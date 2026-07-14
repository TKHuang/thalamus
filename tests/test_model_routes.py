"""Unit tests for the /v1/models listing route's resilience.

Runs standalone (``uv run python tests/test_model_routes.py``) and under pytest.

WHY these tests exist: clients such as Hermes fetch /v1/models to populate their
model picker and fall back to a single configured model when the fetch fails.
Cursor's AvailableModels RPC intermittently returns 401 (ERROR_NOT_LOGGED_IN),
so a naive proxy turns a transient blip into an empty picker. These tests pin
the cache + single-flight + serve-stale behavior that prevents that.
"""

import asyncio
import gzip
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_context import replace_model_context_catalog  # noqa: E402
from core.protobuf_builder import (  # noqa: E402
    build_gzip_framed_protobuf_chat_request_body,
)
from proto import cursor_api_pb2 as pb  # noqa: E402
from routes import model_routes  # noqa: E402


def _models_buffer(*names: str) -> bytes:
    """Serialize an AvailableModelsResponse with the given bare model names."""
    resp = pb.AvailableModelsResponse()
    for name in names:
        resp.models.add().name = name
    return resp.SerializeToString()


def _varint(value: int) -> bytes:
    encoded = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        encoded.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(encoded)


def _merge_unknown_varint(message, field_number: int, value: int) -> None:
    message.MergeFromString(_varint(field_number << 3) + _varint(value))


def _merge_unknown_bytes(message, field_number: int, payload: bytes) -> None:
    tag = _varint((field_number << 3) | 2)
    message.MergeFromString(tag + _varint(len(payload)) + payload)


def _add_tooltip_context(model, context: str, *, max_mode: bool = False) -> None:
    markdown = f"**Model**<br /><br />{context} context window".encode()
    tooltip = _varint((7 << 3) | 2) + _varint(len(markdown)) + markdown
    _merge_unknown_bytes(model, 20 if max_mode else 8, tooltip)


def _decode_chat_request(body: bytes):
    magic = body[0]
    payload_length = struct.unpack(">I", body[1:5])[0]
    payload = body[5 : 5 + payload_length]
    if magic == 1:
        payload = gzip.decompress(payload)
    request = pb.StreamUnifiedChatWithToolsRequest()
    request.ParseFromString(payload)
    return request


def _reset_cache() -> None:
    model_routes._model_cache["ids"] = []
    model_routes._model_cache["metadata"] = {}
    model_routes._model_cache["fetched_at"] = 0.0
    replace_model_context_catalog({})


def _patch_send(monkeypatch, responses: list[dict]) -> dict:
    """Make send_unary_h2_request return `responses` in order (last repeats).

    Returns a counter dict whose ``n`` field records how many upstream calls
    were made. Also no-ops the retry backoff so tests stay fast.
    """
    counter = {"n": 0}

    async def fake_send(path, headers, body):
        idx = min(counter["n"], len(responses) - 1)
        counter["n"] += 1
        return responses[idx]

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(model_routes, "send_unary_h2_request", fake_send)
    monkeypatch.setattr(model_routes.asyncio, "sleep", no_sleep)
    return counter


class _MonkeyPatch:
    """Minimal monkeypatch shim so tests run identically with or without pytest."""

    def __init__(self) -> None:
        self._undo: list = []

    def setattr(self, target, name, value) -> None:
        self._undo.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def undo(self) -> None:
        for target, name, original in reversed(self._undo):
            setattr(target, name, original)
        self._undo.clear()


def test_serves_cached_models_when_upstream_fails(monkeypatch) -> None:
    """A transient upstream 401 must serve the last good list, never empty it."""
    _reset_cache()
    good = {"status": 200, "buffer": _models_buffer("glm-5.2-high", "composer-2.5")}
    bad = {"status": 401, "buffer": b"ERROR_NOT_LOGGED_IN"}
    counter = _patch_send(monkeypatch, [good, bad])

    first = asyncio.run(model_routes._list_model_ids("tok"))
    assert "glm-5.2-high" in first and "composer-2.5" in first

    # Expire the cache so the next call refetches — and upstream now 401s.
    model_routes._model_cache["fetched_at"] = 0.0
    second = asyncio.run(model_routes._list_model_ids("tok"))

    # Serve-stale: the picker keeps the previous list instead of going empty.
    assert second == first
    # Retried _AVAILABLE_MODELS_MAX_ATTEMPTS times on the failing refresh.
    assert counter["n"] == 1 + model_routes._AVAILABLE_MODELS_MAX_ATTEMPTS


def test_cache_hit_skips_upstream(monkeypatch) -> None:
    """A second call within the TTL is served from cache, with no upstream hit."""
    _reset_cache()
    good = {"status": 200, "buffer": _models_buffer("glm-5.2-high")}
    counter = _patch_send(monkeypatch, [good])

    asyncio.run(model_routes._list_model_ids("tok"))
    asyncio.run(model_routes._list_model_ids("tok"))

    assert counter["n"] == 1


def test_no_cache_propagates_failure(monkeypatch) -> None:
    """With no prior success, an upstream failure surfaces (route maps it to 502)."""
    _reset_cache()
    bad = {"status": 401, "buffer": b"ERROR_NOT_LOGGED_IN"}
    _patch_send(monkeypatch, [bad])

    raised = False
    try:
        asyncio.run(model_routes._list_model_ids("tok"))
    except RuntimeError:
        raised = True
    assert raised


def test_live_cursor_tooltip_context_is_cached_for_base_and_variant(monkeypatch) -> None:
    """Use Cursor's live tooltip metadata instead of a local model-name map."""
    _reset_cache()
    upstream = pb.AvailableModelsResponse()
    grok = upstream.models.add()
    grok.name = "grok-4.5"
    _add_tooltip_context(grok, "256k")
    _add_tooltip_context(grok, "256k", max_mode=True)
    _merge_unknown_bytes(grok, 36, b"cursor-grok-4.5-high")
    future = upstream.models.add()
    future.name = "future-model-without-context"

    _patch_send(monkeypatch, [{"status": 200, "buffer": upstream.SerializeToString()}])
    model_ids = asyncio.run(model_routes._list_model_ids("tok"))

    assert model_ids == [
        "grok-4.5",
        "cursor-grok-4.5-high",
        "future-model-without-context",
    ]
    assert model_routes._model_cache["metadata"] == {
        "grok-4.5": {"context_length": 256000, "max_context_length": 256000},
        "cursor-grok-4.5-high": {
            "context_length": 256000,
            "max_context_length": 256000,
        },
    }

    response = model_routes._models_response(model_ids)
    body = json.loads(response.body)
    assert body == {
        "object": "list",
        "data": [
            {
                "id": "grok-4.5",
                "object": "model",
                "created": 0,
                "owned_by": "cursor",
                "context_length": 256000,
            },
            {
                "id": "cursor-grok-4.5-high",
                "object": "model",
                "created": 0,
                "owned_by": "cursor",
                "context_length": 256000,
            },
            {
                "id": "future-model-without-context",
                "object": "model",
                "created": 0,
                "owned_by": "cursor",
            },
        ],
    }


def test_dual_context_models_expose_normal_and_max_mode_ids(monkeypatch) -> None:
    """Every effort variant gets an adjacent synthetic Max Context selection."""
    _reset_cache()
    upstream = pb.AvailableModelsResponse()
    opus = upstream.models.add()
    opus.name = "claude-opus-4-8"
    _add_tooltip_context(opus, "300k")
    _add_tooltip_context(opus, "1m", max_mode=True)
    _merge_unknown_bytes(opus, 36, b"claude-opus-4-8-max")

    _patch_send(monkeypatch, [{"status": 200, "buffer": upstream.SerializeToString()}])
    model_ids = asyncio.run(model_routes._list_model_ids("tok"))

    assert model_ids == [
        "claude-opus-4-8",
        "claude-opus-4-8[1m]",
        "claude-opus-4-8-max",
        "claude-opus-4-8-max[1m]",
    ]
    body = json.loads(model_routes._models_response(model_ids).body)
    assert [item["context_length"] for item in body["data"]] == [
        300000,
        1000000,
        300000,
        1000000,
    ]


def test_context_suffix_routes_same_model_through_max_mode(monkeypatch) -> None:
    """The synthetic suffix selects Max Mode but never reaches Cursor's model name."""
    replace_model_context_catalog(
        {
            "claude-opus-4-8-max": {
                "context_length": 300000,
                "max_context_length": 1000000,
            }
        }
    )
    long_message = [{"role": "user", "content": "x" * 21001}]
    short_message = [{"role": "user", "content": "hello"}]

    normal = _decode_chat_request(
        build_gzip_framed_protobuf_chat_request_body(
            long_message,
            "claude-opus-4-8-max",
            agent_mode=True,
        )
    )
    max_mode = _decode_chat_request(
        build_gzip_framed_protobuf_chat_request_body(
            short_message,
            "claude-opus-4-8-max[1m]",
            agent_mode=True,
        )
    )
    unknown_legacy = _decode_chat_request(
        build_gzip_framed_protobuf_chat_request_body(
            long_message,
            "model-without-dual-context",
            agent_mode=True,
        )
    )

    assert normal.request.model.name == "claude-opus-4-8-max"
    assert normal.request.largeContext == 0
    assert max_mode.request.model.name == "claude-opus-4-8-max"
    assert max_mode.request.largeContext == 1
    assert unknown_legacy.request.largeContext == 1


def test_numeric_context_field_wins_when_cursor_populates_it(monkeypatch) -> None:
    """The stable numeric field takes precedence over human-readable tooltip text."""
    response = pb.AvailableModelsResponse()
    model = response.models.add()
    model.name = "future-model"
    _merge_unknown_varint(model, 15, 300000)
    _merge_unknown_varint(model, 16, 1200000)
    _add_tooltip_context(model, "256k")
    _add_tooltip_context(model, "1m", max_mode=True)

    assert model_routes._extract_model_context_metadata(model) == {
        "context_length": 300000,
        "max_context_length": 1200000,
    }


def test_model_detail_uses_live_cache_and_validated_max_marker(monkeypatch) -> None:
    """Detail probes reuse live metadata and cannot invent unobserved models."""
    model_routes._model_cache["ids"] = ["gpt-next"]
    model_routes._model_cache["metadata"] = {
        "gpt-next": {"context_length": 300000, "max_context_length": 1000000}
    }

    normal = asyncio.run(model_routes.get_model_detail("gpt-next"))
    max_mode = asyncio.run(model_routes.get_model_detail("gpt-next[1m]"))
    unknown = asyncio.run(model_routes.get_model_detail("unknown[1m]"))

    assert json.loads(normal.body)["context_length"] == 300000
    assert json.loads(max_mode.body)["context_length"] == 1000000
    assert unknown.status_code == 404


def test_placeholder_api_key_falls_back_to_env_token(monkeypatch) -> None:
    """A non-Cursor placeholder bearer (the api_key Hermes sends to enable
    discovery) must be ignored in favor of the env token — never forwarded to
    Cursor as a bogus bearer (which 401s) — while a real Cursor token is used."""
    monkeypatch.setattr(model_routes, "get_cursor_access_token", lambda: "user_x::eyJENV")
    # Placeholder api_key (no '::' / 'eyJ') → ignored, env token used.
    assert model_routes._resolve_cursor_token("Bearer thalamus-proxy") == "eyJENV"
    # No header at all → env token.
    assert model_routes._resolve_cursor_token(None) == "eyJENV"
    # A real Cursor-shaped client token IS honored (bare JWT).
    assert model_routes._resolve_cursor_token("Bearer eyJCLIENT") == "eyJCLIENT"
    # Prefixed Cursor token is stripped to the bare JWT.
    assert model_routes._resolve_cursor_token("Bearer user_y::eyJPREF") == "eyJPREF"


def _run_all() -> int:
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in funcs:
        mp = _MonkeyPatch()
        try:
            fn(mp)
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
        finally:
            mp.undo()
            _reset_cache()
    print(f"\n{len(funcs) - failures}/{len(funcs)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
