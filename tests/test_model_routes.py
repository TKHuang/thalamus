"""Unit tests for the /v1/models listing route's resilience.

Runs standalone (``uv run python tests/test_model_routes.py``) and under pytest.

WHY these tests exist: clients such as Hermes fetch /v1/models to populate their
model picker and fall back to a single configured model when the fetch fails.
Cursor's AvailableModels RPC intermittently returns 401 (ERROR_NOT_LOGGED_IN),
so a naive proxy turns a transient blip into an empty picker. These tests pin
the cache + single-flight + serve-stale behavior that prevents that.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proto import cursor_api_pb2 as pb  # noqa: E402
from routes import model_routes  # noqa: E402


def _models_buffer(*names: str) -> bytes:
    """Serialize an AvailableModelsResponse with the given bare model names."""
    resp = pb.AvailableModelsResponse()
    for name in names:
        resp.models.add().name = name
    return resp.SerializeToString()


def _reset_cache() -> None:
    model_routes._model_cache["ids"] = []
    model_routes._model_cache["fetched_at"] = 0.0


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
