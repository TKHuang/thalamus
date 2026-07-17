"""Privacy regressions for compatibility traces and raw payload logging."""

import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code import pipeline  # noqa: E402
from claude_code.compatibility_trace import (  # noqa: E402
    CompatibilityTrace,
    emit_attempt_trace,
    emit_terminal_trace,
)
from core.unified_request import UnifiedRequest  # noqa: E402
from core.protobuf_tool_call_parser import NativeToolCall  # noqa: E402
from utils import llm_payload_logger, thalamus_api_logger  # noqa: E402
from utils.raw_payload_logging import is_raw_payload_logging_enabled  # noqa: E402


SENTINELS = (
    "PROMPT_SECRET_9e3f",
    "CONTENT_SECRET_9e3f",
    "ARGUMENT_SECRET_9e3f",
    "TOOL_RESULT_SECRET_9e3f",
    "TOKEN_SECRET_9e3f",
    "UPSTREAM_ERROR_SECRET_9e3f",
)


class _MonkeyPatch:
    def __init__(self) -> None:
        self._undo: list = []

    def setattr(self, target, name, value) -> None:
        self._undo.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def undo(self) -> None:
        for target, name, original in reversed(self._undo):
            setattr(target, name, original)
        self._undo.clear()


class _DummyStream:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _consumed(text: str, thinking: str, errors: list[object] | None = None) -> dict:
    return {
        "text": text,
        "thinking": thinking,
        "composer_tool_calls": [],
        "native_tool_calls": [],
        "interrupted_tool_state": "",
        "errors": errors or [],
        "had_content": bool(text or thinking),
        "has_fatal_error": False,
        "metrics": {"chunk_count": 1, "first_chunk_latency_ms": 0},
    }


def _native_tool_consumed(
    *,
    call_id: str = "call_1",
    name: str = "write_file",
    arguments: dict | None = None,
    text: str = "",
    thinking: str = "",
    errors: list[object] | None = None,
) -> dict:
    value = _consumed(text, thinking, errors)
    args = arguments or {"content": SENTINELS[2]}
    value["native_tool_calls"] = [
        NativeToolCall(
            enum=49,
            call_id=call_id,
            name=name,
            raw_arguments=json.dumps(args),
            arguments=args,
        )
    ]
    value["had_content"] = True
    return value


def _set_raw_logging(value: str | None) -> str | None:
    original = os.environ.get("THALAMUS_RAW_PAYLOAD_LOGGING")
    if value is None:
        os.environ.pop("THALAMUS_RAW_PAYLOAD_LOGGING", None)
    else:
        os.environ["THALAMUS_RAW_PAYLOAD_LOGGING"] = value
    return original


def _restore_raw_logging(original: str | None) -> None:
    if original is None:
        os.environ.pop("THALAMUS_RAW_PAYLOAD_LOGGING", None)
    else:
        os.environ["THALAMUS_RAW_PAYLOAD_LOGGING"] = original


def test_trace_events_keep_only_approved_stable_fields(monkeypatch):
    captured: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        pipeline,
        "emit_attempt_trace",
        lambda trace: captured.append(("attempt", trace)),
    )
    monkeypatch.setattr(
        pipeline,
        "emit_terminal_trace",
        lambda trace: captured.append(("terminal", trace)),
    )
    monkeypatch.setattr(pipeline, "build_cursor_stream_params", lambda *args: ("/chat", {}, b""))
    monkeypatch.setattr(pipeline, "open_streaming_h2_request", lambda *args: _DummyStream())
    async def fake_agent(*args, **kwargs):
        return _native_tool_consumed(
            text=SENTINELS[1],
            thinking=SENTINELS[3],
            errors=[SENTINELS[5]],
        )

    monkeypatch.setattr(pipeline, "call_cursor_agent", fake_agent)
    monkeypatch.setattr(pipeline, "log_llm_request", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_response", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_api_call", lambda *args, **kwargs: None)

    result = asyncio.run(
        pipeline._call_cursor_direct(
            messages=[
                {"role": "user", "content": SENTINELS[0]},
                {"role": "tool", "content": SENTINELS[3]},
            ],
            model="standard-model",
            tools=[{"name": "write_file", "input_schema": {"type": "object"}}],
            valid_tool_names=["write_file"],
            auth_token=SENTINELS[4],
        )
    )

    assert result["tool_calls"][0]["function"]["name"] == "write_file"
    assert [kind for kind, _trace in captured] == ["attempt", "terminal"]
    serialized = json.dumps([trace for _kind, trace in captured], default=lambda value: value.__dict__)
    for sentinel in SENTINELS:
        assert sentinel not in serialized

    expected_fields = set(CompatibilityTrace.__dataclass_fields__)
    for _kind, trace in captured:
        assert set(trace.__dataclass_fields__) == expected_fields
        assert trace.requested_model == "standard-model"
        assert trace.effective_model == "standard-model"
        assert trace.protocol_adapter == "standard_json_v1"
        assert trace.client_format == "anthropic"
        assert trace.candidate_count == 1
        assert trace.accepted_tool_names == ("write_file",)


def test_plain_text_emits_one_attempt_and_final_trace(monkeypatch):
    captured: list[tuple[str, CompatibilityTrace]] = []
    calls = {"count": 0}

    async def fake_consume_stream(*args, **kwargs):
        calls["count"] += 1
        return _consumed(text="Initial prose response.", thinking=SENTINELS[3])

    monkeypatch.setattr(pipeline, "emit_attempt_trace", lambda trace: captured.append(("attempt", trace)))
    monkeypatch.setattr(pipeline, "emit_terminal_trace", lambda trace: captured.append(("terminal", trace)))
    monkeypatch.setattr(pipeline, "build_cursor_stream_params", lambda *args: ("/chat", {}, b""))
    monkeypatch.setattr(pipeline, "open_streaming_h2_request", lambda *args: _DummyStream())
    monkeypatch.setattr(pipeline, "call_cursor_agent", fake_consume_stream)
    monkeypatch.setattr(pipeline, "log_llm_request", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_response", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_api_call", lambda *args, **kwargs: None)

    result = asyncio.run(
        pipeline._call_cursor_direct(
            messages=[{"role": "user", "content": SENTINELS[0]}],
            model="standard-model",
            tools=[{"name": "write_file", "input_schema": {"type": "object"}}],
            valid_tool_names=["write_file"],
            auth_token=SENTINELS[4],
        )
    )

    assert calls["count"] == 1
    assert result["text"] == "Initial prose response."
    assert [kind for kind, _trace in captured] == ["attempt", "terminal"]
    attempt_ids = [trace.attempt_id for kind, trace in captured if kind == "attempt"]
    assert len(attempt_ids) == len(set(attempt_ids)) == 1
    terminal_trace = captured[-1][1]
    assert terminal_trace.attempt_id == attempt_ids[-1]
    assert terminal_trace.terminal_result == "final_text"
    assert terminal_trace.text_bytes == len("Initial prose response.".encode("utf-8"))
    assert terminal_trace.reasoning_bytes == len(SENTINELS[3].encode("utf-8"))
    assert terminal_trace.tool_candidate_source is None
    assert terminal_trace.candidate_count == 0
    assert terminal_trace.accepted_tool_names == ()
    assert not terminal_trace.repair_attempted
    serialized = json.dumps([trace for _kind, trace in captured], default=lambda value: value.__dict__)
    for sentinel in SENTINELS:
        assert sentinel not in serialized


def test_interrupted_json_repair_failure_emits_error_attempt_and_terminal(monkeypatch):
    captured: list[tuple[str, CompatibilityTrace]] = []
    calls = {"count": 0}

    async def fake_consume_stream(*args, **kwargs):
        calls["count"] += 1
        raise RuntimeError(SENTINELS[5])

    monkeypatch.setattr(pipeline, "emit_attempt_trace", lambda trace: captured.append(("attempt", trace)))
    monkeypatch.setattr(pipeline, "emit_terminal_trace", lambda trace: captured.append(("terminal", trace)))
    monkeypatch.setattr(pipeline, "build_cursor_stream_params", lambda *args: ("/chat", {}, b""))
    monkeypatch.setattr(pipeline, "open_streaming_h2_request", lambda *args: _DummyStream())
    monkeypatch.setattr(pipeline, "call_cursor_agent", fake_consume_stream)
    monkeypatch.setattr(pipeline, "log_llm_request", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_response", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_api_call", lambda *args, **kwargs: None)

    result = asyncio.run(
        pipeline._call_cursor_direct(
            messages=[{"role": "user", "content": SENTINELS[0]}],
            model="standard-model",
            tools=[{"name": "write_file", "input_schema": {"type": "object"}}],
            valid_tool_names=["write_file"],
            auth_token=SENTINELS[4],
        )
    )

    assert SENTINELS[5] in result["error"]
    assert [kind for kind, _trace in captured] == ["attempt", "terminal"]
    attempt_ids = [trace.attempt_id for kind, trace in captured if kind == "attempt"]
    assert len(attempt_ids) == len(set(attempt_ids)) == 1
    terminal_trace = captured[-1][1]
    assert terminal_trace.attempt_id == attempt_ids[-1]
    assert terminal_trace.terminal_result == "upstream_error"
    assert terminal_trace.text_bytes == 0
    assert terminal_trace.reasoning_bytes == 0
    assert terminal_trace.tool_candidate_source is None
    assert terminal_trace.candidate_count == 0
    assert terminal_trace.accepted_tool_names == ()
    serialized = json.dumps([trace for _kind, trace in captured], default=lambda value: value.__dict__)
    for sentinel in SENTINELS:
        assert sentinel not in serialized


def test_fallback_trace_records_safe_reason_and_final_model(monkeypatch):
    captured: list[tuple[str, CompatibilityTrace]] = []
    calls = {"count": 0}
    final_text = SENTINELS[1]

    class FallbackConfig:
        max_attempts = 2

        @staticmethod
        def should_fallback(error_text):
            return error_text == SENTINELS[5]

        @staticmethod
        def select_next_model(requested, tried):
            return "fallback-cursor-model" if tried == ["primary-cursor-model"] else None

    async def fake_consume_stream(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError(SENTINELS[5])
        return _native_tool_consumed(
            call_id="call_fallback",
            arguments={"content": SENTINELS[2]},
            text=SENTINELS[1],
        )

    monkeypatch.setattr(pipeline, "emit_attempt_trace", lambda trace: captured.append(("attempt", trace)))
    monkeypatch.setattr(pipeline, "emit_terminal_trace", lambda trace: captured.append(("terminal", trace)))
    monkeypatch.setattr(pipeline, "build_cursor_stream_params", lambda *args: ("/chat", {}, b""))
    monkeypatch.setattr(pipeline, "open_streaming_h2_request", lambda *args: _DummyStream())
    monkeypatch.setattr(pipeline, "call_cursor_agent", fake_consume_stream)
    monkeypatch.setattr(pipeline, "load_fallback_config", lambda: FallbackConfig())
    monkeypatch.setattr(pipeline, "log_llm_request", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_response", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_api_call", lambda *args, **kwargs: None)

    result = asyncio.run(
        pipeline._call_cursor_direct(
            messages=[{"role": "user", "content": SENTINELS[0]}],
            model="primary-cursor-model",
            tools=[{"name": "write_file", "input_schema": {"type": "object"}}],
            valid_tool_names=["write_file"],
            auth_token=SENTINELS[4],
            requested_model="client-alias-model",
            client_format="openai",
        )
    )

    assert result["tool_calls"][0]["function"]["name"] == "write_file"
    assert [kind for kind, _trace in captured] == ["attempt", "attempt", "terminal"]
    first_attempt, fallback_attempt = [trace for kind, trace in captured if kind == "attempt"]
    terminal_trace = captured[-1][1]
    assert first_attempt.attempt_id != fallback_attempt.attempt_id
    assert first_attempt.effective_model == "primary-cursor-model"
    assert fallback_attempt.requested_model == "client-alias-model"
    assert fallback_attempt.effective_model == "fallback-cursor-model"
    assert fallback_attempt.fallback_reason == "transport_error"
    assert terminal_trace.attempt_id == fallback_attempt.attempt_id
    assert terminal_trace.fallback_reason == "transport_error"
    assert terminal_trace.text_bytes == len(final_text.encode("utf-8"))
    assert terminal_trace.accepted_tool_names == ("write_file",)
    serialized = json.dumps([trace for _kind, trace in captured], default=lambda value: value.__dict__)
    for sentinel in SENTINELS:
        assert sentinel not in serialized


def test_run_pipeline_threads_original_model_and_format_to_direct_call(monkeypatch):
    captured_kwargs: list[dict] = []

    async def fake_call_cursor_direct(*args, **kwargs):
        captured_kwargs.append(kwargs)
        return {
            "text": "Final response.",
            "thinking": "",
            "model": args[1],
            "fallback_attempts": 0,
            "stats": {},
        }

    monkeypatch.setattr(pipeline, "_call_cursor_direct", fake_call_cursor_direct)
    monkeypatch.setattr(pipeline, "get_cursor_access_token", lambda: "token")

    for original_format in ("anthropic", "openai"):
        request = UnifiedRequest(
            messages=[{"role": "user", "content": "hello"}],
            system="",
            tools=[],
            model="cursor-mapped-model",
            stream=False,
            original_model="client-alias-model",
            original_format=original_format,
        )
        result = asyncio.run(pipeline.run_pipeline(request, f"request-{original_format}"))
        assert result["ok"]

    assert captured_kwargs == [
        {
            "compact_tools": False,
            "requested_model": "client-alias-model",
            "client_format": "anthropic",
        },
        {
            "compact_tools": True,
            "requested_model": "client-alias-model",
            "client_format": "openai",
        },
    ]


def test_streaming_pipeline_threads_original_model_and_format_to_direct_call(monkeypatch):
    captured_kwargs: list[dict] = []

    async def fake_call_cursor_direct(*args, **kwargs):
        captured_kwargs.append(kwargs)
        return {
            "text": "Final response.",
            "thinking": "",
            "model": args[1],
            "fallback_attempts": 0,
            "stats": {},
        }

    monkeypatch.setattr(pipeline, "_call_cursor_direct", fake_call_cursor_direct)
    monkeypatch.setattr(pipeline, "get_cursor_access_token", lambda: "token")

    async def consume_stream(result: dict) -> None:
        async for _chunk in result["stream_handler"]():
            pass

    for original_format in ("anthropic", "openai"):
        request = UnifiedRequest(
            messages=[{"role": "user", "content": "hello"}],
            system="",
            tools=[],
            model="cursor-mapped-model",
            stream=True,
            original_model="client-alias-model",
            original_format=original_format,
        )
        result = asyncio.run(pipeline.run_pipeline(request, f"stream-{original_format}"))
        asyncio.run(consume_stream(result))

    assert captured_kwargs == [
        {
            "on_stream_delta": captured_kwargs[0]["on_stream_delta"],
            "on_thinking_delta": captured_kwargs[0]["on_thinking_delta"],
            "compact_tools": False,
            "requested_model": "client-alias-model",
            "client_format": "anthropic",
        },
        {
            "on_stream_delta": captured_kwargs[1]["on_stream_delta"],
            "on_thinking_delta": captured_kwargs[1]["on_thinking_delta"],
            "compact_tools": True,
            "requested_model": "client-alias-model",
            "client_format": "openai",
        },
    ]


def test_trace_preserves_client_alias_and_explicit_format(monkeypatch):
    captured: list[CompatibilityTrace] = []

    monkeypatch.setattr(pipeline, "emit_attempt_trace", captured.append)
    monkeypatch.setattr(pipeline, "emit_terminal_trace", lambda trace: None)
    monkeypatch.setattr(pipeline, "build_cursor_stream_params", lambda *args: ("/chat", {}, b""))
    monkeypatch.setattr(pipeline, "open_streaming_h2_request", lambda *args: _DummyStream())
    monkeypatch.setattr(
        pipeline,
        "consume_stream",
        lambda *args, **kwargs: _async_result(_consumed(text="Final response.", thinking="")),
    )
    monkeypatch.setattr(pipeline, "log_llm_request", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_response", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_api_call", lambda *args, **kwargs: None)

    for client_format, compact_tools in (("anthropic", True), ("openai", False)):
        asyncio.run(
            pipeline._call_cursor_direct(
                messages=[{"role": "user", "content": "hello"}],
                model="cursor-mapped-model",
                tools=[],
                valid_tool_names=[],
                auth_token="token",
                compact_tools=compact_tools,
                requested_model="client-alias-model",
                client_format=client_format,
            )
        )

    assert [(trace.requested_model, trace.effective_model, trace.client_format) for trace in captured] == [
        ("client-alias-model", "cursor-mapped-model", "anthropic"),
        ("client-alias-model", "cursor-mapped-model", "openai"),
    ]


def test_emitters_serialize_trace_as_stable_event_dictionary(monkeypatch):
    captured: list[tuple[str, dict]] = []
    trace = CompatibilityTrace(
        request_id="request-1",
        attempt_id="request-1:1",
        requested_model="requested-model",
        effective_model="effective-model",
        protocol_adapter="standard_json_v1",
        client_format="anthropic",
        fallback_reason=None,
        text_bytes=12,
        reasoning_bytes=7,
        tool_candidate_source="text",
        candidate_count=2,
        accepted_tool_names=("read_file", "write_file"),
        rejection_reason="unknown_tool",
        repair_attempted=False,
        terminal_result="tool_calls",
        latency_ms=42,
    )

    import claude_code.compatibility_trace as compatibility_trace

    monkeypatch.setattr(
        compatibility_trace.logger,
        "info",
        lambda event, payload: captured.append((event, payload)),
    )

    emit_attempt_trace(trace)
    emit_terminal_trace(trace)

    assert [event for event, _payload in captured] == [
        "compatibility_attempt",
        "compatibility_terminal",
    ]
    for _event, payload in captured:
        assert payload["event"] == _event
        assert set(payload["trace"]) == set(CompatibilityTrace.__dataclass_fields__)
        assert payload["trace"]["accepted_tool_names"] == ["read_file", "write_file"]


def test_raw_payload_logging_is_disabled_except_for_literal_true(monkeypatch):
    original = _set_raw_logging(None)
    try:
        assert not is_raw_payload_logging_enabled()
        for value in ("", "false", "true-ish", "1", "yes"):
            _set_raw_logging(value)
            assert not is_raw_payload_logging_enabled()
        _set_raw_logging(" TRUE ")
        assert is_raw_payload_logging_enabled()
    finally:
        _restore_raw_logging(original)


def test_raw_loggers_do_not_create_or_write_files_by_default(monkeypatch):
    original = _set_raw_logging(None)
    try:
        def payload_directory_called() -> str:
            raise AssertionError("raw payload directory should not be created")

        monkeypatch.setattr(llm_payload_logger, "_payload_dir", payload_directory_called)
        monkeypatch.setattr(thalamus_api_logger, "_payload_dir", payload_directory_called)
        monkeypatch.setattr(llm_payload_logger, "_api_log_path", payload_directory_called)
        monkeypatch.setattr(thalamus_api_logger, "_api_log_path", payload_directory_called)

        assert llm_payload_logger.log_llm_request("req", "model", []) == ""
        assert llm_payload_logger.log_llm_response("req", "model", "response") == ""
        assert thalamus_api_logger.log_thalamus_request("req", "/v1/messages", "POST", {}) == ""
        assert thalamus_api_logger.log_thalamus_response("req", "/v1/messages", 200, {}) == ""
        assert llm_payload_logger.log_llm_api_call("req", "model", "OK", 1, "", "") is None
        assert thalamus_api_logger.log_thalamus_api_call(
            "req", "/v1/messages", "POST", 200, 1, "", ""
        ) is None
    finally:
        _restore_raw_logging(original)


def test_raw_loggers_persist_payloads_only_when_enabled(monkeypatch):
    original = _set_raw_logging("true")
    try:
        with tempfile.TemporaryDirectory() as payload_dir:
            writes: list[tuple[str, object]] = []
            monkeypatch.setattr(llm_payload_logger, "_payload_dir", lambda: payload_dir)
            monkeypatch.setattr(thalamus_api_logger, "_payload_dir", lambda: payload_dir)
            monkeypatch.setattr(
                llm_payload_logger,
                "_write_json",
                lambda path, payload: writes.append((path, payload)),
            )
            monkeypatch.setattr(
                thalamus_api_logger,
                "_write_json",
                lambda path, payload: writes.append((path, payload)),
            )

            assert llm_payload_logger.log_llm_request("req", "model", [])
            assert llm_payload_logger.log_llm_response("req", "model", "response")
            assert thalamus_api_logger.log_thalamus_request("req", "/v1/messages", "POST", {})
            assert thalamus_api_logger.log_thalamus_response("req", "/v1/messages", 200, {})

        assert len(writes) == 4
    finally:
        _restore_raw_logging(original)


async def _async_result(value: dict) -> dict:
    return value


def _run_all() -> int:
    functions = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    failures = 0
    for function in functions:
        monkeypatch = _MonkeyPatch()
        try:
            if function.__code__.co_argcount:
                function(monkeypatch)
            else:
                function()
            print(f"PASS {function.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {function.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {function.__name__}: {type(exc).__name__}: {exc}")
        finally:
            monkeypatch.undo()
    print(f"\n{len(functions) - failures}/{len(functions)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
