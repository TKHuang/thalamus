"""Regression tests for model-specific tool protocol attempts."""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code import pipeline  # noqa: E402
from claude_code.composer_tool_protocol import ComposerMarkerV1Adapter  # noqa: E402
from claude_code.tool_prompt_builder import inject_tool_prompt_into_messages  # noqa: E402
from claude_code.tool_protocols import ProtocolDecodeState, ProtocolFragment  # noqa: E402
from claude_code.tool_validation import validate_tool_candidates  # noqa: E402


TOOLS = [{"name": "write_file", "input_schema": {"type": "object"}}]


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


def _composer_marker_block(composer_calls: list[dict]) -> str:
    blocks: list[str] = []
    for call in composer_calls:
        arguments = call.get("arguments") or {}
        body = "\n".join(
            f"{key}\n{value}" for key, value in arguments.items()
        )
        blocks.append(
            "<|tool_call_begin|>"
            f"{call['name']}<|tool_sep|>{body}<|tool_call_end|>"
        )
    return "</think><|tool_calls_begin|>" + "".join(blocks) + "<|tool_calls_end|>"


def _consumed(
    *,
    text: str = "",
    thinking: str = "",
    composer_calls: list[dict] | None = None,
    interrupted_tool_state: str = "",
) -> dict:
    marker_thinking = _composer_marker_block(composer_calls or [])
    return {
        "text": text,
        "thinking": thinking or marker_thinking,
        "composer_tool_calls": composer_calls or [],
        "interrupted_tool_state": interrupted_tool_state,
        "has_fatal_error": False,
        "errors": [],
        "had_content": bool(text or thinking or marker_thinking),
        "metrics": {"chunk_count": 1, "first_chunk_latency_ms": 0},
    }


def _request_text(messages: list[dict]) -> str:
    return "\n".join(str(message.get("content", "")) for message in messages)


def test_composer_adapter_separates_unicode_reasoning_answer_and_tools_across_boundaries():
    marker_block = (
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>write_file<｜tool▁sep｜>"
        "path\n/tmp/émoji-工具.txt<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    )
    payload = f"先分析</think>完成。{marker_block}"

    for boundary in range(1, len(payload)):
        adapter = ComposerMarkerV1Adapter()
        state = ProtocolDecodeState()
        visible_parts: list[str] = []
        candidates = []
        for sequence, chunk in enumerate((payload[:boundary], payload[boundary:])):
            result = adapter.incremental_decode(
                ProtocolFragment(sequence=sequence, lane="reasoning", text=chunk), state
            )
            state = result.state
            visible_parts.append(result.visible_text)
            candidates.extend(result.candidates)
        flushed = adapter.incremental_decode(None, state)
        visible_parts.append(flushed.visible_text)
        candidates.extend(flushed.candidates)

        visible = "".join(visible_parts)
        assert state.reasoning_buffer == "先分析", boundary
        assert visible == "完成。", boundary
        assert "<｜" not in visible, boundary
        assert "</think>" not in visible, boundary
        assert [(candidate.raw_name, candidate.arguments) for candidate in candidates] == [
            ("write_file", {"path": "/tmp/émoji-工具.txt"})
        ], boundary

    continuation = ComposerMarkerV1Adapter().render_continuation(TOOLS, "create a file", "I will do it")
    repair = ComposerMarkerV1Adapter().render_repair(TOOLS, "partial marker call")
    assert "<|tool_calls_begin|>" in continuation
    assert "<|tool_calls_begin|>" in repair
    assert '"type":"tool_use"' not in continuation
    assert '"type":"tool_use"' not in repair


def test_composer_adapter_sends_invalid_json_arguments_to_strict_validation():
    adapter = ComposerMarkerV1Adapter()
    state = ProtocolDecodeState()
    body = '{"name":"write_file","arguments":null}'
    marker = (
        "</think><|tool_calls_begin|><|tool_call_begin|>"
        f"{body}<|tool_call_end|><|tool_calls_end|>"
    )

    decoded = adapter.incremental_decode(
        ProtocolFragment(sequence=0, lane="reasoning", text=marker), state
    )
    validation = validate_tool_candidates(
        decoded.candidates, allowed_names={"write_file"}
    )

    assert len(decoded.candidates) == 1
    assert decoded.candidates[0].arguments is None
    assert validation.accepted == ()
    assert validation.rejected[0].reason == "arguments_not_object"


def test_clean_eof_marker_state_invokes_one_marker_repair():
    calls = {"count": 0}
    rendered_repairs: list[str] = []

    async def open_and_consume(messages):
        calls["count"] += 1
        if calls["count"] == 1:
            return _consumed(interrupted_tool_state="<｜tool▁calls▁beg")
        return _consumed(text="Recovered.")

    def render_repair(partial):
        rendered_repairs.append(partial["interrupted_tool_state"])
        return "marker repair"

    attempt = asyncio.run(
        pipeline._consume_attempt_with_repair(
            messages=[{"role": "user", "content": "create x.txt"}],
            open_and_consume=open_and_consume,
            render_repair=render_repair,
            allow_repair=True,
            is_composer=True,
        )
    )

    assert attempt.repair_attempted
    assert attempt.consumed["text"] == "Recovered."
    assert calls["count"] == 2
    assert rendered_repairs == ["<｜tool▁calls▁beg"]


def test_literal_less_than_state_never_invokes_marker_repair():
    calls = {"count": 0}
    rendered_repairs: list[str] = []

    async def open_and_consume(messages):
        calls["count"] += 1
        return _consumed(
            text="The literal comparison operator is <",
            interrupted_tool_state="<",
        )

    def render_repair(partial):
        rendered_repairs.append(partial["interrupted_tool_state"])
        return "marker repair"

    attempt = asyncio.run(
        pipeline._consume_attempt_with_repair(
            messages=[{"role": "user", "content": "create x.txt"}],
            open_and_consume=open_and_consume,
            render_repair=render_repair,
            allow_repair=True,
            is_composer=True,
        )
    )

    assert not attempt.repair_attempted
    assert attempt.consumed["text"] == "The literal comparison operator is <"
    assert calls["count"] == 1
    assert rendered_repairs == []


def test_injection_uses_the_selected_adapter_manifest():
    messages = inject_tool_prompt_into_messages(
        [{"role": "user", "content": "create x.txt"}],
        TOOLS,
        adapter=ComposerMarkerV1Adapter(),
        compact_tools=True,
    )

    rendered = _request_text(messages)
    assert "<|tool_calls_begin|>" in rendered
    assert '"type":"tool_use"' not in rendered


def test_fallback_rebuilds_the_initial_manifest_for_the_effective_model(monkeypatch):
    class FallbackConfig:
        max_attempts = 2

        @staticmethod
        def should_fallback(error_text):
            return error_text == "retry elsewhere"

        @staticmethod
        def select_next_model(requested, tried):
            return "composer-2.5" if tried == ["standard-model"] else None

    request_messages: list[list[dict]] = []
    calls = {"count": 0}

    def fake_build_cursor_stream_params(auth_token, messages, model):
        request_messages.append(list(messages))
        return "/chat", {}, b""

    async def fake_consume_stream(stream, composer=False, on_text_delta=None, on_thinking_delta=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("retry elsewhere")
        assert composer is False
        return _consumed(
            composer_calls=[{"name": "write_file", "arguments": {"path": "x.txt"}}]
        )

    monkeypatch.setattr(pipeline, "build_cursor_stream_params", fake_build_cursor_stream_params)
    monkeypatch.setattr(pipeline, "open_streaming_h2_request", lambda *args: _DummyStream())
    monkeypatch.setattr(pipeline, "consume_stream", fake_consume_stream)
    monkeypatch.setattr(pipeline, "load_fallback_config", lambda: FallbackConfig())
    monkeypatch.setattr(pipeline, "log_llm_request", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_response", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_api_call", lambda *args, **kwargs: None)

    result = asyncio.run(
        pipeline._call_cursor_direct(
            messages=[{"role": "user", "content": "create x.txt"}],
            model="standard-model",
            tools=TOOLS,
            valid_tool_names=["write_file"],
            auth_token="token",
        )
    )

    assert result["tool_calls"][0]["function"]["name"] == "write_file"
    assert "Available client tools are the following JSON schemas:" in _request_text(request_messages[0])
    assert "<|tool_calls_begin|>" in _request_text(request_messages[1])
    assert '"type":"tool_use"' not in _request_text(request_messages[1])


def test_composer_continuation_and_repair_use_marker_grammar(monkeypatch):
    request_messages: list[list[dict]] = []
    calls = {"count": 0}

    def fake_build_cursor_stream_params(auth_token, messages, model):
        request_messages.append(list(messages))
        return "/chat", {}, b""

    async def fake_consume_stream(stream, composer=False, on_text_delta=None, on_thinking_delta=None):
        assert composer is False
        calls["count"] += 1
        if calls["count"] == 1:
            raise pipeline.PartialStreamConsumptionError(
                "reset",
                _consumed(
                    thinking="reasoning</think><|tool_calls_beg",
                ),
            )
        if calls["count"] == 2:
            return _consumed(text="I will write it.")
        return _consumed(
            composer_calls=[{"name": "write_file", "arguments": {"path": "x.txt"}}]
        )

    created_adapters = []
    original_create_protocol_adapter = pipeline.create_protocol_adapter

    def recording_create_protocol_adapter(protocol):
        adapter = original_create_protocol_adapter(protocol)
        created_adapters.append(adapter)
        return adapter

    monkeypatch.setattr(pipeline, "build_cursor_stream_params", fake_build_cursor_stream_params)
    monkeypatch.setattr(pipeline, "open_streaming_h2_request", lambda *args: _DummyStream())
    monkeypatch.setattr(pipeline, "consume_stream", fake_consume_stream)
    monkeypatch.setattr(pipeline, "create_protocol_adapter", recording_create_protocol_adapter)
    monkeypatch.setattr(pipeline, "log_llm_request", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_response", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_api_call", lambda *args, **kwargs: None)

    result = asyncio.run(
        pipeline._call_cursor_direct(
            messages=[{"role": "user", "content": "create x.txt"}],
            model="composer-2.5",
            tools=TOOLS,
            valid_tool_names=["write_file"],
            auth_token="token",
        )
    )

    assert result["tool_calls"][0]["function"]["name"] == "write_file"
    assert len(request_messages) == 3
    assert len(created_adapters) >= 3
    assert len({id(adapter) for adapter in created_adapters}) == len(created_adapters)
    repair_text = _request_text(request_messages[1])
    continuation_text = _request_text(request_messages[2])
    assert repair_text.count("Repair the interrupted tool call.") == 1
    assert "<|tool_calls_begin|>" in repair_text
    assert "<|tool_calls_begin|>" in continuation_text
    assert '"type":"tool_use"' not in repair_text
    assert '"type":"tool_use"' not in continuation_text


def test_fallback_rebuilds_standard_and_composer_manifests_for_the_next_model(monkeypatch):
    scenarios = [
        ("standard-model", "standard-fallback", False, False),
        ("composer-2.5", "standard-fallback", True, False),
    ]

    for initial_model, fallback_model, initial_composer, fallback_composer in scenarios:
        class FallbackConfig:
            max_attempts = 2

            @staticmethod
            def should_fallback(error_text):
                return error_text == "retry elsewhere"

            @staticmethod
            def select_next_model(requested, tried):
                return fallback_model if tried == [initial_model] else None

        request_messages: list[list[dict]] = []
        calls = {"count": 0}

        def fake_build_cursor_stream_params(auth_token, messages, model):
            request_messages.append(list(messages))
            return "/chat", {}, b""

        async def fake_consume_stream(stream, composer=False, on_text_delta=None, on_thinking_delta=None):
            calls["count"] += 1
            if calls["count"] == 1:
                assert composer is False
                raise RuntimeError("retry elsewhere")
            assert composer is False
            return _consumed(
                text=(
                    '{"type":"tool_use","id":"toolu_1","name":"write_file",'
                    '"input":{"path":"x.txt"}}'
                )
            )

        monkeypatch.setattr(pipeline, "build_cursor_stream_params", fake_build_cursor_stream_params)
        monkeypatch.setattr(pipeline, "open_streaming_h2_request", lambda *args: _DummyStream())
        monkeypatch.setattr(pipeline, "consume_stream", fake_consume_stream)
        monkeypatch.setattr(pipeline, "load_fallback_config", lambda: FallbackConfig())
        monkeypatch.setattr(pipeline, "log_llm_request", lambda *args, **kwargs: "")
        monkeypatch.setattr(pipeline, "log_llm_response", lambda *args, **kwargs: "")
        monkeypatch.setattr(pipeline, "log_llm_api_call", lambda *args, **kwargs: None)

        result = asyncio.run(
            pipeline._call_cursor_direct(
                messages=[{"role": "user", "content": "create x.txt"}],
                model=initial_model,
                tools=TOOLS,
                valid_tool_names=["write_file"],
                auth_token="token",
            )
        )

        assert result["tool_calls"][0]["function"]["name"] == "write_file"
        initial_rendered = _request_text(request_messages[0])
        fallback_rendered = _request_text(request_messages[1])
        if initial_composer:
            assert "<|tool_calls_begin|>" in initial_rendered
            assert '"type":"tool_use"' not in initial_rendered
        else:
            assert "Available client tools are the following JSON schemas:" in initial_rendered
        assert "Available client tools are the following JSON schemas:" in fallback_rendered
        assert "<|tool_calls_begin|>" not in fallback_rendered


def _result_with_text(text: str) -> dict:
    return {
        "text": text,
        "thinking": "",
        "composer_tool_calls": [],
        "interrupted_tool_state": "",
        "has_fatal_error": False,
        "errors": [],
        "had_content": True,
        "metrics": {"chunk_count": 1, "first_chunk_latency_ms": 0},
    }


def _assert_standard_attempt_rejects_non_native_tool_output(monkeypatch, output: str) -> None:
    async def fake_consume_stream(*args, **kwargs):
        return _result_with_text(output)

    monkeypatch.setattr(pipeline, "build_cursor_stream_params", lambda *args: ("/chat", {}, b""))
    monkeypatch.setattr(pipeline, "open_streaming_h2_request", lambda *args: _DummyStream())
    monkeypatch.setattr(pipeline, "consume_stream", fake_consume_stream)
    monkeypatch.setattr(pipeline, "log_llm_request", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_response", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_api_call", lambda *args, **kwargs: None)

    result = asyncio.run(
        pipeline._call_cursor_direct(
            messages=[{"role": "user", "content": "write x.txt"}],
            model="standard-model",
            tools=TOOLS,
            valid_tool_names=["Write"],
            auth_token="token",
        )
    )

    assert result.get("tool_calls") is None


def test_standard_attempt_never_executes_prose_mimicry_or_malformed_json(monkeypatch):
    _assert_standard_attempt_rejects_non_native_tool_output(
        monkeypatch,
        'I will use Write {"file_path":"x.txt","content":"x"}.',
    )
    monkeypatch.undo()
    _assert_standard_attempt_rejects_non_native_tool_output(
        monkeypatch,
        '{"type":"tool_use","id":"toolu_bad","name":"Write",'
        '"input":{"file_path":"x.txt","content":"x",}}',
    )


def _sse_data_objects(sse: str) -> list[dict]:
    objects: list[dict] = []
    for event in sse.split("\n\n"):
        for line in event.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                objects.append(json.loads(line[len("data: "):]))
    return objects


def _visible_stream_text(sse: str, client_format: str) -> str:
    objects = _sse_data_objects(sse)
    if client_format == "openai":
        return "".join(
            (object_.get("choices") or [{}])[0].get("delta", {}).get("content", "")
            for object_ in objects
        )
    return "".join(
        object_.get("delta", {}).get("text", "")
        for object_ in objects
        if object_.get("type") == "content_block_delta"
    )


def _has_public_tool_call(sse: str, client_format: str) -> bool:
    objects = _sse_data_objects(sse)
    if client_format == "openai":
        return any(
            ((object_.get("choices") or [{}])[0].get("delta", {}).get("tool_calls") or [{}])[0]
            .get("function", {})
            .get("name") == "write_file"
            for object_ in objects
        )
    return any(
        (object_.get("content_block") or {}).get("type") == "tool_use"
        and (object_.get("content_block") or {}).get("name") == "write_file"
        for object_ in objects
    )


def test_pipeline_streams_whitespace_native_json_only_as_structured_tool_calls(monkeypatch):
    tool_json = (
        '{ "type" : "tool_use", "id" : "toolu_whitespace", '
        '"name" : "write_file", "input" : {"path":"x.txt","content":"x"} }'
    )
    fragments: list[str] = []

    async def fake_consume_stream(stream, composer=False, on_text_delta=None, on_thinking_delta=None):
        for fragment in fragments:
            if on_text_delta:
                on_text_delta(fragment)
        return _result_with_text(tool_json)

    monkeypatch.setattr(pipeline, "build_cursor_stream_params", lambda *args: ("/chat", {}, b""))
    monkeypatch.setattr(pipeline, "open_streaming_h2_request", lambda *args: _DummyStream())
    monkeypatch.setattr(pipeline, "consume_stream", fake_consume_stream)
    monkeypatch.setattr(pipeline, "log_llm_request", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_response", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_api_call", lambda *args, **kwargs: None)

    async def collect(result: dict) -> str:
        return "".join([chunk async for chunk in result["stream_handler"]()])

    for client_format, builder in (
        ("anthropic", pipeline._build_streaming_result_anthropic),
        ("openai", pipeline._build_streaming_result_openai),
    ):
        for split_point in range(1, len(tool_json)):
            fragments[:] = [tool_json[:split_point], tool_json[split_point:]]
            result = builder(
                request_id=f"whitespace-{client_format}-{split_point}",
                messages=[{"role": "user", "content": "write x.txt"}],
                tools=TOOLS,
                valid_tool_names=["write_file"],
                resolved_model="standard-model",
                max_tokens=None,
                token="token",
                pipeline_start=0,
                base_telemetry={},
            )
            sse = asyncio.run(collect(result))
            visible_text = _visible_stream_text(sse, client_format)

            assert tool_json not in visible_text, (client_format, split_point, sse)
            assert '"tool_use"' not in visible_text, (client_format, split_point, sse)
            assert _has_public_tool_call(sse, client_format), (client_format, split_point, sse)


def _composer_partial_consumed(thinking: str) -> dict:
    return {
        "text": "",
        "thinking": thinking,
        "composer_tool_calls": [],
        "interrupted_tool_state": "",
        "has_fatal_error": False,
        "errors": [],
        "had_content": True,
        "metrics": {"chunk_count": 1, "first_chunk_latency_ms": 0},
    }


def test_callback_fed_composer_partial_markers_repair_once_without_leaking(monkeypatch):
    complete_marker = (
        "</think><|tool_calls_begin|><|tool_call_begin|>write_file<|tool_sep|>"
        "path\nrepaired.txt<|tool_call_end|><|tool_calls_end|>"
    )

    for marker in ("<|tool_calls_beg", "<｜tool▁calls▁beg"):
        for resets_stream in (False, True):
            calls = {"count": 0}
            public_text: list[str] = []
            public_thinking: list[str] = []

            async def fake_consume_stream(
                stream,
                composer=False,
                on_text_delta=None,
                on_thinking_delta=None,
            ):
                calls["count"] += 1
                if calls["count"] == 1:
                    partial = f"reasoning</think>Visible before marker. {marker}"
                    if on_thinking_delta:
                        on_thinking_delta(partial)
                    if resets_stream:
                        raise pipeline.PartialStreamConsumptionError(
                            "stream reset", _composer_partial_consumed(partial)
                        )
                    return _composer_partial_consumed(partial)
                if on_thinking_delta:
                    on_thinking_delta(complete_marker)
                return _composer_partial_consumed(complete_marker)

            monkeypatch.setattr(pipeline, "build_cursor_stream_params", lambda *args: ("/chat", {}, b""))
            monkeypatch.setattr(pipeline, "open_streaming_h2_request", lambda *args: _DummyStream())
            monkeypatch.setattr(pipeline, "consume_stream", fake_consume_stream)
            monkeypatch.setattr(pipeline, "log_llm_request", lambda *args, **kwargs: "")
            monkeypatch.setattr(pipeline, "log_llm_response", lambda *args, **kwargs: "")
            monkeypatch.setattr(pipeline, "log_llm_api_call", lambda *args, **kwargs: None)

            result = asyncio.run(
                pipeline._call_cursor_direct(
                    messages=[{"role": "user", "content": "write repaired.txt"}],
                    model="composer-2.5",
                    tools=TOOLS,
                    valid_tool_names=["write_file"],
                    auth_token="token",
                    on_stream_delta=public_text.append,
                    on_thinking_delta=public_thinking.append,
                )
            )

            emitted = "".join([*public_text, *public_thinking])
            assert calls["count"] == 2, (marker, resets_stream)
            assert result["tool_calls"][0]["function"]["name"] == "write_file"
            assert "repaired.txt" in result["tool_calls"][0]["function"]["arguments"]
            assert "partial" not in result["tool_calls"][0]["function"]["arguments"]
            assert "<|tool" not in emitted, (marker, resets_stream, emitted)
            assert "<｜tool" not in emitted, (marker, resets_stream, emitted)
            assert "</think>" not in emitted, (marker, resets_stream, emitted)
            monkeypatch.undo()


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
