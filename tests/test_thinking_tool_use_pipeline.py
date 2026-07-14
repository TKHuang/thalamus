"""Regression tests for native tool_use JSON emitted through Cursor thinking."""

import asyncio
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proto import cursor_api_pb2 as pb  # noqa: E402
from claude_code import pipeline  # noqa: E402
from claude_code.normalizers import normalize_anthropic, normalize_openai  # noqa: E402
from claude_code.tool_protocols import ToolProtocol, create_protocol_adapter  # noqa: E402


DELTA_TARGET_SIZE = pipeline.DELTA_TARGET_SIZE


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


def _thinking_frame(text: str) -> bytes:
    resp = pb.StreamUnifiedChatWithToolsResponse()
    resp.message.thinking.content = text
    payload = resp.SerializeToString()
    return bytes([0]) + struct.pack(">I", len(payload)) + payload


async def _consume(frames: list[bytes]) -> dict:
    async def _iter():
        for frame in frames:
            yield frame

    return await pipeline.consume_stream(_iter(), composer=False)


def _sse_data_objects(sse: str) -> list[dict]:
    objs = []
    for event in sse.split("\n\n"):
        for line in event.splitlines():
            if line.startswith("data: "):
                payload = line[len("data: "):]
                if payload != "[DONE]":
                    objs.append(json.loads(payload))
    return objs


def test_non_composer_thinking_tool_use_is_parse_candidate():
    payload = (
        "Creating the file now.\n"
        '{"type":"tool_use","id":"toolu_01","name":"Write",'
        '"input":{"file_path":"/tmp/qingming.html","content":"<html></html>"}}'
    )
    consumed = asyncio.run(_consume([_thinking_frame(payload)]))

    decoder = pipeline._ProtocolStreamDecoder(
        create_protocol_adapter(ToolProtocol.STANDARD_JSON_V1)
    )
    decoded = decoder.attach(consumed)
    parsed, source = pipeline._parse_tool_calls_from_consumed(decoded)

    assert source == "reasoning"
    assert parsed[0].raw_name == "Write"
    assert parsed[0].arguments["file_path"] == "/tmp/qingming.html"


def test_anthropic_stream_hides_thinking_tool_use_json(monkeypatch):
    async def fake_call_cursor_direct(
        messages,
        model,
        tools,
        valid_tool_names,
        auth_token,
        on_stream_delta=None,
        on_thinking_delta=None,
        compact_tools=False,
    ):
        if on_thinking_delta:
            on_thinking_delta("Creating the file now.\n")
            on_thinking_delta(
                '{"type":"tool_use","id":"toolu_01","name":"Write",'
                '"input":{"file_path":"/tmp/qingming.html","content":"<html></html>"}}'
            )
        return {
            "tool_calls": [
                {
                    "id": "toolu_01",
                    "type": "function",
                    "function": {
                        "name": "Write",
                        "arguments": '{"file_path":"/tmp/qingming.html","content":"<html></html>"}',
                    },
                }
            ],
            "text": "",
            "thinking": "",
            "model": model,
            "fallback_attempts": 0,
            "stats": {},
        }

    monkeypatch.setattr(pipeline, "_call_cursor_direct", fake_call_cursor_direct)
    result = pipeline._build_streaming_result_anthropic(
        request_id="req_test",
        messages=[],
        tools=[],
        valid_tool_names=["Write"],
        resolved_model="grok-4.5",
        max_tokens=None,
        token="tok",
        pipeline_start=0,
        base_telemetry={},
    )

    async def collect() -> str:
        parts = []
        async for chunk in result["stream_handler"]():
            parts.append(chunk)
        return "".join(parts)

    sse = asyncio.run(collect())
    objs = _sse_data_objects(sse)
    text = "".join(
        obj.get("delta", {}).get("text", "")
        for obj in objs
        if obj.get("type") == "content_block_delta"
    )
    has_tool_block = any(
        (obj.get("content_block") or {}).get("type") == "tool_use"
        and (obj.get("content_block") or {}).get("name") == "Write"
        for obj in objs
    )

    assert '{"type":"tool_use"' not in sse, sse
    assert has_tool_block, sse
    assert "Creating the file now." in text, sse


def test_openai_stream_hides_text_tool_use_json(monkeypatch):
    async def fake_call_cursor_direct(
        messages,
        model,
        tools,
        valid_tool_names,
        auth_token,
        on_stream_delta=None,
        on_thinking_delta=None,
        compact_tools=False,
    ):
        if on_stream_delta:
            on_stream_delta("Creating the file now.\n")
            on_stream_delta(
                '{"type":"tool_use","id":"toolu_01","name":"Write",'
                '"input":{"file_path":"/tmp/qingming.html","content":"<html></html>"}}'
            )
        return {
            "tool_calls": [
                {
                    "id": "toolu_01",
                    "type": "function",
                    "function": {
                        "name": "Write",
                        "arguments": '{"file_path":"/tmp/qingming.html","content":"<html></html>"}',
                    },
                }
            ],
            "text": "",
            "thinking": "",
            "model": model,
            "fallback_attempts": 0,
            "stats": {},
        }

    monkeypatch.setattr(pipeline, "_call_cursor_direct", fake_call_cursor_direct)
    result = pipeline._build_streaming_result_openai(
        request_id="req_test",
        messages=[],
        tools=[],
        valid_tool_names=["Write"],
        resolved_model="grok-4.5",
        max_tokens=None,
        token="tok",
        pipeline_start=0,
        base_telemetry={},
    )

    async def collect() -> str:
        parts = []
        async for chunk in result["stream_handler"]():
            parts.append(chunk)
        return "".join(parts)

    sse = asyncio.run(collect())
    objs = _sse_data_objects(sse)
    text = "".join(
        (obj.get("choices") or [{}])[0].get("delta", {}).get("content", "")
        for obj in objs
    )
    has_tool_call = any(
        (
            ((obj.get("choices") or [{}])[0].get("delta", {}).get("tool_calls") or [{}])[0]
            .get("function", {})
            .get("name")
        ) == "Write"
        for obj in objs
    )

    assert '{"type":"tool_use"' not in text, sse
    assert has_tool_call, sse
    assert "Creating the file now." in text, sse


def test_compact_continuation_suppresses_preliminary_text_when_tool_call_succeeds(monkeypatch):
    class DummyStream:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    calls = {"count": 0}
    streamed: list[str] = []

    def fake_build_cursor_stream_params(auth_token, messages, model):
        return "/chat", {}, b""

    def fake_open_streaming_h2_request(path, headers, body):
        return DummyStream()

    async def fake_consume_stream(
        stream,
        composer=False,
        on_stream_delta=None,
        on_text_delta=None,
        on_thinking_delta=None,
    ):
        text_callback = on_stream_delta or on_text_delta
        calls["count"] += 1
        if calls["count"] == 1:
            if text_callback:
                text_callback("I will create the file.")
            return {
                "text": "I will create the file.",
                "thinking": "",
                "composer_tool_calls": [],
                "has_fatal_error": False,
                "errors": [],
                "had_content": True,
                "metrics": {"chunk_count": 1, "first_chunk_latency_ms": 0},
            }

        if text_callback:
            text_callback("Creating it now.\n")
            text_callback(
                '{"type":"tool_use","id":"toolu_02","name":"write_file",'
                '"input":{"path":"/tmp/q.html","content":"<html></html>"}}'
            )
        return {
            "text": (
                "Creating it now.\n"
                '{"type":"tool_use","id":"toolu_02","name":"write_file",'
                '"input":{"path":"/tmp/q.html","content":"<html></html>"}}'
            ),
            "thinking": "",
            "composer_tool_calls": [],
            "has_fatal_error": False,
            "errors": [],
            "had_content": True,
            "metrics": {"chunk_count": 2, "first_chunk_latency_ms": 0},
        }

    monkeypatch.setattr(pipeline, "build_cursor_stream_params", fake_build_cursor_stream_params)
    monkeypatch.setattr(pipeline, "open_streaming_h2_request", fake_open_streaming_h2_request)
    monkeypatch.setattr(pipeline, "consume_stream", fake_consume_stream)
    monkeypatch.setattr(pipeline, "log_llm_request", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_response", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_api_call", lambda *args, **kwargs: None)

    result = asyncio.run(
        pipeline._call_cursor_direct(
            messages=[{"role": "user", "content": "make q.html"}],
            model="grok-4.5",
            tools=[],
            valid_tool_names=["write_file"],
            auth_token="tok",
            on_stream_delta=streamed.append,
            compact_tools=True,
        )
    )

    assert calls["count"] == 2
    assert streamed == []
    assert result["tool_calls"][0]["function"]["name"] == "write_file"


def test_compact_continuation_prompt_coerces_promised_write_file(monkeypatch):
    class DummyStream:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    calls = {"count": 0}
    request_messages: list[list[dict]] = []

    def fake_build_cursor_stream_params(auth_token, messages, model):
        request_messages.append(list(messages))
        return "/chat", {}, b""

    def fake_open_streaming_h2_request(path, headers, body):
        return DummyStream()

    async def fake_consume_stream(stream, composer=False, on_text_delta=None, on_thinking_delta=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "text": "改用指定的 tool_use JSON 格式直接建立檔案。",
                "thinking": "",
                "composer_tool_calls": [],
                "has_fatal_error": False,
                "errors": [],
                "had_content": True,
                "metrics": {"chunk_count": 1, "first_chunk_latency_ms": 0},
            }

        continuation_prompt = request_messages[-1][-1]["content"]
        assert '"name":"write_file"' in continuation_prompt
        assert '"path":"qingming-3d-grok45.html"' in continuation_prompt
        assert "<ToolName>" not in continuation_prompt

        return {
            "text": (
                '{"type":"tool_use","id":"toolu_write","name":"write_file",'
                '"input":{"path":"qingming-3d-grok45.html","content":"<!DOCTYPE html><html></html>"}}'
            ),
            "thinking": "",
            "composer_tool_calls": [],
            "has_fatal_error": False,
            "errors": [],
            "had_content": True,
            "metrics": {"chunk_count": 1, "first_chunk_latency_ms": 0},
        }

    monkeypatch.setattr(pipeline, "build_cursor_stream_params", fake_build_cursor_stream_params)
    monkeypatch.setattr(pipeline, "open_streaming_h2_request", fake_open_streaming_h2_request)
    monkeypatch.setattr(pipeline, "consume_stream", fake_consume_stream)
    monkeypatch.setattr(pipeline, "log_llm_request", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_response", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_api_call", lambda *args, **kwargs: None)

    result = asyncio.run(
        pipeline._call_cursor_direct(
            messages=[
                {
                    "role": "user",
                    "content": "寫一個 qingming-3d-grok45.html 不准看其他檔案",
                }
            ],
            model="grok-4.5",
            tools=[],
            valid_tool_names=["skill_view", "write_file", "terminal"],
            auth_token="tok",
            compact_tools=True,
        )
    )

    assert calls["count"] == 2
    assert result["tool_calls"][0]["function"]["name"] == "write_file"
    assert "qingming-3d-grok45.html" in result["tool_calls"][0]["function"]["arguments"]


def test_current_openai_tool_result_accepts_any_nonempty_final_text():
    messages = [
        {"role": "user", "content": "make q.html"},
        {
            "role": "assistant",
            "content": "Creating it now.",
            "tool_calls": [
                {
                    "id": "toolu_01",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_01", "content": '{"bytes_written":123}'},
    ]

    assert pipeline._should_accept_final_text_without_continuation(
        messages,
        "完成，檔案在這裡：/tmp/q.html",
    )
    assert pipeline._should_accept_final_text_without_continuation(
        messages,
        "頁面已載入，接著確認 3D 場景是否正常渲染。",
    )
    assert not pipeline._should_accept_final_text_without_continuation(messages, " \n\t ")


def test_final_text_shortcut_requires_a_matching_current_tool_result_id():
    openai_request = normalize_openai(
        {
            "messages": [
                {"role": "user", "content": "Read config.txt."},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_read",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_read", "content": "contents"},
            ]
        }
    )
    anthropic_request = normalize_anthropic(
        {
            "model": "claude-opus-4-6",
            "messages": [
                {"role": "user", "content": "Read config.txt."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_read",
                            "name": "read_file",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_read",
                            "content": "contents",
                        }
                    ],
                },
            ],
        }
    )

    assert pipeline._should_accept_final_text_without_continuation(
        openai_request.messages, "The configuration is valid."
    )
    assert pipeline._should_accept_final_text_without_continuation(
        anthropic_request.messages, "The configuration is valid."
    )

    assistant = openai_request.messages[-2]
    assert not pipeline._should_accept_final_text_without_continuation(
        [assistant, {"role": "tool", "content": "contents"}], "Done."
    )
    assert not pipeline._should_accept_final_text_without_continuation(
        [
            assistant,
            {"role": "tool", "tool_call_id": "call_other", "content": "contents"},
        ],
        "Done.",
    )
    assert not pipeline._should_accept_final_text_without_continuation(
        [
            {"role": "assistant", "tool_calls": [{"id": "", "function": {"name": "read_file"}}]},
            {"role": "tool", "tool_call_id": "call_read", "content": "contents"},
        ],
        "Done.",
    )


def _assert_read_result_text_ends_turn(monkeypatch, messages: list[dict], compact_tools: bool):
    class DummyStream:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    calls = {"count": 0}
    traces: list[tuple[str, object]] = []

    def fake_build_cursor_stream_params(auth_token, messages, model):
        return "/chat", {}, b""

    def fake_open_streaming_h2_request(path, headers, body):
        return DummyStream()

    async def fake_consume_stream(stream, composer=False, on_text_delta=None, on_thinking_delta=None):
        calls["count"] += 1
        return {
            "text": "READ_OPUS_OK",
            "thinking": "",
            "composer_tool_calls": [],
            "has_fatal_error": False,
            "errors": [],
            "had_content": True,
            "metrics": {"chunk_count": 1, "first_chunk_latency_ms": 0},
        }

    monkeypatch.setattr(pipeline, "build_cursor_stream_params", fake_build_cursor_stream_params)
    monkeypatch.setattr(pipeline, "open_streaming_h2_request", fake_open_streaming_h2_request)
    monkeypatch.setattr(pipeline, "consume_stream", fake_consume_stream)
    monkeypatch.setattr(pipeline, "log_llm_request", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_response", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_api_call", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "emit_attempt_trace",
        lambda trace: traces.append(("attempt", trace)),
    )
    monkeypatch.setattr(
        pipeline,
        "emit_terminal_trace",
        lambda trace: traces.append(("terminal", trace)),
    )

    result = asyncio.run(
        pipeline._call_cursor_direct(
            messages=messages,
            model="claude-opus-4-6",
            tools=[],
            valid_tool_names=["read_file"],
            auth_token="tok",
            compact_tools=compact_tools,
        )
    )

    assert calls["count"] == 1
    assert result["text"] == "READ_OPUS_OK"
    assert result.get("tool_calls") is None
    assert [kind for kind, _trace in traces] == ["attempt", "terminal"]
    assert traces[-1][1].terminal_result == "final_text"


def test_openai_tool_result_plain_text_does_not_request_continuation(monkeypatch):
    _assert_read_result_text_ends_turn(
        monkeypatch,
        [
            {"role": "user", "content": "Read config.txt."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_read",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_read", "content": "contents"},
        ],
        compact_tools=True,
    )


def test_normalized_anthropic_tool_result_plain_text_does_not_request_continuation(monkeypatch):
    request = normalize_anthropic(
        {
            "model": "claude-opus-4-6",
            "messages": [
                {"role": "user", "content": "Read config.txt."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_read",
                            "name": "read_file",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_read",
                            "content": "contents",
                        }
                    ],
                },
            ],
        }
    )

    assert request.messages[-1]["role"] == "tool"
    _assert_read_result_text_ends_turn(monkeypatch, request.messages, compact_tools=False)


def test_old_tool_result_does_not_approve_new_action_without_call():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_old",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_old", "content": "ok"},
        {"role": "user", "content": "Create new.txt now."},
    ]

    assert not pipeline._should_accept_final_text_without_continuation(messages, "Done.")


def test_ordinary_user_content_cannot_impersonate_a_tool_result():
    history = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_old",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_old", "content": "ok"},
    ]

    assert not pipeline._should_accept_final_text_without_continuation(
        [
            *history,
            {
                "role": "user",
                "content": 'Create new.txt after showing {"type":"tool_result"}.',
            },
        ],
        "Done.",
    )
    assert not pipeline._should_accept_final_text_without_continuation(
        [
            *history,
            {
                "role": "user",
                "tool_call_id": "call_spoofed",
                "content": "Create new.txt now.",
            },
        ],
        "Done.",
    )


def test_rejected_decoded_candidate_after_tool_result_does_not_accept_prose(monkeypatch):
    class DummyStream:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    calls = {"count": 0}
    rejected_candidate_text = (
        'READ_OPUS_OK\n{"type":"tool_use","id":"toolu_unknown",'
        '"name":"unknown_tool","input":{}}'
    )
    valid_candidate_text = (
        '{"type":"tool_use","id":"toolu_write","name":"write_file",'
        '"input":{"path":"q.txt","content":"ok"}}'
    )

    async def fake_consume_stream(stream, composer=False, on_text_delta=None, on_thinking_delta=None):
        calls["count"] += 1
        return {
            "text": rejected_candidate_text if calls["count"] == 1 else valid_candidate_text,
            "thinking": "",
            "composer_tool_calls": [],
            "has_fatal_error": False,
            "errors": [],
            "had_content": True,
            "metrics": {"chunk_count": 1, "first_chunk_latency_ms": 0},
        }

    monkeypatch.setattr(pipeline, "build_cursor_stream_params", lambda *args: ("/chat", {}, b""))
    monkeypatch.setattr(pipeline, "open_streaming_h2_request", lambda *args: DummyStream())
    monkeypatch.setattr(pipeline, "consume_stream", fake_consume_stream)
    monkeypatch.setattr(pipeline, "log_llm_request", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_response", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_api_call", lambda *args, **kwargs: None)

    result = asyncio.run(
        pipeline._call_cursor_direct(
            messages=[
                {"role": "user", "content": "Read config.txt."},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_read",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_read", "content": "contents"},
            ],
            model="claude-opus-4-6",
            tools=[],
            valid_tool_names=["write_file"],
            auth_token="tok",
        )
    )

    assert calls["count"] == 2
    assert result["tool_calls"][0]["function"]["name"] == "write_file"


def test_thinking_tool_json_does_not_suppress_later_visible_text(monkeypatch):
    async def fake_call_cursor_direct(
        messages,
        model,
        tools,
        valid_tool_names,
        auth_token,
        on_stream_delta=None,
        on_thinking_delta=None,
        compact_tools=False,
    ):
        if on_thinking_delta:
            on_thinking_delta('reasoning {"type":"tool_use","id":"toolu_1",')
        if on_stream_delta:
            on_stream_delta("Visible result text.")
        return {
            "tool_calls": [],
            "text": "",
            "thinking": "",
            "model": model,
            "fallback_attempts": 0,
            "stats": {},
        }

    monkeypatch.setattr(pipeline, "_call_cursor_direct", fake_call_cursor_direct)
    result = pipeline._build_streaming_result_anthropic(
        request_id="req_test",
        messages=[],
        tools=[],
        valid_tool_names=[],
        resolved_model="grok-4.5",
        max_tokens=None,
        token="tok",
        pipeline_start=0,
        base_telemetry={},
    )

    async def collect() -> str:
        return "".join([chunk async for chunk in result["stream_handler"]()])

    sse = asyncio.run(collect())
    text = "".join(
        obj.get("delta", {}).get("text", "")
        for obj in _sse_data_objects(sse)
        if obj.get("type") == "content_block_delta"
    )

    assert "Visible result text." in text, sse
    assert '"type":"tool_use"' not in text, sse


def _assert_initial_interruption_repair(monkeypatch, compact_tools: bool):
    class DummyStream:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    calls = {"count": 0}

    def fake_build_cursor_stream_params(auth_token, messages, model):
        return "/chat", {}, b""

    def fake_open_streaming_h2_request(path, headers, body):
        return DummyStream()

    async def fake_consume_stream(stream, composer=False, on_text_delta=None, on_thinking_delta=None):
        calls["count"] += 1
        if calls["count"] == 1:
            partial = {
                "text": (
                    '直接寫單檔。\n{"type":"tool_use","id":"toolu_03",'
                    '"name":"write_file","input":{"path":"/tmp/q.html","content":"<!DOCTYPE html>'
                ),
                "thinking": "",
                "composer_tool_calls": [],
                "has_fatal_error": False,
                "errors": [],
                "had_content": True,
                "metrics": {"chunk_count": 1200, "first_chunk_latency_ms": 0},
            }
            raise pipeline.PartialStreamConsumptionError(
                "<StreamReset stream_id:1, error_code:2, remote_reset:True>",
                partial,
            )

        return {
            "text": (
                '{"type":"tool_use","id":"toolu_retry","name":"write_file",'
                '"input":{"path":"/tmp/q.html","content":"<!DOCTYPE html><html></html>"}}'
            ),
            "thinking": "",
            "composer_tool_calls": [],
            "has_fatal_error": False,
            "errors": [],
            "had_content": True,
            "metrics": {"chunk_count": 2, "first_chunk_latency_ms": 0},
        }

    monkeypatch.setattr(pipeline, "build_cursor_stream_params", fake_build_cursor_stream_params)
    monkeypatch.setattr(pipeline, "open_streaming_h2_request", fake_open_streaming_h2_request)
    monkeypatch.setattr(pipeline, "consume_stream", fake_consume_stream)
    monkeypatch.setattr(pipeline, "log_llm_request", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_response", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_api_call", lambda *args, **kwargs: None)

    result = asyncio.run(
        pipeline._call_cursor_direct(
            messages=[{"role": "user", "content": "make q.html"}],
            model="grok-4.5",
            tools=[],
            valid_tool_names=["write_file"],
            auth_token="tok",
            compact_tools=compact_tools,
        )
    )

    assert calls["count"] == 2
    assert result.get("error") is None
    assert result["tool_calls"][0]["function"]["name"] == "write_file"
    assert "/tmp/q.html" in result["tool_calls"][0]["function"]["arguments"]


def test_stream_reset_during_tool_json_retries_with_complete_tool_call(monkeypatch):
    _assert_initial_interruption_repair(monkeypatch, compact_tools=True)


def test_noncompact_initial_interruption_repairs_with_complete_tool_call(monkeypatch):
    _assert_initial_interruption_repair(monkeypatch, compact_tools=False)


def _assert_continuation_interruption_repair(monkeypatch, compact_tools: bool):
    class DummyStream:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    calls = {"count": 0}

    def fake_build_cursor_stream_params(auth_token, messages, model):
        return "/chat", {}, b""

    def fake_open_streaming_h2_request(path, headers, body):
        return DummyStream()

    async def fake_consume_stream(stream, composer=False, on_text_delta=None, on_thinking_delta=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "text": "I will create the file.",
                "thinking": "",
                "composer_tool_calls": [],
                "has_fatal_error": False,
                "errors": [],
                "had_content": True,
                "metrics": {"chunk_count": 1, "first_chunk_latency_ms": 0},
            }
        if calls["count"] == 2:
            partial = {
                "text": (
                    '{"type":"tool_use","id":"toolu_partial",'
                    '"name":"write_file","input":{"path":"/tmp/q.html"'
                ),
                "thinking": "",
                "composer_tool_calls": [],
                "has_fatal_error": False,
                "errors": [],
                "had_content": True,
                "metrics": {"chunk_count": 1, "first_chunk_latency_ms": 0},
            }
            raise pipeline.PartialStreamConsumptionError("stream reset", partial)
        return {
            "text": (
                '{"type":"tool_use","id":"toolu_repaired","name":"write_file",'
                '"input":{"path":"/tmp/q.html","content":"ok"}}'
            ),
            "thinking": "",
            "composer_tool_calls": [],
            "has_fatal_error": False,
            "errors": [],
            "had_content": True,
            "metrics": {"chunk_count": 1, "first_chunk_latency_ms": 0},
        }

    monkeypatch.setattr(pipeline, "build_cursor_stream_params", fake_build_cursor_stream_params)
    monkeypatch.setattr(pipeline, "open_streaming_h2_request", fake_open_streaming_h2_request)
    monkeypatch.setattr(pipeline, "consume_stream", fake_consume_stream)
    monkeypatch.setattr(pipeline, "log_llm_request", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_response", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_api_call", lambda *args, **kwargs: None)

    result = asyncio.run(
        pipeline._call_cursor_direct(
            messages=[{"role": "user", "content": "make q.html"}],
            model="grok-4.5",
            tools=[],
            valid_tool_names=["write_file"],
            auth_token="tok",
            compact_tools=compact_tools,
        )
    )

    assert calls["count"] == 3
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "write_file"


def test_continuation_interruption_repairs_once_with_complete_tool_call(monkeypatch):
    _assert_continuation_interruption_repair(monkeypatch, compact_tools=True)


def test_noncompact_continuation_interruption_repairs_with_complete_tool_call(monkeypatch):
    _assert_continuation_interruption_repair(monkeypatch, compact_tools=False)


def test_compact_tool_call_suppresses_stale_attempt_prose(monkeypatch):
    class DummyStream:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    calls = {"count": 0}

    def fake_build_cursor_stream_params(auth_token, messages, model):
        return "/chat", {}, b""

    def fake_open_streaming_h2_request(path, headers, body):
        return DummyStream()

    async def fake_consume_stream(stream, composer=False, on_text_delta=None, on_thinking_delta=None):
        calls["count"] += 1
        callback = on_text_delta
        if calls["count"] == 1:
            if callback:
                callback("This session has no file-writing tool.")
            return {
                "text": "This session has no file-writing tool.",
                "thinking": "",
                "composer_tool_calls": [],
                "has_fatal_error": False,
                "errors": [],
                "had_content": True,
                "metrics": {"chunk_count": 1, "first_chunk_latency_ms": 0},
            }
        tool_json = (
            '{"type":"tool_use","id":"toolu_write","name":"write_file",'
            '"input":{"path":"q.html","content":"<html></html>"}}'
        )
        if callback:
            callback(tool_json)
        return {
            "text": tool_json,
            "thinking": "",
            "composer_tool_calls": [],
            "has_fatal_error": False,
            "errors": [],
            "had_content": True,
            "metrics": {"chunk_count": 1, "first_chunk_latency_ms": 0},
        }

    monkeypatch.setattr(pipeline, "build_cursor_stream_params", fake_build_cursor_stream_params)
    monkeypatch.setattr(pipeline, "open_streaming_h2_request", fake_open_streaming_h2_request)
    monkeypatch.setattr(pipeline, "consume_stream", fake_consume_stream)
    monkeypatch.setattr(pipeline, "log_llm_request", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_response", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_api_call", lambda *args, **kwargs: None)

    result = pipeline._build_streaming_result_openai(
        request_id="req_test",
        messages=[{"role": "user", "content": "make q.html"}],
        tools=[],
        valid_tool_names=["write_file"],
        resolved_model="grok-4.5",
        max_tokens=None,
        token="tok",
        pipeline_start=0,
        base_telemetry={},
    )

    async def collect() -> str:
        return "".join([chunk async for chunk in result["stream_handler"]()])

    sse = asyncio.run(collect())
    objs = _sse_data_objects(sse)
    has_tool_call = any(
        (
            ((obj.get("choices") or [{}])[0].get("delta", {}).get("tool_calls") or [{}])[0]
            .get("function", {})
            .get("name")
        ) == "write_file"
        for obj in objs
    )

    assert calls["count"] == 2
    assert has_tool_call, sse
    assert "This session has no file-writing tool." not in sse
    assert '"type":"tool_use"' not in sse


def test_failed_repair_spends_allowance_before_fallback(monkeypatch):
    class DummyStream:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FallbackConfig:
        max_attempts = 2

        @staticmethod
        def should_fallback(error_text):
            return "connection reset" in error_text

        @staticmethod
        def select_next_model(requested, tried):
            return "fallback-model" if requested == "primary-model" and len(tried) == 1 else None

    calls = {"count": 0}
    requested_models: list[str] = []

    def partial() -> dict:
        return {
            "text": (
                '{"type":"tool_use","id":"toolu_partial",'
                '"name":"write_file","input":{"path":"/tmp/q.html"'
            ),
            "thinking": "",
            "composer_tool_calls": [],
            "has_fatal_error": False,
            "errors": [],
            "had_content": True,
            "metrics": {"chunk_count": 1, "first_chunk_latency_ms": 0},
        }

    def fake_build_cursor_stream_params(auth_token, messages, model):
        requested_models.append(model)
        return "/chat", {}, b""

    def fake_open_streaming_h2_request(path, headers, body):
        return DummyStream()

    async def fake_consume_stream(stream, composer=False, on_text_delta=None, on_thinking_delta=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise pipeline.PartialStreamConsumptionError("initial interruption", partial())
        if calls["count"] == 2:
            raise pipeline.PartialStreamConsumptionError("connection reset", partial())
        raise pipeline.PartialStreamConsumptionError("second initial interruption", partial())

    monkeypatch.setattr(pipeline, "build_cursor_stream_params", fake_build_cursor_stream_params)
    monkeypatch.setattr(pipeline, "open_streaming_h2_request", fake_open_streaming_h2_request)
    monkeypatch.setattr(pipeline, "consume_stream", fake_consume_stream)
    monkeypatch.setattr(pipeline, "load_fallback_config", lambda: FallbackConfig())
    monkeypatch.setattr(pipeline, "log_llm_request", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_response", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_api_call", lambda *args, **kwargs: None)

    result = asyncio.run(
        pipeline._call_cursor_direct(
            messages=[{"role": "user", "content": "make q.html"}],
            model="primary-model",
            tools=[],
            valid_tool_names=["write_file"],
            auth_token="tok",
            compact_tools=False,
        )
    )

    assert calls["count"] == 3
    assert requested_models == ["primary-model", "primary-model", "fallback-model"]
    assert "second initial interruption" in result["error"]


def test_openai_stream_yields_reasoning_before_a_tool_enabled_cursor_call_completes(monkeypatch):
    """Hermes receives reasoning promptly, rather than only after Cursor finishes."""
    reasoning_callback_called = asyncio.Event()
    allow_cursor_return = asyncio.Event()
    cursor_returned = asyncio.Event()

    async def fake_call_cursor_direct(
        messages,
        model,
        tools,
        valid_tool_names,
        auth_token,
        on_stream_delta=None,
        on_thinking_delta=None,
        compact_tools=False,
    ):
        assert tools and valid_tool_names == ["write_file"]
        assert compact_tools is True
        if on_thinking_delta:
            on_thinking_delta("Inspecting the requested path.")
        reasoning_callback_called.set()
        await allow_cursor_return.wait()
        cursor_returned.set()
        return {
            "tool_calls": [],
            "text": "",
            "thinking": "",
            "model": model,
            "fallback_attempts": 0,
            "stats": {},
        }

    monkeypatch.setattr(pipeline, "_call_cursor_direct", fake_call_cursor_direct)
    result = pipeline._build_streaming_result_openai(
        request_id="reasoning-before-complete",
        messages=[{"role": "user", "content": "Write x.txt."}],
        tools=[{"name": "write_file", "input_schema": {"type": "object"}}],
        valid_tool_names=["write_file"],
        resolved_model="gpt-5.6-sol",
        max_tokens=None,
        token="tok",
        pipeline_start=0,
        base_telemetry={},
    )

    async def assert_incremental_reasoning() -> None:
        stream = result["stream_handler"]()
        await anext(stream)  # Initial assistant-role chunk is not reasoning.
        next_event = asyncio.create_task(anext(stream))
        await asyncio.wait_for(reasoning_callback_called.wait(), timeout=0.2)
        reasoning_event = await asyncio.wait_for(next_event, timeout=0.2)

        assert not cursor_returned.is_set()
        delta = _sse_data_objects(reasoning_event)[0]["choices"][0]["delta"]
        assert set(delta) == {"reasoning_content"}
        assert delta["reasoning_content"] in "Inspecting the requested path."

        allow_cursor_return.set()
        async for _chunk in stream:
            pass

    asyncio.run(assert_incremental_reasoning())


def test_openai_reasoning_lane_split_tool_json_is_hidden_and_renders_a_tool_call(monkeypatch):
    """Adapter-decoded native JSON must never leak through either OpenAI delta lane."""
    tool_json = (
        '{"type":"tool_use","id":"toolu_reasoning","name":"write_file",'
        '"input":{"path":"x.txt","content":"ok"}}'
    )

    class DummyStream:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    async def fake_consume_stream(
        stream,
        composer=False,
        on_text_delta=None,
        on_thinking_delta=None,
    ):
        assert composer is False
        assert on_thinking_delta is not None
        for fragment in ("Planning the write. ", tool_json[:37], tool_json[37:]):
            on_thinking_delta(fragment)
        return {
            "text": "",
            "thinking": "Planning the write. " + tool_json,
            "composer_tool_calls": [],
            "interrupted_tool_state": "",
            "has_fatal_error": False,
            "errors": [],
            "had_content": True,
            "metrics": {"chunk_count": 3, "first_chunk_latency_ms": 0},
        }

    monkeypatch.setattr(pipeline, "build_cursor_stream_params", lambda *args: ("/chat", {}, b""))
    monkeypatch.setattr(pipeline, "open_streaming_h2_request", lambda *args: DummyStream())
    monkeypatch.setattr(pipeline, "consume_stream", fake_consume_stream)
    monkeypatch.setattr(pipeline, "log_llm_request", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_response", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_api_call", lambda *args, **kwargs: None)

    result = pipeline._build_streaming_result_openai(
        request_id="reasoning-tool-json",
        messages=[{"role": "user", "content": "Write x.txt."}],
        tools=[{"name": "write_file", "input_schema": {"type": "object"}}],
        valid_tool_names=["write_file"],
        resolved_model="gpt-5.6-sol",
        max_tokens=None,
        token="tok",
        pipeline_start=0,
        base_telemetry={},
    )

    async def collect() -> str:
        return "".join([chunk async for chunk in result["stream_handler"]()])

    sse = asyncio.run(collect())
    deltas = [
        (object_.get("choices") or [{}])[0].get("delta", {})
        for object_ in _sse_data_objects(sse)
    ]
    reasoning = "".join(delta.get("reasoning_content", "") for delta in deltas)
    content = "".join(delta.get("content", "") for delta in deltas)

    assert reasoning == "Planning the write. "
    assert tool_json not in reasoning
    assert tool_json not in content
    assert '"type":"tool_use"' not in sse
    assert any(
        (delta.get("tool_calls") or [{}])[0].get("function", {}).get("name") == "write_file"
        for delta in deltas
    )


def test_openai_reasoning_and_visible_text_share_max_token_budget(monkeypatch):
    """Reasoning must not bypass the same output ceiling enforced for visible text."""
    async def fake_call_cursor_direct(
        messages,
        model,
        tools,
        valid_tool_names,
        auth_token,
        on_stream_delta=None,
        on_thinking_delta=None,
        compact_tools=False,
    ):
        assert on_thinking_delta is not None
        assert on_stream_delta is not None
        on_thinking_delta("abcdef")
        on_stream_delta("visible output")
        return {
            "tool_calls": [],
            "text": "",
            "thinking": "",
            "model": model,
            "fallback_attempts": 0,
            "stats": {},
        }

    monkeypatch.setattr(pipeline, "_call_cursor_direct", fake_call_cursor_direct)
    result = pipeline._build_streaming_result_openai(
        request_id="shared-budget",
        messages=[],
        tools=[],
        valid_tool_names=[],
        resolved_model="gpt-5.6-sol",
        max_tokens=1,
        token="tok",
        pipeline_start=0,
        base_telemetry={},
    )

    async def collect() -> str:
        return "".join([chunk async for chunk in result["stream_handler"]()])

    objects = _sse_data_objects(asyncio.run(collect()))
    deltas = [(object_.get("choices") or [{}])[0].get("delta", {}) for object_ in objects]
    reasoning = "".join(delta.get("reasoning_content", "") for delta in deltas)
    content = "".join(delta.get("content", "") for delta in deltas)

    assert reasoning == "abcd"
    assert content == ""
    assert objects[-1]["choices"][0]["finish_reason"] == "length"


def test_openai_large_reasoning_callback_enqueues_one_sse_event(monkeypatch):
    """A single sanitized upstream callback must not synchronously amplify queue depth."""
    large_reasoning = "r" * (DELTA_TARGET_SIZE * 4 + 1)

    async def fake_call_cursor_direct(
        messages,
        model,
        tools,
        valid_tool_names,
        auth_token,
        on_stream_delta=None,
        on_thinking_delta=None,
        compact_tools=False,
    ):
        assert on_thinking_delta is not None
        on_thinking_delta(large_reasoning)
        return {
            "tool_calls": [],
            "text": "",
            "thinking": "",
            "model": model,
            "fallback_attempts": 0,
            "stats": {},
        }

    monkeypatch.setattr(pipeline, "_call_cursor_direct", fake_call_cursor_direct)
    result = pipeline._build_streaming_result_openai(
        request_id="single-reasoning-event",
        messages=[],
        tools=[],
        valid_tool_names=[],
        resolved_model="gpt-5.6-sol",
        max_tokens=None,
        token="tok",
        pipeline_start=0,
        base_telemetry={},
    )

    async def collect() -> str:
        return "".join([chunk async for chunk in result["stream_handler"]()])

    deltas = [
        (object_.get("choices") or [{}])[0].get("delta", {})
        for object_ in _sse_data_objects(asyncio.run(collect()))
    ]
    reasoning_events = [delta for delta in deltas if "reasoning_content" in delta]

    assert reasoning_events == [{"reasoning_content": large_reasoning}]


def test_closing_openai_stream_cancels_unfinished_cursor_task(monkeypatch):
    """Disconnecting a client must not leave a Cursor request running in the background."""
    cursor_started = asyncio.Event()
    cursor_cancelled = asyncio.Event()
    never_return = asyncio.Event()

    async def fake_call_cursor_direct(
        messages,
        model,
        tools,
        valid_tool_names,
        auth_token,
        on_stream_delta=None,
        on_thinking_delta=None,
        compact_tools=False,
    ):
        cursor_started.set()
        try:
            await never_return.wait()
        except asyncio.CancelledError:
            cursor_cancelled.set()
            raise
        return {"model": model}

    monkeypatch.setattr(pipeline, "_call_cursor_direct", fake_call_cursor_direct)
    result = pipeline._build_streaming_result_openai(
        request_id="cancel-cursor-task",
        messages=[],
        tools=[],
        valid_tool_names=[],
        resolved_model="gpt-5.6-sol",
        max_tokens=None,
        token="tok",
        pipeline_start=0,
        base_telemetry={},
    )

    async def close_stream() -> None:
        stream = result["stream_handler"]()
        await anext(stream)
        next_chunk = asyncio.create_task(anext(stream))
        await asyncio.wait_for(cursor_started.wait(), timeout=0.2)
        next_chunk.cancel()
        try:
            await next_chunk
        except asyncio.CancelledError:
            pass
        await stream.aclose()
        await asyncio.wait_for(cursor_cancelled.wait(), timeout=0.2)

    asyncio.run(close_stream())


def test_openai_reasoning_buffer_bounds_callback_burst_before_consumer_drain(monkeypatch):
    """A synchronous upstream burst cannot grow display buffering without bound."""
    monkeypatch.setattr(pipeline, "MAX_PENDING_REASONING_EVENTS", 2)
    monkeypatch.setattr(pipeline, "MAX_PENDING_REASONING_CHARS", 7)

    async def fake_call_cursor_direct(
        messages,
        model,
        tools,
        valid_tool_names,
        auth_token,
        on_stream_delta=None,
        on_thinking_delta=None,
        compact_tools=False,
    ):
        assert on_thinking_delta is not None
        for _ in range(10):
            on_thinking_delta("xyz")
        return {
            "tool_calls": [],
            "text": "",
            "thinking": "",
            "model": model,
            "fallback_attempts": 0,
            "stats": {},
        }

    monkeypatch.setattr(pipeline, "_call_cursor_direct", fake_call_cursor_direct)
    result = pipeline._build_streaming_result_openai(
        request_id="bounded-reasoning-burst",
        messages=[],
        tools=[],
        valid_tool_names=[],
        resolved_model="gpt-5.6-sol",
        max_tokens=None,
        token="tok",
        pipeline_start=0,
        base_telemetry={},
    )

    async def collect_after_cursor_burst() -> str:
        stream = result["stream_handler"]()
        await anext(stream)
        await asyncio.sleep(0)
        return "".join([chunk async for chunk in stream])

    sse = asyncio.run(collect_after_cursor_burst())
    reasoning_events = [
        (object_.get("choices") or [{}])[0].get("delta", {}).get("reasoning_content", "")
        for object_ in _sse_data_objects(sse)
        if "reasoning_content" in (object_.get("choices") or [{}])[0].get("delta", {})
    ]

    assert len(reasoning_events) <= pipeline.MAX_PENDING_REASONING_EVENTS
    assert sum(map(len, reasoning_events)) <= pipeline.MAX_PENDING_REASONING_CHARS
    assert "data: [DONE]" in sse


def _run_all() -> int:
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in funcs:
        mp = _MonkeyPatch()
        try:
            if fn.__code__.co_argcount:
                fn(mp)
            else:
                fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
        finally:
            mp.undo()
    print(f"\n{len(funcs) - failures}/{len(funcs)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
