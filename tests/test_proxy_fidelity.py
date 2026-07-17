"""Regression tests for request fidelity and per-request tool isolation."""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code import pipeline
from claude_code.normalizers import (
    normalize_anthropic,
    normalize_openai,
    normalize_openai_response,
)
from claude_code.tool_choice import ToolChoiceError, resolve_tool_choice
from core.unified_request import UnifiedRequest
from core.protobuf_tool_call_parser import NativeToolCall


def _tool(name: str) -> dict:
    return {
        "name": name,
        "input_schema": {
            "type": "object",
            "properties": {"session": {"type": "string"}},
            "required": ["session"],
        },
    }


def _consumed(text: str) -> dict:
    return {
        "text": text,
        "thinking": "",
        "composer_tool_calls": [],
        "interrupted_tool_state": "",
        "has_fatal_error": False,
        "errors": [],
        "had_content": bool(text),
        "metrics": {"chunk_count": 2, "first_chunk_latency_ms": 0},
    }


def test_openai_developer_context_is_preserved_as_system_instruction():
    request = normalize_openai({
        "model": "gpt-5.6-sol-xhigh",
        "messages": [
            {
                "role": "developer",
                "content": [
                    {"type": "text", "text": "Working directory: /workspace/project"},
                ],
            },
            {"role": "user", "content": "Create output.html"},
        ],
    })

    assert request.system == "Working directory: /workspace/project"
    assert request.messages == [{"role": "user", "content": "Create output.html"}]


def test_tool_choice_normalizes_supported_client_shapes():
    names = ["read_file", "write_file"]

    assert resolve_tool_choice(None, names).mode == "auto"
    assert "future promise" in resolve_tool_choice(None, names).instruction()
    assert resolve_tool_choice("none", names).mode == "none"
    assert resolve_tool_choice("required", names).mode == "required"
    assert resolve_tool_choice({"type": "any"}, names).mode == "required"
    assert resolve_tool_choice(
        {"type": "tool", "name": "read_file"}, names
    ).name == "read_file"
    try:
        resolve_tool_choice(
            {"type": "function", "function": {"name": "WRITE_FILE"}}, names
        )
    except ToolChoiceError as exc:
        assert "unadvertised tool" in str(exc)
    else:
        raise AssertionError("tool_choice names must be exact")
    assert resolve_tool_choice(
        {"type": "function", "name": "read_file"}, names
    ).name == "read_file"


def test_tool_choice_rejects_unadvertised_specific_tool():
    try:
        resolve_tool_choice(
            {"type": "function", "function": {"name": "terminal"}},
            ["read_file"],
        )
    except ToolChoiceError as exc:
        assert "unadvertised tool" in str(exc)
    else:
        raise AssertionError("unadvertised tool_choice should fail")


def test_all_api_surfaces_preserve_client_tool_schema_verbatim():
    schema = {
        "type": "object",
        "properties": {"url": {"type": "string", "format": "uri"}},
        "required": ["url"],
    }
    requests = [
        normalize_anthropic(
            {
                "model": "gpt-test",
                "messages": [],
                "tools": [{"name": "fetch_exact", "input_schema": schema}],
            }
        ),
        normalize_openai(
            {
                "model": "gpt-test",
                "messages": [],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "fetch_exact", "parameters": schema},
                    }
                ],
            }
        ),
        normalize_openai_response(
            {
                "model": "gpt-test",
                "input": "",
                "tools": [
                    {"type": "function", "name": "fetch_exact", "parameters": schema}
                ],
            }
        ),
    ]

    for request in requests:
        copied = request.tools[0]["function"]["parameters"]
        assert copied == schema
        assert copied is not schema


def test_run_pipeline_rejects_invalid_tool_choice_before_upstream(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "get_cursor_access_token",
        lambda: (_ for _ in ()).throw(AssertionError("must not fetch token")),
    )
    request = UnifiedRequest(
        messages=[{"role": "user", "content": "hello"}],
        system="",
        tools=[_tool("read_file")],
        model="standard-model",
        stream=False,
        tool_choice={"type": "function", "function": {"name": "terminal"}},
    )

    result = asyncio.run(pipeline.run_pipeline(request, "invalid-choice"))

    assert result["status"] == 400
    assert "unadvertised tool" in result["body"]["error"]["message"]


def test_effective_tool_inventory_reaches_wire_builder_after_tool_choice(monkeypatch):
    class DummyStream:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    first = _tool("first_tool")
    second = _tool("second_tool")
    captured: list[list[dict]] = []

    def fake_build_cursor_stream_params(auth_token, messages, model, tools=None):
        captured.append(list(tools or []))
        return "/chat", {}, b""

    async def fake_consume_stream(
        stream,
        composer=False,
        on_text_delta=None,
        on_thinking_delta=None,
    ):
        del stream, composer, on_text_delta, on_thinking_delta
        return _consumed("done")

    async def fake_agent(messages, model, tools, auth_token, **kwargs):
        del messages, model, auth_token, kwargs
        captured.append(list(tools))
        return _consumed("done")

    monkeypatch.setattr(pipeline, "build_cursor_stream_params", fake_build_cursor_stream_params)
    monkeypatch.setattr(pipeline, "open_streaming_h2_request", lambda *args: DummyStream())
    monkeypatch.setattr(pipeline, "consume_stream", fake_consume_stream)
    monkeypatch.setattr(pipeline, "call_cursor_agent", fake_agent)
    monkeypatch.setattr(pipeline, "log_llm_request", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_response", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_api_call", lambda *args, **kwargs: None)

    async def run_calls() -> None:
        await pipeline._call_cursor_direct(
            messages=[{"role": "user", "content": "use the second tool"}],
            model="standard-model",
            tools=[first, second],
            valid_tool_names=["first_tool", "second_tool"],
            auth_token="token",
            tool_choice={"type": "tool", "name": "second_tool"},
        )
        await pipeline._call_cursor_direct(
            messages=[{"role": "user", "content": "plain text only"}],
            model="standard-model",
            tools=[first, second],
            valid_tool_names=["first_tool", "second_tool"],
            auth_token="token",
            tool_choice="none",
        )

    asyncio.run(run_calls())

    assert captured == [[second], []]


def test_parallel_requests_isolate_decoder_callbacks_history_and_trace(monkeypatch):
    class DummyStream:
        def __init__(self, session: str) -> None:
            self.session = session

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    request_histories: dict[str, list[dict]] = {}
    request_tool_names: dict[str, list[str]] = {}
    traces = []

    def fake_build_cursor_stream_params(auth_token, messages, model, tools=None):
        user_text = next(
            message["content"]
            for message in reversed(messages)
            if message.get("role") == "user"
        )
        session = user_text.rsplit(" ", 1)[-1]
        request_histories[session] = json.loads(json.dumps(messages))
        request_tool_names[session] = [
            (tool.get("function") or tool).get("name", "")
            for tool in tools or []
        ]
        return "/chat", {}, session.encode()

    async def fake_agent(
        messages,
        model,
        tools,
        auth_token,
        on_text_delta=None,
        on_thinking_delta=None,
    ):
        del model, auth_token, on_thinking_delta
        user_text = next(
            message["content"]
            for message in reversed(messages)
            if message.get("role") == "user"
        )
        session = user_text.rsplit(" ", 1)[-1]
        request_histories[session] = json.loads(json.dumps(messages))
        request_tool_names[session] = [
            (tool.get("function") or tool).get("name", "") for tool in tools
        ]
        if on_text_delta:
            on_text_delta(f"visible-{session}\n")
        consumed = _consumed(f"visible-{session}\n")
        arguments = {"session": session}
        consumed["native_tool_calls"] = [
            NativeToolCall(
                enum=49,
                call_id=f"call_{session}",
                name=f"tool_{session}",
                raw_arguments=json.dumps(arguments),
                arguments=arguments,
            )
        ]
        consumed["had_content"] = True
        return consumed

    def fake_open_streaming_h2_request(path, headers, body):
        return DummyStream(body.decode())

    async def fake_consume_stream(
        stream,
        composer=False,
        on_text_delta=None,
        on_thinking_delta=None,
    ):
        del composer, on_thinking_delta
        session = stream.session
        raw = (
            f"visible-{session}\n"
            '{"type":"tool_use","id":"call_'
            f'{session}","name":"tool_{session}","input":{{"session":"{session}"}}}}'
        )
        split = len(f"visible-{session}\n") + 1
        if on_text_delta:
            on_text_delta(raw[:split])
        await asyncio.sleep(0)
        if on_text_delta:
            on_text_delta(raw[split:])
        return _consumed(raw)

    monkeypatch.setattr(pipeline, "build_cursor_stream_params", fake_build_cursor_stream_params)
    monkeypatch.setattr(pipeline, "open_streaming_h2_request", fake_open_streaming_h2_request)
    monkeypatch.setattr(pipeline, "consume_stream", fake_consume_stream)
    monkeypatch.setattr(pipeline, "call_cursor_agent", fake_agent)
    monkeypatch.setattr(pipeline, "log_llm_request", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_response", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_api_call", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "emit_attempt_trace", traces.append)
    monkeypatch.setattr(pipeline, "emit_terminal_trace", traces.append)

    async def run_one(session: str):
        streamed: list[str] = []
        result = await pipeline._call_cursor_direct(
            messages=[{"role": "user", "content": f"run session {session}"}],
            model="standard-model",
            tools=[_tool(f"tool_{session}")],
            valid_tool_names=[f"tool_{session}"],
            auth_token="token",
            on_stream_delta=streamed.append,
            compact_tools=True,
        )
        return result, streamed

    async def run_both():
        return await asyncio.gather(run_one("A"), run_one("B"))

    first, second = asyncio.run(run_both())

    for session, (result, streamed) in zip(("A", "B"), (first, second)):
        call = result["tool_calls"][0]
        assert call["function"]["name"] == f"tool_{session}"
        assert json.loads(call["function"]["arguments"]) == {"session": session}
        assert "".join(streamed) == f"visible-{session}\n"
        assert "tool_use" not in "".join(streamed)

        history = json.dumps(request_histories[session], ensure_ascii=False)
        assert f"session {session}" in history
        assert request_tool_names[session] == [f"tool_{session}"]
        other = "B" if session == "A" else "A"
        assert f"tool_{other}" not in history
        assert f"tool_{other}" not in request_tool_names[session]
        assert f"session {other}" not in history

    request_ids = {trace.request_id for trace in traces}
    assert len(request_ids) == 2
    assert all(sum(trace.request_id == request_id for trace in traces) == 2 for request_id in request_ids)


def test_short_plain_text_streams_before_upstream_call_finishes(monkeypatch):
    callback_sent = asyncio.Event()
    allow_return = asyncio.Event()

    async def fake_call_cursor_direct(*args, on_stream_delta=None, **kwargs):
        del args, kwargs
        assert on_stream_delta is not None
        on_stream_delta("short")
        callback_sent.set()
        await allow_return.wait()
        return {
            "text": "short",
            "thinking": "",
            "model": "standard-model",
            "fallback_attempts": 0,
            "stats": {},
        }

    monkeypatch.setattr(pipeline, "_call_cursor_direct", fake_call_cursor_direct)
    result = pipeline._build_streaming_result_openai(
        request_id="immediate-text",
        messages=[{"role": "user", "content": "hello"}],
        tools=[],
        valid_tool_names=[],
        resolved_model="standard-model",
        max_tokens=None,
        token="token",
        pipeline_start=0,
        base_telemetry={},
    )

    async def assert_incremental_text() -> None:
        stream = result["stream_handler"]()
        await anext(stream)
        next_event = asyncio.create_task(anext(stream))
        await asyncio.wait_for(callback_sent.wait(), timeout=0.2)
        event = await asyncio.wait_for(next_event, timeout=0.2)
        payload = json.loads(event.removeprefix("data: ").split("\n", 1)[0])
        assert payload["choices"][0]["delta"]["content"] == "short"
        allow_return.set()
        async for _chunk in stream:
            pass

    asyncio.run(assert_incremental_text())
