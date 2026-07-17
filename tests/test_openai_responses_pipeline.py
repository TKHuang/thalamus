"""End-to-end pipeline contracts for the OpenAI Responses API."""

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code import pipeline
from core.unified_request import UnifiedRequest


def _parse_sse(payload: str) -> list[dict]:
    events = []
    for record in payload.strip().split("\n\n"):
        event_line, data_line = record.split("\n", 1)
        event = json.loads(data_line.removeprefix("data: "))
        assert event_line == f"event: {event['type']}"
        events.append(event)
    return events


def _request(stream: bool) -> UnifiedRequest:
    return UnifiedRequest(
        messages=[{"role": "user", "content": "Hello"}],
        system="",
        tools=[],
        model="gpt-test",
        stream=stream,
        original_format="openai_responses",
        original_model="gpt-test",
    )


def test_unary_pipeline_returns_responses_api_object(monkeypatch):
    async def fake_call_cursor_direct(*args, **kwargs):
        return {"text": "Hello back", "tool_calls": [], "model": "gpt-test"}

    monkeypatch.setattr(pipeline, "_call_cursor_direct", fake_call_cursor_direct)
    result = asyncio.run(pipeline._build_unary_result(
        req=_request(False),
        request_id="req_response_unary",
        messages=[{"role": "user", "content": "Hello"}],
        tools=[],
        valid_tool_names=[],
        resolved_model="gpt-test",
        max_tokens=None,
        token="token",
        original_format="openai_responses",
        pipeline_start=time.monotonic(),
        base_telemetry={},
        requested_model="gpt-test",
    ))

    assert result["ok"]
    assert result["body"]["object"] == "response"
    assert result["body"]["output"][0]["content"][0]["text"] == "Hello back"


def test_streaming_pipeline_returns_responses_api_events(monkeypatch):
    async def fake_call_cursor_direct(*args, **kwargs):
        kwargs["on_thinking_delta"]("Checking the request.")
        kwargs["on_stream_delta"]("Hello back")
        return {
            "thinking": "Checking the request.",
            "text": "Hello back",
            "tool_calls": [],
            "model": "gpt-test",
        }

    monkeypatch.setattr(pipeline, "_call_cursor_direct", fake_call_cursor_direct)
    result = pipeline._build_streaming_result_openai(
        request_id="req_response_stream",
        messages=[{"role": "user", "content": "Hello"}],
        tools=[],
        valid_tool_names=[],
        resolved_model="gpt-test",
        max_tokens=None,
        token="token",
        pipeline_start=time.monotonic(),
        base_telemetry={},
        requested_model="gpt-test",
        client_format="openai_responses",
        thinking={"effort": "low", "summary": "auto"},
    )

    async def collect() -> str:
        return "".join([chunk async for chunk in result["stream_handler"]()])

    payload = asyncio.run(collect())
    events = _parse_sse(payload)
    assert events[0]["type"] == "response.created"
    assert events[-1]["type"] == "response.completed"
    assert any(
        event["type"] == "response.reasoning_summary_text.delta"
        and event["delta"] == "Checking the request."
        for event in events
    )
    assert "".join(
        event.get("delta", "")
        for event in events
        if event["type"] == "response.output_text.delta"
    ) == "Hello back"


def test_stalled_responses_stream_emits_progress_event_and_cancels_upstream(monkeypatch):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def fake_call_cursor_direct(*args, **kwargs):
        del args, kwargs
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(pipeline, "_call_cursor_direct", fake_call_cursor_direct)
    monkeypatch.setattr(pipeline, "SSE_KEEPALIVE_INTERVAL_SECONDS", 0.01)
    result = pipeline._build_streaming_result_openai(
        request_id="req_response_progress",
        messages=[{"role": "user", "content": "Hello"}],
        tools=[],
        valid_tool_names=[],
        resolved_model="gpt-test",
        max_tokens=None,
        token="token",
        pipeline_start=time.monotonic(),
        base_telemetry={},
        requested_model="gpt-test",
        client_format="openai_responses",
    )

    async def verify() -> None:
        stream = result["stream_handler"]()
        initial = await anext(stream)
        assert "response.created" in initial
        keepalive_task = asyncio.create_task(anext(stream))
        await asyncio.wait_for(started.wait(), timeout=0.2)
        keepalive = await asyncio.wait_for(keepalive_task, timeout=0.2)
        event = _parse_sse(keepalive)[0]
        assert event["type"] == "response.in_progress"
        assert event["response"]["status"] == "in_progress"
        await stream.aclose()
        await asyncio.wait_for(cancelled.wait(), timeout=0.2)

    asyncio.run(verify())


def test_quiescent_preamble_item_completes_before_cursor_tool_call(monkeypatch):
    preamble_sent = asyncio.Event()
    allow_cursor_return = asyncio.Event()
    cursor_returned = asyncio.Event()

    async def fake_call_cursor_direct(*args, **kwargs):
        kwargs["on_stream_delta"]("Starting the requested change.")
        preamble_sent.set()
        await allow_cursor_return.wait()
        cursor_returned.set()
        return {
            "text": "Starting the requested change.",
            "tool_calls": [{
                "id": "call_write",
                "function": {
                    "name": "write_file",
                    "arguments": '{"path":"demo.txt","content":"ok"}',
                },
            }],
            "model": "gpt-test",
        }

    monkeypatch.setattr(pipeline, "_call_cursor_direct", fake_call_cursor_direct)
    monkeypatch.setattr(pipeline, "RESPONSES_TEXT_ITEM_IDLE_FLUSH_SECONDS", 0.01)
    result = pipeline._build_streaming_result_openai(
        request_id="req_preamble_flush",
        messages=[{"role": "user", "content": "Write demo.txt"}],
        tools=[{"name": "write_file", "input_schema": {"type": "object"}}],
        valid_tool_names=["write_file"],
        resolved_model="gpt-test",
        max_tokens=None,
        token="token",
        pipeline_start=time.monotonic(),
        base_telemetry={},
        requested_model="gpt-test",
        client_format="openai_responses",
    )

    async def verify() -> None:
        stream = result["stream_handler"]()
        chunks = [await anext(stream)]
        next_chunk = asyncio.create_task(anext(stream))
        await asyncio.wait_for(preamble_sent.wait(), timeout=0.2)
        chunks.append(await asyncio.wait_for(next_chunk, timeout=0.2))

        while "response.output_item.done" not in "".join(chunks):
            chunks.append(await asyncio.wait_for(anext(stream), timeout=0.2))

        assert not cursor_returned.is_set()
        preamble_events = _parse_sse("".join(chunks))
        done_items = [
            event for event in preamble_events
            if event["type"] == "response.output_item.done"
        ]
        assert done_items[-1]["item"]["type"] == "message"
        assert done_items[-1]["item"]["content"][0]["text"] == (
            "Starting the requested change."
        )

        allow_cursor_return.set()
        remaining = [chunk async for chunk in stream]
        all_events = _parse_sse("".join(chunks + remaining))
        assert any(
            event["type"] == "response.function_call_arguments.done"
            and event["name"] == "write_file"
            for event in all_events
        )

    asyncio.run(verify())


def test_reasoning_then_quiescent_preamble_completes_before_cursor_tool_call(
    monkeypatch,
):
    preamble_sent = asyncio.Event()
    allow_cursor_return = asyncio.Event()
    cursor_returned = asyncio.Event()

    async def fake_call_cursor_direct(*args, **kwargs):
        kwargs["on_thinking_delta"]("Planning the requested change.")
        kwargs["on_stream_delta"]("Starting the requested change.")
        preamble_sent.set()
        await allow_cursor_return.wait()
        cursor_returned.set()
        return {
            "thinking": "Planning the requested change.",
            "text": "Starting the requested change.",
            "tool_calls": [{
                "id": "call_write",
                "function": {
                    "name": "write_file",
                    "arguments": '{"path":"demo.txt","content":"ok"}',
                },
            }],
            "model": "gpt-test",
        }

    monkeypatch.setattr(pipeline, "_call_cursor_direct", fake_call_cursor_direct)
    monkeypatch.setattr(pipeline, "RESPONSES_TEXT_ITEM_IDLE_FLUSH_SECONDS", 0.01)
    result = pipeline._build_streaming_result_openai(
        request_id="req_reasoning_preamble_flush",
        messages=[{"role": "user", "content": "Write demo.txt"}],
        tools=[{"name": "write_file", "input_schema": {"type": "object"}}],
        valid_tool_names=["write_file"],
        resolved_model="gpt-test",
        max_tokens=None,
        token="token",
        pipeline_start=time.monotonic(),
        base_telemetry={},
        requested_model="gpt-test",
        client_format="openai_responses",
        thinking={"effort": "low", "summary": "auto"},
    )

    async def verify() -> None:
        stream = result["stream_handler"]()
        chunks = [await anext(stream)]
        next_chunk = asyncio.create_task(anext(stream))
        await asyncio.wait_for(preamble_sent.wait(), timeout=0.2)
        chunks.append(await asyncio.wait_for(next_chunk, timeout=0.2))

        while True:
            chunks.append(await asyncio.wait_for(anext(stream), timeout=0.2))
            events = _parse_sse("".join(chunks))
            if any(
                event["type"] == "response.output_item.done"
                and event["item"]["type"] == "message"
                for event in events
            ):
                break

        assert not cursor_returned.is_set()
        event_types = [event["type"] for event in events]
        reasoning_delta_index = event_types.index(
            "response.reasoning_summary_text.delta"
        )
        reasoning_done_index = event_types.index("response.reasoning_summary_text.done")
        text_delta_index = event_types.index("response.output_text.delta")
        message_done_index = next(
            index
            for index, event in enumerate(events)
            if event["type"] == "response.output_item.done"
            and event["item"]["type"] == "message"
        )
        assert reasoning_delta_index < reasoning_done_index < text_delta_index
        assert text_delta_index < message_done_index

        allow_cursor_return.set()
        remaining = [chunk async for chunk in stream]
        all_events = _parse_sse("".join(chunks + remaining))
        assert any(
            event["type"] == "response.function_call_arguments.done"
            and event["name"] == "write_file"
            for event in all_events
        )

    asyncio.run(verify())
