"""Protocol-valid keepalives for quiet upstream Cursor streams."""

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code import pipeline


def _install_stalled_cursor(monkeypatch):
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
    return started, cancelled


def test_stalled_chat_completions_stream_emits_empty_delta(monkeypatch):
    started, cancelled = _install_stalled_cursor(monkeypatch)
    result = pipeline._build_streaming_result_openai(
        request_id="req_chat_progress",
        messages=[{"role": "user", "content": "Hello"}],
        tools=[],
        valid_tool_names=[],
        resolved_model="gpt-test",
        max_tokens=None,
        token="token",
        pipeline_start=time.monotonic(),
        base_telemetry={},
        requested_model="gpt-test",
        client_format="openai",
    )

    async def verify() -> None:
        stream = result["stream_handler"]()
        await anext(stream)
        keepalive_task = asyncio.create_task(anext(stream))
        await asyncio.wait_for(started.wait(), timeout=0.2)
        raw = await asyncio.wait_for(keepalive_task, timeout=0.2)
        event = json.loads(raw.removeprefix("data: "))
        assert event["object"] == "chat.completion.chunk"
        assert event["choices"] == [{"index": 0, "delta": {}, "finish_reason": None}]
        await stream.aclose()
        await asyncio.wait_for(cancelled.wait(), timeout=0.2)

    asyncio.run(verify())


def test_stalled_anthropic_stream_emits_ping(monkeypatch):
    started, cancelled = _install_stalled_cursor(monkeypatch)
    result = pipeline._build_streaming_result_anthropic(
        request_id="req_anthropic_progress",
        messages=[{"role": "user", "content": "Hello"}],
        tools=[],
        valid_tool_names=[],
        resolved_model="gpt-test",
        max_tokens=None,
        token="token",
        pipeline_start=time.monotonic(),
        base_telemetry={},
        requested_model="gpt-test",
        client_format="anthropic",
    )

    async def verify() -> None:
        stream = result["stream_handler"]()
        await anext(stream)
        keepalive_task = asyncio.create_task(anext(stream))
        await asyncio.wait_for(started.wait(), timeout=0.2)
        raw = await asyncio.wait_for(keepalive_task, timeout=0.2)
        assert raw == 'event: ping\ndata: {"type": "ping"}\n\n'
        await stream.aclose()
        await asyncio.wait_for(cancelled.wait(), timeout=0.2)

    asyncio.run(verify())
