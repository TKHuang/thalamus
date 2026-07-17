"""Tool identity must reach streaming clients before large arguments finish."""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code import pipeline


TOOL_CALL = {
    "id": "call-early",
    "type": "function",
    "function": {
        "name": "write_file",
        "arguments": json.dumps(
            {"path": "large.html", "contents": "complete contents"},
            separators=(",", ":"),
        ),
    },
}


def _fake_direct_call(announced: asyncio.Event, release: asyncio.Event):
    async def call(*args, **kwargs):
        callback = pipeline._TOOL_CALL_START_CALLBACK.get()
        assert callback is not None
        callback("call-early", "write_file")
        announced.set()
        await release.wait()
        return {
            "tool_calls": [TOOL_CALL],
            "text": "",
            "thinking": "",
            "model": args[1],
            "fallback_attempts": 0,
            "stats": {},
        }

    return call


async def _collect_with_early_event(result: dict, announced: asyncio.Event, release: asyncio.Event):
    stream = result["stream_handler"]()
    initial = await anext(stream)
    next_event = asyncio.create_task(anext(stream))
    await asyncio.wait_for(announced.wait(), timeout=0.5)
    early = await asyncio.wait_for(next_event, timeout=0.5)
    assert release.is_set() is False
    release.set()
    tail = "".join([chunk async for chunk in stream])
    return initial, early, tail


def test_anthropic_stream_exposes_tool_start_before_arguments_finish(monkeypatch):
    announced = asyncio.Event()
    release = asyncio.Event()
    monkeypatch.setattr(
        pipeline,
        "_call_cursor_direct",
        _fake_direct_call(announced, release),
    )
    result = pipeline._build_streaming_result_anthropic(
        request_id="early-anthropic",
        messages=[],
        tools=[],
        valid_tool_names=[],
        resolved_model="future-model",
        max_tokens=None,
        token="token",
        pipeline_start=0,
        base_telemetry={},
    )

    _initial, early, tail = asyncio.run(
        _collect_with_early_event(result, announced, release)
    )
    assert '"type": "tool_use"' in early
    assert '"id": "call-early"' in early
    assert '"name": "write_file"' in early
    assert '"input": {}' in early
    assert '"type": "input_json_delta"' in tail
    assert "large.html" in tail
    assert (early + tail).count('"id": "call-early"') == 1


def test_openai_chat_stream_exposes_tool_start_before_arguments_finish(monkeypatch):
    announced = asyncio.Event()
    release = asyncio.Event()
    monkeypatch.setattr(
        pipeline,
        "_call_cursor_direct",
        _fake_direct_call(announced, release),
    )
    result = pipeline._build_streaming_result_openai(
        request_id="early-openai-chat",
        messages=[],
        tools=[],
        valid_tool_names=[],
        resolved_model="future-model",
        max_tokens=None,
        token="token",
        pipeline_start=0,
        base_telemetry={},
    )

    _initial, early, tail = asyncio.run(
        _collect_with_early_event(result, announced, release)
    )
    assert '"id": "call-early"' in early
    assert '"name": "write_file"' in early
    assert '"arguments": ""' in early
    assert "large.html" in tail
    assert (early + tail).count('"id": "call-early"') == 1


def test_openai_responses_stream_exposes_in_progress_function_item(monkeypatch):
    announced = asyncio.Event()
    release = asyncio.Event()
    monkeypatch.setattr(
        pipeline,
        "_call_cursor_direct",
        _fake_direct_call(announced, release),
    )
    result = pipeline._build_streaming_result_openai(
        request_id="early-openai-responses",
        messages=[],
        tools=[],
        valid_tool_names=[],
        resolved_model="future-model",
        max_tokens=None,
        token="token",
        pipeline_start=0,
        base_telemetry={},
        client_format="openai_responses",
    )

    _initial, early, tail = asyncio.run(
        _collect_with_early_event(result, announced, release)
    )
    assert "response.output_item.added" in early
    assert '"type": "function_call"' in early
    assert '"status": "in_progress"' in early
    assert '"call_id": "call-early"' in early
    assert "response.function_call_arguments.delta" in tail
    assert "response.function_call_arguments.done" in tail
    assert "large.html" in tail
