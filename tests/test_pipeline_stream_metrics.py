"""Regression coverage for transport and semantic stream latency metrics."""

from __future__ import annotations

import asyncio
import os
import struct
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code import pipeline
from proto import cursor_api_pb2 as pb


def _frame(response: pb.StreamUnifiedChatWithToolsResponse) -> bytes:
    payload = response.SerializeToString()
    return b"\x00" + struct.pack(">I", len(payload)) + payload


def _empty_frame() -> bytes:
    return _frame(pb.StreamUnifiedChatWithToolsResponse())


def _thinking_frame(value: str) -> bytes:
    response = pb.StreamUnifiedChatWithToolsResponse()
    response.message.thinking.content = value
    return _frame(response)


def _text_frame(value: str) -> bytes:
    response = pb.StreamUnifiedChatWithToolsResponse()
    response.message.content = value
    return _frame(response)


def _tool_frame() -> bytes:
    response = pb.StreamUnifiedChatWithToolsResponse()
    call = response.clientSideToolV2Call
    call.tool = 49
    call.toolCallId = "call_metric"
    call.name = "mcp_call_tool"
    call.isLastMessage = True
    call.callMcpToolParams.toolName = "write_file"
    call.callMcpToolParams.toolArgs.update(
        {"path": "metric.html", "content": "ok"}
    )
    return _frame(response)


def test_consume_stream_distinguishes_first_chunk_and_semantic_events(monkeypatch):
    ticks = iter((10.0, 10.1, 10.2, 10.3, 10.4, 10.5))
    monkeypatch.setattr(
        pipeline,
        "time",
        SimpleNamespace(monotonic=lambda: next(ticks)),
    )

    async def stream():
        yield _empty_frame()
        yield _thinking_frame("reasoning")
        yield _text_frame("visible text")
        yield _tool_frame()

    consumed = asyncio.run(pipeline.consume_stream(stream()))
    metrics = consumed["metrics"]

    assert metrics["first_chunk_latency_ms"] == pytest.approx(100)
    assert metrics["first_reasoning_latency_ms"] == pytest.approx(200)
    assert metrics["first_text_latency_ms"] == pytest.approx(300)
    assert metrics["first_tool_identity_latency_ms"] == pytest.approx(400)
    assert metrics["first_semantic_latency_ms"] == pytest.approx(200)


def test_stream_metric_log_keeps_legacy_key_but_never_calls_it_a_token():
    rendered = pipeline._format_stream_metrics(
        {
            "chunk_count": 7,
            "first_chunk_latency_ms": 12.4,
            "first_semantic_latency_ms": 45.6,
            "first_text_latency_ms": -1,
            "first_reasoning_latency_ms": 45.6,
            "first_tool_identity_latency_ms": 90.2,
        }
    )

    assert "chunks=7" in rendered
    assert "first_chunk_ms=12" in rendered
    assert "first_semantic_ms=46" in rendered
    assert "first_reasoning_ms=46" in rendered
    assert "first_tool_identity_ms=90" in rendered
    assert "first_token" not in rendered


def test_agent_callbacks_measure_semantic_latency_without_core_metric_changes(
    monkeypatch,
):
    captured_metrics: list[dict] = []
    captured_request_ids: list[str | None] = []

    async def fake_agent(
        messages,
        model,
        tools,
        auth_token,
        *,
        on_text_delta=None,
        on_thinking_delta=None,
        on_tool_call_start=None,
        client_request_id=None,
    ):
        del messages, model, tools, auth_token
        captured_request_ids.append(client_request_id)
        on_thinking_delta("plan")
        on_text_delta("working")
        on_tool_call_start("call_1", "write_file")
        return {
            "text": "working",
            "thinking": "plan",
            "composer_tool_calls": [],
            "native_tool_calls": [],
            "interrupted_tool_state": "",
            "errors": [],
            "context_remaining_percent": None,
            "had_content": True,
            "has_fatal_error": False,
            "metrics": {"chunk_count": 3, "first_chunk_latency_ms": 1},
        }

    def capture_response(*args, **kwargs):
        del args
        captured_metrics.append(kwargs["extra"]["stream_metrics"])
        return ""

    # The clock covers pipeline/attempt bookkeeping as well as the three
    # semantic callbacks; exact values are intentionally not asserted here.
    counter = iter(value / 100 for value in range(1000, 1100))
    monkeypatch.setattr(
        pipeline,
        "time",
        SimpleNamespace(monotonic=lambda: next(counter)),
    )
    monkeypatch.setattr(pipeline, "call_cursor_agent", fake_agent)
    monkeypatch.setattr(pipeline, "log_llm_request", lambda *args, **kwargs: "")
    monkeypatch.setattr(pipeline, "log_llm_response", capture_response)
    monkeypatch.setattr(pipeline, "log_llm_api_call", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "emit_attempt_trace", lambda trace: None)
    monkeypatch.setattr(pipeline, "emit_terminal_trace", lambda trace: None)

    result = asyncio.run(
        pipeline._call_cursor_direct(
            messages=[{"role": "user", "content": "create a file"}],
            model="standard-model",
            tools=[
                {
                    "name": "write_file",
                    "input_schema": {"type": "object"},
                }
            ],
            valid_tool_names=["write_file"],
            auth_token="token",
        )
    )

    assert result["text"] == "working"
    assert len(captured_request_ids) == 1
    assert captured_request_ids[0] is not None
    assert captured_request_ids[0].startswith("cc_")
    assert captured_metrics
    metrics = captured_metrics[-1]
    assert metrics["first_chunk_latency_ms"] == 1
    assert metrics["first_reasoning_latency_ms"] >= 0
    assert metrics["first_text_latency_ms"] >= 0
    assert metrics["first_tool_identity_latency_ms"] >= 0
    assert metrics["first_semantic_latency_ms"] == min(
        metrics["first_reasoning_latency_ms"],
        metrics["first_text_latency_ms"],
        metrics["first_tool_identity_latency_ms"],
    )
