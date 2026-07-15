"""Regression tests for non-zero context usage reporting."""

import json
import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code.openai_sse_assembler import (
    StreamingOpenAISession,
    build_unary_openai_response,
)
from claude_code.sse_assembler import (
    StreamingAnthropicSession,
    build_unary_anthropic_response,
)
from core.protobuf_frame_parser import ProtobufFrameParser
from core.token_usage import (
    estimate_input_tokens,
    input_tokens_from_remaining_context,
)


def _parse_anthropic_events(payload: str) -> list[tuple[str, dict]]:
    events = []
    for record in payload.strip().split("\n\n"):
        event_line, data_line = record.split("\n")
        events.append((event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))))
    return events


def _parse_openai_events(payload: str) -> list[dict]:
    events = []
    for record in payload.strip().split("\n\n"):
        data = record.removeprefix("data: ")
        if data != "[DONE]":
            events.append(json.loads(data))
    return events


def test_estimate_input_tokens_counts_outbound_messages_and_tool_schemas():
    messages = [
        {"role": "system", "content": "Follow the caller instructions."},
        {"role": "user", "content": "Inspect this project and fix the bug."},
    ]
    tools = [{
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }]
    without_tools = estimate_input_tokens(messages, [])
    with_tools = estimate_input_tokens(messages, tools)
    assert without_tools > 0
    assert with_tools > without_tools


def test_parser_extracts_cursor_server_remaining_context_percentage():
    nested_usage = b"\x08\x62\x25" + struct.pack("<f", 98.25)
    message = b"\xf2\x01" + bytes([len(nested_usage)]) + nested_usage
    response = b"\x12" + bytes([len(message)]) + message
    frame = b"\x00" + struct.pack(">I", len(response)) + response
    result = ProtobufFrameParser().parse(frame)
    assert result.context_remaining_percent is not None
    assert math.isclose(result.context_remaining_percent, 98.25)


def test_server_remaining_context_converts_to_used_tokens():
    assert input_tokens_from_remaining_context(272_000, 98.25, 123) == 4_760
    assert input_tokens_from_remaining_context(None, 98.25, 123) == 123


def test_anthropic_renderers_report_supplied_input_usage():
    unary = build_unary_anthropic_response(
        message_id="msg_usage",
        model="usage-model",
        text="Done.",
        thinking="",
        tool_calls=[],
        input_tokens=321,
    )
    assert unary["usage"]["input_tokens"] == 321
    session = StreamingAnthropicSession("msg_usage", "usage-model", input_tokens=321)
    events = _parse_anthropic_events(session.emit_message_start() + session.finish())
    message_start = next(data for event, data in events if event == "message_start")
    message_delta = next(data for event, data in events if event == "message_delta")
    assert message_start["message"]["usage"]["input_tokens"] == 321
    assert message_delta["usage"]["input_tokens"] == 321


def test_openai_renderers_report_supplied_prompt_usage():
    unary = build_unary_openai_response(
        completion_id="chatcmpl_usage",
        model="usage-model",
        text="Done.",
        tool_calls=[],
        input_tokens=321,
    )
    assert unary["usage"]["prompt_tokens"] == 321
    assert unary["usage"]["total_tokens"] == 323
    session = StreamingOpenAISession("chatcmpl_usage", "usage-model", input_tokens=321)
    events = _parse_openai_events(session.emit_role_chunk() + session.finish())
    assert events[-1]["usage"]["prompt_tokens"] == 321
    assert events[-1]["usage"]["total_tokens"] == 321


def _run_all() -> int:
    functions = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    failures = 0
    for function in functions:
        try:
            function()
            print(f"PASS {function.__name__}")
        except Exception as exc:
            failures += 1
            print(f"FAIL {function.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(functions) - failures}/{len(functions)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
