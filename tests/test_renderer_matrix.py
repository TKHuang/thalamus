"""Standalone public renderer contracts for accepted canonical tool calls."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code.openai_sse_assembler import (  # noqa: E402
    StreamingOpenAISession,
    build_unary_openai_response,
)
from claude_code.sse_assembler import (  # noqa: E402
    StreamingAnthropicSession,
    build_unary_anthropic_response,
)


CANONICAL_TOOL_CALL = {
    "id": "toolu_renderer_matrix_1",
    "function": {
        "name": "write_file",
        "arguments": (
            '{"file_path":"/tmp/工具.py","content":"print(\\"hello\\")\\n",'
            '"overwrite":true}'
        ),
    },
}
VISIBLE_TEXT = "I will create the requested file."
PROTOCOL_SYNTAX = ("<|tool_", "<｜tool", "</think>", '"type":"tool_use"')


def _parse_openai_sse(payload: str) -> tuple[list[dict], bool]:
    events: list[dict] = []
    saw_done = False
    for record in payload.strip().split("\n\n"):
        data = record.removeprefix("data: ")
        if data == "[DONE]":
            saw_done = True
        else:
            events.append(json.loads(data))
    return events, saw_done


def _parse_anthropic_sse(payload: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for record in payload.strip().split("\n\n"):
        event_line, data_line = record.split("\n")
        events.append((event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))))
    return events


def _assert_no_protocol_syntax(text: str) -> None:
    for syntax in PROTOCOL_SYNTAX:
        assert syntax not in text


def test_openai_renderer_matrix_preserves_tool_call_contracts():
    unary = build_unary_openai_response(
        completion_id="chatcmpl_renderer_matrix",
        model="renderer-model",
        text=VISIBLE_TEXT,
        tool_calls=[CANONICAL_TOOL_CALL],
    )

    message = unary["choices"][0]["message"]
    unary_tool_call = message["tool_calls"][0]
    assert unary["id"] == "chatcmpl_renderer_matrix"
    assert unary["choices"][0]["index"] == 0
    assert unary["choices"][0]["finish_reason"] == "tool_calls"
    assert message["content"] == VISIBLE_TEXT
    assert unary_tool_call["id"] == CANONICAL_TOOL_CALL["id"]
    assert unary_tool_call["function"]["name"] == "write_file"
    assert unary_tool_call["function"]["arguments"] == CANONICAL_TOOL_CALL["function"]["arguments"]
    _assert_no_protocol_syntax(message["content"])

    session = StreamingOpenAISession("chatcmpl_renderer_matrix", "renderer-model")
    events, saw_done = _parse_openai_sse(
        session.emit_role_chunk()
        + session.emit_text_delta(VISIBLE_TEXT)
        + session.emit_tool_use_blocks([CANONICAL_TOOL_CALL])
        + session.finish("tool_use")
    )

    assert saw_done
    assert {event["id"] for event in events} == {"chatcmpl_renderer_matrix"}
    assert [event["choices"][0]["index"] for event in events] == [0] * len(events)
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"

    streamed_content = "".join(
        choice["delta"].get("content", "") for event in events for choice in event["choices"]
    )
    streamed_tool_deltas = [
        tool_call
        for event in events
        for choice in event["choices"]
        for tool_call in choice["delta"].get("tool_calls", [])
    ]
    assert streamed_content == VISIBLE_TEXT
    assert [tool_call["index"] for tool_call in streamed_tool_deltas] == [0, 0]
    assert streamed_tool_deltas[0]["id"] == CANONICAL_TOOL_CALL["id"]
    assert streamed_tool_deltas[0]["function"]["name"] == "write_file"
    assert "".join(
        tool_call["function"].get("arguments", "") for tool_call in streamed_tool_deltas
    ) == CANONICAL_TOOL_CALL["function"]["arguments"]
    _assert_no_protocol_syntax(streamed_content)


def test_openai_renderer_emits_reasoning_in_its_own_delta_and_counts_it_in_usage():
    """OpenAI clients consume reasoning separately; duplicating it as content corrupts answers."""
    session = StreamingOpenAISession("chatcmpl_reasoning", "renderer-model")
    events, saw_done = _parse_openai_sse(
        session.emit_role_chunk()
        + session.emit_reasoning_delta("Inspect inputs. ")
        + session.emit_text_delta("Done.")
        + session.finish()
    )

    assert saw_done
    deltas = [event["choices"][0]["delta"] for event in events]
    assert [delta["reasoning_content"] for delta in deltas if "reasoning_content" in delta] == [
        "Inspect inputs. "
    ]
    assert "".join(delta.get("content", "") for delta in deltas) == "Done."
    assert events[-1]["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 6,
        "total_tokens": 6,
    }


def test_anthropic_renderer_matrix_preserves_tool_use_contracts():
    expected_input = json.loads(CANONICAL_TOOL_CALL["function"]["arguments"])
    unary = build_unary_anthropic_response(
        message_id="msg_renderer_matrix",
        model="renderer-model",
        text=VISIBLE_TEXT,
        thinking="",
        tool_calls=[CANONICAL_TOOL_CALL],
    )

    unary_text, unary_tool_use = unary["content"]
    assert unary["id"] == "msg_renderer_matrix"
    assert unary["stop_reason"] == "tool_use"
    assert unary_text == {"type": "text", "text": VISIBLE_TEXT}
    assert unary_tool_use == {
        "type": "tool_use",
        "id": CANONICAL_TOOL_CALL["id"],
        "name": "write_file",
        "input": expected_input,
    }
    _assert_no_protocol_syntax(unary_text["text"])

    session = StreamingAnthropicSession("msg_renderer_matrix", "renderer-model")
    events = _parse_anthropic_sse(
        session.emit_message_start()
        + session.emit_text_delta(VISIBLE_TEXT)
        + session.close_open_blocks()
        + session.emit_tool_use_blocks([CANONICAL_TOOL_CALL])
        + session.finish("tool_use")
    )

    text_deltas = [
        data["delta"]["text"]
        for event, data in events
        if event == "content_block_delta" and data["delta"]["type"] == "text_delta"
    ]
    tool_start = next(
        data
        for event, data in events
        if event == "content_block_start" and data["content_block"]["type"] == "tool_use"
    )
    tool_delta = next(
        data
        for event, data in events
        if event == "content_block_delta" and data["delta"]["type"] == "input_json_delta"
    )
    tool_stop = next(
        data
        for event, data in events
        if event == "content_block_stop" and data["index"] == tool_start["index"]
    )
    message_delta = next(data for event, data in events if event == "message_delta")

    assert "".join(text_deltas) == VISIBLE_TEXT
    assert tool_start["index"] == tool_delta["index"] == tool_stop["index"] == 1
    assert tool_start["content_block"]["id"] == CANONICAL_TOOL_CALL["id"]
    assert tool_start["content_block"]["name"] == "write_file"
    assert tool_start["content_block"]["type"] == "tool_use"
    assert tool_delta["delta"]["type"] == "input_json_delta"
    assert json.loads(tool_delta["delta"]["partial_json"]) == expected_input
    assert message_delta["delta"]["stop_reason"] == "tool_use"
    _assert_no_protocol_syntax("".join(text_deltas))


def _run_all() -> int:
    functions = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    failures = 0
    for function in functions:
        try:
            function()
            print(f"PASS {function.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {function.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {function.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(functions) - failures}/{len(functions)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
