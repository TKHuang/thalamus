"""OpenAI Responses API compatibility contracts."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code.normalizers import normalize_openai_response
from claude_code.openai_responses_assembler import (
    StreamingOpenAIResponsesSession,
    build_unary_openai_response,
)


def _parse_sse(payload: str) -> list[dict]:
    events = []
    for record in payload.strip().split("\n\n"):
        event_line, data_line = record.split("\n", 1)
        event = json.loads(data_line.removeprefix("data: "))
        assert event_line == f"event: {event['type']}"
        events.append(event)
    return events


def test_normalize_response_accepts_string_input_and_flat_function_tools():
    request = normalize_openai_response({
        "model": "gpt-test",
        "instructions": "Answer precisely.",
        "input": "Hello",
        "max_output_tokens": 123,
        "tools": [{
            "type": "function",
            "name": "get_weather",
            "description": "Get weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            "strict": True,
        }],
    })

    assert request.system == "Answer precisely."
    assert request.messages == [{"role": "user", "content": "Hello"}]
    assert request.tools == [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    assert request.model == "gpt-test"
    assert request.max_tokens == 123
    assert request.original_format == "openai_responses"


def test_unary_response_renders_text_tools_and_usage():
    response = build_unary_openai_response(
        response_id="resp_test",
        model="gpt-test",
        text="Weather found.",
        tool_calls=[{
            "id": "call_weather",
            "function": {
                "name": "get_weather",
                "arguments": '{"city":"Taipei"}',
            },
        }],
        input_tokens=10,
    )

    assert response["id"] == "resp_test"
    assert response["object"] == "response"
    assert response["status"] == "completed"
    message, function_call = response["output"]
    assert message["type"] == "message"
    assert message["content"] == [{
        "type": "output_text",
        "text": "Weather found.",
        "annotations": [],
        "logprobs": [],
    }]
    assert function_call == {
        "id": "fc_call_weather",
        "type": "function_call",
        "status": "completed",
        "call_id": "call_weather",
        "name": "get_weather",
        "arguments": '{"city":"Taipei"}',
    }
    assert response["usage"] == {
        "input_tokens": 10,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 4,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 14,
    }


def test_streaming_response_emits_ordered_text_and_function_call_events():
    session = StreamingOpenAIResponsesSession("resp_stream", "gpt-test", input_tokens=10)
    payload = (
        session.start()
        + session.emit_text_delta("Hello")
        + session.emit_tool_use_blocks([{
            "id": "call_weather",
            "function": {
                "name": "get_weather",
                "arguments": '{"city":"Taipei"}',
            },
        }])
        + session.finish()
    )
    events = _parse_sse(payload)

    assert [event["sequence_number"] for event in events] == list(range(len(events)))
    assert events[0]["type"] == "response.created"
    assert events[1]["type"] == "response.in_progress"
    assert events[0]["response"]["usage"] is None
    added_message = next(
        event for event in events if event["type"] == "response.output_item.added"
    )
    assert added_message["item"]["content"] == []
    assert any(event["type"] == "response.output_text.delta" and event["delta"] == "Hello" for event in events)
    assert any(event["type"] == "response.function_call_arguments.delta" for event in events)
    completed = events[-1]
    assert completed["type"] == "response.completed"
    assert completed["response"]["status"] == "completed"
    assert completed["response"]["output"][0]["content"][0]["text"] == "Hello"
    assert completed["response"]["output"][1]["call_id"] == "call_weather"
    assert completed["response"]["usage"]["input_tokens"] == 10


def test_streaming_response_emits_reasoning_summary_before_visible_text():
    session = StreamingOpenAIResponsesSession("resp_reasoning", "gpt-test")
    payload = (
        session.start()
        + session.emit_reasoning_delta("Checking the request.")
        + session.emit_text_delta("Working on it.")
        + session.finish()
    )
    events = _parse_sse(payload)
    event_types = [event["type"] for event in events]

    reasoning_delta = events[event_types.index("response.reasoning_summary_text.delta")]
    assert reasoning_delta["delta"] == "Checking the request."
    assert event_types.index("response.reasoning_summary_text.done") < event_types.index(
        "response.output_text.delta"
    )
    completed_output = events[-1]["response"]["output"]
    assert [item["type"] for item in completed_output] == ["reasoning", "message"]
    assert completed_output[0]["summary"][0]["text"] == "Checking the request."


def test_streaming_response_can_suppress_unrequested_reasoning_summary():
    session = StreamingOpenAIResponsesSession(
        "resp_hidden_reasoning",
        "gpt-test",
        emit_reasoning_summary=False,
    )
    payload = session.start() + session.emit_reasoning_delta("hidden") + session.finish()
    events = _parse_sse(payload)

    assert all("reasoning" not in event["type"] for event in events)
    assert events[-1]["response"]["output"] == []


def test_streaming_response_can_complete_preamble_before_later_output():
    session = StreamingOpenAIResponsesSession("resp_segments", "gpt-test")
    payload = (
        session.start()
        + session.emit_text_delta("Starting now.")
        + session.flush_text_item()
        + session.emit_text_delta("Finished.")
        + session.finish()
    )
    events = _parse_sse(payload)
    completed_output = events[-1]["response"]["output"]

    assert [item["type"] for item in completed_output] == ["message", "message"]
    assert completed_output[0]["content"][0]["text"] == "Starting now."
    assert completed_output[1]["content"][0]["text"] == "Finished."
    assert [
        event["output_index"]
        for event in events
        if event["type"] == "response.output_item.done"
    ] == [0, 1]


def test_streaming_response_initial_lifecycle_is_emitted_once_and_ordered():
    session = StreamingOpenAIResponsesSession("resp_progress", "gpt-test")
    payload = session.start() + session.finish()
    events = _parse_sse(payload)

    assert [event["sequence_number"] for event in events] == list(range(len(events)))
    assert [event["type"] for event in events] == [
        "response.created",
        "response.in_progress",
        "response.completed",
    ]
    assert "delta" not in events[1]
    assert events[1]["response"]["output"] == []
