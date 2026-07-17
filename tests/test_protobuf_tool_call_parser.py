import asyncio
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code.pipeline import consume_stream
from core.protobuf_frame_parser import ProtobufFrameParser
from core.protobuf_tool_call_parser import extract_native_tool_calls
from proto import cursor_api_pb2 as pb


def _frame(response: pb.StreamUnifiedChatWithToolsResponse) -> bytes:
    raw = response.SerializeToString()
    return b"\x00" + struct.pack(">I", len(raw)) + raw


def _direct_meta_call(
    *,
    enum: int = 49,
    server: str = "hermes",
) -> pb.StreamUnifiedChatWithToolsResponse:
    response = pb.StreamUnifiedChatWithToolsResponse()
    call = response.clientSideToolV2Call
    call.tool = enum
    call.toolCallId = "call_123"
    call.name = "mcp_call_tool"
    call.isLastMessage = True
    params = call.callMcpToolParams
    params.server = server
    params.toolName = "write_file"
    params.toolArgs.update({"path": "/tmp/a", "content": "ok"})
    return response


def test_extracts_direct_call_mcp_tool_and_unwraps_client_name():
    calls = extract_native_tool_calls(_direct_meta_call().SerializeToString())

    assert len(calls) == 1
    assert calls[0].enum == 49
    assert calls[0].call_id == "call_123"
    assert calls[0].name == "write_file"
    assert calls[0].arguments == {"path": "/tmp/a", "content": "ok"}
    assert json.loads(calls[0].raw_arguments) == calls[0].arguments


def test_extracts_nested_streamed_back_tool_call_v2():
    response = pb.StreamUnifiedChatWithToolsResponse()
    call = response.message.toolCallV2
    call.tool = 49
    call.toolCallId = "call_v2"
    call.name = "mcp_call_tool"
    params = call.callMcpToolParams
    params.server = "hermes"
    params.toolName = "write_file"
    params.toolArgs.update({"path": "/tmp/v2"})

    calls = extract_native_tool_calls(response.SerializeToString())

    assert [(call.call_id, call.name) for call in calls] == [
        ("call_v2", "write_file")
    ]
    assert calls[0].arguments == {"path": "/tmp/v2"}


def test_extracts_complete_streamed_raw_meta_arguments():
    response = pb.StreamUnifiedChatWithToolsResponse()
    call = response.message.toolCall
    call.tool = 49
    call.toolCallId = "call_stream"
    call.name = "mcp_call_tool"
    call.rawArgs = json.dumps(
        {
            "server": "thalamus-client",
            "tool_name": "write_file",
            "tool_args": {"path": "/tmp/stream"},
        }
    )

    calls = extract_native_tool_calls(response.SerializeToString())

    assert [(call.call_id, call.name) for call in calls] == [
        ("call_stream", "write_file")
    ]
    assert calls[0].arguments == {"path": "/tmp/stream"}
    assert calls[0].is_streaming is True


def test_ignores_partial_announcements_without_arguments():
    response = pb.StreamUnifiedChatWithToolsResponse()
    partial = response.message.partialToolCall
    partial.tool = 49
    partial.toolCallId = "call_partial"
    partial.name = "mcp_call_tool"

    assert extract_native_tool_calls(response.SerializeToString()) == []


def test_server_label_is_metadata_but_unrelated_cursor_tools_are_rejected():
    calls = extract_native_tool_calls(
        _direct_meta_call(server="runtime-normalized-server").SerializeToString()
    )
    assert [call.name for call in calls] == ["write_file"]
    assert extract_native_tool_calls(
        _direct_meta_call(enum=38).SerializeToString()
    ) == []


def test_accepts_legacy_direct_mcp_shape():
    response = pb.StreamUnifiedChatWithToolsResponse()
    call = response.clientSideToolV2Call
    call.tool = 19
    call.toolCallId = "call_legacy"
    call.name = "write_file"
    call.rawArgs = '{"path":"/tmp/legacy"}'

    calls = extract_native_tool_calls(response.SerializeToString())

    assert [(call.enum, call.name) for call in calls] == [(19, "write_file")]
    assert calls[0].arguments == {"path": "/tmp/legacy"}


def test_frame_parser_handles_split_native_tool_frame():
    frame = _frame(_direct_meta_call())
    parser = ProtobufFrameParser()

    first = parser.parse(frame[:7])
    second = parser.parse(frame[7:])

    assert first.native_tool_calls == []
    assert [call.name for call in second.native_tool_calls] == ["write_file"]


def test_stream_consumer_stops_after_complete_native_call():
    async def exercise():
        async def stream():
            yield _frame(_direct_meta_call())
            raise AssertionError("consumer requested data after a complete tool call")

        return await consume_stream(stream())

    consumed = asyncio.run(exercise())

    assert [call.name for call in consumed["native_tool_calls"]] == ["write_file"]
    assert consumed["had_content"] is True
