"""Decode Cursor-native request-scoped MCP calls from typed protobuf fields."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from google.protobuf.json_format import MessageToDict

from proto import cursor_api_pb2 as pb

MCP_TOOL_ENUMS = frozenset({19, 49})
CALL_MCP_TOOL_ENUM = 49
MCP_META_TOOL_NAMES = frozenset({"call_mcp_tool", "mcp_call_tool"})


@dataclass(frozen=True)
class NativeToolCall:
    enum: int
    call_id: str
    name: str
    raw_arguments: str
    arguments: Any
    is_streaming: bool = False
    is_last: bool = True


def _json_object(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _compact_json(value: dict[str, Any]) -> str | None:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return None


def _translated_meta_call(
    *,
    enum: int,
    call_id: str,
    outer_name: str,
    raw_args: str,
    params: pb.CallMcpToolParams | None,
    is_streaming: bool,
) -> NativeToolCall | None:
    tool_name = ""
    tool_args: dict[str, Any] | None = None

    if params is not None:
        tool_name = params.toolName
        converted = MessageToDict(
            params.toolArgs,
            preserving_proto_field_name=True,
        )
        tool_args = converted if isinstance(converted, dict) else None

    if not tool_name:
        meta_args = _json_object(raw_args)
        if meta_args is not None:
            tool_name = meta_args.get("tool_name", meta_args.get("toolName", ""))
            candidate_args = meta_args.get(
                "tool_args",
                meta_args.get("toolArgs", meta_args.get("arguments")),
            )
            tool_args = candidate_args if isinstance(candidate_args, dict) else None

    # Some server revisions stream enum 49 with the actual client name and
    # arguments already flattened.  Accept that only when it is not a known
    # MCP meta-tool name; the request validator still enforces the exact name.
    if not tool_name and outer_name not in MCP_META_TOOL_NAMES:
        direct_args = _json_object(raw_args)
        if direct_args is not None:
            tool_name = outer_name
            tool_args = direct_args

    if (
        not isinstance(tool_name, str)
        or not tool_name
        or tool_args is None
    ):
        return None

    normalized = _compact_json(tool_args)
    if normalized is None:
        return None
    return NativeToolCall(
        enum=enum,
        call_id=call_id.splitlines()[0],
        name=tool_name,
        raw_arguments=normalized,
        arguments=tool_args,
        is_streaming=is_streaming,
        is_last=True,
    )


def _decode_call(
    call: pb.ClientSideToolV2Call | pb.StreamedBackToolCall | pb.StreamedBackToolCallV2,
    *,
    params: pb.CallMcpToolParams | None = None,
    is_streaming: bool = False,
) -> NativeToolCall | None:
    enum = int(call.tool)
    call_id = call.toolCallId
    outer_name = call.name
    raw_args = call.rawArgs
    if enum not in MCP_TOOL_ENUMS or not call_id:
        return None

    if enum == CALL_MCP_TOOL_ENUM:
        return _translated_meta_call(
            enum=enum,
            call_id=call_id,
            outer_name=outer_name,
            raw_args=raw_args,
            params=params,
            is_streaming=is_streaming,
        )

    arguments = _json_object(raw_args)
    if not outer_name or arguments is None:
        return None
    normalized = _compact_json(arguments)
    if normalized is None:
        return None
    return NativeToolCall(
        enum=enum,
        call_id=call_id.splitlines()[0],
        name=outer_name,
        raw_arguments=normalized,
        arguments=arguments,
        is_streaming=is_streaming,
        is_last=True,
    )


def extract_native_tool_calls(data: bytes) -> list[NativeToolCall]:
    """Return complete MCP calls from the exact Cursor response locations.

    Partial tool announcements are intentionally ignored: they contain no
    executable arguments.  Avoiding a recursive wire walk also prevents nested
    JSON/status messages from being misclassified as client tool calls.
    """
    response = pb.StreamUnifiedChatWithToolsResponse()
    try:
        response.ParseFromString(data)
    except Exception:
        return []

    found: list[NativeToolCall] = []
    if response.HasField("clientSideToolV2Call"):
        call = response.clientSideToolV2Call
        params = call.callMcpToolParams if call.HasField("callMcpToolParams") else None
        decoded = _decode_call(call, params=params, is_streaming=call.isStreaming)
        if decoded is not None:
            found.append(decoded)

    if response.HasField("message"):
        message = response.message
        if message.HasField("toolCallV2"):
            call_v2 = message.toolCallV2
            params = (
                call_v2.callMcpToolParams
                if call_v2.HasField("callMcpToolParams")
                else None
            )
            decoded = _decode_call(call_v2, params=params)
            if decoded is not None:
                found.append(decoded)
        if message.HasField("toolCall"):
            decoded = _decode_call(message.toolCall, is_streaming=True)
            if decoded is not None:
                found.append(decoded)

    unique: dict[tuple[str, str], NativeToolCall] = {}
    for call in found:
        unique[(call.call_id, call.raw_arguments)] = call
    return list(unique.values())
