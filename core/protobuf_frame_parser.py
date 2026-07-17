from __future__ import annotations
"""
Cursor streaming response frame parser with cross-chunk buffering.

Cursor's server-streaming RPC sends data in a custom framing format
(not standard gRPC-Web). Each frame is [1-byte magic][4-byte BE length][payload].
HTTP/2 data frames can split a logical frame across multiple chunks, so we
accumulate a buffer and only decode complete frames.

Magic 0/1 carry protobuf payloads (raw / gzip-compressed).
Magic 2/3 carry JSON error messages (raw / gzip-compressed).

Port of: cursor_streaming_response_protobuf_frame_parser.js
"""

import gzip
import json
import math
import struct
from dataclasses import dataclass, field

from google.protobuf.unknown_fields import UnknownFieldSet

from proto import cursor_api_pb2 as pb
from core.protobuf_tool_call_parser import NativeToolCall, extract_native_tool_calls

from utils.structured_logging import ThalamusStructuredLogger

logger = ThalamusStructuredLogger.get_logger("protobuf-parser", "DEBUG")

CURSOR_ABORT_ERROR_CODE = "ERROR_USER_ABORTED_REQUEST"


@dataclass
class ParsedError:
    raw: str
    parsed: dict | None = None
    detail: str = ""
    error_code: str | None = None


@dataclass
class ParseResult:
    thinking: str = ""
    text: str = ""
    errors: list[ParsedError] = field(default_factory=list)
    context_remaining_percent: float | None = None
    native_tool_calls: list[NativeToolCall] = field(default_factory=list)


def _extract_context_remaining_percent(message) -> float | None:
    """Decode Cursor's server-computed remaining-context percentage."""
    try:
        message_fields = UnknownFieldSet(message)
    except (NotImplementedError, TypeError):
        return None

    remaining_percent: float | None = None
    for field in message_fields:
        if field.field_number != 30 or field.wire_type != 2:
            continue
        value = _decode_remaining_percent_field(field.data)
        if value is not None:
            remaining_percent = value
    return remaining_percent


def _decode_remaining_percent_field(data: bytes) -> float | None:
    """Read fixed32 field 4 from Cursor's undocumented field-30 message."""
    offset = 0
    while offset < len(data):
        decoded_key = _read_varint(data, offset)
        if decoded_key is None:
            return None
        key, offset = decoded_key
        field_number = key >> 3
        wire_type = key & 7

        if wire_type == 0:
            decoded_value = _read_varint(data, offset)
            if decoded_value is None:
                return None
            _, offset = decoded_value
        elif wire_type == 1:
            offset += 8
        elif wire_type == 2:
            decoded_length = _read_varint(data, offset)
            if decoded_length is None:
                return None
            length, offset = decoded_length
            offset += length
        elif wire_type == 5:
            if offset + 4 > len(data):
                return None
            if field_number == 4:
                value = struct.unpack("<f", data[offset:offset + 4])[0]
                if math.isfinite(value) and 0.0 <= value <= 100.0:
                    return value
            offset += 4
        else:
            return None

        if offset > len(data):
            return None
    return None


def _read_varint(data: bytes, offset: int) -> tuple[int, int] | None:
    value = 0
    shift = 0
    while offset < len(data) and shift < 64:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    return None


class ProtobufFrameParser:
    """Stateful parser that accumulates a buffer across successive HTTP/2 chunks."""

    __slots__ = ("_buf",)

    def __init__(self) -> None:
        self._buf = b""

    def parse(self, chunk: bytes) -> ParseResult:
        self._buf += bytes(chunk)

        thinking_parts: list[str] = []
        text_parts: list[str] = []
        errors: list[ParsedError] = []
        context_remaining_percent: float | None = None
        native_tool_calls: list[NativeToolCall] = []

        while len(self._buf) >= 5:
            magic = self._buf[0]
            (data_length,) = struct.unpack(">I", self._buf[1:5])

            if len(self._buf) < 5 + data_length:
                break

            data = self._buf[5 : 5 + data_length]
            self._buf = self._buf[5 + data_length :]

            try:
                if magic in (0, 1):
                    raw_data = gzip.decompress(data) if magic == 1 else data
                    logger.debug(f"Frame magic={magic} raw_len={len(raw_data)} hex_head={raw_data[:40].hex()}")
                    response = pb.StreamUnifiedChatWithToolsResponse()
                    response.ParseFromString(raw_data)
                    native_tool_calls.extend(extract_native_tool_calls(raw_data))

                    if response.HasField("message"):
                        msg = response.message
                        current_remaining = _extract_context_remaining_percent(msg)
                        if current_remaining is not None:
                            context_remaining_percent = current_remaining
                        if msg.HasField("thinking") and msg.thinking.content:
                            thinking_parts.append(msg.thinking.content)
                        if msg.content:
                            text_parts.append(msg.content)

                elif magic in (2, 3):
                    raw_data = gzip.decompress(data) if magic == 3 else data
                    utf8 = raw_data.decode("utf-8", errors="replace")

                    if utf8 and utf8 not in ("{}", "null"):
                        try:
                            err_obj = json.loads(utf8)
                            error_code = _deep_get(err_obj, "error", "details", 0, "debug", "error")
                            detail = (
                                _deep_get(err_obj, "error", "details", 0, "debug", "details", "detail")
                                or _deep_get(err_obj, "error", "message")
                                or utf8
                            )
                            is_abort = error_code == CURSOR_ABORT_ERROR_CODE
                            log_fn = logger.debug if is_abort else logger.warn
                            log_fn(f"Cursor error frame: code={error_code} detail={detail[:200]}")
                            errors.append(ParsedError(
                                raw=utf8,
                                parsed=err_obj,
                                detail=detail,
                                error_code=error_code,
                            ))
                            if not is_abort:
                                text_parts.append(f"[Error] {detail}")
                        except (json.JSONDecodeError, ValueError):
                            logger.warn(f"Cursor error frame (unparseable): {utf8[:300]}")
                            text_parts.append(f"[Error] {utf8}")
                            errors.append(ParsedError(raw=utf8, detail=utf8))

            except Exception as exc:
                logger.warn(f"Frame decode error: {exc}")

        return ParseResult(
            thinking="".join(thinking_parts),
            text="".join(text_parts),
            errors=errors,
            context_remaining_percent=context_remaining_percent,
            native_tool_calls=native_tool_calls,
        )


def _deep_get(obj, *keys):
    """Safely traverse nested dicts/lists. Returns None on any miss."""
    for k in keys:
        if obj is None:
            return None
        if isinstance(k, int):
            if isinstance(obj, list) and 0 <= k < len(obj):
                obj = obj[k]
            else:
                return None
        elif isinstance(obj, dict):
            obj = obj.get(k)
        else:
            return None
    return obj
