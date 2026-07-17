"""Generate Anthropic-format SSE events for streaming and non-streaming responses."""

from __future__ import annotations

import json
import math
import uuid
from typing import Any


def parse_tool_input(input_text: str) -> dict[str, Any]:
    """Parse tool arguments string to dict. Returns {} on failure."""
    if not input_text or not input_text.strip():
        return {}
    try:
        return json.loads(input_text)
    except json.JSONDecodeError:
        return {}


def format_sse(event: str, data: dict[str, Any]) -> str:
    """Format a single SSE event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


class StreamingAnthropicSession:
    """Manages SSE event sequence for a single streaming response."""

    format_sse = staticmethod(format_sse)

    def __init__(self, message_id: str, model: str, input_tokens: int = 0) -> None:
        self.message_id = message_id
        self.model = model
        self.input_tokens = max(0, int(input_tokens))
        self.block_index = 0
        self.thinking_index = -1
        self.text_index = -1
        self.thinking_open = False
        self.text_open = False
        self.tool_open = False
        self.tool_index = -1
        self.tool_id = ""
        self.tool_name = ""
        self.total_text_len = 0
        self.total_thinking_len = 0

    def _format(self, event: str, data: dict[str, Any]) -> str:
        return format_sse(event, data)

    def emit_message_start(self) -> str:
        message_start = self._format(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
                },
            },
        )
        ping = self._format("ping", {"type": "ping"})
        return message_start + ping

    def emit_keepalive(self) -> str:
        """Emit Anthropic's protocol-level, non-visible ping event."""
        return self._format("ping", {"type": "ping"})

    def emit_thinking_delta(self, text: str) -> str:
        out: list[str] = []
        if not self.thinking_open:
            self.thinking_index = self.block_index
            self.block_index += 1
            self.thinking_open = True
            out.append(
                self._format(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": self.thinking_index,
                        "content_block": {"type": "thinking", "thinking": ""},
                    },
                )
            )
        self.total_thinking_len += len(text)
        out.append(
            self._format(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self.thinking_index,
                    "delta": {"type": "thinking_delta", "thinking": text},
                },
            )
        )
        return "".join(out)

    def emit_text_delta(self, text: str) -> str:
        out: list[str] = []
        if not self.text_open:
            if self.thinking_open:
                out.append(
                    self._format(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": self.thinking_index},
                    )
                )
                self.thinking_open = False
            self.text_index = self.block_index
            self.block_index += 1
            self.text_open = True
            out.append(
                self._format(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": self.text_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            )
        self.total_text_len += len(text)
        out.append(
            self._format(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self.text_index,
                    "delta": {"type": "text_delta", "text": text},
                },
            )
        )
        return "".join(out)

    def close_open_blocks(self) -> str:
        out: list[str] = []
        if self.thinking_open:
            out.append(
                self._format(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": self.thinking_index},
                )
            )
            self.thinking_open = False
        if self.text_open:
            out.append(
                self._format(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": self.text_index},
                )
            )
            self.text_open = False
        if self.tool_open:
            out.append(
                self._format(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": self.tool_index},
                )
            )
            self.tool_open = False
        return "".join(out)

    def emit_tool_call_start(self, tc_id: str, name: str) -> str:
        """Expose a native tool as soon as Cursor announces its identity."""
        if self.tool_open:
            return ""
        out = self.close_open_blocks()
        self.tool_index = self.block_index
        self.block_index += 1
        self.tool_id = tc_id
        self.tool_name = name
        self.tool_open = True
        return out + self._format(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": self.tool_index,
                "content_block": {
                    "type": "tool_use",
                    "id": tc_id,
                    "name": name,
                    "input": {},
                },
            },
        )

    def finish_tool_call(self, args: str | dict[str, Any]) -> str:
        """Finish the currently announced tool with its authoritative args."""
        if not self.tool_open:
            return ""
        if isinstance(args, dict):
            partial_json = json.dumps(args)
        else:
            partial_json = args if isinstance(args, str) else "{}"
        out = self._format(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": self.tool_index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": partial_json,
                },
            },
        )
        out += self._format(
            "content_block_stop",
            {"type": "content_block_stop", "index": self.tool_index},
        )
        self.tool_open = False
        return out

    def emit_tool_use_blocks(self, tool_calls: list[dict[str, Any]]) -> str:
        out: list[str] = []
        for tc in tool_calls:
            tc_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:20]}")
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            args_str = fn.get("arguments", "")
            if isinstance(args_str, dict):
                parsed = args_str
            else:
                parsed = parse_tool_input(args_str if isinstance(args_str, str) else "{}")

            out.append(self.emit_tool_call_start(tc_id, name))
            out.append(self.finish_tool_call(parsed))
        return "".join(out)

    def finish(self, stop_reason: str = "end_turn") -> str:
        out: list[str] = []
        out.append(self.close_open_blocks())

        has_content = self.total_text_len > 0 or self.total_thinking_len > 0
        if stop_reason != "tool_use" and not has_content:
            out.append(
                self._format(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": self.block_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            )
            out.append(
                self._format(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": self.block_index},
                )
            )

        output_tokens = math.ceil(
            (self.total_text_len + self.total_thinking_len) / 4
        )
        out.append(
            self._format(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {
                        "input_tokens": self.input_tokens,
                        "output_tokens": output_tokens,
                    },
                },
            )
        )
        out.append(
            self._format("message_stop", {"type": "message_stop"})
        )
        return "".join(out)


def build_unary_anthropic_response(
    message_id: str,
    model: str,
    text: str,
    thinking: str,
    tool_calls: list[dict[str, Any]],
    stop_reason_override: str = "",
    input_tokens: int = 0,
) -> dict[str, Any]:
    """Build a complete non-streaming Anthropic response dict."""
    content: list[dict[str, Any]] = []

    if thinking:
        content.append({"type": "thinking", "thinking": thinking})
    if text:
        content.append({"type": "text", "text": text})

    for tc in tool_calls:
        tc_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:20]}")
        fn = tc.get("function") or {}
        name = fn.get("name", "")
        args_str = fn.get("arguments", "")
        if isinstance(args_str, dict):
            parsed = args_str
        else:
            parsed = parse_tool_input(args_str if isinstance(args_str, str) else "{}")
        content.append(
            {
                "type": "tool_use",
                "id": tc_id,
                "name": name,
                "input": parsed,
            }
        )

    if not content:
        content.append({"type": "text", "text": ""})

    stop_reason = stop_reason_override
    if not stop_reason:
        stop_reason = "tool_use" if tool_calls else "end_turn"

    total_len = len(text or "") + len(thinking or "")
    output_tokens = math.ceil(total_len / 4)

    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": max(0, int(input_tokens)),
            "output_tokens": output_tokens,
        },
    }
