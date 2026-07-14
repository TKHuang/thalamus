"""Generate OpenAI Chat Completions SSE events for streaming and non-streaming responses."""

from __future__ import annotations

import json
import math
import time
import uuid
from typing import Any


def _format_openai_sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"


class StreamingOpenAISession:
    """Manages SSE event sequence for a single OpenAI-format streaming response."""

    def __init__(self, completion_id: str, model: str) -> None:
        self.completion_id = completion_id
        self.model = model
        self.created = int(time.time())
        self._tool_call_index = 0
        self._has_content = False
        self._total_text_len = 0

    def _base_chunk(self, **extra: Any) -> dict[str, Any]:
        chunk: dict[str, Any] = {
            "id": self.completion_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
        }
        chunk.update(extra)
        return chunk

    def emit_role_chunk(self) -> str:
        """Emit the initial chunk that establishes the assistant role."""
        return _format_openai_sse(self._base_chunk(
            choices=[{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }],
        ))

    def emit_reasoning_delta(self, text: str) -> str:
        """Emit one OpenAI-compatible reasoning fragment without visible content."""
        self._total_text_len += len(text)
        return _format_openai_sse(self._base_chunk(
            choices=[{
                "index": 0,
                "delta": {"reasoning_content": text},
                "finish_reason": None,
            }],
        ))

    def emit_text_delta(self, text: str) -> str:
        self._has_content = True
        self._total_text_len += len(text)
        return _format_openai_sse(self._base_chunk(
            choices=[{
                "index": 0,
                "delta": {"content": text},
                "finish_reason": None,
            }],
        ))

    def emit_tool_call_start(self, tc_id: str, name: str) -> str:
        """Emit the first chunk for a tool call (id + function name)."""
        self._has_content = True
        idx = self._tool_call_index
        return _format_openai_sse(self._base_chunk(
            choices=[{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": idx,
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    }],
                },
                "finish_reason": None,
            }],
        ))

    def emit_tool_call_args_delta(self, args_fragment: str) -> str:
        """Emit an arguments fragment for the current tool call."""
        idx = self._tool_call_index
        return _format_openai_sse(self._base_chunk(
            choices=[{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": idx,
                        "function": {"arguments": args_fragment},
                    }],
                },
                "finish_reason": None,
            }],
        ))

    def advance_tool_call(self) -> None:
        self._tool_call_index += 1

    def emit_tool_use_blocks(self, tool_calls: list[dict[str, Any]]) -> str:
        """Emit complete tool call blocks (start + full args + advance)."""
        out: list[str] = []
        for tc in tool_calls:
            tc_id = tc.get("id", f"call_{uuid.uuid4().hex[:24]}")
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            args_str = fn.get("arguments", "{}")
            if isinstance(args_str, dict):
                args_str = json.dumps(args_str)

            out.append(self.emit_tool_call_start(tc_id, name))
            out.append(self.emit_tool_call_args_delta(args_str))
            self.advance_tool_call()
        return "".join(out)

    def finish(self, stop_reason: str = "stop") -> str:
        """Emit the final chunk with finish_reason and [DONE] sentinel."""
        reason_map = {
            "end_turn": "stop",
            "stop": "stop",
            "tool_use": "tool_calls",
            "tool_calls": "tool_calls",
            "max_tokens": "length",
            "length": "length",
        }
        finish_reason = reason_map.get(stop_reason, "stop")

        out: list[str] = []

        if not self._has_content:
            out.append(self.emit_text_delta(""))

        out.append(_format_openai_sse(self._base_chunk(
            choices=[{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }],
            usage={
                "prompt_tokens": 0,
                "completion_tokens": math.ceil(self._total_text_len / 4),
                "total_tokens": math.ceil(self._total_text_len / 4),
            },
        )))
        out.append("data: [DONE]\n\n")
        return "".join(out)


def build_unary_openai_response(
    completion_id: str,
    model: str,
    text: str,
    tool_calls: list[dict[str, Any]],
    stop_reason_override: str = "",
) -> dict[str, Any]:
    """Build a complete non-streaming OpenAI Chat Completion response."""
    message: dict[str, Any] = {"role": "assistant"}

    if text:
        message["content"] = text
    else:
        message["content"] = None

    if tool_calls:
        message["tool_calls"] = []
        for tc in tool_calls:
            tc_id = tc.get("id", f"call_{uuid.uuid4().hex[:24]}")
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            args_str = fn.get("arguments", "{}")
            if isinstance(args_str, dict):
                args_str = json.dumps(args_str)
            message["tool_calls"].append({
                "id": tc_id,
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            })

    if stop_reason_override:
        finish_reason = {"max_tokens": "length", "tool_use": "tool_calls"}.get(
            stop_reason_override, stop_reason_override
        )
    else:
        finish_reason = "tool_calls" if tool_calls else "stop"

    total_len = len(text or "")
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": math.ceil(total_len / 4),
            "total_tokens": math.ceil(total_len / 4),
        },
    }
