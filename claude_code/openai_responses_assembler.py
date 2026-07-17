"""Generate OpenAI Responses API objects and streaming events."""

from __future__ import annotations

import json
import math
import time
import uuid
from typing import Any


def _usage(input_tokens: int, output_text: str) -> dict[str, Any]:
    normalized_input_tokens = max(0, int(input_tokens))
    output_tokens = math.ceil(len(output_text) / 4)
    return {
        "input_tokens": normalized_input_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": normalized_input_tokens + output_tokens,
    }


def _message_item(message_id: str, text: str, status: str = "completed") -> dict[str, Any]:
    return {
        "id": message_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{
            "type": "output_text",
            "text": text,
            "annotations": [],
            "logprobs": [],
        }],
    }


def _reasoning_item(
    reasoning_id: str,
    text: str,
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "id": reasoning_id,
        "type": "reasoning",
        "status": status,
        "summary": ([{"type": "summary_text", "text": text}] if text else []),
    }


def _function_call_item(tool_call: dict[str, Any], status: str = "completed") -> dict[str, Any]:
    call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}"
    function = tool_call.get("function") or {}
    arguments = function.get("arguments", "{}")
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)
    return {
        "id": call_id if str(call_id).startswith("fc_") else f"fc_{call_id}",
        "type": "function_call",
        "status": status,
        "call_id": call_id,
        "name": function.get("name", ""),
        "arguments": arguments,
    }


def _response_object(
    response_id: str,
    model: str,
    output: list[dict[str, Any]],
    text: str,
    input_tokens: int,
    created_at: int,
    status: str,
) -> dict[str, Any]:
    incomplete = status == "incomplete"
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "completed_at": int(time.time()) if status == "completed" else None,
        "status": status,
        "background": False,
        "error": None,
        "incomplete_details": {"reason": "max_output_tokens"} if incomplete else None,
        "instructions": None,
        "max_output_tokens": None,
        "metadata": {},
        "model": model,
        "output": output,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": None,
        "service_tier": "default",
        "store": False,
        "temperature": None,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "usage": None if status == "in_progress" else _usage(input_tokens, text),
        "user": None,
    }


def build_unary_openai_response(
    response_id: str,
    model: str,
    text: str,
    tool_calls: list[dict[str, Any]],
    stop_reason_override: str = "",
    input_tokens: int = 0,
) -> dict[str, Any]:
    """Build a complete non-streaming OpenAI Responses API response."""
    created_at = int(time.time())
    status = "incomplete" if stop_reason_override in ("max_tokens", "length") else "completed"
    output: list[dict[str, Any]] = []
    if text or not tool_calls:
        output.append(_message_item(f"msg_{uuid.uuid4().hex}", text, status=status))
    output.extend(_function_call_item(tool_call) for tool_call in tool_calls)
    return _response_object(response_id, model, output, text, input_tokens, created_at, status)


def _format_sse(event: dict[str, Any]) -> str:
    event_type = event.get("type", "message")
    return f"event: {event_type}\ndata: {json.dumps(event)}\n\n"


class StreamingOpenAIResponsesSession:
    """Manage an ordered OpenAI Responses API SSE stream."""

    def __init__(
        self,
        response_id: str,
        model: str,
        input_tokens: int = 0,
        emit_reasoning_summary: bool = True,
    ) -> None:
        self.response_id = response_id
        self.model = model
        self.input_tokens = max(0, int(input_tokens))
        self.created_at = int(time.time())
        self._sequence_number = 0
        self._message_id = f"msg_{uuid.uuid4().hex}"
        self._reasoning_id = f"rs_{uuid.uuid4().hex}"
        self._text = ""
        self._all_text = ""
        self._reasoning_text = ""
        self._emit_reasoning_summary = emit_reasoning_summary
        self._next_output_index = 0
        self._message_output_index: int | None = None
        self._reasoning_output_index: int | None = None
        self._message_started = False
        self._message_finished = False
        self._reasoning_started = False
        self._reasoning_finished = False
        self._completed_message_items: list[tuple[int, dict[str, Any]]] = []
        self._tool_items: list[tuple[int, dict[str, Any]]] = []
        self._streaming_tool_item: tuple[int, dict[str, Any]] | None = None

    def _event(self, event_type: str, **fields: Any) -> str:
        event = {
            "type": event_type,
            **fields,
            "sequence_number": self._sequence_number,
        }
        self._sequence_number += 1
        return _format_sse(event)

    def _response(self, status: str) -> dict[str, Any]:
        indexed_output: list[tuple[int, dict[str, Any]]] = list(
            self._completed_message_items
        )
        if self._reasoning_started and self._reasoning_output_index is not None:
            reasoning_status = "completed" if self._reasoning_finished else status
            indexed_output.append((
                self._reasoning_output_index,
                _reasoning_item(self._reasoning_id, self._reasoning_text, reasoning_status),
            ))
        if self._message_started:
            indexed_output.append((
                self._message_output_index or 0,
                _message_item(self._message_id, self._text, status=status),
            ))
        indexed_output.extend(self._tool_items)
        if self._streaming_tool_item is not None:
            indexed_output.append(self._streaming_tool_item)
        output = [item for _index, item in sorted(indexed_output, key=lambda entry: entry[0])]
        return _response_object(
            self.response_id,
            self.model,
            output,
            self._all_text,
            self.input_tokens,
            self.created_at,
            status,
        )

    def start(self) -> str:
        response = self._response("in_progress")
        return self._event("response.created", response=response) + self._event(
            "response.in_progress", response=response
        )

    def emit_keepalive(self) -> str:
        """Emit a recognized, non-visible Responses API progress event."""
        return self._event("response.in_progress", response=self._response("in_progress"))

    def _start_message(self) -> str:
        if self._message_started:
            return ""
        self._message_started = True
        self._message_output_index = self._next_output_index
        self._next_output_index += 1
        # Responses stream accumulators append the content part after receiving
        # response.content_part.added.  Starting with an empty output_text here
        # duplicates content_index=0 in strict SDK consumers.
        item = {
            "id": self._message_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        return self._event(
            "response.output_item.added",
            output_index=self._message_output_index,
            item=item,
        ) + self._event(
            "response.content_part.added",
            item_id=self._message_id,
            output_index=self._message_output_index,
            content_index=0,
            part={"type": "output_text", "text": "", "annotations": [], "logprobs": []},
        )

    def emit_text_delta(self, text: str) -> str:
        if not text:
            return ""
        out = self._finish_reasoning() + self._start_message()
        self._text += text
        self._all_text += text
        return out + self._event(
            "response.output_text.delta",
            item_id=self._message_id,
            output_index=self._message_output_index,
            content_index=0,
            delta=text,
            logprobs=[],
        )

    def emit_reasoning_delta(self, text: str) -> str:
        if not text or not self._emit_reasoning_summary or self._reasoning_finished:
            return ""
        out = self._start_reasoning()
        self._reasoning_text += text
        return out + self._event(
            "response.reasoning_summary_text.delta",
            item_id=self._reasoning_id,
            output_index=self._reasoning_output_index,
            summary_index=0,
            delta=text,
        )

    def _start_reasoning(self) -> str:
        if self._reasoning_started:
            return ""
        self._reasoning_started = True
        self._reasoning_output_index = self._next_output_index
        self._next_output_index += 1
        item = _reasoning_item(self._reasoning_id, "", status="in_progress")
        return self._event(
            "response.output_item.added",
            output_index=self._reasoning_output_index,
            item=item,
        ) + self._event(
            "response.reasoning_summary_part.added",
            item_id=self._reasoning_id,
            output_index=self._reasoning_output_index,
            summary_index=0,
            part={"type": "summary_text", "text": ""},
        )

    def _finish_reasoning(self, status: str = "completed") -> str:
        if not self._reasoning_started or self._reasoning_finished:
            return ""
        self._reasoning_finished = True
        part = {"type": "summary_text", "text": self._reasoning_text}
        item = _reasoning_item(self._reasoning_id, self._reasoning_text, status=status)
        return self._event(
            "response.reasoning_summary_text.done",
            item_id=self._reasoning_id,
            output_index=self._reasoning_output_index,
            summary_index=0,
            text=self._reasoning_text,
        ) + self._event(
            "response.reasoning_summary_part.done",
            item_id=self._reasoning_id,
            output_index=self._reasoning_output_index,
            summary_index=0,
            part=part,
        ) + self._event(
            "response.output_item.done",
            output_index=self._reasoning_output_index,
            item=item,
        )

    def _finish_message(self, status: str = "completed") -> str:
        if not self._message_started or self._message_finished:
            return ""
        self._message_finished = True
        part = {
            "type": "output_text",
            "text": self._text,
            "annotations": [],
            "logprobs": [],
        }
        item = _message_item(self._message_id, self._text, status=status)
        out = self._event(
            "response.output_text.done",
            item_id=self._message_id,
            output_index=self._message_output_index,
            content_index=0,
            text=self._text,
            logprobs=[],
        ) + self._event(
            "response.content_part.done",
            item_id=self._message_id,
            output_index=self._message_output_index,
            content_index=0,
            part=part,
        ) + self._event(
            "response.output_item.done",
            output_index=self._message_output_index,
            item=item,
        )
        self._completed_message_items.append((self._message_output_index or 0, item))
        self._message_id = f"msg_{uuid.uuid4().hex}"
        self._text = ""
        self._message_output_index = None
        self._message_started = False
        self._message_finished = False
        return out

    @property
    def has_open_message(self) -> bool:
        return self._message_started and not self._message_finished

    def flush_text_item(self) -> str:
        """Complete a quiescent visible-text item while keeping the response open."""
        return self._finish_message()

    def emit_tool_use_blocks(self, tool_calls: list[dict[str, Any]]) -> str:
        out = self._finish_reasoning() + self._finish_message()
        for tool_call in tool_calls:
            item = _function_call_item(tool_call)
            out += self.emit_tool_call_start(item["call_id"], item["name"])
            out += self.finish_tool_call(tool_call)
        return out

    def emit_tool_call_start(self, call_id: str, name: str) -> str:
        """Emit an in-progress function item before its arguments are complete."""
        if self._streaming_tool_item is not None:
            return ""
        out = self._finish_reasoning() + self._finish_message()
        output_index = self._next_output_index
        self._next_output_index += 1
        item = _function_call_item(
            {
                "id": call_id,
                "function": {"name": name, "arguments": ""},
            },
            status="in_progress",
        )
        self._streaming_tool_item = (output_index, item)
        return out + self._event(
            "response.output_item.added",
            output_index=output_index,
            item=item,
        )

    def finish_tool_call(self, tool_call: dict[str, Any]) -> str:
        """Finish the function item previously exposed by emit_tool_call_start."""
        if self._streaming_tool_item is None:
            return ""
        output_index, in_progress = self._streaming_tool_item
        item = _function_call_item(tool_call)
        if item["call_id"] != in_progress["call_id"]:
            return ""
        out = self._event(
            "response.function_call_arguments.delta",
            item_id=item["id"],
            output_index=output_index,
            delta=item["arguments"],
        )
        out += self._event(
            "response.function_call_arguments.done",
            item_id=item["id"],
            output_index=output_index,
            name=item["name"],
            arguments=item["arguments"],
        )
        out += self._event(
            "response.output_item.done",
            output_index=output_index,
            item=item,
        )
        self._tool_items.append((output_index, item))
        self._streaming_tool_item = None
        return out

    def finish(self, stop_reason: str = "stop") -> str:
        status = "incomplete" if stop_reason in ("max_tokens", "length") else "completed"
        out = self._finish_reasoning(status=status) + self._finish_message(status=status)
        if self._streaming_tool_item is not None:
            output_index, item = self._streaming_tool_item
            completed = {**item, "status": status}
            out += self._event(
                "response.output_item.done",
                output_index=output_index,
                item=completed,
            )
            self._tool_items.append((output_index, completed))
            self._streaming_tool_item = None
        response = self._response(status)
        event_type = "response.incomplete" if status == "incomplete" else "response.completed"
        return out + self._event(event_type, response=response)
