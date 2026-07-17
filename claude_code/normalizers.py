from __future__ import annotations

"""Normalize Anthropic / OpenAI payloads into UnifiedRequest.

Every field CC sends is preserved — nothing is silently dropped.
"""

import copy
import json
import re
from typing import Any

from utils.structured_logging import ThalamusStructuredLogger
from core.unified_request import UnifiedRequest

logger = ThalamusStructuredLogger.get_logger("normalizers", "DEBUG")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _copy_tool_schema(schema: Any) -> Any:
    """Isolate a request schema without changing its client-provided meaning."""
    return copy.deepcopy(schema)


_CLAUDE_FALLBACK = "claude-4.5-haiku"

_CC_TO_CURSOR_MODEL_MAP = {
    "default": "default",
    "inherit": _CLAUDE_FALLBACK,
    "sonnet": "claude-4.5-sonnet",
    "opus": "claude-4.5-opus-high",
    "haiku": "claude-4.5-haiku",
    "fast": "gpt-5.3-codex-spark-preview-low",
    "thalamus": "gemini-3.1-pro",
}

_CC_ANTHROPIC_TO_CURSOR = {
    "claude-sonnet-4": "claude-4-sonnet",
    "claude-sonnet-4-5": "claude-4.5-sonnet",
    "claude-opus-4": "claude-4.5-opus-high",
    "claude-opus-4-5": "claude-4.5-opus-high",
    "claude-haiku-4": "claude-4.5-haiku",
    "claude-haiku-4-5": "claude-4.5-haiku",
    "claude-3-5-sonnet": "claude-4.5-sonnet",
    "claude-3-5-haiku": "claude-4.5-haiku",
    "claude-3-opus": "claude-4.5-opus-high",
}


def resolve_model_name(model_name: str) -> str:
    """Map only documented aliases and unambiguous Anthropic API IDs.

    Cursor-native and future ``claude-*`` IDs pass through unchanged. Cursor
    decides whether those IDs are available instead of silently substituting a
    different model.
    """
    if not model_name or not model_name.strip():
        return _CLAUDE_FALLBACK

    normalized = model_name.strip()
    lower = normalized.lower()

    if lower in _CC_TO_CURSOR_MODEL_MAP:
        resolved = _CC_TO_CURSOR_MODEL_MAP[lower]
        if lower != resolved:
            logger.info(f"Model '{model_name}' mapped to '{resolved}'")
        return resolved

    for api_model, cursor_model in _CC_ANTHROPIC_TO_CURSOR.items():
        if lower == api_model or re.fullmatch(rf"{re.escape(api_model)}-\d{{8}}", lower):
            logger.info(f"Model '{model_name}' mapped to '{cursor_model}'")
            return cursor_model

    return normalized


def _flatten_tool_result_content(content: Any) -> str:
    """Flatten tool_result content to a plain string.

    Handles all three forms observed in CC traffic:
      str           -> pass-through
      list[{text}]  -> join with newlines
      None          -> empty string
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Anthropic normalizer
# ---------------------------------------------------------------------------

def normalize_anthropic(payload: dict) -> UnifiedRequest:
    """Convert an Anthropic Messages API payload to UnifiedRequest.

    Fixes over the old normalize_anthropic_payload():
      - tool_result.content list form correctly flattened (Bug 1)
      - mixed text + tool_result user messages handled without duplication (Bug 2)
      - is_error preserved on role:tool messages (Bug 3)
      - assistant messages with only tool_use never dropped (Bug 4)
      - metadata, thinking, context_management, tool_choice all preserved
    """
    messages: list[dict[str, Any]] = []

    # --- system ---
    system_parts: list[str] = []
    sys_content = payload.get("system")
    if isinstance(sys_content, list):
        for item in sys_content:
            if isinstance(item, dict):
                text = item.get("text", "")
                if item.get("cache_control"):
                    logger.debug(
                        f"system block cache_control={item['cache_control']} "
                        f"(not forwarded to Cursor)"
                    )
            elif isinstance(item, str):
                text = item
            else:
                text = str(item) if item else ""
            if text:
                system_parts.append(text)
    elif isinstance(sys_content, str) and sys_content:
        system_parts.append(sys_content)
    system_text = "\n\n".join(system_parts)

    # --- messages ---
    for msg in payload.get("messages") or []:
        role = msg.get("role", "user")
        raw_content = msg.get("content")

        if role == "assistant":
            _convert_assistant_message(raw_content, messages)
        elif role == "user":
            _convert_user_message(raw_content, messages)
        else:
            messages.append({"role": role, "content": _text_from_content(raw_content)})

    # --- tools ---
    original_tools = payload.get("tools") or []
    ir_tools = [
        {
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": _copy_tool_schema(t.get("input_schema")),
            },
        }
        for t in original_tools
        if t.get("name") != "BatchTool"
    ]

    original_model = payload.get("model", "")

    return UnifiedRequest(
        messages=messages,
        system=system_text,
        tools=ir_tools,
        model=resolve_model_name(original_model),
        stream=payload.get("stream") is True,
        max_tokens=payload.get("max_tokens"),
        original_format="anthropic",
        original_model=original_model,
        original_tools=original_tools,
        metadata=payload.get("metadata"),
        thinking=payload.get("thinking"),
        context_management=payload.get("context_management"),
        tool_choice=payload.get("tool_choice"),
    )


def _convert_assistant_message(
    raw_content: Any, out: list[dict[str, Any]]
) -> None:
    """Convert a single assistant message, handling text + tool_use + thinking.

    Preserves the original Anthropic content array as `anthropic_content`
    so downstream code can serialize it in native format instead of flattening.
    """
    if isinstance(raw_content, str):
        out.append({"role": "assistant", "content": raw_content})
        return
    if not isinstance(raw_content, list):
        out.append({"role": "assistant", "content": str(raw_content) if raw_content else ""})
        return

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    anthropic_blocks: list[dict[str, Any]] = []

    for block in raw_content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            t = block.get("text", "")
            if t:
                text_parts.append(t)
                anthropic_blocks.append(block)
        elif btype == "tool_use":
            tool_calls.append({
                "type": "function",
                "id": block.get("id", ""),
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input") or {}),
                },
            })
            anthropic_blocks.append(block)
        elif btype == "thinking":
            pass

    new_msg: dict[str, Any] = {
        "role": "assistant",
        "content": "\n\n".join(text_parts),
    }
    if tool_calls:
        new_msg["tool_calls"] = tool_calls
    if anthropic_blocks:
        new_msg["anthropic_content"] = anthropic_blocks
    out.append(new_msg)


def _convert_user_message(
    raw_content: Any, out: list[dict[str, Any]]
) -> None:
    """Convert a user message that may contain text, tool_result, or both.

    Emits role:tool messages BEFORE the role:user text (if any), matching
    the expected OpenAI conversation order.  Preserves `anthropic_content`
    for native-format serialization downstream.
    """
    if isinstance(raw_content, str):
        out.append({"role": "user", "content": raw_content})
        return
    if not isinstance(raw_content, list):
        out.append({"role": "user", "content": str(raw_content) if raw_content else ""})
        return

    text_parts: list[str] = []
    tool_results: list[dict[str, Any]] = []
    anthropic_blocks: list[dict[str, Any]] = []

    for block in raw_content:
        if not isinstance(block, dict):
            if isinstance(block, str):
                text_parts.append(block)
            continue
        btype = block.get("type")
        if btype == "tool_result":
            tool_results.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id", ""),
                "content": _flatten_tool_result_content(block.get("content")),
                "is_error": bool(block.get("is_error")),
            })
            anthropic_blocks.append(block)
        elif btype == "text":
            t = block.get("text", "")
            if t:
                text_parts.append(t)
                anthropic_blocks.append(block)
        elif btype == "image":
            text_parts.append("[image]")
            anthropic_blocks.append(block)
        else:
            anthropic_blocks.append(block)

    for tr in tool_results:
        if anthropic_blocks:
            tr["anthropic_content"] = anthropic_blocks
        out.append(tr)

    user_text = "\n\n".join(text_parts)
    if user_text:
        msg: dict[str, Any] = {"role": "user", "content": user_text}
        if anthropic_blocks:
            msg["anthropic_content"] = anthropic_blocks
        out.append(msg)


def _text_from_content(content: Any) -> str:
    """Extract plain text from any content shape."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", ""))
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        return content.get("text", "") or content.get("content", "")
    return str(content) if content is not None else ""


# ---------------------------------------------------------------------------
# OpenAI normalizer
# ---------------------------------------------------------------------------

def normalize_openai(payload: dict) -> UnifiedRequest:
    """Convert an OpenAI Chat Completions API payload to UnifiedRequest."""
    raw_messages = payload.get("messages") or []

    # OpenAI-compatible clients may use either role for high-priority runtime
    # context. Cursor's wire format has no developer role, so fold both into the
    # instruction instead of accidentally serializing developer text as an
    # assistant answer.
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    for msg in raw_messages:
        if msg.get("role") in ("system", "developer"):
            c = msg.get("content", "")
            if isinstance(c, str) and c:
                system_parts.append(c)
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict):
                        system_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        system_parts.append(part)
        else:
            messages.append(msg)

    system_text = "\n\n".join(system_parts)

    # Tools are already in OpenAI format
    raw_tools = payload.get("tools") or []
    ir_tools = []
    for t in raw_tools:
        fn = t.get("function", t)
        ir_tools.append({
            "type": "function",
            "function": {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": _copy_tool_schema(fn.get("parameters")),
            },
        })

    original_model = payload.get("model", "")

    return UnifiedRequest(
        messages=messages,
        system=system_text,
        tools=ir_tools,
        model=resolve_model_name(original_model),
        stream=payload.get("stream") is True,
        max_tokens=payload.get("max_tokens"),
        original_format="openai",
        original_model=original_model,
        original_tools=raw_tools,
        metadata=None,
        thinking=None,
        context_management=None,
        tool_choice=payload.get("tool_choice"),
    )


def normalize_openai_response(payload: dict) -> UnifiedRequest:
    """Convert an OpenAI Responses API payload to UnifiedRequest."""
    system_parts: list[str] = []
    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions:
        system_parts.append(instructions)

    messages: list[dict[str, Any]] = []
    raw_input = payload.get("input", "")
    if isinstance(raw_input, str):
        if raw_input:
            messages.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        for item in raw_input:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type", "message")
            role = item.get("role", "user")
            if item_type == "function_call":
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": item.get("call_id", item.get("id", "")),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", "{}"),
                        },
                    }],
                })
                continue
            if item_type == "function_call_output":
                messages.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": _flatten_tool_result_content(item.get("output")),
                })
                continue
            if item_type == "reasoning":
                continue

            item_content = _responses_content_to_openai(item.get("content", ""))
            if role in ("system", "developer"):
                if item_content:
                    system_parts.append(item_content)
            else:
                messages.append({"role": role, "content": item_content})

    raw_tools = payload.get("tools") or []
    ir_tools: list[dict[str, Any]] = []
    for tool in raw_tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        ir_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": _copy_tool_schema(tool.get("parameters")),
            },
        })

    original_model = payload.get("model", "")
    return UnifiedRequest(
        messages=messages,
        system="\n\n".join(system_parts),
        tools=ir_tools,
        model=resolve_model_name(original_model),
        stream=payload.get("stream") is True,
        max_tokens=payload.get("max_output_tokens"),
        original_format="openai_responses",
        original_model=original_model,
        original_tools=raw_tools,
        metadata=payload.get("metadata"),
        thinking=payload.get("reasoning"),
        context_management=payload.get("context_management"),
        tool_choice=payload.get("tool_choice"),
    )


def _responses_content_to_openai(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content is not None else ""

    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") in ("input_text", "output_text", "text"):
            parts.append(part.get("text", ""))
    return "\n".join(part for part in parts if part)
