from __future__ import annotations

"""Tool call prompt builder — native Anthropic JSON format.

Instead of flattening tool_use/tool_result into custom XML/text, we serialize
them as native Anthropic JSON.  The model sees consistent structured examples
in its conversation history and naturally learns to output the same format.
"""

import json
from typing import TYPE_CHECKING

from claude_code.tool_choice import resolve_tool_choice

if TYPE_CHECKING:
    from claude_code.tool_protocols import ProtocolAdapter

def build_tool_call_prompt(
    tools: list[dict],
    composer: bool = False,
    compact: bool = False,
) -> str:
    """Serialize the client tool manifest without adding execution strategies."""
    del compact
    serialized = json.dumps(
        tools, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    grammar = (
        "Use the Composer marker protocol for a tool call."
        if composer
        else (
            "Use one complete native tool_use JSON object per line for a tool call: "
            '{"type":"tool_use","id":"toolu_<unique>",'
            '"name":"<exact_tool_name>","input":{...}}'
        )
    )
    return (
        "Available client tools are the following JSON schemas:\n"
        f"{serialized}\n\n{grammar}"
    )


def _serialize_anthropic_tool_use(block: dict) -> str:
    """Serialize a single tool_use block to a compact JSON line."""
    return json.dumps({
        "type": "tool_use",
        "id": block.get("id", ""),
        "name": block.get("name", ""),
        "input": block.get("input") or {},
    }, ensure_ascii=False)


def _serialize_anthropic_tool_result(block: dict) -> str:
    """Serialize a single tool_result block to a compact JSON line."""
    content = block.get("content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        content = "\n".join(parts)
    elif not isinstance(content, str):
        content = str(content) if content else ""

    obj: dict = {
        "type": "tool_result",
        "tool_use_id": block.get("tool_use_id", ""),
        "content": content,
    }
    if block.get("is_error"):
        obj["is_error"] = True
    return json.dumps(obj, ensure_ascii=False)


def _serialize_assistant_anthropic(msg: dict) -> str:
    """Serialize an assistant message using Anthropic content blocks.

    If `anthropic_content` is available, use it directly.
    Otherwise fall back to intermediate format (content + tool_calls).
    """
    blocks = msg.get("anthropic_content")
    if blocks and isinstance(blocks, list):
        parts: list[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                t = block.get("text", "")
                if t.strip():
                    parts.append(t)
            elif btype == "tool_use":
                parts.append(_serialize_anthropic_tool_use(block))
        return "\n\n".join(parts) if parts else ""

    # Fallback: reconstruct from intermediate format
    parts = []
    content = msg.get("content", "")
    if isinstance(content, str) and content.strip():
        parts.append(content)

    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or tc
        name = fn.get("name", "unknown")
        raw_args = fn.get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except (json.JSONDecodeError, ValueError):
                args = {}
        else:
            args = raw_args or {}
        tc_id = tc.get("id") or f"toolu_{name[:8]}"
        parts.append(json.dumps({
            "type": "tool_use",
            "id": tc_id,
            "name": name,
            "input": args,
        }, ensure_ascii=False))

    return "\n\n".join(parts) if parts else ""


def _serialize_tool_result_message(msg: dict, id_to_name: dict[str, str]) -> str:
    """Serialize a role:tool message as Anthropic tool_result JSON."""
    tid = msg.get("tool_call_id", "")
    content = msg.get("content", "")
    is_error = msg.get("is_error", False)

    # Check if we have original anthropic_content blocks
    anthropic = msg.get("anthropic_content")
    if anthropic and isinstance(anthropic, list):
        parts = []
        for block in anthropic:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                parts.append(_serialize_anthropic_tool_result(block))
        if parts:
            return "\n\n".join(parts)

    obj: dict = {
        "type": "tool_result",
        "tool_use_id": tid,
        "content": content if isinstance(content, str) else str(content),
    }
    if is_error:
        obj["is_error"] = True
    return json.dumps(obj, ensure_ascii=False)


def _serialize_user_with_tool_results(msg: dict, id_to_name: dict[str, str]) -> str:
    """Serialize a user message that contains tool_result blocks."""
    anthropic = msg.get("anthropic_content")
    if anthropic and isinstance(anthropic, list):
        parts = []
        for block in anthropic:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "tool_result":
                parts.append(_serialize_anthropic_tool_result(block))
            elif btype == "text":
                t = block.get("text", "")
                if t.strip():
                    parts.append(t)
        if parts:
            return "\n\n".join(parts)

    return msg.get("content", "")


def _extract_message_content(msg: dict) -> str:
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content) if content else ""


def _adapter_uses_composer_markers(adapter: ProtocolAdapter | None) -> bool:
    """Whether an adapter requires Composer's marker prompt grammar."""
    return adapter is not None and str(adapter.protocol) == "composer_marker_v1"


def inject_tool_prompt_into_messages(
    messages: list[dict],
    tools: list[dict],
    reminder_interval: int = 10,
    composer: bool = False,
    compact_tools: bool = False,
    adapter: ProtocolAdapter | None = None,
    tool_choice: dict | str | None = None,
    advertised_tool_names: list[str] | None = None,
) -> list[dict]:
    """Place one deterministic tool manifest in the system instruction.

    Existing tool-use and tool-result history is serialized for the text bridge,
    but no fake conversation turns, reminders, or acknowledgements are added.
    """
    del reminder_interval
    result: list[dict] = []
    uses_composer_markers = _adapter_uses_composer_markers(adapter) if adapter else composer
    tool_names = (
        advertised_tool_names
        if advertised_tool_names is not None
        else [(tool.get("function") or tool).get("name", "") for tool in tools]
    )
    policy = resolve_tool_choice(tool_choice, tool_names)
    instruction = policy.instruction()
    tool_prompt = ""
    if tools:
        if adapter is None:
            tool_prompt = build_tool_call_prompt(
                tools,
                composer=uses_composer_markers,
                compact=compact_tools,
            )
            tool_prompt = f"{instruction}\n\n{tool_prompt}"
        else:
            tool_prompt = adapter.render_tool_manifest(
                tools,
                instruction,
            )
    elif tool_choice is not None:
        tool_prompt = instruction

    # Build tool_use_id → tool_name map for context
    _tool_id_to_name: dict[str, str] = {}
    for m in messages:
        for src in [m.get("content", []), m.get("tool_calls", [])]:
            if not isinstance(src, list):
                continue
            for block in src:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    _tool_id_to_name[block.get("id", "")] = block.get("name", "unknown")
                fn = block.get("function")
                if isinstance(fn, dict) and fn.get("name"):
                    _tool_id_to_name[block.get("id", "")] = fn["name"]
        anthropic = m.get("anthropic_content")
        if isinstance(anthropic, list):
            for block in anthropic:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    _tool_id_to_name[block.get("id", "")] = block.get("name", "unknown")

    for m in messages:
        role = m.get("role", "")

        # --- role:tool → serialize as Anthropic tool_result JSON ---
        if role == "tool":
            serialized = _serialize_tool_result_message(m, _tool_id_to_name)
            result.append({"role": "user", "content": serialized})
            continue

        # --- user with tool_result content blocks ---
        if role == "user":
            anthropic = m.get("anthropic_content")
            has_tool_result = False
            if isinstance(anthropic, list):
                has_tool_result = any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in anthropic
                )

            if has_tool_result:
                serialized = _serialize_user_with_tool_results(m, _tool_id_to_name)
                result.append({"role": "user", "content": serialized})
                continue

            raw_content = m.get("content", "")
            if isinstance(raw_content, str) and "<tool_use_error>" in raw_content:
                result.append({
                    "role": "user",
                    "content": json.dumps({
                        "type": "tool_result",
                        "tool_use_id": "unknown",
                        "is_error": True,
                        "content": raw_content,
                    }, ensure_ascii=False),
                })
                continue

        # --- assistant → serialize with Anthropic tool_use JSON ---
        if role == "assistant":
            has_tools = bool(m.get("tool_calls")) or bool(m.get("anthropic_content"))
            if has_tools:
                serialized = _serialize_assistant_anthropic(m)
                if serialized.strip():
                    result.append({"role": "assistant", "content": serialized})
                    continue

        result.append(m)

    if tool_prompt:
        system_index = next(
            (index for index, message in enumerate(result) if message.get("role") == "system"),
            None,
        )
        if system_index is None:
            result.insert(0, {"role": "system", "content": tool_prompt})
        else:
            system_message = result[system_index]
            existing = _extract_message_content(system_message)
            combined = f"{existing}\n\n{tool_prompt}" if existing else tool_prompt
            result[system_index] = {**system_message, "content": combined}

    result = _merge_consecutive_same_role(result)
    return result


def _merge_consecutive_same_role(messages: list[dict]) -> list[dict]:
    """Merge consecutive messages with the same role to avoid API errors."""
    if not messages:
        return messages
    merged: list[dict] = [messages[0]]
    for m in messages[1:]:
        prev = merged[-1]
        cur_text = _extract_message_content(m)
        if m.get("role") == prev.get("role"):
            prev_text = _extract_message_content(prev)
            combined = f"{prev_text}\n\n{cur_text}" if prev_text and cur_text else (prev_text or cur_text)
            merged[-1] = {"role": prev.get("role"), "content": combined}
        else:
            merged.append(m)
    return merged
