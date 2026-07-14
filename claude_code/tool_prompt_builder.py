from __future__ import annotations

"""Tool call prompt builder — native Anthropic JSON format.

Instead of flattening tool_use/tool_result into custom XML/text, we serialize
them as native Anthropic JSON.  The model sees consistent structured examples
in its conversation history and naturally learns to output the same format.
"""

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claude_code.tool_protocols import ProtocolAdapter

from claude_code.tool_prompt_rules import (
    CLIENT_TOOL_INVENTORY_AUTHORITY_RULE,
    POST_TOOL_OUTPUT_FORMAT_RULE,
)
from config.system_prompt import (
    COMPOSER_TOOL_PROMPT_HEADER,
    COMPOSER_TURN1_USER,
    COMPOSER_TURN2_ASSISTANT,
    DECONTAMINATION_REMINDER,
    TURN1_USER,
    TURN2_ASSISTANT,
)

ASK_MODE_CONTAMINATION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ask\s*mode", re.IGNORECASE),
    re.compile(r"read[\s-]*only", re.IGNORECASE),
    re.compile(r"只能.*读"),
    re.compile(r"只能.*分析"),
    re.compile(r"不能.*写入"),
    re.compile(r"不能.*write", re.IGNORECASE),
    re.compile(r"无法.*写入"),
    re.compile(r"无法.*落盘"),
    re.compile(r"无法.*执行.*写"),
    re.compile(r"手动.*写入"),
    re.compile(r"手动.*粘贴"),
    re.compile(r"手动.*复制"),
    re.compile(r"手动.*apply", re.IGNORECASE),
    re.compile(r"可直接粘贴"),
    re.compile(r"可直接复制"),
    re.compile(r"直接粘贴替换"),
    re.compile(r"粘贴.*替换"),
    re.compile(r"copy.*paste", re.IGNORECASE),
    re.compile(r"paste.*into", re.IGNORECASE),
    re.compile(r"directly\s+pasteable", re.IGNORECASE),
    re.compile(
        r"I\s+can(?:'t|not)\s+(?:actually\s+)?(?:write|create|modify|execute)",
        re.IGNORECASE,
    ),
    re.compile(r"I\s+(?:don't|do\s+not)\s+have\s+write", re.IGNORECASE),
    re.compile(r"no\s+write\s+(?:access|permission)", re.IGNORECASE),
    re.compile(
        r"cannot\s+(?:actually\s+)?(?:write|create|save|execute)", re.IGNORECASE
    ),
    re.compile(r"工具.*约束"),
    re.compile(r"(?:读|read)\s*\+\s*(?:分析|analysis)", re.IGNORECASE),
]


def _compact_type(pdef: dict) -> str:
    ptype = pdef.get("type") if isinstance(pdef, dict) else None
    if isinstance(ptype, list):
        return "|".join(str(t) for t in ptype)
    if isinstance(ptype, str):
        return ptype
    if isinstance(pdef, dict):
        if pdef.get("enum"):
            return "enum"
        if pdef.get("anyOf") or pdef.get("oneOf"):
            return "value"
    return "string"


def _compact_tool_signature(tool_def: dict) -> tuple[str, str]:
    fn = tool_def.get("function") or tool_def
    name = fn.get("name", "")
    schema = fn.get("input_schema") or fn.get("parameters") or {}
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])

    params = []
    for pname, pdef in properties.items():
        suffix = "!" if pname in required else ""
        params.append(f"{pname}:{_compact_type(pdef)}{suffix}")
    return name, f"{name}({', '.join(params)})"


def _compact_example_for_tool(tool_def: dict) -> str:
    fn = tool_def.get("function") or tool_def
    name = fn.get("name", "")
    schema = fn.get("input_schema") or fn.get("parameters") or {}
    properties = schema.get("properties") or {}
    required = list(schema.get("required") or [])

    args = {}
    for pname in required[:4]:
        pdef = properties.get(pname) or {}
        ptype = _compact_type(pdef)
        if ptype == "boolean":
            args[pname] = True
        elif ptype in ("integer", "number"):
            args[pname] = 1
        elif ptype == "array":
            args[pname] = []
        elif ptype == "object":
            args[pname] = {}
        else:
            args[pname] = f"<{pname}>"

    return json.dumps(
        {"type": "tool_use", "id": "toolu_01", "name": name, "input": args},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _build_compact_tool_call_prompt(tools: list[dict]) -> str:
    usable = [tool for tool in tools if (tool.get("function") or tool).get("name")]
    example_tool = next(
        (tool for tool in usable if (tool.get("function") or tool).get("name") == "write_file"),
        usable[0] if usable else {},
    )

    lines = [
        "You have access to tools. For actions, output tool_use JSON lines only:",
        '{"type":"tool_use","id":"toolu_<id>","name":"<exact_tool_name>","input":{...}}',
        f"Example: {_compact_example_for_tool(example_tool)}" if example_tool else "",
        'When the task is complete, call {"type":"tool_use","id":"toolu_done","name":"task_complete","input":{"result":"<summary>"}}.',
        "If you plan to create, write, inspect, open, run, search, or verify anything, output the tool_use in the same response.",
        "For direct file creation requests with a named file, call write_file first.",
        "For write_file with generated HTML, prefer concise complete files under 8000 characters.",
        "Use exact CLIENT tool names listed below. Cursor built-in tool names will fail.",
        CLIENT_TOOL_INVENTORY_AUTHORITY_RULE,
        POST_TOOL_OUTPUT_FORMAT_RULE,
        "Tool signatures (! = required):",
    ]

    for index, tool_def in enumerate(usable, 1):
        fn = tool_def.get("function") or tool_def
        desc = " ".join(str(fn.get("description", "")).split())
        if len(desc) > 120:
            desc = desc[:117] + "..."
        _name, signature = _compact_tool_signature(tool_def)
        line = f"{index}. {signature}"
        if desc:
            line += f" — {desc}"
        lines.append(line)

    lines.append('Completion signal: task_complete(result:string!).')
    lines.append(f"Total: {len(usable)} client tool(s) plus task_complete.")
    return "\n".join(line for line in lines if line)


def build_tool_call_prompt(
    tools: list[dict],
    composer: bool = False,
    compact: bool = False,
) -> str:
    """Build a text prompt describing available tools.

    Non-composer models output tool calls as Anthropic native JSON:
      {"type":"tool_use","id":"toolu_xxx","name":"ToolName","input":{...}}

    Composer-2.x models use their native tool-call marker protocol instead, so
    they get a marker-format header and are told to use the client tool names.
    """
    if compact and not composer:
        return _build_compact_tool_call_prompt(tools)

    if composer:
        lines = [COMPOSER_TOOL_PROMPT_HEADER]
    else:
        lines = [
            "You have access to the following tools.\n"
            "When you need to perform an action, output Anthropic native tool_use JSON — "
            "one per line, each on its own line after any text:\n"
            '  {"type":"tool_use","id":"toolu_<unique>","name":"<ToolName>","input":{<params>}}\n\n'
            "Examples:\n"
            '  {"type":"tool_use","id":"toolu_01","name":"Bash","input":{"command":"ls -la"}}\n'
            '  {"type":"tool_use","id":"toolu_02","name":"Read","input":{"file_path":"/tmp/app.go"}}\n'
            '  {"type":"tool_use","id":"toolu_03","name":"Write","input":{"file_path":"/tmp/app.go","content":"package main\\n"}}\n'
            '  {"type":"tool_use","id":"toolu_04","name":"Edit","input":{"file_path":"/tmp/app.go","old_string":"old","new_string":"new"}}\n'
        ]

    count = 0
    for tool_def in tools:
        fn = tool_def.get("function") or tool_def
        name = fn.get("name", "")
        if not name:
            continue

        desc = fn.get("description", "")

        input_schema = fn.get("input_schema") or fn.get("parameters") or {}
        properties = input_schema.get("properties") or {}
        required = set(input_schema.get("required") or [])

        param_parts = []
        for pname, pdef in properties.items():
            ptype = pdef.get("type", "string")
            pdesc = pdef.get("description", "")
            req_marker = " [required]" if pname in required else ""
            line = f"    {pname}: {ptype}{req_marker}"
            if pdesc:
                line += f" — {pdesc}"
            param_parts.append(line)
        params_block = "\n".join(param_parts) if param_parts else "    (no parameters)"

        count += 1
        lines.append(
            f"{count}. Tool: {name}\n"
            f"  Description: {desc}\n"
            f"  Parameters:\n{params_block}"
        )

    if composer:
        lines.append(
            f"\nTotal: {count} tool(s). Call these EXACT names via the marker protocol "
            "for ALL actions — never use built-in tools, never narrate."
        )
    else:
        lines.append(
            f"\nTotal: {count} tool(s). Use tool_use JSON for ALL actions — never narrate."
        )

    return "\n\n".join(lines)


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


def _is_contaminated_assistant_message(content: str) -> bool:
    if not content:
        return False
    return any(p.search(content) for p in ASK_MODE_CONTAMINATION_PATTERNS)


def _adapter_uses_composer_markers(adapter: ProtocolAdapter | None) -> bool:
    """Whether an adapter requires Composer's marker prompt grammar."""
    return adapter is not None and str(adapter.protocol) == "composer_marker_v1"


def _build_brief_reminder(tools: list[dict], composer: bool = False) -> str:
    """Compact tool-name reminder injected periodically into conversation."""
    names = [(t.get("function") or t).get("name", "") for t in tools
             if (t.get("function") or t).get("name")]
    names.append("task_complete")
    if composer:
        return (
            f'[TOOL_REMINDER] {len(names)} client tools: {", ".join(names)}. '
            'ALWAYS execute via the marker protocol using these EXACT names — '
            'never use built-in tools (search_files/read_file/edit_file/skill_view), '
            'never narrate or simulate a call. '
            'Only task_complete signals "done"; no task_complete = keep working. '
            'Format: <|tool_calls_begin|><|tool_call_begin|>NAME<|tool_sep|>arg\\nvalue<|tool_call_end|><|tool_calls_end|>'
        )
    return (
        f'[TOOL_REMINDER] {len(names)} tools: {", ".join(names)}. '
        'ALWAYS execute tools via tool_use JSON — never narrate or simulate a call. '
        'Only task_complete signals "done"; no task_complete = keep working. '
        'Format: {"type":"tool_use","id":"toolu_<id>","name":"NAME","input":{...}}'
    )


def inject_tool_prompt_into_messages(
    messages: list[dict],
    tools: list[dict],
    reminder_interval: int = 10,
    composer: bool = False,
    compact_tools: bool = False,
    adapter: ProtocolAdapter | None = None,
) -> list[dict]:
    """Inject tool schemas and system turns into messages.

    Inserts full tool descriptions (via build_tool_call_prompt) so the model
    knows every available tool and its parameters.  Uses Anthropic native JSON
    format for tool_use/tool_result serialization in conversation history.

    When ``adapter`` is supplied, its manifest defines the active model's
    tool grammar. The legacy ``composer`` flag remains for existing callers.
    """
    result: list[dict] = []
    uses_composer_markers = _adapter_uses_composer_markers(adapter) if adapter else composer

    if not compact_tools:
        result.append({"role": "user", "content": COMPOSER_TURN1_USER if uses_composer_markers else TURN1_USER})
        result.append({"role": "assistant", "content": COMPOSER_TURN2_ASSISTANT if uses_composer_markers else TURN2_ASSISTANT})

    if tools:
        if adapter is None:
            tool_prompt = build_tool_call_prompt(
                tools,
                composer=uses_composer_markers,
                compact=compact_tools,
            )
        else:
            tool_prompt = adapter.render_tool_manifest(
                tools,
                "Use the available client tools to fulfill the request.",
            )
        result.append({"role": "user", "content": tool_prompt})
        result.append({"role": "assistant", "content": "(tools noted, ready to use them)"})

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

            content = _extract_message_content(m)
            if _is_contaminated_assistant_message(content):
                result.append(m)
                result.append({"role": "user", "content": DECONTAMINATION_REMINDER})
                continue

        result.append(m)

    if tools and reminder_interval > 0:
        reminder = _build_brief_reminder(tools, composer=uses_composer_markers)
        result = _inject_periodic_reminders(result, reminder, reminder_interval)

    result = _merge_consecutive_same_role(result)
    return result


def _inject_periodic_reminders(
    messages: list[dict], reminder: str, interval: int
) -> list[dict]:
    """Insert a tool-protocol reminder every `interval` user turns."""
    if interval <= 0 or not reminder:
        return messages

    out: list[dict] = []
    user_count = 0
    skip_first_user = True

    for i, m in enumerate(messages):
        if m.get("role") == "user":
            if skip_first_user:
                skip_first_user = False
                out.append(m)
                continue
            user_count += 1
            if user_count > 0 and user_count % interval == 0:
                content = m.get("content", "")
                is_tool_result = '"type":"tool_result"' in str(content) or '"type": "tool_result"' in str(content)
                if not is_tool_result:
                    logging.getLogger("thalamus.tool-prompt").info(
                        f"Tool reminder injected at user turn {user_count}"
                    )
                    out.append({"role": "user", "content": reminder})
                    out.append({"role": "assistant", "content": "(tools noted)"})
        out.append(m)

    return out


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
            if m.get("role") == "assistant" and not cur_text.strip():
                merged.append({"role": "assistant", "content": "(continued)"})
            else:
                merged.append(m)
    return merged
