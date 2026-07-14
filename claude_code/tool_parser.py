"""
Extract tool call JSON from LLM text output.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from utils.structured_logging import ThalamusStructuredLogger

_logger = ThalamusStructuredLogger.get_logger("tool-parser", "DEBUG")


def extract_balanced_text(
    text: str,
    start_idx: int,
    open_char: str,
    close_char: str,
) -> str | None:
    """Extract balanced delimited text (handles nested braces/brackets)."""
    if not text or start_idx < 0 or start_idx >= len(text) or text[start_idx] != open_char:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start_idx, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == open_char:
            depth += 1
            continue
        if ch == close_char:
            depth -= 1
            if depth == 0:
                return text[start_idx : i + 1]
    return None


def relaxed_json_parse(text: str) -> Any:
    """Try json.loads first. If it fails, apply progressive fixups and retry."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    sanitized = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError:
        pass
    # Fix "key">value  ->  "key":value  (model sometimes emits > instead of :)
    gt_fixed = re.sub(r'">\s*(?="|\{|\[|true|false|null|-?\d)', '":', text)
    if gt_fixed != text:
        gt_fixed = re.sub(r",\s*([}\]])", r"\1", gt_fixed)
        try:
            return json.loads(gt_fixed)
        except json.JSONDecodeError:
            pass
    # Fix "key">unquoted text"  ->  "key":"unquoted text"
    # Handles cases where > replaces : before a value that ends with "
    gt_unquoted = re.sub(r'">([^"]*?)"', lambda m: '":"' + m.group(1) + '"', text)
    if gt_unquoted != text:
        gt_unquoted = re.sub(r",\s*([}\]])", r"\1", gt_unquoted)
        try:
            return json.loads(gt_unquoted)
        except json.JSONDecodeError:
            pass
    replaced = ""
    for ch in text:
        if ord(ch) <= 0x1F:
            if ch == "\n":
                replaced += "\\n"
            elif ch == "\r":
                replaced += "\\r"
            elif ch == "\t":
                replaced += "\\t"
        else:
            replaced += ch
    try:
        return json.loads(replaced)
    except json.JSONDecodeError:
        raise


def _raw_decode_json(text: str, start_index: int) -> tuple[Any, int] | None:
    """Extract the first complete JSON value starting at start_index."""
    if not text or start_index < 0 or start_index >= len(text):
        return None
    open_char = text[start_index]
    if open_char not in ("{", "["):
        return None
    close_char = "}" if open_char == "{" else "]"
    balanced = extract_balanced_text(text, start_index, open_char, close_char)
    if balanced:
        try:
            return relaxed_json_parse(balanced), start_index + len(balanced) - 1
        except (json.JSONDecodeError, ValueError):
            pass
        candidate = balanced
        for _ in range(8):
            if not candidate:
                break
            last = candidate[-1]
            if last in ("}", "]", ","):
                candidate = candidate[:-1]
                try:
                    return relaxed_json_parse(candidate), start_index + len(candidate) - 1
                except (json.JSONDecodeError, ValueError):
                    pass
            else:
                break
    depth = 0
    in_str = False
    esc = False
    for i in range(start_index, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == open_char:
            depth += 1
            continue
        if ch == close_char:
            depth -= 1
            if depth == 0:
                slice_text = text[start_index : i + 1]
                try:
                    return relaxed_json_parse(slice_text), i
                except (json.JSONDecodeError, ValueError):
                    pass
    return None


def extract_all_json_objects(text: str) -> list[dict[str, Any]]:
    """Find all top-level JSON objects in text by scanning for '{' and extracting balanced braces."""
    if not text:
        return []
    results: list[dict[str, Any]] = []
    last_end = -1
    for i in range(len(text)):
        if text[i] != "{":
            continue
        if i <= last_end:
            continue
        decoded = _raw_decode_json(text, i)
        if not decoded:
            continue
        value, end_idx = decoded
        raw = text[i : end_idx + 1]
        results.append({"obj": value, "raw": raw, "start": i})
        last_end = end_idx
    return results


def serialize_tool_arguments(raw_arguments: Any) -> str:
    """Serialize tool arguments to JSON string."""
    if raw_arguments is None:
        return "{}"
    if isinstance(raw_arguments, str):
        trimmed = raw_arguments.strip()
        return trimmed if trimmed else "{}"
    try:
        return json.dumps(raw_arguments)
    except (TypeError, ValueError):
        return "{}"


def is_tool_call_shape(obj: Any) -> bool:
    """Check if obj looks like a tool call: has function.name+arguments, or has name+arguments."""
    if not obj or not isinstance(obj, dict):
        return False
    fn = obj.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str) and "arguments" in fn:
        return True
    if isinstance(fn, str) and fn.strip() and "arguments" in obj:
        return True
    if isinstance(obj.get("name"), str) and "arguments" in obj:
        return True
    return False


def _is_structured_tool_call_array(value: Any) -> bool:
    if not isinstance(value, list) or len(value) == 0:
        return False
    for entry in value:
        if not entry or not isinstance(entry, dict):
            return False
        fn = entry.get("function")
        if isinstance(fn, dict):
            fn_name = fn.get("name", "")
        elif isinstance(fn, str) and fn.strip():
            fn_name = fn
        else:
            fn_name = entry.get("name", "")
        if not isinstance(fn_name, str) or not fn_name.strip():
            return False
    return True


def normalize_tool_calls(tool_calls: list) -> list[dict]:
    """Normalize various formats to standard tool call shape."""
    result: list[dict] = []
    for tc in tool_calls:
        fn = tc.get("function") if isinstance(tc, dict) and isinstance(tc.get("function"), dict) else {}
        fn_is_string = isinstance(tc.get("function"), str) if isinstance(tc, dict) else False
        tool_name = (
            fn.get("name")
            or (tc.get("function") if fn_is_string and isinstance(tc, dict) else "")
            or (tc.get("name") if isinstance(tc, dict) else "")
            or ""
        )
        raw_args = (
            fn.get("arguments")
            if "arguments" in fn
            else (tc.get("arguments") if isinstance(tc, dict) else None)
        )
        result.append(
            {
                "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": serialize_tool_arguments(raw_args),
                },
                "_validation_arguments": raw_args,
            }
        )
    return result


def _parse_tool_calls_json_candidate(raw_text: str) -> list[dict] | None:
    """Parse tool calls from a raw JSON/text candidate."""
    if not raw_text:
        return None
    try:
        parsed = relaxed_json_parse(raw_text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    if parsed and isinstance(parsed.get("tool_calls"), list):
        return normalize_tool_calls(parsed["tool_calls"])
    fc = parsed.get("function_call") if isinstance(parsed, dict) else None
    if (
        fc
        and isinstance(fc, dict)
        and isinstance(fc.get("name"), str)
        and "arguments" in fc
    ):
        return normalize_tool_calls([{"function": fc}])
    if _is_structured_tool_call_array(parsed):
        return normalize_tool_calls(parsed)
    if parsed and isinstance(parsed, dict):
        fn = parsed.get("function")
        has_fn_shape = (
            isinstance(fn, dict)
            and isinstance(fn.get("name"), str)
            and "arguments" in fn
        )
        has_flat_shape = isinstance(parsed.get("name"), str) and "arguments" in parsed
        if has_fn_shape or has_flat_shape:
            return normalize_tool_calls([parsed])
    return None


def parse_tool_calls_from_direct_json(text: str) -> list[dict] | None:
    """Look for {"tool_calls":[...]} wrapper or direct JSON."""
    return _parse_tool_calls_json_candidate(text.strip() if text else "")


def parse_tool_calls_from_code_block(text: str) -> list[dict] | None:
    """Look for ```json ... ``` blocks containing tool_calls."""
    if not text:
        return None
    trimmed = text.strip()
    for m in re.finditer(r"```(?:[a-zA-Z0-9_-]+)?\s*([\s\S]*?)```", trimmed, re.I):
        candidate = (m.group(1) or "").strip()
        if not candidate:
            continue
        parsed = _parse_tool_calls_json_candidate(candidate)
        if parsed:
            return parsed
    return None


def parse_tool_calls_via_raw_decode(text: str) -> list[dict] | None:
    """Extract all JSON objects, find tool_calls arrays, function_call objects, or bare {name, arguments}."""
    if not text or "{" not in text:
        return None
    trimmed = text.strip()
    json_objects = extract_all_json_objects(trimmed)
    if not json_objects:
        return None
    for item in json_objects:
        obj = item["obj"]
        if obj and isinstance(obj.get("tool_calls"), list):
            normalized = normalize_tool_calls(obj["tool_calls"])
            if normalized:
                return normalized
    for item in json_objects:
        obj = item["obj"]
        fc = obj.get("function_call") if isinstance(obj, dict) else None
        if fc and isinstance(fc, dict) and isinstance(fc.get("name"), str):
            return normalize_tool_calls([{"function": fc}])
    bare_calls = [item["obj"] for item in json_objects if is_tool_call_shape(item["obj"])]
    if bare_calls:
        return normalize_tool_calls(bare_calls)
    return None


def parse_tool_calls_from_embedded_array(text: str) -> list[dict] | None:
    """Find 'tool_calls' key and extract the array."""
    if not text:
        return None
    trimmed = text.strip()
    idx = 0
    while True:
        marker_idx = trimmed.find('"tool_calls"', idx)
        if marker_idx < 0:
            break
        array_start = trimmed.find("[", marker_idx)
        if array_start >= 0:
            candidate = extract_balanced_text(trimmed, array_start, "[", "]")
            if candidate:
                try:
                    arr = relaxed_json_parse(candidate)
                    if _is_structured_tool_call_array(arr):
                        return normalize_tool_calls(arr)
                except (json.JSONDecodeError, ValueError):
                    pass
        idx = marker_idx + 1
    return None


def parse_tool_calls_from_embedded_object(text: str) -> list[dict] | None:
    """Find 'tool_calls' in a parent object and parse it."""
    if not text:
        return None
    trimmed = text.strip()
    idx = 0
    while True:
        marker_idx = trimmed.find('"tool_calls"', idx)
        if marker_idx < 0:
            break
        attempts = 0
        start_idx = trimmed.rfind("{", 0, marker_idx + 1)
        while start_idx >= 0 and attempts < 32:
            candidate = extract_balanced_text(trimmed, start_idx, "{", "}")
            if candidate and '"tool_calls"' in candidate:
                parsed = _parse_tool_calls_json_candidate(candidate)
                if parsed:
                    return parsed
            start_idx = trimmed.rfind("{", 0, start_idx)
            attempts += 1
        idx = marker_idx + 1
    return None


def _try_recover_gt_corruption(text: str) -> Any | None:
    """Recover JSON where '>' was used instead of ':' before values.

    Handles patterns where the model emits "key">value instead of "key":value.
    The value may contain raw newlines, braces, and other chars that need escaping.
    """
    if '">' not in text:
        return None

    fixed = text
    for m in re.finditer(r'"(\w+)">', fixed):
        gt_pos = m.end() - 1
        # Replace "> with ":"
        fixed = fixed[:gt_pos] + ':"' + fixed[gt_pos + 1:]

    if fixed == text:
        return None

    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)

    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # The value may contain raw control characters (newlines, tabs) that need escaping.
    # Find each "key":"value" pair where value has control chars and escape them.
    def _escape_control_in_values(s: str) -> str:
        result = []
        i = 0
        in_string = False
        escaped = False
        while i < len(s):
            ch = s[i]
            if in_string:
                if escaped:
                    escaped = False
                    result.append(ch)
                elif ch == '\\':
                    escaped = True
                    result.append(ch)
                elif ch == '"':
                    in_string = False
                    result.append(ch)
                elif ch == '\n':
                    result.append('\\n')
                elif ch == '\r':
                    result.append('\\r')
                elif ch == '\t':
                    result.append('\\t')
                elif ord(ch) < 0x20:
                    result.append(f'\\u{ord(ch):04x}')
                else:
                    result.append(ch)
            else:
                if ch == '"':
                    in_string = True
                result.append(ch)
            i += 1
        return ''.join(result)

    escaped_fixed = _escape_control_in_values(fixed)
    try:
        return json.loads(escaped_fixed)
    except (json.JSONDecodeError, ValueError):
        return None


def parse_tool_calls_from_malformed_payload(text: str) -> list[dict] | None:
    """Recover from malformed JSON by finding 'function' keys."""
    if not text or '"tool_calls"' not in text:
        return None
    trimmed = text.strip()
    recovered: list[dict] = []
    search_idx = 0
    while True:
        fn_key_idx = trimmed.find('"function"', search_idx)
        if fn_key_idx < 0:
            break
        obj_start = trimmed.find("{", fn_key_idx)
        if obj_start < 0:
            break
        fn_obj_text = extract_balanced_text(trimmed, obj_start, "{", "}")
        if not fn_obj_text:
            break
        try:
            fn_obj = relaxed_json_parse(fn_obj_text)
            if (
                fn_obj
                and isinstance(fn_obj, dict)
                and isinstance(fn_obj.get("name"), str)
                and "arguments" in fn_obj
            ):
                recovered.append({"function": fn_obj})
        except (json.JSONDecodeError, ValueError):
            pass
        search_idx = obj_start + len(fn_obj_text)

    if not recovered:
        gt_recovered = _try_recover_gt_corruption(trimmed)
        if gt_recovered and isinstance(gt_recovered, dict):
            tcs = gt_recovered.get("tool_calls")
            if isinstance(tcs, list) and tcs:
                return normalize_tool_calls(tcs)

    if not recovered:
        return None
    return normalize_tool_calls(recovered)


def parse_tool_calls_from_xml_tags(text: str) -> list[dict] | None:
    """Parse <tool_call>{...}</tool_call> format."""
    if not text or "<tool_call>" not in text.lower():
        return None
    trimmed = text.strip()
    recovered: list[dict] = []
    for m in re.finditer(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", trimmed, re.I):
        candidate = (m.group(1) or "").strip()
        if not candidate:
            continue
        parsed = _parse_tool_calls_json_candidate(candidate)
        if parsed:
            recovered.extend(parsed)
    if not recovered:
        open_m = re.search(r"<tool_call>\s*([\s\S]*)$", trimmed, re.I)
        if open_m and open_m.group(1):
            parsed = _parse_tool_calls_json_candidate(open_m.group(1).strip())
            if parsed:
                recovered.extend(parsed)
    return recovered if recovered else None


def parse_tool_calls_from_llama_tags(text: str) -> list[dict] | None:
    """Parse <<function=Name>>{...}<</function>> format."""
    if not text or "<<function=" not in text.lower():
        return None
    trimmed = text.strip()
    recovered: list[dict] = []
    for m in re.finditer(
        r"<<function=([a-zA-Z_][a-zA-Z0-9_.:-]*)>>\s*([\s\S]*?)\s*<<\/function>>",
        trimmed,
        re.I,
    ):
        tool_name = (m.group(1) or "").strip()
        args_text = (m.group(2) or "").strip()
        if not tool_name or not args_text:
            continue
        try:
            parsed_args = json.loads(args_text)
            if not isinstance(parsed_args, dict) or isinstance(parsed_args, list):
                continue
            recovered.extend(
                normalize_tool_calls(
                    [{"function": {"name": tool_name, "arguments": parsed_args}}]
                )
            )
        except json.JSONDecodeError:
            pass
    return recovered if recovered else None


def _legacy_parse_function_object(text: str) -> list[dict] | None:
    """Legacy heuristic: regex for function object pattern."""
    m = re.search(
        r'\{\s*"?(?:function|name)"?\s*:\s*\{?\s*"?name"?\s*:\s*"([^"]+)"\s*,\s*"?arguments"?\s*:\s*(\{[\s\S]*?\})\s*\}?\s*\}',
        text,
    )
    if m:
        return [
            {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": m.group(1), "arguments": m.group(2)},
            }
        ]
    return None


def _legacy_parse_name_arguments(text: str) -> list[dict] | None:
    """Legacy heuristic: regex for separate name + arguments fields."""
    name_m = re.search(
        r'"?(?:name|function_name|tool)"?\s*:\s*"([a-zA-Z_][a-zA-Z0-9_]*)"', text
    )
    args_m = re.search(r'"?arguments"?\s*:\s*(\{[^}]*\})', text)
    if name_m and args_m:
        try:
            args = json.loads(args_m.group(1))
            return [
                {
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": name_m.group(1),
                        "arguments": json.dumps(args),
                    },
                }
            ]
        except json.JSONDecodeError:
            pass
    return None


def _legacy_parse_natural_language(text: str) -> list[dict] | None:
    """Legacy heuristic: natural language pattern call/use/execute ToolName with arguments {...}."""
    m = re.search(
        r"(?:call|use|execute)\s+(?:the\s+)?(?:tool\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:with|using)?\s*(?:arguments?|params?)?\s*[:=]?\s*(\{[^}]+\})?",
        text,
        re.I,
    )
    if m:
        tool_name = m.group(1)
        args = "{}"
        if m.group(2):
            try:
                json.loads(m.group(2))
                args = m.group(2)
            except json.JSONDecodeError:
                pass
        return [
            {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": tool_name, "arguments": args},
            }
        ]
    return None


KNOWN_TOOL_NAMES = {
    "Read", "Write", "Edit", "MultiEdit", "Bash", "Glob", "Grep",
    "WebFetch", "WebSearch", "TodoRead", "TodoWrite", "Task",
}


def _parse_text_mimicry(text: str) -> list[dict] | None:
    """Catch ANY text that mimics a tool call — regardless of surrounding format.

    Patterns caught:
      [Called Tool: X] ... Arguments: {...}
      <tool_executed name="X" input={...} />
      (Executed X with {...})
      X({"key": "val"})
    """
    results: list[dict] = []

    patterns = [
        # [Called Tool: X] (id=...) Arguments: {...}
        re.compile(
            r'\[Called Tool:\s*(\w+)\]\s*(?:\(id=[^)]*\))?\s*(?:Arguments?:\s*)',
            re.DOTALL,
        ),
        # <tool_executed name="X" input={...} />
        re.compile(
            r'<tool_executed\s+name="(\w+)"\s+input=',
            re.DOTALL,
        ),
        # (Executed X with {...})
        re.compile(
            r'\(Executed\s+(\w+)\s+with\s+',
            re.DOTALL,
        ),
    ]

    for pat in patterns:
        for m in pat.finditer(text):
            tool_name = m.group(1)
            if tool_name not in KNOWN_TOOL_NAMES:
                continue
            rest = text[m.end():]
            brace_idx = rest.find("{")
            if brace_idx < 0 or brace_idx > 10:
                continue
            balanced = extract_balanced_text(rest, brace_idx, "{", "}")
            if not balanced:
                continue
            try:
                args_obj = json.loads(balanced)
                if isinstance(args_obj, dict):
                    results.append({
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(args_obj),
                        },
                    })
            except json.JSONDecodeError:
                pass

    if results:
        return results

    # Fallback: ToolName followed by JSON anywhere in text
    for name in KNOWN_TOOL_NAMES:
        for m in re.finditer(re.escape(name) + r'\s*[\(\{:]?\s*', text):
            rest = text[m.end():]
            brace_idx = rest.find("{")
            if brace_idx < 0 or brace_idx > 5:
                continue
            balanced = extract_balanced_text(rest, brace_idx, "{", "}")
            if not balanced:
                continue
            try:
                args_obj = json.loads(balanced)
                if isinstance(args_obj, dict) and args_obj:
                    results.append({
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args_obj),
                        },
                    })
            except json.JSONDecodeError:
                pass

    return results if results else None


def parse_anthropic_tool_use_blocks(text: str) -> list[dict] | None:
    """PRIMARY strategy: extract Anthropic native tool_use JSON blocks.

    Looks for {"type":"tool_use", ...} JSON objects in the text.
    This is the expected output format when the model has learned from
    consistent Anthropic JSON examples in conversation history.
    """
    if not text or '"tool_use"' not in text:
        return None

    results: list[dict] = []
    json_objects = extract_all_json_objects(text)

    for item in json_objects:
        obj = item["obj"]
        if not isinstance(obj, dict):
            continue
        if obj.get("type") != "tool_use":
            continue
        name = obj.get("name", "")
        if not isinstance(name, str) or not name.strip():
            continue
        raw_input = obj.get("input") if "input" in obj else None
        results.append({
            "id": obj.get("id") or f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(raw_input) if not isinstance(raw_input, str) else raw_input,
            },
            "_validation_arguments": raw_input,
        })

    return results if results else None


def try_parse_tool_calls_from_text(
    text: str,
    strict_json_only: bool = True,
) -> list[dict] | None:
    """
    Extract tool calls from LLM text output.
    Returns a list of normalized tool call dicts or None if no tool calls found.

    Strategy order:
      1. anthropic_tool_use  — native Anthropic JSON (PRIMARY, expected format)
      2. text_mimicry        — catch text that mimics tool calls
      3. direct_json         — {"tool_calls":[...]} (legacy)
      4. code_block          — ```json ... ``` blocks (legacy)
      5-8. other fallbacks   — various recovery strategies
    """
    if not text:
        return None
    trimmed = text.strip()
    preview = trimmed[:180]
    snippet = re.sub(r"\s+", " ", trimmed[:160])
    _logger.debug(f"Parsing text len={len(trimmed)} preview={preview}")

    # PRIMARY: Anthropic native tool_use JSON
    anthropic_result = parse_anthropic_tool_use_blocks(trimmed)
    if anthropic_result:
        _logger.info(
            f"parse_strategy=anthropic_tool_use raw_len={len(trimmed)} "
            f"tool_count={len(anthropic_result)} candidate_snippet={snippet}"
        )
        return anthropic_result

    mimicry_result = _parse_text_mimicry(trimmed)
    if mimicry_result:
        _logger.info(
            f"parse_strategy=text_mimicry_intercept raw_len={len(trimmed)} candidate_snippet={snippet}"
        )
        return mimicry_result

    strategies: list[tuple[str, callable]] = [
        ("direct_json", parse_tool_calls_from_direct_json),
        ("code_block", parse_tool_calls_from_code_block),
        ("raw_decode", parse_tool_calls_via_raw_decode),
        ("embedded_array", parse_tool_calls_from_embedded_array),
        ("embedded_object", parse_tool_calls_from_embedded_object),
        ("malformed_payload", parse_tool_calls_from_malformed_payload),
        ("xml_tags", parse_tool_calls_from_xml_tags),
        ("llama_tags", parse_tool_calls_from_llama_tags),
    ]

    for name, parser in strategies:
        result = parser(trimmed)
        if result:
            _logger.info(
                f"parse_strategy=strict_json:{name} raw_len={len(trimmed)} candidate_snippet={snippet}"
            )
            return result

    if strict_json_only:
        _logger.info(
            f"parse_strategy=strict_json:none raw_len={len(trimmed)} candidate_snippet={snippet}"
        )
        _logger.debug(
            "strict_json_only=True and no valid JSON tool_calls found; skip heuristic parsing"
        )
        return None

    legacy_strategies: list[tuple[str, callable]] = [
        ("legacy_function_object", _legacy_parse_function_object),
        ("legacy_name_arguments", _legacy_parse_name_arguments),
        ("legacy_natural_language", _legacy_parse_natural_language),
    ]

    for name, parser in legacy_strategies:
        result = parser(trimmed)
        if result:
            _logger.info(
                f"parse_strategy=legacy_heuristic:{name} raw_len={len(trimmed)} candidate_snippet={snippet}"
            )
            return result

    _logger.info(
        f"parse_strategy=legacy_heuristic:none raw_len={len(trimmed)} candidate_snippet={snippet}"
    )
    return None
