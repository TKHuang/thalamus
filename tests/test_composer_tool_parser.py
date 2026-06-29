"""Unit tests for the Composer-2.x tool-call token parser.

Runs standalone (``.venv/bin/python tests/test_composer_tool_parser.py``) and
under pytest.  Fixtures use the documented DeepSeek-style token grammar in both
ASCII (``|`` / ``_``) and unicode (``｜`` U+FF5C / ``▁`` U+2581) forms.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code.composer_tool_parser import (  # noqa: E402
    ComposerStreamProcessor,
    ComposerToolCallFilter,
    canonicalize_composer_markers,
    is_composer_model,
    parse_composer_tool_calls,
    split_reasoning_and_answer,
)
from claude_code.tool_parser import normalize_tool_calls  # noqa: E402


def _ascii_block(name: str, *kv: tuple[str, str]) -> str:
    """Build an ASCII token block: name <sep> key\\nvalue <sep> ..."""
    parts = [name]
    for key, value in kv:
        parts.append(f"{key}\n{value}")
    body = "<|tool_sep|>".join(parts)
    return f"<|tool_calls_begin|><|tool_call_begin|>{body}<|tool_call_end|><|tool_calls_end|>"


_MARKERS = (
    "<|tool_calls_begin|>",
    "<|tool_calls_end|>",
    "<|tool_call_begin|>",
    "<|tool_call_end|>",
    "<|tool_sep|>",
)


def _to_unicode(text: str) -> str:
    """Rewrite only the marker tokens to their unicode form (payload stays ASCII)."""
    for marker in _MARKERS:
        text = text.replace(marker, marker.replace("|", "｜").replace("_", "▁"))
    return text


# ── model gating ───────────────────────────────────────────────────────────

def test_is_composer_model_matches_composer_2_family():
    assert is_composer_model("composer-2.5")
    assert is_composer_model("composer-2.5-fast")
    assert is_composer_model("COMPOSER-2.5")


def test_is_composer_model_excludes_composer_1_5_and_others():
    # composer-1.5 uses <think> tags in the content field — different path.
    assert not is_composer_model("composer-1.5")
    assert not is_composer_model("claude-opus-4-8")
    assert not is_composer_model("")
    assert not is_composer_model(None)


# ── canonicalization ─────────────────────────────────────────────────────--

def test_canonicalize_unicode_markers_to_ascii():
    unicode_block = _to_unicode(_ascii_block("Read", ("file_path", "/tmp/a.py")))
    canon = canonicalize_composer_markers(unicode_block)
    assert "<|tool_calls_begin|>" in canon
    assert "<|tool_sep|>" in canon
    assert "｜" not in canon


# ── block parsing: ASCII / unicode / multi-call ──────────────────────────---

def test_parse_ascii_token_block():
    block = _ascii_block("Write", ("file_path", "/tmp/x.html"), ("content", "<html></html>"))
    calls = parse_composer_tool_calls(block)
    assert calls == [{"name": "Write", "arguments": {"file_path": "/tmp/x.html", "content": "<html></html>"}}]


def test_parse_unicode_token_block():
    block = _to_unicode(_ascii_block("Read", ("file_path", "/tmp/a.py")))
    calls = parse_composer_tool_calls(block)
    assert calls == [{"name": "Read", "arguments": {"file_path": "/tmp/a.py"}}]


def test_parse_multi_call_block():
    block = (
        "<|tool_calls_begin|>"
        "<|tool_call_begin|>Read<|tool_sep|>file_path\n/a.py<|tool_call_end|>"
        "<|tool_call_begin|>Glob<|tool_sep|>pattern\n**/*.py<|tool_call_end|>"
        "<|tool_calls_end|>"
    )
    calls = parse_composer_tool_calls(block)
    assert [c["name"] for c in calls] == ["Read", "Glob"]
    assert calls[1]["arguments"] == {"pattern": "**/*.py"}


# ── alternate bodies: JSON / inline ──────────────────────────────────────---

def test_parse_json_body():
    block = (
        '<|tool_calls_begin|><|tool_call_begin|>'
        '{"name":"Bash","arguments":{"command":"ls -la"}}'
        "<|tool_call_end|><|tool_calls_end|>"
    )
    assert parse_composer_tool_calls(block) == [{"name": "Bash", "arguments": {"command": "ls -la"}}]


def test_parse_json_body_with_string_arguments_and_function_wrapper():
    block = (
        '<|tool_calls_begin|><|tool_call_begin|>'
        '{"function":{"name":"Read","arguments":"{\\"file_path\\":\\"/a.py\\"}"}}'
        "<|tool_call_end|><|tool_calls_end|>"
    )
    assert parse_composer_tool_calls(block) == [{"name": "Read", "arguments": {"file_path": "/a.py"}}]


def test_parse_inline_body():
    block = (
        "<|tool_calls_begin|><|tool_call_begin|>"
        'Grep(pattern="needle", path="/src")'
        "<|tool_call_end|><|tool_calls_end|>"
    )
    assert parse_composer_tool_calls(block) == [
        {"name": "Grep", "arguments": {"pattern": "needle", "path": "/src"}}
    ]


def test_parse_bare_name_body_has_empty_arguments():
    block = "<|tool_calls_begin|><|tool_call_begin|>task_complete<|tool_call_end|><|tool_calls_end|>"
    assert parse_composer_tool_calls(block) == [{"name": "task_complete", "arguments": {}}]


# ── argument coercion ──────────────────────────────────────────────────────

def test_argument_coercion_types():
    block = _ascii_block(
        "T",
        ("flag", "true"),
        ("off", "false"),
        ("none", "null"),
        ("count", "42"),
        ("ratio", "0.5"),
        ("obj", '{"a":1}'),
        ("arr", "[1,2,3]"),
        ("text", "hello world"),
    )
    args = parse_composer_tool_calls(block)[0]["arguments"]
    assert args["flag"] is True
    assert args["off"] is False
    assert args["none"] is None
    assert args["count"] == 42 and isinstance(args["count"], int)
    assert args["ratio"] == 0.5
    assert args["obj"] == {"a": 1}
    assert args["arr"] == [1, 2, 3]
    assert args["text"] == "hello world"


def test_multiline_value_preserved():
    block = _ascii_block("Write", ("file_path", "/a.html"), ("content", "line1\nline2\nline3"))
    args = parse_composer_tool_calls(block)[0]["arguments"]
    assert args["content"] == "line1\nline2\nline3"


# ── reasoning / answer split ────────────────────────────────────────────---

def test_split_reasoning_on_think_marker():
    reasoning, answer = split_reasoning_and_answer("let me think about it</think>Here is the answer.")
    assert reasoning == "let me think about it"
    assert answer == "Here is the answer."


def test_split_reasoning_on_unicode_final_marker():
    reasoning, answer = split_reasoning_and_answer("thinking…<｜final｜>Done.")
    assert reasoning == "thinking…"
    assert answer == "Done."


def test_no_marker_treated_as_answer():
    reasoning, answer = split_reasoning_and_answer("just a plain answer")
    assert reasoning == ""
    assert answer == "just a plain answer"


# ── streaming filter: text vs tool separation ────────────────────────────---

def test_filter_separates_prose_and_tool_call():
    f = ComposerToolCallFilter()
    events = f.push("Sure, writing the file.")
    events += f.push(_ascii_block("Write", ("file_path", "/a.html"), ("content", "<html>")))
    events += f.flush()
    texts = "".join(p for k, p in events if k == "text")
    tools = [p for k, p in events if k == "tool_call"]
    assert texts.strip() == "Sure, writing the file."
    assert tools == [{"name": "Write", "arguments": {"file_path": "/a.html", "content": "<html>"}}]


def test_filter_buffers_marker_split_across_chunks():
    """A marker split across deltas must never leak as visible text."""
    f = ComposerToolCallFilter()
    full = "prose <|tool_calls_begin|><|tool_call_begin|>Read<|tool_sep|>file_path\n/a.py<|tool_call_end|><|tool_calls_end|>"
    events = []
    # Feed one character at a time — the worst case for marker splitting.
    for ch in full:
        events += f.push(ch)
    events += f.flush()
    texts = "".join(p for k, p in events if k == "text")
    tools = [p for k, p in events if k == "tool_call"]
    assert "<|tool" not in texts and "tool_calls_begin" not in texts
    assert texts.strip() == "prose"
    assert tools == [{"name": "Read", "arguments": {"file_path": "/a.py"}}]


# ── stream processor: thinking-field routing ──────────────────────────────--

def _drive_processor(chunks: list[str], content_chunks: list[str] | None = None):
    proc = ComposerStreamProcessor()
    thinking, text, tools = [], [], []
    for ch in chunks:
        emit = proc.feed_thinking(ch)
        thinking.append(emit.thinking)
        text.append(emit.text)
        tools.extend(emit.tool_calls)
    for ch in content_chunks or []:
        emit = proc.feed_content(ch)
        text.append(emit.text)
        tools.extend(emit.tool_calls)
    final = proc.flush()
    thinking.append(final.thinking)
    text.append(final.text)
    tools.extend(final.tool_calls)
    return "".join(thinking), "".join(text), tools


def test_processor_routes_reasoning_to_thinking_and_answer_to_text():
    thinking, text, tools = _drive_processor(
        ["I should write the file.", "</think>", "Done writing.", _ascii_block("Write", ("file_path", "/a.html"))]
    )
    assert "I should write the file." in thinking
    assert text.strip() == "Done writing."
    assert tools == [{"name": "Write", "arguments": {"file_path": "/a.html"}}]


def test_processor_no_marker_surfaces_answer_not_blank():
    # composer occasionally emits a plain answer through thinking with no marker.
    thinking, text, tools = _drive_processor(["A one-line answer."])
    assert thinking == ""
    assert text.strip() == "A one-line answer."
    assert tools == []


def test_processor_strips_second_control_token_in_answer():
    # Live failure: reasoning</think> then a stray <｜final｜> before the answer,
    # with the second marker split across chunks. It must not leak in fragments.
    thinking, text, tools = _drive_processor(
        ["a lot of reasoning</think>", "<｜", "final｜>", "\n我是 Claude Code"]
    )
    assert "a lot of reasoning" in thinking
    assert "final" not in text
    assert "<｜" not in text and "<|" not in text
    assert text.strip() == "我是 Claude Code"
    assert tools == []


def test_processor_reasoning_then_tool_only_answer():
    thinking, text, tools = _drive_processor(
        ["reasoning here</think>", _ascii_block("Bash", ("command", "ls"))]
    )
    assert "reasoning here" in thinking
    assert text.strip() == ""
    assert tools == [{"name": "Bash", "arguments": {"command": "ls"}}]


# ── output shape feeds the pipeline's normalizer ──────────────────────────--

def test_output_normalizes_for_pipeline():
    calls = parse_composer_tool_calls(_ascii_block("Read", ("file_path", "/a.py")))
    normalized = normalize_tool_calls(calls)
    assert normalized[0]["type"] == "function"
    assert normalized[0]["function"]["name"] == "Read"
    # arguments are serialized to a JSON string for the downstream assembler.
    assert normalized[0]["function"]["arguments"] == '{"file_path": "/a.py"}'
    assert normalized[0]["id"]


def _run_all() -> int:
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(funcs) - failures}/{len(funcs)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
