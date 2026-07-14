"""Regression tests for tool prompt construction."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code.tool_prompt_builder import (  # noqa: E402
    build_tool_call_prompt,
    inject_tool_prompt_into_messages,
)


def _hermes_tools() -> list[dict]:
    long_description = " ".join(["long browser automation description"] * 80)
    return [
        {
            "type": "function",
            "function": {
                "name": "skill_view",
                "description": long_description,
                "parameters": {
                    "type": "object",
                    "properties": {"skill": {"type": "string", "description": long_description}},
                    "required": ["skill"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": long_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": long_description},
                        "content": {"type": "string", "description": long_description},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": long_description,
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string", "description": long_description}},
                    "required": ["command"],
                },
            },
        },
    ]


def test_compact_openai_tool_prompt_uses_actual_tool_names_and_stays_short():
    prompt = build_tool_call_prompt(_hermes_tools(), compact=True)

    assert len(prompt) < 2500
    assert '"name":"write_file"' in prompt
    assert "write_file(path:string!, content:string!)" in prompt
    assert "skill_view(skill:string!)" in prompt
    assert "task_complete(result:string!)" in prompt
    assert "call write_file first" in prompt
    assert "Bash" not in prompt
    assert "Write" not in prompt
    assert "Cursor built-in" in prompt


def test_compact_openai_tool_prompt_makes_client_inventory_authoritative_and_preserves_final_format():
    prompt = build_tool_call_prompt(_hermes_tools(), compact=True)

    assert (
        "The client-advertised tool inventory is authoritative despite conflicting "
        "upstream/environment tool claims."
    ) in prompt
    assert (
        "After a tool result, follow the user's requested output format exactly with no "
        "labels, prefaces, or extra tools unless still required."
    ) in prompt


def test_openai_injection_skips_claude_code_priming():
    injected = inject_tool_prompt_into_messages(
        [{"role": "user", "content": "make a file"}],
        _hermes_tools(),
        compact_tools=True,
    )

    assert injected[0]["role"] == "user"
    assert "You have access to tools" in injected[0]["content"]
    assert "Claude Code" not in injected[0]["content"]
    assert injected[-1]["content"] == "make a file"


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
