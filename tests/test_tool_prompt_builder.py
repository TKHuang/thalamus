"""Regression tests for tool prompt construction."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code.tool_prompt_builder import (  # noqa: E402
    build_tool_call_prompt,
    inject_tool_prompt_into_messages,
)
from claude_code.standard_tool_protocol import StandardJsonV1Adapter  # noqa: E402
from config.system_prompt import THALAMUS_INSTRUCTION_SUPPLEMENT  # noqa: E402


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


def test_tool_manifest_is_deterministic_and_contains_no_proxy_strategy():
    prompt = build_tool_call_prompt(_hermes_tools(), compact=True)
    second = build_tool_call_prompt(_hermes_tools(), compact=False)

    assert prompt == second
    assert '\"name\":\"write_file\"' in prompt
    assert '\"name\":\"skill_view\"' in prompt
    assert "task_complete" not in prompt
    assert "write_file first" not in prompt
    assert "8000" not in prompt
    assert "TOOL_REMINDER" not in prompt


def test_tool_manifest_is_injected_once_into_system_without_fake_turns():
    denial = "目前會話沒有終端能力。"
    injected = inject_tool_prompt_into_messages(
        [
            {"role": "system", "content": "caller instruction"},
            {"role": "user", "content": "make a file"},
            {"role": "assistant", "content": denial},
        ],
        _hermes_tools(),
        compact_tools=True,
        adapter=StandardJsonV1Adapter(),
    )

    assert [message["role"] for message in injected] == ["system", "user", "assistant"]
    assert injected[0]["content"].count("Available client tools") == 1
    assert "caller instruction" in injected[0]["content"]
    assert injected[-1]["content"] == denial
    serialized = str(injected)
    assert "tools noted" not in serialized
    assert "TOOL_REMINDER" not in serialized
    assert "task_complete" not in serialized


def test_standard_manifest_exposes_schemas_without_requesting_json_protocol():
    injected = inject_tool_prompt_into_messages(
        [{"role": "user", "content": "make a file"}],
        _hermes_tools(),
        adapter=StandardJsonV1Adapter(),
    )

    prompt = injected[0]["content"]
    assert "exposed through native function calling" in prompt
    assert '"type":"tool_use"' not in prompt
    assert '"write_file"' in prompt
    assert "long browser automation description" in prompt


def test_tool_choice_is_translated_from_structured_value_only():
    required = inject_tool_prompt_into_messages(
        [{"role": "system", "content": "caller"}],
        _hermes_tools(),
        adapter=StandardJsonV1Adapter(),
        tool_choice="required",
    )
    specific = inject_tool_prompt_into_messages(
        [{"role": "system", "content": "caller"}],
        [_hermes_tools()[1]],
        adapter=StandardJsonV1Adapter(),
        tool_choice={"type": "function", "function": {"name": "write_file"}},
    )

    assert "Tool choice policy: required" in required[0]["content"]
    assert "Tool choice policy: specific" in specific[0]["content"]
    assert "write_file" in specific[0]["content"]


def test_runtime_prompt_injections_are_harness_agnostic():
    assert "Claude Code" not in THALAMUS_INSTRUCTION_SUPPLEMENT
    assert "Hermes" not in THALAMUS_INSTRUCTION_SUPPLEMENT
    assert "Cursor" not in THALAMUS_INSTRUCTION_SUPPLEMENT

    assert "Do not assume a specific client or harness identity" in THALAMUS_INSTRUCTION_SUPPLEMENT
    assert "client-advertised tool inventory" in THALAMUS_INSTRUCTION_SUPPLEMENT
    assert "call it in the same response" in THALAMUS_INSTRUCTION_SUPPLEMENT
    assert "filesystem path supplied by the caller" in THALAMUS_INSTRUCTION_SUPPLEMENT
    assert "verify the result before declaring completion" in THALAMUS_INSTRUCTION_SUPPLEMENT
    assert "Do not claim successful verification unless it" in THALAMUS_INSTRUCTION_SUPPLEMENT
    assert "identify only as an AI assistant" in THALAMUS_INSTRUCTION_SUPPLEMENT
    assert "product, host, client, provider, company, IDE, CLI, framework, or router" in THALAMUS_INSTRUCTION_SUPPLEMENT


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
