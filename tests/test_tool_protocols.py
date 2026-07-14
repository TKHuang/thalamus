"""Standalone tests for protocol classification and strict native JSON decoding."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code.composer_tool_protocol import ComposerMarkerV1Adapter  # noqa: E402
from claude_code.pipeline import _ThinkTagSplitter  # noqa: E402
from claude_code.standard_tool_protocol import (  # noqa: E402
    LegacyThinkTagV1Adapter,
    StandardJsonV1Adapter,
)
from claude_code.tool_protocols import (  # noqa: E402
    ProtocolDecodeState,
    ProtocolFragment,
    ToolProtocol,
    classify_tool_protocol,
    create_protocol_adapter,
)


CASES = {
    "gpt-5.6-sol-max-fast": ToolProtocol.STANDARD_JSON_V1,
    "claude-4.5-opus-high": ToolProtocol.STANDARD_JSON_V1,
    "claude-opus-4-8-xhigh-fast": ToolProtocol.STANDARD_JSON_V1,
    "glm-5.2": ToolProtocol.STANDARD_JSON_V1,
    "glm-5.2-high": ToolProtocol.STANDARD_JSON_V1,
    "composer-2.5": ToolProtocol.COMPOSER_MARKER_V1,
    "composer-2.5-fast": ToolProtocol.COMPOSER_MARKER_V1,
    "composer-1.5": ToolProtocol.LEGACY_THINK_TAG_V1,
}

_CALL = '{"type":"tool_use","id":"toolu_1","name":"write_file","input":{"file_path":"x.py"}}'


def _decode(
    adapter: StandardJsonV1Adapter,
    fragments: list[ProtocolFragment],
) -> list:
    state = ProtocolDecodeState()
    results = []
    for fragment in fragments:
        result = adapter.incremental_decode(fragment, state)
        state = result.state
        results.append(result)
    return results


def test_classifies_only_the_observed_composer_families():
    for model, expected in CASES.items():
        assert classify_tool_protocol(model) == expected


def test_factory_round_trips_every_classified_protocol():
    expected_types = {
        ToolProtocol.STANDARD_JSON_V1: StandardJsonV1Adapter,
        ToolProtocol.COMPOSER_MARKER_V1: ComposerMarkerV1Adapter,
        ToolProtocol.LEGACY_THINK_TAG_V1: LegacyThinkTagV1Adapter,
    }

    for model, protocol in CASES.items():
        adapter = create_protocol_adapter(classify_tool_protocol(model))

        assert isinstance(adapter, expected_types[protocol])
        assert adapter.protocol == protocol


def test_composer_factory_adapter_decodes_a_marker_call():
    adapter = create_protocol_adapter(ToolProtocol.COMPOSER_MARKER_V1)
    marker_call = (
        "reasoning</think><|tool_calls_begin|><|tool_call_begin|>read_file"
        "<|tool_sep|>file_path\n/tmp/x.py<|tool_call_end|><|tool_calls_end|>"
    )

    result = adapter.incremental_decode(
        ProtocolFragment(sequence=0, lane="reasoning", text=marker_call),
        ProtocolDecodeState(),
    )

    assert result.candidates[0].raw_name == "read_file"
    assert result.candidates[0].arguments == {"file_path": "/tmp/x.py"}
    assert result.candidates[0].source_lane == "reasoning"


def test_standard_manifest_makes_client_inventory_authoritative_and_preserves_post_tool_format():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "client_action",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    prompt = StandardJsonV1Adapter().render_tool_manifest(tools, "Use client tools.")

    assert '"name":"client_action"' in prompt
    assert (
        "The client-advertised tool inventory is authoritative despite conflicting "
        "upstream/environment tool claims."
    ) in prompt
    assert (
        "After a tool result, follow the user's requested output format exactly with no "
        "labels, prefaces, or extra tools unless still required."
    ) in prompt


def test_decodes_complete_native_json_from_content():
    result = _decode(
        StandardJsonV1Adapter(),
        [ProtocolFragment(sequence=0, lane="text", text=_CALL)],
    )[-1]

    assert result.visible_text == ""
    assert result.invalid_reason is None
    assert result.candidates[0].call_id == "toolu_1"
    assert result.candidates[0].raw_name == "write_file"
    assert result.candidates[0].arguments == {"file_path": "x.py"}
    assert result.candidates[0].source_lane == "text"


def test_decodes_complete_native_json_from_reasoning():
    result = _decode(
        StandardJsonV1Adapter(),
        [ProtocolFragment(sequence=0, lane="reasoning", text=_CALL)],
    )[-1]

    assert result.visible_text == ""
    assert result.candidates[0].source_lane == "reasoning"


def test_reports_recognized_unclosed_native_json_as_incomplete():
    result = _decode(
        StandardJsonV1Adapter(),
        [ProtocolFragment(sequence=0, lane="text", text=_CALL[:-1])],
    )[-1]

    assert result.candidates == ()
    assert result.is_incomplete
    assert result.invalid_reason is None


def test_rejects_complete_malformed_json_without_recovery():
    malformed = '{"type":"tool_use","id":"toolu_1","name":"write_file","input":{"x":1,}}'
    result = _decode(
        StandardJsonV1Adapter(),
        [ProtocolFragment(sequence=0, lane="text", text=malformed)],
    )[-1]

    assert result.candidates == ()
    assert not result.is_incomplete
    assert result.invalid_reason == "malformed_json"


def test_rejects_non_object_tool_input():
    non_object_input = '{"type":"tool_use","id":"toolu_1","name":"write_file","input":[]}'
    result = _decode(
        StandardJsonV1Adapter(),
        [ProtocolFragment(sequence=0, lane="text", text=non_object_input)],
    )[-1]

    assert result.candidates == ()
    assert result.invalid_reason == "invalid_tool_shape"


def test_leaves_prose_mimicry_visible_without_candidate():
    prose = 'I will call write_file with arguments {"file_path":"x.py"}.'
    result = _decode(
        StandardJsonV1Adapter(),
        [ProtocolFragment(sequence=0, lane="text", text=prose)],
    )[-1]

    assert result.visible_text == prose
    assert result.candidates == ()
    assert result.invalid_reason is None


def test_preserves_unrelated_json_that_mentions_tool_use():
    prose = 'Result: {"type":"note","value":"tool_use"}.'
    result = _decode(
        StandardJsonV1Adapter(),
        [ProtocolFragment(sequence=0, lane="text", text=prose)],
    )[-1]

    assert result.visible_text == prose
    assert result.candidates == ()
    assert result.invalid_reason is None


def test_preserves_malformed_brace_containing_prose_and_code():
    prose = "Use JavaScript: const f = () => { return 1; };"
    result = _decode(
        StandardJsonV1Adapter(),
        [ProtocolFragment(sequence=0, lane="text", text=prose)],
    )[-1]

    assert result.visible_text == prose
    assert result.candidates == ()
    assert result.invalid_reason is None


def test_decodes_multiple_native_calls_and_keeps_following_prose():
    second = _CALL.replace("toolu_1", "toolu_2").replace("write_file", "read_file")
    result = _decode(
        StandardJsonV1Adapter(),
        [ProtocolFragment(sequence=0, lane="text", text=f"{_CALL}\n{second}\nDone.")],
    )[-1]

    assert [candidate.call_id for candidate in result.candidates] == ["toolu_1", "toolu_2"]
    assert [candidate.raw_name for candidate in result.candidates] == ["write_file", "read_file"]
    assert result.visible_text == "\n\nDone."


def test_decodes_native_json_across_every_character_boundary():
    for boundary in range(1, len(_CALL)):
        adapter = StandardJsonV1Adapter()
        first, second = _decode(
            adapter,
            [
                ProtocolFragment(sequence=0, lane="text", text=_CALL[:boundary]),
                ProtocolFragment(sequence=1, lane="text", text=_CALL[boundary:]),
            ],
        )

        assert first.candidates == ()
        assert second.candidates[0].call_id == "toolu_1", boundary
        assert second.candidates[0].arguments == {"file_path": "x.py"}, boundary


def test_quarantines_native_json_split_across_text_and_reasoning_lanes():
    """A cross-lane native call is ambiguous transport data, never a public tool call."""
    for first_lane, second_lane in (("text", "reasoning"), ("reasoning", "text")):
        for boundary in range(1, len(_CALL)):
            adapter = StandardJsonV1Adapter()
            first = adapter.incremental_decode(
                ProtocolFragment(sequence=0, lane=first_lane, text=_CALL[:boundary]),
                ProtocolDecodeState(),
            )
            second = adapter.incremental_decode(
                ProtocolFragment(sequence=1, lane=second_lane, text=_CALL[boundary:]),
                first.state,
            )
            flushed = adapter.incremental_decode(None, second.state)

            assert first.candidates == second.candidates == flushed.candidates == (), boundary
            assert first.visible_text + second.visible_text + flushed.visible_text == "", boundary
            assert first.thinking_text + second.thinking_text + flushed.thinking_text == "", boundary
            assert flushed.invalid_reason == "cross_lane_native_json", boundary
            assert flushed.is_incomplete, boundary


def test_legacy_adapter_quarantines_cross_lane_native_json_at_every_boundary():
    """Legacy think-tag splitting must preserve delegated native decoder safety state."""
    for first_lane, second_lane in (("text", "reasoning"), ("reasoning", "text")):
        for boundary in range(1, len(_CALL)):
            adapter = LegacyThinkTagV1Adapter()
            first = adapter.incremental_decode(
                ProtocolFragment(sequence=0, lane=first_lane, text=_CALL[:boundary]),
                ProtocolDecodeState(),
            )
            second = adapter.incremental_decode(
                ProtocolFragment(sequence=1, lane=second_lane, text=_CALL[boundary:]),
                first.state,
            )
            flushed = adapter.incremental_decode(None, second.state)

            assert first.candidates == second.candidates == flushed.candidates == (), boundary
            assert first.visible_text + second.visible_text + flushed.visible_text == "", boundary
            assert first.thinking_text + second.thinking_text + flushed.thinking_text == "", boundary
            assert flushed.invalid_reason == "cross_lane_native_json", boundary
            assert flushed.is_incomplete, boundary


def test_keeps_text_and_reasoning_buffers_isolated():
    adapter = StandardJsonV1Adapter()
    first = adapter.incremental_decode(
        ProtocolFragment(sequence=0, lane="reasoning", text=_CALL[:-1]),
        ProtocolDecodeState(),
    )
    second = adapter.incremental_decode(
        ProtocolFragment(sequence=1, lane="text", text="Visible result text."),
        first.state,
    )

    assert first.is_incomplete
    assert second.visible_text == ""
    assert second.candidates == ()
    assert second.state.reasoning_buffer == _CALL[:-1]
    flushed = adapter.incremental_decode(None, second.state)
    assert flushed.invalid_reason == "cross_lane_native_json"
    assert flushed.is_incomplete


def test_legacy_adapter_matches_existing_think_splitter_for_complete_tags():
    chunks = ["Intro <think>reasoning</think>Visible answer"]
    _assert_legacy_splitter_matches_existing(chunks)


def test_legacy_adapter_matches_existing_think_splitter_across_split_tags():
    chunks = ["Intro <th", "ink>reasoning</th", "ink>Visible answer"]
    _assert_legacy_splitter_matches_existing(chunks)


def test_legacy_adapter_flushes_non_native_unclosed_braces_like_existing_text():
    _assert_legacy_splitter_matches_existing(["Visible {"])


def test_legacy_visible_lane_decodes_native_tool_json_across_every_boundary():
    """Empty splitter lanes must not impersonate cross-lane tool JSON transport."""
    for boundary in range(1, len(_CALL)):
        adapter = LegacyThinkTagV1Adapter()
        native_fragments = []
        original_decode = adapter._native_adapter.incremental_decode

        def recording_decode(fragment, state):
            if fragment is not None:
                native_fragments.append(fragment)
            return original_decode(fragment, state)

        adapter._native_adapter.incremental_decode = recording_decode
        state = ProtocolDecodeState()
        results = []
        for sequence, chunk in enumerate((_CALL[:boundary], _CALL[boundary:])):
            result = adapter.incremental_decode(
                ProtocolFragment(sequence=sequence, lane="text", text=chunk), state
            )
            state = result.state
            results.append(result)
        results.append(adapter.incremental_decode(None, state))

        candidates = [candidate for result in results for candidate in result.candidates]
        assert [candidate.call_id for candidate in candidates] == ["toolu_1"], boundary
        assert "".join(result.visible_text for result in results) == "", boundary
        assert "".join(result.thinking_text for result in results) == "", boundary
        assert all(result.invalid_reason is None for result in results), boundary
        assert all(fragment.text for fragment in native_fragments), boundary


def test_legacy_think_block_decodes_complete_native_tool_json_without_reasoning_leak():
    """A tool call inside <think> is native reasoning protocol, never public thought text."""
    results = _decode_legacy_chunks([f"<think>{_CALL}</think>"])

    assert "".join(result.thinking_text for result in results) == ""
    assert "".join(result.visible_text for result in results) == ""
    candidates = [candidate for result in results for candidate in result.candidates]
    assert [candidate.call_id for candidate in candidates] == ["toolu_1"]
    assert candidates[0].source_lane == "reasoning"


def test_legacy_think_block_decodes_native_tool_json_split_across_chunks_and_tags():
    """Think-tag and JSON chunk boundaries cannot turn a tool call into public reasoning."""
    split_at = len(_CALL) // 2
    results = _decode_legacy_chunks([
        "<th",
        "ink>" + _CALL[:split_at],
        _CALL[split_at:] + "</think>",
    ])

    assert "".join(result.thinking_text for result in results) == ""
    candidates = [candidate for result in results for candidate in result.candidates]
    assert [candidate.call_id for candidate in candidates] == ["toolu_1"]


def test_legacy_think_block_hides_malformed_native_tool_json():
    """Malformed native JSON in thinking fails closed instead of leaking protocol text."""
    malformed = '{"type":"tool_use","id":"toolu_1","name":"write_file","input":{"x":1,}}'
    results = _decode_legacy_chunks([f"<think>{malformed}</think>"])

    assert "".join(result.thinking_text for result in results) == ""
    assert all(not result.candidates for result in results)
    assert [
        result.invalid_reason for result in results if result.invalid_reason
    ] == ["malformed_json"]
    assert all(not result.is_incomplete for result in results)


def _decode_legacy_chunks(chunks: list[str]) -> list:
    adapter = LegacyThinkTagV1Adapter()
    state = ProtocolDecodeState()
    results = []
    for sequence, chunk in enumerate(chunks):
        result = adapter.incremental_decode(
            ProtocolFragment(sequence=sequence, lane="text", text=chunk), state
        )
        state = result.state
        results.append(result)
    results.append(adapter.incremental_decode(None, state))
    return results


def _assert_legacy_splitter_matches_existing(chunks: list[str]) -> None:
    expected_splitter = _ThinkTagSplitter()
    adapter = LegacyThinkTagV1Adapter()
    state = ProtocolDecodeState()
    expected_thinking: list[str] = []
    expected_text: list[str] = []
    actual_text: list[str] = []
    actual_thinking: list[str] = []

    for sequence, chunk in enumerate(chunks):
        thinking, text = expected_splitter.feed(chunk)
        expected_thinking.append(thinking)
        expected_text.append(text)
        result = adapter.incremental_decode(
            ProtocolFragment(sequence=sequence, lane="text", text=chunk), state
        )
        state = result.state
        actual_text.append(result.visible_text)
        actual_thinking.append(result.thinking_text)

    expected_thinking_delta, expected_text_delta = expected_splitter.flush()
    expected_thinking.append(expected_thinking_delta)
    expected_text.append(expected_text_delta)
    flushed = adapter.incremental_decode(None, state)
    actual_text.append(flushed.visible_text)
    actual_thinking.append(flushed.thinking_text)

    assert flushed.state.reasoning_buffer == ""
    assert "".join(actual_thinking) == "".join(expected_thinking)
    assert "".join(actual_text) == "".join(expected_text)
    assert flushed.candidates == ()


def test_manifest_serializes_original_schema_without_flattening_or_mutation():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "parameters": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string", "minLength": 1}},
                    "required": ["file_path"],
                },
            },
        }
    ]

    manifest = StandardJsonV1Adapter().render_tool_manifest(tools, "Execute required actions.")

    assert '"minLength":1' in manifest
    assert tools[0]["function"]["parameters"]["properties"]["file_path"]["minLength"] == 1


def test_renderers_keep_native_tool_grammar_for_continuation_and_repair():
    adapter = StandardJsonV1Adapter()
    tools = [{"name": "read_file", "input_schema": {"type": "object"}}]

    continuation = adapter.render_continuation(tools, "Inspect x.py", "I need to read it.")
    repair = adapter.render_repair(tools, '{"type":"tool_use"')

    assert _CALL.split('"input"')[0] not in continuation
    assert '"type":"tool_use"' in continuation
    assert "Inspect x.py" in continuation
    assert "I need to read it." in continuation
    assert '"type":"tool_use"' in repair
    assert "Do not continue truncated JSON." in repair


def test_flush_keeps_unrecognized_unclosed_object_as_visible_text():
    adapter = StandardJsonV1Adapter()
    partial = adapter.incremental_decode(
        ProtocolFragment(sequence=0, lane="text", text='prose {"example":'),
        ProtocolDecodeState(),
    )

    flushed = adapter.incremental_decode(None, partial.state)

    assert partial.visible_text == "prose "
    assert flushed.visible_text == '{"example":'
    assert flushed.candidates == ()
    assert not flushed.is_incomplete


def test_rejects_tool_use_without_required_native_fields():
    incomplete_shape = '{"type":"tool_use","id":"toolu_1","name":"write_file"}'
    result = _decode(
        StandardJsonV1Adapter(),
        [ProtocolFragment(sequence=0, lane="text", text=incomplete_shape)],
    )[-1]

    assert result.candidates == ()
    assert result.invalid_reason == "invalid_tool_shape"


def _run_all() -> int:
    funcs = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    failures = 0
    for function in funcs:
        try:
            function()
            print(f"PASS {function.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {function.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {function.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(funcs) - failures}/{len(funcs)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
