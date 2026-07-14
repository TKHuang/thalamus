"""Integration test: consume_stream(composer=True) over synthetic Cursor frames.

Composer-2.x streams its whole output through the protobuf *thinking* field.
This test builds real Cursor stream frames carrying reasoning + answer + a tool
token block in that field, then verifies consume_stream separates them into
clean answer text, reasoning thinking, and parsed composer_tool_calls — and
that nothing leaks across an arbitrary chunk boundary.

Runs standalone (``.venv/bin/python tests/test_composer_pipeline.py``) and under
pytest.
"""

import asyncio
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proto import cursor_api_pb2 as pb  # noqa: E402
from claude_code import pipeline  # noqa: E402

consume_stream = pipeline.consume_stream


def _thinking_frame(text: str) -> bytes:
    """A Cursor stream frame carrying *text* in message.thinking.content."""
    resp = pb.StreamUnifiedChatWithToolsResponse()
    resp.message.thinking.content = text
    payload = resp.SerializeToString()
    return bytes([0]) + struct.pack(">I", len(payload)) + payload


async def _consume(frames: list[bytes], composer: bool = True) -> dict:
    text_deltas: list[str] = []
    thinking_deltas: list[str] = []

    async def _iter():
        for f in frames:
            yield f

    consumed = await consume_stream(
        _iter(),
        on_text_delta=lambda d: text_deltas.append(d),
        on_thinking_delta=lambda d: thinking_deltas.append(d),
        composer=composer,
    )
    consumed["_text_deltas"] = text_deltas
    consumed["_thinking_deltas"] = thinking_deltas
    return consumed


def test_composer_stream_separates_reasoning_answer_and_tools():
    block = (
        "<|tool_calls_begin|><|tool_call_begin|>Write<|tool_sep|>"
        "file_path\n/tmp/cube.html<|tool_sep|>content\n<html>cube</html>"
        "<|tool_call_end|><|tool_calls_end|>"
    )
    full = f"Let me plan the file.</think>Creating the file now.{block}"
    # Split the whole payload into several frames at an awkward offset that lands
    # inside the tool marker, to exercise cross-frame buffering.
    cut = full.index("<|tool_calls") + 5
    frames = [_thinking_frame(full[:cut]), _thinking_frame(full[cut:])]

    consumed = asyncio.run(_consume(frames))

    assert consumed["text"].strip() == "Creating the file now."
    assert "<|tool" not in consumed["text"]
    assert "</think>" not in consumed["text"]
    assert "Let me plan the file." in consumed["thinking"]
    assert consumed["composer_tool_calls"] == [
        {"name": "Write", "arguments": {"file_path": "/tmp/cube.html", "content": "<html>cube</html>"}}
    ]
    # Streamed answer deltas must also be clean (no marker leak to the client).
    assert "<|tool" not in "".join(consumed["_text_deltas"])
    assert consumed["had_content"] is True


def test_interrupted_composer_marker_never_leaks_to_text_deltas_or_partial_text():
    marker_variants = (
        "<|tool_calls_begin|><|tool_call_begin|>write_file",
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>write_file",
    )

    for marker in marker_variants:
        text_deltas: list[str] = []

        async def interrupted_stream():
            yield _thinking_frame(f"Plan.</think>Writing now.{marker}")
            raise ConnectionResetError("stream reset")

        async def run() -> dict:
            try:
                await consume_stream(
                    interrupted_stream(),
                    on_text_delta=text_deltas.append,
                    composer=True,
                )
            except Exception as exc:
                assert type(exc).__name__ == "PartialStreamConsumptionError"
                return exc.consumed
            raise AssertionError("expected interrupted stream error")

        consumed = asyncio.run(run())
        visible_text = "".join(text_deltas)
        assert "Writing now." in visible_text
        assert "<|tool_" not in visible_text
        assert "<｜tool" not in visible_text
        assert "<|tool_" not in consumed["text"]
        assert "<｜tool" not in consumed["text"]
        assert consumed["interrupted_tool_state"] == marker
        assert pipeline._is_interrupted_tool_json(consumed, is_composer=True)


def test_clean_eof_keeps_a_literal_trailing_less_than_visible():
    text_deltas: list[str] = []
    consumed = asyncio.run(
        consume_stream(
            _single_frame("<|final|>The literal comparison operator is <"),
            on_text_delta=text_deltas.append,
            composer=True,
        )
    )

    assert consumed["text"] == "The literal comparison operator is <"
    assert "".join(text_deltas) == "The literal comparison operator is <"
    assert consumed["interrupted_tool_state"] == ""
    assert not pipeline._is_interrupted_tool_json(consumed, is_composer=True)


def test_reset_keeps_a_literal_trailing_less_than_visible_without_marker_state():
    text_deltas: list[str] = []

    async def interrupted_stream():
        yield _thinking_frame("<|final|>The literal comparison operator is <")
        raise ConnectionResetError("stream reset")

    async def run() -> dict:
        try:
            await consume_stream(
                interrupted_stream(),
                on_text_delta=text_deltas.append,
                composer=True,
            )
        except Exception as exc:
            assert type(exc).__name__ == "PartialStreamConsumptionError"
            return exc.consumed
        raise AssertionError("expected interrupted stream error")

    consumed = asyncio.run(run())
    assert consumed["text"] == "The literal comparison operator is <"
    assert "".join(text_deltas) == "The literal comparison operator is <"
    assert consumed["interrupted_tool_state"] == ""
    assert not pipeline._is_interrupted_tool_json(consumed, is_composer=True)


def test_clean_eof_preserves_incomplete_composer_marker_without_visible_leakage():
    marker_variants = (
        "<|tool_calls_begin|><|tool_call_begin|>write_file",
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>write_file",
    )

    for marker in marker_variants:
        text_deltas: list[str] = []
        consumed = asyncio.run(
            consume_stream(
                _single_frame(f"Plan.</think>Writing now.{marker}"),
                on_text_delta=text_deltas.append,
                composer=True,
            )
        )

        assert "Writing now." in "".join(text_deltas)
        assert "<|tool_" not in "".join(text_deltas)
        assert "<｜tool" not in "".join(text_deltas)
        assert "<|tool_" not in consumed["text"]
        assert "<｜tool" not in consumed["text"]
        assert consumed["interrupted_tool_state"] == marker
        assert pipeline._is_interrupted_tool_json(consumed, is_composer=True)


def test_every_marker_prefix_reset_is_partial_without_visible_leakage():
    marker_variants = (
        ("<|final|>", "<|tool_calls_begin|>"),
        ("<｜final｜>", "<｜tool▁calls▁begin｜>"),
    )

    for control, marker in marker_variants:
        # A bare '<' is ordinary prose; '<|' and '<｜' unambiguously begin markers.
        for boundary in range(2, len(marker) + 1):
            text_deltas: list[str] = []

            async def interrupted_stream():
                yield _thinking_frame(control + marker[:boundary])
                raise ConnectionResetError("stream reset")

            async def run() -> dict:
                try:
                    await consume_stream(
                        interrupted_stream(),
                        on_text_delta=text_deltas.append,
                        composer=True,
                    )
                except Exception as exc:
                    assert type(exc).__name__ == "PartialStreamConsumptionError", boundary
                    return exc.consumed
                raise AssertionError(f"expected interruption at boundary {boundary}")

            consumed = asyncio.run(run())
            visible_text = "".join(text_deltas)
            assert "<|tool_" not in visible_text, boundary
            assert "<｜tool" not in visible_text, boundary
            assert "<|tool_" not in consumed["text"], boundary
            assert "<｜tool" not in consumed["text"], boundary
            assert consumed["interrupted_tool_state"] == marker[:boundary], boundary
            assert pipeline._is_interrupted_tool_json(consumed, is_composer=True), boundary

    assert not pipeline._is_interrupted_tool_json(
        {"text": "ordinary prose <", "thinking": ""},
        is_composer=True,
    )


async def _single_frame(text: str):
    yield _thinking_frame(text)


def test_composer_plain_answer_no_marker():
    # composer sometimes answers trivially with no control token / tools.
    frames = [_thinking_frame("用一句话"), _thinking_frame("介绍：我是助手。")]
    consumed = asyncio.run(_consume(frames))
    assert consumed["text"].strip() == "用一句话介绍：我是助手。"
    assert consumed["composer_tool_calls"] == []


def test_non_composer_stream_unaffected():
    # Without composer=True, the thinking field stays thinking (regression guard).
    frames = [_thinking_frame("some reasoning"), _thinking_frame(" more")]
    consumed = asyncio.run(_consume(frames, composer=False))
    assert consumed["thinking"] == "some reasoning more"
    assert consumed["text"] == ""
    assert consumed["composer_tool_calls"] == []


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
