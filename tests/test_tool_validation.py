"""Unit and executor-boundary tests for decoded tool candidates."""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code.tool_protocols import DecodedToolCandidate  # noqa: E402
from claude_code.tool_validation import validate_tool_candidates  # noqa: E402


def _candidate(name: str, arguments: object) -> dict:
    return {
        "id": "call_test",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def test_accepts_an_exactly_advertised_name_and_object_arguments():
    result = validate_tool_candidates(
        [_candidate("write_file", {"path": "émoji-工具.txt", "content": "hello"})],
        allowed_names={"write_file"},
    )

    assert result.rejected == ()
    assert result.accepted[0]["function"] == {
        "name": "write_file",
        "arguments": json.dumps(
            {"path": "émoji-工具.txt", "content": "hello"},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }


def test_resolves_case_and_documented_aliases_only_to_advertised_names():
    case_result = validate_tool_candidates(
        [_candidate("WRITE_FILE", {"path": "x"})],
        allowed_names={"write_file"},
    )
    alias_result = validate_tool_candidates(
        [_candidate("write_file", {"path": "x"})],
        allowed_names={"Write"},
    )
    reverse_alias_result = validate_tool_candidates(
        [_candidate("Write", {"path": "x"})],
        allowed_names={"write_file"},
    )

    assert case_result.accepted[0]["function"]["name"] == "write_file"
    assert alias_result.accepted[0]["function"]["name"] == "Write"
    assert reverse_alias_result.accepted == ()
    assert reverse_alias_result.rejected[0].reason == "unknown_tool"


def test_unknown_cursor_tools_never_reach_executor():
    candidates = [
        _candidate("Write", {"file_path": "x"}),
        _candidate("Edit", {"file_path": "x"}),
        _candidate("terminal", {"command": "pwd"}),
    ]

    result = validate_tool_candidates(candidates, allowed_names={"write_file"})

    assert result.accepted == ()
    assert [(rejection.raw_name, rejection.reason) for rejection in result.rejected] == [
        ("Write", "unknown_tool"),
        ("Edit", "unknown_tool"),
        ("terminal", "unknown_tool"),
    ]


def test_rejects_invalid_json_and_non_object_arguments_without_replacement():
    candidates = [
        _candidate("write_file", '{"path":'),
        _candidate("write_file", "[]"),
        _candidate("write_file", "false"),
        _candidate("write_file", None),
        _candidate("write_file", []),
        _candidate("write_file", 0),
    ]

    result = validate_tool_candidates(candidates, allowed_names={"write_file"})

    assert result.accepted == ()
    assert [rejection.reason for rejection in result.rejected] == [
        "arguments_invalid_json",
        "arguments_not_object",
        "arguments_not_object",
        "arguments_not_object",
        "arguments_not_object",
        "arguments_not_object",
    ]


def test_accepts_protocol_candidate_with_json_object_arguments():
    candidate = DecodedToolCandidate(
        call_id="toolu_1",
        raw_name="write_file",
        arguments={"path": "x.txt"},
        source_lane="text",
    )

    result = validate_tool_candidates([candidate], allowed_names={"write_file"})

    assert result.rejected == ()
    assert result.accepted == (
        {
            "id": "toolu_1",
            "type": "function",
            "function": {"name": "write_file", "arguments": '{"path":"x.txt"}'},
        },
    )


def test_returns_accepted_and_rejected_candidates_in_original_order():
    candidates = [
        _candidate("write_file", {"path": "first.txt"}),
        _candidate("Read", {"file_path": "secret.txt"}),
        _candidate("write_file", "[]"),
        _candidate("write_file", {"path": "last.txt"}),
    ]

    result = validate_tool_candidates(candidates, allowed_names={"write_file"})

    assert [call["function"]["arguments"] for call in result.accepted] == [
        '{"path":"first.txt"}',
        '{"path":"last.txt"}',
    ]
    assert [(rejection.raw_name, rejection.reason) for rejection in result.rejected] == [
        ("Read", "unknown_tool"),
        ("write_file", "arguments_not_object"),
    ]


def test_parser_cannot_coerce_null_arguments_to_an_empty_object():
    from claude_code.tool_parser import try_parse_tool_calls_from_text

    parsed = try_parse_tool_calls_from_text(
        '{"type":"tool_use","id":"toolu_null","name":"write_file","input":null}'
    )
    result = validate_tool_candidates(parsed, allowed_names={"write_file"})

    assert result.accepted == ()
    assert result.rejected[0].reason == "arguments_not_object"


def test_task_complete_requires_explicit_internal_pipeline_permission():
    candidate = _candidate("task_complete", {"result": "done"})

    external_result = validate_tool_candidates([candidate], allowed_names={"write_file"})
    internal_result = validate_tool_candidates(
        [candidate],
        allowed_names={"write_file"},
        allow_internal_task_complete=True,
    )
    alias_result = validate_tool_candidates(
        [_candidate("task_tracker", {"result": "done"})],
        allowed_names={"write_file"},
        allow_internal_task_complete=True,
    )

    assert external_result.accepted == ()
    assert external_result.rejected[0].reason == "unknown_tool"
    assert internal_result.accepted[0]["function"]["name"] == "task_complete"
    assert alias_result.accepted == ()
    assert alias_result.rejected[0].reason == "unknown_tool"


def test_rejects_non_standard_json_constants():
    candidates = [
        _candidate("write_file", '{"value":NaN}'),
        _candidate("write_file", '{"value":Infinity}'),
        _candidate("write_file", '{"value":-Infinity}'),
    ]

    result = validate_tool_candidates(candidates, allowed_names={"write_file"})

    assert result.accepted == ()
    assert [rejection.reason for rejection in result.rejected] == [
        "arguments_invalid_json",
        "arguments_invalid_json",
        "arguments_invalid_json",
    ]


def test_compatibility_helper_rejects_non_standard_json_constants():
    from config.tool_registry import normalize_tool_arguments_as_json_object

    results = [
        normalize_tool_arguments_as_json_object('{"value":NaN}'),
        normalize_tool_arguments_as_json_object('{"value":Infinity}'),
        normalize_tool_arguments_as_json_object('{"value":-Infinity}'),
    ]

    assert results == [(False, "arguments_invalid_json")] * 3


def test_task_complete_requires_a_nonempty_string_result():
    candidates = [
        _candidate("task_complete", {}),
        _candidate("task_complete", {"result": ""}),
        _candidate("task_complete", {"result": 7}),
    ]

    result = validate_tool_candidates(
        candidates,
        allowed_names={"write_file"},
        allow_internal_task_complete=True,
    )

    assert result.accepted == ()
    assert [rejection.reason for rejection in result.rejected] == [
        "invalid_task_complete_result",
        "invalid_task_complete_result",
        "invalid_task_complete_result",
    ]


def _run_all() -> int:
    functions = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    failures = 0
    for function in functions:
        try:
            function()
            print(f"PASS {function.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {function.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {function.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(functions) - failures}/{len(functions)} passed")
    return 1 if failures else 0


def _consumed(text: str) -> dict:
    return {
        "text": text,
        "thinking": "",
        "composer_tool_calls": [],
        "interrupted_tool_state": "",
        "has_fatal_error": False,
        "errors": [],
        "had_content": bool(text),
        "metrics": {"chunk_count": 1, "first_chunk_latency_ms": 0},
    }


def test_initial_unknown_candidate_is_not_returned_to_an_executor():
    from claude_code import pipeline

    class DummyStream:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    originals = {
        "build_cursor_stream_params": pipeline.build_cursor_stream_params,
        "open_streaming_h2_request": pipeline.open_streaming_h2_request,
        "consume_stream": pipeline.consume_stream,
        "log_llm_request": pipeline.log_llm_request,
        "log_llm_response": pipeline.log_llm_response,
        "log_llm_api_call": pipeline.log_llm_api_call,
        "MAX_CONTINUATION_RETRIES": pipeline.MAX_CONTINUATION_RETRIES,
    }
    try:
        pipeline.build_cursor_stream_params = lambda *args: ("/chat", {}, b"")
        pipeline.open_streaming_h2_request = lambda *args: DummyStream()

        async def fake_consume_stream(*args, **kwargs):
            return _consumed(
                '{"type":"tool_use","id":"toolu_1","name":"Write",'
                '"input":{"path":"x.txt"}}'
            )

        pipeline.consume_stream = fake_consume_stream
        pipeline.log_llm_request = lambda *args, **kwargs: ""
        pipeline.log_llm_response = lambda *args, **kwargs: ""
        pipeline.log_llm_api_call = lambda *args, **kwargs: None
        pipeline.MAX_CONTINUATION_RETRIES = 0
        result = __import__("asyncio").run(
            pipeline._call_cursor_direct(
                messages=[{"role": "user", "content": "write x.txt"}],
                model="standard-model",
                tools=[],
                valid_tool_names=["write_file"],
                auth_token="token",
            )
        )
    finally:
        for name, original in originals.items():
            setattr(pipeline, name, original)

    assert "tool_calls" not in result


def test_continuation_unknown_candidate_is_not_returned_to_an_executor():
    from claude_code import pipeline

    class DummyStream:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    calls = {"count": 0}
    originals = {
        "build_cursor_stream_params": pipeline.build_cursor_stream_params,
        "open_streaming_h2_request": pipeline.open_streaming_h2_request,
        "consume_stream": pipeline.consume_stream,
        "log_llm_request": pipeline.log_llm_request,
        "log_llm_response": pipeline.log_llm_response,
        "log_llm_api_call": pipeline.log_llm_api_call,
        "MAX_CONTINUATION_RETRIES": pipeline.MAX_CONTINUATION_RETRIES,
    }
    try:
        pipeline.build_cursor_stream_params = lambda *args: ("/chat", {}, b"")
        pipeline.open_streaming_h2_request = lambda *args: DummyStream()

        async def fake_consume_stream(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                return _consumed("I will write x.txt.")
            return _consumed(
                '{"type":"tool_use","id":"toolu_2","name":"Write",'
                '"input":{"path":"x.txt"}}'
            )

        pipeline.consume_stream = fake_consume_stream
        pipeline.log_llm_request = lambda *args, **kwargs: ""
        pipeline.log_llm_response = lambda *args, **kwargs: ""
        pipeline.log_llm_api_call = lambda *args, **kwargs: None
        pipeline.MAX_CONTINUATION_RETRIES = 1
        result = __import__("asyncio").run(
            pipeline._call_cursor_direct(
                messages=[{"role": "user", "content": "write x.txt"}],
                model="standard-model",
                tools=[],
                valid_tool_names=["write_file"],
                auth_token="token",
            )
        )
    finally:
        for name, original in originals.items():
            setattr(pipeline, name, original)

    assert calls["count"] == 2
    assert "tool_calls" not in result


def _malformed_task_complete_inputs() -> tuple[dict, ...]:
    return ({}, {"result": ""}, {"result": 7})


def _task_complete_json(arguments: dict) -> str:
    return json.dumps(
        {"type": "tool_use", "id": "toolu_bad", "name": "task_complete", "input": arguments},
        separators=(",", ":"),
    )


def test_initial_malformed_task_complete_never_reaches_public_tool_calls():
    from claude_code import pipeline

    class DummyStream:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    originals = {
        "build_cursor_stream_params": pipeline.build_cursor_stream_params,
        "open_streaming_h2_request": pipeline.open_streaming_h2_request,
        "consume_stream": pipeline.consume_stream,
        "log_llm_request": pipeline.log_llm_request,
        "log_llm_response": pipeline.log_llm_response,
        "log_llm_api_call": pipeline.log_llm_api_call,
        "MAX_CONTINUATION_RETRIES": pipeline.MAX_CONTINUATION_RETRIES,
    }
    try:
        pipeline.build_cursor_stream_params = lambda *args: ("/chat", {}, b"")
        pipeline.open_streaming_h2_request = lambda *args: DummyStream()
        pipeline.log_llm_request = lambda *args, **kwargs: ""
        pipeline.log_llm_response = lambda *args, **kwargs: ""
        pipeline.log_llm_api_call = lambda *args, **kwargs: None
        pipeline.MAX_CONTINUATION_RETRIES = 0
        for arguments in _malformed_task_complete_inputs():
            async def fake_consume_stream(*args, **kwargs):
                return _consumed(_task_complete_json(arguments))

            pipeline.consume_stream = fake_consume_stream
            result = asyncio.run(
                pipeline._call_cursor_direct(
                    messages=[{"role": "user", "content": "write x.txt"}],
                    model="standard-model",
                    tools=[],
                    valid_tool_names=["write_file"],
                    auth_token="token",
                )
            )
            assert all(
                call["function"]["name"] != "task_complete"
                for call in result.get("tool_calls", [])
            )
    finally:
        for name, original in originals.items():
            setattr(pipeline, name, original)


def test_continuation_malformed_task_complete_never_reaches_public_tool_calls():
    from claude_code import pipeline

    class DummyStream:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    originals = {
        "build_cursor_stream_params": pipeline.build_cursor_stream_params,
        "open_streaming_h2_request": pipeline.open_streaming_h2_request,
        "consume_stream": pipeline.consume_stream,
        "log_llm_request": pipeline.log_llm_request,
        "log_llm_response": pipeline.log_llm_response,
        "log_llm_api_call": pipeline.log_llm_api_call,
        "MAX_CONTINUATION_RETRIES": pipeline.MAX_CONTINUATION_RETRIES,
    }
    try:
        pipeline.build_cursor_stream_params = lambda *args: ("/chat", {}, b"")
        pipeline.open_streaming_h2_request = lambda *args: DummyStream()
        pipeline.log_llm_request = lambda *args, **kwargs: ""
        pipeline.log_llm_response = lambda *args, **kwargs: ""
        pipeline.log_llm_api_call = lambda *args, **kwargs: None
        pipeline.MAX_CONTINUATION_RETRIES = 1
        for arguments in _malformed_task_complete_inputs():
            calls = {"count": 0}

            async def fake_consume_stream(*args, **kwargs):
                calls["count"] += 1
                if calls["count"] == 1:
                    return _consumed("I will write x.txt.")
                return _consumed(_task_complete_json(arguments))

            pipeline.consume_stream = fake_consume_stream
            result = asyncio.run(
                pipeline._call_cursor_direct(
                    messages=[{"role": "user", "content": "write x.txt"}],
                    model="standard-model",
                    tools=[],
                    valid_tool_names=["write_file"],
                    auth_token="token",
                )
            )
            assert calls["count"] == 2
            assert all(
                call["function"]["name"] != "task_complete"
                for call in result.get("tool_calls", [])
            )
    finally:
        for name, original in originals.items():
            setattr(pipeline, name, original)


def test_assemblers_strip_task_complete_from_direct_results():
    from claude_code import pipeline

    arguments = {}

    async def fake_call_cursor_direct(*args, **kwargs):
        return {
            "tool_calls": [_candidate("task_complete", arguments)],
            "text": "",
            "thinking": "",
            "model": "standard-model",
            "fallback_attempts": 0,
            "stats": {},
        }

    original = pipeline._call_cursor_direct
    pipeline._call_cursor_direct = fake_call_cursor_direct
    try:
        for arguments in _malformed_task_complete_inputs():
            for builder in (
                pipeline._build_streaming_result_anthropic,
                pipeline._build_streaming_result_openai,
            ):
                result = builder(
                    request_id="req_test",
                    messages=[],
                    tools=[],
                    valid_tool_names=["write_file"],
                    resolved_model="standard-model",
                    max_tokens=None,
                    token="token",
                    pipeline_start=0,
                    base_telemetry={},
                )

                async def collect() -> str:
                    return "".join([chunk async for chunk in result["stream_handler"]()])

                assert "task_complete" not in asyncio.run(collect())
    finally:
        pipeline._call_cursor_direct = original


def test_unary_assemblers_strip_task_complete_from_direct_results():
    from claude_code import pipeline

    arguments = {}

    async def fake_call_cursor_direct(*args, **kwargs):
        return {
            "tool_calls": [_candidate("task_complete", arguments)],
            "text": "",
            "thinking": "",
            "model": "standard-model",
            "fallback_attempts": 0,
            "stats": {},
        }

    async def assert_public_unary_responses() -> None:
        for arguments in _malformed_task_complete_inputs():
            for original_format in ("openai", "anthropic"):
                result = await pipeline._build_unary_result(
                    req=None,
                    request_id="req_test",
                    messages=[],
                    tools=[],
                    valid_tool_names=["write_file"],
                    resolved_model="standard-model",
                    max_tokens=None,
                    token="token",
                    original_format=original_format,
                    pipeline_start=0,
                    base_telemetry={},
                )
                assert "task_complete" not in json.dumps(result["body"])

    original = pipeline._call_cursor_direct
    pipeline._call_cursor_direct = fake_call_cursor_direct
    try:
        asyncio.run(assert_public_unary_responses())
    finally:
        pipeline._call_cursor_direct = original


if __name__ == "__main__":
    sys.exit(_run_all())
