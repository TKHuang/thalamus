"""Resource-lifecycle tests for the Cursor HTTP/2 transport."""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import cursor_h2_client


def test_cancelling_consumer_closes_iterator_response_context_and_client(monkeypatch):
    events: list[str] = []
    iterator_started = asyncio.Event()

    class BlockingIterator:
        def __aiter__(self):
            return self

        async def __anext__(self):
            iterator_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                events.append("iterator_cancelled")
                raise

        async def aclose(self):
            events.append("iterator_aclose")

    iterator = BlockingIterator()

    class FakeResponse:
        status_code = 200

        def aiter_bytes(self):
            return iterator

        async def aclose(self):
            events.append("response_aclose")

    class FakeResponseContext:
        async def __aenter__(self):
            events.append("context_enter")
            return FakeResponse()

        async def __aexit__(self, exc_type, exc, traceback):
            assert exc_type is asyncio.CancelledError
            assert isinstance(exc, asyncio.CancelledError)
            events.append("context_exit")

    class FakeClient:
        def stream(self, method, path, headers, content):
            assert method == "POST"
            return FakeResponseContext()

        async def aclose(self):
            events.append("client_aclose")

    monkeypatch.setattr(
        cursor_h2_client,
        "_build_client",
        lambda **_kwargs: FakeClient(),
    )

    async def consume() -> None:
        async with cursor_h2_client.open_streaming_h2_request(
            "/stream", {}, b"body"
        ) as stream:
            async for _chunk in stream:
                raise AssertionError("blocking iterator must not yield")

    async def verify() -> None:
        task = asyncio.create_task(consume())
        await asyncio.wait_for(iterator_started.wait(), timeout=0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(verify())

    assert events == [
        "context_enter",
        "iterator_cancelled",
        "iterator_aclose",
        "response_aclose",
        "context_exit",
        "client_aclose",
    ]


def test_streaming_request_uses_extended_read_timeout(monkeypatch):
    captured: dict[str, object] = {}

    async def empty_stream():
        if False:  # pragma: no cover - keeps this an async generator
            yield b""

    class FakeResponse:
        status_code = 200

        def aiter_bytes(self):
            return empty_stream()

        async def aclose(self):
            pass

    class FakeResponseContext:
        async def __aenter__(self):
            return FakeResponse()

        async def __aexit__(self, exc_type, exc, traceback):
            pass

    class FakeClient:
        def stream(self, method, path, headers, content):
            return FakeResponseContext()

        async def aclose(self):
            pass

    def build_client(*, timeout):
        captured["timeout"] = timeout
        return FakeClient()

    monkeypatch.setattr(cursor_h2_client, "_build_client", build_client)

    async def consume() -> None:
        async with cursor_h2_client.open_streaming_h2_request(
            "/stream", {}, b"body"
        ) as stream:
            assert [chunk async for chunk in stream] == []

    asyncio.run(consume())

    timeout = captured["timeout"]
    assert timeout is cursor_h2_client._STREAMING_TIMEOUT
    assert timeout.connect == 10.0
    assert timeout.read == 600.0
    assert timeout.write == 120.0
    assert timeout.pool == 120.0


def test_stream_timeout_env_parser_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("CURSOR_STREAM_READ_TIMEOUT_SECONDS", "not-a-number")
    assert cursor_h2_client._positive_float_env(
        "CURSOR_STREAM_READ_TIMEOUT_SECONDS", 600.0
    ) == 600.0

    monkeypatch.setenv("CURSOR_STREAM_READ_TIMEOUT_SECONDS", "0")
    assert cursor_h2_client._positive_float_env(
        "CURSOR_STREAM_READ_TIMEOUT_SECONDS", 600.0
    ) == 600.0

    monkeypatch.setenv("CURSOR_STREAM_READ_TIMEOUT_SECONDS", "450")
    assert cursor_h2_client._positive_float_env(
        "CURSOR_STREAM_READ_TIMEOUT_SECONDS", 600.0
    ) == 450.0
