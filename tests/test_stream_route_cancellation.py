"""All public streaming routes must close their owned pipeline iterator."""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import routes.anthropic_messages as anthropic_route
import routes.openai_chat as chat_route


class FakeRequest:
    headers = {}

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    async def json(self):
        return self._payload


def test_chat_and_anthropic_routes_close_inner_pipeline_stream(monkeypatch):
    async def verify_route(module, handler, payload) -> None:
        inner_closed = asyncio.Event()
        never_finish = asyncio.Event()
        emitted = ["initial"]

        async def stream_handler():
            try:
                yield "data: initial\n\n"
                await never_finish.wait()
                emitted.append("late")
                yield "data: late\n\n"
            finally:
                inner_closed.set()

        async def fake_run_pipeline(req, request_id, auth_token=""):
            return {
                "ok": True,
                "stream": True,
                "stream_handler": stream_handler,
            }

        monkeypatch.setattr(module, "run_pipeline", fake_run_pipeline)
        monkeypatch.setattr(module, "log_thalamus_request", lambda *args, **kwargs: "")
        monkeypatch.setattr(module, "log_thalamus_response", lambda *args, **kwargs: "")
        monkeypatch.setattr(module, "log_thalamus_api_call", lambda *args, **kwargs: None)

        response = await handler(FakeRequest(payload))
        body_stream = response.body_iterator
        await anext(body_stream)
        await body_stream.aclose()
        await asyncio.wait_for(inner_closed.wait(), timeout=0.2)
        never_finish.set()
        await asyncio.sleep(0)
        assert emitted == ["initial"]

    async def verify() -> None:
        await verify_route(
            chat_route,
            chat_route.chat_completions,
            {
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
        await verify_route(
            anthropic_route,
            anthropic_route.create_message,
            {
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 32,
                "stream": True,
            },
        )

    asyncio.run(verify())
