"""HTTP route contracts for the OpenAI Responses API."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import routes.openai_responses as responses_route
from server import app


def test_responses_route_normalizes_and_returns_unary_response(monkeypatch):
    captured = {}

    async def fake_run_pipeline(req, request_id, auth_token=""):
        captured["req"] = req
        captured["auth_token"] = auth_token
        return {
            "ok": True,
            "stream": False,
            "body": {
                "id": "resp_test",
                "object": "response",
                "status": "completed",
                "output": [],
            },
        }

    monkeypatch.setattr(responses_route, "run_pipeline", fake_run_pipeline)
    response = TestClient(app).post(
        "/v1/responses",
        headers={"Authorization": "Bearer test-token"},
        json={"model": "gpt-test", "input": "Hello"},
    )

    assert response.status_code == 200
    assert response.json()["object"] == "response"
    assert captured["req"].original_format == "openai_responses"
    assert captured["req"].messages == [{"role": "user", "content": "Hello"}]
    assert captured["auth_token"] == "Bearer test-token"


def test_responses_route_streams_sse_without_chat_done_sentinel(monkeypatch):
    async def stream_handler():
        yield 'data: {"type":"response.created","sequence_number":0}\n\n'
        yield 'data: {"type":"response.completed","sequence_number":1}\n\n'

    async def fake_run_pipeline(req, request_id, auth_token=""):
        return {
            "ok": True,
            "stream": True,
            "stream_handler": stream_handler,
        }

    monkeypatch.setattr(responses_route, "run_pipeline", fake_run_pipeline)
    response = TestClient(app).post(
        "/v1/responses",
        json={"model": "gpt-test", "input": "Hello", "stream": True},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "response.created" in response.text
    assert "response.completed" in response.text
    assert "[DONE]" not in response.text


def test_responses_route_closes_inner_pipeline_stream(monkeypatch):
    class FakeRequest:
        headers = {}

        async def json(self):
            return {"model": "gpt-test", "input": "Hello", "stream": True}

    async def verify() -> None:
        inner_closed = asyncio.Event()
        never_finish = asyncio.Event()
        emitted = ["initial"]

        async def stream_handler():
            try:
                yield 'data: {"type":"response.created"}\n\n'
                await never_finish.wait()
                emitted.append("late")
                yield 'data: {"type":"response.completed"}\n\n'
            finally:
                inner_closed.set()

        async def fake_run_pipeline(req, request_id, auth_token=""):
            return {
                "ok": True,
                "stream": True,
                "stream_handler": stream_handler,
            }

        monkeypatch.setattr(responses_route, "run_pipeline", fake_run_pipeline)
        monkeypatch.setattr(responses_route, "log_thalamus_request", lambda *args, **kwargs: "")
        monkeypatch.setattr(responses_route, "log_thalamus_response", lambda *args, **kwargs: "")
        monkeypatch.setattr(responses_route, "log_thalamus_api_call", lambda *args, **kwargs: None)

        response = await responses_route.create_response(FakeRequest())
        body_stream = response.body_iterator
        await anext(body_stream)
        await body_stream.aclose()
        await asyncio.wait_for(inner_closed.wait(), timeout=0.2)
        never_finish.set()
        await asyncio.sleep(0)
        assert emitted == ["initial"]

    asyncio.run(verify())
