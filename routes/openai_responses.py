import time
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from claude_code.normalizers import normalize_openai_response
from claude_code.pipeline import run_pipeline
from core.token_manager import capture_token_from_request
from utils.structured_logging import ThalamusStructuredLogger
from utils.thalamus_api_logger import (
    log_thalamus_api_call,
    log_thalamus_request,
    log_thalamus_response,
)

logger = ThalamusStructuredLogger.get_logger("openai-responses", "DEBUG")
router = APIRouter()

ENDPOINT = "/v1/responses"


def _headers_summary(request: Request) -> dict:
    summary = {}
    for key, value in request.headers.items():
        key_lower = key.lower()
        if key_lower in ("authorization", "x-api-key"):
            summary[key] = "***" if value else "(absent)"
        elif key_lower in ("content-type", "content-length"):
            summary[key] = value
    return summary


@router.post(ENDPOINT)
async def create_response(request: Request):
    """Handle OpenAI Responses API requests."""
    start_time = time.time()
    auth_token = (
        request.headers.get("authorization", "")
        or request.headers.get("x-api-key", "")
    )
    capture_token_from_request(request.headers.get("authorization", ""))
    request_id = f"req_{time.time_ns() // 1000000}"

    try:
        payload = await request.json()
    except Exception as exc:
        body = {
            "error": {
                "message": str(exc),
                "type": "invalid_request_error",
            }
        }
        return JSONResponse(status_code=400, content=body)

    try:
        request_path = log_thalamus_request(
            request_id,
            ENDPOINT,
            "POST",
            payload,
            _headers_summary(request),
        )
    except Exception:
        request_path = ""

    logger.info(
        f"[{request_id}] POST {ENDPOINT} model={payload.get('model', '?')} "
        f"stream={payload.get('stream', False)} tools={len(payload.get('tools', []))}"
    )

    try:
        normalized = normalize_openai_response(payload)
        result = await run_pipeline(
            req=normalized,
            request_id=request_id,
            auth_token=auth_token,
        )
    except Exception as exc:
        logger.error(f"[{request_id}] pipeline_exception: {exc}")
        body = {"error": {"message": f"Pipeline error: {exc}", "type": "api_error"}}
        return JSONResponse(status_code=500, content=body)

    latency_ms = int((time.time() - start_time) * 1000)
    if not result.get("ok"):
        status = result.get("status", 500)
        body = result.get(
            "body",
            {"error": {"message": "Internal error", "type": "api_error"}},
        )
        try:
            response_path = log_thalamus_response(
                request_id, ENDPOINT, status, body, latency_ms=latency_ms
            )
            log_thalamus_api_call(
                request_id,
                ENDPOINT,
                "POST",
                status,
                latency_ms,
                request_path,
                response_path,
                error=body.get("error", {}).get("message"),
            )
        except Exception:
            pass
        return JSONResponse(status_code=status, content=body)

    if result.get("stream"):
        stream_handler = result.get("stream_handler")
        if not stream_handler:
            body = {"error": {"message": "No stream handler", "type": "api_error"}}
            return JSONResponse(status_code=500, content=body)

        async def wrapped_stream():
            response_size = 0
            upstream_stream = stream_handler()
            try:
                async for chunk in upstream_stream:
                    response_size += len(chunk)
                    yield chunk
            finally:
                try:
                    await upstream_stream.aclose()
                finally:
                    try:
                        final_latency_ms = int((time.time() - start_time) * 1000)
                        response_path = log_thalamus_response(
                            request_id,
                            ENDPOINT,
                            200,
                            {"_raw_sse_len": response_size},
                            latency_ms=final_latency_ms,
                        )
                        log_thalamus_api_call(
                            request_id,
                            ENDPOINT,
                            "POST",
                            200,
                            final_latency_ms,
                            request_path,
                            response_path,
                        )
                    except Exception:
                        pass

        return StreamingResponse(
            wrapped_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Thalamus-Request-Id": request_id,
            },
        )

    body = result.get("body", {})
    try:
        response_path = log_thalamus_response(
            request_id, ENDPOINT, 200, body, latency_ms=latency_ms
        )
        log_thalamus_api_call(
            request_id,
            ENDPOINT,
            "POST",
            200,
            latency_ms,
            request_path,
            response_path,
        )
    except Exception:
        pass
    return JSONResponse(content=body)
