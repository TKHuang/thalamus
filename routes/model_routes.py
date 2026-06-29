from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from config.cursor_client import get_cursor_client_version
from core.bearer_token import extract_bearer_tokens, strip_cursor_user_prefix
from core.cursor_h2_client import send_unary_h2_request
from core.protobuf_builder import generate_obfuscated_machine_id_checksum
from core.token_manager import get_cursor_access_token
from proto import cursor_api_pb2 as pb
from utils.structured_logging import ThalamusStructuredLogger

logger = ThalamusStructuredLogger.get_logger("model-routes", "DEBUG")
router = APIRouter()

CURSOR_CLIENT_VERSION = get_cursor_client_version()


async def _fetch_available_models(token: str) -> list[dict]:
    checksum = generate_obfuscated_machine_id_checksum(token.strip())
    headers = {
        "authorization": f"Bearer {token}",
        "connect-protocol-version": "1",
        "content-type": "application/proto",
        "user-agent": "connect-es/1.6.1",
        "x-cursor-checksum": checksum,
        "x-cursor-client-version": CURSOR_CLIENT_VERSION,
        "x-cursor-config-version": str(uuid4()),
        "x-cursor-timezone": "Asia/Shanghai",
        "x-ghost-mode": "true",
        "Host": "api2.cursor.sh",
    }

    result = await send_unary_h2_request(
        "/aiserver.v1.AiService/AvailableModels",
        headers,
        b"",
    )

    if result["status"] != 200:
        raise RuntimeError(
            f"Cursor AvailableModels returned {result['status']}: "
            f"{result['buffer'][:500]}"
        )

    resp = pb.AvailableModelsResponse()
    resp.ParseFromString(result["buffer"])

    models = []
    for m in resp.models:
        models.append({
            "name": m.name,
            "defaultOn": m.defaultOn,
            "isLongContextOnly": getattr(m, "isLongContextOnly", False),
            "isChatOnly": getattr(m, "isChatOnly", False),
        })
    return models


@router.get("/v1/models")
async def list_models(request: Request):
    """OpenAI-compatible model listing — proxies Cursor's AvailableModels."""
    tokens = extract_bearer_tokens(request.headers.get("authorization"))
    raw_token = strip_cursor_user_prefix(tokens[0]) if tokens else ""
    token = raw_token or strip_cursor_user_prefix(get_cursor_access_token())

    if not token:
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "No auth token", "type": "authentication_error"}},
        )

    try:
        models = await _fetch_available_models(token)
        logger.info(f"AvailableModels returned {len(models)} models")
        return JSONResponse(content={
            "object": "list",
            "data": [
                {
                    "id": m["name"],
                    "object": "model",
                    "created": 0,
                    "owned_by": "cursor",
                }
                for m in models
            ],
        })
    except Exception as exc:
        logger.error(f"AvailableModels failed: {exc}")
        return JSONResponse(
            status_code=502,
            content={"error": {"message": str(exc), "type": "upstream_error"}},
        )


@router.get("/models")
async def list_models_alt(request: Request):
    """Alias without /v1 prefix (legacy compat)."""
    return await list_models(request)
