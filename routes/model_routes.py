from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from google.protobuf.unknown_fields import UnknownFieldSet

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

# AvailableModel carries a repeated string of usable reasoning-effort variant
# ids (e.g. "glm-5.2-high", "glm-5.2-max") in field 36. Our partial .proto only
# defines name/defaultOn/..., so field 36 lands in the unknown-field set.
_MODEL_VARIANT_FIELD = 36


def _extract_variant_model_ids(model) -> list[str]:
    """Read Cursor's effort-suffixed variant ids (field 36) for a model.

    Some models (e.g. glm-5.2) reject their bare name and are only usable via a
    variant, so these must be surfaced in the model listing.
    """
    ids: list[str] = []
    for field in UnknownFieldSet(model):
        if field.field_number == _MODEL_VARIANT_FIELD and field.wire_type == 2:
            try:
                ids.append(field.data.decode("utf-8"))
            except UnicodeDecodeError:
                continue
    return ids


async def _fetch_available_models(token: str) -> list[str]:
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

    # Emit the bare model name plus every effort-suffixed variant, deduped and
    # order-preserved. Bare names are kept (most work as-is); variants make
    # suffix-only models like glm-5.2 selectable.
    model_ids: list[str] = []
    seen: set[str] = set()
    for m in resp.models:
        for model_id in [m.name, *_extract_variant_model_ids(m)]:
            if model_id and model_id not in seen:
                seen.add(model_id)
                model_ids.append(model_id)
    return model_ids


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
        model_ids = await _fetch_available_models(token)
        logger.info(f"AvailableModels returned {len(model_ids)} models")
        return JSONResponse(content={
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "cursor",
                }
                for model_id in model_ids
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
