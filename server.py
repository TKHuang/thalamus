import logging
import os
import sys
from pathlib import Path

from colorama import just_fix_windows_console
from dotenv import load_dotenv

from core.paths import token_env_path

# Enable ANSI color sequences on legacy Windows consoles (cmd.exe / PowerShell);
# no-op on macOS, Linux, and modern terminals.
just_fix_windows_console()

# Dev/local config from the repo-root .env (if present)...
repo_env = Path(__file__).parent / ".env"
if repo_env.exists():
    load_dotenv(repo_env)

# ...then let the persisted runtime token from the platform data dir override it,
# so a captured token survives restarts even when the app is installed in a
# read-only location (e.g. C:\Program Files). Mirrors core/token_manager.py.
data_env = token_env_path()
if data_env.exists():
    load_dotenv(data_env, override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("thalamus-py")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.anthropic_messages import router as anthropic_router
from routes.openai_chat import router as openai_router
from routes.model_routes import router as model_router
from routes.token_routes import router as token_router
from routes.login_routes import router as login_router

app = FastAPI(title="thalamus-py", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(anthropic_router)
app.include_router(openai_router)
app.include_router(model_router)
app.include_router(token_router)
app.include_router(login_router)

@app.get("/")
async def root():
    return {"service": "thalamus-py", "status": "running"}

@app.get("/health")
async def health():
    from core.token_manager import has_cursor_access_token
    return {
        "status": "ok",
        "has_token": has_cursor_access_token(),
    }

@app.get("/api/hello")
async def api_hello():
    """CC SDK calls BASE_API_URL/api/hello as a connectivity check during startup."""
    return {"status": "ok"}

@app.get("/v1/oauth/hello")
async def oauth_hello():
    """CC SDK calls TOKEN_URL/v1/oauth/hello during auth health check."""
    return {"status": "ok"}

@app.post("/v1/messages/count_tokens")
async def count_tokens():
    """CC SDK may call this for token counting; return a dummy response."""
    return {"input_tokens": 0}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "3013"))
    host = os.environ.get("THALAMUS_HOST", "127.0.0.1")
    logger.info(f"Starting thalamus-py on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
