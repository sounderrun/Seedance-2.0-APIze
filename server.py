"""
Dreamina2API FastAPI Microservice
Port: 8090
Endpoints:
  - POST /v1/images/generations (OpenAI format)
  - POST /v1/chat/completions (OpenAI text-to-image chat gateway)
  - GET  /v1/models
  - GET  /v1/session
  - GET  /v1/accounts         (pool status)
  - POST /v1/accounts/add     (add session manually)
  - POST /admin/refresh-session
  - GET  /health
"""
import os
import time
import uuid
import base64
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Header, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

load_dotenv("config.env")

PORT = int(os.getenv("PORT", "8090"))
OUTPUT_DIR = Path(__file__).parent / "outputs"
SESSION_CHECK_INTERVAL = int(os.getenv("SESSION_CHECK_INTERVAL", "3600"))
ENABLE_FARMER = os.getenv("ENABLE_FARMER", "true").lower() in ("true", "1", "yes")

from engine import DreaminaEngine, SessionPool, InsufficientCreditsError, AuthMissingError, COST_PER_IMAGE
from cookie_sync import has_valid_session, get_cached_cookies, reload_cookies, get_session_id, sync_accounts, add_account
from session_refresher import DreaminaSessionRefresher
import sys

# Connect to Vault2API lifecycle engine
VAULT_DIR = Path(__file__).resolve().parent.parent / "vault"
if str(VAULT_DIR) not in sys.path:
    sys.path.insert(0, str(VAULT_DIR))

vault_client = None
try:
    from client.vault_client import VaultClient
    vault_client = VaultClient(
        name="dreamina2api-8090",
        self_base_url=f"http://127.0.0.1:{PORT}",
        providers=["dreamina"],
        vault_url=os.getenv("VAULT_URL", "http://127.0.0.1:8095"),
    )
except Exception:
    vault_client = None

SESSION_ID = os.getenv("DREAMINA_SESSION_ID", "")

# Session pool — load from farmer output if available
FARMER_SESSIONS_FILE = Path(__file__).parent / "farmer" / "output" / "sessions.jsonl"
FARMER_EXHAUSTED_FILE = Path(__file__).parent / "farmer" / "output" / "exhausted.txt"
pool: Optional[SessionPool] = None
if ENABLE_FARMER and FARMER_SESSIONS_FILE.exists():
    pool = SessionPool(sessions_file=FARMER_SESSIONS_FILE, exhausted_file=FARMER_EXHAUSTED_FILE)

engine = DreaminaEngine(session_id=SESSION_ID, output_dir=OUTPUT_DIR, pool=pool, vault_client=vault_client)
refresher = DreaminaSessionRefresher(engine=engine)
engine.refresher = refresher


@asynccontextmanager
async def lifespan(app: FastAPI):
    if vault_client:
        await vault_client.start()
        app.include_router(vault_client.router)

    cookies = get_cached_cookies()
    if not has_valid_session(cookies):
        cookies = reload_cookies()

    if has_valid_session(cookies):
        sid = get_session_id(cookies)
        print(f"[+] Dreamina2API: Firefox session active ({sid[:8]}***, {len(cookies)} cookies loaded).")
        # Prime credit cache in background on startup to eliminate cold-start delay
        import asyncio
        asyncio.create_task(engine.get_credits_info())
    else:
        print("=" * 72)
        print("[!] Dreamina2API: No active session found in Firefox.")
        print("    Please open Firefox and log into https://dreamina.capcut.com to enable Seedream.")
        print("=" * 72)

    refresher.start_background_monitor(SESSION_CHECK_INTERVAL)

    yield

    refresher.stop_background_monitor()

app = FastAPI(title="Dreamina2API", version="1.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageGenRequest(BaseModel):
    prompt: str
    model: Optional[str] = "seedream-5.0"
    n: Optional[int] = 1
    size: Optional[str] = None
    response_format: Optional[str] = "url"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "seedream-5.0"
    messages: List[ChatMessage]

class AddSessionRequest(BaseModel):
    sessionid: str
    sessionid_ss: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None
    cookies: Optional[List[dict]] = None
    name: Optional[str] = None


# ──────────────────────────────────────────────
# Health & Session endpoints
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    cookies = get_cached_cookies()
    has_session = has_valid_session(cookies)
    sid = get_session_id(cookies) or engine.session_id
    # Report the default session's own balance, never whichever account ran last.
    credits_info = engine._credits_by_sid.get(sid) or {
        "gift_credit": "unknown (query /v1/credits to refresh)",
        "generations_remaining": 0,
        "status": "active" if has_session else "auth_missing",
    }
    return {
        "status": "healthy" if has_session else "standby",
        "service": "dreamina2api",
        "session_active": has_session,
        "session_id_masked": f"{sid[:6]}...{sid[-4:]}" if sid else None,
        "cookies_count": len(cookies),
        "pool_active": pool.active_count() if pool else 0,
        "credits": credits_info,
        "last_check_status": refresher.last_check_status,
        "last_check_time": int(refresher.last_check_time) if refresher.last_check_time else None,
        "last_refresh_time": int(refresher.last_refresh_time) if refresher.last_refresh_time else None,
        "last_refresh_result": refresher.last_refresh_result,
        "check_interval_seconds": SESSION_CHECK_INTERVAL,
    }

@app.get("/v1/session")
def get_session():
    return {
        "session_configured": bool(engine.session_id),
        "session_id_masked": f"{engine.session_id[:6]}...{engine.session_id[-4:]}" if engine.session_id else None,
        "last_check_status": refresher.last_check_status,
        "last_check_time": refresher.last_check_time,
        "last_refresh_time": refresher.last_refresh_time,
        "last_refresh_result": refresher.last_refresh_result,
        "has_google_email": bool(refresher.get_config_value("GOOGLE_EMAIL")),
        "has_google_cookies": bool(refresher.get_config_value("GOOGLE_COOKIES")),
    }

@app.post("/admin/refresh-session")
async def trigger_refresh():
    try:
        new_sid = await refresher.refresh_session()
        if not new_sid:
            raise HTTPException(status_code=500, detail=f"Refresh failed: {refresher.last_refresh_result}")
        return {
            "status": "success",
            "message": "Session refreshed successfully",
            "session_id_masked": f"{new_sid[:6]}...{new_sid[-4:]}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# Account pool endpoints
# ──────────────────────────────────────────────

@app.get("/v1/accounts")
async def list_accounts():
    """List every registered account (Firefox containers + default session) with balances."""
    accounts = engine._refresh_accounts(force=False)
    out = []
    total_generations = 0
    for acc in accounts:
        info = await engine.get_credits_info(session_id=acc["sessionid"])
        generations = info.get("generations_remaining", 0)
        total_generations += generations
        out.append({
            "name": acc.get("name"),
            "container": acc.get("container"),
            "sessionid_masked": f"{acc['sessionid'][:8]}...",
            "uid": acc.get("uid"),
            "credits": info.get("gift_credit", 0),
            "generations_remaining": generations,
            "status": info.get("status"),
            "shark_rejected": info.get("shark_rejected", False),
            "last_used": engine._last_used.get(acc["sessionid"]),
            "cooldown_until": engine._cooldowns.get(acc["sessionid"]),
        })
    return {
        "enabled": True,
        "count": len(out),
        "total_generations_remaining": total_generations,
        "accounts": out,
    }

@app.post("/v1/accounts/sync")
def sync_accounts_endpoint():
    """Re-read Firefox containers and register every logged-in Dreamina account."""
    accounts = sync_accounts()
    engine._refresh_accounts(force=True)
    return {
        "status": "synced",
        "count": len(accounts),
        "accounts": [
            {"name": a.get("name"), "container": a.get("container"),
             "sessionid_masked": f"{a['sessionid'][:8]}..."}
            for a in accounts
        ],
    }

@app.post("/v1/accounts/add")
def add_account_endpoint(req: AddSessionRequest):
    """Register an account manually. A full cookie jar is required."""
    if not req.cookies:
        raise HTTPException(
            status_code=400,
            detail="A full cookie jar is required. Use POST /v1/accounts/sync for Firefox containers, "
                   "or pass every .capcut.com cookie in `cookies`.",
        )
    add_account(req.sessionid, req.cookies, name=req.name or req.email or "manual")
    engine._refresh_accounts(force=True)
    return {
        "status": "added",
        "sessionid_masked": f"{req.sessionid[:8]}...",
        "count": len(engine._accounts),
    }


# ──────────────────────────────────────────────
# Model list
# ──────────────────────────────────────────────

@app.get("/v1/models")
def list_models():
    models = [
        {"id": "seedream-5.0", "object": "model", "owned_by": "bytedance"},
        {"id": "seedream-4.5", "object": "model", "owned_by": "bytedance"},
        {"id": "dreamina-4.5", "object": "model", "owned_by": "bytedance"},
        {"id": "seedance-2.5", "object": "model", "owned_by": "bytedance"},
        {"id": "seedance-2.0", "object": "model", "owned_by": "bytedance"},
    ]
    return {"object": "list", "data": models}


# ──────────────────────────────────────────────
# Health & Credits endpoints
# ──────────────────────────────────────────────



@app.get("/v1/credits")
async def get_credits():
    """Return live credit balance and generations remaining."""
    return await engine.get_credits_info()


# ──────────────────────────────────────────────
# Generation endpoints
# ──────────────────────────────────────────────

@app.post("/v1/images/generations")
async def generate_image_endpoint(req: ImageGenRequest, request: Request):
    try:
        file_paths = await engine.generate_images(prompt=req.prompt, count=req.n or 1, size=req.size)
        base_url = str(request.base_url).rstrip("/")

        if req.response_format == "b64_json":
            data = []
            for file_path in file_paths:
                with open(file_path, "rb") as f:
                    data.append({"b64_json": base64.b64encode(f.read()).decode("utf-8")})
            return {"created": int(time.time()), "data": data}

        data = [{"url": f"{base_url}/outputs/{Path(p).name}"} for p in file_paths]
        return {"created": int(time.time()), "data": data}
    except AuthMissingError as e:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": str(e),
                    "type": "auth_missing",
                    "action": "Please open Firefox and log into https://dreamina.capcut.com",
                }
            },
        )
    except InsufficientCreditsError as e:
        return JSONResponse(
            status_code=402,
            content={
                "error": {
                    "message": str(e),
                    "type": "insufficient_credits",
                    "balance": e.balance,
                    "required": COST_PER_IMAGE,
                }
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    try:
        last_prompt = req.messages[-1].content if req.messages else "a beautiful artwork"
        file_path = await engine.generate_image(prompt=last_prompt)
        base_url = str(request.base_url).rstrip("/")
        file_name = Path(file_path).name
        download_url = f"{base_url}/outputs/{file_name}"
        
        md_content = f"Generated with {req.model}:\n\n![Generated Image]({download_url})\n\n[Download Full Resolution]({download_url})"
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": md_content},
                "finish_reason": "stop"
            }]
        }
    except AuthMissingError as e:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": str(e),
                    "type": "auth_missing",
                    "action": "Please open Firefox and log into https://dreamina.capcut.com",
                }
            },
        )
    except InsufficientCreditsError as e:
        return JSONResponse(
            status_code=402,
            content={
                "error": {
                    "message": str(e),
                    "type": "insufficient_credits",
                    "balance": e.balance,
                    "required": 24,
                }
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/outputs/{filename}")
def serve_output(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT)
