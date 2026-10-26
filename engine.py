"""
Dreamina / Seedream & Seedance Camoufox Automation Engine
"""
import asyncio
import json
import math
import os
import re
import tempfile
import time
import uuid
import random
import hashlib
import logging
from pathlib import Path
from typing import Optional, Any
try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
import httpx
from camoufox.async_api import AsyncCamoufox

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("seedance_engine")

from cookie_sync import (
    get_cached_cookies,
    reload_cookies,
    has_valid_session,
    get_session_id,
    get_accounts,
    sync_accounts,
    update_account_cookies,
)

# ByteDance API constants
DEFAULT_ASSISTANT_ID = "513641"
PLATFORM_CODE = "7"
VERSION_CODE = "8.4.0"
APP_SDK_VERSION = "48.0.0"
WEB_VERSION = "7.5.0"
DA_VERSION = "3.3.28"
DRAFT_VERSION = "3.3.7"

# Studio constants
STUDIO_URL = "https://dreamina.capcut.com/ai-tool/generate"
RECORD_ITEM_SELECTOR = '[class*="agentic-image-record-item"]'
RECORD_IMG_SELECTOR = '[class*="agentic-image-record-item"] img'
SUBMIT_BUTTON_SELECTOR = 'button[data-testid="dreamina.generator.submit"]:visible'

# One generation = 24 credits, however many variants the studio decides to render.
COST_PER_IMAGE = 24
VARIANTS_PER_GENERATION = 4
ASPECT_RATIOS = ("1:1", "9:16", "16:9", "3:4", "4:3", "2:3", "3:2", "9:21", "21:9")


def resolve_aspect_ratio(size: Optional[str]) -> Optional[str]:
    """Map an OpenAI-style `size` to the closest aspect ratio the studio offers.

    Returns None for auto/default so the studio keeps choosing the ratio.
    """
    if not size:
        return None
    value = str(size).strip().lower()
    if value in {"auto", "default", "none"}:
        return None
    if value in ASPECT_RATIOS:
        return value

    match = re.match(r"^(\d+)\s*[x:*]\s*(\d+)$", value)
    if not match:
        return None
    width, height = int(match[1]), int(match[2])
    if width <= 0 or height <= 0:
        return None

    divisor = math.gcd(width, height)
    exact = f"{width // divisor}:{height // divisor}"
    if exact in ASPECT_RATIOS:
        return exact

    target = width / height
    return min(ASPECT_RATIOS, key=lambda r: abs(math.log(target / (int(r.split(':')[0]) / int(r.split(':')[1])))))


def generate_sign(uri: str, device_time: int) -> str:
    """ByteDance MD5 signature generator: 9e2c|uri[-7:]|pf|appvr|time||11ac"""
    raw = f"9e2c|{uri[-7:]}|{PLATFORM_CODE}|{VERSION_CODE}|{device_time}||11ac"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def get_base_params() -> dict:
    return {
        "aid": DEFAULT_ASSISTANT_ID,
        "device_platform": "web",
        "region": "US",
        "da_version": DA_VERSION,
        "os": "windows",
        "web_component_open_flag": "1",
        "web_version": WEB_VERSION,
        "aigc_features": "app_lip_sync",
    }


def is_sg_session(cookies: list[dict]) -> bool:
    """Route to the SG datacenter when the session was minted there.

    Only the IDC cookies are meaningful here; scanning every cookie value for
    "sg" matched random base64 blobs and silently selected the wrong host.
    """
    for c in cookies:
        if c.get("name") in ("cc-target-idc", "store-idc"):
            val = str(c.get("value", "")).lower()
            if "alisg" in val or val.endswith("sg") or val.startswith("sg"):
                return True
    return any("sg" in str(c.get("domain", "")).lower() for c in cookies)


def build_overlay_mask(cv_img: np.ndarray, remove_wordmark: bool = True) -> np.ndarray:
    """Build a glyph-level mask for the studio's corner overlays.

    The studio Download carries two overlays: a small "AI" chip top-left and the
    "DreaminaAI" wordmark bottom-right. Both scale with the image width, so their
    boxes are anchored to the corners and sized from `w`.

    Masking the whole wordmark box smears a wide rectangle of real background
    (very visible on bright or textured renders). Instead the wordmark is
    detected as the pixels that sit brighter than a wide local median, which is
    the watermark's own signature: a light overlay on top of the render. A wide
    kernel is required so solid glyph interiors (the logo mark) are compared
    against the surrounding image rather than against themselves. Where the
    overlay is invisible (white on white) the mask stays empty and the pixels
    are left untouched.
    """
    h, w = cv_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if remove_wordmark:
        # Measured on 1728x2304 renders: x 0.731..0.978w, y (from bottom)
        # 0.021..0.064w. The box adds a small margin around those bounds.
        x0, x1 = int(0.725 * w), int(0.985 * w)
        y0, y1 = max(0, h - int(0.070 * w)), max(0, h - int(0.014 * w))
        if x1 - x0 >= 16 and y1 - y0 >= 16:
            roi = cv2.cvtColor(cv_img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            k = max(3, min(101, (min(roi.shape) // 2) * 2 - 1))
            background = cv2.medianBlur(roi, k)
            diff = roi.astype(np.int16) - background.astype(np.int16)
            glyphs = (diff > 12).astype(np.uint8) * 255
            glyphs = cv2.morphologyEx(glyphs, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            # Solid interiors are invisible to the median estimate; fill holes.
            flood = np.zeros((glyphs.shape[0] + 2, glyphs.shape[1] + 2), np.uint8)
            filled = glyphs.copy()
            cv2.floodFill(filled, flood, (0, 0), 255)
            glyphs = cv2.bitwise_or(glyphs, cv2.bitwise_not(filled))
            glyphs = cv2.dilate(glyphs, np.ones((3, 3), np.uint8), iterations=2)
            mask[y0:y1, x0:x1] = glyphs

    # Top-left "AI" chip: a solid rounded badge, so a tight box is correct.
    # Measured at x/y 0.0145..0.058w on a 1728px render, plus a small margin.
    b0, b1 = int(0.012 * w), int(0.060 * w)
    mask[b0:b1, b0:b1] = 255
    return mask


def clean_and_save_image(raw_bytes: bytes, out_path: Path, remove_wordmark: bool = True) -> str:
    """Strip Dreamina render marks and save a clean PNG."""
    try:
        nparr = np.frombuffer(raw_bytes, np.uint8)
        cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if cv_img is not None:
            h, w = cv_img.shape[:2]
            mask = build_overlay_mask(cv_img, remove_wordmark=remove_wordmark)
            clean_img = cv2.inpaint(cv_img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            cv2.imwrite(str(out_path), clean_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            log.info(f"[+] Watermark removed & saved clean PNG ({w}x{h}, {out_path.stat().st_size} bytes)")
        else:
            with open(out_path, "wb") as f:
                f.write(raw_bytes)
    except Exception as e:
        log.error(f"[!] Inpainting error: {e}, falling back to raw save")
        with open(out_path, "wb") as f:
            f.write(raw_bytes)
    return str(out_path)



class InsufficientCreditsError(Exception):
    def __init__(self, message: str = "Insufficient credits for generation", balance: int = 0):
        super().__init__(message)
        self.balance = balance


class GenerationChargedError(RuntimeError):
    """A paid studio run happened but its assets could not be retrieved.

    Resubmitting here would silently charge a second generation, so the
    failover path must never retry this one.
    """


class AuthMissingError(Exception):
    def __init__(self, message: str = "No Dreamina/CapCut session found in Firefox. Please open Firefox, log in to https://dreamina.capcut.com, and retry."):
        super().__init__(message)


# ──────────────────────────────────────────────
# Session Pool
# ──────────────────────────────────────────────

class SessionPool:
    """
    Manages a pool of Dreamina sessionids loaded from farmer/output/sessions.jsonl.
    Tracks which sessions are exhausted and rotates to the next healthy one.
    """

    def __init__(self, sessions_file: Path, exhausted_file: Path):
        self.sessions_file = sessions_file
        self.exhausted_file = exhausted_file
        self._lock = asyncio.Lock()
        self._sessions: list[dict] = []
        self._exhausted: set[str] = set()
        self._shark_cooldowns: dict[str, float] = {}  # session_id -> timestamp until cooldown expires
        self._strikes: dict[str, int] = {}            # session_id -> consecutive failure count
        self._current_index: int = 0
        self.reload()

    def reload(self):
        """Re-read sessions.jsonl and exhausted.txt from disk."""
        sessions = []
        legacy_count = 0
        if self.sessions_file.exists():
            with open(self.sessions_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            s = json.loads(line)
                            # CRITICAL: Legacy rows without verbatim 'cookies' list must NOT be mixed
                            if "cookies" not in s or not isinstance(s["cookies"], list) or len(s["cookies"]) < 5:
                                s["needs_refarm"] = True
                                legacy_count += 1
                            sessions.append(s)
                        except json.JSONDecodeError:
                            pass

        exhausted = set()
        if self.exhausted_file.exists():
            with open(self.exhausted_file, "r", encoding="utf-8") as f:
                exhausted = set(line.strip() for line in f if line.strip())

        self._sessions = [
            s for s in sessions 
            if s.get("sessionid") not in exhausted and not s.get("needs_refarm")
        ]
        self._exhausted = exhausted
        log.info(f"[pool] Loaded {len(self._sessions)} active sessions "
                 f"({len(sessions)} total, {len(exhausted)} exhausted, {legacy_count} legacy/needs_refarm)")

    def get_session_cookies(self, session_id: str) -> Optional[list[dict]]:
        """Return the verbatim cookies array for a given session ID, or None."""
        for s in self._sessions:
            if s.get("sessionid") == session_id:
                return s.get("cookies")
        return None

    def active_count(self) -> int:
        now = time.time()
        # Count sessions that are not exhausted and not in active cooldown
        available = [s for s in self._sessions if self._shark_cooldowns.get(s.get("sessionid", ""), 0) <= now]
        return len(available)

    def get_current(self) -> Optional[dict]:
        """Return an active session not currently in Shark cooldown."""
        if not self._sessions:
            return None
        now = time.time()
        # Look for the first non-cooldown session starting from _current_index
        n = len(self._sessions)
        for offset in range(n):
            idx = (self._current_index + offset) % n
            candidate = self._sessions[idx]
            sid = candidate.get("sessionid", "")
            if self._shark_cooldowns.get(sid, 0) <= now:
                self._current_index = idx
                return candidate

        # If all sessions are in cooldown, return the one closest to expiry
        log.warning("[pool] All sessions in pool are currently in cooldown! Selecting closest to expiry.")
        return self._sessions[self._current_index % n]

    def get_session(self) -> Optional[str]:
        """Return the active sessionid string, or None if pool is empty."""
        curr = self.get_current()
        return curr.get("sessionid") if curr else None

    def rotate(self) -> Optional[dict]:
        """Move to the next session. Returns the new session or None."""
        if not self._sessions:
            return None
        self._current_index = (self._current_index + 1) % len(self._sessions)
        return self.get_current()

    def mark_shark_cooldown(self, session_id: str, cooldown_seconds: float = 7200) -> bool:
        """
        Puts a session into temporary cooldown (default 2 hours) rather than permanently
        exhausting it. Only permanently exhausts if strikes >= 3 AND the session belongs
        to the pool — foreign sessions (e.g. the live Firefox session) are never
        persistently burned into exhausted.txt.
        Returns True if permanently exhausted, False if placed on cooldown.
        """
        strikes = self._strikes.get(session_id, 0) + 1
        self._strikes[session_id] = strikes

        if strikes >= 3 and session_id in {s["sessionid"] for s in self._sessions}:
            log.error(f"[pool] Session {session_id[:8]}*** reached {strikes} strikes. Permanently exhausting.")
            self.mark_exhausted(session_id)
            return True
        else:
            expiry = time.time() + cooldown_seconds
            self._shark_cooldowns[session_id] = expiry
            log.warning(f"[pool] Session {session_id[:8]}*** Shark cooldown (strike {strikes}/3) for {cooldown_seconds/60:.0f} min.")
            self.rotate()
            return False

    def mark_exhausted(self, session_id: str):
        """Mark a session as exhausted/dead and persist to disk (idempotent)."""
        if session_id in self._exhausted:
            return
        log.warning(f"[pool] Marking session {session_id[:8]}*** as exhausted")
        self._exhausted.add(session_id)
        self._sessions = [s for s in self._sessions if s["sessionid"] != session_id]
        self._current_index = 0
        # Persist
        self.exhausted_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.exhausted_file, "a", encoding="utf-8") as f:
            f.write(session_id + "\n")

    def add_session(self, session: dict):
        """Add a new session to the in-memory pool (also persisted by farmer)."""
        sid = session.get("sessionid", "")
        if sid and sid not in self._exhausted:
            existing_ids = {s["sessionid"] for s in self._sessions}
            if sid not in existing_ids:
                self._sessions.append(session)
                log.info(f"[pool] Added session {sid[:8]}*** — pool size: {len(self._sessions)}")


# ──────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────

class DreaminaEngine:
    def __init__(
        self,
        session_id: str,
        output_dir: Path,
        pool: Optional[SessionPool] = None,
        vault_client: Optional[Any] = None,
    ):
        self.session_id = session_id
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self.refresher = None
        self.pool = pool
        self.vault_client = vault_client
        self._cached_credits: Optional[dict] = None
        self._credits_checked_at: float = 0.0
        # Container accounts: one Firefox container per Dreamina login.
        self._accounts: list[dict] = []
        self._accounts_loaded_at: float = 0.0
        self._credits_by_sid: dict[str, dict] = {}
        self._credits_at_by_sid: dict[str, float] = {}
        self._last_used: dict[str, float] = {}
        self._cooldowns: dict[str, float] = {}

    def update_session_id(self, new_id: str):
        """Hot-swap the active session ID without restarting."""
        log.info(f"DreaminaEngine session ID updated to: {new_id[:8]}***")
        self.session_id = new_id

    def _refresh_accounts(self, force: bool = False) -> list[dict]:
        """Reload the container-account registry (cached for five minutes)."""
        if force or not self._accounts or (time.time() - self._accounts_loaded_at) > 300:
            try:
                self._accounts = get_accounts()
            except Exception as exc:
                log.warning(f"[accounts] Could not load container accounts: {exc}")
            self._accounts_loaded_at = time.time()
        return self._accounts

    def _remember_credits(self, sid: str, info: dict) -> None:
        """Cache a balance per session so account selection does not re-query it."""
        self._credits_by_sid[sid] = info
        self._credits_at_by_sid[sid] = time.time()
        self._cached_credits = info
        self._cached_session_id = sid
        self._credits_checked_at = time.time()

    def _deduct_credits(self, sid: str) -> None:
        """Account for the one generation that was just charged on `sid`."""
        info = self._credits_by_sid.get(sid)
        if not info or "gift_credit" not in info:
            return
        info = dict(info)
        info["gift_credit"] = max(0, info["gift_credit"] - COST_PER_IMAGE)
        info["generations_remaining"] = info["gift_credit"] // COST_PER_IMAGE
        info["status"] = "active" if info["gift_credit"] >= COST_PER_IMAGE else ("ready" if info["gift_credit"] > 0 else "exhausted")
        self._credits_by_sid[sid] = info
        self._credits_at_by_sid[sid] = time.time()
        if self._cached_session_id == sid:
            self._cached_credits = info

    async def _sync_balance_after_run(self, sid: str) -> None:
        """Refresh the balance after a paid run, falling back to local bookkeeping.

        The live balance is authoritative: it already reflects the charge and
        any promo credits that landed during the studio visit. Local deduction
        is only used when the balance cannot be read.
        """
        try:
            await self.get_credits_info(force_refresh=True, session_id=sid)
        except Exception as exc:
            log.warning(f"[accounts] Live balance refresh failed for {sid[:8]}***: {exc}")
            self._deduct_credits(sid)

    def _get_session_id(self) -> str:
        """Return the best available session ID: Firefox cookie_sync first (if not exhausted), then pool, then fallback."""
        ff_sid = get_session_id()
        is_ff_exhausted = False
        if ff_sid:
            if self.pool and ff_sid in self.pool._exhausted:
                is_ff_exhausted = True
            elif getattr(self, "_cached_session_id", None) == ff_sid and self._cached_credits and self._cached_credits.get("gift_credit", 0) < 24:
                is_ff_exhausted = True

        if ff_sid and not is_ff_exhausted:
            return ff_sid
        if self.pool:
            sid = self.pool.get_session()
            if sid:
                return sid
        return ff_sid or self.session_id

    def _can_refresh_from_live_firefox(self, sid: str) -> bool:
        """Only the default Firefox session may be re-read from the live profile.

        Re-injecting the default jar into a container account would silently
        generate on a different account than the one that was selected.
        """
        return bool(sid) and sid == get_session_id()

    def _get_cookies_for_session(self, sid: str) -> Optional[list[dict]]:
        """
        Get isolated, complete cookie jar for the specified session ID.
        Never mix one account's cookies with another account's jar.
        """
        if not sid:
            return None

        for acc in self._refresh_accounts():
            if acc.get("sessionid") == sid and acc.get("cookies"):
                return acc["cookies"]

        # Only the default Firefox session may fall back to the live cookie cache.
        if sid == get_session_id():
            cookies = get_cached_cookies()
            if not has_valid_session(cookies):
                cookies = reload_cookies()
            return cookies if has_valid_session(cookies) else None

        if self.pool:
            pool_cookies = self.pool.get_session_cookies(sid)
            if pool_cookies:
                return pool_cookies

        return None

    async def get_credits_info(self, force_refresh: bool = False, session_id: Optional[str] = None) -> dict:
        now = time.time()
        sid = session_id or self._get_session_id()
        cached = self._credits_by_sid.get(sid)
        if not force_refresh and cached and (now - self._credits_at_by_sid.get(sid, 0.0)) < 600:
            return cached

        cookies = self._get_cookies_for_session(sid)
        if not cookies:
            return {
                "gift_credit": 0,
                "vip_credit": 0,
                "purchase_credit": 0,
                "generations_remaining": 0,
                "status": "auth_missing"
            }

        # FAST PATH: Try REST credit check first (takes ~150ms instead of 15s browser boot)
        try:
            is_sg = is_sg_session(cookies)
            comm_base = "https://commerce-api-sg.capcut.com" if is_sg else "https://commerce.us.capcut.com"
            cookie_str = "; ".join(f"{c['name']}={c['value']}" for c in cookies)
            device_time = int(time.time())
            uri = "/commerce/v1/benefits/user_credit"
            headers = {
                "Accept": "application/json, text/plain, */*",
                "App-Sdk-Version": APP_SDK_VERSION,
                "Appid": DEFAULT_ASSISTANT_ID,
                "Appvr": VERSION_CODE,
                "Content-Type": "application/json",
                "Cookie": cookie_str,
                "Device-Time": str(device_time),
                "Origin": "https://dreamina.capcut.com",
                "Pf": PLATFORM_CODE,
                "Referer": "https://dreamina.capcut.com/",
                "Sign": generate_sign(uri, device_time),
                "Sign-Ver": "1",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:135.0) Gecko/20100101 Firefox/135.0",
            }
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(f"{comm_base}{uri}", headers=headers, json={})
                rj = r.json()
                ret = str(rj.get("ret", ""))
                if ret == "0":
                    c_resp = rj.get("response")
                    gift = 0
                    vip = 0
                    purchase = 0
                    if isinstance(c_resp, str):
                        try:
                            c_dict = json.loads(c_resp)
                            cr = c_dict.get("credit", {})
                            gift = cr.get("gift_credit", 0)
                            vip = cr.get("vip_credit", 0)
                            purchase = cr.get("purchase_credit", 0)
                        except Exception:
                            pass
                    elif "data" in rj and isinstance(rj["data"], dict) and "credit" in rj["data"]:
                        cr = rj["data"]["credit"]
                        gift = cr.get("gift_credit", 0)
                        vip = cr.get("vip_credit", 0)
                        purchase = cr.get("purchase_credit", 0)

                    info = {
                        "gift_credit": gift,
                        "vip_credit": vip,
                        "purchase_credit": purchase,
                        "generations_remaining": gift // COST_PER_IMAGE,
                        "status": "active" if gift >= COST_PER_IMAGE else ("ready" if gift > 0 else "exhausted"),
                        "shark_rejected": False
                    }
                    self._remember_credits(sid, info)
                    log.info(f"[credits] Fast REST check succeeded for {sid[:8]}***: {gift} credits")
                    return info
        except Exception as rest_err:
            log.debug(f"[credits] Fast REST check exception: {rest_err}, falling back to Camoufox")

        async with AsyncCamoufox(headless=True, humanize=True, geoip=False, os="windows") as browser:
            context = await browser.new_context(viewport={"width": 1440, "height": 900})
            await context.add_cookies(cookies)
            page = await context.new_page()

            credit_data = {}
            shark_detected = False
            credit_event = asyncio.Event()

            async def on_response(response):
                nonlocal shark_detected
                url = response.url
                if "credit" in url or "benefit" in url:
                    try:
                        data = await response.json()
                        if isinstance(data, dict):
                            errmsg = data.get("errmsg", "")
                            ret = str(data.get("ret", ""))
                            if "shark" in errmsg.lower() or ret == "34070104" or "shark" in str(data).lower():
                                shark_detected = True
                                log.error("[!] ByteDance Shark risk gate rejected credit operation: %s", errmsg or ret)

                            resp_val = data.get("response")
                            if isinstance(resp_val, str):
                                try:
                                    rj = json.loads(resp_val)
                                    if "credit" in rj:
                                        cr = rj["credit"]
                                        credit_data["gift_credit"] = cr.get("gift_credit", 0)
                                        credit_data["vip_credit"] = cr.get("vip_credit", 0)
                                        credit_data["purchase_credit"] = cr.get("purchase_credit", 0)
                                        credit_event.set()
                                except Exception:
                                    pass
                            elif "data" in data and isinstance(data["data"], dict) and "credit" in data["data"]:
                                cr = data["data"]["credit"]
                                credit_data["gift_credit"] = cr.get("gift_credit", 0)
                                credit_data["vip_credit"] = cr.get("vip_credit", 0)
                                credit_data["purchase_credit"] = cr.get("purchase_credit", 0)
                                credit_event.set()
                            elif "receive_quota" in data:
                                quota = data.get("receive_quota", 0)
                                if str(data.get("ret")) == "0" and quota > 0:
                                    log.info(f"[+] Received daily credit grant: +{quota}")
                                    credit_data["gift_credit"] = quota
                                    credit_event.set()
                            elif "total_credit" in data.get("data", {}):
                                credit_data["gift_credit"] = data["data"].get("total_credit", 0)
                                credit_event.set()
                    except Exception:
                        pass

            page.on("response", on_response)
            try:
                # Use domcontentloaded, avoid networkidle hanging on SPA
                await page.goto("https://dreamina.capcut.com/ai-tool/generate", wait_until="domcontentloaded", timeout=30000)

                # Light humanized cursor movement for entropy
                await asyncio.sleep(1.0)
                await page.mouse.move(random.randint(200, 600), random.randint(200, 500))

                # Wait for credit event
                try:
                    await asyncio.wait_for(credit_event.wait(), timeout=12.0)
                except asyncio.TimeoutError:
                    log.debug("Credit response event timed out, checking DOM badge")

                # DOM fallback for credit count (e.g. "✦ 50 Upgrade")
                if credit_data.get("gift_credit", 0) == 0:
                    dom_credits = await page.evaluate('''() => {
                        const text = document.body.innerText || '';
                        const m = text.match(/✦\\s*(\\d+)/);
                        return m ? parseInt(m[1]) : 0;
                    }''')
                    if dom_credits > 0:
                        credit_data["gift_credit"] = dom_credits
                        log.info(f"[+] Detected {dom_credits} gift credits via DOM badge.")
            except Exception as e:
                log.warning(f"Credit check exception: {e}")

            gift = credit_data.get("gift_credit", 0)
            status_val = "shark_rejected" if (shark_detected and gift < 24) else ("active" if gift >= 24 else "exhausted")
            info = {
                "gift_credit": gift,
                "vip_credit": credit_data.get("vip_credit", 0),
                "purchase_credit": credit_data.get("purchase_credit", 0),
                "generations_remaining": gift // 24,
                "status": status_val,
                "shark_rejected": shark_detected
            }
            self._remember_credits(sid, info)
            return info

    async def generate_images(self, prompt: str, count: int = 1, size: Optional[str] = None) -> list[str]:
        """Run exactly one studio generation and return every image it produced.

        `count` is the caller's expectation, not a target to enforce: the studio
        decides how many variants to render, and a second generation is never
        launched implicitly because each run costs COST_PER_IMAGE credits.
        """
        paths = await self._generate_once(prompt, size=size)
        if count and len(paths) != count:
            log.info(
                f"[engine] Requested n={count}, studio produced {len(paths)} image(s); "
                "returning all produced images without another generation."
            )
        return paths

    async def generate_image(self, prompt: str, size: Optional[str] = None) -> str:
        return (await self.generate_images(prompt, 1, size))[0]

    async def _select_account(self) -> tuple[str, dict]:
        """Pick the account with the most headroom, least recently used first.

        A balance below one generation is not permanent death: the daily grant
        lands when the studio page loads, so up to two accounts are woken up
        before the request is rejected.
        """
        accounts = self._refresh_accounts()
        now = time.time()
        candidates: list[str] = []
        for acc in accounts:
            sid = acc.get("sessionid")
            if not sid or self._cooldowns.get(sid, 0.0) > now:
                continue
            candidates.append(sid)

        ff_sid = get_session_id()
        if ff_sid and ff_sid not in candidates and self._cooldowns.get(ff_sid, 0.0) <= now:
            candidates.insert(0, ff_sid)
        candidates.sort(key=lambda s: self._last_used.get(s, 0.0))

        best_sid: Optional[str] = None
        best_info: Optional[dict] = None
        for sid in candidates:
            info = await self.get_credits_info(session_id=sid)
            if info.get("status") == "auth_missing":
                continue
            if best_info is None or info.get("gift_credit", 0) > best_info.get("gift_credit", 0):
                best_sid, best_info = sid, info
            if info.get("gift_credit", 0) >= COST_PER_IMAGE:
                self._last_used[sid] = now
                log.info(f"[accounts] Using {sid[:8]}*** ({info['gift_credit']} credits)")
                return sid, info

        for sid in candidates[:2]:
            info = await self._activate_account(sid)
            if info and info.get("gift_credit", 0) >= COST_PER_IMAGE:
                self._last_used[sid] = now
                return sid, info

        gift = (best_info or {}).get("gift_credit", 0)
        raise InsufficientCreditsError(
            f"No account has {COST_PER_IMAGE} credits (best: {gift}, accounts: {len(candidates)})",
            balance=gift,
        )

    async def _activate_account(self, sid: str) -> Optional[dict]:
        """Load the studio with this account so its daily grant is credited."""
        cookies = self._get_cookies_for_session(sid)
        if not cookies:
            return None
        log.info(f"[accounts] {sid[:8]}*** has no credits; loading the studio to claim the daily grant...")
        try:
            async with AsyncCamoufox(headless=True, humanize=True, geoip=False, os="windows") as browser:
                context = await browser.new_context(viewport={"width": 1440, "height": 900})
                await context.add_cookies(cookies)
                page = await context.new_page()
                try:
                    await page.goto(STUDIO_URL, wait_until="domcontentloaded", timeout=45000)
                    await page.locator("div.tiptap, div[contenteditable='true']").first.wait_for(state="visible", timeout=30000)
                    await page.evaluate("() => document.querySelectorAll('tiktok-cookie-banner').forEach(e => e.remove())")
                    for label in ("Next", "Got it"):
                        btn = page.locator(f"button:has-text('{label}')").first
                        if await btn.count() and await btn.is_visible():
                            await btn.click()
                            await asyncio.sleep(1.5)
                    gift_btn = page.locator("button[aria-label='Free credits'], [class*='task-center'] button").first
                    if await gift_btn.count() and await gift_btn.is_visible():
                        await gift_btn.click()
                        await asyncio.sleep(2.0)
                        claims = page.locator("button:has-text('Claim'), button:has-text('Collect'), button:has-text('Check in')")
                        for idx in range(await claims.count()):
                            claim = claims.nth(idx)
                            if await claim.is_visible():
                                await claim.click()
                                await asyncio.sleep(1.0)
                    await asyncio.sleep(6.0)
                finally:
                    fresh = await context.cookies()
                    if fresh:
                        update_account_cookies(sid, fresh)
                    await page.close()
        except Exception as exc:
            log.warning(f"[accounts] Activation failed for {sid[:8]}***: {exc}")

        info = await self.get_credits_info(force_refresh=True, session_id=sid)
        log.info(f"[accounts] {sid[:8]}*** balance after activation: {info.get('gift_credit', 0)}")
        return info

    async def _generate_once(self, prompt: str, size: Optional[str] = None) -> list[str]:
        # Serialize selection + generation so two concurrent requests can never
        # pick the same account and launch two paid studio sessions on it.
        async with self._lock:
            return await self._generate_once_locked(prompt, size)

    async def _pick_funded_account(self) -> tuple[str, dict]:
        """Select an account and confirm its balance with a live check.

        Selection reads a ten-minute cache; a run started in the studio outside
        the API can drain an account in the meantime, so the chosen balance is
        re-read once before any credits are committed.
        """
        info: dict = {}
        for _ in range(2):
            sid, _cached = await self._select_account()
            info = await self.get_credits_info(force_refresh=True, session_id=sid)
            if info.get("gift_credit", 0) >= COST_PER_IMAGE:
                return sid, info
            self._cooldowns[sid] = time.time() + 60
        raise InsufficientCreditsError(
            f"No account has {COST_PER_IMAGE} credits after a live balance check",
            balance=info.get("gift_credit", 0),
        )

    async def _generate_once_locked(self, prompt: str, size: Optional[str] = None) -> list[str]:
        sid, info = await self._pick_funded_account()
        self._cached_credits = info
        self._cached_session_id = sid
        pre_gift = info.get("gift_credit", 0)

        try:
            paths = await self._generate_image_internal(prompt, size, session_id=sid)
            await self._sync_balance_after_run(sid)
            return paths
        except Exception as e:
            charged = isinstance(e, GenerationChargedError)
            verified = False
            if not charged:
                # The studio can charge a run without handing back a record
                # (slow render, dropped download). Retry on another account
                # only when the balance proves nothing was charged.
                try:
                    post = await self.get_credits_info(force_refresh=True, session_id=sid)
                    verified = True
                    charged = post.get("gift_credit", 0) <= pre_gift - COST_PER_IMAGE
                except Exception as check_err:
                    log.warning(f"[accounts] Could not verify charge on {sid[:8]}***: {check_err}")
                    charged = True  # fail safe: never risk a second paid run
            if charged:
                log.error(
                    f"Session {sid[:8]}*** spent {COST_PER_IMAGE} credits but produced no images: {e}. "
                    "Not retrying, so no second paid run is launched."
                )
                if not verified:
                    await self._sync_balance_after_run(sid)
                self._cooldowns[sid] = time.time() + 600
                raise

            log.warning(f"Session {sid[:8] if sid else 'unknown'}*** generation failed: {e}. Attempting failover...")
            # An empty account is not a broken one; keep its cooldown short.
            self._cooldowns[sid] = time.time() + (600 if isinstance(e, InsufficientCreditsError) else 7200)
            self._cached_credits = None
            try:
                next_sid, _next_info = await self._select_account()
            except InsufficientCreditsError:
                next_sid = None
            if next_sid and next_sid != sid:
                log.info(f"[accounts] Failing over to {next_sid[:8]}***")
                paths = await self._generate_image_internal(prompt, size, session_id=next_sid)
                await self._sync_balance_after_run(next_sid)
                return paths
            if self.vault_client:
                try:
                    async with self.vault_client.lease("dreamina", credits_used=24.0) as lease:
                        vsid = lease["credentials_data"].get("sessionid") or lease["credentials_data"].get("session_id")
                        return await self._generate_image_internal(prompt, size, session_id=vsid)
                except Exception as ve:
                    log.error(f"Vault failover lease failed: {ve}")
            raise

    async def _generate_image_rest(self, prompt: str, aspect_ratio: str = "1:1", session_id: Optional[str] = None) -> Optional[str]:
        """
        Pure HTTP REST Image Generation Engine (No Browser).
        Submits AIGC draft via POST /mweb/v1/aigc_draft/generate with ByteDance MD5 signing
        and polls /mweb/v1/get_history_by_ids.
        Returns local file path to saved, clean PNG, or None if fallback required.
        """
        sid = session_id or self._get_session_id()
        cookies = self._get_cookies_for_session(sid)
        if not cookies:
            return None

        is_sg = is_sg_session(cookies)
        api_base = "https://mweb-api-sg.capcut.com" if is_sg else "https://dreamina-api.us.capcut.com"
        cookie_str = "; ".join(f"{c['name']}={c['value']}" for c in cookies)

        model = "high_aes_general_v50"  # Image 5.0 Lite / Default
        submit_id = str(uuid.uuid4())
        component_id = str(uuid.uuid4())
        draft_id = str(uuid.uuid4())

        ratio_val = 1
        w, h = 1024, 1024
        if aspect_ratio == "16:9":
            ratio_val = 2
            w, h = 1792, 1024
        elif aspect_ratio == "9:16":
            ratio_val = 3
            w, h = 1024, 1792
        elif aspect_ratio == "4:3":
            ratio_val = 4
            w, h = 1408, 1024
        elif aspect_ratio == "3:4":
            ratio_val = 5
            w, h = 1024, 1408

        payload = {
            "extend": {"root_model": model},
            "submit_id": submit_id,
            "metrics_extra": json.dumps({
                "promptSource": "custom",
                "generateCount": 1,
                "enterFrom": "click",
                "position": "page_bottom_box",
                "isBoxSelect": False,
                "isCutout": False,
                "generateId": submit_id,
                "isRegenerate": False
            }),
            "draft_content": json.dumps({
                "type": "draft",
                "id": draft_id,
                "min_version": "3.0.2",
                "min_features": [],
                "is_from_tsn": True,
                "version": DRAFT_VERSION,
                "main_component_id": component_id,
                "component_list": [{
                    "type": "image_base_component",
                    "id": component_id,
                    "min_version": "3.0.2",
                    "aigc_mode": "workbench",
                    "metadata": {
                        "type": "",
                        "id": str(uuid.uuid4()),
                        "created_platform": 3,
                        "created_platform_version": "",
                        "created_time_in_ms": str(int(time.time() * 1000)),
                        "created_did": ""
                    },
                    "generate_type": "generate",
                    "abilities": {
                        "type": "",
                        "id": str(uuid.uuid4()),
                        "generate": {
                            "type": "",
                            "id": str(uuid.uuid4()),
                            "core_param": {
                                "type": "",
                                "id": str(uuid.uuid4()),
                                "model": model,
                                "prompt": prompt,
                                "negative_prompt": "",
                                "seed": random.randint(1000000000, 3000000000),
                                "sample_strength": 0.5,
                                "image_ratio": ratio_val,
                                "large_image_info": {
                                    "type": "",
                                    "id": str(uuid.uuid4()),
                                    "height": h,
                                    "width": w,
                                    "resolution_type": "1k"
                                },
                                "intelligent_ratio": False
                            }
                        },
                        "gen_option": {
                            "type": "",
                            "id": str(uuid.uuid4()),
                            "generate_all": False
                        }
                    }
                }]
            }),
            "http_common_info": {"aid": int(DEFAULT_ASSISTANT_ID)}
        }

        dt = int(time.time())
        gen_uri = "/mweb/v1/aigc_draft/generate"
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9",
            "App-Sdk-Version": APP_SDK_VERSION,
            "Appid": DEFAULT_ASSISTANT_ID,
            "Appvr": VERSION_CODE,
            "Content-Type": "application/json",
            "Cookie": cookie_str,
            "Device-Time": str(dt),
            "Origin": "https://dreamina.capcut.com",
            "Pf": PLATFORM_CODE,
            "Referer": "https://dreamina.capcut.com/",
            "Sign": generate_sign(gen_uri, dt),
            "Sign-Ver": "1",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:135.0) Gecko/20100101 Firefox/135.0",
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            log.info(f"[rest_gen] Submitting prompt via pure REST ({model}) to {api_base}{gen_uri}...")
            r = await client.post(f"{api_base}{gen_uri}", params=get_base_params(), headers=headers, json=payload)
            rj = r.json()
            ret = str(rj.get("ret", ""))
            if ret != "0":
                log.warning(f"[rest_gen] REST submission rejected (ret={ret}, errmsg={rj.get('errmsg')})")
                return None

            data = rj.get("data") or {}
            aigc_data = data.get("aigc_data") or {}
            history_id = aigc_data.get("history_record_id")
            if not history_id:
                return None

            log.info(f"[rest_gen] Submitted successfully! History ID: {history_id}. Polling...")
            poll_uri = "/mweb/v1/get_history_by_ids"
            for attempt in range(1, 40):
                await asyncio.sleep(2.0)
                p_dt = int(time.time())
                headers["Device-Time"] = str(p_dt)
                headers["Sign"] = generate_sign(poll_uri, p_dt)
                poll_resp = await client.post(
                    f"{api_base}{poll_uri}",
                    params=get_base_params(),
                    headers=headers,
                    json={"history_ids": [history_id]}
                )
                p_json = poll_resp.json()
                h_info = (p_json.get("data") or {}).get(history_id, {})
                status = h_info.get("status")
                item_list = h_info.get("item_list", [])

                if status == 10 and item_list:
                    chosen = item_list[0]
                    large_imgs = chosen.get("image", {}).get("large_images", [])
                    img_url = large_imgs[0].get("image_url") if large_imgs else chosen.get("image_url")
                    if img_url:
                        log.info(f"[rest_gen] Asset ready ({attempt*2}s): {img_url[:80]}...")
                        img_resp = await client.get(img_url, headers={"User-Agent": "Mozilla/5.0"})
                        out_path = self.output_dir / f"dreamina_{int(time.time())}.png"
                        clean_path = clean_and_save_image(img_resp.content, out_path)
                        if self._cached_credits and "gift_credit" in self._cached_credits:
                            self._cached_credits["gift_credit"] = max(0, self._cached_credits["gift_credit"] - COST_PER_IMAGE)
                            self._cached_credits["generations_remaining"] = self._cached_credits["gift_credit"] // COST_PER_IMAGE
                        return clean_path
                elif status == 30:
                    log.error(f"[rest_gen] Task failed: code {h_info.get('fail_code')}")
                    return None

            return None

    async def _generate_image_internal(self, prompt: str, size: Optional[str] = None, session_id: Optional[str] = None) -> list[str]:
        sid = session_id or self._get_session_id()
        aspect_ratio = resolve_aspect_ratio(size)
        cookies = self._get_cookies_for_session(sid)
        if not cookies:
            raise AuthMissingError(f"No valid session cookies available for session {sid[:8] if sid else 'None'}***")

        if sid != getattr(self, "_cached_session_id", None):
            self._cached_credits = None
            self._cached_session_id = sid

        log.info(f"Using Dreamina session: {sid[:8]}*** (total {len(cookies)} cookies injected)")

        # The REST draft endpoint is Shark-gated and charges the run even when its
        # result cannot be fetched, which previously led to a paid job followed by
        # a second paid browser submission. It is never used implicitly.
        log.info("[engine] REST submission disabled; generating through the studio UI.")
        return await self._generate_image_browser(prompt, cookies, sid, aspect_ratio)

    async def _apply_aspect_ratio(self, page, aspect_ratio: str) -> None:
        """Lock the requested aspect ratio in the studio settings panel.

        The composer runs in Auto mode by default, where the ratio radio group
        is disabled. The Auto switch must be turned off before a ratio can be
        selected. Raises if the selection does not stick, so a generation is
        never submitted (and charged) with an unexpected ratio.
        """
        panel_open = await page.evaluate(
            """() => Array.from(document.querySelectorAll('[class*=title]'))
                .some(e => (e.innerText || '').trim() === 'Aspect ratio')"""
        )
        if not panel_open:
            trigger = page.locator(
                '[data-testid="dreamina.generator"] button:has-text("Auto"), '
                '[data-testid="dreamina.generator"] button:has-text("Custom")'
            ).first
            await trigger.wait_for(state="visible", timeout=15000)
            await trigger.click()
            await asyncio.sleep(0.8)

        switch = page.locator('button[role="switch"]').first
        await switch.wait_for(state="visible", timeout=10000)
        if await switch.get_attribute("aria-checked") == "true":
            await switch.click()
            await asyncio.sleep(0.6)

        await page.locator(f'label.lv-radio:has(input[value="{aspect_ratio}"])').first.click()
        await asyncio.sleep(0.6)

        selected = await page.evaluate(
            """() => {
            const title = Array.from(document.querySelectorAll('[class*=title]'))
                .find(e => (e.innerText || '').trim() === 'Aspect ratio');
            const group = title ? title.parentElement.querySelector('[role=radiogroup]') : null;
            const checked = group ? group.querySelector('input:checked') : null;
            return checked ? checked.value : null;
        }"""
        )
        if selected != aspect_ratio:
            raise RuntimeError(f"Studio did not accept aspect ratio {aspect_ratio} (selected: {selected!r})")

        # The popup overlaps the composer, so close it before typing. Clicking
        # the editor here would hit the popup instead and time out.
        await page.keyboard.press("Escape")
        await asyncio.sleep(0.6)
        still_open = await page.evaluate(
            """() => Array.from(document.querySelectorAll('[class*=title]'))
                .some(e => (e.innerText || '').trim() === 'Aspect ratio')"""
        )
        if still_open:
            await page.mouse.click(700, 300)
            await asyncio.sleep(0.5)

    async def _restore_auto_settings(self, page) -> None:
        """Best-effort: put the composer back into Auto mode after a run."""
        trigger = page.locator('[data-testid="dreamina.generator"] button:has-text("Custom")').first
        if not await trigger.count():
            return
        await trigger.click()
        await asyncio.sleep(0.8)
        switch = page.locator('button[role="switch"]').first
        if await switch.count() and await switch.get_attribute("aria-checked") == "false":
            await switch.click()
            await asyncio.sleep(0.5)
        await page.keyboard.press("Escape")
        await asyncio.sleep(0.3)

    async def _generate_image_browser(self, prompt: str, cookies: list[dict], sid: str, aspect_ratio: Optional[str] = None) -> list[str]:
        """Render a prompt in the Dreamina studio and pull the finished assets.

        Each generation appends records to the agent grid; every record's detail
        view exposes a Download button that yields the full-resolution 2K asset,
        while the inline <img> is only a small preview (270x360 on 1728x2304
        renders) and is used solely as a last-resort fallback.
        """
        async with AsyncCamoufox(headless=True, humanize=True, geoip=False, os="windows") as browser:
            context = await browser.new_context(viewport={"width": 1440, "height": 900}, accept_downloads=True)
            await context.add_cookies(cookies)
            page = await context.new_page()

            async def on_response(response):
                if "credit" in response.url or "benefit" in response.url:
                    try:
                        data = await response.json()
                        if isinstance(data, dict):
                            resp_val = data.get("response")
                            cr = None
                            if isinstance(resp_val, str):
                                try:
                                    rj = json.loads(resp_val)
                                    if "credit" in rj:
                                        cr = rj["credit"]
                                except Exception:
                                    pass
                            elif "data" in data and isinstance(data["data"], dict) and "credit" in data["data"]:
                                cr = data["data"]["credit"]

                            if cr:
                                gift = cr.get("gift_credit", 0)
                                self._remember_credits(sid, {
                                    "gift_credit": gift,
                                    "vip_credit": cr.get("vip_credit", 0),
                                    "purchase_credit": cr.get("purchase_credit", 0),
                                    "generations_remaining": gift // COST_PER_IMAGE,
                                    "status": "active" if gift >= COST_PER_IMAGE else ("ready" if gift > 0 else "exhausted"),
                                    "shark_rejected": False,
                                })
                    except Exception:
                        pass
            page.on("response", on_response)

            await page.goto(STUDIO_URL, wait_until="domcontentloaded", timeout=45000)

            editor = page.locator("div.tiptap, div[contenteditable='true']").first
            try:
                await editor.wait_for(state="visible", timeout=25000)
            except Exception:
                raise RuntimeError("Dreamina studio editor never rendered. The session may be logged out or the page is blocked.")

            # The consent banner overlays the composer and swallows clicks.
            await page.evaluate("() => document.querySelectorAll('tiktok-cookie-banner').forEach(e => e.remove())")

            is_unauth = await page.evaluate("""() => !!document.querySelector('.login-button-cWDLCG, .login-button-wrapper-kXam7D')""")
            if is_unauth:
                # Only the default Firefox session can be refreshed from the live
                # profile. Re-injecting those cookies into a container account
                # would silently generate on the wrong account.
                if not self._can_refresh_from_live_firefox(sid):
                    raise AuthMissingError(
                        f"Container account {sid[:8]}*** is logged out. Log into Dreamina again in its Firefox container."
                    )
                log.warning("Detected unauthenticated studio page. Reloading fresh cookies from Firefox and retrying...")
                fresh_cookies = reload_cookies()
                if not has_valid_session(fresh_cookies):
                    raise AuthMissingError()
                await context.clear_cookies()
                await context.add_cookies(fresh_cookies)
                await page.goto(STUDIO_URL, wait_until="domcontentloaded", timeout=45000)
                await asyncio.sleep(2)
                is_unauth = await page.evaluate("""() => !!document.querySelector('.login-button-cWDLCG, .login-button-wrapper-kXam7D')""")
                if is_unauth:
                    raise AuthMissingError("Dreamina session remains unauthenticated even after fresh cookie reload. Please log into https://dreamina.capcut.com in Firefox.")
                await page.evaluate("() => document.querySelectorAll('tiktok-cookie-banner').forEach(e => e.remove())")

            # Loading the studio page is what grants the daily credits, so the live
            # badge beats whatever balance was read before launching the browser.
            await asyncio.sleep(1.5)
            dom_credits = await page.evaluate(r"""() => {
                const m = (document.body.innerText || '').match(/✦\s*(\d+)/);
                return m ? parseInt(m[1]) : 0;
            }""")
            if dom_credits > 0:
                self._remember_credits(sid, {
                    "gift_credit": dom_credits,
                    "vip_credit": 0,
                    "purchase_credit": 0,
                    "generations_remaining": dom_credits // COST_PER_IMAGE,
                    "status": "active" if dom_credits >= COST_PER_IMAGE else ("ready" if dom_credits > 0 else "exhausted"),
                    "shark_rejected": False,
                })
                log.info(f"[browser_gen] Live balance on studio page: {dom_credits} credits")

            if self._cached_credits and self._cached_credits.get("gift_credit", 0) < COST_PER_IMAGE:
                gift = self._cached_credits.get("gift_credit", 0)
                raise InsufficientCreditsError(f"Insufficient credits: {gift} < {COST_PER_IMAGE} required", balance=gift)

            if aspect_ratio:
                await self._apply_aspect_ratio(page, aspect_ratio)
                log.info(f"[browser_gen] Aspect ratio locked to {aspect_ratio}")

            baseline = await page.evaluate("""() => Array.from(document.querySelectorAll('[class*="agentic-image-record-item"] img')).map(i => i.src)""")
            log.info(f"[browser_gen] Existing records on page: {len(baseline)}")

            await editor.click()
            await asyncio.sleep(0.3)
            await page.keyboard.type(prompt, delay=8)
            await asyncio.sleep(0.6)
            typed_len = await page.evaluate(
                """() => {
                const ed = document.querySelector('div.tiptap, div[contenteditable=true]');
                return ed ? (ed.textContent || '').trim().length : -1;
            }"""
            )
            log.info(f"[browser_gen] Editor holds {typed_len} chars before submit")

            submit = page.locator(SUBMIT_BUTTON_SELECTOR).first
            try:
                await submit.wait_for(state="visible", timeout=15000)
                await submit.click()
            except Exception as e:
                log.warning(f"[browser_gen] Submit button not clickable ({e}); falling back to Enter")
                await page.keyboard.press("Enter")
            await asyncio.sleep(2.0)
            overlay = await page.evaluate(
                """() => Array.from(document.querySelectorAll('[role=dialog],[class*=modal]'))
                    .filter(e => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; })
                    .map(e => (e.innerText || '').replace(/\\s+/g, ' ').slice(0, 200))
                    .filter(Boolean).slice(0, 3)"""
            )
            if overlay:
                log.warning(f"[browser_gen] Visible overlay after submit: {overlay}")

            log.info("[browser_gen] Prompt submitted, waiting for agent records...")
            fresh: list[str] = []
            last_count = 0
            quiet_until: Optional[float] = None
            accepted_fallback = False
            deadline = time.time() + 240
            while time.time() < deadline:
                await asyncio.sleep(2.5)
                # Turning Auto off pins the VIP-only 5.0 Pro model, so the agent
                # asks for a downgrade confirmation before spending credits.
                if not accepted_fallback:
                    cont = page.locator(
                        'button:has-text("Continue"):visible, [role="button"]:has-text("Continue"):visible'
                    ).first
                    if await cont.count():
                        await cont.click()
                        accepted_fallback = True
                        log.info("[browser_gen] Accepted the non-VIP model fallback (Continue)")
                        continue
                current = await page.evaluate("""() => Array.from(document.querySelectorAll('[class*="agentic-image-record-item"] img')).map(i => i.src)""")
                fresh = [s for s in current if s and s not in baseline]
                if len(fresh) > last_count:
                    last_count = len(fresh)
                    # Keep collecting until the grid has been quiet for a while;
                    # a variant that lands late is still part of the same paid run.
                    quiet_until = time.time() + 30
                    log.info(f"[browser_gen] {last_count} variant(s) visible so far...")
                if len(fresh) >= VARIANTS_PER_GENERATION:
                    break
                if fresh and quiet_until and time.time() >= quiet_until:
                    break

            if not fresh:
                try:
                    shot = Path(tempfile.gettempdir()) / f"dreamina_timeout_{int(time.time())}.png"
                    await page.screenshot(path=str(shot))
                    log.warning(f"[browser_gen] Timeout screenshot: {shot}")
                except Exception:
                    pass
                raise RuntimeError("No rendered image record appeared on the Dreamina studio page within 240s.")
            log.info(f"[browser_gen] {len(fresh)} variant(s) rendered.")

            ts = int(time.time())
            paths: list[str] = []
            for idx, src in enumerate(fresh):
                out_path = self.output_dir / f"dreamina_{ts}_{idx}.png"
                saved = False

                try:
                    grid = page.locator(RECORD_IMG_SELECTOR)
                    dom_srcs = await page.evaluate("""() => Array.from(document.querySelectorAll('[class*="agentic-image-record-item"] img')).map(i => i.src)""")
                    pos = dom_srcs.index(src)
                    card = grid.nth(pos)
                    await card.scroll_into_view_if_needed()
                    # A card that is still animating in can swallow the click, so
                    # the detail panel never opens; one forced retry recovers it.
                    try:
                        await card.click(timeout=20000)
                    except Exception:
                        await asyncio.sleep(1.5)
                        await card.click(force=True, timeout=20000)

                    download_btn = page.locator('button:has-text("Download")').first
                    try:
                        await download_btn.wait_for(state="visible", timeout=25000)
                    except Exception:
                        await card.click(force=True, timeout=15000)
                        await download_btn.wait_for(state="visible", timeout=25000)
                    async with page.expect_download(timeout=90000) as dl_info:
                        await download_btn.click()
                    dl = await dl_info.value
                    raw = Path(await dl.path()).read_bytes()
                    clean_and_save_image(raw, out_path, remove_wordmark=True)
                    saved = True
                    log.info(f"[browser_gen] Variant {idx} saved from full-res download ({Path(dl.suggested_filename).name})")
                except Exception as e:
                    log.warning(f"[browser_gen] Full-res download failed for variant {idx}: {e}")

                # Escape closes the detail view and returns to the record grid.
                try:
                    await page.keyboard.press("Escape")
                    await asyncio.sleep(0.8)
                except Exception:
                    pass

                if not saved:
                    # Last resort: the studio's preview URL carries only the badge.
                    try:
                        async with httpx.AsyncClient(timeout=45.0) as client:
                            r = await client.get(src, headers={"User-Agent": "Mozilla/5.0"})
                            r.raise_for_status()
                        clean_and_save_image(r.content, out_path, remove_wordmark=False)
                        preview = cv2.imread(str(out_path))
                        dims = f"{preview.shape[1]}x{preview.shape[0]}" if preview is not None else "unknown size"
                        log.warning(
                            f"[browser_gen] Variant {idx} saved from preview fallback "
                            f"({dims}; full-res download failed)."
                        )
                        saved = True
                    except Exception as e:
                        log.error(f"[browser_gen] Could not fetch variant {idx}: {e}")

                if saved:
                    paths.append(str(out_path))

            if aspect_ratio:
                try:
                    await self._restore_auto_settings(page)
                except Exception as exc:
                    log.warning(f"[browser_gen] Could not restore Auto settings: {exc}")

            if not paths:
                raise GenerationChargedError(
                    "Dreamina rendered records but no asset could be downloaded."
                )

        log.info(f"[browser_gen] Saved {len(paths)} image(s): {[Path(p).name for p in paths]}")
        return paths
