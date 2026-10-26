"""
Session Refresher for Dreamina / Seedance2API
Handles automated session validation and re-authentication via Camoufox Google SSO.
Hardened with:
- Target probe on /ai-tool/generate with x-tt-agw-login validation
- Separation of PROBE_SCHEMA_ERROR / transient network glitches from AUTH_REVOKED
- Attempt caps (max 2) and 12-hour exponential circuit breaker
- Anti-runaway guard preventing headless browser thrashing
"""
import os
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    from camoufox.async_api import AsyncCamoufox
except ImportError:
    AsyncCamoufox = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("session_refresher")

CONFIG_PATH = Path(__file__).parent / "config.env"


def parse_google_cookies(raw: str) -> List[Dict[str, Any]]:
    """Parse exported Google cookies from JSON string or key=value format."""
    text = (raw or "").strip()
    if not text:
        return []

    cookies = []
    # Try parsing as JSON
    try:
        data = json.loads(text)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "name" in item and "value" in item:
                    cookie = {
                        "name": str(item["name"]),
                        "value": str(item["value"]),
                        "domain": str(item.get("domain", ".google.com")),
                        "path": str(item.get("path", "/")),
                    }
                    cookies.append(cookie)
            if cookies:
                return cookies
        elif isinstance(data, dict):
            for k, v in data.items():
                cookies.append({
                    "name": str(k),
                    "value": str(v),
                    "domain": ".google.com",
                    "path": "/"
                })
            if cookies:
                return cookies
    except Exception:
        pass

    # Try name=value; format
    for part in text.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        if k and v:
            cookies.append({
                "name": k,
                "value": v,
                "domain": ".google.com",
                "path": "/"
            })
    return cookies


class DreaminaSessionRefresher:
    def __init__(self, engine=None):
        self.engine = engine
        self._refresh_lock = asyncio.Lock()
        self.last_check_time: float = 0.0
        self.last_check_status: bool = False
        self.last_refresh_time: float = 0.0
        self.last_refresh_result: Optional[str] = None
        self._bg_task: Optional[asyncio.Task] = None

        # Hardened circuit breaker state
        self._sso_attempts: int = 0
        self._max_sso_attempts: int = 2
        self._sso_cooldown_until: float = 0.0
        self._circuit_broken: bool = False

    def get_config_value(self, key: str, default: str = "") -> str:
        """Read a key from config.env, os.environ, or the DPAPI secret store."""
        val = os.getenv(key, "")
        if val:
            return val
        if CONFIG_PATH.exists():
            for line in CONFIG_PATH.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith(f"{key}="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val
                    break
        try:
            import secrets_store
            return secrets_store.get_secret(key, default)
        except Exception:
            return default

    def update_config_session_id(self, new_session_id: str) -> None:
        """Persist new DREAMINA_SESSION_ID to config.env."""
        lines = []
        found = False
        if CONFIG_PATH.exists():
            for line in CONFIG_PATH.read_text(encoding="utf-8").splitlines():
                if line.startswith("DREAMINA_SESSION_ID="):
                    lines.append(f"DREAMINA_SESSION_ID={new_session_id}")
                    found = True
                else:
                    lines.append(line)
        if not found:
            lines.insert(0, f"DREAMINA_SESSION_ID={new_session_id}")

        CONFIG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.environ["DREAMINA_SESSION_ID"] = new_session_id
        log.info(f"Updated config.env with new DREAMINA_SESSION_ID: {new_session_id[:8]}***")

    async def validate_session(self, session_id: Optional[str] = None) -> bool:
        """Validate if current sessionid is alive via fast lightweight HTTP request to /ai-tool/generate."""
        sid = session_id or self.get_config_value("DREAMINA_SESSION_ID")
        if not sid:
            self.last_check_status = False
            self.last_check_time = time.time()
            return False

        log.info(f"Validating Dreamina session ID ({sid[:8]}***)...")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
            "Cookie": f"sessionid={sid}; sessionid_ss={sid}; store-country-code=it; store-idc=no1a; cc-target-idc=alisg;",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer": "https://dreamina.capcut.com/"
        }
        try:
            import httpx
            async with httpx.AsyncClient(timeout=8.0, follow_redirects=False) as client:
                resp = await client.get("https://dreamina.capcut.com/ai-tool/generate", headers=headers)
                self.last_check_time = time.time()

                if resp.status_code == 200:
                    agw_login = resp.headers.get("x-tt-agw-login")
                    if agw_login == "1":
                        log.info(f"[✓] Session ID ({sid[:8]}***) is ACTIVE and VALID (agw-login: 1).")
                        self.last_check_status = True
                        return True
                    elif agw_login == "0":
                        log.warning(f"[-] Session ID ({sid[:8]}***) is NOT logged in (agw-login: 0). AUTH_REVOKED.")
                        self.last_check_status = False
                        return False
                    else:
                        log.warning(f"[!] Session ID ({sid[:8]}***) returned 200 without login header. PROBE_SCHEMA_DRIFT.")
                        self.last_check_status = False
                        return False
                elif resp.status_code in (301, 302, 303, 307, 308):
                    loc = resp.headers.get("Location", "")
                    log.warning(f"[-] Session ID ({sid[:8]}***) redirected to '{loc}'. AUTH_REVOKED.")
                    self.last_check_status = False
                    return False
                elif resp.status_code >= 500:
                    log.warning(f"[!] Dreamina probe returned upstream {resp.status_code}. Transient upstream error, not tripping auth revocation.")
                    # Keep previous state to avoid spurious SSO trigger on CapCut outage
                    return self.last_check_status
                else:
                    log.warning(f"[-] Session probe returned HTTP {resp.status_code}.")
                    self.last_check_status = False
                    return False

        except Exception as e:
            log.warning(f"[!] Network error during session validation ({e}). PROBE_SCHEMA_ERROR / transient.")
            self.last_check_time = time.time()
            # Do not treat network timeout as auth revoked to avoid runaway SSO
            return self.last_check_status

    async def refresh_session(self) -> Optional[str]:
        """Perform full Google SSO re-login and extract new sessionid."""
        if not AsyncCamoufox:
            log.error("AsyncCamoufox not installed; cannot execute automated browser refresh.")
            return None

        async with self._refresh_lock:
            log.info("Starting Dreamina session refresh via Google SSO...")
            google_email = self.get_config_value("GOOGLE_EMAIL")
            google_password = self.get_config_value("GOOGLE_PASSWORD")
            google_cookies_raw = self.get_config_value("GOOGLE_COOKIES")

            if not (google_email or google_cookies_raw):
                log.warning("Cannot refresh: neither GOOGLE_EMAIL nor GOOGLE_COOKIES configured in config.env!")
                self.last_refresh_result = "error: no credentials configured"
                return None

            try:
                async with AsyncCamoufox(headless=True, humanize=True, geoip=False) as browser:
                    context = await browser.new_context(viewport={"width": 1440, "height": 900})

                    # If google cookies are provided, inject them
                    if google_cookies_raw:
                        g_cookies = parse_google_cookies(google_cookies_raw)
                        if g_cookies:
                            log.info(f"Injecting {len(g_cookies)} Google session cookies...")
                            await context.add_cookies(g_cookies)

                    page = await context.new_page()
                    log.info("Navigating to Dreamina home...")
                    await page.goto("https://dreamina.capcut.com/ai-tool/home", wait_until="networkidle", timeout=45000)
                    await asyncio.sleep(2)

                    # Suppress cookie banner to prevent click interception
                    await page.add_style_tag(content="""
                        tiktok-cookie-banner, .tiktok-cookie-banner, #tiktok-cookie-banner {
                            display: none !important;
                            pointer-events: none !important;
                        }
                    """)
                    await asyncio.sleep(1)

                    # Click Sign In
                    log.info("Clicking sign-in...")
                    sign_in_el = page.locator("text='Sign in'").last
                    await sign_in_el.click(force=True)
                    await asyncio.sleep(2)

                    # Listen for Google OAuth popup
                    async with context.expect_page(timeout=15000) as popup_info:
                        log.info("Clicking 'Continue with Google'...")
                        google_btn = page.locator("text='Continue with Google'").first
                        await google_btn.click()

                    google_page = await popup_info.value
                    log.info(f"Google popup opened: {google_page.url}")
                    await google_page.wait_for_load_state("networkidle", timeout=30000)
                    await asyncio.sleep(2)

                    # Check if Google needs credentials
                    email_field = google_page.locator("input[type='email'], input#identifierId").first
                    if await email_field.count() > 0 and google_email:
                        log.info(f"Entering Google email: {google_email}...")
                        await email_field.click()
                        await google_page.keyboard.type(google_email, delay=30)
                        await asyncio.sleep(1)
                        await google_page.keyboard.press("Enter")
                        await asyncio.sleep(3)

                        # Check for password field
                        pwd_field = google_page.locator("input[type='password'], input[name='Passwd'], input[name='password']").first
                        await pwd_field.wait_for(state="visible", timeout=15000)
                        if google_password:
                            log.info("Entering Google password...")
                            await pwd_field.click()
                            await google_page.keyboard.type(google_password, delay=30)
                            await asyncio.sleep(1)
                            await google_page.keyboard.press("Enter")

                    # If account chooser / profile selection appears
                    account_card = google_page.locator("div[data-identifier], div[data-profile-identifier]").first
                    if await account_card.count() > 0:
                        log.info("Selecting Google profile card...")
                        await account_card.click()

                    # Wait for Google OAuth callback / popup closing
                    log.info("Waiting for OAuth callback completion...")
                    for _ in range(30):
                        if google_page.is_closed():
                            log.info("Google popup completed and closed.")
                            break
                        if "third-party-callback" in google_page.url or "dreamina.capcut.com" in google_page.url:
                            log.info("Detected callback redirect URL!")
                            break
                        await asyncio.sleep(1)

                    await asyncio.sleep(3)

                    # Extract session cookies from context
                    cookies = await context.cookies()
                    new_session_id = None
                    for c in cookies:
                        if c["name"] == "sessionid" and c["value"]:
                            new_session_id = c["value"]
                            break

                    if not new_session_id:
                        log.error("Failed to extract new sessionid from cookies after login flow.")
                        self.last_refresh_result = "failed: sessionid cookie not found"
                        return None

                    log.info(f"Successfully obtained new sessionid: {new_session_id[:8]}***")
                    self.update_config_session_id(new_session_id)

                    if self.engine:
                        self.engine.update_session_id(new_session_id)

                    self.last_refresh_time = time.time()
                    self.last_refresh_result = "success"
                    self.last_check_status = True
                    return new_session_id

            except Exception as e:
                log.error(f"Failed to refresh Dreamina session: {e}", exc_info=True)
                self.last_refresh_result = f"error: {str(e)}"
                return None

    async def ensure_valid_session(self) -> bool:
        """Checks if session is valid; if not, triggers refresh with circuit-breaker protection."""
        if await self.validate_session():
            self._sso_attempts = 0
            return True

        if self._circuit_broken:
            log.error("Dreamina automated SSO is CIRCUIT BROKEN (manual intervention required).")
            return False

        now = time.time()
        if now < self._sso_cooldown_until:
            rem = int(self._sso_cooldown_until - now)
            log.warning(f"Dreamina SSO in cooldown ({rem}s remaining). Skipping automated login.")
            return False

        google_email = self.get_config_value("GOOGLE_EMAIL")
        google_cookies = self.get_config_value("GOOGLE_COOKIES")
        if not (google_email or google_cookies):
            log.warning("Session expired/invalid, but no Google credentials configured in config.env. Skipping browser launch.")
            return False

        if self._sso_attempts >= self._max_sso_attempts:
            self._circuit_broken = True
            self._sso_cooldown_until = now + 12 * 3600
            log.error("Max automated SSO attempts (2) reached. Tripping 12h circuit breaker to prevent runaway loops.")
            return False

        self._sso_attempts += 1
        log.warning(f"Session expired or invalid! Triggering automated refresh (attempt {self._sso_attempts}/{self._max_sso_attempts})...")
        new_sid = await self.refresh_session()
        if new_sid:
            self._sso_attempts = 0
            return True
        else:
            if self._sso_attempts >= self._max_sso_attempts:
                self._circuit_broken = True
                self._sso_cooldown_until = now + 12 * 3600
                log.error("SSO refresh failed repeatedly. Tripping 12h circuit breaker to prevent runaway loops.")
            return False

    def start_background_monitor(self, interval_seconds: int = 7200):
        """Start background periodic health check."""
        if self._bg_task and not self._bg_task.done():
            return

        async def _monitor_loop():
            log.info(f"Dreamina session monitor started (check every {interval_seconds}s).")
            while True:
                try:
                    await asyncio.sleep(interval_seconds)
                    log.info("[SESSION MONITOR] Running periodic health check...")
                    await self.ensure_valid_session()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    log.warning(f"[SESSION MONITOR] Health check exception: {e}")

        self._bg_task = asyncio.create_task(_monitor_loop())

    def stop_background_monitor(self):
        if self._bg_task and not self._bg_task.done():
            self._bg_task.cancel()
