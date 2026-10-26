"""
Dreamina / CapCut Firefox Cookie Auto-Sync & Session Extractor.

Auto-detects active Firefox profile, queries moz_cookies immutably,
and exports all .capcut.com and dreamina.capcut.com cookies into
Playwright-compatible format for Camoufox injection.
"""
import configparser
import json
import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("dreamina_cookie_sync")

_COOKIE_CACHE_FILE = Path(__file__).parent / "cookies_cache.json"
_ACCOUNTS_CACHE_FILE = Path(__file__).parent / "accounts_cache.json"
_CONTAINER_ATTR = re.compile(r"^\^userContextId=(\d+)$")


def detect_firefox_profile_path() -> Optional[Path]:
    """Auto-detect active Firefox profile path via profiles.ini or glob."""
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None

    firefox_dir = Path(appdata) / "Mozilla" / "Firefox"
    profiles_ini = firefox_dir / "profiles.ini"

    if profiles_ini.exists():
        try:
            config = configparser.ConfigParser()
            config.read(profiles_ini, encoding="utf-8")
            for section in config.sections():
                if config.has_option(section, "Default") and config.get(section, "Default") == "1":
                    path_val = config.get(section, "Path", fallback="")
                    is_rel = config.get(section, "IsRelative", fallback="1") == "1"
                    candidate = (firefox_dir / path_val) if is_rel else Path(path_val)
                    if (candidate / "cookies.sqlite").exists():
                        return candidate
        except Exception as e:
            log.warning(f"Failed parsing profiles.ini: {e}")

    # Fallback: search for *.default-release or any profile with cookies.sqlite
    candidates = list(firefox_dir.glob("Profiles/*.default-release"))
    if not candidates:
        candidates = list(firefox_dir.glob("Profiles/*"))

    for c in candidates:
        if (c / "cookies.sqlite").exists():
            return c

    return None


def _read_capcut_cookie_rows() -> List[Tuple]:
    """Read every CapCut cookie row, including Firefox container (originAttributes) rows.

    Tries a read-only connection first so cookies still sitting in the WAL (Firefox
    open) are visible, then falls back to the immutable URI used previously.
    """
    profile_dir = detect_firefox_profile_path()
    if not profile_dir:
        log.warning("No Firefox profile found.")
        return []

    db_path = profile_dir / "cookies.sqlite"
    if not db_path.exists():
        log.warning(f"cookies.sqlite not found in {profile_dir}")
        return []

    query = (
        "SELECT host, name, value, path, expiry, isSecure, isHttpOnly, sameSite, originAttributes "
        "FROM moz_cookies WHERE host LIKE '%capcut.com%'"
    )
    for label, uri in (
        ("mode=ro", f"file:{db_path.as_posix()}?mode=ro"),
        ("immutable", f"file:{db_path.as_posix()}?immutable=1"),
    ):
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=5)
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()
            return rows
        except Exception as e:
            log.debug(f"Cookie read via {label} failed: {e}")
    log.error("Could not read cookies.sqlite in any mode.")
    return []


def _rows_to_cookies(rows) -> List[Dict]:
    same_site_map = {0: "None", 1: "Lax", 2: "Strict"}
    cookies = []
    for host, name, value, path, expiry, is_secure, is_http_only, same_site, _oa in rows:
        c = {
            "name": name,
            "value": value,
            "domain": host,
            "path": path,
            "secure": bool(is_secure),
            "httpOnly": bool(is_http_only),
            "sameSite": same_site_map.get(same_site, "None"),
        }
        if expiry:
            exp_val = int(expiry)
            if exp_val > 10_000_000_000:  # in milliseconds, convert to seconds
                exp_val = exp_val // 1000
            if exp_val > 0:
                c["expires"] = exp_val
        cookies.append(c)
    return cookies


def extract_firefox_capcut_cookies(container_id: Optional[int] = None) -> List[Dict]:
    """Read CapCut cookies for one container (None/0 = the default, non-container profile)."""
    wanted = "" if not container_id else f"^userContextId={int(container_id)}"
    rows = [r for r in _read_capcut_cookie_rows() if (r[8] or "") == wanted]
    return _rows_to_cookies(rows)


def _container_labels() -> Dict[int, str]:
    profile_dir = detect_firefox_profile_path()
    if not profile_dir:
        return {}
    path = profile_dir / "containers.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"Failed reading containers.json: {e}")
        return {}
    labels = {}
    for ident in data.get("identities", []):
        cid = ident.get("userContextId")
        if isinstance(cid, int) and cid < 4_000_000_000:
            labels[cid] = ident.get("name") or f"container-{cid}"
    return labels


def list_firefox_accounts() -> List[Dict]:
    """Return every Firefox container that holds a logged-in CapCut session."""
    rows = _read_capcut_cookie_rows()
    labels = _container_labels()
    grouped: Dict[int, List[Tuple]] = {}
    for row in rows:
        match = _CONTAINER_ATTR.match(row[8] or "")
        if not match:
            continue  # partitioned third-party cookies are not used for direct navigation
        grouped.setdefault(int(match.group(1)), []).append(row)

    accounts = []
    for cid, crows in sorted(grouped.items()):
        cookies = _rows_to_cookies(crows)
        sid = next((c["value"] for c in cookies if c["name"] == "sessionid"), None)
        if not sid:
            continue
        uid = next((c["value"] for c in cookies if c["name"] == "uid_tt"), None)
        accounts.append({
            "container": cid,
            "name": labels.get(cid, f"container-{cid}"),
            "sessionid": sid,
            "uid": uid,
            "cookies": cookies,
        })
    return accounts


def _load_accounts_cache() -> Dict:
    if not _ACCOUNTS_CACHE_FILE.exists():
        return {"accounts": []}
    try:
        data = json.loads(_ACCOUNTS_CACHE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("accounts"), list):
            return data
    except Exception as e:
        log.warning(f"Failed reading accounts cache: {e}")
    return {"accounts": []}


def _save_accounts_cache(accounts: List[Dict]) -> None:
    payload = {"updated_at": int(time.time()), "accounts": accounts}
    try:
        _ACCOUNTS_CACHE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning(f"Failed writing accounts cache: {e}")


def sync_accounts() -> List[Dict]:
    """Merge the live Firefox containers with the cached jars and persist them.

    A cached jar wins over the Firefox row only when it is richer (it may contain
    telemetry cookies added during a studio activation that Firefox never saw).
    """
    previous = {a.get("sessionid"): a for a in _load_accounts_cache().get("accounts", [])}
    merged: List[Dict] = []
    now = int(time.time())

    default_cookies = get_cached_cookies()
    default_sid = get_session_id(default_cookies)
    if default_sid:
        merged.append({
            "container": 0,
            "name": "default",
            "sessionid": default_sid,
            "uid": next((c["value"] for c in default_cookies if c["name"] == "uid_tt"), None),
            "cookies": default_cookies,
            "added_at": previous.get(default_sid, {}).get("added_at", now),
        })

    for acc in list_firefox_accounts():
        prev = previous.get(acc["sessionid"])
        if prev:
            acc["added_at"] = prev.get("added_at", now)
            if len(prev.get("cookies", [])) > len(acc["cookies"]):
                acc["cookies"] = prev["cookies"]
        else:
            acc["added_at"] = now
        merged.append(acc)

    # Deduplicate by sessionid (same account logged in twice), keeping the richer jar.
    deduped: Dict[str, Dict] = {}
    for acc in merged:
        sid = acc["sessionid"]
        if sid not in deduped or len(acc["cookies"]) > len(deduped[sid]["cookies"]):
            deduped[sid] = acc
    accounts = list(deduped.values())
    _save_accounts_cache(accounts)
    log.info(f"Synced {len(accounts)} account(s) from Firefox containers.")
    return accounts


def get_accounts() -> List[Dict]:
    accounts = _load_accounts_cache().get("accounts") or []
    if not accounts:
        return sync_accounts()
    return accounts


def update_account_cookies(session_id: str, cookies: List[Dict]) -> bool:
    """Replace a cached jar after a studio visit added or rotated cookies."""
    accounts = get_accounts()
    for acc in accounts:
        if acc.get("sessionid") == session_id:
            acc["cookies"] = cookies
            _save_accounts_cache(accounts)
            return True
    return False


def add_account(session_id: str, cookies: List[Dict], name: str = "manual") -> None:
    accounts = [a for a in get_accounts() if a.get("sessionid") != session_id]
    accounts.append({
        "container": None,
        "name": name,
        "sessionid": session_id,
        "uid": next((c["value"] for c in cookies if c["name"] == "uid_tt"), None),
        "cookies": cookies,
        "added_at": int(time.time()),
    })
    _save_accounts_cache(accounts)


def get_cached_cookies() -> List[Dict]:
    """Load cached cookies if file exists and has sessionid, else extract freshly."""
    if _COOKIE_CACHE_FILE.exists():
        try:
            with open(_COOKIE_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and any(c.get("name") == "sessionid" for c in data):
                    return data
        except Exception:
            pass

    return reload_cookies()


def check_cookie_staleness(cookies: List[Dict]) -> Tuple[bool, Optional[int]]:
    """Check if sessionid or sid_guard expires in less than 24h (86400s). Returns (is_stale, seconds_left)."""
    now = int(time.time())
    for c in cookies:
        if c.get("name") in ("sessionid", "sid_guard") and "expires" in c:
            sec_left = c["expires"] - now
            if sec_left < 86400:
                log.warning(f"[!] Dreamina session cookie '{c['name']}' expires in {sec_left // 3600}h ({sec_left}s)! Log into Firefox to renew.")
                return True, sec_left
    return False, None


def reload_cookies() -> List[Dict]:
    """Force re-extract cookies from Firefox and save to cache."""
    cookies = extract_firefox_capcut_cookies()
    if cookies and any(c.get("name") == "sessionid" for c in cookies):
        check_cookie_staleness(cookies)
        try:
            with open(_COOKIE_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cookies, f, indent=2)
            log.info(f"Cached {len(cookies)} CapCut cookies from Firefox.")
        except Exception as e:
            log.warning(f"Failed writing cookie cache: {e}")
    return cookies


def has_valid_session(cookies: List[Dict]) -> bool:
    """Check if sessionid cookie is present and not blank."""
    for c in cookies:
        if c.get("name") == "sessionid" and c.get("value", "").strip():
            return True
    return False


def get_session_id(cookies: Optional[List[Dict]] = None) -> Optional[str]:
    """Retrieve sessionid string from cookie list or cache."""
    ck_list = cookies or get_cached_cookies()
    for c in ck_list:
        if c.get("name") == "sessionid":
            return c.get("value")
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = detect_firefox_profile_path()
    print(f"Detected profile: {p}")
    c = reload_cookies()
    print(f"Extracted {len(c)} cookies from Firefox.")
    sid = get_session_id(c)
    print(f"Has sessionid: {bool(sid)} (prefix: {sid[:6] if sid else 'None'}...)")
