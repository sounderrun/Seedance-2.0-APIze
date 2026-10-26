"""
DPAPI-backed local secret store for dreamina2api.

Replaces plaintext credentials in config.env (audit 2026-10-06 finding H1).
Values are encrypted with Windows DPAPI (CryptProtectData) bound to the current
Windows user + machine and stored as base64 blobs in secrets.enc next to this
module. Stdlib only; falls back gracefully so callers keep working when the
store is absent or unreadable.
"""
import base64
import ctypes
import json
import logging
import os
from pathlib import Path

try:
    import ctypes.wintypes  # noqa: F401
except ImportError:
    pass

log = logging.getLogger("secrets_store")

STORE_FILE = Path(__file__).resolve().parent / "secrets.enc"
CREDENTIAL_KEYS = ("GOOGLE_EMAIL", "GOOGLE_PASSWORD", "GOOGLE_COOKIES", "DREAMINA_SESSION_ID")

_CRYPTPROTECT_UI_FORBIDDEN = 0x1


class _DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", ctypes.c_ulong),
        ("pbData", ctypes.POINTER(ctypes.c_byte)),
    ]


def _blob_from_bytes(data: bytes) -> _DATA_BLOB:
    buf = ctypes.create_string_buffer(data, len(data))
    return _DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_byte)))


def _protect(data: bytes) -> bytes:
    in_blob = _blob_from_bytes(data)
    out_blob = _DATA_BLOB()
    if not ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(in_blob), None, None, None, None,
        _CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(out_blob),
    ):
        raise OSError("CryptProtectData failed")
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(out_blob.pbData)


def _unprotect(blob: bytes) -> bytes:
    in_blob = _blob_from_bytes(blob)
    out_blob = _DATA_BLOB()
    if not ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(in_blob), None, None, None, None,
        _CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(out_blob),
    ):
        raise OSError("CryptUnprotectData failed")
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(out_blob.pbData)


def _load() -> dict:
    if not STORE_FILE.exists():
        return {}
    try:
        return json.loads(STORE_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"Could not read secret store {STORE_FILE.name}: {e}")
        return {}


def _save(store: dict) -> None:
    STORE_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")


def has_secret(key: str) -> bool:
    return key in _load()


def get_secret(key: str, default: str = "") -> str:
    """Return the decrypted secret, or default if absent/undecryptable."""
    blob = _load().get(key)
    if not blob:
        return default
    try:
        return _unprotect(base64.b64decode(blob)).decode("utf-8")
    except Exception as e:
        log.warning(f"Could not decrypt secret '{key}': {e}")
        return default


def set_secret(key: str, value: str) -> None:
    store = _load()
    store[key] = base64.b64encode(_protect(value.encode("utf-8"))).decode("ascii")
    _save(store)


def delete_secret(key: str) -> None:
    store = _load()
    if store.pop(key, None) is not None:
        _save(store)


def migrate_config_env(config_path: Path = Path(__file__).resolve().parent / "config.env") -> list:
    """
    One-time scrub: move plaintext credential keys from config.env into the DPAPI
    store and rewrite config.env without them (non-secret keys are preserved).
    Idempotent. Returns the list of migrated key names. Secret values are never
    logged.
    """
    migrated = []
    if not config_path.exists():
        return migrated

    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)
    out_lines = []
    for line in lines:
        stripped = line.strip()
        key = ""
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
        if key in CREDENTIAL_KEYS:
            value = stripped.split("=", 1)[1].strip().strip('"').strip("'")
            if value and not has_secret(key):
                set_secret(key, value)
            if key not in migrated:
                migrated.append(key)
            continue
        out_lines.append(line)

    if migrated:
        config_path.write_text("".join(out_lines), encoding="utf-8")
        log.info(f"Migrated {len(migrated)} credential(s) from config.env to DPAPI store: {', '.join(migrated)}")
    return migrated
