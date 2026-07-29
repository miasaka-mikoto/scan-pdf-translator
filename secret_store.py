from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from pathlib import Path


_MAGIC = b"SPDF-DPAPI-1\0"
_PRODUCT_DIR = "ScanPdfTranslator"
_CREDENTIAL_FILE = "siliconflow-api-key.dpapi"
_CRYPTPROTECT_UI_FORBIDDEN = 0x01


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
    ]


def credential_path() -> Path:
    """Return the per-user credential path, never the application directory."""
    override = os.environ.get("PDF_TRANSLATOR_CREDENTIAL_DIR", "").strip()
    if override:
        base = Path(override)
    else:
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
        base = base / _PRODUCT_DIR / "credentials"
    return base / _CREDENTIAL_FILE


def _input_blob(data: bytes) -> tuple[_DataBlob, ctypes.Array]:
    buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    return _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))), buffer


def _protect(data: bytes) -> bytes:
    if os.name != "nt":
        raise OSError("Windows DPAPI is unavailable on this platform")
    source, source_buffer = _input_blob(data)
    destination = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    ok = crypt32.CryptProtectData(
        ctypes.byref(source),
        "Scan PDF Translator / SiliconFlow",
        None,
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(destination),
    )
    del source_buffer
    if not ok:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(destination.pbData, destination.cbData)
    finally:
        kernel32.LocalFree(destination.pbData)


def _unprotect(data: bytes) -> bytes:
    if os.name != "nt":
        raise OSError("Windows DPAPI is unavailable on this platform")
    source, source_buffer = _input_blob(data)
    destination = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    ok = crypt32.CryptUnprotectData(
        ctypes.byref(source),
        None,
        None,
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(destination),
    )
    del source_buffer
    if not ok:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(destination.pbData, destination.cbData)
    finally:
        kernel32.LocalFree(destination.pbData)


def load_api_key() -> str:
    """Load the current Windows user's DPAPI-protected SiliconFlow key."""
    path = credential_path()
    try:
        payload = path.read_bytes()
        if not payload.startswith(_MAGIC):
            return ""
        return _unprotect(payload[len(_MAGIC) :]).decode("utf-8").strip()
    except (OSError, UnicodeError, ValueError):
        return ""


def save_api_key(value: str) -> str:
    """Validate and protect a key. The returned status never contains the key."""
    key = (value or "").strip()
    if not key:
        return "未保存：API Key 为空。"
    if not key.startswith("sk-") or len(key) < 24 or any(char.isspace() for char in key):
        return "未保存：API Key 格式不完整，请检查后离开输入框再试。"
    if os.name != "nt":
        return "未保存：自动保存仅支持 Windows DPAPI；本次仍可在内存中使用。"
    try:
        path = credential_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(_MAGIC + _protect(key.encode("utf-8")))
        temporary.replace(path)
        return "已使用 Windows DPAPI 加密保存到当前用户，下次启动会自动填充。"
    except OSError:
        return "保存失败：无法写入当前用户的加密凭据目录；本次仍可在内存中使用。"


def clear_api_key() -> tuple[str, str]:
    """Delete the protected credential and clear the textbox value."""
    try:
        credential_path().unlink(missing_ok=True)
        return "", "已清除本机保存的 API Key。"
    except OSError:
        return "", "清除失败：请关闭其他实例后重试。"
