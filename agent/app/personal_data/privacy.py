from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

# 这些规则只用于“禁止长期保存/写日志”的数据治理，不承担身份识别。
_SECRET_KEY_MARKERS = {
    "password",
    "passwd",
    "pwd",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "session",
    "private_key",
    "验证码",
    "密码",
    "密钥",
    "令牌",
}

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "cn_id_card",
        re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"),
    ),
    (
        "bank_card",
        re.compile(r"(?<!\d)(?:\d[ -]?){16,19}(?!\d)"),
    ),
    (
        "mobile_phone",
        re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    ),
    (
        "sms_code",
        re.compile(
            r"(?:验证码|校验码|短信码)\s*[:：]?\s*\d{4,8}",
            re.IGNORECASE,
        ),
    ),
    (
        "api_key",
        re.compile(
            r"(?:sk|ak|api[_-]?key)[-_A-Za-z0-9]{12,}",
            re.IGNORECASE,
        ),
    ),
    (
        "bearer_token",
        re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
    ),
]


def is_sensitive_key(key: str) -> bool:
    normalized = str(key).strip().lower()
    return any(marker in normalized for marker in _SECRET_KEY_MARKERS)


def detect_sensitive_text(text: str) -> list[str]:
    findings: list[str] = []
    for name, pattern in _PATTERNS:
        if pattern.search(text):
            findings.append(name)
    return findings


def redact_sensitive_text(text: str) -> str:
    value = str(text)
    for name, pattern in _PATTERNS:
        value = pattern.sub(f"[redacted:{name}]", value)
    return value


def sanitize_personal_value(
    value: Any,
    *,
    reject_secret_keys: bool = True,
    max_depth: int = 6,
    max_items: int = 100,
    max_text_length: int = 8_000,
    _depth: int = 0,
) -> Any:
    """返回可持久化的脱敏副本，不修改调用方对象。"""
    if _depth > max_depth:
        return "[max_depth]"

    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        text = redact_sensitive_text(value)
        if len(text) > max_text_length:
            text = text[:max_text_length] + "...[truncated]"
        return text

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                result["__truncated__"] = True
                break
            clean_key = str(key)
            if is_sensitive_key(clean_key):
                if reject_secret_keys:
                    raise ValueError(
                        f"字段 {clean_key!r} 属于禁止保存的敏感字段。"
                    )
                result[clean_key] = "[redacted]"
                continue
            result[clean_key] = sanitize_personal_value(
                item,
                reject_secret_keys=reject_secret_keys,
                max_depth=max_depth,
                max_items=max_items,
                max_text_length=max_text_length,
                _depth=_depth + 1,
            )
        return result

    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [
            sanitize_personal_value(
                item,
                reject_secret_keys=reject_secret_keys,
                max_depth=max_depth,
                max_items=max_items,
                max_text_length=max_text_length,
                _depth=_depth + 1,
            )
            for item in list(value)[:max_items]
        ]

    return sanitize_personal_value(
        str(value),
        reject_secret_keys=reject_secret_keys,
        max_depth=max_depth,
        max_items=max_items,
        max_text_length=max_text_length,
        _depth=_depth + 1,
    )


def safe_json_dumps(value: Any) -> str:
    clean = sanitize_personal_value(value, reject_secret_keys=False)
    return json.dumps(clean, ensure_ascii=False, separators=(",", ":"))


def deepcopy_sanitized(value: Any) -> Any:
    return deepcopy(sanitize_personal_value(value))
