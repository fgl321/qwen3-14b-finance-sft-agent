from __future__ import annotations

import hashlib
import hmac
import json
import unicodedata
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Iterable

from pydantic import BaseModel


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("non-finite Decimal is not canonicalizable")
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def _datetime_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("naive datetime is not canonicalizable")
    utc_value = value.astimezone(UTC)
    return utc_value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def canonicalize(
    value: Any,
    *,
    semantic_set_paths: frozenset[str] = frozenset(),
    include_none: bool = True,
    _path: tuple[str, ...] = (),
) -> Any:
    """Convert values to the project's stable JSON domain.

    Lists remain ordered unless their dotted field path is explicitly listed in
    semantic_set_paths.  This prevents accidental sorting of execution steps.
    """

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python", exclude_none=not include_none)
    if isinstance(value, Enum):
        return canonicalize(
            value.value,
            semantic_set_paths=semantic_set_paths,
            include_none=include_none,
            _path=_path,
        )
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        raise TypeError("binary float is forbidden in canonical identity payloads")
    if isinstance(value, Decimal):
        return _decimal_text(value)
    if isinstance(value, datetime):
        return _datetime_text(value)
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for raw_key in sorted(value, key=lambda item: str(item)):
            key = unicodedata.normalize("NFC", str(raw_key))
            item = value[raw_key]
            if item is None and not include_none:
                continue
            result[key] = canonicalize(
                item,
                semantic_set_paths=semantic_set_paths,
                include_none=include_none,
                _path=(*_path, key),
            )
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [
            canonicalize(
                item,
                semantic_set_paths=semantic_set_paths,
                include_none=include_none,
                _path=(*_path, "[]"),
            )
            for item in value
        ]
        dotted_path = ".".join(_path)
        if isinstance(value, (set, frozenset)) or dotted_path in semantic_set_paths:
            encoded = {
                json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                for item in items
            }
            return [json.loads(item) for item in sorted(encoded)]
        return items
    raise TypeError(f"unsupported canonical type: {type(value).__name__}")


def canonical_json(
    value: Any,
    *,
    semantic_set_paths: Iterable[str] = (),
    include_none: bool = True,
) -> str:
    normalized = canonicalize(
        value,
        semantic_set_paths=frozenset(semantic_set_paths),
        include_none=include_none,
    )
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_hash(
    value: Any,
    *,
    semantic_set_paths: Iterable[str] = (),
    include_none: bool = True,
) -> str:
    payload = canonical_json(
        value,
        semantic_set_paths=semantic_set_paths,
        include_none=include_none,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def identity_fingerprint(
    normalized_value: str,
    *,
    key: bytes,
    key_id: str,
    tenant_id: str,
    field_type: str,
    purpose: str,
) -> dict[str, str]:
    if not key or not key_id or not purpose:
        raise ValueError("HMAC key, key_id and purpose are required")
    domain = "\x1f".join(
        [
            "finance-agent-control-plane-v1",
            unicodedata.normalize("NFC", tenant_id),
            unicodedata.normalize("NFC", field_type),
            unicodedata.normalize("NFC", purpose),
            unicodedata.normalize("NFC", normalized_value),
        ]
    ).encode("utf-8")
    digest = hmac.new(key, domain, hashlib.sha256).hexdigest()
    return {
        "algorithm": "HMAC-SHA256",
        "key_id": key_id,
        "purpose": purpose,
        "fingerprint": "hmac-sha256:" + digest,
    }
