from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel


T = TypeVar("T")


class RequestIdempotencyConflict(ValueError):
    """
    同一个幂等键被用于不同请求内容。

    这是客户端请求冲突，不是工具或模型执行错误。
    API 层应将其映射为 HTTP 409。
    """


@dataclass(frozen=True, slots=True)
class IdempotencyExecution(Generic[T]):
    value: T
    replayed: bool
    scope_key_hash: str
    request_fingerprint: str


class RequestIdempotencyStore(Protocol):
    async def execute(
        self,
        *,
        scope_key: str,
        request_fingerprint: str,
        operation: Callable[[], Awaitable[T]],
    ) -> IdempotencyExecution[T]:
        ...


@dataclass(slots=True)
class _Entry(Generic[T]):
    request_fingerprint: str
    task: asyncio.Task[T]
    created_at: float
    last_accessed_at: float


def _normalize_decimal(value: Decimal) -> str:
    if not value.is_finite():
        return str(value)

    normalized = value.normalize()
    rendered = format(normalized, "f")

    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")

    return rendered or "0"


def normalize_idempotency_value(value: Any) -> Any:
    """
    把请求数据转换为稳定、可排序、可 JSON 序列化的结构。

    字典键顺序和集合顺序不会影响最终指纹；列表顺序保留，
    因为历史对话消息的先后顺序具有业务语义。
    """

    if value is None:
        return None

    if isinstance(value, BaseModel):
        return normalize_idempotency_value(
            value.model_dump(mode="json")
        )

    if is_dataclass(value) and not isinstance(value, type):
        return normalize_idempotency_value(asdict(value))

    if isinstance(value, Enum):
        return normalize_idempotency_value(value.value)

    if isinstance(value, Decimal):
        return {
            "__number__": _normalize_decimal(value),
        }

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return {
            "__number__": str(value),
        }

    if isinstance(value, float):
        return {
            "__number__": _normalize_decimal(
                Decimal(str(value))
            ),
        }

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Mapping):
        return {
            str(key): normalize_idempotency_value(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }

    if isinstance(value, (set, frozenset)):
        normalized_items = [
            normalize_idempotency_value(item)
            for item in value
        ]

        return sorted(
            normalized_items,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            normalize_idempotency_value(item)
            for item in value
        ]

    if isinstance(value, (str, bytes, bytearray)):
        if isinstance(value, str):
            return value

        return bytes(value).hex()

    return str(value)


def build_request_fingerprint(payload: Mapping[str, Any]) -> str:
    normalized = normalize_idempotency_value(payload)
    canonical_json = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()


def build_idempotency_scope_key(
    *,
    tenant_id: str,
    user_id: str,
    request_id: str,
) -> str:
    clean_tenant_id = tenant_id.strip() or "default"
    clean_user_id = user_id.strip()
    clean_request_id = request_id.strip()

    if not clean_user_id:
        raise ValueError("user_id 不能为空。")

    if not clean_request_id:
        raise ValueError("request_id 不能为空。")

    raw_scope = (
        f"{clean_tenant_id}\x1f"
        f"{clean_user_id}\x1f"
        f"{clean_request_id}"
    )

    digest = hashlib.sha256(
        raw_scope.encode("utf-8")
    ).hexdigest()

    return f"finance-agent-request:{digest}"


def _scope_key_hash(scope_key: str) -> str:
    return hashlib.sha256(
        scope_key.encode("utf-8")
    ).hexdigest()[:24]


class InMemoryRequestIdempotencyStore:
    """
    单进程异步请求幂等存储。

    - 同一个 scope_key + 相同指纹：等待或复用首次任务；
    - 同一个 scope_key + 不同指纹：抛出冲突；
    - 首次任务异常：删除记录，允许客户端再次重试；
    - 已完成记录按 TTL 清理；
    - 只缓存服务层最终结果，不缓存原始异常堆栈。

    该实现适合当前单 Uvicorn 进程。接口已抽象为 Protocol，
    多 Worker/多实例部署时可替换为 Redis 或数据库实现。
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = 600.0,
        max_entries: int = 2048,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds 必须大于 0。")

        if max_entries <= 0:
            raise ValueError("max_entries 必须大于 0。")

        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._entries: dict[str, _Entry[Any]] = {}
        self._lock = asyncio.Lock()

    async def execute(
        self,
        *,
        scope_key: str,
        request_fingerprint: str,
        operation: Callable[[], Awaitable[T]],
    ) -> IdempotencyExecution[T]:
        clean_scope_key = scope_key.strip()
        clean_fingerprint = request_fingerprint.strip()

        if not clean_scope_key:
            raise ValueError("scope_key 不能为空。")

        if not clean_fingerprint:
            raise ValueError(
                "request_fingerprint 不能为空。"
            )

        now = time.monotonic()

        async with self._lock:
            self._cleanup_locked(now)

            existing = self._entries.get(clean_scope_key)

            if existing is not None:
                if (
                    existing.request_fingerprint
                    != clean_fingerprint
                ):
                    raise RequestIdempotencyConflict(
                        "同一个 request_id 已经用于不同的请求内容，"
                        "请为新请求使用新的 request_id。"
                    )

                existing.last_accessed_at = now
                task: asyncio.Task[T] = existing.task
                replayed = True
            else:
                task = asyncio.create_task(operation())
                entry: _Entry[T] = _Entry(
                    request_fingerprint=(
                        clean_fingerprint
                    ),
                    task=task,
                    created_at=now,
                    last_accessed_at=now,
                )
                self._entries[clean_scope_key] = entry
                replayed = False

                task.add_done_callback(
                    lambda completed_task: (
                        self._schedule_failed_entry_removal(
                            scope_key=clean_scope_key,
                            task=completed_task,
                        )
                    )
                )

                self._enforce_capacity_locked()

        try:
            value = await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                await self._remove_if_same_task(
                    scope_key=clean_scope_key,
                    task=task,
                )
            raise
        except Exception:
            await self._remove_if_same_task(
                scope_key=clean_scope_key,
                task=task,
            )
            raise

        return IdempotencyExecution(
            value=copy.deepcopy(value),
            replayed=replayed,
            scope_key_hash=_scope_key_hash(
                clean_scope_key
            ),
            request_fingerprint=clean_fingerprint,
        )

    def _schedule_failed_entry_removal(
        self,
        *,
        scope_key: str,
        task: asyncio.Task[Any],
    ) -> None:
        failed = task.cancelled()

        if not failed:
            try:
                failed = task.exception() is not None
            except asyncio.CancelledError:
                failed = True

        if not failed:
            return

        try:
            asyncio.get_running_loop().create_task(
                self._remove_if_same_task(
                    scope_key=scope_key,
                    task=task,
                )
            )
        except RuntimeError:
            return

    async def _remove_if_same_task(
        self,
        *,
        scope_key: str,
        task: asyncio.Task[Any],
    ) -> None:
        async with self._lock:
            current = self._entries.get(scope_key)

            if current is not None and current.task is task:
                self._entries.pop(scope_key, None)

    def _cleanup_locked(self, now: float) -> None:
        expired_keys = [
            scope_key
            for scope_key, entry in self._entries.items()
            if entry.task.done()
            and now - entry.last_accessed_at
            >= self._ttl_seconds
        ]

        for scope_key in expired_keys:
            self._entries.pop(scope_key, None)

    def _enforce_capacity_locked(self) -> None:
        overflow = len(self._entries) - self._max_entries

        if overflow <= 0:
            return

        completed_entries = sorted(
            (
                (scope_key, entry)
                for scope_key, entry in self._entries.items()
                if entry.task.done()
            ),
            key=lambda pair: pair[1].last_accessed_at,
        )

        for scope_key, _ in completed_entries[:overflow]:
            self._entries.pop(scope_key, None)
