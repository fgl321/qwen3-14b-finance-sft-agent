from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.personal_data.privacy import redact_sensitive_text


class ShortTermMemoryService:
    """
    Redis 短期记忆。

    兼容旧项目常见调用方式，同时增加：
    - tenant_id 隔离；
    - 只允许 user/assistant；
    - 敏感内容脱敏；
    - TTL、最大消息数、清空和摘要；
    - 读取旧版 short_memory:{user_id}:{thread_id} 键。
    """

    def __init__(
        self,
        settings: Any | None = None,
        *,
        redis_client: Any | None = None,
        redis_url: str | None = None,
        enabled: bool | None = None,
        max_messages: int | None = None,
        ttl_seconds: int | None = None,
        key_prefix: str = "short_memory",
    ) -> None:
        if settings is None and redis_client is None:
            try:
                from app.core.config import get_settings

                settings = get_settings()
            except Exception:
                settings = None

        self.enabled = bool(
            enabled
            if enabled is not None
            else getattr(settings, "short_memory_enabled", True)
        )
        self.max_messages = max(
            2,
            int(
                max_messages
                if max_messages is not None
                else getattr(settings, "short_memory_max_messages", 12)
            ),
        )
        self.ttl_seconds = max(
            60,
            int(
                ttl_seconds
                if ttl_seconds is not None
                else getattr(settings, "short_memory_ttl_seconds", 86_400)
            ),
        )
        self.key_prefix = key_prefix.strip(":") or "short_memory"

        if redis_client is not None:
            self.redis = redis_client
        elif self.enabled:
            try:
                import redis
            except ImportError as exc:  # pragma: no cover - 环境依赖
                raise RuntimeError(
                    "缺少 redis 包，请执行 python -m pip install redis。"
                ) from exc

            final_url = redis_url or getattr(
                settings, "redis_url", "redis://127.0.0.1:6379/0"
            )
            self.redis = redis.Redis.from_url(
                final_url,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
                health_check_interval=30,
            )
        else:
            self.redis = None

    def _key(self, *, user_id: str, thread_id: str, tenant_id: str) -> str:
        return (
            f"{self.key_prefix}:{tenant_id.strip() or 'default'}:"
            f"{user_id.strip()}:{thread_id.strip()}"
        )

    def _legacy_key(self, *, user_id: str, thread_id: str) -> str:
        return f"{self.key_prefix}:{user_id.strip()}:{thread_id.strip()}"

    def _summary_key(self, *, user_id: str, thread_id: str, tenant_id: str) -> str:
        return self._key(
            user_id=user_id, thread_id=thread_id, tenant_id=tenant_id
        ) + ":summary"

    @staticmethod
    def _validate_identity(user_id: str, thread_id: str) -> None:
        if not str(user_id).strip():
            raise ValueError("user_id 不能为空。")
        if not str(thread_id).strip():
            raise ValueError("thread_id 不能为空。")

    @staticmethod
    def _normalize_message(role: str, content: str) -> dict[str, Any]:
        clean_role = str(role).strip().lower()
        if clean_role not in {"user", "assistant"}:
            raise ValueError("短期记忆只允许保存 user 和 assistant 消息。")

        clean_content = redact_sensitive_text(str(content).strip())
        if not clean_content:
            raise ValueError("消息内容不能为空。")
        if len(clean_content) > 8_000:
            clean_content = clean_content[:8_000] + "...[truncated]"

        return {
            "message_id": uuid4().hex,
            "role": clean_role,
            "content": clean_content,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def ping(self) -> bool:
        if not self.enabled:
            return False
        try:
            return bool(self.redis.ping())
        except Exception:
            return False

    def _summarize_messages_before_trim(
        self,
        *,
        key: str,
        summary_key: str,
    ) -> None:
        """把即将被 LTRIM 淘汰的消息压缩到摘要键。"""
        try:
            current_count = int(self.redis.llen(key) or 0)
        except Exception:
            return
        evicted_count = max(0, current_count + 1 - self.max_messages)
        if evicted_count <= 0:
            return
        raw_items = self.redis.lrange(key, 0, evicted_count - 1)
        entries = self._decode_entries(list(raw_items or []))
        if not entries:
            return
        previous = self.redis.get(summary_key) or ""
        if isinstance(previous, bytes):
            previous = previous.decode("utf-8", errors="replace")
        lines = [str(previous).strip()] if str(previous).strip() else []
        lines.append("[较早对话摘要]")
        for item in entries:
            role = "用户" if item["role"] == "user" else "助手"
            content = str(item["content"]).replace("\n", " ").strip()
            lines.append(f"{role}: {content[:500]}")
        summary = redact_sensitive_text("\n".join(lines))[-8_000:]
        self.redis.setex(summary_key, self.ttl_seconds, summary)

    def append_message(
        self,
        *,
        user_id: str,
        thread_id: str,
        role: str,
        content: str,
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        self._validate_identity(user_id, thread_id)
        message = self._normalize_message(role, content)
        if not self.enabled:
            return message

        key = self._key(
            user_id=user_id, thread_id=thread_id, tenant_id=tenant_id
        )
        summary_key = self._summary_key(
            user_id=user_id, thread_id=thread_id, tenant_id=tenant_id
        )
        self._summarize_messages_before_trim(
            key=key, summary_key=summary_key
        )
        payload = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        pipe = self.redis.pipeline(transaction=True)
        pipe.rpush(key, payload)
        pipe.ltrim(key, -self.max_messages, -1)
        pipe.expire(key, self.ttl_seconds)
        pipe.execute()
        return message

    # 旧代码兼容别名。
    add_message = append_message
    save_message = append_message

    def append_messages(
        self,
        *,
        user_id: str,
        thread_id: str,
        messages: list[dict[str, Any]],
        tenant_id: str = "default",
    ) -> int:
        saved = 0
        for message in messages:
            role = str(message.get("role") or "").strip()
            content = str(message.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            self.append_message(
                user_id=user_id,
                thread_id=thread_id,
                tenant_id=tenant_id,
                role=role,
                content=content,
            )
            saved += 1
        return saved

    def save_turn(
        self,
        *,
        user_id: str,
        thread_id: str,
        user_message: str,
        assistant_message: str,
        tenant_id: str = "default",
    ) -> int:
        self.append_message(
            user_id=user_id,
            thread_id=thread_id,
            tenant_id=tenant_id,
            role="user",
            content=user_message,
        )
        self.append_message(
            user_id=user_id,
            thread_id=thread_id,
            tenant_id=tenant_id,
            role="assistant",
            content=assistant_message,
        )
        return 2

    append_turn = save_turn

    @staticmethod
    def _decode_entries(raw_items: list[Any]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for raw in raw_items:
            try:
                item = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            except Exception:
                continue
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip()
            content = str(item.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            entries.append(
                {
                    "message_id": str(item.get("message_id") or uuid4().hex),
                    "role": role,
                    "content": redact_sensitive_text(content),
                    "created_at": str(item.get("created_at") or ""),
                }
            )
        return entries

    def list_entries(
        self,
        *,
        user_id: str,
        thread_id: str,
        tenant_id: str = "default",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self._validate_identity(user_id, thread_id)
        if not self.enabled:
            return []
        final_limit = min(max(int(limit or self.max_messages), 1), self.max_messages)
        key = self._key(
            user_id=user_id, thread_id=thread_id, tenant_id=tenant_id
        )
        raw_items = self.redis.lrange(key, -final_limit, -1)

        # 升级前已有的数据仍可读取；读取后不自动复制，避免重复消息。
        if not raw_items and (tenant_id.strip() or "default") == "default":
            raw_items = self.redis.lrange(
                self._legacy_key(user_id=user_id, thread_id=thread_id),
                -final_limit,
                -1,
            )
        return self._decode_entries(list(raw_items or []))

    def get_messages(
        self,
        *,
        user_id: str,
        thread_id: str,
        tenant_id: str = "default",
        limit: int | None = None,
    ) -> list[dict[str, str]]:
        return [
            {"role": item["role"], "content": item["content"]}
            for item in self.list_entries(
                user_id=user_id,
                thread_id=thread_id,
                tenant_id=tenant_id,
                limit=limit,
            )
        ]

    get_history = get_messages
    load_history = get_messages

    def set_summary(
        self,
        *,
        user_id: str,
        thread_id: str,
        summary: str,
        tenant_id: str = "default",
    ) -> str:
        self._validate_identity(user_id, thread_id)
        clean = redact_sensitive_text(str(summary).strip())[:8_000]
        if not clean:
            raise ValueError("summary 不能为空。")
        if self.enabled:
            key = self._summary_key(
                user_id=user_id, thread_id=thread_id, tenant_id=tenant_id
            )
            self.redis.setex(key, self.ttl_seconds, clean)
        return clean

    def get_summary(
        self,
        *,
        user_id: str,
        thread_id: str,
        tenant_id: str = "default",
    ) -> str:
        self._validate_identity(user_id, thread_id)
        if not self.enabled:
            return ""
        value = self.redis.get(
            self._summary_key(
                user_id=user_id, thread_id=thread_id, tenant_id=tenant_id
            )
        )
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        return redact_sensitive_text(str(value or ""))

    def clear_thread(
        self,
        *,
        user_id: str,
        thread_id: str,
        tenant_id: str = "default",
    ) -> int:
        self._validate_identity(user_id, thread_id)
        if not self.enabled:
            return 0
        keys = [
            self._key(user_id=user_id, thread_id=thread_id, tenant_id=tenant_id),
            self._summary_key(
                user_id=user_id, thread_id=thread_id, tenant_id=tenant_id
            ),
        ]
        if (tenant_id.strip() or "default") == "default":
            keys.append(self._legacy_key(user_id=user_id, thread_id=thread_id))
        return int(self.redis.delete(*keys))

    clear = clear_thread
    delete_thread = clear_thread

    def ttl(
        self,
        *,
        user_id: str,
        thread_id: str,
        tenant_id: str = "default",
    ) -> int:
        if not self.enabled:
            return -2
        return int(
            self.redis.ttl(
                self._key(
                    user_id=user_id, thread_id=thread_id, tenant_id=tenant_id
                )
            )
        )

    def close(self) -> None:
        if self.redis is not None and hasattr(self.redis, "close"):
            self.redis.close()
