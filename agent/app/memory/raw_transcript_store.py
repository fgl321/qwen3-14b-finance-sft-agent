from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.personal_data.privacy import redact_sensitive_text


class RawTranscriptStore:
    """Immutable source-of-record for raw conversation messages.

    Messages are append-only; nothing in this store is ever updated or
    compressed.  Narrative memory and task state are separate derived layers.
    """

    def __init__(
        self,
        *,
        postgres_dsn: str | None = None,
        settings: Any | None = None,
    ) -> None:
        if settings is None and postgres_dsn is None:
            try:
                from app.core.config import get_settings

                settings = get_settings()
            except Exception:
                settings = None
        self.postgres_dsn = postgres_dsn or getattr(
            settings,
            "postgres_dsn",
            "postgresql://agent:agent@127.0.0.1:5432/agent",
        )

    def _connect(self):
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(
            self.postgres_dsn,
            row_factory=dict_row,
        )

    def init_schema(self) -> None:
        sql = """
        CREATE TABLE IF NOT EXISTS conversation_messages (
            id BIGSERIAL PRIMARY KEY,
            tenant_id TEXT NOT NULL DEFAULT 'default',
            user_id TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            request_id TEXT NOT NULL DEFAULT '',
            run_id TEXT NOT NULL DEFAULT '',
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_conversation_messages_thread
            ON conversation_messages (tenant_id, user_id, thread_id, id);
        """
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql)
            connection.commit()

    def append_message(
        self,
        *,
        tenant_id: str,
        user_id: str,
        thread_id: str,
        role: str,
        content: str,
        request_id: str = "",
        run_id: str = "",
    ) -> dict[str, Any]:
        clean_role = str(role).strip().lower()
        if clean_role not in {"user", "assistant"}:
            raise ValueError(
                "raw transcript only stores user/assistant messages"
            )
        clean_content = redact_sensitive_text(
            str(content).strip()
        )
        if not clean_content:
            raise ValueError("message content cannot be empty")
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO conversation_messages
                        (tenant_id, user_id, thread_id, request_id,
                         run_id, role, content, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        str(tenant_id),
                        str(user_id),
                        str(thread_id),
                        str(request_id),
                        str(run_id),
                        clean_role,
                        clean_content[:20_000],
                        created_at,
                    ),
                )
                row = cursor.fetchone()
            connection.commit()
        return {
            "message_id": str(row["id"]) if row else "",
            "role": clean_role,
            "content": clean_content,
            "created_at": created_at,
        }

    def append_turn(
        self,
        *,
        tenant_id: str,
        user_id: str,
        thread_id: str,
        user_message: str,
        assistant_message: str,
        request_id: str = "",
        run_id: str = "",
    ) -> int:
        self.append_message(
            tenant_id=tenant_id,
            user_id=user_id,
            thread_id=thread_id,
            role="user",
            content=user_message,
            request_id=request_id,
            run_id=run_id,
        )
        self.append_message(
            tenant_id=tenant_id,
            user_id=user_id,
            thread_id=thread_id,
            role="assistant",
            content=assistant_message,
            request_id=request_id,
            run_id=run_id,
        )
        return 2

    def list_recent(
        self,
        *,
        tenant_id: str,
        user_id: str,
        thread_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, role, content, created_at
                    FROM conversation_messages
                    WHERE tenant_id = %s
                      AND user_id = %s
                      AND thread_id = %s
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (
                        str(tenant_id),
                        str(user_id),
                        str(thread_id),
                        int(limit),
                    ),
                )
                rows = list(cursor.fetchall() or [])
        return [
            {
                "role": str(row.get("role") or "user"),
                "content": str(row.get("content") or ""),
                "message_id": str(row.get("id") or ""),
                "created_at": str(row.get("created_at") or ""),
            }
            for row in reversed(rows)
        ]
