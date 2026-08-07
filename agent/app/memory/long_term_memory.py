from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from app.personal_data.privacy import sanitize_personal_value


# 个人金融 Agent 允许长期保存的稳定事实。
# 值为允许的 fact_key；"*" 表示该类型下允许任意非敏感 key。
DEFAULT_FACT_WHITELIST: dict[str, frozenset[str]] = {
    "family_finance": frozenset(
        {
            "annual_necessary_expense",
            "monthly_necessary_expense",
            "monthly_income",
            "monthly_expense",
            "available_assets",
            "cash_assets",
            "fund_assets",
            "stock_assets",
            "total_assets",
            "mortgage_balance",
            "debt_balance",
            "total_debt",
        }
    ),
    "insurance": frozenset(
        {
            "life_insurance",
            "existing_life_insurance",
            "husband_life_insurance",
            "wife_life_insurance",
            "spouse_life_insurance",
            "medical_insurance",
            "critical_illness_insurance",
        }
    ),
    "family_profile": frozenset(
        {
            "age",
            "husband_age",
            "wife_age",
            "spouse_age",
            "child_age",
            "children_ages",
            "family_status",
            "city",
            "occupation",
        }
    ),
    "preference": frozenset(
        {
            "risk_preference",
            "investment_experience",
            "answer_style",
        }
    ),
    "goal": frozenset(
        {
            "short_term_goal",
            "medium_term_goal",
            "long_term_goal",
            "emergency_fund_goal",
            "education_goal",
            "retirement_goal",
        }
    ),
}


@dataclass(slots=True)
class LongTermFact:
    # 旧项目依赖的字段保持在前面。
    id: int
    tenant_id: str
    user_id: str
    fact_type: str
    fact_key: str
    fact_value: dict[str, Any]
    confidence: float
    source_thread_id: str | None
    created_at: str
    updated_at: str

    # Stage 4.4 Lite 新增治理字段。
    source_message_id: str | None = None
    status: str = "active"
    version: int = 1
    is_user_confirmed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LongTermFactHistory:
    id: int
    fact_id: int
    tenant_id: str
    user_id: str
    fact_type: str
    fact_key: str
    fact_value: dict[str, Any]
    confidence: float
    source_thread_id: str | None
    source_message_id: str | None
    status: str
    version: int
    is_user_confirmed: bool
    change_reason: str
    metadata: dict[str, Any]
    archived_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LongTermMemoryService:
    """
    PostgreSQL 长期事实服务。

    设计原则：
    - 只保存白名单中的结构化稳定事实；
    - 当前有效值保存在 user_memory_facts；
    - 更新前快照保存在 user_memory_fact_history；
    - 用户明确更正优先；低置信度自动抽取不能覆盖高置信度事实；
    - 敏感字段拒绝，文本中的身份证/银行卡/Token 等自动脱敏；
    - 保留旧版 init_schema/upsert_fact/list_facts/get_fact/delete_user_facts 接口。
    """

    def __init__(
        self,
        postgres_dsn: str | None = None,
        *,
        settings: Any | None = None,
        fact_whitelist: dict[str, frozenset[str]] | None = None,
        strict_whitelist: bool = True,
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
        self.fact_whitelist = fact_whitelist or DEFAULT_FACT_WHITELIST
        self.strict_whitelist = strict_whitelist

    def _connect(self):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - 环境依赖
            raise RuntimeError(
                "缺少 psycopg，请执行 python -m pip install 'psycopg[binary]'。"
            ) from exc
        return psycopg.connect(self.postgres_dsn, row_factory=dict_row)

    @staticmethod
    def _json_value(value: Any) -> Any:
        try:
            from psycopg.types.json import Jsonb

            return Jsonb(value)
        except Exception:  # pragma: no cover - 旧 psycopg 兼容
            return json.dumps(value, ensure_ascii=False)

    def validate_fact_key(self, *, fact_type: str, fact_key: str) -> None:
        clean_type = str(fact_type).strip()
        clean_key = str(fact_key).strip()
        if not clean_type:
            raise ValueError("fact_type 不能为空。")
        if not clean_key:
            raise ValueError("fact_key 不能为空。")
        allowed = self.fact_whitelist.get(clean_type)
        if self.strict_whitelist and (
            allowed is None or ("*" not in allowed and clean_key not in allowed)
        ):
            raise ValueError(
                f"长期记忆不允许保存事实 {clean_type}.{clean_key}。"
            )

    @staticmethod
    def _row_to_fact(row: dict[str, Any] | None) -> LongTermFact | None:
        if row is None:
            return None
        normalized = dict(row)
        normalized.setdefault("source_message_id", None)
        normalized.setdefault("status", "active")
        normalized.setdefault("version", 1)
        normalized.setdefault("is_user_confirmed", False)
        normalized.setdefault("metadata", {})
        return LongTermFact(**normalized)

    @staticmethod
    def _row_to_history(
        row: dict[str, Any] | None,
    ) -> LongTermFactHistory | None:
        return LongTermFactHistory(**dict(row)) if row else None

    def init_schema(self) -> None:
        sql = """
        CREATE TABLE IF NOT EXISTS user_memory_facts (
            id BIGSERIAL PRIMARY KEY,
            tenant_id TEXT NOT NULL DEFAULT 'default',
            user_id TEXT NOT NULL,
            fact_type TEXT NOT NULL,
            fact_key TEXT NOT NULL,
            fact_value JSONB NOT NULL,
            confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
            source_thread_id TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT user_memory_facts_confidence_check
                CHECK (confidence >= 0 AND confidence <= 1),
            CONSTRAINT user_memory_facts_unique_key
                UNIQUE (tenant_id, user_id, fact_type, fact_key)
        );

        ALTER TABLE user_memory_facts
            ADD COLUMN IF NOT EXISTS source_message_id TEXT;
        ALTER TABLE user_memory_facts
            ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active';
        ALTER TABLE user_memory_facts
            ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1;
        ALTER TABLE user_memory_facts
            ADD COLUMN IF NOT EXISTS is_user_confirmed BOOLEAN NOT NULL DEFAULT FALSE;
        ALTER TABLE user_memory_facts
            ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

        CREATE TABLE IF NOT EXISTS user_memory_fact_history (
            id BIGSERIAL PRIMARY KEY,
            fact_id BIGINT NOT NULL,
            tenant_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            fact_type TEXT NOT NULL,
            fact_key TEXT NOT NULL,
            fact_value JSONB NOT NULL,
            confidence DOUBLE PRECISION NOT NULL,
            source_thread_id TEXT,
            source_message_id TEXT,
            status TEXT NOT NULL,
            version INTEGER NOT NULL,
            is_user_confirmed BOOLEAN NOT NULL DEFAULT FALSE,
            change_reason TEXT NOT NULL DEFAULT 'updated',
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            archived_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_user_memory_facts_user
            ON user_memory_facts (tenant_id, user_id);
        CREATE INDEX IF NOT EXISTS idx_user_memory_facts_type
            ON user_memory_facts (tenant_id, user_id, fact_type);
        CREATE INDEX IF NOT EXISTS idx_user_memory_facts_updated_at
            ON user_memory_facts (updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_user_memory_fact_history_lookup
            ON user_memory_fact_history
            (tenant_id, user_id, fact_type, fact_key, version DESC);
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()

    def _archive_current(
        self,
        *,
        cur: Any,
        current: dict[str, Any],
        change_reason: str,
        archived_status: str = "superseded",
    ) -> None:
        cur.execute(
            """
            INSERT INTO user_memory_fact_history (
                fact_id, tenant_id, user_id, fact_type, fact_key,
                fact_value, confidence, source_thread_id,
                source_message_id, status, version, is_user_confirmed,
                change_reason, metadata
            ) VALUES (
                %(id)s, %(tenant_id)s, %(user_id)s, %(fact_type)s,
                %(fact_key)s, %(fact_value)s, %(confidence)s,
                %(source_thread_id)s, %(source_message_id)s,
                %(status)s, %(version)s, %(is_user_confirmed)s,
                %(change_reason)s, %(metadata)s
            );
            """,
            {
                **current,
                "fact_value": self._json_value(current.get("fact_value") or {}),
                "metadata": self._json_value(current.get("metadata") or {}),
                "status": archived_status,
                "change_reason": change_reason,
            },
        )

    def upsert_fact(
        self,
        *,
        user_id: str,
        fact_type: str,
        fact_key: str,
        fact_value: dict[str, Any],
        tenant_id: str = "default",
        confidence: float = 1.0,
        source_thread_id: str | None = None,
        source_message_id: str | None = None,
        is_user_confirmed: bool = False,
        change_reason: str = "updated",
        metadata: dict[str, Any] | None = None,
        force: bool = False,
    ) -> LongTermFact:
        clean_user = str(user_id).strip()
        clean_tenant = str(tenant_id).strip() or "default"
        clean_type = str(fact_type).strip()
        clean_key = str(fact_key).strip()
        if not clean_user:
            raise ValueError("user_id 不能为空。")
        self.validate_fact_key(fact_type=clean_type, fact_key=clean_key)
        if not 0 <= float(confidence) <= 1:
            raise ValueError("confidence 必须在 0 到 1 之间。")

        clean_value = sanitize_personal_value(fact_value)
        if not isinstance(clean_value, dict):
            raise ValueError("fact_value 必须是对象。")
        clean_metadata = sanitize_personal_value(metadata or {})

        select_sql = """
        SELECT id, tenant_id, user_id, fact_type, fact_key, fact_value,
               confidence, source_thread_id, source_message_id, status,
               version, is_user_confirmed, metadata,
               created_at::text AS created_at,
               updated_at::text AS updated_at
        FROM user_memory_facts
        WHERE tenant_id=%(tenant_id)s AND user_id=%(user_id)s
          AND fact_type=%(fact_type)s AND fact_key=%(fact_key)s
        FOR UPDATE;
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                params = {
                    "tenant_id": clean_tenant,
                    "user_id": clean_user,
                    "fact_type": clean_type,
                    "fact_key": clean_key,
                }
                cur.execute(select_sql, params)
                current = cur.fetchone()

                if current:
                    # 用户确认事实不会被较低置信度的自动抽取静默覆盖。
                    if (
                        not force
                        and bool(current.get("is_user_confirmed"))
                        and not is_user_confirmed
                    ):
                        return self._row_to_fact(current)  # type: ignore[return-value]
                    if (
                        not force
                        and not is_user_confirmed
                        and float(confidence) < float(current.get("confidence") or 0)
                    ):
                        return self._row_to_fact(current)  # type: ignore[return-value]

                    self._archive_current(
                        cur=cur,
                        current=dict(current),
                        change_reason=change_reason,
                    )
                    cur.execute(
                        """
                        UPDATE user_memory_facts
                        SET fact_value=%(fact_value)s,
                            confidence=%(confidence)s,
                            source_thread_id=%(source_thread_id)s,
                            source_message_id=%(source_message_id)s,
                            status='active',
                            version=version+1,
                            is_user_confirmed=%(is_user_confirmed)s,
                            metadata=%(metadata)s,
                            updated_at=NOW()
                        WHERE id=%(id)s
                        RETURNING id, tenant_id, user_id, fact_type, fact_key,
                                  fact_value, confidence, source_thread_id,
                                  source_message_id, status, version,
                                  is_user_confirmed, metadata,
                                  created_at::text AS created_at,
                                  updated_at::text AS updated_at;
                        """,
                        {
                            "id": current["id"],
                            "fact_value": self._json_value(clean_value),
                            "confidence": float(confidence),
                            "source_thread_id": source_thread_id,
                            "source_message_id": source_message_id,
                            "is_user_confirmed": bool(is_user_confirmed),
                            "metadata": self._json_value(clean_metadata),
                        },
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO user_memory_facts (
                            tenant_id, user_id, fact_type, fact_key,
                            fact_value, confidence, source_thread_id,
                            source_message_id, status, version,
                            is_user_confirmed, metadata
                        ) VALUES (
                            %(tenant_id)s, %(user_id)s, %(fact_type)s,
                            %(fact_key)s, %(fact_value)s, %(confidence)s,
                            %(source_thread_id)s, %(source_message_id)s,
                            'active', 1, %(is_user_confirmed)s, %(metadata)s
                        )
                        RETURNING id, tenant_id, user_id, fact_type, fact_key,
                                  fact_value, confidence, source_thread_id,
                                  source_message_id, status, version,
                                  is_user_confirmed, metadata,
                                  created_at::text AS created_at,
                                  updated_at::text AS updated_at;
                        """,
                        {
                            **params,
                            "fact_value": self._json_value(clean_value),
                            "confidence": float(confidence),
                            "source_thread_id": source_thread_id,
                            "source_message_id": source_message_id,
                            "is_user_confirmed": bool(is_user_confirmed),
                            "metadata": self._json_value(clean_metadata),
                        },
                    )
                row = cur.fetchone()
            conn.commit()
        return self._row_to_fact(row)  # type: ignore[return-value]

    def list_facts(
        self,
        *,
        user_id: str,
        tenant_id: str = "default",
        fact_type: str | None = None,
        include_deleted: bool = False,
    ) -> list[LongTermFact]:
        clauses = ["tenant_id=%(tenant_id)s", "user_id=%(user_id)s"]
        params: dict[str, Any] = {
            "tenant_id": tenant_id.strip() or "default",
            "user_id": user_id.strip(),
        }
        if fact_type:
            clauses.append("fact_type=%(fact_type)s")
            params["fact_type"] = fact_type.strip()
        if not include_deleted:
            clauses.append("status='active'")
        sql = f"""
        SELECT id, tenant_id, user_id, fact_type, fact_key, fact_value,
               confidence, source_thread_id, source_message_id, status,
               version, is_user_confirmed, metadata,
               created_at::text AS created_at,
               updated_at::text AS updated_at
        FROM user_memory_facts
        WHERE {' AND '.join(clauses)}
        ORDER BY updated_at DESC, id DESC;
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [self._row_to_fact(row) for row in rows if row]  # type: ignore[misc]

    def get_fact(
        self,
        *,
        user_id: str,
        fact_type: str,
        fact_key: str,
        tenant_id: str = "default",
        include_deleted: bool = False,
    ) -> LongTermFact | None:
        sql = """
        SELECT id, tenant_id, user_id, fact_type, fact_key, fact_value,
               confidence, source_thread_id, source_message_id, status,
               version, is_user_confirmed, metadata,
               created_at::text AS created_at,
               updated_at::text AS updated_at
        FROM user_memory_facts
        WHERE tenant_id=%(tenant_id)s AND user_id=%(user_id)s
          AND fact_type=%(fact_type)s AND fact_key=%(fact_key)s
        """ + ("" if include_deleted else " AND status='active'") + " LIMIT 1;"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    {
                        "tenant_id": tenant_id.strip() or "default",
                        "user_id": user_id.strip(),
                        "fact_type": fact_type.strip(),
                        "fact_key": fact_key.strip(),
                    },
                )
                row = cur.fetchone()
        return self._row_to_fact(row)

    def list_fact_history(
        self,
        *,
        user_id: str,
        fact_type: str,
        fact_key: str,
        tenant_id: str = "default",
        limit: int = 100,
    ) -> list[LongTermFactHistory]:
        sql = """
        SELECT id, fact_id, tenant_id, user_id, fact_type, fact_key,
               fact_value, confidence, source_thread_id,
               source_message_id, status, version, is_user_confirmed,
               change_reason, metadata,
               archived_at::text AS archived_at
        FROM user_memory_fact_history
        WHERE tenant_id=%(tenant_id)s AND user_id=%(user_id)s
          AND fact_type=%(fact_type)s AND fact_key=%(fact_key)s
        ORDER BY version DESC, id DESC
        LIMIT %(limit)s;
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    {
                        "tenant_id": tenant_id.strip() or "default",
                        "user_id": user_id.strip(),
                        "fact_type": fact_type.strip(),
                        "fact_key": fact_key.strip(),
                        "limit": min(max(int(limit), 1), 500),
                    },
                )
                rows = cur.fetchall()
        return [LongTermFactHistory(**dict(row)) for row in rows]

    def delete_fact(
        self,
        *,
        user_id: str,
        fact_type: str,
        fact_key: str,
        tenant_id: str = "default",
        hard_delete: bool = False,
        change_reason: str = "user_deleted",
    ) -> bool:
        current = self.get_fact(
            user_id=user_id,
            tenant_id=tenant_id,
            fact_type=fact_type,
            fact_key=fact_key,
            include_deleted=True,
        )
        if current is None:
            return False
        with self._connect() as conn:
            with conn.cursor() as cur:
                if hard_delete:
                    # 隐私意义上的硬删除必须同时清理所有历史版本。
                    cur.execute(
                        """
                        DELETE FROM user_memory_fact_history
                        WHERE tenant_id=%(tenant_id)s AND user_id=%(user_id)s
                          AND fact_type=%(fact_type)s AND fact_key=%(fact_key)s;
                        """,
                        {
                            "tenant_id": tenant_id.strip() or "default",
                            "user_id": user_id.strip(),
                            "fact_type": fact_type.strip(),
                            "fact_key": fact_key.strip(),
                        },
                    )
                    cur.execute(
                        "DELETE FROM user_memory_facts WHERE id=%(id)s;",
                        {"id": current.id},
                    )
                else:
                    current_dict = current.to_dict()
                    self._archive_current(
                        cur=cur,
                        current=current_dict,
                        change_reason=change_reason,
                        archived_status="deleted",
                    )
                    cur.execute(
                        """
                        UPDATE user_memory_facts
                        SET status='deleted', version=version+1,
                            updated_at=NOW()
                        WHERE id=%(id)s;
                        """,
                        {"id": current.id},
                    )
            conn.commit()
        return True

    def delete_user_facts(
        self,
        *,
        user_id: str,
        tenant_id: str = "default",
        hard_delete: bool = True,
    ) -> int:
        params = {
            "tenant_id": tenant_id.strip() or "default",
            "user_id": user_id.strip(),
        }
        with self._connect() as conn:
            with conn.cursor() as cur:
                if hard_delete:
                    # 先统计当前事实，再清理当前值和历史快照。
                    cur.execute(
                        """
                        SELECT COUNT(*) AS count FROM user_memory_facts
                        WHERE tenant_id=%(tenant_id)s AND user_id=%(user_id)s;
                        """,
                        params,
                    )
                    row = cur.fetchone() or {"count": 0}
                    count = int(row["count"] or 0)
                    cur.execute(
                        """
                        DELETE FROM user_memory_fact_history
                        WHERE tenant_id=%(tenant_id)s AND user_id=%(user_id)s;
                        """,
                        params,
                    )
                    cur.execute(
                        """
                        DELETE FROM user_memory_facts
                        WHERE tenant_id=%(tenant_id)s AND user_id=%(user_id)s;
                        """,
                        params,
                    )
                else:
                    cur.execute(
                        """
                        UPDATE user_memory_facts
                        SET status='deleted', version=version+1, updated_at=NOW()
                        WHERE tenant_id=%(tenant_id)s AND user_id=%(user_id)s
                          AND status <> 'deleted';
                        """,
                        params,
                    )
                    count = int(cur.rowcount or 0)
            conn.commit()
        return count

    clear_user = delete_user_facts

    def export_user_facts(
        self, *, user_id: str, tenant_id: str = "default"
    ) -> dict[str, Any]:
        facts = self.list_facts(user_id=user_id, tenant_id=tenant_id)
        return {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "facts": [fact.to_dict() for fact in facts],
        }
