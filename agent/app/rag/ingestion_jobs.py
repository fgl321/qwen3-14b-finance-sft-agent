from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Literal
from uuid import uuid4


JobStatus = Literal["queued", "processing", "completed", "failed"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IngestionJobStore:
    """Durable, single-node ingestion job registry.

    SQLite is appropriate for this personal deployment: job state survives an
    API restart without introducing a second queue product.  Model-heavy work
    still runs outside the event loop.  A multi-replica deployment can replace
    this class with Celery/Redis without changing the HTTP contract.
    """

    def __init__(self, path: str | Path = "data/ingestion_jobs.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._init_schema()
        self._mark_interrupted_jobs()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL DEFAULT 'queued',
                    progress_percent REAL NOT NULL DEFAULT 0,
                    progress_message TEXT NOT NULL DEFAULT '',
                    file_name TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL,
                    knowledge_base_id TEXT NOT NULL,
                    result_json TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(ingestion_jobs)")
            }
            migrations = {
                "phase": "TEXT NOT NULL DEFAULT 'queued'",
                "progress_percent": "REAL NOT NULL DEFAULT 0",
                "progress_message": "TEXT NOT NULL DEFAULT ''",
            }
            for name, definition in migrations.items():
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE ingestion_jobs ADD COLUMN {name} {definition}"
                    )

    def _mark_interrupted_jobs(self) -> None:
        now = _utc_now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE ingestion_jobs
                SET status='failed', phase='failed',
                    progress_message='服务重启中断了文档处理，请重新上传。',
                    error_code='WORKER_RESTARTED',
                    error_message='服务重启中断了文档处理，请重新上传。', updated_at=?
                WHERE status IN ('queued', 'processing')
                """,
                (now,),
            )

    def create(
        self,
        *,
        file_name: str,
        stored_path: str,
        tenant_id: str,
        owner_user_id: str,
        knowledge_base_id: str,
    ) -> dict[str, Any]:
        job_id = f"ingest_{uuid4().hex}"
        now = _utc_now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingestion_jobs (
                    job_id, status, phase, progress_percent, progress_message,
                    file_name, stored_path, tenant_id,
                    owner_user_id, knowledge_base_id, created_at, updated_at
                ) VALUES (?, 'queued', 'queued', 0, '文件已接收，等待处理', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    file_name,
                    stored_path,
                    tenant_id,
                    owner_user_id,
                    knowledge_base_id,
                    now,
                    now,
                ),
            )
        return self.require(job_id)

    def set_processing(self, job_id: str) -> None:
        self._update(
            job_id,
            status="processing",
            phase="parsing",
            progress_percent=1,
            progress_message="正在解析文档",
        )

    def set_progress(
        self,
        job_id: str,
        *,
        phase: str,
        percent: float,
        message: str,
    ) -> None:
        self._update(
            job_id,
            status="processing",
            phase=phase[:50],
            progress_percent=max(0.0, min(99.0, float(percent))),
            progress_message=message[:300],
        )

    def set_completed(self, job_id: str, result: dict[str, Any]) -> None:
        self._update(
            job_id,
            status="completed",
            phase="completed",
            progress_percent=100,
            progress_message="文档索引已完成",
            result_json=json.dumps(result, ensure_ascii=False, default=str),
            error_code=None,
            error_message=None,
        )

    def set_failed(self, job_id: str, *, code: str, message: str) -> None:
        self._update(
            job_id,
            status="failed",
            phase="failed",
            progress_message=message[:300],
            error_code=code[:100],
            error_message=message[:500],
        )

    def _update(self, job_id: str, **changes: Any) -> None:
        allowed = {
            "status",
            "phase",
            "progress_percent",
            "progress_message",
            "result_json",
            "error_code",
            "error_message",
        }
        fields = {key: value for key, value in changes.items() if key in allowed}
        fields["updated_at"] = _utc_now()
        assignments = ", ".join(f"{key}=?" for key in fields)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE ingestion_jobs SET {assignments} WHERE job_id=?",
                (*fields.values(), job_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(job_id)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ingestion_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        raw_result = payload.pop("result_json", None)
        payload["result"] = json.loads(raw_result) if raw_result else None
        return payload

    def require(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job
