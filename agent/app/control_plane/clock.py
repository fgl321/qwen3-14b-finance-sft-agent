from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


def utc_now() -> datetime:
    return datetime.now(UTC)


def require_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timezone-aware UTC datetime is required")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class MonotonicDeadline:
    deadline_at_utc: datetime
    original_budget_ms: int
    monotonic_deadline: float

    @classmethod
    def start(cls, budget_ms: int, *, now_utc: datetime | None = None) -> "MonotonicDeadline":
        if budget_ms <= 0:
            raise ValueError("budget_ms must be positive")
        wall_now = require_utc(now_utc or utc_now())
        return cls(
            deadline_at_utc=wall_now + timedelta(milliseconds=budget_ms),
            original_budget_ms=budget_ms,
            monotonic_deadline=time.monotonic() + budget_ms / 1000,
        )

    @classmethod
    def restore(
        cls,
        *,
        deadline_at_utc: datetime,
        original_budget_ms: int,
        now_utc: datetime | None = None,
    ) -> "MonotonicDeadline":
        wall_now = require_utc(now_utc or utc_now())
        deadline = require_utc(deadline_at_utc)
        remaining_ms = max(0, int((deadline - wall_now).total_seconds() * 1000))
        remaining_ms = min(remaining_ms, original_budget_ms)
        return cls(
            deadline_at_utc=deadline,
            original_budget_ms=original_budget_ms,
            monotonic_deadline=time.monotonic() + remaining_ms / 1000,
        )

    def remaining_ms(self) -> int:
        return max(0, int((self.monotonic_deadline - time.monotonic()) * 1000))

    def expired(self) -> bool:
        return self.remaining_ms() <= 0
