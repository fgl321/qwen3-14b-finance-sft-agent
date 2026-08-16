from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from app.control_plane.clock import utc_now


RED_LINE_METRICS = (
    "required_drop_count",
    "forbidden_execution_count",
    "silent_scope_expansion_count",
    "guard_false_validated_count",
    "duplicate_side_effect_count",
    "citation_scope_violation_total",
    "contract_execution_violation_total",
)


@dataclass(slots=True)
class ControlPlaneMetrics:
    """Run-local acceptance counters; production exporters may consume snapshots."""

    runtime_revision: str
    schema_versions: dict[str, str]
    measurement_started_at_utc: datetime = field(default_factory=utc_now)
    request_count: int = 0
    eligible_request_count: int = 0
    counters: dict[str, int] = field(default_factory=dict)

    def observe_request(self, *, eligible: bool = True) -> None:
        self.request_count += 1
        self.eligible_request_count += int(eligible)

    def increment(self, name: str, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("metric increments cannot be negative")
        self.counters[name] = self.counters.get(name, 0) + amount

    def count(self, name: str) -> int:
        return self.counters.get(name, 0)

    def rate(self, name: str) -> Decimal:
        denominator = self.eligible_request_count
        return Decimal(self.count(name)) / Decimal(denominator) if denominator else Decimal(0)

    def red_line_violations(self) -> dict[str, int]:
        return {name: self.count(name) for name in RED_LINE_METRICS if self.count(name)}

    def acceptance_passed(self) -> bool:
        return not self.red_line_violations()

    def snapshot(self) -> dict[str, object]:
        names = set(self.counters).union(RED_LINE_METRICS)
        return {
            "measurement_window": {
                "started_at_utc": self.measurement_started_at_utc.isoformat(),
                "ended_at_utc": utc_now().isoformat(),
            },
            "request_count": self.request_count,
            "eligible_request_count": self.eligible_request_count,
            "metrics": {
                name: {"event_count": self.count(name), "rate": self.rate(name)}
                for name in sorted(names)
            },
            "runtime_revision": self.runtime_revision,
            "schema_versions": dict(self.schema_versions),
        }
