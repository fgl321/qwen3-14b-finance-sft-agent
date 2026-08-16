from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.control_plane.canonical import content_hash
from app.control_plane.metrics import ControlPlaneMetrics


ACCEPTANCE_GATES = {
    "PASS_SHADOW",
    "PASS_CANARY",
    "FAIL_CONTRACT_INVARIANT",
    "FAIL_SIDE_EFFECT_SAFETY",
    "FAIL_STATUS_SEMANTICS",
    "FAIL_OBSERVABILITY",
}


@dataclass(frozen=True, slots=True)
class AcceptanceCaseResult:
    test_id: str
    fixture_hash: str
    passed: bool
    expected: dict[str, object]
    actual: dict[str, object]
    reason_code_diff: tuple[str, ...] = ()
    owner_component: str = "control_plane"


def build_acceptance_report(
    *,
    runtime_revision: str,
    schema_versions: dict[str, str],
    cases: tuple[AcceptanceCaseResult, ...],
    metrics: ControlPlaneMetrics,
) -> dict[str, object]:
    red_lines = metrics.red_line_violations()
    gate = "PASS_SHADOW" if all(case.passed for case in cases) and not red_lines else "FAIL_CONTRACT_INVARIANT"
    return {
        "runtime_revision": runtime_revision,
        "schema_versions": schema_versions,
        "cases": [asdict(case) for case in cases],
        "slo": metrics.snapshot(),
        "red_line_violations": red_lines,
        "gate": gate,
    }


def write_acceptance_report(report: dict[str, object], *, output_dir: Path) -> tuple[Path, Path]:
    """Explicit offline artifact writer; never used by request-time Shadow runtime."""
    output_dir.mkdir(parents=True, exist_ok=True)
    revision = str(report["runtime_revision"]).replace("/", "_")
    json_path = output_dir / f"control_plane_acceptance_{revision}.json"
    md_path = output_dir / f"control_plane_acceptance_{revision}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md_path.write_text(
        "# Control Plane Acceptance\n\n"
        f"- Runtime revision: `{report['runtime_revision']}`\n"
        f"- Gate: **{report['gate']}**\n"
        f"- Report hash: `{content_hash(report)}`\n",
        encoding="utf-8",
    )
    return json_path, md_path
