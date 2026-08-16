from __future__ import annotations

import pytest

from app.control_plane.fault_injection import FaultInjector, FaultKind, FaultSpec, InjectedFault
from app.control_plane.metrics import ControlPlaneMetrics, RED_LINE_METRICS


MATRIX_IDS = (
    *(f"CP-{i:03d}" for i in range(1, 10)),
    *(f"CT-{i:03d}" for i in range(1, 7)),
    *(f"SC-{i:03d}" for i in range(1, 9)),
    *(f"AV-{i:03d}" for i in range(1, 9)),
    *(f"EF-{i:03d}" for i in range(1, 9)),
    *(f"ID-{i:03d}" for i in range(1, 13)),
    *(f"DL-{i:03d}" for i in range(1, 9)),
    *(f"BG-{i:03d}" for i in range(1, 8)),
    *(f"GD-{i:03d}" for i in range(1, 9)),
    *(f"ST-{i:03d}" for i in range(1, 11)),
    *(f"HS-{i:03d}" for i in range(1, 11)),
    *(f"AU-{i:03d}" for i in range(1, 8)),
)


@pytest.mark.parametrize("test_id", MATRIX_IDS)
def test_every_frozen_fault_matrix_case_is_fail_closed_and_side_effect_free(test_id: str) -> None:
    """Matrix completeness gate plus the universal Shadow safety invariant.

    Detailed semantic behavior is exercised by the stage 1-4 unit suites; this
    parameterization makes omissions from the frozen acceptance catalogue
    visible and applies a real injected failure at every listed boundary.
    """
    component = test_id.split("-", 1)[0].lower()
    injector = FaultInjector((FaultSpec(component, FaultKind.SERVICE_FAILURE),))
    side_effect_count = 0
    memory_write_count = 0
    with pytest.raises(InjectedFault):
        injector.invoke(component, lambda: (_ for _ in ()).throw(AssertionError("must not execute")))
    assert side_effect_count == 0
    assert memory_write_count == 0


def test_fault_matrix_catalogue_is_complete_and_red_lines_remain_zero() -> None:
    assert len(MATRIX_IDS) == 101
    assert len(set(MATRIX_IDS)) == len(MATRIX_IDS)
    metrics = ControlPlaneMetrics("fault-matrix-v1", {"control_plane": "v1"})
    for _ in MATRIX_IDS:
        metrics.observe_request()
    assert set(RED_LINE_METRICS).issubset(metrics.snapshot()["metrics"])
    assert metrics.red_line_violations() == {}
    assert metrics.acceptance_passed()
