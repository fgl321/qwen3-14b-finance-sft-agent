from __future__ import annotations

import pytest

from app.agent_graph.semantic_route import SemanticRouteDecision
from app.control_plane.enums import (
    Authority,
    EnforcementStrength,
    PermissionLevel,
    RequirementLevel,
)
from app.control_plane.floor import ExplicitConstraintParser
from app.control_plane.production_adapter import ControlPlaneBlocked, production_control_preflight
from app.control_plane.schemas import (
    CapabilityConstraint,
    ConstraintSource,
    ExplicitRequirementFloor,
)


def _route(capability: str = "general_explanation") -> SemanticRouteDecision:
    return SemanticRouteDecision.model_validate({
        "orchestration_mode": "tool" if capability == "financial_calculation" else "direct",
        "required_capabilities": [capability],
        "task_requirements": [{
            "id": "answer_request", "description": "answer safely", "required": True,
            "capabilities": [capability], "evidence_tool_names": [],
            "requires_citations": False,
        }],
        "confidence": 1.0, "reason_summary": "test route",
    })


def test_v2_control_plane_takes_over_gate_and_emits_hash_chain() -> None:
    audit = production_control_preflight(
        request_id="req", run_id="run", user_message="请解释紧急备用金。",
        route=_route(),
    )
    assert audit["mode"] == "v2_execution"
    assert audit["sealed_contract_hash"].startswith("sha256:")
    assert audit["strategy_hash"].startswith("sha256:")


def _web_conflict_floor() -> ExplicitRequirementFloor:
    def constraint(
        requirement: RequirementLevel,
        permission: PermissionLevel,
        identifier: str,
    ) -> CapabilityConstraint:
        return CapabilityConstraint(
            capability="web_access",
            requirement=requirement,
            permission=permission,
            source=ConstraintSource(
                constraint_id=identifier,
                authority=Authority.USER_EXPLICIT,
                enforcement_strength=EnforcementStrength.EXPLICIT_CONSTRAINT,
                rule_id="test-conflict",
            ),
        )

    return ExplicitRequirementFloor(
        request_id="req",
        constraints=(
            constraint(
                RequirementLevel.REQUIRED,
                PermissionLevel.ALLOWED,
                "web-required",
            ),
            constraint(
                RequirementLevel.NOT_NEEDED,
                PermissionLevel.FORBIDDEN,
                "web-forbidden",
            ),
        ),
        parser_version="test",
    )


def test_v2_control_plane_blocks_genuine_required_forbidden_conflict() -> None:
    with pytest.raises(ControlPlaneBlocked) as caught:
        production_control_preflight(
            request_id="req", run_id="run",
            user_message="",
            constraints=_web_conflict_floor(),
            route=_route(),
        )
    assert "CONTRACT_PERMISSION_CONFLICT" in caught.value.reason_codes


def test_v2_control_plane_does_not_fabricate_conflict_from_nl() -> None:
    floor = ExplicitConstraintParser().parse(
        request_id="req",
        user_message="必须联网查询当前信息，但是不要访问互联网。",
    )
    assert floor.constraints == ()
    audit = production_control_preflight(
        request_id="req",
        run_id="run",
        user_message="必须联网查询当前信息，但是不要访问互联网。",
        route=_route(),
    )
    assert audit["strategy_status"] == "ready"
    assert "CONTRACT_PERMISSION_CONFLICT" not in audit["reason_codes"]
