from __future__ import annotations

import pytest

from app.agent_graph.semantic_route import (
    FactUpdate,
    SemanticRouteDecision,
    SemanticRouteProtocolError,
    SemanticRouter,
    assess_task_admission,
    validate_semantic_consistency,
)


def _route(
    *,
    state_update_only: bool = False,
    required_capabilities: list[str] | None = None,
    retrieval_requirement: str = "not_needed",
    capability_constraints: dict | None = None,
    needs_exact_calculation: bool = False,
) -> SemanticRouteDecision:
    capabilities = required_capabilities or [
        "complex_reasoning"
    ]
    return SemanticRouteDecision(
        orchestration_mode="direct",
        required_capabilities=capabilities,
        task_requirements=[
            {
                "id": "T1",
                "description": "任务",
                "capabilities": capabilities,
                "task_kind": "reasoning",
            }
        ],
        retrieval_requirement=retrieval_requirement,  # type: ignore[arg-type]
        capability_constraints=capability_constraints or {},
        needs_exact_calculation=needs_exact_calculation,
        state_update_only=state_update_only,
        confidence=0.9,
        reason_summary="test",
    )


def test_state_update_only_rejects_executable_capability() -> None:
    route = _route(
        state_update_only=True,
        required_capabilities=["financial_calculation"],
    )
    violations = validate_semantic_consistency(route)
    assert (
        "state_update_only_with_executable_capability"
        in violations
    )


def test_state_update_only_allows_passive_memory_read() -> None:
    route = _route(
        state_update_only=True,
        required_capabilities=["memory_read"],
    )
    violations = validate_semantic_consistency(route)
    assert (
        "state_update_only_with_executable_capability"
        not in violations
    )


def test_state_update_only_allows_empty_requirements() -> None:
    route = SemanticRouteDecision(
        orchestration_mode="direct",
        required_capabilities=[],
        task_requirements=[],
        state_update_only=True,
        fact_updates=[
            FactUpdate(
                field="down_payment",
                operation="replace",
                value=250000,
            )
        ],
        confidence=0.9,
        reason_summary="pure state ack",
    )
    violations = validate_semantic_consistency(route)
    assert violations == []


def test_conversational_turn_without_requirements_is_valid() -> None:
    route = SemanticRouteDecision(
        orchestration_mode="direct",
        required_capabilities=[],
        task_requirements=[],
        confidence=0.9,
        reason_summary="greeting",
    )
    violations = validate_semantic_consistency(route)
    assert violations == []


def test_task_goal_without_requirements_rejected() -> None:
    route = SemanticRouteDecision(
        orchestration_mode="direct",
        required_capabilities=[],
        task_requirements=[],
        resolved_goal="计算备用金",
        confidence=0.9,
        reason_summary="task without requirements",
    )
    violations = validate_semantic_consistency(route)
    assert (
        "invalid_task_proposal_without_requirements"
        in violations
    )


def test_state_update_only_without_mutation_rejected() -> None:
    route = SemanticRouteDecision(
        orchestration_mode="direct",
        required_capabilities=[],
        task_requirements=[],
        state_update_only=True,
        confidence=0.9,
        reason_summary="empty state update",
    )
    violations = validate_semantic_consistency(route)
    assert "state_update_only_without_mutation" in violations


def test_task_admission_conversational() -> None:
    route = SemanticRouteDecision(
        orchestration_mode="direct",
        required_capabilities=[],
        task_requirements=[],
        confidence=0.9,
        reason_summary="greeting",
    )
    admission = assess_task_admission(route)
    assert admission["admitted"] is False
    assert admission["kind"] == "conversational"


def test_task_admission_state_mutation() -> None:
    route = SemanticRouteDecision(
        orchestration_mode="direct",
        required_capabilities=[],
        task_requirements=[],
        state_update_only=True,
        fact_updates=[
            FactUpdate(
                field="down_payment",
                operation="replace",
                value=250000,
            )
        ],
        confidence=0.9,
        reason_summary="change down payment",
    )
    admission = assess_task_admission(route)
    assert admission["admitted"] is True
    assert admission["kind"] == "state_mutation"


def test_task_admission_new_task() -> None:
    route = SemanticRouteDecision(
        orchestration_mode="direct",
        required_capabilities=["financial_calculation"],
        task_requirements=[
            {
                "id": "T1",
                "description": "计算",
                "capabilities": ["financial_calculation"],
                "task_kind": "calculation",
            }
        ],
        resolved_goal="计算剩余资金",
        confidence=0.9,
        reason_summary="calc",
    )
    admission = assess_task_admission(route)
    assert admission["admitted"] is True
    assert admission["kind"] == "new_task"


def test_state_update_only_rejects_exact_calculation() -> None:
    route = _route(
        state_update_only=True,
        needs_exact_calculation=True,
    )
    violations = validate_semantic_consistency(route)
    assert (
        "state_update_only_with_exact_calculation"
        in violations
    )


def test_required_retrieval_rejects_not_needed_capability() -> None:
    route = SemanticRouteDecision(
        orchestration_mode="rag",
        required_capabilities=[
            "knowledge_retrieval",
            "complex_reasoning",
        ],
        task_requirements=[
            {
                "id": "T1",
                "description": "检索",
                "capabilities": [
                    "knowledge_retrieval",
                    "complex_reasoning",
                ],
                "requires_citations": True,
                "task_kind": "retrieval",
                "evidence_requirements": ["内容"],
            }
        ],
        retrieval_requirement="required",
        citation_requirement="required",
        grounding_requirement="authoritative",
        retrieval_scope="selected_documents",
        capability_constraints={"knowledge_retrieval": "not_needed"},
        confidence=0.9,
        reason_summary="test",
    )
    violations = validate_semantic_consistency(route)
    assert (
        "required_retrieval_with_not_needed_capability"
        in violations
    )


def test_consistent_direct_route_has_no_violations() -> None:
    route = _route()
    assert validate_semantic_consistency(route) == []


class FakeClient:
    def __init__(self, payload: dict):
        self.payload = payload

    async def chat(self, **kwargs):
        import json

        return {
            "message": {
                "role": "assistant",
                "content": json.dumps(
                    self.payload,
                    ensure_ascii=False,
                ),
            },
            "model": "fake",
            "finish_reason": "stop",
        }


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_router_fails_closed_on_inconsistent_proposal() -> None:
    payload = {
        "orchestration_mode": "direct",
        "required_capabilities": ["financial_calculation"],
        "task_requirements": [
            {
                "id": "T1",
                "description": "计算",
                "capabilities": ["financial_calculation"],
                "evidence_tool_names": ["yearly_expense_to_monthly"],
                "task_kind": "calculation",
            }
        ],
        "state_update_only": True,
        "needs_exact_calculation": True,
        "confidence": 0.9,
        "reason_summary": "test",
    }
    router = SemanticRouter(
        llm_client=FakeClient(payload),  # type: ignore[arg-type]
        max_repairs=0,
    )
    with pytest.raises(SemanticRouteProtocolError):
        await router.route(
            "先记录，不计算",
            tool_catalog=[
                {
                    "name": "yearly_expense_to_monthly",
                    "description": "年度转月度",
                }
            ],
        )
