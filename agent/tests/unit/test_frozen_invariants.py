from __future__ import annotations

from app.agent_graph.conversation_state import (
    FactUpdate,
    apply_turn_patch,
    default_conversation_state,
    update_conversation_state,
    validate_referential_integrity,
)
from app.agent_graph.semantic_route import (
    SemanticRouteDecision,
    assess_task_admission,
    validate_semantic_consistency,
)


def _route(
    *,
    state_update_only: bool = False,
    resolved_goal: str | None = None,
    required_capabilities: list[str] | None = None,
    fact_updates: list[FactUpdate] | None = None,
    task_requirements: list[dict] | None = None,
) -> SemanticRouteDecision:
    capabilities = required_capabilities or []
    return SemanticRouteDecision(
        orchestration_mode="direct",
        required_capabilities=capabilities,
        task_requirements=task_requirements or [],
        resolved_goal=resolved_goal,
        fact_updates=fact_updates or [],
        state_update_only=state_update_only,
        confidence=0.9,
        reason_summary="frozen invariant test",
    )


def test_inv_2_3_semantic_contract_invalid_after_repair_fails_closed() -> None:
    route = _route(state_update_only=True)
    assert "state_update_only_without_mutation" in (
        validate_semantic_consistency(route)
    )

    invalid_task = _route(resolved_goal="计算")
    assert "invalid_task_proposal_without_requirements" in (
        validate_semantic_consistency(invalid_task)
    )


def test_inv_4_5_turn_not_task_no_allocation() -> None:
    greeting = _route()
    admission = assess_task_admission(greeting)
    assert admission["admitted"] is False
    assert admission["kind"] == "conversational"

    state = default_conversation_state()
    updated = update_conversation_state(
        state=state,
        semantic_route=greeting,
        final_answer="你好！",
        resolved_resources=[],
        completed=True,
        admit_task=False,
        result_artifact={},
    )
    assert updated.active_task is None
    assert updated.recent_results == []


def test_inv_6_state_update_only_requires_mutation() -> None:
    with_mutation = _route(
        state_update_only=True,
        fact_updates=[
            FactUpdate(
                field="down_payment",
                operation="replace",
                value=250000,
            )
        ],
    )
    assert (
        "state_update_only_without_mutation"
        not in validate_semantic_consistency(with_mutation)
    )


def test_inv_18_19_only_committed_qualified_refs_referenceable() -> None:
    violations = validate_referential_integrity(
        used_fact_refs=[],
        used_derivation_ids=[],
        used_result_artifact_refs=[".CALC_1", "RESULT_99.CALC_1"],
        used_citation_ids=[],
        canonical_fact_fields=[],
        known_derivation_ids=[],
        known_sub_artifact_ids=["RESULT_2.CALC_1"],
        allowed_citation_ids=[],
    )
    assert "sub_artifact:.CALC_1" in violations
    assert "sub_artifact:RESULT_99.CALC_1" in violations
    assert "sub_artifact:RESULT_2.CALC_1" not in violations


def test_inv_13_evidence_statuses_never_conflated() -> None:
    from app.rag.rag_types import EVIDENCE_STATUS_USER_TEXT

    texts = set(EVIDENCE_STATUS_USER_TEXT.values())
    assert len(texts) == len(EVIDENCE_STATUS_USER_TEXT)
    assert (
        EVIDENCE_STATUS_USER_TEXT["not_observed"]
        != EVIDENCE_STATUS_USER_TEXT["insufficient_evidence"]
    )
    assert (
        EVIDENCE_STATUS_USER_TEXT["insufficient_evidence"]
        != EVIDENCE_STATUS_USER_TEXT["technical_unavailable"]
    )


def test_inv_10_11_12_requirement_universe_preserved() -> None:
    from app.api.routes.chat_graph_v2 import (
        _finalize_requirement_coverage,
    )

    rag = {
        "citations": [],
        "physical_queries": [
            {
                "id": "Q1",
                "query": "存款保险",
                "source_requirement_ids": ["E1", "E2"],
                "merged_from_query_ids": [],
            }
        ],
        "requirement_coverage": [
            {
                "requirement_id": "E1",
                "task_id": "T1",
                "status": "direct_support",
                "source_query_ids": ["Q1"],
                "citation_ids": [],
                "conflict_ids": [],
            }
        ],
    }
    rag = _finalize_requirement_coverage(
        rag,
        ["E1", "E2", "E3"],
    )
    assert rag["coverage_integrity"]["status"] == "violated"
    assert rag["coverage_integrity"]["missing_observation_ids"] == [
        "E2",
        "E3",
    ]
    by_id = {
        item["requirement_id"]: item["status"]
        for item in rag["requirement_coverage"]
    }
    assert by_id["E2"] == "not_observed"
    assert by_id["E3"] == "not_observed"


def test_inv_16_calc_satisfied_requires_verified_calc() -> None:
    from app.api.routes.chat_graph_v2 import (
        _apply_completion_contract,
    )

    route = _route(
        required_capabilities=["financial_calculation"],
        task_requirements=[
            {
                "id": "T1",
                "description": "计算",
                "capabilities": ["financial_calculation"],
                "task_kind": "calculation",
            }
        ],
        resolved_goal="计算剩余",
    )
    result = _apply_completion_contract(
        result={
            "status": "completed",
            "final_answer": "剩余70万",
            "tool_results": [],
            "final_response_result": {
                "synthesis": {
                    "used_derivation_ids": ["CALC_1"],
                    "used_fact_refs": [],
                }
            },
        },
        route=route,
        materialized_artifacts=[
            {
                "handle": "CALC_1",
                "artifact_type": "calc",
                "operation": "SUBTRACT",
                "output": None,
                "verification_status": "unsupported_operation",
            }
        ],
        rag_outcome={
            "status": "not_attempted",
            "requirement_coverage": [],
        },
    )
    calc = result["capability_outcomes"]["financial_calculation"]
    assert calc["status"] != "satisfied"
    assert calc["result_refs"] == []


def test_inv_22_23_guard_failure_does_not_rewrite_execution() -> None:
    from app.api.routes.chat_graph_v2 import (
        _apply_completion_contract,
    )

    route = _route(
        required_capabilities=["financial_calculation"],
        task_requirements=[
            {
                "id": "T1",
                "description": "计算",
                "capabilities": ["financial_calculation"],
                "task_kind": "calculation",
            }
        ],
        resolved_goal="计算剩余",
    )
    result = _apply_completion_contract(
        result={
            "status": "fallback",
            "final_answer": "抱歉，暂时无法安全完成。",
            "tool_results": [],
            "final_response_result": {
                "synthesis": {
                    "used_derivation_ids": ["CALC_1"],
                    "used_fact_refs": [],
                }
            },
        },
        route=route,
        materialized_artifacts=[
            {
                "handle": "CALC_1",
                "artifact_type": "calc",
                "operation": "SUBTRACT",
                "output": 700000,
                "verification_status": "verified",
            }
        ],
        rag_outcome={
            "status": "not_attempted",
            "requirement_coverage": [],
        },
    )
    assert result["execution_status"] == "success"
    assert result["status"] == "fallback"
