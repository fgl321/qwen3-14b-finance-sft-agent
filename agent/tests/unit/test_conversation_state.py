from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agent_graph.conversation_state import (
    EffectiveOrchestrationDecision,
    FactUpdate,
    apply_turn_patch,
    build_resource_catalog,
    default_conversation_state,
    resource_handles_to_document_ids,
    update_conversation_state,
)
from app.agent_graph.semantic_route import (
    PendingActionResolution,
    SemanticRouteDecision,
    TaskReference,
)


def _route(
    *,
    relation: str = "new_task",
    task_status: str = "none",
    pending_status: str = "none",
) -> SemanticRouteDecision:
    return SemanticRouteDecision(
        orchestration_mode="direct",
        required_capabilities=["complex_reasoning"],
        task_requirements=[
            {
                "id": "T1",
                "description": "任务",
                "capabilities": ["complex_reasoning"],
                "task_kind": "reasoning",
            }
        ],
        conversation_relation=relation,  # type: ignore[arg-type]
        resolved_goal="测试目标",
        task_reference=TaskReference(
            status=task_status,  # type: ignore[arg-type]
            reference_type=(
                "active_task" if task_status == "resolved" else None
            ),
            task_handle="TASK_1",
            confidence=0.9,
        ),
        pending_action_resolution=PendingActionResolution(
            status=pending_status,  # type: ignore[arg-type]
            action_handle=(
                "ACTION_1" if pending_status != "none" else None
            ),
        ),
        confidence=0.9,
        reason_summary="test",
    )


def test_default_state() -> None:
    state = default_conversation_state()
    assert state.turn_count == 0
    assert state.active_task is None
    assert state.pending_action is None
    assert state.resource_handle_map == {}


def test_resource_catalog_handles_are_stable() -> None:
    candidates_a = [
        {"document_id": "doc-b", "title": "B"},
        {"document_id": "doc-a", "title": "A"},
    ]
    catalog_a, state = build_resource_catalog(candidates_a)
    by_id = {
        "doc-a": None,
        "doc-b": None,
    }
    for ref in catalog_a:
        by_id[state.resource_handle_map[ref.handle]] = ref.handle
    assert by_id["doc-a"] == "DOC_1"
    assert by_id["doc-b"] == "DOC_2"

    catalog_b, state_b = build_resource_catalog(
        [
            {"document_id": "doc-b", "title": "B"},
            {"document_id": "doc-a", "title": "A"},
            {"document_id": "doc-c", "title": "C"},
        ],
        state=state,
    )
    assert state_b.resource_handle_map["DOC_1"] == "doc-a"
    assert state_b.resource_handle_map["DOC_2"] == "doc-b"
    assert state_b.resource_handle_map["DOC_3"] == "doc-c"


def test_resource_handle_validation_rejects_unknown_and_scope() -> None:
    candidates = [{"document_id": "doc-a", "title": "A"}]
    catalog, state = build_resource_catalog(candidates)

    resolved, violations = resource_handles_to_document_ids(
        selected_handles=["DOC_1"],
        catalog=catalog,
        state=state,
        allowed_document_ids=["doc-a"],
    )
    assert resolved == ["doc-a"]
    assert violations == []

    _resolved, violations = resource_handles_to_document_ids(
        selected_handles=["DOC_99"],
        catalog=catalog,
        state=state,
        allowed_document_ids=["doc-a"],
    )
    assert any("unknown_handle:DOC_99" in item for item in violations)

    _resolved, violations = resource_handles_to_document_ids(
        selected_handles=["DOC_1"],
        catalog=catalog,
        state=state,
        allowed_document_ids=["doc-other"],
    )
    assert any("scope_conflict:DOC_1:doc-a" in item for item in violations)


def test_update_state_new_task_creates_task_and_result() -> None:
    state = default_conversation_state()
    updated = update_conversation_state(
        state=state,
        semantic_route=_route(relation="new_task"),
        final_answer="完成了",
        resolved_resources=[],
        completed=True,
    )
    assert updated.turn_count == 1
    assert updated.active_task is not None
    assert updated.active_task.status == "completed"
    assert len(updated.recent_results) == 1
    assert updated.recent_results[0].handle.startswith("RESULT_")


def test_update_state_confirmation_completes_pending_action() -> None:
    state = default_conversation_state()
    state.active_task = None
    updated = update_conversation_state(
        state=state,
        semantic_route=_route(relation="new_task"),
        final_answer="需要确认",
        proposed_action={
            "action_type": "resource_catalog_query",
            "description": "查询文档数量",
        },
        completed=True,
    )
    assert updated.pending_action is not None
    assert updated.pending_action.status == "pending_confirmation"

    confirmed = update_conversation_state(
        state=updated,
        semantic_route=_route(
            relation="confirmation",
            task_status="resolved",
            pending_status="confirmed",
        ),
        final_answer="已执行",
        completed=True,
    )
    assert confirmed.pending_action is None
    assert confirmed.active_task is not None
    assert confirmed.active_task.status == "completed"


def test_new_task_inherits_only_session_and_durable_facts() -> None:
    state = default_conversation_state()
    state = apply_turn_patch(
        state=state,
        route=_route(relation="new_task").model_copy(
            update={
                "fact_updates": [
                    FactUpdate(
                        field="age",
                        operation="set",
                        value=35,
                        scope="session",
                    ),
                    FactUpdate(
                        field="cash",
                        operation="set",
                        value=900000,
                        scope="task",
                    )
                ]
            }
        ),
    )
    assert state.active_task is not None
    first_handle = state.active_task.handle
    assert {
        fact.field for fact in state.active_task.canonical_facts
    } == {"age", "cash"}

    state.turn_count = 1
    state = apply_turn_patch(
        state=state,
        route=_route(relation="new_task"),
    )
    assert state.active_task is not None
    assert state.active_task.handle != first_handle
    fields = {
        fact.field for fact in state.active_task.canonical_facts
    }
    assert "age" in fields
    assert "cash" not in fields


def test_same_value_fact_write_is_idempotent_noop() -> None:
    state = default_conversation_state()
    state = apply_turn_patch(
        state=state,
        route=_route(relation="new_task").model_copy(
            update={
                "fact_updates": [
                    FactUpdate(
                        field="age",
                        operation="set",
                        value=35,
                        scope="session",
                    ),
                    FactUpdate(
                        field="age",
                        operation="set",
                        value=35,
                        scope="session",
                    ),
                ]
            }
        ),
    )
    assert state.active_task is not None
    age_facts = [
        fact
        for fact in state.active_task.canonical_facts
        if fact.field == "age"
    ]
    assert len(age_facts) == 1
    assert age_facts[0].value == 35
    assert len(state.active_task.superseded_facts) == 0


def test_duplicate_fact_updates_and_extracted_facts_canonicalized() -> None:
    state = default_conversation_state()
    state = apply_turn_patch(
        state=state,
        route=_route(relation="new_task").model_copy(
            update={
                "fact_updates": [
                    FactUpdate(
                        field="cash",
                        operation="set",
                        value=900000,
                    )
                ],
                "extracted_facts": [
                    FactUpdate(
                        field="cash",
                        operation="set",
                        value=900000,
                    )
                ],
            }
        ),
    )
    assert state.active_task is not None
    cash_facts = [
        fact
        for fact in state.active_task.canonical_facts
        if fact.field == "cash"
    ]
    assert len(cash_facts) == 1
    assert cash_facts[0].value == 900000
    assert state.active_task.superseded_facts == []


def test_replace_preserves_existing_scope() -> None:
    state = default_conversation_state()
    state = apply_turn_patch(
        state=state,
        route=_route(relation="new_task").model_copy(
            update={
                "fact_updates": [
                    FactUpdate(
                        field="down_payment",
                        operation="set",
                        value=200000,
                        scope="task",
                    )
                ]
            }
        ),
    )
    state.turn_count = 1
    state = apply_turn_patch(
        state=state,
        route=_route(relation="continuation").model_copy(
            update={
                "fact_updates": [
                    FactUpdate(
                        field="down_payment",
                        operation="replace",
                        value=250000,
                        scope=None,
                    )
                ]
            }
        ),
    )
    assert state.active_task is not None
    active = [
        fact
        for fact in state.active_task.canonical_facts
        if fact.field == "down_payment"
    ]
    superseded = [
        fact
        for fact in state.active_task.superseded_facts
        if fact.field == "down_payment"
    ]
    assert len(active) == 1
    assert active[0].value == 250000
    assert active[0].scope == "task"
    assert len(superseded) == 1
    assert superseded[0].value == 200000
    assert superseded[0].scope == "task"


def test_first_create_without_scope_defaults_task() -> None:
    state = default_conversation_state()
    state = apply_turn_patch(
        state=state,
        route=_route(relation="new_task").model_copy(
            update={
                "state_update_only": True,
                "fact_updates": [
                    FactUpdate(
                        field="age",
                        operation="set",
                        value=35,
                        scope=None,
                    )
                ],
            }
        ),
    )
    assert state.active_task is not None
    age = [
        fact
        for fact in state.active_task.canonical_facts
        if fact.field == "age"
    ]
    assert len(age) == 1
    assert age[0].scope == "task"


def test_conversational_turn_does_not_allocate_task() -> None:
    state = default_conversation_state()
    updated = update_conversation_state(
        state=state,
        semantic_route=_route(relation="new_task"),
        final_answer="你好！有什么可以帮你的？",
        resolved_resources=[],
        completed=True,
        admit_task=False,
        result_artifact={},
    )
    assert updated.active_task is None
    assert updated.recent_results == []
    assert updated.turn_count == 1


def test_effective_orchestration_consistency() -> None:
    decision = EffectiveOrchestrationDecision(
        mode="clarify",
        needs_clarification=True,
        reason_code="remaining_ambiguity",
    )
    assert decision.needs_clarification is True

    with pytest.raises(ValidationError):
        EffectiveOrchestrationDecision(
            mode="clarify",
            needs_clarification=False,
            reason_code="bad",
        )
