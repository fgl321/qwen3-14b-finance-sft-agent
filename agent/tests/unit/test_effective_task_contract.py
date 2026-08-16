from __future__ import annotations

from app.agent_graph.conversation_state import (
    EffectiveTaskContract,
    MemoryPromotionGate,
    PolicySnapshot,
    TaskFact,
    apply_turn_patch,
    build_effective_task_contract,
    build_result_artifact,
    default_conversation_state,
    reconcile_facts,
)
from app.agent_graph.semantic_route import (
    ConstraintUpdate,
    FactUpdate,
    SemanticRouteDecision,
)


def _route(
    *,
    fact_updates: list[FactUpdate] | None = None,
    extracted_facts: list[FactUpdate] | None = None,
    constraint_updates: list[ConstraintUpdate] | None = None,
    relation: str = "refinement",
    authority: dict | None = None,
) -> SemanticRouteDecision:
    data = {
        "orchestration_mode": "tool",
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
        "needs_exact_calculation": True,
        "conversation_relation": relation,
        "resolved_goal": "首付款改为25万",
        "fact_updates": fact_updates or [],
        "extracted_facts": extracted_facts or [],
        "constraint_updates": constraint_updates or [],
        "confidence": 0.9,
        "reason_summary": "test",
    }
    if authority is not None:
        data["source_authority"] = authority
    return SemanticRouteDecision(**data)


def test_new_task_does_not_inherit_old_contract() -> None:
    state = default_conversation_state()
    state = apply_turn_patch(
        state=state,
        route=_route(
            relation="new_task",
            fact_updates=[
                FactUpdate(
                    field="cash",
                    operation="set",
                    value=900000,
                )
            ],
            authority={
                "current_user_facts": "allowed",
                "selected_documents": "allowed",
                "deterministic_derivation": "allowed",
                "memory": "forbidden",
                "general_model_knowledge": "forbidden",
                "domain_heuristics": "forbidden",
                "web": "forbidden",
            },
        ),
    )
    old_handle = state.active_task.handle
    old_facts = list(state.active_task.canonical_facts)
    state.turn_count = 1

    state = apply_turn_patch(
        state=state,
        route=_route(
            relation="new_task",
            fact_updates=[
                FactUpdate(
                    field="cash",
                    operation="set",
                    value=500000,
                )
            ],
        ),
    )
    state.turn_count = 1
    assert state.active_task is not None
    assert state.active_task.handle != old_handle
    cash_facts = [
        fact
        for fact in state.active_task.canonical_facts
        if fact.field == "cash"
    ]
    assert len(cash_facts) == 1
    assert cash_facts[0].value == 500000
    assert not any(
        fact.field == "cash" and fact.value == 900000
        for fact in state.active_task.canonical_facts
    )

    contract = build_effective_task_contract(
        state=state,
        route=_route(relation="new_task"),
        resolved_resources=[],
    )
    assert contract.source_authority.get("memory") != "forbidden"


def test_continuation_still_inherits_authority() -> None:
    state = default_conversation_state()
    state = apply_turn_patch(
        state=state,
        route=_route(
            relation="new_task",
            authority={
                "current_user_facts": "allowed",
                "selected_documents": "allowed",
                "deterministic_derivation": "allowed",
                "memory": "forbidden",
                "general_model_knowledge": "forbidden",
                "domain_heuristics": "forbidden",
                "web": "forbidden",
            },
        ),
    )
    state.turn_count = 1
    contract = build_effective_task_contract(
        state=state,
        route=_route(relation="follow_up"),
        resolved_resources=[],
    )
    assert contract.source_authority["memory"] == "forbidden"
    assert contract.memory_policy == "forbidden"


def test_constraint_update_patches_inherited_authority() -> None:
    state = default_conversation_state()
    state = apply_turn_patch(
        state=state,
        route=_route(
            relation="new_task",
            authority={
                "current_user_facts": "allowed",
                "selected_documents": "allowed",
                "deterministic_derivation": "allowed",
                "memory": "forbidden",
                "general_model_knowledge": "forbidden",
                "domain_heuristics": "forbidden",
                "web": "forbidden",
            },
        ),
    )
    changed = _route(
        relation="follow_up",
        constraint_updates=[
            ConstraintUpdate(
                name="general_model_knowledge",
                value="allowed",
            )
        ],
    )
    contract = build_effective_task_contract(
        state=state,
        route=changed,
        resolved_resources=[],
    )
    assert contract.source_authority["memory"] == "forbidden"
    assert contract.source_authority["domain_heuristics"] == (
        "forbidden"
    )
    assert contract.source_authority["general_model_knowledge"] == (
        "allowed"
    )
    assert contract.source_authority["web"] == "forbidden"


def test_typed_contract_boundary_no_enum_leak() -> None:
    state = default_conversation_state()
    state = apply_turn_patch(
        state=state,
        route=_route(
            relation="new_task",
            authority={
                "current_user_facts": "allowed",
                "selected_documents": "allowed",
                "deterministic_derivation": "allowed",
                "memory": "allowed",
                "general_model_knowledge": "allowed",
                "domain_heuristics": "allowed",
                "web": "forbidden",
            },
        ),
    )
    contract = build_effective_task_contract(
        state=state,
        route=_route(relation="follow_up"),
        resolved_resources=[],
    )
    assert contract.memory_policy == "not_needed"
    assert contract.memory_policy != "allowed"
    assert contract.web_policy == "forbidden"


def test_original_route_helper_unchanged() -> None:
    route = _route()
    assert route.conversation_relation == "refinement"
    assert route.required_capabilities == ["financial_calculation"]


def test_apply_turn_patch_supersedes_old_fact() -> None:
    state = default_conversation_state()
    state = apply_turn_patch(
        state=state,
        route=_route(
            fact_updates=[
                FactUpdate(
                    field="down_payment",
                    operation="set",
                    value=200000,
                )
            ]
        ),
    )
    state = apply_turn_patch(
        state=state,
        route=_route(
            fact_updates=[
                FactUpdate(
                    field="down_payment",
                    operation="replace",
                    value=250000,
                )
            ]
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
    assert active[0].value == 250000
    assert superseded[0].value == 200000
    assert superseded[0].status == "superseded"


def test_extracted_facts_seed_new_task() -> None:
    state = default_conversation_state()
    state = apply_turn_patch(
        state=state,
        route=_route(
            relation="new_task",
            extracted_facts=[
                FactUpdate(
                    field="cash",
                    operation="set",
                    value=900000,
                ),
                FactUpdate(
                    field="down_payment",
                    operation="set",
                    value=200000,
                ),
            ],
        ),
    )
    assert state.active_task is not None
    by_field = {
        fact.field: fact
        for fact in state.active_task.canonical_facts
    }
    assert by_field["cash"].value == 900000
    assert by_field["down_payment"].value == 200000


def test_effective_contract_inherits_authority_when_unmodified() -> None:
    state = default_conversation_state()
    state = apply_turn_patch(state=state, route=_route())
    assert state.active_task is not None
    state.active_task.source_authority = {
        "current_user_facts": "allowed",
        "selected_documents": "allowed",
        "deterministic_derivation": "allowed",
        "memory": "forbidden",
        "general_model_knowledge": "forbidden",
        "domain_heuristics": "forbidden",
        "web": "forbidden",
    }
    contract = build_effective_task_contract(
        state=state,
        route=_route(),
        resolved_resources=[],
    )
    assert contract.source_authority["memory"] == "forbidden"
    assert contract.memory_policy == "forbidden"

    changed = _route(
        constraint_updates=[
            ConstraintUpdate(
                name="web",
                value="allowed",
            )
        ]
    )
    contract_changed = build_effective_task_contract(
        state=state,
        route=changed,
        resolved_resources=[],
    )
    assert contract_changed.web_policy == "not_needed"


def test_reconcile_facts_precedence_and_memory_gate() -> None:
    contract = EffectiveTaskContract(
        goal="计算",
        canonical_facts=[
            TaskFact(
                field="cash",
                value=900000,
                source="task_canonical",
                updated_turn=1,
            ),
            TaskFact(
                field="down_payment",
                value=250000,
                source="current_turn",
                updated_turn=2,
            ),
        ],
        memory_policy="optional",
        source_authority={},
    )
    reconciled, shadowed = reconcile_facts(
        contract=contract,
        memory_facts=[
            {
                "fact_key": "available_assets",
                "fact_value": 300000,
            },
            {
                "fact_key": "cash",
                "fact_value": 100000,
            },
        ],
    )
    by_field = {fact.field: fact for fact in reconciled}
    assert by_field["cash"].value == 900000
    assert by_field["down_payment"].value == 250000
    assert by_field["available_assets"].value == 300000
    assert any("cash shadowed" in item for item in shadowed)

    forbidden = contract.model_copy(
        update={"memory_policy": "forbidden"}
    )
    reconciled_forbidden, _ = reconcile_facts(
        contract=forbidden,
        memory_facts=[
            {
                "fact_key": "available_assets",
                "fact_value": 300000,
            }
        ],
    )
    assert all(
        fact.field != "available_assets"
        for fact in reconciled_forbidden
    )


def test_memory_promotion_gate() -> None:
    gate = MemoryPromotionGate()
    ok, _reason = gate.may_promote(
        fact_type="identity",
        fact_key="age",
        fact_value={"value": 35},
    )
    assert ok is True

    ok, reason = gate.may_promote(
        fact_type="calculation",
        fact_key="remaining_after_down_payment",
        fact_value={"value": 700000},
    )
    assert ok is False
    assert "fact_type_not_promotable" in reason

    ok, reason = gate.may_promote(
        fact_type="personal",
        fact_key="calc:remaining",
        fact_value={"value": 700000},
    )
    assert ok is False
    assert "ephemeral_key" in reason


def test_policy_snapshot_shape() -> None:
    snapshot = PolicySnapshot(
        tenant_id="personal",
        user_id="owner",
        knowledge_base_id="kb_finance_basic",
        max_scope_document_ids=["doc-a"],
    )
    assert snapshot.allow_web is False
    assert snapshot.max_agent_rounds == 3


def test_build_result_artifact_structured() -> None:
    route = _route()
    artifact = build_result_artifact(
        result={
            "tool_results": [
                {
                    "tool_call_id": "call_1",
                    "tool_name": "yearly_expense_to_monthly",
                    "success": True,
                    "output": {"monthly": "15000.00"},
                }
            ],
            "citations": [
                {
                    "citation_id": 1,
                    "document_id": "doc-a",
                    "file_name": "a.pdf",
                },
                {
                    "citation_id": 2,
                    "document_id": "doc-b",
                    "file_name": "b.pdf",
                },
            ],
            "final_response_result": {
                "synthesis": {
                    "used_citation_ids": ["2"],
                    "used_result_artifact_refs": [
                        "RESULT_1.CLAIM_1"
                    ],
                    "primary_response_focus": {
                        "type": "calculation",
                        "handle": "CALC_1",
                    },
                    "case_verdicts": {
                        "case_a": "determined",
                        "case_b": "conditional",
                    },
                }
            },
        },
        route=route,
    )
    assert len(artifact["calculations"]) == 1
    assert artifact["calculations"][0]["tool_name"] == (
        "yearly_expense_to_monthly"
    )
    assert len(artifact["citations"]) == 1
    assert artifact["citations"][0]["citation_id"] == "2"
    assert len(artifact["conclusions"]) == 2
    assert artifact["conclusions"][0]["handle"] == "CONCLUSION_1"
    assert artifact["claims"][0]["handle"] == "CLAIM_1"
    assert artifact["claims"][0]["grounding_type"] == (
        "document_citation"
    )
    assert artifact["primary_response_focus"]["handle"] == (
        "CALC_1"
    )
    assert "CALC_1" in artifact["sub_artifact_handles"]
    assert "CLAIM_1" in artifact["sub_artifact_handles"]
    assert "CONCLUSION_1" in artifact["sub_artifact_handles"]
