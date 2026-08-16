from __future__ import annotations

import json

import pytest

from app.agent_graph.semantic_route import (
    RequestRequirementContract,
    SemanticRouteDecision,
    SemanticRouter,
    TaskRequirement,
    conservative_route_fallback,
)
from app.agent_graph.schemas.planner_schema import (
    PlannerDecision,
    ToolCallRequest,
    resolve_typed_references,
)
from app.agent_graph.runtime_nodes.tool_executor_node import (
    execute_planner_tool_calls,
)
from app.tools.runtime_registry import build_production_tool_registry
from app.tools.tool_executor import ProductionToolExecutor, ToolExecutionContext
from app.api.routes.chat_graph_v2 import (
    ProductionChatRequest,
    _apply_caller_route_constraints,
    _apply_completion_contract,
    _required_outputs_delivered,
    _rag_outcome,
    _rag_pipeline_metrics,
)


class FakeClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": json.dumps(self.payloads.pop(0))}}


def hybrid_route_payload():
    return {
        "orchestration_mode": "hybrid",
        "required_capabilities": [
            "knowledge_retrieval",
            "financial_calculation",
            "complex_reasoning",
        ],
        "task_requirements": [
            {
                "id": "calculate",
                "description": "calculate verified financial values",
                "required": True,
                "capabilities": ["financial_calculation"],
            },
            {
                "id": "use_document",
                "description": "support material conclusions with document evidence",
                "required": True,
                "capabilities": ["knowledge_retrieval"],
                "evidence_requirements": ["use_document"],
            },
            {
                "id": "answer",
                "description": "integrate the verified results",
                "required": True,
                "capabilities": ["complex_reasoning"],
            },
        ],
        "retrieval_requirement": "required",
        "citation_requirement": "required",
        "grounding_requirement": "authoritative",
        "retrieval_scope": "selected_documents",
        "needs_exact_calculation": True,
        "needs_latest_data": False,
        "needs_clarification": False,
        "risk_level": "medium",
        "confidence": 0.94,
        "ambiguities": [],
        "reason_summary": "The request requires both document evidence and calculations.",
    }


@pytest.mark.anyio
async def test_semantic_router_returns_valid_hybrid_contract():
    client = FakeClient([hybrid_route_payload()])
    route = await SemanticRouter(llm_client=client).route("answer from my file and calculate")
    assert route.orchestration_mode == "hybrid"
    assert route.citation_requirement == "required"
    assert client.calls[0]["response_format"] == {"type": "json_object"}


def test_technical_retrieval_failure_is_not_no_evidence():
    outcome = _rag_outcome(
        {"attempted": True, "degraded": True, "error": "AttributeError"},
        None,
    )
    assert outcome["status"] == "failed_technical"


def test_rag_pipeline_metrics_explains_zero_citations():
    metrics = _rag_pipeline_metrics(
        {
            "retrieved_chunks": [
                {"metadata": {"rerank_probability": 0.42}},
                {"metadata": {"rrf_score": 0.11}},
            ],
            "evidence_assessment": {
                "sufficient": False,
                "reason": "证据与问题不相关。",
                "relevant_evidence_numbers": [],
            },
            "citations": [],
        }
    )
    assert metrics["retrieved_count"] == 2
    assert metrics["reranked_count"] == 2
    assert metrics["sufficient_evidence_count"] == 0
    assert metrics["citation_count"] == 0
    assert metrics["evidence_rejection_reason"] == "证据与问题不相关。"


def test_completion_contract_marks_required_evidence_partial():
    client = FakeClient([])
    route = __import__(
        "app.agent_graph.semantic_route", fromlist=["SemanticRouteDecision"]
    ).SemanticRouteDecision.model_validate(hybrid_route_payload())
    result = _apply_completion_contract(
        result={
            "status": "completed",
            "final_answer": "The verified calculation completed.",
            "tool_results": [
                {
                    "tool_call_id": "calc_1",
                    "tool_name": "yearly_expense_to_monthly",
                    "success": True,
                    "output": {"monthly_necessary_expense": 15000},
                }
            ],
        },
        route=route,
        rag_outcome={
            "status": "failed_technical",
            "error_code": "AttributeError",
            "retrieved_count": 0,
            "citation_count": 0,
        },
    )
    assert result["status"] == "partial"
    assert result["fulfillment_status"] == "partial"
    assert "required_citations" in result["completion_contract"]["missing_requirements"]
    assert result["claim_ledger"][0]["source_ids"] == ["calc_1"]


def test_completion_contract_no_evidence_still_completes_citation_validation():
    route = __import__(
        "app.agent_graph.semantic_route", fromlist=["SemanticRouteDecision"]
    ).SemanticRouteDecision.model_validate(hybrid_route_payload())
    result = _apply_completion_contract(
        result={
            "status": "completed",
            "final_answer": "已检索指定文档，但没有找到足够依据支持该结论。",
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
            "status": "completed_no_evidence",
            "execution_ok": True,
            "evidence_insufficient": True,
            "retrieved_count": 0,
            "citation_count": 0,
            "unresolved_required_conflicts": [],
            "missing_retrieval_requirements": [],
        },
    )
    assert (
        result["capability_outcomes"]["citation_validation"]["status"]
        == "failed"
    )
    assert result["status"] == "partial"
    missing = result["completion_contract"]["missing_requirements"]
    assert "required_citations" in missing
    assert "required_task:use_document" in missing


def test_completion_contract_protocol_failure_is_task_local():
    route = __import__(
        "app.agent_graph.semantic_route", fromlist=["SemanticRouteDecision"]
    ).SemanticRouteDecision.model_validate(hybrid_route_payload())
    result = _apply_completion_contract(
        result={
            "status": "completed",
            "final_answer": "已完成计算；文档证据审核部分异常，该部分未完成。",
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
            "status": "completed_with_partial_evidence",
            "execution_ok": True,
            "evidence_insufficient": False,
            "retrieved_count": 24,
            "citation_count": 14,
            "unresolved_required_conflicts": [],
            "missing_retrieval_requirements": ["use_document"],
            "requirement_coverage": [
                {
                    "requirement_id": "calculate",
                    "status": "direct_support",
                    "citation_count": 0,
                },
                {
                    "requirement_id": "use_document:1",
                    "status": "assessment_protocol_failed",
                    "citation_count": 0,
                },
            ],
        },
    )
    task_by_id = {
        item["id"]: item
        for item in result["completion_contract"]["task_outcomes"]
    }
    assert task_by_id["calculate"]["status"] == "completed"
    assert task_by_id["use_document"]["status"] == "failed_requirement"
    assert task_by_id["use_document"]["missing_retrieval_evidence"] is True
    missing = result["completion_contract"]["missing_requirements"]
    assert "required_task:use_document" in missing
    assert "required_task:calculate" not in missing
    assert result["status"] == "partial"


def test_required_outputs_delivered_keyphrase():
    outputs = [
        "金融机构查询个人信息通常需要什么授权条件",
        "个人查询本人信用报告在文档中记载有什么权利",
    ]
    assert _required_outputs_delivered(
        outputs,
        "信用报告包含基本信息；查询需取得授权；"
        "个人每年两次免费获取的权利。",
    ) == []
    missing = _required_outputs_delivered(
        outputs,
        "信用报告包含基本信息；个人每年两次免费获取。",
    )
    assert "授权" in missing


def test_required_outputs_ignores_low_information_fragments():
    outputs = [
        "根据文档中的房贷“三一定律”",
        "消费者应采取的防范措施",
        "参考文档中的风险提示",
    ]
    assert _required_outputs_delivered(
        outputs,
        "三一定律：月供不超过收入三分之一；"
        "消费者应妥善保管身份证件，避免信息泄露。",
    ) == []


def test_completion_contract_requires_sub_outputs():
    route = __import__(
        "app.agent_graph.semantic_route", fromlist=["SemanticRouteDecision"]
    ).SemanticRouteDecision.model_validate(
        {
            "orchestration_mode": "rag",
            "required_capabilities": [
                "knowledge_retrieval",
            ],
            "task_requirements": [
                {
                    "id": "task_8",
                    "description": "查询授权条件、个人查询权利",
                    "required": True,
                    "capabilities": ["knowledge_retrieval"],
                    "requires_citations": True,
                    "task_kind": "retrieval",
                    "required_outputs": ["授权条件", "查询权利"],
                    "evidence_requirements": ["授权条件", "查询权利"],
                }
            ],
            "retrieval_requirement": "required",
            "citation_requirement": "required",
            "grounding_requirement": "authoritative",
            "retrieval_scope": "uploaded_documents",
            "needs_exact_calculation": False,
            "needs_latest_data": False,
            "needs_clarification": False,
            "risk_level": "low",
            "confidence": 0.9,
            "ambiguities": [],
            "reason_summary": "test",
        }
    )
    result = _apply_completion_contract(
        result={
            "status": "completed",
            "final_answer": "信用报告包含基本信息。",
            "tool_results": [],
        },
        route=route,
        rag_outcome={
            "status": "completed_with_evidence",
            "execution_ok": True,
            "evidence_insufficient": False,
            "retrieved_count": 4,
            "citation_count": 3,
            "unresolved_required_conflicts": [],
            "missing_retrieval_requirements": [],
            "requirement_coverage": [
                {
                    "requirement_id": "task_8:1",
                    "status": "direct_support",
                    "citation_count": 2,
                },
                {
                    "requirement_id": "task_8:2",
                    "status": "direct_support",
                    "citation_count": 1,
                }
            ],
        },
    )
    task = result["completion_contract"]["task_outcomes"][0]
    assert task["status"] == "completed"
    assert "授权" in task["coverage_warnings"]
    assert result["status"] == "completed"


def test_delivery_coverage_does_not_pollute_financial_calculation():
    route = __import__(
        "app.agent_graph.semantic_route", fromlist=["SemanticRouteDecision"]
    ).SemanticRouteDecision.model_validate(
        {
            "orchestration_mode": "hybrid",
            "required_capabilities": [
                "knowledge_retrieval",
                "financial_calculation",
            ],
            "task_requirements": [
                {
                    "id": "task_3",
                    "description": "计算月供并说明授权条件",
                    "required": True,
                    "capabilities": [
                        "knowledge_retrieval",
                        "financial_calculation",
                    ],
                    "requires_citations": True,
                    "task_kind": "retrieval",
                    "required_outputs": ["授权条件"],
                    "evidence_requirements": ["E1"],
                }
            ],
            "retrieval_requirement": "required",
            "citation_requirement": "required",
            "grounding_requirement": "authoritative",
            "retrieval_scope": "uploaded_documents",
            "needs_exact_calculation": True,
            "needs_latest_data": False,
            "needs_clarification": False,
            "risk_level": "low",
            "confidence": 0.9,
            "ambiguities": [],
            "reason_summary": "test",
        }
    )
    result = _apply_completion_contract(
        result={
            "status": "completed",
            "final_answer": "月供约1.2万元。",
            "tool_results": [
                {
                    "tool_call_id": "calc_1",
                    "tool_name": "yearly_expense_to_monthly",
                    "success": True,
                    "output": {"monthly_necessary_expense": 15000},
                }
            ],
        },
        route=route,
        rag_outcome={
            "status": "completed_with_evidence",
            "execution_ok": True,
            "evidence_insufficient": False,
            "retrieved_count": 4,
            "citation_count": 3,
            "unresolved_required_conflicts": [],
            "missing_retrieval_requirements": [],
            "requirement_coverage": [
                {
                    "requirement_id": "task_3:1",
                    "status": "direct_support",
                    "citation_count": 3,
                }
            ],
        },
    )
    assert (
        result["capability_outcomes"]["financial_calculation"]["status"]
        == "satisfied"
    )
    assert (
        result["capability_outcomes"]["financial_calculation"][
            "satisfaction_source"
        ]
        == "tool"
    )
    assert result["status"] == "completed"
    assert any(
        "授权" in item.get("coverage_warnings") or []
        for item in result["completion_contract"]["task_outcomes"]
    )


def _insurance_route_payload():
    return {
        "orchestration_mode": "hybrid",
        "required_capabilities": [
            "knowledge_retrieval",
            "financial_calculation",
        ],
        "task_requirements": [
            {
                "id": "task_3",
                "description": "案例B分析",
                "required": True,
                "capabilities": [
                    "knowledge_retrieval",
                    "financial_calculation",
                ],
                "requires_citations": True,
                "task_kind": "retrieval",
                "evidence_requirements": ["R1", "R2"],
                "required_outputs": [],
            }
        ],
        "retrieval_requirement": "required",
        "citation_requirement": "required",
        "grounding_requirement": "authoritative",
        "retrieval_scope": "uploaded_documents",
        "needs_exact_calculation": True,
        "needs_latest_data": False,
        "needs_clarification": False,
        "risk_level": "low",
        "confidence": 0.9,
        "ambiguities": [],
        "reason_summary": "test",
    }


def test_completion_contract_evidence_requirements_aggregate():
    route = __import__(
        "app.agent_graph.semantic_route", fromlist=["SemanticRouteDecision"]
    ).SemanticRouteDecision.model_validate(_insurance_route_payload())
    result = _apply_completion_contract(
        result={
            "status": "completed",
            "final_answer": "条件性结论：9万。",
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
                "output": 90000,
                "verification_status": "verified",
            }
        ],
        rag_outcome={
            "status": "completed_with_partial_evidence",
            "execution_ok": True,
            "evidence_insufficient": False,
            "retrieved_count": 10,
            "citation_count": 2,
            "unresolved_required_conflicts": [],
            "missing_retrieval_requirements": [],
            "requirement_coverage": [
                {
                    "requirement_id": "task_3:1",
                    "status": "direct_support",
                    "citation_count": 1,
                },
                {
                    "requirement_id": "task_3:2",
                    "status": "irrelevant",
                    "citation_count": 0,
                },
            ],
        },
    )
    task = result["completion_contract"]["task_outcomes"][0]
    # irrelevant/insufficient_evidence means the system executed normally but
    # evidence quality is limited; that is not an execution failure.
    assert task["status"] == "completed"
    assert task["missing_retrieval_evidence"] is False
    assert (
        result["capability_outcomes"]["financial_calculation"]["status"]
        == "satisfied"
    )
    assert (
        result["capability_outcomes"]["financial_calculation"][
            "result_refs"
        ]
        == ["CALC_1"]
    )
    assert result["status"] == "completed"

    full = _apply_completion_contract(
        result={
            "status": "completed",
            "final_answer": "条件性结论：9万。",
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
                "output": 90000,
                "verification_status": "verified",
            }
        ],
        rag_outcome={
            "status": "completed_with_partial_evidence",
            "execution_ok": True,
            "evidence_insufficient": False,
            "retrieved_count": 10,
            "citation_count": 2,
            "unresolved_required_conflicts": [],
            "missing_retrieval_requirements": [],
            "requirement_coverage": [
                {
                    "requirement_id": "task_3:1",
                    "status": "direct_support",
                    "citation_count": 1,
                },
                {
                    "requirement_id": "task_3:2",
                    "status": "partial_support",
                    "citation_count": 1,
                },
            ],
        },
    )
    task_full = full["completion_contract"]["task_outcomes"][0]
    assert task_full["status"] == "completed"


def test_retrieval_queries_expand_evidence_requirements():
    from app.api.routes.chat_graph_v2 import _retrieval_queries

    route = __import__(
        "app.agent_graph.semantic_route", fromlist=["SemanticRouteDecision"]
    ).SemanticRouteDecision.model_validate(_insurance_route_payload())
    queries = _retrieval_queries(route, "保险理赔")
    assert {item["requirement_id"] for item in queries} == {
        "task_3:1",
        "task_3:2",
    }
    assert "R1" in queries[0]["query"]


def test_delivery_fallback_does_not_pollute_execution_status():
    route = __import__(
        "app.agent_graph.semantic_route", fromlist=["SemanticRouteDecision"]
    ).SemanticRouteDecision.model_validate(_insurance_route_payload())
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
                "output": 650000,
                "verification_status": "verified",
            }
        ],
        rag_outcome={
            "status": "completed_with_partial_evidence",
            "execution_ok": True,
            "evidence_insufficient": False,
            "retrieved_count": 10,
            "citation_count": 2,
            "unresolved_required_conflicts": [],
            "missing_retrieval_requirements": [],
            "requirement_coverage": [
                {
                    "requirement_id": "task_3:1",
                    "status": "direct_support",
                    "citation_count": 1,
                },
                {
                    "requirement_id": "task_3:2",
                    "status": "direct_support",
                    "citation_count": 1,
                },
            ],
        },
    )
    # Delivery fallback must not rewrite execution truth.
    assert result["execution_status"] == "success"
    assert result["fulfillment_status"] == "fulfilled"
    assert result["status"] == "fallback"
    assert (
        result["capability_outcomes"]["financial_calculation"]["status"]
        == "satisfied"
    )


def test_normalize_payload_fills_evidence_requirements():
    from app.agent_graph.semantic_route import SemanticRouter

    payload = {
        "orchestration_mode": "rag",
        "required_capabilities": ["knowledge_retrieval"],
        "task_requirements": [
            {
                "id": "T1",
                "description": "等待期、保险责任、责任免除",
                "required": True,
                "capabilities": ["knowledge_retrieval"],
                "requires_citations": True,
                "task_kind": "retrieval",
            }
        ],
        "retrieval_requirement": "required",
        "citation_requirement": "required",
        "grounding_requirement": "authoritative",
        "retrieval_scope": "uploaded_documents",
        "needs_exact_calculation": False,
        "needs_latest_data": False,
        "needs_clarification": False,
        "risk_level": "low",
        "confidence": 0.9,
        "ambiguities": [],
        "reason_summary": "test",
    }
    normalized = SemanticRouter._normalize_payload(payload)
    task = normalized["task_requirements"][0]
    assert task["evidence_requirements"] == [
        "等待期",
        "保险责任",
        "责任免除",
    ]

    aliased = dict(payload)
    aliased["task_requirements"][0]["evidence_queries"] = [
        "hospital_definition",
        "exclusions",
    ]
    normalized_aliased = SemanticRouter._normalize_payload(aliased)
    assert normalized_aliased["task_requirements"][0][
        "evidence_requirements"
    ] == ["hospital_definition", "exclusions"]


def test_normalize_payload_auto_generates_protocol_ids():
    payload = {
        "orchestration_mode": "rag",
        "required_capabilities": ["knowledge_retrieval"],
        "task_requirements": [
            {
                "description": "检索责任免除",
                "required": True,
                "capabilities": ["knowledge_retrieval"],
                "requires_citations": True,
                "task_kind": "retrieval",
            },
            {
                "id": 1,
                "description": "判断等待期",
                "required": True,
                "capabilities": ["knowledge_retrieval"],
                "requires_citations": True,
                "task_kind": "retrieval",
            },
        ],
        "retrieval_requirement": "required",
        "citation_requirement": "required",
        "grounding_requirement": "authoritative",
        "retrieval_scope": "uploaded_documents",
        "needs_exact_calculation": False,
        "needs_latest_data": False,
        "needs_clarification": False,
        "risk_level": "low",
        "confidence": 0.9,
        "ambiguities": [],
        "reason_summary": "test",
    }
    normalized = SemanticRouter._normalize_payload(payload)
    tasks = normalized["task_requirements"]
    assert [task["id"] for task in tasks] == ["T1", "T2"]
    route = SemanticRouteDecision.model_validate(normalized)
    assert [task.id for task in route.task_requirements] == [
        "T1",
        "T2",
    ]


def test_normalize_payload_remaps_dependencies_after_renumber():
    payload = {
        "orchestration_mode": "rag",
        "required_capabilities": ["knowledge_retrieval"],
        "task_requirements": [
            {
                "id": "B",
                "description": "first retrieval",
                "required": True,
                "capabilities": ["knowledge_retrieval"],
                "requires_citations": True,
                "task_kind": "retrieval",
                "depends_on": ["B"],
            },
            {
                "id": "B",
                "description": "second retrieval",
                "required": True,
                "capabilities": ["knowledge_retrieval"],
                "requires_citations": True,
                "task_kind": "retrieval",
                "depends_on": [],
            },
        ],
        "retrieval_requirement": "required",
        "citation_requirement": "required",
        "grounding_requirement": "authoritative",
        "retrieval_scope": "uploaded_documents",
        "needs_exact_calculation": False,
        "needs_latest_data": False,
        "needs_clarification": False,
        "risk_level": "low",
        "confidence": 0.9,
        "ambiguities": [],
        "reason_summary": "test",
    }
    normalized = SemanticRouter._normalize_payload(payload)
    tasks = normalized["task_requirements"]
    assert [task["id"] for task in tasks] == ["T1", "T2"]
    assert tasks[0]["depends_on"] == ["T1"]
    route = SemanticRouteDecision.model_validate(normalized)
    assert route.task_requirements[0].depends_on == ["T1"]


def _typed_route_payload(
    *,
    capabilities: list[str],
    task_capabilities: list[str],
    capability_constraints: dict[str, str],
    resource_constraints: dict | None = None,
    retrieval_requirement: str = "required",
    citation_requirement: str = "required",
) -> dict:
    return {
        "orchestration_mode": "rag",
        "required_capabilities": capabilities,
        "task_requirements": [
            {
                "id": "T1",
                "description": "retrieve and answer",
                "capabilities": task_capabilities,
                "requires_citations": True,
                "task_kind": "retrieval",
            }
        ],
        "retrieval_requirement": retrieval_requirement,
        "citation_requirement": citation_requirement,
        "grounding_requirement": "authoritative",
        "retrieval_scope": "uploaded_documents",
        "capability_constraints": capability_constraints,
        "resource_constraints": resource_constraints or {},
        "confidence": 0.9,
        "reason_summary": "test",
    }


def test_typed_contract_capability_resource_orthogonality() -> None:
    cases = [
        (
            "required_include",
            _typed_route_payload(
                capabilities=["knowledge_retrieval"],
                task_capabilities=[
                    "knowledge_retrieval",
                    "citation_validation",
                ],
                capability_constraints={
                    "knowledge_retrieval": "required",
                    "citation_validation": "required",
                },
                resource_constraints={
                    "include_documents": [
                        {
                            "reference": "A",
                            "reference_type": "title",
                        }
                    ]
                },
            ),
            {"knowledge_retrieval": "required"},
            {"include_count": 1, "exclusive": False},
        ),
        (
            "exclusive",
            _typed_route_payload(
                capabilities=["knowledge_retrieval"],
                task_capabilities=[
                    "knowledge_retrieval",
                    "citation_validation",
                ],
                capability_constraints={
                    "knowledge_retrieval": "required"
                },
                resource_constraints={
                    "include_documents": [
                        {
                            "reference": "金融知识普及读本（第二版）",
                            "reference_type": "title",
                        }
                    ],
                    "exclusive": True,
                },
            ),
            {"knowledge_retrieval": "required"},
            {"include_count": 1, "exclusive": True},
        ),
        (
            "exclude",
            _typed_route_payload(
                capabilities=["knowledge_retrieval"],
                task_capabilities=[
                    "knowledge_retrieval",
                    "citation_validation",
                ],
                capability_constraints={
                    "knowledge_retrieval": "required"
                },
                resource_constraints={
                    "exclude_documents": [
                        {
                            "reference": "A",
                            "reference_type": "title",
                        }
                    ]
                },
            ),
            {"knowledge_retrieval": "required"},
            {"exclude_count": 1, "exclusive": False},
        ),
        (
            "forbidden",
            {
                "orchestration_mode": "direct",
                "required_capabilities": ["complex_reasoning"],
                "task_requirements": [
                    {
                        "id": "answer",
                        "description": "answer",
                        "capabilities": ["complex_reasoning"],
                        "task_kind": "synthesis",
                    }
                ],
                "capability_constraints": {
                    "knowledge_retrieval": "forbidden"
                },
                "confidence": 0.9,
                "reason_summary": "test",
            },
            {"knowledge_retrieval": "forbidden"},
            {"include_count": 0, "exclusive": False},
        ),
        (
            "web_ban_retrieval_ok",
            _typed_route_payload(
                capabilities=["knowledge_retrieval"],
                task_capabilities=[
                    "knowledge_retrieval",
                    "citation_validation",
                ],
                capability_constraints={
                    "knowledge_retrieval": "required",
                    "citation_validation": "required",
                    "web_search": "forbidden",
                },
            ),
            {
                "knowledge_retrieval": "required",
                "web_search": "forbidden",
            },
            {"include_count": 0, "exclusive": False},
        ),
    ]
    for name, payload, expected_capabilities, expected_resource in cases:
        normalized = SemanticRouter._normalize_payload(payload)
        route = SemanticRouteDecision.model_validate(normalized)
        for capability, value in expected_capabilities.items():
            assert route.capability_constraints[capability] == value, name
        resource = route.resource_constraints
        assert (
            len(resource.include_documents)
            == expected_resource.get("include_count", 0)
        ), name
        assert (
            len(resource.exclude_documents)
            == expected_resource.get("exclude_count", 0)
        ), name
        assert resource.exclusive == expected_resource["exclusive"], name


def test_typed_contract_required_and_forbidden_is_protocol_error() -> None:
    payload = _typed_route_payload(
        capabilities=["knowledge_retrieval"],
        task_capabilities=[
            "knowledge_retrieval",
            "citation_validation",
        ],
        capability_constraints={
            "knowledge_retrieval": "forbidden"
        },
    )
    with pytest.raises(ValueError, match="both forbidden and required"):
        SemanticRouter._normalize_payload(payload)


def test_fallback_strips_memory_read_from_preserved_contract() -> None:
    contract = RequestRequirementContract(
        retrieval_requirement="required",
        citation_requirement="required",
        required_capabilities=[
            "knowledge_retrieval",
            "citation_validation",
            "memory_read",
        ],
        task_requirements=[
            TaskRequirement(
                id="t_retrieve",
                description="retrieve",
                capabilities=[
                    "knowledge_retrieval",
                    "citation_validation",
                ],
                requires_citations=True,
                task_kind="retrieval",
            ),
            TaskRequirement(
                id="t_memory",
                description="memory",
                capabilities=["memory_read"],
                task_kind="reasoning",
            ),
        ],
    )
    route = conservative_route_fallback(
        enable_rag=True,
        allowed_tool_groups=["financial_calculation"],
        error_type="test",
        requirement_contract=contract,
    )
    assert "memory_read" not in route.required_capabilities
    assert all(
        "memory_read" not in task.capabilities
        for task in route.task_requirements
    )


def test_normalize_payload_demotes_memory_read() -> None:
    payload = {
        "orchestration_mode": "rag",
        "required_capabilities": [
            "knowledge_retrieval",
            "memory_read",
        ],
        "task_requirements": [
            {
                "id": "T1",
                "description": "检索文档",
                "capabilities": ["knowledge_retrieval"],
                "requires_citations": True,
                "task_kind": "retrieval",
            }
        ],
        "retrieval_requirement": "required",
        "citation_requirement": "required",
        "grounding_requirement": "authoritative",
        "retrieval_scope": "uploaded_documents",
        "confidence": 0.9,
        "reason_summary": "test",
    }
    normalized = SemanticRouter._normalize_payload(payload)
    assert "memory_read" not in normalized["required_capabilities"]
    route = SemanticRouteDecision.model_validate(normalized)
    assert "memory_read" not in route.required_capabilities


def test_caller_route_constraints_all_uploaded_uses_valid_scope():
    route = __import__(
        "app.agent_graph.semantic_route", fromlist=["SemanticRouteDecision"]
    ).SemanticRouteDecision.model_validate(hybrid_route_payload())
    payload = ProductionChatRequest(
        user_message="必须检索我上传的文档。",
        thread_id="thread-1",
        rag_mode="required",
        document_scope={
            "mode": "all_uploaded",
            "document_ids": [],
        },
    )
    updated = _apply_caller_route_constraints(
        payload=payload,
        route=route,
        scope_mode="all_uploaded",
    )
    assert updated.retrieval_scope == "uploaded_documents"


def test_typed_dependency_reference_resolves_without_string_substitution():
    decision = PlannerDecision(
        action="call_tools",
        tool_calls=[
            ToolCallRequest(
                tool_call_id="call_1",
                step_id="monthly",
                tool_name="yearly_expense_to_monthly",
                arguments={"yearly_necessary_expense": 180000},
            ),
            ToolCallRequest(
                tool_call_id="call_2",
                step_id="reserve",
                depends_on=["monthly"],
                tool_name="emergency_fund_range",
                arguments={
                    "monthly_necessary_expense": {
                        "$ref": {
                            "step_id": "monthly",
                            "path": ["monthly_necessary_expense"],
                        }
                    },
                    "min_months": 3,
                    "max_months": 6,
                },
            ),
        ],
    )
    resolved = resolve_typed_references(
        decision.tool_calls[1].arguments,
        {"monthly": {"monthly_necessary_expense": 15000}},
    )
    assert resolved["monthly_necessary_expense"] == 15000


def test_typed_dependency_reference_must_be_declared():
    with pytest.raises(ValueError, match="declared in depends_on"):
        PlannerDecision(
            action="call_tools",
            tool_calls=[
                ToolCallRequest(
                    tool_call_id="call_1",
                    step_id="monthly",
                    tool_name="yearly_expense_to_monthly",
                    arguments={"yearly_necessary_expense": 180000},
                ),
                ToolCallRequest(
                    tool_call_id="call_2",
                    step_id="reserve",
                    tool_name="emergency_fund_range",
                    arguments={
                        "monthly_necessary_expense": {
                            "$ref": {"step_id": "monthly", "path": []}
                        }
                    },
                ),
            ],
        )


@pytest.mark.anyio
async def test_failed_dag_step_returns_structured_dependency_failures():
    executor = ProductionToolExecutor(registry=build_production_tool_registry())
    decision = PlannerDecision(
        action="call_tools",
        tool_calls=[
            ToolCallRequest(
                tool_call_id="bad_monthly",
                step_id="monthly",
                tool_name="yearly_expense_to_monthly",
                arguments={"yearly_necessary_expense": -1},
            ),
            ToolCallRequest(
                tool_call_id="blocked_reserve",
                step_id="reserve",
                depends_on=["monthly"],
                tool_name="emergency_fund_range",
                arguments={
                    "monthly_necessary_expense": {
                        "$ref": {
                            "step_id": "monthly",
                            "path": ["monthly_necessary_expense"],
                        }
                    }
                },
            ),
        ],
    )
    result = await execute_planner_tool_calls(
        decision=decision,
        executor=executor,
        context=ToolExecutionContext(
            request_id="request",
            run_id="run",
            tenant_id="personal",
            user_id="owner",
            role="user",
            remaining_tool_calls=4,
        ),
    )

    assert result.failed_call_count == 2
    by_id = {item.tool_call_id: item for item in result.tool_results}
    assert by_id["bad_monthly"].error.code == "ARGUMENT_SCHEMA_ERROR"
    assert by_id["blocked_reserve"].error.code == "DEPENDENCY_UNAVAILABLE"


def test_required_rag_api_control_is_a_hard_constraint():
    from app.agent_graph.semantic_route import conservative_route_fallback

    route = conservative_route_fallback(
        enable_rag=True,
        allowed_tool_groups=["financial_calculation"],
        error_type="test",
    )
    constrained = _apply_caller_route_constraints(
        payload=ProductionChatRequest(
            user_message="answer from the knowledge base",
            thread_id="thread",
            rag_mode="required",
            document_ids=["doc_1"],
        ),
        route=route,
    )
    assert constrained.retrieval_requirement == "required"
    assert constrained.citation_requirement == "required"
    assert constrained.retrieval_scope == "selected_documents"


def test_requires_citations_without_evidence_universe_not_fulfilled():
    route = __import__(
        "app.agent_graph.semantic_route", fromlist=["SemanticRouteDecision"]
    ).SemanticRouteDecision.model_validate(
        {
            "orchestration_mode": "rag",
            "required_capabilities": ["knowledge_retrieval"],
            "task_requirements": [
                {
                    "id": "task_3",
                    "description": "存款保险限额",
                    "required": True,
                    "capabilities": ["knowledge_retrieval"],
                    "requires_citations": True,
                    "task_kind": "retrieval",
                    "evidence_requirements": [],
                    "required_outputs": ["最高偿付限额"],
                }
            ],
            "retrieval_requirement": "required",
            "citation_requirement": "required",
            "grounding_requirement": "authoritative",
            "retrieval_scope": "uploaded_documents",
            "needs_exact_calculation": False,
            "needs_latest_data": False,
            "needs_clarification": False,
            "risk_level": "low",
            "confidence": 0.9,
            "ambiguities": [],
            "reason_summary": "test",
        }
    )
    result = _apply_completion_contract(
        result={
            "status": "completed",
            "final_answer": "存款保险最高偿付限额为50万元。",
            "tool_results": [],
        },
        route=route,
        rag_outcome={
            "status": "not_attempted",
            "execution_ok": False,
            "retrieved_count": 0,
            "citation_count": 0,
            "unresolved_required_conflicts": [],
            "missing_retrieval_requirements": [],
            "requirement_coverage": [],
        },
    )
    missing = result["completion_contract"]["missing_requirements"]
    assert "evidence_universe_missing:task_3" in missing
    assert result["fulfillment_status"] != "fulfilled"
    task = result["completion_contract"]["task_outcomes"][0]
    assert task["evidence_requirement_universe_missing"] is True
