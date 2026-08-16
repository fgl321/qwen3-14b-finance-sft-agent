from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.agent_graph.semantic_route import SemanticRouteDecision
from app.api.routes.chat_graph_v2 import (
    _apply_completion_contract,
    _build_logical_evidence_requirements,
    _build_physical_queries,
    _finalize_requirement_coverage,
    _merge_rag_requirement_results,
    _retry_missing_retrieval_requirements,
)


def _route(*tasks: dict) -> SemanticRouteDecision:
    return SemanticRouteDecision.model_validate(
        {
            "orchestration_mode": "rag",
            "required_capabilities": ["knowledge_retrieval"],
            "task_requirements": list(tasks),
            "retrieval_requirement": "required",
            "citation_requirement": "required",
            "grounding_requirement": "authoritative",
            "retrieval_scope": "uploaded_documents",
            "confidence": 0.9,
            "reason_summary": "test",
        }
    )


def _task(
    task_id: str,
    description: str,
    evidence: list[str],
) -> dict:
    return {
        "id": task_id,
        "description": description,
        "required": True,
        "capabilities": [
            "knowledge_retrieval",
            "citation_validation",
        ],
        "requires_citations": True,
        "task_kind": "retrieval",
        "evidence_requirements": evidence,
    }


def _rag_result(
    *,
    sufficient: bool = False,
    support_level: str = "irrelevant",
    citations: list[dict] | None = None,
    assessor_status: str = "completed",
) -> dict:
    return {
        "evidence_assessment": {
            "sufficient": sufficient,
            "support_level": support_level,
            "evidence_conflicts": [],
        },
        "stage_status": {
            "evidence_assessment_status": assessor_status,
        },
        "citations": citations or [],
        "retrieved_chunks": [],
        "answer": "基于证据的结论",
    }


def _citation(
    citation_id: int,
    *,
    support_level: str = "direct_support",
) -> dict:
    return {
        "citation_id": citation_id,
        "document_id": "doc",
        "file_name": "file.pdf",
        "page_start": 1,
        "page_end": 1,
        "chunk_id": f"chunk-{citation_id}",
        "score": 1.0,
        "metadata": {
            "support_level": support_level,
            "requirement_ids": [],
        },
    }


def _run_merge(
    physical_queries: list[dict],
    rag_by_query: dict[str, dict],
) -> dict:
    return _merge_rag_requirement_results(
        [
            (query, rag_by_query[query["id"]])
            for query in physical_queries
        ]
    )


def test_7_logical_5_physical_7_observations() -> None:
    route = _route(
        _task(
            "T1",
            "合同解除条件",
            [
                "合同解除条件",
                "违约金计算依据",
                "合同解除条件",
                "提前还款条件",
                "信息披露要求",
                "提前还款条件",
                "适用范围",
            ],
        )
    )
    requirements = _build_logical_evidence_requirements(route)
    physical_queries, raw_count, merged_count = _build_physical_queries(
        requirements,
        "合同解除",
    )
    assert len(requirements) == 7
    assert len(physical_queries) == 5
    assert raw_count == 7
    assert merged_count == 2
    rag = _run_merge(
        physical_queries,
        {
            query["id"]: _rag_result(
                sufficient=True,
                support_level="direct_support",
                citations=[_citation(index)],
            )
            for index, query in enumerate(physical_queries, start=1)
        },
    )
    rag["physical_queries"] = physical_queries
    rag = _finalize_requirement_coverage(
        rag,
        list(requirements),
        raw_query_count=raw_count,
        merged_query_count=merged_count,
    )
    assert len(rag["requirement_coverage"]) == 7
    assert {item["requirement_id"] for item in rag["requirement_coverage"]} == set(
        requirements
    )
    assert rag["evidence_pipeline"]["physical_query_count"] == 5
    assert rag["evidence_pipeline"]["requirement_observation_count"] == 7
    assert rag["evidence_pipeline"]["missing_observation_ids"] == []


def test_shared_query_preserves_both_observations() -> None:
    route = _route(
        _task(
            "T1",
            "信息披露要求",
            ["信息披露要求", "信息披露要求"],
        )
    )
    requirements = _build_logical_evidence_requirements(route)
    physical_queries, _, _ = _build_physical_queries(
        requirements,
        "信息披露",
    )
    assert len(physical_queries) == 1
    assert set(physical_queries[0]["source_requirement_ids"]) == {
        "T1:1",
        "T1:2",
    }
    rag = _run_merge(
        physical_queries,
        {
            physical_queries[0]["id"]: _rag_result(
                sufficient=True,
                support_level="direct_support",
                citations=[_citation(1)],
            )
        },
    )
    rag["physical_queries"] = physical_queries
    rag = _finalize_requirement_coverage(
        rag,
        list(requirements),
        raw_query_count=2,
        merged_query_count=1,
    )
    assert {
        item["requirement_id"]
        for item in rag["requirement_coverage"]
    } == {
        "T1:1",
        "T1:2",
    }


def test_legacy_topic_expansion_preserves_logical_requirement() -> None:
    from app.api.routes.chat_graph_v2 import (
        _build_physical_queries,
    )

    requirements = {
        "T1:1": {
            "id": "T1:1",
            "task_id": "T1",
            "description": "存款保险最高偿付限额",
            "query": "存款保险：最高偿付限额",
            "required": True,
        }
    }
    physical_queries, _, _ = _build_physical_queries(
        requirements,
        "存款保险和双十定律",
    )
    assert len(physical_queries) > 1
    assert all(
        set(query["source_requirement_ids"]) == {"T1:1"}
        for query in physical_queries
    )
    assert not any(
        str(query.get("requirement_id") or "").startswith(
            "document_topic_"
        )
        for query in physical_queries
    )


def test_no_requirements_returns_no_physical_queries() -> None:
    from app.api.routes.chat_graph_v2 import (
        _build_physical_queries,
    )

    physical_queries, _, _ = _build_physical_queries(
        {},
        "随便什么",
    )
    assert physical_queries == []


def test_same_pool_different_statuses_allowed() -> None:
    citation = _citation(5)
    citation["metadata"]["requirement_ids"] = ["T1:1", "T1:2"]
    rag = {
        "citations": [citation],
        "requirement_coverage": [
            {
                "requirement_id": "T1:1",
                "task_id": "T1",
                "status": "direct_support",
                "source_query_ids": ["Q7"],
                "citation_ids": [5],
                "conflict_ids": [],
            },
            {
                "requirement_id": "T1:2",
                "task_id": "T1",
                "status": "irrelevant",
                "source_query_ids": ["Q7"],
                "citation_ids": [5],
                "conflict_ids": [],
            },
        ],
    }
    rag = _finalize_requirement_coverage(rag, ["T1:1", "T1:2"])
    by_id = {
        item["requirement_id"]: item["status"]
        for item in rag["requirement_coverage"]
    }
    assert by_id == {"T1:1": "direct_support", "T1:2": "irrelevant"}


def test_dedup_merges_provenance() -> None:
    requirements = {
        "E1": {
            "id": "E1",
            "task_id": "T1",
            "description": "违约金计算依据",
            "query": "违约金计算依据",
            "required": True,
        },
        "E2": {
            "id": "E2",
            "task_id": "T1",
            "description": "违约金计算依据",
            "query": "违约金计算依据",
            "required": True,
        },
    }
    physical_queries, _, merged = _build_physical_queries(
        requirements,
        "违约金",
    )
    assert len(physical_queries) == 1
    assert set(physical_queries[0]["source_requirement_ids"]) == {
        "E1",
        "E2",
    }
    assert len(physical_queries[0]["merged_from_query_ids"]) == 1
    assert merged == 1


def test_physical_query_technical_failure_fans_out() -> None:
    physical_queries = [
        {
            "id": "Q1",
            "query": "提前还款条件",
            "source_requirement_ids": ["E1", "E2"],
            "merged_from_query_ids": [],
        }
    ]
    rag = _run_merge(
        physical_queries,
        {
            "Q1": _rag_result(
                assessor_status="service_failed",
            )
        },
    )
    rag["physical_queries"] = physical_queries
    rag = _finalize_requirement_coverage(rag, ["E1", "E2"])
    by_id = {
        item["requirement_id"]: item["status"]
        for item in rag["requirement_coverage"]
    }
    assert by_id == {
        "E1": "service_failed",
        "E2": "service_failed",
    }


def test_query_success_without_evidence_is_not_technical() -> None:
    physical_queries = [
        {
            "id": "Q1",
            "query": "合同解除条件",
            "source_requirement_ids": ["E1"],
            "merged_from_query_ids": [],
        }
    ]
    rag = _run_merge(
        physical_queries,
        {
            "Q1": _rag_result(
                sufficient=False,
                support_level="irrelevant",
                assessor_status="completed",
            )
        },
    )
    rag["physical_queries"] = physical_queries
    rag = _finalize_requirement_coverage(rag, ["E1"])
    assert rag["requirement_coverage"][0]["status"] == "irrelevant"


def test_assessor_failure_is_requirement_local() -> None:
    physical_queries = [
        {
            "id": "Q1",
            "query": "适用范围",
            "source_requirement_ids": ["E1"],
            "merged_from_query_ids": [],
        },
        {
            "id": "Q2",
            "query": "信息披露要求",
            "source_requirement_ids": ["E2"],
            "merged_from_query_ids": [],
        },
    ]
    rag = _run_merge(
        physical_queries,
        {
            "Q1": _rag_result(
                sufficient=True,
                support_level="direct_support",
                citations=[_citation(1)],
            ),
            "Q2": _rag_result(
                assessor_status="protocol_failed",
            ),
        },
    )
    rag["physical_queries"] = physical_queries
    rag = _finalize_requirement_coverage(rag, ["E1", "E2"])
    by_id = {
        item["requirement_id"]: item["status"]
        for item in rag["requirement_coverage"]
    }
    assert by_id["E1"] == "direct_support"
    assert by_id["E2"] == "assessment_protocol_failed"


def test_missing_observation_becomes_not_observed() -> None:
    rag = {
        "citations": [],
        "requirement_coverage": [
            {
                "requirement_id": "E1",
                "task_id": "T1",
                "status": "direct_support",
                "source_query_ids": ["Q1"],
                "citation_ids": [1],
                "conflict_ids": [],
            },
            {
                "requirement_id": "E2",
                "task_id": "T1",
                "status": "partial_support",
                "source_query_ids": ["Q1"],
                "citation_ids": [1],
                "conflict_ids": [],
            },
        ],
    }
    rag = _finalize_requirement_coverage(rag, ["E1", "E2", "E3"])
    assert rag["coverage_integrity"]["status"] == "violated"
    assert rag["coverage_integrity"]["missing_observation_ids"] == ["E3"]
    by_id = {
        item["requirement_id"]: item["status"]
        for item in rag["requirement_coverage"]
    }
    assert by_id["E3"] == "not_observed"
    assert rag["evidence_pipeline"]["not_observed_count"] == 1


def test_citation_dedup_keeps_both_coverages() -> None:
    physical_queries = [
        {
            "id": "Q1",
            "query": "违约金计算依据",
            "source_requirement_ids": ["E1", "E2"],
            "merged_from_query_ids": [],
        }
    ]
    rag = _run_merge(
        physical_queries,
        {
            "Q1": _rag_result(
                sufficient=True,
                support_level="direct_support",
                citations=[_citation(5)],
            )
        },
    )
    rag["physical_queries"] = physical_queries
    rag = _finalize_requirement_coverage(rag, ["E1", "E2"])
    by_id = {
        item["requirement_id"]: item
        for item in rag["requirement_coverage"]
    }
    assert 5 in by_id["E1"]["citation_ids"]
    assert 5 in by_id["E2"]["citation_ids"]


@pytest.mark.asyncio
async def test_retry_keeps_requirement_id_stable_and_merges() -> None:
    class FakeRagService:
        async def answer(self, **kwargs):
            return _rag_result(
                sufficient=True,
                support_level="direct_support",
                citations=[_citation(33)],
            )

    rag = {
        "citations": [_citation(11)],
        "retrieved_chunks": [],
        "usage": {"decomposed_queries": 1},
        "requirement_coverage": [
            {
                "requirement_id": "T1:5",
                "task_id": "T1",
                "query": "提前还款条件",
                "query_id": "Q7",
                "source_query_ids": ["Q7"],
                "status": "insufficient_evidence",
                "citation_ids": [],
                "conflict_ids": [],
                "citation_count": 0,
                "retrieved_count": 0,
                "retryable": True,
                "assessor_status": "completed",
            }
        ],
    }
    payload = SimpleNamespace(
        user_message="提前还款条件",
        user_id="owner",
        tenant_id="personal",
        knowledge_base_id="kb_finance_basic",
        document_ids=[],
    )
    rag = await _retry_missing_retrieval_requirements(
        rag_service=FakeRagService(),
        payload=payload,
        rag=rag,
        document_ids=[],
    )
    entries = [
        item
        for item in rag["requirement_coverage"]
        if item["requirement_id"] == "T1:5"
    ]
    assert len(entries) == 1
    assert entries[0]["status"] == "direct_support"
    assert len(entries[0]["source_query_ids"]) == 2
    assert 33 in entries[0]["citation_ids"]


def test_completion_contract_ignores_physical_query_count() -> None:
    route = _route(
        _task(
            "T1",
            "合同解除条件",
            ["合同解除条件", "违约金计算依据", "提前还款条件"],
        )
    )
    coverage = [
        {
            "requirement_id": "T1:1",
            "status": "direct_support",
            "citation_count": 1,
        },
        {
            "requirement_id": "T1:2",
            "status": "direct_support",
            "citation_count": 1,
        },
        {
            "requirement_id": "T1:3",
            "status": "partial_support",
            "citation_count": 1,
        },
    ]
    result = _apply_completion_contract(
        result={
            "status": "completed",
            "final_answer": "根据文档给出条件性结论。",
            "tool_results": [],
        },
        route=route,
        rag_outcome={
            "status": "completed_with_partial_evidence",
            "execution_ok": True,
            "evidence_insufficient": False,
            "retrieved_count": 4,
            "citation_count": 2,
            "unresolved_required_conflicts": [],
            "missing_retrieval_requirements": [],
            "requirement_coverage": coverage,
        },
    )
    task = result["completion_contract"]["task_outcomes"][0]
    assert task["status"] == "completed"
    assert result["status"] == "completed"


def test_required_observation_count_equals_logical_count() -> None:
    rag = {
        "citations": [_citation(1)],
        "physical_queries": [
            {
                "id": "Q1",
                "query": "适用范围",
                "source_requirement_ids": ["E1", "E2", "E3"],
                "merged_from_query_ids": [],
            }
        ],
        "requirement_coverage": [
            {
                "requirement_id": rid,
                "task_id": "T1",
                "status": "direct_support",
                "source_query_ids": ["Q1"],
                "citation_ids": [1],
                "conflict_ids": [],
            }
            for rid in ("E1", "E2", "E3")
        ],
    }
    rag = _finalize_requirement_coverage(rag, ["E1", "E2", "E3"])
    pipeline = rag["evidence_pipeline"]
    assert (
        pipeline["required_logical_requirement_count"]
        == pipeline["required_observation_count"]
        == 3
    )
    assert pipeline["physical_query_count"] == 1
    assert pipeline["missing_observation_ids"] == []


def test_optional_requirement_missing_does_not_fail_required_task() -> None:
    route = _route(
        _task(
            "T1",
            "合同解除条件",
            ["合同解除条件"],
        )
    )
    result = _apply_completion_contract(
        result={
            "status": "completed",
            "final_answer": "合同解除条件已根据文档说明。",
            "tool_results": [],
        },
        route=route,
        rag_outcome={
            "status": "completed_with_evidence",
            "execution_ok": True,
            "evidence_insufficient": False,
            "retrieved_count": 2,
            "citation_count": 1,
            "unresolved_required_conflicts": [],
            "missing_retrieval_requirements": [],
            "requirement_coverage": [
                {
                    "requirement_id": "T1:1",
                    "status": "direct_support",
                    "citation_count": 1,
                }
            ],
        },
    )
    assert result["completion_contract"]["task_outcomes"][0]["status"] == (
        "completed"
    )


def test_direct_support_requires_direct_citation() -> None:
    rag = {
        "citations": [
            _citation(
                9,
                support_level="partial_support",
            )
        ],
        "requirement_coverage": [
            {
                "requirement_id": "T1:1",
                "task_id": "T1",
                "status": "direct_support",
                "source_query_ids": ["Q1"],
                "citation_ids": [9],
                "conflict_ids": [],
            }
        ],
    }
    rag = _finalize_requirement_coverage(rag, ["T1:1"])
    entry = rag["requirement_coverage"][0]
    assert entry["status"] == "partial_support"
    assert entry["reason"] == "direct_support_requires_direct_citation"
    assert rag["coverage_integrity"]["downgraded_direct_support_ids"] == [
        "T1:1"
    ]


def test_insurance_regression_all_requirements_observed() -> None:
    route = _route(
        _task(
            "T1",
            "保险条款分析",
            [
                "一般医疗费用保险责任",
                "等待期作用",
                "医院要求",
                "免赔额作用",
                "给付比例",
                "医疗费用补偿原则",
                "责任免除",
            ],
        ),
        _task(
            "T4",
            "案例B证据链",
            [
                "一般医疗费用保险责任",
                "医院条件",
                "必要且合理",
                "责任免除",
                "既往症",
                "社会医疗保险补偿",
            ],
        ),
    )
    requirements = _build_logical_evidence_requirements(route)
    physical_queries, raw_count, merged_count = _build_physical_queries(
        requirements,
        "保险理赔",
    )
    rag = _run_merge(
        physical_queries,
        {
            query["id"]: _rag_result(
                sufficient=True,
                support_level="direct_support",
                citations=[_citation(index)],
            )
            for index, query in enumerate(physical_queries, start=1)
        },
    )
    rag["physical_queries"] = physical_queries
    rag = _finalize_requirement_coverage(
        rag,
        list(requirements),
        raw_query_count=raw_count,
        merged_query_count=merged_count,
    )
    observed = {
        item["requirement_id"]
        for item in rag["requirement_coverage"]
    }
    assert {
        "T1:1",
        "T1:2",
        "T1:3",
        "T1:4",
        "T1:5",
        "T1:6",
        "T1:7",
    }.issubset(observed)
    assert {
        "T4:1",
        "T4:2",
        "T4:3",
        "T4:4",
        "T4:5",
        "T4:6",
    }.issubset(observed)
    assert rag["coverage_integrity"]["status"] == "ok"
    assert rag["evidence_pipeline"]["missing_observation_ids"] == []
