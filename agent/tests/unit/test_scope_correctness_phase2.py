from __future__ import annotations

import pytest

from app.agent_graph.final_response_pipeline import (
    FinalResponsePipeline,
    FinalResponseRequest,
    _citation_scope_violations,
)
from app.agent_graph.llm_output_guard import (
    OutputGuardInvocationResult,
)
from app.agent_graph.llm_synthesizer import (
    SynthesisInvocationResult,
)
from app.agent_graph.runtime.agent_errors import (
    AgentExecutionError,
)
from app.agent_graph.schemas.loop_schema import (
    AgentLoopResult,
)
from app.agent_graph.schemas.planner_schema import (
    PlannerDecision,
)
from app.agent_graph.schemas.synthesis_schema import (
    OutputGuardResult,
    SynthesisResult,
)
from app.agent_graph.schemas.tool_schema import ToolResult
from app.api.routes.chat_graph_v2 import (
    ProductionChatRequest,
    _assert_retrieval_within_scope,
    _citation_scope_violations as route_citation_violations,
    _rag_answer_cache_key,
    _rag_outcome,
    _rag_pipeline_metrics,
    _rag_request_fingerprint,
    _scope_snapshot_hash,
)


def _payload() -> ProductionChatRequest:
    return ProductionChatRequest(
        user_message="必须检索我上传的文档。",
        thread_id="thread-1",
        user_id="owner",
        tenant_id="personal",
        knowledge_base_id="kb_finance_basic",
        document_ids=["doc_a"],
    )


def test_scope_snapshot_hash_is_stable_and_sensitive() -> None:
    snapshot = {
        "doc_a": {
            "document_id": "doc_a",
            "document_version": 1,
            "content_hash": "h1",
        }
    }
    assert (
        _scope_snapshot_hash(snapshot)
        == _scope_snapshot_hash(dict(snapshot))
    )
    changed = {
        "doc_a": {
            "document_id": "doc_a",
            "document_version": 2,
            "content_hash": "h1",
        }
    }
    assert _scope_snapshot_hash(snapshot) != _scope_snapshot_hash(changed)
    assert _scope_snapshot_hash(None) == ""


def test_request_fingerprint_and_cache_key_include_scope_snapshot() -> None:
    payload = _payload()
    base_hash = _scope_snapshot_hash(
        {
            "doc_a": {
                "document_id": "doc_a",
                "document_version": 1,
                "content_hash": "h1",
            }
        }
    )
    other_hash = _scope_snapshot_hash(
        {
            "doc_a": {
                "document_id": "doc_a",
                "document_version": 2,
                "content_hash": "h2",
            }
        }
    )
    assert (
        _rag_request_fingerprint(payload, scope_snapshot_hash=base_hash)
        != _rag_request_fingerprint(payload, scope_snapshot_hash=other_hash)
    )
    assert (
        _rag_answer_cache_key(
            payload=payload,
            retrieval_query="q",
            provider="qwen",
            kb_fingerprint=1,
            scope_snapshot_hash=base_hash,
        )
        != _rag_answer_cache_key(
            payload=payload,
            retrieval_query="q",
            provider="qwen",
            kb_fingerprint=1,
            scope_snapshot_hash=other_hash,
        )
    )


def test_assert_retrieval_within_scope() -> None:
    _assert_retrieval_within_scope(
        {
            "retrieved_chunks": [
                {"document_id": "doc_a", "chunk_id": "c1"}
            ]
        },
        ["doc_a"],
    )
    with pytest.raises(AgentExecutionError):
        _assert_retrieval_within_scope(
            {
                "retrieved_chunks": [
                    {"document_id": "doc_b", "chunk_id": "c2"}
                ]
            },
            ["doc_a"],
        )


def test_citation_scope_violations_route_level() -> None:
    rag = {
        "citations": [
            {"citation_id": 1, "document_id": "doc_a"},
            {"citation_id": 2, "document_id": "doc_b"},
        ]
    }
    snapshot = {
        "doc_a": {"document_id": "doc_a", "document_version": "1"}
    }
    assert route_citation_violations(rag, snapshot) == ["doc_b"]
    rag["citations"][0]["document_version"] = "9"
    assert route_citation_violations(rag, snapshot) == [
        "doc_a@9",
        "doc_b",
    ]


def test_pipeline_citation_scope_violations() -> None:
    citations = [
        {"citation_id": 1, "document_id": "doc_b"},
        {"citation_id": 2, "document_id": "doc_a"},
    ]
    synthesis = SynthesisResult(
        answer="x",
        used_tool_call_ids=[],
        used_citation_ids=["1", "2"],
        uncertainty=None,
        disclaimer_required=False,
    )
    violations = _citation_scope_violations(
        citations,
        synthesis,
        allowed_document_ids=["doc_a"],
        scope_snapshot={
            "doc_a": {"document_id": "doc_a", "document_version": "1"}
        },
    )
    assert violations == ["citation_scope:doc_b"]


def test_rag_outcome_execution_flags() -> None:
    outcome = _rag_outcome(
        {
            "attempted": True,
            "degraded": False,
            "sufficient": False,
            "retrieved_count": 0,
        },
        None,
    )
    assert outcome["status"] == "completed_no_evidence"
    assert outcome["execution_ok"] is True
    assert outcome["evidence_insufficient"] is True
    degraded = _rag_outcome(
        {
            "attempted": True,
            "degraded": True,
            "error": "RAG_SERVICE_UNAVAILABLE",
        },
        None,
    )
    assert degraded["execution_ok"] is False


def test_rag_outcome_partial_support_is_not_missing_requirement() -> None:
    outcome = _rag_outcome(
        {
            "attempted": True,
            "degraded": False,
            "sufficient": False,
            "retrieved_count": 13,
            "requirement_coverage": [
                {
                    "requirement_id": "task_4",
                    "status": "partial_support",
                    "citation_count": 3,
                },
                {
                    "requirement_id": "task_5",
                    "status": "direct_support",
                    "citation_count": 3,
                },
                {
                    "requirement_id": "task_irrelevant",
                    "status": "irrelevant",
                    "citation_count": 0,
                },
            ],
        },
        {"citations": []},
    )
    assert outcome["missing_retrieval_requirements"] == []


def test_rag_pipeline_metrics_clears_rejection_reason_when_partial_satisfied() -> None:
    metrics = _rag_pipeline_metrics(
        {
            "retrieved_chunks": [],
            "evidence_assessment": {
                "sufficient": False,
                "reason": (
                    "one or more retrieval requirements "
                    "lack acceptable support"
                ),
            },
            "citations": [],
            "requirement_coverage": [
                {
                    "requirement_id": "task_4",
                    "status": "partial_support",
                    "citation_count": 3,
                }
            ],
        }
    )
    assert metrics["evidence_rejection_reason"] is None
    assert metrics["requirement_coverage"] != []


def test_rag_outcome_partial_protocol_failure_is_task_local() -> None:
    outcome = _rag_outcome(
        {
            "attempted": True,
            "degraded": False,
            "sufficient": False,
            "evidence_assessment_status": "partial_protocol_failure",
            "retrieved_count": 24,
            "requirement_coverage": [
                {
                    "requirement_id": "task_3",
                    "status": "direct_support",
                    "citation_count": 2,
                },
                {
                    "requirement_id": "task_9",
                    "status": "assessment_protocol_failed",
                    "citation_count": 0,
                },
            ],
        },
        {"citations": []},
    )
    assert outcome["status"] == "completed_with_partial_evidence"
    assert outcome["missing_retrieval_requirements"] == ["task_9"]
    assert outcome["protocol_failed_task_ids"] == ["task_9"]
    assert outcome["execution_ok"] is True


def _loop_result() -> AgentLoopResult:
    return AgentLoopResult(
        status="completed",
        final_decision=PlannerDecision(
            action="respond",
            clarification_question=None,
            decision_reason="test",
            confidence="high",
            plan_version=1,
        ),
        tool_results=[],
        agent_rounds=1,
        total_tool_calls=0,
        finish_reason="test",
    )


def _synthesis(answer: str, used_citation_ids: list[str]) -> SynthesisInvocationResult:
    return SynthesisInvocationResult(
        result=SynthesisResult(
            answer=answer,
            used_tool_call_ids=[],
            used_citation_ids=used_citation_ids,
            uncertainty=None,
            disclaimer_required=False,
        )
    )


def _guard_pass() -> OutputGuardInvocationResult:
    return OutputGuardInvocationResult(
        result=OutputGuardResult(
            verdict="pass",
            reason="ok",
            risk_flags=[],
            rewrite_instructions=None,
        )
    )


class _FakeSynthesizer:
    def __init__(self, results: list[SynthesisInvocationResult]) -> None:
        self.results = list(results)
        self.calls = 0

    async def synthesize(self, request: object) -> SynthesisInvocationResult:
        self.calls += 1
        return self.results.pop(0)


class _FakeGuard:
    def __init__(self, results: list[OutputGuardInvocationResult]) -> None:
        self.results = list(results)
        self.calls = 0

    async def guard(self, request: object) -> OutputGuardInvocationResult:
        self.calls += 1
        return self.results.pop(0)


def _scope_request() -> FinalResponseRequest:
    return FinalResponseRequest(
        request_id="req-1",
        run_id="run-1",
        user_message="必须基于 A.pdf 回答。",
        loop_result=_loop_result(),
        citations=[
            {"citation_id": "1", "document_id": "doc_b"},
        ],
        allowed_document_ids=["doc_a"],
        scope_snapshot={
            "doc_a": {
                "document_id": "doc_a",
                "document_version": "1",
            }
        },
    )


@pytest.mark.anyio
async def test_pipeline_regenerates_once_then_passes() -> None:
    synthesizer = _FakeSynthesizer(
        [
            _synthesis("使用 doc_b 的引用。", ["1"]),
            _synthesis("基于 doc_a 回答。", []),
        ]
    )
    guard = _FakeGuard([_guard_pass()])
    pipeline = FinalResponsePipeline(
        synthesizer=synthesizer,
        output_guard=guard,
    )
    result = await pipeline.run(_scope_request())
    assert result.status == "completed"
    assert result.citation_regeneration_count == 1
    assert synthesizer.calls == 2
    assert guard.calls == 1


@pytest.mark.anyio
async def test_pipeline_fails_closed_after_one_regeneration() -> None:
    synthesizer = _FakeSynthesizer(
        [
            _synthesis("使用 doc_b 的引用。", ["1"]),
            _synthesis("仍然使用 doc_b 的引用。", ["1"]),
        ]
    )
    guard = _FakeGuard([])
    pipeline = FinalResponsePipeline(
        synthesizer=synthesizer,
        output_guard=guard,
    )
    result = await pipeline.run(_scope_request())
    assert result.finish_reason == "citation_scope_violation"
    assert result.citation_regeneration_count == 1
    assert guard.calls == 0
