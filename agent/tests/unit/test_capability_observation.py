from __future__ import annotations

import pytest

from app.api.routes.chat_graph_v2 import (
    _apply_completion_contract,
    _persist_conversation_state,
)
from app.agent_graph.semantic_route import (
    SemanticRouteDecision,
)


def _memory_route(constraint: str) -> SemanticRouteDecision:
    return SemanticRouteDecision(
        orchestration_mode="direct",
        required_capabilities=["memory_read"],
        task_requirements=[
            {
                "id": "T1",
                "description": "读取记忆",
                "capabilities": ["memory_read"],
                "task_kind": "reasoning",
            }
        ],
        memory_constraint=constraint,  # type: ignore[arg-type]
        confidence=0.9,
        reason_summary="test",
    )


def _run_memory(
    *,
    constraint: str,
    memory_audit: dict,
) -> dict:
    result = _apply_completion_contract(
        result={
            "status": "completed",
            "final_answer": "回答",
            "tool_results": [],
        },
        route=_memory_route(constraint),
        rag_outcome={
            "status": "not_attempted",
            "requirement_coverage": [],
        },
        memory_audit=memory_audit,
    )
    return result["capability_outcomes"]["memory_read"]


def test_memory_satisfied_when_records_found() -> None:
    outcome = _run_memory(
        constraint="required",
        memory_audit={
            "long_memory_attempted": True,
            "long_memory_loaded": 2,
        },
    )
    assert outcome["status"] == "satisfied"
    assert outcome["attempted"] is True
    assert outcome["execution_ok"] is True
    assert outcome["satisfaction_source"] == "memory"


def test_memory_empty_only_after_successful_execution() -> None:
    outcome = _run_memory(
        constraint="required",
        memory_audit={
            "long_memory_attempted": True,
            "long_memory_loaded": 0,
        },
    )
    assert outcome["status"] == "empty"
    assert outcome["attempted"] is True
    assert outcome["execution_ok"] is True
    assert outcome["satisfaction_source"] == "memory"


def test_memory_technical_unavailable_not_empty() -> None:
    outcome = _run_memory(
        constraint="required",
        memory_audit={
            "long_memory_attempted": True,
            "long_memory_loaded": 0,
            "degraded": [
                {
                    "stage": "long_memory_read",
                    "error": "ConnectionError",
                }
            ],
        },
    )
    assert outcome["status"] == "technical_unavailable"
    assert outcome["attempted"] is True
    assert outcome["execution_ok"] is False
    assert outcome["retryable"] is True


def test_memory_not_needed_is_not_failed() -> None:
    outcome = _run_memory(
        constraint="not_needed",
        memory_audit={"long_memory_attempted": False},
    )
    assert outcome["status"] == "not_needed"
    assert outcome["attempted"] is False
    assert outcome["satisfaction_source"] is None


def test_memory_required_but_not_executed_is_not_observed() -> None:
    outcome = _run_memory(
        constraint="required",
        memory_audit={"long_memory_attempted": False},
    )
    assert outcome["status"] == "not_observed"
    assert outcome["attempted"] is False


class _FakePayload:
    user_id = "u1"
    thread_id = "t1"
    tenant_id = "personal"


class _FailingService:
    def set_conversation_state(self, **kwargs):
        raise RuntimeError("redis down")


@pytest.mark.asyncio
async def test_atomic_commit_failure_propagates() -> None:
    from app.agent_graph.conversation_state import (
        default_conversation_state,
    )

    with pytest.raises(RuntimeError):
        await _persist_conversation_state(
            service=_FailingService(),  # type: ignore[arg-type]
            payload=_FakePayload(),  # type: ignore[arg-type]
            state=default_conversation_state(),
        )
