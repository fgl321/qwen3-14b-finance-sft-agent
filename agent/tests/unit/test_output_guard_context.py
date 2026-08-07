from __future__ import annotations

import json

from app.agent_graph.llm_output_guard import (
    LLMOutputGuard,
    OutputGuardRequest,
)
from app.agent_graph.schemas.loop_schema import AgentLoopResult
from app.agent_graph.schemas.planner_schema import PlannerDecision
from app.agent_graph.schemas.synthesis_schema import SynthesisResult


class _FakeGuardLLM:
    async def chat(self, **kwargs):
        return {
            "model": "fake",
            "message": {"content": "{}"},
            "usage": {},
        }


def _loop_result() -> AgentLoopResult:
    return AgentLoopResult(
        status="completed",
        final_decision=PlannerDecision(
            action="respond",
            decision_reason="直接回答",
            confidence="high",
        ),
        finish_reason="planner_finished",
        tool_results=[],
        tool_traces=[],
    )


def test_guard_payload_includes_user_context() -> None:
    guard = LLMOutputGuard(llm_client=_FakeGuardLLM())
    request = OutputGuardRequest(
        request_id="r1",
        run_id="run1",
        user_message="我刚才说的月收入是多少？",
        loop_result=_loop_result(),
        synthesis=SynthesisResult(
            answer="您的月收入是2万元。",
            used_tool_call_ids=[],
            used_citation_ids=[],
        ),
        citations=[],
        context_summary="用户长期记忆：family_finance.monthly_income = 2万",
    )

    messages = guard.build_messages(request)
    user_content = messages[-1]["content"]
    payload = json.loads(
        user_content[user_content.find("{") :]
    )

    user_context = payload["user_context"]
    assert "2万" in user_context["context_summary"]
    assert "合法回答依据" in user_context["note"]
