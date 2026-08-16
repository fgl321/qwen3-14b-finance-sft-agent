from __future__ import annotations

import json

from app.agent_graph.llm_output_guard import (
    OutputGuardRequest,
    deterministic_output_flags,
)
from app.agent_graph.llm_synthesizer import (
    LLMAnswerSynthesizer,
    SynthesisRequest,
)
from app.agent_graph.schemas.loop_schema import (
    AgentLoopResult,
)
from app.agent_graph.schemas.planner_schema import (
    PlannerDecision,
)
from app.agent_graph.schemas.synthesis_schema import (
    SynthesisResult,
)
from app.rag.rag_types import SourceAuthorityContract


def _loop_result() -> AgentLoopResult:
    return AgentLoopResult(
        status="completed",
        final_decision=PlannerDecision(
            action="respond",
            decision_reason="done",
            confidence="high",
            plan_version=3,
        ),
        tool_results=[],
        agent_rounds=1,
        total_tool_calls=0,
        finish_reason="planner_finished",
    )


def _synthesis(
    *,
    answer: str,
    used_tool_call_ids: list[str] | None = None,
    used_citation_ids: list[str] | None = None,
    used_fact_refs: list[str] | None = None,
    used_derivation_ids: list[str] | None = None,
    used_result_artifact_refs: list[str] | None = None,
) -> SynthesisResult:
    return SynthesisResult(
        answer=answer,
        used_tool_call_ids=used_tool_call_ids or [],
        used_citation_ids=used_citation_ids or [],
        used_fact_refs=used_fact_refs or [],
        used_derivation_ids=used_derivation_ids or [],
        used_result_artifact_refs=(
            used_result_artifact_refs or []
        ),
        uncertainty=None,
        disclaimer_required=False,
    )


def test_synthesis_messages_include_source_authority_contract() -> None:
    class FakeClient:
        async def chat(self, **kwargs):  # pragma: no cover
            raise AssertionError("chat should not be called")

    synthesizer = LLMAnswerSynthesizer(
        llm_client=FakeClient(),  # type: ignore[arg-type]
    )
    authority = SourceAuthorityContract(
        general_model_knowledge="forbidden",
        domain_heuristics="forbidden",
    )
    messages = synthesizer.build_messages(
        SynthesisRequest(
            request_id="request_test",
            run_id="run_test",
            user_message="请只根据文档分析",
            loop_result=_loop_result(),
            source_authority=authority,
        )
    )

    system_contents = [
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "system"
    ]
    assert any(
        "<source_authority_contract>" in content
        and "general_model_knowledge" in content
        and "domain_heuristics=forbidden" in content
        for content in system_contents
    )

    user_payload = json.loads(
        str(messages[-1].get("content") or "").split("\n", 1)[1]
    )
    assert user_payload["source_authority"]["general_model_knowledge"] == (
        "forbidden"
    )


def test_guard_blocks_citations_when_documents_forbidden() -> None:
    authority = SourceAuthorityContract(
        selected_documents="forbidden",
    )
    synthesis = _synthesis(
        answer="根据文档，该规则如此。[1]",
        used_citation_ids=["1"],
    )
    flags = deterministic_output_flags(
        synthesis,
        loop_result=_loop_result(),
        citations=[{"citation_id": 1}],
        source_authority=authority,
    )
    assert "source_authority_citation_forbidden" in flags


def test_guard_blocks_general_knowledge_when_forbidden() -> None:
    authority = SourceAuthorityContract(
        general_model_knowledge="forbidden",
    )
    synthesis = _synthesis(
        answer="根据通用金融知识，高收益必然伴随高风险。"
    )
    flags = deterministic_output_flags(
        synthesis,
        loop_result=_loop_result(),
        citations=[],
        source_authority=authority,
    )
    assert "source_authority_general_knowledge_used" in flags


def test_guard_requires_citing_referenced_result() -> None:
    authority = SourceAuthorityContract(
        general_model_knowledge="forbidden",
    )
    synthesis = _synthesis(
        answer="平台仍然不保证本金安全。"
    )
    flags = deterministic_output_flags(
        synthesis,
        loop_result=_loop_result(),
        citations=[],
        source_authority=authority,
        result_reference_context={
            "resolved_handles": ["RESULT_1"],
            "has_claims": True,
            "has_citations": True,
        },
    )
    assert (
        "source_authority_result_reference_not_cited"
        in flags
    )


def test_guard_blocks_long_ungrounded_explanation() -> None:
    authority = SourceAuthorityContract(
        general_model_knowledge="forbidden",
        domain_heuristics="forbidden",
    )
    synthesis = _synthesis(
        answer=(
            "平台本金是否安全取决于资金存管、合规备案、底层资产真实性等"
            "机制性因素，投入金额只影响风险敞口大小，不影响风险性质本身；"
            "高收益承诺缺乏保障机制时，无论投入多少都面临违约或跑路风险；"
            "因此投入金额从三十万改为十万，不会把不安全变成安全；"
            "判断平台安全性应关注资金存管、备案资质、底层资产真实性等"
            "机制性因素，而不是纠结投入金额，任何投资都应使用闲置资金，"
            "避免借款或加杠杆投入，具体平台安全性需结合实际资质评估；"
            "这一判断不因投入金额变化而改变，投入金额只影响单笔损失上限，"
            "不影响平台本身是否存在资金保障机制、是否具备合规备案、"
            "底层资产是否真实、是否存在期限错配或资金池运作等问题；"
            "因此无论投入十万元还是三十万元，都需要先核实平台资质、"
            "资金存管与底层资产真实性，再决定是否投入以及投入多少。"
        )
    )
    flags = deterministic_output_flags(
        synthesis,
        loop_result=_loop_result(),
        citations=[],
        source_authority=authority,
    )
    assert (
        "source_authority_ungrounded_explanation"
        in flags
    )


def test_guard_allows_fact_and_derivation_grounding() -> None:
    authority = SourceAuthorityContract(
        general_model_knowledge="forbidden",
        domain_heuristics="forbidden",
    )
    synthesis = _synthesis(
        answer=(
            "70万元来自您的数据：90万元现金减20万元首付款等于70万元。"
            "这是一个确定性推导，不依赖文档或通用知识。"
        ),
        used_fact_refs=["cash", "down_payment"],
        used_derivation_ids=["CALC_1"],
    )
    flags = deterministic_output_flags(
        synthesis,
        loop_result=_loop_result(),
        citations=[],
        source_authority=authority,
        canonical_fact_fields=["cash", "down_payment"],
        known_derivation_ids=["CALC_1"],
    )
    assert "source_authority_ungrounded_explanation" not in flags
    assert "invalid_used_fact_refs" not in flags
    assert "invalid_used_derivation_ids" not in flags


def test_guard_rejects_unknown_fact_refs() -> None:
    authority = SourceAuthorityContract(
        general_model_knowledge="forbidden",
    )
    synthesis = _synthesis(
        answer="根据 cash 字段得出结果。",
        used_fact_refs=["cash", "nonexistent_field"],
    )
    flags = deterministic_output_flags(
        synthesis,
        loop_result=_loop_result(),
        citations=[],
        source_authority=authority,
        canonical_fact_fields=["cash"],
    )
    assert "invalid_used_fact_refs" in flags


def test_guard_allows_result_artifact_grounding() -> None:
    authority = SourceAuthorityContract(
        general_model_knowledge="forbidden",
    )
    synthesis = _synthesis(
        answer=(
            "存款保险最高偿付限额为50万元，这是此前文档结论的重复。"
        ),
        used_result_artifact_refs=["RESULT_1.CLAIM_2"],
    )
    flags = deterministic_output_flags(
        synthesis,
        loop_result=_loop_result(),
        citations=[],
        source_authority=authority,
        known_sub_artifact_ids=["RESULT_1.CLAIM_2"],
    )
    assert "invalid_used_result_artifact_refs" not in flags
    assert "source_authority_ungrounded_explanation" not in flags


def test_guard_rejects_unknown_result_artifact_ref() -> None:
    authority = SourceAuthorityContract(
        general_model_knowledge="forbidden",
    )
    synthesis = _synthesis(
        answer="根据 RESULT_1.CLAIM_99 说明。",
        used_result_artifact_refs=["RESULT_1.CLAIM_99"],
    )
    flags = deterministic_output_flags(
        synthesis,
        loop_result=_loop_result(),
        citations=[],
        source_authority=authority,
        known_sub_artifact_ids=["RESULT_1.CLAIM_2"],
    )
    assert "invalid_used_result_artifact_refs" in flags


def test_guard_blocks_ungrounded_answer_when_grounding_required() -> None:
    authority = SourceAuthorityContract(
        general_model_knowledge="forbidden",
        domain_heuristics="forbidden",
    )
    synthesis = _synthesis(
        answer="按照一般经验，应该保留 6 个月备用金。"
    )
    flags = deterministic_output_flags(
        synthesis,
        loop_result=_loop_result(),
        citations=[],
        source_authority=authority,
        delivery_contract=(
            "<delivery_contract> T1: status=required; "
            "required_outputs=['waiting_period_rule'] "
            "</delivery_contract>"
        ),
    )
    assert "source_authority_ungrounded_answer" in flags


def test_guard_allows_disclosed_ungrounded_answer() -> None:
    authority = SourceAuthorityContract(
        general_model_knowledge="forbidden",
        domain_heuristics="forbidden",
    )
    synthesis = _synthesis(
        answer="当前没有检索到足够文档证据，无法确认该结论。"
    )
    flags = deterministic_output_flags(
        synthesis,
        loop_result=_loop_result(),
        citations=[],
        source_authority=authority,
        delivery_contract=(
            "<delivery_contract> T1: status=required; "
            "required_outputs=['waiting_period_rule'] "
            "</delivery_contract>"
        ),
    )
    assert "source_authority_ungrounded_answer" not in flags


def test_guard_messages_include_contract_and_observations() -> None:
    class FakeClient:
        async def chat(self, **kwargs):  # pragma: no cover
            raise AssertionError("chat should not be called")

    from app.agent_graph.llm_output_guard import LLMOutputGuard

    guard = LLMOutputGuard(
        llm_client=FakeClient(),  # type: ignore[arg-type]
    )
    authority = SourceAuthorityContract(
        domain_heuristics="forbidden",
    )
    messages = guard.build_messages(
        OutputGuardRequest(
            request_id="request_test",
            run_id="run_test",
            user_message="请只根据文档分析",
            loop_result=_loop_result(),
            synthesis=_synthesis(answer="回答内容。"),
            citations=[],
            source_authority=authority,
            requirement_observations=[
                {
                    "requirement_id": "T1:1",
                    "status": "direct_support",
                    "citation_ids": [1],
                }
            ],
        )
    )

    system_contents = [
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "system"
    ]
    assert any(
        "<source_authority_contract>" in content
        for content in system_contents
    )

    user_payload = json.loads(
        str(messages[-1].get("content") or "").split("\n", 1)[1]
    )
    assert (
        user_payload["source_authority"]["domain_heuristics"]
        == "forbidden"
    )
    assert user_payload["requirement_observations"][0][
        "requirement_id"
    ] == "T1:1"
