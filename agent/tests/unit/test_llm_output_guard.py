import json

import pytest

from app.agent_graph.llm_output_guard import (
    LLMOutputGuard,
    OutputGuardRequest,
    _detect_claim_evidence_attribution_mismatches,
    deterministic_output_flags,
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
from app.agent_graph.schemas.tool_schema import ToolResult


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    async def chat(self, **kwargs):
        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response


def guard_response(
    verdict,
    *,
    reason="检查完成。",
    flags=None,
    rewrite=None,
):
    return {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "guard_1",
                    "type": "function",
                    "function": {
                        "name": (
                            "submit_output_guard_result"
                        ),
                        "arguments": json.dumps(
                            {
                                "verdict": verdict,
                                "reason": reason,
                                "risk_flags": (
                                    flags or []
                                ),
                                "rewrite_instructions": (
                                    rewrite
                                ),
                            },
                            ensure_ascii=False,
                        ),
                    },
                }
            ],
        },
        "model": "deepseek-test",
        "finish_reason": "tool_calls",
        "usage": {},
    }


def loop_result():
    return AgentLoopResult(
        status="completed",
        final_decision=PlannerDecision(
            action="respond",
            decision_reason="完成。",
            confidence="high",
            plan_version=3,
        ),
        tool_results=[
            ToolResult(
                tool_call_id="call_1",
                tool_name="emergency_fund_range",
                success=True,
                output={
                    "min_amount": "45000.00",
                    "max_amount": "90000.00",
                },
            )
        ],
        agent_rounds=3,
        total_tool_calls=1,
        finish_reason="planner_finished",
    )


def request(answer, *, used_tool_call_ids=None):
    return OutputGuardRequest(
        request_id="request_test",
        run_id="run_test",
        user_message="请计算紧急备用金。",
        loop_result=loop_result(),
        synthesis=SynthesisResult(
            answer=answer,
            used_tool_call_ids=(
                used_tool_call_ids
                if used_tool_call_ids is not None
                else ["call_1"]
            ),
            used_citation_ids=[],
            uncertainty=None,
            disclaimer_required=True,
        ),
    )


def test_deterministic_guard_only_detects_structural_issues():
    synthesis = SynthesisResult(
        answer="这是零风险收益，建议贷款投资。",
        used_tool_call_ids=[],
        used_citation_ids=[],
        uncertainty=None,
        disclaimer_required=False,
    )

    flags = deterministic_output_flags(
        synthesis,
        loop_result=loop_result(),
    )

    # 语义安全（承诺收益/诱导杠杆）交由 LLM Guard 判断，
    # 确定性检查不得用关键词误杀否定句。
    assert "guaranteed_return" not in flags
    assert "leverage_encouragement" not in flags

    bad = SynthesisResult(
        answer="回答内容。",
        used_tool_call_ids=["bad_call"],
        used_citation_ids=[],
        uncertainty=None,
        disclaimer_required=False,
    )
    bad_flags = deterministic_output_flags(
        bad,
        loop_result=loop_result(),
    )
    assert "invalid_used_tool_call_ids" in bad_flags


@pytest.mark.anyio
async def test_should_parse_pass():
    guard = LLMOutputGuard(
        llm_client=FakeClient(
            [guard_response("pass")]
        )
    )

    result = await guard.guard(
        request(
            "紧急备用金建议为4.5万至9万元。"
        )
    )

    assert result.result.verdict == "pass"


@pytest.mark.anyio
async def test_deterministic_issue_should_request_rewrite():
    guard = LLMOutputGuard(
        llm_client=FakeClient([])
    )

    # 结构化问题（思考标签泄露）由确定性检查直接拦截，
    # 不依赖第二个模型。
    result = await guard.guard(
        request(
            "回答内容。",
            used_tool_call_ids=["bad_call"],
        )
    )

    assert result.result.verdict == "rewrite"

    assert "invalid_used_tool_call_ids" in (
        result.result.risk_flags
    )

@pytest.mark.anyio
async def test_guard_failure_should_fallback():
    guard = LLMOutputGuard(
        llm_client=FakeClient(
            [ConnectionError("测试异常")]
        )
    )

    result = await guard.guard(
        request(
            "紧急备用金建议为4.5万至9万元。"
        )
    )

    assert result.result.verdict == "fallback"
    assert result.error == "ConnectionError"


def test_claim_evidence_attribution_mismatch_detected():
    citations = [
        {
            "citation_id": 5,
            "quote": (
                "第四十七条 本条例自2013年3月15日起施行。\n"
                "附录五\n"
                "存款保险条例\n"
                "第一条 本条例自2015年5月1日起施行。"
            ),
        }
    ]
    synthesis = SynthesisResult(
        answer="《存款保险条例》自2013年3月15日起施行 [5]",
        used_tool_call_ids=[],
        used_citation_ids=["5"],
        uncertainty=None,
        disclaimer_required=False,
    )
    flags = deterministic_output_flags(
        synthesis,
        citations=citations,
    )
    assert "claim_evidence_attribution_mismatch" in flags


def test_claim_evidence_attribution_ok_when_same_sentence():
    citations = [
        {
            "citation_id": 5,
            "quote": (
                "2015年5月1日，我国《存款保险条例》施行，"
                "存款保险制度正式建立。"
            ),
        }
    ]
    synthesis = SynthesisResult(
        answer="《存款保险条例》自2015年5月1日起施行 [5]",
        used_tool_call_ids=[],
        used_citation_ids=["5"],
        uncertainty=None,
        disclaimer_required=False,
    )
    assert (
        _detect_claim_evidence_attribution_mismatches(
            answer=synthesis.answer,
            citations=citations,
        )
        == []
    )
    flags = deterministic_output_flags(
        synthesis,
        citations=citations,
    )
    assert "claim_evidence_attribution_mismatch" not in flags


def test_claim_evidence_attribution_ignores_ocr_spaces():
    citations = [
        {
            "citation_id": 5,
            "quote": (
                "2015 年5 月1 日，我国《存款保险条例》施行，"
                "存款保险制度正式建立。"
            ),
        }
    ]
    synthesis = SynthesisResult(
        answer="《存款保险条例》自2015年5月1日起施行 [5]",
        used_tool_call_ids=[],
        used_citation_ids=["5"],
        uncertainty=None,
        disclaimer_required=False,
    )
    flags = deterministic_output_flags(
        synthesis,
        citations=citations,
    )
    assert "claim_evidence_attribution_mismatch" not in flags


def test_claim_evidence_attribution_flags_spaced_pollution():
    citations = [
        {
            "citation_id": 5,
            "quote": (
                "第四十七条 本条例自2013 年3 月15 日起施行。\n"
                "附录五\n"
                "存款保险条例\n"
                "第一条 本条例自2015 年5 月1 日起施行。"
            ),
        }
    ]
    synthesis = SynthesisResult(
        answer="《存款保险条例》自2013年3月15日起施行 [5]",
        used_tool_call_ids=[],
        used_citation_ids=["5"],
        uncertainty=None,
        disclaimer_required=False,
    )
    flags = deterministic_output_flags(
        synthesis,
        citations=citations,
    )
    assert "claim_evidence_attribution_mismatch" in flags


def test_case_verdicts_are_structured_and_enum_validated():
    valid = SynthesisResult(
        answer="案例A与案例B的结论。",
        used_tool_call_ids=[],
        used_citation_ids=[],
        uncertainty=None,
        disclaimer_required=False,
        case_verdicts={
            "case_A": "determined",
            "case_B": "conditional",
        },
    )
    assert valid.case_verdicts["case_A"] == "determined"
    with pytest.raises(Exception):
        SynthesisResult(
            answer="x",
            used_tool_call_ids=[],
            used_citation_ids=[],
            uncertainty=None,
            disclaimer_required=False,
            case_verdicts={"case_A": "unknown"},
        )


def test_unverified_state_mutation_claim_detected():
    synthesis = SynthesisResult(
        answer="好的，已记住您的信息：年龄35岁，家庭年收入36万元。",
        used_tool_call_ids=[],
        used_citation_ids=[],
        uncertainty=None,
        disclaimer_required=False,
    )
    flags = deterministic_output_flags(
        synthesis,
        result_reference_context={"has_mutation_intent": False},
    )
    assert "unverified_state_mutation_claim" in flags


def test_unverified_state_mutation_claim_detected_saved_paraphrase():
    synthesis = SynthesisResult(
        answer="这些信息已作为您的个人基础资料保存，后续分析会以此为准。",
        used_tool_call_ids=[],
        used_citation_ids=[],
        uncertainty=None,
        disclaimer_required=False,
    )
    flags = deterministic_output_flags(
        synthesis,
        result_reference_context={"has_mutation_intent": False},
    )
    assert "unverified_state_mutation_claim" in flags


def test_state_mutation_claim_allowed_with_mutation_intent():
    synthesis = SynthesisResult(
        answer="您的年龄已保存到长期记忆。",
        used_tool_call_ids=[],
        used_citation_ids=[],
        uncertainty=None,
        disclaimer_required=False,
    )
    flags = deterministic_output_flags(
        synthesis,
        result_reference_context={"has_mutation_intent": True},
    )
    assert "unverified_state_mutation_claim" not in flags


def test_no_state_mutation_claim_without_intent_is_ok():
    synthesis = SynthesisResult(
        answer="建议每月预留 1.5 万元作为紧急备用金。",
        used_tool_call_ids=[],
        used_citation_ids=[],
        uncertainty=None,
        disclaimer_required=False,
    )
    flags = deterministic_output_flags(
        synthesis,
        result_reference_context={"has_mutation_intent": False},
    )
    assert "unverified_state_mutation_claim" not in flags


def test_committed_state_reference_is_not_flagged():
    synthesis = SynthesisResult(
        answer="根据当前已确认的首付款25万计算，剩余资金为65万。",
        used_tool_call_ids=[],
        used_citation_ids=[],
        uncertainty=None,
        disclaimer_required=False,
    )
    flags = deterministic_output_flags(
        synthesis,
        result_reference_context={
            "has_mutation_intent": False,
            "committed_fact_fields": ["down_payment"],
        },
    )
    assert "unverified_state_mutation_claim" not in flags


def test_committed_state_reference_flagged_when_no_committed_facts():
    synthesis = SynthesisResult(
        answer="根据当前已确认的首付款25万计算，剩余资金为65万。",
        used_tool_call_ids=[],
        used_citation_ids=[],
        uncertainty=None,
        disclaimer_required=False,
    )
    flags = deterministic_output_flags(
        synthesis,
        result_reference_context={
            "has_mutation_intent": False,
            "committed_fact_fields": [],
        },
    )
    assert "unverified_state_mutation_claim" in flags


def test_committed_state_update_reference_is_not_flagged():
    synthesis = SynthesisResult(
        answer="已将首付款更新为25万，因此剩余资金为65万。",
        used_tool_call_ids=[],
        used_citation_ids=[],
        uncertainty=None,
        disclaimer_required=False,
    )
    flags = deterministic_output_flags(
        synthesis,
        result_reference_context={
            "has_mutation_intent": False,
            "committed_fact_fields": ["down_payment", "cash"],
        },
    )
    assert "unverified_state_mutation_claim" not in flags


def test_typed_committed_state_reference_validated():
    synthesis = SynthesisResult(
        answer="已修改后的首付款25万元，剩余资金为65万。",
        used_tool_call_ids=[],
        used_citation_ids=[],
        uncertainty=None,
        disclaimer_required=False,
        state_claim_bindings=[
            {
                "claim_id": "CLAIM_LOCAL_1",
                "claim_type": "committed_state_reference",
                "fact_refs": ["down_payment"],
            }
        ],
    )
    flags = deterministic_output_flags(
        synthesis,
        result_reference_context={
            "has_mutation_intent": False,
            "committed_fact_fields": ["down_payment", "cash"],
        },
    )
    assert "unverified_committed_state_reference" not in flags
    assert "unverified_state_mutation_claim" not in flags


def test_typed_committed_state_reference_rejected_when_fact_unknown():
    synthesis = SynthesisResult(
        answer="已修改后的首付款25万元，剩余资金为65万。",
        used_tool_call_ids=[],
        used_citation_ids=[],
        uncertainty=None,
        disclaimer_required=False,
        state_claim_bindings=[
            {
                "claim_id": "CLAIM_LOCAL_1",
                "claim_type": "committed_state_reference",
                "fact_refs": ["down_payment"],
            }
        ],
    )
    flags = deterministic_output_flags(
        synthesis,
        result_reference_context={
            "has_mutation_intent": False,
            "committed_fact_fields": ["cash"],
        },
    )
    assert "unverified_committed_state_reference" in flags


def test_typed_current_turn_ack_requires_mutation_intent():
    synthesis = SynthesisResult(
        answer="已帮你把首付款修改为25万元。",
        used_tool_call_ids=[],
        used_citation_ids=[],
        uncertainty=None,
        disclaimer_required=False,
        state_claim_bindings=[
            {
                "claim_id": "CLAIM_LOCAL_2",
                "claim_type": "current_turn_mutation_ack",
                "fact_refs": ["down_payment"],
                "mutation_receipt_ref": "CURRENT_TURN",
            }
        ],
    )
    flags = deterministic_output_flags(
        synthesis,
        result_reference_context={
            "has_mutation_intent": True,
            "current_turn_mutation_fields": ["down_payment"],
        },
    )
    assert "unverified_state_mutation_claim" not in flags

    flags = deterministic_output_flags(
        synthesis,
        result_reference_context={
            "has_mutation_intent": False,
            "current_turn_mutation_fields": ["down_payment"],
        },
    )
    assert "unverified_state_mutation_claim" in flags


def test_delivery_truth_conflict_when_negating_technical_failure():
    synthesis = SynthesisResult(
        answer="本轮检索正常完成，没有技术异常。",
        used_tool_call_ids=[],
        used_citation_ids=[],
        uncertainty=None,
        disclaimer_required=False,
    )
    flags = deterministic_output_flags(
        synthesis,
        result_reference_context={
            "technical_failures": [
                {
                    "requirement_id": "T5:4",
                    "stage": "evidence_assessment",
                    "status": "assessment_protocol_failed",
                }
            ]
        },
    )
    assert "delivery_truth_conflict" in flags


def test_delivery_discloses_technical_failure_without_conflict():
    synthesis = SynthesisResult(
        answer="检索服务技术异常，该部分证据验证未完成。",
        used_tool_call_ids=[],
        used_citation_ids=[],
        uncertainty=None,
        disclaimer_required=False,
    )
    flags = deterministic_output_flags(
        synthesis,
        result_reference_context={
            "technical_failures": [
                {
                    "requirement_id": "T5:4",
                    "stage": "evidence_assessment",
                    "status": "assessment_protocol_failed",
                }
            ]
        },
    )
    assert "delivery_truth_conflict" not in flags


def test_no_technical_failures_never_conflicts():
    synthesis = SynthesisResult(
        answer="本轮没有技术异常。",
        used_tool_call_ids=[],
        used_citation_ids=[],
        uncertainty=None,
        disclaimer_required=False,
    )
    flags = deterministic_output_flags(
        synthesis,
        result_reference_context={"technical_failures": []},
    )
    assert "delivery_truth_conflict" not in flags


def test_blocked_evidence_citation_used_detected():
    synthesis = SynthesisResult(
        answer="根据文档，存款保险最高偿付限额为50万元 [5]。",
        used_tool_call_ids=[],
        used_citation_ids=["5"],
        uncertainty=None,
        disclaimer_required=False,
    )
    flags = deterministic_output_flags(
        synthesis,
        result_reference_context={
            "requirement_observations": [
                {
                    "requirement_id": "T3:E2",
                    "status": "not_observed",
                    "citation_ids": [5],
                }
            ]
        },
    )
    assert "blocked_evidence_citation_used" in flags


def test_supported_evidence_citation_not_blocked():
    synthesis = SynthesisResult(
        answer="根据文档，存款保险最高偿付限额为50万元 [5]。",
        used_tool_call_ids=[],
        used_citation_ids=["5"],
        uncertainty=None,
        disclaimer_required=False,
    )
    flags = deterministic_output_flags(
        synthesis,
        result_reference_context={
            "requirement_observations": [
                {
                    "requirement_id": "T3:E2",
                    "status": "direct_support",
                    "citation_ids": [5],
                }
            ]
        },
    )
    assert "blocked_evidence_citation_used" not in flags
