from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any

from app.agent.guards.finance_safety_guard import FinanceSafetyGuard
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.llm.deepseek_client import DeepSeekClient
from app.rag.embedding_factory import build_embedding_provider
from app.rag.qdrant_store import QdrantRagStore
from app.rag.rag_service import RagAnswerService
from app.tools.tool_registry import RAG_TOOL_NAME, TOOL_SCHEMAS, execute_tool, is_rag_tool


MAX_TOOL_ROUNDS = 5

logger = get_logger(__name__)


@dataclass
class FinanceAgentResult:
    request_id: str
    answer: str
    executed_tools: list[dict[str, Any]]
    usage: dict[str, Any]
    finish_reason: str
    message_count: int
    safety_check: dict[str, Any]


class FinanceAgent:
    def __init__(
        self,
        llm_client: DeepSeekClient,
        settings: Settings | None = None,
        rag_service: RagAnswerService | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.settings = settings or get_settings()
        self.safety_guard = FinanceSafetyGuard(llm_client)

        self.rag_service = rag_service or RagAnswerService(
            llm_client=llm_client,
            store=QdrantRagStore(settings=self.settings),
            embedding_provider=build_embedding_provider(
                settings=self.settings,
            ),
        )

    async def run(
            self,
            *,
            user_message: str,
            user_id: str,
            thread_id: str | None = None,
            request_id: str | None = None,
            tenant_id: str = "tenant_001",
            knowledge_base_id: str = "kb_finance_basic",
            history_messages: list[dict[str, str]] | None = None,
    ) -> FinanceAgentResult:
        request_id = request_id or str(uuid.uuid4())

        usage_summary: dict[str, Any] = {
            "input_safety": None,
            "agent_rounds": [],
            "tool_executions": [],
            "output_safety": None,
            "rewrite": None,
            "output_safety_after_rewrite": None,
        }

        input_guard_result = await self.safety_guard.assess_input(
            user_message=user_message,
        )

        input_assessment = input_guard_result.assessment
        usage_summary["input_safety"] = input_guard_result.usage

        logger.info(
            "input_safety_assessed",
            request_id=request_id,
            safe=input_assessment.safe,
            decision=input_assessment.decision,
            risk_level=input_assessment.risk_level,
            finding_count=len(input_assessment.findings),
            guard_model=input_guard_result.model,
            guard_finish_reason=input_guard_result.finish_reason,
        )

        if input_assessment.decision == "refuse":
            return FinanceAgentResult(
                request_id=request_id,
                answer=input_assessment.user_message
                or (
                    "这个问题涉及具体投资产品推荐、收益承诺或交易时机判断，"
                    "我不能提供这类建议。可以提供一般性的风险评估框架供你参考。"
                ),
                executed_tools=[],
                usage=usage_summary,
                finish_reason="input_safety_refused",
                message_count=0,
                safety_check={
                    **input_assessment.model_dump(),
                    "stage": "input",
                    "rewritten": False,
                    "guard_model": input_guard_result.model,
                    "guard_finish_reason": input_guard_result.finish_reason,
                    "guard_usage": input_guard_result.usage,
                },
            )

        if input_assessment.decision == "ask_clarification":
            return FinanceAgentResult(
                request_id=request_id,
                answer=input_assessment.user_message
                or "为了避免误导，我需要更多信息后才能继续分析。",
                executed_tools=[],
                usage=usage_summary,
                finish_reason="input_safety_ask_clarification",
                message_count=0,
                safety_check={
                    **input_assessment.model_dump(),
                    "stage": "input",
                    "rewritten": False,
                    "guard_model": input_guard_result.model,
                    "guard_finish_reason": input_guard_result.finish_reason,
                    "guard_usage": input_guard_result.usage,
                },
            )

        messages = [
            {
                "role": "system",
                "content": self._build_system_prompt(),
            }
        ]

        if history_messages:
            messages.extend(history_messages)

        messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        executed_tools: list[dict[str, Any]] = []
        last_finish_reason = ""

        for round_index in range(1, MAX_TOOL_ROUNDS + 1):
            logger.info(
                "agent_round_started",
                request_id=request_id,
                user_id=user_id,
                thread_id=thread_id,
                round_index=round_index,
            )

            result = await self.llm_client.chat(
                messages=messages,
                tools=TOOL_SCHEMAS,
                thinking_enabled=False,
                max_completion_tokens=2048,
            )

            assistant_message = result["message"]
            messages.append(assistant_message)

            last_usage = result.get("usage", {})
            last_finish_reason = result.get("finish_reason", "")

            tool_calls = assistant_message.get("tool_calls") or []

            usage_summary["agent_rounds"].append(
                {
                    "round_index": round_index,
                    "model": result.get("model"),
                    "finish_reason": last_finish_reason,
                    "tool_call_count": len(tool_calls),
                    "usage": last_usage,
                }
            )

            logger.info(
                "agent_round_finished",
                request_id=request_id,
                user_id=user_id,
                thread_id=thread_id,
                round_index=round_index,
                finish_reason=last_finish_reason,
                tool_call_count=len(tool_calls),
                usage=last_usage,
            )

            if not tool_calls:
                raw_answer = assistant_message.get("content", "")

                final_answer, safety_payload, safety_usage = (
                    await self._make_answer_safe_if_needed(
                        user_message=user_message,
                        answer=raw_answer,
                        executed_tools=executed_tools,
                        request_id=request_id,
                    )
                )

                usage_summary.update(safety_usage)

                return FinanceAgentResult(
                    request_id=request_id,
                    answer=final_answer,
                    executed_tools=executed_tools,
                    usage=usage_summary,
                    finish_reason=last_finish_reason,
                    message_count=len(messages),
                    safety_check=safety_payload,
                )

            current_round_tool_payloads: list[dict[str, Any]] = []

            for tool_call in tool_calls:
                tool_payload = await self._execute_one_tool(
                    tool_call=tool_call,
                    request_id=request_id,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    knowledge_base_id=knowledge_base_id,
                )

                executed_tools.append(tool_payload)
                current_round_tool_payloads.append(tool_payload)

                usage_summary["tool_executions"].append(
                    {
                        "tool_name": tool_payload.get("tool_name"),
                        "ok": tool_payload.get("ok"),
                        "usage": tool_payload.get("usage"),
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(
                            tool_payload,
                            ensure_ascii=False,
                        ),
                    }
                )

            direct_rag_answer = self._try_build_direct_rag_answer(
                user_message=user_message,
                current_round_tool_payloads=current_round_tool_payloads,
            )

            if direct_rag_answer is not None:
                logger.info(
                    "rag_direct_answer_selected",
                    request_id=request_id,
                    user_id=user_id,
                    thread_id=thread_id,
                )

                final_answer, safety_payload, safety_usage = (
                    await self._make_answer_safe_if_needed(
                        user_message=user_message,
                        answer=direct_rag_answer,
                        executed_tools=executed_tools,
                        request_id=request_id,
                    )
                )

                usage_summary.update(safety_usage)

                return FinanceAgentResult(
                    request_id=request_id,
                    answer=final_answer,
                    executed_tools=executed_tools,
                    usage=usage_summary,
                    finish_reason="rag_direct_answer",
                    message_count=len(messages),
                    safety_check=safety_payload,
                )

        fallback_answer = (
            "工具调用轮数超过系统上限，已停止。"
            "请检查问题是否过于复杂，或是否存在工具循环。"
        )

        final_answer, safety_payload, safety_usage = (
            await self._make_answer_safe_if_needed(
                user_message=user_message,
                answer=fallback_answer,
                executed_tools=executed_tools,
                request_id=request_id,
            )
        )

        usage_summary.update(safety_usage)

        return FinanceAgentResult(
            request_id=request_id,
            answer=final_answer,
            executed_tools=executed_tools,
            usage=usage_summary,
            finish_reason="max_tool_rounds_exceeded",
            message_count=len(messages),
            safety_check=safety_payload,
        )

    async def _execute_one_tool(
        self,
        *,
        tool_call: dict[str, Any],
        request_id: str,
        user_id: str,
        tenant_id: str,
        knowledge_base_id: str,
    ) -> dict[str, Any]:
        tool_name = tool_call["function"]["name"]
        raw_arguments = tool_call["function"]["arguments"]

        try:
            arguments = self._safe_json_loads(raw_arguments)

            if is_rag_tool(tool_name):
                return await self._execute_rag_tool(
                    arguments=arguments,
                    request_id=request_id,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    knowledge_base_id=knowledge_base_id,
                )

            tool_result = execute_tool(tool_name, arguments)

            payload = {
                "ok": True,
                "tool_name": tool_name,
                "arguments": arguments,
                "result": tool_result,
                "usage": None,
            }

            logger.info(
                "tool_executed",
                request_id=request_id,
                tool_name=tool_name,
                ok=True,
            )

            return payload

        except Exception as exc:
            payload = {
                "ok": False,
                "tool_name": tool_name,
                "raw_arguments": raw_arguments,
                "error": str(exc),
                "usage": None,
            }

            logger.warning(
                "tool_execution_failed",
                request_id=request_id,
                tool_name=tool_name,
                ok=False,
                error=str(exc),
            )

            return payload

    async def _execute_rag_tool(
        self,
        *,
        arguments: dict[str, Any],
        request_id: str,
        user_id: str,
        tenant_id: str,
        knowledge_base_id: str,
    ) -> dict[str, Any]:
        query = str(arguments.get("query") or "").strip()

        if not query:
            raise ValueError("search_knowledge_base 缺少 query 参数。")

        result = await self.rag_service.answer(
            query=query,
            tenant_id=tenant_id,
            owner_user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            child_limit=8,
            parent_limit=4,
        )

        payload = {
            "ok": True,
            "tool_name": RAG_TOOL_NAME,
            "arguments": {
                "query": query,
                "tenant_id": tenant_id,
                "owner_user_id": user_id,
                "knowledge_base_id": knowledge_base_id,
            },
            "result": {
                "answer": result.answer,
                "evidence_assessment": result.evidence_assessment.model_dump(),
                "citations": [
                    citation.model_dump()
                    for citation in result.citations
                ],
                "retrieved_count": len(result.retrieved_chunks),
            },
            "usage": result.usage,
        }

        logger.info(
            "rag_tool_executed",
            request_id=request_id,
            ok=True,
            query=query,
            retrieved_count=len(result.retrieved_chunks),
            sufficient=result.evidence_assessment.sufficient,
        )

        return payload

    def _try_build_direct_rag_answer(
        self,
        *,
        user_message: str,
        current_round_tool_payloads: list[dict[str, Any]],
    ) -> str | None:
        if len(current_round_tool_payloads) != 1:
            return None

        tool_payload = current_round_tool_payloads[0]

        if tool_payload.get("tool_name") != RAG_TOOL_NAME:
            return None

        if not tool_payload.get("ok"):
            return None

        if self._looks_like_mixed_calculation_task(user_message):
            return None

        result = tool_payload.get("result") or {}
        answer = str(result.get("answer") or "").strip()

        if not answer:
            return None

        return answer

    @staticmethod
    def _looks_like_mixed_calculation_task(
        user_message: str,
    ) -> bool:
        calculation_keywords = [
            "计算",
            "测算",
            "算一下",
            "帮我算",
            "多少钱",
            "多少元",
            "多少万",
            "缺口是多少",
            "建议范围",
            "保额",
            "金额",
        ]

        return any(
            keyword in user_message
            for keyword in calculation_keywords
        )

    async def _make_answer_safe_if_needed(
        self,
        *,
        user_message: str,
        answer: str,
        executed_tools: list[dict[str, Any]],
        request_id: str,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        guard_result = await self.safety_guard.assess_output(
            user_message=user_message,
            answer=answer,
            executed_tools=executed_tools,
        )

        assessment = guard_result.assessment

        safety_usage: dict[str, Any] = {
            "output_safety": guard_result.usage,
            "rewrite": None,
            "output_safety_after_rewrite": None,
        }

        safety_payload: dict[str, Any] = assessment.model_dump()
        safety_payload.update(
            {
                "stage": "output",
                "rewritten": False,
                "guard_usage": guard_result.usage,
                "guard_model": guard_result.model,
                "guard_finish_reason": guard_result.finish_reason,
            }
        )

        logger.info(
            "answer_safety_assessed",
            request_id=request_id,
            safe=assessment.safe,
            decision=assessment.decision,
            risk_level=assessment.risk_level,
            finding_count=len(assessment.findings),
            guard_model=guard_result.model,
            guard_finish_reason=guard_result.finish_reason,
        )

        if assessment.decision == "allow":
            return answer, safety_payload, safety_usage

        if assessment.decision == "ask_clarification":
            clarification_answer = assessment.user_message or (
                "为了避免误导，我需要更多信息后才能继续分析。"
            )
            return clarification_answer, safety_payload, safety_usage

        if assessment.decision == "refuse":
            refusal_answer = assessment.user_message or (
                "这个问题涉及具体投资产品推荐、收益承诺或交易时机判断，"
                "我不能提供这类建议。可以提供一般性的风险评估框架供你参考。"
            )
            return refusal_answer, safety_payload, safety_usage

        rewrite_result = await self.safety_guard.rewrite_output(
            user_message=user_message,
            unsafe_answer=answer,
            assessment=assessment,
        )

        rewritten_answer = rewrite_result.answer
        safety_usage["rewrite"] = rewrite_result.usage

        second_guard_result = await self.safety_guard.assess_output(
            user_message=user_message,
            answer=rewritten_answer,
            executed_tools=executed_tools,
        )

        second_assessment = second_guard_result.assessment
        safety_usage["output_safety_after_rewrite"] = second_guard_result.usage

        safety_payload.update(
            {
                "rewritten": True,
                "rewrite_usage": rewrite_result.usage,
                "rewrite_model": rewrite_result.model,
                "rewrite_finish_reason": rewrite_result.finish_reason,
                "safe_after_rewrite": second_assessment.safe,
                "decision_after_rewrite": second_assessment.decision,
                "risk_level_after_rewrite": second_assessment.risk_level,
                "findings_after_rewrite": [
                    item.model_dump() for item in second_assessment.findings
                ],
                "guard_usage_after_rewrite": second_guard_result.usage,
                "guard_model_after_rewrite": second_guard_result.model,
                "guard_finish_reason_after_rewrite": second_guard_result.finish_reason,
            }
        )

        logger.info(
            "answer_reassessed_after_rewrite",
            request_id=request_id,
            safe=second_assessment.safe,
            decision=second_assessment.decision,
            risk_level=second_assessment.risk_level,
            guard_model=second_guard_result.model,
            guard_finish_reason=second_guard_result.finish_reason,
        )

        if second_assessment.decision == "allow":
            return rewritten_answer, safety_payload, safety_usage

        fallback_answer = (
            "这个问题涉及具体投资产品推荐、收益承诺或交易时机判断，"
            "我不能提供“稳赚不赔”“马上买入”这类建议，也不能替你做具体投资决策。"
            "如果你愿意，我可以改为提供一般性的风险评估框架，帮助你了解如何判断产品风险。"
        )

        return fallback_answer, safety_payload, safety_usage

    @staticmethod
    def _safe_json_loads(raw_arguments: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(raw_arguments, dict):
            return raw_arguments

        try:
            arguments = json.loads(raw_arguments)
        except JSONDecodeError as exc:
            raise ValueError(f"工具参数不是合法 JSON：{raw_arguments}") from exc

        if not isinstance(arguments, dict):
            raise ValueError(f"工具参数必须是 JSON object：{raw_arguments}")

        return arguments

    @staticmethod
    def _build_system_prompt() -> str:
        return (
            "你是一个稳健、谨慎、可解释的中文金融规划 Agent。"
            "你的任务是帮助用户进行家庭现金流、紧急备用金、寿险缺口、"
            "基础保障规划和金融知识库问答。"
            "\n\n"
            "重要规则："
            "\n1. 凡是涉及金额、保额、缺口、备用金、月支出、年支出的计算，"
            "必须优先调用金融计算工具，不要自己口算。"
            "\n2. 凡是用户询问金融概念、知识库内容、基础公式、规划原则、文档中的说明，"
            "应优先调用 search_knowledge_base 工具。"
            "\n3. 凡是用户明确说“基于知识库回答”“根据知识库”“文档里怎么说”，"
            "必须调用 search_knowledge_base 工具，不能直接凭常识回答。"
            "\n4. 你可以连续调用多个工具，直到工具结果足够回答用户问题。"
            "\n5. 如果上一个工具结果还不能完整回答用户问题，必须继续调用必要工具。"
            "\n6. 工具返回结果后，你要用中文解释计算过程、关键结果和建议。"
            "\n7. 如果调用了 search_knowledge_base，最终回答必须保留工具返回中的引用编号，例如 [1]。"
            "\n8. 不要编造用户没有提供的信息。"
            "\n9. 不要承诺收益，不要推荐具体金融产品。"
            "\n10. 如果信息不足，要明确说明缺少哪些信息。"
            "\n11. 回答要区分：已知事实、工具计算结果、知识库证据、一般性建议。"
        )
