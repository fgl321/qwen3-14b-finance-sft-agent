from __future__ import annotations

import json
from typing import Any

from app.agent.guards.finance_policy import FINANCE_SAFETY_POLICY
from app.agent.guards.safety_types import (
    SafetyAssessment,
    SafetyGuardResult,
    SafetyRewriteResult,
)
from app.llm.deepseek_client import DeepSeekClient


class FinanceSafetyGuard:
    def __init__(self, llm_client: DeepSeekClient) -> None:
        self.llm_client = llm_client

    async def assess_input(
        self,
        *,
        user_message: str,
    ) -> SafetyGuardResult:
        messages = [
            {
                "role": "system",
                "content": (
                    FINANCE_SAFETY_POLICY
                    + "\n\n你现在审核的是用户输入，而不是 Agent 输出。"
                    + "\n你必须只输出 JSON，不要输出 Markdown，不要输出解释性正文。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请审核下面这个用户金融请求是否适合进入 Agent 主流程。\n\n"
                    f"【用户原始问题】\n{user_message}\n\n"
                    "判断规则：\n"
                    "1. 如果用户只是咨询一般金融知识、家庭保障、现金流、备用金、寿险缺口，可以 allow。\n"
                    "2. 如果用户要求推荐具体产品、稳赚不赔、马上买入、卖出、加杠杆、梭哈，应 refuse。\n"
                    "3. 如果用户信息不足但方向安全，可以 ask_clarification。\n"
                    "4. 如果用户问题本身危险，但可以转成风险教育，可以 refuse，并在 user_message 中给出安全替代说明。\n\n"
                    "请输出如下 JSON：\n"
                    "{\n"
                    '  "safe": false,\n'
                    '  "decision": "refuse",\n'
                    '  "risk_level": "high",\n'
                    '  "findings": [\n'
                    "    {\n"
                    '      "category": "具体投资建议",\n'
                    '      "severity": "high",\n'
                    '      "evidence": "原文片段",\n'
                    '      "reason": "原因"\n'
                    "    }\n"
                    "  ],\n"
                    '  "explanation": "给开发者看的判断说明",\n'
                    '  "user_message": "给用户看的安全回应"\n'
                    "}\n\n"
                    "decision 只能是 allow、refuse、rewrite、ask_clarification。"
                ),
            },
        ]

        result = await self.llm_client.chat(
            messages=messages,
            thinking_enabled=False,
            max_completion_tokens=1024,
            response_format={"type": "json_object"},
        )

        raw = result["message"].get("content", "{}")

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"输入安全裁判返回了非法 JSON：{raw}") from exc

        assessment = SafetyAssessment.model_validate(payload)

        return SafetyGuardResult(
            assessment=assessment,
            usage=result.get("usage", {}),
            model=result.get("model"),
            finish_reason=result.get("finish_reason"),
        )

    async def assess_output(
        self,
        *,
        user_message: str,
        answer: str,
        executed_tools: list[dict[str, Any]],
    ) -> SafetyGuardResult:
        messages = [
            {
                "role": "system",
                "content": (
                    FINANCE_SAFETY_POLICY
                    + "\n\n你现在审核的是 Agent 准备展示给用户的最终回答。"
                    + "\n你必须只输出 JSON，不要输出 Markdown，不要输出解释性正文。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请审核下面这个金融 Agent 回答是否安全。\n\n"
                    f"【用户原始问题】\n{user_message}\n\n"
                    f"【Agent 回答】\n{answer}\n\n"
                    f"【已执行工具】\n"
                    f"{json.dumps(executed_tools, ensure_ascii=False)}\n\n"
                    "请输出如下 JSON：\n"
                    "{\n"
                    '  "safe": true,\n'
                    '  "decision": "allow",\n'
                    '  "risk_level": "low",\n'
                    '  "findings": [],\n'
                    '  "explanation": "判断说明",\n'
                    '  "user_message": "给用户看的安全提示"\n'
                    "}\n\n"
                    "decision 只能是 allow、refuse、rewrite、ask_clarification。"
                ),
            },
        ]

        result = await self.llm_client.chat(
            messages=messages,
            thinking_enabled=False,
            max_completion_tokens=1024,
            response_format={"type": "json_object"},
        )

        raw = result["message"].get("content", "{}")

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"输出安全裁判返回了非法 JSON：{raw}") from exc

        assessment = SafetyAssessment.model_validate(payload)

        return SafetyGuardResult(
            assessment=assessment,
            usage=result.get("usage", {}),
            model=result.get("model"),
            finish_reason=result.get("finish_reason"),
        )

    async def rewrite_output(
        self,
        *,
        user_message: str,
        unsafe_answer: str,
        assessment: SafetyAssessment,
    ) -> SafetyRewriteResult:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是金融回答安全改写助手。"
                    "请在不改变事实和计算结果的前提下，改写回答，使其符合金融安全要求。"
                    "要求："
                    "1. 不承诺收益。"
                    "2. 不推荐具体金融产品。"
                    "3. 不给买入、卖出、加仓、满仓、梭哈等交易指令。"
                    "4. 不替用户做投资决策。"
                    "5. 可以保留风险教育、一般性知识和工具计算结果。"
                    "6. 如果用户要求稳赚不赔或马上买入，要明确拒绝这类要求。"
                    "7. 输出用户可见答案，不要输出 JSON。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"【用户原始问题】\n{user_message}\n\n"
                    f"【不安全回答】\n{unsafe_answer}\n\n"
                    f"【安全审核结果】\n"
                    f"{assessment.model_dump_json(indent=2)}\n\n"
                    "请输出改写后的用户可见答案。"
                ),
            },
        ]

        result = await self.llm_client.chat(
            messages=messages,
            thinking_enabled=False,
            max_completion_tokens=2048,
        )

        return SafetyRewriteResult(
            answer=result["message"].get("content", ""),
            usage=result.get("usage", {}),
            model=result.get("model"),
            finish_reason=result.get("finish_reason"),
        )
