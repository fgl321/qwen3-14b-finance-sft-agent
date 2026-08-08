from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.agent_graph.prompts.synthesis_prompt import (
    OUTPUT_GUARD_REPAIR_PROMPT,
    OUTPUT_GUARD_SYSTEM_PROMPT,
)
from app.agent_graph.schemas.loop_schema import AgentLoopResult
from app.agent_graph.schemas.synthesis_schema import (
    OutputGuardResult,
    SynthesisResult,
)
from app.core.logging import get_logger


logger = get_logger(__name__)


SUBMIT_GUARD_TOOL = "submit_output_guard_result"


_FORBIDDEN_PATTERNS = {
    "hidden_reasoning": re.compile(
        r"<\s*/?\s*think\b",
        re.IGNORECASE,
    ),
}


class OutputGuardLLMClient(Protocol):
    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        thinking_enabled: bool = False,
        max_completion_tokens: int = 800,
    ) -> dict[str, Any]:
        ...


@dataclass(slots=True)
class OutputGuardRequest:
    request_id: str
    run_id: str

    user_message: str

    loop_result: AgentLoopResult
    synthesis: SynthesisResult

    citations: list[dict[str, Any]] = field(
        default_factory=list
    )
    # 用户上下文（短期记忆历史摘要 + 长期记忆事实）。
    # 这些是用户明确提供或已保存的个人事实，属于合法回答依据。
    context_summary: str = field(default="")


class OutputGuardInvocationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: OutputGuardResult

    model: str | None = None
    finish_reason: str = ""

    usage: dict[str, Any] = Field(default_factory=dict)

    attempts: int = Field(default=1, ge=1)
    protocol_repaired: bool = False

    error: str | None = None


class OutputGuardProtocolError(ValueError):
    pass


def _guard_tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": SUBMIT_GUARD_TOOL,
            "description": (
                "提交最终金融回答的安全与一致性检查结果。"
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": [
                            "pass",
                            "rewrite",
                            "fallback",
                        ],
                    },
                    "reason": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 1500,
                    },
                    "risk_flags": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },
                    "rewrite_instructions": {
                        "type": [
                            "string",
                            "null",
                        ],
                    },
                },
                "required": [
                    "verdict",
                    "reason",
                    "risk_flags",
                    "rewrite_instructions",
                ],
            },
        },
    }


def _parse_arguments(
    raw_arguments: Any,
) -> dict[str, Any]:
    if isinstance(raw_arguments, dict):
        return raw_arguments

    if not isinstance(raw_arguments, str):
        raise OutputGuardProtocolError(
            "Output Guard arguments 格式错误。"
        )

    try:
        payload = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise OutputGuardProtocolError(
            "Output Guard arguments 不是合法 JSON。"
        ) from exc

    if not isinstance(payload, dict):
        raise OutputGuardProtocolError(
            "Output Guard arguments 顶层必须是对象。"
        )

    return payload

def _normalize_guard_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    兼容模型可能产生的常见同义字段。

    最终仍统一转换为：
    - verdict
    - reason
    - risk_flags
    - rewrite_instructions
    """

    normalized = dict(payload)

    if "verdict" not in normalized:
        normalized["verdict"] = (
            normalized.pop("decision", None)
            or normalized.pop("action", None)
            or normalized.pop("result", None)
        )

    raw_verdict = str(
        normalized.get("verdict") or ""
    ).strip().lower()

    verdict_aliases = {
        "pass": "pass",
        "allow": "pass",
        "allowed": "pass",
        "approve": "pass",
        "approved": "pass",
        "safe": "pass",
        "ok": "pass",

        "rewrite": "rewrite",
        "revise": "rewrite",
        "modify": "rewrite",
        "repair": "rewrite",

        "fallback": "fallback",
        "reject": "fallback",
        "rejected": "fallback",
        "block": "fallback",
        "blocked": "fallback",
        "unsafe": "fallback",
    }

    normalized["verdict"] = (
        verdict_aliases.get(
            raw_verdict,
            raw_verdict,
        )
    )

    if "reason" not in normalized:
        normalized["reason"] = (
            normalized.pop(
                "explanation",
                None,
            )
            or normalized.pop(
                "message",
                None,
            )
            or normalized.pop(
                "feedback",
                None,
            )
            or "输出检查已完成。"
        )

    if "risk_flags" not in normalized:
        normalized["risk_flags"] = (
            normalized.pop("issues", None)
            or normalized.pop("findings", None)
            or normalized.pop("risks", None)
            or []
        )

    risk_flags = normalized.get(
        "risk_flags"
    )

    if risk_flags is None:
        normalized["risk_flags"] = []

    elif isinstance(risk_flags, str):
        cleaned_flag = risk_flags.strip()

        normalized["risk_flags"] = (
            [cleaned_flag]
            if cleaned_flag
            else []
        )

    elif not isinstance(risk_flags, list):
        normalized["risk_flags"] = [
            str(risk_flags)
        ]

    if "rewrite_instructions" not in normalized:
        normalized[
            "rewrite_instructions"
        ] = (
            normalized.pop(
                "rewrite_instruction",
                None,
            )
            or normalized.pop(
                "instructions",
                None,
            )
            or normalized.pop(
                "revision_instructions",
                None,
            )
        )

    rewrite_instructions = normalized.get(
        "rewrite_instructions"
    )

    if isinstance(rewrite_instructions, str):
        cleaned_rewrite = rewrite_instructions.strip()

        if (
            not cleaned_rewrite
            or cleaned_rewrite.lower()
            in {"null", "none", "nil"}
        ):
            normalized[
                "rewrite_instructions"
            ] = None
        else:
            normalized[
                "rewrite_instructions"
            ] = cleaned_rewrite

    if (
        normalized.get("verdict")
        == "rewrite"
        and not normalized.get(
            "rewrite_instructions"
        )
    ):
        normalized[
            "rewrite_instructions"
        ] = (
            str(normalized.get("reason") or "").strip()
            or "根据输出检查结果修正回答。"
        )

    # 删除模型可能附带、但不属于正式协议的字段。
    allowed_fields = {
        "verdict",
        "reason",
        "risk_flags",
        "rewrite_instructions",
    }

    return {
        key: value
        for key, value in normalized.items()
        if key in allowed_fields
    }


_SUCCESSFUL_RESULT_CLAIM_PATTERN = re.compile(
    r"计算结果|结果为|结果是|合理区间|应为|等于|"
    r"\d+(?:\.\d+)?\s*(?:元|万元|%|％)",
    re.IGNORECASE,
)

_FAILURE_DISCLOSURE_PATTERN = re.compile(
    r"无法|失败|未能|不能|暂时不能|没有产生|未产生",
    re.IGNORECASE,
)


def deterministic_output_flags(
    synthesis: SynthesisResult,
    *,
    loop_result: AgentLoopResult | None = None,
    citations: list[dict[str, Any]] | None = None,
) -> list[str]:
    """
    不依赖第二个模型判断的最小证据一致性检查。

    这里不规定所有请求都必须调用工具：
    - 没有走工具路径时，允许 used_tool_call_ids 为空；
    - 一旦存在成功工具结果，引用的工具调用编号必须真实；
    - 工具全部失败时，不允许生成看起来成功的确定性计算结论。
    """

    flags: list[str] = []

    answer = synthesis.answer

    if not answer.strip():
        flags.append("empty_answer")

    if len(answer) > 8000:
        flags.append("answer_too_long")

    for flag_name, pattern in (
        _FORBIDDEN_PATTERNS.items()
    ):
        if pattern.search(answer):
            flags.append(flag_name)

    if loop_result is not None:
        successful_tool_ids = {
            item.tool_call_id
            for item in loop_result.tool_results
            if item.success
        }
        failed_tool_ids = {
            item.tool_call_id
            for item in loop_result.tool_results
            if not item.success
        }
        used_tool_ids = set(
            synthesis.used_tool_call_ids
        )

        if used_tool_ids - successful_tool_ids:
            flags.append(
                "invalid_used_tool_call_ids"
            )

        if (
            successful_tool_ids
            and not used_tool_ids
        ):
            flags.append(
                "missing_used_tool_call_ids"
            )

        if used_tool_ids & failed_tool_ids:
            flags.append(
                "failed_tool_result_referenced"
            )

        if (
            failed_tool_ids
            and not successful_tool_ids
            and _SUCCESSFUL_RESULT_CLAIM_PATTERN.search(
                answer
            )
            and not _FAILURE_DISCLOSURE_PATTERN.search(
                answer
            )
        ):
            flags.append(
                "successful_conclusion_after_tool_failure"
            )

    allowed_citation_ids = {
        str(item.get("citation_id"))
        for item in (citations or [])
        if item.get("citation_id")
    }
    used_citation_ids = set(
        synthesis.used_citation_ids
    )

    if used_citation_ids - allowed_citation_ids:
        flags.append(
            "invalid_used_citation_ids"
        )

    return list(dict.fromkeys(flags))


def _deterministic_rewrite_instructions(
    flags: list[str],
) -> str:
    instructions: list[str] = []

    if "missing_used_tool_call_ids" in flags:
        instructions.append(
            "如果回答使用了成功工具结果，"
            "必须填写真实的 used_tool_call_ids。"
        )

    if (
        "invalid_used_tool_call_ids" in flags
        or "failed_tool_result_referenced" in flags
    ):
        instructions.append(
            "删除不存在或失败的工具调用编号，"
            "只能引用成功工具结果。"
        )

    if (
        "successful_conclusion_after_tool_failure"
        in flags
    ):
        instructions.append(
            "工具没有产生成功结果，"
            "不得输出成功计算结论；"
            "应明确说明本次无法得到可靠结果。"
        )

    if "invalid_used_citation_ids" in flags:
        instructions.append(
            "删除不存在的引用编号，"
            "只能使用系统提供的 citation_id。"
        )

    if "hidden_reasoning" in flags:
        instructions.append(
            "删除思考标签，不要输出 <think> 等隐藏推理内容。"
        )

    if not instructions:
        instructions.append(
            "根据确定性输出检查结果修正回答，"
            "只保留有真实依据的内容。"
        )

    return "".join(instructions)


class LLMOutputGuard:
    def __init__(
        self,
        *,
        llm_client: OutputGuardLLMClient,
        max_completion_tokens: int = 800,
        max_protocol_repairs: int = 1,
    ) -> None:
        if max_completion_tokens <= 0:
            raise ValueError(
                "max_completion_tokens 必须大于 0。"
            )

        if max_protocol_repairs < 0:
            raise ValueError(
                "max_protocol_repairs 不能小于 0。"
            )

        self.llm_client = llm_client
        self.max_completion_tokens = (
            max_completion_tokens
        )
        self.max_protocol_repairs = (
            max_protocol_repairs
        )

    def build_messages(
        self,
        request: OutputGuardRequest,
    ) -> list[dict[str, Any]]:
        payload = {
            "user_message": request.user_message,
            "draft_synthesis": (
                request.synthesis.model_dump(
                    mode="json"
                )
            ),
            "successful_tool_results": [
                item.model_dump(mode="json")
                for item in request.loop_result.tool_results
                if item.success
            ],
            "failed_tool_results": [
                item.model_dump(mode="json")
                for item in request.loop_result.tool_results
                if not item.success
            ],
            "evidence_contract": {
                "used_tool_call_ids": list(
                    request.synthesis.used_tool_call_ids
                ),
                "successful_tool_call_ids": [
                    item.tool_call_id
                    for item
                    in request.loop_result.tool_results
                    if item.success
                ],
                "failed_tool_call_ids": [
                    item.tool_call_id
                    for item
                    in request.loop_result.tool_results
                    if not item.success
                ],
                "direct_answer_without_tools_allowed": (
                    not request.loop_result.tool_results
                ),
            },
            "citations": request.citations,
            "user_context": {
                "context_summary": request.context_summary,
                "note": (
                    "context_summary 中的内容来自短期对话历史或"
                    "用户已确认的长期记忆事实，是合法回答依据。"
                    "基于这些事实回答（例如用户自己提供的收入、"
                    "家庭支出）不得判定为伪造或缺少证据。"
                ),
            },
        }

        return [
            {
                "role": "system",
                "content": OUTPUT_GUARD_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    "请检查以下最终回答草稿：\n"
                    f"{json.dumps(payload, ensure_ascii=False)}"
                ),
            },
        ]

    async def guard(
        self,
        request: OutputGuardRequest,
    ) -> OutputGuardInvocationResult:
        deterministic_flags = (
            deterministic_output_flags(
                request.synthesis,
                loop_result=request.loop_result,
                citations=request.citations,
            )
        )

        if deterministic_flags:
            return OutputGuardInvocationResult(
                result=OutputGuardResult(
                    verdict="rewrite",
                    reason=(
                        "确定性输出规则检测到安全或"
                        "证据一致性问题。"
                    ),
                    risk_flags=deterministic_flags,
                    rewrite_instructions=(
                        _deterministic_rewrite_instructions(
                            deterministic_flags
                        )
                    ),
                )
            )

        messages = self.build_messages(request)

        total_attempts = (
            self.max_protocol_repairs + 1
        )

        last_error: str | None = None

        logger.info(
            "llm_output_guard_started",
            request_id=request.request_id,
            run_id=request.run_id,
        )

        for attempt_index in range(
            1,
            total_attempts + 1,
        ):
            try:
                response = await self.llm_client.chat(
                    messages=messages,
                    tools=[_guard_tool_definition()],
                    thinking_enabled=False,
                    max_completion_tokens=(
                        self.max_completion_tokens
                    ),
                )
            except Exception as exc:
                error_name = type(exc).__name__

                logger.error(
                    "llm_output_guard_call_failed",
                    request_id=request.request_id,
                    run_id=request.run_id,
                    error_type=error_name,
                )

                return OutputGuardInvocationResult(
                    result=OutputGuardResult(
                        verdict="fallback",
                        reason=(
                            "输出安全检查服务不可用，"
                            "当前草稿不能直接返回。"
                        ),
                        risk_flags=[
                            "guard_service_unavailable"
                        ],
                    ),
                    attempts=attempt_index,
                    protocol_repaired=(
                        attempt_index > 1
                    ),
                    error=error_name,
                )

            assistant_message = (
                response.get("message") or {}
            )

            try:
                guard_result = (
                    self._parse_assistant_message(
                        assistant_message
                    )
                )
            except OutputGuardProtocolError as exc:
                last_error = str(exc)

                logger.warning(
                    "llm_output_guard_protocol_error",
                    request_id=request.request_id,
                    run_id=request.run_id,
                    attempt=attempt_index,
                    error=last_error,
                )

                if attempt_index < total_attempts:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"{OUTPUT_GUARD_REPAIR_PROMPT}\n"
                                f"协议错误摘要：{last_error}"
                            ),
                        }
                    )

                continue

            logger.info(
                "llm_output_guard_finished",
                request_id=request.request_id,
                run_id=request.run_id,
                verdict=guard_result.verdict,
                attempts=attempt_index,
                protocol_repaired=(
                    attempt_index > 1
                ),
                model=response.get("model"),
                finish_reason=response.get(
                    "finish_reason",
                    "",
                ),
                usage=response.get("usage", {}),
            )

            return OutputGuardInvocationResult(
                result=guard_result,
                model=response.get("model"),
                finish_reason=response.get(
                    "finish_reason",
                    "",
                ),
                usage=response.get("usage") or {},
                attempts=attempt_index,
                protocol_repaired=(
                    attempt_index > 1
                ),
            )

        return OutputGuardInvocationResult(
            result=OutputGuardResult(
                verdict="fallback",
                reason=(
                    "输出检查器连续返回无效协议。"
                ),
                risk_flags=[
                    "guard_protocol_failure"
                ],
            ),
            attempts=total_attempts,
            protocol_repaired=(
                self.max_protocol_repairs > 0
            ),
            error=last_error,
        )

    def _parse_assistant_message(
        self,
        assistant_message: dict[str, Any],
    ) -> OutputGuardResult:
        if not isinstance(assistant_message, dict):
            raise OutputGuardProtocolError(
                "Output Guard message 不是对象。"
            )

        tool_calls = (
            assistant_message.get("tool_calls") or []
        )

        if (
            not isinstance(tool_calls, list)
            or len(tool_calls) != 1
        ):
            raise OutputGuardProtocolError(
                "Output Guard 必须调用一次 "
                "submit_output_guard_result。"
            )

        function_payload = tool_calls[0].get(
            "function"
        )

        if not isinstance(function_payload, dict):
            raise OutputGuardProtocolError(
                "Output Guard 工具调用缺少 function。"
            )

        tool_name = str(
            function_payload.get("name") or ""
        )

        if tool_name != SUBMIT_GUARD_TOOL:
            raise OutputGuardProtocolError(
                f"Output Guard 调用了非法工具："
                f"{tool_name}"
            )

        raw_payload = _parse_arguments(
            function_payload.get("arguments")
        )

        payload = _normalize_guard_payload(
            raw_payload
        )

        try:
            return OutputGuardResult.model_validate(
                payload
            )
        except Exception as exc:
            raise OutputGuardProtocolError(
                "OutputGuardResult 校验失败："
                f"{type(exc).__name__}: {exc}"
            ) from exc
