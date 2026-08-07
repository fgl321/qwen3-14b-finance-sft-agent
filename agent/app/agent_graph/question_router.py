from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class QuestionCapability(str, Enum):
    """
    解决一个用户问题时，系统可能需要调用的能力。

    注意：
    一个问题可以同时需要多个能力，不能再使用单一 route 四选一。
    """

    GENERAL_EXPLANATION = "general_explanation"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    FINANCIAL_CALCULATION = "financial_calculation"
    MEMORY_READ = "memory_read"
    COMPLEX_REASONING = "complex_reasoning"


class RuleConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# 固定能力顺序，保证日志、测试和后续执行计划稳定。
CAPABILITY_ORDER: tuple[QuestionCapability, ...] = (
    QuestionCapability.MEMORY_READ,
    QuestionCapability.KNOWLEDGE_RETRIEVAL,
    QuestionCapability.FINANCIAL_CALCULATION,
    QuestionCapability.GENERAL_EXPLANATION,
    QuestionCapability.COMPLEX_REASONING,
)


@dataclass(frozen=True)
class HardRuleRoutingResult:
    """
    硬规则层的判断结果。

    capabilities:
        已经被确定性规则锁定的能力。

    matched_rules:
        命中的规则名称，用于日志审计和问题排查。

    needs_semantic_router:
        True 表示硬规则无法明确判断，需要交给 DeepSeek 语义路由器。

    is_empty_input:
        输入为空时使用，后续可以交给接口校验层处理。
    """

    capabilities: tuple[QuestionCapability, ...]
    matched_rules: tuple[str, ...]
    confidence: RuleConfidence
    needs_semantic_router: bool
    reason: str
    is_empty_input: bool = False

    def has_capability(self, capability: QuestionCapability) -> bool:
        return capability in self.capabilities

    def to_dict(self) -> dict[str, object]:
        return {
            "capabilities": [
                capability.value
                for capability in self.capabilities
            ],
            "matched_rules": list(self.matched_rules),
            "confidence": self.confidence.value,
            "needs_semantic_router": self.needs_semantic_router,
            "reason": self.reason,
            "is_empty_input": self.is_empty_input,
            "router": "hard_rule",
        }


def _compile_patterns(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in patterns
    )


# 只判断用户是否明确要求基于资料或证据回答。
# 不要加入大量金融主题词。
KNOWLEDGE_PATTERNS = _compile_patterns(
    r"(?:根据|结合|按照|参考).{0,12}"
    r"(?:知识库|文档|资料|报告|附件|"
    r"上传(?:的)?(?:文件|资料|文档)|保单|原文)",

    r"(?:知识库|文档|资料|报告|附件|保单|原文).{0,12}"
    r"(?:怎么说|如何说明|有没有|是否提到|"
    r"写了什么|内容是什么)",

    r"(?:引用|标明|给出).{0,8}"
    r"(?:出处|来源|原文|页码|证据)",
)


# 只判断明确的计算意图。
# 特别注意：不能因为出现“18万元”就直接认定为计算问题。
CALCULATION_PATTERNS = _compile_patterns(
    r"(?:帮我)?(?:计算|算一下|测算|换算|折算|估算)",

    r"(?:缺口|金额|比例|比率|范围|月(?:度)?支出|"
    r"备用金|保费|保额).{0,10}"
    r"(?:是多少|多少|怎么算|如何计算)",

    r"(?:应该|需要|大概|建议).{0,8}"
    r"(?:准备|预留|留出|配置).{0,8}"
    r"(?:多少|几个月)",
)


GENERAL_EXPLANATION_PATTERNS = _compile_patterns(
    r"(?:什么是|是什么意思|定义是什么|概念是什么)",

    r"(?:有何区别|有什么区别|区别是什么)",

    r"(?:请|帮我)?(?:用.{0,8})?"
    r"(?:解释|说明).{0,8}"
    r"(?:概念|含义|作用)",
)


MEMORY_PATTERNS = _compile_patterns(
    r"(?:我刚才|我之前|前面我|上次我|"
    r"根据我之前|结合我之前|你还记得|"
    r"刚刚提到|之前提到)",

    r"(?:根据|结合)我的(?:实际)?情况",
)


COMPLEX_REASONING_PATTERNS = _compile_patterns(
    r"(?:应该先.+还是.+|先.+还是.+)",

    r"(?:如何|怎么).{0,6}"
    r"(?:规划|配置|安排|权衡)",

    r"(?:给我|制定|设计|给出|提供).{0,8}"
    r"(?:方案|规划|建议|调整建议|配置建议)",

    r"(?:该不该|要不要|是否应该)",

    r"(?:综合分析|优先级|权衡利弊|资产配置)",
)


def _normalize_text(text: str | None) -> str:
    """
    清理首尾空格，并把连续换行和空白压缩成一个空格。
    """

    return re.sub(r"\s+", " ", text or "").strip()


def _matches_any(
    text: str,
    patterns: tuple[re.Pattern[str], ...],
) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def normalize_capabilities(
    values: Iterable[QuestionCapability | str],
) -> tuple[QuestionCapability, ...]:
    """
    校验、去重并按照固定顺序返回能力列表。

    后续 DeepSeek 返回能力列表后，也复用这个函数做合法性校验。
    """

    parsed: set[QuestionCapability] = set()

    for value in values:
        if isinstance(value, QuestionCapability):
            parsed.add(value)
        else:
            # 非法字符串会抛出 ValueError，
            # 后续由大模型路由异常回退逻辑捕获。
            parsed.add(QuestionCapability(value))

    return tuple(
        capability
        for capability in CAPABILITY_ORDER
        if capability in parsed
    )


def route_by_hard_rules(
    user_message: str,
) -> HardRuleRoutingResult:
    """
    使用确定性规则识别用户明确表达出来的能力需求。

    设计原则：

    1. 规则可以同时命中多个能力。
    2. 不根据单纯数字判断计算问题。
    3. 不按照寿险、房贷、基金等金融主题分类。
    4. 没命中明确规则时，交给 DeepSeek 语义路由器。
    """

    text = _normalize_text(user_message)

    if not text:
        return HardRuleRoutingResult(
            capabilities=(
                QuestionCapability.COMPLEX_REASONING,
            ),
            matched_rules=("empty_input_fallback",),
            confidence=RuleConfidence.LOW,
            needs_semantic_router=False,
            reason=(
                "用户输入为空，采用保守回退，"
                "交给现有 Agent 或输入校验层处理。"
            ),
            is_empty_input=True,
        )

    capabilities: list[QuestionCapability] = []
    matched_rules: list[str] = []

    if _matches_any(text, MEMORY_PATTERNS):
        capabilities.append(
            QuestionCapability.MEMORY_READ
        )
        matched_rules.append(
            "explicit_memory_reference"
        )

    if _matches_any(text, KNOWLEDGE_PATTERNS):
        capabilities.append(
            QuestionCapability.KNOWLEDGE_RETRIEVAL
        )
        matched_rules.append(
            "explicit_knowledge_request"
        )

    if _matches_any(text, CALCULATION_PATTERNS):
        capabilities.append(
            QuestionCapability.FINANCIAL_CALCULATION
        )
        matched_rules.append(
            "explicit_calculation_request"
        )

    if _matches_any(
        text,
        GENERAL_EXPLANATION_PATTERNS,
    ):
        capabilities.append(
            QuestionCapability.GENERAL_EXPLANATION
        )
        matched_rules.append(
            "explicit_concept_explanation"
        )

    if _matches_any(
        text,
        COMPLEX_REASONING_PATTERNS,
    ):
        capabilities.append(
            QuestionCapability.COMPLEX_REASONING
        )
        matched_rules.append(
            "explicit_complex_planning"
        )

    normalized_capabilities = normalize_capabilities(
        capabilities
    )

    if not normalized_capabilities:
        return HardRuleRoutingResult(
            capabilities=(),
            matched_rules=(),
            confidence=RuleConfidence.LOW,
            needs_semantic_router=True,
            reason=(
                "没有命中确定性硬规则，"
                "需要交给语义路由器识别所需能力。"
            ),
        )

    hard_boundary_capabilities = {
        QuestionCapability.KNOWLEDGE_RETRIEVAL,
        QuestionCapability.FINANCIAL_CALCULATION,
    }

    if (
        hard_boundary_capabilities.intersection(
            normalized_capabilities
        )
        or len(normalized_capabilities) >= 2
    ):
        confidence = RuleConfidence.HIGH
    else:
        confidence = RuleConfidence.MEDIUM

    capability_text = "、".join(
        capability.value
        for capability in normalized_capabilities
    )

    return HardRuleRoutingResult(
        capabilities=normalized_capabilities,
        matched_rules=tuple(matched_rules),
        confidence=confidence,
        needs_semantic_router=False,
        reason=(
            "命中明确表达，已锁定所需能力："
            f"{capability_text}。"
        ),
    )
