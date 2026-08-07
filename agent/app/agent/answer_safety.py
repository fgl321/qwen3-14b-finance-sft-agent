from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SafetyCheckResult:
    safe: bool
    risk_level: str
    reasons: list[str]
    matched_terms: list[str]


FORBIDDEN_TERMS = [
    "保证收益",
    "稳赚不赔",
    "保本保收益",
    "本金绝对安全",
    "无风险收益",
    "一定赚钱",
    "必赚",
    "马上买入",
    "立刻买入",
    "全仓买入",
    "梭哈",
    "闭眼买",
    "强烈推荐买入",
]

PRODUCT_RECOMMENDATION_TERMS = [
    "推荐你买",
    "建议你买入",
    "可以买入",
    "值得买入",
    "直接买",
    "买这只",
]

HIGH_RISK_ASSET_TERMS = [
    "股票",
    "基金",
    "期货",
    "期权",
    "虚拟货币",
    "比特币",
    "杠杆",
    "融资融券",
]


def check_answer_safety(answer: str) -> SafetyCheckResult:
    matched_terms: list[str] = []
    reasons: list[str] = []

    for term in FORBIDDEN_TERMS:
        if term in answer:
            matched_terms.append(term)
            reasons.append(f"包含高风险承诺或诱导性表达：{term}")

    for term in PRODUCT_RECOMMENDATION_TERMS:
        if term in answer:
            matched_terms.append(term)
            reasons.append(f"包含具体买入倾向表达：{term}")

    has_product_recommendation = any(
        term in answer for term in PRODUCT_RECOMMENDATION_TERMS
    )
    has_high_risk_asset = any(
        term in answer for term in HIGH_RISK_ASSET_TERMS
    )

    if has_product_recommendation and has_high_risk_asset:
        reasons.append("同时出现买入建议和高风险资产，存在个性化投资建议风险。")

    if reasons:
        return SafetyCheckResult(
            safe=False,
            risk_level="high",
            reasons=reasons,
            matched_terms=matched_terms,
        )

    return SafetyCheckResult(
        safe=True,
        risk_level="low",
        reasons=[],
        matched_terms=[],
    )
