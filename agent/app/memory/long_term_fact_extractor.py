from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ExtractedFact:
    fact_type: str
    fact_key: str
    fact_value: dict[str, Any]
    confidence: float


class LongTermFactExtractor:
    """
    长期记忆事实抽取器。

    当前使用规则抽取，先保证金融字段稳定、可控、可测试。

    这一版修复的问题：
    1. “丈夫定寿30万元”不会被误抽成丈夫年龄30岁
    2. 支持“夫妻两人分别35岁和33岁”这种表达
    3. 支持孩子年龄、支出、房贷、资产、寿险、风险偏好
    """

    MONEY_FACT_PATTERNS = [
        {
            "fact_type": "family_finance",
            "fact_key": "annual_necessary_expense",
            "keywords": [
                "家庭年度必要支出",
                "年度必要支出",
                "一年必要支出",
                "每年必要支出",
                "年必要支出",
            ],
            "unit": "year",
        },
        {
            "fact_type": "family_finance",
            "fact_key": "monthly_necessary_expense",
            "keywords": [
                "月度必要支出",
                "每月必要支出",
                "月必要支出",
                "一个月必要支出",
            ],
            "unit": "month",
        },
        {
            "fact_type": "family_finance",
            "fact_key": "mortgage_balance",
            "keywords": [
                "房贷余额",
                "剩余房贷",
                "房贷还剩",
                "房贷还有",
                "贷款余额",
            ],
            "unit": None,
        },
        {
            "fact_type": "family_finance",
            "fact_key": "available_assets",
            "keywords": [
                "已有可用资产",
                "可用资产",
                "现金和低风险资产",
                "现金低风险资产",
                "现金类资产",
                "已有资产",
            ],
            "unit": None,
        },
        {
            "fact_type": "family_finance",
            "fact_key": "annual_income",
            "keywords": [
                "税后年收入",
                "家庭年收入",
                "年度收入",
                "年收入",
            ],
            "unit": "year",
        },
        {
            "fact_type": "family_finance",
            "fact_key": "monthly_income",
            "keywords": [
                "税后月收入",
                "家庭月收入",
                "每月收入",
                "月收入",
            ],
            "unit": "month",
        },
    ]

    RISK_KEYWORDS = {
        "conservative": ["保守", "稳健", "风险低", "低风险", "不能接受亏损"],
        "balanced": ["平衡", "均衡", "中等风险", "适中"],
        "aggressive": ["进取", "激进", "高风险", "能接受较大波动"],
    }

    def extract(self, text: str) -> list[ExtractedFact]:
        if not text:
            return []

        normalized_text = self._normalize_text(text)

        facts: list[ExtractedFact] = []

        facts.extend(self._extract_money_facts(normalized_text, text))
        facts.extend(self._extract_life_insurance(normalized_text, text))
        facts.extend(self._extract_risk_preference(normalized_text, text))
        facts.extend(self._extract_family_ages(normalized_text, text))

        return self._deduplicate(facts)

    def _extract_money_facts(
        self,
        normalized_text: str,
        original_text: str,
    ) -> list[ExtractedFact]:
        facts: list[ExtractedFact] = []

        for item in self.MONEY_FACT_PATTERNS:
            for keyword in item["keywords"]:
                amount = self._find_money_after_keyword(normalized_text, keyword)

                if amount is None:
                    continue

                fact_value: dict[str, Any] = {
                    "amount": amount,
                    "currency": "CNY",
                    "original_text": original_text,
                }

                if item["unit"]:
                    fact_value["unit"] = item["unit"]

                facts.append(
                    ExtractedFact(
                        fact_type=item["fact_type"],
                        fact_key=item["fact_key"],
                        fact_value=fact_value,
                        confidence=0.95,
                    )
                )

                break

        return facts

    def _extract_life_insurance(
        self,
        normalized_text: str,
        original_text: str,
    ) -> list[ExtractedFact]:
        facts: list[ExtractedFact] = []

        wife_no_patterns = [
            "妻子无寿险",
            "妻子没有寿险",
            "老婆无寿险",
            "老婆没有寿险",
            "女方无寿险",
            "女方没有寿险",
        ]

        husband_no_patterns = [
            "丈夫无寿险",
            "丈夫没有寿险",
            "老公无寿险",
            "老公没有寿险",
            "男方无寿险",
            "男方没有寿险",
        ]

        if any(pattern in normalized_text for pattern in wife_no_patterns):
            facts.append(
                ExtractedFact(
                    fact_type="insurance",
                    fact_key="wife_life_insurance",
                    fact_value={
                        "amount": 0,
                        "currency": "CNY",
                        "original_text": original_text,
                    },
                    confidence=0.98,
                )
            )

        if any(pattern in normalized_text for pattern in husband_no_patterns):
            facts.append(
                ExtractedFact(
                    fact_type="insurance",
                    fact_key="husband_life_insurance",
                    fact_value={
                        "amount": 0,
                        "currency": "CNY",
                        "original_text": original_text,
                    },
                    confidence=0.98,
                )
            )

        husband_amount = self._find_money_after_any_keyword(
            normalized_text,
            [
                "丈夫定寿",
                "丈夫寿险",
                "丈夫已有寿险",
                "老公定寿",
                "老公寿险",
                "男方定寿",
                "男方寿险",
            ],
        )

        if husband_amount is not None:
            facts.append(
                ExtractedFact(
                    fact_type="insurance",
                    fact_key="husband_life_insurance",
                    fact_value={
                        "amount": husband_amount,
                        "currency": "CNY",
                        "original_text": original_text,
                    },
                    confidence=0.95,
                )
            )

        wife_amount = self._find_money_after_any_keyword(
            normalized_text,
            [
                "妻子定寿",
                "妻子寿险",
                "妻子已有寿险",
                "老婆定寿",
                "老婆寿险",
                "女方定寿",
                "女方寿险",
            ],
        )

        if wife_amount is not None:
            facts.append(
                ExtractedFact(
                    fact_type="insurance",
                    fact_key="wife_life_insurance",
                    fact_value={
                        "amount": wife_amount,
                        "currency": "CNY",
                        "original_text": original_text,
                    },
                    confidence=0.95,
                )
            )

        return facts

    def _extract_risk_preference(
        self,
        normalized_text: str,
        original_text: str,
    ) -> list[ExtractedFact]:
        for risk_level, keywords in self.RISK_KEYWORDS.items():
            if any(keyword in normalized_text for keyword in keywords):
                return [
                    ExtractedFact(
                        fact_type="preference",
                        fact_key="risk_preference",
                        fact_value={
                            "value": risk_level,
                            "original_text": original_text,
                        },
                        confidence=0.85,
                    )
                ]

        return []

    def _extract_family_ages(
        self,
        normalized_text: str,
        original_text: str,
    ) -> list[ExtractedFact]:
        facts: list[ExtractedFact] = []

        couple_facts = self._extract_couple_ages(normalized_text, original_text)
        facts.extend(couple_facts)

        husband_age = self._find_person_age(
            normalized_text,
            ["丈夫", "老公", "先生", "男方"],
        )

        if husband_age is not None:
            facts.append(
                ExtractedFact(
                    fact_type="family_profile",
                    fact_key="husband_age",
                    fact_value={
                        "age": husband_age,
                        "original_text": original_text,
                    },
                    confidence=0.92,
                )
            )

        wife_age = self._find_person_age(
            normalized_text,
            ["妻子", "老婆", "太太", "女方"],
        )

        if wife_age is not None:
            facts.append(
                ExtractedFact(
                    fact_type="family_profile",
                    fact_key="wife_age",
                    fact_value={
                        "age": wife_age,
                        "original_text": original_text,
                    },
                    confidence=0.92,
                )
            )

        child_age = self._find_person_age(
            normalized_text,
            ["孩子", "小孩", "子女", "儿子", "女儿"],
        )

        if child_age is not None:
            facts.append(
                ExtractedFact(
                    fact_type="family_profile",
                    fact_key="child_age",
                    fact_value={
                        "age": child_age,
                        "original_text": original_text,
                    },
                    confidence=0.92,
                )
            )

        return facts

    def _extract_couple_ages(
        self,
        normalized_text: str,
        original_text: str,
    ) -> list[ExtractedFact]:
        """
        专门处理：
        - 夫妻两人分别35岁和33岁
        - 夫妻分别35岁、33岁
        - 夫妻年龄分别为35岁和33岁

        当前默认：
        第一个年龄 = 丈夫年龄
        第二个年龄 = 妻子年龄
        """
        patterns = [
            r"夫妻(?:两人)?(?:年龄)?分别(?:为|是)?(\d{1,3})岁?(?:和|、|,)(\d{1,3})岁?",
            r"夫妻(?:两人)?(?:年龄)?(?:为|是)(\d{1,3})岁?(?:和|、|,)(\d{1,3})岁?",
        ]

        for pattern in patterns:
            match = re.search(pattern, normalized_text)

            if not match:
                continue

            first_age = int(match.group(1))
            second_age = int(match.group(2))

            if not self._is_valid_age(first_age):
                return []

            if not self._is_valid_age(second_age):
                return []

            return [
                ExtractedFact(
                    fact_type="family_profile",
                    fact_key="husband_age",
                    fact_value={
                        "age": first_age,
                        "original_text": original_text,
                    },
                    confidence=0.95,
                ),
                ExtractedFact(
                    fact_type="family_profile",
                    fact_key="wife_age",
                    fact_value={
                        "age": second_age,
                        "original_text": original_text,
                    },
                    confidence=0.95,
                ),
            ]

        return []

    def _find_person_age(
        self,
        text: str,
        keywords: list[str],
    ) -> int | None:
        """
        找某个人的年龄。

        关键修复点：
        只接受数字后面明确跟着“岁”或“周岁”的情况。
        所以：
        - 丈夫35岁：可以抽取
        - 丈夫定寿30万元：不会抽取
        """
        for keyword in keywords:
            escaped_keyword = re.escape(keyword)

            patterns = [
                escaped_keyword + r".{0,8}?(\d{1,3})(?:岁|周岁)",
                r"(\d{1,3})(?:岁|周岁).{0,8}?" + escaped_keyword,
            ]

            for pattern in patterns:
                match = re.search(pattern, text)

                if not match:
                    continue

                age = int(match.group(1))

                if self._is_valid_age(age):
                    return age

        return None

    def _find_money_after_any_keyword(
        self,
        text: str,
        keywords: list[str],
    ) -> int | None:
        for keyword in keywords:
            amount = self._find_money_after_keyword(text, keyword)

            if amount is not None:
                return amount

        return None

    def _find_money_after_keyword(
        self,
        text: str,
        keyword: str,
    ) -> int | None:
        """
        从关键词附近抽取金额。

        支持：
        - 年度必要支出18万元
        - 年度必要支出是18万
        - 房贷余额80万元
        - 可用资产为250000元
        """
        escaped_keyword = re.escape(keyword)

        pattern = (
            escaped_keyword
            + r".{0,10}?"
            + r"(\d+(?:\.\d+)?)"
            + r"\s*"
            + r"(万元|万|元)?"
        )

        match = re.search(pattern, text)

        if not match:
            return None

        number = float(match.group(1))
        unit = match.group(2) or "元"

        if unit in {"万元", "万"}:
            return int(number * 10000)

        return int(number)

    def _normalize_text(self, text: str) -> str:
        return (
            text.replace("，", ",")
            .replace("。", ".")
            .replace("：", ":")
            .replace("；", ";")
            .replace("、", ",")
            .replace(" ", "")
            .replace("\n", "")
            .strip()
        )

    def _is_valid_age(self, age: int) -> bool:
        return 0 < age <= 120

    def _deduplicate(self, facts: list[ExtractedFact]) -> list[ExtractedFact]:
        result: dict[tuple[str, str], ExtractedFact] = {}

        for fact in facts:
            key = (fact.fact_type, fact.fact_key)

            old_fact = result.get(key)

            if old_fact is None:
                result[key] = fact
                continue

            if fact.confidence >= old_fact.confidence:
                result[key] = fact

        return list(result.values())
