from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class RagCitationAuditResult:
    has_text_citation: bool
    text_citation_numbers: list[int]

    has_rag_tool: bool
    rag_tool_count: int

    has_structured_citation: bool
    structured_citation_ids: list[int]

    citation_consistent: bool
    issue: str | None
    severity: str

    missing_structured_citation_numbers: list[int]
    orphan_structured_citation_ids: list[int]

    def model_dump(self) -> dict[str, Any]:
        return {
            "has_text_citation": self.has_text_citation,
            "text_citation_numbers": self.text_citation_numbers,
            "has_rag_tool": self.has_rag_tool,
            "rag_tool_count": self.rag_tool_count,
            "has_structured_citation": self.has_structured_citation,
            "structured_citation_ids": self.structured_citation_ids,
            "citation_consistent": self.citation_consistent,
            "issue": self.issue,
            "severity": self.severity,
            "missing_structured_citation_numbers": self.missing_structured_citation_numbers,
            "orphan_structured_citation_ids": self.orphan_structured_citation_ids,
        }


class RagCitationAuditor:
    """
    RAG 引用审计器。

    它不改变模型行为，也不强制调用 RAG。

    它只负责检查：
    1. 最终 answer 里有没有 [1] / [2] 这种文本引用。
    2. executed_tools 里有没有 search_knowledge_base 工具调用。
    3. RAG 工具结果里有没有结构化 citations。
    4. answer 里的引用编号和结构化 citations 是否一致。

    生产意义：
    - 保留模型自主工具选择。
    - 同时发现“模型伪造引用”的问题。
    - 把问题记录到 usage.rag_audit，方便后续排查和评估。
    """

    RAG_TOOL_NAME = "search_knowledge_base"

    def audit(
        self,
        *,
        answer: str,
        executed_tools: list[dict[str, Any]],
    ) -> RagCitationAuditResult:
        text_citation_numbers = self.extract_text_citation_numbers(answer)
        structured_citations = self.extract_structured_rag_citations(executed_tools)

        structured_citation_ids = sorted(
            {
                int(item.get("citation_id"))
                for item in structured_citations
                if self._can_cast_int(item.get("citation_id"))
            }
        )

        rag_tool_payloads = [
            item
            for item in executed_tools
            if item.get("tool_name") == self.RAG_TOOL_NAME
        ]

        has_text_citation = len(text_citation_numbers) > 0
        has_rag_tool = len(rag_tool_payloads) > 0
        has_structured_citation = len(structured_citation_ids) > 0

        missing_structured_citation_numbers = [
            number
            for number in text_citation_numbers
            if number not in structured_citation_ids
        ]

        orphan_structured_citation_ids = [
            number
            for number in structured_citation_ids
            if number not in text_citation_numbers
        ]

        issue: str | None = None

        if has_text_citation and not has_rag_tool:
            issue = "answer_contains_citation_but_rag_tool_not_called"

        elif has_text_citation and has_rag_tool and not has_structured_citation:
            issue = "answer_contains_citation_but_rag_tool_has_no_structured_citations"

        elif missing_structured_citation_numbers:
            issue = "answer_citation_numbers_missing_from_structured_citations"

        elif has_structured_citation and not has_text_citation:
            issue = "structured_citations_exist_but_answer_does_not_reference_them"

        citation_consistent = issue is None

        severity = "none"

        if issue in {
            "answer_contains_citation_but_rag_tool_not_called",
            "answer_contains_citation_but_rag_tool_has_no_structured_citations",
            "answer_citation_numbers_missing_from_structured_citations",
        }:
            severity = "warning"

        elif issue == "structured_citations_exist_but_answer_does_not_reference_them":
            severity = "info"

        return RagCitationAuditResult(
            has_text_citation=has_text_citation,
            text_citation_numbers=text_citation_numbers,
            has_rag_tool=has_rag_tool,
            rag_tool_count=len(rag_tool_payloads),
            has_structured_citation=has_structured_citation,
            structured_citation_ids=structured_citation_ids,
            citation_consistent=citation_consistent,
            issue=issue,
            severity=severity,
            missing_structured_citation_numbers=missing_structured_citation_numbers,
            orphan_structured_citation_ids=orphan_structured_citation_ids,
        )

    @staticmethod
    def extract_text_citation_numbers(answer: str) -> list[int]:
        if not answer:
            return []

        numbers: list[int] = []

        for match in re.finditer(r"\[(\d{1,3})\]", answer):
            value = int(match.group(1))

            if value not in numbers:
                numbers.append(value)

        return sorted(numbers)

    @classmethod
    def extract_structured_rag_citations(
        cls,
        executed_tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []

        for tool_payload in executed_tools:
            if tool_payload.get("tool_name") != cls.RAG_TOOL_NAME:
                continue

            result = tool_payload.get("result") or {}
            raw_citations = result.get("citations") or []

            for item in raw_citations:
                if isinstance(item, dict):
                    citations.append(item)

        return citations

    @staticmethod
    def _can_cast_int(value: Any) -> bool:
        try:
            int(value)
            return True
        except Exception:
            return False
