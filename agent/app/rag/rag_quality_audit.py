from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.rag.rag_audit import RagCitationAuditor


@dataclass
class RagQualityAuditResult:
    rag_used: bool
    rag_sufficient: bool | None
    retrieved_count: int | None
    citation_count: int
    answer_has_citation: bool
    citation_consistent: bool
    quality_level: str
    issues: list[dict[str, Any]]

    def model_dump(self) -> dict[str, Any]:
        return {
            "rag_used": self.rag_used,
            "rag_sufficient": self.rag_sufficient,
            "retrieved_count": self.retrieved_count,
            "citation_count": self.citation_count,
            "answer_has_citation": self.answer_has_citation,
            "citation_consistent": self.citation_consistent,
            "quality_level": self.quality_level,
            "issues": self.issues,
        }


class RagQualityAuditor:
    """
    RAG 质量审计器。

    它不改变模型行为，不强制模型调用 RAG。
    它只负责对一次回答的 RAG 质量进行结构化评估。

    它和 RagCitationAuditor 的区别：

    RagCitationAuditor:
    - 专门检查答案里的 [1] 和结构化 citations 是否一致。

    RagQualityAuditor:
    - 检查整条 RAG 链路质量。
    - 包括是否调用 RAG、是否检索到内容、证据是否充分、引用是否为空、引用是否一致。
    """

    def __init__(self) -> None:
        self.citation_auditor = RagCitationAuditor()

    def audit(
        self,
        *,
        answer: str,
        executed_tools: list[dict[str, Any]],
        rag_payload: dict[str, Any],
    ) -> RagQualityAuditResult:
        citation_audit = self.citation_auditor.audit(
            answer=answer,
            executed_tools=executed_tools,
        )

        rag_used = bool(rag_payload.get("used"))
        rag_sufficient = rag_payload.get("sufficient")
        retrieved_count = rag_payload.get("retrieved_count")
        citations = rag_payload.get("citations") or []

        citation_count = len(citations)
        issues: list[dict[str, Any]] = []

        if not rag_used and citation_audit.has_text_citation:
            issues.append(
                {
                    "type": "text_citation_without_rag",
                    "severity": "warning",
                    "message": "答案文本出现引用编号，但本轮没有结构化 RAG 工具调用。",
                }
            )

        if rag_used and retrieved_count == 0:
            issues.append(
                {
                    "type": "rag_retrieved_zero_chunks",
                    "severity": "info",
                    "message": "RAG 工具被调用，但没有检索到候选证据。",
                }
            )

        if rag_used and rag_sufficient is False:
            issues.append(
                {
                    "type": "rag_evidence_insufficient",
                    "severity": "info",
                    "message": "RAG 检索或证据审核认为证据不足。",
                }
            )

        if rag_used and rag_sufficient is True and citation_count == 0:
            issues.append(
                {
                    "type": "rag_sufficient_but_no_citations",
                    "severity": "warning",
                    "message": "RAG 证据充分，但结构化 citations 为空。",
                }
            )

        if not citation_audit.citation_consistent:
            issues.append(
                {
                    "type": "rag_citation_inconsistent",
                    "severity": citation_audit.severity,
                    "message": "答案文本引用编号和结构化 citations 不一致。",
                    "detail": citation_audit.model_dump(),
                }
            )

        quality_level = self._decide_quality_level(
            rag_used=rag_used,
            rag_sufficient=rag_sufficient,
            retrieved_count=retrieved_count,
            citation_count=citation_count,
            citation_consistent=citation_audit.citation_consistent,
            issues=issues,
        )

        return RagQualityAuditResult(
            rag_used=rag_used,
            rag_sufficient=rag_sufficient,
            retrieved_count=retrieved_count,
            citation_count=citation_count,
            answer_has_citation=citation_audit.has_text_citation,
            citation_consistent=citation_audit.citation_consistent,
            quality_level=quality_level,
            issues=issues,
        )

    @staticmethod
    def _decide_quality_level(
        *,
        rag_used: bool,
        rag_sufficient: bool | None,
        retrieved_count: int | None,
        citation_count: int,
        citation_consistent: bool,
        issues: list[dict[str, Any]],
    ) -> str:
        warning_count = sum(
            1
            for issue in issues
            if issue.get("severity") == "warning"
        )

        if warning_count > 0:
            return "warning"

        if not rag_used:
            return "not_used"

        if retrieved_count == 0:
            return "no_evidence"

        if rag_sufficient is False:
            return "insufficient_evidence"

        if rag_sufficient is True and citation_count > 0 and citation_consistent:
            return "grounded"

        return "unknown"
