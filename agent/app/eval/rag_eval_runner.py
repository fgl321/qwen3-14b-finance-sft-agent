from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass
class RagEvalCase:
    case_id: str
    category: str
    message: str
    expected_keywords_any: list[str]
    forbidden_keywords: list[str]
    allow_known_issue: bool = True


@dataclass
class RagEvalResult:
    case_id: str
    category: str
    status: str
    reason: str
    answer: str
    finish_reason: str | None
    rag_used: bool | None
    rag_sufficient: bool | None
    retrieved_count: int | None
    citation_consistent: bool | None
    rag_quality_level: str | None
    rag_quality_issues: list[dict[str, Any]]
    tool_total_calls: int | None
    tool_failed_calls: int | None
    latency_ms: int

    def model_dump(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "status": self.status,
            "reason": self.reason,
            "answer": self.answer,
            "finish_reason": self.finish_reason,
            "rag_used": self.rag_used,
            "rag_sufficient": self.rag_sufficient,
            "retrieved_count": self.retrieved_count,
            "citation_consistent": self.citation_consistent,
            "rag_quality_level": self.rag_quality_level,
            "rag_quality_issues": self.rag_quality_issues,
            "tool_total_calls": self.tool_total_calls,
            "tool_failed_calls": self.tool_failed_calls,
            "latency_ms": self.latency_ms,
        }


class RagEvalRunner:
    """
    RAG 回归评估器。

    它不是为了让所有 case 都强行通过。
    它的作用是：
    1. 固定一组 RAG 问题。
    2. 调用真实 /api/chat。
    3. 收集 answer、rag、rag_audit、rag_quality_audit、tool_audit。
    4. 区分 passed / known_issue / failed。
    5. 给后续调提示词、检索参数、Agent 编排提供依据。
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000",
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def load_cases(self, case_file: str | Path) -> list[RagEvalCase]:
        path = Path(case_file)

        cases: list[RagEvalCase] = []

        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()

            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON：{line}") from exc

            cases.append(
                RagEvalCase(
                    case_id=str(payload["case_id"]),
                    category=str(payload.get("category") or "unknown"),
                    message=str(payload["message"]),
                    expected_keywords_any=list(payload.get("expected_keywords_any") or []),
                    forbidden_keywords=list(payload.get("forbidden_keywords") or []),
                    allow_known_issue=bool(payload.get("allow_known_issue", True)),
                )
            )

        return cases

    def run_cases(
        self,
        *,
        cases: list[RagEvalCase],
        tenant_id: str = "default",
        knowledge_base_id: str = "kb_finance_basic",
    ) -> list[RagEvalResult]:
        results: list[RagEvalResult] = []

        for case in cases:
            result = self.run_one_case(
                case=case,
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
            )

            results.append(result)

        return results

    def run_one_case(
        self,
        *,
        case: RagEvalCase,
        tenant_id: str,
        knowledge_base_id: str,
    ) -> RagEvalResult:
        user_id = f"rag_eval_user_{uuid.uuid4()}"
        thread_id = f"rag_eval_thread_{case.case_id}_{uuid.uuid4()}"

        payload = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "thread_id": thread_id,
            "knowledge_base_id": knowledge_base_id,
            "message": case.message,
        }

        started = time.perf_counter()

        with httpx.Client(timeout=self.timeout, trust_env=False) as client:
            response = client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )

        latency_ms = int((time.perf_counter() - started) * 1000)

        if response.status_code != 200:
            return RagEvalResult(
                case_id=case.case_id,
                category=case.category,
                status="failed",
                reason=f"HTTP 状态码不是 200：{response.status_code}，body={response.text}",
                answer="",
                finish_reason=None,
                rag_used=None,
                rag_sufficient=None,
                retrieved_count=None,
                citation_consistent=None,
                rag_quality_level=None,
                rag_quality_issues=[],
                tool_total_calls=None,
                tool_failed_calls=None,
                latency_ms=latency_ms,
            )

        data = response.json()

        answer = str(data.get("answer") or "")
        finish_reason = data.get("finish_reason")

        rag = data.get("rag") or {}
        usage = data.get("usage") or {}

        rag_audit = usage.get("rag_audit") or {}
        rag_quality_audit = usage.get("rag_quality_audit") or {}
        tool_audit = usage.get("tool_audit") or {}

        status, reason = self._judge_case(
            case=case,
            answer=answer,
            rag=rag,
            rag_audit=rag_audit,
            rag_quality_audit=rag_quality_audit,
            tool_audit=tool_audit,
        )

        return RagEvalResult(
            case_id=case.case_id,
            category=case.category,
            status=status,
            reason=reason,
            answer=answer,
            finish_reason=finish_reason,
            rag_used=rag.get("used"),
            rag_sufficient=rag.get("sufficient"),
            retrieved_count=rag.get("retrieved_count"),
            citation_consistent=rag_audit.get("citation_consistent"),
            rag_quality_level=rag_quality_audit.get("quality_level"),
            rag_quality_issues=list(rag_quality_audit.get("issues") or []),
            tool_total_calls=tool_audit.get("total_tool_calls"),
            tool_failed_calls=tool_audit.get("failed_tool_calls"),
            latency_ms=latency_ms,
        )

    def _judge_case(
        self,
        *,
        case: RagEvalCase,
        answer: str,
        rag: dict[str, Any],
        rag_audit: dict[str, Any],
        rag_quality_audit: dict[str, Any],
        tool_audit: dict[str, Any],
    ) -> tuple[str, str]:
        forbidden_hits = [
            keyword
            for keyword in case.forbidden_keywords
            if keyword and keyword in answer
        ]

        if forbidden_hits:
            return (
                "failed",
                f"命中禁止词：{forbidden_hits}",
            )

        failed_tool_calls = int(tool_audit.get("failed_tool_calls") or 0)

        if failed_tool_calls > 0:
            if case.allow_known_issue:
                return (
                    "known_issue",
                    f"存在工具调用失败，failed_tool_calls={failed_tool_calls}",
                )

            return (
                "failed",
                f"存在工具调用失败，failed_tool_calls={failed_tool_calls}",
            )

        keyword_hits = [
            keyword
            for keyword in case.expected_keywords_any
            if keyword and keyword in answer
        ]

        if case.expected_keywords_any and not keyword_hits:
            if case.allow_known_issue:
                return (
                    "known_issue",
                    "答案没有命中任何期望关键词，可能是检索未命中或模型未按预期回答。",
                )

            return (
                "failed",
                "答案没有命中任何期望关键词。",
            )

        rag_quality_level = rag_quality_audit.get("quality_level")
        rag_issue_count = len(rag_quality_audit.get("issues") or [])

        if rag_quality_level == "warning":
            if case.allow_known_issue:
                return (
                    "known_issue",
                    "RAG 质量审计出现 warning，但该 case 允许记录为 known issue。",
                )

            return (
                "failed",
                "RAG 质量审计出现 warning。",
            )

        if rag_issue_count > 0 and case.allow_known_issue:
            return (
                "known_issue",
                f"RAG 质量审计存在 {rag_issue_count} 个 issue，记录为 known issue。",
            )

        return (
            "passed",
            "基础检查通过。",
        )

    @staticmethod
    def summarize(results: list[RagEvalResult]) -> dict[str, Any]:
        summary = {
            "total": len(results),
            "passed": 0,
            "known_issue": 0,
            "failed": 0,
            "avg_latency_ms": 0,
            "results": [item.model_dump() for item in results],
        }

        if results:
            summary["avg_latency_ms"] = int(
                sum(item.latency_ms for item in results) / len(results)
            )

        for item in results:
            if item.status == "passed":
                summary["passed"] += 1
            elif item.status == "known_issue":
                summary["known_issue"] += 1
            elif item.status == "failed":
                summary["failed"] += 1

        return summary
