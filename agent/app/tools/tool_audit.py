from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolAuditResult:
    total_tool_calls: int
    successful_tool_calls: int
    failed_tool_calls: int
    tool_names: list[str]
    failed_tool_names: list[str]
    has_rag_tool: bool
    has_calculation_tool: bool
    issues: list[dict[str, Any]]

    def model_dump(self) -> dict[str, Any]:
        return {
            "total_tool_calls": self.total_tool_calls,
            "successful_tool_calls": self.successful_tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
            "tool_names": self.tool_names,
            "failed_tool_names": self.failed_tool_names,
            "has_rag_tool": self.has_rag_tool,
            "has_calculation_tool": self.has_calculation_tool,
            "issues": self.issues,
        }


class ToolCallAuditor:
    """
    工具调用审计器。

    它不改变工具执行结果，只做观测和记录。

    生产意义：
    - Agent 出问题时，能看出来是模型没调工具、调错工具、参数错误，还是工具本身失败。
    - 可以后续接入 trace / dashboard。
    """

    RAG_TOOL_NAME = "search_knowledge_base"

    CALCULATION_TOOL_NAMES = {
        "yearly_expense_to_monthly",
        "emergency_fund_range",
        "life_insurance_gap",
    }

    def audit(
        self,
        *,
        executed_tools: list[dict[str, Any]],
    ) -> ToolAuditResult:
        tool_names: list[str] = []
        failed_tool_names: list[str] = []
        issues: list[dict[str, Any]] = []

        successful_tool_calls = 0
        failed_tool_calls = 0

        for index, tool_payload in enumerate(executed_tools):
            tool_name = str(tool_payload.get("tool_name") or "unknown")
            ok = bool(tool_payload.get("ok"))

            tool_names.append(tool_name)

            if ok:
                successful_tool_calls += 1
            else:
                failed_tool_calls += 1
                failed_tool_names.append(tool_name)

                issues.append(
                    {
                        "type": "tool_execution_failed",
                        "severity": "warning",
                        "tool_name": tool_name,
                        "tool_index": index,
                        "error": tool_payload.get("error"),
                        "raw_arguments": tool_payload.get("raw_arguments"),
                    }
                )

            if tool_name == self.RAG_TOOL_NAME and ok:
                result = tool_payload.get("result") or {}
                retrieved_count = result.get("retrieved_count")

                if retrieved_count == 0:
                    issues.append(
                        {
                            "type": "rag_retrieved_zero_chunks",
                            "severity": "info",
                            "tool_name": tool_name,
                            "tool_index": index,
                        }
                    )

        return ToolAuditResult(
            total_tool_calls=len(executed_tools),
            successful_tool_calls=successful_tool_calls,
            failed_tool_calls=failed_tool_calls,
            tool_names=tool_names,
            failed_tool_names=failed_tool_names,
            has_rag_tool=self.RAG_TOOL_NAME in tool_names,
            has_calculation_tool=any(
                item in self.CALCULATION_TOOL_NAMES
                for item in tool_names
            ),
            issues=issues,
        )
