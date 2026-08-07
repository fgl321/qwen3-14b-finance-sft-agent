from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class RagEvalReportWriter:
    """
    RAG 评估报告生成器。

    输入：
    - RagEvalRunner.summarize(results) 得到的 summary

    输出：
    - JSON 报告：方便机器读取、后续做 dashboard
    - Markdown 报告：方便人看、写简历、写项目文档、面试展示
    """

    def __init__(
        self,
        *,
        output_dir: str | Path = "reports",
    ) -> None:
        self.output_dir = Path(output_dir)

    def write_reports(
        self,
        *,
        summary: dict[str, Any],
        report_name: str = "rag_eval_summary",
    ) -> dict[str, str]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        json_path = self.output_dir / f"{report_name}.json"
        md_path = self.output_dir / f"{report_name}.md"

        json_path.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        md_path.write_text(
            self._build_markdown(summary),
            encoding="utf-8",
        )

        return {
            "json_path": str(json_path),
            "markdown_path": str(md_path),
        }

    def _build_markdown(
        self,
        summary: dict[str, Any],
    ) -> str:
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        total = int(summary.get("total") or 0)
        passed = int(summary.get("passed") or 0)
        known_issue = int(summary.get("known_issue") or 0)
        failed = int(summary.get("failed") or 0)
        avg_latency_ms = int(summary.get("avg_latency_ms") or 0)

        results = list(summary.get("results") or [])

        lines: list[str] = []

        lines.append("# RAG 回归评估报告")
        lines.append("")
        lines.append(f"- 生成时间：{generated_at}")
        lines.append(f"- 总用例数：{total}")
        lines.append(f"- 通过：{passed}")
        lines.append(f"- 已知问题：{known_issue}")
        lines.append(f"- 失败：{failed}")
        lines.append(f"- 平均延迟：{avg_latency_ms} ms")
        lines.append("")

        lines.append("## 总览")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|---|---:|")
        lines.append(f"| Total | {total} |")
        lines.append(f"| Passed | {passed} |")
        lines.append(f"| Known Issue | {known_issue} |")
        lines.append(f"| Failed | {failed} |")
        lines.append(f"| Avg Latency(ms) | {avg_latency_ms} |")
        lines.append("")

        lines.append("## Case 明细")
        lines.append("")
        lines.append(
            "| Case ID | Category | Status | RAG Used | RAG Sufficient | "
            "Retrieved | Citation Consistent | Quality | Tool Failed | Latency(ms) |"
        )
        lines.append("|---|---|---|---:|---:|---:|---:|---|---:|---:|")

        for item in results:
            lines.append(
                "| "
                + " | ".join(
                    [
                        self._escape_md(str(item.get("case_id"))),
                        self._escape_md(str(item.get("category"))),
                        self._escape_md(str(item.get("status"))),
                        self._escape_md(str(item.get("rag_used"))),
                        self._escape_md(str(item.get("rag_sufficient"))),
                        self._escape_md(str(item.get("retrieved_count"))),
                        self._escape_md(str(item.get("citation_consistent"))),
                        self._escape_md(str(item.get("rag_quality_level"))),
                        self._escape_md(str(item.get("tool_failed_calls"))),
                        self._escape_md(str(item.get("latency_ms"))),
                    ]
                )
                + " |"
            )

        lines.append("")

        lines.append("## 问题与原因")
        lines.append("")

        for item in results:
            status = str(item.get("status"))
            case_id = str(item.get("case_id"))
            reason = str(item.get("reason") or "")
            issues = list(item.get("rag_quality_issues") or [])

            if status == "passed" and not issues:
                continue

            lines.append(f"### {self._escape_md(case_id)}")
            lines.append("")
            lines.append(f"- 状态：{self._escape_md(status)}")
            lines.append(f"- 原因：{self._escape_md(reason)}")
            lines.append(f"- RAG 质量：{self._escape_md(str(item.get('rag_quality_level')))}")
            lines.append(f"- 引用一致性：{self._escape_md(str(item.get('citation_consistent')))}")

            if issues:
                lines.append("- RAG issues：")

                for issue in issues:
                    issue_type = self._escape_md(str(issue.get("type")))
                    severity = self._escape_md(str(issue.get("severity")))
                    message = self._escape_md(str(issue.get("message")))

                    lines.append(f"  - `{issue_type}` / `{severity}`：{message}")

            answer = self._shorten(str(item.get("answer") or ""), max_length=500)

            if answer:
                lines.append("")
                lines.append("回答摘要：")
                lines.append("")
                lines.append("> " + self._escape_blockquote(answer))

            lines.append("")

        lines.append("## 结论")
        lines.append("")

        if failed > 0:
            lines.append(
                "本次 RAG 回归评估存在失败用例，说明本次修改可能引入了不可接受的退化，"
                "需要优先排查 failed case。"
            )
        elif known_issue > 0:
            lines.append(
                "本次 RAG 回归评估没有 failed case，但存在 known issue。"
                "这些问题当前被允许保留，后续可通过检索策略、提示词、工具调用策略或知识库质量继续优化。"
            )
        else:
            lines.append(
                "本次 RAG 回归评估全部通过，当前 RAG 基础链路稳定。"
            )

        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _escape_md(text: str) -> str:
        return text.replace("|", "\\|").replace("\n", " ")

    @staticmethod
    def _escape_blockquote(text: str) -> str:
        return text.replace("\n", "\n> ")

    @staticmethod
    def _shorten(
        text: str,
        *,
        max_length: int,
    ) -> str:
        text = text.strip()

        if len(text) <= max_length:
            return text

        return text[:max_length] + "..."
