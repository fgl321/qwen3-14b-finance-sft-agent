from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.eval.production_eval_runner import ProductionEvalRunner


SUPPORTED_EVAL_DOC_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".json",
    ".jsonl",
    ".csv",
    ".pdf",
    ".docx",
}


def ingest_documents(
    *,
    doc_dir: str | Path,
    tenant_id: str,
    user_id: str,
    knowledge_base_id: str,
) -> list[dict]:
    from app.rag.document_lifecycle import RagDocumentLifecycleService

    service = RagDocumentLifecycleService()
    service.init_schema()

    results: list[dict] = []
    for path in sorted(Path(doc_dir).iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EVAL_DOC_EXTENSIONS:
            continue
        result = service.ingest_file(
            path=path,
            title=path.stem,
            tenant_id=tenant_id,
            owner_user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            source="eval_doc",
            replace_same_title=False,
        )
        results.append(
            {
                "file": path.name,
                "document_id": result.get("document_id"),
                "status": result.get("status"),
                "duplicate": result.get("duplicate", False),
            }
        )
    return results


def render_markdown(summary: dict) -> str:
    lines = [
        "# 生产链路评测报告",
        "",
        f"- 生成时间：{summary.get('generated_at')}",
        f"- 总用例：{summary.get('total')}",
        f"- 通过：{summary.get('passed')}",
        f"- 已知问题：{summary.get('known_issue')}",
        f"- 失败：{summary.get('failed')}",
        f"- 通过率：{summary.get('pass_rate')}",
        f"- 平均延迟：{summary.get('avg_latency_ms')} ms",
        "",
        "## 检索与回答指标",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
    ]
    metrics = summary.get("metrics") or {}
    for key in (
        "mean_recall_at_3",
        "mean_recall_at_5",
        "mean_mrr",
        "mean_ndcg_at_5",
        "citation_hit_rate",
        "mean_citation_precision",
    ):
        lines.append(f"| {key} | {metrics.get(key, 0.0)} |")

    lines.append("")
    lines.append("## 分类统计")
    lines.append("")
    lines.append("| 分类 | 总数 | 通过 | 已知问题 | 失败 |")
    lines.append("|---|---:|---:|---:|---:|")
    category_stats = summary.get("category_stats") or {}
    for category, stats in sorted(category_stats.items()):
        lines.append(
            f"| {category} | {stats.get('total')} | {stats.get('passed')} "
            f"| {stats.get('known_issue')} | {stats.get('failed')} |"
        )

    lines.append("")
    lines.append("## Case 明细")
    lines.append("")
    lines.append("| Case | 分类 | 状态 | 原因 |")
    lines.append("|---|---|---|---|")
    for case in summary.get("cases") or []:
        reason = str(case.get("reason") or "").replace("\n", " ")[:160]
        lines.append(
            f"| {case.get('case_id')} | {case.get('category')} "
            f"| {case.get('status')} | {reason} |"
        )

    lines.append("")
    lines.append("## 失败与已知问题明细")
    lines.append("")
    for case in summary.get("cases") or []:
        if case.get("status") == "passed":
            continue
        lines.append(f"### {case.get('case_id')} ({case.get('status')})")
        lines.append("")
        lines.append(f"{case.get('reason')}")
        for turn in case.get("turns") or []:
            if turn.get("status") == "passed":
                continue
            lines.append("")
            lines.append(f"- turn{turn.get('turn_index')}: {turn.get('reason')}")
            lines.append(f"- 回答：{str(turn.get('answer') or '')[:300]}")
        lines.append("")
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="生产链路评测：入库评测文档并调用 /api/chat/graph-v2"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument(
        "--cases",
        default="docs/eval/cases.jsonl",
        help="评测用例文件（JSONL）",
    )
    parser.add_argument(
        "--ingest-dir",
        default=None,
        help="评测文档目录，提供后先入库再跑用例",
    )
    parser.add_argument("--output-dir", default="reports/production_eval")
    parser.add_argument("--tenant-id", default="default")
    parser.add_argument("--user-id", default="eval_user")
    parser.add_argument(
        "--knowledge-base-id",
        default="kb_finance_basic",
    )
    args = parser.parse_args()

    ingest_results: list[dict] = []
    if args.ingest_dir:
        ingest_results = ingest_documents(
            doc_dir=args.ingest_dir,
            tenant_id=args.tenant_id,
            user_id=args.user_id,
            knowledge_base_id=args.knowledge_base_id,
        )
        print("=== 入库结果 ===")
        print(json.dumps(ingest_results, ensure_ascii=False, indent=2))

    runner = ProductionEvalRunner(
        base_url=args.base_url,
        tenant_id=args.tenant_id,
        user_id=args.user_id,
        knowledge_base_id=args.knowledge_base_id,
    )
    cases = runner.load_cases(args.cases)
    print(f"=== 加载 {len(cases)} 个用例 ===")

    results = []
    for case in cases:
        result = await runner.run_case(case)
        results.append(result)
        print(f"[{result.status}] {case.case_id} ({case.category})")
        if result.reason:
            print(f"    {result.reason}")

    summary = runner.summarize(results)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "production_eval.json"
    md_path = output_dir / "production_eval.md"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(summary), encoding="utf-8")

    print("=== 汇总 ===")
    print(json.dumps(
        {
            "total": summary["total"],
            "passed": summary["passed"],
            "known_issue": summary["known_issue"],
            "failed": summary["failed"],
            "pass_rate": summary["pass_rate"],
            "metrics": summary["metrics"],
            "report_json": str(json_path),
            "report_md": str(md_path),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    asyncio.run(main())
