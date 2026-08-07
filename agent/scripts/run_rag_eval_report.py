from __future__ import annotations

import json
from pathlib import Path

import httpx

from app.eval.rag_eval_runner import RagEvalRunner
from app.eval.report_writer import RagEvalReportWriter


BASE_URL = "http://127.0.0.1:8000"


def request_health(path: str) -> dict:
    with httpx.Client(timeout=30, trust_env=False) as client:
        response = client.get(f"{BASE_URL}{path}")

    print("GET", path, "status_code:", response.status_code)

    if response.status_code != 200:
        print(response.text)

    response.raise_for_status()
    return response.json()


def main() -> None:
    print()
    print("=" * 80)
    print("RAG Eval Report Started")
    print("=" * 80)

    health = request_health("/health")
    assert health["status"] == "ok"

    memory_health = request_health("/health/memory")
    assert memory_health["status"] == "ok"

    case_file = Path("eval_cases/rag_cases.jsonl")
    assert case_file.exists(), f"评估集不存在：{case_file}"

    runner = RagEvalRunner(base_url=BASE_URL)

    cases = runner.load_cases(case_file)

    results = runner.run_cases(
        cases=cases,
        tenant_id="default",
        knowledge_base_id="kb_finance_basic",
    )

    summary = runner.summarize(results)

    writer = RagEvalReportWriter(
        output_dir="reports",
    )

    paths = writer.write_reports(
        summary=summary,
        report_name="rag_eval_summary",
    )

    print()
    print("=" * 80)
    print("RAG Eval Summary")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print()
    print("=" * 80)
    print("Reports Written")
    print("=" * 80)
    print("json_path:", paths["json_path"])
    print("markdown_path:", paths["markdown_path"])

    if summary["failed"] > 0:
        raise AssertionError(
            f"RAG 评估存在 failed case，数量={summary['failed']}。"
            "请查看 reports\\rag_eval_summary.md。"
        )

    print()
    print("=" * 80)
    print("RAG Eval Report Passed")
    print("=" * 80)


if __name__ == "__main__":
    main()
