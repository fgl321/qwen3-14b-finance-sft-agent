from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckResult:
    name: str
    command: list[str]
    ok: bool
    return_code: int
    elapsed_seconds: float


class CheckRunner:
    """
    统一回归检查入口。

    设计目标：
    1. 不再记一堆测试脚本。
    2. 每个阶段独立运行，方便定位失败点。
    3. 失败后立即停止，避免后续错误干扰判断。
    4. 输出总耗时和每个阶段结果。
    """

    def __init__(self) -> None:
        self.project_root = Path.cwd()
        self.python = sys.executable

    def run_all(self) -> list[CheckResult]:
        checks = [
            (
                "Stage 1 Foundation",
                [self.python, "scripts/test_stage_1_foundation.py"],
            ),
            (
                "Stage 2 RAG Foundation",
                [self.python, "scripts/test_stage_2_rag_foundation.py"],
            ),
            (
                "Stage 2 RAG Eval",
                [self.python, "scripts/test_stage_2_rag_eval.py"],
            ),
            (
                "RAG Eval Report",
                [self.python, "scripts/run_rag_eval_report.py"],
            ),
        ]

        results: list[CheckResult] = []

        for name, command in checks:
            result = self._run_one(
                name=name,
                command=command,
            )

            results.append(result)

            if not result.ok:
                self._print_summary(results)
                raise SystemExit(result.return_code)

        self._print_summary(results)
        return results

    def _run_one(
        self,
        *,
        name: str,
        command: list[str],
    ) -> CheckResult:
        self._print_header(name, command)

        started = time.perf_counter()

        completed = subprocess.run(
            command,
            cwd=self.project_root,
            text=True,
        )

        elapsed_seconds = time.perf_counter() - started

        ok = completed.returncode == 0

        if ok:
            print()
            print(f"[OK] {name} passed, elapsed={elapsed_seconds:.2f}s")
        else:
            print()
            print(f"[FAILED] {name} failed, return_code={completed.returncode}, elapsed={elapsed_seconds:.2f}s")

        return CheckResult(
            name=name,
            command=command,
            ok=ok,
            return_code=completed.returncode,
            elapsed_seconds=elapsed_seconds,
        )

    @staticmethod
    def _print_header(
        name: str,
        command: list[str],
    ) -> None:
        print()
        print("=" * 100)
        print(f"Running: {name}")
        print("=" * 100)
        print("command:", " ".join(command))
        print()

    @staticmethod
    def _print_summary(
        results: list[CheckResult],
    ) -> None:
        print()
        print("=" * 100)
        print("All Checks Summary")
        print("=" * 100)

        total_elapsed = 0.0

        for item in results:
            total_elapsed += item.elapsed_seconds
            status = "PASSED" if item.ok else "FAILED"

            print(
                f"{status:8} | "
                f"{item.elapsed_seconds:8.2f}s | "
                f"{item.name}"
            )

        print("-" * 100)
        print(f"Total elapsed: {total_elapsed:.2f}s")

        failed_items = [
            item
            for item in results
            if not item.ok
        ]

        if failed_items:
            print()
            print("Result: FAILED")
            print("请优先查看第一个 FAILED 阶段的终端输出。")
        else:
            print()
            print("Result: PASSED")
            print("所有阶段检查通过。")


def main() -> None:
    runner = CheckRunner()
    runner.run_all()


if __name__ == "__main__":
    main()
