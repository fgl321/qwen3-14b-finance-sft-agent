from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from app.agent_graph.release_contract import (
    STAGE_4_2_8_UNIT_TESTS,
    STAGE_4_2_8_VERSION,
)


def _run(command: list[str], *, root: Path) -> None:
    print("\n$ " + " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
    )

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "执行 Stage 4.2.8 最终发布验收。"
        )
    )
    parser.add_argument(
        "--with-http",
        action="store_true",
        help=(
            "单元回归通过后继续执行真实 HTTP 验收；"
            "使用前需在另一终端启动生产 API。"
        ),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    python = sys.executable

    print(
        f"Stage 4.2.8 final acceptance: "
        f"{STAGE_4_2_8_VERSION}"
    )

    _run(
        [
            python,
            "-m",
            "scripts.verify_stage_4_2_8_release",
        ],
        root=root,
    )
    _run(
        [
            python,
            "-m",
            "pytest",
            *STAGE_4_2_8_UNIT_TESTS,
            "-q",
        ],
        root=root,
    )

    if args.with_http:
        _run(
            [
                python,
                "-m",
                "scripts.test_stage_4_2_8_final_http",
            ],
            root=root,
        )

    print(
        "\nStage 4.2.8F final acceptance passed."
    )


if __name__ == "__main__":
    main()
