import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def run_command(command: list[str]) -> None:
    print("\n" + "=" * 80)
    print("RUN:", " ".join(command))
    print("=" * 80)

    result = subprocess.run(
        command,
        cwd=ROOT_DIR,
        text=True,
    )

    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 3 LangGraph checks.",
    )

    parser.add_argument(
        "--with-api",
        action="store_true",
        help="同时运行需要 FastAPI 服务的 /api/chat/graph 接口测试。",
    )

    args = parser.parse_args()

    print("========== Stage 3 Checks Started ==========")

    # 1. 不依赖大模型的质量门控单元测试
    run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/test_agent_graph_quality_gate.py",
            "-v",
        ]
    )

    # 2. LangGraph 最小闭环测试
    run_command(
        [
            sys.executable,
            "scripts/test_stage_3_langgraph_minimal.py",
        ]
    )

    # 3. LangGraph 质量门控集成测试
    run_command(
        [
            sys.executable,
            "scripts/test_stage_3_langgraph_quality_gate.py",
        ]
    )

    # 4. Graph Service 测试
    run_command(
        [
            sys.executable,
            "scripts/test_stage_3_graph_service.py",
        ]
    )

    # 5. HTTP API 回归测试，需要你提前启动 uvicorn
    if args.with_api:
        run_command(
            [
                sys.executable,
                "scripts/test_stage_3_chat_graph_api.py",
            ]
        )

        run_command(
            [
                sys.executable,
                "scripts/test_stage_3_api_regression.py",
            ]
        )

    print("\n========== Stage 3 Checks Passed ==========")


if __name__ == "__main__":
    main()
