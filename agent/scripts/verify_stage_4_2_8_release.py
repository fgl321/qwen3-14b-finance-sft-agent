from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.agent_graph.production_nodes import (
    prepare_production_run_node,
)
from app.agent_graph.release_contract import (
    STAGE_4_2_8_CAPABILITIES,
    STAGE_4_2_8_RELEASE_NAME,
    STAGE_4_2_8_REQUIRED_FILES,
    STAGE_4_2_8_UNIT_TESTS,
    STAGE_4_2_8_VERSION,
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def verify_release(root: Path | None = None) -> dict[str, Any]:
    final_root = (root or project_root()).resolve()
    failures: list[str] = []

    missing_files = [
        relative_path
        for relative_path in STAGE_4_2_8_REQUIRED_FILES
        if not (final_root / relative_path).is_file()
    ]
    missing_tests = [
        relative_path
        for relative_path in STAGE_4_2_8_UNIT_TESTS
        if not (final_root / relative_path).is_file()
    ]

    if missing_files:
        failures.append(
            "缺少发布必需文件：" + ", ".join(missing_files)
        )

    if missing_tests:
        failures.append(
            "缺少回归测试文件：" + ", ".join(missing_tests)
        )

    prepared = prepare_production_run_node(
        {
            "request_id": "stage_4_2_8f_verify",
            "run_id": "stage_4_2_8f_verify_run",
            "user_message": "发布契约校验",
            "user_id": "release_verifier",
            "thread_id": "release_verifier_thread",
        }
    )
    runtime_version = str(
        prepared.get("graph_version") or ""
    )

    if runtime_version != STAGE_4_2_8_VERSION:
        failures.append(
            "生产 Graph 版本与发布契约不一致："
            f"runtime={runtime_version!r}, "
            f"contract={STAGE_4_2_8_VERSION!r}"
        )

    error_module = (
        final_root
        / "app/agent_graph/runtime/agent_errors.py"
    )
    obsolete_tool_error_import = False

    if error_module.is_file():
        error_text = error_module.read_text(
            encoding="utf-8"
        )
        obsolete_tool_error_import = (
            "schemas.tool_schema import ToolError"
            in error_text
        )

    if obsolete_tool_error_import:
        failures.append(
            "agent_errors.py 仍依赖不存在的 ToolError 类。"
        )

    return {
        "release_name": STAGE_4_2_8_RELEASE_NAME,
        "version": STAGE_4_2_8_VERSION,
        "runtime_version": runtime_version,
        "capabilities": list(STAGE_4_2_8_CAPABILITIES),
        "required_file_count": len(
            STAGE_4_2_8_REQUIRED_FILES
        ),
        "unit_test_file_count": len(
            STAGE_4_2_8_UNIT_TESTS
        ),
        "missing_files": missing_files,
        "missing_tests": missing_tests,
        "obsolete_tool_error_import": (
            obsolete_tool_error_import
        ),
        "passed": not failures,
        "failures": failures,
    }


def main() -> None:
    result = verify_release()
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )

    if not result["passed"]:
        raise SystemExit(1)

    print(
        "\nStage 4.2.8F release-contract verification passed."
    )


if __name__ == "__main__":
    main()
