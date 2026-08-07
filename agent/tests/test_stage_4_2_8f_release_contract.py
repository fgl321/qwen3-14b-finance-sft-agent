from __future__ import annotations

from pathlib import Path

from app.agent_graph.production_nodes import (
    prepare_production_run_node,
)
from app.agent_graph.release_contract import (
    STAGE_4_2_8_CAPABILITIES,
    STAGE_4_2_8_REQUIRED_FILES,
    STAGE_4_2_8_UNIT_TESTS,
    STAGE_4_2_8_VERSION,
)
from scripts.verify_stage_4_2_8_release import (
    verify_release,
)


def test_final_version_is_centralized() -> None:
    result = prepare_production_run_node(
        {
            "request_id": "request_001",
            "run_id": "run_001",
            "user_message": "问题",
            "user_id": "user_001",
            "thread_id": "thread_001",
        }
    )

    assert STAGE_4_2_8_VERSION == "stage_4_2_8f"
    assert result["graph_version"] == STAGE_4_2_8_VERSION


def test_release_contract_lists_all_reliability_layers() -> None:
    assert set(STAGE_4_2_8_CAPABILITIES) == {
        "execution_policy",
        "successful_tool_result_reuse",
        "no_progress_loop_detection",
        "request_idempotency",
        "unified_error_model",
        "release_acceptance_contract",
    }


def test_release_contract_includes_final_test() -> None:
    assert (
        "tests/test_stage_4_2_8f_release_contract.py"
        in STAGE_4_2_8_UNIT_TESTS
    )
    assert (
        "app/agent_graph/release_contract.py"
        in STAGE_4_2_8_REQUIRED_FILES
    )


def test_verify_release_passes_for_complete_fixture(
    tmp_path: Path,
) -> None:
    for relative_path in (
        *STAGE_4_2_8_REQUIRED_FILES,
        *STAGE_4_2_8_UNIT_TESTS,
    ):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")

    agent_errors = (
        tmp_path
        / "app/agent_graph/runtime/agent_errors.py"
    )
    agent_errors.write_text(
        "from typing import Protocol\n",
        encoding="utf-8",
    )

    result = verify_release(tmp_path)

    assert result["passed"] is True
    assert result["missing_files"] == []
    assert result["missing_tests"] == []
    assert result["runtime_version"] == (
        STAGE_4_2_8_VERSION
    )


def test_verify_release_detects_missing_files(
    tmp_path: Path,
) -> None:
    result = verify_release(tmp_path)

    assert result["passed"] is False
    assert result["missing_files"]
    assert result["missing_tests"]


def test_verify_release_rejects_obsolete_tool_error_import(
    tmp_path: Path,
) -> None:
    for relative_path in (
        *STAGE_4_2_8_REQUIRED_FILES,
        *STAGE_4_2_8_UNIT_TESTS,
    ):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")

    agent_errors = (
        tmp_path
        / "app/agent_graph/runtime/agent_errors.py"
    )
    agent_errors.write_text(
        "from app.agent_graph.schemas.tool_schema import ToolError\n",
        encoding="utf-8",
    )

    result = verify_release(tmp_path)

    assert result["passed"] is False
    assert result["obsolete_tool_error_import"] is True
