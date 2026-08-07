from __future__ import annotations


STAGE_4_2_8_VERSION = "stage_4_2_8f"
STAGE_4_2_8_RELEASE_NAME = (
    "Stage 4.2.8 Agent Loop Reliability"
)

STAGE_4_2_8_CAPABILITIES: tuple[str, ...] = (
    "execution_policy",
    "successful_tool_result_reuse",
    "no_progress_loop_detection",
    "request_idempotency",
    "unified_error_model",
    "release_acceptance_contract",
)

STAGE_4_2_8_UNIT_TESTS: tuple[str, ...] = (
    "tests/test_stage_4_2_7b_planner_consistency.py",
    "tests/test_stage_4_2_7b_tool_registry_filters.py",
    "tests/test_stage_4_2_7b_tool_executor_filters.py",
    "tests/test_stage_4_2_7b_output_guard_evidence.py",
    "tests/test_stage_4_2_8_execution_policy.py",
    "tests/test_stage_4_2_8b_successful_tool_reuse.py",
    "tests/test_stage_4_2_8c_no_progress_detection.py",
    "tests/test_stage_4_2_8d_request_idempotency.py",
    "tests/test_stage_4_2_8e_unified_error_model.py",
    "tests/test_stage_4_2_8f_release_contract.py",
)

STAGE_4_2_8_REQUIRED_FILES: tuple[str, ...] = (
    "app/agent_graph/agent_loop.py",
    "app/agent_graph/production_nodes.py",
    "app/agent_graph/production_service.py",
    "app/agent_graph/release_contract.py",
    "app/agent_graph/runtime/agent_errors.py",
    "app/agent_graph/runtime/agent_limits.py",
    "app/agent_graph/runtime/request_idempotency.py",
    "app/agent_graph/runtime_nodes/tool_executor_node.py",
    "app/agent_graph/schemas/error_schema.py",
    "app/agent_graph/schemas/loop_schema.py",
    "app/api/routes/chat_graph_v2.py",
    "app/tools/tool_executor.py",
    "scripts/run_stage_4_2_8_acceptance.py",
    "scripts/test_stage_4_2_8_final_http.py",
    "scripts/verify_stage_4_2_8_release.py",
)
