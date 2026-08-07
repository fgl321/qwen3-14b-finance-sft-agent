from __future__ import annotations

import asyncio
import json
import os
from typing import Any
from uuid import uuid4

import httpx

from app.agent_graph.release_contract import (
    STAGE_4_2_8_VERSION,
)


os.environ.setdefault(
    "NO_PROXY",
    "127.0.0.1,localhost,::1",
)
os.environ.setdefault(
    "no_proxy",
    "127.0.0.1,localhost,::1",
)


EXPECTED_TOOL_NAMES = [
    "yearly_expense_to_monthly",
    "emergency_fund_range",
]


def _string_value(
    payload: dict[str, Any],
    key: str,
) -> str:
    value = payload.get(key)

    if value is None:
        return ""

    return str(value).replace(",", "").strip()


def _require_mapping(
    value: Any,
    *,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AssertionError(
            f"{field_name} 必须是对象，实际为：{type(value).__name__}"
        )

    return value


def _require_list(
    value: Any,
    *,
    field_name: str,
) -> list[Any]:
    if not isinstance(value, list):
        raise AssertionError(
            f"{field_name} 必须是列表，实际为：{type(value).__name__}"
        )

    return value


def _assert_health(
    payload: dict[str, Any],
) -> None:
    if payload.get("status") != "ok":
        raise AssertionError(
            f"生产 Graph 健康检查失败：{payload}"
        )

    if payload.get("graph_service_ready") is not True:
        raise AssertionError(
            "graph_service_ready 不是 true。"
        )

    if payload.get("graph_runtime_ready") is not True:
        raise AssertionError(
            "graph_runtime_ready 不是 true。"
        )

    if payload.get("checkpointer") != "postgresql":
        raise AssertionError(
            "生产主图没有使用 PostgreSQL Checkpointer。"
        )


def _assert_real_tool_chain(
    result: dict[str, Any],
) -> set[str]:
    loop_result = _require_mapping(
        result.get("agent_loop_result"),
        field_name="agent_loop_result",
    )

    if loop_result.get("status") != "completed":
        raise AssertionError(
            "Agent Loop 未完成："
            f"{loop_result.get('status')}"
        )

    if loop_result.get("finish_reason") != "planner_finished":
        raise AssertionError(
            "Agent Loop 不是由合法 planner_finish 结束："
            f"{loop_result.get('finish_reason')}"
        )

    if loop_result.get("total_tool_calls") != 2:
        raise AssertionError(
            "本用例必须真实执行两个工具，实际 total_tool_calls="
            f"{loop_result.get('total_tool_calls')}"
        )

    reused_tool_calls = _require_list(
        loop_result.get("reused_tool_calls"),
        field_name="agent_loop_result.reused_tool_calls",
    )

    if reused_tool_calls:
        raise AssertionError(
            "正常工具链不应出现重复结果复用，实际为："
            f"{reused_tool_calls}"
        )

    if loop_result.get("reused_tool_call_count") != 0:
        raise AssertionError(
            "正常工具链的 reused_tool_call_count 应为0，"
            f"实际为：{loop_result.get('reused_tool_call_count')}"
        )

    no_progress_events = _require_list(
        loop_result.get("no_progress_events"),
        field_name="agent_loop_result.no_progress_events",
    )

    if no_progress_events:
        raise AssertionError(
            "正常工具链不应出现无进展事件，实际为："
            f"{no_progress_events}"
        )

    if loop_result.get("no_progress_round_count") != 0:
        raise AssertionError(
            "正常工具链的 no_progress_round_count 应为0，"
            f"实际为：{loop_result.get('no_progress_round_count')}"
        )

    if loop_result.get("last_progress_round") != 2:
        raise AssertionError(
            "正常两工具链的 last_progress_round 应为2，"
            f"实际为：{loop_result.get('last_progress_round')}"
        )

    raw_tool_results = _require_list(
        loop_result.get("tool_results"),
        field_name="agent_loop_result.tool_results",
    )

    if len(raw_tool_results) != 2:
        raise AssertionError(
            "本用例必须产生两个工具结果，实际数量="
            f"{len(raw_tool_results)}"
        )

    tool_results = [
        _require_mapping(
            item,
            field_name=f"tool_results[{index}]",
        )
        for index, item in enumerate(raw_tool_results)
    ]

    actual_tool_names = [
        str(item.get("tool_name") or "")
        for item in tool_results
    ]

    if actual_tool_names != EXPECTED_TOOL_NAMES:
        raise AssertionError(
            "工具链顺序不正确。\n"
            f"期望：{EXPECTED_TOOL_NAMES}\n"
            f"实际：{actual_tool_names}"
        )

    for item in tool_results:
        if item.get("success") is not True:
            raise AssertionError(
                "工具执行失败，不能进入成功回答：\n"
                f"{json.dumps(item, ensure_ascii=False, indent=2)}"
            )

    first_output = _require_mapping(
        tool_results[0].get("output"),
        field_name="yearly_expense_to_monthly.output",
    )
    second_output = _require_mapping(
        tool_results[1].get("output"),
        field_name="emergency_fund_range.output",
    )

    monthly_expense = _string_value(
        first_output,
        "monthly_necessary_expense",
    )
    min_amount = _string_value(
        second_output,
        "min_amount",
    )
    max_amount = _string_value(
        second_output,
        "max_amount",
    )

    if monthly_expense not in {"15000", "15000.0", "15000.00"}:
        raise AssertionError(
            "月度必要支出工具结果不正确："
            f"{monthly_expense!r}"
        )

    if min_amount not in {"45000", "45000.0", "45000.00"}:
        raise AssertionError(
            "紧急备用金下限工具结果不正确："
            f"{min_amount!r}"
        )

    if max_amount not in {"90000", "90000.0", "90000.00"}:
        raise AssertionError(
            "紧急备用金上限工具结果不正确："
            f"{max_amount!r}"
        )

    successful_ids = {
        str(item.get("tool_call_id") or "").strip()
        for item in tool_results
    }

    if "" in successful_ids or len(successful_ids) != 2:
        raise AssertionError(
            "成功工具结果缺少有效且唯一的 tool_call_id。"
        )

    planner_invocations = _require_list(
        loop_result.get("planner_invocations"),
        field_name="agent_loop_result.planner_invocations",
    )

    raw_function_names: list[str] = []

    for invocation in planner_invocations:
        invocation_payload = _require_mapping(
            invocation,
            field_name="planner_invocation",
        )
        names = invocation_payload.get(
            "raw_tool_call_names"
        ) or []

        if not isinstance(names, list):
            raise AssertionError(
                "raw_tool_call_names 必须是列表。"
            )

        raw_function_names.extend(
            str(name)
            for name in names
        )

    expected_function_order = [
        "yearly_expense_to_monthly",
        "emergency_fund_range",
        "planner_finish",
    ]

    if raw_function_names != expected_function_order:
        raise AssertionError(
            "Planner 原始函数调用顺序不正确。\n"
            f"期望：{expected_function_order}\n"
            f"实际：{raw_function_names}"
        )

    return successful_ids


def _assert_final_response_evidence(
    result: dict[str, Any],
    *,
    successful_tool_call_ids: set[str],
) -> None:
    final_response = _require_mapping(
        result.get("final_response_result"),
        field_name="final_response_result",
    )

    synthesis = _require_mapping(
        final_response.get("synthesis"),
        field_name="final_response_result.synthesis",
    )
    guard = _require_mapping(
        final_response.get("guard"),
        field_name="final_response_result.guard",
    )

    used_tool_call_ids_raw = _require_list(
        synthesis.get("used_tool_call_ids"),
        field_name="synthesis.used_tool_call_ids",
    )
    used_tool_call_ids = {
        str(item).strip()
        for item in used_tool_call_ids_raw
    }

    if not used_tool_call_ids:
        raise AssertionError(
            "最终回答没有声明任何 used_tool_call_ids，"
            "无法证明金额来自真实工具结果。"
        )

    invalid_ids = (
        used_tool_call_ids
        - successful_tool_call_ids
    )

    if invalid_ids:
        raise AssertionError(
            "最终回答引用了不存在或失败的工具调用 ID："
            f"{sorted(invalid_ids)}"
        )

    if not successful_tool_call_ids.issubset(
        used_tool_call_ids
    ):
        raise AssertionError(
            "本用例最终回答必须使用两个工具结果。\n"
            "成功工具 ID："
            f"{sorted(successful_tool_call_ids)}\n"
            "回答使用 ID："
            f"{sorted(used_tool_call_ids)}"
        )

    if guard.get("verdict") != "pass":
        raise AssertionError(
            "Output Guard 没有通过："
            f"{guard.get('verdict')}"
        )

    rewrite_instructions = guard.get(
        "rewrite_instructions"
    )

    if rewrite_instructions is not None:
        raise AssertionError(
            "guard.rewrite_instructions 应为真正的 null/None，"
            f"实际为：{rewrite_instructions!r}"
        )


def _assert_answer_amounts(
    result: dict[str, Any],
) -> None:
    answer = str(
        result.get("final_answer") or ""
    )

    lower_amount_found = any(
        item in answer
        for item in (
            "4.5万元",
            "4.5万",
            "45000",
            "45,000",
        )
    )
    upper_amount_found = any(
        item in answer
        for item in (
            "9万元",
            "9万",
            "90000",
            "90,000",
        )
    )

    if not lower_amount_found:
        raise AssertionError(
            "最终回答缺少备用金下限4.5万元。"
        )

    if not upper_amount_found:
        raise AssertionError(
            "最终回答缺少备用金上限9万元。"
        )


async def main() -> None:
    thread_id = (
        "stage_4_2_7b_postgres_"
        f"{uuid4().hex}"
    )

    request_id = f"stage_4_2_8f_request_{uuid4().hex}"

    payload = {
        "request_id": request_id,
        "user_message": (
            "我的家庭年度必要支出是18万元，"
            "请计算3到6个月的紧急备用金。"
        ),
        "user_id": "production_user_001",
        "thread_id": thread_id,
        "tenant_id": "default",
        "route_context": {
            "complexity": "medium",
            "risk_level": "low",
        },
        "allowed_tool_groups": [
            "financial_calculation",
        ],
        "execution_policy": "require_tool",
        "remaining_tool_calls": 12,
        "allow_side_effects": False,
    }

    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8000",
        timeout=180.0,
        trust_env=False,
    ) as client:
        health_response = await client.get(
            "/health/production-graph"
        )
        health_response.raise_for_status()
        health_payload = health_response.json()

        print("========== health ==========")
        print(
            json.dumps(
                health_payload,
                ensure_ascii=False,
                indent=2,
            )
        )

        _assert_health(health_payload)

        response = await client.post(
            "/api/chat/graph-v2",
            json=payload,
        )

        print("\n========== status_code ==========")
        print(response.status_code)

        print("\n========== raw_response ==========")
        print(response.text)

        response.raise_for_status()
        result = response.json()

        print("\n========== parsed_result ==========")
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )

        if result.get("user_message") != payload["user_message"]:
            raise AssertionError(
                "服务端收到的 user_message 与发送内容不一致。"
            )

        if result.get("execution_policy") != "require_tool":
            raise AssertionError(
                "本专项链式工具测试必须使用 require_tool 策略。"
            )

        if result.get("graph_version") != STAGE_4_2_8_VERSION:
            raise AssertionError(
                "生产主图版本不符合最终发布契约："
                f"{result.get('graph_version')}"
            )

        if result.get("status") != "completed":
            raise AssertionError(
                "生产主图没有完成。\n"
                f"status={result.get('status')}\n"
                f"finish_reason={result.get('finish_reason')}\n"
                f"error={result.get('error')}"
            )

        if result.get("finish_reason") != "output_guard_passed":
            raise AssertionError(
                "最终回答没有通过 Output Guard："
                f"{result.get('finish_reason')}"
            )

        idempotency = result.get("idempotency") or {}

        if result.get("idempotency_replayed") is not False:
            raise AssertionError(
                "首次请求不应标记为幂等重放。"
            )

        if idempotency.get("request_id") != request_id:
            raise AssertionError(
                "首次响应的幂等 request_id 不正确。"
            )

        if idempotency.get("replayed") is not False:
            raise AssertionError(
                "首次响应 idempotency.replayed 应为 false。"
            )

        successful_tool_call_ids = (
            _assert_real_tool_chain(result)
        )
        _assert_final_response_evidence(
            result,
            successful_tool_call_ids=(
                successful_tool_call_ids
            ),
        )
        _assert_answer_amounts(result)

        replay_response = await client.post(
            "/api/chat/graph-v2",
            json=payload,
        )

        print(
            "\n========== replay_status_code =========="
        )
        print(replay_response.status_code)

        replay_response.raise_for_status()
        replay_result = replay_response.json()

        print(
            "\n========== replay_result =========="
        )
        print(
            json.dumps(
                replay_result,
                ensure_ascii=False,
                indent=2,
            )
        )

        if replay_result.get("idempotency_replayed") is not True:
            raise AssertionError(
                "第二次相同请求没有命中幂等重放。"
            )

        replay_idempotency = (
            replay_result.get("idempotency") or {}
        )

        if replay_idempotency.get("replayed") is not True:
            raise AssertionError(
                "第二次响应 idempotency.replayed 应为 true。"
            )

        if replay_result.get("run_id") != result.get("run_id"):
            raise AssertionError(
                "幂等重放没有复用首次执行的 run_id。"
            )

        if (
            replay_result.get("final_answer")
            != result.get("final_answer")
        ):
            raise AssertionError(
                "幂等重放的最终回答与首次结果不一致。"
            )

        replay_loop = (
            replay_result.get("agent_loop_result") or {}
        )
        original_loop = (
            result.get("agent_loop_result") or {}
        )

        if (
            replay_loop.get("tool_results")
            != original_loop.get("tool_results")
        ):
            raise AssertionError(
                "幂等重放重新生成或修改了工具结果。"
            )

        conflict_payload = dict(payload)
        conflict_payload["user_message"] = (
            "使用相同 request_id 提交另一条不同问题。"
        )

        conflict_response = await client.post(
            "/api/chat/graph-v2",
            json=conflict_payload,
        )

        print(
            "\n========== conflict_status_code =========="
        )
        print(conflict_response.status_code)

        if conflict_response.status_code != 409:
            raise AssertionError(
                "同一 request_id 的不同请求内容应返回 HTTP 409，"
                f"实际为 {conflict_response.status_code}。"
            )

        conflict_result = conflict_response.json()
        conflict_detail = (
            conflict_result.get("detail") or {}
        )

        print(
            "\n========== conflict_result =========="
        )
        print(
            json.dumps(
                conflict_result,
                ensure_ascii=False,
                indent=2,
            )
        )

        if conflict_detail.get("code") != (
            "REQUEST_ID_CONFLICT"
        ):
            raise AssertionError(
                "幂等冲突没有返回统一错误码："
                f"{conflict_detail}"
            )

        if conflict_detail.get("category") != "conflict":
            raise AssertionError(
                "幂等冲突的错误分类不正确。"
            )

        if conflict_detail.get("stage") != "idempotency":
            raise AssertionError(
                "幂等冲突的错误阶段不正确。"
            )

        if conflict_detail.get("http_status") != 409:
            raise AssertionError(
                "统一错误模型中的 http_status 不正确。"
            )

        if conflict_detail.get("retryable") is not False:
            raise AssertionError(
                "幂等冲突不应被标记为可重试。"
            )

        error_id = str(
            conflict_detail.get("error_id") or ""
        )

        if not error_id.startswith("err_"):
            raise AssertionError(
                "统一错误模型缺少合法 error_id。"
            )

        print(
            "\nStage 4.2.8F final-release "
            "HTTP correctness test passed."
        )


if __name__ == "__main__":
    asyncio.run(main())
