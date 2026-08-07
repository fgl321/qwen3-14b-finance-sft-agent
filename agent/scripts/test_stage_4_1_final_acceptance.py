import json
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


API_URL = "http://127.0.0.1:8000/api/chat/graph"


def print_json(data: dict) -> None:
    print(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        )
    )


def assert_equal(
    actual,
    expected,
    message: str,
) -> None:
    if actual != expected:
        raise AssertionError(
            f"{message}\n"
            f"expected: {expected}\n"
            f"actual:   {actual}"
        )


def run_case(
    *,
    client: httpx.Client,
    name: str,
    body: dict,
    expected_capabilities: list[str],
    expected_plan: list[str],
) -> None:
    print("=" * 80)
    print(name)
    print("=" * 80)

    response = client.post(
        API_URL,
        json=body,
    )

    print("status_code:", response.status_code)
    response.raise_for_status()

    data = response.json()

    summary = {
        "request_id": data.get("request_id"),
        "question_capabilities": data.get("question_capabilities"),
        "question_router": data.get("question_router"),
        "question_router_confidence": data.get(
            "question_router_confidence"
        ),
        "question_router_reason": data.get(
            "question_router_reason"
        ),
        "execution_plan": data.get("execution_plan"),
        "finish_reason": data.get("finish_reason"),
        "fallback_used": data.get("fallback_used"),
        "executed_tool_names": [
            item.get("tool_name")
            for item in data.get("executed_tools", [])
        ],
        "answer_preview": str(data.get("answer") or "")[:120],
    }

    print_json(summary)

    assert_equal(
        data.get("question_capabilities"),
        expected_capabilities,
        f"{name} 能力识别不符合预期",
    )

    assert_equal(
        data.get("execution_plan"),
        expected_plan,
        f"{name} 执行计划不符合预期",
    )

    assert data.get("answer"), f"{name} answer 不能为空"

    print(f"{name} 验收通过")


def main() -> None:
    with httpx.Client(
        timeout=120.0,
        trust_env=False,
    ) as client:
        run_case(
            client=client,
            name="验收 1：简单金融概念题",
            body={
                "message": "什么是紧急备用金？",
                "user_id": "acceptance_user_simple",
                "thread_id": "acceptance_thread_simple",
                "request_id": "stage-4-1-acceptance-simple",
            },
            expected_capabilities=[
                "general_explanation",
            ],
            expected_plan=[
                "general_finance_answer",
            ],
        )

        run_case(
            client=client,
            name="验收 2：金融计算题",
            body={
                "message": (
                    "我家年度必要支出是18万元，"
                    "请帮我换算成月度必要支出，"
                    "并计算3到6个月紧急备用金范围。"
                ),
                "user_id": "acceptance_user_calc",
                "thread_id": "acceptance_thread_calc",
                "request_id": "stage-4-1-acceptance-calc",
            },
            expected_capabilities=[
                "financial_calculation",
            ],
            expected_plan=[
                "finance_agent",
            ],
        )

    print("=" * 80)
    print("Stage 4.1 final acceptance passed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
