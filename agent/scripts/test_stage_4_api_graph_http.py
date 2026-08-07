import json
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def print_json(data: dict) -> None:
    print(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    url = "http://127.0.0.1:8000/api/chat/graph"

    cases = [
        {
            "name": "简单概念题",
            "body": {
                "message": "什么是紧急备用金？",
                "user_id": "api_graph_user_simple",
                "thread_id": "api_graph_thread_simple",
                "request_id": "api-graph-http-simple-001",
            },
        },
        {
            "name": "复杂计算题",
            "body": {
                "message": (
                    "我家年度必要支出是18万元，"
                    "请帮我换算成月度必要支出，"
                    "并计算3到6个月紧急备用金范围。"
                ),
                "user_id": "api_graph_user_complex",
                "thread_id": "api_graph_thread_complex",
                "request_id": "api-graph-http-complex-001",
            },
        },
    ]

    with httpx.Client(
        timeout=120.0,
        trust_env=False,
    ) as client:
        for case in cases:
            print("=" * 80)
            print(case["name"])
            print("=" * 80)

            response = client.post(
                url,
                json=case["body"],
            )

            print("status_code:", response.status_code)
            print("content_type:", response.headers.get("content-type"))

            response.raise_for_status()

            data = response.json()

            debug_data = {
                "request_id": data.get("request_id"),
                "question_capabilities": data.get("question_capabilities"),
                "question_router": data.get("question_router"),
                "question_router_confidence": data.get(
                    "question_router_confidence"
                ),
                "question_router_reason": data.get(
                    "question_router_reason"
                ),
                "question_router_used_fallback": data.get(
                    "question_router_used_fallback"
                ),
                "question_router_matched_rules": data.get(
                    "question_router_matched_rules"
                ),
                "execution_plan": data.get("execution_plan"),
                "finish_reason": data.get("finish_reason"),
                "fallback_used": data.get("fallback_used"),
                "answer": data.get("answer"),
                "executed_tools": data.get("executed_tools"),
                "quality_gate": data.get("quality_gate"),
                "usage_keys": list((data.get("usage") or {}).keys()),
            }

            print_json(debug_data)


if __name__ == "__main__":
    main()
