from __future__ import annotations

import json
import sys
import urllib.request


BASE = "http://127.0.0.1:8002"


def post(request_id: str, user_message: str, thread_id: str) -> dict:
    payload = {
        "request_id": request_id,
        "user_message": user_message,
        "user_id": "e2e-core-user",
        "thread_id": thread_id,
        "tenant_id": "personal",
        "rag_mode": "off",
        "synthesis_llm_provider": "deepseek",
        "use_short_memory": True,
        "use_long_memory": False,
        "extract_long_memory": False,
        "save_memory": True,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        BASE + "/api/chat/graph-v2",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=240) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"http_error": exc.code, "detail": body[:2000]}


def after_state(r: dict) -> dict:
    return (r.get("commit_observability") or {}).get("after") or {}


def active_facts(r: dict) -> list[dict]:
    task = after_state(r).get("active_task") or {}
    return list(task.get("canonical_facts") or [])


def fact_by_field(r: dict, field: str) -> dict | None:
    for fact in active_facts(r):
        if fact.get("field") == field:
            return fact
    return None


def superseded_facts(r: dict) -> list[dict]:
    task = after_state(r).get("active_task") or {}
    return list(task.get("superseded_facts") or [])


def find_calc(r: dict, ref: str) -> dict | None:
    for result in after_state(r).get("recent_results") or []:
        for calc in result.get("calculations") or []:
            if calc.get("handle") == ref:
                return calc
    return None


def show(label: str, r: dict) -> None:
    print(f"===== {label} =====")
    print("status:", r.get("status"), "| overall:", r.get("overall_status"))
    print(
        "final_answer:",
        str(r.get("final_answer") or "")[:180],
    )
    print("turn_commit_receipt:", r.get("turn_commit_receipt"))
    print(
        "mutation_receipt:",
        json.dumps(r.get("mutation_receipt"), ensure_ascii=False)[:500],
    )
    calc = (r.get("capability_outcomes") or {}).get(
        "financial_calculation"
    ) or {}
    print(
        "financial_calculation:",
        json.dumps(calc, ensure_ascii=False)[:500],
    )
    guard = r.get("output_guard_result") or {}
    print("guard_verdict:", guard.get("verdict"))
    print("guard_risk_flags:", guard.get("risk_flags"))
    print("planner_invocation_count:", r.get("planner_invocation_count"))
    print("execution_round:", r.get("execution_round"))
    sr = r.get("semantic_route") or {}
    print(
        "semantic_route:",
        {
            "relation": sr.get("conversation_relation"),
            "state_update_only": sr.get("state_update_only"),
            "orchestration_mode": sr.get("orchestration_mode"),
            "fact_updates": sr.get("fact_updates"),
            "extracted_facts": sr.get("extracted_facts"),
        },
    )
    print(
        "active_facts:",
        json.dumps(active_facts(r), ensure_ascii=False),
    )
    print(
        "superseded_facts:",
        json.dumps(superseded_facts(r), ensure_ascii=False),
    )
    if "http_error" in r:
        print("http_error:", r["http_error"], r.get("detail"))


def main() -> int:
    # Case 0: greeting must NOT create a Task / business mutation / ACK.
    greet = post(
        "e2e-greet-1",
        "你好",
        "e2e-greet-20260816-2035",
    )
    show("GREETING 你好", greet)
    if greet.get("http_error"):
        return 1
    admission0 = greet.get("task_admission") or {}
    answer0 = str(greet.get("final_answer") or "")
    conversational_no_task = bool(
        admission0.get("kind") == "conversational"
        and admission0.get("admitted") is False
        and after_state(greet).get("active_task") is None
    )
    mutation0 = greet.get("mutation_receipt") or {}
    greet_ok = bool(
        (
            conversational_no_task
            or admission0.get("admitted") is True
        )
        and mutation0.get("applied_fact_updates") == []
        and mutation0.get("applied_constraint_updates") == []
        and "已更新" not in answer0
        and "已保存" not in answer0
        and "无事实或约束变更" not in answer0
        and greet.get("planner_invocation_count") == 0
        and greet.get("overall_status") == "completed"
    )
    print("GREETING_OK:", greet_ok)
    print("GREETING_CONVERSATIONAL_NO_TASK:", conversational_no_task)
    print("GREETING_TASK_ADMISSION:", admission0)

    thread = "e2e-core-20260816-2035"
    results: list[dict] = []

    t1 = post(
        "e2e-core-1",
        "记录：我有90万现金，首付款20万。",
        thread,
    )
    show("T1 记录 90万/20万", t1)
    results.append(t1)
    if t1.get("http_error"):
        return 1

    t2 = post("e2e-core-2", "支付首付后还剩多少钱？", thread)
    show("T2 计算 70万", t2)
    results.append(t2)
    if t2.get("http_error"):
        return 1

    t3 = post("e2e-core-3", "这个结果是怎么算出来的？", thread)
    show("T3 解释来源", t3)
    results.append(t3)
    if t3.get("http_error"):
        return 1

    t4 = post(
        "e2e-core-4",
        "把首付款从20万改成25万，其他条件保持不变，只保存。",
        thread,
    )
    show("T4 改 25万", t4)
    results.append(t4)
    if t4.get("http_error"):
        return 1

    t5 = post("e2e-core-5", "现在支付首付后还剩多少钱？", thread)
    show("T5 计算 65万", t5)
    results.append(t5)
    if t5.get("http_error"):
        return 1

    # --- T1: facts active, no superseded, scope task ---
    cash1 = fact_by_field(t1, "cash")
    dp1 = fact_by_field(t1, "down_payment")
    t1_ok = bool(
        cash1
        and dp1
        and cash1.get("value") == 900000
        and dp1.get("value") == 200000
        and cash1.get("scope") == "task"
        and dp1.get("scope") == "task"
        and superseded_facts(t1) == []
    )
    print("T1_FACTS_OK:", t1_ok)

    # --- T2: verified CALC 700000 + satisfied + Planner=0 ---
    calc2 = (t2.get("capability_outcomes") or {}).get(
        "financial_calculation"
    ) or {}
    refs2 = list(calc2.get("result_refs") or [])
    calc_artifact2 = find_calc(t2, refs2[0]) if refs2 else None
    t2_ok = bool(
        calc2.get("status") == "satisfied"
        and refs2
        and calc_artifact2
        and calc_artifact2.get("verification_status") == "verified"
        and calc_artifact2.get("output") == 700000.0
        and t2.get("planner_invocation_count") == 0
        and t2.get("overall_status") == "completed"
    )
    print("T2_CALC_VERIFIED:", json.dumps(calc_artifact2, ensure_ascii=False))
    print("T2_OK:", t2_ok)

    # --- T3: explanation references prior artifact, guard pass ---
    guard3 = t3.get("output_guard_result") or {}
    synthesis3 = (
        (t3.get("final_response_result") or {}).get("synthesis") or {}
    )
    answer3 = str(t3.get("final_answer") or "")
    calc3 = (t3.get("capability_outcomes") or {}).get(
        "financial_calculation"
    ) or {}
    t3_ok = bool(
        guard3.get("verdict") == "pass"
        and t3.get("overall_status") == "completed"
        and ("70万" in answer3 or "700000" in answer3)
        and calc3.get("status") == "satisfied"
        and calc3.get("result_refs")
    )
    print(
        "T3_RESULT_ARTIFACT_REFS:",
        synthesis3.get("used_result_artifact_refs"),
    )
    print("T3_FINANCIAL_CALC:", json.dumps(calc3, ensure_ascii=False))
    print("T3_OK:", t3_ok)

    # --- T4: replace 20->25, scope preserved task, old superseded ---
    dp_active4 = fact_by_field(t4, "down_payment")
    dp_superseded4 = [
        f for f in superseded_facts(t4) if f.get("field") == "down_payment"
    ]
    t4_ok = bool(
        dp_active4
        and dp_active4.get("value") == 250000
        and dp_active4.get("scope") == "task"
        and dp_superseded4
        and dp_superseded4[0].get("value") == 200000
    )
    print("T4_SCOPE_PRESERVED:", t4_ok)

    # --- T5: verified CALC 650000 + satisfied + guard pass ---
    calc5 = (t5.get("capability_outcomes") or {}).get(
        "financial_calculation"
    ) or {}
    refs5 = list(calc5.get("result_refs") or [])
    calc_artifact5 = find_calc(t5, refs5[0]) if refs5 else None
    if calc_artifact5 is None:
        print(
            "T5_RECENT_RESULTS:",
            json.dumps(
                after_state(t5).get("recent_results"),
                ensure_ascii=False,
            )[:2000],
        )
    guard5 = t5.get("output_guard_result") or {}
    task5 = (after_state(t5).get("active_task") or {})
    t5_ok = bool(
        calc5.get("status") == "satisfied"
        and refs5
        and calc_artifact5
        and calc_artifact5.get("verification_status") == "verified"
        and calc_artifact5.get("output") == 650000.0
        and guard5.get("verdict") == "pass"
        and t5.get("overall_status") == "completed"
        and t5.get("planner_invocation_count") == 0
        and t5.get("execution_status") == "success"
        and t5.get("fulfillment_status") == "fulfilled"
        and t5.get("delivery_status") == "completed"
        and task5.get("status") != "awaiting_information"
        and bool(t5.get("committed_result_refs"))
    )
    print("T5_CALC_VERIFIED:", json.dumps(calc_artifact5, ensure_ascii=False))
    print("T5_EXECUTION_STATUS:", t5.get("execution_status"))
    print("T5_FULFILLMENT_STATUS:", t5.get("fulfillment_status"))
    print("T5_DELIVERY_STATUS:", t5.get("delivery_status"))
    print("T5_TASK_STATUS:", task5.get("status"))
    print("T5_COMMITTED_RESULT_REFS:", t5.get("committed_result_refs"))
    print("T5_OK:", t5_ok)

    versions = [
        (r.get("turn_commit_receipt") or {}).get("after_version")
        for r in results
    ]
    print("VERSION_CHAIN:", versions)
    print("VERSION_CHAIN_OK:", versions == [1, 2, 3, 4, 5])
    print(
        "CORE_CHAIN_PASS:",
        t1_ok and t2_ok and t3_ok and t4_ok and t5_ok
        and versions == [1, 2, 3, 4, 5],
    )
    return (
        0
        if (
            greet_ok
            and t1_ok
            and t2_ok
            and t3_ok
            and t4_ok
            and t5_ok
        )
        else 2
    )


if __name__ == "__main__":
    sys.exit(main())
