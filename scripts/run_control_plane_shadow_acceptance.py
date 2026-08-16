from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from app.agent_graph.semantic_route import SemanticRouteDecision
from app.control_plane.enums import (
    Authority, DeliveryStatus, EnforcementStrength, InvocationStatus,
    PermissionLevel, RequirementLevel, RuntimeCapabilityStatus, StrategyStatus,
    SystemHealth, TaskStatus,
)
from app.control_plane.metrics import ControlPlaneMetrics
from app.control_plane.schemas import (
    CONTROL_PLANE_SCHEMA_VERSION, CapabilityAvailability, CapabilityConstraint,
    ConstraintSource, FinalRunStatus, PreliminaryStrategy, ResolvedResourceRef,
    ResolvedResourceScope, RuntimeCapabilitySnapshot, SemanticRequirementContract,
    TaskRequirement,
)
from app.control_plane.shadow import ShadowCapabilityRegistry, ShadowControlPlane
from app.core.config import Settings
from app.llm.deepseek_client import DeepSeekClient
from tests.acceptance.finance_shadow_cases import FINANCE_SHADOW_CASES


DOC_ID = "06b200f7-e17b-5235-84b6-93a331bc9b4a"
DOC_HASH = "7df4ac5a3d57fc3bbd12c97c937fd14bef528e220c770db094315b492d333c47"
CAPABILITIES = (
    "general_explanation", "knowledge_retrieval", "financial_calculation",
    "complex_reasoning", "citation_validation", "web_access",
)


EXTRACTOR_PROMPT = """You are an independent semantic requirement extractor, not a router.
Extract user requirements without choosing providers. Return JSON only:
{"constraints":[{"capability":"...","requirement":"required|optional|not_needed","permission":"allowed|forbidden","scope_ref":null|"uploaded_documents"}],
 "tasks":[{"task_id":"snake_case","description":"...","capabilities":["..."],"depends_on":[],"evidence_tool_names":[],"requires_citations":false}],"confidence":0.0}
Capabilities: general_explanation, knowledge_retrieval, financial_calculation,
complex_reasoning, citation_validation, web_access. Exact numerical financial
calculation requires financial_calculation. Explicit document use requires
knowledge_retrieval; explicit citations require citation_validation. "Do not"
means forbidden; "not necessary" without prohibition means not_needed+allowed.
A prohibition on switching to other knowledge bases is a scope restriction,
not a prohibition on knowledge_retrieval itself. Preserve genuine conflicts as
separate constraints. A request for a "guaranteed profit/highest return"
product is a financial-safety explanation task; do not invent retrieval or
calculation requirements. A multi-source synthesis or comparison that combines
tool results and several evidence topics requires complex_reasoning. "Give the
source/citation" requires citation_validation. Never answer the user."""


EXPECTED_CONTROLS = {
    "FIN-ACCEPT-001": ({"knowledge_retrieval", "citation_validation", "financial_calculation", "complex_reasoning"}, {"web_access"}),
    "FIN-REG-001": (set(), {"knowledge_retrieval", "financial_calculation"}),
    "FIN-REG-002": ({"financial_calculation"}, {"knowledge_retrieval"}),
    "FIN-REG-003": ({"knowledge_retrieval", "citation_validation"}, {"financial_calculation", "web_access"}),
    "FIN-REG-004": ({"financial_calculation"}, set()),
    "FIN-REG-005": ({"knowledge_retrieval", "citation_validation"}, {"web_access"}),
    "FIN-REG-006": (set(), set()),  # expected unresolved same-authority conflict
    "FIN-REG-007": ({"financial_calculation"}, {"web_access"}),
    "FIN-REG-008": ({"knowledge_retrieval"}, set()),
    "FIN-REG-009": ({"knowledge_retrieval"}, set()),
    "FIN-REG-010": ({"financial_calculation"}, set()),
    "FIN-REG-011": ({"knowledge_retrieval"}, set()),
    "FIN-REG-012": ({"general_explanation"}, set()),
}


def _constraint(raw: dict[str, Any], index: int) -> CapabilityConstraint:
    return CapabilityConstraint(
        capability=str(raw["capability"]),
        requirement=RequirementLevel(str(raw.get("requirement", "not_needed"))),
        # Semantic inference may discover work, but it cannot invent a deny.
        # Explicit prohibitions are authoritatively captured by the Floor.
        permission=PermissionLevel.ALLOWED,
        scope_ref=raw.get("scope_ref"),
        source=ConstraintSource(
            constraint_id=f"extractor:{index}:{raw['capability']}",
            authority=Authority.SEMANTIC_EXTRACTOR,
            enforcement_strength=EnforcementStrength.INFERRED,
            rule_id="shadow-semantic-extractor-v1",
        ),
    )


async def _extract(client: DeepSeekClient, request_id: str, message: str) -> SemanticRequirementContract:
    contract = None
    for attempt in range(2):
        try:
            prompt = EXTRACTOR_PROMPT if attempt == 0 else EXTRACTOR_PROMPT + "\nPrevious output failed schema validation. Return only a corrected object matching the schema exactly."
            response = await client.chat(
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": message}],
                thinking_enabled=False, max_completion_tokens=1800,
                response_format={"type": "json_object"},
            )
            raw = json.loads(response["message"]["content"])
            constraints = tuple(_constraint(item, i) for i, item in enumerate(raw.get("constraints", []), 1))
            tasks = tuple(TaskRequirement.model_validate(item) for item in raw.get("tasks", []))
            contract = SemanticRequirementContract(
                request_id=request_id, constraints=constraints, task_requirements=tasks,
                confidence=Decimal(str(raw.get("confidence", 0))),
                invocation_status=InvocationStatus.SUCCESS if attempt == 0 else InvocationStatus.REPAIRED,
            )
            break
        except Exception:
            continue
    if contract is None:
        contract = SemanticRequirementContract(request_id=request_id, invocation_status=InvocationStatus.PROTOCOL_FAILED)
    return contract.model_copy(update={"canonical_hash": contract.calculate_hash()})


def _preliminary(request_id: str, route: dict[str, Any] | None) -> PreliminaryStrategy | None:
    if not route:
        return None
    try:
        decision = SemanticRouteDecision.model_validate(route)
        tasks = tuple(
            TaskRequirement(
                task_id=item.id, description=item.description, required=item.required,
                capabilities=tuple(item.capabilities), depends_on=tuple(item.depends_on),
                evidence_tool_names=tuple(item.evidence_tool_names), requires_citations=item.requires_citations,
            ) for item in decision.task_requirements
        )
        value = PreliminaryStrategy(
            request_id=request_id, orchestration_mode=decision.orchestration_mode,
            proposed_capabilities=tuple(decision.required_capabilities), proposed_tasks=tasks,
            confidence=Decimal(str(decision.confidence)), invocation_status=InvocationStatus.SUCCESS,
        )
        return value.model_copy(update={"canonical_hash": value.calculate_hash()})
    except Exception:
        return None


def _snapshot(run_id: str) -> RuntimeCapabilitySnapshot:
    now = datetime.now(UTC)
    value = RuntimeCapabilitySnapshot(
        run_id=run_id, observed_at_utc=now,
        capabilities=tuple(CapabilityAvailability(
            capability=cap, provider_or_tool=f"local:{cap}", status=RuntimeCapabilityStatus.AVAILABLE,
            checked_at_utc=now,
        ) for cap in CAPABILITIES),
    )
    return value.model_copy(update={"canonical_hash": value.calculate_hash()})


def _scope() -> ResolvedResourceScope:
    value = ResolvedResourceScope(
        scope_id="uploaded_documents", requested_scope_hash="fixed-benchmark-scope",
        resources=(ResolvedResourceRef(tenant_id="personal", knowledge_base_id="kb_finance_basic",
            document_id=DOC_ID, document_version=1, content_hash=DOC_HASH),),
        allowed_source_types=("pdf",), authorization_snapshot_id="local-shadow-acceptance",
        resolved_at_utc=datetime.now(UTC), resolution_status="resolved",
    )
    return value.model_copy(update={"canonical_hash": value.calculate_hash()})


def _prediction(strategy_status: StrategyStatus, legacy: dict[str, Any]) -> FinalRunStatus:
    degraded = tuple(legacy.get("system_health", {}).get("degraded_components", [])) if isinstance(legacy.get("system_health"), dict) else ()
    if strategy_status == StrategyStatus.BLOCKED:
        task, delivery = TaskStatus.BLOCKED, DeliveryStatus.NOT_GENERATED
    elif strategy_status == StrategyStatus.UNFULFILLABLE:
        task, delivery = TaskStatus.FAILED, DeliveryStatus.NOT_GENERATED
    else:
        task, delivery = TaskStatus.COMPLETED, DeliveryStatus.VALIDATED
    return FinalRunStatus(task_status=task, system_health=SystemHealth.DEGRADED if degraded else SystemHealth.HEALTHY,
        delivery_status=delivery, degraded_components=degraded, legacy_overall_status=task.value)


async def _legacy(http: httpx.AsyncClient, case, request_id: str) -> dict[str, Any]:
    payload = {
        "request_id": request_id, "user_message": case.prompt, "user_id": "owner",
        "thread_id": f"shadow-{case.test_id.lower()}", "tenant_id": "personal",
        "knowledge_base_id": "kb_finance_basic", "document_ids": [DOC_ID] if case.document_scoped else [],
        "rag_mode": "auto", "synthesis_llm_provider": "deepseek", "extract_long_memory": False,
        "save_memory": False,
    }
    response = await http.post("/api/chat/graph-v2", json=payload, timeout=300)
    if response.is_error:
        try:
            detail = response.json()
        except Exception:
            detail = {"body": response.text[:1000]}
        return {"status": "blocked", "finish_reason": f"http_{response.status_code}", "http_error": detail}
    return response.json()


async def run(output: Path, *, refresh_shadow: bool = False) -> int:
    settings = Settings(_env_file=Path("agent/.env"))
    metrics = ControlPlaneMetrics("control-plane-shadow-v1", {"control_plane": CONTROL_PLANE_SCHEMA_VERSION})
    shadow = ShadowControlPlane(registry=ShadowCapabilityRegistry())
    llm = DeepSeekClient(settings)
    output.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    if output.exists():
        try:
            reports = list(json.loads(output.read_text(encoding="utf-8")).get("cases", []))
        except Exception:
            reports = []
    completed_ids = {item.get("test_id") for item in reports}
    prior_by_id = {item.get("test_id"): item for item in reports}
    for item in (() if refresh_shadow else reports):
        metrics.observe_request()
        if item.get("semantic_extractor", {}).get("invocation_status") in {"protocol_failed", "service_failed", "timeout"}:
            metrics.increment("extractor_protocol_degraded_rate")
        if "STRATEGY_RECONCILED" in (item.get("shadow_diff", {}).get("reason_codes") or []):
            metrics.increment("strategy_reconcile_rate")
        prior_diff = item.get("shadow_diff") or {}
        for metric_name, field_name in (
            ("required_drop_count", "required_dropped"),
            ("forbidden_execution_count", "forbidden_planned"),
            ("silent_scope_expansion_count", "scope_expansions"),
        ):
            if prior_diff.get(field_name):
                metrics.increment(metric_name, len(prior_diff[field_name]))
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8002") as http:
        for case in FINANCE_SHADOW_CASES:
            if case.test_id in completed_ids and not refresh_shadow:
                continue
            print(f"START {case.test_id}", flush=True)
            request_id = f"shadow-{case.test_id.lower()}-{uuid4()}"
            prior = prior_by_id.get(case.test_id)
            if refresh_shadow and prior:
                legacy = {
                    **prior["legacy_actual_execution"],
                    "semantic_route": prior.get("legacy_route"),
                    "run_id": prior.get("shadow_diff", {}).get("production_run_id"),
                    "runtime_revision": prior.get("shadow_diff", {}).get("production_revision"),
                }
            else:
                legacy = await _legacy(http, case, request_id)
            extractor = await _extract(llm, request_id, case.prompt)
            route = legacy.get("semantic_route") or (legacy.get("route_context") or {}).get("semantic_route")
            preliminary = _preliminary(request_id, route)
            result, diff = shadow.evaluate(
                request_id=request_id, production_run_id=str(legacy.get("run_id", "unknown")),
                production_revision=str(legacy.get("runtime_revision", legacy.get("graph_version", "v3.1"))),
                user_message=case.prompt, extractor_contract=extractor, preliminary_strategy=preliminary,
                runtime_snapshot=_snapshot(f"shadow:{request_id}"), metrics=metrics,
                resolved_scopes=(_scope(),) if case.document_scoped else (),
            )
            strategy_status = "blocked" if diff.forbidden_planned else "ready"
            case_report = {
                "test_id": case.test_id,
                "legacy_route": route,
                "requirement_floor_hash": result.shadow_floor_hash,
                "semantic_extractor": extractor.model_dump(mode="json"),
                "sealed_effective_contract_hash": result.shadow_contract_hash,
                "sealed_effective_contract": result.sealed_contract,
                "shadow_effective_strategy_hash": result.shadow_strategy_hash,
                "shadow_effective_strategy": result.effective_strategy,
                "legacy_actual_execution": {
                    "status": legacy.get("status"), "finish_reason": legacy.get("finish_reason"),
                    "execution_path": legacy.get("execution_path"), "tool_results": legacy.get("tool_results", []),
                    "rag": legacy.get("rag"), "task_status": legacy.get("task_status"),
                    "system_health": legacy.get("system_health"), "delivery_status": legacy.get("delivery_status"),
                },
                "shadow_status_prediction": strategy_status,
                "shadow_diff": asdict(diff),
            }
            if prior:
                reports = [item for item in reports if item.get("test_id") != case.test_id]
            reports.append(case_report)
            reports.sort(key=lambda item: next(i for i, case_item in enumerate(FINANCE_SHADOW_CASES) if case_item.test_id == item["test_id"]))
            partial_report = {"suite": "finance-shadow-13", "cases": reports,
                "slo": metrics.snapshot(), "red_line_violations": metrics.red_line_violations(), "gate": "RUNNING"}
            output.write_text(json.dumps(partial_report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            print(f"DONE {case.test_id}", flush=True)
    await llm.close()
    expected_unsealed = {"FIN-REG-006"}
    extractor_ok = all(
        item["semantic_extractor"]["invocation_status"] in {"success", "repaired"}
        for item in reports
    )
    extractor_fail_closed = all(
        item["semantic_extractor"]["invocation_status"] in {"success", "repaired"}
        or (
            not item["semantic_extractor"]["constraints"]
            and not item["semantic_extractor"]["task_requirements"]
        )
        for item in reports
    )
    sealing_ok = all(
        bool(item["sealed_effective_contract_hash"]) == (item["test_id"] not in expected_unsealed)
        for item in reports
    )
    semantic_checks: dict[str, bool] = {}
    for item in reports:
        expected_required, expected_forbidden = EXPECTED_CONTROLS[item["test_id"]]
        contract = item.get("sealed_effective_contract") or {}
        constraints = contract.get("constraints") or []
        actual_required = {x["capability"] for x in constraints if x["requirement"] == "required"}
        actual_forbidden = {x["capability"] for x in constraints if x["permission"] == "forbidden"}
        if item["test_id"] == "FIN-REG-006":
            semantic_checks[item["test_id"]] = (
                not item["sealed_effective_contract_hash"]
                and "CONTRACT_PERMISSION_CONFLICT" in item["shadow_diff"]["reason_codes"]
            )
        else:
            semantic_checks[item["test_id"]] = (
                expected_required.issubset(actual_required)
                and expected_forbidden.issubset(actual_forbidden)
            )
    semantics_ok = all(semantic_checks.values())
    normal_passed = metrics.acceptance_passed() and extractor_fail_closed and sealing_ok and semantics_ok
    report = {"suite": "finance-shadow-13", "cases": reports, "slo": metrics.snapshot(),
        "red_line_violations": metrics.red_line_violations(),
        "acceptance_checks": {"extractor_ok": extractor_ok, "sealing_ok": sealing_ok,
            "extractor_fail_closed": extractor_fail_closed, "semantics_ok": semantics_ok,
            "per_case_semantics": semantic_checks},
        "gate": "PASS_SHADOW" if normal_passed else "FAIL_CONTRACT_INVARIANT"}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(output), "case_count": len(reports), "gate": report["gate"], "red_lines": report["red_line_violations"]}, ensure_ascii=False))
    return 0 if report["gate"] == "PASS_SHADOW" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/control_plane/shadow_finance_13.json"))
    parser.add_argument("--refresh-shadow", action="store_true")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args.output, refresh_shadow=args.refresh_shadow)))
