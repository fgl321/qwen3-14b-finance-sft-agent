from __future__ import annotations

from app.agent_graph.semantic_route import (
    SemanticRouteDecision,
    SemanticRouter,
)
from app.api.routes.chat_graph_v2 import memory_requirement_satisfied
from app.rag.rag_types import SourceAuthorityContract
from app.tools.runtime_registry import build_production_tool_registry
from app.tools.tool_specs import tool_allowed


def test_memory_requirement_satisfied_matrix() -> None:
    assert memory_requirement_satisfied("forbidden", None) is True
    assert memory_requirement_satisfied("not_needed", None) is True
    assert memory_requirement_satisfied("optional", None) is True
    assert memory_requirement_satisfied(
        "required",
        {"status": "succeeded"},
    ) is True
    assert memory_requirement_satisfied(
        "required",
        {"status": "not_observed"},
    ) is False
    assert memory_requirement_satisfied("unknown", None) is False


def test_normalize_memory_constraint_and_source_authority() -> None:
    payload = {
        "orchestration_mode": "direct",
        "required_capabilities": ["complex_reasoning"],
        "task_requirements": [
            {
                "id": "answer",
                "description": "answer",
                "capabilities": ["complex_reasoning"],
                "task_kind": "synthesis",
            }
        ],
        "capability_constraints": {
            "memory_read": "forbidden",
            "web_search": "forbidden",
        },
        "confidence": 0.9,
        "reason_summary": "test",
    }
    normalized = SemanticRouter._normalize_payload(payload)
    route = SemanticRouteDecision.model_validate(normalized)
    assert route.memory_constraint == "forbidden"
    assert route.source_authority.memory == "forbidden"
    assert route.source_authority.web == "forbidden"


def test_source_authority_defaults_web_forbidden() -> None:
    contract = SourceAuthorityContract()
    assert contract.web == "forbidden"
    assert contract.memory == "allowed"


def test_tool_allowed_gate() -> None:
    math_only = SourceAuthorityContract(
        current_user_facts="allowed",
        selected_documents="allowed",
        deterministic_derivation="allowed",
        domain_heuristics="forbidden",
        web="forbidden",
    )
    assert tool_allowed("pure_math", math_only) is True
    assert tool_allowed("user_fact_transform", math_only) is True
    assert tool_allowed("domain_heuristic", math_only) is False
    assert tool_allowed("external_data", math_only) is False
    assert tool_allowed("unknown_class", math_only) is False

    heuristic_ok = SourceAuthorityContract(domain_heuristics="allowed")
    assert tool_allowed("domain_heuristic", heuristic_ok) is True

    no_math = SourceAuthorityContract(
        deterministic_derivation="forbidden"
    )
    assert tool_allowed("pure_math", no_math) is False


def test_tool_registry_source_classification() -> None:
    registry = build_production_tool_registry()
    yearly = registry.require("yearly_expense_to_monthly")
    emergency = registry.require("emergency_fund_range")
    assert yearly.source_class == "pure_math"
    assert emergency.source_class == "domain_heuristic"
