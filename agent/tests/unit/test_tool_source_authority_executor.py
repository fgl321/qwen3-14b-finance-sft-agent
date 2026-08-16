from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from app.agent_graph.schemas.planner_schema import ToolCallRequest
from app.rag.rag_types import SourceAuthorityContract
from app.tools.runtime_registry import (
    ToolRegistry,
    build_production_tool_registry,
)
from app.tools.tool_executor import (
    ProductionToolExecutor,
    ToolExecutionContext,
    normalize_source_authority,
    source_authority_from_route_context,
)
from app.tools.tool_specs import ToolSpec


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _context(
    *,
    source_authority: Any = None,
) -> ToolExecutionContext:
    return ToolExecutionContext(
        request_id="request_test",
        run_id="run_test",
        tenant_id="default",
        user_id="user_test",
        role="user",
        source_authority=source_authority,
    )


def _math_call() -> ToolCallRequest:
    return ToolCallRequest(
        tool_call_id="call_math",
        tool_name="yearly_expense_to_monthly",
        arguments={"yearly_necessary_expense": 180000},
    )


def _heuristic_call() -> ToolCallRequest:
    return ToolCallRequest(
        tool_call_id="call_heuristic",
        tool_name="emergency_fund_range",
        arguments={
            "monthly_necessary_expense": 15000,
            "min_months": 3,
            "max_months": 6,
        },
    )


@pytest.mark.anyio
async def test_source_authority_gate_allows_pure_math() -> None:
    registry = build_production_tool_registry()
    executor = ProductionToolExecutor(registry=registry)
    authority = SourceAuthorityContract(
        deterministic_derivation="allowed",
        domain_heuristics="forbidden",
    )
    outcome = await executor.execute_one(
        _math_call(),
        context=_context(source_authority=authority),
    )
    assert outcome.result.success is True
    assert outcome.trace.status == "succeeded"
    assert outcome.result.output["monthly_necessary_expense"] == "15000.00"


@pytest.mark.anyio
async def test_source_authority_gate_denies_domain_heuristic() -> None:
    registry = build_production_tool_registry()
    executor = ProductionToolExecutor(registry=registry)
    authority = SourceAuthorityContract(
        deterministic_derivation="allowed",
        domain_heuristics="forbidden",
    )
    outcome = await executor.execute_one(
        _heuristic_call(),
        context=_context(source_authority=authority),
    )
    assert outcome.result.success is False
    assert outcome.result.error is not None
    assert outcome.result.error.code == "SOURCE_AUTHORITY_DENIED"
    assert outcome.trace.status == "rejected"
    assert outcome.trace.error_code == "SOURCE_AUTHORITY_DENIED"


@pytest.mark.anyio
async def test_source_authority_gate_accepts_dict_contract() -> None:
    registry = build_production_tool_registry()
    executor = ProductionToolExecutor(registry=registry)
    authority = {
        "current_user_facts": "allowed",
        "selected_documents": "allowed",
        "deterministic_derivation": "allowed",
        "memory": "allowed",
        "general_model_knowledge": "allowed",
        "domain_heuristics": "allowed",
        "web": "forbidden",
    }
    outcome = await executor.execute_one(
        _heuristic_call(),
        context=_context(source_authority=authority),
    )
    assert outcome.result.success is True


@pytest.mark.anyio
async def test_source_authority_gate_fails_closed_on_unknown_class() -> None:
    class EmptyInput(BaseModel):
        model_config = ConfigDict(extra="forbid")

    called: list[str] = []

    def handler() -> dict[str, bool]:
        called.append("handler")
        return {"ok": True}

    spec = ToolSpec(
        name="unknown_class_tool",
        description="unknown source class must fail closed",
        input_model=EmptyInput,
        handler=handler,
        source_class="unknown_class",  # type: ignore[arg-type]
    )
    registry = ToolRegistry()
    registry.register(spec)
    registry.freeze()
    executor = ProductionToolExecutor(registry=registry)

    outcome = await executor.execute_one(
        ToolCallRequest(
            tool_call_id="call_unknown",
            tool_name="unknown_class_tool",
            arguments={},
        ),
        context=_context(source_authority=SourceAuthorityContract()),
    )

    assert called == []
    assert outcome.result.success is False
    assert outcome.result.error is not None
    assert outcome.result.error.code == "SOURCE_AUTHORITY_DENIED"


@pytest.mark.anyio
async def test_source_authority_gate_disabled_without_contract() -> None:
    registry = build_production_tool_registry()
    executor = ProductionToolExecutor(registry=registry)
    outcome = await executor.execute_one(
        _heuristic_call(),
        context=_context(source_authority=None),
    )
    assert outcome.result.success is True


def test_source_authority_denied_blocks_reuse_signature() -> None:
    registry = build_production_tool_registry()
    executor = ProductionToolExecutor(registry=registry)
    authority = SourceAuthorityContract(
        domain_heuristics="forbidden",
    )
    signature = executor.build_reuse_signature(
        _heuristic_call(),
        context=_context(source_authority=authority),
    )
    assert signature is None


def test_source_authority_helpers_normalize_dict() -> None:
    raw = {
        "domain_heuristics": "forbidden",
        "web": "forbidden",
    }
    contract = normalize_source_authority(raw)
    assert isinstance(contract, SourceAuthorityContract)
    assert contract.domain_heuristics == "forbidden"

    assert normalize_source_authority(None) is None

    route_context = {
        "semantic_route": {
            "source_authority": raw,
        },
    }
    extracted = source_authority_from_route_context(route_context)
    assert extracted == raw

    assert (
        source_authority_from_route_context({"semantic_route": {}})
        is None
    )
