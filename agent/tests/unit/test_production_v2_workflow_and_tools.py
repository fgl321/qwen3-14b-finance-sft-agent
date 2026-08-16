from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.agent_graph.llm_output_guard import OutputGuardInvocationResult
from app.agent_graph.llm_synthesizer import SynthesisInvocationResult
from app.agent_graph.llm_task_planner import PlannerInvocationResult
from app.agent_graph.production_dependencies import ProductionGraphDependencies
from app.agent_graph.production_graph import build_production_finance_graph
from app.agent_graph.explicit_workflow import (
    build_capability_validator_node,
    build_observation_validator_node,
    route_after_capability_validation,
    route_after_observation,
)
from app.agent_graph.production_service import ProductionFinanceGraphService
from app.agent_graph.runtime.agent_limits import AgentLimits
from app.agent_graph.schemas.planner_schema import PlannerDecision
from app.agent_graph.schemas.synthesis_schema import OutputGuardResult, SynthesisResult
from app.tools.financial_analytics_tools import (
    asset_allocation_rebalance,
    bond_analytics,
    cashflow_npv_irr,
    compound_interest_projection,
    financial_ratio_analysis,
    loan_amortization_compare,
    portfolio_risk_metrics,
)
from app.tools.runtime_registry import build_production_tool_registry


class DirectPlanner:
    async def plan(self, request):
        return PlannerInvocationResult(
            decision=PlannerDecision(
                action="respond",
                decision_reason="无需工具。",
                confidence="high",
            ),
            model="planner-test",
            usage={"total_tokens": 2},
        )


class Reviewer:
    def should_review(self, **_):
        return False


class Executor:
    pass


class Synthesizer:
    async def synthesize(self, request):
        return SynthesisInvocationResult(
            result=SynthesisResult(answer=f"正式回答：{request.user_message}"),
            model="qwen-test",
            usage={"total_tokens": 3},
        )


class Guard:
    async def guard(self, _request):
        return OutputGuardInvocationResult(
            result=OutputGuardResult(verdict="pass", reason="一致性检查通过。"),
            model="guard-test",
            usage={"total_tokens": 1},
        )


@pytest.mark.anyio
async def test_explicit_graph_exposes_production_nodes():
    dependencies = ProductionGraphDependencies(
        agent_loop=None,  # type: ignore[arg-type]
        final_response_pipeline=None,  # type: ignore[arg-type]
        planner=DirectPlanner(),  # type: ignore[arg-type]
        reviewer=Reviewer(),  # type: ignore[arg-type]
        executor=Executor(),  # type: ignore[arg-type]
        synthesizer=Synthesizer(),  # type: ignore[arg-type]
        output_guard=Guard(),  # type: ignore[arg-type]
        limits=AgentLimits(max_agent_rounds=2, max_total_tool_calls=6),
    )
    graph = build_production_finance_graph(
        dependencies=dependencies,
        checkpointer=InMemorySaver(),
    )
    service = ProductionFinanceGraphService(graph=graph)
    result = await service.run(
        request_id="req-explicit",
        run_id="run-explicit",
        user_message="什么是久期？",
        user_id="owner",
        thread_id="thread-explicit",
        tenant_id="personal",
    )
    assert result["status"] == "completed"
    assert result["final_answer"] == "正式回答：什么是久期？"
    assert result["usage"]["total_tokens"] == 6
    names = [entry["node"] for entry in result["node_trace"]]
    assert names == [
        "intent_router",
        "planner",
        "agent_result_assembler",
        "answer_synthesis",
        "output_guard",
        "trace_finalizer",
    ]


def test_registry_contains_exactly_ten_financial_tools():
    registry = build_production_tool_registry()
    assert len(registry.names()) == 10
    assert {
        "compound_interest_projection",
        "loan_amortization_compare",
        "cashflow_npv_irr",
        "bond_analytics",
        "portfolio_risk_metrics",
        "asset_allocation_rebalance",
        "financial_ratio_analysis",
    } <= set(registry.names())


def test_financial_analytics_reference_cases():
    compound = compound_interest_projection(
        initial_principal=Decimal("1000"),
        annual_rate_percent=Decimal("0"),
        years=1,
    )
    assert compound["future_value"] == Decimal("1000.00")

    loan = loan_amortization_compare(
        principal=Decimal("120000"),
        annual_rate_percent=Decimal("0"),
        years=1,
    )
    assert loan["baseline"]["equal_payment"]["monthly_payment"] == Decimal("10000.00")

    cashflow = cashflow_npv_irr(
        cash_flows=[Decimal("-100"), Decimal("110")],
        discount_rate_percent=Decimal("10"),
    )
    assert cashflow["npv"] == Decimal("0.00")
    assert abs(cashflow["irr_percent"] - Decimal("10")) <= Decimal("0.0001")

    bond = bond_analytics(
        face_value=Decimal("1000"),
        annual_coupon_rate_percent=Decimal("5"),
        annual_yield_percent=Decimal("5"),
        years_to_maturity=2,
        payments_per_year=1,
    )
    assert bond["dirty_price"] == Decimal("1000.00")

    risk = portfolio_risk_metrics(
        periodic_returns_percent=[Decimal("10"), Decimal("-10")],
        periods_per_year=2,
    )
    assert risk["cumulative_return_percent"] == Decimal("-1.0000")
    assert risk["max_drawdown_percent"] == Decimal("-10.0000")

    rebalance = asset_allocation_rebalance(
        current_amounts={"stock": Decimal("600"), "bond": Decimal("400")},
        target_weights_percent={"stock": Decimal("50"), "bond": Decimal("50")},
    )
    by_asset = {item["asset"]: item for item in rebalance["trades"]}
    assert by_asset["stock"]["trade"] == "sell"
    assert by_asset["stock"]["trade_amount"] == Decimal("100.00")

    ratios = financial_ratio_analysis(
        revenue=Decimal("1000"),
        net_income=Decimal("100"),
        total_assets=Decimal("500"),
        total_liabilities=Decimal("200"),
        total_equity=Decimal("300"),
        current_assets=Decimal("200"),
        current_liabilities=Decimal("100"),
    )
    assert ratios["current_ratio"] == Decimal("2.000000")
    assert ratios["net_margin_percent"] == Decimal("10.0000")


@pytest.mark.anyio
async def test_last_round_verified_results_are_synthesized_instead_of_discarded():
    dependencies = SimpleNamespace(limits=AgentLimits(max_agent_rounds=3))
    node = build_observation_validator_node(dependencies)  # type: ignore[arg-type]
    state = {
        "request_id": "req-budget",
        "planner_round": 3,
        "node_trace": [],
        "current_tool_results": [
            {
                "tool_call_id": "call-final",
                "tool_name": "emergency_fund_range",
                "success": True,
                "output": {"min_amount": "45000.00", "max_amount": "90000.00"},
                "duration_ms": 1,
            }
        ],
        "tool_results": [
            {
                "tool_call_id": "call-final",
                "tool_name": "emergency_fund_range",
                "success": True,
                "output": {"min_amount": "45000.00", "max_amount": "90000.00"},
                "duration_ms": 1,
            }
        ],
        "error_counts": {},
        "repeated_error_count": 0,
        "consecutive_no_progress_rounds": 0,
    }

    update = await node(state)  # type: ignore[arg-type]

    assert update["loop_status"] == "completed"
    assert update["loop_finish_reason"] == (
        "max_agent_rounds_completed_with_verified_results"
    )
    assert update["current_decision"]["action"] == "respond"
    assert route_after_observation({**state, **update}) == "validate"  # type: ignore[arg-type]

    capability_node = build_capability_validator_node(dependencies)  # type: ignore[arg-type]
    capability_update = await capability_node({**state, **update})  # type: ignore[arg-type]
    combined = {**state, **update, **capability_update}
    assert route_after_capability_validation(combined) == "assemble"  # type: ignore[arg-type]
    assert capability_update["last_execution_observation"]["execution_round"] == 3
    assert "replan_count" not in capability_update


@pytest.mark.anyio
async def test_partial_evidence_is_completed_not_partial():
    dependencies = SimpleNamespace(
        limits=AgentLimits(max_agent_rounds=3)
    )
    node = build_capability_validator_node(dependencies)  # type: ignore[arg-type]
    state = {
        "request_id": "req-partial",
        "execution_round": 1,
        "loop_status": "running",
        "node_trace": [],
        "execution_round_history": [],
        "remaining_tool_calls": 6,
        "total_tool_calls": 3,
        "current_tool_results": [],
        "tool_results": [
            {
                "tool_call_id": "call_1",
                "tool_name": "life_insurance_gap",
                "success": True,
                "output": {"gap_5y": "1150000"},
                "duration_ms": 1,
            }
        ],
        "route_context": {
            "retrieval_outcome": {
                "status": "completed_with_partial_evidence",
                "missing_retrieval_requirements": [],
                "unresolved_required_conflicts": [],
            },
            "semantic_route": {
                "required_capabilities": [
                    "knowledge_retrieval",
                    "citation_validation",
                ],
                "task_requirements": [
                    {
                        "id": "t_rag",
                        "description": "检索并引用上传文档",
                        "required": True,
                        "capabilities": ["knowledge_retrieval"],
                        "evidence_tool_names": [],
                        "requires_citations": True,
                        "task_kind": "retrieval",
                    }
                ],
            },
        },
    }

    update = await node(state)  # type: ignore[arg-type]

    assert update["loop_status"] == "completed"
    assert update["loop_finish_reason"] == (
        "result_validation_completed"
    )
    capability_status = update[
        "last_execution_observation"
    ]["capability_status"]
    assert capability_status["knowledge_retrieval"] == (
        "partial_evidence"
    )
