import asyncio
from decimal import Decimal

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.agent_graph.runtime.agent_limits import AgentLimits
from app.agent_graph.schemas.planner_schema import ToolCallRequest
from app.tools.runtime_registry import (
    ToolRegistry,
    build_production_tool_registry,
)
from app.tools.tool_executor import (
    ProductionToolExecutor,
    ToolExecutionContext,
    summarize_for_trace,
)
from app.tools.tool_specs import ToolSpec


@pytest.fixture
def anyio_backend():
    return "asyncio"


def build_context(
    *,
    allowed_tool_names: frozenset[str] | None = None,
    allowed_tool_groups: frozenset[str] | None = None,
    remaining_tool_calls: int = 12,
    allow_side_effects: bool = False,
) -> ToolExecutionContext:
    return ToolExecutionContext(
        request_id="request_test",
        run_id="run_test",
        tenant_id="default",
        user_id="user_test",
        role="user",
        allowed_tool_names=allowed_tool_names,
        allowed_tool_groups=allowed_tool_groups,
        remaining_tool_calls=remaining_tool_calls,
        allow_side_effects=allow_side_effects,
    )


def test_registry_should_be_frozen_and_explicit() -> None:
    registry = build_production_tool_registry()

    assert registry.frozen is True

    assert registry.names() == (
        "emergency_fund_range",
        "life_insurance_gap",
        "yearly_expense_to_monthly",
    )

    with pytest.raises(RuntimeError):
        registry.register(
            registry.require("emergency_fund_range")
        )


def test_registry_should_reject_duplicate_names() -> None:
    class EmptyInput(BaseModel):
        model_config = ConfigDict(extra="forbid")

    def handler() -> dict:
        return {"ok": True}

    spec = ToolSpec(
        name="duplicate_tool",
        description="重复工具测试。",
        input_model=EmptyInput,
        handler=handler,
    )

    registry = ToolRegistry()
    registry.register(spec)

    with pytest.raises(ValueError):
        registry.register(spec)


def test_registry_should_filter_llm_definitions() -> None:
    registry = build_production_tool_registry()

    definitions = registry.get_llm_tool_definitions(
        allowed_tool_names={
            "yearly_expense_to_monthly",
        }
    )

    assert len(definitions) == 1
    assert (
        definitions[0]["function"]["name"]
        == "yearly_expense_to_monthly"
    )


@pytest.mark.anyio
async def test_execute_yearly_expense_to_monthly() -> None:
    registry = build_production_tool_registry()

    executor = ProductionToolExecutor(
        registry=registry
    )

    outcome = await executor.execute_one(
        ToolCallRequest(
            tool_call_id="call_yearly",
            tool_name="yearly_expense_to_monthly",
            arguments={
                "yearly_necessary_expense": 180000,
            },
        ),
        context=build_context(),
    )

    assert outcome.result.success is True
    assert outcome.result.error is None

    assert (
        outcome.result.output[
            "monthly_necessary_expense"
        ]
        == "15000.00"
    )

    assert outcome.trace.status == "succeeded"


@pytest.mark.anyio
async def test_execute_emergency_fund_range() -> None:
    executor = ProductionToolExecutor(
        registry=build_production_tool_registry()
    )

    outcome = await executor.execute_one(
        ToolCallRequest(
            tool_call_id="call_emergency",
            tool_name="emergency_fund_range",
            arguments={
                "monthly_necessary_expense": 15000,
                "min_months": 3,
                "max_months": 6,
            },
        ),
        context=build_context(),
    )

    assert outcome.result.success is True
    assert outcome.result.output["min_amount"] == "45000.00"
    assert outcome.result.output["max_amount"] == "90000.00"


@pytest.mark.anyio
async def test_execute_life_insurance_gap() -> None:
    executor = ProductionToolExecutor(
        registry=build_production_tool_registry()
    )

    outcome = await executor.execute_one(
        ToolCallRequest(
            tool_call_id="call_life",
            tool_name="life_insurance_gap",
            arguments={
                "annual_necessary_expense": 180000,
                "coverage_years": 10,
                "outstanding_debt": 800000,
                "education_fund": 0,
                "other_family_responsibilities": 0,
                "available_assets": 250000,
                "existing_life_insurance": 300000,
            },
        ),
        context=build_context(),
    )

    assert outcome.result.success is True

    assert (
        outcome.result.output["life_insurance_gap"]
        == "2050000.00"
    )


@pytest.mark.anyio
async def test_unknown_tool_should_return_repairable_error() -> None:
    executor = ProductionToolExecutor(
        registry=build_production_tool_registry()
    )

    outcome = await executor.execute_one(
        ToolCallRequest(
            tool_call_id="call_unknown",
            tool_name="not_existing_tool",
            arguments={},
        ),
        context=build_context(),
    )

    assert outcome.result.success is False
    assert outcome.result.error is not None
    assert outcome.result.error.code == "TOOL_NOT_FOUND"
    assert outcome.result.error.model_repairable is True


@pytest.mark.anyio
async def test_invalid_arguments_should_return_schema_error() -> None:
    executor = ProductionToolExecutor(
        registry=build_production_tool_registry()
    )

    outcome = await executor.execute_one(
        ToolCallRequest(
            tool_call_id="call_invalid",
            tool_name="yearly_expense_to_monthly",
            arguments={
                "monthly_expense": 180000,
            },
        ),
        context=build_context(),
    )

    assert outcome.result.success is False
    assert outcome.result.error is not None

    assert (
        outcome.result.error.code
        == "ARGUMENT_SCHEMA_ERROR"
    )

    assert outcome.result.error.model_repairable is True

    details = outcome.result.error.details

    assert "validation_errors" in details

    validation_errors = details["validation_errors"]

    assert validation_errors

    # 每条错误只能保留安全字段，不能包含 Pydantic 原始 input。
    for error_item in validation_errors:
        assert set(error_item.keys()) == {
            "location",
            "type",
            "message",
        }

        assert "input" not in error_item

    # 参数名称可以返回给 Planner 用于修复，
    # 但不能把用户提交的原始参数值复制进错误详情。
    assert "180000" not in str(details)


@pytest.mark.anyio
async def test_domain_month_range_should_fail() -> None:
    executor = ProductionToolExecutor(
        registry=build_production_tool_registry()
    )

    outcome = await executor.execute_one(
        ToolCallRequest(
            tool_call_id="call_bad_range",
            tool_name="emergency_fund_range",
            arguments={
                "monthly_necessary_expense": 15000,
                "min_months": 9,
                "max_months": 3,
            },
        ),
        context=build_context(),
    )

    assert outcome.result.success is False
    assert outcome.result.error is not None

    # 当前错误发生于 Pydantic 输入模型，因此是结构错误。
    assert (
        outcome.result.error.code
        == "ARGUMENT_SCHEMA_ERROR"
    )


@pytest.mark.anyio
async def test_route_permission_should_be_enforced() -> None:
    executor = ProductionToolExecutor(
        registry=build_production_tool_registry()
    )

    outcome = await executor.execute_one(
        ToolCallRequest(
            tool_call_id="call_denied",
            tool_name="life_insurance_gap",
            arguments={
                "annual_necessary_expense": 180000,
            },
        ),
        context=build_context(
            allowed_tool_names=frozenset(
                {"yearly_expense_to_monthly"}
            )
        ),
    )

    assert outcome.result.success is False
    assert outcome.result.error is not None
    assert outcome.result.error.code == "PERMISSION_DENIED"


@pytest.mark.anyio
async def test_tool_budget_should_be_enforced() -> None:
    executor = ProductionToolExecutor(
        registry=build_production_tool_registry()
    )

    outcome = await executor.execute_one(
        ToolCallRequest(
            tool_call_id="call_budget",
            tool_name="yearly_expense_to_monthly",
            arguments={
                "yearly_necessary_expense": 180000,
            },
        ),
        context=build_context(
            remaining_tool_calls=0
        ),
    )

    assert outcome.result.success is False
    assert outcome.result.error is not None

    assert (
        outcome.result.error.code
        == "AGENT_BUDGET_EXCEEDED"
    )


@pytest.mark.anyio
async def test_timeout_should_be_wrapped() -> None:
    class SlowInput(BaseModel):
        model_config = ConfigDict(extra="forbid")

        delay: float = Field(gt=0)

    async def slow_handler(
        *,
        delay: float,
    ) -> dict:
        await asyncio.sleep(delay)
        return {"completed": True}

    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="slow_tool",
            description="超时测试工具。",
            input_model=SlowInput,
            handler=slow_handler,
            timeout_seconds=0.02,
            max_infrastructure_retries=0,
        )
    )

    registry.freeze()

    executor = ProductionToolExecutor(
        registry=registry
    )

    outcome = await executor.execute_one(
        ToolCallRequest(
            tool_call_id="call_timeout",
            tool_name="slow_tool",
            arguments={
                "delay": 0.2,
            },
        ),
        context=build_context(),
    )

    assert outcome.result.success is False
    assert outcome.result.error is not None
    assert outcome.result.error.code == "TOOL_TIMEOUT"
    assert outcome.trace.status == "timed_out"


@pytest.mark.anyio
async def test_internal_exception_should_not_escape_graph() -> None:
    class CrashInput(BaseModel):
        model_config = ConfigDict(extra="forbid")

    def crash_handler() -> dict:
        raise RuntimeError("内部测试异常")

    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="crash_tool",
            description="异常封装测试工具。",
            input_model=CrashInput,
            handler=crash_handler,
            timeout_seconds=1,
        )
    )

    registry.freeze()

    executor = ProductionToolExecutor(
        registry=registry
    )

    outcome = await executor.execute_one(
        ToolCallRequest(
            tool_call_id="call_crash",
            tool_name="crash_tool",
            arguments={},
        ),
        context=build_context(),
    )

    assert outcome.result.success is False
    assert outcome.result.error is not None

    assert (
        outcome.result.error.code
        == "TOOL_INTERNAL_ERROR"
    )

    # 不把原始异常内容直接返回给模型。
    assert "内部测试异常" not in outcome.result.error.message


@pytest.mark.anyio
async def test_parallel_execution_should_preserve_order() -> None:
    class DelayInput(BaseModel):
        model_config = ConfigDict(extra="forbid")

        value: int
        delay: float = Field(ge=0)

    async def delayed_handler(
        *,
        value: int,
        delay: float,
    ) -> dict:
        await asyncio.sleep(delay)
        return {"value": value}

    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="delayed_tool",
            description="并行顺序测试工具。",
            input_model=DelayInput,
            handler=delayed_handler,
            timeout_seconds=1,
            parallel_safe=True,
        )
    )

    registry.freeze()

    executor = ProductionToolExecutor(
        registry=registry,
        limits=AgentLimits(
            max_parallel_tool_calls=3,
        ),
    )

    calls = [
        ToolCallRequest(
            tool_call_id="call_1",
            tool_name="delayed_tool",
            arguments={"value": 1, "delay": 0.04},
        ),
        ToolCallRequest(
            tool_call_id="call_2",
            tool_name="delayed_tool",
            arguments={"value": 2, "delay": 0.01},
        ),
        ToolCallRequest(
            tool_call_id="call_3",
            tool_name="delayed_tool",
            arguments={"value": 3, "delay": 0.02},
        ),
    ]

    outcomes = await executor.execute_many(
        calls,
        context=build_context(),
    )

    assert [
        outcome.result.tool_call_id
        for outcome in outcomes
    ] == [
        "call_1",
        "call_2",
        "call_3",
    ]

    assert [
        outcome.result.output["value"]
        for outcome in outcomes
    ] == [1, 2, 3]


def test_trace_should_redact_credentials_and_finance() -> None:
    summary = summarize_for_trace(
        {
            "api_key": "test-secret-value",
            "annual_income": 500000,
            "mortgage_balance": 800000,
            "normal_field": "visible",
        }
    )

    assert summary["api_key"] == "[redacted]"

    assert (
        summary["annual_income"]
        == "[financial_value]"
    )

    assert (
        summary["mortgage_balance"]
        == "[financial_value]"
    )

    assert summary["normal_field"] == "visible"
