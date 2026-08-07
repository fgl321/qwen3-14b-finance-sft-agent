from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.agent_graph.production_dependencies import (
    ProductionGraphDependencies,
)
from app.agent_graph.production_graph import (
    build_production_finance_graph,
)
from app.agent_graph.production_service import (
    ProductionFinanceGraphService,
    build_checkpoint_thread_id,
)
from app.agent_graph.schemas.final_response_schema import (
    FinalResponsePipelineResult,
)
from app.agent_graph.schemas.loop_schema import (
    AgentLoopResult,
)
from app.agent_graph.schemas.planner_schema import (
    PlannerDecision,
)
from app.agent_graph.schemas.synthesis_schema import (
    OutputGuardResult,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeAgentLoop:
    def __init__(self):
        self.requests = []
        self.should_fail = False

    async def run(
        self,
        *,
        request,
        execution_context,
    ):
        self.requests.append(
            {
                "request": request,
                "execution_context": (
                    execution_context
                ),
            }
        )

        if self.should_fail:
            raise RuntimeError(
                "内部测试异常"
            )

        return AgentLoopResult(
            status="completed",
            final_decision=PlannerDecision(
                action="respond",
                decision_reason=(
                    "测试工具循环完成。"
                ),
                confidence="high",
                plan_version=1,
            ),
            agent_rounds=1,
            total_tool_calls=0,
            finish_reason=(
                "planner_finished"
            ),
        )


class FakeFinalResponsePipeline:
    def __init__(self):
        self.requests = []

    async def run(self, request):
        self.requests.append(request)

        return FinalResponsePipelineResult(
            status="completed",
            answer=(
                f"已完成："
                f"{request.user_message}"
            ),
            guard=OutputGuardResult(
                verdict="pass",
                reason="测试通过。",
                risk_flags=[],
                rewrite_instructions=None,
            ),
            finish_reason=(
                "output_guard_passed"
            ),
        )


def build_service():
    loop = FakeAgentLoop()
    pipeline = FakeFinalResponsePipeline()

    dependencies = ProductionGraphDependencies(
        agent_loop=loop,
        final_response_pipeline=pipeline,
    )

    graph = build_production_finance_graph(
        dependencies=dependencies,
        checkpointer=InMemorySaver(),
    )

    service = ProductionFinanceGraphService(
        graph=graph
    )

    return service, loop, pipeline


def test_checkpoint_thread_id_should_isolate_users():
    first = build_checkpoint_thread_id(
        tenant_id="tenant_a",
        user_id="user_a",
        thread_id="thread_1",
    )

    second = build_checkpoint_thread_id(
        tenant_id="tenant_a",
        user_id="user_b",
        thread_id="thread_1",
    )

    assert first != second

    assert first.startswith("finance-agent:")
    assert len(first) == len("finance-agent:") + 64


@pytest.mark.anyio
async def test_graph_should_return_final_answer():
    service, loop, pipeline = (
        build_service()
    )

    result = await service.run(
        request_id="request_1",
        run_id="run_1",
        user_message="测试问题一",
        user_id="user_1",
        thread_id="thread_1",
    )

    assert result["status"] == "completed"

    assert (
        result["finish_reason"]
        == "output_guard_passed"
    )

    assert result["final_answer"] == (
        "已完成：测试问题一"
    )

    assert result["graph_version"] == (
        "stage_4_2_8f"
    )

    assert len(loop.requests) == 1
    assert len(pipeline.requests) == 1


@pytest.mark.anyio
async def test_checkpoint_should_store_latest_state():
    service, _, _ = build_service()

    await service.run(
        request_id="request_checkpoint",
        run_id="run_checkpoint",
        user_message="检查持久化状态",
        user_id="user_checkpoint",
        thread_id="thread_checkpoint",
    )

    state = (
        await service.get_checkpoint_state(
            user_id="user_checkpoint",
            thread_id="thread_checkpoint",
        )
    )

    assert (
        state["request_id"]
        == "request_checkpoint"
    )

    assert state["status"] == "completed"

    assert state["final_answer"] == (
        "已完成：检查持久化状态"
    )


@pytest.mark.anyio
async def test_same_thread_should_reset_previous_turn():
    service, _, _ = build_service()

    await service.run(
        request_id="request_first",
        run_id="run_first",
        user_message="第一轮问题",
        user_id="user_same",
        thread_id="thread_same",
    )

    second = await service.run(
        request_id="request_second",
        run_id="run_second",
        user_message="第二轮问题",
        user_id="user_same",
        thread_id="thread_same",
    )

    assert (
        second["request_id"]
        == "request_second"
    )

    assert second["run_id"] == "run_second"

    assert second["final_answer"] == (
        "已完成：第二轮问题"
    )

    assert "第一轮问题" not in (
        second["final_answer"]
    )


@pytest.mark.anyio
async def test_agent_loop_failure_should_use_safe_fallback():
    service, loop, _ = build_service()

    loop.should_fail = True

    result = await service.run(
        user_message="触发测试异常",
        user_id="user_error",
        thread_id="thread_error",
    )

    assert result["status"] == "fallback"

    assert (
        result["finish_reason"]
        == "agent_loop_node_failed"
    )

    assert "RuntimeError" not in (
        result["final_answer"]
    )

    assert result["final_answer"] == (
        "系统暂时无法安全完成本次处理，"
        "请稍后重新提交问题。"
    )
