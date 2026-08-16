from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent_graph.agent_loop import AgentToolLoop
from app.agent_graph.final_response_pipeline import (
    FinalResponsePipeline,
)
from app.agent_graph.llm_output_guard import (
    LLMOutputGuard,
)
from app.agent_graph.llm_plan_reviewer import (
    LLMPlanReviewer,
)
from app.agent_graph.llm_synthesizer import (
    LLMAnswerSynthesizer,
)
from app.agent_graph.llm_task_planner import (
    LLMTaskPlanner,
)
from app.agent_graph.runtime.agent_limits import (
    AgentLimits,
    DEFAULT_AGENT_LIMITS,
)
from app.tools.runtime_registry import (
    build_production_tool_registry,
)
from app.tools.tool_executor import (
    ProductionToolExecutor,
)


@dataclass(slots=True)
class ProductionGraphDependencies:
    """
    生产图依赖。

    依赖通过闭包注入节点，不写入 LangGraph State。
    """

    agent_loop: AgentToolLoop
    final_response_pipeline: FinalResponsePipeline
    planner: LLMTaskPlanner | None = None
    reviewer: LLMPlanReviewer | None = None
    executor: ProductionToolExecutor | None = None
    synthesizer: LLMAnswerSynthesizer | None = None
    output_guard: LLMOutputGuard | None = None
    limits: AgentLimits = DEFAULT_AGENT_LIMITS

    @property
    def explicit_workflow_ready(self) -> bool:
        return all(
            item is not None
            for item in (
                self.planner,
                self.reviewer,
                self.executor,
                self.synthesizer,
                self.output_guard,
            )
        )


def build_production_graph_dependencies(
    *,
    llm_client: Any,
    synthesis_llm_client: Any | None = None,
    limits: AgentLimits = DEFAULT_AGENT_LIMITS,
) -> ProductionGraphDependencies:
    """
    复用同一个 DeepSeekClient 构造全链路组件。

    不在每个节点中重复创建模型客户端。
    """

    registry = build_production_tool_registry()

    planner = LLMTaskPlanner(
        llm_client=llm_client,
        registry=registry,
    )

    reviewer = LLMPlanReviewer(
        llm_client=llm_client,
        registry=registry,
    )

    executor = ProductionToolExecutor(
        registry=registry,
        limits=limits,
    )

    agent_loop = AgentToolLoop(
        planner=planner,
        reviewer=reviewer,
        executor=executor,
        limits=limits,
    )

    synthesizer = LLMAnswerSynthesizer(
        llm_client=synthesis_llm_client or llm_client,
    )

    output_guard = LLMOutputGuard(
        llm_client=llm_client,
    )

    final_response_pipeline = FinalResponsePipeline(
        synthesizer=synthesizer,
        output_guard=output_guard,
        limits=limits,
    )

    return ProductionGraphDependencies(
        agent_loop=agent_loop,
        final_response_pipeline=(
            final_response_pipeline
        ),
        planner=planner,
        reviewer=reviewer,
        executor=executor,
        synthesizer=synthesizer,
        output_guard=output_guard,
        limits=limits,
    )
