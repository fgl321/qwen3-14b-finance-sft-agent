import asyncio

from app.agent_graph import nodes
from app.agent_graph.plan_compiler import (
    FINANCE_AGENT_STEP,
    GENERAL_FINANCE_ANSWER_STEP,
)
from app.agent_graph.question_router import (
    QuestionCapability,
    RuleConfidence,
)


class FakeDeepSeekClient:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakeRouteResult:
    def __init__(
        self,
        *,
        capabilities,
        confidence=RuleConfidence.HIGH,
        reason="测试路由原因",
        router="hard_rule",
        matched_rules=(),
        used_fallback=False,
    ) -> None:
        self.capabilities = tuple(capabilities)
        self.confidence = confidence
        self.reason = reason
        self.router = router
        self.matched_rules = tuple(matched_rules)
        self.used_fallback = used_fallback

    def to_dict(self) -> dict[str, object]:
        return {
            "capabilities": [
                capability.value
                for capability in self.capabilities
            ],
            "confidence": self.confidence.value,
            "reason": self.reason,
            "router": self.router,
            "matched_rules": list(
                self.matched_rules
            ),
            "used_fallback": self.used_fallback,
        }


class FakeHybridQuestionRouter:
    def __init__(
        self,
        route_result: FakeRouteResult,
    ) -> None:
        self.route_result = route_result
        self.received_user_message = ""

    async def route(
        self,
        user_message: str,
    ) -> FakeRouteResult:
        self.received_user_message = user_message
        return self.route_result


def test_general_explanation_should_compile_general_answer_plan(
    monkeypatch,
):
    fake_client = FakeDeepSeekClient()

    fake_route_result = FakeRouteResult(
        capabilities=[
            QuestionCapability.GENERAL_EXPLANATION,
        ],
        matched_rules=[
            "explicit_concept_explanation",
        ],
    )

    fake_router = FakeHybridQuestionRouter(
        fake_route_result
    )

    monkeypatch.setattr(
        nodes,
        "_build_deepseek_client",
        lambda settings: fake_client,
    )

    monkeypatch.setattr(
        nodes,
        "build_hybrid_question_router",
        lambda llm_client: fake_router,
    )

    result = asyncio.run(
        nodes.question_router_node(
            {
                "user_message": "什么是紧急备用金？",
                "user_id": "user_001",
                "thread_id": "thread_001",
            }
        )
    )

    assert result["question_capabilities"] == [
        "general_explanation",
    ]

    assert result["execution_plan"] == [
        GENERAL_FINANCE_ANSWER_STEP,
    ]

    assert result["question_router"] == "hard_rule"

    assert result[
        "question_router_confidence"
    ] == "high"

    assert result[
        "question_router_matched_rules"
    ] == [
        "explicit_concept_explanation",
    ]

    assert fake_router.received_user_message == (
        "什么是紧急备用金？"
    )

    assert fake_client.closed is True


def test_multiple_capabilities_should_compile_finance_agent_plan(
    monkeypatch,
):
    fake_client = FakeDeepSeekClient()

    fake_route_result = FakeRouteResult(
        capabilities=[
            QuestionCapability.KNOWLEDGE_RETRIEVAL,
            QuestionCapability.FINANCIAL_CALCULATION,
            QuestionCapability.COMPLEX_REASONING,
        ],
        confidence=RuleConfidence.MEDIUM,
        router="llm_semantic_router",
    )

    fake_router = FakeHybridQuestionRouter(
        fake_route_result
    )

    monkeypatch.setattr(
        nodes,
        "_build_deepseek_client",
        lambda settings: fake_client,
    )

    monkeypatch.setattr(
        nodes,
        "build_hybrid_question_router",
        lambda llm_client: fake_router,
    )

    result = asyncio.run(
        nodes.question_router_node(
            {
                "user_message": (
                    "根据保单计算寿险缺口并给出方案。"
                ),
                "user_id": "user_001",
                "thread_id": "thread_001",
            }
        )
    )

    assert result["question_capabilities"] == [
        "knowledge_retrieval",
        "financial_calculation",
        "complex_reasoning",
    ]

    assert result["execution_plan"] == [
        FINANCE_AGENT_STEP,
    ]

    assert result["question_router"] == (
        "llm_semantic_router"
    )

    assert result[
        "question_router_confidence"
    ] == "medium"

    assert fake_client.closed is True


def test_router_exception_should_fallback_to_finance_agent(
    monkeypatch,
):
    fake_client = FakeDeepSeekClient()

    def raise_router_error(
        llm_client,
    ):
        raise RuntimeError(
            "router build failed"
        )

    monkeypatch.setattr(
        nodes,
        "_build_deepseek_client",
        lambda settings: fake_client,
    )

    monkeypatch.setattr(
        nodes,
        "build_hybrid_question_router",
        raise_router_error,
    )

    result = asyncio.run(
        nodes.question_router_node(
            {
                "user_message": "我应该怎么规划？",
                "user_id": "user_001",
                "thread_id": "thread_001",
            }
        )
    )

    assert result["question_capabilities"] == [
        "complex_reasoning",
    ]

    assert result["execution_plan"] == [
        FINANCE_AGENT_STEP,
    ]

    assert result["question_router"] == (
        "question_router_node_fallback"
    )

    assert result[
        "question_router_used_fallback"
    ] is True

    assert "RuntimeError" in result[
        "question_route_detail"
    ]["node_error"]

    assert "error" not in result

    assert fake_client.closed is True
