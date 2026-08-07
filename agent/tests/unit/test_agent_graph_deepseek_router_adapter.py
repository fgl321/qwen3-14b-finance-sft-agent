import asyncio
from typing import Any

import pytest

from app.agent_graph.deepseek_router_adapter import (
    DeepSeekRouterCompletion,
    build_hybrid_question_router,
)
from app.agent_graph.question_router import (
    QuestionCapability,
    RuleConfidence,
)


class FakeDeepSeekClient:
    """
    模拟项目现有的 DeepSeekClient。

    用来验证：
    1. 适配器是否调用 chat()；
    2. 参数是否正确传递；
    3. 返回内容是否正确提取；
    4. 异常时是否正确处理；
    5. 构造出的混合路由器是否正常工作。
    """

    def __init__(
        self,
        *,
        result: Any = None,
        error: Exception | None = None,
        delay_seconds: float = 0,
    ) -> None:
        self.result = result
        self.error = error
        self.delay_seconds = delay_seconds

        self.call_count = 0
        self.last_kwargs: dict[str, Any] | None = None

    async def chat(
        self,
        **kwargs: Any,
    ) -> Any:
        self.call_count += 1
        self.last_kwargs = kwargs

        if self.delay_seconds > 0:
            await asyncio.sleep(
                self.delay_seconds
            )

        if self.error is not None:
            raise self.error

        return self.result


def test_adapter_should_reuse_existing_deepseek_chat():
    """
    适配器应复用传入的 DeepSeekClient.chat()，
    而不是创建新的客户端。
    """

    fake_client = FakeDeepSeekClient(
        result={
            "message": {
                "role": "assistant",
                "content": """
                {
                  "required_capabilities": [
                    "general_explanation"
                  ],
                  "confidence": "high",
                  "reason": "用户询问的是普通金融概念。"
                }
                """,
            },
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 30,
            },
        }
    )

    completion = DeepSeekRouterCompletion(
        llm_client=fake_client,
        max_completion_tokens=256,
    )

    messages = [
        {
            "role": "system",
            "content": "系统提示词",
        },
        {
            "role": "user",
            "content": "用户问题",
        },
    ]

    result = asyncio.run(
        completion(messages)
    )

    assert fake_client.call_count == 1
    assert '"required_capabilities"' in result

    assert fake_client.last_kwargs is not None
    assert fake_client.last_kwargs["messages"] == messages

    assert (
        fake_client.last_kwargs["thinking_enabled"]
        is False
    )

    assert (
        fake_client.last_kwargs[
            "max_completion_tokens"
        ]
        == 256
    )

    assert fake_client.last_kwargs[
        "response_format"
    ] == {
        "type": "json_object",
    }


def test_adapter_should_strip_content_whitespace():
    """
    适配器应该清理返回正文两侧的空白字符。
    """

    fake_client = FakeDeepSeekClient(
        result={
            "message": {
                "role": "assistant",
                "content": "   测试内容   ",
            }
        }
    )

    completion = DeepSeekRouterCompletion(
        llm_client=fake_client,
    )

    result = asyncio.run(
        completion(
            [
                {
                    "role": "user",
                    "content": "测试问题",
                }
            ]
        )
    )

    assert result == "测试内容"


def test_adapter_should_reject_non_dict_result():
    """
    DeepSeekClient 返回值不是字典时，
    适配器应抛出明确异常。
    """

    fake_client = FakeDeepSeekClient(
        result="非法返回结果"
    )

    completion = DeepSeekRouterCompletion(
        llm_client=fake_client,
    )

    with pytest.raises(
        RuntimeError,
        match="返回值不是字典",
    ):
        asyncio.run(
            completion(
                [
                    {
                        "role": "user",
                        "content": "测试问题",
                    }
                ]
            )
        )


def test_adapter_should_reject_missing_message():
    """
    返回结果缺少 message 字段时，
    适配器应抛出异常。
    """

    fake_client = FakeDeepSeekClient(
        result={
            "usage": {},
        }
    )

    completion = DeepSeekRouterCompletion(
        llm_client=fake_client,
    )

    with pytest.raises(
        RuntimeError,
        match="缺少 message",
    ):
        asyncio.run(
            completion(
                [
                    {
                        "role": "user",
                        "content": "测试问题",
                    }
                ]
            )
        )


def test_adapter_should_reject_invalid_message_type():
    """
    message 不是字典时，
    适配器应抛出异常。
    """

    fake_client = FakeDeepSeekClient(
        result={
            "message": "非法 message",
        }
    )

    completion = DeepSeekRouterCompletion(
        llm_client=fake_client,
    )

    with pytest.raises(
        RuntimeError,
        match="缺少 message",
    ):
        asyncio.run(
            completion(
                [
                    {
                        "role": "user",
                        "content": "测试问题",
                    }
                ]
            )
        )


def test_adapter_should_reject_invalid_content_type():
    """
    message.content 不是字符串时，
    适配器应抛出异常。
    """

    fake_client = FakeDeepSeekClient(
        result={
            "message": {
                "role": "assistant",
                "content": None,
            }
        }
    )

    completion = DeepSeekRouterCompletion(
        llm_client=fake_client,
    )

    with pytest.raises(
        RuntimeError,
        match="content 不是字符串",
    ):
        asyncio.run(
            completion(
                [
                    {
                        "role": "user",
                        "content": "测试问题",
                    }
                ]
            )
        )


def test_adapter_should_reject_empty_content():
    """
    message.content 为空字符串时，
    适配器应抛出异常。
    """

    fake_client = FakeDeepSeekClient(
        result={
            "message": {
                "role": "assistant",
                "content": "   ",
            }
        }
    )

    completion = DeepSeekRouterCompletion(
        llm_client=fake_client,
    )

    with pytest.raises(
        RuntimeError,
        match="空内容",
    ):
        asyncio.run(
            completion(
                [
                    {
                        "role": "user",
                        "content": "测试问题",
                    }
                ]
            )
        )


def test_adapter_should_propagate_client_exception():
    """
    底层 DeepSeekClient 抛出异常时，
    适配器不应该吞掉异常。

    异常应继续交给上层 LLMQuestionRouter，
    由上层执行保守回退。
    """

    fake_client = FakeDeepSeekClient(
        error=RuntimeError(
            "模拟 DeepSeek 网络异常"
        )
    )

    completion = DeepSeekRouterCompletion(
        llm_client=fake_client,
    )

    with pytest.raises(
        RuntimeError,
        match="模拟 DeepSeek 网络异常",
    ):
        asyncio.run(
            completion(
                [
                    {
                        "role": "user",
                        "content": "测试问题",
                    }
                ]
            )
        )

    assert fake_client.call_count == 1


def test_adapter_should_reject_invalid_max_tokens():
    """
    最大输出长度必须大于 0。
    """

    fake_client = FakeDeepSeekClient()

    with pytest.raises(
        ValueError,
        match="max_completion_tokens 必须大于 0",
    ):
        DeepSeekRouterCompletion(
            llm_client=fake_client,
            max_completion_tokens=0,
        )


def test_factory_should_build_working_hybrid_router():
    """
    验证构造函数可以生成完整的混合路由器：

    硬规则没有命中
    → 调用 DeepSeek
    → 解析 JSON
    → 返回多个能力。
    """

    fake_client = FakeDeepSeekClient(
        result={
            "message": {
                "role": "assistant",
                "content": """
                {
                  "required_capabilities": [
                    "general_explanation",
                    "complex_reasoning"
                  ],
                  "confidence": "high",
                  "reason": "用户既需要了解基础概念，也存在个人决策意图。"
                }
                """,
            }
        }
    )

    router = build_hybrid_question_router(
        llm_client=fake_client,
        timeout_seconds=2.0,
        max_completion_tokens=256,
    )

    result = asyncio.run(
        router.route(
            "我是不是应该先留一笔应急的钱？"
        )
    )

    assert fake_client.call_count == 1

    assert result.capabilities == (
        QuestionCapability.GENERAL_EXPLANATION,
        QuestionCapability.COMPLEX_REASONING,
    )

    assert result.confidence == (
        RuleConfidence.HIGH
    )

    assert result.router == (
        "llm_semantic_router"
    )

    assert result.used_fallback is False
    assert result.llm_result is not None


def test_factory_should_not_call_llm_when_rule_matches():
    """
    硬规则已经能够判断问题时，
    不应该继续调用 DeepSeek。
    """

    fake_client = FakeDeepSeekClient(
        result={
            "message": {
                "content": (
                    "这个模型返回内容不应该被使用"
                ),
            }
        }
    )

    router = build_hybrid_question_router(
        llm_client=fake_client,
    )

    result = asyncio.run(
        router.route(
            "什么是紧急备用金？"
        )
    )

    assert fake_client.call_count == 0

    assert result.router == "hard_rule"

    assert result.capabilities == (
        QuestionCapability.GENERAL_EXPLANATION,
    )

    assert result.used_fallback is False


def test_factory_should_support_multiple_hard_rule_capabilities():
    """
    硬规则能够同时识别知识库、计算和复杂规划能力。
    """

    fake_client = FakeDeepSeekClient(
        result={
            "message": {
                "content": (
                    "这个模型返回内容不应该被使用"
                ),
            }
        }
    )

    router = build_hybrid_question_router(
        llm_client=fake_client,
    )

    result = asyncio.run(
        router.route(
            "请根据上传的保单，"
            "计算寿险保障缺口，"
            "并给出调整方案。"
        )
    )

    assert fake_client.call_count == 0

    assert result.capabilities == (
        QuestionCapability.KNOWLEDGE_RETRIEVAL,
        QuestionCapability.FINANCIAL_CALCULATION,
        QuestionCapability.COMPLEX_REASONING,
    )

    assert result.router == "hard_rule"
    assert result.used_fallback is False


def test_factory_should_fallback_when_deepseek_raises_exception():
    """
    硬规则未命中且 DeepSeek 调用异常时，
    应保守回退到 complex_reasoning。
    """

    fake_client = FakeDeepSeekClient(
        error=RuntimeError(
            "模拟 DeepSeek API 异常"
        )
    )

    router = build_hybrid_question_router(
        llm_client=fake_client,
        timeout_seconds=2.0,
    )

    result = asyncio.run(
        router.route(
            "最近总觉得钱放着也不安心，"
            "这种情况一般先考虑什么？"
        )
    )

    assert fake_client.call_count == 1

    assert result.capabilities == (
        QuestionCapability.COMPLEX_REASONING,
    )

    assert result.confidence == (
        RuleConfidence.LOW
    )

    assert result.router == "llm_fallback"
    assert result.used_fallback is True

    assert result.llm_result is not None

    assert "RuntimeError" in (
        result.llm_result.validation_error
    )

    assert "模拟 DeepSeek API 异常" in (
        result.llm_result.validation_error
    )


def test_factory_should_fallback_when_deepseek_times_out():
    """
    DeepSeek 路由调用超时时，
    应保守回退到 complex_reasoning。
    """

    fake_client = FakeDeepSeekClient(
        result={
            "message": {
                "content": """
                {
                  "required_capabilities": [
                    "general_explanation"
                  ],
                  "confidence": "high",
                  "reason": "普通金融问题。"
                }
                """,
            }
        },
        delay_seconds=0.05,
    )

    router = build_hybrid_question_router(
        llm_client=fake_client,
        timeout_seconds=0.01,
    )

    result = asyncio.run(
        router.route(
            "最近总觉得钱放着也不安心，"
            "这种情况一般先考虑什么？"
        )
    )

    assert fake_client.call_count == 1

    assert result.capabilities == (
        QuestionCapability.COMPLEX_REASONING,
    )

    assert result.router == "llm_fallback"
    assert result.used_fallback is True

    assert result.llm_result is not None

    assert (
        result.llm_result.validation_error
        == "llm_router_timeout"
    )


def test_factory_should_fallback_when_json_is_invalid():
    """
    DeepSeek 返回非法 JSON 时，
    应保守回退到 complex_reasoning。
    """

    fake_client = FakeDeepSeekClient(
        result={
            "message": {
                "role": "assistant",
                "content": "这不是合法 JSON",
            }
        }
    )

    router = build_hybrid_question_router(
        llm_client=fake_client,
        timeout_seconds=2.0,
    )

    result = asyncio.run(
        router.route(
            "最近总觉得钱放着也不安心，"
            "这种情况一般先考虑什么？"
        )
    )

    assert fake_client.call_count == 1

    assert result.capabilities == (
        QuestionCapability.COMPLEX_REASONING,
    )

    assert result.router == "llm_fallback"
    assert result.used_fallback is True

    assert result.llm_result is not None
    assert result.llm_result.validation_error


def test_factory_should_reject_invalid_timeout():
    """
    路由调用超时时间必须大于 0。
    """

    fake_client = FakeDeepSeekClient()

    with pytest.raises(
        ValueError,
        match="timeout_seconds 必须大于 0",
    ):
        build_hybrid_question_router(
            llm_client=fake_client,
            timeout_seconds=0,
        )
