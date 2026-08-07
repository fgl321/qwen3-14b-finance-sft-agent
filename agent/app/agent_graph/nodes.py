import inspect
import os
from dataclasses import asdict, is_dataclass
from typing import Any
from uuid import uuid4

import httpx
from openai import AsyncOpenAI

from app.agent.finance_agent import FinanceAgent
from app.agent_graph.deepseek_router_adapter import (
    build_hybrid_question_router,
)
from app.agent_graph.plan_compiler import (
    FINANCE_AGENT_STEP,
    compile_execution_plan,
)
from app.agent_graph.quality_gate import build_quality_gate_result
from app.agent_graph.question_router import QuestionCapability
from app.agent_graph.state import FinanceAgentGraphState
from app.core.config import get_settings
from app.llm.deepseek_client import DeepSeekClient


def _build_deepseek_client(settings: Any) -> DeepSeekClient:
    """
    兼容不同 DeepSeekClient 构造方式。
    """
    try:
        return DeepSeekClient(settings=settings)
    except TypeError:
        try:
            return DeepSeekClient(settings)
        except TypeError:
            return DeepSeekClient()


def _to_plain_dict(obj: Any) -> dict[str, Any]:
    """
    把 FinanceAgentResult 转成普通 dict。
    """
    if obj is None:
        return {}

    if isinstance(obj, dict):
        return obj

    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    if is_dataclass(obj):
        return asdict(obj)

    result: dict[str, Any] = {}

    for field in [
        "request_id",
        "answer",
        "executed_tools",
        "usage",
        "finish_reason",
        "message_count",
        "safety_check",
    ]:
        if hasattr(obj, field):
            result[field] = getattr(obj, field)

    return result


async def _maybe_close_client(client: Any) -> None:
    """
    如果 DeepSeekClient 里有 close / aclose 方法，就安全关闭。
    """
    close_func = getattr(client, "aclose", None)

    if close_func is None:
        close_func = getattr(client, "close", None)

    if close_func is None:
        return

    close_result = close_func()

    if inspect.isawaitable(close_result):
        await close_result


def _read_setting(
    settings: Any,
    attr_names: list[str],
    env_names: list[str],
    default: str | None = None,
) -> str | None:
    """
    兼容不同配置字段名。

    先从 settings 里读；
    读不到再从环境变量里读；
    最后返回 default。
    """
    for attr_name in attr_names:
        if hasattr(settings, attr_name):
            value = getattr(settings, attr_name)
            if value is not None and str(value).strip():
                return str(value).strip()

    for env_name in env_names:
        value = os.getenv(env_name)
        if value is not None and value.strip():
            return value.strip()

    return default


async def _call_deepseek_for_general_finance_answer(
    user_message: str,
) -> tuple[str, dict[str, Any]]:
    """
    通用金融解释兜底。

    注意：
    这里不基于知识库回答，也不伪造 citations。
    只做通用金融知识解释，并且禁止具体投资产品推荐。
    """
    settings = get_settings()

    api_key = _read_setting(
        settings=settings,
        attr_names=[
            "deepseek_api_key",
            "api_key",
            "openai_api_key",
        ],
        env_names=[
            "DEEPSEEK_API_KEY",
            "OPENAI_API_KEY",
        ],
        default=None,
    )

    base_url = _read_setting(
        settings=settings,
        attr_names=[
            "deepseek_base_url",
            "deepseek_api_base",
            "openai_base_url",
            "base_url",
            "api_base_url",
        ],
        env_names=[
            "DEEPSEEK_BASE_URL",
            "DEEPSEEK_API_BASE",
            "OPENAI_BASE_URL",
        ],
        default="https://api.deepseek.com",
    )

    model = _read_setting(
        settings=settings,
        attr_names=[
            "model",
            "deepseek_model",
            "llm_model",
        ],
        env_names=[
            "DEEPSEEK_MODEL",
            "MODEL",
            "LLM_MODEL",
        ],
        default="deepseek-v4-flash",
    )

    if not api_key:
        raise RuntimeError(
            "无法读取 DeepSeek API Key，请检查 .env 中是否配置了 DEEPSEEK_API_KEY。"
        )

    http_client = httpx.AsyncClient(
        timeout=60.0,
        trust_env=False,
    )

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=512,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个中文家庭金融规划助手。"
                        "当前回答不是基于知识库引用，而是基于通用金融常识。"
                        "你只能做通用解释、概念说明、基础计算思路说明。"
                        "不要推荐具体股票、基金、保险产品或平台。"
                        "不要承诺收益。"
                        "不要编造引用编号。"
                        "回答要简洁、稳健、适合普通用户理解。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "用户原始问题如下：\n"
                        f"{user_message}\n\n"
                        "请直接回答用户问题。"
                        "如果这是概念解释题，优先用 1 到 3 句话说明。"
                        "不要出现 [1]、[2] 这类引用编号。"
                    ),
                },
            ],
        )

        answer = response.choices[0].message.content or ""

        usage: dict[str, Any] = {}

        if response.usage is not None:
            if hasattr(response.usage, "model_dump"):
                usage = response.usage.model_dump()
            else:
                usage = dict(response.usage)

        return answer.strip(), {
            "model": model,
            "usage": usage,
        }

    finally:
        await http_client.aclose()

async def question_router_node(
    state: FinanceAgentGraphState,
) -> dict[str, Any]:
    """
    Stage 4.1 前置问题路由节点。

    执行过程：

    1. 读取用户问题。
    2. 运行“硬规则 + DeepSeek 语义路由”。
    3. 得到一个或多个问题能力。
    4. 将能力编译成 LangGraph 执行计划。
    5. 把完整路由信息写入状态。

    安全策略：

    如果路由节点自身出现未预料异常，
    不让整个请求直接失败，而是保守地交给旧 FinanceAgent。
    """
    user_message = state.get(
        "user_message",
        "",
    ).strip()

    if not user_message:
        raise ValueError(
            "state.user_message 不能为空"
        )

    settings = get_settings()
    llm_client = _build_deepseek_client(settings)

    try:
        question_router = build_hybrid_question_router(
            llm_client=llm_client,
        )

        route_result = await question_router.route(
            user_message
        )

        capability_values = [
            capability.value
            for capability in route_result.capabilities
        ]

        execution_plan = compile_execution_plan(
            route_result.capabilities
        )

        route_detail = route_result.to_dict()

        return {
            "question_capabilities": capability_values,
            "question_router": route_result.router,
            "question_router_confidence": (
                route_result.confidence.value
            ),
            "question_router_reason": (
                route_result.reason
            ),
            "question_router_used_fallback": (
                route_result.used_fallback
            ),
            "question_router_matched_rules": list(
                route_result.matched_rules
            ),
            "question_route_detail": route_detail,
            "execution_plan": execution_plan,
        }

    except Exception as exc:
        """
        这里不能直接写入 state.error。

        因为路由失败后仍然要继续执行旧 FinanceAgent。
        如果写入全局 error，后面的质量门控会误以为
        FinanceAgent 也执行失败了。
        """
        fallback_capabilities = [
            QuestionCapability.COMPLEX_REASONING.value
        ]

        fallback_plan = compile_execution_plan(
            fallback_capabilities
        )

        router_error = (
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "question_capabilities": fallback_capabilities,
            "question_router": "question_router_node_fallback",
            "question_router_confidence": "low",
            "question_router_reason": (
                "问题路由节点执行异常，"
                "系统保守交给旧 FinanceAgent 处理。"
            ),
            "question_router_used_fallback": True,
            "question_router_matched_rules": [],
            "question_route_detail": {
                "capabilities": fallback_capabilities,
                "confidence": "low",
                "reason": (
                    "问题路由节点执行异常，"
                    "系统保守交给旧 FinanceAgent 处理。"
                ),
                "router": "question_router_node_fallback",
                "matched_rules": [],
                "used_fallback": True,
                "node_error": router_error,
            },
            "execution_plan": fallback_plan,
        }

    finally:
        await _maybe_close_client(llm_client)


async def general_finance_answer_node(
    state: FinanceAgentGraphState,
) -> dict[str, Any]:
    """
    通用金融知识直接回答节点。

    使用场景：
    问题路由器判断用户只需要 general_explanation，
    例如：

    - 什么是紧急备用金？
    - 什么是寿险缺口？
    - 资产和负债有什么区别？

    这类问题不需要知识库、计算工具、记忆或复杂规划，
    因此不进入完整 FinanceAgent，直接调用大模型回答。

    容错策略：
    如果直接回答失败，不写入全局 error，
    而是把执行计划改成 finance_agent。
    后续由 LangGraph 自动退回旧 Agent。
    """
    user_message = state.get(
        "user_message",
        "",
    ).strip()

    if not user_message:
        raise ValueError(
            "state.user_message 不能为空"
        )

    try:
        answer, answer_metadata = (
            await _call_deepseek_for_general_finance_answer(
                user_message=user_message,
            )
        )

        if not answer:
            raise RuntimeError(
                "通用金融直接回答返回了空内容"
            )

        old_usage = state.get("usage") or {}

        merged_usage = {
            **old_usage,
            "langgraph_general_finance_answer": {
                "used": True,
                **answer_metadata,
            },
        }

        old_agent_result = (
            state.get("agent_result") or {}
        )

        merged_agent_result = {
            **old_agent_result,
            "answer": answer,
            "langgraph_general_finance_answer": {
                "used": True,
                "answer": answer,
            },
        }

        return {
            "final_answer": answer,
            "agent_result": merged_agent_result,
            "executed_tools": (
                state.get("executed_tools") or []
            ),
            "usage": merged_usage,
            "finish_reason": (
                "langgraph_general_finance_answer"
            ),
            "message_count": (
                state.get("message_count") or 0
            ),
            "safety_check": (
                state.get("safety_check") or {}
            ),
            "fallback_used": False,
        }

    except Exception as exc:
        """
        不返回全局 error。

        否则后续进入 FinanceAgent 后，
        旧 error 可能仍残留在 LangGraph 状态中，
        干扰回答质量门控。
        """
        direct_answer_error = (
            f"{type(exc).__name__}: {exc}"
        )

        old_route_detail = (
            state.get("question_route_detail") or {}
        )

        updated_route_detail = {
            **old_route_detail,
            "general_finance_answer": {
                "used": False,
                "error": direct_answer_error,
                "fallback_to": FINANCE_AGENT_STEP,
            },
        }

        return {
            "final_answer": "",
            "finish_reason": (
                "general_finance_answer_error"
            ),
            "question_route_detail": (
                updated_route_detail
            ),
            "execution_plan": [
                FINANCE_AGENT_STEP,
            ],
        }


async def finance_agent_node(
    state: FinanceAgentGraphState,
) -> dict[str, Any]:
    """
    Stage 3 基础节点。

    先复用已经稳定的旧 FinanceAgent。
    """
    user_message = state.get("user_message", "").strip()
    user_id = state.get("user_id", "").strip()
    thread_id = state.get("thread_id", "").strip()

    if not user_message:
        raise ValueError("state.user_message 不能为空")

    if not user_id:
        raise ValueError("state.user_id 不能为空")

    if not thread_id:
        raise ValueError("state.thread_id 不能为空")

    request_id = state.get("request_id") or f"stage3-{uuid4()}"
    tenant_id = state.get("tenant_id") or "tenant_001"
    knowledge_base_id = state.get("knowledge_base_id") or "kb_finance_basic"
    history_messages = state.get("history_messages") or []

    settings = get_settings()
    llm_client = _build_deepseek_client(settings)

    try:
        agent = FinanceAgent(
            llm_client=llm_client,
            settings=settings,
        )

        agent_result = await agent.run(
            user_message=user_message,
            user_id=user_id,
            thread_id=thread_id,
            request_id=request_id,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            history_messages=history_messages,
        )

        result_dict = _to_plain_dict(agent_result)

        final_answer = result_dict.get("answer") or ""

        return {
            "request_id": request_id,
            "final_answer": final_answer,
            "agent_result": result_dict,
            "executed_tools": result_dict.get("executed_tools", []),
            "usage": result_dict.get("usage", {}),
            "finish_reason": result_dict.get("finish_reason", ""),
            "message_count": result_dict.get("message_count", 0),
            "safety_check": result_dict.get("safety_check", {}),
            "fallback_used": False,
        }

    except Exception as exc:
        return {
            "request_id": request_id,
            "final_answer": "",
            "agent_result": {},
            "executed_tools": [],
            "usage": {},
            "finish_reason": "error",
            "message_count": 0,
            "safety_check": {},
            "fallback_used": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    finally:
        await _maybe_close_client(llm_client)


async def answer_quality_gate_node(
    state: FinanceAgentGraphState,
) -> dict[str, Any]:
    """
    回答质量门控节点。

    当前只判断一种情况：
    知识库无证据 + 用户没有要求必须基于知识库回答。
    """
    if state.get("error"):
        return {
            "quality_gate": {
                "needs_general_finance_fallback": False,
                "reason": "上游节点已经报错，不进入兜底。",
            },
            "needs_general_finance_fallback": False,
        }

    quality_gate = build_quality_gate_result(state)

    return {
        "quality_gate": quality_gate,
        "needs_general_finance_fallback": quality_gate[
            "needs_general_finance_fallback"
        ],
        "fallback_reason": quality_gate["reason"],
    }


async def general_finance_fallback_node(
    state: FinanceAgentGraphState,
) -> dict[str, Any]:
    """
    通用金融解释兜底节点。

    只在 quality_gate 判断需要兜底时运行。
    """
    user_message = state.get("user_message", "").strip()
    fallback_reason = state.get("fallback_reason", "")

    try:
        answer, fallback_usage = await _call_deepseek_for_general_finance_answer(
            user_message=user_message,
        )

        old_usage = state.get("usage") or {}

        merged_usage = {
            **old_usage,
            "langgraph_fallback": {
                "used": True,
                "reason": fallback_reason,
                **fallback_usage,
            },
        }

        old_agent_result = state.get("agent_result") or {}

        merged_agent_result = {
            **old_agent_result,
            "langgraph_fallback": {
                "used": True,
                "reason": fallback_reason,
                "answer": answer,
            },
        }

        return {
            "final_answer": answer,
            "usage": merged_usage,
            "agent_result": merged_agent_result,
            "fallback_used": True,
            "finish_reason": "langgraph_fallback_answer",
        }

    except Exception as exc:
        old_usage = state.get("usage") or {}

        merged_usage = {
            **old_usage,
            "langgraph_fallback": {
                "used": False,
                "reason": fallback_reason,
                "error": f"{type(exc).__name__}: {exc}",
            },
        }

        return {
            "usage": merged_usage,
            "fallback_used": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
