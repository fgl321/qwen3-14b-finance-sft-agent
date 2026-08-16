from __future__ import annotations

import hashlib
from typing import Any
from uuid import uuid4

from app.agent_graph.runtime.agent_errors import (
    AgentExecutionError,
    exception_to_agent_error,
)
from app.agent_graph.runtime.request_idempotency import (
    InMemoryRequestIdempotencyStore,
    RequestIdempotencyStore,
    build_idempotency_scope_key,
    build_request_fingerprint,
)
from app.agent_graph.schemas.planner_schema import (
    ExecutionPolicy,
    normalize_execution_policy,
)


def build_checkpoint_thread_id(
    *,
    tenant_id: str,
    user_id: str,
    thread_id: str,
) -> str:
    """
    构造数据库中的 Checkpoint thread_id。

    不直接把用户 ID、租户 ID 写入 checkpoint 主键，
    而是生成稳定哈希，减少内部标识暴露。
    """

    clean_tenant_id = (
        tenant_id.strip() or "default"
    )
    clean_user_id = user_id.strip()
    clean_thread_id = thread_id.strip()

    if not clean_user_id:
        raise ValueError(
            "user_id 不能为空。"
        )

    if not clean_thread_id:
        raise ValueError(
            "thread_id 不能为空。"
        )

    raw_identity = (
        f"{clean_tenant_id}\x1f"
        f"{clean_user_id}\x1f"
        f"{clean_thread_id}"
    )

    digest = hashlib.sha256(
        raw_identity.encode("utf-8")
    ).hexdigest()

    return f"finance-agent:{digest}"


class ProductionFinanceGraphService:
    """
    最终生产 LangGraph 服务层。

    API 层只依赖该类，不直接调用 graph.ainvoke。
    """

    def __init__(
        self,
        *,
        graph: Any,
        idempotency_store: (
            RequestIdempotencyStore | None
        ) = None,
    ) -> None:
        if graph is None:
            raise ValueError(
                "graph 不能为空。"
            )

        self._graph = graph
        self._idempotency_store = (
            idempotency_store
            or InMemoryRequestIdempotencyStore()
        )

    async def run(
        self,
        *,
        user_message: str,
        user_id: str,
        thread_id: str,
        request_id: str | None = None,
        run_id: str | None = None,
        tenant_id: str = "default",
        knowledge_base_id: str = (
            "kb_finance_basic"
        ),
        history_messages: list[
            dict[str, Any]
        ] | None = None,
        context_summary: str = "",
        route_context: dict[
            str,
            Any,
        ] | None = None,
        citations: list[dict[str, Any]] | None = None,
        allowed_tool_names: list[
            str
        ] | None = None,
        allowed_tool_groups: list[
            str
        ] | None = None,
        remaining_tool_calls: int = 12,
        allow_side_effects: bool = False,
        execution_policy: ExecutionPolicy = "auto",
    ) -> dict[str, Any]:
        clean_message = user_message.strip()
        clean_user_id = user_id.strip()
        clean_thread_id = thread_id.strip()
        clean_tenant_id = (
            tenant_id.strip() or "default"
        )
        clean_knowledge_base_id = (
            knowledge_base_id.strip()
            or "kb_finance_basic"
        )
        final_execution_policy = (
            normalize_execution_policy(
                execution_policy
            )
        )

        if not clean_message:
            raise ValueError(
                "user_message 不能为空。"
            )

        if not clean_user_id:
            raise ValueError(
                "user_id 不能为空。"
            )

        if not clean_thread_id:
            raise ValueError(
                "thread_id 不能为空。"
            )

        if remaining_tool_calls < 0:
            raise ValueError(
                "remaining_tool_calls 不能小于 0。"
            )

        final_request_id = str(
            request_id
            or f"prod-request-{uuid4()}"
        ).strip()

        if not final_request_id:
            raise ValueError(
                "request_id 不能为空。"
            )

        if len(final_request_id) > 200:
            raise ValueError(
                "request_id 长度不能超过 200。"
            )

        final_history_messages = list(
            history_messages or []
        )
        final_context_summary = context_summary.strip()
        final_route_context = dict(
            route_context
            or {
                "complexity": "medium",
                "risk_level": "low",
            }
        )
        final_citations = list(citations or [])
        final_allowed_tool_names = sorted(
            {
                str(item).strip()
                for item in (allowed_tool_names or [])
                if str(item).strip()
            }
        )
        final_allowed_tool_groups = sorted(
            {
                str(item).strip()
                for item in (
                    allowed_tool_groups
                    or ["financial_calculation"]
                )
                if str(item).strip()
            }
        )

        fingerprint_payload = {
            "user_message": clean_message,
            "user_id": clean_user_id,
            "thread_id": clean_thread_id,
            "tenant_id": clean_tenant_id,
            "knowledge_base_id": (
                clean_knowledge_base_id
            ),
            "history_messages": (
                final_history_messages
            ),
            "context_summary": (
                final_context_summary
            ),
            "route_context": final_route_context,
            "citations": final_citations,
            "allowed_tool_names": (
                final_allowed_tool_names
            ),
            "allowed_tool_groups": (
                final_allowed_tool_groups
            ),
            "remaining_tool_calls": (
                remaining_tool_calls
            ),
            "allow_side_effects": (
                allow_side_effects
            ),
            "execution_policy": (
                final_execution_policy
            ),
        }

        request_fingerprint = (
            build_request_fingerprint(
                fingerprint_payload
            )
        )
        scope_key = build_idempotency_scope_key(
            tenant_id=clean_tenant_id,
            user_id=clean_user_id,
            request_id=final_request_id,
        )

        async def invoke_graph() -> dict[str, Any]:
            final_run_id = str(
                run_id
                or f"prod-run-{uuid4()}"
            ).strip()

            if not final_run_id:
                raise ValueError(
                    "run_id 不能为空。"
                )

            checkpoint_thread_id = (
                build_checkpoint_thread_id(
                    tenant_id=clean_tenant_id,
                    user_id=clean_user_id,
                    thread_id=clean_thread_id,
                )
            )

            config = {
                "configurable": {
                    "thread_id": (
                        checkpoint_thread_id
                    ),
                },
                "metadata": {
                    "request_id": final_request_id,
                    "run_id": final_run_id,
                    "tenant_id": clean_tenant_id,
                    "graph_name": (
                        "production_finance_agent"
                    ),
                    "execution_policy": (
                        final_execution_policy
                    ),
                    "request_fingerprint": (
                        request_fingerprint
                    ),
                },
                "recursion_limit": 30,
            }

            graph_input = {
                "request_id": final_request_id,
                "run_id": final_run_id,

                "user_message": clean_message,
                "user_id": clean_user_id,
                "thread_id": clean_thread_id,
                "tenant_id": clean_tenant_id,

                "knowledge_base_id": (
                    clean_knowledge_base_id
                ),

                "history_messages": (
                    final_history_messages
                ),

                "context_summary": (
                    final_context_summary
                ),

                "route_context": (
                    final_route_context
                ),

                "citations": final_citations,

                "allowed_tool_names": (
                    final_allowed_tool_names
                ),

                "allowed_tool_groups": (
                    final_allowed_tool_groups
                ),

                "execution_policy": (
                    final_execution_policy
                ),

                "remaining_tool_calls": (
                    remaining_tool_calls
                ),

                "allow_side_effects": (
                    allow_side_effects
                ),

                # 每一轮都显式清理临时执行字段，
                # 防止同一 thread 继承上一轮最终输出。
                "agent_loop_result": None,
                "final_response_result": None,
                "status": "pending",
                "final_answer": "",
                "finish_reason": "",
                "usage": {},
                "error": None,
            }

            try:
                result = await self._graph.ainvoke(
                    graph_input,
                    config=config,
                )
            except AgentExecutionError:
                raise
            except Exception as exc:
                error = exception_to_agent_error(
                    exc,
                    stage="graph",
                    request_id=final_request_id,
                    run_id=final_run_id,
                )
                raise AgentExecutionError(
                    error
                ) from exc

            return dict(result)

        execution = await self._idempotency_store.execute(
            scope_key=scope_key,
            request_fingerprint=request_fingerprint,
            operation=invoke_graph,
        )

        response = dict(execution.value)
        response["idempotency"] = {
            "request_id": final_request_id,
            "replayed": execution.replayed,
            "scope_key_hash": (
                execution.scope_key_hash
            ),
            "request_fingerprint": (
                execution.request_fingerprint
            ),
        }
        response["idempotency_replayed"] = (
            execution.replayed
        )

        return response

    async def get_checkpoint_state(
        self,
        *,
        user_id: str,
        thread_id: str,
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        checkpoint_thread_id = (
            build_checkpoint_thread_id(
                tenant_id=tenant_id,
                user_id=user_id,
                thread_id=thread_id,
            )
        )

        config = {
            "configurable": {
                "thread_id": (
                    checkpoint_thread_id
                )
            }
        }

        snapshot = await self._graph.aget_state(
            config
        )

        return dict(snapshot.values)

    async def get_checkpoint_history(
        self,
        *,
        user_id: str,
        thread_id: str,
        tenant_id: str = "default",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError(
                "limit 必须大于 0。"
            )

        checkpoint_thread_id = (
            build_checkpoint_thread_id(
                tenant_id=tenant_id,
                user_id=user_id,
                thread_id=thread_id,
            )
        )

        config = {
            "configurable": {
                "thread_id": (
                    checkpoint_thread_id
                )
            }
        }

        history: list[dict[str, Any]] = []

        async for snapshot in (
            self._graph.aget_state_history(
                config
            )
        ):
            history.append(
                {
                    "values": dict(
                        snapshot.values
                    ),
                    "next": list(
                        snapshot.next
                    ),
                    "metadata": dict(
                        snapshot.metadata
                    ),
                    "created_at": (
                        snapshot.created_at
                    ),
                    "config": dict(
                        snapshot.config
                    ),
                }
            )

            if len(history) >= limit:
                break

        return history
