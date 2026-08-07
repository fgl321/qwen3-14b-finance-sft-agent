from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.agent_graph.production_service import (
    ProductionFinanceGraphService,
)
from app.agent_graph.runtime.request_idempotency import (
    InMemoryRequestIdempotencyStore,
    RequestIdempotencyConflict,
    build_idempotency_scope_key,
    build_request_fingerprint,
)


class CountingGraph:
    def __init__(
        self,
        *,
        delay_seconds: float = 0.0,
        fail_first: bool = False,
    ) -> None:
        self.delay_seconds = delay_seconds
        self.fail_first = fail_first
        self.calls = 0

    async def ainvoke(
        self,
        graph_input: dict[str, Any],
        *,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls += 1

        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)

        if self.fail_first and self.calls == 1:
            raise RuntimeError("temporary failure")

        return {
            **graph_input,
            "status": "completed",
            "finish_reason": "test_completed",
            "final_answer": "ok",
            "graph_version": "stage_4_2_8d",
            "checkpoint_thread_id": (
                config["configurable"]["thread_id"]
            ),
        }


async def _run_service(
    service: ProductionFinanceGraphService,
    *,
    request_id: str = "request_001",
    user_message: str = "什么是紧急备用金？",
    user_id: str = "user_001",
) -> dict[str, Any]:
    return await service.run(
        request_id=request_id,
        user_message=user_message,
        user_id=user_id,
        thread_id="thread_001",
        tenant_id="tenant_001",
        execution_policy="auto",
    )


def test_request_fingerprint_is_stable() -> None:
    left = build_request_fingerprint(
        {
            "route_context": {
                "risk_level": "low",
                "complexity": "medium",
            },
            "allowed_tool_groups": {
                "knowledge_base",
                "financial_calculation",
            },
            "remaining_tool_calls": 12,
        }
    )
    right = build_request_fingerprint(
        {
            "remaining_tool_calls": 12.0,
            "allowed_tool_groups": {
                "financial_calculation",
                "knowledge_base",
            },
            "route_context": {
                "complexity": "medium",
                "risk_level": "low",
            },
        }
    )

    assert left == right


def test_scope_key_isolated_by_user_and_tenant() -> None:
    first = build_idempotency_scope_key(
        tenant_id="tenant_a",
        user_id="user_a",
        request_id="request_001",
    )
    second = build_idempotency_scope_key(
        tenant_id="tenant_a",
        user_id="user_b",
        request_id="request_001",
    )
    third = build_idempotency_scope_key(
        tenant_id="tenant_b",
        user_id="user_a",
        request_id="request_001",
    )

    assert first != second
    assert first != third


def test_sequential_duplicate_replays_first_result() -> None:
    async def scenario() -> None:
        graph = CountingGraph()
        service = ProductionFinanceGraphService(
            graph=graph
        )

        first = await _run_service(service)
        second = await _run_service(service)

        assert graph.calls == 1
        assert first["idempotency_replayed"] is False
        assert second["idempotency_replayed"] is True
        assert first["run_id"] == second["run_id"]
        assert first["final_answer"] == second["final_answer"]
        assert (
            first["idempotency"]["request_fingerprint"]
            == second["idempotency"]["request_fingerprint"]
        )

    asyncio.run(scenario())


def test_concurrent_duplicate_waits_for_first_result() -> None:
    async def scenario() -> None:
        graph = CountingGraph(delay_seconds=0.05)
        service = ProductionFinanceGraphService(
            graph=graph
        )

        first, second = await asyncio.gather(
            _run_service(service),
            _run_service(service),
        )

        assert graph.calls == 1
        assert sorted(
            [
                first["idempotency_replayed"],
                second["idempotency_replayed"],
            ]
        ) == [False, True]
        assert first["run_id"] == second["run_id"]

    asyncio.run(scenario())


def test_same_request_id_with_different_payload_conflicts() -> None:
    async def scenario() -> None:
        graph = CountingGraph()
        service = ProductionFinanceGraphService(
            graph=graph
        )

        await _run_service(
            service,
            user_message="什么是紧急备用金？",
        )

        with pytest.raises(
            RequestIdempotencyConflict
        ):
            await _run_service(
                service,
                user_message="什么是定期寿险？",
            )

        assert graph.calls == 1

    asyncio.run(scenario())


def test_failed_first_execution_is_not_cached() -> None:
    async def scenario() -> None:
        graph = CountingGraph(fail_first=True)
        store = InMemoryRequestIdempotencyStore()
        service = ProductionFinanceGraphService(
            graph=graph,
            idempotency_store=store,
        )

        with pytest.raises(RuntimeError):
            await _run_service(service)

        await asyncio.sleep(0)
        result = await _run_service(service)

        assert graph.calls == 2
        assert result["idempotency_replayed"] is False
        assert result["status"] == "completed"

    asyncio.run(scenario())


def test_same_request_id_in_different_user_scope_executes_again() -> None:
    async def scenario() -> None:
        graph = CountingGraph()
        service = ProductionFinanceGraphService(
            graph=graph
        )

        first = await _run_service(
            service,
            user_id="user_001",
        )
        second = await _run_service(
            service,
            user_id="user_002",
        )

        assert graph.calls == 2
        assert first["idempotency_replayed"] is False
        assert second["idempotency_replayed"] is False
        assert first["run_id"] != second["run_id"]

    asyncio.run(scenario())


def test_expired_completed_entry_executes_again() -> None:
    async def scenario() -> None:
        graph = CountingGraph()
        store = InMemoryRequestIdempotencyStore(
            ttl_seconds=0.01,
            max_entries=8,
        )
        service = ProductionFinanceGraphService(
            graph=graph,
            idempotency_store=store,
        )

        first = await _run_service(service)
        await asyncio.sleep(0.02)
        second = await _run_service(service)

        assert graph.calls == 2
        assert first["idempotency_replayed"] is False
        assert second["idempotency_replayed"] is False

    asyncio.run(scenario())


def test_api_maps_idempotency_conflict_to_http_409() -> None:
    from types import SimpleNamespace

    from fastapi import HTTPException, Request

    from app.api.routes.chat_graph_v2 import (
        ProductionChatRequest,
        production_chat_graph,
    )

    class ConflictService:
        async def run(self, **_: Any) -> dict[str, Any]:
            raise RequestIdempotencyConflict(
                "request id conflict"
            )

    async def scenario() -> None:
        app = SimpleNamespace(
            state=SimpleNamespace(
                production_graph_service=(
                    ConflictService()
                )
            )
        )
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/chat/graph-v2",
                "headers": [],
                "app": app,
            }
        )
        payload = ProductionChatRequest(
            request_id="request_001",
            user_message="问题A",
            user_id="user_001",
            thread_id="thread_001",
        )

        with pytest.raises(HTTPException) as exc_info:
            await production_chat_graph(
                payload,
                request,
            )

        assert exc_info.value.status_code == 409

    asyncio.run(scenario())
