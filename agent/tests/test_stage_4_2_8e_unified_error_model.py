from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException, Request

from app.agent_graph.production_nodes import (
    build_agent_loop_node,
    prepare_production_run_node,
)
from app.agent_graph.production_service import (
    ProductionFinanceGraphService,
)
from app.agent_graph.runtime.agent_errors import (
    AgentExecutionError,
    exception_to_agent_error,
    sanitize_error_details,
    tool_error_to_agent_error,
)
from app.agent_graph.runtime.request_idempotency import (
    RequestIdempotencyConflict,
)
from app.api.routes.chat_graph_v2 import (
    ProductionChatRequest,
    production_chat_graph,
)


def _request_with_service(service: Any) -> Request:
    app = SimpleNamespace(
        state=SimpleNamespace(
            production_graph_service=service
        )
    )

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/chat/graph-v2",
            "headers": [],
            "app": app,
        }
    )


def _payload() -> ProductionChatRequest:
    return ProductionChatRequest(
        request_id="request_001",
        user_message="什么是紧急备用金？",
        user_id="user_001",
        thread_id="thread_001",
    )


def test_sensitive_error_details_are_redacted() -> None:
    details = sanitize_error_details(
        {
            "api_key": "secret-value",
            "nested": {
                "authorization": "Bearer secret",
                "safe": "visible",
            },
        }
    )

    assert details["api_key"] == "[redacted]"
    assert (
        details["nested"]["authorization"]
        == "[redacted]"
    )
    assert details["nested"]["safe"] == "visible"


def test_idempotency_conflict_has_stable_error_code() -> None:
    error = exception_to_agent_error(
        RequestIdempotencyConflict("raw conflict"),
        stage="service",
        request_id="request_001",
    )

    assert error.code == "REQUEST_ID_CONFLICT"
    assert error.category == "conflict"
    assert error.stage == "idempotency"
    assert error.http_status == 409
    assert error.retryable is False
    assert error.request_id == "request_001"


def test_timeout_is_retryable_gateway_timeout() -> None:
    error = exception_to_agent_error(
        TimeoutError("raw timeout"),
        stage="graph",
        request_id="request_001",
        run_id="run_001",
    )

    assert error.code == "AGENT_EXECUTION_TIMEOUT"
    assert error.category == "timeout"
    assert error.http_status == 504
    assert error.retryable is True
    assert error.run_id == "run_001"


def test_internal_error_does_not_expose_raw_message() -> None:
    error = exception_to_agent_error(
        RuntimeError("database password is secret"),
        stage="graph",
    )

    assert error.code == "AGENT_INTERNAL_ERROR"
    assert error.http_status == 500
    assert "password" not in error.message.lower()
    assert "secret" not in error.message.lower()
    assert (
        error.details["exception_type"]
        == "RuntimeError"
    )


def test_tool_error_maps_to_same_envelope() -> None:
    error = tool_error_to_agent_error(
        SimpleNamespace(
            code="TOOL_TIMEOUT",
            message="工具超时。",
            model_repairable=False,
            infrastructure_retryable=True,
            details={
                "api_key": "must-not-leak",
                "dependency": "calculator",
            },
        ),
        tool_name="finance_calculator",
        request_id="request_001",
        run_id="run_001",
    )

    assert error.code == "TOOL_TIMEOUT"
    assert error.category == "timeout"
    assert error.stage == "tool"
    assert error.http_status == 504
    assert error.retryable is True
    assert (
        error.details["tool_details"]["api_key"]
        == "[redacted]"
    )


def test_service_wraps_graph_failure_with_unified_error() -> None:
    class FailedGraph:
        async def ainvoke(
            self,
            graph_input: dict[str, Any],
            *,
            config: dict[str, Any],
        ) -> dict[str, Any]:
            raise RuntimeError("raw graph failure")

    async def scenario() -> None:
        service = ProductionFinanceGraphService(
            graph=FailedGraph()
        )

        with pytest.raises(
            AgentExecutionError
        ) as exc_info:
            await service.run(
                request_id="request_001",
                user_message="问题",
                user_id="user_001",
                thread_id="thread_001",
            )

        error = exc_info.value.error

        assert error.code == "AGENT_INTERNAL_ERROR"
        assert error.stage == "graph"
        assert error.request_id == "request_001"
        assert error.run_id is not None

    asyncio.run(scenario())


def test_agent_loop_node_stores_structured_error() -> None:
    class FailedLoop:
        async def run(self, **_: Any) -> Any:
            raise TimeoutError("raw timeout")

    dependencies = SimpleNamespace(
        agent_loop=FailedLoop()
    )
    node = build_agent_loop_node(dependencies)
    state = prepare_production_run_node(
        {
            "request_id": "request_001",
            "run_id": "run_001",
            "user_message": "问题",
            "user_id": "user_001",
            "thread_id": "thread_001",
        }
    )

    result = asyncio.run(node(state))

    assert result["status"] == "fallback"
    assert result["error"]["code"] == (
        "AGENT_EXECUTION_TIMEOUT"
    )
    assert result["error"]["stage"] == "agent_loop"
    assert result["error"]["request_id"] == "request_001"


def test_api_returns_structured_conflict_detail() -> None:
    class ConflictService:
        async def run(self, **_: Any) -> dict[str, Any]:
            raise RequestIdempotencyConflict(
                "raw conflict"
            )

    async def scenario() -> None:
        with pytest.raises(HTTPException) as exc_info:
            await production_chat_graph(
                _payload(),
                _request_with_service(
                    ConflictService()
                ),
            )

        http_error = exc_info.value
        detail = http_error.detail

        assert http_error.status_code == 409
        assert detail["code"] == "REQUEST_ID_CONFLICT"
        assert detail["category"] == "conflict"
        assert detail["stage"] == "idempotency"
        assert detail["request_id"] == "request_001"
        assert (
            http_error.headers["X-Agent-Error-Code"]
            == "REQUEST_ID_CONFLICT"
        )

    asyncio.run(scenario())


def test_api_masks_internal_service_error() -> None:
    class FailedService:
        async def run(self, **_: Any) -> dict[str, Any]:
            raise RuntimeError(
                "database password is secret"
            )

    async def scenario() -> None:
        with pytest.raises(HTTPException) as exc_info:
            await production_chat_graph(
                _payload(),
                _request_with_service(FailedService()),
            )

        http_error = exc_info.value
        detail = http_error.detail

        assert http_error.status_code == 500
        assert detail["code"] == "AGENT_INTERNAL_ERROR"
        assert "secret" not in detail["message"].lower()
        assert detail["error_id"].startswith("err_")

    asyncio.run(scenario())
