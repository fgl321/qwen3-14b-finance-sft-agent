from __future__ import annotations

import json

import pytest

from app.agent_graph.conversation_state import (
    AuthorizedResourceRef,
    ConversationState,
    build_capability_catalog,
)
from app.agent_graph.semantic_route import (
    SemanticRouteProtocolError,
    SemanticRouter,
)


class FakeRouterClient:
    def __init__(self, payload: dict):
        self.payload = payload
        self.captured_messages: list[dict] = []

    async def chat(self, **kwargs):
        self.captured_messages = list(kwargs.get("messages") or [])
        return {
            "message": {
                "role": "assistant",
                "content": json.dumps(
                    self.payload,
                    ensure_ascii=False,
                ),
            },
            "model": "fake",
            "finish_reason": "stop",
        }


def _valid_payload(**overrides) -> dict:
    payload = {
        "orchestration_mode": "direct",
        "required_capabilities": ["complex_reasoning"],
        "task_requirements": [
            {
                "id": "T1",
                "description": "回答",
                "capabilities": ["complex_reasoning"],
                "task_kind": "reasoning",
            }
        ],
        "confidence": 0.9,
        "reason_summary": "test",
        "conversation_relation": "new_task",
        "resolved_goal": "回答",
        "task_reference": {
            "status": "none",
            "reference_type": None,
            "task_handle": None,
            "confidence": 0.0,
        },
        "pending_action_resolution": {
            "status": "none",
            "action_handle": None,
        },
        "resource_references": [],
        "result_references": [],
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_router_includes_conversation_context() -> None:
    client = FakeRouterClient(_valid_payload())
    router = SemanticRouter(
        llm_client=client,  # type: ignore[arg-type]
        max_repairs=0,
    )
    await router.route(
        "那每个月呢？",
        conversation_state=ConversationState(
            turn_count=1,
            active_task=None,
        ),
        recent_messages=[
            {"role": "user", "content": "我的年必要支出18万元。"},
        ],
        resource_catalog=[
            AuthorizedResourceRef(
                handle="DOC_1",
                resource_type="document",
                title="金融知识普及读本（第二版）",
            )
        ],
        capability_catalog=build_capability_catalog(),
        scope_snapshot={},
    )
    user_content = str(client.captured_messages[-1]["content"])
    assert "<conversation_context>" in user_content
    assert "authorized_resource_catalog" in user_content
    assert "capability_catalog" in user_content
    assert "那每个月呢？" in user_content


@pytest.mark.anyio
async def test_router_accepts_continuation_and_resource_reference() -> None:
    client = FakeRouterClient(
        _valid_payload(
            orchestration_mode="rag",
            required_capabilities=[
                "knowledge_retrieval",
                "complex_reasoning",
            ],
            task_requirements=[
                {
                    "id": "T1",
                    "description": "读取第二个文档",
                    "capabilities": [
                        "knowledge_retrieval",
                        "complex_reasoning",
                    ],
                    "requires_citations": True,
                    "task_kind": "retrieval",
                    "evidence_requirements": ["内容"],
                }
            ],
            retrieval_requirement="required",
            citation_requirement="required",
            grounding_requirement="authoritative",
            retrieval_scope="selected_documents",
            conversation_relation="follow_up",
            resolved_goal="读取第二个文档",
            resource_references=[
                {
                    "resource_type": "document",
                    "reference_form": "ordinal",
                    "selected_handles": ["DOC_2"],
                    "confidence": 0.95,
                    "status": "resolved",
                }
            ],
        )
    )
    router = SemanticRouter(
        llm_client=client,  # type: ignore[arg-type]
        max_repairs=0,
    )
    decision = await router.route(
        "第二个主要讲什么？",
        conversation_state=ConversationState(
            turn_count=1,
            focused_resources=[],
        ),
        resource_catalog=[
            AuthorizedResourceRef(
                handle="DOC_1",
                resource_type="document",
                title="金融知识普及读本（第二版）",
            ),
            AuthorizedResourceRef(
                handle="DOC_2",
                resource_type="document",
                title="平安医疗费用保险（D款）条款",
            ),
        ],
        capability_catalog=build_capability_catalog(),
    )
    assert decision.conversation_relation == "follow_up"
    assert decision.resource_references[0].status == "resolved"
    assert decision.resource_references[0].selected_handles == [
        "DOC_2"
    ]


@pytest.mark.anyio
async def test_router_rejects_contradictory_clarify() -> None:
    client = FakeRouterClient(
        _valid_payload(
            orchestration_mode="clarify",
            needs_clarification=False,
            ambiguities=["缺少主题"],
        )
    )
    router = SemanticRouter(
        llm_client=client,  # type: ignore[arg-type]
        max_repairs=0,
    )
    with pytest.raises(SemanticRouteProtocolError):
        await router.route("继续")


@pytest.mark.anyio
async def test_router_accepts_confirmation_resolution() -> None:
    client = FakeRouterClient(
        _valid_payload(
            orchestration_mode="direct",
            conversation_relation="confirmation",
            resolved_goal="确认执行资源目录查询",
            task_reference={
                "status": "resolved",
                "reference_type": "active_task",
                "task_handle": "TASK_1",
                "confidence": 0.95,
            },
            pending_action_resolution={
                "status": "confirmed",
                "action_handle": "ACTION_1",
            },
        )
    )
    router = SemanticRouter(
        llm_client=client,  # type: ignore[arg-type]
        max_repairs=0,
    )
    decision = await router.route(
        "执行",
        conversation_state=ConversationState(
            turn_count=1,
            active_task=None,
        ),
        resource_catalog=[],
        capability_catalog=build_capability_catalog(),
    )
    assert decision.conversation_relation == "confirmation"
    assert (
        decision.pending_action_resolution.status == "confirmed"
    )
    assert decision.pending_action_resolution.action_handle == (
        "ACTION_1"
    )


@pytest.mark.anyio
async def test_router_accepts_correction_relation() -> None:
    client = FakeRouterClient(
        _valid_payload(
            orchestration_mode="tool",
            required_capabilities=["financial_calculation"],
            task_requirements=[
                {
                    "id": "T1",
                    "description": "按15万重新计算",
                    "capabilities": ["financial_calculation"],
                    "evidence_tool_names": [
                        "yearly_expense_to_monthly"
                    ],
                    "task_kind": "calculation",
                }
            ],
            needs_exact_calculation=True,
            conversation_relation="correction",
            resolved_goal="放弃刚才的18万，重新按15万年度必要支出计算",
            task_reference={
                "status": "resolved",
                "reference_type": "previous_task",
                "task_handle": "TASK_1",
                "confidence": 0.9,
            },
        )
    )
    router = SemanticRouter(
        llm_client=client,  # type: ignore[arg-type]
        max_repairs=0,
    )
    decision = await router.route(
        "不算刚才那个了，重新按15万算",
        tool_catalog=[
            {
                "name": "yearly_expense_to_monthly",
                "description": "年度支出转月度",
            }
        ],
        conversation_state=ConversationState(
            turn_count=1,
            active_task=None,
        ),
        resource_catalog=[],
        capability_catalog=build_capability_catalog(),
    )
    assert decision.conversation_relation == "correction"
    assert decision.needs_exact_calculation is True
    assert "15万" in (decision.resolved_goal or "")


@pytest.mark.anyio
async def test_router_accepts_state_update_and_extracted_facts() -> None:
    client = FakeRouterClient(
        _valid_payload(
            orchestration_mode="direct",
            conversation_relation="new_task",
            state_update_only=True,
            extracted_facts=[
                {
                    "field": "cash",
                    "operation": "set",
                    "value": 900000,
                    "source": "current_turn",
                }
            ],
        )
    )
    router = SemanticRouter(
        llm_client=client,  # type: ignore[arg-type]
        max_repairs=0,
    )
    decision = await router.route(
        "我有90万现金，首付款20万。",
        resource_catalog=[],
        capability_catalog=build_capability_catalog(),
    )
    assert decision.state_update_only is True
    assert decision.extracted_facts[0].value == 900000


@pytest.mark.anyio
async def test_router_accepts_result_sub_artifact_reference() -> None:
    client = FakeRouterClient(
        _valid_payload(
            orchestration_mode="direct",
            conversation_relation="follow_up",
            resolved_goal="重复存款保险结论",
            result_references=[
                {
                    "handle": "RESULT_1",
                    "artifact_handle": "CLAIM_2",
                    "status": "resolved",
                    "confidence": 0.95,
                }
            ],
        )
    )
    router = SemanticRouter(
        llm_client=client,  # type: ignore[arg-type]
        max_repairs=0,
    )
    decision = await router.route(
        "刚才存款保险那个结论呢？",
        resource_catalog=[],
        capability_catalog=build_capability_catalog(),
    )
    assert decision.result_references[0].handle == "RESULT_1"
    assert decision.result_references[0].artifact_handle == (
        "CLAIM_2"
    )
