from __future__ import annotations

from app.agent_graph.conversation_state import (
    build_resource_catalog,
)
from app.agent_graph.semantic_route import (
    ResourceReference,
    SemanticRouteDecision,
)
from app.api.routes.chat_graph_v2 import (
    _build_catalog_response,
    _resolved_resources_from_route,
)


def _route_with_reference(handle: str) -> SemanticRouteDecision:
    return SemanticRouteDecision(
        orchestration_mode="direct",
        required_capabilities=[
            "resource_catalog_read",
            "general_explanation",
        ],
        task_requirements=[
            {
                "id": "T1",
                "description": "查询文档目录",
                "capabilities": [
                    "resource_catalog_read",
                    "general_explanation",
                ],
                "task_kind": "reasoning",
            }
        ],
        resource_references=[
            ResourceReference(
                resource_type="document",
                reference_form="explicit",
                selected_handles=[handle],
                confidence=0.95,
                status="resolved",
            )
        ],
        confidence=0.9,
        reason_summary="test",
    )


def test_build_catalog_response_filters_by_scope() -> None:
    candidates = [
        {"document_id": "doc-a", "title": "金融读本"},
        {"document_id": "doc-b", "title": "保险条款"},
    ]
    catalog, state = build_resource_catalog(candidates)
    response = _build_catalog_response(
        request_id="request_test",
        run_id=None,
        resource_catalog=catalog,
        catalog_state=state,
        scope_plan={
            "allowed_document_ids": ["doc-a"],
            "audit": {"mode": "selected"},
        },
        semantic_route=_route_with_reference("DOC_1"),
        control_plane_audit=None,
    )
    assert response["status"] == "completed"
    assert response["finish_reason"] == "catalog_direct"
    assert response["catalog"]["document_count"] == 1
    assert response["catalog"]["documents"][0]["document_id"] == (
        "doc-a"
    )
    assert "金融读本" in response["final_answer"]


def test_resolved_resources_maps_handles_to_documents() -> None:
    candidates = [
        {"document_id": "doc-a", "title": "金融读本"},
        {"document_id": "doc-b", "title": "保险条款"},
    ]
    catalog, state = build_resource_catalog(candidates)
    resolved = _resolved_resources_from_route(
        route=_route_with_reference("DOC_2"),
        conversation_state=state,
        resource_catalog=catalog,
        allowed_document_ids=["doc-a", "doc-b"],
    )
    assert resolved == [
        {
            "handle": "DOC_2",
            "resource_type": "document",
            "title": "保险条款",
            "document_id": "doc-b",
        }
    ]
