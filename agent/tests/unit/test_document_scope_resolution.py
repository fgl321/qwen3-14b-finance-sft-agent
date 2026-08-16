from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.api.routes.chat_graph_v2 import (
    ProductionChatRequest,
    _apply_route_scope_intent,
    _effective_document_scope,
    _extract_requested_title,
    _filter_document_candidates,
    _load_authorized_document_candidates,
    _reconcile_document_scope,
    _requirement_contract_from_floor,
    _resolve_document_scope,
    _task_aware_retrieval_query,
)
from app.agent_graph.semantic_route import (
    RequestRequirementContract,
    SemanticRouteDecision,
    TaskRequirement,
    conservative_route_fallback,
)
from app.control_plane.floor import ExplicitConstraintParser
from app.control_plane.production_adapter import (
    ControlPlaneBlocked,
    production_control_preflight,
    semantic_contract_from_route,
)
from app.control_plane.scope import resolve_resource_scope
from app.control_plane.schemas import (
    RequestedResourceScope,
    ResolvedResourceRef,
)
from app.core.config import Settings


def _row(
    document_id: str,
    *,
    title: str = "",
    file_name: str = "",
    version: str = "1",
    content_hash: str = "hash",
    aliases: list[str] | None = None,
) -> dict:
    metadata = {"visibility": "private"}
    if aliases:
        metadata["aliases"] = list(aliases)
    return {
        "document_id": document_id,
        "title": title or document_id,
        "file_name": file_name or f"{title or document_id}.pdf",
        "version": version,
        "content_hash": content_hash,
        "expired_date": None,
        "metadata": metadata,
    }


class FakeLifecycle:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def list_documents(self, **kwargs: object) -> list[dict]:
        return list(self.rows)


class FakeRagStore:
    def __init__(self, docs: list[dict]) -> None:
        self.docs = docs

    def list_documents(self, **kwargs: object) -> list[dict]:
        return list(self.docs)


def _request(rows: list[dict]) -> SimpleNamespace:
    state = SimpleNamespace(
        settings=Settings(),
        rag_document_lifecycle=FakeLifecycle(rows),
        rag_store=None,
        embedding_provider=None,
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


def _payload(message: str, **overrides: object) -> ProductionChatRequest:
    values: dict[str, object] = {
        "user_message": message,
        "thread_id": "thread-1",
        "user_id": "owner",
        "tenant_id": "personal",
        "knowledge_base_id": "kb_finance_basic",
    }
    values.update(overrides)
    return ProductionChatRequest(**values)


def _include_route(
    reference: str,
    *,
    strength: str = "required",
    exclusive: bool = False,
) -> SemanticRouteDecision:
    return SemanticRouteDecision.model_validate(
        {
            "orchestration_mode": "rag",
            "required_capabilities": ["knowledge_retrieval"],
            "task_requirements": [
                {
                    "id": "T1",
                    "description": "检索文档",
                    "capabilities": [
                        "knowledge_retrieval",
                        "citation_validation",
                    ],
                    "requires_citations": True,
                    "task_kind": "retrieval",
                }
            ],
            "retrieval_requirement": "required",
            "citation_requirement": "required",
            "grounding_requirement": "authoritative",
            "retrieval_scope": "uploaded_documents",
            "resource_constraints": {
                "include_documents": [
                    {
                        "reference": reference,
                        "reference_type": "title",
                        "strength": strength,
                    }
                ],
                "exclusive": exclusive,
            },
            "confidence": 0.9,
            "reason_summary": "test",
        }
    )


def test_floor_is_security_boundary_not_nl_parser() -> None:
    floor = ExplicitConstraintParser().parse(
        request_id="req-1",
        user_message="必须检索我上传的文档并且必须给出引用，不要联网。",
    )
    # Natural-language semantics are owned by the Semantic Router; the Floor
    # must not fabricate capability constraints from NL regexes.
    assert floor.constraints == ()
    assert floor.resource_constraints.document_exclusive is False


def test_candidate_filter_selected() -> None:
    rows = [_row("doc_a"), _row("doc_b")]
    assert [
        item["document_id"]
        for item in _filter_document_candidates(
            mode="selected",
            document_ids=["doc_a"],
            requested_title=None,
            candidates=rows,
        )
    ] == ["doc_a"]
    assert (
        _filter_document_candidates(
            mode="selected",
            document_ids=["doc_missing"],
            requested_title=None,
            candidates=rows,
        )
        == []
    )


def test_effective_document_scope_handles_raw_dict() -> None:
    payload = _payload("继续按照刚才那份分析。").model_copy(
        update={
            "document_scope": {
                "mode": "selected",
                "document_ids": ["doc_a"],
            }
        }
    )
    mode, ids = _effective_document_scope(payload)
    assert mode == "selected"
    assert ids == ["doc_a"]


def test_task_aware_retrieval_query_uses_semantic_contract() -> None:
    query = _task_aware_retrieval_query(
        "判断30万元是否适合全部用于投资并说明原因",
        "我12个月内要付首付，可用资金30万，年必要支出14.4万，请问怎么安排？",
    )
    assert query == "判断30万元是否适合全部用于投资并说明原因"

    credit_query = _task_aware_retrieval_query(
        "征信不良记录保存多久",
        "信用卡逾期已还清18个月，征信记录应该怎么处理？",
    )
    assert credit_query == "征信不良记录保存多久"


def test_task_aware_retrieval_query_no_keyword_branches() -> None:
    query = _task_aware_retrieval_query(
        "检索保险条款",
        "《平安医疗费用保险（D款）条款》的等待期、责任免除、免赔额是多少？",
    )
    assert query == "检索保险条款"

    family_query = _task_aware_retrieval_query(
        "家庭保险配置",
        "家庭寿险保障缺口怎么算？",
    )
    assert family_query == "家庭保险配置"


def test_candidate_filter_title_tiers() -> None:
    rows = [
        _row("doc_a", title="金融知识普及读本（第二版）"),
        _row("doc_b", title="金融知识普及读本（第一版）"),
    ]
    # Exact title wins even when another candidate contains the same prefix.
    assert [
        item["document_id"]
        for item in _filter_document_candidates(
            mode="missing",
            document_ids=[],
            requested_title="金融知识普及读本（第二版）",
            candidates=rows,
        )
    ] == ["doc_a"]
    # A unique contains match resolves to that candidate.
    rows2 = [
        _row("doc_a", title="金融知识普及读本（第二版）"),
        _row("doc_b", title="保险法基础"),
    ]
    assert [
        item["document_id"]
        for item in _filter_document_candidates(
            mode="missing",
            document_ids=[],
            requested_title="金融知识",
            candidates=rows2,
        )
    ] == ["doc_a"]
    # Multiple contains matches stay ambiguous (resolver decides status).
    assert (
        len(
            _filter_document_candidates(
                mode="missing",
                document_ids=[],
                requested_title="金融知识",
                candidates=rows,
            )
        )
        == 2
    )
    # File stem exact match tier.
    rows3 = [_row("doc_a", title="报告", file_name="金融知识普及读本（第二版）.pdf")]
    assert [
        item["document_id"]
        for item in _filter_document_candidates(
            mode="missing",
            document_ids=[],
            requested_title="金融知识普及读本（第二版）",
            candidates=rows3,
        )
    ] == ["doc_a"]


def test_candidate_filter_aliases_tier() -> None:
    rows = [
        _row(
            "insurance_doc",
            title="pingananzhenwuyoubaoxiantiaokuan.pdf",
            file_name="pingananzhenwuyoubaoxiantiaokuan.pdf",
            aliases=[
                "平安安诊无忧保险条款",
                "pingananzhenwuyoubaoxiantiaokuan",
            ],
        ),
        _row(
            "book_doc",
            title="金融知识普及读本（第二版）",
            file_name="jrzspjdb2.pdf",
        ),
    ]
    matched = _filter_document_candidates(
        mode="missing",
        document_ids=[],
        requested_title="平安安诊无忧保险条款",
        candidates=rows,
    )
    assert [item["document_id"] for item in matched] == [
        "insurance_doc"
    ]

    # A shared alias prefix stays ambiguous; the resolver must not guess.
    rows2 = [
        _row(
            "a",
            title="a.pdf",
            aliases=["平安保险条款A"],
        ),
        _row(
            "b",
            title="b.pdf",
            aliases=["平安保险条款B"],
        ),
    ]
    contains = _filter_document_candidates(
        mode="missing",
        document_ids=[],
        requested_title="平安保险条款",
        candidates=rows2,
    )
    assert len(contains) == 2


def test_candidate_filter_bidirectional_contains_and_canonical() -> None:
    rows = [
        _row(
            "book_doc",
            title="金融知识普及读本",
            file_name="jrzspjdb2.pdf",
            aliases=["金融知识普及读本"],
        )
    ]
    for title in [
        "金融知识普及读本（第二版）",
        "金融知识普及读本(第二版)",
        "金融知识普及读本 第二版",
        "《金融知识普及读本（第二版）》",
    ]:
        matched = _filter_document_candidates(
            mode="missing",
            document_ids=[],
            requested_title=title,
            candidates=rows,
        )
        assert [item["document_id"] for item in matched] == [
            "book_doc"
        ], title


def test_scope_algebra_matrix() -> None:
    rows = [
        _row("doc_a", title="A 文档"),
        _row("doc_b", title="B 文档"),
        _row("doc_c", title="C 文档"),
    ]

    def matched(mode: str, title: str | None, ids: list[str]) -> list[str]:
        filtered, error = _reconcile_document_scope(
            mode=mode,
            explicit_ids=ids,
            requested_title=title,
            candidates=rows,
        )
        assert error is None
        return [item["document_id"] for item in filtered]

    # all_uploaded + no title -> all active docs
    assert matched("all_uploaded", None, []) == ["doc_a", "doc_b", "doc_c"]
    # all_uploaded + named title -> narrowed to that doc
    assert matched("all_uploaded", "B 文档", []) == ["doc_b"]
    # all_uploaded + missing title -> empty (NOT_FOUND downstream)
    assert matched("all_uploaded", "不存在的文档", []) == []
    # selected A + no title / named A -> A
    assert matched("selected", None, ["doc_a"]) == ["doc_a"]
    assert matched("selected", "A 文档", ["doc_a"]) == ["doc_a"]

    # selected A + named B -> CONFLICT
    filtered, error = _reconcile_document_scope(
        mode="selected",
        explicit_ids=["doc_a"],
        requested_title="B 文档",
        candidates=rows,
    )
    assert filtered == []
    assert error == "conflict"

    # none + named title -> CONFLICT
    filtered, error = _reconcile_document_scope(
        mode="none",
        explicit_ids=[],
        requested_title="A 文档",
        candidates=rows,
    )
    assert filtered == []
    assert error == "conflict"


def test_scope_algebra_ambiguous_base_title() -> None:
    rows = [
        _row("v1", title="金融知识普及读本（第一版）"),
        _row("v2", title="金融知识普及读本（第二版）"),
    ]
    filtered, error = _reconcile_document_scope(
        mode="all_uploaded",
        explicit_ids=[],
        requested_title="金融知识普及读本",
        candidates=rows,
    )
    assert error is None
    assert {item["document_id"] for item in filtered} == {"v1", "v2"}


def test_extract_requested_title_ignores_emphasis_quotes() -> None:
    message = (
        "请结合上传文档分析，凡是文档中没有依据的内容，"
        "请明确标记为“通用金融建议”，不要伪造引用。"
    )
    assert _extract_requested_title(message) is None
    assert (
        _extract_requested_title(
            "请必须检索我上传的《金融知识普及读本（第二版）》。"
        )
        == "金融知识普及读本（第二版）"
    )


def test_resolve_resource_scope_statuses() -> None:
    requested = RequestedResourceScope(
        scope_id="uploaded_documents",
        requested_description="必须检索我上传的文档",
    )
    refs = [
        ResolvedResourceRef(
            tenant_id="personal",
            knowledge_base_id="kb",
            document_id="doc_a",
            document_version=1,
            content_hash="h",
        )
    ]
    resolved = resolve_resource_scope(
        requested=requested,
        authorized_candidates=refs,
        authorization_snapshot_id="auth:snapshot",
    )
    assert resolved.resolution_status.value == "resolved"
    ambiguous = resolve_resource_scope(
        requested=requested,
        authorized_candidates=[
            *refs,
            ResolvedResourceRef(
                tenant_id="personal",
                knowledge_base_id="kb",
                document_id="doc_b",
                document_version=1,
                content_hash="h2",
            ),
        ],
        authorization_snapshot_id="auth:snapshot",
    )
    assert ambiguous.resolution_status.value == "ambiguous"


@pytest.mark.asyncio
async def test_load_candidates_falls_back_to_qdrant() -> None:
    state = SimpleNamespace(
        settings=Settings(),
        rag_document_lifecycle=FakeLifecycle([]),
        rag_store=FakeRagStore(
            [
                {
                    "document_id": "legacy_doc",
                    "file_name": "legacy.pdf",
                    "document_version": "1",
                    "file_sha256": "hash1",
                    "visibility": "private",
                }
            ]
        ),
        embedding_provider=None,
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=state)
    )
    rows, source = await _load_authorized_document_candidates(
        request,
        tenant_id="personal",
        user_id="owner",
        knowledge_base_id="kb_finance_basic",
    )
    assert source == "qdrant_legacy_fallback"
    assert rows[0]["document_id"] == "legacy_doc"
    assert rows[0]["title"] == "legacy.pdf"


@pytest.mark.asyncio
async def test_resolve_document_scope_selected_unknown_not_found() -> None:
    plan = await _resolve_document_scope(
        request=_request([]),
        payload=_payload(
            "分析这个文档。",
            document_scope={
                "mode": "selected",
                "document_ids": ["missing_doc"],
            },
        ),
        constraints=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message="分析这个文档。",
        ),
        request_id="req-1",
    )
    assert plan["error"] is not None
    assert plan["error"].code == "DOCUMENT_SCOPE_NOT_FOUND"
    assert plan["error"].http_status == 404


@pytest.mark.asyncio
async def test_semantic_scope_ambiguous_base_title() -> None:
    rows = [
        _row("v1", title="金融知识普及读本（第一版）"),
        _row("v2", title="金融知识普及读本（第二版）"),
    ]
    plan = await _resolve_document_scope(
        request=_request(rows),
        payload=_payload("根据我上传的文档分析。"),
        constraints=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message="根据我上传的文档分析。",
        ),
        request_id="req-1",
    )
    route = SemanticRouteDecision.model_validate(
        {
            "orchestration_mode": "rag",
            "required_capabilities": ["knowledge_retrieval"],
            "task_requirements": [
                {
                    "id": "T1",
                    "description": "检索文档",
                    "capabilities": [
                        "knowledge_retrieval",
                        "citation_validation",
                    ],
                    "requires_citations": True,
                    "task_kind": "retrieval",
                }
            ],
            "retrieval_requirement": "required",
            "citation_requirement": "required",
            "grounding_requirement": "authoritative",
            "retrieval_scope": "uploaded_documents",
            "resource_constraints": {
                "include_documents": [
                    {
                        "reference": "金融知识普及读本",
                        "reference_type": "title",
                    }
                ],
                "exclusive": True,
            },
            "confidence": 0.9,
            "reason_summary": "test",
        }
    )
    updated = await _apply_route_scope_intent(
        request=_request(rows),
        payload=_payload("根据我上传的文档分析。"),
        scope_plan=plan,
        route=route,
        floor=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message="根据我上传的文档分析。",
        ),
        request_id="req-1",
    )
    assert updated["error"] is not None
    assert updated["error"].code == "DOCUMENT_SCOPE_AMBIGUOUS"


@pytest.mark.asyncio
async def test_semantic_required_include_not_found() -> None:
    rows = [_row("doc_a", title="保险法基础")]
    message = "请必须检索我上传的《金融知识普及读本（第二版）》。"
    plan = await _resolve_document_scope(
        request=_request(rows),
        payload=_payload(message),
        constraints=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message=message,
        ),
        request_id="req-1",
    )
    updated = await _apply_route_scope_intent(
        request=_request(rows),
        payload=_payload(message),
        scope_plan=plan,
        route=_include_route("金融知识普及读本（第二版）"),
        floor=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message=message,
        ),
        request_id="req-1",
    )
    assert updated["error"] is not None
    assert updated["error"].code == "DOCUMENT_SCOPE_NOT_FOUND"
    assert updated["error"].http_status == 404
    assert updated["error"].details["action"] == "select_document"


@pytest.mark.asyncio
async def test_resolve_document_scope_selected_resolved() -> None:
    rows = [_row("doc_a"), _row("doc_b")]
    plan = await _resolve_document_scope(
        request=_request(rows),
        payload=_payload(
            "必须检索我上传的文档。",
            document_scope={
                "mode": "selected",
                "document_ids": ["doc_a"],
            },
        ),
        constraints=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message="必须检索我上传的文档。",
        ),
        request_id="req-1",
    )
    assert plan["error"] is None
    assert plan["allowed_document_ids"] == ["doc_a"]
    assert plan["resolved_scope"] is not None
    assert plan["skip_answer_cache"] is True


@pytest.mark.asyncio
async def test_semantic_required_include_conflicts_with_selection() -> None:
    rows = [
        _row("insurance_doc", title="平安医疗费用保险（D款）条款"),
        _row("old_doc", title="金融知识普及读本"),
    ]
    message = "必须检索《平安医疗费用保险（D款）条款》。"
    plan = await _resolve_document_scope(
        request=_request(rows),
        payload=_payload(
            message,
            document_scope={
                "mode": "selected",
                "document_ids": ["old_doc"],
            },
        ),
        constraints=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message=message,
        ),
        request_id="req-1",
    )
    updated = await _apply_route_scope_intent(
        request=_request(rows),
        payload=_payload(
            message,
            document_scope={
                "mode": "selected",
                "document_ids": ["old_doc"],
            },
        ),
        scope_plan=plan,
        route=_include_route("平安医疗费用保险（D款）条款"),
        floor=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message=message,
        ),
        request_id="req-1",
    )
    assert updated["error"] is not None
    assert updated["error"].code == "DOCUMENT_SCOPE_CONFLICT"
    assert updated["error"].details["action"] == "select_document"


@pytest.mark.asyncio
async def test_semantic_required_include_resolves_title() -> None:
    rows = [
        _row("insurance_doc", title="平安医疗费用保险（D款）条款")
    ]
    message = "必须检索《平安医疗费用保险（D款）条款》。"
    plan = await _resolve_document_scope(
        request=_request(rows),
        payload=_payload(message),
        constraints=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message=message,
        ),
        request_id="req-1",
    )
    updated = await _apply_route_scope_intent(
        request=_request(rows),
        payload=_payload(message),
        scope_plan=plan,
        route=_include_route("平安医疗费用保险（D款）条款"),
        floor=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message=message,
        ),
        request_id="req-1",
    )
    assert updated["error"] is None
    assert updated["allowed_document_ids"] == ["insurance_doc"]
    assert updated["audit"]["source"] == "semantic_resource"


@pytest.mark.asyncio
async def test_resolve_document_scope_title_matches_selection() -> None:
    rows = [
        _row("insurance_doc", title="平安医疗费用保险（D款）条款"),
        _row("old_doc", title="金融知识普及读本"),
    ]
    message = "必须检索《平安医疗费用保险（D款）条款》。"
    plan = await _resolve_document_scope(
        request=_request(rows),
        payload=_payload(
            message,
            document_scope={
                "mode": "selected",
                "document_ids": ["insurance_doc"],
            },
        ),
        constraints=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message=message,
        ),
        request_id="req-1",
    )
    assert plan["error"] is None
    assert plan["allowed_document_ids"] == ["insurance_doc"]


@pytest.mark.asyncio
async def test_resolve_document_scope_title_without_floor_rule() -> None:
    rows = [
        _row("insurance_doc", title="平安医疗费用保险（D款）条款")
    ]
    message = "请检索《平安医疗费用保险（D款）条款》并分析责任免除。"
    plan = await _resolve_document_scope(
        request=_request(rows),
        payload=_payload(message),
        constraints=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message=message,
        ),
        request_id="req-1",
    )
    assert plan["audit"]["requested_title"] == (
        "平安医疗费用保险（D款）条款"
    )
    assert plan["error"] is None
    assert plan["allowed_document_ids"] == []
    assert plan["audit"]["needs_resolution"] is False


@pytest.mark.asyncio
async def test_route_scope_intent_mention_only_does_not_lock() -> None:
    rows = [
        _row("insurance_doc", title="平安医疗费用保险（D款）条款")
    ]
    message = "我之前在《平安医疗费用保险（D款）条款》里看过等待期，现在帮我算一下。"
    plan = await _resolve_document_scope(
        request=_request(rows),
        payload=_payload(message),
        constraints=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message=message,
        ),
        request_id="req-1",
    )
    route = SimpleNamespace(
        scope_strength="mention_only",
        retrieval_requirement="optional",
        citation_requirement="not_needed",
    )
    updated = await _apply_route_scope_intent(
        request=_request(rows),
        payload=_payload(message),
        scope_plan=plan,
        route=route,
        floor=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message=message,
        ),
        request_id="req-1",
    )
    assert updated["allowed_document_ids"] == []
    assert updated["audit"]["title_is_mention_only"] is True


@pytest.mark.asyncio
async def test_route_scope_intent_preferred_resolves_title() -> None:
    rows = [
        _row("insurance_doc", title="平安医疗费用保险（D款）条款")
    ]
    message = "《平安医疗费用保险（D款）条款》里这种产品一般适合什么人？"
    plan = await _resolve_document_scope(
        request=_request(rows),
        payload=_payload(message),
        constraints=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message=message,
        ),
        request_id="req-1",
    )
    route = SimpleNamespace(
        scope_strength="explicit_preferred",
        retrieval_requirement="preferred",
        citation_requirement="preferred",
    )
    updated = await _apply_route_scope_intent(
        request=_request(rows),
        payload=_payload(message),
        scope_plan=plan,
        route=route,
        floor=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message=message,
        ),
        request_id="req-1",
    )
    assert updated["allowed_document_ids"] == ["insurance_doc"]
    assert updated["audit"]["scope_strength"] == "explicit_preferred"


@pytest.mark.asyncio
async def test_resolve_document_scope_selection_authoritative_when_title_unmatched() -> None:
    rows = [
        _row(
            "insurance_doc",
            title="pingananzhenwuyoubaoxiantiaokuan",
        ),
        _row("old_doc", title="金融知识普及读本"),
    ]
    message = "请检索《平安医疗费用保险（D款）条款》并分析责任免除。"
    plan = await _resolve_document_scope(
        request=_request(rows),
        payload=_payload(
            message,
            document_scope={
                "mode": "selected",
                "document_ids": ["insurance_doc"],
            },
        ),
        constraints=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message=message,
        ),
        request_id="req-1",
    )
    assert plan["error"] is None
    assert plan["allowed_document_ids"] == ["insurance_doc"]


@pytest.mark.asyncio
async def test_semantic_required_include_resolves_book() -> None:
    rows = [_row("doc_a", title="金融知识普及读本（第二版）")]
    message = "请必须检索我上传的《金融知识普及读本（第二版）》。"
    plan = await _resolve_document_scope(
        request=_request(rows),
        payload=_payload(message),
        constraints=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message=message,
        ),
        request_id="req-1",
    )
    updated = await _apply_route_scope_intent(
        request=_request(rows),
        payload=_payload(message),
        scope_plan=plan,
        route=_include_route("金融知识普及读本（第二版）"),
        floor=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message=message,
        ),
        request_id="req-1",
    )
    assert updated["error"] is None
    assert updated["allowed_document_ids"] == ["doc_a"]
    assert updated["audit"]["source"] == "semantic_resource"


@pytest.mark.asyncio
async def test_semantic_mention_only_does_not_conflict_with_selection() -> None:
    rows = [
        _row("insurance_doc", title="平安医疗费用保险（D款）条款"),
        _row("book_doc", title="金融知识普及读本（第二版）"),
    ]
    message = "我之前在《金融知识普及读本（第二版）》里看过，现在帮我分析保险条款。"
    plan = await _resolve_document_scope(
        request=_request(rows),
        payload=_payload(
            message,
            document_scope={
                "mode": "selected",
                "document_ids": ["insurance_doc"],
            },
        ),
        constraints=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message=message,
        ),
        request_id="req-1",
    )
    updated = await _apply_route_scope_intent(
        request=_request(rows),
        payload=_payload(
            message,
            document_scope={
                "mode": "selected",
                "document_ids": ["insurance_doc"],
            },
        ),
        scope_plan=plan,
        route=_include_route(
            "金融知识普及读本（第二版）",
            strength="mention_only",
        ),
        floor=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message=message,
        ),
        request_id="req-1",
    )
    assert updated["error"] is None
    assert updated["allowed_document_ids"] == ["insurance_doc"]
    assert updated["audit"]["title_is_mention_only"] is True


@pytest.mark.asyncio
async def test_semantic_preferred_outside_selection_stays_selected() -> None:
    rows = [
        _row("insurance_doc", title="平安医疗费用保险（D款）条款"),
        _row("book_doc", title="金融知识普及读本（第二版）"),
    ]
    message = "如果《金融知识普及读本（第二版）》里有相关规定也可以参考。"
    plan = await _resolve_document_scope(
        request=_request(rows),
        payload=_payload(
            message,
            document_scope={
                "mode": "selected",
                "document_ids": ["insurance_doc"],
            },
        ),
        constraints=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message=message,
        ),
        request_id="req-1",
    )
    updated = await _apply_route_scope_intent(
        request=_request(rows),
        payload=_payload(
            message,
            document_scope={
                "mode": "selected",
                "document_ids": ["insurance_doc"],
            },
        ),
        scope_plan=plan,
        route=_include_route(
            "金融知识普及读本（第二版）",
            strength="preferred",
        ),
        floor=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message=message,
        ),
        request_id="req-1",
    )
    assert updated["error"] is None
    assert updated["allowed_document_ids"] == ["insurance_doc"]
    assert updated["audit"].get("resource_warnings")


@pytest.mark.asyncio
async def test_semantic_required_include_not_found_with_selection() -> None:
    rows = [_row("insurance_doc", title="平安医疗费用保险（D款）条款")]
    message = "必须根据《不存在的文档C》回答。"
    plan = await _resolve_document_scope(
        request=_request(rows),
        payload=_payload(
            message,
            document_scope={
                "mode": "selected",
                "document_ids": ["insurance_doc"],
            },
        ),
        constraints=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message=message,
        ),
        request_id="req-1",
    )
    updated = await _apply_route_scope_intent(
        request=_request(rows),
        payload=_payload(
            message,
            document_scope={
                "mode": "selected",
                "document_ids": ["insurance_doc"],
            },
        ),
        scope_plan=plan,
        route=_include_route("不存在的文档C"),
        floor=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message=message,
        ),
        request_id="req-1",
    )
    assert updated["error"] is not None
    assert updated["error"].code == "DOCUMENT_SCOPE_NOT_FOUND"


@pytest.mark.asyncio
async def test_resolve_document_scope_all_uploaded() -> None:
    rows = [_row("doc_a"), _row("doc_b")]
    plan = await _resolve_document_scope(
        request=_request(rows),
        payload=_payload(
            "分析一下这些材料。",
            document_scope={"mode": "all_uploaded", "document_ids": []},
        ),
        constraints=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message="分析一下这些材料。",
        ),
        request_id="req-1",
    )
    assert plan["error"] is None
    assert set(plan["allowed_document_ids"]) == {"doc_a", "doc_b"}


def _route_stub() -> SimpleNamespace:
    return SimpleNamespace(
        required_capabilities=["knowledge_retrieval"],
        task_requirements=[
            SimpleNamespace(
                id="t_rag",
                description="检索并引用上传文档",
                required=True,
                capabilities=("knowledge_retrieval",),
                depends_on=(),
                evidence_tool_names=(),
                requires_citations=True,
            )
        ],
        confidence=0.9,
        orchestration_mode="rag",
    )


def test_preflight_blocks_when_scoped_retrieval_missing_scope() -> None:
    message = "必须检索我上传的文档。"
    route = SemanticRouteDecision.model_validate(
        {
            "orchestration_mode": "rag",
            "required_capabilities": ["knowledge_retrieval"],
            "task_requirements": [
                {
                    "id": "T1",
                    "description": "检索文档",
                    "capabilities": ["knowledge_retrieval"],
                    "requires_citations": True,
                    "task_kind": "retrieval",
                }
            ],
            "retrieval_requirement": "required",
            "citation_requirement": "required",
            "grounding_requirement": "authoritative",
            "retrieval_scope": "uploaded_documents",
            "resource_constraints": {
                "include_documents": [
                    {
                        "reference": "A",
                        "reference_type": "title",
                    }
                ],
                "exclusive": True,
            },
            "confidence": 0.9,
            "reason_summary": "test",
        }
    )
    with pytest.raises(ControlPlaneBlocked) as exc_info:
        production_control_preflight(
            request_id="req-1",
            run_id="run-1",
            user_message=message,
            route=route,
            constraints=ExplicitConstraintParser().parse(
                request_id="req-1",
                user_message=message,
            ),
            scopes=[],
        )
    assert "SCOPE_RESOLUTION_FAILED" in exc_info.value.reason_codes


def test_preflight_passes_when_retrieval_required_without_named_scope() -> None:
    route = SemanticRouteDecision.model_validate(
        {
            "orchestration_mode": "rag",
            "required_capabilities": ["knowledge_retrieval"],
            "task_requirements": [
                {
                    "id": "T1",
                    "description": "检索文档",
                    "capabilities": ["knowledge_retrieval"],
                    "requires_citations": True,
                    "task_kind": "retrieval",
                }
            ],
            "retrieval_requirement": "required",
            "citation_requirement": "required",
            "grounding_requirement": "authoritative",
            "retrieval_scope": "uploaded_documents",
            "confidence": 0.9,
            "reason_summary": "test",
        }
    )
    audit = production_control_preflight(
        request_id="req-1",
        run_id="run-1",
        user_message="必须根据私有知识库回答这个问题。",
        route=route,
        constraints=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message="必须根据私有知识库回答这个问题。",
        ),
        scopes=[],
    )
    assert "SCOPE_RESOLUTION_FAILED" not in audit["reason_codes"]


def test_preflight_passes_with_resolved_scope() -> None:
    message = "必须检索我上传的文档。"
    requested = RequestedResourceScope(
        scope_id="uploaded_documents",
        requested_description="必须检索我上传的文档",
    )
    resolved = resolve_resource_scope(
        requested=requested,
        authorized_candidates=[
            ResolvedResourceRef(
                tenant_id="personal",
                knowledge_base_id="kb_finance_basic",
                document_id="doc_a",
                document_version=1,
                content_hash="h",
            )
        ],
        authorization_snapshot_id="auth:snapshot",
    )
    audit = production_control_preflight(
        request_id="req-1",
        run_id="run-1",
        user_message=message,
        route=_route_stub(),
        constraints=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message=message,
        ),
        scopes=[resolved],
    )
    assert audit["strategy_status"] == "ready"
    assert "SCOPE_RESOLUTION_FAILED" not in audit["reason_codes"]


def test_fallback_preserves_router_requirement_contract() -> None:
    contract = RequestRequirementContract(
        retrieval_requirement="required",
        citation_requirement="required",
        required_capabilities=[
            "knowledge_retrieval",
            "citation_validation",
        ],
        task_requirements=[
            TaskRequirement(
                id="floor_required_knowledge_retrieval",
                description="retrieve",
                capabilities=["knowledge_retrieval"],
                requires_citations=True,
                task_kind="retrieval",
            ),
            TaskRequirement(
                id="floor_required_citation_validation",
                description="validate citations",
                capabilities=["citation_validation"],
                requires_citations=True,
                task_kind="validation",
            ),
        ],
    )
    route = conservative_route_fallback(
        enable_rag=True,
        allowed_tool_groups=["financial_calculation"],
        error_type="SemanticRouteProtocolError",
        requirement_contract=contract,
    )
    assert route.retrieval_requirement == "required"
    assert route.citation_requirement == "required"
    assert "knowledge_retrieval" in route.required_capabilities
    assert "citation_validation" in route.required_capabilities

    resolved = resolve_resource_scope(
        requested=RequestedResourceScope(
            scope_id="uploaded_documents",
            requested_description="test",
        ),
        authorized_candidates=[
            ResolvedResourceRef(
                tenant_id="personal",
                knowledge_base_id="kb_finance_basic",
                document_id="doc_a",
                document_version=1,
                content_hash="h",
            )
        ],
        authorization_snapshot_id="auth:snapshot",
    )
    audit = production_control_preflight(
        request_id="req-1",
        run_id="run-1",
        user_message="必须检索我上传的文档并给出引用。",
        route=route,
        constraints=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message="必须检索我上传的文档并给出引用。",
        ),
        scopes=[resolved],
    )
    assert audit["strategy_status"] == "ready"
    assert "CAPABILITY_UNAVAILABLE" not in audit["reason_codes"]


def test_semantic_contract_exclusive_does_not_forbid_retrieval() -> None:
    route = SemanticRouteDecision.model_validate(
        {
            "orchestration_mode": "rag",
            "required_capabilities": ["knowledge_retrieval"],
            "task_requirements": [
                {
                    "id": "T1",
                    "description": "检索文档",
                    "capabilities": [
                        "knowledge_retrieval",
                        "citation_validation",
                    ],
                    "requires_citations": True,
                    "task_kind": "retrieval",
                }
            ],
            "retrieval_requirement": "required",
            "citation_requirement": "required",
            "grounding_requirement": "authoritative",
            "retrieval_scope": "uploaded_documents",
            "capability_constraints": {
                "knowledge_retrieval": "required",
                "citation_validation": "required",
            },
            "resource_constraints": {
                "include_documents": [
                    {
                        "reference": "金融知识普及读本（第二版）",
                        "reference_type": "title",
                    }
                ],
                "exclusive": True,
            },
            "confidence": 0.9,
            "reason_summary": "test",
        }
    )
    assert route.resource_constraints.exclusive is True
    contract = semantic_contract_from_route(
        request_id="req-1",
        route=route,
    )
    retrieval = [
        item
        for item in contract.constraints
        if item.capability == "knowledge_retrieval"
    ]
    assert any(
        item.requirement.value == "required"
        for item in retrieval
    )
    assert all(
        item.permission.value == "allowed"
        for item in retrieval
    )
    resolved = resolve_resource_scope(
        requested=RequestedResourceScope(
            scope_id="uploaded_documents",
            requested_description="test",
        ),
        authorized_candidates=[
            ResolvedResourceRef(
                tenant_id="personal",
                knowledge_base_id="kb_finance_basic",
                document_id="book_doc",
                document_version=1,
                content_hash="h",
            )
        ],
        authorization_snapshot_id="auth:snapshot",
    )
    audit = production_control_preflight(
        request_id="req-1",
        run_id="run-1",
        user_message="必须检索《金融知识普及读本（第二版）》并给出引用，不要使用其他文档。",
        route=route,
        constraints=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message="必须检索《金融知识普及读本（第二版）》并给出引用，不要使用其他文档。",
        ),
        scopes=[resolved],
    )
    assert audit["strategy_status"] == "ready"
    assert "CONTRACT_PERMISSION_CONFLICT" not in audit["reason_codes"]


def test_semantic_contract_real_retrieval_ban_still_forbidden() -> None:
    route = SemanticRouteDecision.model_validate(
        {
            "orchestration_mode": "direct",
            "required_capabilities": ["complex_reasoning"],
            "task_requirements": [
                {
                    "id": "answer",
                    "description": "answer",
                    "capabilities": ["complex_reasoning"],
                    "task_kind": "synthesis",
                }
            ],
            "capability_constraints": {
                "knowledge_retrieval": "forbidden"
            },
            "confidence": 0.9,
            "reason_summary": "test",
        }
    )
    contract = semantic_contract_from_route(
        request_id="req-1",
        route=route,
    )
    retrieval = [
        item
        for item in contract.constraints
        if item.capability == "knowledge_retrieval"
    ]
    assert any(
        item.permission.value == "forbidden"
        for item in retrieval
    )


def test_semantic_contract_web_ban_does_not_touch_retrieval() -> None:
    route = SemanticRouteDecision.model_validate(
        {
            "orchestration_mode": "rag",
            "required_capabilities": ["knowledge_retrieval"],
            "task_requirements": [
                {
                    "id": "T1",
                    "description": "检索文档",
                    "capabilities": ["knowledge_retrieval"],
                    "requires_citations": True,
                    "task_kind": "retrieval",
                }
            ],
            "retrieval_requirement": "required",
            "citation_requirement": "required",
            "grounding_requirement": "authoritative",
            "retrieval_scope": "uploaded_documents",
            "capability_constraints": {
                "knowledge_retrieval": "required",
                "citation_validation": "required",
                "web_search": "forbidden",
            },
            "confidence": 0.9,
            "reason_summary": "test",
        }
    )
    contract = semantic_contract_from_route(
        request_id="req-1",
        route=route,
    )
    web = [
        item
        for item in contract.constraints
        if item.capability == "web_search"
    ]
    retrieval = [
        item
        for item in contract.constraints
        if item.capability == "knowledge_retrieval"
    ]
    assert web and web[0].permission.value == "forbidden"
    assert retrieval and retrieval[0].permission.value == "allowed"
