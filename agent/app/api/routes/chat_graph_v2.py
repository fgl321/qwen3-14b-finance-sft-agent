from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import time
import unicodedata
from collections import OrderedDict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from app.agent_graph.runtime.agent_errors import (
    AgentExecutionError,
    build_agent_error,
    exception_to_agent_error,
    log_event,
    raise_agent_http_exception,
)
from app.agent_graph.schemas.planner_schema import ExecutionPolicy
from app.agent_graph.semantic_route import (
    DocumentReference,
    RequestRequirementContract,
    SemanticRouteProtocolError,
    SemanticRouteDecision,
    SemanticRouter,
    TaskRequirement,
    assess_task_admission,
    conservative_route_fallback,
)
from app.agent_graph.runtime.request_idempotency import (
    RequestIdempotencyConflict,
)
from app.agent_graph.conversation_state import (
    ConversationState,
    EffectiveTaskContract,
    MemoryPromotionGate,
    PolicySnapshot,
    apply_turn_patch,
    build_effective_task_contract,
    build_result_artifact,
    build_capability_catalog,
    build_resource_catalog,
    default_conversation_state,
    materialize_new_artifacts,
    resource_handles_to_document_ids,
    update_conversation_state,
)
from app.agent_graph.release_contract import PRODUCTION_RUNTIME_REVISION
from app.agent_graph.events import (
    publish_event,
    reset_event_sink,
    set_event_sink,
)
from app.core.logging import get_logger
from app.core.config import get_settings
from app.control_plane.floor import ExplicitConstraintParser
from app.control_plane.production_adapter import ControlPlaneBlocked, production_control_preflight
from app.control_plane.scope import resolve_resource_scope
from app.control_plane.enums import (
    PermissionLevel,
    RequirementLevel,
    ScopeResolutionStatus,
)
from app.control_plane.schemas import (
    RequestedResourceScope,
    ResolvedResourceRef,
    ResolvedResourceScope,
)
from app.core.request_boundary import personal_request_identity
from app.llm.synthesis_proxy import (
    current_synthesis_provider,
    reset_synthesis_provider,
    set_synthesis_provider,
)
from app.memory.llm_fact_extractor import LLMFactExtractor
from app.memory.long_term_memory import LongTermMemoryService
from app.memory.narrative_memory import (
    compress_messages_to_summary,
    narrative_segment_token_estimate,
    select_history_strategy,
)
from app.memory.raw_transcript_store import (
    RawTranscriptStore,
)
from app.memory.short_term_memory import ShortTermMemoryService
from app.personal_data.models import PERSONAL_DATA_VERSION
from app.rag.document_lifecycle import RagDocumentLifecycleService
from app.rag.query_rewriter import QueryRewriter
from app.rag.context_governance import (
    build_evidence_context,
    estimate_tokens,
    select_history_messages,
    trim_context_summary,
)
from app.rag.rag_types import SourceAuthorityContract
from app.tools.runtime_registry import build_production_tool_registry


logger = get_logger(__name__)
router = APIRouter(tags=["production-chat-graph"])


class DocumentScopePayload(BaseModel):
    """Explicit document scope selected by the caller.

    ``missing`` (the legacy ``document_ids`` absence) is represented by
    ``document_scope=None`` on the request; this model only carries explicit
    intents: selected, none (explicitly no documents) or all_uploaded.
    """

    mode: Literal["selected", "none", "all_uploaded"]
    document_ids: list[str] = Field(default_factory=list)

    @field_validator("document_ids")
    @classmethod
    def clean_document_ids(cls, value: list[str]) -> list[str]:
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if len(cleaned) > 50:
            raise ValueError("document_scope.document_ids 最多允许 50 个。")
        return list(dict.fromkeys(cleaned))

    @model_validator(mode="after")
    def validate_scope_self_consistency(self) -> "DocumentScopePayload":
        if self.mode == "selected" and not self.document_ids:
            raise ValueError("document_scope.mode=selected 时 document_ids 不能为空。")
        if self.mode in {"none", "all_uploaded"} and self.document_ids:
            raise ValueError(
                f"document_scope.mode={self.mode} 时 document_ids 必须为空。"
            )
        return self


class ProductionChatRequest(BaseModel):
    user_message: str = Field(min_length=1, max_length=12_000)
    user_id: str = Field(default="owner", min_length=1, max_length=200)
    thread_id: str = Field(min_length=1, max_length=200)
    tenant_id: str = Field(default="personal", min_length=1, max_length=200)
    knowledge_base_id: str = Field(
        default="kb_finance_basic", min_length=1, max_length=200
    )
    request_id: str | None = Field(default=None, min_length=1, max_length=200)
    history_messages: list[dict[str, Any]] = Field(default_factory=list)
    context_summary: str = ""
    route_context: dict[str, Any] = Field(default_factory=dict)
    allowed_tool_names: list[str] = Field(default_factory=list)
    allowed_tool_groups: list[str] = Field(
        default_factory=lambda: ["financial_calculation"]
    )
    remaining_tool_calls: int = Field(default=12, ge=0, le=12)
    allow_side_effects: bool = False
    execution_policy: ExecutionPolicy = "auto"

    # Stage 4.4 Lite：个人使用默认开启记忆和 RAG 能力。
    use_short_memory: bool = True
    use_long_memory: bool = True
    save_memory: bool = True
    extract_long_memory: bool = True
    enable_rag: bool = True
    rag_mode: Literal["off", "auto", "required"] = "auto"
    # 最终回答模型：qwen=本地蒸馏模型，deepseek=DeepSeek API（默认取服务端配置）。
    synthesis_llm_provider: Literal["qwen", "deepseek"] | None = Field(
        default=None
    )
    # 把检索范围限定到指定文档（“我上传的这个文档”场景）。
    document_ids: list[str] = Field(default_factory=list)
    # 显式文档作用域（新协议）。旧客户端继续使用 document_ids。
    document_scope: DocumentScopePayload | None = Field(default=None)

    @field_validator("user_message", "context_summary")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("history_messages")
    @classmethod
    def bound_history(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(value) > 40:
            raise ValueError("history_messages 最多允许 40 条。")
        return value

    @field_validator("document_ids")
    @classmethod
    def bound_document_scope(cls, value: list[str]) -> list[str]:
        if len(value) > 50:
            raise ValueError("document_ids 最多允许 50 个。")
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return list(dict.fromkeys(cleaned))

    @model_validator(mode="after")
    def bound_context(self) -> "ProductionChatRequest":
        if len(self.context_summary) > 16_000:
            raise ValueError("context_summary 不能超过 16000 个字符。")
        encoded_route = json.dumps(
            self.route_context,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        if len(encoded_route) > 16_384:
            raise ValueError("route_context 不能超过 16 KiB。")
        if self.document_scope is not None and self.document_ids:
            raise ValueError(
                "document_scope 与 document_ids 不能同时指定，请只使用 document_scope。"
            )
        return self



def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ensure_request_id(value: str | None) -> str:
    """Server-side request id: never allow None into the agent graph."""

    if value and str(value).strip():
        return str(value).strip()
    return f"api-prod-{uuid4()}"


def _scope_snapshot_hash(
    scope_snapshot: dict[str, Any] | None,
) -> str:
    """Stable hash of the resolved scope snapshot (ids + versions + hashes)."""
    if not scope_snapshot:
        return ""
    normalized = [
        {
            "document_id": str(item.get("document_id") or ""),
            "document_version": str(
                item.get("document_version") or ""
            ),
            "content_hash": str(item.get("content_hash") or ""),
        }
        for item in scope_snapshot.values()
    ]
    normalized.sort(key=lambda item: item["document_id"])
    return _canonical_hash(normalized)


def _rag_request_fingerprint(
    payload: ProductionChatRequest,
    scope_snapshot_hash: str = "",
) -> str:
    return _canonical_hash(
        {
            "tenant_id": payload.tenant_id,
            "user_id": payload.user_id,
            "thread_id": payload.thread_id,
            "knowledge_base_id": payload.knowledge_base_id,
            "user_message": payload.user_message,
            "rag_mode": payload.rag_mode,
            "enable_rag": payload.enable_rag,
            "document_ids": sorted(payload.document_ids),
            "scope_snapshot_hash": scope_snapshot_hash,
        }
    )


def _rag_attempt_cache(request: Request) -> OrderedDict:
    cache = getattr(request.app.state, "personal_rag_attempt_cache", None)
    if cache is None:
        cache = OrderedDict()
        request.app.state.personal_rag_attempt_cache = cache
    return cache


def _cached_rag_attempt(
    request: Request,
    *,
    payload: ProductionChatRequest,
    request_id: str,
    scope_snapshot_hash: str = "",
) -> dict[str, Any] | None:
    cache = _rag_attempt_cache(request)
    key = (payload.tenant_id, payload.user_id, request_id)
    item = cache.get(key)
    if item is None:
        return None
    fingerprint = _rag_request_fingerprint(
        payload,
        scope_snapshot_hash=scope_snapshot_hash,
    )
    if item["fingerprint"] != fingerprint:
        raise RequestIdempotencyConflict(
            "同一个 request_id 已经用于不同的 RAG 请求内容。"
        )
    cache.move_to_end(key)
    return copy.deepcopy(item)


def _store_rag_attempt(
    request: Request,
    *,
    payload: ProductionChatRequest,
    request_id: str,
    rag: dict[str, Any],
    run_id: str | None,
    scope_snapshot_hash: str = "",
) -> None:
    cache = _rag_attempt_cache(request)
    key = (payload.tenant_id, payload.user_id, request_id)
    cache[key] = {
        "fingerprint": _rag_request_fingerprint(
            payload,
            scope_snapshot_hash=scope_snapshot_hash,
        ),
        "rag": copy.deepcopy(rag),
        "run_id": run_id,
    }
    cache.move_to_end(key)
    while len(cache) > 2048:
        cache.popitem(last=False)


def _serialize_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _serialize_model(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_model(item) for item in value]
    return value


def _get_document_lifecycle(request: Request) -> RagDocumentLifecycleService:
    service = getattr(request.app.state, "rag_document_lifecycle", None)
    if service is not None:
        return service
    settings = getattr(request.app.state, "settings", None)
    service = RagDocumentLifecycleService(
        settings=settings,
        rag_store=getattr(request.app.state, "rag_store", None),
        embedding_provider=getattr(
            request.app.state, "embedding_provider", None
        ),
    )
    service.init_schema()
    request.app.state.rag_document_lifecycle = service
    return service


def _effective_document_scope(payload: ProductionChatRequest) -> tuple[str, list[str]]:
    """Return (mode, document_ids) after normalizing the legacy payload."""
    if payload.document_scope is not None:
        scope = payload.document_scope
        if isinstance(scope, dict):
            # Defensive: model_copy(update=...) does not coerce nested models.
            scope = DocumentScopePayload.model_validate(scope)
        return scope.mode, list(scope.document_ids)
    if payload.document_ids:
        return "selected", list(payload.document_ids)
    return "missing", []


def _get_metrics(request: Request) -> Any:
    metrics = getattr(request.app.state, "control_plane_metrics", None)
    if metrics is not None:
        return metrics
    from app.control_plane.metrics import ControlPlaneMetrics

    metrics = ControlPlaneMetrics(
        runtime_revision=PRODUCTION_RUNTIME_REVISION,
        schema_versions={"control-plane": "v2"},
    )
    request.app.state.control_plane_metrics = metrics
    return metrics


def _canonicalize_scope_text(value: Any) -> str:
    """Deterministic canonical form for document identity matching.

    NFKC unifies full-width/half-width forms (including 中文/英文括号),
    book-title marks are removed, whitespace is collapsed and case is
    normalized.  This is identity matching only; it never reinterprets which
    document the user means.
    """

    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("《", "").replace("》", "")
    return re.sub(r"\s+", "", text).strip().lower()


def _normalize_scope_text(value: Any) -> str:
    return _canonicalize_scope_text(value)


def _bidirectional_contains(left: str, right: str) -> bool:
    return bool(left and right and (left in right or right in left))


def _file_stem(value: Any) -> str:
    return Path(str(value or "")).stem


def _document_aliases(item: dict[str, Any]) -> list[str]:
    metadata = item.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    raw = metadata.get("aliases") or []
    if isinstance(raw, str):
        raw = [raw]
    return [
        str(value).strip()
        for value in raw
        if str(value).strip()
    ]


def _extract_requested_title(user_message: str) -> str | None:
    """Deterministically extract an explicitly quoted document title."""
    # Only book-title marks are treated as document titles.  Chinese quotes
    # (“…”) are too commonly used for emphasis (e.g. “通用金融建议”) and must
    # not be misread as a document reference.
    match = re.search(r"[《「]([^》」]{1,120})[》」]", user_message)
    if match is None:
        return None
    cleaned = str(match.group(1)).strip()
    return cleaned or None


def _explicit_command_with_title(user_message: str) -> bool:
    """High-confidence explicit command + quoted title, without relying on
    the word 文档/资料 (e.g. “必须检索《平安医疗费用保险（D款）条款》”)."""
    return bool(
        re.search(
            r"(?:必须|务必|严格要求).{0,12}"
            r"(?:检索|使用|结合|只根据).{0,20}[《「]",
            user_message,
        )
    )


def _document_scope_error(
    *,
    code: str,
    message: str,
    action: str,
    request_id: str,
    run_id: str | None,
    http_status: int = 422,
) -> Any:
    return build_agent_error(
        code=code,
        category=(
            "conflict"
            if code
            in {
                "DOCUMENT_SCOPE_AMBIGUOUS",
                "DOCUMENT_SCOPE_CONFLICT",
            }
            else "validation"
        ),
        stage="api",
        message=message,
        retryable=False,
        http_status=http_status,
        request_id=request_id,
        run_id=run_id,
        details={
            "reason_codes": [code],
            "user_action_required": True,
            "action": action,
            "scope_id": "uploaded_documents",
        },
    )


def _authorization_snapshot_id(
    *,
    tenant_id: str,
    user_id: str,
    knowledge_base_id: str,
    document_ids: list[str],
) -> str:
    digest = hashlib.sha256(
        json.dumps(
            sorted(document_ids),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"auth:{tenant_id}:{user_id}:{knowledge_base_id}:{digest}"


async def _load_authorized_document_candidates(
    request: Request,
    *,
    tenant_id: str,
    user_id: str,
    knowledge_base_id: str,
) -> tuple[list[dict[str, Any]], str]:
    """Load authorized candidates and return (rows, source_store).

    PostgreSQL is the primary resource authority.  For documents ingested
    before the Postgres lifecycle table existed (legacy Qdrant-only uploads),
    the resolver falls back to Qdrant payload metadata so users do not see
    their uploaded files disappear from document scope resolution.
    """
    rows: list[dict[str, Any]] = []
    source = "postgres"
    postgres_error: Exception | None = None
    try:
        lifecycle = _get_document_lifecycle(request)
        rows = await asyncio.to_thread(
            lifecycle.list_documents,
            tenant_id=tenant_id,
            owner_user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            status="active",
            limit=500,
        )
    except SemanticRouteProtocolError as exc:
        # A real router protocol failure must fail closed: Python must not
        # guess user semantics (memory/web/correction/constraints) to build a
        # substitute contract.
        logger.warning(
            "semantic_contract_unresolved",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise_agent_http_exception(
            build_agent_error(
                code="SEMANTIC_CONTRACT_UNRESOLVED",
                category="protocol",
                stage="api",
                message=(
                    "语义路由多次修复后仍无法形成一致契约，"
                    "已拒绝执行；不会用 Python 猜测用户语义。"
                ),
                retryable=True,
                http_status=422,
                request_id=request_id,
                details={
                    "reason_codes": [
                        "SEMANTIC_CONTRACT_UNRESOLVED"
                    ]
                },
            )
        )
    except Exception as exc:
        logger.warning(
            "document_scope_postgres_load_failed",
            error_type=type(exc).__name__,
        )
        postgres_error = exc
        rows = []
    if not rows:
        store = getattr(request.app.state, "rag_store", None)
        if store is not None:
            try:
                qdrant_docs = await asyncio.to_thread(
                    store.list_documents,
                    tenant_id=tenant_id,
                    owner_user_id=user_id,
                    knowledge_base_id=knowledge_base_id,
                    limit=500,
                )
            except Exception as exc:
                logger.warning(
                    "document_scope_qdrant_load_failed",
                    error_type=type(exc).__name__,
                )
                qdrant_docs = []
            if qdrant_docs:
                rows = [
                    {
                        "document_id": str(doc.get("document_id") or ""),
                        "title": str(
                            doc.get("title")
                            or doc.get("document_title")
                            or doc.get("file_name")
                            or doc.get("document_id")
                            or ""
                        ),
                        "file_name": doc.get("file_name"),
                        "version": str(
                            doc.get("document_version") or "1"
                        ),
                        "content_hash": str(
                            doc.get("file_sha256")
                            or doc.get("content_hash")
                            or ""
                        ),
                        "expired_date": None,
                        "metadata": {
                            "visibility": (
                                doc.get("visibility") or "private"
                            ),
                            "aliases": (
                                doc.get("aliases")
                                or doc.get("metadata", {}).get("aliases")
                                or []
                            ),
                        },
                    }
                    for doc in qdrant_docs
                    if doc.get("document_id")
                ]
                source = "qdrant_legacy_fallback"
    if postgres_error is not None and not rows:
        # Both sources failed: surface the dependency error instead of
        # pretending the knowledge base is empty.
        raise postgres_error
    today = date.today()
    result: list[dict[str, Any]] = []
    for row in rows:
        expired = str(row.get("expired_date") or "").strip()
        if expired:
            try:
                if date.fromisoformat(expired[:10]) < today:
                    continue
            except ValueError:
                pass
        metadata = row.get("metadata") or {}
        if isinstance(metadata, dict) and metadata.get("visibility") not in {
            None,
            "private",
        }:
            continue
        result.append(dict(row))
    return result, source


def _filter_document_candidates(
    *,
    mode: str,
    document_ids: list[str],
    requested_title: str | None,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deterministic candidate filtering by explicit ids, then title tiers."""
    if mode == "selected":
        wanted = set(document_ids)
        found = [
            item
            for item in candidates
            if str(item.get("document_id") or "") in wanted
        ]
        found_ids = {str(item.get("document_id") or "") for item in found}
        if wanted - found_ids:
            return []
        return found
    if mode == "all_uploaded":
        return list(candidates)
    if mode == "none":
        return []
    if requested_title:
        normalized = _normalize_scope_text(requested_title)
        exact_title = [
            item
            for item in candidates
            if _normalize_scope_text(item.get("title")) == normalized
        ]
        if exact_title:
            return exact_title
        exact_alias = [
            item
            for item in candidates
            if normalized
            in {
                _normalize_scope_text(alias)
                for alias in _document_aliases(item)
            }
        ]
        if exact_alias:
            return exact_alias
        exact_filename = [
            item
            for item in candidates
            if _normalize_scope_text(item.get("file_name")) == normalized
        ]
        if exact_filename:
            return exact_filename
        exact_stem = [
            item
            for item in candidates
            if _normalize_scope_text(_file_stem(item.get("file_name")))
            == normalized
        ]
        if exact_stem:
            return exact_stem
        contains = [
            item
            for item in candidates
            if normalized
            and (
                _bidirectional_contains(
                    normalized,
                    _normalize_scope_text(item.get("title")),
                )
                or _bidirectional_contains(
                    normalized,
                    _normalize_scope_text(item.get("file_name")),
                )
                or any(
                    _bidirectional_contains(
                        normalized,
                        _normalize_scope_text(alias),
                    )
                    for alias in _document_aliases(item)
                )
            )
        ]
        return contains
    return list(candidates)


def _build_scope_snapshot(
    *,
    tenant_id: str,
    knowledge_base_id: str,
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("document_id") or ""): {
            "tenant_id": tenant_id,
            "knowledge_base_id": knowledge_base_id,
            "document_id": str(row.get("document_id") or ""),
            "document_version": int(row.get("version") or 1),
            "content_hash": str(row.get("content_hash") or ""),
        }
        for row in rows
        if row.get("document_id")
    }


def _reconcile_document_scope(
    *,
    mode: str,
    explicit_ids: list[str],
    requested_title: str | None,
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    """Scope algebra over one candidate set.

    The UI/API scope is the maximum allowed resource set; natural language can
    narrow within that set but never expand it:

    - selected(A) + no title / named A       -> {A}
    - selected(A) + named B (required)       -> CONFLICT
    - all_uploaded + no title                -> all active candidates
    - all_uploaded + named B                 -> {B}
    - all_uploaded + named missing D         -> [] (NOT_FOUND downstream)
    - missing/unspecified + named title      -> title match
    - none + named title                     -> CONFLICT

    Returns (filtered_rows, error_kind) where error_kind is ``conflict``,
    ``not_found`` or ``None``.  Ambiguity is adjudicated later by
    ``resolve_resource_scope()`` when more than one row survives.
    """

    if mode == "none":
        if requested_title:
            return [], "conflict"
        return [], None

    if not requested_title:
        return (
            _filter_document_candidates(
                mode=mode,
                document_ids=explicit_ids,
                requested_title=None,
                candidates=candidates,
            ),
            None,
        )

    title_matched = _filter_document_candidates(
        mode="missing",
        document_ids=[],
        requested_title=requested_title,
        candidates=candidates,
    )
    if mode == "selected":
        if not title_matched:
            # The explicitly selected document_id is authoritative identity;
            # a mismatched or missing title must not erase the selection.
            return (
                _filter_document_candidates(
                    mode="selected",
                    document_ids=explicit_ids,
                    requested_title=None,
                    candidates=candidates,
                ),
                None,
            )
        selected_ids = set(explicit_ids)
        matched_ids = {
            str(item.get("document_id") or "")
            for item in title_matched
        }
        if selected_ids != matched_ids:
            return [], "conflict"
        return title_matched, None

    # all_uploaded / missing: a named title narrows the maximum allowed set.
    return title_matched, None


async def _resolve_document_scope(
    *,
    request: Request,
    payload: ProductionChatRequest,
    constraints: Any | None,
    request_id: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Resolve ``uploaded_documents`` to concrete authorized documents.

    This is the single scope fact source in the production chain: it loads
    authorized candidates from PostgreSQL, applies deterministic filters and
    delegates final status adjudication to ``resolve_resource_scope()``.
    """
    mode, explicit_ids = _effective_document_scope(payload)
    explicit_mode = mode not in {"missing", "none"}
    needs_resolution = explicit_mode
    base: dict[str, Any] = {
        "resolved_scope": None,
        "allowed_document_ids": [],
        "scope_requirement": (
            "optional" if explicit_mode else "none"
        ),
        "explicit_mode": explicit_mode,
        "error": None,
        "scope_snapshot": None,
        "skip_answer_cache": False,
        "audit": {
            "mode": mode,
            "needs_resolution": needs_resolution,
            "requested_title": None,
        },
    }
    requested_title = _extract_requested_title(payload.user_message)
    # The title is only a structured resource-reference candidate here.
    # Semantic strength (required/preferred/mention_only) is decided by the
    # Semantic Router; Python must not turn 《...》 into a scope decision.
    base["audit"]["requested_title"] = requested_title
    if mode in {"missing", "none"}:
        if mode == "missing":
            try:
                candidates, _candidate_source = (
                    await _load_authorized_document_candidates(
                        request,
                        tenant_id=payload.tenant_id,
                        user_id=payload.user_id,
                        knowledge_base_id=payload.knowledge_base_id,
                    )
                )
                base["authorized_candidates"] = candidates
            except Exception:
                base["authorized_candidates"] = []
        else:
            base["authorized_candidates"] = []
        return base

    try:
        candidates, candidate_source = (
            await _load_authorized_document_candidates(
                request,
                tenant_id=payload.tenant_id,
                user_id=payload.user_id,
                knowledge_base_id=payload.knowledge_base_id,
            )
        )
    except Exception as exc:
        logger.warning(
            "document_scope_candidate_load_failed",
            request_id=request_id,
            error_type=type(exc).__name__,
        )
        base["error"] = build_agent_error(
            code="DOCUMENT_SCOPE_SOURCE_UNAVAILABLE",
            category="dependency",
            stage="api",
            message="文档授权服务暂时不可用，请稍后重试。",
            retryable=True,
            http_status=503,
            request_id=request_id,
            run_id=run_id,
            details={"reason_codes": ["DOCUMENT_SCOPE_SOURCE_UNAVAILABLE"]},
        )
        return base

    base["authorized_candidates"] = candidates

    filtered, reconcile_error = _reconcile_document_scope(
        mode=mode,
        explicit_ids=explicit_ids,
        requested_title=None,
        candidates=candidates,
    )
    if reconcile_error == "conflict":
        base["error"] = _document_scope_error(
            code="DOCUMENT_SCOPE_CONFLICT",
            message=(
                "当前选中的文档与您点名要求的文档不一致，"
                "请选择或上传对应文档。"
                if mode == "selected"
                else "您点名要求文档，但又指定不使用文档，请重新选择。"
            ),
            action="select_document",
            request_id=request_id,
            run_id=run_id,
            http_status=422,
        )
        return base
    refs = [
        ResolvedResourceRef(
            tenant_id=payload.tenant_id,
            knowledge_base_id=payload.knowledge_base_id,
            document_id=str(row.get("document_id") or ""),
            document_version=int(row.get("version") or 1),
            content_hash=str(row.get("content_hash") or ""),
        )
        for row in filtered
        if row.get("document_id")
    ]
    source_text = requested_title or ""
    requested = RequestedResourceScope(
        scope_id="uploaded_documents",
        source_constraint_ids=tuple(
            item.source.constraint_id
            for item in (
                constraints.constraints
                if constraints is not None
                else ()
            )
        ),
        requested_description=(source_text or payload.user_message)[:500],
    )
    authorization_snapshot_id = _authorization_snapshot_id(
        tenant_id=payload.tenant_id,
        user_id=payload.user_id,
        knowledge_base_id=payload.knowledge_base_id,
        document_ids=[item.document_id for item in refs],
    )
    if mode in {"selected", "all_uploaded"}:
        # Explicit selections (a concrete id set or "all uploaded") are by
        # definition unambiguous.  The count-based resolver only adjudicates
        # natural-language/title resolution.
        if refs:
            resolved = ResolvedResourceScope(
                scope_id=requested.scope_id,
                requested_scope_hash=(
                    requested.canonical_hash or requested.calculate_hash()
                ),
                resources=tuple(refs),
                allowed_source_types=requested.allowed_source_types,
                forbidden_source_types=requested.forbidden_source_types,
                web_access=requested.web_access,
                authorization_snapshot_id=authorization_snapshot_id,
                resolved_at_utc=datetime.now(timezone.utc),
                resolution_status=ScopeResolutionStatus.RESOLVED,
            )
            resolved = resolved.model_copy(
                update={"canonical_hash": resolved.calculate_hash()}
            )
            status = resolved.resolution_status
        else:
            resolved = resolve_resource_scope(
                requested=requested,
                authorized_candidates=(),
                authorization_snapshot_id=authorization_snapshot_id,
            )
            status = resolved.resolution_status
    else:
        resolved = resolve_resource_scope(
            requested=requested,
            authorized_candidates=refs,
            authorization_snapshot_id=authorization_snapshot_id,
        )
        status = resolved.resolution_status
    status = resolved.resolution_status
    base["resolved_scope"] = resolved
    base["audit"].update(
        {
            "source_store": candidate_source,
            "candidate_count": len(candidates),
            "filtered_count": len(refs),
            "resolution_status": status.value,
            "authorization_snapshot_id": resolved.authorization_snapshot_id,
            "source": (
                "explicit_ids"
                if mode == "selected"
                else "all_uploaded"
                if mode == "all_uploaded"
                else "title"
                if requested_title
                else "auto_single_candidate"
                if len(candidates) == 1
                else "candidates"
            ),
        }
    )
    if status == ScopeResolutionStatus.RESOLVED:
        _get_metrics(request).increment(
            "scope_resolution_resolved_total"
        )
        allowed_ids = [item.document_id for item in refs]
        base["allowed_document_ids"] = allowed_ids
        base["scope_snapshot"] = _build_scope_snapshot(
            tenant_id=payload.tenant_id,
            knowledge_base_id=payload.knowledge_base_id,
            rows=filtered,
        )
        base["skip_answer_cache"] = True
        return base

    explicit_reference = mode == "selected" or bool(requested_title)
    if status == ScopeResolutionStatus.AMBIGUOUS:
        code = "DOCUMENT_SCOPE_AMBIGUOUS"
        message = "检测到多份可用文档，请选择本次需要使用的文档。"
        action = "select_document"
        http_status = 422
    elif explicit_reference:
        code = "DOCUMENT_SCOPE_NOT_FOUND"
        message = "未找到你指定的可用文档，请重新选择文档后重试。"
        action = "select_document"
        http_status = 404
    else:
        code = "DOCUMENT_SCOPE_EMPTY"
        message = (
            "你要求必须使用上传文档，但当前知识库中没有可用文档。"
            "请先上传文档后重试。"
        )
        action = "upload_document"
        http_status = 422
    _get_metrics(request).increment(
        {
            "DOCUMENT_SCOPE_AMBIGUOUS": (
                "scope_resolution_ambiguous_total"
            ),
            "DOCUMENT_SCOPE_NOT_FOUND": (
                "scope_resolution_not_found_total"
            ),
            "DOCUMENT_SCOPE_EMPTY": (
                "scope_resolution_empty_total"
            ),
            "DOCUMENT_SCOPE_CONFLICT": (
                "scope_resolution_conflict_total"
            ),
        }[code]
    )
    base["error"] = _document_scope_error(
        code=code,
        message=message,
        action=action,
        request_id=request_id,
        run_id=run_id,
        http_status=http_status,
    )
    return base


def _resolve_document_reference(
    *,
    reference: Any,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deterministically resolve one typed document reference."""

    ref = getattr(reference, "reference", None)
    ref_type = getattr(reference, "reference_type", "title")
    if not ref:
        return []
    if ref_type == "document_id":
        return _filter_document_candidates(
            mode="selected",
            document_ids=[str(ref)],
            requested_title=None,
            candidates=candidates,
        )
    return _filter_document_candidates(
        mode="missing",
        document_ids=[],
        requested_title=str(ref),
        candidates=candidates,
    )


def _apply_resource_refs(
    *,
    base_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    include_refs: list[Any],
    exclude_refs: list[Any],
    exclusive: bool,
    mode: str,
    explicit_ids: list[str],
) -> tuple[list[dict[str, Any]], str | None, list[str]]:
    """Apply typed resource constraints (include/exclude/exclusive).

    UI/API scope is the maximum allowed set; the semantic contract can only
    narrow it.  Returns (allowed_rows, error_kind, warnings) where error_kind
    is ``conflict`` / ``not_found`` / ``ambiguous`` / ``None``.
    """

    base_ids = {
        str(row.get("document_id") or "")
        for row in base_rows
    }
    warnings: list[str] = []

    include_ids: set[str] = set()
    required_include_ids: set[str] = set()
    for ref in include_refs:
        strength = str(getattr(ref, "strength", "required") or "required")
        if strength == "mention_only":
            continue
        matched = _resolve_document_reference(
            reference=ref,
            candidates=candidates,
        )
        if not matched:
            if strength == "required":
                return [], "not_found", warnings
            warnings.append(
                f"preferred_resource_unavailable:{ref.reference}"
            )
            continue
        if len(matched) > 1:
            if strength == "required":
                return [], "ambiguous", warnings
            warnings.append(
                f"preferred_resource_ambiguous:{ref.reference}"
            )
            continue
        document_id = str(matched[0].get("document_id") or "")
        include_ids.add(document_id)
        if strength == "required":
            required_include_ids.add(document_id)

    if include_ids:
        if mode == "selected" and not required_include_ids.issubset(base_ids):
            return [], "conflict", warnings
        dropped_preferred = include_ids - base_ids
        if dropped_preferred:
            warnings.append(
                "preferred_resource_outside_scope:"
                + ",".join(sorted(dropped_preferred))
            )
            include_ids -= dropped_preferred
        allowed = [
            row
            for row in base_rows
            if str(row.get("document_id") or "") in include_ids
        ]
    else:
        allowed = list(base_rows)

    exclude_ids: set[str] = set()
    for ref in exclude_refs:
        matched = _resolve_document_reference(
            reference=ref,
            candidates=candidates,
        )
        if not matched:
            warnings.append(
                f"exclude_resource_unavailable:{ref.reference}"
            )
            continue
        if len(matched) > 1:
            warnings.append(
                f"exclude_resource_ambiguous:{ref.reference}"
            )
            continue
        exclude_ids.add(str(matched[0].get("document_id") or ""))
    if exclude_ids:
        allowed = [
            row
            for row in allowed
            if str(row.get("document_id") or "") not in exclude_ids
        ]

    if exclusive and not allowed:
        return [], "not_found", warnings
    return allowed, None, warnings


async def _apply_route_scope_intent(
    *,
    request: Request,
    payload: ProductionChatRequest,
    scope_plan: dict[str, Any],
    route: SemanticRouteDecision,
    floor: Any,
    request_id: str,
    resolved_document_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Post-router scope intent from the typed semantic resource contract.

    The Semantic Router owns natural-language resource semantics; this
    function only resolves typed references against the authorized registry.
    """

    if scope_plan.get("error") is not None:
        return scope_plan

    resource = getattr(route, "resource_constraints", None)
    include_refs = list(
        getattr(resource, "include_documents", None) or []
    )
    exclude_refs = list(
        getattr(resource, "exclude_documents", None) or []
    )
    exclusive = bool(getattr(resource, "exclusive", False))
    mention_only_refs = [
        ref
        for ref in include_refs
        if str(getattr(ref, "strength", "required") or "required")
        == "mention_only"
    ]
    active_include_refs = [
        ref
        for ref in include_refs
        if str(getattr(ref, "strength", "required") or "required")
        != "mention_only"
    ]
    if resolved_document_ids:
        # Python already resolved the router's semantic handles to real
        # document ids and validated them against the authorization snapshot
        # and the caller scope.  These ids become the scope fact source so
        # title/alias re-resolution cannot fail on shorthand names.
        active_include_refs = [
            DocumentReference(
                reference=document_id,
                reference_type="document_id",
                strength="required",
            )
            for document_id in resolved_document_ids
        ]
        include_refs = [
            *mention_only_refs,
            *active_include_refs,
        ]
        exclusive = True
        scope_plan["audit"]["source"] = (
            "semantic_resource_resolved_handles"
        )
    scope_plan["audit"]["document_exclusive"] = exclusive
    requested_title = scope_plan.get("audit", {}).get(
        "requested_title"
    )
    mode, explicit_ids = _effective_document_scope(payload)

    if route.scope_strength != "semantic_inferred":
        strength = route.scope_strength
    elif (
        route.retrieval_requirement in {"required", "preferred"}
        or route.citation_requirement in {"required", "preferred"}
    ):
        strength = "explicit_preferred"
    else:
        strength = "mention_only"
    scope_plan["audit"]["scope_strength"] = strength
    if (
        strength == "mention_only"
        and not active_include_refs
        and not exclude_refs
        and not exclusive
    ):
        scope_plan["audit"]["title_is_mention_only"] = True
        return scope_plan
    if (
        mention_only_refs
        and not active_include_refs
        and not exclude_refs
        and not exclusive
    ):
        scope_plan["audit"]["title_is_mention_only"] = True
        return scope_plan

    if (
        not active_include_refs
        and not exclude_refs
        and not exclusive
    ):
        if mode == "selected":
            scope_plan["audit"]["selection_authoritative"] = True
            return scope_plan
        if not requested_title:
            return scope_plan

    candidates, candidate_source = (
        await _load_authorized_document_candidates(
            request,
            tenant_id=payload.tenant_id,
            user_id=payload.user_id,
            knowledge_base_id=payload.knowledge_base_id,
        )
    )

    if scope_plan.get("resolved_scope") is not None:
        resolved_ids = {
            item.document_id
            for item in scope_plan["resolved_scope"].resources
        }
        base_rows = [
            row
            for row in candidates
            if str(row.get("document_id") or "") in resolved_ids
        ]
    else:
        base_rows = _filter_document_candidates(
            mode=mode,
            document_ids=explicit_ids,
            requested_title=None,
            candidates=candidates,
        )

    allowed_rows: list[dict[str, Any]]
    error_kind: str | None
    warnings: list[str] = []
    if active_include_refs or exclude_refs or exclusive:
        allowed_rows, error_kind, warnings = _apply_resource_refs(
            base_rows=base_rows,
            candidates=candidates,
            include_refs=active_include_refs,
            exclude_refs=exclude_refs,
            exclusive=exclusive,
            mode=mode,
            explicit_ids=explicit_ids,
        )
    elif requested_title:
        matched = _filter_document_candidates(
            mode="missing",
            document_ids=[],
            requested_title=requested_title,
            candidates=candidates,
        )
        if not matched:
            scope_plan["audit"]["title_unresolved"] = True
            return scope_plan
        allowed_rows = matched
        error_kind = None
    else:
        allowed_rows = base_rows
        error_kind = None
    if warnings:
        scope_plan["audit"]["resource_warnings"] = warnings

    if error_kind == "conflict":
        scope_plan["audit"]["source"] = "semantic_resource"
        scope_plan["error"] = _document_scope_error(
            code="DOCUMENT_SCOPE_CONFLICT",
            message=(
                "当前选中的文档与您点名要求的文档不一致，"
                "请选择或上传对应文档。"
            ),
            action="select_document",
            request_id=request_id,
            run_id=None,
            http_status=422,
        )
        return scope_plan
    if error_kind == "not_found":
        scope_plan["audit"]["source"] = "semantic_resource"
        scope_plan["error"] = _document_scope_error(
            code="DOCUMENT_SCOPE_NOT_FOUND",
            message="未找到你指定的可用文档，请重新选择文档后重试。",
            action="select_document",
            request_id=request_id,
            run_id=None,
            http_status=404,
        )
        return scope_plan
    if error_kind == "ambiguous":
        scope_plan["audit"]["source"] = "semantic_resource"
        scope_plan["error"] = _document_scope_error(
            code="DOCUMENT_SCOPE_AMBIGUOUS",
            message="检测到多份可用文档，请选择本次需要使用的文档。",
            action="select_document",
            request_id=request_id,
            run_id=None,
            http_status=422,
        )
        return scope_plan

    refs = [
        ResolvedResourceRef(
            tenant_id=payload.tenant_id,
            knowledge_base_id=payload.knowledge_base_id,
            document_id=str(row.get("document_id") or ""),
            document_version=int(row.get("version") or 1),
            content_hash=str(row.get("content_hash") or ""),
        )
        for row in allowed_rows
        if row.get("document_id")
    ]
    requested = RequestedResourceScope(
        scope_id="uploaded_documents",
        requested_description=(
            str(
                (include_refs[0].reference if include_refs else None)
                or requested_title
                or payload.user_message
            )[:500]
        ),
    )
    resolved = resolve_resource_scope(
        requested=requested,
        authorized_candidates=refs,
        authorization_snapshot_id=_authorization_snapshot_id(
            tenant_id=payload.tenant_id,
            user_id=payload.user_id,
            knowledge_base_id=payload.knowledge_base_id,
            document_ids=[item.document_id for item in refs],
        ),
    )
    if resolved.resolution_status != ScopeResolutionStatus.RESOLVED:
        scope_plan["audit"]["title_unresolved"] = True
        return scope_plan
    scope_plan["resolved_scope"] = resolved
    scope_plan["allowed_document_ids"] = [
        item.document_id for item in refs
    ]
    scope_plan["scope_snapshot"] = _build_scope_snapshot(
        tenant_id=payload.tenant_id,
        knowledge_base_id=payload.knowledge_base_id,
        rows=allowed_rows,
    )
    scope_plan["skip_answer_cache"] = True
    scope_plan["scope_requirement"] = "optional"
    scope_plan["audit"].update(
        {
            "source_store": candidate_source,
            "source": (
                "semantic_resource"
                if active_include_refs or exclude_refs or exclusive
                else "semantic_title"
            ),
            "resolution_status": resolved.resolution_status.value,
        }
    )
    return scope_plan


def _rag_sufficient(rag: dict[str, Any]) -> bool:
    assessment = rag.get("evidence_assessment") or {}
    return bool(assessment.get("sufficient"))


def _rag_has_citable_support(rag: dict[str, Any] | None) -> bool:
    if not rag:
        return False
    assessment = dict(rag.get("evidence_assessment") or {})
    return bool(
        rag.get("citations")
        and assessment.get("support_level")
        in {"direct_support", "partial_support", "background_support"}
    )


def _route_allows_rag_direct(route: SemanticRouteDecision) -> bool:
    """RAG direct is only safe for retrieval-only requests."""

    if route.orchestration_mode != "rag" or route.needs_exact_calculation:
        return False
    if route.confidence <= 0 or any(
        str(item).startswith("semantic_router_degraded:")
        for item in route.ambiguities
    ):
        return False
    forbidden = {
        "financial_calculation",
        "complex_reasoning",
        "memory_read",
    }
    if forbidden & set(route.required_capabilities):
        return False
    if any(
        task.required and task.evidence_tool_names
        for task in route.task_requirements
    ):
        return False
    return all(
        set(task.capabilities)
        <= {"knowledge_retrieval", "citation_validation", "general_explanation"}
        for task in route.task_requirements
        if task.required
    )


def _legacy_rag_direct_is_safe(payload: ProductionChatRequest, request: Request) -> bool:
    """Narrow compatibility path when the semantic router is unavailable.

    Explicit required RAG can still return/refuse from verified retrieval.
    Auto mode is limited to an unmistakable knowledge-base question without
    numeric calculation signals. This does not weaken the v2 execution path.
    """
    if getattr(request.app.state, "deepseek", None) is not None:
        return False
    if payload.rag_mode == "required":
        return True
    message = payload.user_message.lower()
    retrieval_signal = any(term in message for term in ("知识库", "文档", "资料", "knowledge base"))
    calculation_signal = any(char.isdigit() for char in message) or any(
        term in message for term in ("计算", "测算", "比例", "缺口")
    )
    return payload.rag_mode == "auto" and retrieval_signal and not calculation_signal


def _rag_direct_execution_path(
    rag: dict[str, Any],
    payload: ProductionChatRequest,
) -> str:
    """细分 RAG 直接回答的来源路径。

    - attachment_direct：用户明确指定文档（文档问答，即使来源是生成内容也允许）；
    - kb_direct：知识库常规直接回答（权威证据充分）。
    """
    chunks = rag.get("retrieved_chunks") or []
    positional_fallback = any(
        str((chunk.get("metadata") or {}).get("retrieval_mode", ""))
        == "document_scope_positional_fallback"
        for chunk in chunks
    )
    if positional_fallback:
        return "attachment_direct"
    scoped_ids = {
        str(document_id)
        for document_id in (payload.document_ids or [])
    }
    if scoped_ids:
        chunk_doc_ids = {
            str(chunk.get("document_id") or "")
            for chunk in chunks
        }
        if chunk_doc_ids and chunk_doc_ids <= scoped_ids:
            return "attachment_direct"
    return "kb_direct"


def _build_rag_direct_result(
    *,
    payload: ProductionChatRequest,
    request_id: str,
    rag: dict[str, Any],
    run_id: str,
    replayed: bool,
    scope_snapshot_hash: str = "",
) -> dict[str, Any]:
    sufficient = _rag_sufficient(rag)
    fingerprint = _rag_request_fingerprint(
        payload,
        scope_snapshot_hash=scope_snapshot_hash,
    )
    return {
        "request_id": request_id,
        "run_id": run_id,
        "user_id": payload.user_id,
        "thread_id": payload.thread_id,
        "tenant_id": payload.tenant_id,
        "knowledge_base_id": payload.knowledge_base_id,
        "user_message": payload.user_message,
        "status": "completed",
        "final_answer": str(rag.get("answer") or "").strip(),
        "finish_reason": (
            "rag_direct_answer"
            if sufficient
            else "rag_evidence_insufficient"
        ),
        "usage": rag.get("usage") or {},
        "error": None,
        # 保留原生产图版本，另用 execution_path/personal_data_version
        # 表示本次请求由 Stage 4.4 RAG 快速路径完成。
        "graph_version": "stage_4_2_8f",
        "personal_data_version": PERSONAL_DATA_VERSION,
        "execution_path": _rag_direct_execution_path(
            rag,
            payload,
        ),
        "rag": rag,
        "synthesis_llm_provider": current_synthesis_provider(),
        "idempotency": {
            "request_id": request_id,
            "replayed": replayed,
            "scope_key_hash": _canonical_hash(
                [payload.tenant_id, payload.user_id, request_id]
            )[:24],
            "request_fingerprint": fingerprint,
        },
        "idempotency_replayed": replayed,
    }


def _rag_answer_redis(request: Request) -> Any | None:
    """返回用于 RAG 答案缓存的 Redis 客户端，不可用时返回 None。"""
    short_memory = _get_short_memory(request)
    redis = getattr(short_memory, "redis", None)
    return redis if redis is not None else None


def _rag_kb_fingerprint(request: Request) -> int:
    """知识库指纹：任意文档增删都会改变，用于缓存失效。"""
    store = getattr(request.app.state, "rag_store", None)
    if store is None:
        return 0
    try:
        return int(store.count_points())
    except Exception:
        return 0


def _rag_answer_cache_key(
    *,
    payload: ProductionChatRequest,
    retrieval_query: str,
    provider: str,
    kb_fingerprint: int,
    scope_snapshot_hash: str = "",
) -> str:
    raw = "|".join(
        [
            payload.tenant_id,
            payload.user_id,
            payload.knowledge_base_id,
            ",".join(sorted(payload.document_ids)),
            payload.user_message,
            retrieval_query,
            provider,
            str(kb_fingerprint),
            scope_snapshot_hash,
            PRODUCTION_RUNTIME_REVISION,
        ]
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"rag_answer:{digest}"


def _rag_pipeline_metrics(rag: dict[str, Any] | None) -> dict[str, Any]:
    """Expose retrieval stages without leaking document text into traces."""
    if not rag:
        return {
            "retrieved_count": 0,
            "reranked_count": 0,
            "evidence_candidate_count": 0,
            "sufficient_evidence_count": 0,
            "citable_evidence_count": 0,
            "evidence_support_level": "irrelevant",
            "citation_count": 0,
            "provisional_citation_count": 0,
            "retrieval_status": "not_run",
            "rerank_status": "not_run",
            "evidence_assessment_status": "not_run",
            "conflict_detection_status": "not_run",
            "protocol_error_stage": None,
            "evidence_rejection_reason": None,
        }
    chunks = list(rag.get("retrieved_chunks") or [])
    stages = dict(rag.get("stage_status") or {})
    assessment = dict(rag.get("evidence_assessment") or {})
    relevant = list(assessment.get("relevant_evidence_numbers") or [])
    coverage = list(rag.get("requirement_coverage") or [])
    missing_coverage = [
        item
        for item in coverage
        if item.get("status")
        not in {
            "direct_support",
            "partial_support",
            "background_support",
        }
    ]
    reranked = sum(
        1
        for chunk in chunks
        if isinstance(chunk, dict)
        and any(
            key in (chunk.get("metadata") or {})
            for key in (
                "rerank_probability",
                "rerank_score",
                "rrf_score",
                "retrieval_mode",
            )
        )
    )
    sufficient = bool(assessment.get("sufficient"))
    support_level = str(
        assessment.get("support_level")
        or ("direct_support" if sufficient else "irrelevant")
    )
    return {
        "retrieved_count": len(chunks),
        "reranked_count": reranked or len(chunks),
        "evidence_candidate_count": len(chunks),
        "sufficient_evidence_count": len(relevant) if sufficient else 0,
        "citable_evidence_count": len(relevant),
        "evidence_support_level": support_level,
        "citation_count": len(rag.get("citations") or []),
        "provisional_citation_count": len(
            rag.get("provisional_citations") or []
        ),
        "retrieval_status": stages.get("retrieval_status", "completed"),
        "rerank_status": stages.get("rerank_status", "completed"),
        "evidence_assessment_status": stages.get(
            "evidence_assessment_status", "completed"
        ),
        "conflict_detection_status": stages.get(
            "conflict_detection_status", "completed"
        ),
        "protocol_error_stage": stages.get("protocol_error_stage"),
        "error_code": (
            "assessor_protocol_error"
            if stages.get("evidence_assessment_status") == "protocol_failed"
            else "assessor_service_error"
            if stages.get("evidence_assessment_status") == "service_failed"
            else None
        ),
        "retryable": False,
        "evidence_rejection_reason": (
            None
            if sufficient
            or (coverage and not missing_coverage)
            else str(assessment.get("reason") or "") or None
        ),
        "requirement_coverage": coverage,
        "retrieval_retry_count": int(
            ((rag.get("usage") or {}).get("retrieval_retry_count") or 0)
        ),
        "evidence_pipeline": rag.get("evidence_pipeline"),
        "coverage_integrity": rag.get("coverage_integrity"),
    }


def _normalize_query_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _build_logical_evidence_requirements(
    route: SemanticRouteDecision,
) -> dict[str, dict[str, Any]]:
    """Logical evidence requirements are the semantic contract of RAG.

    They are derived once from the router's task requirements and must never
    be removed by query dedup/merge/retry optimizations.
    """

    requirements: dict[str, dict[str, Any]] = {}
    for task in route.task_requirements:
        if (
            not task.required
            or task.task_kind != "retrieval"
            or "knowledge_retrieval" not in task.capabilities
        ):
            continue
        if task.evidence_requirements:
            for index, requirement in enumerate(
                task.evidence_requirements, start=1
            ):
                requirement_id = f"{task.id}:{index}"
                requirements[requirement_id] = {
                    "id": requirement_id,
                    "task_id": task.id,
                    "description": str(requirement),
                    "query": f"{task.description}：{requirement}",
                    "required": True,
                }
        else:
            requirements[str(task.id)] = {
                "id": str(task.id),
                "task_id": task.id,
                "description": task.description,
                "query": task.description,
                "required": True,
            }
    return requirements


def _build_physical_queries(
    requirements: dict[str, dict[str, Any]],
    user_message: str,
) -> tuple[list[dict[str, Any]], int, int]:
    """Build physical queries and merge exact-duplicate executions.

    Dedup merges provenance: a merged query keeps the union of
    ``source_requirement_ids`` so no logical requirement can silently vanish.
    """

    raw_queries: list[dict[str, str]] = [
        {
            "requirement_id": requirement_id,
            "query": requirement["query"],
        }
        for requirement_id, requirement in requirements.items()
    ]
    if len(raw_queries) <= 1:
        topic_specs = (
            ("4321", "4321定律"),
            ("双十", "家庭保险 双十定律"),
            ("三一", "房贷 三一定律 月供 家庭月收入"),
            ("80定律", "80定律 年龄 风险资产比例"),
            ("72定律", "72定律 收益率 翻倍年数"),
            ("存款保险", "存款保险 最高偿付限额 本金 利息 同一投保机构 超过限额"),
            ("住房贷款", "住房贷款 还款方式 等额本金 等额本息 提前还款"),
            ("不良记录", "个人征信 不良信息 保存期限 起算点 超过期限删除"),
            ("保存期", "个人征信 不良信息 保存期限 起算点 超过期限删除"),
            ("征信", "征信报告 查询授权 同意 隐私保护 异议处理"),
            ("隐私", "征信查询 授权同意 个人隐私保护 查询目的"),
            ("国债", "国债 储蓄国债 凭证式 电子式 风险 收益 特征"),
        )
        known_topics = [
            query for marker, query in topic_specs if marker in user_message
        ]
        known_topics = list(dict.fromkeys(known_topics))
        if len(known_topics) > 1 and raw_queries:
            # Legacy topic expansion is a PHYSICAL query group only: every
            # expanded query keeps the original logical requirement id, so
            # document_topic_n can never replace the Requirement Universe.
            base_requirement_id = str(
                raw_queries[0]["requirement_id"]
            )
            raw_queries = [
                {
                    "requirement_id": base_requirement_id,
                    "query": topic,
                }
                for topic in known_topics
            ]

    by_normalized: dict[str, dict[str, Any]] = {}
    physical_queries: list[dict[str, Any]] = []
    merged_count = 0
    for index, item in enumerate(raw_queries, start=1):
        query_id = f"Q{index}"
        normalized = _normalize_query_text(item["query"])
        existing = by_normalized.get(normalized)
        if existing is not None:
            existing["source_requirement_ids"] = sorted(
                set(existing["source_requirement_ids"])
                | {str(item["requirement_id"])}
            )
            existing["merged_from_query_ids"] = sorted(
                set(existing["merged_from_query_ids"]) | {query_id}
            )
            merged_count += 1
            continue
        physical = {
            "id": query_id,
            "query": item["query"],
            "requirement_id": str(item["requirement_id"]),
            "source_requirement_ids": [str(item["requirement_id"])],
            "merged_from_query_ids": [],
        }
        by_normalized[normalized] = physical
        physical_queries.append(physical)
    return physical_queries, len(raw_queries), merged_count


def _retrieval_queries(
    route: SemanticRouteDecision,
    user_message: str,
) -> list[dict[str, Any]]:
    requirements = _build_logical_evidence_requirements(route)
    query_text = (
        str(route.resolved_goal or "").strip()
        or user_message
    )
    physical_queries, _, _ = _build_physical_queries(
        requirements,
        query_text,
    )
    return physical_queries


_COVERAGE_STATUS_RANK = {
    "direct_support": 7,
    "partial_support": 6,
    "background_support": 5,
    "insufficient_evidence": 4,
    "irrelevant": 3,
    "conflict": 2,
    "assessment_protocol_failed": 1,
    "service_failed": 1,
    "technical_unavailable": 1,
    "not_observed": 0,
}


def _coverage_status_from_rag(rag: dict[str, Any]) -> str:
    assessment = dict(rag.get("evidence_assessment") or {})
    local_stages = dict(rag.get("stage_status") or {})
    assessor_status = str(
        local_stages.get("evidence_assessment_status") or "not_run"
    )
    if assessor_status == "protocol_failed":
        return "assessment_protocol_failed"
    if assessor_status == "service_failed":
        return "service_failed"
    if assessment.get("evidence_conflicts"):
        return "conflict"
    if assessment.get("sufficient"):
        return "direct_support"
    support = str(assessment.get("support_level") or "irrelevant")
    return support if support in _COVERAGE_STATUS_RANK else "irrelevant"


def _merge_rag_requirement_results(
    items: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    """Merge physical query results and fan out coverage per logical
    requirement.

    One physical query may serve many logical requirements; the EvidencePool
    (candidate chunks/citations) is shared, but every source requirement gets
    its own RequirementObservation entry so none can silently disappear.
    """

    chunks: list[dict[str, Any]] = []
    seen_chunks: set[str] = set()
    citations: list[dict[str, Any]] = []
    provisional_citations: list[dict[str, Any]] = []
    coverage_by_id: dict[str, dict[str, Any]] = {}
    answers: list[str] = []
    for physical_query, rag in items:
        source_ids = [
            str(item)
            for item in (
                physical_query.get("source_requirement_ids")
                or [physical_query.get("requirement_id")]
            )
            if str(item)
        ]
        query_id = str(
            physical_query.get("id")
            or physical_query.get("requirement_id")
            or "query"
        )
        query_text = str(physical_query.get("query") or "")
        assessment = dict(rag.get("evidence_assessment") or {})
        status = _coverage_status_from_rag(rag)
        local_citations = list(rag.get("citations") or [])
        local_conflict_entries = [
            {**conflict, "requirement_id": source_id}
            for source_id in source_ids
            for conflict in (assessment.get("evidence_conflicts") or [])
        ]
        for source_id in source_ids:
            existing = coverage_by_id.get(source_id)
            if existing is not None:
                existing["source_query_ids"] = sorted(
                    set(existing["source_query_ids"]) | {query_id}
                )
                existing["citation_ids"] = sorted(
                    set(existing["citation_ids"])
                    | {
                        int(citation.get("citation_id") or 0)
                        for citation in local_citations
                        if citation.get("citation_id") is not None
                    }
                )
                existing["conflict_ids"] = sorted(
                    set(existing["conflict_ids"])
                    | {
                        int(conflict.get("conflict_id") or 0)
                        for conflict in local_conflict_entries
                        if conflict.get("conflict_id") is not None
                    }
                )
                existing["citation_count"] = len(existing["citation_ids"])
                existing["retrieved_count"] = int(
                    existing.get("retrieved_count") or 0
                ) + len(rag.get("retrieved_chunks") or [])
                if (
                    _COVERAGE_STATUS_RANK.get(status, 0)
                    > _COVERAGE_STATUS_RANK.get(
                        str(existing["status"]),
                        0,
                    )
                ):
                    existing["status"] = status
                    existing["assessor_status"] = str(
                        (rag.get("stage_status") or {}).get(
                            "evidence_assessment_status"
                        )
                        or "completed"
                    )
                continue
            coverage_by_id[source_id] = {
                "requirement_id": source_id,
                "task_id": source_id.split(":", 1)[0],
                "query": query_text,
                "query_id": query_id,
                "source_query_ids": [query_id],
                "status": status,
                "citation_count": len(local_citations),
                "retrieved_count": len(rag.get("retrieved_chunks") or []),
                "citation_ids": [
                    int(citation.get("citation_id") or 0)
                    for citation in local_citations
                    if citation.get("citation_id") is not None
                ],
                "conflict_ids": [
                    int(conflict.get("conflict_id") or 0)
                    for conflict in local_conflict_entries
                    if conflict.get("conflict_id") is not None
                ],
                "retryable": status
                in {
                    "direct_support",
                    "partial_support",
                    "background_support",
                    "insufficient_evidence",
                    "irrelevant",
                },
                "assessor_status": str(
                    (rag.get("stage_status") or {}).get(
                        "evidence_assessment_status"
                    )
                    or "completed"
                ),
            }
        for chunk in rag.get("retrieved_chunks") or []:
            chunk_id = str(chunk.get("chunk_id") or "")
            if chunk_id and chunk_id in seen_chunks:
                continue
            if chunk_id:
                seen_chunks.add(chunk_id)
            chunks.append(chunk)
        for citation in local_citations:
            item = dict(citation)
            item["citation_id"] = len(citations) + 1
            metadata = dict(item.get("metadata") or {})
            metadata["requirement_id"] = source_ids[0] if source_ids else None
            metadata["requirement_ids"] = source_ids
            item["metadata"] = metadata
            citations.append(item)
        for citation in rag.get("provisional_citations") or []:
            item = dict(citation)
            item["citation_id"] = len(provisional_citations) + 1
            metadata = dict(item.get("metadata") or {})
            metadata["requirement_id"] = source_ids[0] if source_ids else None
            metadata["requirement_ids"] = source_ids
            metadata["verification_status"] = "provisional_unassessed"
            item["metadata"] = metadata
            provisional_citations.append(item)
        if str(rag.get("answer") or "").strip():
            answers.append(str(rag["answer"]).strip())
    coverage = list(coverage_by_id.values())
    complete = bool(coverage) and all(
        item["status"] == "direct_support" for item in coverage
    )
    satisfied = bool(coverage) and all(
        item["status"]
        in {
            "direct_support",
            "partial_support",
            "background_support",
            "insufficient_evidence",
            "irrelevant",
        }
        for item in coverage
    )
    has_support = any(item["citation_count"] > 0 for item in coverage)
    support_level = (
        "direct_support" if complete
        else "partial_support" if has_support
        else "irrelevant"
    )
    stage_items = [dict(rag.get("stage_status") or {}) for _, rag in items]
    assessment_statuses = {
        str(item.get("evidence_assessment_status") or "not_run")
        for item in stage_items
    }
    failed_coverage = [
        item
        for item in coverage
        if item.get("status")
        in {
            "assessment_protocol_failed",
            "service_failed",
            "technical_unavailable",
            "not_observed",
        }
    ]
    satisfied_coverage = [
        item
        for item in coverage
        if item.get("status")
        in {
            "direct_support",
            "partial_support",
            "background_support",
            "insufficient_evidence",
            "irrelevant",
        }
    ]
    if "protocol_failed" in assessment_statuses:
        merged_assessment_status = (
            "partial_protocol_failure"
            if failed_coverage and satisfied_coverage
            else "protocol_failed"
        )
    elif "service_failed" in assessment_statuses:
        merged_assessment_status = (
            "partial_protocol_failure"
            if failed_coverage and satisfied_coverage
            else "service_failed"
        )
    elif "repaired" in assessment_statuses:
        merged_assessment_status = "repaired"
    else:
        merged_assessment_status = "completed"
    return {
        "query": "multi_requirement_retrieval",
        "answer": "\n\n".join(answers),
        "evidence_assessment": {
            "sufficient": complete,
            "support_level": support_level,
            "confidence": "high" if coverage else "low",
            "reason": (
                "all retrieval requirements covered"
                if satisfied
                else "one or more retrieval requirements lack acceptable support"
            ),
            "relevant_evidence_numbers": list(range(1, len(chunks) + 1)),
            "direct_evidence_numbers": [],
            "partial_evidence_numbers": list(range(1, len(chunks) + 1)),
            "background_evidence_numbers": [],
            "missing_info": [
                item["requirement_id"]
                for item in coverage
                if item["status"]
                not in {
                    "direct_support",
                    "partial_support",
                    "background_support",
                    "insufficient_evidence",
                    "irrelevant",
                }
            ],
        },
        "citations": citations,
        "provisional_citations": provisional_citations,
        "retrieved_chunks": chunks,
        "stage_status": {
            "retrieval_status": "completed",
            "rerank_status": (
                "completed"
                if any(item.get("rerank_status") == "completed" for item in stage_items)
                else "not_run"
            ),
            "evidence_assessment_status": merged_assessment_status,
            "conflict_detection_status": (
                "completed"
                if merged_assessment_status in {"completed", "repaired"}
                else "not_run"
            ),
            "retrieved_count": len(chunks),
            "reranked_count": sum(
                int(item.get("reranked_count") or 0) for item in stage_items
            ),
            "candidate_count": len(chunks),
            "protocol_failed_task_ids": [
                str(item.get("requirement_id") or "")
                for item in failed_coverage
            ],
            "protocol_error_stage": (
                "evidence_sufficiency_assessor"
                if failed_coverage
                else None
            ),
        },
        "requirement_coverage": coverage,
        "usage": {"decomposed_queries": len(items)},
        "evidence_conflicts": [
            conflict
            for _, rag in items
            for conflict in (
                (rag.get("evidence_assessment") or {}).get(
                    "evidence_conflicts"
                )
                or []
            )
        ],
    }


def _enforce_direct_support_citation_binding(
    coverage: list[dict[str, Any]],
    citations: list[dict[str, Any]],
) -> list[str]:
    """direct_support must be backed by a citation the assessor marked
    direct_support for that requirement; otherwise downgrade."""

    direct_by_requirement: dict[str, list[dict[str, Any]]] = {}
    for citation in citations:
        metadata = dict(citation.get("metadata") or {})
        requirement_ids = metadata.get("requirement_ids") or (
            [metadata["requirement_id"]]
            if metadata.get("requirement_id")
            else []
        )
        if metadata.get("support_level") != "direct_support":
            continue
        for requirement_id in requirement_ids:
            direct_by_requirement.setdefault(
                str(requirement_id),
                [],
            ).append(citation)
    downgraded: list[str] = []
    for item in coverage:
        if item.get("status") != "direct_support":
            continue
        requirement_id = str(item.get("requirement_id") or "")
        if direct_by_requirement.get(requirement_id):
            continue
        item["status"] = "partial_support"
        item["reason"] = "direct_support_requires_direct_citation"
        downgraded.append(requirement_id)
    return downgraded


def _finalize_requirement_coverage(
    rag: dict[str, Any],
    logical_requirement_ids: list[str],
    *,
    raw_query_count: int | None = None,
    merged_query_count: int | None = None,
) -> dict[str, Any]:
    """Deterministic coverage integrity check before Completion Contract."""

    required_ids = {str(item) for item in logical_requirement_ids}
    coverage = list(rag.get("requirement_coverage") or [])
    by_id = {
        str(item.get("requirement_id") or ""): item
        for item in coverage
    }
    original_observed_ids = set(by_id)
    missing_before_synthesis = sorted(required_ids - original_observed_ids)
    for requirement_id in sorted(required_ids):
        if requirement_id in by_id:
            continue
        by_id[requirement_id] = {
            "requirement_id": requirement_id,
            "task_id": requirement_id.split(":", 1)[0],
            "status": "not_observed",
            "source_query_ids": [],
            "citation_ids": [],
            "conflict_ids": [],
            "assessor_status": "coverage_integrity_violation",
            "reason": "coverage_integrity_violation",
            "retryable": False,
            "citation_count": 0,
            "retrieved_count": 0,
            "query": None,
            "query_id": None,
        }
    coverage = list(by_id.values())
    downgraded = _enforce_direct_support_citation_binding(
        coverage,
        list(rag.get("citations") or []),
    )
    observed_ids = {
        str(item.get("requirement_id") or "")
        for item in coverage
    }
    missing = sorted(required_ids - observed_ids)
    status_counts: dict[str, int] = {}
    for item in coverage:
        status = str(item.get("status") or "not_observed")
        status_counts[status] = status_counts.get(status, 0) + 1
    rag["requirement_coverage"] = coverage
    rag["coverage_integrity"] = {
        "status": (
            "ok"
            if not missing_before_synthesis
            else "violated"
        ),
        "missing_observation_ids": missing_before_synthesis,
        "downgraded_direct_support_ids": downgraded,
    }
    rag["evidence_pipeline"] = {
        "logical_requirement_count": len(required_ids),
        "required_logical_requirement_count": len(required_ids),
        "raw_query_count": int(raw_query_count or 0),
        "physical_query_count": len(
            rag.get("physical_queries") or []
        ),
        "merged_query_count": int(merged_query_count or 0),
        "requirement_observation_count": len(coverage),
        "required_observation_count": sum(
            1
            for item in coverage
            if str(item.get("requirement_id") or "") in required_ids
        ),
        "direct_support_count": status_counts.get("direct_support", 0),
        "partial_support_count": status_counts.get(
            "partial_support",
            0,
        ),
        "insufficient_evidence_count": status_counts.get(
            "insufficient_evidence",
            0,
        ),
        "irrelevant_count": status_counts.get("irrelevant", 0),
        "technical_unavailable_count": (
            status_counts.get("technical_unavailable", 0)
            + status_counts.get("assessment_protocol_failed", 0)
            + status_counts.get("service_failed", 0)
        ),
        "not_observed_count": status_counts.get("not_observed", 0),
        "missing_observation_ids": missing_before_synthesis,
    }
    return rag


async def _retry_missing_retrieval_requirements(
    *,
    rag_service: Any,
    payload: ProductionChatRequest,
    rag: dict[str, Any],
    document_ids: list[str] | None = None,
) -> dict[str, Any]:
    coverage = list(rag.get("requirement_coverage") or [])
    missing = [
        item for item in coverage
        if item.get("retryable", True)
        and item.get("status")
        in {
            "insufficient_evidence",
            "irrelevant",
            "assessment_protocol_failed",
            "service_failed",
            "technical_unavailable",
        }
    ]
    if not missing:
        return rag

    async def retry(
        item: dict[str, Any],
        index: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        query = _task_aware_retrieval_query(
            str(item.get("query") or ""),
            payload.user_message,
        )
        raw = await rag_service.answer(
            query=query,
            retrieval_query=query,
            tenant_id=payload.tenant_id,
            owner_user_id=payload.user_id,
            knowledge_base_id=payload.knowledge_base_id,
            document_ids=(
                list(document_ids)
                if document_ids is not None
                else list(payload.document_ids)
            ),
            relevance_gate=None,
        )
        return (
            {
                "id": f"Q-retry-{index}",
                "requirement_id": str(
                    item.get("requirement_id") or "retry"
                ),
                "query": query,
                "source_requirement_ids": [
                    str(item.get("requirement_id") or "retry")
                ],
                "merged_from_query_ids": [],
            },
            _serialize_model(raw),
        )

    retried = list(
        await asyncio.gather(
            *(retry(item, index) for index, item in enumerate(missing, 1))
        )
    )
    coverage_by_id = {
        str(item.get("requirement_id") or ""): item
        for item in coverage
    }
    seen_chunks = {
        str(chunk.get("chunk_id") or "")
        for chunk in (rag.get("retrieved_chunks") or [])
        if chunk.get("chunk_id")
    }
    citations = list(rag.get("citations") or [])
    for physical_query, raw_rag in retried:
        source_ids = [
            str(item)
            for item in (
                physical_query.get("source_requirement_ids")
                or [physical_query.get("requirement_id")]
            )
            if str(item)
        ]
        query_id = str(physical_query.get("id") or "retry")
        status = _coverage_status_from_rag(raw_rag)
        for source_id in source_ids:
            entry = coverage_by_id.get(source_id)
            if entry is None:
                entry = {
                    "requirement_id": source_id,
                    "task_id": source_id.split(":", 1)[0],
                    "status": status,
                    "source_query_ids": [],
                    "citation_ids": [],
                    "conflict_ids": [],
                    "citation_count": 0,
                    "retrieved_count": 0,
                    "retryable": False,
                    "query": physical_query.get("query"),
                    "query_id": query_id,
                    "assessor_status": str(
                        (raw_rag.get("stage_status") or {}).get(
                            "evidence_assessment_status"
                        )
                        or "completed"
                    ),
                }
                coverage_by_id[source_id] = entry
            entry["source_query_ids"] = sorted(
                set(entry["source_query_ids"]) | {query_id}
            )
            entry["citation_ids"] = sorted(
                set(entry["citation_ids"])
                | {
                    int(citation.get("citation_id") or 0)
                    for citation in (raw_rag.get("citations") or [])
                    if citation.get("citation_id") is not None
                }
            )
            entry["citation_count"] = len(entry["citation_ids"])
            entry["retrieved_count"] = int(
                entry.get("retrieved_count") or 0
            ) + len(raw_rag.get("retrieved_chunks") or [])
            entry["query"] = physical_query.get("query")
            entry["query_id"] = query_id
            if (
                _COVERAGE_STATUS_RANK.get(status, 0)
                >= _COVERAGE_STATUS_RANK.get(
                    str(entry["status"]),
                    0,
                )
            ):
                entry["status"] = status
                entry["assessor_status"] = str(
                    (raw_rag.get("stage_status") or {}).get(
                        "evidence_assessment_status"
                    )
                    or "completed"
                )
            entry["retryable"] = status in {
                "direct_support",
                "partial_support",
                "background_support",
                "insufficient_evidence",
                "irrelevant",
            }
        for citation in raw_rag.get("citations") or []:
            item = dict(citation)
            item["citation_id"] = len(citations) + 1
            metadata = dict(item.get("metadata") or {})
            metadata["requirement_id"] = source_ids[0] if source_ids else None
            metadata["requirement_ids"] = source_ids
            item["metadata"] = metadata
            citations.append(item)
        for chunk in raw_rag.get("retrieved_chunks") or []:
            chunk_id = str(chunk.get("chunk_id") or "")
            if chunk_id and chunk_id in seen_chunks:
                continue
            if chunk_id:
                seen_chunks.add(chunk_id)
            rag.setdefault("retrieved_chunks", []).append(chunk)
    rag["requirement_coverage"] = list(coverage_by_id.values())
    rag["citations"] = citations
    rag["usage"]["retrieval_retry_count"] = len(retried)
    return rag


def _task_aware_retrieval_query(
    query: str,
    user_message: str,
) -> str:
    """Build a retrieval query that fits the task instead of a fixed legal
    suffix.  The old generic suffix (“原文 条款 规定 期限 金额 起算点”) was
    designed for contract/regulation retrieval and actively hurt family
    finance questions by shifting the search toward irrelevant legal terms.
    """
    # Retrieval queries come from the resolved semantic contract (Router
    # evidence requirements + resolved_goal); keyword-based suffix branches are
    # removed in favor of the LLM-decided evidence requirements.
    return query.strip()
    parts = [query.strip()]
    if any(
        marker in user_message
        for marker in ("首付", "买房", "购房", "住房", "房贷")
    ):
        parts.append("住房首付款 短期资金 通知存款 流动性 投资风险")
    if any(
        marker in user_message
        for marker in (
            "家庭理财",
            "家庭",
            "资产配置",
            "资金安排",
            "资金",
            "理财",
            "储蓄",
            "投资",
        )
    ):
        parts.append("家庭理财 资金预留 大额支出规划 储蓄 风险投资比例 4321定律 80定律")
    medical_insurance_markers = (
        "条款",
        "医疗",
        "等待期",
        "责任免除",
        "免赔",
        "给付比例",
        "医院",
        "既往症",
        "理赔",
        "保险责任",
        "补偿原则",
    )
    is_medical_insurance = any(
        marker in user_message
        for marker in medical_insurance_markers
    )
    if is_medical_insurance:
        parts.append(
            "保险责任 等待期 医院 必要且合理 免赔额 "
            "给付比例 社会医疗保险 责任免除 既往症"
        )
    elif any(
        marker in user_message
        for marker in ("家庭保险", "寿险", "保额", "保障缺口", "保险配置", "保费")
    ):
        parts.append("保险 寿险 保障 双十定律 三一定律")
    elif any(
        marker in user_message
        for marker in ("保险", "保障")
    ):
        parts.append("保险 保障")
    if any(
        marker in user_message
        for marker in ("征信", "信用", "逾期")
    ):
        parts.append("征信 不良记录 保存期限 起算点")
    if any(
        marker in user_message
        for marker in ("贷款", "等额本息", "等额本金")
    ):
        parts.append("住房贷款 还款方式 等额本息 等额本金")
    return " ".join(dict.fromkeys(parts)).strip()


def _assert_retrieval_within_scope(
    rag: dict[str, Any],
    allowed_document_ids: list[str],
) -> None:
    """Fail closed when retrieval returns chunks outside the resolved scope."""
    allowed = {str(item) for item in (allowed_document_ids or [])}
    if not allowed:
        return
    for chunk in rag.get("retrieved_chunks") or []:
        document_id = str(chunk.get("document_id") or "")
        if document_id and document_id not in allowed:
            raise AgentExecutionError(
                build_agent_error(
                    code="RETRIEVAL_SCOPE_VIOLATION",
                    category="internal",
                    stage="service",
                    message="检索结果超出已解析的文档范围。",
                    retryable=False,
                    http_status=500,
                    details={"document_id": document_id},
                )
            )


def _citation_scope_violations(
    rag: dict[str, Any] | None,
    scope_snapshot: dict[str, Any] | None,
) -> list[str]:
    """Return citation document ids that escape the resolved scope."""
    if not rag or not scope_snapshot:
        return []
    allowed = set(scope_snapshot.keys())
    violations: list[str] = []
    for citation in rag.get("citations") or []:
        document_id = str(citation.get("document_id") or "")
        if not document_id:
            continue
        if document_id not in allowed:
            violations.append(document_id)
            continue
        entry = scope_snapshot.get(document_id) or {}
        citation_version = str(
            citation.get("document_version")
            or citation.get("version")
            or ""
        ).strip()
        snapshot_version = str(
            entry.get("document_version") or ""
        ).strip()
        if (
            citation_version
            and snapshot_version
            and citation_version != snapshot_version
        ):
            violations.append(f"{document_id}@{citation_version}")
    return violations


async def _run_rag_attempt(
    *,
    payload: ProductionChatRequest,
    request: Request,
    request_id: str,
    history_messages: list[dict[str, Any]],
    allow_direct: bool = True,
    required_failure_is_fatal: bool = True,
    retrieval_queries: list[dict[str, str]] | None = None,
    skip_answer_cache: bool = False,
    scope_snapshot_hash: str = "",
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any] | None]:
    audit: dict[str, Any] = {
        "attempted": False,
        "mode": payload.rag_mode,
        "sufficient": False,
        "replayed": False,
        "degraded": False,
    }
    if not payload.enable_rag or payload.rag_mode == "off":
        return None, audit, None

    cached = _cached_rag_attempt(
        request,
        payload=payload,
        request_id=request_id,
        scope_snapshot_hash=scope_snapshot_hash,
    )
    if cached is not None:
        rag = dict(cached["rag"])
        sufficient = _rag_sufficient(rag)
        audit.update(
            {
                "attempted": True,
                "sufficient": sufficient,
                "replayed": True,
                **_rag_pipeline_metrics(rag),
            }
        )
        if allow_direct and (sufficient or payload.rag_mode == "required"):
            run_id = str(cached.get("run_id") or f"rag-run-{uuid4()}")
            return (
                _build_rag_direct_result(
                    payload=payload,
                    request_id=request_id,
                    rag=rag,
                    run_id=run_id,
                    replayed=True,
                    scope_snapshot_hash=scope_snapshot_hash,
                ),
                audit,
                rag,
            )
        return None, audit, rag

    rag_service = getattr(request.app.state, "rag_service", None)
    if rag_service is None:
        audit.update(
            {
                "attempted": True,
                "degraded": True,
                "error": "RAG_SERVICE_UNAVAILABLE",
            }
        )
        if payload.rag_mode == "required" and required_failure_is_fatal:
            raise_agent_http_exception(
                build_agent_error(
                    code="RAG_SERVICE_UNAVAILABLE",
                    category="unavailable",
                    stage="rag",
                    message="知识库检索服务尚未初始化。",
                    retryable=True,
                    http_status=503,
                    request_id=request_id,
                )
            )
        return None, audit, None

    try:
        settings = getattr(request.app.state, "settings", None)
        retrieval_query = payload.user_message
        # 多轮查询改写只在存在真实对话上下文时启用：
        # 首问没有历史，直接使用原始问题，避免每次检索都多一次 LLM 调用。
        rewrite_enabled = bool(
            getattr(settings, "rag_query_rewrite_enabled", False)
        )
        if rewrite_enabled and len(history_messages or []) >= 2:
            rewriter = QueryRewriter(
                llm_client=getattr(
                    request.app.state, "deepseek", None
                ),
                enabled=True,
                max_tokens=int(
                    getattr(
                        settings,
                        "rag_query_rewrite_max_tokens",
                        256,
                    )
                    or 256
                ),
            )
            retrieval_query = await rewriter.rewrite(
                query=payload.user_message,
                history_messages=history_messages,
            )

        cache = _rag_answer_redis(request)
        cache_key: str | None = None
        cached_rag: dict[str, Any] | None = None
        if cache is not None and not skip_answer_cache:
            cache_key = _rag_answer_cache_key(
                payload=payload,
                retrieval_query=retrieval_query,
                provider=current_synthesis_provider(),
                kb_fingerprint=_rag_kb_fingerprint(request),
                scope_snapshot_hash=scope_snapshot_hash,
            )
            try:
                cached_raw = cache.get(cache_key)
            except Exception:
                cached_raw = None
            if cached_raw:
                try:
                    cached_rag = json.loads(cached_raw)
                except Exception:
                    cached_rag = None

        if cached_rag is not None:
            rag = dict(cached_rag)
            sufficient = _rag_sufficient(rag)
            audit.update(
                {
                    "attempted": True,
                    "sufficient": sufficient,
                    "replayed": False,
                    "cache_hit": True,
                    **_rag_pipeline_metrics(rag),
                }
            )
            if allow_direct and (sufficient or payload.rag_mode == "required"):
                return (
                    _build_rag_direct_result(
                        payload=payload,
                        request_id=request_id,
                        rag=rag,
                        run_id=f"rag-run-{uuid4()}",
                        replayed=False,
                        scope_snapshot_hash=scope_snapshot_hash,
                    ),
                    audit,
                    rag,
                )
            return None, audit, rag

        focused_queries = list(retrieval_queries or [])
        logical_requirement_ids = sorted(
            {
                str(requirement_id)
                for query in focused_queries
                for requirement_id in (
                    query.get("source_requirement_ids")
                    or [query.get("requirement_id")]
                )
                if str(requirement_id)
            }
        )
        raw_query_count = len(focused_queries) + sum(
            len(query.get("merged_from_query_ids") or [])
            for query in focused_queries
        )
        merged_query_count = sum(
            len(query.get("merged_from_query_ids") or [])
            for query in focused_queries
        )
        if len(focused_queries) > 1:
            async def retrieve_one(
                item: dict[str, Any],
            ) -> tuple[dict[str, Any], dict[str, Any]]:
                raw_item = await rag_service.answer(
                    query=item["query"],
                    retrieval_query=item["query"],
                    tenant_id=payload.tenant_id,
                    owner_user_id=payload.user_id,
                    knowledge_base_id=payload.knowledge_base_id,
                    document_ids=payload.document_ids,
                    relevance_gate=None,
                )
                return item, _serialize_model(raw_item)

            rag = _merge_rag_requirement_results(
                list(
                    await asyncio.gather(
                        *(
                            retrieve_one(item)
                            for item in focused_queries
                        )
                    )
                )
            )
        else:
            single_query = (
                dict(focused_queries[0])
                if focused_queries
                else {
                    "id": "Q1",
                    "query": retrieval_query,
                    "requirement_id": "knowledge_lookup",
                    "source_requirement_ids": ["knowledge_lookup"],
                    "merged_from_query_ids": [],
                }
            )
            if (
                len(focused_queries) == 1
                and retrieval_query != payload.user_message
            ):
                single_query["query"] = retrieval_query
            raw = await rag_service.answer(
                query=single_query["query"],
                retrieval_query=single_query["query"],
                tenant_id=payload.tenant_id,
                owner_user_id=payload.user_id,
                knowledge_base_id=payload.knowledge_base_id,
                document_ids=payload.document_ids,
                relevance_gate=(
                    float(
                        getattr(
                            settings,
                            "rag_auto_min_rerank_score",
                            0.5,
                        )
                        or 0.5
                    )
                    if payload.rag_mode != "required"
                    else None
                ),
            )
            rag = _merge_rag_requirement_results(
                [(single_query, _serialize_model(raw))]
            )
        rag["physical_queries"] = [
            {
                "id": str(query.get("id") or ""),
                "query": str(query.get("query") or ""),
                "source_requirement_ids": [
                    str(item)
                    for item in (
                        query.get("source_requirement_ids")
                        or [query.get("requirement_id")]
                    )
                    if str(item)
                ],
                "merged_from_query_ids": [
                    str(item)
                    for item in (
                        query.get("merged_from_query_ids") or []
                    )
                    if str(item)
                ],
            }
            for query in focused_queries
        ]
        rag = await _retry_missing_retrieval_requirements(
            rag_service=rag_service,
            payload=payload,
            rag=rag,
            document_ids=payload.document_ids,
        )
        rag = _finalize_requirement_coverage(
            rag,
            logical_requirement_ids,
            raw_query_count=raw_query_count,
            merged_query_count=merged_query_count,
        )
        if not isinstance(rag, dict):
            raise TypeError("RAG 服务返回值必须可序列化为对象。")
        _assert_retrieval_within_scope(rag, payload.document_ids)
        if cache is not None and cache_key and not skip_answer_cache:
            try:
                cache.setex(
                    cache_key,
                    int(
                        getattr(
                            settings,
                            "rag_answer_cache_ttl_seconds",
                            300,
                        )
                        or 300
                    ),
                    json.dumps(rag, ensure_ascii=False),
                )
            except Exception:
                pass
        sufficient = _rag_sufficient(rag)
        run_id = f"rag-run-{uuid4()}" if (
            sufficient or payload.rag_mode == "required"
        ) else None
        _store_rag_attempt(
            request,
            payload=payload,
            request_id=request_id,
            rag=rag,
            run_id=run_id,
            scope_snapshot_hash=scope_snapshot_hash,
        )
        audit.update(
            {
                "attempted": True,
                "sufficient": sufficient,
                **_rag_pipeline_metrics(rag),
            }
        )
        if allow_direct and (sufficient or payload.rag_mode == "required"):
            return (
                _build_rag_direct_result(
                    payload=payload,
                    request_id=request_id,
                    rag=rag,
                    run_id=str(run_id),
                    replayed=False,
                ),
                audit,
                rag,
            )
        return None, audit, rag
    except RequestIdempotencyConflict:
        raise
    except Exception as exc:
        audit.update(
            {
                "attempted": True,
                "degraded": True,
                "error": type(exc).__name__,
                "error_code": (
                    "rag_contract_error_nonretryable"
                    if isinstance(exc, (ValueError, TypeError))
                    else "rag_transient_error_retryable"
                ),
                "retryable": not isinstance(exc, (ValueError, TypeError)),
                "error_detail": str(exc)[:500],
            }
        )
        if payload.rag_mode == "required" and required_failure_is_fatal:
            error = exception_to_agent_error(
                exc, stage="rag", request_id=request_id
            )
            raise_agent_http_exception(error)
        logger.exception(
            "rag_attempt_failed",
            request_id=request_id,
            error_type=type(exc).__name__,
        )
        return None, audit, None


async def _resolve_semantic_route(
    *,
    payload: ProductionChatRequest,
    request: Request,
    scope_snapshot_hash: str = "",
    floor: Any | None = None,
    conversation_state: ConversationState | None = None,
    recent_messages: list[dict[str, Any]] | None = None,
    resource_catalog: list[Any] | None = None,
    capability_catalog: list[Any] | None = None,
    scope_snapshot: dict[str, Any] | None = None,
    narrative_segments: list[dict[str, Any]] | None = None,
) -> SemanticRouteDecision:
    cache = getattr(request.app.state, "semantic_route_cache", None)
    if cache is None:
        cache = OrderedDict()
        request.app.state.semantic_route_cache = cache
    cache_key = (
        payload.tenant_id,
        payload.user_id,
        payload.request_id,
        _rag_request_fingerprint(
            payload,
            scope_snapshot_hash=scope_snapshot_hash,
        ),
    )
    cached = cache.get(cache_key)
    if cached is not None:
        cache.move_to_end(cache_key)
        return SemanticRouteDecision.model_validate(cached)
    client = getattr(request.app.state, "deepseek", None)
    if client is None:
        decision = conservative_route_fallback(
            enable_rag=payload.enable_rag and payload.rag_mode != "off",
            allowed_tool_groups=payload.allowed_tool_groups,
            error_type="router_client_unavailable",
        )
        cache[cache_key] = decision.model_dump(mode="json")
        return decision
    semantic_router: SemanticRouter | None = None
    floor_contract = _requirement_contract_from_floor(floor)
    try:
        tool_catalog = [
            {"name": spec.name, "description": spec.description}
            for spec in build_production_tool_registry().list_specs()
        ]
        semantic_router = SemanticRouter(llm_client=client)
        decision = await semantic_router.route(
            payload.user_message,
            tool_catalog=tool_catalog,
            conversation_state=conversation_state,
            recent_messages=recent_messages,
            resource_catalog=resource_catalog,
            capability_catalog=capability_catalog,
            scope_snapshot=scope_snapshot,
            narrative_segments=narrative_segments,
        )
        decision = _merge_requirement_floor(
            route=decision,
            floor=_merge_requirement_contracts(
                floor_contract,
                semantic_router.last_requirement_contract,
            ),
        )
    except Exception as exc:
        requirement_contract = (
            exc.requirement_contract
            if isinstance(exc, SemanticRouteProtocolError)
            else semantic_router.last_requirement_contract
            if semantic_router is not None
            else None
        )
        if floor_contract is not None:
            requirement_contract = _merge_requirement_contracts(
                floor_contract,
                requirement_contract,
            )
        logger.warning(
            "semantic_route_degraded",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
            error_code="route_schema_validation_failed",
            schema_version=getattr(exc, "schema_version", "semantic-route-v3"),
            validation_errors=getattr(exc, "validation_errors", []),
            preserved_requirement_contract=(
                requirement_contract.model_dump(mode="json")
                if requirement_contract is not None
                else None
            ),
        )
        decision = conservative_route_fallback(
            enable_rag=payload.enable_rag and payload.rag_mode != "off",
            allowed_tool_groups=payload.allowed_tool_groups,
            error_type=type(exc).__name__,
            requirement_contract=requirement_contract,
        )
        decision = decision.model_copy(
            update={
                "semantic_contract_source": "python_fallback",
                "semantic_contract_status": "degraded",
                "protocol_repaired": True,
            }
        )
    if floor_contract is not None:
        decision = _merge_requirement_floor(
            route=decision,
            floor=floor_contract,
        )
    cache[cache_key] = decision.model_dump(mode="json")
    cache.move_to_end(cache_key)
    while len(cache) > 2048:
        cache.popitem(last=False)
    return decision


def _apply_caller_route_constraints(
    *,
    payload: ProductionChatRequest,
    route: SemanticRouteDecision,
    scope_mode: str | None = None,
) -> SemanticRouteDecision:
    """Merge explicit API controls without reinterpreting natural language."""
    if payload.rag_mode != "required" or not payload.enable_rag:
        return route
    data = route.model_dump(mode="json")
    capabilities = list(data["required_capabilities"])
    if "knowledge_retrieval" not in capabilities:
        capabilities.append("knowledge_retrieval")
    tasks = list(data["task_requirements"])
    if not any("knowledge_retrieval" in item["capabilities"] for item in tasks if item["required"]):
        tasks.append(
            {
                "id": "required_knowledge_lookup",
                "description": "Retrieve and cite evidence from the caller-selected knowledge scope",
                "required": True,
                "capabilities": ["knowledge_retrieval"],
                "evidence_tool_names": [],
                "requires_citations": True,
            }
        )
    data.update(
        required_capabilities=capabilities,
        task_requirements=tasks,
        retrieval_requirement="required",
        citation_requirement="required",
        grounding_requirement="authoritative",
        retrieval_scope=(
            "uploaded_documents"
            if scope_mode == "all_uploaded"
            else "selected_documents"
            if scope_mode == "selected" or payload.document_ids
            else "all_accessible_knowledge_base"
        ),
        orchestration_mode=(
            "hybrid" if "financial_calculation" in capabilities else "rag"
        ),
    )
    return SemanticRouteDecision.model_validate(data)


_OUTPUT_SPLIT_RE = re.compile(r"[；;。，,、]")
_OUTPUT_KEY_MARKERS = (
    "授权",
    "权利",
    "包含",
    "期限",
    "限额",
    "起算点",
    "流动性",
    "比例",
    "月供",
    "首付",
    "备用金",
    "收益率",
    "通知存款",
)
_LOW_INFORMATION_KEYPHRASES = {
    "律'",
    "措施",
    "风险",
    "参考",
    "内容",
    "方式",
    "原则",
    "规定",
    "情况",
    "建议",
    "信息",
}


def _required_output_keyphrase(output: str) -> str:
    cleaned = re.sub(r"[\s'\"“”‘’]", "", str(output or ""))
    for marker in _OUTPUT_KEY_MARKERS:
        if marker in cleaned:
            return marker
    return ""


def _required_outputs_delivered(
    required_outputs: list[str],
    answer: str,
) -> list[str]:
    """Return required-output keyphrases that are missing from the answer."""
    compact = re.sub(r"\s+", "", answer or "")
    missing: list[str] = []
    for output in required_outputs:
        keyphrase = _required_output_keyphrase(str(output or ""))
        if not keyphrase:
            # 低信息量/无法稳定识别的子要求只作为 debug 提示，
            # 不得改变 task 完成状态。
            continue
        if (
            keyphrase in _LOW_INFORMATION_KEYPHRASES
            or keyphrase not in compact
        ):
            missing.append(keyphrase)
    return missing


def _enrich_task_required_outputs(
    route: SemanticRouteDecision,
) -> SemanticRouteDecision:
    """Split retrieval/citation task descriptions into required_outputs."""
    tasks = []
    for task in route.task_requirements:
        outputs = list(task.required_outputs or [])
        if not outputs and (
            "knowledge_retrieval" in task.capabilities
            or task.requires_citations
        ):
            outputs = [
                part.strip()
                for part in _OUTPUT_SPLIT_RE.split(task.description)
                if part.strip()
            ][:8]
        tasks.append(
            task.model_copy(update={"required_outputs": outputs})
        )
    return route.model_copy(update={"task_requirements": tasks})


def _merge_requirement_floor(
    *,
    route: SemanticRouteDecision,
    floor: RequestRequirementContract | None,
) -> SemanticRouteDecision:
    """Monotonic merge: downstream routing may strengthen, never weaken."""
    if floor is None:
        return route
    priority = {"not_needed": 0, "optional": 1, "preferred": 1, "required": 2}
    data = route.model_dump(mode="json")
    retrieval = max(
        [route.retrieval_requirement, floor.retrieval_requirement],
        key=lambda item: priority[item],
    )
    citation = max(
        [route.citation_requirement, floor.citation_requirement],
        key=lambda item: priority[item],
    )
    capabilities = list(
        dict.fromkeys([*route.required_capabilities, *floor.required_capabilities])
    )
    tasks = {task["id"]: task for task in data["task_requirements"]}
    for task in floor.task_requirements:
        tasks.setdefault(task.id, task.model_dump(mode="json"))
    calculation_required = floor.calculation_requirement == "required"
    data.update(
        retrieval_requirement=retrieval,
        citation_requirement=citation,
        needs_exact_calculation=(route.needs_exact_calculation or floor.needs_exact_calculation),
        required_capabilities=capabilities,
        task_requirements=list(tasks.values()),
        grounding_requirement=(
            "authoritative" if retrieval == "required" else data["grounding_requirement"]
        ),
        retrieval_scope=(
            data["retrieval_scope"]
            if data["retrieval_scope"] != "none"
            else "all_accessible_knowledge_base" if retrieval != "not_needed"
            else "none"
        ),
        orchestration_mode=(
            "hybrid" if retrieval != "not_needed" and calculation_required
            else data["orchestration_mode"]
        ),
    )
    return SemanticRouteDecision.model_validate(data)


_CAPABILITY_TASK_KIND = {
    "knowledge_retrieval": "retrieval",
    "financial_calculation": "calculation",
    "citation_validation": "validation",
    "complex_reasoning": "synthesis",
    "general_explanation": "reasoning",
    "memory_read": "reasoning",
}


def _requirement_contract_from_floor(
    floor: Any | None,
) -> RequestRequirementContract | None:
    """Translate the deterministic explicit floor into a router contract.

    Fallback routing is monotonic: a degraded model route may remove optional
    model behavior, but it must never remove a user-required capability that
    the explicit floor already made REQUIRED.
    """

    if floor is None:
        return None
    required = [
        item
        for item in (getattr(floor, "constraints", None) or ())
        if item.requirement == RequirementLevel.REQUIRED
        and item.permission != PermissionLevel.FORBIDDEN
    ]
    if not required:
        return None

    capabilities: list[str] = []
    for item in required:
        if item.capability not in capabilities:
            capabilities.append(item.capability)

    retrieval_required = "knowledge_retrieval" in capabilities
    citation_required = "citation_validation" in capabilities
    if citation_required and not retrieval_required:
        # Citations are meaningless without retrieval; promoting retrieval is
        # the same monotonic rule the semantic router already enforces.
        capabilities.insert(0, "knowledge_retrieval")
        retrieval_required = True

    tasks: list[TaskRequirement] = []
    for capability in capabilities:
        tasks.append(
            TaskRequirement(
                id=f"floor_required_{capability}",
                description=(
                    f"Complete the user-required capability: {capability}"
                ),
                required=True,
                capabilities=[capability],
                evidence_tool_names=[],
                requires_citations=(
                    capability == "knowledge_retrieval"
                    and citation_required
                )
                or capability == "citation_validation",
                task_kind=_CAPABILITY_TASK_KIND.get(
                    capability,
                    "reasoning",
                ),
                depends_on=[],
            )
        )

    return RequestRequirementContract(
        retrieval_requirement=(
            "required" if retrieval_required else "not_needed"
        ),
        citation_requirement=(
            "required" if citation_required else "not_needed"
        ),
        calculation_requirement=(
            "required"
            if "financial_calculation" in capabilities
            else "not_needed"
        ),
        needs_exact_calculation=False,
        required_capabilities=capabilities,
        task_requirements=tasks,
    )


def _merge_requirement_contracts(
    base: RequestRequirementContract | None,
    other: RequestRequirementContract | None,
) -> RequestRequirementContract:
    """Monotonic max over two requirement contracts."""

    if other is None:
        return base
    if base is None:
        return other
    priority = {"not_needed": 0, "optional": 1, "preferred": 1, "required": 2}
    capabilities = list(
        dict.fromkeys(
            [*other.required_capabilities, *base.required_capabilities]
        )
    )
    tasks = list(other.task_requirements)
    existing_ids = {task.id for task in tasks}
    for task in base.task_requirements:
        if task.id not in existing_ids:
            tasks.append(task)
            existing_ids.add(task.id)
    return RequestRequirementContract(
        retrieval_requirement=max(
            [other.retrieval_requirement, base.retrieval_requirement],
            key=lambda item: priority[item],
        ),
        citation_requirement=max(
            [other.citation_requirement, base.citation_requirement],
            key=lambda item: priority[item],
        ),
        calculation_requirement=(
            "required"
            if (
                other.calculation_requirement == "required"
                or base.calculation_requirement == "required"
            )
            else "not_needed"
        ),
        needs_exact_calculation=(
            other.needs_exact_calculation
            or base.needs_exact_calculation
        ),
        required_capabilities=capabilities,
        task_requirements=tasks,
    )


def _rag_outcome(
    audit: dict[str, Any], rag: dict[str, Any] | None
) -> dict[str, Any]:
    assessment_status = str(
        audit.get("evidence_assessment_status") or "not_run"
    )
    coverage = list(audit.get("requirement_coverage") or [])
    failed_coverage = [
        item
        for item in coverage
        if item.get("status")
        in {"assessment_protocol_failed", "service_failed"}
    ]
    satisfied_coverage = [
        item
        for item in coverage
        if item.get("status")
        in {"direct_support", "partial_support", "background_support"}
    ]
    if not audit.get("attempted"):
        status = "not_attempted"
    elif audit.get("degraded"):
        status = "failed_technical"
    elif (
        assessment_status
        in {
            "protocol_failed",
            "service_failed",
            "partial_protocol_failure",
        }
        and rag
    ):
        # A per-task protocol failure must not erase the tasks that already
        # produced valid evidence.  Only the failed tasks remain missing.
        status = (
            "completed_with_partial_evidence"
            if satisfied_coverage
            else "completed_with_unassessed_evidence"
        )
    elif rag and _rag_sufficient(rag):
        status = "completed_with_evidence"
    elif rag and _rag_has_citable_support(rag):
        status = "completed_with_partial_evidence"
    else:
        status = "completed_no_evidence"
    execution_ok = bool(
        audit.get("attempted")
        and not audit.get("degraded")
        and status != "failed_technical"
    )
    missing_coverage = [
        str(item.get("requirement_id") or "retrieval_requirement")
        for item in coverage
        if item.get("status")
        not in {
            "direct_support",
            "partial_support",
            "background_support",
            "insufficient_evidence",
            "irrelevant",
        }
    ]
    conflict_source = list((rag or {}).get("evidence_conflicts") or [])
    if not conflict_source:
        conflict_source = list(
            (((rag or {}).get("evidence_assessment") or {}).get("evidence_conflicts") or [])
        )
    unresolved_conflicts = [
        str(conflict.get("conflict_id") or "evidence_conflict")
        for conflict in conflict_source
        if conflict.get("unresolved", True)
    ]
    return {
        "status": status,
        "execution_ok": execution_ok,
        "evidence_insufficient": status == "completed_no_evidence",
        "protocol_failed_task_ids": [
            str(item.get("requirement_id") or "")
            for item in failed_coverage
        ],
        "error_code": audit.get("error_code") or audit.get("error"),
        "error_type": audit.get("error"),
        "retryable": bool(audit.get("retryable")),
        "retrieved_count": int(audit.get("retrieved_count") or 0),
        "reranked_count": int(audit.get("reranked_count") or 0),
        "evidence_candidate_count": int(
            audit.get("evidence_candidate_count") or 0
        ),
        "sufficient_evidence_count": int(
            audit.get("sufficient_evidence_count") or 0
        ),
        "citation_count": int(audit.get("citation_count") or 0),
        "citable_evidence_count": int(
            audit.get("citable_evidence_count") or 0
        ),
        "evidence_support_level": audit.get("evidence_support_level"),
        "retrieval_status": audit.get("retrieval_status"),
        "rerank_status": audit.get("rerank_status"),
        "evidence_assessment_status": assessment_status,
        "conflict_detection_status": audit.get("conflict_detection_status"),
        "protocol_error_stage": audit.get("protocol_error_stage"),
        "provisional_citation_count": int(
            audit.get("provisional_citation_count") or 0
        ),
        "requirement_coverage": coverage,
        "missing_retrieval_requirements": missing_coverage,
        "unresolved_required_conflicts": unresolved_conflicts,
        "retrieval_retry_count": int(audit.get("retrieval_retry_count") or 0),
        "evidence_rejection_reason": audit.get("evidence_rejection_reason"),
    }


def _rag_evidence_context(rag: dict[str, Any] | None) -> str:
    """Build a governed evidence context string.

    The context is compressed per RequirementObservation and citations are
    deduplicated; logical requirements are never dropped.
    """

    if not rag:
        return ""

    evidence_text, stats = build_evidence_context(rag)
    rag["context_governance"] = stats
    return evidence_text


def _build_performance_summary(
    *,
    started_at: float,
    result: dict[str, Any],
    retrieval_outcome: dict[str, Any] | None = None,
    memory_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Real token/latency observability for the final response."""

    latency_ms = max(
        0,
        int((time.perf_counter() - started_at) * 1000),
    )

    final_response_result = result.get(
        "final_response_result"
    ) or {}
    if isinstance(final_response_result, str):
        try:
            final_response_result = json.loads(
                final_response_result
            )
        except json.JSONDecodeError:
            final_response_result = {}
    if not isinstance(final_response_result, dict):
        final_response_result = {}

    usage_by_stage = (
        final_response_result.get("usage_by_stage")
        or result.get("usage_by_stage")
        or {}
    )

    invocation_tokens: dict[str, int] = {}
    for invocation in (
        final_response_result.get("model_invocations") or []
    ):
        stage = str(invocation.get("stage") or "unknown")
        usage = invocation.get("usage") or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(
            usage.get("completion_tokens") or 0
        )
        invocation_tokens[stage] = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    governance: dict[str, Any] = {}
    if retrieval_outcome:
        governance["evidence"] = (
            retrieval_outcome.get("context_governance")
            or {}
        )
    if memory_audit:
        governance["history"] = (
            memory_audit.get("history_governance")
            or {}
        )
        governance["memory"] = (
            memory_audit.get("context_governance")
            or {}
        )

    return {
        "latency_ms": latency_ms,
        "usage_by_stage": usage_by_stage,
        "synthesis_guard_tokens": invocation_tokens,
        "context_governance": governance,
    }


def _raise_context_contract_error(
    *,
    code: str,
    message: str,
    request_id: str,
    run_id: str | None,
    violations: list[str],
) -> None:
    raise_agent_http_exception(
        build_agent_error(
            code=code,
            category="validation",
            stage="api",
            message=message,
            retryable=False,
            http_status=422,
            request_id=request_id,
            run_id=run_id,
            details={
                "reason_codes": [code],
                "violations": violations,
            },
        )
    )


def _validate_context_references(
    *,
    route: SemanticRouteDecision,
    conversation_state: ConversationState,
    resource_catalog: list[Any],
    allowed_document_ids: list[str],
) -> None:
    """Python-only validation of handles/tasks/actions/results."""

    selected_handles: list[str] = []
    for reference in route.resource_references:
        if reference.status != "resolved":
            continue
        selected_handles.extend(
            reference.selected_handles or []
        )

    _document_ids, violations = resource_handles_to_document_ids(
        selected_handles=selected_handles,
        catalog=resource_catalog,
        state=conversation_state,
        allowed_document_ids=allowed_document_ids,
    )
    if violations:
        unknown = [
            item for item in violations
            if item.startswith("unknown_handle:")
        ]
        scope_conflicts = [
            item for item in violations
            if item.startswith("scope_conflict:")
        ]
        if unknown:
            _raise_context_contract_error(
                code="RESOURCE_HANDLE_UNKNOWN",
                message="模型引用了不存在的资源 handle。",
                request_id=route.request_id
                if hasattr(route, "request_id")
                else "",
                run_id=None,
                violations=unknown,
            )
        if scope_conflicts:
            _raise_context_contract_error(
                code="RESOURCE_SCOPE_CONFLICT",
                message=(
                    "模型解析出的资源超出了当前授权范围，"
                    "无法绕过已解析的文档作用域。"
                ),
                request_id=route.request_id
                if hasattr(route, "request_id")
                else "",
                run_id=None,
                violations=scope_conflicts,
            )

    if (
        route.task_reference.status == "resolved"
        and route.task_reference.task_handle
        and conversation_state.active_task is not None
        and route.task_reference.task_handle
        != conversation_state.active_task.handle
    ):
        _raise_context_contract_error(
            code="TASK_REFERENCE_UNKNOWN",
            message="模型引用了当前会话中不存在的任务 handle。",
            request_id=route.request_id
            if hasattr(route, "request_id")
            else "",
            run_id=None,
            violations=[
                f"task_handle:{route.task_reference.task_handle}"
            ],
        )
    if (
        route.task_reference.status == "resolved"
        and route.task_reference.task_handle
        and conversation_state.active_task is None
    ):
        _raise_context_contract_error(
            code="TASK_REFERENCE_UNKNOWN",
            message="当前会话没有可引用的活动任务。",
            request_id=route.request_id
            if hasattr(route, "request_id")
            else "",
            run_id=None,
            violations=[
                f"task_handle:{route.task_reference.task_handle}"
            ],
        )

    if route.pending_action_resolution.status in {
        "confirmed",
        "rejected",
    }:
        action_handle = (
            route.pending_action_resolution.action_handle
        )
        pending = conversation_state.pending_action
        if pending is None or (
            action_handle
            and action_handle != pending.handle
        ):
            _raise_context_contract_error(
                code="PENDING_ACTION_CONFLICT",
                message=(
                    "模型确认/拒绝的动作与当前待确认动作不一致。"
                ),
                request_id=route.request_id
                if hasattr(route, "request_id")
                else "",
                run_id=None,
                violations=[
                    f"action_handle:{action_handle}"
                ],
            )

    known_results = {
        item.handle
        for item in conversation_state.recent_results
    }
    for reference in route.result_references:
        if (
            reference.status == "resolved"
            and reference.handle
            and reference.handle not in known_results
        ):
            _raise_context_contract_error(
                code="RESULT_REFERENCE_UNKNOWN",
                message="模型引用了当前会话中不存在的结果 handle。",
                request_id=route.request_id
                if hasattr(route, "request_id")
                else "",
                run_id=None,
                violations=[
                    f"result_handle:{reference.handle}"
                ],
            )

    recent_by_handle = {
        item.handle: item
        for item in conversation_state.recent_results
    }
    for reference in route.result_references:
        if (
            reference.status != "resolved"
            or not reference.handle
            or not reference.artifact_handle
        ):
            continue
        artifact = recent_by_handle.get(reference.handle)
        if artifact is None:
            continue
        if reference.artifact_handle not in set(
            artifact.sub_artifact_handles
        ):
            _raise_context_contract_error(
                code="RESULT_ARTIFACT_UNKNOWN",
                message=(
                    "模型引用了结构化结果中不存在的子产物。"
                ),
                request_id=route.request_id
                if hasattr(route, "request_id")
                else "",
                run_id=None,
                violations=[
                    "sub_artifact:"
                    f"{reference.handle}."
                    f"{reference.artifact_handle}"
                ],
            )


def _resolved_resources_from_route(
    *,
    route: SemanticRouteDecision,
    conversation_state: ConversationState,
    resource_catalog: list[Any],
    allowed_document_ids: list[str],
) -> list[dict[str, Any]]:
    """Map resolved handles to real document ids (Python-owned)."""

    selected_handles: list[str] = []
    for reference in route.resource_references:
        if reference.status != "resolved":
            continue
        selected_handles.extend(
            reference.selected_handles or []
        )
    document_ids, _violations = resource_handles_to_document_ids(
        selected_handles=selected_handles,
        catalog=resource_catalog,
        state=conversation_state,
        allowed_document_ids=allowed_document_ids,
    )
    catalog_by_handle = {
        item.handle: item for item in resource_catalog
    }
    resolved: list[dict[str, Any]] = []
    seen_documents: set[str] = set()
    for handle in selected_handles:
        entry = catalog_by_handle.get(handle)
        document_id = conversation_state.resource_handle_map.get(
            handle
        )
        if (
            entry is None
            or not document_id
            or document_id in seen_documents
        ):
            continue
        seen_documents.add(document_id)
        resolved.append(
            {
                "handle": handle,
                "resource_type": entry.resource_type,
                "title": entry.title,
                "document_id": document_id,
            }
        )
    return resolved


def memory_requirement_satisfied(
    requirement: str,
    observation: dict[str, Any] | None,
) -> bool:
    """Memory completion semantics: never an implicit gate."""

    if requirement in {"forbidden", "not_needed", "optional"}:
        return True
    if requirement == "required":
        return bool(
            observation
            and observation.get("status") == "succeeded"
        )
    return False


def _apply_completion_contract(
    *,
    result: dict[str, Any],
    route: SemanticRouteDecision,
    rag_outcome: dict[str, Any],
    memory_audit: dict[str, Any] | None = None,
    materialized_artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    tool_results = list(result.get("tool_results") or [])
    successful_tool_names = {
        str(item.get("tool_name") or "")
        for item in tool_results
        if bool(item.get("success"))
    }
    successful_tools = bool(successful_tool_names)
    retrieval_ok = rag_outcome["status"] in {
        "completed_with_evidence",
        "completed_with_partial_evidence",
        "completed_no_evidence",
        "completed_with_unassessed_evidence",
    }
    citable_evidence_ok = rag_outcome["status"] in {
        "completed_with_evidence", "completed_with_partial_evidence"
    }
    executed_without_evidence = bool(
        rag_outcome.get("execution_ok")
        and rag_outcome.get("evidence_insufficient")
    )
    coverage_by_task = {
        str(item.get("requirement_id") or ""): item
        for item in (rag_outcome.get("requirement_coverage") or [])
    }
    _EVIDENCE_COMPLETED_STATUSES = {
        "direct_support",
        "partial_support",
        "background_support",
        "insufficient_evidence",
        "irrelevant",
    }

    def task_evidence_satisfied(task: Any) -> bool:
        """Per-evidence-requirement coverage is the single fact source."""
        evidence_ids = [
            f"{task.id}:{index}"
            for index in range(
                1, len(task.evidence_requirements) + 1
            )
        ]
        if evidence_ids:
            entries = [
                coverage_by_task.get(requirement_id)
                for requirement_id in evidence_ids
            ]
            if not any(entries):
                # Task 要求逐条检索，但没有任何子证据 observation：
                # 不能判 completed（not_observed）。
                return False
            return all(
                bool(entry)
                and entry.get("status")
                in _EVIDENCE_COMPLETED_STATUSES
                for entry in entries
            )
        item = coverage_by_task.get(str(task.id))
        if item is None:
            # No per-requirement coverage (e.g. single-query RAG): fall back
            # to the global execution state.
            return citable_evidence_ok or executed_without_evidence
        return bool(
            item.get("status")
            in _EVIDENCE_COMPLETED_STATUSES
        )

    # “引用校验”与“引用可用性”解耦：
    # RAG 已执行且引用（若有）合法，即使最终 citation_count=0
    # （没有可接受证据），也属于正常完成结果；披露由 Output Guard 负责。
    retrieval_tasks = [
        task
        for task in route.task_requirements
        if task.required and "knowledge_retrieval" in task.capabilities
    ]
    citation_tasks = [
        task
        for task in route.task_requirements
        if task.required and task.requires_citations
    ]
    retrieval_tasks_ok = (
        all(task_evidence_satisfied(task) for task in retrieval_tasks)
        if retrieval_tasks
        else retrieval_ok or executed_without_evidence
    )
    any_retrieval_ok = (
        any(task_evidence_satisfied(task) for task in retrieval_tasks)
        if retrieval_tasks
        else retrieval_tasks_ok
    )
    citation_tasks_ok = (
        all(task_evidence_satisfied(task) for task in citation_tasks)
        if citation_tasks
        else citable_evidence_ok or executed_without_evidence
    )
    any_citation_ok = (
        any(task_evidence_satisfied(task) for task in citation_tasks)
        if citation_tasks
        else citation_tasks_ok
    )
    evidence_ok = retrieval_tasks_ok and citation_tasks_ok
    insufficient_disclosed = bool(
        executed_without_evidence
        and any(
            marker in str(result.get("final_answer") or "")
            for marker in (
                "没有找到足够依据",
                "未找到足够依据",
                "没有足够依据",
                "没有找到相关证据",
                "未能找到足够",
                "知识库中没有找到",
                "文档中没有找到",
            )
        )
    )
    task_outcomes: list[dict[str, Any]] = []
    synthesis_out = (
        (result.get("final_response_result") or {}).get(
            "synthesis"
        )
        if isinstance(
            (result.get("final_response_result") or {}).get(
                "synthesis"
            ),
            dict,
        )
        else {}
    )
    used_derivations = list(
        synthesis_out.get("used_derivation_ids") or []
    )
    used_fact_refs = list(
        synthesis_out.get("used_fact_refs") or []
    )
    used_result_artifact_refs = list(
        synthesis_out.get("used_result_artifact_refs") or []
    )
    materialized_calcs = {
        str(item.get("handle") or ""): item
        for item in (materialized_artifacts or [])
        if str(item.get("artifact_type") or "").lower()
        in {"calc", "calculation"}
    }
    prior_verified_calcs: dict[str, dict[str, Any]] = {}
    for artifact in (
        (result.get("route_context") or {}).get(
            "resolved_result_artifacts"
        )
        or []
    ):
        if not isinstance(artifact, dict):
            continue
        for calc in artifact.get("calculations") or []:
            if (
                isinstance(calc, dict)
                and calc.get("verification_status") == "verified"
                and calc.get("output") is not None
            ):
                prior_verified_calcs[
                    str(calc.get("handle") or "")
                ] = calc
    prior_derivation_refs = [
        str(ref).rsplit(".", 1)[-1]
        for ref in used_result_artifact_refs
        if str(ref).rsplit(".", 1)[-1].startswith("CALC_")
    ]
    verified_calc_refs = [
        ref
        for ref in used_derivations
        if ref in materialized_calcs
        and materialized_calcs[ref].get(
            "verification_status"
        )
        == "verified"
        and materialized_calcs[ref].get("output") is not None
    ]
    verified_calc_refs = list(
        dict.fromkeys(
            [
                *verified_calc_refs,
                *[
                    ref
                    for ref in prior_derivation_refs
                    if ref in prior_verified_calcs
                ],
            ]
        )
    )
    all_verified_calcs = {
        **prior_verified_calcs,
        **materialized_calcs,
    }
    for task in route.task_requirements:
        evidence_tools = set(task.evidence_tool_names)
        derivation_satisfies_financial = bool(
            verified_calc_refs
            and "financial_calculation" in task.capabilities
        )
        missing_tools = (
            []
            if derivation_satisfies_financial
            else sorted(evidence_tools - successful_tool_names)
        )
        if (
            "financial_calculation" in task.capabilities
            and not derivation_satisfies_financial
            and not successful_tools
        ):
            missing_tools = sorted(
                {*missing_tools, "CALC_VERIFIED"}
            )
        missing_evidence = bool(
            (
                "knowledge_retrieval" in task.capabilities
                or task.requires_citations
            )
            and not task_evidence_satisfied(task)
        )
        missing_citations = bool(
            task.requires_citations
            and not task_evidence_satisfied(task)
        )
        evidence_universe_missing = bool(
            task.requires_citations
            and (
                "knowledge_retrieval" in task.capabilities
                or task.task_kind == "retrieval"
            )
            and not (task.evidence_requirements or [])
        )
        if evidence_universe_missing:
            missing_evidence = True
            missing_citations = True
        completed = not missing_tools and not missing_evidence
        # 语义子要求覆盖不由字符串关键词决定 task 状态：
        # 只在这里生成 debug 提示，真正的语义判定交给 Output Guard
        # （它在 LLM 侧逐条检查 delivery_contract.required_outputs）。
        coverage_warnings: list[str] = []
        if evidence_universe_missing:
            coverage_warnings.append(
                "evidence_requirement_universe_missing"
            )
        task_evidence_ids = [
            f"{task.id}:{index}"
            for index in range(
                1,
                len(task.evidence_requirements or []) + 1,
            )
        ]
        if task.required_outputs and (
            str(task.id) in coverage_by_task
            or any(
                requirement_id in coverage_by_task
                for requirement_id in task_evidence_ids
            )
        ):
            coverage_warnings = _required_outputs_delivered(
                list(task.required_outputs),
                str(result.get("final_answer") or ""),
            )
        task_outcomes.append(
            {
                "id": task.id,
                "description": task.description,
                "required": task.required,
                "capabilities": list(task.capabilities),
                "status": "completed" if completed else "failed_requirement",
                "evidence_tool_names": sorted(evidence_tools),
                "completed_tool_names": sorted(
                    evidence_tools & successful_tool_names
                ),
                "missing_tool_names": missing_tools,
                "requires_citations": task.requires_citations,
                "required_outputs": list(task.required_outputs),
                "missing_retrieval_evidence": missing_evidence,
                "coverage_warnings": coverage_warnings,
                "missing_citations": missing_citations,
                "evidence_requirement_universe_missing": (
                    evidence_universe_missing
                ),
            }
        )
    outcomes: dict[str, dict[str, Any]] = {
        "knowledge_retrieval": {
            **rag_outcome,
            # Keep the capability contract status separate from the detailed
            # retrieval lifecycle status (for example
            # ``completed_with_evidence``).  The previous order allowed
            # rag_outcome["status"] to overwrite "succeeded", which made a
            # successful required retrieval fail during final aggregation.
            # partial_evidence / completed_no_evidence mean the retrieval
            # executed; evidence quality is reported separately and must not
            # downgrade the execution capability to failed.
            "status": (
                "succeeded"
                if retrieval_tasks_ok
                else "partial"
                if any_retrieval_ok
                else "failed"
            ),
            "retrieval_status": rag_outcome["status"],
            "satisfaction_source": "retrieval",
        },
        "financial_calculation": {
            "status": (
                "satisfied"
                if (verified_calc_refs or successful_tools)
                else "failed"
            ),
            "satisfaction_source": (
                "derivation"
                if verified_calc_refs
                else "tool"
                if successful_tools
                else None
            ),
            "successful_tool_result": successful_tools,
            "successful_tool_names": sorted(successful_tool_names),
            "result_refs": verified_calc_refs,
            "materialized_artifact_refs": verified_calc_refs,
            "attempted": bool(
                successful_tools
                or used_derivations
                or used_fact_refs
            ),
            "execution_ok": bool(
                verified_calc_refs or successful_tools
            ),
        },
        "complex_reasoning": {
            "status": "succeeded" if str(result.get("final_answer") or "").strip() else "failed",
            "satisfaction_source": "reasoning",
        },
        "general_explanation": {
            "status": "succeeded" if str(result.get("final_answer") or "").strip() else "failed",
            "satisfaction_source": "synthesis",
        },
        "citation_validation": {
            "status": (
                "succeeded"
                if citation_tasks_ok
                else "partial"
                if any_citation_ok
                else "failed"
            ),
            "satisfaction_source": "retrieval",
        },
    }
    calc_outcome = outcomes.get("financial_calculation") or {}
    if (
        calc_outcome.get("status") == "satisfied"
        and calc_outcome.get("satisfaction_source") == "derivation"
    ):
        refs = list(calc_outcome.get("result_refs") or [])
        if not refs:
            raise RuntimeError(
                "invariant: satisfied derivation without result_refs"
            )
        for ref in refs:
            calc = all_verified_calcs.get(str(ref))
            if (
                calc is None
                or calc.get("verification_status")
                != "verified"
                or calc.get("output") is None
            ):
                raise RuntimeError(
                    "invariant: satisfied derivation references "
                    f"unverified CALC {ref}"
                )
    memory_requirement = str(
        getattr(route, "memory_constraint", "not_needed")
        or "not_needed"
    )
    memory_degraded = any(
        str(item.get("stage") or "") == "long_memory_read"
        for item in (memory_audit or {}).get("degraded") or []
    )
    memory_attempted = bool(
        (memory_audit or {}).get(
            "long_memory_attempted"
        )
    )
    memory_loaded = int(
        (memory_audit or {}).get("long_memory_loaded") or 0
    )
    if memory_requirement in {"not_needed", "forbidden"}:
        memory_status = memory_requirement
        memory_attempted = False
        memory_exec_ok = False
    elif memory_degraded:
        memory_status = "technical_unavailable"
        memory_attempted = True
        memory_exec_ok = False
    elif memory_loaded > 0:
        memory_status = "satisfied"
        memory_attempted = True
        memory_exec_ok = True
    elif memory_attempted:
        memory_status = "empty"
        memory_attempted = True
        memory_exec_ok = True
    else:
        memory_status = "not_observed"
        memory_attempted = False
        memory_exec_ok = False
    outcomes["memory_read"] = {
        "capability": "memory_read",
        "requirement": memory_requirement,
        "effective_policy": memory_requirement,
        "attempted": memory_attempted,
        "execution_ok": memory_exec_ok,
        "status": memory_status,
        "satisfaction_source": (
            "memory"
            if memory_exec_ok
            and memory_status not in {"not_needed", "forbidden"}
            else None
        ),
        "result_refs": [],
        "error_code": (
            "LTM_UNAVAILABLE"
            if memory_status == "technical_unavailable"
            else None
        ),
        "retryable": memory_status
        == "technical_unavailable",
    }
    catalog_payload = (
        (result.get("route_context") or {}).get(
            "resource_catalog_payload"
        )
    )
    outcomes["resource_catalog_read"] = {
        "status": (
            "succeeded"
            if catalog_payload is not None
            else "requires_final_synthesis"
        ),
        "satisfaction_source": "catalog",
    }
    missing = [
        capability
        for capability in route.required_capabilities
        if not _capability_is_satisfied(
            capability=capability,
            outcome=outcomes.get(capability, {}),
            route=route,
            task_outcomes=task_outcomes,
        )
    ]
    if route.citation_requirement == "required" and not citation_tasks_ok:
        missing.append("required_citations")
    if (
        route.grounding_requirement in {"authoritative", "exclusive"}
        and not retrieval_tasks_ok
    ):
        missing.append("required_grounding")
    missing.extend(
        f"required_task:{task['id']}"
        for task in task_outcomes
        if task["required"] and task["status"] != "completed"
    )
    missing.extend(
        f"evidence_universe_missing:{task.id}"
        for task in route.task_requirements
        if task.required
        and task.requires_citations
        and (
            "knowledge_retrieval" in task.capabilities
            or task.task_kind == "retrieval"
        )
        and not (task.evidence_requirements or [])
    )
    missing = list(dict.fromkeys(missing))
    fulfillment = "fulfilled" if not missing else (
        "unfulfilled"
        if route.grounding_requirement == "exclusive" and not evidence_ok
        else "partial"
    )
    original_status = str(result.get("status") or "completed")
    execution_status = (
        "degraded"
        if rag_outcome["status"] == "failed_technical"
        or missing
        else "success"
    )
    source_ledger: list[dict[str, Any]] = []
    claim_ledger: list[dict[str, Any]] = []
    for item in tool_results:
        if item.get("success"):
            source_id = item.get("tool_call_id")
            source_ledger.append(
                {
                    "source_type": "tool_result",
                    "source_id": source_id,
                    "name": item.get("tool_name"),
                    "payload": item.get("output"),
                }
            )
            output = item.get("output")
            if isinstance(output, dict):
                for field, value in output.items():
                    if isinstance(value, (int, float, str, bool)):
                        claim_ledger.append(
                            {
                                "claim_type": "verified_tool_value",
                                "claim": f"{item.get('tool_name')}.{field}={value}",
                                "source_ids": [source_id],
                            }
                        )
    for citation in list((result.get("rag") or {}).get("citations") or result.get("citations") or []):
        citation_id = citation.get("citation_id")
        source_ledger.append(
            {
                "source_type": "document_citation",
                "source_id": citation_id,
                "document_id": citation.get("document_id"),
                "page": citation.get("page_number") or citation.get("page_start"),
            }
        )
        claim_ledger.append(
            {
                "claim_type": "document_evidence",
                "claim": str(
                    citation.get("quote")
                    or citation.get("text")
                    or (citation.get("metadata") or {}).get("evidence_excerpt")
                    or ""
                )[:500],
                "source_ids": [citation_id],
            }
        )
    for claim in list(
        (((result.get("rag") or {}).get("evidence_assessment") or {}).get("evidence_claims") or [])
    ):
        evidence_number = int(claim.get("evidence_number") or 0)
        matching = [
            citation.get("citation_id")
            for citation in ((result.get("rag") or {}).get("citations") or [])
            if int(citation.get("citation_id") or 0) == evidence_number
        ]
        claim_ledger.append(
            {
                "claim_id": claim.get("claim_id"),
                "claim_type": "document_direct"
                if claim.get("support_level") == "direct"
                else "document_derived",
                "claim": {
                    "subject": claim.get("subject"),
                    "attribute": claim.get("attribute"),
                    "value": claim.get("value"),
                    "unit": claim.get("unit"),
                },
                "source_ids": matching,
            }
        )
    result.update(
        semantic_route=route.model_dump(mode="json"),
        capability_outcomes=outcomes,
        completion_contract={
            "required_tasks": [item.model_dump(mode="json") for item in route.task_requirements],
            "task_outcomes": task_outcomes,
            "missing_requirements": missing,
        },
        execution_status=execution_status,
        fulfillment_status=fulfillment,
        source_ledger=source_ledger,
        claim_ledger=claim_ledger,
    )
    if fulfillment != "fulfilled" and original_status == "completed":
        result["status"] = fulfillment
        if result.get("finish_reason") not in {"rag_evidence_insufficient"}:
            result["finish_reason"] = "required_capability_incomplete"
    if insufficient_disclosed:
        result["finish_reason"] = "insufficient_scoped_evidence"
    return result


def _capability_is_satisfied(
    *,
    capability: str,
    outcome: dict[str, Any],
    route: SemanticRouteDecision,
    task_outcomes: list[dict[str, Any]],
) -> bool:
    """The single policy entry point for final capability satisfaction."""
    if capability == "memory_read":
        return memory_requirement_satisfied(
            route.memory_constraint,
            outcome,
        )
    if capability == "knowledge_retrieval":
        if outcome.get("missing_retrieval_requirements"):
            return False
        if (
            outcome.get("retrieval_status")
            in {"completed_with_evidence", "completed_with_partial_evidence"}
            and not outcome.get("missing_retrieval_requirements")
            and not outcome.get("unresolved_required_conflicts")
        ):
            return True
        if (
            outcome.get("execution_ok")
            and outcome.get("evidence_insufficient")
        ):
            return True
        return False
    if capability == "citation_validation":
        return (
            outcome.get("status") == "succeeded"
            and not any(
                task.get("required") and task.get("missing_citations")
                for task in task_outcomes
            )
        )
    if capability == "financial_calculation":
        required_calculation_tasks = [
            task
            for task in task_outcomes
            if task.get("required")
            and "financial_calculation" in (task.get("capabilities") or [])
        ]
        return bool(required_calculation_tasks) and all(
            task.get("status") == "completed"
            for task in required_calculation_tasks
        )
    if capability == "resource_catalog_read":
        # System capability pre-executed by Python; the payload is injected
        # into the synthesis context.  Final synthesis is still required to
        # include the catalog data (Guarded), so this is not a free pass.
        return outcome.get("status") in {
            "succeeded",
            "verified",
            "requires_final_synthesis",
        }
    return outcome.get("status") == "succeeded"


def _attach_runtime_contract(
    *,
    result: dict[str, Any],
    route: SemanticRouteDecision,
    effective_rag_mode: str,
    retrieval_enabled: bool,
) -> dict[str, Any]:
    """Make the active orchestration contract visible even on cache replay."""

    result["runtime_revision"] = PRODUCTION_RUNTIME_REVISION
    result["runtime_contract"] = {
        "revision": PRODUCTION_RUNTIME_REVISION,
        "graph_version": result.get("graph_version"),
        "execution_round_unit": "execute_observe_result_validate",
        "max_execution_rounds": 3,
        "planner_attempts_count_as_execution_rounds": False,
        "plan_reviews_count_as_execution_rounds": False,
        "requirement_contract_monotonic": True,
        "request_requirement_contract": {
            "retrieval_requirement": route.retrieval_requirement,
            "citation_requirement": route.citation_requirement,
            "calculation_requirement": (
                "required"
                if any(
                    task.required and task.task_kind == "calculation"
                    for task in route.task_requirements
                )
                else "not_needed"
            ),
            "needs_exact_calculation": route.needs_exact_calculation,
            "required_capabilities": list(route.required_capabilities),
        },
    }
    result["effective_rag"] = {
        "enabled": retrieval_enabled,
        "mode": effective_rag_mode,
        "retrieval_requirement": route.retrieval_requirement,
        "citation_requirement": route.citation_requirement,
        "retrieval_scope": route.retrieval_scope,
    }
    loop_result = dict(result.get("agent_loop_result") or {})
    execution_round = int(
        loop_result.get("execution_round")
        or loop_result.get("completed_execution_rounds")
        or loop_result.get("agent_rounds")
        or 0
    )
    result.update(
        planner_invocation_count=int(
            loop_result.get("planner_invocation_count") or 0
        ),
        plan_attempt_in_round=int(
            loop_result.get("plan_attempt_in_round") or 0
        ),
        target_execution_round=int(
            loop_result.get("target_execution_round")
            or max(1, execution_round)
        ),
        execution_round=execution_round,
        replan_count=int(loop_result.get("replan_count") or 0),
        execution_round_history=list(
            loop_result.get("execution_round_history") or []
        ),
    )
    result["round_state"] = {
        "planner_invocation_count": result["planner_invocation_count"],
        "plan_attempt_in_round": result["plan_attempt_in_round"],
        "target_execution_round": result["target_execution_round"],
        "execution_round": result["execution_round"],
        "replan_count": result["replan_count"],
        "execution_round_history": result["execution_round_history"],
    }
    capability_outcomes = dict(result.get("capability_outcomes") or {})
    routing_degraded = route.confidence <= 0 or any(
        str(item).startswith("semantic_router_degraded:")
        for item in route.ambiguities
    )
    required_capabilities = list(route.required_capabilities)
    completed_capabilities = [
        capability
        for capability in required_capabilities
        if _capability_is_satisfied(
            capability=capability,
            outcome=capability_outcomes.get(capability, {}),
            route=route,
            task_outcomes=list(
                (result.get("completion_contract") or {}).get("task_outcomes") or []
            ),
        )
    ]
    failed_capabilities = [
        capability
        for capability in required_capabilities
        if capability not in completed_capabilities
        and (
            capability_outcomes.get(capability) or {}
        ).get("status") != "partial"
    ]
    partial_capabilities = [
        capability
        for capability in required_capabilities
        if (capability_outcomes.get(capability) or {}).get("status")
        == "partial"
    ]
    final_response_fallback = (
        str(result.get("status") or "") in {"fallback", "failed"}
        or str(result.get("finish_reason") or "") in {
            "max_output_rewrites_blocking_violation",
            "max_output_rewrites_exceeded",
            "output_guard_fallback",
        }
    )
    result.update(
        required_capabilities=required_capabilities,
        completed_capabilities=completed_capabilities,
        partial_capabilities=partial_capabilities,
        failed_capabilities=failed_capabilities,
        delivery_status=(
            "fallback"
            if final_response_fallback
            else "completed"
        ),
        routing_status="degraded" if routing_degraded else "resolved",
        overall_status=(
            "partial"
            if final_response_fallback
            and (
                result.get("final_answer")
                or completed_capabilities
            )
            else "failed"
            if final_response_fallback
            else
            "completed"
            if not failed_capabilities
            and not partial_capabilities
            and not routing_degraded
            else "partial"
            if result.get("final_answer") or completed_capabilities
            else "failed"
        ),
    )
    if routing_degraded:
        result.setdefault("completion_contract", {}).setdefault(
            "missing_requirements", []
        ).append("semantic_route_unresolved")
        result["fulfillment_status"] = "partial"
        result["execution_status"] = "degraded"
    result["status"] = result["overall_status"]
    _assert_runtime_invariants(result)
    return result


def _assert_runtime_invariants(result: dict[str, Any]) -> None:
    """Fail closed when independently produced runtime states disagree."""
    required = list(result.get("required_capabilities") or [])
    failed = list(result.get("failed_capabilities") or [])
    completed = list(result.get("completed_capabilities") or [])
    contract = dict(result.get("completion_contract") or {})
    if result.get("overall_status") == "completed" and failed:
        raise RuntimeError("invariant: completed response has failed capabilities")
    if not contract.get("missing_requirements") and set(required) - set(completed):
        raise RuntimeError("invariant: contract fulfilled but required capability missing")
    execution_round = int(result.get("execution_round") or 0)
    successful_tools = [item for item in (result.get("tool_results") or []) if item.get("success")]
    if execution_round == 0 and successful_tools:
        raise RuntimeError("invariant: successful tool result without execution round")
    if int(result.get("planner_invocation_count") or 0) < execution_round:
        raise RuntimeError("invariant: execution rounds exceed planner invocations")
    if int(result.get("replan_count") or 0) > max(0, execution_round - 1):
        raise RuntimeError("invariant: replan count exceeds completed execution rounds")


def _request_memory_snapshot(
    request: Request,
    *,
    tenant_id: str,
    user_id: str,
    request_id: str,
) -> dict[str, Any] | None:
    cache = getattr(request.app.state, "personal_request_memory_cache", None)
    if cache is None:
        cache = OrderedDict()
        request.app.state.personal_request_memory_cache = cache
    key = (tenant_id, user_id, request_id)
    snapshot = cache.get(key)
    if snapshot is not None:
        cache.move_to_end(key)
    return snapshot


def _store_request_memory_snapshot(
    request: Request,
    *,
    tenant_id: str,
    user_id: str,
    request_id: str,
    history_messages: list[dict[str, Any]],
    context_summary: str,
    memory_audit: dict[str, Any],
) -> None:
    cache = getattr(request.app.state, "personal_request_memory_cache", None)
    if cache is None:
        cache = OrderedDict()
        request.app.state.personal_request_memory_cache = cache
    key = (tenant_id, user_id, request_id)
    cache[key] = {
        "history_messages": list(history_messages),
        "context_summary": context_summary,
        "memory_audit": dict(memory_audit),
    }
    cache.move_to_end(key)
    while len(cache) > 2048:
        cache.popitem(last=False)

def _get_short_memory(request: Request) -> ShortTermMemoryService | None:
    service = getattr(request.app.state, "short_memory", None)
    if service is not None:
        return service
    settings = getattr(request.app.state, "settings", None)
    try:
        service = ShortTermMemoryService(settings=settings)
        request.app.state.short_memory = service
        return service
    except Exception:
        return None


def _get_raw_transcript_store(
    request: Request,
) -> RawTranscriptStore | None:
    store = getattr(
        request.app.state, "raw_transcript_store", None
    )
    if store is not None:
        return store
    settings = getattr(request.app.state, "settings", None)
    try:
        store = RawTranscriptStore(settings=settings)
        store.init_schema()
        request.app.state.raw_transcript_store = store
        return store
    except Exception:
        return None


def _get_long_memory(request: Request) -> LongTermMemoryService | None:
    service = getattr(request.app.state, "personal_long_memory", None)
    if service is not None:
        return service
    settings = getattr(request.app.state, "settings", None)
    try:
        service = LongTermMemoryService(settings=settings)
        service.init_schema()
        request.app.state.personal_long_memory = service
        return service
    except Exception:
        return None


def _merge_history(
    short_history: list[dict[str, Any]],
    explicit_history: list[dict[str, Any]],
    *,
    max_messages: int,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in [*short_history, *explicit_history]:
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        signature = (role, content)
        if signature in seen:
            continue
        seen.add(signature)
        # 连续的用户消息（如“需要我补充什么”+“你缺什么信息”）合并为一条，
        # 避免把相邻同角色消息直接拼进模型历史，保证对话结构干净。
        if role == "user" and merged and merged[-1]["role"] == "user":
            merged[-1]["content"] = (
                f"{merged[-1]['content']}\n{content}"
            )
        else:
            merged.append({"role": role, "content": content})
    return merged[-max(max_messages, 2) :]


_MEMORY_ALL_FACTS_LIMIT = 12
_MEMORY_RANK_TOP_K = 8
_MEMORY_RANK_LIMIT = 100
_MEMORY_MIN_SIMILARITY = 0.15


def _fact_text(fact: Any) -> str:
    value = getattr(fact, "fact_value", {}) or {}
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return (
        f"{getattr(fact, 'fact_type', '')}."
        f"{getattr(fact, 'fact_key', '')}: {value}"
    )


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _select_memory_facts(
    facts: list[Any],
    *,
    query: str,
    embedding_provider: Any,
) -> list[Any]:
    """
    长期记忆的智能注入策略（参考 LangGraph 向量记忆）：
    - 事实量少时全量注入，保证小规模场景行为稳定；
    - 事实量多时，按当前问题的语义相关性排序，只注入 top-K，
      避免无关记忆撑爆上下文。
    """
    total = list(facts)
    if not total:
        return []
    if (
        len(total) <= _MEMORY_ALL_FACTS_LIMIT
        or not query.strip()
        or embedding_provider is None
    ):
        return total[:_MEMORY_RANK_LIMIT]
    try:
        query_embedding = await asyncio.to_thread(
            embedding_provider.embed_query,
            query,
        )
        candidates = total[:_MEMORY_RANK_LIMIT]
        texts = [_fact_text(fact) for fact in candidates]
        embeddings = await asyncio.to_thread(
            embedding_provider.embed_documents,
            texts,
        )
        scored = sorted(
            (
                (_cosine_similarity(query_embedding.dense, embedding.dense), fact)
                for embedding, fact in zip(embeddings, candidates)
            ),
            key=lambda item: -item[0],
        )
        return [
            fact
            for score, fact in scored[:_MEMORY_RANK_TOP_K]
            if score >= _MEMORY_MIN_SIMILARITY
        ]
    except Exception:
        # 嵌入服务不可用时降级为全量注入，保证记忆功能不中断。
        return total[:_MEMORY_RANK_LIMIT]


async def _long_memory_context(
    facts: list[Any],
    *,
    query: str = "",
    embedding_provider: Any = None,
    blocked_fields: set[str] | None = None,
    usage: str = "answer_context",
) -> str:
    if not facts:
        return ""
    blocked = blocked_fields or set()
    eligible = [
        fact
        for fact in facts
        if getattr(fact, "status", "active") == "active"
        and (
            not (getattr(fact, "metadata", {}) or {}).get("allowed_usage")
            or usage in (getattr(fact, "metadata", {}) or {}).get("allowed_usage", [])
        )
        and f"{getattr(fact, 'fact_type', '')}.{getattr(fact, 'fact_key', '')}"
        not in blocked
    ]
    selected = await _select_memory_facts(
        eligible,
        query=query,
        embedding_provider=embedding_provider,
    )
    lines = [
        "以下是历史会话中已保存的事实（不是本轮用户输入）。"
        "只能作为背景参考，不得写入“用户原始数据”板块，"
        "也不得用于与当前问题无关的分析；"
        "每条事实的来源会话见标注："
    ]
    for fact in selected:
        value = getattr(fact, "fact_value", {})
        source_thread = str(
            getattr(fact, "source_thread_id", "") or ""
        ).strip()
        lines.append(
            f"- {getattr(fact, 'fact_type', '')}."
            f"{getattr(fact, 'fact_key', '')} = "
            f"{json.dumps(value, ensure_ascii=False)}"
            + (
                f"（来源会话：{source_thread[:24]}）"
                if source_thread
                else "（来源会话：历史上下文）"
            )
        )
    return "\n".join(lines)


async def _save_memory_after_success(
    *,
    payload: ProductionChatRequest,
    request: Request,
    request_id: str,
    result: dict[str, Any],
    short_memory: ShortTermMemoryService | None,
    long_memory: LongTermMemoryService | None,
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "short_memory_saved": False,
        "long_memory_changes": 0,
        "errors": [],
    }
    if result.get("idempotency_replayed"):
        audit["skipped_reason"] = "idempotency_replay"
        return audit

    final_answer = str(result.get("final_answer") or "").strip()
    if payload.save_memory and short_memory is not None and final_answer:
        try:
            await asyncio.to_thread(
                short_memory.save_turn,
                user_id=payload.user_id,
                thread_id=payload.thread_id,
                tenant_id=payload.tenant_id,
                user_message=payload.user_message,
                assistant_message=final_answer,
            )
            audit["short_memory_saved"] = True
        except Exception as exc:
            audit["errors"].append(
                {
                    "stage": "short_memory_write",
                    "error": type(exc).__name__,
                }
            )

    transcript_store = _get_raw_transcript_store(request)
    if transcript_store is not None and final_answer:
        try:
            await asyncio.to_thread(
                transcript_store.append_turn,
                tenant_id=payload.tenant_id,
                user_id=payload.user_id,
                thread_id=payload.thread_id,
                user_message=payload.user_message,
                assistant_message=final_answer,
                request_id=request_id,
                run_id=str(result.get("run_id") or ""),
            )
            audit["raw_transcript_saved"] = True
        except Exception as exc:
            audit["errors"].append(
                {
                    "stage": "raw_transcript_write",
                    "error": type(exc).__name__,
                }
            )

    if (
        payload.save_memory
        and payload.extract_long_memory
        and long_memory is not None
    ):
        llm_client = getattr(request.app.state, "deepseek", None)
        if llm_client is not None:
            try:
                extractor = LLMFactExtractor(
                    llm_client=llm_client,
                    memory_service=long_memory,
                )
                changes = await extractor.extract(
                    user_message=payload.user_message
                )
                promotion_gate = MemoryPromotionGate()
                blocked_promotions: list[str] = []
                filtered_changes = []
                for change in changes:
                    ok, reason = promotion_gate.may_promote(
                        fact_type=change.fact_type,
                        fact_key=change.fact_key,
                        fact_value=change.fact_value,
                    )
                    if not ok:
                        blocked_promotions.append(reason)
                        continue
                    filtered_changes.append(change)
                changes = filtered_changes
                audit["memory_promotion_blocked"] = (
                    blocked_promotions
                )
                saved = 0
                deleted = 0
                for change in changes:
                    if change.action == "delete":
                        ok = await asyncio.to_thread(
                            long_memory.delete_fact,
                            user_id=payload.user_id,
                            tenant_id=payload.tenant_id,
                            fact_type=change.fact_type,
                            fact_key=change.fact_key,
                            change_reason=change.change_reason,
                        )
                        deleted += int(ok)
                    else:
                        await asyncio.to_thread(
                            long_memory.upsert_fact,
                            user_id=payload.user_id,
                            tenant_id=payload.tenant_id,
                            fact_type=change.fact_type,
                            fact_key=change.fact_key,
                            fact_value=change.fact_value,
                            confidence=change.confidence,
                            source_thread_id=payload.thread_id,
                            source_message_id=request_id,
                            is_user_confirmed=change.is_user_confirmed,
                            change_reason=change.change_reason,
                        )
                        saved += 1
                audit["long_memory_changes"] = saved + deleted
                audit["long_memory_saved"] = saved
                audit["long_memory_deleted"] = deleted
            except Exception as exc:
                # 记忆抽取失败不影响主回答。
                audit["errors"].append(
                    {
                        "stage": "long_memory_extract",
                        "error": type(exc).__name__,
                    }
                )
    return audit


async def _persist_thread_scope(
    *,
    request: Request,
    payload: ProductionChatRequest,
    scope_plan: dict[str, Any],
) -> None:
    """Persist/clear the thread-level active resource scope."""
    service = getattr(request.app.state, "short_memory", None)
    if service is None:
        service = _get_short_memory(request)
    if service is None:
        return
    try:
        if scope_plan.get("allowed_document_ids"):
            await asyncio.to_thread(
                service.set_thread_meta,
                user_id=payload.user_id,
                thread_id=payload.thread_id,
                tenant_id=payload.tenant_id,
                metadata={
                    "active_resource_scope": {
                        "scope_id": "uploaded_documents",
                        "document_ids": scope_plan[
                            "allowed_document_ids"
                        ],
                        "updated_at": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "source": scope_plan.get("audit", {}).get(
                            "source", "explicit_selection"
                        ),
                    }
                },
            )
        elif _effective_document_scope(payload)[0] == "none":
            await asyncio.to_thread(
                service.delete_thread_meta,
                user_id=payload.user_id,
                thread_id=payload.thread_id,
                tenant_id=payload.tenant_id,
            )
    except Exception:
        pass


def _build_catalog_response(
    *,
    request_id: str,
    run_id: str | None,
    resource_catalog: list[Any],
    catalog_state: ConversationState,
    scope_plan: dict[str, Any],
    semantic_route: SemanticRouteDecision,
    control_plane_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    """Execute the resource_catalog_read system capability deterministically."""

    allowed = {
        str(item)
        for item in (scope_plan.get("allowed_document_ids") or [])
    }
    documents: list[dict[str, Any]] = []
    for ref in resource_catalog:
        document_id = catalog_state.resource_handle_map.get(
            ref.handle
        )
        if not document_id:
            continue
        if allowed and document_id not in allowed:
            continue
        documents.append(
            {
                "handle": ref.handle,
                "resource_type": ref.resource_type,
                "title": ref.title,
                "aliases": ref.aliases,
                "status": ref.status,
                "document_id": document_id,
            }
        )
    count = len(documents)
    lines = [
        f"当前知识库中可访问的文档有 {count} 个：",
    ]
    lines.extend(
        f"- {item['handle']}：{item['title']}"
        for item in documents
    )
    if not documents:
        lines.append("（当前授权范围内没有可访问文档。）")
    answer = "\n".join(lines)

    return {
        "request_id": request_id,
        "run_id": run_id,
        "status": "completed",
        "finish_reason": "catalog_direct",
        "final_answer": answer,
        "answer": answer,
        "overall_status": "completed",
        "execution_path": "catalog_direct",
        "catalog": {
            "document_count": count,
            "documents": documents,
        },
        "citations": [],
        "rag": {},
        "scope_resolution": scope_plan.get("audit") or {},
        "semantic_route": semantic_route.model_dump(
            mode="json"
        ),
        "control_plane": control_plane_audit,
        "runtime_revision": PRODUCTION_RUNTIME_REVISION,
        "synthesis_llm_provider": current_synthesis_provider(),
        "performance_summary": {
            "latency_ms": 0,
            "usage_by_stage": {},
            "synthesis_guard_tokens": {},
            "context_governance": {},
        },
    }


def _build_confirmation_response(
    *,
    request_id: str,
    run_id: str | None,
    description: str,
    scope_plan: dict[str, Any],
    semantic_route: SemanticRouteDecision,
    control_plane_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    """Deterministic confirmation request for a router-proposed action."""

    answer = (
        f"需要确认后才能执行：{description}。"
        "回复“执行”确认，或“取消”放弃。"
    )
    return {
        "request_id": request_id,
        "run_id": run_id,
        "status": "completed",
        "finish_reason": "action_confirmation_requested",
        "final_answer": answer,
        "answer": answer,
        "overall_status": "completed",
        "execution_path": "action_confirmation",
        "citations": [],
        "rag": {},
        "scope_resolution": scope_plan.get("audit") or {},
        "semantic_route": semantic_route.model_dump(
            mode="json"
        ),
        "control_plane": control_plane_audit,
        "runtime_revision": PRODUCTION_RUNTIME_REVISION,
        "synthesis_llm_provider": current_synthesis_provider(),
        "performance_summary": {
            "latency_ms": 0,
            "usage_by_stage": {},
            "synthesis_guard_tokens": {},
            "context_governance": {},
        },
    }


def _build_state_update_response(
    *,
    request_id: str,
    run_id: str | None,
    semantic_route: SemanticRouteDecision,
    scope_plan: dict[str, Any],
    control_plane_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    """Deterministic ACK for a state-update-only turn."""

    lines: list[str] = []
    for patch in semantic_route.extracted_facts:
        lines.append(
            f"- 初始事实：{patch.field} = {patch.value}"
        )
    for patch in semantic_route.fact_updates:
        lines.append(
            f"- 事实更新：{patch.field} "
            f"{patch.operation} = {patch.value}"
        )
    for patch in semantic_route.constraint_updates:
        lines.append(
            f"- 约束更新：{patch.name} -> {patch.value}"
        )
    if not lines:
        lines.append("- 无事实或约束变更")
    answer = "好的，已更新当前任务状态：\n" + "\n".join(lines)

    return {
        "request_id": request_id,
        "run_id": run_id,
        "status": "completed",
        "finish_reason": "state_update_acked",
        "final_answer": answer,
        "answer": answer,
        "overall_status": "completed",
        "execution_path": "state_update_only",
        "citations": [],
        "rag": {},
        "scope_resolution": scope_plan.get("audit") or {},
        "semantic_route": semantic_route.model_dump(
            mode="json"
        ),
        "control_plane": control_plane_audit,
        "runtime_revision": PRODUCTION_RUNTIME_REVISION,
        "synthesis_llm_provider": current_synthesis_provider(),
        "performance_summary": {
            "latency_ms": 0,
            "usage_by_stage": {},
            "synthesis_guard_tokens": {},
            "context_governance": {},
        },
    }


async def _persist_conversation_state(
    *,
    service: Any,
    payload: ProductionChatRequest,
    state: ConversationState,
    expected_state: dict[str, Any] | None = None,
    expected_version: int | None = None,
) -> dict[str, Any]:
    """Atomic critical commit with CAS and a TurnCommitReceipt.

    Task State / Fact versions / RESULT / Artifacts / Response Focus / Serial
    counters live in one ConversationState key.  The commit writes the whole
    snapshot with Redis Lua CAS: if another request already advanced the
    version, the write is rejected (STATE_VERSION_CONFLICT) and the turn fails
    instead of overwriting newer state.
    """

    if service is None:
        raise RuntimeError(
            "conversation state persistence unavailable"
        )

    before_version = int(expected_version or 0)
    persist = state.model_copy(
        update={"state_version": before_version + 1}
    )
    payload_json = persist.model_dump(mode="json")
    await asyncio.to_thread(
        service.set_conversation_state,
        user_id=payload.user_id,
        thread_id=payload.thread_id,
        tenant_id=payload.tenant_id,
        state=payload_json,
        expected_version=before_version,
        expected_state=expected_state,
    )
    state_hash = hashlib.sha256(
        json.dumps(
            payload_json,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "request_id": payload.request_id,
        "thread_id": payload.thread_id,
        "before_version": before_version,
        "after_version": before_version + 1,
        "state_hash": state_hash,
        "committed": True,
    }


def _build_state_mutation_receipt(
    *,
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    before_version: int,
    after_version: int,
) -> dict[str, Any]:
    """Diff before/after ConversationState snapshots into a mutation receipt."""

    def facts_by_field(
        state: dict[str, Any],
    ) -> dict[str, Any]:
        task = state.get("active_task") or {}
        return {
            str(fact.get("field") or ""): fact.get("value")
            for fact in (task.get("canonical_facts") or [])
            if fact.get("status") == "active"
        }

    def constraints_by_name(
        state: dict[str, Any],
    ) -> dict[str, str]:
        task = state.get("active_task") or {}
        return {
            str(item.get("name") or ""): str(
                item.get("value") or ""
            )
            for item in (task.get("user_constraints") or [])
        }

    before_facts = facts_by_field(before_state)
    after_facts = facts_by_field(after_state)
    fact_updates: list[dict[str, Any]] = []
    for field in sorted(
        set(before_facts) | set(after_facts)
    ):
        before = before_facts.get(field)
        after = after_facts.get(field)
        if before != after:
            fact_updates.append(
                {
                    "field": field,
                    "before": before,
                    "after": after,
                }
            )

    before_constraints = constraints_by_name(before_state)
    after_constraints = constraints_by_name(after_state)
    constraint_updates: list[dict[str, Any]] = []
    for name in sorted(
        set(before_constraints) | set(after_constraints)
    ):
        before = before_constraints.get(name)
        after = after_constraints.get(name)
        if before != after:
            constraint_updates.append(
                {
                    "name": name,
                    "before": before,
                    "after": after,
                }
            )

    task_handle = (
        (after_state.get("active_task") or {}).get(
            "handle"
        )
        or None
    )
    return {
        "task_handle": task_handle,
        "applied_fact_updates": fact_updates,
        "applied_constraint_updates": constraint_updates,
        "before_version": before_version,
        "after_version": after_version,
        "status": "committed",
    }


@router.post("/api/chat/graph-v2")
async def production_chat_graph(
    payload: ProductionChatRequest,
    request: Request,
) -> dict[str, Any]:
    started_at = time.perf_counter()

    settings = getattr(request.app.state, "settings", None)
    if settings is not None and settings.single_user_mode:
        identity = personal_request_identity(settings)
        payload = payload.model_copy(
            update={
                "tenant_id": identity.tenant_id,
                "user_id": identity.user_id,
            }
        )

    request_id = ensure_request_id(
        payload.request_id
        or request.headers.get("X-Request-ID")
    )
    payload = payload.model_copy(update={"request_id": request_id})
    provider = str(
        payload.synthesis_llm_provider
        or getattr(settings, "synthesis_llm_provider", "qwen")
        or "qwen"
    ).strip().lower()
    publish_event(
        "request_accepted",
        request_id=request_id,
        node="request_boundary",
        detail={"synthesis_provider": provider or "default"},
    )
    scope_mode, scope_ids = _effective_document_scope(payload)
    if (
        scope_mode != "missing"
        and payload.document_scope is None
        and not payload.document_ids
    ):
        payload = payload.model_copy(
            update={
                "document_scope": DocumentScopePayload(
                    mode=scope_mode,
                    document_ids=scope_ids,
                )
            }
        )
    # Deterministic document scope resolution happens before any LLM call.
    # This is the single scope fact source in the production chain; failures
    # for explicit/required document scopes fail fast with a business error.
    floor = ExplicitConstraintParser().parse(
        request_id=request_id,
        user_message=payload.user_message,
    )
    scope_plan = await _resolve_document_scope(
        request=request,
        payload=payload,
        constraints=floor,
        request_id=request_id,
        run_id=None,
    )
    if scope_plan["error"] is not None and (
        scope_plan["scope_requirement"] == "required"
        or scope_plan["explicit_mode"]
    ):
        raise_agent_http_exception(scope_plan["error"])
    scope_snapshot_hash = _scope_snapshot_hash(
        scope_plan["scope_snapshot"]
    )

    provider_token = set_synthesis_provider(provider)
    service = getattr(request.app.state, "production_graph_service", None)
    if service is None:
        raise_agent_http_exception(
            build_agent_error(
                code="GRAPH_SERVICE_UNAVAILABLE",
                category="unavailable",
                stage="api",
                message="生产 LangGraph 服务尚未初始化。",
                retryable=True,
                http_status=503,
                request_id=request_id,
            )
        )

    short_memory = _get_short_memory(request) if payload.use_short_memory else None
    long_memory = _get_long_memory(request) if payload.use_long_memory else None

    # 服务端加载的记忆会在首轮完成后发生变化。若相同 request_id 重试，
    # 必须复用首轮记忆快照，否则会改变 Service 的请求指纹并误报 409。
    snapshot = _request_memory_snapshot(
        request,
        tenant_id=payload.tenant_id,
        user_id=payload.user_id,
        request_id=request_id,
    )
    narrative_segments: list[dict[str, Any]] = []
    if snapshot is not None:
        final_history = list(snapshot["history_messages"])
        final_context_summary = str(snapshot["context_summary"])
        memory_audit = dict(snapshot["memory_audit"])
        memory_audit["request_memory_snapshot_reused"] = True
    else:
        memory_audit = {
            "short_memory_loaded": 0,
            "long_memory_loaded": 0,
            "degraded": [],
            "request_memory_snapshot_reused": False,
        }
        short_history: list[dict[str, Any]] = []
        if short_memory is not None:
            try:
                short_history = await asyncio.to_thread(
                    short_memory.get_messages,
                    user_id=payload.user_id,
                    thread_id=payload.thread_id,
                    tenant_id=payload.tenant_id,
                )
                memory_audit["short_memory_loaded"] = len(short_history)
            except Exception as exc:
                memory_audit["degraded"].append(
                    {"stage": "short_memory_read", "error": type(exc).__name__}
                )
        if not short_history:
            transcript_store = _get_raw_transcript_store(request)
            if transcript_store is not None:
                try:
                    raw_history = await asyncio.to_thread(
                        transcript_store.list_recent,
                        tenant_id=payload.tenant_id,
                        user_id=payload.user_id,
                        thread_id=payload.thread_id,
                        limit=80,
                    )
                    short_history = [
                        {
                            "role": str(item.get("role") or "user"),
                            "content": str(
                                item.get("content") or ""
                            ),
                        }
                        for item in raw_history
                    ]
                    memory_audit["raw_transcript_loaded"] = len(
                        short_history
                    )
                except Exception as exc:
                    memory_audit["degraded"].append(
                        {
                            "stage": "raw_transcript_read",
                            "error": type(exc).__name__,
                        }
                    )

        final_history = _merge_history(
            short_history,
            payload.history_messages,
            max_messages=(short_memory.max_messages if short_memory else 12),
        )

        final_history, history_governance = (
            select_history_messages(
                final_history,
                payload.user_message,
            )
        )
        memory_audit["history_governance"] = (
            history_governance
        )

        if short_memory is not None:
            try:
                narrative_segments = await asyncio.to_thread(
                    short_memory.get_narrative_segments,
                    user_id=payload.user_id,
                    thread_id=payload.thread_id,
                    tenant_id=payload.tenant_id,
                )
            except Exception as exc:
                memory_audit["degraded"].append(
                    {
                        "stage": "narrative_memory_read",
                        "error": type(exc).__name__,
                    }
                )
        raw_history_tokens = sum(
            estimate_tokens(str(item.get("content") or ""))
            for item in final_history
        )
        narrative_strategy = select_history_strategy(
            raw_history_tokens
        )
        memory_audit["narrative_strategy"] = {
            "raw_history_tokens": raw_history_tokens,
            "level": narrative_strategy.level,
            "should_compress": narrative_strategy.should_compress,
            "segment_count": len(narrative_segments),
        }
        if (
            narrative_strategy.should_compress
            and not narrative_segments
            and _get_raw_transcript_store(request) is not None
        ):
            try:
                llm_client = getattr(
                    request.app.state, "deepseek", None
                )
                transcript_store = _get_raw_transcript_store(
                    request
                )
                older_messages = await asyncio.to_thread(
                    transcript_store.list_recent,
                    tenant_id=payload.tenant_id,
                    user_id=payload.user_id,
                    thread_id=payload.thread_id,
                    limit=200,
                )
                compressible = [
                    {
                        "role": str(item.get("role") or "user"),
                        "content": str(
                            item.get("content") or ""
                        ),
                    }
                    for item in older_messages[:100]
                ]
                if llm_client is not None and compressible:
                    summary = await compress_messages_to_summary(
                        llm_client=llm_client,
                        messages=compressible,
                        level=narrative_strategy.level,
                    )
                    if summary:
                        narrative_segments = [
                            {
                                "segment_id": (
                                    f"SEG_{len(narrative_segments)+1}"
                                ),
                                "turn_range": (
                                    f"1-{len(compressible)//2}"
                                ),
                                "summary": summary,
                                "level": narrative_strategy.level,
                                "tokens": estimate_tokens(summary),
                            }
                        ]
                        await asyncio.to_thread(
                            short_memory.set_narrative_segments,
                            user_id=payload.user_id,
                            thread_id=payload.thread_id,
                            tenant_id=payload.tenant_id,
                            segments=narrative_segments,
                        )
                        memory_audit["narrative_compressed"] = True
            except Exception as exc:
                memory_audit["degraded"].append(
                    {
                        "stage": "narrative_compress",
                        "error": type(exc).__name__,
                    }
                )

        # LTM 读取移到 Semantic Route 解析之后，由 Memory Load Gate 决定是否
        # 物理读取（forbidden/not_needed 绝不调用 list_facts）。
        long_context = ""
        final_context_summary, context_governance = (
            trim_context_summary(payload.context_summary.strip())
        )
        memory_audit["context_governance"] = (
            context_governance
        )
        _store_request_memory_snapshot(
            request,
            tenant_id=payload.tenant_id,
            user_id=payload.user_id,
            request_id=request_id,
            history_messages=final_history,
            context_summary=final_context_summary,
            memory_audit=memory_audit,
        )

    try:
        conversation_state = default_conversation_state()
        if short_memory is not None:
            try:
                raw_state = await asyncio.to_thread(
                    short_memory.get_conversation_state,
                    user_id=payload.user_id,
                    thread_id=payload.thread_id,
                    tenant_id=payload.tenant_id,
                )
                if raw_state:
                    conversation_state = (
                        ConversationState.model_validate(
                            raw_state
                        )
                    )
            except Exception as exc:
                memory_audit[
                    "conversation_state_load_error"
                ] = type(exc).__name__

        authorized_candidates = list(
            scope_plan.get("authorized_candidates") or []
        )
        resource_catalog, catalog_state = (
            build_resource_catalog(
                authorized_candidates,
                state=conversation_state,
            )
        )
        if (
            not scope_plan.get("allowed_document_ids")
            and catalog_state.turn_count > 0
            and catalog_state.active_task is not None
            and short_memory is not None
        ):
            try:
                meta = await asyncio.to_thread(
                    short_memory.get_thread_meta,
                    user_id=payload.user_id,
                    thread_id=payload.thread_id,
                    tenant_id=payload.tenant_id,
                )
                active = (
                    (meta or {}).get(
                        "active_resource_scope"
                    )
                    or {}
                )
                if active.get("document_ids"):
                    scope_plan["allowed_document_ids"] = [
                        str(item)
                        for item in active["document_ids"]
                    ]
            except Exception:
                pass
        catalog_document_ids = [
            str(item.get("document_id") or "")
            for item in authorized_candidates
            if str(item.get("document_id") or "")
        ]
        handle_allowed_document_ids = (
            scope_plan.get("allowed_document_ids")
            or catalog_document_ids
        )
        memory_audit["resource_catalog_count"] = len(
            resource_catalog
        )
        memory_audit["conversation_state"] = (
            catalog_state.model_dump(mode="json")
        )

        semantic_route = await _resolve_semantic_route(
            payload=payload,
            request=request,
            scope_snapshot_hash=scope_snapshot_hash,
            floor=floor,
            conversation_state=catalog_state,
            recent_messages=final_history,
            resource_catalog=resource_catalog,
            capability_catalog=build_capability_catalog(),
            scope_snapshot=(
                scope_plan.get("scope_snapshot") or {}
            ),
            narrative_segments=narrative_segments,
        )
        _validate_context_references(
            route=semantic_route,
            conversation_state=catalog_state,
            resource_catalog=resource_catalog,
            allowed_document_ids=handle_allowed_document_ids,
        )
        resolved_resources = _resolved_resources_from_route(
            route=semantic_route,
            conversation_state=catalog_state,
            resource_catalog=resource_catalog,
            allowed_document_ids=handle_allowed_document_ids,
        )
        resolved_document_ids = [
            str(item.get("document_id") or "")
            for item in resolved_resources
        ]
        policy_snapshot = PolicySnapshot(
            tenant_id=payload.tenant_id,
            user_id=payload.user_id,
            knowledge_base_id=payload.knowledge_base_id,
            max_scope_document_ids=catalog_document_ids,
            allow_web=(
                semantic_route.source_authority.web == "allowed"
            ),
            allow_side_effects=bool(
                payload.allow_side_effects
            ),
            max_agent_rounds=3,
            max_tool_calls=payload.remaining_tool_calls,
        )
        task_admission = assess_task_admission(semantic_route)
        conversational_turn = (
            task_admission["kind"] == "conversational"
        )
        if not conversational_turn:
            catalog_state = apply_turn_patch(
                state=catalog_state,
                route=semantic_route,
            )
        commit_before_state = (
            conversation_state.model_dump(mode="json")
        )
        working_task_state = catalog_state.model_dump(
            mode="json"
        )
        effective_contract = build_effective_task_contract(
            state=catalog_state,
            route=semantic_route,
            resolved_resources=resolved_resources,
            evidence_requirements=[
                requirement
                for task in semantic_route.task_requirements
                for requirement in (
                    task.evidence_requirements or []
                )
            ],
        )
        resolved_result_handles = [
            str(reference.handle)
            for reference in semantic_route.result_references
            if reference.status == "resolved"
            and reference.handle
        ]
        result_artifacts = [
            item.model_dump(mode="json")
            for item in catalog_state.recent_results
            if item.handle in resolved_result_handles
        ]
        recall_citations = [
            dict(citation)
            for artifact in result_artifacts
            for citation in (artifact.get("citations") or [])
        ]
        semantic_route = semantic_route.model_copy(
            update={
                "source_authority": SourceAuthorityContract(
                    **effective_contract.source_authority
                ),
                "memory_constraint": (
                    effective_contract.memory_policy
                ),
            }
        )
        semantic_route = _apply_caller_route_constraints(
            payload=payload,
            route=semantic_route,
            scope_mode=scope_plan["audit"]["mode"],
        )
        semantic_route = _enrich_task_required_outputs(semantic_route)
        # Memory Load Gate: forbidden/not_needed 物理阻断 LTM 读取。
        memory_constraint = str(
            getattr(
                semantic_route,
                "memory_constraint",
                "not_needed",
            )
            or "not_needed"
        )
        if not memory_audit.get(
            "request_memory_snapshot_reused"
        ):
            if (
                long_memory is not None
                and memory_constraint in {"required", "optional"}
            ):
                memory_audit["long_memory_attempted"] = True
                try:
                    facts = await asyncio.to_thread(
                        long_memory.list_facts,
                        user_id=payload.user_id,
                        tenant_id=payload.tenant_id,
                    )
                    memory_audit["long_memory_loaded"] = len(
                        facts
                    )
                    long_context = await _long_memory_context(
                        facts,
                        query=payload.user_message,
                        embedding_provider=getattr(
                            request.app.state,
                            "embedding_provider",
                            None,
                        ),
                        blocked_fields={
                            str(item).strip()
                            for item in (
                                (
                                    payload.route_context
                                    or {}
                                ).get(
                                    "blocked_memory_fields"
                                )
                                or []
                            )
                            if str(item).strip()
                        },
                    )
                    memory_audit[
                        "blocked_memory_fields"
                    ] = list(
                        (
                            payload.route_context or {}
                        ).get(
                            "blocked_memory_fields"
                        )
                        or []
                    )
                except Exception as exc:
                    memory_audit["degraded"].append(
                        {
                            "stage": "long_memory_read",
                            "error": type(exc).__name__,
                        }
                    )
            else:
                memory_audit["long_memory_attempted"] = False
                memory_audit["long_memory_loaded"] = 0
                long_context = ""
            context_parts = [
                part
                for part in [
                    payload.context_summary.strip(),
                    long_context,
                ]
                if part
            ]
            raw_context_summary = "\n\n".join(context_parts)
            final_context_summary, context_governance = (
                trim_context_summary(raw_context_summary)
            )
            memory_audit["context_governance"] = (
                context_governance
            )
            _store_request_memory_snapshot(
                request,
                tenant_id=payload.tenant_id,
                user_id=payload.user_id,
                request_id=request_id,
                history_messages=final_history,
                context_summary=final_context_summary,
                memory_audit=memory_audit,
            )
        scope_plan = await _apply_route_scope_intent(
            request=request,
            payload=payload,
            scope_plan=scope_plan,
            route=semantic_route,
            floor=floor,
            request_id=request_id,
            resolved_document_ids=resolved_document_ids,
        )
        scope_snapshot_hash = _scope_snapshot_hash(
            scope_plan["scope_snapshot"]
        )
        if scope_plan["error"] is not None and (
            scope_plan["scope_requirement"] == "required"
            or scope_plan["explicit_mode"]
            or scope_plan["audit"].get("source")
            in {"semantic_resource", "semantic_title"}
        ):
            raise_agent_http_exception(scope_plan["error"])
        runtime_settings = getattr(request.app.state, "settings", None) or get_settings()
        control_plane_audit: dict[str, Any] | None = None
        if runtime_settings.control_plane_v2_execution_enabled:
            try:
                control_plane_audit = production_control_preflight(
                    request_id=request_id,
                    run_id=f"control:{request_id}",
                    user_message=payload.user_message,
                    route=semantic_route,
                    constraints=floor,
                    scopes=(
                        [scope_plan["resolved_scope"]]
                        if scope_plan["resolved_scope"] is not None
                        else []
                    ),
                )
            except ControlPlaneBlocked as exc:
                error = build_agent_error(
                    code="CONTROL_PLANE_CONTRACT_BLOCKED",
                    category="conflict",
                    stage="api",
                    message="请求与当前控制面契约冲突，无法执行。",
                    retryable=False,
                    http_status=422,
                    request_id=request_id,
                    run_id=f"control:{request_id}",
                    details={
                        "reason_codes": list(exc.reason_codes),
                        "control_plane": exc.audit,
                    },
                )
                raise_agent_http_exception(error)
        publish_event(
            "semantic_route_resolved",
            request_id=request_id,
            node="semantic_router",
            detail={
                "mode": semantic_route.orchestration_mode,
                "required_capabilities": semantic_route.required_capabilities,
                "confidence": semantic_route.confidence,
            },
        )
        retrieval_enabled = (
            payload.enable_rag
            and payload.rag_mode != "off"
            and semantic_route.retrieval_requirement != "not_needed"
        )
        effective_rag_mode = (
            "required"
            if semantic_route.retrieval_requirement == "required"
            else payload.rag_mode
        )
        effective_payload = payload.model_copy(
            update={
                "enable_rag": retrieval_enabled,
                "rag_mode": effective_rag_mode,
                "document_ids": (
                    resolved_document_ids
                    or scope_plan["allowed_document_ids"]
                    or payload.document_ids
                ),
            }
        )
        if conversational_turn:
            # Conversational Direct: no RAG, no tools, no business state
            # mutation, no Task allocation.  The route has no capabilities by
            # definition, so this only guarantees the safety boundary.
            retrieval_enabled = False
            effective_rag_mode = "off"
            effective_payload = effective_payload.model_copy(
                update={
                    "enable_rag": False,
                    "rag_mode": "off",
                }
            )
        if semantic_route.proposed_action is not None:
            confirmation_response = (
                _build_confirmation_response(
                    request_id=request_id,
                    run_id=None,
                    description=(
                        semantic_route.proposed_action.description
                    ),
                    scope_plan=scope_plan,
                    semantic_route=semantic_route,
                    control_plane_audit=control_plane_audit,
                )
            )
            confirmation_response["performance_summary"] = (
                _build_performance_summary(
                    started_at=started_at,
                    result=confirmation_response,
                    retrieval_outcome={},
                    memory_audit=memory_audit,
                )
            )
            confirmation_response["policy_snapshot"] = (
                policy_snapshot.model_dump(mode="json")
            )
            confirmation_response["effective_task_contract"] = (
                effective_contract.model_dump(mode="json")
            )
            catalog_state = update_conversation_state(
                state=catalog_state,
                semantic_route=semantic_route,
                final_answer=(
                    confirmation_response["final_answer"]
                ),
                resolved_resources=resolved_resources,
                proposed_action={
                    "action_type": (
                        semantic_route.proposed_action.action_type
                    ),
                    "description": (
                        semantic_route.proposed_action.description
                    ),
                    "proposed_by": (
                        semantic_route.proposed_action.proposed_by
                    ),
                },
                completed=True,
                result_artifact=build_result_artifact(
                    result=confirmation_response,
                    route=semantic_route,
                ),
            )
            confirmation_response["commit_observability"] = {
                "before": commit_before_state,
                "working": working_task_state,
                "after": catalog_state.model_dump(mode="json"),
            }
            turn_commit_receipt = await _persist_conversation_state(
                service=short_memory,
                payload=payload,
                state=catalog_state,
                expected_state=commit_before_state,
                expected_version=(
                    conversation_state.state_version
                ),
            )
            confirmation_response["turn_commit_receipt"] = (
                turn_commit_receipt
            )
            catalog_state = catalog_state.model_copy(
                update={
                    "state_version": (
                        turn_commit_receipt["after_version"]
                    )
                }
            )
            confirmation_response["commit_observability"]["after"] = (
                catalog_state.model_dump(mode="json")
            )
            confirmation_response["mutation_receipt"] = (
                _build_state_mutation_receipt(
                    before_state=commit_before_state,
                    after_state=catalog_state.model_dump(
                        mode="json"
                    ),
                    before_version=(
                        turn_commit_receipt["before_version"]
                    ),
                    after_version=(
                        turn_commit_receipt["after_version"]
                    ),
                )
            )
            write_audit = await _save_memory_after_success(
                payload=payload,
                request=request,
                request_id=request_id,
                result=confirmation_response,
                short_memory=short_memory,
                long_memory=long_memory,
            )
            confirmation_response["personal_memory"] = {
                **memory_audit,
                **write_audit,
            }
            await _persist_thread_scope(
                request=request,
                payload=payload,
                scope_plan=scope_plan,
            )
            return confirmation_response
        if semantic_route.state_update_only:
            if not (
                semantic_route.fact_updates
                or semantic_route.extracted_facts
                or semantic_route.constraint_updates
            ):
                raise_agent_http_exception(
                    build_agent_error(
                        code="SEMANTIC_CONTRACT_UNRESOLVED",
                        category="protocol",
                        stage="api",
                        message=(
                            "语义合同未解析：state_update_only "
                            "缺少语义变更。"
                        ),
                        retryable=False,
                        http_status=422,
                        request_id=request_id,
                        details={
                            "reason_codes": [
                                "STATE_UPDATE_ONLY_WITHOUT_MUTATION"
                            ]
                        },
                    )
                )
            state_update_response = (
                _build_state_update_response(
                    request_id=request_id,
                    run_id=None,
                    semantic_route=semantic_route,
                    scope_plan=scope_plan,
                    control_plane_audit=control_plane_audit,
                )
            )
            state_update_response["performance_summary"] = (
                _build_performance_summary(
                    started_at=started_at,
                    result=state_update_response,
                    retrieval_outcome={},
                    memory_audit=memory_audit,
                )
            )
            state_update_response["policy_snapshot"] = (
                policy_snapshot.model_dump(mode="json")
            )
            state_update_response["effective_task_contract"] = (
                effective_contract.model_dump(mode="json")
            )
            catalog_state = update_conversation_state(
                state=catalog_state,
                semantic_route=semantic_route,
                final_answer=(
                    state_update_response["final_answer"]
                ),
                resolved_resources=resolved_resources,
                proposed_action=None,
                completed=True,
                result_artifact=build_result_artifact(
                    result=state_update_response,
                    route=semantic_route,
                ),
            )
            state_update_response["commit_observability"] = {
                "before": commit_before_state,
                "working": working_task_state,
                "after": catalog_state.model_dump(mode="json"),
            }
            turn_commit_receipt = await _persist_conversation_state(
                service=short_memory,
                payload=payload,
                state=catalog_state,
                expected_state=commit_before_state,
                expected_version=(
                    conversation_state.state_version
                ),
            )
            state_update_response["turn_commit_receipt"] = (
                turn_commit_receipt
            )
            catalog_state = catalog_state.model_copy(
                update={
                    "state_version": (
                        turn_commit_receipt["after_version"]
                    )
                }
            )
            state_update_response["commit_observability"]["after"] = (
                catalog_state.model_dump(mode="json")
            )
            state_update_response["mutation_receipt"] = (
                _build_state_mutation_receipt(
                    before_state=commit_before_state,
                    after_state=catalog_state.model_dump(
                        mode="json"
                    ),
                    before_version=(
                        turn_commit_receipt["before_version"]
                    ),
                    after_version=(
                        turn_commit_receipt["after_version"]
                    ),
                )
            )
            write_audit = await _save_memory_after_success(
                payload=payload,
                request=request,
                request_id=request_id,
                result=state_update_response,
                short_memory=short_memory,
                long_memory=long_memory,
            )
            state_update_response["personal_memory"] = {
                **memory_audit,
                **write_audit,
            }
            await _persist_thread_scope(
                request=request,
                payload=payload,
                scope_plan=scope_plan,
            )
            return state_update_response
        catalog_capability_required = (
            "resource_catalog_read"
            in semantic_route.required_capabilities
        )
        pending_catalog_confirmed = bool(
            semantic_route.pending_action_resolution.status
            == "confirmed"
            and catalog_state.pending_action is not None
            and catalog_state.pending_action.action_type
            == "resource_catalog_query"
        )
        catalog_sole_capability = set(
            semantic_route.required_capabilities
        ) <= {"resource_catalog_read", "general_explanation"}
        if (
            catalog_sole_capability
            and (
                catalog_capability_required
                or pending_catalog_confirmed
            )
        ):
            catalog_direct = _build_catalog_response(
                request_id=request_id,
                run_id=None,
                resource_catalog=resource_catalog,
                catalog_state=catalog_state,
                scope_plan=scope_plan,
                semantic_route=semantic_route,
                control_plane_audit=control_plane_audit,
            )
            catalog_direct["performance_summary"] = (
                _build_performance_summary(
                    started_at=started_at,
                    result=catalog_direct,
                    retrieval_outcome={},
                    memory_audit=memory_audit,
                )
            )
            catalog_direct["policy_snapshot"] = (
                policy_snapshot.model_dump(mode="json")
            )
            catalog_direct["effective_task_contract"] = (
                effective_contract.model_dump(mode="json")
            )
            catalog_state = update_conversation_state(
                state=catalog_state,
                semantic_route=semantic_route,
                final_answer=catalog_direct["final_answer"],
                resolved_resources=resolved_resources,
                proposed_action=None,
                completed=True,
                result_artifact=build_result_artifact(
                    result=catalog_direct,
                    route=semantic_route,
                ),
            )
            catalog_direct["commit_observability"] = {
                "before": commit_before_state,
                "working": working_task_state,
                "after": catalog_state.model_dump(mode="json"),
            }
            turn_commit_receipt = (
                await _persist_conversation_state(
                    service=short_memory,
                    payload=payload,
                    state=catalog_state,
                    expected_state=commit_before_state,
                    expected_version=(
                        conversation_state.state_version
                    ),
                )
            )
            catalog_direct["turn_commit_receipt"] = (
                turn_commit_receipt
            )
            catalog_state = catalog_state.model_copy(
                update={
                    "state_version": (
                        turn_commit_receipt["after_version"]
                    )
                }
            )
            catalog_direct["commit_observability"]["after"] = (
                catalog_state.model_dump(mode="json")
            )
            catalog_direct["mutation_receipt"] = (
                _build_state_mutation_receipt(
                    before_state=commit_before_state,
                    after_state=catalog_state.model_dump(
                        mode="json"
                    ),
                    before_version=(
                        turn_commit_receipt["before_version"]
                    ),
                    after_version=(
                        turn_commit_receipt["after_version"]
                    ),
                )
            )
            write_audit = await _save_memory_after_success(
                payload=payload,
                request=request,
                request_id=request_id,
                result=catalog_direct,
                short_memory=short_memory,
                long_memory=long_memory,
            )
            catalog_direct["personal_memory"] = {
                **memory_audit,
                **write_audit,
            }
            await _persist_thread_scope(
                request=request,
                payload=payload,
                scope_plan=scope_plan,
            )
            return catalog_direct
        allow_rag_direct = _route_allows_rag_direct(semantic_route) or _legacy_rag_direct_is_safe(
            payload, request
        )
        if retrieval_enabled:
            publish_event(
                "rag_started",
                request_id=request_id,
                node="rag_subgraph",
                detail={
                    "mode": effective_rag_mode,
                    "semantic_requirement": (
                        semantic_route.retrieval_requirement
                    ),
                },
            )
        rag_direct, rag_audit, rag_result = await _run_rag_attempt(
            payload=effective_payload,
            request=request,
            request_id=request_id,
            history_messages=final_history,
            allow_direct=allow_rag_direct,
            required_failure_is_fatal=allow_rag_direct,
            retrieval_queries=_retrieval_queries(
                semantic_route, payload.user_message
            ),
            skip_answer_cache=bool(scope_plan["skip_answer_cache"]),
            scope_snapshot_hash=scope_snapshot_hash,
        )
        retrieval_outcome = _rag_outcome(rag_audit, rag_result)
        if (
            semantic_route.retrieval_requirement == "required"
            and not rag_audit.get("attempted")
        ):
            _get_metrics(request).increment(
                "contract_execution_violation_total"
            )
            raise_agent_http_exception(
                build_agent_error(
                    code="CONTRACT_EXECUTION_VIOLATION",
                    category="protocol",
                    stage="api",
                    message="必须执行的文档检索没有执行。",
                    retryable=False,
                    http_status=500,
                    request_id=request_id,
                    details={
                        "reason_codes": [
                            "CONTRACT_EXECUTION_VIOLATION"
                        ]
                    },
                )
            )
        citation_violations = _citation_scope_violations(
            rag_result,
            scope_plan["scope_snapshot"],
        )
        citation_required = bool(
            semantic_route.citation_requirement == "required"
            or scope_plan["scope_requirement"] == "required"
        )
        if citation_violations:
            if citation_required:
                _get_metrics(request).increment(
                    "citation_scope_violation_total"
                )
                raise_agent_http_exception(
                    build_agent_error(
                        code="CITATION_SCOPE_VIOLATION",
                        category="internal",
                        stage="final_response",
                        message="最终引用超出了已解析的文档范围。",
                        retryable=False,
                        http_status=500,
                        request_id=request_id,
                        details={
                            "reason_codes": [
                                "CITATION_SCOPE_VIOLATION"
                            ],
                            "document_ids": citation_violations,
                        },
                    )
                )
            allowed_ids = set(
                scope_plan["allowed_document_ids"]
            )
            rag_result["citations"] = [
                citation
                for citation in (rag_result.get("citations") or [])
                if str(citation.get("document_id") or "") in allowed_ids
            ]
        memory_audit["rag"] = rag_audit
        if retrieval_enabled:
            publish_event(
                "rag_finished",
                request_id=request_id,
                node="rag_subgraph",
                status="completed",
                detail={
                    "attempted": rag_audit.get("attempted", False),
                    "sufficient": rag_audit.get("sufficient", False),
                    "retrieved_count": rag_audit.get("retrieved_count", 0),
                    "reranked_count": rag_audit.get("reranked_count", 0),
                    "evidence_candidate_count": rag_audit.get(
                        "evidence_candidate_count", 0
                    ),
                    "sufficient_evidence_count": rag_audit.get(
                        "sufficient_evidence_count", 0
                    ),
                    "citation_count": rag_audit.get("citation_count", 0),
                    "evidence_rejection_reason": rag_audit.get(
                        "evidence_rejection_reason"
                    ),
                    "retrieval_status": rag_audit.get("retrieval_status"),
                    "rerank_status": rag_audit.get("rerank_status"),
                    "evidence_assessment_status": rag_audit.get(
                        "evidence_assessment_status"
                    ),
                    "conflict_detection_status": rag_audit.get(
                        "conflict_detection_status"
                    ),
                    "protocol_error_stage": rag_audit.get(
                        "protocol_error_stage"
                    ),
                },
            )
        if rag_direct is not None:
            if control_plane_audit is not None:
                rag_direct["control_plane"] = control_plane_audit
            rag_direct["scope_resolution"] = scope_plan["audit"]
            rag_direct = _apply_completion_contract(
                result=rag_direct,
                route=semantic_route,
                rag_outcome=retrieval_outcome,
                memory_audit=memory_audit,
            )
            rag_direct = _attach_runtime_contract(
                result=rag_direct,
                route=semantic_route,
                effective_rag_mode=effective_rag_mode,
                retrieval_enabled=retrieval_enabled,
            )
            if not rag_direct.get("idempotency_replayed"):
                write_audit = await _save_memory_after_success(
                    payload=payload,
                    request=request,
                    request_id=request_id,
                    result=rag_direct,
                    short_memory=short_memory,
                    long_memory=long_memory,
                )
                rag_direct["personal_memory"] = {
                    **memory_audit,
                    **write_audit,
                }
            else:
                rag_direct["personal_memory"] = {
                    **memory_audit,
                    "skipped_reason": "idempotency_replay",
                }
            await _persist_thread_scope(
                request=request,
                payload=payload,
                scope_plan=scope_plan,
            )
            rag_direct["performance_summary"] = (
                _build_performance_summary(
                    started_at=started_at,
                    result=rag_direct,
                    retrieval_outcome=retrieval_outcome,
                    memory_audit=memory_audit,
                )
            )
            rag_direct["policy_snapshot"] = (
                policy_snapshot.model_dump(mode="json")
            )
            rag_direct["effective_task_contract"] = (
                effective_contract.model_dump(mode="json")
            )
            catalog_state = update_conversation_state(
                state=catalog_state,
                semantic_route=semantic_route,
                final_answer=(
                    rag_direct.get("final_answer")
                    or rag_direct.get("answer")
                    or ""
                ),
                resolved_resources=resolved_resources,
                proposed_action=None,
                completed=True,
                result_artifact=build_result_artifact(
                    result=rag_direct,
                    route=semantic_route,
                ),
            )
            rag_direct["commit_observability"] = {
                "before": commit_before_state,
                "working": working_task_state,
                "after": catalog_state.model_dump(mode="json"),
            }
            turn_commit_receipt = (
                await _persist_conversation_state(
                    service=short_memory,
                    payload=payload,
                    state=catalog_state,
                    expected_state=commit_before_state,
                    expected_version=(
                        conversation_state.state_version
                    ),
                )
            )
            rag_direct["turn_commit_receipt"] = (
                turn_commit_receipt
            )
            catalog_state = catalog_state.model_copy(
                update={
                    "state_version": (
                        turn_commit_receipt["after_version"]
                    )
                }
            )
            rag_direct["commit_observability"]["after"] = (
                catalog_state.model_dump(mode="json")
            )
            rag_direct["mutation_receipt"] = (
                _build_state_mutation_receipt(
                    before_state=commit_before_state,
                    after_state=catalog_state.model_dump(
                        mode="json"
                    ),
                    before_version=(
                        turn_commit_receipt["before_version"]
                    ),
                    after_version=(
                        turn_commit_receipt["after_version"]
                    ),
                )
            )
            return rag_direct

        allowed_groups = (
            []
            if conversational_turn
            else list(payload.allowed_tool_groups)
        )
        if retrieval_enabled and "knowledge_retrieval" not in allowed_groups:
            allowed_groups.append("knowledge_retrieval")

        memory_audit["constraint"] = semantic_route.memory_constraint
        if semantic_route.memory_constraint in {"forbidden", "not_needed"}:
            # Memory is not allowed for this request: drop any loaded long-term
            # context so it never reaches Synthesis.
            final_context_summary = payload.context_summary.strip()
        evidence_context = _rag_evidence_context(rag_result)
        retrieval_outcome["context_governance"] = (
            (rag_result or {}).get("context_governance")
            or {}
        )
        graph_context_parts = [part for part in [final_context_summary, evidence_context] if part]
        catalog_payload: dict[str, Any] | None = None
        if catalog_capability_required:
            catalog_payload = {
                "document_count": len(resource_catalog),
                "documents": [
                    {
                        "handle": ref.handle,
                        "resource_type": ref.resource_type,
                        "title": ref.title,
                        "aliases": ref.aliases,
                    }
                    for ref in resource_catalog
                ],
            }
            graph_context_parts.append(
                "<resource_catalog>\n"
                + json.dumps(
                    catalog_payload,
                    ensure_ascii=False,
                )
                + "\n</resource_catalog>"
            )
        graph_context_parts.append(
            "<effective_task_contract>\n"
            + json.dumps(
                effective_contract.model_dump(mode="json"),
                ensure_ascii=False,
            )
            + "\n</effective_task_contract>"
        )
        if result_artifacts:
            graph_context_parts.append(
                "<result_artifacts>\n"
                + json.dumps(
                    result_artifacts,
                    ensure_ascii=False,
                )
                + "\n</result_artifacts>"
            )
        graph_context_summary = "\n\n".join(graph_context_parts)
        technical_failures: list[dict[str, Any]] = []
        for item in (
            retrieval_outcome.get("requirement_coverage")
            or []
        ):
            status = str(item.get("status") or "")
            if status in {
                "technical_unavailable",
                "assessment_protocol_failed",
            }:
                technical_failures.append(
                    {
                        "requirement_id": str(
                            item.get("requirement_id") or ""
                        ),
                        "stage": (
                            "evidence_assessment"
                            if status
                            == "assessment_protocol_failed"
                            else "retrieval"
                        ),
                        "status": status,
                    }
                )
        for item in (memory_audit.get("degraded") or []):
            technical_failures.append(
                {
                    "requirement_id": None,
                    "stage": str(
                        item.get("stage") or "unknown"
                    ),
                    "status": "technical_unavailable",
                    "error": str(item.get("error") or ""),
                }
            )
        semantic_context = {
            **(payload.route_context or {}),
            "complexity": (payload.route_context or {}).get("complexity", "medium"),
            "risk_level": semantic_route.risk_level,
            "orchestration_mode": semantic_route.orchestration_mode,
            "semantic_route": semantic_route.model_dump(mode="json"),
            "policy_snapshot": (
                policy_snapshot.model_dump(mode="json")
            ),
            "effective_task_contract": (
                effective_contract.model_dump(mode="json")
            ),
            "resolved_result_artifacts": result_artifacts,
            "narrative_memory": narrative_segments,
            "narrative_strategy": (
                memory_audit.get("narrative_strategy") or {}
            ),
            "conversation_state": (
                catalog_state.model_dump(mode="json")
            ),
            "resolved_resources": resolved_resources,
            "resource_catalog": [
                ref.model_dump(mode="json")
                for ref in resource_catalog
            ],
            "resource_catalog_payload": catalog_payload,
            "effective_conversation_relation": (
                semantic_route.conversation_relation
            ),
            "retrieval_outcome": retrieval_outcome,
            "control_plane": control_plane_audit,
            "allowed_document_ids": scope_plan["allowed_document_ids"],
            "scope_snapshot": scope_plan["scope_snapshot"] or {},
            "technical_failures": technical_failures,
        }

        effective_execution_policy: ExecutionPolicy = payload.execution_policy
        if (
            "financial_calculation" in semantic_route.required_capabilities
            and semantic_route.needs_exact_calculation
        ):
            effective_execution_policy = "require_tool"

        result = await service.run(
            request_id=request_id,
            user_message=payload.user_message,
            user_id=payload.user_id,
            thread_id=payload.thread_id,
            tenant_id=payload.tenant_id,
            knowledge_base_id=payload.knowledge_base_id,
            history_messages=final_history,
            context_summary=graph_context_summary,
            route_context=semantic_context,
            citations=(
                list((rag_result or {}).get("citations") or [])
                + recall_citations
            ),
            allowed_tool_names=payload.allowed_tool_names,
            allowed_tool_groups=allowed_groups,
            remaining_tool_calls=payload.remaining_tool_calls,
            allow_side_effects=payload.allow_side_effects,
            execution_policy=effective_execution_policy,
        )
        if result.get("finish_reason") == "citation_scope_violation":
            _get_metrics(request).increment(
                "citation_scope_violation_total"
            )
            raise_agent_http_exception(
                build_agent_error(
                    code="CITATION_SCOPE_VIOLATION",
                    category="internal",
                    stage="final_response",
                    message="最终回答引用了指定范围之外的文档。",
                    retryable=False,
                    http_status=500,
                    request_id=request_id,
                    run_id=result.get("run_id"),
                    details={
                        "reason_codes": [
                            "CITATION_SCOPE_VIOLATION"
                        ]
                    },
                )
            )
        if control_plane_audit is not None:
            result["control_plane"] = control_plane_audit
        result["rag"] = rag_result
        final_response_result = (
            result.get("final_response_result") or {}
        )
        if isinstance(final_response_result, str):
            try:
                final_response_result = json.loads(
                    final_response_result
                )
            except json.JSONDecodeError:
                final_response_result = {}
        if not isinstance(final_response_result, dict):
            final_response_result = {}
        synthesis_data = (
            final_response_result.get("synthesis")
            if isinstance(
                final_response_result.get("synthesis"),
                dict,
            )
            else {}
        )
        # P0 chain: Python materializes and verifies CALC artifacts BEFORE
        # capability observation.  A verified current-turn CALC is bound into
        # used_derivation_ids so financial_calculation can only reach
        # satisfied via a real Verified CALC (or a successful tool result).
        materialized = materialize_new_artifacts(
            catalog_state,
            synthesis_data,
        )
        verified_new_calc_handles = [
            str(item.get("handle") or "")
            for item in materialized["artifacts"]
            if str(item.get("artifact_type") or "").lower()
            in {"calc", "calculation"}
            and item.get("verification_status") == "verified"
            and item.get("output") is not None
        ]
        if verified_new_calc_handles:
            existing_derivations = [
                str(item)
                for item in (
                    synthesis_data.get("used_derivation_ids")
                    or []
                )
            ]
            synthesis_data["used_derivation_ids"] = list(
                dict.fromkeys(
                    [
                        *existing_derivations,
                        *verified_new_calc_handles,
                    ]
                )
            )
            final_response_result["synthesis"] = synthesis_data
            result["final_response_result"] = (
                final_response_result
            )
        result = _apply_completion_contract(
            result=result,
            route=semantic_route,
            rag_outcome=retrieval_outcome,
            memory_audit=memory_audit,
            materialized_artifacts=materialized["artifacts"],
        )
        result = _attach_runtime_contract(
            result=result,
            route=semantic_route,
            effective_rag_mode=effective_rag_mode,
            retrieval_enabled=retrieval_enabled,
        )
        result["execution_path"] = "agent_path"
        result["synthesis_llm_provider"] = current_synthesis_provider()
        result["scope_resolution"] = scope_plan["audit"]
        write_audit = await _save_memory_after_success(
            payload=payload,
            request=request,
            request_id=request_id,
            result=result,
            short_memory=short_memory,
            long_memory=long_memory,
        )
        result["personal_memory"] = {**memory_audit, **write_audit}
        await _persist_thread_scope(
            request=request,
            payload=payload,
            scope_plan=scope_plan,
        )
        result["performance_summary"] = (
            _build_performance_summary(
                started_at=started_at,
                result=result,
                retrieval_outcome=retrieval_outcome,
                memory_audit=memory_audit,
            )
        )
        proposed_action = (
            (final_response_result.get("synthesis") or {}).get(
                "proposed_action"
            )
            if isinstance(
                final_response_result.get("synthesis"),
                dict,
            )
            else None
        )
        result_artifact = build_result_artifact(
            result=result,
            route=semantic_route,
        )
        for artifact_item in materialized["artifacts"]:
            artifact_type = str(
                artifact_item.get("artifact_type") or ""
            )
            target = (
                "calculations"
                if artifact_type in {"calc", "calculation"}
                else "claims"
                if artifact_type == "claim"
                else "conclusions"
            )
            result_artifact.setdefault(target, []).append(
                {
                    key: value
                    for key, value in artifact_item.items()
                    if key
                    not in {
                        "local_key",
                        "artifact_type",
                        "grounding",
                    }
                }
            )
            result_artifact.setdefault(
                "sub_artifact_handles", []
            ).append(artifact_item.get("handle"))
        result_artifact["materialized_artifacts"] = (
            materialized["artifacts"]
        )
        catalog_state = update_conversation_state(
            state=catalog_state,
            semantic_route=semantic_route,
            final_answer=(
                result.get("final_answer")
                or result.get("answer")
                or ""
            ),
            resolved_resources=resolved_resources,
            proposed_action=(
                dict(proposed_action)
                if isinstance(proposed_action, dict)
                else None
            ),
            completed=(
                not (
                    (result.get("completion_contract") or {}).get(
                        "missing_requirements"
                    )
                    or []
                )
                and not semantic_route.needs_clarification
            ),
            admit_task=bool(task_admission["admitted"]),
            result_artifact=result_artifact,
            response_focus_candidate=materialized["focus"],
        )
        result["task_admission"] = task_admission
        result["commit_observability"] = {
            "before": commit_before_state,
            "working": working_task_state,
            "after": catalog_state.model_dump(mode="json"),
        }
        result["policy_snapshot"] = (
            policy_snapshot.model_dump(mode="json")
        )
        result["effective_task_contract"] = (
            effective_contract.model_dump(mode="json")
        )
        turn_commit_receipt = await _persist_conversation_state(
            service=short_memory,
            payload=payload,
            state=catalog_state,
            expected_state=commit_before_state,
            expected_version=(
                conversation_state.state_version
            ),
        )
        result["turn_commit_receipt"] = turn_commit_receipt
        catalog_state = catalog_state.model_copy(
            update={
                "state_version": (
                    turn_commit_receipt["after_version"]
                )
            }
        )
        result["commit_observability"]["after"] = (
            catalog_state.model_dump(mode="json")
        )
        result["mutation_receipt"] = (
            _build_state_mutation_receipt(
                before_state=commit_before_state,
                after_state=catalog_state.model_dump(
                    mode="json"
                ),
                before_version=(
                    turn_commit_receipt["before_version"]
                ),
                after_version=(
                    turn_commit_receipt["after_version"]
                ),
            )
        )
        committed_result_handle = ""
        if catalog_state.recent_results:
            committed_result_handle = str(
                catalog_state.recent_results[0].handle or ""
            )
        committed_calc_refs = [
            f"{committed_result_handle}."
            f"{str(calc.get('handle') or '')}"
            for calc in (result_artifact.get("calculations") or [])
            if str(calc.get("handle") or "").startswith("CALC_")
        ]
        result["committed_result_refs"] = committed_calc_refs
        calc_outcome = (
            (result.get("capability_outcomes") or {}).get(
                "financial_calculation"
            )
        )
        if isinstance(calc_outcome, dict):
            calc_outcome["committed_result_refs"] = (
                committed_calc_refs
            )
            result["capability_outcomes"][
                "financial_calculation"
            ] = calc_outcome
        return result

    except HTTPException:
        raise
    except Exception as exc:
        error = exception_to_agent_error(
            exc,
            stage="api",
            request_id=request_id,
        )
        log_payload = {
            "request_id": request_id,
            "error_id": error.error_id,
            "error_code": error.code,
            "error_type": type(exc).__name__,
            "http_status": error.http_status,
        }
        if error.http_status >= 500:
            log_event(
                logger,
                "exception",
                "production_chat_graph_failed",
                **log_payload,
            )
        else:
            log_event(
                logger,
                "warning",
                "production_chat_graph_rejected",
                **log_payload,
            )
        raise_agent_http_exception(error)
    finally:
        if provider_token is not None:
            reset_synthesis_provider(provider_token)


def _sse(event: dict[str, Any]) -> str:
    name = str(event.get("event") or "message")
    payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    return f"event: {name}\ndata: {payload}\n\n"


@router.post("/api/chat/graph-v2/stream")
async def production_chat_graph_stream(
    payload: ProductionChatRequest,
    request: Request,
) -> StreamingResponse:
    """Stream safe workflow events, then the guarded final response.

    Model reasoning and unguarded answer tokens are never sent to the browser.
    """

    async def event_stream():
        request_id = ensure_request_id(
            payload.request_id
            or request.headers.get("X-Request-ID")
        )
        stream_payload = payload.model_copy(
            update={"request_id": request_id}
        )
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)

        def sink(event: dict[str, Any]) -> None:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Progress events are advisory; dropping one must not break the
                # financial answer or bypass final validation.
                return

        def enqueue_terminal(_task: Any) -> None:
            try:
                queue.put_nowait({"event": "__terminal__"})
            except asyncio.QueueFull:
                # The queue is full of real events; they will all be yielded
                # before the consumer notices the task finished.
                return

        token = set_event_sink(sink)
        task = asyncio.create_task(
            production_chat_graph(stream_payload, request)
        )
        task.add_done_callback(enqueue_terminal)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if event.get("event") == "__terminal__":
                    break
                yield _sse(event)

            result = await task
            if not isinstance(result, dict):
                raise RuntimeError(
                    "production_chat_graph returned a non-dict result"
                )
            yield _sse(
                {
                    "event": "completed",
                    "request_id": result.get("request_id"),
                    "status": result.get("status"),
                    "result": result,
                }
            )
        except HTTPException as exc:
            detail = exc.detail
            if isinstance(detail, dict) and detail.get("error_id"):
                error = detail
            else:
                error = build_agent_error(
                    code="AGENT_HTTP_ERROR",
                    category="validation",
                    stage="api",
                    message=str(detail) or "请求处理失败。",
                    retryable=False,
                    http_status=exc.status_code,
                    request_id=request_id,
                    details={},
                ).model_dump(mode="json")
            logger.warning(
                "production_chat_stream_failed",
                request_id=request_id,
                error_code=error.get("code"),
                error_id=error.get("error_id"),
                reason_codes=(error.get("details") or {}).get(
                    "reason_codes"
                ),
            )
            yield _sse(
                {
                    "event": "error",
                    "status": "failed",
                    "request_id": request_id,
                    "error": error,
                    # Backward-compatible alias during the protocol migration.
                    "detail": error,
                }
            )
        except Exception as exc:
            logger.exception(
                "production_chat_stream_unhandled",
                request_id=request_id,
                error_type=type(exc).__name__,
            )
            error = build_agent_error(
                code="AGENT_INTERNAL_ERROR",
                category="internal",
                stage="api",
                message="生产 Agent 执行失败，请稍后重试。",
                retryable=True,
                http_status=500,
                request_id=request_id,
                details={"exception_type": type(exc).__name__},
            ).model_dump(mode="json")
            yield _sse(
                {
                    "event": "error",
                    "status": "failed",
                    "request_id": request_id,
                    "error": error,
                    "detail": error,
                }
            )
        except asyncio.CancelledError:
            logger.warning(
                "production_chat_stream_client_disconnected",
                request_id=request_id,
            )
            task.cancel()
            raise
        finally:
            if not task.done():
                task.cancel()
            reset_event_sink(token)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/api/chat/history/{thread_id}")
async def delete_chat_history(
    thread_id: str,
    request: Request,
    user_id: str,
    tenant_id: str = "default",
) -> dict[str, Any]:
    """清空某个会话的短期记忆（Redis），用于前端删除会话时同步清理。"""
    settings = getattr(request.app.state, "settings", None)
    clean_thread = str(thread_id).strip()
    clean_user = (
        personal_request_identity(settings).user_id
        if settings is not None and settings.single_user_mode
        else str(user_id).strip()
    )
    tenant_id = (
        personal_request_identity(settings).tenant_id
        if settings is not None and settings.single_user_mode
        else tenant_id
    )
    fallback_request_id = f"api-prod-{uuid4()}"
    if not clean_thread or not clean_user:
        raise_agent_http_exception(
            build_agent_error(
                code="INVALID_HISTORY_TARGET",
                category="invalid_input",
                stage="api",
                message="thread_id 和 user_id 不能为空。",
                retryable=False,
                http_status=400,
                request_id=(
                    request.headers.get("X-Request-ID")
                    or fallback_request_id
                ),
            )
        )
    short_memory = _get_short_memory(request)
    if short_memory is None:
        raise_agent_http_exception(
            build_agent_error(
                code="SHORT_MEMORY_UNAVAILABLE",
                category="unavailable",
                stage="api",
                message="短期记忆服务尚未初始化。",
                retryable=True,
                http_status=503,
                request_id=(
                    request.headers.get("X-Request-ID")
                    or fallback_request_id
                ),
            )
        )
    deleted = await asyncio.to_thread(
        short_memory.delete_thread,
        user_id=clean_user,
        thread_id=clean_thread,
        tenant_id=str(tenant_id).strip() or "default",
    )
    return {
        "ok": True,
        "thread_id": clean_thread,
        "deleted_keys": deleted,
    }
