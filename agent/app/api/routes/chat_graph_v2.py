from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from collections import OrderedDict
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.agent_graph.runtime.agent_errors import (
    build_agent_error,
    exception_to_agent_error,
    log_event,
    raise_agent_http_exception,
)
from app.agent_graph.schemas.planner_schema import ExecutionPolicy
from app.agent_graph.runtime.request_idempotency import (
    RequestIdempotencyConflict,
)
from app.core.logging import get_logger
from app.llm.synthesis_proxy import (
    reset_synthesis_provider,
    set_synthesis_provider,
)
from app.memory.llm_fact_extractor import LLMFactExtractor
from app.memory.long_term_memory import LongTermMemoryService
from app.memory.short_term_memory import ShortTermMemoryService
from app.personal_data.models import PERSONAL_DATA_VERSION
from app.rag.query_rewriter import QueryRewriter


logger = get_logger(__name__)
router = APIRouter(tags=["production-chat-graph"])


class ProductionChatRequest(BaseModel):
    user_message: str = Field(min_length=1, max_length=12_000)
    user_id: str = Field(min_length=1, max_length=200)
    thread_id: str = Field(min_length=1, max_length=200)
    tenant_id: str = Field(default="default", min_length=1, max_length=200)
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
    synthesis_llm_provider: str | None = Field(
        default=None, min_length=1, max_length=20
    )
    # 把检索范围限定到指定文档（“我上传的这个文档”场景）。
    document_ids: list[str] = Field(default_factory=list)



def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rag_request_fingerprint(payload: ProductionChatRequest) -> str:
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
) -> dict[str, Any] | None:
    cache = _rag_attempt_cache(request)
    key = (payload.tenant_id, payload.user_id, request_id)
    item = cache.get(key)
    if item is None:
        return None
    fingerprint = _rag_request_fingerprint(payload)
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
) -> None:
    cache = _rag_attempt_cache(request)
    key = (payload.tenant_id, payload.user_id, request_id)
    cache[key] = {
        "fingerprint": _rag_request_fingerprint(payload),
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


def _rag_sufficient(rag: dict[str, Any]) -> bool:
    assessment = rag.get("evidence_assessment") or {}
    return bool(assessment.get("sufficient"))


def _build_rag_direct_result(
    *,
    payload: ProductionChatRequest,
    request_id: str,
    rag: dict[str, Any],
    run_id: str,
    replayed: bool,
) -> dict[str, Any]:
    sufficient = _rag_sufficient(rag)
    fingerprint = _rag_request_fingerprint(payload)
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
        "execution_path": "rag_direct",
        "rag": rag,
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


async def _run_rag_attempt(
    *,
    payload: ProductionChatRequest,
    request: Request,
    request_id: str,
    history_messages: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    audit: dict[str, Any] = {
        "attempted": False,
        "mode": payload.rag_mode,
        "sufficient": False,
        "replayed": False,
        "degraded": False,
    }
    if not payload.enable_rag or payload.rag_mode == "off":
        return None, audit

    cached = _cached_rag_attempt(
        request, payload=payload, request_id=request_id
    )
    if cached is not None:
        rag = dict(cached["rag"])
        sufficient = _rag_sufficient(rag)
        audit.update(
            {
                "attempted": True,
                "sufficient": sufficient,
                "replayed": True,
                "retrieved_count": int(rag.get("retrieved_count") or 0),
            }
        )
        if sufficient or payload.rag_mode == "required":
            run_id = str(cached.get("run_id") or f"rag-run-{uuid4()}")
            return (
                _build_rag_direct_result(
                    payload=payload,
                    request_id=request_id,
                    rag=rag,
                    run_id=run_id,
                    replayed=True,
                ),
                audit,
            )
        return None, audit

    rag_service = getattr(request.app.state, "rag_service", None)
    if rag_service is None:
        audit.update(
            {
                "attempted": True,
                "degraded": True,
                "error": "RAG_SERVICE_UNAVAILABLE",
            }
        )
        if payload.rag_mode == "required":
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
        return None, audit

    try:
        settings = getattr(request.app.state, "settings", None)
        rewrite_enabled = bool(
            getattr(settings, "rag_query_rewrite_enabled", False)
        )
        retrieval_query = payload.user_message
        if rewrite_enabled:
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

        raw = await rag_service.answer(
            query=payload.user_message,
            retrieval_query=retrieval_query,
            tenant_id=payload.tenant_id,
            owner_user_id=payload.user_id,
            knowledge_base_id=payload.knowledge_base_id,
            document_ids=payload.document_ids,
        )
        rag = _serialize_model(raw)
        if not isinstance(rag, dict):
            raise TypeError("RAG 服务返回值必须可序列化为对象。")
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
        )
        audit.update(
            {
                "attempted": True,
                "sufficient": sufficient,
                "retrieved_count": int(rag.get("retrieved_count") or 0),
                "citation_count": len(rag.get("citations") or []),
            }
        )
        if sufficient or payload.rag_mode == "required":
            return (
                _build_rag_direct_result(
                    payload=payload,
                    request_id=request_id,
                    rag=rag,
                    run_id=str(run_id),
                    replayed=False,
                ),
                audit,
            )
        return None, audit
    except RequestIdempotencyConflict:
        raise
    except Exception as exc:
        audit.update(
            {
                "attempted": True,
                "degraded": True,
                "error": type(exc).__name__,
            }
        )
        if payload.rag_mode == "required":
            error = exception_to_agent_error(
                exc, stage="rag", request_id=request_id
            )
            raise_agent_http_exception(error)
        return None, audit


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


def _long_memory_context(facts: list[Any]) -> str:
    if not facts:
        return ""
    lines = [
        "以下是用户已明确提供并允许长期保存的事实。回答时可使用，"
        "但不得推测未记录内容："
    ]
    for fact in facts[:50]:
        value = getattr(fact, "fact_value", {})
        lines.append(
            f"- {getattr(fact, 'fact_type', '')}."
            f"{getattr(fact, 'fact_key', '')} = "
            f"{json.dumps(value, ensure_ascii=False)}"
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


@router.post("/api/chat/graph-v2")
async def production_chat_graph(
    payload: ProductionChatRequest,
    request: Request,
) -> dict[str, Any]:
    request_id = (
        payload.request_id
        or request.headers.get("X-Request-ID")
        or f"api-prod-{uuid4()}"
    )
    provider = str(payload.synthesis_llm_provider or "").strip().lower()
    provider_token = None
    if provider in {"qwen", "deepseek"}:
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

        final_history = _merge_history(
            short_history,
            payload.history_messages,
            max_messages=(short_memory.max_messages if short_memory else 12),
        )

        long_context = ""
        if long_memory is not None:
            try:
                facts = await asyncio.to_thread(
                    long_memory.list_facts,
                    user_id=payload.user_id,
                    tenant_id=payload.tenant_id,
                )
                memory_audit["long_memory_loaded"] = len(facts)
                long_context = _long_memory_context(facts)
            except Exception as exc:
                memory_audit["degraded"].append(
                    {"stage": "long_memory_read", "error": type(exc).__name__}
                )

        context_parts = [
            part
            for part in [payload.context_summary.strip(), long_context]
            if part
        ]
        final_context_summary = "\n\n".join(context_parts)
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
        rag_direct, rag_audit = await _run_rag_attempt(
            payload=payload,
            request=request,
            request_id=request_id,
            history_messages=final_history,
        )
        memory_audit["rag"] = rag_audit
        if rag_direct is not None:
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
            return rag_direct

        allowed_groups = list(payload.allowed_tool_groups)
        if payload.enable_rag and "knowledge_retrieval" not in allowed_groups:
            allowed_groups.append("knowledge_retrieval")

        result = await service.run(
            request_id=request_id,
            user_message=payload.user_message,
            user_id=payload.user_id,
            thread_id=payload.thread_id,
            tenant_id=payload.tenant_id,
            knowledge_base_id=payload.knowledge_base_id,
            history_messages=final_history,
            context_summary=final_context_summary,
            route_context=(
                payload.route_context
                or {"complexity": "medium", "risk_level": "low"}
            ),
            allowed_tool_names=payload.allowed_tool_names,
            allowed_tool_groups=allowed_groups,
            remaining_tool_calls=payload.remaining_tool_calls,
            allow_side_effects=payload.allow_side_effects,
            execution_policy=payload.execution_policy,
        )
        write_audit = await _save_memory_after_success(
            payload=payload,
            request=request,
            request_id=request_id,
            result=result,
            short_memory=short_memory,
            long_memory=long_memory,
        )
        result["personal_memory"] = {**memory_audit, **write_audit}
        return result

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
