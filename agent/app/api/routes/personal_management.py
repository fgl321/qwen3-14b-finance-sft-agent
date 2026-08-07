from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from app.memory.long_term_memory import LongTermMemoryService
from app.memory.short_term_memory import ShortTermMemoryService
from app.personal_data.models import PERSONAL_DATA_VERSION
from app.personal_data.privacy import redact_sensitive_text
from app.rag.document_lifecycle import RagDocumentLifecycleService


router = APIRouter(tags=["personal-data-management"])


class LongFactUpsertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str = "default"
    user_id: str = Field(min_length=1, max_length=200)
    fact_type: str = Field(min_length=1, max_length=100)
    fact_key: str = Field(min_length=1, max_length=100)
    fact_value: dict[str, Any]
    confidence: float = Field(default=1.0, ge=0, le=1)
    source_thread_id: str | None = Field(default=None, max_length=200)
    source_message_id: str | None = Field(default=None, max_length=200)
    is_user_confirmed: bool = True
    change_reason: str = Field(default="user_updated", max_length=200)
    metadata: dict[str, Any] = Field(default_factory=dict)
    force: bool = False


class RagTextIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str = "default"
    owner_user_id: str = Field(min_length=1, max_length=200)
    knowledge_base_id: str = Field(default="kb_finance_basic", max_length=200)
    title: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=10, max_length=2_000_000)
    source: str = Field(default="personal_upload", max_length=500)
    version: str = Field(default="1", max_length=100)
    effective_date: str | None = None
    expired_date: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    replace_same_title: bool = True


class RagFilePathIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str = "default"
    owner_user_id: str = Field(min_length=1, max_length=200)
    knowledge_base_id: str = Field(default="kb_finance_basic", max_length=200)
    file_path: str = Field(min_length=1, max_length=2_000)
    title: str | None = Field(default=None, max_length=500)
    source: str = Field(default="personal_upload", max_length=500)
    version: str = Field(default="1", max_length=100)
    effective_date: str | None = None
    expired_date: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    replace_same_title: bool = True


class RagStatusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str = "default"
    owner_user_id: str = Field(min_length=1, max_length=200)
    knowledge_base_id: str = Field(default="kb_finance_basic", max_length=200)
    enabled: bool


class RagQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str = "default"
    owner_user_id: str = Field(min_length=1, max_length=200)
    knowledge_base_id: str = Field(default="kb_finance_basic", max_length=200)
    query: str = Field(min_length=1, max_length=5_000)


def _safe_error(exc: Exception) -> str:
    return redact_sensitive_text(
        f"{type(exc).__name__}: {str(exc)[:200]}"
    )


def _settings(request: Request) -> Any:
    settings = getattr(request.app.state, "settings", None)
    if settings is not None:
        return settings
    try:
        from app.core.config import get_settings

        return get_settings()
    except Exception as exc:
        raise RuntimeError("无法加载项目配置。") from exc


def _short_memory(request: Request) -> ShortTermMemoryService:
    service = getattr(request.app.state, "short_memory", None)
    if service is None:
        service = ShortTermMemoryService(settings=_settings(request))
        request.app.state.short_memory = service
    return service


def _long_memory(request: Request) -> LongTermMemoryService:
    service = getattr(request.app.state, "personal_long_memory", None)
    if service is None:
        service = LongTermMemoryService(settings=_settings(request))
        service.init_schema()
        request.app.state.personal_long_memory = service
    return service


def _rag_lifecycle(request: Request) -> RagDocumentLifecycleService:
    service = getattr(request.app.state, "rag_document_lifecycle", None)
    if service is None:
        service = RagDocumentLifecycleService(
            settings=_settings(request),
            rag_store=getattr(request.app.state, "rag_store", None),
            embedding_provider=getattr(
                request.app.state, "embedding_provider", None
            ),
        )
        service.init_schema()
        request.app.state.rag_document_lifecycle = service
    return service


async def _thread_call(function: Any, /, *args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(function, *args, **kwargs)


def _serialize(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


@router.get("/health/personal-data")
async def personal_data_health(request: Request) -> dict[str, Any]:
    short = _short_memory(request)
    checks: dict[str, Any] = {
        "short_memory": {
            "ok": await _thread_call(short.ping),
            "max_messages": short.max_messages,
            "ttl_seconds": short.ttl_seconds,
        }
    }
    try:
        long = _long_memory(request)
        await _thread_call(long.init_schema)
        checks["long_memory"] = {"ok": True, "backend": "postgresql"}
    except Exception as exc:
        checks["long_memory"] = {
            "ok": False,
            "error": _safe_error(exc),
        }
    try:
        lifecycle = _rag_lifecycle(request)
        await _thread_call(lifecycle.init_schema)
        client = await _thread_call(lifecycle._get_qdrant_client)
        await _thread_call(client.get_collections)
        checks["rag"] = {
            "ok": True,
            "metadata_backend": "postgresql",
            "vector_backend": "qdrant",
        }
    except Exception as exc:
        checks["rag"] = {
            "ok": False,
            "error": _safe_error(exc),
        }
    return {
        "status": "ok" if all(v.get("ok") for v in checks.values()) else "degraded",
        "version": PERSONAL_DATA_VERSION,
        "checks": checks,
    }


@router.get("/api/personal/short-memory")
async def get_short_memory(
    request: Request,
    user_id: str,
    thread_id: str,
    tenant_id: str = "default",
) -> dict[str, Any]:
    service = _short_memory(request)
    entries = await _thread_call(
        service.list_entries,
        user_id=user_id,
        thread_id=thread_id,
        tenant_id=tenant_id,
    )
    summary = await _thread_call(
        service.get_summary,
        user_id=user_id,
        thread_id=thread_id,
        tenant_id=tenant_id,
    )
    ttl = await _thread_call(
        service.ttl,
        user_id=user_id,
        thread_id=thread_id,
        tenant_id=tenant_id,
    )
    return {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "thread_id": thread_id,
        "message_count": len(entries),
        "messages": entries,
        "summary": summary,
        "ttl_seconds_remaining": ttl,
        "max_messages": service.max_messages,
    }


@router.delete("/api/personal/short-memory")
async def clear_short_memory(
    request: Request,
    user_id: str,
    thread_id: str,
    tenant_id: str = "default",
) -> dict[str, Any]:
    deleted = await _thread_call(
        _short_memory(request).clear_thread,
        user_id=user_id,
        thread_id=thread_id,
        tenant_id=tenant_id,
    )
    return {"cleared": True, "deleted_key_count": deleted}


@router.get("/api/personal/long-memory/facts")
async def list_long_facts(
    request: Request,
    user_id: str,
    tenant_id: str = "default",
    fact_type: str | None = None,
) -> dict[str, Any]:
    facts = await _thread_call(
        _long_memory(request).list_facts,
        user_id=user_id,
        tenant_id=tenant_id,
        fact_type=fact_type,
    )
    return {"count": len(facts), "facts": _serialize(facts)}


@router.put("/api/personal/long-memory/facts")
async def upsert_long_fact(
    payload: LongFactUpsertRequest,
    request: Request,
) -> dict[str, Any]:
    try:
        fact = await _thread_call(
            _long_memory(request).upsert_fact,
            **payload.model_dump(),
        )
        return {"saved": True, "fact": _serialize(fact)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/api/personal/long-memory/facts/{fact_type}/{fact_key}/history")
async def get_long_fact_history(
    fact_type: str,
    fact_key: str,
    request: Request,
    user_id: str,
    tenant_id: str = "default",
) -> dict[str, Any]:
    history = await _thread_call(
        _long_memory(request).list_fact_history,
        user_id=user_id,
        tenant_id=tenant_id,
        fact_type=fact_type,
        fact_key=fact_key,
    )
    return {"count": len(history), "history": _serialize(history)}


@router.delete("/api/personal/long-memory/facts/{fact_type}/{fact_key}")
async def delete_long_fact(
    fact_type: str,
    fact_key: str,
    request: Request,
    user_id: str,
    tenant_id: str = "default",
    hard_delete: bool = False,
) -> dict[str, Any]:
    deleted = await _thread_call(
        _long_memory(request).delete_fact,
        user_id=user_id,
        tenant_id=tenant_id,
        fact_type=fact_type,
        fact_key=fact_key,
        hard_delete=hard_delete,
    )
    return {"deleted": deleted}


@router.delete("/api/personal/long-memory/users/{user_id}")
async def clear_long_memory(
    user_id: str,
    request: Request,
    tenant_id: str = "default",
    hard_delete: bool = True,
) -> dict[str, Any]:
    count = await _thread_call(
        _long_memory(request).delete_user_facts,
        user_id=user_id,
        tenant_id=tenant_id,
        hard_delete=hard_delete,
    )
    return {"cleared": True, "deleted_count": count}


@router.post("/api/personal/rag/documents/text")
async def ingest_rag_text(
    payload: RagTextIngestRequest,
    request: Request,
) -> dict[str, Any]:
    try:
        result = await _thread_call(
            _rag_lifecycle(request).ingest_text,
            **payload.model_dump(),
        )
        return _serialize(result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail="RAG 文档服务暂时不可用，请检查 PostgreSQL、Qdrant 和 Embedding。",
        ) from exc


@router.post("/api/personal/rag/documents/file-path")
async def ingest_rag_file_path(
    payload: RagFilePathIngestRequest,
    request: Request,
) -> dict[str, Any]:
    path = Path(payload.file_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="本地文件不存在。")
    try:
        result = await _thread_call(
            _rag_lifecycle(request).ingest_file,
            path=path,
            title=payload.title,
            tenant_id=payload.tenant_id,
            owner_user_id=payload.owner_user_id,
            knowledge_base_id=payload.knowledge_base_id,
            source=payload.source,
            version=payload.version,
            effective_date=payload.effective_date,
            expired_date=payload.expired_date,
            metadata=payload.metadata,
            replace_same_title=payload.replace_same_title,
        )
        return _serialize(result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail="RAG 文档服务暂时不可用，请检查 PostgreSQL、Qdrant 和 Embedding。",
        ) from exc


@router.get("/api/personal/rag/documents")
async def list_rag_documents(
    request: Request,
    owner_user_id: str,
    tenant_id: str = "default",
    knowledge_base_id: str = "kb_finance_basic",
    status: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    documents = await _thread_call(
        _rag_lifecycle(request).list_documents,
        tenant_id=tenant_id,
        owner_user_id=owner_user_id,
        knowledge_base_id=knowledge_base_id,
        status=status,
        limit=limit,
    )
    return {"count": len(documents), "documents": documents}


@router.get("/api/personal/rag/documents/{document_id}")
async def get_rag_document(
    document_id: str,
    request: Request,
    owner_user_id: str,
    tenant_id: str = "default",
    knowledge_base_id: str = "kb_finance_basic",
) -> dict[str, Any]:
    document = await _thread_call(
        _rag_lifecycle(request).get_document,
        document_id=document_id,
        tenant_id=tenant_id,
        owner_user_id=owner_user_id,
        knowledge_base_id=knowledge_base_id,
    )
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在。")
    return document


@router.patch("/api/personal/rag/documents/{document_id}/enabled")
async def set_rag_document_enabled(
    document_id: str,
    payload: RagStatusRequest,
    request: Request,
) -> dict[str, Any]:
    try:
        return await _thread_call(
            _rag_lifecycle(request).set_document_enabled,
            document_id=document_id,
            **payload.model_dump(),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail="RAG 文档服务暂时不可用，请检查 PostgreSQL、Qdrant 和 Embedding。",
        ) from exc


@router.post("/api/personal/rag/documents/{document_id}/rebuild")
async def rebuild_rag_document(
    document_id: str,
    request: Request,
    owner_user_id: str,
    tenant_id: str = "default",
    knowledge_base_id: str = "kb_finance_basic",
) -> dict[str, Any]:
    try:
        return await _thread_call(
            _rag_lifecycle(request).rebuild_document,
            document_id=document_id,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail="RAG 文档服务暂时不可用，请检查 PostgreSQL、Qdrant 和 Embedding。",
        ) from exc


@router.delete("/api/personal/rag/documents/{document_id}")
async def delete_rag_document(
    document_id: str,
    request: Request,
    owner_user_id: str,
    tenant_id: str = "default",
    knowledge_base_id: str = "kb_finance_basic",
    hard_delete: bool = False,
) -> dict[str, Any]:
    return await _thread_call(
        _rag_lifecycle(request).delete_document,
        document_id=document_id,
        tenant_id=tenant_id,
        owner_user_id=owner_user_id,
        knowledge_base_id=knowledge_base_id,
        hard_delete=hard_delete,
    )


async def _call_rag_service(service: Any, payload: RagQueryRequest) -> Any:
    method = getattr(service, "answer", None) or getattr(service, "run", None)
    if method is None:
        raise RuntimeError("RagAnswerService 缺少 answer/run 方法。")
    signature = inspect.signature(method)
    candidates = {
        "query": payload.query,
        "question": payload.query,
        "user_message": payload.query,
        "tenant_id": payload.tenant_id,
        "owner_user_id": payload.owner_user_id,
        "user_id": payload.owner_user_id,
        "knowledge_base_id": payload.knowledge_base_id,
    }
    kwargs = {
        name: candidates[name]
        for name in signature.parameters
        if name in candidates
    }
    result = method(**kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


@router.post("/api/personal/rag/query")
async def query_rag(
    payload: RagQueryRequest,
    request: Request,
) -> dict[str, Any]:
    rag_service = getattr(request.app.state, "rag_service", None)
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG 回答服务尚未初始化。")
    try:
        result = await _call_rag_service(rag_service, payload)
        serialized = _serialize(result)
        if isinstance(serialized, dict):
            return serialized
        return {"result": serialized}
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"RAG 查询失败：{_safe_error(exc)}",
        ) from exc


@router.get("/api/personal/quality/cases")
async def list_quality_cases() -> dict[str, Any]:
    path = Path("data/eval/personal_quality_cases.jsonl")
    if not path.exists():
        return {"count": 0, "cases": []}
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return {"count": len(cases), "cases": cases}


@router.get("/personal-console", response_class=HTMLResponse)
async def personal_console() -> str:
    return _CONSOLE_HTML


_CONSOLE_HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>个人金融 Agent 管理台</title>
<style>
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;max-width:1100px;margin:30px auto;padding:0 18px;background:#f6f7f9;color:#172033}
h1{margin-bottom:6px}.muted{color:#667085}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;margin-top:20px}
.card{background:white;border:1px solid #e4e7ec;border-radius:14px;padding:18px;box-shadow:0 2px 10px rgba(16,24,40,.04)}
label{display:block;margin:10px 0 4px;font-size:13px;color:#475467}input,textarea,select{width:100%;box-sizing:border-box;padding:10px;border:1px solid #d0d5dd;border-radius:8px}textarea{min-height:120px}
button{margin-top:12px;padding:10px 14px;border:0;border-radius:8px;background:#175cd3;color:#fff;cursor:pointer}.danger{background:#b42318}.secondary{background:#344054}
pre{white-space:pre-wrap;word-break:break-word;background:#101828;color:#d1e9ff;padding:12px;border-radius:9px;min-height:80px;max-height:420px;overflow:auto}
</style></head>
<body>
<h1>个人金融 Agent 管理台</h1>
<div class="muted">管理短期记忆、长期事实和 RAG 文档。接口文档仍可在 <a href="/docs">/docs</a> 查看。</div>
<div class="grid">
<section class="card"><h2>服务检查</h2><button onclick="health()">检查</button><pre id="health"></pre></section>
<section class="card"><h2>短期记忆</h2><label>用户 ID</label><input id="su" value="personal_user"/><label>线程 ID</label><input id="st" value="personal_thread"/><button onclick="loadShort()">查看</button> <button class="danger" onclick="clearShort()">清空</button><pre id="short"></pre></section>
<section class="card"><h2>长期事实</h2><label>用户 ID</label><input id="lu" value="personal_user"/><label>事实类型</label><select id="lt"><option>family_finance</option><option>insurance</option><option>family_profile</option><option>preference</option><option>goal</option></select><label>事实键</label><input id="lk" value="annual_necessary_expense"/><label>JSON 值</label><textarea id="lv">{"amount":180000,"currency":"CNY"}</textarea><button onclick="saveFact()">保存/更正</button> <button class="secondary" onclick="listFacts()">查看全部</button><pre id="long"></pre></section>
<section class="card"><h2>RAG 文档</h2><label>所有者用户 ID</label><input id="ru" value="personal_user"/><label>标题</label><input id="rt" value="我的金融知识"/><label>文档正文</label><textarea id="rx">紧急备用金通常用于覆盖失业、疾病或意外支出。个人应结合收入稳定性与家庭责任确定储备月数。</textarea><button onclick="ingest()">入库</button> <button class="secondary" onclick="listDocs()">查看文档</button><pre id="rag"></pre></section>
</div>
<script>
const out=(id,v)=>document.getElementById(id).textContent=JSON.stringify(v,null,2);
async function req(url,opt){const r=await fetch(url,opt);let j;try{j=await r.json()}catch{j={text:await r.text()}}if(!r.ok)throw j;return j}
async function health(){try{out('health',await req('/health/personal-data'))}catch(e){out('health',e)}}
async function loadShort(){try{out('short',await req(`/api/personal/short-memory?user_id=${encodeURIComponent(su.value)}&thread_id=${encodeURIComponent(st.value)}`))}catch(e){out('short',e)}}
async function clearShort(){try{out('short',await req(`/api/personal/short-memory?user_id=${encodeURIComponent(su.value)}&thread_id=${encodeURIComponent(st.value)}`,{method:'DELETE'}))}catch(e){out('short',e)}}
async function listFacts(){try{out('long',await req(`/api/personal/long-memory/facts?user_id=${encodeURIComponent(lu.value)}`))}catch(e){out('long',e)}}
async function saveFact(){try{out('long',await req('/api/personal/long-memory/facts',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({user_id:lu.value,fact_type:lt.value,fact_key:lk.value,fact_value:JSON.parse(lv.value),is_user_confirmed:true,force:true})}))}catch(e){out('long',e)}}
async function ingest(){try{out('rag',await req('/api/personal/rag/documents/text',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({owner_user_id:ru.value,title:rt.value,text:rx.value})}))}catch(e){out('rag',e)}}
async function listDocs(){try{out('rag',await req(`/api/personal/rag/documents?owner_user_id=${encodeURIComponent(ru.value)}`))}catch(e){out('rag',e)}}
health();
</script></body></html>
"""
