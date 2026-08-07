from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Query

from app.memory.long_term_memory import LongTermMemoryService


router = APIRouter(
    prefix="/api/memory",
    tags=["memory"],
)


@router.get("/facts")
def list_memory_facts(
    user_id: str = Query(...),
    tenant_id: str = Query("default"),
    fact_type: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    service = LongTermMemoryService()
    service.init_schema()

    facts = service.list_facts(
        user_id=user_id,
        tenant_id=tenant_id,
        fact_type=fact_type,
        limit=limit,
    )

    return {
        "ok": True,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "fact_type": fact_type,
        "count": len(facts),
        "facts": [
            {
                "id": fact.id,
                "tenant_id": fact.tenant_id,
                "user_id": fact.user_id,
                "fact_type": fact.fact_type,
                "fact_key": fact.fact_key,
                "fact_value": fact.fact_value,
                "confidence": fact.confidence,
                "source_thread_id": fact.source_thread_id,
                "created_at": fact.created_at,
                "updated_at": fact.updated_at,
            }
            for fact in facts
        ],
    }


@router.post("/facts")
def upsert_memory_fact(
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    user_id = str(payload.get("user_id") or "").strip()
    tenant_id = str(payload.get("tenant_id") or "default").strip()
    fact_type = str(payload.get("fact_type") or "").strip()
    fact_key = str(payload.get("fact_key") or "").strip()
    fact_value = payload.get("fact_value")
    confidence = float(payload.get("confidence", 1.0))
    source_thread_id = payload.get("source_thread_id")

    if not user_id:
        raise ValueError("user_id 不能为空")

    if not fact_type:
        raise ValueError("fact_type 不能为空")

    if not fact_key:
        raise ValueError("fact_key 不能为空")

    if not isinstance(fact_value, dict):
        raise ValueError("fact_value 必须是 object")

    service = LongTermMemoryService()
    service.init_schema()

    fact = service.upsert_fact(
        user_id=user_id,
        tenant_id=tenant_id,
        fact_type=fact_type,
        fact_key=fact_key,
        fact_value=fact_value,
        confidence=confidence,
        source_thread_id=source_thread_id,
    )

    return {
        "ok": True,
        "fact": {
            "id": fact.id,
            "tenant_id": fact.tenant_id,
            "user_id": fact.user_id,
            "fact_type": fact.fact_type,
            "fact_key": fact.fact_key,
            "fact_value": fact.fact_value,
            "confidence": fact.confidence,
            "source_thread_id": fact.source_thread_id,
            "created_at": fact.created_at,
            "updated_at": fact.updated_at,
        },
    }


@router.delete("/facts")
def delete_user_memory_facts(
    user_id: str = Query(...),
    tenant_id: str = Query("default"),
) -> dict[str, Any]:
    service = LongTermMemoryService()
    service.init_schema()

    deleted_count = service.delete_user_facts(
        user_id=user_id,
        tenant_id=tenant_id,
    )

    return {
        "ok": True,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "deleted_count": deleted_count,
    }


@router.delete("/facts/{fact_type}/{fact_key}")
def delete_one_memory_fact(
    fact_type: str,
    fact_key: str,
    user_id: str = Query(...),
    tenant_id: str = Query("default"),
) -> dict[str, Any]:
    service = LongTermMemoryService()
    service.init_schema()

    deleted_count = service.delete_fact(
        user_id=user_id,
        tenant_id=tenant_id,
        fact_type=fact_type,
        fact_key=fact_key,
    )

    return {
        "ok": True,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "fact_type": fact_type,
        "fact_key": fact_key,
        "deleted_count": deleted_count,
    }
