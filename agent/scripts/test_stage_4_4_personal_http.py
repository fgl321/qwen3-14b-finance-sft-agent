from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import httpx


BASE_URL = "http://127.0.0.1:8000"


def show(title: str, value: object) -> None:
    print(f"\n========== {title} ==========")
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _sufficient(result: dict) -> bool:
    assessment = result.get("evidence_assessment") or {}
    return bool(assessment.get("sufficient"))


async def main() -> None:
    suffix = uuid4().hex
    user_id = f"stage_4_4_user_{suffix}"
    thread_id = f"stage_4_4_thread_{suffix}"
    title = f"Stage 4.4 测试知识_{suffix}"
    knowledge_base_id = "kb_finance_basic"

    async with httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=300.0,
        trust_env=False,
    ) as client:
        health = await client.get("/health/personal-data")
        show("personal_health", health.json())
        health.raise_for_status()
        checks = health.json().get("checks") or {}
        if not all((checks.get(name) or {}).get("ok") for name in (
            "short_memory", "long_memory", "rag"
        )):
            raise AssertionError("Redis/PostgreSQL/Qdrant 个人数据依赖未全部就绪。")

        # 1. 真实聊天写入短期记忆，并触发白名单长期事实抽取。
        first_chat = await client.post(
            "/api/chat/graph-v2",
            json={
                "request_id": f"stage_4_4_chat_1_{suffix}",
                "user_id": user_id,
                "thread_id": thread_id,
                "user_message": "请记住：我的月度必要支出是15000元。",
                "execution_policy": "direct_allowed",
                "enable_rag": False,
                "use_short_memory": True,
                "use_long_memory": True,
                "save_memory": True,
                "extract_long_memory": True,
            },
        )
        show("first_memory_chat", first_chat.json())
        first_chat.raise_for_status()
        first_result = first_chat.json()
        if first_result.get("status") != "completed":
            raise AssertionError("首轮记忆聊天没有完成。")
        if not (first_result.get("personal_memory") or {}).get(
            "short_memory_saved"
        ):
            raise AssertionError("首轮回答没有写入短期记忆。")

        second_chat = await client.post(
            "/api/chat/graph-v2",
            json={
                "request_id": f"stage_4_4_chat_2_{suffix}",
                "user_id": user_id,
                "thread_id": thread_id,
                "user_message": "我刚才说的月度必要支出是多少？",
                "execution_policy": "direct_allowed",
                "enable_rag": False,
                "use_short_memory": True,
                "use_long_memory": True,
                "save_memory": True,
                "extract_long_memory": False,
            },
        )
        show("second_memory_chat", second_chat.json())
        second_chat.raise_for_status()
        second_result = second_chat.json()
        if (second_result.get("personal_memory") or {}).get(
            "short_memory_loaded", 0
        ) < 2:
            raise AssertionError("第二轮没有读取首轮短期记忆。")
        answer = str(second_result.get("final_answer") or "")
        if "15000" not in answer.replace(",", "") and "1.5万" not in answer:
            raise AssertionError(f"第二轮没有正确使用短期记忆：{answer}")

        short_memory = await client.get(
            "/api/personal/short-memory",
            params={"user_id": user_id, "thread_id": thread_id},
        )
        show("short_memory", short_memory.json())
        short_memory.raise_for_status()
        if short_memory.json().get("message_count", 0) < 4:
            raise AssertionError("短期记忆没有保存两轮完整对话。")

        # 2. 长期记忆人工更正、历史与跨线程事实管理。
        facts_after_extract = await client.get(
            "/api/personal/long-memory/facts",
            params={"user_id": user_id},
        )
        show("facts_after_auto_extract", facts_after_extract.json())
        facts_after_extract.raise_for_status()
        extracted_keys = {
            item.get("fact_key")
            for item in facts_after_extract.json().get("facts", [])
        }
        if "monthly_necessary_expense" not in extracted_keys:
            raise AssertionError("明确事实没有被抽取为长期记忆。")

        fact_payload = {
            "user_id": user_id,
            "fact_type": "family_finance",
            "fact_key": "annual_necessary_expense",
            "fact_value": {"amount": 180000, "currency": "CNY"},
            "is_user_confirmed": True,
            "force": True,
        }
        saved = await client.put(
            "/api/personal/long-memory/facts", json=fact_payload
        )
        show("long_memory_saved", saved.json())
        saved.raise_for_status()

        fact_payload["fact_value"] = {"amount": 200000, "currency": "CNY"}
        corrected = await client.put(
            "/api/personal/long-memory/facts", json=fact_payload
        )
        show("long_memory_corrected", corrected.json())
        corrected.raise_for_status()

        history = await client.get(
            "/api/personal/long-memory/facts/"
            "family_finance/annual_necessary_expense/history",
            params={"user_id": user_id},
        )
        show("long_memory_history", history.json())
        history.raise_for_status()
        if history.json().get("count", 0) < 1:
            raise AssertionError("长期记忆更正没有留下历史。")

        # 3. RAG：入库、去重、版本替换、引用、禁用、重启用、重建、删除、拒答。
        v1_payload = {
            "owner_user_id": user_id,
            "knowledge_base_id": knowledge_base_id,
            "title": title,
            "text": (
                "本个人金融知识库规定：紧急备用金通常覆盖三到六个月必要支出。"
                "资金应保持较高流动性。"
            ),
            "version": "1",
        }
        document_v1 = await client.post(
            "/api/personal/rag/documents/text", json=v1_payload
        )
        show("rag_document_v1", document_v1.json())
        document_v1.raise_for_status()
        document_id_v1 = document_v1.json()["document_id"]

        duplicate = await client.post(
            "/api/personal/rag/documents/text", json=v1_payload
        )
        show("rag_duplicate", duplicate.json())
        duplicate.raise_for_status()
        if duplicate.json().get("duplicate") is not True:
            raise AssertionError("相同文档没有正确去重。")

        v2_payload = {
            **v1_payload,
            "text": (
                "本个人金融方案的最新版本规定：收入明显不稳定时，"
                "紧急备用金建议覆盖七个月必要支出。旧版三到六个月规则"
                "不再作为本人的最新方案。"
            ),
            "version": "2",
        }
        document_v2 = await client.post(
            "/api/personal/rag/documents/text", json=v2_payload
        )
        show("rag_document_v2", document_v2.json())
        document_v2.raise_for_status()
        document_id_v2 = document_v2.json()["document_id"]
        if document_id_v1 not in document_v2.json().get(
            "superseded_document_ids", []
        ):
            raise AssertionError("新版本入库后旧版本没有失效。")

        rag_positive = await client.post(
            "/api/personal/rag/query",
            json={
                "owner_user_id": user_id,
                "knowledge_base_id": knowledge_base_id,
                "query": "我的最新个人方案中，收入明显不稳定时备用金覆盖几个月？",
            },
        )
        show("rag_positive_query", rag_positive.json())
        rag_positive.raise_for_status()
        positive_result = rag_positive.json()
        if not _sufficient(positive_result):
            raise AssertionError("精确知识库问题没有被判定为证据充分。")
        citations = positive_result.get("citations") or []
        if not citations:
            raise AssertionError("证据充分的 RAG 回答没有引用。")
        if document_id_v2 not in {
            str(item.get("document_id")) for item in citations
        }:
            raise AssertionError("RAG 引用没有指向最新文档版本。")

        # 主聊天接口也必须真正进入 RAG 快速路径，而不是只提供管理查询。
        rag_chat = await client.post(
            "/api/chat/graph-v2",
            json={
                "request_id": f"stage_4_4_rag_chat_{suffix}",
                "user_id": user_id,
                "thread_id": thread_id,
                "knowledge_base_id": knowledge_base_id,
                "user_message": (
                    "必须根据我的知识库回答：收入明显不稳定时，"
                    "最新方案要求备用金覆盖几个月？"
                ),
                "rag_mode": "required",
                "enable_rag": True,
                "extract_long_memory": False,
            },
        )
        show("rag_chat_required", rag_chat.json())
        rag_chat.raise_for_status()
        rag_chat_result = rag_chat.json()
        if rag_chat_result.get("execution_path") != "rag_direct":
            raise AssertionError("主聊天接口没有进入 RAG 快速路径。")
        if rag_chat_result.get("finish_reason") != "rag_direct_answer":
            raise AssertionError("主聊天接口没有返回证据充分的 RAG 回答。")
        rag_chat_citations = (rag_chat_result.get("rag") or {}).get(
            "citations"
        ) or []
        if document_id_v2 not in {
            str(item.get("document_id")) for item in rag_chat_citations
        }:
            raise AssertionError("主聊天接口的 RAG 引用不是最新版本。")

        rag_chat_replay = await client.post(
            "/api/chat/graph-v2",
            json={
                "request_id": f"stage_4_4_rag_chat_{suffix}",
                "user_id": user_id,
                "thread_id": thread_id,
                "knowledge_base_id": knowledge_base_id,
                "user_message": (
                    "必须根据我的知识库回答：收入明显不稳定时，"
                    "最新方案要求备用金覆盖几个月？"
                ),
                "rag_mode": "required",
                "enable_rag": True,
                "extract_long_memory": False,
            },
        )
        show("rag_chat_replay", rag_chat_replay.json())
        rag_chat_replay.raise_for_status()
        if rag_chat_replay.json().get("idempotency_replayed") is not True:
            raise AssertionError("RAG 主聊天重复请求没有命中幂等重放。")

        disabled = await client.patch(
            f"/api/personal/rag/documents/{document_id_v2}/enabled",
            json={
                "owner_user_id": user_id,
                "knowledge_base_id": knowledge_base_id,
                "enabled": False,
            },
        )
        show("rag_document_disabled", disabled.json())
        disabled.raise_for_status()
        if disabled.json().get("status") != "disabled":
            raise AssertionError("文档禁用失败。")

        rag_disabled = await client.post(
            "/api/personal/rag/query",
            json={
                "owner_user_id": user_id,
                "knowledge_base_id": knowledge_base_id,
                "query": "我的最新个人方案中，收入明显不稳定时备用金覆盖几个月？",
            },
        )
        show("rag_query_after_disable", rag_disabled.json())
        rag_disabled.raise_for_status()
        if _sufficient(rag_disabled.json()) or rag_disabled.json().get("citations"):
            raise AssertionError("禁用文档后仍然被作为有效证据引用。")

        enabled = await client.patch(
            f"/api/personal/rag/documents/{document_id_v2}/enabled",
            json={
                "owner_user_id": user_id,
                "knowledge_base_id": knowledge_base_id,
                "enabled": True,
            },
        )
        show("rag_document_enabled", enabled.json())
        enabled.raise_for_status()
        if enabled.json().get("status") != "active":
            raise AssertionError("文档重新启用失败。")

        rebuilt = await client.post(
            f"/api/personal/rag/documents/{document_id_v2}/rebuild",
            params={
                "owner_user_id": user_id,
                "knowledge_base_id": knowledge_base_id,
            },
        )
        show("rag_document_rebuilt", rebuilt.json())
        rebuilt.raise_for_status()
        if rebuilt.json().get("status") != "active":
            raise AssertionError("文档重建失败。")

        deleted = await client.delete(
            f"/api/personal/rag/documents/{document_id_v2}",
            params={
                "owner_user_id": user_id,
                "knowledge_base_id": knowledge_base_id,
            },
        )
        show("rag_document_deleted", deleted.json())
        deleted.raise_for_status()

        rag_deleted = await client.post(
            "/api/personal/rag/query",
            json={
                "owner_user_id": user_id,
                "knowledge_base_id": knowledge_base_id,
                "query": "我的最新个人方案中，收入明显不稳定时备用金覆盖几个月？",
            },
        )
        show("rag_query_after_delete", rag_deleted.json())
        rag_deleted.raise_for_status()
        if _sufficient(rag_deleted.json()) or rag_deleted.json().get("citations"):
            raise AssertionError("删除文档后仍然被检索或引用。")

        # 清理测试数据。
        await client.delete(
            f"/api/personal/rag/documents/{document_id_v1}",
            params={
                "owner_user_id": user_id,
                "knowledge_base_id": knowledge_base_id,
                "hard_delete": True,
            },
        )
        cleared_short = await client.delete(
            "/api/personal/short-memory",
            params={"user_id": user_id, "thread_id": thread_id},
        )
        show("short_memory_cleared", cleared_short.json())
        cleared_short.raise_for_status()

        cleared_long = await client.delete(
            f"/api/personal/long-memory/users/{user_id}"
        )
        show("long_memory_cleared", cleared_long.json())
        cleared_long.raise_for_status()

    print(
        "\nStage 4.4 Lite personal memory/RAG full HTTP correctness "
        "test passed."
    )


if __name__ == "__main__":
    asyncio.run(main())
