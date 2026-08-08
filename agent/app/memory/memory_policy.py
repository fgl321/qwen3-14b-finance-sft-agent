from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.memory.llm_fact_extractor import LLMFactExtractor
from app.memory.long_term_memory import LongTermMemoryService


@dataclass
class MemoryPolicyResult:
    loaded: bool
    saved_count: int
    saved_facts: list[dict[str, Any]]
    prompt_context: str
    error: str | None

    def model_dump(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "saved_count": self.saved_count,
            "saved_facts": self.saved_facts,
            "prompt_context": self.prompt_context,
            "error": self.error,
        }


class LongTermMemoryPolicy:
    """
    长期记忆策略层。

    LongTermMemoryService 只负责数据库读写。
    LLMFactExtractor 负责通过 LLM 从文本抽取/删除事实。
    LongTermMemoryPolicy 负责把二者组合成业务策略。

    当前策略：
    1. 每轮消息通过 LLM 抽取长期稳定事实。
    2. upsert 写入 PostgreSQL，delete 删除指定事实。
    3. 再读取该用户全部长期记忆，格式化后注入给 Agent。
    4. 出错时不影响主对话，只把错误写入 usage.long_memory.error。
    """

    def __init__(
        self,
        *,
        llm_client: Any,
        extractor: LLMFactExtractor | None = None,
        memory_service: LongTermMemoryService | None = None,
    ) -> None:
        self.memory_service = memory_service or LongTermMemoryService()
        self.extractor = extractor or LLMFactExtractor(
            llm_client=llm_client,
            memory_service=self.memory_service,
        )

    async def process_user_message(
        self,
        *,
        user_message: str,
        user_id: str,
        tenant_id: str,
        thread_id: str,
    ) -> MemoryPolicyResult:
        try:
            self.memory_service.init_schema()

            changes = await self.extractor.extract(
                user_message=user_message,
            )

            saved_facts: list[dict[str, Any]] = []

            for change in changes:
                if change.action == "delete":
                    self.memory_service.delete_fact(
                        user_id=user_id,
                        tenant_id=tenant_id,
                        fact_type=change.fact_type,
                        fact_key=change.fact_key,
                        change_reason=change.change_reason,
                    )
                    continue

                saved_fact = self.memory_service.upsert_fact(
                    user_id=user_id,
                    tenant_id=tenant_id,
                    fact_type=change.fact_type,
                    fact_key=change.fact_key,
                    fact_value=change.fact_value,
                    confidence=change.confidence,
                    source_thread_id=thread_id,
                )

                saved_facts.append(
                    {
                        "fact_type": saved_fact.fact_type,
                        "fact_key": saved_fact.fact_key,
                        "fact_value": saved_fact.fact_value,
                        "confidence": saved_fact.confidence,
                        "source_thread_id": saved_fact.source_thread_id,
                    }
                )

            prompt_context = self.memory_service.format_facts_for_prompt(
                user_id=user_id,
                tenant_id=tenant_id,
                limit=30,
            )

            loaded = bool(prompt_context and prompt_context != "暂无长期记忆。")

            return MemoryPolicyResult(
                loaded=loaded,
                saved_count=len(saved_facts),
                saved_facts=saved_facts,
                prompt_context=prompt_context if loaded else "",
                error=None,
            )

        except Exception as exc:
            return MemoryPolicyResult(
                loaded=False,
                saved_count=0,
                saved_facts=[],
                prompt_context="",
                error=str(exc),
            )
