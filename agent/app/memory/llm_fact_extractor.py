from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.memory.long_term_memory import LongTermMemoryService


_MEMORY_TOOL = "submit_memory_changes"


def _fact_value_non_empty(value: dict[str, Any]) -> bool:
    """递归判断事实值是否包含真实内容，避免存下空对象 {}。"""
    for item in value.values():
        if item is None or item == "" or item == []:
            continue
        if isinstance(item, dict):
            if _fact_value_non_empty(item):
                return True
            continue
        return True
    return False


_SYSTEM_PROMPT = """
你是长期记忆抽取器，只处理用户明确表达、适合跨会话保留的稳定事实。

规则：
1. 只抽取用户当前消息中明确出现的事实，不推测、不计算、不补全。
2. 临时问题、情绪、闲聊、系统提示、工具错误、模型回答不得保存。
3. 身份证、银行卡、密码、验证码、API Key、Token、完整住址、医疗诊断原文不得保存。
4. 用户明确说“更正、改为、现在是、不是……而是……”时，action=upsert，is_user_confirmed=true。
5. 用户明确要求忘记某个白名单事实时，action=delete。
6. 没有需要变更的长期事实时，也必须调用工具并返回 changes=[]。
7. fact_type/fact_key 必须来自工具描述中的白名单。
8. fact_value 必须包含实际值对象，例如 {"value": 25} 或
   {"value": "金融行业相关工作"}；不允许返回空对象 {}。
""".strip()


@dataclass(slots=True)
class MemoryChange:
    action: str
    fact_type: str
    fact_key: str
    fact_value: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.95
    is_user_confirmed: bool = False
    change_reason: str = "llm_extracted"


class LLMFactExtractor:
    def __init__(
        self,
        *,
        llm_client: Any,
        memory_service: LongTermMemoryService,
        max_completion_tokens: int = 900,
    ) -> None:
        self.llm_client = llm_client
        self.memory_service = memory_service
        self.max_completion_tokens = max_completion_tokens

    def _tool_definition(self) -> dict[str, Any]:
        allowed = {
            fact_type: sorted(keys)
            for fact_type, keys in self.memory_service.fact_whitelist.items()
        }
        return {
            "type": "function",
            "function": {
                "name": _MEMORY_TOOL,
                "description": (
                    "提交长期记忆变更。白名单："
                    + json.dumps(allowed, ensure_ascii=False)
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "changes": {
                            "type": "array",
                            "maxItems": 20,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "action": {
                                        "type": "string",
                                        "enum": ["upsert", "delete"],
                                    },
                                    "fact_type": {"type": "string"},
                                    "fact_key": {"type": "string"},
                                    "fact_value": {
                                        "type": "object",
                                        "description": (
                                            "实际值对象，必须非空，"
                                            '例如 {"value": 25}。'
                                        ),
                                    },
                                    "confidence": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1,
                                    },
                                    "is_user_confirmed": {"type": "boolean"},
                                    "change_reason": {"type": "string"},
                                },
                                "required": [
                                    "action",
                                    "fact_type",
                                    "fact_key",
                                    "fact_value",
                                    "confidence",
                                    "is_user_confirmed",
                                    "change_reason",
                                ],
                            },
                        }
                    },
                    "required": ["changes"],
                },
            },
        }

    @staticmethod
    def _parse_arguments(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        raise ValueError("长期记忆抽取器返回了非法 arguments。")

    async def extract(self, *, user_message: str) -> list[MemoryChange]:
        clean = str(user_message).strip()
        if not clean:
            return []
        response = await self.llm_client.chat(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": clean},
            ],
            tools=[self._tool_definition()],
            thinking_enabled=False,
            max_completion_tokens=self.max_completion_tokens,
        )
        message = response.get("message") or {}
        calls = message.get("tool_calls") or []
        for call in calls:
            function = call.get("function") or {}
            if function.get("name") != _MEMORY_TOOL:
                continue
            arguments = self._parse_arguments(function.get("arguments"))
            changes: list[MemoryChange] = []
            for item in arguments.get("changes") or []:
                if not isinstance(item, dict):
                    continue
                try:
                    action = str(item.get("action") or "").strip()
                    fact_type = str(item.get("fact_type") or "").strip()
                    fact_key = str(item.get("fact_key") or "").strip()
                    self.memory_service.validate_fact_key(
                        fact_type=fact_type,
                        fact_key=fact_key,
                    )
                    if action not in {"upsert", "delete"}:
                        continue
                    raw_value = (
                        item.get("fact_value")
                        if isinstance(item.get("fact_value"), dict)
                        else {}
                    )
                    if action == "upsert" and not _fact_value_non_empty(
                        raw_value
                    ):
                        continue
                    changes.append(
                        MemoryChange(
                            action=action,
                            fact_type=fact_type,
                            fact_key=fact_key,
                            fact_value=raw_value,
                            confidence=float(item.get("confidence", 0.95)),
                            is_user_confirmed=bool(
                                item.get("is_user_confirmed", False)
                            ),
                            change_reason=str(
                                item.get("change_reason") or "llm_extracted"
                            )[:200],
                        )
                    )
                except (TypeError, ValueError):
                    continue
            return changes
        return []

    async def extract_and_apply(
        self,
        *,
        user_message: str,
        user_id: str,
        tenant_id: str = "default",
        source_thread_id: str | None = None,
        source_message_id: str | None = None,
    ) -> dict[str, Any]:
        changes = await self.extract(user_message=user_message)
        saved: list[dict[str, Any]] = []
        deleted: list[dict[str, str]] = []
        for change in changes:
            if change.action == "delete":
                ok = self.memory_service.delete_fact(
                    user_id=user_id,
                    tenant_id=tenant_id,
                    fact_type=change.fact_type,
                    fact_key=change.fact_key,
                    change_reason=change.change_reason,
                )
                if ok:
                    deleted.append(
                        {
                            "fact_type": change.fact_type,
                            "fact_key": change.fact_key,
                        }
                    )
                continue
            fact = self.memory_service.upsert_fact(
                user_id=user_id,
                tenant_id=tenant_id,
                fact_type=change.fact_type,
                fact_key=change.fact_key,
                fact_value=change.fact_value,
                confidence=change.confidence,
                source_thread_id=source_thread_id,
                source_message_id=source_message_id,
                is_user_confirmed=change.is_user_confirmed,
                change_reason=change.change_reason,
            )
            saved.append(fact.to_dict())
        return {"saved": saved, "deleted": deleted, "change_count": len(changes)}
