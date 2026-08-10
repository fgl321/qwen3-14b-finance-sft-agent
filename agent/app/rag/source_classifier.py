from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from app.core.json_utils import extract_json_object


@dataclass(frozen=True)
class SourceMetadata:
    """知识源治理元数据：决定某个来源能否作为权威证据。"""

    source_type: str = "user_upload"
    content_type: str = "unclassified"
    scope: str = "thread"
    trust_level: str = "unverified"
    generated_content: bool = False
    allow_rag_direct: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# 仅作为“线索”提供给 LLM 分类器，不是判定依据。
# 判定完全由 DeepSeek 按语义完成。
_STRUCTURAL_HINTS = (
    '"request_id"',
    '"run_id"',
    '"final_answer"',
    '"final_response_result"',
    '"agent_loop_result"',
    '"output_guard"',
    '"synthesis"',
    '"rag_direct_answer"',
    '"planner_finish"',
    '"submit_output_guard_result"',
)


_CLASSIFY_SYSTEM_PROMPT = (
    "你是文档来源治理分类器。你的任务是判断一份用户上传的文档，"
    "它属于什么性质、是否适合作为权威金融知识证据。\n"
    "只输出 JSON，不要输出其他内容，格式：\n"
    '{"content_type": "financial_knowledge|agent_debug_log|'
    'general_document|code|personal|other", '
    '"trust_level": "verified|unverified", '
    '"generated_content": true|false, '
    '"allow_rag_direct": true|false, "reason": "简要理由"}\n'
    "判定规则：\n"
    "1. 如果文档是 AI/Agent 的运行日志、原始响应、生成内容（例如包含 "
    "request_id、final_answer、synthesis、guard 等运行字段，或明显是"
    "上一次模型回答的转储），则 generated_content=true、"
    "allow_rag_direct=false、trust_level=unverified；\n"
    "2. 如果文档是正式金融知识、教材、官方/内部资料，则 "
    "content_type=financial_knowledge、trust_level=verified、"
    "allow_rag_direct=true；\n"
    "3. 其余一般用户文档（通用资料、个人文件、代码等）按内容分类，"
    "trust_level=unverified，allow_rag_direct=true；\n"
    "4. 无法判断时选择最接近的类别，不要编造。"
)


def _default_metadata() -> SourceMetadata:
    return SourceMetadata()


class SourceClassifier:
    """基于 DeepSeek 语义判断的文档来源分类器（同步调用，供入库线程使用）。"""

    def __init__(self, settings: Any | None = None) -> None:
        self.settings = settings

    def classify(
        self,
        *,
        text: str,
        file_name: str = "",
    ) -> SourceMetadata:
        try:
            payload = self._llm_classify_sync(text, file_name)
            return self._payload_to_metadata(payload)
        except Exception:
            # LLM 分类失败时保守兜底：标记为未验证，
            # 检索侧证据门槛仍会要求 LLM 判断充分性。
            return _default_metadata()

    def _llm_classify_sync(self, text: str, file_name: str) -> dict[str, Any]:
        import httpx

        if self.settings is None:
            from app.core.config import get_settings

            self.settings = get_settings()
        api_key = str(
            getattr(self.settings, "deepseek_api_key", "") or ""
        ).strip()
        if not api_key:
            raise RuntimeError("DeepSeek API key 未配置。")
        base_url = str(
            getattr(self.settings, "deepseek_base_url", "")
            or "https://api.deepseek.com"
        ).rstrip("/")
        model = str(
            getattr(self.settings, "deepseek_model", "")
            or "deepseek-chat"
        )
        hints = [
            marker
            for marker in _STRUCTURAL_HINTS
            if marker in str(text)[:16000]
        ]
        hint_text = (
            f"（提示：内容疑似包含 Agent 运行字段：{hints}）"
            if hints
            else ""
        )
        user_content = (
            f"文件名：{file_name or '未知'}\n"
            f"文档开头内容：\n{str(text)[:4000]}\n"
            f"{hint_text}"
        )
        with httpx.Client(timeout=20.0) as client:
            response = client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": _CLASSIFY_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": user_content,
                        },
                    ],
                    "temperature": 0.0,
                    "max_tokens": 256,
                },
            )
            response.raise_for_status()
            data = response.json()
        content = (
            (data.get("choices") or [{}])[0]
            .get("message", {})
            .get("content")
            or "{}"
        )
        return extract_json_object(str(content))

    @staticmethod
    def _payload_to_metadata(payload: dict[str, Any]) -> SourceMetadata:
        content_type = str(payload.get("content_type") or "other")
        if content_type not in {
            "financial_knowledge",
            "agent_debug_log",
            "general_document",
            "code",
            "personal",
            "other",
        }:
            content_type = "other"
        trust = str(payload.get("trust_level") or "unverified")
        trust = trust if trust in {"verified", "unverified"} else "unverified"
        generated = bool(payload.get("generated_content"))
        allow_direct = bool(payload.get("allow_rag_direct"))
        if generated:
            allow_direct = False
            trust = "unverified"
            content_type = "agent_debug_log"
        return SourceMetadata(
            source_type="user_upload",
            content_type=content_type,
            scope="thread",
            trust_level=trust,
            generated_content=generated,
            allow_rag_direct=allow_direct,
        )
