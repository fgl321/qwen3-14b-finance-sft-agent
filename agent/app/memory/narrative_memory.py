from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from app.rag.context_governance import estimate_tokens


CompressionLevel = Literal["none", "light", "narrative", "episode"]


@dataclass(frozen=True, slots=True)
class HistoryCompressionStrategy:
    raw_tokens: int
    soft_budget: int
    hard_budget: int
    level: CompressionLevel
    should_compress: bool


def select_history_strategy(
    raw_tokens: int,
    *,
    soft_budget: int = 60_000,
    hard_budget: int = 100_000,
) -> HistoryCompressionStrategy:
    """Soft/hard budget thresholds for narrative compaction."""

    if raw_tokens < soft_budget:
        return HistoryCompressionStrategy(
            raw_tokens=raw_tokens,
            soft_budget=soft_budget,
            hard_budget=hard_budget,
            level="none",
            should_compress=False,
        )
    if raw_tokens < hard_budget:
        return HistoryCompressionStrategy(
            raw_tokens=raw_tokens,
            soft_budget=soft_budget,
            hard_budget=hard_budget,
            level="light",
            should_compress=True,
        )
    return HistoryCompressionStrategy(
        raw_tokens=raw_tokens,
        soft_budget=soft_budget,
        hard_budget=hard_budget,
        level="narrative",
        should_compress=True,
    )


class NarrativeLLMClient(Protocol):
    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        thinking_enabled: bool = False,
        max_completion_tokens: int = 1200,
    ) -> dict[str, Any]: ...


_COMPRESSION_SYSTEM_PROMPT = """你是多轮金融对话的历史叙事压缩器。
你的任务是把较早的原始对话压缩成自然语言叙事摘要，供未来轮次恢复“我们之前在做什么”的语境。
这不是 JSON 抽取，也不是关键词列表；请输出连贯的中文叙事段落。

必须保留：
- 核心事件、事实、数字、时间与前后因果；
- 用户目标、决定、约束、承诺与未完成事项；
- 否定、纠正、时间关系（“此前…现在…”）必须原样保留，禁止把旧值和新值并列成等价事实；
- 用户态度与重要情绪、人物与关系。

删除：
- 语气词、重复句、同义反复、无意义寒暄、无关枝节。

如果原文包含“30万是以前，现在现金是90万”，必须写成“用户澄清：30万元是此前情况，当前现金为90万元”，
不得写成“用户提到现金30万和90万”。"""


async def compress_messages_to_summary(
    *,
    llm_client: NarrativeLLMClient,
    messages: list[dict[str, Any]],
    level: CompressionLevel,
) -> str:
    """Compress older raw messages into one narrative segment summary."""

    if not messages:
        return ""
    transcript = "\n".join(
        f"{'用户' if str(item.get('role')) == 'user' else '助手'}: "
        f"{str(item.get('content') or '')}"
        for item in messages
    )
    level_hint = {
        "light": (
            "轻压缩：去掉语气词与重复，尽量保持原话语义。"
        ),
        "narrative": (
            "叙事摘要：20k 级历史压缩为 3k~5k token 的自然语言叙事。"
        ),
        "episode": (
            "长期 Episode 摘要：把多个分段的更高层概括写成对话故事。"
        ),
    }.get(level, "轻压缩：去掉语气词与重复，尽量保持原话语义。")
    response = await llm_client.chat(
        messages=[
            {
                "role": "system",
                "content": _COMPRESSION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"{level_hint}\n"
                    "以下是需要压缩的原始对话：\n"
                    f"<transcript>\n{transcript[:40_000]}\n"
                    "</transcript>\n"
                    "请只输出压缩后的中文叙事摘要。"
                ),
            },
        ],
        thinking_enabled=False,
        max_completion_tokens=2000,
    )
    raw = str(
        (response.get("message") or {}).get("content") or ""
    ).strip()
    return raw[:8000]


def narrative_segment_token_estimate(
    segments: list[dict[str, Any]],
) -> int:
    return sum(
        estimate_tokens(str(item.get("summary") or ""))
        for item in segments
    )
