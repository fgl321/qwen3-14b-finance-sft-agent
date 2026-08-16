from __future__ import annotations

import pytest

from app.memory.narrative_memory import (
    compress_messages_to_summary,
    narrative_segment_token_estimate,
    select_history_strategy,
)


def test_history_strategy_thresholds() -> None:
    assert select_history_strategy(10_000).level == "none"
    assert select_history_strategy(10_000).should_compress is False

    soft = select_history_strategy(70_000)
    assert soft.level == "light"
    assert soft.should_compress is True

    hard = select_history_strategy(120_000)
    assert hard.level == "narrative"
    assert hard.should_compress is True


class FakeClient:
    def __init__(self, content: str):
        self.content = content

    async def chat(self, **kwargs):
        return {
            "message": {
                "role": "assistant",
                "content": self.content,
            },
            "model": "fake",
            "finish_reason": "stop",
        }


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_compress_messages_to_summary() -> None:
    client = FakeClient("用户澄清：30万元是此前情况，当前现金为90万元。")
    summary = await compress_messages_to_summary(
        llm_client=client,  # type: ignore[arg-type]
        messages=[
            {
                "role": "user",
                "content": "我不是有30万，我之前有30万，现在现金是90万。",
            }
        ],
        level="narrative",
    )
    assert "此前" in summary
    assert "90万" in summary


def test_narrative_segment_token_estimate() -> None:
    segments = [
        {"summary": "用户讨论了任务状态设计。"},
        {"summary": "随后决定加入 Canonical Facts。"},
    ]
    assert narrative_segment_token_estimate(segments) > 0
