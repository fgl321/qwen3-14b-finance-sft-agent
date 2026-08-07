from __future__ import annotations

import pytest

from app.rag.query_rewriter import QueryRewriter


class _FakeLLM:
    def __init__(self, content: str = "改写后的查询", error: Exception | None = None):
        self.content = content
        self.error = error
        self.calls: list[dict] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {
            "model": "fake",
            "message": {"content": self.content},
            "usage": {},
        }


@pytest.mark.asyncio
async def test_disabled_returns_original() -> None:
    rewriter = QueryRewriter(llm_client=_FakeLLM(), enabled=False)
    result = await rewriter.rewrite(query="它是什么", history_messages=[{"role": "user", "content": "前面"}])
    assert result == "它是什么"


@pytest.mark.asyncio
async def test_no_history_returns_original() -> None:
    rewriter = QueryRewriter(llm_client=_FakeLLM())
    result = await rewriter.rewrite(query="它是什么", history_messages=[])
    assert result == "它是什么"


@pytest.mark.asyncio
async def test_failure_falls_back_to_original() -> None:
    rewriter = QueryRewriter(
        llm_client=_FakeLLM(error=RuntimeError("boom"))
    )
    result = await rewriter.rewrite(
        query="它是什么",
        history_messages=[{"role": "user", "content": "前面"}],
    )
    assert result == "它是什么"


@pytest.mark.asyncio
async def test_success_returns_rewritten_query() -> None:
    fake = _FakeLLM(content="我家的月支出是多少")
    rewriter = QueryRewriter(llm_client=fake)
    result = await rewriter.rewrite(
        query="是多少",
        history_messages=[
            {"role": "user", "content": "我家的年支出是18万"},
            {"role": "assistant", "content": "已记录。"},
        ],
    )
    assert result == "我家的月支出是多少"
    assert fake.calls
