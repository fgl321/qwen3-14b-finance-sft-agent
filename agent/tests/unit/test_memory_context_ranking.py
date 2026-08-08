from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.api.routes.chat_graph_v2 import (
    _long_memory_context,
    _select_memory_facts,
)


def _fact(fact_type: str, fact_key: str, value: object):
    return SimpleNamespace(
        fact_type=fact_type,
        fact_key=fact_key,
        fact_value={"value": value},
    )


class _StubEmbedder:
    """字符袋 embedding：用于验证排序机制，不承担真实语义。"""

    def __init__(self, alphabet: str):
        self._index = {ch: i for i, ch in enumerate(sorted(set(alphabet)))}

    def _vec(self, text: str) -> list[float]:
        vector = [0.0] * len(self._index)
        for ch in text:
            if ch in self._index:
                vector[self._index[ch]] += 1.0
        return vector

    def embed_query(self, text: str):
        return SimpleNamespace(dense=self._vec(text), sparse=None)

    def embed_documents(self, texts: list[str]):
        return [
            SimpleNamespace(dense=self._vec(text), sparse=None)
            for text in texts
        ]


def _alphabet(*texts: str) -> str:
    return "".join(texts)


@pytest.mark.anyio
async def test_small_fact_set_injects_all_without_embedding():
    facts = [_fact("family_profile", "name", "范广路")]
    called = {"count": 0}

    class NeverCalled:
        def embed_query(self, text):
            called["count"] += 1
            raise AssertionError("不应调用 embedding")

    text = await _long_memory_context(
        facts,
        query="你是谁",
        embedding_provider=NeverCalled(),
    )
    assert called["count"] == 0
    assert "范广路" in text


@pytest.mark.anyio
async def test_large_fact_set_ranks_by_relevance():
    facts = [
        _fact("family_profile", "name", "范广路"),
        _fact("family_profile", "major", "机器人工程"),
        _fact("preference", "hobby", "打羽毛球"),
        _fact("preference", "risk_preference", "稳健"),
        _fact("goal", "long_term_goal", "金融AI"),
        _fact("family_finance", "annual_necessary_expense", 180000),
        _fact("family_profile", "city", "上海"),
        _fact("insurance", "life_insurance", "有"),
        _fact("family_profile", "age", 25),
        _fact("family_profile", "occupation", "金融从业"),
        _fact("goal", "short_term_goal", "学Python"),
        _fact("goal", "education_goal", "考研"),
        _fact("family_profile", "family_status", "单身"),
        _fact("family_finance", "monthly_income", 15000),
    ]
    alphabet = _alphabet(
        "范广路机器人工程打羽毛球稳健金融AI上海有金融从业学Python考研单身",
        "范广路 机器人工程 我的情况",
    )
    embedder = _StubEmbedder(alphabet)

    selected = await _select_memory_facts(
        facts,
        query="范广路 机器人工程 我的情况",
        embedding_provider=embedder,
    )

    assert len(selected) <= 8
    selected_keys = {
        (fact.fact_type, fact.fact_key) for fact in selected
    }
    assert ("family_profile", "name") in selected_keys
    assert ("family_profile", "major") in selected_keys


@pytest.mark.anyio
async def test_embedding_failure_falls_back_to_all():
    facts = [
        _fact("family_profile", f"key_{i}", f"value_{i}")
        for i in range(15)
    ]

    class BrokenEmbedder:
        def embed_query(self, text):
            raise RuntimeError("embedding down")

        def embed_documents(self, texts):
            raise RuntimeError("embedding down")

    selected = await _select_memory_facts(
        facts,
        query="随便问问",
        embedding_provider=BrokenEmbedder(),
    )
    assert len(selected) == 15
