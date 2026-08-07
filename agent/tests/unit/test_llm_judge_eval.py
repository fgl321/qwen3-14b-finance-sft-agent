from __future__ import annotations

import pytest

from app.eval.production_eval_runner import (
    EvalTurn,
    ProductionEvalRunner,
)


class _FakeJudge:
    def __init__(self, content: str, error: Exception | None = None):
        self.content = content
        self.error = error
        self.calls = 0

    async def chat(self, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return {
            "model": "fake-judge",
            "message": {"content": self.content},
            "usage": {},
        }


def _runner(judge=None) -> ProductionEvalRunner:
    return ProductionEvalRunner(
        base_url="http://127.0.0.1:8002",
        judge_llm_client=judge,
    )


def _turn(**overrides) -> EvalTurn:
    values = {"message": "问题", "expectation": "应正确回答"}
    values.update(overrides)
    return EvalTurn(**values)


@pytest.mark.asyncio
async def test_judge_pass() -> None:
    judge = _FakeJudge(
        '{"verdict":"pass","score":9,"reason":"回答正确","issues":[]}'
    )
    status, _ = await _runner(judge)._judge_turn_dispatch(
        turn=_turn(),
        answer="正确的回答",
        finish_reason="completed",
        rag={},
        metrics=None,
    )
    assert status == "passed"
    assert judge.calls == 1


@pytest.mark.asyncio
async def test_judge_fail() -> None:
    judge = _FakeJudge(
        '{"verdict":"fail","score":2,"reason":"编造答案","issues":["无依据"]}'
    )
    status, reason = await _runner(judge)._judge_turn_dispatch(
        turn=_turn(),
        answer="错误的回答",
        finish_reason="completed",
        rag={},
        metrics=None,
    )
    assert status == "failed"
    assert "编造答案" in reason


@pytest.mark.asyncio
async def test_forbidden_keyword_fails_without_judge() -> None:
    judge = _FakeJudge(
        '{"verdict":"pass","score":10,"reason":"ok","issues":[]}'
    )
    status, reason = await _runner(judge)._judge_turn_dispatch(
        turn=_turn(forbidden_keywords=["已被攻击"]),
        answer="回答中包含 已被攻击",
        finish_reason="completed",
        rag={},
        metrics=None,
    )
    assert status == "failed"
    assert "禁止词" in reason
    assert judge.calls == 0


@pytest.mark.asyncio
async def test_judge_error_falls_back_to_deterministic() -> None:
    judge = _FakeJudge(
        content="",
        error=RuntimeError("judge down"),
    )
    status, _ = await _runner(judge)._judge_turn_dispatch(
        turn=_turn(expected_keywords_any=["正确"]),
        answer="正确的回答",
        finish_reason="completed",
        rag={},
        metrics=None,
    )
    assert status == "passed"


@pytest.mark.asyncio
async def test_expected_refusal_marker_shortcut() -> None:
    judge = _FakeJudge(
        '{"verdict":"fail","score":1,"reason":"x","issues":[]}'
    )
    status, _ = await _runner(judge)._judge_turn_dispatch(
        turn=_turn(expected_refusal=True),
        answer="当前信息不足，无法给出确定回答。",
        finish_reason="rag_evidence_insufficient",
        rag={},
        metrics=None,
    )
    assert status == "passed"
    assert judge.calls == 0
