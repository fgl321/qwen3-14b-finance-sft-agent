from __future__ import annotations

from app.eval.production_eval_runner import (
    EvalTurn,
    ProductionEvalRunner,
    RetrievalMetrics,
)


def _metrics(
    retrieved: list[str],
    expected: list[str],
    citations: list[str] | None = None,
) -> RetrievalMetrics:
    return ProductionEvalRunner._compute_metrics(
        retrieved_chunks=[
            {"file_name": name} for name in retrieved
        ],
        citations=[
            {"file_name": name} for name in (citations or [])
        ],
        expected_file_names=expected,
    )


def test_recall_mrr_ndcg_perfect_first_hit() -> None:
    m = _metrics(
        retrieved=["a.md", "b.md", "c.md"],
        expected=["a.md"],
        citations=["a.md"],
    )
    assert m.recall_at_3 == 1.0
    assert m.recall_at_5 == 1.0
    assert m.mrr == 1.0
    assert m.ndcg_at_5 == 1.0
    assert m.citation_hit is True
    assert m.citation_precision == 1.0


def test_recall_partial_and_mrr_second_rank() -> None:
    m = _metrics(
        retrieved=["x.md", "a.md", "y.md", "b.md"],
        expected=["a.md", "b.md"],
        citations=["x.md"],
    )
    assert m.recall_at_3 == 1 / 2
    assert m.recall_at_5 == 1.0
    assert m.mrr == 0.5
    assert m.citation_hit is False
    assert m.citation_precision == 0.0


def test_no_expected_returns_none() -> None:
    m = _metrics(
        retrieved=["a.md"],
        expected=[],
    )
    assert m is None


def test_judge_refusal() -> None:
    status, _ = ProductionEvalRunner._judge_turn(
        turn=EvalTurn(
            message="q",
            expected_refusal=True,
        ),
        answer="我在当前知识库中没有找到足够依据回答这个问题。",
        finish_reason="rag_evidence_insufficient",
        rag={
            "evidence_assessment": {"sufficient": False},
            "retrieved_count": 0,
            "citations": [],
        },
        metrics=None,
    )
    assert status == "passed"


def test_judge_missing_citation_fails() -> None:
    status, _ = ProductionEvalRunner._judge_turn(
        turn=EvalTurn(
            message="q",
            expected_has_citations=True,
        ),
        answer="没有引用的回答",
        finish_reason="completed",
        rag={
            "evidence_assessment": {"sufficient": True},
            "retrieved_count": 3,
            "citations": [],
        },
        metrics=None,
    )
    assert status == "failed"
