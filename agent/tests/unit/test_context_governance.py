from __future__ import annotations

from app.rag.context_governance import (
    build_evidence_context,
    compact_json,
    compact_tool_results,
    content_hash,
    dedupe_messages,
    estimate_tokens,
    select_evidence_citations,
    select_history_messages,
    trim_context_summary,
    trim_text,
)


def test_estimate_tokens_and_trim() -> None:
    assert estimate_tokens("") == 0
    cjk = "中文金融助手测试"
    assert estimate_tokens(cjk) >= 1
    long_text = "字" * 3000
    trimmed = trim_text(long_text, 100)
    assert trimmed.endswith("...[truncated]")
    assert estimate_tokens(trimmed) <= 120


def test_dedupe_messages_keeps_latest_occurrence() -> None:
    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "回复A"},
        {"role": "user", "content": "你好"},
    ]
    unique = dedupe_messages(messages)
    assert [item["content"] for item in unique] == [
        "回复A",
        "你好",
    ]


def test_select_history_keeps_recent_without_keywords() -> None:
    history = [
        {"role": "user", "content": f"历史问题{i}"}
        for i in range(10)
    ]
    selected, stats = select_history_messages(
        history,
        "这是一个自包含的新问题",
    )
    assert stats["continuation"] is None
    assert 1 <= len(selected) <= 8


def test_select_history_dedupes_without_keywords() -> None:
    history = [
        {"role": "user", "content": "继续分析"},
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": "继续分析"},
    ]
    selected, stats = select_history_messages(
        history,
        "继续刚才的分析",
    )
    assert stats["continuation"] is None
    assert stats["deduplicated_current_count"] == 0
    contents = [item["content"] for item in selected]
    assert contents.count("继续分析") == 1


def test_compact_json_preserves_numbers() -> None:
    value = {
        "amount": "180000.00",
        "ok": True,
        "long_text": "x" * 5000,
        "items": [{"id": 1} for _ in range(100)],
    }
    compacted = compact_json(value, max_chars=200)
    assert compacted["amount"] == "180000.00"
    assert compacted["ok"] is True
    assert len(compacted["long_text"]) < 5000
    assert len(compacted["items"]) <= 41


def test_compact_tool_results_budget() -> None:
    results = [
        {
            "tool_call_id": f"call_{i}",
            "tool_name": "tool",
            "success": True,
            "output": {"value": i, "text": "字" * 100},
        }
        for i in range(30)
    ]
    selected, stats = compact_tool_results(results)
    assert stats["selected_count"] <= 24
    assert stats["selected_tokens"] <= 4000
    assert selected[0]["output"]["value"] == 0


def test_select_evidence_citations_strongest_and_dedup() -> None:
    citations = [
        {
            "citation_id": 1,
            "document_id": "doc",
            "file_name": "a.pdf",
            "text": "直接证据A",
            "score": 0.9,
            "metadata": {"support_level": "direct_support"},
        },
        {
            "citation_id": 2,
            "document_id": "doc",
            "file_name": "a.pdf",
            "text": "直接证据B",
            "score": 0.8,
            "metadata": {"support_level": "direct_support"},
        },
        {
            "citation_id": 3,
            "document_id": "doc",
            "file_name": "a.pdf",
            "text": "部分证据",
            "score": 0.7,
            "metadata": {"support_level": "partial_support"},
        },
    ]
    observations = [
        {
            "requirement_id": "T1:1",
            "status": "direct_support",
            "citation_ids": [1, 2],
        },
        {
            "requirement_id": "T1:2",
            "status": "partial_support",
            "citation_ids": [3],
        },
    ]
    selected, stats = select_evidence_citations(
        citations,
        observations,
    )
    assert stats["requirement_observation_count"] == 2
    ids = [item["citation_id"] for item in selected]
    assert 1 in ids and 2 in ids and 3 in ids
    supported = {
        item["citation_id"]: (
            item.get("metadata") or {}
        ).get("supported_requirement_ids")
        for item in selected
    }
    assert supported[1] == ["T1:1"]
    assert supported[3] == ["T1:2"]


def test_build_evidence_context_provisional() -> None:
    rag = {
        "stage_status": {
            "evidence_assessment_status": "protocol_failed",
        },
        "retrieved_chunks": [
            {
                "document_id": "doc",
                "text": "候选内容",
            }
        ],
    }
    text, stats = build_evidence_context(rag)
    assert stats["provisional"] is True
    assert "Provisional Evidence" in text


def test_compact_citation_preserves_evidence_excerpt() -> None:
    from app.rag.context_governance import (
        compact_citation,
    )

    citation = {
        "citation_id": 9,
        "document_id": "doc",
        "file_name": "a.pdf",
        "metadata": {
            "support_level": "direct_support",
            "evidence_excerpt": "第六条 一般医疗费用保险责任……",
        },
    }
    compacted = compact_citation(citation, max_chars=200)
    assert compacted["text"].startswith("第六条")
    assert (
        compacted["metadata"]["evidence_excerpt"]
        == "第六条 一般医疗费用保险责任……"
    )


def test_trim_context_summary() -> None:
    summary = "长" * 10000
    trimmed, stats = trim_context_summary(summary)
    assert stats["truncated"] is True
    assert stats["trimmed_tokens"] <= 3100


def test_content_hash_is_normalized() -> None:
    assert content_hash("你好 世界") == content_hash("你好世界")
