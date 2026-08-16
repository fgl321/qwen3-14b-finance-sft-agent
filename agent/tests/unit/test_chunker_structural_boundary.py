from __future__ import annotations

from app.rag.chunker import ChunkingConfig, ParentChildChunker
from app.rag.file_utils import clean_text


def test_parent_chunks_do_not_cross_regulation_boundary() -> None:
    chunker = ParentChildChunker(
        config=ChunkingConfig(
            parent_token_limit=900,
            parent_overlap_tokens=120,
            child_token_limit=280,
            child_overlap_tokens=60,
        )
    )
    segments = [
        (1, "第四十七条 本条例自2013年3月15日起施行。", []),
        (1, "（上一部法规结束）", []),
        (1, "附录五", ["附录五"]),
        (1, "存款保险条例", ["附录五"]),
        (1, "第一条 本条例自2015年5月1日起施行。", ["附录五"]),
    ]
    windows = chunker._merge_segments_to_windows(
        segments=segments,
        token_limit=900,
        overlap_tokens=120,
    )
    joined = ["\n".join(item[1] for item in window) for window in windows]
    assert len(joined) == 2
    for text in joined:
        assert not (
            "2013年3月15日" in text and "存款保险条例" in text
        )
    assert "附录五" in joined[1]


def test_structural_boundary_detection() -> None:
    assert ParentChildChunker._is_structural_boundary(
        "附录五"
    )
    assert ParentChildChunker._is_structural_boundary(
        "第三章 存款保险"
    )
    assert ParentChildChunker._is_structural_boundary(
        "第一条 本条例自2015年5月1日起施行。"
    )
    assert ParentChildChunker._is_structural_boundary(
        "《存款保险条例》"
    )
    assert not ParentChildChunker._is_structural_boundary(
        "本条例自2013年3月15日起施行。"
    )


def test_clean_text_normalizes_ocr_dates() -> None:
    cleaned = clean_text(
        "第二十三条 本条例自2015 年5 月1 日起施行。"
    )
    assert "2015年5月1日" in cleaned
