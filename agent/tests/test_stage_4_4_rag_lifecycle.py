from pathlib import Path
from uuid import UUID

from app.rag.document_lifecycle import RagDocumentLifecycleService
from app.rag.document_parser import (
    _merge_title_lines,
    normalize_document_text,
    parse_document,
)


def test_rag_content_hash_is_stable() -> None:
    service = RagDocumentLifecycleService(postgres_dsn="postgresql://unused")
    assert service.content_hash("第一行\n\n第二行") == service.content_hash(
        "第一行\r\n\r\n第二行   "
    )


def test_parent_child_chunking_has_traceable_ids() -> None:
    service = RagDocumentLifecycleService(postgres_dsn="postgresql://unused")
    text = "。".join([f"第{i}条金融知识" for i in range(300)])
    chunks = service.chunk_text(document_id="doc1", text=text)
    parents = [item for item in chunks if item.chunk_type == "parent"]
    children = [item for item in chunks if item.chunk_type == "child"]
    assert parents
    assert children
    parent_ids = {item.chunk_id for item in parents}
    assert all(item.parent_chunk_id in parent_ids for item in children)
    assert all(str(UUID(item.chunk_id)) == item.chunk_id for item in chunks)
    assert all(
        item.parent_chunk_id is None
        or str(UUID(item.parent_chunk_id)) == item.parent_chunk_id
        for item in chunks
    )


def test_text_document_parser(tmp_path: Path) -> None:
    path = tmp_path / "finance.txt"
    path.write_text("紧急备用金\r\n\r\n覆盖必要支出。", encoding="utf-8")
    assert parse_document(path) == "紧急备用金\n\n覆盖必要支出。"


def test_merge_title_lines_joins_edition_continuation() -> None:
    assert _merge_title_lines(["金融知识普及读本", "（第二版）"]) == (
        "金融知识普及读本（第二版）"
    )
    assert _merge_title_lines(
        [
            "中国平安财产保险股份有限公司",
            "平安医疗费用保险（D 款）条款",
        ]
    ) == "平安医疗费用保险（D 款）条款"
    assert _merge_title_lines(["普通段落内容", "不是标题"]) is None
