from __future__ import annotations

import tempfile
from pathlib import Path

from app.rag.ingestion_service import RagIngestionService
from app.rag.rag_types import ParsedDocument


class FakeParser:
    def parse(self, **kwargs):
        class _Page:
            page_number = 1
            text = "家庭年度必要支出 18 万元。"

        class _Parsed:
            file_name = "family.md"
            source_type = "md"
            pages = [_Page()]
            metadata = {
                "title": "家庭金融规划指南",
                "aliases": ["family.md", "家庭金融规划指南"],
            }

        return _Parsed()


class FakeChunker:
    def __init__(self):
        self.parsed = None

    def chunk(self, parsed):
        self.parsed = parsed
        return []


class FakeStore:
    def __init__(self):
        self.deleted = None

    def delete_document(self, **kwargs):
        self.deleted = kwargs
        return {"deleted_count_estimate": 0}

    def upsert_chunks(self, **kwargs):
        return {
            "ok": True,
            "upserted_count": 0,
            "batch_count": 0,
            "collection_name": "finance_knowledge",
        }

    def count_points(self):
        return 0


class FakeSettings:
    embedding_provider = "fake"


def test_ingest_file_builds_typed_parsed_document() -> None:
    parser = FakeParser()
    chunker = FakeChunker()
    store = FakeStore()

    service = RagIngestionService(
        settings=FakeSettings(),
        embedding_provider=object(),
        store=store,
    )
    service.parser = parser
    service.chunker = chunker

    with tempfile.NamedTemporaryFile(
        suffix=".md",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        handle.write("家庭年度必要支出 18 万元。")
        path = Path(handle.name)

    try:
        result = service.ingest_file(
            file_path=path,
            original_file_name="family.md",
            tenant_id="t",
            owner_user_id="u",
            knowledge_base_id="kb",
            visibility="private",
        )
    finally:
        path.unlink(missing_ok=True)

    assert result["ok"] is True
    assert isinstance(chunker.parsed, ParsedDocument)
    assert chunker.parsed.meta.tenant_id == "t"
    assert chunker.parsed.meta.file_name == "family.md"
    assert chunker.parsed.meta.title == "家庭金融规划指南"
    assert "家庭金融规划指南" in chunker.parsed.meta.aliases
    assert len(chunker.parsed.meta.file_sha256) == 64
    assert chunker.parsed.pages[0].page_number == 1
    assert store.deleted is not None
