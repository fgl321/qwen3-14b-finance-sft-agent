from __future__ import annotations

from app.memory.long_term_memory import LongTermMemoryService
from app.rag.document_lifecycle import RagDocumentLifecycleService


def main() -> None:
    long_memory = LongTermMemoryService()
    long_memory.init_schema()
    lifecycle = RagDocumentLifecycleService()
    lifecycle.init_schema()
    print("long_memory_schema_ok")
    print("rag_document_lifecycle_schema_ok")
    print("Stage 4.4 Lite personal-data schema initialized.")


if __name__ == "__main__":
    main()
