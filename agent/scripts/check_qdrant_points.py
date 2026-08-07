from __future__ import annotations

import json
from collections import Counter

from app.rag.qdrant_store import QdrantRagStore


def main() -> None:
    store = QdrantRagStore()

    total = store.count_points()
    points = store.scroll_points(limit=50)

    counter = Counter(
        point["payload"].get("chunk_type")
        for point in points
    )

    print("\n========== Qdrant 点数量 ==========")
    print("total_points:", total)

    print("\n========== 当前滚动样本统计 ==========")
    print("parent_count:", counter.get("parent", 0))
    print("child_count:", counter.get("child", 0))

    print("\n========== 点预览 ==========")
    for point in points:
        payload = point["payload"]

        preview = {
            "id": point["id"],
            "chunk_id": payload.get("chunk_id"),
            "parent_id": payload.get("parent_id"),
            "chunk_type": payload.get("chunk_type"),
            "tenant_id": payload.get("tenant_id"),
            "owner_user_id": payload.get("owner_user_id"),
            "knowledge_base_id": payload.get("knowledge_base_id"),
            "file_name": payload.get("file_name"),
            "page_start": payload.get("page_start"),
            "page_end": payload.get("page_end"),
            "text_preview": (payload.get("text") or "")[:120],
        }

        print("-" * 80)
        print(json.dumps(preview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
