from __future__ import annotations

import json
import time

from app.core.config import get_settings
from app.rag.embedding_factory import build_embedding_provider


def main() -> None:
    settings = get_settings()

    print("\n========== Embedding 配置 ==========")
    print("embedding_provider:", settings.embedding_provider)
    print("bge_m3_model_name:", settings.bge_m3_model_name)
    print("bge_m3_device:", settings.bge_m3_device or "auto")
    print("bge_m3_use_fp16:", settings.bge_m3_use_fp16)
    print("bge_m3_batch_size:", settings.bge_m3_batch_size)
    print("bge_m3_max_length:", settings.bge_m3_max_length)

    provider = build_embedding_provider(
        settings=settings,
    )

    texts = [
        "紧急备用金通常建议覆盖3到6个月必要支出。",
        "寿险保障缺口需要结合家庭负债和已有寿险保额计算。",
        "新能源汽车电池质保政策包括期限和覆盖范围。",
    ]

    start = time.perf_counter()

    embeddings = provider.embed_documents(texts)

    duration = round(time.perf_counter() - start, 3)

    print("\n========== Embedding 输出 ==========")
    print("duration_seconds:", duration)

    for index, embedding in enumerate(embeddings, start=1):
        print("-" * 80)
        print("text:", texts[index - 1])
        print("dense_size:", len(embedding.dense))
        print("dense_preview:", embedding.dense[:5])
        print("sparse_indices_count:", len(embedding.sparse.indices))
        print("sparse_values_count:", len(embedding.sparse.values))
        print(
            "sparse_preview:",
            list(
                zip(
                    embedding.sparse.indices[:10],
                    embedding.sparse.values[:10],
                )
            ),
        )

        if len(embedding.dense) != settings.rag_dense_vector_size:
            raise AssertionError(
                "dense 向量维度不符合 Qdrant collection："
                f"期望 {settings.rag_dense_vector_size}，"
                f"实际 {len(embedding.dense)}"
            )

    query_embedding = provider.embed_query("怎么计算寿险保障缺口？")

    print("\n========== Query Embedding ==========")
    print("dense_size:", len(query_embedding.dense))
    print("sparse_indices_count:", len(query_embedding.sparse.indices))

    result = {
        "ok": True,
        "embedding_provider": settings.embedding_provider,
        "dense_size": len(query_embedding.dense),
        "sparse_indices_count": len(query_embedding.sparse.indices),
    }

    print("\n========== 测试结果 ==========")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
