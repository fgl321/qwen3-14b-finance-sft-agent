from __future__ import annotations

import os

# 必须放在 qdrant_client 导入之前。
# 目的：确保本地 Qdrant 请求不走系统代理。
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
os.environ["no_proxy"] = "127.0.0.1,localhost,::1"

for proxy_key in [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]:
    os.environ.pop(proxy_key, None)

from qdrant_client import QdrantClient

from app.core.config import get_settings


def main() -> None:
    settings = get_settings()

    print("qdrant url:", settings.qdrant_url)
    print("qdrant collection:", settings.qdrant_collection)

    client = QdrantClient(
        url=settings.qdrant_url,
        timeout=30,
        prefer_grpc=False,
    )

    collections = client.get_collections()
    print("qdrant connected:", collections)

    collection_name = settings.qdrant_collection

    exists = client.collection_exists(collection_name)
    print("collection exists:", exists)

    if not exists:
        return

    info = client.get_collection(collection_name)

    print("status:", info.status)
    print("points_count:", info.points_count)
    print("indexed_vectors_count:", info.indexed_vectors_count)
    print("vectors config:", info.config.params.vectors)
    print("sparse vectors config:", info.config.params.sparse_vectors)


if __name__ == "__main__":
    main()
