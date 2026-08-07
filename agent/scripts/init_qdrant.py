from __future__ import annotations

import os

# 必须放在 qdrant_client 导入之前。
# 目的：避免 Windows 系统代理把 127.0.0.1:6333 转发到代理，导致 502 Bad Gateway。
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
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from app.core.config import get_settings


def create_payload_indexes(
    client: QdrantClient,
    collection_name: str,
) -> None:
    index_fields = {
        "tenant_id": PayloadSchemaType.KEYWORD,
        "owner_user_id": PayloadSchemaType.KEYWORD,
        "knowledge_base_id": PayloadSchemaType.KEYWORD,
        "document_id": PayloadSchemaType.KEYWORD,
        "chunk_id": PayloadSchemaType.KEYWORD,
        "parent_id": PayloadSchemaType.KEYWORD,
        "chunk_type": PayloadSchemaType.KEYWORD,
        "visibility": PayloadSchemaType.KEYWORD,
        "file_sha256": PayloadSchemaType.KEYWORD,
        "page_start": PayloadSchemaType.INTEGER,
        "page_end": PayloadSchemaType.INTEGER,
    }

    for field_name, field_type in index_fields.items():
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_type,
            )
            print(f"payload index created: {field_name}")
        except Exception as exc:
            message = str(exc).lower()
            if "already exists" in message or "exists" in message:
                print(f"payload index already exists: {field_name}")
            else:
                raise


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

    if client.collection_exists(collection_name):
        print(f"collection already exists: {collection_name}")
        create_payload_indexes(client, collection_name)

        info = client.get_collection(collection_name)
        print("collection status:", info.status)
        print("vectors:", info.config.params.vectors)
        print("sparse vectors:", info.config.params.sparse_vectors)
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            settings.rag_dense_vector_name: VectorParams(
                size=settings.rag_dense_vector_size,
                distance=Distance.COSINE,
                on_disk=True,
            )
        },
        sparse_vectors_config={
            settings.rag_sparse_vector_name: SparseVectorParams(
                index=SparseIndexParams(
                    on_disk=True,
                )
            )
        },
    )

    print(f"collection created: {collection_name}")

    create_payload_indexes(client, collection_name)

    info = client.get_collection(collection_name)
    print("collection status:", info.status)
    print("vectors:", info.config.params.vectors)
    print("sparse vectors:", info.config.params.sparse_vectors)


if __name__ == "__main__":
    main()
