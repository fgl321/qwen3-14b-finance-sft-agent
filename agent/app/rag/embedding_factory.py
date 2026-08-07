from __future__ import annotations

from threading import Lock
from typing import Any

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.rag.embeddings import (
    BgeM3EmbeddingProvider,
    EmbeddingProvider,
    FakeEmbeddingProvider,
)


logger = get_logger(__name__)


_EMBEDDING_PROVIDER_CACHE: dict[
    tuple[Any, ...],
    EmbeddingProvider,
] = {}

_EMBEDDING_PROVIDER_CACHE_LOCK = Lock()


def _build_cache_key(
    settings: Settings,
) -> tuple[Any, ...]:
    """
    根据 embedding 配置生成缓存 key。

    只要这些关键配置没有变，就复用同一个 embedding provider。
    """
    provider_name = settings.embedding_provider.strip().lower()

    if provider_name == "fake":
        return (
            "fake",
            settings.rag_dense_vector_size,
        )

    if provider_name in {"bge-m3", "bge_m3", "bgem3"}:
        return (
            "bge-m3",
            settings.bge_m3_model_name,
            settings.bge_m3_batch_size,
            settings.bge_m3_max_length,
            settings.bge_m3_device or "auto",
            settings.bge_m3_use_fp16,
        )

    return (
        "unknown",
        settings.embedding_provider,
    )


def clear_embedding_provider_cache() -> None:
    """
    清空 embedding provider 缓存。

    主要用于单元测试。
    业务代码一般不需要调用。
    """
    with _EMBEDDING_PROVIDER_CACHE_LOCK:
        _EMBEDDING_PROVIDER_CACHE.clear()


def _create_embedding_provider(
    settings: Settings,
) -> EmbeddingProvider:
    provider_name = settings.embedding_provider.strip().lower()

    if provider_name == "fake":
        logger.info(
            "embedding_provider_selected",
            provider="fake",
            dense_size=settings.rag_dense_vector_size,
        )

        return FakeEmbeddingProvider(
            dense_size=settings.rag_dense_vector_size,
        )

    if provider_name in {"bge-m3", "bge_m3", "bgem3"}:
        logger.info(
            "embedding_provider_selected",
            provider="bge-m3",
            model_name=settings.bge_m3_model_name,
            batch_size=settings.bge_m3_batch_size,
            max_length=settings.bge_m3_max_length,
            device=settings.bge_m3_device or "auto",
            use_fp16=settings.bge_m3_use_fp16,
        )

        return BgeM3EmbeddingProvider(
            model_name_or_path=settings.bge_m3_model_name,
            use_fp16=settings.bge_m3_use_fp16,
            batch_size=settings.bge_m3_batch_size,
            max_length=settings.bge_m3_max_length,
            device=settings.bge_m3_device,
        )

    raise ValueError(
        "未知 EMBEDDING_PROVIDER："
        f"{settings.embedding_provider}。可选值：fake、bge-m3"
    )


def build_embedding_provider(
    settings: Settings | None = None,
) -> EmbeddingProvider:
    """
    构建 embedding provider。

    BGE-M3 是重模型，不能每次请求都重新加载。
    因此这里做进程内缓存，同一套配置只初始化一次。
    """
    settings = settings or get_settings()

    cache_key = _build_cache_key(
        settings
    )

    with _EMBEDDING_PROVIDER_CACHE_LOCK:
        cached_provider = _EMBEDDING_PROVIDER_CACHE.get(
            cache_key
        )

        if cached_provider is not None:
            return cached_provider

        provider = _create_embedding_provider(
            settings
        )

        _EMBEDDING_PROVIDER_CACHE[
            cache_key
        ] = provider

        return provider
