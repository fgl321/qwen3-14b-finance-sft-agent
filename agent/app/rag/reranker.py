from __future__ import annotations

import math
from typing import Any, Protocol

from app.core.logging import get_logger
from app.rag.rag_types import RetrievedChunk


logger = get_logger(__name__)


class Reranker(Protocol):
    """检索后重排器协议。"""

    def rerank(
        self,
        *,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        ...


class NoopReranker:
    """不重排，保持候选顺序。"""

    def rerank(
        self,
        *,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        del query
        return list(candidates)


class BgeReranker:
    """
    BGE-Reranker 交叉编码器重排器。

    - 使用 FlagEmbedding 自带的 FlagReranker（与 BGE-M3 同包，无新增依赖）。
    - 模型首次使用时加载；加载失败时由 build_reranker 降级为 NoopReranker。
    - 重排分数归一化为 0~100 展示分，原始分数保留在 metadata 中。
    """

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "",
        use_fp16: bool = True,
        top_k: int = 6,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.device = device or None
        self.use_fp16 = use_fp16
        self.top_k = max(1, int(top_k))
        self.batch_size = max(1, int(batch_size))
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:  # pragma: no cover - 环境依赖
            raise RuntimeError(
                "缺少 FlagEmbedding，无法使用 BGE-Reranker。"
            ) from exc

        kwargs: dict[str, Any] = {
            "model_name_or_path": self.model_name,
            "use_fp16": self.use_fp16,
        }
        if self.device:
            kwargs["device"] = self.device
        self._model = FlagReranker(**kwargs)
        return self._model

    def rerank(
        self,
        *,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        if len(candidates) == 1:
            return list(candidates)

        model = self._load_model()
        pairs = [[query, chunk.text] for chunk in candidates]

        raw_scores = model.compute_score(
            pairs,
            normalize=False,
            batch_size=self.batch_size,
        )

        if isinstance(raw_scores, (int, float)):
            score_list = [float(raw_scores)]
        else:
            score_list = [float(score) for score in raw_scores]

        if len(score_list) != len(candidates):
            logger.warning(
                "rag_rerank_score_count_mismatch",
                score_count=len(score_list),
                candidate_count=len(candidates),
            )
            return list(candidates)

        scored = list(zip(candidates, score_list))
        scored.sort(key=lambda item: item[1], reverse=True)

        top_candidates = [
            self._apply_rerank_score(chunk=chunk, raw_score=score)
            for chunk, score in scored[: self.top_k]
        ]

        logger.info(
            "rag_rerank_finished",
            candidate_count=len(candidates),
            top_k=len(top_candidates),
            model=self.model_name,
        )

        return top_candidates

    @staticmethod
    def _apply_rerank_score(
        *,
        chunk: RetrievedChunk,
        raw_score: float,
    ) -> RetrievedChunk:
        metadata = dict(chunk.metadata or {})
        metadata["rerank_model"] = "bge-reranker-v2-m3"
        metadata["rerank_raw_score"] = round(raw_score, 6)
        try:
            probability = 1.0 / (1.0 + math.exp(-float(raw_score)))
        except (OverflowError, ValueError):
            probability = 0.0 if float(raw_score) < 0 else 1.0
        metadata["rerank_probability"] = round(probability, 6)

        return chunk.model_copy(
            update={
                "score": 0.0,
                "score_display": None,
                "metadata": metadata,
            }
        )

    @staticmethod
    def normalize_scores(
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """把 rerank 原始分数 min-max 归一化为 0~100 展示分。"""
        if not chunks:
            return chunks

        raw_values = [
            float(chunk.metadata.get("rerank_raw_score", chunk.score) or 0.0)
            for chunk in chunks
        ]
        low = min(raw_values)
        high = max(raw_values)

        normalized: list[RetrievedChunk] = []
        for chunk, raw in zip(chunks, raw_values):
            if high > low:
                display = round((raw - low) / (high - low) * 100, 4)
            else:
                display = 100.0
            metadata = dict(chunk.metadata or {})
            metadata["rerank_score"] = round(display, 4)
            normalized.append(
                chunk.model_copy(
                    update={
                        "score": display,
                        "score_display": f"{display:.2f}/100",
                        "metadata": metadata,
                    }
                )
            )
        return normalized


class HttpReranker:
    """
    通过 HTTP 调用远程 GPU rerank 服务（见 embed_server.py）。
    """

    def __init__(
        self,
        *,
        base_url: str,
        top_k: int = 6,
        timeout: float = 60.0,
    ) -> None:
        import httpx

        self.base_url = base_url.rstrip("/")
        self.top_k = max(1, int(top_k))
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            trust_env=False,
        )

    def rerank(
        self,
        *,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        if len(candidates) == 1:
            return list(candidates)

        response = self._client.post(
            f"{self.base_url}/rerank",
            json={
                "query": query,
                "texts": [chunk.text for chunk in candidates],
            },
        )
        response.raise_for_status()
        scores = list(response.json().get("scores") or [])

        if len(scores) != len(candidates):
            logger.warning(
                "rag_http_rerank_score_count_mismatch",
                score_count=len(scores),
                candidate_count=len(candidates),
            )
            return list(candidates)

        scored = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            BgeReranker._apply_rerank_score(
                chunk=chunk,
                raw_score=float(score),
            )
            for chunk, score in scored[: self.top_k]
        ]


def build_reranker(
    settings: Any,
) -> Reranker:
    """根据配置构建重排器；任何失败都降级为 NoopReranker。"""
    provider = str(
        getattr(settings, "rag_rerank_provider", "local")
    ).strip().lower()

    if provider == "http":
        try:
            url = str(
                getattr(settings, "rag_rerank_http_url", "")
            ).strip()
            if not url:
                raise ValueError("rag_rerank_http_url 为空")
            reranker = HttpReranker(
                base_url=url,
                top_k=int(getattr(settings, "rag_rerank_top_k", 6) or 6),
            )
            logger.info(
                "rag_reranker_ready",
                provider="http",
                url=url,
            )
            return reranker
        except Exception as exc:
            logger.warning(
                "rag_http_reranker_disabled_fallback_to_noop",
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )
            return NoopReranker()

    if not bool(getattr(settings, "rag_rerank_enabled", False)):
        return NoopReranker()

    try:
        reranker = BgeReranker(
            model_name=str(
                getattr(settings, "rag_rerank_model", "BAAI/bge-reranker-v2-m3")
            ),
            device=str(getattr(settings, "rag_rerank_device", "") or ""),
            use_fp16=bool(getattr(settings, "rag_rerank_use_fp16", True)),
            top_k=int(getattr(settings, "rag_rerank_top_k", 6) or 6),
            batch_size=int(
                getattr(settings, "rag_rerank_batch_size", 8) or 8
            ),
        )
        # 启动阶段即验证模型可加载；失败立刻降级，避免请求期才报错。
        reranker._load_model()
        logger.info(
            "rag_reranker_ready",
            model=reranker.model_name,
        )
        return reranker
    except Exception as exc:
        logger.warning(
            "rag_reranker_disabled_fallback_to_noop",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        return NoopReranker()
