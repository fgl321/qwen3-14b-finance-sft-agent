from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from typing import Any

# 本地 Qdrant/GPU 服务请求不走系统代理。
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

from fastapi import FastAPI
from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)


class RerankRequest(BaseModel):
    query: str = Field(min_length=1)
    texts: list[str] = Field(min_length=1, max_length=256)


_embed_model: Any | None = None
_rerank_model: Any | None = None
_inference_lock = threading.Lock()


def _load_models() -> None:
    global _embed_model, _rerank_model
    model_name = os.environ.get(
        "BGE_M3_MODEL_NAME",
        "/home/yjq/models/bge-m3",
    )
    rerank_model = os.environ.get(
        "RAG_RERANK_MODEL",
        "/home/yjq/models/bge-reranker-v2-m3",
    )
    device = os.environ.get("EMBED_SERVER_DEVICE", "cuda")

    from FlagEmbedding import BGEM3FlagModel, FlagReranker

    _embed_model = BGEM3FlagModel(
        model_name,
        use_fp16=True,
        device=device,
    )
    _rerank_model = FlagReranker(
        rerank_model,
        use_fp16=True,
        device=device,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_models()
    yield


app = FastAPI(title="Finance GPU Embedding Server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "embed_ready": _embed_model is not None,
        "rerank_ready": _rerank_model is not None,
        "device": os.environ.get("EMBED_SERVER_DEVICE", "cuda"),
    }


@app.post("/v1/embed")
async def embed(request: EmbedRequest) -> dict[str, Any]:
    if _embed_model is None:
        raise RuntimeError("embedding 模型未就绪")
    with _inference_lock:
        output = _embed_model.encode(
            request.texts,
            batch_size=64,
            max_length=1024,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
    dense_vecs = output.get("dense_vecs")
    lexical_weights = output.get("lexical_weights") or []

    dense_list = (
        dense_vecs.tolist()
        if hasattr(dense_vecs, "tolist")
        else list(dense_vecs)
    )

    embeddings: list[dict[str, Any]] = []
    for index, dense in enumerate(dense_list):
        weights = (
            lexical_weights[index]
            if index < len(lexical_weights)
            else {}
        )
        sparse: dict[str, list[Any]] = {"indices": [], "values": []}
        if isinstance(weights, dict):
            for raw_index, raw_value in weights.items():
                try:
                    sparse["indices"].append(int(raw_index))
                    sparse["values"].append(float(raw_value))
                except (TypeError, ValueError):
                    continue
        embeddings.append(
            {
                "dense": [float(item) for item in dense],
                "sparse": sparse,
            }
        )
    return {"embeddings": embeddings}


@app.post("/v1/rerank")
async def rerank(request: RerankRequest) -> dict[str, Any]:
    if _rerank_model is None:
        raise RuntimeError("rerank 模型未就绪")
    pairs = [[request.query, text] for text in request.texts]
    with _inference_lock:
        scores = _rerank_model.compute_score(
            pairs,
            normalize=False,
            batch_size=16,
        )
    if isinstance(scores, (int, float)):
        score_list = [float(scores)]
    else:
        score_list = [float(score) for score in scores]
    return {"scores": score_list}
