from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse

from app.agent_graph.production_runtime import (
    open_production_graph_runtime,
)
from app.agent_graph.release_contract import PRODUCTION_RUNTIME_REVISION
from app.api.routes import chat_graph_v2, memory
from app.api.routes.knowledge import router as knowledge_router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.core.request_boundary import personal_request_identity
from app.llm.deepseek_client import DeepSeekClient
from app.llm.qwen_client import QwenClient
from app.llm.synthesis_proxy import SynthesisClientProxy
from app.memory.short_term_memory import (
    ShortTermMemoryService,
)
from app.memory.raw_transcript_store import (
    RawTranscriptStore,
)
from app.rag.embedding_factory import (
    build_embedding_provider,
)
from app.rag.document_lifecycle import (
    RagDocumentLifecycleService,
)
from app.rag.reranker import build_reranker
from app.rag.qdrant_store import QdrantRagStore
from app.rag.rag_service import RagAnswerService
from app.rag.ingestion_jobs import IngestionJobStore


setup_logging()
logger = get_logger(__name__)

# get_settings() 使用 lru_cache，安全在模块级别调用。
settings = get_settings()


def _production_limits():
    from app.agent_graph.runtime.agent_limits import AgentLimits

    return AgentLimits(
        max_agent_rounds=settings.production_max_agent_rounds,
        max_total_tool_calls=settings.production_max_total_tool_calls,
        max_parallel_tool_calls=settings.production_max_parallel_tool_calls,
        max_plan_repairs_per_execution_round=(
            settings.production_max_plan_repairs_per_execution_round
        ),
        max_plan_revisions=settings.production_max_plan_revisions,
        max_output_rewrites=settings.production_max_output_rewrites,
        total_run_timeout_seconds=settings.production_total_timeout_seconds,
    )


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    """
    FastAPI 应用生命周期。

    启动阶段：
    1. 创建共享 DeepSeekClient。
    2. 加载向量模型。
    3. 创建 Qdrant RAG 服务。
    4. 创建 Redis 短期记忆服务。
    5. 创建 PostgreSQL LangGraph Checkpointer。
    6. 编译生产 LangGraph。

    关闭阶段：
    1. 关闭 PostgreSQL Checkpointer。
    2. 关闭 DeepSeekClient。
    """

    app.state.settings = settings
    app.state.request_identity = personal_request_identity(settings)
    app.state.ingestion_jobs = IngestionJobStore()
    app.state.ingestion_tasks = set()

    app.state.deepseek = DeepSeekClient(
        settings
    )
    app.state.qwen = QwenClient(settings)

    # 前端可以在每个请求里切换最终回答模型（qwen=蒸馏模型，deepseek=DeepSeek API），
    # 无需重启；默认值来自 SYNTHESIS_LLM_PROVIDER。
    synthesis_llm_client = SynthesisClientProxy(
        qwen_client=app.state.qwen,
        deepseek_client=app.state.deepseek,
        default_provider=settings.synthesis_llm_provider,
    )

    try:
        logger.info(
            "embedding_provider_loading",
            provider=settings.embedding_provider,
            model_name=settings.bge_m3_model_name,
            device=(
                settings.bge_m3_device
                or "auto"
            ),
        )

        app.state.embedding_provider = (
            build_embedding_provider(
                settings=settings,
            )
        )

        logger.info(
            "embedding_provider_ready",
            provider=settings.embedding_provider,
            model_name=settings.bge_m3_model_name,
        )

        app.state.rag_store = QdrantRagStore(
            settings=settings,
        )

        try:
            lifecycle = RagDocumentLifecycleService(
                settings=settings,
                rag_store=app.state.rag_store,
            )
            lifecycle.init_schema()
            app.state.rag_document_lifecycle = lifecycle
            transcript_store = RawTranscriptStore(
                settings=settings,
            )
            transcript_store.init_schema()
            app.state.raw_transcript_store = transcript_store
            await asyncio.to_thread(
                lifecycle.sync_index_status,
                tenant_id=settings.personal_tenant_id,
                owner_user_id=settings.personal_user_id,
                knowledge_base_id="kb_finance_basic",
            )
        except Exception as exc:
            logger.warning(
                "document_registry_startup_sync_failed",
                error_type=type(exc).__name__,
            )

        app.state.reranker = build_reranker(
            settings=settings,
        )

        app.state.rag_service = RagAnswerService(
            llm_client=app.state.deepseek,
            store=app.state.rag_store,
            embedding_provider=(
                app.state.embedding_provider
            ),
            answer_llm_client=synthesis_llm_client,
            reranker=app.state.reranker,
            settings=settings,
        )

        logger.info(
            "rag_service_ready",
        )

        app.state.short_memory = (
            ShortTermMemoryService(
                settings=settings,
            )
        )

        logger.info(
            "short_memory_ready",
            enabled=(
                settings.short_memory_enabled
            ),
            max_messages=(
                settings.short_memory_max_messages
            ),
            ttl_seconds=(
                settings.short_memory_ttl_seconds
            ),
        )

        logger.info(
            "production_graph_initializing",
            postgres_enabled=True,
        )

        async with open_production_graph_runtime(
            llm_client=app.state.deepseek,
            synthesis_llm_client=synthesis_llm_client,
            postgres_dsn=(
                settings.postgres_dsn
            ),

            # 数据库迁移已经通过
            # setup_langgraph_checkpointer.py
            # 独立完成。
            setup_checkpointer=False,
            limits=_production_limits(),
        ) as production_runtime:
            app.state.production_graph_runtime = (
                production_runtime
            )

            app.state.production_graph_service = (
                production_runtime.service
            )

            logger.info(
                "production_finance_graph_ready",
            )

            logger.info(
                "app_started",
                app_name=settings.app_name,
                app_env=settings.app_env,
                model=settings.deepseek_model,
                embedding_provider=(
                    settings.embedding_provider
                ),
            )

            yield

    finally:
        try:
            await app.state.deepseek.close()

        except Exception as exc:
            logger.exception(
                "deepseek_client_close_failed",
                error_type=type(exc).__name__,
            )

        try:
            await app.state.qwen.close()
        except Exception as exc:
            logger.exception(
                "qwen_client_close_failed",
                error_type=type(exc).__name__,
            )

        logger.info(
            "app_stopped",
        )


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_log_middleware(
    request: Request,
    call_next,
):
    """
    为每个 HTTP 请求生成 request_id，
    并记录请求耗时和执行结果。
    """

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            body_size = int(content_length)
        except ValueError:
            body_size = -1
        if body_size < 0 or body_size > settings.max_http_body_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "error_code": "REQUEST_BODY_TOO_LARGE",
                    "message": "请求体超过服务允许的大小。",
                },
            )

    supplied_request_id = request.headers.get("X-Request-ID", "").strip()
    request_id = (
        supplied_request_id
        if 0 < len(supplied_request_id) <= 128
        and supplied_request_id.replace("-", "").replace("_", "").isalnum()
        else str(uuid.uuid4())
    )

    request.state.request_id = request_id

    start_time = time.perf_counter()

    logger.info(
        "request_started",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )

    try:
        response = await call_next(
            request
        )

    except Exception as exc:
        duration_ms = round(
            (
                time.perf_counter()
                - start_time
            )
            * 1000,
            2,
        )

        logger.exception(
            "request_failed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            duration_ms=duration_ms,
            error_type=type(exc).__name__,
        )

        return JSONResponse(
            status_code=500,
            content={
                "request_id": request_id,
                "error_code": (
                    "INTERNAL_SERVER_ERROR"
                ),
                "message": (
                    "服务内部处理失败，"
                    "请重新提交请求。"
                ),
            },
            headers={
                "X-Request-ID": request_id,
            },
        )

    duration_ms = round(
        (
            time.perf_counter()
            - start_time
        )
        * 1000,
        2,
    )

    logger.info(
        "request_finished",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )

    response.headers[
        "X-Request-ID"
    ] = request_id
    response.headers["X-Agent-Runtime-Revision"] = PRODUCTION_RUNTIME_REVISION

    return response


@app.get("/health")
async def health() -> dict[str, str]:
    """
    基础存活检查。
    """

    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "runtime_revision": PRODUCTION_RUNTIME_REVISION,
    }


async def _check_llm_health(
    client: DeepSeekClient | QwenClient,
    label: str,
) -> dict:
    """公共 LLM 健康检查，避免 deepseek/qwen 两个端点重复。"""
    result = await client.chat(
        messages=[{"role": "user", "content": "只回答 OK"}],
        thinking_enabled=False,
        max_completion_tokens=32,
    )
    return {
        "status": "ok",
        "service": label,
        "model": result["model"],
        "answer": result["message"].get("content"),
        "usage": result["usage"],
    }


@app.get("/health/deepseek")
async def deepseek_health(request: Request) -> dict:
    return await _check_llm_health(request.app.state.deepseek, "deepseek")


@app.get("/health/qwen")
async def qwen_health(request: Request) -> dict:
    return await _check_llm_health(request.app.state.qwen, "qwen")


@app.get("/health/embedding")
async def embedding_health(
    request: Request,
) -> dict:
    """
    检查向量模型是否可用。
    """

    settings = (
        request.app.state.settings
    )

    provider = (
        request.app.state
        .embedding_provider
    )

    embedding = provider.embed_query(
        "测试向量模型是否可用"
    )

    return {
        "status": "ok",
        "embedding_provider": (
            settings.embedding_provider
        ),
        "dense_size": len(
            embedding.dense
        ),
        "sparse_indices_count": len(
            embedding.sparse.indices
        ),
    }


@app.get("/health/memory")
async def memory_health(
    request: Request,
) -> dict:
    """
    检查 Redis 短期记忆是否可用。
    """

    short_memory: (
        ShortTermMemoryService
    ) = request.app.state.short_memory

    return {
        "status": "ok",
        "memory_type": (
            "redis_short_term_memory"
        ),
        "redis_ping": (
            short_memory.ping()
        ),
    }


@app.get("/health/production-graph")
async def production_graph_health(
    request: Request,
) -> dict:
    """
    检查生产 LangGraph 是否已经初始化。
    """

    service = getattr(
        request.app.state,
        "production_graph_service",
        None,
    )

    runtime = getattr(
        request.app.state,
        "production_graph_runtime",
        None,
    )

    return {
        "status": (
            "ok"
            if service is not None
            else "unavailable"
        ),
        "graph_service_ready": (
            service is not None
        ),
        "graph_runtime_ready": (
            runtime is not None
        ),
        "checkpointer": (
            "postgresql"
            if runtime is not None
            else None
        ),
    }


# 记忆相关接口。
app.include_router(
    memory.router
)

# 知识库相关接口。
app.include_router(
    knowledge_router
)

# 唯一生产聊天接口：
# POST /api/chat/graph-v2
app.include_router(
    chat_graph_v2.router
)


@app.get("/", include_in_schema=False)
async def web_frontend() -> FileResponse:
    frontend_index = (
        Path(__file__).resolve().parents[2]
        / "frontend"
        / "dist"
        / "index.html"
    )
    if frontend_index.is_file():
        return FileResponse(frontend_index)
    return FileResponse(Path(__file__).parent / "static" / "index.html")


# React 构建产物存在时，托管其静态资源；API 路由优先于该挂载。
_FRONTEND_DIST_ASSETS = (
    Path(__file__).resolve().parents[2]
    / "frontend"
    / "dist"
    / "assets"
)
if _FRONTEND_DIST_ASSETS.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=_FRONTEND_DIST_ASSETS),
        name="frontend-assets",
    )
