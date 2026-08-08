from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    项目统一配置。

    规则：
    - Python 代码里用小写字段，例如 settings.http_trust_env。
    - .env 文件里用大写变量，例如 HTTP_TRUST_ENV=false。
    - pydantic-settings 会自动映射。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # =========================
    # App
    # =========================
    app_name: str = "qwen3-14b-bf16-finance-agent"
    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8002
    app_version: str = "0.2.0"
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # =========================
    # HTTP / Proxy
    # =========================
    # False 表示 httpx 不读取系统代理环境变量。
    # 你本地有系统代理，所以这里建议保持 False。
    http_trust_env: bool = False

    # =========================
    # DeepSeek
    # =========================
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"

    deepseek_timeout_seconds: float = 120.0

    deepseek_connect_timeout: float = 15.0
    deepseek_read_timeout: float = 120.0
    deepseek_write_timeout: float = 30.0
    deepseek_pool_timeout: float = 30.0

    deepseek_max_retries: int = 2

    deepseek_http2: bool = True
    deepseek_max_connections: int = 100
    deepseek_max_keepalive_connections: int = 20
    deepseek_keepalive_expiry: float = 30.0

    # Qwen3-14B distilled model: final answer generation only.
    qwen_api_key: str = "local-qwen"
    qwen_base_url: str = "http://127.0.0.1:8001/v1"
    qwen_model: str = "qwen3-14b-bf16-finance-sft"
    qwen_connect_timeout: float = 15.0
    qwen_read_timeout: float = 180.0
    qwen_write_timeout: float = 30.0
    qwen_pool_timeout: float = 30.0
    qwen_max_retries: int = 2

    # 最终回答生成模型：qwen=本地蒸馏模型，deepseek=DeepSeek API。
    # 先用于全链路验证，验证通过后切回 qwen。
    synthesis_llm_provider: str = "qwen"

    # =========================
    # PostgreSQL
    # =========================
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_db: str = "finance_agent"
    postgres_user: str = "finance_agent"
    postgres_password: str = "finance_agent"
    postgres_dsn: str = (
        "postgresql://finance_agent:finance_agent@127.0.0.1:5432/finance_agent"
    )

    # =========================
    # Redis
    # =========================
    redis_url: str = "redis://127.0.0.1:6379/0"

    short_memory_enabled: bool = True
    short_memory_max_messages: int = 12
    short_memory_ttl_seconds: int = 86400

    # =========================
    # Qdrant
    # =========================
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "finance_knowledge"
    qdrant_timeout: float = 60.0

    # =========================
    # RAG Vector
    # =========================
    rag_dense_vector_name: str = "dense"
    rag_sparse_vector_name: str = "sparse"
    rag_dense_vector_size: int = 1024

    # RAG 检索参数
    rag_child_limit: int = 8
    rag_parent_limit: int = 4
    rag_fusion_dense_weight: float = 0.65
    rag_fusion_sparse_weight: float = 0.35
    rag_fusion_rrf_k: int = 60
    # 对外展示分数（0~100）低于该阈值的证据会被过滤，0 表示不过滤。
    rag_min_score: float = 0.0
    # 高置信度快速通道：重排概率（sigmoid 后的 0~1）达到该阈值时，
    # 跳过 LLM 证据评估，直接判定证据充分（省一次模型调用）。
    rag_fast_path_min_score: float = 0.9

    # auto 模式相关性门槛：重排概率低于该值视为证据与问题无关，
    # 不进入知识库直接回答，回落到 Agent 正常回答。
    rag_auto_min_rerank_score: float = 0.5

    # 生成/评估提示词中的证据裁剪，缩短解码注意力长度。
    rag_evidence_max_chunks: int = 3
    rag_evidence_max_chars_per_chunk: int = 1800

    # RAG 直接答案的 Redis 缓存 TTL（秒）。
    rag_answer_cache_ttl_seconds: int = 300

    # 图片/扫描件 OCR：对无文字层的 PDF 页和图片文件做文字识别。
    ocr_enabled: bool = True

    # Rerank
    # 默认开启；模型加载失败时会自动降级为不重排，不影响主链路。
    rag_rerank_enabled: bool = True
    rag_rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rag_rerank_provider: str = "local"
    rag_rerank_http_url: str = ""
    rag_rerank_top_k: int = 6
    rag_rerank_batch_size: int = 8
    rag_rerank_device: str = ""
    rag_rerank_use_fp16: bool = True

    # 多轮查询改写
    rag_query_rewrite_enabled: bool = True
    rag_query_rewrite_max_tokens: int = 256

    # =========================
    # Production Agent Defaults
    # =========================
    production_default_tenant_id: str = "default"
    production_default_kb_id: str = "kb_finance_basic"
    production_default_capabilities: list[str] = ["financial_calculation"]
    production_recursion_limit: int = 30

    # =========================
    # Embedding
    # =========================
    embedding_provider: str = "bge-m3"
    # embedding_provider=http 时使用远程 GPU embedding 服务
    embedding_http_url: str = ""

    bge_m3_model_name: str = "BAAI/bge-m3"
    bge_m3_use_fp16: bool = True
    bge_m3_batch_size: int = 4
    bge_m3_max_length: int = 8192
    bge_m3_device: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
