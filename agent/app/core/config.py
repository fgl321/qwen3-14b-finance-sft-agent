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

    # Qwen3-14B distilled model: final answer generation only.
    qwen_api_key: str = "local-qwen"
    qwen_base_url: str = "http://127.0.0.1:8001/v1"
    qwen_model: str = "qwen3-14b-bf16-finance-sft"
    qwen_connect_timeout: float = 15.0
    qwen_read_timeout: float = 180.0
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

    # =========================
    # RAG Vector
    # =========================
    rag_dense_vector_name: str = "dense"
    rag_sparse_vector_name: str = "sparse"
    rag_dense_vector_size: int = 1024

    # RAG 检索参数
    rag_child_limit: int = 8
    rag_parent_limit: int = 4
    # 对外展示分数（0~100）低于该阈值的证据会被过滤，0 表示不过滤。
    rag_min_score: float = 0.0

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
