from __future__ import annotations

import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.llm.qwen_client import QwenClient


ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="Qwen3-14B Finance Agent Preview")


class PreviewRequest(BaseModel):
    user_message: str = Field(min_length=1, max_length=12_000)
    user_id: str = "preview-user"
    thread_id: str = "preview-thread"


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(ROOT / "app/static/index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "mode": "qwen-preview"}


@app.get("/health/qwen")
async def qwen_health() -> dict[str, str]:
    client = QwenClient(get_settings())
    try:
        result = await client.chat(
            [{"role": "user", "content": "只回答 OK"}],
            temperature=0,
            max_completion_tokens=16,
        )
        return {"status": "ok", "model": result["model"]}
    finally:
        await client.close()


@app.post("/api/chat/graph-v2")
async def preview_chat(payload: PreviewRequest) -> dict:
    client = QwenClient(get_settings())
    try:
        result = await client.chat(
            messages=[
                {"role": "system", "content": (
                    "你是中文个人金融助手。回答要准确、清晰、审慎；需要计算时写明关键步骤；"
                    "不承诺收益，不编造用户信息，并提示必要风险。"
                )},
                {"role": "user", "content": payload.user_message},
            ],
            thinking_enabled=False,
            temperature=0.2,
            max_completion_tokens=2048,
        )
        return {
            "status": "completed",
            "mode": "qwen-preview",
            "final_answer": result["message"].get("content") or "",
            "model": result["model"],
            "usage": result["usage"],
            "notice": "预览模式未启用 DeepSeek 规划、RAG、工具和持久化记忆。",
        }
    finally:
        await client.close()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(app, host=settings.app_host, port=settings.app_port)
