from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .model_loader import load_finance_model


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = os.environ.get("QWEN_MODEL", "qwen3-14b-bf16-finance-sft")
BASE_MODEL = os.environ.get("QWEN_BASE_MODEL", "Qwen/Qwen3-14B")
ADAPTER_DIR = Path(
    os.environ.get("QWEN_ADAPTER_DIR", str(ROOT / "model" / "artifacts" / "final_adapter"))
)
API_KEY = os.environ.get("QWEN_SERVER_API_KEY", "").strip()
LOCAL_FILES_ONLY = os.environ.get("QWEN_LOCAL_FILES_ONLY", "false").lower() == "true"

if not API_KEY:
    raise RuntimeError("QWEN_SERVER_API_KEY must be set")

model, tokenizer = load_finance_model(
    BASE_MODEL,
    ADAPTER_DIR,
    local_files_only=LOCAL_FILES_ONLY,
)
generation_lock = Lock()
app = FastAPI(title="Qwen3-14B Finance SFT", version="1.0.0")


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[dict[str, Any]]
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=4096)


def require_auth(authorization: str | None) -> None:
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="invalid API key")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "base_model": BASE_MODEL,
        "adapter_dir": str(ADAPTER_DIR),
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.get("/v1/models")
def models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    require_auth(authorization)
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}


@app.post("/v1/chat/completions")
def chat_completions(
    request: ChatRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    require_auth(authorization)
    prompt = tokenizer.apply_chat_template(
        request.messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    device = model.get_input_embeddings().weight.device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    do_sample = request.temperature > 0

    with generation_lock, torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=request.max_tokens,
            do_sample=do_sample,
            temperature=request.temperature if do_sample else None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = generated[0, encoded["input_ids"].shape[1] :]
    content = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    prompt_tokens = int(encoded["input_ids"].shape[1])
    completion_tokens = int(new_ids.shape[0])
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
