from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.llm.qwen_client import QwenClient


async def main() -> None:
    client = QwenClient(get_settings())
    try:
        result = await client.chat(
            messages=[{"role": "user", "content": "什么是紧急备用金？请用一句话回答。"}],
            max_completion_tokens=128,
            temperature=0,
        )
        print(json.dumps({
            "status": "ok", "model": result["model"],
            "answer": result["message"].get("content"),
            "usage": result["usage"],
        }, ensure_ascii=False))
        structured = await client.chat(
            messages=[{"role": "user", "content": "用一句话解释复利。"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "submit_final_answer",
                    "parameters": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                },
            }],
            max_completion_tokens=256,
            temperature=0,
        )
        tool_call = structured["message"]["tool_calls"][0]
        arguments = json.loads(tool_call["function"]["arguments"])
        assert tool_call["function"]["name"] == "submit_final_answer"
        assert str(arguments.get("answer") or "").strip()
        print(json.dumps({"tool_protocol": "ok", "model": structured["model"]}))
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
