import asyncio

from app.core.config import get_settings
from app.llm.deepseek_client import DeepSeekClient


async def main() -> None:
    settings = get_settings()
    client = DeepSeekClient(settings)

    try:
        result = await client.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个稳健的中文金融规划助手。"
                        "你必须谨慎回答，不知道就明确说不知道。"
                        "涉及金融建议时，必须说明这不是个性化投资建议。"
                    ),
                },
                {
                    "role": "user",
                    "content": "请用一句话解释什么是紧急备用金。",
                },
            ],
            thinking_enabled=False,
            temperature=0.2,
            max_completion_tokens=512,
        )

        print("=" * 60)
        print("模型：", result["model"])
        print("=" * 60)
        print("回答：")
        print(result["message"].get("content"))
        print("=" * 60)
        print("结束原因：", result["finish_reason"])
        print("用量：", result["usage"])
        print("=" * 60)

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
