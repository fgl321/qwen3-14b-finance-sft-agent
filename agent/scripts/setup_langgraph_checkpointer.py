from __future__ import annotations

import asyncio
import sys


def configure_windows_event_loop() -> None:
    """
    Psycopg 异步连接在 Windows 下不能使用默认的
    ProactorEventLoop，必须切换为 SelectorEventLoop。

    必须在 asyncio.run() 创建事件循环之前执行。
    """

    if sys.platform != "win32":
        return

    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )


configure_windows_event_loop()


from langgraph.checkpoint.postgres.aio import (
    AsyncPostgresSaver,
)

from app.core.config import get_settings


async def main() -> None:
    settings = get_settings()

    postgres_dsn = str(
        settings.postgres_dsn
    ).strip()

    if not postgres_dsn:
        raise RuntimeError(
            "settings.postgres_dsn 不能为空。"
        )

    print(
        "正在初始化 LangGraph PostgreSQL "
        "Checkpointer..."
    )

    async with (
        AsyncPostgresSaver.from_conn_string(
            postgres_dsn
        )
    ) as checkpointer:
        await checkpointer.setup()

    print(
        "LangGraph PostgreSQL "
        "checkpoint tables ready."
    )


if __name__ == "__main__":
    asyncio.run(main())
