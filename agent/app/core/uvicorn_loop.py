from __future__ import annotations

import asyncio


def selector_loop_factory(
) -> asyncio.AbstractEventLoop:
    """
    为 Windows 下的 Psycopg 异步连接创建
    SelectorEventLoop。

    Uvicorn 调用本函数一次后，
    必须直接得到事件循环实例。
    """

    return asyncio.SelectorEventLoop()
