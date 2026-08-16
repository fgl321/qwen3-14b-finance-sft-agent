from __future__ import annotations

import contextvars
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any


EventSink = Callable[[dict[str, Any]], None]

_event_sink: contextvars.ContextVar[EventSink | None] = contextvars.ContextVar(
    "finance_agent_event_sink",
    default=None,
)


def set_event_sink(sink: EventSink) -> contextvars.Token:
    return _event_sink.set(sink)


def reset_event_sink(token: contextvars.Token) -> None:
    _event_sink.reset(token)


def publish_event(
    event: str,
    *,
    request_id: str = "",
    node: str | None = None,
    status: str = "running",
    detail: dict[str, Any] | None = None,
) -> None:
    sink = _event_sink.get()
    if sink is None:
        return
    sink(
        {
            "event": event,
            "request_id": request_id,
            "node": node,
            "status": status,
            "detail": detail or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
