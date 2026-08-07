from __future__ import annotations

from typing import Any

from app.memory.short_term_memory import ShortTermMemoryService


class FakePipeline:
    def __init__(self, redis: "FakeRedis") -> None:
        self.redis = redis
        self.operations: list[tuple[str, tuple[Any, ...]]] = []

    def rpush(self, *args: Any):
        self.operations.append(("rpush", args))
        return self

    def ltrim(self, *args: Any):
        self.operations.append(("ltrim", args))
        return self

    def expire(self, *args: Any):
        self.operations.append(("expire", args))
        return self

    def execute(self):
        for name, args in self.operations:
            getattr(self.redis, name)(*args)
        return [True] * len(self.operations)


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, list[str]] = {}
        self.values: dict[str, str] = {}
        self.expiries: dict[str, int] = {}

    def ping(self):
        return True

    def pipeline(self, transaction: bool = True):
        return FakePipeline(self)

    def rpush(self, key: str, value: str):
        self.data.setdefault(key, []).append(value)

    def ltrim(self, key: str, start: int, end: int):
        values = self.data.get(key, [])
        self.data[key] = values[start:] if start < 0 else values[start : end + 1]

    def expire(self, key: str, seconds: int):
        self.expiries[key] = seconds

    def lrange(self, key: str, start: int, end: int):
        values = self.data.get(key, [])
        return values[start:] if start < 0 else values[start : end + 1]

    def llen(self, key: str):
        return len(self.data.get(key, []))

    def setex(self, key: str, seconds: int, value: str):
        self.values[key] = value
        self.expiries[key] = seconds

    def get(self, key: str):
        return self.values.get(key)

    def delete(self, *keys: str):
        count = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                count += 1
            if key in self.values:
                del self.values[key]
                count += 1
        return count

    def ttl(self, key: str):
        return self.expiries.get(key, -2)


def test_short_memory_filters_trims_and_clears() -> None:
    fake = FakeRedis()
    service = ShortTermMemoryService(
        redis_client=fake,
        max_messages=2,
        ttl_seconds=600,
    )
    service.append_message(
        user_id="u1", thread_id="t1", role="user", content="第一条"
    )
    service.append_message(
        user_id="u1", thread_id="t1", role="assistant", content="第二条"
    )
    service.append_message(
        user_id="u1", thread_id="t1", role="user", content="第三条"
    )
    messages = service.get_messages(user_id="u1", thread_id="t1")
    assert [item["content"] for item in messages] == ["第二条", "第三条"]
    summary = service.get_summary(user_id="u1", thread_id="t1")
    assert "第一条" in summary
    assert service.ttl(user_id="u1", thread_id="t1") == 600
    assert service.clear_thread(user_id="u1", thread_id="t1") >= 1
    assert service.get_messages(user_id="u1", thread_id="t1") == []


def test_short_memory_rejects_system_messages() -> None:
    service = ShortTermMemoryService(redis_client=FakeRedis())
    try:
        service.append_message(
            user_id="u1", thread_id="t1", role="system", content="secret"
        )
    except ValueError as exc:
        assert "只允许" in str(exc)
    else:
        raise AssertionError("system 消息不应写入短期记忆。")
