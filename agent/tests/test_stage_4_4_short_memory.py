from __future__ import annotations

import json
from typing import Any

from app.memory.short_term_memory import (
    ShortTermMemoryService,
    StateVersionConflictError,
)


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

    def eval(self, script: str, numkeys: int, *args: Any):
        # 最小 Lua CAS 语义：按 state_version 原子比较，旧版 key 视为版本 0。
        key, _expected, new_value, expected_version, _ttl = args[:5]
        current = self.values.get(key)
        current_version = 0
        if current is not None:
            try:
                current_version = int(
                    json.loads(current).get("state_version") or 0
                )
            except Exception:
                current_version = -1
        allowed = current_version == int(expected_version)
        if not allowed:
            return 0
        self.values[key] = new_value
        return 1

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


def test_conversation_state_cas_commit_and_version_conflict() -> None:
    fake = FakeRedis()
    service = ShortTermMemoryService(redis_client=fake)
    state_v0 = {
        "state_version": 0,
        "turn_count": 0,
        "active_task": None,
    }
    state_v1 = {
        "state_version": 1,
        "turn_count": 1,
        "active_task": {"handle": "TASK_1"},
    }
    state_v2 = {
        "state_version": 2,
        "turn_count": 2,
        "active_task": {"handle": "TASK_1"},
    }

    service.set_conversation_state(
        user_id="u1",
        thread_id="t1",
        state=state_v1,
        expected_version=0,
        expected_state=state_v0,
    )
    saved = fake.get(
        service._conversation_state_key(
            user_id="u1",
            thread_id="t1",
            tenant_id="default",
        )
    )
    assert "TASK_1" in saved

    # 并发写入：仍以 v0 为基准应冲突。
    try:
        service.set_conversation_state(
            user_id="u1",
            thread_id="t1",
            state=state_v1,
            expected_version=0,
            expected_state=state_v0,
        )
    except StateVersionConflictError:
        pass
    else:
        raise AssertionError("过期版本提交应触发 StateVersionConflictError")

    # 基于最新 v1 提交成功，version 递增到 2。
    service.set_conversation_state(
        user_id="u1",
        thread_id="t1",
        state=state_v2,
        expected_version=1,
        expected_state=state_v1,
    )
    saved = fake.get(
        service._conversation_state_key(
            user_id="u1",
            thread_id="t1",
            tenant_id="default",
        )
    )
    assert '"state_version": 2' in saved


def test_conversation_state_cas_requires_expected_state() -> None:
    service = ShortTermMemoryService(redis_client=FakeRedis())
    try:
        service.set_conversation_state(
            user_id="u1",
            thread_id="t1",
            state={"state_version": 1},
            expected_version=0,
            expected_state=None,
        )
    except ValueError as exc:
        assert "expected_state" in str(exc)
    else:
        raise AssertionError("CAS 提交必须提供 expected_state")
