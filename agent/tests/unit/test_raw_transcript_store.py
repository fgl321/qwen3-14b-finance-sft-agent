from __future__ import annotations

from app.memory.raw_transcript_store import RawTranscriptStore


class FakeCursor:
    def __init__(self, rows=None, returning=None):
        self.rows = rows or []
        self.returning = returning
        self.executed: list[tuple] = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self.returning

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_append_message_and_list_recent(tmp_path, monkeypatch) -> None:
    store = RawTranscriptStore(
        postgres_dsn="postgresql://fake:fake@127.0.0.1:1/fake"
    )
    cursor = FakeCursor(returning={"id": 7})
    connection = FakeConnection(cursor)
    monkeypatch.setattr(store, "_connect", lambda: connection)

    message = store.append_message(
        tenant_id="personal",
        user_id="owner",
        thread_id="t1",
        role="user",
        content="我的现金90万",
        request_id="req_1",
        run_id="run_1",
    )
    assert message["message_id"] == "7"
    assert connection.committed is True
    assert "conversation_messages" in cursor.executed[0][0]

    read_cursor = FakeCursor(
        rows=[
            {
                "role": "user",
                "content": "我的现金90万",
                "id": 7,
                "created_at": "2026-08-16T00:00:00+00:00",
            }
        ]
    )
    read_connection = FakeConnection(read_cursor)
    monkeypatch.setattr(store, "_connect", lambda: read_connection)
    recent = store.list_recent(
        tenant_id="personal",
        user_id="owner",
        thread_id="t1",
        limit=10,
    )
    assert len(recent) == 1
    assert recent[0]["content"] == "我的现金90万"
    assert recent[0]["message_id"] == "7"


def test_append_turn_writes_two_messages(tmp_path, monkeypatch) -> None:
    store = RawTranscriptStore(
        postgres_dsn="postgresql://fake:fake@127.0.0.1:1/fake"
    )
    cursor = FakeCursor(returning={"id": 8})
    connection = FakeConnection(cursor)
    monkeypatch.setattr(store, "_connect", lambda: connection)

    count = store.append_turn(
        tenant_id="personal",
        user_id="owner",
        thread_id="t1",
        user_message="你好",
        assistant_message="你好！",
    )
    assert count == 2
    assert len(cursor.executed) == 2
