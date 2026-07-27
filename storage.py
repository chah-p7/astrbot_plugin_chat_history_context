from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class ChatRecord:
    row_id: int
    umo: str
    ts: float
    sender_id: str
    sender_name: str
    content: str
    platform: str
    group_id: str
    message_id: str | None


class ChatHistoryStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS group_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    umo TEXT NOT NULL,
                    ts REAL NOT NULL,
                    sender_id TEXT NOT NULL DEFAULT '',
                    sender_name TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL,
                    platform TEXT NOT NULL DEFAULT '',
                    group_id TEXT NOT NULL DEFAULT '',
                    message_id TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_group_messages_umo_ts "
                "ON group_messages(umo, ts)"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_group_messages_umo_message_id "
                "ON group_messages(umo, message_id) WHERE message_id IS NOT NULL"
            )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> ChatRecord:
        return ChatRecord(
            row_id=int(row["id"]),
            umo=str(row["umo"]),
            ts=float(row["ts"]),
            sender_id=str(row["sender_id"]),
            sender_name=str(row["sender_name"]),
            content=str(row["content"]),
            platform=str(row["platform"]),
            group_id=str(row["group_id"]),
            message_id=row["message_id"],
        )

    def append(
        self,
        *,
        umo: str,
        ts: float,
        sender_id: str,
        sender_name: str,
        content: str,
        platform: str,
        group_id: str,
        message_id: str | None,
    ) -> int:
        with self._connection() as connection:
            try:
                cursor = connection.execute(
                    """
                    INSERT INTO group_messages(
                        umo, ts, sender_id, sender_name, content,
                        platform, group_id, message_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        umo,
                        ts,
                        sender_id,
                        sender_name,
                        content,
                        platform,
                        group_id,
                        message_id,
                    ),
                )
                return int(cursor.lastrowid)
            except sqlite3.IntegrityError:
                if message_id is None:
                    raise
                row = connection.execute(
                    "SELECT id FROM group_messages WHERE umo = ? AND message_id = ?",
                    (umo, message_id),
                ).fetchone()
                if row is None:
                    raise
                return int(row["id"])

    def query(
        self,
        *,
        umo: str,
        start_ts: float,
        end_ts: float,
        exclude_row_id: int | None = None,
    ) -> list[ChatRecord]:
        sql = (
            "SELECT * FROM group_messages "
            "WHERE umo = ? AND ts >= ? AND ts <= ?"
        )
        params: list[object] = [umo, start_ts, end_ts]
        if exclude_row_id is not None:
            sql += " AND id != ?"
            params.append(exclude_row_id)
        sql += " ORDER BY ts ASC, id ASC"
        with self._connection() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._from_row(row) for row in rows]

    def count_since(self, *, umo: str, start_ts: float) -> int:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS cnt FROM group_messages WHERE umo = ? AND ts >= ?",
                (umo, start_ts),
            ).fetchone()
        return int(row["cnt"]) if row else 0

    def prune_before(self, cutoff_ts: float) -> int:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM group_messages WHERE ts < ?",
                (cutoff_ts,),
            )
            return max(0, int(cursor.rowcount))
