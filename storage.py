from __future__ import annotations

import sqlite3
import uuid
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
    logical_group_id: str = ""
    logical_event_id: str = ""
    canonical_sender_id: str = ""
    source_bot_id: str = ""


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
            existing_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(group_messages)")
            }
            migrations = {
                "logical_group_id": "TEXT NOT NULL DEFAULT ''",
                "logical_event_id": "TEXT NOT NULL DEFAULT ''",
                "canonical_sender_id": "TEXT NOT NULL DEFAULT ''",
                "source_bot_id": "TEXT NOT NULL DEFAULT ''",
            }
            for column, definition in migrations.items():
                if column not in existing_columns:
                    connection.execute(
                        f"ALTER TABLE group_messages ADD COLUMN {column} {definition}"
                    )
            self._repair_duplicate_event_umos(connection)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_group_messages_umo_ts "
                "ON group_messages(umo, ts)"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_group_messages_umo_message_id "
                "ON group_messages(umo, message_id) WHERE message_id IS NOT NULL"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_group_messages_logical_ts "
                "ON group_messages(logical_group_id, ts)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_group_messages_logical_event "
                "ON group_messages(logical_group_id, logical_event_id)"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "idx_group_messages_event_umo_unique "
                "ON group_messages(logical_event_id, umo) "
                "WHERE logical_event_id != ''"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sender_aliases (
                    logical_group_id TEXT NOT NULL,
                    alias_id TEXT NOT NULL,
                    canonical_sender_id TEXT NOT NULL,
                    display_name TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0,
                    source TEXT NOT NULL DEFAULT '',
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(logical_group_id, alias_id)
                )
                """
            )

    @staticmethod
    def _repair_duplicate_event_umos(connection: sqlite3.Connection) -> None:
        """Split legacy over-merged events before adding the uniqueness guard."""
        event_ids = connection.execute(
            "SELECT logical_event_id FROM group_messages "
            "WHERE logical_event_id != '' "
            "GROUP BY logical_event_id "
            "HAVING COUNT(*) > COUNT(DISTINCT umo)"
        ).fetchall()
        for event_row in event_ids:
            event_id = str(event_row["logical_event_id"] or "")
            rows = connection.execute(
                "SELECT id, umo FROM group_messages "
                "WHERE logical_event_id = ? ORDER BY ts ASC, id ASC",
                (event_id,),
            ).fetchall()
            by_umo: dict[str, list[int]] = {}
            for row in rows:
                by_umo.setdefault(str(row["umo"]), []).append(int(row["id"]))
            max_occurrences = max((len(items) for items in by_umo.values()), default=1)
            for occurrence in range(1, max_occurrences):
                replacement = uuid.uuid4().hex
                for row_ids in by_umo.values():
                    if occurrence < len(row_ids):
                        connection.execute(
                            "UPDATE group_messages SET logical_event_id = ? WHERE id = ?",
                            (replacement, row_ids[occurrence]),
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
            logical_group_id=str(row["logical_group_id"] or ""),
            logical_event_id=str(row["logical_event_id"] or ""),
            canonical_sender_id=str(row["canonical_sender_id"] or ""),
            source_bot_id=str(row["source_bot_id"] or ""),
        )

    @staticmethod
    def _mapped_sender_id(
        connection: sqlite3.Connection,
        logical_group_id: str,
        sender_id: str,
    ) -> str:
        if not logical_group_id or not sender_id:
            return sender_id
        row = connection.execute(
            "SELECT canonical_sender_id FROM sender_aliases "
            "WHERE logical_group_id = ? AND alias_id = ?",
            (logical_group_id, sender_id),
        ).fetchone()
        return str(row["canonical_sender_id"] or sender_id) if row else sender_id

    @staticmethod
    def _save_sender_alias(
        connection: sqlite3.Connection,
        *,
        logical_group_id: str,
        alias_id: str,
        canonical_sender_id: str,
        display_name: str,
        confidence: float,
        source: str,
        updated_at: float,
    ) -> None:
        if not logical_group_id or not alias_id or not canonical_sender_id:
            return
        connection.execute(
            """
            INSERT INTO sender_aliases(
                logical_group_id, alias_id, canonical_sender_id,
                display_name, confidence, source, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(logical_group_id, alias_id) DO UPDATE SET
                canonical_sender_id = excluded.canonical_sender_id,
                display_name = CASE
                    WHEN excluded.display_name != '' THEN excluded.display_name
                    ELSE sender_aliases.display_name
                END,
                confidence = MAX(sender_aliases.confidence, excluded.confidence),
                source = excluded.source,
                updated_at = excluded.updated_at
            """,
            (
                logical_group_id,
                alias_id,
                canonical_sender_id,
                display_name,
                max(0.0, min(1.0, float(confidence))),
                source,
                updated_at,
            ),
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
        logical_group_id: str = "",
        source_bot_id: str = "",
    ) -> int:
        with self._connection() as connection:
            logical_group_id = str(logical_group_id or "").strip()
            canonical_sender_id = self._mapped_sender_id(
                connection,
                logical_group_id,
                sender_id,
            )
            logical_event_id = ""
            if logical_group_id:
                duplicate = connection.execute(
                    """
                    SELECT logical_event_id, canonical_sender_id, sender_id
                    FROM group_messages
                    WHERE logical_group_id = ?
                      AND umo != ?
                      AND sender_name = ?
                      AND content = ?
                      AND ABS(ts - ?) <= 1.5
                      AND logical_event_id != ''
                      AND NOT EXISTS (
                          SELECT 1 FROM group_messages AS current
                          WHERE current.logical_event_id = group_messages.logical_event_id
                            AND current.umo = ?
                      )
                    ORDER BY ABS(ts - ?) ASC, id ASC
                    LIMIT 1
                    """,
                    (
                        logical_group_id,
                        umo,
                        sender_name,
                        content,
                        ts,
                        umo,
                        ts,
                    ),
                ).fetchone()
                if duplicate is not None:
                    logical_event_id = str(duplicate["logical_event_id"] or "")
                    canonical_sender_id = str(
                        duplicate["canonical_sender_id"]
                        or duplicate["sender_id"]
                        or canonical_sender_id
                    )
                    self._save_sender_alias(
                        connection,
                        logical_group_id=logical_group_id,
                        alias_id=sender_id,
                        canonical_sender_id=canonical_sender_id,
                        display_name=sender_name,
                        confidence=0.75,
                        source="cross_account_duplicate",
                        updated_at=ts,
                    )
                    self._save_sender_alias(
                        connection,
                        logical_group_id=logical_group_id,
                        alias_id=str(duplicate["sender_id"] or ""),
                        canonical_sender_id=canonical_sender_id,
                        display_name=sender_name,
                        confidence=0.75,
                        source="cross_account_duplicate",
                        updated_at=ts,
                    )
                else:
                    self._save_sender_alias(
                        connection,
                        logical_group_id=logical_group_id,
                        alias_id=sender_id,
                        canonical_sender_id=canonical_sender_id,
                        display_name=sender_name,
                        confidence=0.5,
                        source="first_observation",
                        updated_at=ts,
                    )
                logical_event_id = logical_event_id or uuid.uuid4().hex
            try:
                cursor = connection.execute(
                    """
                    INSERT INTO group_messages(
                        umo, ts, sender_id, sender_name, content,
                        platform, group_id, message_id, logical_group_id,
                        logical_event_id, canonical_sender_id, source_bot_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        logical_group_id,
                        logical_event_id,
                        canonical_sender_id,
                        source_bot_id,
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

    def query_logical(
        self,
        *,
        logical_group_id: str,
        start_ts: float,
        end_ts: float,
        exclude_row_id: int | None = None,
        limit: int = 200,
    ) -> list[ChatRecord]:
        sql = (
            "SELECT * FROM group_messages "
            "WHERE logical_group_id = ? AND ts >= ? AND ts <= ?"
        )
        params: list[object] = [logical_group_id, start_ts, end_ts]
        if exclude_row_id is not None:
            sql += " AND id != ?"
            params.append(exclude_row_id)
        sql += " ORDER BY ts DESC, id DESC LIMIT ?"
        params.append(max(1, int(limit)) * 3)
        with self._connection() as connection:
            rows = connection.execute(sql, params).fetchall()

        selected: list[sqlite3.Row] = []
        seen_events: set[str] = set()
        for row in rows:
            logical_event_id = str(row["logical_event_id"] or "")
            fingerprint = logical_event_id or f"row:{int(row['id'])}"
            if fingerprint in seen_events:
                continue
            seen_events.add(fingerprint)
            selected.append(row)
            if len(selected) >= max(1, int(limit)):
                break
        selected.reverse()
        return [self._from_row(row) for row in selected]

    def aliases_for_group(self, logical_group_id: str) -> list[dict[str, object]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM sender_aliases WHERE logical_group_id = ? "
                "ORDER BY canonical_sender_id, alias_id",
                (logical_group_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def backfill_logical_group(
        self,
        *,
        logical_group_id: str,
        umos: list[str],
    ) -> int:
        logical_group_id = str(logical_group_id or "").strip()
        umos = [str(item or "").strip() for item in umos if str(item or "").strip()]
        if not logical_group_id or not umos:
            return 0
        placeholders = ",".join("?" for _ in umos)
        with self._connection() as connection:
            cursor = connection.execute(
                f"UPDATE group_messages SET logical_group_id = ? "
                f"WHERE umo IN ({placeholders}) AND logical_group_id = ''",
                (logical_group_id, *umos),
            )
            changed = max(0, int(cursor.rowcount))
            rows = connection.execute(
                f"""
                SELECT id, umo, ts, sender_id, sender_name, content,
                       logical_event_id, canonical_sender_id
                FROM group_messages
                WHERE logical_group_id = ? AND umo IN ({placeholders})
                ORDER BY ts ASC, id ASC
                """,
                (logical_group_id, *umos),
            ).fetchall()
            recent: list[dict[str, object]] = []
            event_umos: dict[str, set[str]] = {}
            for existing in rows:
                existing_event = str(existing["logical_event_id"] or "")
                if existing_event:
                    event_umos.setdefault(existing_event, set()).add(
                        str(existing["umo"])
                    )
            for row in rows:
                sender_id = str(row["sender_id"] or "")
                canonical = self._mapped_sender_id(
                    connection,
                    logical_group_id,
                    sender_id,
                )
                event_id = str(row["logical_event_id"] or "")
                duplicate: dict[str, object] | None = None
                if not event_id:
                    for candidate in reversed(recent):
                        if float(row["ts"]) - float(candidate["ts"]) > 1.5:
                            break
                        if (
                            str(candidate["umo"]) != str(row["umo"])
                            and str(candidate["sender_name"]) == str(row["sender_name"])
                            and str(candidate["content"]) == str(row["content"])
                            and str(row["umo"])
                            not in event_umos.get(
                                str(candidate["logical_event_id"]), set()
                            )
                        ):
                            duplicate = candidate
                            break
                    if duplicate is not None:
                        event_id = str(duplicate["logical_event_id"])
                        canonical = str(duplicate["canonical_sender_id"] or canonical)
                        self._save_sender_alias(
                            connection,
                            logical_group_id=logical_group_id,
                            alias_id=sender_id,
                            canonical_sender_id=canonical,
                            display_name=str(row["sender_name"] or ""),
                            confidence=0.75,
                            source="backfilled_cross_account_duplicate",
                            updated_at=float(row["ts"]),
                        )
                        self._save_sender_alias(
                            connection,
                            logical_group_id=logical_group_id,
                            alias_id=str(duplicate["sender_id"] or ""),
                            canonical_sender_id=canonical,
                            display_name=str(row["sender_name"] or ""),
                            confidence=0.75,
                            source="backfilled_cross_account_duplicate",
                            updated_at=float(row["ts"]),
                        )
                    else:
                        event_id = uuid.uuid4().hex
                        self._save_sender_alias(
                            connection,
                            logical_group_id=logical_group_id,
                            alias_id=sender_id,
                            canonical_sender_id=canonical,
                            display_name=str(row["sender_name"] or ""),
                            confidence=0.5,
                            source="backfilled_observation",
                            updated_at=float(row["ts"]),
                        )
                    connection.execute(
                        "UPDATE group_messages SET logical_event_id = ?, "
                        "canonical_sender_id = ? WHERE id = ?",
                        (event_id, canonical, int(row["id"])),
                    )
                event_umos.setdefault(event_id, set()).add(str(row["umo"]))
                recent.append(
                    {
                        "umo": str(row["umo"]),
                        "ts": float(row["ts"]),
                        "sender_id": sender_id,
                        "sender_name": str(row["sender_name"]),
                        "content": str(row["content"]),
                        "logical_event_id": event_id,
                        "canonical_sender_id": canonical,
                    }
                )
                while recent and float(row["ts"]) - float(recent[0]["ts"]) > 1.5:
                    recent.pop(0)
            return changed

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
