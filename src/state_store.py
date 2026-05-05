from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any


class SqliteStateStore:
    def __init__(self, *, path: str, max_snapshots: int) -> None:
        self._path = Path(path)
        self._max_snapshots = max_snapshots
        self._lock = Lock()
        self._conn: sqlite3.Connection | None = None

    def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS incident_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_kind TEXT NOT NULL,
                    batch_id INTEGER,
                    recorded_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def record_snapshot(
        self,
        *,
        snapshot_kind: str,
        batch_id: int | None,
        recorded_at: str,
        payload: dict[str, Any],
    ) -> None:
        if self._conn is None:
            return
        serialized = json.dumps(payload, default=str)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO incident_snapshots (snapshot_kind, batch_id, recorded_at, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (snapshot_kind, batch_id, recorded_at, serialized),
            )
            self._conn.execute(
                """
                DELETE FROM incident_snapshots
                WHERE id NOT IN (
                    SELECT id FROM incident_snapshots ORDER BY id DESC LIMIT ?
                )
                """,
                (self._max_snapshots,),
            )
            self._conn.commit()

    def list_snapshots(
        self,
        *,
        snapshot_kind: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if self._conn is None:
            return []
        query = """
            SELECT id, snapshot_kind, batch_id, recorded_at, payload_json
            FROM incident_snapshots
        """
        params: list[Any] = []
        if snapshot_kind is not None:
            query += " WHERE snapshot_kind = ?"
            params.append(snapshot_kind)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()

        return [
            {
                "id": row[0],
                "snapshot_kind": row[1],
                "batch_id": row[2],
                "recorded_at": row[3],
                "payload": json.loads(row[4]),
            }
            for row in rows
        ]

    def get_stats(self) -> dict[str, Any]:
        if self._conn is None:
            return {"enabled": False, "path": str(self._path), "snapshots": 0}
        with self._lock:
            count = self._conn.execute("SELECT COUNT(*) FROM incident_snapshots").fetchone()[0]
        return {
            "enabled": True,
            "path": str(self._path),
            "snapshots": count,
            "max_snapshots": self._max_snapshots,
        }
