"""
Thread-sicheres SQLite-Backend (WAL-Modus) + atomare JSON-Helfer.

Ersetzt verstreute, race-condition-anfällige JSON-Schreibvorgänge durch eine
zentrale lokale Datenbank (`assistant.db`) und bietet gleichzeitig Migration &
Rückwärtskompatibilität für `local_state.json`, `rss_state.json` und `scheduler_state.json`.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger("core.storage")

_DB_LOCK = threading.RLock()
_FILE_LOCKS: dict[str, threading.RLock] = {}
_FILE_LOCKS_GUARD = threading.Lock()


def _get_file_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _FILE_LOCKS_GUARD:
        if key not in _FILE_LOCKS:
            _FILE_LOCKS[key] = threading.RLock()
        return _FILE_LOCKS[key]


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Schreibt JSON atomar über eine temporäre Datei im selben Verzeichnis."""
    lock = _get_file_lock(path)
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=indent), encoding="utf-8")
        os.replace(tmp, path)


def safe_read_json(path: Path, default: Any) -> Any:
    lock = _get_file_lock(path)
    with lock:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("JSON-Datei %s nicht lesbar (%s) – nutze Default.", path.name, exc)
            return default


class StateStore:
    """Thread-sicherer SQLite-Store im WAL-Modus."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_schema(self) -> None:
        with _DB_LOCK:
            with self._connect() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS kv (
                        key TEXT PRIMARY KEY,
                        value_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        bot TEXT NOT NULL,
                        chat_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_chat_hist_lookup
                        ON chat_history (bot, chat_id, id DESC);

                    CREATE TABLE IF NOT EXISTS undo_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        bot TEXT NOT NULL,
                        chat_id TEXT NOT NULL,
                        action_type TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        undone INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS rss_seen (
                        fp TEXT PRIMARY KEY,
                        seen_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS rss_knowledge (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        topic_id TEXT NOT NULL,
                        item_key TEXT NOT NULL,
                        title TEXT NOT NULL,
                        date TEXT,
                        category TEXT,
                        summary TEXT,
                        link TEXT,
                        relevance TEXT,
                        notion_page_id TEXT,
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_rss_know_topic
                        ON rss_knowledge (topic_id, id DESC);

                    CREATE TABLE IF NOT EXISTS rss_runs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        at TEXT NOT NULL,
                        new_count INTEGER NOT NULL,
                        digest_url TEXT
                    );

                    CREATE TABLE IF NOT EXISTS rss_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        topic_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        link TEXT,
                        feedback TEXT NOT NULL,
                        reason TEXT,
                        created_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS feed_health (
                        source_key TEXT PRIMARY KEY,
                        topic_id TEXT NOT NULL,
                        source_name TEXT NOT NULL,
                        fail_count INTEGER NOT NULL DEFAULT 0,
                        last_error TEXT,
                        last_ok_at TEXT,
                        disabled_until TEXT
                    );

                    CREATE TABLE IF NOT EXISTS work_busy_slots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        start_time TEXT,
                        end_time TEXT,
                        source TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_work_busy_date ON work_busy_slots (date);

                    CREATE TABLE IF NOT EXISTS pre_meeting_pings (
                        event_key TEXT PRIMARY KEY,
                        pinged_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS callback_payloads (
                        token TEXT PRIMARY KEY,
                        kind TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    """
                )

    # ------------------------------------------------------------------
    # Key-Value Helpers (Offsets, Chat-IDs, Scheduler, Task-Meta)
    # ------------------------------------------------------------------

    def get_kv(self, key: str, default: Any = None) -> Any:
        with _DB_LOCK:
            with self._connect() as conn:
                row = conn.execute("SELECT value_json FROM kv WHERE key = ?", (key,)).fetchone()
                if not row:
                    return default
                try:
                    return json.loads(row["value_json"])
                except Exception:
                    return default

    def set_kv(self, key: str, value: Any) -> None:
        now = dt.datetime.now().isoformat(timespec="seconds")
        raw = json.dumps(value, ensure_ascii=False)
        with _DB_LOCK:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO kv (key, value_json, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value_json = excluded.value_json,
                        updated_at = excluded.updated_at
                    """,
                    (key, raw, now),
                )

    # ------------------------------------------------------------------
    # Chat-Memory (Multi-Turn)
    # ------------------------------------------------------------------

    def append_chat_turn(self, bot: str, chat_id: str, role: str, content: str, keep_last: int = 20) -> None:
        if not chat_id or not content:
            return
        now = dt.datetime.now().isoformat(timespec="seconds")
        with _DB_LOCK:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO chat_history (bot, chat_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                    (bot, str(chat_id), role, content[:2500], now),
                )
                conn.execute(
                    """
                    DELETE FROM chat_history
                    WHERE bot = ? AND chat_id = ?
                      AND id NOT IN (
                          SELECT id FROM chat_history
                          WHERE bot = ? AND chat_id = ?
                          ORDER BY id DESC LIMIT ?
                      )
                    """,
                    (bot, str(chat_id), bot, str(chat_id), keep_last),
                )

    def get_recent_chat_turns(self, bot: str, chat_id: str, limit: int = 8) -> list[dict[str, str]]:
        if not chat_id:
            return []
        with _DB_LOCK:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT role, content, created_at
                    FROM chat_history
                    WHERE bot = ? AND chat_id = ?
                    ORDER BY id DESC LIMIT ?
                    """,
                    (bot, str(chat_id), limit),
                ).fetchall()
        return [{"role": r["role"], "content": r["content"], "at": r["created_at"]} for r in reversed(rows)]

    # ------------------------------------------------------------------
    # Undo-Log
    # ------------------------------------------------------------------

    def push_undo(self, bot: str, chat_id: str, action_type: str, payload: dict[str, Any]) -> int:
        now = dt.datetime.now().isoformat(timespec="seconds")
        with _DB_LOCK:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO undo_log (bot, chat_id, action_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
                    (bot, str(chat_id), action_type, json.dumps(payload, ensure_ascii=False), now),
                )
                return int(cur.lastrowid or 0)

    def pop_last_undo(self, bot: str, chat_id: str) -> dict[str, Any] | None:
        with _DB_LOCK:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT id, action_type, payload_json, created_at
                    FROM undo_log
                    WHERE bot = ? AND chat_id = ? AND undone = 0
                    ORDER BY id DESC LIMIT 1
                    """,
                    (bot, str(chat_id)),
                ).fetchone()
                if not row:
                    return None
                conn.execute("UPDATE undo_log SET undone = 1 WHERE id = ?", (row["id"],))
                return {
                    "id": row["id"],
                    "action_type": row["action_type"],
                    "payload": json.loads(row["payload_json"]),
                    "created_at": row["created_at"],
                }

    # ------------------------------------------------------------------
    # Callback Payloads (kompakte Token für Telegram 64-Byte-Limit)
    # ------------------------------------------------------------------

    def put_callback_payload(self, token: str, kind: str, payload: dict[str, Any]) -> None:
        now = dt.datetime.now().isoformat(timespec="seconds")
        with _DB_LOCK:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO callback_payloads (token, kind, payload_json, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(token) DO UPDATE SET
                        kind = excluded.kind,
                        payload_json = excluded.payload_json,
                        created_at = excluded.created_at
                    """,
                    (token, kind, json.dumps(payload, ensure_ascii=False), now),
                )
                # Alte Payloads (>14 Tage) aufräumen
                cutoff = (dt.datetime.now() - dt.timedelta(days=14)).isoformat(timespec="seconds")
                conn.execute("DELETE FROM callback_payloads WHERE created_at < ?", (cutoff,))

    def get_callback_payload(self, token: str) -> tuple[str, dict[str, Any]] | None:
        with _DB_LOCK:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT kind, payload_json FROM callback_payloads WHERE token = ?",
                    (token,),
                ).fetchone()
                if not row:
                    return None
                return row["kind"], json.loads(row["payload_json"])

    # ------------------------------------------------------------------
    # RSS Feedback, Feed Health & Knowledge Search
    # ------------------------------------------------------------------

    def add_rss_feedback(
        self,
        topic_id: str,
        title: str,
        link: str = "",
        feedback: str = "irrelevant",
        reason: str = "",
    ) -> None:
        now = dt.datetime.now().isoformat(timespec="seconds")
        with _DB_LOCK:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO rss_feedback (topic_id, title, link, feedback, reason, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (topic_id, title[:240], link[:400], feedback, reason[:200], now),
                )

    def get_recent_negative_feedback(self, topic_id: str, limit: int = 8) -> list[str]:
        with _DB_LOCK:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT title FROM rss_feedback
                    WHERE topic_id = ? AND feedback = 'irrelevant'
                    ORDER BY id DESC LIMIT ?
                    """,
                    (topic_id, limit),
                ).fetchall()
        return [r["title"] for r in rows]

    def record_feed_success(self, source_key: str, topic_id: str, source_name: str) -> None:
        now = dt.datetime.now().isoformat(timespec="seconds")
        with _DB_LOCK:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO feed_health (source_key, topic_id, source_name, fail_count, last_error, last_ok_at, disabled_until)
                    VALUES (?, ?, ?, 0, NULL, ?, NULL)
                    ON CONFLICT(source_key) DO UPDATE SET
                        fail_count = 0,
                        last_error = NULL,
                        last_ok_at = excluded.last_ok_at,
                        disabled_until = NULL
                    """,
                    (source_key, topic_id, source_name, now),
                )

    def record_feed_failure(
        self,
        source_key: str,
        topic_id: str,
        source_name: str,
        error_msg: str,
        trip_threshold: int = 5,
        cooldown_hours: int = 12,
    ) -> bool:
        """Erhöht Fehlerzähler und öffnet ggf. den Circuit Breaker (gibt True zurück wenn pausiert)."""
        with _DB_LOCK:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT fail_count FROM feed_health WHERE source_key = ?",
                    (source_key,),
                ).fetchone()
                new_count = (int(row["fail_count"]) if row else 0) + 1
                disabled_until = None
                if new_count >= trip_threshold:
                    disabled_until = (dt.datetime.now() + dt.timedelta(hours=cooldown_hours)).isoformat(timespec="seconds")
                conn.execute(
                    """
                    INSERT INTO feed_health (source_key, topic_id, source_name, fail_count, last_error, disabled_until)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_key) DO UPDATE SET
                        fail_count = excluded.fail_count,
                        last_error = excluded.last_error,
                        disabled_until = excluded.disabled_until
                    """,
                    (source_key, topic_id, source_name, new_count, error_msg[:300], disabled_until),
                )
                return disabled_until is not None

    def is_feed_tripped(self, source_key: str) -> bool:
        now_iso = dt.datetime.now().isoformat(timespec="seconds")
        with _DB_LOCK:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT disabled_until FROM feed_health WHERE source_key = ?",
                    (source_key,),
                ).fetchone()
                if not row or not row["disabled_until"]:
                    return False
                return str(row["disabled_until"]) > now_iso

    def get_degraded_feeds(self) -> list[dict[str, Any]]:
        with _DB_LOCK:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT topic_id, source_name, fail_count, last_error, disabled_until
                    FROM feed_health
                    WHERE fail_count >= 2
                    ORDER BY fail_count DESC
                    """
                ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # RSS Knowledge Persistence & Hybrid Search
    # ------------------------------------------------------------------

    def add_knowledge_record(self, rec: dict[str, Any]) -> None:
        now = dt.datetime.now().isoformat(timespec="seconds")
        with _DB_LOCK:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO rss_knowledge
                    (topic_id, item_key, title, date, category, summary, link, relevance, notion_page_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rec.get("topic_id", ""),
                        rec.get("key", ""),
                        rec.get("title", ""),
                        rec.get("date") or rec.get("published") or dt.date.today().isoformat(),
                        rec.get("category", ""),
                        rec.get("summary", ""),
                        rec.get("link", ""),
                        rec.get("relevance", "Mittel"),
                        rec.get("notion_page_id", ""),
                        now,
                    ),
                )

    def get_recent_knowledge(self, topic_id: str | None = None, days: int = 7, limit: int = 50) -> list[dict[str, Any]]:
        cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat()
        with _DB_LOCK:
            with self._connect() as conn:
                if topic_id:
                    rows = conn.execute(
                        """
                        SELECT * FROM rss_knowledge
                        WHERE topic_id = ? AND (date >= ? OR created_at >= ?)
                        ORDER BY id DESC LIMIT ?
                        """,
                        (topic_id, cutoff, cutoff, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT * FROM rss_knowledge
                        WHERE date >= ? OR created_at >= ?
                        ORDER BY id DESC LIMIT ?
                        """,
                        (cutoff, cutoff, limit),
                    ).fetchall()
        return [dict(r) for r in rows]

    def search_knowledge_bm25(self, query: str, limit: int = 6) -> list[dict[str, Any]]:
        """Durchsucht gespeicherte RSS-Einträge per Token-Overlap / TF-IDF-ähnlichem Scoring."""
        tokens = [t for t in re.findall(r"[a-z0-9äöüß]{3,}", query.lower())]
        if not tokens:
            return []
        with _DB_LOCK:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT topic_id, item_key, title, date, category, summary, link, relevance FROM rss_knowledge ORDER BY id DESC LIMIT 800"
                ).fetchall()
        if not rows:
            return []

        docs = [dict(r) for r in rows]
        n_docs = len(docs)
        df: dict[str, int] = {tok: 0 for tok in tokens}
        for d in docs:
            hay = f"{d.get('title', '')} {d.get('summary', '')} {d.get('category', '')}".lower()
            for tok in tokens:
                if tok in hay:
                    df[tok] += 1

        scored: list[tuple[float, dict[str, Any]]] = []
        for d in docs:
            title_l = (d.get("title") or "").lower()
            sum_l = (d.get("summary") or "").lower()
            score = 0.0
            for tok in tokens:
                tf = title_l.count(tok) * 2.5 + sum_l.count(tok) * 1.0
                if tf > 0:
                    idf = math.log((n_docs + 1) / (df[tok] + 1)) + 1.0
                    score += tf * idf
            if score > 0:
                scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:limit]]

    # ------------------------------------------------------------------
    # Work Calendar Busy Slots (STRIKT ANONYM)
    # ------------------------------------------------------------------

    def replace_work_busy_slots(self, slots: list[dict[str, Any]], source: str) -> int:
        """Speichert ausschließlich anonyme Zeitblöcke (date, start_time, end_time) – niemals Titel!"""
        now = dt.datetime.now().isoformat(timespec="seconds")
        clean_rows: list[tuple[str, str | None, str | None, str, str]] = []
        for s in slots:
            d = str(s.get("date") or "").strip()
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
                continue
            st = s.get("start_time")
            et = s.get("end_time")
            clean_rows.append((d, st, et, source, now))

        with _DB_LOCK:
            with self._connect() as conn:
                conn.execute("DELETE FROM work_busy_slots WHERE source = ?", (source,))
                conn.executemany(
                    "INSERT INTO work_busy_slots (date, start_time, end_time, source, updated_at) VALUES (?, ?, ?, ?, ?)",
                    clean_rows,
                )
        return len(clean_rows)

    def get_work_busy_slots(self, start_date: str, end_date: str) -> list[dict[str, Any]]:
        with _DB_LOCK:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT date, start_time, end_time, source
                    FROM work_busy_slots
                    WHERE date >= ? AND date <= ?
                    ORDER BY date ASC, COALESCE(start_time, '00:00') ASC
                    """,
                    (start_date, end_date),
                ).fetchall()
        return [
            {
                "title": "Arbeit: Belegt",
                "date": r["date"],
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "location": "",
                "source": f"Work Free/Busy ({r['source']})",
                "is_work_busy_only": True,
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Pre-Meeting Pings & Task Defer Tracking
    # ------------------------------------------------------------------

    def was_meeting_pinged(self, event_key: str) -> bool:
        with _DB_LOCK:
            with self._connect() as conn:
                row = conn.execute("SELECT 1 FROM pre_meeting_pings WHERE event_key = ?", (event_key,)).fetchone()
                return row is not None

    def mark_meeting_pinged(self, event_key: str) -> None:
        now = dt.datetime.now().isoformat(timespec="seconds")
        with _DB_LOCK:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO pre_meeting_pings (event_key, pinged_at) VALUES (?, ?)",
                    (event_key, now),
                )
                cutoff = (dt.datetime.now() - dt.timedelta(days=14)).isoformat(timespec="seconds")
                conn.execute("DELETE FROM pre_meeting_pings WHERE pinged_at < ?", (cutoff,))

    def increment_task_defer(self, task_key: str) -> int:
        counts = self.get_kv("task_defer_counts", {}) or {}
        val = int(counts.get(task_key, 0)) + 1
        counts[task_key] = val
        self.set_kv("task_defer_counts", counts)
        return val

    def get_task_defer_count(self, task_key: str) -> int:
        counts = self.get_kv("task_defer_counts", {}) or {}
        return int(counts.get(task_key, 0))

    # ------------------------------------------------------------------
    # Legacy JSON Migration
    # ------------------------------------------------------------------

    def migrate_from_legacy_json(self, base_dir: Path) -> None:
        """Importiert vorhandene rss_state.json / scheduler_state.json einmalig in SQLite."""
        if self.get_kv("legacy_migrated", False):
            return
        rss_json = base_dir / "rss_state.json"
        if rss_json.exists():
            data = safe_read_json(rss_json, {})
            if isinstance(data, dict):
                with _DB_LOCK:
                    with self._connect() as conn:
                        for fp, ts in (data.get("seen") or {}).items():
                            conn.execute(
                                "INSERT OR IGNORE INTO rss_seen (fp, seen_at) VALUES (?, ?)",
                                (str(fp), str(ts)),
                            )
                        for tid, entries in (data.get("knowledge") or {}).items():
                            for e in entries or []:
                                conn.execute(
                                    """
                                    INSERT INTO rss_knowledge (topic_id, item_key, title, date, category, link, created_at)
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        str(tid),
                                        str(e.get("key", "")),
                                        str(e.get("title", "")),
                                        str(e.get("date", "")),
                                        str(e.get("category", "")),
                                        str(e.get("link", "")),
                                        dt.datetime.now().isoformat(timespec="seconds"),
                                    ),
                                )
        self.set_kv("legacy_migrated", True)
