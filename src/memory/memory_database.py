#!/usr/bin/env python3
"""SQLite storage for the unified memory prototype.

The schema deliberately keeps a few legacy table names used by existing
benchmark scripts (`memory_facts`, `memory_observations`,
`memory_interpretations`, `entity_nodes`) while adding the new unified line:

    memory_episodes -> memory_facts -> memory_states/actionable_items
    -> memory_index_entries

`memory_index_entries` is the MemPalace-style directory layer: every retrievable
memory object writes one index card that points back to its source row.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

_HAS_FAISS = False
EMBEDDING_DIM = 384


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _embedding_to_blob(embedding: Optional[np.ndarray]) -> Optional[bytes]:
    if embedding is None:
        return None
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    return vector.tobytes()


def _blob_to_embedding(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        vector = np.frombuffer(bytes(value), dtype=np.float32)
    except Exception:
        return None
    return vector.reshape(1, -1).astype(np.float32)


class SessionDB:
    """Small DB facade compatible with the current LongMemEval scripts."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,
                episode_type TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                participants TEXT NOT NULL DEFAULT '[]',
                started_at TEXT,
                ended_at TEXT,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER,
                source_type TEXT NOT NULL DEFAULT 'assistant_wakeup',
                fact_type TEXT NOT NULL DEFAULT 'episodic',
                fact_kind TEXT NOT NULL DEFAULT 'context',
                fact_subject TEXT NOT NULL DEFAULT 'user',
                summary TEXT NOT NULL,
                keywords TEXT NOT NULL DEFAULT '',
                entities TEXT NOT NULL DEFAULT '[]',
                canonical_topics TEXT NOT NULL DEFAULT '[]',
                time_key TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.85,
                importance REAL NOT NULL DEFAULT 0.5,
                metadata TEXT NOT NULL DEFAULT '{}',
                embedding BLOB,
                embedding_text TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(episode_id) REFERENCES memory_episodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state_type TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'unified',
                canonical_name TEXT NOT NULL,
                summary TEXT NOT NULL,
                evidence_fact_ids TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL DEFAULT 0.75,
                metadata TEXT NOT NULL DEFAULT '{}',
                embedding BLOB,
                embedding_text TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(source_type, state_type, canonical_name)
            );

            CREATE TABLE IF NOT EXISTS memory_actionable_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_type TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'unified',
                canonical_name TEXT NOT NULL,
                summary TEXT NOT NULL,
                owner TEXT NOT NULL DEFAULT 'unknown',
                status TEXT NOT NULL DEFAULT 'unknown',
                due_at TEXT NOT NULL DEFAULT '',
                evidence_fact_ids TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL DEFAULT 0.75,
                importance REAL NOT NULL DEFAULT 0.6,
                metadata TEXT NOT NULL DEFAULT '{}',
                embedding BLOB,
                embedding_text TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(source_type, item_type, canonical_name)
            );

            CREATE TABLE IF NOT EXISTS memory_index_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,
                target_table TEXT NOT NULL,
                target_id INTEGER NOT NULL,
                index_level TEXT NOT NULL,
                memory_path TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                summary_for_retrieval TEXT NOT NULL,
                keywords TEXT NOT NULL DEFAULT '',
                entities TEXT NOT NULL DEFAULT '[]',
                canonical_topics TEXT NOT NULL DEFAULT '[]',
                participants TEXT NOT NULL DEFAULT '[]',
                time_start TEXT,
                time_end TEXT,
                importance REAL NOT NULL DEFAULT 0.5,
                confidence REAL NOT NULL DEFAULT 0.8,
                embedding BLOB,
                embedding_text TEXT NOT NULL DEFAULT '',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(target_table, target_id, index_level)
            );

            CREATE TABLE IF NOT EXISTS entity_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                type TEXT NOT NULL DEFAULT 'OTHER',
                created_at TEXT NOT NULL DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_memory_facts_time ON memory_facts(time_key);
            CREATE INDEX IF NOT EXISTS idx_memory_facts_source ON memory_facts(source_type);
            CREATE INDEX IF NOT EXISTS idx_memory_index_source ON memory_index_entries(source_type);
            CREATE INDEX IF NOT EXISTS idx_memory_index_time ON memory_index_entries(time_start);
            CREATE INDEX IF NOT EXISTS idx_memory_states_source ON memory_states(source_type, state_type);
            CREATE INDEX IF NOT EXISTS idx_memory_actionable_source
            ON memory_actionable_items(source_type, item_type, status);
            """
        )
        self._conn.commit()

    def insert_episode(
        self,
        *,
        source_type: str,
        episode_type: str,
        title: str,
        summary: str,
        participants: Sequence[str],
        started_at: str,
        ended_at: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        now = utc_now_text()
        cur = self._conn.execute(
            """
            INSERT INTO memory_episodes (
                source_type, episode_type, title, summary, participants,
                started_at, ended_at, metadata, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_type,
                episode_type,
                title,
                summary,
                _json_dumps(list(participants or [])),
                started_at,
                ended_at,
                _json_dumps(metadata or {}),
                now,
                now,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def upsert_state(
        self,
        *,
        state_type: str,
        source_type: str,
        canonical_name: str,
        summary: str,
        evidence_fact_ids: Sequence[int],
        confidence: float,
        metadata: Optional[Dict[str, Any]],
        embedding: Optional[np.ndarray],
        embedding_text: str,
    ) -> int:
        now = utc_now_text()
        self._conn.execute(
            """
            INSERT INTO memory_states (
                state_type, source_type, canonical_name, summary,
                evidence_fact_ids, confidence, metadata, embedding,
                embedding_text, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_type, state_type, canonical_name) DO UPDATE SET
                summary = excluded.summary,
                evidence_fact_ids = excluded.evidence_fact_ids,
                confidence = excluded.confidence,
                metadata = excluded.metadata,
                embedding = excluded.embedding,
                embedding_text = excluded.embedding_text,
                updated_at = excluded.updated_at
            """,
            (
                str(state_type or "other"),
                str(source_type or "unified"),
                str(canonical_name or "general").strip(),
                str(summary or "").strip(),
                _json_dumps([int(value) for value in evidence_fact_ids or []]),
                float(confidence),
                _json_dumps(metadata or {}),
                _embedding_to_blob(embedding),
                str(embedding_text or ""),
                now,
                now,
            ),
        )
        self._conn.commit()
        row = self._conn.execute(
            """
            SELECT id FROM memory_states
            WHERE source_type = ? AND state_type = ? AND canonical_name = ?
            """,
            (
                str(source_type or "unified"),
                str(state_type or "other"),
                str(canonical_name or "general").strip(),
            ),
        ).fetchone()
        return int(row["id"]) if row else 0

    def upsert_actionable_item(
        self,
        *,
        item_type: str,
        source_type: str,
        canonical_name: str,
        summary: str,
        owner: str,
        status: str,
        due_at: str,
        evidence_fact_ids: Sequence[int],
        confidence: float,
        importance: float,
        metadata: Optional[Dict[str, Any]],
        embedding: Optional[np.ndarray],
        embedding_text: str,
    ) -> int:
        now = utc_now_text()
        self._conn.execute(
            """
            INSERT INTO memory_actionable_items (
                item_type, source_type, canonical_name, summary, owner,
                status, due_at, evidence_fact_ids, confidence, importance,
                metadata, embedding, embedding_text, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_type, item_type, canonical_name) DO UPDATE SET
                summary = excluded.summary,
                owner = excluded.owner,
                status = excluded.status,
                due_at = excluded.due_at,
                evidence_fact_ids = excluded.evidence_fact_ids,
                confidence = excluded.confidence,
                importance = excluded.importance,
                metadata = excluded.metadata,
                embedding = excluded.embedding,
                embedding_text = excluded.embedding_text,
                updated_at = excluded.updated_at
            """,
            (
                str(item_type or "other"),
                str(source_type or "unified"),
                str(canonical_name or "general").strip(),
                str(summary or "").strip(),
                str(owner or "unknown").strip(),
                str(status or "unknown").strip(),
                str(due_at or "").strip(),
                _json_dumps([int(value) for value in evidence_fact_ids or []]),
                float(confidence),
                float(importance),
                _json_dumps(metadata or {}),
                _embedding_to_blob(embedding),
                str(embedding_text or ""),
                now,
                now,
            ),
        )
        self._conn.commit()
        row = self._conn.execute(
            """
            SELECT id FROM memory_actionable_items
            WHERE source_type = ? AND item_type = ? AND canonical_name = ?
            """,
            (
                str(source_type or "unified"),
                str(item_type or "other"),
                str(canonical_name or "general").strip(),
            ),
        ).fetchone()
        return int(row["id"]) if row else 0

    def recent_facts(
        self,
        *,
        source_types: Optional[Sequence[str]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if source_types:
            placeholders = ",".join("?" for _ in source_types)
            clauses.append(f"source_type IN ({placeholders})")
            params.extend(source_types)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn.execute(
            f"""
            SELECT * FROM memory_facts
            {where}
            ORDER BY time_key DESC, id DESC
            LIMIT ?
            """,
            (*params, max(1, int(limit or 100))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def recent_states(
        self,
        *,
        source_types: Optional[Sequence[str]] = None,
        limit: int = 80,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if source_types:
            placeholders = ",".join("?" for _ in source_types)
            clauses.append(f"source_type IN ({placeholders})")
            params.extend(source_types)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn.execute(
            f"""
            SELECT * FROM memory_states
            {where}
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            (*params, max(1, int(limit or 80))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def recent_actionable_items(
        self,
        *,
        source_types: Optional[Sequence[str]] = None,
        limit: int = 80,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if source_types:
            placeholders = ",".join("?" for _ in source_types)
            clauses.append(f"source_type IN ({placeholders})")
            params.extend(source_types)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn.execute(
            f"""
            SELECT * FROM memory_actionable_items
            {where}
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            (*params, max(1, int(limit or 80))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def insert_fact(
        self,
        *,
        episode_id: Optional[int],
        source_type: str,
        fact_type: str,
        fact_kind: str,
        fact_subject: str,
        summary: str,
        keywords: str,
        entities: Sequence[str],
        canonical_topics: Sequence[str],
        time_key: str,
        confidence: float,
        importance: float,
        metadata: Optional[Dict[str, Any]],
        embedding: Optional[np.ndarray],
        embedding_text: str,
    ) -> int:
        now = utc_now_text()
        cur = self._conn.execute(
            """
            INSERT INTO memory_facts (
                episode_id, source_type, fact_type, fact_kind, fact_subject,
                summary, keywords, entities, canonical_topics, time_key,
                confidence, importance, metadata, embedding, embedding_text,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode_id,
                source_type,
                fact_type,
                fact_kind,
                fact_subject,
                summary,
                keywords,
                _json_dumps(list(entities or [])),
                _json_dumps(list(canonical_topics or [])),
                time_key,
                float(confidence),
                float(importance),
                _json_dumps(metadata or {}),
                _embedding_to_blob(embedding),
                embedding_text,
                now,
                now,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def upsert_index_entry(
        self,
        *,
        source_type: str,
        target_table: str,
        target_id: int,
        index_level: str,
        memory_path: str,
        title: str,
        summary_for_retrieval: str,
        keywords: str,
        entities: Sequence[str],
        canonical_topics: Sequence[str],
        participants: Sequence[str],
        time_start: str,
        time_end: str,
        importance: float,
        confidence: float,
        embedding: Optional[np.ndarray],
        embedding_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        now = utc_now_text()
        self._conn.execute(
            """
            INSERT INTO memory_index_entries (
                source_type, target_table, target_id, index_level, memory_path,
                title, summary_for_retrieval, keywords, entities, canonical_topics,
                participants, time_start, time_end, importance, confidence,
                embedding, embedding_text, metadata, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(target_table, target_id, index_level) DO UPDATE SET
                source_type = excluded.source_type,
                memory_path = excluded.memory_path,
                title = excluded.title,
                summary_for_retrieval = excluded.summary_for_retrieval,
                keywords = excluded.keywords,
                entities = excluded.entities,
                canonical_topics = excluded.canonical_topics,
                participants = excluded.participants,
                time_start = excluded.time_start,
                time_end = excluded.time_end,
                importance = excluded.importance,
                confidence = excluded.confidence,
                embedding = excluded.embedding,
                embedding_text = excluded.embedding_text,
                metadata = excluded.metadata,
                updated_at = excluded.updated_at
            """,
            (
                source_type,
                target_table,
                int(target_id),
                index_level,
                memory_path,
                title,
                summary_for_retrieval,
                keywords,
                _json_dumps(list(entities or [])),
                _json_dumps(list(canonical_topics or [])),
                _json_dumps(list(participants or [])),
                time_start,
                time_end,
                float(importance),
                float(confidence),
                _embedding_to_blob(embedding),
                embedding_text,
                _json_dumps(metadata or {}),
                now,
                now,
            ),
        )
        self._conn.commit()
        row = self._conn.execute(
            """
            SELECT id FROM memory_index_entries
            WHERE target_table = ? AND target_id = ? AND index_level = ?
            """,
            (target_table, int(target_id), index_level),
        ).fetchone()
        return int(row["id"]) if row else 0

    def add_entity_names(self, names: Iterable[str]) -> None:
        now = utc_now_text()
        for name in names:
            clean = str(name or "").strip()
            if not clean:
                continue
            self._conn.execute(
                "INSERT OR IGNORE INTO entity_nodes (name, type, created_at) VALUES (?, ?, ?)",
                (clean, "OTHER", now),
            )
        self._conn.commit()

    def search_index_entries(
        self,
        *,
        source_types: Optional[Sequence[str]] = None,
        index_levels: Optional[Sequence[str]] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if source_types:
            placeholders = ",".join("?" for _ in source_types)
            clauses.append(f"source_type IN ({placeholders})")
            params.extend(source_types)
        if index_levels:
            placeholders = ",".join("?" for _ in index_levels)
            clauses.append(f"index_level IN ({placeholders})")
            params.extend(index_levels)
        if time_start:
            clauses.append("(time_start IS NULL OR time_start = '' OR time_start >= ?)")
            params.append(str(time_start))
        if time_end:
            clauses.append("(time_start IS NULL OR time_start = '' OR time_start <= ?)")
            params.append(str(time_end))
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn.execute(
            f"""
            SELECT * FROM memory_index_entries
            {where}
            ORDER BY time_start DESC, id DESC
            LIMIT ?
            """,
            (*params, max(1, int(limit or 200))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def memory_facts_by_ids(self, fact_ids: Sequence[int]) -> List[Dict[str, Any]]:
        ids = [int(value) for value in fact_ids if value is not None]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT * FROM memory_facts WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        by_id = {int(row["id"]): self._row_to_dict(row) for row in rows}
        return [by_id[item] for item in ids if item in by_id]

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        item = dict(row)
        for key in ("entities", "canonical_topics", "participants", "metadata", "evidence_fact_ids"):
            if key in item:
                item[key] = _json_loads(item[key], [] if key != "metadata" else {})
        for key in ("embedding",):
            if key in item:
                item[key] = _blob_to_embedding(item[key])
        return item
