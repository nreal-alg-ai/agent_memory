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
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

_HAS_FAISS = False
EMBEDDING_DIM = 384


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_reference_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value or "").strip()
        if not text:
            dt = datetime.now().astimezone()
        else:
            try:
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return dt


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
                entity_ids TEXT NOT NULL DEFAULT '[]',
                canonical_topics TEXT NOT NULL DEFAULT '[]',
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
                entity_ids TEXT NOT NULL DEFAULT '[]',
                fact_root_topic TEXT NOT NULL DEFAULT '',
                fact_aspect_topic TEXT NOT NULL DEFAULT '',
                time_key TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.85,
                importance REAL NOT NULL DEFAULT 0.5,
                processed_for_memory_state INTEGER NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}',
                embedding BLOB,
                embedding_text TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(episode_id) REFERENCES memory_episodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state_scope TEXT NOT NULL DEFAULT 'entity_state',
                state_type TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'unified',
                entity_key TEXT NOT NULL DEFAULT '',
                canonical_name TEXT NOT NULL,
                summary TEXT NOT NULL,
                time_line TEXT NOT NULL DEFAULT '[]',
                entity_ids TEXT NOT NULL DEFAULT '[]',
                evidence_fact_ids TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL DEFAULT 0.75,
                metadata TEXT NOT NULL DEFAULT '{}',
                identity_text_embedding BLOB,
                canonical_name_embedding BLOB,
                identity_text TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(source_type, state_scope, state_type, entity_key, canonical_name)
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
                entity_ids TEXT NOT NULL DEFAULT '[]',
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
            CREATE INDEX IF NOT EXISTS idx_memory_facts_state_processing
            ON memory_facts(processed_for_memory_state, created_at);
            CREATE INDEX IF NOT EXISTS idx_memory_index_source ON memory_index_entries(source_type);
            CREATE INDEX IF NOT EXISTS idx_memory_index_time ON memory_index_entries(time_start);
            CREATE INDEX IF NOT EXISTS idx_memory_states_source ON memory_states(source_type, state_type);
            CREATE INDEX IF NOT EXISTS idx_memory_states_scope
            ON memory_states(source_type, state_scope, state_type);
            CREATE INDEX IF NOT EXISTS idx_memory_actionable_source
            ON memory_actionable_items(source_type, item_type, status);
            """
        )
        self._ensure_entity_ids_schema()
        self._ensure_memory_states_scope_schema()
        self._ensure_memory_states_time_line_schema()
        self._ensure_memory_states_entity_key_schema()
        self._ensure_memory_states_canonical_name_embedding_schema()
        self._init_index_fts()
        self._conn.commit()

    def _ensure_entity_ids_schema(self) -> None:
        """Ensure all primary memory tables expose direct entity id mappings."""
        for table in (
            "memory_episodes",
            "memory_facts",
            "memory_states",
            "memory_actionable_items",
        ):
            columns = {
                str(row["name"])
                for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if "entity_ids" not in columns:
                self._conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN entity_ids TEXT NOT NULL DEFAULT '[]'"
                )

    def _ensure_memory_states_scope_schema(self) -> None:
        """Normalize the state scope columns for databases created earlier."""
        columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(memory_states)").fetchall()
        }
        if "state_scope" not in columns:
            self._conn.execute(
                "ALTER TABLE memory_states ADD COLUMN state_scope TEXT NOT NULL DEFAULT 'entity_state'"
            )
            self._conn.execute(
                """
                UPDATE memory_states
                SET state_scope = 'topic_state', state_type = 'topic'
                WHERE state_type = 'topic_state'
                """
            )
        else:
            self._conn.execute(
                """
                UPDATE memory_states
                SET state_scope = 'topic_state', state_type = 'topic'
                WHERE state_type = 'topic_state'
                """
            )

    def _ensure_memory_states_time_line_schema(self) -> None:
        """Add the state change timeline column to existing databases."""
        columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(memory_states)").fetchall()
        }
        if "time_line" not in columns:
            self._conn.execute(
                "ALTER TABLE memory_states ADD COLUMN time_line TEXT NOT NULL DEFAULT '[]'"
            )

    def _ensure_memory_states_entity_key_schema(self) -> None:
        """Keep same-named entity states separate from topic states."""
        columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(memory_states)").fetchall()
        }
        if "entity_key" in columns:
            return
        rows = self._conn.execute("SELECT * FROM memory_states ORDER BY id").fetchall()
        self._conn.execute("DROP INDEX IF EXISTS idx_memory_states_source")
        self._conn.execute("DROP INDEX IF EXISTS idx_memory_states_scope")
        self._conn.execute("ALTER TABLE memory_states RENAME TO memory_states_legacy")
        self._conn.execute(
            """
            CREATE TABLE memory_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state_scope TEXT NOT NULL DEFAULT 'entity_state',
                state_type TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'unified',
                entity_key TEXT NOT NULL DEFAULT '',
                canonical_name TEXT NOT NULL,
                summary TEXT NOT NULL,
                time_line TEXT NOT NULL DEFAULT '[]',
                entity_ids TEXT NOT NULL DEFAULT '[]',
                evidence_fact_ids TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL DEFAULT 0.75,
                metadata TEXT NOT NULL DEFAULT '{}',
                identity_text_embedding BLOB,
                canonical_name_embedding BLOB,
                identity_text TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(source_type, state_scope, state_type, entity_key, canonical_name)
            )
            """
        )
        for row in rows:
            metadata = _json_loads(row["metadata"], {})
            scope = str(row["state_scope"] or "entity_state")
            entity_key = ""
            if scope == "entity_state" and isinstance(metadata, dict):
                entity_key = str(
                    metadata.get("entity_key")
                    or metadata.get("entity")
                    or ""
                ).strip().lower()
            self._conn.execute(
                """
                INSERT INTO memory_states (
                    id, state_scope, state_type, source_type, entity_key,
                    canonical_name, summary, time_line, entity_ids, evidence_fact_ids,
                    confidence, metadata, identity_text_embedding, canonical_name_embedding,
                    identity_text,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"], row["state_scope"], row["state_type"],
                    row["source_type"], entity_key, row["canonical_name"],
                    row["summary"], row["time_line"], row["entity_ids"],
                    row["evidence_fact_ids"], row["confidence"], row["metadata"],
                    row["identity_text_embedding"],
                    row["canonical_name_embedding"]
                    if "canonical_name_embedding" in row.keys()
                    else None,
                    row["identity_text"], row["created_at"],
                    row["updated_at"],
                ),
            )
        self._conn.execute("DROP TABLE memory_states_legacy")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_states_source ON memory_states(source_type, state_type)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_states_scope ON memory_states(source_type, state_scope, state_type)"
        )

    def _ensure_memory_states_canonical_name_embedding_schema(self) -> None:
        """Add the persisted canonical-name vector used by state matching."""
        columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(memory_states)").fetchall()
        }
        if "canonical_name_embedding" not in columns:
            self._conn.execute(
                "ALTER TABLE memory_states ADD COLUMN canonical_name_embedding BLOB"
            )

    def _init_index_fts(self) -> None:
        """Create the lightweight lexical index used for first-pass recall."""
        self._conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_index_entries_fts USING fts5(
                title,
                summary_for_retrieval,
                keywords,
                entities,
                canonical_topics,
                participants,
                memory_path
            )
            """
        )
        self._backfill_index_fts()

    def _backfill_index_fts(self) -> None:
        """Populate FTS rows for existing index cards not yet synchronized."""
        self._conn.execute(
            """
            INSERT INTO memory_index_entries_fts (
                rowid, title, summary_for_retrieval, keywords, entities,
                canonical_topics, participants, memory_path
            )
            SELECT
                i.id,
                i.title,
                i.summary_for_retrieval,
                i.keywords,
                i.entities,
                i.canonical_topics,
                i.participants,
                i.memory_path
            FROM memory_index_entries i
            WHERE NOT EXISTS (
                SELECT 1
                FROM memory_index_entries_fts f
                WHERE f.rowid = i.id
            )
            """
        )

    @staticmethod
    def _terms_to_fts_query(terms: Sequence[str]) -> str:
        quoted: List[str] = []
        for term in terms or []:
            clean = re.sub(r"\s+", " ", str(term or "").strip())
            clean = clean.replace('"', '""')
            if clean:
                quoted.append(f'"{clean}"')
            if len(quoted) >= 12:
                break
        return " OR ".join(quoted)

    @staticmethod
    def _normalize_search_terms(terms: Optional[Sequence[str]]) -> List[str]:
        normalized: List[str] = []
        for term in terms or []:
            clean = re.sub(r"\s+", " ", str(term or "").strip().lower())
            if clean and clean not in normalized:
                normalized.append(clean)
            if len(normalized) >= 16:
                break
        return normalized

    def _sync_index_entry_fts(
        self,
        *,
        row_id: int,
        title: str,
        summary_for_retrieval: str,
        keywords: str,
        entities: Sequence[str],
        canonical_topics: Sequence[str],
        participants: Sequence[str],
        memory_path: str,
    ) -> None:
        self._conn.execute(
            "DELETE FROM memory_index_entries_fts WHERE rowid = ?",
            (int(row_id),),
        )
        self._conn.execute(
            """
            INSERT INTO memory_index_entries_fts (
                rowid, title, summary_for_retrieval, keywords, entities,
                canonical_topics, participants, memory_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(row_id),
                str(title or ""),
                str(summary_for_retrieval or ""),
                str(keywords or ""),
                " ".join(str(item or "") for item in entities or []),
                " ".join(str(item or "") for item in canonical_topics or []),
                " ".join(str(item or "") for item in participants or []),
                str(memory_path or ""),
            ),
        )

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
        canonical_topics: Optional[Sequence[str]] = None,
        entity_ids: Optional[Sequence[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        now = utc_now_text()
        episode_metadata = dict(metadata or {})
        episode_metadata.pop("canonical_topics", None)
        normalized_topics = list(canonical_topics or [])
        cur = self._conn.execute(
            """
            INSERT INTO memory_episodes (
                source_type, episode_type, title, summary, participants,
                entity_ids, canonical_topics, started_at, ended_at,
                metadata, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_type,
                episode_type,
                title,
                summary,
                _json_dumps(list(participants or [])),
                _json_dumps([int(value) for value in entity_ids or []]),
                _json_dumps(normalized_topics),
                started_at,
                ended_at,
                _json_dumps(episode_metadata),
                now,
                now,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def upsert_state(
        self,
        *,
        state_scope: str,
        state_type: str,
        source_type: str,
        entity_key: str,
        canonical_name: str,
        summary: str,
        time_line: Optional[Sequence[Dict[str, Any]]],
        entity_ids: Optional[Sequence[int]],
        evidence_fact_ids: Sequence[int],
        confidence: float,
        metadata: Optional[Dict[str, Any]],
        identity_text_embedding: Optional[np.ndarray],
        canonical_name_embedding: Optional[np.ndarray],
        identity_text: str,
    ) -> int:
        now = utc_now_text()
        normalized_scope = str(state_scope or "entity_state").strip()
        normalized_type = str(state_type or "profile").strip()
        normalized_source = str(source_type or "unified")
        normalized_entity_key = str(entity_key or "").strip().lower()
        normalized_name = str(canonical_name or "general").strip()
        values = (
            normalized_scope,
            normalized_type,
            normalized_source,
            normalized_entity_key,
            normalized_name,
            str(summary or "").strip(),
            _json_dumps(list(time_line or [])),
            _json_dumps([int(value) for value in entity_ids or []]),
            _json_dumps([int(value) for value in evidence_fact_ids or []]),
            float(confidence),
            _json_dumps(metadata or {}),
            _embedding_to_blob(identity_text_embedding),
            _embedding_to_blob(canonical_name_embedding),
            str(identity_text or ""),
        )
        existing = self._conn.execute(
            """
            SELECT id FROM memory_states
            WHERE source_type = ? AND state_scope = ? AND state_type = ?
              AND entity_key = ? AND canonical_name = ?
            """,
            (
                normalized_source, normalized_scope, normalized_type,
                normalized_entity_key, normalized_name,
            ),
        ).fetchone()
        if existing:
            self._conn.execute(
                """
                UPDATE memory_states
                SET summary = ?, time_line = ?, entity_ids = ?, evidence_fact_ids = ?, confidence = ?,
                    metadata = ?, identity_text_embedding = ?, canonical_name_embedding = ?,
                    identity_text = ?, updated_at = ?
                WHERE id = ?
                """,
                (*values[5:], now, int(existing["id"])),
            )
        else:
            self._conn.execute(
                """
                INSERT INTO memory_states (
                    state_scope, state_type, source_type, entity_key, canonical_name, summary,
                    time_line, entity_ids, evidence_fact_ids, confidence, metadata,
                    identity_text_embedding, canonical_name_embedding, identity_text,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (*values, now, now),
            )
        self._conn.commit()
        row = self._conn.execute(
            """
            SELECT id FROM memory_states
            WHERE source_type = ? AND state_scope = ? AND state_type = ?
              AND entity_key = ? AND canonical_name = ?
            """,
            (
                normalized_source, normalized_scope, normalized_type,
                normalized_entity_key, normalized_name,
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
        entity_ids: Optional[Sequence[int]],
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
                status, due_at, entity_ids, evidence_fact_ids, confidence, importance,
                metadata, embedding, embedding_text, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_type, item_type, canonical_name) DO UPDATE SET
                summary = excluded.summary,
                owner = excluded.owner,
                status = excluded.status,
                due_at = excluded.due_at,
                entity_ids = excluded.entity_ids,
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
                _json_dumps([int(value) for value in entity_ids or []]),
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

    def get_unprocessed_facts_for_states(
        self,
        *,
        reference_timestamp: Any,
        source_types: Optional[Sequence[str]] = None,
        limit: int = 100,
        restrict_to_today: bool = True,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = ["processed_for_memory_state = 0"]
        params: List[Any] = []
        if source_types:
            placeholders = ",".join("?" for _ in source_types)
            clauses.append(f"source_type IN ({placeholders})")
            params.extend(source_types)
        if restrict_to_today:
            local_now = _coerce_reference_datetime(reference_timestamp).astimezone()
            event_date = local_now.date().isoformat()
            clauses.append("substr(time_key, 1, 10) = ?")
            params.append(event_date)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn.execute(
            f"""
            SELECT * FROM memory_facts
            {where}
            ORDER BY replace(substr(time_key, 1, 19), 'T', ' ') ASC, created_at ASC, id ASC
            LIMIT ?
            """,
            (*params, max(1, int(limit or 100))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def mark_facts_processed_for_memory_state(self, fact_ids: Sequence[int]) -> int:
        ids = [int(value) for value in fact_ids if value is not None]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        cur = self._conn.execute(
            f"""
            UPDATE memory_facts
            SET processed_for_memory_state = 1,
                updated_at = ?
            WHERE id IN ({placeholders})
            """,
            (utc_now_text(), *ids),
        )
        self._conn.commit()
        return int(cur.rowcount or 0)

    def get_recent_memory_states(
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
        entity_ids: Optional[Sequence[int]],
        fact_root_topic: str,
        fact_aspect_topic: str,
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
                summary, keywords, entities, entity_ids, fact_root_topic,
                fact_aspect_topic, time_key,
                confidence, importance, metadata, embedding, embedding_text,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                _json_dumps([int(value) for value in entity_ids or []]),
                str(fact_root_topic or ""),
                str(fact_aspect_topic or ""),
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
        row_id = int(row["id"]) if row else 0
        if row_id:
            self._sync_index_entry_fts(
                row_id=row_id,
                title=title,
                summary_for_retrieval=summary_for_retrieval,
                keywords=keywords,
                entities=entities,
                canonical_topics=canonical_topics,
                participants=participants,
                memory_path=memory_path,
            )
        self._conn.commit()
        return row_id

    def add_entity_names(self, names: Iterable[str]) -> Dict[str, int]:
        now = utc_now_text()
        normalized: List[str] = []
        for name in names:
            clean = str(name or "").strip()
            if not clean or clean in normalized:
                continue
            normalized.append(clean)
            self._conn.execute(
                "INSERT OR IGNORE INTO entity_nodes (name, type, created_at) VALUES (?, ?, ?)",
                (clean, "OTHER", now),
            )
        self._conn.commit()
        if not normalized:
            return {}
        placeholders = ",".join("?" for _ in normalized)
        rows = self._conn.execute(
            f"SELECT id, name FROM entity_nodes WHERE name IN ({placeholders})",
            normalized,
        ).fetchall()
        return {str(row["name"]): int(row["id"]) for row in rows}

    def search_index_entries(
        self,
        *,
        source_types: Optional[Sequence[str]] = None,
        index_levels: Optional[Sequence[str]] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        limit: int = 200,
        terms: Optional[Sequence[str]] = None,
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
        candidate_limit = max(1, int(limit or 200))
        normalized_terms = self._normalize_search_terms(terms)
        candidate_ids: List[int] = []
        candidate_sources: Dict[int, List[str]] = {}

        def add_ids(rows: Sequence[sqlite3.Row], source: str) -> None:
            for row in rows:
                row_id = int(row["id"])
                if row_id not in candidate_ids:
                    candidate_ids.append(row_id)
                candidate_sources.setdefault(row_id, [])
                if source not in candidate_sources[row_id]:
                    candidate_sources[row_id].append(source)

        if normalized_terms:
            match_query = self._terms_to_fts_query(normalized_terms)
            if match_query:
                try:
                    rows = self._conn.execute(
                        f"""
                        SELECT i.id
                        FROM memory_index_entries_fts f
                        JOIN memory_index_entries i ON i.id = f.rowid
                        {where}
                        AND memory_index_entries_fts MATCH ?
                        ORDER BY bm25(memory_index_entries_fts), i.time_start DESC, i.id DESC
                        LIMIT ?
                        """ if where else
                        """
                        SELECT i.id
                        FROM memory_index_entries_fts f
                        JOIN memory_index_entries i ON i.id = f.rowid
                        WHERE memory_index_entries_fts MATCH ?
                        ORDER BY bm25(memory_index_entries_fts), i.time_start DESC, i.id DESC
                        LIMIT ?
                        """,
                        (*params, match_query, candidate_limit * 3) if where else (match_query, candidate_limit * 3),
                    ).fetchall()
                    add_ids(rows, "fts")
                except sqlite3.Error:
                    pass

            like_clauses: List[str] = []
            like_params: List[Any] = []
            searchable_fields = (
                "title",
                "summary_for_retrieval",
                "keywords",
                "entities",
                "canonical_topics",
                "participants",
                "memory_path",
            )
            for term in normalized_terms[:8]:
                per_term = " OR ".join(f"LOWER({field}) LIKE ?" for field in searchable_fields)
                like_clauses.append(f"({per_term})")
                like_params.extend([f"%{term}%"] * len(searchable_fields))
            structured_where = list(clauses)
            structured_params = list(params)
            if like_clauses:
                structured_where.append("(" + " OR ".join(like_clauses) + ")")
                structured_params.extend(like_params)
                rows = self._conn.execute(
                    f"""
                    SELECT id FROM memory_index_entries
                    WHERE {" AND ".join(structured_where)}
                    ORDER BY time_start DESC, updated_at DESC, id DESC
                    LIMIT ?
                    """,
                    (*structured_params, candidate_limit * 3),
                ).fetchall()
                add_ids(rows, "like")

        rows = self._conn.execute(
            f"""
            SELECT id FROM memory_index_entries
            {where}
            ORDER BY time_start DESC, id DESC
            LIMIT ?
            """,
            (*params, candidate_limit),
        ).fetchall()
        add_ids(rows, "recent")
        if not candidate_ids:
            return []
        selected_ids = candidate_ids[: candidate_limit * 4]
        placeholders = ",".join("?" for _ in selected_ids)
        rows = self._conn.execute(
            f"SELECT * FROM memory_index_entries WHERE id IN ({placeholders})",
            selected_ids,
        ).fetchall()
        by_id = {int(row["id"]): self._row_to_dict(row) for row in rows}
        out: List[Dict[str, Any]] = []
        for row_id in selected_ids:
            item = by_id.get(row_id)
            if not item:
                continue
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            metadata = dict(metadata)
            metadata["_matched_via"] = candidate_sources.get(row_id, [])
            item["metadata"] = metadata
            out.append(item)
        return out

    def _search_memory_rows(
        self,
        *,
        table: str,
        searchable_fields: Sequence[str],
        time_field: str,
        terms: Optional[Sequence[str]],
        source_types: Optional[Sequence[str]],
        time_start: Optional[str],
        time_end: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Return raw rows for direct recall without using index cards.

        Table and field names are internal constants supplied by the three
        public wrappers below; user input is only ever bound as SQL values.
        Keyword hits are merged with recent rows so semantic reranking still
        has a chance to recover older records without scanning unbounded data.
        """
        base_clauses: List[str] = []
        base_params: List[Any] = []
        if source_types:
            placeholders = ",".join("?" for _ in source_types)
            base_clauses.append(f"source_type IN ({placeholders})")
            base_params.extend(source_types)
        time_expression = (
            f"substr({time_field}, 1, 19)"
            if time_field == "time_key"
            else time_field
        )
        time_clauses: List[str] = []
        time_params: List[Any] = []
        if time_start:
            time_clauses.append(f"{time_expression} >= ?")
            time_params.append(str(time_start))
        if time_end:
            time_clauses.append(f"{time_expression} <= ?")
            time_params.append(str(time_end))
        base_where = " AND ".join(base_clauses) if base_clauses else "1=1"
        timed_where = " AND ".join([base_where, *time_clauses]) if time_clauses else base_where
        row_limit = max(1, int(limit or 80))
        normalized_terms = self._normalize_search_terms(terms)
        row_ids: List[int] = []

        def add_ids(rows: Sequence[sqlite3.Row]) -> None:
            for row in rows:
                row_id = int(row["id"])
                if row_id not in row_ids:
                    row_ids.append(row_id)

        def add_keyword_matches(*, where: str, params: Sequence[Any], limit_value: int) -> None:
            if not normalized_terms:
                return
            match_parts: List[str] = []
            match_params: List[Any] = []
            for term in normalized_terms[:12]:
                per_term = " OR ".join(
                    f"LOWER(COALESCE({field}, '')) LIKE ?" for field in searchable_fields
                )
                match_parts.append(f"({per_term})")
                match_params.extend([f"%{term}%"] * len(searchable_fields))
            if match_parts:
                rows = self._conn.execute(
                    f"""
                    SELECT id FROM {table}
                    WHERE {where} AND ({" OR ".join(match_parts)})
                    ORDER BY {time_expression} DESC, id DESC
                    LIMIT ?
                    """,
                    (*params, *match_params, limit_value),
                ).fetchall()
                add_ids(rows)

        def add_recent(*, where: str, params: Sequence[Any], limit_value: int) -> None:
            rows = self._conn.execute(
                f"""
                SELECT id FROM {table}
                WHERE {where}
                ORDER BY {time_expression} DESC, id DESC
                LIMIT ?
                """,
                (*params, limit_value),
            ).fetchall()
            add_ids(rows)

        timed_params = [*base_params, *time_params]
        add_keyword_matches(where=timed_where, params=timed_params, limit_value=row_limit * 2)
        add_recent(where=timed_where, params=timed_params, limit_value=row_limit)
        if time_clauses and len(row_ids) < row_limit:
            # Time range is a strong preference, not a brittle hard stop. Pad
            # with broader keyword/recent candidates so downstream reranking
            # can still recover facts with coarse or slightly shifted times.
            add_keyword_matches(where=base_where, params=base_params, limit_value=row_limit * 2)
            add_recent(where=base_where, params=base_params, limit_value=row_limit)

        selected_ids = row_ids[: row_limit * 3]
        if not selected_ids:
            return []
        placeholders = ",".join("?" for _ in selected_ids)
        rows = self._conn.execute(
            f"SELECT * FROM {table} WHERE id IN ({placeholders})",
            selected_ids,
        ).fetchall()
        by_id = {int(row["id"]): self._row_to_dict(row) for row in rows}
        return [by_id[row_id] for row_id in selected_ids if row_id in by_id]

    def search_memory_facts(
        self,
        *,
        terms: Optional[Sequence[str]] = None,
        source_types: Optional[Sequence[str]] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        return self._search_memory_rows(
            table="memory_facts",
            searchable_fields=(
                "summary", "keywords", "entities", "entity_ids", "fact_root_topic",
                "fact_aspect_topic", "fact_kind", "fact_subject",
                "embedding_text", "metadata",
            ),
            time_field="time_key",
            terms=terms,
            source_types=source_types,
            time_start=time_start,
            time_end=time_end,
            limit=limit,
        )

    def search_memory_states(
        self,
        *,
        terms: Optional[Sequence[str]] = None,
        source_types: Optional[Sequence[str]] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        return self._search_memory_rows(
            table="memory_states",
            searchable_fields=(
                "canonical_name", "summary", "time_line", "entity_ids", "entity_key",
                "state_scope", "state_type", "identity_text", "metadata",
            ),
            time_field="updated_at",
            terms=terms,
            source_types=source_types,
            time_start=time_start,
            time_end=time_end,
            limit=limit,
        )

    def search_memory_actionable_items(
        self,
        *,
        terms: Optional[Sequence[str]] = None,
        source_types: Optional[Sequence[str]] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        return self._search_memory_rows(
            table="memory_actionable_items",
            searchable_fields=(
                "canonical_name", "summary", "item_type", "owner", "status",
                "due_at", "entity_ids", "embedding_text", "metadata",
            ),
            time_field="updated_at",
            terms=terms,
            source_types=source_types,
            time_start=time_start,
            time_end=time_end,
            limit=limit,
        )

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

    def memory_episodes_by_ids(self, episode_ids: Sequence[int]) -> List[Dict[str, Any]]:
        ids = [int(value) for value in episode_ids if value is not None]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT * FROM memory_episodes WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        by_id = {int(row["id"]): self._row_to_dict(row) for row in rows}
        return [by_id[item] for item in ids if item in by_id]

    def memory_states_by_ids(self, state_ids: Sequence[int]) -> List[Dict[str, Any]]:
        ids = [int(value) for value in state_ids if value is not None]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT * FROM memory_states WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        by_id = {int(row["id"]): self._row_to_dict(row) for row in rows}
        return [by_id[item] for item in ids if item in by_id]

    def memory_actionable_items_by_ids(
        self,
        item_ids: Sequence[int],
    ) -> List[Dict[str, Any]]:
        ids = [int(value) for value in item_ids if value is not None]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT * FROM memory_actionable_items WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        by_id = {int(row["id"]): self._row_to_dict(row) for row in rows}
        return [by_id[item] for item in ids if item in by_id]

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        item = dict(row)
        for key in (
            "entities",
            "entity_ids",
            "canonical_topics",
            "participants",
            "metadata",
            "evidence_fact_ids",
            "time_line",
        ):
            if key in item:
                item[key] = _json_loads(item[key], [] if key != "metadata" else {})
        for key in (
            "embedding",
            "identity_text_embedding",
            "canonical_name_embedding",
        ):
            if key in item:
                item[key] = _blob_to_embedding(item[key])
        return item
