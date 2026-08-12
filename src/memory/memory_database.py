#!/usr/bin/env python3
"""SQLite storage for the unified memory prototype.

The schema deliberately keeps a few legacy table names used by existing
benchmark scripts (`memory_facts`, `memory_observations`,
`memory_interpretations`, `memory_entity_nodes`) while adding the new unified line:

    memory_episodes -> memory_facts -> memory_states/actionable_items

`memory_index_entries` is the MemPalace-style directory layer: every retrievable
memory object writes one index card that points back to its source row.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

_HAS_FAISS = False
EMBEDDING_DIM = 384

_IDENTITY_FTS_TABLES = {
    "memory_facts": "memory_facts_identity_fts",
    "memory_states": "memory_states_identity_fts",
    "memory_actionable_items": "memory_actionable_items_identity_fts",
}


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
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._transaction_depth = 0
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def open_reader(self) -> "SessionDB":
        """Open a query-only connection without running schema initialization."""
        reader = object.__new__(SessionDB)
        reader.db_path = self.db_path
        reader._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=5.0,
        )
        reader._conn.row_factory = sqlite3.Row
        reader._transaction_depth = 0
        reader._conn.execute("PRAGMA query_only=ON")
        reader._conn.execute("PRAGMA foreign_keys=ON")
        reader._conn.execute("PRAGMA busy_timeout=5000")
        return reader

    @contextmanager
    def reader_transaction(self):
        """Read one committed SQLite snapshot and close its connection."""
        reader = self.open_reader()
        try:
            reader._conn.execute("BEGIN")
            yield reader
        finally:
            try:
                reader._conn.rollback()
            finally:
                reader.close()

    @contextmanager
    def transaction(self):
        """Group database mutations into one commit or rollback boundary."""
        is_outermost = self._transaction_depth == 0
        if is_outermost:
            self._conn.execute("BEGIN")
        self._transaction_depth += 1
        try:
            yield self
        except Exception:
            self._transaction_depth -= 1
            if is_outermost:
                self._conn.rollback()
            raise
        else:
            self._transaction_depth -= 1
            if is_outermost:
                self._conn.commit()

    def _commit_if_needed(self) -> None:
        """Commit standalone writes while deferring commits in a transaction."""
        if self._transaction_depth == 0:
            self._conn.commit()

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
                summary TEXT NOT NULL,
                keywords TEXT NOT NULL DEFAULT '',
                entities TEXT NOT NULL DEFAULT '[]',
                entity_ids TEXT NOT NULL DEFAULT '[]',
                fact_root_topic TEXT NOT NULL DEFAULT '',
                fact_aspect_topic TEXT NOT NULL DEFAULT '',
                event_time_key TEXT NOT NULL DEFAULT '',
                dialogue_time_key TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.85,
                importance REAL NOT NULL DEFAULT 0.5,
                processed_for_memory_state INTEGER NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}',
                identity_text_embedding BLOB,
                identity_text TEXT NOT NULL DEFAULT '',
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
                identity_text_embedding BLOB,
                identity_text TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(source_type, item_type, canonical_name)
            );

            CREATE TABLE IF NOT EXISTS memory_entity_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                type TEXT NOT NULL DEFAULT 'OTHER',
                created_at TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS memory_entity_mapping (
                entity_id INTEGER PRIMARY KEY,
                episode_id TEXT NOT NULL DEFAULT '[]',
                fact_id TEXT NOT NULL DEFAULT '[]',
                state_id TEXT NOT NULL DEFAULT '[]',
                actionable_item_id TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_memory_facts_event_time ON memory_facts(event_time_key);
            CREATE INDEX IF NOT EXISTS idx_memory_facts_dialogue_time ON memory_facts(dialogue_time_key);
            CREATE INDEX IF NOT EXISTS idx_memory_facts_source ON memory_facts(source_type);
            CREATE INDEX IF NOT EXISTS idx_memory_facts_state_processing
            ON memory_facts(processed_for_memory_state, created_at);
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
        self._init_identity_fts()
        self._commit_if_needed()

    def _init_identity_fts(self) -> None:
        """Create and backfill one BM25 index for each memory table."""
        for source_table, fts_table in _IDENTITY_FTS_TABLES.items():
            self._conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {fts_table} USING fts5(
                    identity_text
                )
                """
            )
            self._conn.execute(
                f"""
                INSERT INTO {fts_table} (rowid, identity_text)
                SELECT source.id, source.identity_text
                FROM {source_table} source
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM {fts_table} fts_row
                    WHERE fts_row.rowid = source.id
                )
                """
            )

    def _sync_identity_fts(
        self,
        *,
        source_table: str,
        row_id: int,
        identity_text: str,
    ) -> None:
        """Keep the table-specific BM25 document synchronized with its row."""
        fts_table = _IDENTITY_FTS_TABLES[source_table]
        self._conn.execute(
            f"DELETE FROM {fts_table} WHERE rowid = ?",
            (int(row_id),),
        )
        self._conn.execute(
            f"INSERT INTO {fts_table} (rowid, identity_text) VALUES (?, ?)",
            (int(row_id), str(identity_text or "")),
        )

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
        episode_id = int(cur.lastrowid)
        self.insert_entity_memory_mappings([
            {
                "entity_id": int(entity_id),
                "episode_id": [episode_id],
            }
            for entity_id in entity_ids or []
        ])
        self._commit_if_needed()
        return episode_id

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
        state_id = int(row["id"]) if row else 0
        if state_id:
            self._sync_identity_fts(
                source_table="memory_states",
                row_id=state_id,
                identity_text=str(identity_text or ""),
            )
            self.insert_entity_memory_mappings([
                {
                    "entity_id": int(entity_id),
                    "state_id": [state_id],
                }
                for entity_id in entity_ids or []
            ])
        self._commit_if_needed()
        return state_id

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
        identity_text_embedding: Optional[np.ndarray],
        identity_text: str,
    ) -> int:
        now = utc_now_text()
        self._conn.execute(
            """
            INSERT INTO memory_actionable_items (
                item_type, source_type, canonical_name, summary, owner,
                status, due_at, entity_ids, evidence_fact_ids, confidence, importance,
                metadata, identity_text_embedding, identity_text, created_at, updated_at
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
                identity_text_embedding = excluded.identity_text_embedding,
                identity_text = excluded.identity_text,
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
                _embedding_to_blob(identity_text_embedding),
                str(identity_text or ""),
                now,
                now,
            ),
        )
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
        actionable_item_id = int(row["id"]) if row else 0
        if actionable_item_id:
            self._sync_identity_fts(
                source_table="memory_actionable_items",
                row_id=actionable_item_id,
                identity_text=str(identity_text or ""),
            )
            self.insert_entity_memory_mappings([
                {
                    "entity_id": int(entity_id),
                    "actionable_item_id": [actionable_item_id],
                }
                for entity_id in entity_ids or []
            ])
        self._commit_if_needed()
        return actionable_item_id

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
            clauses.append("substr(dialogue_time_key, 1, 10) = ?")
            params.append(event_date)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn.execute(
            f"""
            SELECT * FROM memory_facts
            {where}
            ORDER BY replace(substr(dialogue_time_key, 1, 19), 'T', ' ') ASC, created_at ASC, id ASC
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
        self._commit_if_needed()
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
        summary: str,
        keywords: str,
        entities: Sequence[str],
        entity_ids: Optional[Sequence[int]],
        fact_root_topic: str,
        fact_aspect_topic: str,
        event_time_key: str,
        dialogue_time_key: str,
        confidence: float,
        importance: float,
        metadata: Optional[Dict[str, Any]],
        identity_text_embedding: Optional[np.ndarray],
        identity_text: str,
    ) -> int:
        now = utc_now_text()
        cur = self._conn.execute(
            """
            INSERT INTO memory_facts (
                episode_id, source_type, fact_type, fact_kind,
                summary, keywords, entities, entity_ids, fact_root_topic,
                fact_aspect_topic, event_time_key, dialogue_time_key,
                confidence, importance, metadata, identity_text_embedding, identity_text,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode_id,
                source_type,
                fact_type,
                fact_kind,
                summary,
                keywords,
                _json_dumps(list(entities or [])),
                _json_dumps([int(value) for value in entity_ids or []]),
                str(fact_root_topic or ""),
                str(fact_aspect_topic or ""),
                event_time_key,
                dialogue_time_key,
                float(confidence),
                float(importance),
                _json_dumps(metadata or {}),
                _embedding_to_blob(identity_text_embedding),
                identity_text,
                now,
                now,
            ),
        )
        fact_id = int(cur.lastrowid)
        self._sync_identity_fts(
            source_table="memory_facts",
            row_id=fact_id,
            identity_text=str(identity_text or ""),
        )
        self.insert_entity_memory_mappings([
            {
                "entity_id": int(entity_id),
                "episode_id": [episode_id] if episode_id is not None else [],
                "fact_id": [fact_id],
            }
            for entity_id in entity_ids or []
        ])
        self._commit_if_needed()
        return fact_id

    def add_entity_names(self, names: Iterable[str]) -> Dict[str, int]:
        now = utc_now_text()
        normalized: List[str] = []
        for name in names:
            clean = str(name or "").strip()
            if not clean or clean in normalized:
                continue
            normalized.append(clean)
            self._conn.execute(
                "INSERT OR IGNORE INTO memory_entity_nodes (name, type, created_at) VALUES (?, ?, ?)",
                (clean, "OTHER", now),
            )
        self._commit_if_needed()
        if not normalized:
            return {}
        placeholders = ",".join("?" for _ in normalized)
        rows = self._conn.execute(
            f"SELECT id, name FROM memory_entity_nodes WHERE name IN ({placeholders})",
            normalized,
        ).fetchall()
        return {str(row["name"]): int(row["id"]) for row in rows}

    def find_entity_nodes_in_text(
        self,
        text: str,
        *,
        limit: int = 12,
    ) -> List[Dict[str, Any]]:
        """Find stored entity names occurring verbatim in a query text."""
        clean_text = str(text or "").strip()
        if not clean_text:
            return []
        rows = self._conn.execute(
            """
            SELECT id, name
            FROM memory_entity_nodes
            WHERE length(name) >= 2
              AND instr(lower(?), lower(name)) > 0
            ORDER BY length(name) DESC, id ASC
            LIMIT ?
            """,
            (clean_text, max(1, int(limit or 12))),
        ).fetchall()
        return [dict(row) for row in rows]

    def memory_entity_mappings_by_entity_ids(
        self,
        entity_ids: Sequence[int],
    ) -> List[Dict[str, Any]]:
        """Load mapping rows for a bounded set of entity IDs."""
        ids = list(dict.fromkeys(
            int(value)
            for value in entity_ids
            if value is not None
        ))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"""
            SELECT entity_id, episode_id, fact_id, state_id, actionable_item_id
            FROM memory_entity_mapping
            WHERE entity_id IN ({placeholders})
            """,
            ids,
        ).fetchall()
        return [
            {
                "entity_id": int(row["entity_id"]),
                "episode_id": _json_loads(row["episode_id"], []),
                "fact_id": _json_loads(row["fact_id"], []),
                "state_id": _json_loads(row["state_id"], []),
                "actionable_item_id": _json_loads(row["actionable_item_id"], []),
            }
            for row in rows
        ]

    def insert_entity_memory_mappings(
        self,
        mappings: Sequence[Dict[str, Any]],
    ) -> int:
        """Merge episode, fact, and future memory links into one row per entity."""
        if not mappings:
            return 0

        mapping_fields = (
            "episode_id",
            "fact_id",
            "state_id",
            "actionable_item_id",
        )

        def normalize_ids(value: Any) -> List[int]:
            if isinstance(value, str):
                parsed = _json_loads(value, None)
                values = parsed if isinstance(parsed, list) else [value]
            else:
                values = value if isinstance(value, (list, tuple, set)) else [value]
            normalized: List[int] = []
            for item in values:
                if item in (None, ""):
                    continue
                try:
                    item_id = int(item)
                except (TypeError, ValueError):
                    continue
                if item_id not in normalized:
                    normalized.append(item_id)
            return normalized

        grouped: Dict[int, Dict[str, List[int]]] = {}
        for mapping in mappings:
            try:
                entity_id = int(mapping["entity_id"])
            except (KeyError, TypeError, ValueError):
                continue
            entity_mapping = grouped.setdefault(
                entity_id,
                {field: [] for field in mapping_fields},
            )
            for field in mapping_fields:
                for item_id in normalize_ids(mapping.get(field)):
                    if item_id not in entity_mapping[field]:
                        entity_mapping[field].append(item_id)

        if not grouped:
            return 0

        now = utc_now_text()
        changed_count = 0
        for entity_id, mapping in grouped.items():
            existing = self._conn.execute(
                """
                SELECT episode_id, fact_id, state_id, actionable_item_id
                FROM memory_entity_mapping
                WHERE entity_id = ?
                """,
                (entity_id,),
            ).fetchone()
            merged = dict(mapping)
            if existing:
                for field in mapping_fields:
                    previous_ids = normalize_ids(existing[field])
                    merged[field] = previous_ids + [
                        item_id
                        for item_id in mapping[field]
                        if item_id not in previous_ids
                    ]
                self._conn.execute(
                    """
                    UPDATE memory_entity_mapping
                    SET episode_id = ?, fact_id = ?, state_id = ?,
                        actionable_item_id = ?, updated_at = ?
                    WHERE entity_id = ?
                    """,
                    (
                        _json_dumps(merged["episode_id"]),
                        _json_dumps(merged["fact_id"]),
                        _json_dumps(merged["state_id"]),
                        _json_dumps(merged["actionable_item_id"]),
                        now,
                        entity_id,
                    ),
                )
            else:
                self._conn.execute(
                    """
                    INSERT INTO memory_entity_mapping (
                        entity_id, episode_id, fact_id, state_id,
                        actionable_item_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entity_id,
                        _json_dumps(merged["episode_id"]),
                        _json_dumps(merged["fact_id"]),
                        _json_dumps(merged["state_id"]),
                        _json_dumps(merged["actionable_item_id"]),
                        now,
                        now,
                    ),
                )
            changed_count += 1
        self._commit_if_needed()
        return changed_count

    def _search_memory_rows(
        self,
        *,
        table: str,
        identity_fts_table: str,
        time_fields: Optional[Sequence[str]] = None,
        terms: Optional[Sequence[str]],
        source_types: Optional[Sequence[str]],
        time_start: Optional[str],
        time_end: Optional[str],
        limit: int,
        strict_time_filter: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return raw rows ranked by BM25 over their identity text.

        Table and field names are internal constants supplied by the three
        public wrappers below; user input is only ever bound as SQL values.
        BM25 hits are merged with recent rows so semantic reranking still has
        a chance to recover older records without scanning unbounded data.
        """
        base_clauses: List[str] = []
        base_params: List[Any] = []
        if source_types:
            placeholders = ",".join("?" for _ in source_types)
            base_clauses.append(f"source_type IN ({placeholders})")
            base_params.extend(source_types)
        selected_time_fields = [
            str(field).strip()
            for field in (time_fields or [])
            if str(field).strip()
        ]
        time_expressions = [
            f"substr({field}, 1, 19)"
            if field.endswith("_time_key")
            else field
            for field in selected_time_fields
        ]
        time_expression = (
            time_expressions[0]
            if len(time_expressions) == 1
            else "updated_at"
            if not time_expressions
            else "COALESCE(" + ", ".join(time_expressions) + ")"
        )
        time_clauses: List[str] = []
        time_params: List[Any] = []
        for field_expression in time_expressions:
            field_clauses: List[str] = []
            field_params: List[str] = []
            if time_start:
                field_clauses.append(f"{field_expression} >= ?")
                field_params.append(str(time_start))
            if time_end:
                field_clauses.append(f"{field_expression} <= ?")
                field_params.append(str(time_end))
            if field_clauses:
                time_clauses.append("(" + " AND ".join(field_clauses) + ")")
                time_params.extend(field_params)
        if len(time_clauses) > 1:
            time_filter = "(" + " OR ".join(time_clauses) + ")"
            time_clauses = [time_filter]
        base_where = " AND ".join(base_clauses) if base_clauses else "1=1"
        timed_where = " AND ".join([base_where, *time_clauses]) if time_clauses else base_where
        row_limit = int(limit)
        if row_limit <= 0:
            return []
        normalized_terms = self._normalize_search_terms(terms)
        row_ids: List[int] = []
        bm25_scores: Dict[int, float] = {}

        def add_ids(rows: Sequence[sqlite3.Row]) -> None:
            for row in rows:
                row_id = int(row["id"])
                if row_id not in row_ids:
                    row_ids.append(row_id)
                if "bm25_score" in row.keys():
                    try:
                        bm25_scores[row_id] = float(row["bm25_score"])
                    except (TypeError, ValueError):
                        pass

        def add_bm25_matches(
            *,
            where: str,
            params: Sequence[Any],
            limit_value: int,
        ) -> bool:
            if not normalized_terms:
                return True
            match_query = self._terms_to_fts_query(normalized_terms)
            if not match_query:
                return True
            try:
                rows = self._conn.execute(
                    f"""
                    SELECT source.id, bm25({identity_fts_table}) AS bm25_score
                    FROM {identity_fts_table}
                    JOIN {table} source ON source.id = {identity_fts_table}.rowid
                    WHERE {where} AND {identity_fts_table} MATCH ?
                    ORDER BY bm25({identity_fts_table}) ASC,
                             {time_expression} DESC, source.id DESC
                    LIMIT ?
                    """,
                    (*params, match_query, limit_value),
                ).fetchall()
            except sqlite3.Error:
                return False
            add_ids(rows)
            return True

        def add_identity_like_matches(
            *,
            where: str,
            params: Sequence[Any],
            limit_value: int,
        ) -> None:
            """Fallback for SQLite builds without FTS5 support."""
            if not normalized_terms:
                return
            like_clauses = [
                "LOWER(COALESCE(source.identity_text, '')) LIKE ?"
                for _term in normalized_terms[:12]
            ]
            if not like_clauses:
                return
            rows = self._conn.execute(
                f"""
                SELECT source.id
                FROM {table} source
                WHERE {where} AND ({" OR ".join(like_clauses)})
                ORDER BY {time_expression} DESC, source.id DESC
                LIMIT ?
                """,
                (*params, *[f"%{term}%" for term in normalized_terms[:12]], limit_value),
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
        bm25_available = add_bm25_matches(
            where=timed_where,
            params=timed_params,
            limit_value=row_limit * 2,
        )
        if not bm25_available:
            add_identity_like_matches(
                where=timed_where,
                params=timed_params,
                limit_value=row_limit * 2,
            )
        add_recent(where=timed_where, params=timed_params, limit_value=row_limit)
        if time_clauses and len(row_ids) < row_limit and not strict_time_filter:
            # Time range is a strong preference, not a brittle hard stop. Pad
            # with broader keyword/recent candidates so downstream reranking
            # can still recover facts with coarse or slightly shifted times.
            if bm25_available:
                add_bm25_matches(
                    where=base_where,
                    params=base_params,
                    limit_value=row_limit * 2,
                )
            else:
                add_identity_like_matches(
                    where=base_where,
                    params=base_params,
                    limit_value=row_limit * 2,
                )
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
        out: List[Dict[str, Any]] = []
        for row_id in selected_ids:
            item = by_id.get(row_id)
            if not item:
                continue
            if row_id in bm25_scores:
                item["_bm25_score"] = bm25_scores[row_id]
            out.append(item)
        return out

    def search_memory_facts(
        self,
        *,
        terms: Optional[Sequence[str]] = None,
        source_types: Optional[Sequence[str]] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        temporal_mode: str = "dialogue_time",
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        temporal_mode = str(temporal_mode or "dialogue_time").strip().lower()
        if temporal_mode == "event_time":
            time_fields = ["event_time_key"]
        elif temporal_mode == "both":
            time_fields = ["event_time_key", "dialogue_time_key"]
        elif temporal_mode == "none":
            time_fields = []
        else:
            time_fields = ["dialogue_time_key"]
        return self._search_memory_rows(
            table="memory_facts",
            identity_fts_table="memory_facts_identity_fts",
            time_fields=time_fields or ["dialogue_time_key"],
            terms=terms,
            source_types=source_types,
            time_start=time_start if temporal_mode != "none" else None,
            time_end=time_end if temporal_mode != "none" else None,
            limit=limit,
            strict_time_filter=True,
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
            identity_fts_table="memory_states_identity_fts",
            time_fields=[],
            terms=terms,
            source_types=source_types,
            time_start=None,
            time_end=None,
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
            identity_fts_table="memory_actionable_items_identity_fts",
            time_fields=[],
            terms=terms,
            source_types=source_types,
            time_start=None,
            time_end=None,
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

    def memory_facts_by_episode_ids(
        self,
        episode_ids: Sequence[int],
        *,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Return facts belonging to a bounded set of episodes."""
        ids = list(dict.fromkeys(
            int(value)
            for value in episode_ids
            if value is not None
        ))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"""
            SELECT * FROM memory_facts
            WHERE episode_id IN ({placeholders})
            ORDER BY dialogue_time_key ASC, id ASC
            LIMIT ?
            """,
            (*ids, max(1, int(limit or 200))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

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
