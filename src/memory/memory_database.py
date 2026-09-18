#!/usr/bin/env python3
"""SQLite storage for the unified memory prototype.

The schema deliberately keeps a few legacy table names used by existing
benchmark scripts (`memory_facts`, `memory_observations`,
`memory_interpretations`, `memory_entity_nodes`) while adding the new unified line:

    memory_episodes -> memory_facts -> entity_claims / intent-execution

`memory_index_entries` is the MemPalace-style directory layer: every retrievable
memory object writes one index card that points back to its source row.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

try:
    import jieba
except ImportError:  # pragma: no cover - exercised only in minimal installs
    jieba = None

_IDENTITY_FTS_TABLES = {
    "memory_facts": "memory_facts_identity_fts",
}

_LEXICAL_DATE_PATTERNS = (
    re.compile(r"(?<!\d)\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}(?:日)?(?!\d)"),
    re.compile(r"(?<!\d)\d{1,2}月\d{1,2}日?(?!\d)"),
)


def _lexical_index_text(identity_text: Any) -> str:
    """Convert display identity text into deterministic FTS search tokens.

    The source tables keep the original ``identity_text`` for embeddings and
    display. FTS receives a separate token stream so Chinese text can be
    searched by words instead of relying on SQLite's default tokenizer.
    Dates are excluded here because fact time is filtered through structured
    time columns rather than lexical coincidence.
    """
    text_lines: List[str] = []
    for line in str(identity_text or "").splitlines():
        clean_line = line.strip()
        colon_positions = [
            position
            for position in (clean_line.find(":"), clean_line.find("："))
            if position >= 0
        ]
        if colon_positions:
            # identity_text is formatted as one ``field: value`` per line.
            # Index only the value so labels such as ``summary`` and
            # ``entities`` do not become searchable memory content.
            clean_line = clean_line[min(colon_positions) + 1 :].strip()
        if clean_line:
            text_lines.append(clean_line)
    text = "\n".join(text_lines)
    for pattern in _LEXICAL_DATE_PATTERNS:
        text = pattern.sub(" ", text)
    if not text.strip():
        return ""

    if jieba is not None:
        # Search mode keeps useful sub-tokens for Chinese compounds while the
        # regular cut preserves the complete domain phrase when available.
        raw_tokens = [
            *jieba.lcut(text, HMM=False),
            *jieba.cut_for_search(text, HMM=False),
        ]
    else:
        # Keep the database usable in minimal environments. This fallback is
        # deliberately simple; production installs should include jieba.
        raw_tokens = []
        for match in re.findall(
            r"[A-Za-z][A-Za-z0-9_.$'-]*|\d+(?:/\d+)?|[\u4e00-\u9fff]+",
            text,
        ):
            if re.fullmatch(r"[\u4e00-\u9fff]+", match):
                raw_tokens.append(match)
                raw_tokens.extend(match)
                raw_tokens.extend(
                    match[index : index + 2]
                    for index in range(len(match) - 1)
                )
            else:
                raw_tokens.append(match)

    tokens: List[str] = []
    seen: set[str] = set()
    for raw_token in raw_tokens:
        token = re.sub(r"\s+", "", str(raw_token or "")).strip()
        if not token or re.fullmatch(r"[^\w\u4e00-\u9fff]+", token):
            continue
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return " ".join(tokens)


def local_now_text() -> str:
    return datetime.now().astimezone().isoformat()


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
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=30.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._transaction_depth = 0
        self._conn.execute("PRAGMA busy_timeout=30000")
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
            timeout=30.0,
        )
        reader._conn.row_factory = sqlite3.Row
        reader._transaction_depth = 0
        reader._conn.execute("PRAGMA query_only=ON")
        reader._conn.execute("PRAGMA foreign_keys=ON")
        reader._conn.execute("PRAGMA busy_timeout=30000")
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
                keywords TEXT NOT NULL DEFAULT '[]',
                entities TEXT NOT NULL DEFAULT '[]',
                entity_ids TEXT NOT NULL DEFAULT '[]',
                fact_root_topic TEXT NOT NULL DEFAULT '',
                fact_aspect_topic TEXT NOT NULL DEFAULT '',
                event_time_key TEXT NOT NULL DEFAULT '',
                dialogue_time_key TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.85,
                importance REAL NOT NULL DEFAULT 0.5,
                processed_for_memory_entity_claim INTEGER NOT NULL DEFAULT 0,
                processed_for_memory_entity_claim_induction INTEGER NOT NULL DEFAULT 0,
                processed_for_memory_intent_execution INTEGER NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}',
                identity_text_embedding BLOB,
                identity_text TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(episode_id) REFERENCES memory_episodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_entity_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                type TEXT NOT NULL DEFAULT 'OTHER',
                created_at TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS memory_fact_entity_claim_signal_mapping (
                fact_id INTEGER NOT NULL,
                subject_entity_id INTEGER NOT NULL,
                claim_type_hint TEXT NOT NULL,
                signal_kind TEXT NOT NULL,
                claim_anchor TEXT NOT NULL,
                claim_anchor_key TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(
                    fact_id, subject_entity_id, claim_type_hint, claim_anchor_key
                ),
                FOREIGN KEY(fact_id) REFERENCES memory_facts(id) ON DELETE CASCADE,
                FOREIGN KEY(subject_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_entity_mapping (
                entity_id INTEGER PRIMARY KEY,
                episode_id TEXT NOT NULL DEFAULT '[]',
                fact_id TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_fact_episode_mapping (
                fact_id INTEGER NOT NULL,
                episode_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(fact_id, episode_id),
                FOREIGN KEY(fact_id) REFERENCES memory_facts(id) ON DELETE CASCADE,
                FOREIGN KEY(episode_id) REFERENCES memory_episodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_topic_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_kind TEXT NOT NULL,
                topic_name TEXT NOT NULL,
                topic_key TEXT NOT NULL,
                fact_occurrence_count INTEGER NOT NULL DEFAULT 0,
                episode_occurrence_count INTEGER NOT NULL DEFAULT 0,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(topic_kind, topic_key)
            );

            CREATE TABLE IF NOT EXISTS memory_topic_mapping (
                topic_item_id INTEGER PRIMARY KEY,
                fact_ids TEXT NOT NULL DEFAULT '[]',
                episode_ids TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(topic_item_id) REFERENCES memory_topic_items(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_entity_claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_entity_id INTEGER NOT NULL,
                predicate TEXT NOT NULL,
                object_entity_id INTEGER NOT NULL DEFAULT 0,
                claim_text TEXT NOT NULL DEFAULT '',
                claim_type TEXT NOT NULL,
                claim_origin TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'candidate',
                confidence REAL NOT NULL DEFAULT 0.7,
                valid_from TEXT NOT NULL DEFAULT '',
                valid_to TEXT NOT NULL DEFAULT '',
                source_actor_entity_id INTEGER,
                extractor_version TEXT NOT NULL DEFAULT '',
                prompt_version TEXT NOT NULL DEFAULT '',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(subject_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE,
                FOREIGN KEY(source_actor_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS memory_entity_claim_evidence (
                claim_id INTEGER NOT NULL,
                evidence_type TEXT NOT NULL,
                evidence_id INTEGER NOT NULL,
                role TEXT NOT NULL DEFAULT 'support',
                weight REAL NOT NULL DEFAULT 1.0,
                observed_at TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(claim_id, evidence_type, evidence_id, role),
                FOREIGN KEY(claim_id) REFERENCES memory_entity_claims(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_entity_claim_induction (
                claim_id INTEGER PRIMARY KEY,
                condition_text TEXT NOT NULL DEFAULT '',
                support_count INTEGER NOT NULL DEFAULT 0,
                first_observed_at TEXT NOT NULL DEFAULT '',
                last_observed_at TEXT NOT NULL DEFAULT '',
                consolidation_version TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(claim_id) REFERENCES memory_entity_claims(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_entity_claim_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_claim_id INTEGER NOT NULL,
                trigger_claim_id INTEGER,
                event_type TEXT NOT NULL DEFAULT 'status_transition',
                previous_status TEXT NOT NULL,
                new_status TEXT NOT NULL,
                semantic_relation TEXT NOT NULL DEFAULT '',
                effective_at TEXT NOT NULL DEFAULT '',
                decision_source TEXT NOT NULL DEFAULT '',
                semantic_confidence REAL,
                semantic_reason TEXT NOT NULL DEFAULT '',
                policy_reason TEXT NOT NULL DEFAULT '',
                details TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(target_claim_id) REFERENCES memory_entity_claims(id) ON DELETE CASCADE,
                FOREIGN KEY(trigger_claim_id) REFERENCES memory_entity_claims(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS memory_goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                world_owner_entity_id INTEGER NOT NULL,
                owner_entity_id INTEGER NOT NULL,
                canonical_key TEXT NOT NULL,
                summary TEXT NOT NULL,
                desired_outcome TEXT NOT NULL DEFAULT '',
                success_criteria TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                target_at TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.7,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(world_owner_entity_id, owner_entity_id, canonical_key),
                FOREIGN KEY(world_owner_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE,
                FOREIGN KEY(owner_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                world_owner_entity_id INTEGER NOT NULL,
                actor_entity_id INTEGER NOT NULL,
                canonical_key TEXT NOT NULL,
                summary TEXT NOT NULL,
                event_or_activity TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'planned',
                start_at TEXT NOT NULL DEFAULT '',
                end_at TEXT NOT NULL DEFAULT '',
                time_precision TEXT NOT NULL DEFAULT 'unknown',
                location_entity_id INTEGER,
                location_text TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.7,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(world_owner_entity_id, actor_entity_id, canonical_key),
                FOREIGN KEY(world_owner_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE,
                FOREIGN KEY(actor_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE,
                FOREIGN KEY(location_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS memory_plan_entities (
                plan_id INTEGER NOT NULL,
                entity_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(plan_id, entity_id, role),
                FOREIGN KEY(plan_id) REFERENCES memory_plans(id) ON DELETE CASCADE,
                FOREIGN KEY(entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_work_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                world_owner_entity_id INTEGER NOT NULL,
                responsible_entity_id INTEGER NOT NULL,
                canonical_key TEXT NOT NULL,
                summary TEXT NOT NULL,
                action_text TEXT NOT NULL DEFAULT '',
                deliverable TEXT NOT NULL DEFAULT '',
                responsibility_type TEXT NOT NULL DEFAULT 'personal_action',
                status TEXT NOT NULL DEFAULT 'open',
                due_at TEXT NOT NULL DEFAULT '',
                start_at TEXT NOT NULL DEFAULT '',
                priority TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.7,
                completed_at TEXT NOT NULL DEFAULT '',
                extractor_version TEXT NOT NULL DEFAULT '',
                prompt_version TEXT NOT NULL DEFAULT '',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(world_owner_entity_id, responsible_entity_id, canonical_key),
                FOREIGN KEY(world_owner_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE,
                FOREIGN KEY(responsible_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_work_item_entities (
                work_item_id INTEGER NOT NULL,
                entity_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(work_item_id, entity_id, role),
                FOREIGN KEY(work_item_id) REFERENCES memory_work_items(id) ON DELETE CASCADE,
                FOREIGN KEY(entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_intent_evidence (
                object_type TEXT NOT NULL,
                object_id INTEGER NOT NULL,
                evidence_type TEXT NOT NULL,
                evidence_id INTEGER NOT NULL,
                role TEXT NOT NULL DEFAULT 'support',
                observed_at TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(object_type, object_id, evidence_type, evidence_id, role)
            );

            CREATE TABLE IF NOT EXISTS memory_intent_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_type TEXT NOT NULL,
                object_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                previous_status TEXT NOT NULL DEFAULT '',
                new_status TEXT NOT NULL DEFAULT '',
                previous_payload TEXT NOT NULL DEFAULT '{}',
                new_payload TEXT NOT NULL DEFAULT '{}',
                evidence_fact_ids TEXT NOT NULL DEFAULT '[]',
                effective_at TEXT NOT NULL DEFAULT '',
                decision_source TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_goal_work_item_mappings (
                goal_id INTEGER NOT NULL,
                work_item_id INTEGER NOT NULL,
                relation TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(goal_id, work_item_id, relation),
                FOREIGN KEY(goal_id) REFERENCES memory_goals(id) ON DELETE CASCADE,
                FOREIGN KEY(work_item_id) REFERENCES memory_work_items(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_plan_work_item_mappings (
                plan_id INTEGER NOT NULL,
                work_item_id INTEGER NOT NULL,
                relation TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(plan_id, work_item_id, relation),
                FOREIGN KEY(plan_id) REFERENCES memory_plans(id) ON DELETE CASCADE,
                FOREIGN KEY(work_item_id) REFERENCES memory_work_items(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_work_item_relations (
                source_work_item_id INTEGER NOT NULL,
                target_work_item_id INTEGER NOT NULL,
                relation TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(source_work_item_id, target_work_item_id, relation),
                FOREIGN KEY(source_work_item_id) REFERENCES memory_work_items(id) ON DELETE CASCADE,
                FOREIGN KEY(target_work_item_id) REFERENCES memory_work_items(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_memory_facts_event_time ON memory_facts(event_time_key);
            CREATE INDEX IF NOT EXISTS idx_memory_facts_dialogue_time ON memory_facts(dialogue_time_key);
            CREATE INDEX IF NOT EXISTS idx_memory_facts_source ON memory_facts(source_type);
            CREATE INDEX IF NOT EXISTS idx_memory_fact_episode_episode
            ON memory_fact_episode_mapping(episode_id, updated_at);
            CREATE INDEX IF NOT EXISTS idx_memory_topic_items_kind_seen
            ON memory_topic_items(topic_kind, last_seen_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_entity_claims_subject
            ON memory_entity_claims(subject_entity_id, claim_type, claim_origin, status);
            CREATE INDEX IF NOT EXISTS idx_memory_entity_claim_evidence_claim
            ON memory_entity_claim_evidence(claim_id, role, evidence_type);
            CREATE INDEX IF NOT EXISTS idx_memory_fact_entity_claim_signal_group
            ON memory_fact_entity_claim_signal_mapping(
                subject_entity_id, claim_type_hint, claim_anchor_key, fact_id
            );
            CREATE INDEX IF NOT EXISTS idx_memory_entity_claim_events_target
            ON memory_entity_claim_events(target_claim_id, effective_at DESC, id DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_entity_claim_events_trigger
            ON memory_entity_claim_events(trigger_claim_id, id DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_facts_intent_execution_processing
            ON memory_facts(processed_for_memory_intent_execution, created_at);
            CREATE INDEX IF NOT EXISTS idx_memory_goals_owner_status
            ON memory_goals(world_owner_entity_id, owner_entity_id, status);
            CREATE INDEX IF NOT EXISTS idx_memory_plans_actor_status_time
            ON memory_plans(world_owner_entity_id, actor_entity_id, status, start_at);
            CREATE INDEX IF NOT EXISTS idx_memory_work_items_responsible_status_due
            ON memory_work_items(world_owner_entity_id, responsible_entity_id, status, due_at);
            CREATE INDEX IF NOT EXISTS idx_memory_intent_evidence_object
            ON memory_intent_evidence(object_type, object_id, role);
            CREATE INDEX IF NOT EXISTS idx_memory_intent_events_object
            ON memory_intent_events(object_type, object_id, effective_at DESC, id DESC);
            """
        )
        self._ensure_entity_ids_schema()
        self._ensure_memory_entity_claim_processing_schema()
        self._ensure_memory_intent_execution_processing_schema()
        self._ensure_memory_entity_claims_schema()
        self._ensure_memory_entity_claim_induction_schema()
        self._backfill_fact_episode_mappings()
        self._init_identity_fts()
        self._commit_if_needed()
    
    def _init_identity_fts(self) -> None:
        """Create and backfill one tokenized BM25 index per memory table."""
        for source_table, fts_table in _IDENTITY_FTS_TABLES.items():
            existing_columns = {
                str(row["name"])
                for row in self._conn.execute(
                    f"PRAGMA table_info({fts_table})"
                ).fetchall()
            }
            if existing_columns and "lexical_index_text" not in existing_columns:
                # FTS is a derived index. Rebuilding it is safe and keeps
                # existing memory rows untouched when the index format changes.
                self._conn.execute(f"DROP TABLE IF EXISTS {fts_table}")
            self._conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {fts_table} USING fts5(
                    lexical_index_text
                )
                """
            )
            existing_ids = {
                int(row["rowid"])
                for row in self._conn.execute(
                    f"SELECT rowid FROM {fts_table}"
                ).fetchall()
            }
            source_rows = self._conn.execute(
                f"SELECT id, identity_text FROM {source_table}"
            ).fetchall()
            for row in source_rows:
                row_id = int(row["id"])
                if row_id in existing_ids:
                    continue
                self._conn.execute(
                    f"INSERT INTO {fts_table} (rowid, lexical_index_text) VALUES (?, ?)",
                    (row_id, _lexical_index_text(row["identity_text"])),
                )

    def _sync_identity_fts(
        self,
        *,
        source_table: str,
        row_id: int,
        identity_text: str,
    ) -> None:
        """Keep the tokenized BM25 document synchronized with its source row."""
        fts_table = _IDENTITY_FTS_TABLES[source_table]
        self._conn.execute(
            f"DELETE FROM {fts_table} WHERE rowid = ?",
            (int(row_id),),
        )
        self._conn.execute(
            f"INSERT INTO {fts_table} (rowid, lexical_index_text) VALUES (?, ?)",
            (int(row_id), _lexical_index_text(identity_text)),
        )

    def _ensure_entity_ids_schema(self) -> None:
        """Ensure all primary memory tables expose direct entity id mappings."""
        for table in (
            "memory_episodes",
            "memory_facts",
        ):
            columns = {
                str(row["name"])
                for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if "entity_ids" not in columns:
                self._conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN entity_ids TEXT NOT NULL DEFAULT '[]'"
                )

    def _ensure_memory_entity_claim_processing_schema(self) -> None:
        """Add independent reflect cursors for the claim projections."""
        fact_columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(memory_facts)").fetchall()
        }
        if "processed_for_memory_entity_claim" not in fact_columns:
            self._conn.execute(
                "ALTER TABLE memory_facts ADD COLUMN "
                "processed_for_memory_entity_claim INTEGER NOT NULL DEFAULT 0"
            )
        if "processed_for_memory_entity_claim_induction" not in fact_columns:
            self._conn.execute(
                "ALTER TABLE memory_facts ADD COLUMN "
                "processed_for_memory_entity_claim_induction INTEGER NOT NULL DEFAULT 0"
            )
        episode_columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(memory_episodes)").fetchall()
        }
        if "processed_for_memory_entity_claim_induction" in episode_columns:
            self._conn.execute(
                "DROP INDEX IF EXISTS idx_memory_episodes_entity_claim_induction"
            )
            self._conn.execute(
                "ALTER TABLE memory_episodes DROP COLUMN "
                "processed_for_memory_entity_claim_induction"
            )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_facts_entity_claim_processing "
            "ON memory_facts(processed_for_memory_entity_claim, created_at)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_facts_entity_claim_induction "
            "ON memory_facts(processed_for_memory_entity_claim_induction, created_at)"
        )

    def _ensure_memory_intent_execution_processing_schema(self) -> None:
        """Give Intent & Execution its own fact cursor during schema upgrades."""
        columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(memory_facts)").fetchall()
        }
        if "processed_for_memory_intent_execution" not in columns:
            self._conn.execute(
                "ALTER TABLE memory_facts ADD COLUMN "
                "processed_for_memory_intent_execution INTEGER NOT NULL DEFAULT 0"
            )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_facts_intent_execution_processing "
            "ON memory_facts(processed_for_memory_intent_execution, created_at)"
        )

    def _ensure_memory_entity_claims_schema(self) -> None:
        """Migrate legacy claim rows away from the retired normalized value."""
        columns = {
            str(row["name"])
            for row in self._conn.execute(
                "PRAGMA table_info(memory_entity_claims)"
            ).fetchall()
        }
        if "normalized_value" not in columns:
            return
        if "claim_text" not in columns:
            self._conn.execute(
                "ALTER TABLE memory_entity_claims "
                "ADD COLUMN claim_text TEXT NOT NULL DEFAULT ''"
            )
        self._rebuild_memory_entity_claims_without_normalized_value()

    def _ensure_memory_entity_claim_induction_schema(self) -> None:
        """Remove retired induction fields while preserving its support history."""
        columns = {
            str(row["name"])
            for row in self._conn.execute(
                "PRAGMA table_info(memory_entity_claim_induction)"
            ).fetchall()
        }
        for column in ("behavior_or_outcome_text", "counterexample_count"):
            if column in columns:
                self._conn.execute(
                    "ALTER TABLE memory_entity_claim_induction "
                    f"DROP COLUMN {column}"
                )

    def _rebuild_memory_entity_claims_without_normalized_value(self) -> None:
        """Drop the legacy key while preserving claim IDs and dependents."""
        self._conn.commit()
        self._conn.execute("PRAGMA foreign_keys = OFF")
        try:
            self._conn.executescript(
                """
                CREATE TABLE memory_entity_claims_rebuilt (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject_entity_id INTEGER NOT NULL,
                    predicate TEXT NOT NULL,
                    object_entity_id INTEGER NOT NULL DEFAULT 0,
                    claim_text TEXT NOT NULL DEFAULT '',
                    claim_type TEXT NOT NULL,
                    claim_origin TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'candidate',
                    confidence REAL NOT NULL DEFAULT 0.7,
                    valid_from TEXT NOT NULL DEFAULT '',
                    valid_to TEXT NOT NULL DEFAULT '',
                    source_actor_entity_id INTEGER,
                    extractor_version TEXT NOT NULL DEFAULT '',
                    prompt_version TEXT NOT NULL DEFAULT '',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(subject_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY(source_actor_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE SET NULL
                );

                INSERT INTO memory_entity_claims_rebuilt (
                    id, subject_entity_id, predicate, object_entity_id, claim_text,
                    claim_type, claim_origin, status, confidence, valid_from, valid_to,
                    source_actor_entity_id, extractor_version, prompt_version, metadata,
                    created_at, updated_at
                )
                SELECT id, subject_entity_id, predicate, object_entity_id, claim_text,
                       claim_type, claim_origin, status, confidence, valid_from, valid_to,
                       source_actor_entity_id, extractor_version, prompt_version, metadata,
                       created_at, updated_at
                FROM memory_entity_claims;

                DROP TABLE memory_entity_claims;
                ALTER TABLE memory_entity_claims_rebuilt RENAME TO memory_entity_claims;
                CREATE INDEX IF NOT EXISTS idx_memory_entity_claims_subject
                ON memory_entity_claims(subject_entity_id, claim_type, claim_origin, status);
                """
            )
        finally:
            self._conn.execute("PRAGMA foreign_keys = ON")

    def _backfill_fact_episode_mappings(self) -> None:
        """Mirror legacy fact episode references into the relation table."""
        episode_rows = self._conn.execute(
            "SELECT id, episode_id FROM memory_facts WHERE episode_id IS NOT NULL"
        ).fetchall()
        self.insert_fact_episode_mappings([
            {
                "fact_id": int(row["id"]),
                "episode_id": int(row["episode_id"]),
            }
            for row in episode_rows
            if row["episode_id"] is not None
        ])

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
        """Normalize lexical terms that were already tokenized upstream.

        Query tokenization belongs to ``MemoryNodeManager`` so recall and
        reflect use one consistent lexical policy. The database layer only
        deduplicates whitespace-normalized terms before building the FTS
        expression; jieba remains part of index-text construction above.
        """
        normalized: List[str] = []
        for term in terms or []:
            clean = re.sub(r"\s+", " ", str(term or "").strip()).lower()
            if clean and clean not in normalized:
                normalized.append(clean)
            if len(normalized) >= 32:
                return normalized
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
        now = local_now_text()
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

    def upsert_memory_topic_items(
        self,
        items: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Upsert topic registry items and merge their fact/episode IDs.

        ``memory_topic_items`` holds one reusable topic name per
        ``(topic_kind, topic_key)``. Its companion mapping row deliberately
        stores the aggregated evidence IDs as JSON arrays, matching the
        compact topic-directory model used by this project.
        """
        grouped: Dict[str, Dict[str, Any]] = {}

        def normalize_ids(value: Any) -> List[int]:
            values = value if isinstance(value, (list, tuple, set)) else [value]
            normalized: List[int] = []
            for raw in values:
                try:
                    item_id = int(raw)
                except (TypeError, ValueError):
                    continue
                if item_id > 0 and item_id not in normalized:
                    normalized.append(item_id)
            return normalized

        for item in items or []:
            if not isinstance(item, dict):
                continue
            topic_kind = str(item.get("topic_kind") or "").strip().lower()
            topic_name = str(item.get("topic_name") or "").strip()
            topic_key = str(item.get("topic_key") or "").strip().lower()
            if topic_kind not in {"canonical", "aspect"} or not topic_name or not topic_key:
                continue
            group_key = f"{topic_kind}\x1f{topic_key}"
            grouped_item = grouped.setdefault(
                group_key,
                {
                    "topic_kind": topic_kind,
                    "topic_name": topic_name,
                    "topic_key": topic_key,
                    "fact_ids": [],
                    "episode_ids": [],
                },
            )
            for field in ("fact_ids", "episode_ids"):
                for item_id in normalize_ids(item.get(field)):
                    if item_id not in grouped_item[field]:
                        grouped_item[field].append(item_id)

        report = {
            "topic_item_ids": [],
            "created_count": 0,
            "updated_count": 0,
            "fact_links_added": 0,
            "episode_links_added": 0,
        }
        if not grouped:
            return report

        now = local_now_text()
        for item in grouped.values():
            existing = self._conn.execute(
                """
                SELECT id FROM memory_topic_items
                WHERE topic_kind = ? AND topic_key = ?
                """,
                (item["topic_kind"], item["topic_key"]),
            ).fetchone()
            if existing is None:
                cur = self._conn.execute(
                    """
                    INSERT INTO memory_topic_items (
                        topic_kind, topic_name, topic_key,
                        fact_occurrence_count, episode_occurrence_count,
                        first_seen_at, last_seen_at, created_at, updated_at
                    ) VALUES (?, ?, ?, 0, 0, ?, ?, ?, ?)
                    """,
                    (
                        item["topic_kind"],
                        item["topic_name"],
                        item["topic_key"],
                        now,
                        now,
                        now,
                        now,
                    ),
                )
                topic_item_id = int(cur.lastrowid)
                report["created_count"] += 1
            else:
                topic_item_id = int(existing["id"])

            mapping = self._conn.execute(
                """
                SELECT fact_ids, episode_ids FROM memory_topic_mapping
                WHERE topic_item_id = ?
                """,
                (topic_item_id,),
            ).fetchone()
            existing_fact_ids = normalize_ids(
                _json_loads(mapping["fact_ids"], []) if mapping else []
            )
            existing_episode_ids = normalize_ids(
                _json_loads(mapping["episode_ids"], []) if mapping else []
            )
            new_fact_ids = [
                fact_id for fact_id in item["fact_ids"]
                if fact_id not in existing_fact_ids
            ]
            new_episode_ids = [
                episode_id for episode_id in item["episode_ids"]
                if episode_id not in existing_episode_ids
            ]
            merged_fact_ids = [*existing_fact_ids, *new_fact_ids]
            merged_episode_ids = [*existing_episode_ids, *new_episode_ids]

            self._conn.execute(
                """
                INSERT INTO memory_topic_mapping (
                    topic_item_id, fact_ids, episode_ids, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(topic_item_id) DO UPDATE SET
                    fact_ids = excluded.fact_ids,
                    episode_ids = excluded.episode_ids,
                    updated_at = excluded.updated_at
                """,
                (
                    topic_item_id,
                    _json_dumps(merged_fact_ids),
                    _json_dumps(merged_episode_ids),
                    now,
                    now,
                ),
            )
            if new_fact_ids or new_episode_ids:
                self._conn.execute(
                    """
                    UPDATE memory_topic_items
                    SET fact_occurrence_count = fact_occurrence_count + ?,
                        episode_occurrence_count = episode_occurrence_count + ?,
                        last_seen_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        len(new_fact_ids),
                        len(new_episode_ids),
                        now,
                        now,
                        topic_item_id,
                    ),
                )
                report["updated_count"] += 1
            report["topic_item_ids"].append(topic_item_id)
            report["fact_links_added"] += len(new_fact_ids)
            report["episode_links_added"] += len(new_episode_ids)

        self._commit_if_needed()
        return report

    def list_memory_topic_items(self, *, limit: int = 240) -> List[Dict[str, Any]]:
        """Load recent topic registry items with their compact evidence IDs."""
        rows = self._conn.execute(
            """
            SELECT item.*, mapping.fact_ids, mapping.episode_ids
            FROM memory_topic_items AS item
            LEFT JOIN memory_topic_mapping AS mapping
                ON mapping.topic_item_id = item.id
            ORDER BY item.last_seen_at DESC, item.id DESC
            LIMIT ?
            """,
            (max(1, int(limit or 240)),),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_unprocessed_facts(
        self,
        *,
        processing_target: str = "entity_claim",
        reference_timestamp: Any,
        source_types: Optional[Sequence[str]] = None,
        limit: int = 100,
        restrict_to_today: bool = True,
        require_episode: bool = False,
    ) -> List[Dict[str, Any]]:
        processing_columns = {
            "entity_claim": "processed_for_memory_entity_claim",
            "entity_claim_induction": "processed_for_memory_entity_claim_induction",
            "intent_execution": "processed_for_memory_intent_execution",
        }
        target = str(processing_target or "entity_claim").strip().lower()
        try:
            processing_column = processing_columns[target]
        except KeyError as exc:
            raise ValueError(
                "processing_target must be 'entity_claim', "
                "'entity_claim_induction', or 'intent_execution'"
            ) from exc

        clauses: List[str] = [f"{processing_column} = 0"]
        params: List[Any] = []
        if source_types:
            placeholders = ",".join("?" for _ in source_types)
            clauses.append(f"source_type IN ({placeholders})")
            params.extend(source_types)
        if require_episode:
            clauses.append("episode_id IS NOT NULL")
        if restrict_to_today:
            local_now = _coerce_reference_datetime(reference_timestamp).astimezone()
            event_date = local_now.date().isoformat()
            clauses.append("substr(created_at, 1, 10) = ?")
            params.append(event_date)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn.execute(
            f"""
            SELECT * FROM memory_facts
            {where}
            ORDER BY replace(substr(created_at, 1, 19), 'T', ' ') ASC, id ASC
            LIMIT ?
            """,
            (*params, max(1, int(limit or 100))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_memory_facts_after_id(
        self,
        *,
        fact_id: int = 0,
        source_type: Optional[str] = None,
        only_unassigned: bool = False,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Return facts newer than a runtime cursor in insertion order."""
        clauses = ["id > ?"]
        params: List[Any] = [int(fact_id or 0)]
        if source_type:
            clauses.append("source_type = ?")
            params.append(str(source_type))
        if only_unassigned:
            clauses.append("episode_id IS NULL")
        rows = self._conn.execute(
            f"SELECT * FROM memory_facts WHERE {' AND '.join(clauses)} "
            "ORDER BY id ASC LIMIT ?",
            (*params, max(1, int(limit or 500))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def update_facts_episode_id(
        self,
        *,
        fact_ids: Sequence[int],
        episode_id: int,
    ) -> int:
        """Attach existing facts to an episode and mirror entity mappings."""
        normalized_ids = [int(value) for value in fact_ids if str(value).strip().isdigit()]
        if not normalized_ids:
            return 0
        placeholders = ",".join("?" for _ in normalized_ids)
        rows = self._conn.execute(
            f"SELECT id, entity_ids FROM memory_facts WHERE id IN ({placeholders})",
            normalized_ids,
        ).fetchall()
        now = local_now_text()
        cur = self._conn.execute(
            f"UPDATE memory_facts SET episode_id = ?, updated_at = ? WHERE id IN ({placeholders})",
            (int(episode_id), now, *normalized_ids),
        )
        self._conn.execute(
            f"DELETE FROM memory_fact_episode_mapping "
            f"WHERE fact_id IN ({placeholders}) AND episode_id != ?",
            (*normalized_ids, int(episode_id)),
        )
        self.insert_fact_episode_mappings([
            {
                "fact_id": fact_id,
                "episode_id": int(episode_id),
            }
            for fact_id in normalized_ids
        ])
        mappings = []
        for row in rows:
            for entity_id in _json_loads(row["entity_ids"], default=[]):
                if str(entity_id).strip().isdigit():
                    mappings.append({"entity_id": int(entity_id), "episode_id": [int(episode_id)]})
        if mappings:
            self.insert_entity_memory_mappings(mappings)
        self._commit_if_needed()
        return int(cur.rowcount or 0)

    def insert_fact_episode_mappings(
        self,
        mappings: Sequence[Dict[str, Any]],
    ) -> int:
        """Persist stable membership links from facts to memory episodes."""
        normalized_pairs = {
            (int(mapping["fact_id"]), int(mapping["episode_id"]))
            for mapping in mappings or []
            if str(mapping.get("fact_id") or "").strip().isdigit()
            and str(mapping.get("episode_id") or "").strip().isdigit()
            and int(mapping["fact_id"]) > 0
            and int(mapping["episode_id"]) > 0
        }
        if not normalized_pairs:
            return 0
        fact_ids = sorted({fact_id for fact_id, _episode_id in normalized_pairs})
        episode_ids = sorted(
            {episode_id for _fact_id, episode_id in normalized_pairs}
        )
        fact_placeholders = ",".join("?" for _ in fact_ids)
        episode_placeholders = ",".join("?" for _ in episode_ids)
        existing_fact_ids = {
            int(row["id"])
            for row in self._conn.execute(
                f"SELECT id FROM memory_facts WHERE id IN ({fact_placeholders})",
                fact_ids,
            ).fetchall()
        }
        existing_episode_ids = {
            int(row["id"])
            for row in self._conn.execute(
                f"SELECT id FROM memory_episodes WHERE id IN ({episode_placeholders})",
                episode_ids,
            ).fetchall()
        }
        now = local_now_text()
        changed_count = 0
        for fact_id, episode_id in normalized_pairs:
            if fact_id not in existing_fact_ids or episode_id not in existing_episode_ids:
                continue
            self._conn.execute(
                """
                INSERT INTO memory_fact_episode_mapping (
                    fact_id, episode_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(fact_id, episode_id) DO NOTHING
                """,
                (fact_id, episode_id, now, now),
            )
            changed_count += 1
        self._commit_if_needed()
        return changed_count

    def update_episode(
        self,
        *,
        episode_id: int,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        canonical_topics: Optional[Sequence[str]] = None,
        started_at: Optional[str] = None,
        ended_at: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update generated episode-level fields after fact aggregation."""
        assignments: List[str] = []
        params: List[Any] = []
        for column, value in (
            ("title", title),
            ("summary", summary),
            ("canonical_topics", _json_dumps(list(canonical_topics or [])) if canonical_topics is not None else None),
            ("started_at", started_at),
            ("ended_at", ended_at),
            ("metadata", _json_dumps(dict(metadata or {})) if metadata is not None else None),
        ):
            if value is not None:
                assignments.append(f"{column} = ?")
                params.append(value)
        if not assignments:
            return False
        assignments.append("updated_at = ?")
        params.extend([local_now_text(), int(episode_id)])
        cur = self._conn.execute(
            f"UPDATE memory_episodes SET {', '.join(assignments)} WHERE id = ?",
            params,
        )
        self._commit_if_needed()
        return bool(cur.rowcount)

    def mark_facts_processed(
        self,
        *,
        processing_target: str,
        fact_ids: Sequence[int],
    ) -> int:
        processing_columns = {
            "entity_claim": "processed_for_memory_entity_claim",
            "entity_claim_induction": "processed_for_memory_entity_claim_induction",
            "intent_execution": "processed_for_memory_intent_execution",
        }
        target = str(processing_target or "").strip().lower()
        try:
            processing_column = processing_columns[target]
        except KeyError as exc:
            raise ValueError(
                "processing_target must be 'entity_claim', "
                "'entity_claim_induction', or 'intent_execution'"
            ) from exc
        ids = [int(value) for value in fact_ids if value is not None]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        cur = self._conn.execute(
            f"""
            UPDATE memory_facts
            SET {processing_column} = 1,
                updated_at = ?
            WHERE id IN ({placeholders})
            """,
            (local_now_text(), *ids),
        )
        self._commit_if_needed()
        return int(cur.rowcount or 0)

    def upsert_fact_entity_claim_signal_mappings(
        self,
        mappings: Sequence[Dict[str, Any]],
    ) -> int:
        """Index fact-level claim signals for exact induction-group expansion."""
        now = local_now_text()
        changed = 0
        for mapping in mappings or []:
            try:
                fact_id = int(mapping["fact_id"])
                subject_entity_id = int(mapping["subject_entity_id"])
            except (KeyError, TypeError, ValueError):
                continue
            claim_type_hint = str(mapping.get("claim_type_hint") or "").strip()
            signal_kind = str(mapping.get("signal_kind") or "").strip()
            claim_anchor = str(mapping.get("claim_anchor") or "").strip()
            claim_anchor_key = str(mapping.get("claim_anchor_key") or "").strip()
            if (
                fact_id <= 0
                or subject_entity_id <= 0
                or not claim_type_hint
                or not signal_kind
                or not claim_anchor
                or not claim_anchor_key
            ):
                continue
            self._conn.execute(
                """
                INSERT INTO memory_fact_entity_claim_signal_mapping (
                    fact_id, subject_entity_id, claim_type_hint, signal_kind,
                    claim_anchor, claim_anchor_key, confidence, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(
                    fact_id, subject_entity_id, claim_type_hint, claim_anchor_key
                ) DO UPDATE SET
                    signal_kind = excluded.signal_kind,
                    claim_anchor = excluded.claim_anchor,
                    confidence = MAX(
                        memory_fact_entity_claim_signal_mapping.confidence,
                        excluded.confidence
                    ),
                    updated_at = excluded.updated_at
                """,
                (
                    fact_id, subject_entity_id, claim_type_hint, signal_kind,
                    claim_anchor, claim_anchor_key,
                    float(mapping.get("confidence") or 0.0), now, now,
                ),
            )
            changed += 1
        self._commit_if_needed()
        return changed

    def memory_facts_for_entity_claim_signal_group(
        self,
        *,
        subject_entity_id: int,
        claim_type_hint: str,
        claim_anchor_key: str,
        limit: int = 64,
    ) -> List[Dict[str, Any]]:
        """Load bounded historical facts for one exact induction group."""
        rows = self._conn.execute(
            """
            SELECT fact.*
            FROM memory_fact_entity_claim_signal_mapping AS signal
            JOIN memory_facts AS fact ON fact.id = signal.fact_id
            WHERE signal.subject_entity_id = ?
              AND signal.claim_type_hint = ?
              AND signal.claim_anchor_key = ?
              AND fact.episode_id IS NOT NULL
            ORDER BY fact.dialogue_time_key DESC, fact.id DESC
            LIMIT ?
            """,
            (
                int(subject_entity_id), str(claim_type_hint or ""),
                str(claim_anchor_key or ""), max(1, int(limit or 64)),
            ),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_fact_entity_claim_signal_mappings(
        self,
        fact_id: int,
    ) -> List[Dict[str, Any]]:
        """Load the normalized claim signals persisted for one stored fact."""
        if int(fact_id or 0) <= 0:
            return []
        rows = self._conn.execute(
            """
            SELECT signal.*, entity.name AS subject
            FROM memory_fact_entity_claim_signal_mapping AS signal
            JOIN memory_entity_nodes AS entity
                ON entity.id = signal.subject_entity_id
            WHERE signal.fact_id = ?
            ORDER BY signal.claim_type_hint ASC,
                     signal.claim_anchor_key ASC
            """,
            (int(fact_id),),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def upsert_entity_claim(
        self,
        *,
        subject_entity_id: int,
        predicate: str,
        object_entity_id: Optional[int] = None,
        claim_text: str = "",
        claim_type: str,
        claim_origin: str,
        status: str = "candidate",
        confidence: float = 0.7,
        valid_from: str = "",
        valid_to: str = "",
        source_actor_entity_id: Optional[int] = None,
        extractor_version: str = "",
        prompt_version: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[int, bool]:
        """Insert a semantic claim or refresh its confidence and payload."""
        now = local_now_text()
        object_id = int(object_entity_id or 0)
        subject_id = int(subject_entity_id)
        predicate = str(predicate or "").strip()
        claim_text = re.sub(r"\s+", " ", str(claim_text or "")).strip()
        existing = self._conn.execute(
            """
            SELECT id, confidence, status FROM memory_entity_claims
            WHERE subject_entity_id = ? AND predicate = ? AND object_entity_id = ?
              AND claim_text = ? AND claim_type = ? AND claim_origin = ?
            """,
            (subject_id, predicate, object_id, claim_text, claim_type, claim_origin),
        ).fetchone()
        if existing:
            claim_id = int(existing["id"])
            self._conn.execute(
                """
                UPDATE memory_entity_claims
                SET claim_text = ?, status = ?, confidence = ?, valid_from = ?, valid_to = ?,
                    source_actor_entity_id = ?, extractor_version = ?, prompt_version = ?,
                    metadata = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    claim_text, status,
                    max(float(existing["confidence"] or 0.0), float(confidence or 0.0)),
                    valid_from, valid_to, source_actor_entity_id, extractor_version,
                    prompt_version, _json_dumps(metadata or {}), now, claim_id,
                ),
            )
            self._commit_if_needed()
            return claim_id, False
        cur = self._conn.execute(
            """
            INSERT INTO memory_entity_claims (
                subject_entity_id, predicate, object_entity_id, claim_text,
                claim_type, claim_origin, status, confidence, valid_from, valid_to,
                source_actor_entity_id, extractor_version, prompt_version, metadata,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                subject_id, predicate, object_id, claim_text,
                claim_type, claim_origin, status, float(confidence or 0.0),
                valid_from, valid_to, source_actor_entity_id, extractor_version,
                prompt_version, _json_dumps(metadata or {}), now, now,
            ),
        )
        self._commit_if_needed()
        return int(cur.lastrowid), True

    def transition_entity_claim_status(
        self,
        *,
        target_claim_id: int,
        new_status: str,
        trigger_claim_id: Optional[int] = None,
        semantic_relation: str = "",
        effective_at: str = "",
        decision_source: str = "",
        semantic_confidence: Optional[float] = None,
        semantic_reason: str = "",
        policy_reason: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Change a claim status and retain the causally linked history event.

        The event is deliberately written only for an actual status transition;
        evidence additions and duplicate merges do not make a claim appear to
        have been abandoned or revived.
        """
        target_id = int(target_claim_id)
        status = str(new_status or "").strip()
        if not status:
            return False
        occurred_at = str(effective_at or "").strip()
        now = local_now_text()
        with self.transaction():
            row = self._conn.execute(
                "SELECT status FROM memory_entity_claims WHERE id = ?",
                (target_id,),
            ).fetchone()
            if not row:
                return False
            previous_status = str(row["status"] or "")
            if previous_status == status:
                return False
            # A superseding claim closes the prior claim's validity interval.
            # We leave valid_to untouched for a weakened claim: it may still
            # describe a partially valid or temporarily interrupted pattern.
            if status == "superseded" and occurred_at:
                self._conn.execute(
                    """
                    UPDATE memory_entity_claims
                    SET status = ?,
                        valid_to = CASE
                            WHEN valid_to = '' OR valid_to > ? THEN ?
                            ELSE valid_to
                        END,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (status, occurred_at, occurred_at, now, target_id),
                )
            else:
                self._conn.execute(
                    "UPDATE memory_entity_claims SET status = ?, updated_at = ? WHERE id = ?",
                    (status, now, target_id),
                )
            self._conn.execute(
                """
                INSERT INTO memory_entity_claim_events (
                    target_claim_id, trigger_claim_id, event_type,
                    previous_status, new_status, semantic_relation, effective_at,
                    decision_source, semantic_confidence, semantic_reason,
                    policy_reason, details, created_at
                ) VALUES (?, ?, 'status_transition', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    target_id,
                    int(trigger_claim_id) if trigger_claim_id is not None else None,
                    previous_status,
                    status,
                    str(semantic_relation or "").strip(),
                    occurred_at,
                    str(decision_source or "").strip(),
                    float(semantic_confidence) if semantic_confidence is not None else None,
                    str(semantic_reason or "").strip(),
                    str(policy_reason or "").strip(),
                    _json_dumps(details or {}),
                    now,
                ),
            )
        return True

    @staticmethod
    def _intent_object_spec(object_type: str) -> tuple[str, tuple[str, ...]]:
        normalized = str(object_type or "").strip().lower()
        specs = {
            "goal": ("memory_goals", (
                "world_owner_entity_id", "owner_entity_id", "canonical_key", "summary",
                "desired_outcome", "success_criteria", "status", "target_at",
                "confidence", "metadata",
            )),
            "plan": ("memory_plans", (
                "world_owner_entity_id", "actor_entity_id", "canonical_key", "summary",
                "event_or_activity", "status", "start_at", "end_at", "time_precision",
                "location_entity_id", "location_text", "confidence", "metadata",
            )),
            "work_item": ("memory_work_items", (
                "world_owner_entity_id", "responsible_entity_id", "canonical_key", "summary",
                "action_text", "deliverable", "responsibility_type", "status", "due_at",
                "start_at", "priority", "confidence", "completed_at", "extractor_version",
                "prompt_version", "metadata",
            )),
        }
        try:
            return specs[normalized]
        except KeyError as exc:
            raise ValueError("object_type must be goal, plan, or work_item") from exc

    def create_intent_object(self, *, object_type: str, payload: Dict[str, Any]) -> int:
        """Persist one normalized Goal, Plan, or Work item candidate."""
        table, columns = self._intent_object_spec(object_type)
        now = local_now_text()
        values: List[Any] = []
        for column in columns:
            value = payload.get(column)
            if column == "metadata":
                value = _json_dumps(value if isinstance(value, dict) else {})
            elif column == "confidence":
                value = float(value or 0.0)
            elif column.endswith("_entity_id"):
                value = int(value) if value not in (None, "", 0) else None
            else:
                value = str(value or "")
            values.append(value)
        placeholders = ", ".join("?" for _ in columns)
        cur = self._conn.execute(
            f"INSERT INTO {table} ({', '.join(columns)}, created_at, updated_at) "
            f"VALUES ({placeholders}, ?, ?)",
            (*values, now, now),
        )
        self._commit_if_needed()
        return int(cur.lastrowid)

    def update_intent_object(
        self, *, object_type: str, object_id: int, payload: Dict[str, Any]
    ) -> bool:
        """Update a known object without erasing fields absent from an event."""
        table, allowed_columns = self._intent_object_spec(object_type)
        assignments: List[str] = []
        values: List[Any] = []
        for column in allowed_columns:
            if column not in payload:
                continue
            value = payload[column]
            if column == "metadata":
                value = _json_dumps(value if isinstance(value, dict) else {})
            elif column == "confidence":
                value = float(value or 0.0)
            elif column.endswith("_entity_id"):
                value = int(value) if value not in (None, "", 0) else None
            else:
                value = str(value or "")
            assignments.append(f"{column} = ?")
            values.append(value)
        if not assignments:
            return False
        assignments.append("updated_at = ?")
        values.extend([local_now_text(), int(object_id)])
        cur = self._conn.execute(
            f"UPDATE {table} SET {', '.join(assignments)} WHERE id = ?", values
        )
        self._commit_if_needed()
        return bool(cur.rowcount)

    def get_intent_objects(
        self,
        *,
        object_type: str,
        world_owner_entity_id: Optional[int] = None,
        statuses: Optional[Sequence[str]] = None,
        limit: int = 120,
    ) -> List[Dict[str, Any]]:
        table, _columns = self._intent_object_spec(object_type)
        clauses: List[str] = []
        params: List[Any] = []
        if world_owner_entity_id is not None:
            clauses.append("world_owner_entity_id = ?")
            params.append(int(world_owner_entity_id))
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(str(status) for status in statuses)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM {table}{where} ORDER BY updated_at DESC, id DESC LIMIT ?",
            (*params, max(1, int(limit or 120))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_intent_object(
        self, *, object_type: str, object_id: int
    ) -> Optional[Dict[str, Any]]:
        table, _columns = self._intent_object_spec(object_type)
        row = self._conn.execute(
            f"SELECT * FROM {table} WHERE id = ?", (int(object_id),)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def upsert_intent_evidence(self, evidence: Sequence[Dict[str, Any]]) -> int:
        now = local_now_text()
        changed = 0
        for item in evidence or []:
            object_type = str(item.get("object_type") or "").strip().lower()
            evidence_type = str(item.get("evidence_type") or "fact").strip().lower()
            role = str(item.get("role") or "support").strip().lower()
            try:
                object_id = int(item["object_id"])
                evidence_id = int(item["evidence_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                object_type not in {"goal", "plan", "work_item"}
                or evidence_type not in {"fact", "episode"}
                or object_id <= 0 or evidence_id <= 0
            ):
                continue
            self._conn.execute(
                """
                INSERT INTO memory_intent_evidence (
                    object_type, object_id, evidence_type, evidence_id, role,
                    observed_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(object_type, object_id, evidence_type, evidence_id, role)
                DO UPDATE SET observed_at = excluded.observed_at, updated_at = excluded.updated_at
                """,
                (object_type, object_id, evidence_type, evidence_id, role,
                 str(item.get("observed_at") or ""), now, now),
            )
            changed += 1
        self._commit_if_needed()
        return changed

    def insert_intent_event(
        self,
        *,
        object_type: str,
        object_id: int,
        event_type: str,
        previous_status: str = "",
        new_status: str = "",
        previous_payload: Optional[Dict[str, Any]] = None,
        new_payload: Optional[Dict[str, Any]] = None,
        evidence_fact_ids: Optional[Sequence[int]] = None,
        effective_at: str = "",
        decision_source: str = "",
        reason: str = "",
    ) -> int:
        normalized_type = str(object_type or "").strip().lower()
        if normalized_type not in {"goal", "plan", "work_item"}:
            return 0
        fact_ids = [int(value) for value in (evidence_fact_ids or []) if str(value).strip().isdigit()]
        cur = self._conn.execute(
            """
            INSERT INTO memory_intent_events (
                object_type, object_id, event_type, previous_status, new_status,
                previous_payload, new_payload, evidence_fact_ids, effective_at,
                decision_source, reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (normalized_type, int(object_id), str(event_type or "update"),
             str(previous_status or ""), str(new_status or ""),
             _json_dumps(previous_payload or {}), _json_dumps(new_payload or {}),
             _json_dumps(list(dict.fromkeys(fact_ids))), str(effective_at or ""),
             str(decision_source or ""), str(reason or ""), local_now_text()),
        )
        self._commit_if_needed()
        return int(cur.lastrowid)

    def upsert_intent_entities(
        self,
        *,
        object_type: str,
        object_id: int,
        entities: Sequence[Dict[str, Any]],
    ) -> int:
        normalized_type = str(object_type or "").strip().lower()
        table = {"plan": "memory_plan_entities", "work_item": "memory_work_item_entities"}.get(normalized_type)
        id_column = "plan_id" if normalized_type == "plan" else "work_item_id"
        if not table or int(object_id or 0) <= 0:
            return 0
        now = local_now_text()
        changed = 0
        for entity in entities or []:
            try:
                entity_id = int(entity["entity_id"])
            except (KeyError, TypeError, ValueError):
                continue
            role = str(entity.get("role") or "").strip().lower()
            if entity_id <= 0 or not role:
                continue
            self._conn.execute(
                f"""
                INSERT INTO {table} ({id_column}, entity_id, role, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT({id_column}, entity_id, role) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (int(object_id), entity_id, role, now, now),
            )
            changed += 1
        self._commit_if_needed()
        return changed

    def upsert_goal_work_item_mapping(self, *, goal_id: int, work_item_id: int, relation: str) -> bool:
        if int(goal_id or 0) <= 0 or int(work_item_id or 0) <= 0:
            return False
        now = local_now_text()
        self._conn.execute(
            """
            INSERT INTO memory_goal_work_item_mappings (goal_id, work_item_id, relation, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(goal_id, work_item_id, relation) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (int(goal_id), int(work_item_id), str(relation or "advances"), now, now),
        )
        self._commit_if_needed()
        return True

    def upsert_plan_work_item_mapping(self, *, plan_id: int, work_item_id: int, relation: str) -> bool:
        if int(plan_id or 0) <= 0 or int(work_item_id or 0) <= 0:
            return False
        now = local_now_text()
        self._conn.execute(
            """
            INSERT INTO memory_plan_work_item_mappings (plan_id, work_item_id, relation, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(plan_id, work_item_id, relation) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (int(plan_id), int(work_item_id), str(relation or "prepares"), now, now),
        )
        self._commit_if_needed()
        return True

    def get_entity_claim_events(
        self,
        claim_id: int,
        *,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return status-history events for one claim, newest event first."""
        rows = self._conn.execute(
            """
            SELECT * FROM memory_entity_claim_events
            WHERE target_claim_id = ?
            ORDER BY effective_at DESC, id DESC
            LIMIT ?
            """,
            (int(claim_id), max(1, int(limit or 100))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_entity_claims(
        self,
        *,
        subject_entity_id: Optional[int] = None,
        claim_type: Optional[str] = None,
        claim_origin: Optional[str] = None,
        predicate: Optional[str] = None,
        statuses: Optional[Sequence[str]] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        for column, value in (
            ("subject_entity_id", subject_entity_id),
            ("claim_type", claim_type),
            ("claim_origin", claim_origin),
            ("predicate", predicate),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(str(value) for value in statuses)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM memory_entity_claims{where} ORDER BY updated_at DESC, id DESC LIMIT ?",
            (*params, max(1, int(limit or 200))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def upsert_entity_claim_evidence(self, evidence: Sequence[Dict[str, Any]]) -> int:
        now = local_now_text()
        changed = 0
        for item in evidence or []:
            try:
                claim_id = int(item["claim_id"])
                evidence_id = int(item["evidence_id"])
            except (KeyError, TypeError, ValueError):
                continue
            evidence_type = str(item.get("evidence_type") or "fact")
            role = str(item.get("role") or "support")
            if claim_id <= 0 or evidence_id <= 0 or evidence_type not in {"fact", "episode"}:
                continue
            self._conn.execute(
                """
                INSERT INTO memory_entity_claim_evidence (
                    claim_id, evidence_type, evidence_id, role, weight, observed_at,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(claim_id, evidence_type, evidence_id, role) DO UPDATE SET
                    weight = MAX(memory_entity_claim_evidence.weight, excluded.weight),
                    observed_at = excluded.observed_at, updated_at = excluded.updated_at
                """,
                (claim_id, evidence_type, evidence_id, role, float(item.get("weight") or 1.0),
                 str(item.get("observed_at") or ""), now, now),
            )
            changed += 1
        self._commit_if_needed()
        return changed

    def get_entity_claim_support_fact_summary(self, claim_id: int) -> Dict[str, Any]:
        """Aggregate unique support facts so incremental induction never shrinks counts."""
        row = self._conn.execute(
            """
            SELECT
                COUNT(DISTINCT evidence_id) AS support_count,
                MIN(NULLIF(observed_at, '')) AS first_observed_at,
                MAX(NULLIF(observed_at, '')) AS last_observed_at
            FROM memory_entity_claim_evidence
            WHERE claim_id = ? AND evidence_type = 'fact' AND role = 'support'
            """,
            (int(claim_id),),
        ).fetchone()
        return {
            "support_count": int(row["support_count"] or 0) if row else 0,
            "first_observed_at": str(row["first_observed_at"] or "") if row else "",
            "last_observed_at": str(row["last_observed_at"] or "") if row else "",
        }

    def upsert_entity_claim_induction(
        self,
        *,
        claim_id: int,
        condition_text: str,
        support_count: int,
        first_observed_at: str,
        last_observed_at: str,
        consolidation_version: str = "v1",
    ) -> None:
        now = local_now_text()
        self._conn.execute(
            """
            INSERT INTO memory_entity_claim_induction (
                claim_id, condition_text, support_count,
                first_observed_at, last_observed_at,
                consolidation_version, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(claim_id) DO UPDATE SET
                condition_text = excluded.condition_text,
                support_count = excluded.support_count,
                first_observed_at = excluded.first_observed_at,
                last_observed_at = excluded.last_observed_at,
                consolidation_version = excluded.consolidation_version,
                updated_at = excluded.updated_at
            """,
            (int(claim_id), str(condition_text or ""),
             max(0, int(support_count or 0)),
             str(first_observed_at or ""), str(last_observed_at or ""),
             str(consolidation_version or "v1"), now, now),
        )
        self._commit_if_needed()

    def insert_fact(
        self,
        *,
        episode_id: Optional[int],
        source_type: str,
        fact_type: str,
        fact_kind: str,
        summary: str,
        keywords: Sequence[str],
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
        now = local_now_text()
        keyword_values = (
            [
                value
                for value in str(keywords).split()
                if value
            ]
            if isinstance(keywords, str)
            else [
                str(value).strip()
                for value in keywords or []
                if str(value).strip()
            ]
        )
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
                _json_dumps(keyword_values),
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
        if episode_id is not None:
            self.insert_fact_episode_mappings([
                {
                    "fact_id": fact_id,
                    "episode_id": int(episode_id),
                }
            ])
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
        now = local_now_text()
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
            SELECT entity_id, episode_id, fact_id
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
            }
            for row in rows
        ]

    def insert_entity_memory_mappings(
        self,
        mappings: Sequence[Dict[str, Any]],
    ) -> int:
        """Merge fact and episode links into one row per entity."""
        if not mappings:
            return 0

        mapping_fields = (
            "episode_id",
            "fact_id",
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

        now = local_now_text()
        changed_count = 0
        for entity_id, mapping in grouped.items():
            existing = self._conn.execute(
                """
                SELECT episode_id, fact_id
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
                    SET episode_id = ?, fact_id = ?, updated_at = ?
                    WHERE entity_id = ?
                    """,
                    (
                        _json_dumps(merged["episode_id"]),
                        _json_dumps(merged["fact_id"]),
                        now,
                        entity_id,
                    ),
                )
            else:
                self._conn.execute(
                    """
                    INSERT INTO memory_entity_mapping (
                        entity_id, episode_id, fact_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        entity_id,
                        _json_dumps(merged["episode_id"]),
                        _json_dumps(merged["fact_id"]),
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

        Table and field names are internal constants supplied by the public
        fact-search wrapper; user input is only ever bound as SQL values.
        Only identity-text lexical hits are returned. BM25 results are merged
        with the LIKE fallback when FTS5 is unavailable, while time filtering
        remains controlled by the caller.
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
        if time_clauses and len(row_ids) < row_limit and not strict_time_filter:
            # Time range is a strong preference, not a brittle hard stop. Pad
            # with broader lexical candidates so downstream reranking can
            # still recover facts with coarse or slightly shifted times.
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

    def memory_facts_with_identity_embeddings(
        self,
        *,
        source_types: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Load every embeddable fact for bounded in-process vector ranking."""
        clauses = ["identity_text_embedding IS NOT NULL"]
        params: List[Any] = []
        if source_types:
            placeholders = ",".join("?" for _ in source_types)
            clauses.append(f"source_type IN ({placeholders})")
            params.extend(source_types)
        where = " WHERE " + " AND ".join(clauses)
        rows = self._conn.execute(
            f"SELECT * FROM memory_facts{where} ORDER BY id ASC",
            params,
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

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

    def memory_episode_facts_for_entity_id(
        self,
        entity_id: int,
        *,
        limit: int = 240,
    ) -> List[Dict[str, Any]]:
        """Load completed-episode facts directly linked to one entity."""
        entity_token = f"%,{int(entity_id)},%"
        rows = self._conn.execute(
            """
            SELECT * FROM memory_facts
            WHERE episode_id IS NOT NULL
              AND (',' || replace(replace(replace(replace(entity_ids, ' ', ''), '\n', ''), '[', ''), ']', '') || ',') LIKE ?
            ORDER BY dialogue_time_key ASC, id ASC
            LIMIT ?
            """,
            (entity_token, max(1, int(limit or 240))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def related_fact_pairs_by_episode_fact_ids(
        self,
        fact_ids: Sequence[int],
        *,
        limit: int = 200,
    ) -> List[Dict[str, int]]:
        """Return other facts sharing an episode with each supplied fact."""
        ids = list(dict.fromkeys(
            int(value)
            for value in fact_ids or []
            if str(value).strip().isdigit() and int(value) > 0
        ))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"""
            SELECT seed.fact_id AS seed_fact_id,
                   related.fact_id AS related_fact_id
            FROM memory_fact_episode_mapping AS seed
            INNER JOIN memory_fact_episode_mapping AS related
                ON related.episode_id = seed.episode_id
            WHERE seed.fact_id IN ({placeholders})
              AND related.fact_id != seed.fact_id
            ORDER BY seed.fact_id ASC, related.fact_id ASC
            LIMIT ?
            """,
            (*ids, max(1, int(limit or 200))),
        ).fetchall()
        return [
            {
                "seed_fact_id": int(row["seed_fact_id"]),
                "related_fact_id": int(row["related_fact_id"]),
            }
            for row in rows
        ]

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        item = dict(row)
        if "keywords" in item:
            raw_keywords = item["keywords"]
            parsed_keywords = _json_loads(raw_keywords, None)
            if isinstance(parsed_keywords, list):
                item["keywords"] = [
                    str(value).strip()
                    for value in parsed_keywords
                    if str(value).strip()
                ]
            else:
                # Facts stored before keyword lists were serialized used a
                # whitespace-joined string. Their original phrase boundaries
                # cannot be recovered, so retain the former token behavior.
                item["keywords"] = [
                    value
                    for value in str(raw_keywords or "").split()
                    if value
                ]
        for key in (
            "entities",
            "entity_ids",
            "canonical_topics",
            "participants",
            "metadata",
            "details",
            "previous_payload",
            "new_payload",
            "evidence_fact_ids",
            "fact_ids",
            "episode_ids",
        ):
            if key in item:
                item[key] = _json_loads(
                    item[key],
                    {} if key in {"metadata", "details"} else [],
                )
        for key in (
            "embedding",
            "identity_text_embedding",
        ):
            if key in item:
                item[key] = _blob_to_embedding(item[key])
        return item
