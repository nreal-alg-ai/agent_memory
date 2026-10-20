"""Durable task records and compact, batch-level consolidation checkpoints."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _ids(value: Any) -> List[int]:
    if isinstance(value, str):
        value = json.loads(value or "[]")
    return list(dict.fromkeys(
        int(item) for item in (value or []) if str(item).isdigit() and int(item) > 0
    ))


def _locked(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


class TaskExecutionDB:
    """One row per producer task; memory.db owns the referenced content."""

    _SIGNAL_BATCHES = {
        "claim": "user_claim_signal_consolidation_task_info",
        "intent": "user_intent_signal_consolidation_task_info",
        "action": "user_action_signal_consolidation_task_info",
    }
    _PENDING_COLUMNS = {
        "explicit_user_claim": (
            "claim", "pending_user_claim_signal_ids", "processed_explicit_user_claim",
        ),
        "user_intent_update": (
            "intent", "pending_user_intent_signal_ids", "processed_user_intent",
        ),
        "user_action_signal_consolidation": (
            "action", "pending_user_action_signal_ids", "processed_user_action",
        ),
    }

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._run_id = uuid.uuid4().hex
        self._conn = sqlite3.connect(
            str(self.path), timeout=30.0, check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA journal_mode=WAL")
        for table, required_column in (
            ("task_signal_progress", None),
            ("user_claim_signal_consolidation_task_info", "signal_ids"),
            ("user_intent_signal_consolidation_task_info", "signal_ids"),
            ("user_action_signal_consolidation_task_info", "signal_ids"),
            ("user_claim_consolidation_task_info", "claim_ids"),
        ):
            exists = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if exists and (required_column is None or required_column not in {
                row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})")
            }):
                self._conn.close()
                raise RuntimeError(
                    f"task_scheduler.db uses the retired {table} schema; "
                    "back it up and use a fresh task database"
                )
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS task_execution_records (
                run_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                task_kind TEXT NOT NULL,
                status TEXT NOT NULL,
                submitted_at TEXT NOT NULL,
                completed_at TEXT,
                result_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY(run_id, task_id)
            );
            CREATE TABLE IF NOT EXISTS user_prediction_intent_progress (
                object_type TEXT NOT NULL,
                object_id INTEGER NOT NULL,
                fingerprint TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                PRIMARY KEY(object_type, object_id)
            );
            CREATE TABLE IF NOT EXISTS user_claim_signal_consolidation_task_info (
                batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_task_id TEXT NOT NULL UNIQUE,
                signal_ids TEXT NOT NULL DEFAULT '[]',
                pending_user_claim_signal_ids TEXT NOT NULL DEFAULT '[]',
                processed_explicit_user_claim INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_claim_signal_batches_pending
            ON user_claim_signal_consolidation_task_info(
                processed_explicit_user_claim, batch_id
            );
            CREATE TABLE IF NOT EXISTS user_intent_signal_consolidation_task_info (
                batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_task_id TEXT NOT NULL UNIQUE,
                signal_ids TEXT NOT NULL DEFAULT '[]',
                pending_user_intent_signal_ids TEXT NOT NULL DEFAULT '[]',
                processed_user_intent INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_intent_signal_batches_pending
            ON user_intent_signal_consolidation_task_info(
                processed_user_intent, batch_id
            );
            CREATE TABLE IF NOT EXISTS user_action_signal_consolidation_task_info (
                batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_task_id TEXT NOT NULL UNIQUE,
                signal_ids TEXT NOT NULL DEFAULT '[]',
                pending_user_action_signal_ids TEXT NOT NULL DEFAULT '[]',
                processed_user_action INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_action_signal_batches_pending
            ON user_action_signal_consolidation_task_info(
                processed_user_action, batch_id
            );
            CREATE TABLE IF NOT EXISTS user_claim_consolidation_task_info (
                batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_task_id TEXT NOT NULL UNIQUE,
                claim_ids TEXT NOT NULL DEFAULT '[]',
                pending_user_state_claim_ids TEXT NOT NULL DEFAULT '[]',
                processed_user_state INTEGER NOT NULL DEFAULT 0,
                reference_timestamp TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_claim_batches_pending
            ON user_claim_consolidation_task_info(processed_user_state, batch_id);
        """)
        # Queued tasks are process-local. Restore unfinished reservations after
        # a restart; rows with no pending IDs remain processed.
        with self._conn:
            for signal_type, table in self._SIGNAL_BATCHES.items():
                _, pending_column, processed_column = next(
                    columns for columns in self._PENDING_COLUMNS.values()
                    if columns[0] == signal_type
                )
                self._conn.execute(
                    f"UPDATE {table} SET {processed_column}=0 "
                    f"WHERE {processed_column}=1 AND {pending_column} != '[]'"
                )
        self._registered_signal_ids = {
            signal_type: self._load_registered_signal_ids(table)
            for signal_type, table in self._SIGNAL_BATCHES.items()
        }

    @_locked
    def close(self) -> None:
        self._conn.close()

    @_locked
    def record_submission(self, task_id: str, task_kind: str) -> None:
        self._conn.execute(
            """INSERT INTO task_execution_records
               (run_id, task_id, task_kind, status, submitted_at)
               VALUES (?, ?, ?, 'queued', ?)""",
            (self._run_id, task_id, task_kind, _now()),
        )
        self._conn.commit()

    @_locked
    def record_completion(
        self, task_id: str, *, status: str, result: Any,
    ) -> None:
        try:
            result_json = json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            result_json = "{}"
        self._conn.execute(
            """UPDATE task_execution_records
               SET status = ?, completed_at = ?, result_json = ?
               WHERE run_id = ? AND task_id = ?""",
            (status, _now(), result_json, self._run_id, task_id),
        )
        self._conn.commit()

    @_locked
    def completed_prediction_intents(
        self, intents: Sequence[Dict[str, Any]],
    ) -> set[Tuple[str, int, str]]:
        completed: set[Tuple[str, int, str]] = set()
        for intent in intents:
            object_type = str(intent["object_type"])
            object_id = int(intent["id"])
            row = self._conn.execute(
                """SELECT fingerprint FROM user_prediction_intent_progress
                   WHERE object_type = ? AND object_id = ?""",
                (object_type, object_id),
            ).fetchone()
            fingerprint = str(intent["fingerprint"])
            if row and row["fingerprint"] == fingerprint:
                completed.add((object_type, object_id, fingerprint))
        return completed

    @_locked
    def complete_prediction_intents(
        self, intents: Sequence[Dict[str, Any]],
    ) -> None:
        self._conn.executemany(
            """INSERT INTO user_prediction_intent_progress
               (object_type, object_id, fingerprint, completed_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(object_type, object_id) DO UPDATE SET
                 fingerprint = excluded.fingerprint,
                 completed_at = excluded.completed_at""",
            [(str(item["object_type"]), int(item["id"]),
              str(item["fingerprint"]), _now()) for item in intents],
        )
        self._conn.commit()

    def _source_reference(self, source_task_id: str) -> str:
        return f"{self._run_id}:{source_task_id or 'startup_recovery'}"

    def _load_registered_signal_ids(self, table: str) -> set[int]:
        registered: set[int] = set()
        for row in self._conn.execute(f"SELECT signal_ids FROM {table}"):
            registered.update(_ids(row["signal_ids"]))
        return registered

    def _register_signal_batch(
        self, signal_type: str, signal_ids: Sequence[int], source_task_id: str,
    ) -> List[int]:
        table = self._SIGNAL_BATCHES[signal_type]
        new_ids = sorted(
            set(_ids(signal_ids)) - self._registered_signal_ids[signal_type]
        )
        if not new_ids:
            return []
        source = self._source_reference(source_task_id)
        existing = self._conn.execute(
            f"SELECT * FROM {table} WHERE source_task_id = ?", (source,),
        ).fetchone()
        now = _now()
        if signal_type == "claim":
            if existing:
                all_ids = _ids(existing["signal_ids"]) + new_ids
                explicit = _ids(existing["pending_user_claim_signal_ids"]) + new_ids
                self._conn.execute(
                    f"""UPDATE {table} SET signal_ids=?,
                        pending_user_claim_signal_ids=?,
                        processed_explicit_user_claim=0, updated_at=?
                        WHERE batch_id=?""",
                    (json.dumps(all_ids), json.dumps(explicit), now, existing["batch_id"]),
                )
            else:
                values = json.dumps(new_ids)
                self._conn.execute(
                    f"""INSERT INTO {table}
                        (source_task_id, signal_ids, pending_user_claim_signal_ids,
                         created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?)""",
                    (source, values, values, now, now),
                )
        elif signal_type == "action":
            if existing:
                all_ids = _ids(existing["signal_ids"]) + new_ids
                pending = _ids(existing["pending_user_action_signal_ids"]) + new_ids
                self._conn.execute(
                    f"""UPDATE {table} SET signal_ids=?,
                        pending_user_action_signal_ids=?, processed_user_action=0,
                        updated_at=? WHERE batch_id=?""",
                    (json.dumps(all_ids), json.dumps(pending), now, existing["batch_id"]),
                )
            else:
                values = json.dumps(new_ids)
                self._conn.execute(
                    f"""INSERT INTO {table}
                        (source_task_id, signal_ids, pending_user_action_signal_ids,
                         created_at, updated_at) VALUES (?, ?, ?, ?, ?)""",
                    (source, values, values, now, now),
                )
        elif existing:
            all_ids = _ids(existing["signal_ids"]) + new_ids
            pending = _ids(existing["pending_user_intent_signal_ids"]) + new_ids
            self._conn.execute(
                f"""UPDATE {table} SET signal_ids=?,
                    pending_user_intent_signal_ids=?, processed_user_intent=0,
                    updated_at=? WHERE batch_id=?""",
                (json.dumps(all_ids), json.dumps(pending), now, existing["batch_id"]),
            )
        else:
            values = json.dumps(new_ids)
            self._conn.execute(
                f"""INSERT INTO {table}
                    (source_task_id, signal_ids, pending_user_intent_signal_ids,
                     created_at, updated_at) VALUES (?, ?, ?, ?, ?)""",
                (source, values, values, now, now),
            )
        return new_ids

    @_locked
    def register_signals(
        self,
        *,
        claim_signal_ids: Sequence[int],
        intent_signal_ids: Sequence[int],
        action_signal_ids: Sequence[int] = (),
        source_task_id: str = "",
    ) -> int:
        """Append only IDs absent from prior producer batches."""
        with self._conn:
            claim_added = self._register_signal_batch(
                "claim", claim_signal_ids, source_task_id,
            )
            intent_added = self._register_signal_batch(
                "intent", intent_signal_ids, source_task_id,
            )
            action_added = self._register_signal_batch(
                "action", action_signal_ids, source_task_id,
            )
        self._registered_signal_ids["claim"].update(claim_added)
        self._registered_signal_ids["intent"].update(intent_added)
        self._registered_signal_ids["action"].update(action_added)
        return len(claim_added) + len(intent_added) + len(action_added)

    @_locked
    def pending_signal_ids(self, target: str) -> List[int]:
        try:
            signal_type, pending_column, processed_column = self._PENDING_COLUMNS[target]
        except KeyError as exc:
            raise ValueError(f"Invalid signal processing target: {target}") from exc
        table = self._SIGNAL_BATCHES[signal_type]
        pending: List[int] = []
        for row in self._conn.execute(
            f"SELECT {pending_column} FROM {table} "
            f"WHERE {processed_column}=0 ORDER BY batch_id"
        ):
            pending.extend(_ids(row[pending_column]))
        return list(dict.fromkeys(pending))

    @_locked
    def reserve_signal_ids(self, target: str, signal_ids: Sequence[int]) -> List[int]:
        """Hide submitted signals from later task snapshots until completion."""
        try:
            signal_type, pending_column, processed_column = self._PENDING_COLUMNS[target]
        except KeyError as exc:
            raise ValueError(f"Invalid signal processing target: {target}") from exc
        selected = set(_ids(signal_ids))
        if not selected:
            return []
        table = self._SIGNAL_BATCHES[signal_type]
        reserved: set[int] = set()
        with self._conn:
            for row in self._conn.execute(
                f"SELECT batch_id, {pending_column} FROM {table} "
                f"WHERE {processed_column}=0 ORDER BY batch_id"
            ).fetchall():
                matched = set(_ids(row[pending_column])) & selected
                if not matched:
                    continue
                reserved.update(matched)
                self._conn.execute(
                    f"UPDATE {table} SET {processed_column}=1, updated_at=? "
                    "WHERE batch_id=?",
                    (_now(), row["batch_id"]),
                )
            if reserved != selected:
                raise ValueError("Signal IDs are no longer pending")
        return sorted(reserved)

    @_locked
    def finish_signal_task(
        self, target: str, selected_ids: Sequence[int], completed_ids: Sequence[int],
    ) -> List[int]:
        """Acknowledge completed IDs and release uncompleted IDs for retry."""
        try:
            signal_type, pending_column, processed_column = self._PENDING_COLUMNS[target]
        except KeyError as exc:
            raise ValueError(f"Invalid signal processing target: {target}") from exc
        selected = set(_ids(selected_ids))
        completed = set(_ids(completed_ids)) & selected
        if not selected:
            return []
        table = self._SIGNAL_BATCHES[signal_type]
        changed: set[int] = set()
        with self._conn:
            for row in self._conn.execute(
                f"SELECT batch_id, {pending_column} FROM {table} "
                f"WHERE {processed_column}=1 ORDER BY batch_id"
            ).fetchall():
                pending = _ids(row[pending_column])
                if not set(pending) & selected:
                    continue
                changed.update(set(pending) & completed)
                remaining = [signal_id for signal_id in pending if signal_id not in completed]
                self._conn.execute(
                    f"UPDATE {table} SET {pending_column}=?, {processed_column}=?, "
                    "updated_at=? WHERE batch_id=?",
                    (json.dumps(remaining), int(not remaining), _now(), row["batch_id"]),
                )
        return sorted(changed)

    @_locked
    def mark_signal_ids_completed(
        self, target: str, signal_ids: Sequence[int],
    ) -> List[int]:
        try:
            signal_type, pending_column, processed_column = self._PENDING_COLUMNS[target]
        except KeyError as exc:
            raise ValueError(f"Invalid signal processing target: {target}") from exc
        completed = set(_ids(signal_ids))
        if not completed:
            return []
        table = self._SIGNAL_BATCHES[signal_type]
        changed: set[int] = set()
        with self._conn:
            for row in self._conn.execute(
                f"SELECT batch_id, {pending_column} FROM {table} "
                f"WHERE {processed_column}=0 ORDER BY batch_id"
            ).fetchall():
                pending = _ids(row[pending_column])
                matched = set(pending) & completed
                if not matched:
                    continue
                changed.update(matched)
                remaining = [signal_id for signal_id in pending if signal_id not in matched]
                self._conn.execute(
                    f"""UPDATE {table} SET {pending_column}=?,
                        {processed_column}=?, updated_at=? WHERE batch_id=?""",
                    (json.dumps(remaining), int(not remaining), _now(), row["batch_id"]),
                )
        return sorted(changed)

    @_locked
    def record_pending_user_state_claims(
        self,
        claim_ids: Sequence[int],
        *,
        reference_timestamp: Any = None,
        source_task_id: str = "",
    ) -> int:
        ids = sorted(_ids(claim_ids))
        if not ids:
            return 0
        source = self._source_reference(source_task_id or uuid.uuid4().hex)
        now = _now()
        with self._conn:
            existing = self._conn.execute(
                "SELECT batch_id FROM user_claim_consolidation_task_info "
                "WHERE source_task_id=?", (source,),
            ).fetchone()
            if existing:
                return 0
            values = json.dumps(ids)
            self._conn.execute(
                """INSERT INTO user_claim_consolidation_task_info
                   (source_task_id, claim_ids, pending_user_state_claim_ids,
                    reference_timestamp, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (source, values, values, str(reference_timestamp or ""), now, now),
            )
        return len(ids)

    @_locked
    def get_pending_user_state_claims(self, *, limit: int = 80) -> Dict[str, Any]:
        by_claim: Dict[int, Dict[str, Any]] = {}
        for row in self._conn.execute(
            """SELECT batch_id, pending_user_state_claim_ids, reference_timestamp,
                      created_at
               FROM user_claim_consolidation_task_info
               WHERE processed_user_state=0 ORDER BY batch_id"""
        ):
            for claim_id in _ids(row["pending_user_state_claim_ids"]):
                item = by_claim.setdefault(claim_id, {
                    "claim_id": claim_id,
                    "generation": int(row["batch_id"]),
                    "first_detected_at": str(row["created_at"]),
                    "last_detected_at": str(row["created_at"]),
                    "reference_timestamp": str(row["reference_timestamp"] or ""),
                })
                item["generation"] = int(row["batch_id"])
                item["last_detected_at"] = str(row["created_at"])
                item["reference_timestamp"] = str(row["reference_timestamp"] or "")
        ordered = sorted(
            by_claim.values(), key=lambda item: (item["first_detected_at"], item["claim_id"]),
        )
        return {
            "pending_count": len(ordered),
            "first_detected_at": ordered[0]["first_detected_at"] if ordered else "",
            "items": ordered[:max(1, int(limit or 80))],
        }

    @_locked
    def acknowledge_user_state_claims(
        self, versions: Sequence[Tuple[int, int]],
    ) -> int:
        limits = {int(claim_id): int(generation) for claim_id, generation in versions}
        if not limits:
            return 0
        changed: set[int] = set()
        with self._conn:
            for row in self._conn.execute(
                """SELECT batch_id, pending_user_state_claim_ids
                   FROM user_claim_consolidation_task_info
                   WHERE processed_user_state=0 ORDER BY batch_id"""
            ).fetchall():
                pending = _ids(row["pending_user_state_claim_ids"])
                matched = {
                    claim_id for claim_id in pending
                    if claim_id in limits and row["batch_id"] <= limits[claim_id]
                }
                if not matched:
                    continue
                changed.update(matched)
                remaining = [claim_id for claim_id in pending if claim_id not in matched]
                self._conn.execute(
                    """UPDATE user_claim_consolidation_task_info
                       SET pending_user_state_claim_ids=?, processed_user_state=?,
                           updated_at=? WHERE batch_id=?""",
                    (json.dumps(remaining), int(not remaining), _now(), row["batch_id"]),
                )
        return len(changed)
