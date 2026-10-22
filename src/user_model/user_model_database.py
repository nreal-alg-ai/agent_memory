"""SQLite projection store for durable user states and hypotheses.

The memory database owns evidence; cross-database evidence IDs here are logical
references, not SQLite foreign keys. A single FIFO worker orders updates.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from memory.memory_database import _json_dumps, _json_loads, local_now_text


class UserModelDB:
    """Own user-state and prediction projections, separately from memory.db."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._transaction_depth = 0
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self):
        outermost = self._transaction_depth == 0
        if outermost:
            self._conn.execute("BEGIN")
        self._transaction_depth += 1
        try:
            yield self
        except Exception:
            self._transaction_depth -= 1
            if outermost:
                self._conn.rollback()
            raise
        else:
            self._transaction_depth -= 1
            if outermost:
                self._conn.commit()

    def _commit_if_needed(self) -> None:
        if self._transaction_depth == 0:
            self._conn.commit()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        return dict(row)

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS user_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state_domain TEXT NOT NULL,
                state_type TEXT NOT NULL,
                state_kind TEXT NOT NULL,
                state_topic TEXT NOT NULL,
                state_text TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                confidence REAL NOT NULL DEFAULT 0.0,
                conditions TEXT NOT NULL DEFAULT '{}',
                valid_from TEXT NOT NULL DEFAULT '',
                valid_to TEXT NOT NULL DEFAULT '',
                last_observed_at TEXT NOT NULL DEFAULT '',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(state_domain, state_type, state_kind, state_topic)
            );
            CREATE TABLE IF NOT EXISTS user_state_evidence (
                state_id INTEGER NOT NULL,
                evidence_type TEXT NOT NULL,
                evidence_id INTEGER NOT NULL,
                role TEXT NOT NULL DEFAULT 'support',
                weight REAL NOT NULL DEFAULT 1.0,
                observed_at TEXT NOT NULL DEFAULT '',
                details TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(state_id, evidence_type, evidence_id, role),
                FOREIGN KEY(state_id) REFERENCES user_states(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS user_state_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state_id INTEGER NOT NULL,
                operation TEXT NOT NULL,
                previous_payload TEXT NOT NULL DEFAULT '{}',
                new_payload TEXT NOT NULL DEFAULT '{}',
                trigger_claim_ids TEXT NOT NULL DEFAULT '[]',
                trigger_intent_objects TEXT NOT NULL DEFAULT '[]',
                effective_at TEXT NOT NULL DEFAULT '',
                decision_source TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.0,
                reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY(state_id) REFERENCES user_states(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS user_prediction_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_entity_id INTEGER NOT NULL,
                prediction_kind TEXT NOT NULL,
                as_of TEXT NOT NULL,
                horizon_start TEXT NOT NULL,
                horizon_end TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('ok', 'empty', 'error')),
                input_snapshot TEXT NOT NULL DEFAULT '{}',
                predictor_version TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS user_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                user_entity_id INTEGER NOT NULL,
                prediction_key TEXT NOT NULL,
                basis_hash TEXT NOT NULL,
                behavior_text TEXT NOT NULL,
                window_start TEXT NOT NULL,
                window_end TEXT NOT NULL,
                conditions TEXT NOT NULL DEFAULT '[]',
                likelihood_band TEXT NOT NULL,
                uncertainties TEXT NOT NULL DEFAULT '[]',
                rationale TEXT NOT NULL DEFAULT '',
                basis TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                outcome TEXT NOT NULL DEFAULT '{}',
                resolved_at TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES user_prediction_runs(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS user_proactive_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                user_entity_id INTEGER NOT NULL,
                policy_version TEXT NOT NULL,
                decision TEXT NOT NULL CHECK(decision IN (
                    'recommend_service', 'defer', 'suppress'
                )),
                reason_code TEXT NOT NULL DEFAULT '',
                rationale TEXT NOT NULL DEFAULT '',
                runtime_context TEXT NOT NULL DEFAULT '{}',
                prediction_snapshot TEXT NOT NULL DEFAULT '{}',
                source_prediction_task_id TEXT NOT NULL DEFAULT '',
                evaluated_at TEXT NOT NULL,
                UNIQUE(prediction_id, policy_version),
                FOREIGN KEY(prediction_id) REFERENCES user_predictions(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_user_states_domain_type_status
            ON user_states(state_domain, state_type, status, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_user_state_evidence_state
            ON user_state_evidence(state_id, role, evidence_type);
            CREATE INDEX IF NOT EXISTS idx_user_state_events_state
            ON user_state_events(state_id, effective_at DESC, id DESC);
            CREATE INDEX IF NOT EXISTS idx_user_prediction_runs_owner
            ON user_prediction_runs(user_entity_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_user_predictions_active
            ON user_predictions(user_entity_id, status, window_end);
            CREATE INDEX IF NOT EXISTS idx_user_proactive_decisions_user_time
            ON user_proactive_decisions(user_entity_id, evaluated_at DESC, id DESC);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_user_predictions_pending_key
            ON user_predictions(user_entity_id, prediction_key)
            WHERE status = 'pending';
        """)
        self._commit_if_needed()

    def insert_user_prediction_run(self, payload: Dict[str, Any]) -> int:
        cursor = self._conn.execute(
            """INSERT INTO user_prediction_runs (
                user_entity_id, prediction_kind, as_of, horizon_start, horizon_end,
                status, input_snapshot, predictor_version, error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                int(payload["user_entity_id"]), str(payload["prediction_kind"]),
                str(payload["as_of"]), str(payload["horizon_start"]),
                str(payload["horizon_end"]), str(payload["status"]),
                _json_dumps(payload.get("input_snapshot") or {}),
                str(payload.get("predictor_version") or ""),
                str(payload.get("error") or ""), local_now_text(),
            ),
        )
        self._commit_if_needed()
        return int(cursor.lastrowid)

    def save_user_prediction(self, payload: Dict[str, Any]) -> Tuple[int, str]:
        """Reuse an unchanged pending forecast, supersede it when evidence changes."""
        owner_id = int(payload["user_entity_id"])
        prediction_key = str(payload["prediction_key"])
        basis_hash = str(payload["basis_hash"])
        existing = self._conn.execute(
            """SELECT id, basis_hash FROM user_predictions
            WHERE user_entity_id = ? AND prediction_key = ? AND status = 'pending'""",
            (owner_id, prediction_key),
        ).fetchone()
        if existing and existing["basis_hash"] == basis_hash:
            return int(existing["id"]), "reused"
        if existing:
            self._conn.execute(
                """UPDATE user_predictions SET status = 'superseded',
                resolved_at = ? WHERE id = ?""",
                (local_now_text(), int(existing["id"])),
            )
        cursor = self._conn.execute(
            """INSERT INTO user_predictions (
                run_id, user_entity_id, prediction_key, basis_hash, behavior_text,
                window_start, window_end, conditions, likelihood_band,
                uncertainties, rationale, basis, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                int(payload["run_id"]), owner_id, prediction_key, basis_hash,
                str(payload["behavior_text"]), str(payload["window_start"]),
                str(payload["window_end"]),
                _json_dumps(payload.get("conditions") or []),
                str(payload["likelihood_band"]),
                _json_dumps(payload.get("uncertainties") or []),
                str(payload.get("rationale") or ""),
                _json_dumps(payload.get("basis") or {}), local_now_text(),
            ),
        )
        self._commit_if_needed()
        return int(cursor.lastrowid), "created"

    @staticmethod
    def _prediction_row_to_api(row: sqlite3.Row) -> Dict[str, Any]:
        item = dict(row)
        for field, fallback in (("conditions", []), ("uncertainties", []),
                                ("basis", {}), ("outcome", {})):
            item[field] = _json_loads(item.get(field), fallback)
        return item

    def get_active_user_predictions(
        self, *, user_entity_id: int, as_of: str, limit: int = 100,
    ) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT * FROM user_predictions
            WHERE user_entity_id = ? AND status = 'pending'
              AND julianday(window_end) > julianday(?)
            ORDER BY window_start, id LIMIT ?""",
            (int(user_entity_id), str(as_of), max(1, int(limit))),
        ).fetchall()
        return [self._prediction_row_to_api(row) for row in rows]

    def get_user_predictions_by_ids(
        self, prediction_ids: Sequence[int],
    ) -> List[Dict[str, Any]]:
        ids = sorted({int(value) for value in prediction_ids
                      if str(value).isdigit() and int(value) > 0})
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT * FROM user_predictions WHERE id IN ({placeholders}) ORDER BY id",
            ids,
        ).fetchall()
        return [self._prediction_row_to_api(row) for row in rows]

    def save_user_proactive_decision(
        self, payload: Dict[str, Any],
    ) -> Tuple[int, bool]:
        cursor = self._conn.execute(
            """INSERT OR IGNORE INTO user_proactive_decisions (
                prediction_id, user_entity_id, policy_version, decision,
                reason_code, rationale, runtime_context, prediction_snapshot,
                source_prediction_task_id, evaluated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                int(payload["prediction_id"]), int(payload["user_entity_id"]),
                str(payload["policy_version"]), str(payload["decision"]),
                str(payload.get("reason_code") or ""),
                str(payload.get("rationale") or ""),
                _json_dumps(payload.get("runtime_context") or {}),
                _json_dumps(payload.get("prediction_snapshot") or {}),
                str(payload.get("source_prediction_task_id") or ""),
                str(payload["evaluated_at"]),
            ),
        )
        self._commit_if_needed()
        created = cursor.rowcount == 1
        row = self._conn.execute(
            """SELECT id FROM user_proactive_decisions
               WHERE prediction_id = ? AND policy_version = ?""",
            (int(payload["prediction_id"]), str(payload["policy_version"])),
        ).fetchone()
        return int(row["id"]), created

    def get_user_proactive_decisions(
        self, *, user_entity_id: Optional[int] = None, limit: int = 100,
    ) -> List[Dict[str, Any]]:
        clauses = []
        params: List[Any] = []
        if user_entity_id is not None:
            clauses.append("user_entity_id = ?")
            params.append(int(user_entity_id))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"""SELECT * FROM user_proactive_decisions {where}
                ORDER BY evaluated_at DESC, id DESC LIMIT ?""",
            (*params, max(1, int(limit))),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["runtime_context"] = _json_loads(item.get("runtime_context"), {})
            item["prediction_snapshot"] = _json_loads(item.get("prediction_snapshot"), {})
            result.append(item)
        return result

    def record_user_prediction_outcome(
        self, *, prediction_id: int, outcome_status: str,
        outcome: Dict[str, Any],
    ) -> bool:
        if outcome_status not in {"confirmed", "contradicted", "unverifiable", "invalidated"}:
            raise ValueError("Unsupported prediction outcome status")
        cursor = self._conn.execute(
            """UPDATE user_predictions
            SET status = ?, outcome = ?, resolved_at = ?
            WHERE id = ? AND status = 'pending'""",
            (outcome_status, _json_dumps(outcome), local_now_text(), int(prediction_id)),
        )
        self._commit_if_needed()
        return cursor.rowcount == 1

    def get_user_states(
        self,
        *,
        state_domains: Optional[Sequence[str]] = None,
        state_types: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[str]] = None,
        limit: int = 160,
        include_evidence: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return durable states, optionally attaching all of each state's evidence."""
        clauses: List[str] = []
        params: List[Any] = []
        if state_domains:
            placeholders = ",".join("?" for _ in state_domains)
            clauses.append(f"state_domain IN ({placeholders})")
            params.extend(str(value) for value in state_domains)
        if state_types:
            placeholders = ",".join("?" for _ in state_types)
            clauses.append(f"state_type IN ({placeholders})")
            params.extend(str(value) for value in state_types)
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(str(value) for value in statuses)
        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"""
            SELECT * FROM user_states
            {where_clause}
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            (*params, max(1, int(limit or 160))),
        ).fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows:
            item = self._row_to_dict(row)
            item["conditions"] = _json_loads(item.get("conditions"), {})
            item["metadata"] = _json_loads(item.get("metadata"), {})
            result.append(item)
        if include_evidence:
            states_by_id = {int(state["id"]): state for state in result}
            for state in result:
                state["evidence_items"] = []
            for evidence in self.get_user_state_evidence_items(
                list(states_by_id), limit=None,
            ):
                states_by_id[int(evidence["state_id"])]["evidence_items"].append(evidence)
        return result

    def get_user_states_by_ids(self, state_ids: Sequence[int]) -> List[Dict[str, Any]]:
        ids = list(dict.fromkeys(
            int(value) for value in state_ids or []
            if str(value).strip().isdigit() and int(value) > 0
        ))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT * FROM user_states WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        by_id = {int(row["id"]): self._row_to_dict(row) for row in rows}
        result: List[Dict[str, Any]] = []
        for state_id in ids:
            row = by_id.get(state_id)
            if not row:
                continue
            row["conditions"] = _json_loads(row.get("conditions"), {})
            row["metadata"] = _json_loads(row.get("metadata"), {})
            result.append(row)
        return result

    def user_state_identity_exists(
        self, *, state_domain: str,
        state_type: str, state_kind: str, state_topic: str,
    ) -> bool:
        row = self._conn.execute(
            """
            SELECT 1 FROM user_states
            WHERE state_domain = ? AND state_type = ? AND state_kind = ?
              AND state_topic = ? LIMIT 1
            """,
            (
                str(state_domain), str(state_type),
                str(state_kind), str(state_topic),
            ),
        ).fetchone()
        return row is not None

    def get_user_state_evidence_items(
        self, state_ids: Sequence[int], *, evidence_type: Optional[str] = None,
        limit: Optional[int] = 480,
    ) -> List[Dict[str, Any]]:
        """Load evidence for selected states; limit=None returns all matching rows."""
        ids = sorted({
            int(value) for value in state_ids or []
            if str(value).strip().isdigit() and int(value) > 0
        })
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        type_filter = " AND evidence_type = ?" if evidence_type is not None else ""
        params: List[Any] = [*ids]
        if evidence_type is not None:
            params.append(str(evidence_type))
        limit_clause = " LIMIT ?" if limit is not None else ""
        if limit is not None:
            params.append(max(1, int(limit)))
        rows = self._conn.execute(
            "SELECT * FROM user_state_evidence "
            f"WHERE state_id IN ({placeholders}){type_filter} "
            f"ORDER BY state_id, evidence_type, evidence_id{limit_clause}",
            params,
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def replace_user_state_claim_evidence(
        self, *, state_id: int, evidence: Sequence[Dict[str, Any]],
    ) -> None:
        """Replace claim evidence after a full claim-only state resolution."""
        self._conn.execute(
            "DELETE FROM user_state_evidence WHERE state_id = ? AND evidence_type = 'user_claim'",
            (int(state_id),),
        )
        self.upsert_user_state_evidence([
            {**item, "state_id": int(state_id)} for item in evidence
        ])
        self._commit_if_needed()

    def upsert_user_state(
        self,
        *,
        state_domain: str,
        state_type: str,
        state_kind: str,
        state_topic: str,
        state_text: str,
        status: str,
        confidence: float,
        conditions: Optional[Dict[str, Any]] = None,
        valid_from: str = "",
        valid_to: str = "",
        last_observed_at: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        target_state_id: Optional[int] = None,
    ) -> tuple[int, bool, Optional[Dict[str, Any]]]:
        """Create or incrementally update one state and return its old value."""
        now = local_now_text()
        target_id = int(target_state_id or 0)
        topic = str(state_topic or "").strip()
        if not topic:
            raise ValueError("state_topic is required")
        existing = None
        if target_id > 0:
            existing = self._conn.execute(
                "SELECT * FROM user_states WHERE id = ?",
                (target_id,),
            ).fetchone()
            if existing is None:
                raise ValueError(f"Unknown target user state: {target_id}")
        if existing is None:
            existing = self._conn.execute(
                """
                SELECT * FROM user_states
                WHERE state_domain = ? AND state_type = ? AND state_kind = ?
                  AND state_topic = ?
                """,
                (
                    str(state_domain), str(state_type),
                    str(state_kind), topic,
                ),
            ).fetchone()
        old = self._row_to_dict(existing) if existing else None
        if old:
            if (
                old["state_domain"] != state_domain
                or old["state_type"] != state_type
                or old["state_kind"] != state_kind
                or old["state_topic"] != topic
            ):
                raise ValueError("Target user state identity cannot be changed")
            state_id = int(old["id"])
            self._conn.execute(
                """
                UPDATE user_states
                SET state_text = ?, status = ?, confidence = ?, conditions = ?,
                    valid_from = ?, valid_to = ?, last_observed_at = ?,
                    metadata = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    str(state_text or old.get("state_text") or ""),
                    str(status or old.get("status") or "active"),
                    float(confidence),
                    _json_dumps(conditions if isinstance(conditions, dict) else {}),
                    str(valid_from or old.get("valid_from") or ""),
                    str(valid_to or old.get("valid_to") or ""),
                    str(last_observed_at or old.get("last_observed_at") or ""),
                    _json_dumps(metadata if isinstance(metadata, dict) else {}),
                    now,
                    state_id,
                ),
            )
            self._commit_if_needed()
            return state_id, False, old
        cur = self._conn.execute(
            """
            INSERT INTO user_states (
                state_domain, state_type, state_kind,
                state_topic, state_text, status,
                confidence, conditions, valid_from, valid_to, last_observed_at,
                metadata, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(state_domain), str(state_type), str(state_kind),
                topic, str(state_text),
                str(status or "active"), float(confidence),
                _json_dumps(conditions if isinstance(conditions, dict) else {}),
                str(valid_from or ""), str(valid_to or ""),
                str(last_observed_at or ""),
                _json_dumps(metadata if isinstance(metadata, dict) else {}), now, now,
            ),
        )
        self._commit_if_needed()
        return int(cur.lastrowid), True, None

    def upsert_user_state_evidence(self, evidence: Sequence[Dict[str, Any]]) -> int:
        now = local_now_text()
        changed = 0
        for item in evidence or []:
            try:
                state_id = int(item.get("state_id") or 0)
                evidence_id = int(item.get("evidence_id") or 0)
            except (TypeError, ValueError):
                continue
            evidence_type = str(item.get("evidence_type") or "").strip()
            role = str(item.get("role") or "support").strip()
            if state_id <= 0 or evidence_id <= 0 or not evidence_type:
                continue
            self._conn.execute(
                """
                INSERT INTO user_state_evidence (
                    state_id, evidence_type, evidence_id, role, weight,
                    observed_at, details, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(state_id, evidence_type, evidence_id, role)
                DO UPDATE SET weight = excluded.weight,
                              observed_at = excluded.observed_at,
                              details = excluded.details,
                              updated_at = excluded.updated_at
                """,
                (
                    state_id, evidence_type, evidence_id, role,
                    float(item.get("weight") or 0.0),
                    str(item.get("observed_at") or ""),
                    _json_dumps(item.get("details") or {}), now, now,
                ),
            )
            changed += 1
        self._commit_if_needed()
        return changed

    def record_user_state_event(
        self,
        *,
        state_id: int,
        operation: str,
        previous_payload: Optional[Dict[str, Any]],
        new_payload: Optional[Dict[str, Any]],
        trigger_claim_ids: Sequence[int],
        trigger_intent_objects: Sequence[Dict[str, Any]],
        effective_at: str,
        decision_source: str,
        confidence: float,
        reason: str,
    ) -> int:
        normalized_claim_ids: List[int] = []
        for value in trigger_claim_ids or []:
            try:
                claim_id = int(value)
            except (TypeError, ValueError):
                continue
            if claim_id > 0:
                normalized_claim_ids.append(claim_id)
        cur = self._conn.execute(
            """
            INSERT INTO user_state_events (
                state_id, operation, previous_payload, new_payload,
                trigger_claim_ids, trigger_intent_objects, effective_at,
                decision_source, confidence, reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(state_id), str(operation), _json_dumps(previous_payload or {}),
                _json_dumps(new_payload or {}),
                _json_dumps(sorted(set(normalized_claim_ids))),
                _json_dumps(list(trigger_intent_objects or [])), str(effective_at or ""),
                str(decision_source or ""), float(confidence), str(reason or ""),
                local_now_text(),
            ),
        )
        self._commit_if_needed()
        return int(cur.lastrowid)

