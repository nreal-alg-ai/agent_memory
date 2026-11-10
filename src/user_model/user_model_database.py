"""SQLite projection store for durable user states and hypotheses.

The memory database owns evidence; cross-database evidence IDs here are logical
references, not SQLite foreign keys. A single priority worker orders updates.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from storage_utils import json_dumps, json_loads, local_now_text


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
            CREATE TABLE IF NOT EXISTS user_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_entity_id INTEGER NOT NULL,
                prediction_key TEXT NOT NULL,
                behavior_text TEXT NOT NULL,
                window_start TEXT NOT NULL,
                window_end TEXT NOT NULL,
                conditions TEXT NOT NULL DEFAULT '[]',
                likelihood_band TEXT NOT NULL,
                uncertainties TEXT NOT NULL DEFAULT '[]',
                rationale TEXT NOT NULL DEFAULT '',
                basis TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                resolved_at TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS user_prediction_evidence (
                prediction_id INTEGER NOT NULL,
                intent_id INTEGER NOT NULL CHECK(intent_id > 0),
                created_at TEXT NOT NULL,
                PRIMARY KEY(prediction_id, intent_id),
                FOREIGN KEY(prediction_id) REFERENCES user_predictions(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS user_prediction_reassessment_schedule (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intent_id INTEGER NOT NULL CHECK(intent_id > 0),
                user_entity_id INTEGER NOT NULL,
                next_evaluate_at TEXT NOT NULL,
                reason TEXT NOT NULL,
                evaluated_at TEXT NOT NULL,
                intent_event_ids TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                CHECK(julianday(next_evaluate_at) IS NOT NULL
                      AND julianday(evaluated_at) IS NOT NULL
                      AND julianday(next_evaluate_at) > julianday(evaluated_at))
            );
            CREATE TABLE IF NOT EXISTS user_proactive_services (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                user_entity_id INTEGER NOT NULL,
                service_key TEXT NOT NULL,
                service_type TEXT NOT NULL CHECK(service_type IN (
                    'notify', 'suggest', 'clarify', 'offer_assistance', 'prepare', 'delegate'
                )),
                objective TEXT NOT NULL,
                agent_guidance TEXT NOT NULL,
                constraints TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL CHECK(status IN ('ready', 'waiting', 'cancelled', 'expired')),
                reason TEXT NOT NULL,
                next_evaluate_at TEXT,
                recheck_condition TEXT NOT NULL DEFAULT '',
                expires_at TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                evaluated_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(prediction_id, service_key),
                FOREIGN KEY(prediction_id) REFERENCES user_predictions(id) ON DELETE CASCADE,
                CHECK(julianday(expires_at) IS NOT NULL),
                CHECK(next_evaluate_at IS NULL OR (
                    julianday(next_evaluate_at) > julianday(evaluated_at)
                    AND julianday(next_evaluate_at) < julianday(expires_at)
                )),
                CHECK(status != 'ready' OR next_evaluate_at IS NULL)
            );
            CREATE TABLE IF NOT EXISTS user_prediction_proactive_service_mapping (
                prediction_id INTEGER PRIMARY KEY,
                proactive_service_ids TEXT NOT NULL DEFAULT '[]',
                evaluated_at TEXT,
                FOREIGN KEY(prediction_id) REFERENCES user_predictions(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_prediction_proactive_service_pending
            ON user_prediction_proactive_service_mapping(evaluated_at, prediction_id);
            CREATE TABLE IF NOT EXISTS user_proactive_service_feedback_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                service_id INTEGER NOT NULL,
                execution_id TEXT NOT NULL,
                execution_status TEXT NOT NULL CHECK(execution_status IN (
                    'not_executed', 'presented', 'running', 'succeeded', 'failed', 'cancelled'
                )),
                user_response TEXT NOT NULL CHECK(user_response IN (
                    'unknown', 'accepted', 'declined', 'deferred', 'not_needed'
                )),
                user_feedback TEXT NOT NULL DEFAULT '',
                execution_result TEXT NOT NULL DEFAULT '{}',
                agent_note TEXT NOT NULL DEFAULT '',
                context TEXT NOT NULL DEFAULT '{}',
                observed_at TEXT NOT NULL,
                received_at TEXT NOT NULL,
                FOREIGN KEY(service_id) REFERENCES user_proactive_services(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_proactive_feedback_service
            ON user_proactive_service_feedback_events(service_id, observed_at, id);
            CREATE INDEX IF NOT EXISTS idx_user_states_domain_type_status
            ON user_states(state_domain, state_type, status, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_user_state_evidence_state
            ON user_state_evidence(state_id, role, evidence_type);
            CREATE INDEX IF NOT EXISTS idx_user_state_evidence_claim
            ON user_state_evidence(evidence_type, evidence_id, role, state_id);
            CREATE INDEX IF NOT EXISTS idx_user_state_events_state
            ON user_state_events(state_id, effective_at DESC, id DESC);
            CREATE INDEX IF NOT EXISTS idx_user_predictions_active
            ON user_predictions(user_entity_id, status, window_end);
            CREATE INDEX IF NOT EXISTS idx_user_prediction_evidence_intent
            ON user_prediction_evidence(intent_id, prediction_id);
            CREATE INDEX IF NOT EXISTS idx_user_prediction_reassessment_due
            ON user_prediction_reassessment_schedule(julianday(next_evaluate_at), intent_id);
            CREATE INDEX IF NOT EXISTS idx_user_proactive_services_user_status
            ON user_proactive_services(user_entity_id, status, evaluated_at DESC);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_user_predictions_write_key
            ON user_predictions(user_entity_id, prediction_key);
        """)
        self._commit_if_needed()

    def upsert_user_prediction_reassessment_schedule(
        self, *, intent_id: int, user_entity_id: int, next_evaluate_at: str,
        reason: str, evaluated_at: str, intent_event_ids: Sequence[int],
        assessment_id: Optional[int] = None,
    ) -> int:
        """Create a deferred direction, or update the same record by its stable ID."""
        event_ids = sorted({int(value) for value in intent_event_ids})
        if (int(intent_id) <= 0 or int(user_entity_id) <= 0 or not event_ids
                or any(value <= 0 for value in event_ids) or not str(reason).strip()
                or (assessment_id is not None and (type(assessment_id) is not int or assessment_id <= 0))):
            raise ValueError("Invalid prediction reassessment schedule")
        with self.transaction():
            existing = self._conn.execute(
                "SELECT user_entity_id FROM user_prediction_reassessment_schedule WHERE intent_id = ? LIMIT 1",
                (int(intent_id),),
            ).fetchone()
            if existing and existing["user_entity_id"] != int(user_entity_id):
                raise ValueError("Cannot replace another user's reassessment schedule")
            now = local_now_text()
            if assessment_id is not None:
                cursor = self._conn.execute(
                    """UPDATE user_prediction_reassessment_schedule SET next_evaluate_at = ?, reason = ?,
                        evaluated_at = ?, intent_event_ids = ?, updated_at = ?
                        WHERE id = ? AND intent_id = ? AND user_entity_id = ?""",
                    (str(next_evaluate_at), str(reason).strip(), str(evaluated_at), json_dumps(event_ids), now,
                     assessment_id, int(intent_id), int(user_entity_id)),
                )
                if cursor.rowcount != 1:
                    raise ValueError("Missing or foreign reassessment schedule")
                return assessment_id
            cursor = self._conn.execute(
                """INSERT INTO user_prediction_reassessment_schedule (
                    intent_id, user_entity_id, next_evaluate_at, reason, evaluated_at,
                    intent_event_ids, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (int(intent_id), int(user_entity_id), str(next_evaluate_at), str(reason).strip(),
                 str(evaluated_at), json_dumps(event_ids), now, now),
            )
            return int(cursor.lastrowid)

    def get_user_prediction_reassessment_schedules(
        self, *, intent_ids: Optional[Sequence[int]] = None,
        user_entity_id: Optional[int] = None, due_at: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Read deferred assessments; due_at only filters, it does not submit tasks."""
        ids = None if intent_ids is None else sorted({int(value) for value in intent_ids if int(value) > 0})
        chunks = [None] if ids is None else [ids[offset:offset + 500] for offset in range(0, len(ids), 500)]
        schedules = []
        for chunk in chunks:
            clauses, params = [], []
            if chunk is not None:
                clauses.append(f"intent_id IN ({','.join('?' for _ in chunk)})")
                params.extend(chunk)
            if user_entity_id is not None:
                clauses.append("user_entity_id = ?")
                params.append(int(user_entity_id))
            if due_at is not None:
                clauses.append("julianday(next_evaluate_at) <= julianday(?)")
                params.append(str(due_at))
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            rows = self._conn.execute(
                f"SELECT *, julianday(next_evaluate_at) AS due_order "
                f"FROM user_prediction_reassessment_schedule {where}", params,
            ).fetchall()
            for row in rows:
                item = dict(row)
                item["intent_event_ids"] = json_loads(item["intent_event_ids"], [])
                schedules.append(item)
        schedules.sort(key=lambda item: (item["due_order"], item["intent_id"], item["id"]))
        for item in schedules:
            item.pop("due_order")
        return schedules

    def delete_user_prediction_reassessment_schedules(
        self, intent_ids: Sequence[int], *, user_entity_id: int,
    ) -> int:
        ids = sorted({int(value) for value in intent_ids if int(value) > 0})
        deleted = 0
        with self.transaction():
            for offset in range(0, len(ids), 500):
                chunk = ids[offset:offset + 500]
                cursor = self._conn.execute(
                    f"DELETE FROM user_prediction_reassessment_schedule "
                    f"WHERE intent_id IN ({','.join('?' for _ in chunk)}) AND user_entity_id = ?",
                    [*chunk, int(user_entity_id)],
                )
                deleted += cursor.rowcount
        return deleted

    def delete_user_prediction_reassessment_schedule(
        self, *, assessment_id: int, user_entity_id: int,
    ) -> int:
        """Consume one assessment without clearing other behaviors of the intent."""
        cursor = self._conn.execute(
            "DELETE FROM user_prediction_reassessment_schedule "
            "WHERE id = ? AND user_entity_id = ?",
            (int(assessment_id), int(user_entity_id)),
        )
        self._commit_if_needed()
        return cursor.rowcount

    def save_user_prediction(self, payload: Dict[str, Any]) -> Tuple[int, str]:
        """Deduplicate an exact write across all statuses; never refresh an existing forecast."""
        owner_id = int(payload["user_entity_id"])
        prediction_key = str(payload["prediction_key"])
        basis = payload.get("basis") or {}
        with self.transaction():
            existing = self._conn.execute(
                """SELECT * FROM user_predictions
                WHERE user_entity_id = ? AND prediction_key = ?""",
                (owner_id, prediction_key),
            ).fetchone()
            forecast_fields = {
                "behavior_text": str(payload["behavior_text"]),
                "window_start": str(payload["window_start"]),
                "window_end": str(payload["window_end"]),
                "conditions": payload.get("conditions") or [],
                "likelihood_band": str(payload["likelihood_band"]),
                "uncertainties": payload.get("uncertainties") or [],
                "rationale": str(payload.get("rationale") or ""),
            }
            existing_payload = self._prediction_row_to_api(existing) if existing else {}
            if existing:
                if (not all(existing_payload[key] == value for key, value in forecast_fields.items())
                        or existing_payload["basis"].get("intent", {}).get("id") != basis.get("intent", {}).get("id")):
                    raise ValueError("Prediction write key reused for different forecast content")
                prediction_id, save_action = int(existing["id"]), "reused"
                basis = json_loads(existing["basis"], {})
            else:
                cursor = self._conn.execute(
                    """INSERT INTO user_predictions (
                        user_entity_id, prediction_key, behavior_text,
                        window_start, window_end, conditions, likelihood_band,
                        uncertainties, rationale, basis, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        owner_id, prediction_key,
                        str(payload["behavior_text"]), str(payload["window_start"]),
                        str(payload["window_end"]),
                        json_dumps(payload.get("conditions") or []),
                        str(payload["likelihood_band"]),
                        json_dumps(payload.get("uncertainties") or []),
                        str(payload.get("rationale") or ""),
                        json_dumps(payload.get("basis") or {}), local_now_text(),
                    ),
                )
                prediction_id, save_action = int(cursor.lastrowid), "created"
                self._conn.execute(
                    "INSERT INTO user_prediction_proactive_service_mapping(prediction_id) VALUES (?)",
                    (prediction_id,),
                )
            intent = basis.get("intent")
            if intent is not None:
                self.upsert_user_prediction_evidence([{
                    "prediction_id": prediction_id, "intent_id": intent["id"],
                }])
        return prediction_id, save_action

    def upsert_user_prediction_evidence(self, evidence: Sequence[Dict[str, Any]]) -> int:
        """Link predictions to memory-db intents through logical IDs, without duplicates."""
        added = 0
        with self.transaction():
            for item in evidence:
                prediction_id, intent_id = int(item["prediction_id"]), int(item["intent_id"])
                if prediction_id <= 0 or intent_id <= 0:
                    raise ValueError("Prediction evidence IDs must be positive")
                cursor = self._conn.execute(
                    """INSERT OR IGNORE INTO user_prediction_evidence
                    (prediction_id, intent_id, created_at) VALUES (?, ?, ?)""",
                    (prediction_id, intent_id, local_now_text()),
                )
                added += cursor.rowcount
        return added

    def get_user_prediction_evidence(
        self, prediction_ids: Sequence[int],
    ) -> List[Dict[str, Any]]:
        ids = sorted({int(value) for value in prediction_ids if int(value) > 0})
        evidence: List[Dict[str, Any]] = []
        for offset in range(0, len(ids), 500):
            chunk = ids[offset:offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            evidence.extend(dict(row) for row in self._conn.execute(
                "SELECT * FROM user_prediction_evidence "
                f"WHERE prediction_id IN ({placeholders}) ORDER BY prediction_id, intent_id",
                chunk,
            ))
        return evidence

    def get_user_predictions_by_intent_ids(
        self, intent_ids: Sequence[int], *,
        statuses: Optional[Sequence[str]] = None,
        user_entity_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Read linked forecasts, including invalid or expired ones unless filtered."""
        ids = sorted({int(value) for value in intent_ids if int(value) > 0})
        predictions: Dict[int, Dict[str, Any]] = {}
        for offset in range(0, len(ids), 500):
            chunk = ids[offset:offset + 500]
            clauses = [f"evidence.intent_id IN ({','.join('?' for _ in chunk)})"]
            params: List[Any] = list(chunk)
            if statuses:
                clauses.append(f"prediction.status IN ({','.join('?' for _ in statuses)})")
                params.extend(statuses)
            if user_entity_id is not None:
                clauses.append("prediction.user_entity_id = ?")
                params.append(int(user_entity_id))
            rows = self._conn.execute(
                "SELECT DISTINCT prediction.* FROM user_predictions AS prediction "
                "JOIN user_prediction_evidence AS evidence ON evidence.prediction_id = prediction.id "
                f"WHERE {' AND '.join(clauses)} ORDER BY prediction.id", params,
            ).fetchall()
            for row in rows:
                predictions[int(row["id"])] = self._prediction_row_to_api(row)
        return [predictions[prediction_id] for prediction_id in sorted(predictions)]

    def expire_user_predictions(
        self, *, user_entity_id: int, as_of: str,
    ) -> Dict[str, List[int]]:
        """Expire ended forecasts and close their unprovided services atomically.

        Preserve forecast content and evidence, and also close any open services
        of already-expired forecasts so repeated maintenance converges safely.
        """
        owner_id = int(user_entity_id)
        if owner_id <= 0 or self._conn.execute(
            "SELECT julianday(?)", (as_of,),
        ).fetchone()[0] is None:
            raise ValueError("Invalid prediction expiration owner or reference time")
        due_query = """SELECT id FROM user_predictions
            WHERE user_entity_id = ? AND status = 'pending'
              AND julianday(window_end) <= julianday(?)"""
        service_source_query = """SELECT id FROM user_predictions
            WHERE user_entity_id = ? AND (
                status = 'expired' OR (status = 'pending'
                    AND julianday(window_end) <= julianday(?))
            )"""
        service_filter = f"""user_entity_id = ? AND status IN ('ready', 'waiting')
            AND prediction_id IN ({service_source_query})"""
        service_params = (owner_id, owner_id, as_of)
        with self.transaction():
            prediction_ids = [int(row["id"]) for row in self._conn.execute(
                f"{due_query} ORDER BY id", (owner_id, as_of),
            ).fetchall()]
            service_ids = [int(row["id"]) for row in self._conn.execute(
                f"SELECT id FROM user_proactive_services WHERE {service_filter} ORDER BY id",
                service_params,
            ).fetchall()]
            # Close services before changing prediction status; both writes
            # share this transaction and neither invokes policy generation.
            self._conn.execute(
                "UPDATE user_proactive_services SET status = 'expired', "
                "reason = ?, next_evaluate_at = NULL, recheck_condition = '', "
                f"evaluated_at = ?, updated_at = ? WHERE {service_filter}",
                ("Source prediction window ended", as_of, local_now_text(), *service_params),
            )
            self._conn.execute(
                "UPDATE user_predictions SET status = 'expired', resolved_at = ? "
                f"WHERE id IN ({due_query})",
                (as_of, owner_id, as_of),
            )
        return {"expired_prediction_ids": prediction_ids, "expired_service_ids": service_ids}

    def invalidate_user_predictions(
        self, prediction_ids: Sequence[int], *, user_entity_id: int, as_of: Optional[str] = None,
    ) -> List[int]:
        """Invalidate only specified pending forecasts, preserving content and basis."""
        ids = sorted({int(value) for value in prediction_ids if int(value) > 0})
        invalidated = []
        with self.transaction():
            for offset in range(0, len(ids), 500):
                chunk = ids[offset:offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                params = [int(user_entity_id), *chunk]
                rows = self._conn.execute(
                    "SELECT id FROM user_predictions WHERE user_entity_id = ? "
                    f"AND id IN ({placeholders}) AND status = 'pending' ORDER BY id",
                    params,
                ).fetchall()
                self._conn.execute(
                    "UPDATE user_predictions SET status = 'invalidated', resolved_at = ? "
                    f"WHERE user_entity_id = ? AND id IN ({placeholders}) AND status = 'pending'",
                    [as_of or local_now_text(), *params],
                )
                invalidated.extend(int(row["id"]) for row in rows)
        return invalidated

    @staticmethod
    def _prediction_row_to_api(row: sqlite3.Row) -> Dict[str, Any]:
        item = dict(row)
        for field, fallback in (("conditions", []), ("uncertainties", []), ("basis", {})):
            item[field] = json_loads(item.get(field), fallback)
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

    def get_user_prediction_proactive_service_mapping(
        self, *, prediction_id: int, user_entity_id: int,
    ) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT mapping.* FROM user_prediction_proactive_service_mapping AS mapping "
            "JOIN user_predictions AS prediction ON prediction.id = mapping.prediction_id "
            "WHERE mapping.prediction_id = ? AND prediction.user_entity_id = ?",
            (prediction_id, user_entity_id),
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["proactive_service_ids"] = json_loads(result["proactive_service_ids"], [])
        return result

    def get_pending_proactive_prediction_ids(self, *, user_entity_id: int) -> List[int]:
        """Read unevaluated or waiting-service sources after lifecycle cleanup.

        Waiting sources precede new forecasts, then nearer window endings first.
        Effective-prediction checks still belong to the prediction reader.
        """
        rows = self._conn.execute(
            """WITH candidates AS (
                SELECT mapping.prediction_id, mapping.evaluated_at, prediction.window_end,
                    EXISTS (
                        SELECT 1 FROM user_proactive_services AS service
                        WHERE service.prediction_id = mapping.prediction_id
                            AND service.user_entity_id = prediction.user_entity_id
                            AND service.status = 'waiting'
                    ) AS has_waiting_service
                FROM user_prediction_proactive_service_mapping AS mapping
                JOIN user_predictions AS prediction ON prediction.id = mapping.prediction_id
                WHERE prediction.user_entity_id = ? AND prediction.status = 'pending'
            )
            SELECT prediction_id FROM candidates
            WHERE evaluated_at IS NULL OR has_waiting_service
            ORDER BY has_waiting_service DESC, julianday(window_end), prediction_id""",
            (int(user_entity_id),),
        ).fetchall()
        return [int(row["prediction_id"]) for row in rows]

    def mark_user_prediction_proactive_service_evaluated(
        self, *, prediction_id: int, user_entity_id: int, evaluated_at: str,
    ) -> None:
        """Mark a successful decision, including one producing no services."""
        cursor = self._conn.execute(
            "UPDATE user_prediction_proactive_service_mapping SET evaluated_at = ? "
            "WHERE prediction_id = ? AND prediction_id IN ("
            "SELECT id FROM user_predictions WHERE user_entity_id = ?)",
            (evaluated_at, int(prediction_id), int(user_entity_id)),
        )
        if cursor.rowcount != 1:
            raise ValueError("No proactive service mapping for this user's prediction")
        self._commit_if_needed()

    def save_user_proactive_service(
        self, payload: Dict[str, Any],
    ) -> Tuple[int, str]:
        """Create or update a service, never a delivery or execution record."""
        prediction_id, owner_id = int(payload["prediction_id"]), int(payload["user_entity_id"])
        with self.transaction():
            prediction = self._conn.execute(
                "SELECT user_entity_id FROM user_predictions WHERE id = ?", (prediction_id,),
            ).fetchone()
            if not prediction or prediction["user_entity_id"] != owner_id:
                raise ValueError("Service must reference the same owner's prediction")
            service_id = payload.get("service_id")
            if service_id is not None:
                existing = self._conn.execute(
                    "SELECT * FROM user_proactive_services WHERE id = ?", (int(service_id),),
                ).fetchone()
                if (not existing or existing["prediction_id"] != prediction_id
                        or existing["user_entity_id"] != owner_id):
                    raise ValueError("Service update references an unrelated service")
            else:
                existing = self._conn.execute(
                    "SELECT * FROM user_proactive_services WHERE prediction_id = ? AND service_key = ?",
                    (prediction_id, payload["service_key"]),
                ).fetchone()
            fields = {
                key: payload[key] for key in (
                    "service_key", "service_type", "objective", "agent_guidance", "status", "reason",
                    "next_evaluate_at", "recheck_condition", "expires_at", "policy_version",
                    "evaluated_at",
                )
            }
            fields["constraints"] = json_dumps(payload["constraints"])
            now = local_now_text()
            if existing:
                if existing["status"] in {"cancelled", "expired"}:
                    raise ValueError("Closed services cannot be silently reactivated")
                if all(existing[key] == value for key, value in fields.items()):
                    service_id, action = int(existing["id"]), "reused"
                else:
                    self._conn.execute(
                        f"UPDATE user_proactive_services SET {', '.join(key + ' = ?' for key in fields)}, "
                        "updated_at = ? WHERE id = ?",
                        [*fields.values(), now, int(existing["id"])],
                    )
                    service_id, action = int(existing["id"]), "updated"
            else:
                fields = {"prediction_id": prediction_id, "user_entity_id": owner_id, **fields,
                          "created_at": now, "updated_at": now}
                cursor = self._conn.execute(
                    f"INSERT INTO user_proactive_services ({', '.join(fields)}) "
                    f"VALUES ({','.join('?' for _ in fields)})", list(fields.values()),
                )
                service_id, action = int(cursor.lastrowid), "created"
            mapping = self.get_user_prediction_proactive_service_mapping(
                prediction_id=prediction_id, user_entity_id=owner_id,
            )
            if mapping is None:
                raise ValueError("Prediction has no proactive service mapping")
            if service_id not in mapping["proactive_service_ids"]:
                self._conn.execute(
                    "UPDATE user_prediction_proactive_service_mapping SET proactive_service_ids = ? "
                    "WHERE prediction_id = ?",
                    (json_dumps([*mapping["proactive_service_ids"], service_id]), prediction_id),
                )
            return service_id, action

    def get_user_proactive_service(self, service_id: int) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM user_proactive_services WHERE id = ?", (int(service_id),),
        ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["constraints"] = json_loads(item["constraints"], [])
        return item

    def insert_user_proactive_service_feedback(self, payload: Dict[str, Any]) -> int:
        """Append a report, not a prediction verdict or an intent completion."""
        cursor = self._conn.execute(
            """INSERT INTO user_proactive_service_feedback_events (
                service_id, execution_id, execution_status, user_response,
                user_feedback, execution_result, agent_note, context, observed_at, received_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (payload["service_id"], payload["execution_id"], payload["execution_status"],
             payload["user_response"], payload["user_feedback"],
             json_dumps(payload["execution_result"]), payload["agent_note"],
             json_dumps(payload["context"]), payload["observed_at"], local_now_text()),
        )
        self._commit_if_needed()
        return int(cursor.lastrowid)

    def get_user_proactive_service_feedback(
        self, *, user_entity_id: int, service_id: Optional[int] = None,
        prediction_id: Optional[int] = None, observed_before: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        clauses, params = ["service.user_entity_id = ?"], [int(user_entity_id)]
        if service_id is not None:
            clauses.append("feedback.service_id = ?")
            params.append(int(service_id))
        if prediction_id is not None:
            clauses.append("service.prediction_id = ?")
            params.append(int(prediction_id))
        if observed_before is not None:
            clauses.append("julianday(feedback.observed_at) <= julianday(?)")
            params.append(observed_before)
        params.append(max(1, int(limit)))
        rows = self._conn.execute(
            "SELECT feedback.* FROM user_proactive_service_feedback_events AS feedback "
            "JOIN user_proactive_services AS service ON service.id = feedback.service_id "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY julianday(feedback.observed_at) DESC, feedback.id DESC LIMIT ?", params,
        ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            for key in ("execution_result", "context"):
                item[key] = json_loads(item[key], {})
            items.append(item)
        return items

    def get_user_proactive_services(
        self, *, user_entity_id: Optional[int] = None, limit: Optional[int] = 100,
        prediction_id: Optional[int] = None, statuses: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        clauses = []
        params: List[Any] = []
        if user_entity_id is not None:
            clauses.append("user_entity_id = ?")
            params.append(int(user_entity_id))
        if prediction_id is not None:
            clauses.append("prediction_id = ?")
            params.append(int(prediction_id))
        if statuses:
            clauses.append(f"status IN ({','.join('?' for _ in statuses)})")
            params.extend(statuses)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_sql = ""
        if limit is not None:
            limit_sql = " LIMIT ?"
            params.append(max(1, int(limit)))
        rows = self._conn.execute(
            f"SELECT * FROM user_proactive_services {where} "
            f"ORDER BY evaluated_at DESC, id DESC{limit_sql}", params,
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["constraints"] = json_loads(item.get("constraints"), [])
            result.append(item)
        return result

    def close_user_proactive_services(
        self, *, prediction_id: int, user_entity_id: int, status: str, reason: str,
        as_of: str, expired_only: bool = False,
    ) -> List[int]:
        """Close unprovided services without rewriting their original content."""
        if status not in {"cancelled", "expired"}:
            raise ValueError("Unsupported closed service status")
        clauses = ["prediction_id = ?", "user_entity_id = ?", "status IN ('ready', 'waiting')"]
        params: List[Any] = [int(prediction_id), int(user_entity_id)]
        if expired_only:
            clauses.append("julianday(expires_at) <= julianday(?)")
            params.append(str(as_of))
        where = " AND ".join(clauses)
        with self.transaction():
            rows = self._conn.execute(f"SELECT id FROM user_proactive_services WHERE {where}", params).fetchall()
            self._conn.execute(
                "UPDATE user_proactive_services SET status = ?, reason = ?, next_evaluate_at = NULL, recheck_condition = '', "
                f"evaluated_at = ?, updated_at = ? WHERE {where}",
                [status, str(reason), str(as_of), local_now_text(), *params],
            )
        return [int(row["id"]) for row in rows]

    def get_user_states(
        self,
        *,
        state_domains: Optional[Sequence[str]] = None,
        state_types: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[str]] = None,
        limit: int = 160,
        include_evidence: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return user states, optionally attaching all of each state's evidence."""
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
            item["conditions"] = json_loads(item.get("conditions"), {})
            item["metadata"] = json_loads(item.get("metadata"), {})
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
            row["conditions"] = json_loads(row.get("conditions"), {})
            row["metadata"] = json_loads(row.get("metadata"), {})
            result.append(row)
        return result

    def user_state_identity_exists(
        self, *, state_domain: str,
        state_type: str, state_kind: str, state_topic: str,
    ) -> bool:
        return self.get_user_state_by_identity(
            state_domain=state_domain, state_type=state_type,
            state_kind=state_kind, state_topic=state_topic,
        ) is not None

    def get_user_state_by_identity(
        self, *, state_domain: str,
        state_type: str, state_kind: str, state_topic: str,
    ) -> Optional[Dict[str, Any]]:
        """Return identity/status for conflict diagnostics, including expired states."""
        row = self._conn.execute(
            """
            SELECT id, status, state_domain, state_type, state_kind, state_topic FROM user_states
            WHERE state_domain = ? AND state_type = ? AND state_kind = ?
              AND state_topic = ? LIMIT 1
            """,
            (
                str(state_domain), str(state_type),
                str(state_kind), str(state_topic),
            ),
        ).fetchone()
        return dict(row) if row is not None else None

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

    def get_user_states_for_thread_ids(
        self, thread_ids: Sequence[int],
    ) -> List[Dict[str, Any]]:
        """Keep linked active situational states even outside the recent-state limit."""
        ids = sorted({int(value) for value in thread_ids if int(value) > 0})
        state_ids = set()
        for start in range(0, len(ids), 400):
            chunk = ids[start:start + 400]
            rows = self._conn.execute(
                "SELECT DISTINCT s.id FROM user_states s "
                "JOIN user_state_evidence e ON e.state_id=s.id "
                "WHERE s.state_type='situational_state' "
                "AND s.status IN ('active','weakened') "
                "AND e.evidence_type='user_thread' AND e.evidence_id IN ("
                + ','.join('?' for _ in chunk) + ')', chunk,
            ).fetchall()
            state_ids.update(int(row['id']) for row in rows)
        states = self.get_user_states_by_ids(sorted(state_ids))
        evidence_by_state: Dict[int, List[Dict[str, Any]]] = {}
        for item in self.get_user_state_evidence_items(sorted(state_ids), limit=None):
            evidence_by_state.setdefault(int(item['state_id']), []).append(item)
        for state in states:
            state['evidence_items'] = evidence_by_state.get(int(state['id']), [])
        return states

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
                    json_dumps(conditions if isinstance(conditions, dict) else {}),
                    str(valid_from or old.get("valid_from") or ""),
                    str(valid_to or old.get("valid_to") or ""),
                    str(last_observed_at or old.get("last_observed_at") or ""),
                    json_dumps(metadata if isinstance(metadata, dict) else {}),
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
                json_dumps(conditions if isinstance(conditions, dict) else {}),
                str(valid_from or ""), str(valid_to or ""),
                str(last_observed_at or ""),
                json_dumps(metadata if isinstance(metadata, dict) else {}), now, now,
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
                    json_dumps(item.get("details") or {}), now, now,
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
                int(state_id), str(operation), json_dumps(previous_payload or {}),
                json_dumps(new_payload or {}),
                json_dumps(sorted(set(normalized_claim_ids))),
                json_dumps(list(trigger_intent_objects or [])), str(effective_at or ""),
                str(decision_source or ""), float(confidence), str(reason or ""),
                local_now_text(),
            ),
        )
        self._commit_if_needed()
        return int(cur.lastrowid)

