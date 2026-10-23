"""Evidence-linked, near-term forecasts of user preparation behavior."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from prompts_en import USER_PREDICTION_NEAR_TERM_PREPARATION_PROMPT_EN
from prompts_zh import USER_PREDICTION_NEAR_TERM_PREPARATION_PROMPT_ZH

if TYPE_CHECKING:
    from llm_client import LLMClient
    from memory.memory_database import UserMemoryDB
    from task_scheduler import TaskScheduler
    from .user_model_database import UserModelDB


class UserPrediction:
    """Generate, inspect and verify preparation forecasts without changing intent."""

    TASK_KIND = "user_prediction_near_term_preparation"
    MODEL_VERSION = "near_term_preparation_v1"

    def __init__(
        self, *, db: UserMemoryDB, model_db: UserModelDB, llm_client: LLMClient,
        task_scheduler: TaskScheduler,
        world_owner_entity_name: str = "用户本人",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._db = db
        self._model_db = model_db
        self._llm_client = llm_client
        self._task_scheduler = task_scheduler
        self._owner_name = world_owner_entity_name
        self._logger = logger or logging.getLogger(__name__)
        task_scheduler.register_handler(self.TASK_KIND, self.predict_near_term_preparation)
        task_scheduler.register_prediction_candidate_provider(self.due_intents)

    @staticmethod
    def _as_datetime(value: Any, reference: datetime) -> Optional[datetime]:
        if isinstance(value, datetime):
            result = value
        else:
            text = str(value or "").strip()
            if not text:
                return None
            try:
                result = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                return None
        if result.tzinfo is None:
            result = result.replace(tzinfo=reference.tzinfo)
        return result.astimezone(reference.tzinfo)

    @staticmethod
    def _now(as_of: Optional[datetime]) -> datetime:
        local = datetime.now().astimezone()
        if as_of is None:
            return local
        if as_of.tzinfo is None:
            return as_of.replace(tzinfo=local.tzinfo)
        return as_of

    @staticmethod
    def _hash(value: Any) -> str:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _text_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()][:8]

    @staticmethod
    def _intent_anchor_text(intent: Dict[str, Any]) -> str:
        object_type = intent.get("object_type")
        if object_type == "plan":
            return str(intent.get("start_at") or "")
        if object_type == "work_item":
            return str(intent.get("due_at") or "")
        return str(intent.get("target_at") or "")

    @classmethod
    def _intent_fingerprint(cls, intent: Dict[str, Any]) -> str:
        return cls._hash({
            "object_type": intent.get("object_type"), "id": intent.get("id"),
            "status": intent.get("status"), "summary": intent.get("summary"),
            "anchor": cls._intent_anchor_text(intent),
            "updated_at": intent.get("updated_at"),
        })

    def due_intents(
        self, *, as_of: datetime, horizon_hours: float = 6.0,
        intent_refs: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Select precise, user-owned intent anchors in the rolling window."""
        owner_id = self._db.get_entity_id_by_name(self._owner_name)
        if owner_id is None:
            return []
        requested = None if intent_refs is None else {
            (str(ref.get("object_type") or ""), int(ref.get("object_id") or 0)):
            str(ref.get("fingerprint") or "")
            for ref in intent_refs if isinstance(ref, dict)
        }
        end = as_of + timedelta(hours=horizon_hours)
        intents = []
        for intent in self._db.get_time_bearing_user_intents(
            world_owner_entity_id=owner_id,
        ):
            identity = (str(intent["object_type"]), int(intent["id"]))
            if requested is not None and identity not in requested:
                continue
            fingerprint = self._intent_fingerprint(intent)
            if (requested is not None and requested[identity]
                    and requested[identity] != fingerprint):
                continue
            anchor_text = self._intent_anchor_text(intent)
            # Date-only and day/week precision must not become invented midnight deadlines.
            if not re.search(r"[T\s]\d{2}:\d{2}", anchor_text):
                continue
            if intent["object_type"] == "plan" and intent.get("time_precision") in {
                "day", "week", "unknown",
            }:
                continue
            anchor = self._as_datetime(anchor_text, as_of)
            if anchor is None or not as_of < anchor <= end:
                continue
            intents.append({
                "object_type": identity[0], "id": identity[1],
                "summary": intent.get("summary") or "",
                "anchor_at": anchor.isoformat(),
                "primary_domain": intent.get("primary_domain") or "general",
                "status": intent.get("status") or "",
                "details": intent.get("details") or {},
                "updated_at": intent.get("updated_at") or "",
                "fingerprint": fingerprint,
            })
        return intents

    def _input_snapshot(
        self, *, owner_id: int, as_of: datetime,
        horizon_end: datetime, intent_refs: Optional[Sequence[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        intents = self.due_intents(
            as_of=as_of, horizon_hours=(horizon_end - as_of).total_seconds() / 3600,
            intent_refs=intent_refs,
        )

        states = self._model_db.get_user_states(
            state_types=["recurring_state"],
            statuses=["active"], limit=500,
        )
        active_states = []
        for state in states:
            valid_from = self._as_datetime(state.get("valid_from"), as_of)
            valid_to = self._as_datetime(state.get("valid_to"), as_of)
            if valid_from and valid_from >= horizon_end:
                continue
            if valid_to and valid_to <= as_of:
                continue
            active_states.append({
                "id": int(state["id"]), "state_text": state.get("state_text") or "",
                "state_domain": state.get("state_domain") or "general",
                "state_kind": state.get("state_kind") or "",
                "conditions": state.get("conditions") or {},
                "confidence": state.get("confidence"),
                "updated_at": state.get("updated_at") or "",
            })
        return {"intents": intents, "recurring_states": active_states}

    def predict_near_term_preparation(
        self, *, as_of: Optional[datetime] = None,
        horizon_hours: float = 6.0,
        intent_refs: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run one bounded forecast, recording even empty and failed runs."""
        now = self._now(as_of)
        end = now + timedelta(hours=max(0.1, float(horizon_hours)))
        owner_id = self._db.get_entity_id_by_name(self._owner_name)
        if owner_id is None:
            owner_id = self._db.add_entity_names([self._owner_name])[self._owner_name]
        snapshot = self._input_snapshot(
            owner_id=owner_id, as_of=now, horizon_end=end, intent_refs=intent_refs,
        )
        intents = snapshot["intents"]
        states = snapshot["recurring_states"]
        predictions: List[Dict[str, Any]] = []
        status, error = "empty", ""
        if intents:
            source = " ".join(str(intent["summary"]) for intent in intents)
            language = "zh" if re.search(r"[\u4e00-\u9fff]", source) else "en"
            template = (
                USER_PREDICTION_NEAR_TERM_PREPARATION_PROMPT_ZH
                if language == "zh"
                else USER_PREDICTION_NEAR_TERM_PREPARATION_PROMPT_EN
            )
            prompt = template.format(
                as_of=now.isoformat(), horizon_start=now.isoformat(),
                horizon_end=end.isoformat(),
                intents=json.dumps(intents, ensure_ascii=False),
                states=json.dumps(states, ensure_ascii=False),
            )
            try:
                response = self._llm_client.call_llm(prompt)
                parsed = self._llm_client.parse_json_object_from_llm_text(response or "")
                if not isinstance(parsed, dict) or not isinstance(parsed.get("predictions"), list):
                    raise ValueError("Prediction response is not a JSON predictions array")
                predictions = self._validate_predictions(
                    parsed["predictions"], intents=intents, states=states,
                    as_of=now, horizon_end=end,
                )
                status = "ok" if predictions else "empty"
            except Exception as exc:
                status, error = "error", f"{type(exc).__name__}: {exc}"
                self._logger.exception("Near-term preparation prediction failed")
        with self._model_db.transaction():
            run_id = self._model_db.insert_user_prediction_run({
                "user_entity_id": owner_id, "prediction_kind": "near_term_preparation",
                "as_of": now.isoformat(), "horizon_start": now.isoformat(),
                "horizon_end": end.isoformat(), "status": status,
                "input_snapshot": snapshot, "predictor_version": self.MODEL_VERSION,
                "error": error,
            })
            saved = []
            for item in predictions:
                prediction_id, save_action = self._model_db.save_user_prediction({
                    **item, "run_id": run_id, "user_entity_id": owner_id,
                })
                saved.append({"id": prediction_id, "save_action": save_action, **item})
        return {"status": status, "run_id": run_id, "predictions": saved,
                "processed_intents": [
                    {"object_type": item["object_type"], "id": item["id"],
                     "fingerprint": item["fingerprint"]} for item in intents
                ], "error": error}

    def _validate_predictions(
        self, raw: Sequence[Any], *, intents: Sequence[Dict[str, Any]],
        states: Sequence[Dict[str, Any]], as_of: datetime,
        horizon_end: datetime,
    ) -> List[Dict[str, Any]]:
        intent_by_key = {(item["object_type"], item["id"]): item for item in intents}
        state_by_id = {item["id"]: item for item in states}
        accepted = []
        for candidate in raw[:5]:
            if not isinstance(candidate, dict):
                continue
            try:
                intent_type = str(candidate.get("intent_type") or "")
                intent_id = int(candidate.get("intent_id"))
                state_ids = sorted({int(value) for value in candidate.get("state_ids") or []})
            except (TypeError, ValueError):
                continue
            intent = intent_by_key.get((intent_type, intent_id))
            if intent is None or any(
                state_id not in state_by_id for state_id in state_ids
            ):
                continue
            behavior = str(candidate.get("behavior_text") or "").strip()
            if re.sub(r"\s+", "", behavior).casefold() == re.sub(
                r"\s+", "", str(intent["summary"])
            ).casefold():
                continue
            band = str(candidate.get("likelihood_band") or "").lower().strip()
            beginning = self._as_datetime(candidate.get("window_start"), as_of)
            ending = self._as_datetime(candidate.get("window_end"), as_of)
            intent_anchor = self._as_datetime(intent["anchor_at"], as_of)
            if (not behavior or band not in {"low", "medium", "high"}
                    or beginning is None or ending is None or intent_anchor is None
                    or beginning < as_of or ending > horizon_end
                    or not beginning < ending <= intent_anchor):
                continue
            basis = {
                "intent": intent,
                "recurring_states": [state_by_id[value] for value in state_ids],
            }
            key = self._hash({
                "kind": "near_term_preparation", "intent_type": intent_type,
                "intent_id": intent_id,
                "behavior": re.sub(r"\s+", " ", behavior).casefold(),
                "anchor": intent_anchor.isoformat(),
            })
            accepted.append({
                "prediction_key": key, "basis_hash": self._hash(basis),
                "behavior_text": behavior, "window_start": beginning.isoformat(),
                "window_end": ending.isoformat(),
                "conditions": self._text_list(candidate.get("conditions")),
                "likelihood_band": band,
                "uncertainties": self._text_list(candidate.get("uncertainties")),
                "rationale": str(candidate.get("rationale") or "").strip(),
                "basis": basis,
            })
        return accepted

    def get_active_predictions(
        self, *, as_of: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Read unexpired, unresolved hypotheses; no delivery decision here."""
        owner_id = self._db.get_entity_id_by_name(self._owner_name)
        if owner_id is None:
            return []
        reference = self._now(as_of)
        rows = self._model_db.get_active_user_predictions(
            user_entity_id=owner_id, as_of=reference.isoformat(),
        )
        current = []
        for row in rows:
            basis = row.get("basis") or {}
            intent_basis = basis.get("intent") or {}
            try:
                intent_type = str(intent_basis["object_type"])
                intent_id = int(intent_basis["id"])
                state_basis = basis["recurring_states"]
                state_ids = [int(state["id"]) for state in state_basis]
            except (KeyError, TypeError, ValueError):
                continue
            intent = self._db.get_intent_object(
                object_type=intent_type, object_id=intent_id,
            )
            valid_statuses = {
                "goal": {"active"}, "plan": {"planned", "rescheduled"},
                "work_item": {"open", "in_progress", "blocked"},
            }
            if (not intent or intent.get("status") not in valid_statuses.get(intent_type, set())
                    or int(intent.get("world_owner_entity_id") or 0) != owner_id
                    or intent.get("updated_at") != intent_basis.get("updated_at")):
                continue
            states = {int(item["id"]): item for item in self._model_db.get_user_states_by_ids(state_ids)}
            if any(
                state_id not in states
                or states[state_id].get("status") != "active"
                or states[state_id].get("updated_at") != source.get("updated_at")
                or (
                    self._as_datetime(states[state_id].get("valid_to"), reference) is not None
                    and self._as_datetime(states[state_id].get("valid_to"), reference) <= reference
                )
                for state_id, source in zip(state_ids, state_basis)
            ):
                continue
            current.append(row)
        return current

    def record_prediction_outcome(
        self, *, prediction_id: int, status: str,
        observed_at: Optional[datetime] = None,
        evidence_refs: Optional[Sequence[Dict[str, Any]]] = None,
        note: str = "",
    ) -> bool:
        """Attach an external validation result without creating memory evidence."""
        outcome = {
            "observed_at": self._now(observed_at).isoformat(),
            "evidence_refs": list(evidence_refs or []), "note": str(note),
        }
        return self._model_db.record_user_prediction_outcome(
            prediction_id=prediction_id, outcome_status=status, outcome=outcome,
        )
