"""Intent-event-driven forecasts of user preparation behavior."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from prompts_en import (
    USER_PREDICTION_NEAR_TERM_PREPARATION_PROMPT_EN,
    USER_PREDICTION_INTENT_UPDATE_PROMPT_EN,
)
from prompts_zh import (
    USER_PREDICTION_NEAR_TERM_PREPARATION_PROMPT_ZH,
    USER_PREDICTION_INTENT_UPDATE_PROMPT_ZH,
)

if TYPE_CHECKING:
    from llm_client import LLMClient
    from memory.memory_database import UserMemoryDB
    from task_scheduler import TaskScheduler
    from .user_model_database import UserModelDB


class UserPrediction:
    """Generate, inspect and verify preparation forecasts without changing intent."""

    TASK_KIND = "user_prediction_update_from_intent_events"
    MODEL_VERSION = "intent_preparation_v4"
    ACTIVE_INTENT_STATUSES = {
        "goal": {"active"}, "plan": {"planned", "rescheduled"},
        "work_item": {"open", "in_progress", "blocked"},
    }

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
        task_scheduler.register_handler(
            self.TASK_KIND, self.process_user_prediction_update_from_intent_events_task,
        )

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

    def _intent_snapshot(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "object_type": intent["object_type"], "id": int(intent["id"]),
            "summary": intent.get("summary") or "",
            "anchor_at": self._intent_anchor_text(intent),
            "time_start": intent.get("time_start") or "",
            "time_end": intent.get("time_end") or "",
            "time_precision": intent.get("time_precision") or "unknown",
            "primary_domain": intent.get("primary_domain") or "general",
            "status": intent.get("status") or "", "details": intent.get("details") or {},
            "updated_at": intent.get("updated_at") or "",
        }

    def _build_user_prediction_prediction_inputs_using_intents(
        self, *, owner_id: int, as_of: datetime,
        horizon_end: datetime, intent_event_ids: Sequence[int],
    ) -> Dict[str, Any]:
        events = self._db.get_user_intent_events(event_ids=intent_event_ids)
        if {int(event["id"]) for event in events} != set(intent_event_ids):
            raise ValueError("Prediction input contains missing intent events")
        intent_ids = sorted({int(event["intent_id"]) for event in events})
        rows = self._db.get_user_intent_items_by_ids(intent_ids)
        if {int(intent["id"]) for intent in rows} != set(intent_ids):
            raise ValueError("Prediction event references a missing intent")
        if any(
            int(intent.get("world_owner_entity_id") or 0) != owner_id
            or int(intent.get("primary_entity_id") or 0) != owner_id
            for intent in rows
        ):
            raise ValueError("Prediction events must refer to the world owner's intents")
        intents = [self._intent_snapshot(intent) for intent in rows]
        existing = self._model_db.get_user_predictions_by_intent_ids(
            intent_ids, statuses=["pending"], user_entity_id=owner_id,
        )
        evidence = self._model_db.get_user_prediction_evidence([row["id"] for row in existing])
        links = {}
        for link in evidence:
            links.setdefault(link["intent_id"], set()).add(link["prediction_id"])
        cases = []
        for row, intent in zip(rows, intents):
            related_events = [event for event in events if int(event["intent_id"]) == intent["id"]]
            latest = related_events[-1]
            # An older creation event can reach this task after a later update was persisted.
            changed_since_event = any(
                row.get(key) != value for key, value in latest["new_payload"].items()
            )
            cases.append({
                "event_type": "updated" if changed_since_event or any(
                    event["event_type"] == "updated" for event in related_events
                ) else "created",
                "current_intent": intent, "intent_events": related_events,
                "existing_predictions": [prediction for prediction in existing
                                         if prediction["id"] in links.get(intent["id"], set())],
            })

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
        return {"intent_cases": cases, "recurring_states": active_states}

    def process_user_prediction_update_from_intent_events_task(
        self, *, intent_event_ids: Sequence[int],
        reference_timestamp: Optional[Any] = None,
        horizon_hours: float = 6.0,
    ) -> Dict[str, Any]:
        """Generate for created intents; review and regenerate for updated intents."""
        event_ids = sorted({int(value) for value in intent_event_ids if int(value) > 0})
        if not event_ids:
            return {"status": "empty", "predictions": [], "intent_assessments": [],
                    "processed_intent_event_ids": []}
        local_now = self._now(None)
        now = self._as_datetime(reference_timestamp, local_now) or local_now
        end = now + timedelta(hours=max(0.1, float(horizon_hours)))
        owner_id = self._db.get_entity_id_by_name(self._owner_name)
        if owner_id is None:
            raise ValueError("Prediction world owner is missing")
        snapshot = self._build_user_prediction_prediction_inputs_using_intents(
            owner_id=owner_id, as_of=now, horizon_end=end, intent_event_ids=event_ids,
        )
        states = snapshot["recurring_states"]
        predictions, decisions, assessments = [], [], []
        status, error = "empty", ""
        try:
            created_cases = [case for case in snapshot["intent_cases"] if case["event_type"] == "created"]
            updated_cases = [case for case in snapshot["intent_cases"] if case["event_type"] == "updated"]
            if created_cases:
                generated, assessed = self._generate_predictions_for_created_intents_with_llm(
                    created_cases, states=states, as_of=now, horizon_end=end,
                )
                predictions.extend(generated)
                assessments.extend(assessed)
            if updated_cases:
                generated, reviewed = self._update_predictions_for_updated_intents_with_llm(
                    updated_cases, states=states, as_of=now, horizon_end=end,
                )
                predictions.extend(generated)
                decisions.extend(reviewed)
            status = "ok" if predictions or decisions or any(
                item["decision"] == "defer" for item in assessments
            ) else "empty"
        except Exception as exc:
            predictions, decisions, assessments = [], [], []
            status, error = "error", f"{type(exc).__name__}: {exc}"
            self._logger.exception("Intent-event prediction evaluation failed")
        with self._model_db.transaction():
            saved = []
            for item in predictions:
                prediction_id, save_action = self._model_db.save_user_prediction({
                    **item, "user_entity_id": owner_id,
                })
                saved.append({"id": prediction_id, "save_action": save_action, **item})
            for decision in decisions:
                replacements = [item["id"] for item in saved
                                if decision["prediction_id"] in item.get("replaces_prediction_ids", [])]
                review_id = self._model_db.save_user_prediction_review(
                    prediction_id=decision["prediction_id"], user_entity_id=owner_id,
                    intent_event_ids=decision["intent_event_ids"],
                    decision=decision["decision"], reason=decision["reason"],
                    intent_snapshot=decision["intent_snapshot"],
                    replacement_prediction_ids=replacements,
                )
                decision["review_id"] = review_id
                decision["replacement_prediction_ids"] = replacements
            if status != "error":
                # An intent change invalidates its old deferral. The update prompt
                # still only reviews/regenerates forecasts in this first version.
                clear_ids = [case["current_intent"]["id"] for case in updated_cases]
                clear_ids.extend(item["intent_id"] for item in assessments if item["decision"] != "defer")
                self._model_db.delete_user_prediction_reassessment_schedules(
                    clear_ids, user_entity_id=owner_id,
                )
                for assessment in assessments:
                    if assessment["decision"] == "defer":
                        self._model_db.upsert_user_prediction_reassessment_schedule(
                            intent_id=assessment["intent_id"], user_entity_id=owner_id,
                            next_evaluate_at=assessment["next_evaluate_at"], reason=assessment["reason"],
                            evaluated_at=assessment["evaluated_at"], intent_basis_hash=assessment["intent_basis_hash"],
                            intent_event_ids=assessment["intent_event_ids"],
                        )
        self._logger.info(
            "intent prediction update status=%s events=%s predictions=%s reviews=%s assessments=%s",
            status, event_ids, len(saved), len(decisions),
            [{"intent_id": item["intent_id"], "decision": item["decision"],
              "next_evaluate_at": item["next_evaluate_at"]} for item in assessments],
        )
        return {"status": status, "predictions": saved,
                "prediction_decisions": decisions,
                "intent_assessments": assessments,
                "prediction_context": {
                    "as_of": now.isoformat(), "horizon_start": now.isoformat(),
                    "horizon_end": end.isoformat(), "predictor_version": self.MODEL_VERSION,
                    "intent_event_ids": event_ids,
                },
                "processed_intent_event_ids": event_ids if status in {"ok", "empty"} else [],
                "error": error}

    def _generate_predictions_for_created_intents_with_llm(
        self, cases: Sequence[Dict[str, Any]], *, states: Sequence[Dict[str, Any]],
        as_of: datetime, horizon_end: datetime,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Assess each new intent: predict now, defer, or wait for changed evidence."""
        source = " ".join(case["current_intent"]["summary"] for case in cases)
        chinese = bool(re.search(r"[\u4e00-\u9fff]", source))
        template = (USER_PREDICTION_NEAR_TERM_PREPARATION_PROMPT_ZH if chinese
                    else USER_PREDICTION_NEAR_TERM_PREPARATION_PROMPT_EN)
        prompt = template.format(
            as_of=as_of.isoformat(), horizon_start=as_of.isoformat(), horizon_end=horizon_end.isoformat(),
            intent_cases=json.dumps(cases, ensure_ascii=False), states=json.dumps(states, ensure_ascii=False),
        )
        raw = self._llm_client.call_llm(prompt)
        parsed = self._llm_client.parse_json_object_from_llm_text(raw or "")
        if not isinstance(parsed, dict):
            raise ValueError("Prediction response must be a JSON object")
        return self._normalize_created_intent_prediction_results(
            parsed.get("intent_results"), cases=cases, states=states, as_of=as_of, horizon_end=horizon_end,
        )

    def _normalize_created_intent_prediction_results(
        self, results: Any, *, cases: Sequence[Dict[str, Any]], states: Sequence[Dict[str, Any]],
        as_of: datetime, horizon_end: datetime,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not isinstance(results, list) or len(results) != len(cases):
            raise ValueError("Every new intent requires exactly one assessment")
        cases_by_id = {case["current_intent"]["id"]: case for case in cases}
        generated, assessments, seen = [], [], set()
        for result in results:
            if not isinstance(result, dict):
                raise ValueError("Invalid intent prediction assessment")
            intent_id = int(result["intent_id"])
            if intent_id not in cases_by_id or intent_id in seen:
                raise ValueError("Unknown or duplicate intent assessment")
            seen.add(intent_id)
            case = cases_by_id[intent_id]
            current = case["current_intent"]
            decision = result.get("decision")
            reason = str(result.get("reason") or "").strip()
            if decision not in {"predict_now", "defer", "wait_for_change"} or not reason:
                raise ValueError("Invalid intent assessment decision or reason")
            new = self._normalize_case_predictions(
                result.get("predictions"), cases=[case], states=states,
                as_of=as_of, horizon_end=horizon_end,
            )
            if bool(new) != (decision == "predict_now"):
                raise ValueError("Only predict_now must contain one or more valid forecasts")
            next_evaluate_at = None
            if decision == "defer":
                value = result.get("next_evaluate_at")
                next_time = self._as_datetime(value, as_of)
                if (not isinstance(value, str) or not re.search(r"[T\s]\d{2}:\d{2}", value)
                        or next_time is None or next_time <= as_of):
                    raise ValueError("Deferred assessment requires a future evaluation datetime")
                if current["status"] not in self.ACTIVE_INTENT_STATUSES.get(current["object_type"], set()):
                    raise ValueError("Terminal intents cannot be deferred for preparation prediction")
                next_evaluate_at = next_time.isoformat()
            elif result.get("next_evaluate_at") is not None:
                raise ValueError("Only defer may specify next_evaluate_at")
            assessments.append({
                "intent_id": intent_id, "decision": decision, "reason": reason,
                "next_evaluate_at": next_evaluate_at, "evaluated_at": as_of.isoformat(),
                "intent_basis_hash": self._hash(current),
                "intent_event_ids": sorted({int(event["id"]) for event in case["intent_events"]}),
            })
            generated.extend(new)
        return generated, assessments

    def _update_predictions_for_updated_intents_with_llm(
        self, cases: Sequence[Dict[str, Any]], *, states: Sequence[Dict[str, Any]],
        as_of: datetime, horizon_end: datetime,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Review old forecasts and generate replacements in one LLM call."""
        source = " ".join(case["current_intent"]["summary"] for case in cases)
        chinese = bool(re.search(r"[\u4e00-\u9fff]", source))
        template = (USER_PREDICTION_INTENT_UPDATE_PROMPT_ZH if chinese
                    else USER_PREDICTION_INTENT_UPDATE_PROMPT_EN)
        prompt = template.format(
            as_of=as_of.isoformat(), horizon_start=as_of.isoformat(), horizon_end=horizon_end.isoformat(),
            intent_cases=json.dumps(cases, ensure_ascii=False), states=json.dumps(states, ensure_ascii=False),
        )
        raw = self._llm_client.call_llm(prompt)
        parsed = self._llm_client.parse_json_object_from_llm_text(raw or "")
        if not isinstance(parsed, dict):
            raise ValueError("Prediction response must be a JSON object")
        generated, reviews = [], []
        results = parsed.get("intent_results")
        if not isinstance(results, list) or len(results) != len(cases):
            raise ValueError("Every updated intent requires exactly one result")
        cases_by_id = {case["current_intent"]["id"]: case for case in cases}
        seen = set()
        for result in results:
            if not isinstance(result, dict):
                raise ValueError("Invalid updated intent result")
            intent_id = int(result["intent_id"])
            if intent_id not in cases_by_id or intent_id in seen:
                raise ValueError("Unknown or duplicate intent result")
            seen.add(intent_id)
            case = cases_by_id[intent_id]
            current = case["current_intent"]
            old = {item["id"]: item for item in case["existing_predictions"]}
            raw_decisions = result.get("existing_prediction_decisions")
            if not isinstance(raw_decisions, list):
                raise ValueError("Missing prediction decisions")
            decisions_by_id = {}
            for decision in raw_decisions:
                if not isinstance(decision, dict):
                    raise ValueError("Invalid prediction decision")
                prediction_id = int(decision["prediction_id"])
                action = decision.get("decision")
                reason = str(decision.get("reason") or "").strip()
                if (prediction_id not in old or prediction_id in decisions_by_id
                        or action not in {"retain", "invalidate", "replace"} or not reason):
                    raise ValueError("Invalid, unknown or duplicate prediction decision")
                if (action == "retain" and current["status"] not in
                        self.ACTIVE_INTENT_STATUSES.get(current["object_type"], set())):
                    raise ValueError("Terminal intents cannot retain preparation predictions")
                decisions_by_id[prediction_id] = {
                    "prediction_id": prediction_id, "decision": action,
                    "reason": reason, "intent_snapshot": current,
                    "intent_event_ids": sorted({int(event["id"]) for event in case["intent_events"]}),
                }
            if set(decisions_by_id) != set(old):
                raise ValueError("Every existing prediction must be reviewed explicitly")
            new = self._normalize_case_predictions(
                result.get("new_predictions"), cases=[case], states=states,
                as_of=as_of, horizon_end=horizon_end,
            )
            replaced = set()
            for item in new:
                for prediction_id in item["replaces_prediction_ids"]:
                    if (prediction_id not in decisions_by_id
                            or decisions_by_id[prediction_id]["decision"] != "replace"):
                        raise ValueError("Replacement references a prediction not marked replace")
                    replaced.add(prediction_id)
            if replaced != {key for key, decision in decisions_by_id.items()
                            if decision["decision"] == "replace"}:
                raise ValueError("Every replace decision requires a valid new prediction")
            retained_texts = {self._normalize_behavior_text(old[key]["behavior_text"])
                              for key, decision in decisions_by_id.items() if decision["decision"] == "retain"}
            if any(self._normalize_behavior_text(item["behavior_text"]) in retained_texts for item in new):
                raise ValueError("New prediction duplicates a retained prediction")
            reviews.extend(decisions_by_id.values())
            generated.extend(new)
        return generated, reviews

    @staticmethod
    def _normalize_behavior_text(text: str) -> str:
        return re.sub(r"\s+", "", text).casefold()

    def _normalize_case_predictions(
        self, raw: Any, *, cases: Sequence[Dict[str, Any]], states: Sequence[Dict[str, Any]],
        as_of: datetime, horizon_end: datetime,
    ) -> List[Dict[str, Any]]:
        if not isinstance(raw, list) or len(raw) > 5:
            raise ValueError("Expected at most five predictions")
        intents = [case["current_intent"] for case in cases if case["current_intent"]["status"]
                   in self.ACTIVE_INTENT_STATUSES.get(case["current_intent"]["object_type"], set())]
        normalized = []
        seen = set()
        for candidate in raw:
            accepted = self._validate_predictions(
                [candidate], intents=intents, states=states, as_of=as_of, horizon_end=horizon_end,
                intent_events=[event for case in cases for event in case["intent_events"]],
            )
            if len(accepted) != 1:
                raise ValueError("Prediction references invalid evidence, behavior or time window")
            item = accepted[0]
            replacements = candidate.get("replaces_prediction_ids", [])
            if not isinstance(replacements, list):
                raise ValueError("Replacement IDs must be an array")
            item["replaces_prediction_ids"] = sorted({int(value) for value in replacements})
            if any(value <= 0 for value in item["replaces_prediction_ids"]):
                raise ValueError("Replacement IDs must be positive")
            if any(case["event_type"] == "created" for case in cases) and replacements:
                raise ValueError("New intent predictions cannot replace existing predictions")
            identity = (item["basis"]["intent"]["id"], self._normalize_behavior_text(item["behavior_text"]))
            if identity in seen:
                raise ValueError("Duplicate predicted behavior")
            seen.add(identity)
            normalized.append(item)
        return normalized

    def _validate_predictions(
        self, raw: Sequence[Any], *, intents: Sequence[Dict[str, Any]],
        states: Sequence[Dict[str, Any]], as_of: datetime,
        horizon_end: datetime, intent_events: Sequence[Dict[str, Any]],
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
            anchor_text = str(intent.get("anchor_at") or "")
            precise_anchor = bool(re.search(r"[T\s]\d{2}:\d{2}", anchor_text))
            if intent["object_type"] == "plan" and intent.get("time_precision") in {
                "day", "week", "unknown",
            }:
                precise_anchor = False
            # Keep coarse or unknown anchors as uncertainty, never midnight deadlines.
            intent_anchor = self._as_datetime(anchor_text, as_of) if precise_anchor else None
            if (not behavior or band not in {"low", "medium", "high"}
                    or beginning is None or ending is None
                    or beginning < as_of or ending > horizon_end
                    or not beginning < ending
                    or (intent_anchor is not None and ending > intent_anchor)):
                continue
            basis = {
                "intent": intent,
                "recurring_states": [state_by_id[value] for value in state_ids],
                "prediction_context": {
                    "as_of": as_of.isoformat(), "horizon_start": as_of.isoformat(),
                    "horizon_end": horizon_end.isoformat(), "predictor_version": self.MODEL_VERSION,
                    "intent_event_ids": sorted({int(event["id"]) for event in intent_events
                                                if int(event["intent_id"]) == intent_id}),
                },
            }
            key = self._hash({
                "kind": "near_term_preparation", "intent_type": intent_type,
                "intent_id": intent_id,
                "behavior": re.sub(r"\s+", " ", behavior).casefold(),
                "anchor": intent_anchor.isoformat() if intent_anchor is not None else anchor_text,
                "basis_hash": self._hash(basis),
                "window_start": beginning.isoformat(), "window_end": ending.isoformat(),
                "conditions": self._text_list(candidate.get("conditions")),
                "likelihood_band": band,
                "uncertainties": self._text_list(candidate.get("uncertainties")),
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
        reviews = self._model_db.get_latest_user_prediction_reviews([row["id"] for row in rows])
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
            review = reviews.get(int(row["id"]))
            validated_intent = review["intent_snapshot"] if review and review["decision"] == "retain" else intent_basis
            if (not intent or intent.get("status") not in self.ACTIVE_INTENT_STATUSES.get(intent_type, set())
                    or int(intent.get("world_owner_entity_id") or 0) != owner_id
                    or int(intent.get("primary_entity_id") or 0) != owner_id
                    or self._intent_snapshot(intent) != validated_intent):
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
