"""Intent-event forecasts and time-driven reassessment of deferred behaviors."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from pwm_logging import as_pwm_logger
from intent_lifecycle import (
    ACTIVE_USER_INTENT_STATUSES,
    CLOSED_USER_INTENT_STATUSES,
    USER_INTENT_EVENT_TYPES,
)
from prompts_en import (
    USER_PREDICTION_REASSESSMENT_PROMPT_EN,
    USER_PREDICTION_GENERATED_FROM_NEW_INTENTS_PROMPT_EN,
    USER_PREDICTION_UPDATED_FROM_CHANGED_INTENTS_PROMPT_EN,
)
from prompts_zh import (
    USER_PREDICTION_REASSESSMENT_PROMPT_ZH,
    USER_PREDICTION_GENERATED_FROM_NEW_INTENTS_PROMPT_ZH,
    USER_PREDICTION_UPDATED_FROM_CHANGED_INTENTS_PROMPT_ZH,
)

if TYPE_CHECKING:
    from llm_client import LLMClient
    from memory.memory_database import UserMemoryDB
    from task_scheduler import TaskScheduler
    from .user_model_database import UserModelDB


class UserPrediction:
    """Generate, inspect and verify preparation forecasts without changing intent."""

    NEW_INTENT_TRIGGERED_TASK_NAME = "user_prediction_update_from_intent_events"
    TIME_TICK_TRIGGERED_TASK_NAME = "user_prediction_reassessment"
    WINDOW_EXPIRATION_TASK_NAME = "user_prediction_window_expiration"
    MODEL_VERSION = "intent_preparation_v5"

    def __init__(
        self, *, db: UserMemoryDB, model_db: UserModelDB, llm_client: LLMClient,
        task_scheduler: TaskScheduler,
        world_owner_entity_name: str = "用户本人",
        logger: Optional[logging.Logger | logging.LoggerAdapter] = None,
    ) -> None:
        self._user_memory_db = db
        self._model_db = model_db
        self._llm_client = llm_client
        self._task_scheduler = task_scheduler
        self._world_owner_entity_name = str(world_owner_entity_name or "用户本人").strip()
        self._world_owner_entity_id = int(self._user_memory_db.add_entity_names(
            [self._world_owner_entity_name]
        )[self._world_owner_entity_name])
        self._logger = as_pwm_logger(logger or logging.getLogger(__name__))
        task_scheduler.register_handler(
            self.NEW_INTENT_TRIGGERED_TASK_NAME, self.process_user_prediction_update_from_intent_events_task,
        )
        task_scheduler.register_handler(
            self.TIME_TICK_TRIGGERED_TASK_NAME, self.process_user_prediction_reassessment_task,
        )
        task_scheduler.register_handler(
            self.WINDOW_EXPIRATION_TASK_NAME, self.process_user_prediction_window_expiration_task,
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
    def _text_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()][:8]

    def _intent_snapshot(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "object_type": intent["object_type"], "id": int(intent["id"]),
            "summary": intent.get("summary") or "",
            "current_progress": intent.get("current_progress") or "",
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
        events = self._user_memory_db.get_user_intent_events(event_ids=intent_event_ids)
        if {int(event["id"]) for event in events} != set(intent_event_ids):
            raise ValueError("Prediction input contains missing intent events")
        if any(event["event_type"] not in USER_INTENT_EVENT_TYPES for event in events):
            raise ValueError("Prediction input contains unknown intent event types")
        intent_ids = sorted({int(event["intent_id"]) for event in events})
        rows = self._user_memory_db.get_user_intent_items_by_ids(intent_ids)
        if {int(intent["id"]) for intent in rows} != set(intent_ids):
            raise ValueError("Prediction event references a missing intent")
        if any(
            int(intent.get("world_owner_entity_id") or 0) != owner_id
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
        schedules = self._model_db.get_user_prediction_reassessment_schedules(
            intent_ids=intent_ids, user_entity_id=owner_id,
        )
        cases = []
        for row, intent in zip(rows, intents):
            related_events = [event for event in events if int(event["intent_id"]) == intent["id"]]
            latest = related_events[-1]
            # An older creation event can reach this task after a later update was persisted.
            changed_since_event = any(
                row.get(key) != value for key, value in latest["new_payload"].items()
            )
            existing_predictions = [prediction for prediction in existing
                if prediction["id"] in links.get(intent["id"], set())
                and (self._as_datetime(prediction.get("window_end"), as_of) or as_of) > as_of]
            deferred = [schedule for schedule in schedules if schedule["intent_id"] == intent["id"]]
            if intent["status"] in CLOSED_USER_INTENT_STATUSES:
                path = "ruling_cleanup"
            elif intent["status"] not in ACTIVE_USER_INTENT_STATUSES:
                raise ValueError(f"Unsupported current intent status: {intent['status']!r}")
            elif (changed_since_event or existing_predictions or deferred
                  or any(event["event_type"] != "created" for event in related_events)):
                path = "LLM_update"
            else:
                path = "LLM_create"
            fact_ids = self._user_memory_db.get_intent_evidence_fact_ids(
                [(intent["object_type"], intent["id"])],
            ).get((intent["object_type"], intent["id"]), []) if path != "ruling_cleanup" else []
            cases.append({
                "event_type": latest["event_type"], "processing_path": path,
                "current_intent": intent, "intent_events": related_events,
                "existing_predictions": existing_predictions,
                "existing_deferred_assessments": [{
                    "assessment_id": schedule["id"], **{key: schedule[key] for key in (
                        "reason", "evaluated_at", "next_evaluate_at", "intent_event_ids",
                    )},
                } for schedule in deferred],
                "related_facts": [{
                    "fact_id": fact["id"], "summary": fact.get("summary") or "",
                    "event_time_key": fact.get("event_time_key"),
                    "dialogue_time_key": fact.get("dialogue_time_key"),
                } for fact in self._user_memory_db.get_memory_facts_by_ids(fact_ids)],
            })

        return {"intent_cases": cases, "recurring_states": self._get_prediction_recurring_states(
            as_of=as_of, horizon_end=horizon_end,
        ) if any(case["processing_path"] != "ruling_cleanup" for case in cases) else []}

    def _get_prediction_recurring_states(
        self, *, as_of: datetime, horizon_end: datetime,
    ) -> List[Dict[str, Any]]:
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
        return active_states

    def process_user_prediction_window_expiration_task(
        self, *, reference_timestamp: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """End expired forecast windows and their open services without an LLM."""
        started_at = time.monotonic()
        reference = self._now(None)
        if reference_timestamp is not None:
            parsed = self._as_datetime(reference_timestamp, reference)
            if parsed is None:
                raise ValueError("Invalid prediction expiration reference_timestamp")
            reference = parsed
        self._logger.log_json_format(self.WINDOW_EXPIRATION_TASK_NAME, "window_expiration_start", {
            "as_of": reference.isoformat(), "owner_id": self._world_owner_entity_id,
        })
        try:
            expired = self._model_db.expire_user_predictions(
                user_entity_id=self._world_owner_entity_id, as_of=reference.isoformat(),
            )
        except Exception:
            self._logger.exception(
                "Prediction window expiration failed as_of=%s transaction_rolled_back=True",
                reference.isoformat(),
            )
            raise
        report = {
            "status": "ok" if expired["expired_prediction_ids"] or expired["expired_service_ids"] else "empty",
            "reference_timestamp": reference.isoformat(), **expired,
        }
        self._logger.log_json_format(self.WINDOW_EXPIRATION_TASK_NAME, "window_expiration_finish", {
            **report, "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
        })
        return report

    def process_user_prediction_reassessment_task(
        self, *, reference_timestamp: Optional[Any] = None, horizon_hours: float = 6.0,
    ) -> Dict[str, Any]:
        """Consume only due defer assessments, using current evidence and a new horizon."""
        started_at = time.monotonic()
        now = self._now(None)
        if reference_timestamp is not None:
            parsed_time = self._as_datetime(reference_timestamp, now)
            if parsed_time is None:
                raise ValueError("Invalid prediction reassessment reference_timestamp")
            now = parsed_time
        end = now + timedelta(hours=max(0.1, float(horizon_hours)))
        report: Dict[str, Any] = {
            "status": "empty", "predictions": [], "prediction_assessments": [],
            "invalidated_prediction_ids": [], "processed_assessment_ids": [],
            "prediction_context": {
                "as_of": now.isoformat(), "horizon_start": now.isoformat(),
                "horizon_end": end.isoformat(), "trigger_source": "time_tick",
                "predictor_version": self.MODEL_VERSION,
            },
        }
        owner_id = self._world_owner_entity_id
        due = self._model_db.get_user_prediction_reassessment_schedules(
            user_entity_id=owner_id, due_at=now.isoformat(),
        )
        self._logger.log_json_format(self.TIME_TICK_TRIGGERED_TASK_NAME, "reassessment_start", {
            "as_of": now.isoformat(), "horizon_end": end.isoformat(), "due_count": len(due),
            "assessment_ids": [item["id"] for item in due],
        })
        if not due:
            return report
        try:
            rows = self._user_memory_db.get_user_intent_items_by_ids(sorted({item["intent_id"] for item in due}))
            intents = {item["id"]: item for item in rows}
            cases, obsolete = [], []
            for schedule in due:
                intent = intents.get(schedule["intent_id"])
                if (intent is None or intent.get("world_owner_entity_id") != owner_id
                        or intent.get("status") not in ACTIVE_USER_INTENT_STATUSES):
                    obsolete.append(schedule)
                    continue
                snapshot = self._intent_snapshot(intent)
                fact_ids = self._user_memory_db.get_intent_evidence_fact_ids(
                    [(intent["object_type"], intent["id"])],
                ).get((intent["object_type"], intent["id"]), [])
                existing = self._model_db.get_user_predictions_by_intent_ids(
                    [intent["id"]], statuses=["pending"], user_entity_id=owner_id,
                )
                cases.append({
                    "assessment_id": schedule["id"],
                    "current_intent": snapshot,
                    "intent_events": self._user_memory_db.get_user_intent_events(event_ids=schedule["intent_event_ids"]),
                    "previous_assessment": {
                        key: schedule[key] for key in ("reason", "evaluated_at", "next_evaluate_at")
                    },
                    "related_facts": [{
                        "fact_id": fact["id"], "summary": fact.get("summary") or "",
                        "event_time_key": fact.get("event_time_key"),
                        "dialogue_time_key": fact.get("dialogue_time_key"),
                    } for fact in self._user_memory_db.get_memory_facts_by_ids(fact_ids)],
                    "existing_predictions": [item for item in existing
                        if (self._as_datetime(item.get("window_end"), now) or now) > now],
                })
            states = self._get_prediction_recurring_states(as_of=now, horizon_end=end) if cases else []
            self._logger.log_json_format(self.TIME_TICK_TRIGGERED_TASK_NAME, "reassessment_inputs_loaded", {
                "cases": cases, "recurring_states": states,
            })
            predictions, assessments = self._reassess_predictions_with_llm(
                cases, states=states, as_of=now, horizon_end=end,
            ) if cases else ([], [])
            # No event is fabricated or re-consumed. Only the selected schedules
            # are replaced; unrelated predictions and future schedules stay intact.
            with self._model_db.transaction():
                saved = []
                for item in predictions:
                    prediction_id, action = self._model_db.save_user_prediction({
                        **item, "user_entity_id": owner_id,
                    })
                    committed = (self._model_db.get_user_predictions_by_ids([prediction_id])[0]
                                 if action == "reused" else {"id": prediction_id, **item})
                    saved.append({**committed, "save_action": action})
                deferred_by_id = {item["assessment_id"]: item for item in assessments if item["decision"] == "defer"}
                for schedule in due:
                    item = deferred_by_id.get(schedule["id"])
                    if item is not None:
                        self._model_db.upsert_user_prediction_reassessment_schedule(
                            user_entity_id=owner_id, **{key: item[key] for key in (
                                "intent_id", "assessment_id", "next_evaluate_at", "reason",
                                "evaluated_at", "intent_event_ids",
                            )},
                        )
                    else:
                        self._model_db.delete_user_prediction_reassessment_schedule(
                            assessment_id=schedule["id"], user_entity_id=owner_id,
                        )
            report.update(
                status="ok", predictions=saved, prediction_assessments=assessments,
                processed_assessment_ids=[item["id"] for item in due],
                obsolete_assessment_ids=[item["id"] for item in obsolete],
            )
        except Exception as exc:
            self._logger.exception("Prediction reassessment failed as_of=%s schedules_preserved=True", now.isoformat())
            report.update(status="error", error=f"{type(exc).__name__}: {exc}")
        self._logger.log_json_format(self.TIME_TICK_TRIGGERED_TASK_NAME, "reassessment_finish", {
            **report, "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
        })
        return report

    def _reassess_predictions_with_llm(
        self, cases: Sequence[Dict[str, Any]], *, states: Sequence[Dict[str, Any]],
        as_of: datetime, horizon_end: datetime,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        chinese = bool(re.search(r"[\u4e00-\u9fff]", " ".join(case["current_intent"]["summary"] for case in cases)))
        template = USER_PREDICTION_REASSESSMENT_PROMPT_ZH if chinese else USER_PREDICTION_REASSESSMENT_PROMPT_EN
        prompt = template.format(
            as_of=as_of.isoformat(), horizon_start=as_of.isoformat(), horizon_end=horizon_end.isoformat(),
            assessment_cases="\n".join(json.dumps(case, ensure_ascii=False) for case in cases),
            states="\n".join(json.dumps(state, ensure_ascii=False) for state in states),
        )
        stage = "user_prediction_reassessment"
        raw = self._llm_client.call_llm(prompt, stage_name=stage)
        try:
            parsed = self._llm_client.parse_json_object_from_llm_text(raw or "", stage_name=stage)
            if not isinstance(parsed, dict) or set(parsed) != {"prediction_results"}:
                raise ValueError("Reassessment response requires only prediction_results")
            results = parsed["prediction_results"]
            if not isinstance(results, list):
                raise ValueError("prediction_results must be an array")
            by_id = {case["assessment_id"]: case for case in cases}
            seen, predictions, assessments = set(), [], []
            for result in results:
                if not isinstance(result, dict) or set(result) != {
                    "assessment_id", "decision", "reason", "next_evaluate_at", "prediction",
                }:
                    raise ValueError("Invalid reassessment result fields")
                assessment_id = result["assessment_id"]
                if type(assessment_id) is not int or assessment_id not in by_id or assessment_id in seen:
                    raise ValueError("Unknown or duplicate assessment ID")
                seen.add(assessment_id)
                case = by_id[assessment_id]
                generated, assessed = self._normalize_prediction_results(
                    [{"intent_id": case["current_intent"]["id"], **{
                        name: value for name, value in result.items() if name != "assessment_id"
                    }}], cases=[case], states=states, as_of=as_of, horizon_end=horizon_end,
                )
                existing_behaviors = {self._normalize_behavior_text(item["behavior_text"])
                                      for item in case["existing_predictions"]}
                for item in generated:
                    identity = (item["basis"]["intent"]["id"], self._normalize_behavior_text(item["behavior_text"]))
                    if identity[1] in existing_behaviors or any(
                        identity == (old["basis"]["intent"]["id"], self._normalize_behavior_text(old["behavior_text"]))
                        for old in predictions
                    ):
                        raise ValueError("Reassessment duplicates an existing or newly generated behavior")
                assessed[0]["assessment_id"] = assessment_id
                predictions.extend(generated)
                assessments.extend(assessed)
            if seen != set(by_id):
                raise ValueError("Missing reassessment results")
            return predictions, assessments
        except Exception as exc:
            self._log_prediction_response_failure(
                "reassessment", raw=raw, error=exc, cases=cases, states=states,
                as_of=as_of, horizon_end=horizon_end,
            )
            raise

    def process_user_prediction_update_from_intent_events_task(
        self, *, intent_event_ids: Sequence[int],
        reference_timestamp: Optional[Any] = None,
        horizon_hours: float = 6.0,
    ) -> Dict[str, Any]:
        """Route latest intents to creation, change review, or deterministic cleanup."""
        started_at = time.monotonic()
        event_ids = sorted({int(value) for value in intent_event_ids if int(value) > 0})
        self._logger.log_json_format(self.NEW_INTENT_TRIGGERED_TASK_NAME, "user_prediction_update_start", {
            "intent_event_ids": event_ids, "reference_timestamp": reference_timestamp,
            "horizon_hours": horizon_hours,
        })
        if not event_ids:
            self._logger.log_json_format(self.NEW_INTENT_TRIGGERED_TASK_NAME, "user_prediction_update_finish", {
                "status": "empty", "reason": "no_intent_events",
                "total_elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
            })
            return {"status": "empty", "predictions": [], "prediction_assessments": [],
                    "retained_prediction_ids": [], "invalidated_prediction_ids": [],
                    "processed_intent_event_ids": []}
        now = self._now(None)
        if reference_timestamp is not None:
            parsed_time = self._as_datetime(reference_timestamp, now)
            if parsed_time is None:
                raise ValueError("Invalid intent-event prediction reference_timestamp")
            now = parsed_time
        end = now + timedelta(hours=max(0.1, float(horizon_hours)))
        owner_id = self._world_owner_entity_id
        try:
            snapshot = self._build_user_prediction_prediction_inputs_using_intents(
                owner_id=owner_id, as_of=now, horizon_end=end, intent_event_ids=event_ids,
            )
        except Exception:
            self._logger.exception("Prediction input loading failed intent_event_ids=%s as_of=%s",
                                   event_ids, now.isoformat())
            raise
        states = snapshot["recurring_states"]
        self._logger.log_json_format(self.NEW_INTENT_TRIGGERED_TASK_NAME, "inputs_loaded", {
            "intent_event_ids": event_ids, "owner_id": owner_id,
            "horizon_start": now.isoformat(), "horizon_end": end.isoformat(),
            "intent_cases": [{
                "event_type": case["event_type"], "processing_path": case["processing_path"],
                "current_intent": case["current_intent"],
                "intent_event_ids": [item["id"] for item in case["intent_events"]],
                "existing_prediction_ids": [item["id"] for item in case["existing_predictions"]],
                "existing_deferred_assessments": case["existing_deferred_assessments"],
            } for case in snapshot["intent_cases"]],
            "recurring_states": states,
        })
        # Window expiration is independent of intent adjudication, and is never
        # rolled back because another intent's LLM evaluation fails.
        expired = self._model_db.expire_user_predictions(user_entity_id=owner_id, as_of=now.isoformat())
        saved, retained_prediction_ids, assessments = [], [], []
        invalidated_prediction_ids, closed_service_ids, processed_event_ids, failures = [], [], [], []
        groups = [[case] for case in snapshot["intent_cases"] if case["processing_path"] == "ruling_cleanup"]
        groups.extend([case for case in snapshot["intent_cases"] if case["processing_path"] == path]
                      for path in ("LLM_create", "LLM_update"))
        for cases in groups:
            if not cases:
                continue
            path = cases[0]["processing_path"]
            selected_events = sorted({event["id"] for case in cases for event in case["intent_events"]})
            try:
                retained = []
                if path == "ruling_cleanup":
                    generated, assessed = [], []
                    committed = self._cleanup_predictions_for_closed_intents(cases, as_of=now)
                else:
                    if path == "LLM_create":
                        generated, assessed = self._generate_predictions_for_created_intents_with_llm(
                            cases, states=states, as_of=now, horizon_end=end,
                        )
                    else:
                        generated, retained, assessed = self._update_predictions_for_updated_intents_with_llm(
                            cases, states=states, as_of=now, horizon_end=end,
                        )
                    committed = self._save_intent_prediction_results(
                        cases, predictions=generated, retained_prediction_ids=retained,
                        assessments=assessed, as_of=now,
                    )
                saved.extend(committed["predictions"])
                invalidated_prediction_ids.extend(committed["invalidated_prediction_ids"])
                closed_service_ids.extend(committed["closed_service_ids"])
                retained_prediction_ids.extend(retained)
                assessments.extend(assessed)
                processed_event_ids.extend(selected_events)
                self._logger.log_json_format(self.NEW_INTENT_TRIGGERED_TASK_NAME, "intent_path_results_saved", {
                    "processing_path": path, "intent_event_ids": selected_events,
                    "intent_ids": [case["current_intent"]["id"] for case in cases],
                    **committed, "retained_prediction_ids": retained, "prediction_assessments": assessed,
                })
            except Exception as exc:
                failures.append({
                    "processing_path": path, "intent_event_ids": selected_events,
                    "intent_ids": [case["current_intent"]["id"] for case in cases],
                    "error": f"{type(exc).__name__}: {exc}",
                })
                self._logger.exception(
                    "Prediction path failed path=%s intent_event_ids=%s as_of=%s "
                    "other_committed_paths_preserved=True", path, selected_events, now.isoformat(),
                )
        status = ("partial" if processed_event_ids else "error") if failures else (
            "ok" if saved or retained_prediction_ids or invalidated_prediction_ids or closed_service_ids
            or any(expired.values()) or any(item["decision"] == "defer" for item in assessments) else "empty"
        )
        error = "; ".join(item["error"] for item in failures)
        self._logger.log_json_format(self.NEW_INTENT_TRIGGERED_TASK_NAME, "finish", {
            "status": status, "intent_event_ids": event_ids, "prediction_count": len(saved),
            "retained_prediction_ids": retained_prediction_ids,
            "invalidated_prediction_ids": invalidated_prediction_ids,
            "assessment_count": len(assessments), "error": error,
            "processed_intent_event_ids": sorted(processed_event_ids), "failures": failures,
            "closed_service_ids": closed_service_ids, **expired,
            "total_elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
        })
        return {"status": status, "predictions": saved,
                "retained_prediction_ids": retained_prediction_ids,
                "invalidated_prediction_ids": invalidated_prediction_ids,
                "closed_service_ids": closed_service_ids, **expired,
                "prediction_assessments": assessments,
                "prediction_context": {
                    "as_of": now.isoformat(), "horizon_start": now.isoformat(),
                    "horizon_end": end.isoformat(), "predictor_version": self.MODEL_VERSION,
                    "intent_event_ids": event_ids,
                },
                "processed_intent_event_ids": sorted(processed_event_ids),
                "failed_intent_event_ids": sorted(set(event_ids) - set(processed_event_ids)),
                "failures": failures,
                "error": error}

    def _cleanup_predictions_for_closed_intents(
        self, cases: Sequence[Dict[str, Any]], *, as_of: datetime,
    ) -> Dict[str, Any]:
        """Invalidate pending forecasts, close open services, and remove defers atomically."""
        if any(case["current_intent"]["status"] not in CLOSED_USER_INTENT_STATUSES for case in cases):
            raise ValueError("Prediction cleanup requires closed intents")
        owner_id = self._world_owner_entity_id
        invalidated, closed = [], []
        with self._model_db.transaction():
            for case in cases:
                intent = case["current_intent"]
                prediction_ids = self._model_db.invalidate_user_predictions(
                    [item["id"] for item in case["existing_predictions"]], user_entity_id=owner_id,
                    as_of=as_of.isoformat(),
                )
                invalidated.extend(prediction_ids)
                for prediction_id in prediction_ids:
                    closed.extend(self._model_db.close_user_proactive_services(
                        prediction_id=prediction_id, user_entity_id=owner_id, status="cancelled",
                        reason=f"Source intent {intent['status']}", as_of=as_of.isoformat(),
                    ))
            self._model_db.delete_user_prediction_reassessment_schedules(
                [case["current_intent"]["id"] for case in cases], user_entity_id=owner_id,
            )
        return {"predictions": [], "invalidated_prediction_ids": invalidated, "closed_service_ids": closed}

    def _save_intent_prediction_results(
        self, cases: Sequence[Dict[str, Any]], *, predictions: Sequence[Dict[str, Any]],
        retained_prediction_ids: Sequence[int], assessments: Sequence[Dict[str, Any]], as_of: datetime,
    ) -> Dict[str, Any]:
        """Commit an active intent evaluation, including service cleanup and defer replacement."""
        owner_id = self._world_owner_entity_id
        provided = {item["id"] for case in cases for item in case["existing_predictions"]}
        write_keys = {item["prediction_key"] for item in predictions}
        reused_ids = {item["id"] for case in cases for item in case["existing_predictions"]
                      if item["prediction_key"] in write_keys}
        with self._model_db.transaction():
            invalidated = self._model_db.invalidate_user_predictions(
                sorted(provided - set(retained_prediction_ids) - reused_ids), user_entity_id=owner_id,
                as_of=as_of.isoformat(),
            )
            closed = []
            for prediction_id in invalidated:
                closed.extend(self._model_db.close_user_proactive_services(
                    prediction_id=prediction_id, user_entity_id=owner_id, status="cancelled",
                    reason="Source intent changed; prediction no longer applies", as_of=as_of.isoformat(),
                ))
            saved = []
            for item in predictions:
                prediction_id, action = self._model_db.save_user_prediction({**item, "user_entity_id": owner_id})
                committed = (self._model_db.get_user_predictions_by_ids([prediction_id])[0]
                             if action == "reused" else {"id": prediction_id, **item})
                saved.append({**committed, "save_action": action})
            self._model_db.delete_user_prediction_reassessment_schedules(
                [case["current_intent"]["id"] for case in cases], user_entity_id=owner_id,
            )
            for assessment in assessments:
                if assessment["decision"] == "defer":
                    assessment["assessment_id"] = self._model_db.upsert_user_prediction_reassessment_schedule(
                        user_entity_id=owner_id, **{key: assessment[key] for key in (
                            "intent_id", "next_evaluate_at", "reason", "evaluated_at", "intent_event_ids",
                        )},
                    )
        return {"predictions": saved, "invalidated_prediction_ids": invalidated, "closed_service_ids": closed}

    def _format_intent_cases_view_for_new_intent_prediction_prompt(
        self, cases: Sequence[Dict[str, Any]],
    ) -> str:
        """Keep only prediction evidence in the prompt; leave internal cases intact."""
        detail_fields = {
            "goal": ("desired_outcome", "success_criteria", "target_at"),
            "plan": ("event_or_activity", "time_precision", "location_text", "participants"),
            "work_item": ("action_text", "deliverable", "responsibility_type", "beneficiary_entities",
                          "delegator_entities", "collaborator_entities", "priority", "completed_at"),
        }
        lines = []
        for case in cases:
            intent = case["current_intent"]
            intent_view = {key: intent[key] for key in (
                "object_type", "id", "summary", "current_progress", "time_start", "time_end", "time_precision", "status",
            ) if key in intent and intent[key] not in ("", [], {}, None)}
            details = intent.get("details") or {}
            detail_view = {key: details[key] for key in detail_fields[intent["object_type"]]
                           if key in details and details[key] not in ("", [], {}, None)}
            if detail_view.get("target_at") == intent_view.get("time_end"):
                detail_view.pop("target_at", None)
            if detail_view.get("time_precision") == intent_view.get("time_precision"):
                detail_view.pop("time_precision", None)
            if detail_view:
                intent_view["details"] = detail_view
            fact_views = [{key: fact.get(key) for key in (
                "fact_id", "summary", "event_time_key", "dialogue_time_key",
            )} for fact in case["related_facts"]]
            lines.append(json.dumps({"current_intent": intent_view, "related_facts": fact_views}, ensure_ascii=False))
        return "\n".join(lines)

    def _generate_predictions_for_created_intents_with_llm(
        self, cases: Sequence[Dict[str, Any]], *, states: Sequence[Dict[str, Any]],
        as_of: datetime, horizon_end: datetime,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Assess individual behavior directions for newly created intents."""
        source = " ".join(case["current_intent"]["summary"] for case in cases)
        chinese = bool(re.search(r"[\u4e00-\u9fff]", source))
        template = (USER_PREDICTION_GENERATED_FROM_NEW_INTENTS_PROMPT_ZH if chinese
                    else USER_PREDICTION_GENERATED_FROM_NEW_INTENTS_PROMPT_EN)
        prompt = template.format(
            as_of=as_of.isoformat(), horizon_start=as_of.isoformat(), horizon_end=horizon_end.isoformat(),
            intent_cases=self._format_intent_cases_view_for_new_intent_prediction_prompt(cases),
            states=json.dumps(states, ensure_ascii=False),
        )
        started_at = time.monotonic()
        self._logger.log_json_format(self.NEW_INTENT_TRIGGERED_TASK_NAME, "llm_started", {
            "stage": "created", "intent_ids": [case["current_intent"]["id"] for case in cases],
            "prompt_language": "zh" if chinese else "en", "prompt_chars": len(prompt),
            "state_ids": [state["id"] for state in states],
            "horizon_start": as_of.isoformat(), "horizon_end": horizon_end.isoformat(),
        })
        raw = None
        try:
            raw = self._llm_client.call_llm(prompt, stage_name='user_prediction_from_created_intents')
            parsed = self._llm_client.parse_json_object_from_llm_text(raw or "", stage_name='user_prediction_from_created_intents')
            if not isinstance(parsed, dict) or set(parsed) != {"prediction_results"}:
                raise ValueError("Created-intent response must contain only prediction_results")
            predictions, assessments = self._normalize_prediction_results(
                parsed["prediction_results"], cases=cases, states=states, as_of=as_of, horizon_end=horizon_end,
            )
            return predictions, assessments
        except Exception as exc:
            self._log_prediction_response_failure(
                "created", raw=raw, error=exc, cases=cases, states=states,
                as_of=as_of, horizon_end=horizon_end,
            )
            raise

    def _log_prediction_response_failure(
        self, stage: str, *, raw: Any, error: Exception,
        cases: Sequence[Dict[str, Any]], states: Sequence[Dict[str, Any]],
        as_of: datetime, horizon_end: datetime,
    ) -> None:
        """Keep model output and validation scope together, only on failure."""
        payload = {
            "stage": stage, "error": f"{type(error).__name__}: {error}",
            "horizon_start": as_of.isoformat(), "horizon_end": horizon_end.isoformat(),
            "intent_cases": [{
                "current_intent": case["current_intent"],
                "intent_event_ids": [event["id"] for event in case["intent_events"]],
                "existing_prediction_ids": [item["id"] for item in case["existing_predictions"]],
            } for case in cases],
            "allowed_state_ids": [state["id"] for state in states],
            "raw_response": raw,
        }
        self._logger.error(
            "User prediction response rejected diagnostics=%s",
            json.dumps(payload, ensure_ascii=False, default=str),
        )

    def _normalize_prediction_results(
        self, results: Any, *, cases: Sequence[Dict[str, Any]], states: Sequence[Dict[str, Any]],
        as_of: datetime, horizon_end: datetime,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Normalize one assessment per behavior, allowing several per intent."""
        if not isinstance(results, list):
            raise ValueError("prediction_results must be an array")
        cases_by_id = {case["current_intent"]["id"]: case for case in cases}
        generated, assessments, seen_behaviors, seen_assessments = [], [], set(), set()
        counts = {}
        for index, result in enumerate(results):
            if not isinstance(result, dict) or set(result) != {
                "intent_id", "decision", "reason", "next_evaluate_at", "prediction",
            }:
                raise ValueError(f"Invalid prediction_results[{index}] fields")
            intent_id = result["intent_id"]
            if type(intent_id) is not int or intent_id not in cases_by_id:
                raise ValueError(f"Unknown intent_id in prediction_results[{index}]: {intent_id!r}")
            case = cases_by_id[intent_id]
            current = case["current_intent"]
            decision = result.get("decision")
            reason = result["reason"]
            if (decision not in {"predict_now", "defer", "wait_for_change"}
                    or not isinstance(reason, str) or not reason.strip()):
                raise ValueError(f"Invalid prediction_results[{index}] decision or reason")
            reason = reason.strip()
            if decision == "predict_now":
                prediction = result["prediction"]
                if not isinstance(prediction, dict) or set(prediction) != {
                    "intent_type", "state_ids", "behavior_text", "window_start", "window_end",
                    "conditions", "likelihood_band", "uncertainties", "rationale",
                }:
                    raise ValueError(f"prediction_results[{index}] requires one prediction object without intent_id")
                try:
                    new = self._normalize_case_predictions(
                        [{**prediction, "intent_id": intent_id}], cases=[case], states=states,
                        as_of=as_of, horizon_end=horizon_end,
                    )
                except ValueError as exc:
                    raise ValueError(f"prediction_results[{index}]: {exc}") from exc
                identity = (intent_id, self._normalize_behavior_text(new[0]["behavior_text"]))
                if identity in seen_behaviors:
                    raise ValueError("Duplicate predicted behavior")
                seen_behaviors.add(identity)
                counts[intent_id] = counts.get(intent_id, 0) + 1
                if counts[intent_id] > 5:
                    raise ValueError("Expected at most five new predictions per intent")
                generated.extend(new)
            elif result["prediction"] is not None:
                raise ValueError(f"prediction_results[{index}]: only predict_now may contain a prediction")
            next_evaluate_at = None
            if decision == "defer":
                value = result.get("next_evaluate_at")
                next_time = self._as_datetime(value, as_of)
                if (not isinstance(value, str) or not re.search(r"[T\s]\d{2}:\d{2}", value)
                        or next_time is None or next_time.tzinfo is None
                        or not re.search(r"(?:Z|[+-]\d{2}:\d{2})$", value)
                        or next_time <= as_of):
                    raise ValueError("Deferred assessment requires a future evaluation datetime")
                if current["status"] not in ACTIVE_USER_INTENT_STATUSES:
                    raise ValueError("Closed intents cannot be deferred for preparation prediction")
                next_evaluate_at = next_time.isoformat()
            elif result.get("next_evaluate_at") is not None:
                raise ValueError("Only defer may specify next_evaluate_at")
            identity = (intent_id, decision, reason, next_evaluate_at,
                        new[0]["prediction_key"] if decision == "predict_now" else None)
            if identity in seen_assessments:
                raise ValueError("Duplicate behavior assessment")
            seen_assessments.add(identity)
            assessments.append({
                "intent_id": intent_id, "decision": decision, "reason": reason,
                "next_evaluate_at": next_evaluate_at, "evaluated_at": as_of.isoformat(),
                "intent_event_ids": sorted({int(event["id"]) for event in case["intent_events"]}),
            })
        return generated, assessments

    def _update_predictions_for_updated_intents_with_llm(
        self, cases: Sequence[Dict[str, Any]], *, states: Sequence[Dict[str, Any]],
        as_of: datetime, horizon_end: datetime,
    ) -> tuple[List[Dict[str, Any]], List[int], List[Dict[str, Any]]]:
        """Reconcile retained IDs and per-behavior assessments in one LLM call."""
        source = " ".join(case["current_intent"]["summary"] for case in cases)
        chinese = bool(re.search(r"[\u4e00-\u9fff]", source))
        template = (USER_PREDICTION_UPDATED_FROM_CHANGED_INTENTS_PROMPT_ZH if chinese
                    else USER_PREDICTION_UPDATED_FROM_CHANGED_INTENTS_PROMPT_EN)
        prompt = template.format(
            as_of=as_of.isoformat(), horizon_start=as_of.isoformat(), horizon_end=horizon_end.isoformat(),
            intent_cases="\n".join(json.dumps(case, ensure_ascii=False) for case in cases),
            states=json.dumps(states, ensure_ascii=False),
        )
        started_at = time.monotonic()
        self._logger.log_json_format(self.NEW_INTENT_TRIGGERED_TASK_NAME, "llm_started", {
            "stage": "updated", "intent_ids": [case["current_intent"]["id"] for case in cases],
            "existing_prediction_ids": [item["id"] for case in cases for item in case["existing_predictions"]],
            "prompt_language": "zh" if chinese else "en", "prompt_chars": len(prompt),
            "state_ids": [state["id"] for state in states],
            "horizon_start": as_of.isoformat(), "horizon_end": horizon_end.isoformat(),
        })
        self._logger.debug("user_prediction_from_updated_intents llm_request prompt=%s", prompt)
        raw = None
        try:
            raw = self._llm_client.call_llm(prompt, stage_name='user_prediction_from_updated_intents')
            self._logger.debug("user_prediction_from_updated_intents llm_response raw_response=%s", raw)
            parsed = self._llm_client.parse_json_object_from_llm_text(raw or "", stage_name='user_prediction_from_updated_intents')
            if not isinstance(parsed, dict) or set(parsed) != {"retained_prediction_ids", "prediction_results"}:
                raise ValueError("Updated-intent response requires retained_prediction_ids and prediction_results")
            predictions, retained, assessments = self._normalize_updated_intent_prediction_results(
                parsed["prediction_results"], retained_prediction_ids=parsed["retained_prediction_ids"],
                cases=cases, states=states,
                as_of=as_of, horizon_end=horizon_end,
            )
            self._logger.log_json_format(self.NEW_INTENT_TRIGGERED_TASK_NAME, "llm_finished", {
                "stage": "updated", "intent_ids": [case["current_intent"]["id"] for case in cases],
                "prediction_count": len(predictions), "retained_prediction_ids": retained,
                "proposed_invalidated_prediction_ids": sorted({
                    item["id"] for case in cases for item in case["existing_predictions"]
                } - set(retained)),
                "assessments": assessments, "response_chars": len(raw or ""),
                "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
            })
            return predictions, retained, assessments
        except Exception as exc:
            self._log_prediction_response_failure(
                "updated", raw=raw, error=exc, cases=cases, states=states,
                as_of=as_of, horizon_end=horizon_end,
            )
            raise

    def _normalize_updated_intent_prediction_results(
        self, results: Any, *, retained_prediction_ids: Any,
        cases: Sequence[Dict[str, Any]], states: Sequence[Dict[str, Any]],
        as_of: datetime, horizon_end: datetime,
    ) -> tuple[List[Dict[str, Any]], List[int], List[Dict[str, Any]]]:
        if (not isinstance(retained_prediction_ids, list)
                or any(type(value) is not int or value <= 0 for value in retained_prediction_ids)
                or len(set(retained_prediction_ids)) != len(retained_prediction_ids)):
            raise ValueError("retained_prediction_ids must contain unique positive integer IDs")
        retained = set(retained_prediction_ids)
        provided = {item["id"] for case in cases for item in case["existing_predictions"]}
        if retained - provided:
            raise ValueError(f"Unknown retained prediction IDs: {sorted(retained - provided)}")
        generated, assessments = self._normalize_prediction_results(
            results, cases=cases, states=states, as_of=as_of, horizon_end=horizon_end,
        )
        for case in cases:
            current = case["current_intent"]
            old = {item["id"]: item for item in case["existing_predictions"]}
            if retained.intersection(old) and current["status"] not in ACTIVE_USER_INTENT_STATUSES:
                raise ValueError("Closed intents cannot retain preparation predictions")
            retained_texts = {self._normalize_behavior_text(item["behavior_text"])
                              for key, item in old.items() if key in retained}
            if any(item["basis"]["intent"]["id"] == current["id"]
                   and self._normalize_behavior_text(item["behavior_text"]) in retained_texts
                   for item in generated):
                raise ValueError("New prediction duplicates a retained prediction")
        return generated, sorted(retained), assessments

    @staticmethod
    def _normalize_behavior_text(text: str) -> str:
        return re.sub(r"\s+", "", text).casefold()

    def _normalize_case_predictions(
        self, raw: Any, *, cases: Sequence[Dict[str, Any]], states: Sequence[Dict[str, Any]],
        as_of: datetime, horizon_end: datetime,
    ) -> List[Dict[str, Any]]:
        if not isinstance(raw, list) or len(raw) > 5:
            raise ValueError("Expected at most five predictions")
        intents = [case["current_intent"] for case in cases
                   if case["current_intent"]["status"] in ACTIVE_USER_INTENT_STATUSES]
        normalized = []
        seen = set()
        for index, candidate in enumerate(raw):
            try:
                accepted = self._validate_predictions(
                    [candidate], intents=intents, states=states, as_of=as_of, horizon_end=horizon_end,
                    intent_events=[event for case in cases for event in case["intent_events"]],
                    source_assessment_id=cases[0].get("assessment_id") if len(cases) == 1 else None,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Prediction candidate[{index}] for intent_ids="
                    f"{[case['current_intent']['id'] for case in cases]} rejected: {exc}"
                ) from exc
            item = accepted[0]
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
        source_assessment_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        intent_by_key = {(item["object_type"], item["id"]): item for item in intents}
        state_by_id = {item["id"]: item for item in states}
        accepted = []
        for candidate in raw[:5]:
            if not isinstance(candidate, dict):
                raise ValueError("candidate_not_object")
            try:
                intent_type = str(candidate.get("intent_type") or "")
                intent_id = int(candidate.get("intent_id"))
                state_ids = sorted({int(value) for value in candidate.get("state_ids") or []})
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid_intent_or_state_id_format") from exc
            intent = intent_by_key.get((intent_type, intent_id))
            if intent is None:
                raise ValueError(
                    f"unknown_or_inactive_intent_reference: intent_type={intent_type!r}, "
                    f"intent_id={intent_id}, allowed_intents={sorted(intent_by_key)}"
                )
            unknown_state_ids = [value for value in state_ids if value not in state_by_id]
            if unknown_state_ids:
                raise ValueError(
                    f"unknown_state_ids: {unknown_state_ids}, allowed_state_ids={sorted(state_by_id)}"
                )
            behavior = str(candidate.get("behavior_text") or "").strip()
            if not behavior:
                raise ValueError("empty_behavior_text")
            if re.sub(r"\s+", "", behavior).casefold() == re.sub(
                r"\s+", "", str(intent["summary"])
            ).casefold():
                raise ValueError("behavior_restates_intent_summary")
            band = str(candidate.get("likelihood_band") or "").lower().strip()
            beginning = self._as_datetime(candidate.get("window_start"), as_of)
            ending = self._as_datetime(candidate.get("window_end"), as_of)
            anchor_text = str(intent.get("time_start" if intent_type == "plan" else "time_end") or "")
            precise_anchor = bool(re.search(r"[T\s]\d{2}:\d{2}", anchor_text))
            if intent["object_type"] == "plan" and intent.get("time_precision") in {
                "day", "week", "unknown",
            }:
                precise_anchor = False
            # Keep coarse or unknown anchors as uncertainty, never midnight deadlines.
            intent_anchor = self._as_datetime(anchor_text, as_of) if precise_anchor else None
            reasons = []
            if band not in {"low", "medium", "high"}:
                reasons.append(f"invalid_likelihood_band: {band!r}")
            if beginning is None:
                reasons.append(f"invalid_window_start: {candidate.get('window_start')!r}")
            elif beginning < as_of:
                reasons.append(f"window_start_before_horizon: {beginning.isoformat()} < {as_of.isoformat()}")
            if ending is None:
                reasons.append(f"invalid_window_end: {candidate.get('window_end')!r}")
            else:
                if ending > horizon_end:
                    reasons.append(f"window_end_after_horizon: {ending.isoformat()} > {horizon_end.isoformat()}")
                if intent_anchor is not None and ending > intent_anchor:
                    reasons.append(f"window_end_after_intent_anchor: {ending.isoformat()} > {intent_anchor.isoformat()}")
            if beginning is not None and ending is not None and not beginning < ending:
                reasons.append(f"non_positive_window: {beginning.isoformat()} >= {ending.isoformat()}")
            if reasons:
                raise ValueError("; ".join(reasons))
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
            if source_assessment_id is not None:
                basis["prediction_context"]["source_assessment_id"] = source_assessment_id
            # Exact write fingerprint, not a semantic identity or an input snapshot hash.
            forecast = {
                "behavior_text": behavior,
                "window_start": beginning.isoformat(), "window_end": ending.isoformat(),
                "conditions": self._text_list(candidate.get("conditions")),
                "likelihood_band": band,
                "uncertainties": self._text_list(candidate.get("uncertainties")),
                "rationale": str(candidate.get("rationale") or "").strip(),
            }
            key = hashlib.sha256(json.dumps({
                "intent_type": intent_type, "intent_id": intent_id,
                "intent_event_ids": basis["prediction_context"]["intent_event_ids"],
                "source_assessment_id": source_assessment_id, "forecast": forecast,
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            accepted.append({
                "prediction_key": key, **forecast, "basis": basis,
            })
        return accepted

    def get_active_predictions(
        self, *, as_of: Optional[datetime] = None,
        prediction_ids: Optional[Sequence[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Read unexpired, unresolved hypotheses; no delivery decision here."""
        owner_id = self._world_owner_entity_id
        reference = self._now(as_of)
        if prediction_ids is None:
            rows = self._model_db.get_active_user_predictions(
                user_entity_id=owner_id, as_of=reference.isoformat(),
            )
        else:
            rows = self._model_db.get_user_predictions_by_ids(prediction_ids)
        current = []
        for row in rows:
            ending = self._as_datetime(row.get("window_end"), reference)
            if (row.get("user_entity_id") != owner_id or row.get("status") != "pending"
                    or ending is None or ending <= reference):
                self._logger.debug(
                    "active_prediction_skipped prediction_id=%s reason=owner_status_or_window_invalid "
                    "status=%s window_end=%s as_of=%s",
                    row["id"], row.get("status"), row.get("window_end"), reference.isoformat(),
                )
                continue
            basis = row.get("basis") or {}
            intent_basis = basis.get("intent") or {}
            try:
                intent_type = str(intent_basis["object_type"])
                intent_id = int(intent_basis["id"])
                state_basis = basis["recurring_states"]
                state_ids = [int(state["id"]) for state in state_basis]
            except (KeyError, TypeError, ValueError):
                self._logger.debug("active_prediction_skipped prediction_id=%s reason=invalid_basis", row["id"])
                continue
            intent = self._user_memory_db.get_intent_object(
                object_type=intent_type, object_id=intent_id,
            )
            if (not intent or intent.get("status") not in ACTIVE_USER_INTENT_STATUSES
                    or int(intent.get("world_owner_entity_id") or 0) != owner_id):
                self._logger.debug(
                    "active_prediction_skipped prediction_id=%s intent_id=%s "
                    "reason=source_intent_missing_inactive_or_foreign intent_status=%s",
                    row["id"], intent_id, intent.get("status") if intent else None,
                )
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
                self._logger.debug(
                    "active_prediction_skipped prediction_id=%s state_ids=%s "
                    "reason=supporting_state_missing_changed_inactive_or_expired",
                    row["id"], state_ids,
                )
                continue
            current.append(row)
        self._logger.debug(
            "active_predictions_read as_of=%s requested_prediction_ids=%s loaded_count=%s active_prediction_ids=%s",
            reference.isoformat(), prediction_ids, len(rows), [row["id"] for row in current],
        )
        return current
