"""Generate persistent service opportunities without dispatching them to an Agent."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

from pwm_logging import as_pwm_logger

if TYPE_CHECKING:
    from llm_client import LLMClient
    from task_scheduler import TaskScheduler
    from memory.memory_database import UserMemoryDB
    from .user_model_database import UserModelDB

from prompts_en import USER_PROACTIVE_POLICY_EVALUATION_PROMPT_EN
from prompts_zh import USER_PROACTIVE_POLICY_EVALUATION_PROMPT_ZH


class UserProactiveService:
    """LLM policy owns service content/timing; execution belongs to a downstream Agent."""

    TASK_KIND = "user_proactive_policy_evaluation"
    POLICY_VERSION = "llm_services_v4"
    SERVICE_TYPES = {"notify", "suggest", "clarify", "offer_assistance", "prepare", "delegate"}

    def __init__(
        self, *, memory_db: UserMemoryDB, model_db: UserModelDB,
        prediction_reader: Callable[..., List[Dict[str, Any]]], task_scheduler: TaskScheduler,
        llm_client: LLMClient,
        world_owner_entity_name: str = "用户本人",
        logger: Optional[logging.Logger | logging.LoggerAdapter] = None,
    ) -> None:
        self._model_db = model_db
        self._memory_db = memory_db
        self._prediction_reader = prediction_reader
        self._owner_name = world_owner_entity_name
        self._owner_entity_id = self._memory_db.get_entity_id_by_name(self._owner_name)
        self._task_scheduler = task_scheduler
        self._llm_client = llm_client
        self._logger = as_pwm_logger(logger or logging.getLogger(__name__))
        task_scheduler.register_handler(self.TASK_KIND, self.generate_proactive_services_from_predictions)

    @staticmethod
    def _as_datetime(value: Any, reference: datetime) -> Optional[datetime]:
        try:
            result = value if isinstance(value, datetime) else datetime.fromisoformat(
                str(value or "").replace("Z", "+00:00")
            )
        except ValueError:
            return None
        if result.tzinfo is None:
            result = result.replace(tzinfo=reference.tzinfo)
        return result.astimezone(reference.tzinfo)

    @classmethod
    def _active_user_states(
        cls, *, as_of: datetime, model_db: UserModelDB,
    ) -> List[Dict[str, Any]]:
        states = model_db.get_user_states(
            statuses=["active"], limit=500,
        )
        active = []
        for state in states:
            valid_from = cls._as_datetime(state.get("valid_from"), as_of)
            valid_to = cls._as_datetime(state.get("valid_to"), as_of)
            if valid_from and valid_from > as_of:
                continue
            if valid_to and valid_to <= as_of:
                continue
            active.append(state)
        return active

    def _build_service_evaluation_input(
        self, predictions: Sequence[Dict[str, Any]], *, states: Sequence[Dict[str, Any]], as_of: datetime,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build the round's cases in order, isolating individual input failures."""
        prediction_cases, failures = [], []
        for prediction in predictions:
            started_at = time.monotonic()
            try:
                basis = prediction.get("basis") or {}
                source = basis.get("intent") or {}
                intent = self._memory_db.get_intent_object(
                    object_type=source["object_type"], object_id=int(source["id"]),
                )
                key = (intent["object_type"], int(intent["id"]))
                fact_ids = self._memory_db.get_intent_evidence_fact_ids([key], limit=20).get(key, [])
                facts, excluded_future_fact_ids = [], []
                for fact in self._memory_db.get_memory_facts_by_ids(fact_ids):
                    observed = self._as_datetime(fact.get("dialogue_time_key"), as_of)
                    if observed and observed > as_of:
                        excluded_future_fact_ids.append(fact["id"])
                        continue
                    facts.append({name: fact.get(name) for name in (
                        "id", "summary", "fact_type", "event_time_key", "dialogue_time_key",
                    )})
                supporting_ids = {int(item["id"]) for item in basis.get("recurring_states") or []}
                relevant = [state for state in states if (
                    int(state["id"]) in supporting_ids
                    or state.get("state_domain") in {intent.get("primary_domain"), "general"}
                )]
                relevant.sort(key=lambda state: int(state["id"]) not in supporting_ids)
                case = {
                    "prediction": prediction, "current_intent": intent, "related_facts": facts,
                    "relevant_user_states": [{name: state.get(name) for name in (
                        "id", "state_domain", "state_type", "state_kind", "state_topic", "state_text",
                        "conditions", "confidence", "valid_from", "valid_to", "last_observed_at",
                    )} for state in relevant[:40]],
                    "existing_services": self._model_db.get_user_proactive_services(
                        prediction_id=int(prediction["id"]), user_entity_id=int(prediction["user_entity_id"]),
                        limit=None,
                    ),
                    "recent_service_feedback": self._model_db.get_user_proactive_service_feedback(
                        user_entity_id=int(prediction["user_entity_id"]),
                        prediction_id=int(prediction["id"]), observed_before=as_of.isoformat(), limit=20,
                    ),
                }
                self._logger.log_json_format(self.TASK_KIND, "inputs_loaded", {
                    "prediction": {key: prediction.get(key) for key in (
                        "id", "behavior_text", "window_start", "window_end", "conditions",
                        "likelihood_band", "uncertainties", "rationale",
                    )},
                    "current_intent": intent, "related_facts": facts,
                    "future_fact_ids_excluded": excluded_future_fact_ids,
                    "relevant_user_states": case["relevant_user_states"],
                    "existing_services": [{key: item.get(key) for key in (
                        "id", "service_type", "objective", "status", "reason",
                        "next_evaluate_at", "recheck_condition", "expires_at",
                    )} for item in case["existing_services"]],
                    "as_of": as_of.isoformat(),
                })
                reason = "waiting_service_reassessment" if any(
                    service["status"] == "waiting" for service in case["existing_services"]
                ) else "new_prediction"
                self._logger.log_json_format(self.TASK_KIND, "evaluation_selected", {
                    "prediction_id": prediction["id"], "reason": reason,
                })
                prediction_cases.append(case)
            except Exception as exc:
                self._logger.exception(
                    "proactive service input building failed prediction_id=%s "
                    "elapsed_ms=%.2f existing_unexpired_services_unchanged=True",
                    prediction["id"], (time.monotonic() - started_at) * 1000,
                )
                failures.append({
                    "prediction_id": prediction["id"], "stage": "build_inputs",
                    "error": f"{type(exc).__name__}: {exc}",
                })
        return prediction_cases, failures

    @staticmethod
    def _format_evaluation_case_for_prompt(case: Dict[str, Any]) -> str:
        """One compact business-only case; leave persisted/internal records intact."""
        def select(item, fields):
            return {name: item[name] for name in fields if name in item}

        def intent_view(intent):
            result = select(intent, (
                "id", "object_type", "summary", "primary_domain", "status", "confidence",
                "time_start", "time_end", "time_precision",
            ))
            if "details" in intent:
                result["details"] = select(intent.get("details") or {}, (
                    "desired_outcome", "success_criteria", "target_at",
                    "event_or_activity", "time_precision", "location_text", "participants",
                    "action_text", "deliverable", "responsibility_type", "priority", "completed_at",
                    "beneficiary_entities", "delegator_entities", "collaborator_entities",
                ))
            return result

        state_fields = (
            "id", "state_domain", "state_type", "state_kind", "state_topic", "state_text",
            "conditions", "confidence", "valid_from", "valid_to", "last_observed_at",
        )
        prediction = case["prediction"]
        prediction_view = select(prediction, (
            "id", "behavior_text", "window_start", "window_end", "conditions",
            "likelihood_band", "uncertainties", "rationale",
        ))
        if "basis" in prediction:
            basis = prediction.get("basis") or {}
            basis_view = {}
            if "intent" in basis:
                basis_view["intent"] = intent_view(basis["intent"])
            if "recurring_states" in basis:
                basis_view["recurring_states"] = [select(item, state_fields) for item in basis["recurring_states"]]
            context = basis.get("prediction_context") or {}
            if "as_of" in context:
                basis_view["prediction_context"] = {"as_of": context["as_of"]}
            prediction_view["basis"] = basis_view
        view = {"prediction": prediction_view, "current_intent": intent_view(case["current_intent"])}
        if "related_facts" in case:
            view["related_facts"] = [select(item, (
                "id", "summary", "event_time_key", "dialogue_time_key",
            )) for item in case["related_facts"]]
        if "relevant_user_states" in case:
            view["relevant_user_states"] = [select(item, state_fields) for item in case["relevant_user_states"]]
        if "existing_services" in case:
            view["existing_services"] = [select(item, (
                "id", "service_type", "objective", "agent_guidance", "constraints", "status", "reason",
                "evaluated_at", "next_evaluate_at", "recheck_condition", "expires_at",
            )) for item in case["existing_services"]]
        if "recent_service_feedback" in case:
            view["recent_service_feedback"] = [select(item, (
                "id", "service_id", "execution_status", "user_response", "user_feedback",
                "execution_result", "observed_at",
            )) for item in case["recent_service_feedback"]]
        return json.dumps(view, ensure_ascii=False)

    def record_service_feedback(
        self, *, service_id: int, execution_id: str, execution_status: str,
        user_response: str = "unknown", user_feedback: str = "",
        execution_result: Optional[Dict[str, Any]] = None,
        agent_note: str = "", observed_at: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Persist execution reports independently of the service proposal status.

        Called by PWM's shared-worker handler. Late reports remain useful even
        when the proposal or prediction has expired; nothing is reactivated.
        """
        if type(service_id) is not int or service_id <= 0:
            raise ValueError("service_id must be a positive integer")
        if not isinstance(execution_id, str) or not execution_id.strip():
            raise ValueError("execution_id must be nonempty text")
        if not isinstance(execution_status, str) or execution_status not in {
            "not_executed", "presented", "running", "succeeded", "failed", "cancelled",
        }:
            raise ValueError("Unsupported service execution_status")
        if not isinstance(user_response, str) or user_response not in {
            "unknown", "accepted", "declined", "deferred", "not_needed",
        }:
            raise ValueError("Unsupported service user_response")
        if not isinstance(user_feedback, str) or not isinstance(agent_note, str):
            raise ValueError("user_feedback and agent_note must be text")
        if execution_result is not None and not isinstance(execution_result, dict):
            raise ValueError("execution_result must be an object")
        result = dict(execution_result or {})
        # Reject unserializable reports before any write. A report is structured
        # data, never executable content or a source of tool authorization.
        json.dumps(result, ensure_ascii=False, allow_nan=False)
        now = datetime.now().astimezone()
        observed = self._as_datetime(observed_at, now) if observed_at is not None else now
        if observed is None:
            raise ValueError("Invalid feedback observed_at")
        service = self._model_db.get_user_proactive_service(service_id)
        if service is None or service["user_entity_id"] != self._owner_entity_id:
            raise ValueError("Unknown service or service belongs to another user")
        predictions = self._model_db.get_user_predictions_by_ids([service["prediction_id"]])
        if not predictions:
            raise ValueError("Feedback service has no source prediction")
        prediction = predictions[0]
        source = (prediction.get("basis") or {}).get("intent") or {}
        intent = self._memory_db.get_intent_object(
            object_type=source["object_type"], object_id=int(source["id"]),
        )
        context = {
            "service": {key: service[key] for key in (
                "id", "prediction_id", "service_type", "objective", "agent_guidance", "constraints",
                "status", "evaluated_at", "expires_at",
            )},
            "prediction": {key: prediction.get(key) for key in (
                "id", "behavior_text", "window_start", "window_end", "status",
            )},
            "current_intent": {key: intent.get(key) for key in (
                "id", "object_type", "summary", "primary_domain", "status",
                "time_start", "time_end", "time_precision", "details",
            )} if intent else None,
        }
        payload = {
            "service_id": service_id, "execution_id": execution_id.strip(),
            "execution_status": execution_status, "user_response": user_response,
            "user_feedback": user_feedback, "execution_result": result,
            "agent_note": agent_note, "observed_at": observed.isoformat(), "context": context,
        }
        payload["id"] = self._model_db.insert_user_proactive_service_feedback(payload)
        self._logger.log_json_format("proactive_service_feedback", "service_feedback_recorded", {
            "feedback_id": payload["id"], "service_id": service_id,
            "execution_id": payload["execution_id"], "execution_status": execution_status,
            "user_response": user_response, "observed_at": payload["observed_at"],
        })
        return payload

    def get_service_feedback(
        self, *, service_id: Optional[int] = None, limit: int = 50,
    ) -> List[Dict[str, Any]]:
        if self._owner_entity_id is None:
            return []
        return self._model_db.get_user_proactive_service_feedback(
            user_entity_id=self._owner_entity_id, service_id=service_id, limit=limit,
        )

    def _evaluate_with_llm(self, cases: Sequence[Dict[str, Any]], *, as_of: datetime,
                           runtime_context: Dict[str, Any], agent_capabilities: Dict[str, Any]) -> Any:
        source_text = "\n".join(self._format_evaluation_case_for_prompt(case) for case in cases)
        chinese = bool(re.search(r"[\u4e00-\u9fff]", source_text))
        prompt_template = (
            USER_PROACTIVE_POLICY_EVALUATION_PROMPT_ZH
            if chinese
            else USER_PROACTIVE_POLICY_EVALUATION_PROMPT_EN
        )
        prompt = prompt_template.format(
            as_of=as_of.isoformat(),
            evaluation_cases=source_text,
            runtime_context=json.dumps(runtime_context, ensure_ascii=False),
            agent_capabilities=json.dumps(agent_capabilities, ensure_ascii=False),
        )
        started_at = time.monotonic()
        prediction_ids = [case["prediction"]["id"] for case in cases]
        self._logger.log_json_format(self.TASK_KIND, "llm_started", {
            "prediction_ids": prediction_ids, "as_of": as_of.isoformat(),
            "prompt_language": "zh" if chinese else "en", "prompt_chars": len(prompt),
            "runtime_context": runtime_context, "agent_capabilities": agent_capabilities,
        })
        raw = None
        try:
            raw = self._llm_client.call_llm(prompt, stage_name='user_proactive_policy_evaluation')
            parsed = self._llm_client.parse_json_object_from_llm_text(raw or "", stage_name='user_proactive_policy_evaluation')
            if not isinstance(parsed, dict):
                self._logger.error(
                    "proactive service invalid JSON prediction_ids=%s response=%r", prediction_ids, raw,
                )
            return parsed
        except Exception:
            self._logger.exception(
                "proactive service LLM call or parse failed prediction_ids=%s elapsed_ms=%.2f response=%r",
                prediction_ids, (time.monotonic() - started_at) * 1000, raw,
            )
            raise

    def _normalize_service_results(
        self, response: Any, *, case: Dict[str, Any], as_of: datetime,
        runtime_context: Dict[str, Any], agent_capabilities: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not isinstance(response, dict) or not isinstance(response.get("service_results"), list):
            raise ValueError("Proactive response must contain a service_results list")
        existing = {int(item["id"]): item for item in case["existing_services"]}
        open_ids = {key for key, item in existing.items() if item["status"] == "waiting"}
        seen_ids, seen_keys, results = set(), set(), []

        def future_time(value: Any) -> datetime:
            if not isinstance(value, str):
                raise ValueError("Service time must be a timezone-aware ISO timestamp")
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None or parsed <= as_of:
                raise ValueError(f"Service time must be after evaluation time: {value!r}")
            return parsed.astimezone(as_of.tzinfo)

        for raw in response["service_results"]:
            fields = {"service_id", "prediction_id", "decision", "reason", "service",
                      "next_evaluate_at", "recheck_condition", "expires_at"}
            if not isinstance(raw, dict) or set(raw) != fields:
                raise ValueError("Invalid service result fields")
            item = dict(raw)
            if (type(item["prediction_id"]) is not int
                    or item["prediction_id"] != case["prediction"]["id"]):
                raise ValueError("Service references an unrelated prediction")
            decision = item["decision"]
            if not isinstance(decision, str) or decision not in {"recommend_service", "defer", "suppress"}:
                raise ValueError("Unsupported proactive decision")
            if not isinstance(item["reason"], str) or not item["reason"].strip():
                raise ValueError("Service decision requires a reason")
            service_id = item["service_id"]
            if service_id is not None and (type(service_id) is not int or service_id not in open_ids):
                raise ValueError("Service update must reference an open service of this prediction")
            if decision == "suppress":
                if (item["service"] is not None or item["expires_at"] is not None
                        or item["next_evaluate_at"] is not None or item["recheck_condition"] != ""):
                    raise ValueError("Suppression must not include a new service or schedule")
            else:
                service = item["service"]
                if (not isinstance(service, dict) or set(service) != {
                    "service_type", "objective", "agent_guidance", "constraints",
                }):
                    raise ValueError("Service requires type, objective, Agent guidance and constraints")
                kind = service["service_type"]
                if not isinstance(kind, str) or kind not in self.SERVICE_TYPES:
                    raise ValueError("Unsupported service type")
                for name in ("objective", "agent_guidance"):
                    if not isinstance(service[name], str) or not service[name].strip():
                        raise ValueError(f"Missing service {name}")
                if not isinstance(service["constraints"], list) or any(
                    not isinstance(value, str) or not value.strip() for value in service["constraints"]
                ):
                    raise ValueError("Service constraints must be a list of nonempty strings")
                key_text = kind + ":" + re.sub(r"\s+", "", service["objective"]).casefold()
                item["service_key"] = hashlib.sha256(key_text.encode()).hexdigest()
                if item["service_key"] in seen_keys:
                    raise ValueError("Duplicate service objective in one response")
                seen_keys.add(item["service_key"])
                if service_id is None:
                    match = next((old for old in existing.values()
                                  if old["service_key"] == item["service_key"]), None)
                    if match:
                        service_id = int(match["id"])
                        if service_id not in open_ids:
                            raise ValueError("Closed services cannot be silently reactivated")
                        item["service_id"] = service_id
                expiry = future_time(item["expires_at"])
                item["expires_at"] = expiry.isoformat()
                if not isinstance(item["recheck_condition"], str):
                    raise ValueError("Recheck condition must be text")
                if decision == "recommend_service":
                    if item["next_evaluate_at"] is not None or item["recheck_condition"]:
                        raise ValueError("Ready services cannot have a waiting schedule")
                    if kind not in agent_capabilities.get("supported_service_types", []):
                        raise ValueError("Ready service requires explicitly available Agent capability")
                    if kind in {"prepare", "delegate"}:
                        if kind not in agent_capabilities.get("authorized_service_types", []):
                            raise ValueError("Background service requires explicit authorization")
                    elif runtime_context.get("can_present") is not True:
                        raise ValueError("Ready conversational service requires known presentation availability")
                else:
                    next_at = item["next_evaluate_at"]
                    if next_at is not None:
                        next_time = future_time(next_at)
                        if next_time >= expiry:
                            raise ValueError("Reevaluation must precede service expiry")
                        item["next_evaluate_at"] = next_time.isoformat()
                    elif not item["recheck_condition"].strip():
                        raise ValueError("Waiting service requires a reevaluation time or condition")
            if service_id is not None:
                if service_id in seen_ids:
                    raise ValueError("Existing service appears more than once")
                seen_ids.add(service_id)
            results.append(item)
        if seen_ids != open_ids:
            raise ValueError("Every waiting service must be explicitly updated or suppressed")
        return results

    def _normalize_round_service_results(
        self, response: Any, *, cases: Sequence[Dict[str, Any]], as_of: datetime,
        runtime_context: Dict[str, Any], agent_capabilities: Dict[str, Any],
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Validate the complete joint decision before writing any services."""
        if not isinstance(response, dict) or not isinstance(response.get("service_results"), list):
            raise ValueError("Proactive response must contain a service_results list")
        grouped = {case["prediction"]["id"]: [] for case in cases}
        for item in response["service_results"]:
            if (not isinstance(item, dict) or type(item.get("prediction_id")) is not int
                    or item["prediction_id"] not in grouped):
                raise ValueError("Service references an unrelated prediction")
            grouped[item["prediction_id"]].append(item)
        normalized = {}
        recommendation_count = 0
        for case in cases:
            prediction_id = case["prediction"]["id"]
            results = self._normalize_service_results(
                {"service_results": grouped[prediction_id]}, case=case, as_of=as_of,
                runtime_context=runtime_context, agent_capabilities=agent_capabilities,
            )
            recommendation_count += sum(item["decision"] == "recommend_service" for item in results)
            normalized[prediction_id] = results
        if recommendation_count > 1:
            raise ValueError("At most one service may be recommended per round, regardless of service type")
        return normalized

    def _save_service_results(
        self, results: Sequence[Dict[str, Any]], *, case: Dict[str, Any], as_of: datetime,
    ) -> List[Dict[str, Any]]:
        existing = {int(item["id"]): item for item in case["existing_services"]}
        saved = []
        with self._model_db.transaction():
            for item in results:
                if item["decision"] == "suppress" and item["service_id"] is None:
                    saved.append({**item, "save_action": "not_saved"})
                    continue
                if item["decision"] == "suppress":
                    payload = {**existing[item["service_id"]], "service_id": item["service_id"],
                               "status": "cancelled", "reason": item["reason"],
                               "next_evaluate_at": None, "recheck_condition": ""}
                else:
                    payload = {**item["service"], "service_id": item["service_id"],
                               "service_key": item["service_key"], "reason": item["reason"],
                               "status": "ready" if item["decision"] == "recommend_service" else "waiting",
                               **{name: item[name] for name in (
                                   "next_evaluate_at", "recheck_condition", "expires_at",
                               )}}
                payload.update(
                    prediction_id=case["prediction"]["id"],
                    user_entity_id=case["prediction"]["user_entity_id"],
                    policy_version=self.POLICY_VERSION, evaluated_at=as_of.isoformat(),
                )
                service_id, action = self._model_db.save_user_proactive_service(payload)
                saved.append({**item, "service_id": service_id, "status": payload["status"],
                              "save_action": action})
            self._model_db.mark_user_prediction_proactive_service_evaluated(
                prediction_id=case["prediction"]["id"],
                user_entity_id=case["prediction"]["user_entity_id"],
                evaluated_at=as_of.isoformat(),
            )
        self._logger.log_json_format(self.TASK_KIND, "services_saved", {
            "prediction_id": case["prediction"]["id"],
            "service_results": saved,
        })
        return saved

    def cleanup_proactive_services(
        self, *, reference_timestamp: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Clean services independently of generation, without calling an LLM.

        Return closed service IDs only.
        Prediction reads here are limited to sources of open services.
        Like generation, this write operation belongs on the shared worker.
        """
        now = datetime.now().astimezone()
        if reference_timestamp is not None:
            reference = self._as_datetime(reference_timestamp, now)
            if reference is None:
                raise ValueError("Invalid service cleanup reference_timestamp")
            now = reference
        if self._owner_entity_id is None:
            return {"closed_service_ids": []}
        open_services = self._model_db.get_user_proactive_services(
            user_entity_id=self._owner_entity_id, statuses=["ready", "waiting"], limit=None,
        )
        source_ids = sorted({service["prediction_id"] for service in open_services})
        active = {
            int(row["id"]): row for row in self._prediction_reader(as_of=now, prediction_ids=source_ids)
        } if source_ids else {}
        closed_ids = self._cleanup_proactive_service_lifecycle(
            open_services=open_services, active_predictions=active, as_of=now,
        )
        return {"closed_service_ids": closed_ids}

    def _cleanup_proactive_service_lifecycle(
        self, *, open_services: Sequence[Dict[str, Any]],
        active_predictions: Dict[int, Dict[str, Any]], as_of: datetime,
    ) -> List[int]:
        """Close services with ineffective sources or expired service windows.

        This is deterministic maintenance, independent of LLM evaluation and
        its per-call limit. Prediction records themselves are not changed.
        """
        source_ids = sorted({service["prediction_id"] for service in open_services})
        sources = {row["id"]: row for row in self._model_db.get_user_predictions_by_ids(source_ids)}
        closed_ids = []
        for prediction_id in source_ids:
            prediction = sources.get(prediction_id)
            if not prediction or prediction.get("user_entity_id") != self._owner_entity_id:
                self._logger.log_json_format(self.TASK_KIND, "lifecycle_source_skipped", {
                    "prediction_id": prediction_id, "reason": "missing_or_foreign_prediction",
                })
                continue
            if prediction_id not in active_predictions:
                ending = self._as_datetime(prediction.get("window_end"), as_of)
                status = "expired" if prediction["status"] == "expired" or (
                    prediction["status"] == "pending" and ending and ending <= as_of
                ) else "cancelled"
                closed = self._model_db.close_user_proactive_services(
                    prediction_id=prediction_id, user_entity_id=self._owner_entity_id, status=status,
                    reason="Source prediction is no longer effective", as_of=as_of.isoformat(),
                )
                closed_ids.extend(closed)
                self._logger.log_json_format(self.TASK_KIND, "services_closed_for_inactive_prediction", {
                    "prediction_id": prediction_id, "prediction_status": prediction["status"],
                    "closed_service_ids": closed, "closed_service_status": status,
                    "as_of": as_of.isoformat(),
                })
                continue
            expired = self._model_db.close_user_proactive_services(
                prediction_id=prediction_id, user_entity_id=self._owner_entity_id, status="expired",
                reason="Service validity period ended", as_of=as_of.isoformat(), expired_only=True,
            )
            closed_ids.extend(expired)
            if expired:
                self._logger.log_json_format(self.TASK_KIND, "services_expired", {
                    "prediction_id": prediction_id, "expired_service_ids": expired, "as_of": as_of.isoformat(),
                })
        return closed_ids

    def generate_proactive_services_from_predictions(
        self, *, runtime_context: Optional[Dict[str, Any]] = None,
        agent_capabilities: Optional[Dict[str, Any]] = None,
        reference_timestamp: Optional[Any] = None,
        max_predictions: int = 10,
    ) -> Dict[str, Any]:
        """Shared-worker handler: evaluate a host tick and return current help.

        Jointly evaluate eligible forecasts and their waiting services in one LLM call.
        The pending query includes unevaluated forecasts and waiting-service
        sources, reconsidered on each host trigger. Evaluated forecasts without
        a waiting service are not reselected. Ready is not proof of delivery.
        """
        started_at = time.monotonic()
        self._logger.log_json_format(self.TASK_KIND, "proactive_services_generation_start", {
            "reference_timestamp": reference_timestamp,
            "runtime_context": runtime_context, "agent_capabilities": agent_capabilities,
        })
        now = datetime.now().astimezone()
        if reference_timestamp is not None:
            reference = self._as_datetime(reference_timestamp, now)
            if reference is None:
                raise ValueError("Invalid proactive policy reference_timestamp")
            now = reference
        if runtime_context is not None and not isinstance(runtime_context, dict):
            raise ValueError("runtime_context must be an object")
        context = dict(runtime_context or {})
        if agent_capabilities is not None and not isinstance(agent_capabilities, dict):
            raise ValueError("agent_capabilities must be an object")
        capabilities = dict(agent_capabilities or {})
        for name in ("supported_service_types", "authorized_service_types"):
            values = capabilities.get(name, [])
            if not isinstance(values, list) or any(
                not isinstance(value, str) or value not in self.SERVICE_TYPES for value in values
            ):
                raise ValueError(f"Invalid capability list: {name}")
            capabilities[name] = sorted(set(values))
        if type(max_predictions) is not int or max_predictions <= 0:
            raise ValueError("max_predictions must be a positive integer")
        if self._owner_entity_id is None:
            return {"status": "empty", "assistance_items": [], "failed_predictions": []}
        cleanup = self.cleanup_proactive_services(reference_timestamp=now)
        closed_ids = cleanup["closed_service_ids"]
        candidate_ids = self._model_db.get_pending_proactive_prediction_ids(user_entity_id=self._owner_entity_id)
        active = {
            int(row["id"]): row for row in self._prediction_reader(as_of=now, prediction_ids=candidate_ids)
        } if candidate_ids else {}
        # Preserve the pending query's waiting-first, nearer-window ordering.
        ids = [prediction_id for prediction_id in candidate_ids if prediction_id in active]
        states = self._active_user_states(as_of=now, model_db=self._model_db) if ids else []
        self._logger.log_json_format(self.TASK_KIND, "predictions_loaded", {
            "candidate_prediction_ids": ids,
            "pending_prediction_ids": candidate_ids,
            "active_prediction_ids": sorted(active), "owner_id": self._owner_entity_id,
            "active_state_count": len(states), "as_of": now.isoformat(),
        })
        results, failures, skipped, processed, assistance, prediction_cases = [], [], [], [], [], []
        predictions = []
        for prediction_id in ids:
            prediction = active[prediction_id]
            if not prediction or prediction.get("user_entity_id") != self._owner_entity_id:
                skipped.append({"prediction_id": prediction_id, "reason": "missing_or_foreign_prediction"})
                self._logger.log_json_format(self.TASK_KIND, "prediction_skipped", skipped[-1])
                continue
            if len(predictions) >= max_predictions:
                skipped.append({"prediction_id": prediction_id, "reason": "evaluation_limit"})
                continue
            predictions.append(prediction)

        if predictions:
            prediction_cases, input_failures = self._build_service_evaluation_input(predictions, states=states, as_of=now)
            failures.extend(input_failures)

        normalized = {}
        if prediction_cases:
            response = None
            stage = "llm_evaluation"
            try:
                response = self._evaluate_with_llm(
                    prediction_cases, as_of=now, runtime_context=context, agent_capabilities=capabilities,
                )
                stage = "normalize_response"
                normalized = self._normalize_round_service_results(
                    response, cases=prediction_cases, as_of=now,
                    runtime_context=context, agent_capabilities=capabilities,
                )
                self._logger.log_json_format(self.TASK_KIND, "round_evaluated", {
                    "prediction_ids": list(normalized),
                    "result_count": sum(len(items) for items in normalized.values()),
                })
            except Exception as exc:
                self._logger.exception(
                    "Proactive joint evaluation failed stage=%s; services and mappings unchanged response=%s",
                    stage, response,
                )
                failures.extend({
                    "prediction_id": case["prediction"]["id"],
                    "error": f"{type(exc).__name__}: {exc}", "stage": stage,
                } for case in prediction_cases)
                prediction_cases = []

        for case in prediction_cases:
            prediction_id = case["prediction"]["id"]
            try:
                saved = self._save_service_results(
                    normalized[prediction_id], case=case, as_of=now,
                )
                results.extend(saved)
                previous_ready = {item["id"] for item in case["existing_services"] if item["status"] == "ready"}
                assistance.extend({
                    "service_id": item["service_id"], "prediction_id": prediction_id,
                    **item["service"], "reason": item["reason"], "expires_at": item["expires_at"],
                } for item in saved if item.get("status") == "ready" and item["service_id"] not in previous_ready)
                processed.append(prediction_id)
                self._logger.log_json_format(self.TASK_KIND, "prediction_finished", {"prediction_id": prediction_id, "status": "ok",
                                                       "result_count": len(saved)})
            except Exception as exc:
                self._logger.exception("Proactive service save failed prediction_id=%s", prediction_id)
                failures.append({"prediction_id": prediction_id, "error": f"{type(exc).__name__}: {exc}", "stage": "save_services"})
        closed_ids.extend(item["service_id"] for item in results if item.get("status") == "cancelled")
        report = {
            "status": ("partial" if failures and (processed or closed_ids) else "error" if failures
                       else "ok" if processed or closed_ids else "empty"),
            "evaluated_count": len(processed), "failed_count": len(failures),
            "processed_prediction_ids": processed,
            "new_service_ids": [item["service_id"] for item in results if item["save_action"] == "created"],
            "updated_service_ids": [item["service_id"] for item in results if item["save_action"] == "updated"],
            "ready_service_ids": [item["service_id"] for item in results if item.get("status") == "ready"],
            "waiting_service_ids": [item["service_id"] for item in results if item.get("status") == "waiting"],
            "closed_service_ids": sorted(set(closed_ids)),
            "skipped_predictions": skipped, "failed_predictions": failures, "service_results": results,
        }
        self._logger.log_json_format(self.TASK_KIND, "finish", {
            **{key: value for key, value in report.items() if key != "service_results"},
            "candidate_prediction_ids": ids,
            "result_count": len(results),
            "total_elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
        })
        return {"status": report["status"], "assistance_items": assistance, "failed_predictions": failures}

    def get_services(
        self, *, user_entity_id: Optional[int] = None, statuses: Optional[Sequence[str]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Read saved opportunities, not a dispatch/interruptibility decision."""
        services = self._model_db.get_user_proactive_services(
            user_entity_id=user_entity_id, statuses=statuses, limit=limit,
        )
        self._logger.log_json_format(self.TASK_KIND, "services_read", {
            "user_entity_id": user_entity_id, "statuses": statuses, "limit": limit,
            "service_ids": [item["id"] for item in services],
        })
        return services
