"""LLM-based assessment of whether a prediction warrants proactive service."""

from __future__ import annotations

import logging
import json
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

if TYPE_CHECKING:
    from llm_client import LLMClient
    from task_scheduler import TaskScheduler
    from .user_model_database import UserModelDB

from prompts_en import USER_PROACTIVE_POLICY_EVALUATION_PROMPT_EN
from prompts_zh import USER_PROACTIVE_POLICY_EVALUATION_PROMPT_ZH


class UserProactiveService:
    """Evaluate new predictions and persist service recommendations without execution."""

    TASK_KIND = "user_proactive_policy_evaluation"
    POLICY_VERSION = "llm_policy_v1"

    def __init__(
        self, *, model_db: UserModelDB, task_scheduler: TaskScheduler,
        llm_client: LLMClient,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._model_db = model_db
        self._task_scheduler = task_scheduler
        self._llm_client = llm_client
        self._logger = logger or logging.getLogger(__name__)
        task_scheduler.register_handler(self.TASK_KIND, self.evaluate_new_predictions)

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

    def _evaluate_with_llm(
        self, prediction: Dict[str, Any], *, as_of: datetime,
        runtime_context: Dict[str, Any],
        current_user_states: Sequence[Dict[str, Any]],
    ) -> Dict[str, str]:
        source_text = " ".join((
            str(prediction.get("behavior_text") or ""),
            str(prediction.get("rationale") or ""),
            json.dumps(current_user_states, ensure_ascii=False),
            json.dumps(runtime_context, ensure_ascii=False),
        ))
        prompt_template = (
            USER_PROACTIVE_POLICY_EVALUATION_PROMPT_ZH
            if re.search(r"[\u4e00-\u9fff]", source_text)
            else USER_PROACTIVE_POLICY_EVALUATION_PROMPT_EN
        )
        prompt = prompt_template.format(
            as_of=as_of.isoformat(),
            prediction=json.dumps(prediction, ensure_ascii=False, sort_keys=True),
            current_user_states=json.dumps(
                current_user_states, ensure_ascii=False, sort_keys=True,
            ),
            runtime_context=json.dumps(
                runtime_context, ensure_ascii=False, sort_keys=True,
            ),
        )
        raw = self._llm_client.call_llm(prompt)
        parsed = self._llm_client.parse_json_object_from_llm_text(raw or "")
        if not isinstance(parsed, dict):
            raise ValueError("Proactive policy response is not a JSON object")
        decision = str(parsed.get("decision") or "").strip()
        if decision not in {"recommend_service", "defer", "suppress"}:
            raise ValueError(f"Unsupported proactive decision: {decision!r}")
        rationale = str(parsed.get("rationale") or "").strip()
        reason_code = str(parsed.get("reason_code") or "llm_assessed").strip()
        if not rationale:
            raise ValueError("Proactive policy response is missing rationale")
        return {
            "decision": decision,
            "reason_code": reason_code[:120],
            "rationale": rationale[:2000],
        }

    def evaluate_new_predictions(
        self, *, prediction_ids: Sequence[int], source_prediction_task_id: str = "",
        runtime_context: Optional[Dict[str, Any]] = None,
        reference_timestamp: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Persist one policy decision per newly created prediction."""
        now = datetime.now().astimezone()
        if reference_timestamp is not None:
            reference = self._as_datetime(reference_timestamp, now)
            if reference is None:
                raise ValueError("Invalid proactive policy reference_timestamp")
            now = reference
        context = dict(runtime_context) if isinstance(runtime_context, dict) else {}
        predictions = self._model_db.get_user_predictions_by_ids(prediction_ids)
        decisions = []
        for prediction in predictions:
            user_id = int(prediction["user_entity_id"])
            current_states = self._active_user_states(
                as_of=now, model_db=self._model_db,
            )
            context_snapshot = {
                "evaluated_at": now.isoformat(),
                "runtime_context": context,
                "current_user_states": current_states,
            }
            try:
                decision = self._evaluate_with_llm(
                    prediction, as_of=now, runtime_context=context,
                    current_user_states=current_states,
                )
            except Exception as exc:
                self._logger.exception(
                    "LLM proactive policy evaluation failed prediction_id=%s",
                    prediction.get("id"),
                )
                decisions.append({
                    "prediction_id": int(prediction["id"]),
                    "decision": "evaluation_failed",
                    "reason_code": "llm_evaluation_failed",
                    "rationale": f"{type(exc).__name__}: {exc}"[:2000],
                })
                continue
            decision_id, created = self._model_db.save_user_proactive_decision({
                "prediction_id": int(prediction["id"]),
                "user_entity_id": user_id,
                "policy_version": self.POLICY_VERSION,
                **decision,
                "runtime_context": context_snapshot,
                "prediction_snapshot": prediction,
                "source_prediction_task_id": source_prediction_task_id,
                "evaluated_at": now.isoformat(),
            })
            decisions.append({
                "decision_id": decision_id,
                "prediction_id": int(prediction["id"]),
                "created": created,
                **decision,
            })
        failed_count = sum(
            item["decision"] == "evaluation_failed" for item in decisions
        )
        status = "error" if failed_count else "ok" if decisions else "empty"
        self._logger.info(
            "user proactive policy evaluated predictions=%s decisions=%s failed=%s",
            len(predictions), len(decisions) - failed_count, failed_count,
        )
        return {
            "status": status, "evaluated_count": len(decisions) - failed_count,
            "failed_count": failed_count,
            "decisions": decisions,
        }

    def get_recent_decisions(
        self, *, user_entity_id: Optional[int] = None, limit: int = 100,
    ) -> List[Dict[str, Any]]:
        return self._model_db.get_user_proactive_decisions(
            user_entity_id=user_entity_id, limit=limit,
        )
