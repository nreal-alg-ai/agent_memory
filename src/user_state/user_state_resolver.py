"""Incremental, evidence-linked user-state projection helpers.

Entity claims stay the durable proposition ledger. This module prepares a
bounded update view for the LLM and validates/applies its state operations; it
does not manufacture a user model through hand-tuned scores.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

from prompts_en import USER_STATE_UPDATE_PROMPT_EN
from prompts_zh import USER_STATE_UPDATE_PROMPT_ZH

if TYPE_CHECKING:
    from memory.memory_database import SessionDB
    from llm_client import LLMClient
    from memory.task_scheduler import TaskScheduler


class UserStateResolver:
    """Turn changed claims and intent objects into incremental state updates."""

    RESOLVER_VERSION = "user_state_incremental_v1"
    STATE_CLASSES = {"enduring_state", "recurring_state", "situational_state"}
    STATE_DOMAINS = {
        "general", "health", "work", "life", "information", "finance",
        "family", "learning", "social", "interest", "travel",
    }
    STATE_KEYS_BY_CLASS = {
        "enduring_state": {
            "communication_style",
            "decision_style",
            "value_priority",
            "risk_preference",
            "privacy_preference",
            "time_preference",
            "work_style",
            "lifestyle_preference",
            "long_term_constraint",
        },
        "recurring_state": {
            "daily_routine",
            "weekly_routine",
            "work_routine",
            "health_routine",
            "social_routine",
            "conditional_behavior",
        },
        "situational_state": {
            "current_focus",
            "current_constraint",
            "current_availability",
            "current_energy",
            "current_stress",
            "current_social_engagement",
            "current_preparation",
        },
    }
    OPERATIONS = {"create", "reinforce", "revise", "weaken", "expire", "no_change"}
    ACTIVE_INTENT_STATUSES = {
        "goal": {"active"},
        "plan": {"planned", "rescheduled"},
        "work_item": {"open", "blocked"},
    }
    EVIDENCE_TYPES = {"user_claim", "goal", "plan", "work_item"}
    EVIDENCE_ROLES = {"support", "change_trigger", "counterexample", "context"}

    TASK_KIND = "memory_user_state_update"

    def __init__(
        self,
        *,
        db: "SessionDB",
        task_scheduler: TaskScheduler,
        llm_client: "LLMClient",
        memory_config: Optional[Mapping[str, Any]] = None,
        world_owner_entity_name: str = "用户本人",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._db = db
        self._task_scheduler = task_scheduler
        self._llm_client = llm_client
        self._world_owner_entity_name = str(world_owner_entity_name or "用户本人").strip()
        self._logger = logger or logging.getLogger(__name__)
        config = dict(memory_config or {})
        self._memory_enabled = self._config_bool(
            config.get("memory_enabled"), True,
        )
        self._enabled = self._config_bool(
            config.get("enable_memory_user_state_update"), True,
        )
        self._update_limit = max(
            1,
            int(config.get("user_state_update_limit", 80) or 80),
        )
        self._prompt_language_mode = str(
            config.get("memory_prompt_language_mode")
            or config.get("prompt_language_mode")
            or "source"
        ).strip().lower()
        self._task_scheduler.register_handler(self.TASK_KIND, self.process_update_task)

    @staticmethod
    def _config_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
            return True
        if text in {"0", "false", "no", "n", "off", "disable", "disabled"}:
            return False
        return default

    def set_logger(self, logger: Optional[logging.Logger]) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def submit_memory_user_state_update_task(
        self,
        *,
        changed_claim_ids: Optional[Sequence[int]] = None,
        changed_intent_objects: Optional[Sequence[Dict[str, Any]]] = None,
        reference_timestamp: Optional[Any] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
        completion_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Validate and enqueue a user-state update on the shared FIFO."""
        if not self._memory_enabled or not self._enabled:
            return self._task_scheduler.reject(
                self.TASK_KIND,
                "memory_disabled" if not self._memory_enabled
                else "user_state_update_disabled",
            )
        claim_ids = sorted({
            int(value)
            for value in changed_claim_ids or []
            if str(value).strip().isdigit() and int(value) > 0
        })
        intent_objects = [
            dict(value)
            for value in changed_intent_objects or []
            if isinstance(value, dict)
        ]
        if not claim_ids and not intent_objects:
            return self._task_scheduler.reject(
                self.TASK_KIND, "no_changed_state_evidence"
            )

        payload = {
            "changed_claim_ids": claim_ids,
            "changed_intent_objects": intent_objects,
            "reference_timestamp": reference_timestamp,
            "runtime_context": dict(runtime_context or {}),
        }
        submission = self._task_scheduler.submit(
            self.TASK_KIND,
            payload,
            completion_context=completion_context,
        )
        if not submission["queued"]:
            self._logger.warning(
                "User state task rejected reason=%s task_id=%s",
                submission["reason"],
                submission["task_id"],
            )
        return submission

    def process_update_task(
        self,
        *,
        changed_claim_ids: Optional[Sequence[int]] = None,
        changed_intent_objects: Optional[Sequence[Dict[str, Any]]] = None,
        reference_timestamp: Optional[Any] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build, prompt for, validate, and persist incremental state updates."""
        owner_mapping = self._db.add_entity_names([self._world_owner_entity_name])
        owner_id = int(owner_mapping[self._world_owner_entity_name])
        update_inputs = self.build_domain_update_inputs(
            user_entity_id=owner_id,
            changed_claim_ids=list(changed_claim_ids or []),
            changed_intent_objects=list(changed_intent_objects or []),
            reference_time=reference_timestamp,
            runtime_context=runtime_context or {},
            limit_per_domain=self._update_limit,
        )
        report: Dict[str, Any] = {
            "status": "empty",
            "domain_count": len(update_inputs),
            "changed_claim_count": sum(
                len(item["changed_claims"]) for item in update_inputs.values()
            ),
            "changed_intent_object_count": sum(
                len(item["changed_intent_objects"]) for item in update_inputs.values()
            ),
            "existing_state_count": sum(
                len(item["existing_states"]) for item in update_inputs.values()
            ),
            "operation_count": 0,
            "created": 0,
            "reinforced": 0,
            "revised": 0,
            "weakened": 0,
            "expired": 0,
            "state_ids": [],
            "domain_reports": {},
        }
        if not update_inputs:
            self._log_update(report)
            return report

        failed_domain_count = 0
        for state_domain, update_input in update_inputs.items():
            domain_report: Dict[str, Any] = {
                "changed_claim_count": len(update_input["changed_claims"]),
                "changed_intent_object_count": len(update_input["changed_intent_objects"]),
                "existing_state_count": len(update_input["existing_states"]),
                "operation_count": 0,
                "state_ids": [],
            }
            if not self.has_update_evidence(update_input):
                domain_report["status"] = "empty"
                report["domain_reports"][state_domain] = domain_report
                continue
            source_text = "\n".join([
                *(str(item.get("claim_text") or "") for item in update_input["changed_claims"]),
                *(str(item.get("summary") or "") for item in update_input["changed_intent_objects"]),
            ])
            language = self._resolve_prompt_language(source_text)
            prompt_template = (
                USER_STATE_UPDATE_PROMPT_EN
                if language == "en"
                else USER_STATE_UPDATE_PROMPT_ZH
            )
            prompt = prompt_template
            for key in (
                "state_domain", "reference_time", "runtime_context", "existing_states",
                "changed_claims", "changed_intent_objects", "active_intent_objects",
            ):
                value = update_input[key]
                serialized = (
                    json.dumps(value, ensure_ascii=False, indent=2)
                    if isinstance(value, (dict, list)) else str(value or "")
                )
                prompt = prompt.replace("{" + key + "}", serialized)
            parsed = self._llm_client.parse_json_object_from_llm_text(
                self._llm_client.call_llm(prompt) or ""
            )
            if parsed is None or not isinstance(parsed.get("operations"), list):
                domain_report.update({
                    "status": "error",
                    "error": "invalid_llm_user_state_update_response",
                })
                failed_domain_count += 1
                report["domain_reports"][state_domain] = domain_report
                continue
            operations = self.normalize_operations(
                parsed["operations"], update_input=update_input,
            )
            if not operations:
                domain_report["status"] = "ok"
                report["domain_reports"][state_domain] = domain_report
                continue
            with self._db.transaction():
                applied = self.apply_operations(
                    user_entity_id=owner_id,
                    operations=operations,
                    reference_time=update_input["reference_time"],
                )
            domain_report.update(applied)
            domain_report["status"] = "ok"
            report["domain_reports"][state_domain] = domain_report
            for key in (
                "operation_count", "created", "reinforced", "revised",
                "weakened", "expired",
            ):
                report[key] += int(applied.get(key) or 0)
            report["state_ids"].extend(applied.get("state_ids") or [])
        report["state_ids"] = sorted(set(report["state_ids"]))
        report["status"] = "error" if failed_domain_count == len(update_inputs) else "ok"
        if failed_domain_count:
            report["failed_domain_count"] = failed_domain_count
        self._log_update(report)
        return report

    def _resolve_prompt_language(self, text: str) -> str:
        if self._prompt_language_mode in {"en", "english", "force_en"}:
            return "en"
        if self._prompt_language_mode in {"zh", "chinese", "force_zh"}:
            return "zh"
        if re.search(r"[\u4e00-\u9fff]", str(text or "")):
            return "zh"
        return "zh"

    def _log_update(self, payload: Dict[str, Any]) -> None:
        record = {"scope": "memory_user_state", "event": "finish", "payload": payload}
        self._logger.info(
            "\n%s",
            json.dumps(record, ensure_ascii=False, sort_keys=False, indent=2, default=str),
        )

    def build_domain_update_inputs(
        self,
        *,
        user_entity_id: int,
        changed_claim_ids: Sequence[int],
        changed_intent_objects: Sequence[Mapping[str, Any]],
        reference_time: Optional[Any] = None,
        runtime_context: Optional[Mapping[str, Any]] = None,
        limit_per_domain: int = 100,
    ) -> Dict[str, Dict[str, Any]]:
        """Partition changed evidence into independently updateable domains."""
        owner_id = int(user_entity_id)
        changed_claims = [
            claim
            for claim in self._db.get_user_claims_by_ids(changed_claim_ids)
            if (
                int(claim.get("subject_entity_id") or 0) == owner_id
                and self._normalize_state_domain(claim.get("primary_domain"))
            )
        ]
        changed_intents = self._changed_intent_views(
            changed_intent_objects,
            user_entity_id=owner_id,
        )
        claim_evidence = self._db.get_user_claim_evidence_fact_ids(
            [int(claim["id"]) for claim in changed_claims],
            limit=480,
        )
        grouped_claims: Dict[str, List[Dict[str, Any]]] = {}
        grouped_intents: Dict[str, List[Dict[str, Any]]] = {}
        for claim in changed_claims:
            domain = self._normalize_state_domain(claim.get("primary_domain"))
            grouped_claims.setdefault(domain, []).append(claim)
        for item in changed_intents:
            domain = self._normalize_state_domain(item.get("primary_domain"))
            if domain:
                grouped_intents.setdefault(domain, []).append(item)

        update_inputs: Dict[str, Dict[str, Any]] = {}
        for domain in sorted(set(grouped_claims) | set(grouped_intents)):
            domain_claims = grouped_claims.get(domain, [])[:max(1, limit_per_domain)]
            domain_intents = grouped_intents.get(domain, [])[:max(1, limit_per_domain)]
            active_states = self._db.get_user_states(
                user_entity_id=owner_id,
                state_domains=[domain],
                statuses=["active", "weakened"],
                limit=max(1, limit_per_domain),
            )
            changed_intent_keys = {
                (str(item.get("object_type") or ""), int(item.get("id") or 0))
                for item in domain_intents
            }
            active_intents = [
                item for item in self._active_intents(
                    owner_id, primary_domain=domain,
                )
                if (str(item.get("object_type") or ""), int(item.get("id") or 0))
                not in changed_intent_keys
            ][:max(1, limit_per_domain)]
            update_inputs[domain] = {
                "user_entity_id": owner_id,
                "state_domain": domain,
                "reference_time": self._timestamp_text(reference_time),
                "runtime_context": dict(runtime_context or {}),
                "existing_states": [
                    self._state_prompt_view(state) for state in active_states
                ],
                "changed_claims": [
                    self._claim_prompt_view(
                        claim, claim_evidence.get(int(claim["id"]), []),
                    )
                    for claim in domain_claims
                ],
                "changed_intent_objects": list(domain_intents),
                "active_intent_objects": active_intents,
            }
        return update_inputs

    @staticmethod
    def has_update_evidence(update_input: Mapping[str, Any]) -> bool:
        return bool(
            update_input.get("changed_claims")
            or update_input.get("changed_intent_objects")
        )

    def normalize_operations(
        self,
        raw_operations: Any,
        *,
        update_input: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        """Reject LLM operations not grounded in this update input."""
        if not isinstance(raw_operations, list):
            return []
        existing_states = {
            int(item["id"]): item
            for item in self._db.get_user_states_by_ids([
                state.get("id") for state in update_input.get("existing_states", [])
                if isinstance(state, Mapping)
            ])
        }
        state_domain = self._normalize_state_domain(update_input.get("state_domain"))
        if not state_domain:
            return []
        allowed_evidence = self._allowed_evidence(update_input)
        normalized: List[Dict[str, Any]] = []
        seen_targets: set[tuple[str, int, str]] = set()
        for raw in raw_operations[:24]:
            item = self._normalize_operation(
                raw,
                existing_states=existing_states,
                allowed_evidence=allowed_evidence,
                state_domain=state_domain,
                reference_time=str(update_input.get("reference_time") or ""),
            )
            if item is None:
                continue
            target = (
                str(item["operation"]),
                int(item.get("target_state_id") or 0),
                str(item.get("state_domain") or ""),
                str(item.get("state_key") or ""),
            )
            if target not in seen_targets:
                seen_targets.add(target)
                normalized.append(item)
        return normalized

    def apply_operations(
        self,
        *,
        user_entity_id: int,
        operations: Sequence[Mapping[str, Any]],
        reference_time: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Persist validated operations and their evidence/event trail."""
        owner_id = int(user_entity_id)
        effective_at = self._timestamp_text(reference_time)
        report = {
            "operation_count": 0,
            "created": 0,
            "reinforced": 0,
            "revised": 0,
            "weakened": 0,
            "expired": 0,
            "state_ids": [],
        }
        for operation in operations:
            action = str(operation.get("operation") or "")
            target_id = int(operation.get("target_state_id") or 0)
            state_id, created, previous = self._db.upsert_user_state(
                user_entity_id=owner_id,
                target_state_id=target_id or None,
                state_domain=str(operation["state_domain"]),
                state_class=str(operation["state_class"]),
                state_key=str(operation["state_key"]),
                state_text=str(operation["state_text"]),
                status=str(operation["status"]),
                confidence=float(operation["confidence"]),
                conditions=dict(operation.get("conditions") or {}),
                valid_from=str(operation.get("valid_from") or ""),
                valid_to=str(operation.get("valid_to") or ""),
                last_observed_at=effective_at,
                metadata={
                    "resolver_version": self.RESOLVER_VERSION,
                    "last_operation": action,
                    "state_domain": str(operation["state_domain"]),
                },
            )
            current = self._db.get_user_states_by_ids([state_id])[0]
            evidence = [
                {**item, "state_id": state_id, "observed_at": effective_at}
                for item in operation.get("evidence") or []
            ]
            self._db.upsert_user_state_evidence(evidence)
            self._db.record_user_state_event(
                state_id=state_id,
                operation=action,
                previous_payload=previous or {},
                new_payload=current,
                trigger_claim_ids=[
                    item["evidence_id"] for item in evidence
                    if item["evidence_type"] == "user_claim"
                ],
                trigger_intent_objects=[
                    {"object_type": item["evidence_type"], "object_id": item["evidence_id"]}
                    for item in evidence
                    if item["evidence_type"] in {"goal", "plan", "work_item"}
                ],
                effective_at=effective_at,
                decision_source=self.RESOLVER_VERSION,
                confidence=float(operation["confidence"]),
                reason=str(operation.get("reason") or ""),
            )
            report["operation_count"] += 1
            report["state_ids"].append(state_id)
            if created:
                report["created"] += 1
            elif action == "reinforce":
                report["reinforced"] += 1
            elif action == "revise":
                report["revised"] += 1
            elif action == "weaken":
                report["weakened"] += 1
            elif action == "expire":
                report["expired"] += 1
        report["state_ids"] = sorted(set(report["state_ids"]))
        return report

    def get_current_states(
        self,
        *,
        user_entity_id: int,
        state_domains: Optional[Sequence[str]] = None,
        state_classes: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        return self._db.get_user_states(
            user_entity_id=int(user_entity_id),
            state_domains=state_domains,
            state_classes=state_classes,
            statuses=statuses or ["active", "weakened"],
        )

    def _normalize_operation(
        self,
        raw: Any,
        *,
        existing_states: Mapping[int, Mapping[str, Any]],
        allowed_evidence: Mapping[tuple[str, int], Mapping[str, Any]],
        state_domain: str,
        reference_time: str,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, Mapping):
            return None
        action = str(raw.get("operation") or "").strip().lower()
        state_class = str(raw.get("state_class") or "").strip().lower()
        if action not in self.OPERATIONS or state_class not in self.STATE_CLASSES:
            return None
        target_id = self._positive_int(raw.get("target_state_id"))
        target = existing_states.get(target_id)
        if action in {"reinforce", "revise", "weaken", "expire"} and target is None:
            return None
        if target is not None and str(target.get("state_class") or "") != state_class:
            return None
        if target is not None and str(target.get("state_domain") or "") != state_domain:
            return None
        if action == "create" and target_id:
            return None
        state_key = self._normalize_state_key(
            (target or {}).get("state_key") if target is not None
            else raw.get("state_key")
        )
        if target is None and state_key not in self.STATE_KEYS_BY_CLASS[state_class]:
            return None
        state_text = self._compact(raw.get("state_text") or (target or {}).get("state_text"))
        evidence = self._normalize_evidence(raw.get("evidence"), allowed_evidence)
        if not state_key or not state_text or not evidence:
            return None
        status = {
            "create": "active", "reinforce": "active", "revise": "active",
            "weaken": "weakened", "expire": "expired", "no_change": "",
        }[action]
        conditions = raw.get("conditions")
        if not isinstance(conditions, Mapping):
            conditions = dict((target or {}).get("conditions") or {})
        return {
            "operation": action,
            "target_state_id": target_id,
            "state_domain": state_domain,
            "state_class": state_class,
            "state_key": state_key,
            "state_text": state_text[:600],
            "status": status,
            "confidence": self._clamp(raw.get("confidence"), fallback=(target or {}).get("confidence", 0.7)),
            "conditions": dict(conditions),
            "valid_from": self._text(raw.get("valid_from") or (target or {}).get("valid_from") or reference_time),
            "valid_to": self._text(raw.get("valid_to") or (target or {}).get("valid_to")),
            "evidence": evidence,
            "reason": self._compact(raw.get("reason"))[:300],
        }

    def _allowed_evidence(self, update_input: Mapping[str, Any]) -> Dict[tuple[str, int], Mapping[str, Any]]:
        allowed: Dict[tuple[str, int], Mapping[str, Any]] = {}
        state_domain = self._normalize_state_domain(update_input.get("state_domain"))
        for claim in update_input.get("changed_claims") or []:
            if (
                isinstance(claim, Mapping)
                and self._normalize_state_domain(claim.get("primary_domain")) == state_domain
                and self._positive_int(claim.get("id"))
            ):
                allowed[("user_claim", int(claim["id"]))] = claim
        for item in update_input.get("changed_intent_objects") or []:
            if not isinstance(item, Mapping):
                continue
            object_type = str(item.get("object_type") or "")
            object_id = self._positive_int(item.get("id"))
            if (
                object_type in {"goal", "plan", "work_item"}
                and self._normalize_state_domain(item.get("primary_domain")) == state_domain
                and object_id
            ):
                allowed[(object_type, object_id)] = item
        return allowed

    def _normalize_evidence(self, raw_items: Any, allowed: Mapping[tuple[str, int], Mapping[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(raw_items, list):
            return []
        evidence: List[Dict[str, Any]] = []
        seen: set[tuple[str, int, str]] = set()
        for raw in raw_items[:24]:
            if not isinstance(raw, Mapping):
                continue
            evidence_type = str(raw.get("evidence_type") or "").strip()
            evidence_id = self._positive_int(raw.get("evidence_id"))
            role = str(raw.get("role") or "support").strip()
            key = (evidence_type, evidence_id, role)
            if (
                evidence_type not in self.EVIDENCE_TYPES or role not in self.EVIDENCE_ROLES
                or not evidence_id or (evidence_type, evidence_id) not in allowed or key in seen
            ):
                continue
            seen.add(key)
            evidence.append({
                "evidence_type": evidence_type,
                "evidence_id": evidence_id,
                "role": role,
                "weight": self._clamp(raw.get("weight"), fallback=1.0),
                "details": {"source": "user_state_incremental_llm"},
            })
        return evidence

    def _active_intents(
        self,
        user_entity_id: int,
        *,
        primary_domain: str,
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        owner_columns = {"goal": "owner_entity_id", "plan": "actor_entity_id", "work_item": "responsible_entity_id"}
        for object_type, statuses in self.ACTIVE_INTENT_STATUSES.items():
            for item in self._db.get_intent_objects(
                object_type=object_type, world_owner_entity_id=user_entity_id,
                primary_domain=primary_domain,
                statuses=sorted(statuses), limit=80,
            ):
                if int(item.get(owner_columns[object_type]) or 0) == user_entity_id:
                    result.append(self._intent_prompt_view(object_type, item))
        return result

    def _changed_intent_views(
        self,
        changed: Sequence[Mapping[str, Any]],
        *,
        user_entity_id: int,
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for raw in changed or []:
            if not isinstance(raw, Mapping):
                continue
            object_type = str(raw.get("object_type") or "").strip()
            object_id = UserStateResolver._positive_int(raw.get("object_id"))
            key = (object_type, object_id)
            if object_type not in {"goal", "plan", "work_item"} or not object_id or key in seen:
                continue
            stored = self._db.get_intent_object(
                object_type=object_type,
                object_id=object_id,
            )
            if not self._intent_belongs_to_user(
                object_type, stored, user_entity_id=user_entity_id,
            ):
                continue
            item = self._intent_prompt_view(object_type, stored)
            if self._normalize_state_domain(item.get("primary_domain")):
                result.append(item)
                seen.add(key)
        return result

    @staticmethod
    def _intent_belongs_to_user(
        object_type: str,
        item: Optional[Mapping[str, Any]],
        *,
        user_entity_id: int,
    ) -> bool:
        if not isinstance(item, Mapping):
            return False
        owner_column = {
            "goal": "owner_entity_id",
            "plan": "actor_entity_id",
            "work_item": "responsible_entity_id",
        }.get(object_type)
        return bool(
            owner_column
            and int(item.get("world_owner_entity_id") or 0) == int(user_entity_id)
            and int(item.get(owner_column) or 0) == int(user_entity_id)
        )

    @staticmethod
    def _claim_prompt_view(claim: Mapping[str, Any], evidence_fact_ids: Sequence[int]) -> Dict[str, Any]:
        return {
            "id": int(claim["id"]), "claim_text": str(claim.get("claim_text") or ""),
            "claim_type": str(claim.get("claim_type") or ""), "predicate": str(claim.get("predicate") or ""),
            "primary_domain": str(claim.get("primary_domain") or ""),
            "claim_origin": str(claim.get("claim_origin") or ""), "status": str(claim.get("status") or ""),
            "confidence": float(claim.get("confidence") or 0.0), "valid_from": str(claim.get("valid_from") or ""),
            "valid_to": str(claim.get("valid_to") or ""), "evidence_fact_ids": [int(value) for value in evidence_fact_ids or []],
        }

    @staticmethod
    def _state_prompt_view(state: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "id": int(state["id"]),
            "state_domain": str(state.get("state_domain") or ""),
            "state_class": str(state.get("state_class") or ""),
            "state_key": str(state.get("state_key") or ""), "state_text": str(state.get("state_text") or ""),
            "status": str(state.get("status") or ""), "confidence": float(state.get("confidence") or 0.0),
            "conditions": dict(state.get("conditions") or {}), "valid_from": str(state.get("valid_from") or ""),
            "valid_to": str(state.get("valid_to") or ""), "last_observed_at": str(state.get("last_observed_at") or ""),
        }

    @staticmethod
    def _intent_prompt_view(object_type: str, item: Mapping[str, Any]) -> Dict[str, Any]:
        fields = {
            "goal": ("summary", "desired_outcome", "success_criteria", "target_at"),
            "plan": ("summary", "event_or_activity", "start_at", "end_at", "location_text"),
            "work_item": ("summary", "action_text", "deliverable", "due_at", "start_at", "priority"),
        }
        result = {
            "object_type": object_type,
            "id": int(item["id"]),
            "primary_domain": str(item.get("primary_domain") or ""),
            "status": str(item.get("status") or ""),
            "confidence": float(item.get("confidence") or 0.0),
        }
        result.update({field: str(item.get(field) or "") for field in fields[object_type]})
        return result

    @staticmethod
    def _normalize_state_key(value: Any) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", str(value or "").lower()).strip("_")[:96]

    @classmethod
    def _normalize_state_domain(cls, value: Any) -> str:
        domain = re.sub(r"[^a-z]+", "_", str(value or "").lower()).strip("_")
        return domain if domain in cls.STATE_DOMAINS else ""

    @staticmethod
    def _compact(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @staticmethod
    def _text(value: Any) -> str:
        return str(value or "").strip()

    @staticmethod
    def _positive_int(value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 0
        return parsed if parsed > 0 else 0

    @staticmethod
    def _clamp(value: Any, *, fallback: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = float(fallback or 0.0)
        return max(0.0, min(1.0, parsed))

    @staticmethod
    def _timestamp_text(value: Optional[Any]) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value or "").strip() or datetime.now().astimezone().isoformat()
