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

from prompts_en import (
    USER_ACTIVITY_STATE_UPDATE_PROMPT_EN,
    USER_CLAIM_STATE_RESOLUTION_PROMPT_EN,
    USER_OBSERVED_PATTERN_STATE_UPDATE_PROMPT_EN,
)
from prompts_zh import (
    USER_ACTIVITY_STATE_UPDATE_PROMPT_ZH,
    USER_CLAIM_STATE_RESOLUTION_PROMPT_ZH,
    USER_OBSERVED_PATTERN_STATE_UPDATE_PROMPT_ZH,
)

if TYPE_CHECKING:
    from memory.memory_database import UserMemoryDB
    from llm_client import LLMClient
    from task_scheduler import TaskScheduler
    from .user_model_database import UserModelDB


class UserStateResolver:
    """Turn new activity and changed claims into incremental user states."""

    RESOLVER_VERSION = "user_state_incremental_v1"
    CLAIM_RESOLVER_VERSION = "user_claim_state_joint_v1"
    STATE_TYPES = {"enduring_state", "recurring_state", "situational_state"}
    STATE_DOMAINS = {
        "general", "health", "work", "life", "information", "finance",
        "family", "learning", "social", "interest", "travel",
    }
    STATE_KINDS_BY_TYPE = {
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
    OPERATIONS = {"create", "reinforce", "revise", "weaken", "expire"}
    ACTIVE_INTENT_STATUSES = {
        "goal": {"active"},
        "plan": {"planned", "rescheduled"},
        "work_item": {"open", "blocked"},
    }
    EVIDENCE_TYPES = {
        "user_claim", "goal", "plan", "work_item", "user_activity", "fact",
        "user_observed_pattern", "user_action_signal",
    }
    EVIDENCE_ROLES = {"support", "change_trigger", "counterexample", "context"}

    CLAIM_STATE_TASK_KIND = "memory_user_claim_state_update"
    ACTIVITY_STATE_TASK_KIND = "memory_user_activity_state_update"
    OBSERVED_PATTERN_STATE_TASK_KIND = "memory_user_observed_pattern_state_update"

    def __init__(
        self,
        *,
        memory_db: "UserMemoryDB",
        model_db: "UserModelDB",
        task_scheduler: TaskScheduler,
        llm_client: "LLMClient",
        user_state_config: Optional[Mapping[str, Any]] = None,
        world_owner_entity_name: str = "用户本人",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._user_memory_db = memory_db
        self._user_model_db = model_db
        self._llm_client = llm_client
        self._world_owner_entity_name = str(world_owner_entity_name or "用户本人").strip()
        self._logger = logger or logging.getLogger(__name__)
        config = dict(user_state_config or {})
        self._update_limit = max(
            1,
            int(config.get("user_state_update_limit", 80) or 80),
        )
        self._prompt_language_mode = str(
            config.get("prompt_language_mode")
            or "source"
        ).strip().lower()
        task_scheduler.register_handler(
            self.CLAIM_STATE_TASK_KIND,
            self.process_user_claim_state_update_task,
        )
        task_scheduler.register_handler(
            self.ACTIVITY_STATE_TASK_KIND,
            self.process_user_activity_state_update_task,
        )
        task_scheduler.register_handler(
            self.OBSERVED_PATTERN_STATE_TASK_KIND,
            self.process_user_observed_pattern_state_update_task,
        )

    def set_logger(self, logger: Optional[logging.Logger]) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def process_user_activity_state_update_task(
        self,
        *,
        changed_user_activity_ids: Optional[Sequence[int]] = None,
        reference_timestamp: Optional[Any] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update situational states from a new activity window and linked evidence."""
        activity_ids = list(changed_user_activity_ids or [])
        reference_time = reference_timestamp
        owner_id = int(self._user_memory_db.add_entity_names([self._world_owner_entity_name])[
            self._world_owner_entity_name
        ])
        activities = self._user_memory_db.get_user_activity_items_by_ids(activity_ids)
        if not activities:
            result = {
                "status": "empty", "activity_count": 0,
                "operation_count": 0, "state_ids": [],
            }
            self._log_update(result, task_kind=self.ACTIVITY_STATE_TASK_KIND)
            return result
        activity_timestamps = [
            str(item.get("ended_at") or item.get("started_at") or "").strip()
            for item in activities
            if str(item.get("ended_at") or item.get("started_at") or "").strip()
        ]
        effective_reference_time = reference_time or (
            max(activity_timestamps) if activity_timestamps else None
        )

        fact_ids = sorted({
            int(fact_id)
            for activity in activities
            for fact_id in activity.get("evidence_fact_ids") or []
            if self._positive_int(fact_id)
        })
        facts = self._user_memory_db.get_memory_facts_by_ids(fact_ids) if fact_ids else []
        facts_by_id = {int(item["id"]): item for item in facts}
        activity_fact_ids = {
            int(activity["id"]): [
                int(fact_id) for fact_id in activity.get("evidence_fact_ids") or []
                if self._positive_int(fact_id) in facts_by_id
            ]
            for activity in activities
        }
        linked_fact_ids = sorted({
            fact_id for values in activity_fact_ids.values() for fact_id in values
        })

        related_intents: List[Dict[str, Any]] = []
        intent_owner_columns = {
            "goal": "owner_entity_id",
            "plan": "actor_entity_id",
            "work_item": "responsible_entity_id",
        }
        for object_type, active_statuses in self.ACTIVE_INTENT_STATUSES.items():
            items = self._user_memory_db.get_intent_objects(
                object_type=object_type,
                world_owner_entity_id=owner_id,
                statuses=sorted(active_statuses),
                limit=self._update_limit,
            )
            owned_items = [
                item for item in items
                if int(item.get(intent_owner_columns[object_type]) or 0) == owner_id
            ]
            evidence_by_object = self._user_memory_db.get_intent_evidence_fact_ids(
                [(object_type, int(item["id"])) for item in owned_items],
                limit=max(120, self._update_limit * 4),
            )
            for item in owned_items:
                key = (object_type, int(item["id"]))
                supporting_fact_ids = evidence_by_object.get(key, [])
                if not set(supporting_fact_ids).intersection(linked_fact_ids):
                    continue
                view = self._intent_prompt_view(object_type, item)
                view["evidence_fact_ids"] = [
                    int(value) for value in supporting_fact_ids
                    if int(value) in facts_by_id
                ]
                related_intents.append(view)

        existing_states = self._user_model_db.get_user_states(
            state_types=["situational_state"],
            statuses=["active", "weakened"],
            limit=max(40, self._update_limit * 2),
        )
        activity_views = [
            {
                "id": int(item["id"]),
                "activity_text": str(item.get("activity_text") or ""),
                "user_role": str(item.get("user_role") or ""),
                "confidence": float(item.get("confidence") or 0.0),
                "started_at": str(item.get("started_at") or ""),
                "ended_at": str(item.get("ended_at") or ""),
                "evidence_fact_ids": activity_fact_ids.get(int(item["id"]), []),
            }
            for item in activities
        ]
        fact_views = [
            {
                "id": int(item["id"]),
                "fact_type": str(item.get("fact_type") or ""),
                "summary": str(item.get("summary") or ""),
                "fact_topic": str(item.get("fact_topic") or ""),
                "event_time_key": str(item.get("event_time_key") or ""),
                "dialogue_time_key": str(item.get("dialogue_time_key") or ""),
                "entities": item.get("entities") or [],
                "confidence": float(item.get("confidence") or 0.0),
            }
            for item in facts
        ]
        fact_views_by_id = {int(item["id"]): item for item in fact_views}
        activity_cases = [
            {
                "activity": {
                    key: value for key, value in activity.items()
                    if key != "evidence_fact_ids"
                },
                "linked_facts": [
                    fact_views_by_id[fact_id]
                    for fact_id in activity["evidence_fact_ids"]
                ],
            }
            for activity in activity_views
        ]
        update_input = {
            "reference_time": self._timestamp_text(effective_reference_time),
            "runtime_context": dict(runtime_context or {}),
            "activity_cases": activity_cases,
            "related_intents": related_intents,
            "existing_states": [self._state_prompt_view(item) for item in existing_states],
        }
        source_text = "\n".join(
            [item["activity_text"] for item in activity_views]
            + [item["summary"] for item in fact_views]
        )
        template = (
            USER_ACTIVITY_STATE_UPDATE_PROMPT_EN
            if self._resolve_prompt_language(source_text) == "en"
            else USER_ACTIVITY_STATE_UPDATE_PROMPT_ZH
        )
        prompt = template
        for key, value in update_input.items():
            serialized = json.dumps(value, ensure_ascii=False, indent=2)
            prompt = prompt.replace("{" + key + "}", serialized)
        parsed = self._llm_client.parse_json_object_from_llm_text(
            self._llm_client.call_llm(prompt) or ""
        )
        report: Dict[str, Any] = {
            "status": "empty",
            "activity_count": len(activity_views),
            "linked_fact_count": len(fact_views),
            "related_intent_count": len(related_intents),
            "existing_state_count": len(existing_states),
            "operation_count": 0,
            "created": 0,
            "reinforced": 0,
            "revised": 0,
            "weakened": 0,
            "expired": 0,
            "state_ids": [],
        }
        if not isinstance(parsed, Mapping) or not isinstance(parsed.get("operations"), list):
            report.update({"status": "error", "error": "invalid_llm_activity_state_response"})
            self._log_update(report, task_kind=self.ACTIVITY_STATE_TASK_KIND)
            return report

        existing_by_id = {int(item["id"]): item for item in existing_states}
        allowed_evidence: Dict[tuple[str, int], Mapping[str, Any]] = {}
        for item in activity_views:
            allowed_evidence[("user_activity", int(item["id"]))] = item
        for item in fact_views:
            allowed_evidence[("fact", int(item["id"]))] = item
        for item in related_intents:
            key = (str(item["object_type"]), int(item["id"]))
            allowed_evidence[key] = item

        operations: List[Dict[str, Any]] = []
        seen: set[tuple[str, int, str, str, str]] = set()
        for raw in parsed["operations"][:24]:
            if not isinstance(raw, Mapping):
                continue
            if str(raw.get("operation") or "").strip().lower() not in {
                "create", "reinforce", "revise", "weaken", "expire",
            }:
                continue
            domain = self._normalize_state_domain(raw.get("state_domain"))
            if not domain or str(raw.get("state_type") or "") != "situational_state":
                continue
            normalized = self._normalize_operation(
                raw,
                existing_states=existing_by_id,
                allowed_evidence=allowed_evidence,
                state_domain=domain,
                reference_time=update_input["reference_time"],
            )
            if normalized is None:
                continue
            if normalized["operation"] == "create" and self._user_model_db.user_state_identity_exists(
                state_domain=normalized["state_domain"],
                state_type=normalized["state_type"],
                state_kind=normalized["state_kind"],
                state_topic=normalized["state_topic"],
            ):
                continue
            cited_activity_ids = {
                item["evidence_id"] for item in normalized["evidence"]
                if item["evidence_type"] == "user_activity"
            }
            if not cited_activity_ids:
                continue
            cited_activity_fact_ids = {
                fact_id
                for activity_id in cited_activity_ids
                for fact_id in activity_fact_ids.get(activity_id, [])
            }
            if any(
                item["evidence_type"] == "fact"
                and item["evidence_id"] not in cited_activity_fact_ids
                for item in normalized["evidence"]
            ):
                continue
            dedupe_key = (
                str(normalized["operation"]),
                int(normalized.get("target_state_id") or 0),
                str(normalized["state_domain"]),
                str(normalized["state_kind"]),
                str(normalized["state_topic"]),
            )
            if dedupe_key not in seen:
                seen.add(dedupe_key)
                operations.append(normalized)

        if not operations:
            report["status"] = "ok"
            self._log_update(report, task_kind=self.ACTIVITY_STATE_TASK_KIND)
            return report
        with self._user_model_db.transaction():
            applied = self.apply_operations(
                operations=operations,
                reference_time=update_input["reference_time"],
                decision_source="user_activity_situational_state_v1",
            )
        report.update(applied)
        report["status"] = "ok"
        self._log_update(report, task_kind=self.ACTIVITY_STATE_TASK_KIND)
        return report

    def process_user_observed_pattern_state_update_task(
        self,
        *,
        changed_observed_pattern_ids: Optional[Sequence[int]] = None,
        affected_observed_pattern_ids: Optional[Sequence[int]] = None,
        trigger_user_action_signal_ids: Optional[Sequence[int]] = None,
        reference_timestamp: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Project changed observed behavior into recurring states by domain."""
        changed_pattern_ids = list(changed_observed_pattern_ids or [])
        affected_pattern_ids = list(affected_observed_pattern_ids or [])
        trigger_signal_ids = list(trigger_user_action_signal_ids or [])
        reference_time = reference_timestamp
        owner_id = int(self._user_memory_db.add_entity_names([self._world_owner_entity_name])[
            self._world_owner_entity_name
        ])
        changed_ids = {int(value) for value in changed_pattern_ids if int(value) > 0}
        affected_ids = {int(value) for value in affected_pattern_ids if int(value) > 0}
        trigger_ids = {int(value) for value in trigger_signal_ids if int(value) > 0}
        patterns = self._user_memory_db.get_user_observed_patterns_by_ids(changed_ids | affected_ids)
        patterns = [
            item for item in patterns
            if int(item.get("subject_entity_id") or 0) == owner_id
            and self._normalize_state_domain(item.get("primary_domain"))
        ]
        events = self._user_memory_db.get_user_observed_pattern_events_for_signal_ids(trigger_ids)
        pattern_by_id = {int(item["id"]): item for item in patterns}
        events = [
            item for item in events
            if int(item["target_pattern_id"]) in pattern_by_id
        ]
        representative_ids = {
            int(signal_id)
            for pattern in patterns
            for signal_id in (
                (pattern.get("support_signal_ids") or [])[:1]
                + (pattern.get("support_signal_ids") or [])[-3:]
            )
        }
        signal_by_id = {
            int(item["signal_id"]): item
            for item in self._user_memory_db.get_user_action_signals_by_ids(
                representative_ids | trigger_ids
            )
        }
        fact_by_id = {
            int(item["id"]): item
            for item in self._user_memory_db.get_memory_facts_by_ids([
                int(item["fact_id"]) for item in signal_by_id.values()
            ])
        }

        def signal_view(signal_id: int) -> Dict[str, Any]:
            signal = signal_by_id[signal_id]
            fact = fact_by_id.get(int(signal["fact_id"])) or {}
            return {
                "signal_id": signal_id,
                "action_text": str(signal.get("action_text") or ""),
                "action_kind": str(signal.get("action_kind") or ""),
                "event_time": str(signal.get("event_time_key") or ""),
                "location": str(signal.get("location") or ""),
                "fact_id": int(signal["fact_id"]),
                "fact_summary": str(fact.get("summary") or ""),
            }

        def pattern_view(pattern: Mapping[str, Any]) -> Dict[str, Any]:
            signal_ids = pattern.get("support_signal_ids") or []
            representative = list(dict.fromkeys(signal_ids[:1] + signal_ids[-3:]))
            return {
                "id": int(pattern["id"]),
                "primary_domain": str(pattern.get("primary_domain") or ""),
                "pattern_text": str(pattern.get("pattern_text") or ""),
                "condition_text": str(pattern.get("condition_text") or ""),
                "confidence": float(pattern.get("confidence") or 0.0),
                "support_signal_count": len(signal_ids),
                "supporting_actions": [
                    signal_view(int(value)) for value in representative
                    if int(value) in signal_by_id
                ],
            }

        report: Dict[str, Any] = {
            "status": "empty", "changed_pattern_count": len(changed_ids),
            "affected_pattern_count": len(affected_ids), "domain_count": 0,
            "operation_count": 0, "created": 0, "reinforced": 0,
            "revised": 0, "weakened": 0, "expired": 0,
            "state_ids": [], "domain_reports": {},
        }
        domains = sorted({str(item["primary_domain"]) for item in patterns})
        report["domain_count"] = len(domains)
        failed = 0
        for domain in domains:
            domain_patterns = {
                int(item["id"]): item for item in patterns
                if item["primary_domain"] == domain
            }
            domain_changed = changed_ids & domain_patterns.keys()
            domain_affected = affected_ids & domain_patterns.keys()
            domain_events = [
                item for item in events
                if int(item["target_pattern_id"]) in domain_patterns
            ]
            domain_trigger_ids = {
                int(signal_id)
                for event in domain_events
                for signal_id in event["support_signal_ids"]
                if int(signal_id) in trigger_ids and int(signal_id) in signal_by_id
            }
            domain_trigger_ids.update({
                signal_id for signal_id in trigger_ids
                if signal_id in signal_by_id
                and any(signal_id in (item.get("support_signal_ids") or [])
                        for item in domain_patterns.values())
            })
            existing_states = self._user_model_db.get_user_states(
                state_domains=[domain],
                state_types=["recurring_state"], statuses=["active", "weakened"],
                limit=self._update_limit + 1,
            )
            if len(existing_states) > self._update_limit:
                failed += 1
                report["domain_reports"][domain] = {
                    "status": "error", "error": "observed_pattern_state_input_limit_exceeded",
                }
                continue
            state_evidence = self._user_model_db.get_user_state_evidence_items(
                [int(item["id"]) for item in existing_states],
                limit=max(480, self._update_limit * 12),
            )
            claim_ids = {
                int(item["evidence_id"]) for item in state_evidence
                if item["evidence_type"] == "user_claim"
            }
            claim_by_id = {
                int(item["id"]): item
                for item in self._user_memory_db.get_user_claims_by_ids(sorted(claim_ids))
            }
            supporting_claims = [
                self._claim_prompt_view(item, []) for item in claim_by_id.values()
            ]
            explicitly_supported_state_ids = {
                int(item["state_id"])
                for item in state_evidence
                if item["evidence_type"] == "user_claim"
                and item["role"] in {"support", "change_trigger"}
                and (claim := claim_by_id.get(int(item["evidence_id"])))
                and claim.get("claim_origin") == "explicit"
                and claim.get("status") == "active"
            }
            update_input = {
                "reference_time": self._timestamp_text(reference_time),
                "state_domain": domain,
                "changed_patterns": [
                    pattern_view(domain_patterns[value]) for value in sorted(domain_changed)
                ],
                "affected_patterns": [
                    pattern_view(domain_patterns[value])
                    for value in sorted(domain_affected - domain_changed)
                ],
                "relation_events": [
                    {
                        "target_pattern_id": int(item["target_pattern_id"]),
                        "trigger_pattern_id": int(item.get("trigger_pattern_id") or 0),
                        "semantic_relation": str(item["semantic_relation"]),
                        "support_signal_ids": item["support_signal_ids"],
                        "candidate_snapshot": item.get("candidate_snapshot") or {},
                        "semantic_reason": str(item.get("semantic_reason") or ""),
                    }
                    for item in domain_events
                ],
                "trigger_signals": [
                    signal_view(value) for value in sorted(domain_trigger_ids)
                ],
                "existing_states": [
                    self._state_prompt_view(item) for item in existing_states
                ],
                "state_evidence": [
                    {"state_id": int(item["state_id"]),
                     "evidence_type": str(item["evidence_type"]),
                     "evidence_id": int(item["evidence_id"]),
                     "role": str(item["role"])}
                    for item in state_evidence
                ],
                "supporting_claims": supporting_claims,
            }
            source = "\n".join(
                item["pattern_text"] for item in update_input["changed_patterns"]
            )
            template = (
                USER_OBSERVED_PATTERN_STATE_UPDATE_PROMPT_EN
                if self._resolve_prompt_language(source) == "en"
                else USER_OBSERVED_PATTERN_STATE_UPDATE_PROMPT_ZH
            )
            prompt = template
            for key, value in update_input.items():
                prompt = prompt.replace(
                    "{" + key + "}",
                    json.dumps(value, ensure_ascii=False, indent=2)
                    if isinstance(value, (dict, list)) else str(value or ""),
                )
            parsed = self._llm_client.parse_json_object_from_llm_text(
                self._llm_client.call_llm(prompt) or ""
            )
            if not isinstance(parsed, Mapping) or not isinstance(parsed.get("operations"), list):
                failed += 1
                report["domain_reports"][domain] = {
                    "status": "error", "error": "invalid_llm_observed_pattern_state_response",
                }
                continue
            existing_by_id = {int(item["id"]): item for item in existing_states}
            allowed = {
                ("user_observed_pattern", pattern_id): pattern
                for pattern_id, pattern in domain_patterns.items()
            }
            allowed.update({
                ("user_action_signal", signal_id): signal_by_id[signal_id]
                for signal_id in domain_trigger_ids
            })
            linked_patterns_by_state = {
                int(item["state_id"]): set() for item in state_evidence
            }
            for item in state_evidence:
                if item["evidence_type"] == "user_observed_pattern":
                    linked_patterns_by_state[int(item["state_id"])].add(
                        int(item["evidence_id"])
                    )
            contradicting_signals_by_target: Dict[int, set[int]] = {}
            for item in domain_events:
                if item["semantic_relation"] == "contradicts":
                    contradicting_signals_by_target.setdefault(
                        int(item["target_pattern_id"]), set(),
                    ).update(int(value) for value in item["support_signal_ids"])
            operations: List[Dict[str, Any]] = []
            seen_targets: set[tuple[Any, ...]] = set()
            for raw in parsed["operations"][:24]:
                normalized = self._normalize_operation(
                    raw, existing_states=existing_by_id,
                    allowed_evidence=allowed, state_domain=domain,
                    reference_time=update_input["reference_time"],
                )
                if not normalized or normalized["state_type"] != "recurring_state":
                    continue
                action = normalized["operation"]
                if action == "expire":
                    continue
                evidence = normalized["evidence"]
                has_changed_pattern = any(
                    item["evidence_type"] == "user_observed_pattern"
                    and item["evidence_id"] in domain_changed
                    for item in evidence
                )
                has_trigger_signal = any(
                    item["evidence_type"] == "user_action_signal"
                    and item["evidence_id"] in domain_trigger_ids
                    for item in evidence
                )
                if not (has_changed_pattern or has_trigger_signal):
                    continue
                if action in {"create", "reinforce", "revise"} and not any(
                    item["evidence_type"] == "user_observed_pattern"
                    and item["evidence_id"] in domain_changed
                    and item["role"] in {"support", "change_trigger"}
                    for item in evidence
                ):
                    continue
                target_counterexample_signal_ids = set().union(*[
                    contradicting_signals_by_target.get(pattern_id, set())
                    for pattern_id in linked_patterns_by_state.get(
                        normalized["target_state_id"], set()
                    )
                ])
                independent_counterexample_times = {
                    str(signal_by_id[signal_id].get("event_time_key") or "").strip()
                    for signal_id in target_counterexample_signal_ids
                    if signal_id in signal_by_id
                    and str(signal_by_id[signal_id].get("event_time_key") or "").strip()
                }
                if action == "weaken" and not (
                    has_trigger_signal
                    and normalized["target_state_id"] not in explicitly_supported_state_ids
                    and len(independent_counterexample_times) >= 2
                    and any(
                        item["evidence_type"] == "user_action_signal"
                        and item["role"] == "counterexample"
                        and item["evidence_id"] in target_counterexample_signal_ids
                        for item in evidence
                    )
                ):
                    continue
                if action == "create" and self._user_model_db.user_state_identity_exists(
                    state_domain=domain,
                    state_type="recurring_state",
                    state_kind=normalized["state_kind"],
                    state_topic=normalized["state_topic"],
                ):
                    continue
                target = (
                    ("id", normalized["target_state_id"])
                    if normalized["target_state_id"]
                    else ("new", normalized["state_kind"], normalized["state_topic"])
                )
                if target in seen_targets:
                    continue
                seen_targets.add(target)
                operations.append(normalized)
            if operations:
                with self._user_model_db.transaction():
                    applied = self.apply_operations(
                        operations=operations,
                        reference_time=update_input["reference_time"],
                        decision_source="user_observed_pattern_state_v1",
                    )
                for key in ("operation_count", "created", "reinforced", "revised", "weakened", "expired"):
                    report[key] += int(applied.get(key) or 0)
                report["state_ids"].extend(applied["state_ids"])
            report["domain_reports"][domain] = {
                "status": "ok", "changed_pattern_count": len(domain_changed),
                "affected_pattern_count": len(domain_affected),
                "operation_count": len(operations),
            }
        report["state_ids"] = sorted(set(report["state_ids"]))
        report["status"] = (
            "error" if failed == len(domains) and domains
            else "partial" if failed else "ok" if domains else "empty"
        )
        self._log_update(report, task_kind=self.OBSERVED_PATTERN_STATE_TASK_KIND)
        return report

    def _resolve_prompt_language(self, text: str) -> str:
        if self._prompt_language_mode in {"en", "english", "force_en"}:
            return "en"
        if self._prompt_language_mode in {"zh", "chinese", "force_zh"}:
            return "zh"
        if re.search(r"[\u4e00-\u9fff]", str(text or "")):
            return "zh"
        return "zh"

    def _log_update(self, payload: Dict[str, Any], *, task_kind: str) -> None:
        record = {"scope": task_kind, "event": "finish", "payload": payload}
        self._logger.info(
            "\n%s",
            json.dumps(record, ensure_ascii=False, sort_keys=False, indent=2, default=str),
        )

    def _build_user_claim_update_inputs(
        self, *, world_owner_entity_id: int, new_claim_ids: Sequence[int],
        claim_relations: Sequence[Mapping[str, Any]],
        reference_time: Optional[Any],
    ) -> Dict[str, Dict[str, Any]]:
        claims_by_id = {
            int(claim["id"]): claim
            for claim in self._user_memory_db.get_user_claims_by_ids(new_claim_ids)
            if int(claim.get("subject_entity_id") or 0) == world_owner_entity_id
            and self._normalize_state_domain(claim.get("primary_domain"))
        }
        relation_refs = {
            (int(item["new_claim_id"]), int(item["relation_event_id"]))
            for item in claim_relations
        }
        new_claim_id_set = {int(claim_id) for claim_id in new_claim_ids}
        if any(claim_id not in new_claim_id_set or event_id <= 0
               for claim_id, event_id in relation_refs):
            raise ValueError("Claim relation reference does not belong to this update")
        event_ids = sorted({event_id for _, event_id in relation_refs})
        relation_events = self._user_memory_db.get_user_claim_relation_events_by_ids(
            event_ids,
        )
        loaded_refs = {
            (int(event["trigger_claim_id"]), int(event["id"]))
            for event in relation_events
        }
        if loaded_refs != relation_refs:
            raise ValueError("Claim relation event is missing or mismatched")
        relations_by_trigger: Dict[int, List[Dict[str, Any]]] = {}
        for event in relation_events:
            relations_by_trigger.setdefault(int(event["trigger_claim_id"]), []).append(event)
        by_domain: Dict[str, List[int]] = {}
        for claim_id, claim in claims_by_id.items():
            by_domain.setdefault(str(claim["primary_domain"]), []).append(claim_id)
        result: Dict[str, Dict[str, Any]] = {}
        evidence_limit = max(480, self._update_limit * 12)
        effective_reference_time = self._timestamp_text(reference_time)
        for domain, changed_ids in by_domain.items():
            relations = [
                event for claim_id in changed_ids
                for event in relations_by_trigger.get(claim_id, [])
            ]
            states = self._user_model_db.get_user_states(
                state_domains=[domain],
                state_types=["enduring_state", "recurring_state"],
                statuses=["active", "weakened"], limit=self._update_limit + 1,
                include_evidence=True,
            )
            claim_evidence = [
                item for state in states for item in state["evidence_items"]
                if item["evidence_type"] == "user_claim"
            ]
            pattern_evidence = [
                item for state in states for item in state["evidence_items"]
                if item["evidence_type"] == "user_observed_pattern"
            ]
            required_claim_ids = (
                set(changed_ids)
                | {int(event["target_claim_id"]) for event in relations}
                | {int(item["evidence_id"]) for item in claim_evidence}
            )
            claims_by_id.update({
                int(claim["id"]): claim
                for claim in self._user_memory_db.get_user_claims_by_ids(
                    sorted(required_claim_ids - claims_by_id.keys()),
                )
                if int(claim.get("subject_entity_id") or 0) == world_owner_entity_id
            })
            fact_evidence = self._user_memory_db.get_user_claim_evidence_fact_ids(
                sorted(required_claim_ids & claims_by_id.keys()), limit=evidence_limit,
            )
            claim_views_by_id = {
                claim_id: self._claim_prompt_view(
                    claims_by_id[claim_id], fact_evidence.get(claim_id, []),
                )
                for claim_id in required_claim_ids if claim_id in claims_by_id
            }
            patterns_by_id = {
                int(pattern["id"]): {
                    "id": int(pattern["id"]),
                    "pattern_text": str(pattern.get("pattern_text") or ""),
                    "condition_text": str(pattern.get("condition_text") or ""),
                    "confidence": float(pattern.get("confidence") or 0.0),
                }
                for pattern in self._user_memory_db.get_user_observed_patterns_by_ids(
                    sorted({int(item["evidence_id"]) for item in pattern_evidence}),
                )
            }
            state_contexts = [
                {
                    "state": self._state_prompt_view(state),
                    "claim_evidence": [
                        {"claim": claim_views_by_id[int(item["evidence_id"])],
                         "role": str(item.get("role") or ""),
                         "weight": float(item.get("weight") or 0.0)}
                        for item in state["evidence_items"]
                        if item["evidence_type"] == "user_claim"
                        and int(item["evidence_id"]) in claim_views_by_id
                    ],
                    "pattern_evidence": [
                        {"pattern_id": int(item["evidence_id"]),
                         "pattern": patterns_by_id.get(int(item["evidence_id"])),
                         "role": str(item.get("role") or "")}
                        for item in state["evidence_items"]
                        if item["evidence_type"] == "user_observed_pattern"
                    ],
                }
                for state in states
            ]
            new_claim_cases = [
                {
                    "new_claim": claim_views_by_id[claim_id],
                    "related_claims": [
                        {
                            "relation_event": {key: event.get(key) for key in (
                                "id", "target_claim_id", "trigger_claim_id",
                                "semantic_relation", "semantic_confidence",
                                "effective_at", "semantic_reason",
                            )},
                            "existing_claim": claim_views_by_id[int(event["target_claim_id"])],
                        }
                        for event in relations_by_trigger.get(claim_id, [])
                        if int(event["target_claim_id"]) in claim_views_by_id
                    ],
                }
                for claim_id in sorted(changed_ids)
            ]
            result[domain] = {
                "input_truncated": (
                    len(relations) > max(400, self._update_limit * 8)
                    or len(states) > self._update_limit
                    or len(claim_evidence) > evidence_limit
                    or len(pattern_evidence) > evidence_limit
                ),
                "state_domain": domain,
                "reference_time": effective_reference_time,
                "new_claim_cases": new_claim_cases,
                "state_contexts": state_contexts,
            }
        return result

    def _normalize_user_claim_joint_decision(
        self, parsed: Mapping[str, Any], *, update_input: Mapping[str, Any],
    ) -> Optional[tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
        raw_updates = parsed.get("claim_updates")
        raw_operations = parsed.get("operations")
        if not isinstance(raw_updates, list) or not isinstance(raw_operations, list):
            return None
        if len(raw_updates) > 24 or len(raw_operations) > 24:
            return None
        changed_ids: set[int] = set()
        related_by_id: Dict[int, Dict[str, Any]] = {}
        events_by_id: Dict[int, Dict[str, Any]] = {}
        claims: Dict[int, Dict[str, Any]] = {}
        for case in update_input["new_claim_cases"]:
            new_claim = case["new_claim"]
            claim_id = int(new_claim["id"])
            changed_ids.add(claim_id)
            claims[claim_id] = new_claim
            for relation in case["related_claims"]:
                old_claim = relation["existing_claim"]
                event = relation["relation_event"]
                old_claim_id = int(old_claim["id"])
                related_by_id[old_claim_id] = old_claim
                claims[old_claim_id] = old_claim
                events_by_id[int(event["id"])] = event
        existing_states = {}
        state_contexts_by_id = {}
        for context in update_input["state_contexts"]:
            state = context["state"]
            state_id = int(state["id"])
            existing_states[state_id] = state
            state_contexts_by_id[state_id] = context
            for evidence in context["claim_evidence"]:
                claim = evidence["claim"]
                claims[int(claim["id"])] = claim
        claim_updates: List[Dict[str, Any]] = []
        seen_claims: set[int] = set()
        for raw in raw_updates:
            if not isinstance(raw, Mapping):
                return None
            claim_id = self._positive_int(raw.get("claim_id"))
            event_id = self._positive_int(raw.get("relation_event_id"))
            status = str(raw.get("new_status") or "").strip()
            event = events_by_id.get(event_id)
            old_claim = related_by_id.get(claim_id)
            relation = str((event or {}).get("semantic_relation") or "")
            if (
                not old_claim or not event or claim_id in seen_claims
                or int(event["target_claim_id"]) != claim_id
                or int(event["trigger_claim_id"]) not in changed_ids
                or str(old_claim.get("status") or "") not in {"active", "weakened"}
                or status not in {"weakened", "superseded"}
                or (status == "superseded" and relation != "supersedes")
                or (status == "weakened" and relation not in {"contradicts", "supersedes"})
            ):
                return None
            seen_claims.add(claim_id)
            claim_updates.append({
                "claim_id": claim_id,
                "new_status": status,
                "relation_event_id": event_id,
                "trigger_claim_id": int(event["trigger_claim_id"]),
                "reason": self._compact(raw.get("reason"))[:300],
            })

        projected_status = {claim_id: str(item.get("status") or "") for claim_id, item in claims.items()}
        projected_status.update({
            item["claim_id"]: item["new_status"] for item in claim_updates
        })
        allowed_evidence = {("user_claim", claim_id): item for claim_id, item in claims.items()}
        operations: List[Dict[str, Any]] = []
        seen_states: set[tuple[str, ...]] = set()
        for raw in raw_operations:
            operation = self._normalize_operation(
                raw, existing_states=existing_states,
                allowed_evidence=allowed_evidence,
                state_domain=str(update_input["state_domain"]),
                reference_time=str(update_input["reference_time"]),
            )
            if not operation or operation["state_type"] not in {
                "enduring_state", "recurring_state",
            } or operation["operation"] == "no_change":
                return None
            if (operation["operation"] in {"weaken", "expire"}
                    and operation["state_type"] == "recurring_state"
                    and any(
                        item["role"] in {"support", "change_trigger"}
                        for item in state_contexts_by_id.get(
                            operation["target_state_id"], {}
                        ).get("pattern_evidence", [])
                    )):
                continue
            raw_evidence = raw.get("evidence") if isinstance(raw, Mapping) else None
            if not isinstance(raw_evidence, list) or len(raw_evidence) != len(operation["evidence"]):
                return None
            if not any(
                item["evidence_type"] == "user_claim"
                and item["evidence_id"] in changed_ids
                for item in operation["evidence"]
            ):
                return None
            if operation["operation"] in {"create", "reinforce", "revise"} and not any(
                item["role"] in {"support", "change_trigger"}
                and projected_status[item["evidence_id"]] in {"active", "weakened"}
                for item in operation["evidence"]
            ):
                return None
            if any(
                item["role"] == "support"
                and projected_status[item["evidence_id"]] not in {"active", "weakened"}
                for item in operation["evidence"]
            ):
                return None
            target_key = (
                "id", str(operation["target_state_id"])
            ) if operation["target_state_id"] else (
                operation["state_type"], operation["state_kind"],
                operation["state_topic"],
            )
            if target_key in seen_states:
                return None
            seen_states.add(target_key)
            if operation["operation"] == "create" and any(
                state["state_type"] == operation["state_type"]
                and state["state_kind"] == operation["state_kind"]
                and state["state_topic"] == operation["state_topic"]
                for state in existing_states.values()
            ):
                return None
            if operation["operation"] == "create" and self._user_model_db.user_state_identity_exists(
                state_domain=str(update_input["state_domain"]),
                state_type=operation["state_type"],
                state_kind=operation["state_kind"],
                state_topic=operation["state_topic"],
            ):
                return None
            operations.append(operation)
        return claim_updates, operations

    def process_user_claim_state_update_task(
        self,
        *,
        new_claim_ids: Optional[Sequence[int]] = None,
        claim_relations: Optional[Sequence[Mapping[str, Any]]] = None,
        reference_timestamp: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Update enduring and recurring states from changed user claims."""
        new_claim_ids = list(new_claim_ids or [])
        reference_time = reference_timestamp
        world_owner_entity_id = int(self._user_memory_db.add_entity_names([self._world_owner_entity_name])[
            self._world_owner_entity_name
        ])
        update_inputs = self._build_user_claim_update_inputs(
            world_owner_entity_id=world_owner_entity_id,
            new_claim_ids=new_claim_ids,
            claim_relations=list(claim_relations or []),
            reference_time=reference_time,
        )
        report: Dict[str, Any] = {
            "status": "empty", "domain_count": len(update_inputs),
            "changed_claim_count": sum(len(item["new_claim_cases"]) for item in update_inputs.values()),
            "changed_intent_object_count": 0,
            "existing_state_count": sum(len(item["state_contexts"]) for item in update_inputs.values()),
            "claim_status_update_count": 0, "operation_count": 0,
            "created": 0, "reinforced": 0, "revised": 0, "weakened": 0,
            "expired": 0, "state_ids": [], "domain_reports": {},
        }
        failed_count = 0
        for domain, update_input in update_inputs.items():
            if update_input["input_truncated"]:
                failed_count += 1
                report["domain_reports"][domain] = {
                    "status": "error", "error": "user_claim_state_input_limit_exceeded",
                }
                continue
            source = "\n".join(
                case["new_claim"]["claim_text"]
                for case in update_input["new_claim_cases"]
            )
            template = (
                USER_CLAIM_STATE_RESOLUTION_PROMPT_EN
                if self._resolve_prompt_language(source) == "en"
                else USER_CLAIM_STATE_RESOLUTION_PROMPT_ZH
            )
            prompt = template
            for key in ("reference_time", "state_domain", "new_claim_cases", "state_contexts"):
                value = update_input[key]
                prompt = prompt.replace("{" + key + "}", json.dumps(
                    value, ensure_ascii=False, indent=2,
                ) if isinstance(value, (dict, list)) else str(value or ""))
            parsed = self._llm_client.parse_json_object_from_llm_text(
                self._llm_client.call_llm(prompt) or ""
            )
            normalized = self._normalize_user_claim_joint_decision(
                parsed, update_input=update_input,
            ) if isinstance(parsed, Mapping) else None
            if normalized is None:
                failed_count += 1
                report["domain_reports"][domain] = {
                    "status": "error", "error": "invalid_llm_user_claim_state_decision",
                }
                continue
            claim_updates, operations = normalized
            # State is the projection; write it first. If this fails, source
            # claims remain unchanged and the scheduler can retry the batch.
            with self._user_model_db.transaction():
                applied = self.apply_operations(
                    operations=operations,
                    reference_time=update_input["reference_time"],
                    replace_claim_evidence=True,
                    decision_source=self.CLAIM_RESOLVER_VERSION,
                )
            # Memory and model databases cannot share an SQLite transaction.
            # Claim transitions are idempotent, so an interrupted second write
            # remains retryable through the scheduler's unacknowledged batch.
            with self._user_memory_db.transaction():
                changed_status_count = sum(
                    self._user_memory_db.apply_user_claim_status_decision(
                        claim_id=item["claim_id"], new_status=item["new_status"],
                        trigger_claim_id=item["trigger_claim_id"],
                        relation_event_id=item["relation_event_id"],
                        effective_at=str(update_input["reference_time"]),
                        reason=item["reason"], decision_source=self.CLAIM_RESOLVER_VERSION,
                    )
                    for item in claim_updates
                )
            report["claim_status_update_count"] += changed_status_count
            for key in ("operation_count", "created", "reinforced", "revised", "weakened", "expired"):
                report[key] += int(applied.get(key) or 0)
            report["state_ids"].extend(applied["state_ids"])
            report["domain_reports"][domain] = {
                "status": "ok", "changed_claim_count": len(update_input["new_claim_cases"]),
                "related_claim_count": len({
                    int(relation["existing_claim"]["id"])
                    for case in update_input["new_claim_cases"]
                    for relation in case["related_claims"]
                }),
                "claim_status_update_count": changed_status_count,
                **applied,
            }
        report["state_ids"] = sorted(set(report["state_ids"]))
        report["status"] = (
            "error" if failed_count == len(update_inputs) and update_inputs
            else "partial" if failed_count else "ok" if update_inputs else "empty"
        )
        if failed_count:
            report["failed_domain_count"] = failed_count
        self._log_update(report, task_kind=self.CLAIM_STATE_TASK_KIND)
        return report

    def apply_operations(
        self,
        *,
        operations: Sequence[Mapping[str, Any]],
        reference_time: Optional[Any] = None,
        replace_claim_evidence: bool = False,
        decision_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist validated operations and their evidence/event trail."""
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
            previous_claim_evidence = (
                self._user_model_db.get_user_state_evidence_items(
                    [target_id], evidence_type="user_claim",
                )
                if replace_claim_evidence and target_id else []
            )
            state_id, created, previous = self._user_model_db.upsert_user_state(
                target_state_id=target_id or None,
                state_domain=str(operation["state_domain"]),
                state_type=str(operation["state_type"]),
                state_kind=str(operation["state_kind"]),
                state_topic=str(operation["state_topic"]),
                state_text=str(operation["state_text"]),
                status=str(operation["status"]),
                confidence=float(operation["confidence"]),
                conditions=dict(operation.get("conditions") or {}),
                valid_from=str(operation.get("valid_from") or ""),
                valid_to=str(operation.get("valid_to") or ""),
                last_observed_at=effective_at,
                metadata={
                    "resolver_version": decision_source or self.RESOLVER_VERSION,
                    "last_operation": action,
                    "state_domain": str(operation["state_domain"]),
                },
            )
            current = self._user_model_db.get_user_states_by_ids([state_id])[0]
            evidence = [
                {**item, "state_id": state_id, "observed_at": effective_at}
                for item in operation.get("evidence") or []
            ]
            if replace_claim_evidence:
                self._user_model_db.replace_user_state_claim_evidence(
                    state_id=state_id, evidence=evidence,
                )
            else:
                self._user_model_db.upsert_user_state_evidence(evidence)
            self._user_model_db.record_user_state_event(
                state_id=state_id,
                operation=action,
                previous_payload={
                    **(previous or {}),
                    "claim_evidence": previous_claim_evidence,
                } if replace_claim_evidence else previous or {},
                new_payload={
                    **current,
                    "claim_evidence": evidence,
                } if replace_claim_evidence else current,
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
                decision_source=decision_source or self.RESOLVER_VERSION,
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
        state_domains: Optional[Sequence[str]] = None,
        state_types: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        return self._user_model_db.get_user_states(
            state_domains=state_domains,
            state_types=state_types,
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
        state_type = str(raw.get("state_type") or "").strip().lower()
        if action not in self.OPERATIONS or state_type not in self.STATE_TYPES:
            return None
        target_id = self._positive_int(raw.get("target_state_id"))
        target = existing_states.get(target_id)
        if action in {"reinforce", "revise", "weaken", "expire"} and target is None:
            return None
        if target is not None and str(target.get("state_type") or "") != state_type:
            return None
        if target is not None and str(target.get("state_domain") or "") != state_domain:
            return None
        if action == "create" and target_id:
            return None
        state_kind = self._normalize_state_kind(
            (target or {}).get("state_kind") if target is not None
            else raw.get("state_kind")
        )
        if target is None and state_kind not in self.STATE_KINDS_BY_TYPE[state_type]:
            return None
        state_text = self._compact(raw.get("state_text") or (target or {}).get("state_text"))
        evidence = self._normalize_evidence(raw.get("evidence"), allowed_evidence)
        if not state_kind or not state_text or not evidence:
            return None
        if target is not None:
            state_topic = str(target.get("state_topic") or "")
        else:
            state_topic = self._normalize_state_topic(
                raw.get("state_topic")
            )
            if not state_topic:
                trigger = next(
                    (item for item in evidence if item["role"] == "change_trigger"),
                    evidence[0],
                )
                state_topic = (
                    f"{state_kind}:{trigger['evidence_type']}:{trigger['evidence_id']}"
                )
        status = {
            "create": "active", "reinforce": "active", "revise": "active",
            "weaken": "weakened", "expire": "expired",
        }[action]
        conditions = raw.get("conditions")
        if not isinstance(conditions, Mapping):
            conditions = dict((target or {}).get("conditions") or {})
        return {
            "operation": action,
            "target_state_id": target_id,
            "state_domain": state_domain,
            "state_type": state_type,
            "state_kind": state_kind,
            "state_topic": state_topic,
            "state_text": state_text[:600],
            "status": status,
            "confidence": self._clamp(raw.get("confidence"), fallback=(target or {}).get("confidence", 0.7)),
            "conditions": dict(conditions),
            "valid_from": self._text(raw.get("valid_from") or (target or {}).get("valid_from") or reference_time),
            "valid_to": self._text(raw.get("valid_to") or (target or {}).get("valid_to")),
            "evidence": evidence,
            "reason": self._compact(raw.get("reason"))[:300],
        }

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
            "state_type": str(state.get("state_type") or ""),
            "state_kind": str(state.get("state_kind") or ""),
            "state_topic": str(state.get("state_topic") or ""),
            "state_text": str(state.get("state_text") or ""),
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
    def _normalize_state_kind(value: Any) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", str(value or "").lower()).strip("_")[:96]

    @staticmethod
    def _normalize_state_topic(value: Any) -> str:
        return re.sub(r"[^\w]+", "_", str(value or "").casefold()).strip("_")[:96]

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
