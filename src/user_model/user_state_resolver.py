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

from semantic_domains import SEMANTIC_DOMAINS
from prompts_en import (
    USER_STATE_UPDATE_FROM_USER_THREAD_PROMPT_EN,
    USER_STATE_UPDATE_FROM_USER_CLAIM_PROMPT_EN,
    USER_STATE_UPDATE_FROM_USER_OBSERVED_PATTERN_PROMPT_EN,
)
from prompts_zh import (
    USER_STATE_UPDATE_FROM_USER_THREAD_PROMPT_ZH,
    USER_STATE_UPDATE_FROM_USER_CLAIM_PROMPT_ZH,
    USER_STATE_UPDATE_FROM_USER_OBSERVED_PATTERN_PROMPT_ZH,
)

if TYPE_CHECKING:
    from memory.memory_database import UserMemoryDB
    from llm_client import LLMClient
    from task_scheduler import TaskScheduler
    from .user_model_database import UserModelDB


class UserStateResolver:
    """Project claims, observed patterns, and thread changes into user states."""

    RESOLVER_VERSION = "user_state_incremental_v1"
    CLAIM_RESOLVER_VERSION = "user_claim_state_only_v1"
    STATE_TYPES = {"enduring_state", "recurring_state", "situational_state"}
    STATE_DOMAINS = SEMANTIC_DOMAINS
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
    EVIDENCE_TYPES = {
        "user_claim", "goal", "plan", "work_item", "user_activity", "fact",
        "user_observed_pattern", "user_action_signal", "user_thread", "user_thread_event",
    }
    EVIDENCE_ROLES = {"support", "change_trigger", "counterexample", "context"}

    USER_STATE_UPDATE_FROM_USER_CLAIMS_TASK_KIND = "user_state_update_from_user_claims"
    USER_STATE_UPDATE_FROM_USER_THREADS_TASK_KIND = "user_state_update_from_user_threads"
    USER_STATE_UPDATE_FROM_USER_OBSERVED_PATTERNS_TASK_KIND = "user_state_update_from_user_observed_patterns"

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
        self._world_owner_entity_id = int(self._user_memory_db.add_entity_names(
            [self._world_owner_entity_name]
        )[self._world_owner_entity_name])
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
            self.USER_STATE_UPDATE_FROM_USER_CLAIMS_TASK_KIND,
            self.process_user_state_update_from_user_claims_task,
        )
        task_scheduler.register_handler(
            self.USER_STATE_UPDATE_FROM_USER_THREADS_TASK_KIND,
            self.process_user_state_update_from_user_threads_task,
        )
        task_scheduler.register_handler(
            self.USER_STATE_UPDATE_FROM_USER_OBSERVED_PATTERNS_TASK_KIND,
            self.process_user_state_update_from_user_observed_patterns_task,
        )

    def set_logger(self, logger: Optional[logging.Logger]) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def _build_user_state_update_inputs_using_threads(
        self, thread_changes: Sequence[Mapping[str, Any]], *,
        reference_timestamp: Optional[Any] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build thread-centered cases from fixed updates, not all thread history."""
        changes_by_id: Dict[int, Dict[str, Any]] = {}
        for change in thread_changes:
            thread = dict(change["thread"])
            thread_id = self._positive_int(thread.get("id"))
            if not thread_id:
                raise ValueError("Expected a positive thread ID")
            if thread_id not in changes_by_id:
                changes_by_id[thread_id] = {
                    "thread": thread, "previous_thread": change.get("previous_thread"),
                    "thread_event_ids": set(), "evidence_activity_ids": set(),
                    "evidence_fact_ids": set(), "reference_time": change["reference_time"],
                }
            merged = changes_by_id[thread_id]
            merged["thread"] = thread
            merged["reference_time"] = change["reference_time"]
            for field in ("thread_event_ids", "evidence_activity_ids", "evidence_fact_ids"):
                merged[field].update(int(value) for value in change.get(field) or [])

        event_ids = sorted({value for change in changes_by_id.values()
                            for value in change["thread_event_ids"]})
        if {int(thread["id"]) for thread in self._user_memory_db.get_user_threads_by_ids(
            list(changes_by_id)
        )} != set(changes_by_id):
            raise ValueError("Unknown thread IDs")
        events = self._user_memory_db.get_user_thread_events_by_ids(event_ids)
        if {int(event["id"]) for event in events} != set(event_ids):
            raise ValueError("Unknown thread change event IDs")
        events_by_id = {int(event["id"]): event for event in events}
        activity_ids = sorted({value for change in changes_by_id.values()
                               for value in change["evidence_activity_ids"]})
        activities = self._user_memory_db.get_user_activity_items_by_ids(activity_ids)
        if {int(item["id"]) for item in activities} != set(activity_ids):
            raise ValueError("Unknown thread activity evidence IDs")
        activities_by_id = {int(item["id"]): item for item in activities}
        fact_ids = sorted({value for change in changes_by_id.values()
                           for value in change["evidence_fact_ids"]})
        facts_by_id = {
            int(item["id"]): item
            for item in self._user_memory_db.get_memory_facts_by_ids(fact_ids)
        }
        if set(facts_by_id) != set(fact_ids):
            raise ValueError("Unknown thread fact evidence IDs")

        states = self._user_model_db.get_user_states(
            state_types=["situational_state"], statuses=["active", "weakened"],
            limit=max(40, self._update_limit * 2), include_evidence=True,
        )
        states_by_id = {int(state["id"]): state for state in states}
        for state in self._user_model_db.get_user_states_for_thread_ids(list(changes_by_id)):
            states_by_id[int(state["id"])] = state

        cases = []
        detailed_state_ids = set()
        for thread_id, change in changes_by_id.items():
            case_events = [events_by_id[value] for value in sorted(change["thread_event_ids"])]
            if any(int(event["thread_id"]) != thread_id for event in case_events):
                raise ValueError("Thread event belongs to another thread")
            case_activities = [
                activities_by_id[value] for value in sorted(change["evidence_activity_ids"])
            ]
            case_fact_ids = change["evidence_fact_ids"]
            linked_facts = {int(value) for activity in case_activities
                            for value in activity.get("evidence_fact_ids") or []}
            if not case_fact_ids.issubset(linked_facts):
                raise ValueError("Thread facts are not linked to the supplied activities")
            for event in case_events:
                if (not set(event["evidence_activity_ids"]).issubset(change["evidence_activity_ids"])
                        or not set(event["evidence_fact_ids"]).issubset(case_fact_ids)):
                    raise ValueError("Thread event evidence exceeds this update")

            linked_state_ids = [
                state_id for state_id, state in states_by_id.items()
                if any(
                    (item["evidence_type"] == "user_thread"
                     and int(item["evidence_id"]) == thread_id)
                    or (item["evidence_type"] == "fact"
                        and int(item["evidence_id"]) in case_fact_ids)
                    or (item["evidence_type"] == "user_activity"
                        and int(item["evidence_id"]) in change["evidence_activity_ids"])
                    for item in state.get("evidence_items") or []
                )
            ]
            detailed_state_ids.update(linked_state_ids)
            cases.append({
                "thread": change["thread"], "previous_thread": change["previous_thread"],
                "reference_time": str(change["reference_time"]),
                "change_events": [
                    {key: event.get(key) for key in (
                        "id", "thread_id", "event_type", "summary", "occurred_at",
                        "known_at", "evidence_activity_ids", "evidence_fact_ids",
                    )} for event in case_events
                ],
                "activity_cases": [
                    {
                        "activity": {key: activity.get(key) for key in (
                            "id", "activity_text", "user_role", "confidence",
                            "started_at", "ended_at",
                        )},
                        "linked_facts": [
                            {key: facts_by_id[int(value)].get(key) for key in (
                                "id", "summary", "fact_topic", "event_time_key",
                                "dialogue_time_key", "confidence",
                            )}
                            for value in activity.get("evidence_fact_ids") or []
                            if int(value) in case_fact_ids
                        ],
                    } for activity in case_activities
                ],
                "linked_state_ids": linked_state_ids,
            })
        times = [str(change["reference_time"]) for change in changes_by_id.values()]
        effective_time = reference_timestamp or (
            max(times, key=lambda value: datetime.fromisoformat(
                value.replace("Z", "+00:00")
            ).astimezone()) if times else None
        )
        return {
            "update_input": {
                "reference_time": self._timestamp_text(effective_time),
                "runtime_context": dict(runtime_context or {}),
                "thread_cases": cases,
                "state_contexts": [
                    self._build_state_context(
                        state, detailed=state_id in detailed_state_ids,
                        world_owner_entity_id=self._world_owner_entity_id,
                    ) for state_id, state in states_by_id.items()
                ],
            },
            "existing_states": states_by_id,
        }

    @staticmethod
    def _format_thread_cases_view_for_user_state_prompt(
        thread_cases: Sequence[Mapping[str, Any]],
    ) -> str:
        """Render compact thread cases without changing validation snapshots."""
        if not thread_cases:
            return "[]"
        lines = []
        for case in thread_cases:
            previous = case["previous_thread"]
            previous_view = None if previous is None else {
                key: previous[key]
                for key in ("title", "identity_description", "current_summary", "status")
                if key in previous and previous[key] != case["thread"].get(key)
            }
            lines.append(json.dumps({
                "thread": case["thread"],
                "previous_thread": previous_view,
                "reference_time": case["reference_time"],
                "change_events": [
                    {key: value for key, value in event.items()
                     if key != "thread_id" and not (
                         key == "known_at" and value == case["reference_time"]
                     )}
                    for event in case["change_events"]
                ],
                "activity_cases": [
                    {
                        "activity": activity_case["activity"],
                        "linked_facts": [
                            {key: value for key, value in fact.items() if key != "fact_topic"}
                            for fact in activity_case["linked_facts"]
                        ],
                    } for activity_case in case["activity_cases"]
                ],
                "linked_state_ids": case["linked_state_ids"],
            }, ensure_ascii=False, separators=(",", ":")))
        return "[\n" + ",\n".join(lines) + "\n]"

    def _normalize_user_thread_state_operations(
        self, raw_operations: Sequence[Any], *,
        update_input: Mapping[str, Any],
        existing_states: Mapping[int, Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Limit this route to situational states and each cited thread's evidence."""
        cases = {int(case["thread"]["id"]): case for case in update_input["thread_cases"]}
        allowed: Dict[tuple[str, int], Mapping[str, Any]] = {}
        scope_by_thread: Dict[int, set[tuple[str, int]]] = {}
        for thread_id, case in cases.items():
            scope = {("user_thread", thread_id)}
            allowed[("user_thread", thread_id)] = case["thread"]
            for event in case["change_events"]:
                key = ("user_thread_event", int(event["id"]))
                scope.add(key)
                allowed[key] = event
            for activity_case in case["activity_cases"]:
                activity = activity_case["activity"]
                key = ("user_activity", int(activity["id"]))
                scope.add(key)
                allowed[key] = activity
                for fact in activity_case["linked_facts"]:
                    key = ("fact", int(fact["id"]))
                    scope.add(key)
                    allowed[key] = fact
            scope_by_thread[thread_id] = scope

        operations = []
        seen_targets = set()
        seen_identities = set()
        for raw in raw_operations:
            if not isinstance(raw, Mapping) or raw.get("state_type") != "situational_state":
                continue
            domain = self._normalize_state_domain(raw.get("state_domain"))
            if not domain:
                continue
            evidence = raw.get("evidence")
            if not isinstance(evidence, list) or any(
                not isinstance(item, Mapping)
                or (str(item.get("evidence_type") or ""), self._positive_int(item.get("evidence_id")))
                not in allowed for item in evidence
            ):
                continue
            normalized = self._normalize_operation(
                raw, existing_states=existing_states, allowed_evidence=allowed,
                state_domain=domain, reference_time=update_input["reference_time"],
            )
            if normalized is None:
                continue
            cited_threads = {
                item["evidence_id"] for item in normalized["evidence"]
                if item["evidence_type"] == "user_thread"
            }
            if not cited_threads:
                continue
            cited_scope = set().union(*(scope_by_thread[value] for value in cited_threads))
            if any((item["evidence_type"], item["evidence_id"]) not in cited_scope
                   for item in normalized["evidence"]):
                continue
            if not any(item["evidence_type"] != "user_thread"
                       and item["role"] != "context" for item in normalized["evidence"]):
                continue
            target_id = normalized["target_state_id"]
            identity = (domain, normalized["state_kind"], normalized["state_topic"])
            if target_id and target_id in seen_targets:
                continue
            if normalized["operation"] == "create":
                if identity in seen_identities or self._user_model_db.user_state_identity_exists(
                    state_domain=domain, state_type="situational_state",
                    state_kind=normalized["state_kind"], state_topic=normalized["state_topic"],
                ):
                    continue
                seen_identities.add(identity)
            if target_id:
                seen_targets.add(target_id)
            for item in normalized["evidence"]:
                if item["evidence_type"] == "user_thread":
                    case = cases[item["evidence_id"]]
                    item["details"] = {
                        "source": "user_thread_situational_state_v1",
                        "thread_snapshot": case["thread"],
                        "thread_event_ids": [event["id"] for event in case["change_events"]],
                        "reference_time": case["reference_time"],
                    }
            operations.append(normalized)
        return operations

    def process_user_state_update_from_user_threads_task(
        self, *, thread_changes: Optional[Sequence[Mapping[str, Any]]] = None,
        reference_timestamp: Optional[Any] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Assess personal-state implications of newly committed thread changes."""
        task_kind = self.USER_STATE_UPDATE_FROM_USER_THREADS_TASK_KIND
        if not thread_changes:
            report = {"status": "empty", "thread_count": 0, "operation_count": 0, "state_ids": []}
            self._log_update(report, task_kind=task_kind)
            return report
        inputs = self._build_user_state_update_inputs_using_threads(
            thread_changes, reference_timestamp=reference_timestamp, runtime_context=runtime_context,
        )
        update_input = inputs["update_input"]
        language = self._resolve_prompt_language(
            "\n".join(case["thread"]["current_summary"] for case in update_input["thread_cases"])
        )
        template = (USER_STATE_UPDATE_FROM_USER_THREAD_PROMPT_EN if language == "en"
                    else USER_STATE_UPDATE_FROM_USER_THREAD_PROMPT_ZH)
        prompt = template
        for key, value in update_input.items():
            if key == "thread_cases":
                text = self._format_thread_cases_view_for_user_state_prompt(value)
            elif key == "state_contexts":
                text = "[\n" + ",\n".join(
                    json.dumps(context, ensure_ascii=False, separators=(",", ":"))
                    for context in value
                ) + "\n]" if value else "[]"
            else:
                text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            prompt = prompt.replace("{" + key + "}", text)
        self._logger.info("\n%s", json.dumps({
            "scope": task_kind, "event": "input", "payload": update_input,
        }, ensure_ascii=False, indent=2, default=str))
        raw_output = self._llm_client.call_llm(prompt, stage_name='user_state_update_from_user_threads')
        parsed = self._llm_client.parse_json_object_from_llm_text(raw_output or "", stage_name='user_state_update_from_user_threads')
        self._logger.info("\n%s", json.dumps({
            "scope": task_kind, "event": "llm_result", "payload": parsed,
        }, ensure_ascii=False, indent=2, default=str))
        report = {
            "status": "ok", "thread_count": len(update_input["thread_cases"]),
            "thread_event_count": sum(len(case["change_events"])
                                      for case in update_input["thread_cases"]),
            "operation_count": 0, "state_ids": [],
        }
        if not isinstance(parsed, Mapping) or not isinstance(parsed.get("operations"), list):
            report.update(status="error", error="invalid_llm_thread_state_response")
        else:
            operations = self._normalize_user_thread_state_operations(
                parsed["operations"], update_input=update_input,
                existing_states=inputs["existing_states"],
            )
            report["rejected_operation_count"] = len(parsed["operations"]) - len(operations)
            if parsed["operations"] and not operations:
                report.update(status="error", error="no_valid_thread_state_operations")
            else:
                with self._user_model_db.transaction():
                    report.update(self.save_user_state_update_operations(
                        operations=operations, reference_time=update_input["reference_time"],
                        decision_source="user_thread_situational_state_v1",
                    ))
        self._log_update(report, task_kind=task_kind)
        return report

    def process_user_state_update_from_user_observed_patterns_task(
        self,
        *,
        new_pattern_ids: Optional[Sequence[int]] = None,
        pattern_relation_event_ids: Optional[Sequence[int]] = None,
        reference_timestamp: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Project changed observed behavior into recurring states by domain."""
        reference_time = reference_timestamp
        update_inputs = self._build_user_state_update_inputs_using_observed_patterns(
            world_owner_entity_id=self._world_owner_entity_id,
            new_pattern_ids=new_pattern_ids or [],
            pattern_relation_event_ids=pattern_relation_event_ids or [],
            reference_time=reference_time,
        )
        report: Dict[str, Any] = {
            "status": "empty", "changed_pattern_count": update_inputs["changed_pattern_count"],
            "affected_pattern_count": update_inputs["affected_pattern_count"],
            "domain_count": len(update_inputs["domains"]),
            "operation_count": 0, "created": 0, "reinforced": 0,
            "revised": 0, "weakened": 0, "expired": 0,
            "state_ids": [], "domain_reports": {},
        }
        domains = update_inputs["domains"]
        failed = 0
        for domain, domain_input in domains.items():
            update_input = domain_input["update_input"]
            domain_patterns = domain_input["domain_patterns"]
            domain_changed = domain_input["domain_changed"]
            domain_affected = domain_input["domain_affected"]
            domain_events = domain_input["domain_events"]
            domain_trigger_ids = domain_input["domain_trigger_ids"]
            signal_by_id = domain_input["signal_by_id"]
            existing_states = domain_input["existing_states"]
            explicitly_supported_state_ids = domain_input["explicitly_supported_state_ids"]
            pattern_cases = update_input["pattern_cases"]
            source = "\n".join(
                item["pattern"]["pattern_text"] for item in pattern_cases
            )
            template = (
                USER_STATE_UPDATE_FROM_USER_OBSERVED_PATTERN_PROMPT_EN
                if self._resolve_prompt_language(source) == "en"
                else USER_STATE_UPDATE_FROM_USER_OBSERVED_PATTERN_PROMPT_ZH
            )
            prompt = template
            for key, value in update_input.items():
                prompt = prompt.replace(
                    "{" + key + "}",
                    json.dumps(value, ensure_ascii=False, indent=2)
                    if isinstance(value, (dict, list)) else str(value or ""),
                )
            raw_output = self._llm_client.call_llm(prompt, stage_name='user_state_update_from_user_observed_patterns')
            parsed = self._llm_client.parse_json_object_from_llm_text(raw_output or "", stage_name='user_state_update_from_user_observed_patterns')
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
                int(state["id"]): {
                    int(item["evidence_id"])
                    for item in state["evidence_items"]
                    if item["evidence_type"] == "user_observed_pattern"
                }
                for state in existing_states
            }
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
                    applied = self.save_user_state_update_operations(
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
        self._log_update(report, task_kind=self.USER_STATE_UPDATE_FROM_USER_OBSERVED_PATTERNS_TASK_KIND)
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

    def _build_user_state_update_inputs_using_observed_patterns(
        self, *, world_owner_entity_id: int, new_pattern_ids: Sequence[int],
        pattern_relation_event_ids: Sequence[int], reference_time: Optional[Any],
    ) -> Dict[str, Any]:
        new_ids = {int(value) for value in new_pattern_ids if int(value) > 0}
        relation_event_ids = {int(value) for value in pattern_relation_event_ids}
        if any(event_id <= 0 for event_id in relation_event_ids):
            raise ValueError("Pattern relation event IDs must be positive")
        events = self._user_memory_db.get_user_observed_pattern_relation_events_by_ids(
            relation_event_ids
        )
        if {int(item["id"]) for item in events} != relation_event_ids:
            raise ValueError("Pattern relation event is missing")
        changed_ids = new_ids | {
            int(item["target_pattern_id"]) for item in events
            if item["semantic_relation"] in {"supports", "duplicate"}
        }
        affected_ids = {
            int(item["target_pattern_id"]) for item in events
            if item["semantic_relation"] in {"refines", "contradicts"}
        }
        trigger_ids = {
            int(value) for item in events for value in item["support_signal_ids"]
        }
        patterns = self._user_memory_db.get_user_observed_patterns_by_ids(
            changed_ids | affected_ids
        )
        patterns = [
            item for item in patterns
            if int(item.get("subject_entity_id") or 0) == world_owner_entity_id
            and self._normalize_state_domain(item.get("primary_domain"))
        ]
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
        domains: Dict[str, Dict[str, Any]] = {}
        for domain in sorted({str(item["primary_domain"]) for item in patterns}):
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
            events_by_pattern: Dict[int, List[Dict[str, Any]]] = {}
            for event in domain_events:
                trigger_id = int(event.get("trigger_pattern_id") or 0)
                case_id = (
                    trigger_id if trigger_id in domain_changed
                    else int(event["target_pattern_id"])
                )
                events_by_pattern.setdefault(case_id, []).append(event)
            pattern_cases = []
            for pattern_id in sorted(domain_changed | events_by_pattern.keys()):
                pattern_cases.append({
                    "pattern": self._user_observed_pattern_prompt_view(
                        domain_patterns[pattern_id], signal_by_id,
                    ),
                    "update_kind": (
                        "new" if pattern_id in new_ids
                        else "supported" if pattern_id in domain_changed
                        else "relation_only"
                    ),
                    "related_patterns": [
                        {
                            "relation_event": {
                                key: event.get(key) for key in (
                                    "id", "target_pattern_id", "trigger_pattern_id",
                                    "semantic_relation", "support_signal_ids",
                                    "candidate_snapshot", "semantic_reason",
                                )
                            },
                            "existing_pattern": self._user_observed_pattern_prompt_view(
                                domain_patterns[int(event["target_pattern_id"])],
                                signal_by_id,
                            ),
                            "trigger_signals": [
                                self._user_action_signal_prompt_view(signal_by_id[int(value)])
                                for value in event["support_signal_ids"]
                                if int(value) in signal_by_id
                            ],
                            "linked_state_ids": [],
                        }
                        for event in events_by_pattern.get(pattern_id, [])
                    ],
                    "related_states": [],
                })
            existing_states = self._user_model_db.get_user_states(
                state_domains=[domain],
                state_types=["recurring_state"], statuses=["active", "weakened"],
                limit=self._update_limit,
                include_evidence=True,
            )
            claim_ids = {
                int(item["evidence_id"])
                for state in existing_states
                for item in state["evidence_items"]
                if item["evidence_type"] == "user_claim"
            }
            claim_by_id = {
                int(item["id"]): item
                for item in self._user_memory_db.get_user_claims_by_ids(sorted(claim_ids))
            }
            explicitly_supported_state_ids = {
                int(state["id"])
                for state in existing_states
                for item in state["evidence_items"]
                if item["evidence_type"] == "user_claim"
                and item["role"] in {"support", "change_trigger"}
                and (claim := claim_by_id.get(int(item["evidence_id"])))
                and claim.get("claim_origin") == "explicit"
                and claim.get("status") == "active"
            }
            linked_state_ids_by_pattern: Dict[int, set[int]] = {}
            for state in existing_states:
                state_id = int(state["id"])
                for item in state["evidence_items"]:
                    if item["evidence_type"] == "user_observed_pattern":
                        linked_state_ids_by_pattern.setdefault(
                            int(item["evidence_id"]), set(),
                        ).add(state_id)
            detailed_state_ids = set()
            for case in pattern_cases:
                related_pattern_ids = {
                    int(related["existing_pattern"]["id"])
                    for related in case["related_patterns"]
                } | {int(case["pattern"]["id"])}
                related_state_ids = {
                    state_id for related_id in related_pattern_ids
                    for state_id in linked_state_ids_by_pattern.get(related_id, set())
                }
                detailed_state_ids.update(related_state_ids)
            state_contexts = {
                int(state["id"]): self._build_state_context(
                    state, detailed=int(state["id"]) in detailed_state_ids,
                    world_owner_entity_id=world_owner_entity_id,
                )
                for state in existing_states
            }
            for case in pattern_cases:
                related_pattern_ids = {
                    int(related["existing_pattern"]["id"])
                    for related in case["related_patterns"]
                } | {int(case["pattern"]["id"])}
                related_state_ids = {
                    state_id for pattern_id in related_pattern_ids
                    for state_id in linked_state_ids_by_pattern.get(pattern_id, set())
                }
                case["related_states"] = [
                    state_contexts[value] for value in sorted(related_state_ids)
                ]
                for related in case["related_patterns"]:
                    related["linked_state_ids"] = sorted(
                        linked_state_ids_by_pattern.get(
                            int(related["existing_pattern"]["id"]), set(),
                        ),
                    )
            domains[domain] = {
                "update_input": {
                    "reference_time": self._timestamp_text(reference_time),
                    "state_domain": domain,
                    "pattern_cases": pattern_cases,
                    "existing_states": [
                        state_contexts[int(item["id"])] for item in existing_states
                        if int(item["id"]) not in detailed_state_ids
                    ],
                },
                "domain_patterns": domain_patterns,
                "domain_changed": domain_changed,
                "domain_affected": domain_affected,
                "domain_events": domain_events,
                "domain_trigger_ids": domain_trigger_ids,
                "signal_by_id": signal_by_id,
                "existing_states": existing_states,
                "explicitly_supported_state_ids": explicitly_supported_state_ids,
            }
        return {
            "changed_pattern_count": len(changed_ids),
            "affected_pattern_count": len(affected_ids),
            "domains": domains,
        }

    def _build_user_state_update_inputs_using_claim(
        self, *, world_owner_entity_id: int, new_claim_ids: Sequence[int],
        claim_relations: Sequence[Mapping[str, Any]],
        reference_time: Optional[Any],
    ) -> Dict[str, Dict[str, Any]]:
        claims_by_id = {
            int(claim["id"]): claim
            for claim in self._user_memory_db.get_user_claims_by_ids(new_claim_ids)
            if int(claim.get("subject_entity_id") or 0) == world_owner_entity_id
            and self._normalize_state_domain(claim.get("primary_domain"))
            and claim.get("status") in {"active", "weakened"}
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
        effective_reference_time = self._timestamp_text(reference_time)
        for domain, changed_ids in by_domain.items():
            relations = [
                event for claim_id in changed_ids
                for event in relations_by_trigger.get(claim_id, [])
            ]
            related_old_claim_ids = {
                int(event["target_claim_id"]) for event in relations
            }
            claims_by_id.update({
                int(claim["id"]): claim
                for claim in self._user_memory_db.get_user_claims_by_ids(
                    sorted(related_old_claim_ids - claims_by_id.keys()),
                )
                if int(claim.get("subject_entity_id") or 0) == world_owner_entity_id
            })
            new_claim_cases = [
                {
                    "new_claim": self._claim_prompt_view(claims_by_id[claim_id]),
                    "related_claims": [
                        {
                            "relation_event": {key: event.get(key) for key in (
                                "id", "target_claim_id", "trigger_claim_id",
                                "semantic_relation", "semantic_confidence",
                                "effective_at", "semantic_reason",
                            )},
                            "existing_claim": self._claim_prompt_view(
                                claims_by_id[int(event["target_claim_id"])],
                            ),
                            "linked_state_ids": [],
                        }
                        for event in relations_by_trigger.get(claim_id, [])
                        if int(event["target_claim_id"]) in claims_by_id
                    ],
                }
                for claim_id in sorted(changed_ids)
            ]
            states = self._user_model_db.get_user_states(
                state_domains=[domain],
                state_types=sorted(self.STATE_TYPES),
                statuses=["active", "weakened"], limit=self._update_limit,
                include_evidence=True,
            )
            detailed_state_ids = {
                int(state["id"]) for state in states
                if any(
                    item["evidence_type"] == "user_claim"
                    and int(item["evidence_id"]) in related_old_claim_ids
                    for item in state["evidence_items"]
                )
            }
            state_contexts = []
            linked_state_ids_by_claim: Dict[int, List[int]] = {}
            for state in states:
                is_detailed = int(state["id"]) in detailed_state_ids
                state_contexts.append(self._build_state_context(
                    state, detailed=is_detailed,
                    world_owner_entity_id=world_owner_entity_id,
                ))
                for item in state["evidence_items"]:
                    if item["evidence_type"] == "user_claim":
                        linked_state_ids_by_claim.setdefault(
                            int(item["evidence_id"]), [],
                        ).append(int(state["id"]))
            for case in new_claim_cases:
                for related in case["related_claims"]:
                    related["linked_state_ids"] = linked_state_ids_by_claim.get(
                        int(related["existing_claim"]["id"]), [],
                    )
            result[domain] = {
                "state_domain": domain,
                "reference_time": effective_reference_time,
                "new_claim_cases": new_claim_cases,
                "state_contexts": state_contexts,
            }
        return result

    @staticmethod
    def _format_claim_cases_view_for_user_state_prompt(
        new_claim_cases: Sequence[Mapping[str, Any]],
    ) -> str:
        """Flatten claim relations for the prompt without changing internal events."""
        if not new_claim_cases:
            return "[]"
        lines = []
        for case in new_claim_cases:
            related_claims = []
            for related in case["related_claims"]:
                event = related["relation_event"]
                related_claims.append({
                    **{key: value for key, value in related["existing_claim"].items()
                       if key != "predicate"},
                    "semantic_relation": event.get("semantic_relation"),
                    "semantic_confidence": event.get("semantic_confidence"),
                    "semantic_reason": event.get("semantic_reason"),
                    "relation_effective_at": event.get("effective_at"),
                    "linked_state_ids": related["linked_state_ids"],
                })
            lines.append(json.dumps({
                "new_claim": {
                    key: value for key, value in case["new_claim"].items()
                    if key not in {"predicate", "primary_domain"}
                },
                "related_claims": related_claims,
            }, ensure_ascii=False, separators=(",", ":")))
        return "[\n" + ",\n".join(lines) + "\n]"

    def _build_state_context(
        self, state: Mapping[str, Any], *, detailed: bool,
        world_owner_entity_id: int,
    ) -> Dict[str, Any]:
        """Show a compact state, expanding its supporting objects when relevant."""
        evidence_items = state.get("evidence_items") or []
        context = {
            "state": self._state_prompt_view(state),
            "evidence_detail_available": detailed,
            "claim_support_count": sum(
                item["evidence_type"] == "user_claim"
                and item["role"] in {"support", "change_trigger"}
                for item in evidence_items
            ),
            "pattern_support_count": sum(
                item["evidence_type"] == "user_observed_pattern"
                and item["role"] in {"support", "change_trigger"}
                for item in evidence_items
            ),
            "thread_support_count": sum(
                item["evidence_type"] == "user_thread"
                and item["role"] in {"support", "change_trigger"}
                for item in evidence_items
            ),
        }
        if not detailed:
            return context

        claim_ids = sorted({
            int(item["evidence_id"]) for item in evidence_items
            if item["evidence_type"] == "user_claim"
        })
        claims = {
            int(claim["id"]): claim
            for claim in self._user_memory_db.get_user_claims_by_ids(claim_ids)
            if int(claim.get("subject_entity_id") or 0) == world_owner_entity_id
        }
        context["claim_evidence"] = [
            {
                "claim": {
                    key: value for key, value in self._claim_prompt_view(
                        claims[int(item["evidence_id"])]
                    ).items() if key != "predicate"
                },
                "role": str(item.get("role") or ""),
            }
            for item in evidence_items
            if item["evidence_type"] == "user_claim"
            and int(item["evidence_id"]) in claims
        ]
        pattern_ids = sorted({
            int(item["evidence_id"]) for item in evidence_items
            if item["evidence_type"] == "user_observed_pattern"
        })
        patterns = {
            int(pattern["id"]): pattern
            for pattern in self._user_memory_db.get_user_observed_patterns_by_ids(pattern_ids)
        }
        context["pattern_evidence"] = []
        for item in evidence_items:
            if item["evidence_type"] != "user_observed_pattern":
                continue
            evidence = {
                "id": int(item["evidence_id"]),
                "role": str(item.get("role") or ""),
            }
            pattern = patterns.get(evidence["id"])
            if pattern is not None:
                evidence.update({
                    "pattern_text": str(pattern.get("pattern_text") or ""),
                    "condition_text": str(pattern.get("condition_text") or ""),
                    "confidence": float(pattern.get("confidence") or 0.0),
                })
            context["pattern_evidence"].append(evidence)
        context["thread_evidence"] = []
        event_ids = [int(item["evidence_id"]) for item in evidence_items
                     if item["evidence_type"] == "user_thread_event"]
        events_by_id = {
            int(event["id"]): event
            for event in self._user_memory_db.get_user_thread_events_by_ids(event_ids)
        }
        for item in evidence_items:
            if item["evidence_type"] not in {"user_thread", "user_thread_event"}:
                continue
            evidence = {
                "evidence_type": item["evidence_type"],
                "evidence_id": int(item["evidence_id"]),
                "role": item["role"],
            }
            if item.get("observed_at"):
                evidence["observed_at"] = item["observed_at"]
            if item["evidence_type"] == "user_thread":
                details = item.get("details") or {}
                if isinstance(details, str):
                    details = json.loads(details)
                if details.get("thread_snapshot") is not None:
                    evidence["thread_snapshot"] = details["thread_snapshot"]
            else:
                event = events_by_id.get(evidence["evidence_id"])
                if event is not None:
                    evidence["event"] = {key: event.get(key) for key in (
                        "thread_id", "event_type", "summary", "occurred_at", "known_at",
                        "evidence_activity_ids", "evidence_fact_ids",
                    )}
            context["thread_evidence"].append(evidence)
        return context

    def _log_state_validation(
        self, reason: str, *, context: Mapping[str, Any], raw: Any = None,
        outcome: str = "rejected", **details: Any,
    ) -> None:
        """Explain a validation branch without changing its result or retry policy."""
        record = dict(context)
        if isinstance(raw, Mapping):
            record.update({key: raw.get(key) for key in (
                "operation", "target_state_id", "state_type", "state_kind", "state_topic",
            ) if key in raw})
        record.update(outcome=outcome, reason=reason, **details)
        log = self._logger.info if outcome == "skipped" else self._logger.warning
        log("User state validation %s", json.dumps(
            record, ensure_ascii=False, separators=(",", ":"), default=str,
        ))

    def _normalize_user_claim_state_operations(
        self, parsed: Mapping[str, Any], *, update_input: Mapping[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
        context = {
            "stage_name": "user_state_update_from_user_claims",
            "state_domain": update_input["state_domain"],
            "reference_time": update_input["reference_time"],
            "new_claim_ids": [int(case["new_claim"]["id"])
                              for case in update_input["new_claim_cases"]],
        }
        raw_operations = parsed.get("operations")
        if not isinstance(raw_operations, list):
            self._log_state_validation("operations_not_list", context=context,
                                       actual_type=type(raw_operations).__name__)
            return None
        if len(raw_operations) > 24:
            self._log_state_validation("operation_limit_exceeded", context=context,
                                       actual_count=len(raw_operations), limit=24)
            return None
        # Input construction guarantees these are effective claims. Historical
        # claims remain context, never new evidence for this domain's update.
        allowed_evidence = {
            ("user_claim", int(case["new_claim"]["id"])): case["new_claim"]
            for case in update_input["new_claim_cases"]
        }
        existing_states = {}
        state_contexts_by_id = {}
        for state_context in update_input["state_contexts"]:
            state = state_context["state"]
            state_id = int(state["id"])
            existing_states[state_id] = state
            state_contexts_by_id[state_id] = state_context
        operations: List[Dict[str, Any]] = []
        seen_states: set[tuple[str, ...]] = set()
        for index, raw in enumerate(raw_operations):
            operation_context = {**context, "operation_index": index}
            operation = self._normalize_operation(
                raw, existing_states=existing_states,
                allowed_evidence=allowed_evidence,
                state_domain=str(update_input["state_domain"]),
                reference_time=str(update_input["reference_time"]),
                diagnostic_context=operation_context,
            )
            if operation is None:
                return None
            if (operation["operation"] in {"weaken", "expire"}
                    and operation["state_type"] == "recurring_state"
                    and state_contexts_by_id.get(
                        operation["target_state_id"], {}
                    ).get("pattern_support_count", 0) > 0):
                self._log_state_validation(
                    "recurring_state_protected_by_pattern", context=operation_context,
                    raw=raw, outcome="skipped",
                    pattern_support_count=state_contexts_by_id[
                        operation["target_state_id"]
                    ]["pattern_support_count"],
                )
                continue
            raw_evidence = raw.get("evidence") if isinstance(raw, Mapping) else None
            if not isinstance(raw_evidence, list) or len(raw_evidence) != len(operation["evidence"]):
                self._log_state_validation(
                    "evidence_filtered", context=operation_context, raw=raw,
                    raw_evidence_count=len(raw_evidence) if isinstance(raw_evidence, list) else None,
                    valid_evidence_count=len(operation["evidence"]),
                )
                return None
            if operation["operation"] in {"create", "reinforce", "revise"} and not any(
                item["role"] in {"support", "change_trigger"}
                for item in operation["evidence"]
            ):
                self._log_state_validation(
                    "missing_effective_support", context=operation_context, raw=raw,
                    required_roles=["support", "change_trigger"],
                )
                return None
            target_key = (
                "id", str(operation["target_state_id"])
            ) if operation["target_state_id"] else (
                operation["state_type"], operation["state_kind"],
                operation["state_topic"],
            )
            if target_key in seen_states:
                self._log_state_validation("duplicate_target_operation",
                                           context=operation_context, raw=raw, target_key=target_key)
                return None
            seen_states.add(target_key)
            if operation["operation"] == "create" and any(
                state["state_type"] == operation["state_type"]
                and state["state_kind"] == operation["state_kind"]
                and state["state_topic"] == operation["state_topic"]
                for state in existing_states.values()
            ):
                self._log_state_validation(
                    "state_identity_already_in_input", context=operation_context, raw=raw,
                    conflicting_states=[{"id": state["id"], "status": state.get("status")}
                                        for state in existing_states.values()
                                        if state["state_type"] == operation["state_type"]
                                        and state["state_kind"] == operation["state_kind"]
                                        and state["state_topic"] == operation["state_topic"]],
                )
                return None
            if operation["operation"] == "create" and self._user_model_db.user_state_identity_exists(
                state_domain=str(update_input["state_domain"]),
                state_type=operation["state_type"],
                state_kind=operation["state_kind"],
                state_topic=operation["state_topic"],
            ):
                conflict = self._user_model_db.get_user_state_by_identity(
                    state_domain=str(update_input["state_domain"]),
                    state_type=operation["state_type"], state_kind=operation["state_kind"],
                    state_topic=operation["state_topic"],
                )
                self._log_state_validation(
                    "state_identity_already_exists", context=operation_context, raw=raw,
                    conflicting_state=conflict,
                )
                return None
            operations.append(operation)
        return operations

    def process_user_state_update_from_user_claims_task(
        self,
        *,
        new_claim_ids: Optional[Sequence[int]] = None,
        claim_relations: Optional[Sequence[Mapping[str, Any]]] = None,
        reference_timestamp: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Update enduring, recurring and situational states from new claims."""
        new_claim_ids = list(new_claim_ids or [])
        reference_time = reference_timestamp
        update_inputs = self._build_user_state_update_inputs_using_claim(
            world_owner_entity_id=self._world_owner_entity_id,
            new_claim_ids=new_claim_ids,
            claim_relations=list(claim_relations or []),
            reference_time=reference_time,
        )
        report: Dict[str, Any] = {
            "status": "empty", "domain_count": len(update_inputs),
            "changed_claim_count": sum(len(item["new_claim_cases"]) for item in update_inputs.values()),
            "changed_intent_object_count": 0,
            "existing_state_count": sum(len(item["state_contexts"]) for item in update_inputs.values()),
            "operation_count": 0,
            "created": 0, "reinforced": 0, "revised": 0, "weakened": 0,
            "expired": 0, "state_ids": [],
            "domain_reports": {},
        }
        failed_count = 0
        for domain, update_input in update_inputs.items():
            context = {
                "stage_name": "user_state_update_from_user_claims", "state_domain": domain,
                "reference_time": update_input["reference_time"],
                "new_claim_ids": [int(case["new_claim"]["id"])
                                  for case in update_input["new_claim_cases"]],
            }
            source = "\n".join(
                case["new_claim"]["claim_text"]
                for case in update_input["new_claim_cases"]
            )
            template = (
                USER_STATE_UPDATE_FROM_USER_CLAIM_PROMPT_EN
                if self._resolve_prompt_language(source) == "en"
                else USER_STATE_UPDATE_FROM_USER_CLAIM_PROMPT_ZH
            )
            prompt = template
            for key in ("reference_time", "state_domain", "new_claim_cases", "state_contexts"):
                value = update_input[key]
                if key == "new_claim_cases":
                    text = self._format_claim_cases_view_for_user_state_prompt(value)
                elif key == "state_contexts":
                    text = "[\n" + ",\n".join(
                        json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                        for item in value
                    ) + "\n]" if value else "[]"
                else:
                    text = json.dumps(
                        value, ensure_ascii=False, indent=2,
                    ) if isinstance(value, (dict, list)) else str(value or "")
                prompt = prompt.replace("{" + key + "}", text)
            raw_output = self._llm_client.call_llm(prompt, stage_name='user_state_update_from_user_claims')
            parsed = self._llm_client.parse_json_object_from_llm_text(raw_output or "", stage_name='user_state_update_from_user_claims')
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug("User state LLM result %s", json.dumps(
                    {**context, "parsed_result": parsed}, ensure_ascii=False, default=str,
                ))
            if not isinstance(parsed, Mapping):
                self._log_state_validation("parsed_result_not_object", context=context,
                                           actual_type=type(parsed).__name__)
            operations = self._normalize_user_claim_state_operations(
                parsed, update_input=update_input,
            ) if isinstance(parsed, Mapping) else None
            if operations is None:
                self._log_state_validation("domain_operations_rejected", context=context,
                                           discarded_domain=True)
                failed_count += 1
                report["domain_reports"][domain] = {
                    "status": "error", "error": "invalid_llm_user_claim_state_decision",
                }
                continue
            self._logger.info("User state decision %s", json.dumps({
                **context, "outcome": "validated",
                "reason": ("model_no_operations" if not parsed["operations"] else
                           "no_applicable_operations" if not operations else "operations_validated"),
                "raw_operation_count": len(parsed["operations"]), "operation_count": len(operations),
            }, ensure_ascii=False, separators=(",", ":"), default=str))
            with self._user_model_db.transaction():
                applied = self.save_user_state_update_operations(
                    operations=operations,
                    reference_time=update_input["reference_time"],
                    merge_claim_evidence=True,
                    decision_source=self.CLAIM_RESOLVER_VERSION,
                )
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
                **applied,
            }
        report["state_ids"] = sorted(set(report["state_ids"]))
        report["status"] = (
            "error" if failed_count == len(update_inputs) and update_inputs
            else "partial" if failed_count else "ok" if update_inputs else "empty"
        )
        if failed_count:
            report["failed_domain_count"] = failed_count
        self._log_update(report, task_kind=self.USER_STATE_UPDATE_FROM_USER_CLAIMS_TASK_KIND)
        return report

    def save_user_state_update_operations(
        self,
        *,
        operations: Sequence[Mapping[str, Any]],
        reference_time: Optional[Any] = None,
        merge_claim_evidence: bool = False,
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
                    [target_id], evidence_type="user_claim", limit=None,
                )
                if merge_claim_evidence and target_id else []
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
            if merge_claim_evidence:
                claim_evidence = self._merge_user_claim_state_evidence(
                    previous_claim_evidence, evidence,
                )
                self._user_model_db.replace_user_state_claim_evidence(
                    state_id=state_id, evidence=claim_evidence,
                )
            else:
                self._user_model_db.upsert_user_state_evidence(evidence)
            self._user_model_db.record_user_state_event(
                state_id=state_id,
                operation=action,
                previous_payload={
                    **(previous or {}),
                    "claim_evidence": previous_claim_evidence,
                } if merge_claim_evidence else previous or {},
                new_payload={
                    **current,
                    "claim_evidence": claim_evidence,
                } if merge_claim_evidence else {**current, "evidence": evidence},
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

    @staticmethod
    def _merge_user_claim_state_evidence(
        previous: Sequence[Mapping[str, Any]], incoming: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Preserve unseen support while applying this decision's claim evidence."""
        incoming_claim_ids = {
            int(item["evidence_id"]) for item in incoming
            if item["evidence_type"] == "user_claim"
        }
        merged: Dict[tuple[int, str], Dict[str, Any]] = {}
        for item in previous:
            claim_id = int(item["evidence_id"])
            if claim_id in incoming_claim_ids:
                continue
            role = str(item["role"])
            details = item.get("details") or {}
            if isinstance(details, str):
                try:
                    details = json.loads(details)
                except json.JSONDecodeError:
                    details = {}
            merged[(claim_id, role)] = {
                "evidence_type": "user_claim", "evidence_id": claim_id,
                "role": role, "weight": float(item.get("weight") or 0.0),
                "observed_at": str(item.get("observed_at") or ""),
                "details": details if isinstance(details, dict) else {},
            }
        for item in incoming:
            if item["evidence_type"] != "user_claim":
                continue
            claim_id = int(item["evidence_id"])
            role = str(item["role"])
            merged[(claim_id, role)] = {**item, "role": role}
        return list(merged.values())

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
        diagnostic_context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        context = {"stage_name": "user_state_operation_validation",
                   "state_domain": state_domain, "reference_time": reference_time,
                   **dict(diagnostic_context or {})}
        if not isinstance(raw, Mapping):
            self._log_state_validation("operation_not_object", context=context,
                                       actual_type=type(raw).__name__)
            return None
        action = str(raw.get("operation") or "").strip().lower()
        state_type = str(raw.get("state_type") or "").strip().lower()
        if action not in self.OPERATIONS:
            self._log_state_validation("invalid_operation", context=context, raw=raw,
                                       allowed_operations=sorted(self.OPERATIONS))
            return None
        if state_type not in self.STATE_TYPES:
            self._log_state_validation("invalid_state_type", context=context, raw=raw,
                                       allowed_state_types=sorted(self.STATE_TYPES))
            return None
        target_id = self._positive_int(raw.get("target_state_id"))
        target = existing_states.get(target_id)
        if action in {"reinforce", "revise", "weaken", "expire"} and target is None:
            self._log_state_validation("target_state_not_in_input", context=context, raw=raw,
                                       allowed_state_ids=sorted(existing_states))
            return None
        if target is not None and str(target.get("state_type") or "") != state_type:
            self._log_state_validation("target_state_type_mismatch", context=context, raw=raw,
                                       expected_state_type=target.get("state_type"))
            return None
        if target is not None and str(target.get("state_domain") or "") != state_domain:
            self._log_state_validation("target_state_domain_mismatch", context=context, raw=raw,
                                       expected_state_domain=target.get("state_domain"))
            return None
        if action == "create" and target_id:
            self._log_state_validation("create_with_target_state_id", context=context, raw=raw)
            return None
        state_kind = self._normalize_state_kind(
            (target or {}).get("state_kind") if target is not None
            else raw.get("state_kind")
        )
        if target is None and state_kind not in self.STATE_KINDS_BY_TYPE[state_type]:
            self._log_state_validation("invalid_state_kind", context=context, raw=raw,
                                       normalized_state_kind=state_kind,
                                       allowed_state_kinds=sorted(self.STATE_KINDS_BY_TYPE[state_type]))
            return None
        state_text = self._compact(raw.get("state_text") or (target or {}).get("state_text"))
        evidence = self._normalize_evidence(
            raw.get("evidence"), allowed_evidence,
            diagnostic_context={**context, **{key: raw.get(key) for key in (
                "operation", "target_state_id", "state_type", "state_kind", "state_topic",
            ) if key in raw}},
        )
        if not state_kind or not state_text or not evidence:
            reason = ("missing_state_kind" if not state_kind else
                      "missing_state_text" if not state_text else "no_valid_evidence")
            self._log_state_validation(reason, context=context, raw=raw)
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

    def _normalize_evidence(
        self, raw_items: Any, allowed: Mapping[tuple[str, int], Mapping[str, Any]], *,
        diagnostic_context: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        context = {"stage_name": "user_state_operation_validation",
                   **dict(diagnostic_context or {})}
        if not isinstance(raw_items, list):
            self._log_state_validation("evidence_not_list", context=context,
                                       actual_type=type(raw_items).__name__)
            return []
        if len(raw_items) > 24:
            self._log_state_validation("evidence_limit_exceeded", context=context,
                                       raw_evidence_count=len(raw_items), limit=24)
        evidence: List[Dict[str, Any]] = []
        seen: set[tuple[str, int, str]] = set()
        for index, raw in enumerate(raw_items[:24]):
            evidence_context = {**context, "evidence_index": index}
            if not isinstance(raw, Mapping):
                self._log_state_validation("evidence_not_object", context=evidence_context,
                                           actual_type=type(raw).__name__)
                continue
            evidence_type = str(raw.get("evidence_type") or "").strip()
            evidence_id = self._positive_int(raw.get("evidence_id"))
            role = str(raw.get("role") or "support").strip()
            key = (evidence_type, evidence_id, role)
            reason = (
                "invalid_evidence_type" if evidence_type not in self.EVIDENCE_TYPES else
                "invalid_evidence_role" if role not in self.EVIDENCE_ROLES else
                "invalid_evidence_id" if not evidence_id else
                "evidence_not_in_input" if (evidence_type, evidence_id) not in allowed else
                "duplicate_evidence" if key in seen else ""
            )
            if reason:
                self._log_state_validation(
                    reason, context=evidence_context,
                    evidence_type=evidence_type, evidence_id=raw.get("evidence_id"), role=role,
                    allowed_evidence=[{"evidence_type": kind, "evidence_id": value}
                                      for kind, value in sorted(allowed)]
                    if reason == "evidence_not_in_input" else None,
                )
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
    def _user_action_signal_prompt_view(
        signal: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return {
            "signal_id": int(signal["signal_id"]),
            "action_text": str(signal.get("action_text") or ""),
            "action_kind": str(signal.get("action_kind") or ""),
            "event_time": str(signal.get("event_time_key") or ""),
            "location": str(signal.get("location") or ""),
            "source_type": str(signal["source_type"]),
            "source_id": int(signal["source_id"]),
            "time_scope": str(signal.get("time_scope") or "reported_event"),
            "event_ended_at": str(signal.get("event_ended_at") or ""),
            "source_unit_refs": signal.get("source_unit_refs") or [],
        }

    @classmethod
    def _user_observed_pattern_prompt_view(
        cls, pattern: Mapping[str, Any],
        signals_by_id: Mapping[int, Mapping[str, Any]],
    ) -> Dict[str, Any]:
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
                cls._user_action_signal_prompt_view(signals_by_id[int(value)])
                for value in representative if int(value) in signals_by_id
            ],
        }

    @staticmethod
    def _claim_prompt_view(claim: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "id": int(claim["id"]), "claim_text": str(claim.get("claim_text") or ""),
            "claim_type": str(claim.get("claim_type") or ""), "predicate": str(claim.get("predicate") or ""),
            "primary_domain": str(claim.get("primary_domain") or ""),
            "claim_origin": str(claim.get("claim_origin") or ""), "status": str(claim.get("status") or ""),
            "confidence": float(claim.get("confidence") or 0.0), "valid_from": str(claim.get("valid_from") or ""),
            "valid_to": str(claim.get("valid_to") or ""),
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
