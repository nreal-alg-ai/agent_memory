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

from prompts_en import USER_ACTIVITY_STATE_UPDATE_PROMPT_EN, USER_CLAIM_STATE_RESOLUTION_PROMPT_EN
from prompts_zh import USER_ACTIVITY_STATE_UPDATE_PROMPT_ZH, USER_CLAIM_STATE_RESOLUTION_PROMPT_ZH

if TYPE_CHECKING:
    from memory.memory_database import SessionDB
    from llm_client import LLMClient
    from memory.task_scheduler import TaskScheduler


class UserStateResolver:
    """Turn new activity and changed claims into incremental user states."""

    RESOLVER_VERSION = "user_state_incremental_v1"
    CLAIM_RESOLVER_VERSION = "user_claim_state_joint_v1"
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
    OPERATIONS = {"create", "reinforce", "revise", "weaken", "expire"}
    ACTIVE_INTENT_STATUSES = {
        "goal": {"active"},
        "plan": {"planned", "rescheduled"},
        "work_item": {"open", "blocked"},
    }
    EVIDENCE_TYPES = {"user_claim", "goal", "plan", "work_item", "user_activity", "fact"}
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
        self._llm_client = llm_client
        self._world_owner_entity_name = str(world_owner_entity_name or "用户本人").strip()
        self._logger = logger or logging.getLogger(__name__)
        config = dict(memory_config or {})
        self._update_limit = max(
            1,
            int(config.get("user_state_update_limit", 80) or 80),
        )
        self._prompt_language_mode = str(
            config.get("memory_prompt_language_mode")
            or config.get("prompt_language_mode")
            or "source"
        ).strip().lower()
        task_scheduler.register_handler(self.TASK_KIND, self.process_update_task)

    def set_logger(self, logger: Optional[logging.Logger]) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def process_update_task(
        self,
        *,
        changed_claim_ids: Optional[Sequence[int]] = None,
        changed_user_activity_ids: Optional[Sequence[int]] = None,
        reference_timestamp: Optional[Any] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build, prompt for, validate, and persist incremental state updates."""
        reports: List[Dict[str, Any]] = []
        activity_ids = list(changed_user_activity_ids or [])
        if changed_claim_ids:
            reports.append(self._process_user_claim_state_updates(
                changed_claim_ids=changed_claim_ids,
                reference_time=reference_timestamp,
            ))
        if activity_ids:
            reports.append(
                self._process_user_activity_situational_state_updates(
                    activity_ids,
                    reference_time=reference_timestamp,
                    runtime_context=runtime_context or {},
                )
            )
        report: Optional[Dict[str, Any]] = None
        for item in reports:
            report = self._combine_update_reports(report, item)
        result = report or {
            "status": "empty", "operation_count": 0, "state_ids": [],
        }
        self._log_update(result)
        return result

    def _process_user_activity_situational_state_updates(
        self,
        activity_ids: Sequence[int],
        *,
        reference_time: Optional[Any],
        runtime_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update situational states from a new activity window and linked evidence."""
        owner_id = int(self._db.add_entity_names([self._world_owner_entity_name])[
            self._world_owner_entity_name
        ])
        activities = self._db.get_user_activity_items_by_ids(activity_ids)
        if not activities:
            return {"status": "empty", "activity_count": 0, "operation_count": 0, "state_ids": []}
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
        facts = self._db.get_memory_facts_by_ids(fact_ids) if fact_ids else []
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
            items = self._db.get_intent_objects(
                object_type=object_type,
                world_owner_entity_id=owner_id,
                statuses=sorted(active_statuses),
                limit=self._update_limit,
            )
            owned_items = [
                item for item in items
                if int(item.get(intent_owner_columns[object_type]) or 0) == owner_id
            ]
            evidence_by_object = self._db.get_intent_evidence_fact_ids(
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

        existing_states = self._db.get_user_states(
            user_entity_id=owner_id,
            state_classes=["situational_state"],
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
        update_input = {
            "reference_time": self._timestamp_text(effective_reference_time),
            "allowed_domains": sorted(self.STATE_DOMAINS),
            "runtime_context": dict(runtime_context or {}),
            "activities": activity_views,
            "facts": fact_views,
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
        seen: set[tuple[str, int, str, str]] = set()
        for raw in parsed["operations"][:24]:
            if not isinstance(raw, Mapping):
                continue
            if str(raw.get("operation") or "").strip().lower() not in {
                "create", "reinforce", "revise", "weaken", "expire",
            }:
                continue
            domain = self._normalize_state_domain(raw.get("state_domain"))
            if not domain or str(raw.get("state_class") or "") != "situational_state":
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
            if not any(
                item.get("evidence_type") == "user_activity"
                for item in normalized["evidence"]
            ):
                continue
            dedupe_key = (
                str(normalized["operation"]),
                int(normalized.get("target_state_id") or 0),
                str(normalized["state_domain"]),
                str(normalized["state_key"]),
            )
            if dedupe_key not in seen:
                seen.add(dedupe_key)
                operations.append(normalized)

        if not operations:
            report["status"] = "ok"
            return report
        with self._db.transaction():
            applied = self.apply_operations(
                user_entity_id=owner_id,
                operations=operations,
                reference_time=update_input["reference_time"],
                decision_source="user_activity_situational_state_v1",
            )
        report.update(applied)
        report["status"] = "ok"
        return report

    @staticmethod
    def _combine_update_reports(
        primary: Optional[Dict[str, Any]],
        secondary: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if primary is None:
            return dict(secondary) if secondary is not None else None
        if secondary is None:
            return primary
        combined = dict(primary)
        count_keys = {
            "domain_count", "changed_claim_count", "changed_intent_object_count",
            "existing_state_count", "operation_count", "created", "reinforced",
            "revised", "weakened", "expired", "activity_count", "linked_fact_count",
            "related_intent_count", "claim_status_update_count",
        }
        for key in count_keys:
            if key in primary or key in secondary:
                combined[key] = int(primary.get(key) or 0) + int(secondary.get(key) or 0)
        combined["state_ids"] = sorted(set(
            list(primary.get("state_ids") or []) + list(secondary.get("state_ids") or [])
        ))
        domain_reports = dict(primary.get("domain_reports") or {})
        for key, value in (secondary.get("domain_reports") or {}).items():
            domain_reports[f"activity:{key}"] = value
        if domain_reports:
            combined["domain_reports"] = domain_reports
        statuses = {str(primary.get("status") or ""), str(secondary.get("status") or "")}
        if "error" in statuses:
            combined["status"] = "partial" if "ok" in statuses else "error"
        elif "partial" in statuses:
            combined["status"] = "partial"
        elif "ok" in statuses:
            combined["status"] = "ok"
        else:
            combined["status"] = "empty"
        if secondary.get("error"):
            combined["activity_error"] = secondary["error"]
        return combined

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

    def _build_user_claim_update_inputs(
        self, *, user_entity_id: int, changed_claim_ids: Sequence[int],
        reference_time: Optional[Any],
    ) -> Dict[str, Dict[str, Any]]:
        changed = [
            claim for claim in self._db.get_user_claims_by_ids(changed_claim_ids)
            if int(claim.get("subject_entity_id") or 0) == user_entity_id
            and self._normalize_state_domain(claim.get("primary_domain"))
        ]
        by_domain: Dict[str, List[Dict[str, Any]]] = {}
        for claim in changed:
            by_domain.setdefault(str(claim["primary_domain"]), []).append(claim)
        result: Dict[str, Dict[str, Any]] = {}
        for domain, domain_claims in by_domain.items():
            changed_ids = {int(claim["id"]) for claim in domain_claims}
            relation_limit = max(400, self._update_limit * 8)
            relations = self._db.get_user_claim_relations_for_claim_ids(
                sorted(changed_ids), limit=relation_limit + 1,
            )
            related_ids = {
                int(event[key])
                for event in relations
                for key in ("target_claim_id", "trigger_claim_id")
            } - changed_ids
            linked_state_ids = self._db.get_user_state_ids_for_claim_ids(
                sorted(changed_ids | related_ids), limit=self._update_limit + 1,
            )
            linked_states = [
                state for state in self._db.get_user_states_by_ids(linked_state_ids)
                if int(state.get("user_entity_id") or 0) == user_entity_id
                and state.get("state_domain") == domain
                and state.get("status") in {"active", "weakened"}
            ]
            domain_states = self._db.get_user_states(
                user_entity_id=user_entity_id, state_domains=[domain],
                statuses=["active", "weakened"], limit=self._update_limit + 1,
            )
            states_by_id = {
                int(state["id"]): state
                for state in [*linked_states, *domain_states]
            }
            states = list(states_by_id.values())[:self._update_limit]
            state_evidence_limit = max(480, self._update_limit * 12)
            state_evidence = self._db.get_user_state_claim_evidence(
                [int(state["id"]) for state in states],
                limit=state_evidence_limit + 1,
            )
            supporting_ids = {
                int(item["evidence_id"]) for item in state_evidence
            } - changed_ids - related_ids
            all_ids = changed_ids | related_ids | supporting_ids
            claims_by_id = {
                int(claim["id"]): claim
                for claim in self._db.get_user_claims_by_ids(sorted(all_ids))
                if int(claim.get("subject_entity_id") or 0) == user_entity_id
            }
            fact_evidence = self._db.get_user_claim_evidence_fact_ids(
                sorted(claims_by_id), limit=max(480, self._update_limit * 12),
            )
            def claim_view(claim_id: int) -> Dict[str, Any]:
                return self._claim_prompt_view(
                    claims_by_id[claim_id], fact_evidence.get(claim_id, []),
                )
            result[domain] = {
                "user_entity_id": user_entity_id,
                "input_truncated": (
                    len(relations) > relation_limit
                    or len(linked_state_ids) > self._update_limit
                    or len(domain_states) > self._update_limit
                    or len(state_evidence) > state_evidence_limit
                ),
                "state_domain": domain,
                "reference_time": self._timestamp_text(reference_time),
                "changed_claims": [
                    claim_view(claim_id) for claim_id in sorted(changed_ids)
                    if claim_id in claims_by_id
                ],
                "related_claims": [
                    claim_view(claim_id) for claim_id in sorted(related_ids)
                    if claim_id in claims_by_id
                ],
                "supporting_claims": [
                    claim_view(claim_id) for claim_id in sorted(supporting_ids)
                    if claim_id in claims_by_id
                ],
                "relation_events": [
                    {
                        "id": int(event["id"]),
                        "target_claim_id": int(event["target_claim_id"]),
                        "trigger_claim_id": int(event["trigger_claim_id"]),
                        "semantic_relation": str(event.get("semantic_relation") or ""),
                        "semantic_confidence": event.get("semantic_confidence"),
                        "effective_at": str(event.get("effective_at") or ""),
                        "semantic_reason": str(event.get("semantic_reason") or ""),
                    }
                    for event in relations
                    if int(event["target_claim_id"]) in claims_by_id
                    and int(event["trigger_claim_id"]) in claims_by_id
                ],
                "existing_states": [self._state_prompt_view(state) for state in states],
                "state_claim_evidence": [
                    {
                        "state_id": int(item["state_id"]),
                        "claim_id": int(item["evidence_id"]),
                        "role": str(item.get("role") or ""),
                        "weight": float(item.get("weight") or 0.0),
                    }
                    for item in state_evidence
                    if int(item["evidence_id"]) in claims_by_id
                ],
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
        changed_ids = {int(item["id"]) for item in update_input["changed_claims"]}
        related_by_id = {
            int(item["id"]): item for item in update_input["related_claims"]
        }
        events_by_id = {
            int(item["id"]): item for item in update_input["relation_events"]
        }
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

        claims = {
            int(item["id"]): item
            for field in ("changed_claims", "related_claims", "supporting_claims")
            for item in update_input[field]
        }
        projected_status = {claim_id: str(item.get("status") or "") for claim_id, item in claims.items()}
        projected_status.update({
            item["claim_id"]: item["new_status"] for item in claim_updates
        })
        allowed_evidence = {("user_claim", claim_id): item for claim_id, item in claims.items()}
        existing_states = {
            int(state["id"]): state for state in update_input["existing_states"]
        }
        operations: List[Dict[str, Any]] = []
        seen_states: set[tuple[str, str]] = set()
        for raw in raw_operations:
            operation = self._normalize_operation(
                raw, existing_states=existing_states,
                allowed_evidence=allowed_evidence,
                state_domain=str(update_input["state_domain"]),
                reference_time=str(update_input["reference_time"]),
            )
            if not operation or operation["state_class"] not in {
                "enduring_state", "recurring_state",
            } or operation["operation"] == "no_change":
                return None
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
                operation["state_class"], operation["state_key"]
            )
            if target_key in seen_states:
                return None
            seen_states.add(target_key)
            if operation["operation"] == "create" and any(
                state["state_class"] == operation["state_class"]
                and state["state_key"] == operation["state_key"]
                for state in existing_states.values()
            ):
                return None
            if operation["operation"] == "create" and self._db.user_state_identity_exists(
                user_entity_id=int(update_input["user_entity_id"]),
                state_domain=str(update_input["state_domain"]),
                state_class=operation["state_class"],
                state_key=operation["state_key"],
            ):
                return None
            operations.append(operation)
        return claim_updates, operations

    def _process_user_claim_state_updates(
        self, *, changed_claim_ids: Sequence[int], reference_time: Optional[Any],
    ) -> Dict[str, Any]:
        owner_id = int(self._db.add_entity_names([self._world_owner_entity_name])[
            self._world_owner_entity_name
        ])
        update_inputs = self._build_user_claim_update_inputs(
            user_entity_id=owner_id, changed_claim_ids=changed_claim_ids,
            reference_time=reference_time,
        )
        report: Dict[str, Any] = {
            "status": "empty", "domain_count": len(update_inputs),
            "changed_claim_count": sum(len(item["changed_claims"]) for item in update_inputs.values()),
            "changed_intent_object_count": 0,
            "existing_state_count": sum(len(item["existing_states"]) for item in update_inputs.values()),
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
            source = "\n".join(item["claim_text"] for item in update_input["changed_claims"])
            template = (
                USER_CLAIM_STATE_RESOLUTION_PROMPT_EN
                if self._resolve_prompt_language(source) == "en"
                else USER_CLAIM_STATE_RESOLUTION_PROMPT_ZH
            )
            prompt = template
            for key, value in update_input.items():
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
            with self._db.transaction():
                changed_status_count = sum(
                    self._db.apply_user_claim_status_decision(
                        claim_id=item["claim_id"], new_status=item["new_status"],
                        trigger_claim_id=item["trigger_claim_id"],
                        relation_event_id=item["relation_event_id"],
                        effective_at=str(update_input["reference_time"]),
                        reason=item["reason"], decision_source=self.CLAIM_RESOLVER_VERSION,
                    )
                    for item in claim_updates
                )
                applied = self.apply_operations(
                    user_entity_id=owner_id, operations=operations,
                    reference_time=update_input["reference_time"],
                    replace_claim_evidence=True,
                    decision_source=self.CLAIM_RESOLVER_VERSION,
                )
            report["claim_status_update_count"] += changed_status_count
            for key in ("operation_count", "created", "reinforced", "revised", "weakened", "expired"):
                report[key] += int(applied.get(key) or 0)
            report["state_ids"].extend(applied["state_ids"])
            report["domain_reports"][domain] = {
                "status": "ok", "changed_claim_count": len(update_input["changed_claims"]),
                "related_claim_count": len(update_input["related_claims"]),
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
        return report

    def apply_operations(
        self,
        *,
        user_entity_id: int,
        operations: Sequence[Mapping[str, Any]],
        reference_time: Optional[Any] = None,
        replace_claim_evidence: bool = False,
        decision_source: Optional[str] = None,
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
            previous_claim_evidence = (
                self._db.get_user_state_claim_evidence([target_id])
                if replace_claim_evidence and target_id else []
            )
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
                    "resolver_version": decision_source or self.RESOLVER_VERSION,
                    "last_operation": action,
                    "state_domain": str(operation["state_domain"]),
                },
            )
            current = self._db.get_user_states_by_ids([state_id])[0]
            evidence = [
                {**item, "state_id": state_id, "observed_at": effective_at}
                for item in operation.get("evidence") or []
            ]
            if replace_claim_evidence:
                self._db.replace_user_state_claim_evidence(
                    state_id=state_id, evidence=evidence,
                )
            else:
                self._db.upsert_user_state_evidence(evidence)
            self._db.record_user_state_event(
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
            "weaken": "weakened", "expire": "expired",
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
