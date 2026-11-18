"""Shared business fields and lifecycle vocabulary for user intents.

Status describes the current stage, not the object's type or a past change.
Lifecycle evidence describes an observed change, not the resulting stage.
Only matched, validated and actually applied changes become persisted events.
"""

ACTIVE_USER_INTENT_STATUSES = frozenset({"pending", "in_progress", "blocked"})
CLOSED_USER_INTENT_STATUSES = frozenset({"completed", "cancelled"})
USER_INTENT_STATUSES = ACTIVE_USER_INTENT_STATUSES | CLOSED_USER_INTENT_STATUSES
USER_INTENT_LIFECYCLE_EVIDENCE_TYPES = frozenset({
    "updated", "rescheduled", "started", "blocked", "resumed", "completed", "cancelled", "reopened",
})
USER_INTENT_EVENT_TYPES = USER_INTENT_LIFECYCLE_EVIDENCE_TYPES | {"created"}
USER_INTENT_FIELD_ONLY_EVIDENCE_TYPES = frozenset({"updated", "rescheduled"})

USER_INTENT_FIELDS = (
    "world_owner_entity_id", "canonical_key", "summary", "primary_domain", "status",
    "current_progress", "time_start", "time_end", "confidence", "details", "metadata",
)
USER_INTENT_DETAIL_FIELDS = frozenset({
    "user_role", "completion_condition", "time_precision", "time_start_meaning",
    "time_end_meaning", "location_text", "participants",
})


def normalize_user_intent_details(value: dict) -> dict:
    """Validate optional shared details without requiring a completion anchor.

    Omitted values are unknown; sparse lifecycle patches retain existing details.
    Temporal meanings describe the supplied time, not an intent category.
    """
    if not isinstance(value, dict) or set(value) - USER_INTENT_DETAIL_FIELDS:
        raise ValueError("Invalid user intent details fields")
    result = {}
    for key, item in value.items():
        if key == "participants":
            if not isinstance(item, list) or any(not isinstance(v, str) or not v.strip() for v in item):
                raise ValueError("Intent participants must be a list of nonempty names")
            if item:
                result[key] = list(dict.fromkeys(v.strip() for v in item))
            continue
        if not isinstance(item, str):
            raise ValueError(f"Intent detail {key} must be text")
        item = item.strip()
        if not item:
            continue
        allowed = {
            "time_precision": {"exact", "day", "week", "month", "relative", "unknown"},
            "time_start_meaning": {"start", "available_from", "unknown"},
            "time_end_meaning": {"end", "deadline", "target", "unknown"},
        }.get(key)
        if allowed is not None and item not in allowed:
            raise ValueError(f"Invalid intent detail {key}: {item!r}")
        result[key] = item
    return result


def user_intent_status_after_evidence(current_status: str, evidence_type: str) -> str:
    """Resolve a matched activity's change against the current persisted stage."""
    if current_status not in USER_INTENT_STATUSES:
        raise ValueError(f"Invalid intent status: {current_status}")
    if evidence_type not in USER_INTENT_LIFECYCLE_EVIDENCE_TYPES:
        raise ValueError(f"Invalid lifecycle evidence: {evidence_type}")
    if evidence_type in USER_INTENT_FIELD_ONLY_EVIDENCE_TYPES:
        return current_status
    if evidence_type == "reopened":
        if current_status not in CLOSED_USER_INTENT_STATUSES | {"pending"}:
            raise ValueError("Reopening requires a closed intent")
        return "pending"
    if current_status in CLOSED_USER_INTENT_STATUSES:
        if evidence_type == current_status:
            return current_status
        raise ValueError("A closed intent requires explicit reopening before execution changes")
    if evidence_type in {"started", "resumed"}:
        # Pending means no prior execution evidence, not proof that execution
        # never happened. A newly observed resumption can fill that evidence gap.
        return "in_progress"
    return evidence_type  # blocked, completed or cancelled
