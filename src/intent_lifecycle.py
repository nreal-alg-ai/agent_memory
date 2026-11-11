"""Shared lifecycle vocabulary for goals, plans and work items.

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
