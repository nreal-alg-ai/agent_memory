"""Controlled user-state vocabulary shared by resolution and retrieval."""

USER_STATE_TYPES = {"enduring_state", "recurring_state", "situational_state"}

USER_STATE_KINDS_BY_TYPE = {
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

USER_STATE_STATUSES = {"active", "weakened", "expired"}
