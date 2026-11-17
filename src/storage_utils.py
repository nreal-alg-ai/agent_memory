"""Standard-library storage helpers shared by independent database modules."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterable


def local_now_text() -> str:
    return datetime.now().astimezone().isoformat()


def normalize_reference_timestamp(value: Any) -> str:
    """Normalize a supplied business clock; never replace invalid input with wall time."""
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return parsed.isoformat()


def latest_timestamp(values: Iterable[Any]) -> str | None:
    timestamps = [normalize_reference_timestamp(value) for value in values if value not in (None, "")]
    return max(timestamps, key=lambda value: datetime.fromisoformat(value).timestamp(), default=None)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def json_loads(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except Exception:
        return default
