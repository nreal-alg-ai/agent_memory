"""Standard-library storage helpers shared by independent database modules."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any


def local_now_text() -> str:
    return datetime.now().astimezone().isoformat()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def json_loads(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except Exception:
        return default
