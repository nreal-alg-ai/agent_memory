"""Conservative occurrence grouping for fact-grounded user action signals.

An action signal is evidence, not an independent repetition.  Signals retain
their evidence rows and share an occurrence_group_id when they describe one
occurrence. Coarse or missing times cannot establish separate repetitions.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict


_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T].*)?$")
_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
_YEAR_RE = re.compile(r"^\d{4}$")
_OWNER_WORDS = ("speaker_用户本人", "用户本人", "本人", "the user", "user")
_TIME_WORDS = (
    "今天", "昨天", "前天", "最近", "刚刚", "上周末", "本周", "上周",
    "today", "yesterday", "recently", "last week",
)
_NEGATIVE_MARKERS = ("没有", "没去", "未", "取消", "not", "didnt")
_AGGREGATE_MARKERS = (
    "经常", "通常", "总是", "每周", "每天", "每月", "多次", "几次", "两次",
    "反复", "习惯", "often", "usually", "every", "multiple", "several",
)
_VAGUE_TIME_MARKERS = (
    "周末", "最近", "近期", "前段时间", "这几天", "上个月", "这个月",
    "去年", "今年夏天", "around", "recently", "last weekend", "last month",
)
_DAYPART_WORDS = {
    "morning": ("凌晨", "清晨", "早上", "上午", "morning"),
    "noon": ("中午", "正午", "noon"),
    "afternoon": ("下午", "afternoon"),
    "evening": ("傍晚", "晚上", "夜里", "夜晚", "evening", "night"),
}


def occurrence_profile(event_time_key: Any, time_confidence: Any) -> Dict[str, str]:
    """Classify the *event* time; never substitute dialogue/ingestion time."""
    key = str(event_time_key or "").strip()
    confidence = str(time_confidence or "").strip()
    if not key or confidence == "unknown":
        return {"occurrence_key": "", "time_precision": "unknown",
                "independence_status": "unresolved"}
    if _DAY_RE.fullmatch(key) and len(key) > 10:
        try:
            instant = datetime.fromisoformat(key.replace("Z", "+00:00"))
        except ValueError:
            pass
        else:
            return {
                "occurrence_key": instant.isoformat(timespec="minutes"),
                "time_precision": "minute",
                "independence_status": "countable",
            }
    if _DAY_RE.fullmatch(key):
        return {"occurrence_key": key[:10], "time_precision": "day",
                "independence_status": "countable"}
    if _MONTH_RE.fullmatch(key):
        return {"occurrence_key": key, "time_precision": "month",
                "independence_status": "unresolved"}
    if _YEAR_RE.fullmatch(key):
        return {"occurrence_key": key, "time_precision": "year",
                "independence_status": "unresolved"}
    return {"occurrence_key": "", "time_precision": "unknown",
            "independence_status": "unresolved"}


def normalize_action_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    for word in (*_OWNER_WORDS, *_TIME_WORDS):
        text = text.replace(word, "")
    return "".join(char for char in text if char.isalnum())


def describes_aggregate_action(value: Any) -> bool:
    """Repeated/habitual wording is not one observed occurrence."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return any(marker in text for marker in _AGGREGATE_MARKERS)


def has_vague_occurrence_time(value: Any) -> bool:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return any(marker in text for marker in _VAGUE_TIME_MARKERS)


def _daypart(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return next(
        (part for part, words in _DAYPART_WORDS.items()
         if any(word in text for word in words)),
        "",
    )


def same_action(left: Any, right: Any) -> bool:
    """Accept close paraphrases, but not merely a shared broad topic."""
    a, b = normalize_action_text(left), normalize_action_text(right)
    if not a or not b:
        return False
    if _has_negative_marker(a) != _has_negative_marker(b):
        return False
    if a == b:
        return True
    if min(len(a), len(b)) >= 6 and (a in b or b in a):
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.86


def _has_negative_marker(text: str) -> bool:
    return any(marker in text for marker in _NEGATIVE_MARKERS)


def potential_same_occurrence(signal: Dict[str, Any], grouped: Dict[str, Any]) -> bool:
    """Structural guard before either lexical or semantic action comparison."""
    if int(signal["subject_entity_id"]) != int(grouped["subject_entity_id"]):
        return False
    if signal["action_kind"] != grouped["action_kind"]:
        return False
    if signal["occurrence_key"] != grouped["occurrence_key"]:
        return False
    if signal["time_precision"] != grouped["time_precision"]:
        return False
    if signal["independence_status"] != grouped["independence_status"]:
        return False
    # An unknown/month-level time cannot establish that two facts report one
    # occurrence.  Only duplicate signals inside the same fact may coalesce.
    if signal["independence_status"] != "countable" and (
        int(signal["fact_id"]) != int(grouped["fact_id"])
    ):
        return False
    signal_location = normalize_action_text(signal.get("location") or "")
    grouped_location = normalize_action_text(grouped.get("location") or "")
    if signal_location and grouped_location and signal_location != grouped_location:
        return False
    signal_daypart = _daypart(signal["action_text"])
    grouped_daypart = _daypart(grouped["action_text"])
    if signal_daypart != grouped_daypart:
        return False
    if _has_negative_marker(normalize_action_text(signal["action_text"])) != (
        _has_negative_marker(normalize_action_text(grouped["action_text"]))
    ):
        return False
    return True


def same_occurrence(signal: Dict[str, Any], grouped: Dict[str, Any]) -> bool:
    if not potential_same_occurrence(signal, grouped):
        return False
    return same_action(signal["action_text"], grouped["action_text"])
