"""音频事件预处理：final 过滤、说话人标注、敏感信息脱敏。"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


# 敏感信息模式：验证码、密码、银行卡、身份证等，命中后整段不写入记忆。
_SENSITIVE_PATTERNS = (
    re.compile(r"(验证码|校验码|动态码)\s*[是为：:]?\s*[0-9A-Za-z]{4,8}", re.IGNORECASE),
    re.compile(r"(密码|口令|pin码|pin)\s*[是为：:]?\s*[0-9A-Za-z@#!._-]{4,32}", re.IGNORECASE),
    re.compile(r"\b\d{4}\s?-?\d{4}\s?-?\d{4}\s?-?\d{4}\b"),
    re.compile(r"\b\d{17}[\dXx]\b"),
    re.compile(r"(api[_ -]?key|access[_ -]?token|secret)\b", re.IGNORECASE),
)


def _contains_sensitive_text(text: str) -> bool:
    return any(pattern.search(text) for pattern in _SENSITIVE_PATTERNS)


def preprocess_audio_event(
    event: Dict[str, Any],
    private: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """返回标准化事件；不符合条件（partial/拒绝/空/敏感）返回 None。"""
    if event.get("final") is not True:
        return None
    event_type = str(event.get("type") or "")
    if event_type in {"speech_rejected", "partial", "speaker_update"}:
        return None
    text = str(event.get("text") or "").strip()
    if not text:
        return None
    if _contains_sensitive_text(text):
        return None

    speaker = event.get("speaker") if isinstance(event.get("speaker"), dict) else {}
    speaker_label = str(speaker.get("label") or "").strip() or "unknown"
    start_ms = event.get("start_ms")
    try:
        timestamp = float(start_ms or 0) / 1000.0
    except (TypeError, ValueError):
        timestamp = None

    return {
        "event_id": str(event.get("event_id") or ""),
        "lane": str(event.get("lane") or "ambient"),
        "speaker": speaker_label,
        "text": text,
        "source_type": str(event.get("source_type") or "ambient_audio"),
        "timestamp": timestamp,
    }


def is_assistant_query(event: Dict[str, Any]) -> bool:
    return str(event.get("lane") or "") == "assistant" or str(event.get("source_type") or "") == "wake_query"


def is_speaker_enrollment(event: Dict[str, Any]) -> bool:
    return str(event.get("type") or "") == "speaker_update" or str(event.get("source_type") or "") == "speaker_enrollment"
