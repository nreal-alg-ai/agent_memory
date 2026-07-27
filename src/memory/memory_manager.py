#!/usr/bin/env python3
"""Unified memory manager inspired by MemPalace.

The public surface mirrors the current project's `MemoryNodeManager`, but the
internal model is deliberately unified:

1. assistant_wakeup turns and future allday transcript episodes both become
   `memory_episodes`.
2. Extracted evidence becomes narrative `memory_facts`.
3. Longer-running preferences/topics/tasks can become `memory_states`.
4. Concrete decisions/tasks/commitments become `memory_actionable_items`.
5. Every retrievable object writes a `memory_index_entries` card. Recall starts
   from this shared index instead of hard-routing between assistant/allday lines.

This file intentionally avoids ASR dependencies.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests

from .embedding_client import EmbeddingClient
from .memory_database import SessionDB
from .prompts_en import (
    RECALL_QUERY_ANALYSIS_PROMPT_EN,
    UNIFIED_ACTIONABLE_ITEM_EXTRACTION_PROMPT_EN,
    UNIFIED_MEMORY_EXTRACTION_PROMPT_EN,
    UNIFIED_STATE_UPDATE_PROMPT_EN,
)
from .prompts_zh import (
    RECALL_QUERY_ANALYSIS_PROMPT_ZH,
    UNIFIED_ACTIONABLE_ITEM_EXTRACTION_PROMPT_ZH,
    UNIFIED_MEMORY_EXTRACTION_PROMPT_ZH,
    UNIFIED_STATE_UPDATE_PROMPT_ZH,
)

logger = logging.getLogger(__name__)

DEFAULT_LLM_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_LLM_MODEL = "deepseek-v4-flash"

_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "about", "what", "which", "where", "when", "who", "why", "how", "did",
    "do", "does", "i", "me", "my", "you", "your", "we", "our", "is", "are",
    "was", "were", "be", "been", "being", "can", "could", "would", "should",
    "that", "this", "it", "as", "at", "by", "from", "have", "had", "has",
}

_COURTESY_PATTERNS = (
    "希望这个方法能帮到",
    "希望这能帮到",
    "希望对你有帮助",
    "希望对您有帮助",
    "有其他问题",
    "继续沟通",
    "随时告诉我",
    "不客气",
    "别客气",
    "很高兴能帮",
    "祝你",
    "祝您",
    "hope this helps",
    "hope that helps",
    "let me know if",
    "feel free to ask",
    "happy to help",
    "you are welcome",
    "you're welcome",
)

_WEAK_TRY_PATTERNS = (
    "愿意尝试",
    "决定尝试",
    "打算尝试",
    "尝试使用",
    "尝试选择",
    "可以试一试",
    "试一试",
    "听起来不错",
    "听起来可以",
    "可以考虑",
    "觉得可以",
    "might try",
    "may try",
    "willing to try",
    "could try",
    "sounds good",
    "sounds okay",
    "may consider",
)

_ACTIONABLE_HARD_MARKERS = (
    "提醒",
    "跟进",
    "后续",
    "确认",
    "安排",
    "预约",
    "截止",
    "待办",
    "承诺",
    "决定",
    "必须",
    "需要完成",
    "明天",
    "下周",
    "每天",
    "每周",
    "每月",
    "remind",
    "follow up",
    "confirm",
    "schedule",
    "appointment",
    "deadline",
    "todo",
    "commit",
    "decide",
    "must",
    "need to complete",
    "tomorrow",
    "next week",
    "daily",
    "weekly",
    "monthly",
)


def _now_text() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_timestamp_text(value: Any) -> str:
    if isinstance(value, datetime):
        # The LongMemEval scripts pass time filters as "YYYY-MM-DD HH:MM:SS".
        # Keep the same sortable format so SQLite string range filters work.
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value or "").strip()


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _json_safe(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class MemoryNodeManager:
    """Compatibility manager backed by a unified index-first memory line."""

    def __init__(
        self,
        db: SessionDB,
        *,
        embedding_config: Optional[Dict[str, Any]] = None,
        memory_config: Optional[Dict[str, Any]] = None,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
    ) -> None:
        self._db = db
        self._embedding_cfg = dict(embedding_config or {})
        self._memory_cfg = dict(memory_config or {})
        self._llm_model = llm_model or DEFAULT_LLM_MODEL
        self._llm_base_url = self._normalize_llm_base_url(llm_base_url or DEFAULT_LLM_BASE_URL)
        self._llm_api_key = llm_api_key or ""
        if not self._llm_api_key:
            self._llm_api_key = self._resolve_env(self._memory_cfg.get("llm_api_key"))
        self._llm_timeout = int(self._memory_cfg.get("llm_timeout", 120) or 120)
        self._llm_json_mode = self._config_bool(
            self._memory_cfg.get("llm_json_mode", True),
            True,
        )
        self._llm_thinking = str(self._memory_cfg.get("llm_thinking", "disabled") or "disabled")
        self._memory_prompt_language = str(
            self._memory_cfg.get("memory_prompt_language_mode")
            or self._memory_cfg.get("prompt_language_mode")
            or "source"
        )
        self._enabled = bool(self._memory_cfg.get("memory_enabled", True))
        self._top_k = int(self._memory_cfg.get("retrieval_top_k", 8) or 8)
        self._recall_budget = str(self._memory_cfg.get("recall_budget", "mid") or "mid")
        self._min_dialogue_turns_before_store = max(
            1,
            int(
                self._memory_cfg.get("min_dialogue_turns_before_store")
                or self._memory_cfg.get("min_dilaogue_turns_before_store")
                or self._memory_cfg.get("min_turns_before_store")
                or 1
            ),
        )
        self._max_dialogue_chars_before_store = max(
            1,
            int(
                self._memory_cfg.get("max_dialogue_chars_before_store")
                or self._memory_cfg.get("max_dilaogue_chars_before_store")
                or self._memory_cfg.get("max_chars_before_store")
                or 2000
            ),
        )
        self._pending_store_turns: List[Dict[str, Any]] = []
        self._embedding_client: Optional[EmbeddingClient] = None

    @staticmethod
    def _config_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
            return True
        if text in {"0", "false", "no", "n", "off", "disable", "disabled"}:
            return False
        return default

    @staticmethod
    def _resolve_env(value: Any) -> str:
        text = str(value or "").strip()
        match = re.fullmatch(r"\${([A-Za-z_][A-Za-z0-9_]*)}", text)
        if match:
            return os.environ.get(match.group(1), "").strip()
        return text

    @staticmethod
    def _normalize_llm_base_url(value: str) -> str:
        text = str(value or DEFAULT_LLM_BASE_URL).strip().rstrip("/")
        if text == "https://api.deepseek.com":
            return "https://api.deepseek.com/v1"
        return text or DEFAULT_LLM_BASE_URL

    # ── Runtime helpers used by benchmark scripts ───────────────────────

    def _ensure_embedding_client(self) -> bool:
        if self._embedding_client is None:
            self._embedding_client = EmbeddingClient(self._embedding_cfg)
        return True

    @staticmethod
    def _as_embedding_vector(value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        vector = np.asarray(value, dtype=np.float32)
        if vector.ndim > 1:
            vector = vector.reshape(-1)
        if vector.size == 0:
            return None
        return vector

    @classmethod
    def _cal_embedding_similarity(cls, left: Any, right: Any) -> float:
        a = cls._as_embedding_vector(left)
        b = cls._as_embedding_vector(right)
        if a is None or b is None:
            return 0.0
        av = a.reshape(-1)
        bv = b.reshape(-1)
        keep = min(av.size, bv.size)
        if keep <= 0:
            return 0.0
        av = av[:keep]
        bv = bv[:keep]
        denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
        if denom <= 0:
            return 0.0
        return float(np.dot(av, bv) / denom)

    # ── Store path: raw turn -> episode -> facts -> index cards ──────────

    def store_turn(
        self,
        user_message: str,
        assistant_response: str = "",
        *,
        tags: Optional[List[str]] = None,
        turn_timestamp: Optional[Any] = None,
        **extra: Any,
    ) -> bool:
        if not self._enabled:
            return False
        if turn_timestamp is None:
            turn_timestamp = extra.get("timestamp")
        turn = {
            "user_message": _compact_whitespace(user_message),
            "assistant_response": _compact_whitespace(assistant_response),
            "tags": list(tags or []),
            "turn_timestamp": _to_timestamp_text(turn_timestamp) or _now_text(),
        }
        if not turn["user_message"] and not turn["assistant_response"]:
            return False
        self._pending_store_turns.append(turn)
        pending_chars = sum(
            len(item.get("user_message", "")) + len(item.get("assistant_response", ""))
            for item in self._pending_store_turns
        )
        if (
            len(self._pending_store_turns) >= self._min_dialogue_turns_before_store
            or pending_chars >= self._max_dialogue_chars_before_store
        ):
            return self.flush_pending_store_turns()
        return False

    def flush_pending_store_turns(self) -> bool:
        if not self._pending_store_turns:
            return False
        turns = list(self._pending_store_turns)
        self._pending_store_turns.clear()
        return self._store_interaction_episode(turns)

    def _store_interaction_episode(self, turns: List[Dict[str, Any]]) -> bool:
        segments = self._turns_to_memory_segments(turns)
        tags = sorted({tag for turn in turns for tag in turn.get("tags", [])})
        return self._store_memory_episode(
            segments=segments,
            source_type="assistant_wakeup",
            episode_type="interaction",
            tags=tags,
            source_ref="store_turn",
        )

    def store_transcript_segments(
        self,
        segments: Sequence[Dict[str, Any]],
        *,
        source_type: str = "allday_recording",
        episode_type: str = "ambient_transcript",
        source_ref: str = "",
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Store a multi-speaker transcript episode using the unified pipeline."""
        if not self._enabled:
            return False
        normalized = self._normalize_transcript_segments(segments)
        if not normalized:
            return False
        return self._store_memory_episode(
            segments=normalized,
            source_type=source_type,
            episode_type=episode_type,
            tags=list(tags or []),
            source_ref=source_ref,
        )

    def process_transcript_segments(
        self,
        segments: Sequence[Dict[str, Any]],
        **kwargs: Any,
    ) -> bool:
        return self.store_transcript_segments(segments, **kwargs)

    def _store_memory_episode(
        self,
        *,
        segments: List[Dict[str, Any]],
        source_type: str,
        episode_type: str,
        tags: List[str],
        source_ref: str = "",
    ) -> bool:
        if not segments:
            return False
        participants = self._participants_from_segments(segments)
        started_at = segments[0].get("started_at") or _now_text()
        ended_at = segments[-1].get("ended_at") or started_at
        extracted = self._extract_memory_from_segments(segments)
        title = extracted.get("episode_title") or self._episode_title_from_segments(segments)
        summary = extracted.get("episode_summary") or self._episode_summary_from_segments(segments)
        canonical_topics = (
            extracted.get("canonical_topics")
            or self._topic_candidates(summary)
            or ["general"]
        )
        episode_id = self._db.insert_episode(
            source_type=source_type,
            episode_type=episode_type,
            title=title,
            summary=summary,
            participants=participants,
            started_at=started_at,
            ended_at=ended_at,
            metadata={
                "tags": tags,
                "segment_count": len(segments),
                "canonical_topics": canonical_topics,
                "source_ref": source_ref,
            },
        )
        self._index_episode(
            episode_id=episode_id,
            source_type=source_type,
            episode_type=episode_type,
            title=title,
            summary=summary,
            started_at=started_at,
            ended_at=ended_at,
            tags=tags,
            canonical_topics=canonical_topics,
            participants=participants,
            source_ref=source_ref,
        )

        fact_count = 0
        facts = extracted.get("facts") or []
        if not facts:
            for segment_index, segment in enumerate(segments, 1):
                fallback_facts = self._extract_segment_facts(segment, segment_index=segment_index)
                for fact in fallback_facts:
                    if not fact.get("canonical_topics"):
                        fact["canonical_topics"] = canonical_topics
                facts.extend(fallback_facts)
        for fact in facts:
            self._store_fact(
                episode_id=episode_id,
                fact=fact,
                tags=tags,
                source_type=source_type,
                participants=participants,
            )
            fact_count += 1
        return fact_count > 0

    def _extract_memory_from_turns(self, turns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract episode metadata and facts with LLM, falling back to heuristics."""
        return self._extract_memory_from_segments(self._turns_to_memory_segments(turns))

    def _extract_memory_from_segments(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract episode metadata and facts with LLM, falling back to heuristics."""
        data = self._extract_memory_with_llm(segments)
        if data and data.get("facts"):
            return data
        summary = self._episode_summary_from_segments(segments)
        return {
            "episode_title": self._episode_title_from_segments(segments),
            "episode_summary": summary,
            "canonical_topics": self._topic_candidates(summary),
            "facts": [],
        }

    def _extract_memory_with_llm(self, segments: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not self._llm_api_key or not self._llm_base_url or str(self._llm_base_url).lower() == "none":
            return None
        prompt_language = self._resolve_prompt_language_from_segments(segments)
        prompt_template = (
            UNIFIED_MEMORY_EXTRACTION_PROMPT_EN
            if prompt_language == "en"
            else UNIFIED_MEMORY_EXTRACTION_PROMPT_ZH
        )
        topic_candidates = self._collect_long_term_topic_candidates(limit=60)
        prompt = (
            prompt_template
            .replace(
                "{existing_long_term_topics}",
                self._format_existing_long_term_topics_for_prompt(topic_candidates),
            )
            .replace(
                "{dialogue_batch}",
                self._build_memory_segments_for_prompt(
                    segments,
                    prompt_language=prompt_language,
                ),
            )
        )
        for attempt in range(2):
            result = self._call_llm(prompt)
            parsed = self._parse_json_object_from_llm_text(result or "")
            if parsed is not None:
                normalized = self._normalize_llm_memory_payload(
                    parsed,
                    segments,
                    topic_candidates=topic_candidates,
                )
                if normalized is not None:
                    return normalized
            if attempt == 0:
                logger.debug("Unified memory LLM extraction failed, retrying")
        return None

    def _call_llm(self, prompt: str) -> Optional[str]:
        url = f"{self._llm_base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._llm_api_key:
            headers["Authorization"] = f"Bearer {self._llm_api_key}"
        payload: Dict[str, Any] = {
            "model": self._llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "stream": False,
            "max_tokens": 2048,
        }
        if self._llm_json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self._llm_thinking in {"disabled", "enabled"}:
            payload["thinking"] = {"type": self._llm_thinking}

        attempts = [payload]
        if "thinking" in payload:
            stripped = dict(payload)
            stripped.pop("thinking", None)
            attempts.append(stripped)
        if "response_format" in payload:
            stripped = dict(payload)
            stripped.pop("response_format", None)
            attempts.append(stripped)

        seen = set()
        for item in attempts:
            marker = json.dumps(sorted(item.keys()), ensure_ascii=False)
            if marker in seen:
                continue
            seen.add(marker)
            try:
                response = requests.post(url, json=item, headers=headers, timeout=self._llm_timeout)
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices") or []
                if choices:
                    message = choices[0].get("message") or {}
                    content = message.get("content")
                    if content:
                        return str(content)
            except requests.RequestException as exc:
                text = str(exc).lower()
                if "response_format" in text or "json" in text or "thinking" in text:
                    continue
                logger.warning("Unified memory LLM call failed: %s", exc)
                return None
        return None

    @staticmethod
    def _parse_json_object_from_llm_text(text: str) -> Optional[Dict[str, Any]]:
        raw = str(text or "").strip()
        if not raw:
            return None
        if raw.startswith("```"):
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                raw = raw[start : end + 1]
        else:
            start = raw.find("{")
            end = raw.rfind("}")
            if 0 <= start < end and (start > 0 or end < len(raw) - 1):
                raw = raw[start : end + 1]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def _normalize_llm_memory_payload(
        self,
        data: Dict[str, Any],
        segments: List[Dict[str, Any]],
        *,
        topic_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        raw_facts = data.get("facts")
        if not isinstance(raw_facts, list):
            return None
        episode_summary = _compact_whitespace(
            data.get("episode_summary") or self._episode_summary_from_segments(segments)
        )
        episode_topics = self._normalize_episode_canonical_topics(
            data.get("canonical_topics") or data.get("episode_canonical_topics"),
            topic_candidates=topic_candidates or [],
            fallback_text=episode_summary,
            limit=5,
        )
        facts: List[Dict[str, Any]] = []
        fallback_timestamp = segments[0].get("started_at") if segments else _now_text()
        for index, raw_fact in enumerate(raw_facts, 1):
            if not isinstance(raw_fact, dict):
                continue
            text = _compact_whitespace(raw_fact.get("text") or raw_fact.get("summary") or "")
            if not text:
                continue
            priority = self._normalize_priority(raw_fact.get("priority", 70))
            if priority < 60:
                continue
            fact_subject = self._normalize_fact_subject(raw_fact.get("fact_subject"))
            if fact_subject == "assistant" and self._is_low_value_assistant_closing(text):
                continue
            if fact_subject == "user" and self._is_low_value_user_acknowledgement(text):
                continue
            keywords = self._normalize_string_list(raw_fact.get("keywords"), limit=18)
            if not keywords:
                keywords = self._keywords(text, limit=18)
            entities = self._normalize_entity_names(raw_fact.get("entities"))
            if not entities:
                entities = self._entities(text)
            primary_topic = _compact_whitespace(raw_fact.get("primary_topic") or "")
            if not primary_topic:
                primary_topic = " ".join(keywords[:3]) if keywords else "general"
            fact_topics = self._normalize_episode_canonical_topics(
                [primary_topic],
                topic_candidates=[
                    {"canonical_topic": topic}
                    for topic in episode_topics
                ] + list(topic_candidates or []),
                fallback_text=text,
                limit=3,
            )
            timestamp = (
                _compact_whitespace(raw_fact.get("occurred_start") or "")
                or _compact_whitespace(raw_fact.get("occurred_end") or "")
                or fallback_timestamp
            )
            facts.append({
                "summary": text,
                "source_text": text,
                "fact_subject": fact_subject,
                "fact_kind": self._normalize_fact_kind(raw_fact.get("fact_kind")),
                "fact_type": self._normalize_fact_type(raw_fact.get("fact_type")),
                "time_key": f"{timestamp}#llm:{index:02d}",
                "keywords": " ".join(keywords),
                "entities": entities,
                "canonical_topics": fact_topics or episode_topics or [primary_topic],
                "importance": max(0.6, min(1.0, priority / 100.0)),
                "confidence": 0.9,
                "metadata": {
                    "extractor": "llm",
                    "priority": priority,
                    "occurred_start": _compact_whitespace(raw_fact.get("occurred_start") or ""),
                    "occurred_end": _compact_whitespace(raw_fact.get("occurred_end") or ""),
                    "time_confidence": _compact_whitespace(raw_fact.get("time_confidence") or "unknown"),
                    "where": _compact_whitespace(raw_fact.get("where") or ""),
                    "primary_topic": primary_topic,
                },
            })
        return {
            "episode_title": _compact_whitespace(
                data.get("episode_title") or self._episode_title_from_segments(segments)
            ),
            "episode_summary": episode_summary,
            "canonical_topics": episode_topics or self._topic_candidates(episode_summary),
            "facts": facts,
        }

    def _build_dialogue_batch_for_prompt(
        self,
        turns: List[Dict[str, Any]],
        *,
        prompt_language: str,
    ) -> str:
        is_en = prompt_language == "en"
        time_label = "Conversation timestamp" if is_en else "对话发生时间"
        user_label = "User" if is_en else "用户"
        assistant_label = "Assistant" if is_en else "助手"
        blocks: List[str] = []
        for index, turn in enumerate(turns, 1):
            blocks.append(
                "\n".join([
                    f"[Turn {index}]",
                    f"{time_label}: {turn.get('turn_timestamp') or ''}",
                    f"{user_label}: {turn.get('user_message') or ''}",
                    f"{assistant_label}: {turn.get('assistant_response') or ''}",
                ])
            )
        return "\n\n".join(blocks)

    def _turns_to_memory_segments(self, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        segments: List[Dict[str, Any]] = []
        for turn_index, turn in enumerate(turns, 1):
            timestamp = _to_timestamp_text(turn.get("turn_timestamp")) or _now_text()
            tags = list(turn.get("tags") or [])
            user_text = _compact_whitespace(turn.get("user_message") or "")
            assistant_text = _compact_whitespace(turn.get("assistant_response") or "")
            if user_text:
                segments.append({
                    "speaker": "user",
                    "role": "user",
                    "text": user_text,
                    "started_at": timestamp,
                    "ended_at": timestamp,
                    "tags": tags,
                    "turn_index": turn_index,
                })
            if assistant_text:
                segments.append({
                    "speaker": "assistant",
                    "role": "assistant",
                    "text": assistant_text,
                    "started_at": timestamp,
                    "ended_at": timestamp,
                    "tags": tags,
                    "turn_index": turn_index,
                })
        return segments

    def _normalize_transcript_segments(
        self,
        segments: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for index, segment in enumerate(segments, 1):
            if not isinstance(segment, dict):
                continue
            text = _compact_whitespace(
                segment.get("text")
                or segment.get("asr_text")
                or segment.get("reference_text")
                or segment.get("utterance")
                or ""
            )
            if not text:
                continue
            speaker = _compact_whitespace(
                segment.get("speaker")
                or segment.get("speaker_name")
                or segment.get("speaker_id")
                or "unknown_speaker"
            )
            role = _compact_whitespace(segment.get("role") or speaker or "speaker")
            started_at = _to_timestamp_text(
                segment.get("started_at")
                or segment.get("start_timestamp")
                or segment.get("timestamp")
                or segment.get("start")
            )
            ended_at = _to_timestamp_text(
                segment.get("ended_at")
                or segment.get("end_timestamp")
                or segment.get("timestamp_end")
                or segment.get("end")
                or started_at
            )
            normalized.append({
                "speaker": speaker or "unknown_speaker",
                "role": role or "speaker",
                "text": text,
                "started_at": started_at or _now_text(),
                "ended_at": ended_at or started_at or _now_text(),
                "tags": list(segment.get("tags") or []),
                "segment_index": int(segment.get("segment_index") or index),
                "metadata": dict(segment.get("metadata") or {}),
            })
        return sorted(
            normalized,
            key=lambda item: (
                str(item.get("started_at") or ""),
                str(item.get("ended_at") or ""),
                str(item.get("speaker") or ""),
            ),
        )

    @staticmethod
    def _participants_from_segments(segments: List[Dict[str, Any]]) -> List[str]:
        participants: List[str] = []
        seen: set[str] = set()
        for segment in segments:
            speaker = _compact_whitespace(segment.get("speaker") or segment.get("role") or "")
            if not speaker:
                continue
            key = speaker.lower()
            if key in seen:
                continue
            seen.add(key)
            participants.append(speaker)
        return participants or ["unknown_speaker"]

    def _build_memory_segments_for_prompt(
        self,
        segments: List[Dict[str, Any]],
        *,
        prompt_language: str,
    ) -> str:
        is_en = prompt_language == "en"
        time_label = "Time" if is_en else "时间"
        speaker_label = "Speaker" if is_en else "说话人"
        role_label = "Role" if is_en else "角色"
        text_label = "Text" if is_en else "文本"
        blocks: List[str] = []
        for index, segment in enumerate(segments, 1):
            started_at = segment.get("started_at") or ""
            ended_at = segment.get("ended_at") or started_at
            time_text = started_at if started_at == ended_at else f"{started_at} - {ended_at}"
            blocks.append(
                "\n".join([
                    f"[Segment {index}]",
                    f"{time_label}: {time_text}",
                    f"{speaker_label}: {segment.get('speaker') or ''}",
                    f"{role_label}: {segment.get('role') or ''}",
                    f"{text_label}: {segment.get('text') or ''}",
                ])
            )
        return "\n\n".join(blocks)

    def _collect_long_term_topic_candidates(self, *, limit: int = 60) -> List[Dict[str, Any]]:
        """Collect durable topic names that should guide new episode topics."""
        try:
            states = self._db.recent_states(limit=max(1, int(limit or 60)))
        except Exception as exc:
            logger.debug("Failed to load long-term topic candidates: %s", exc)
            return []
        rows: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for state in states:
            metadata = state.get("metadata") or {}
            raw_topics: List[Any] = [
                state.get("canonical_name"),
                *(metadata.get("canonical_topics") or []),
            ]
            for raw_topic in raw_topics:
                topic = self._normalize_topic_name(raw_topic)
                if not topic:
                    continue
                key = topic.lower()
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "canonical_topic": topic,
                    "state_id": state.get("id"),
                    "source_type": state.get("source_type"),
                    "state_type": state.get("state_type"),
                    "summary": _compact_whitespace(state.get("summary") or "")[:240],
                })
                if len(rows) >= limit:
                    return rows
        return rows

    @staticmethod
    def _format_existing_long_term_topics_for_prompt(topics: List[Dict[str, Any]]) -> str:
        if not topics:
            return "[]"
        rows = [
            {
                "canonical_topic": item.get("canonical_topic"),
                "source_type": item.get("source_type"),
                "state_type": item.get("state_type"),
                "summary": item.get("summary"),
            }
            for item in topics[:60]
        ]
        return json.dumps(rows, ensure_ascii=False, indent=2)

    def _normalize_episode_canonical_topics(
        self,
        value: Any,
        *,
        topic_candidates: List[Dict[str, Any]],
        fallback_text: str,
        limit: int,
    ) -> List[str]:
        raw_topics = self._coerce_topic_list(value)
        if not raw_topics:
            raw_topics = self._topic_candidates(fallback_text)
        normalized: List[str] = []
        seen: set[str] = set()
        for raw_topic in raw_topics:
            topic = self._normalize_topic_name(raw_topic)
            if not topic:
                continue
            topic = self._resolve_existing_topic_name(topic, topic_candidates) or topic
            key = topic.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(topic)
            if len(normalized) >= max(1, int(limit or 5)):
                break
        return normalized

    @staticmethod
    def _coerce_topic_list(value: Any) -> List[Any]:
        if isinstance(value, str):
            return re.split(r"[,，;；\n]+", value)
        if isinstance(value, list):
            out: List[Any] = []
            for item in value:
                if isinstance(item, dict):
                    out.append(
                        item.get("canonical_topic")
                        or item.get("topic")
                        or item.get("name")
                        or item.get("text")
                    )
                else:
                    out.append(item)
            return out
        return []

    @staticmethod
    def _normalize_topic_name(value: Any) -> str:
        text = _compact_whitespace(value)
        text = text.strip("'\".,:;!?，。！？、；：（）()[]{}")
        if not text:
            return ""
        lower = text.lower()
        if lower in _STOPWORDS:
            return ""
        if any(pattern in lower for pattern in _COURTESY_PATTERNS):
            return ""
        if re.search(r"[。！？!?；;，,]", text):
            return ""
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        if chinese_chars:
            if not (2 <= len(chinese_chars) <= 18):
                return ""
        elif not (2 <= len(text.split()) <= 6):
            return ""
        generic_topics = {
            "方案确定", "产品设计讨论", "部门协作", "问题讨论", "用户咨询",
            "solution finalized", "product design discussion",
            "team collaboration", "problem discussion", "user consultation",
        }
        if lower in generic_topics:
            return ""
        return text

    def _resolve_existing_topic_name(
        self,
        topic: str,
        topic_candidates: List[Dict[str, Any]],
    ) -> Optional[str]:
        if not topic_candidates:
            return None
        topic_key = topic.lower()
        candidate_names = [
            self._normalize_topic_name(item.get("canonical_topic"))
            for item in topic_candidates
        ]
        candidate_names = [item for item in candidate_names if item]
        for candidate in candidate_names:
            if candidate.lower() == topic_key:
                return candidate
        for candidate in candidate_names:
            candidate_key = candidate.lower()
            if len(candidate_key) >= 4 and (candidate_key in topic_key or topic_key in candidate_key):
                return candidate
        best_name = ""
        best_score = 0.0
        for candidate in candidate_names:
            score = self._topic_name_similarity(topic, candidate)
            if score > best_score:
                best_score = score
                best_name = candidate
        return best_name if best_score >= 0.62 else None

    def _topic_name_similarity(self, left: str, right: str) -> float:
        left_terms = set(self._topic_similarity_terms(left))
        right_terms = set(self._topic_similarity_terms(right))
        if not left_terms or not right_terms:
            return 0.0
        overlap = len(left_terms & right_terms)
        union = len(left_terms | right_terms)
        return overlap / max(1, union)

    def _topic_similarity_terms(self, text: str) -> List[str]:
        clean = _compact_whitespace(text).lower()
        chinese_chars = "".join(re.findall(r"[\u4e00-\u9fff]", clean))
        if len(chinese_chars) >= 3:
            return [chinese_chars[i : i + 2] for i in range(len(chinese_chars) - 1)]
        return self._keywords(clean, limit=12)

    def _resolve_prompt_language(self, turns: List[Dict[str, Any]]) -> str:
        mode = str(self._memory_prompt_language or "source").strip().lower()
        if mode in {"en", "english", "force_en"}:
            return "en"
        if mode in {"zh", "chinese", "force_zh"}:
            return "zh"
        sample = "\n".join(
            f"{turn.get('user_message','')}\n{turn.get('assistant_response','')}"
            for turn in turns[:3]
        )
        return "zh" if re.search(r"[\u4e00-\u9fff]", sample) else "en"

    def _resolve_prompt_language_from_segments(self, segments: List[Dict[str, Any]]) -> str:
        mode = str(self._memory_prompt_language or "source").strip().lower()
        if mode in {"en", "english", "force_en"}:
            return "en"
        if mode in {"zh", "chinese", "force_zh"}:
            return "zh"
        sample = "\n".join(str(segment.get("text") or "") for segment in segments[:12])
        return "zh" if re.search(r"[\u4e00-\u9fff]", sample) else "en"

    def _resolve_prompt_language_from_text(self, text: str, *, fallback: str = "zh") -> str:
        mode = str(self._memory_prompt_language or "source").strip().lower()
        if mode in {"en", "english", "force_en"}:
            return "en"
        if mode in {"zh", "chinese", "force_zh"}:
            return "zh"
        if re.search(r"[\u4e00-\u9fff]", str(text or "")):
            return "zh"
        return "en" if str(fallback).lower().startswith("en") else "zh"

    def _episode_title(self, turns: List[Dict[str, Any]]) -> str:
        for turn in turns:
            text = turn.get("user_message") or turn.get("assistant_response") or ""
            if text:
                return _compact_whitespace(text)[:96]
        return "assistant interaction episode"

    def _episode_summary(self, turns: List[Dict[str, Any]]) -> str:
        chunks: List[str] = []
        for turn in turns[:6]:
            if turn.get("user_message"):
                chunks.append(f"User: {turn['user_message']}")
            if turn.get("assistant_response"):
                chunks.append(f"Assistant: {turn['assistant_response'][:600]}")
        return "\n".join(chunks)

    def _episode_title_from_segments(self, segments: List[Dict[str, Any]]) -> str:
        for segment in segments:
            text = segment.get("text") or ""
            if text:
                return _compact_whitespace(text)[:96]
        return "memory episode"

    def _episode_summary_from_segments(self, segments: List[Dict[str, Any]]) -> str:
        chunks: List[str] = []
        for segment in segments[:10]:
            speaker = segment.get("speaker") or segment.get("role") or "speaker"
            text = _compact_whitespace(segment.get("text") or "")
            if not text:
                continue
            started_at = segment.get("started_at") or ""
            chunks.append(f"{started_at} {speaker}: {text[:600]}")
        return "\n".join(chunks)

    def _index_episode(
        self,
        *,
        episode_id: int,
        source_type: str,
        episode_type: str,
        title: str,
        summary: str,
        started_at: str,
        ended_at: str,
        tags: List[str],
        canonical_topics: List[str],
        participants: List[str],
        source_ref: str = "",
    ) -> None:
        keywords = " ".join(self._keywords(summary, limit=24))
        embedding_text = f"{title}\n{summary}\nkeywords: {keywords}"
        embedding = self._embed(embedding_text)
        self._db.upsert_index_entry(
            source_type=source_type,
            target_table="memory_episodes",
            target_id=episode_id,
            index_level="episode",
            memory_path=f"{source_type}/episodes/{episode_type}",
            title=title,
            summary_for_retrieval=summary,
            keywords=keywords,
            entities=participants,
            canonical_topics=canonical_topics,
            participants=participants,
            time_start=started_at,
            time_end=ended_at,
            importance=0.55,
            confidence=0.8,
            embedding=embedding,
            embedding_text=embedding_text,
            metadata={"tags": tags, "source_ref": source_ref},
        )

    def _extract_turn_facts(self, turn: Dict[str, Any], *, turn_index: int) -> List[Dict[str, Any]]:
        facts: List[Dict[str, Any]] = []
        timestamp = turn.get("turn_timestamp") or _now_text()
        user_text = turn.get("user_message") or ""
        assistant_text = turn.get("assistant_response") or ""
        if user_text:
            if not self._is_low_value_user_acknowledgement(user_text):
                facts.append(self._build_fact(
                    summary=f"User said: {user_text}",
                    source_text=user_text,
                    fact_subject="user",
                    fact_kind=self._infer_fact_kind(user_text, speaker="user"),
                    fact_type="episodic",
                    timestamp=timestamp,
                    turn_index=turn_index,
                    role="user",
                    importance=0.75,
                ))
            for detail in self._split_high_value_details(user_text):
                if detail != user_text:
                    facts.append(self._build_fact(
                        summary=f"User mentioned: {detail}",
                        source_text=detail,
                        fact_subject="user",
                        fact_kind=self._infer_fact_kind(detail, speaker="user"),
                        fact_type="episodic",
                        timestamp=timestamp,
                        turn_index=turn_index,
                        role="user_detail",
                        importance=0.82,
                    ))
        if assistant_text and not self._is_low_value_assistant_closing(assistant_text):
            facts.append(self._build_fact(
                summary=f"Assistant responded: {assistant_text[:1200]}",
                source_text=assistant_text,
                fact_subject="assistant",
                fact_kind=self._infer_fact_kind(assistant_text, speaker="assistant"),
                fact_type="semantic" if self._looks_like_recommendation(assistant_text) else "episodic",
                timestamp=timestamp,
                turn_index=turn_index,
                role="assistant",
                importance=0.62,
            ))
            for detail in self._assistant_answer_details(assistant_text):
                if self._is_low_value_assistant_closing(detail):
                    continue
                facts.append(self._build_fact(
                    summary=f"Assistant recommended or stated: {detail}",
                    source_text=detail,
                    fact_subject="assistant",
                    fact_kind="recommendation",
                    fact_type="semantic",
                    timestamp=timestamp,
                    turn_index=turn_index,
                    role="assistant_detail",
                    importance=0.68,
                ))
        return facts

    def _extract_segment_facts(
        self,
        segment: Dict[str, Any],
        *,
        segment_index: int,
    ) -> List[Dict[str, Any]]:
        text = _compact_whitespace(segment.get("text") or "")
        if not text:
            return []
        role = str(segment.get("role") or "").lower()
        speaker = _compact_whitespace(segment.get("speaker") or role or "speaker")
        fact_subject = "user" if role == "user" else "assistant" if role == "assistant" else "other"
        if fact_subject == "user" and self._is_low_value_user_acknowledgement(text):
            return []
        if fact_subject == "assistant" and self._is_low_value_assistant_closing(text):
            return []
        timestamp = segment.get("started_at") or _now_text()
        return [
            self._build_fact(
                summary=f"{speaker} said: {text}",
                source_text=text,
                fact_subject=fact_subject,
                fact_kind=self._infer_fact_kind(text, speaker=role or speaker),
                fact_type="episodic",
                timestamp=timestamp,
                turn_index=segment_index,
                role=speaker,
                importance=0.7,
            )
        ]

    def _is_low_value_assistant_closing(self, text: str) -> bool:
        clean = _compact_whitespace(text)
        if not clean:
            return True
        lower = clean.lower()
        if any(pattern in lower for pattern in _COURTESY_PATTERNS):
            has_specific_action = any(
                marker in lower
                for marker in (
                    "建议", "需要", "决定", "计划", "截止", "预约", "购买",
                    "recommend", "suggest", "need to", "decide", "plan", "deadline",
                )
            )
            return not has_specific_action
        return False

    @staticmethod
    def _is_low_value_user_acknowledgement(text: str) -> bool:
        clean = _compact_whitespace(text)
        if not clean:
            return True
        lower = clean.lower()
        if len(clean) > 48:
            return False
        if any(marker in lower for marker in ("?", "？", "帮我", "需要", "想要", "计划", "决定", "安排", "提醒", "购买", "预约", "need", "want", "plan", "decide", "remind")):
            return False
        acknowledgement_markers = (
            "好的", "可以", "行", "嗯", "谢谢", "试一试", "听起来", "明白",
            "ok", "okay", "thanks", "thank you", "sounds good", "i'll try",
        )
        return any(marker in lower for marker in acknowledgement_markers)

    def _build_fact(
        self,
        *,
        summary: str,
        source_text: str,
        fact_subject: str,
        fact_kind: str,
        fact_type: str,
        timestamp: str,
        turn_index: int,
        role: str,
        importance: float,
    ) -> Dict[str, Any]:
        keywords = self._keywords(source_text, limit=18)
        entities = self._entities(source_text)
        return {
            "summary": _compact_whitespace(summary),
            "source_text": source_text,
            "fact_subject": fact_subject,
            "fact_kind": fact_kind,
            "fact_type": fact_type,
            "time_key": f"{timestamp}#{turn_index:02d}:{role}",
            "keywords": " ".join(keywords),
            "entities": entities,
            "canonical_topics": self._topic_candidates(source_text),
            "importance": importance,
            "confidence": 0.88,
            "metadata": {"turn_index": turn_index, "role": role},
        }

    def _store_fact(
        self,
        *,
        episode_id: int,
        fact: Dict[str, Any],
        tags: List[str],
        source_type: str,
        participants: List[str],
    ) -> int:
        embedding_text = "\n".join([
            fact["summary"],
            f"keywords: {fact['keywords']}",
            f"entities: {', '.join(fact['entities'])}",
        ])
        embedding = self._embed(embedding_text)
        fact_id = self._db.insert_fact(
            episode_id=episode_id,
            source_type=source_type,
            fact_type=fact["fact_type"],
            fact_kind=fact["fact_kind"],
            fact_subject=fact["fact_subject"],
            summary=fact["summary"],
            keywords=fact["keywords"],
            entities=fact["entities"],
            canonical_topics=fact["canonical_topics"],
            time_key=fact["time_key"],
            confidence=fact["confidence"],
            importance=fact["importance"],
            metadata={**fact["metadata"], "tags": tags, "source_text": fact["source_text"]},
            embedding=embedding,
            embedding_text=embedding_text,
        )
        self._db.add_entity_names(fact["entities"])
        self._db.upsert_index_entry(
            source_type=source_type,
            target_table="memory_facts",
            target_id=fact_id,
            index_level="fact",
            memory_path=f"{source_type}/facts/{fact['fact_subject']}/{fact['fact_kind']}",
            title=fact["summary"][:96],
            summary_for_retrieval=fact["summary"],
            keywords=fact["keywords"],
            entities=fact["entities"],
            canonical_topics=fact["canonical_topics"],
            participants=participants,
            time_start=fact["time_key"].split("#", 1)[0],
            time_end=fact["time_key"].split("#", 1)[0],
            importance=fact["importance"],
            confidence=fact["confidence"],
            embedding=embedding,
            embedding_text=embedding_text,
            metadata={"tags": tags, **fact["metadata"]},
        )
        return fact_id

    # ── Reflection: facts -> evolving states ─────────────────────────────

    def reflect(self, *_, **kwargs: Any) -> Dict[str, int]:
        """Update durable states and actionable items from recent facts."""
        if self._pending_store_turns:
            self.flush_pending_store_turns()
        if not self._enabled:
            return {"states_updated": 0, "actionable_items_updated": 0}
        if not self._llm_api_key:
            return {"states_updated": 0, "actionable_items_updated": 0}
        limit = max(1, int(kwargs.get("limit") or self._memory_cfg.get("reflect_limit") or 100))
        facts = self._db.recent_facts(limit=limit)
        if not facts:
            return {"states_updated": 0, "actionable_items_updated": 0}
        existing_states = self._db.recent_states(limit=80)
        state_updates = self._extract_state_updates_with_llm(
            facts=facts,
            existing_states=existing_states,
        )
        states_updated = 0
        for state in state_updates:
            state_id = self._store_state(state)
            if state_id:
                states_updated += 1
        actionable_updates = self._extract_actionable_items_with_llm(facts=facts)
        actionable_items_updated = 0
        for item in actionable_updates:
            item_id = self._store_actionable_item(item)
            if item_id:
                actionable_items_updated += 1
        return {
            "states_updated": states_updated,
            "actionable_items_updated": actionable_items_updated,
        }

    def _extract_state_updates_with_llm(
        self,
        *,
        facts: List[Dict[str, Any]],
        existing_states: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        prompt_language = self._resolve_prompt_language_from_text(
            "\n".join(str(item.get("summary") or "") for item in facts[:20])
        )
        prompt_template = (
            UNIFIED_STATE_UPDATE_PROMPT_EN
            if prompt_language == "en"
            else UNIFIED_STATE_UPDATE_PROMPT_ZH
        )
        prompt = (
            prompt_template
            .replace("{existing_states}", self._format_existing_states_for_prompt(existing_states))
            .replace("{facts}", self._format_facts_for_state_prompt(facts))
        )
        result = self._call_llm(prompt)
        parsed = self._parse_json_object_from_llm_text(result or "")
        if not parsed:
            return []
        raw_states = parsed.get("states")
        if not isinstance(raw_states, list):
            return []
        normalized: List[Dict[str, Any]] = []
        valid_fact_ids = {int(item["id"]) for item in facts if item.get("id") is not None}
        for raw in raw_states:
            if not isinstance(raw, dict):
                continue
            summary = _compact_whitespace(raw.get("summary") or "")
            canonical_name = _compact_whitespace(raw.get("canonical_name") or "")
            if not summary or not canonical_name:
                continue
            evidence_ids = [
                int(value)
                for value in (raw.get("evidence_fact_ids") or [])
                if str(value).strip().isdigit() and int(value) in valid_fact_ids
            ]
            if not evidence_ids:
                continue
            state_type = self._normalize_state_type(raw.get("state_type"))
            source_type = self._state_source_type_for_facts(facts, evidence_ids)
            normalized.append({
                "state_type": state_type,
                "source_type": source_type,
                "canonical_name": canonical_name,
                "summary": summary,
                "evidence_fact_ids": evidence_ids[:24],
                "keywords": self._normalize_string_list(raw.get("keywords"), limit=18),
                "entities": self._normalize_string_list(raw.get("entities"), limit=18),
                "canonical_topics": self._normalize_string_list(raw.get("canonical_topics"), limit=8),
                "importance": self._clamp_float(raw.get("importance"), 0.0, 1.0, 0.65),
                "confidence": self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.75),
                "status": _compact_whitespace(raw.get("status") or "active") or "active",
            })
        return normalized

    def _store_state(self, state: Dict[str, Any]) -> int:
        keywords = state.get("keywords") or self._keywords(state["summary"], limit=18)
        entities = state.get("entities") or self._entities(state["summary"])
        canonical_topics = state.get("canonical_topics") or [state["canonical_name"]]
        evidence_fact_ids = [int(value) for value in state.get("evidence_fact_ids") or []]
        embedding_text = "\n".join([
            state["canonical_name"],
            state["summary"],
            f"state_type: {state['state_type']}",
            f"keywords: {' '.join(keywords)}",
            f"entities: {', '.join(entities)}",
        ])
        embedding = self._embed(embedding_text)
        state_id = self._db.upsert_state(
            state_type=state["state_type"],
            source_type=state["source_type"],
            canonical_name=state["canonical_name"],
            summary=state["summary"],
            evidence_fact_ids=evidence_fact_ids,
            confidence=state["confidence"],
            metadata={
                "keywords": keywords,
                "entities": entities,
                "canonical_topics": canonical_topics,
                "importance": state["importance"],
                "status": state["status"],
            },
            embedding=embedding,
            embedding_text=embedding_text,
        )
        if state_id:
            self._db.upsert_index_entry(
                source_type=state["source_type"],
                target_table="memory_states",
                target_id=state_id,
                index_level="state",
                memory_path=f"{state['source_type']}/states/{state['state_type']}",
                title=state["canonical_name"],
                summary_for_retrieval=state["summary"],
                keywords=" ".join(keywords),
                entities=entities,
                canonical_topics=canonical_topics,
                participants=["user", "assistant"],
                time_start="",
                time_end="",
                importance=state["importance"],
                confidence=state["confidence"],
                embedding=embedding,
                embedding_text=embedding_text,
                metadata={
                    "state_type": state["state_type"],
                    "status": state["status"],
                    "evidence_fact_ids": evidence_fact_ids,
                },
            )
        return state_id

    def _extract_actionable_items_with_llm(
        self,
        *,
        facts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        prompt_language = self._resolve_prompt_language_from_text(
            "\n".join(str(item.get("summary") or "") for item in facts[:20])
        )
        prompt_template = (
            UNIFIED_ACTIONABLE_ITEM_EXTRACTION_PROMPT_EN
            if prompt_language == "en"
            else UNIFIED_ACTIONABLE_ITEM_EXTRACTION_PROMPT_ZH
        )
        prompt = prompt_template.replace("{facts}", self._format_facts_for_state_prompt(facts))
        result = self._call_llm(prompt)
        parsed = self._parse_json_object_from_llm_text(result or "")
        if not parsed:
            return []
        raw_items = parsed.get("actionable_items")
        if not isinstance(raw_items, list):
            return []
        normalized: List[Dict[str, Any]] = []
        valid_fact_ids = {int(item["id"]) for item in facts if item.get("id") is not None}
        facts_by_id = {int(item["id"]): item for item in facts if item.get("id") is not None}
        seen_keys: set[str] = set()
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            summary = _compact_whitespace(raw.get("summary") or "")
            canonical_name = _compact_whitespace(raw.get("canonical_name") or "")
            if not summary or not canonical_name:
                continue
            evidence_ids = [
                int(value)
                for value in (raw.get("evidence_fact_ids") or [])
                if str(value).strip().isdigit() and int(value) in valid_fact_ids
            ]
            if not evidence_ids:
                continue
            source_type = self._state_source_type_for_facts(facts, evidence_ids)
            item = {
                "item_type": self._normalize_actionable_item_type(raw.get("item_type")),
                "source_type": source_type,
                "canonical_name": canonical_name,
                "summary": summary,
                "owner": self._normalize_actionable_owner(raw.get("owner")),
                "status": self._normalize_actionable_status(raw.get("status")),
                "due_at": _compact_whitespace(raw.get("due_at") or ""),
                "evidence_fact_ids": evidence_ids[:24],
                "keywords": self._normalize_string_list(raw.get("keywords"), limit=18),
                "entities": self._normalize_string_list(raw.get("entities"), limit=18),
                "canonical_topics": self._normalize_string_list(raw.get("canonical_topics"), limit=8),
                "importance": self._clamp_float(raw.get("importance"), 0.0, 1.0, 0.7),
                "confidence": self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.75),
            }
            if not self._is_high_value_actionable_item(item, facts_by_id=facts_by_id):
                continue
            dedupe_key = self._actionable_dedupe_key(item)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            normalized.append(item)
        return normalized

    def _is_high_value_actionable_item(
        self,
        item: Dict[str, Any],
        *,
        facts_by_id: Dict[int, Dict[str, Any]],
    ) -> bool:
        item_type = str(item.get("item_type") or "other")
        status = str(item.get("status") or "unknown")
        summary = _compact_whitespace(item.get("summary") or "")
        canonical_name = _compact_whitespace(item.get("canonical_name") or "")
        if not summary or item_type == "other":
            return False
        if float(item.get("confidence") or 0.0) < 0.6:
            return False

        joined = "\n".join([canonical_name, summary, str(item.get("due_at") or "")]).lower()
        evidence_text = "\n".join(
            str(facts_by_id.get(int(fact_id), {}).get("summary") or "")
            for fact_id in item.get("evidence_fact_ids") or []
        ).lower()
        all_text = f"{joined}\n{evidence_text}"
        has_hard_marker = self._has_actionable_hard_marker(all_text) or bool(item.get("due_at"))
        has_weak_try = any(pattern in all_text for pattern in _WEAK_TRY_PATTERNS)

        if has_weak_try and not self._has_explicit_followup_or_commitment(all_text, item_type=item_type):
            return False

        if item_type in {"task", "commitment", "reminder"}:
            return has_hard_marker or item_type in {"commitment", "reminder"}

        if item_type == "decision":
            if status == "decided":
                return True
            return self._has_strong_decision_marker(all_text) and not has_weak_try

        if item_type in {"follow_up", "open_question"}:
            if self._is_low_value_followup_question(all_text):
                return False
            return self._has_followup_marker(all_text)

        if item_type == "risk":
            return self._has_blocking_marker(all_text)

        if item_type == "recommendation":
            return self._has_explicit_followup_or_commitment(all_text, item_type=item_type)

        if item_type == "constraint":
            return status == "blocked" and self._has_blocking_marker(all_text) and has_hard_marker

        return False

    @staticmethod
    def _actionable_dedupe_key(item: Dict[str, Any]) -> str:
        terms: List[str] = []
        for field in ("canonical_name", "summary"):
            text = _compact_whitespace(item.get(field) or "").lower()
            text = re.sub(r"(用户|助手|agent|assistant|user)", "", text)
            text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
            terms.extend(part for part in text.split() if len(part) > 1)
        compact = "".join(terms)
        return f"{item.get('source_type')}|{item.get('item_type')}|{compact[:80]}"

    @staticmethod
    def _has_actionable_hard_marker(text: str) -> bool:
        lower = str(text or "").lower()
        if any(marker in lower for marker in _ACTIONABLE_HARD_MARKERS):
            return True
        return bool(re.search(r"(每\s*\d+\s*(分钟|小时|天|周|月)|\d+\s*(分钟|小时|天|周|月)\s*后)", lower))

    @staticmethod
    def _has_explicit_followup_or_commitment(text: str, *, item_type: str) -> bool:
        lower = str(text or "").lower()
        if item_type == "follow_up":
            return True
        if re.search(r"(提醒我|帮我提醒|请提醒|帮我记|请记住)", lower):
            return True
        return any(
            marker in lower
            for marker in (
                "跟进", "后续确认", "下次", "明天", "截止", "承诺",
                "决定执行", "已经决定", "明确采纳", "请记住", "帮我记",
                "remind", "follow up", "next time", "deadline", "commit",
                "decided to", "explicitly accepted", "remember this",
            )
        )

    @staticmethod
    def _has_strong_decision_marker(text: str) -> bool:
        lower = str(text or "").lower()
        return any(
            marker in lower
            for marker in (
                "决定", "明确", "拒绝", "否定", "放弃", "采纳", "接受",
                "不再", "已经", "最终", "decided", "explicitly", "rejected",
                "declined", "accepted", "will not", "no longer",
            )
        )

    @staticmethod
    def _has_followup_marker(text: str) -> bool:
        lower = str(text or "").lower()
        return any(
            marker in lower
            for marker in (
                "后续", "跟进", "确认", "未解决", "仍需", "需要进一步",
                "开放问题", "下次", "follow up", "confirm", "unresolved",
                "still need", "open question", "next time",
            )
        )

    @staticmethod
    def _is_low_value_followup_question(text: str) -> bool:
        lower = str(text or "").lower()
        if any(marker in lower for marker in ("提醒我", "帮我提醒", "请提醒", "帮我记", "请记住", "remind me", "remember this")):
            return False
        return any(
            marker in lower
            for marker in (
                "未明确接受", "未明确拒绝", "未明确回应", "是否愿意尝试",
                "是否采纳", "是否接受", "是否愿意", "用户未明确",
                "not explicitly accepted", "not explicitly rejected",
                "did not clearly respond", "whether the user is willing to try",
                "whether the user accepts",
            )
        )

    @staticmethod
    def _has_blocking_marker(text: str) -> bool:
        lower = str(text or "").lower()
        return any(
            marker in lower
            for marker in (
                "阻塞", "影响", "限制", "风险", "担心", "冲突", "无法",
                "不现实", "拒绝", "否定", "blocked", "blocking", "risk",
                "concern", "constraint", "prevents", "cannot", "unrealistic",
            )
        )

    def _store_actionable_item(self, item: Dict[str, Any]) -> int:
        keywords = item.get("keywords") or self._keywords(item["summary"], limit=18)
        entities = item.get("entities") or self._entities(item["summary"])
        canonical_topics = item.get("canonical_topics") or [item["canonical_name"]]
        evidence_fact_ids = [int(value) for value in item.get("evidence_fact_ids") or []]
        embedding_text = "\n".join([
            item["canonical_name"],
            item["summary"],
            f"item_type: {item['item_type']}",
            f"owner: {item['owner']}",
            f"status: {item['status']}",
            f"due_at: {item['due_at']}",
            f"keywords: {' '.join(keywords)}",
            f"entities: {', '.join(entities)}",
        ])
        embedding = self._embed(embedding_text)
        item_id = self._db.upsert_actionable_item(
            item_type=item["item_type"],
            source_type=item["source_type"],
            canonical_name=item["canonical_name"],
            summary=item["summary"],
            owner=item["owner"],
            status=item["status"],
            due_at=item["due_at"],
            evidence_fact_ids=evidence_fact_ids,
            confidence=item["confidence"],
            importance=item["importance"],
            metadata={
                "keywords": keywords,
                "entities": entities,
                "canonical_topics": canonical_topics,
            },
            embedding=embedding,
            embedding_text=embedding_text,
        )
        if item_id:
            self._db.upsert_index_entry(
                source_type=item["source_type"],
                target_table="memory_actionable_items",
                target_id=item_id,
                index_level="actionable_item",
                memory_path=f"{item['source_type']}/actionable_items/{item['item_type']}",
                title=item["canonical_name"],
                summary_for_retrieval=item["summary"],
                keywords=" ".join(keywords),
                entities=entities,
                canonical_topics=canonical_topics,
                participants=["user", "assistant"],
                time_start=item["due_at"],
                time_end=item["due_at"],
                importance=item["importance"],
                confidence=item["confidence"],
                embedding=embedding,
                embedding_text=embedding_text,
                metadata={
                    "item_type": item["item_type"],
                    "owner": item["owner"],
                    "status": item["status"],
                    "due_at": item["due_at"],
                    "evidence_fact_ids": evidence_fact_ids,
                },
            )
        return item_id

    def _format_existing_states_for_prompt(self, states: List[Dict[str, Any]]) -> str:
        if not states:
            return "[]"
        rows: List[Dict[str, Any]] = []
        for state in states[:80]:
            rows.append({
                "id": state.get("id"),
                "source_type": state.get("source_type"),
                "state_type": state.get("state_type"),
                "canonical_name": state.get("canonical_name"),
                "summary": state.get("summary"),
                "evidence_fact_ids": state.get("evidence_fact_ids") or [],
                "confidence": state.get("confidence"),
            })
        return json.dumps(rows, ensure_ascii=False, indent=2)

    def _format_facts_for_state_prompt(self, facts: List[Dict[str, Any]]) -> str:
        rows: List[Dict[str, Any]] = []
        for fact in facts[:160]:
            rows.append({
                "id": fact.get("id"),
                "source_type": fact.get("source_type"),
                "fact_type": fact.get("fact_type"),
                "fact_kind": fact.get("fact_kind"),
                "fact_subject": fact.get("fact_subject"),
                "time_key": fact.get("time_key"),
                "summary": fact.get("summary"),
                "keywords": fact.get("keywords"),
                "entities": fact.get("entities") or [],
                "canonical_topics": fact.get("canonical_topics") or [],
            })
        return json.dumps(rows, ensure_ascii=False, indent=2)

    @staticmethod
    def _normalize_state_type(value: Any) -> str:
        text = str(value or "other").strip().lower()
        allowed = {
            "preference", "task_state", "project_state", "relationship",
            "routine", "topic_state", "commitment", "constraint", "risk",
            "profile", "other",
        }
        return text if text in allowed else "other"

    @staticmethod
    def _normalize_actionable_item_type(value: Any) -> str:
        text = str(value or "other").strip().lower()
        allowed = {
            "task", "commitment", "decision", "follow_up", "open_question",
            "risk", "reminder", "recommendation", "constraint", "other",
        }
        return text if text in allowed else "other"

    @staticmethod
    def _normalize_actionable_status(value: Any) -> str:
        text = str(value or "unknown").strip().lower()
        allowed = {
            "open", "in_progress", "done", "blocked", "decided", "noted",
            "unknown",
        }
        return text if text in allowed else "unknown"

    @staticmethod
    def _normalize_actionable_owner(value: Any) -> str:
        text = str(value or "unknown").strip().lower()
        allowed = {"user", "assistant", "other", "unknown"}
        return text if text in allowed else "unknown"

    @staticmethod
    def _clamp_float(value: Any, low: float, high: float, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = default
        return max(low, min(high, number))

    @staticmethod
    def _state_source_type_for_facts(facts: List[Dict[str, Any]], fact_ids: List[int]) -> str:
        by_id = {int(item["id"]): item for item in facts if item.get("id") is not None}
        sources = {
            str(by_id[item].get("source_type") or "")
            for item in fact_ids
            if item in by_id
        }
        sources.discard("")
        if len(sources) == 1:
            return next(iter(sources))
        return "unified"

    # ── Recall path: query -> unified index -> formatted evidence ────────

    def recall(
        self,
        query: str,
        top_k: int = None,
        budget: str = None,
        tags: Optional[List[str]] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        recall_gate_mode: Optional[str] = None,
        memory_source_override: Optional[Sequence[str]] = None,
    ) -> str:
        del tags, recall_gate_mode
        if not self._enabled or not str(query or "").strip():
            return ""
        if self._pending_store_turns:
            self.flush_pending_store_turns()
        k = max(1, int(top_k or self._top_k or 8))
        b = str(budget or self._recall_budget or "mid")
        recall_plan = self._analyze_recall_query(query)
        source_types = self._normalize_source_override(memory_source_override)
        if source_types is None:
            source_types = self._normalize_source_override(recall_plan.get("source_types") or [])
        index_levels = self._normalize_index_levels(recall_plan.get("index_levels") or [])
        query_embedding = self._embed(self._query_embedding_text(query))
        terms = self._query_terms(query)
        candidate_limit = self._candidate_limit(query=query, top_k=k, budget=b)
        candidates = self._db.search_index_entries(
            source_types=source_types,
            index_levels=index_levels,
            time_start=time_start,
            time_end=time_end,
            limit=candidate_limit,
        )
        ranked = self._rank_index_entries(
            candidates,
            query=query,
            terms=terms,
            query_embedding=query_embedding,
        )
        selected = ranked[: self._final_limit(query=query, top_k=k)]
        return self._format_recall_context(query=query, entries=selected)

    def _analyze_recall_query(self, query: str) -> Dict[str, Any]:
        if not self._llm_api_key:
            return {}
        prompt_language = self._resolve_prompt_language_from_text(query, fallback="en")
        prompt_template = (
            RECALL_QUERY_ANALYSIS_PROMPT_EN
            if prompt_language == "en"
            else RECALL_QUERY_ANALYSIS_PROMPT_ZH
        )
        result = self._call_llm(prompt_template.replace("{query}", str(query or "")))
        parsed = self._parse_json_object_from_llm_text(result or "")
        return parsed if isinstance(parsed, dict) else {}

    def _normalize_source_override(self, value: Optional[Sequence[str]]) -> Optional[List[str]]:
        if not value:
            return None
        aliases = {
            "assistant": "assistant_wakeup",
            "interaction": "assistant_wakeup",
            "assistant_wakeup": "assistant_wakeup",
            "allday": "allday_recording",
            "all_day": "allday_recording",
            "transcript": "allday_recording",
            "allday_recording": "allday_recording",
        }
        out: List[str] = []
        for item in value:
            normalized = aliases.get(str(item or "").strip().lower())
            if normalized and normalized not in out:
                out.append(normalized)
        return out or None

    def _normalize_index_levels(self, value: Optional[Sequence[str]]) -> Optional[List[str]]:
        if not value:
            return None
        allowed = {"episode", "fact", "state", "actionable_item"}
        out: List[str] = []
        for item in value:
            text = str(item or "").strip().lower()
            if text in allowed and text not in out:
                out.append(text)
        return out or None

    def _rank_index_entries(
        self,
        entries: List[Dict[str, Any]],
        *,
        query: str,
        terms: List[str],
        query_embedding: Optional[np.ndarray],
    ) -> List[Dict[str, Any]]:
        query_lower = str(query or "").lower()
        scored: List[Tuple[float, str, int, Dict[str, Any]]] = []
        for position, entry in enumerate(entries):
            text = " ".join(
                str(entry.get(key) or "")
                for key in (
                    "title", "summary_for_retrieval", "keywords",
                    "memory_path", "canonical_topics", "entities",
                )
            ).lower()
            term_hits = sum(1 for term in terms if term in text)
            phrase_bonus = 0.0
            for quoted in re.findall(r"'([^']+)'|\"([^\"]+)\"", query):
                phrase = (quoted[0] or quoted[1]).lower()
                if phrase and phrase in text:
                    phrase_bonus += 3.0
            for ngram in self._important_ngrams(query_lower):
                if ngram in text:
                    phrase_bonus += 1.0
            sim = self._cal_embedding_similarity(query_embedding, entry.get("embedding"))
            level_bonus = 0.35 if entry.get("index_level") == "fact" else 0.15
            importance = float(entry.get("importance") or 0.5)
            confidence = float(entry.get("confidence") or 0.8)
            score = (
                term_hits * 0.85
                + phrase_bonus
                + max(0.0, sim) * 2.2
                + importance * 0.55
                + confidence * 0.35
                + level_bonus
                + (1.0 / max(1, position + 1)) * 0.25
            )
            item = dict(entry)
            item["_recall_score"] = round(score, 4)
            item["_embedding_similarity"] = round(float(sim), 4)
            item.pop("embedding", None)
            scored.append((score, str(entry.get("time_start") or ""), -position, item))
        scored.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
        return [item for _, _, _, item in scored]

    def _format_recall_context(self, *, query: str, entries: List[Dict[str, Any]]) -> str:
        if not entries:
            return ""
        lines = [
            "[Unified Memory Recall — MemPalace-style index entries]",
            "System note: All memories are retrieved from one shared index. "
            "Each entry keeps source_type, memory_path, time, and a pointer to the original table.",
            f"Query: {query}",
            "",
        ]
        for rank, entry in enumerate(entries, 1):
            meta = {
                "source": entry.get("source_type"),
                "level": entry.get("index_level"),
                "path": entry.get("memory_path"),
                "target": f"{entry.get('target_table')}#{entry.get('target_id')}",
                "score": entry.get("_recall_score"),
            }
            time_text = entry.get("time_start") or entry.get("time_end") or "unknown-time"
            lines.append(
                f"{rank}. [{time_text}] {entry.get('summary_for_retrieval')}\n"
                f"   metadata: {_json_safe(meta)}\n"
                f"   keywords: {entry.get('keywords') or ''}"
            )
        return "\n".join(lines)

    # ── Lightweight NLP heuristics ───────────────────────────────────────

    def _embed(self, text: str) -> Optional[np.ndarray]:
        self._ensure_embedding_client()
        return self._embedding_client.embed_text(text) if self._embedding_client else None

    def _query_embedding_text(self, query: str) -> str:
        terms = self._query_terms(query)
        return f"{query}\nkeywords: {' '.join(terms)}"

    def _query_terms(self, text: str) -> List[str]:
        terms = self._keywords(text, limit=32)
        for phrase in re.findall(r"'([^']+)'|\"([^\"]+)\"", str(text or "")):
            clean = _compact_whitespace(phrase[0] or phrase[1]).lower()
            if clean and clean not in terms:
                terms.insert(0, clean)
        return terms

    def _keywords(self, text: str, *, limit: int) -> List[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_.$'-]*|\d+(?:/\d+)?|[\u4e00-\u9fff]{2,}", str(text or "").lower())
        counts = Counter(
            self._normalize_keyword_term(token)
            for token in tokens
            if self._is_valid_keyword_term(self._normalize_keyword_term(token))
        )
        return [term for term, _count in counts.most_common(limit)]

    def _entities(self, text: str) -> List[str]:
        entities: List[str] = []
        for match in re.findall(r"\b[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,4}\b", str(text or "")):
            clean = _compact_whitespace(match)
            if clean.lower() not in _STOPWORDS and clean not in entities:
                entities.append(clean)
            if len(entities) >= 12:
                break
        return entities

    def _topic_candidates(self, text: str) -> List[str]:
        keywords = self._keywords(text, limit=6)
        if not keywords:
            return []
        topics: List[str] = []
        for size in (3, 2):
            if len(keywords) >= size:
                topics.append(" ".join(keywords[:size]))
        topics.append(keywords[0])
        return list(dict.fromkeys(topics))[:3]

    def _important_ngrams(self, text: str) -> List[str]:
        tokens = [term for term in self._keywords(text, limit=12) if len(term) > 2]
        ngrams: List[str] = []
        for i in range(len(tokens) - 1):
            ngrams.append(f"{tokens[i]} {tokens[i + 1]}")
        return ngrams[:10]

    def _split_high_value_details(self, text: str) -> List[str]:
        raw = _compact_whitespace(text)
        if not raw:
            return []
        markers = r"\b(?:by the way|also|actually|i just|i still|i need|i have|i attended|i bought|i got|i went|i started|i prefer|i like|i want|i realized)\b"
        pieces = [raw]
        for part in re.split(markers, raw, flags=re.IGNORECASE):
            clean = _compact_whitespace(part)
            if len(clean) >= 28:
                pieces.append(clean)
        for sentence in re.split(r"(?<=[.!?])\s+", raw):
            clean = _compact_whitespace(sentence)
            if len(clean) >= 28:
                pieces.append(clean)
        return list(dict.fromkeys(pieces))[:8]

    def _assistant_answer_details(self, text: str) -> List[str]:
        details: List[str] = []
        for line in str(text or "").splitlines():
            clean = _compact_whitespace(re.sub(r"^[*\-\d.)\s]+", "", line))
            if 20 <= len(clean) <= 260:
                details.append(clean)
            if len(details) >= 8:
                break
        return list(dict.fromkeys(details))

    def _looks_like_recommendation(self, text: str) -> bool:
        lower = str(text or "").lower()
        return any(word in lower for word in ("recommend", "suggest", "try", "consider", "tips", "advice"))

    def _infer_fact_kind(self, text: str, *, speaker: str) -> str:
        lower = str(text or "").lower()
        if any(word in lower for word in ("prefer", "favorite", "like", "dislike", "would rather")):
            return "preference"
        if any(word in lower for word in ("decided", "i'll", "i will", "plan to", "going to")):
            return "decision" if speaker == "user" else "recommendation"
        if any(word in lower for word in ("need to", "have to", "should", "todo", "pick up", "return")):
            return "action"
        if any(word in lower for word in ("recommend", "suggest", "consider", "try")):
            return "recommendation"
        if "?" in text:
            return "request"
        return "context"

    @staticmethod
    def _normalize_priority(value: Any) -> int:
        try:
            return max(0, min(100, int(round(float(value)))))
        except (TypeError, ValueError):
            return 70

    @staticmethod
    def _normalize_fact_type(value: Any) -> str:
        text = str(value or "episodic").strip().lower()
        return text if text in {"semantic", "episodic"} else "episodic"

    @staticmethod
    def _normalize_fact_subject(value: Any) -> str:
        text = str(value or "other").strip().lower()
        allowed = {"user", "assistant", "world", "project", "system", "other"}
        return text if text in allowed else "other"

    @staticmethod
    def _normalize_fact_kind(value: Any) -> str:
        text = str(value or "context").strip().lower()
        allowed = {
            "preference", "decision", "request", "recommendation", "action",
            "commitment", "open_question", "risk", "error", "context",
            "instruction", "other",
        }
        return text if text in allowed else "context"

    @staticmethod
    def _normalize_string_list(value: Any, *, limit: int = 12) -> List[str]:
        if isinstance(value, str):
            raw = re.split(r"[,，;；\n]+", value)
        elif isinstance(value, list):
            raw = value
        else:
            raw = []
        out: List[str] = []
        seen = set()
        for item in raw:
            text = MemoryNodeManager._normalize_keyword_term(item)
            if not MemoryNodeManager._is_valid_keyword_term(text) or text in seen:
                continue
            seen.add(text)
            out.append(text)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _normalize_keyword_term(value: Any) -> str:
        text = _compact_whitespace(value)
        return text.strip("'\".,:;!?，。！？、；：（）()[]{}")

    @staticmethod
    def _is_valid_keyword_term(text: str) -> bool:
        clean = _compact_whitespace(text)
        if not clean:
            return False
        lower = clean.lower()
        if lower in _STOPWORDS:
            return False
        if any(pattern in lower for pattern in _COURTESY_PATTERNS):
            return False
        if re.search(r"[。！？!?；;，,]", clean):
            return False
        if re.search(r"(好的|谢谢|不客气|继续沟通|有其他问题|帮到您|帮到你)", clean):
            return False
        if re.search(r"^(好的|谢谢|嗯|行|可以|ok|okay|thanks)$", lower):
            return False
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", clean)
        if chinese_chars and len(chinese_chars) > 10:
            return False
        if not chinese_chars and len(clean.split()) > 4:
            return False
        return len(clean) > 1

    @staticmethod
    def _normalize_entity_names(value: Any, *, limit: int = 16) -> List[str]:
        if isinstance(value, str):
            raw = re.split(r"[,，;；\n]+", value)
        elif isinstance(value, list):
            raw = value
        else:
            raw = []
        out: List[str] = []
        seen = set()
        for item in raw:
            if isinstance(item, dict):
                text = _compact_whitespace(item.get("name") or item.get("text") or "")
            else:
                text = _compact_whitespace(item)
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
            if len(out) >= limit:
                break
        return out

    def _candidate_limit(self, *, query: str, top_k: int, budget: str) -> int:
        multiplier = {"low": 10, "mid": 18, "high": 28}.get(str(budget).lower(), 18)
        if self._needs_broad_evidence(query):
            multiplier = max(multiplier, 28)
        return max(80, int(top_k) * multiplier)

    def _final_limit(self, *, query: str, top_k: int) -> int:
        if self._needs_broad_evidence(query):
            return max(int(top_k), 20)
        return max(int(top_k), 10)

    def _needs_broad_evidence(self, query: str) -> bool:
        lower = str(query or "").lower()
        return any(
            marker in lower
            for marker in (
                "how many", "count", "total", "which", "first", "before",
                "after", "between", "compare", "all", "what were",
            )
        )
