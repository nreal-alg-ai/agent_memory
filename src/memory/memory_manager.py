#!/usr/bin/env python3
"""Unified memory manager inspired by MemPalace.

The public surface mirrors the current project's `MemoryNodeManager`, but the
internal model is deliberately unified:

1. assistant_wakeup turns and future allday transcript episodes both become
   `memory_episodes`.
2. Extracted evidence becomes narrative `memory_facts`.
3. Longer-running preferences/topics/constraints can become `memory_states`.
4. Concrete decisions/tasks/commitments become `memory_actionable_items`.
5. Every retrievable object writes a `memory_index_entries` card. The cards are
   retained as an optional future directory, while the default recall path
   searches the raw facts/states/actionable-items tables directly.

The current recall path searches the raw memory tables first. The shared
`memory_index_entries` layer remains available for a future optional retrieval
mode, but it is not a hard gate for recall.

This file intentionally avoids ASR dependencies.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests

from .embedding_client import EmbeddingClient
from .memory_database import SessionDB
from .prompts_en import (
    RECALL_QUERY_ANALYSIS_PROMPT_EN,
    UNIFIED_ACTIONABLE_ITEM_EXTRACTION_PROMPT_EN,
    UNIFIED_ENTITY_STATE_UPDATE_PROMPT_EN,
    UNIFIED_MEMORY_EXTRACTION_PROMPT_EN,
    UNIFIED_TOPIC_STATE_UPDATE_PROMPT_EN,
)
from .prompts_zh import (
    RECALL_QUERY_ANALYSIS_PROMPT_ZH,
    UNIFIED_ACTIONABLE_ITEM_EXTRACTION_PROMPT_ZH,
    UNIFIED_ENTITY_STATE_UPDATE_PROMPT_ZH,
    UNIFIED_MEMORY_EXTRACTION_PROMPT_ZH,
    UNIFIED_TOPIC_STATE_UPDATE_PROMPT_ZH,
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
        self._recall_fact_min_embedding_similarity = self._clamp_float(
            self._memory_cfg.get("recall_fact_min_embedding_similarity"),
            0.0,
            1.0,
            0.35,
        )
        self._recall_state_min_embedding_similarity = self._clamp_float(
            self._memory_cfg.get("recall_state_min_embedding_similarity"),
            0.0,
            1.0,
            0.35,
        )
        self._recall_actionable_item_min_embedding_similarity = self._clamp_float(
            self._memory_cfg.get("recall_actionable_item_min_embedding_similarity"),
            0.0,
            1.0,
            0.35,
        )
        self._recall_context_char_budgets = {
            "low": max(1200, int(self._memory_cfg.get("recall_context_chars_low", 3200) or 3200)),
            "mid": max(1800, int(self._memory_cfg.get("recall_context_chars_mid", 6000) or 6000)),
            "high": max(2400, int(self._memory_cfg.get("recall_context_chars_high", 10000) or 10000)),
        }
        shared_recall_context_budget = self._memory_cfg.get("recall_context_max_chars")
        if shared_recall_context_budget not in (None, ""):
            shared_budget = max(1200, int(shared_recall_context_budget or 0))
            self._recall_context_char_budgets = {
                key: shared_budget for key in self._recall_context_char_budgets
            }
        self._recall_entry_char_budgets = {
            "low": max(260, int(self._memory_cfg.get("recall_entry_chars_low", 520) or 520)),
            "mid": max(320, int(self._memory_cfg.get("recall_entry_chars_mid", 760) or 760)),
            "high": max(420, int(self._memory_cfg.get("recall_entry_chars_high", 1100) or 1100)),
        }
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
        self._enable_topic_state_resolution = self._config_bool(
            self._memory_cfg.get("enable_topic_state_resolution", True),
            True,
        )
        self._topic_state_max_topics_per_episode = max(
            1,
            int(self._memory_cfg.get("topic_state_max_topics_per_episode", 3) or 3),
        )
        self._topic_state_resolution_similarity_threshold = float(
            self._memory_cfg.get("topic_state_resolution_similarity_threshold", 0.62) or 0.62
        )
        self._topic_state_grounding_similarity_threshold = float(
            self._memory_cfg.get("topic_state_grounding_similarity_threshold", 0.42) or 0.42
        )
        self._topic_identity_embedding_similarity_threshold = float(
            self._memory_cfg.get("topic_identity_embedding_similarity_threshold", 0.78) or 0.78
        )
        self._topic_identity_grounding_similarity_threshold = float(
            self._memory_cfg.get("topic_identity_grounding_similarity_threshold", 0.64) or 0.64
        )
        self._pending_unresolved_topic_max = max(
            1,
            int(self._memory_cfg.get("pending_unresolved_topic_max", 200) or 200),
        )
        self._pending_unresolved_topics: List[Dict[str, Any]] = []
        self._enable_entity_scoped_state_resolution = self._config_bool(
            self._memory_cfg.get("enable_entity_scoped_state_resolution", True),
            True,
        )
        self._entity_state_max_entities_per_fact = max(
            1,
            int(self._memory_cfg.get("entity_state_max_entities_per_fact", 4) or 4),
        )
        self._entity_state_resolution_similarity_threshold = float(
            self._memory_cfg.get("entity_state_resolution_similarity_threshold", 0.72) or 0.72
        )
        self._entity_state_attribute_similarity_threshold = float(
            self._memory_cfg.get("entity_state_attribute_similarity_threshold", 0.62) or 0.62
        )

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
        memory_state_context = self._collect_memory_state_context(limit=12)
        prompt = (
            prompt_template
            .replace(
                "{existing_memory_states}",
                self._format_memory_states_for_prompt(memory_state_context),
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
        fallback_timestamp = _to_timestamp_text(
            segments[0].get("started_at") if segments else ""
        ) or _now_text()
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
            primary_entity = self._normalize_primary_entity(
                raw_fact.get("primary_entity"),
                entities=entities,
                fact_subject=fact_subject,
            )
            if primary_entity:
                primary_entity_name = primary_entity["name"]
                if primary_entity_name not in entities:
                    entities = [primary_entity_name, *entities]
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
            facts.append({
                "summary": text,
                "source_text": text,
                "fact_subject": fact_subject,
                "fact_kind": self._normalize_fact_kind(raw_fact.get("fact_kind")),
                "fact_type": self._normalize_fact_type(raw_fact.get("fact_type")),
                "time_key": f"{fallback_timestamp}#llm:{index:02d}",
                "keywords": " ".join(keywords),
                "entities": entities,
                "primary_entity": primary_entity,
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
                    "primary_entity": primary_entity,
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
            if str(state.get("state_scope") or "") != "topic_state":
                continue
            if str(state.get("state_type") or "") != "topic":
                continue
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
                    "state_scope": state.get("state_scope"),
                    "state_type": state.get("state_type"),
                    "summary": _compact_whitespace(state.get("summary") or "")[:240],
                })
                if len(rows) >= limit:
                    return rows
        return rows

    def _collect_memory_state_context(self, *, limit: int = 12) -> List[Dict[str, Any]]:
        """Collect a small, balanced state reference set for fact extraction."""
        try:
            states = self._db.recent_states(limit=max(40, int(limit or 12) * 6))
        except Exception as exc:
            logger.debug("Failed to load memory state context: %s", exc)
            return []
        rows: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str]] = set()
        scope_counts: Counter[str] = Counter()
        type_counts: Counter[Tuple[str, str]] = Counter()
        scope_limits = {"topic_state": 4, "entity_state": 8}
        type_limits = {"topic": 4}
        max_items = max(1, int(limit or 12))
        for state in states:
            scope = _compact_whitespace(state.get("state_scope") or "")
            state_type = _compact_whitespace(state.get("state_type") or "")
            canonical_name = _compact_whitespace(state.get("canonical_name") or "")
            summary = self._normalize_state_summary(
                state.get("summary") or "",
                max_chars=120,
            )
            if scope not in scope_limits or not state_type or not canonical_name or not summary:
                continue
            key = (scope, state_type, canonical_name.lower())
            if key in seen:
                continue
            if scope_counts[scope] >= scope_limits[scope]:
                continue
            type_key = (scope, state_type)
            if type_counts[type_key] >= type_limits.get(state_type, 2):
                continue
            seen.add(key)
            scope_counts[scope] += 1
            type_counts[type_key] += 1
            rows.append({
                "state_scope": scope,
                "state_type": state_type,
                "canonical_name": canonical_name[:60],
                "summary": summary,
            })
            if len(rows) >= max_items:
                break
        return rows

    @staticmethod
    def _format_memory_states_for_prompt(
        states: List[Dict[str, Any]],
        *,
        max_chars: int = 1800,
    ) -> str:
        if not states:
            return "[]"
        rows = list(states)
        while rows:
            text = json.dumps(rows, ensure_ascii=False, indent=2)
            if len(text) <= max_chars:
                return text
            rows.pop()
        return "[]"

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

    @staticmethod
    def _truncate_index_text(text: Any, *, max_chars: int = 700) -> str:
        clean = _compact_whitespace(text or "")
        if len(clean) <= max_chars:
            return clean
        return clean[: max(0, max_chars - 3)].rstrip() + "..."

    def _build_index_embedding_text(
        self,
        *,
        title: str,
        summary: str,
        keywords: Any,
        entities: Sequence[str],
        canonical_topics: Sequence[str],
        memory_path: str,
        max_summary_chars: int = 520,
    ) -> str:
        if isinstance(keywords, str):
            keyword_text = keywords
        else:
            keyword_text = " ".join(str(item or "") for item in keywords)
        return "\n".join(
            part
            for part in (
                f"title: {self._truncate_index_text(title, max_chars=120)}",
                f"summary: {self._truncate_index_text(summary, max_chars=max_summary_chars)}",
                f"topics: {', '.join(str(item or '') for item in canonical_topics[:8])}",
                f"entities: {', '.join(str(item or '') for item in entities[:12])}",
                f"keywords: {self._truncate_index_text(keyword_text, max_chars=180)}",
                f"path: {memory_path}",
            )
            if part.strip()
        )

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
        memory_path = f"{source_type}/episodes/{episode_type}"
        summary_card = self._truncate_index_text(summary, max_chars=700)
        embedding_text = self._build_index_embedding_text(
            title=title,
            summary=summary_card,
            keywords=keywords,
            entities=participants,
            canonical_topics=canonical_topics,
            memory_path=memory_path,
        )
        embedding = self._embed(embedding_text)
        self._db.upsert_index_entry(
            source_type=source_type,
            target_table="memory_episodes",
            target_id=episode_id,
            index_level="episode",
            memory_path=memory_path,
            title=title,
            summary_for_retrieval=summary_card,
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
        primary_entity = self._normalize_primary_entity(
            None,
            entities=entities,
            fact_subject=fact_subject,
        )
        return {
            "summary": _compact_whitespace(summary),
            "source_text": source_text,
            "fact_subject": fact_subject,
            "fact_kind": fact_kind,
            "fact_type": fact_type,
            "time_key": f"{timestamp}#{turn_index:02d}:{role}",
            "keywords": " ".join(keywords),
            "entities": entities,
            "primary_entity": primary_entity,
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
            f"primary_entity: {(fact.get('primary_entity') or {}).get('name', '') if isinstance(fact.get('primary_entity'), dict) else ''}",
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
        memory_path = f"{source_type}/facts/{fact['fact_subject']}/{fact['fact_kind']}"
        summary_card = self._truncate_index_text(fact["summary"], max_chars=760)
        index_embedding_text = self._build_index_embedding_text(
            title=fact["summary"][:96],
            summary=summary_card,
            keywords=fact["keywords"],
            entities=fact["entities"],
            canonical_topics=fact["canonical_topics"],
            memory_path=memory_path,
            max_summary_chars=620,
        )
        index_embedding = self._embed(index_embedding_text)
        self._db.upsert_index_entry(
            source_type=source_type,
            target_table="memory_facts",
            target_id=fact_id,
            index_level="fact",
            memory_path=memory_path,
            title=fact["summary"][:96],
            summary_for_retrieval=summary_card,
            keywords=fact["keywords"],
            entities=fact["entities"],
            canonical_topics=fact["canonical_topics"],
            participants=participants,
            time_start=fact["time_key"].split("#", 1)[0],
            time_end=fact["time_key"].split("#", 1)[0],
            importance=fact["importance"],
            confidence=fact["confidence"],
            embedding=index_embedding,
            embedding_text=index_embedding_text,
            metadata={"tags": tags, **fact["metadata"]},
        )
        return fact_id

    # ── Reflection: facts -> evolving states ─────────────────────────────

    def reflect(self, *_, **kwargs: Any) -> Dict[str, int]:
        """Update topic/entity projections and actionable items from recent facts."""
        if self._pending_store_turns:
            self.flush_pending_store_turns()
        if not self._enabled:
            return {"states_updated": 0, "actionable_items_updated": 0}
        if not self._llm_api_key:
            return {"states_updated": 0, "actionable_items_updated": 0}
        limit = max(1, int(kwargs.get("limit") or self._memory_cfg.get("reflect_limit") or 100))
        reflect_timestamp = kwargs.get("reflect_timestamp")
        if reflect_timestamp is None:
            reflect_timestamp = kwargs.get("timestamp") or _now_text()
        facts = self._db.get_unprocessed_facts_for_states(
            limit=limit,
            reference_timestamp=reflect_timestamp,
        )
        if not facts:
            return {"states_updated": 0, "actionable_items_updated": 0}
        fact_ids = [
            int(fact["id"])
            for fact in facts
            if str(fact.get("id") or "").strip().isdigit()
        ]
        topic_facts = [fact for fact in facts if self._fact_can_seed_topic_state(fact)]
        entity_facts = [fact for fact in facts if self._fact_can_seed_entity_state(fact)]
        existing_states = self._db.recent_states(limit=80)
        topic_report = self._resolve_and_update_topic_states_from_facts(
            facts=topic_facts,
            existing_states=existing_states,
        )
        existing_states = self._db.recent_states(limit=80)
        entity_report = self._resolve_and_update_entity_scoped_states_from_facts(
            facts=entity_facts,
            existing_states=existing_states,
        )
        facts_marked_processed = self._db.mark_facts_processed_for_memory_state(fact_ids)
        actionable_updates = self._extract_actionable_items_with_llm(facts=facts)
        actionable_items_updated = 0
        for item in actionable_updates:
            item_id = self._store_actionable_item(item)
            if item_id:
                actionable_items_updated += 1
        return {
            "states_updated": (
                int(topic_report.get("updated", 0) or 0)
                + int(entity_report.get("updated", 0) or 0)
            ),
            "topic_facts_considered": len(topic_facts),
            "entity_facts_considered": len(entity_facts),
            "evidence_only_facts": max(0, len(facts) - len(set(
                int(fact["id"])
                for fact in [*topic_facts, *entity_facts]
                if str(fact.get("id") or "").strip().isdigit()
            ))),
            "topic_states_updated": int(topic_report.get("updated", 0) or 0),
            "topic_candidates_unresolved": int(topic_report.get("unresolved", 0) or 0),
            "pending_unresolved_topics": int(topic_report.get("pending_unresolved", 0) or 0),
            "entity_states_updated": int(entity_report.get("updated", 0) or 0),
            "facts_marked_processed_for_memory_state": facts_marked_processed,
            "actionable_items_updated": actionable_items_updated,
        }

    @classmethod
    def _fact_has_durable_state_signal(cls, fact: Dict[str, Any]) -> bool:
        """Return whether a fact contains durable state signal, not just an event."""
        kind = str(fact.get("fact_kind") or "").strip().lower()
        fact_type = str(fact.get("fact_type") or "").strip().lower()
        summary = str(fact.get("summary") or "").lower()
        if kind in {"action", "request", "commitment", "other"} and fact_type != "semantic":
            return any(
                marker in summary
                for marker in (
                    "长期", "持续", "反复", "一直", "通常", "习惯", "偏好", "喜欢",
                    "计划", "决定", "策略", "风险", "约束", "长期", "ongoing",
                    "persistent", "usually", "habit", "prefer", "decided", "strategy",
                    "risk", "constraint",
                )
            )
        if fact_type == "semantic":
            return True
        return kind in {
            "preference", "decision", "risk", "error", "open_question",
            "context", "instruction",
        }

    @classmethod
    def _fact_can_seed_topic_state(cls, fact: Dict[str, Any]) -> bool:
        """Topic states require an anchored topic and durable topic signal."""
        topics = cls._coerce_topic_list(fact.get("canonical_topics"))
        return bool(topics) and cls._fact_has_durable_state_signal(fact)

    def _fact_can_seed_entity_state(self, fact: Dict[str, Any]) -> bool:
        """Entity states require both a durable signal and an entity target."""
        if not self._fact_has_durable_state_signal(fact):
            return False
        if not self._infer_entity_scoped_state_types_for_fact(fact):
            return False
        return bool(self._entities_for_entity_state_fact(fact))

    def _resolve_and_update_topic_states_from_facts(
        self,
        *,
        facts: List[Dict[str, Any]],
        existing_states: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        if not self._enable_topic_state_resolution:
            return {"enabled": 0, "updated": 0, "unresolved": 0, "pending_unresolved": len(self._pending_unresolved_topics)}
        existing_topic_states = [
            state for state in existing_states
            if str(state.get("state_scope") or "") == "topic_state"
            and str(state.get("state_type") or "") == "topic"
        ]
        candidates = self._build_topic_state_candidates_from_facts(facts)
        updated = 0
        unresolved = 0
        for candidate in candidates:
            matched_state, match_info = self._match_topic_candidate_to_existing_state(
                candidate=candidate,
                existing_topic_states=existing_topic_states,
            )
            grounded, chosen_state, grounding_info = self._ground_topic_state_candidate(
                candidate=candidate,
                matched_state=matched_state,
                match_info=match_info,
                existing_topic_states=existing_topic_states,
            )
            if not grounded:
                unresolved += 1
                self._remember_pending_unresolved_topic(candidate, grounding_info)
                continue
            if chosen_state:
                candidate["canonical_name"] = str(
                    chosen_state.get("canonical_name")
                    or candidate.get("canonical_name")
                    or "general"
                )
            state_update = self._extract_topic_state_update_with_llm(
                candidate=candidate,
                existing_state=chosen_state,
            )
            if not state_update or not state_update.get("summary"):
                continue
            state_update.setdefault("source_type", candidate.get("source_type") or "unified")
            state_id = self._store_state(state_update)
            if state_id:
                updated += 1
                refreshed = self._db.recent_states(
                    source_types=[state_update["source_type"]],
                    limit=200,
                )
                existing_topic_states = [
                    state for state in refreshed
                    if str(state.get("state_scope") or "") == "topic_state"
                    and str(state.get("state_type") or "") == "topic"
                ]
        return {
            "enabled": 1,
            "candidate_count": len(candidates),
            "updated": updated,
            "unresolved": unresolved,
            "pending_unresolved": len(self._pending_unresolved_topics),
        }

    def _build_topic_state_candidates_from_facts(
        self,
        facts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        by_episode: Dict[str, List[Dict[str, Any]]] = {}
        for fact in facts:
            episode_key = str(fact.get("episode_id") or "unknown")
            by_episode.setdefault(episode_key, []).append(fact)

        candidates: List[Dict[str, Any]] = []
        for episode_id, episode_facts in by_episode.items():
            topic_counts: Counter[str] = Counter()
            topic_original: Dict[str, str] = {}
            for fact in episode_facts:
                raw_topics = self._coerce_topic_list(fact.get("canonical_topics"))
                if not raw_topics:
                    raw_topics = self._topic_candidates(str(fact.get("summary") or ""))
                for raw_topic in raw_topics:
                    topic = self._normalize_topic_name(raw_topic)
                    if not topic:
                        continue
                    key = self._topic_state_key(topic)
                    topic_counts[key] += 1
                    topic_original.setdefault(key, topic)
            if not topic_counts:
                continue
            single_topic = len(topic_counts) == 1
            for topic_key, _count in topic_counts.most_common(self._topic_state_max_topics_per_episode):
                topic_name = topic_original.get(topic_key) or topic_key
                matched_facts = [
                    fact for fact in episode_facts
                    if single_topic or self._fact_matches_topic_name(fact, topic_name)
                ]
                if not matched_facts:
                    matched_facts = list(episode_facts)
                fact_ids = [
                    int(fact["id"])
                    for fact in matched_facts
                    if str(fact.get("id") or "").strip().isdigit()
                ]
                if not fact_ids:
                    continue
                source_type = self._state_source_type_for_facts(matched_facts, fact_ids)
                aliases = self._merge_topic_aliases([
                    topic_name,
                    topic_key,
                    *[
                        topic for fact in matched_facts
                        for topic in self._coerce_topic_list(fact.get("canonical_topics"))
                    ],
                ])
                identity_keywords = self._topic_identity_keywords(matched_facts, limit=12)
                topic_identity_text = self._topic_identity_text(
                    canonical_name=topic_name,
                    aliases=aliases,
                    keywords=identity_keywords,
                )
                candidates.append({
                    "episode_id": episode_id,
                    "topic_key": topic_key,
                    "canonical_name": topic_name,
                    "aliases": aliases,
                    "identity_keywords": identity_keywords,
                    "topic_identity_text": topic_identity_text,
                    "facts": matched_facts,
                    "fact_ids": fact_ids,
                    "source_type": source_type,
                    "summary_text": "\n".join(
                        str(fact.get("summary") or "") for fact in matched_facts
                    )[:2400],
                })
        return candidates

    def _fact_matches_topic_name(self, fact: Dict[str, Any], topic_name: str) -> bool:
        topic_key = self._topic_state_key(topic_name)
        fact_topics = [
            self._topic_state_key(topic)
            for topic in self._coerce_topic_list(fact.get("canonical_topics"))
        ]
        if topic_key in fact_topics:
            return True
        for fact_topic in fact_topics:
            if fact_topic and topic_key and (fact_topic in topic_key or topic_key in fact_topic):
                return True
        summary = str(fact.get("summary") or "")
        return bool(topic_name and topic_name in summary)

    @staticmethod
    def _topic_state_key(value: Any) -> str:
        text = _compact_whitespace(value).lower()
        text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", text)
        return text or "general"

    def _merge_topic_aliases(self, values: Sequence[Any], *, limit: int = 20) -> List[str]:
        aliases: List[str] = []
        seen: set[str] = set()
        for value in values:
            topic = self._normalize_topic_name(value) or _compact_whitespace(value)
            if not topic:
                continue
            key = topic.lower()
            if key in seen:
                continue
            seen.add(key)
            aliases.append(topic)
            if len(aliases) >= limit:
                break
        return aliases

    def _topic_aliases_from_state(self, state: Dict[str, Any]) -> List[str]:
        metadata = state.get("metadata") or {}
        return self._merge_topic_aliases([
            state.get("canonical_name"),
            *(metadata.get("canonical_topics") or []),
            *(metadata.get("topic_aliases") or []),
        ])

    def _topic_identity_keywords(
        self,
        facts: Sequence[Dict[str, Any]],
        *,
        limit: int = 12,
    ) -> List[str]:
        values: List[str] = []
        for fact in facts:
            values.extend(str(fact.get("keywords") or "").split())
            values.extend(str(entity) for entity in (fact.get("entities") or []))
        generic = {
            "用户", "助手", "建议", "认为", "表示", "接受", "拒绝", "尝试",
            "方案", "问题", "工作", "时间", "方法", "讨论", "不现实",
            "user", "assistant", "suggestion", "plan", "issue", "work",
            "time", "method", "discussion",
        }
        out: List[str] = []
        seen: set[str] = set()
        for value in values:
            clean = self._normalize_topic_name(value) or _compact_whitespace(value)
            if not clean:
                continue
            if clean in generic:
                continue
            key = self._topic_state_key(clean)
            if key in seen:
                continue
            seen.add(key)
            out.append(clean)
            if len(out) >= limit:
                break
        return out

    def _topic_identity_text(
        self,
        *,
        canonical_name: Any,
        aliases: Sequence[Any],
        keywords: Sequence[Any],
    ) -> str:
        canonical = self._normalize_topic_name(canonical_name) or _compact_whitespace(canonical_name)
        alias_values = self._merge_topic_aliases([item for item in aliases if item != canonical], limit=8)
        keyword_values = self._merge_topic_aliases(keywords, limit=12)
        return "\n".join([
            f"canonical_topic: {canonical}",
            f"aliases: {', '.join(alias_values)}",
            f"anchor_terms: {', '.join(keyword_values)}",
        ])

    def _topic_identity_text_for_state(self, state: Dict[str, Any]) -> str:
        metadata = state.get("metadata") or {}
        existing_text = _compact_whitespace(metadata.get("topic_identity_text") or "")
        if existing_text:
            return existing_text
        return self._topic_identity_text(
            canonical_name=state.get("canonical_name"),
            aliases=self._topic_aliases_from_state(state),
            keywords=[
                *(metadata.get("keywords") or []),
                *(metadata.get("entities") or []),
            ],
        )

    def _topic_embedding_similarity(self, left_text: str, right_text: str) -> float:
        left_embedding = self._embed(left_text)
        right_embedding = self._embed(right_text)
        if left_embedding is None or right_embedding is None:
            return 0.0
        left = np.asarray(left_embedding, dtype=np.float32).reshape(-1)
        right = np.asarray(right_embedding, dtype=np.float32).reshape(-1)
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom <= 0:
            return 0.0
        return float(np.dot(left, right) / denom)

    def _topic_resolution_text(
        self,
        *,
        candidate: Optional[Dict[str, Any]] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> str:
        parts: List[str] = []
        if candidate:
            parts.extend(candidate.get("aliases") or [])
            parts.append(str(candidate.get("summary_text") or ""))
            for fact in candidate.get("facts") or []:
                parts.append(str(fact.get("summary") or "")[:500])
        if state:
            metadata = state.get("metadata") or {}
            parts.extend(self._topic_aliases_from_state(state))
            parts.append(str(state.get("summary") or ""))
            parts.extend(str(item) for item in metadata.get("canonical_topics") or [])
        return "\n".join(part for part in parts if str(part or "").strip())

    def _match_topic_candidate_to_existing_state(
        self,
        *,
        candidate: Dict[str, Any],
        existing_topic_states: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        if not existing_topic_states:
            return None, {"matched": False, "reason": "no_existing_topic_states"}
        candidate_aliases = candidate.get("aliases") or [candidate.get("canonical_name", "")]
        candidate_keys = {
            self._topic_state_key(alias)
            for alias in candidate_aliases
            if str(alias or "").strip()
        }
        candidate_text = self._topic_resolution_text(candidate=candidate)
        candidate_identity_text = (
            _compact_whitespace(candidate.get("topic_identity_text") or "")
            or self._topic_identity_text(
                canonical_name=candidate.get("canonical_name"),
                aliases=candidate_aliases,
                keywords=candidate.get("identity_keywords") or [],
            )
        )
        best_state: Optional[Dict[str, Any]] = None
        best_info: Dict[str, Any] = {"score": 0.0}
        for state in existing_topic_states:
            state_aliases = self._topic_aliases_from_state(state)
            state_keys = {
                self._topic_state_key(alias)
                for alias in state_aliases
                if str(alias or "").strip()
            }
            exact_alias_match = bool(candidate_keys & state_keys)
            name_overlap = self._topic_name_overlap(candidate_aliases, state_aliases)
            text_overlap = self._topic_name_similarity(
                candidate_text[:1200],
                self._topic_resolution_text(state=state)[:1200],
            )
            embedding_similarity = self._topic_embedding_similarity(
                candidate_identity_text,
                self._topic_identity_text_for_state(state),
            )
            strong_name_match = exact_alias_match or name_overlap >= self._topic_state_resolution_similarity_threshold
            strong_embedding_match = (
                embedding_similarity >= self._topic_identity_embedding_similarity_threshold
                and name_overlap >= 0.25
            )
            matched = strong_name_match or strong_embedding_match
            score = max(
                1.0 if exact_alias_match else 0.0,
                name_overlap,
                embedding_similarity,
            )
            if score > float(best_info.get("score", 0.0)):
                best_state = state
                best_info = {
                    "matched": matched,
                    "reason": "matched_existing_topic_state" if matched else "best_match_below_threshold",
                    "score": round(score, 4),
                    "exact_alias_match": exact_alias_match,
                    "name_overlap": round(name_overlap, 4),
                    "text_overlap": round(text_overlap, 4),
                    "embedding_similarity": round(embedding_similarity, 4),
                    "strong_name_match": strong_name_match,
                    "strong_embedding_match": strong_embedding_match,
                    "existing_state_id": state.get("id"),
                    "existing_canonical_name": state.get("canonical_name"),
                }
        if best_state and best_info.get("matched"):
            return best_state, best_info
        return None, best_info

    def _topic_name_overlap(self, left_aliases: Sequence[str], right_aliases: Sequence[str]) -> float:
        best = 0.0
        for left in left_aliases:
            left_key = self._topic_state_key(left)
            for right in right_aliases:
                right_key = self._topic_state_key(right)
                if left_key and right_key and left_key == right_key:
                    return 1.0
                if left_key and right_key and (left_key in right_key or right_key in left_key):
                    best = max(best, 0.9)
                best = max(best, self._topic_name_similarity(str(left), str(right)))
        return best

    @classmethod
    def _topic_anchor_remainder(cls, text: str) -> str:
        clean = re.sub(r"speaker[_-]?\d+", "", str(text or "").lower())
        generic_terms = [
            "产品", "方案", "策略", "讨论", "活动", "项目", "计划", "协作",
            "落实", "筹备", "管理", "设计", "推广", "促销", "目标", "用户",
            "定位", "成本", "质量", "部门", "会议", "总结", "后续", "安排",
            "执行", "上线", "选择", "方式", "内容", "问题", "建议", "相关",
            "事项", "工作", "阶段", "品牌", "赠品", "供应商", "价格", "折扣",
            "定价", "合作", "沟通", "配合", "跟进", "确认", "时间", "排期",
            "颜色", "外观", "风格", "提议", "参考", "偏好", "使用", "要求",
            "会后", "各", "与", "和", "及", "的", "了",
            "product", "products", "plan", "plans", "strategy", "strategies",
            "discussion", "coordination", "implementation", "execution", "design",
            "activity", "preparation", "management", "topic", "topics", "followup",
            "follow-up", "meeting", "department", "departments", "scheme",
            "solution", "solutions", "project", "projects", "promotion", "pricing",
            "discount", "quality", "cost", "vendor", "vendors", "supplier",
            "suppliers", "launch", "schedule", "scheduling", "and", "or", "the",
            "a", "an", "to", "of", "for", "with",
        ]
        for term in sorted(generic_terms, key=len, reverse=True):
            clean = clean.replace(term, "")
        clean = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", clean)
        return clean.strip()

    def _topic_candidate_has_anchor(self, candidate: Dict[str, Any]) -> bool:
        for value in [candidate.get("canonical_name"), *(candidate.get("aliases") or [])]:
            if len(self._topic_anchor_remainder(str(value or ""))) >= 2:
                return True
        return False

    def _ground_topic_state_candidate(
        self,
        *,
        candidate: Dict[str, Any],
        matched_state: Optional[Dict[str, Any]],
        match_info: Dict[str, Any],
        existing_topic_states: List[Dict[str, Any]],
    ) -> Tuple[bool, Optional[Dict[str, Any]], Dict[str, Any]]:
        if matched_state:
            return True, matched_state, {
                "grounded": True,
                "reason": "matched_existing_topic_state",
                "candidate_has_anchor": self._topic_candidate_has_anchor(candidate),
                "match": match_info,
            }
        candidate_has_anchor = self._topic_candidate_has_anchor(candidate)
        if candidate_has_anchor:
            return True, None, {
                "grounded": True,
                "reason": "candidate_has_concrete_anchor",
                "candidate_has_anchor": True,
                "match": match_info,
            }
        best_state = None
        if (
            existing_topic_states
            and float(match_info.get("embedding_similarity", 0.0) or 0.0) >= self._topic_identity_grounding_similarity_threshold
            and float(match_info.get("name_overlap", 0.0) or 0.0) >= 0.2
        ):
            best_id = match_info.get("existing_state_id")
            for state in existing_topic_states:
                if best_id is not None and int(state.get("id") or -1) == int(best_id):
                    best_state = state
                    break
        if best_state:
            return True, best_state, {
                "grounded": True,
                "reason": "inherited_existing_topic_for_unanchored_candidate",
                "candidate_has_anchor": False,
                "match": match_info,
            }
        return False, None, {
            "grounded": False,
            "reason": "missing_concrete_topic_anchor",
            "candidate_has_anchor": False,
            "match": match_info,
        }

    def _remember_pending_unresolved_topic(
        self,
        candidate: Dict[str, Any],
        grounding_info: Dict[str, Any],
    ) -> None:
        record = {
            "created_at": _now_text(),
            "episode_id": candidate.get("episode_id"),
            "canonical_name": candidate.get("canonical_name"),
            "topic_key": candidate.get("topic_key"),
            "aliases": candidate.get("aliases", []),
            "fact_ids": candidate.get("fact_ids", []),
            "source_type": candidate.get("source_type"),
            "grounding": grounding_info,
        }
        self._pending_unresolved_topics.append(record)
        if len(self._pending_unresolved_topics) > self._pending_unresolved_topic_max:
            overflow = len(self._pending_unresolved_topics) - self._pending_unresolved_topic_max
            del self._pending_unresolved_topics[:overflow]

    def _extract_topic_state_update_with_llm(
        self,
        *,
        candidate: Dict[str, Any],
        existing_state: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        facts = list(candidate.get("facts") or [])
        prompt_language = self._resolve_prompt_language_from_text(
            "\n".join(str(item.get("summary") or "") for item in facts[:12])
        )
        prompt_template = (
            UNIFIED_TOPIC_STATE_UPDATE_PROMPT_EN
            if prompt_language == "en"
            else UNIFIED_TOPIC_STATE_UPDATE_PROMPT_ZH
        )
        prompt = (
            prompt_template
            .replace("{canonical_topic}", json.dumps({
                "canonical_name": candidate.get("canonical_name"),
                "topic_key": candidate.get("topic_key"),
                "aliases": candidate.get("aliases", []),
                "identity_keywords": candidate.get("identity_keywords", []),
                "topic_identity_text": candidate.get("topic_identity_text") or "",
            }, ensure_ascii=False, indent=2))
            .replace("{existing_topic_state}", json.dumps(
                self._format_existing_topic_state_for_prompt(existing_state),
                ensure_ascii=False,
                indent=2,
            ))
            .replace("{facts}", self._format_facts_for_state_prompt(facts))
        )
        result = self._call_llm(prompt)
        parsed = self._parse_json_object_from_llm_text(result or "")
        if parsed:
            if not self._config_bool(parsed.get("update_needed", True), True):
                logger.debug(
                    "LLM declined topic-state update for topic=%s",
                    candidate.get("canonical_name"),
                )
                return None
            normalized = self._normalize_topic_state_update_payload(
                parsed,
                candidate=candidate,
                existing_state=existing_state,
            )
            if normalized:
                return normalized
        return self._fallback_topic_state_update(candidate, existing_state)

    @staticmethod
    def _format_existing_topic_state_for_prompt(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not state:
            return {}
        return {
            "id": state.get("id"),
            "state_scope": state.get("state_scope") or "topic_state",
            "source_type": state.get("source_type"),
            "canonical_name": state.get("canonical_name"),
            "summary": state.get("summary"),
            "time_line": MemoryNodeManager._normalize_time_line(
                state.get("time_line"),
                limit=8,
                max_chars=1000,
            ),
            "evidence_fact_ids": state.get("evidence_fact_ids") or [],
            "confidence": state.get("confidence"),
            "metadata": state.get("metadata") or {},
        }

    @staticmethod
    def _normalize_state_summary(value: Any, *, max_chars: int = 280) -> str:
        """Keep state summaries as short current snapshots, not history logs."""
        text = _compact_whitespace(value)
        if len(text) <= max_chars:
            return text
        boundary = max(
            text.rfind("。", 0, max_chars),
            text.rfind("！", 0, max_chars),
            text.rfind("？", 0, max_chars),
            text.rfind(".", 0, max_chars),
            text.rfind("!", 0, max_chars),
            text.rfind("?", 0, max_chars),
        )
        if boundary >= max_chars // 2:
            return text[: boundary + 1]
        return text[:max_chars].rstrip("，,；; ") + "..."

    @staticmethod
    def _normalize_time_line(
        value: Any,
        *,
        limit: int = 20,
        max_chars: int = 2400,
        valid_fact_ids: Optional[set[int]] = None,
    ) -> List[Dict[str, Any]]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                value = []
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            return []
        events: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str, Tuple[int, ...]]] = set()
        for raw in value:
            if not isinstance(raw, dict):
                continue
            summary = MemoryNodeManager._normalize_state_summary(
                raw.get("summary") or raw.get("text"),
                max_chars=180,
            )
            if not summary:
                continue
            fact_ids: List[int] = []
            for fact_id in raw.get("fact_ids") or raw.get("evidence_fact_ids") or []:
                if not str(fact_id).strip().isdigit():
                    continue
                normalized_id = int(fact_id)
                if valid_fact_ids is None or normalized_id in valid_fact_ids:
                    fact_ids.append(normalized_id)
            fact_ids = list(dict.fromkeys(fact_ids))[:12]
            occurred_at = _compact_whitespace(
                raw.get("occurred_at")
                or raw.get("timestamp")
                or raw.get("time")
                or ""
            )[:80]
            change_type = _compact_whitespace(
                raw.get("change_type") or raw.get("type") or "updated"
            )[:40]
            event = {
                "occurred_at": occurred_at,
                "change_type": change_type,
                "summary": summary,
                "fact_ids": fact_ids,
            }
            key = (occurred_at, change_type, summary, tuple(fact_ids))
            if key in seen:
                continue
            seen.add(key)
            events.append(event)
        events = events[-max(1, int(limit or 20)):]
        while events and len(json.dumps(events, ensure_ascii=False)) > max_chars:
            events.pop(0)
        return events

    def _fallback_time_line_update(
        self,
        candidate: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        facts = [
            fact for fact in candidate.get("facts") or []
            if _compact_whitespace(fact.get("summary") or "")
        ]
        if not facts:
            return []
        fact_ids = [
            int(fact["id"])
            for fact in facts
            if str(fact.get("id") or "").strip().isdigit()
        ]
        latest = facts[-1]
        time_key = _compact_whitespace(latest.get("time_key") or "")
        occurred_at = time_key.split("#", 1)[0] if time_key else ""
        return [{
            "occurred_at": occurred_at,
            "change_type": "updated",
            "summary": self._normalize_state_summary(
                latest.get("summary") or "",
                max_chars=120,
            ),
            "fact_ids": fact_ids[-12:],
        }]

    def _build_state_time_line(
        self,
        *,
        raw_updates: Any,
        candidate: Dict[str, Any],
        existing_state: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        valid_fact_ids = {
            int(fact["id"])
            for fact in candidate.get("facts") or []
            if str(fact.get("id") or "").strip().isdigit()
        }
        updates = raw_updates
        if not updates:
            updates = self._fallback_time_line_update(candidate)
        existing_events = self._normalize_time_line(
            (existing_state or {}).get("time_line"),
            limit=20,
            max_chars=2400,
        )
        update_events = self._normalize_time_line(
            updates,
            limit=20,
            max_chars=2400,
            valid_fact_ids=valid_fact_ids,
        )
        return self._normalize_time_line(
            [
                *existing_events,
                *update_events,
            ],
            limit=20,
            max_chars=2400,
        )

    def _normalize_topic_state_update_payload(
        self,
        raw: Dict[str, Any],
        *,
        candidate: Dict[str, Any],
        existing_state: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not self._config_bool(raw.get("update_needed", True), True):
            return None
        summary = self._normalize_state_summary(raw.get("summary") or "", max_chars=120)
        if not summary:
            return None
        valid_fact_ids = {
            int(fact["id"])
            for fact in candidate.get("facts", [])
            if str(fact.get("id") or "").strip().isdigit()
        }
        evidence_ids = [
            int(value)
            for value in (raw.get("evidence_fact_ids") or [])
            if str(value).strip().isdigit() and int(value) in valid_fact_ids
        ]
        if not evidence_ids:
            evidence_ids = list(candidate.get("fact_ids") or [])[:24]
        existing_ids = [
            int(value)
            for value in ((existing_state or {}).get("evidence_fact_ids") or [])
            if str(value).strip().isdigit()
        ]
        evidence_ids = list(dict.fromkeys([*existing_ids, *evidence_ids]))[:80]
        canonical_name = (
            _compact_whitespace((existing_state or {}).get("canonical_name") or "")
            or self._normalize_topic_name(raw.get("canonical_name"))
            or _compact_whitespace(candidate.get("canonical_name") or "")
            or "general"
        )
        canonical_topics = self._merge_topic_aliases([
            canonical_name,
            *(raw.get("canonical_topics") or []),
            *(candidate.get("aliases") or []),
        ], limit=8)
        time_line = self._build_state_time_line(
            raw_updates=raw.get("time_line_updates") or raw.get("time_line"),
            candidate=candidate,
            existing_state=existing_state,
        )
        return {
            "state_scope": "topic_state",
            "state_type": "topic",
            "source_type": candidate.get("source_type") or (existing_state or {}).get("source_type") or "unified",
            "canonical_name": canonical_name,
            "summary": summary,
            "time_line": time_line,
            "evidence_fact_ids": evidence_ids,
            "keywords": self._normalize_string_list(raw.get("keywords"), limit=18),
            "entities": self._normalize_string_list(raw.get("entities"), limit=18),
            "canonical_topics": canonical_topics or [canonical_name],
            "importance": self._clamp_float(raw.get("importance"), 0.0, 1.0, 0.7),
            "confidence": self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.75),
            "status": _compact_whitespace(raw.get("status") or "active") or "active",
            "metadata": {
                "topic_key": candidate.get("topic_key"),
                "topic_aliases": candidate.get("aliases", []),
                "topic_identity_text": candidate.get("topic_identity_text") or "",
                "identity_keywords": candidate.get("identity_keywords", []),
                "episode_id": candidate.get("episode_id"),
            },
        }

    def _fallback_topic_state_update(
        self,
        candidate: Dict[str, Any],
        existing_state: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        fact_summaries = [
            _compact_whitespace(fact.get("summary") or "")
            for fact in candidate.get("facts", [])
            if _compact_whitespace(fact.get("summary") or "")
        ][:5]
        base = _compact_whitespace((existing_state or {}).get("summary") or "")
        update_text = "；".join(fact_summaries)
        summary_source = (
            f"{base}；最新变化：{update_text}"
            if base and update_text
            else update_text or base or _compact_whitespace(candidate.get("summary_text") or "")
        )
        summary = self._normalize_state_summary(summary_source, max_chars=120)
        existing_ids = [
            int(value)
            for value in ((existing_state or {}).get("evidence_fact_ids") or [])
            if str(value).strip().isdigit()
        ]
        evidence_ids = list(dict.fromkeys([*existing_ids, *(candidate.get("fact_ids") or [])]))[:80]
        canonical_name = (
            _compact_whitespace((existing_state or {}).get("canonical_name") or "")
            or _compact_whitespace(candidate.get("canonical_name") or "")
            or "general"
        )
        return {
            "state_scope": "topic_state",
            "state_type": "topic",
            "source_type": candidate.get("source_type") or (existing_state or {}).get("source_type") or "unified",
            "canonical_name": canonical_name,
            "summary": summary,
            "time_line": self._build_state_time_line(
                raw_updates=None,
                candidate=candidate,
                existing_state=existing_state,
            ),
            "evidence_fact_ids": evidence_ids,
            "keywords": self._keywords(summary, limit=18),
            "entities": self._entities(summary),
            "canonical_topics": self._merge_topic_aliases([canonical_name, *(candidate.get("aliases") or [])], limit=8),
            "importance": 0.7,
            "confidence": 0.58,
            "status": "active",
            "metadata": {
                "topic_key": candidate.get("topic_key"),
                "topic_aliases": candidate.get("aliases", []),
                "topic_identity_text": candidate.get("topic_identity_text") or "",
                "identity_keywords": candidate.get("identity_keywords", []),
                "episode_id": candidate.get("episode_id"),
                "extractor": "fallback_topic_state_update",
            },
        }

    def _resolve_and_update_entity_scoped_states_from_facts(
        self,
        *,
        facts: List[Dict[str, Any]],
        existing_states: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        if not self._enable_entity_scoped_state_resolution:
            return {"enabled": 0, "updated": 0}
        existing_entity_states = [
            state for state in existing_states
            if str(state.get("state_scope") or "") == "entity_state"
            and str(state.get("state_type") or "") in self._entity_scoped_state_types()
        ]
        candidates = self._build_entity_state_candidates_from_facts(facts)
        updated = 0
        for candidate in candidates:
            existing_state, match_info = self._match_entity_state_candidate(
                candidate=candidate,
                existing_entity_states=existing_entity_states,
            )
            if existing_state:
                candidate["canonical_name"] = str(
                    existing_state.get("canonical_name")
                    or candidate.get("canonical_name")
                    or "general"
                )
            state_update = self._extract_entity_state_update_with_llm(
                candidate=candidate,
                existing_state=existing_state,
                match_info=match_info,
            )
            if not state_update or not state_update.get("summary"):
                continue
            state_id = self._store_state(state_update)
            if state_id:
                updated += 1
                refreshed = self._db.recent_states(
                    source_types=[state_update["source_type"]],
                    limit=200,
                )
                existing_entity_states = [
                    state for state in refreshed
                    if str(state.get("state_scope") or "") == "entity_state"
                    and str(state.get("state_type") or "") in self._entity_scoped_state_types()
                ]
        return {"enabled": 1, "candidate_count": len(candidates), "updated": updated}

    @staticmethod
    def _entity_scoped_state_types() -> set[str]:
        return {"preference", "profile", "routine", "relationship", "constraint", "risk"}

    def _build_entity_state_candidates_from_facts(
        self,
        facts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        grouped: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        for fact in facts:
            state_types = self._infer_entity_scoped_state_types_for_fact(fact)
            if not state_types:
                continue
            source_type = str(fact.get("source_type") or "unified")
            for state_type in state_types:
                entities = self._entities_for_entity_state_fact(
                    fact,
                    state_type=state_type,
                )
                if not entities:
                    continue
                attribute_topics = self._entity_state_attribute_topics(fact)
                for entity in entities[:1]:
                    entity_key = self._entity_state_key(entity)
                    for attribute_name in attribute_topics:
                        attribute_key = self._topic_state_key(attribute_name)
                        key = (source_type, state_type, entity_key, attribute_key)
                        canonical_name = self._entity_state_candidate_name(
                            attribute_name=attribute_name,
                        )
                        item = grouped.setdefault(key, {
                            "source_type": source_type,
                            "state_scope": "entity_state",
                            "state_type": state_type,
                            "entity": entity,
                            "entity_key": entity_key,
                            "canonical_name": canonical_name,
                            "attribute_key": attribute_key,
                            "attribute_name": attribute_name,
                            "attribute_aliases": [attribute_name],
                            "facts": [],
                            "fact_ids": [],
                            "aliases": self._merge_topic_aliases([entity, attribute_name], limit=10),
                        })
                        item["attribute_aliases"] = self._merge_topic_aliases([
                            *(item.get("attribute_aliases") or []),
                            *self._coerce_topic_list(fact.get("canonical_topics")),
                        ], limit=10)
                        item["facts"].append(fact)
                        if str(fact.get("id") or "").strip().isdigit():
                            item["fact_ids"].append(int(fact["id"]))
        candidates: List[Dict[str, Any]] = []
        for item in grouped.values():
            item["fact_ids"] = list(dict.fromkeys(item.get("fact_ids") or []))
            if not item["fact_ids"]:
                continue
            item["summary_text"] = "\n".join(
                str(fact.get("summary") or "") for fact in item.get("facts") or []
            )[:2400]
            item["attribute_text"] = "\n".join([
                str(item.get("attribute_name") or ""),
                " ".join(item.get("attribute_aliases") or []),
                item["summary_text"],
            ])[:2800]
            candidates.append(item)
        return candidates

    def _entity_state_attribute_topics(self, fact: Dict[str, Any]) -> List[str]:
        topics = [
            self._normalize_topic_name(topic)
            for topic in self._coerce_topic_list(fact.get("canonical_topics"))
        ]
        topics = list(dict.fromkeys(topic for topic in topics if topic))
        if not topics:
            fallback = self._topic_candidates(str(fact.get("summary") or ""))
            topics = [self._normalize_topic_name(topic) for topic in fallback if topic]
        return topics[:3] or ["general"]

    def _infer_entity_scoped_state_types_for_fact(self, fact: Dict[str, Any]) -> List[str]:
        kind = str(fact.get("fact_kind") or "").strip().lower()
        subject = str(fact.get("fact_subject") or "").strip().lower()
        summary = str(fact.get("summary") or "").lower()
        state_types: List[str] = []
        if kind == "preference" or any(marker in summary for marker in ("偏好", "喜欢", "倾向", "更愿意", "prefer", "likes", "tends to")):
            state_types.append("preference")
        if any(marker in summary for marker in ("习惯", "经常", "通常", "每天", "每周", "routine", "usually", "often")):
            state_types.append("routine")
        if kind in {"context", "instruction"} and subject in {"user", "project", "world", "other"}:
            state_types.append("profile")
        if kind == "risk" or any(marker in summary for marker in ("约束", "限制", "不能", "预算", "deadline", "constraint", "limited", "budget")):
            state_types.append("constraint")
        if kind == "risk" or any(marker in summary for marker in ("风险", "担心", "隐患", "risk", "concern", "hazard")):
            state_types.append("risk")
        if any(marker in summary for marker in ("关系", "同事", "朋友", "家人", "老板", "客户", "relationship", "colleague", "friend", "family", "client")):
            state_types.append("relationship")
        return [item for item in dict.fromkeys(state_types) if item in self._entity_scoped_state_types()]

    def _entities_for_entity_state_fact(
        self,
        fact: Dict[str, Any],
        *,
        state_type: Optional[str] = None,
    ) -> List[str]:
        metadata = fact.get("metadata") if isinstance(fact.get("metadata"), dict) else {}
        primary_entity = fact.get("primary_entity") or metadata.get("primary_entity")
        if isinstance(primary_entity, dict):
            primary_name = _compact_whitespace(
                primary_entity.get("name") or primary_entity.get("text") or ""
            )
        else:
            primary_name = _compact_whitespace(primary_entity)
        if primary_name:
            normalized_primary = primary_name.lower()
            if normalized_primary in {"用户", "user", "the user"}:
                return ["user"]
            if normalized_primary in {"助手", "assistant", "agent", "the assistant"}:
                return ["assistant"]
            return [primary_name]

        entities = [
            _compact_whitespace(value)
            for value in (fact.get("entities") or [])
            if _compact_whitespace(value)
        ]
        subject = str(fact.get("fact_subject") or "").strip().lower()
        if subject in {"user", "assistant"}:
            # A fact without an explicit primary_entity still belongs to one
            # speaker; never fan it out to every mentioned entity.
            return [subject]
        if not entities and subject in {"project", "world", "other"}:
            topics = self._coerce_topic_list(fact.get("canonical_topics"))
            entities.extend(self._normalize_topic_name(topic) for topic in topics)
        out: List[str] = []
        seen: set[str] = set()
        for entity in entities:
            clean = _compact_whitespace(entity)
            if not clean:
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(clean)
        return out[:1]

    @staticmethod
    def _entity_state_key(value: Any) -> str:
        return _compact_whitespace(value).lower()

    @staticmethod
    def _entity_state_candidate_name(
        *,
        attribute_name: str,
    ) -> str:
        return _compact_whitespace(attribute_name) or "general"

    def _match_entity_state_candidate(
        self,
        *,
        candidate: Dict[str, Any],
        existing_entity_states: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        candidate_name = str(candidate.get("canonical_name") or "")
        candidate_entity = str(candidate.get("entity") or "")
        candidate_type = str(candidate.get("state_type") or "")
        candidate_entity_key = self._entity_state_key(candidate.get("entity_key") or candidate_entity)
        candidate_attribute_aliases = self._merge_topic_aliases([
            candidate.get("attribute_name"),
            *(candidate.get("attribute_aliases") or []),
        ], limit=12)
        candidate_text = str(candidate.get("attribute_text") or candidate.get("summary_text") or "")
        best_state: Optional[Dict[str, Any]] = None
        best_score = 0.0
        best_info: Dict[str, Any] = {"matched": False, "score": 0.0}
        for state in existing_entity_states:
            if str(state.get("source_type") or "") != str(candidate.get("source_type") or ""):
                continue
            if str(state.get("state_scope") or "") != "entity_state":
                continue
            if str(state.get("state_type") or "") != candidate_type:
                continue
            metadata = state.get("metadata") or {}
            state_entity_key = self._entity_state_key(
                metadata.get("entity_key")
                or metadata.get("entity")
                or candidate_entity
            )
            if state_entity_key != candidate_entity_key:
                continue
            state_attribute_aliases = self._merge_topic_aliases([
                metadata.get("attribute_name"),
                *(metadata.get("attribute_aliases") or []),
                *(metadata.get("canonical_topics") or []),
                state.get("canonical_name"),
            ], limit=16)
            attribute_overlap = self._topic_name_overlap(
                candidate_attribute_aliases,
                state_attribute_aliases,
            )
            state_text = _compact_whitespace(
                metadata.get("entity_state_identity_text")
                or "\n".join([
                    str(state.get("canonical_name") or ""),
                    " ".join(state_attribute_aliases),
                    str(state.get("summary") or ""),
                ])
            )
            embedding_similarity = self._topic_embedding_similarity(
                candidate_text[:1600],
                state_text[:1600],
            )
            name_similarity = self._topic_name_similarity(
                candidate_name,
                str(state.get("canonical_name") or ""),
            )
            exact_attribute_match = any(
                self._topic_state_key(left) == self._topic_state_key(right)
                for left in candidate_attribute_aliases
                for right in state_attribute_aliases
                if left and right
            )
            matched = (
                exact_attribute_match
                or attribute_overlap >= self._entity_state_attribute_similarity_threshold
                or (
                    embedding_similarity >= self._entity_state_resolution_similarity_threshold
                    and attribute_overlap >= 0.2
                )
            )
            score = max(
                1.0 if exact_attribute_match else 0.0,
                attribute_overlap,
                embedding_similarity,
                name_similarity * 0.8,
            )
            if score > best_score:
                best_score = score
                best_state = state
                best_info = {
                    "matched": matched,
                    "score": round(score, 4),
                    "attribute_overlap": round(attribute_overlap, 4),
                    "embedding_similarity": round(embedding_similarity, 4),
                    "exact_attribute_match": exact_attribute_match,
                    "existing_state_id": state.get("id"),
                    "existing_canonical_name": state.get("canonical_name"),
                }
        if best_state and best_info.get("matched"):
            return best_state, {
                "matched": True,
                "score": round(best_score, 4),
                "existing_state_id": best_state.get("id"),
                "existing_canonical_name": best_state.get("canonical_name"),
                "attribute_overlap": best_info.get("attribute_overlap", 0.0),
                "embedding_similarity": best_info.get("embedding_similarity", 0.0),
            }
        return None, best_info

    def _extract_entity_state_update_with_llm(
        self,
        *,
        candidate: Dict[str, Any],
        existing_state: Optional[Dict[str, Any]],
        match_info: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        facts = list(candidate.get("facts") or [])
        prompt_language = self._resolve_prompt_language_from_text(
            "\n".join(str(item.get("summary") or "") for item in facts[:12])
        )
        prompt_template = (
            UNIFIED_ENTITY_STATE_UPDATE_PROMPT_EN
            if prompt_language == "en"
            else UNIFIED_ENTITY_STATE_UPDATE_PROMPT_ZH
        )
        prompt = (
            prompt_template
            .replace("{entity_state_target}", json.dumps({
                "entity": candidate.get("entity"),
                "entity_key": candidate.get("entity_key"),
                "state_type": candidate.get("state_type"),
                "canonical_name_hint": candidate.get("canonical_name"),
                "attribute_name": candidate.get("attribute_name"),
                "attribute_key": candidate.get("attribute_key"),
                "attribute_aliases": candidate.get("attribute_aliases", []),
                "aliases": candidate.get("aliases", []),
            }, ensure_ascii=False, indent=2))
            .replace("{existing_entity_state}", json.dumps(
                self._format_existing_topic_state_for_prompt(existing_state),
                ensure_ascii=False,
                indent=2,
            ))
            .replace("{facts}", self._format_facts_for_state_prompt(facts))
        )
        result = self._call_llm(prompt)
        parsed = self._parse_json_object_from_llm_text(result or "")
        if parsed:
            if not self._config_bool(parsed.get("update_needed", True), True):
                logger.debug(
                    "LLM declined entity-state update for entity=%s attribute=%s",
                    candidate.get("entity"),
                    candidate.get("attribute_name"),
                )
                return None
            normalized = self._normalize_entity_state_update_payload(
                parsed,
                candidate=candidate,
                existing_state=existing_state,
                match_info=match_info,
            )
            if normalized:
                return normalized
        return self._fallback_entity_state_update(candidate, existing_state, match_info)

    def _normalize_entity_state_update_payload(
        self,
        raw: Dict[str, Any],
        *,
        candidate: Dict[str, Any],
        existing_state: Optional[Dict[str, Any]],
        match_info: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self._config_bool(raw.get("update_needed", True), True):
            return None
        summary = self._normalize_state_summary(raw.get("summary") or "", max_chars=120)
        if not summary:
            return None
        valid_fact_ids = {
            int(fact["id"])
            for fact in candidate.get("facts", [])
            if str(fact.get("id") or "").strip().isdigit()
        }
        evidence_ids = [
            int(value)
            for value in (raw.get("evidence_fact_ids") or [])
            if str(value).strip().isdigit() and int(value) in valid_fact_ids
        ] or list(candidate.get("fact_ids") or [])[:24]
        existing_ids = [
            int(value)
            for value in ((existing_state or {}).get("evidence_fact_ids") or [])
            if str(value).strip().isdigit()
        ]
        evidence_ids = list(dict.fromkeys([*existing_ids, *evidence_ids]))[:80]
        canonical_name = (
            _compact_whitespace(candidate.get("attribute_name") or "")
            or _compact_whitespace(raw.get("canonical_name") or "")
            or _compact_whitespace((existing_state or {}).get("canonical_name") or "")
        )
        if canonical_name.lower() in self._entity_scoped_state_types() or len(canonical_name) < 3:
            canonical_name = _compact_whitespace(candidate.get("canonical_name") or "")
        canonical_topics = self._merge_topic_aliases(
            [
                *(raw.get("canonical_topics") or []),
                candidate.get("attribute_name"),
                *(candidate.get("attribute_aliases") or []),
            ],
            limit=8,
        )
        time_line = self._build_state_time_line(
            raw_updates=raw.get("time_line_updates") or raw.get("time_line"),
            candidate=candidate,
            existing_state=existing_state,
        )
        return {
            "state_scope": "entity_state",
            "state_type": candidate["state_type"],
            "source_type": candidate.get("source_type") or (existing_state or {}).get("source_type") or "unified",
            "canonical_name": canonical_name,
            "summary": summary,
            "time_line": time_line,
            "evidence_fact_ids": evidence_ids,
            "keywords": self._normalize_string_list(raw.get("keywords"), limit=18),
            "entities": self._merge_topic_aliases([
                candidate.get("entity"),
                *(raw.get("entities") or []),
            ], limit=18),
            "canonical_topics": canonical_topics,
            "importance": self._clamp_float(raw.get("importance"), 0.0, 1.0, 0.68),
            "confidence": self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.74),
            "status": _compact_whitespace(raw.get("status") or "active") or "active",
            "metadata": {
                "entity": candidate.get("entity"),
                "entity_key": candidate.get("entity_key"),
                "entity_aliases": candidate.get("aliases", []),
                "attribute_key": candidate.get("attribute_key"),
                "attribute_name": candidate.get("attribute_name"),
                "attribute_aliases": candidate.get("attribute_aliases", []),
                "entity_state_identity_text": candidate.get("attribute_text") or "",
                "entity_state_resolution": match_info,
                "extractor": "entity_scoped_state_update",
            },
        }

    def _fallback_entity_state_update(
        self,
        candidate: Dict[str, Any],
        existing_state: Optional[Dict[str, Any]],
        match_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        summaries = [
            _compact_whitespace(fact.get("summary") or "")
            for fact in candidate.get("facts", [])
            if _compact_whitespace(fact.get("summary") or "")
        ][:5]
        base = _compact_whitespace((existing_state or {}).get("summary") or "")
        update_text = "；".join(summaries)
        summary_source = (
            f"{base}；最新变化：{update_text}"
            if base and update_text
            else update_text or base or _compact_whitespace(candidate.get("summary_text") or "")
        )
        summary = self._normalize_state_summary(summary_source, max_chars=120)
        existing_ids = [
            int(value)
            for value in ((existing_state or {}).get("evidence_fact_ids") or [])
            if str(value).strip().isdigit()
        ]
        evidence_ids = list(dict.fromkeys([*existing_ids, *(candidate.get("fact_ids") or [])]))[:80]
        return {
            "state_scope": "entity_state",
            "state_type": candidate["state_type"],
            "source_type": candidate.get("source_type") or (existing_state or {}).get("source_type") or "unified",
            "canonical_name": _compact_whitespace(
                candidate.get("attribute_name")
                or candidate.get("canonical_name")
                or (existing_state or {}).get("canonical_name")
                or "general"
            ),
            "summary": summary,
            "time_line": self._build_state_time_line(
                raw_updates=None,
                candidate=candidate,
                existing_state=existing_state,
            ),
            "evidence_fact_ids": evidence_ids,
            "keywords": self._keywords(summary, limit=18),
            "entities": [candidate.get("entity")] if candidate.get("entity") else [],
            "canonical_topics": self._merge_topic_aliases([
                candidate.get("attribute_name"),
                *(candidate.get("attribute_aliases") or []),
            ], limit=8),
            "importance": 0.66,
            "confidence": 0.58,
            "status": "active",
            "metadata": {
                "entity": candidate.get("entity"),
                "entity_key": candidate.get("entity_key"),
                "entity_aliases": candidate.get("aliases", []),
                "attribute_key": candidate.get("attribute_key"),
                "attribute_name": candidate.get("attribute_name"),
                "attribute_aliases": candidate.get("attribute_aliases", []),
                "entity_state_identity_text": candidate.get("attribute_text") or "",
                "entity_state_resolution": match_info,
                "extractor": "fallback_entity_scoped_state_update",
            },
        }

    def _store_state(self, state: Dict[str, Any]) -> int:
        state_scope = self._normalize_state_scope(
            state.get("state_scope"),
            state.get("state_type"),
        )
        state_type = self._normalize_state_type(state.get("state_type"))
        if not state_scope or not state_type:
            logger.debug("Skipping state with invalid scope/type: %s", state)
            return 0
        if state_scope == "topic_state" and state_type != "topic":
            logger.debug("Skipping topic state with non-topic type: %s", state)
            return 0
        if state_scope == "entity_state" and state_type not in self._entity_scoped_state_types():
            logger.debug("Skipping entity state with invalid type: %s", state)
            return 0
        keywords = state.get("keywords") or self._keywords(state["summary"], limit=18)
        entities = state.get("entities") or self._entities(state["summary"])
        canonical_topics = state.get("canonical_topics") or [state["canonical_name"]]
        evidence_fact_ids = [int(value) for value in state.get("evidence_fact_ids") or []]
        state_metadata = dict(state.get("metadata") or {})
        entity_key = ""
        if state_scope == "entity_state":
            entity_key = self._entity_state_key(
                state_metadata.get("entity_key")
                or state_metadata.get("entity")
                or ""
            )
        embedding_text = "\n".join([
            state["canonical_name"],
            state["summary"],
            f"state_scope: {state_scope}",
            f"state_type: {state_type}",
            f"keywords: {' '.join(keywords)}",
            f"entities: {', '.join(entities)}",
        ])
        embedding = self._embed(embedding_text)
        state_id = self._db.upsert_state(
            state_scope=state_scope,
            state_type=state_type,
            source_type=state["source_type"],
            entity_key=entity_key,
            canonical_name=state["canonical_name"],
            summary=state["summary"],
            time_line=state.get("time_line") or [],
            evidence_fact_ids=evidence_fact_ids,
            confidence=state["confidence"],
            metadata={
                **state_metadata,
                "entity_key": entity_key,
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
            memory_path = f"{state['source_type']}/states/{state_scope}/{state_type}"
            # State summaries are maintained as concise current snapshots. Reuse
            # the same text in the index so retrieval sees exactly that snapshot.
            summary_card = _compact_whitespace(state["summary"])
            evidence_time_start, evidence_time_end = self._event_time_bounds_from_facts(
                self._db.memory_facts_by_ids(evidence_fact_ids),
            )
            index_embedding_text = self._build_index_embedding_text(
                title=state["canonical_name"],
                summary=summary_card,
                keywords=keywords,
                entities=entities,
                canonical_topics=canonical_topics,
                memory_path=memory_path,
            )
            index_embedding = self._embed(index_embedding_text)
            self._db.upsert_index_entry(
                source_type=state["source_type"],
                target_table="memory_states",
                target_id=state_id,
                index_level="state",
                memory_path=memory_path,
                title=state["canonical_name"],
                summary_for_retrieval=summary_card,
                keywords=" ".join(keywords),
                entities=entities,
                canonical_topics=canonical_topics,
                participants=["user", "assistant"],
                time_start=evidence_time_start,
                time_end=evidence_time_end,
                importance=state["importance"],
                confidence=state["confidence"],
                embedding=index_embedding,
                embedding_text=index_embedding_text,
                metadata={
                    "state_scope": state_scope,
                    "state_type": state_type,
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
            memory_path = f"{item['source_type']}/actionable_items/{item['item_type']}"
            summary_card = self._truncate_index_text(item["summary"], max_chars=700)
            evidence_time_start, evidence_time_end = self._event_time_bounds_from_facts(
                self._db.memory_facts_by_ids(evidence_fact_ids),
            )
            index_embedding_text = self._build_index_embedding_text(
                title=item["canonical_name"],
                summary=summary_card,
                keywords=keywords,
                entities=entities,
                canonical_topics=canonical_topics,
                memory_path=memory_path,
            )
            index_embedding = self._embed(index_embedding_text)
            self._db.upsert_index_entry(
                source_type=item["source_type"],
                target_table="memory_actionable_items",
                target_id=item_id,
                index_level="actionable_item",
                memory_path=memory_path,
                title=item["canonical_name"],
                summary_for_retrieval=summary_card,
                keywords=" ".join(keywords),
                entities=entities,
                canonical_topics=canonical_topics,
                participants=["user", "assistant"],
                time_start=evidence_time_start,
                time_end=evidence_time_end,
                importance=item["importance"],
                confidence=item["confidence"],
                embedding=index_embedding,
                embedding_text=index_embedding_text,
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
                "state_scope": state.get("state_scope"),
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
            metadata = fact.get("metadata") if isinstance(fact.get("metadata"), dict) else {}
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
                "primary_entity": fact.get("primary_entity") or metadata.get("primary_entity"),
                "canonical_topics": fact.get("canonical_topics") or [],
            })
        return json.dumps(rows, ensure_ascii=False, indent=2)

    @staticmethod
    def _normalize_state_scope(value: Any, state_type: Any = None) -> str:
        text = str(value or "").strip().lower()
        if text in {"topic_state", "topic"}:
            return "topic_state"
        if text in {"entity_state", "entity"}:
            return "entity_state"
        if str(state_type or "").strip().lower() in {"topic_state", "topic"}:
            return "topic_state"
        if str(state_type or "").strip().lower() in {
            "preference", "profile", "routine", "relationship", "constraint", "risk",
        }:
            return "entity_state"
        return ""

    @staticmethod
    def _normalize_state_type(value: Any) -> str:
        text = str(value or "").strip().lower()
        allowed = {
            "topic", "preference", "profile", "routine", "relationship", "constraint", "risk",
        }
        if text == "topic_state":
            return "topic"
        return text if text in allowed else ""

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

    @classmethod
    def _log_info(cls, scope: str, event: str, payload: Dict[str, Any]) -> None:
        record = {
            "scope": scope,
            "event": event,
            "payload": payload,
        }
        try:
            body = json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=False,
                indent=2,
                default=str,
            )
        except (TypeError, ValueError):
            body = json.dumps(
                {
                    "scope": scope,
                    "event": event,
                    "payload": str(payload),
                },
                ensure_ascii=False,
                sort_keys=False,
                indent=2,
            )
        logger.info("\n%s", body)

    @staticmethod
    def _log_text(value: Any, *, limit: int = 500) -> str:
        text = _compact_whitespace(value)
        if limit <= 0 or len(text) <= limit:
            return text
        return text[:limit] + "...[truncated]"

    @staticmethod
    def _normalize_recall_time_bound(value: Any, *, default_to_now: bool = False) -> str:
        if isinstance(value, datetime):
            return value.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
        text = str(value or "").strip()
        if not text:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S") if default_to_now else ""
        normalized = text.replace("T", " ")
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(normalized).replace(tzinfo=None).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except ValueError:
            pass
        match = re.search(r"(\d{4}-\d{1,2}-\d{1,2})\s+(\d{1,2}:\d{1,2}:\d{1,2})", normalized)
        if match:
            try:
                return datetime.strptime(
                    f"{match.group(1)} {match.group(2)}",
                    "%Y-%m-%d %H:%M:%S",
                ).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
        match = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", normalized)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y-%m-%d").strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            except ValueError:
                pass
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S") if default_to_now else text

    def _recall_log_candidate_items(
        self,
        items: Sequence[Dict[str, Any]],
        *,
        limit: int = 12,
    ) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        for item in list(items or [])[:limit]:
            raw = item.get("_hydrated") if isinstance(item.get("_hydrated"), dict) else {}
            summary = (
                raw.get("summary")
                or item.get("summary_for_retrieval")
                or item.get("summary")
                or item.get("title")
                or ""
            )
            support_ids: List[int] = []
            for support_fact in item.get("_supporting_facts") or []:
                try:
                    support_ids.append(int(support_fact.get("id")))
                except (TypeError, ValueError):
                    continue
            rows.append({
                "target": f"{item.get('target_table')}#{item.get('target_id')}",
                "level": item.get("index_level") or item.get("_recall_type"),
                "source_type": item.get("source_type"),
                "score": item.get("_recall_score"),
                "type_score": item.get("_recall_type_score"),
                "rank": item.get("_recall_rank"),
                "embedding_similarity": item.get("embedding_similarity"),
                "time_start": item.get("time_start"),
                "summary": self._log_text(summary, limit=240),
                "support_fact_ids": support_ids,
            })
        return {
            "count": len(items or []),
            "items": rows,
        }

    @staticmethod
    def _parse_time_expression(
        query: str,
        *,
        reference_time: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str], str]:
        """Parse lightweight time expressions from a recall query.

        This mirrors the voice_recording recall path but uses the recall
        timestamp as the relative-time anchor when one is provided. That keeps
        benchmark queries anchored to the question date instead of wall-clock
        time.
        """

        def parse_reference_time(value: Optional[str]) -> datetime:
            text = str(value or "").strip()
            if not text:
                return datetime.now()
            normalized = text.replace("T", " ")
            if normalized.endswith("Z"):
                normalized = normalized[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(normalized)
            except ValueError:
                pass
            candidates = [normalized, normalized[:19], normalized[:16], normalized[:10]]
            for candidate in candidates:
                for fmt_text in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M"):
                    try:
                        return datetime.strptime(candidate, fmt_text)
                    except ValueError:
                        continue
            return datetime.now()

        def fmt(value: datetime) -> str:
            return value.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")

        text = str(query or "")
        now = parse_reference_time(reference_time)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        clean_query = text

        m = re.search(r"(?:最近|近|过去)\s*(\d+)\s*(天|日|周|星期|个月|月|年)", text)
        if m:
            num = int(m.group(1))
            unit = m.group(2)
            if unit in ("天", "日"):
                delta = timedelta(days=num)
            elif unit in ("周", "星期"):
                delta = timedelta(weeks=num)
            elif unit in ("个月", "月"):
                delta = timedelta(days=num * 30)
            elif unit == "年":
                delta = timedelta(days=num * 365)
            else:
                delta = timedelta(days=num)
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(now - delta), fmt(now), clean_query.strip()

        m = re.search(r"\b(?:last|past|previous|recent)\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)\b", text, re.IGNORECASE)
        if m:
            num = int(m.group(1))
            unit = m.group(2).lower()
            if unit.startswith("day"):
                delta = timedelta(days=num)
            elif unit.startswith("week"):
                delta = timedelta(weeks=num)
            elif unit.startswith("month"):
                delta = timedelta(days=num * 30)
            else:
                delta = timedelta(days=num * 365)
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(now - delta), fmt(now), clean_query.strip()

        m = re.search(r"最近\s*", text)
        if m:
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(now - timedelta(days=7)), fmt(now), clean_query.strip()

        m = re.search(r"\b(?:recently|lately)\b", text, re.IGNORECASE)
        if m:
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(now - timedelta(days=7)), fmt(now), clean_query.strip()

        m = re.search(r"(?:从)?\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*(?:到|至)\s*(?:(\d{4})\s*年)?\s*(\d{1,2})\s*月", text)
        if m:
            year1 = int(m.group(1))
            month1 = int(m.group(2))
            year2 = int(m.group(3)) if m.group(3) else year1
            month2 = int(m.group(4))
            try:
                start = datetime(year1, month1, 1)
                end = (
                    datetime(year2, month2 + 1, 1) - timedelta(seconds=1)
                    if month2 < 12
                    else datetime(year2, 12, 31, 23, 59, 59)
                )
                clean_query = text[:m.start()] + text[m.end():]
                return fmt(start), fmt(end), clean_query.strip()
            except ValueError:
                pass

        m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月", text)
        if m:
            year = int(m.group(1))
            month = int(m.group(2))
            try:
                start = datetime(year, month, 1)
                end = (
                    datetime(year, month + 1, 1) - timedelta(seconds=1)
                    if month < 12
                    else datetime(year, 12, 31, 23, 59, 59)
                )
                clean_query = text[:m.start()] + text[m.end():]
                return fmt(start), fmt(end), clean_query.strip()
            except ValueError:
                pass

        m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
        if m:
            try:
                start = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                clean_query = text[:m.start()] + text[m.end():]
                return fmt(start), fmt(start + timedelta(days=1)), clean_query.strip()
            except ValueError:
                pass

        m = re.search(r"上(?:个)?(?:月|星期|周)", text)
        if m:
            unit = m.group()[1:]
            if "月" in unit:
                first_of_month = today_start.replace(day=1)
                end_of_last_month = first_of_month - timedelta(seconds=1)
                start_of_last_month = end_of_last_month.replace(day=1, hour=0, minute=0, second=0)
                start, end = start_of_last_month, end_of_last_month
            else:
                start_of_this_week = today_start - timedelta(days=today_start.weekday())
                start = start_of_this_week - timedelta(days=7)
                end = start_of_this_week
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(start), fmt(end), clean_query.strip()

        m = re.search(r"\b(last month|last week|previous month|previous week)\b", text, re.IGNORECASE)
        if m:
            phrase = m.group(1).lower()
            if "month" in phrase:
                first_of_month = today_start.replace(day=1)
                end = first_of_month - timedelta(seconds=1)
                start = end.replace(day=1, hour=0, minute=0, second=0)
            else:
                end = today_start - timedelta(days=today_start.weekday())
                start = end - timedelta(days=7)
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(start), fmt(end), clean_query.strip()

        m = re.search(r"(?:这个月|本月)", text)
        if m:
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(today_start.replace(day=1)), fmt(now), clean_query.strip()

        m = re.search(r"\b(this month|current month)\b", text, re.IGNORECASE)
        if m:
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(today_start.replace(day=1)), fmt(now), clean_query.strip()

        m = re.search(r"(?:本周|这一周)", text)
        if m:
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(today_start - timedelta(days=today_start.weekday())), fmt(now), clean_query.strip()

        m = re.search(r"\b(this week|current week)\b", text, re.IGNORECASE)
        if m:
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(today_start - timedelta(days=today_start.weekday())), fmt(now), clean_query.strip()

        m = re.search(r"昨天|昨日", text)
        if m:
            start = today_start - timedelta(days=1)
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(start), fmt(start + timedelta(days=1)), clean_query.strip()

        m = re.search(r"\byesterday\b", text, re.IGNORECASE)
        if m:
            start = today_start - timedelta(days=1)
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(start), fmt(start + timedelta(days=1)), clean_query.strip()

        m = re.search(r"前天|前日", text)
        if m:
            start = today_start - timedelta(days=2)
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(start), fmt(start + timedelta(days=1)), clean_query.strip()

        m = re.search(r"今天|今日", text)
        if m:
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(today_start), fmt(now), clean_query.strip()

        m = re.search(r"\btoday\b", text, re.IGNORECASE)
        if m:
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(today_start), fmt(now), clean_query.strip()

        return None, None, text

    # ── Recall path: raw candidates -> unified rerank -> formatted evidence ─

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
        if not self._enabled or not str(query or "").strip():
            return ""
        started_at = time.monotonic()
        try:
            if self._pending_store_turns:
                self.flush_pending_store_turns()
            k = max(1, int(top_k or self._top_k or 8))
            b = str(budget or self._recall_budget or "mid")
            recall_time_end = self._normalize_recall_time_bound(
                time_end,
                default_to_now=True,
            )
            self._log_info("memory_recall", "start", {
                "query": self._log_text(query, limit=500),
                "top_k": k,
                "budget": b,
                "tags": tags or [],
                "time_start": time_start,
                "time_end": recall_time_end,
                "recall_gate_mode": recall_gate_mode,
                "memory_source_override": list(memory_source_override or []),
            })

            parsed_time_start, parsed_time_end, clean_query = self._parse_time_expression(
                query,
                reference_time=recall_time_end,
            )
            effective_time_start = (
                self._normalize_recall_time_bound(time_start)
                or parsed_time_start
            )
            effective_time_end = (
                self._normalize_recall_time_bound(time_end)
                if time_end and not (parsed_time_start or parsed_time_end)
                else (parsed_time_end or recall_time_end)
            )
            search_query = clean_query or query
            self._log_info("memory_recall", "query_prepared", {
                "search_query": self._log_text(search_query, limit=500),
                "clean_query": self._log_text(clean_query, limit=500),
                "parsed_time_start": parsed_time_start,
                "parsed_time_end": parsed_time_end,
                "effective_time_start": effective_time_start,
                "effective_time_end": effective_time_end,
                "recall_time_end": recall_time_end,
            })

            recall_plan = self._analyze_recall_query(search_query)
            forced_source_types = self._normalize_source_override(memory_source_override)
            preferred_source_types = forced_source_types or self._normalize_source_override(
                recall_plan.get("source_types") or []
            )
            preferred_index_levels = self._normalize_index_levels(
                recall_plan.get("index_levels") or []
            )
            llm_keywords = self._normalize_string_list(
                recall_plan.get("keywords"),
                limit=12,
            )
            llm_entities = self._normalize_entity_names(
                recall_plan.get("entities"),
                limit=12,
            )
            terms = self._build_recall_search_terms(
                search_query,
                keywords=llm_keywords,
                entities=llm_entities,
            )
            retrieval_text = (
                recall_plan.get("search_text")
                or recall_plan.get("query_rewrite")
                or ""
            )
            query_embedding_text = self._query_embedding_text(
                search_query,
                retrieval_text=retrieval_text,
                keywords=llm_keywords,
                entities=llm_entities,
            )
            query_embedding = self._embed(query_embedding_text)
            final_candidate_limits = self._final_recall_candidate_limits(
                query=search_query,
                top_k=k,
                budget=b,
                preferred_index_levels=preferred_index_levels,
            )
            raw_candidate_limits = self._cal_raw_candidate_limits_from_recall_limits(
                final_candidate_limits,
                budget=b,
            )
            self._log_info("memory_recall", "query_analyzed", {
                "recall_plan": recall_plan,
                "forced_source_types": forced_source_types or [],
                "preferred_source_types": preferred_source_types or [],
                "preferred_index_levels": preferred_index_levels or [],
                "keywords": llm_keywords,
                "entities": llm_entities,
                "terms": terms,
                "retrieval_text": self._log_text(retrieval_text, limit=500),
                "embedding_text": self._log_text(query_embedding_text, limit=500),
                "query_embedding_available": query_embedding is not None,
                "final_candidate_limits": final_candidate_limits,
                "raw_candidate_limits": raw_candidate_limits,
                "budget": b,
                "parsed_time_start": parsed_time_start,
                "parsed_time_end": parsed_time_end,
                "effective_time_start": effective_time_start,
                "effective_time_end": effective_time_end,
            })
            fact_candidates, state_candidates, actionable_candidates = self._retrieve_recall_raw_candidates(
                terms=terms,
                source_types=forced_source_types,
                time_start=effective_time_start,
                time_end=effective_time_end,
                raw_candidate_limits=raw_candidate_limits,
            )
            self._log_info("memory_recall", "candidates_found", {
                "facts": self._recall_log_candidate_items(fact_candidates),
                "states": self._recall_log_candidate_items(state_candidates),
                "actionable_items": self._recall_log_candidate_items(actionable_candidates),
            })
            ranked = self._rank_recall_raw_candidates(
                facts=fact_candidates,
                states=state_candidates,
                actionable_items=actionable_candidates,
                query=search_query,
                terms=terms,
                query_embedding=query_embedding,
                preferred_source_types=preferred_source_types,
                preferred_index_levels=preferred_index_levels,
                final_candidate_limits=final_candidate_limits,
                fact_type_preference=str(
                    recall_plan.get("fact_type_preference") or "both"
                ).strip().lower(),
            )
            self._log_info("memory_recall", "ranked", {
                "facts": self._recall_log_candidate_items([
                    item for item in ranked if item.get("index_level") == "fact"
                ]),
                "states": self._recall_log_candidate_items([
                    item for item in ranked if item.get("index_level") == "state"
                ]),
                "actionable_items": self._recall_log_candidate_items([
                    item for item in ranked if item.get("index_level") == "actionable_item"
                ]),
                "all_ranked": self._recall_log_candidate_items(ranked, limit=20),
            })
            memory_text = self._build_memory_retrieved_format_text(
                entries=ranked,
                query=search_query,
                budget=b,
            )
            self._log_info("memory_recall", "finish", {
                "status": "ok" if memory_text else "empty",
                "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
                "recall_context_chars": len(memory_text or ""),
                "recall_context": memory_text,
            })
            return memory_text
        except Exception as exc:
            self._log_info("memory_recall", "error", {
                "query": self._log_text(query, limit=500),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
            })
            raise

    def _retrieve_recall_raw_candidates(
        self,
        *,
        terms: List[str],
        source_types: Optional[Sequence[str]],
        time_start: Optional[str],
        time_end: Optional[str],
        raw_candidate_limits: Dict[str, int],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build separate raw candidate groups before type-specific ranking.

        This intentionally bypasses `memory_index_entries` for the default
        path. Each candidate still points to its source row, so formatting and
        evidence hydration remain shared with the index-based implementation.
        """
        table_specs = (
            ("memory_facts", "fact", self._db.search_memory_facts),
            ("memory_states", "state", self._db.search_memory_states),
            ("memory_actionable_items", "actionable_item", self._db.search_memory_actionable_items),
        )
        rows_by_table: Dict[str, List[Dict[str, Any]]] = {}
        raw_limit_keys = {
            "memory_facts": "facts",
            "memory_states": "states",
            "memory_actionable_items": "actionable_items",
        }
        for table, _level, loader in table_specs:
            rows_by_table[table] = loader(
                terms=terms,
                source_types=source_types,
                # States and actionable items do not have an event-time
                # column. Their evidence facts provide the temporal filter.
                time_start=time_start if table == "memory_facts" else None,
                time_end=time_end if table == "memory_facts" else None,
                limit=max(1, int(raw_candidate_limits.get(raw_limit_keys[table], 1) or 1)),
            )

        evidence_ids: List[int] = []
        for table in ("memory_states", "memory_actionable_items"):
            for row in rows_by_table[table]:
                for value in row.get("evidence_fact_ids") or []:
                    try:
                        fact_id = int(value)
                    except (TypeError, ValueError):
                        continue
                    if fact_id not in evidence_ids:
                        evidence_ids.append(fact_id)
        evidence_by_id = {
            int(fact["id"]): fact
            for fact in self._db.memory_facts_by_ids(evidence_ids)
            if fact.get("id") is not None
        }

        candidates_by_level: Dict[str, List[Dict[str, Any]]] = {
            "fact": [],
            "state": [],
            "actionable_item": [],
        }
        for table, level, _loader in table_specs:
            for row in rows_by_table[table]:
                try:
                    target_id = int(row.get("id"))
                except (TypeError, ValueError):
                    continue
                source_type = str(row.get("source_type") or "unified")
                if table == "memory_facts":
                    title = _compact_whitespace(row.get("summary") or "")[:120]
                    summary = _compact_whitespace(row.get("summary") or "")
                    time_value = self._normalize_event_time_text(row.get("time_key"))
                    entities = row.get("entities") or []
                    topics = row.get("canonical_topics") or []
                elif table == "memory_states":
                    title = _compact_whitespace(row.get("canonical_name") or "")
                    summary = _compact_whitespace(row.get("summary") or "")
                    time_value = self._normalize_event_time_text(row.get("updated_at"))
                    entities = [row.get("entity_key")] if row.get("entity_key") else []
                    topics = [row.get("canonical_name")] if row.get("canonical_name") else []
                else:
                    title = _compact_whitespace(row.get("canonical_name") or "")
                    summary = _compact_whitespace(row.get("summary") or "")
                    time_value = self._normalize_event_time_text(
                        row.get("due_at") or row.get("updated_at") or row.get("created_at")
                    )
                    entities = [row.get("owner")] if row.get("owner") else []
                    topics = [row.get("canonical_name")] if row.get("canonical_name") else []

                support_facts = []
                if table in {"memory_states", "memory_actionable_items"}:
                    for value in row.get("evidence_fact_ids") or []:
                        try:
                            fact_id = int(value)
                        except (TypeError, ValueError):
                            continue
                        fact = evidence_by_id.get(fact_id)
                        if fact:
                            support_facts.append(dict(fact))
                        if len(support_facts) >= 4:
                            break
                    support_start, support_end = self._event_time_bounds_from_facts(support_facts)
                    if support_start or support_end:
                        time_value = support_start or support_end
                        time_end_value = support_end or support_start
                    else:
                        time_end_value = time_value
                    if time_start and time_end_value and time_end_value < str(time_start):
                        continue
                    if time_end and time_value and time_value > str(time_end):
                        continue
                else:
                    time_end_value = time_value

                hydrated = dict(row)
                hydrated.pop("embedding", None)
                metadata = dict(row.get("metadata") or {})
                metadata["_matched_via"] = ["direct"]
                candidates_by_level[level].append({
                    "source_type": source_type,
                    "target_table": table,
                    "target_id": target_id,
                    "index_level": level,
                    "memory_path": f"{source_type}/{level}",
                    "title": title,
                    "summary_for_retrieval": summary,
                    "keywords": row.get("keywords") or "",
                    "entities": entities,
                    "canonical_topics": topics,
                    "participants": row.get("participants") or [],
                    "time_start": time_value,
                    "time_end": time_end_value,
                    "importance": row.get("importance") or 0.5,
                    "confidence": row.get("confidence") or 0.8,
                    "embedding": row.get("embedding"),
                    "metadata": metadata,
                    "_hydrated": hydrated,
                    "_supporting_facts": support_facts,
                    "_recall_candidate_source": "direct",
                })
        logger.debug(
            "Direct recall candidates: facts=%d states=%d actionable_items=%d total=%d",
            len(rows_by_table["memory_facts"]),
            len(rows_by_table["memory_states"]),
            len(rows_by_table["memory_actionable_items"]),
            sum(len(items) for items in candidates_by_level.values()),
        )
        return (
            candidates_by_level["fact"],
            candidates_by_level["state"],
            candidates_by_level["actionable_item"],
        )

    def _recall_type_weights(
        self,
        query: str,
        preferred_index_levels: Optional[Sequence[str]],
    ) -> Dict[str, float]:
        """Build query-intent weights for the three raw memory layers."""
        weights = {"fact": 1.0, "state": 1.0, "actionable_item": 1.0}
        preferred = set(preferred_index_levels or [])
        if preferred:
            for level in weights:
                weights[level] = 1.3 if level in preferred else 0.82

        lower = str(query or "").lower()
        fact_markers = (
            "when", "where", "who", "which", "what happened", "before", "after",
            "什么时候", "哪里", "谁", "哪一个", "发生了什么", "之前", "之后", "具体说过",
        )
        state_markers = (
            "prefer", "usually", "habit", "profile", "relationship", "trend", "state",
            "偏好", "通常", "习惯", "画像", "关系", "趋势", "长期", "状态", "一般怎么",
        )
        actionable_markers = (
            "todo", "task", "remind", "decision", "commit", "deadline", "follow up",
            "待办", "任务", "提醒", "决定", "承诺", "截止", "跟进", "下一步", "风险",
        )
        if any(marker in lower for marker in fact_markers):
            weights["fact"] *= 1.28
        if any(marker in lower for marker in state_markers):
            weights["state"] *= 1.28
        if any(marker in lower for marker in actionable_markers):
            weights["actionable_item"] *= 1.35
        return weights

    def _make_recall_fact_candidate(
        self,
        fact: Dict[str, Any],
        *,
        candidate_source: str,
    ) -> Optional[Dict[str, Any]]:
        """Convert a raw/supporting fact into the shared display shape."""
        try:
            fact_id = int(fact.get("id"))
        except (TypeError, ValueError):
            return None
        source_type = str(fact.get("source_type") or "unified")
        summary = _compact_whitespace(fact.get("summary") or "")
        time_value = self._normalize_event_time_text(fact.get("time_key"))
        hydrated = dict(fact)
        hydrated.pop("embedding", None)
        metadata = dict(fact.get("metadata") or {})
        metadata["_matched_via"] = [candidate_source]
        return {
            "source_type": source_type,
            "target_table": "memory_facts",
            "target_id": fact_id,
            "index_level": "fact",
            "memory_path": f"{source_type}/fact",
            "title": summary[:120],
            "summary_for_retrieval": summary,
            "keywords": fact.get("keywords") or "",
            "entities": fact.get("entities") or [],
            "canonical_topics": fact.get("canonical_topics") or [],
            "participants": fact.get("participants") or [],
            "time_start": time_value,
            "time_end": time_value,
            "importance": fact.get("importance") or 0.5,
            "confidence": fact.get("confidence") or 0.8,
            "embedding": fact.get("embedding"),
            "metadata": metadata,
            "_hydrated": hydrated,
            "_supporting_facts": [],
            "_recall_candidate_source": candidate_source,
        }

    def _rank_recall_candidates_by_type(
        self,
        candidates: List[Dict[str, Any]],
        *,
        memory_type: str,
        terms: Sequence[str],
        query: str,
        query_embedding: Optional[np.ndarray],
        min_embedding_similarity: Optional[float],
        fact_type_preference: str = "both",
    ) -> List[Dict[str, Any]]:
        """Score one memory type with fields appropriate to its semantics."""
        query_lower = str(query or "").lower()
        threshold = (
            None
            if min_embedding_similarity is None
            else self._clamp_float(min_embedding_similarity, 0.0, 1.0, 0.0)
        )
        scored: List[Tuple[float, str, int, Dict[str, Any]]] = []
        for position, entry in enumerate(candidates):
            raw = entry.get("_hydrated") if isinstance(entry.get("_hydrated"), dict) else {}
            retrieval_summary = (
                entry.get("summary_for_retrieval")
                or entry.get("summary")
                or raw.get("summary")
            )
            if memory_type == "fact":
                fields = (
                    entry.get("title"), retrieval_summary,
                    entry.get("keywords"), entry.get("entities"),
                    entry.get("canonical_topics"), raw.get("fact_kind"),
                    raw.get("fact_subject"), raw.get("fact_type"), raw.get("time_key"),
                )
            elif memory_type == "state":
                fields = (
                    entry.get("title"), retrieval_summary,
                    entry.get("canonical_topics"), entry.get("entities"),
                    raw.get("state_scope"), raw.get("state_type"),
                    raw.get("entity_key"), raw.get("time_line"),
                )
            else:
                fields = (
                    entry.get("title"), retrieval_summary,
                    entry.get("keywords"), entry.get("entities"),
                    entry.get("canonical_topics"), raw.get("item_type"),
                    raw.get("owner"), raw.get("status"), raw.get("due_at"),
                )
            text = " ".join(
                _compact_whitespace(_json_safe(value) if isinstance(value, (list, dict)) else value)
                for value in fields
                if value
            ).lower()
            term_hits = sum(1 for term in terms if term and term.lower() in text)
            term_coverage = term_hits / max(1, min(len(terms), 8))
            phrase_bonus = 0.0
            for quoted in re.findall(r"'([^']+)'|\"([^\"]+)\"", query):
                phrase = (quoted[0] or quoted[1]).lower()
                if phrase and phrase in text:
                    phrase_bonus += 1.5
            similarity = max(0.0, self._cal_embedding_similarity(
                query_embedding,
                entry.get("embedding"),
            ))
            if (
                threshold is not None
                and query_embedding is not None
                and similarity < threshold
            ):
                continue
            importance = self._clamp_float(entry.get("importance"), 0.0, 1.0, 0.5)
            confidence = self._clamp_float(entry.get("confidence"), 0.0, 1.0, 0.8)
            support_score = self._clamp_float(
                entry.get("_recall_support_score"),
                0.0,
                1.0,
                0.0,
            )
            score = (
                term_coverage * 2.4
                + term_hits * 0.28
                + phrase_bonus
                + similarity * 2.0
                + importance * 0.4
                + confidence * 0.25
                + support_score * 0.55
            )
            if memory_type == "fact":
                if fact_type_preference in {"semantic", "episodic"} and str(
                    raw.get("fact_type") or ""
                ).lower() == fact_type_preference:
                    score += 0.35
                if any(marker in query_lower for marker in ("when", "before", "after", "什么时候", "之前", "之后")):
                    score += 0.2
            elif memory_type == "state":
                if raw.get("canonical_name") and any(
                    str(term).lower() in str(raw.get("canonical_name")).lower()
                    for term in terms
                ):
                    score += 0.35
            else:
                if any(marker in query_lower for marker in (
                    "todo", "task", "remind", "decision", "commit", "待办", "任务", "提醒", "决定", "承诺",
                )):
                    score += 0.35
                if str(raw.get("status") or "").lower() in {"open", "in_progress", "blocked"}:
                    score += 0.1
            item = dict(entry)
            item.pop("embedding", None)
            if query_embedding is not None:
                item["embedding_similarity"] = round(float(similarity), 4)
            scored.append((score, str(entry.get("time_start") or ""), -position, item))

        if not scored:
            return []
        raw_scores = [row[0] for row in scored]
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        score_span = max_score - min_score
        ranked: List[Dict[str, Any]] = []
        for rank, (score, time_value, position, item) in enumerate(
            sorted(scored, key=lambda row: (row[0], row[1], row[2]), reverse=True),
            1,
        ):
            normalized = (
                (score - min_score) / score_span
                if score_span > 1e-6
                else min(1.0, max(0.0, score / 4.0))
            )
            item["_recall_type"] = memory_type
            item["_recall_type_score"] = round(float(normalized), 4)
            item["_recall_rank"] = rank
            ranked.append(item)
        return ranked

    def _rank_recall_fact_candidates(
        self,
        candidates: List[Dict[str, Any]],
        *,
        terms: Sequence[str],
        query: str,
        query_embedding: Optional[np.ndarray],
        min_embedding_similarity: Optional[float],
        fact_type_preference: str = "both",
    ) -> List[Dict[str, Any]]:
        return self._rank_recall_candidates_by_type(
            candidates,
            memory_type="fact",
            terms=terms,
            query=query,
            query_embedding=query_embedding,
            min_embedding_similarity=min_embedding_similarity,
            fact_type_preference=fact_type_preference,
        )

    def _rank_recall_state_candidates(
        self,
        candidates: List[Dict[str, Any]],
        *,
        terms: Sequence[str],
        query: str,
        query_embedding: Optional[np.ndarray],
        min_embedding_similarity: Optional[float],
    ) -> List[Dict[str, Any]]:
        return self._rank_recall_candidates_by_type(
            candidates,
            memory_type="state",
            terms=terms,
            query=query,
            query_embedding=query_embedding,
            min_embedding_similarity=min_embedding_similarity,
        )

    def _rank_recall_actionable_candidates(
        self,
        candidates: List[Dict[str, Any]],
        *,
        terms: Sequence[str],
        query: str,
        query_embedding: Optional[np.ndarray],
        min_embedding_similarity: Optional[float],
    ) -> List[Dict[str, Any]]:
        return self._rank_recall_candidates_by_type(
            candidates,
            memory_type="actionable_item",
            terms=terms,
            query=query,
            query_embedding=query_embedding,
            min_embedding_similarity=min_embedding_similarity,
        )

    def _rank_recall_raw_candidates(
        self,
        *,
        facts: List[Dict[str, Any]],
        states: List[Dict[str, Any]],
        actionable_items: List[Dict[str, Any]],
        query: str,
        terms: Sequence[str],
        query_embedding: Optional[np.ndarray],
        preferred_source_types: Optional[Sequence[str]] = None,
        preferred_index_levels: Optional[Sequence[str]] = None,
        final_candidate_limits: Optional[Dict[str, int]] = None,
        fact_type_preference: str = "both",
        fact_min_embedding_similarity: Optional[float] = None,
        state_min_embedding_similarity: Optional[float] = None,
        actionable_item_min_embedding_similarity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Rank each raw layer independently, then merge normalized scores."""
        weights = self._recall_type_weights(query, preferred_index_levels)
        ranked_states = self._rank_recall_state_candidates(
            states,
            terms=terms,
            query=query,
            query_embedding=query_embedding,
            min_embedding_similarity=(
                self._recall_state_min_embedding_similarity
                if state_min_embedding_similarity is None
                else state_min_embedding_similarity
            ),
        )
        ranked_actionable = self._rank_recall_actionable_candidates(
            actionable_items,
            terms=terms,
            query=query,
            query_embedding=query_embedding,
            min_embedding_similarity=(
                self._recall_actionable_item_min_embedding_similarity
                if actionable_item_min_embedding_similarity is None
                else actionable_item_min_embedding_similarity
            ),
        )
        final_limits = final_candidate_limits or {
            "facts": len(facts),
            "states": len(states),
            "actionable_items": len(actionable_items),
        }
        ranked_states = ranked_states[: max(0, int(final_limits.get("states", 0) or 0))]
        ranked_actionable = ranked_actionable[: max(
            0,
            int(final_limits.get("actionable_items", 0) or 0),
        )]

        all_fact_candidates = [dict(item) for item in facts]
        fact_ids = {
            int(item.get("target_id"))
            for item in all_fact_candidates
            if str(item.get("target_id") or "").isdigit()
        }
        for parent in [*ranked_states, *ranked_actionable]:
            parent_score = float(parent.get("_recall_type_score") or 0.0)
            for support_fact in parent.get("_supporting_facts") or []:
                try:
                    fact_id = int(support_fact.get("id"))
                except (TypeError, ValueError):
                    continue
                if fact_id in fact_ids:
                    continue
                candidate = self._make_recall_fact_candidate(
                    support_fact,
                    candidate_source=f"support_{parent.get('_recall_type', 'memory')}",
                )
                if candidate:
                    candidate["_recall_support_score"] = round(parent_score * 0.85, 4)
                    all_fact_candidates.append(candidate)
                    fact_ids.add(fact_id)
        ranked_facts = self._rank_recall_fact_candidates(
            all_fact_candidates,
            terms=terms,
            query=query,
            query_embedding=query_embedding,
            min_embedding_similarity=(
                self._recall_fact_min_embedding_similarity
                if fact_min_embedding_similarity is None
                else fact_min_embedding_similarity
            ),
            fact_type_preference=fact_type_preference,
        )
        ranked_facts = ranked_facts[: max(0, int(final_limits.get("facts", 0) or 0))]

        merged: List[Dict[str, Any]] = []
        for item in [*ranked_states, *ranked_actionable, *ranked_facts]:
            candidate = dict(item)
            level = str(candidate.get("index_level") or candidate.get("_recall_type") or "")
            type_score = float(candidate.get("_recall_type_score") or 0.0)
            source_bonus = 0.0
            if preferred_source_types and candidate.get("source_type") in set(preferred_source_types):
                source_bonus = 0.08
            candidate["_recall_score"] = round(
                type_score * weights.get(level, 1.0) + source_bonus,
                4,
            )
            merged.append(candidate)
        merged.sort(
            key=lambda item: (
                float(item.get("_recall_score") or 0.0),
                str(item.get("time_start") or ""),
            ),
            reverse=True,
        )

        selected: List[Dict[str, Any]] = []
        seen_targets: set[Tuple[str, int]] = set()
        for item in merged:
            try:
                target = (str(item.get("target_table") or ""), int(item.get("target_id")))
            except (TypeError, ValueError):
                continue
            if target in seen_targets:
                continue
            seen_targets.add(target)
            selected.append(item)

        return selected

    def _rank_recall_raw_candidates_unified(
        self,
        entries: List[Dict[str, Any]],
        *,
        query: str,
        terms: List[str],
        query_embedding: Optional[np.ndarray],
        preferred_source_types: Optional[Sequence[str]] = None,
        preferred_index_levels: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Legacy unified scorer retained for comparison and fallback."""
        query_lower = str(query or "").lower()
        preferred_sources = set(preferred_source_types or [])
        preferred_levels = set(preferred_index_levels or [])
        scored: List[Tuple[float, str, int, Dict[str, Any]]] = []
        seen_targets: set[Tuple[str, int]] = set()
        for position, entry in enumerate(entries):
            try:
                target = (str(entry.get("target_table") or ""), int(entry.get("target_id")))
            except (TypeError, ValueError):
                continue
            if target in seen_targets:
                continue
            seen_targets.add(target)
            raw = entry.get("_hydrated") if isinstance(entry.get("_hydrated"), dict) else {}
            text_parts = [
                entry.get("title"),
                entry.get("summary_for_retrieval"),
                entry.get("keywords"),
                entry.get("entities"),
                entry.get("canonical_topics"),
                raw.get("fact_kind"),
                raw.get("fact_subject"),
                raw.get("state_scope"),
                raw.get("state_type"),
                raw.get("entity_key"),
                raw.get("time_line"),
                raw.get("item_type"),
                raw.get("owner"),
                raw.get("status"),
                raw.get("due_at"),
            ]
            text = " ".join(
                _compact_whitespace(_json_safe(part) if isinstance(part, (list, dict)) else part)
                for part in text_parts
                if part
            ).lower()
            term_hits = sum(1 for term in terms if term and term in text)
            phrase_bonus = 0.0
            for quoted in re.findall(r"'([^']+)'|\"([^\"]+)\"", query):
                phrase = (quoted[0] or quoted[1]).lower()
                if phrase and phrase in text:
                    phrase_bonus += 3.0
            for ngram in self._important_ngrams(query_lower):
                if ngram in text:
                    phrase_bonus += 1.0
            sim = self._cal_embedding_similarity(query_embedding, entry.get("embedding"))
            importance = self._clamp_float(entry.get("importance"), 0.0, 1.0, 0.5)
            confidence = self._clamp_float(entry.get("confidence"), 0.0, 1.0, 0.8)
            level = str(entry.get("index_level") or "")
            base_level_bonus = {"fact": 0.35, "state": 0.28, "actionable_item": 0.32}.get(level, 0.15)
            source_bonus = 0.7 if preferred_sources and entry.get("source_type") in preferred_sources else 0.0
            level_bonus = 0.55 if preferred_levels and level in preferred_levels else 0.0
            direct_bonus = 0.12
            score = (
                term_hits * 0.85
                + phrase_bonus
                + max(0.0, sim) * 2.2
                + importance * 0.55
                + confidence * 0.35
                + base_level_bonus
                + source_bonus
                + level_bonus
                + direct_bonus
                + (1.0 / max(1, position + 1)) * 0.25
            )
            item = dict(entry)
            item.pop("embedding", None)
            item["_recall_score"] = round(score, 4)
            item["_embedding_similarity"] = round(float(sim), 4)
            item["_source_preference_bonus"] = round(source_bonus, 4)
            item["_level_preference_bonus"] = round(level_bonus, 4)
            item["_retrieval_signal_bonus"] = direct_bonus
            scored.append((score, str(entry.get("time_start") or ""), -position, item))
        scored.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
        return [item for _, _, _, item in scored]

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

    def _build_recall_search_terms(
        self,
        query: str,
        *,
        keywords: Sequence[str],
        entities: Sequence[str],
    ) -> List[str]:
        """Combine LLM query concepts with deterministic fallback terms."""
        terms: List[str] = []
        seen = set()
        for value in [*keywords, *entities, *self._query_terms(query)]:
            clean = self._normalize_keyword_term(value).lower()
            if not clean or clean in seen:
                continue
            if len(clean) > 80 or re.search(r"[。！？!?；;，,]", clean):
                continue
            seen.add(clean)
            terms.append(clean)
            if len(terms) >= 32:
                break
        return terms

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
        preferred_source_types: Optional[Sequence[str]] = None,
        preferred_index_levels: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        query_lower = str(query or "").lower()
        preferred_sources = set(preferred_source_types or [])
        preferred_levels = set(preferred_index_levels or [])
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
            base_level_bonus = 0.35 if entry.get("index_level") == "fact" else 0.15
            importance = float(entry.get("importance") or 0.5)
            confidence = float(entry.get("confidence") or 0.8)
            score = (
                term_hits * 0.85
                + phrase_bonus
                + max(0.0, sim) * 2.2
                + importance * 0.55
                + confidence * 0.35
                + base_level_bonus
                + (1.0 / max(1, position + 1)) * 0.25
            )
            item = dict(entry)
            source_bonus = 0.0
            if preferred_sources and entry.get("source_type") in preferred_sources:
                source_bonus = 0.7
            preference_level_bonus = 0.0
            if preferred_levels and entry.get("index_level") in preferred_levels:
                preference_level_bonus = 0.55
            retrieval_bonus = 0.0
            metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            matched_via = metadata.get("_matched_via") or []
            matched_via_text = " ".join(str(item) for item in matched_via)
            if "fts" in matched_via_text:
                retrieval_bonus += 0.55
            if "like" in matched_via_text:
                retrieval_bonus += 0.35
            if "recent" in matched_via_text:
                retrieval_bonus += 0.08
            item["_recall_score"] = round(score, 4)
            item["_embedding_similarity"] = round(float(sim), 4)
            item["_source_preference_bonus"] = round(source_bonus, 4)
            item["_level_preference_bonus"] = round(preference_level_bonus, 4)
            item["_retrieval_signal_bonus"] = round(retrieval_bonus, 4)
            item.pop("embedding", None)
            score += source_bonus + preference_level_bonus + retrieval_bonus
            item["_recall_score"] = round(score, 4)
            scored.append((score, str(entry.get("time_start") or ""), -position, item))
        scored.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
        return [item for _, _, _, item in scored]

    def _hydrate_recall_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ids_by_table: Dict[str, List[int]] = {}
        for entry in entries:
            table = str(entry.get("target_table") or "")
            try:
                target_id = int(entry.get("target_id"))
            except (TypeError, ValueError):
                continue
            ids_by_table.setdefault(table, [])
            if target_id not in ids_by_table[table]:
                ids_by_table[table].append(target_id)

        hydrated_by_table: Dict[str, Dict[int, Dict[str, Any]]] = {}
        table_loaders = {
            "memory_episodes": self._db.memory_episodes_by_ids,
            "memory_facts": self._db.memory_facts_by_ids,
            "memory_states": self._db.memory_states_by_ids,
            "memory_actionable_items": self._db.memory_actionable_items_by_ids,
        }
        for table, ids in ids_by_table.items():
            loader = table_loaders.get(table)
            if not loader:
                continue
            loaded = loader(ids)
            hydrated_by_table[table] = {
                int(item["id"]): item
                for item in loaded
                if item.get("id") is not None
            }

        support_fact_ids: List[int] = []
        for table, rows_by_id in hydrated_by_table.items():
            if table not in {"memory_states", "memory_actionable_items"}:
                continue
            for item in rows_by_id.values():
                for value in item.get("evidence_fact_ids") or []:
                    try:
                        fact_id = int(value)
                    except (TypeError, ValueError):
                        continue
                    if fact_id not in support_fact_ids:
                        support_fact_ids.append(fact_id)
        support_facts_by_id = {
            int(fact["id"]): fact
            for fact in self._db.memory_facts_by_ids(support_fact_ids)
            if fact.get("id") is not None
        } if support_fact_ids else {}

        hydrated_entries: List[Dict[str, Any]] = []
        for entry in entries:
            item = dict(entry)
            table = str(item.get("target_table") or "")
            try:
                target_id = int(item.get("target_id"))
            except (TypeError, ValueError):
                target_id = 0
            raw = hydrated_by_table.get(table, {}).get(target_id)
            if raw:
                raw_item = dict(raw)
                raw_item.pop("embedding", None)
                item["_hydrated"] = raw_item
                if table in {"memory_states", "memory_actionable_items"}:
                    evidence_ids = [
                        int(value)
                        for value in raw_item.get("evidence_fact_ids") or []
                        if str(value).strip().isdigit()
                    ]
                    item["_supporting_facts"] = [
                        support_facts_by_id[fact_id]
                        for fact_id in evidence_ids[:4]
                        if fact_id in support_facts_by_id
                    ]
                    for fact in item["_supporting_facts"]:
                        fact.pop("embedding", None)
            hydrated_entries.append(item)
        return hydrated_entries

    def _recall_context_char_budget(self, budget: str) -> int:
        return int(
            self._recall_context_char_budgets.get(
                str(budget or "mid").lower(),
                self._recall_context_char_budgets["mid"],
            )
        )

    def _recall_entry_char_budget(self, budget: str) -> int:
        return int(
            self._recall_entry_char_budgets.get(
                str(budget or "mid").lower(),
                self._recall_entry_char_budgets["mid"],
            )
        )

    @staticmethod
    def _truncate_recall_line(text: Any, *, max_chars: int) -> str:
        clean = _compact_whitespace(text or "")
        if len(clean) <= max_chars:
            return clean
        return clean[: max(0, max_chars - 18)].rstrip() + "...[truncated]"

    @staticmethod
    def _normalize_event_time_text(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        text = text.split("#", 1)[0].strip().replace("T", " ")
        if len(text) >= 19:
            return text[:19]
        if len(text) >= 10:
            return text[:10]
        return text

    @classmethod
    def _event_time_bounds_from_facts(cls, facts: Sequence[Dict[str, Any]]) -> Tuple[str, str]:
        times = [
            cls._normalize_event_time_text(fact.get("time_key"))
            for fact in facts or []
        ]
        times = sorted(time for time in times if time)
        if not times:
            return "", ""
        return times[0], times[-1]

    @staticmethod
    def _format_event_time_range(start: str, end: str) -> str:
        if start and end and start != end:
            return f"{start} - {end}"
        return start or end or "unknown-event-time"

    def _recall_event_time_text(self, entry: Dict[str, Any], raw: Dict[str, Any]) -> str:
        target_table = str(entry.get("target_table") or "")
        if target_table == "memory_facts":
            return (
                self._normalize_event_time_text(raw.get("time_key"))
                or self._normalize_event_time_text(entry.get("time_start"))
                or self._normalize_event_time_text(entry.get("time_end"))
                or "unknown-event-time"
            )
        if target_table == "memory_episodes":
            start = self._normalize_event_time_text(raw.get("started_at") or entry.get("time_start"))
            end = self._normalize_event_time_text(raw.get("ended_at") or entry.get("time_end"))
            return self._format_event_time_range(start, end)
        if target_table in {"memory_states", "memory_actionable_items"}:
            evidence_start, evidence_end = self._event_time_bounds_from_facts(
                entry.get("_supporting_facts") or [],
            )
            if evidence_start or evidence_end:
                return self._format_event_time_range(evidence_start, evidence_end)
            start = self._normalize_event_time_text(entry.get("time_start"))
            end = self._normalize_event_time_text(entry.get("time_end"))
            if start or end:
                return self._format_event_time_range(start, end)
            return "unknown-event-time"
        return (
            self._normalize_event_time_text(raw.get("time_key"))
            or self._normalize_event_time_text(raw.get("started_at"))
            or self._normalize_event_time_text(entry.get("time_start"))
            or self._normalize_event_time_text(entry.get("time_end"))
            or "unknown-event-time"
        )

    def _build_memory_retrieved_format_text(
        self,
        *,
        entries: List[Dict[str, Any]],
        query: str = "",
        budget: str = "mid",
    ) -> str:
        """Format ranked raw memories with source-specific semantic fields."""
        del query, budget
        if not entries:
            return ""

        def values_text(value: Any) -> str:
            if isinstance(value, list):
                values: List[str] = []
                for item in value:
                    if isinstance(item, dict):
                        text = _compact_whitespace(item.get("name") or item.get("text") or "")
                    else:
                        text = _compact_whitespace(item)
                    if text:
                        values.append(text)
                return ", ".join(values)
            return _compact_whitespace(value)

        def metadata_line(entry: Dict[str, Any]) -> str:
            source = str(entry.get("source_type") or "unified")
            target = f"{entry.get('target_table')}#{entry.get('target_id')}"
            score = entry.get("_recall_score")
            return f"   source: {source}; target: {target}; recall_score: {score}"

        grouped = {
            "state": [entry for entry in entries if entry.get("index_level") == "state"],
            "actionable_item": [
                entry for entry in entries
                if entry.get("index_level") == "actionable_item"
            ],
            "fact": [entry for entry in entries if entry.get("index_level") == "fact"],
        }
        section_specs = (
            (
                "[Long-term States]",
                "These are evolving state projections derived from memory facts. "
                "Treat them as summarized context, not direct user quotations.",
                "state",
            ),
            (
                "[Actionable Items]",
                "These are decisions, tasks, commitments, risks, or open questions "
                "that may require follow-up.",
                "actionable_item",
            ),
            (
                "[Retrieved Facts]",
                "These are ranked narrative facts retrieved directly from memory_facts.",
                "fact",
            ),
        )
        lines = [
            "[Unified Memory]",
            "System note: Memories are grouped by semantic role. States and actionable "
            "items provide compact summaries; facts provide traceable evidence.",
        ]
        for title, note, group_key in section_specs:
            group = grouped[group_key]
            if not group:
                continue
            section_header = [title, f"System note: {note}"]
            lines.extend(["", *section_header])
            for index, entry in enumerate(group, 1):
                raw = entry.get("_hydrated") if isinstance(entry.get("_hydrated"), dict) else {}
                time_text = self._recall_event_time_text(entry, raw)
                if group_key == "fact":
                    block_lines = [
                        f"{index}. [{time_text}] narrative fact",
                        metadata_line(entry),
                        f"   fact_type: {raw.get('fact_type') or ''}; fact_kind: {raw.get('fact_kind') or ''}; subject: {raw.get('fact_subject') or ''}",
                        f"   summary: {raw.get('summary') or entry.get('summary_for_retrieval') or ''}",
                        f"   topics: {values_text(raw.get('canonical_topics'))}",
                        f"   entities: {values_text(raw.get('entities'))}",
                        f"   keywords: {values_text(raw.get('keywords'))}",
                    ]
                elif group_key == "state":
                    timeline = self._format_state_timeline(raw.get("time_line"))
                    block_lines = [
                        f"{index}. [{time_text}] long-term state",
                        metadata_line(entry),
                        f"   state_scope: {raw.get('state_scope') or ''}; state_type: {raw.get('state_type') or ''}",
                        f"   canonical_name: {raw.get('canonical_name') or ''}",
                        f"   entity: {raw.get('entity_key') or ''}",
                        f"   summary: {raw.get('summary') or ''}",
                    ]
                    if timeline:
                        block_lines.append(f"   timeline: {timeline}")
                else:
                    block_lines = [
                        f"{index}. [{time_text}] actionable item",
                        metadata_line(entry),
                        f"   item_type: {raw.get('item_type') or ''}; status: {raw.get('status') or ''}",
                        f"   canonical_name: {raw.get('canonical_name') or ''}",
                        f"   owner: {raw.get('owner') or ''}; due_at: {raw.get('due_at') or ''}",
                        f"   summary: {raw.get('summary') or ''}",
                    ]
                lines.append("\n".join(block_lines))
        return "\n".join(lines).strip()

    def _format_recall_context(
        self,
        *,
        query: str,
        entries: List[Dict[str, Any]],
        budget: str = "mid",
    ) -> str:
        if not entries:
            return ""
        max_chars = self._recall_context_char_budget(budget)
        entry_max_chars = self._recall_entry_char_budget(budget)
        lines = [
            "[Unified Memory Recall]",
            "System note: Candidate selection searches raw memory_facts, "
            "memory_states, and memory_actionable_items, then reranks the "
            "candidates. memory_index_entries is retained as an optional future "
            "retrieval layer.",
            f"Recall budget: {budget} max_chars={max_chars}",
            f"Query: {query}",
            "",
        ]
        emitted = 0
        omitted = 0
        for rank, entry in enumerate(entries, 1):
            raw = entry.get("_hydrated") if isinstance(entry.get("_hydrated"), dict) else {}
            meta = {
                "source": entry.get("source_type"),
                "level": entry.get("index_level"),
                "path": entry.get("memory_path"),
                "target": f"{entry.get('target_table')}#{entry.get('target_id')}",
                "score": entry.get("_recall_score"),
                "matched_via": (entry.get("metadata") or {}).get("_matched_via", []),
            }
            time_text = self._recall_event_time_text(entry, raw)
            content = self._truncate_recall_line(
                self._format_hydrated_recall_content(entry, raw),
                max_chars=entry_max_chars,
            )
            block_lines = [
                f"{rank}. [{time_text}] {content}",
                f"   metadata: {_json_safe(meta)}",
            ]
            supporting_facts = entry.get("_supporting_facts") or []
            if supporting_facts:
                evidence_lines = []
                for fact in supporting_facts[:4]:
                    fact_time = self._normalize_event_time_text(fact.get("time_key"))
                    prefix = f"#{fact.get('id')}"
                    if fact_time:
                        prefix = f"{prefix} [{fact_time}]"
                    evidence_lines.append(
                        f"{prefix}: {self._truncate_recall_line(fact.get('summary') or '', max_chars=240)}"
                    )
                block_lines.append(f"   supporting_facts: {' | '.join(evidence_lines)}")
            block = "\n".join(block_lines)
            current = "\n".join(lines)
            separator = "\n" if current else ""
            if len(current) + len(separator) + len(block) > max_chars:
                omitted = len(entries) - emitted
                break
            lines.append(block)
            emitted += 1
        if omitted > 0:
            note = f"... omitted {omitted} lower-ranked recall entries due to budget."
            current = "\n".join(lines)
            if len(current) + 1 + len(note) <= max_chars:
                lines.append(note)
        return "\n".join(lines)

    def _format_state_timeline(
        self,
        value: Any,
        *,
        max_events: int = 8,
        max_chars: int = 520,
    ) -> str:
        events = self._normalize_time_line(
            value,
            limit=max_events,
            max_chars=max_chars,
        )
        if not events:
            return ""
        parts: List[str] = []
        for event in events:
            occurred_at = self._normalize_event_time_text(event.get("occurred_at"))
            change_type = _compact_whitespace(event.get("change_type") or "updated")
            summary = self._truncate_recall_line(
                event.get("summary") or "",
                max_chars=150,
            )
            if not summary:
                continue
            time_label = occurred_at or "unknown-time"
            parts.append(f"[{time_label} {change_type}] {summary}")
        if not parts:
            return ""
        return self._truncate_recall_line(
            "; ".join(parts),
            max_chars=max_chars,
        )

    def _format_hydrated_recall_content(
        self,
        entry: Dict[str, Any],
        raw: Dict[str, Any],
    ) -> str:
        if not raw:
            return _compact_whitespace(entry.get("summary_for_retrieval") or "")
        target_table = str(entry.get("target_table") or "")
        if target_table == "memory_episodes":
            title = _compact_whitespace(raw.get("title") or "")
            summary = _compact_whitespace(raw.get("summary") or "")
            participants = raw.get("participants") or []
            participant_text = f" participants={participants}" if participants else ""
            return f"episode: {title}. {summary}{participant_text}".strip()
        if target_table == "memory_facts":
            fact_kind = str(raw.get("fact_kind") or "")
            subject = str(raw.get("fact_subject") or "")
            summary = _compact_whitespace(raw.get("summary") or "")
            return f"fact({subject}/{fact_kind}): {summary}".strip()
        if target_table == "memory_states":
            state_scope = str(raw.get("state_scope") or "")
            state_type = str(raw.get("state_type") or "")
            name = _compact_whitespace(raw.get("canonical_name") or "")
            summary = _compact_whitespace(raw.get("summary") or "")
            label = "/".join(part for part in (state_scope, state_type) if part)
            timeline = self._format_state_timeline(raw.get("time_line"))
            content = f"state({label}) {name}: {summary}".strip()
            if timeline:
                content = f"{content}; timeline: {timeline}"
            return content
        if target_table == "memory_actionable_items":
            item_type = str(raw.get("item_type") or "")
            status = str(raw.get("status") or "")
            owner = str(raw.get("owner") or "")
            name = _compact_whitespace(raw.get("canonical_name") or "")
            summary = _compact_whitespace(raw.get("summary") or "")
            extras = ", ".join(part for part in (f"owner={owner}", f"status={status}") if part)
            return f"actionable({item_type}, {extras}) {name}: {summary}".strip()
        return _compact_whitespace(raw.get("summary") or entry.get("summary_for_retrieval") or "")

    # ── Lightweight NLP heuristics ───────────────────────────────────────

    def _embed(self, text: str) -> Optional[np.ndarray]:
        self._ensure_embedding_client()
        return self._embedding_client.embed_text(text) if self._embedding_client else None

    def _query_embedding_text(
        self,
        query: str,
        *,
        retrieval_text: str = "",
        keywords: Optional[Sequence[str]] = None,
        entities: Optional[Sequence[str]] = None,
    ) -> str:
        terms = list(keywords or [])
        if not terms:
            terms = self._query_terms(query)
        parts = [str(query or "").strip()]
        if retrieval_text and str(retrieval_text).strip() != str(query or "").strip():
            parts.append(f"retrieval: {str(retrieval_text).strip()}")
        if terms:
            parts.append(f"keywords: {' '.join(str(item) for item in terms)}")
        if entities:
            parts.append(f"entities: {' '.join(str(item) for item in entities)}")
        return "\n".join(parts)

    def _query_terms(self, text: str) -> List[str]:
        terms = self._keywords(text, limit=32)
        for phrase in re.findall(r"'([^']+)'|\"([^\"]+)\"", str(text or "")):
            clean = _compact_whitespace(phrase[0] or phrase[1]).lower()
            if clean and clean not in terms:
                terms.insert(0, clean)
        expanded: List[str] = []
        for term in terms:
            expanded.append(term)
            if re.fullmatch(r"[\u4e00-\u9fff]{3,}", term):
                for idx in range(0, len(term) - 1):
                    bigram = term[idx : idx + 2]
                    if bigram not in expanded:
                        expanded.append(bigram)
        terms = expanded
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

    @classmethod
    def _normalize_primary_entity(
        cls,
        value: Any,
        *,
        entities: Sequence[str],
        fact_subject: str,
    ) -> Optional[Dict[str, str]]:
        """Normalize the single entity used for entity-state assignment."""
        name = ""
        entity_type = "CONCEPT"
        if isinstance(value, dict):
            name = _compact_whitespace(value.get("name") or value.get("text") or "")
            entity_type = _compact_whitespace(value.get("type") or "CONCEPT").upper()
        else:
            name = _compact_whitespace(value)
        if not name:
            subject = str(fact_subject or "").strip().lower()
            if subject in {"user", "assistant"}:
                name = subject
                entity_type = "PERSON"
            elif entities:
                name = _compact_whitespace(entities[0])
        if not name:
            return None
        allowed_types = {
            "PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "PROJECT",
            "TECHNOLOGY", "CONCEPT", "TOPIC", "PREFERENCE", "OTHER",
        }
        if entity_type not in allowed_types:
            entity_type = "CONCEPT"
        return {"name": name, "type": entity_type}

    @staticmethod
    def _cal_raw_candidate_limits_from_recall_limits(
        final_candidate_limits: Dict[str, int],
        *,
        budget: str,
    ) -> Dict[str, int]:
        """Expand per-type final limits into broad raw candidate pools.

        The raw pool is intentionally independent for facts, states, and
        actionable items. This mirrors the voice_recording recall path: final
        output size must not determine how many records another memory type is
        allowed to contribute during first-pass retrieval.
        """
        multipliers = {"low": 8, "mid": 12, "high": 18}
        multiplier = multipliers.get(str(budget or "mid").lower(), 12)
        floors = {"facts": 40, "states": 28, "actionable_items": 24}
        ceilings = {"facts": 180, "states": 120, "actionable_items": 100}
        return {
            key: max(
                floors[key],
                min(
                    ceilings[key],
                    max(1, int(final_candidate_limits.get(key, 0) or 0)) * multiplier,
                ),
            )
            for key in ("facts", "states", "actionable_items")
        }

    def _final_recall_candidate_limits(
        self,
        *,
        query: str,
        top_k: int,
        budget: str,
        preferred_index_levels: Optional[Sequence[str]],
    ) -> Dict[str, int]:
        """Allocate independent post-ranking limits for each memory type."""
        k = max(1, int(top_k or 1))
        lower = str(query or "").lower()
        limits = {
            "facts": max(4, int(math.ceil(k * 0.60))),
            "states": max(3, int(math.ceil(k * 0.30))),
            "actionable_items": max(3, int(math.ceil(k * 0.25))),
        }
        if self._needs_broad_evidence(query):
            limits["facts"] = max(limits["facts"], int(math.ceil(k * 0.85)))
        preferred = set(preferred_index_levels or [])
        aliases = {"actionable": "actionable_items", "action": "actionable_items"}
        preferred = {aliases.get(level, level) for level in preferred}
        if preferred:
            for key in limits:
                if key.rstrip("s") in preferred or key in preferred:
                    limits[key] += max(1, int(math.ceil(k * 0.15)))
        if any(marker in lower for marker in ("todo", "task", "remind", "decision", "commit", "待办", "任务", "提醒", "决定", "承诺")):
            limits["actionable_items"] = max(limits["actionable_items"], int(math.ceil(k * 0.55)))
        if any(marker in lower for marker in ("prefer", "usually", "habit", "偏好", "通常", "习惯", "长期", "状态")):
            limits["states"] = max(limits["states"], int(math.ceil(k * 0.5)))
        return limits

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
