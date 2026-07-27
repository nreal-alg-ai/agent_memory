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
        **_: Any,
    ) -> bool:
        if not self._enabled:
            return False
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
        started_at = turns[0].get("turn_timestamp") or _now_text()
        ended_at = turns[-1].get("turn_timestamp") or started_at
        extracted = self._extract_memory_from_turns(turns)
        title = extracted.get("episode_title") or self._episode_title(turns)
        summary = extracted.get("episode_summary") or self._episode_summary(turns)
        tags = sorted({tag for turn in turns for tag in turn.get("tags", [])})
        episode_id = self._db.insert_episode(
            source_type="assistant_wakeup",
            episode_type="interaction",
            title=title,
            summary=summary,
            participants=["user", "assistant"],
            started_at=started_at,
            ended_at=ended_at,
            metadata={"tags": tags, "turn_count": len(turns)},
        )
        self._index_episode(
            episode_id=episode_id,
            title=title,
            summary=summary,
            started_at=started_at,
            ended_at=ended_at,
            tags=tags,
        )

        fact_count = 0
        facts = extracted.get("facts") or []
        if not facts:
            for turn_index, turn in enumerate(turns, 1):
                facts.extend(self._extract_turn_facts(turn, turn_index=turn_index))
        for fact in facts:
            self._store_fact(episode_id=episode_id, fact=fact, tags=tags)
            fact_count += 1
        return fact_count > 0

    def _extract_memory_from_turns(self, turns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract episode metadata and facts with LLM, falling back to heuristics."""
        data = self._extract_memory_with_llm(turns)
        if data and data.get("facts"):
            return data
        return {
            "episode_title": self._episode_title(turns),
            "episode_summary": self._episode_summary(turns),
            "facts": [],
        }

    def _extract_memory_with_llm(self, turns: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not self._llm_api_key or not self._llm_base_url or str(self._llm_base_url).lower() == "none":
            return None
        prompt_language = self._resolve_prompt_language(turns)
        prompt_template = (
            UNIFIED_MEMORY_EXTRACTION_PROMPT_EN
            if prompt_language == "en"
            else UNIFIED_MEMORY_EXTRACTION_PROMPT_ZH
        )
        prompt = prompt_template.replace(
            "{dialogue_batch}",
            self._build_dialogue_batch_for_prompt(
                turns,
                prompt_language=prompt_language,
            ),
        )
        for attempt in range(2):
            result = self._call_llm(prompt)
            parsed = self._parse_json_object_from_llm_text(result or "")
            if parsed is not None:
                normalized = self._normalize_llm_memory_payload(parsed, turns)
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
        turns: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        raw_facts = data.get("facts")
        if not isinstance(raw_facts, list):
            return None
        facts: List[Dict[str, Any]] = []
        fallback_timestamp = turns[0].get("turn_timestamp") if turns else _now_text()
        for index, raw_fact in enumerate(raw_facts, 1):
            if not isinstance(raw_fact, dict):
                continue
            text = _compact_whitespace(raw_fact.get("text") or raw_fact.get("summary") or "")
            if not text:
                continue
            priority = self._normalize_priority(raw_fact.get("priority", 70))
            if priority < 60:
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
            timestamp = (
                _compact_whitespace(raw_fact.get("occurred_start") or "")
                or _compact_whitespace(raw_fact.get("occurred_end") or "")
                or fallback_timestamp
            )
            facts.append({
                "summary": text,
                "source_text": text,
                "fact_subject": self._normalize_fact_subject(raw_fact.get("fact_subject")),
                "fact_kind": self._normalize_fact_kind(raw_fact.get("fact_kind")),
                "fact_type": self._normalize_fact_type(raw_fact.get("fact_type")),
                "time_key": f"{timestamp}#llm:{index:02d}",
                "keywords": " ".join(keywords),
                "entities": entities,
                "canonical_topics": [primary_topic],
                "importance": max(0.6, min(1.0, priority / 100.0)),
                "confidence": 0.9,
                "metadata": {
                    "extractor": "llm",
                    "priority": priority,
                    "occurred_start": _compact_whitespace(raw_fact.get("occurred_start") or ""),
                    "occurred_end": _compact_whitespace(raw_fact.get("occurred_end") or ""),
                    "time_confidence": _compact_whitespace(raw_fact.get("time_confidence") or "unknown"),
                    "where": _compact_whitespace(raw_fact.get("where") or ""),
                },
            })
        return {
            "episode_title": _compact_whitespace(data.get("episode_title") or self._episode_title(turns)),
            "episode_summary": _compact_whitespace(data.get("episode_summary") or self._episode_summary(turns)),
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

    def _index_episode(
        self,
        *,
        episode_id: int,
        title: str,
        summary: str,
        started_at: str,
        ended_at: str,
        tags: List[str],
    ) -> None:
        keywords = " ".join(self._keywords(summary, limit=24))
        embedding_text = f"{title}\n{summary}\nkeywords: {keywords}"
        embedding = self._embed(embedding_text)
        self._db.upsert_index_entry(
            source_type="assistant_wakeup",
            target_table="memory_episodes",
            target_id=episode_id,
            index_level="episode",
            memory_path="assistant_wakeup/episodes/interaction",
            title=title,
            summary_for_retrieval=summary,
            keywords=keywords,
            entities=["user", "assistant"],
            canonical_topics=self._topic_candidates(summary),
            participants=["user", "assistant"],
            time_start=started_at,
            time_end=ended_at,
            importance=0.55,
            confidence=0.8,
            embedding=embedding,
            embedding_text=embedding_text,
            metadata={"tags": tags},
        )

    def _extract_turn_facts(self, turn: Dict[str, Any], *, turn_index: int) -> List[Dict[str, Any]]:
        facts: List[Dict[str, Any]] = []
        timestamp = turn.get("turn_timestamp") or _now_text()
        user_text = turn.get("user_message") or ""
        assistant_text = turn.get("assistant_response") or ""
        if user_text:
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
        if assistant_text:
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

    def _store_fact(self, *, episode_id: int, fact: Dict[str, Any], tags: List[str]) -> int:
        embedding_text = "\n".join([
            fact["summary"],
            f"keywords: {fact['keywords']}",
            f"entities: {', '.join(fact['entities'])}",
        ])
        embedding = self._embed(embedding_text)
        fact_id = self._db.insert_fact(
            episode_id=episode_id,
            source_type="assistant_wakeup",
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
            source_type="assistant_wakeup",
            target_table="memory_facts",
            target_id=fact_id,
            index_level="fact",
            memory_path=f"assistant_wakeup/facts/{fact['fact_subject']}/{fact['fact_kind']}",
            title=fact["summary"][:96],
            summary_for_retrieval=fact["summary"],
            keywords=fact["keywords"],
            entities=fact["entities"],
            canonical_topics=fact["canonical_topics"],
            participants=["user", "assistant"],
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
            normalized.append({
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
            })
        return normalized

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
            token.strip("'\".,:;!?()[]{}")
            for token in tokens
            if len(token.strip("'\".,:;!?()[]{}")) > 1
            and token.strip("'\".,:;!?()[]{}") not in _STOPWORDS
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
            text = _compact_whitespace(item)
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
            if len(out) >= limit:
                break
        return out

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
