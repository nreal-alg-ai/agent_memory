#!/usr/bin/env python3
"""Unified memory manager inspired by MemPalace.

The public surface mirrors the current project's `MemoryNodeManager`, but the
internal model is deliberately unified:

1. Submitted source units become narrative `memory_facts`.
2. Longer source windows and their facts yield grounded user-activity observations.
3. Signal-bearing facts directly support user states; repeated action patterns live separately.
4. User intents share one schema for future intentions and evidence-backed
   lifecycle changes.

"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import time
import unicodedata
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from storage_utils import local_now_text, latest_timestamp, json_dumps

try:
    import jieba
except ImportError:  # pragma: no cover - exercised only in minimal installs
    jieba = None

from embedding_client import EmbeddingClient
from recall_embeddings import RecallEmbeddingIndexer
from llm_client import LLMClient
from intent_lifecycle import (
    USER_INTENT_FIELDS,
    normalize_user_intent_details,
    USER_INTENT_FIELD_ONLY_EVIDENCE_TYPES,
    USER_INTENT_LIFECYCLE_EVIDENCE_TYPES,
    USER_INTENT_STATUSES,
    user_intent_status_after_evidence,
)
from semantic_domains import SEMANTIC_DOMAINS
from .memory_database import UserMemoryDB
from .memory_types import MemoryUnit
from pwm_logging import CompactLogValue, as_pwm_logger
from .user_thread_updater import UserThreadUpdater
from task_scheduler import TaskScheduler
from prompts_en import (
    USER_OBSERVED_PATTERN_RECONCILIATION_PROMPT_EN,
    USER_OBSERVED_PATTERN_SIGNAL_CLUSTER_RELATION_PROMPT_EN,
    USER_OBSERVED_PATTERN_EXTRACTION_PROMPT_EN,
    USER_INTENT_UPDATE_FROM_FUTURE_INTENT_SIGNALS_PROMPT_EN,
    USER_INTENT_UPDATE_FROM_INTENT_LIFECYCLE_OBSERVATIONS_PROMPT_EN,
    USER_INTENT_LIFECYCLE_FROM_ACTIVITIES_PROMPT_EN,
    USER_ACTIVITY_EXTRACTION_PROMPT_EN,
    UNIFIED_MEMORY_EXTRACTION_PROMPT_EN,
    PROACTIVE_SERVICE_FEEDBACK_EXTRACTION_GUIDANCE_EN,
)
from prompts_zh import (
    USER_OBSERVED_PATTERN_RECONCILIATION_PROMPT_ZH,
    USER_OBSERVED_PATTERN_SIGNAL_CLUSTER_RELATION_PROMPT_ZH,
    USER_OBSERVED_PATTERN_EXTRACTION_PROMPT_ZH,
    USER_INTENT_UPDATE_FROM_FUTURE_INTENT_SIGNALS_PROMPT_ZH,
    USER_INTENT_UPDATE_FROM_INTENT_LIFECYCLE_OBSERVATIONS_PROMPT_ZH,
    USER_INTENT_LIFECYCLE_FROM_ACTIVITIES_PROMPT_ZH,
    USER_ACTIVITY_EXTRACTION_PROMPT_ZH,
    UNIFIED_MEMORY_EXTRACTION_PROMPT_ZH,
    PROACTIVE_SERVICE_FEEDBACK_EXTRACTION_GUIDANCE_ZH,
)
from .utils import _cal_embedding_cosine_similarity
from .user_action_similarity import action_lexical_similarity

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

_ORDINARY_TIME_ENTITY_PATTERNS = (
    r"^(今天|昨天|前天|明天|后天|上周|本周|下周|上个月|这个月|下个月|最近|近期)$",
    r"^最近\d+(天|周|个月|月|年)$",
    r"^过去\d+(天|周|个月|月|年)$",
    r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$",
    r"^\d{1,2}:\d{2}(?::\d{2})?$",
    r"^\d+(分钟|小时|天|周|个月|月|年)$",
    r"^(today|yesterday|tomorrow|last week|this week|next week|last month|this month|next month|recently|lately)$",
    r"^(last|past|previous|next)\s+\d+\s+(day|days|week|weeks|month|months|year|years)$",
    r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$",
)

_ATTRIBUTE_ONLY_ENTITY_PATTERNS = (
    r"^(低|高|强|弱|轻|重|小|大|快|慢|短|长|稳定|灵活|固定|频繁|高频|低频|长期|短期).{0,8}$",
    r"^(low|high|strong|weak|light|heavy|fast|slow|short|long|stable|flexible|fixed|frequent)\s+[\w -]{0,24}$",
)

# ``用户本人`` / ``user self`` identify the wearer whose personal world model
# this database represents. They are intentionally distinct from ordinary
# mentions of product users, customers, or consumers in ambient recordings.
_WORLD_OWNER_ENTITY_REFERENCE_ALIASES = {
    "用户本人", "user self", "user_self",
}
_AMBIGUOUS_USER_ENTITY_ALIASES = {
    "用户", "user", "the user",
}


def _now_text() -> str:
    return local_now_text()


def _to_timestamp_text(value: Any) -> str:
    if isinstance(value, datetime):
        # The LongMemEval scripts pass time filters as "YYYY-MM-DD HH:MM:SS".
        # Keep the same sortable format so SQLite string range filters work.
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value or "").strip()


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


class MemoryNodeManager:
    """Compatibility manager backed by a unified index-first memory line."""

    def __init__(
        self,
        db: UserMemoryDB,
        *,
        embedding_config: Optional[Dict[str, Any]] = None,
        memory_manager_config: Optional[Dict[str, Any]] = None,
        task_scheduler: TaskScheduler,
        llm_client: LLMClient,
        logger: Optional[logging.Logger | logging.LoggerAdapter] = None,
    ) -> None:
        self._db = db
        self._logger = as_pwm_logger(logger or logging.getLogger(__name__))
        self._task_scheduler = task_scheduler
        self._llm_client = llm_client
        self._memory_cfg = dict(memory_manager_config or {})
        self._embedding_cfg = dict(embedding_config or {})
        self._memory_prompt_language = str(
            self._memory_cfg.get("memory_prompt_language_mode")
            or "source"
        )
        self._memory_enabled = bool(self._memory_cfg.get("memory_enabled", True))
        configured_world_owner_name = _compact_whitespace(
            self._memory_cfg.get("world_owner_entity_name") or ""
        )
        if configured_world_owner_name.lower() in _AMBIGUOUS_USER_ENTITY_ALIASES:
            raise ValueError(
                "memory.world_owner_entity_name must use '用户本人', "
                "'user self', or a concrete owner name; bare '用户' / 'user' "
                "is ambiguous with product users and consumers."
            )
        self._world_owner_entity_name = (
            configured_world_owner_name or "用户本人"
        )
        self._user_observed_pattern_min_activities = max(
            2, int(self._memory_cfg.get("user_observed_pattern_min_activities") or 2),
        )
        self._embedding_client: Optional[EmbeddingClient] = None
        self._recall_embedding_indexer: Optional[RecallEmbeddingIndexer] = None
        self._user_thread_updater = UserThreadUpdater(
            memory_db=self._db, llm_client=self._llm_client,
            resolve_language=self._resolve_prompt_language_from_text,
            log=lambda scope, event, payload: self._logger.log_json_format(scope, event, payload),
            batch_size=self._memory_cfg.get("user_thread_activity_batch_size", 8),
            candidate_limit=self._memory_cfg.get("user_thread_candidate_limit", 12),
            event_context_limit=self._memory_cfg.get("user_thread_event_context_limit", 16),
        )
        self._task_scheduler.register_handler(
            "memory_embedding_cache_refresh", self._generate_memory_recall_document_embeddings,
        )
        self._task_scheduler.register_handler(
            "memory_fact_extraction", self._process_memory_fact_extraction_task,
        )
        self._task_scheduler.register_handler(
            "memory_user_activity_extraction", self._process_memory_user_activity_extraction_task,
        )
        self._task_scheduler.register_handler(
            "memory_user_thread_update", self._process_memory_user_thread_update_task,
        )
        self._task_scheduler.register_handler(
            "memory_user_observed_pattern_update",
            self._process_memory_user_observed_pattern_update_task,
        )
        self._task_scheduler.register_handler(
            "memory_user_intent_update", self._process_memory_user_intent_update_task,
        )
        self._task_scheduler.register_handler(
            "memory_user_intent_update_from_activities",
            self._process_memory_user_intent_update_from_activities_task,
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

    def _ensure_embedding_client(self) -> bool:
        if self._embedding_client is None:
            self._embedding_client = EmbeddingClient(self._embedding_cfg)
        return True

    # ── Store path: source units -> facts -> user activity ───────────────

    @property
    def enabled(self) -> bool:
        return self._memory_enabled

    @property
    def world_owner_entity_name(self) -> str:
        return self._world_owner_entity_name

    def set_logger(self, logger: Optional[logging.Logger | logging.LoggerAdapter]) -> None:
        """Set the logger used by memory extraction and persistence."""
        self._logger = as_pwm_logger(logger or logging.getLogger(__name__))

    def set_embedding_client(self, embedding_client: Optional[EmbeddingClient]) -> None:
        """Share a pre-initialized embedding client with memory runtime callers."""
        self._embedding_client = embedding_client

    def get_embedding_client(self) -> EmbeddingClient:
        """Share the same embedding client with the PWM recall coordinator."""
        self._ensure_embedding_client()
        return self._embedding_client

    def initialize_embedding_cache(self) -> None:
        """Invalidate incompatible profiles before readers and workers start."""
        if self._get_recall_embedding_indexer().initialize():
            self._task_scheduler.submit("memory_embedding_cache_refresh", {"force": True})

    def _generate_memory_recall_document_embeddings(self, *, force: bool = False) -> Dict[str, Any]:
        return self._get_recall_embedding_indexer().generate(force=force)

    def set_recall_embedding_indexer(self, indexer: RecallEmbeddingIndexer) -> None:
        self._recall_embedding_indexer = indexer

    def _get_recall_embedding_indexer(self) -> RecallEmbeddingIndexer:
        if self._recall_embedding_indexer is None:
            self._recall_embedding_indexer = RecallEmbeddingIndexer(
                databases=[self._db], client_provider=self.get_embedding_client,
                enabled=self._embedding_cfg.get("enabled", True), logger=self._logger,
            )
        return self._recall_embedding_indexer

    def _process_memory_user_thread_update_task(
        self, *, activity_ids: List[int], reference_timestamp: Any = None,
    ) -> Dict[str, Any]:
        """Resolve activity membership first, then incrementally update matters."""
        return self._user_thread_updater.process_update_task(
            activity_ids=activity_ids, reference_timestamp=reference_timestamp,
        )

    def _process_memory_user_activity_extraction_task(
        self,
        *,
        tags: List[str],
        reference_timestamp: Any = None,
    ) -> Dict[str, Any]:
        source_units = self._db.get_unprocessed_memory_source_segments_for_user_activity(
            limit=240,
            reference_timestamp=reference_timestamp,
        )
        if not source_units:
            return {
                "status": "empty",
                "reason": "no_unprocessed_source_units",
                "new_user_activity_count": 0,
                "fact_count": 0,
                "source_unit_count": 0,
                "source_segment_ids": [],
            }
        source_row_ids = list(dict.fromkeys(
            int(unit.source_unit_row_id)
            for unit in source_units
            if unit.source_unit_row_id is not None
        ))
        facts = self._db.get_memory_facts_for_source_segments(
            source_segment_ids=source_row_ids, limit=240,
        )
        activity_items = self._extract_user_activity_from_source_units(
            source_units=source_units,
            facts=facts,
            reference_timestamp=reference_timestamp,
        )
        if activity_items is None:
            return {"status": "failed", "reason": "invalid_user_activity_response"}
        source_started_at = _compact_whitespace(source_units[0].start_time) or _now_text()
        source_ended_at = latest_timestamp(
            entry["ended_at"] for unit in source_units for entry in unit.entries()
        ) or source_started_at
        fact_ids = {int(fact["id"]) for fact in facts}
        activity_ids: List[int] = []
        with self._db.transaction():
            for item in activity_items:
                supported_fact_ids = [
                    int(value) for value in item.get("evidence_fact_ids") or []
                    if str(value).isdigit() and int(value) in fact_ids
                ]
                activity_id = self._db.insert_user_activity_item(
                    activity_text=item["activity_text"],
                    user_role=item["user_role"],
                    confidence=item["confidence"],
                    started_at=source_started_at,
                    ended_at=source_ended_at,
                    evidence_fact_ids=supported_fact_ids,
                    source_segment_ids=source_row_ids,
                    metadata={"tags": list(tags or []), "time_scope": "source_window"},
                    reference_timestamp=reference_timestamp,
                )
                activity_ids.append(activity_id)
                if item.get("user_action_signals"):
                    owner_entity_id = self._intent_world_owner_entity_id(reference_timestamp=reference_timestamp)
                    self._db.upsert_user_action_signals([
                        {**signal, "source_type": "user_activity", "source_id": activity_id,
                         "subject_entity_id": owner_entity_id}
                        for signal in item["user_action_signals"]
                    ], reference_timestamp=reference_timestamp)
            self._upsert_memory_user_activities_into_recall_document_table(
                activity_ids, reference_timestamp=reference_timestamp,
            )
            processed_rows = self._db.mark_memory_source_segments_processed_for_user_activity(
                source_segment_ids=source_row_ids,
            )
        if activity_ids:
            self._generate_memory_recall_document_embeddings()
        report = {
            "status": "ok",
            "new_user_activity_count": len(activity_ids),
            "user_activity_item_ids": activity_ids,
            "user_action_signal_ids": [
                int(signal["signal_id"])
                for signal in self._db.get_user_signals_by_source_ids(
                    "action", source_type="user_activity", source_ids=activity_ids,
                )
            ],
            "user_action_signal_count": sum(len(item.get("user_action_signals") or []) for item in activity_items),
            "user_action_signal_activity_count": sum(bool(item.get("user_action_signals")) for item in activity_items),
            "reference_timestamp": reference_timestamp or source_ended_at,
            "source_segment_ids": source_row_ids,
            "fact_count": len(facts),
            "source_unit_count": len(source_units),
            "source_unit_row_count": processed_rows,
        }
        self._logger.log_json_format("memory_user_activity_extraction", "finish", report)
        return report

    def _process_memory_fact_extraction_task(
        self,
        *,
        units: List[MemoryUnit],
        preceding_context_units: Optional[List[MemoryUnit]] = None,
        tags: List[str],
        feedback_context: Optional[Dict[str, Any]] = None,
        reference_timestamp: Any = None,
    ) -> Dict[str, Any]:
        store_started_at = time.monotonic()
        stage_name = (
            "memory_proactive_service_feedback_extraction"
            if feedback_context is not None else "memory_fact_extraction"
        )
        units = self._valid_memory_units(units)
        preceding_units = list(preceding_context_units or [])
        if not units:
            elapsed_ms = round((time.monotonic() - store_started_at) * 1000, 2)
            report = {
                "status": "empty",
                "reason": "no_units",
                "attempt_count": 0,
                "new_fact_count": 0,
                "fact_ids": [],
                "source_unit_row_ids": [],
                "source_unit_count": 0,
                "source_unit_row_count": 0,
                "total_elapsed_ms": elapsed_ms,
            }
            self._logger.log_json_format(stage_name, "finish", report)
            return report
        # Feedback's authoritative raw source is its dedicated event record.
        # Do not add tool reports to the ordinary activity observation window.
        source_unit_row_ids = []
        if feedback_context is None:
            with self._db.transaction():
                source_unit_row_ids = self._db.insert_memory_source_segments(units=units, reference_timestamp=reference_timestamp)
        fact_info = self._extract_memory_fact_from_units(
            units,
            preceding_context_units=preceding_units,
            **({"feedback_context": feedback_context} if feedback_context is not None else {}),
            reference_timestamp=reference_timestamp,
        )
        if fact_info["status"] != "ok":
            report = {
                "status": fact_info["status"],
                "reason": fact_info["reason"],
                "attempt_count": fact_info["attempt_count"],
                "new_fact_count": 0,
                "fact_ids": [],
                "source_unit_row_ids": source_unit_row_ids,
                "source_unit_count": len(units),
                "source_unit_row_count": len(source_unit_row_ids),
                "total_elapsed_ms": round((time.monotonic() - store_started_at) * 1000, 2),
            }
            if fact_info.get("error_type"):
                report["error_type"] = fact_info["error_type"]
            self._logger.log_json_format(stage_name, "finish", report)
            return report
        facts = list(fact_info.get("facts") or [])
        if feedback_context is not None:
            for fact in facts:
                fact["metadata"].update({
                    "source": "proactive_service_feedback",
                    **{key: feedback_context[key] for key in (
                        "feedback_id", "service_id", "execution_id", "prediction_id",
                    )},
                })
        self._log_extracted_fact_info(facts=facts)
        with self._db.transaction():
            save_fact_info = self._store_extracted_memory_facts_into_db(
                facts=facts,
                tags=tags,
                reference_timestamp=reference_timestamp,
            )
            self._db.insert_source_mappings(
                fact_ids=save_fact_info.get("fact_ids") or [],
                source_segment_ids=source_unit_row_ids,
            )
        self._generate_memory_recall_document_embeddings()
        report = {
            "status": fact_info["status"],
            "reason": fact_info["reason"],
            "attempt_count": fact_info["attempt_count"],
            "new_fact_count": len(list(save_fact_info.get("fact_ids") or [])),
            "fact_ids": list(save_fact_info.get("fact_ids") or []),
            "user_claim_signal_fact_count": int(
                save_fact_info.get("user_claim_signal_fact_count") or 0
            ),
            "user_claim_signal_count": int(
                save_fact_info.get("user_claim_signal_count") or 0
            ),
            "user_action_signal_fact_count": int(
                save_fact_info.get("user_action_signal_fact_count") or 0
            ),
            "user_action_signal_count": int(
                save_fact_info.get("user_action_signal_count") or 0
            ),
            "user_intent_signal_fact_count": int(
                save_fact_info.get("user_intent_signal_fact_count") or 0
            ),
            "user_intent_signal_count": int(
                save_fact_info.get("user_intent_signal_count") or 0
            ),
            "source_unit_row_ids": source_unit_row_ids,
            "source_unit_count": len(units),
            "source_unit_row_count": len(source_unit_row_ids),
            "total_elapsed_ms": round((time.monotonic() - store_started_at) * 1000, 2),
        }
        self._logger.log_json_format(stage_name, "finish", report)
        return report

    def extract_memory_from_proactive_service_feedback(
        self, feedback: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process actual feedback sources directly, bypassing Runtime windows.

        MemoryUnit is reused only as the source/evidence representation. This
        does not inject a synthetic conversation into Runtime or trigger an
        activity/thread observation for an Agent's tool execution.
        """
        sources = []
        if feedback["user_feedback"].strip():
            sources.append(("speaker_用户本人", feedback["user_feedback"]))
        if feedback["execution_result"]:
            sources.append(("speaker_执行结果报告", json.dumps(
                feedback["execution_result"], ensure_ascii=False,
            )))
        if not sources:
            return {"status": "empty", "reason": "no_memory_evidence", "fact_ids": [],
                    "new_fact_count": 0, "attempt_count": 0}
        timestamp = feedback["observed_at"]
        tags = ["proactive_service_feedback", f"service_feedback:{feedback['id']}"]
        units = [MemoryUnit(
            unit_type="transcript", texts=[text], speakers=[speaker],
            started_ats=[timestamp], ended_ats=[timestamp],
            token_count=max(1, len(text) // 2), tags=tags,
        ) for speaker, text in sources]
        context = {
            "feedback_id": feedback["id"], "service_id": feedback["service_id"],
            "execution_id": feedback["execution_id"],
            "prediction_id": feedback["context"]["service"]["prediction_id"],
            "execution_status": feedback["execution_status"],
            "user_response": feedback["user_response"],
            "observed_at": timestamp, "agent_note": feedback["agent_note"],
            **feedback["context"],
        }
        return self._process_memory_fact_extraction_task(
            units=units, tags=tags, feedback_context=context, reference_timestamp=timestamp,
        )

    @staticmethod
    def _valid_memory_units(
        units: Sequence[MemoryUnit],
    ) -> List[MemoryUnit]:
        """Keep task units with usable text, without rebuilding their order."""
        return [
            unit
            for unit in units or ()
            if isinstance(unit, MemoryUnit)
            and _compact_whitespace(unit.text)
        ]

    def _upsert_memory_recall_document(
        self,
        *,
        object_type: str,
        object_id: int,
        identity_lines: Sequence[str],
        title: str = "",
        summary: str = "",
        entity_ids: Optional[Sequence[int]] = None,
        topic_keys: Optional[Sequence[str]] = None,
        time_start: str = "",
        time_end: str = "",
        status: str = "",
        confidence: float = 0.0,
        importance: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        reference_timestamp: Any = None,
    ) -> int:
        """Write one canonical retrieval projection without ranking it.

        The source tables own domain state. This helper owns only the stable
        textual representation shared by lexical search and embeddings.
        """
        identity_text = self._format_recall_document_identity_text(identity_lines)
        return self._db.upsert_memory_recall_document(
            object_type=object_type,
            object_id=object_id,
            identity_text=identity_text,
            title=title,
            summary=summary,
            entity_ids=entity_ids,
            topic_keys=topic_keys,
            time_start=time_start,
            time_end=time_end,
            status=status,
            confidence=confidence,
            importance=importance,
            metadata=metadata,
            reference_timestamp=reference_timestamp,
        )

    @staticmethod
    def _format_recall_document_identity_text(identity_lines: Sequence[str]) -> str:
        return "\n".join(
            _compact_whitespace(line)
            for line in identity_lines
            if _compact_whitespace(line)
        )

    def _upsert_memory_facts_into_recall_document_table(
        self, fact_ids: Sequence[int], *, reference_timestamp: Any = None,
    ) -> None:
        """Project persisted facts into lexical documents; vectors are filled later."""
        for fact in self._db.get_memory_facts_by_ids(fact_ids):
            fact_topic = fact["fact_topic"]
            time_key = fact.get("event_time_key") or fact.get("dialogue_time_key") or ""
            self._upsert_memory_recall_document(
                object_type="fact", object_id=fact["id"],
                identity_lines=[
                    f"summary: {_compact_whitespace(fact['summary'])}",
                    f"keywords: {' '.join(str(value) for value in fact['keywords'] if value)}",
                    f"entities: {', '.join(fact.get('entities') or [])}",
                    f"fact_topic: {fact_topic}",
                ],
                title=_compact_whitespace(fact["summary"])[:120], summary=fact["summary"],
                entity_ids=fact.get("entity_ids") or [], topic_keys=[fact_topic],
                time_start=time_key, time_end=time_key,
                confidence=float(fact["confidence"]), importance=float(fact["importance"]),
                metadata={"projection_version": "v1"}, reference_timestamp=reference_timestamp,
            )

    def _upsert_memory_user_activities_into_recall_document_table(
        self, activity_ids: Sequence[int], *, reference_timestamp: Any = None,
    ) -> None:
        """Project stored activities without expanding their facts or raw sources."""
        activities = self._db.get_user_activity_items_by_ids(activity_ids)
        if not activities:
            return
        owner_entity_id = self._intent_world_owner_entity_id(reference_timestamp=reference_timestamp)
        for activity in activities:
            self._upsert_memory_recall_document(
                object_type="user_activity", object_id=activity["id"],
                identity_lines=[
                    f"activity_text: {activity['activity_text']}",
                    f"user_role: {activity['user_role']}",
                ],
                title=_compact_whitespace(activity["activity_text"])[:120],
                summary=activity["activity_text"], entity_ids=[owner_entity_id],
                time_start=activity.get("started_at") or "",
                time_end=activity.get("ended_at") or "",
                confidence=float(activity["confidence"]),
                metadata={"projection_version": "v1", "user_role": activity["user_role"]},
                reference_timestamp=reference_timestamp,
            )

    def _upsert_memory_user_intents_into_recall_document_table(
        self, intent_ids: Sequence[int], *, reference_timestamp: Any = None,
    ) -> None:
        """Project current persisted intents; history remains in intent events."""
        for intent in self._db.get_user_intent_items_by_ids(intent_ids):
            details = intent["details"]
            identity_lines = [f"summary: {intent['summary']}"]
            if details.get("user_role"):
                identity_lines.append(f"user_role: {details['user_role']}")
            if intent.get("current_progress"):
                identity_lines.append(f"current_progress: {intent['current_progress']}")
            for key in ("completion_condition", "location_text"):
                if details.get(key):
                    identity_lines.append(f"{key}: {details[key]}")
            if details.get("participants"):
                identity_lines.append(f"participants: {', '.join(details['participants'])}")
            self._upsert_memory_recall_document(
                object_type="user_intent", object_id=intent["id"],
                identity_lines=identity_lines,
                title=_compact_whitespace(intent["summary"])[:120], summary=intent["summary"],
                entity_ids=[intent["world_owner_entity_id"]], status=intent["status"],
                time_start=intent.get("time_start") or "",
                time_end=intent.get("time_end") or "",
                confidence=float(intent["confidence"]),
                metadata={
                    "projection_version": "v1", "primary_domain": intent["primary_domain"],
                    **{key: details[key] for key in (
                        "time_precision", "time_start_meaning", "time_end_meaning",
                    ) if details.get(key)},
                },
                reference_timestamp=reference_timestamp,
            )

    def _upsert_memory_user_observed_patterns_into_recall_document_table(
        self, pattern_ids: Sequence[int], *, reference_timestamp: Any = None,
    ) -> None:
        """Project persisted patterns and their conditions without expanding evidence."""
        for pattern in self._db.get_user_observed_patterns_by_ids(pattern_ids):
            self._upsert_memory_recall_document(
                object_type="user_observed_pattern", object_id=pattern["id"],
                identity_lines=[
                    f"pattern: {pattern['pattern_text']}",
                    *([f"condition: {pattern['condition_text']}"] if pattern["condition_text"] else []),
                ],
                title=_compact_whitespace(pattern["pattern_text"])[:120],
                summary=pattern["pattern_text"], entity_ids=[pattern["subject_entity_id"]],
                confidence=float(pattern["confidence"]),
                metadata={"projection_version": "v1", "primary_domain": pattern["primary_domain"]},
                reference_timestamp=reference_timestamp,
            )

    def _log_extracted_fact_info(
        self,
        *,
        facts: List[Dict[str, Any]],
    ) -> None:
        if not facts:
            return
        for index, fact in enumerate(facts, 1):
            metadata = fact.get("metadata") if isinstance(fact.get("metadata"), dict) else {}
            self._logger.log_json_format(
                "memory_fact_extraction",
                "fact_info",
                {
                    "fact_index": index,
                    "fact_count": len(facts),
                    "summary": fact.get("summary") or "",
                    "fact_type": fact.get("fact_type"),
                    "event_time_key": fact.get("event_time_key") or "",
                    "dialogue_time_key": fact.get("dialogue_time_key") or "",
                    "keywords": fact.get("keywords") or "",
                    "entities": fact.get("entities") or [],
                    "fact_topic": fact.get("fact_topic") or "",
                    "user_claim_signals": fact.get("user_claim_signals") or [],
                    "user_action_signals": fact.get("user_action_signals") or [],
                    "user_intent_signals": fact.get("user_intent_signals") or [],
                    "importance": fact.get("importance"),
                    "confidence": fact.get("confidence"),
                    "time_confidence": metadata.get("time_confidence") or "",
                    "where": metadata.get("where") or "",
                    "metadata": metadata,
                    "batch_fact_index": index,
                    "batch_fact_count": len(facts),
                },
            )

    def _extract_memory_fact_from_units(
        self,
        units: Sequence[MemoryUnit],
        *,
        preceding_context_units: Optional[Sequence[MemoryUnit]] = None,
        feedback_context: Optional[Dict[str, Any]] = None,
        reference_timestamp: Any = None,
    ) -> Dict[str, Any]:
        """Return ok/empty/failed, facts, a reason and the extraction attempt count."""
        current_units = self._valid_memory_units(units)
        if not current_units:
            return {"status": "empty", "facts": [], "reason": "no_units", "attempt_count": 0}
        preceding_units = self._valid_memory_units(preceding_context_units or [])
        prompt_language = self._resolve_prompt_language_from_memory_units(
            [*current_units, *preceding_units],
        )
        stage_name = (
            "memory_proactive_service_feedback_extraction"
            if feedback_context is not None else "memory_fact_extraction"
        )
        self._logger.log_json_format(stage_name, "start", {
            "source_unit_count": len(current_units),
            "preceding_context_unit_count": len(preceding_units),
            "current_units": self._memory_units_log_view(current_units),
            "preceding_context_units": self._memory_units_log_view(preceding_units),
        })
        prompt_template = (
            UNIFIED_MEMORY_EXTRACTION_PROMPT_EN
            if prompt_language == "en"
            else UNIFIED_MEMORY_EXTRACTION_PROMPT_ZH
        )
        known_world_owner_aliases = [
            item["alias_name"]
            for item in self._db.get_owner_entity_aliases(
                self._intent_world_owner_entity_id(reference_timestamp=reference_timestamp),
                status="confirmed",
                limit=128,
            )
        ]
        if feedback_context is not None:
            # The service is already verified as belonging to this wearer.
            # This permits attribution without labelling Agent output as user speech.
            known_world_owner_aliases = [self._world_owner_entity_name, *known_world_owner_aliases]
        prompt = (
            prompt_template
            .replace(
                "{known_world_owner_aliases}",
                json.dumps(known_world_owner_aliases[:32], ensure_ascii=False),
            )
            .replace(
                "{dialogue_batch}",
                self._format_memory_units_view_for_fact_prompt(
                    current_units,
                    prompt_language=prompt_language,
                ),
            )
            .replace(
                "{preceding_context}",
                self._format_memory_units_view_for_fact_prompt(
                    preceding_units,
                    prompt_language=prompt_language,
                ) or ("(none)" if prompt_language == "en" else "（无）"),
            )
        )
        if feedback_context is not None:
            guidance = (
                PROACTIVE_SERVICE_FEEDBACK_EXTRACTION_GUIDANCE_EN
                if prompt_language == "en" else PROACTIVE_SERVICE_FEEDBACK_EXTRACTION_GUIDANCE_ZH
            )
            prompt += "\n\n" + guidance.replace(
                "{feedback_context}", json.dumps(feedback_context, ensure_ascii=False),
            )
        failure_info: Dict[str, Any] = {}
        for attempt in range(2):
            try:
                result = self._llm_client.call_llm(prompt, stage_name=stage_name)
            except Exception as exc:
                failure_info = {"reason": "llm_request_error", "error_type": type(exc).__name__}
                continue
            response_text = str(result or "")
            if not response_text.strip():
                failure_info = {"reason": "empty_llm_response"}
                if attempt == 0:
                    self._logger.debug("Unified memory LLM extraction failed, retrying")
                continue
            parsed = self._llm_client.parse_json_object_from_llm_text(result or "", stage_name=stage_name)
            if parsed is None:
                failure_info = {"reason": "invalid_llm_json_object"}
            else:
                normalization_diagnostics: Dict[str, Any] = {}
                normalized = self._normalize_memory_fact_extraction_llm_output(
                    parsed,
                    current_units,
                    prompt_language=prompt_language,
                    known_world_owner_aliases=known_world_owner_aliases,
                    diagnostics=normalization_diagnostics,
                )
                if normalized is not None:
                    facts = normalized["facts"]
                    return {
                        "status": "ok" if facts else "empty",
                        "facts": facts,
                        "reason": (
                            "" if facts else "no_retained_facts" if parsed["facts"] else "no_facts"
                        ),
                        "attempt_count": attempt + 1,
                    }
                failure_info = {
                    "reason": normalization_diagnostics.get("normalization_status")
                    or "invalid_llm_fact_response",
                }
            if attempt == 0:
                self._logger.debug("Unified memory LLM extraction failed, retrying")
        return {"status": "failed", "facts": [], "attempt_count": 2, **failure_info}

    def _extract_user_activity_from_source_units(
        self,
        *,
        source_units: Sequence[MemoryUnit],
        facts: Sequence[Dict[str, Any]],
        reference_timestamp: Any = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Observe what the user is doing across a source window.

        Raw units are authoritative; facts provide compact evidence anchors.
        An empty list is valid when the window does not establish user activity.
        """
        unit_list = self._valid_memory_units(source_units)
        if not unit_list:
            return []
        prompt_language = self._resolve_prompt_language_from_memory_units(
            unit_list,
        )
        fact_list = [
            item for item in facts or []
            if isinstance(item, dict)
            and _compact_whitespace(item.get("summary") or item.get("text") or "")
        ]
        source_text = self._build_user_activity_units_for_prompt(
            unit_list,
            prompt_language=prompt_language,
        )
        prompt_template = (
            USER_ACTIVITY_EXTRACTION_PROMPT_EN
            if prompt_language == "en"
            else USER_ACTIVITY_EXTRACTION_PROMPT_ZH
        )
        prompt = (
            prompt_template
            .replace("{source_units}", source_text)
            .replace(
                "{evidence_facts}",
                self._format_facts_view_for_user_activity_prompt(fact_list),
            )
        )
        allowed_fact_ids = {
            int(item["id"]) for item in fact_list
            if str(item.get("id") or "").isdigit()
        }
        for _ in range(2):
            parsed = self._llm_client.parse_json_object_from_llm_text(
                self._llm_client.call_llm(prompt, stage_name='memory_user_activity_extraction') or "", stage_name='memory_user_activity_extraction'
            ) or {}
            raw_items = parsed.get("user_activities")
            if not isinstance(raw_items, list):
                continue
            activities: List[Dict[str, Any]] = []
            seen: set[str] = set()
            for raw in raw_items[:12]:
                if not isinstance(raw, dict):
                    continue
                activity_text = _compact_whitespace(raw.get("activity_text") or "")
                user_role = str(raw.get("user_role") or "").strip().lower()
                if not activity_text or user_role not in {"actor", "participant", "contextual"}:
                    continue
                if activity_text in seen:
                    continue
                seen.add(activity_text)
                raw_evidence_ids = raw.get("evidence_fact_ids")
                if not isinstance(raw_evidence_ids, list):
                    raw_evidence_ids = []
                confidence = self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.5)
                if not math.isfinite(confidence):
                    confidence = 0.5
                activities.append({
                    "activity_text": activity_text,
                    "user_role": user_role,
                    "confidence": confidence,
                    "evidence_fact_ids": list(dict.fromkeys(
                        int(value) for value in raw_evidence_ids
                        if str(value).isdigit() and int(value) in allowed_fact_ids
                    )),
                    "user_action_signals": self._normalize_activity_user_action_signals(
                        raw.get("user_action_signals"), source_units=unit_list,
                        reference_timestamp=reference_timestamp,
                    ),
                })
            return activities
        return None

    def _normalize_activity_user_action_signals(
        self, value: Any, *, source_units: Sequence[MemoryUnit],
        reference_timestamp: Any = None,
    ) -> List[Dict[str, Any]]:
        """Validate window observations against exact, attributed source excerpts.

        Referenced entry times bound the observation, not the action's exact
        onset/duration. Reported actions continue through the fact path.
        """
        if not isinstance(value, list) or not value:
            return []
        owner_name = self._world_owner_entity_name
        aliases = [owner_name]
        aliases.extend(row["alias_name"] for row in self._db.get_owner_entity_aliases(
            self._intent_world_owner_entity_id(reference_timestamp=reference_timestamp), status="confirmed", limit=128,
        ))
        signals: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for raw in value:
            if not isinstance(raw, dict):
                continue
            if not self._is_world_owner_entity_reference(raw.get("subject_entity")):
                continue
            kind = str(raw.get("action_kind") or "")
            text = _compact_whitespace(raw.get("action_text") or "")[:480]
            basis = _compact_whitespace(raw.get("evidence_basis") or "")[:480]
            evidence = raw.get("source_evidence")
            if (kind not in {"action", "choice", "participation"} or not text or not basis
                    or not isinstance(evidence, list) or not evidence):
                continue
            refs: List[Dict[str, Any]] = []
            valid = True
            for ref in evidence:
                if not isinstance(ref, dict):
                    valid = False
                    break
                try:
                    unit_index, entry_index = int(ref["unit_index"]), int(ref["entry_index"])
                    if unit_index < 1 or entry_index < 1:
                        raise ValueError("source indices must be positive")
                    unit = source_units[unit_index - 1]
                    entry = unit.entries()[entry_index - 1]
                except (KeyError, ValueError, TypeError, IndexError):
                    valid = False
                    break
                quote = _compact_whitespace(ref.get("quote") or "")
                if not quote or quote not in _compact_whitespace(entry["text"]):
                    valid = False
                    break
                resolved = {
                    "source_unit_row_id": unit.source_unit_row_id,
                    "unit_index": unit_index, "entry_index": entry_index,
                    "speaker": entry["speaker"], "quote": quote,
                    "started_at": entry["started_at"], "ended_at": entry["ended_at"],
                }
                if resolved not in refs:
                    refs.append(resolved)
            if not valid or not any(
                ref["speaker"] == "speaker_用户本人" or (
                    ref["speaker"] != "speaker_助手" and any(
                        alias.casefold() in ref["quote"].casefold() for alias in aliases
                    )
                ) for ref in refs
            ):
                self._logger.debug("Activity action rejected: ungrounded or owner-unattributed source evidence")
                continue
            refs.sort(key=lambda ref: (ref["unit_index"], ref["entry_index"]))
            fingerprint = hashlib.sha256(json.dumps(
                [kind, text.casefold(), refs], ensure_ascii=False, sort_keys=True,
            ).encode("utf-8")).hexdigest()[:16]
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            signals.append({
                "signal_key": f"activity|{fingerprint}",
                "subject_entity": owner_name, "action_kind": kind, "action_text": text,
                "evidence_basis": basis, "owner_evidence_type": "observed_activity",
                "owner_evidence_text": "", "other_evidence_text": "",
                "time_scope": "source_window",
                "event_time_key": refs[0]["started_at"],
                "event_ended_at": refs[-1]["ended_at"],
                "source_unit_refs": refs,
                "confidence": self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.7),
            })
        return signals

    @staticmethod
    def _format_facts_view_for_user_activity_prompt(
        facts: Sequence[Dict[str, Any]],
    ) -> str:
        """Render evidence facts as a JSON array with one complete fact per line."""
        if not facts:
            return "[]"
        return "[\n" + ",\n".join(
            json.dumps({
                "id": item.get("id"),
                "summary": _compact_whitespace(item.get("summary") or item.get("text") or ""),
                "dialogue_time_key": item.get("dialogue_time_key") or "",
                "event_time_key": item.get("event_time_key") or "",
            }, ensure_ascii=False, separators=(",", ":"))
            for item in facts
        ) + "\n]"

    def _normalize_memory_fact_extraction_llm_output(
        self,
        data: Dict[str, Any],
        units: Sequence[MemoryUnit],
        *,
        prompt_language: str,
        known_world_owner_aliases: Sequence[str],
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        diagnostics = diagnostics if diagnostics is not None else {}
        raw_facts = data.get("facts")
        if not isinstance(raw_facts, list):
            diagnostics.update({
                "normalization_status": "invalid_facts_field",
                "facts_field_type": type(raw_facts).__name__,
                "raw_fact_count": None,
                "normalized_fact_count": 0,
                "dropped_fact_counts": {},
                "fact_candidates": [],
            })
            return None
        facts: List[Dict[str, Any]] = []
        dropped_fact_counts: Dict[str, int] = {}
        candidate_diagnostics: List[Dict[str, Any]] = []

        def record_drop(reason: str) -> None:
            dropped_fact_counts[reason] = dropped_fact_counts.get(reason, 0) + 1

        def record_candidate(
            *,
            index: int,
            raw_fact: Any,
            result: str,
        ) -> None:
            if len(candidate_diagnostics) >= 12:
                return
            raw_payload = raw_fact if isinstance(raw_fact, dict) else {}
            candidate_diagnostics.append({
                "index": index,
                "result": result,
                "text": self._format_log_text(
                    raw_payload.get("text") or raw_payload.get("summary") or "",
                    limit=300,
                ),
                "fact_type": _compact_whitespace(raw_payload.get("fact_type") or ""),
            })

        dialogue_time_key = _to_timestamp_text(
            units[0].start_time if units else ""
        ) or _now_text()
        newly_confirmed_aliases: List[str] = []
        for index, raw_fact in enumerate(raw_facts, 1):
            if not isinstance(raw_fact, dict):
                record_drop("non_object_fact")
                record_candidate(
                    index=index,
                    raw_fact=raw_fact,
                    result="dropped_non_object_fact",
                )
                continue
            text = _compact_whitespace(raw_fact.get("text") or raw_fact.get("summary") or "")
            if not text:
                record_drop("missing_text")
                record_candidate(
                    index=index,
                    raw_fact=raw_fact,
                    result="dropped_missing_text",
                )
                continue
            keywords = self._normalize_string_list(raw_fact.get("keywords"), limit=18)
            if not keywords:
                keywords = self._keywords(text, limit=18)
            raw_entity_descriptors = self._normalize_entity_descriptors(
                raw_fact.get("entities"), limit=16,
            )
            alias_candidates = self._normalize_world_owner_alias_candidates(
                raw_fact.get("world_owner_alias_candidates"),
                fact_text=text,
                entity_names=[item["name"] for item in raw_entity_descriptors],
                source_units=units,
            )
            confirmed_aliases = self._confirmed_owner_aliases_in_fact(
                entity_names=[item["name"] for item in raw_entity_descriptors],
                fact_text=text,
                confirmed_aliases=[
                    *known_world_owner_aliases,
                    *newly_confirmed_aliases,
                    *(
                        item["alias_name"] for item in alias_candidates
                        if item["status"] == "confirmed"
                    ),
                ],
            )
            owner_alias_reference = bool(confirmed_aliases)
            allows_world_owner_reference = (
                self._units_include_world_owner(units) or owner_alias_reference
            )
            entity_descriptors = self._normalize_fact_entity_descriptors(
                raw_fact.get("entities"),
                allows_world_owner_reference=allows_world_owner_reference,
            )
            if owner_alias_reference:
                self._merge_entity_descriptor(entity_descriptors, {
                    "name": self._world_owner_entity_name,
                    "type": "PERSON",
                })
            entities = [item["name"] for item in entity_descriptors]
            if not entities:
                entity_descriptors = self._normalize_fact_entity_descriptors(
                    self._entities(text),
                    allows_world_owner_reference=allows_world_owner_reference,
                )
                entities = [item["name"] for item in entity_descriptors]
            fact_topic_fallback = " ".join(keywords[:3]) if keywords else "general"
            fact_topic = (
                self._normalize_topic_name(raw_fact.get("fact_topic"))
                or self._normalize_topic_name(fact_topic_fallback)
                or "general"
            )
            user_claim_signals = self._normalize_user_claim_signal(
                raw_fact.get("user_claim_signals"),
                allows_world_owner_reference=allows_world_owner_reference,
                source_units=units,
                owner_aliases=confirmed_aliases,
            )
            user_action_signals = self._normalize_user_action_signals(
                raw_fact.get("user_action_signals"),
                fact_entities=entities,
                allows_world_owner_reference=allows_world_owner_reference,
                source_units=units,
                owner_aliases=confirmed_aliases,
            )
            user_intent_signals = self._normalize_user_intent_signals(
                raw_fact.get("user_intent_signals"),
                allows_world_owner_reference=allows_world_owner_reference,
                source_units=units,
                owner_aliases=confirmed_aliases,
            )
            event_time_key = _compact_whitespace(raw_fact.get("event_time_key") or "")
            facts.append({
                "summary": text,
                "fact_type": self._normalize_fact_type(raw_fact.get("fact_type")),
                "event_time_key": event_time_key,
                "dialogue_time_key": dialogue_time_key,
                "keywords": keywords,
                "entities": entities,
                "entity_descriptors": entity_descriptors,
                "user_claim_signals": user_claim_signals,
                "user_action_signals": user_action_signals,
                "user_intent_signals": user_intent_signals,
                "world_owner_alias_candidates": alias_candidates,
                "resolved_world_owner_aliases": confirmed_aliases,
                "fact_topic": fact_topic,
                "importance": 0.5,
                "confidence": 0.9,
                "metadata": {
                    "extractor": "llm",
                    "time_confidence": _compact_whitespace(raw_fact.get("time_confidence") or "unknown"),
                    "where": _compact_whitespace(raw_fact.get("where") or ""),
                    "has_world_owner_speaker": (
                        any(
                            signal.get("owner_evidence_type") in {
                                "self_statement", "confirmed_response",
                            }
                            for signal in user_claim_signals
                        )
                        or any(
                            signal.get("assertion_source") in {
                                "self_statement", "confirmed_response",
                            }
                            for signal in user_intent_signals
                        )
                        or (
                            self._units_include_world_owner(units)
                            and any(
                                signal.get("assertion_source") == "observed_event"
                                for signal in user_intent_signals
                            )
                        )
                    ),
                    "has_world_owner_alias_reference": owner_alias_reference,
                },
            })
            newly_confirmed_aliases.extend(
                item["alias_name"] for item in alias_candidates
                if item["status"] == "confirmed"
            )
            record_candidate(
                index=index,
                raw_fact=raw_fact,
                result="accepted",
            )
        diagnostics.update({
            "normalization_status": "ok",
            "facts_field_type": "list",
            "raw_fact_count": len(raw_facts),
            "normalized_fact_count": len(facts),
            "dropped_fact_counts": dropped_fact_counts,
            "fact_candidates": candidate_diagnostics,
            "omitted_fact_candidate_diagnostics": max(
                0,
                len(raw_facts) - len(candidate_diagnostics),
            ),
        })
        return {"facts": facts}

    def _normalize_user_claim_signal(
        self,
        value: Any,
        *,
        allows_world_owner_reference: bool = False,
        source_units: Optional[Sequence[MemoryUnit]] = None,
        owner_aliases: Sequence[str] = (),
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        max_items = max(
            0,
            int(
                limit
                if limit is not None
                else self._memory_cfg.get("user_claim_signal_max_per_fact", 3) or 3
            ),
        )
        if max_items <= 0:
            return []
        allowed_types = self._user_claim_types()
        normalized: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str]] = set()
        for raw in value:
            if not isinstance(raw, dict):
                continue
            claim_type_hint = str(raw.get("claim_type_hint") or "").strip().lower()
            if claim_type_hint not in allowed_types:
                continue
            claim_anchor = _compact_whitespace(
                raw.get("claim_anchor")
                or raw.get("anchor")
                or raw.get("attribute_name")
                or ""
            )
            evidence_basis = _compact_whitespace(
                raw.get("evidence_basis")
                or raw.get("evidence")
                or raw.get("reason")
                or ""
            )
            if (
                not claim_anchor
                or not evidence_basis
            ):
                continue
            confidence = self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.75)
            subject_entity = _compact_whitespace(raw.get("subject_entity") or "")
            if not subject_entity:
                continue
            # user_claim signals are evidence about the configured wearer
            # only. A third-party entity may still appear in the fact or as
            # the object of a relation, but cannot be the claim subject.
            is_owner_reference = (
                self._is_world_owner_entity_reference(subject_entity)
                or subject_entity.lower() == self._world_owner_entity_name.lower()
            )
            if not allows_world_owner_reference or not is_owner_reference:
                continue
            owner_evidence = self._normalize_user_signal_owner_evidence(
                raw,
                source_units=source_units,
                owner_aliases=owner_aliases,
            )
            if owner_evidence is None:
                continue
            if owner_evidence["owner_evidence_type"] == "third_party_report":
                confidence = min(confidence, 0.65)
            key = (
                claim_type_hint,
                claim_anchor.lower(),
                evidence_basis.lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            item: Dict[str, Any] = {
                "subject_entity": self._world_owner_entity_name,
                "claim_type_hint": claim_type_hint,
                "claim_anchor": claim_anchor,
                "evidence_basis": evidence_basis,
                "confidence": confidence,
                **owner_evidence,
            }
            normalized.append(item)
            if len(normalized) >= max_items:
                break
        return normalized

    def _normalize_user_action_signals(
        self,
        value: Any,
        *,
        fact_entities: Sequence[str],
        allows_world_owner_reference: bool,
        source_units: Optional[Sequence[MemoryUnit]] = None,
        owner_aliases: Sequence[str] = (),
    ) -> List[Dict[str, Any]]:
        """Retain only grounded, wearer-attributed concrete action evidence."""
        if not isinstance(value, list) or not allows_world_owner_reference:
            return []
        owner_name = self._world_owner_entity_name
        if owner_name not in fact_entities:
            return []
        normalized: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str, str]] = set()
        for raw in value:
            if not isinstance(raw, dict):
                continue
            subject = _compact_whitespace(raw.get("subject_entity") or "")
            if not self._is_world_owner_entity_reference(subject) and (
                subject.lower() != owner_name.lower()
            ):
                continue
            action_kind = str(raw.get("action_kind") or "").strip().lower()
            if action_kind not in {"action", "choice", "participation"}:
                continue
            action_text = _compact_whitespace(raw.get("action_text") or "")[:480]
            evidence_basis = _compact_whitespace(raw.get("evidence_basis") or "")[:480]
            if not action_text or not evidence_basis:
                continue
            owner_evidence = self._normalize_user_signal_owner_evidence(
                raw,
                source_units=source_units,
                owner_aliases=owner_aliases,
            )
            if owner_evidence is None:
                continue
            key = (
                action_kind, action_text.lower(),
                owner_evidence["owner_evidence_text"].lower(),
                owner_evidence["other_evidence_text"].lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            confidence = self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.7)
            if owner_evidence["owner_evidence_type"] == "third_party_report":
                confidence = min(confidence, 0.65)
            normalized.append({
                "subject_entity": owner_name,
                "action_kind": action_kind,
                "action_text": action_text,
                "evidence_basis": evidence_basis,
                "confidence": confidence,
                **owner_evidence,
            })
            if len(normalized) >= 3:
                break
        return normalized

    def _normalize_user_intent_signals(
        self,
        value: Any,
        *,
        limit: Optional[int] = None,
        allows_world_owner_reference: bool = False,
        source_units: Optional[Sequence[MemoryUnit]] = None,
        owner_aliases: Sequence[str] = (),
    ) -> List[Dict[str, Any]]:
        """Normalize fact-level evidence for the prospective world model."""
        if not isinstance(value, list):
            return []
        max_items = max(
            0,
            int(
                limit
                if limit is not None
                else self._memory_cfg.get("user_intent_signal_max_per_fact", 2) or 2
            ),
        )
        if max_items <= 0:
            return []
        allowed_signal_purposes = {"future_intent", "lifecycle_update"}
        normalized: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str, str, str]] = set()
        for raw in value:
            if not isinstance(raw, dict):
                continue
            signal_purpose = str(raw.get("signal_purpose") or "").strip().lower()
            if signal_purpose not in allowed_signal_purposes:
                continue
            user_role = str(raw.get("user_role") or "").strip()
            if not user_role:
                continue
            assertion_source = str(raw.get("assertion_source") or "").strip().lower()
            if assertion_source not in {
                "self_statement", "confirmed_response", "third_party_report",
                "observed_event",
            }:
                continue
            subject = raw.get("subject_entity") or raw.get("subject")
            if isinstance(subject, dict):
                subject_name = _compact_whitespace(
                    subject.get("name") or subject.get("text") or ""
                )
            else:
                subject_name = _compact_whitespace(subject)
            if not allows_world_owner_reference or not (
                self._is_world_owner_entity_reference(subject_name)
                or subject_name.lower() == self._world_owner_entity_name.lower()
            ):
                continue
            subject_name = self._world_owner_entity_name
            user_intent_anchor = _compact_whitespace(
                raw.get("user_intent_anchor") or raw.get("intent_anchor") or ""
            )[:240]
            evidence_basis = _compact_whitespace(
                raw.get("evidence_basis") or raw.get("evidence") or raw.get("reason") or ""
            )[:480]
            if not subject_name or not user_intent_anchor or not evidence_basis:
                continue
            if assertion_source == "observed_event":
                if raw.get("owner_evidence_text") or raw.get("other_evidence_text"):
                    continue
                owner_evidence = {
                    "owner_evidence_text": "",
                    "other_evidence_text": "",
                }
            else:
                owner_evidence = self._normalize_user_signal_owner_evidence(
                    {**raw, "owner_evidence_type": assertion_source},
                    source_units=source_units,
                    owner_aliases=owner_aliases,
                )
                if owner_evidence is None:
                    continue
                owner_evidence.pop("owner_evidence_type")
            confidence = self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.75)
            if assertion_source == "third_party_report":
                confidence = min(confidence, 0.65)
            key = (
                subject_name.lower(),
                signal_purpose,
                self._generate_topic_name_key(user_intent_anchor),
                user_role.casefold(),
                evidence_basis.casefold(),
            )
            if key in seen:
                continue
            seen.add(key)
            normalized.append({
                "subject_entity": subject_name,
                "signal_purpose": signal_purpose,
                "user_role": user_role,
                "user_intent_anchor": user_intent_anchor,
                "assertion_source": assertion_source,
                "evidence_basis": evidence_basis,
                "confidence": confidence,
                **owner_evidence,
            })
            if len(normalized) >= max_items:
                break
        return normalized

    @staticmethod
    def _is_world_owner_entity_reference(value: Any) -> bool:
        return _compact_whitespace(value).lower() in {
            alias.lower() for alias in _WORLD_OWNER_ENTITY_REFERENCE_ALIASES
        }

    @staticmethod
    def _units_include_world_owner(units: Sequence[MemoryUnit]) -> bool:
        """Return whether this fact task contains a verified wearer speaker."""
        return any(
            isinstance(unit, MemoryUnit)
            and any(
                _compact_whitespace(speaker) == "speaker_用户本人"
                for speaker in unit.speakers or []
            )
            for unit in units or ()
        )

    def _confirmed_owner_aliases_in_fact(
        self,
        *,
        entity_names: Sequence[str],
        fact_text: str,
        confirmed_aliases: Sequence[str],
    ) -> List[str]:
        """Resolve aliases co-referenced with the wearer, without another DB lookup."""
        if not any(
            self._is_world_owner_entity_reference(name)
            or _compact_whitespace(name).lower() == self._world_owner_entity_name.lower()
            for name in entity_names
        ):
            return []
        confirmed_keys = {
            self._db.normalize_owner_alias_key(name)
            for name in confirmed_aliases
        }
        aliases: List[str] = []
        for name in entity_names:
            normalized_name = _compact_whitespace(name)
            if (
                normalized_name
                and self._db.normalize_owner_alias_key(normalized_name) in confirmed_keys
                and normalized_name.casefold() in fact_text.casefold()
                and normalized_name not in aliases
            ):
                aliases.append(normalized_name)
        return aliases

    def _normalize_world_owner_alias_candidates(
        self,
        value: Any,
        *,
        fact_text: str,
        entity_names: Sequence[str],
        source_units: Sequence[MemoryUnit],
    ) -> List[Dict[str, Any]]:
        """Validate proposed owner names against current, speaker-attributed audio text."""
        if not isinstance(value, list):
            return []
        source_entries = [
            (_compact_whitespace(entry.get("speaker")), _compact_whitespace(entry.get("text")))
            for unit in source_units
            for entry in unit.entries()
        ]
        entity_keys = {
            self._db.normalize_owner_alias_key(name)
            for name in entity_names
        }
        candidates: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for raw in value[:4]:
            if not isinstance(raw, dict):
                continue
            alias_name = _compact_whitespace(raw.get("alias_name") or "")[:80]
            alias_key = self._db.normalize_owner_alias_key(alias_name)
            evidence_type = str(raw.get("evidence_type") or "").strip().lower()
            evidence_text = _compact_whitespace(raw.get("evidence_text") or "")[:480]
            response_text = _compact_whitespace(raw.get("owner_response_text") or "")[:240]
            if (
                len(alias_key) < 2 or alias_key in seen
                or alias_key not in entity_keys or alias_name not in fact_text
                or alias_name.lower() in _AMBIGUOUS_USER_ENTITY_ALIASES
                or self._is_world_owner_entity_reference(alias_name)
                or evidence_type not in {
                    "self_identification", "direct_address", "third_party_reference",
                }
                or not evidence_text or alias_name not in evidence_text
            ):
                continue
            matching_positions = [
                index for index, (speaker, text) in enumerate(source_entries)
                if evidence_text in text and (
                    speaker == "speaker_用户本人"
                    if evidence_type == "self_identification"
                    else speaker not in {"speaker_用户本人", "speaker_助手"}
                )
            ]
            if not matching_positions:
                continue
            if evidence_type == "self_identification":
                if not re.search(
                    rf"(?:我叫|我是|我的名字(?:是)?|本人叫|my name is|i am|i'm)\s*{re.escape(alias_name)}",
                    evidence_text,
                    re.IGNORECASE,
                ):
                    continue
                status = "confirmed"
                source_speaker = "speaker_用户本人"
            elif evidence_type == "direct_address":
                if not response_text or response_text in {
                    "嗯", "哦", "好", "好的", "知道了", "收到", "明白了",
                } or not any(
                    speaker == "speaker_用户本人" and response_text in text
                    for position in matching_positions
                    for speaker, text in source_entries[position + 1:]
                ):
                    continue
                status = "confirmed"
                source_speaker = source_entries[matching_positions[0]][0]
            else:
                if response_text:
                    continue
                status = "candidate"
                source_speaker = source_entries[matching_positions[0]][0]
            seen.add(alias_key)
            candidates.append({
                "alias_name": alias_name,
                "status": status,
                "confidence": self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.75),
                "evidence_type": evidence_type,
                "evidence_text": evidence_text,
                "source_speaker": source_speaker,
            })
        return candidates

    @staticmethod
    def _split_user_signal_evidence_text(text: str) -> List[str]:
        """Treat Unicode punctuation as separators between quoted fragments."""
        separated = "".join(
            "\n" if unicodedata.category(char).startswith("P") else char
            for char in text
        )
        return [part for part in (
            _compact_whitespace(value) for value in separated.splitlines()
        ) if part]

    @staticmethod
    def _match_user_signal_evidence_fragments(
        fragments: Sequence[str], source_entries: Sequence[Tuple[int, str]],
    ) -> Optional[Tuple[int, int]]:
        """Match all fragments in order without reusing source characters."""
        if not fragments:
            return None
        entry_index, text_offset = 0, 0
        first_position, last_position = -1, -1
        for fragment in fragments:
            while entry_index < len(source_entries):
                position, text = source_entries[entry_index]
                match_offset = text.find(fragment, text_offset)
                if match_offset >= 0:
                    if first_position < 0:
                        first_position = position
                    last_position = position
                    text_offset = match_offset + len(fragment)
                    break
                entry_index += 1
                text_offset = 0
            else:
                return None
        return first_position, last_position

    def _normalize_user_signal_owner_evidence(
        self,
        signal: Dict[str, Any],
        *,
        source_units: Optional[Sequence[MemoryUnit]],
        owner_aliases: Sequence[str] = (),
    ) -> Optional[Dict[str, str]]:
        """Validate ordered quote fragments, preserving speaker and response order."""
        evidence_type = str(signal.get("owner_evidence_type") or "").strip().lower()
        if evidence_type not in {"self_statement", "confirmed_response", "third_party_report"}:
            return None
        if any(value is not None and not isinstance(value, str) for value in (
            signal.get("owner_evidence_text"), signal.get("other_evidence_text"),
        )):
            return None
        owner_text = _compact_whitespace(
            signal.get("owner_evidence_text") or ""
        ).strip("\"'“”")
        other_text = _compact_whitespace(
            signal.get("other_evidence_text") or ""
        ).strip("\"'“”")
        owner_fragments = self._split_user_signal_evidence_text(owner_text)
        other_fragments = self._split_user_signal_evidence_text(other_text)
        source_entries = [
            (
                _compact_whitespace(entry.get("speaker") or ""),
                _compact_whitespace(entry.get("text") or ""),
            )
            for unit in source_units or []
            for entry in unit.entries()
        ]
        if evidence_type == "third_party_report":
            if (
                owner_text or not other_fragments or not any(
                    self._db.normalize_owner_alias_key(alias)
                    in self._db.normalize_owner_alias_key(other_text)
                    for alias in owner_aliases
                )
            ):
                return None
            if source_units is not None and self._match_user_signal_evidence_fragments(
                other_fragments, [
                    (index, text) for index, (speaker, text) in enumerate(source_entries)
                    if speaker not in {"speaker_用户本人", "speaker_助手"}
                ],
            ) is None:
                return None
            return {
                "owner_evidence_type": evidence_type,
                "owner_evidence_text": "",
                "other_evidence_text": other_text,
            }
        if not owner_fragments or (
            evidence_type == "confirmed_response" and not other_fragments
        ) or (
            evidence_type == "self_statement" and other_text
        ):
            return None
        if evidence_type == "confirmed_response" and all(fragment in {
            "嗯", "哦", "好", "好的", "知道了", "收到", "明白了",
        } for fragment in owner_fragments):
            return None
        if source_units is not None:
            response_after = -1
            if evidence_type == "confirmed_response":
                other_positions = self._match_user_signal_evidence_fragments(
                    other_fragments, [
                        (index, text) for index, (speaker, text) in enumerate(source_entries)
                        if speaker != "speaker_用户本人"
                    ],
                )
                if other_positions is None:
                    return None
                response_after = other_positions[1]
            if self._match_user_signal_evidence_fragments(
                owner_fragments, [
                    (index, text) for index, (speaker, text) in enumerate(source_entries)
                    if speaker == "speaker_用户本人" and index > response_after
                ],
            ) is None:
                return None
        return {
            "owner_evidence_type": evidence_type,
            "owner_evidence_text": owner_text,
            "other_evidence_text": other_text,
        }

    def _normalize_fact_entity_descriptors(
        self,
        value: Any,
        *,
        allows_world_owner_reference: bool,
        limit: int = 16,
    ) -> List[Dict[str, str]]:
        """Apply source-aware identity boundaries to extracted entities.

        A bare ``用户`` is ambiguous: in ambient audio it generally denotes a
        product user or customer, not the wearer. Only the reserved
        ``用户本人`` / ``user self`` reference from a tagged interaction may
        resolve to this database's world owner.
        """
        descriptors = self._normalize_entity_descriptors(value, limit=limit)
        normalized: List[Dict[str, str]] = []
        for descriptor in descriptors:
            name = _compact_whitespace(descriptor.get("name") or "")
            if not name or name.lower() in _AMBIGUOUS_USER_ENTITY_ALIASES:
                continue
            if self._is_world_owner_entity_reference(name):
                if not allows_world_owner_reference:
                    continue
                descriptor = {
                    "name": self._world_owner_entity_name,
                    "type": "PERSON",
                }
            self._merge_entity_descriptor(normalized, descriptor)
            if len(normalized) >= limit:
                break
        return normalized

    def _build_dialogue_batch_for_prompt(
        self,
        turns: List[Dict[str, Any]],
        *,
        prompt_language: str,
    ) -> str:
        is_en = prompt_language == "en"
        time_label = "Conversation timestamp" if is_en else "对话发生时间"
        user_label = "speaker_用户本人"
        assistant_label = "speaker_助手"
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

    @staticmethod
    def _parse_participants_from_units(units: Sequence[MemoryUnit]) -> List[str]:
        participants: List[str] = []
        seen: set[str] = set()
        for unit in units or []:
            for value in unit.speakers or []:
                speaker = _compact_whitespace(value)
                if not speaker:
                    continue
                key = speaker.lower()
                if key in seen:
                    continue
                seen.add(key)
                participants.append(speaker)
        return participants or ["unknown_speaker"]

    @staticmethod
    def _append_unique_text(values: List[str], value: Any, *, limit: int = 64) -> None:
        text = _compact_whitespace(value)
        if not text:
            return
        seen = {item.lower() for item in values}
        if text.lower() in seen:
            return
        values.append(text)
        if len(values) > limit:
            del values[limit:]

    @classmethod
    def _merge_entity_descriptor(
        cls,
        descriptors: List[Dict[str, str]],
        value: Any,
    ) -> None:
        """Merge one typed entity without downgrading a concrete type."""
        if not isinstance(value, dict):
            return
        name = _compact_whitespace(value.get("name") or value.get("text") or "")
        if not cls._is_valid_entity_name(name):
            return
        entity_type = cls._normalize_entity_type(value.get("type"))
        for descriptor in descriptors:
            if _compact_whitespace(descriptor.get("name") or "") != name:
                continue
            if descriptor.get("type") == "OTHER" and entity_type != "OTHER":
                descriptor["type"] = entity_type
            return
        descriptors.append({"name": name, "type": entity_type})

    @classmethod
    def _fact_entity_descriptors(
        cls,
        fact: Dict[str, Any],
        *,
        entities: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, str]]:
        """Return typed entity descriptors while retaining name-only fallback."""
        metadata = fact.get("metadata") if isinstance(fact.get("metadata"), dict) else {}
        descriptor_source = fact.get("entity_descriptors")
        if descriptor_source is None:
            descriptor_source = metadata.get("entity_descriptors")
        descriptors = cls._normalize_entity_descriptors(
            descriptor_source if descriptor_source is not None else fact.get("entities"),
            limit=32,
        )
        names = (
            list(entities)
            if entities is not None
            else cls._normalize_entity_names(fact.get("entities"), limit=32)
        )
        for name in names:
            cls._merge_entity_descriptor(
                descriptors,
                {"name": name, "type": "OTHER"},
            )
        return descriptors

    @staticmethod
    def _memory_units_log_view(units: Sequence[MemoryUnit]) -> List[CompactLogValue]:
        """Render each complete unit on one line, independently of log events."""
        return [
            CompactLogValue({
                "unit_type": unit.unit_type,
                "speakers": list(unit.speakers or []),
                "texts": list(unit.texts or []),
                "started_ats": list(unit.started_ats or []),
                "ended_ats": list(unit.ended_ats or []),
                "tags": list(unit.tags or []),
            })
            for unit in units
        ]

    @staticmethod
    def _build_user_activity_units_for_prompt(
        units: Sequence[MemoryUnit],
        *,
        prompt_language: str,
    ) -> str:
        """Compact activity evidence without losing source types or entry boundaries."""
        unit_entries = [(unit, unit.entries()) for unit in units]
        if not unit_entries:
            return ""
        is_en = prompt_language == "en"

        raw_times = [
            entry[field] for _, entries in unit_entries for entry in entries
            for field in ("started_at", "ended_at")
        ]
        parsed_times: List[datetime] = []
        for value in raw_times:
            try:
                parsed_times.append(datetime.fromisoformat(value.replace("Z", "+00:00")))
            except ValueError:
                break
        shared_clock = bool(parsed_times) and len(parsed_times) == len(raw_times) and all(
            value.date() == parsed_times[0].date()
            and value.utcoffset() == parsed_times[0].utcoffset()
            for value in parsed_times
        )
        lines: List[str] = []
        if shared_clock:
            first = parsed_times[0]
            offset = first.timetz().isoformat()[len(first.time().isoformat()):]
            zone = offset or ("unspecified" if is_en else "未指定")
            lines.append(
                f"Window date: {first.date().isoformat()}; timezone: {zone}"
                if is_en else f"窗口日期：{first.date().isoformat()}；时区：{zone}"
            )
        def display_time(value: str) -> str:
            if shared_clock:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).time().isoformat()
            return value or "?"

        for index, (unit, entries) in enumerate(unit_entries, 1):
            entry_texts = []
            for entry in entries:
                speaker = entry["speaker"] or "unknown_speaker"
                text = entry["text"].replace("\r", "\\r").replace("\n", "\\n")
                entry_texts.append(
                    f"[{display_time(entry['started_at'])}—{display_time(entry['ended_at'])}] "
                    f"{speaker}: {text}"
                )
            lines.append(f"[U{index} {unit.unit_type}] " + " | ".join(entry_texts))
        return "\n".join(lines)

    def _format_memory_units_view_for_fact_prompt(
        self,
        units: Sequence[MemoryUnit],
        *,
        prompt_language: str,
    ) -> str:
        """Keep each complete unit on one line, escaping embedded line breaks."""
        is_en = prompt_language == "en"
        time_label = "Time" if is_en else "时间"
        unit_type_label = "Unit type" if is_en else "单元类型"
        speaker_label = "Speaker" if is_en else "说话人"
        text_label = "Text" if is_en else "文本"
        blocks: List[str] = []
        for index, unit in enumerate(units, 1):
            entry_blocks: List[str] = []
            for entry_index, text in enumerate(unit.texts or [], 1):
                started_at = (
                    unit.started_ats[entry_index - 1]
                    if entry_index <= len(unit.started_ats)
                    else ""
                ) or ""
                ended_at = (
                    unit.ended_ats[entry_index - 1]
                    if entry_index <= len(unit.ended_ats)
                    else started_at
                ) or started_at
                time_text = (
                    started_at
                    if started_at == ended_at
                    else f"{started_at} - {ended_at}"
                )
                speaker = (
                    unit.speakers[entry_index - 1]
                    if entry_index <= len(unit.speakers)
                    else ""
                )
                entry_blocks.append(" | ".join([
                    f"{time_label}: {time_text}",
                    f"{speaker_label}: {speaker}",
                    f"{text_label}: {text}",
                ]))
            blocks.append(" | ".join([
                f"[Unit {index}]",
                f"{unit_type_label}: {unit.unit_type or 'transcript'}",
                *entry_blocks,
            ]).replace("\r", "\\r").replace("\n", "\\n"))
        return "\n".join(blocks)

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
        if jieba is not None and chinese_chars:
            raw_tokens = [
                *jieba.lcut(clean, HMM=False),
                *jieba.cut_for_search(clean, HMM=False),
            ]
            tokens: List[str] = []
            seen: set[str] = set()
            for token in raw_tokens:
                normalized = _compact_whitespace(token).strip(
                    "'\".,:;!?，。！？、；：（）()[]{}"
                )
                if not normalized or not re.search(r"[0-9a-zA-Z\u4e00-\u9fff]", normalized):
                    continue
                key = normalized.lower()
                if key in seen:
                    continue
                seen.add(key)
                tokens.append(key)
            if tokens:
                return tokens
        if len(chinese_chars) >= 3:
            return [chinese_chars[i : i + 2] for i in range(len(chinese_chars) - 1)]
        return self._keywords(clean, limit=12)

    def _resolve_prompt_language_from_text(self, text: str, *, fallback: str = "zh") -> str:
        mode = str(self._memory_prompt_language or "source").strip().lower()
        if mode in {"en", "english", "force_en"}:
            return "en"
        if mode in {"zh", "chinese", "force_zh"}:
            return "zh"
        if re.search(r"[\u4e00-\u9fff]", str(text or "")):
            return "zh"
        return "en" if str(fallback).lower().startswith("en") else "zh"

    def _resolve_prompt_language_from_memory_units(
        self,
        units: Sequence[MemoryUnit],
    ) -> str:
        """Resolve one LLM task's prompt language from its actual source text."""
        text = "\n".join(
            _compact_whitespace(unit.text)
            for unit in units
            if isinstance(unit, MemoryUnit)
        )
        return self._resolve_prompt_language_from_text(text)

    def _store_extracted_memory_facts_into_db(
        self,
        *,
        facts: Sequence[Dict[str, Any]],
        tags: List[str],
        reference_timestamp: Any = None,
    ) -> Dict[str, Any]:
        fact_ids: List[int] = []
        user_claim_signal_mapping_updates: List[Dict[str, Any]] = []
        user_action_signal_updates: List[Dict[str, Any]] = []
        user_intent_signal_mapping_updates: List[Dict[str, Any]] = []
        user_claim_signal_fact_ids: List[int] = []
        user_action_signal_fact_ids: List[int] = []
        user_intent_signal_fact_ids: List[int] = []
        owner_entity_id = self._intent_world_owner_entity_id(reference_timestamp=reference_timestamp)
        stored_confirmed_alias_keys = {
            str(row["alias_key"])
            for row in self._db.get_owner_entity_aliases(
                owner_entity_id, status="confirmed", limit=4096,
            )
        }
        for fact in facts:
            keywords = self._normalize_string_list(fact.get("keywords"), limit=18)
            if not keywords:
                keywords = self._keywords(fact.get("summary") or "", limit=18)
            fact_entities = list(fact.get("entities") or [])
            entity_descriptors = list(fact.get("entity_descriptors") or [])
            raw_metadata = fact.get("metadata")
            metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
            resolved_world_owner_aliases = list(fact["resolved_world_owner_aliases"])
            fact_topic = self._normalize_topic_name(fact.get("fact_topic"))
            if not fact_topic:
                fallback_topics = self._topic_candidates(fact["summary"])
                fact_topic = self._normalize_topic_name(
                    fallback_topics[0] if fallback_topics else ""
                ) or "general"
            owner_alias_keys = {
                self._db.normalize_owner_alias_key(alias)
                for alias in resolved_world_owner_aliases
            }
            entity_mapping = self._db.upsert_entity_nodes([
                descriptor for descriptor in entity_descriptors
                if self._db.normalize_owner_alias_key(descriptor.get("name") or "")
                not in owner_alias_keys
            ], reference_timestamp=reference_timestamp)
            if owner_alias_keys:
                for descriptor in entity_descriptors:
                    name = str(descriptor.get("name") or "")
                    if self._db.normalize_owner_alias_key(name) in owner_alias_keys:
                        entity_mapping[name] = owner_entity_id
            entity_ids = [
                entity_mapping[entity_name]
                for entity_name in fact_entities
                if entity_name in entity_mapping
            ]
            fact_metadata = {
                **metadata,
                "tags": tags,
                "entity_descriptors": entity_descriptors,
            }
            fact_id = self._db.insert_fact(
                fact_type=fact["fact_type"],
                summary=fact["summary"],
                keywords=keywords,
                entities=fact_entities,
                entity_ids=entity_ids,
                fact_topic=fact_topic,
                event_time_key=fact.get("event_time_key") or "",
                dialogue_time_key=fact.get("dialogue_time_key") or "",
                confidence=fact["confidence"],
                importance=fact["importance"],
                metadata=fact_metadata,
                reference_timestamp=reference_timestamp,
            )
            confirmed_candidates_by_key = {
                self._db.normalize_owner_alias_key(alias["alias_name"]): alias
                for alias in fact.get("world_owner_alias_candidates") or []
                if isinstance(alias, dict) and alias.get("status") == "confirmed"
            }
            for alias_name in resolved_world_owner_aliases:
                alias_key = self._db.normalize_owner_alias_key(alias_name)
                alias = confirmed_candidates_by_key.get(alias_key)
                if alias_key in stored_confirmed_alias_keys or alias is None:
                    # Known aliases need no new row, and fresh aliases need
                    # current-fact confirmation evidence before persistence.
                    continue
                self._db.upsert_owner_entity_alias(
                    owner_entity_id=owner_entity_id,
                    alias_name=alias_name,
                    status="confirmed",
                    confidence=alias["confidence"],
                    evidence_fact_id=fact_id,
                    evidence_type=alias["evidence_type"],
                    evidence_text=alias["evidence_text"],
                    source_speaker=alias["source_speaker"],
                    reference_timestamp=reference_timestamp,
                )
                stored_confirmed_alias_keys.add(alias_key)
            fact_ids.append(fact_id)
            user_claim_updates = (
                self._fact_user_claim_signal_mapping_updates(fact_id, fact, reference_timestamp=reference_timestamp)
            )
            user_claim_signal_mapping_updates.extend(user_claim_updates)
            if user_claim_updates:
                user_claim_signal_fact_ids.append(fact_id)
            action_updates = self._fact_user_action_signal_updates(
                fact_id=fact_id,
                fact=fact,
                owner_entity_id=owner_entity_id,
            )
            user_action_signal_updates.extend(action_updates)
            if action_updates:
                user_action_signal_fact_ids.append(fact_id)
            user_intent_updates = (
                self._fact_user_intent_signal_mapping_updates(fact_id, fact, reference_timestamp=reference_timestamp)
            )
            user_intent_signal_mapping_updates.extend(user_intent_updates)
            if user_intent_updates:
                user_intent_signal_fact_ids.append(fact_id)
        self._db.upsert_user_claim_signals(
            user_claim_signal_mapping_updates,
            reference_timestamp=reference_timestamp,
        )
        self._db.upsert_user_action_signals(user_action_signal_updates, reference_timestamp=reference_timestamp)
        self._db.upsert_user_intent_signals(
            user_intent_signal_mapping_updates,
            reference_timestamp=reference_timestamp,
        )
        self._upsert_memory_facts_into_recall_document_table(
            fact_ids, reference_timestamp=reference_timestamp,
        )
        return {
            "fact_ids": fact_ids,
            "user_claim_signal_fact_count": len(user_claim_signal_fact_ids),
            "user_claim_signal_count": len(user_claim_signal_mapping_updates),
            "user_action_signal_fact_count": len(user_action_signal_fact_ids),
            "user_action_signal_count": len(user_action_signal_updates),
            "user_intent_signal_fact_count": len(
                user_intent_signal_fact_ids
            ),
            "user_intent_signal_count": len(user_intent_signal_mapping_updates),
        }

    def _process_memory_user_observed_pattern_update_task(
        self, *, signal_ids: Optional[Sequence[int]] = None,
        reference_timestamp: Optional[Any] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Evaluate affected action clusters against existing patterns or discovery gates."""
        selected_ids = list(dict.fromkeys(
            int(value) for value in signal_ids or [] if int(value) > 0
        ))
        if not selected_ids:
            return {"status": "empty", "signal_count": 0,
                    "pattern_event_ids": [],
                    "_completed_signal_ids": []}
        started_at = time.monotonic()
        update = self._update_user_observed_patterns_from_action_signals(
            selected_ids,
            reference_timestamp=reference_timestamp,
        )
        completed = bool(update.get("completed"))
        report = {
            "status": "ok" if completed else "failed",
            "signal_count": len(selected_ids),
            "similar_action_batches": int(update.get("similar_action_batches") or 0),
            "skipped_activity_gate_count": int(update.get("skipped_activity_gate_count") or 0),
            "pattern_relations_recorded": int(update.get("relations_recorded") or 0),
            "direct_support_signal_count": int(
                update.get("direct_support_signal_count") or 0
            ),
            "user_observed_patterns_created": int(update.get("created") or 0),
            "user_observed_patterns_updated": int(update.get("updated") or 0),
            "pattern_event_ids": update["pattern_event_ids"] if completed else [],
            "reference_timestamp": reference_timestamp,
            "total_elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
            "_completed_signal_ids": update.get("processed_signal_ids", []) if completed else [],
        }
        if update.get("error"):
            report["error"] = update["error"]
        self._generate_memory_recall_document_embeddings()
        self._logger.log_json_format("memory_user_observed_pattern_update", "finish", report)
        return report

    def _build_activity_intent_lifecycle_cases(
        self, activities: Sequence[Dict[str, Any]], *, world_owner_id: int,
    ) -> Tuple[List[Dict[str, Any]], List[int]]:
        """Recall prior intents, excluding creations from this activity's facts."""
        limit = max(1, int(self._memory_cfg.get("activity_intent_candidate_limit") or 8))
        intents = self._db.get_intent_objects(world_owner_entity_id=world_owner_id, limit=360)
        cases, replay_event_ids = [], []
        for activity in activities:
            facts = self._db.get_memory_facts_by_ids(activity["evidence_fact_ids"])
            created_intent_ids = set(self._db.get_intent_ids_created_from_facts(
                activity["evidence_fact_ids"],
            ))
            ranked = []
            for intent in intents:
                # The source window must not immediately advance its own new intents.
                if intent["id"] in created_intent_ids:
                    continue
                score = max(self._score_intent_candidate_against_existing_object(
                    {"world_owner_entity_id": world_owner_id, "summary": text}, intent,
                ) for text in [activity["activity_text"], *[fact["summary"] for fact in facts]])
                if score < 0.12:
                    continue
                if self._db.get_activity_intent_lifecycle_observation(activity["id"], intent["id"]):
                    replay_event_ids.extend(event["id"] for event in self._db.get_user_intent_events(
                        source_activity_id=activity["id"],
                    ) if event["intent_id"] == intent["id"])
                    continue
                ranked.append((score, intent))
            ranked.sort(key=lambda pair: (-pair[0], -int(pair[1]["id"])))
            if not ranked:
                continue
            cases.append({
                "activity": {key: activity[key] for key in (
                    "id", "activity_text", "user_role", "started_at", "ended_at",
                )},
                "facts": [self._format_fact_view_for_user_intent_prompt(fact) for fact in facts],
                "existing_intents": [{
                    **self._intent_object_prompt_view(item),
                    "time_start": item["time_start"], "time_end": item["time_end"],
                    "details": item["details"],
                    "latest_event": self._activity_intent_latest_event_view(item["id"]),
                } for _, item in ranked[:limit]],
            })
        return cases, sorted(set(replay_event_ids))

    def _activity_intent_latest_event_view(self, intent_id: int) -> Optional[Dict[str, Any]]:
        event = self._db.get_latest_user_intent_event(intent_id)
        return {key: event[key] for key in ("id", "event_type", "effective_at")} if event else None

    def _process_memory_user_intent_update_from_activities_task(
        self, *, activity_ids: Sequence[int], reference_timestamp: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """One LLM pass matches new activities and adjudicates old-intent progress."""
        scope = "memory_user_intent_update_from_activities"
        ids = sorted({int(value) for value in activity_ids if int(value) > 0})
        activities = self._db.get_user_activity_items_by_ids(ids)
        reference = str(reference_timestamp or max(
            (item["ended_at"] for item in activities), default=_now_text(),
        ))
        report: Dict[str, Any] = {
            "status": "empty", "reference_timestamp": reference, "intent_event_ids": [],
            "processed_activity_ids": [], "failed_activity_ids": sorted(set(ids) - {a["id"] for a in activities}),
        }
        has_decision_errors = False
        try:
            as_of = datetime.fromisoformat(reference.replace("Z", "+00:00")).astimezone()
        except (ValueError, TypeError):
            return {**report, "status": "failed", "reason": "invalid_reference_timestamp",
                    "failed_activity_ids": ids}
        eligible = []
        for activity in activities:
            try:
                end = datetime.fromisoformat(activity["ended_at"].replace("Z", "+00:00")).astimezone()
                if end > as_of:
                    raise ValueError("activity_after_reference_time")
                eligible.append(activity)
            except (ValueError, TypeError):
                report["failed_activity_ids"].append(activity["id"])
        owner_id = self._intent_world_owner_entity_id(reference_timestamp=reference_timestamp)
        batch_size = max(1, int(self._memory_cfg.get("activity_intent_batch_size") or 4))
        for offset in range(0, len(eligible), batch_size):
            batch = eligible[offset:offset + batch_size]
            batch_ids = {item["id"] for item in batch}
            try:
                cases, replay_ids = self._build_activity_intent_lifecycle_cases(batch, world_owner_id=owner_id)
                report["intent_event_ids"].extend(replay_ids)
                case_ids = {case["activity"]["id"] for case in cases}
                report["processed_activity_ids"].extend(batch_ids - case_ids)
                if not cases:
                    continue
                language = self._resolve_prompt_language_from_text("\n".join(a["activity_text"] for a in batch))
                template = (USER_INTENT_LIFECYCLE_FROM_ACTIVITIES_PROMPT_EN if language == "en"
                            else USER_INTENT_LIFECYCLE_FROM_ACTIVITIES_PROMPT_ZH)
                prompt = (template.replace("{reference_timestamp}", reference)
                          .replace("{activity_cases}", "[\n" + ",\n".join(
                              json.dumps(case, ensure_ascii=False, separators=(",", ":")) for case in cases) + "\n]"))
                self._logger.log_json_format(scope, "start", {"activity_ids": sorted(case_ids),
                    "candidate_count": sum(len(case["existing_intents"]) for case in cases),
                    "reference_timestamp": reference})
                raw = self._llm_client.call_llm(prompt, stage_name=scope)
                parsed = self._llm_client.parse_json_object_from_llm_text(raw or "", stage_name=scope)
                if not isinstance(parsed, dict) or not isinstance(parsed.get("decisions"), list):
                    raise ValueError("invalid_activity_intent_lifecycle_response")
            except Exception as exc:
                report["failed_activity_ids"].extend(batch_ids)
                self._logger.log_json_format(scope, "batch_error", {
                    "activity_ids": sorted(batch_ids), "error_type": type(exc).__name__, "error": str(exc)})
                continue
            by_activity: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {value: [] for value in case_ids}
            for decision_index, decision in enumerate(parsed["decisions"]):
                if (not isinstance(decision, dict) or type(decision.get("activity_id")) is not int
                        or decision["activity_id"] not in by_activity):
                    has_decision_errors = True
                    self._logger.log_json_format(scope, "decision_error", {
                        "decision_index": decision_index,
                        "activity_id": decision.get("activity_id") if isinstance(decision, dict) else None,
                        "phase": "grouping", "error": "invalid_activity_decision_id",
                    })
                    continue
                by_activity[decision["activity_id"]].append((decision_index, decision))
            observations, validation_cases, raw_decisions = [], [], {}
            for case in cases:
                activity_id = case["activity"]["id"]
                rows = by_activity[activity_id]
                if not rows:
                    report["failed_activity_ids"].append(activity_id)
                    self._logger.log_json_format(scope, "activity_error", {
                        "activity_id": activity_id, "phase": "normalization",
                        "error": "missing_activity_decision",
                    })
                    continue
                candidates = {item["id"]: item for item in case["existing_intents"]}
                fact_ids = {fact["id"] for fact in case["facts"]}
                target_counts = Counter(row.get("target_intent_id") for _, row in rows
                                        if type(row.get("target_intent_id")) is int)
                for decision_index, row in rows:
                    try:
                        if set(row) != {"activity_id", "target_intent_id", "intent_lifecycle_observation", "field_updates", "reason"}:
                            raise ValueError("invalid_activity_decision_fields")
                        target_id = row["target_intent_id"]
                        if not isinstance(row["reason"], str):
                            raise ValueError("invalid_activity_reason")
                        if not isinstance(row["field_updates"], dict):
                            raise ValueError("invalid_activity_field_updates")
                        if target_id is None:
                            if row["intent_lifecycle_observation"] is not None or row["field_updates"] != {}:
                                raise ValueError("invalid_unmatched_activity_decision")
                            report["processed_activity_ids"].append(activity_id)
                            continue
                        if type(target_id) is not int or target_id not in candidates:
                            raise ValueError("activity_target_not_in_candidates")
                        if target_counts[target_id] > 1:
                            raise ValueError("duplicate_activity_target")
                        raw_observation = row["intent_lifecycle_observation"]
                        if raw_observation is None and not row["field_updates"]:
                            # A reliable match alone is not a lifecycle change.
                            report["processed_activity_ids"].append(activity_id)
                            continue
                        if not isinstance(raw_observation, dict):
                            raise ValueError("missing_activity_lifecycle_observation")
                        evidence_ids = raw_observation.get("evidence_fact_ids")
                        if (not isinstance(evidence_ids, list)
                                or any(type(value) is not int or value not in fact_ids for value in evidence_ids)):
                            raise ValueError("activity_fact_evidence_not_in_case")
                        observation = self._normalize_single_intent_lifecycle_observation(raw_observation)
                        timestamp = observation["effective_at"]
                        if timestamp and datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone() > as_of:
                            raise ValueError("activity_change_after_reference_time")
                        observation.update(source_activity_id=activity_id, target_intent_id=target_id,
                                           observed_at=case["activity"]["ended_at"])
                        target = candidates[target_id]
                        evidence_type = observation["lifecycle_evidence"]
                        # This path has already matched and assessed the old
                        # intent in one LLM pass. Derive its stage rather than
                        # asking for a second, potentially conflicting status.
                        update = None
                        latest = target.get("latest_event") or {}
                        change_at = latest.get("effective_at")
                        historical = (
                            latest.get("event_type") != "created" and bool(change_at)
                            and datetime.fromisoformat(observation["observed_at"].replace("Z", "+00:00")).astimezone()
                            < datetime.fromisoformat(change_at.replace("Z", "+00:00")).astimezone()
                        )
                        if not historical and (
                            evidence_type not in USER_INTENT_FIELD_ONLY_EVIDENCE_TYPES or row["field_updates"]
                        ):
                            update = {
                                "status": user_intent_status_after_evidence(target["status"], evidence_type),
                                "field_updates": row["field_updates"],
                                "effective_at": observation["effective_at"],
                            }
                        # Publish only a fully built decision; a failed sibling
                        # must not leave an incomplete validation case behind.
                        index = len(observations)
                        observations.append(observation)
                        validation_cases.append({"intent_lifecycle_observation": observation, "existing_intents": [target]})
                        raw_decisions[index] = [{"observation_index": index, "target_intent_id": target["id"],
                            "intent_update": update, "reason": (
                                "Activity predates the latest intent change" if historical else row["reason"]
                            )}]
                    except Exception as exc:
                        report["failed_activity_ids"].append(activity_id)
                        self._logger.log_json_format(scope, "decision_error", {
                            "decision_index": decision_index, "activity_id": activity_id,
                            "target_intent_id": row.get("target_intent_id"),
                            "phase": "normalization", "error": str(exc),
                        })
            # Conflict checks concern one intent, not unrelated decisions in
            # the same LLM response. Isolate normalization failures accordingly.
            indices_by_target: Dict[int, List[int]] = {}
            for index, observation in enumerate(observations):
                indices_by_target.setdefault(observation["target_intent_id"], []).append(index)
            for target_id, indices in indices_by_target.items():
                target_observations = [observations[index] for index in indices]
                try:
                    decisions = self._normalize_intent_lifecycle_decisions(
                        [validation_cases[index] for index in indices],
                        {local_index: [{**raw_decisions[index][0], "observation_index": local_index}]
                         for local_index, index in enumerate(indices)},
                        scope=scope,
                    )
                except Exception as exc:
                    report["failed_activity_ids"].extend(
                        observation["source_activity_id"] for observation in target_observations
                    )
                    self._logger.log_json_format(scope, "decision_error", {
                        "target_intent_id": target_id, "phase": "lifecycle_reconciliation",
                        "error": str(exc),
                    })
                    continue
                for index, observation in enumerate(target_observations):
                    try:
                        saved = self._save_intent_lifecycle_observations(
                            [observation], {0: decisions[index]} if index in decisions else {},
                            world_owner_id=owner_id, scope=scope,
                            reference_timestamp=reference_timestamp,
                        )
                        report["intent_event_ids"].extend(saved["intent_event_ids"])
                        report["processed_activity_ids"].extend(saved["completed_activity_ids"])
                        report["failed_activity_ids"].extend(saved["failed_activity_ids"])
                    except Exception as exc:
                        report["failed_activity_ids"].append(observation["source_activity_id"])
                        self._logger.log_json_format(scope, "decision_error", {
                            "activity_id": observation["source_activity_id"],
                            "target_intent_id": target_id, "phase": "lifecycle_observation_save",
                            "error": str(exc),
                        })
        report["intent_event_ids"] = sorted(set(report["intent_event_ids"]))
        report["failed_activity_ids"] = sorted(set(report["failed_activity_ids"]))
        completed = set(report["processed_activity_ids"])
        report["processed_activity_ids"] = sorted(completed - set(report["failed_activity_ids"]))
        if report["failed_activity_ids"] or has_decision_errors:
            # One activity may produce several independently committed intent
            # updates. A failed sibling must not hide its already applied events.
            report["status"] = "partial" if completed or report["intent_event_ids"] else "failed"
        elif report["intent_event_ids"]:
            report["status"] = "ok"
        self._generate_memory_recall_document_embeddings()
        self._logger.log_json_format(scope, "finish", report)
        return report

    def _process_memory_user_intent_update_task(
        self,
        *,
        reference_timestamp: Optional[Any] = None,
        signal_ids: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Load intent evidence and aggregate the two signal-update paths."""
        reference_timestamp = reference_timestamp or _now_text()
        selected_signal_ids = list(dict.fromkeys(
            int(value) for value in signal_ids or [] if int(value) > 0
        ))
        pending_signals = self._db.get_user_intent_signals_by_ids(
            selected_signal_ids,
        )
        facts = self._db.get_memory_facts_by_ids(
            sorted({int(signal["source_id"]) for signal in pending_signals if signal["source_type"] == "fact"})
        )
        report: Dict[str, Any] = {
            "status": "empty",
            "reference_timestamp": str(reference_timestamp),
            "intent_event_ids": [],
            "_completed_signal_ids": [],
            "failed_signal_ids": [],
        }
        facts_by_id = {
            int(fact["id"]): fact
            for fact in facts
            if str(fact.get("id") or "").strip().isdigit()
        }
        activity_contexts_by_fact = self._db.get_user_activity_contexts_for_facts(
            list(facts_by_id),
        )
        signal_items: List[Dict[str, Any]] = []
        for signal in pending_signals:
            if signal["source_type"] != "fact":
                continue
            fact_id = int(signal["source_id"])
            fact = facts_by_id.get(fact_id)
            if fact is None:
                continue
            signal_items.append({
                "signal_id": int(signal["signal_id"]),
                "fact_id": fact_id,
                "fact": fact,
                "signal": signal,
                "activity_contexts": activity_contexts_by_fact.get(fact_id, []),
            })
        self._log_user_intent_signals_loaded(signal_items, reference_timestamp)
        available_signal_ids = {item["signal_id"] for item in signal_items}
        report["failed_signal_ids"] = sorted(set(selected_signal_ids) - available_signal_ids)
        if report["failed_signal_ids"]:
            self._logger.log_json_format("memory_user_intent_update", "batch_error", {
                "phase": "input_validation",
                "signal_ids": report["failed_signal_ids"],
                "error": "missing_signal_or_supporting_fact",
            })
        if not signal_items:
            if report["failed_signal_ids"]:
                report["status"] = "failed"
            self._logger.log_json_format("memory_user_intent_update", "finish", report)
            return report

        world_owner_id = self._intent_world_owner_entity_id(reference_timestamp=reference_timestamp)
        future_signals = [item for item in signal_items
                          if item["signal"].get("signal_purpose") == "future_intent"]
        lifecycle_signals = [item for item in signal_items
                             if item["signal"].get("signal_purpose") == "lifecycle_update"]
        # Finish every future batch before retrieving targets for lifecycle updates.
        future_result = self._update_user_intents_using_future_intent_signals(
            future_signals, world_owner_id=world_owner_id, reference_timestamp=reference_timestamp,
        )
        lifecycle_result = self._update_user_intents_using_intent_lifecycle_observations(
            lifecycle_signals, world_owner_id=world_owner_id, reference_timestamp=reference_timestamp,
        )
        for result in (future_result, lifecycle_result):
            report["intent_event_ids"].extend(result["intent_event_ids"])
            report["_completed_signal_ids"].extend(result["completed_signal_ids"])
            report["failed_signal_ids"].extend(result["failed_signal_ids"])

        report["intent_event_ids"] = sorted(set(report["intent_event_ids"]))
        report["_completed_signal_ids"] = sorted(set(report["_completed_signal_ids"]))
        # Never silently lose an input that received no terminal handler result.
        report["failed_signal_ids"] = sorted(
            set(report["failed_signal_ids"])
            | (set(selected_signal_ids) - set(report["_completed_signal_ids"]))
        )
        report["status"] = (
            "partial" if report["failed_signal_ids"] and report["_completed_signal_ids"]
            else "failed" if report["failed_signal_ids"] else "ok"
        )
        self._generate_memory_recall_document_embeddings()
        self._logger.log_json_format("memory_user_intent_update", "finish", report)
        return report


    def _update_user_intents_using_future_intent_signals(
        self, signal_items: Sequence[Dict[str, Any]], *,
        world_owner_id: int, reference_timestamp: Any,
    ) -> Dict[str, Any]:
        """Batch future-intent signals through extraction, matching, validation and storage."""
        report: Dict[str, Any] = {
            "intent_event_ids": [], "completed_signal_ids": [], "failed_signal_ids": [],
        }
        facts_by_id = {item["fact_id"]: item["fact"] for item in signal_items}
        for batch_index, batch in enumerate(self._iter_user_intent_signal_batches(signal_items)):
            phase = "signal_reconciliation"
            try:
                cases = self._build_user_intent_signal_cases(batch, world_owner_id=world_owner_id)
                language = self._resolve_prompt_language_from_text("\n".join(
                    str(item["fact"].get("summary") or "") for item in batch
                ))
                results = self._call_user_intent_signal_llm(
                    cases, purpose="future_intent", prompt_language=language,
                    reference_timestamp=reference_timestamp,
                )
                phase = "normalization"
                normalized = self._normalize_intent_signal_decisions(
                    results, signal_cases=cases, world_owner_id=world_owner_id,
                )
                report["completed_signal_ids"].extend(normalized["empty_signal_ids"])
                report["failed_signal_ids"].extend(normalized["failed_signal_ids"])
                phase = "save"
                saved = self._save_future_intent_candidates(
                    normalized["future_intent_candidates"], normalized["future_intent_decisions"],
                    facts_by_id=facts_by_id, reference_timestamp=reference_timestamp,
                )
                report["completed_signal_ids"].extend(saved["completed_signal_ids"])
                report["failed_signal_ids"].extend(saved["failed_signal_ids"])
                report["intent_event_ids"].extend(saved["intent_event_ids"])
                self._logger.log_json_format("memory_user_intent_update", "batch_finish", {
                    "signal_purpose": "future_intent", "batch_index": batch_index,
                    "signal_ids": [item["signal_id"] for item in batch],
                    "intent_event_ids": saved["intent_event_ids"],
                    "failed_signal_ids": normalized["failed_signal_ids"] + saved["failed_signal_ids"],
                })
            except sqlite3.Error:
                raise
            except Exception as exc:
                report["failed_signal_ids"].extend(item["signal_id"] for item in batch)
                self._logger.log_json_format("memory_user_intent_update", "batch_error", {
                    "signal_purpose": "future_intent", "batch_index": batch_index,
                    "signal_ids": [item["signal_id"] for item in batch],
                    "phase": phase, "error_type": type(exc).__name__, "error": str(exc),
                })
        return report

    def _update_user_intents_using_intent_lifecycle_observations(
        self, signal_items: Sequence[Dict[str, Any]], *,
        world_owner_id: int, reference_timestamp: Any,
    ) -> Dict[str, Any]:
        """Batch lifecycle signals into observations and apply matched-intent updates."""
        report: Dict[str, Any] = {
            "intent_event_ids": [], "completed_signal_ids": [], "failed_signal_ids": [],
        }
        for batch_index, batch in enumerate(self._iter_user_intent_signal_batches(signal_items)):
            phase = "signal_reconciliation"
            try:
                cases = self._build_user_intent_signal_cases(batch, world_owner_id=world_owner_id)
                language = self._resolve_prompt_language_from_text("\n".join(
                    str(item["fact"].get("summary") or "") for item in batch
                ))
                results = self._call_user_intent_signal_llm(
                    cases, purpose="lifecycle_update", prompt_language=language,
                    reference_timestamp=reference_timestamp,
                )
                phase = "normalization"
                normalized = self._normalize_intent_signal_decisions(
                    results, signal_cases=cases, world_owner_id=world_owner_id,
                )
                report["completed_signal_ids"].extend(normalized["empty_signal_ids"])
                report["failed_signal_ids"].extend(normalized["failed_signal_ids"])
                phase = "save"
                saved = self._save_intent_lifecycle_observations(
                    normalized["intent_lifecycle_observations"], normalized["lifecycle_decisions"],
                    world_owner_id=world_owner_id, reference_timestamp=reference_timestamp,
                )
                report["completed_signal_ids"].extend(saved["completed_signal_ids"])
                report["failed_signal_ids"].extend(saved["failed_signal_ids"])
                report["intent_event_ids"].extend(saved["intent_event_ids"])
                self._logger.log_json_format("memory_user_intent_update", "batch_finish", {
                    "signal_purpose": "lifecycle_update", "batch_index": batch_index,
                    "signal_ids": [item["signal_id"] for item in batch],
                    "intent_event_ids": saved["intent_event_ids"],
                    "failed_signal_ids": normalized["failed_signal_ids"] + saved["failed_signal_ids"],
                })
            except sqlite3.Error:
                raise
            except Exception as exc:
                report["failed_signal_ids"].extend(item["signal_id"] for item in batch)
                self._logger.log_json_format("memory_user_intent_update", "batch_error", {
                    "signal_purpose": "lifecycle_update", "batch_index": batch_index,
                    "signal_ids": [item["signal_id"] for item in batch],
                    "phase": phase, "error_type": type(exc).__name__, "error": str(exc),
                })
        return report

    def _iter_user_intent_signal_batches(
        self, signal_items: Sequence[Dict[str, Any]],
    ) -> Iterable[List[Dict[str, Any]]]:
        """Keep each fact's signals together; the signal budget is a soft limit."""
        batch_size = max(1, int(self._memory_cfg.get("user_intent_signal_batch_size") or 8))
        signals_by_fact: Dict[int, List[Dict[str, Any]]] = {}
        for item in signal_items:
            signals_by_fact.setdefault(item["fact_id"], []).append(item)
        batch: List[Dict[str, Any]] = []
        for fact_signals in signals_by_fact.values():
            if batch and len(batch) + len(fact_signals) > batch_size:
                yield batch
                batch = []
            batch.extend(fact_signals)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _format_fact_cases_for_user_intent_prompt(
        self,
        signal_items: Sequence[Dict[str, Any]],
    ) -> str:
        """Render each fact once, with its selected signals and per-signal targets."""
        if not signal_items:
            return "[]"
        cases_by_fact: Dict[int, Dict[str, Any]] = {}
        for item in signal_items:
            case = cases_by_fact.setdefault(item["fact_id"], {
                "fact": self._format_fact_view_for_user_intent_prompt(item["fact"]),
                "activity_context_ids": [],
                "user_intent_signals": [],
            })
            for activity in item.get("activity_contexts") or []:
                activity_id = int(activity["activity_id"])
                if activity_id not in case["activity_context_ids"]:
                    case["activity_context_ids"].append(activity_id)
            signal = item["signal"]
            case["user_intent_signals"].append({
                "signal_id": item["signal_id"],
                **{
                    key: signal.get(key)
                    for key in (
                        "signal_purpose", "subject_entity",
                        "user_role", "user_intent_anchor", "assertion_source",
                        "evidence_basis", "confidence",
                    )
                },
                **({"existing_intents": [
                    self._intent_object_prompt_view(intent) for intent in item["existing_intents"]
                ]} if "existing_intents" in item else {}),
            })
        return "[\n" + ",\n".join(
            json.dumps(case, ensure_ascii=False, separators=(",", ":"))
            for case in cases_by_fact.values()
        ) + "\n]"

    def _build_user_intent_signal_cases(
        self, signal_items: Sequence[Dict[str, Any]], *, world_owner_id: int,
    ) -> List[Dict[str, Any]]:
        """Retrieve an independent set of old intents for each evidence-backed signal."""
        existing = self._db.get_intent_objects(world_owner_entity_id=world_owner_id, limit=360)
        cases = []
        for item in signal_items:
            signal = item["signal"]
            anchor = _compact_whitespace(signal.get("user_intent_anchor") or "")
            texts = self._normalize_unique_labels([
                anchor, signal.get("evidence_basis"), item["fact"].get("summary"),
            ], limit=8)
            ranked = []
            for intent in existing:
                score = max((self._score_intent_candidate_against_existing_object(
                    {"summary": text, "world_owner_entity_id": world_owner_id}, intent,
                ) for text in texts), default=0.0)
                if score >= 0.2:
                    ranked.append((score, intent))
            ranked.sort(key=lambda pair: (-pair[0], -int(pair[1]["id"])))
            cases.append({**item, "existing_intents": [intent for _, intent in ranked[:6]]})
        return cases

    def _build_intent_object_match_profile(self, item: Dict[str, Any]) -> Dict[str, Any]:
        details = item.get("details") or {}
        return {
            "text_values": self._normalize_unique_labels([
                item.get("summary"), details.get("completion_condition"),
                details.get("user_role"), details.get("location_text"),
            ], limit=6),
            "entity_id": int(item.get("world_owner_entity_id") or 0),
            "time_value": item.get("time_start") or item.get("time_end") or "",
            "canonical_key": self._generate_topic_name_key(item["canonical_key"]) if item.get("canonical_key") else "",
        }

    def _score_intent_candidate_against_existing_object(
        self,
        candidate: Dict[str, Any],
        existing: Dict[str, Any],
    ) -> float:
        """Rank a related existing intent without requiring key equality."""
        candidate_profile = self._build_intent_object_match_profile(candidate)
        existing_profile = self._build_intent_object_match_profile(existing)
        text_score = self._topic_name_best_pair_similarity(
            candidate_profile["text_values"],
            existing_profile["text_values"],
        )
        canonical_score = self._topic_name_best_pair_similarity(
            [candidate_profile["canonical_key"]],
            [existing_profile["canonical_key"]],
        )
        candidate_entity_id = int(candidate_profile["entity_id"] or 0)
        existing_entity_id = int(existing_profile["entity_id"] or 0)
        entity_score = (
            1.0
            if candidate_entity_id and candidate_entity_id == existing_entity_id
            else 0.0
        )
        candidate_day = str(candidate_profile["time_value"] or "")[:10]
        existing_day = str(existing_profile["time_value"] or "")[:10]
        time_score = 1.0 if candidate_day and candidate_day == existing_day else 0.0
        return round(
            min(
                1.0,
                text_score * 0.6
                + canonical_score * 0.25
                + entity_score * 0.1
                + time_score * 0.05,
            ),
            6,
        )

    def _intent_world_owner_entity_id(self, *, reference_timestamp: Any = None) -> int:
        mapping = self._db.add_entity_names([self._world_owner_entity_name], reference_timestamp=reference_timestamp)
        return int(mapping[self._world_owner_entity_name])

    def _format_fact_view_for_user_intent_prompt(self, fact: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": fact.get("id"), "summary": fact.get("summary") or "",
            "fact_type": fact.get("fact_type") or "",
            "event_time": fact.get("event_time_key") or "",
            "dialogue_time": fact.get("dialogue_time_key") or "",
        }

    def _call_user_intent_signal_llm(
        self, cases: Sequence[Dict[str, Any]], *, purpose: str,
        prompt_language: str, reference_timestamp: Any,
    ) -> Dict[int, Dict[str, Any]]:
        """Assess fact-centered input, retaining one output decision per signal."""
        if purpose == "future_intent":
            template = (USER_INTENT_UPDATE_FROM_FUTURE_INTENT_SIGNALS_PROMPT_EN if prompt_language == "en"
                        else USER_INTENT_UPDATE_FROM_FUTURE_INTENT_SIGNALS_PROMPT_ZH)
            stage = "memory_future_intent_reconciliation"
        else:
            template = (USER_INTENT_UPDATE_FROM_INTENT_LIFECYCLE_OBSERVATIONS_PROMPT_EN if prompt_language == "en"
                        else USER_INTENT_UPDATE_FROM_INTENT_LIFECYCLE_OBSERVATIONS_PROMPT_ZH)
            stage = "memory_intent_lifecycle_reconciliation"
        prompt = (template.replace("{reference_timestamp}", str(reference_timestamp))
                  .replace("{activity_contexts}", self._format_activity_contexts_for_signal_prompt(cases))
                  .replace("{fact_cases}", self._format_fact_cases_for_user_intent_prompt(cases)))
        raw = self._llm_client.call_llm(prompt, stage_name=stage)
        parsed = self._llm_client.parse_json_object_from_llm_text(raw or "", stage_name=stage)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("decisions"), list):
            raise ValueError("invalid_user_intent_signal_response")
        expected = {item["signal_id"] for item in cases}
        by_signal: Dict[int, List[Dict[str, Any]]] = {}
        for index, result in enumerate(parsed["decisions"]):
            if (not isinstance(result, dict) or type(result.get("signal_id")) is not int
                    or result["signal_id"] not in expected):
                self._logger.log_json_format("memory_user_intent_update", "signal_error", {
                    "result_index": index, "phase": purpose, "error": "invalid_or_unknown_signal_id",
                })
                continue
            by_signal.setdefault(result["signal_id"], []).append(result)
        return {signal_id: rows[0] for signal_id, rows in by_signal.items() if len(rows) == 1}

    def _normalize_intent_signal_decisions(
        self, results: Dict[int, Dict[str, Any]], *,
        signal_cases: Sequence[Dict[str, Any]], world_owner_id: int,
    ) -> Dict[str, Any]:
        """Validate final per-signal decisions, reusing the existing storage contracts."""
        candidates, observations, lifecycle_cases = [], [], []
        future_decisions, raw_lifecycle_decisions = {}, {}
        empty_signal_ids, failed_signal_ids = [], []
        for case in signal_cases:
            signal_id = case["signal_id"]
            purpose = case["signal"]["signal_purpose"]
            try:
                result = results.get(signal_id)
                if result is None:
                    raise ValueError("missing_or_duplicate_signal_decision")
                existing_ids = {item["id"] for item in case["existing_intents"]}
                target_id = result.get("target_intent_id")
                if target_id is not None and (type(target_id) is not int or target_id not in existing_ids):
                    raise ValueError("intent_target_not_in_signal_case")
                fact_id = case["fact_id"]
                update = result.get("existing_intent_update")
                if update is not None and target_id is None:
                    raise ValueError("intent_update_requires_matched_target")
                if purpose == "future_intent":
                    if (set(result) != {"signal_id", "new_intent", "target_intent_id",
                                      "relation", "existing_intent_update", "reason"}
                            or not isinstance(result["reason"], str)):
                        raise ValueError("invalid_future_intent_decision_fields")
                    relation = result["relation"]
                    if target_id is None:
                        if relation is not None:
                            raise ValueError("unmatched_future_intent_has_relation_or_update")
                    elif relation not in {"same_intent", "supersedes", "conflicts"}:
                        raise ValueError("invalid_future_intent_target_or_relation")
                    raw_new_intent = result["new_intent"]
                    if relation == "same_intent" and raw_new_intent is not None:
                        raise ValueError("same_intent_must_not_create_new_intent")
                    candidate = {
                        "signal_id": signal_id, "world_owner_entity_id": world_owner_id,
                        "evidence_fact_ids": [fact_id], "activity_contexts": case.get("activity_contexts") or [],
                    }
                    new_intent = None
                    if raw_new_intent is not None:
                        if (not isinstance(raw_new_intent, dict) or raw_new_intent.get("evidence_fact_ids") != [fact_id]
                                or any(type(value) is not int for value in raw_new_intent["evidence_fact_ids"])):
                            raise ValueError("future_intent_requires_current_fact_evidence")
                        new_intent = self._normalize_single_intent_candidate(
                            raw_new_intent, facts_by_id={fact_id: case["fact"]}, world_owner_id=world_owner_id,
                        )
                    normalized_update = self._normalize_intent_update(update) if update is not None else None
                    index = len(candidates)
                    candidates.append(candidate)
                    future_decisions[index] = {
                        "new_intent": new_intent, "target_intent_id": target_id,
                        "relation": relation, "existing_intent_update": normalized_update,
                        "reason": _compact_whitespace(result["reason"])[:480],
                    }
                else:
                    if (set(result) != {"signal_id", "intent_lifecycle_observation",
                                      "target_intent_id", "existing_intent_update", "reason"}
                            or not isinstance(result["reason"], str)):
                        raise ValueError("invalid_lifecycle_decision_fields")
                    raw_observation = result["intent_lifecycle_observation"]
                    if raw_observation is None:
                        if update is not None:
                            raise ValueError("intent_update_requires_lifecycle_observation")
                        empty_signal_ids.append(signal_id)
                        continue
                    if (not isinstance(raw_observation, dict)
                            or raw_observation.get("evidence_fact_ids") != [fact_id]
                            or any(type(value) is not int for value in raw_observation["evidence_fact_ids"])):
                        raise ValueError("lifecycle_observation_requires_current_fact_evidence")
                    observation = self._normalize_single_intent_lifecycle_observation(raw_observation)
                    observation["signal_id"] = signal_id
                    # Validate before publishing shared buffers so one malformed
                    # result cannot leave a half-built observation behind.
                    normalized_update = self._normalize_intent_update(update) if update is not None else None
                    index = len(observations)
                    observations.append(observation)
                    lifecycle_cases.append({
                        "intent_lifecycle_observation": observation,
                        "existing_intents": case["existing_intents"],
                    })
                    raw_lifecycle_decisions[index] = [{
                        "observation_index": index, "target_intent_id": target_id,
                        "intent_update": normalized_update, "reason": result["reason"],
                    }]
            except sqlite3.Error:
                raise
            except Exception as exc:
                failed_signal_ids.append(signal_id)
                self._logger.log_json_format("memory_user_intent_update", "signal_error", {
                    "signal_id": signal_id, "fact_id": case["fact_id"],
                    "signal_purpose": purpose, "phase": "normalization", "error": str(exc),
                })
        return {
            "future_intent_candidates": candidates,
            "future_intent_decisions": self._filter_conflicting_future_intent_decisions(candidates, future_decisions),
            "intent_lifecycle_observations": observations,
            "lifecycle_decisions": self._normalize_intent_lifecycle_decisions(
                lifecycle_cases, raw_lifecycle_decisions,
            ),
            "empty_signal_ids": empty_signal_ids, "failed_signal_ids": failed_signal_ids,
        }

    def _normalize_single_intent_lifecycle_observation(
        self, raw: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize observed changes without reconstructing or updating an intent."""
        if not isinstance(raw, dict):
            raise ValueError(f"field=observation reason=invalid_type actual={raw!r} expected=object")
        for key in (
            "target_description", "observation_text", "lifecycle_evidence",
        ):
            if not isinstance(raw.get(key), str):
                raise ValueError(
                    f"field={key} reason=invalid_type actual={raw.get(key)!r} expected=string"
                )
        if not isinstance(raw.get("effective_at", ""), str):
            raise ValueError(
                f"field=effective_at reason=invalid_type actual={raw.get('effective_at')!r} expected=string"
            )
        lifecycle_evidence = str(raw.get("lifecycle_evidence") or "").strip().lower()
        target = _compact_whitespace(raw.get("target_description") or "")[:480]
        text = _compact_whitespace(raw.get("observation_text") or "")[:800]
        if not target:
            raise ValueError("field=target_description reason=empty_value actual='' expected=nonempty_text")
        if not text:
            raise ValueError("field=observation_text reason=empty_value actual='' expected=nonempty_text")
        if lifecycle_evidence not in USER_INTENT_LIFECYCLE_EVIDENCE_TYPES:
            raise ValueError(
                f"field=lifecycle_evidence reason=invalid_enum actual={raw.get('lifecycle_evidence')!r} "
                f"expected={sorted(USER_INTENT_LIFECYCLE_EVIDENCE_TYPES)!r}"
            )
        effective_at = _compact_whitespace(raw.get("effective_at") or "")
        if effective_at:
            try:
                datetime.fromisoformat(effective_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(
                    f"field=effective_at reason=invalid_datetime actual={effective_at!r} expected=ISO_datetime_or_empty"
                ) from exc
        raw_details = raw.get("details", {})
        if not isinstance(raw_details, dict):
            raise ValueError(
                f"field=details reason=invalid_type actual={raw_details!r} expected=object"
            )
        details: Dict[str, str] = {}
        for key in ("time_start", "time_end", "location_text", "progress_text"):
            value = raw_details.get(key)
            if value is None or value == "":
                continue
            if not isinstance(value, str):
                raise ValueError(
                    f"field=details.{key} reason=invalid_type actual={value!r} expected=string_or_empty"
                )
            text_value = _compact_whitespace(value)[:480]
            if text_value:
                details[key] = text_value
        return {
            "target_description": target,
            "observation_text": text,
            "lifecycle_evidence": lifecycle_evidence,
            "effective_at": effective_at,
            "confidence": self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.7),
            "evidence_fact_ids": list(raw["evidence_fact_ids"]),
            "details": details,
        }

    def _normalize_single_intent_candidate(
        self, raw: Any, *, facts_by_id: Dict[int, Dict[str, Any]],
        world_owner_id: int,
    ) -> Dict[str, Any]:
        """Validate a future intent without requiring a category or completion anchor."""
        if not isinstance(raw, dict):
            raise ValueError("field=candidate reason=invalid_type expected=object")
        summary = _compact_whitespace(raw.get("summary") or "")
        domain = self._normalize_primary_domain(raw.get("primary_domain"))
        evidence_ids = list(dict.fromkeys(int(v) for v in raw.get("evidence_fact_ids", [])
                                        if type(v) is int and v in facts_by_id))
        if not summary or not domain or not evidence_ids:
            raise ValueError("Intent requires a summary, valid domain and supporting facts")
        owner_supported = any(bool((facts_by_id[v].get("metadata") or {}).get("has_world_owner_speaker")
                                  or (facts_by_id[v].get("metadata") or {}).get("has_world_owner_alias_reference"))
                              for v in evidence_ids)
        if not owner_supported:
            raise ValueError("field=evidence_fact_ids reason=owner_reference_not_supported")
        details = normalize_user_intent_details(raw.get("details", {}))
        times = {}
        for key in ("time_start", "time_end"):
            value = raw.get(key, "")
            if not isinstance(value, str):
                raise ValueError(f"field={key} reason=invalid_type expected=string")
            value = value.strip()
            times[key] = value
        return {
            "summary": summary, "subject_entity": self._world_owner_entity_name,
            "world_owner_entity_id": world_owner_id,
            "canonical_key": self._generate_topic_name_key(raw.get("canonical_key") or summary),
            "primary_domain": domain, "confidence": self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.7),
            "details": details, "evidence_fact_ids": evidence_ids, **times,
        }

    @staticmethod
    def _normalize_intent_update(update: Any) -> Dict[str, Any]:
        """Validate the shared sparse lifecycle patch for either source path."""
        if (not isinstance(update, dict) or set(update) != {"status", "field_updates", "effective_at"}
                or update["status"] not in USER_INTENT_STATUSES):
            raise ValueError("invalid_intent_update")
        patch = update["field_updates"]
        if (not isinstance(patch, dict)
                or set(patch) - {"summary", "current_progress", "time_start", "time_end", "details"}):
            raise ValueError("invalid_intent_field_updates")
        normalized = {}
        for key, value in patch.items():
            if key == "details":
                details = normalize_user_intent_details(value)
                if set(details) != set(value):
                    raise ValueError("intent_patch_must_not_clear_details")
                normalized[key] = details
            elif not isinstance(value, str) or not value.strip():
                raise ValueError("intent_patch_must_not_clear_values")
            else:
                normalized[key] = value.strip()
        effective_at = update["effective_at"]
        if not isinstance(effective_at, str):
            raise ValueError("invalid_intent_effective_at")
        effective_at = effective_at.strip()
        if effective_at:
            datetime.fromisoformat(effective_at.replace("Z", "+00:00"))
        return {"status": update["status"], "field_updates": normalized, "effective_at": effective_at}

    def _filter_conflicting_future_intent_decisions(
        self, candidates: Sequence[Dict[str, Any]], decisions: Dict[int, Dict[str, Any]],
    ) -> Dict[int, Dict[str, Any]]:
        """Reject conflicting sparse updates without discarding other signal results."""
        # Do not apply contradictory patches to the same object in list order.
        updates_by_target: Dict[int, Dict[str, Any]] = {}
        conflicting_targets = set()
        for decision in decisions.values():
            update = decision["existing_intent_update"]
            if update is None:
                continue
            target_id = decision["target_intent_id"]
            fields = {"status": update["status"]}
            for field, value in update["field_updates"].items():
                if field == "details":
                    fields.update({f"details.{key}": detail for key, detail in value.items()})
                else:
                    fields[field] = value
            previous = updates_by_target.setdefault(target_id, {})
            if any(key in previous and previous[key] != value for key, value in fields.items()):
                conflicting_targets.add(target_id)
            previous.update(fields)
        for index in list(decisions):
            if (decisions[index]["existing_intent_update"] is not None
                    and decisions[index]["target_intent_id"] in conflicting_targets):
                del decisions[index]
                self._logger.log_json_format("memory_user_intent_update", "signal_error", {
                    "signal_id": candidates[index].get("signal_id"), "candidate_index": index,
                    "phase": "future_intent_reconciliation", "error": "conflicting_future_intent_updates",
                })
        return decisions

    @staticmethod
    def _intent_object_prompt_view(item: Dict[str, Any]) -> Dict[str, Any]:
        return {key: item.get(key) for key in (
            "id", "summary", "status", "current_progress", "time_start", "time_end", "details",
        )}

    def _normalize_intent_lifecycle_decisions(
        self, cases: Sequence[Dict[str, Any]], decisions_by_index: Dict[int, List[Dict[str, Any]]],
        *, scope: str = "memory_user_intent_update",
    ) -> Dict[int, Dict[str, Any]]:
        """Apply the same lifecycle patch and conflict checks to both evidence sources."""
        decisions: Dict[int, Dict[str, Any]] = {}

        for index, case in enumerate(cases):
            try:
                if not case["existing_intents"]:
                    decision = {"target_intent_id": None, "intent_update": None, "reason": "no_related_intent"}
                else:
                    results = decisions_by_index.get(index, [])
                    if len(results) != 1:
                        raise ValueError("missing_or_duplicate_lifecycle_matching_decision")
                    decision = results[0]
                    if set(decision) != {"observation_index", "target_intent_id", "intent_update", "reason"}:
                        raise ValueError("invalid_lifecycle_decision_fields")
                if not isinstance(decision.get("reason"), str):
                    raise ValueError("invalid_lifecycle_decision_fields")
                target_id = decision["target_intent_id"]
                update = decision["intent_update"]
                normalized_update = None
                if target_id is not None:
                    if (type(target_id) is not int or target_id <= 0
                            or target_id not in {item["id"] for item in case["existing_intents"]}):
                        raise ValueError("invalid_lifecycle_matching_target")
                if update is not None:
                    if target_id is None:
                        raise ValueError("intent_update_requires_matched_target")
                    normalized_update = self._normalize_intent_update(update)
                decisions[index] = {
                    "target_intent_id": target_id, "intent_update": normalized_update,
                    "reason": _compact_whitespace(decision["reason"])[:480],
                }
            except Exception as exc:
                self._logger.log_json_format(scope, "signal_error", {
                    "observation_index": index, "phase": "lifecycle_reconciliation", "error": str(exc),
                })
        # Do not let incompatible batch updates overwrite one another in list order.
        updates_by_target: Dict[int, Dict[str, Any]] = {}
        conflicting_targets = set()
        for index, decision in decisions.items():
            update = decision["intent_update"]
            if update is None:
                continue
            target_id = decision["target_intent_id"]
            observation = cases[index]["intent_lifecycle_observation"]
            field_only_activity = (
                observation.get("source_activity_id") is not None
                and observation["lifecycle_evidence"] in USER_INTENT_FIELD_ONLY_EVIDENCE_TYPES
            )
            fields = {} if field_only_activity else {"status": update["status"]}
            for field, value in update["field_updates"].items():
                if field == "details":
                    fields.update({f"details.{key}": detail for key, detail in value.items()})
                else:
                    fields[field] = value
            previous = updates_by_target.setdefault(target_id, {})
            if any(key in previous and previous[key] != value for key, value in fields.items()):
                conflicting_targets.add(target_id)
            previous.update(fields)
        for index in list(decisions):
            if (decisions[index]["intent_update"] is not None
                    and decisions[index]["target_intent_id"] in conflicting_targets):
                del decisions[index]
                self._logger.log_json_format(scope, "signal_error", {
                    "observation_index": index, "phase": "lifecycle_reconciliation",
                    "error": "conflicting_lifecycle_updates",
                })
        return decisions

    def _save_intent_lifecycle_observations(
        self, observations: Sequence[Dict[str, Any]], decisions: Dict[int, Dict[str, Any]], *,
        world_owner_id: int, scope: str = "memory_user_intent_update",
        reference_timestamp: Any = None,
    ) -> Dict[str, Any]:
        """Atomically save either source's observation, applied changes and evidence."""
        report: Dict[str, Any] = {
            "intent_lifecycle_observation_ids": [],
            "intent_event_ids": [],
            "unmatched_observation_ids": [], "completed_signal_ids": [], "failed_signal_ids": [],
            "completed_activity_ids": [], "failed_activity_ids": [],
        }
        for index, observation in enumerate(observations):
            signal_id = observation.get("signal_id")
            activity_id = observation.get("source_activity_id")
            target_id = observation.get("target_intent_id")
            completed_key = "completed_signal_ids" if signal_id is not None else "completed_activity_ids"
            failed_key = "failed_signal_ids" if signal_id is not None else "failed_activity_ids"
            source_id = signal_id if signal_id is not None else activity_id
            try:
                if (signal_id is None) == (activity_id is None):
                    raise ValueError("observation_requires_exactly_one_source")
                recorded_observation = (
                    self._db.get_activity_intent_lifecycle_observation(activity_id, target_id)
                    if activity_id is not None else None
                )
                recorded_events = self._db.get_user_intent_events(
                    source_signal_id=signal_id, source_activity_id=activity_id,
                )
                if activity_id is not None:
                    recorded_events = [event for event in recorded_events if event["intent_id"] == target_id]
                if recorded_events:
                    report["intent_event_ids"].extend(event["id"] for event in recorded_events)
                    report["intent_lifecycle_observation_ids"].extend(
                        sorted({event["lifecycle_observation_id"] for event in recorded_events
                                if event["lifecycle_observation_id"] is not None})
                    )
                    report[completed_key].append(source_id)
                    continue
                if recorded_observation:
                    report["intent_lifecycle_observation_ids"].append(recorded_observation["id"])
                    report[completed_key].append(source_id)
                    continue
                decision = decisions.get(index)
                if decision is None:
                    raise ValueError("missing_valid_lifecycle_matching_decision")
                target_id = decision["target_intent_id"]
                if activity_id is not None and target_id != observation["target_intent_id"]:
                    raise ValueError("activity_observation_target_mismatch")
                update = decision["intent_update"]
                event_id = None
                with self._db.transaction():
                    observation_id = self._db.upsert_user_intent_lifecycle_observation(observation, reference_timestamp=reference_timestamp)
                    if target_id is not None:

                        target = self._db.get_intent_object(intent_id=target_id)
                        if (not target
                                or target["world_owner_entity_id"] != world_owner_id):
                            raise ValueError("invalid_lifecycle_matching_target")
                        if update:
                            if activity_id is not None:
                                latest = self._db.get_latest_user_intent_event(target_id)
                                change_at = (latest or {}).get("effective_at") or ""
                                observed_at = observation.get("observed_at") or ""
                                if change_at and observed_at and (latest or {}).get("event_type") != "created":
                                    change_time = datetime.fromisoformat(change_at.replace("Z", "+00:00")).astimezone()
                                    observed_time = datetime.fromisoformat(observed_at.replace("Z", "+00:00")).astimezone()
                                    if observed_time < change_time:
                                        update = None
                                        decision = {**decision, "intent_update": None,
                                                    "reason": "Activity predates the latest intent change"}
                        if update:
                            patch = update["field_updates"]
                            payload = self._intent_storage_payload({
                                **target, **patch,
                                "details": {**target["details"], **patch.get("details", {})},
                            }, status=update["status"])
                            # Resolve against persisted state, including earlier
                            # batch commits, failed siblings and chronology checks.
                            if activity_id is not None:
                                update["status"] = user_intent_status_after_evidence(
                                    target["status"], observation["lifecycle_evidence"],
                                )
                            payload["status"] = update["status"]
                            if any(payload[key] != target.get(key) for key in payload):
                                if not self._db.update_intent_object(
                                    intent_id=target_id, payload=payload,
                                    reference_timestamp=reference_timestamp,
                                ):
                                    raise ValueError("lifecycle_intent_update_failed")
                                updated_target = self._db.get_intent_object(
                                    intent_id=target_id,
                                )
                                if not updated_target:
                                    raise ValueError("updated_lifecycle_intent_missing")
                                event_id = self._db.insert_user_intent_event(
                                   intent_id=target_id,
                                    previous_payload=target,
                                    new_payload=updated_target,
                                    source_signal_id=signal_id,
                                    source_activity_id=activity_id,
                                    lifecycle_observation_id=observation_id,
                                    effective_at=update["effective_at"],
                                    reference_timestamp=reference_timestamp,
                                )
                        facts = self._db.get_memory_facts_by_ids(observation["evidence_fact_ids"])
                        if len(facts) != len(observation["evidence_fact_ids"]):
                            raise ValueError("missing_lifecycle_evidence_fact")
                        self._db.upsert_intent_evidence([{
                            "intent_id": target_id,
                            "evidence_type": "fact", "evidence_id": fact["id"],
                            "role": observation["lifecycle_evidence"],
                            "observed_at": str(fact.get("dialogue_time_key") or fact.get("event_time_key") or ""),
                        } for fact in facts], reference_timestamp=reference_timestamp)
                        if activity_id is not None:
                            self._db.upsert_intent_evidence([{
                                "intent_id": target_id,
                                "evidence_type": "user_activity", "evidence_id": activity_id,
                                "role": observation["lifecycle_evidence"],
                                "observed_at": observation.get("observed_at") or "",
                            }], reference_timestamp=reference_timestamp)
                    if event_id is not None:
                        self._upsert_memory_user_intents_into_recall_document_table(
                            [target_id], reference_timestamp=reference_timestamp,
                        )
            except Exception as exc:
                report[failed_key].append(source_id)
                self._logger.log_json_format(scope, "signal_error", {
                    "signal_id": signal_id, "activity_id": activity_id,
                    "phase": "lifecycle_observation_save", "error": str(exc),
                })
                continue
            report["intent_lifecycle_observation_ids"].append(observation_id)
            report[completed_key].append(source_id)
            if event_id is not None:
                report["intent_event_ids"].append(event_id)
            if target_id is None:
                report["unmatched_observation_ids"].append(observation_id)
            self._logger.log_json_format(scope, "lifecycle_relation", {
                **decision, "signal_id": signal_id, "activity_id": activity_id,
                "observation_id": observation_id, "intent_event_id": event_id,
            })
        return report

    def _save_future_intent_candidates(
        self,
        candidates: Sequence[Dict[str, Any]],
        decisions: Dict[int, Dict[str, Any]],
        *,
        facts_by_id: Dict[int, Dict[str, Any]],
        reference_timestamp: Any,
    ) -> Dict[str, Any]:
        """Atomically save new intents, existing-intent changes, and their fact evidence."""
        report: Dict[str, Any] = {
            "intent_event_ids": [],
            "completed_signal_ids": [],
            "failed_signal_ids": [],
        }
        for index, candidate in enumerate(candidates):
            signal_id = candidate["signal_id"]
            try:
                recorded_events = self._db.get_user_intent_events(source_signal_id=signal_id)
                if recorded_events:
                    report["intent_event_ids"].extend(event["id"] for event in recorded_events)
                    report["completed_signal_ids"].append(signal_id)
                    continue
                signal_event_ids: List[int] = []
                fact = facts_by_id[candidate["evidence_fact_ids"][0]]
                observed_at = _to_timestamp_text(
                    fact.get("dialogue_time_key")
                    or fact.get("event_time_key")
                    or reference_timestamp
                )
                decision = decisions.get(index)
                if decision is None:
                    raise ValueError("missing_valid_future_intent_decision")

                target_id = decision["target_intent_id"]
                update = decision["existing_intent_update"]
                changed_intent_ids: List[int] = []
                with self._db.transaction():
                    target = self._db.get_intent_object(
                        intent_id=target_id,
                    ) if target_id is not None else None
                    if target_id is not None and (
                        not target or target["world_owner_entity_id"] != candidate["world_owner_entity_id"]
                    ):
                        raise ValueError("future_intent_target_missing_or_owner_mismatch")
                    if decision["new_intent"] is not None:
                        object_id = self._db.create_intent_object(
                            payload=self._intent_storage_payload(decision["new_intent"], status="pending"),
                            reference_timestamp=reference_timestamp,
                        )
                        stored_object = self._db.get_intent_object(
                            intent_id=object_id,
                        )
                        if not stored_object:
                            raise ValueError("created_future_intent_missing")
                        self._db.upsert_intent_evidence([{
                            "intent_id": object_id,
                            "evidence_type": "fact", "evidence_id": fact_id,
                            "role": "creation", "observed_at": observed_at,
                        } for fact_id in candidate["evidence_fact_ids"]], reference_timestamp=reference_timestamp)
                        signal_event_ids.append(self._db.insert_user_intent_event(
                           intent_id=object_id,
                            event_type="created",
                            new_payload=stored_object,
                            source_signal_id=signal_id,
                            reference_timestamp=reference_timestamp,
                        ))
                        changed_intent_ids.append(object_id)
                    if target:
                        if update:
                            patch = update["field_updates"]
                            payload = self._intent_storage_payload({
                                **target, **patch,
                                "details": {**target["details"], **patch.get("details", {})},
                            }, status=update["status"])
                            payload["status"] = update["status"]
                            if any(payload[key] != target.get(key) for key in payload):
                                if not self._db.update_intent_object(
                                    intent_id=target_id, payload=payload,
                                    reference_timestamp=reference_timestamp,
                                ):
                                    raise ValueError("future_intent_update_failed")
                                updated_target = self._db.get_intent_object(
                                    intent_id=target_id,
                                )
                                if not updated_target:
                                    raise ValueError("updated_future_intent_missing")
                                signal_event_ids.append(self._db.insert_user_intent_event(
                                   intent_id=target_id,
                                    previous_payload=target,
                                    new_payload=updated_target,
                                    source_signal_id=signal_id,
                                    effective_at=update["effective_at"],
                                    reference_timestamp=reference_timestamp,
                                ))
                                changed_intent_ids.append(target_id)
                        self._db.upsert_intent_evidence([{
                            "intent_id": target_id,
                            "evidence_type": "fact", "evidence_id": fact_id,
                            "role": decision["relation"], "observed_at": observed_at,
                        } for fact_id in candidate["evidence_fact_ids"]], reference_timestamp=reference_timestamp)
                    if changed_intent_ids:
                        self._upsert_memory_user_intents_into_recall_document_table(
                            changed_intent_ids, reference_timestamp=reference_timestamp,
                        )
            except Exception as exc:
                report["failed_signal_ids"].append(signal_id)
                self._logger.log_json_format("memory_user_intent_update", "signal_error", {
                    "signal_id": signal_id, "error": str(exc),
                })
                continue
            report["completed_signal_ids"].append(signal_id)
            report["intent_event_ids"].extend(signal_event_ids)
            self._logger.log_json_format("memory_user_intent_update", "future_intent_saved", {
                "candidate_index": index, "signal_id": signal_id,
                "intent_event_ids": signal_event_ids, **decision,
            })
        return report

    def _intent_storage_payload(self, candidate: Dict[str, Any], *, status: str) -> Dict[str, Any]:
        return {**{key: candidate[key] for key in USER_INTENT_FIELDS if key in candidate}, "status": status}

    @staticmethod
    def _user_claim_types() -> set[str]:
        return {
            "identity_profile", "affiliation", "relationship", "judgment", "preference",
            "constraint", "behavior_pattern",
        }

    @staticmethod
    def _format_activity_contexts_for_signal_prompt(
        signal_items: Sequence[Dict[str, Any]],
    ) -> str:
        """Share only this LLM batch's activities, with one context per line."""
        contexts = {
            int(activity["activity_id"]): {
                "activity_id": int(activity["activity_id"]),
                "activity_text": activity["activity_text"],
            }
            for item in signal_items for activity in item.get("activity_contexts") or []
        }
        if not contexts:
            return "[]"
        return "[\n" + ",\n".join(
            json.dumps(contexts[key], ensure_ascii=False, separators=(",", ":"))
            for key in sorted(contexts)
        ) + "\n]"

    def _update_user_observed_patterns_from_action_signals(
        self, signal_ids: Sequence[int],
        *, reference_timestamp: Any = None,
    ) -> Dict[str, Any]:
        """Cluster unassigned evidence, then maintain or discover patterns per cluster."""
        report: Dict[str, Any] = {
            "candidate_count": 0, "updated": 0, "created": 0,
            "completed": True,
            "similar_action_batches": 0, "relations_recorded": 0,
            "direct_support_signal_count": 0, "processed_signal_ids": [],
            "pattern_event_ids": [], "skipped_activity_gate_count": 0,
        }
        unassigned_signals = self._db.get_unassigned_user_action_signals()
        seed_signals = self._db.get_user_action_signals_by_ids(signal_ids)
        seed_ids = {int(row["signal_id"]) for row in seed_signals}
        report["processed_signal_ids"] = [int(row["signal_id"]) for row in seed_signals]
        if not seed_signals:
            return report
        subject_id = int(seed_signals[0]["subject_entity_id"])
        # Assigned signals stay available as representative pattern evidence,
        # but never re-enter the historical discovery pool.
        history = [row for row in unassigned_signals if int(row["signal_id"]) not in seed_ids]
        existing_patterns = self._db.get_user_observed_patterns(subject_entity_id=subject_id, limit=None)
        pattern_evidence = self._db.get_user_action_signals_by_ids(sorted({
            int(value) for pattern in existing_patterns for value in pattern["support_signal_ids"]
        }))
        pool = {int(row["signal_id"]): row for row in [*history, *seed_signals, *pattern_evidence]}
        vectors = self._user_action_vectors_for_signals(list(pool.values()))
        clusters = self._cluster_user_action_signals(
            seeds=seed_signals, history=history, vectors=vectors,
        )
        report["similar_action_batches"] = len(clusters)
        supported_signal_ids: set[int] = set()
        discovery_clusters: List[Dict[str, Any]] = []
        for cluster_id, (cluster, new_signal_ids) in enumerate(clusters):
            contexts = self._user_action_signal_activity_contexts(cluster)
            related = self._retrieve_similar_user_observed_patterns(
                signals=cluster, signals_by_id=pool,
                vectors=vectors,
            )
            cluster_matched = False
            if related:
                direct_update = self._update_observed_patterns_from_new_action_signals(
                    new_signals=cluster, existing_patterns=related,
                    reference_timestamp=reference_timestamp,
                )
                if direct_update is None:
                    report.update(completed=False, error="invalid_action_pattern_relation_response")
                    return report
                supported_signal_ids.update(direct_update["supported_signal_ids"])
                report["updated"] += len(direct_update["updated_pattern_ids"])
                report["relations_recorded"] += direct_update["relations_recorded"]
                report["pattern_event_ids"].extend(direct_update.get("pattern_event_ids") or [])
                cluster_matched = bool(direct_update["supported_signal_ids"])
            activity_count = self._user_action_cluster_activity_count(cluster, contexts)
            self._logger.log_json_format("memory_user_observed_pattern_update", "cluster_evaluated", {
                "cluster_id": cluster_id,
                "signal_ids": [int(row["signal_id"]) for row in cluster],
                "new_signal_ids": sorted(new_signal_ids),
                "existing_pattern_ids": [int(pattern["id"]) for pattern in related],
                "matched_existing_pattern": cluster_matched,
                "activity_count": activity_count,
            })
            # The cluster is the decision unit: do not split unused fragments
            # out of a matched cluster to discover another pattern in this run.
            if cluster_matched:
                continue
            if activity_count < self._user_observed_pattern_min_activities:
                report["skipped_activity_gate_count"] += 1
                continue
            discovery_clusters.append({
                "cluster_id": cluster_id, "signals": cluster,
                "new_signal_ids": new_signal_ids,
            })

        # First finish matching every cluster. Only then jointly extract new
        # patterns, keeping each unmatched cluster as an isolated input case.
        report["candidate_count"] = len(discovery_clusters)
        if discovery_clusters:
            cluster_results = self._extract_user_observed_patterns(
                clusters=discovery_clusters,
            )
            if cluster_results is None:
                report.update(completed=False, error="invalid_observed_pattern_response")
                return report
            existing_by_id = {int(pattern["id"]): pattern
                for pattern in self._db.get_user_observed_patterns(subject_entity_id=subject_id, limit=None)}
            for cluster_result in cluster_results:
                for pattern in cluster_result["patterns"]:
                    applied = self._reconcile_user_observed_pattern(
                        pattern, subject_entity_id=subject_id, existing_patterns=list(existing_by_id.values()),
                        reference_timestamp=reference_timestamp,
                    )
                    if applied is None:
                        report.update(completed=False, error="invalid_pattern_reconciliation")
                        return report
                    report["created"] += int(applied["created"])
                    report["updated"] += len(applied["updated_pattern_ids"])
                    report["relations_recorded"] += int(applied["relations_recorded"])
                    report["pattern_event_ids"].extend(applied["pattern_event_ids"])
                    changed_ids = set(applied["updated_pattern_ids"])
                    if applied["pattern_id"] is not None:
                        changed_ids.add(int(applied["pattern_id"]))
                    existing_by_id.update({int(row["id"]): row for row in
                        self._db.get_user_observed_patterns_by_ids(sorted(changed_ids))})
        report["direct_support_signal_count"] = len(supported_signal_ids)
        report["pattern_event_ids"] = sorted(set(report["pattern_event_ids"]))
        return report

    def _update_observed_patterns_from_new_action_signals(
        self, *, new_signals: Sequence[Dict[str, Any]],
        existing_patterns: Sequence[Dict[str, Any]],
        reference_timestamp: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """Classify a cluster's unassigned evidence against existing patterns."""
        result: Dict[str, Any] = {
            "supported_signal_ids": set(), "updated_pattern_ids": set(),
            "relations_recorded": 0, "pattern_event_ids": [],
        }
        if not new_signals or not existing_patterns:
            return result
        facts_by_id = {int(fact["id"]): fact for fact in self._db.get_memory_facts_by_ids(
            [self._user_action_signal_fact_id(signal) for signal in new_signals
             if signal["source_type"] == "fact"],
        )}
        def representative_ids(pattern: Dict[str, Any]) -> List[int]:
            ids = pattern.get("support_signal_ids") or []
            return ids if len(ids) <= 4 else [ids[0], *ids[-3:]]

        evidence_ids = [
            signal_id
            for pattern in existing_patterns
            for signal_id in representative_ids(pattern)
        ]
        evidence_signals = self._db.get_user_action_signals_by_ids(evidence_ids)
        evidence_facts = {
            int(fact["id"]): fact
            for fact in self._db.get_memory_facts_by_ids([
                self._user_action_signal_fact_id(signal) for signal in evidence_signals
                if signal["source_type"] == "fact"
            ])
        }
        evidence_by_id = {
            int(signal["signal_id"]): signal for signal in evidence_signals
        }

        activities_by_id = self._user_action_signal_activities([*new_signals, *evidence_signals])

        new_by_id = {int(signal["signal_id"]): signal for signal in new_signals}
        pattern_by_id = {int(pattern["id"]): pattern for pattern in existing_patterns}
        language = self._resolve_prompt_language_from_text(
            "\n".join(str(signal.get("action_text") or "") for signal in new_signals)
        )
        template = (
            USER_OBSERVED_PATTERN_SIGNAL_CLUSTER_RELATION_PROMPT_EN
            if language == "en" else USER_OBSERVED_PATTERN_SIGNAL_CLUSTER_RELATION_PROMPT_ZH
        )
        prompt = template.replace("{new_signals}", "[\n" + ",\n".join(
            json.dumps(
                self._format_action_signal_view_for_prompt(
                    signal, facts_by_id.get(self._user_action_signal_fact_id(signal)) or {},
                    activities_by_id.get(int(signal["source_id"])),
                ), ensure_ascii=False, separators=(",", ":"),
            )
            for signal in new_signals
        ) + "\n]").replace("{existing_patterns}", "[\n" + ",\n".join(
            json.dumps(
                self._format_user_observed_pattern_view_for_prompt(
                    pattern, supporting_actions=[
                        self._format_action_signal_view_for_prompt(
                            evidence_by_id[signal_id], evidence_facts.get(
                                self._user_action_signal_fact_id(evidence_by_id[signal_id])
                            ) or {},
                            activities_by_id.get(int(evidence_by_id[signal_id]["source_id"])),
                        )
                        for signal_id in representative_ids(pattern)
                        if signal_id in evidence_by_id
                    ],
                ), ensure_ascii=False, separators=(",", ":"),
            )
            for pattern in existing_patterns
        ) + "\n]")
        raw = self._llm_client.call_llm(
            prompt, stage_name='memory_user_observed_pattern_signal_relation',
        )
        parsed = self._llm_client.parse_json_object_from_llm_text(raw or "", stage_name='memory_user_observed_pattern_signal_relation')
        if not parsed or not isinstance(parsed.get("decisions"), list):
            return None
        decisions: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for item in parsed["decisions"]:
            if not isinstance(item, dict):
                continue
            try:
                signal_id = int(item.get("signal_id"))
                pattern_id = int(item.get("existing_pattern_id"))
            except (TypeError, ValueError):
                continue
            relation = str(item.get("semantic_relation") or "").strip().lower()
            if (
                signal_id not in new_by_id or pattern_id not in pattern_by_id
                or relation not in {"supports", "refines", "contradicts", "unrelated"}
            ):
                continue
            confidence = self._clamp_float(item.get("confidence"), 0.0, 1.0, 0.0)
            if confidence < 0.7 or relation == "unrelated":
                continue
            key = (signal_id, pattern_id)
            if key not in decisions or confidence > decisions[key]["confidence"]:
                decisions[key] = {
                    "semantic_relation": relation,
                    "confidence": confidence,
                    "reason": _compact_whitespace(item.get("reason") or "")[:240],
                }
        with self._db.transaction():
            for (signal_id, pattern_id), decision in decisions.items():
                signal = new_by_id[signal_id]
                snapshot = self._format_action_signal_view_for_prompt(
                    signal, facts_by_id.get(self._user_action_signal_fact_id(signal)) or {},
                    activities_by_id.get(int(signal["source_id"])),
                )
                if decision["semantic_relation"] == "supports":
                    added = self._db.add_user_observed_pattern_signals(
                        pattern_id=pattern_id, signal_ids=[signal_id],
                        reference_timestamp=reference_timestamp,
                    )
                    if added:
                        self._db.refresh_user_observed_pattern(
                            pattern_id=pattern_id,
                            confidence=float(pattern_by_id[pattern_id]["confidence"]),
                            reference_timestamp=reference_timestamp,
                        )
                        result["updated_pattern_ids"].add(pattern_id)
                    result["supported_signal_ids"].add(signal_id)
                    if not added:
                        continue
                event_id = self._record_user_observed_pattern_event(
                    target_pattern_id=pattern_id, trigger_pattern_id=None,
                    semantic_relation=decision["semantic_relation"],
                    support_signal_ids=[signal_id],
                    candidate_snapshot=snapshot,
                    semantic_confidence=decision["confidence"],
                    semantic_reason=decision["reason"],
                    reference_timestamp=reference_timestamp,
                )
                if event_id is not None:
                    result["relations_recorded"] += 1
                    result["pattern_event_ids"].append(event_id)
            if result["updated_pattern_ids"]:
                self._upsert_memory_user_observed_patterns_into_recall_document_table(
                    sorted(result["updated_pattern_ids"]), reference_timestamp=reference_timestamp,
                )
        return result

    def _user_action_vectors_for_signals(
        self, signals: Sequence[Dict[str, Any]],
    ) -> Dict[int, np.ndarray]:
        """Read cached action vectors and embed only missing action texts."""
        self._ensure_embedding_client()
        assert self._embedding_client is not None
        dimension = self._embedding_client.dimension
        vectors: Dict[int, np.ndarray] = {}
        missing: List[Dict[str, Any]] = []
        for signal in signals:
            signal_id = int(signal["signal_id"])
            blob = signal.get("action_text_embedding")
            vector = np.frombuffer(blob, dtype=np.float32) if blob else None
            if vector is not None and vector.size == dimension and np.all(np.isfinite(vector)):
                vectors[signal_id] = vector
            else:
                missing.append(signal)
        if missing:
            embedded = self._embedding_client.embed_batch(
                str(signal.get("action_text") or "") for signal in missing
            )
            updates: List[Tuple[int, np.ndarray]] = []
            for signal, vector in zip(missing, embedded):
                if vector is None:
                    continue
                array = np.asarray(vector, dtype=np.float32)
                if array.size != dimension or not np.all(np.isfinite(array)):
                    continue
                signal_id = int(signal["signal_id"])
                vectors[signal_id] = array
                updates.append((signal_id, array))
            if updates:
                self._db.set_user_action_signal_embeddings(updates)
        return vectors

    @staticmethod
    def _user_action_signal_similarity(
        left: Dict[str, Any], right: Dict[str, Any],
        vectors: Dict[int, np.ndarray],
    ) -> float:
        lexical = action_lexical_similarity(
            left.get("action_text"), right.get("action_text"),
        )
        left_vector = vectors.get(int(left["signal_id"]))
        right_vector = vectors.get(int(right["signal_id"]))
        semantic = (
            max(0.0, _cal_embedding_cosine_similarity(left_vector, right_vector))
            if left_vector is not None and right_vector is not None else 0.0
        )
        return max(lexical, semantic * 0.82)

    def _retrieve_similar_user_observed_patterns(
        self, *, signals: Sequence[Dict[str, Any]],
        signals_by_id: Dict[int, Dict[str, Any]],
        vectors: Dict[int, np.ndarray],
    ) -> List[Dict[str, Any]]:
        """Retrieve patterns for the whole affected cluster, not only its new seed."""
        if not signals:
            return []
        patterns = self._db.get_user_observed_patterns(
            subject_entity_id=int(signals[0]["subject_entity_id"]), limit=None,
        )
        ranked: List[Tuple[float, Dict[str, Any]]] = []
        for pattern in patterns:
            score = max((action_lexical_similarity(
                signal.get("action_text"), pattern.get("pattern_text"),
            ) for signal in signals), default=0.0)
            for support_id in pattern.get("support_signal_ids") or []:
                support = signals_by_id.get(int(support_id))
                if support is not None:
                    score = max(score, *(self._user_action_signal_similarity(
                        signal, support, vectors,
                    ) for signal in signals))
            if score >= 0.60:
                ranked.append((score, pattern))
        ranked.sort(key=lambda item: (-item[0], int(item[1]["id"])))
        return [pattern for _, pattern in ranked[:8]]

    def _cluster_user_action_signals(
        self, *, seeds: Sequence[Dict[str, Any]],
        history: Sequence[Dict[str, Any]], vectors: Dict[int, np.ndarray],
    ) -> List[Tuple[List[Dict[str, Any]], set[int]]]:
        """Greedy complete-link clustering; only return clusters touched by new input.

        Cluster membership never proves independent observations. Requiring
        pairwise similarity avoids joining unrelated endpoints through a chain.
        Clusters do not overlap; downstream decisions use each complete cluster.
        """
        if not seeds:
            return []
        seed_ids = {int(row["signal_id"]) for row in seeds}
        pool = {int(row["signal_id"]): row for row in [*history, *seeds]}
        clusters: List[List[Dict[str, Any]]] = []
        for signal_id in sorted(pool):
            signal = pool[signal_id]
            scores = [min(self._user_action_signal_similarity(signal, member, vectors)
                          for member in cluster) for cluster in clusters]
            if scores and max(scores) >= 0.60:
                clusters[scores.index(max(scores))].append(signal)
            else:
                clusters.append([signal])
        return [(cluster, seed_ids & {int(row["signal_id"]) for row in cluster})
                for cluster in clusters if any(int(row["signal_id"]) in seed_ids for row in cluster)]

    @staticmethod
    def _user_action_cluster_activity_count(
        signals: Sequence[Dict[str, Any]], contexts: Dict[int, List[Dict[str, Any]]],
    ) -> int:
        """Count activity windows by ID, excluding ambiguous signal associations."""
        activity_ids: Set[int] = set()
        for signal in signals:
            associated_ids = {
                int(activity["id"])
                for activity in contexts.get(int(signal["signal_id"]), [])
            }
            if len(associated_ids) == 1:
                activity_ids.update(associated_ids)
        return len(activity_ids)

    def _fact_user_claim_signal_mapping_updates(
        self,
        fact_id: int,
        fact: Dict[str, Any],
        *, reference_timestamp: Any = None,
    ) -> List[Dict[str, Any]]:
        """Persist every normalized user-claim signal for one fact."""
        updates: List[Dict[str, Any]] = []
        raw_signals = self._normalize_user_claim_signal(
            fact.get("user_claim_signals"),
            allows_world_owner_reference=(
                isinstance(fact.get("metadata"), dict)
                and bool(fact["metadata"].get("has_world_owner_speaker")
                         or fact["metadata"].get("has_world_owner_alias_reference"))
            ),
            owner_aliases=fact.get("resolved_world_owner_aliases") or [],
        )
        for signal in raw_signals:
            claim_type_hint = str(signal.get("claim_type_hint") or "").lower()
            subject_entity = self._subject_entity_for_user_claim_signal(signal, fact)
            if not subject_entity:
                continue
            entity_mapping = self._db.upsert_entity_nodes([{
                "name": subject_entity,
                "type": "PERSON",
            }], reference_timestamp=reference_timestamp)
            if subject_entity not in entity_mapping:
                continue
            claim_anchor = _compact_whitespace(signal.get("claim_anchor") or "")
            if not claim_anchor:
                continue
            claim_anchor_key = self._generate_topic_name_key(claim_anchor)
            if not claim_anchor_key:
                continue
            evidence_basis = _compact_whitespace(
                signal.get("evidence_basis") or ""
            )
            signal_fingerprint = hashlib.sha256(
                "\x1f".join((
                    claim_type_hint,
                    claim_anchor_key,
                    evidence_basis.lower(),
                )).encode("utf-8")
            ).hexdigest()[:16]
            updates.append({
                "source_type": "fact", "source_id": int(fact_id),
                "signal_key": (
                    f"{int(entity_mapping[subject_entity])}|{claim_type_hint}|"
                    f"{claim_anchor_key}|{signal_fingerprint}"
                ),
                "subject": subject_entity,
                "subject_entity_id": int(entity_mapping[subject_entity]),
                "claim_type_hint": claim_type_hint,
                "claim_anchor": claim_anchor,
                "claim_anchor_key": claim_anchor_key,
                "evidence_basis": evidence_basis,
                "confidence": float(signal.get("confidence") or 0.0),
            })
        return updates

    def _fact_user_action_signal_updates(
        self,
        *,
        fact_id: int,
        fact: Dict[str, Any],
        owner_entity_id: int,
    ) -> List[Dict[str, Any]]:
        """Prepare persisted action evidence without routing it to claims."""
        if self._world_owner_entity_name not in (fact.get("entities") or []):
            return []
        updates: List[Dict[str, Any]] = []
        for signal in fact.get("user_action_signals") or []:
            if not isinstance(signal, dict):
                continue
            action_kind = str(signal.get("action_kind") or "")
            action_text = _compact_whitespace(signal.get("action_text") or "")
            owner_evidence_type = str(signal.get("owner_evidence_type") or "")
            if (
                signal.get("subject_entity") != self._world_owner_entity_name
                or action_kind not in {"action", "choice", "participation"}
                or not action_text
                or owner_evidence_type not in {
                    "self_statement", "confirmed_response", "third_party_report",
                }
            ):
                continue
            fingerprint = hashlib.sha256(
                "\x1f".join((
                    action_kind,
                    action_text.lower(),
                    owner_evidence_type,
                    str(signal.get("owner_evidence_text") or "").lower(),
                    str(signal.get("other_evidence_text") or "").lower(),
                )).encode("utf-8")
            ).hexdigest()[:16]
            updates.append({
                "source_type": "fact", "source_id": int(fact_id),
                "signal_key": f"{owner_entity_id}|{action_kind}|{fingerprint}",
                "subject_entity_id": int(owner_entity_id),
                "action_kind": action_kind,
                "action_text": action_text,
                "event_time_key": str(fact.get("event_time_key") or ""),
                "time_confidence": str(
                    (fact.get("metadata") or {}).get("time_confidence") or "unknown"
                ),
                "location": str((fact.get("metadata") or {}).get("where") or ""),
                "owner_evidence_type": owner_evidence_type,
                "owner_evidence_text": signal.get("owner_evidence_text") or "",
                "other_evidence_text": signal.get("other_evidence_text") or "",
                "evidence_basis": signal.get("evidence_basis") or "",
                "confidence": float(signal.get("confidence") or 0.0),
            })
        return updates

    def _fact_user_intent_signal_mapping_updates(
        self,
        fact_id: int,
        fact: Dict[str, Any],
        *, reference_timestamp: Any = None,
    ) -> List[Dict[str, Any]]:
        """Project one fact's normalized prospective signals into DB rows."""
        updates: List[Dict[str, Any]] = []
        raw_signals = self._normalize_user_intent_signals(
            fact.get("user_intent_signals"),
            allows_world_owner_reference=(
                isinstance(fact.get("metadata"), dict)
                and bool(fact["metadata"].get("has_world_owner_speaker")
                         or fact["metadata"].get("has_world_owner_alias_reference"))
            ),
            owner_aliases=fact.get("resolved_world_owner_aliases") or [],
        )
        fact_entities_by_name = {
            str(descriptor["name"]).lower(): descriptor
            for descriptor in self._fact_entity_descriptors(fact)
            if _compact_whitespace(descriptor.get("name") or "")
        }
        for signal in raw_signals:
            subject_name = _compact_whitespace(signal.get("subject_entity") or "")
            entity_descriptor = fact_entities_by_name.get(subject_name.lower())
            if entity_descriptor is None:
                continue
            subject_name = str(entity_descriptor["name"])
            entity_mapping = self._db.upsert_entity_nodes([entity_descriptor], reference_timestamp=reference_timestamp)
            if subject_name not in entity_mapping:
                continue
            user_intent_anchor = _compact_whitespace(
                signal.get("user_intent_anchor") or ""
            )
            user_intent_anchor_key = self._generate_topic_name_key(user_intent_anchor)
            if not user_intent_anchor or user_intent_anchor_key == "general":
                continue
            updates.append({
                "source_type": "fact", "source_id": int(fact_id),
                "signal_key": json_dumps([
                    int(entity_mapping[subject_name]), signal["signal_purpose"],
                    user_intent_anchor_key, signal["user_role"], signal["evidence_basis"],
                ]),
                "subject_entity_id": int(entity_mapping[subject_name]),
                "signal_purpose": signal["signal_purpose"],
                "user_role": signal["user_role"],
                "user_intent_anchor": user_intent_anchor,
                "user_intent_anchor_key": user_intent_anchor_key,
                "assertion_source": signal["assertion_source"],
                "evidence_basis": signal["evidence_basis"],
                "confidence": signal["confidence"],
            })
        return updates

    @staticmethod
    def _user_action_signal_fact_id(signal: Dict[str, Any]) -> int:
        if signal["source_type"] not in {"fact", "user_activity"}:
            raise ValueError(f"Unsupported action signal source: {signal['source_type']}")
        return int(signal["source_id"]) if signal["source_type"] == "fact" else 0

    def _user_action_signal_activities(self, signals: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        return {int(item["id"]): item for item in self._db.get_user_activity_items_by_ids(
            sorted({int(signal["source_id"]) for signal in signals if signal["source_type"] == "user_activity"}),
        )}

    def _user_action_signal_activity_contexts(
        self, signals: Sequence[Dict[str, Any]],
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Resolve direct activity sources and fact-linked contexts without inventing events."""
        fact_contexts = self._db.get_user_activity_contexts_for_facts([
            int(row["source_id"]) for row in signals if row["source_type"] == "fact"
        ])
        ids_by_signal = {
            int(row["signal_id"]): (
                [int(row["source_id"])] if row["source_type"] == "user_activity"
                else [int(item["activity_id"]) for item in fact_contexts.get(int(row["source_id"]), [])]
            ) for row in signals
        }
        activities = self._db.get_user_activity_items_by_ids(sorted({
            value for ids in ids_by_signal.values() for value in ids
        }))
        by_id = {int(row["id"]): {key: row.get(key) for key in (
            "id", "activity_text", "user_role", "started_at", "ended_at",
        )} for row in activities}
        return {signal_id: [by_id[value] for value in sorted(set(ids)) if value in by_id]
                for signal_id, ids in ids_by_signal.items()}

    @staticmethod
    def _format_action_signal_view_for_prompt(
        signal: Dict[str, Any], fact: Dict[str, Any], activity: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Expose provenance so multiple records/windows cannot masquerade as events."""
        view = {
            "signal_id": int(signal["signal_id"]),
            "action_kind": signal.get("action_kind") or "",
            "action_text": signal.get("action_text") or "",
            "event_time": signal.get("event_time_key") or "",
            "event_ended_at": signal.get("event_ended_at") or "",
            "time_scope": signal.get("time_scope") or "reported_event",
            "location": signal.get("location") or "",
            "source_type": signal["source_type"], "source_id": int(signal["source_id"]),
            "owner_evidence_type": signal.get("owner_evidence_type") or "",
            "owner_evidence_text": signal.get("owner_evidence_text") or "",
            "other_evidence_text": signal.get("other_evidence_text") or "",
            "evidence_basis": signal.get("evidence_basis") or "",
        }
        if signal["source_type"] == "fact":
            view.update(fact_summary=fact.get("summary") or "",
                        dialogue_time=fact.get("dialogue_time_key") or "")
        elif signal["source_type"] == "user_activity":
            view["activity_text"] = (activity or {}).get("activity_text") or ""
        return view

    @staticmethod
    def _format_user_observed_pattern_view_for_prompt(
        pattern: Dict[str, Any], *, supporting_actions: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Format candidate and persisted patterns with their selected action evidence."""
        view = {
            "pattern_text": pattern.get("pattern_text") or "",
            "condition_text": pattern.get("condition_text") or "",
            "primary_domain": pattern.get("primary_domain") or "",
            "confidence": pattern.get("confidence"),
            "support_signal_ids": list(pattern.get("support_signal_ids") or []),
            "supporting_actions": list(supporting_actions),
        }
        if pattern.get("id") is not None:
            view["id"] = int(pattern["id"])
        return view

    def _extract_user_observed_patterns(
        self,
        *,
        clusters: Sequence[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        """Extract unmatched clusters together without mixing their evidence or results."""
        if not clusters:
            return []
        signals = [signal for cluster in clusters for signal in cluster["signals"]]
        language = self._resolve_prompt_language_from_text(
            "\n".join(str(signal.get("action_text") or "") for signal in signals)
        )
        prompt_template = (
            USER_OBSERVED_PATTERN_EXTRACTION_PROMPT_EN
            if language == "en" else USER_OBSERVED_PATTERN_EXTRACTION_PROMPT_ZH
        )
        activities_by_id = self._user_action_signal_activities(signals)
        activity_contexts = self._user_action_signal_activity_contexts(signals)
        facts_by_id = {int(fact["id"]): fact for fact in self._db.get_memory_facts_by_ids(
            sorted({int(signal["source_id"]) for signal in signals if signal["source_type"] == "fact"}),
        )}
        cluster_inputs = [{
            "cluster_id": int(cluster["cluster_id"]),
            "new_signal_ids": sorted(cluster["new_signal_ids"]),
            "action_signals": [self._format_action_signal_view_for_prompt(
                signal, facts_by_id.get(self._user_action_signal_fact_id(signal)) or {},
                activities_by_id.get(int(signal["source_id"])),
            )
                for signal in cluster["signals"]],
        } for cluster in clusters]
        prompt = prompt_template.replace(
            "{signal_clusters}", "[\n" + ",\n".join(
                json.dumps(case, ensure_ascii=False, separators=(",", ":")) for case in cluster_inputs
            ) + "\n]",
        )
        raw = self._llm_client.call_llm(
            prompt, stage_name='memory_user_observed_pattern_extraction',
        )
        parsed = self._llm_client.parse_json_object_from_llm_text(raw or "", stage_name='memory_user_observed_pattern_extraction')
        if parsed is None or not isinstance(parsed.get("cluster_results"), list):
            return None
        clusters_by_id = {int(cluster["cluster_id"]): cluster for cluster in clusters}
        outputs_by_id: Dict[int, Dict[str, Any]] = {}
        for output in parsed["cluster_results"]:
            if not isinstance(output, dict) or not str(output.get("cluster_id")).isdigit():
                return None
            cluster_id = int(output["cluster_id"])
            if (cluster_id not in clusters_by_id or cluster_id in outputs_by_id
                    or not isinstance(output.get("patterns"), list)):
                return None
            outputs_by_id[cluster_id] = output
        if outputs_by_id.keys() != clusters_by_id.keys():
            return None
        result = []
        for cluster_id, cluster in clusters_by_id.items():
            patterns = []
            for raw_pattern in outputs_by_id[cluster_id]["patterns"]:
                pattern = self._normalize_user_observed_pattern(
                    raw_pattern,
                    new_signal_ids=set(cluster["new_signal_ids"]),
                    available_signal_ids={int(signal["signal_id"]) for signal in cluster["signals"]},
                    activity_contexts=activity_contexts,
                )
                if pattern:
                    patterns.append(pattern)
            result.append({"cluster_id": cluster_id, "patterns": patterns})
        return result

    def _normalize_user_observed_pattern(
        self,
        raw: Any,
        *,
        new_signal_ids: set[int],
        available_signal_ids: set[int],
        activity_contexts: Dict[int, List[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        primary_domain = self._normalize_primary_domain(raw.get("primary_domain"))
        if not primary_domain:
            return None
        if (
            _compact_whitespace(raw.get("subject_entity") or "")
            != self._world_owner_entity_name
        ):
            return None
        raw_ids = raw.get("support_signal_ids")
        if not isinstance(raw_ids, list) or any(
            not str(value).isdigit() or int(value) not in available_signal_ids for value in raw_ids
        ):
            return None
        signal_ids = list(dict.fromkeys(
            int(value) for value in (raw.get("support_signal_ids") or [])
            if str(value).strip().isdigit() and int(value) in available_signal_ids
        ))
        if (
            not new_signal_ids.intersection(signal_ids)
            or self._user_action_cluster_activity_count(
                [{"signal_id": value} for value in signal_ids], activity_contexts,
            ) < self._user_observed_pattern_min_activities
        ):
            return None
        pattern_text = _compact_whitespace(raw.get("pattern_text") or "")[:480]
        if not pattern_text:
            return None
        confidence = self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.7)
        return {
            "pattern_text": pattern_text,
            "condition_text": _compact_whitespace(raw.get("condition_text") or "")[:240],
            "primary_domain": primary_domain,
            "confidence": confidence,
            "support_signal_ids": signal_ids,
        }

    def _record_user_observed_pattern_event(self, reference_timestamp: Any = None, **event: Any) -> Optional[int]:
        """Capture a new event ID in the FIFO writer's transaction; duplicates return None."""
        with self._db.transaction():
            if not self._db.record_user_observed_pattern_event(**event, reference_timestamp=reference_timestamp):
                return None
            # The DB write API returns bool. Read the inserted ID immediately,
            # before another event can be committed, without scanning signal history.
            return self._db.get_user_observed_pattern_event_cursor()

    def _reconcile_user_observed_pattern(
        self, pattern: Dict[str, Any], *, subject_entity_id: int,
        existing_patterns: Optional[Sequence[Dict[str, Any]]] = None,
        reference_timestamp: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """Preserve semantic changes, then apply each candidate-to-pattern relation."""
        existing = list(existing_patterns) if existing_patterns is not None else self._db.get_user_observed_patterns(
            subject_entity_id=subject_entity_id, limit=200,
        )
        # Domain is a preference for the shortlist, not an exclusion: a pattern
        # can be assigned to a different domain in a later extraction.
        ranked = sorted(existing, key=lambda row: (
            int(row.get("primary_domain") == pattern["primary_domain"]),
            action_lexical_similarity(
                pattern["pattern_text"], row.get("pattern_text"),
            ),
        ), reverse=True)
        related = ranked[:16]
        relations = self._classify_user_observed_pattern_relations(pattern, related)
        if relations is None:
            return None
        existing_by_id = {int(row["id"]): row for row in related}
        relation_types = {item["semantic_relation"] for item in relations}
        save_candidate = bool(relation_types & {"refines", "contradicts", "unrelated"}) or not (
            relation_types & {"duplicate", "supports"}
        )
        candidate_id = None
        created = False
        updated_pattern_ids: Set[int] = set()
        pattern_event_ids: List[int] = []
        recorded_count = 0
        with self._db.transaction():
            if save_candidate:
                pattern_key = hashlib.sha256(json.dumps({
                    "pattern_text": pattern["pattern_text"].casefold(),
                    "condition_text": pattern["condition_text"].casefold(),
                    "primary_domain": pattern["primary_domain"],
                }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
                candidate_id, created = self._db.upsert_user_observed_pattern(
                    subject_entity_id=subject_entity_id, pattern_key=pattern_key,
                    pattern_text=pattern["pattern_text"],
                    condition_text=pattern["condition_text"],
                    primary_domain=pattern["primary_domain"],
                    confidence=pattern["confidence"],
                    reference_timestamp=reference_timestamp,
                )
                added = self._db.add_user_observed_pattern_signals(
                    pattern_id=candidate_id, signal_ids=pattern["support_signal_ids"],
                    reference_timestamp=reference_timestamp,
                )
                if created:
                    event_id = self._record_user_observed_pattern_event(
                        event_type="created", target_pattern_id=candidate_id,
                        trigger_pattern_id=None, semantic_relation="",
                        support_signal_ids=pattern["support_signal_ids"],
                        candidate_snapshot={}, semantic_confidence=0.0,
                        semantic_reason="",
                        reference_timestamp=reference_timestamp,
                    )
                    if event_id is not None:
                        pattern_event_ids.append(event_id)
                elif added:
                    updated_pattern_ids.add(candidate_id)
                    event_id = self._record_user_observed_pattern_event(
                        target_pattern_id=candidate_id, trigger_pattern_id=None,
                        semantic_relation="supports",
                        support_signal_ids=pattern["support_signal_ids"],
                        candidate_snapshot=pattern, semantic_confidence=pattern["confidence"],
                        semantic_reason="New action signals supported the existing pattern key.",
                        reference_timestamp=reference_timestamp,
                    )
                    if event_id is not None:
                        recorded_count += 1
                        pattern_event_ids.append(event_id)
            for relation in relations:
                relation_type = relation["semantic_relation"]
                target_id = relation["existing_pattern_id"]
                if relation_type in {"duplicate", "supports"}:
                    added = self._db.add_user_observed_pattern_signals(
                        pattern_id=target_id, signal_ids=pattern["support_signal_ids"],
                        reference_timestamp=reference_timestamp,
                    )
                    if added:
                        updated_pattern_ids.add(target_id)
                        self._db.refresh_user_observed_pattern(
                            pattern_id=target_id,
                            confidence=float(existing_by_id[target_id]["confidence"]),
                            reference_timestamp=reference_timestamp,
                        )
                # The source is C, never one of its merge targets. An unpersisted
                # C is represented by its candidate snapshot without inventing an ID.
                event_id = self._record_user_observed_pattern_event(
                    target_pattern_id=target_id,
                    trigger_pattern_id=candidate_id if candidate_id != target_id else None,
                    semantic_relation=relation_type,
                    support_signal_ids=pattern["support_signal_ids"],
                    candidate_snapshot=pattern,
                    semantic_confidence=relation["confidence"],
                    semantic_reason=relation["reason"],
                    reference_timestamp=reference_timestamp,
                )
                if event_id is not None:
                    recorded_count += 1
                    pattern_event_ids.append(event_id)
            recall_pattern_ids = updated_pattern_ids | ({candidate_id} if candidate_id is not None else set())
            if recall_pattern_ids:
                self._upsert_memory_user_observed_patterns_into_recall_document_table(
                    sorted(recall_pattern_ids), reference_timestamp=reference_timestamp,
                )
        return {"pattern_id": candidate_id, "created": created,
                "updated_pattern_ids": sorted(updated_pattern_ids),
                "pattern_event_ids": pattern_event_ids,
                "relations_recorded": recorded_count}

    def _classify_user_observed_pattern_relations(
        self, candidate: Dict[str, Any], existing: Sequence[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        if not existing:
            return []
        evidence_ids = list(candidate["support_signal_ids"][-8:])
        evidence_ids.extend(
            signal_id
            for row in existing
            for signal_id in (row.get("support_signal_ids") or [])[-4:]
        )
        signals_by_id = {
            int(row["signal_id"]): row
            for row in self._db.get_user_action_signals_by_ids(evidence_ids)
        }
        facts_by_id = {
            int(row["id"]): row
            for row in self._db.get_memory_facts_by_ids([
                self._user_action_signal_fact_id(signal) for signal in signals_by_id.values()
                if signal["source_type"] == "fact"
            ])
        }

        activities_by_id = self._user_action_signal_activities(list(signals_by_id.values()))

        def evidence_view(signal_ids: Sequence[int], limit: int) -> List[Dict[str, Any]]:
            result: List[Dict[str, Any]] = []
            for signal_id in signal_ids[-limit:]:
                signal = signals_by_id.get(int(signal_id))
                if signal is None:
                    continue
                result.append(self._format_action_signal_view_for_prompt(
                    signal, facts_by_id.get(self._user_action_signal_fact_id(signal)) or {},
                    activities_by_id.get(int(signal["source_id"])),
                ))
            return result

        language = self._resolve_prompt_language_from_text(
            candidate["pattern_text"],
        )
        template = (
            USER_OBSERVED_PATTERN_RECONCILIATION_PROMPT_EN
            if language == "en" else USER_OBSERVED_PATTERN_RECONCILIATION_PROMPT_ZH
        )
        prompt = template.replace("{candidate_pattern}", json.dumps(
            self._format_user_observed_pattern_view_for_prompt(
                candidate, supporting_actions=evidence_view(candidate["support_signal_ids"], 8),
            ), ensure_ascii=False, separators=(",", ":"),
        )).replace("{existing_patterns}", "[\n" + ",\n".join(
            json.dumps(
                self._format_user_observed_pattern_view_for_prompt(
                    row, supporting_actions=evidence_view(
                        row.get("support_signal_ids") or [], 4,
                    ),
                ), ensure_ascii=False, separators=(",", ":"),
            )
            for row in existing
        ) + "\n]")
        raw = self._llm_client.call_llm(
            prompt, stage_name='memory_user_observed_pattern_reconciliation',
        )
        parsed = self._llm_client.parse_json_object_from_llm_text(raw or "", stage_name='memory_user_observed_pattern_reconciliation')
        if not parsed or not isinstance(parsed.get("relations"), list):
            return None
        allowed_ids = {int(row["id"]) for row in existing}
        allowed_relations = {"duplicate", "supports", "refines", "contradicts", "unrelated"}
        result: List[Dict[str, Any]] = []
        for item in parsed["relations"]:
            if not isinstance(item, dict):
                continue
            try:
                target_id = int(item.get("existing_pattern_id"))
            except (TypeError, ValueError):
                continue
            relation = str(item.get("semantic_relation") or "").strip().lower()
            if target_id not in allowed_ids or relation not in allowed_relations:
                continue
            result.append({
                "existing_pattern_id": target_id,
                "semantic_relation": relation,
                "confidence": self._clamp_float(item.get("confidence"), 0.0, 1.0, 0.7),
                "reason": _compact_whitespace(item.get("reason") or "")[:240],
            })
        return result

    def _log_user_intent_signals_loaded(
        self,
        signal_items: Sequence[Dict[str, Any]],
        reference_timestamp: Any,
    ) -> None:
        """Log signal-bearing facts once, keeping selected signal IDs visible."""
        fact_views = json.loads(self._format_fact_cases_for_user_intent_prompt(signal_items))
        self._logger.log_json_format("memory_user_intent_update", "signals_loaded", {
            "signal_count": len(signal_items),
            "fact_count": len(fact_views),
            "signal_ids": [item["signal_id"] for item in signal_items],
            "reference_timestamp": reference_timestamp,
            "fact_cases": [CompactLogValue(view) for view in fact_views],
        })

    @staticmethod
    def _generate_topic_name_key(value: Any) -> str:
        text = _compact_whitespace(value).lower()
        text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", text)
        return text or "general"

    @staticmethod
    def _normalize_primary_domain(value: Any) -> str:
        """Return a controlled semantic domain or an empty invalid marker."""
        domain = re.sub(
            r"[^a-z]+", "_", str(value or "").strip().lower(),
        ).strip("_")
        return domain if domain in SEMANTIC_DOMAINS else ""

    @staticmethod
    def _normalize_unique_labels(
        values: Sequence[Any],
        *,
        limit: int = 20,
    ) -> List[str]:
        labels: List[str] = []
        seen: set[str] = set()
        for value in values:
            label = _compact_whitespace(value)
            if not label:
                continue
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            labels.append(label)
            if len(labels) >= limit:
                break
        return labels

    def _topic_name_best_pair_similarity(
        self,
        left_aliases: Sequence[str],
        right_aliases: Sequence[str],
        *,
        allow_substring: bool = True,
    ) -> float:
        """Return the strongest similarity between any two topic aliases."""
        best = 0.0
        for left in left_aliases:
            left_key = self._generate_topic_name_key(left)
            for right in right_aliases:
                right_key = self._generate_topic_name_key(right)
                if left_key and right_key and left_key == right_key:
                    return 1.0
                if (
                    allow_substring
                    and left_key
                    and right_key
                    and (left_key in right_key or right_key in left_key)
                ):
                    best = max(best, 0.9)
                left_terms = set(self._topic_similarity_terms(str(left)))
                right_terms = set(self._topic_similarity_terms(str(right)))
                if not left_terms or not right_terms:
                    continue
                shared_terms = left_terms & right_terms
                if not shared_terms:
                    continue
                jaccard = len(shared_terms) / max(1, len(left_terms | right_terms))
                best = max(best, jaccard)

                # A shorter topic can be a lexical specialization of a
                # longer topic, e.g. "手机推广策略" and
                # "新手机产品推广策略". Require at least two shared
                # tokens so a single generic word cannot create a strong
                # match by itself.
                if len(shared_terms) >= 2:
                    left_coverage = len(shared_terms) / max(1, len(left_terms))
                    right_coverage = len(shared_terms) / max(1, len(right_terms))
                    best = max(best, left_coverage, right_coverage)
        return best

    def _subject_entity_for_user_claim_signal(
        self,
        signal: Dict[str, Any],
        fact: Dict[str, Any],
    ) -> str:
        """Return the wearer subject only when it is a fact entity."""
        name = _compact_whitespace(signal.get("subject_entity") or "")
        if name.lower() != self._world_owner_entity_name.lower():
            return ""
        for fact_entity in fact.get("entities") or []:
            canonical_name = _compact_whitespace(fact_entity)
            if canonical_name and canonical_name.lower() == name.lower():
                return canonical_name
        return ""

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
    def _clamp_float(value: Any, low: float, high: float, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = default
        return max(low, min(high, number))

    @staticmethod
    def _format_log_text(value: Any, *, limit: int = 500) -> str:
        text = _compact_whitespace(value)
        if limit <= 0 or len(text) <= limit:
            return text
        return text[:limit] + "...[truncated]"
    
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
            if self._is_valid_entity_name(clean) and clean not in entities:
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
    @staticmethod
    def _normalize_fact_type(value: Any) -> str:
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
            if not MemoryNodeManager._is_valid_entity_name(text) or text in seen:
                continue
            seen.add(text)
            out.append(text)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _normalize_entity_type(value: Any) -> str:
        entity_type = _compact_whitespace(value or "OTHER").upper()
        allowed_types = {
            "PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "PROJECT",
            "TECHNOLOGY", "CONCEPT", "TOPIC", "PREFERENCE", "OTHER",
        }
        return entity_type if entity_type in allowed_types else "OTHER"

    @classmethod
    def _normalize_entity_descriptors(
        cls,
        value: Any,
        *,
        limit: int = 16,
    ) -> List[Dict[str, str]]:
        """Normalize prompt entity objects without discarding their types."""
        if isinstance(value, str):
            raw = re.split(r"[,，;；\n]+", value)
        elif isinstance(value, list):
            raw = value
        else:
            raw = []
        descriptors: List[Dict[str, str]] = []
        for item in raw:
            if isinstance(item, dict):
                candidate = {
                    "name": _compact_whitespace(
                        item.get("name") or item.get("text") or "",
                    ),
                    "type": cls._normalize_entity_type(item.get("type")),
                }
            else:
                candidate = {
                    "name": _compact_whitespace(item),
                    "type": "OTHER",
                }
            if not cls._is_valid_entity_name(candidate["name"]):
                continue
            cls._merge_entity_descriptor(descriptors, candidate)
            if len(descriptors) >= limit:
                break
        return descriptors

    @staticmethod
    def _is_valid_entity_name(value: Any) -> bool:
        """Validate entity anchors using the shared extraction guidance."""
        text = _compact_whitespace(value).strip("'\".,:;!?，。！？、；：（）()[]{}")
        if not text:
            return False
        lower = text.lower()
        if lower in _STOPWORDS:
            return False
        if any(pattern in lower for pattern in _COURTESY_PATTERNS):
            return False
        if re.search(r"[。！？!?；;，,]", text):
            return False
        if len(text) > 48:
            return False
        if any(
            re.fullmatch(pattern, text, flags=re.IGNORECASE)
            for pattern in _ORDINARY_TIME_ENTITY_PATTERNS
        ):
            return False
        if any(
            re.fullmatch(pattern, text, flags=re.IGNORECASE)
            for pattern in _ATTRIBUTE_ONLY_ENTITY_PATTERNS
        ):
            if not re.search(
                r"(工作|压力|负担|时间|作息|活动|场景|问题|任务|状态|沟通|管理|"
                r"fatigue|burden|pressure|schedule|activity|scenario|task|condition|communication|management)",
                lower,
            ):
                return False
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        if chinese_chars and len(chinese_chars) > 16:
            return False
        if not chinese_chars and len(text.split()) > 5:
            return False
        return len(text) > 1
