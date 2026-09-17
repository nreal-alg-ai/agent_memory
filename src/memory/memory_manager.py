#!/usr/bin/env python3
"""Unified memory manager inspired by MemPalace.

The public surface mirrors the current project's `MemoryNodeManager`, but the
internal model is deliberately unified:

1. assistant_wakeup turns and future allday transcript episodes both become
   `memory_episodes`.
2. Extracted evidence becomes narrative `memory_facts`.
3. Traceable explicit and inductive propositions live in entity claims.
4. The legacy actionable-item projection is temporarily disabled while the
   intent and work-item layer is redesigned.

"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import re
import threading
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests

try:
    import jieba
except ImportError:  # pragma: no cover - exercised only in minimal installs
    jieba = None

from .embedding_client import EmbeddingClient
from .memory_database import SessionDB
from .prompts_en import (
    ENTITY_CLAIM_RECONCILIATION_PROMPT_EN,
    EXPLICIT_ENTITY_CLAIM_EXTRACTION_PROMPT_EN,
    INDUCTIVE_ENTITY_CLAIM_EXTRACTION_PROMPT_EN,
    EPISODE_SUMMARY_PROMPT_EN,
    MEMORY_RETRIEVED_FORMAT_PROMPT_EN,
    MEMORY_RETRIEVED_SECTION_SPECS_EN,
    RECALL_QUERY_ANALYSIS_PROMPT_EN,
    UNIFIED_MEMORY_EXTRACTION_PROMPT_EN,
)
from .prompts_zh import (
    ENTITY_CLAIM_RECONCILIATION_PROMPT_ZH,
    EXPLICIT_ENTITY_CLAIM_EXTRACTION_PROMPT_ZH,
    INDUCTIVE_ENTITY_CLAIM_EXTRACTION_PROMPT_ZH,
    EPISODE_SUMMARY_PROMPT_ZH,
    MEMORY_RETRIEVED_FORMAT_PROMPT_ZH,
    MEMORY_RETRIEVED_SECTION_SPECS_ZH,
    RECALL_QUERY_ANALYSIS_PROMPT_ZH,
    UNIFIED_MEMORY_EXTRACTION_PROMPT_ZH,
)
from .utils import _cal_embedding_cosine_similarity

RecallTimeBounds = Optional[Tuple[Optional[str], Optional[str]]]

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

# These labels identify conversational roles rather than concrete entities.
# They may still be useful as diagnostics, but a role match alone must not be
# treated as an entity strong anchor during recall.
_LOW_VALUE_ENTITY_ALIASES = {
    "user": {
        "user", "the user", "用户", "我", "我自己", "me", "myself",
    },
    "assistant": {
        "assistant", "the assistant", "助手", "ai", "人工智能", "bot",
        "机器人", "agent",
    },
    "system": {
        "system", "the system", "系统",
    },
    "speaker": {
        "speaker", "说话人", "unknown", "unknown_speaker",
        "speaker_1", "speaker_2",
    },
}

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
    return datetime.now().astimezone().isoformat()


def _to_timestamp_text(value: Any) -> str:
    if isinstance(value, datetime):
        # The LongMemEval scripts pass time filters as "YYYY-MM-DD HH:MM:SS".
        # Keep the same sortable format so SQLite string range filters work.
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value or "").strip()


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


class MemoryOperationReporter:
    """Collect asynchronous memory operation results for benchmark callers."""

    def __init__(self, *, recent_task_limit: int = 200) -> None:
        self._lock = threading.Lock()
        self._task_seq = 0
        self._recent_task_limit = max(1, int(recent_task_limit or 200))
        self._counts: Dict[str, Dict[str, Any]] = {}
        self._latest_reports: Dict[str, Dict[str, Any]] = {}
        self._recent_tasks: List[Dict[str, Any]] = []

    @staticmethod
    def _empty_counts() -> Dict[str, Any]:
        return {
            "submitted": 0,
            "completed": 0,
            "succeeded": 0,
            "failed": 0,
            "rejected": 0,
            "inflight": 0,
            "total_elapsed_ms": 0.0,
        }

    def next_task_id(self, operation_type: str) -> str:
        clean_type = str(operation_type or "memory_task").strip() or "memory_task"
        with self._lock:
            self._task_seq += 1
            return f"{clean_type}-{self._task_seq}"

    def on_task_submitted(
        self,
        *,
        operation_type: str,
        task_id: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        safe_payload = self._json_compatible(payload or {})
        with self._lock:
            counts = self._counts.setdefault(operation_type, self._empty_counts())
            counts["submitted"] += 1
            counts["inflight"] += 1
            self._append_recent_locked({
                "event": "submitted",
                "operation_type": operation_type,
                "task_id": task_id,
                "payload": safe_payload,
                "timestamp": _now_text(),
            })

    def on_task_rejected(
        self,
        *,
        operation_type: str,
        task_id: str,
        reason: str,
    ) -> None:
        report = {
            "accepted": False,
            "status": "rejected",
            "reason": reason,
            "task_id": task_id,
        }
        with self._lock:
            counts = self._counts.setdefault(operation_type, self._empty_counts())
            counts["rejected"] += 1
            self._latest_reports[operation_type] = dict(report)
            self._append_recent_locked({
                "event": "rejected",
                "operation_type": operation_type,
                "task_id": task_id,
                "reason": reason,
                "timestamp": _now_text(),
            })

    def on_task_finished(
        self,
        *,
        operation_type: str,
        task_id: str,
        started_at: float,
        result: Any = None,
        error: Optional[BaseException] = None,
    ) -> None:
        elapsed_ms = round((time.monotonic() - started_at) * 1000, 2)
        succeeded = error is None and self._operation_succeeded(operation_type, result)
        report = self._operation_result_report(
            operation_type=operation_type,
            task_id=task_id,
            result=result,
            error=error,
            succeeded=succeeded,
            elapsed_ms=elapsed_ms,
        )
        with self._lock:
            counts = self._counts.setdefault(operation_type, self._empty_counts())
            counts["completed"] += 1
            counts["inflight"] = max(0, int(counts.get("inflight") or 0) - 1)
            counts["total_elapsed_ms"] = round(
                float(counts.get("total_elapsed_ms") or 0.0) + elapsed_ms,
                2,
            )
            counts["succeeded" if succeeded else "failed"] += 1
            self._latest_reports[operation_type] = dict(report)
            recent_event = {
                "event": "finished",
                "operation_type": operation_type,
                "task_id": task_id,
                "status": report.get("status"),
                "succeeded": succeeded,
                "elapsed_ms": elapsed_ms,
                "timestamp": _now_text(),
            }
            if error is not None:
                recent_event["error"] = str(error)
                recent_event["error_type"] = type(error).__name__
            self._append_recent_locked(recent_event)

    def on_recall_finished(self, report: Dict[str, Any]) -> None:
        elapsed_ms = float(report.get("elapsed_ms") or 0.0)
        status = str(report.get("status") or "").strip().lower()
        succeeded = status not in {"error", "failed"}
        recall_report = {
            key: value
            for key, value in report.items()
            if key != "memory_context"
        }
        with self._lock:
            counts = self._counts.setdefault("recall", self._empty_counts())
            counts["submitted"] += 1
            counts["completed"] += 1
            counts["total_elapsed_ms"] = round(
                float(counts.get("total_elapsed_ms") or 0.0) + elapsed_ms,
                2,
            )
            counts["succeeded" if succeeded else "failed"] += 1
            self._latest_reports["recall"] = recall_report
            self._append_recent_locked({
                "event": "finished",
                "operation_type": "recall",
                "task_id": f"recall-{counts['completed']}",
                "status": recall_report.get("status"),
                "actual_recall_mode": recall_report.get("actual_recall_mode"),
                "elapsed_ms": elapsed_ms,
                "timestamp": _now_text(),
            })

    def operation_report(self, operation_type: str) -> Dict[str, Any]:
        with self._lock:
            counts = dict(
                self._counts.get(operation_type) or self._empty_counts()
            )
            latest = self._latest_reports.get(operation_type)
        if latest:
            counts["latest_report"] = dict(latest)
        return counts

    def latest_report(self, operation_type: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest_reports.get(operation_type) or {})

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "counts": {
                    key: dict(value)
                    for key, value in self._counts.items()
                },
                "latest_reports": {
                    key: dict(value)
                    for key, value in self._latest_reports.items()
                },
                "recent_tasks": [dict(item) for item in self._recent_tasks],
            }

    def _append_recent_locked(self, event: Dict[str, Any]) -> None:
        self._recent_tasks.append(event)
        if len(self._recent_tasks) > self._recent_task_limit:
            del self._recent_tasks[: len(self._recent_tasks) - self._recent_task_limit]

    @classmethod
    def _json_compatible(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            return _to_timestamp_text(value)
        if isinstance(value, dict):
            return {
                str(key): cls._json_compatible(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._json_compatible(item) for item in value]
        return str(value)

    @staticmethod
    def _operation_succeeded(operation_type: str, result: Any) -> bool:
        if operation_type == "memory_store":
            if not isinstance(result, dict):
                return bool(result)
            return str(result.get("status") or "ok").strip().lower() not in {
                "failed",
                "skipped",
                "error",
                "queue_rejected",
            }
        if operation_type == "memory_reflect":
            if not isinstance(result, dict):
                return False
            return str(result.get("status") or "ok").strip().lower() not in {
                "failed",
                "error",
                "queue_rejected",
                "skipped",
            }
        return True

    @staticmethod
    def _operation_result_report(
        *,
        operation_type: str,
        task_id: str,
        result: Any,
        error: Optional[BaseException],
        succeeded: bool,
        elapsed_ms: float,
    ) -> Dict[str, Any]:
        if isinstance(result, dict):
            report = dict(result)
        else:
            report = {"result": result}
        result_status = str(report.get("status") or "").strip().lower()
        report.update({
            "accepted": True,
            "task_id": task_id,
            "operation_type": operation_type,
            "status": (
                "failed"
                if error is not None or not succeeded
                else (result_status or "ok")
            ),
            "total_elapsed_ms": elapsed_ms,
        })
        if operation_type in {"memory_store", "memory_episode_summary"}:
            report["stored"] = bool(succeeded and error is None)
        if error is not None:
            report["error_type"] = type(error).__name__
            report["error"] = str(error)
        return report


class MemoryNodeManager:
    """Compatibility manager backed by a unified index-first memory line."""

    def __init__(
        self,
        db: SessionDB,
        *,
        embedding_config: Optional[Dict[str, Any]] = None,
        memory_manager_config: Optional[Dict[str, Any]] = None,
        operation_reporter: Optional[MemoryOperationReporter] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._db = db
        self._logger = logger or logging.getLogger(__name__)
        self._operation_reporter = operation_reporter or MemoryOperationReporter()
        self._memory_cfg = dict(memory_manager_config or {})
        configured_embedding = self._memory_cfg.get("embedding")
        self._embedding_cfg = dict(
            embedding_config
            or (configured_embedding if isinstance(configured_embedding, dict) else {})
        )
        configured_llm = self._memory_cfg.get("llm")
        self._llm_cfg = dict(
            configured_llm if isinstance(configured_llm, dict) else {}
        )
        self._llm_model = str(self._llm_cfg.get("llm_name") or DEFAULT_LLM_MODEL)
        self._llm_base_url = self._normalize_llm_base_url(
            str(self._llm_cfg.get("llm_base_url") or DEFAULT_LLM_BASE_URL)
        )
        self._llm_api_key = self._resolve_env(self._llm_cfg.get("llm_api_key"))
        self._llm_timeout = int(self._llm_cfg.get("llm_timeout", 120) or 120)
        self._llm_json_mode = self._config_bool(
            self._llm_cfg.get("llm_json_mode", True),
            True,
        )
        self._llm_thinking = str(
            self._llm_cfg.get("llm_thinking", "disabled") or "disabled"
        )
        self._memory_prompt_language = str(
            self._memory_cfg.get("memory_prompt_language_mode")
            or self._memory_cfg.get("prompt_language_mode")
            or "source"
        )
        self._memory_enabled = bool(self._memory_cfg.get("memory_enabled", True))
        self._enable_memory_actionable_item_update = self._config_bool(
            self._memory_cfg.get("enable_memory_actionable_item_update", True),
            True,
        )
        self._enable_memory_entity_claim_update = self._config_bool(
            self._memory_cfg.get("enable_memory_entity_claim_update", True),
            True,
        )
        self._entity_claim_explicit_min_confidence = self._clamp_float(
            self._memory_cfg.get("entity_claim_explicit_min_confidence"),
            0.0,
            1.0,
            0.72,
        )
        self._entity_claim_induction_min_episodes = max(
            3,
            int(self._memory_cfg.get("entity_claim_induction_min_episodes", 3) or 3),
        )
        self._entity_claim_induction_min_time_windows = max(
            2,
            int(self._memory_cfg.get("entity_claim_induction_min_time_windows", 2) or 2),
        )
        self._initialize_recall_config()
        self._embedding_client: Optional[EmbeddingClient] = None
        self._task_queue_maxsize = max(
            1,
            int(self._memory_cfg.get("task_queue_maxsize", 100) or 100),
        )
        self._task_queue: queue.Queue[Dict[str, Any]] = queue.Queue(
            maxsize=self._task_queue_maxsize,
        )
        self._task_worker_thread: Optional[threading.Thread] = None
        self._task_worker_lock = threading.Lock()
        self._task_shutdown_event = threading.Event()
        self._memory_operation_lock = threading.RLock()
        # Kept by the serialized task worker: advance only after an episode and
        # its fact links have both been persisted successfully.
        self._episode_summary_last_fact_id_by_source: Dict[str, int] = {}

    def _initialize_recall_config(self) -> None:
        """Parse nested recall settings and keep legacy flat overrides working."""
        configured_recall = self._memory_cfg.get("recall")
        self._recall_cfg = dict(
            configured_recall if isinstance(configured_recall, dict) else {}
        )
        configured_stage1 = self._recall_cfg.get("recall_stage1")
        self._recall_stage1_cfg = dict(
            configured_stage1 if isinstance(configured_stage1, dict) else {}
        )
        configured_stage2 = self._recall_cfg.get("recall_stage2")
        self._recall_stage2_cfg = dict(
            configured_stage2 if isinstance(configured_stage2, dict) else {}
        )

        def recall_value(
            key: str,
            default: Any = None,
            *,
            stage: Optional[str] = None,
        ) -> Any:
            # Keep flat values as explicit overrides so existing callers that
            # patch ``memory_manager_config`` (for example CLI flags) retain
            # their current behavior while config.yaml uses the nested shape.
            if self._memory_cfg.get(key) is not None:
                return self._memory_cfg.get(key)
            stage_cfg = {
                "stage1": self._recall_stage1_cfg,
                "stage2": self._recall_stage2_cfg,
            }.get(stage, {})
            if key in stage_cfg and stage_cfg.get(key) is not None:
                return stage_cfg.get(key)
            if key in self._recall_cfg and self._recall_cfg.get(key) is not None:
                return self._recall_cfg.get(key)
            return default

        self._top_k = max(1, int(recall_value("recall_top_k", 8) or 8))
        self._recall_detailed_logging = self._config_bool(
            recall_value("recall_detailed_logging", False),
            False,
        )
        self._recall_budget = str(recall_value("recall_budget", "mid") or "mid")
        configured_source_override = recall_value("retrieval_source_override")
        if isinstance(configured_source_override, str):
            configured_source_override = [
                item.strip()
                for item in configured_source_override.split(",")
                if item.strip()
            ]
        self._retrieval_source_override = (
            self._normalize_source_override(configured_source_override)
            if isinstance(configured_source_override, (list, tuple, set))
            else None
        )
        self._recall_mode = str(
            recall_value("recall_mode", "normal") or "normal"
        ).strip().lower()

        self._recall_stage1_entity_matched_score = self._clamp_float(
            recall_value(
                "recall_stage1_entity_matched_score",
                0.30,
                stage="stage1",
            ),
            0.0,
            1.0,
            0.30,
        )
        self._recall_stage1_topic_overlap_score_weight = self._clamp_float(
            recall_value(
                "recall_stage1_topic_overlap_score_weight",
                0.54,
                stage="stage1",
            ),
            0.0,
            1.0,
            0.54,
        )
        self._recall_stage1_keyword_overlap_score_weight = self._clamp_float(
            recall_value(
                "recall_stage1_keyword_overlap_score_weight",
                0.12,
                stage="stage1",
            ),
            0.0,
            1.0,
            0.12,
        )
        self._recall_stage1_time_score_weight = self._clamp_float(
            recall_value(
                "recall_stage1_time_score_weight",
                None,
                stage="stage1",
            ),
            0.0,
            1.0,
            0.12,
        )
        self._recall_stage1_contextual_time_score_multiplier = max(
            1.0,
            float(
                recall_value(
                    "recall_stage1_contextual_time_score_multiplier",
                    2.0,
                    stage="stage1",
                )
                or 2.0
            ),
        )
        self._recall_stage1_min_term_coverage = self._clamp_float(
            recall_value(
                "recall_stage1_evidence_profile_min_term_coverage",
                0.5,
                stage="stage1",
            ),
            0.0,
            1.0,
            0.5,
        )
        self._recall_stage1_actionable_min_importance = self._clamp_float(
            recall_value(
                "recall_stage1_actionable_min_importance",
                0.75,
                stage="stage1",
            ),
            0.0,
            1.0,
            0.75,
        )
        self._recall_stage1_episode_propagation_decay = self._clamp_float(
            recall_value(
                "recall_stage1_episode_propagation_decay",
                0.70,
                stage="stage1",
            ),
            0.0,
            1.0,
            0.70,
        )
        self._recall_stage1_state_propagation_decay = self._clamp_float(
            recall_value(
                "recall_stage1_state_propagation_decay",
                0.80,
                stage="stage1",
            ),
            0.0,
            1.0,
            0.80,
        )
        self._recall_stage1_association_propagation_decay = self._clamp_float(
            recall_value(
                "recall_stage1_association_propagation_decay",
                0.80,
                stage="stage1",
            ),
            0.0,
            1.0,
            0.80,
        )

        self._recall_stage2_fact_min_embedding_similarity = self._clamp_float(
            recall_value(
                "recall_stage2_fact_min_embedding_similarity",
                0.35,
                stage="stage2",
            ),
            0.0,
            1.0,
            0.35,
        )
        self._recall_stage2_state_min_embedding_similarity = self._clamp_float(
            recall_value(
                "recall_stage2_state_min_embedding_similarity",
                0.35,
                stage="stage2",
            ),
            0.0,
            1.0,
            0.35,
        )
        # The seed-retrieval floors are intentionally lower than the
        # thresholds required for a candidate to become direct evidence.
        self._recall_stage2_fact_strong_embedding_similarity = (
            self._clamp_float(
                recall_value(
                    "recall_stage2_fact_strong_embedding_similarity",
                    0.45,
                    stage="stage2",
                ),
                0.0,
                1.0,
                0.45,
            )
        )
        self._recall_stage2_state_strong_embedding_similarity = (
            self._clamp_float(
                recall_value(
                    "recall_stage2_state_strong_embedding_similarity",
                    0.40,
                    stage="stage2",
                ),
                0.0,
                1.0,
                0.40,
            )
        )
        self._recall_stage2_strong_topic_pair_score = self._clamp_float(
            recall_value(
                "recall_stage2_strong_topic_pair_score",
                0.90,
                stage="stage2",
            ),
            0.0,
            1.0,
            0.90,
        )
        self._recall_stage2_embedding_score_weight = self._clamp_float(
            recall_value(
                "recall_stage2_embedding_score_weight",
                0.42,
                stage="stage2",
            ),
            0.0,
            1.0,
            0.42,
        )
        self._recall_stage2_topic_overlap_score_weight = self._clamp_float(
            recall_value(
                "recall_stage2_topic_overlap_score_weight",
                0.25,
                stage="stage2",
            ),
            0.0,
            1.0,
            0.25,
        )
        self._recall_stage2_keyword_match_score_weight = self._clamp_float(
            recall_value(
                "recall_stage2_keyword_match_score_weight",
                0.12,
                stage="stage2",
            ),
            0.0,
            1.0,
            0.12,
        )
        self._recall_stage2_bm25_score_weight = self._clamp_float(
            recall_value(
                "recall_stage2_bm25_score_weight",
                0.20,
                stage="stage2",
            ),
            0.0,
            1.0,
            0.20,
        )
        self._recall_stage2_entity_matched_score = self._clamp_float(
            recall_value(
                "recall_stage2_entity_matched_score",
                0.20,
                stage="stage2",
            ),
            0.0,
            1.0,
            0.20,
        )
        self._recall_stage2_time_score_weight = self._clamp_float(
            recall_value(
                "recall_stage2_time_score_weight",
                None,
                stage="stage2",
            ),
            0.0,
            1.0,
            0.12,
        )
        self._recall_stage2_contextual_time_score_multiplier = max(
            1.0,
            float(
                recall_value(
                    "recall_stage2_contextual_time_score_multiplier",
                    2.0,
                    stage="stage2",
                )
                or 2.0
            ),
        )
        self._recall_fact_time_score_half_life_seconds = max(
            1,
            int(recall_value("recall_fact_time_score_half_life_seconds", 604800) or 604800),
        )
        self._recall_persistent_state_time_score_half_life_seconds = max(
            1,
            int(recall_value("recall_persistent_state_time_score_half_life_seconds", 7776000) or 7776000),
        )

        self._recall_context_char_budgets = {
            "low": max(
                1200,
                int(recall_value("recall_context_chars_low", 3200) or 3200),
            ),
            "mid": max(
                1800,
                int(recall_value("recall_context_chars_mid", 6000) or 6000),
            ),
            "high": max(
                2400,
                int(recall_value("recall_context_chars_high", 10000) or 10000),
            ),
        }
        shared_recall_context_budget = recall_value("recall_context_max_chars")
        if shared_recall_context_budget not in (None, ""):
            shared_budget = max(1200, int(shared_recall_context_budget or 0))
            self._recall_context_char_budgets = {
                key: shared_budget for key in self._recall_context_char_budgets
            }
        self._recall_entry_char_budgets = {
            "low": max(
                260,
                int(recall_value("recall_entry_chars_low", 520) or 520),
            ),
            "mid": max(
                320,
                int(recall_value("recall_entry_chars_mid", 760) or 760),
            ),
            "high": max(
                420,
                int(recall_value("recall_entry_chars_high", 1100) or 1100),
            ),
        }

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

    @staticmethod
    def _episode_type_for_source_type(source_type: str) -> str:
        normalized = str(source_type or "").strip().lower()
        if normalized == "assistant_wakeup":
            return "interaction"
        if normalized == "allday_recording":
            return "ambient_transcript"
        return normalized or "memory"

    # ── Runtime helpers used by benchmark scripts ───────────────────────

    def _ensure_embedding_client(self) -> bool:
        if self._embedding_client is None:
            self._embedding_client = EmbeddingClient(self._embedding_cfg)
        return True

    # ── Store path: raw segments -> episode -> facts -> index cards ──────

    @property
    def enabled(self) -> bool:
        return self._memory_enabled

    def set_logger(self, logger: Optional[logging.Logger]) -> None:
        """Set the logger used by memory store, reflect, and recall operations."""
        self._logger = logger or logging.getLogger(__name__)

    def set_embedding_client(self, embedding_client: Optional[EmbeddingClient]) -> None:
        """Share a pre-initialized embedding client with memory runtime callers."""
        self._embedding_client = embedding_client

    def _task_worker_loop(self) -> None:
        while not self._task_shutdown_event.is_set() or not self._task_queue.empty():
            try:
                task = self._task_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            task_kind = str(task.get("kind") or "")
            task_id = str(task.get("task_id") or "")
            started_at = float(task.get("started_at") or time.monotonic())
            try:
                with self._memory_operation_lock:
                    if task_kind == "memory_store":
                        result = self._process_memory_store_task(**task["payload"])
                    elif task_kind == "memory_episode_summary":
                        result = self._process_memory_episode_summary_task(**task["payload"])
                    elif task_kind == "memory_reflect":
                        result = self._process_memory_reflect_task(**task["payload"])
                    else:
                        raise ValueError(f"Unsupported memory async task: {task_kind}")
                self._operation_reporter.on_task_finished(
                    operation_type=task_kind,
                    task_id=task_id,
                    started_at=started_at,
                    result=result,
                )
            except Exception as exc:
                self._logger.exception("Async memory %s failed: %s", task.get("kind"), exc)
                self._operation_reporter.on_task_finished(
                    operation_type=task_kind,
                    task_id=task_id,
                    started_at=started_at,
                    error=exc,
                )
            finally:
                self._task_queue.task_done()

    def _ensure_task_worker_locked(self) -> None:
        if self._task_worker_thread and self._task_worker_thread.is_alive():
            return
        self._task_worker_thread = threading.Thread(
            target=self._task_worker_loop,
            daemon=True,
            name="memory-node-worker",
        )
        self._task_worker_thread.start()

    def _submit_memory_task(
        self,
        *,
        task_kind: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        task_id = self._operation_reporter.next_task_id(task_kind)
        with self._task_worker_lock:
            if self._task_shutdown_event.is_set():
                self._logger.warning("Memory worker is shut down; dropping %s task", task_kind)
                return self._reject_memory_task(
                    task_kind=task_kind,
                    task_id=task_id,
                    reason="worker_shutdown",
                )
            try:
                self._task_queue.put_nowait({
                    "kind": task_kind,
                    "payload": payload,
                    "task_id": task_id,
                    "started_at": time.monotonic(),
                })
            except queue.Full:
                self._logger.warning(
                    "Memory task queue is full; dropping %s (maxsize=%d)",
                    task_kind,
                    self._task_queue_maxsize,
                )
                return self._reject_memory_task(
                    task_kind=task_kind,
                    task_id=task_id,
                    reason="worker_queue_full",
                )
            self._operation_reporter.on_task_submitted(
                operation_type=task_kind,
                task_id=task_id,
                payload={
                    key: value
                    for key, value in payload.items()
                    if key != "raw_segments"
                },
            )
            self._ensure_task_worker_locked()
        return {
            "queued": True,
            "status": "queued",
            "task_id": task_id,
            "operation_type": task_kind,
        }

    def _reject_memory_task(
        self,
        *,
        task_kind: str,
        task_id: str,
        reason: str,
    ) -> Dict[str, Any]:
        self._operation_reporter.on_task_rejected(
            operation_type=task_kind,
            task_id=task_id,
            reason=reason,
        )
        return {
            "queued": False,
            "status": "rejected",
            "reason": reason,
            "task_id": task_id,
            "operation_type": task_kind,
        }

    def flush_task_queue(self, timeout: Optional[float] = None) -> bool:
        """Wait until all queued asynchronous memory tasks finish."""
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while self._task_queue.unfinished_tasks:
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True

    def shutdown_task_worker(
        self,
        *,
        wait: bool = True,
        timeout: Optional[float] = None,
    ) -> bool:
        """Stop accepting tasks and optionally drain the memory worker.

        When ``wait`` is true, do not report shutdown as successful until every
        queued task has called ``task_done`` and the worker thread has exited.
        This is important because callers close the database immediately after
        shutdown; closing it while a reflection transaction is still running
        rolls back all state updates made by that transaction.
        """
        with self._task_worker_lock:
            self._task_shutdown_event.set()
            worker = self._task_worker_thread
        if not wait or worker is None:
            return not self._task_queue.unfinished_tasks

        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while self._task_queue.unfinished_tasks:
            if not worker.is_alive():
                return False
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        worker.join(timeout=remaining)
        return not worker.is_alive() and not self._task_queue.unfinished_tasks

    def submit_memory_store_task(
        self,
        *,
        raw_segments: List[Dict[str, Any]],
        source_type: str,
        tags: List[str],
        prompt_language: str,
    ) -> Dict[str, Any]:
        """Queue one normalized episode for ordered background storage."""
        if not self._memory_enabled or not raw_segments:
            reason = "memory_disabled" if not self._memory_enabled else "no_raw_segments"
            task_id = self._operation_reporter.next_task_id("memory_store")
            return self._reject_memory_task(
                task_kind="memory_store",
                task_id=task_id,
                reason=reason,
            )
        return self._submit_memory_task(
            task_kind="memory_store",
            payload={
                "raw_segments": raw_segments,
                "source_type": source_type,
                "tags": tags,
                "prompt_language": prompt_language,
            },
        )

    def submit_memory_episode_summary_task(
        self,
        *,
        source_type: str,
        tags: List[str],
        prompt_language: str,
    ) -> Dict[str, Any]:
        """Queue episode summarization after preceding fact-store tasks."""
        if not self._memory_enabled:
            task_id = self._operation_reporter.next_task_id("memory_episode_summary")
            return self._reject_memory_task(
                task_kind="memory_episode_summary", task_id=task_id, reason="memory_disabled"
            )
        return self._submit_memory_task(
            task_kind="memory_episode_summary",
            payload={
                "source_type": source_type,
                "tags": list(tags or []),
                "prompt_language": prompt_language,
            },
        )

    def _process_memory_episode_summary_task(
        self,
        *,
        source_type: str,
        tags: List[str],
        prompt_language: str,
    ) -> Dict[str, Any]:
        cursor_key = str(source_type or "")
        fact_id_after = self._episode_summary_last_fact_id_by_source.get(cursor_key, 0)
        facts = self._db.get_memory_facts_after_id(
            fact_id=fact_id_after,
            source_type=source_type,
            only_unassigned=True,
            limit=80,
        )
        episode_info = self.generate_episode_from_facts(
            facts=facts,
            prompt_language=prompt_language,
        )
        if episode_info.get("status") != "ok":
            return episode_info
        with self._db.transaction():
            episode_id = self._db.insert_episode(
                source_type=source_type,
                episode_type=self._episode_type_for_source_type(source_type),
                title=episode_info["title"],
                summary=episode_info["summary"],
                participants=episode_info.get("participants") or [],
                started_at=episode_info.get("started_at") or _now_text(),
                ended_at=episode_info.get("ended_at") or episode_info.get("started_at") or _now_text(),
                canonical_topics=episode_info.get("canonical_topics") or [],
                entity_ids=episode_info.get("entity_ids") or [],
                metadata={"tags": list(tags or []), "fact_count": len(facts), "generated_from_facts": True},
            )
            topic_report = self._db.upsert_memory_topic_items(
                self._build_memory_topic_item_updates(
                    canonical_topics=episode_info.get("canonical_topics") or [],
                    episode_id=episode_id,
                )
            )
            attached = self._db.update_facts_episode_id(
                fact_ids=episode_info.get("fact_ids") or [], episode_id=episode_id,
            )
        fact_ids = [int(fact_id) for fact_id in (episode_info.get("fact_ids") or [])]
        last_fact_id = fact_id_after
        if attached and fact_ids:
            last_fact_id = max(fact_ids)
            self._episode_summary_last_fact_id_by_source[cursor_key] = last_fact_id
        episode_info.update({
            "episode_id": episode_id,
            "fact_count": attached,
            "new_episode_count": 1,
            "topic_items_created": int(topic_report.get("created_count", 0) or 0),
            "topic_items_updated": int(topic_report.get("updated_count", 0) or 0),
            "fact_id_after": fact_id_after,
            "last_fact_id": last_fact_id,
        })
        self._log_info("memory_store", "episode_summary_generated", {
            "episode_id": episode_id,
            "fact_count": attached,
            "fact_id_after": fact_id_after,
            "last_fact_id": last_fact_id,
            "source_type": source_type,
            "title": episode_info.get("title") or "",
            "topic_items_created": episode_info.get("topic_items_created", 0),
            "topic_items_updated": episode_info.get("topic_items_updated", 0),
        })
        return episode_info

    def _process_memory_store_task(
        self,
        *,
        raw_segments: List[Dict[str, Any]],
        source_type: str,
        tags: List[str],
        prompt_language: str,
    ) -> Dict[str, Any]:
        store_started_at = time.monotonic()
        self._log_info("memory_store", "start", {
            "source_type": source_type,
            "source_segment_count": len(raw_segments),
            "raw_segments": self._build_memory_segments_for_prompt(
                raw_segments,
                prompt_language=prompt_language,
            ),
        })
        if not raw_segments:
            elapsed_ms = round((time.monotonic() - store_started_at) * 1000, 2)
            self._log_info("memory_store", "finish", {
                "status": "skipped",
                "reason": "no_raw_segments",
                "total_elapsed_ms": elapsed_ms,
            })
            return {
                "status": "skipped",
                "reason": "no_raw_segments",
                "new_episode_count": 0,
                "new_fact_count": 0,
                "total_elapsed_ms": elapsed_ms,
            }
        extracted_info = self._extract_memory_fact_from_raw_segments(
            raw_segments,
            prompt_language=prompt_language,
        )
        facts = list(extracted_info.get("facts") or [])
        self._log_extracted_fact_info(facts=facts)
        with self._db.transaction():
            save_entity_info = self._store_extracted_memory_entities_into_db(
                participants=[], raw_segments=raw_segments, facts=facts, episode_summary="",
            )
            save_fact_info = self._store_extracted_memory_facts_into_db(
                episode_id=None,
                facts=facts,
                tags=tags,
                source_type=source_type,
                episode_context_topics=None,
                entity_info=save_entity_info,
            )
            topic_report = self._db.upsert_memory_topic_items(
                save_fact_info.get("topic_item_updates") or []
            )
        report = {
            "status": "ok",
            "new_episode_count": 0,
            "new_fact_count": len(list(save_fact_info.get("fact_ids") or [])),
            "fact_ids": list(save_fact_info.get("fact_ids") or []),
            "topic_items_created": int(topic_report.get("created_count", 0) or 0),
            "topic_items_updated": int(topic_report.get("updated_count", 0) or 0),
            "total_elapsed_ms": round((time.monotonic() - store_started_at) * 1000, 2),
        }
        self._log_info("memory_store", "finish", {
            **report,
            "source_type": source_type,
            "source_segment_count": len(raw_segments),
        })
        return report

    def _store_extracted_memory_entities_into_db(
        self,
        *,
        participants: List[str],
        raw_segments: List[Dict[str, Any]],
        facts: List[Dict[str, Any]],
        episode_summary: str,
    ) -> Dict[str, int]:
        """Persist all episode entities once and return name-to-id mappings."""
        entity_names = self._episode_entity_names(
            participants=participants,
            segments=raw_segments,
            facts=facts,
            summary=episode_summary,
        )
        mapping = self._db.add_entity_names(entity_names)
        return {
            str(entity_name): int(entity_id)
            for entity_name, entity_id in mapping.items()
            if str(entity_name).strip() and str(entity_id).strip().isdigit()
        }

    def _log_extracted_fact_info(
        self,
        *,
        facts: List[Dict[str, Any]],
    ) -> None:
        if not facts:
            return
        for index, fact in enumerate(facts, 1):
            metadata = fact.get("metadata") if isinstance(fact.get("metadata"), dict) else {}
            self._log_info(
                "memory_store",
                "extract_fact_signals",
                {
                    "fact_index": index,
                    "fact_count": len(facts),
                    "summary": fact.get("summary") or "",
                    "fact_type": fact.get("fact_type"),
                    "fact_kind": fact.get("fact_kind"),
                    "event_time_key": fact.get("event_time_key") or "",
                    "dialogue_time_key": fact.get("dialogue_time_key") or "",
                    "keywords": fact.get("keywords") or "",
                    "entities": fact.get("entities") or [],
                    "primary_entity": fact.get("primary_entity"),
                    "fact_root_topic": fact.get("fact_root_topic") or "",
                    "fact_aspect_topic": fact.get("fact_aspect_topic") or "",
                    "entity_claim_signal": fact.get("entity_claim_signal") or [],
                    "action_signal": fact.get("action_signal") or [],
                    "importance": fact.get("importance"),
                    "confidence": fact.get("confidence"),
                    "time_confidence": metadata.get("time_confidence") or "",
                    "where": metadata.get("where") or "",
                    "metadata": metadata,
                    "batch_fact_index": index,
                    "batch_fact_count": len(facts),
                },
            )

    def _extract_memory_fact_from_raw_segments(
        self,
        raw_segments: List[Dict[str, Any]],
        *,
        prompt_language: str,
    ) -> Dict[str, Any]:
        """Extract narrative facts with the unified fact prompt."""
        data = self._extract_memory_fact_with_llm(
            raw_segments,
            prompt_language=prompt_language,
        )
        if data and data.get("facts"):
            return data
        return {"facts": []}

    def generate_episode_from_facts(
        self,
        *,
        facts: Sequence[Dict[str, Any]],
        prompt_language: str = "zh",
    ) -> Dict[str, Any]:
        """Summarize a logical episode from facts already persisted by extraction."""
        fact_list = [item for item in facts if isinstance(item, dict) and _compact_whitespace(item.get("summary") or item.get("text") or "")]
        if not fact_list:
            return {"status": "empty", "new_episode_count": 0, "fact_count": 0}
        payload = [
            {
                "summary": _compact_whitespace(item.get("summary") or item.get("text") or ""),
                "fact_kind": item.get("fact_kind") or "",
                "fact_root_topic": item.get("fact_root_topic") or "",
                "fact_aspect_topic": item.get("fact_aspect_topic") or "",
                "dialogue_time_key": item.get("dialogue_time_key") or "",
                "event_time_key": item.get("event_time_key") or "",
            }
            for item in fact_list
        ]
        prompt_template = EPISODE_SUMMARY_PROMPT_EN if prompt_language == "en" else EPISODE_SUMMARY_PROMPT_ZH
        prompt = prompt_template.replace("{facts}", json.dumps(payload, ensure_ascii=False))
        parsed: Dict[str, Any] = {}
        for _ in range(2):
            parsed = self._parse_json_object_from_llm_text(self._call_llm(prompt) or "") or {}
            if _compact_whitespace(parsed.get("summary") or ""):
                break
        summary = _compact_whitespace(parsed.get("summary") or "")
        if not summary:
            summary = "；".join(item["summary"] for item in payload)[:2000]
        title = _compact_whitespace(parsed.get("title") or "")
        if not title:
            title = _compact_whitespace(payload[0].get("fact_root_topic") or "本次对话")[:80]
        topics = self._normalize_episode_canonical_topics(
            parsed.get("canonical_topics"), fallback_text=" ".join(item["summary"] for item in payload), limit=3,
        )
        if not topics:
            topics = self._normalize_unique_labels([item.get("fact_root_topic") for item in payload if item.get("fact_root_topic")])[:3] or self._topic_candidates(summary)[:3]
        participants = self._normalize_unique_labels([
            entity
            for item in fact_list
            for entity in (item.get("entities") or [])
            if isinstance(entity, str)
            and entity.strip().lower() in {"user", "assistant", "用户", "助手", "speaker_1", "speaker_2"}
        ])[:20]
        episode_entity_ids = sorted({
            int(entity_id)
            for item in fact_list
            for entity_id in (item.get("entity_ids") or [])
            if str(entity_id).strip().isdigit()
        })
        dialogue_times = [
            _compact_whitespace(item.get("dialogue_time_key") or "")
            for item in fact_list
            if _compact_whitespace(item.get("dialogue_time_key") or "")
        ]
        start = dialogue_times[0] if dialogue_times else _now_text()
        end = dialogue_times[-1] if dialogue_times else start
        fact_ids = [int(item["id"]) for item in fact_list if str(item.get("id", "")).isdigit()]
        return {
            "status": "ok",
            "new_episode_count": 0,
            "title": title,
            "summary": summary,
            "canonical_topics": topics,
            "participants": participants,
            "entity_ids": episode_entity_ids,
            "fact_ids": fact_ids,
            "started_at": start,
            "ended_at": end,
        }

    def _extract_memory_fact_with_llm(
        self,
        segments: List[Dict[str, Any]],
        *,
        prompt_language: str,
    ) -> Optional[Dict[str, Any]]:
        prompt_template = (
            UNIFIED_MEMORY_EXTRACTION_PROMPT_EN
            if prompt_language == "en"
            else UNIFIED_MEMORY_EXTRACTION_PROMPT_ZH
        )
        memory_state_context = self._collect_memory_state_context(limit=12)
        memory_topic_item_context = (
            self._collect_memory_topic_item_context(segments=segments)
            if prompt_language != "en"
            else {"canonical_topics": [], "aspect_topics": []}
        )
        prompt = (
            prompt_template
            .replace(
                "{existing_memory_states}",
                self._format_memory_states_for_prompt(memory_state_context),
            )
            .replace(
                "{existing_memory_topic_items}",
                self._format_memory_topic_items_for_prompt(memory_topic_item_context),
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
                normalized = self._normalize_memory_fact_extraction_llm_output(
                    parsed,
                    segments,
                    prompt_language=prompt_language,
                )
                if normalized is not None:
                    return normalized
            if attempt == 0:
                self._logger.debug("Unified memory LLM extraction failed, retrying")
        return None

    def _call_llm(self, prompt: str) -> Optional[str]:
        if (
            not self._llm_api_key
            or not self._llm_base_url
            or str(self._llm_base_url).strip().lower() == "none"
        ):
            self._logger.debug(
                "Skipping LLM call because llm_api_key or llm_base_url is not configured"
            )
            return None
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
                self._logger.warning("Unified memory LLM call failed: %s", exc)
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

    def _normalize_memory_fact_extraction_llm_output(
        self,
        data: Dict[str, Any],
        raw_segments: List[Dict[str, Any]],
        *,
        prompt_language: str,
    ) -> Optional[Dict[str, Any]]:
        raw_facts = data.get("facts")
        if not isinstance(raw_facts, list):
            return None
        facts: List[Dict[str, Any]] = []
        dialogue_time_key = _to_timestamp_text(
            raw_segments[0].get("started_at") if raw_segments else ""
        ) or _now_text()
        for raw_fact in raw_facts:
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
            primary_entity = self._normalize_primary_entity(
                raw_fact.get("primary_entity"),
                entities=entities,
            )
            if primary_entity:
                primary_entity_name = primary_entity["name"]
                if primary_entity_name not in entities:
                    entities = [primary_entity_name, *entities]
            primary_entity_name = _compact_whitespace(
                (primary_entity or {}).get("name") or ""
            ).lower()
            if primary_entity_name in {"assistant", "agent", "the assistant", "助手"} \
                    and self._is_low_value_assistant_closing(text):
                continue
            if primary_entity_name in {"user", "the user", "用户"} \
                    and self._is_low_value_user_acknowledgement(text):
                continue
            fact_topic_fallback = " ".join(keywords[:3]) if keywords else "general"
            fact_root_topic, fact_aspect_topic = self._normalize_fact_topic_fields(
                raw_fact.get("fact_root_topic"),
                raw_fact.get("fact_aspect_topic"),
                fallback_root_topic=fact_topic_fallback,
                fallback_aspect_topic=fact_topic_fallback,
            )
            entity_claim_signal = self._normalize_entity_claim_signal(
                raw_fact.get("entity_claim_signal"),
                fallback_entity=primary_entity,
            )
            action_signal = self._normalize_action_signal(
                raw_fact.get("action_signal"),
            )
            event_time_key = _compact_whitespace(raw_fact.get("event_time_key") or "")
            facts.append({
                "summary": text,
                "fact_kind": self._normalize_fact_kind(raw_fact.get("fact_kind")),
                "fact_type": self._normalize_fact_type(raw_fact.get("fact_type")),
                "event_time_key": event_time_key,
                "dialogue_time_key": dialogue_time_key,
                "keywords": keywords,
                "entities": entities,
                "primary_entity": primary_entity,
                "entity_claim_signal": entity_claim_signal,
                "action_signal": action_signal,
                "fact_root_topic": fact_root_topic,
                "fact_aspect_topic": fact_aspect_topic,
                "importance": max(0.6, min(1.0, priority / 100.0)),
                "confidence": 0.9,
                "metadata": {
                    "extractor": "llm",
                    "priority": priority,
                    "time_confidence": _compact_whitespace(raw_fact.get("time_confidence") or "unknown"),
                    "where": _compact_whitespace(raw_fact.get("where") or ""),
                },
            })
        return {"facts": facts}

    def _normalize_fact_topic_fields(
        self,
        root_topic: Any,
        aspect_topic: Any,
        *,
        fallback_root_topic: Any,
        fallback_aspect_topic: Any,
    ) -> Tuple[str, str]:
        normalized_root = (
            self._normalize_topic_name(root_topic)
            or self._normalize_topic_name(fallback_root_topic)
            or "general"
        )
        normalized_aspect = (
            self._normalize_topic_name(aspect_topic)
            or self._normalize_topic_name(fallback_aspect_topic)
            or normalized_root
        )
        return normalized_root, normalized_aspect

    def _normalize_entity_claim_signal(
        self,
        value: Any,
        *,
        fallback_entity: Optional[Dict[str, str]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        max_items = max(
            0,
            int(
                limit
                if limit is not None
                else self._memory_cfg.get("entity_claim_signal_max_per_fact", 3) or 3
            ),
        )
        if max_items <= 0:
            return []
        allowed_types = self._entity_claim_types()
        allowed_kinds = {
            "explicit_assertion", "pattern_observation", "counterexample",
        }
        normalized: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str, str]] = set()
        for raw in value:
            if not isinstance(raw, dict):
                continue
            signal_kind = str(raw.get("signal_kind") or "").strip().lower()
            claim_type_hint = str(raw.get("claim_type_hint") or "").strip().lower()
            if signal_kind not in allowed_kinds or claim_type_hint not in allowed_types:
                continue
            if (
                signal_kind == "explicit_assertion"
                and claim_type_hint == "behavior_pattern"
            ):
                continue
            if (
                signal_kind in {"pattern_observation", "counterexample"}
                and claim_type_hint not in {"preference", "behavior_pattern"}
            ):
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
            entity = raw.get("entity") or raw.get("primary_entity") or fallback_entity
            if isinstance(entity, dict):
                entity_name = _compact_whitespace(entity.get("name") or entity.get("text") or "")
                entity_type = _compact_whitespace(entity.get("type") or "CONCEPT").upper()
            else:
                entity_name = _compact_whitespace(entity)
                entity_type = "CONCEPT"
            entity_payload = (
                {"name": entity_name, "type": entity_type}
                if entity_name
                else None
            )
            key = (
                signal_kind,
                claim_type_hint,
                claim_anchor.lower(),
                evidence_basis.lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            item: Dict[str, Any] = {
                "signal_kind": signal_kind,
                "claim_type_hint": claim_type_hint,
                "claim_anchor": claim_anchor,
                "evidence_basis": evidence_basis,
                "confidence": confidence,
            }
            if entity_payload:
                item["entity"] = entity_payload
            normalized.append(item)
            if len(normalized) >= max_items:
                break
        return normalized

    def _normalize_action_signal(
        self,
        value: Any,
        *,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        max_items = max(
            0,
            int(
                limit
                if limit is not None
                else self._memory_cfg.get("action_signal_max_per_fact", 2) or 2
            ),
        )
        if max_items <= 0:
            return []
        allowed_types = {
            "task", "commitment", "decision", "follow_up", "open_question",
            "risk", "reminder", "recommendation", "constraint",
        }
        allowed_strengths = {
            "assigned", "committed", "pending_decision", "follow_up",
        }
        normalized: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str]] = set()
        for raw in value:
            if not isinstance(raw, dict):
                continue
            item_type = self._normalize_actionable_item_type(raw.get("item_type"))
            if item_type not in allowed_types:
                continue
            action_strength = _compact_whitespace(
                raw.get("action_strength") or ""
            ).lower()
            if action_strength not in allowed_strengths:
                continue
            evidence_basis = _compact_whitespace(
                raw.get("evidence_basis")
                or raw.get("evidence")
                or raw.get("reason")
                or ""
            )
            if not evidence_basis:
                continue
            due_at = _compact_whitespace(raw.get("due_at") or "")
            confidence = self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.75)
            if confidence < 0.7:
                continue
            if item_type == "decision" and action_strength != "pending_decision":
                continue
            key = (item_type, evidence_basis.lower())
            if key in seen:
                continue
            seen.add(key)
            normalized.append({
                "item_type": item_type,
                "action_strength": action_strength,
                "due_at": due_at,
                "evidence_basis": evidence_basis,
                "confidence": confidence,
            })
            if len(normalized) >= max_items:
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

    @staticmethod
    def _parse_participants_from_raw_segments(raw_segments: List[Dict[str, Any]]) -> List[str]:
        participants: List[str] = []
        seen: set[str] = set()
        for segment in raw_segments:
            speaker = _compact_whitespace(segment.get("speaker") or "")
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

    def _entity_ids_for_names(self, names: Sequence[Any], *, limit: int = 64) -> List[int]:
        normalized: List[str] = []
        for name in names or []:
            self._append_unique_text(normalized, name, limit=limit)
        mapping = self._db.add_entity_names(normalized)
        ids: List[int] = []
        for name in normalized:
            entity_id = mapping.get(name)
            if entity_id and entity_id not in ids:
                ids.append(entity_id)
        return ids

    def _entity_ids_from_names_and_facts(
        self,
        *,
        names: Sequence[Any],
        facts: Sequence[Dict[str, Any]],
        limit: int = 64,
    ) -> List[int]:
        ids: List[int] = []
        for fact in facts or []:
            for value in fact.get("entity_ids") or []:
                try:
                    entity_id = int(value)
                except (TypeError, ValueError):
                    continue
                if entity_id and entity_id not in ids:
                    ids.append(entity_id)
                if len(ids) >= limit:
                    return ids
        for entity_id in self._entity_ids_for_names(names, limit=limit):
            if entity_id not in ids:
                ids.append(entity_id)
            if len(ids) >= limit:
                break
        return ids

    def _fact_entity_names(
        self,
        fact: Dict[str, Any],
        *,
        entities: Optional[Sequence[str]] = None,
    ) -> List[str]:
        names: List[str] = []
        normalized_entities = (
            list(entities)
            if entities is not None
            else self._normalize_entity_names(fact.get("entities"), limit=32)
        )
        for entity in normalized_entities:
            self._append_unique_text(names, entity)
        primary = fact.get("primary_entity")
        if isinstance(primary, dict):
            self._append_unique_text(names, primary.get("name") or primary.get("text"))
        else:
            self._append_unique_text(names, primary)
        return names

    def _episode_entity_names(
        self,
        *,
        participants: Sequence[str],
        segments: Sequence[Dict[str, Any]],
        facts: Sequence[Dict[str, Any]],
        summary: str,
    ) -> List[str]:
        names: List[str] = []
        for participant in participants or []:
            self._append_unique_text(names, participant)
        for fact in facts or []:
            for entity in self._fact_entity_names(fact):
                self._append_unique_text(names, entity)
        for entity in self._entities(summary):
            self._append_unique_text(names, entity)
        if not names:
            for segment in segments[:12]:
                for entity in self._entities(segment.get("text") or ""):
                    self._append_unique_text(names, entity)
        return names

    def _build_memory_segments_for_prompt(
        self,
        segments: List[Dict[str, Any]],
        *,
        prompt_language: str,
    ) -> str:
        is_en = prompt_language == "en"
        time_label = "Time" if is_en else "时间"
        speaker_label = "Speaker" if is_en else "说话人"
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
                    f"{text_label}: {segment.get('text') or ''}",
                ])
            )
        return "\n\n".join(blocks)

    def _collect_memory_state_context(self, *, limit: int = 12) -> List[Dict[str, Any]]:
        """Collect entity-state references for fact extraction only."""
        try:
            states = self._db.get_recent_memory_states(
                state_scope="entity_state",
                limit=max(40, int(limit or 12) * 6),
            )
        except Exception as exc:
            self._logger.debug("Failed to load memory state context: %s", exc)
            return []
        rows: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str]] = set()
        type_counts: Counter[Tuple[str, str]] = Counter()
        max_items = max(1, int(limit or 12))
        for state in states:
            scope = _compact_whitespace(state.get("state_scope") or "")
            state_type = _compact_whitespace(state.get("state_type") or "")
            canonical_name = _compact_whitespace(state.get("canonical_name") or "")
            summary = self._normalize_state_summary(
                state.get("summary") or "",
                max_chars=120,
            )
            if scope != "entity_state" or not state_type or not canonical_name or not summary:
                continue
            key = (scope, state_type, canonical_name.lower())
            if key in seen:
                continue
            type_key = (scope, state_type)
            if type_counts[type_key] >= 2:
                continue
            seen.add(key)
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

    def _is_indexable_memory_topic(self, value: Any) -> bool:
        """Reject fallback labels that must not become reusable topic names."""
        topic = self._normalize_topic_name(value)
        if not topic:
            return False
        return self._generate_topic_name_key(topic) not in {
            "general",
            "通用",
            "其他",
            "其它",
            "unknown",
            "未知",
        }

    def _build_memory_topic_item_updates(
        self,
        *,
        canonical_topics: Sequence[Any],
        aspect_topics: Sequence[Any] = (),
        fact_id: Optional[int] = None,
        episode_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Create compact topic-registry updates from persisted memory fields."""
        normalized_fact_id = int(fact_id or 0)
        normalized_episode_id = int(episode_id or 0)
        updates: List[Dict[str, Any]] = []
        canonical_keys: set[str] = set()

        def add(topic_kind: str, value: Any) -> None:
            topic_name = self._normalize_topic_name(value)
            if not topic_name or not self._is_indexable_memory_topic(topic_name):
                return
            topic_key = self._generate_topic_name_key(topic_name)
            if topic_kind == "aspect" and topic_key in canonical_keys:
                # An aspect equal to its fact root adds no finer-grained
                # vocabulary and would only duplicate prompt candidates.
                return
            item: Dict[str, Any] = {
                "topic_kind": topic_kind,
                "topic_name": topic_name,
                "topic_key": topic_key,
                "fact_ids": [normalized_fact_id] if normalized_fact_id > 0 else [],
                "episode_ids": [normalized_episode_id] if normalized_episode_id > 0 else [],
            }
            if not any(
                existing["topic_kind"] == topic_kind
                and existing["topic_key"] == topic_key
                for existing in updates
            ):
                updates.append(item)

        for topic in canonical_topics or ():
            topic_name = self._normalize_topic_name(topic)
            if topic_name and self._is_indexable_memory_topic(topic_name):
                canonical_keys.add(self._generate_topic_name_key(topic_name))
            add("canonical", topic)
        # Aspect topics are defined only by facts. Episode callers therefore
        # leave this sequence empty.
        if normalized_fact_id > 0:
            for topic in aspect_topics or ():
                add("aspect", topic)
        return updates

    def _collect_memory_topic_item_context(
        self,
        *,
        segments: Sequence[Dict[str, Any]],
        canonical_limit: int = 12,
        aspect_limit: int = 12,
    ) -> Dict[str, List[str]]:
        """Return only topic names lexically related to the incoming evidence."""
        query_text = " ".join(
            _compact_whitespace(segment.get("text") or "")
            for segment in segments or ()
            if isinstance(segment, dict)
        )
        query_terms = self._lexical_search_terms_for_text(
            query_text,
            limit=24,
            preserve_phrase=False,
        )
        if not query_terms:
            return {"canonical_topics": [], "aspect_topics": []}
        try:
            rows = self._db.list_memory_topic_items(limit=240)
        except Exception as exc:
            self._logger.debug("Failed to load memory topic item context: %s", exc)
            return {"canonical_topics": [], "aspect_topics": []}

        ranked: List[Tuple[float, Dict[str, Any]]] = []
        query_key = self._generate_topic_name_key(query_text)
        for row in rows:
            topic_name = self._normalize_topic_name(row.get("topic_name") or "")
            topic_key = self._generate_topic_name_key(topic_name) if topic_name else ""
            if not topic_name or not topic_key:
                continue
            score = self._topic_name_best_pair_similarity(
                query_terms,
                [topic_name],
            )
            if query_key and topic_key and topic_key in query_key:
                score = max(score, 0.9)
            if score < 0.5:
                continue
            ranked.append((score, row))

        result: Dict[str, List[str]] = {
            "canonical_topics": [],
            "aspect_topics": [],
        }
        limits = {
            "canonical": max(1, int(canonical_limit or 12)),
            "aspect": max(1, int(aspect_limit or 12)),
        }
        for _score, row in sorted(
            ranked,
            key=lambda item: (
                item[0],
                str(item[1].get("last_seen_at") or ""),
                int(item[1].get("fact_occurrence_count") or 0)
                + int(item[1].get("episode_occurrence_count") or 0),
            ),
            reverse=True,
        ):
            kind = str(row.get("topic_kind") or "").strip().lower()
            result_key = f"{kind}_topics"
            topic_name = self._normalize_topic_name(row.get("topic_name") or "")
            if (
                kind not in limits
                or not topic_name
                or topic_name in result[result_key]
                or len(result[result_key]) >= limits[kind]
            ):
                continue
            result[result_key].append(topic_name)
        return result

    @staticmethod
    def _format_memory_topic_items_for_prompt(
        topic_items: Dict[str, List[str]],
        *,
        max_chars: int = 1200,
    ) -> str:
        payload = {
            "canonical_topics": list(topic_items.get("canonical_topics") or []),
            "aspect_topics": list(topic_items.get("aspect_topics") or []),
        }
        while payload["canonical_topics"] or payload["aspect_topics"]:
            text = json.dumps(payload, ensure_ascii=False, indent=2)
            if len(text) <= max_chars:
                return text
            if len(payload["aspect_topics"]) >= len(payload["canonical_topics"]):
                payload["aspect_topics"].pop()
            else:
                payload["canonical_topics"].pop()
        return "[]"

    def _normalize_episode_canonical_topics(
        self,
        value: Any,
        *,
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

    def _fallback_generate_episode_title_from_raw_segments(self, raw_segments: List[Dict[str, Any]]) -> str:
        for segment in raw_segments:
            text = segment.get("text") or ""
            if text:
                return _compact_whitespace(text)[:96]
        return "memory episode"

    def _fallback_generate_episode_summary_from_raw_segments(self, raw_segments: List[Dict[str, Any]]) -> str:
        chunks: List[str] = []
        for segment in raw_segments[:10]:
            speaker = segment.get("speaker") or "speaker"
            text = _compact_whitespace(segment.get("text") or "")
            if not text:
                continue
            started_at = segment.get("started_at") or ""
            chunks.append(f"{started_at} {speaker}: {text[:600]}")
        return "\n".join(chunks)

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

    def _store_extracted_memory_facts_into_db(
        self,
        *,
        episode_id: Optional[int],
        facts: List[Dict[str, Any]],
        tags: List[str],
        source_type: str,
        episode_context_topics: Optional[Sequence[str]] = None,
        entity_info: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        fact_ids: List[int] = []
        topic_item_updates: List[Dict[str, Any]] = []
        normalized_entity_info = {
            str(entity_name): int(entity_id)
            for entity_name, entity_id in (entity_info or {}).items()
            if str(entity_name).strip() and str(entity_id).strip().isdigit()
        }
        episode_context_entities = list(normalized_entity_info.keys())
        for fact in facts:
            keywords = self._normalize_string_list(
                fact.get("keywords"),
                limit=18,
            )
            if not keywords:
                keywords = self._keywords(fact.get("summary") or "", limit=18)
            keyword_text = " ".join(keywords)
            entities = self._normalize_entity_names(fact.get("entities"))
            raw_metadata = fact.get("metadata")
            metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
            fallback_topics = self._topic_candidates(fact["summary"])
            fact_root_topic = self._normalize_topic_name(
                fact.get("fact_root_topic")
                or next(iter(episode_context_topics or []), "")
            ) or self._normalize_topic_name(
                fallback_topics[0] if fallback_topics else ""
            ) or "general"
            fact_aspect_topic = self._normalize_topic_name(
                fact.get("fact_aspect_topic")
                or fact_root_topic
            ) or fact_root_topic
            identity_text = "\n".join([
                f"summary: {_compact_whitespace(fact['summary'])}",
                f"keywords: {keyword_text}",
                f"entities: {', '.join(entities)}",
                f"primary_entity: {(fact.get('primary_entity') or {}).get('name', '') if isinstance(fact.get('primary_entity'), dict) else ''}",
                f"fact_root_topic: {fact_root_topic}",
                f"fact_aspect_topic: {fact_aspect_topic}",
            ])
            identity_text_embedding = self._generate_embedding_vector(identity_text)
            fact_entities = self._fact_entity_names(fact, entities=entities)
            entity_ids = [
                normalized_entity_info[entity_name]
                for entity_name in fact_entities
                if entity_name in normalized_entity_info
            ]
            fact_metadata = {
                **metadata,
                "tags": tags,
                "entity_claim_signal": fact.get("entity_claim_signal") or [],
                "action_signal": fact.get("action_signal") or [],
                "episode_context_topics": list(episode_context_topics or []),
                "episode_context_entities": list(episode_context_entities or []),
            }
            fact_id = self._db.insert_fact(
                episode_id=episode_id,
                source_type=source_type,
                fact_type=fact["fact_type"],
                fact_kind=fact["fact_kind"],
                summary=fact["summary"],
                keywords=keywords,
                entities=entities,
                entity_ids=entity_ids,
                fact_root_topic=fact_root_topic,
                fact_aspect_topic=fact_aspect_topic,
                event_time_key=fact.get("event_time_key") or "",
                dialogue_time_key=fact.get("dialogue_time_key") or "",
                confidence=fact["confidence"],
                importance=fact["importance"],
                metadata=fact_metadata,
                identity_text_embedding=identity_text_embedding,
                identity_text=identity_text,
            )
            fact_ids.append(fact_id)
            topic_item_updates.extend(
                self._build_memory_topic_item_updates(
                    canonical_topics=[fact_root_topic],
                    aspect_topics=[fact_aspect_topic],
                    fact_id=fact_id,
                )
            )
        return {
            "fact_ids": fact_ids,
            "topic_item_updates": topic_item_updates,
        }

    # ── Reflection: facts/episodes -> entity claims ──────────────────────

    def submit_memory_reflect_task(self, *_, **kwargs: Any) -> Dict[str, Any]:
        """Queue reflection after all previously accepted memory tasks."""
        if not self._memory_enabled:
            task_id = self._operation_reporter.next_task_id("memory_reflect")
            return self._reject_memory_task(
                task_kind="memory_reflect",
                task_id=task_id,
                reason="memory_disabled",
            )
        return self._submit_memory_task(
            task_kind="memory_reflect",
            payload=dict(kwargs),
        )

    def _process_memory_reflect_task(
        self,
        limit: Optional[int] = None,
        reflect_timestamp: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Project new facts and completed episodes into entity claims."""
        reflect_started_at = time.monotonic()
        limit = max(1, int(limit or self._memory_cfg.get("reflect_limit") or 100))
        if reflect_timestamp is None:
            reflect_timestamp = kwargs.get("timestamp") or _now_text()
        self._log_info("memory_reflect", "start", {
            "limit": limit,
            "reflect_timestamp": reflect_timestamp,
        })
        with self._db.transaction():
            claim_report = self._update_memory_entity_claims(
                limit=limit,
                reference_timestamp=reflect_timestamp,
            )
        report = {
            "status": (
                "ok"
                if (
                    claim_report.get("explicit", {}).get("fact_count", 0)
                    or claim_report.get("inductive", {}).get("episode_count", 0)
                )
                else "empty"
            ),
            "explicit_claims_updated": int(
                claim_report.get("explicit", {}).get("updated", 0) or 0
            ),
            "inductive_claims_updated": int(
                claim_report.get("inductive", {}).get("updated", 0) or 0
            ),
            "facts_marked_processed_for_memory_entity_claim": int(
                claim_report.get("explicit", {}).get("facts_marked_processed", 0) or 0
            ),
            "episodes_marked_processed_for_entity_claim_induction": int(
                claim_report.get("inductive", {}).get("episodes_marked_processed", 0) or 0
            ),
            "legacy_actionable_item_flow": "disabled_pending_work_item_redesign",
            "total_elapsed_ms": round(
                (time.monotonic() - reflect_started_at) * 1000,
                2,
            ),
        }
        self._log_info("memory_reflect", "finish", report)
        return report

    @staticmethod
    def _entity_claim_types() -> set[str]:
        return {
            "identity_profile", "affiliation", "relationship", "preference",
            "constraint", "behavior_pattern",
        }

    def _update_memory_entity_claims(
        self,
        *,
        limit: int,
        reference_timestamp: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """Project facts and completed episodes into traceable claim records."""
        disabled = {
            "enabled": 0, "updated": 0, "facts_marked_processed": 0,
            "episodes_marked_processed": 0,
        }
        if not self._enable_memory_entity_claim_update:
            return {"explicit": dict(disabled), "inductive": dict(disabled)}

        explicit_facts = self._db.get_unprocessed_facts(
            processing_target="entity_claim",
            limit=limit,
            reference_timestamp=reference_timestamp,
        )
        explicit_report = self._update_explicit_entity_claims_from_facts(explicit_facts)
        # A valid empty result is a completed projection.  An unavailable or
        # malformed LLM response is retried on the next reflect task.
        if explicit_report.pop("completed", False):
            explicit_report["facts_marked_processed"] = self._db.mark_facts_processed(
                processing_target="entity_claim",
                fact_ids=[fact.get("id") for fact in explicit_facts],
            )
        else:
            explicit_report["facts_marked_processed"] = 0

        episodes = self._db.get_unprocessed_episodes_for_entity_claim_induction(
            limit=max(1, min(limit, 24)),
        )
        inductive_report = self._update_inductive_entity_claims_from_episodes(episodes)
        if inductive_report.pop("completed", False):
            inductive_report["episodes_marked_processed"] = (
                self._db.mark_episodes_processed_for_entity_claim_induction(
                    [episode.get("id") for episode in episodes]
                )
            )
        else:
            inductive_report["episodes_marked_processed"] = 0
        self._log_info("memory_reflect", "entity_claim_update_finish", {
            "explicit": explicit_report,
            "inductive": inductive_report,
        })
        return {"explicit": explicit_report, "inductive": inductive_report}

    def _claim_fact_prompt_view(self, fact: Dict[str, Any]) -> Dict[str, Any]:
        metadata = fact.get("metadata") if isinstance(fact.get("metadata"), dict) else {}
        return {
            "fact_id": fact.get("id"),
            "summary": fact.get("summary") or "",
            "fact_kind": fact.get("fact_kind") or "",
            "entities": fact.get("entities") or [],
            "primary_entity": fact.get("primary_entity") or metadata.get("primary_entity"),
            "keywords": fact.get("keywords") or [],
            "entity_claim_signal": (
                fact.get("entity_claim_signal")
                or metadata.get("entity_claim_signal")
                or []
            ),
            "event_time": fact.get("event_time_key") or "",
            "dialogue_time": fact.get("dialogue_time_key") or "",
            "episode_id": fact.get("episode_id"),
        }

    def _claim_entity_name_to_id(self, facts: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        names = [
            name
            for fact in facts
            for name in self._normalize_entity_names(fact.get("entities") or [], limit=24)
        ]
        return self._db.add_entity_names(names)

    @staticmethod
    def _entity_claim_origin_priority(origin: Any) -> int:
        """Origin is a hard reconciliation precedence, not a soft score."""
        return {"explicit": 3, "inductive": 2, "derived": 1}.get(
            str(origin or "").strip().lower(),
            0,
        )

    @staticmethod
    def _entity_claim_storage_payload(candidate: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in candidate.items()
            if key not in {
                "evidence_fact_ids", "support_fact_ids", "counterexample_fact_ids",
            }
        }

    @staticmethod
    def _entity_claim_evidence_ids(
        candidate: Dict[str, Any],
    ) -> Tuple[List[int], List[int]]:
        support_ids = candidate.get("support_fact_ids")
        if support_ids is None:
            support_ids = candidate.get("evidence_fact_ids") or []
        return (
            [int(value) for value in support_ids if str(value).strip().isdigit()],
            [
                int(value)
                for value in candidate.get("counterexample_fact_ids") or []
                if str(value).strip().isdigit()
            ],
        )

    def _retrieve_related_entity_claims(
        self,
        candidate: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Bound relation candidates to one entity and compatible claim type."""
        rows = self._db.get_entity_claims(
            subject_entity_id=int(candidate["subject_entity_id"]),
            claim_type=str(candidate["claim_type"]),
            statuses=["active", "candidate", "weakened"],
            limit=80,
        )
        predicate = str(candidate.get("predicate") or "")
        normalized_value = str(candidate.get("normalized_value") or "")
        rows.sort(
            key=lambda row: (
                str(row.get("predicate") or "") != predicate,
                str(row.get("normalized_value") or "") != normalized_value,
                -self._entity_claim_origin_priority(row.get("claim_origin")),
                -float(row.get("confidence") or 0.0),
            )
        )
        return rows[:24]

    def _finalize_entity_claim_relation_decision(
        self,
        candidate: Dict[str, Any],
        target: Optional[Dict[str, Any]],
        semantic_decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Translate text-only relation into a safe origin-aware write plan."""
        semantic_relation = str(
            semantic_decision.get("semantic_relation")
            or semantic_decision.get("relation")
            or "unrelated"
        )
        decision = {
            **semantic_decision,
            "semantic_relation": semantic_relation,
            "merge_into_target": False,
            "candidate_status": str(candidate.get("status") or "candidate"),
            "target_status": "",
            "target_evidence_role": "",
        }
        if not target:
            return decision
        candidate_priority = self._entity_claim_origin_priority(
            candidate.get("claim_origin")
        )
        target_priority = self._entity_claim_origin_priority(
            target.get("claim_origin")
        )
        if semantic_relation == "duplicate":
            if candidate_priority <= target_priority:
                decision["merge_into_target"] = True
                decision["target_evidence_role"] = (
                    "context" if candidate_priority < target_priority else "support"
                )
            else:
                # Keep a direct explicit assertion as its own, stronger claim,
                # while using it to support the older inductive conclusion.
                decision["target_evidence_role"] = "support"
            return decision
        if semantic_relation in {"supports", "refines"}:
            decision["target_evidence_role"] = (
                "context" if candidate_priority < target_priority else "support"
            )
            return decision
        if semantic_relation not in {"contradicts", "supersedes"}:
            return decision
        if candidate_priority < target_priority:
            # An inferred pattern can coexist as a tentative competing claim,
            # but it never changes an explicit claim's state.
            decision["candidate_status"] = "candidate"
            return decision
        decision["target_evidence_role"] = "counterexample"
        decision["target_status"] = (
            "superseded" if semantic_relation == "supersedes" else "weakened"
        )
        return decision

    def _classify_entity_claim_relations(
        self,
        candidates: Sequence[Dict[str, Any]],
        related_by_index: Dict[int, List[Dict[str, Any]]],
    ) -> List[List[Dict[str, Any]]]:
        if not any(related_by_index.values()):
            return [[] for _candidate in candidates]
        language = self._resolve_prompt_language_from_text(
            "\n".join(str(candidate.get("claim_text") or "") for candidate in candidates)
        )
        template = (
            ENTITY_CLAIM_RECONCILIATION_PROMPT_EN
            if language == "en" else ENTITY_CLAIM_RECONCILIATION_PROMPT_ZH
        )
        prompt_candidates = [
            {
                "candidate_claim_index": index,
                "claim_text": candidate.get("claim_text") or "",
            }
            for index, candidate in enumerate(candidates)
        ]
        existing_by_id = {
            int(claim["id"]): claim
            for claims in related_by_index.values()
            for claim in claims
            if str(claim.get("id") or "").strip().isdigit()
        }
        raw = self._call_llm(
            template.replace("{candidate_claims}", json.dumps(
                prompt_candidates, ensure_ascii=False, indent=2,
            )).replace("{existing_claims}", json.dumps(
                [
                    {"id": claim.get("id"), "claim_text": claim.get("claim_text") or ""}
                    for claim in existing_by_id.values()
                ],
                ensure_ascii=False, indent=2,
            ))
        )
        parsed = self._parse_json_object_from_llm_text(raw or "")
        if not parsed or not isinstance(parsed.get("decisions"), list):
            return [[] for _candidate in candidates]
        else:
            allowed_relations = {
                "duplicate", "supports", "contradicts", "refines", "supersedes",
            }
            semantic_resolved: List[List[Dict[str, Any]]] = [
                [] for _candidate in candidates
            ]
            for raw_decision in parsed["decisions"]:
                if not isinstance(raw_decision, dict):
                    continue
                try:
                    index = int(raw_decision.get("candidate_claim_index"))
                except (TypeError, ValueError):
                    continue
                if index < 0 or index >= len(candidates):
                    continue
                raw_relations = raw_decision.get("relations")
                if not isinstance(raw_relations, list):
                    continue
                allowed_ids = {
                    int(claim["id"])
                    for claim in related_by_index.get(index, [])
                    if str(claim.get("id") or "").strip().isdigit()
                }
                parsed_by_id: Dict[int, Dict[str, Any]] = {}
                for raw_relation in raw_relations:
                    if not isinstance(raw_relation, dict):
                        continue
                    try:
                        target_id = int(raw_relation.get("existing_claim_id"))
                    except (TypeError, ValueError):
                        continue
                    relation = str(
                        raw_relation.get("semantic_relation") or ""
                    ).strip().lower()
                    if target_id not in allowed_ids or relation not in allowed_relations:
                        continue
                    parsed_by_id[target_id] = {
                        "existing_claim_id": target_id,
                        "semantic_relation": relation,
                        "confidence": self._clamp_float(
                            raw_relation.get("confidence"), 0.0, 1.0, 0.7,
                        ),
                        "reason": _compact_whitespace(
                            raw_relation.get("reason") or ""
                        )[:240],
                    }
                semantic_resolved[index] = list(parsed_by_id.values())
        finalized: List[List[Dict[str, Any]]] = []
        for index, candidate in enumerate(candidates):
            candidate_decisions: List[Dict[str, Any]] = []
            for semantic_decision in semantic_resolved[index]:
                target_id = semantic_decision.get("existing_claim_id")
                target = next(
                    (
                        claim for claim in related_by_index.get(index, [])
                        if int(claim.get("id") or 0) == int(target_id or 0)
                    ),
                    None,
                )
                if target:
                    candidate_decisions.append(
                        self._finalize_entity_claim_relation_decision(
                            candidate, target, semantic_decision,
                        )
                    )
            finalized.append(candidate_decisions)
        return finalized

    def _write_entity_claim_evidence(
        self,
        *,
        claim_id: int,
        support_ids: Sequence[int],
        counterexample_ids: Sequence[int],
        facts_by_id: Dict[int, Dict[str, Any]],
        confidence: float,
        support_role: str = "support",
    ) -> None:
        evidence: List[Dict[str, Any]] = []
        for role, fact_ids in (
            (support_role, support_ids),
            ("counterexample", counterexample_ids),
        ):
            for fact_id in fact_ids:
                fact = facts_by_id.get(int(fact_id))
                if not fact:
                    continue
                evidence.append({
                    "claim_id": claim_id,
                    "evidence_type": "fact",
                    "evidence_id": int(fact_id),
                    "role": role,
                    "weight": confidence,
                    "observed_at": fact.get("event_time_key")
                    or fact.get("dialogue_time_key") or "",
                })
        self._db.upsert_entity_claim_evidence(evidence)

    @staticmethod
    def _entity_claim_transition_effective_at(
        candidate: Dict[str, Any],
        *,
        support_ids: Sequence[int],
        counterexample_ids: Sequence[int],
        facts_by_id: Dict[int, Dict[str, Any]],
    ) -> str:
        """Use the candidate's own temporal assertion, then its newest fact."""
        valid_from = _compact_whitespace(candidate.get("valid_from") or "")
        if valid_from:
            return valid_from
        observed_at = [
            _compact_whitespace(
                fact.get("event_time_key") or fact.get("dialogue_time_key") or ""
            )
            for fact_id in [*support_ids, *counterexample_ids]
            for fact in [facts_by_id.get(int(fact_id))]
            if fact
        ]
        return max((value for value in observed_at if value), default="")

    def _transition_entity_claim_target(
        self,
        *,
        target: Dict[str, Any],
        trigger_claim_id: int,
        candidate: Dict[str, Any],
        decision: Dict[str, Any],
        effective_at: str,
    ) -> bool:
        """Persist an origin-aware claim-state transition with its audit trail."""
        target_status = str(decision.get("target_status") or "")
        if not target_status:
            return False
        candidate_origin = str(candidate.get("claim_origin") or "")
        target_origin = str(target.get("claim_origin") or "")
        candidate_priority = self._entity_claim_origin_priority(candidate_origin)
        target_priority = self._entity_claim_origin_priority(target_origin)
        details = {
            "trigger_claim_snapshot": {
                "claim_text": candidate.get("claim_text") or "",
                "claim_origin": candidate_origin,
                "claim_type": candidate.get("claim_type") or "",
                "confidence": candidate.get("confidence"),
            },
            "target_claim_snapshot": {
                "claim_text": target.get("claim_text") or "",
                "claim_origin": target_origin,
                "claim_type": target.get("claim_type") or "",
                "confidence": target.get("confidence"),
            },
            "origin_priorities": {
                "trigger": candidate_priority,
                "target": target_priority,
            },
        }
        return self._db.transition_entity_claim_status(
            target_claim_id=int(target["id"]),
            new_status=target_status,
            trigger_claim_id=trigger_claim_id,
            semantic_relation=str(decision.get("semantic_relation") or ""),
            effective_at=effective_at,
            decision_source="entity_claim_reconciliation_v1:llm_semantics+origin_policy",
            semantic_confidence=self._clamp_float(
                decision.get("confidence"), 0.0, 1.0, 0.0,
            ),
            semantic_reason=_compact_whitespace(decision.get("reason") or "")[:240],
            policy_reason=(
                f"origin_precedence: trigger={candidate_origin}({candidate_priority}), "
                f"target={target_origin}({target_priority}), "
                f"target_status={target_status}"
            ),
            details=details,
        )

    def _reconcile_entity_claims(
        self,
        candidates: Sequence[Dict[str, Any]],
        *,
        facts_by_id: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Resolve new candidate claims before they become durable claims."""
        unique_candidates: List[Dict[str, Any]] = []
        seen: set[Tuple[Any, ...]] = set()
        for candidate in candidates:
            key = (
                candidate.get("subject_entity_id"), candidate.get("claim_type"),
                candidate.get("predicate"), candidate.get("object_entity_id") or 0,
                candidate.get("normalized_value"), candidate.get("claim_origin"),
            )
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(candidate)
        related_by_index = {
            index: self._retrieve_related_entity_claims(candidate)
            for index, candidate in enumerate(unique_candidates)
        }
        decisions = self._classify_entity_claim_relations(
            unique_candidates, related_by_index,
        )
        applied: List[Dict[str, Any]] = []
        for index, candidate in enumerate(unique_candidates):
            candidate_decisions = decisions[index]
            candidate_origin = str(candidate.get("claim_origin") or "")
            support_ids, counterexample_ids = self._entity_claim_evidence_ids(candidate)
            effective_at = self._entity_claim_transition_effective_at(
                candidate,
                support_ids=support_ids,
                counterexample_ids=counterexample_ids,
                facts_by_id=facts_by_id,
            )
            candidate_status = str(candidate.get("status") or "candidate")
            if any(
                str(decision.get("candidate_status") or "") == "candidate"
                for decision in candidate_decisions
            ):
                candidate_status = "candidate"

            def target_for(decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                target_id = decision.get("existing_claim_id")
                return next(
                    (
                        item for item in related_by_index[index]
                        if int(item.get("id") or 0) == int(target_id or 0)
                    ),
                    None,
                )

            merge_decisions = [
                decision for decision in candidate_decisions
                if decision.get("merge_into_target") and target_for(decision)
            ]
            if merge_decisions:
                primary_merge = max(
                    merge_decisions,
                    key=lambda decision: self._entity_claim_origin_priority(
                        (target_for(decision) or {}).get("claim_origin")
                    ),
                )
                primary_target = target_for(primary_merge)
                assert primary_target is not None
                claim_id = int(primary_target["id"])
                for decision in candidate_decisions:
                    target = target_for(decision)
                    if not target:
                        continue
                    target_status = str(decision.get("target_status") or "")
                    target_evidence_role = str(
                        decision.get("target_evidence_role") or ""
                    )
                    if target_status:
                        self._transition_entity_claim_target(
                            target=target,
                            trigger_claim_id=claim_id,
                            candidate=candidate,
                            decision=decision,
                            effective_at=effective_at,
                        )
                    if target_evidence_role:
                        self._write_entity_claim_evidence(
                            claim_id=int(target["id"]),
                            support_ids=support_ids,
                            counterexample_ids=counterexample_ids,
                            facts_by_id=facts_by_id,
                            confidence=float(candidate.get("confidence") or 0.0),
                            support_role=target_evidence_role,
                        )
                applied.append({
                    "claim": candidate, "claim_id": claim_id, "created": False,
                    "effective_origin": primary_target.get("claim_origin") or "",
                    "relations": candidate_decisions, "merged": True,
                })
                continue

            storage_payload = self._entity_claim_storage_payload(candidate)
            storage_payload["status"] = candidate_status
            claim_id, created = self._db.upsert_entity_claim(**storage_payload)
            self._write_entity_claim_evidence(
                claim_id=claim_id,
                support_ids=support_ids,
                counterexample_ids=counterexample_ids,
                facts_by_id=facts_by_id,
                confidence=float(candidate.get("confidence") or 0.0),
            )
            for decision in candidate_decisions:
                target = target_for(decision)
                if not target:
                    continue
                target_status = str(decision.get("target_status") or "")
                target_evidence_role = str(
                    decision.get("target_evidence_role") or ""
                )
                if target_status:
                    self._transition_entity_claim_target(
                        target=target,
                        trigger_claim_id=claim_id,
                        candidate=candidate,
                        decision=decision,
                        effective_at=effective_at,
                    )
                if target_evidence_role:
                    self._write_entity_claim_evidence(
                        claim_id=int(target["id"]),
                        support_ids=support_ids,
                        counterexample_ids=counterexample_ids,
                        facts_by_id=facts_by_id,
                        confidence=float(candidate.get("confidence") or 0.0),
                        support_role=target_evidence_role,
                    )
            applied.append({
                "claim": candidate, "claim_id": claim_id, "created": created,
                "effective_origin": candidate_origin,
                "relations": candidate_decisions,
                "merged": False,
            })
        self._log_info("memory_reflect", "entity_claim_reconciled", {
            "candidate_count": len(unique_candidates),
            "applied_count": len(applied),
            "relations": [
                {
                    "claim_id": item["claim_id"], "origin": item["claim"].get("claim_origin"),
                    "relations": [
                        {
                            "existing_claim_id": decision.get("existing_claim_id"),
                            "semantic_relation": decision.get("semantic_relation"),
                        }
                        for decision in item["relations"]
                    ],
                    "merged": item["merged"],
                }
                for item in applied
            ],
        })
        return applied

    def _update_explicit_entity_claims_from_facts(
        self,
        facts: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "enabled": 1, "fact_count": len(facts), "claim_count": 0,
            "updated": 0, "created": 0, "completed": True,
        }
        if not facts:
            return report
        prompt_language = self._resolve_prompt_language_from_text(
            "\n".join(str(fact.get("summary") or "") for fact in facts[:20])
        )
        template = (
            EXPLICIT_ENTITY_CLAIM_EXTRACTION_PROMPT_EN
            if prompt_language == "en" else EXPLICIT_ENTITY_CLAIM_EXTRACTION_PROMPT_ZH
        )
        raw = self._call_llm(template.replace(
            "{facts}", json.dumps(
                [self._claim_fact_prompt_view(fact) for fact in facts[:40]],
                ensure_ascii=False, indent=2,
            ),
        ))
        parsed = self._parse_json_object_from_llm_text(raw or "")
        if parsed is None or not isinstance(parsed.get("claims"), list):
            report["completed"] = False
            report["error"] = "invalid_llm_claim_response"
            return report
        entity_ids = self._claim_entity_name_to_id(facts)
        facts_by_id = {
            int(fact["id"]): fact for fact in facts
            if str(fact.get("id") or "").strip().isdigit()
        }
        candidates: List[Dict[str, Any]] = []
        for raw_claim in parsed["claims"][:32]:
            claim = self._normalize_explicit_entity_claim(
                raw_claim, facts_by_id=facts_by_id, entity_ids=entity_ids,
            )
            if not claim:
                continue
            candidates.append(claim)
        applied = self._reconcile_entity_claims(candidates, facts_by_id=facts_by_id)
        report["updated"] = len(applied)
        report["created"] = sum(int(item["created"]) for item in applied)
        report["merged"] = sum(int(item["merged"]) for item in applied)
        report["claim_count"] = len(candidates)
        return report

    def _normalize_explicit_entity_claim(
        self,
        raw: Any,
        *,
        facts_by_id: Dict[int, Dict[str, Any]],
        entity_ids: Dict[str, int],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        claim_type = str(raw.get("claim_type") or "").strip().lower()
        if claim_type not in self._entity_claim_types() - {"behavior_pattern"}:
            return None
        subject = _compact_whitespace(raw.get("subject_entity") or "")
        subject_id = entity_ids.get(subject)
        predicate = re.sub(r"[^a-z0-9_]+", "_", str(raw.get("predicate") or "").lower()).strip("_")
        if not subject_id or not predicate:
            return None
        evidence_ids = list(dict.fromkeys(
            int(value) for value in (raw.get("evidence_fact_ids") or [])
            if str(value).strip().isdigit() and int(value) in facts_by_id
        ))[:12]
        if not evidence_ids or not all(
            subject in (facts_by_id[fact_id].get("entities") or [])
            for fact_id in evidence_ids
        ):
            return None
        object_name = _compact_whitespace(raw.get("object_entity") or "")
        object_id = entity_ids.get(object_name) if object_name else None
        normalized_value = _compact_whitespace(raw.get("normalized_value") or "")[:160]
        if not object_id and not normalized_value:
            return None
        confidence = self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.7)
        claim_text = self._normalize_entity_claim_text(
            raw.get("claim_text"),
            subject=subject,
            predicate=predicate,
            object_name=object_name,
            normalized_value=normalized_value,
        )
        return {
            "subject_entity_id": subject_id,
            "predicate": predicate,
            "object_entity_id": object_id,
            "normalized_value": normalized_value,
            "claim_text": claim_text,
            "claim_type": claim_type,
            "claim_origin": "explicit",
            "status": "active" if confidence >= self._entity_claim_explicit_min_confidence else "candidate",
            "confidence": confidence,
            "valid_from": _compact_whitespace(raw.get("valid_from") or ""),
            "valid_to": _compact_whitespace(raw.get("valid_to") or ""),
            "extractor_version": "entity_claim_explicit_v1",
            "prompt_version": "v1",
            "metadata": {"source_fact_count": len(evidence_ids)},
            "evidence_fact_ids": evidence_ids,
        }

    def _update_inductive_entity_claims_from_episodes(
        self,
        episodes: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "enabled": 1, "episode_count": len(episodes), "candidate_count": 0,
            "updated": 0, "created": 0, "completed": True,
        }
        if not episodes:
            return report
        seed_facts = self._db.memory_facts_by_episode_ids(
            [episode.get("id") for episode in episodes], limit=360,
        )
        groups: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
        for fact in seed_facts:
            for signal in self._entity_claim_signals_from_fact(fact):
                claim_type_hint = str(signal.get("claim_type_hint") or "").lower()
                if claim_type_hint not in {"preference", "behavior_pattern"}:
                    continue
                if str(signal.get("signal_kind") or "") not in {
                    "explicit_assertion", "pattern_observation", "counterexample",
                }:
                    continue
                names = self._entities_for_entity_claim_signal(signal, fact)
                mapping = self._db.add_entity_names(names)
                if not names or names[0] not in mapping:
                    continue
                claim_anchor = _compact_whitespace(signal.get("claim_anchor") or "")
                if not claim_anchor:
                    continue
                key = (
                    int(mapping[names[0]]),
                    claim_type_hint,
                    self._generate_topic_name_key(claim_anchor),
                )
                groups.setdefault(key, {
                    "subject": names[0], "subject_entity_id": int(mapping[names[0]]),
                    "claim_type_hint": claim_type_hint,
                    "claim_anchor": claim_anchor,
                })
        report["candidate_count"] = len(groups)
        for group in list(groups.values())[:12]:
            all_entity_facts = self._db.memory_episode_facts_for_entity_id(
                group["subject_entity_id"], limit=240,
            )
            evidence_facts = [
                fact for fact in all_entity_facts
                if any(
                    str(signal.get("claim_type_hint") or "").lower()
                    == group["claim_type_hint"]
                    and self._generate_topic_name_key(signal.get("claim_anchor") or "")
                    == self._generate_topic_name_key(group["claim_anchor"])
                    and group["subject"]
                    in self._entities_for_entity_claim_signal(signal, fact)
                    for signal in self._entity_claim_signals_from_fact(fact)
                )
            ]
            distinct_episodes = {
                int(fact["episode_id"]) for fact in evidence_facts
                if str(fact.get("episode_id") or "").strip().isdigit()
            }
            time_windows = {
                str(fact.get("event_time_key") or fact.get("dialogue_time_key") or "")[:10]
                for fact in evidence_facts
                if str(fact.get("event_time_key") or fact.get("dialogue_time_key") or "")
            }
            if (
                len(distinct_episodes) < self._entity_claim_induction_min_episodes
                or len(time_windows) < self._entity_claim_induction_min_time_windows
            ):
                continue
            outcome = self._extract_inductive_entity_claims(
                group=group, facts=evidence_facts,
            )
            if outcome is None:
                # Do not mark the episode cursor on a transport/format error:
                # the same completed evidence must remain eligible for retry.
                report["completed"] = False
                report["error"] = "invalid_llm_induction_response"
                return report
            by_id = {int(fact["id"]): fact for fact in evidence_facts}
            applied = self._reconcile_entity_claims(outcome, facts_by_id=by_id)
            for item in applied:
                claim = item["claim"]
                if item["effective_origin"] != "inductive":
                    continue
                support_ids, counterexample_ids = self._entity_claim_evidence_ids(claim)
                if not support_ids:
                    continue
                claim_id = int(item["claim_id"])
                self._db.upsert_entity_claim_induction(
                    claim_id=claim_id,
                    condition_text=str(claim.get("metadata", {}).get("condition_text") or ""),
                    behavior_or_outcome_text=str(claim.get("metadata", {}).get("behavior_or_outcome_text") or ""),
                    support_count=len(support_ids),
                    counterexample_count=len(counterexample_ids),
                    first_observed_at=min(
                        (by_id[item].get("event_time_key") or by_id[item].get("dialogue_time_key") or "")
                        for item in support_ids
                    ),
                    last_observed_at=max(
                        (by_id[item].get("event_time_key") or by_id[item].get("dialogue_time_key") or "")
                        for item in support_ids
                    ),
                )
            report["updated"] += len(applied)
            report["created"] += sum(int(item["created"]) for item in applied)
        return report

    def _extract_inductive_entity_claims(
        self,
        *,
        group: Dict[str, Any],
        facts: Sequence[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        language = self._resolve_prompt_language_from_text(
            "\n".join(str(fact.get("summary") or "") for fact in facts[:16])
        )
        template = (
            INDUCTIVE_ENTITY_CLAIM_EXTRACTION_PROMPT_EN
            if language == "en" else INDUCTIVE_ENTITY_CLAIM_EXTRACTION_PROMPT_ZH
        )
        raw = self._call_llm(
            template.replace("{induction_target}", json.dumps({
                "subject_entity": group["subject"],
                "claim_type_hint": group["claim_type_hint"],
                "claim_anchor": group["claim_anchor"],
            }, ensure_ascii=False, indent=2)).replace("{facts}", json.dumps([
                self._claim_fact_prompt_view(fact) for fact in facts[:32]
            ], ensure_ascii=False, indent=2))
        )
        parsed = self._parse_json_object_from_llm_text(raw or "")
        if parsed is None or not isinstance(parsed.get("claims"), list):
            return None
        facts_by_id = {
            int(fact["id"]): fact for fact in facts
            if str(fact.get("id") or "").strip().isdigit()
        }
        result: List[Dict[str, Any]] = []
        for raw_claim in parsed["claims"][:4]:
            claim = self._normalize_inductive_entity_claim(
                raw_claim, group=group, facts_by_id=facts_by_id,
            )
            if claim:
                result.append(claim)
        return result

    def _normalize_inductive_entity_claim(
        self,
        raw: Any,
        *,
        group: Dict[str, Any],
        facts_by_id: Dict[int, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        claim_type = str(raw.get("claim_type") or "").strip().lower()
        if claim_type not in {"preference", "behavior_pattern"}:
            return None
        if _compact_whitespace(raw.get("subject_entity") or "") != group["subject"]:
            return None
        predicate = re.sub(r"[^a-z0-9_]+", "_", str(raw.get("predicate") or "").lower()).strip("_")
        if predicate not in {"prefers", "dislikes", "usually_does", "avoids", "has_routine"}:
            return None
        support_ids = list(dict.fromkeys(
            int(value) for value in (raw.get("support_fact_ids") or [])
            if str(value).strip().isdigit() and int(value) in facts_by_id
        ))[:24]
        counterexample_ids = list(dict.fromkeys(
            int(value) for value in (raw.get("counterexample_fact_ids") or [])
            if str(value).strip().isdigit() and int(value) in facts_by_id
        ))[:24]
        episode_ids = {
            int(facts_by_id[fact_id]["episode_id"]) for fact_id in support_ids
            if str(facts_by_id[fact_id].get("episode_id") or "").strip().isdigit()
        }
        windows = {
            str(facts_by_id[fact_id].get("event_time_key") or facts_by_id[fact_id].get("dialogue_time_key") or "")[:10]
            for fact_id in support_ids
        } - {""}
        if (
            len(episode_ids) < self._entity_claim_induction_min_episodes
            or len(windows) < self._entity_claim_induction_min_time_windows
        ):
            return None
        normalized_value = _compact_whitespace(raw.get("normalized_value") or "")[:160]
        if not normalized_value:
            return None
        confidence = self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.7)
        claim_text = self._normalize_entity_claim_text(
            raw.get("claim_text") or raw.get("behavior_or_outcome_text"),
            subject=group["subject"],
            predicate=predicate,
            normalized_value=normalized_value,
        )
        return {
            "subject_entity_id": group["subject_entity_id"], "predicate": predicate,
            "normalized_value": normalized_value,
            "claim_text": claim_text,
            "claim_type": claim_type, "claim_origin": "inductive",
            "status": "weakened" if counterexample_ids else "active",
            "confidence": confidence,
            "extractor_version": "entity_claim_induction_v1", "prompt_version": "v1",
            "metadata": {
                "condition_text": _compact_whitespace(raw.get("condition_text") or ""),
                "behavior_or_outcome_text": _compact_whitespace(raw.get("behavior_or_outcome_text") or ""),
                "claim_anchor": group["claim_anchor"],
            },
            "support_fact_ids": support_ids,
            "counterexample_fact_ids": counterexample_ids,
        }

    @staticmethod
    def _normalize_entity_claim_text(
        value: Any,
        *,
        subject: str,
        predicate: str,
        object_name: str = "",
        normalized_value: str = "",
    ) -> str:
        """Keep a readable proposition separate from the compact merge key."""
        text = _compact_whitespace(value or "")[:480]
        if text:
            return text
        target = _compact_whitespace(object_name or normalized_value)
        return _compact_whitespace(f"{subject} {predicate} {target}")[:480]

    def _log_reflect_facts_loaded(
        self,
        processing_target: str,
        facts: List[Dict[str, Any]],
        limit: int,
        reference_timestamp: Any,
    ) -> None:
        """Log the independent fact batch consumed by one reflect projection."""
        source_counts = Counter(
            str(fact.get("source_type")) for fact in facts
        )
        self._log_info("memory_reflect", "facts_loaded", {
            "processing_target": processing_target,
            "fact_count": len(facts),
            "fact_ids": [fact.get("id") for fact in facts],
            "source_counts": dict(source_counts),
            "limit": limit,
            "reference_timestamp": reference_timestamp,
            "time_start": facts[0].get("dialogue_time_key") if facts else "",
            "time_end": facts[-1].get("dialogue_time_key") if facts else "",
        })

    @staticmethod
    def _generate_topic_name_key(value: Any) -> str:
        text = _compact_whitespace(value).lower()
        text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", text)
        return text or "general"

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

    def _recall_calculate_search_terms_overlap_with_topic_values(
        self,
        query_terms: Sequence[str],
        topic_values: Sequence[str],
        *,
        allow_substring: bool = True,
        minimum_pair_score: float = 0.5,
    ) -> Dict[str, Any]:
        """Measure how many distinct query-side topic terms a candidate covers.

        Unlike ``_topic_name_best_pair_similarity``, this is intentionally
        asymmetric: a candidate with many aliases cannot inflate its score.
        Each distinct query term contributes at most one match when any topic
        value provides sufficiently strong exact, substring, or token-overlap
        evidence.
        """
        normalized_terms: List[Tuple[str, str]] = []
        seen_term_keys: set[str] = set()
        for value in query_terms or ():
            term = self._recall_stage1_clean_anchor(value)
            term_key = self._generate_topic_name_key(term) if term else ""
            if not term or not term_key or term_key in seen_term_keys:
                continue
            seen_term_keys.add(term_key)
            normalized_terms.append((term, term_key))
        # Search-mode tokenization can emit both a compound term and its
        # nested fragments. Only the longest form should contribute to
        # query-side coverage, otherwise one topic phrase is counted several
        # times as independent evidence.
        normalized_terms = [
            (term, term_key)
            for term, term_key in normalized_terms
            if not any(
                term_key != other_key
                and len(term_key) < len(other_key)
                and term_key in other_key
                for _other_term, other_key in normalized_terms
            )
        ]

        normalized_topics: List[Tuple[str, str]] = []
        seen_topic_keys: set[str] = set()
        for value in topic_values or ():
            topic = self._recall_stage1_clean_anchor(value)
            topic_key = self._generate_topic_name_key(topic) if topic else ""
            if not topic or not topic_key or topic_key in seen_topic_keys:
                continue
            seen_topic_keys.add(topic_key)
            normalized_topics.append((topic, topic_key))

        matched_terms: List[str] = []
        matched_topic_values: List[str] = []
        best_pair_score = 0.0
        required_pair_score = self._clamp_float(
            minimum_pair_score,
            0.0,
            1.0,
            0.5,
        )
        for term, _term_key in normalized_terms:
            best_topic = ""
            best_term_score = 0.0
            for topic, _topic_key in normalized_topics:
                pair_score = self._topic_name_best_pair_similarity(
                    [term],
                    [topic],
                    allow_substring=allow_substring,
                )
                if pair_score > best_term_score:
                    best_term_score = pair_score
                    best_topic = topic
            best_pair_score = max(best_pair_score, best_term_score)
            if best_term_score < required_pair_score:
                continue
            matched_terms.append(term)
            if best_topic and best_topic not in matched_topic_values:
                matched_topic_values.append(best_topic)

        term_count = len(normalized_terms)
        matched_term_count = len(matched_terms)
        return {
            "matched_term_count": matched_term_count,
            "term_count": term_count,
            "coverage": round(matched_term_count / term_count, 4)
            if term_count
            else 0.0,
            "matched_terms": matched_terms,
            "matched_topic_values": matched_topic_values,
            "best_pair_score": round(best_pair_score, 4),
        }

    def _recall_calculate_search_terms_overlap_with_candidate_topics(
        self,
        candidate: Dict[str, Any],
        query_terms: Sequence[str],
        *,
        allow_substring: bool = True,
        minimum_pair_score: float = 0.5,
    ) -> Dict[str, Any]:
        """Extract one candidate's canonical topic values and score overlap."""
        raw = (
            candidate.get("_hydrated")
            if isinstance(candidate.get("_hydrated"), dict)
            else {}
        )
        metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        topic_values: List[str] = []

        def extend_values(value: Any) -> None:
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    extend_values(item)
                return
            topic = self._recall_stage1_clean_anchor(value)
            if topic and topic not in topic_values:
                topic_values.append(topic)

        index_level = str(candidate.get("index_level") or "").strip().lower()
        if index_level == "fact":
            extend_values(raw.get("fact_root_topic"))
            extend_values(raw.get("fact_aspect_topic"))
        elif index_level == "state":
            extend_values(raw.get("canonical_name"))
            extend_values(metadata.get("attribute_name_aliases"))
            extend_values(metadata.get("canonical_topics"))
        else:
            extend_values(raw.get("canonical_name"))
            extend_values(metadata.get("canonical_topics"))
            extend_values(raw.get("canonical_topics"))

        overlap = self._recall_calculate_search_terms_overlap_with_topic_values(
            query_terms,
            topic_values,
            allow_substring=allow_substring,
            minimum_pair_score=minimum_pair_score,
        )
        return {
            **overlap,
            "topic_values": topic_values,
        }

    def _recall_calculate_search_terms_overlap_with_candidate_keywords(
        self,
        candidate: Dict[str, Any],
        query_terms: Sequence[str],
        *,
        allow_substring: bool = True,
        minimum_pair_score: float = 0.5,
    ) -> Dict[str, Any]:
        """Measure query-term overlap with one fact's extracted keywords.

        Keywords are a supplementary, fact-only signal.  They intentionally
        remain separate from canonical topic matching so callers can use them
        for ranking without treating them as a topic strong anchor.
        """
        if str(candidate.get("index_level") or "").strip().lower() != "fact":
            overlap = self._recall_calculate_search_terms_overlap_with_topic_values(
                query_terms,
                [],
                allow_substring=allow_substring,
                minimum_pair_score=minimum_pair_score,
            )
            return {
                key: value
                for key, value in overlap.items()
                if key != "matched_topic_values"
            } | {
                "matched_keyword_values": list(
                    overlap.get("matched_topic_values") or []
                ),
                "keyword_values": [],
            }

        raw = (
            candidate.get("_hydrated")
            if isinstance(candidate.get("_hydrated"), dict)
            else {}
        )
        raw_keywords = raw.get("keywords") or candidate.get("keywords") or ""
        keyword_values: List[str] = []
        seen_keyword_keys: set[str] = set()

        def add_keyword(value: Any) -> None:
            keyword = self._recall_stage1_clean_anchor(value)
            keyword_key = self._generate_topic_name_key(keyword) if keyword else ""
            if not keyword or not keyword_key or keyword_key in seen_keyword_keys:
                return
            seen_keyword_keys.add(keyword_key)
            keyword_values.append(keyword)

        keywords_are_structured = isinstance(raw_keywords, (list, tuple, set))
        raw_values = (
            raw_keywords
            if keywords_are_structured
            else re.split(r"[,，;；\n]+", str(raw_keywords))
        )
        for raw_value in raw_values:
            add_keyword(raw_value)
            # Legacy facts persist a whitespace-joined string. New facts use
            # a structured list so multi-word keywords retain their boundary.
            if not keywords_are_structured and isinstance(raw_value, str):
                for token in raw_value.split():
                    add_keyword(token)

        overlap = self._recall_calculate_search_terms_overlap_with_topic_values(
            query_terms,
            keyword_values,
            allow_substring=allow_substring,
            minimum_pair_score=minimum_pair_score,
        )
        return {
            key: value
            for key, value in overlap.items()
            if key != "matched_topic_values"
        } | {
            "matched_keyword_values": list(
                overlap.get("matched_topic_values") or []
            ),
            "keyword_values": keyword_values,
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

    def _entity_claim_signals_from_fact(self, fact: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Read claim-level extraction signals from a stored fact."""
        metadata = fact.get("metadata") if isinstance(fact.get("metadata"), dict) else {}
        raw = fact.get("entity_claim_signal") or metadata.get("entity_claim_signal")
        fallback_entity = fact.get("primary_entity")
        return self._normalize_entity_claim_signal(raw, fallback_entity=fallback_entity)

    def _entities_for_entity_claim_signal(
        self,
        signal: Dict[str, Any],
        fact: Dict[str, Any],
    ) -> List[str]:
        entity = signal.get("entity") or signal.get("primary_entity")
        if isinstance(entity, dict):
            name = _compact_whitespace(entity.get("name") or entity.get("text") or "")
        else:
            name = _compact_whitespace(entity)
        if name:
            return [name]
        return self._entities_for_entity_claim_fact(fact)

    def _entities_for_entity_claim_fact(
        self,
        fact: Dict[str, Any],
    ) -> List[str]:
        metadata = fact.get("metadata") if isinstance(fact.get("metadata"), dict) else {}
        primary_entity = fact.get("primary_entity")
        if isinstance(primary_entity, dict):
            primary_name = _compact_whitespace(
                primary_entity.get("name") or primary_entity.get("text") or ""
            )
        else:
            primary_name = _compact_whitespace(primary_entity)
        if primary_name:
            return [primary_name]

        entities = [
            _compact_whitespace(value)
            for value in (fact.get("entities") or [])
            if _compact_whitespace(value)
        ]
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

    def _action_signals_from_fact(self, fact: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return lightweight action signals for reflection gating."""
        metadata = fact.get("metadata") if isinstance(fact.get("metadata"), dict) else {}
        raw = fact.get("action_signal") or metadata.get("action_signal")
        return self._normalize_action_signal(raw)

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
                # A completed decision is durable context, not an open
                # actionable item. It remains available through facts/state.
                return False
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
        canonical_topics = item.get("canonical_topics") or [item["canonical_name"]]
        evidence_fact_ids = [int(value) for value in item.get("evidence_fact_ids") or []]
        owner = _compact_whitespace(item.get("owner") or "")
        entity_ids = (
            self._entity_ids_for_names([owner])
            if owner and owner.lower() != "unknown"
            else []
        )
        identity_text = "\n".join([
            item["canonical_name"],
            item["summary"],
            f"item_type: {item['item_type']}",
            f"owner: {item['owner']}",
            f"status: {item['status']}",
            f"due_at: {item['due_at']}",
            f"keywords: {' '.join(keywords)}",
        ])
        identity_text_embedding = self._generate_embedding_vector(identity_text)
        item_id = self._db.upsert_actionable_item(
            item_type=item["item_type"],
            source_type=item["source_type"],
            canonical_name=item["canonical_name"],
            summary=item["summary"],
            owner=item["owner"],
            status=item["status"],
            due_at=item["due_at"],
            entity_ids=entity_ids,
            evidence_fact_ids=evidence_fact_ids,
            confidence=item["confidence"],
            importance=item["importance"],
            metadata={
                "keywords": keywords,
                "canonical_topics": canonical_topics,
            },
            identity_text_embedding=identity_text_embedding,
            identity_text=identity_text,
        )

        return item_id
    
    def _format_facts_for_actionable_prompt(
        self,
        facts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for fact in facts[:80]:
            action_signals = self._action_signals_from_fact(fact)
            rows.append({
                "id": fact.get("id"),
                "source_type": fact.get("source_type"),
                "fact_type": fact.get("fact_type"),
                "fact_kind": fact.get("fact_kind"),
                "event_time_key": fact.get("event_time_key"),
                "dialogue_time_key": fact.get("dialogue_time_key"),
                "summary": fact.get("summary"),
                "keywords": fact.get("keywords"),
                "entities": fact.get("entities") or [],
                "primary_entity": fact.get("primary_entity"),
                "fact_root_topic": fact.get("fact_root_topic") or "",
                "fact_aspect_topic": fact.get("fact_aspect_topic") or "",
                "action_signal": action_signals,
            })
        return rows

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
        text = _compact_whitespace(value).strip("'\".,:;!?，。！？、；：（）()[]{}")
        if not text:
            return "unknown"
        normalized = text.lower()
        if normalized in {"用户", "user", "the user"}:
            return "user"
        if normalized in {"助手", "assistant", "agent", "the assistant"}:
            return "assistant"
        if normalized in {"未知", "unknown"}:
            return "unknown"
        if normalized in {"其他", "other"}:
            return "unknown"
        return text

    @staticmethod
    def _clamp_float(value: Any, low: float, high: float, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = default
        return max(low, min(high, number))

    def _log_info(self, scope: str, event: str, payload: Dict[str, Any]) -> None:
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
        self._logger.info("\n%s", body)

    @staticmethod
    def _format_log_text(value: Any, *, limit: int = 500) -> str:
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
        detailed: bool = False,
        limit: Optional[int] = 12,
        stage: str = "",
    ) -> Dict[str, Any]:
        """Serialize recall candidates for compact or detailed diagnostics."""
        candidates = list(items or [])
        visible_candidates = (
            candidates
            if limit is None
            else candidates[:max(0, int(limit or 0))]
        )
        rows: List[Dict[str, Any]] = []
        for item in visible_candidates:
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
            score_details_key = (
                "_recall_stage2_match_details"
                if str(stage or "").startswith("stage2")
                else "_recall_fast_match_details"
            )
            score_details = item.get(score_details_key)
            if not isinstance(score_details, dict) and not stage:
                score_details = item.get("_recall_stage2_match_details")
            score_details_present = isinstance(score_details, dict)
            score_details = (
                score_details if isinstance(score_details, dict) else {}
            )
            score_components = dict(
                score_details.get("score_components") or {}
            )
            if not score_details_present:
                score_components = dict(
                    item.get("_recall_score_components") or {}
                )
            row: Dict[str, Any] = {
                "target": f"{item.get('target_table')}#{item.get('target_id')}",
                "level": item.get("index_level") or item.get("_recall_type"),
                "source_type": item.get("source_type"),
                "score": item.get("_recall_score"),
                "rank": item.get("_recall_rank"),
                "embedding_similarity": item.get("embedding_similarity"),
                "bm25_score": (
                    item.get("_recall_bm25_score")
                    if item.get("_recall_bm25_score") is not None
                    else score_components.get("bm25_component_score")
                ),
                "score_components": score_components,
                "has_strong_anchor": bool(item.get("has_strong_anchor")),
                "strong_anchor_reasons": list(
                    item.get("strong_anchor_reasons") or []
                ),
                "time_start": item.get("time_start"),
                "summary": self._format_log_text(summary, limit=240),
                "support_fact_ids": support_ids,
            }
            if detailed:
                match_details_key = (
                    "_recall_stage2_match_details"
                    if str(stage or "").startswith("stage2")
                    else "_recall_fast_match_details"
                )
                match_details = item.get(match_details_key)
                if not isinstance(match_details, dict) and not stage:
                    match_details = item.get("_recall_stage2_match_details")
                match_details = (
                    dict(match_details)
                    if isinstance(match_details, dict)
                    else {}
                )
                decision = item.get("_recall_decision")
                decision = dict(decision) if isinstance(decision, dict) else {}
                source = str(item.get("_recall_candidate_source") or "")
                reasons: List[str] = []
                for value in (
                    item.get("evidence")
                    or item.get("_recall_fast_match_evidence")
                    or []
                ):
                    if str(value) and str(value) not in reasons:
                        reasons.append(str(value))
                for value in self._recall_candidate_source_channels(source):
                    if value not in reasons:
                        reasons.append(value)
                decision_reason = (
                    decision.get("decision_reason")
                    or match_details.get("filter_reason")
                    or item.get("_recall_drop_reason")
                    or ""
                )
                row.update({
                    "stage": stage,
                    "candidate_source": source,
                    "retrieval_reasons": reasons,
                    "title": self._format_log_text(
                        item.get("title") or raw.get("canonical_name") or "",
                        limit=240,
                    ),
                    "summary": self._format_log_text(summary, limit=500),
                    "identity_text": self._format_log_text(
                        item.get("identity_text") or raw.get("identity_text") or "",
                        limit=500,
                    ),
                    "time_end": item.get("time_end"),
                    "match": match_details,
                    "bm25_raw_score": item.get("_bm25_score"),
                    "bm25_score": item.get("_recall_bm25_score"),
                    "episode_seed_targets": list(
                        item.get("_stage2_episode_seed_targets") or []
                    ),
                    "accepted": decision.get("accepted"),
                    "decision_reason": decision_reason,
                })
            rows.append(row)
        return {
            "count": len(candidates),
            "items": rows,
        }

    @staticmethod
    def _parse_time_expression(
        query: str,
        *,
        reference_time: Optional[str] = None,
    ) -> Tuple[Optional[str], str, str]:
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

        # Parse a day-level Chinese date range before the single-date branch.
        # The end bound is exclusive, so a query covering Apr 27 through Apr
        # 29 searches until the start of Apr 30.
        m = re.search(
            r"(?<!\d)"
            r"(?:(?P<year1>\d{4})\s*年\s*)?"
            r"(?P<month1>\d{1,2})\s*月\s*(?P<day1>\d{1,2})\s*(?:日|号)?"
            r"\s*(?:到|至|[-~～])\s*"
            r"(?:(?P<year2>\d{4})\s*年\s*)?"
            r"(?P<month2>\d{1,2})\s*月\s*(?P<day2>\d{1,2})\s*(?:日|号)?"
            r"(?!\d)",
            text,
        )
        if m:
            year1_text = m.group("year1")
            year2_text = m.group("year2")
            month1 = int(m.group("month1"))
            day1 = int(m.group("day1"))
            month2 = int(m.group("month2"))
            day2 = int(m.group("day2"))
            year1 = int(year1_text) if year1_text else now.year
            if not year1_text:
                try:
                    if datetime(year1, month1, day1).date() > now.date():
                        year1 -= 1
                except ValueError:
                    pass
            if year2_text:
                year2 = int(year2_text)
            elif year1_text:
                year2 = year1
            else:
                year2 = year1 + (1 if (month2, day2) < (month1, day1) else 0)
            try:
                start = datetime(year1, month1, day1)
                end = datetime(year2, month2, day2) + timedelta(days=1)
                if end > start:
                    clean_query = text[:m.start()] + text[m.end():]
                    return fmt(start), fmt(end), clean_query.strip()
            except ValueError:
                pass

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

        m = re.search(
            r"(?<!\d)(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?",
            text,
        )
        if m:
            try:
                start = datetime(
                    int(m.group(1)),
                    int(m.group(2)),
                    int(m.group(3)),
                )
                clean_query = text[:m.start()] + text[m.end():]
                return None, fmt(start + timedelta(days=1)), clean_query.strip()
            except ValueError:
                pass

        # A month/day without a year is interpreted relative to the recall
        # reference year. For historical-memory queries, a date later than
        # the reference date most naturally refers to the previous year.
        m = re.search(
            r"(?<!\d)(\d{1,2})\s*月\s*(\d{1,2})\s*(?:日|号)?",
            text,
        )
        if m:
            month = int(m.group(1))
            day = int(m.group(2))
            try:
                start = datetime(now.year, month, day)
                if start.date() > now.date():
                    start = datetime(now.year - 1, month, day)
                clean_query = text[:m.start()] + text[m.end():]
                return None, fmt(start + timedelta(days=1)), clean_query.strip()
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

        m = re.search(
            r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)",
            text,
        )
        if m:
            try:
                start = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                clean_query = text[:m.start()] + text[m.end():]
                return None, fmt(start + timedelta(days=1)), clean_query.strip()
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

        return None, fmt(now), text

    # ── Recall path: raw candidates -> unified rerank -> formatted evidence ─

    def process_memory_recall_immediately(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        time_end: Optional[str] = None,
        prompt_language: str = "zh",
    ) -> Dict[str, Any]:
        """Run recall immediately against the latest committed memory snapshot."""
        # Recall is read-only and must not wait for the store/reflect worker's
        # long-running LLM or embedding work. A separate WAL reader gives it
        # a consistent committed snapshot without sharing the writer connection.
        with self._db.reader_transaction() as reader_db:
            return self._recall_sync(
                query=query,
                tags=tags,
                time_end=time_end,
                memory_source_override=self._retrieval_source_override,
                recall_mode=self._recall_mode,
                prompt_language=prompt_language,
                database=reader_db,
            )

    def _recall_sync(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        time_end: Optional[str] = None,
        memory_source_override: Optional[Sequence[str]] = None,
        recall_mode: str = "normal",
        prompt_language: str = "zh",
        database: Optional[SessionDB] = None,
    ) -> Dict[str, Any]:
        requested_recall_mode = str(recall_mode or "normal").strip().lower()
        started_at = time.monotonic()
        if not self._memory_enabled or not str(query or "").strip():
            recall_report = {
                "memory_context": "",
                "requested_recall_mode": requested_recall_mode,
                "actual_recall_mode": "none",
                "status": "empty",
                "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
            }
            self._operation_reporter.on_recall_finished(recall_report)
            return recall_report
        try:
            normalized_recall_mode = requested_recall_mode
            if normalized_recall_mode not in {"stage1", "stage2", "normal"}:
                raise ValueError(
                    "recall_mode must be one of: stage1, stage2, normal"
                )
            reference_time = self._normalize_recall_time_bound(
                time_end,
                default_to_now=True,
            )
            self._log_info("memory_recall", "start", {
                "query": self._format_log_text(query, limit=500),
                "top_k": self._top_k,
                "budget": self._recall_budget,
                "tags": tags or [],
                "requested_time_end": time_end,
                "time_end": reference_time,
                "memory_source_override": list(memory_source_override or []),
                "recall_mode": normalized_recall_mode,
                "prompt_language": prompt_language,
            })

            parsed_time_start, parsed_time_end, time_stripped_query = self._parse_time_expression(
                query,
                reference_time=reference_time,
            )
            temporal_bounds: RecallTimeBounds = (
                parsed_time_start,
                parsed_time_end,
            )
            temporal_mode = self._infer_recall_temporal_mode(query)
            # Keep a successfully parsed time-only query empty for text
            # retrieval rather than reintroducing the removed time expression.
            self._log_info("memory_recall", "query_prepared", {
                "time_stripped_query": self._format_log_text(
                    time_stripped_query,
                    limit=500,
                ),
                "parsed_time_start": parsed_time_start,
                "parsed_time_end": parsed_time_end,
                "temporal_mode": temporal_mode,
            })

            memory_text: Optional[str]
            actual_recall_mode: str
            if normalized_recall_mode == "stage2":
                actual_recall_mode = "stage2"
                stage1_report = self._process_recall_stage1(
                    original_query=query,
                    time_stripped_query=time_stripped_query,
                    temporal_bounds=temporal_bounds,
                    memory_source_override=memory_source_override,
                    temporal_mode=temporal_mode,
                    prompt_language=prompt_language,
                    database=database,
                )
                memory_text = self._process_recall_stage2(
                    original_query=query,
                    time_stripped_query=time_stripped_query,
                    temporal_bounds=temporal_bounds,
                    reference_time=reference_time,
                    memory_source_override=memory_source_override,
                    temporal_mode=temporal_mode,
                    stage1_report=stage1_report,
                    prompt_language=prompt_language,
                    database=database,
                )
            else:
                stage1_report = self._process_recall_stage1(
                    original_query=query,
                    time_stripped_query=time_stripped_query,
                    temporal_bounds=temporal_bounds,
                    memory_source_override=memory_source_override,
                    temporal_mode=temporal_mode,
                    prompt_language=prompt_language,
                    database=database,
                )
                if normalized_recall_mode == "stage1":
                    actual_recall_mode = "stage1"
                    memory_text = str(stage1_report.get("memory_context") or "")
                elif not stage1_report.get("trusted"):
                    actual_recall_mode = "stage2"
                    memory_text = self._process_recall_stage2(
                        original_query=query,
                        time_stripped_query=time_stripped_query,
                        temporal_bounds=temporal_bounds,
                        reference_time=reference_time,
                        memory_source_override=memory_source_override,
                        temporal_mode=temporal_mode,
                        stage1_report=stage1_report,
                        prompt_language=prompt_language,
                        database=database,
                    )
                else:
                    memory_text = str(stage1_report.get("memory_context") or "")
                    actual_recall_mode = "stage1"
            recall_status = "ok" if memory_text else "empty"
            recall_report = {
                "memory_context": memory_text or "",
                "requested_recall_mode": normalized_recall_mode,
                "actual_recall_mode": actual_recall_mode,
                "temporal_mode": temporal_mode,
                "status": recall_status,
                "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
                "recall_context_chars": len(memory_text or ""),
            }
            self._log_info("memory_recall", "finish", {
                "status": recall_status,
                "recall_mode": normalized_recall_mode,
                "actual_recall_mode": actual_recall_mode,
                "temporal_mode": temporal_mode,
                "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
                "recall_context_chars": len(memory_text or ""),
                "recall_context": memory_text,
            })
            self._operation_reporter.on_recall_finished(recall_report)
            return recall_report
        except Exception as exc:
            self._log_info("memory_recall", "error", {
                "query": self._format_log_text(query, limit=500),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
            })
            raise

    def _retrieve_recall_stage1_seed_candidates(
        self,
        *,
        terms: Sequence[str],
        query_entity_names: Sequence[str],
        source_types: Optional[Sequence[str]],
        temporal_bounds: RecallTimeBounds,
        temporal_mode: str,
        candidate_limits: Dict[str, int],
        database: Optional[SessionDB] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve and merge Stage 1's direct lexical fact/state seeds."""
        seed_search_terms = self._recall_stage1_build_seed_search_terms(
            terms=terms,
            query_entity_names=query_entity_names,
        )
        fact_candidates, state_candidates = (
            self._retrieve_recall_raw_candidates_lexical_search(
                terms=seed_search_terms,
                candidate_source_prefix="stage1",
                source_types=source_types,
                temporal_bounds=temporal_bounds,
                temporal_mode=temporal_mode,
                candidate_limits=candidate_limits,
                database=database,
            )
        )
        return [*fact_candidates, *state_candidates]

    def _recall_stage1_build_seed_search_terms(
        self,
        *,
        terms: Sequence[str],
        query_entity_names: Sequence[str],
    ) -> List[str]:
        """Build lexical terms for Stage 1 seed retrieval.

        Concrete query entities are intentionally added only here.  Candidate
        scoring and evidence coverage continue to use the original query
        terms plus their dedicated entity-matching signal.
        """
        high_value_entity_names = [
            entity_name
            for entity_name in self._normalize_entity_names(query_entity_names)
            if self._recall_stage1_entity_value_class(
                self._recall_stage1_normalize_match_text(entity_name)
            ) == "high"
        ]
        return self._build_recall_search_terms(
            "",
            keywords=[*high_value_entity_names, *list(terms or [])],
            entities=[],
        )

    def _process_recall_stage1(
        self,
        *,
        original_query: str,
        time_stripped_query: str,
        temporal_bounds: RecallTimeBounds,
        memory_source_override: Optional[Sequence[str]] = None,
        temporal_mode: str = "dialogue_time",
        prompt_language: str = "zh",
        database: Optional[SessionDB] = None,
    ) -> Dict[str, Any]:
        """Run a deterministic, no-LLM recall path for high-confidence hits.

        Stage 1 is deliberately conservative. It returns a formatted context
        only when the retrieved candidates provide sufficient direct matching
        evidence. Otherwise it returns ``None`` so the caller can fall through
        to Stage 2 semantic retrieval.
        """
        started_at = time.monotonic()
        source_types = self._normalize_source_override(memory_source_override)
        terms = self._build_recall_search_terms(
            time_stripped_query,
            keywords=[],
            entities=[],
        )
        is_contextual_query = self._recall_stage1_is_contextual_query(original_query)

        candidate_limits = self._recall_stage1_candidate_limits(
            top_k=self._top_k,
        )
        seed_candidate_limits = candidate_limits["seed_limits"]
        selected_candidate_limits = candidate_limits["selected_limits"]
        association_per_relation_limit = candidate_limits[
            "association_per_relation_limit"
        ]
        query_entity_names = self._recall_stage1_resolve_query_entity_names(
            query=original_query,
            database=database,
        )
        self._log_info("memory_recall_stage1", "start", {
            "original_query": self._format_log_text(original_query, limit=500),
            "time_stripped_query": self._format_log_text(
                time_stripped_query,
                limit=500,
            ),
            "top_k": self._top_k,
            "budget": self._recall_budget,
            "terms": terms,
            "time_start": (temporal_bounds or (None, None))[0],
            "time_end": (temporal_bounds or (None, None))[1],
            "temporal_mode": temporal_mode,
            "memory_source_override": list(memory_source_override or []),
            "candidate_limits": candidate_limits,
            "query_entity_names": query_entity_names,
        })
        # direct recall
        seed_candidates = self._retrieve_recall_stage1_seed_candidates(
            terms=terms,
            query_entity_names=query_entity_names,
            source_types=source_types,
            temporal_bounds=temporal_bounds,
            temporal_mode=temporal_mode,
            candidate_limits=seed_candidate_limits,
            database=database,
        )
        self._log_recall_stage1_seed_candidates(
            seed_candidates=seed_candidates,
            seed_candidate_limits=seed_candidate_limits,
        )
        direct_candidates = self._recall_stage1_calculate_candidate_matching_score(
            candidates=seed_candidates,
            search_terms=terms,
            query_entity_names=query_entity_names,
            is_contextual_query=is_contextual_query,
            temporal_bounds=temporal_bounds,
        )
        self._log_recall_direct_candidates(
            stage_name="stage1",
            seed_candidates=seed_candidates,
            direct_candidates=direct_candidates,
        )
        # associative recall
        association_candidates = (
            self._retrieve_association_candidates_using_seed_candidates(
                seed_candidates=direct_candidates,
                source_types=source_types,
                temporal_bounds=temporal_bounds,
                temporal_mode=temporal_mode,
                limit=association_per_relation_limit,
                candidate_source_prefix="stage1",
                database=database,
            )
        )
        expanded_candidates = self._merge_recall_stage2_direct_and_associative_candidates(
            direct_candidates=direct_candidates,
            association_candidates=association_candidates,
            stage_name="stage1",
        )
        self._log_recall_association_candidates(
            stage_name="stage1",
            association_candidates=association_candidates,
            expanded_candidates=expanded_candidates,
        )
        semantic_query = self._recall_stage1_requires_semantic_search(
            time_stripped_query
        )
        selected_candidates = self._recall_stage1_rank_and_select_candidates(
            candidates=expanded_candidates,
            layer_limits=selected_candidate_limits,
        )
        self._log_recall_selected_candidates(
            stage_name="stage1",
            selected_candidates=selected_candidates,
            expanded_candidates=expanded_candidates,
            actionable_candidates=[],
        )

        evidence_profile = self._recall_stage1_build_evidence_profile(
            candidates=selected_candidates,
            query_terms=terms,
            query_entity_names=query_entity_names,
        )
        evidence_gate = bool(evidence_profile.get("trusted"))

        trusted = bool(selected_candidates) and evidence_gate
        memory_text = self._build_memory_retrieved_format_text(
            entries=selected_candidates,
            prompt_language=prompt_language,
        )
        stage1_finish_payload = self._log_recall_stage1_finish_payload(
            selected_candidates=selected_candidates,
            semantic_query=semantic_query,
            evidence_profile=evidence_profile,
            trusted=trusted,
            memory_text=memory_text,
            started_at=started_at,
        )
        return {
            "memory_context": memory_text or "",
            # Stage 2 receives the original direct lexical pool, not the
            # Stage 1 association results or final presentation selection.
            # It will score these candidates again under its own policy.
            "seed_candidates": list(seed_candidates),
            "evidence_profile": evidence_profile,
            "trusted": bool(trusted and memory_text),
            "elapsed_ms": stage1_finish_payload["elapsed_ms"],
        }

    @staticmethod
    def _recall_count_candidates_by_level(
        candidates: Sequence[Dict[str, Any]],
    ) -> Dict[str, int]:
        counts = {"fact": 0, "state": 0, "actionable_item": 0}
        for candidate in candidates or []:
            level = str(candidate.get("index_level") or "")
            if level in counts:
                counts[level] += 1
        return counts

    def _log_recall_stage1_seed_candidates(
        self,
        *,
        seed_candidates: Sequence[Dict[str, Any]],
        seed_candidate_limits: Dict[str, int],
    ) -> None:
        """Log the lexical retrieval output before any Stage 1 scoring."""
        seed_candidates = list(seed_candidates or [])
        payload: Dict[str, Any] = {
            "seed_candidate_count": len(seed_candidates),
            "seed_candidate_limits": dict(seed_candidate_limits or {}),
            "seed_by_level": self._recall_count_candidates_by_level(
                seed_candidates
            ),
            "facts": self._recall_log_candidate_items(
                [
                    candidate
                    for candidate in seed_candidates
                    if str(candidate.get("index_level") or "") == "fact"
                ],
                detailed=self._recall_detailed_logging,
                limit=None if self._recall_detailed_logging else 12,
                stage="stage1_seed",
            ),
            "states": self._recall_log_candidate_items(
                [
                    candidate
                    for candidate in seed_candidates
                    if str(candidate.get("index_level") or "") == "state"
                ],
                detailed=self._recall_detailed_logging,
                limit=None if self._recall_detailed_logging else 12,
                stage="stage1_seed",
            ),
        }
        self._log_info("memory_recall_stage1", "seeds_retrieved", payload)

    def _log_recall_direct_candidates(
        self,
        *,
        stage_name: str,
        seed_candidates: Sequence[Dict[str, Any]],
        direct_candidates: Sequence[Dict[str, Any]],
    ) -> None:
        """Log direct matcher acceptance for one recall stage."""
        stage_name = str(stage_name or "").strip().lower()
        if stage_name not in {"stage1", "stage2"}:
            raise ValueError(f"Unsupported recall stage: {stage_name}")
        seed_candidates = list(seed_candidates or [])
        direct_candidates = list(direct_candidates or [])
        payload: Dict[str, Any] = {
            "scored_candidate_count": len(seed_candidates),
            "direct_candidate_count": len(direct_candidates),
            "rejected_candidate_count": max(
                0, len(seed_candidates) - len(direct_candidates)
            ),
            "direct_by_level": self._recall_count_candidates_by_level(
                direct_candidates
            ),
            "direct_candidates": self._recall_log_candidate_items(
                direct_candidates,
                detailed=self._recall_detailed_logging,
                limit=None if self._recall_detailed_logging else 12,
                stage=f"{stage_name}_direct",
            ),
        }
        self._log_info(
            f"memory_recall_{stage_name}",
            "direct_candidates_scored",
            payload,
        )

    def _log_recall_association_candidates(
        self,
        *,
        stage_name: str,
        association_candidates: Sequence[Dict[str, Any]],
        expanded_candidates: Sequence[Dict[str, Any]],
    ) -> None:
        """Log association expansion and the expanded candidates."""
        stage_name = str(stage_name or "").strip().lower()
        if stage_name not in {"stage1", "stage2"}:
            raise ValueError(f"Unsupported recall stage: {stage_name}")
        association_candidates = list(association_candidates or [])
        expanded_candidates = list(expanded_candidates or [])
        association_by_relation = {"same_episode": 0, "same_state": 0}
        for candidate in association_candidates:
            relation = str(
                candidate.get("_recall_association_relation")
                or ""
            )
            if relation in association_by_relation:
                association_by_relation[relation] += 1
        payload: Dict[str, Any] = {
            "association_candidate_count": len(association_candidates),
            "association_by_relation": association_by_relation,
            "expanded_candidate_count": len(expanded_candidates),
            "expanded_by_level": self._recall_count_candidates_by_level(
                expanded_candidates
            ),
            "facts": self._recall_log_candidate_items(
                [
                    candidate
                    for candidate in expanded_candidates
                    if str(candidate.get("index_level") or "") == "fact"
                ],
                detailed=self._recall_detailed_logging,
                limit=None if self._recall_detailed_logging else 12,
                stage=f"{stage_name}_expanded",
            ),
            "states": self._recall_log_candidate_items(
                [
                    candidate
                    for candidate in expanded_candidates
                    if str(candidate.get("index_level") or "") == "state"
                ],
                detailed=self._recall_detailed_logging,
                limit=None if self._recall_detailed_logging else 12,
                stage=f"{stage_name}_expanded",
            ),
        }
        self._log_info(
            f"memory_recall_{stage_name}",
            "association_candidates_merged",
            payload,
        )

    def _log_recall_selected_candidates(
        self,
        *,
        stage_name: str,
        selected_candidates: Sequence[Dict[str, Any]],
        expanded_candidates: Sequence[Dict[str, Any]],
        actionable_candidates: Sequence[Dict[str, Any]],
    ) -> None:
        """Log final candidate selection for one recall stage."""
        stage_name = str(stage_name or "").strip().lower()
        if stage_name not in {"stage1", "stage2"}:
            raise ValueError(f"Unsupported recall stage: {stage_name}")
        selected_candidates = list(selected_candidates or [])
        expanded_candidates = list(expanded_candidates or [])
        actionable_candidates = list(actionable_candidates or [])
        payload: Dict[str, Any] = {
            "expanded_candidate_count": len(expanded_candidates),
            "selected_candidate_count": len(selected_candidates),
            "actionable_candidate_count": len(actionable_candidates),
            "selected_by_level": self._recall_count_candidates_by_level(
                selected_candidates
            ),
            "selected_candidates": self._recall_log_candidate_items(
                selected_candidates,
                detailed=self._recall_detailed_logging,
                limit=None if self._recall_detailed_logging else 12,
                stage=f"{stage_name}_selected",
            ),
            "selected_targets": [
                f"{item.get('target_table')}#{item.get('target_id')}"
                for item in selected_candidates
            ],
        }
        self._log_info(
            f"memory_recall_{stage_name}",
            "candidates_selected",
            payload,
        )

    def _log_recall_stage1_finish_payload(
        self,
        *,
        selected_candidates: Sequence[Dict[str, Any]],
        semantic_query: bool,
        evidence_profile: Dict[str, Any],
        trusted: bool,
        memory_text: str,
        started_at: float,
    ) -> Dict[str, Any]:
        """Build and log only the final Stage 1 trust/output decision."""
        selected_candidates = list(selected_candidates or [])
        status = "hit" if trusted and memory_text else "miss"
        if status == "hit":
            reason = ""
        elif not selected_candidates:
            reason = "no_selected_candidates"
        elif not memory_text:
            reason = "empty_formatted_context"
        elif semantic_query:
            reason = "semantic_query_requires_stage2"
        else:
            reason = "evidence_profile_below_threshold"
        stage1_finish_payload: Dict[str, Any] = {
            "status": status,
            "reason": reason,
            "trusted": bool(trusted),
            "semantic_query": bool(semantic_query),
            "selected_candidate_count": len(selected_candidates),
            "selected_by_level": self._recall_count_candidates_by_level(
                selected_candidates
            ),
            "strong_anchor_reasons": sorted({
                str(reason)
                for item in selected_candidates
                for reason in item.get("strong_anchor_reasons") or []
                if str(reason)
            }),
            "evidence_profile": dict(evidence_profile or {}),
            "retrieved_chars": len(memory_text or ""),
            "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
        }
        self._log_info("memory_recall_stage1", "finish", stage1_finish_payload)
        return stage1_finish_payload

    def _retrieve_association_candidates_using_seed_candidates(
        self,
        *,
        seed_candidates: Sequence[Dict[str, Any]],
        source_types: Optional[Sequence[str]],
        temporal_bounds: RecallTimeBounds,
        temporal_mode: str,
        limit: int,
        candidate_source_prefix: str = "stage1",
        database: Optional[SessionDB] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve associated facts from already accepted fact seeds."""
        db = database or self._db
        fact_seed_candidates = [
            candidate
            for candidate in seed_candidates or []
            if str(candidate.get("index_level") or "") == "fact"
        ]
        seed_scores: Dict[int, float] = {}
        for candidate in fact_seed_candidates or []:
            try:
                fact_id = int(candidate.get("target_id"))
            except (TypeError, ValueError):
                continue
            seed_scores[fact_id] = max(
                seed_scores.get(fact_id, 0.0),
                self._clamp_float(candidate.get("_recall_score"), 0.0, 1.0, 0.0),
            )
        if not seed_scores:
            return []

        per_relation_limit = max(1, int(limit or 24))
        related_scores: Dict[int, Tuple[str, float]] = {}
        relation_specs = (
            ("same_episode", db.related_fact_pairs_by_episode_fact_ids),
            ("same_state", db.related_fact_pairs_by_state_fact_ids),
        )
        for relation, loader in relation_specs:
            for pair in loader(
                list(seed_scores),
                limit=per_relation_limit,
            ):
                seed_score = seed_scores.get(int(pair["seed_fact_id"]), 0.0)
                decay = self._recall_fact_association_decay(relation)
                propagated_score = round(seed_score * decay, 4)
                related_fact_id = int(pair["related_fact_id"])
                existing = related_scores.get(related_fact_id)
                if existing is None or propagated_score > existing[1]:
                    related_scores[related_fact_id] = (
                        relation,
                        propagated_score,
                    )
        if not related_scores:
            return []

        allowed_sources = set(source_types or [])
        expanded: List[Dict[str, Any]] = []
        for fact in db.memory_facts_by_ids(list(related_scores)):
            fact_id = int(fact.get("id") or 0)
            relation_info = related_scores.get(fact_id)
            if not relation_info:
                continue
            if allowed_sources and fact.get("source_type") not in allowed_sources:
                continue
            relation, propagated_score = relation_info
            candidate = self._make_recall_memory_candidate(
                level="fact",
                row=fact,
                candidate_source=(
                    f"{candidate_source_prefix}_episode_association"
                    if relation == "same_episode"
                    else f"{candidate_source_prefix}_state_association"
                ),
                temporal_bounds=temporal_bounds,
                temporal_mode=temporal_mode,
            )
            if not candidate:
                continue
            candidate["_recall_association_score"] = propagated_score
            candidate["_recall_association_relation"] = relation
            candidate["_recall_score"] = propagated_score
            candidate["evidence"] = ["associative_recall"]
            candidate["matched"] = True
            candidate["candidate_score_threshold"] = 0.0
            candidate["filter_reason"] = ""
            candidate["has_strong_anchor"] = False
            candidate["strong_anchor_reasons"] = []
            match_details_key = (
                "_recall_stage2_match_details"
                if candidate_source_prefix == "stage2"
                else "_recall_fast_match_details"
            )
            candidate[match_details_key] = {
                "topic_match_info": {},
                "entity_match_info": {},
                "time_score_info": {},
                "score_components": {},
            }
            candidate["_recall_decision"] = {
                "accepted": True,
                "decision_reason": "accepted_associative_recall",
            }
            expanded.append(candidate)
        return expanded

    def _recall_fact_association_decay(self, relation: str) -> float:
        """Return the configured score decay for one fact association edge."""
        return {
            "same_episode": self._recall_stage1_episode_propagation_decay,
            "same_state": self._recall_stage1_state_propagation_decay,
        }.get(relation, self._recall_stage1_association_propagation_decay)

    @staticmethod
    def _recall_stage1_candidate_limits(
        *,
        top_k: int,
    ) -> Dict[str, Any]:
        """Build explicit Stage 1 retrieval, expansion, and output budgets.

        ``top_k`` controls the primary fact quota. Entity states are
        supplementary evidence. Contextual wording changes time scoring, not
        the amount of memory that Stage 1 is allowed to retrieve or return.
        """
        k = max(1, int(top_k or 1))
        seed_per_level_limit = max(8, min(24, k * 2))
        supplementary_limit = max(1, int(math.ceil(k / 2)))

        return {
            # Direct lexical recall keeps a modest, fixed over-fetch ratio so
            # score filtering and association can work without inflating the
            # Stage 1 latency or candidate pool for contextual wording.
            "seed_limits": {
                "fact": seed_per_level_limit,
                "state": seed_per_level_limit,
            },
            # The association loader applies this limit independently to each
            # relation (same episode and same state).
            "association_per_relation_limit": max(4, min(12, k)),
            # Facts are the primary answer evidence. States supplement them;
            # the ranker may reuse an empty layer's capacity for the other.
            "selected_limits": {
                "fact": k,
                "state": supplementary_limit,
            },
            # The legacy state→actionable expansion is disabled until
            # memory_work_items replaces memory_actionable_items.
            "actionable_item_limit": 0,
        }

    @staticmethod
    def _recall_candidate_source_channels(value: Any) -> set[str]:
        """Normalize prefixed candidate sources to retrieval channels."""
        source = str(value or "").strip().lower()
        if source == "both" or source.endswith("_both"):
            return {"entity_mapping", "lexical"}
        channels: set[str] = set()
        if source == "entity_mapping" or source.endswith("_entity_mapping"):
            channels.add("entity_mapping")
        if source == "lexical" or source.endswith("_lexical"):
            channels.add("lexical")
        return channels

    def _recall_stage1_calculate_candidate_matching_score(
        self,
        *,
        candidates: Sequence[Dict[str, Any]],
        search_terms: Sequence[str] = (),
        query_entity_names: Sequence[str] = (),
        is_contextual_query: bool = False,
        temporal_bounds: RecallTimeBounds = None,
    ) -> List[Dict[str, Any]]:
        """Score direct Stage 1 candidates and return accepted ones.

        This method scores only direct lexical candidates. Associated recall
        candidates receive their propagated score at expansion time instead.
        """
        direct_candidates: List[Dict[str, Any]] = []
        for candidate in candidates or []:
            matching_score_info = (
                self._recall_stage1_calculate_single_candidate_matching_score(
                    candidate,
                    search_terms=search_terms,
                    query_entity_names=query_entity_names,
                    is_contextual_query=is_contextual_query,
                    temporal_bounds=temporal_bounds,
                )
            )
            match_details = dict(
                matching_score_info.get("_recall_fast_match_details") or {}
            )
            candidate.update({
                key: value
                for key, value in matching_score_info.items()
                if key != "_recall_fast_match_details"
            })
            # Direct Stage 1 matching is represented by the normalized
            # strong-anchor fields; do not carry the legacy evidence list
            # forward on the candidate.
            candidate.pop("evidence", None)
            candidate.pop("_recall_fast_match_evidence", None)
            candidate["_recall_fast_match_details"] = match_details
            candidate["_recall_score"] = self._clamp_float(
                matching_score_info.get("score"),
                0.0,
                1.0,
                0.0,
            )
            candidate["_recall_candidate_source"] = (
                candidate.get("_recall_candidate_source") or "stage1_lexical"
            )
            entity_match_info = match_details.get("entity_match_info")
            if not isinstance(entity_match_info, dict):
                entity_match_info = {}
            if entity_match_info.get("matched_entity_names"):
                candidate["_recall_entity_names"] = list(
                    entity_match_info.get("matched_entity_names") or []
                )
            accepted = bool(
                matching_score_info.get("matched")
            )
            candidate["_recall_decision"] = {
                "accepted": accepted,
                "decision_reason": (
                    "accepted"
                    if accepted
                    else str(matching_score_info.get("filter_reason") or "not_matched")
                ),
            }
            if accepted:
                direct_candidates.append(candidate)
        return direct_candidates

    def _recall_stage1_rank_and_select_candidates(
        self,
        *,
        candidates: Sequence[Dict[str, Any]],
        layer_limits: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """Rank accepted Stage 1 candidates and select them by layer limits."""
        ranked_by_level: Dict[str, List[Dict[str, Any]]] = {
            "fact": [],
            "state": [],
            "actionable_item": [],
        }
        for candidate in candidates or []:
            if not bool((candidate.get("_recall_decision") or {}).get("accepted")):
                continue
            level = str(candidate.get("index_level") or "")
            if level in ranked_by_level:
                ranked_by_level[level].append(candidate)

        def rank_key(item: Dict[str, Any]) -> Tuple[float, str, int]:
            return (
                float(item.get("_recall_score") or 0.0),
                str(item.get("time_start") or ""),
                int(item.get("target_id") or 0),
            )

        for candidates_for_level in ranked_by_level.values():
            candidates_for_level.sort(key=rank_key, reverse=True)

        layer_limits = {
            str(layer): max(0, int(limit or 0))
            for layer, limit in (layer_limits or {}).items()
        }
        selected_candidates: List[Dict[str, Any]] = []
        seen_targets: set[Tuple[str, int]] = set()
        max_selected_candidates = max(1, sum(layer_limits.values()))

        def append_candidate(candidate: Dict[str, Any]) -> bool:
            try:
                target = (
                    str(candidate.get("target_table") or ""),
                    int(candidate.get("target_id")),
                )
            except (TypeError, ValueError):
                return False
            if target in seen_targets or len(selected_candidates) >= max_selected_candidates:
                return False
            seen_targets.add(target)
            selected_candidates.append(candidate)
            return True

        selected_by_layer: Dict[str, int] = {layer: 0 for layer in layer_limits}
        for layer, limit in layer_limits.items():
            for candidate in ranked_by_level.get(layer, []):
                if selected_by_layer[layer] >= limit:
                    break
                if append_candidate(candidate):
                    selected_by_layer[layer] += 1

        # Reuse empty layer quota so a missing layer does not reduce the
        # available output capacity. The first pass enforces each layer's
        # preferred limit; this pass can use unused capacity from another
        # layer until the global sum of layer limits is reached.
        for layer in layer_limits:
            for candidate in ranked_by_level.get(layer, []):
                if len(selected_candidates) >= max_selected_candidates:
                    break
                if append_candidate(candidate):
                    selected_by_layer[layer] += 1
            if len(selected_candidates) >= max_selected_candidates:
                break
        selected_by_layer = {
            layer: sum(
                1
                for candidate in selected_candidates
                if str(candidate.get("index_level") or "") == layer
            )
            for layer in layer_limits
        }
        return selected_candidates

    def _recall_stage1_calculate_single_candidate_matching_score(
        self,
        candidate: Dict[str, Any],
        *,
        search_terms: Sequence[str] = (),
        query_entity_names: Sequence[str] = (),
        is_contextual_query: bool = False,
        temporal_bounds: RecallTimeBounds = None,
    ) -> Dict[str, Any]:
        """Score topic/entity anchors and temporal relevance for one candidate.
        """
        topic_match_info = (
            self._recall_calculate_search_terms_overlap_with_candidate_topics(
                candidate,
                search_terms,
                allow_substring=True,
            )
        )
        topic_coverage_ratio = float(topic_match_info.get("coverage") or 0.0)
        keyword_match_info = (
            self._recall_calculate_search_terms_overlap_with_candidate_keywords(
                candidate,
                search_terms,
                allow_substring=True,
            )
        )
        keyword_coverage_ratio = float(
            keyword_match_info.get("coverage") or 0.0
        )

        candidate_source = str(candidate.get("_recall_candidate_source") or "")
        candidate_sources = self._recall_candidate_source_channels(candidate_source)
        entity_match_info = self._recall_stage1_matched_entity_names(
            query_entity_names=query_entity_names,
            candidate_entity_names=candidate.get("entities") or [],
        )
        high_value_entity_matched = bool(
            entity_match_info.get("high_value_entity_matched")
        )

        topic_matched_overlap_score = min(
            self._recall_stage1_topic_overlap_score_weight,
            self._recall_stage1_topic_overlap_score_weight
            * topic_coverage_ratio)
        keyword_overlap_score = min(
            self._recall_stage1_keyword_overlap_score_weight,
            self._recall_stage1_keyword_overlap_score_weight
            * keyword_coverage_ratio,
        )
        entity_score = (
            self._recall_stage1_entity_matched_score
            if high_value_entity_matched
            else 0.0
        )
        time_score_info = self._calculate_recall_candidate_time_score(
            candidate,
            temporal_bounds=temporal_bounds,
        )
        time_weight = self._recall_stage1_time_score_weight
        if is_contextual_query:
            time_weight *= self._recall_stage1_contextual_time_score_multiplier

        score_components = {
            "topic_overlap": round(topic_matched_overlap_score, 4),
            "keyword_overlap": round(keyword_overlap_score, 4),
            "entity_matched": round(entity_score, 4),
            "time_score": round(
                time_weight * float(time_score_info.get("time_score") or 0.0),
                4,
            ),
        }
        score = min(1.0, sum(score_components.values()))
        strong_anchor_reasons: List[str] = []
        if topic_coverage_ratio > 0.0:
            strong_anchor_reasons.append("topic_match")
        if entity_score > 0.0:
            strong_anchor_reasons.append("entity_match")
        has_strong_anchor = bool(strong_anchor_reasons)

        fast_match_details = {
            "topic_match_info": {
                **topic_match_info,
                "keyword_match_info": dict(keyword_match_info),
            },
            "entity_match_info": dict(entity_match_info),
            "time_score_info": dict(time_score_info),
            "score_components": dict(score_components),
        }

        if not has_strong_anchor:
            return {
                "matched": False,
                "score": 0.0,
                "candidate_source": candidate_source,
                "candidate_sources": sorted(candidate_sources - {""}),
                "has_strong_anchor": has_strong_anchor,
                "strong_anchor_reasons": strong_anchor_reasons,
                "filter_reason": "no_strong_anchor",
                "_recall_fast_match_details": fast_match_details,
            }
        return {
            "matched": True,
            "score": round(score, 4),
            "candidate_source": candidate_source,
            "candidate_sources": sorted(candidate_sources - {""}),
            "filter_reason": "",
            "has_strong_anchor": has_strong_anchor,
            "strong_anchor_reasons": strong_anchor_reasons,
            "_recall_fast_match_details": fast_match_details,
        }

    def _recall_stage1_build_effective_evidence_terms(
        self,
        *,
        query_terms: Sequence[str],
    ) -> Dict[str, Any]:
        """Filter query terms used only by Stage 1 evidence coverage.

        Retrieval and candidate scoring deliberately continue to use the full
        lexical query.  The profile only removes fixed question and dialogue
        framing terms that cannot be evidence anchors.
        """
        ignored_terms = {
            "什么", "哪个", "哪位", "哪种", "哪本", "哪部", "多少", "几",
            "何时", "哪里", "怎么", "如何", "是否", "吗", "呢",
            "what", "which", "who", "whom", "whose", "when", "where",
            "how", "whether", "howmany", "howmuch", "whichone",
            "whichkind", "whichbook", "whichmovie",
        }
        coverage_terms: List[str] = []
        seen_term_keys: set[str] = set()
        excluded_terms: List[str] = []
        for value in query_terms or ():
            term = self._recall_stage1_clean_anchor(value)
            term_key = self._generate_topic_name_key(term) if term else ""
            if not term or not term_key or term_key in seen_term_keys:
                continue
            seen_term_keys.add(term_key)
            if term_key in ignored_terms:
                excluded_terms.append(term)
            else:
                coverage_terms.append(term)

        return {
            "coverage_terms": coverage_terms,
            "excluded_terms": excluded_terms,
        }

    def _recall_stage1_build_evidence_profile(
        self,
        *,
        candidates: Sequence[Dict[str, Any]],
        query_terms: Sequence[str],
        query_entity_names: Sequence[str],
    ) -> Dict[str, Any]:
        """Determine whether selected candidates cover the query's anchors.

        This deliberately consumes only the cached direct-match diagnostics in
        ``_recall_fast_match_details``.  It does not re-score candidates or
        create new topic/entity matches while evaluating the final Stage 1
        result.
        """
        effective_terms_info = self._recall_stage1_build_effective_evidence_terms(
            query_terms=query_terms,
        )
        query_term_by_key: Dict[str, str] = {}
        for value in effective_terms_info.get("coverage_terms") or []:
            term = self._recall_stage1_clean_anchor(value)
            term_key = self._generate_topic_name_key(term) if term else ""
            if not term_key or term_key in query_term_by_key:
                continue
            query_term_by_key[term_key] = term
        effective_term_keys = list(query_term_by_key)
        effective_term_key_set = set(effective_term_keys)
        high_value_entity_by_key: Dict[str, str] = {}
        for value in self._normalize_entity_names(query_entity_names or []):
            entity_key = self._recall_stage1_normalize_match_text(value)
            if (
                entity_key
                and self._recall_stage1_entity_value_class(entity_key) == "high"
            ):
                high_value_entity_by_key.setdefault(entity_key, value)

        matched_topic_term_keys: set[str] = set()
        matched_keyword_term_keys: set[str] = set()
        matched_entity_keys: set[str] = set()
        cached_match_candidate_count = 0
        for candidate in candidates or ():
            details = candidate.get("_recall_fast_match_details")
            if not isinstance(details, dict):
                continue
            topic_match_info = details.get("topic_match_info")
            topic_match_info = (
                topic_match_info
                if isinstance(topic_match_info, dict)
                else {}
            )
            keyword_match_info = topic_match_info.get("keyword_match_info")
            keyword_match_info = (
                keyword_match_info
                if isinstance(keyword_match_info, dict)
                else {}
            )
            entity_match_info = details.get("entity_match_info")
            entity_match_info = (
                entity_match_info
                if isinstance(entity_match_info, dict)
                else {}
            )
            candidate_matched = False
            for value in topic_match_info.get("matched_terms") or []:
                term_key = self._generate_topic_name_key(str(value or ""))
                if term_key in effective_term_key_set:
                    matched_topic_term_keys.add(term_key)
                    candidate_matched = True
            for value in keyword_match_info.get("matched_terms") or []:
                term_key = self._generate_topic_name_key(str(value or ""))
                if term_key in effective_term_key_set:
                    matched_keyword_term_keys.add(term_key)
                    candidate_matched = True
            for value in (
                entity_match_info.get("high_value_matched_entity_names") or []
            ):
                entity_key = self._recall_stage1_normalize_match_text(value)
                if entity_key in high_value_entity_by_key:
                    matched_entity_keys.add(entity_key)
                    candidate_matched = True
            if candidate_matched:
                cached_match_candidate_count += 1

        matched_term_keys = (
            matched_topic_term_keys | matched_keyword_term_keys
        )
        matched_query_terms = [
            query_term_by_key[key]
            for key in effective_term_keys
            if key in matched_term_keys
        ]
        missing_query_terms = [
            query_term_by_key[key]
            for key in effective_term_keys
            if key not in matched_term_keys
        ]
        matched_topic_query_terms = [
            query_term_by_key[key]
            for key in effective_term_keys
            if key in matched_topic_term_keys
        ]
        matched_keyword_query_terms = [
            query_term_by_key[key]
            for key in effective_term_keys
            if key in matched_keyword_term_keys
        ]
        query_term_count = len(effective_term_keys)
        query_term_coverage_ratio = (
            len(matched_term_keys) / query_term_count
            if query_term_count
            else 1.0
        )
        query_term_coverage_sufficient = (
            query_term_coverage_ratio >= self._recall_stage1_min_term_coverage
        )

        matched_high_value_entities = [
            high_value_entity_by_key[key]
            for key in high_value_entity_by_key
            if key in matched_entity_keys
        ]
        missing_high_value_entities = [
            high_value_entity_by_key[key]
            for key in high_value_entity_by_key
            if key not in matched_entity_keys
        ]
        high_value_entity_count = len(high_value_entity_by_key)
        entity_coverage = (
            len(matched_entity_keys) / high_value_entity_count
            if high_value_entity_count
            else 1.0
        )
        query_entity_coverage_sufficient = not missing_high_value_entities
        has_primary_query_coverage = bool(
            matched_topic_term_keys or matched_high_value_entities
        )
        trusted = bool(candidates) and has_primary_query_coverage and (
            query_term_coverage_sufficient and query_entity_coverage_sufficient
        )
        return {
            "trusted": trusted,
            "candidate_count": len(candidates),
            "cached_match_candidate_count": cached_match_candidate_count,
            "coverage_terms": list(effective_terms_info.get("coverage_terms") or []),
            "excluded_query_terms": list(effective_terms_info.get("excluded_terms") or []),
            "query_term_count": query_term_count,
            "matched_query_terms": matched_query_terms,
            "matched_topic_query_terms": matched_topic_query_terms,
            "matched_keyword_query_terms": matched_keyword_query_terms,
            "missing_query_terms": missing_query_terms,
            "query_term_coverage_ratio": round(query_term_coverage_ratio, 4),
            "query_term_coverage_threshold": (
                self._recall_stage1_min_term_coverage
            ),
            "query_term_coverage_sufficient": query_term_coverage_sufficient,
            "high_value_query_entity_count": high_value_entity_count,
            "matched_high_value_entities": matched_high_value_entities,
            "missing_high_value_entities": missing_high_value_entities,
            "entity_coverage": round(entity_coverage, 4),
            "query_entity_coverage_sufficient": query_entity_coverage_sufficient,
        }

    @staticmethod
    def _recall_stage1_normalize_match_text(value: Any) -> str:
        return _compact_whitespace(value).lower().strip("'\".,:;!?，。！？、；：（）()[]{}")

    @staticmethod
    def _recall_stage1_clean_anchor(value: Any) -> str:
        if isinstance(value, dict):
            value = value.get("name") or value.get("text") or ""
        anchor = _compact_whitespace(value)
        if not anchor:
            return ""
        normalized = anchor.lower().strip("'\".,:;!?，。！？、；：（）()[]{}")
        if normalized in {
            "general", "topic", "state", "entity", "user", "assistant",
            "用户", "助手", "unknown", "unknown_speaker",
        }:
            return ""
        if len(normalized) < 2 or len(normalized) > 80:
            return ""
        return anchor

    def _recall_stage1_resolve_query_entity_names(
        self,
        *,
        query: str,
        database: Optional[SessionDB] = None,
    ) -> List[str]:
        """Resolve entity anchors explicitly present in the query."""
        db = database or self._db
        rows = db.find_entity_nodes_in_text(str(query or ""), limit=12)
        return self._normalize_entity_names([
            row.get("name")
            for row in rows
        ], limit=12)

    def _recall_stage1_matched_entity_names(
        self,
        *,
        query_entity_names: Sequence[str],
        candidate_entity_names: Sequence[Any],
    ) -> Dict[str, Any]:
        """Classify exact entity matches by their anchor value.

        Role-like labels such as ``用户`` and ``助手`` remain visible in the
        diagnostics, but only concrete (high-value) entity matches qualify as
        an entity strong anchor.
        """
        candidate_names = self._normalize_entity_names(
            candidate_entity_names,
            limit=24,
        )
        candidate_high_value_keys: set[str] = set()
        candidate_low_value_keys: Dict[str, set[str]] = {}
        for name in candidate_names:
            normalized = self._recall_stage1_normalize_match_text(name)
            if not normalized:
                continue
            value_class = self._recall_stage1_entity_value_class(normalized)
            if value_class == "high":
                candidate_high_value_keys.add(normalized)
            else:
                candidate_low_value_keys.setdefault(value_class, set()).add(
                    normalized
                )

        matched: List[str] = []
        high_value_matched: List[str] = []
        low_value_matched: List[str] = []
        seen_matched: set[str] = set()
        for name in self._normalize_entity_names(list(query_entity_names or [])):
            normalized = self._recall_stage1_normalize_match_text(name)
            if not normalized:
                continue
            value_class = self._recall_stage1_entity_value_class(normalized)
            if value_class == "high":
                is_match = normalized in candidate_high_value_keys
            else:
                # Low-value aliases represent the same conversational role
                # even when the query and candidate use different languages
                # (for example ``用户`` and ``user``).
                is_match = bool(candidate_low_value_keys.get(value_class))
            if not is_match or normalized in seen_matched:
                continue
            seen_matched.add(normalized)
            matched.append(name)
            if value_class == "high":
                high_value_matched.append(name)
            else:
                low_value_matched.append(name)

        high_value_matched = list(high_value_matched)
        low_value_matched = list(low_value_matched)
        return {
            "matched_entity_names": matched,
            "high_value_matched_entity_names": high_value_matched,
            "low_value_matched_entity_names": low_value_matched,
            "entity_matched": bool(matched),
            "high_value_entity_matched": bool(high_value_matched),
            "low_value_entity_matched": bool(low_value_matched),
            "entity_strong_anchor": bool(high_value_matched),
            "high_value_entity_match_count": len(high_value_matched),
            "low_value_entity_match_count": len(low_value_matched),
        }

    @classmethod
    def _recall_stage1_entity_value_class(cls, value: Any) -> str:
        normalized = cls._recall_stage1_normalize_match_text(value)
        for value_class, aliases in _LOW_VALUE_ENTITY_ALIASES.items():
            if normalized in {
                cls._recall_stage1_normalize_match_text(alias)
                for alias in aliases
            }:
                return value_class
        return "high"

    @classmethod
    def _recall_stage1_contains_anchor(cls, query: str, anchor: str) -> bool:
        query_text = _compact_whitespace(query).lower()
        anchor_text = _compact_whitespace(anchor).lower()
        if not query_text or not anchor_text:
            return False
        if re.search(r"[\u4e00-\u9fff]", anchor_text):
            return anchor_text.replace(" ", "") in query_text.replace(" ", "")
        pattern = rf"(?<![a-z0-9]){re.escape(anchor_text)}(?![a-z0-9])"
        return re.search(pattern, query_text) is not None

    def _calculate_recall_candidate_time_score(
        self,
        candidate: Dict[str, Any],
        *,
        temporal_bounds: RecallTimeBounds = None,
    ) -> Dict[str, Any]:
        """Calculate continuous temporal proximity for one recall candidate.

        A closed ``[start, end]`` window gives every in-range candidate full
        temporal relevance. For an open start, proximity is measured from
        ``temporal_bounds.end`` so older memories before that end remain
        distinguishable.
        """
        raw = (
            candidate.get("_hydrated")
            if isinstance(candidate.get("_hydrated"), dict)
            else {}
        )
        level = str(candidate.get("index_level") or "").strip().lower()
        window_start_text, window_end_text = temporal_bounds or (None, None)
        window_start = self._recall_stage1_parse_datetime(window_start_text)
        window_end = self._recall_stage1_parse_datetime(window_end_text)
        reference = window_end
        if reference is None:
            return {
                "time_score": 0.0,
                "time_distance_seconds": None,
                "candidate_time": "",
                "within_temporal_bounds": False,
                "half_life_seconds": 0,
            }

        time_values: List[Tuple[datetime, str]] = []

        def add_time(value: Any) -> None:
            parsed = self._recall_stage1_parse_datetime(value)
            text = _compact_whitespace(value)
            if parsed is not None and text:
                time_values.append((parsed, text))

        if level == "fact":
            add_time(raw.get("dialogue_time_key"))
            add_time(candidate.get("time_end"))
            add_time(candidate.get("time_start"))
            half_life_seconds = self._recall_fact_time_score_half_life_seconds
        elif level == "state":
            for event in self._normalize_time_line(
                raw.get("time_line"),
                limit=20,
                max_chars=2400,
            ):
                add_time(event.get("occurred_at"))
            add_time(raw.get("updated_at"))
            add_time(candidate.get("time_end"))
            add_time(candidate.get("time_start"))
            half_life_seconds = self._recall_persistent_state_time_score_half_life_seconds
        else:
            add_time(candidate.get("time_end"))
            add_time(candidate.get("time_start"))
            half_life_seconds = self._recall_persistent_state_time_score_half_life_seconds

        if not time_values:
            return {
                "time_score": 0.0,
                "time_distance_seconds": None,
                "candidate_time": "",
                "within_temporal_bounds": False,
                "half_life_seconds": half_life_seconds,
            }

        if window_start is not None and window_end is not None:
            def distance_to_window(value: datetime) -> float:
                if window_start is not None and value < window_start:
                    return (window_start - value).total_seconds()
                if window_end is not None and value > window_end:
                    return (value - window_end).total_seconds()
                return 0.0

            candidate_time, candidate_time_text = min(
                time_values,
                key=lambda item: distance_to_window(item[0]),
            )
            distance_seconds = distance_to_window(candidate_time)
        else:
            candidate_time, candidate_time_text = max(
                time_values,
                key=lambda item: item[0],
            )
            distance_seconds = abs((reference - candidate_time).total_seconds())

        time_score = math.exp(
            -max(0.0, distance_seconds) / max(1, half_life_seconds)
        )
        return {
            "time_score": round(float(time_score), 4),
            "time_distance_seconds": round(float(distance_seconds), 2),
            "candidate_time": candidate_time_text,
            "within_temporal_bounds": bool(distance_seconds <= 0.0),
            "half_life_seconds": int(half_life_seconds),
        }

    def _recall_stage1_is_high_priority_actionable(
        self,
        candidate: Dict[str, Any],
        *,
        reference_time: str,
    ) -> bool:
        raw = candidate.get("_hydrated") if isinstance(candidate.get("_hydrated"), dict) else {}
        status = str(raw.get("status") or "").lower()
        if status not in {"open", "in_progress", "blocked", "pending", "unknown"}:
            return False
        importance = self._clamp_float(
            raw.get("importance"),
            0.0,
            1.0,
            0.0,
        )
        due_at = self._recall_stage1_parse_datetime(raw.get("due_at"))
        reference = self._recall_stage1_parse_datetime(reference_time)
        due_soon = False
        if due_at is not None and reference is not None:
            due_soon = (due_at - reference).total_seconds() <= 24 * 60 * 60
        min_importance = self._recall_stage1_actionable_min_importance
        return importance >= min_importance or status == "blocked" or due_soon

    @staticmethod
    def _recall_stage1_parse_datetime(value: Any) -> Optional[datetime]:
        text = str(value or "").strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            try:
                parsed = datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return None
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed

    @staticmethod
    def _recall_stage1_is_contextual_query(query: str) -> bool:
        lower = str(query or "").lower()
        markers = (
            "这个", "那个", "刚才", "前面", "继续", "然后", "目前", "接下来",
            "what about it", "that one", "continue", "then", "next",
        )
        return any(marker in lower for marker in markers)

    @staticmethod
    def _recall_stage1_is_actionable_query(query: str) -> bool:
        lower = str(query or "").lower()
        markers = (
            "待办", "任务", "提醒", "截止", "跟进", "下一步", "承诺", "决定",
            "风险", "阻塞", "什么时候完成", "还要做什么",
            "todo", "task", "remind", "deadline", "follow up", "next step",
            "commit", "decision", "risk", "blocked",
        )
        return any(marker in lower for marker in markers)

    @staticmethod
    def _recall_stage1_requires_semantic_search(query: str) -> bool:
        lower = str(query or "").lower()
        markers = (
            "为什么", "为何", "原因", "如何", "怎么", "比较", "区别", "历史",
            "趋势", "变化", "全部", "所有", "之前", "之后", "最早", "第一次",
            "why", "how", "compare", "difference", "history", "trend", "before",
            "after", "all", "first",
        )
        return any(marker in lower for marker in markers)

    def _merge_recall_stage2_seed_candidates(
        self,
        *,
        stage1_lexical_candidates: Sequence[Dict[str, Any]],
        stage2_lexical_candidates: Sequence[Dict[str, Any]],
        stage2_embedding_candidates: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Deduplicate the independent Stage 2 direct-retrieval channels."""
        seed_candidates = [
            *list(stage1_lexical_candidates or []),
            *list(stage2_lexical_candidates or []),
            *list(stage2_embedding_candidates or []),
        ]

        merged_candidates: Dict[Tuple[str, int], Dict[str, Any]] = {}

        def merge_candidate_group(
            candidates: Sequence[Dict[str, Any]],
        ) -> None:
            for raw_candidate in candidates or []:
                try:
                    target = (
                        str(raw_candidate.get("target_table") or ""),
                        int(raw_candidate.get("target_id")),
                    )
                except (TypeError, ValueError):
                    continue
                candidate = dict(raw_candidate)
                existing = merged_candidates.get(target)
                if existing is None:
                    merged_candidates[target] = candidate
                    continue
                if float(candidate.get("_recall_bm25_score") or 0.0) > float(
                    existing.get("_recall_bm25_score") or 0.0
                ):
                    existing["_recall_bm25_score"] = candidate.get(
                        "_recall_bm25_score"
                    )
                    existing["_bm25_score"] = candidate.get("_bm25_score")
                entity_names = self._normalize_entity_names([
                    *(existing.get("_recall_entity_names") or []),
                    *(candidate.get("_recall_entity_names") or []),
                ], limit=24)
                if entity_names:
                    existing["_recall_entity_names"] = entity_names

        merge_candidate_group(stage1_lexical_candidates)
        merge_candidate_group(stage2_lexical_candidates)
        merge_candidate_group(stage2_embedding_candidates)
        merged_by_level: Dict[str, List[Dict[str, Any]]] = {
            "fact": [],
            "state": [],
            "actionable_item": [],
        }
        for candidate in merged_candidates.values():
            level = str(candidate.get("index_level") or "")
            if level in merged_by_level:
                merged_by_level[level].append(candidate)
        return {
            "by_level": merged_by_level,
            "seed_candidates": seed_candidates,
            "stage1_lexical_candidates": list(stage1_lexical_candidates or []),
            "stage2_lexical_candidates": list(stage2_lexical_candidates or []),
            "stage2_embedding_candidates": list(stage2_embedding_candidates or []),
            "merged_candidates": list(merged_candidates.values()),
            "merged_count": len(merged_candidates),
            "seed_count": len(seed_candidates),
        }

    def _recall_stage2_calculate_candidate_matching_score(
        self,
        *,
        candidates: Sequence[Dict[str, Any]],
        search_terms: Sequence[str],
        query_terms: Sequence[str],
        query_embedding: Optional[np.ndarray],
        query_entity_names: Sequence[str] = (),
        is_contextual_query: bool = False,
        temporal_bounds: RecallTimeBounds = None,
    ) -> List[Dict[str, Any]]:
        """Score Stage 2 seeds in place and return accepted direct candidates."""
        direct_candidates: List[Dict[str, Any]] = []
        for candidate in candidates or []:
            memory_type = str(candidate.get("index_level") or "").strip().lower()
            if memory_type not in {"fact", "state"}:
                continue
            matching_score_info = (
                self._recall_stage2_calculate_single_candidate_matching_score(
                    candidate,
                    memory_type=memory_type,
                    search_terms=search_terms,
                    query_terms=query_terms,
                    query_embedding=query_embedding,
                    query_entity_names=query_entity_names,
                    is_contextual_query=is_contextual_query,
                    temporal_bounds=temporal_bounds,
                )
            )
            match_details = dict(
                matching_score_info.get("_recall_stage2_match_details") or {}
            )
            candidate.update({
                key: value
                for key, value in matching_score_info.items()
                if key != "_recall_stage2_match_details"
            })
            candidate["_recall_stage2_match_details"] = match_details
            candidate["_recall_type"] = memory_type
            candidate["_recall_score"] = self._clamp_float(
                matching_score_info.get("score"),
                0.0,
                1.0,
                0.0,
            )
            filter_reason = str(
                matching_score_info.get("filter_reason") or ""
            )
            if filter_reason:
                candidate["_recall_drop_reason"] = filter_reason
            matched = bool(matching_score_info.get("matched"))
            candidate["_recall_decision"] = {
                "accepted": matched,
                "decision_reason": (
                    "accepted_stage2_direct_retrieval"
                    if matched
                    else filter_reason or "stage2_direct_score_zero"
                ),
            }
            if bool(
                (candidate.get("_recall_decision") or {}).get("accepted")
            ):
                direct_candidates.append(candidate)
        return direct_candidates

    def _merge_recall_stage2_direct_and_associative_candidates(
        self,
        *,
        direct_candidates: Sequence[Dict[str, Any]],
        association_candidates: Sequence[Dict[str, Any]],
        stage_name: str,
    ) -> List[Dict[str, Any]]:
        """Merge direct and propagated candidates without re-scoring relations.

        The same merge path is shared by Stage 1 and Stage 2. Association
        candidates carry common score and relation fields; their source marks
        which recall stage created the association.
        """
        if stage_name not in {"stage1", "stage2"}:
            raise ValueError(f"Unsupported recall stage: {stage_name}")
        candidates_by_target: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for candidate in direct_candidates or []:
            try:
                target = (
                    str(candidate.get("target_table") or ""),
                    int(candidate.get("target_id")),
                )
            except (TypeError, ValueError):
                continue
            candidates_by_target[target] = candidate
        for association_candidate in association_candidates or []:
            try:
                target = (
                    str(association_candidate.get("target_table") or ""),
                    int(association_candidate.get("target_id")),
                )
            except (TypeError, ValueError):
                continue
            association_score = self._clamp_float(
                association_candidate.get("_recall_association_score"),
                0.0,
                1.0,
                0.0,
            )
            association_relation = str(
                association_candidate.get("_recall_association_relation") or ""
            )
            existing = candidates_by_target.get(target)
            if existing is None:
                candidates_by_target[target] = association_candidate
                continue
            existing_association_score = self._clamp_float(
                existing.get("_recall_association_score"),
                0.0,
                1.0,
                0.0,
            )
            if association_score >= existing_association_score:
                existing["_recall_association_score"] = association_score
                existing["_recall_association_relation"] = association_relation
            else:
                association_score = existing_association_score
                association_relation = str(
                    existing.get("_recall_association_relation") or ""
                )
            existing["_recall_association_score"] = association_score
            existing["_recall_association_relation"] = association_relation
            current_score = self._clamp_float(
                existing.get("_recall_score"),
                0.0,
                1.0,
                0.0,
            )
            existing["_recall_score"] = round(max(
                current_score,
                association_score,
            ), 4)
            evidence = list(
                existing.get("evidence")
                or existing.get("_recall_fast_match_evidence")
                or []
            )
            if "associative_recall" not in evidence:
                evidence.append("associative_recall")
            existing["evidence"] = evidence
            existing["matched"] = True
            existing["filter_reason"] = ""
            if not bool(
                (existing.get("_recall_decision") or {}).get("accepted")
            ):
                existing["_recall_decision"] = {
                    "accepted": True,
                    "decision_reason": "accepted_associative_recall",
                }

            # Preserve the best lexical BM25 evidence when an associated fact
            # is the same target as a direct seed.
            try:
                existing_bm25 = float(existing.get("_bm25_score"))
            except (TypeError, ValueError):
                existing_bm25 = None
            try:
                candidate_bm25 = float(association_candidate.get("_bm25_score"))
            except (TypeError, ValueError):
                candidate_bm25 = None
            if candidate_bm25 is not None and (
                existing_bm25 is None or candidate_bm25 < existing_bm25
            ):
                existing["_bm25_score"] = candidate_bm25
                existing["_recall_bm25_score"] = self._clamp_float(
                    association_candidate.get("_recall_bm25_score"),
                    0.0,
                    1.0,
                    0.0,
                )
                existing["_recall_bm25_rank"] = association_candidate.get(
                    "_recall_bm25_rank"
                )

        return list(candidates_by_target.values())

    def _recall_stage2_rank_and_select_candidates(
        self,
        *,
        candidates: Sequence[Dict[str, Any]],
        final_candidate_limits: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """Rank direct and associative candidates by their already-final score."""
        candidates_by_level: Dict[str, List[Dict[str, Any]]] = {
            "fact": [],
            "state": [],
        }
        for candidate in candidates or []:
            if not bool((candidate.get("_recall_decision") or {}).get("accepted")):
                continue
            level = str(candidate.get("index_level") or "")
            if level in candidates_by_level:
                candidates_by_level[level].append(candidate)
        ranked_by_level: Dict[str, List[Dict[str, Any]]] = {}
        for level in ("fact", "state"):
            ranked_by_level[level] = sorted(
                candidates_by_level[level],
                key=lambda item: (
                    float(item.get("_recall_score") or 0.0),
                    str(item.get("time_start") or ""),
                    int(item.get("target_id") or 0),
                ),
                reverse=True,
            )
        return self._assemble_recall_evidence_candidates(
            ranked_fact_candidates=ranked_by_level["fact"],
            ranked_state_candidates=ranked_by_level["state"],
            layer_limits=final_candidate_limits,
        )

    def _retrieve_recall_stage2_seed_candidates(
        self,
        *,
        stage1_lexical_candidates: Sequence[Dict[str, Any]],
        search_terms: Sequence[str],
        query_embedding: Optional[np.ndarray],
        source_types: Optional[Sequence[str]],
        temporal_bounds: RecallTimeBounds,
        temporal_mode: str,
        seed_channel_limits: Dict[str, Dict[str, int]],
        database: Optional[SessionDB] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve and deduplicate all direct Stage 2 seed channels.

        Stage 1 contributes only its original lexical seeds. Stage 2 adds
        lexical candidates retrieved with its expanded search terms and the
        top full-corpus fact and state embedding candidates. Association
        expansion is intentionally performed by the caller after direct
        scoring.
        """
        lexical_candidate_limits = dict(
            seed_channel_limits.get("stage2_lexical") or {}
        )
        embedding_candidate_limits = dict(
            seed_channel_limits.get("stage2_embedding") or {}
        )
        if search_terms:
            lexical_fact_candidates, lexical_state_candidates = (
                self._retrieve_recall_raw_candidates_lexical_search(
                    terms=list(search_terms),
                    candidate_source_prefix="stage2",
                    source_types=source_types,
                    temporal_bounds=temporal_bounds,
                    temporal_mode=temporal_mode,
                    candidate_limits=lexical_candidate_limits,
                    database=database,
                )
            )
        else:
            lexical_fact_candidates, lexical_state_candidates = [], []

        embedding_fact_candidates, embedding_state_candidates = (
            self._retrieve_recall_full_embedding_candidates(
                query_embedding=query_embedding,
                candidate_source_prefix="stage2",
                source_types=source_types,
                temporal_bounds=temporal_bounds,
                temporal_mode=temporal_mode,
                candidate_limits=embedding_candidate_limits,
                database=database,
            )
        )
        merged_report = self._merge_recall_stage2_seed_candidates(
            stage1_lexical_candidates=stage1_lexical_candidates,
            stage2_lexical_candidates=[
                *lexical_fact_candidates,
                *lexical_state_candidates,
            ],
            stage2_embedding_candidates=[
                *embedding_fact_candidates,
                *embedding_state_candidates,
            ],
        )
        return list(merged_report.get("merged_candidates") or [])

    def _log_recall_stage2_seed_candidates(
        self,
        *,
        seed_candidates: Sequence[Dict[str, Any]],
    ) -> None:
        """Log all Stage 2 seed channels before direct matching."""
        seed_candidates = list(seed_candidates or [])
        source_counts = {
            source: sum(
                1
                for candidate in seed_candidates
                if candidate.get("_recall_candidate_source") == source
            )
            for source in (
                "stage1_lexical",
                "stage2_lexical",
                "stage2_embedding",
            )
        }
        payload: Dict[str, Any] = {
            "seed_candidate_count": len(seed_candidates),
            "seed_by_level": self._recall_count_candidates_by_level(
                seed_candidates
            ),
            "stage1_lexical_seed_count": source_counts["stage1_lexical"],
            "stage2_lexical_seed_count": source_counts["stage2_lexical"],
            "stage2_embedding_seed_count": source_counts["stage2_embedding"],
            "facts": self._recall_log_candidate_items(
                [
                    candidate
                    for candidate in seed_candidates
                    if str(candidate.get("index_level") or "") == "fact"
                ],
                detailed=self._recall_detailed_logging,
                limit=None if self._recall_detailed_logging else 12,
                stage="stage2_seed",
            ),
            "states": self._recall_log_candidate_items(
                [
                    candidate
                    for candidate in seed_candidates
                    if str(candidate.get("index_level") or "") == "state"
                ],
                detailed=self._recall_detailed_logging,
                limit=None if self._recall_detailed_logging else 12,
                stage="stage2_seed",
            ),
        }
        self._log_info("memory_recall_stage2", "seeds_retrieved", payload)

    def _process_recall_stage2(
        self,
        *,
        original_query: str,
        time_stripped_query: str,
        temporal_bounds: RecallTimeBounds,
        reference_time: str,
        memory_source_override: Optional[Sequence[str]] = None,
        temporal_mode: str = "dialogue_time",
        stage1_report: Optional[Dict[str, Any]] = None,
        prompt_language: str = "zh",
        database: Optional[SessionDB] = None,
    ) -> str:
        """Run the existing LLM and semantic-search recall pipeline.

        This is intentionally separated from ``recall`` so a deterministic
        Stage 1 fast path can decide whether this more expensive path is
        necessary without duplicating its query preparation and logging.
        """
        stage_started_at = time.monotonic()
        time_stripped_query = str(time_stripped_query or "")
        fallback_temporal_bounds = temporal_bounds
        fallback_temporal_mode = self._normalize_recall_temporal_mode(
            temporal_mode
        )
        analysis_reference_time = str(reference_time)
        self._log_info("memory_recall_stage2", "start", {
            "original_query": self._format_log_text(original_query, limit=500),
            "time_stripped_query": self._format_log_text(
                time_stripped_query,
                limit=500,
            ),
            "top_k": self._top_k,
            "budget": self._recall_budget,
            "fallback_time_start": (fallback_temporal_bounds or (None, None))[0],
            "fallback_time_end": (fallback_temporal_bounds or (None, None))[1],
            "fallback_temporal_mode": fallback_temporal_mode,
            "reference_time": analysis_reference_time,
            "prompt_language": prompt_language,
            "memory_source_override": list(memory_source_override or []),
        })
        query_analysis_info = self._analyze_recall_query(
            original_query,
            reference_time=analysis_reference_time,
            prompt_language=prompt_language,
        )
        temporal_resolution = self._resolve_recall_stage2_temporal_constraints(
            original_query=original_query,
            fallback_temporal_bounds=fallback_temporal_bounds,
            fallback_temporal_mode=fallback_temporal_mode,
            llm_temporal_bounds=query_analysis_info.get("temporal_bounds"),
            llm_temporal_mode=query_analysis_info.get("temporal_mode"),
        )
        temporal_bounds = temporal_resolution["effective_temporal_bounds"]
        temporal_mode = temporal_resolution["effective_temporal_mode"]
        forced_source_types = self._normalize_source_override(memory_source_override)
        preferred_source_types = forced_source_types or self._normalize_source_override(
            query_analysis_info.get("source_types") or []
        )
        layer_preference = query_analysis_info.get("layer_preference")
        preferred_layer_preferences = self._normalize_recall_layer_preference(
            layer_preference or []
        )
        llm_keywords = self._normalize_string_list(
            query_analysis_info.get("keywords"),
            limit=12,
        )
        llm_entities = self._normalize_entity_names(
            query_analysis_info.get("entities"),
            limit=12,
        )
        query_entity_names = self._normalize_entity_names(
            [
                *llm_entities,
                *self._recall_stage1_resolve_query_entity_names(
                    query=original_query,
                    database=database,
                ),
            ],
            limit=24,
        )
        is_contextual_query = self._recall_stage1_is_contextual_query(
            original_query
        )
        reference_time = (
            (temporal_bounds or (None, None))[1]
            or analysis_reference_time
        )
        # Stage 1 contributes its direct lexical seeds as one Stage 2 source.
        # The Stage 2 lexical source itself is reserved for LLM-derived terms.
        supplement_terms = self._build_recall_search_terms(
            "",
            keywords=llm_keywords,
            entities=llm_entities,
        )
        query_terms = self._lexical_search_terms_for_text(
            time_stripped_query,
            limit=32,
            preserve_phrase=False,
        )
        retrieval_text = query_analysis_info.get("query_rewrite") or ""
        rewrite_terms = self._lexical_search_terms_for_text(
            retrieval_text,
            limit=32,
            preserve_phrase=False,
        )
        search_terms = list(dict.fromkeys([
            *supplement_terms,
            *rewrite_terms,
            *query_terms,
        ]))
        query_identity_text = self._format_recall_query_identity_text(
            time_stripped_query,
            retrieval_text=retrieval_text,
            keywords=llm_keywords,
            entities=llm_entities,
        )
        query_identity_embedding = self._generate_embedding_vector(query_identity_text)
        candidate_limits = self._recall_stage2_candidate_limits(
            top_k=self._top_k,
            preferred_layer_preferences=preferred_layer_preferences,
        )
        seed_channel_limits = candidate_limits["seed_channel_limits"]
        selected_candidate_limits = candidate_limits["selected_limits"]
        association_per_relation_limit = candidate_limits[
            "association_per_relation_limit"
        ]
        self._log_info("memory_recall_stage2", "query_analyzed", {
            "query_analysis_info": query_analysis_info,
            "temporal_resolution": temporal_resolution,
            "forced_source_types": forced_source_types or [],
            "preferred_source_types": preferred_source_types or [],
            "preferred_layer_preferences": preferred_layer_preferences or [],
            "keywords": llm_keywords,
            "entities": llm_entities,
            "query_entity_names": query_entity_names,
            "is_contextual_query": is_contextual_query,
            "reference_time": reference_time,
            "supplement_terms": supplement_terms,
            "query_terms": query_terms,
            "rewrite_terms": rewrite_terms,
            "search_terms": search_terms,
            "retrieval_text": self._format_log_text(retrieval_text, limit=500),
            "identity_text": self._format_log_text(query_identity_text, limit=500),
            "query_embedding_available": query_identity_embedding is not None,
            "candidate_limits": candidate_limits,
            "budget": self._recall_budget,
            "temporal_mode": temporal_mode,
        })
        stage1_lexical_seed_candidates = list(
            (stage1_report or {}).get("seed_candidates") or []
        )
        seed_candidates = self._retrieve_recall_stage2_seed_candidates(
            stage1_lexical_candidates=stage1_lexical_seed_candidates,
            search_terms=search_terms,
            query_embedding=query_identity_embedding,
            source_types=forced_source_types,
            temporal_bounds=temporal_bounds,
            temporal_mode=temporal_mode,
            seed_channel_limits=seed_channel_limits,
            database=database,
        )
        self._log_recall_stage2_seed_candidates(
            seed_candidates=seed_candidates,
        )
        direct_candidates = self._recall_stage2_calculate_candidate_matching_score(
            candidates=seed_candidates,
            search_terms=search_terms,
            query_terms=query_terms,
            query_embedding=query_identity_embedding,
            query_entity_names=query_entity_names,
            is_contextual_query=is_contextual_query,
            temporal_bounds=temporal_bounds,
        )
        self._log_recall_direct_candidates(
            stage_name="stage2",
            seed_candidates=seed_candidates,
            direct_candidates=direct_candidates,
        )
        association_candidates = (
            self._retrieve_association_candidates_using_seed_candidates(
                seed_candidates=direct_candidates,
                source_types=forced_source_types,
                temporal_bounds=temporal_bounds,
                temporal_mode=temporal_mode,
                limit=association_per_relation_limit,
                candidate_source_prefix="stage2",
                database=database,
            )
        )
        expanded_candidates = self._merge_recall_stage2_direct_and_associative_candidates(
            direct_candidates=direct_candidates,
            association_candidates=association_candidates,
            stage_name="stage2",
        )
        self._log_recall_association_candidates(
            stage_name="stage2",
            expanded_candidates=expanded_candidates,
            association_candidates=association_candidates,
        )

        ranked_candidates = self._recall_stage2_rank_and_select_candidates(
            candidates=expanded_candidates,
            final_candidate_limits=selected_candidate_limits,
        )
        self._log_recall_selected_candidates(
            stage_name="stage2",
            selected_candidates=ranked_candidates,
            expanded_candidates=expanded_candidates,
            actionable_candidates=[],
        )
        memory_text = self._build_memory_retrieved_format_text(
            entries=ranked_candidates,
            prompt_language=prompt_language,
        )
        self._log_info("memory_recall_stage2", "finish", {
            "status": "ok" if memory_text else "empty",
            "elapsed_ms": round((time.monotonic() - stage_started_at) * 1000, 2),
            "retrieved_chars": len(memory_text or ""),
        })
        return memory_text

    def _make_recall_memory_candidate(
        self,
        *,
        level: str,
        row: Dict[str, Any],
        candidate_source: str,
        supporting_facts: Optional[Sequence[Dict[str, Any]]] = None,
        temporal_bounds: RecallTimeBounds = None,
        temporal_mode: str = "dialogue_time",
    ) -> Optional[Dict[str, Any]]:
        """Convert a memory row into the shared recall candidate shape."""
        target_table = {
            "fact": "memory_facts",
            "state": "memory_states",
            "actionable_item": "memory_actionable_items",
        }.get(level)
        if not target_table:
            return None
        try:
            target_id = int(row.get("id"))
        except (TypeError, ValueError):
            return None
        source_type = row.get("source_type")
        support_facts = [dict(fact) for fact in supporting_facts or []]
        if level == "fact":
            title = _compact_whitespace(row.get("summary") or "")[:120]
            summary = _compact_whitespace(row.get("summary") or "")
            fact_times = self._fact_time_values(row, temporal_mode)
            if not self._fact_matches_time_bounds(
                row,
                temporal_mode=temporal_mode,
                temporal_bounds=temporal_bounds,
            ):
                return None
            time_value = fact_times[0] if fact_times else ""
            entities = row.get("entities") or []
            topics: List[str] = []
            keywords = row.get("keywords") or ""
            time_end_value = fact_times[-1] if fact_times else ""
        elif level == "state":
            title = _compact_whitespace(row.get("canonical_name") or "")
            summary = _compact_whitespace(row.get("summary") or "")
            time_value = self._normalize_event_time_text(row.get("updated_at"))
            state_metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            entities = self._normalize_entity_names([
                row.get("entity_key"),
                *(state_metadata.get("context_entities") or []),
                *(state_metadata.get("entities") or []),
            ], limit=18)
            topics = self._normalize_unique_labels([
                row.get("canonical_name"),
                *(state_metadata.get("canonical_topics") or []),
                *(state_metadata.get("parent_topics") or []),
                *(state_metadata.get("aspect_topic_names") or []),
            ], limit=24)
            keywords = " ".join(self._normalize_string_list(
                state_metadata.get("keywords"),
                limit=24,
            ))
            support_start, support_end = self._event_time_bounds_from_facts(
                support_facts,
                temporal_mode=temporal_mode,
            )
            if support_start or support_end:
                time_value = support_start or support_end
                time_end_value = support_end or support_start
            else:
                time_end_value = time_value
        elif level == "actionable_item":
            title = _compact_whitespace(row.get("canonical_name") or "")
            summary = _compact_whitespace(row.get("summary") or "")
            time_value = self._normalize_event_time_text(
                row.get("due_at") or row.get("updated_at") or row.get("created_at")
            )
            entities = [row.get("owner")] if row.get("owner") else []
            topics = [row.get("canonical_name")] if row.get("canonical_name") else []
            keywords = ""
            support_start, support_end = self._event_time_bounds_from_facts(
                support_facts,
                temporal_mode=temporal_mode,
            )
            if support_start or support_end:
                time_value = support_start or support_end
                time_end_value = support_end or support_start
            else:
                time_end_value = time_value
        else:
            return None

        if level == "fact":
            time_start, time_end = temporal_bounds or (None, None)
            if time_start and time_end_value and time_end_value < str(time_start):
                return None
            if time_end and time_value and time_value > str(time_end):
                return None

        hydrated = dict(row)
        if level == "fact":
            hydrated.pop("episode_id", None)
        hydrated.pop("embedding", None)
        hydrated.pop("identity_text_embedding", None)
        hydrated.pop("canonical_name_embedding", None)
        metadata = dict(row.get("metadata") or {})
        metadata["_matched_via"] = [candidate_source]
        candidate = {
            "source_type": source_type,
            "target_table": target_table,
            "target_id": target_id,
            "index_level": level,
            "memory_path": f"{source_type}/{level}",
            "title": title,
            "summary_for_retrieval": summary,
            "identity_text": _compact_whitespace(row.get("identity_text") or ""),
            "keywords": keywords,
            "entities": entities,
            "participants": row.get("participants") or [],
            "time_start": time_value,
            "time_end": time_end_value,
            "importance": row.get("importance") or 0.5,
            "confidence": row.get("confidence") or 0.8,
            "embedding": row.get("identity_text_embedding"),
            "metadata": metadata,
            "_hydrated": hydrated,
            "_supporting_facts": support_facts,
            "_bm25_score": row.get("_bm25_score"),
            "_recall_candidate_source": candidate_source,
        }
        if level != "fact":
            candidate["canonical_topics"] = topics
        return candidate

    def _retrieve_recall_full_embedding_candidates(
        self,
        *,
        query_embedding: Optional[np.ndarray],
        candidate_source_prefix: str,
        source_types: Optional[Sequence[str]],
        temporal_bounds: RecallTimeBounds,
        temporal_mode: str,
        candidate_limits: Dict[str, int],
        database: Optional[SessionDB] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Rank all local fact/state embeddings and retain per-layer seeds.

        Stage 2 currently runs against a bounded personal-memory corpus, so
        this intentionally evaluates every persisted identity embedding rather
        than restricting semantic retrieval to lexical hits first.
        """
        if query_embedding is None:
            return [], []
        db = database or self._db
        rows_by_level = {
            "fact": db.memory_facts_with_identity_embeddings(
                source_types=source_types,
            ),
            "state": db.memory_states_with_identity_embeddings(
                source_types=source_types,
                state_scope="entity_state",
            ),
        }
        minimum_similarity_by_level = {
            "fact": self._recall_stage2_fact_min_embedding_similarity,
            "state": self._recall_stage2_state_min_embedding_similarity,
        }
        candidates_by_level: Dict[str, List[Dict[str, Any]]] = {
            "fact": [],
            "state": [],
        }
        for level, rows in rows_by_level.items():
            minimum_similarity = minimum_similarity_by_level[level]
            for row in rows:
                candidate = self._make_recall_memory_candidate(
                    level=level,
                    row=row,
                    candidate_source=(
                        f"{str(candidate_source_prefix).strip()}_embedding"
                    ),
                    temporal_bounds=temporal_bounds,
                    temporal_mode=temporal_mode,
                )
                if not candidate:
                    continue
                similarity = max(0.0, _cal_embedding_cosine_similarity(
                    query_embedding,
                    candidate.get("embedding"),
                ))
                if similarity < minimum_similarity:
                    continue
                candidate["_recall_embedding_seed_similarity"] = round(
                    float(similarity),
                    4,
                )
                candidates_by_level[level].append(candidate)
            candidates_by_level[level].sort(
                key=lambda item: (
                    float(item.get("_recall_embedding_seed_similarity") or 0.0),
                    str(item.get("time_start") or ""),
                    int(item.get("target_id") or 0),
                ),
                reverse=True,
            )
            candidates_by_level[level] = candidates_by_level[level][
                : max(0, int(candidate_limits.get(level, 0) or 0))
            ]
        return (
            candidates_by_level["fact"],
            candidates_by_level["state"],
        )

    def _retrieve_recall_raw_candidates_lexical_search(
        self,
        *,
        terms: List[str],
        candidate_source_prefix: str,
        source_types: Optional[Sequence[str]],
        temporal_bounds: RecallTimeBounds,
        temporal_mode: str,
        candidate_limits: Dict[str, int],
        database: Optional[SessionDB] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build separate raw candidate groups before type-specific ranking.

        This intentionally bypasses `memory_index_entries` for the default
        path. Each candidate still points to its source row; relationship
        expansion is handled by the Stage 1/Stage 2 merge layer.
        """
        db = database or self._db
        time_start, time_end = temporal_bounds or (None, None)
        level_specs = (
            ("fact", db.search_memory_facts),
            ("state", db.search_memory_states),
        )
        rows_by_level: Dict[str, List[Dict[str, Any]]] = {}
        for level, loader in level_specs:
            loader_kwargs = {
                "terms": terms,
                "source_types": source_types,
                # States and actionable items do not have an event-time
                # column, so their own update/due time is used below.
                "time_start": time_start if level == "fact" else None,
                "time_end": time_end if level == "fact" else None,
                "limit": max(1, int(candidate_limits.get(level, 1) or 1)),
            }
            if level == "fact":
                loader_kwargs["temporal_mode"] = temporal_mode
            else:
                loader_kwargs["state_scope"] = "entity_state"
            rows_by_level[level] = loader(**loader_kwargs)

        candidates_by_level: Dict[str, List[Dict[str, Any]]] = {
            "fact": [],
            "state": [],
        }
        for level, _loader in level_specs:
            for row in rows_by_level[level]:
                candidate = self._make_recall_memory_candidate(
                    level=level,
                    row=row,
                    candidate_source=(
                        f"{str(candidate_source_prefix).strip()}_lexical"
                    ),
                    temporal_bounds=temporal_bounds,
                    temporal_mode=temporal_mode,
                )
                if candidate:
                    candidates_by_level[level].append(candidate)

        # SQLite FTS5 BM25 returns lower (normally negative) values for more
        # relevant documents. Its magnitude is table-dependent, so normalize
        # only among the BM25 hits in each returned memory layer.
        for level_candidates in candidates_by_level.values():
            scored_candidates: List[Tuple[float, Dict[str, Any]]] = []
            for candidate in level_candidates:
                try:
                    raw_bm25_score = float(candidate.get("_bm25_score"))
                except (TypeError, ValueError):
                    candidate["_recall_bm25_score"] = 0.0
                    continue
                if not math.isfinite(raw_bm25_score):
                    candidate["_recall_bm25_score"] = 0.0
                    continue
                scored_candidates.append((raw_bm25_score, candidate))
            scored_candidates.sort(key=lambda item: item[0])
            count = len(scored_candidates)
            for position, (_raw_bm25_score, candidate) in enumerate(scored_candidates):
                normalized_bm25_score = (
                    0.80
                    if count == 1
                    else 0.42 + 0.50 * (1.0 - position / (count - 1))
                )
                candidate["_recall_bm25_score"] = round(normalized_bm25_score, 4)
                candidate["_recall_bm25_rank"] = position + 1
        self._logger.debug(
            "Direct recall candidates: facts=%d states=%d total=%d",
            len(rows_by_level["fact"]),
            len(rows_by_level["state"]),
            sum(len(items) for items in candidates_by_level.values()),
        )
        return (
            candidates_by_level["fact"],
            candidates_by_level["state"],
        )

    def _recall_stage2_calculate_single_candidate_matching_score(
        self,
        candidate: Dict[str, Any],
        *,
        memory_type: str,
        search_terms: Sequence[str],
        query_terms: Sequence[str],
        query_embedding: Optional[np.ndarray],
        query_entity_names: Sequence[str] = (),
        is_contextual_query: bool = False,
        temporal_bounds: RecallTimeBounds = None,
    ) -> Dict[str, Any]:
        """Calculate matching score for one Stage 2 direct candidate only."""
        search_topic_match_info = (
            self._recall_calculate_search_terms_overlap_with_candidate_topics(
                candidate,
                search_terms,
                allow_substring=True,
            )
        )
        query_topic_match_info = (
            self._recall_calculate_search_terms_overlap_with_candidate_topics(
                candidate,
                query_terms,
                allow_substring=True,
            )
        )
        all_topic_values = list(
            search_topic_match_info.get("topic_values") or []
        )
        topic_match_ratio = float(
            search_topic_match_info.get("coverage") or 0.0
        )
        keyword_match_info = (
            self._recall_calculate_search_terms_overlap_with_candidate_keywords(
                candidate,
                search_terms,
                allow_substring=True,
            )
        )
        all_keyword_values = list(keyword_match_info.get("keyword_values") or [])
        keyword_match_ratio = float(keyword_match_info.get("coverage") or 0.0)
        similarity = max(0.0, _cal_embedding_cosine_similarity(
            query_embedding,
            candidate.get("embedding"),
        ))
        bm25_score = self._clamp_float(
            candidate.get("_recall_bm25_score"),
            0.0,
            1.0,
            0.0,
        )
        entity_match_info = self._recall_stage1_matched_entity_names(
            query_entity_names=query_entity_names,
            candidate_entity_names=candidate.get("entities") or [],
        )
        high_value_entity_matched = bool(
            entity_match_info.get("high_value_entity_matched")
        )
        entity_strong_anchor = bool(
            entity_match_info.get("entity_strong_anchor")
        )
        time_score_info = self._calculate_recall_candidate_time_score(
            candidate,
            temporal_bounds=temporal_bounds,
        )
        time_weight = self._recall_stage2_time_score_weight
        if is_contextual_query:
            time_weight *= self._recall_stage2_contextual_time_score_multiplier
        time_component_score = time_weight * float(
            time_score_info.get("time_score") or 0.0
        )

        strong_embedding_threshold = {
            "fact": self._recall_stage2_fact_strong_embedding_similarity,
            "state": self._recall_stage2_state_strong_embedding_similarity,
        }.get(memory_type, 1.0)
        embedding_strong_anchor = bool(
            query_embedding is not None
            and candidate.get("embedding") is not None
            and similarity >= strong_embedding_threshold
        )
        best_topic_pair_score = float(
            query_topic_match_info.get("best_pair_score") or 0.0
        )
        term_strong_anchor = bool(
            int(query_topic_match_info.get("matched_term_count") or 0) > 0
            and best_topic_pair_score
            >= self._recall_stage2_strong_topic_pair_score
        )
        keyword_strong_anchor = bool(
            memory_type == "fact"
            and int(keyword_match_info.get("matched_term_count") or 0) > 0
            and float(keyword_match_info.get("best_pair_score") or 0.0)
            >= self._recall_stage2_strong_topic_pair_score
        )
        strong_anchor_reasons: List[str] = []
        if embedding_strong_anchor:
            strong_anchor_reasons.append("embedding_similarity")
        if term_strong_anchor:
            strong_anchor_reasons.append("topic_match")
        if keyword_strong_anchor:
            strong_anchor_reasons.append("keyword_match")
        if entity_strong_anchor:
            strong_anchor_reasons.append("entity_match")
        has_strong_anchor = bool(strong_anchor_reasons)

        embedding_score = (
            self._recall_stage2_embedding_score_weight * similarity
            if query_embedding is not None and candidate.get("embedding") is not None
            else 0.0
        )
        topic_match_score = (
            self._recall_stage2_topic_overlap_score_weight * topic_match_ratio
            if search_terms and all_topic_values
            else 0.0
        )
        keyword_match_score = (
            self._recall_stage2_keyword_match_score_weight * keyword_match_ratio
            if search_terms and all_keyword_values
            else 0.0
        )
        bm25_component_score = (
            self._recall_stage2_bm25_score_weight * bm25_score
        )
        entity_matching_score = (
            self._recall_stage2_entity_matched_score
            if high_value_entity_matched
            else 0.0
        )
        score = self._clamp_float(
            embedding_score
            + topic_match_score
            + keyword_match_score
            + bm25_component_score
            + entity_matching_score
            + time_component_score,
            0.0,
            1.0,
            0.0,
        )
        score_components = {
            "embedding_score": round(float(embedding_score), 4),
            "topic_match_score": round(float(topic_match_score), 4),
            "keyword_match_score": round(float(keyword_match_score), 4),
            "bm25_component_score": round(float(bm25_component_score), 4),
            "entity_matching_score": round(float(entity_matching_score), 4),
            "time_component_score": round(float(time_component_score), 4),
        }
        stage2_match_details = {
            "topic_match_info": {
                **search_topic_match_info,
                "query_anchor_match_info": dict(query_topic_match_info),
                "keyword_match_info": dict(keyword_match_info),
            },
            "entity_match_info": dict(entity_match_info),
            "time_score_info": dict(time_score_info),
            "score_components": dict(score_components),
        }
        return {
            "score": round(float(score), 4),
            "matched": has_strong_anchor,
            "filter_reason": "" if has_strong_anchor else "no_strong_anchor",
            "embedding_similarity": round(float(similarity), 4),
            "contextual_time_weight_applied": bool(is_contextual_query),
            "has_strong_anchor": has_strong_anchor,
            "strong_anchor_reasons": strong_anchor_reasons,
            "_recall_stage2_match_details": stage2_match_details,
        }

    @staticmethod
    def _recall_candidate_evidence_fact_ids(
        candidate: Dict[str, Any],
    ) -> set[int]:
        """Return the persisted evidence fact ids attached to a candidate."""
        raw = (
            candidate.get("_hydrated")
            if isinstance(candidate.get("_hydrated"), dict)
            else {}
        )
        values = candidate.get("evidence_fact_ids")
        if values is None:
            values = raw.get("evidence_fact_ids")
        if isinstance(values, str):
            try:
                values = json.loads(values)
            except (TypeError, ValueError, json.JSONDecodeError):
                values = []
        if values is None:
            return set()
        if not isinstance(values, (list, tuple, set)):
            values = [values]
        evidence_fact_ids: set[int] = set()
        for value in values:
            try:
                evidence_fact_ids.add(int(value))
            except (TypeError, ValueError):
                continue
        return evidence_fact_ids

    def _assemble_recall_evidence_candidates(
        self,
        *,
        ranked_fact_candidates: Sequence[Dict[str, Any]],
        ranked_state_candidates: Sequence[Dict[str, Any]],
        layer_limits: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """Assemble ranked candidates by removing state/action duplicates.

        Facts are already ranked and deduplicated before this step, so they
        establish the evidence baseline. A state or actionable item whose
        persisted evidence_fact_ids are all present in the selected facts is
        omitted as a derived duplicate. Candidates without evidence ids are
        retained conservatively because coverage cannot be proven.
        """
        limits = {
            str(layer): max(0, int(limit or 0))
            for layer, limit in (layer_limits or {}).items()
        }
        selected_candidates: List[Dict[str, Any]] = []
        selected_fact_ids: set[int] = set()

        fact_limit = limits.get("fact", len(ranked_fact_candidates))
        for candidate in ranked_fact_candidates:
            if sum(
                1
                for item in selected_candidates
                if str(item.get("index_level") or "") == "fact"
            ) >= fact_limit:
                break
            selected_candidates.append(candidate)
            try:
                selected_fact_ids.add(int(candidate.get("target_id")))
            except (TypeError, ValueError):
                continue

        def append_uncovered_candidates(
            candidates: Sequence[Dict[str, Any]],
            *,
            layer: str,
        ) -> None:
            layer_limit = limits.get(layer, len(candidates))
            selected_count = sum(
                1
                for item in selected_candidates
                if str(item.get("index_level") or "") == layer
            )
            for candidate in candidates:
                if selected_count >= layer_limit:
                    break
                evidence_fact_ids = self._recall_candidate_evidence_fact_ids(
                    candidate
                )
                if evidence_fact_ids and evidence_fact_ids.issubset(
                    selected_fact_ids
                ):
                    continue
                selected_candidates.append(candidate)
                selected_count += 1

        append_uncovered_candidates(
            ranked_state_candidates,
            layer="state",
        )
        return selected_candidates

    @classmethod
    def _normalize_llm_recall_time_bound(cls, value: Any) -> Optional[str]:
        """Validate one LLM-supplied recall bound and normalize it to ISO."""
        normalized = cls._normalize_recall_time_bound(value, default_to_now=False)
        if not normalized:
            return None
        try:
            return datetime.strptime(normalized, "%Y-%m-%d %H:%M:%S").strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except ValueError:
            return None

    @staticmethod
    def _recall_query_has_explicit_calendar_date(query: str) -> bool:
        """Return whether the query states a concrete calendar date/range."""
        text = str(query or "")
        return bool(re.search(
            r"\d{4}\s*(?:年|[-/.])\s*\d{1,2}"
            r"|\d{1,2}\s*月\s*\d{1,2}\s*(?:日|号)?",
            text,
        ))

    def _resolve_recall_stage2_temporal_constraints(
        self,
        *,
        original_query: str,
        fallback_temporal_bounds: RecallTimeBounds,
        fallback_temporal_mode: str,
        llm_temporal_bounds: Any,
        llm_temporal_mode: Any,
    ) -> Dict[str, Any]:
        """Validate Stage 2 temporal analysis and choose effective constraints."""
        fallback_start, fallback_end = fallback_temporal_bounds or (None, None)
        fallback_bounds = (
            self._normalize_llm_recall_time_bound(fallback_start),
            self._normalize_llm_recall_time_bound(fallback_end),
        )
        raw_llm_bounds = (
            llm_temporal_bounds if isinstance(llm_temporal_bounds, dict) else {}
        )
        llm_bounds = (
            self._normalize_llm_recall_time_bound(raw_llm_bounds.get("start")),
            self._normalize_llm_recall_time_bound(raw_llm_bounds.get("end")),
        )
        llm_bounds_valid = bool(llm_bounds[0] or llm_bounds[1])
        if llm_bounds_valid and all(llm_bounds):
            llm_bounds_valid = bool(llm_bounds[0] < llm_bounds[1])

        fallback_has_bounds = bool(fallback_bounds[0] or fallback_bounds[1])
        explicit_rule_bounds = bool(
            fallback_has_bounds
            and self._recall_query_has_explicit_calendar_date(original_query)
        )
        if llm_bounds_valid and not explicit_rule_bounds:
            effective_bounds = llm_bounds
            bounds_source = "llm"
        elif fallback_has_bounds:
            effective_bounds = fallback_bounds
            bounds_source = (
                "rule_conflict_override"
                if llm_bounds_valid and explicit_rule_bounds
                else "rule"
            )
        else:
            effective_bounds = (None, None)
            bounds_source = "none"

        llm_mode = self._normalize_recall_temporal_mode(llm_temporal_mode)
        fallback_mode = self._normalize_recall_temporal_mode(fallback_temporal_mode)
        effective_has_bounds = bool(effective_bounds[0] or effective_bounds[1])
        if effective_has_bounds:
            if llm_mode in {"event_time", "dialogue_time", "both"}:
                effective_mode = llm_mode
                mode_source = "llm"
            elif fallback_mode in {"event_time", "dialogue_time", "both"}:
                effective_mode = fallback_mode
                mode_source = "rule"
            else:
                effective_mode = "both"
                mode_source = "bounds_default"
        else:
            effective_mode = llm_mode if llm_mode != "none" else fallback_mode
            mode_source = "llm" if llm_mode != "none" else "rule"

        return {
            "fallback_temporal_bounds": fallback_bounds,
            "fallback_temporal_mode": fallback_mode,
            "llm_temporal_bounds": llm_bounds,
            "llm_temporal_bounds_valid": llm_bounds_valid,
            "llm_temporal_mode": llm_mode,
            "effective_temporal_bounds": effective_bounds,
            "effective_temporal_mode": effective_mode,
            "temporal_bounds_source": bounds_source,
            "temporal_mode_source": mode_source,
        }

    def _analyze_recall_query(
        self,
        query: str,
        *,
        reference_time: str,
        prompt_language: str,
    ) -> Dict[str, Any]:
        prompt_language = (
            "en"
            if str(prompt_language or "").strip().lower().startswith("en")
            else "zh"
        )
        prompt_template = (
            RECALL_QUERY_ANALYSIS_PROMPT_EN
            if prompt_language == "en"
            else RECALL_QUERY_ANALYSIS_PROMPT_ZH
        )
        prompt = (
            prompt_template
            .replace("{query}", str(query or ""))
            .replace("{reference_time}", str(reference_time or ""))
        )
        result = self._call_llm(prompt)
        parsed = self._parse_json_object_from_llm_text(result or "")
        return parsed if isinstance(parsed, dict) else {}

    def _build_recall_search_terms(
        self,
        query: str,
        *,
        keywords: Sequence[str],
        entities: Sequence[str],
    ) -> List[str]:
        """Build the shared, already-tokenized lexical query representation."""
        terms: List[str] = []
        seen: set[str] = set()

        def add_terms(values: Sequence[str]) -> None:
            for value in values:
                clean = re.sub(r"\s+", " ", str(value or "").strip()).lower()
                if not clean or clean in seen:
                    continue
                if len(clean) > 80 or re.search(r"[。！？!?；;，,]", clean):
                    continue
                seen.add(clean)
                terms.append(clean)
                if len(terms) >= 32:
                    return

        for value in [*keywords, *entities]:
            add_terms(self._lexical_search_terms_for_text(value))
            if len(terms) >= 32:
                break
        if len(terms) < 32:
            add_terms(
                self._lexical_search_terms_for_text(
                    query,
                    limit=32,
                    preserve_phrase=False,
                )
            )
        return terms

    @staticmethod
    def _recall_entity_lookup_aliases(query: Any) -> List[str]:
        """Return canonical role entities implied by first/second-person text.

        Memory entities use stable role names (``用户``/``助手`` or
        ``user``/``assistant``), while recall questions naturally use
        pronouns such as ``我`` and ``你``. These aliases are only used to
        query the entity-node index; the original query remains unchanged for
        time parsing, lexical search, scoring, and LLM analysis.
        """
        text = str(query or "")
        aliases: List[str] = []

        # Do not treat ``我们``/``你们`` as a single speaker role. The
        # conversational schema models the direct user and assistant roles
        # separately.
        is_chinese = bool(re.search(r"[\u4e00-\u9fff]", text))
        if re.search(r"我(?!们)", text) or re.search(
            r"\b(?:i|me|my|mine|myself)\b", text, re.IGNORECASE
        ):
            aliases.append("用户" if is_chinese else "user")
        if re.search(r"你(?!们)", text) or re.search(
            r"\b(?:you|your|yours|yourself)\b", text, re.IGNORECASE
        ):
            aliases.append("助手" if is_chinese else "assistant")
        return list(dict.fromkeys(aliases))

    def _lexical_search_terms_for_text(
        self,
        text: Any,
        *,
        limit: int = 32,
        preserve_phrase: bool = True,
    ) -> List[str]:
        """Tokenize one lexical value for the database FTS contract.

        The regular jieba tokens and search-mode sub-tokens mirror the stream
        used when ``memory_database`` builds ``lexical_index_text``. A
        whitespace-joined regular-token phrase is retained before individual
        tokens so short topic phrases remain searchable as a unit.
        """
        clean_text = _compact_whitespace(text)
        if not clean_text:
            return []
        values: List[str] = []
        seen: set[str] = set()

        def add(value: Any) -> None:
            if len(values) >= max(1, int(limit or 32)):
                return
            clean = re.sub(r"\s+", " ", str(value or "").strip()).lower()
            if not clean or clean in seen:
                return
            if len(clean) > 80 or re.search(r"[。！？!?；;，,]", clean):
                return
            chinese_count = len(re.findall(r"[\u4e00-\u9fff]", clean))
            if chinese_count and chinese_count < 2 and len(clean) < 2:
                return
            if not chinese_count and len(clean) < 2:
                return
            seen.add(clean)
            values.append(clean)

        chinese_text = "".join(re.findall(r"[\u4e00-\u9fff]", clean_text))
        if jieba is not None and chinese_text:
            regular_tokens = [
                _compact_whitespace(token)
                for token in jieba.lcut(clean_text, HMM=False)
            ]
            regular_tokens = [
                token for token in regular_tokens
                if token and re.search(r"[0-9a-zA-Z\u4e00-\u9fff]", token)
            ]
            if preserve_phrase and len(regular_tokens) > 1:
                add(" ".join(regular_tokens))
            for token in regular_tokens:
                add(token)
            for token in jieba.cut_for_search(clean_text, HMM=False):
                add(token)
            return values

        if preserve_phrase and not chinese_text:
            add(clean_text)

        # Minimal-install fallback: keep the complete Chinese run and the
        # same bigram coverage used by the legacy lexical path.
        for token in re.findall(
            r"[A-Za-z][A-Za-z0-9_.$'-]*|\d+(?:/\d+)?|[\u4e00-\u9fff]+",
            clean_text,
        ):
            if preserve_phrase or not re.fullmatch(r"[\u4e00-\u9fff]+", token):
                add(token)
            if re.fullmatch(r"[\u4e00-\u9fff]+", token):
                for index in range(len(token) - 1):
                    add(token[index : index + 2])
        return values

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

    def _normalize_recall_layer_preference(self, value: Optional[Sequence[str]]) -> Optional[List[str]]:
        if not value:
            return None
        allowed = {"episode", "fact", "state", "actionable_item"}
        out: List[str] = []
        for item in value:
            text = str(item or "").strip().lower()
            if text in allowed and text not in out:
                out.append(text)
        return out or None
    
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

    @staticmethod
    def _normalize_recall_temporal_mode(value: Any) -> str:
        mode = str(value or "").strip().lower()
        aliases = {
            "event": "event_time",
            "event-time": "event_time",
            "dialogue": "dialogue_time",
            "dialogue-time": "dialogue_time",
            "conversation": "dialogue_time",
            "all": "both",
        }
        mode = aliases.get(mode, mode)
        return mode if mode in {"event_time", "dialogue_time", "both", "none"} else "none"

    @classmethod
    def _infer_recall_temporal_mode(cls, query: str) -> str:
        text = _compact_whitespace(query).lower()
        dialogue_markers = (
            "讨论", "聊", "提到", "问过", "说过", "谈过", "对话", "交流",
            "讨论了", "提及", "discuss", "talk", "mentioned", "asked",
            "conversation", "chat", "said",
        )
        event_markers = (
            "做了", "做过", "发生", "买了", "买过", "去过", "参加", "完成",
            "经历", "遇到", "使用过", "发生了什么", "what happened", "did",
            "bought", "visited", "attended", "completed", "experienced",
        )
        has_dialogue = any(marker in text for marker in dialogue_markers)
        has_event = any(marker in text for marker in event_markers)
        if has_dialogue and has_event:
            return "both"
        if has_event:
            return "event_time"
        if has_dialogue:
            return "dialogue_time"
        return "none"

    @classmethod
    def _fact_time_values(
        cls,
        fact: Dict[str, Any],
        temporal_mode: str,
    ) -> List[str]:
        mode = cls._normalize_recall_temporal_mode(temporal_mode)
        event_time = cls._normalize_event_time_text(fact.get("event_time_key"))
        dialogue_time = cls._normalize_event_time_text(fact.get("dialogue_time_key"))
        if mode == "event_time":
            return [event_time] if event_time else []
        if mode == "dialogue_time":
            return [dialogue_time] if dialogue_time else []
        if mode == "both":
            return sorted({value for value in (event_time, dialogue_time) if value})
        return []

    @classmethod
    def _fact_matches_time_bounds(
        cls,
        fact: Dict[str, Any],
        *,
        temporal_mode: str,
        temporal_bounds: RecallTimeBounds,
    ) -> bool:
        time_start, time_end = temporal_bounds or (None, None)
        mode = cls._normalize_recall_temporal_mode(temporal_mode)
        if mode == "none" or not (time_start or time_end):
            return True
        values = cls._fact_time_values(fact, mode)
        if not values:
            return False
        return any(
            (not time_start or value >= str(time_start))
            and (not time_end or value <= str(time_end))
            for value in values
        )

    @classmethod
    def _event_time_bounds_from_facts(
        cls,
        facts: Sequence[Dict[str, Any]],
        *,
        temporal_mode: str = "event_time",
    ) -> Tuple[str, str]:
        times = [
            value
            for fact in facts or []
            for value in cls._fact_time_values(fact, temporal_mode)
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
                self._normalize_event_time_text(raw.get("event_time_key"))
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
            self._normalize_event_time_text(raw.get("event_time_key"))
            or self._normalize_event_time_text(raw.get("started_at"))
            or self._normalize_event_time_text(entry.get("time_start"))
            or self._normalize_event_time_text(entry.get("time_end"))
            or "unknown-event-time"
        )

    def _build_memory_retrieved_format_text(
        self,
        *,
        entries: List[Dict[str, Any]],
        prompt_language: str,
    ) -> str:
        """Format ranked raw memories with source-specific semantic fields."""
        if not entries:
            return ""

        is_en = str(prompt_language or "").strip().lower().startswith("en")
        format_template = (
            MEMORY_RETRIEVED_FORMAT_PROMPT_EN
            if is_en
            else MEMORY_RETRIEVED_FORMAT_PROMPT_ZH
        )
        section_specs = (
            MEMORY_RETRIEVED_SECTION_SPECS_EN
            if is_en
            else MEMORY_RETRIEVED_SECTION_SPECS_ZH
        )
        note_prefix = "System note: " if is_en else "系统说明："
        if is_en:
            labels = {
                "fact": "narrative fact",
                "dialogue_time": "dialogue_time",
                "event_time": "event_time",
                "summary": "summary",
                "fact_root_topic": "fact_root_topic",
                "fact_aspect_topic": "fact_aspect_topic",
                "state": "long-term state",
                "state_scope": "state_scope",
                "state_type": "state_type",
                "canonical_name": "canonical_name",
                "entity": "entity",
                "timeline": "timeline",
                "actionable_item": "actionable item",
                "item_type": "item_type",
                "status": "status",
                "owner": "owner",
                "due_at": "due_at",
            }
        else:
            labels = {
                "fact": "叙事事实",
                "dialogue_time": "对话时间",
                "event_time": "事件时间",
                "summary": "摘要",
                "fact_root_topic": "事实根主题",
                "fact_aspect_topic": "事实方面主题",
                "state": "长期状态",
                "state_scope": "状态范围",
                "state_type": "状态类型",
                "canonical_name": "规范名称",
                "entity": "实体",
                "timeline": "时间线",
                "actionable_item": "行动事项",
                "item_type": "事项类型",
                "status": "状态",
                "owner": "负责人",
                "due_at": "截止时间",
            }

        grouped = {
            "state": [entry for entry in entries if entry.get("index_level") == "state"],
            "actionable_item": [
                entry for entry in entries
                if entry.get("index_level") == "actionable_item"
            ],
            "fact": [entry for entry in entries if entry.get("index_level") == "fact"],
        }
        sections: List[str] = []
        for title, note, group_key in section_specs:
            group = grouped[group_key]
            if not group:
                continue
            section_lines = [title, f"{note_prefix}{note}"]
            for index, entry in enumerate(group, 1):
                raw = entry.get("_hydrated") if isinstance(entry.get("_hydrated"), dict) else {}
                time_text = self._recall_event_time_text(entry, raw)
                if group_key == "fact":
                    dialogue_time = (
                        self._normalize_event_time_text(raw.get("dialogue_time_key"))
                        or "unknown-dialogue-time"
                    )
                    event_time = (
                        self._normalize_event_time_text(raw.get("event_time_key"))
                        or "unknown-event-time"
                    )
                    block_lines = [
                        f"{index}. {labels['fact']}",
                        f"   {labels['dialogue_time']}: {dialogue_time}",
                        f"   {labels['event_time']}: {event_time}",
                        f"   {labels['summary']}: {raw.get('summary') or entry.get('summary_for_retrieval') or ''}",
                        f"   {labels['fact_root_topic']}: {raw.get('fact_root_topic') or ''}; {labels['fact_aspect_topic']}: {raw.get('fact_aspect_topic') or ''}",
                    ]
                elif group_key == "state":
                    timeline = self._format_state_timeline(raw.get("time_line"))
                    block_lines = [
                        f"{index}. [{time_text}] {labels['state']}",
                        f"   {labels['state_scope']}: {raw.get('state_scope') or ''}; {labels['state_type']}: {raw.get('state_type') or ''}",
                        f"   {labels['canonical_name']}: {raw.get('canonical_name') or ''}",
                        f"   {labels['entity']}: {raw.get('entity_key') or ''}",
                        f"   {labels['summary']}: {raw.get('summary') or ''}",
                    ]
                    if timeline:
                        block_lines.append(f"   {labels['timeline']}: {timeline}")
                else:
                    block_lines = [
                        f"{index}. [{time_text}] {labels['actionable_item']}",
                        f"   {labels['item_type']}: {raw.get('item_type') or ''}; {labels['status']}: {raw.get('status') or ''}",
                        f"   {labels['canonical_name']}: {raw.get('canonical_name') or ''}",
                        f"   {labels['owner']}: {raw.get('owner') or ''}; {labels['due_at']}: {raw.get('due_at') or ''}",
                        f"   {labels['summary']}: {raw.get('summary') or ''}",
                    ]
                section_lines.append("\n".join(block_lines))
            sections.append("\n".join(section_lines))
        return format_template.replace(
            "{memory_sections}",
            "\n\n".join(sections),
        ).strip()
    
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
    
    # ── Lightweight NLP heuristics ───────────────────────────────────────

    def _generate_embedding_vector(self, text: str) -> Optional[np.ndarray]:
        self._ensure_embedding_client()
        return self._embedding_client.embed_text(text) if self._embedding_client else None

    def _format_recall_query_identity_text(
        self,
        query: str,
        *,
        retrieval_text: str = "",
        keywords: Optional[Sequence[str]] = None,
        entities: Optional[Sequence[str]] = None,
    ) -> str:
        terms = list(keywords or [])
        if not terms:
            terms = self._lexical_search_terms_for_text(
                query,
                limit=32,
                preserve_phrase=False,
            )
        parts = [str(query or "").strip()]
        if retrieval_text and str(retrieval_text).strip() != str(query or "").strip():
            parts.append(f"retrieval: {str(retrieval_text).strip()}")
        if terms:
            parts.append(f"keywords: {' '.join(str(item) for item in terms)}")
        if entities:
            parts.append(f"entities: {' '.join(str(item) for item in entities)}")
        return "\n".join(parts)

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
            if not MemoryNodeManager._is_valid_entity_name(text) or text in seen:
                continue
            seen.add(text)
            out.append(text)
            if len(out) >= limit:
                break
        return out

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

    @classmethod
    def _normalize_primary_entity(
        cls,
        value: Any,
        *,
        entities: Sequence[str],
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
            if entities:
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

    def _recall_stage2_candidate_limits(
        self,
        *,
        top_k: int,
        preferred_layer_preferences: Optional[Sequence[str]],
    ) -> Dict[str, Any]:
        """Build explicit Stage 2 retrieval, expansion, and output budgets.

        Stage 2 broadens retrieval through three independent seed channels,
        while keeping final context compact. Layer preference reallocates the
        fixed fact/state output budget instead of increasing it.
        """
        k = max(1, int(top_k or 1))
        preferred = set(preferred_layer_preferences or [])
        supplementary_limit = max(1, int(math.ceil(k / 2)))
        reallocation = max(1, int(math.ceil(k / 4)))
        state_preferred = "state" in preferred
        fact_limit = k
        state_limit = supplementary_limit
        if state_preferred:
            # Entity-state queries may trade some fact quota for durable
            # entity evidence without growing the final context.
            fact_limit = max(1, fact_limit - reallocation)
            state_limit += reallocation

        # Stage 2's LLM-expanded lexical and full-embedding channels need a
        # little more depth than the final fact quota to surface semantic
        # alternatives before direct scoring.
        seed_per_channel_limit = max(6, min(16, int(math.ceil(k * 1.5))))
        seed_limits = {
            "fact": seed_per_channel_limit,
            "state": seed_per_channel_limit,
        }
        return {
            # No merged-seed cap is needed: each independent channel is
            # already bounded, and deduplication can only reduce the pool.
            # Stage 1 lexical seeds retain their own Stage 1 retrieval limit.
            "seed_channel_limits": {
                "stage2_lexical": dict(seed_limits),
                "stage2_embedding": dict(seed_limits),
            },
            # Applied independently to same-episode and same-state expansion.
            "association_per_relation_limit": max(4, min(12, k)),
            "selected_limits": {
                "fact": fact_limit,
                "state": state_limit,
            },
            "actionable_item_limit": 0,
        }
