#!/usr/bin/env python3
"""Unified memory manager inspired by MemPalace.

The public surface mirrors the current project's `MemoryNodeManager`, but the
internal model is deliberately unified:

1. assistant_wakeup turns and future allday transcript episodes both become
   `memory_episodes`.
2. Extracted evidence becomes narrative `memory_facts`.
3. Longer-running preferences/topics/constraints can become `memory_states`.
4. Concrete decisions/tasks/commitments become `memory_actionable_items`.

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
    EPISODE_SUMMARY_PROMPT_EN,
    MEMORY_RETRIEVED_FORMAT_PROMPT_EN,
    MEMORY_RETRIEVED_SECTION_SPECS_EN,
    RECALL_QUERY_ANALYSIS_PROMPT_EN,
    UNIFIED_ACTIONABLE_ITEM_EXTRACTION_PROMPT_EN,
    UNIFIED_ENTITY_STATE_UPDATE_PROMPT_EN,
    UNIFIED_MEMORY_EXTRACTION_PROMPT_EN,
    UNIFIED_TOPIC_STATE_UPDATE_PROMPT_EN,
)
from .prompts_zh import (
    EPISODE_SUMMARY_PROMPT_ZH,
    MEMORY_RETRIEVED_FORMAT_PROMPT_ZH,
    MEMORY_RETRIEVED_SECTION_SPECS_ZH,
    RECALL_QUERY_ANALYSIS_PROMPT_ZH,
    UNIFIED_ACTIONABLE_ITEM_EXTRACTION_PROMPT_ZH,
    UNIFIED_ENTITY_STATE_UPDATE_PROMPT_ZH,
    UNIFIED_MEMORY_EXTRACTION_PROMPT_ZH,
    UNIFIED_TOPIC_STATE_UPDATE_PROMPT_ZH,
)

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
                "actual_recall_path": recall_report.get("actual_recall_path"),
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
        if operation_type == "memory_store":
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
        llm_config: Optional[Dict[str, Any]] = None,
        operation_reporter: Optional[MemoryOperationReporter] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._db = db
        self._logger = logger or logging.getLogger(__name__)
        self._operation_reporter = operation_reporter or MemoryOperationReporter()
        self._embedding_cfg = dict(embedding_config or {})
        self._memory_cfg = dict(memory_manager_config or {})
        self._llm_cfg = dict(llm_config or {})
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
        self._enable_memory_state_update = self._config_bool(
            self._memory_cfg.get("enable_memory_state_update", True),
            True,
        )
        self._enable_memory_actionable_item_update = self._config_bool(
            self._memory_cfg.get("enable_memory_actionable_item_update", True),
            True,
        )
        self._top_k = int(self._memory_cfg.get("retrieval_top_k", 8) or 8)
        self._recall_detailed_logging = self._config_bool(
            self._memory_cfg.get("recall_detailed_logging", False),
            False,
        )
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
        self._recall_fast_candidate_score_threshold = self._clamp_float(
            self._memory_cfg.get("recall_fast_candidate_score_threshold"),
            0.0,
            1.0,
            0.65,
        )
        self._recall_fast_candidate_min_score = self._clamp_float(
            self._memory_cfg.get("recall_fast_candidate_min_score"),
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
        self._task_queue_maxsize = max(
            1,
            int(self._memory_cfg.get("store_queue_maxsize", 100) or 100),
        )
        self._task_queue: queue.Queue[Dict[str, Any]] = queue.Queue(
            maxsize=self._task_queue_maxsize,
        )
        self._task_worker_thread: Optional[threading.Thread] = None
        self._task_worker_lock = threading.Lock()
        self._task_shutdown_event = threading.Event()
        self._memory_operation_lock = threading.RLock()
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

    # ── Store path: raw segments -> episode -> facts -> index cards ──────

    @property
    def enabled(self) -> bool:
        return self._memory_enabled

    def set_logger(self, logger: Optional[logging.Logger]) -> None:
        """Set the logger used by memory store, reflect, and recall operations."""
        self._logger = logger or logging.getLogger(__name__)

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
        timeout: Optional[float] = 5.0,
    ) -> bool:
        """Stop accepting tasks and optionally drain the memory worker."""
        with self._task_worker_lock:
            self._task_shutdown_event.set()
            worker = self._task_worker_thread
        if not wait or worker is None:
            return not self._task_queue.unfinished_tasks
        worker.join(timeout=None if timeout is None else max(0.0, timeout))
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

    def _process_memory_store_task(
        self,
        *,
        raw_segments: List[Dict[str, Any]],
        source_type: str,
        tags: List[str],
        prompt_language: str,
    ) -> Dict[str, Any]:
        store_started_at = time.monotonic()
        episode_type = self._episode_type_for_source_type(source_type)
        self._log_info("memory_store", "start", {
            "source_type": source_type,
            "episode_type": episode_type,
            "source_segment_count": len(raw_segments),
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
        episode_participants = self._parse_participants_from_raw_segments(raw_segments)
        episode_started_at = raw_segments[0].get("started_at") or _now_text()
        episode_ended_at = raw_segments[-1].get("ended_at") or episode_started_at
        extracted_info = self._extract_memory_fact_from_raw_segments(
            raw_segments,
            prompt_language=prompt_language,
        )
        episode_title = (
            _compact_whitespace(extracted_info.get("episode_title") or "")
            or self._fallback_generate_episode_title_from_raw_segments(raw_segments)
        )
        episode_summary = (
            _compact_whitespace(extracted_info.get("episode_summary") or "")
            or self._fallback_generate_episode_summary_from_raw_segments(raw_segments)
        )
        episode_canonical_topics = (
            extracted_info.get("canonical_topics")
            or self._topic_candidates(episode_summary)
            or ["general"]
        )
        facts = list(extracted_info.get("facts") or [])

        self._log_extracted_fact_info(
            raw_segments=raw_segments,
            facts=facts,
            source_type=source_type,
            episode_type=episode_type,
            prompt_language=prompt_language,
        )

        save_entity_info = self._store_extracted_memory_entities_into_db(
            participants=episode_participants,
            raw_segments=raw_segments,
            facts=facts,
            episode_summary=episode_summary,
        )
        save_episode_info = self._store_extracted_memory_episode_into_db(
            participants=episode_participants,
            raw_segments=raw_segments,
            entity_info=save_entity_info,
            source_type=source_type,
            episode_type=episode_type,
            tags=tags,
            episode_title=episode_title,
            episode_summary=episode_summary,
            canonical_topics=episode_canonical_topics,
            started_at=episode_started_at,
            ended_at=episode_ended_at,
        )
        save_fact_info = self._store_extracted_memory_facts_into_db(
            episode_id=int(save_episode_info["episode_id"]),
            facts=facts,
            tags=tags,
            source_type=source_type,
            episode_context_topics=episode_canonical_topics,
            entity_info=save_entity_info,
        )
        report = {
            "status": "ok",
            "new_episode_count": 1,
            "new_fact_count": len(list(save_fact_info.get("fact_ids") or [])),
            "total_elapsed_ms": round((time.monotonic() - store_started_at) * 1000, 2),
        }
        self._log_info("memory_store", "finish", {
            **report,
            "source_type": source_type,
            "episode_type": episode_type,
            "source_segment_count": len(raw_segments),
        })
        return report

    def _store_extracted_memory_episode_into_db(
        self,
        *,
        participants: List[str],
        raw_segments: List[Dict[str, Any]],
        entity_info: Dict[str, int],
        source_type: str,
        episode_type: str,
        tags: List[str],
        episode_title: str,
        episode_summary: str,
        canonical_topics: List[str],
        started_at: str,
        ended_at: str,
    ) -> Dict[str, Any]:
        episode_entity_names = list(entity_info.keys())
        episode_entity_ids = [
            int(entity_id)
            for entity_id in entity_info.values()
            if str(entity_id).strip().isdigit()
        ]
        episode_id = self._db.insert_episode(
            source_type=source_type,
            episode_type=episode_type,
            title=episode_title,
            summary=episode_summary,
            participants=participants,
            started_at=started_at,
            ended_at=ended_at,
            entity_ids=episode_entity_ids,
            canonical_topics=canonical_topics,
            metadata={
                "tags": tags,
                "segment_count": len(raw_segments),
                "entities": episode_entity_names,
            },
        )

        return {
            "episode_id": episode_id,
            "participants": participants,
            "entity_names": episode_entity_names,
            "entity_ids": episode_entity_ids,
            "entity_info": entity_info,
            "title": episode_title,
            "summary": episode_summary,
            "canonical_topics": canonical_topics,
        }

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
        raw_segments: List[Dict[str, Any]],
        facts: List[Dict[str, Any]],
        source_type: str,
        episode_type: str,
        prompt_language: str,
    ) -> None:
        if not facts:
            return
        self._log_info(
            "memory_store",
            "extract_fact_state_aspects",
            {
                "source_type": source_type,
                "episode_type": episode_type,
                "raw_segments": self._build_memory_segments_for_prompt(
                    raw_segments,
                    prompt_language=prompt_language,
                ),
                "source_segment_count": len(raw_segments),
            },
        )
        for index, fact in enumerate(facts, 1):
            metadata = fact.get("metadata") if isinstance(fact.get("metadata"), dict) else {}
            self._log_info(
                "memory_store",
                "extract_fact_state_aspects",
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
                    "state_aspects": fact.get("state_aspects") or [],
                    "actionable_aspects": fact.get("actionable_aspects") or [],
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
        """Extract episode metadata and facts with LLM, falling back to heuristics."""
        data = self._extract_memory_fact_with_llm(
            raw_segments,
            prompt_language=prompt_language,
        )
        if data and data.get("facts"):
            return data
        llm_episode_summary = self._generate_episode_summary_directly_with_llm(
            raw_segments,
            prompt_language=prompt_language,
        ) or {}
        summary = (
            llm_episode_summary.get("summary")
            or self._fallback_generate_episode_summary_from_raw_segments(raw_segments)
        )
        title = (llm_episode_summary.get("title")
                or self._fallback_generate_episode_title_from_raw_segments(raw_segments)
        )
        return {
            "episode_title": title,
            "episode_summary": summary,
            "canonical_topics": self._topic_candidates(summary),
            "facts": [],
        }

    def _generate_episode_summary_directly_with_llm(
        self,
        segments: List[Dict[str, Any]],
        *,
        prompt_language: str,
    ) -> Optional[Dict[str, str]]:
        if not segments:
            return None
        prompt_template = (
            EPISODE_SUMMARY_PROMPT_EN
            if prompt_language == "en"
            else EPISODE_SUMMARY_PROMPT_ZH
        )
        prompt = prompt_template.replace(
            "{dialogue_batch}",
            self._build_memory_segments_for_prompt(
                segments,
                prompt_language=prompt_language,
            ),
        )
        for attempt in range(2):
            result = self._call_llm(prompt)
            parsed = self._parse_json_object_from_llm_text(result or "")
            if parsed is not None:
                title = _compact_whitespace(
                    parsed.get("title")
                    or parsed.get("episode_title")
                    or ""
                )
                summary = _compact_whitespace(
                    parsed.get("summary")
                    or parsed.get("episode_summary")
                    or ""
                )
                if summary:
                    self._log_info("memory_store", "episode_summary_fallback", {
                        "title": title,
                        "summary": summary,
                        "summary_chars": len(summary),
                        "source_segment_count": len(segments),
                    })
                    return {"title": title, "summary": summary}
            if attempt == 0:
                self._logger.debug("Episode summary LLM fallback failed, retrying")
        return None

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
        raw_episode_title = _compact_whitespace(data.get("episode_title") or "")
        raw_episode_summary = _compact_whitespace(data.get("episode_summary") or "")
        llm_episode_summary: Dict[str, str] = {}
        if not raw_episode_title or not raw_episode_summary:
            llm_episode_summary = self._generate_episode_summary_directly_with_llm(
                raw_segments,
                prompt_language=prompt_language,
            ) or {}
        episode_summary = (
            raw_episode_summary
            or llm_episode_summary.get("summary")
            or self._fallback_generate_episode_summary_from_raw_segments(raw_segments)
        )
        episode_topics = self._normalize_episode_canonical_topics(
            data.get("canonical_topics") or data.get("episode_canonical_topics"),
            fallback_text=episode_summary,
            limit=5,
        )
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
                fallback_root_topic=(episode_topics[0] if episode_topics else fact_topic_fallback),
                fallback_aspect_topic=fact_topic_fallback,
            )
            state_aspects = self._normalize_state_aspects(
                raw_fact.get("state_aspects"),
                fallback_entity=primary_entity,
            )
            actionable_aspects = self._normalize_actionable_aspects(
                raw_fact.get("actionable_aspects"),
            )
            event_time_key = _compact_whitespace(raw_fact.get("event_time_key") or "")
            facts.append({
                "summary": text,
                "fact_kind": self._normalize_fact_kind(raw_fact.get("fact_kind")),
                "fact_type": self._normalize_fact_type(raw_fact.get("fact_type")),
                "event_time_key": event_time_key,
                "dialogue_time_key": dialogue_time_key,
                "keywords": " ".join(keywords),
                "entities": entities,
                "primary_entity": primary_entity,
                "state_aspects": state_aspects,
                "actionable_aspects": actionable_aspects,
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
        return {
            "episode_title": (
                raw_episode_title
                or llm_episode_summary.get("title")
                or self._fallback_generate_episode_title_from_raw_segments(raw_segments)
            ),
            "episode_summary": episode_summary,
            "canonical_topics": episode_topics or self._topic_candidates(episode_summary),
            "facts": facts,
        }

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

    def _normalize_state_aspects(
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
                else self._memory_cfg.get("state_aspect_max_per_fact", 3) or 3
            ),
        )
        if max_items <= 0:
            return []
        allowed_types = self._entity_scoped_state_types()
        normalized: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str]] = set()
        for raw in value:
            if not isinstance(raw, dict):
                continue
            state_type = str(raw.get("state_type") or "").strip().lower()
            if state_type not in allowed_types:
                continue
            attribute_name = _compact_whitespace(
                raw.get("attribute_name")
                or raw.get("canonical_name")
                or raw.get("attribute")
                or ""
            )
            aspect_summary = _compact_whitespace(
                raw.get("aspect_summary")
                or raw.get("summary")
                or raw.get("text")
                or ""
            )
            evidence_basis = _compact_whitespace(
                raw.get("evidence_basis")
                or raw.get("evidence")
                or raw.get("reason")
                or ""
            )
            if not attribute_name or not aspect_summary:
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
                state_type,
                attribute_name.lower(),
                aspect_summary.lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            item: Dict[str, Any] = {
                "state_type": state_type,
                "attribute_name": attribute_name,
                "aspect_summary": aspect_summary,
                "evidence_basis": evidence_basis,
                "confidence": confidence,
            }
            if entity_payload:
                item["entity"] = entity_payload
            normalized.append(item)
            if len(normalized) >= max_items:
                break
        return normalized

    def _normalize_actionable_aspects(
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
                else self._memory_cfg.get("actionable_aspect_max_per_fact", 2) or 2
            ),
        )
        if max_items <= 0:
            return []
        allowed_types = {
            "task", "commitment", "decision", "follow_up", "open_question",
            "risk", "reminder", "recommendation", "constraint",
        }
        normalized: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str]] = set()
        for raw in value:
            if not isinstance(raw, dict):
                continue
            item_type = self._normalize_actionable_item_type(raw.get("item_type"))
            if item_type not in allowed_types:
                continue
            action_summary = _compact_whitespace(
                raw.get("action_summary")
                or raw.get("summary")
                or raw.get("text")
                or ""
            )
            trigger_basis = _compact_whitespace(
                raw.get("trigger_basis")
                or raw.get("evidence_basis")
                or raw.get("evidence")
                or raw.get("reason")
                or ""
            )
            if not action_summary or not trigger_basis:
                continue
            owner = self._normalize_actionable_owner(raw.get("owner"))
            status = self._normalize_actionable_status(raw.get("status"))
            due_at = _compact_whitespace(raw.get("due_at") or "")
            confidence = self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.75)
            key = (item_type, action_summary.lower())
            if key in seen:
                continue
            seen.add(key)
            normalized.append({
                "item_type": item_type,
                "action_summary": action_summary,
                "owner": owner,
                "status": status,
                "due_at": due_at,
                "trigger_basis": trigger_basis,
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
        """Collect a small, balanced state reference set for fact extraction."""
        try:
            states = self._db.get_recent_memory_states(limit=max(40, int(limit or 12) * 6))
        except Exception as exc:
            self._logger.debug("Failed to load memory state context: %s", exc)
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
        episode_id: int,
        facts: List[Dict[str, Any]],
        tags: List[str],
        source_type: str,
        episode_context_topics: Optional[Sequence[str]] = None,
        entity_info: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        fact_ids: List[int] = []
        normalized_entity_info = {
            str(entity_name): int(entity_id)
            for entity_name, entity_id in (entity_info or {}).items()
            if str(entity_name).strip() and str(entity_id).strip().isdigit()
        }
        episode_context_entities = list(normalized_entity_info.keys())
        for fact in facts:
            keywords = fact.get("keywords") or ""
            if isinstance(keywords, (list, tuple, set)):
                keywords = " ".join(str(item).strip() for item in keywords if str(item).strip())
            else:
                keywords = str(keywords).strip()
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
                f"keywords: {keywords}",
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
                "state_aspects": fact.get("state_aspects") or [],
                "actionable_aspects": fact.get("actionable_aspects") or [],
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
        return {
            "fact_ids": fact_ids,
        }

    # ── Reflection: facts -> evolving states ─────────────────────────────

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
        """Update topic/entity projections and actionable items from recent facts."""
        reflect_started_at = time.monotonic()
        limit = max(1, int(limit or self._memory_cfg.get("reflect_limit") or 100))
        if reflect_timestamp is None:
            reflect_timestamp = kwargs.get("timestamp") or _now_text()
        self._log_info("memory_reflect", "start", {
            "limit": limit,
            "reflect_timestamp": reflect_timestamp,
        })
        # Keep the projection apply atomic. The expensive extraction and
        # embedding work still happens inside the existing worker, while all
        # state/actionable writes and processed markers share one commit point.
        with self._db.transaction():
            state_report = self._update_memory_states_using_facts(
                limit=limit,
                reference_timestamp=reflect_timestamp,
            )
            topic_report = state_report["topic_report"]
            entity_report = state_report["entity_report"]

            actionable_report = self._update_memory_actionable_items_using_facts(
                limit=limit,
                reference_timestamp=reflect_timestamp,
            )
        report = {
            "status": (
                "ok"
                if state_report.get("fact_count", 0)
                or actionable_report.get("fact_count", 0)
                else "empty"
            ),
            "states_updated": int(state_report.get("states_updated", 0) or 0),
            "topic_facts_considered": int(
                state_report.get("topic_facts_considered", 0) or 0
            ),
            "entity_facts_considered": int(
                state_report.get("entity_facts_considered", 0) or 0
            ),
            "evidence_only_facts": int(
                state_report.get("evidence_only_facts", 0) or 0
            ),
            "topic_states_updated": int(topic_report.get("updated", 0) or 0),
            "topic_candidates_unresolved": int(topic_report.get("unresolved", 0) or 0),
            "pending_unresolved_topics": int(topic_report.get("pending_unresolved", 0) or 0),
            "entity_states_updated": int(entity_report.get("updated", 0) or 0),
            "actionable_facts_considered": int(
                actionable_report.get("candidate_fact_count", 0) or 0
            ),
            "facts_marked_processed_for_memory_state": int(
                state_report.get("facts_marked_processed", 0) or 0
            ),
            "facts_marked_processed_for_memory_actionable_item": int(
                actionable_report.get("facts_marked_processed", 0) or 0
            ),
            "actionable_items_updated": int(
                actionable_report.get("stored_count", 0) or 0
            ),
            "actionable_item_ids": actionable_report.get("item_ids") or [],
            "total_elapsed_ms": round(
                (time.monotonic() - reflect_started_at) * 1000,
                2,
            ),
        }
        self._log_info("memory_reflect", "finish", report)
        return report

    def _update_memory_states_using_facts(
        self,
        *,
        limit: int,
        reference_timestamp: Any,
    ) -> Dict[str, Any]:
        """Update topic and entity state projections from the current facts."""

        started_at = time.monotonic()
        if not self._enable_memory_state_update:
            report = {
                "enabled": 0,
                "fact_count": 0,
                "fact_ids": [],
                "topic_report": {
                    "enabled": 0,
                    "updated": 0,
                    "unresolved": 0,
                    "pending_unresolved": len(self._pending_unresolved_topics),
                },
                "entity_report": {
                    "enabled": 0,
                    "updated": 0,
                },
                "topic_facts_considered": 0,
                "entity_facts_considered": 0,
                "evidence_only_facts": 0,
                "states_updated": 0,
                "facts_marked_processed": 0,
                "total_elapsed_ms": round(
                    (time.monotonic() - started_at) * 1000,
                    2,
                ),
            }
            self._log_info("memory_reflect", "state_update_skipped", report)
            return report
        facts = self._db.get_unprocessed_facts(
            processing_target="state",
            limit=limit,
            reference_timestamp=reference_timestamp,
        )
        self._log_reflect_facts_loaded("state", facts, limit, reference_timestamp)
        topic_facts = [fact for fact in facts if self._fact_can_seed_topic_state(fact)]
        entity_facts = [fact for fact in facts if self._fact_can_seed_entity_state(fact)]
        self._log_info("memory_reflect", "state_fact_candidates", {
            "topic_fact_count": len(topic_facts),
            "topic_fact_ids": [
                fact.get("id") for fact in topic_facts if fact.get("id") is not None
            ],
            "entity_fact_count": len(entity_facts),
            "entity_fact_ids": [
                fact.get("id") for fact in entity_facts if fact.get("id") is not None
            ],
        })

        topic_report = self._resolve_and_update_topic_states_from_facts(
            facts=topic_facts,
        )
        entity_report = self._resolve_and_update_entity_scoped_states_from_facts(
            facts=entity_facts,
        )
        facts_marked_processed = self._db.mark_facts_processed(
            processing_target="state",
            fact_ids=[fact.get("id") for fact in facts],
        )
        report = {
            "fact_count": len(facts),
            "fact_ids": [fact.get("id") for fact in facts],
            "topic_report": topic_report,
            "entity_report": entity_report,
            "topic_facts_considered": len(topic_facts),
            "entity_facts_considered": len(entity_facts),
            "evidence_only_facts": max(0, len(facts) - len(set(
                int(fact["id"])
                for fact in [*topic_facts, *entity_facts]
                if str(fact.get("id") or "").strip().isdigit()
            ))),
            "states_updated": (
                int(topic_report.get("updated", 0) or 0)
                + int(entity_report.get("updated", 0) or 0)
            ),
            "facts_marked_processed": facts_marked_processed,
            "total_elapsed_ms": round(
                (time.monotonic() - started_at) * 1000,
                2,
            ),
        }
        self._log_info("memory_reflect", "state_update_finish", report)
        return report

    def _update_memory_actionable_items_using_facts(
        self,
        *,
        limit: int,
        reference_timestamp: Any,
    ) -> Dict[str, Any]:
        """Extract and persist actionable items from the current reflect facts."""
        started_at = time.monotonic()
        if not self._enable_memory_actionable_item_update:
            report = {
                "enabled": 0,
                "fact_count": 0,
                "fact_ids": [],
                "candidate_fact_count": 0,
                "candidate_fact_ids": [],
                "actionable_update_count": 0,
                "requested_store_count": 0,
                "stored_count": 0,
                "item_ids": [],
                "facts_marked_processed": 0,
                "total_elapsed_ms": round(
                    (time.monotonic() - started_at) * 1000,
                    2,
                ),
            }
            self._log_info(
                "memory_reflect",
                "actionable_item_update_skipped",
                report,
            )
            return report
        facts = self._db.get_unprocessed_facts(
            processing_target="actionable_item",
            limit=limit,
            reference_timestamp=reference_timestamp,
        )
        self._log_reflect_facts_loaded(
            "actionable_item", facts, limit, reference_timestamp
        )
        actionable_facts = self._filter_facts_for_actionable_item_extraction(facts)
        self._log_info("memory_reflect", "actionable_item_extraction_start", {
            "candidate_fact_count": len(actionable_facts),
            "candidate_fact_ids": [
                fact.get("id") for fact in actionable_facts if fact.get("id") is not None
            ],
        })
        actionable_updates = self._extract_actionable_items_with_llm(facts=actionable_facts)
        self._log_info("memory_reflect", "actionable_item_extraction_finish", {
            "candidate_fact_count": len(actionable_facts),
            "actionable_update_count": len(actionable_updates),
            "actionable_updates": [
                {
                    "item_type": item.get("item_type"),
                    "canonical_name": item.get("canonical_name"),
                    "owner": item.get("owner"),
                    "status": item.get("status"),
                    "due_at": item.get("due_at"),
                    "evidence_fact_ids": item.get("evidence_fact_ids") or [],
                    "confidence": item.get("confidence"),
                    "importance": item.get("importance"),
                }
                for item in actionable_updates
            ],
        })
        actionable_items_updated = 0
        actionable_item_ids: List[int] = []
        for item in actionable_updates:
            item_id = self._store_actionable_item(item)
            if item_id:
                actionable_items_updated += 1
                actionable_item_ids.append(item_id)

        facts_marked_processed = self._db.mark_facts_processed(
            processing_target="actionable_item",
            fact_ids=[fact.get("id") for fact in facts],
        )
        
        report = {
            "fact_count": len(facts),
            "fact_ids": [fact.get("id") for fact in facts],
            "candidate_fact_count": len(actionable_facts),
            "candidate_fact_ids": [
                fact.get("id") for fact in actionable_facts
                if fact.get("id") is not None
            ],
            "actionable_update_count": len(actionable_updates),
            "requested_store_count": len(actionable_updates),
            "stored_count": actionable_items_updated,
            "item_ids": actionable_item_ids,
            "facts_marked_processed": facts_marked_processed,
            "total_elapsed_ms": round(
                (time.monotonic() - started_at) * 1000,
                2,
            ),
        }
        self._log_info(
            "memory_reflect",
            "actionable_item_update_finish",
            report,
        )
        return report

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
        topics = cls._fact_topic_names(fact)
        return bool(topics) and cls._fact_has_durable_state_signal(fact)

    @classmethod
    def _fact_topic_names(cls, fact: Dict[str, Any]) -> List[str]:
        topics = [
            cls._normalize_topic_name(fact.get("fact_root_topic")),
            cls._normalize_topic_name(fact.get("fact_aspect_topic")),
        ]
        return list(dict.fromkeys(topic for topic in topics if topic))

    def _fact_can_seed_entity_state(self, fact: Dict[str, Any]) -> bool:
        """Entity states require an explicit projected aspect."""
        state_aspects = self._state_aspects_from_fact(fact)
        return any(
            str(aspect.get("state_type") or "").strip().lower()
            in self._entity_scoped_state_types()
            and bool(self._entities_for_state_aspect(aspect, fact))
            for aspect in state_aspects
        )

    def _resolve_and_update_topic_states_from_facts(
        self,
        *,
        facts: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        update_started_at = time.monotonic()
        if not self._enable_topic_state_resolution:
            report = {
                "enabled": 0,
                "updated": 0,
                "unresolved": 0,
                "pending_unresolved": len(self._pending_unresolved_topics),
            }
            self._log_info("memory_reflect", "topic_state_update_finish", {
                **report,
            })
            return report
        candidates = self._build_topic_state_candidates_from_facts(facts)
        updated = 0
        unresolved = 0
        for candidate in candidates:
            candidate_existing_topic_states = (
                self._retrieve_existing_topic_states_for_candidate(
                    candidate=candidate,
                    limit=16,
                )
            )
            self._log_info(
                "memory_reflect",
                "topic_state_candidates_retrieved",
                {
                    "candidate_root_topic": candidate.get("topic_name"),
                    "candidate_aspect_topics": candidate.get("aspect_topics") or [],
                    "candidate_fact_ids": candidate.get("fact_ids") or [],
                    "existing_state_ids": [
                        state.get("id")
                        for state in candidate_existing_topic_states
                        if state.get("id") is not None
                    ],
                    "existing_state_count": len(candidate_existing_topic_states),
                },
            )
            matched_state, match_info = self._match_topic_state_candidate_to_existing_state(
                candidate=candidate,
                existing_topic_states=candidate_existing_topic_states,
            )
            grounded, chosen_state, grounding_info = self._ground_topic_state_candidate(
                candidate=candidate,
                matched_state=matched_state,
                match_info=match_info,
                existing_topic_states=candidate_existing_topic_states,
            )
            if not grounded:
                unresolved += 1
                self._remember_pending_unresolved_topic(candidate, grounding_info)
                continue
            if chosen_state:
                candidate["topic_name"] = str(
                    chosen_state.get("canonical_name")
                    or candidate.get("topic_name")
                    or "general"
                )
            state_update = self._extract_topic_state_update_with_llm(
                candidate=candidate,
                existing_state=chosen_state,
            )
            if not state_update or not state_update.get("summary"):
                continue
            state_update.setdefault("source_type", candidate.get("source_type"))
            state_id = self._store_state(state_update)
            if state_id:
                self._log_info(
                    "memory_reflect",
                    "topic_state_updated",
                    self._state_update_log_payload(
                        state_id=state_id,
                        state_update=state_update,
                        candidate=candidate,
                        existing_state=chosen_state,
                    ),
                )
                updated += 1
        report = {
            "enabled": 1,
            "candidate_count": len(candidates),
            "updated": updated,
            "unresolved": unresolved,
            "pending_unresolved": len(self._pending_unresolved_topics),
        }
        self._log_info("memory_reflect", "topic_state_update_finish", {
            **report,
            "elapsed_ms": round(
                (time.monotonic() - update_started_at) * 1000,
                2,
            ),
        })
        return report

    def _retrieve_existing_topic_states_for_candidate(
        self,
        *,
        candidate: Dict[str, Any],
        limit: int = 16,
    ) -> List[Dict[str, Any]]:
        """Retrieve only existing topic states relevant to one topic candidate."""
        candidate_limit = max(1, int(limit or 16))
        states_by_id: Dict[int, Dict[str, Any]] = {}

        def add_states(rows: Sequence[Dict[str, Any]]) -> None:
            for row in rows or []:
                if str(row.get("state_type") or "").strip().lower() != "topic":
                    continue
                try:
                    state_id = int(row.get("id"))
                except (TypeError, ValueError):
                    continue
                states_by_id[state_id] = row

        root_topic = self._normalize_topic_name(candidate.get("topic_name"))
        if root_topic and self._generate_topic_name_key(root_topic) != "general":
            add_states(
                self._db.search_memory_states(
                    terms=self._build_recall_search_terms(
                        "",
                        keywords=[root_topic],
                        entities=[],
                    ),
                    state_type="topic",
                    limit=candidate_limit,
                )
            )

        aspect_topics = self._normalize_unique_labels(
            candidate.get("aspect_topics") or [],
            limit=16,
        )
        if aspect_topics:
            add_states(
                self._db.search_memory_states(
                    terms=self._build_recall_search_terms(
                        "",
                        keywords=aspect_topics,
                        entities=[],
                    ),
                    state_type="topic",
                    limit=candidate_limit,
                )
            )

        supplementary_terms = self._normalize_unique_labels(
            [
                *(candidate.get("keywords") or []),
                *(candidate.get("context_entities") or []),
            ],
            limit=20,
        )
        if supplementary_terms:
            add_states(
                self._db.search_memory_states(
                    terms=self._build_recall_search_terms(
                        "",
                        keywords=supplementary_terms,
                        entities=[],
                    ),
                    state_type="topic",
                    limit=candidate_limit,
                )
            )

        # Keep a small recency fallback for underspecified candidates. This is
        # intentionally candidate-local and is not used as a shared match pool.
        recent_states = self._db.get_recent_memory_states(
            state_type="topic",
            limit=min(8, candidate_limit),
        )
        add_states(recent_states)
        return list(states_by_id.values())

    def _build_topic_state_candidates_from_facts(
        self,
        facts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        by_source_and_root: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for fact in facts:
            source_type = str(fact.get("source_type"))
            root_topic_name = (
                self._normalize_topic_name(fact.get("fact_root_topic"))
                or "general"
            )
            root_topic_key = self._generate_topic_name_key(root_topic_name)
            group = by_source_and_root.setdefault(
                (source_type, root_topic_key),
                {
                    "source_type": source_type,
                    "root_topic_key": root_topic_key,
                    "root_topic_name": root_topic_name,
                    "facts": [],
                },
            )
            group["facts"].append(fact)

        candidates: List[Dict[str, Any]] = []
        for group in by_source_and_root.values():
            matched_facts = sorted(
                group["facts"],
                key=lambda fact: (
                    str(fact.get("dialogue_time_key") or ""),
                    int(fact.get("id") or 0),
                ),
            )
            fact_ids = [
                int(fact["id"])
                for fact in matched_facts
                if str(fact.get("id") or "").strip().isdigit()
            ]
            if not fact_ids:
                continue
            root_topic_name = str(group["root_topic_name"])
            root_topic_key = str(group["root_topic_key"])
            aspect_topics = self._normalize_unique_labels([
                self._normalize_topic_name(fact.get("fact_aspect_topic"))
                or self._normalize_topic_name(fact.get("fact_root_topic"))
                or "general"
                for fact in matched_facts
            ], limit=16)
            parent_topics = self._normalize_unique_labels([
                topic
                for fact in matched_facts
                for topic in (
                    (fact.get("metadata") or {}).get("episode_context_topics") or []
                )
            ], limit=12)
            aspect_topic_keys = {
                self._generate_topic_name_key(aspect)
                for aspect in aspect_topics
            }
            parent_topics = [
                topic
                for topic in parent_topics
                if self._generate_topic_name_key(topic) != root_topic_key
                and self._generate_topic_name_key(topic) not in aspect_topic_keys
            ]
            context_entities = self._normalize_entity_names([
                entity
                for fact in matched_facts
                for entity in fact.get("entities") or []
            ], limit=18)
            keywords = self._generate_topic_candidate_identity_keywords(
                matched_facts,
                limit=12,
            )
            fact_summaries = [
                _compact_whitespace(fact.get("summary") or "")
                for fact in matched_facts
                if _compact_whitespace(fact.get("summary") or "")
            ]
            identity_text = self._generate_topic_candidate_identity_text(
                canonical_name=root_topic_name,
                keywords=keywords,
                fact_summaries=fact_summaries,
                context_topics=[*parent_topics, *aspect_topics],
                context_entities=context_entities,
            )
            candidates.append({
                "topic_key": root_topic_key,
                "topic_name": root_topic_name,
                "keywords": keywords,
                "fact_summaries": fact_summaries,
                "identity_text": identity_text,
                "aspect_topics": aspect_topics,
                "parent_topics": parent_topics,
                "context_entities": context_entities,
                "facts": matched_facts,
                "fact_ids": fact_ids,
                "source_type": group["source_type"],
                "summary_text": "\n".join(
                    str(fact.get("summary") or "") for fact in matched_facts
                )[:2400],
            })
        return candidates

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

    def _generate_topic_candidate_identity_keywords(
        self,
        facts: Sequence[Dict[str, Any]],
        *,
        limit: int = 12,
    ) -> List[str]:
        values: List[str] = []
        for fact in facts:
            values.extend(str(fact.get("keywords") or "").split())
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
            key = self._generate_topic_name_key(clean)
            if key in seen:
                continue
            seen.add(key)
            out.append(clean)
            if len(out) >= limit:
                break
        return out

    def _generate_topic_candidate_identity_text(
        self,
        *,
        canonical_name: Any,
        keywords: Sequence[Any],
        fact_summaries: Optional[Sequence[Any]] = None,
        context_topics: Optional[Sequence[Any]] = None,
        context_entities: Optional[Sequence[Any]] = None,
    ) -> str:
        canonical = self._normalize_topic_name(canonical_name) or _compact_whitespace(canonical_name)
        keyword_values = self._normalize_unique_labels(keywords, limit=12)
        raw_summaries = [fact_summaries] if isinstance(fact_summaries, str) else fact_summaries or []
        summary_values = self._normalize_unique_labels(raw_summaries, limit=5)
        summary_values = [value[:320] for value in summary_values]
        context_topic_values = self._normalize_unique_labels(context_topics or [], limit=8)
        context_entity_values = self._normalize_entity_names(
            list(context_entities or []),
            limit=12,
        )
        return "\n".join([
            f"root_topic: {canonical}",
            f"context_topics: {', '.join(context_topic_values)}",
            f"context_entities: {', '.join(context_entity_values)}",
            f"keywords: {', '.join(keyword_values)}",
            f"fact_summaries: {' | '.join(summary_values)}",
        ])

    def _match_topic_state_candidate_to_existing_state(
        self,
        *,
        candidate: Dict[str, Any],
        existing_topic_states: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        if not existing_topic_states:
            return None, {"matched": False, "reason": "no_existing_topic_states"}
        candidate_root_name = (
            self._normalize_topic_name(candidate.get("topic_name"))
            or _compact_whitespace(candidate.get("topic_key") or "")
            or "general"
        )
        candidate_root_key = self._generate_topic_name_key(
            candidate.get("topic_key") or candidate_root_name
        )
        candidate_aspect_topics = self._normalize_unique_labels(
            candidate.get("aspect_topics") or [],
            limit=16,
        )
        candidate_aspect_keys = {
            self._generate_topic_name_key(topic)
            for topic in candidate_aspect_topics
            if self._generate_topic_name_key(topic)
        }
        candidate_identity_text = (
            _compact_whitespace(candidate.get("identity_text") or "")
            or self._generate_topic_candidate_identity_text(
                canonical_name=candidate_root_name,
                keywords=candidate.get("keywords") or [],
                fact_summaries=candidate.get("fact_summaries") or candidate.get("summary_text"),
            )
        )
        best_state: Optional[Dict[str, Any]] = None
        best_info: Dict[str, Any] = {"score": 0.0}
        candidate_identity_embedding: Optional[np.ndarray] = None
        candidate_name_embedding: Optional[np.ndarray] = None
        if any(
            state.get("identity_text_embedding") is not None
            for state in existing_topic_states
        ):
            candidate_identity_embedding = self._generate_embedding_vector(candidate_identity_text)
        if any(
            state.get("canonical_name_embedding") is not None
            for state in existing_topic_states
        ):
            candidate_name_embedding = self._generate_embedding_vector(candidate_root_name)
        for state in existing_topic_states:
            state_metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
            state_aspect_topics = self._normalize_unique_labels([
                *(state_metadata.get("aspect_topic_names") or []),
            ], limit=16)
            state_aspect_keys = {
                self._generate_topic_name_key(topic)
                for topic in state_aspect_topics
                if self._generate_topic_name_key(topic)
            }
            state_root_name = (
                self._normalize_topic_name(state.get("canonical_name"))
                or "general"
            )
            state_root_key = self._generate_topic_name_key(state_root_name)
            exact_root_match = candidate_root_key == state_root_key
            matched_aspect_keys = candidate_aspect_keys & state_aspect_keys
            exact_aspect_match = bool(matched_aspect_keys)
            aspect_to_root_match = state_root_key in candidate_aspect_keys
            root_name_overlap = self._topic_name_overlap(
                [candidate_root_name],
                [state_root_name],
                allow_substring=False,
            )
            identity_embedding_similarity = self._cal_embedding_similarity(
                candidate_identity_embedding,
                state.get("identity_text_embedding"),
            )
            canonical_name_embedding_similarity = self._cal_embedding_similarity(
                candidate_name_embedding,
                state.get("canonical_name_embedding"),
            )
            embedding_similarity = max(
                identity_embedding_similarity,
                canonical_name_embedding_similarity,
            )
            strong_name_match = root_name_overlap >= self._topic_state_resolution_similarity_threshold
            strong_embedding_match = (
                embedding_similarity >= self._topic_identity_embedding_similarity_threshold
                and root_name_overlap >= 0.5
            )
            aspect_supported_name_threshold = max(
                0.4,
                self._topic_state_resolution_similarity_threshold - 0.15,
            )
            aspect_supported_match = exact_aspect_match and (
                root_name_overlap >= aspect_supported_name_threshold
                or embedding_similarity >= self._topic_identity_embedding_similarity_threshold
            )
            matched = (
                exact_root_match
                or strong_name_match
                or strong_embedding_match
                or aspect_supported_match
            )
            aspect_match_bonus = 0.16 if exact_aspect_match else 0.0
            score = max(
                1.0 if exact_root_match else 0.0,
                min(1.0, max(root_name_overlap, embedding_similarity) + aspect_match_bonus),
            )
            if score > float(best_info.get("score", 0.0)):
                best_state = state
                best_info = {
                    "matched": matched,
                    "reason": "matched_existing_topic_state" if matched else "best_match_below_threshold",
                    "score": round(score, 4),
                    "exact_aspect_match": exact_aspect_match,
                    "aspect_to_root_match": aspect_to_root_match,
                    "exact_root_match": exact_root_match,
                    "candidate_root_topic": candidate_root_name,
                    "candidate_aspect_topics": candidate_aspect_topics,
                    "existing_aspect_topics": state_aspect_topics,
                    "matched_aspect_keys": sorted(matched_aspect_keys),
                    "existing_root_topic": state_root_name,
                    "root_name_overlap": round(root_name_overlap, 4),
                    "embedding_similarity": round(embedding_similarity, 4),
                    "identity_embedding_similarity": round(identity_embedding_similarity, 4),
                    "canonical_name_embedding_similarity": round(
                        canonical_name_embedding_similarity,
                        4,
                    ),
                    "strong_name_match": strong_name_match,
                    "strong_embedding_match": strong_embedding_match,
                    "aspect_supported_match": aspect_supported_match,
                    "aspect_supported_name_threshold": round(
                        aspect_supported_name_threshold,
                        4,
                    ),
                    "existing_state_id": state.get("id"),
                    "existing_canonical_name": state.get("canonical_name"),
                }
        if best_state and best_info.get("matched"):
            return best_state, best_info
        return None, best_info

    def _topic_name_overlap(
        self,
        left_aliases: Sequence[str],
        right_aliases: Sequence[str],
        *,
        allow_substring: bool = True,
    ) -> float:
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
        return len(
            self._topic_anchor_remainder(str(candidate.get("topic_name") or ""))
        ) >= 2

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
            and float(match_info.get("root_name_overlap", 0.0) or 0.0) >= 0.2
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
            "topic_name": candidate.get("topic_name"),
            "topic_key": candidate.get("topic_key"),
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
            .replace("{candidate_topic_state}", json.dumps(
                self._format_topic_state_candidate_for_prompt(candidate),
                ensure_ascii=False,
                indent=2,
            ))
            .replace("{existing_topic_state}", json.dumps(
                self._format_existing_topic_state_for_prompt(existing_state),
                ensure_ascii=False,
                indent=2,
            ))
        )
        result = self._call_llm(prompt)
        parsed = self._parse_json_object_from_llm_text(result or "")
        if parsed:
            if not self._config_bool(parsed.get("update_needed", True), True):
                self._logger.debug(
                    "LLM declined topic-state update for topic=%s",
                    candidate.get("topic_name"),
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
    def _format_topic_state_candidate_for_prompt(
        candidate: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Format the resolved topic candidate without duplicating raw facts."""
        candidate = candidate or {}
        return {
            "root_topic_name": candidate.get("topic_name") or "",
            "topic_key": candidate.get("topic_key") or "",
            "identity_text": candidate.get("identity_text") or "",
            "aspect_topics": list(candidate.get("aspect_topics") or []),
            "parent_topics": list(candidate.get("parent_topics") or []),
            "keywords": list(candidate.get("keywords") or []),
            "context_entities": list(candidate.get("context_entities") or []),
            "fact_summaries": list(candidate.get("fact_summaries") or []),
            "fact_ids": [
                int(fact_id)
                for fact_id in candidate.get("fact_ids") or []
                if str(fact_id).strip().isdigit()
            ],
            "source_type": candidate.get("source_type") or "",
        }

    @staticmethod
    def _format_existing_topic_state_for_prompt(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not state:
            return {}
        metadata = state.get("metadata") or {}
        return {
            "state_scope": state.get("state_scope") or "topic_state",
            "source_type": state.get("source_type"),
            "canonical_name": state.get("canonical_name"),
            "summary": state.get("summary"),
            "time_line": MemoryNodeManager._normalize_time_line(
                state.get("time_line"),
                limit=8,
                max_chars=1000,
            ),
            "confidence": state.get("confidence"),
            "aspect_topic_names": metadata.get("aspect_topic_names") or [],
        }

    @staticmethod
    def _format_existing_entity_state_for_prompt(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Format only the semantic context needed to update an entity state."""
        if not state:
            return {}
        metadata = state.get("metadata") or {}
        return {
            "state_scope": state.get("state_scope") or "entity_state",
            "state_type": state.get("state_type"),
            "entity": metadata.get("entity") or "",
            "entity_key": metadata.get("entity_key") or "",
            "canonical_name": state.get("canonical_name"),
            "attribute_name_aliases": metadata.get("attribute_name_aliases") or [],
            "summary": state.get("summary"),
            "time_line": MemoryNodeManager._normalize_time_line(
                state.get("time_line"),
                limit=8,
                max_chars=1000,
            ),
            "confidence": state.get("confidence"),
            "status": state.get("status") or "active",
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
        aspect_items = [
            aspect for aspect in candidate.get("state_aspects") or []
            if isinstance(aspect, dict)
            and _compact_whitespace(aspect.get("aspect_summary") or "")
        ]
        if aspect_items:
            fact_ids = [
                int(aspect.get("fact_id"))
                for aspect in aspect_items
                if str(aspect.get("fact_id") or "").strip().isdigit()
            ]
            latest = aspect_items[-1]
            occurred_at = _compact_whitespace(
                str(
                    latest.get("fact_dialogue_time_key")
                    or latest.get("fact_event_time_key")
                    or ""
                ).split("#", 1)[0]
            )
            return [{
                "occurred_at": occurred_at,
                "change_type": "updated",
                "summary": self._normalize_state_summary(
                    latest.get("aspect_summary") or "",
                    max_chars=120,
                ),
                "fact_ids": list(dict.fromkeys(fact_ids))[-12:],
            }]
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
        dialogue_time_key = _compact_whitespace(latest.get("dialogue_time_key") or "")
        occurred_at = dialogue_time_key if dialogue_time_key else ""
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
            or _compact_whitespace(candidate.get("topic_name") or "")
            or self._normalize_topic_name(raw.get("canonical_name"))
            or "general"
        )
        existing_metadata = dict((existing_state or {}).get("metadata") or {})
        parent_topics = self._normalize_unique_labels([
            *(existing_metadata.get("parent_topics") or []),
            *(candidate.get("parent_topics") or []),
        ], limit=12)
        context_entities = self._normalize_entity_names([
            *(existing_metadata.get("context_entities") or []),
            *(candidate.get("context_entities") or []),
            *(raw.get("entities") or []),
        ], limit=18)
        aspect_topic_names = self._normalize_unique_labels([
            *(existing_metadata.get("aspect_topic_names") or []),
            *(candidate.get("aspect_topics") or []),
        ], limit=16)
        aspect_fact_ids: Dict[str, List[int]] = {
            str(key): [
                int(value)
                for value in values
                if str(value).strip().isdigit()
            ][:80]
            for key, values in (existing_metadata.get("aspect_fact_ids") or {}).items()
            if isinstance(values, list)
        }
        for fact in candidate.get("facts") or []:
            fact_id = fact.get("id")
            if not str(fact_id or "").strip().isdigit():
                continue
            aspect = self._normalize_topic_name(fact.get("fact_aspect_topic")) or canonical_name
            current_ids = aspect_fact_ids.setdefault(str(aspect), [])
            current_ids.append(int(fact_id))
            aspect_fact_ids[str(aspect)] = list(dict.fromkeys(current_ids))[-80:]
        canonical_topics = self._normalize_unique_labels([
            canonical_name,
            *(raw.get("canonical_topics") or []),
            *parent_topics,
            *aspect_topic_names,
        ], limit=8)
        keywords = self._normalize_unique_labels([
            *self._normalize_string_list(raw.get("keywords"), limit=18),
            *(candidate.get("keywords") or []),
            *parent_topics,
            *aspect_topic_names,
            *context_entities,
        ], limit=24)
        time_line = self._build_state_time_line(
            raw_updates=raw.get("time_line"),
            candidate=candidate,
            existing_state=existing_state,
        )
        return {
            "state_scope": "topic_state",
            "state_type": "topic",
            "source_type": candidate.get("source_type") or (existing_state or {}).get("source_type"),
            "canonical_name": canonical_name,
            "summary": summary,
            "time_line": time_line,
            "evidence_fact_ids": evidence_ids,
            "keywords": keywords,
            "entities": context_entities,
            "canonical_topics": canonical_topics or [canonical_name],
            "importance": self._clamp_float(raw.get("importance"), 0.0, 1.0, 0.7),
            "confidence": self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.75),
            "status": _compact_whitespace(raw.get("status") or "active") or "active",
            "metadata": {
                "parent_topics": parent_topics,
                "aspect_topic_names": aspect_topic_names,
                "aspect_fact_ids": aspect_fact_ids,
                "context_entities": context_entities,
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
            or _compact_whitespace(candidate.get("topic_name") or "")
            or "general"
        )
        existing_metadata = dict((existing_state or {}).get("metadata") or {})
        for key in (
            "topic_key",
            "identity_text",
            "keywords",
            "episode_id",
        ):
            existing_metadata.pop(key, None)
        parent_topics = self._normalize_unique_labels([
            *(existing_metadata.get("parent_topics") or []),
            *(candidate.get("parent_topics") or []),
        ], limit=12)
        context_entities = self._normalize_entity_names([
            *(existing_metadata.get("context_entities") or []),
            *(candidate.get("context_entities") or []),
        ], limit=18)
        aspect_topic_names = self._normalize_unique_labels([
            *(existing_metadata.get("aspect_topic_names") or []),
            *(candidate.get("aspect_topics") or []),
        ], limit=16)
        aspect_fact_ids: Dict[str, List[int]] = {
            str(key): [
                int(value)
                for value in values
                if str(value).strip().isdigit()
            ][:80]
            for key, values in (existing_metadata.get("aspect_fact_ids") or {}).items()
            if isinstance(values, list)
        }
        for fact in candidate.get("facts") or []:
            fact_id = fact.get("id")
            if not str(fact_id or "").strip().isdigit():
                continue
            aspect = self._normalize_topic_name(fact.get("fact_aspect_topic")) or canonical_name
            current_ids = aspect_fact_ids.setdefault(str(aspect), [])
            current_ids.append(int(fact_id))
            aspect_fact_ids[str(aspect)] = list(dict.fromkeys(current_ids))[-80:]
        return {
            "state_scope": "topic_state",
            "state_type": "topic",
            "source_type": candidate.get("source_type") or (existing_state or {}).get("source_type"),
            "canonical_name": canonical_name,
            "summary": summary,
            "time_line": self._build_state_time_line(
                raw_updates=None,
                candidate=candidate,
                existing_state=existing_state,
            ),
            "evidence_fact_ids": evidence_ids,
            "keywords": self._normalize_unique_labels([
                *self._keywords(summary, limit=18),
                *(candidate.get("keywords") or []),
                *parent_topics,
                *aspect_topic_names,
                *context_entities,
            ], limit=24),
            "entities": context_entities or self._entities(summary),
            "canonical_topics": self._normalize_unique_labels([
                canonical_name,
                *parent_topics,
                *aspect_topic_names,
            ], limit=8),
            "importance": 0.7,
            "confidence": 0.58,
            "status": "active",
            "metadata": {
                "extractor": "fallback_topic_state_update",
                "parent_topics": parent_topics,
                "aspect_topic_names": aspect_topic_names,
                "aspect_fact_ids": aspect_fact_ids,
                "context_entities": context_entities,
            },
        }

    def _resolve_and_update_entity_scoped_states_from_facts(
        self,
        *,
        facts: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        update_started_at = time.monotonic()
        if not self._enable_entity_scoped_state_resolution:
            report = {"enabled": 0, "updated": 0}
            self._log_info("memory_reflect", "entity_state_update_finish", {
                **report,
                "elapsed_ms": round(
                    (time.monotonic() - update_started_at) * 1000,
                    2,
                ),
            })
            return report
        existing_entity_states = self._db.get_recent_memory_states(
            state_type=sorted(self._entity_scoped_state_types()),
            limit=80,
        )
        candidates = self._build_entity_state_candidates_from_facts(facts)
        updated = 0
        for candidate in candidates:
            existing_state, match_info = self._match_entity_state_candidate_existing_states(
                candidate=candidate,
                existing_entity_states=existing_entity_states,
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
                self._log_info(
                    "memory_reflect",
                    "entity_state_updated",
                    self._state_update_log_payload(
                        state_id=state_id,
                        state_update=state_update,
                        candidate=candidate,
                        existing_state=existing_state,
                    ),
                )
                updated += 1
                refreshed_state = self._db.get_memory_state_by_id(state_id)
                if refreshed_state:
                    replaced = False
                    for index, existing in enumerate(existing_entity_states):
                        if int(existing.get("id") or -1) == int(state_id):
                            existing_entity_states[index] = refreshed_state
                            replaced = True
                            break
                    if not replaced:
                        existing_entity_states.append(refreshed_state)
        report = {"enabled": 1, "candidate_count": len(candidates), "updated": updated}
        self._log_info("memory_reflect", "entity_state_update_finish", {
            **report,
            "elapsed_ms": round(
                (time.monotonic() - update_started_at) * 1000,
                2,
            ),
        })
        return report

    def _state_update_log_payload(
        self,
        *,
        state_id: int,
        state_update: Dict[str, Any],
        candidate: Dict[str, Any],
        existing_state: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        facts = list(candidate.get("facts") or [])
        return {
            "state_id": state_id,
            "candidate": {
                "state_scope": candidate.get("state_scope"),
                "state_type": candidate.get("state_type"),
                "source_type": candidate.get("source_type"),
                "entity": candidate.get("entity"),
                "entity_key": candidate.get("entity_key"),
                "attribute_name": candidate.get("attribute_name"),
                "fact_ids": candidate.get("fact_ids") or [],
            },
            "existing_state": self._state_log_view(existing_state),
            "participating_facts": [
                self._fact_log_view_for_state_update(fact)
                for fact in facts
            ],
            "updated_state": self._state_log_view(
                {
                    **state_update,
                    "id": state_id,
                }
            ),
        }

    def _fact_log_view_for_state_update(self, fact: Dict[str, Any]) -> Dict[str, Any]:
        metadata = fact.get("metadata") if isinstance(fact.get("metadata"), dict) else {}
        return {
            "id": fact.get("id"),
            "episode_id": fact.get("episode_id"),
            "source_type": fact.get("source_type"),
            "fact_type": fact.get("fact_type"),
            "fact_kind": fact.get("fact_kind"),
            "event_time_key": fact.get("event_time_key"),
            "dialogue_time_key": fact.get("dialogue_time_key"),
            "summary": self._format_log_text(fact.get("summary") or "", limit=1200),
            "keywords": fact.get("keywords"),
            "entities": fact.get("entities") or [],
            "primary_entity": fact.get("primary_entity"),
            "fact_root_topic": fact.get("fact_root_topic") or "",
            "fact_aspect_topic": fact.get("fact_aspect_topic") or "",
            "state_aspects": fact.get("state_aspects") or metadata.get("state_aspects") or [],
            "actionable_aspects": fact.get("actionable_aspects") or metadata.get("actionable_aspects") or [],
            "confidence": fact.get("confidence"),
            "importance": fact.get("importance"),
        }

    def _state_log_view(self, state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not state:
            return {}
        metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
        return {
            "id": state.get("id"),
            "state_scope": state.get("state_scope"),
            "state_type": state.get("state_type"),
            "source_type": state.get("source_type"),
            "entity_key": state.get("entity_key") or metadata.get("entity_key") or "",
            "canonical_name": state.get("canonical_name"),
            "summary": self._format_log_text(state.get("summary") or "", limit=1200),
            "time_line": self._normalize_time_line(
                state.get("time_line"),
                limit=20,
                max_chars=2400,
            ),
            "evidence_fact_ids": state.get("evidence_fact_ids") or [],
            "confidence": state.get("confidence"),
            "metadata": metadata,
        }

    @staticmethod
    def _entity_scoped_state_types() -> set[str]:
        return {"preference", "profile", "routine", "relationship", "constraint", "risk"}

    def _build_entity_state_candidates_from_facts(
        self,
        facts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        grouped: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        for fact in facts:
            source_type = fact.get("source_type")
            state_aspects = self._state_aspects_from_fact(fact)
            if state_aspects:
                for aspect in state_aspects:
                    state_type = str(aspect.get("state_type") or "").strip().lower()
                    if state_type not in self._entity_scoped_state_types():
                        continue
                    entities = self._entities_for_state_aspect(aspect, fact)
                    if not entities:
                        continue
                    attribute_name = _compact_whitespace(aspect.get("attribute_name") or "")
                    if not attribute_name:
                        continue
                    for entity in entities[:1]:
                        entity_key = self._generate_entity_name_key(entity)
                        attribute_key = self._generate_topic_name_key(attribute_name)
                        key = (source_type, state_type, entity_key, attribute_key)
                        item = grouped.setdefault(key, {
                            "source_type": source_type,
                            "state_scope": "entity_state",
                            "state_type": state_type,
                            "entity": entity,
                            "entity_key": entity_key,
                            "attribute_key": attribute_key,
                            "attribute_name": attribute_name,
                            "attribute_name_aliases": [attribute_name],
                            "facts": [],
                            "fact_ids": [],
                            "state_aspects": [],
                        })
                        item["facts"].append(fact)
                        fact_id = None
                        if str(fact.get("id") or "").strip().isdigit():
                            fact_id = int(fact["id"])
                            item["fact_ids"].append(fact_id)
                        item["state_aspects"].append({
                            **aspect,
                            "fact_id": fact_id,
                            "fact_summary": fact.get("summary") or "",
                            "fact_event_time_key": fact.get("event_time_key") or "",
                            "fact_dialogue_time_key": fact.get("dialogue_time_key") or "",
                        })
        candidates: List[Dict[str, Any]] = []
        for item in grouped.values():
            item["fact_ids"] = list(dict.fromkeys(item.get("fact_ids") or []))
            if not item["fact_ids"]:
                continue
            aspect_summaries = [
                _compact_whitespace(aspect.get("aspect_summary") or "")
                for aspect in item.get("state_aspects") or []
                if _compact_whitespace(aspect.get("aspect_summary") or "")
            ]
            aspect_evidence = [
                _compact_whitespace(aspect.get("evidence_basis") or "")
                for aspect in item.get("state_aspects") or []
                if _compact_whitespace(aspect.get("evidence_basis") or "")
            ]
            item["summary_text"] = "\n".join(
                aspect_summaries
                or [str(fact.get("summary") or "") for fact in item.get("facts") or []]
            )[:2400]
            item["identity_text"] = "\n".join([
                str(item.get("attribute_name") or ""),
                " ".join(item.get("attribute_name_aliases") or []),
                item["summary_text"],
                "\n".join(aspect_evidence),
            ])[:2800]
            candidates.append(item)
        return candidates

    def _state_aspects_from_fact(self, fact: Dict[str, Any]) -> List[Dict[str, Any]]:
        metadata = fact.get("metadata") if isinstance(fact.get("metadata"), dict) else {}
        raw = fact.get("state_aspects") or metadata.get("state_aspects")
        fallback_entity = fact.get("primary_entity")
        return self._normalize_state_aspects(raw, fallback_entity=fallback_entity)

    def _entities_for_state_aspect(
        self,
        aspect: Dict[str, Any],
        fact: Dict[str, Any],
    ) -> List[str]:
        entity = aspect.get("entity") or aspect.get("primary_entity")
        if isinstance(entity, dict):
            name = _compact_whitespace(entity.get("name") or entity.get("text") or "")
        else:
            name = _compact_whitespace(entity)
        if name:
            return [name]
        return self._entities_for_entity_state_fact(fact)

    def _entities_for_entity_state_fact(
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
        if not entities:
            entities.extend(self._fact_topic_names(fact))
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
    def _generate_entity_name_key(value: Any) -> str:
        return _compact_whitespace(value).lower()

    def _match_entity_state_candidate_existing_states(
        self,
        *,
        candidate: Dict[str, Any],
        existing_entity_states: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        candidate_name = str(candidate.get("attribute_name") or "")
        candidate_entity = str(candidate.get("entity") or "")
        candidate_type = str(candidate.get("state_type") or "")
        candidate_entity_key = self._generate_entity_name_key(candidate.get("entity_key") or candidate_entity)
        candidate_attribute_aliases = self._normalize_unique_labels([
            candidate.get("attribute_name"),
            *(candidate.get("attribute_name_aliases") or []),
        ], limit=12)
        candidate_identity_text = str(candidate.get("identity_text") or candidate.get("summary_text") or "")
        best_state: Optional[Dict[str, Any]] = None
        best_score = 0.0
        best_info: Dict[str, Any] = {"matched": False, "score": 0.0}
        matching_states: List[Dict[str, Any]] = []
        for state in existing_entity_states:
            if str(state.get("source_type") or "") != str(candidate.get("source_type") or ""):
                continue
            if str(state.get("state_scope") or "") != "entity_state":
                continue
            if str(state.get("state_type") or "") != candidate_type:
                continue
            metadata = state.get("metadata") or {}
            state_entity_key = self._generate_entity_name_key(
                metadata.get("entity_key")
                or metadata.get("entity")
                or candidate_entity
            )
            if state_entity_key != candidate_entity_key:
                continue
            matching_states.append(state)

        if not matching_states:
            return None, best_info

        candidate_name_embedding: Optional[np.ndarray] = None
        candidate_identity_embedding: Optional[np.ndarray] = None
        if any(state.get("canonical_name_embedding") is not None for state in matching_states):
            candidate_name_embedding = self._generate_embedding_vector(
                candidate.get("attribute_name") or candidate_name
            )
        if any(
            state.get("identity_text_embedding") is not None
            for state in matching_states
        ):
            candidate_identity_embedding = self._generate_embedding_vector(candidate_identity_text[:1600])

        for state in matching_states:
            metadata = state.get("metadata") or {}
            state_attribute_aliases = self._normalize_unique_labels([
                *(metadata.get("attribute_name_aliases") or []),
                state.get("canonical_name"),
            ], limit=16)
            attribute_overlap = self._topic_name_overlap(
                candidate_attribute_aliases,
                state_attribute_aliases,
            )
            identity_embedding_similarity = self._cal_embedding_similarity(
                candidate_identity_embedding,
                state.get("identity_text_embedding"),
            )
            canonical_name_embedding_similarity = self._cal_embedding_similarity(
                candidate_name_embedding,
                state.get("canonical_name_embedding"),
            )
            embedding_similarity = max(
                identity_embedding_similarity,
                canonical_name_embedding_similarity,
            )
            exact_attribute_match = any(
                self._generate_topic_name_key(left) == self._generate_topic_name_key(right)
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
            )
            if score > best_score:
                best_score = score
                best_state = state
                best_info = {
                    "matched": matched,
                    "score": round(score, 4),
                    "attribute_overlap": round(attribute_overlap, 4),
                    "embedding_similarity": round(embedding_similarity, 4),
                    "identity_embedding_similarity": round(identity_embedding_similarity, 4),
                    "canonical_name_embedding_similarity": round(
                        canonical_name_embedding_similarity,
                        4,
                    ),
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
                "identity_embedding_similarity": best_info.get(
                    "identity_embedding_similarity",
                    0.0,
                ),
                "canonical_name_embedding_similarity": best_info.get(
                    "canonical_name_embedding_similarity",
                    0.0,
                ),
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
                "attribute_name": candidate.get("attribute_name"),
                "attribute_key": candidate.get("attribute_key"),
                "attribute_name_aliases": candidate.get("attribute_name_aliases", []),
                "state_aspect_summaries": [
                    {
                        "fact_id": aspect.get("fact_id"),
                        "aspect_summary": aspect.get("aspect_summary") or "",
                        "evidence_basis": aspect.get("evidence_basis") or "",
                        "confidence": aspect.get("confidence"),
                    }
                    for aspect in candidate.get("state_aspects") or []
                    if isinstance(aspect, dict)
                ],
            }, ensure_ascii=False, indent=2))
            .replace("{existing_entity_state}", json.dumps(
                self._format_existing_entity_state_for_prompt(existing_state),
                ensure_ascii=False,
                indent=2,
            ))
        )
        result = self._call_llm(prompt)
        parsed = self._parse_json_object_from_llm_text(result or "")
        if parsed:
            if not self._config_bool(parsed.get("update_needed", True), True):
                self._logger.debug(
                    "LLM declined entity-state update for entity=%s attribute=%s",
                    candidate.get("entity"),
                    candidate.get("attribute_name"),
                )
                return None
            normalized = self._normalize_entity_state_update_payload(
                parsed,
                candidate=candidate,
                existing_state=existing_state,
            )
            if normalized:
                return normalized
        return self._fallback_entity_state_update(candidate, existing_state)

    def _entity_state_canonical_topics_from_facts(
        self,
        *,
        candidate: Dict[str, Any],
        existing_state: Optional[Dict[str, Any]],
    ) -> List[str]:
        fact_topics = [
            topic
            for fact in candidate.get("facts") or []
            if isinstance(fact, dict)
            for topic in (
                fact.get("fact_root_topic"),
                fact.get("fact_aspect_topic"),
            )
        ]
        return self._normalize_unique_labels([
            *fact_topics,
            *((existing_state or {}).get("canonical_topics") or []),
        ], limit=24)

    def _normalize_entity_state_update_payload(
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
        ] or list(candidate.get("fact_ids") or [])[:24]
        existing_ids = [
            int(value)
            for value in ((existing_state or {}).get("evidence_fact_ids") or [])
            if str(value).strip().isdigit()
        ]
        evidence_ids = list(dict.fromkeys([*existing_ids, *evidence_ids]))[:80]
        canonical_name = (
            _compact_whitespace((existing_state or {}).get("canonical_name") or "")
            or _compact_whitespace(candidate.get("attribute_name") or "")
            or _compact_whitespace(raw.get("canonical_name") or "")
        )
        if canonical_name.lower() in self._entity_scoped_state_types() or len(canonical_name) < 3:
            canonical_name = _compact_whitespace(candidate.get("attribute_name") or "")
        existing_metadata = dict((existing_state or {}).get("metadata") or {})
        attribute_name_aliases = self._normalize_unique_labels([
            (existing_state or {}).get("canonical_name"),
            *(existing_metadata.get("attribute_name_aliases") or []),
            candidate.get("attribute_name"),
        ], limit=16)
        canonical_topics = self._entity_state_canonical_topics_from_facts(
            candidate=candidate,
            existing_state=existing_state,
        )
        time_line = self._build_state_time_line(
            raw_updates=raw.get("time_line"),
            candidate=candidate,
            existing_state=existing_state,
        )
        entity_name = self._normalize_entity_names([
            candidate.get("entity")
            or existing_metadata.get("entity")
            or candidate.get("entity_key")
        ], limit=1)
        return {
            "state_scope": "entity_state",
            "state_type": candidate["state_type"],
            "source_type": candidate.get("source_type") or (existing_state or {}).get("source_type"),
            "canonical_name": canonical_name,
            "summary": summary,
            "time_line": time_line,
            "evidence_fact_ids": evidence_ids,
            "keywords": self._normalize_string_list(raw.get("keywords"), limit=18),
            "entities": entity_name,
            "canonical_topics": canonical_topics,
            "importance": self._clamp_float(raw.get("importance"), 0.0, 1.0, 0.68),
            "confidence": self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.74),
            "status": _compact_whitespace(raw.get("status") or "active") or "active",
            "metadata": {
                "entity": candidate.get("entity"),
                "entity_key": candidate.get("entity_key"),
                "attribute_name_aliases": attribute_name_aliases,
                "extractor": "entity_scoped_state_update",
            },
        }

    def _fallback_entity_state_update(
        self,
        candidate: Dict[str, Any],
        existing_state: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        summaries = [
            _compact_whitespace(aspect.get("aspect_summary") or "")
            for aspect in candidate.get("state_aspects") or []
            if isinstance(aspect, dict)
            and _compact_whitespace(aspect.get("aspect_summary") or "")
        ][:5]
        if not summaries:
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
        existing_metadata = dict((existing_state or {}).get("metadata") or {})
        attribute_name_aliases = self._normalize_unique_labels([
            (existing_state or {}).get("canonical_name"),
            *(existing_metadata.get("attribute_name_aliases") or []),
            candidate.get("attribute_name"),
        ], limit=16)
        canonical_topics = self._entity_state_canonical_topics_from_facts(
            candidate=candidate,
            existing_state=existing_state,
        )
        existing_ids = [
            int(value)
            for value in ((existing_state or {}).get("evidence_fact_ids") or [])
            if str(value).strip().isdigit()
        ]
        evidence_ids = list(dict.fromkeys([*existing_ids, *(candidate.get("fact_ids") or [])]))[:80]
        return {
            "state_scope": "entity_state",
            "state_type": candidate["state_type"],
            "source_type": candidate.get("source_type") or (existing_state or {}).get("source_type"),
            "canonical_name": _compact_whitespace(
                (existing_state or {}).get("canonical_name")
                or candidate.get("attribute_name")
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
            "canonical_topics": canonical_topics,
            "importance": 0.66,
            "confidence": 0.58,
            "status": "active",
            "metadata": {
                "entity": candidate.get("entity"),
                "entity_key": candidate.get("entity_key"),
                "attribute_name_aliases": attribute_name_aliases,
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
            self._logger.debug("Skipping state with invalid scope/type: %s", state)
            return 0
        if state_scope == "topic_state" and state_type != "topic":
            self._logger.debug("Skipping topic state with non-topic type: %s", state)
            return 0
        if state_scope == "entity_state" and state_type not in self._entity_scoped_state_types():
            self._logger.debug("Skipping entity state with invalid type: %s", state)
            return 0
        keywords = state.get("keywords") or self._keywords(state["summary"], limit=18)
        entities = state.get("entities") or self._entities(state["summary"])
        canonical_topics = state.get("canonical_topics") or [state["canonical_name"]]
        evidence_fact_ids = [int(value) for value in state.get("evidence_fact_ids") or []]
        state_metadata = dict(state.get("metadata") or {})

        entity_key = ""
        if state_scope == "entity_state":
            entity_key = self._generate_entity_name_key(
                state_metadata.get("entity_key")
                or state_metadata.get("entity")
                or ""
            )
        entity_names = list(entities or [])
        if entity_key:
            entity_names.append(entity_key)
        entity_ids = self._entity_ids_for_names(entity_names, limit=64)
        identity_text = "\n".join([
            state["canonical_name"],
            state["summary"],
            f"keywords: {' '.join(keywords)}",
            f"entities: {', '.join(entities)}",
        ])
        if state_scope == "topic_state":
            identity_text = "\n".join([
                identity_text,
                f"parent_topics: {', '.join(state_metadata.get('parent_topics') or [])}",
                f"aspects: {', '.join(state_metadata.get('aspect_topic_names') or [])}",
        ])
        identity_text_embedding = self._generate_embedding_vector(identity_text)
        canonical_name_embedding = self._generate_embedding_vector(state["canonical_name"])
        state_id = self._db.upsert_state(
            state_scope=state_scope,
            state_type=state_type,
            source_type=state["source_type"],
            entity_key=entity_key,
            canonical_name=state["canonical_name"],
            summary=state["summary"],
            time_line=state.get("time_line") or [],
            entity_ids=entity_ids,
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
            identity_text_embedding=identity_text_embedding,
            canonical_name_embedding=canonical_name_embedding,
            identity_text=identity_text,
        )
        return state_id

    def _extract_actionable_items_with_llm(
        self,
        *,
        facts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not facts:
            self._log_info("memory_reflect", "actionable_llm_skipped", {
                "reason": "no_candidate_facts",
            })
            return []
        prompt_language = self._resolve_prompt_language_from_text(
            "\n".join(str(item.get("summary") or "") for item in facts[:20])
        )
        prompt_template = (
            UNIFIED_ACTIONABLE_ITEM_EXTRACTION_PROMPT_EN
            if prompt_language == "en"
            else UNIFIED_ACTIONABLE_ITEM_EXTRACTION_PROMPT_ZH
        )
        prompt = prompt_template.replace("{facts}", self._format_facts_for_actionable_prompt(facts))
        self._log_info("memory_reflect", "actionable_llm_call_start", {
            "candidate_fact_count": len(facts),
            "candidate_fact_ids": [
                fact.get("id") for fact in facts if fact.get("id") is not None
            ],
            "prompt_language": prompt_language,
            "prompt_chars": len(prompt),
        })
        result = self._call_llm(prompt)
        parsed = self._parse_json_object_from_llm_text(result or "")
        if not parsed:
            self._log_info("memory_reflect", "actionable_llm_parse_failed", {
                "candidate_fact_count": len(facts),
                "response_chars": len(result or ""),
                "response_preview": self._format_log_text(result or "", limit=600),
            })
            return []
        raw_items = parsed.get("actionable_items")
        if not isinstance(raw_items, list):
            self._log_info("memory_reflect", "actionable_llm_schema_failed", {
                "candidate_fact_count": len(facts),
                "parsed_keys": sorted(str(key) for key in parsed.keys()),
            })
            return []
        normalized: List[Dict[str, Any]] = []
        valid_fact_ids = {int(item["id"]) for item in facts if item.get("id") is not None}
        facts_by_id = {int(item["id"]): item for item in facts if item.get("id") is not None}
        seen_keys: set[str] = set()
        rejected_counts: Counter[str] = Counter()
        for raw in raw_items:
            if not isinstance(raw, dict):
                rejected_counts["non_object"] += 1
                continue
            summary = _compact_whitespace(raw.get("summary") or "")
            canonical_name = _compact_whitespace(raw.get("canonical_name") or "")
            if not summary or not canonical_name:
                rejected_counts["missing_summary_or_name"] += 1
                continue
            evidence_ids = [
                int(value)
                for value in (raw.get("evidence_fact_ids") or [])
                if str(value).strip().isdigit() and int(value) in valid_fact_ids
            ]
            if not evidence_ids:
                rejected_counts["missing_valid_evidence"] += 1
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
                rejected_counts["low_value"] += 1
                continue
            dedupe_key = self._actionable_dedupe_key(item)
            if dedupe_key in seen_keys:
                rejected_counts["duplicate"] += 1
                continue
            seen_keys.add(dedupe_key)
            normalized.append(item)
        self._log_info("memory_reflect", "actionable_llm_normalized", {
            "candidate_fact_count": len(facts),
            "raw_item_count": len(raw_items),
            "normalized_item_count": len(normalized),
            "rejected_counts": dict(rejected_counts),
        })
        return normalized

    def _filter_facts_for_actionable_item_extraction(
        self,
        facts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        seen_ids: set[int] = set()
        for fact in facts:
            fact_id = fact.get("id")
            numeric_id = int(fact_id) if str(fact_id or "").strip().isdigit() else 0
            if numeric_id and numeric_id in seen_ids:
                continue
            if self._actionable_aspects_from_fact(fact) or self._fact_can_seed_actionable_item(fact):
                selected.append(fact)
                if numeric_id:
                    seen_ids.add(numeric_id)
        limit = max(
            1,
            int(self._memory_cfg.get("actionable_fact_candidate_limit", 40) or 40),
        )
        selected = selected[:limit]
        aspect_seed_count = sum(
            1 for fact in selected
            if self._actionable_aspects_from_fact(fact)
        )
        self._log_info("memory_reflect", "actionable_fact_candidates", {
            "candidate_count": len(selected),
            "input_fact_count": len(facts),
            "candidate_limit": limit,
            "candidate_fact_ids": [
                fact.get("id")
                for fact in selected
                if fact.get("id") is not None
            ],
            "aspect_seed_count": aspect_seed_count,
            "heuristic_seed_count": max(0, len(selected) - aspect_seed_count),
        })
        return selected

    def _actionable_aspects_from_fact(self, fact: Dict[str, Any]) -> List[Dict[str, Any]]:
        metadata = fact.get("metadata") if isinstance(fact.get("metadata"), dict) else {}
        raw = fact.get("actionable_aspects") or metadata.get("actionable_aspects")
        return self._normalize_actionable_aspects(raw)

    def _fact_can_seed_actionable_item(self, fact: Dict[str, Any]) -> bool:
        kind = str(fact.get("fact_kind") or "").strip().lower()
        fact_type = str(fact.get("fact_type") or "").strip().lower()
        summary = str(fact.get("summary") or "")
        all_text = f"{summary}".lower()
        if not summary:
            return False
        if any(pattern in all_text for pattern in _WEAK_TRY_PATTERNS):
            if not self._has_explicit_followup_or_commitment(all_text, item_type="task"):
                return False
        if self._has_actionable_hard_marker(all_text):
            return True
        if kind in {"commitment", "open_question", "request"}:
            return True
        if kind == "decision":
            return self._has_strong_decision_marker(all_text)
        if kind == "risk":
            return self._has_blocking_marker(all_text)
        primary_entity = fact.get("primary_entity")
        primary_entity_name = (
            primary_entity.get("name")
            if isinstance(primary_entity, dict)
            else primary_entity
        )
        if kind == "instruction" and str(primary_entity_name or "").strip().lower() in {
            "user", "the user", "用户", "assistant", "the assistant", "助手", "system", "系统",
        }:
            return True
        if kind == "action" and fact_type == "episodic":
            return self._has_explicit_followup_or_commitment(all_text, item_type="task")
        if kind == "recommendation":
            return self._has_explicit_followup_or_commitment(all_text, item_type="recommendation")
        return False

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
        evidence_facts = self._db.memory_facts_by_ids(evidence_fact_ids)
        entity_ids = self._entity_ids_from_names_and_facts(
            names=[*entities, item.get("owner") or ""],
            facts=evidence_facts,
        )
        identity_text = "\n".join([
            item["canonical_name"],
            item["summary"],
            f"item_type: {item['item_type']}",
            f"owner: {item['owner']}",
            f"status: {item['status']}",
            f"due_at: {item['due_at']}",
            f"keywords: {' '.join(keywords)}",
            f"entities: {', '.join(entities)}",
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
                "entities": entities,
                "canonical_topics": canonical_topics,
            },
            identity_text_embedding=identity_text_embedding,
            identity_text=identity_text,
        )

        return item_id
    
    def _format_facts_for_actionable_prompt(self, facts: List[Dict[str, Any]]) -> str:
        rows: List[Dict[str, Any]] = []
        for fact in facts[:80]:
            metadata = fact.get("metadata") if isinstance(fact.get("metadata"), dict) else {}
            actionable_aspects = self._actionable_aspects_from_fact(fact)
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
                "actionable_aspects": actionable_aspects,
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
        if text in {"用户", "user", "the user"}:
            return "user"
        if text in {"助手", "assistant", "agent", "the assistant"}:
            return "assistant"
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
                "score": item.get(
                    "_recall_score",
                    item.get("_recall_type_score"),
                ),
                "type_score": item.get("_recall_type_score"),
                "rank": item.get("_recall_rank"),
                "embedding_similarity": item.get("embedding_similarity"),
                "bm25_score": (
                    item.get("_recall_score_components") or {}
                ).get("bm25_identity_text"),
                "score_components": item.get("_recall_score_components") or {},
                "provenance": {
                    key: value
                    for key, value in (item.get("_recall_provenance") or {}).items()
                    if key in {
                        "candidate_source",
                        "stage2_provenance",
                        "direct_index",
                        "evidence_expansion",
                        "episode_expansion",
                        "provenance_bonus",
                        "expansion_penalty",
                    }
                },
                "time_start": item.get("time_start"),
                "summary": self._format_log_text(summary, limit=240),
                "support_fact_ids": support_ids,
            })
        return {
            "count": len(items or []),
            "items": rows,
        }

    def _recall_detailed_candidate_items(
        self,
        items: Sequence[Dict[str, Any]],
        *,
        decision_by_target: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
        accepted_targets: Optional[set[Tuple[str, int]]] = None,
        selected_targets: Optional[set[Tuple[str, int]]] = None,
        stage: str = "",
    ) -> List[Dict[str, Any]]:
        """Serialize detailed recall diagnostics without embeddings or raw rows."""
        rows: List[Dict[str, Any]] = []
        decision_by_target = decision_by_target or {}
        has_accepted_targets = accepted_targets is not None
        has_selected_targets = selected_targets is not None
        accepted_targets = accepted_targets or set()
        selected_targets = selected_targets or set()
        for item in items or []:
            table = str(item.get("target_table") or "")
            target_id = str(item.get("target_id") or "")
            target_key = (table, target_id)
            try:
                target = (table, int(target_id))
            except (TypeError, ValueError):
                target = (table, -1)
            raw = item.get("_hydrated") if isinstance(item.get("_hydrated"), dict) else {}
            match_details = item.get("_recall_fast_match_details")
            match_details = dict(match_details) if isinstance(match_details, dict) else {}
            decision = dict(decision_by_target.get(target_key) or {})
            item_decision = item.get("_recall_decision")
            if isinstance(item_decision, dict):
                decision.update(item_decision)
            source = str(item.get("_recall_candidate_source") or "")
            provenance = list(item.get("_stage2_provenance") or [])
            reasons: List[str] = []
            for value in item.get("_recall_fast_match_evidence") or []:
                if str(value) and str(value) not in reasons:
                    reasons.append(str(value))
            for value in self._recall_candidate_source_channels(source):
                if value not in reasons:
                    reasons.append(value)
            if source.endswith("_evidence_expansion") and "evidence_expansion" not in reasons:
                reasons.append("evidence_expansion")
            if source.endswith("_episode_expansion") and "episode_expansion" not in reasons:
                reasons.append("episode_expansion")
            for value in provenance:
                if value not in reasons:
                    reasons.append(value)
            filter_reason = (
                decision.get("decision_reason")
                or match_details.get("filter_reason")
                or ""
            )
            accepted = decision.get("accepted")
            if accepted is None:
                accepted = target in accepted_targets if has_accepted_targets else None
            selected = target in selected_targets if has_selected_targets else None
            rows.append({
                "stage": stage,
                "target": f"{table}#{target_id}",
                "level": item.get("index_level") or item.get("_recall_type"),
                "source_type": item.get("source_type"),
                "candidate_source": source,
                "stage2_provenance": provenance,
                "retrieval_reasons": reasons,
                "title": self._format_log_text(
                    item.get("title") or raw.get("canonical_name") or "",
                    limit=240,
                ),
                "summary": self._format_log_text(
                    raw.get("summary")
                    or item.get("summary_for_retrieval")
                    or item.get("summary")
                    or "",
                    limit=500,
                ),
                "identity_text": self._format_log_text(
                    item.get("identity_text") or raw.get("identity_text") or "",
                    limit=500,
                ),
                "time_start": item.get("time_start"),
                "time_end": item.get("time_end"),
                "match": {
                    key: value
                    for key, value in match_details.items()
                    if key not in {"anchor"} or value
                },
                "score": item.get("_recall_score"),
                "type_score": item.get("_recall_type_score"),
                "embedding_similarity": item.get("embedding_similarity"),
                "bm25_raw_score": item.get("_bm25_score"),
                "bm25_score": item.get("_recall_bm25_score"),
                "score_components": item.get("_recall_score_components") or {},
                "provenance_profile": item.get("_recall_provenance") or {},
                "support_fact_ids": [
                    fact.get("id")
                    for fact in item.get("_supporting_facts") or []
                    if isinstance(fact, dict) and fact.get("id") is not None
                ],
                "episode_seed_targets": list(
                    item.get("_stage2_episode_seed_targets") or []
                ),
                "accepted": accepted,
                "selected": selected,
                "decision_reason": (
                    filter_reason
                    or item.get("_recall_drop_reason")
                    or (
                        "selected"
                        if selected is True
                        else "not_selected"
                        if selected is False
                        else ""
                    )
                ),
            })
        return rows

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
        top_k: int = None,
        budget: str = None,
        tags: Optional[List[str]] = None,
        time_end: Optional[str] = None,
        recall_gate_mode: Optional[str] = None,
        memory_source_override: Optional[Sequence[str]] = None,
        recall_path: str = "normal",
        prompt_language: str = "zh",
    ) -> Dict[str, Any]:
        """Run recall immediately against the latest committed memory snapshot."""
        # Recall is read-only and must not wait for the store/reflect worker's
        # long-running LLM or embedding work. A separate WAL reader gives it
        # a consistent committed snapshot without sharing the writer connection.
        with self._db.reader_transaction() as reader_db:
            return self._recall_sync(
                query=query,
                top_k=top_k,
                budget=budget,
                tags=tags,
                time_end=time_end,
                recall_gate_mode=recall_gate_mode,
                memory_source_override=memory_source_override,
                recall_path=recall_path,
                prompt_language=prompt_language,
                database=reader_db,
            )

    def _recall_sync(
        self,
        query: str,
        top_k: int = None,
        budget: str = None,
        tags: Optional[List[str]] = None,
        time_end: Optional[str] = None,
        recall_gate_mode: Optional[str] = None,
        memory_source_override: Optional[Sequence[str]] = None,
        recall_path: str = "normal",
        prompt_language: str = "zh",
        database: Optional[SessionDB] = None,
    ) -> Dict[str, Any]:
        requested_recall_path = str(recall_path or "normal").strip().lower()
        started_at = time.monotonic()
        if not self._memory_enabled or not str(query or "").strip():
            recall_report = {
                "memory_context": "",
                "requested_recall_path": requested_recall_path,
                "actual_recall_path": "none",
                "status": "empty",
                "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
            }
            self._operation_reporter.on_recall_finished(recall_report)
            return recall_report
        try:
            normalized_recall_path = requested_recall_path
            if normalized_recall_path not in {"stage1", "stage2", "normal"}:
                raise ValueError(
                    "recall_path must be one of: stage1, stage2, normal"
                )
            k = max(1, int(top_k or self._top_k or 8))
            b = str(budget or self._recall_budget or "mid")
            reference_time = self._normalize_recall_time_bound(
                time_end,
                default_to_now=True,
            )
            self._log_info("memory_recall", "start", {
                "query": self._format_log_text(query, limit=500),
                "top_k": k,
                "budget": b,
                "tags": tags or [],
                "requested_time_end": time_end,
                "time_end": reference_time,
                "recall_gate_mode": recall_gate_mode,
                "memory_source_override": list(memory_source_override or []),
                "recall_path": normalized_recall_path,
                "prompt_language": prompt_language,
            })

            parsed_time_start, parsed_time_end, clean_query = self._parse_time_expression(
                query,
                reference_time=reference_time,
            )
            temporal_bounds: RecallTimeBounds = (
                parsed_time_start,
                parsed_time_end,
            )
            temporal_mode = self._infer_recall_temporal_mode(query)
            search_query = clean_query or query
            self._log_info("memory_recall", "query_prepared", {
                "search_query": self._format_log_text(search_query, limit=500),
                "clean_query": self._format_log_text(clean_query, limit=500),
                "parsed_time_start": parsed_time_start,
                "parsed_time_end": parsed_time_end,
                "temporal_mode": temporal_mode,
            })

            memory_text: Optional[str]
            actual_recall_path: str
            if normalized_recall_path == "stage2":
                actual_recall_path = "stage2"
                stage1_report = self._process_recall_stage1(
                    query=search_query,
                    top_k=k,
                    budget=b,
                    temporal_bounds=temporal_bounds,
                    memory_source_override=memory_source_override,
                    temporal_mode=temporal_mode,
                    recent_reference_time=parsed_time_end,
                    prompt_language=prompt_language,
                    database=database,
                )
                memory_text = self._process_recall_stage2(
                    query=search_query,
                    top_k=k,
                    budget=b,
                    temporal_bounds=temporal_bounds,
                    memory_source_override=memory_source_override,
                    temporal_mode=temporal_mode,
                    stage1_report=stage1_report,
                    prompt_language=prompt_language,
                    database=database,
                )
            else:
                has_explicit_time_window = bool(
                    parsed_time_start
                )
                stage1_report = self._process_recall_stage1(
                    query=search_query,
                    top_k=k,
                    budget=b,
                    temporal_bounds=temporal_bounds,
                    memory_source_override=memory_source_override,
                    temporal_mode=temporal_mode,
                    recent_reference_time=(
                        parsed_time_end
                        if has_explicit_time_window
                        else datetime.now(timezone.utc).isoformat()
                    ),
                    prompt_language=prompt_language,
                    database=database,
                )
                if normalized_recall_path == "stage1":
                    actual_recall_path = "stage1"
                    memory_text = str(stage1_report.get("memory_context") or "")
                elif not stage1_report.get("trusted"):
                    actual_recall_path = "stage2"
                    memory_text = self._process_recall_stage2(
                        query=search_query,
                        analysis_query=query,
                        top_k=k,
                        budget=b,
                        temporal_bounds=temporal_bounds,
                        memory_source_override=memory_source_override,
                        temporal_mode=temporal_mode,
                        stage1_report=stage1_report,
                        prompt_language=prompt_language,
                        database=database,
                    )
                else:
                    memory_text = str(stage1_report.get("memory_context") or "")
                    actual_recall_path = "stage1"
            recall_status = "ok" if memory_text else "empty"
            recall_report = {
                "memory_context": memory_text or "",
                "requested_recall_path": normalized_recall_path,
                "actual_recall_path": actual_recall_path,
                "temporal_mode": temporal_mode,
                "status": recall_status,
                "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
                "recall_context_chars": len(memory_text or ""),
            }
            self._log_info("memory_recall", "finish", {
                "status": recall_status,
                "recall_path": normalized_recall_path,
                "actual_recall_path": actual_recall_path,
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

    def _process_recall_stage1(
        self,
        *,
        query: str,
        top_k: int,
        budget: str,
        temporal_bounds: RecallTimeBounds,
        memory_source_override: Optional[Sequence[str]] = None,
        temporal_mode: str = "dialogue_time",
        recent_reference_time: Optional[str] = None,
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
            query,
            keywords=[],
            entities=[],
        )
        is_contextual_query = self._recall_stage1_is_contextual_query(query)
        is_actionable_query = self._recall_stage1_is_actionable_query(query)

        raw_candidate_limits, layer_limits = self._recall_stage1_candidate_limits(
            top_k=top_k,
            is_contextual_query=is_contextual_query,
            is_actionable_query=is_actionable_query,
        )
        entity_mapping_candidate_limits, lexical_candidate_limits = (
            self._split_recall_stage1_source_limits(raw_candidate_limits)
        )
        self._log_info("memory_recall_stage1", "start", {
            "query": self._format_log_text(query, limit=500),
            "top_k": top_k,
            "budget": budget,
            "terms": terms,
            "time_start": (temporal_bounds or (None, None))[0],
            "time_end": (temporal_bounds or (None, None))[1],
            "temporal_mode": temporal_mode,
            "recent_reference_time": recent_reference_time,
            "memory_source_override": list(memory_source_override or []),
            "raw_candidate_limits": raw_candidate_limits,
            "entity_mapping_candidate_limits": entity_mapping_candidate_limits,
            "lexical_candidate_limits": lexical_candidate_limits,
        })

        lexical_fact_candidates, lexical_state_candidates, lexical_actionable_candidates = (
            self._retrieve_recall_raw_candidates_lexical_search(
                terms=terms,
                candidate_source_prefix="stage1",
                source_types=source_types,
                temporal_bounds=temporal_bounds,
                temporal_mode=temporal_mode,
                candidate_limits=lexical_candidate_limits,
                database=database,
            )
        )
        entity_fact_candidates, entity_state_candidates, entity_actionable_candidates = (
            self._retrieve_recall_entity_mapping_candidates(
                query=query,
                candidate_source_prefix="stage1",
                source_types=source_types,
                temporal_bounds=temporal_bounds,
                temporal_mode=temporal_mode,
                candidate_limits=entity_mapping_candidate_limits,
                database=database,
            )
        )
        raw_candidates, raw_candidates_by_level = (
            self._expand_and_merge_recall_stage1_raw_candidates(
                entity_candidates=[
                    *entity_fact_candidates,
                    *entity_state_candidates,
                    *entity_actionable_candidates,
                ],
                lexical_candidates=[
                    *lexical_fact_candidates,
                    *lexical_state_candidates,
                    *lexical_actionable_candidates,
                ],
                source_types=source_types,
                temporal_bounds=temporal_bounds,
                temporal_mode=temporal_mode,
                database=database,
            )
        )
        reference_time = recent_reference_time or datetime.now(timezone.utc).isoformat()
        (
            ranked_fact_candidates,
            ranked_state_candidates,
            ranked_actionable_items,
        ) = self._recall_stage1_calculate_candidate_matching_score(
            candidates=raw_candidates,
            query=query,
            search_terms=terms,
            is_contextual_query=is_contextual_query,
            reference_time=reference_time,
        )
        filtered_candidates: List[Dict[str, Any]] = [
            *ranked_fact_candidates,
            *ranked_state_candidates,
            *ranked_actionable_items,
        ]
        semantic_query = self._recall_stage1_requires_semantic_search(query)

        selected_candidates, selected_by_layer = (
            self._recall_get_selected_candidates_by_layer(
                ranked_fact_candidates=ranked_fact_candidates,
                ranked_state_candidates=ranked_state_candidates,
                ranked_actionable_items=ranked_actionable_items,
                layer_limits=layer_limits,
            )
        )

        evidence_profile = self._recall_stage1_build_evidence_profile(
            selected_candidates,
        )
        evidence_gate = bool(evidence_profile.get("trusted"))

        actionable_hit = any(
            self._recall_stage1_candidate_match_type(item)
            in {"high_priority_actionable", "exact_actionable"}
            for item in selected_candidates
        )
        trusted = bool(selected_candidates) and (
            evidence_gate
            or actionable_hit
        )
        memory_text = self._build_memory_retrieved_format_text(
            entries=selected_candidates,
            prompt_language=prompt_language,
        )
        stage1_finish_payload = {
            "status": "hit" if trusted and memory_text else "miss",
            "reason": "" if trusted and memory_text else (
                "empty_formatted_context"
                if not memory_text
                else "semantic_query_requires_stage2"
                if semantic_query
                else "evidence_profile_below_threshold"
            ),
            "raw_candidate_limits": raw_candidate_limits,
            "entity_mapping_candidate_limits": entity_mapping_candidate_limits,
            "lexical_candidate_limits": lexical_candidate_limits,
            "entity_mapping_fact_candidate_count": len(entity_fact_candidates),
            "entity_mapping_state_candidate_count": len(entity_state_candidates),
            "entity_mapping_actionable_candidate_count": len(entity_actionable_candidates),
            "lexical_fact_candidate_count": len(lexical_fact_candidates),
            "lexical_state_candidate_count": len(lexical_state_candidates),
            "lexical_actionable_candidate_count": len(lexical_actionable_candidates),
            "evidence_expansion_candidate_count": sum(
                1
                for candidate in raw_candidates
                if str(candidate.get("_recall_candidate_source") or "")
                == "stage1_evidence_expansion"
            ),
            "raw_candidate_counts_by_level": {
                level: len(candidates)
                for level, candidates in raw_candidates_by_level.items()
            },
            "filtered_candidate_count": len(filtered_candidates),
            "selected_count": len(selected_candidates),
            "trusted": trusted,
            "match_types": [
                self._recall_stage1_candidate_match_type(item)
                for item in selected_candidates
            ],
            "evidence_profile": evidence_profile,
            "layer_limits": layer_limits,
            "selected_by_layer": selected_by_layer,
            "selected_scores": self._recall_stage1_score_log(
                selected_candidates
            ),
            "targets": [
                f"{item.get('target_table')}#{item.get('target_id')}"
                for item in selected_candidates
            ],
            "retrieved_chars": len(memory_text),
            "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
        }
        if self._recall_detailed_logging:
            selected_targets = {
                (
                    str(candidate.get("target_table") or ""),
                    int(candidate.get("target_id")),
                )
                for candidate in selected_candidates
                if str(candidate.get("target_id") or "").isdigit()
            }
            filtered_targets = {
                (
                    str(candidate.get("target_table") or ""),
                    int(candidate.get("target_id")),
                )
                for candidate in filtered_candidates
                if str(candidate.get("target_id") or "").isdigit()
            }
            stage1_finish_payload["candidate_diagnostics"] = {
                "raw_candidates": self._recall_detailed_candidate_items(
                    raw_candidates,
                    accepted_targets=filtered_targets,
                    selected_targets=selected_targets,
                    stage="stage1_raw",
                ),
                "filtered_candidates": self._recall_detailed_candidate_items(
                    filtered_candidates,
                    accepted_targets=filtered_targets,
                    selected_targets=selected_targets,
                    stage="stage1_filtered",
                ),
                "selected_candidates": self._recall_detailed_candidate_items(
                    selected_candidates,
                    accepted_targets=filtered_targets,
                    selected_targets=selected_targets,
                    stage="stage1_selected",
                ),
            }
        self._log_info("memory_recall_stage1", "finish", stage1_finish_payload)
        return {
            "memory_context": memory_text or "",
            "raw_candidates": raw_candidates,
            "filtered_candidates": filtered_candidates,
            "selected_candidates": selected_candidates,
            "evidence_profile": evidence_profile,
            "trusted": bool(trusted and memory_text),
            "raw_candidate_count": len(raw_candidates),
            "filtered_candidate_count": len(filtered_candidates),
            "selected_count": len(selected_candidates),
            "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
        }

    def _expand_and_merge_recall_stage1_raw_candidates(
        self,
        *,
        entity_candidates: Sequence[Dict[str, Any]],
        lexical_candidates: Sequence[Dict[str, Any]],
        source_types: Optional[Sequence[str]] = None,
        temporal_bounds: RecallTimeBounds = None,
        temporal_mode: str = "dialogue_time",
        database: Optional[SessionDB] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """Merge Stage 1 candidates and expand state/actionable evidence facts."""
        raw_candidates: List[Dict[str, Any]] = []
        candidates_by_target: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for source_name, source_candidates in (
            ("stage1_entity_mapping", entity_candidates),
            ("stage1_lexical", lexical_candidates),
        ):
            for candidate in source_candidates:
                candidate["_recall_candidate_source"] = source_name
                self._merge_recall_stage1_candidate(
                    candidate=candidate,
                    source_name=source_name,
                    candidates_by_target=candidates_by_target,
                    raw_candidates=raw_candidates,
                )
        
        seed_candidates = [
            *list(entity_candidates or []),
            *list(lexical_candidates or []),
        ]
        evidence_candidates = self._expand_recall_evidence_facts_from_candidates(
            candidates=seed_candidates,
            candidate_source_prefix="stage1",
            source_types=source_types,
            temporal_bounds=temporal_bounds,
            temporal_mode=temporal_mode,
            limit=max(12, min(48, max(len(raw_candidates), 12))),
            database=database,
        )
        for evidence_candidate in evidence_candidates:
            self._merge_recall_stage1_candidate(
                candidate=evidence_candidate,
                source_name="stage1_evidence_expansion",
                candidates_by_target=candidates_by_target,
                raw_candidates=raw_candidates,
            )

        candidates_by_level: Dict[str, List[Dict[str, Any]]] = {
            "fact": [],
            "state": [],
            "actionable_item": [],
        }
        for candidate in raw_candidates:
            level = str(candidate.get("index_level") or "")
            if level in candidates_by_level:
                candidates_by_level[level].append(candidate)
        return raw_candidates, candidates_by_level

    def _merge_recall_stage1_candidate(
        self,
        *,
        candidate: Dict[str, Any],
        source_name: str,
        candidates_by_target: Dict[Tuple[str, int], Dict[str, Any]],
        raw_candidates: List[Dict[str, Any]],
    ) -> None:
        """Merge one Stage 1 candidate while preserving source provenance."""
        try:
            target = (
                str(candidate.get("target_table") or ""),
                int(candidate.get("target_id")),
            )
        except (TypeError, ValueError):
            return
        existing_candidate = candidates_by_target.get(target)
        if existing_candidate is None:
            candidates_by_target[target] = candidate
            raw_candidates.append(candidate)
            return

        existing_source = str(
            existing_candidate.get("_recall_candidate_source") or source_name
        )
        direct_source_names: set[str] = set()
        if existing_source == "stage1_both":
            direct_source_names.update({"stage1_entity_mapping", "stage1_lexical"})
        elif existing_source in {"stage1_entity_mapping", "stage1_lexical"}:
            direct_source_names.add(existing_source)
        if source_name == "stage1_both":
            direct_source_names.update({"stage1_entity_mapping", "stage1_lexical"})
        elif source_name in {"stage1_entity_mapping", "stage1_lexical"}:
            direct_source_names.add(source_name)
        if direct_source_names == {"stage1_entity_mapping", "stage1_lexical"}:
            existing_candidate["_recall_candidate_source"] = "stage1_both"
        elif direct_source_names:
            # A direct index hit remains the primary source when the same fact
            # is also returned through Stage 1 evidence expansion.
            existing_candidate["_recall_candidate_source"] = next(
                iter(direct_source_names)
            )
        elif existing_source == source_name:
            existing_candidate["_recall_candidate_source"] = existing_source
        else:
            existing_candidate["_recall_candidate_source"] = source_name

        try:
            existing_bm25 = float(existing_candidate.get("_bm25_score"))
        except (TypeError, ValueError):
            existing_bm25 = None
        try:
            candidate_bm25 = float(candidate.get("_bm25_score"))
        except (TypeError, ValueError):
            candidate_bm25 = None
        # SQLite FTS5 BM25 ranks lower values as more relevant.
        if candidate_bm25 is not None and (
            existing_bm25 is None or candidate_bm25 < existing_bm25
        ):
            existing_candidate["_bm25_score"] = candidate_bm25
            existing_candidate["_recall_bm25_score"] = self._clamp_float(
                candidate.get("_recall_bm25_score"),
                0.0,
                1.0,
                0.0,
            )
            existing_candidate["_recall_bm25_rank"] = candidate.get(
                "_recall_bm25_rank"
            )

    @staticmethod
    def _recall_stage1_candidate_limits(
        *,
        top_k: int,
        is_contextual_query: Optional[bool] = None,
        is_actionable_query: Optional[bool] = None,
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Build the raw retrieval and final per-layer Stage 1 budgets.

        Raw limits are the combined per-layer budget for the two direct Stage
        1 sources. The caller splits each raw limit between entity mapping and
        lexical retrieval. Layer limits are per-layer caps for the final
        selection. The final
        candidate count is the sum of these caps rather than ``top_k`` itself;
        ``top_k`` is used as the base cap for each memory layer.
        """
        k = max(1, int(top_k or 1))
        contextual = bool(is_contextual_query)
        actionable = bool(is_actionable_query)
        base_raw_limit = max(8, min(32, k * 3))
        raw_candidate_limits = {
            "facts": base_raw_limit,
            "states": base_raw_limit,
            "actionable_items": base_raw_limit,
        }

        # Contextual recall needs a wider recent-fact and active-topic pool;
        # actionable recall needs more actionable candidates for the
        # high-priority pass. These are raw retrieval budgets, not additional
        # final output slots.
        contextual_extra = max(4, min(16, k * 2))
        actionable_extra = max(4, min(16, k * 2))
        if contextual:
            raw_candidate_limits["facts"] += contextual_extra
            raw_candidate_limits["states"] += contextual_extra
        if actionable:
            raw_candidate_limits["actionable_items"] += actionable_extra
        raw_candidate_limits = {
            key: min(48, value)
            for key, value in raw_candidate_limits.items()
        }

        base_layer_limit = k
        contextual_layer_extra = int(math.ceil(base_layer_limit * 0.5))
        actionable_layer_extra = int(math.ceil(base_layer_limit * 0.5))
        limits = {
            "fact": base_layer_limit,
            "state": base_layer_limit,
            "actionable_item": base_layer_limit,
        }
        if contextual:
            limits["fact"] += contextual_layer_extra
            limits["state"] += contextual_layer_extra
        if actionable:
            limits["actionable_item"] += actionable_layer_extra
        return raw_candidate_limits, limits

    @staticmethod
    def _split_recall_stage1_source_limits(
        raw_candidate_limits: Dict[str, int],
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Split each Stage 1 raw budget between mapping and lexical search.

        The two direct retrieval sources receive complementary halves. When a
        budget is odd, lexical search receives the extra candidate so the two
        limits never exceed the combined raw budget.
        """
        entity_mapping_limits: Dict[str, int] = {}
        lexical_limits: Dict[str, int] = {}
        for key, raw_limit in (raw_candidate_limits or {}).items():
            normalized_limit = max(0, int(raw_limit or 0))
            entity_limit = normalized_limit // 2
            entity_mapping_limits[key] = entity_limit
            lexical_limits[key] = normalized_limit - entity_limit
        return entity_mapping_limits, lexical_limits

    @staticmethod
    def _recall_stage1_candidate_match_type(
        candidate: Dict[str, Any],
    ) -> str:
        """Read a candidate match type from details or fallback evidence."""
        details = candidate.get("_recall_fast_match_details")
        if isinstance(details, dict) and details.get("match_type"):
            return str(details.get("match_type"))
        known_types = {
            "exact_actionable",
            "exact_topic",
            "exact_name",
            "topic_overlap",
            "name_overlap",
            "entity_mapping",
            "bm25_lexical",
            "high_priority_actionable",
            "recent_fact_context",
            "recent_active_topic",
        }
        for value in candidate.get("_recall_fast_match_evidence") or []:
            if str(value) in known_types:
                return str(value)
        return ""

    @staticmethod
    def _recall_stage1_score_log(
        candidates: Sequence[Dict[str, Any]],
        *,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        """Return compact score evidence for Stage 1 diagnostics."""
        rows: List[Dict[str, Any]] = []
        for candidate in list(candidates)[: max(1, int(limit or 1))]:
            rows.append({
                "target": (
                    f"{candidate.get('target_table')}#"
                    f"{candidate.get('target_id')}"
                ),
                "match_type": MemoryNodeManager._recall_stage1_candidate_match_type(
                    candidate
                ),
                "candidate_source": candidate.get("_recall_candidate_source") or "",
                "evidence": list(
                    candidate.get("_recall_fast_match_evidence") or []
                ),
                "matched_keywords": list(
                    candidate.get("_recall_fast_matched_keywords") or []
                ),
                "rank_score": round(
                    float(candidate.get("_recall_score") or 0.0),
                    4,
                ),
                "candidate_score_passed": bool(
                    (candidate.get("_recall_fast_match_details") or {}).get(
                        "candidate_score_passed"
                    )
                ),
                "candidate_score_threshold": (
                    candidate.get("_recall_fast_match_details") or {}
                ).get("candidate_score_threshold"),
                "bm25_score": round(
                    float(candidate.get("_recall_bm25_score") or 0.0),
                    4,
                ),
                "bm25_raw_score": candidate.get("_bm25_score"),
            })
        return rows

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
        query: str,
        search_terms: Sequence[str] = (),
        is_contextual_query: bool = False,
        reference_time: Optional[str] = None,
    ) -> Tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
    ]:
        """Score, filter, and rank Stage 1 candidates by memory layer.

        Contextual recency and actionable priority are score components here;
        neither category bypasses the normal matching gate.
        """
        ranked_by_level: Dict[str, List[Dict[str, Any]]] = {
            "fact": [],
            "state": [],
            "actionable_item": [],
        }
        for candidate in candidates or []:
            match = self._recall_stage1_calculate_single_candidate_matching_score(
                candidate,
                query,
                search_terms=search_terms,
            )
            score_components = dict(match.get("score_components") or {})
            index_level = str(candidate.get("index_level") or "")
            effective_reference_time = (
                reference_time or datetime.now(timezone.utc).isoformat()
            )

            if index_level == "actionable_item":
                high_priority = self._recall_stage1_is_high_priority_actionable(
                    candidate,
                    reference_time=effective_reference_time,
                )
                high_priority_score = (
                    self._clamp_float(
                        self._memory_cfg.get(
                            "recall_fast_high_priority_actionable_score",
                            0.20,
                        ),
                        0.0,
                        1.0,
                        0.20,
                    )
                    if high_priority
                    else 0.0
                )
                score_components["high_priority_actionable"] = round(
                    high_priority_score,
                    4,
                )
                if high_priority and match.get("match_type"):
                    match.setdefault("evidence", []).append(
                        "high_priority_actionable"
                    )
                    if not match.get("strong_anchor"):
                        match["match_type"] = "high_priority_actionable"

            if is_contextual_query and index_level in {"fact", "state"}:
                if index_level == "fact":
                    is_recent = self._recall_stage1_is_recent_fact(
                        candidate,
                        reference_time=effective_reference_time,
                    )
                else:
                    is_recent = self._recall_stage1_is_recent_active_topic(
                        candidate,
                        reference_time=effective_reference_time,
                    )
                contextual_time_score = (
                    self._clamp_float(
                        self._memory_cfg.get(
                            "recall_fast_contextual_time_score",
                            0.20,
                        ),
                        0.0,
                        1.0,
                        0.20,
                    )
                    if is_recent
                    else 0.0
                )
                score_components["contextual_recency"] = round(
                    contextual_time_score,
                    4,
                )
                if contextual_time_score and match.get("match_type"):
                    match.setdefault("evidence", []).append(
                        "contextual_recency"
                    )

            rank_score = min(1.0, sum(score_components.values()))
            if score_components:
                match["score_components"] = score_components
                match["rank_score"] = round(rank_score, 4)
                candidate_min_score = self._clamp_float(
                    match.get("candidate_score_threshold"),
                    0.0,
                    1.0,
                    0.35,
                )
                match["candidate_score_passed"] = bool(
                    match.get("strong_anchor")
                    or (
                        bool(match.get("match_type"))
                        and rank_score >= candidate_min_score
                    )
                )
                match["matched"] = bool(
                    match.get("match_type")
                    and match.get("candidate_score_passed")
                )
                if not match["matched"] and match.get("filter_reason") == "":
                    match["filter_reason"] = "candidate_score_below_threshold"
            candidate["_recall_score"] = match.get("rank_score", 0.0)
            candidate["_recall_type_score"] = match.get("rank_score", 0.0)
            candidate["_recall_candidate_source"] = (
                candidate.get("_recall_candidate_source") or "stage1_lexical"
            )
            candidate["_recall_fast_match_evidence"] = list(
                match.get("evidence") or []
            )
            candidate["_recall_fast_matched_keywords"] = list(
                match.get("matched_keywords") or []
            )
            candidate["_recall_fast_match_details"] = dict(match)
            accepted = bool(
                match.get("matched")
                and match.get("candidate_score_passed", False)
            )
            candidate["_recall_decision"] = {
                "accepted": accepted,
                "decision_reason": (
                    "accepted"
                    if accepted
                    else str(match.get("filter_reason") or "not_matched")
                ),
            }
            if not (
                match.get("matched")
                and match.get("candidate_score_passed", False)
            ):
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
        return (
            ranked_by_level["fact"],
            ranked_by_level["state"],
            ranked_by_level["actionable_item"],
        )

    def _recall_get_selected_candidates_by_layer(
        self,
        *,
        ranked_fact_candidates: Sequence[Dict[str, Any]],
        ranked_state_candidates: Sequence[Dict[str, Any]],
        ranked_actionable_items: Sequence[Dict[str, Any]],
        layer_limits: Dict[str, int],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Select already-ranked candidates with per-layer preferred limits."""
        candidates_by_layer: Dict[str, Sequence[Dict[str, Any]]] = {
            "facts": ranked_fact_candidates,
            "states": ranked_state_candidates,
            "actionable_items": ranked_actionable_items,
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

        selected_by_layer: Dict[str, int] = {
            layer: 0 for layer in layer_limits
        }
        for layer, limit in layer_limits.items():
            for candidate in candidates_by_layer.get(layer, []):
                if selected_by_layer[layer] >= limit:
                    break
                if append_candidate(candidate):
                    selected_by_layer[layer] += 1

        # Reuse empty layer quota so a missing layer does not reduce the
        # available output capacity. The first pass enforces each layer's
        # preferred limit; this pass can use unused capacity from another
        # layer until the global sum of layer limits is reached.
        for layer in layer_limits:
            for candidate in candidates_by_layer.get(layer, []):
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
        return selected_candidates, selected_by_layer

    def _recall_stage1_calculate_single_candidate_matching_score(
        self,
        candidate: Dict[str, Any],
        query: str,
        *,
        search_terms: Sequence[str] = (),
    ) -> Dict[str, Any]:
        """Collect exact and lexical-overlap evidence for one candidate.

        This method deliberately does not decide whether the whole recall is
        trustworthy.  That decision is made from all filtered candidates by
        ``_recall_stage1_build_evidence_profile``.
        """
        raw = candidate.get("_hydrated") if isinstance(candidate.get("_hydrated"), dict) else {}
        metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        topic_values: List[str] = []
        name_values: List[str] = []

        def extend_values(target: List[str], value: Any) -> None:
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    extend_values(target, item)
                return
            anchor = self._recall_stage1_clean_anchor(value)
            if anchor and anchor not in target:
                target.append(anchor)

        index_level = str(candidate.get("index_level") or "")
        if index_level == "state":
            state_scope = str(raw.get("state_scope") or "")
            if state_scope == "topic_state":
                extend_values(topic_values, [
                    raw.get("canonical_name"),
                    metadata.get("canonical_topics"),
                    metadata.get("parent_topics"),
                    metadata.get("aspect_topic_names"),
                ])
            else:
                extend_values(name_values, [
                    raw.get("canonical_name"),
                    metadata.get("attribute_name_aliases"),
                ])
        elif index_level == "fact":
            extend_values(topic_values, [
                raw.get("fact_root_topic"),
                raw.get("fact_aspect_topic"),
            ])
        elif index_level == "actionable_item":
            extend_values(name_values, raw.get("canonical_name"))
            extend_values(topic_values, [
                metadata.get("canonical_topics"),
                raw.get("canonical_topics"),
            ])

        exact_topics = [
            anchor for anchor in topic_values
            if self._recall_stage1_contains_anchor(query, anchor)
        ]
        exact_names = [
            anchor for anchor in name_values
            if self._recall_stage1_contains_anchor(query, anchor)
        ]

        normalized_search_terms = [
            cleaned
            for value in search_terms or ()
            if (cleaned := self._recall_stage1_clean_anchor(value))
        ]

        topic_overlap = self._topic_name_overlap(
            normalized_search_terms,
            topic_values,
            allow_substring=False,
        ) if normalized_search_terms and topic_values else 0.0
        name_overlap = self._topic_name_overlap(
            normalized_search_terms,
            name_values,
            allow_substring=False,
        ) if normalized_search_terms and name_values else 0.0
        
        strong_overlap_threshold = 0.90
        strong_topic_overlap = topic_overlap >= strong_overlap_threshold
        strong_name_overlap = name_overlap >= strong_overlap_threshold

        evidence: List[str] = []
        if exact_topics:
            evidence.append("exact_topic")
        if exact_names:
            evidence.append("exact_name")
        if topic_overlap > 0.0:
            evidence.append("topic_overlap")
        if name_overlap > 0.0:
            evidence.append("name_overlap")
        bm25_score = self._clamp_float(
            candidate.get("_recall_bm25_score"),
            0.0,
            1.0,
            0.0,
        )
        candidate_source = str(candidate.get("_recall_candidate_source") or "")
        candidate_sources = self._recall_candidate_source_channels(candidate_source)
        entity_mapping_hit = "entity_mapping" in candidate_sources
        lexical_source_hit = "lexical" in candidate_sources
        if entity_mapping_hit:
            evidence.append("entity_mapping")

        if exact_topics:
            exact_topic_score = min(
                0.54,
                0.48 + 0.03 * (len(exact_topics) - 1),
            )
        else:
            exact_topic_score = 0.0
        # A lexical overlap is weaker than an exact query anchor, but it must
        # still affect ranking when no exact topic was found.
        topic_overlap_score = (
            min(0.36, 0.36 * topic_overlap)
            if not exact_topics
            else 0.0
        )
        if exact_names:
            exact_name_score = min(
                0.48,
                0.42 + 0.03 * (len(exact_names) - 1),
            )
        else:
            exact_name_score = 0.0
        name_overlap_score = (
            min(0.30, 0.30 * name_overlap)
            if not exact_names
            else 0.0
        )
        bm25_component = 0.45 * bm25_score if lexical_source_hit else 0.0
        mapping_score = 0.30 if entity_mapping_hit else 0.0
        score_components = {
            "exact_topic": round(exact_topic_score, 4),
            "exact_name": round(exact_name_score, 4),
            "topic_overlap": round(topic_overlap_score, 4),
            "name_overlap": round(name_overlap_score, 4),
            "entity_mapping": round(mapping_score, 4),
            "bm25_identity_text": round(bm25_component, 4),
        }
        rank_score = min(1.0, sum(score_components.values()))
        lexical_bm25_match = bool(
            lexical_source_hit
            and bm25_score >= 0.55
        )
        strong_anchor = bool(
            exact_topics
            or exact_names
            or strong_topic_overlap
            or strong_name_overlap
        )
        candidate_min_score = self._clamp_float(
            getattr(self, "_recall_fast_candidate_min_score", None),
            0.0,
            1.0,
            0.35,
        )
        candidate_score_passed = bool(
            strong_anchor or rank_score >= candidate_min_score
        )

        if index_level == "actionable_item" and (exact_topics or exact_names):
            match_type = "exact_actionable"
            anchor = max(
                [*exact_topics, *exact_names],
                key=len,
            )
        elif exact_topics:
            match_type = "exact_topic"
            anchor = max(exact_topics, key=len)
        elif exact_names:
            match_type = "exact_name"
            anchor = max(exact_names, key=len)
        elif entity_mapping_hit:
            match_type = "entity_mapping"
            anchor = ", ".join(candidate.get("_recall_entity_names") or []) or (
                candidate.get("title") or candidate.get("summary_for_retrieval") or ""
            )
        elif lexical_bm25_match:
            match_type = "bm25_lexical"
            anchor = (
                candidate.get("identity_text")
                or raw.get("identity_text")
                or candidate.get("summary_for_retrieval")
                or candidate.get("title")
                or ""
            )
        elif topic_overlap > 0.0:
            match_type = "topic_overlap"
            anchor = ", ".join(topic_values)
        elif name_overlap > 0.0:
            match_type = "name_overlap"
            anchor = ", ".join(name_values)
        else:
            return {
                "matched": False,
                "match_type": "",
                "anchor": "",
                "rank_score": 0.0,
                "evidence": [],
                "matched_keywords": [],
                "exact_topics": [],
                "exact_names": [],
                "candidate_source": candidate_source,
                "candidate_sources": sorted(candidate_sources - {""}),
                "entity_mapping_hit": entity_mapping_hit,
                "strong_anchor": False,
                "candidate_score_passed": False,
                "candidate_score_threshold": candidate_min_score,
                "filter_reason": "no_matching_evidence",
                "bm25_score": round(bm25_score, 4),
                "bm25_raw_score": candidate.get("_bm25_score"),
                "topic_overlap": round(topic_overlap, 4),
                "name_overlap": round(name_overlap, 4),
                "score_components": score_components,
            }

        if lexical_source_hit and bm25_score:
            evidence.append("bm25_identity_text")
        if not candidate_score_passed:
            return {
                "matched": False,
                "match_type": match_type,
                "anchor": anchor,
                "rank_score": round(rank_score, 4),
                "evidence": evidence,
                "matched_keywords": [],
                "exact_topics": exact_topics,
                "exact_names": exact_names,
                "candidate_source": candidate_source,
                "candidate_sources": sorted(candidate_sources - {""}),
                "entity_mapping_hit": entity_mapping_hit,
                "strong_anchor": strong_anchor,
                "candidate_score_passed": False,
                "candidate_score_threshold": candidate_min_score,
                "filter_reason": "candidate_score_below_threshold",
                "bm25_score": round(bm25_score, 4),
                "bm25_raw_score": candidate.get("_bm25_score"),
                "topic_overlap": round(topic_overlap, 4),
                "name_overlap": round(name_overlap, 4),
                "score_components": score_components,
            }
        return {
            "matched": True,
            "match_type": match_type,
            "anchor": anchor,
            "rank_score": round(rank_score, 4),
            "evidence": evidence,
            "matched_keywords": [],
            "exact_topics": exact_topics,
            "exact_names": exact_names,
            "candidate_source": candidate_source,
            "candidate_sources": sorted(candidate_sources - {""}),
            "entity_mapping_hit": entity_mapping_hit,
            "candidate_score_passed": True,
            "candidate_score_threshold": candidate_min_score,
            "filter_reason": "",
            "bm25_score": round(bm25_score, 4),
            "bm25_raw_score": candidate.get("_bm25_score"),
            "topic_overlap": round(topic_overlap, 4),
            "name_overlap": round(name_overlap, 4),
            "score_components": score_components,
            "strong_anchor": strong_anchor,
        }

    def _recall_stage1_build_evidence_profile(
        self,
        candidates: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate evidence already computed for the filtered candidates."""
        exact_topic_anchors: set[str] = set()
        exact_name_anchors: set[str] = set()
        exact_topic_count = 0
        exact_name_count = 0
        entity_mapping_candidate_count = 0
        max_bm25_score = 0.0
        bm25_support_candidate_count = 0
        max_candidate_score = 0.0
        strong_anchor_candidate_count = 0

        for candidate in candidates:
            details = candidate.get("_recall_fast_match_details")
            if not isinstance(details, dict):
                continue
            candidate_score = self._clamp_float(
                details.get("rank_score"),
                0.0,
                1.0,
                0.0,
            )
            max_candidate_score = max(max_candidate_score, candidate_score)
            if details.get("strong_anchor"):
                strong_anchor_candidate_count += 1
            topics = [str(value) for value in details.get("exact_topics") or []]
            names = [str(value) for value in details.get("exact_names") or []]
            exact_topic_count += len(topics)
            exact_name_count += len(names)
            if details.get("entity_mapping_hit"):
                entity_mapping_candidate_count += 1
            exact_topic_anchors.update(
                self._recall_stage1_normalize_match_text(value)
                for value in topics
            )
            exact_name_anchors.update(
                self._recall_stage1_normalize_match_text(value)
                for value in names
            )
            bm25_score = self._clamp_float(
                details.get("bm25_score"),
                0.0,
                1.0,
                0.0,
            )
            candidate_sources = {
                str(value)
                for value in details.get("candidate_sources") or []
            }
            if bm25_score and "lexical" in candidate_sources:
                max_bm25_score = max(max_bm25_score, bm25_score)
                bm25_support_candidate_count += 1

        # Candidate-level evidence already includes exact topic/name, entity
        # mapping, and BM25 components. Reuse the strongest candidate score
        # and BM25 score instead of rescanning identity_text here.
        lexical_support_score = max_bm25_score
        score = max(max_candidate_score, lexical_support_score)
        score = round(min(1.0, score), 4)

        strong_anchor = bool(
            exact_topic_anchors
            or exact_name_anchors
        )
        bm25_lexical_support = bool(
            bm25_support_candidate_count
            and max_bm25_score >= 0.75
        )
        trusted = bool(candidates) and (
            strong_anchor
            or (
                score >= self._recall_fast_candidate_score_threshold
                and bm25_lexical_support
            )
        )
        return {
            "score": score,
            "best_candidate_score": round(max_candidate_score, 4),
            "lexical_support_score": round(lexical_support_score, 4),
            "trusted": trusted,
            "threshold": self._recall_fast_candidate_score_threshold,
            "candidate_count": len(candidates),
            "exact_topic_count": exact_topic_count,
            "unique_exact_topic_count": len(exact_topic_anchors),
            "exact_name_count": exact_name_count,
            "unique_exact_name_count": len(exact_name_anchors),
            "entity_mapping_candidate_count": entity_mapping_candidate_count,
            "strong_anchor_candidate_count": strong_anchor_candidate_count,
            "max_bm25_score": round(max_bm25_score, 4),
            "bm25_support_candidate_count": bm25_support_candidate_count,
            "bm25_lexical_support": bm25_lexical_support,
            "strong_anchor": strong_anchor,
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

    def _recall_stage1_is_recent_active_topic(
        self,
        candidate: Dict[str, Any],
        *,
        reference_time: str,
        recent_fact_ids: Optional[set[int]] = None,
    ) -> bool:
        raw = candidate.get("_hydrated") if isinstance(candidate.get("_hydrated"), dict) else {}
        metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        state_scope = str(raw.get("state_scope") or candidate.get("state_scope") or "")
        status = str(raw.get("status") or metadata.get("status") or "active").lower()
        if state_scope != "topic_state" or status != "active":
            return False
        reference = self._recall_stage1_parse_datetime(reference_time)
        if reference is None:
            return False
        recent_window = max(
            1,
            int(self._memory_cfg.get("recall_fast_recent_topic_seconds", 300) or 300),
        )
        evidence_fact_ids = {
            int(value)
            for value in raw.get("evidence_fact_ids") or []
            if str(value).strip().isdigit()
        }
        if recent_fact_ids and evidence_fact_ids & recent_fact_ids:
            return True
        supporting_facts = candidate.get("_supporting_facts") or []
        if supporting_facts:
            return any(
                self._recall_stage1_is_recent_fact(
                    {
                        "_hydrated": fact,
                        "index_level": "fact",
                    },
                    reference_time=reference_time,
                )
                for fact in supporting_facts
            )
        updated_at = self._recall_stage1_parse_datetime(raw.get("updated_at"))
        if updated_at is None:
            return False
        age_seconds = (reference - updated_at).total_seconds()
        return 0 <= age_seconds <= recent_window

    def _recall_stage1_is_recent_fact(
        self,
        candidate: Dict[str, Any],
        *,
        reference_time: str,
    ) -> bool:
        raw = candidate.get("_hydrated") if isinstance(candidate.get("_hydrated"), dict) else {}
        fact_time = self._recall_stage1_parse_datetime(
            raw.get("dialogue_time_key") or candidate.get("time_start")
        )
        reference = self._recall_stage1_parse_datetime(reference_time)
        if fact_time is None or reference is None:
            return False
        recent_window = max(
            1,
            int(self._memory_cfg.get("recall_fast_recent_topic_seconds", 300) or 300),
        )
        age_seconds = (reference - fact_time).total_seconds()
        return 0 <= age_seconds <= recent_window

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
        min_importance = self._clamp_float(
            self._memory_cfg.get("recall_fast_actionable_min_importance"),
            0.0,
            1.0,
            0.75,
        )
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

    def _expand_recall_episode_facts_from_candidates(
        self,
        *,
        candidates: Sequence[Dict[str, Any]],
        candidate_source_prefix: str,
        source_types: Optional[Sequence[str]],
        temporal_bounds: RecallTimeBounds,
        temporal_mode: str,
        limit: int,
        database: Optional[SessionDB] = None,
    ) -> List[Dict[str, Any]]:
        """Add bounded facts from episodes represented by Stage 1 candidates."""
        db = database or self._db
        episode_ids: List[int] = []
        seed_targets: Dict[int, List[str]] = {}
        evidence_fact_ids: List[int] = []
        for candidate in candidates or []:
            raw = candidate.get("_hydrated") if isinstance(candidate.get("_hydrated"), dict) else {}
            raw_episode_ids = [raw.get("episode_id")]
            raw_episode_ids.extend(
                fact.get("episode_id")
                for fact in candidate.get("_supporting_facts") or []
                if isinstance(fact, dict)
            )
            for value in raw.get("evidence_fact_ids") or []:
                try:
                    fact_id = int(value)
                except (TypeError, ValueError):
                    continue
                if fact_id not in evidence_fact_ids:
                    evidence_fact_ids.append(fact_id)
            for value in raw_episode_ids:
                if not str(value or "").strip().isdigit():
                    continue
                episode_id = int(value)
                if episode_id not in episode_ids:
                    episode_ids.append(episode_id)
                seed_targets.setdefault(episode_id, []).append(
                    f"{candidate.get('target_table')}#{candidate.get('target_id')}"
                )
                if len(episode_ids) >= 8:
                    break
            if len(episode_ids) >= 8:
                break
        if evidence_fact_ids:
            for fact in db.memory_facts_by_ids(evidence_fact_ids):
                episode_id = fact.get("episode_id")
                if not str(episode_id or "").strip().isdigit():
                    continue
                episode_id = int(episode_id)
                if episode_id not in episode_ids:
                    episode_ids.append(episode_id)
                seed_targets.setdefault(episode_id, [])
                for candidate in candidates or []:
                    target = f"{candidate.get('target_table')}#{candidate.get('target_id')}"
                    if target not in seed_targets[episode_id]:
                        seed_targets[episode_id].append(target)
                if len(episode_ids) >= 8:
                    break
        if not episode_ids:
            return []

        allowed_sources = set(source_types or [])
        facts = db.memory_facts_by_episode_ids(
            episode_ids,
            limit=max(1, int(limit or 24)),
        )
        expanded: List[Dict[str, Any]] = []
        for fact in facts:
            if allowed_sources and fact.get("source_type") not in allowed_sources:
                continue
            candidate = self._make_recall_fact_candidate(
                fact,
                candidate_source=(
                    f"{str(candidate_source_prefix).strip()}_episode_expansion"
                ),
                temporal_mode=temporal_mode,
            )
            if not candidate:
                continue
            if not self._fact_matches_time_bounds(
                fact,
                temporal_mode=temporal_mode,
                temporal_bounds=temporal_bounds,
            ):
                continue
            episode_id = fact.get("episode_id")
            candidate["_stage2_episode_seed_targets"] = list(
                dict.fromkeys(seed_targets.get(int(episode_id), []))
            ) if str(episode_id or "").strip().isdigit() else []
            expanded.append(candidate)
        return expanded

    def _expand_recall_evidence_facts_from_candidates(
        self,
        *,
        candidates: Sequence[Dict[str, Any]],
        candidate_source_prefix: str,
        source_types: Optional[Sequence[str]],
        temporal_bounds: RecallTimeBounds,
        temporal_mode: str,
        limit: int,
        database: Optional[SessionDB] = None,
    ) -> List[Dict[str, Any]]:
        """Add facts directly cited by state and actionable-item candidates."""
        db = database or self._db
        evidence_fact_ids: List[int] = []
        for candidate in candidates or []:
            if str(candidate.get("index_level") or "") not in {
                "state",
                "actionable_item",
            }:
                continue
            raw = candidate.get("_hydrated")
            raw = raw if isinstance(raw, dict) else {}
            raw_fact_ids = list(raw.get("evidence_fact_ids") or [])
            raw_fact_ids.extend(
                fact.get("id")
                for fact in candidate.get("_supporting_facts") or []
                if isinstance(fact, dict)
            )
            for value in raw_fact_ids:
                try:
                    fact_id = int(value)
                except (TypeError, ValueError):
                    continue
                evidence_fact_ids.append(fact_id)

        if not evidence_fact_ids:
            return []
        allowed_sources = set(source_types or [])
        facts = db.memory_facts_by_ids(evidence_fact_ids[: max(1, int(limit or 24))])
        expanded: List[Dict[str, Any]] = []
        for fact in facts:
            if allowed_sources and fact.get("source_type") not in allowed_sources:
                continue
            candidate = self._make_recall_fact_candidate(
                fact,
                candidate_source=(
                    f"{str(candidate_source_prefix).strip()}_evidence_expansion"
                ),
                temporal_mode=temporal_mode,
            )
            if not candidate:
                continue
            if not self._fact_matches_time_bounds(
                fact,
                temporal_mode=temporal_mode,
                temporal_bounds=temporal_bounds,
            ):
                continue
            expanded.append(candidate)
        return expanded

    def _merge_and_expand_recall_stage2_candidates(
        self,
        *,
        stage1_raw_candidates: Sequence[Dict[str, Any]],
        stage2_lexical_candidates: Sequence[Dict[str, Any]],
        stage2_entity_candidates: Sequence[Dict[str, Any]],
        source_types: Optional[Sequence[str]],
        temporal_bounds: RecallTimeBounds,
        temporal_mode: str,
        top_k: int,
        database: Optional[SessionDB] = None,
    ) -> Dict[str, Any]:
        """Merge Stage 1/2 seeds and expand episode and evidence relations."""
        seed_candidates = [
            *list(stage2_lexical_candidates or []),
            *list(stage2_entity_candidates or []),
        ]
        expansion_limit = max(12, min(64, max(1, int(top_k or 1)) * 6))
        episode_candidates = self._expand_recall_episode_facts_from_candidates(
            candidates=seed_candidates,
            candidate_source_prefix="stage2",
            source_types=source_types,
            temporal_bounds=temporal_bounds,
            temporal_mode=temporal_mode,
            limit=expansion_limit,
            database=database,
        )
        evidence_candidates = self._expand_recall_evidence_facts_from_candidates(
            candidates=seed_candidates,
            candidate_source_prefix="stage2",
            source_types=source_types,
            temporal_bounds=temporal_bounds,
            temporal_mode=temporal_mode,
            limit=expansion_limit,
            database=database,
        )

        merged_candidates: Dict[Tuple[str, int], Dict[str, Any]] = {}

        def merge_candidate_group(
            candidates: Sequence[Dict[str, Any]],
            provenance: str,
            *,
            seed: bool = False,
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
                provenance_values = list(candidate.get("_stage2_provenance") or [])
                if provenance not in provenance_values:
                    provenance_values.append(provenance)
                candidate["_stage2_provenance"] = provenance_values
                if seed:
                    candidate["_stage1_seed_score"] = float(
                        candidate.get("_recall_score") or 0.0
                    )
                existing = merged_candidates.get(target)
                if existing is None:
                    merged_candidates[target] = candidate
                    continue
                existing_provenance = list(existing.get("_stage2_provenance") or [])
                for value in provenance_values:
                    if value not in existing_provenance:
                        existing_provenance.append(value)
                existing["_stage2_provenance"] = existing_provenance
                if seed and not existing.get("_stage1_seed_score"):
                    existing["_stage1_seed_score"] = float(
                        candidate.get("_stage1_seed_score") or 0.0
                    )

        merge_candidate_group(stage1_raw_candidates, "stage1_seed", seed=True)
        merge_candidate_group(stage2_lexical_candidates, "stage2_lexical_supplement")
        merge_candidate_group(stage2_entity_candidates, "stage2_entity_mapping_supplement")
        merge_candidate_group(
            evidence_candidates,
            "evidence_expansion",
        )
        merge_candidate_group(
            episode_candidates,
            "episode_expansion",
        )

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
            "stage1_seed_candidates": list(stage1_raw_candidates or []),
            "stage2_lexical_candidates": list(stage2_lexical_candidates or []),
            "stage2_entity_candidates": list(stage2_entity_candidates or []),
            "episode_candidates": episode_candidates,
            "evidence_candidates": evidence_candidates,
            "merged_candidates": list(merged_candidates.values()),
            "merged_count": len(merged_candidates),
            "seed_count": len(seed_candidates),
            "episode_expansion_count": len(episode_candidates),
            "evidence_expansion_count": len(evidence_candidates),
        }

    def _process_recall_stage2(
        self,
        *,
        query: str,
        analysis_query: Optional[str] = None,
        top_k: int,
        budget: str,
        temporal_bounds: RecallTimeBounds,
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
        self._log_info("memory_recall_stage2", "start", {
            "query": self._format_log_text(query, limit=500),
            "top_k": top_k,
            "budget": budget,
            "time_start": (temporal_bounds or (None, None))[0],
            "time_end": (temporal_bounds or (None, None))[1],
            "temporal_mode": temporal_mode,
            "prompt_language": prompt_language,
            "memory_source_override": list(memory_source_override or []),
        })
        recall_plan = self._analyze_recall_query(
            analysis_query or query,
            prompt_language=prompt_language,
        )
        temporal_mode = self._normalize_recall_temporal_mode(
            recall_plan.get("temporal_mode") or temporal_mode
        )
        forced_source_types = self._normalize_source_override(memory_source_override)
        preferred_source_types = forced_source_types or self._normalize_source_override(
            recall_plan.get("source_types") or []
        )
        layer_preference = recall_plan.get("layer_preference")
        if layer_preference is None:
            # Keep old local/test LLM adapters compatible while the prompt
            # contract migrates from index_levels to layer_preference.
            layer_preference = recall_plan.get("index_levels") or []
        preferred_layer_preferences = self._normalize_recall_layer_preference(
            layer_preference or []
        )
        llm_keywords = self._normalize_string_list(
            recall_plan.get("keywords"),
            limit=12,
        )
        llm_entities = self._normalize_entity_names(
            recall_plan.get("entities"),
            limit=12,
        )
        # Stage 1 already performed lexical retrieval with the raw user query.
        # Stage 2 lexical retrieval is only for LLM-derived concepts; the raw
        # query is added back later only for ranking the merged candidates.
        supplement_terms = self._build_recall_search_terms(
            "",
            keywords=llm_keywords,
            entities=llm_entities,
        )
        ranking_terms = list(dict.fromkeys([
            *self._lexical_search_terms_for_text(
                query,
                limit=32,
                preserve_phrase=False,
            ),
            *supplement_terms,
        ]))
        retrieval_text = (
            recall_plan.get("query_rewrite")
            or ""
        )
        query_identity_text = self._format_recall_query_identity_text(
            query,
            retrieval_text=retrieval_text,
            keywords=llm_keywords,
            entities=llm_entities,
        )
        query_identity_embedding = self._generate_embedding_vector(query_identity_text)
        final_candidate_limits, supplement_candidate_limits = (
            self._recall_stage2_candidate_limits(
                query=query,
                top_k=top_k,
                budget=budget,
                preferred_layer_preferences=preferred_layer_preferences,
            )
        )
        supplement_entity_mapping_limits, supplement_lexical_limits = (
            self._split_recall_stage2_source_limits(supplement_candidate_limits)
        )
        self._log_info("memory_recall_stage2", "query_analyzed", {
            "recall_plan": recall_plan,
            "forced_source_types": forced_source_types or [],
            "preferred_source_types": preferred_source_types or [],
            "preferred_layer_preferences": preferred_layer_preferences or [],
            "keywords": llm_keywords,
            "entities": llm_entities,
            "supplement_terms": supplement_terms,
            "ranking_terms": ranking_terms,
            "retrieval_text": self._format_log_text(retrieval_text, limit=500),
            "identity_text": self._format_log_text(query_identity_text, limit=500),
            "query_embedding_available": query_identity_embedding is not None,
            "final_candidate_limits": final_candidate_limits,
            "supplement_candidate_limits": supplement_candidate_limits,
            "supplement_entity_mapping_limits": supplement_entity_mapping_limits,
            "supplement_lexical_limits": supplement_lexical_limits,
            "budget": budget,
            "temporal_mode": temporal_mode,
        })
        if supplement_terms:
            lexical_fact_candidates, lexical_state_candidates, lexical_actionable_candidates = (
                self._retrieve_recall_raw_candidates_lexical_search(
                    terms=supplement_terms,
                    candidate_source_prefix="stage2",
                    source_types=forced_source_types,
                    temporal_bounds=temporal_bounds,
                    temporal_mode=temporal_mode,
                    candidate_limits=supplement_lexical_limits,
                    database=database,
                )
            )
        else:
            lexical_fact_candidates, lexical_state_candidates, lexical_actionable_candidates = (
                [], [], []
            )
        stage1_raw_candidates = list(
            (stage1_report or {}).get("raw_candidates") or []
        )
        stage1_entity_names = {
            self._recall_stage1_normalize_match_text(value)
            for candidate in stage1_raw_candidates
            for value in candidate.get("_recall_entity_names") or []
            if self._recall_stage1_normalize_match_text(value)
        }
        new_llm_entities = [
            entity for entity in llm_entities
            if self._recall_stage1_normalize_match_text(entity) not in stage1_entity_names
        ]
        stage2_entity_fact_candidates, stage2_entity_state_candidates, stage2_entity_actionable_candidates = (
            self._retrieve_recall_entity_mapping_candidates(
                query=" ".join(new_llm_entities),
                candidate_source_prefix="stage2",
                source_types=forced_source_types,
                temporal_bounds=temporal_bounds,
                temporal_mode=temporal_mode,
                candidate_limits=supplement_entity_mapping_limits,
                database=database,
            ) if new_llm_entities else ([], [], [])
        )
        stage2_candidate_report = self._merge_and_expand_recall_stage2_candidates(
            stage1_raw_candidates=stage1_raw_candidates,
            stage2_lexical_candidates=[
                *lexical_fact_candidates,
                *lexical_state_candidates,
                *lexical_actionable_candidates,
            ],
            stage2_entity_candidates=[
                *stage2_entity_fact_candidates,
                *stage2_entity_state_candidates,
                *stage2_entity_actionable_candidates,
            ],
            source_types=forced_source_types,
            temporal_bounds=temporal_bounds,
            temporal_mode=temporal_mode,
            top_k=top_k,
            database=database,
        )
        merged_by_level = stage2_candidate_report["by_level"]
        fact_candidates = merged_by_level["fact"]
        state_candidates = merged_by_level["state"]
        actionable_candidates = merged_by_level["actionable_item"]
        stage2_candidates_payload = {
            "facts": self._recall_log_candidate_items(fact_candidates),
            "states": self._recall_log_candidate_items(state_candidates),
            "actionable_items": self._recall_log_candidate_items(actionable_candidates),
            "stage1_seed_count": len(stage1_raw_candidates),
            "merged_count": stage2_candidate_report["merged_count"],
            "episode_expansion_count": stage2_candidate_report["episode_expansion_count"],
            "evidence_expansion_count": stage2_candidate_report["evidence_expansion_count"],
        }
        if self._recall_detailed_logging:
            stage2_candidates_payload["candidate_diagnostics"] = {
                "stage1_seed_candidates": self._recall_detailed_candidate_items(
                    stage2_candidate_report.get("stage1_seed_candidates") or [],
                    stage="stage2_stage1_seed",
                ),
                "stage2_lexical_seeds": self._recall_detailed_candidate_items(
                    stage2_candidate_report.get("stage2_lexical_candidates") or [],
                    stage="stage2_lexical_seed",
                ),
                "stage2_entity_mapping_seeds": self._recall_detailed_candidate_items(
                    stage2_candidate_report.get("stage2_entity_candidates") or [],
                    stage="stage2_entity_mapping_seed",
                ),
                "evidence_expansions": self._recall_detailed_candidate_items(
                    stage2_candidate_report.get("evidence_candidates") or [],
                    stage="stage2_evidence_expansion",
                ),
                "episode_expansions": self._recall_detailed_candidate_items(
                    stage2_candidate_report.get("episode_candidates") or [],
                    stage="stage2_episode_expansion",
                ),
                "merged_candidates": self._recall_detailed_candidate_items(
                    stage2_candidate_report.get("merged_candidates") or [],
                    stage="stage2_merged",
                ),
            }
        self._log_info("memory_recall_stage2", "candidates_found", stage2_candidates_payload)
        ranked = self._rank_recall_raw_candidates(
            facts=fact_candidates,
            states=state_candidates,
            actionable_items=actionable_candidates,
            query=query,
            terms=ranking_terms,
            query_embedding=query_identity_embedding,
            final_candidate_limits=final_candidate_limits,
        )
        
        ranked_payload = {
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
        }
        if self._recall_detailed_logging:
            ranked_targets = {
                (
                    str(item.get("target_table") or ""),
                    int(item.get("target_id")),
                )
                for item in ranked
                if str(item.get("target_id") or "").isdigit()
            }
            ranked_payload["candidate_diagnostics"] = {
                "candidate_pool": self._recall_detailed_candidate_items(
                    stage2_candidate_report.get("merged_candidates") or [],
                    selected_targets=ranked_targets,
                    stage="stage2_rank_pool",
                ),
                "ranked_candidates": self._recall_detailed_candidate_items(
                    ranked,
                    selected_targets=ranked_targets,
                    stage="stage2_ranked",
                ),
            }
        self._log_info("memory_recall_stage2", "ranked", ranked_payload)
        memory_text = self._build_memory_retrieved_format_text(
            entries=ranked,
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
        table: str,
        level: str,
        row: Dict[str, Any],
        candidate_source: str,
        supporting_facts: Optional[Sequence[Dict[str, Any]]] = None,
        temporal_bounds: RecallTimeBounds = None,
        temporal_mode: str = "dialogue_time",
    ) -> Optional[Dict[str, Any]]:
        """Convert a memory row into the shared recall candidate shape."""
        try:
            target_id = int(row.get("id"))
        except (TypeError, ValueError):
            return None
        source_type = row.get("source_type")
        support_facts = [dict(fact) for fact in supporting_facts or []]
        if table == "memory_facts":
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
        elif table == "memory_states":
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
        elif table == "memory_actionable_items":
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

        if table == "memory_facts":
            time_start, time_end = temporal_bounds or (None, None)
            if time_start and time_end_value and time_end_value < str(time_start):
                return None
            if time_end and time_value and time_value > str(time_end):
                return None

        hydrated = dict(row)
        hydrated.pop("embedding", None)
        hydrated.pop("identity_text_embedding", None)
        hydrated.pop("canonical_name_embedding", None)
        metadata = dict(row.get("metadata") or {})
        metadata["_matched_via"] = [candidate_source]
        candidate = {
            "source_type": source_type,
            "target_table": table,
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
        if table != "memory_facts":
            candidate["canonical_topics"] = topics
        return candidate

    def _retrieve_recall_entity_mapping_candidates(
        self,
        *,
        query: str,
        candidate_source_prefix: str,
        source_types: Optional[Sequence[str]],
        temporal_bounds: RecallTimeBounds,
        temporal_mode: str,
        candidate_limits: Dict[str, int],
        database: Optional[SessionDB] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Retrieve direct memory candidates through query-matched entities."""
        db = database or self._db
        entity_lookup_aliases = self._recall_entity_lookup_aliases(query)
        entity_lookup_text = " ".join(
            [str(query or "").strip(), *entity_lookup_aliases]
        ).strip()
        entity_rows = db.find_entity_nodes_in_text(entity_lookup_text, limit=12)
        entity_ids = [int(row["id"]) for row in entity_rows if row.get("id") is not None]
        matched_entity_names = [
            str(row.get("name") or "")
            for row in entity_rows
            if str(row.get("name") or "").strip()
        ]
        if not entity_ids:
            return [], [], []
        mappings = db.memory_entity_mappings_by_entity_ids(entity_ids)
        if not mappings:
            return [], [], []
        allowed_sources = set(source_types or [])
        ids_by_level: Dict[str, List[int]] = {
            "fact": [],
            "state": [],
            "actionable_item": [],
        }
        raw_limit_keys = {
            "fact": "facts",
            "state": "states",
            "actionable_item": "actionable_items",
        }
        for mapping in mappings:
            for field, level in (
                ("fact_id", "fact"),
                ("state_id", "state"),
                ("actionable_item_id", "actionable_item"),
            ):
                level_limit = max(
                    0,
                    int(candidate_limits.get(raw_limit_keys[level], 0) or 0),
                )
                if len(ids_by_level[level]) >= level_limit:
                    continue
                for value in mapping.get(field) or []:
                    try:
                        target_id = int(value)
                    except (TypeError, ValueError):
                        continue
                    if len(ids_by_level[level]) >= level_limit:
                        break
                    if target_id not in ids_by_level[level]:
                        ids_by_level[level].append(target_id)

        facts = db.memory_facts_by_ids(ids_by_level["fact"])
        states = db.memory_states_by_ids(ids_by_level["state"])
        actionable_items = db.memory_actionable_items_by_ids(ids_by_level["actionable_item"])
        candidates_by_level: Dict[str, List[Dict[str, Any]]] = {
            "fact": [],
            "state": [],
            "actionable_item": [],
        }
        rows_by_level = {
            "fact": facts,
            "state": states,
            "actionable_item": actionable_items,
        }
        table_by_level = {
            "fact": "memory_facts",
            "state": "memory_states",
            "actionable_item": "memory_actionable_items",
        }
        for level, rows in rows_by_level.items():
            for row in rows:
                if allowed_sources and row.get("source_type") not in allowed_sources:
                    continue
                candidate = self._make_recall_memory_candidate(
                    table=table_by_level[level],
                    level=level,
                    row=row,
                    candidate_source=(
                        f"{str(candidate_source_prefix).strip()}_entity_mapping"
                    ),
                    temporal_bounds=temporal_bounds,
                    temporal_mode=temporal_mode,
                )
                if candidate:
                    candidate["_recall_entity_names"] = matched_entity_names
                    candidates_by_level[level].append(candidate)
        return (
            candidates_by_level["fact"],
            candidates_by_level["state"],
            candidates_by_level["actionable_item"],
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
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build separate raw candidate groups before type-specific ranking.

        This intentionally bypasses `memory_index_entries` for the default
        path. Each candidate still points to its source row; relationship
        expansion is handled by the Stage 1/Stage 2 merge layer.
        """
        db = database or self._db
        time_start, time_end = temporal_bounds or (None, None)
        table_specs = (
            ("memory_facts", "fact", db.search_memory_facts),
            ("memory_states", "state", db.search_memory_states),
            ("memory_actionable_items", "actionable_item", db.search_memory_actionable_items),
        )
        rows_by_table: Dict[str, List[Dict[str, Any]]] = {}
        raw_limit_keys = {
            "memory_facts": "facts",
            "memory_states": "states",
            "memory_actionable_items": "actionable_items",
        }
        for table, _level, loader in table_specs:
            loader_kwargs = {
                "terms": terms,
                "source_types": source_types,
                # States and actionable items do not have an event-time
                # column, so their own update/due time is used below.
                "time_start": time_start if table == "memory_facts" else None,
                "time_end": time_end if table == "memory_facts" else None,
                "limit": max(1, int(candidate_limits.get(raw_limit_keys[table], 1) or 1)),
            }
            if table == "memory_facts":
                loader_kwargs["temporal_mode"] = temporal_mode
            rows_by_table[table] = loader(**loader_kwargs)

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
                source_type = row.get("source_type")
                if table == "memory_facts":
                    title = _compact_whitespace(row.get("summary") or "")[:120]
                    summary = _compact_whitespace(row.get("summary") or "")
                    fact_times = self._fact_time_values(row, temporal_mode)
                    time_value = fact_times[0] if fact_times else ""
                    time_end_value = fact_times[-1] if fact_times else ""
                    entities = row.get("entities") or []
                    topics = self._normalize_unique_labels([
                        row.get("fact_root_topic"),
                        row.get("fact_aspect_topic"),
                    ], limit=12)
                elif table == "memory_states":
                    title = _compact_whitespace(row.get("canonical_name") or "")
                    summary = _compact_whitespace(row.get("summary") or "")
                    time_value = self._normalize_event_time_text(row.get("updated_at"))
                    state_metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                    entities = [
                        row.get("entity_key"),
                        *(state_metadata.get("context_entities") or []),
                        *(state_metadata.get("entities") or []),
                    ]
                    entities = self._normalize_entity_names(entities, limit=18)
                    topics = self._normalize_unique_labels([
                        row.get("canonical_name"),
                        *(state_metadata.get("canonical_topics") or []),
                        *(state_metadata.get("parent_topics") or []),
                        *(state_metadata.get("aspect_topic_names") or []),
                    ], limit=24)
                    state_keywords = self._normalize_string_list(
                        state_metadata.get("keywords"),
                        limit=24,
                    )
                else:
                    title = _compact_whitespace(row.get("canonical_name") or "")
                    summary = _compact_whitespace(row.get("summary") or "")
                    time_value = self._normalize_event_time_text(
                        row.get("due_at") or row.get("updated_at") or row.get("created_at")
                    )
                    entities = [row.get("owner")] if row.get("owner") else []
                    topics = [row.get("canonical_name")] if row.get("canonical_name") else []

                if table == "memory_facts" and not self._fact_matches_time_bounds(
                    row,
                    temporal_mode=temporal_mode,
                    temporal_bounds=temporal_bounds,
                ):
                    continue
                if table != "memory_facts":
                    time_end_value = time_value
                if table == "memory_facts":
                    if time_start and time_end_value and time_end_value < str(time_start):
                        continue
                    if time_end and time_value and time_value > str(time_end):
                        continue

                candidate_embedding = row.get("identity_text_embedding")
                hydrated = dict(row)
                hydrated.pop("embedding", None)
                hydrated.pop("identity_text_embedding", None)
                hydrated.pop("canonical_name_embedding", None)
                metadata = dict(row.get("metadata") or {})
                metadata["_matched_via"] = ["direct"]
                candidate = {
                    "source_type": source_type,
                    "target_table": table,
                    "target_id": target_id,
                    "index_level": level,
                    "memory_path": f"{source_type}/{level}",
                    "title": title,
                    "summary_for_retrieval": summary,
                    "identity_text": _compact_whitespace(row.get("identity_text") or ""),
                    "keywords": (
                        " ".join(state_keywords)
                        if table == "memory_states"
                        else row.get("keywords") or ""
                    ),
                    "entities": entities,
                    "participants": row.get("participants") or [],
                    "time_start": time_value,
                    "time_end": time_end_value,
                    "importance": row.get("importance") or 0.5,
                    "confidence": row.get("confidence") or 0.8,
                    "embedding": candidate_embedding,
                    "metadata": metadata,
                    "_hydrated": hydrated,
                    "_supporting_facts": [],
                    "_bm25_score": row.get("_bm25_score"),
                    "_recall_candidate_source": (
                        f"{str(candidate_source_prefix).strip()}_lexical"
                    ),
                }
                if table != "memory_facts":
                    candidate["canonical_topics"] = topics
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

    def _make_recall_fact_candidate(
        self,
        fact: Dict[str, Any],
        *,
        candidate_source: str,
        temporal_mode: str = "dialogue_time",
    ) -> Optional[Dict[str, Any]]:
        """Convert a raw/supporting fact into the shared display shape."""
        try:
            fact_id = int(fact.get("id"))
        except (TypeError, ValueError):
            return None
        source_type = fact.get("source_type")
        summary = _compact_whitespace(fact.get("summary") or "")
        fact_times = self._fact_time_values(fact, temporal_mode)
        time_value = fact_times[0] if fact_times else ""
        time_end_value = fact_times[-1] if fact_times else ""
        hydrated = dict(fact)
        hydrated.pop("identity_text_embedding", None)
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
            "identity_text": _compact_whitespace(fact.get("identity_text") or ""),
            "keywords": fact.get("keywords") or "",
            "entities": fact.get("entities") or [],
            "participants": fact.get("participants") or [],
            "time_start": time_value,
            "time_end": time_end_value,
            "importance": fact.get("importance") or 0.5,
            "confidence": fact.get("confidence") or 0.8,
            "embedding": fact.get("identity_text_embedding"),
            "metadata": metadata,
            "_hydrated": hydrated,
            "_supporting_facts": [],
            "_recall_candidate_source": candidate_source,
        }

    @staticmethod
    def _recall_candidate_provenance_profile(
        candidate: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Summarize retrieval provenance without changing candidate content."""
        provenance = {
            str(value)
            for value in candidate.get("_stage2_provenance") or []
            if str(value).strip()
        }
        candidate_source = str(
            candidate.get("_recall_candidate_source") or ""
        ).strip()
        source_names = MemoryNodeManager._recall_candidate_source_channels(
            candidate_source
        )
        stage1_seed = "stage1_seed" in provenance
        stage1_entity_mapping = bool(
            stage1_seed and "entity_mapping" in source_names
        )
        stage1_lexical = bool(stage1_seed and "lexical" in source_names)
        stage1_direct_index = bool(
            stage1_seed
            and candidate_source in {
                "stage1_entity_mapping",
                "stage1_lexical",
                "stage1_both",
            }
        )
        stage2_entity_mapping = "stage2_entity_mapping_supplement" in provenance
        stage2_lexical = "stage2_lexical_supplement" in provenance
        direct_index = bool(
            stage1_direct_index
            or stage2_entity_mapping
            or stage2_lexical
        )
        evidence_expansion = bool(
            "evidence_expansion" in provenance
            or candidate_source.endswith("_evidence_expansion")
        )
        episode_expansion = bool(
            "episode_expansion" in provenance
            or candidate_source.endswith("_episode_expansion")
        )
        fast_match = candidate.get("_recall_fast_match_details")
        fast_match = fast_match if isinstance(fast_match, dict) else {}
        strong_anchor = bool(fast_match.get("strong_anchor"))

        # These bonuses are deliberately small. Retrieval provenance should
        # break ties between semantically relevant candidates, not rescue a
        # candidate whose identity_text is unrelated to the query.
        provenance_bonus = 0.0
        if stage1_entity_mapping:
            provenance_bonus += 0.08
        if stage1_lexical:
            provenance_bonus += 0.04
        if stage2_entity_mapping:
            provenance_bonus += 0.05
        if stage2_lexical:
            provenance_bonus += 0.025
        if (stage1_entity_mapping or stage2_entity_mapping) and (
            stage1_lexical or stage2_lexical
        ):
            provenance_bonus += 0.04
        if not provenance and "entity_mapping" in source_names:
            provenance_bonus += 0.04
        elif not provenance and "lexical" in source_names:
            provenance_bonus += 0.02
        if strong_anchor:
            provenance_bonus += 0.04

        hop = 0 if direct_index else 1 if (
            evidence_expansion or episode_expansion
        ) else 0
        if direct_index:
            expansion_penalty = 0.0
        elif evidence_expansion:
            # Evidence facts are explicitly cited by a state/actionable item,
            # so they are weaker than direct hits but stronger than same-episode
            # expansion facts.
            expansion_penalty = min(0.08, 0.03 * max(1, hop))
        elif episode_expansion:
            expansion_penalty = min(0.18, 0.08 * max(1, hop))
        else:
            expansion_penalty = 0.0

        return {
            "candidate_source": candidate_source,
            "source_names": sorted(source_names),
            "stage2_provenance": sorted(provenance),
            "stage1_seed": stage1_seed,
            "stage1_direct_index": stage1_direct_index,
            "stage1_entity_mapping": stage1_entity_mapping,
            "stage1_lexical": stage1_lexical,
            "stage2_entity_mapping": stage2_entity_mapping,
            "stage2_lexical": stage2_lexical,
            "direct_index": direct_index,
            "evidence_expansion": evidence_expansion,
            "episode_expansion": episode_expansion,
            "hop": hop,
            "strong_anchor": strong_anchor,
            "provenance_bonus": round(min(0.22, provenance_bonus), 4),
            "expansion_penalty": round(expansion_penalty, 4),
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
    ) -> List[Dict[str, Any]]:
        """Score one memory type with calibrated relevance and provenance."""
        query_lower = str(query or "").lower()
        threshold = (
            None
            if min_embedding_similarity is None
            else self._clamp_float(min_embedding_similarity, 0.0, 1.0, 0.0)
        )
        scored: List[Tuple[float, str, int, Dict[str, Any]]] = []
        for position, entry in enumerate(candidates):
            raw = entry.get("_hydrated") if isinstance(entry.get("_hydrated"), dict) else {}
            metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}

            def extend_values(target: List[str], value: Any) -> None:
                if isinstance(value, (list, tuple, set)):
                    for item in value:
                        extend_values(target, item)
                    return
                clean = _compact_whitespace(value)
                if clean and clean not in target:
                    target.append(clean)

            topic_values: List[str] = []
            name_values: List[str] = []
            state_scope = str(raw.get("state_scope") or "")
            if memory_type == "fact":
                extend_values(topic_values, raw.get("fact_root_topic"))
                extend_values(topic_values, raw.get("fact_aspect_topic"))
            elif memory_type == "state":
                extend_values(name_values, raw.get("canonical_name"))
                if state_scope == "topic_state":
                    extend_values(topic_values, raw.get("canonical_name"))
                    extend_values(
                        topic_values,
                        metadata.get("aspect_topic_names"),
                    )
                    extend_values(
                        topic_values,
                        metadata.get("canonical_topics"),
                    )
                else:
                    extend_values(
                        name_values,
                        metadata.get("attribute_name_aliases"),
                    )
                    extend_values(
                        topic_values,
                        metadata.get("canonical_topics"),
                    )
            else:
                extend_values(name_values, raw.get("canonical_name"))
                extend_values(topic_values, metadata.get("canonical_topics"))
                extend_values(topic_values, raw.get("canonical_topics"))

            topic_overlap = (
                self._topic_name_overlap(
                    terms,
                    topic_values,
                    allow_substring=False,
                )
                if terms and topic_values
                else 0.0
            )
            name_overlap = (
                self._topic_name_overlap(
                    terms,
                    name_values,
                    allow_substring=False,
                )
                if terms and name_values
                else 0.0
            )
            structured_overlap = max(topic_overlap, name_overlap)
            similarity = max(0.0, self._cal_embedding_similarity(
                query_embedding,
                entry.get("embedding"),
            ))
            if (
                threshold is not None
                and query_embedding is not None
                and similarity < threshold
            ):
                entry["_recall_drop_reason"] = "embedding_below_threshold"
                entry["_recall_embedding_threshold"] = threshold
                entry["_recall_embedding_similarity"] = round(float(similarity), 4)
                continue
            provenance_profile = self._recall_candidate_provenance_profile(entry)
            bm25_score = self._clamp_float(
                entry.get("_recall_bm25_score"),
                0.0,
                1.0,
                0.0,
            )
            has_bm25_signal = bool(
                entry.get("_recall_bm25_score") is not None
                or entry.get("_bm25_score") is not None
            )

            # Use a weighted average over available signals. Entity-mapping
            # candidates often have no BM25 result, and candidates from an
            # embedding-only path may have no lexical terms; neither should be
            # penalized merely because a signal is unavailable.
            relevance_parts: List[Tuple[float, float]] = []
            if query_embedding is not None and entry.get("embedding") is not None:
                relevance_parts.append((0.42, similarity))
            if has_bm25_signal:
                relevance_parts.append((0.28, bm25_score))
            if terms and (topic_values or name_values):
                relevance_parts.append((0.25, structured_overlap))
            relevance_weight = sum(weight for weight, _value in relevance_parts)
            relevance_score = (
                sum(weight * value for weight, value in relevance_parts)
                / relevance_weight
                if relevance_weight > 1e-6
                else 0.0
            )
            type_bonus = 0.0
            if memory_type == "fact":
                if any(marker in query_lower for marker in ("when", "before", "after", "什么时候", "之前", "之后")):
                    type_bonus += 0.03
            elif memory_type == "state":
                if structured_overlap >= 0.90:
                    type_bonus += 0.05
            else:
                if any(marker in query_lower for marker in (
                    "todo", "task", "remind", "decision", "commit", "待办", "任务", "提醒", "决定", "承诺",
                )):
                    type_bonus += 0.05
                if str(raw.get("status") or "").lower() in {"open", "in_progress", "blocked"}:
                    type_bonus += 0.02
            score = self._clamp_float(
                relevance_score
                + type_bonus
                + float(provenance_profile["provenance_bonus"])
                - float(provenance_profile["expansion_penalty"]),
                0.0,
                1.0,
                0.0,
            )
            item = dict(entry)
            item.pop("embedding", None)
            if query_embedding is not None:
                item["embedding_similarity"] = round(float(similarity), 4)
            item["_recall_score_components"] = {
                "embedding_similarity": round(float(similarity), 4),
                "bm25_identity_text": round(float(bm25_score), 4),
                "topic_overlap": round(float(topic_overlap), 4),
                "name_overlap": round(float(name_overlap), 4),
                "structured_overlap": round(float(structured_overlap), 4),
                "relevance_score": round(float(relevance_score), 4),
                "type_bonus": round(float(type_bonus), 4),
                "provenance_bonus": provenance_profile["provenance_bonus"],
                "expansion_penalty": provenance_profile["expansion_penalty"],
                "final_type_score": round(float(score), 4),
            }
            item["_recall_provenance"] = provenance_profile
            item["_stage2_hop_penalty"] = provenance_profile["expansion_penalty"]
            scored.append((score, str(entry.get("time_start") or ""), -position, item))

        if not scored:
            return []
        ranked: List[Dict[str, Any]] = []
        for rank, (score, time_value, position, item) in enumerate(
            sorted(scored, key=lambda row: (row[0], row[1], row[2]), reverse=True),
            1,
        ):
            item["_recall_type"] = memory_type
            item["_recall_type_score"] = round(float(score), 4)
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
    ) -> List[Dict[str, Any]]:
        return self._rank_recall_candidates_by_type(
            candidates,
            memory_type="fact",
            terms=terms,
            query=query,
            query_embedding=query_embedding,
            min_embedding_similarity=min_embedding_similarity,
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
        final_candidate_limits: Optional[Dict[str, int]] = None,
        fact_min_embedding_similarity: Optional[float] = None,
        state_min_embedding_similarity: Optional[float] = None,
        actionable_item_min_embedding_similarity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Rank each raw layer independently, then merge the ranked candidates."""

        final_limits = final_candidate_limits or {
            "facts": len(facts),
            "states": len(states),
            "actionable_items": len(actionable_items),
        }
        ranked_facts = self._rank_recall_fact_candidates(
            facts,
            terms=terms,
            query=query,
            query_embedding=query_embedding,
            min_embedding_similarity=(
                self._recall_fact_min_embedding_similarity
                if fact_min_embedding_similarity is None
                else fact_min_embedding_similarity
            ),
        )
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

        selected_candidates, _selected_by_layer = (
            self._recall_get_selected_candidates_by_layer(
                ranked_fact_candidates=ranked_facts,
                ranked_state_candidates=ranked_states,
                ranked_actionable_items=ranked_actionable,
                layer_limits=final_limits,
            )
        )
        return selected_candidates
    
    def _analyze_recall_query(
        self,
        query: str,
        *,
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
        query: str,
        top_k: int,
        budget: str,
        preferred_layer_preferences: Optional[Sequence[str]],
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Build Stage 2 final and supplement budgets consistently with Stage 1."""
        k = max(1, int(top_k or 1))

        base_raw_limit = max(8, min(32, k * 3))
        supplement_limits = {
            "facts": base_raw_limit,
            "states": base_raw_limit,
            "actionable_items": base_raw_limit,
        }

        supplement_limits = {
            key: min(48, value)
            for key, value in supplement_limits.items()
        }

        final_limits = {
            "facts": k,
            "states": k,
            "actionable_items": k,
        }

        lower = str(query or "").lower()
        if self._needs_broad_evidence(query):
            final_limits["facts"] = max(final_limits["facts"], int(math.ceil(k * 0.85)))
        preferred = set(preferred_layer_preferences or [])
        aliases = {"actionable": "actionable_items", "action": "actionable_items"}
        preferred = {aliases.get(level, level) for level in preferred}
        if preferred:
            for key in final_limits:
                if key.rstrip("s") in preferred or key in preferred:
                    final_limits[key] += max(1, int(math.ceil(k * 0.15)))
        if any(marker in lower for marker in ("todo", "task", "remind", "decision", "commit", "待办", "任务", "提醒", "决定", "承诺")):
            final_limits["actionable_items"] = max(final_limits["actionable_items"], int(math.ceil(k * 0.55)))
        if any(marker in lower for marker in ("prefer", "usually", "habit", "偏好", "通常", "习惯", "长期", "状态")):
            final_limits["states"] = max(final_limits["states"], int(math.ceil(k * 0.5)))

        return final_limits, supplement_limits

    @staticmethod
    def _split_recall_stage2_source_limits(
        supplement_candidate_limits: Dict[str, int],
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Split Stage 2 supplement budgets between mapping and lexical search."""
        entity_mapping_limits: Dict[str, int] = {}
        lexical_limits: Dict[str, int] = {}
        for key, supplement_limit in (supplement_candidate_limits or {}).items():
            normalized_limit = max(0, int(supplement_limit or 0))
            entity_limit = normalized_limit // 2
            entity_mapping_limits[key] = entity_limit
            lexical_limits[key] = normalized_limit - entity_limit
        return entity_mapping_limits, lexical_limits
    
    def _needs_broad_evidence(self, query: str) -> bool:
        lower = str(query or "").lower()
        return any(
            marker in lower
            for marker in (
                "how many", "count", "total", "which", "first", "before",
                "after", "between", "compare", "all", "what were",
            )
        )
