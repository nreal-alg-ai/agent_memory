"""Runtime adapter between application input events and memory storage.

``MemoryNodeManager`` owns episode persistence, reflection, and recall.  This
adapter owns the short-lived interaction buffer and converts frontend-shaped
turns or transcript segments into the manager's raw episode segments.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .embedding_client import EmbeddingClient
from .memory_database import SessionDB

from .memory_manager import (
    MemoryOperationReporter,
    MemoryNodeManager,
    _compact_whitespace,
    _now_text,
    _to_timestamp_text,
)
from .memory_segmentation import (
    OnlineSemanticSegmenter,
    build_online_segmentation_config,
    convert_interaction_turn_to_online_exchange
)

INTERACTION_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_.$'-]+|[^\s]")


class MemoryRuntime:
    """Normalize application input and batch it before episode storage."""

    def __init__(
        self,
        *,
        db_path: Path | str,
        memory_runtime_config: Optional[Dict[str, Any]] = None,
        memory_manager_config: Optional[Dict[str, Any]] = None,
        operation_reporter: Optional[MemoryOperationReporter] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize memory storage, manager, and runtime batching.

        Application-facing callers provide ``db_path`` and the two memory
        config mappings. The runtime owns the ``SessionDB``, embedding client,
        and standard ``MemoryNodeManager`` lifecycle.
        """
        manager_config = dict(memory_manager_config or {})
        configured_embedding = manager_config.get("embedding")
        embedding_config = (
            dict(configured_embedding)
            if isinstance(configured_embedding, dict)
            else {}
        )

        database = SessionDB(Path(db_path).expanduser().resolve())
        try:
            manager = MemoryNodeManager(
                database,
                embedding_config=dict(embedding_config or {}),
                memory_manager_config=manager_config,
                operation_reporter=operation_reporter,
                logger=logger,
            )
        except Exception:
            database.close()
            raise
        self._memory_database = database
        self._memory_manager = manager
        self._logger = logger or logging.getLogger(__name__)
        if logger is not None:
            self._memory_manager.set_logger(logger)
        runtime_config = dict(memory_runtime_config or {})
        self._prompt_language_mode = str(
            runtime_config.get("memory_prompt_language_mode")
            or "source"
        ).strip().lower()
        self._embedding_client = EmbeddingClient(
            dict(embedding_config or getattr(manager, "_embedding_cfg", {}) or {}),
        )
        if hasattr(self._memory_manager, "set_embedding_client"):
            self._memory_manager.set_embedding_client(self._embedding_client)
        interaction_segmentation_config = build_online_segmentation_config(runtime_config)
        self._interaction_segmenter = OnlineSemanticSegmenter(
            self._embedding_client,
            interaction_segmentation_config,
        )
        self._transcript_episode_max_segments = max(
            1,
            int(runtime_config.get("transcript_episode_max_segments") or 80),
        )
        self._transcript_episode_max_chars = max(
            1,
            int(runtime_config.get("transcript_episode_max_chars") or 12000),
        )
        transcript_gap_seconds = runtime_config.get("transcript_episode_max_gap_seconds")
        self._transcript_episode_max_gap_seconds = float(
            60.0 if transcript_gap_seconds is None else transcript_gap_seconds
        )
        self._pending_transcript_segments: List[Dict[str, Any]] = []
        self._pending_transcript_context: Optional[Dict[str, Any]] = None

    def close(self, timeout: Optional[float] = 30.0) -> None:
        """Drain owned tasks and release resources created by this runtime."""
        try:
            self.flush_task_queue(timeout=timeout)
        finally:
            try:
                self._memory_manager.shutdown_task_worker(
                    wait=True,
                    timeout=timeout,
                )
            finally:
                if self._memory_database is not None:
                    self._memory_database.close()

    @property
    def manager(self) -> MemoryNodeManager:
        """Return the manager owned by this runtime for diagnostics/adapters."""
        return self._memory_manager

    @property
    def database(self) -> SessionDB:
        """Return the database used by the owned manager."""
        if self._memory_database is not None:
            return self._memory_database
        return self._memory_manager._db

    def accept_single_interaction_turn(
        self,
        user_message: str,
        assistant_response: str = "",
        *,
        tags: Optional[List[str]] = None,
        turn_timestamp: Optional[Any] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Buffer one interaction and store a complete batch when due."""
        if not self._memory_manager.enabled:
            return {"queued": False, "reason": "memory_disabled"}
        if turn_timestamp is None:
            turn_timestamp = extra.get("timestamp")
        turn = {
            "user_message": _compact_whitespace(user_message),
            "assistant_response": _compact_whitespace(assistant_response),
            "tags": list(tags or []),
            "turn_timestamp": _to_timestamp_text(turn_timestamp) or _now_text(),
        }
        if not turn["user_message"] and not turn["assistant_response"]:
            return {"queued": False, "reason": "empty_turn"}

        queued = False
        reason = "threshold_not_reached"
        pending_exchanges = self._interaction_segmenter.pending_exchange_snapshot()
        incoming_exchange = convert_interaction_turn_to_online_exchange(
            turn,
            len(pending_exchanges) + 1,
        )
        if pending_exchanges:
            should_finalize, boundary_decision = self._interaction_segmenter.should_finalize_pending_exchanges(
                incoming_exchange,
            )
            if should_finalize:
                flush_report = self._flush_pending_interaction_turns()
                queued = bool(flush_report.get("queued")) or queued
                reason = "" if queued else str(flush_report.get("reason") or boundary_decision.reason)
                if self._interaction_segmenter.has_pending_exchanges():
                    return {"queued": queued, "reason": reason}

        self._interaction_segmenter.append_pending_exchange(incoming_exchange)
        return {"queued": queued, "reason": "" if queued else reason}

    def _flush_pending_interaction_turns(self) -> Dict[str, Any]:
        """Submit the buffered interaction turns and clear them when queued."""
        if not self._interaction_segmenter.has_pending_exchanges():
            return {"queued": False, "reason": "no_pending_turns"}
        turns = self.get_pending_interaction_turns()
        queue_report = self._process_interaction_turns(turns)
        queued = bool(queue_report.get("queued"))
        if queued:
            self._interaction_segmenter.clear_pending_exchanges()
        return {
            "queued": queued,
            "reason": (
                ""
                if queued
                else str(queue_report.get("reason") or "queue_rejected")
            ),
        }

    def accept_single_transcript_segment(
        self,
        segment: Dict[str, Any],
        *,
        source_type: str = "allday_recording",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Buffer one transcript segment and queue completed transcript episodes."""
        if not self._memory_manager.enabled:
            return {"queued": False, "reason": "memory_disabled"}
        if not isinstance(segment, dict):
            return {"queued": False, "reason": "invalid_segment"}
        if not self._transcript_segment_text(segment):
            return {"queued": False, "reason": "empty_segment"}

        context = {
            "source_type": str(source_type or "allday_recording"),
            "tags": sorted(
                {
                    str(tag)
                    for tag in tags or []
                    if tag is not None and str(tag).strip()
                }
            ),
        }
        queued = False
        reason = "threshold_not_reached"
        should_flush = self._should_flush_pending_transcript_before(segment, context)
        if should_flush:
            flush_report = self._flush_pending_transcript_segments()
            queued = bool(flush_report.get("queued")) or queued
            if queued:
                reason = ""
            elif flush_report.get("reason"):
                reason = str(flush_report.get("reason"))
            if self._pending_transcript_segments:
                return {"queued": queued, "reason": reason}

        self._pending_transcript_context = dict(context)
        self._pending_transcript_segments.append(dict(segment))
        return {"queued": queued, "reason": "" if queued else reason}

    def _flush_pending_transcript_segments(self) -> Dict[str, Any]:
        """Normalize and submit buffered transcript segments as one episode."""
        if not self._pending_transcript_segments:
            return {"queued": False, "reason": "no_pending_segments"}
        segments = list(self._pending_transcript_segments)
        context = dict(self._pending_transcript_context or {})
        raw_segments, prompt_language = self._normalize_transcript_segments_into_memory_raw_segments(
            segments,
        )
        if not raw_segments:
            self._pending_transcript_segments.clear()
            self._pending_transcript_context = None
            return {"queued": False, "reason": "invalid_pending_segments"}
        queue_report = self._process_transcript_segments(
            raw_segments,
            context,
            prompt_language=prompt_language,
        )
        queued = bool(queue_report.get("queued"))
        if queued:
            self._pending_transcript_segments.clear()
            self._pending_transcript_context = None
        return {
            "queued": queued,
            "reason": (
                ""
                if queued
                else str(queue_report.get("reason") or "queue_rejected")
            ),
        }

    def trigger_memory_reflect(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Queue reflection after pending interaction and transcript storage."""
        interaction_flush_report = self._flush_pending_interaction_turns()
        transcript_flush_report = self._flush_pending_transcript_segments()

        report = self._memory_manager.submit_memory_reflect_task(*args, **kwargs) or {}
        report["pending_interaction_flush"] = interaction_flush_report
        report["pending_transcript_flush"] = transcript_flush_report
        
        return report
    
    def trigger_memory_recall(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Run recall immediately through the manager without adding buffering."""
        if not kwargs.get("prompt_language"):
            query = kwargs.get("query")
            if query is None and args:
                query = args[0]
            kwargs["prompt_language"] = self._resolve_prompt_language_from_segments(
                [{"text": str(query or "")}]
            )
        return self._memory_manager.process_memory_recall_immediately(*args, **kwargs)

    def flush_task_queue(self, timeout: Optional[float] = None) -> bool:
        """Wait for queued store and reflection tasks at an explicit boundary."""
        if self._interaction_segmenter.has_pending_exchanges():
            self._flush_pending_interaction_turns()
        if self._pending_transcript_segments:
            self._flush_pending_transcript_segments()
        return self._memory_manager.flush_task_queue(timeout=timeout)

    def get_pending_interaction_turns(self) -> List[Dict[str, Any]]:
        """Return raw turns currently held by the interaction segmenter."""
        turns: List[Dict[str, Any]] = []
        for exchange in self._interaction_segmenter.pending_exchange_snapshot():
            if isinstance(exchange.raw, dict):
                turns.append(dict(exchange.raw))
        return turns

    def has_pending_interaction_turns(self) -> bool:
        """Return whether interaction turns are waiting for storage."""
        return self._interaction_segmenter.has_pending_exchanges()

    def _process_interaction_turns(self, turns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert interaction turns and submit them as an assistant episode."""
        (
            raw_segments,
            prompt_language,
        ) = self._normalize_interaction_turns_to_memory_raw_segments(turns)
        tags = sorted({tag for turn in turns for tag in turn.get("tags", [])})
        return self._memory_manager.submit_memory_store_task(
            raw_segments=raw_segments,
            source_type="assistant_wakeup",
            tags=tags,
            prompt_language=prompt_language,
        )

    def _process_transcript_segments(
        self,
        raw_segments: Sequence[Dict[str, Any]],
        context: Dict[str, Any],
        *,
        prompt_language: str,
    ) -> Dict[str, Any]:
        """Submit normalized transcript segments with their source context."""
        tags = {
            str(tag)
            for tag in context.get("tags") or []
            if tag is not None and str(tag).strip()
        }
        for segment in raw_segments:
            tags.update(
                str(tag)
                for tag in segment.get("tags") or []
                if tag is not None and str(tag).strip()
            )
        return self._memory_manager.submit_memory_store_task(
            raw_segments=list(raw_segments),
            source_type=str(context.get("source_type") or "allday_recording"),
            tags=sorted(tags),
            prompt_language=prompt_language,
        )

    def _resolve_prompt_language_from_segments(
        self,
        records: Sequence[Dict[str, Any]],
    ) -> str:
        """Resolve prompt language for normalized segments or interaction turns."""
        mode = self._prompt_language_mode
        if mode in {"en", "english", "force_en"}:
            return "en"
        if mode in {"zh", "chinese", "force_zh"}:
            return "zh"
        sample = "\n".join(
            text
            for record in list(records)[:12]
            if isinstance(record, dict)
            for text in (
                str(record.get("text") or "").strip()
                or str(record.get("user_message") or "").strip()
                or str(record.get("assistant_response") or "").strip(),
            )
            if text
        )
        return "zh" if re.search(r"[\u4e00-\u9fff]", sample) else "en"

    def _should_flush_pending_transcript_before(
        self,
        next_segment: Dict[str, Any],
        next_context: Dict[str, Any],
    ) -> bool:
        """Check whether the next segment must start a new transcript episode."""
        if not self._pending_transcript_segments:
            return False
        if self._pending_transcript_context != next_context:
            return True
        pending_chars = sum(
            len(self._transcript_segment_text(segment))
            for segment in self._pending_transcript_segments
        )
        next_chars = len(self._transcript_segment_text(next_segment))
        previous = self._pending_transcript_segments[-1]
        gap_seconds = OnlineSemanticSegmenter.timestamp_gap_seconds(
            self._transcript_segment_end_time(previous)
            or self._transcript_segment_start_time(previous),
            self._transcript_segment_start_time(next_segment),
        )
        return bool(
            len(self._pending_transcript_segments)
            >= self._transcript_episode_max_segments
            or pending_chars + next_chars > self._transcript_episode_max_chars
            or (
                self._transcript_episode_max_gap_seconds >= 0
                and gap_seconds is not None
                and gap_seconds > self._transcript_episode_max_gap_seconds
            )
        )

    @staticmethod
    def _transcript_segment_text(segment: Dict[str, Any]) -> str:
        """Extract and compact the transcript text from supported input fields."""
        return _compact_whitespace(
            segment.get("text")
            or segment.get("asr_text")
            or segment.get("reference_text")
            or segment.get("utterance")
            or ""
        )

    @staticmethod
    def _transcript_segment_start_time(segment: Dict[str, Any]) -> str:
        """Extract the normalized start timestamp from a transcript segment."""
        return _to_timestamp_text(
            segment.get("started_at")
            or segment.get("start_timestamp")
            or segment.get("timestamp")
            or segment.get("start")
        )

    @staticmethod
    def _transcript_segment_end_time(segment: Dict[str, Any]) -> str:
        """Extract the normalized end timestamp, falling back to the start time."""
        started_at = MemoryRuntime._transcript_segment_start_time(segment)
        return _to_timestamp_text(
            segment.get("ended_at")
            or segment.get("end_timestamp")
            or segment.get("timestamp_end")
            or segment.get("end")
            or started_at
        )

    def _normalize_interaction_turns_to_memory_raw_segments(
        self,
        turns: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Convert frontend interaction turns into manager-compatible segments."""

        prompt_language = self._resolve_prompt_language_from_segments(turns)
        segments: List[Dict[str, Any]] = []
        for turn_index, turn in enumerate(turns, 1):
            timestamp = _to_timestamp_text(turn.get("turn_timestamp")) or _now_text()
            tags = list(turn.get("tags") or [])
            user_text = _compact_whitespace(turn.get("user_message") or "")
            assistant_text = _compact_whitespace(turn.get("assistant_response") or "")
            if user_text:
                segments.append({
                    "speaker": "user" if prompt_language == "en" else "用户",
                    "text": user_text,
                    "started_at": timestamp,
                    "ended_at": timestamp,
                    "tags": tags,
                    "turn_index": turn_index,
                })
            if assistant_text:
                segments.append({
                    "speaker": "assistant" if prompt_language == "en" else "助手",
                    "text": assistant_text,
                    "started_at": timestamp,
                    "ended_at": timestamp,
                    "tags": tags,
                    "turn_index": turn_index,
                })
        return segments, prompt_language

    def _normalize_transcript_segments_into_memory_raw_segments(
        self,
        segments: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Validate and normalize transcript segments for episode storage."""
        normalized: List[Dict[str, Any]] = []
        for index, segment in enumerate(segments, 1):
            if not isinstance(segment, dict):
                continue
            text = self._transcript_segment_text(segment)
            if not text:
                continue
            speaker = _compact_whitespace(
                segment.get("speaker")
                or segment.get("speaker_name")
                or segment.get("speaker_id")
                or "unknown_speaker"
            )
            started_at = self._transcript_segment_start_time(segment)
            ended_at = self._transcript_segment_end_time(segment)
            normalized.append({
                "speaker": speaker or "unknown_speaker",
                "text": text,
                "started_at": started_at or _now_text(),
                "ended_at": ended_at or started_at or _now_text(),
                "tags": list(segment.get("tags") or []),
                "segment_index": int(segment.get("segment_index") or index),
                "metadata": dict(segment.get("metadata") or {}),
            })
        normalized = sorted(
            normalized,
            key=lambda item: (
                str(item.get("started_at") or ""),
                str(item.get("ended_at") or ""),
                str(item.get("speaker") or ""),
            ),
        )
        return normalized, self._resolve_prompt_language_from_segments(normalized)
