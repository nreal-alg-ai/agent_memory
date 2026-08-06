"""Runtime adapter between application input events and memory storage.

``MemoryNodeManager`` owns episode persistence, reflection, and recall.  This
adapter owns the short-lived interaction buffer and converts frontend-shaped
turns or transcript segments into the manager's raw episode segments.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from .memory_manager import (
    MemoryNodeManager,
    _compact_whitespace,
    _now_text,
    _to_timestamp_text,
)


class MemoryRuntime:
    """Normalize application input and batch it before episode storage."""

    def __init__(
        self,
        manager: MemoryNodeManager,
        *,
        memory_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._manager = manager
        config = dict(memory_config or {})
        self._min_dialogue_turns_before_store = max(
            1,
            int(
                config.get("min_dialogue_turns_before_store")
                or config.get("min_dilaogue_turns_before_store")
                or config.get("min_turns_before_store")
                or 1
            ),
        )
        self._max_dialogue_chars_before_store = max(
            1,
            int(
                config.get("max_dialogue_chars_before_store")
                or config.get("max_dilaogue_chars_before_store")
                or config.get("max_chars_before_store")
                or 2000
            ),
        )
        self._pending_interaction_turns: List[Dict[str, Any]] = []
        self._transcript_episode_max_segments = max(
            1,
            int(config.get("transcript_episode_max_segments") or 80),
        )
        self._transcript_episode_max_chars = max(
            1,
            int(config.get("transcript_episode_max_chars") or 12000),
        )
        transcript_gap_seconds = config.get("transcript_episode_max_gap_seconds")
        self._transcript_episode_max_gap_seconds = float(
            60.0 if transcript_gap_seconds is None else transcript_gap_seconds
        )
        self._pending_transcript_segments: List[Dict[str, Any]] = []
        self._pending_transcript_context: Optional[Dict[str, Any]] = None

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
        if not self._manager.enabled:
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

        self._pending_interaction_turns.append(turn)
        if self.should_flush_pending_interaction_turns():
            return self._flush_pending_interaction_turns()
        return {"queued": False, "reason": "threshold_not_reached"}

    def should_flush_pending_interaction_turns(
        self,
        turns: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> bool:
        """Return whether an interaction batch has reached a store threshold."""
        pending_turns = (
            list(turns)
            if turns is not None
            else self._pending_interaction_turns
        )
        if not pending_turns:
            return False
        return (
            len(pending_turns) >= self._min_dialogue_turns_before_store
            or self.interaction_turns_character_count(pending_turns)
            >= self._max_dialogue_chars_before_store
        )

    @staticmethod
    def interaction_turns_character_count(
        turns: Sequence[Dict[str, Any]],
    ) -> int:
        """Count user+assistant content characters in interaction turns."""
        return sum(
            len(str(item.get("user_message") or ""))
            + len(str(item.get("assistant_response") or ""))
            for item in turns
        )

    def _flush_pending_interaction_turns(self) -> Dict[str, Any]:
        if not self._pending_interaction_turns:
            return {"queued": False, "reason": "no_pending_turns"}
        turns = list(self._pending_interaction_turns)
        queue_report = self._process_interaction_turns(turns)
        queued = bool(queue_report.get("accepted"))
        if queued:
            self._pending_interaction_turns.clear()
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
        episode_type: str = "ambient_transcript",
        source_ref: str = "",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Buffer one transcript segment and queue completed transcript episodes."""
        if not self._manager.enabled:
            return {"queued": False, "reason": "memory_disabled"}
        if not isinstance(segment, dict):
            return {"queued": False, "reason": "invalid_segment"}
        if not self._transcript_segment_text(segment):
            return {"queued": False, "reason": "empty_segment"}

        context = {
            "source_type": str(source_type or "allday_recording"),
            "episode_type": str(episode_type or "ambient_transcript"),
            "source_ref": str(source_ref or ""),
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
        if (
            self._pending_transcript_segments
            and self._pending_transcript_context != context
        ):
            flush_report = self._flush_pending_transcript_segments()
            queued = bool(flush_report.get("queued"))
            reason = "" if queued else str(flush_report.get("reason") or "queue_rejected")
            if self._pending_transcript_segments:
                return {"queued": queued, "reason": reason}

        if self._should_flush_pending_transcript_before(segment):
            flush_report = self._flush_pending_transcript_segments()
            queued = bool(flush_report.get("queued")) or queued
            if queued:
                reason = ""
            elif flush_report.get("reason"):
                reason = str(flush_report.get("reason"))
        if self._pending_transcript_context is None:
            self._pending_transcript_context = dict(context)
        self._pending_transcript_segments.append(dict(segment))
        return {"queued": queued, "reason": "" if queued else reason}

    def _flush_pending_transcript_segments(self) -> Dict[str, Any]:
        if not self._pending_transcript_segments:
            return {"queued": False, "reason": "no_pending_segments"}
        segments = list(self._pending_transcript_segments)
        context = dict(self._pending_transcript_context or {})
        raw_segments = self._normalize_transcript_segments_into_memory_raw_segments(
            segments,
        )
        if not raw_segments:
            self._pending_transcript_segments.clear()
            self._pending_transcript_context = None
            return {"queued": False, "reason": "invalid_pending_segments"}
        queue_report = self._process_transcript_segments(raw_segments, context)
        queued = bool(queue_report.get("accepted"))
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

        report = self._manager.submit_memory_reflect_task(*args, **kwargs) or {}
        report["pending_interaction_flush"] = interaction_flush_report
        report["pending_transcript_flush"] = transcript_flush_report
        
        return report
    
    def trigger_memory_recall(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self._manager.process_memory_recall_immediately(*args, **kwargs)

    def flush_store_queue(self, timeout: Optional[float] = None) -> bool:
        """Wait for queued store and reflection tasks at an explicit boundary."""
        if self._pending_interaction_turns:
            self._flush_pending_interaction_turns()
        if self._pending_transcript_segments:
            self._flush_pending_transcript_segments()
        return self._manager.flush_store_queue(timeout=timeout)

    def _process_interaction_turns(self, turns: List[Dict[str, Any]]) -> Dict[str, Any]:
        raw_segments = self._normalize_interaction_turns_to_memory_raw_segments(turns)
        tags = sorted({tag for turn in turns for tag in turn.get("tags", [])})
        return self._manager.submit_memory_store_task(
            raw_segments=raw_segments,
            source_type="assistant_wakeup",
            episode_type="interaction",
            tags=tags,
            source_ref="store_turn",
        )

    def _process_transcript_segments(
        self,
        raw_segments: Sequence[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
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
        return self._manager.submit_memory_store_task(
            raw_segments=list(raw_segments),
            source_type=str(context.get("source_type") or "allday_recording"),
            episode_type=str(context.get("episode_type") or "ambient_transcript"),
            tags=sorted(tags),
            source_ref=str(context.get("source_ref") or ""),
        )

    def _should_flush_pending_transcript_before(
        self,
        next_segment: Dict[str, Any],
    ) -> bool:
        if not self._pending_transcript_segments:
            return False
        pending_chars = sum(
            len(self._transcript_segment_text(segment))
            for segment in self._pending_transcript_segments
        )
        next_chars = len(self._transcript_segment_text(next_segment))
        previous = self._pending_transcript_segments[-1]
        gap_seconds = self._timestamp_gap_seconds(
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
        return _compact_whitespace(
            segment.get("text")
            or segment.get("asr_text")
            or segment.get("reference_text")
            or segment.get("utterance")
            or ""
        )

    @staticmethod
    def _transcript_segment_start_time(segment: Dict[str, Any]) -> str:
        return _to_timestamp_text(
            segment.get("started_at")
            or segment.get("start_timestamp")
            or segment.get("timestamp")
            or segment.get("start")
        )

    @staticmethod
    def _transcript_segment_end_time(segment: Dict[str, Any]) -> str:
        started_at = MemoryRuntime._transcript_segment_start_time(segment)
        return _to_timestamp_text(
            segment.get("ended_at")
            or segment.get("end_timestamp")
            or segment.get("timestamp_end")
            or segment.get("end")
            or started_at
        )

    @staticmethod
    def _timestamp_gap_seconds(
        previous_end: Any,
        current_start: Any,
    ) -> Optional[float]:
        if not previous_end or not current_start:
            return None
        try:
            previous = datetime.fromisoformat(
                str(previous_end).replace("Z", "+00:00"),
            )
            current = datetime.fromisoformat(
                str(current_start).replace("Z", "+00:00"),
            )
        except ValueError:
            return None
        return (current - previous).total_seconds()

    def _normalize_interaction_turns_to_memory_raw_segments(
        self,
        turns: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
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

    def _normalize_transcript_segments_into_memory_raw_segments(
        self,
        segments: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
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
            role = _compact_whitespace(segment.get("role") or speaker or "speaker")
            started_at = self._transcript_segment_start_time(segment)
            ended_at = self._transcript_segment_end_time(segment)
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
