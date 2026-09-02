"""Runtime adapter between application input events and memory storage.

``MemoryNodeManager`` owns episode persistence, reflection, and recall.  This
adapter owns the short-lived interaction buffer and converts frontend-shaped
turns or transcript segments into the manager's raw episode segments.
"""

from __future__ import annotations

import logging
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    TranscriptSemanticUnit,
    TranscriptUtteranceAssembler,
    build_online_segmentation_config,
    build_transcript_aggregation_config,
    build_transcript_segmentation_config,
    convert_interaction_turn_to_online_exchange,
    convert_transcript_unit_to_online_exchange,
)

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
        transcript_segmentation_config = build_transcript_segmentation_config(
            runtime_config,
        )
        self._transcript_segmenter = OnlineSemanticSegmenter(
            self._embedding_client,
            transcript_segmentation_config,
        )
        self._transcript_utterance_assembler = TranscriptUtteranceAssembler(
            build_transcript_aggregation_config(runtime_config),
        )
        allday_segmentation_config = runtime_config.get(
            "allday_recording_segmentation",
        )
        if not isinstance(allday_segmentation_config, dict):
            allday_segmentation_config = {}
        self._transcript_segmentation_log_decisions = bool(
            allday_segmentation_config.get("log_decisions", False),
        )
        self._episode_summary_trigger_gap_seconds = float(
            allday_segmentation_config.get("episode_summary_trigger_gap_seconds")
            or allday_segmentation_config.get("max_time_gap_seconds")
            or 60.0
        )
        self._transcript_previous_segment_end: Optional[datetime] = None
        self._transcript_episode_source_type = "allday_recording"
        self._transcript_episode_tags: List[str] = []
        self._transcript_episode_prompt_language = "zh"
        self._transcript_has_pending_episode_facts = False

    def close(self, timeout: Optional[float] = 30.0) -> None:
        """Drain owned tasks and release resources created by this runtime."""
        try:
            self.flush_task_queue(timeout=timeout)
        finally:
            try:
                shutdown_ok = self._memory_manager.shutdown_task_worker(
                    wait=True,
                    timeout=timeout,
                )
                if not shutdown_ok:
                    # A bounded shutdown timeout must not allow the database
                    # to close while a reflection transaction is still in
                    # flight. Continue draining without a deadline so all
                    # queued writes are committed before the connection closes.
                    self._logger.warning(
                        "Memory worker did not stop within timeout=%s; "
                        "continuing to drain queued tasks before closing database",
                        timeout,
                    )
                    shutdown_ok = self._memory_manager.shutdown_task_worker(
                        wait=True,
                        timeout=None,
                    )
                if not shutdown_ok:
                    raise RuntimeError(
                        "Memory worker stopped before completing all queued tasks"
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
        incoming_embedding = self._interaction_segmenter.embed_exchange(
            incoming_exchange,
        ).embedding
        if pending_exchanges:
            should_finalize, boundary_decision = self._interaction_segmenter.should_finalize_pending_exchanges(
                incoming_exchange,
                incoming_embedding,
            )
            if should_finalize:
                flush_report = self._flush_pending_interaction_turns()
                queued = bool(flush_report.get("queued")) or queued
                reason = "" if queued else str(flush_report.get("reason") or boundary_decision.reason)
                if self._interaction_segmenter.has_pending_exchanges():
                    return {"queued": queued, "reason": reason}

        self._interaction_segmenter.append_pending_exchange(
            incoming_exchange,
            incoming_embedding,
        )
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
        is_last_segment: bool = False,
    ) -> Dict[str, Any]:
        """Buffer one transcript segment and queue completed transcript episodes."""
        if not self._memory_manager.enabled:
            return {"queued": False, "reason": "memory_disabled"}
        if not isinstance(segment, dict):
            return {"queued": False, "reason": "invalid_segment"}
        normalized_segment = self._normalize_single_transcript_segment(segment)
        if normalized_segment is None:
            return {"queued": False, "reason": "empty_segment"}

        current_start = self._parse_runtime_timestamp(normalized_segment.get("started_at"))
        gap_trigger = self.should_trigger_episode_summary(normalized_segment)
        if gap_trigger and (
            self._transcript_segmenter.has_pending_exchanges()
            or self._transcript_utterance_assembler.has_pending_segments()
            or self._transcript_has_pending_episode_facts
        ):
            gap_summary_report = self._finalize_transcript_episode_summary(reason="time_gap")
        else:
            gap_summary_report = None

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
        normalized_segment["_memory_context"] = dict(context)
        self._transcript_episode_source_type = context["source_type"]
        self._transcript_episode_tags = sorted(set(self._transcript_episode_tags).union(context["tags"]))
        self._transcript_previous_segment_end = self._parse_runtime_timestamp(normalized_segment.get("ended_at")) or current_start

        queued = bool((gap_summary_report or {}).get("queued"))
        completed_unit = self._transcript_utterance_assembler.append(
            normalized_segment,
        )
        if completed_unit is None:
            final_summary_report = None
            if is_last_segment:
                final_summary_report = self._finalize_transcript_episode_summary(reason="last_segment")
            return {
                "queued": queued or bool((final_summary_report or {}).get("queued")),
                "reason": "threshold_not_reached",
                "episode_summary": final_summary_report,
            }
        append_report = self._append_transcript_semantic_unit(completed_unit)
        queued = bool(append_report.get("queued")) or queued
        if not append_report.get("accepted"):
            return {
                "queued": queued,
                "reason": str(append_report.get("reason") or "queue_rejected"),
            }
        if is_last_segment:
            final_summary_report = self._finalize_transcript_episode_summary(reason="last_segment")
            queued = queued or bool(final_summary_report.get("queued"))
        else:
            final_summary_report = None
        return {
            "queued": queued,
            "reason": "" if queued else "threshold_not_reached",
            "episode_summary": final_summary_report or gap_summary_report,
        }

    def should_trigger_episode_summary(
        self,
        current_segment: Dict[str, Any],
        previous_segment: Optional[Dict[str, Any]] = None,
        *,
        is_last_segment: bool = False,
    ) -> bool:
        """Return whether a logical transcript episode boundary was reached."""
        if is_last_segment:
            return True
        previous_end = self._transcript_previous_segment_end
        if previous_segment is not None:
            previous_end = self._parse_runtime_timestamp(
                previous_segment.get("ended_at") or previous_segment.get("end_time")
            )
        if previous_end is None:
            return False
        current_start = self._parse_runtime_timestamp(
            current_segment.get("started_at") or current_segment.get("start_time")
        )
        if current_start is None:
            return False
        return (current_start - previous_end).total_seconds() > self._episode_summary_trigger_gap_seconds

    def _parse_runtime_timestamp(self, value: Any) -> Optional[datetime]:
        text = _to_timestamp_text(value)
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
            return parsed
        except (TypeError, ValueError):
            return None

    def _finalize_transcript_episode_summary(self, *, reason: str) -> Dict[str, Any]:
        flush_report = self._flush_pending_transcript_segments(reason=reason)
        report = self._memory_manager.submit_memory_episode_summary_task(
            source_type=self._transcript_episode_source_type,
            tags=self._transcript_episode_tags,
            prompt_language=self._transcript_episode_prompt_language,
        )
        self._transcript_has_pending_episode_facts = False
        self._transcript_episode_tags = []
        report["trigger_reason"] = reason
        report["transcript_flush"] = flush_report
        return report

    def _append_transcript_semantic_unit(
        self,
        unit: TranscriptSemanticUnit,
    ) -> Dict[str, Any]:
        """Use one assembled utterance to decide whether the prior episode ends."""
        queued = False
        incoming_exchange = convert_transcript_unit_to_online_exchange(unit)
        incoming_embedding = self._transcript_segmenter.embed_exchange(
            incoming_exchange,
        ).embedding
        pending_exchanges = self._transcript_segmenter.pending_exchange_snapshot()
        if pending_exchanges:
            should_finalize, decision = (
                self._transcript_segmenter.should_finalize_pending_exchanges(
                    incoming_exchange,
                    incoming_embedding,
                )
            )
            self._log_transcript_segmentation_decision(
                unit,
                decision=decision,
            )
            flush_report = (
                self._process_pending_transcript_exchanges(
                    reason=decision.reason,
                )
                if should_finalize
                else None
            )
            if flush_report is not None:
                queued = bool(flush_report.get("queued")) or queued
                if self._transcript_segmenter.has_pending_exchanges():
                    return {
                        "accepted": False,
                        "queued": queued,
                        "reason": str(flush_report.get("reason") or "queue_rejected"),
                    }
        else:
            _, decision = self._transcript_segmenter.should_finalize_pending_exchanges(
                incoming_exchange,
                incoming_embedding,
            )
            self._log_transcript_segmentation_decision(unit, decision=decision)

        self._transcript_segmenter.append_pending_exchange(
            incoming_exchange,
            incoming_embedding,
        )
        return {"accepted": True, "queued": queued, "reason": ""}

    def _flush_pending_transcript_segments(
        self,
        *,
        reason: str = "explicit_flush",
    ) -> Dict[str, Any]:
        """Finalize the current utterance, then submit pending transcript exchanges."""
        queued = False
        completed_unit = self._transcript_utterance_assembler.flush()
        if completed_unit is not None:
            append_report = self._append_transcript_semantic_unit(completed_unit)
            queued = bool(append_report.get("queued")) or queued
            if not append_report.get("accepted"):
                return {
                    "queued": queued,
                    "reason": str(append_report.get("reason") or "queue_rejected"),
                }
        store_report = self._process_pending_transcript_exchanges(reason=reason)
        return {
            "queued": bool(store_report.get("queued")) or queued,
            "reason": str(store_report.get("reason") or ""),
        }

    def _process_pending_transcript_exchanges(
        self,
        *,
        reason: str,
    ) -> Dict[str, Any]:
        """Normalize and submit transcript spans held by the semantic segmenter."""
        pending_exchanges = self._transcript_segmenter.pending_exchange_snapshot()
        if not pending_exchanges:
            return {"queued": False, "reason": "no_pending_segments"}
        segments = self._transcript_raw_segments_from_exchanges(pending_exchanges)
        context = self._transcript_context_from_exchanges(pending_exchanges)
        raw_segments, prompt_language = self._normalize_transcript_segments_into_memory_raw_segments(
            segments,
        )
        if not raw_segments:
            self._transcript_segmenter.clear_pending_exchanges()
            return {"queued": False, "reason": "invalid_pending_segments"}
        self._log_info(
            "memory_runtime",
            "transcript_episode_batch_detail",
            {
                "reason": reason,
                "source_type": context.get("source_type"),
                "tags": context.get("tags") or [],
                "raw_segment_count": len(segments),
                "semantic_unit_count": len(pending_exchanges),
                "prompt_language": prompt_language,
                "segments": raw_segments,
            },
        )
        queue_report = self._process_transcript_segments(
            raw_segments,
            context,
            prompt_language=prompt_language,
        )
        self._transcript_episode_prompt_language = prompt_language
        queued = bool(queue_report.get("queued"))
        if queued:
            self._transcript_has_pending_episode_facts = True
            self._logger.info(
                "transcript episode queued reason=%s raw_segment_count=%s semantic_unit_count=%s",
                reason,
                len(segments),
                len(pending_exchanges),
            )
            self._transcript_segmenter.clear_pending_exchanges()
        return {
            "queued": queued,
            "reason": (
                ""
                if queued
                else str(queue_report.get("reason") or "queue_rejected")
            ),
        }

    @staticmethod
    def _transcript_raw_segments_from_exchanges(
        exchanges: Sequence[Any],
    ) -> List[Dict[str, Any]]:
        """Expand the original ASR spans retained by transcript exchanges."""
        segments: List[Dict[str, Any]] = []
        for exchange in exchanges:
            raw = exchange.raw if isinstance(exchange.raw, dict) else {}
            raw_segments = raw.get("raw_segments")
            if not isinstance(raw_segments, list):
                continue
            segments.extend(
                dict(segment)
                for segment in raw_segments
                if isinstance(segment, dict)
            )
        return segments

    @staticmethod
    def _transcript_context_from_exchanges(
        exchanges: Sequence[Any],
    ) -> Dict[str, Any]:
        """Derive one storage context from the raw spans in pending exchanges."""
        source_type = "allday_recording"
        tags: set[str] = set()
        for exchange in exchanges:
            raw = exchange.raw if isinstance(exchange.raw, dict) else {}
            raw_segments = raw.get("raw_segments")
            if not isinstance(raw_segments, list):
                continue
            for segment in raw_segments:
                if not isinstance(segment, dict):
                    continue
                context = segment.get("_memory_context")
                if not isinstance(context, dict):
                    continue
                resolved_source_type = str(context.get("source_type") or "").strip()
                if resolved_source_type:
                    source_type = resolved_source_type
                tags.update(
                    str(tag)
                    for tag in context.get("tags") or []
                    if tag is not None and str(tag).strip()
                )
        return {"source_type": source_type, "tags": sorted(tags)}

    def _log_transcript_segmentation_decision(
        self,
        unit: TranscriptSemanticUnit,
        *,
        decision: Optional[Any] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Emit per-unit semantic boundary details when transcript logging is enabled."""
        if not self._transcript_segmentation_log_decisions:
            return
        resolved_reason = str(reason or getattr(decision, "reason", "append"))
        self._logger.info(
            "transcript segmentation decision unit=%s reason=%s raw_segment_count=%s "
            "token_count=%s speakers=%s cut_probability=%s score=%s "
            "semantic_surprise=%s cohesion_drop=%s time_gap_seconds=%s "
            "scoring_mode=%s rolling_tail_units=%s rolling_tail_tokens=%s text=%s",
            unit.index,
            resolved_reason,
            len(unit.raw_segments),
            unit.token_count,
            ",".join(unit.speaker_labels),
            getattr(decision, "cut_probability", None),
            getattr(decision, "score", None),
            getattr(decision, "semantic_surprise", None),
            getattr(decision, "cohesion_drop", None),
            getattr(decision, "time_gap_seconds", None),
            getattr(decision, "scoring_mode", None),
            getattr(decision, "rolling_window_tail_units", None),
            getattr(decision, "rolling_window_tail_tokens", None),
            unit.text[:240],
        )

    def _log_info(self, scope: str, event: str, payload: Dict[str, Any]) -> None:
        """Emit a structured JSON log record for runtime diagnostics."""
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

    def trigger_memory_reflect(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Queue reflection after pending interaction and transcript storage."""
        interaction_flush_report = self._flush_pending_interaction_turns()
        transcript_flush_report = self._flush_pending_transcript_segments(
            reason="reflect",
        )
        if transcript_flush_report.get("queued") or self._transcript_has_pending_episode_facts:
            transcript_flush_report["episode_summary"] = self._memory_manager.submit_memory_episode_summary_task(
                source_type=self._transcript_episode_source_type,
                tags=self._transcript_episode_tags,
                prompt_language=self._transcript_episode_prompt_language,
            )
            self._transcript_has_pending_episode_facts = False
            self._transcript_episode_tags = []

        report = self._memory_manager.submit_memory_reflect_task(*args, **kwargs) or {}
        report["pending_interaction_flush"] = interaction_flush_report
        report["pending_transcript_flush"] = transcript_flush_report
        
        return report
    
    def trigger_memory_recall(
        self,
        query: str,
        *,
        tags: Optional[List[str]] = None,
        time_end: Optional[str] = None,
        prompt_language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run recall immediately through the manager without adding buffering."""
        resolved_prompt_language = str(prompt_language or "").strip().lower()
        if not resolved_prompt_language:
            resolved_prompt_language = self._resolve_prompt_language_from_segments(
                [{"text": str(query or "")}]
            )
        return self._memory_manager.process_memory_recall_immediately(
            query=str(query or ""),
            tags=tags,
            time_end=time_end,
            prompt_language=resolved_prompt_language,
        )

    def flush_task_queue(self, timeout: Optional[float] = None) -> bool:
        """Submit buffered work; the manager worker drains tasks in FIFO order."""
        if self._interaction_segmenter.has_pending_exchanges():
            interaction_report = self._flush_pending_interaction_turns()
            if not interaction_report.get("queued") and interaction_report.get("reason") not in {"", "no_pending_segments"}:
                return False
        transcript_flush_report = self._flush_pending_transcript_segments(reason="explicit_input_boundary")
        return not (
            not transcript_flush_report.get("queued")
            and transcript_flush_report.get("reason") not in {"", "no_pending_segments"}
        )

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
            normalized_segment = self._normalize_single_transcript_segment(
                segment,
                fallback_index=index,
            )
            if normalized_segment is not None:
                normalized.append(normalized_segment)
        normalized = sorted(
            normalized,
            key=lambda item: (
                str(item.get("started_at") or ""),
                str(item.get("ended_at") or ""),
                str(item.get("speaker") or ""),
            ),
        )
        return normalized, self._resolve_prompt_language_from_segments(normalized)

    def _normalize_single_transcript_segment(
        self,
        segment: Any,
        *,
        fallback_index: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """Normalize one frontend transcript record before utterance assembly."""
        if not isinstance(segment, dict):
            return None
        text = self._transcript_segment_text(segment)
        if not text:
            return None
        speaker = _compact_whitespace(
            segment.get("speaker")
            or segment.get("speaker_name")
            or segment.get("speaker_id")
            or "unknown_speaker"
        )
        started_at = self._transcript_segment_start_time(segment)
        ended_at = self._transcript_segment_end_time(segment)
        try:
            segment_index = int(segment.get("segment_index") or fallback_index)
        except (TypeError, ValueError):
            segment_index = fallback_index
        return {
            "speaker": speaker or "unknown_speaker",
            "text": text,
            "started_at": started_at or _now_text(),
            "ended_at": ended_at or started_at or _now_text(),
            "tags": list(segment.get("tags") or []),
            "segment_index": segment_index,
            "metadata": dict(segment.get("metadata") or {}),
        }
