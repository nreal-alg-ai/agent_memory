"""Runtime adapter between application input events and memory storage.

``MemoryNodeManager`` owns episode persistence, reflection, and recall.  This
adapter owns the short-lived interaction buffer and converts frontend-shaped
turns or transcript segments into the manager's raw episode segments.
"""

from __future__ import annotations

import logging
import time
import re
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .embedding_client import EmbeddingClient
from .utils import _as_embedding_vector, _cal_embedding_cosine_similarity, _sigmoid, _centroid, _cohesion

from .memory_manager import (
    MemoryNodeManager,
    _compact_whitespace,
    _now_text,
    _to_timestamp_text,
)

INTERACTION_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_.$'-]+|[^\s]")


@dataclass
class SegmentDecision:
    exchange_index: int
    reason: str
    cut_probability: Optional[float] = None
    score: Optional[float] = None
    semantic_surprise: Optional[float] = None
    robust_surprise: Optional[float] = None
    absolute_surprise: Optional[float] = None
    cohesion_before: Optional[float] = None
    cohesion_after: Optional[float] = None
    cohesion_drop: Optional[float] = None
    length_signal: Optional[float] = None
    turn_signal: Optional[float] = None
    centroid_similarity: Optional[float] = None
    recent_similarity: Optional[float] = None
    prospective_tokens: Optional[int] = None
    prospective_exchanges: Optional[int] = None
    time_gap_seconds: Optional[float] = None


@dataclass
class OnlineSegmentExchange:
    index: int
    text: str
    token_count: int
    timestamp: str = ""
    raw: Optional[Dict[str, Any]] = None


@dataclass
class ActiveExchange:
    exchange: Any
    embedding: np.ndarray


@dataclass
class OnlineSegmentationConfig:
    threshold: float = 0.60
    bias: float = -1.10
    surprise_history_window: int = 64
    min_surprise_history: int = 5
    robust_surprise_weight: float = 0.8
    absolute_surprise_weight: float = 0.8
    cohesion_drop_weight: float = 1.0
    length_weight: float = 0.40
    turn_count_weight: float = 0.40
    max_pending_turns: int = 0
    max_pending_tokens: int = 500
    min_pending_tokens: int = 100
    min_pending_turns: int = 2
    min_segment_override_probability: float = 0.90
    max_time_gap_seconds: float = -1.0


def _estimate_interaction_token_count(text: str) -> int:
    return max(1, len(INTERACTION_TOKEN_RE.findall(str(text or ""))))


def _exchange_text_from_turn(turn: Dict[str, Any]) -> str:
    user_text = _compact_whitespace(turn.get("user_message") or "")
    assistant_text = _compact_whitespace(turn.get("assistant_response") or "")
    return f"用户：{user_text}\n助手：{assistant_text}".strip()

def _clipped(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def robust_surprise_signal(
    surprise: float,
    history: Sequence[float],
    min_history: int,
) -> float:
    required_history = max(1, int(min_history))
    if len(history) < required_history:
        return 0.0
    values = np.asarray(list(history), dtype=np.float32)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1e-6, mad)
    return _clipped((surprise - median) / scale, -2.0, 4.0)


def absolute_surprise_signal(surprise: float) -> float:
    return _clipped((surprise - 0.20) / 0.14, -1.0, 2.5)


def _linear(value: float, start: float, end: float, low: float, high: float) -> float:
    if end <= start:
        return high
    ratio = _clipped((value - start) / (end - start), 0.0, 1.0)
    return low + ratio * (high - low)


def length_pressure(token_count: int, config: OnlineSegmentationConfig) -> float:
    min_tokens = max(1, int(config.min_pending_tokens))
    max_tokens = max(min_tokens + 1, int(config.max_pending_tokens))
    length = float(max(0, token_count))
    early_boundary = 0.7 * min_tokens

    if length < early_boundary:
        return -1.30
    if length < min_tokens:
        return _linear(length, early_boundary, min_tokens, -0.80, 0.0)
    if length < max_tokens:
        return _linear(length, min_tokens, max_tokens, 0.45, 2.80)
    return 3.00


def turn_count_pressure(exchange_count: int) -> float:
    count = max(1, int(exchange_count))
    if count == 1:
        return -0.85
    if count == 2:
        return -0.15
    if count == 3:
        return 0.15
    return min(1.0, 0.30 + 0.15 * (count - 4))


class OnlineSemanticSegmenter:
    """Embedding-based online semantic boundary detector for dialogue exchanges."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        config: Optional[OnlineSegmentationConfig] = None,
    ) -> None:
        self.embedding_client = embedding_client
        self.config = config or OnlineSegmentationConfig()
        self.surprise_history: Deque[float] = deque(
            maxlen=max(1, self.config.surprise_history_window),
        )
        self._pending_online_exchanges: List[OnlineSegmentExchange] = []
        self._pending_online_exchanges_embedding: List[np.ndarray] = []
        self._prepared_incoming_embedding: Optional[
            Tuple[OnlineSegmentExchange, np.ndarray]
        ] = None

    def append_pending_exchange(self, exchange: OnlineSegmentExchange) -> None:
        """Append an exchange to the segmenter's pending online buffer."""
        embedding: Optional[np.ndarray] = None
        if (
            self._prepared_incoming_embedding is not None
            and self._prepared_incoming_embedding[0] is exchange
        ):
            embedding = self._prepared_incoming_embedding[1]
        elif self._prepared_incoming_embedding is not None:
            self._prepared_incoming_embedding = None
        if embedding is None:
            embedding = self.embed_exchange(exchange).embedding
        self._pending_online_exchanges.append(exchange)
        self._pending_online_exchanges_embedding.append(embedding)
        self._prepared_incoming_embedding = None

    def pending_exchange_snapshot(self) -> List[OnlineSegmentExchange]:
        """Return a shallow copy of the current pending online exchanges."""
        return list(self._pending_online_exchanges)

    def has_pending_exchanges(self) -> bool:
        """Return whether the segmenter has exchanges waiting for storage."""
        return bool(self._pending_online_exchanges)

    def clear_pending_exchanges(self) -> None:
        """Clear exchanges after the corresponding memory task was queued."""
        self._pending_online_exchanges.clear()
        self._pending_online_exchanges_embedding.clear()

    def embed_exchange(self, exchange: Any) -> ActiveExchange:
        embedding = self.embedding_client.embed_text(self.exchange_text(exchange))
        vector = _as_embedding_vector(embedding)
        if vector is None:
            raise ValueError("Embedding provider returned an invalid vector")
        return ActiveExchange(exchange=exchange, embedding=vector)

    def should_finalize_pending_exchanges(
        self,
        incoming_exchange: Any,
    ) -> Tuple[bool, SegmentDecision]:
        active_exchanges = self._pending_online_exchanges
        time_gap_seconds = (
            self.exchange_time_gap_seconds(active_exchanges[-1], incoming_exchange)
            if active_exchanges
            else None
        )
        if not active_exchanges:
            decision = SegmentDecision(
                exchange_index=self.exchange_index(incoming_exchange),
                reason="start_segment",
                prospective_tokens=self.exchange_token_count(incoming_exchange),
                prospective_exchanges=1,
                time_gap_seconds=time_gap_seconds,
            )
            return False, decision
        existing_capacity_reason = self._pending_capacity_reason(active_exchanges)
        if existing_capacity_reason:
            return True, SegmentDecision(
                exchange_index=self.exchange_index(incoming_exchange),
                reason=existing_capacity_reason,
                prospective_tokens=sum(
                    self.exchange_token_count(item) for item in active_exchanges
                ),
                prospective_exchanges=len(active_exchanges),
                time_gap_seconds=time_gap_seconds,
            )
        if self.time_gap_exceeded(time_gap_seconds):
            return True, SegmentDecision(
                exchange_index=self.exchange_index(incoming_exchange),
                reason="time_gap",
                prospective_tokens=sum(
                    self.exchange_token_count(item) for item in active_exchanges
                ),
                prospective_exchanges=len(active_exchanges),
                time_gap_seconds=time_gap_seconds,
            )
        active = [
            ActiveExchange(exchange=exchange, embedding=embedding)
            for exchange, embedding in zip(
                active_exchanges,
                self._pending_online_exchanges_embedding,
            )
        ]
        incoming = self.embed_exchange(incoming_exchange)
        self._prepared_incoming_embedding = (
            incoming_exchange,
            incoming.embedding,
        )
        decision = self.score_boundary(active, incoming)
        decision.time_gap_seconds = time_gap_seconds
        if self.semantic_boundary_allowed(active, decision):
            decision.reason = "semantic_boundary"
            return True, decision
        decision.reason = "append"
        return False, decision

    def _pending_capacity_reason(
        self,
        exchanges: Sequence[Any],
    ) -> Optional[str]:
        """Return the pending turn or character hard-limit reason."""
        if not exchanges:
            return None
        if (
            self.config.max_pending_turns > 0
            and len(exchanges) >= self.config.max_pending_turns
        ):
            return "pending_turn_limit"
        token_count = sum(self.exchange_token_count(exchange) for exchange in exchanges)
        if (
            self.config.max_pending_tokens > 0
            and token_count >= self.config.max_pending_tokens
        ):
            return "pending_token_limit"
        return None

    def score_boundary(
        self,
        active: Sequence[ActiveExchange],
        incoming: ActiveExchange,
    ) -> SegmentDecision:
        active_embeddings = [item.embedding for item in active]
        active_centroid = _centroid(active_embeddings)
        recent_embedding = active[-1].embedding
        centroid_sim = _cal_embedding_cosine_similarity(incoming.embedding, active_centroid)
        recent_sim = _cal_embedding_cosine_similarity(incoming.embedding, recent_embedding)
        semantic_surprise = 1.0 - max(centroid_sim, recent_sim)

        robust_surprise = robust_surprise_signal(
            semantic_surprise,
            list(self.surprise_history),
            self.config.min_surprise_history,
        )
        absolute_surprise = absolute_surprise_signal(semantic_surprise)

        cohesion_before = _cohesion(active_embeddings)
        cohesion_after = _cohesion([*active_embeddings, incoming.embedding])
        cohesion_drop = max(0.0, cohesion_before - cohesion_after)
        prospective_tokens = sum(self.exchange_token_count(item.exchange) for item in active) + self.exchange_token_count(incoming.exchange)
        prospective_exchanges = len(active) + 1
        length_signal = length_pressure(prospective_tokens, self.config)
        turn_signal = turn_count_pressure(prospective_exchanges)
        score = (
            self.config.robust_surprise_weight * robust_surprise
            + self.config.absolute_surprise_weight * absolute_surprise
            + self.config.cohesion_drop_weight * cohesion_drop
            + self.config.length_weight * length_signal
            + self.config.turn_count_weight * turn_signal
        )
        cut_probability = _sigmoid(self.config.bias + score)
        self.surprise_history.append(float(semantic_surprise))

        return SegmentDecision(
            exchange_index=self.exchange_index(incoming.exchange),
            reason="score",
            cut_probability=cut_probability,
            score=score,
            semantic_surprise=semantic_surprise,
            robust_surprise=robust_surprise,
            absolute_surprise=absolute_surprise,
            cohesion_before=cohesion_before,
            cohesion_after=cohesion_after,
            cohesion_drop=cohesion_drop,
            length_signal=length_signal,
            turn_signal=turn_signal,
            centroid_similarity=centroid_sim,
            recent_similarity=recent_sim,
            prospective_tokens=prospective_tokens,
            prospective_exchanges=prospective_exchanges,
        )

    def semantic_boundary_allowed(
        self,
        active: Sequence[ActiveExchange],
        decision: SegmentDecision,
    ) -> bool:
        if (decision.cut_probability or 0.0) < self.config.threshold:
            return False
        min_exchanges = max(1, int(self.config.min_pending_turns))
        if len(active) >= min_exchanges:
            return True
        return (decision.cut_probability or 0.0) >= float(
            self.config.min_segment_override_probability,
        )

    def time_gap_exceeded(self, time_gap_seconds: Optional[float]) -> bool:
        return bool(
            self.config.max_time_gap_seconds >= 0
            and time_gap_seconds is not None
            and time_gap_seconds > self.config.max_time_gap_seconds
        )

    @staticmethod
    def exchange_text(exchange: Any) -> str:
        return str(getattr(exchange, "text", "") or "")

    @staticmethod
    def exchange_token_count(exchange: Any) -> int:
        value = getattr(exchange, "token_count", None)
        if value is not None:
            return max(1, int(value))
        return _estimate_interaction_token_count(OnlineSemanticSegmenter.exchange_text(exchange))

    @staticmethod
    def exchange_index(exchange: Any) -> int:
        return int(getattr(exchange, "index", 0) or 0)

    @staticmethod
    def exchange_timestamp(exchange: Any) -> str:
        return _to_timestamp_text(getattr(exchange, "timestamp", "")) or ""

    @classmethod
    def exchange_time_gap_seconds(
        cls,
        previous_exchange: Any,
        incoming_exchange: Any,
    ) -> Optional[float]:
        return cls.timestamp_gap_seconds(
            cls.exchange_timestamp(previous_exchange),
            cls.exchange_timestamp(incoming_exchange),
        )

    @staticmethod
    def timestamp_gap_seconds(
        previous_end: Any,
        current_start: Any,
    ) -> Optional[float]:
        """Return the gap between two timestamp values, if both are valid."""
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

class MemoryRuntime:
    """Normalize application input and batch it before episode storage."""

    def __init__(
        self,
        manager: MemoryNodeManager,
        *,
        memory_runtime_config: Optional[Dict[str, Any]] = None,
        embedding_config: Optional[Dict[str, Any]] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the runtime buffers and batching thresholds."""
        self._manager = manager
        self._logger = logger or logging.getLogger(__name__)
        if logger is not None:
            self._manager.set_logger(logger)
        config = dict(memory_runtime_config or {})
        self._prompt_language_mode = str(
            config.get("memory_prompt_language_mode")
            or "source"
        ).strip().lower()
        self._embedding_client = embedding_client or EmbeddingClient(
            dict(embedding_config or getattr(manager, "_embedding_cfg", {}) or {}),
        )
        if hasattr(self._manager, "set_embedding_client"):
            self._manager.set_embedding_client(self._embedding_client)
        interaction_segmentation_config = self._build_online_segmentation_config(config)
        self._interaction_segmenter = OnlineSemanticSegmenter(
            self._embedding_client,
            interaction_segmentation_config,
        )
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

    def _build_online_segmentation_config(
        self,
        config: Dict[str, Any],
    ) -> OnlineSegmentationConfig:
        """Build the assistant-wakeup segmentation config from runtime config.

        ``memory_runtime`` contains separate configurations for assistant
        wakeup and all-day recording.  The interaction segmenter currently
        consumes only ``assistant_wakeup_segmentation``; the all-day recording
        configuration remains reserved for the transcript pipeline.
        """
        segmentation_config = config.get("assistant_wakeup_segmentation")
        if not isinstance(segmentation_config, dict):
            segmentation_config = {}
        max_gap = segmentation_config.get("max_time_gap_seconds")
        return OnlineSegmentationConfig(
            threshold=self._config_float(segmentation_config, "threshold", 0.60),
            bias=self._config_float(segmentation_config, "bias", -1.10),
            surprise_history_window=max(
                1,
                self._config_int(segmentation_config, "surprise_history_window", 64),
            ),
            min_surprise_history=max(
                0,
                self._config_int(segmentation_config, "min_surprise_history", 5),
            ),
            robust_surprise_weight=self._config_float(
                segmentation_config,
                "robust_surprise_weight",
                0.8,
            ),
            absolute_surprise_weight=self._config_float(
                segmentation_config,
                "absolute_surprise_weight",
                0.8,
            ),
            cohesion_drop_weight=self._config_float(
                segmentation_config,
                "cohesion_drop_weight",
                1.0,
            ),
            length_weight=self._config_float(segmentation_config, "length_weight", 0.40),
            turn_count_weight=self._config_float(
                segmentation_config,
                "turn_count_weight",
                0.40,
            ),
            max_pending_turns=max(
                1,
                self._config_int(
                    segmentation_config,
                    "max_pending_interaction_turns",
                    5,
                ),
            ),
            max_pending_tokens=max(
                1,
                self._config_int(
                    segmentation_config,
                    "max_pending_interaction_tokens",
                    500,
                ),
            ),
            min_pending_tokens=max(
                1,
                self._config_int(segmentation_config, "min_pending_interaction_tokens", 100),
            ),
            min_pending_turns=max(
                1,
                self._config_int(segmentation_config, "min_pending_interaction_turns", 2),
            ),
            min_segment_override_probability=self._config_float(
                segmentation_config,
                "min_segment_override_probability",
                0.90,
            ),
            max_time_gap_seconds=float(-1.0 if max_gap in (None, "") else max_gap),
        )

    @staticmethod
    def _config_int(config: Dict[str, Any], key: str, default: int) -> int:
        value = config.get(key)
        if value in (None, ""):
            return int(default)
        return int(value)

    @staticmethod
    def _config_float(config: Dict[str, Any], key: str, default: float) -> float:
        value = config.get(key)
        if value in (None, ""):
            return float(default)
        return float(value)

    @staticmethod
    def _interaction_turn_to_online_exchange(
        turn: Dict[str, Any],
        index: int,
    ) -> OnlineSegmentExchange:
        text = _exchange_text_from_turn(turn)
        timestamp = _to_timestamp_text(turn.get("turn_timestamp")) or ""
        return OnlineSegmentExchange(
            index=index,
            text=text,
            token_count=_estimate_interaction_token_count(text),
            timestamp=timestamp,
            raw=dict(turn),
        )

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

        queued = False
        reason = "threshold_not_reached"
        pending_exchanges = self._interaction_segmenter.pending_exchange_snapshot()
        incoming_exchange = self._interaction_turn_to_online_exchange(
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
        if not self._manager.enabled:
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

        report = self._manager.submit_memory_reflect_task(*args, **kwargs) or {}
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
        return self._manager.process_memory_recall_immediately(*args, **kwargs)

    def flush_task_queue(self, timeout: Optional[float] = None) -> bool:
        """Wait for queued store and reflection tasks at an explicit boundary."""
        if self._interaction_segmenter.has_pending_exchanges():
            self._flush_pending_interaction_turns()
        if self._pending_transcript_segments:
            self._flush_pending_transcript_segments()
        return self._manager.flush_task_queue(timeout=timeout)

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
        return self._manager.submit_memory_store_task(
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
        return self._manager.submit_memory_store_task(
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
