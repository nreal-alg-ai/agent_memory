from dataclasses import dataclass
from datetime import datetime
from collections import deque
import re
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .embedding_client import EmbeddingClient
from .utils import _as_embedding_vector, _cal_embedding_cosine_similarity, _sigmoid, _centroid, _cohesion
from .memory_manager import (
    _compact_whitespace,
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

def build_online_segmentation_config(
    runtime_config: Dict[str, Any],
) -> OnlineSegmentationConfig:
    """Build the assistant-wakeup segmentation runtime_config from runtime config.

    ``memory_runtime`` contains separate configurations for assistant
    wakeup and all-day recording.  The interaction segmenter currently
    consumes only ``assistant_wakeup_segmentation``; the all-day recording
    configuration remains reserved for the transcript pipeline.
    """
    segmentation_config = runtime_config.get("assistant_wakeup_segmentation")
    if not isinstance(segmentation_config, dict):
        segmentation_config = {}
    max_gap = segmentation_config.get("max_time_gap_seconds")
    return OnlineSegmentationConfig(
        threshold=float(segmentation_config.get("threshold", 0.60)),
        bias=float(segmentation_config.get("bias", -1.10)),
        surprise_history_window=max(
            1,
            int(segmentation_config.get("surprise_history_window", 64)),
        ),
        min_surprise_history=max(
            0,
            int(segmentation_config.get("min_surprise_history", 5)),
        ),
        robust_surprise_weight=float(
            segmentation_config.get("robust_surprise_weight", 0.8),
        ),
        absolute_surprise_weight=float(
            segmentation_config.get("absolute_surprise_weight", 0.8),
        ),
        cohesion_drop_weight=float(
            segmentation_config.get("cohesion_drop_weight", 1.0),
        ),
        length_weight=float(segmentation_config.get("length_weight", 0.40)),
        turn_count_weight=float(
            segmentation_config.get("turn_count_weight", 0.40),
        ),
        max_pending_turns=max(
            1,
            int(segmentation_config.get("max_pending_interaction_turns", 5)),
        ),
        max_pending_tokens=max(
            1,
            int(segmentation_config.get("max_pending_interaction_tokens", 500)),
        ),
        min_pending_tokens=max(
            1,
            int(segmentation_config.get("min_pending_interaction_tokens", 100)),
        ),
        min_pending_turns=max(
            1,
            int(segmentation_config.get("min_pending_interaction_turns", 2)),
        ),
        min_segment_override_probability=float(
            segmentation_config.get("min_segment_override_probability", 0.90),
        ),
        max_time_gap_seconds=float(-1.0 if max_gap in (None, "") else max_gap),
    )


def convert_interaction_turn_to_online_exchange(
    turn: Dict[str, Any],
    index: int,
) -> OnlineSegmentExchange:
    """Normalize one interaction turn into the segmenter's exchange format."""
    text = _exchange_text_from_turn(turn)
    timestamp = _to_timestamp_text(turn.get("turn_timestamp")) or ""
    return OnlineSegmentExchange(
        index=index,
        text=text,
        token_count=_estimate_interaction_token_count(text),
        timestamp=timestamp,
        raw=dict(turn),
    )


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
