#!/usr/bin/env python3
"""Test online semantic segmentation on CLongEval conversations.

This script implements the Online Semantic Segmentation method described in
LycheeMemory V2. It reads CLongEval JSONL records, parses dated user/assistant
conversation pairs, embeds each exchange, and decides semantic segment
boundaries with surprise, cohesion-drop, length-pressure, and turn-count
signals. It is intentionally standalone and does not modify the existing
CLongEval memory runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (SRC_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from memory.config import split_memory_config
from memory.embedding_client import EmbeddingClient


DEFAULT_INPUT = (
    REPO_ROOT
    / "test_data"
    / "user_dialogue"
    / "CLongEval"
    / "1-2_long_conversation_memory"
    / "small.jsonl"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tmp" / "online_segmentation"

DATE_HEADER_RE = re.compile(
    r"以下是(?P<date>\d{4}年\d{1,2}月\d{1,2}日)的对话记录\s*[:：]?"
)
ROLE_LINE_RE = re.compile(r"^[“”]?\s*(用户|AI|助手)\s*[:：]\s*(.*)$")
CONTEXT_END_MARKERS = ("请记住以上全部对话记录", "问题：", "问题:")
TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_.$'-]+|[^\s]")


@dataclass
class Exchange:
    index: int
    day_index: int
    pair_index: int
    date_text: str
    user: str
    assistant: str
    text: str
    token_count: int


@dataclass
class ActiveExchange:
    exchange: Exchange
    embedding: np.ndarray


@dataclass
class SegmentDecision:
    exchange_index: int
    reason: str
    cut_probability: Optional[float] = None
    score: Optional[float] = None
    semantic_surprise: Optional[float] = None
    robust_surprise: Optional[float] = None
    cohesion_before: Optional[float] = None
    cohesion_after: Optional[float] = None
    cohesion_drop: Optional[float] = None
    length_signal: Optional[float] = None
    turn_signal: Optional[float] = None
    centroid_similarity: Optional[float] = None
    recent_similarity: Optional[float] = None
    prospective_tokens: Optional[int] = None
    prospective_exchanges: Optional[int] = None


@dataclass
class FinalizedSegment:
    segment_index: int
    reason: str
    exchanges: List[ActiveExchange]
    decision: SegmentDecision


@dataclass
class SegmentationConfig:
    threshold: float = 0.60
    bias: float = -1.10
    surprise_history_window: int = 16
    min_surprise_history: int = 5
    robust_surprise_weight: float = 1.0
    robust_surprise_scale_floor: float = 0.05
    cohesion_drop_weight: float = 1.0
    length_weight: float = 0.40
    turn_count_weight: float = 0.40
    min_chunk_tokens: int = 300
    target_chunk_tokens: int = 600
    max_chunk_tokens: int = 900
    max_exchanges: int = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run LycheeMemory-style online semantic segmentation on CLongEval "
            "conversation contexts."
        )
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all records.")
    parser.add_argument(
        "--question-id",
        action="append",
        help="Only use the specified CLongEval record id. Can be passed multiple times.",
    )
    parser.add_argument(
        "--context-limit",
        type=int,
        default=0,
        help="Maximum number of unique context groups to segment. 0 means all groups.",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Do not include raw exchange text in the JSON output.",
    )
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--bias", type=float, default=-1.10)
    parser.add_argument("--surprise-history-window", type=int, default=16)
    parser.add_argument("--min-surprise-history", type=int, default=5)
    parser.add_argument("--robust-surprise-weight", type=float, default=1.0)
    parser.add_argument("--robust-surprise-scale-floor", type=float, default=0.05)
    parser.add_argument("--cohesion-drop-weight", type=float, default=1.0)
    parser.add_argument("--length-weight", type=float, default=0.40)
    parser.add_argument("--turn-count-weight", type=float, default=0.40)
    parser.add_argument("--min-chunk-tokens", type=int, default=300)
    parser.add_argument("--target-chunk-tokens", type=int, default=600)
    parser.add_argument("--max-chunk-tokens", type=int, default=900)
    parser.add_argument("--max-exchanges", type=int, default=10)
    parser.add_argument(
        "--embedding-provider",
        help="Override memory_manager.embedding.provider, for example local_hash.",
    )
    parser.add_argument("--embedding-model", help="Override the embedding model name.")
    parser.add_argument("--embedding-base-url", help="Override the embedding base URL.")
    parser.add_argument("--embedding-api-key", help="Override the embedding API key.")
    parser.add_argument("--embedding-api-key-env", help="Override the embedding API key env var.")
    parser.add_argument("--embedding-dimensions", type=int, help="Override embedding dimensions.")
    parser.add_argument("--embedding-timeout", type=int, help="Override embedding timeout seconds.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_project_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to read the project config") from exc

    config_path = path.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return loaded


def load_records(path: Path) -> List[Dict[str, Any]]:
    input_path = path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"CLongEval input does not exist: {input_path}")

    records: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {input_path}:{line_number}: {exc}") from exc
            if not isinstance(item, dict):
                continue
            context = str(item.get("context") or "").strip()
            query = str(item.get("query") or "").strip()
            if not context or not query:
                logging.warning("Skipping record without context/query at line %s", line_number)
                continue
            records.append({
                **item,
                "context": context,
                "query": query,
                "answer": str(item.get("answer") or "").strip(),
                "id": str(item.get("id") or f"line_{line_number}"),
                "_source_line": line_number,
            })
    return records


def filter_records(
    records: Sequence[Dict[str, Any]],
    *,
    question_ids: Optional[Sequence[str]],
    start: int,
    limit: int,
) -> List[Dict[str, Any]]:
    selected = list(records)
    if question_ids:
        wanted = {str(value).strip() for value in question_ids if str(value).strip()}
        selected = [item for item in selected if str(item.get("id")) in wanted]
    if start:
        selected = selected[max(0, int(start)) :]
    if limit:
        selected = selected[: max(0, int(limit))]
    return selected


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y年%m月%d日")


def _flush_message(messages: List[Tuple[str, str]], role: Optional[str], parts: List[str]) -> None:
    text = "\n".join(part for part in parts if part).strip().strip("“”")
    if text and role:
        messages.append((role, text))


def parse_context_days(context: str) -> List[Dict[str, Any]]:
    """Parse dated 用户/AI blocks from a CLongEval context string."""
    headers = list(DATE_HEADER_RE.finditer(context))
    if not headers:
        raise ValueError("Context does not contain a dated conversation header")

    days: List[Dict[str, Any]] = []
    for index, header in enumerate(headers):
        section_end = headers[index + 1].start() if index + 1 < len(headers) else len(context)
        section = context[header.end() : section_end]
        for marker in CONTEXT_END_MARKERS:
            marker_index = section.find(marker)
            if marker_index >= 0:
                section = section[:marker_index]
                break
        section = section.strip().strip("“”")

        messages: List[Tuple[str, str]] = []
        current_role: Optional[str] = None
        current_parts: List[str] = []
        for raw_line in section.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = ROLE_LINE_RE.match(line)
            if match:
                _flush_message(messages, current_role, current_parts)
                current_role = "user" if match.group(1) == "用户" else "assistant"
                current_parts = [match.group(2).strip()]
            elif current_role:
                current_parts.append(line.strip("“”"))
        _flush_message(messages, current_role, current_parts)

        pairs: List[Tuple[str, str]] = []
        pending_user: Optional[str] = None
        for role, text in messages:
            if role == "user":
                pending_user = text if not pending_user else f"{pending_user}\n{text}"
            elif pending_user:
                pairs.append((pending_user, text))
                pending_user = None

        if not pairs:
            logging.warning("No user/assistant pairs parsed for %s", header.group("date"))
            continue
        days.append({
            "date_text": header.group("date"),
            "date": parse_date(header.group("date")),
            "pairs": pairs,
        })

    if not days:
        raise ValueError("Context did not contain any complete user/assistant pairs")
    return days


def group_records_by_context(
    records: Sequence[Dict[str, Any]],
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    grouped: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for record in records:
        grouped.setdefault(str(record["context"]), []).append(record)
    return list(grouped.items())


def context_group_id(context: str, group_index: int) -> str:
    digest = hashlib.sha1(context.encode("utf-8")).hexdigest()[:12]
    return f"context_{group_index:04d}_{digest}"


def estimate_token_count(text: str) -> int:
    """Lightweight token pressure proxy for Chinese/English benchmark text."""
    return max(1, len(TOKEN_RE.findall(text or "")))


def exchange_text(user: str, assistant: str) -> str:
    return f"用户：{user.strip()}\n助手：{assistant.strip()}".strip()


def build_exchanges(days: Sequence[Dict[str, Any]]) -> List[Exchange]:
    exchanges: List[Exchange] = []
    for day_index, day in enumerate(days, 1):
        for pair_index, (user, assistant) in enumerate(day["pairs"], 1):
            text = exchange_text(user, assistant)
            exchanges.append(Exchange(
                index=len(exchanges) + 1,
                day_index=day_index,
                pair_index=pair_index,
                date_text=str(day["date_text"]),
                user=user,
                assistant=assistant,
                text=text,
                token_count=estimate_token_count(text),
            ))
    return exchanges


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def as_vector(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim > 1:
        vector = vector.reshape(-1)
    if vector.size == 0:
        raise ValueError("Embedding provider returned an empty vector")
    return vector


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    keep = min(left.size, right.size)
    if keep <= 0:
        return 0.0
    a = left[:keep].reshape(-1)
    b = right[:keep].reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def centroid(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("Cannot compute centroid for an empty segment")
    return np.mean(np.vstack([as_vector(item) for item in embeddings]), axis=0)


def cohesion(embeddings: Sequence[np.ndarray]) -> float:
    if not embeddings:
        return 0.0
    center = centroid(embeddings)
    return float(np.mean([cosine_similarity(item, center) for item in embeddings]))


def clipped(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def robust_surprise_signal(
    surprise: float,
    history: Sequence[float],
    min_history: int,
    scale_floor: float,
) -> float:
    required_history = max(1, int(min_history))
    if len(history) < required_history:
        return 0.0
    values = np.asarray(list(history), dtype=np.float32)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(float(scale_floor), 1.4826 * mad)
    return clipped((surprise - median) / scale, -2.0, 2.0)


def linear(value: float, start: float, end: float, low: float, high: float) -> float:
    if end <= start:
        return high
    ratio = clipped((value - start) / (end - start), 0.0, 1.0)
    return low + ratio * (high - low)


def length_pressure(token_count: int, config: SegmentationConfig) -> float:
    min_tokens = max(1, int(config.min_chunk_tokens))
    target_tokens = max(min_tokens + 1, int(config.target_chunk_tokens))
    max_tokens = max(target_tokens + 1, int(config.max_chunk_tokens))
    length = float(max(0, token_count))
    early_boundary = 0.7 * min_tokens

    if length < early_boundary:
        return -1.30
    if length < min_tokens:
        return linear(length, early_boundary, min_tokens, -0.80, 0.0)
    if length < target_tokens:
        return linear(length, min_tokens, target_tokens, 0.45, 1.90)
    if length < max_tokens:
        return linear(length, target_tokens, max_tokens, 1.90, 2.80)
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


def round_float(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


class OnlineSemanticSegmenter:
    def __init__(self, embedding_client: EmbeddingClient, config: SegmentationConfig) -> None:
        self.embedding_client = embedding_client
        self.config = config
        self.surprise_history: Deque[float] = deque(
            maxlen=max(1, config.surprise_history_window)
        )

    def embed_exchange(self, exchange: Exchange) -> ActiveExchange:
        embedding = self.embedding_client.embed_text(exchange.text)
        return ActiveExchange(exchange=exchange, embedding=as_vector(embedding))

    def segment(self, exchanges: Sequence[Exchange]) -> List[FinalizedSegment]:
        active: List[ActiveExchange] = []
        finalized: List[FinalizedSegment] = []

        for exchange in exchanges:
            incoming = self.embed_exchange(exchange)
            if not active:
                active.append(incoming)
                startup_decision = SegmentDecision(
                    exchange_index=exchange.index,
                    reason="start_segment",
                    prospective_tokens=exchange.token_count,
                    prospective_exchanges=1,
                )
                if self._should_finalize_after_append(active):
                    finalized.append(self._finalize(
                        finalized,
                        active,
                        self._post_append_reason(active),
                        startup_decision,
                    ))
                    active = []
                continue

            decision = self._score_boundary(active, incoming)

            if decision.prospective_tokens and decision.prospective_tokens > self.config.max_chunk_tokens:
                finalized.append(self._finalize(finalized, active, "capacity_limit", decision))
                active = [incoming]
                if self._should_finalize_after_append(active):
                    finalized.append(self._finalize(
                        finalized,
                        active,
                        self._post_append_reason(active),
                        SegmentDecision(
                            exchange_index=exchange.index,
                            reason="post_capacity_new_segment",
                            prospective_tokens=exchange.token_count,
                            prospective_exchanges=1,
                        ),
                    ))
                    active = []
            elif (decision.cut_probability or 0.0) >= self.config.threshold:
                finalized.append(self._finalize(finalized, active, "semantic_boundary", decision))
                active = [incoming]
                if self._should_finalize_after_append(active):
                    finalized.append(self._finalize(
                        finalized,
                        active,
                        self._post_append_reason(active),
                        SegmentDecision(
                            exchange_index=exchange.index,
                            reason="post_boundary_new_segment",
                            prospective_tokens=exchange.token_count,
                            prospective_exchanges=1,
                        ),
                    ))
                    active = []
            else:
                active.append(incoming)
                if self._should_finalize_after_append(active):
                    finalized.append(self._finalize(
                        finalized,
                        active,
                        self._post_append_reason(active),
                        decision,
                    ))
                    active = []

        if active:
            finalized.append(self._finalize(
                finalized,
                active,
                "session_flush",
                SegmentDecision(
                    exchange_index=active[-1].exchange.index,
                    reason="session_flush",
                    prospective_tokens=sum(item.exchange.token_count for item in active),
                    prospective_exchanges=len(active),
                ),
            ))
        return finalized

    def _score_boundary(
        self,
        active: Sequence[ActiveExchange],
        incoming: ActiveExchange,
    ) -> SegmentDecision:
        active_embeddings = [item.embedding for item in active]
        active_centroid = centroid(active_embeddings)
        recent_embedding = active[-1].embedding
        centroid_sim = cosine_similarity(incoming.embedding, active_centroid)
        recent_sim = cosine_similarity(incoming.embedding, recent_embedding)
        semantic_surprise = 1.0 - max(centroid_sim, recent_sim)

        robust_surprise = robust_surprise_signal(
            semantic_surprise,
            list(self.surprise_history),
            self.config.min_surprise_history,
            self.config.robust_surprise_scale_floor,
        )

        cohesion_before = cohesion(active_embeddings)
        cohesion_after = cohesion([*active_embeddings, incoming.embedding])
        cohesion_drop = max(0.0, cohesion_before - cohesion_after)
        prospective_tokens = sum(item.exchange.token_count for item in active) + incoming.exchange.token_count
        prospective_exchanges = len(active) + 1
        length_signal = length_pressure(prospective_tokens, self.config)
        turn_signal = turn_count_pressure(prospective_exchanges)
        score = (
            self.config.robust_surprise_weight * robust_surprise
            + self.config.cohesion_drop_weight * cohesion_drop
            + self.config.length_weight * length_signal
            + self.config.turn_count_weight * turn_signal
        )
        cut_probability = sigmoid(self.config.bias + score)
        self.surprise_history.append(float(semantic_surprise))

        return SegmentDecision(
            exchange_index=incoming.exchange.index,
            reason="score",
            cut_probability=cut_probability,
            score=score,
            semantic_surprise=semantic_surprise,
            robust_surprise=robust_surprise,
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

    def _should_finalize_after_append(self, active: Sequence[ActiveExchange]) -> bool:
        if not active:
            return False
        token_count = sum(item.exchange.token_count for item in active)
        if token_count >= self.config.max_chunk_tokens:
            return True
        if len(active) >= self.config.max_exchanges:
            return True
        return False

    def _post_append_reason(self, active: Sequence[ActiveExchange]) -> str:
        token_count = sum(item.exchange.token_count for item in active)
        if token_count >= self.config.max_chunk_tokens:
            return "capacity_limit"
        if len(active) >= self.config.max_exchanges:
            return "exchange_count_limit"
        return "segment_complete"

    def _finalize(
        self,
        finalized: Sequence[FinalizedSegment],
        active: Sequence[ActiveExchange],
        reason: str,
        decision: SegmentDecision,
    ) -> FinalizedSegment:
        decision.reason = reason
        return FinalizedSegment(
            segment_index=len(finalized) + 1,
            reason=reason,
            exchanges=list(active),
            decision=decision,
        )


def build_segmentation_config(args: argparse.Namespace) -> SegmentationConfig:
    if args.min_chunk_tokens <= 0:
        raise ValueError("--min-chunk-tokens must be positive")
    if args.target_chunk_tokens < args.min_chunk_tokens:
        raise ValueError("--target-chunk-tokens must be >= --min-chunk-tokens")
    if args.max_chunk_tokens < args.target_chunk_tokens:
        raise ValueError("--max-chunk-tokens must be >= --target-chunk-tokens")
    if args.max_exchanges <= 0:
        raise ValueError("--max-exchanges must be positive")
    if args.robust_surprise_scale_floor <= 0:
        raise ValueError("--robust-surprise-scale-floor must be positive")
    return SegmentationConfig(
        threshold=float(args.threshold),
        bias=float(args.bias),
        surprise_history_window=max(1, int(args.surprise_history_window)),
        min_surprise_history=max(0, int(args.min_surprise_history)),
        robust_surprise_weight=float(args.robust_surprise_weight),
        robust_surprise_scale_floor=float(args.robust_surprise_scale_floor),
        cohesion_drop_weight=float(args.cohesion_drop_weight),
        length_weight=float(args.length_weight),
        turn_count_weight=float(args.turn_count_weight),
        min_chunk_tokens=int(args.min_chunk_tokens),
        target_chunk_tokens=int(args.target_chunk_tokens),
        max_chunk_tokens=int(args.max_chunk_tokens),
        max_exchanges=int(args.max_exchanges),
    )


def build_embedding_config(args: argparse.Namespace) -> Dict[str, Any]:
    _runtime_config, _manager_config, _llm_config, embedding_config = split_memory_config(
        load_project_config(args.config)
    )
    overrides = {
        "provider": args.embedding_provider,
        "model": args.embedding_model,
        "base_url": args.embedding_base_url,
        "api_key": args.embedding_api_key,
        "api_key_env": args.embedding_api_key_env,
        "dimensions": args.embedding_dimensions,
        "timeout": args.embedding_timeout,
    }
    for key, value in overrides.items():
        if value is not None:
            embedding_config[key] = value
    return embedding_config


def serialize_decision(decision: SegmentDecision) -> Dict[str, Any]:
    return {
        "exchange_index": decision.exchange_index,
        "reason": decision.reason,
        "cut_probability": round_float(decision.cut_probability),
        "score": round_float(decision.score),
        "semantic_surprise": round_float(decision.semantic_surprise),
        "robust_surprise": round_float(decision.robust_surprise),
        "cohesion_before": round_float(decision.cohesion_before),
        "cohesion_after": round_float(decision.cohesion_after),
        "cohesion_drop": round_float(decision.cohesion_drop),
        "length_signal": round_float(decision.length_signal),
        "turn_signal": round_float(decision.turn_signal),
        "centroid_similarity": round_float(decision.centroid_similarity),
        "recent_similarity": round_float(decision.recent_similarity),
        "prospective_tokens": decision.prospective_tokens,
        "prospective_exchanges": decision.prospective_exchanges,
    }


def serialize_exchange(item: ActiveExchange, include_text: bool) -> Dict[str, Any]:
    data = {
        "index": item.exchange.index,
        "day_index": item.exchange.day_index,
        "pair_index": item.exchange.pair_index,
        "date": item.exchange.date_text,
        "token_count": item.exchange.token_count,
    }
    if include_text:
        data.update({
            "user": item.exchange.user,
            "assistant": item.exchange.assistant,
            "text": item.exchange.text,
        })
    return data


def serialize_segment(segment: FinalizedSegment, include_text: bool) -> Dict[str, Any]:
    exchanges = [serialize_exchange(item, include_text) for item in segment.exchanges]
    token_count = sum(item["token_count"] for item in exchanges)
    return {
        "segment_index": segment.segment_index,
        "reason": segment.reason,
        "exchange_count": len(exchanges),
        "token_count": token_count,
        "exchange_start": exchanges[0]["index"] if exchanges else None,
        "exchange_end": exchanges[-1]["index"] if exchanges else None,
        "date_start": exchanges[0]["date"] if exchanges else None,
        "date_end": exchanges[-1]["date"] if exchanges else None,
        "boundary_decision": serialize_decision(segment.decision),
        "exchanges": exchanges,
    }


def summarize_segments(segments: Sequence[FinalizedSegment]) -> Dict[str, Any]:
    segment_count = len(segments)
    exchange_counts = [len(segment.exchanges) for segment in segments]
    token_counts = [
        sum(item.exchange.token_count for item in segment.exchanges)
        for segment in segments
    ]
    reasons = Counter(segment.reason for segment in segments)
    return {
        "segment_count": segment_count,
        "exchange_count": sum(exchange_counts),
        "avg_exchanges_per_segment": (
            round(sum(exchange_counts) / segment_count, 4) if segment_count else 0.0
        ),
        "avg_tokens_per_segment": (
            round(sum(token_counts) / segment_count, 4) if segment_count else 0.0
        ),
        "min_tokens_per_segment": min(token_counts) if token_counts else 0,
        "max_tokens_per_segment": max(token_counts) if token_counts else 0,
        "finalize_reasons": dict(sorted(reasons.items())),
    }


def segment_context_group(
    *,
    group_index: int,
    context: str,
    records: Sequence[Dict[str, Any]],
    embedding_client: EmbeddingClient,
    segmentation_config: SegmentationConfig,
    include_text: bool,
) -> Dict[str, Any]:
    group_id = context_group_id(context, group_index)
    days = parse_context_days(context)
    exchanges = build_exchanges(days)
    segmenter = OnlineSemanticSegmenter(embedding_client, segmentation_config)
    segments = segmenter.segment(exchanges)
    logging.info(
        "Segmented %s: records=%s days=%s exchanges=%s segments=%s",
        group_id,
        len(records),
        len(days),
        len(exchanges),
        len(segments),
    )
    return {
        "context_group_id": group_id,
        "record_ids": [str(item.get("id")) for item in records],
        "source_lines": [item.get("_source_line") for item in records],
        "record_count": len(records),
        "days": len(days),
        "context_chars": len(context),
        "summary": summarize_segments(segments),
        "segments": [serialize_segment(segment, include_text) for segment in segments],
    }


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    segmentation_config = build_segmentation_config(args)
    embedding_config = build_embedding_config(args)
    logging.info(
        "Embedding config: provider=%s model=%s base_url=%s dimensions=%s api_key_present=%s",
        embedding_config.get("provider"),
        embedding_config.get("model"),
        embedding_config.get("base_url"),
        embedding_config.get("dimensions"),
        bool(str(embedding_config.get("api_key") or "").strip())
        or bool(os.environ.get(str(embedding_config.get("api_key_env") or "").strip(), "")),
    )
    embedding_client = EmbeddingClient(embedding_config)

    records = filter_records(
        load_records(args.input),
        question_ids=args.question_id,
        start=args.start,
        limit=args.limit,
    )
    if not records:
        raise RuntimeError("No CLongEval records selected")

    groups = group_records_by_context(records)
    if args.context_limit:
        groups = groups[: max(0, int(args.context_limit))]
    if not groups:
        raise RuntimeError("No CLongEval context groups selected")

    include_text = not bool(args.no_text)
    group_outputs: List[Dict[str, Any]] = []
    total_segments = 0
    total_exchanges = 0
    reason_counts: Counter[str] = Counter()
    for group_index, (context, group_records) in enumerate(groups, 1):
        group_output = segment_context_group(
            group_index=group_index,
            context=context,
            records=group_records,
            embedding_client=embedding_client,
            segmentation_config=segmentation_config,
            include_text=include_text,
        )
        group_outputs.append(group_output)
        summary = group_output["summary"]
        total_segments += int(summary["segment_count"])
        total_exchanges += int(summary["exchange_count"])
        reason_counts.update(summary["finalize_reasons"])

    output_dir = args.output_dir.expanduser().resolve()
    output_file = (
        args.output_file.expanduser().resolve()
        if args.output_file
        else output_dir / f"clongeval_online_segmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    result = {
        "input": str(args.input.expanduser().resolve()),
        "config": str(args.config.expanduser().resolve()),
        "output_file": str(output_file),
        "records_processed": len(records),
        "context_groups": len(groups),
        "segmentation_config": segmentation_config.__dict__,
        "embedding_config": {
            key: value
            for key, value in embedding_config.items()
            if key not in {"api_key"}
        },
        "summary": {
            "segment_count": total_segments,
            "exchange_count": total_exchanges,
            "avg_exchanges_per_segment": (
                round(total_exchanges / total_segments, 4) if total_segments else 0.0
            ),
            "finalize_reasons": dict(sorted(reason_counts.items())),
        },
        "groups": group_outputs,
    }
    write_json(output_file, result)

    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote segmentation output to {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
