#!/usr/bin/env python3
"""Store Eval_Ali TextGrid transcripts through the unified memory manager."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memory.memory_database import SessionDB
from memory.memory_manager import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL, MemoryNodeManager


DEFAULT_TEXTGRID = (
    REPO_ROOT
    / "test_data/ambient_transcript/Eval_Ali/Eval_Ali_far/textgrid_dir/R8001_M8004.TextGrid"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp/ambient_transcript/Ali_eval"

_ENV_REF_RE = re.compile(r"\${([A-Za-z_][A-Za-z0-9_]*)}")
_NAME_RE = re.compile(r'name\s*=\s*"(?P<name>.*)"')
_XMIN_RE = re.compile(r"xmin\s*=\s*(?P<value>[-+]?\d+(?:\.\d+)?)")
_XMAX_RE = re.compile(r"xmax\s*=\s*(?P<value>[-+]?\d+(?:\.\d+)?)")
_TEXT_RE = re.compile(r'text\s*=\s*"(?P<text>(?:[^"]|"")*)"')


@dataclass
class TextGridSegment:
    speaker: str
    start_s: float
    end_s: float
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read Eval_Ali TextGrid intervals in time order and store them as "
            "ambient transcript memory episodes."
        )
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--textgrid", type=Path, default=DEFAULT_TEXTGRID)
    parser.add_argument("--output-root-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--db-name", default="memory.db")
    parser.add_argument("--session-start", default="", help="ISO timestamp used as audio time zero.")
    parser.add_argument("--source-type", default="allday_recording")
    parser.add_argument("--episode-type", default="ambient_transcript")
    parser.add_argument("--max-segments-per-episode", type=int, default=0)
    parser.add_argument("--max-chars-per-episode", type=int, default=0)
    parser.add_argument("--max-gap-seconds", type=float, default=-1.0)
    parser.add_argument("--enable-reflect", action="store_true", help="Run manager.reflect() after storing episodes.")
    parser.add_argument("--disable-llm", action="store_true", help="Use heuristic fallback extraction only.")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    memory_config = dict(config.get("memory") or {})
    embedding_config = dict(config.get("embedding") or {})
    if args.disable_llm:
        memory_config["llm_api_key"] = ""
    max_segments = int(
        args.max_segments_per_episode
        or memory_config.get("transcript_episode_max_segments")
        or 80
    )
    max_chars = int(
        args.max_chars_per_episode
        or memory_config.get("transcript_episode_max_chars")
        or 12000
    )
    max_gap_s = (
        float(args.max_gap_seconds)
        if args.max_gap_seconds >= 0
        else float(memory_config.get("transcript_episode_max_gap_seconds") or 60.0)
    )

    textgrid_path = args.textgrid.expanduser().resolve()
    if not textgrid_path.exists():
        raise FileNotFoundError(f"TextGrid file not found: {textgrid_path}")
    segments = load_textgrid_segments(textgrid_path)
    if not segments:
        raise RuntimeError(f"No non-empty intervals found in TextGrid: {textgrid_path}")

    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir / "run.log", args.log_level)
    db_path = output_dir / args.db_name
    transcript_path = output_dir / "transcript.txt"
    report_path = output_dir / "report.json"

    session_start = parse_session_start(args.session_start)
    memory_segments = [
        textgrid_segment_to_memory_segment(item, session_start=session_start, index=index)
        for index, item in enumerate(segments, 1)
    ]
    write_transcript_txt(transcript_path, memory_segments)
    chunks = list(chunk_memory_segments(
        memory_segments,
        max_segments=max_segments,
        max_chars=max_chars,
        max_gap_s=max_gap_s,
    ))

    llm_model = args.llm_model or memory_config.get("llm_name") or DEFAULT_LLM_MODEL
    llm_base_url = args.llm_base_url or memory_config.get("llm_base_url") or DEFAULT_LLM_BASE_URL
    llm_api_key = args.llm_api_key or memory_config.get("llm_api_key") or ""
    llm_api_key = expand_env_refs(llm_api_key)

    db = SessionDB(db_path)
    manager = MemoryNodeManager(
        db,
        embedding_config=embedding_config,
        memory_config=memory_config,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
    )

    stored_count = 0
    for chunk_index, chunk in enumerate(chunks, 1):
        source_ref = f"{textgrid_path.name}#chunk_{chunk_index:03d}"
        ok = manager.store_transcript_segments(
            chunk,
            source_type=args.source_type,
            episode_type=args.episode_type,
            source_ref=source_ref,
            tags=["Eval_Ali", textgrid_path.stem],
        )
        stored_count += int(ok)
        logging.info(
            "Stored transcript chunk %s/%s segments=%s chars=%s ok=%s",
            chunk_index,
            len(chunks),
            len(chunk),
            sum(len(item.get("text", "")) for item in chunk),
            ok,
        )

    reflect_result: Dict[str, Any] = {}
    if args.enable_reflect:
        reflect_timestamp = (
            memory_segments[-1].get("ended_at")
            if memory_segments
            else session_start.isoformat()
        )
        reflect_result = manager.reflect(reflect_timestamp=reflect_timestamp)
        logging.info("Reflect result: %s", reflect_result)

    report = {
        "textgrid": str(textgrid_path),
        "db_path": str(db_path),
        "transcript_path": str(transcript_path),
        "segment_count": len(memory_segments),
        "chunk_count": len(chunks),
        "stored_episode_count": stored_count,
        "source_type": args.source_type,
        "episode_type": args.episode_type,
        "session_start": session_start.isoformat(),
        "reflect_result": reflect_result,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Wrote report: %s", report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def load_project_config(config_path: Path) -> Dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML config support requires PyYAML.") from exc
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping in config file: {config_path}")
    return expand_env_refs(loaded)


def expand_env_refs(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_env_refs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_refs(item) for item in value]
    if isinstance(value, str):
        return _ENV_REF_RE.sub(lambda match: os.getenv(match.group(1), ""), value)
    return value


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return args.output_dir.expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (args.output_root_dir / f"{timestamp}_{args.textgrid.stem}").expanduser().resolve()


def configure_logging(log_path: Path, log_level: str) -> None:
    level = getattr(logging, str(log_level or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def parse_session_start(value: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.now(timezone.utc)
    normalized = text.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def load_textgrid_segments(textgrid_path: Path) -> List[TextGridSegment]:
    lines = textgrid_path.read_text(encoding="utf-8").splitlines()
    segments: List[TextGridSegment] = []
    current_tier = ""
    pending_start = None
    pending_end = None
    for raw_line in lines:
        line = raw_line.strip()
        name_match = _NAME_RE.fullmatch(line)
        if name_match:
            current_tier = unescape_text(name_match.group("name"))
            continue
        start_match = _XMIN_RE.fullmatch(line)
        if start_match:
            pending_start = float(start_match.group("value"))
            continue
        end_match = _XMAX_RE.fullmatch(line)
        if end_match:
            pending_end = float(end_match.group("value"))
            continue
        text_match = _TEXT_RE.fullmatch(line)
        if not text_match or pending_start is None or pending_end is None:
            continue
        text = unescape_text(text_match.group("text")).strip()
        if current_tier and text and pending_end > pending_start:
            segments.append(TextGridSegment(
                speaker=current_tier,
                start_s=pending_start,
                end_s=pending_end,
                text=text,
            ))
        pending_start = None
        pending_end = None
    return sorted(segments, key=lambda item: (item.start_s, item.end_s, item.speaker))


def unescape_text(value: str) -> str:
    return value.replace('""', '"')


def textgrid_segment_to_memory_segment(
    segment: TextGridSegment,
    *,
    session_start: datetime,
    index: int,
) -> Dict[str, Any]:
    started_at = session_start + timedelta(seconds=segment.start_s)
    ended_at = session_start + timedelta(seconds=segment.end_s)
    return {
        "speaker": segment.speaker,
        "role": "ambient_speaker",
        "text": segment.text,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "segment_index": index,
        "metadata": {
            "audio_start_s": round(segment.start_s, 3),
            "audio_end_s": round(segment.end_s, 3),
        },
    }


def chunk_memory_segments(
    segments: Sequence[Dict[str, Any]],
    *,
    max_segments: int,
    max_chars: int,
    max_gap_s: float,
) -> Iterable[List[Dict[str, Any]]]:
    chunk: List[Dict[str, Any]] = []
    chunk_chars = 0
    previous_end = ""
    for segment in segments:
        text_len = len(segment.get("text", ""))
        gap_s = timestamp_gap_seconds(previous_end, str(segment.get("started_at") or ""))
        should_flush = bool(chunk) and (
            len(chunk) >= max(1, max_segments)
            or chunk_chars + text_len > max(1, max_chars)
            or (max_gap_s >= 0 and gap_s is not None and gap_s > max_gap_s)
        )
        if should_flush:
            yield chunk
            chunk = []
            chunk_chars = 0
        chunk.append(dict(segment))
        chunk_chars += text_len
        previous_end = str(segment.get("ended_at") or segment.get("started_at") or "")
    if chunk:
        yield chunk


def timestamp_gap_seconds(previous_end: str, current_start: str) -> float | None:
    if not previous_end or not current_start:
        return None
    try:
        previous = datetime.fromisoformat(previous_end.replace("Z", "+00:00"))
        current = datetime.fromisoformat(current_start.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (current - previous).total_seconds()


def write_transcript_txt(path: Path, segments: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for segment in segments:
            meta = segment.get("metadata") or {}
            audio_start = float(meta.get("audio_start_s") or 0.0)
            audio_end = float(meta.get("audio_end_s") or 0.0)
            file.write(
                f"{segment.get('speaker')}\t"
                f"{audio_start:.3f}-{audio_end:.3f}\t"
                f"{segment.get('started_at')} - {segment.get('ended_at')}\t"
                f"{segment.get('text')}\n"
            )


if __name__ == "__main__":
    main()
