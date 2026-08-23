#!/usr/bin/env python3
"""Exercise the VoiceRuntime -> MemoryRuntime ingestion path."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from memory_mcp_server import (  # noqa: E402
    _resolve_configured_path,
    build_service,
    load_config,
)


DEFAULT_AUDIO_DIR = (
    ROOT
    / "test_data/ambient_transcript/Eval_Ali/Eval_Ali_far/audio_dir"
)
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test VoiceRuntime transcription and MemoryRuntime storage together."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Override memory_mcp_server.db_path from config.yaml.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Override memory_mcp_server.log_path from config.yaml.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Override memory_mcp_server.report_path from config.yaml.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Delete existing database, log, and report files before testing.",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--queue-timeout", type=float, default=60.0)
    parser.add_argument("--source-type", default="allday_recording")
    parser.add_argument("--session-start")
    parser.add_argument("--tag", dest="tags", action="append", default=[])
    parser.add_argument("--max-files", type=int)
    parser.add_argument(
        "--query",
        help="Optional query to run after audio ingestion and queue flushing.",
    )
    return parser.parse_args()


def collect_audio_files(audio_dir: Path, max_files: int | None) -> List[Path]:
    directory = audio_dir.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {directory}")
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
    )
    if max_files is not None:
        if max_files <= 0:
            raise ValueError("--max-files must be positive")
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No supported audio files found in {directory}")
    return files


def build_service_args(args: argparse.Namespace) -> Namespace:
    """Build the small Namespace expected by memory_mcp_server.build_service."""
    return Namespace(
        config=args.config.expanduser().resolve(),
        db_path=(args.db_path.expanduser().resolve() if args.db_path else None),
        log_path=(args.log_path.expanduser().resolve() if args.log_path else None),
        log_level=args.log_level,
        queue_timeout=args.queue_timeout,
    )


def _remove_existing_file(path: Path) -> None:
    """Remove one output file while refusing to delete a directory."""
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir():
        raise IsADirectoryError(f"Refusing to delete directory: {path}")
    path.unlink()


def remove_existing_outputs(
    db_path: Path,
    log_path: Path | None,
    report_path: Path,
) -> None:
    """Remove files that could mix results from an earlier test run."""
    paths = [
        db_path,
        Path(f"{db_path}-wal"),
        Path(f"{db_path}-shm"),
        log_path,
        report_path,
    ]
    seen: set[Path] = set()
    for path in paths:
        if path is None:
            continue
        resolved_path = path.expanduser().resolve()
        if resolved_path in seen:
            continue
        seen.add(resolved_path)
        _remove_existing_file(resolved_path)


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    server_config = config.get("memory_mcp_server") or {}
    if not isinstance(server_config, dict):
        raise ValueError("memory_mcp_server in config.yaml must be a mapping")
    db_path = (
        args.db_path.expanduser().resolve()
        if args.db_path
        else _resolve_configured_path(server_config.get("db_path"), config_path)
    )
    log_path = (
        args.log_path.expanduser().resolve()
        if args.log_path
        else _resolve_configured_path(server_config.get("log_path"), config_path)
    )
    report_path = (
        args.report_path.expanduser().resolve()
        if args.report_path
        else _resolve_configured_path(server_config.get("report_path"), config_path)
    )
    if db_path is None:
        raise ValueError("memory_mcp_server.db_path must be configured")
    if report_path is None:
        raise ValueError("memory_mcp_server.report_path must be configured")
    if args.override:
        remove_existing_outputs(db_path, log_path, report_path)
    audio_files = collect_audio_files(args.audio_dir, args.max_files)
    service, logger = build_service(build_service_args(args))
    try:
        logger.info(
            "Voice/memory runtime test started audio_dir=%s file_count=%s db_path=%s",
            args.audio_dir.expanduser().resolve(),
            len(audio_files),
            db_path,
        )
        processing_report = service.process_audio_files(
            files=[{"audio_path": str(path)} for path in audio_files],
            source_type=args.source_type,
            session_start=args.session_start,
            tags=list(args.tags),
        )
        result: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config_path": str(config_path),
            "audio_dir": str(args.audio_dir.expanduser().resolve()),
            "audio_files": [str(path) for path in audio_files],
            "db_path": str(db_path),
            "override": bool(args.override),
            "processing": processing_report,
        }
        if args.query:
            result["recall"] = service.trigger_memory_recall(query=args.query)

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Voice/memory runtime test finished report_path=%s", report_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
