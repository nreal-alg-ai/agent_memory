#!/usr/bin/env python3
"""Extract PWM memory/state from Feishu-exported Markdown transcripts in order.

Example (run from the repository root)::

    python scripts/memory/ambient_transcript/run_feishu_meeting_extraction.py \
        --meeting-file '/path/to/文字记录：会议名称 2025年12月24日.md' \
        --owner-name 沈劲阳 --parse-only

Use --meeting-dir DIR --file-limit N to select the earliest N Markdown meetings
in that directory (non-recursive; 0 means all). Pass filenames after
--meeting-file together with --meeting-dir to select specific files, or pass
file paths without a directory. Multiple paths and repeated --meeting-file
are supported. All selected meetings are ordered by their actual meeting start
time, not filename or modification time, and share one PWM and databases.
Across the selected speech in each run, the wearer keeps speaker_用户本人;
other speakers receive speaker_1, speaker_2, ... in first-appearance order.
Named speakers share IDs across files; anonymous labels are file-scoped.
Original labels remain in speaker_mapping and segment metadata, not replaced
inside utterance text. Unselected segments do not consume speaker IDs.
Omit --parse-only to run extraction, without recall. Exact owner names may
include Feishu's " 在 <meeting room>" suffix; anonymous room/device speakers
are never inferred to be that person. Exported offsets are utterance STARTS,
not measured speech durations: ends are estimated from text length and capped
at the next start. The estimate and its rate are recorded in metadata/report;
use --speech-rate-chars-per-second to change the default engineering assumption.
Replay a logical clock without real-time sleeps. Deliver only finished speech,
finalize at each meeting end, then wait for memory, due state updates, prediction
maintenance, and proactive evaluation in that order. Between meetings and during
the configurable post-roll, time keeps advancing without fabricated input.
Proactive availability is explicitly simulated only outside meetings; no Agent
tools are executed and no background execution is authorized.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from config import split_memory_config
from pwm_manager import PersonWorldModel
from task_scheduler import TaskOperationReporter
from user_model.user_proactive_service import UserProactiveService

try:  # Support both direct CLI execution and package imports in tests.
    from .run_transcript_Ali_eval import (configure_logging, write_transcript_txt)
    from .feishu_test_speaker_mapping import remap_meeting_speakers
except ImportError:
    from run_transcript_Ali_eval import (configure_logging, write_transcript_txt)
    from feishu_test_speaker_mapping import remap_meeting_speakers


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp/ambient_transcript/feishu_meeting"
from memory.feishu_transcript import (
    SHANGHAI, OWNER_SPEAKER, HEADER_RE, DATE_RE, TIME_RANGE_RE, TIMEZONE_RE,
    MARKDOWN_ESCAPE_RE, plain_markdown, speaker_name, resolve_meeting_start,
    read_meeting_metadata, parse_meeting, DEFAULT_SPEECH_RATE_CHARS_PER_SECOND,
)
MEMORY_TABLES = (
    "memory_source_segments", "memory_facts", "memory_user_activity_items",
    "memory_user_claims", "memory_user_observed_patterns", "memory_user_intent_items",
    "memory_user_threads", "memory_user_claim_signals",
    "memory_user_intent_signals", "memory_user_action_signals",
)


def select_meeting_files(args: argparse.Namespace) -> List[Path]:
    """Resolve explicit files or sort directory exports before applying the limit."""
    directory = args.meeting_dir.expanduser().resolve() if args.meeting_dir else None
    if directory is not None and not directory.is_dir():
        raise NotADirectoryError(f"Meeting directory does not exist or is not a directory: {directory}")
    if args.meeting_file:
        candidates = []
        for path in args.meeting_file:
            source = path.expanduser()
            if directory is not None and not source.is_absolute():
                source = directory / source
            candidates.append(source.resolve())
    else:
        candidates = [
            path.resolve() for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".md"
        ]
    if not candidates:
        raise ValueError(f"No Markdown meeting files found in {directory}")
    dated_files = []
    for source in dict.fromkeys(candidates):
        if not source.is_file():
            raise FileNotFoundError(f"Meeting file does not exist or is not a file: {source}")
        with source.open(encoding="utf-8-sig") as stream:
            metadata = read_meeting_metadata(stream)
        try:
            start = resolve_meeting_start(
                title=metadata["heading"] or metadata["title"],
                time_description=metadata["time_description"], filename=source.name,
                override=args.session_start,
            )
        except ValueError as exc:
            raise ValueError(f"Invalid meeting time in {source}: {exc}") from exc
        dated_files.append((start.timestamp(), str(source), source))
    dated_files.sort(key=lambda item: (item[0], item[1]))
    paths = [item[2] for item in dated_files]
    return paths[:args.file_limit] if args.file_limit else paths


def register_test_owner(pwm: PersonWorldModel, owner_name: str) -> None:
    """Register operator-supplied identity, not a fabricated fact or LLM guess.

    The production alias upsert requires a fact ID. A test fixture has none, so
    use the nullable evidence_fact_id and explicitly record test configuration
    as the evidence source. Call before submitting any inputs to the FIFO.
    """
    db = pwm.database
    canonical = pwm.memory_manager.world_owner_entity_name
    with db.transaction():
        owner_id = db.upsert_entity_nodes([{"name": canonical, "type": "PERSON"}])[canonical]
        if owner_name == canonical:
            return
        timestamp = datetime.now().astimezone().isoformat()
        db._conn.execute(
            """INSERT INTO memory_owner_entity_aliases (
                owner_entity_id, alias_name, alias_key, status, confidence,
                evidence_fact_id, evidence_type, evidence_text, source_speaker,
                created_at, updated_at
            ) VALUES (?, ?, ?, 'confirmed', 1.0, NULL, 'test_configuration', ?, ?, ?, ?)""",
            (owner_id, owner_name, db.normalize_owner_alias_key(owner_name),
             f"Test operator specified --owner-name={owner_name}", OWNER_SPEAKER,
             timestamp, timestamp),
        )


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meeting-dir", type=Path,
                        help="Directory containing Feishu Markdown exports; non-recursive")
    parser.add_argument("--meeting-file", type=Path, nargs="+", action="extend", default=[],
                        help="Specific files, optionally relative to --meeting-dir; may be repeated; ordered by meeting time")
    parser.add_argument("--file-limit", type=int, default=0,
                        help="First N files after chronological sorting; 0 means all")
    parser.add_argument("--owner-name", required=True, help="Exact owner name in the export, e.g. 沈劲阳")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--output-dir", type=Path, help="New or empty directory; existing data is never reused")
    parser.add_argument("--session-start", default="", help="Single-file time override; naive ISO time means GMT+08")
    parser.add_argument("--segment-limit", type=int, default=0, help="First N chronological utterances per file; 0 means all")
    parser.add_argument(
        "--speech-rate-chars-per-second", type=float,
        default=DEFAULT_SPEECH_RATE_CHARS_PER_SECOND,
        help="Estimated equivalent characters/second for missing speech ends (default: 4; not measured timing)",
    )
    parser.add_argument("--parse-only", action="store_true", help="Export normalized inputs only; no models or databases")
    parser.add_argument("--clock-step-minutes", type=int, default=5,
                        help="Logical clock base step; meeting boundaries and task deadlines are also visited (default: 5)")
    parser.add_argument("--state-interval-minutes", type=int, default=30,
                        help="Periodic claim-to-state check interval; 0 disables (default: 30)")
    parser.add_argument("--prediction-interval-minutes", type=int, default=5,
                        help="Prediction expiration/defer tick interval; 0 disables (default: 5)")
    parser.add_argument("--proactive-interval-minutes", type=int, default=15,
                        help="Service opportunity interval, skipped during meetings; 0 disables (default: 15)")
    parser.add_argument("--post-roll-hours", type=float, default=24,
                        help="Continue the clock after the last selected speech ends (default: 24)")
    parser.add_argument("--clock-end", default="",
                        help="Explicit clock end, overriding post-roll; naive ISO time means GMT+08")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args(argv)
    if not args.meeting_file and args.meeting_dir is None:
        parser.error("supply --meeting-dir or --meeting-file")
    if args.file_limit < 0:
        parser.error("--file-limit must be nonnegative")
    if args.segment_limit < 0:
        parser.error("--segment-limit must be nonnegative")
    if args.clock_step_minutes <= 0:
        parser.error("--clock-step-minutes must be positive")
    for name in ("state_interval_minutes", "prediction_interval_minutes", "proactive_interval_minutes"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if not math.isfinite(args.post_roll_hours) or args.post_roll_hours < 0:
        parser.error("--post-roll-hours must be finite and nonnegative")
    if args.clock_end:
        try:
            args.clock_end = replay_datetime(args.clock_end).isoformat()
        except ValueError:
            parser.error("--clock-end must be an ISO timestamp")
    if not math.isfinite(args.speech_rate_chars_per_second) or args.speech_rate_chars_per_second <= 0:
        parser.error("--speech-rate-chars-per-second must be finite and positive")
    if args.session_start and len(args.meeting_file) != 1:
        parser.error("--session-start requires one explicit --meeting-file; other files use their own meeting times")
    return args


def meeting_output_dir(output_dir: Path, index: int, count: int) -> Path:
    return output_dir if count == 1 else output_dir / f"meeting_{index:04d}"


def replay_datetime(value: Any) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
        str(value).strip().replace("Z", "+00:00")
    )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=SHANGHAI)
    return parsed.astimezone(SHANGHAI)


def replay_clock_settings(args: argparse.Namespace, meetings: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Fix the replay range and reject overlapping sessions before starting PWM."""
    previous_end = None
    for meeting in meetings:
        start = replay_datetime(meeting["session_start"])
        end = replay_datetime(meeting["last_utterance_end"])
        if end < start:
            raise ValueError(f"Meeting ends before it starts: {meeting['source_file']}")
        if previous_end is not None and start < previous_end:
            raise ValueError("Overlapping meetings cannot share a single session-finalization boundary")
        previous_end = end
    start = replay_datetime(meetings[0]["session_start"])
    last_end = replay_datetime(meetings[-1]["last_utterance_end"])
    end = (replay_datetime(args.clock_end) if args.clock_end
           else last_end + timedelta(hours=args.post_roll_hours))
    if end < last_end:
        raise ValueError("--clock-end must not precede the last selected meeting's speech end")
    return {
        "start": start.isoformat(), "end": end.isoformat(),
        "clock_step_minutes": args.clock_step_minutes,
        "state_interval_minutes": args.state_interval_minutes,
        "prediction_interval_minutes": args.prediction_interval_minutes,
        "proactive_interval_minutes": args.proactive_interval_minutes,
        "post_roll_hours": args.post_roll_hours if not args.clock_end else None,
        "availability_scenario": "simulated_available_outside_meetings",
        "ticks_completed": 0, "failed_tick_count": 0,
    }


def iter_replay_clock(
    settings: Dict[str, Any], meetings: Sequence[Dict[str, Any]],
) -> Iterable[Dict[str, Any]]:
    """Merge base steps, exact periodic deadlines and exact meeting boundaries."""
    start, end = replay_datetime(settings["start"]), replay_datetime(settings["end"])
    intervals = {
        "step": settings["clock_step_minutes"],
        "state": settings["state_interval_minutes"],
        "prediction": settings["prediction_interval_minutes"],
        "proactive": settings["proactive_interval_minutes"],
    }
    deltas = {name: timedelta(minutes=minutes) for name, minutes in intervals.items() if minutes}
    next_times = {name: start if name == "step" else start + delta
                  for name, delta in deltas.items()}
    boundaries = sorted({
        replay_datetime(meeting[key])
        for meeting in meetings for key in ("session_start", "last_utterance_end")
    })
    boundary_index = 0
    while True:
        candidates = [end, *[value for value in next_times.values() if value <= end]]
        if boundary_index < len(boundaries):
            candidates.append(boundaries[boundary_index])
        current = min(candidates)
        due = [name for name in ("state", "prediction", "proactive")
               if next_times.get(name) == current]
        yield {"reference_timestamp": current.isoformat(), "due": due}
        if current == end:
            return
        for name, value in next_times.items():
            if value == current:
                next_times[name] = value + deltas[name]
        while boundary_index < len(boundaries) and boundaries[boundary_index] <= current:
            boundary_index += 1


def table_counts(pwm: PersonWorldModel) -> Dict[str, int]:
    counts = {
        table: int(pwm.database._conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        for table in MEMORY_TABLES
    }
    for table in ("user_states", "user_predictions", "user_proactive_services"):
        counts[table] = int(
            pwm.user_model_database._conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        )
    return counts


def operation_changes(reporter: TaskOperationReporter, before: Dict[str, Any]) -> Dict[str, Any]:
    """Attribute task failures to this interval, not all previous meetings."""
    snapshot = reporter.snapshot()
    delta = {}
    for kind, counts in snapshot["counts"].items():
        previous = before["counts"].get(kind, {})
        change = {key: round(value - previous.get(key, 0), 2) for key, value in counts.items()}
        if any(change.values()):
            delta[kind] = change
    previous_task_ids = {item["task_id"] for item in before["recent_tasks"]}
    events = [item for item in snapshot["recent_tasks"] if item["task_id"] not in previous_task_ids]
    return {
        "operation_delta": delta,
        "task_results": [item for item in events if item.get("event") == "finished"],
        "failed_tasks": [item for item in events
                         if item.get("event") == "finished" and item.get("succeeded") is False],
        "partial_tasks": [item for item in events
                          if item.get("event") == "finished" and item.get("status") == "partial"],
        "unexpected_rejections": [item for item in events
                                 if item.get("event") == "rejected" and item.get("reason") not in {
                                     "task_disabled", "no_pending_signals", "no_pending_activities",
                                     "signals_already_submitted",
                                 }],
    }


def has_task_failures(changes: Dict[str, Any]) -> bool:
    return bool(
        any(counts.get("failed") for counts in changes["operation_delta"].values())
        or changes["failed_tasks"] or changes["partial_tasks"] or changes["unexpected_rejections"]
    )


def evaluate_proactive_services(pwm: PersonWorldModel, reference_timestamp: str) -> Dict[str, Any]:
    """Simulate an available host; never infer availability from meeting content."""
    evaluation = {
        "scenario": "simulated_available_outside_meetings",
        "reference_timestamp": reference_timestamp,
        "runtime_context": {"can_present": True},
        "agent_capabilities": {
            "supported_service_types": ["notify", "suggest", "clarify", "offer_assistance"],
            "authorized_service_types": [],
        },
    }
    try:
        result = pwm.generate_proactive_services_from_predictions(
            reference_timestamp=reference_timestamp,
            runtime_context=evaluation["runtime_context"],
            agent_capabilities=evaluation["agent_capabilities"],
        )
        if not isinstance(result, dict) or result.get("status") not in {
            "ok", "empty", "partial", "error", "failed", "skipped",
        }:
            raise ValueError(f"Invalid proactive service result: {result!r}")
    except Exception as exc:
        if isinstance(exc, RuntimeError) and str(exc) == "Task not queued: task_disabled":
            result = {"status": "skipped", "reason": "task_disabled", "assistance_items": []}
        else:
            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}", "assistance_items": []}
            logging.exception("Proactive evaluation failed reference_timestamp=%s; continuing", reference_timestamp)
    evaluation["result"] = result
    return evaluation


def trigger_periodic_task(pwm: PersonWorldModel, kind: str, reference_timestamp: str) -> Dict[str, Any]:
    """Wait at each phase boundary, without sleeping for simulated time."""
    try:
        if kind == "state":
            result = pwm.trigger_user_state_update()
        else:
            result = pwm.trigger_user_prediction_time_tick(reference_timestamp=reference_timestamp)
        if not isinstance(result, dict):
            raise ValueError(f"Invalid {kind} tick result: {result!r}")
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        logging.exception("Periodic %s trigger failed reference_timestamp=%s; continuing", kind, reference_timestamp)
    if not pwm.wait_for_memory_tasks():
        raise RuntimeError(f"{kind.capitalize()} tasks did not drain")
    return result


def replay_meetings(
    pwm: PersonWorldModel, reporter: TaskOperationReporter,
    meetings: Sequence[Dict[str, Any]], meeting_reports: List[Dict[str, Any]],
    output_dir: Path, settings: Dict[str, Any],
) -> None:
    """Replay inputs and due work serially; never expose a later meeting early."""
    cursors = [0] * len(meetings)
    meeting_baselines: Dict[int, Dict[str, Any]] = {}
    with (output_dir / "clock_events.jsonl").open("w", encoding="utf-8") as stream:
        for event in iter_replay_clock(settings, meetings):
            current = replay_datetime(event["reference_timestamp"])
            before = reporter.snapshot()
            event.update(status="running", input_batches=[], finalized_meetings=[])
            ending = []
            try:
                for index, (meeting, report) in enumerate(zip(meetings, meeting_reports)):
                    start = replay_datetime(meeting["session_start"])
                    if start > current or report["status"] in {"completed", "completed_with_failures"}:
                        continue
                    if report["status"] == "pending":
                        meeting_baselines[index] = reporter.snapshot()
                        report.update(status="running", input_batches=[], clock_failure_count=0)
                        logging.info("Meeting replay start index=%s source=%s time=%s",
                                     index + 1, meeting["source_file"], current.isoformat())
                    first = cursors[index]
                    segments = meeting["segments"]
                    while (cursors[index] < len(segments)
                           and replay_datetime(segments[cursors[index]]["ended_at"]) <= current):
                        cursors[index] += 1
                    tags = ["feishu_meeting", f"meeting:{meeting['meeting_title']}"]
                    if cursors[index] > first:
                        result = pwm.accept_memory_input(
                            transcript_segments=segments[first:cursors[index]], tags=tags,
                            ambient_recording_enabled=True,
                        )
                        batch = {"meeting_index": index + 1, "reference_timestamp": current.isoformat(),
                                 "segment_count": cursors[index] - first, "input_report": result}
                        event["input_batches"].append(batch)
                        report["input_batches"].append(batch)
                        if result.get("accepted") is False or result.get("reason") in {
                            "memory_disabled", "empty_transcript_batch", "queue_rejected",
                            "worker_queue_full", "worker_shutdown", "task_handler_not_registered", "task_disabled",
                        }:
                            raise RuntimeError(f"Meeting input rejected: {result}")
                    if current >= replay_datetime(meeting["last_utterance_end"]):
                        report["finalization"] = pwm.finalize_memory_session(
                            reason="feishu_meeting_complete", tags=tags,
                        )
                        if not report["finalization"].get("accepted", False):
                            raise RuntimeError(f"Meeting finalization rejected: {report['finalization']}")
                        event["finalized_meetings"].append(index + 1)
                        ending.append(index)
                if not pwm.wait_for_memory_tasks():
                    raise RuntimeError("Memory extraction tasks did not drain")
                if "state" in event["due"]:
                    event["state_tick"] = trigger_periodic_task(pwm, "state", current.isoformat())
                if "prediction" in event["due"]:
                    event["prediction_tick"] = trigger_periodic_task(pwm, "prediction", current.isoformat())
                event["active_meeting_indices"] = [
                    index + 1 for index, meeting in enumerate(meetings)
                    if replay_datetime(meeting["session_start"]) <= current
                    < replay_datetime(meeting["last_utterance_end"])
                ]
                if "proactive" in event["due"]:
                    if event["active_meeting_indices"]:
                        event["proactive_service_evaluation"] = {
                            "reference_timestamp": current.isoformat(),
                            "result": {"status": "skipped", "reason": "simulated_meeting_in_progress",
                                       "assistance_items": []},
                        }
                    else:
                        event["proactive_service_evaluation"] = evaluate_proactive_services(pwm, current.isoformat())
                event.update(operation_changes(reporter, before))
                direct_results = [event.get("state_tick", {}), event.get("prediction_tick", {}),
                                  event.get("proactive_service_evaluation", {}).get("result", {})]
                failed = (has_task_failures(event)
                          or any(result.get("status") in {"partial", "error", "failed"} for result in direct_results))
                event["status"] = "completed_with_failures" if failed else "completed"
                event["table_counts"] = table_counts(pwm)
                for report in meeting_reports:
                    if report["status"] == "running" and failed:
                        report["clock_failure_count"] += 1
                for index in ending:
                    report = meeting_reports[index]
                    report.update(operation_changes(reporter, meeting_baselines[index]))
                    report["table_counts"] = event["table_counts"]
                    report["status"] = ("completed_with_failures" if has_task_failures(report)
                                        or report["clock_failure_count"] else "completed")
                    logging.info("Meeting replay complete index=%s status=%s counts=%s",
                                 index + 1, report["status"], report["table_counts"])
                    if len(meetings) > 1:
                        write_json(meeting_output_dir(output_dir, index + 1, len(meetings)) / "report.json", report)
                blocking = [item for item in event["unexpected_rejections"]
                            if item.get("operation_type") != UserProactiveService.TASK_KIND]
                if blocking:
                    raise RuntimeError(f"Task submissions rejected: {blocking}")
            except Exception as exc:
                event.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                for index, report in enumerate(meeting_reports):
                    if report["status"] == "running" or index in ending:
                        report.update(status="failed", error=event["error"])
                        if len(meetings) > 1:
                            write_json(meeting_output_dir(output_dir, index + 1, len(meetings)) / "report.json", report)
                raise
            finally:
                event.update(operation_changes(reporter, before))
                settings["ticks_completed"] += 1
                settings["failed_tick_count"] += int(event["status"] != "completed")
                stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
                stream.flush()
                logging.info("Replay tick time=%s due=%s input_batches=%s finalized=%s status=%s",
                             current.isoformat(), event["due"], len(event["input_batches"]),
                             event["finalized_meetings"], event["status"])


def run_extraction(args: argparse.Namespace, meetings: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    clock = replay_clock_settings(args, meetings)
    sections = split_memory_config(args.config)
    sections.memory_manager["user_memory_database_path"] = str(output_dir / "memory.db")
    sections.memory_manager["world_owner_entity_name"] = "用户本人"
    sections.user_state["user_model_database_path"] = str(output_dir / "user_model.db")
    for key, value in (("llm_name", args.llm_model), ("llm_base_url", args.llm_base_url)):
        if value is not None:
            sections.llm_model[key] = value
    reporter = TaskOperationReporter(recent_task_limit=10000)
    pwm = None
    meeting_reports = [
        {**{key: value for key, value in meeting.items() if key != "segments"},
         "meeting_index": index, "status": "pending"}
        for index, meeting in enumerate(meetings, 1)
    ]
    report: Dict[str, Any] = (
        {key: value for key, value in meetings[0].items() if key != "segments"}
        if len(meetings) == 1 else {}
    )
    report.update({
        "status": "running", "config": str(args.config.expanduser().resolve()),
        "meeting_count": len(meetings), "meetings": meeting_reports,
        "memory_db_path": str(output_dir / "memory.db"),
        "user_model_db_path": str(output_dir / "user_model.db"),
        "task_db_path": str(output_dir / "task_scheduler.db"),
        "clock": clock,
        "clock_events_path": str(output_dir / "clock_events.jsonl"),
    })
    try:
        pwm = PersonWorldModel(
            memory_runtime_config=sections.memory_runtime,
            memory_manager_config=sections.memory_manager,
            user_state_config=sections.user_state,
            llm_model_config=sections.llm_model,
            embedding_model_config=sections.embedding_model,
            task_scheduling_config=sections.task_scheduling,
            operation_reporter=reporter,
            logging_config={**sections.logging, "directory": str((output_dir / "logs").resolve())},
        )
        register_test_owner(pwm, meetings[0]["owner_name"])
        for index in range(1, len(meetings) + 1):
            directory = meeting_output_dir(output_dir, index, len(meetings))
            directory.mkdir(parents=True, exist_ok=True)
        replay_meetings(pwm, reporter, meetings, meeting_reports, output_dir, clock)
        report["table_counts"] = table_counts(pwm)
        report["status"] = (
            "completed_with_failures"
            if clock["failed_tick_count"]
            or any(item["status"] == "completed_with_failures" for item in meeting_reports)
            else "completed"
        )
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        logging.exception("Meeting extraction failed")
        raise
    finally:
        try:
            if pwm is not None:
                pwm.close()
        finally:
            if len(meetings) == 1:
                for key in (
                    "input_batches", "finalization", "clock_failure_count",
                    "operation_delta", "task_results", "failed_tasks", "partial_tasks", "unexpected_rejections",
                ):
                    if key in meeting_reports[0]:
                        report[key] = meeting_reports[0][key]
            report["operations"] = reporter.snapshot()
            write_json(output_dir / "report.json", report)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    meetings = []
    # Sort by export timestamps and apply the file limit before parsing speech.
    # Validate all selected sources before starting models or creating output.
    for source in select_meeting_files(args):
        meeting = parse_meeting(
            source, owner_name=args.owner_name, session_start=args.session_start,
            speech_rate_chars_per_second=args.speech_rate_chars_per_second,
        )
        if args.segment_limit:
            meeting["segments"] = meeting["segments"][:args.segment_limit]
        meeting["selected_segment_count"] = len(meeting["segments"])
        meeting["selected_owner_segment_count"] = sum(
            item["speaker"] == OWNER_SPEAKER for item in meeting["segments"]
        )
        meeting["segment_limit"] = args.segment_limit
        meeting["last_utterance_start"] = meeting["segments"][-1]["started_at"]
        meeting["last_utterance_end"] = meeting["segments"][-1]["ended_at"]
        meetings.append(meeting)
    remap_meeting_speakers(meetings)
    replay_clock_settings(args, meetings)
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else
        DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir / "run.log", args.log_level)
    for index, meeting in enumerate(meetings, 1):
        directory = meeting_output_dir(output_dir, index, len(meetings))
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "parsed_meeting.json", meeting)
        write_transcript_txt(directory / "transcript.txt", meeting["segments"])
        logging.info(
            "Meeting parsed index=%s/%s source=%s owner=%s start=%s speakers=%s segments=%s/%s owner_segments=%s output=%s",
            index, len(meetings), meeting["source_file"], meeting["owner_name"], meeting["session_start"],
            len(meeting["speaker_mapping"]), meeting["selected_segment_count"], meeting["source_segment_count"],
            meeting["selected_owner_segment_count"], directory,
        )
    if len(meetings) > 1:
        write_json(output_dir / "parsed_meetings.json", [
            {**{key: value for key, value in meeting.items() if key != "segments"},
             "meeting_index": index, "output_dir": str(meeting_output_dir(output_dir, index, len(meetings)))}
            for index, meeting in enumerate(meetings, 1)
        ])
    if not args.parse_only:
        run_extraction(args, meetings, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
