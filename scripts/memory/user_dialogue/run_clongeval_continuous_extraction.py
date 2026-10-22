#!/usr/bin/env python3
"""Continuously extract memory and user state from CLongEval contexts.

Unlike run_clongeval_conversation.py, this runner never recalls or answers a
question. Distinct ``context`` values are replayed in JSON/JSONL input order
through one PersonWorldModel and one pair of databases. Each context forms a
session boundary, and the script drains memory tasks plus the claim-state
window before recording a progress snapshot.

CLongEval contexts may represent different fictional people. A shared database
is useful for pipeline testing, but is not an identity-isolated benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (SRC_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from config import split_memory_config
from pwm_manager import PersonWorldModel
from task_scheduler import TaskOperationReporter
from scripts.memory.user_dialogue.run_clongeval_conversation import (
    configure_memory_logger,
    normalize_unique_timestamp,
    parse_context_days,
    remove_memory_logger_handler,
)
from scripts.memory.user_dialogue.test_memory_store_fact_extraction import (
    configure_logging,
    validate_embedding_runtime,
)


DEFAULT_INPUT = (
    REPO_ROOT / "test_data" / "user_dialogue" / "CLongEval"
    / "1-2_long_conversation_memory" / "small.jsonl"
)
MEMORY_COUNT_TABLES = (
    "memory_source_segments",
    "memory_facts",
    "memory_user_activity_items",
    "memory_user_claims",
    "memory_user_observed_patterns",
    "memory_user_intent_items",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir", type=Path,
        help="Fresh result directory; defaults to tmp/clongeval_continuous/<timestamp>.",
    )
    parser.add_argument("--start", type=int, default=0,
                        help="Skip this many distinct contexts in input order.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Maximum distinct contexts; 0 means all.")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key")
    parser.add_argument("--llm-timeout", type=int)
    parser.add_argument("--skip-embedding-validation", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--manager-log-level", default="INFO")
    args = parser.parse_args(argv)
    if args.start < 0 or args.limit < 0:
        parser.error("--start and --limit must be nonnegative")
    return args


def _json_records(path: Path) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Read JSONL records or a JSON object/list without requiring query fields."""
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}")
                yield str(line_number), item
        return

    loaded = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(loaded, dict):
        records = loaded.get("records", loaded.get("data", loaded))
    else:
        records = loaded
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON object or list of objects: {path}")
    for index, item in enumerate(records, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Expected an object at {path} item {index}")
        yield str(index), item


def select_contexts(
    path: Path, *, start: int, limit: int,
) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"CLongEval input does not exist: {path}")
    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for source_index, record in _json_records(path):
        context = str(record.get("context") or "").strip()
        if not context:
            logging.warning("Skipping empty context at input item %s", source_index)
            continue
        digest = hashlib.sha1(context.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        if len(seen) <= start:
            continue
        selected.append({
            "context": context,
            "context_id": f"context_{len(seen):04d}_{digest[:12]}",
            "source_index": source_index,
            "source_id": str(record.get("id") or ""),
        })
        if limit and len(selected) >= limit:
            break
    return selected


def _table_counts(pwm: PersonWorldModel) -> Dict[str, int]:
    memory_conn = pwm.database._conn
    model_conn = pwm.user_model_database._conn
    counts = {
        table: int(memory_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in MEMORY_COUNT_TABLES
    }
    counts["user_states"] = int(
        model_conn.execute("SELECT COUNT(*) FROM user_states").fetchone()[0]
    )
    return counts


def _operation_counts(reporter: TaskOperationReporter) -> Dict[str, Dict[str, int]]:
    return {
        kind: {
            field: int(stats.get(field) or 0)
            for field in ("submitted", "completed", "succeeded", "failed", "rejected")
        }
        for kind, stats in (reporter.snapshot().get("counts") or {}).items()
    }


def _operation_delta(
    before: Dict[str, Dict[str, int]], after: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, int]]:
    return {
        kind: {
            field: int(values.get(field, 0)) - int(before.get(kind, {}).get(field, 0))
            for field in values
        }
        for kind, values in after.items()
        if any(
            int(value) != int(before.get(kind, {}).get(field, 0))
            for field, value in values.items()
        )
    }


def _classify_task_outcomes(
    reporter: TaskOperationReporter,
    delta: Dict[str, Dict[str, int]],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, str]]]:
    """Treat an empty signal queue as a skipped follow-up, not a failed task."""
    failures: Dict[str, Dict[str, Any]] = {}
    skipped: List[Dict[str, str]] = []
    events = reporter.snapshot().get("recent_tasks") or []
    for kind, counts in delta.items():
        if counts.get("failed"):
            failures[kind] = {"counts": counts, "reason": "task_failed"}
        rejected_count = int(counts.get("rejected") or 0)
        if not rejected_count:
            continue
        rejected_events = [
            event for event in events
            if event.get("event") == "rejected"
            and event.get("operation_type") == kind
        ][-rejected_count:]
        if len(rejected_events) != rejected_count:
            failures[kind] = {
                "counts": counts,
                "reason": "rejection_details_not_available",
            }
            continue
        for event in rejected_events:
            reason = str(event.get("reason") or "")
            if reason == "no_pending_signals":
                skipped.append({"task_kind": kind, "reason": reason})
            else:
                failures[kind] = {"counts": counts, "reason": reason}
    return failures, skipped


def _state_snapshot(pwm: PersonWorldModel) -> List[Dict[str, Any]]:
    fields = (
        "id", "state_domain", "state_type", "state_kind", "state_topic",
        "state_text", "status", "confidence", "valid_from", "valid_to",
    )
    return [
        {field: state.get(field) for field in fields}
        for state in pwm.get_user_states(statuses=["active", "weakened", "expired"])
    ]


def replay_context(
    pwm: PersonWorldModel, context_item: Dict[str, Any],
    seen_timestamps: set[str],
) -> Dict[str, Any]:
    context_id = str(context_item["context_id"])
    days = parse_context_days(str(context_item["context"]))
    turns = 0
    last_timestamp: Optional[datetime] = None
    for day in days:
        for pair_index, (user, assistant) in enumerate(day["pairs"]):
            timestamp = normalize_unique_timestamp(
                day["date"] + timedelta(seconds=pair_index), seen_timestamps,
            )
            report = pwm.accept_memory_input(
                interaction_turn={
                    "user_message": user,
                    "assistant_response": assistant,
                    "turn_timestamp": timestamp,
                },
                tags=["clongeval_continuous", f"context:{context_id}",
                      f"date:{day['date_text']}"],
                ambient_recording_enabled=False,
            )
            if report.get("reason") in {
                "expected_one_input_kind", "memory_disabled", "empty_turn",
                "queue_rejected", "worker_queue_full", "worker_shutdown",
                "task_handler_not_registered", "task_disabled",
            }:
                raise RuntimeError(f"Input rejected for {context_id}: {report}")
            turns += 1
            last_timestamp = timestamp

    finalization = pwm.finalize_memory_session(
        reason="clongeval_context_boundary",
        tags=["clongeval_continuous", f"context:{context_id}"],
    )
    if not finalization.get("accepted", False):
        raise RuntimeError(f"Finalization rejected for {context_id}: {finalization}")
    if not pwm.wait_for_memory_tasks():
        raise RuntimeError(f"Memory task queue did not drain for {context_id}")
    # The normal claim-state task uses a thirty-minute window. A benchmark
    # context boundary explicitly drains it so this context's state is visible.
    state_tick = pwm.trigger_user_state_time_tick(force=True)
    if state_tick.get("reason") in {
        "worker_queue_full", "worker_shutdown", "task_handler_not_registered",
    }:
        raise RuntimeError(f"Claim-state tick rejected for {context_id}: {state_tick}")
    if not pwm.wait_for_memory_tasks():
        raise RuntimeError(f"State task queue did not drain for {context_id}")
    return {
        "parsed_days": len(days),
        "turn_pairs": turns,
        "last_turn_timestamp": last_timestamp.isoformat() if last_timestamp else None,
        "finalization": finalization,
        "claim_state_tick": state_tick,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_path = args.input.expanduser().resolve()
    contexts = select_contexts(input_path, start=args.start, limit=args.limit)
    if not contexts:
        raise RuntimeError("No distinct CLongEval contexts selected")

    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else
        REPO_ROOT / "tmp" / "clongeval_continuous"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir / "run.log", args.log_level, args.manager_log_level)
    logging.info(
        "Continuous extraction started contexts=%s input=%s output=%s; "
        "different CLongEval contexts may represent different users",
        len(contexts), input_path, output_dir,
    )

    sections = split_memory_config(args.config)
    llm_config = sections.llm_model
    embedding_config = sections.embedding_model
    runtime_config = sections.memory_runtime
    scheduler_config = sections.task_scheduling
    manager_config = sections.memory_manager
    state_config = sections.user_state
    for key, value in (
        ("llm_name", args.llm_model),
        ("llm_base_url", args.llm_base_url),
        ("llm_api_key", args.llm_api_key),
        ("llm_timeout", args.llm_timeout),
    ):
        if value is not None:
            llm_config[key] = value
    manager_config["user_memory_database_path"] = str(output_dir / "memory.db")
    state_config["user_model_database_path"] = str(output_dir / "user_model.db")

    memory_logger, memory_log_handler = configure_memory_logger(
        output_dir / "memory.log", args.manager_log_level,
        "memory.pipeline.clongeval_continuous",
    )
    # Retain rejection reasons across several contexts for accurate diagnostics.
    reporter = TaskOperationReporter(recent_task_limit=10000)
    pwm: Optional[PersonWorldModel] = None
    results: List[Dict[str, Any]] = []
    seen_timestamps: set[str] = set()
    try:
        pwm = PersonWorldModel(
            memory_runtime_config=runtime_config,
            memory_manager_config=manager_config,
            user_state_config=state_config,
            llm_model_config=llm_config,
            embedding_model_config=embedding_config,
            task_scheduling_config=scheduler_config,
            operation_reporter=reporter,
            logger=memory_logger,
        )
        if not args.skip_embedding_validation:
            validate_embedding_runtime(pwm.memory_manager, pwm.database, embedding_config)
        for index, context_item in enumerate(contexts, 1):
            before = _operation_counts(reporter)
            logging.info(
                "Context started index=%s/%s id=%s source_item=%s chars=%s",
                index, len(contexts), context_item["context_id"],
                context_item["source_index"], len(context_item["context"]),
            )
            replay = replay_context(pwm, context_item, seen_timestamps)
            delta = _operation_delta(before, _operation_counts(reporter))
            failures, skipped_tasks = _classify_task_outcomes(reporter, delta)
            result = {
                "context_index": index,
                "context_id": context_item["context_id"],
                "source_index": context_item["source_index"],
                "source_id": context_item["source_id"],
                "context_chars": len(context_item["context"]),
                "replay": replay,
                "table_counts": _table_counts(pwm),
                "operation_delta": delta,
                "skipped_tasks": skipped_tasks,
                "states": _state_snapshot(pwm),
            }
            results.append(result)
            _write_json(output_dir / "context_progress.json", results)
            logging.info(
                "Context completed index=%s/%s id=%s turns=%s facts=%s "
                "claims=%s states=%s skipped_tasks=%s failures=%s",
                index, len(contexts), context_item["context_id"],
                replay["turn_pairs"], result["table_counts"]["memory_facts"],
                result["table_counts"]["memory_user_claims"],
                result["table_counts"]["user_states"], skipped_tasks, failures,
            )
            if failures:
                raise RuntimeError(
                    f"Tasks failed while processing {context_item['context_id']}: {failures}"
                )

        summary = {
            "input": str(input_path),
            "output_dir": str(output_dir),
            "contexts_processed": len(results),
            "turn_pairs_processed": sum(row["replay"]["turn_pairs"] for row in results),
            "memory_db_path": str(output_dir / "memory.db"),
            "user_model_db_path": str(output_dir / "user_model.db"),
            "task_db_path": str(output_dir / "task_scheduler.db"),
            "table_counts": _table_counts(pwm),
            "operation_counts": _operation_counts(reporter),
            "states": _state_snapshot(pwm),
            "contexts": results,
        }
        _write_json(output_dir / "summary.json", summary)
        logging.info(
            "Continuous extraction finished contexts=%s turns=%s facts=%s states=%s",
            summary["contexts_processed"], summary["turn_pairs_processed"],
            summary["table_counts"]["memory_facts"],
            summary["table_counts"]["user_states"],
        )
        return 0
    finally:
        try:
            if pwm is not None:
                pwm.close()
        finally:
            remove_memory_logger_handler(memory_logger, memory_log_handler)


if __name__ == "__main__":
    raise SystemExit(main())
