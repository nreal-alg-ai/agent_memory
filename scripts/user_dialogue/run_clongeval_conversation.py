#!/usr/bin/env python3
"""Run MemoryNodeManager on the CLongEval conversation benchmark.

CLongEval conversation records are JSONL objects with this shape::

    {
        "context": "...dated Chinese conversations...",
        "query": "...question...",
        "answer": "...gold answer...",
        "id": "..."
    }

The same context is often reused by several questions. This runner groups
records by exact context, replays each context into one isolated memory DB,
and then runs all of that context's questions against the same manager. It
uses the project config.yaml and the StoreFactExtractionManager test adapter
so memory extraction behavior stays aligned with the existing store test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = REPO_ROOT / "scripts" / "user_dialogue"
for import_root in (SRC_ROOT, REPO_ROOT, SCRIPT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from memory.memory_database import SessionDB
from test_memory_store_fact_extraction import (
    StoreFactExtractionManager,
    configure_logging,
    load_project_config,
    log_memory_index_state,
    resolve_llm_args,
    validate_embedding_runtime,
)


DEFAULT_INPUT = (
    REPO_ROOT
    / "test_data"
    / "user_dialogue"
    / "CLongEval"
    / "1-2_long_conversation_memory"
    / "small.jsonl"
)
DEFAULT_OUTPUT_ROOT_DIR = REPO_ROOT / "tmp" / "clongeval_conversation"

DATE_HEADER_RE = re.compile(
    r"以下是(?P<date>\d{4}年\d{1,2}月\d{1,2}日)的对话记录\s*[:：]?"
)
ROLE_LINE_RE = re.compile(r"^[“”]?\s*(用户|AI|助手)\s*[:：]\s*(.*)$")
CONTEXT_END_MARKERS = ("请记住以上全部对话记录", "问题：", "问题:")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MemoryNodeManager on CLongEval conversation JSONL data."
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root-dir", type=Path, default=DEFAULT_OUTPUT_ROOT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all records.")
    parser.add_argument(
        "--question-id",
        action="append",
        help="Only run the specified record id. Can be passed multiple times.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key")
    parser.add_argument("--llm-timeout", type=int)
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=8192,
        help="Maximum output tokens for each memory extraction call.",
    )
    parser.add_argument(
        "--llm-thinking",
        choices=("disabled", "enabled", "auto"),
        default=None,
        help="Defaults to memory.llm_thinking in config.yaml.",
    )
    parser.add_argument(
        "--no-llm-json-mode",
        action="store_false",
        dest="llm_json_mode",
        help="Do not request provider-enforced JSON output.",
    )
    parser.set_defaults(llm_json_mode=None)
    parser.add_argument("--min-dialogue-turns-before-store", type=int)
    parser.add_argument("--max-dialogue-chars-before-store", type=int)
    parser.add_argument("--enable-reflect", action="store_true")
    parser.add_argument(
        "--reflect-every-days",
        type=int,
        default=1,
        help="Run reflect after every N parsed conversation dates when enabled.",
    )
    parser.add_argument("--reflect-limit", type=int)
    parser.add_argument("--recall-top-k", type=int)
    parser.add_argument("--recall-budget")
    parser.add_argument(
        "--recall-path",
        choices=("stage1", "stage2", "normal"),
        default="normal",
        help="Recall path passed to MemoryNodeManager.recall().",
    )
    parser.add_argument("--skip-embedding-validation", action="store_true")
    parser.add_argument(
        "--run-reader",
        action="store_true",
        help="Use the configured memory LLM to answer each question from recall context.",
    )
    parser.add_argument("--reader-max-context-chars", type=int, default=12000)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--manager-log-level", default="INFO")
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"CLongEval input does not exist: {path}")
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
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


def normalize_unique_timestamp(candidate: datetime, seen: set[str]) -> datetime:
    current = candidate
    while current.strftime("%Y-%m-%d %H:%M:%S") in seen:
        current += timedelta(seconds=1)
    seen.add(current.strftime("%Y-%m-%d %H:%M:%S"))
    return current


def context_group_id(context: str, group_index: int) -> str:
    digest = hashlib.sha1(context.encode("utf-8")).hexdigest()[:12]
    return f"context_{group_index:04d}_{digest}"


def group_records_by_context(
    records: Sequence[Dict[str, Any]],
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    grouped: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for record in records:
        grouped.setdefault(str(record["context"]), []).append(record)
    return list(grouped.items())


def db_counts(db: SessionDB) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for table in (
        "memory_episodes",
        "memory_facts",
        "memory_states",
        "memory_actionable_items",
        "memory_index_entries",
    ):
        row = db._conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
        counts[table] = int(row["count"] if row else 0)
    return counts


def replay_context_into_memory(
    *,
    manager: StoreFactExtractionManager,
    db: SessionDB,
    days: Sequence[Dict[str, Any]],
    group_id: str,
    enable_reflect: bool,
    reflect_every_days: int,
    reflect_limit: int,
) -> Dict[str, Any]:
    seen_timestamps: set[str] = set()
    pair_count = 0
    store_batches = 0
    reflect_runs = 0
    reflect_reports: List[Dict[str, Any]] = []
    last_reflected_day = 0
    every_days = max(1, int(reflect_every_days or 1))

    for day_index, day in enumerate(days, 1):
        last_timestamp: Optional[datetime] = None
        for pair_index, (user, assistant) in enumerate(day["pairs"]):
            pair_count += 1
            turn_timestamp = normalize_unique_timestamp(
                day["date"] + timedelta(seconds=pair_index),
                seen_timestamps,
            )
            last_timestamp = turn_timestamp
            stored = manager.store_turn(
                user,
                assistant,
                tags=[
                    "clongeval_conversation",
                    f"context_group:{group_id}",
                    f"date:{day['date_text']}",
                    f"day_index:{day_index}",
                ],
                turn_timestamp=turn_timestamp,
            )
            if stored:
                store_batches += 1

        should_reflect = enable_reflect and (
            day_index % every_days == 0 or day_index == len(days)
        )
        if should_reflect:
            if manager._pending_interaction_turns:
                if manager.flush_pending_interaction_turns():
                    store_batches += 1
            reflect_timestamp = last_timestamp or day["date"]
            report = manager.reflect(
                limit=max(1, int(reflect_limit or 100)),
                reflect_timestamp=reflect_timestamp,
            )
            reflect_runs += 1
            last_reflected_day = day_index
            reflect_reports.append({
                "day_index": day_index,
                "date": day["date_text"],
                "report": report,
            })

    if manager._pending_interaction_turns:
        if manager.flush_pending_interaction_turns():
            store_batches += 1

    return {
        "parsed_days": len(days),
        "turn_pairs": pair_count,
        "store_batches": store_batches,
        "reflect_runs": reflect_runs,
        "last_reflected_day": last_reflected_day,
        "reflect_reports": reflect_reports,
        "db_counts": db_counts(db),
    }


def normalize_for_match(value: Any) -> str:
    text = str(value or "").lower()
    return re.sub(r"\s+", "", text)


def gold_answer_in_context(answer: str, context: str) -> bool:
    normalized_answer = normalize_for_match(answer)
    return bool(normalized_answer) and normalized_answer in normalize_for_match(context)


def truncate_text(value: str, max_chars: int) -> str:
    text = str(value or "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 24)] + "...[truncated]"


def build_reader_prompt(query: str, memory_context: str) -> str:
    return (
        "你是一个长期记忆问答评测器。只能依据下面的记忆上下文回答问题，不要补充上下文之外的信息。\n"
        "如果上下文中有直接证据，简洁返回答案；如果没有足够证据，返回空字符串。\n"
        "只返回 JSON：{\"answer\": \"...\"}\n\n"
        f"问题：{query}\n\n"
        "记忆上下文：\n"
        f"{memory_context or '[empty]'}"
    )


def run_reader(manager: StoreFactExtractionManager, query: str, memory_context: str, max_chars: int) -> str:
    if not memory_context.strip():
        return ""
    raw = manager._call_llm(
        build_reader_prompt(query, truncate_text(memory_context, max_chars))
    )
    parsed = manager._parse_json_object_from_llm_text(raw or "")
    if isinstance(parsed, dict):
        return str(parsed.get("answer") or parsed.get("final_answer") or "").strip()
    return str(raw or "").strip()


def prepare_runtime(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    resolve_llm_args(args)
    config = load_project_config(args.config)
    embedding_config = dict(config.get("embedding") or {}) if isinstance(config.get("embedding"), dict) else {}
    memory_config = dict(config.get("memory") or {}) if isinstance(config.get("memory"), dict) else {}
    if args.llm_thinking is None:
        args.llm_thinking = str(memory_config.get("llm_thinking") or "disabled")
    if args.llm_json_mode is None:
        args.llm_json_mode = bool(memory_config.get("llm_json_mode", True))
    args.reflect_limit = max(1, int(args.reflect_limit or memory_config.get("reflect_limit", 100) or 100))
    args.recall_top_k = max(1, int(args.recall_top_k or memory_config.get("retrieval_top_k", 8) or 8))
    args.recall_budget = str(args.recall_budget or memory_config.get("recall_budget", "mid") or "mid")
    memory_config["min_dialogue_turns_before_store"] = args.min_dialogue_turns_before_store
    memory_config["max_dialogue_chars_before_store"] = args.max_dialogue_chars_before_store
    memory_config["llm_timeout"] = args.llm_timeout
    memory_config["enable_entity_extraction"] = False
    return embedding_config, memory_config


def resolve_output_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root_dir = Path(args.output_root_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else output_root_dir / f"{args.input.stem}_{timestamp}"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}. Pass --overwrite.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_root_dir = output_root_dir
    args.output_dir = output_dir
    return output_dir


def add_context_log_handler(log_path: Path) -> logging.Handler:
    """Mirror the current context's records into its own output directory."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.NOTSET)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(handler)
    return handler


def remove_context_log_handler(handler: Optional[logging.Handler]) -> None:
    if handler is None:
        return
    root_logger = logging.getLogger()
    root_logger.removeHandler(handler)
    handler.close()


def main() -> int:
    args = parse_args()
    records = filter_records(
        load_records(args.input),
        question_ids=args.question_id,
        start=args.start,
        limit=args.limit,
    )
    if not records:
        raise RuntimeError("No CLongEval records selected")
    output_dir = resolve_output_dir(args)
    log_path = output_dir / "run_clongeval_conversation.log"
    configure_logging(log_path, args.log_level, args.manager_log_level)
    logging.info("Loaded CLongEval records=%s input=%s", len(records), args.input)

    embedding_config, memory_config = prepare_runtime(args)
    groups = group_records_by_context(records)
    results_path = output_dir / "clongeval_conversation_results.jsonl"
    group_summaries: List[Dict[str, Any]] = []
    result_count = 0
    recall_nonempty = 0
    gold_covered = 0
    reader_answered = 0

    with results_path.open("w", encoding="utf-8") as result_file:
        for group_index, (context, group_records) in enumerate(groups, 1):
            group_id = context_group_id(context, group_index)
            group_dir = output_dir / group_id
            group_dir.mkdir(parents=True, exist_ok=True)
            db_path = group_dir / "memory.db"
            logging.info(
                "Processing context group %s/%s id=%s records=%s context_chars=%s",
                group_index,
                len(groups),
                group_id,
                len(group_records),
                len(context),
            )
            days = parse_context_days(context)
            db = SessionDB(db_path)
            context_log_path = group_dir / "context.log"
            context_log_handler = add_context_log_handler(context_log_path)
            try:
                logging.info(
                    "Context processing started id=%s records=%s parsed_days=%s",
                    group_id,
                    len(group_records),
                    len(days),
                )
                manager = StoreFactExtractionManager(
                    db,
                    embedding_config=embedding_config,
                    memory_config=memory_config,
                    llm_model=args.llm_model,
                    llm_base_url=args.llm_base_url,
                    llm_api_key=args.llm_api_key,
                    report_rows=[],
                    llm_max_tokens=args.llm_max_tokens,
                    llm_thinking=args.llm_thinking or "disabled",
                    llm_json_mode=args.llm_json_mode,
                )
                if not args.skip_embedding_validation:
                    validate_embedding_runtime(manager, db, embedding_config)
                log_memory_index_state(db, f"before_context:{group_id}")
                replay_stats = replay_context_into_memory(
                    manager=manager,
                    db=db,
                    days=days,
                    group_id=group_id,
                    enable_reflect=args.enable_reflect,
                    reflect_every_days=args.reflect_every_days,
                    reflect_limit=args.reflect_limit,
                )
                log_memory_index_state(db, f"after_context:{group_id}")
                counts = db_counts(db)
                group_summary = {
                    "context_group_id": group_id,
                    "record_count": len(group_records),
                    "context_chars": len(context),
                    "db_path": str(db_path),
                    "context_log_path": str(context_log_path),
                    "days": len(days),
                    "replay": replay_stats,
                    "db_counts": counts,
                }
                group_summaries.append(group_summary)

                for record in group_records:
                    recall_context = manager.recall(
                        str(record["query"]),
                        top_k=args.recall_top_k,
                        budget=args.recall_budget,
                        recall_gate_mode=str(memory_config.get("recall_gate_mode") or "auto"),
                        recall_path=args.recall_path,
                    )
                    covered = gold_answer_in_context(record.get("answer", ""), recall_context)
                    hypothesis = ""
                    if args.run_reader:
                        hypothesis = run_reader(
                            manager,
                            str(record["query"]),
                            recall_context,
                            args.reader_max_context_chars,
                        )
                    result = {
                        "id": record["id"],
                        "source_line": record.get("_source_line"),
                        "context_group_id": group_id,
                        "query": record["query"],
                        "answer": record.get("answer", ""),
                        "hypothesis": hypothesis,
                        "answer_in_recall_context": covered,
                        "recall_path": args.recall_path,
                        "recall_context_chars": len(recall_context or ""),
                        "recall_context": recall_context,
                        "db_path": str(db_path),
                        "db_counts": counts,
                    }
                    result_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    result_file.flush()
                    result_count += 1
                    recall_nonempty += int(bool(recall_context))
                    gold_covered += int(covered)
                    reader_answered += int(bool(hypothesis))
                    logging.info(
                        "[%s/%s] id=%s group=%s recall_chars=%s answer_in_context=%s",
                        result_count,
                        len(records),
                        record["id"],
                        group_id,
                        len(recall_context or ""),
                        covered,
                    )
            finally:
                log_memory_index_state(db, f"finished_context:{group_id}")
                db.close()
                remove_context_log_handler(context_log_handler)

    summary = {
        "input": str(args.input.resolve()),
        "config": str(args.config.resolve()),
        "output_dir": str(output_dir),
        "results_path": str(results_path),
        "log_path": str(log_path),
        "records_processed": result_count,
        "context_groups": len(groups),
        "recall_nonempty": recall_nonempty,
        "gold_answer_in_recall_context": gold_covered,
        "reader_answered": reader_answered,
        "gold_coverage_rate": gold_covered / result_count if result_count else 0.0,
        "recall_path": args.recall_path,
        "enable_reflect": bool(args.enable_reflect),
        "group_summaries": group_summaries,
    }
    summary_path = output_dir / "clongeval_conversation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info(
        "Finished CLongEval records=%s groups=%s recall_nonempty=%s gold_coverage=%s/%s",
        result_count,
        len(groups),
        recall_nonempty,
        gold_covered,
        result_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
