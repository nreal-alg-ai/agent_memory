#!/usr/bin/env python3
"""Run memory-node store fact extraction on dialogue test samples.

Samples can come from a history_dialogue.json file or from
PYTHON_TEST_SAMPLES below. Both sources are normalized into the same turn
stream, fed to MemoryRuntime.store_interaction_turns(), and saved in an isolated
SessionDB under tmp/ by default. Fact extraction follows the batching interval
configured in the project-level config.yaml.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (SRC_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from memory.memory_manager import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    MemoryNodeManager,
    MemoryOperationReporter,
)
from memory.memory_database import SessionDB
from memory.memory_runtime import MemoryRuntime


DEFAULT_INPUT = Path("/Users/zhouboyu/Documents/agent_memory/test_data/user_dialogue/history_dialogue.json")
DEFAULT_OUTPUT_ROOT_DIR = REPO_ROOT / "tmp" / "memory_store_fact_test"

SAMPLE_ID_RE = re.compile(r"(?:^|_)sample(\d+)$")
_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

class StoreFactExtractionManager(MemoryNodeManager):
    """Use real retain extraction, but skip unrelated async graph work."""

    def __init__(
        self,
        *args: Any,
        report_rows: List[Dict[str, Any]],
        llm_max_tokens: int,
        llm_thinking: str,
        llm_json_mode: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.report_rows = report_rows
        self._test_llm_max_tokens = max(1, int(llm_max_tokens))
        self._test_llm_thinking = str(llm_thinking)
        self._test_llm_json_mode = bool(llm_json_mode)
        self._test_llm_call_count = 0

    def _call_llm(self, prompt: str) -> str | None:
        self._test_llm_call_count += 1
        call_kind = (
            "fact_extraction"
            if "记忆提炼模块" in prompt or "memory extraction module" in prompt
            else "state_update"
            if "state 更新模块" in prompt or "state update module" in prompt
            else "actionable_item_extraction"
            if "actionable item 提取模块" in prompt or "actionable-item extraction module" in prompt
            else "recall_analysis"
            if "recall query 分析器" in prompt or "recall query analyzer" in prompt
            else "other"
        )
        if (
            not self._llm_api_key
            or not self._llm_base_url
            or str(self._llm_base_url).strip().lower() == "none"
        ):
            logging.info(
                "LLM call #%s kind=%s skipped: missing llm_api_key or llm_base_url",
                self._test_llm_call_count,
                call_kind,
            )
            return None
        url = f"{self._llm_base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._llm_api_key:
            headers["Authorization"] = f"Bearer {self._llm_api_key}"
        payload = {
            "model": self._llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": self._test_llm_max_tokens,
            "stream": False,
        }
        if self._test_llm_thinking != "auto":
            payload["thinking"] = {"type": self._test_llm_thinking}
        if self._test_llm_json_mode:
            payload["response_format"] = {"type": "json_object"}
        logging.info(
            "LLM call #%s kind=%s model=%s prompt_chars=%s max_tokens=%s "
            "thinking=%s json_mode=%s",
            self._test_llm_call_count,
            call_kind,
            self._llm_model,
            len(prompt),
            self._test_llm_max_tokens,
            self._test_llm_thinking,
            self._test_llm_json_mode,
        )
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=self._llm_timeout)
            response.raise_for_status()
            response_data = response.json()
            choices = response_data.get("choices", [])
            if choices:
                choice = choices[0]
                content = choice.get("message", {}).get("content", "") or ""
                finish_reason = choice.get("finish_reason")
                logging.info(
                    "LLM result #%s kind=%s finish_reason=%s response_chars=%s usage=%s",
                    self._test_llm_call_count,
                    call_kind,
                    finish_reason,
                    len(content),
                    response_data.get("usage"),
                )
                # logging.info(
                #     "LLM raw response #%s kind=%s:\n%s",
                #     self._test_llm_call_count,
                #     call_kind,
                #     content,
                # )
                if finish_reason == "length":
                    logging.warning(
                        "LLM result #%s was truncated; increase --llm-max-tokens",
                        self._test_llm_call_count,
                    )
                return content
            logging.error(
                "LLM response #%s kind=%s contained no choices: %s",
                self._test_llm_call_count,
                call_kind,
                response_data,
            )
        except requests.exceptions.RequestException as exc:
            logging.error("LLM request failed for %s with model %s: %s", url, self._llm_model, exc)
        except (KeyError, ValueError, TypeError) as exc:
            logging.error("LLM response parse failed for %s with model %s: %s", url, self._llm_model, exc)
        return ""

    def _start_async_work(self, **kwargs: Any) -> None:
        return None


def strip_jsonc(text: str) -> str:
    """Remove JSONC comments and trailing commas without touching strings."""
    output: List[str] = []
    in_string = False
    escape = False
    idx = 0
    while idx < len(text):
        char = text[idx]
        next_char = text[idx + 1] if idx + 1 < len(text) else ""
        if in_string:
            output.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            idx += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            idx += 1
            continue
        if char == "/" and next_char == "/":
            idx += 2
            while idx < len(text) and text[idx] not in "\r\n":
                idx += 1
            continue
        if char == "/" and next_char == "*":
            idx += 2
            while idx + 1 < len(text) and not (text[idx] == "*" and text[idx + 1] == "/"):
                idx += 1
            idx += 2
            continue
        output.append(char)
        idx += 1
    return re.sub(r",\s*([}\]])", r"\1", "".join(output))


def sample_hour_offset(sample_id: str, fallback_offset: int) -> int:
    match = SAMPLE_ID_RE.search(sample_id)
    if match:
        return int(match.group(1))
    return fallback_offset


def flatten_dialogue_data(data: Any) -> List[Tuple[str, int, str, str, int, bool]]:
    """Normalize exported or in-file dialogue samples into store turns."""
    dialogue = data.get("dialogue") if isinstance(data, dict) else data
    if not isinstance(dialogue, list):
        raise ValueError("Expected JSON to contain a top-level dialogue list")

    turns: List[Tuple[str, int, str, str, int]] = []
    fallback_sample_offsets: Dict[str, int] = {}
    for group in dialogue:
        if not isinstance(group, dict):
            continue
        for sample_id, sample_turns in group.items():
            if not isinstance(sample_turns, list):
                continue
            sample_key = str(sample_id)
            if sample_key not in fallback_sample_offsets:
                fallback_sample_offsets[sample_key] = len(fallback_sample_offsets)
            hour_offset = sample_hour_offset(sample_key, fallback_sample_offsets[sample_key])
            for turn_index, turn in enumerate(sample_turns):
                if not isinstance(turn, dict):
                    continue
                user = str(turn.get("user") or "").strip()
                assistant = str(turn.get("assistant") or "").strip()
                if turn_index == (len(sample_turns) - 1):
                    is_last_turn = True
                else:
                    is_last_turn = False 
                if user and assistant:
                    turns.append((sample_key, turn_index, user, assistant, hour_offset, is_last_turn))
    return turns


def flatten_dialogue(path: Path) -> List[Tuple[str, int, str, str, int, bool]]:
    data = json.loads(strip_jsonc(path.read_text(encoding="utf-8")))
    return flatten_dialogue_data(data)


def load_python_test_samples() -> List[Dict[str, List[Dict[str, str]]]]:
    try:
        from scripts.test_data_sample import (
            AGENT_FIFTH,
            AGENT_FIRST,
            AGENT_FORTH,
            AGENT_SECOND,
            AGENT_THIRD,
            USER_FIFTH,
            USER_FIRST,
            USER_FORTH,
            USER_SECOND,
            USER_THIRD,
        )
    except ModuleNotFoundError as first_error:
        try:
            from test_data_sample import (
                AGENT_FIFTH,
                AGENT_FIRST,
                AGENT_FORTH,
                AGENT_SECOND,
                AGENT_THIRD,
                USER_FIFTH,
                USER_FIRST,
                USER_FORTH,
                USER_SECOND,
                USER_THIRD,
            )
        except ModuleNotFoundError as second_error:
            raise RuntimeError(
                "--sample-source python requires scripts/test_data_sample.py."
            ) from second_error

    return [
        {
            "python_sample1": [
                {"user": USER_FIRST, "assistant": AGENT_FIRST},
                {"user": USER_SECOND, "assistant": AGENT_SECOND},
                {"user": USER_THIRD, "assistant": AGENT_THIRD},
                {"user": USER_FORTH, "assistant": AGENT_FORTH},
                {"user": USER_FIFTH, "assistant": AGENT_FIFTH},
            ]
        }
    ]


def load_test_turns(
    source: str,
    input_path: Path,
) -> List[Tuple[str, int, str, str, int, bool]]:
    if source == "python":
        return flatten_dialogue_data(load_python_test_samples())
    return flatten_dialogue(input_path)


def iter_stored_nodes(db: SessionDB, start_id: int) -> Iterable[Dict[str, Any]]:
    rows = db._conn.execute(
        """SELECT id, episode_id, source_type, time_key, summary, keywords,
                  fact_type, fact_kind, fact_subject, entities,
                  fact_root_topic, fact_aspect_topic,
                  confidence, importance, metadata, created_at, updated_at
             FROM memory_facts
            WHERE id > ?
            ORDER BY id""",
        (start_id,),
    ).fetchall()
    for row in rows:
        item = dict(row)
        for key, default in (
            ("entities", []),
            ("metadata", {}),
        ):
            try:
                item[key] = json.loads(item.get(key) or json.dumps(default))
            except json.JSONDecodeError:
                item[key] = default
        yield item


def load_project_config(config_path: Path) -> Dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "YAML config support requires PyYAML. Install it with: "
            "micromamba run -n voice_recording pip install pyyaml"
        ) from exc
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping in config file: {config_path}")
    return _expand_env_refs(loaded)


def _expand_env_refs(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env_refs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_refs(item) for item in value]
    if isinstance(value, str):
        return _ENV_REF_RE.sub(lambda match: os.getenv(match.group(1), ""), value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise MemoryRuntime.store_interaction_turns fact extraction against JSON or in-file Python samples."
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--sample-source",
        choices=("json", "python"),
        default="json",
        help=(
            "Read --input as JSON, or use PYTHON_TEST_SAMPLES defined in this "
            "script. Defaults to json."
        ),
    )
    parser.add_argument("--output-root-dir", type=Path, default=DEFAULT_OUTPUT_ROOT_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Optional explicit run output directory. By default a run directory "
            "is created under --output-root-dir from the test dataset name and timestamp."
        ),
    )
    parser.add_argument("--db-name", default="memory_store_fact_test.db")
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Limit turns for smoke testing; 0 means all.")
    parser.add_argument("--start", type=int, default=0, help="Start offset in flattened turns.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output DB/report files.")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key")
    parser.add_argument("--llm-timeout", type=int)
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=8192,
        help="Maximum output tokens for each extraction/summary LLM call. Defaults to 8192.",
    )
    parser.add_argument(
        "--llm-thinking",
        choices=("disabled", "enabled", "auto"),
        default="disabled",
        help=(
            "Control provider thinking mode. Defaults to disabled because "
            "structured extraction does not need a large reasoning budget; "
            "auto omits the provider-specific parameter."
        ),
    )
    parser.add_argument(
        "--no-llm-json-mode",
        action="store_false",
        dest="llm_json_mode",
        help="Do not request provider-enforced JSON output.",
    )
    parser.set_defaults(llm_json_mode=True)
    parser.add_argument(
        "--min-dialogue-turns-before-store",
        type=int,
        help=(
            "Extract facts once per N completed turns. Defaults to "
            "memory.min_dialogue_turns_before_store from config.yaml."
        ),
    )
    parser.add_argument(
        "--max-dialogue-chars-before-store",
        type=int,
        help=(
            "Extract facts early when pending dialogue exceeds this many "
            "characters. Defaults to memory.max_dialogue_chars_before_store."
        ),
    )
    parser.add_argument(
        "--enable-reflect",
        action="store_true",
        help='Enable reflect',
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--manager-log-level", default="INFO")
    return parser.parse_args()


def _safe_output_name(value: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in str(value or "").strip()
    ).strip("._")
    return cleaned or "memory_store_fact_test"


def resolve_output_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    dataset_name = (
        args.input.stem
        if args.sample_source == "json"
        else "python_test_samples"
    )
    dataset_stem = _safe_output_name(dataset_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_root_dir = Path(args.output_root_dir or DEFAULT_OUTPUT_ROOT_DIR)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else output_root_dir / f"{dataset_stem}_{timestamp}"
    )
    args.output_root_dir = output_root_dir
    args.output_dir = output_dir
    args.log_path = (
        Path(args.log_path)
        if args.log_path
        else output_dir / "memory_store_extraction_test.log"
    )

    db_path = output_dir / args.db_name
    report_path = output_dir / "memory_store_fact_report.jsonl"
    return db_path, report_path


def remove_existing_outputs(db_path: Path, report_path: Path, overwrite: bool) -> None:
    related = [
        db_path,
        report_path,
    ]
    existing = [path for path in related if path.exists()]
    if existing and not overwrite:
        joined = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists. Pass --overwrite to replace:\n  {joined}")
    for path in existing:
        path.unlink()

def resolve_llm_args(args: argparse.Namespace) -> None:
    config = load_project_config(args.config)
    memory_config = config.get("memory", {}) if isinstance(config.get("memory"), dict) else {}
    args.llm_model = (
        args.llm_model
        or str(memory_config.get("llm_name") or "")
        or DEFAULT_LLM_MODEL
    )
    args.llm_base_url = (
        args.llm_base_url
        or str(memory_config.get("llm_base_url") or "")
        or DEFAULT_LLM_BASE_URL
    )
    if args.llm_base_url.rstrip("/") == "https://api.deepseek.com":
        args.llm_base_url = "https://api.deepseek.com/v1"
    args.llm_api_key = (
        args.llm_api_key
        or str(memory_config.get("llm_api_key") or "")
        or ""
    )
    args.llm_timeout = (
        args.llm_timeout
        or int(str(memory_config.get("llm_timeout", 120)))
    )
    args.min_dialogue_turns_before_store = max(
        1,
        int(
            args.min_dialogue_turns_before_store
            or memory_config.get("min_dialogue_turns_before_store", 1)
            or 1
        ),
    )
    args.max_dialogue_chars_before_store = max(
        1,
        int(
            args.max_dialogue_chars_before_store
            or memory_config.get("max_dialogue_chars_before_store", 2000)
            or 2000
        ),
    )

def configure_logging(log_path: Path, log_level: str, manager_log_level: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, str(log_level).upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    logging.getLogger("agent.memory_node_manager").setLevel(
        getattr(logging, str(manager_log_level).upper(), logging.INFO)
    )


def _configured_embedding_api_key_env(embedding_config: Dict[str, Any]) -> str:
    api_key_env = str(embedding_config.get("api_key_env") or "").strip()
    if api_key_env:
        return api_key_env
    api_key = str(embedding_config.get("api_key") or "").strip()
    match = re.fullmatch(r"\${([A-Za-z_][A-Za-z0-9_]*)}", api_key)
    return match.group(1) if match else ""


def validate_embedding_runtime(
    manager: MemoryNodeManager,
    db: SessionDB,
    embedding_config: Dict[str, Any],
) -> None:
    """Fail early when the configured embedding client cannot produce vectors."""
    del db
    logging.info(
        "Embedding runtime: executable=%s python=%s configured_provider=%s "
        "configured_model=%s configured_dimensions=%s",
        sys.executable,
        sys.version.split()[0],
        embedding_config.get("provider"),
        embedding_config.get("model"),
        embedding_config.get("dimensions"),
    )
    configured_api_key_env = _configured_embedding_api_key_env(embedding_config)
    if (
        configured_api_key_env
        and not os.getenv(configured_api_key_env)
        and not os.getenv("EMBEDDING_API_KEY")
    ):
        logging.warning(
            "Embedding API key environment variable is not set: "
            f"{configured_api_key_env}. config.yaml configures embedding.api_key "
            f"or embedding.api_key_env to use this variable. The agent_memory "
            f"EmbeddingClient will use deterministic local fallback vectors."
        )
    if not manager._ensure_embedding_client():
        raise RuntimeError("Failed to initialize the configured embedding client")

    probe = manager._embedding_client.embed_text(
        "memory embedding runtime validation"
    )
    if probe is None:
        raise RuntimeError(
            "The configured embedding provider returned no vector. Check "
            "config.yaml embedding credentials, base_url, and model."
        )
    probe_vector = manager._as_embedding_vector(probe)
    if probe_vector is None:
        raise RuntimeError("The configured embedding provider returned an invalid vector")
    expected_dim = int(getattr(manager._embedding_client, "dimension", probe_vector.size))
    if probe_vector.size != expected_dim:
        raise RuntimeError(
            "Embedding dimension mismatch: provider returned "
            f"{probe_vector.size}, but EmbeddingClient is configured for "
            f"{expected_dim}."
        )
    logging.info(
        "Embedding probe succeeded: dimensions=%s normalized_norm=%.6f",
        probe_vector.size,
        float((probe_vector @ probe_vector) ** 0.5),
    )


def log_memory_index_state(db: SessionDB, event: str) -> None:
    index_count = db._conn.execute(
        "SELECT COUNT(*) AS count FROM memory_index_entries"
    ).fetchone()["count"]
    fact_count = db._conn.execute(
        "SELECT COUNT(*) AS count FROM memory_facts"
    ).fetchone()["count"]
    state_count = db._conn.execute(
        "SELECT COUNT(*) AS count FROM memory_states"
    ).fetchone()["count"]
    actionable_count = db._conn.execute(
        "SELECT COUNT(*) AS count FROM memory_actionable_items"
    ).fetchone()["count"]
    logging.info(
        "Memory index state event=%s facts=%s states=%s actionable_items=%s index_entries=%s",
        event,
        fact_count,
        state_count,
        actionable_count,
        index_count,
    )


def main() -> int:
    args = parse_args()
    db_path, report_path = resolve_output_paths(args)
    resolve_llm_args(args)
    configure_logging(args.log_path, args.log_level, args.manager_log_level)
    logging.info("Loaded memory store test config from: %s", args.config)
    
    turns = load_test_turns(args.sample_source, args.input)
    if args.start:
        turns = turns[args.start :]
    if args.limit:
        turns = turns[: args.limit]
    if not turns:
        raise RuntimeError("No user/assistant turns found in input")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    remove_existing_outputs(db_path, report_path, args.overwrite)

    report_rows: List[Dict[str, Any]] = []
    db = SessionDB(db_path=db_path)
    config = load_project_config(args.config)
    embedding_config = (
        dict(config.get("embedding") or {})
        if isinstance(config.get("embedding"), dict)
        else {}
    )
    memory_config = (
        dict(config.get("memory") or {})
        if isinstance(config.get("memory"), dict)
        else {}
    )
    memory_config["min_dialogue_turns_before_store"] = args.min_dialogue_turns_before_store
    memory_config["max_dialogue_chars_before_store"] = args.max_dialogue_chars_before_store
    memory_config["llm_timeout"] = args.llm_timeout
    memory_config["enable_entity_extraction"] = False
    operation_reporter = MemoryOperationReporter()
    manager = StoreFactExtractionManager(
        db,
        embedding_config=embedding_config,
        memory_config=memory_config,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        operation_reporter=operation_reporter,
        report_rows=report_rows,
        llm_max_tokens=args.llm_max_tokens,
        llm_thinking=args.llm_thinking,
        llm_json_mode=args.llm_json_mode,
    )
    runtime = MemoryRuntime(manager, memory_config=memory_config)
    try:
        validate_embedding_runtime(manager, db, embedding_config)
        log_memory_index_state(db, "initialized")
    except Exception:
        db.close()
        raise

    stored_turns = 0
    stored_facts = 0
    reflect_runs = 0
    base_turn_timestamp = datetime.now().astimezone()
    try:
        with report_path.open("w", encoding="utf-8") as report:
            for local_index, (sample_id, turn_index, user, assistant, hour_offset, is_last_turn) in enumerate(turns):
                flat_index = args.start + local_index
                # Keep sample-level spacing at one hour for reflect tests, while
                # giving each turn a unique timestamp so memory_nodes.time_key
                # does not collide on "#00" across turns in the same sample.
                turn_timestamp = base_turn_timestamp + timedelta(hours=hour_offset, seconds=turn_index)
                before_id = db._conn.execute("SELECT COALESCE(MAX(id), 0) AS max_id FROM memory_facts").fetchone()["max_id"]
                pending_before = [dict(turn) for turn in runtime._pending_interaction_turns]
                pending_with_current = pending_before + [{
                    "user_message": user,
                    "assistant_response": assistant,
                }]
                pending_character_count = (
                    runtime.interaction_turns_character_count(pending_with_current)
                )
                extraction_due = runtime.should_flush_pending_interaction_turns(
                    pending_with_current
                )
                ok = runtime.store_interaction_turns(
                    user,
                    assistant,
                    tags=["store_fact_test", f"sample:{sample_id}", f"turn:{turn_index}"],
                    turn_timestamp=turn_timestamp,
                )
                if ok.get("queued") and not runtime.flush_store_queue():
                    raise RuntimeError("Timed out while draining queued memory stores")
                nodes = list(iter_stored_nodes(db, before_id))
                store_operation_report = operation_reporter.latest_report("store_episode")
                if ok.get("queued"):
                    stored_turns += 1
                    stored_facts += len(nodes)
                row = {
                    "event": "store_turn",
                    "flat_index": flat_index,
                    "sample_id": sample_id,
                    "turn_index": turn_index,
                    "sample_hour_offset": hour_offset,
                    "turn_second_offset": turn_index,
                    "turn_timestamp": turn_timestamp.isoformat(),
                    "min_dialogue_turns_before_store": runtime._min_dialogue_turns_before_store,
                    "max_dialogue_chars_before_store": runtime._max_dialogue_chars_before_store,
                    "pending_character_count": pending_character_count,
                    "fact_extraction_due": extraction_due,
                    "source_turn_count": (
                        len(pending_before) + 1 if extraction_due else 0
                    ),
                    "pending_turn_count": len(runtime._pending_interaction_turns),
                    "queued": bool(ok.get("queued")),
                    "store_total_elapsed_ms": float(
                        store_operation_report.get("elapsed_ms") or 0.0
                    ),
                    "fact_count": len(nodes),
                    "user": user,
                    "assistant": assistant,
                    "facts": nodes,
                }
                report.write(json.dumps(row, ensure_ascii=False) + "\n")
                report.flush()
                logging.info(
                    "[%s/%s] %s turn=%s extraction_due=%s "
                    "source_turns=%s pending=%s stored=%s facts=%s",
                    flat_index - args.start + 1,
                    len(turns),
                    sample_id,
                    turn_index,
                    extraction_due,
                    len(pending_before) + 1 if extraction_due else 0,
                    len(runtime._pending_interaction_turns),
                    ok,
                    len(nodes),
                )
                
                if args.enable_reflect and is_last_turn:
                    log_memory_index_state(db, f"before_reflect:{sample_id}")
                    logging.info(
                        "Running reflect after sample %s",
                        sample_id,
                    )
                    reflect_submit = runtime.reflect_async(
                        reflect_timestamp=turn_timestamp,
                    )
                    if reflect_submit.get("accepted") and not runtime.flush_store_queue():
                        raise RuntimeError("Timed out while draining queued memory reflect")
                    reflect_report = (
                        operation_reporter.latest_report("reflect")
                        or reflect_submit
                    )
                    reflect_row = {
                        "event": "reflect",
                        "after_sample_id": sample_id,
                        "after_flat_index": flat_index,
                        "report": reflect_report,
                        "submit_report": reflect_submit,
                    }
                    report.write(json.dumps(reflect_row, ensure_ascii=False, default=str) + "\n")
                    report.flush()
                    logging.info(
                        "Reflect after %s states_updated=%s actionable_items_updated=%s",
                        sample_id,
                        reflect_report.get("states_updated"),
                        reflect_report.get("actionable_items_updated"),
                    )
    finally:
        log_memory_index_state(db, "finished")
        db.close()

    memory_operation_report = operation_reporter.snapshot()
    operation_counts = memory_operation_report.get("counts") or {}
    store_operation_report = operation_counts.get("store_episode") or {}
    reflect_operation_report = operation_counts.get("reflect") or {}
    summary = {
        "sample_source": args.sample_source,
        "input": str(args.input) if args.sample_source == "json" else "PYTHON_TEST_SAMPLES",
        "output_root_dir": str(args.output_root_dir),
        "output_dir": str(args.output_dir),
        "log_path": str(args.log_path),
        "db_path": str(db_path),
        "report_path": str(report_path),
        "turns_processed": len(turns),
        "turns_with_facts": stored_turns,
        "facts_stored": stored_facts,
        "min_dialogue_turns_before_store": runtime._min_dialogue_turns_before_store,
        "max_dialogue_chars_before_store": runtime._max_dialogue_chars_before_store,
        "pending_turns": len(runtime._pending_interaction_turns),
        "store_episode_submitted": int(store_operation_report.get("submitted") or 0),
        "store_episode_succeeded": int(store_operation_report.get("succeeded") or 0),
        "store_total_elapsed_ms": float(store_operation_report.get("total_elapsed_ms") or 0.0),
        "reflect_runs": int(reflect_operation_report.get("submitted") or 0),
        "reflect_total_elapsed_ms": float(reflect_operation_report.get("total_elapsed_ms") or 0.0),
        "llm_model": args.llm_model,
        "llm_base_url": args.llm_base_url,
        "llm_max_tokens": args.llm_max_tokens,
        "llm_thinking": args.llm_thinking,
        "llm_json_mode": args.llm_json_mode,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
