"""Plan-first recall of direct memory objects, without evidence expansion."""

from __future__ import annotations

import json
import logging
import math
import time
from contextlib import ExitStack
from datetime import datetime
from typing import Any, Mapping, Optional

import numpy as np

from intent_lifecycle import ACTIVE_USER_INTENT_STATUSES, USER_INTENT_STATUSES
from memory.memory_database import UserMemoryDB, USER_THREAD_STATUSES
from recall_document_store import _lexical_index_text
from user_model.user_model_database import UserModelDB
from user_model.user_state_schema import USER_STATE_TYPES, USER_STATE_KINDS_BY_TYPE, USER_STATE_STATUSES
from prompts_en import RECALL_QUERY_ANALYSIS_PROMPT_EN
from prompts_zh import RECALL_QUERY_ANALYSIS_PROMPT_ZH
from pwm_logging import as_pwm_logger
from semantic_domains import SEMANTIC_DOMAINS
from storage_utils import normalize_reference_timestamp


class MemoryRecall:
    """Analyze once, validate the plan, then retrieve only requested types."""

    DATA_TYPES = {"fact", "user_activity", "user_intent", "user_observed_pattern", "user_state", "user_thread"}
    TIME_FIELDS = {
        "fact": {"event_time", "dialogue_time", "both"},
        "user_activity": {"activity_time"},
        "user_intent": {"scheduled_time"},
    }
    RESULT_FIELDS = {
        "fact": ("summary", "event_time_key", "dialogue_time_key", "confidence"),
        "user_activity": ("activity_text", "user_role", "started_at", "ended_at", "confidence"),
        "user_intent": ("summary", "current_progress", "status", "time_start", "time_end",
                        "primary_domain", "details", "confidence"),
        "user_observed_pattern": ("pattern_text", "condition_text", "primary_domain", "confidence"),
        "user_thread": ("title", "identity_description", "current_summary", "status"),
        "user_state": ("state_domain", "state_type", "state_kind", "state_topic", "state_text",
                       "conditions", "status", "valid_from", "valid_to", "confidence", "last_observed_at"),
    }
    TYPE_DESCRIPTIONS = {
        "zh": {
            "fact": "从任意说话人的对话中提炼的有价值信息，可包含事实、观点或判断，不一定描述用户本人。",
            "user_activity": "采集窗口内用户本人实际参与的活动及其角色，不等同于谈话中提到的全部行动。",
            "user_thread": "用户本人参与的持续事项及其身份、最新进展；不是完整历史记录。",
            "user_intent": "用户本人的未来事项及后续进展；按 status 区分未完成、已完成或取消。",
            "user_observed_pattern": "从用户实际行为证据归纳的重复规律，不代表每次都如此，需保留适用条件。",
            "user_state": "对用户本人稳定倾向、重复行为或当前个人处境的综合判断，需保留条件、状态与置信度。",
        },
        "en": {
            "fact": "Valuable information extracted from any speaker, including facts, opinions or judgments; not necessarily about user self.",
            "user_activity": "User self's actual participation and role during an observation window, not every action mentioned in dialogue.",
            "user_thread": "The identity and latest progress of an ongoing matter involving user self, not its complete history.",
            "user_intent": "User self's future matter and subsequent progress; status distinguishes unfinished, completed or cancelled matters.",
            "user_observed_pattern": "A recurring pattern inferred from actual behavior, conditional rather than universally true.",
            "user_state": "An integrated judgment of user self's stable tendencies, recurring behavior or current circumstances; retain conditions, status and confidence.",
        },
    }
    BUSINESS_RELATIONSHIPS = (
        (("fact", "user_activity"), "fact 描述具体内容，activity 描述当时的实际参与；谈到某个行动不代表当时执行了它。",
         "Facts describe content; activities describe actual participation. Mentioning an action does not establish its execution."),
        (("user_activity", "user_thread"), "activity 是窗口内的参与记录，thread 组织持续事项的最新情况。",
         "Activities record participation in a window; threads organize the latest situation of an ongoing matter."),
        (("user_activity", "user_intent"), "activity 可提供 intent 的进展线索，但参与相关讨论不等于履行或完成 intent。",
         "Activities may inform intent progress, but participating in related discussions does not establish fulfillment or completion."),
        (("user_thread", "user_intent"), "thread 描述持续事项，intent 描述个人未来要做的事；事项整体进展不等于个人待办完成。",
         "Threads describe ongoing matters; intents describe personal future actions. Overall project progress does not establish completion of a personal task."),
        (("fact", "user_state"), "fact 是具体信息，state 是个人状态判断；单次表达或项目要求不自动构成稳定个人特征。",
         "Facts are concrete information; states are personal judgments. A single statement or project requirement does not automatically establish a stable personal trait."),
        (("user_thread", "user_state"), "thread 描述事项进展，state 描述用户本人处境；项目技术细节不是个人状态。",
         "Threads describe matter progress; states describe user self's circumstances. Project technical details are not personal states."),
        (("user_observed_pattern", "user_state"), "pattern 是行为证据归纳的规律，可用于更新 state，但不等同于完整个人状态。",
         "Observed patterns are behavioral regularities that may inform state updates, not complete personal states themselves."),
    )

    def __init__(
        self, *, memory_db: UserMemoryDB, llm_client: Any, embedding_client: Any = None,
        model_db: Optional[UserModelDB] = None,
        config: Optional[Mapping[str, Any]] = None, enabled: bool = True,
        prompt_language_mode: str = "source", operation_reporter: Any = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._db = memory_db
        self._model_db = model_db
        self._llm_client = llm_client
        self._embedding_client = embedding_client
        self._enabled = enabled
        self._prompt_language_mode = prompt_language_mode
        self._reporter = operation_reporter
        self._logger = as_pwm_logger(logger or logging.getLogger(__name__))
        settings = dict(config or {})
        self._top_k = max(1, min(100, int(settings.get("recall_top_k", 8))))
        self._candidate_limit = max(self._top_k, min(500, int(settings.get("recall_candidate_limit_per_type", 40))))
        budget = str(settings.get("recall_budget", "mid"))
        self._context_budget = max(128, int(settings.get("recall_context_max_chars", {
            "low": 3200, "mid": 6000, "high": 10000,
        }.get(budget, 6000))))
        self._entry_budget = max(80, int(settings.get("recall_entry_max_chars", 1200)))
        self._input_context_budget = max(0, int(settings.get("recall_input_context_max_chars", 4000)))
        self._min_similarity = float(settings.get("recall_min_embedding_similarity", .3))
        if not 0 <= self._min_similarity <= 1:
            raise ValueError("recall_min_embedding_similarity must be between 0 and 1")

    def recall(
        self, query: str, *, context: Optional[str] = None,
        reference_timestamp: Any = None, tags: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        report: dict[str, Any] = {"status": "empty", "memory_context": "", "items": [],
                                  "retrieval_plan": None, "retrieval_strategy": "planned"}
        try:
            if not isinstance(query, str) or (context is not None and not isinstance(context, str)):
                raise ValueError("query and context must be text")
            if self._enabled and query.strip():
                reference = normalize_reference_timestamp(
                    datetime.now().astimezone() if reference_timestamp is None else reference_timestamp,
                )
                recent_context = (context or "")[-self._input_context_budget:] if self._input_context_budget else ""
                self._logger.log_json_format("memory_recall", "start", {
                    "query": query, "context": recent_context, "reference_timestamp": reference, "tags": tags or [],
                })
                plan = self._analyze_query(query, recent_context, reference)
                report["retrieval_plan"] = plan
                vectors = self._embed_queries(plan["requests"])
                # Open the WAL snapshot after network calls; do not share the writer's
                # connection or keep a read transaction open during query analysis.
                if plan["requests"]:
                    with ExitStack() as stack:
                        readers = {}
                        groups: dict[str, list[dict[str, Any]]] = {}
                        for request in plan["requests"]:
                            database = self._model_db if request["data_type"] == "user_state" else self._db
                            if database not in readers:
                                readers[database] = stack.enter_context(database.reader_transaction())
                            reader = readers[database]
                            items = self._retrieve_request(reader, request, vectors.get(request["search_query"]), reference)
                            groups.setdefault(request["data_type"], []).extend(items)
                        report["items"], report["memory_context"] = self._organize_results(
                            groups, english=self._use_english(query + recent_context),
                        )
                report["status"] = "ok" if report["items"] else "empty"
                report["reference_timestamp"] = reference
        except Exception as exc:
            report.update(status="error", error=f"{type(exc).__name__}: {exc}")
            self._logger.exception("Memory recall failed")
        report["elapsed_ms"] = round((time.monotonic() - started) * 1000, 2)
        report["recall_context_chars"] = len(report["memory_context"])
        self._logger.log_json_format("memory_recall", "finish", report)
        if self._reporter is not None:
            self._reporter.on_recall_finished(report)
        return report

    def _use_english(self, text: str) -> bool:
        english = str(self._prompt_language_mode).lower() in {"en", "english", "force_en"}
        if str(self._prompt_language_mode).lower() == "source":
            english = not any("\u4e00" <= char <= "\u9fff" for char in text)
        return english

    def _analyze_query(self, query: str, context: str, reference: str) -> dict[str, Any]:
        english = self._use_english(query + context)
        template = RECALL_QUERY_ANALYSIS_PROMPT_EN if english else RECALL_QUERY_ANALYSIS_PROMPT_ZH
        prompt = template.replace("{recall_input}", json.dumps({
            "context": context, "query": query, "reference_timestamp": reference,
        }, ensure_ascii=False, separators=(",", ":")))
        raw = self._llm_client.call_llm(prompt, stage_name="memory_recall_query_analysis")
        parsed = self._llm_client.parse_json_object_from_llm_text(raw or "", stage_name="memory_recall_query_analysis")
        plan = self._validate_plan(parsed, reference)
        self._logger.log_json_format("memory_recall", "plan", plan)
        return plan

    def _validate_plan(self, value: Any, reference: str) -> dict[str, Any]:
        if (not isinstance(value, dict) or set(value) != {"resolved_query", "requests"}
                or not isinstance(value["resolved_query"], str) or not value["resolved_query"].strip()
                or not isinstance(value["requests"], list) or len(value["requests"]) > 8):
            raise ValueError("invalid_recall_plan")
        requests = []
        for request in value["requests"]:
            if not isinstance(request, dict) or set(request) != {"data_type", "search_query", "filters"}:
                raise ValueError("invalid_recall_request")
            kind = request["data_type"]
            search = request["search_query"]
            filters = request["filters"]
            if (not isinstance(kind, str) or kind not in self.DATA_TYPES or not isinstance(search, str)
                    or len(search) > 2000 or not isinstance(filters, dict)):
                raise ValueError("invalid_recall_request_type_or_query")
            allowed = set()
            if kind in self.TIME_FIELDS:
                allowed.update({"time_start", "time_end", "time_field"})
            if kind in {"user_intent", "user_observed_pattern"}:
                allowed.add("primary_domain")
            if kind in {"user_intent", "user_thread"}:
                allowed.add("statuses")
            if kind == "user_state":
                if self._model_db is None:
                    raise ValueError("user_model_database_not_configured")
                allowed.update({"statuses", "state_domain", "state_type", "state_kind"})
            if set(filters) - allowed:
                raise ValueError("unsupported_recall_filter")
            filters = {key: val for key, val in filters.items() if val is not None}
            if "statuses" in filters:
                statuses = filters["statuses"]
                allowed_statuses = {"user_state": USER_STATE_STATUSES, "user_intent": USER_INTENT_STATUSES,
                                    "user_thread": USER_THREAD_STATUSES}[kind]
                if (not isinstance(statuses, list) or not statuses
                        or any(not isinstance(s, str) or s not in allowed_statuses for s in statuses)):
                    raise ValueError("invalid_recall_statuses")
                filters["statuses"] = sorted(set(statuses))
            elif kind == "user_intent":
                filters["statuses"] = sorted(ACTIVE_USER_INTENT_STATUSES)
            elif kind == "user_state":
                filters["statuses"] = ["active", "weakened"]
            elif kind == "user_thread":
                filters["statuses"] = sorted(USER_THREAD_STATUSES)
            if "state_domain" in filters and filters["state_domain"] not in SEMANTIC_DOMAINS:
                raise ValueError("invalid_recall_state_domain")
            if "state_type" in filters and filters["state_type"] not in USER_STATE_TYPES:
                raise ValueError("invalid_recall_state_type")
            if "state_kind" in filters:
                kinds = USER_STATE_KINDS_BY_TYPE.get(filters.get("state_type"))
                if kinds is None:
                    kinds = set().union(*USER_STATE_KINDS_BY_TYPE.values())
                if filters["state_kind"] not in kinds:
                    raise ValueError("invalid_recall_state_kind")
            if "primary_domain" in filters and filters["primary_domain"] not in SEMANTIC_DOMAINS:
                raise ValueError("invalid_recall_domain")
            if "time_field" in filters and filters["time_field"] not in self.TIME_FIELDS[kind]:
                raise ValueError("invalid_recall_time_field")
            for key in ("time_start", "time_end"):
                if key in filters:
                    if not isinstance(filters[key], str) or not filters[key].strip():
                        raise ValueError("invalid_recall_time_bound")
                    bound = datetime.fromisoformat(filters[key].replace("Z", "+00:00"))
                    if bound.tzinfo is None:
                        bound = bound.replace(tzinfo=datetime.fromisoformat(reference).tzinfo)
                    filters[key] = bound.isoformat()
            if ("time_start" in filters and "time_end" in filters
                    and datetime.fromisoformat(filters["time_start"]) >= datetime.fromisoformat(filters["time_end"])):
                raise ValueError("invalid_recall_time_range")
            if not search.strip() and not any(k in filters for k in ("statuses", "time_start", "time_end", "primary_domain")):
                raise ValueError("unbounded_empty_recall_query")
            normalized = {"data_type": kind, "search_query": search.strip(), "filters": filters}
            if normalized not in requests:
                requests.append(normalized)
        return {"resolved_query": value["resolved_query"].strip(), "requests": requests}

    def _embed_queries(self, requests: list[dict[str, Any]]) -> dict[str, Any]:
        texts = list(dict.fromkeys(r["search_query"] for r in requests if r["search_query"]))
        if self._embedding_client is None or not texts:
            return {}
        try:
            vectors = self._embedding_client.embed_batch(texts)
            if len(vectors) != len(texts):
                raise ValueError("query_embedding_batch_length_mismatch")
            return dict(zip(texts, vectors))
        except Exception:
            self._logger.warning("Recall query embeddings unavailable; using lexical retrieval", exc_info=True)
            return {}

    def _retrieve_request(self, db, request: dict[str, Any], vector: Any, reference: str) -> list[dict[str, Any]]:
        kind, filters = request["data_type"], request["filters"]
        terms = list(dict.fromkeys(_lexical_index_text(request["search_query"]).split()))[:32]
        # A nonempty but untokenizable query must not turn into a recent-items scan.
        lexical = db.retrieve_recall_documents(
            object_type=kind, filters=filters, terms=terms, limit=self._candidate_limit,
            reference_timestamp=reference,
        ) if terms or not request["search_query"] else []
        semantic = []
        if vector is not None and request["search_query"]:
            query_vector = np.asarray(vector).reshape(-1)
            norm = np.linalg.norm(query_vector)
            for row in db.retrieve_recall_documents(
                object_type=kind, filters=filters, embeddings_only=True, reference_timestamp=reference,
            ):
                stored = row["identity_text_embedding"]
                stored = np.asarray(stored).reshape(-1)
                if stored.shape != query_vector.shape or not norm or not np.linalg.norm(stored):
                    continue
                similarity = float(np.dot(query_vector, stored) / (norm * np.linalg.norm(stored)))
                if math.isfinite(similarity) and similarity >= self._min_similarity:
                    semantic.append((similarity, row))
            semantic.sort(key=lambda pair: (-pair[0], pair[1]["object_id"]))
        scores: dict[int, float] = {}
        for channel in (lexical, [row for _, row in semantic[:self._candidate_limit]]):
            for rank, row in enumerate(channel, 1):
                key = int(row["object_id"])
                scores[key] = scores.get(key, 0) + 1 / (60 + rank)
        ranked = sorted(scores, key=lambda key: (-scores[key], key))[:self._candidate_limit]
        loader = getattr(db, {"fact": "get_memory_facts_by_ids", "user_activity": "get_user_activity_items_by_ids",
                              "user_intent": "get_user_intent_items_by_ids",
                              "user_observed_pattern": "get_user_observed_patterns_by_ids",
                              "user_thread": "get_user_threads_by_ids",
                              "user_state": "get_user_states_by_ids"}[kind])
        rows = {int(row["id"]): row for row in loader(ranked)}
        items = [{"object_type": kind, "object_id": key, "data": {
            field: rows[key].get(field) for field in self.RESULT_FIELDS[kind]
        }, "_rank_score": scores[key]} for key in ranked if key in rows]
        self._logger.log_json_format("memory_recall", "retrieved", {
            "request": request, "lexical_count": len(lexical), "semantic_count": len(semantic),
            "object_ids": [item["object_id"] for item in items],
        })
        return items

    def _format_memory_context_guidance(self, kinds: list[str], *, english: bool) -> str:
        """Explain returned types without asserting links between concrete records."""
        language = "en" if english else "zh"
        lines = ["[Memory type meanings]" if english else "【记忆数据含义】"]
        lines.extend(f"{kind}: {self.TYPE_DESCRIPTIONS[language][kind]}" for kind in kinds)
        relationships = [en if english else zh for types, zh, en in self.BUSINESS_RELATIONSHIPS
                         if all(kind in kinds for kind in types)]
        if relationships:
            lines.extend(["[Business relationships between types]" if english else "【数据类型之间的业务关系】",
                          *relationships])
        lines.append(
            "These are type-level explanations, not verified links between these records. Similar content across levels is not independent corroboration."
            if english else "以上仅说明数据类型分工，不代表本次记录间存在已核实关联；不同层级的相似内容不能视为独立佐证。"
        )
        lines.append("[Retrieved memories]" if english else "【召回记忆】")
        return "\n".join(lines) + "\n"

    def _organize_results(
        self, groups: dict[str, list[dict[str, Any]]], *, english: bool = False,
    ) -> tuple[list[dict[str, Any]], str]:
        queues = []
        for group in groups.values():
            unique = {}
            for item in group:
                old = unique.get(item["object_id"])
                if old is None or item["_rank_score"] > old["_rank_score"]:
                    unique[item["object_id"]] = item
            queues.append(sorted(unique.values(), key=lambda item: (-item["_rank_score"], item["object_id"])))
        result, lines, size = [], [], 0
        # Interleave types rather than comparing raw scores across different types.
        while any(queues) and len(result) < self._top_k:
            for queue in queues:
                if not queue or len(result) >= self._top_k:
                    continue
                item = queue.pop(0)
                item = {key: value for key, value in item.items() if key != "_rank_score"}
                data = dict(item["data"])
                item["data"] = data
                entry_budget = min(self._entry_budget, self._context_budget - size - bool(lines))
                line = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                text_keys = ("summary", "activity_text", "pattern_text", "current_progress", "state_text",
                             "title", "identity_description", "current_summary")
                while len(line) > entry_budget:
                    reducible = [key for key in text_keys if isinstance(data.get(key), str) and len(data[key]) > 41]
                    if not reducible:
                        break
                    key = max(reducible, key=lambda key: len(data[key]))
                    keep = max(40, len(data[key]) - (len(line) - entry_budget) - 20)
                    data[key] = data[key][:keep] + "…"
                    item["truncated"] = True
                    line = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                extra = len(line) + bool(lines)
                if len(line) > entry_budget:
                    continue
                result.append(item)
                lines.append(line)
                size += extra
        if not result:
            return [], ""
        kinds = list(dict.fromkeys(item["object_type"] for item in result))
        guidance = self._format_memory_context_guidance(kinds, english=english)
        return result, guidance + "\n".join(lines)
