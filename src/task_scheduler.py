"""Priority execution and operation reporting; FIFO within each priority."""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from concurrent.futures import Future
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

from task_execution_database import TaskExecutionDB

if TYPE_CHECKING:
    from memory.memory_database import UserMemoryDB


TaskHandler = Callable[..., Any]


def _report_now_text() -> str:
    return datetime.now().astimezone().isoformat()


def _report_timestamp_text(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


class TaskOperationReporter:
    """Collect asynchronous memory operation results for benchmark callers."""

    def __init__(self, *, recent_task_limit: int = 200) -> None:
        self._lock = threading.Lock()
        self._task_seq = 0
        self._recent_task_limit = max(1, int(recent_task_limit or 200))
        self._counts: Dict[str, Dict[str, Any]] = {}
        self._latest_reports: Dict[str, Dict[str, Any]] = {}
        self._recent_tasks: List[Dict[str, Any]] = []

    @staticmethod
    def _empty_counts() -> Dict[str, Any]:
        return {
            "submitted": 0,
            "completed": 0,
            "succeeded": 0,
            "failed": 0,
            "rejected": 0,
            "inflight": 0,
            "total_elapsed_ms": 0.0,
        }

    def next_task_id(self, operation_type: str) -> str:
        clean_type = str(operation_type or "memory_task").strip() or "memory_task"
        with self._lock:
            self._task_seq += 1
            return f"{clean_type}-{self._task_seq}"

    def on_task_submitted(
        self,
        *,
        operation_type: str,
        task_id: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        safe_payload = self._json_compatible(payload or {})
        with self._lock:
            counts = self._counts.setdefault(operation_type, self._empty_counts())
            counts["submitted"] += 1
            counts["inflight"] += 1
            self._append_recent_locked({
                "event": "submitted",
                "operation_type": operation_type,
                "task_id": task_id,
                "payload": safe_payload,
                "timestamp": _report_now_text(),
            })

    def on_task_rejected(
        self,
        *,
        operation_type: str,
        task_id: str,
        reason: str,
    ) -> None:
        report = {
            "accepted": False,
            "status": "rejected",
            "reason": reason,
            "task_id": task_id,
        }
        with self._lock:
            counts = self._counts.setdefault(operation_type, self._empty_counts())
            counts["rejected"] += 1
            self._latest_reports[operation_type] = dict(report)
            self._append_recent_locked({
                "event": "rejected",
                "operation_type": operation_type,
                "task_id": task_id,
                "reason": reason,
                "timestamp": _report_now_text(),
            })

    def on_task_finished(
        self,
        *,
        operation_type: str,
        task_id: str,
        started_at: float,
        result: Any = None,
        error: Optional[BaseException] = None,
    ) -> bool:
        elapsed_ms = round((time.monotonic() - started_at) * 1000, 2)
        succeeded = error is None and self._operation_succeeded(operation_type, result)
        report = self._operation_result_report(
            operation_type=operation_type,
            task_id=task_id,
            result=result,
            error=error,
            succeeded=succeeded,
            elapsed_ms=elapsed_ms,
        )
        with self._lock:
            counts = self._counts.setdefault(operation_type, self._empty_counts())
            counts["completed"] += 1
            counts["inflight"] = max(0, int(counts.get("inflight") or 0) - 1)
            counts["total_elapsed_ms"] = round(
                float(counts.get("total_elapsed_ms") or 0.0) + elapsed_ms,
                2,
            )
            counts["succeeded" if succeeded else "failed"] += 1
            self._latest_reports[operation_type] = dict(report)
            recent_event = {
                "event": "finished",
                "operation_type": operation_type,
                "task_id": task_id,
                "status": report.get("status"),
                "succeeded": succeeded,
                "result": self._json_compatible(report),
                "elapsed_ms": elapsed_ms,
                "timestamp": _report_now_text(),
            }
            if error is not None:
                recent_event["error"] = str(error)
                recent_event["error_type"] = type(error).__name__
            self._append_recent_locked(recent_event)
        return succeeded

    def on_recall_finished(self, report: Dict[str, Any]) -> None:
        elapsed_ms = float(report.get("elapsed_ms") or 0.0)
        status = str(report.get("status") or "").strip().lower()
        succeeded = status not in {"error", "failed"}
        recall_report = {
            key: value
            for key, value in report.items()
            if key != "memory_context"
        }
        with self._lock:
            counts = self._counts.setdefault("recall", self._empty_counts())
            counts["submitted"] += 1
            counts["completed"] += 1
            counts["total_elapsed_ms"] = round(
                float(counts.get("total_elapsed_ms") or 0.0) + elapsed_ms,
                2,
            )
            counts["succeeded" if succeeded else "failed"] += 1
            self._latest_reports["recall"] = recall_report
            self._append_recent_locked({
                "event": "finished",
                "operation_type": "recall",
                "task_id": f"recall-{counts['completed']}",
                "status": recall_report.get("status"),
                "actual_recall_mode": recall_report.get("actual_recall_mode"),
                "elapsed_ms": elapsed_ms,
                "timestamp": _report_now_text(),
            })

    def operation_report(self, operation_type: str) -> Dict[str, Any]:
        with self._lock:
            counts = dict(
                self._counts.get(operation_type) or self._empty_counts()
            )
            latest = self._latest_reports.get(operation_type)
        if latest:
            counts["latest_report"] = dict(latest)
        return counts

    def latest_report(self, operation_type: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest_reports.get(operation_type) or {})

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "counts": {
                    key: dict(value)
                    for key, value in self._counts.items()
                },
                "latest_reports": {
                    key: dict(value)
                    for key, value in self._latest_reports.items()
                },
                "recent_tasks": [dict(item) for item in self._recent_tasks],
            }

    def _append_recent_locked(self, event: Dict[str, Any]) -> None:
        self._recent_tasks.append(event)
        if len(self._recent_tasks) > self._recent_task_limit:
            del self._recent_tasks[: len(self._recent_tasks) - self._recent_task_limit]

    @classmethod
    def _json_compatible(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            return _report_timestamp_text(value)
        if isinstance(value, dict):
            return {
                str(key): cls._json_compatible(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._json_compatible(item) for item in value]
        return str(value)

    @staticmethod
    def _operation_succeeded(operation_type: str, result: Any) -> bool:
        failure_statuses = {
            "failed",
            "skipped",
            "error",
            "queue_rejected",
        }
        if operation_type == "memory_fact_extraction":
            if not isinstance(result, dict):
                return bool(result)
            return (
                str(result.get("status") or "ok").strip().lower()
                not in failure_statuses
            )
        if operation_type in {
            "memory_flush_signal_updates",
            "memory_user_activity_extraction",
            "memory_user_thread_update",
            "memory_user_claim_update",
            "memory_user_observed_pattern_update",
            "memory_user_intent_update",
            "memory_user_intent_update_from_activities",
            "user_state_update_from_user_claims",
            "user_state_update_from_user_threads",
            "user_state_update_from_user_observed_patterns",
            "user_prediction_update_from_intent_events",
            "user_prediction_reassessment",
            "user_prediction_window_expiration",
            "user_proactive_policy_evaluation",
            "proactive_service_feedback",
        }:
            if not isinstance(result, dict):
                return False
            return (
                str(result.get("status") or "ok").strip().lower()
                not in failure_statuses
            )
        return True

    @staticmethod
    def _operation_result_report(
        *,
        operation_type: str,
        task_id: str,
        result: Any,
        error: Optional[BaseException],
        succeeded: bool,
        elapsed_ms: float,
    ) -> Dict[str, Any]:
        if isinstance(result, dict):
            report = dict(result)
        else:
            report = {"result": result}
        result_status = str(report.get("status") or "").strip().lower()
        report.update({
            "accepted": True,
            "task_id": task_id,
            "operation_type": operation_type,
            "status": (
                "failed"
                if error is not None or not succeeded
                else (result_status or "ok")
            ),
            "total_elapsed_ms": elapsed_ms,
        })
        if operation_type in {"memory_fact_extraction", "memory_user_activity_extraction"}:
            report["stored"] = bool(succeeded and error is None)
        if error is not None:
            report["error_type"] = type(error).__name__
            report["error"] = str(error)
        return report


class TaskScheduler:
    """Own queued tasks on one worker, with FIFO within each priority."""

    _SIGNAL_UPDATE_TASK_KINDS = (
        "memory_user_claim_update",
        "memory_user_observed_pattern_update",
        "memory_user_intent_update",
    )
    _SIGNAL_TASK_TARGETS = {
        "memory_user_claim_update": "user_claim_signal",
        "memory_user_observed_pattern_update": "user_action_signal",
        "memory_user_intent_update": "user_intent_signal",
    }

    def __init__(
        self,
        *,
        operation_reporter: TaskOperationReporter,
        maxsize: int = 100,
        logger: Optional[logging.Logger] = None,
        worker_name: str = "memory-task-worker",
        task_scheduling_config: Optional[Dict[str, Any]] = None,
        user_memory_db: Optional[UserMemoryDB] = None,
        memory_enabled: Any = True,
    ) -> None:
        self._operation_reporter = operation_reporter
        self._handlers: Dict[str, TaskHandler] = {}
        self._queue: queue.PriorityQueue = queue.PriorityQueue(
            maxsize=max(1, int(maxsize)),
        )
        self._submission_sequence = 0
        self._logger = logger or logging.getLogger(__name__)
        self._worker_name = worker_name
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_lock = threading.Lock()
        self._execution_lock = threading.RLock()
        self._shutdown_event = threading.Event()
        settings = dict(
            task_scheduling_config
            if isinstance(task_scheduling_config, dict)
            else {}
        )
        self._memory_enabled = self._config_bool(memory_enabled, True)
        self._signal_lookback_hours = {
            "user_claim_signal": float(settings.get("user_claim_signal_lookback_hours", 6)),
            "user_intent_signal": float(settings.get("user_intent_signal_lookback_hours", 6)),
        }
        for target, hours in self._signal_lookback_hours.items():
            if not 0 < hours < float("inf"):
                raise ValueError(f"Signal lookback hours must be finite and positive: {target}")
        user_state_updates_enabled = self._config_bool(
            settings.get("enable_memory_user_state_update"), True,
        )
        self._task_enabled = {
            "memory_user_thread_update": self._config_bool(
                settings.get("enable_memory_user_thread_update"), True,
            ),
            "memory_user_claim_update": self._config_bool(
                settings.get("enable_memory_user_claim_update"), True,
            ),
            "memory_user_observed_pattern_update": self._config_bool(
                settings.get("enable_memory_user_observed_pattern_update"), True,
            ),
            "memory_user_intent_update": self._config_bool(
                settings.get("enable_memory_user_intent_update"), True,
            ),
            "memory_user_intent_update_from_activities": self._config_bool(
                settings.get("enable_memory_user_intent_update_from_activities"),
                self._config_bool(settings.get("enable_memory_user_intent_update"), True),
            ),
            "user_state_update_from_user_claims": user_state_updates_enabled,
            "user_state_update_from_user_threads": user_state_updates_enabled,
            "user_state_update_from_user_observed_patterns": user_state_updates_enabled,
            "user_prediction_update_from_intent_events": self._config_bool(
                settings.get("enable_user_prediction"), True,
            ),
            "user_prediction_reassessment": self._config_bool(
                settings.get("enable_user_prediction"), True,
            ),
            "user_prediction_window_expiration": self._config_bool(
                settings.get("enable_user_prediction"), True,
            ),
            "user_proactive_policy_evaluation": self._config_bool(
                settings.get("enable_user_proactive_policy_evaluation"), True,
            ),
            "proactive_service_feedback": self._config_bool(
                settings.get("enable_proactive_service_feedback"), True,
            ),
        }
        self._user_prediction_horizon_hours = max(
            0.1, float(settings.get("user_prediction_horizon_hours", 6) or 6),
        )
        self._prediction_lock = threading.RLock()
        self._prediction_events_inflight: set[int] = set()
        self._prediction_reassessment_pending = False
        self._prediction_window_expiration_pending = False
        self._prediction_reassessment_retry_at = 0.0
        self._prediction_reassessment_retry_seconds = max(
            1.0, float(settings.get("user_prediction_reassessment_retry_seconds", 300)),
        )
        self._user_memory_db = user_memory_db
        self._task_execution_db: Optional[TaskExecutionDB] = None
        if user_memory_db is not None and hasattr(user_memory_db, "db_path"):
            task_db_path = user_memory_db.db_path.with_name("task_scheduler.db")
            self._task_execution_db = TaskExecutionDB(task_db_path)
            try:
                signal_references = user_memory_db.get_user_signal_batches()
                recovered = self._task_execution_db.register_signals(**signal_references)
                if recovered:
                    self._logger.warning(
                        "Recovered unregistered signals without producer input time count=%s; "
                        "undated claim/intent batches are excluded from time-window scheduling",
                        recovered,
                    )
            except Exception:
                self._task_execution_db.close()
                raise
        self._claim_state_batch_limit = max(
            1, int(settings.get("user_state_claim_batch_limit", 80) or 80),
        )
        self._claim_state_lock = threading.Lock()
        self._claim_state_task_pending = False
        self._claim_state_force_requested = False
        self._latest_input_time_end = ""
        self.register_handler(
            "memory_flush_signal_updates",
            self._process_flush_signal_update_tasks,
        )

    @property
    def execution_lock(self) -> threading.RLock:
        """Shared lock for queued writes and synchronized domain reads."""
        return self._execution_lock

    @property
    def operation_reporter(self) -> TaskOperationReporter:
        """Reporter shared with synchronous operations such as recall."""
        return self._operation_reporter

    def _select_pending_signal_ids(
        self, target: str, *, reference_timestamp: Any = None,
    ) -> List[int]:
        if self._task_execution_db is None or self._user_memory_db is None:
            return []
        signal_type = {
            "user_intent_signal": "intent",
            "user_action_signal": "action",
        }.get(target, "claim")
        all_pending_ids = self._task_execution_db.pending_signal_ids(target)
        pending_ids = all_pending_ids
        if target in self._signal_lookback_hours:
            reference = (reference_timestamp or self._latest_input_time_end
                         or datetime.now().astimezone().isoformat())
            pending_ids = self._task_execution_db.pending_signal_ids(
                target, reference_timestamp=reference,
                lookback_hours=self._signal_lookback_hours[target],
            )
            self._logger.info(
                "signal selection target=%s reference_timestamp=%s lookback_hours=%s "
                "pending_count=%s within_window_count=%s excluded_by_time_count=%s",
                target, reference, self._signal_lookback_hours[target],
                len(all_pending_ids), len(pending_ids), len(all_pending_ids) - len(pending_ids),
            )
        if not pending_ids:
            return []
        with self._user_memory_db.reader_transaction() as reader_db:
            pending_signals = reader_db.get_user_signals_by_ids(
                signal_type, pending_ids,
            )
        valid_ids = {int(row["signal_id"]) for row in pending_signals}
        missing = set(pending_ids) - valid_ids
        if missing:
            self._task_execution_db.mark_signal_ids_completed(target, sorted(missing))
        return [signal_id for signal_id in pending_ids if signal_id in valid_ids]

    def _prepare_signal_task_payload(
        self, task_kind: str, payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Freeze pending signal IDs; task handlers enforce their own limits."""
        prepared = dict(payload)
        if task_kind in {"memory_user_claim_update", "memory_user_intent_update"}:
            prepared["reference_timestamp"] = (
                prepared.get("reference_timestamp") or self._latest_input_time_end
                or datetime.now().astimezone().isoformat()
            )
        if task_kind == "memory_user_claim_update":
            if "signal_ids" not in prepared:
                prepared["signal_ids"] = self._select_pending_signal_ids(
                    "user_claim_signal",
                    reference_timestamp=prepared["reference_timestamp"],
                )
        elif task_kind == "memory_user_intent_update":
            if "signal_ids" not in prepared:
                prepared["signal_ids"] = self._select_pending_signal_ids(
                    "user_intent_signal",
                    reference_timestamp=prepared["reference_timestamp"],
                )
        elif task_kind == "memory_user_observed_pattern_update":
            if "signal_ids" not in prepared:
                prepared["signal_ids"] = self._select_pending_signal_ids(
                    "user_action_signal",
                )
        return prepared

    def _complete_signal_task_result(
        self, task_kind: str, payload: Dict[str, Any], result: Any,
    ) -> None:
        """Complete the submitted snapshot and release unfinished IDs."""
        target = self._SIGNAL_TASK_TARGETS.get(task_kind)
        if target is None or self._task_execution_db is None:
            return
        completed_ids = result.pop("_completed_signal_ids", []) if isinstance(result, dict) else []
        if (task_kind == "memory_user_observed_pattern_update"
                and isinstance(result, dict)
                and str(result.get("status") or "").lower() not in {"ok", "empty"}):
            completed_ids = []
        completed = self._task_execution_db.finish_signal_task(
            target, payload.get("signal_ids") or [], completed_ids,
        )
        if isinstance(result, dict):
            result["signals_marked_processed"] = len(completed)

    def record_pending_user_state_claims(
        self, claim_ids: List[int], *,
        claim_relations: Sequence[Dict[str, Any]] = (),
        reference_timestamp: Any = None,
        source_task_id: str = "",
    ) -> int:
        if self._task_execution_db is None:
            return 0
        return self._task_execution_db.record_pending_user_state_claims(
            claim_ids,
            claim_relations=claim_relations,
            reference_timestamp=reference_timestamp,
            source_task_id=source_task_id,
        )

    def register_handler(self, task_kind: str, handler: TaskHandler) -> None:
        """Register a domain handler before its first task is submitted."""
        with self._worker_lock:
            if task_kind in self._handlers:
                raise ValueError(f"Task handler already registered: {task_kind}")
            self._handlers[task_kind] = handler

    @staticmethod
    def _config_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
            return True
        if text in {"0", "false", "no", "n", "off", "disable", "disabled"}:
            return False
        return default

    def start(self) -> None:
        """Start the shared worker after all task handlers are registered."""
        with self._worker_lock:
            if not self._shutdown_event.is_set():
                self._ensure_worker_locked()

    def submit(
        self,
        task_kind: str,
        payload: Dict[str, Any],
        *,
        completion_context: Optional[Dict[str, Any]] = None,
        report_payload: Optional[Dict[str, Any]] = None,
        priority: Optional[int] = None,
        result_future: Optional[Future] = None,
    ) -> Dict[str, Any]:
        """Enqueue a non-preemptive task; lower priority numbers run first.

        Proactive service defaults to 0 (highest), other tasks to 10. Tasks
        with equal priority execute in submission order.
        """
        resolved_priority = priority
        if resolved_priority is None:
            resolved_priority = 0 if task_kind == "user_proactive_policy_evaluation" else 10
        if type(resolved_priority) is not int or resolved_priority < 0:
            raise ValueError("priority must be a nonnegative integer")
        # Enqueueing must not wait for the worker's long-running LLM handler.
        # The worker still holds _execution_lock for its shared writer connection;
        # signal validation above uses its own read-only connection.
        with self._worker_lock:
            task_id = self._operation_reporter.next_task_id(task_kind)
            if self._shutdown_event.is_set():
                reason = "worker_shutdown"
            elif not self._memory_enabled:
                reason = "memory_disabled"
            elif task_kind in self._task_enabled and not self._task_enabled[task_kind]:
                reason = "task_disabled"
            elif task_kind not in self._handlers:
                reason = "task_handler_not_registered"
            else:
                reason = None
            if reason is not None:
                return self._reject_locked(task_kind, task_id, reason)
            task_payload = self._prepare_signal_task_payload(task_kind, payload)
            task_completion_context = dict(completion_context or {})
            task = {
                "kind": task_kind,
                "payload": task_payload,
                "task_id": task_id,
                "started_at": time.monotonic(),
                "completion_context": task_completion_context,
                "priority": resolved_priority,
                "result_future": result_future,
            }
            if self._queue.full():
                return self._reject_locked(task_kind, task_id, "worker_queue_full")
            if task_kind == "user_prediction_reassessment" and self._prediction_reassessment_pending:
                return self._reject_locked(task_kind, task_id, "reassessment_already_pending")
            if task_kind == "user_prediction_window_expiration" and self._prediction_window_expiration_pending:
                return self._reject_locked(task_kind, task_id, "window_expiration_already_pending")
            if task_kind == "memory_user_thread_update":
                if self._task_execution_db is None:
                    return self._reject_locked(task_kind, task_id, "task_execution_database_unavailable")
                activity_ids = task_payload.get("activity_ids")
                if activity_ids is None:
                    activity_ids = self._task_execution_db.pending_user_thread_activity_ids()
                task_payload["activity_ids"] = self._task_execution_db.reserve_user_thread_activities(
                    activity_ids, task_id=task_id,
                )
                if not task_payload["activity_ids"]:
                    return self._reject_locked(task_kind, task_id, "no_pending_activities")
            signal_target = self._SIGNAL_TASK_TARGETS.get(task_kind)
            if signal_target is not None:
                signal_ids = task_payload.get("signal_ids") or []
                if not signal_ids:
                    return self._reject_locked(task_kind, task_id, "no_pending_signals")
                if self._task_execution_db is None:
                    return self._reject_locked(task_kind, task_id, "task_execution_database_unavailable")
                try:
                    self._task_execution_db.reserve_signal_ids(signal_target, signal_ids)
                except ValueError:
                    return self._reject_locked(task_kind, task_id, "signals_already_submitted")
                except sqlite3.Error:
                    self._logger.exception(
                        "Failed to reserve signal IDs kind=%s id=%s", task_kind, task_id,
                    )
                    return self._reject_locked(task_kind, task_id, "signal_reservation_failed")
            try:
                self._operation_reporter.on_task_submitted(
                    operation_type=task_kind,
                    task_id=task_id,
                    payload=report_payload if report_payload is not None else task_payload,
                )
            except Exception:
                self._logger.exception(
                    "Failed to report task submission kind=%s id=%s",
                    task_kind,
                    task_id,
                )
            if task_kind == "memory_fact_extraction":
                self._record_input_time_end(
                    task["completion_context"].get("input_time_end")
                )
            if self._task_execution_db is not None:
                try:
                    self._task_execution_db.record_task_submission_info(
                        task_id, task_kind,
                    )
                except Exception:
                    self._logger.exception(
                        "Failed to persist task submission kind=%s id=%s",
                        task_kind, task_id,
                    )
            # Publish only after the reporter and task DB know the task. The
            # worker may take it immediately after put_nowait returns.
            if task_kind == "user_prediction_reassessment":
                self._prediction_reassessment_pending = True
            if task_kind == "user_prediction_window_expiration":
                self._prediction_window_expiration_pending = True
            self._submission_sequence += 1
            self._queue.put_nowait((resolved_priority, self._submission_sequence, task))
            self._logger.info(
                "scheduled task queued task_kind=%s task_id=%s priority=%s queue_depth=%s",
                task_kind, task_id, resolved_priority, self._queue.qsize(),
            )
            self._ensure_worker_locked()
        return {
            "queued": True,
            "status": "queued",
            "task_id": task_id,
            "operation_type": task_kind,
        }

    def submit_and_wait(
        self, task_kind: str, payload: Dict[str, Any], *, timeout: Optional[float] = None,
    ) -> Any:
        """Wait for this task's result, without holding the worker execution lock.

        A timeout stops waiting, not execution. A handler must not wait on its
        own worker; background tasks should use submit instead.
        """
        if threading.current_thread() is self._worker_thread:
            raise RuntimeError("Cannot wait for a queued task from its worker")
        future = Future()
        report = self.submit(task_kind, payload, result_future=future)
        if not report.get("queued"):
            raise RuntimeError(f"Task not queued: {report.get('reason')}")
        return future.result(timeout=timeout)

    def reject(self, task_kind: str, reason: str) -> Dict[str, Any]:
        """Record a domain-level rejection before a task enters the queue."""
        with self._worker_lock:
            task_id = self._operation_reporter.next_task_id(task_kind)
            return self._reject_locked(task_kind, task_id, reason)

    def _reject_locked(
        self, task_kind: str, task_id: str, reason: str,
    ) -> Dict[str, Any]:
        self._operation_reporter.on_task_rejected(
            operation_type=task_kind,
            task_id=task_id,
            reason=reason,
        )
        return {
            "queued": False,
            "status": "rejected",
            "reason": reason,
            "task_id": task_id,
            "operation_type": task_kind,
        }

    def set_logger(self, logger: Optional[logging.Logger]) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def flush(self, timeout: Optional[float] = None) -> bool:
        """Wait until every accepted task has completed."""
        started_at = time.monotonic()
        deadline = None if timeout is None else started_at + max(0.0, timeout)
        next_progress_log_at = started_at + 30.0
        while self._queue.unfinished_tasks:
            worker = self._worker_thread
            if worker is not None and not worker.is_alive():
                return False
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                return False
            if now >= next_progress_log_at:
                self._logger.warning(
                    "scheduled task drain waiting elapsed_seconds=%.1f "
                    "unfinished_tasks=%s queue_depth=%s worker_alive=%s",
                    now - started_at, self._queue.unfinished_tasks,
                    self._queue.qsize(), bool(worker and worker.is_alive()),
                )
                next_progress_log_at = now + 30.0
            time.sleep(0.01)
        return True

    def shutdown(
        self,
        *,
        wait: bool = True,
        timeout: Optional[float] = None,
    ) -> bool:
        """Stop accepting tasks and optionally drain and join the worker."""
        with self._worker_lock:
            self._shutdown_event.set()
            worker = self._worker_thread
        if worker is None:
            if self._task_execution_db is not None:
                self._task_execution_db.close()
                self._task_execution_db = None
            return not self._queue.unfinished_tasks
        if not wait:
            return not self._queue.unfinished_tasks

        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while self._queue.unfinished_tasks:
            if not worker.is_alive():
                return False
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        worker.join(timeout=remaining)
        stopped = not worker.is_alive() and not self._queue.unfinished_tasks
        if stopped and self._task_execution_db is not None:
            self._task_execution_db.close()
            self._task_execution_db = None
        return stopped

    def _ensure_worker_locked(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=self._worker_name,
        )
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        while not self._shutdown_event.is_set() or not self._queue.empty():
            try:
                _, _, task = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._execute_scheduled_task(task)
            except Exception as exc:
                self._logger.exception(
                    "Scheduled task handler failed task_kind=%s task_id=%s",
                    task.get("kind"),
                    task.get("task_id"),
                )
                future = task.get("result_future")
                if future is not None and not future.done():
                    future.set_exception(exc)
            finally:
                self._queue.task_done()

    def _execute_scheduled_task(self, task: Dict[str, Any]) -> None:
        """Execute, apply result bookkeeping, report, then schedule follow-ups."""
        task_kind = str(task["kind"])
        task_id = str(task["task_id"])
        result: Any = None
        error: Optional[BaseException] = None
        handler_started_at = time.monotonic()
        self._logger.info(
            "scheduled task started task_kind=%s task_id=%s priority=%s queue_wait_ms=%.1f",
            task_kind, task_id, task.get("priority"),
            (handler_started_at - float(task["started_at"])) * 1000,
        )
        try:
            with self._execution_lock:
                result = self._handlers[task_kind](**task["payload"])
        except Exception as exc:
            error = exc
            self._logger.exception(
                "Scheduled task failed task_kind=%s task_id=%s", task_kind, task_id,
            )
        handler_elapsed_ms = (time.monotonic() - handler_started_at) * 1000
        event = {
            "task_kind": task_kind,
            "task_id": task_id,
            "payload": task["payload"],
            "result": result,
            "error": error,
            "succeeded": error is None and self._operation_reporter._operation_succeeded(
                task_kind, result,
            ),
            "completion_context": dict(task.get("completion_context") or {}),
        }
        completion: Dict[str, Any] = {"postprocessed": False}
        try:
            completion = self._postprocess_completed_task(event)
        except Exception as exc:
            error = error or exc
            event["error"] = error
            event["succeeded"] = False
            self._logger.exception(
                "Scheduled task postprocessing failed task_kind=%s task_id=%s",
                task_kind, task_id,
            )
        try:
            succeeded = self._operation_reporter.on_task_finished(
                operation_type=task_kind,
                task_id=task_id,
                started_at=float(task["started_at"]),
                result=result,
                error=error,
            )
        except Exception as exc:
            succeeded = False
            error = error or exc
            self._logger.exception(
                "Failed to report task completion task_kind=%s task_id=%s",
                task_kind, task_id,
            )
        event["error"] = error
        event["succeeded"] = bool(succeeded)
        result_status = str(result.get("status") or "")[:80] if isinstance(result, dict) else ""
        self._logger.info(
            "scheduled task handler finished task_kind=%s task_id=%s succeeded=%s "
            "result_status=%s handler_elapsed_ms=%.1f total_elapsed_ms=%.1f",
            task_kind, task_id, bool(succeeded), result_status,
            handler_elapsed_ms, (time.monotonic() - float(task["started_at"])) * 1000,
        )
        if self._task_execution_db is not None:
            try:
                self._task_execution_db.record_task_completion_info(
                    task_id,
                    status=(result_status or "ok") if succeeded else "failed",
                    result=result if error is None else {
                        "error_type": type(error).__name__,
                    },
                )
            except Exception:
                self._logger.exception(
                    "Failed to persist task completion kind=%s id=%s",
                    task_kind, task_id,
                )
        self._schedule_completed_task(event, completion)
        self._logger.info(
            "scheduled task follow-up finished task_kind=%s task_id=%s queue_depth=%s",
            task_kind, task_id, self._queue.qsize(),
        )
        future = task.get("result_future")
        if future is not None and not future.done():
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(result)

    def trigger_user_prediction_time_tick(
        self, *, reference_timestamp: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Queue window cleanup first, then independently consider due-defer work.

        Each subreport describes one task; queued means at least one was queued.
        Defer retry cooldown must never suppress deterministic expiration.
        """
        reference = datetime.now().astimezone()
        if reference_timestamp is not None:
            reference = (reference_timestamp if isinstance(reference_timestamp, datetime)
                         else datetime.fromisoformat(str(reference_timestamp).replace("Z", "+00:00")))
            if reference.tzinfo is None:
                reference = reference.replace(tzinfo=datetime.now().astimezone().tzinfo)
        with self._prediction_lock:
            timestamp = reference.isoformat() if reference_timestamp is not None else None
            if self._prediction_window_expiration_pending:
                expiration_report = {
                    "queued": False, "status": "skipped", "reason": "window_expiration_already_pending",
                }
            else:
                expiration_report = self.submit(
                    "user_prediction_window_expiration", {"reference_timestamp": timestamp},
                )
            if self._prediction_reassessment_pending:
                reassessment_report = {
                    "queued": False, "status": "skipped", "reason": "reassessment_already_pending",
                }
            elif reference.timestamp() < self._prediction_reassessment_retry_at:
                reassessment_report = {
                    "queued": False, "status": "skipped", "reason": "reassessment_retry_cooldown",
                }
            else:
                # Resolve real time at execution; preserve an explicit replay clock.
                reassessment_report = self.submit("user_prediction_reassessment", {
                    "reference_timestamp": timestamp,
                    "horizon_hours": self._user_prediction_horizon_hours,
                })
        queued = bool(expiration_report.get("queued") or reassessment_report.get("queued"))
        report = {
            "queued": queued, "status": "queued" if queued else "skipped",
            "window_expiration_task": expiration_report, "reassessment_task": reassessment_report,
        }
        self._logger.info("prediction time tick reference_timestamp=%s report=%s", reference.isoformat(), report)
        return report

    def trigger_user_state_update(self, *, drain: bool = False) -> Dict[str, Any]:
        """Inspect persisted claims now and enqueue only the actual state update.

        TaskExecutionDB owns a separate connection and lock. Do not acquire
        _execution_lock here or wait for model calls / future claim results.
        Timing and cadence are exclusively owned by the application host.
        """
        self._logger.info(
            "trigger_user_state_update received drain=%s batch_limit=%s queue_depth=%s",
            drain, self._claim_state_batch_limit, self._queue.qsize(),
        )
        if self._task_execution_db is None:
            self._logger.warning(
                "trigger_user_state_update skipped drain=%s queued=False "
                "reason=task_execution_database_unavailable", drain,
            )
            return {"queued": False, "reason": "task_execution_database_unavailable"}
        with self._claim_state_lock:
            if self._claim_state_task_pending:
                if drain:
                    self._claim_state_force_requested = True
                self._logger.info(
                    "trigger_user_state_update skipped drain=%s queued=False "
                    "reason=claim_state_task_pending drain_requested=%s",
                    drain, self._claim_state_force_requested,
                )
                return {"queued": False, "reason": "claim_state_task_pending"}
            self._claim_state_task_pending = True
        queued = False
        try:
            pending = self._task_execution_db.get_pending_user_state_claims(
                limit=self._claim_state_batch_limit,
            )
            items = pending["items"]
            if not items:
                self._logger.info(
                    "trigger_user_state_update skipped drain=%s pending_claims=0 "
                    "selected_claims=0 queued=False reason=no_changed_claims", drain,
                )
                return {"status": "empty", "queued": False, "reason": "no_changed_claims"}
            versions = [(int(item["claim_id"]), int(item["generation"])) for item in items]
            newest = max(items, key=lambda item: item["last_detected_at"])
            with self._worker_lock:
                latest_input_time_end = self._latest_input_time_end
            # This is evidence time, not the time the host sent its signal.
            reference_timestamp = newest["reference_timestamp"] or latest_input_time_end or None
            self._logger.info(
                "trigger_user_state_update submitting drain=%s pending_claims=%s "
                "selected_claims=%s relation_events=%s",
                drain, pending["pending_count"], len(versions), len(pending["claim_relations"]),
            )
            report = self.submit(
                "user_state_update_from_user_claims",
                {"new_claim_ids": [claim_id for claim_id, _ in versions],
                 "claim_relations": pending["claim_relations"],
                 "reference_timestamp": reference_timestamp},
                completion_context={"claim_state_window_versions": versions,
                                    "claim_state_final_flush": drain},
                report_payload={
                    "new_claim_ids": [claim_id for claim_id, _ in versions],
                    "relation_event_count": len(pending["claim_relations"]),
                    "reference_timestamp": reference_timestamp,
                },
            )
            queued = bool(report.get("queued"))
            self._logger.info(
                "trigger_user_state_update evaluated drain=%s pending_claims=%s "
                "selected_claims=%s queued=%s task_id=%s reason=%s",
                drain, pending["pending_count"], len(versions), queued,
                report.get("task_id") or "", report.get("reason") or "",
            )
            return {**report, "pending_claim_count": pending["pending_count"],
                    "selected_claim_count": len(versions)}
        except Exception:
            self._logger.exception(
                "trigger_user_state_update failed drain=%s queued=%s", drain, queued,
            )
            raise
        finally:
            if not queued:
                with self._claim_state_lock:
                    self._claim_state_task_pending = False
                    self._claim_state_force_requested = False

    def drain_pending_user_state_claims(self) -> Dict[str, Any]:
        """Explicitly process successive batches; never start a periodic clock."""
        return self.trigger_user_state_update(drain=True)

    def submit_flush_signal_update_tasks(self) -> Dict[str, Any]:
        """Queue a normal-priority boundary; resolve signals when it executes."""
        return self.submit("memory_flush_signal_updates", {})

    def _process_flush_signal_update_tasks(self) -> Dict[str, Any]:
        """Drain unreserved signals after preceding fact and activity tasks."""
        signal_reports = self._submit_final_signal_update_tasks()
        failed_task_kinds = [
            task_kind
            for task_kind, report in signal_reports.items()
            if not report.get("queued")
            and report.get("reason") not in {"no_pending_signals", "task_disabled"}
        ]
        self._logger.info(
            "final signal updates submitted claim_queued=%s "
            "observed_pattern_queued=%s intent_queued=%s",
            bool(signal_reports["memory_user_claim_update"].get("queued")),
            bool(signal_reports["memory_user_observed_pattern_update"].get("queued")),
            bool(signal_reports["memory_user_intent_update"].get("queued")),
        )
        return {
            "status": "failed" if failed_task_kinds else "ok",
            "signal_update_tasks": signal_reports,
            "failed_task_kinds": failed_task_kinds,
        }

    def _submit_final_signal_update_tasks(self) -> Dict[str, Dict[str, Any]]:
        with self._worker_lock:
            resolved_timestamp = self._latest_input_time_end
        reports: Dict[str, Dict[str, Any]] = {}
        for task_kind in self._SIGNAL_UPDATE_TASK_KINDS:
            payload: Dict[str, Any] = {
                "reference_timestamp": resolved_timestamp or None,
            }
            reports[task_kind] = self.submit(
                task_kind,
                payload,
                report_payload=payload,
            )
        return reports

    def _postprocess_completed_task(self, event: Dict[str, Any]) -> Dict[str, Any]:
        task_kind = str(event.get("task_kind") or "")
        postprocessors = {
            "memory_fact_extraction": self._postprocess_fact_extraction_task,
            "proactive_service_feedback": self._postprocess_fact_extraction_task,
            "memory_user_activity_extraction": self._postprocess_user_activity_task,
            "memory_user_thread_update": self._postprocess_user_thread_update_task,
            "memory_user_claim_update": self._postprocess_user_claim_update_task,
            "memory_user_observed_pattern_update": self._postprocess_signal_update_task,
            "memory_user_intent_update": self._postprocess_signal_update_task,
            "user_state_update_from_user_claims": (
                self._postprocess_user_state_update_from_user_claims_task
            ),
            "user_prediction_update_from_intent_events": self._postprocess_user_prediction_task,
            "user_prediction_reassessment": self._postprocess_prediction_reassessment_task,
            "user_prediction_window_expiration": self._postprocess_prediction_window_expiration_task,
        }
        completion: Dict[str, Any] = {"postprocessed": False}
        postprocess = postprocessors.get(task_kind)
        if postprocess is not None:
            completion.update(postprocess(event) or {})
        completion["postprocessed"] = True
        return completion

    def _schedule_completed_task(
        self, event: Dict[str, Any], completion: Dict[str, Any],
    ) -> None:
        task_kind = str(event.get("task_kind") or "")
        schedulers = {
            "memory_user_activity_extraction": self._schedule_after_user_activity_task,
            "memory_user_thread_update": self._schedule_after_user_thread_update_task,
            "memory_user_observed_pattern_update": (
                self._schedule_after_user_observed_pattern_update_task
            ),
            "memory_user_intent_update": self._schedule_after_user_intent_update_task,
            "memory_user_intent_update_from_activities": self._schedule_after_user_intent_update_task,
            "user_state_update_from_user_claims": (
                self._schedule_after_user_state_update_from_user_claims_task
            ),
        }
        schedule = schedulers.get(task_kind)
        if schedule is None:
            return
        if completion.get("postprocessed"):
            schedule(event, completion)

    def _postprocess_fact_extraction_task(
        self, event: Dict[str, Any],
    ) -> None:
        result = event.get("result")
        context = dict(event.get("completion_context") or {})
        if event.get("task_kind") == "proactive_service_feedback" and isinstance(result, dict):
            context.update(
                input_time_end=result.get("reference_timestamp"),
                contains_interaction_turn=result.get("contains_interaction_turn", False),
            )
        if (isinstance(result, dict) and self._task_execution_db is not None
                and self._user_memory_db is not None):
            signal_references = self._user_memory_db.get_user_signal_batches(
                source_type="fact", source_ids=result.get("fact_ids") or [],
            )
            self._task_execution_db.register_signals(
                **signal_references,
                source_task_id=str(event.get("task_id") or ""),
                source_time_end=context.get("input_time_end") or "",
            )
        if event.get("succeeded") and isinstance(result, dict):
            if event.get("task_kind") == "proactive_service_feedback":
                self._record_input_time_end(result.get("reference_timestamp"))
            self._record_fact_extraction_task_result(
                result, completion_context=context,
            )

    def _postprocess_user_activity_task(self, event: Dict[str, Any]) -> None:
        result = event.get("result")
        if event.get("succeeded") and isinstance(result, dict) and self._task_execution_db:
            self._task_execution_db.register_user_thread_activities(
                result.get("user_activity_item_ids") or [],
            )
            signal_ids = result.get("user_action_signal_ids") or []
            if signal_ids:
                added = self._task_execution_db.register_signals(
                    claim_signal_ids=[], intent_signal_ids=[], action_signal_ids=signal_ids,
                    source_task_id=str(event.get("task_id") or ""),
                    source_time_end=result.get("reference_timestamp") or "",
                )
                self._logger.info(
                    "activity action signals registered task_id=%s signal_ids=%s added=%s",
                    event.get("task_id"), signal_ids, added,
                )

    def _postprocess_user_thread_update_task(self, event: Dict[str, Any]) -> None:
        if self._task_execution_db is None:
            return
        result = event.get("result")
        # Earlier batches may have committed even if a later batch failed.
        completed = result.get("processed_activity_ids", []) if isinstance(result, dict) else []
        self._task_execution_db.finish_user_thread_activities(
            task_id=event["task_id"], activity_ids=event["payload"].get("activity_ids") or [],
            completed_ids=completed,
        )

    def _schedule_after_user_activity_task(
        self, event: Dict[str, Any], completion: Dict[str, Any],
    ) -> None:
        result = event.get("result")
        if not event.get("succeeded") or not isinstance(result, dict):
            return
        # Use the processed source window, even when no activity was extracted.
        # Do not mix in pending signals from unrelated or later windows.
        self._schedule_signal_updates_for_activity_window(result)
        activity_ids = result.get("user_activity_item_ids") or []
        if not activity_ids:
            return
        # New activities are the sole automatic entry. Do not scan historical
        # activities from the intent-update completion path.
        lifecycle_kind = "memory_user_intent_update_from_activities"
        if self._task_enabled[lifecycle_kind] and lifecycle_kind in self._handlers:
            report = self.submit(lifecycle_kind, {
                "activity_ids": list(activity_ids),
                "reference_timestamp": result.get("reference_timestamp") or self._latest_input_time_end or None,
            })
            if not report.get("queued"):
                self._logger.warning("Activity intent lifecycle update not queued: %s", report)
        if self._task_enabled["memory_user_thread_update"]:
            report = self.submit("memory_user_thread_update", {})
            if not report.get("queued"):
                self._logger.warning("User thread update not queued: %s", report)

    def _schedule_after_user_thread_update_task(
        self, event: Dict[str, Any], completion: Dict[str, Any],
    ) -> None:
        result = event.get("result")
        if not self._task_enabled["user_state_update_from_user_threads"]:
            return
        if event.get("error") is not None or not isinstance(result, dict):
            return
        changes = result.get("thread_changes") or []
        if not changes:
            return
        # A failed later batch may still have committed earlier changes. Those
        # completed batches also need their state projection.
        report = self.submit(
            "user_state_update_from_user_threads",
            {"thread_changes": changes, "runtime_context": {}},
            report_payload={
                "thread_ids": [change["thread"]["id"] for change in changes],
                "thread_event_ids": result.get("thread_event_ids") or [],
            },
        )
        if not report.get("queued"):
            self._logger.warning("Thread state update not queued: %s", report)

    def _postprocess_user_prediction_task(
        self, event: Dict[str, Any],
    ) -> None:
        context = event.get("completion_context") or {}
        event_ids = context.get("prediction_intent_event_ids") or []
        result = event.get("result") or {}
        with self._prediction_lock:
            self._prediction_events_inflight.difference_update(event_ids)
            if (event.get("succeeded") and isinstance(result, dict)
                    and result.get("status") in {"ok", "empty", "partial"}
                    and self._task_execution_db is not None):
                self._task_execution_db.complete_prediction_intent_events(
                    sorted(set(result.get("processed_intent_event_ids") or []) & set(event_ids)),
                )

    def _postprocess_prediction_window_expiration_task(self, event: Dict[str, Any]) -> None:
        with self._prediction_lock:
            self._prediction_window_expiration_pending = False

    def _postprocess_prediction_reassessment_task(self, event: Dict[str, Any]) -> None:
        with self._prediction_lock:
            try:
                if event.get("succeeded"):
                    self._prediction_reassessment_retry_at = 0.0
                else:
                    # Replays use their logical clock; real-time retries wait
                    # from completion, not from the start of a slow LLM call.
                    reference = (event.get("payload") or {}).get("reference_timestamp")
                    now = datetime.now().astimezone()
                    if reference is not None:
                        try:
                            now = datetime.fromisoformat(str(reference).replace("Z", "+00:00"))
                        except ValueError:
                            self._logger.warning("Invalid reassessment retry timestamp: %r", reference)
                    self._prediction_reassessment_retry_at = (
                        now.timestamp() + self._prediction_reassessment_retry_seconds
                    )
            finally:
                self._prediction_reassessment_pending = False

    def _postprocess_user_state_update_from_user_claims_task(
        self, event: Dict[str, Any],
    ) -> Dict[str, Any]:
        context = event.get("completion_context") or {}
        versions = context.get("claim_state_window_versions")
        if not isinstance(versions, list):
            return {"completed_successfully": False, "force_requested": False}
        completed_successfully = False
        try:
            result = event.get("result")
            result_status = str(
                (result or {}).get("status")
                if isinstance(result, dict) else ""
            ).strip().lower()
            if event.get("succeeded") and result_status in {"ok", "empty"}:
                with self._execution_lock:
                    acknowledged = self._task_execution_db.acknowledge_user_state_claims(
                        versions,
                    )
                completed_successfully = True
                self._logger.info(
                    "claim state window completed task_id=%s acknowledged_claims=%s",
                    event.get("task_id"), acknowledged,
                )
            else:
                self._logger.warning(
                    "claim state window retained pending claims task_id=%s status=%s",
                    event.get("task_id"), result_status or "error",
                )
        finally:
            with self._claim_state_lock:
                self._claim_state_task_pending = False
                force_requested = self._claim_state_force_requested
                self._claim_state_force_requested = False
        return {
            "completed_successfully": completed_successfully,
            "force_requested": force_requested,
        }

    def _schedule_after_user_state_update_from_user_claims_task(
        self, event: Dict[str, Any], completion: Dict[str, Any],
    ) -> None:
        context = event.get("completion_context") or {}
        if (completion.get("completed_successfully") and (
                context.get("claim_state_final_flush")
                or completion.get("force_requested"))):
            self.drain_pending_user_state_claims()

    def _postprocess_signal_update_task(
        self, event: Dict[str, Any],
    ) -> None:
        """Acknowledge the signal IDs completed by this task's handler."""
        task_kind = str(event.get("task_kind") or "")
        result = event.get("result")
        self._complete_signal_task_result(
            task_kind, event.get("payload") or {}, result,
        )

    def _postprocess_user_claim_update_task(
        self, event: Dict[str, Any],
    ) -> None:
        result = event.get("result")
        try:
            if event.get("succeeded") and isinstance(result, dict):
                self.record_pending_user_state_claims(
                    list(result.get("new_claim_ids") or []),
                    claim_relations=result.get("claim_relations") or [],
                    reference_timestamp=result.get("reference_timestamp"),
                    source_task_id=str(event.get("task_id") or ""),
                )
        except Exception:
            if isinstance(result, dict):
                result["_completed_signal_ids"] = []
            raise
        finally:
            self._postprocess_signal_update_task(event)

    def _schedule_after_user_observed_pattern_update_task(
        self, event: Dict[str, Any], completion: Dict[str, Any],
    ) -> None:
        result = event.get("result")
        if (event.get("succeeded") and isinstance(result, dict)):
            new_pattern_ids = list(result.get("new_pattern_ids") or [])
            pattern_relation_event_ids = list(
                result.get("pattern_relation_event_ids") or []
            )
            if new_pattern_ids or pattern_relation_event_ids:
                payload = {
                    "new_pattern_ids": new_pattern_ids,
                    "pattern_relation_event_ids": pattern_relation_event_ids,
                    "reference_timestamp": result.get("reference_timestamp"),
                }
                self.submit(
                    "user_state_update_from_user_observed_patterns", payload,
                    report_payload={
                        "new_pattern_ids": new_pattern_ids,
                        "pattern_relation_event_ids": pattern_relation_event_ids,
                    },
                )

    def _schedule_after_user_intent_update_task(
        self, event: Dict[str, Any], completion: Dict[str, Any],
    ) -> None:
        result = event.get("result")
        if (not event.get("succeeded") or not isinstance(result, dict)
                or not result.get("intent_event_ids")):
            return
        event_ids = sorted({int(value) for value in result["intent_event_ids"] if int(value) > 0})
        with self._prediction_lock:
            completed = (
                self._task_execution_db.completed_prediction_intent_events(event_ids)
                if self._task_execution_db is not None else set()
            )
            selected = sorted(set(event_ids) - completed - self._prediction_events_inflight)
            if not selected:
                return
            payload = {
                "intent_event_ids": selected,
                "reference_timestamp": result.get("reference_timestamp"),
                "horizon_hours": self._user_prediction_horizon_hours,
            }
            report = self.submit(
                "user_prediction_update_from_intent_events", payload,
                completion_context={"prediction_intent_event_ids": selected},
                report_payload={"trigger_source": "intent_update", **payload},
            )
            if report.get("queued"):
                self._prediction_events_inflight.update(selected)

    def _record_fact_extraction_task_result(
        self, result: Dict[str, Any], *, completion_context: Any,
    ) -> None:
        context = dict(completion_context) if isinstance(completion_context, dict) else {}
        contains_interaction = bool(context.get("contains_interaction_turn"))
        signal_stats = {
            "memory_user_claim_update": {
                "signal_count": int(result.get("user_claim_signal_count") or 0),
                "signal_fact_count": int(result.get("user_claim_signal_fact_count") or 0),
            },
            "memory_user_intent_update": {
                "signal_count": int(result.get("user_intent_signal_count") or 0),
                "signal_fact_count": int(result.get("user_intent_signal_fact_count") or 0),
            },
            "memory_user_observed_pattern_update": {
                "signal_count": int(result.get("user_action_signal_count") or 0),
                "signal_fact_count": int(result.get("user_action_signal_fact_count") or 0),
            },
        }
        self._logger.info(
            "memory store completion observed facts=%s user_claim_signals=%s "
            "user_claim_signal_facts=%s user_intent_signals=%s "
            "user_intent_signal_facts=%s user_action_signals=%s "
            "user_action_signal_facts=%s contains_interaction_turn=%s input_time_end=%s",
            int(result.get("new_fact_count") or 0),
            signal_stats["memory_user_claim_update"]["signal_count"],
            signal_stats["memory_user_claim_update"]["signal_fact_count"],
            signal_stats["memory_user_intent_update"]["signal_count"],
            signal_stats["memory_user_intent_update"]["signal_fact_count"],
            signal_stats["memory_user_observed_pattern_update"]["signal_count"],
            signal_stats["memory_user_observed_pattern_update"]["signal_fact_count"],
            contains_interaction,
            context.get("input_time_end") or "",
        )

    def _schedule_signal_updates_for_activity_window(self, result: Dict[str, Any]) -> None:
        """Freeze unreserved signal IDs belonging to this completed window."""
        if self._user_memory_db is None or self._task_execution_db is None:
            return
        source_ids = result.get("source_segment_ids") or []
        activity_ids = result.get("user_activity_item_ids") or []
        signals = self._user_memory_db.get_user_signal_batches_for_activity_window(
            source_segment_ids=source_ids, activity_ids=activity_ids,
        )
        reference_timestamp = result.get("reference_timestamp") or self._latest_input_time_end or None
        signal_keys = {
            "memory_user_claim_update": "claim_signal_ids",
            "memory_user_observed_pattern_update": "action_signal_ids",
            "memory_user_intent_update": "intent_signal_ids",
        }
        for kind in self._SIGNAL_UPDATE_TASK_KINDS:
            if not self._task_enabled[kind]:
                continue
            pending_ids = set(self._task_execution_db.pending_signal_ids(self._SIGNAL_TASK_TARGETS[kind]))
            selected_ids = sorted(set(signals[signal_keys[kind]]) & pending_ids)
            if not selected_ids:
                continue
            payload = {"signal_ids": selected_ids, "reference_timestamp": reference_timestamp}
            report = self.submit(kind, payload)
            self._logger.info(
                "activity-window signal update kind=%s signal_ids=%s "
                "source_segment_ids=%s activity_ids=%s reference_timestamp=%s queued=%s reason=%s",
                kind, selected_ids, source_ids, activity_ids, reference_timestamp,
                bool(report.get("queued")), report.get("reason") or "",
            )
            if not report.get("queued"):
                self._logger.warning("Activity-window signal update not queued: %s", report)

    def _record_input_time_end(self, value: Any) -> None:
        """Advance the input watermark while submit() holds _worker_lock."""
        candidate = str(value or "").strip()
        if not candidate:
            return
        current = self._latest_input_time_end
        if not current or self._is_later_timestamp(candidate, current):
            self._latest_input_time_end = candidate

    @staticmethod
    def _is_later_timestamp(candidate: str, current: str) -> bool:
        def epoch(value: str) -> Optional[float]:
            try:
                return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).timestamp()
            except (TypeError, ValueError, OSError):
                return None
        candidate_epoch = epoch(candidate)
        current_epoch = epoch(current)
        if candidate_epoch is not None and current_epoch is not None:
            return candidate_epoch > current_epoch
        return candidate > current
