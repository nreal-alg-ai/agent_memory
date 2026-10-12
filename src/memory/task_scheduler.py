"""FIFO execution and operation reporting for asynchronous memory tasks."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Protocol


TaskHandler = Callable[..., Any]
TaskCompletionListener = Callable[[Dict[str, Any]], None]


class TaskOperationReporter(Protocol):
    def next_task_id(self, operation_type: str) -> str: ...

    def on_task_submitted(
        self, *, operation_type: str, task_id: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None: ...

    def on_task_rejected(
        self, *, operation_type: str, task_id: str, reason: str,
    ) -> None: ...

    def on_task_finished(
        self, *, operation_type: str, task_id: str, started_at: float,
        result: Any = None, error: Optional[BaseException] = None,
    ) -> bool: ...

    def on_recall_finished(self, report: Dict[str, Any]) -> None: ...


class TaskScheduler:
    """Own the complete lifecycle of queued tasks on one FIFO worker."""

    def __init__(
        self,
        *,
        operation_reporter: TaskOperationReporter,
        maxsize: int = 100,
        logger: Optional[logging.Logger] = None,
        worker_name: str = "memory-task-worker",
    ) -> None:
        self._operation_reporter = operation_reporter
        self._handlers: Dict[str, TaskHandler] = {}
        self._queue: queue.Queue[Dict[str, Any]] = queue.Queue(
            maxsize=max(1, int(maxsize)),
        )
        self._logger = logger or logging.getLogger(__name__)
        self._worker_name = worker_name
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_lock = threading.Lock()
        self._execution_lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self._listener_lock = threading.Lock()
        self._completion_listeners: List[TaskCompletionListener] = []

    @property
    def execution_lock(self) -> threading.RLock:
        """Shared lock for queued writes and synchronized domain reads."""
        return self._execution_lock

    @property
    def operation_reporter(self) -> TaskOperationReporter:
        """Reporter shared with synchronous operations such as recall."""
        return self._operation_reporter

    def register_handler(self, task_kind: str, handler: TaskHandler) -> None:
        """Register a domain handler before its first task is submitted."""
        with self._worker_lock:
            if task_kind in self._handlers:
                raise ValueError(f"Task handler already registered: {task_kind}")
            self._handlers[task_kind] = handler

    def submit(
        self,
        task_kind: str,
        payload: Dict[str, Any],
        *,
        completion_context: Optional[Dict[str, Any]] = None,
        report_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Enqueue and report a task, or return a recorded rejection."""
        with self._worker_lock:
            task_id = self._operation_reporter.next_task_id(task_kind)
            if self._shutdown_event.is_set():
                reason = "worker_shutdown"
            elif task_kind not in self._handlers:
                reason = "task_handler_not_registered"
            else:
                reason = None
            if reason is not None:
                return self._reject_locked(task_kind, task_id, reason)
            task = {
                "kind": task_kind,
                "payload": dict(payload),
                "task_id": task_id,
                "started_at": time.monotonic(),
                "completion_context": dict(completion_context or {}),
            }
            try:
                self._queue.put_nowait(task)
            except queue.Full:
                return self._reject_locked(task_kind, task_id, "worker_queue_full")
            try:
                self._operation_reporter.on_task_submitted(
                    operation_type=task_kind,
                    task_id=task_id,
                    payload=report_payload if report_payload is not None else payload,
                )
            except Exception:
                self._logger.exception(
                    "Failed to report task submission kind=%s id=%s",
                    task_kind,
                    task_id,
                )
            self._ensure_worker_locked()
        return {
            "queued": True,
            "status": "queued",
            "task_id": task_id,
            "operation_type": task_kind,
        }

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

    def register_completion_listener(
        self,
        listener: TaskCompletionListener,
    ) -> None:
        with self._listener_lock:
            if listener not in self._completion_listeners:
                self._completion_listeners.append(listener)

    def set_logger(self, logger: Optional[logging.Logger]) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def unregister_completion_listener(
        self,
        listener: TaskCompletionListener,
    ) -> None:
        with self._listener_lock:
            self._completion_listeners = [
                candidate
                for candidate in self._completion_listeners
                if candidate != listener
            ]

    def flush(self, timeout: Optional[float] = None) -> bool:
        """Wait until every accepted task has completed."""
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while self._queue.unfinished_tasks:
            worker = self._worker_thread
            if worker is not None and not worker.is_alive():
                return False
            if deadline is not None and time.monotonic() >= deadline:
                return False
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
        if not wait or worker is None:
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
        return not worker.is_alive() and not self._queue.unfinished_tasks

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
                task = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._execute_scheduled_task(task)
            except Exception:
                self._logger.exception(
                    "Scheduled task handler failed task_kind=%s task_id=%s",
                    task.get("kind"),
                    task.get("task_id"),
                )
            finally:
                self._queue.task_done()

    def _execute_scheduled_task(self, task: Dict[str, Any]) -> None:
        """Execute, report once, then notify completion listeners."""
        task_kind = str(task["kind"])
        task_id = str(task["task_id"])
        result: Any = None
        error: Optional[BaseException] = None
        try:
            with self._execution_lock:
                result = self._handlers[task_kind](**task["payload"])
        except Exception as exc:
            error = exc
            self._logger.exception(
                "Scheduled task failed task_kind=%s task_id=%s", task_kind, task_id,
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
        self._emit_completion({
            "task_kind": task_kind,
            "task_id": task_id,
            "result": result,
            "error": error,
            "succeeded": bool(succeeded),
            "completion_context": dict(task.get("completion_context") or {}),
        })

    def _emit_completion(self, event: Dict[str, Any]) -> None:
        with self._listener_lock:
            listeners = list(self._completion_listeners)
        for listener in listeners:
            try:
                listener(dict(event))
            except Exception:
                self._logger.exception(
                    "Task completion listener failed task_kind=%s task_id=%s",
                    event.get("task_kind"),
                    event.get("task_id"),
                )
