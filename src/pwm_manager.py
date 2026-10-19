"""Composition root for the personal world model."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from llm_client import LLMClient
from memory.memory_database import SessionDB
from memory.memory_manager import MemoryNodeManager
from memory.memory_runtime import MemoryRuntime
from memory.task_scheduler import TaskOperationReporter, TaskScheduler
from user_state import UserStateResolver


class PersonWorldModel:
    """Own the shared memory worker, evidence runtime, and user model."""

    def __init__(
        self,
        *,
        db_path: Path | str,
        memory_runtime_config: Optional[Dict[str, Any]] = None,
        memory_manager_config: Optional[Dict[str, Any]] = None,
        operation_reporter: Optional[TaskOperationReporter] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        manager_config = dict(memory_manager_config or {})
        resolved_db_path = Path(db_path).expanduser().resolve()
        self._operation_reporter = operation_reporter or TaskOperationReporter()
        self._database = SessionDB(resolved_db_path)
        try:
            self._llm_client = LLMClient(
                memory_config=manager_config,
                logger=self._logger.getChild("llm_client"),
            )
            # Both memory and user-state handlers are registered before the
            # shared worker starts processing its FIFO.
            self._task_scheduler = TaskScheduler(
                operation_reporter=self._operation_reporter,
                maxsize=int(manager_config.get("task_queue_maxsize", 100) or 100),
                logger=self._logger.getChild("task_scheduler"),
                claim_state_database=self._database,
                task_scheduling_config=(
                    (memory_runtime_config or {}).get("task_scheduling")
                ),
                memory_enabled=manager_config.get("memory_enabled", True),
            )
            self._memory_runtime = MemoryRuntime(
                db_path=resolved_db_path,
                database=self._database,
                task_scheduler=self._task_scheduler,
                llm_client=self._llm_client,
                memory_runtime_config=memory_runtime_config,
                memory_manager_config=manager_config,
                logger=self._logger.getChild("memory_runtime"),
            )
            self._user_state_resolver = UserStateResolver(
                db=self._database,
                task_scheduler=self._task_scheduler,
                llm_client=self._llm_client,
                memory_config=manager_config,
                world_owner_entity_name=(
                    self._memory_runtime.manager.world_owner_entity_name
                ),
                logger=self._logger.getChild("user_state_resolver"),
            )
            self._task_scheduler.start()
        except Exception:
            if hasattr(self, "_task_scheduler"):
                self._task_scheduler.shutdown(wait=True, timeout=0)
            self._database.close()
            raise
        self._closed = False
        self._memory_inputs_need_flush = False

    @property
    def memory_runtime(self) -> MemoryRuntime:
        return self._memory_runtime

    @property
    def memory_manager(self) -> MemoryNodeManager:
        return self._memory_runtime.manager

    @property
    def user_state_resolver(self) -> UserStateResolver:
        return self._user_state_resolver

    @property
    def task_scheduler(self) -> TaskScheduler:
        return self._task_scheduler

    @property
    def llm_client(self) -> LLMClient:
        return self._llm_client

    @property
    def database(self) -> SessionDB:
        return self._database

    @property
    def operation_reporter(self) -> TaskOperationReporter:
        return self._operation_reporter

    def accept_memory_input(
        self,
        *,
        interaction_turn: Optional[Dict[str, Any]] = None,
        transcript_segments: Optional[Sequence[Dict[str, Any]]] = None,
        tags: Optional[List[str]] = None,
        ambient_recording_enabled: bool = False,
    ) -> Dict[str, Any]:
        report = self._memory_runtime.accept_memory_input(
            interaction_turn=interaction_turn,
            transcript_segments=transcript_segments,
            tags=tags,
            ambient_recording_enabled=ambient_recording_enabled,
        )
        if report.get("reason") not in {
            "expected_one_input_kind", "memory_disabled", "empty_transcript_batch",
        }:
            self._memory_inputs_need_flush = True
        return report

    def update_ambient_asr_watermark(
        self,
        value: Any,
        *,
        evaluate_user_activity: bool = True,
    ) -> Dict[str, Any]:
        """Release buffered inputs once ambient ASR has covered their time range."""
        return self._memory_runtime.update_ambient_asr_watermark(
            value,
            evaluate_user_activity=evaluate_user_activity,
        )

    def finalize_memory_session(
        self,
        *,
        reason: str = "explicit_finalize",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        report = self._memory_runtime.finalize_memory_session(
            reason=reason,
            tags=tags,
        )
        input_flush = report.get("input_flush") or {}
        subreports = [
            input_flush,
            input_flush.get("user_activity") or {},
            *((input_flush.get("fact_followup_tasks") or {}).values()),
        ]
        if not any(
            item.get("reason") in {
                "queue_rejected", "worker_queue_full", "worker_shutdown",
                "task_handler_not_registered",
            }
            for item in subreports
            if isinstance(item, dict)
        ):
            self._memory_inputs_need_flush = False
        return report

    def trigger_memory_recall(
        self,
        query: str,
        *,
        tags: Optional[List[str]] = None,
        time_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._memory_runtime.trigger_memory_recall(
            query,
            tags=tags,
            time_end=time_end,
        )

    def flush_pending_memory_inputs(
        self,
        timeout: Optional[float] = None,
        *,
        tags: Optional[List[str]] = None,
    ) -> bool:
        flushed = self._memory_runtime.flush_pending_memory_inputs(
            timeout=timeout,
            tags=tags,
        )
        if flushed:
            self._memory_inputs_need_flush = False
        return flushed

    def wait_for_memory_tasks(self, timeout: Optional[float] = None) -> bool:
        return self._task_scheduler.flush(timeout=timeout)

    def trigger_user_state_time_tick(
        self, *, now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Evaluate the thirty-minute claim-to-state update window now."""
        return self._task_scheduler.trigger_user_state_time_tick(now=now)

    def get_user_states(
        self,
        *,
        state_domains: Optional[Sequence[str]] = None,
        state_types: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Read the current user model through the resolver."""
        with self._task_scheduler.execution_lock:
            owner_name = self.memory_manager.world_owner_entity_name
            owner_id = self._database.add_entity_names([owner_name])[owner_name]
            return self._user_state_resolver.get_current_states(
                user_entity_id=owner_id,
                state_domains=state_domains,
                state_types=state_types,
                statuses=statuses,
            )

    def close(self, timeout: Optional[float] = 30.0) -> None:
        if self._closed:
            return
        safe_to_close_database = False
        try:
            if self._memory_inputs_need_flush and not self.flush_pending_memory_inputs(timeout=timeout):
                raise RuntimeError("Pending memory inputs could not be submitted")
        finally:
            try:
                if not self._task_scheduler.flush(timeout=timeout):
                    self._logger.warning(
                        "PWM tasks did not drain within timeout=%s; continuing to drain",
                        timeout,
                    )
                    if not self._task_scheduler.flush(timeout=None):
                        raise RuntimeError("PWM task worker stopped with pending tasks")
                stopped = self._task_scheduler.shutdown(wait=True, timeout=timeout)
                if not stopped:
                    stopped = self._task_scheduler.shutdown(wait=True, timeout=None)
                if not stopped:
                    raise RuntimeError("PWM task worker did not stop cleanly")
                safe_to_close_database = True
            finally:
                if safe_to_close_database:
                    self._database.close()
                    self._closed = True
