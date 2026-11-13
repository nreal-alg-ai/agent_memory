"""Composition root for the personal world model."""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from llm_client import LLMClient
from pwm_logging import PWMLoggingManager
from memory.memory_database import UserMemoryDB
from memory.memory_manager import MemoryNodeManager
from memory.memory_runtime import MemoryRuntime
from task_scheduler import TaskOperationReporter, TaskScheduler
from user_model.user_model_database import UserModelDB
from user_model.user_prediction import UserPrediction
from user_model.user_proactive_service import UserProactiveService
from user_model.user_state_resolver import UserStateResolver


class PersonWorldModel:
    """Own the shared memory worker, evidence runtime, and user model."""

    PROACTIVE_SERVICE_FEEDBACK_TASK_KIND = "proactive_service_feedback"

    def __init__(
        self,
        *,
        memory_runtime_config: Dict[str, Any],
        memory_manager_config: Dict[str, Any],
        user_state_config: Dict[str, Any],
        llm_model_config: Dict[str, Any],
        embedding_model_config: Dict[str, Any],
        task_scheduling_config: Dict[str, Any],
        operation_reporter: Optional[TaskOperationReporter] = None,
        logger: Optional[logging.Logger] = None,
        logging_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        runtime_config = dict(memory_runtime_config)
        manager_config = dict(memory_manager_config)
        llm_config = dict(llm_model_config)
        embedding_config = dict(embedding_model_config)
        scheduler_config = dict(task_scheduling_config)
        memory_db_value = manager_config.get("user_memory_database_path")
        user_model_db_value = user_state_config.get("user_model_database_path")
        if not memory_db_value:
            raise ValueError(
                "memory_manager_config.user_memory_database_path is required"
            )
        if not user_model_db_value:
            raise ValueError(
                "user_state_config.user_model_database_path is required"
            )
        memory_db_path = Path(memory_db_value).expanduser().resolve()
        model_db_path = Path(user_model_db_value).expanduser().resolve()
        if model_db_path == memory_db_path:
            raise ValueError("User memory and user model database paths must be distinct")
        self._logging_manager = PWMLoggingManager(
            memory_db_path=memory_db_path, config=logging_config,
        )
        # An injected application logger overrides only PWM lifecycle messages.
        self._logger = logger or self._logging_manager.get_logger("pwm")
        self._operation_reporter = operation_reporter or TaskOperationReporter()
        try:
            self._user_memory_database = UserMemoryDB(memory_db_path)
            self._user_model_database = UserModelDB(model_db_path)
            self._llm_client = LLMClient(
                memory_config={"llm": llm_config},
                logger=self._logging_manager.get_logger("llm_client"),
            )
            # Both memory and user-state handlers are registered before the
            # shared worker starts processing its priority queue.
            self._task_scheduler = TaskScheduler(
                operation_reporter=self._operation_reporter,
                maxsize=int(scheduler_config.get("task_queue_maxsize", 100) or 100),
                logger=self._logging_manager.get_logger("task_scheduler"),
                user_memory_db=self._user_memory_database,
                task_scheduling_config=scheduler_config,
                memory_enabled=manager_config.get("memory_enabled", True),
            )
            self._memory_runtime = MemoryRuntime(
                db_path=memory_db_path,
                database=self._user_memory_database,
                task_scheduler=self._task_scheduler,
                llm_client=self._llm_client,
                memory_runtime_config=runtime_config,
                memory_manager_config=manager_config,
                embedding_config=embedding_config,
                logger=self._logging_manager.get_logger("memory_runtime"),
                memory_manager_logger=self._logging_manager.get_logger("memory_manager"),
            )
            self._user_state_resolver = UserStateResolver(
                memory_db=self._user_memory_database,
                model_db=self._user_model_database,
                task_scheduler=self._task_scheduler,
                llm_client=self._llm_client,
                user_state_config=user_state_config,
                world_owner_entity_name=(
                    self._memory_runtime.manager.world_owner_entity_name
                ),
                logger=self._logging_manager.get_logger("user_state"),
            )
            self._user_prediction = UserPrediction(
                db=self._user_memory_database,
                model_db=self._user_model_database,
                task_scheduler=self._task_scheduler,
                llm_client=self._llm_client,
                world_owner_entity_name=(
                    self._memory_runtime.manager.world_owner_entity_name
                ),
                logger=self._logging_manager.get_logger("user_prediction"),
            )
            self._user_proactive_service = UserProactiveService(
                memory_db=self._user_memory_database,
                model_db=self._user_model_database,
                prediction_reader=self._user_prediction.get_active_predictions,
                task_scheduler=self._task_scheduler,
                llm_client=self._llm_client,
                world_owner_entity_name=self._memory_runtime.manager.world_owner_entity_name,
                logger=self._logging_manager.get_logger("user_proactive_service"),
            )
            self._task_scheduler.register_handler(
                self.PROACTIVE_SERVICE_FEEDBACK_TASK_KIND,
                self._process_proactive_service_feedback_task,
            )
            self._memory_runtime.manager.initialize_embedding_cache()
            self._task_scheduler.start()
        except Exception:
            self._logger.exception("PWM initialization failed")
            try:
                if hasattr(self, "_task_scheduler"):
                    self._task_scheduler.shutdown(wait=True, timeout=0)
                if hasattr(self, "_user_model_database"):
                    self._user_model_database.close()
                if hasattr(self, "_user_memory_database"):
                    self._user_memory_database.close()
            finally:
                self._logging_manager.close()
            raise
        self._closed = False
        self._memory_inputs_need_flush = False
        self._logger.info("PWM initialized log_directory=%s", self._logging_manager.directory)

    @property
    def log_directory(self) -> Path:
        return self._logging_manager.directory

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
    def user_prediction(self) -> UserPrediction:
        return self._user_prediction

    @property
    def user_proactive_service(self) -> UserProactiveService:
        return self._user_proactive_service

    @property
    def task_scheduler(self) -> TaskScheduler:
        return self._task_scheduler

    @property
    def llm_client(self) -> LLMClient:
        return self._llm_client

    @property
    def database(self) -> UserMemoryDB:
        return self._user_memory_database

    @property
    def user_model_database(self) -> UserModelDB:
        return self._user_model_database

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
        report = self._memory_runtime.flush_pending_memory_inputs(
            reason=reason,
            tags=tags,
        )
        input_flush = report.get("input_flush") or {}
        subreports = [
            input_flush,
            input_flush.get("user_activity") or {},
            input_flush.get("signal_update_dispatch") or {},
        ]
        if bool(report.get("accepted", False)) and not any(
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
        if self._closed:
            raise RuntimeError("PWM is closed")
        return self._memory_runtime.trigger_memory_recall(
            query,
            tags=tags,
            time_end=time_end,
        )

    def wait_for_memory_tasks(self, timeout: Optional[float] = None) -> bool:
        return self._task_scheduler.flush(timeout=timeout)

    def trigger_user_state_update(self) -> Dict[str, Any]:
        """Check persisted claims on a host signal and enqueue their state update."""
        return self._task_scheduler.trigger_user_state_update()

    def trigger_user_prediction_time_tick(
        self, *, reference_timestamp: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Queue expiration and defer reassessment on the shared worker."""
        return self._task_scheduler.trigger_user_prediction_time_tick(
            reference_timestamp=reference_timestamp,
        )

    def drain_pending_user_state_claims(self) -> Dict[str, Any]:
        return self._task_scheduler.drain_pending_user_state_claims()

    def get_user_states(
        self,
        *,
        state_domains: Optional[Sequence[str]] = None,
        state_types: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Read the current user model through the resolver."""
        with self._task_scheduler.execution_lock:
            return self._user_state_resolver.get_current_states(
                state_domains=state_domains,
                state_types=state_types,
                statuses=statuses,
            )

    def get_active_predictions(
        self, *, as_of: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        with self._task_scheduler.execution_lock:
            return self._user_prediction.get_active_predictions(as_of=as_of)

    def accept_proactive_service_feedback(
        self, *, service_id: int, execution_id: str, execution_status: str,
        user_response: str = "unknown", user_feedback: str = "",
        execution_result: Optional[Dict[str, Any]] = None,
        agent_note: str = "", observed_at: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Record and extract one report on the shared worker, then return its result.

        execution_id identifies the external Agent's run. user_feedback must
        be the user's actual words; execution_result is JSON-compatible Agent/
        tool output, while agent_note is interpretation, not memory evidence.
        A failed memory_status does not discard the recorded raw feedback.
        Follow-up claim/intent/state work remains asynchronous and follows the
        existing scheduler thresholds. No dialogue deduplication is performed.
        """
        if self._closed:
            raise RuntimeError("PWM is closed")
        report = self._task_scheduler.submit_and_wait(
            self.PROACTIVE_SERVICE_FEEDBACK_TASK_KIND,
            deepcopy({
                "service_id": service_id, "execution_id": execution_id,
                "execution_status": execution_status, "user_response": user_response,
                "user_feedback": user_feedback, "execution_result": execution_result,
                "agent_note": agent_note, "observed_at": observed_at,
            }),
        )
        if report.get("fact_ids"):
            # Ensure finalize/close can drain below-threshold feedback signals,
            # even if no ordinary conversation has entered MemoryRuntime.
            self._memory_inputs_need_flush = True
        return report

    def _process_proactive_service_feedback_task(self, **payload: Any) -> Dict[str, Any]:
        """Coordinate service records and memory extraction, not domain decisions."""
        feedback = self._user_proactive_service.record_service_feedback(**payload)
        try:
            memory = self.memory_manager.extract_memory_from_proactive_service_feedback(feedback)
        except Exception as exc:
            self._logger.exception(
                "Service feedback memory extraction failed feedback_id=%s service_id=%s; raw report retained",
                feedback["id"], feedback["service_id"],
            )
            memory = {"status": "failed", "reason": "feedback_memory_extraction_error",
                      "error_type": type(exc).__name__, "fact_ids": [], "new_fact_count": 0}
        report = {
            **memory,
            "status": "failed" if memory["status"] == "failed" else "ok",
            "memory_status": memory["status"], "feedback_recorded": True,
            "feedback_id": feedback["id"], "service_id": feedback["service_id"],
            "execution_id": feedback["execution_id"],
            "reference_timestamp": feedback["observed_at"],
            "contains_interaction_turn": bool(feedback["user_feedback"].strip()),
        }
        self._logger.info(
            "Service feedback processed feedback_id=%s service_id=%s status=%s memory_status=%s fact_ids=%s",
            report["feedback_id"], report["service_id"], report["status"],
            report["memory_status"], report.get("fact_ids", []),
        )
        return report

    def get_proactive_service_feedback(
        self, *, service_id: Optional[int] = None, limit: int = 50,
    ) -> List[Dict[str, Any]]:
        with self._task_scheduler.execution_lock:
            return self._user_proactive_service.get_service_feedback(service_id=service_id, limit=limit)

    def generate_proactive_services_from_predictions(
        self, *, runtime_context: Optional[Dict[str, Any]] = None,
        agent_capabilities: Optional[Dict[str, Any]] = None,
        reference_timestamp: Optional[Any] = None,
        max_predictions: int = 10,
    ) -> Dict[str, Any]:
        """Host-driven assessment; return current help, not a queue receipt.

        Queues at highest priority, waits for this task, and returns assistance.
        Does not interrupt a running task or flush background work. Waiting
        services are persisted internally; no tools are executed.
        """
        if self._closed:
            raise RuntimeError("PWM is closed")
        if runtime_context is not None and not isinstance(runtime_context, dict):
            raise ValueError("runtime_context must be an object")
        if agent_capabilities is not None and not isinstance(agent_capabilities, dict):
            raise ValueError("agent_capabilities must be an object")
        return self._task_scheduler.submit_and_wait(
            UserProactiveService.TASK_KIND,
            {
                "runtime_context": deepcopy(runtime_context),
                "agent_capabilities": deepcopy(agent_capabilities),
                "reference_timestamp": reference_timestamp,
                "max_predictions": max_predictions,
            },
        )

    def get_proactive_services(
        self, *, statuses: Optional[Sequence[str]] = None, limit: int = 100,
    ) -> List[Dict[str, Any]]:
        with self._task_scheduler.execution_lock:
            owner_name = self.memory_manager.world_owner_entity_name
            owner_id = self._user_memory_database.get_entity_id_by_name(owner_name)
            if owner_id is None:
                return []
            return self._user_proactive_service.get_services(
                user_entity_id=owner_id, statuses=statuses, limit=limit,
            )

    def close(self, timeout: Optional[float] = 30.0) -> None:
        if self._closed:
            return
        if self._memory_inputs_need_flush:
            self.finalize_memory_session(reason="pwm_close")
            if self._memory_inputs_need_flush:
                raise RuntimeError("Pending memory inputs could not be submitted")
        safe_to_close_database = False
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
                try:
                    self._user_model_database.close()
                    self._user_memory_database.close()
                    self._closed = True
                    self._logger.info("PWM closed")
                finally:
                    self._logging_manager.close()
