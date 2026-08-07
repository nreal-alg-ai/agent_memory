"""agent_memory 包装层：统一初始化、写入、召回、反思与关闭接口。"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Optional


class AgentMemoryRuntime:
    """包装 agent_memory 的 SessionDB + MemoryNodeManager + MemoryRuntime。"""

    def __init__(
        self,
        db_path: str | Path,
        *,
        llm_config: Optional[Dict[str, Any]] = None,
        embedding_config: Optional[Dict[str, Any]] = None,
        memory_manager_config: Optional[Dict[str, Any]] = None,
        memory_runtime_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        from memory.memory_database import SessionDB
        from memory.memory_manager import MemoryNodeManager
        from memory.memory_runtime import MemoryRuntime

        self._db_path = str(db_path)
        self._db = SessionDB(db_path=self._db_path)
        self._manager = MemoryNodeManager(
            self._db,
            embedding_config=dict(embedding_config or {}),
            memory_manager_config=dict(memory_manager_config or {}),
            llm_config=dict(llm_config or {}),
        )
        self._runtime = MemoryRuntime(
            self._manager,
            memory_runtime_config=dict(memory_runtime_config or {}),
        )
        self._lock = threading.RLock()
        self._closed = False

    @property
    def db_path(self) -> str:
        return self._db_path

    def ingest_ambient_segment(
        self,
        *,
        speaker: str,
        text: str,
        timestamp: Optional[float] = None,
        source_type: str = "allday_recording",
    ) -> Dict[str, Any]:
        """环境转写片段 -> agent_memory，并等待后台任务落库。"""
        segment: Dict[str, Any] = {"speaker": speaker or "unknown", "text": text}
        if timestamp is not None:
            segment["timestamp"] = timestamp
        with self._lock:
            if self._closed:
                return {"queued": False, "reason": "runtime_closed"}
            result = self._runtime.accept_single_transcript_segment(
                segment=segment,
                source_type=source_type,
            )
            self._runtime.flush_task_queue(timeout=10.0)
            return result

    def store_interaction(self, user_message: str, assistant_reply: str) -> Dict[str, Any]:
        """对话轮次 -> agent_memory，并等待后台任务落库。"""
        with self._lock:
            if self._closed:
                return {"queued": False, "reason": "runtime_closed"}
            result = self._runtime.accept_single_interaction_turn(
                user_message=user_message,
                assistant_response=assistant_reply,
            )
            self._runtime.flush_task_queue(timeout=10.0)
            return result

    def recall(self, query: str, top_k: int = 8, budget: str = "mid") -> Dict[str, Any]:
        """召回记忆上下文，返回 agent_memory 的完整报告。"""
        with self._lock:
            if self._closed:
                return {"memory_context": "", "status": "closed", "elapsed_ms": 0.0}
            return self._runtime.trigger_memory_recall(query=query, top_k=top_k, budget=budget)

    def trigger_reflect(self) -> Dict[str, Any]:
        """触发 state 反思，并等待反思任务完成。"""
        with self._lock:
            if self._closed:
                return {"queued": False, "reason": "runtime_closed"}
            report = self._runtime.trigger_memory_reflect()
            self._runtime.flush_task_queue(timeout=20.0)
            return report

    def flush(self, timeout: float = 10.0) -> bool:
        with self._lock:
            return self._runtime.flush_task_queue(timeout=timeout)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._runtime.flush_task_queue(timeout=10.0)
            except Exception:
                pass
            try:
                self._db.close()
            except Exception:
                pass
