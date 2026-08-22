"""MCP adapter for the unified memory runtime.

The project intentionally keeps the MCP transport small and dependency-free.
The public service class is independent from the stdio protocol, so it can be
embedded in another host later without duplicating memory-runtime behavior.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TextIO

from .memory_database import SessionDB
from .memory_manager import MemoryNodeManager
from .memory_runtime import MemoryRuntime


MCP_PROTOCOL_VERSION = "2024-11-05"


def _json_compatible(value: Any) -> Any:
    """Convert runtime reports into values accepted by JSON-RPC responses."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_compatible(item) for item in value]
    # Handles numpy scalar values without making numpy a server dependency.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_compatible(item())
        except Exception:
            pass
    return str(value)


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(value: Any, field_name: str) -> Optional[List[str]]:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array of strings")
    return [str(item) for item in value if item is not None and str(item).strip()]


class MemoryMCPService:
    """Expose the four application-facing operations over a memory runtime."""

    def __init__(
        self,
        runtime: MemoryRuntime,
        manager: MemoryNodeManager,
        database: SessionDB,
        *,
        queue_timeout: float = 30.0,
    ) -> None:
        self.runtime = runtime
        self.manager = manager
        self.database = database
        self.queue_timeout = max(0.0, float(queue_timeout))

    def accept_single_interaction_turn(
        self,
        *,
        user_message: str,
        assistant_response: str = "",
        tags: Optional[List[str]] = None,
        turn_timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Accept one user/assistant exchange and return its queue report."""
        if not str(user_message or "").strip() and not str(assistant_response or "").strip():
            raise ValueError("user_message or assistant_response must be non-empty")
        report = self.runtime.accept_single_interaction_turn(
            user_message=str(user_message or ""),
            assistant_response=str(assistant_response or ""),
            tags=tags,
            turn_timestamp=turn_timestamp,
        )
        return _json_compatible(report)

    def accept_single_transcript_segment(
        self,
        *,
        segment: Dict[str, Any],
        source_type: str = "allday_recording",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Accept one transcript segment and return its queue report."""
        if not isinstance(segment, dict):
            raise ValueError("segment must be an object")
        report = self.runtime.accept_single_transcript_segment(
            dict(segment),
            source_type=str(source_type or "allday_recording"),
            tags=tags,
        )
        return _json_compatible(report)

    def trigger_memory_reflect(
        self,
        *,
        limit: Optional[int] = None,
        reflect_timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Flush pending input, queue reflection, and return its report."""
        kwargs: Dict[str, Any] = {}
        if limit is not None:
            kwargs["limit"] = max(1, int(limit))
        if reflect_timestamp is not None:
            kwargs["reflect_timestamp"] = reflect_timestamp
        report = self.runtime.trigger_memory_reflect(**kwargs)
        return _json_compatible(report)

    def trigger_memory_recall(
        self,
        *,
        query: str,
        top_k: Optional[int] = None,
        budget: Optional[str] = None,
        tags: Optional[List[str]] = None,
        time_end: Optional[str] = None,
        recall_gate_mode: Optional[str] = None,
        memory_source_override: Optional[List[str]] = None,
        recall_path: str = "normal",
        prompt_language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run immediate recall and return memory context plus recall metadata."""
        if not str(query or "").strip():
            raise ValueError("query must be non-empty")
        kwargs: Dict[str, Any] = {
            "query": str(query),
            "recall_path": str(recall_path or "normal"),
        }
        if top_k is not None:
            kwargs["top_k"] = max(1, int(top_k))
        if budget is not None:
            kwargs["budget"] = str(budget)
        if tags is not None:
            kwargs["tags"] = tags
        if time_end is not None:
            kwargs["time_end"] = time_end
        if recall_gate_mode is not None:
            kwargs["recall_gate_mode"] = str(recall_gate_mode)
        if memory_source_override is not None:
            kwargs["memory_source_override"] = memory_source_override
        if prompt_language is not None:
            kwargs["prompt_language"] = str(prompt_language)
        report = self.runtime.trigger_memory_recall(**kwargs)
        return _json_compatible(report)

    def close(self) -> None:
        """Drain queued writes and release the database/worker resources."""
        try:
            self.runtime.flush_task_queue(timeout=self.queue_timeout)
        finally:
            try:
                self.manager.shutdown_task_worker(
                    wait=True,
                    timeout=self.queue_timeout,
                )
            finally:
                self.database.close()


TOOLS: List[Dict[str, Any]] = [
    {
        "name": "accept_single_interaction_turn",
        "description": (
            "Accept one user/assistant interaction turn. MemoryRuntime buffers "
            "turns and queues extraction when the configured boundary is reached."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_message": {"type": "string"},
                "assistant_response": {"type": "string", "default": ""},
                "tags": {"type": "array", "items": {"type": "string"}},
                "turn_timestamp": {
                    "type": "string",
                    "description": "ISO-8601 timestamp; omitted to use current time.",
                },
            },
            "required": ["user_message"],
            "additionalProperties": False,
        },
    },
    {
        "name": "accept_single_transcript_segment",
        "description": (
            "Accept one ambient transcript segment. MemoryRuntime groups compatible "
            "segments before submitting an episode for extraction."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "segment": {
                    "type": "object",
                    "description": "Transcript object containing text and optional timestamps.",
                },
                "source_type": {"type": "string", "default": "allday_recording"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["segment"],
            "additionalProperties": False,
        },
    },
    {
        "name": "trigger_memory_reflect",
        "description": (
            "Flush pending input and queue reflection to update topic/entity states "
            "and actionable items."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1},
                "reflect_timestamp": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "trigger_memory_recall",
        "description": (
            "Search committed memory immediately and return memory_context, "
            "recall path metadata, and timing information."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1},
                "budget": {"type": "string", "enum": ["low", "mid", "high"]},
                "tags": {"type": "array", "items": {"type": "string"}},
                "time_end": {"type": "string"},
                "recall_gate_mode": {"type": "string", "enum": ["auto", "force", "off"]},
                "memory_source_override": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "recall_path": {"type": "string", "enum": ["stage1", "stage2", "normal"]},
                "prompt_language": {"type": "string", "enum": ["zh", "en", "source"]},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]


class StdioMCPServer:
    """Minimal MCP stdio transport for ``MemoryMCPService``."""

    def __init__(
        self,
        service: MemoryMCPService,
        *,
        input_stream: Optional[TextIO] = None,
        output_stream: Optional[TextIO] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.service = service
        self.input_stream = input_stream or sys.stdin
        self.output_stream = output_stream or sys.stdout
        self.logger = logger or logging.getLogger(__name__)

    def serve_forever(self) -> None:
        """Process MCP JSON-RPC messages until the client closes stdin."""
        try:
            while True:
                request = self._read_message()
                if request is None:
                    break
                response = self._dispatch(request)
                if response is not None:
                    self._write_message(response)
        finally:
            self.service.close()

    def _read_message(self) -> Optional[Dict[str, Any]]:
        stream = self.input_stream
        headers: Dict[str, str] = {}
        while True:
            line = stream.buffer.readline() if hasattr(stream, "buffer") else stream.readline()
            if not line:
                return None
            if isinstance(line, bytes):
                line = line.decode("ascii")
            line = line.strip("\r\n")
            if not line:
                break
            if ":" not in line:
                raise ValueError(f"Invalid MCP header: {line}")
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
        length = int(headers.get("content-length", "0"))
        if length <= 0:
            raise ValueError("MCP message is missing Content-Length")
        raw = (
            stream.buffer.read(length)
            if hasattr(stream, "buffer")
            else stream.read(length)
        )
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        message = json.loads(raw)
        if not isinstance(message, dict):
            raise ValueError("MCP message must be a JSON object")
        return message

    def _write_message(self, message: Dict[str, Any]) -> None:
        payload = json.dumps(
            _json_compatible(message),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        stream = self.output_stream
        header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
        if hasattr(stream, "buffer"):
            stream.buffer.write(header + payload)
            stream.buffer.flush()
        else:
            try:
                stream.write((header + payload).decode("utf-8"))
            except TypeError:
                # Useful for embedding/tests that provide a raw BytesIO.
                stream.write(header + payload)
            stream.flush()

    def _dispatch(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        request_id = request.get("id")
        method = str(request.get("method") or "")
        params = request.get("params") or {}
        if request_id is None:
            # MCP notifications, including notifications/initialized, do not
            # receive a JSON-RPC response.
            return None
        try:
            if method == "initialize":
                client_version = str(params.get("protocolVersion") or MCP_PROTOCOL_VERSION)
                result = {
                    "protocolVersion": client_version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": "agent-memory",
                        "version": "0.1.0",
                    },
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                result = self._call_tool(params)
            else:
                return self._error(request_id, -32601, f"Unknown method: {method}")
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except (TypeError, ValueError, KeyError) as exc:
            self.logger.warning("MCP invalid request method=%s error=%s", method, exc)
            return self._error(request_id, -32602, str(exc))
        except Exception as exc:  # pragma: no cover - protects the stdio loop
            self.logger.exception("MCP tool failed method=%s", method)
            return self._error(request_id, -32603, str(exc))

    def _call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise ValueError("tools/call arguments must be an object")
        if name == "accept_single_interaction_turn":
            result = self.service.accept_single_interaction_turn(
                user_message=str(arguments.get("user_message") or ""),
                assistant_response=str(arguments.get("assistant_response") or ""),
                tags=_string_list(arguments.get("tags"), "tags"),
                turn_timestamp=_optional_string(arguments.get("turn_timestamp")),
            )
        elif name == "accept_single_transcript_segment":
            segment = arguments.get("segment")
            if not isinstance(segment, dict):
                raise ValueError("segment must be an object")
            result = self.service.accept_single_transcript_segment(
                segment=segment,
                source_type=str(arguments.get("source_type") or "allday_recording"),
                tags=_string_list(arguments.get("tags"), "tags"),
            )
        elif name == "trigger_memory_reflect":
            result = self.service.trigger_memory_reflect(
                limit=arguments.get("limit"),
                reflect_timestamp=_optional_string(arguments.get("reflect_timestamp")),
            )
        elif name == "trigger_memory_recall":
            result = self.service.trigger_memory_recall(
                query=str(arguments.get("query") or ""),
                top_k=arguments.get("top_k"),
                budget=_optional_string(arguments.get("budget")),
                tags=_string_list(arguments.get("tags"), "tags"),
                time_end=_optional_string(arguments.get("time_end")),
                recall_gate_mode=_optional_string(arguments.get("recall_gate_mode")),
                memory_source_override=_string_list(
                    arguments.get("memory_source_override"),
                    "memory_source_override",
                ),
                recall_path=str(arguments.get("recall_path") or "normal"),
                prompt_language=_optional_string(arguments.get("prompt_language")),
            )
        else:
            raise ValueError(f"Unknown tool: {name}")
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, ensure_ascii=False),
                }
            ],
            "structuredContent": result,
            "isError": False,
        }

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }


__all__ = ["MemoryMCPService", "StdioMCPServer", "TOOLS"]
