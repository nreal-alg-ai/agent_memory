"""MCP adapter for audio-backed memory ingestion and memory recall."""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from memory.memory_runtime import MemoryRuntime
from voice.voice_runtime import VoiceRuntime


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
    """Expose audio-to-memory ingestion and memory recall over MCP."""

    def __init__(
        self,
        memory_runtime: MemoryRuntime,
        voice_runtime: Optional["VoiceRuntime"] = None,
        *,
        voice_runtime_factory: Optional[Callable[[], "VoiceRuntime"]] = None,
        queue_timeout: float = 30.0,
        asr_result_dir: Optional[Path | str] = None,
    ) -> None:
        self.memory_runtime = memory_runtime
        self.voice_runtime = voice_runtime
        self._voice_runtime_factory = voice_runtime_factory
        self._voice_runtime_lock = threading.Lock()
        self._asr_result_lock = threading.Lock()
        self._stdio_framing: Optional[str] = None
        self.queue_timeout = max(0.0, float(queue_timeout))
        self.asr_result_dir = (
            Path(asr_result_dir).expanduser().resolve()
            if asr_result_dir is not None and str(asr_result_dir).strip()
            else None
        )

    def _get_voice_runtime(self) -> "VoiceRuntime":
        """Create the voice runtime only when audio processing is requested."""
        if self.voice_runtime is not None:
            return self.voice_runtime
        if self._voice_runtime_factory is None:
            raise RuntimeError("Voice runtime is not configured")
        with self._voice_runtime_lock:
            if self.voice_runtime is None:
                self.voice_runtime = self._voice_runtime_factory()
        return self.voice_runtime

    def process_audio_file(
        self,
        *,
        audio_path: str,
        source_type: str = "allday_recording",
        session_start: Optional[str] = None,
        tags: Optional[List[str]] = None,
        run_reflect: bool = True,
    ) -> Dict[str, Any]:
        """Transcribe an audio file and submit its segments to memory."""
        if not str(audio_path or "").strip():
            raise ValueError("audio_path must be non-empty")
        source = str(source_type or "allday_recording").strip()
        voice_report = self._get_voice_runtime().process_audio_file(
            audio_path,
            session_start=session_start,
        )
        segments = list(voice_report.get("segments") or [])
        asr_result_path = self._save_asr_result(voice_report)
        store_reports: List[Dict[str, Any]] = []
        for segment in segments:
            store_reports.append(
                self.memory_runtime.accept_single_transcript_segment(
                    segment,
                    source_type=source,
                    tags=tags,
                )
            )

        queue_flushed = self.memory_runtime.flush_task_queue(timeout=self.queue_timeout)
        reflect_report: Optional[Dict[str, Any]] = None
        reflect_flushed: Optional[bool] = None
        if run_reflect and queue_flushed and segments:
            reflect_report = self.memory_runtime.trigger_memory_reflect()
            reflect_flushed = self.memory_runtime.flush_task_queue(
                timeout=self.queue_timeout,
            )

        queued_count = sum(bool(report.get("queued")) for report in store_reports)
        report = {
            "status": "ok" if queue_flushed else "queue_flush_timeout",
            "source_type": source,
            "audio": {
                key: value
                for key, value in voice_report.items()
                if key != "segments"
            },
            "asr_result_path": str(asr_result_path) if asr_result_path else None,
            "transcript_segments": segments,
            "submitted_segment_count": len(store_reports),
            "store_queue_event_count": queued_count,
            "store_reports": store_reports,
            "store_queue_flushed": queue_flushed,
            "run_reflect": bool(run_reflect),
            "reflect_report": reflect_report,
            "reflect_queue_flushed": reflect_flushed,
        }
        return _json_compatible(report)

    def _save_asr_result(self, voice_report: Dict[str, Any]) -> Optional[Path]:
        """Persist one complete VoiceRuntime report after audio transcription."""
        if self.asr_result_dir is None:
            return None

        audio_path = Path(str(voice_report.get("audio_path") or "audio"))
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", audio_path.stem).strip("._")
        safe_stem = safe_stem or "audio"
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        with self._asr_result_lock:
            self.asr_result_dir.mkdir(parents=True, exist_ok=True)
            result_path = self.asr_result_dir / f"{timestamp}_{safe_stem}.json"
            payload = {
                "created_at": datetime.now().astimezone().isoformat(),
                **voice_report,
            }
            serialized = json.dumps(
                _json_compatible(payload),
                ensure_ascii=False,
                indent=2,
            )
            temp_path = result_path.with_suffix(result_path.suffix + ".tmp")
            temp_path.write_text(serialized, encoding="utf-8")
            temp_path.replace(result_path)
        return result_path

    def process_audio_files(
        self,
        *,
        files: List[Dict[str, Any]],
        source_type: str = "allday_recording",
        session_start: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Process multiple files sequentially using the single-file workflow.

        Each file independently runs ``process_audio_file``. This preserves the
        existing per-file MemoryRuntime flush and reflect behavior while keeping
        the voice models resident for the duration of the batch.
        """
        if not isinstance(files, list) or not files:
            raise ValueError("files must be a non-empty array")

        file_reports: List[Dict[str, Any]] = []
        for index, file_spec in enumerate(files, 1):
            if not isinstance(file_spec, dict):
                file_reports.append({
                    "status": "failed",
                    "file_index": index,
                    "error": "each files item must be an object",
                })
                continue
            path = str(file_spec.get("audio_path") or "").strip()
            if not path:
                file_reports.append({
                    "status": "failed",
                    "file_index": index,
                    "error": "audio_path must be non-empty",
                })
                continue

            file_tags = list(tags or [])
            item_tags = file_spec.get("tags")
            if item_tags is not None:
                if not isinstance(item_tags, list):
                    file_reports.append({
                        "status": "failed",
                        "file_index": index,
                        "audio_path": path,
                        "error": "files[].tags must be an array of strings",
                    })
                    continue
                file_tags.extend(str(tag) for tag in item_tags if str(tag).strip())

            try:
                report = self.process_audio_file(
                    audio_path=path,
                    source_type=str(file_spec.get("source_type") or source_type),
                    session_start=(
                        _optional_string(file_spec.get("session_start"))
                        or session_start
                    ),
                    tags=file_tags,
                    run_reflect=True,
                )
                report["file_index"] = index
                file_reports.append(report)
            except Exception as exc:
                file_reports.append({
                    "status": "failed",
                    "file_index": index,
                    "audio_path": path,
                    "error": str(exc),
                })

        succeeded = [report for report in file_reports if report.get("status") == "ok"]
        failed = [report for report in file_reports if report.get("status") == "failed"]
        status = "ok" if not failed else "partial_failure" if succeeded else "failed"
        return _json_compatible({
            "status": status,
            "file_count": len(file_reports),
            "succeeded_file_count": len(succeeded),
            "failed_file_count": len(failed),
            "submitted_segment_count": sum(
                int(report.get("submitted_segment_count") or 0)
                for report in succeeded
            ),
            "store_queue_event_count": sum(
                int(report.get("store_queue_event_count") or 0)
                for report in succeeded
            ),
            "files": file_reports,
        })

    def trigger_memory_recall(
        self,
        *,
        query: str,
        tags: Optional[List[str]] = None,
        time_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run immediate recall and return memory context plus metadata."""
        if not str(query or "").strip():
            raise ValueError("query must be non-empty")
        report = self.memory_runtime.trigger_memory_recall(
            str(query),
            tags=tags,
            time_end=time_end,
        )
        return _json_compatible(report)

    def close(self) -> None:
        """Drain queued writes and release memory and voice resources."""
        try:
            self.memory_runtime.close(timeout=self.queue_timeout)
        finally:
            close_voice = getattr(self.voice_runtime, "close", None)
            if callable(close_voice):
                close_voice()


TOOLS: List[Dict[str, Any]] = [
    {
        "name": "process_audio_files",
        "description": (
            "Process multiple audio files sequentially with shared VAD, ASR, and "
            "speaker models. Each file is independently stored and reflected."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "audio_path": {"type": "string"},
                            "source_type": {"type": "string"},
                            "session_start": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["audio_path"],
                        "additionalProperties": False,
                    },
                },
                "source_type": {"type": "string", "default": "allday_recording"},
                "session_start": {
                    "type": "string",
                    "description": "Optional ISO-8601 timestamp for audio time zero.",
                },
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["files"],
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
                "tags": {"type": "array", "items": {"type": "string"}},
                "time_end": {"type": "string"},
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
        first_line = stream.buffer.readline() if hasattr(stream, "buffer") else stream.readline()
        if not first_line:
            return None
        if isinstance(first_line, bytes):
            first_line = first_line.decode("utf-8")
        first_line = first_line.strip("\r\n")
        if not first_line:
            return self._read_message()

        # MCP stdio normally uses one JSON-RPC object per line. Keep accepting
        # LSP-style Content-Length frames for older local MCP clients.
        if first_line.lstrip().startswith(("{", "[")):
            self._stdio_framing = "newline"
            message = json.loads(first_line)
            if not isinstance(message, dict):
                raise ValueError("MCP message must be a JSON object")
            return message

        headers: Dict[str, str] = {}
        if ":" not in first_line:
            raise ValueError(f"Invalid MCP header: {first_line}")
        key, value = first_line.split(":", 1)
        headers[key.strip().lower()] = value.strip()
        while True:
            line = stream.buffer.readline() if hasattr(stream, "buffer") else stream.readline()
            if not line:
                return None
            if isinstance(line, bytes):
                line = line.decode("utf-8")
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
        self._stdio_framing = "content-length"
        return message

    def _write_message(self, message: Dict[str, Any]) -> None:
        payload = json.dumps(
            _json_compatible(message),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        stream = self.output_stream
        if self._stdio_framing == "content-length":
            output = (
                f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
                + payload
            )
        else:
            output = payload + b"\n"
        if hasattr(stream, "buffer"):
            stream.buffer.write(output)
            stream.buffer.flush()
        else:
            try:
                stream.write(output.decode("utf-8"))
            except TypeError:
                stream.write(output)
            stream.flush()

    def _dispatch(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        request_id = request.get("id")
        method = str(request.get("method") or "")
        params = request.get("params") or {}
        if request_id is None:
            return None
        try:
            if method == "initialize":
                client_version = str(params.get("protocolVersion") or MCP_PROTOCOL_VERSION)
                result = {
                    "protocolVersion": client_version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "agent-memory", "version": "0.1.0"},
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
        if name == "process_audio_files":
            files = arguments.get("files")
            if not isinstance(files, list):
                raise ValueError("files must be an array")
            result = self.service.process_audio_files(
                files=files,
                source_type=str(arguments.get("source_type") or "allday_recording"),
                session_start=_optional_string(arguments.get("session_start")),
                tags=_string_list(arguments.get("tags"), "tags"),
            )
        elif name == "trigger_memory_recall":
            result = self.service.trigger_memory_recall(
                query=str(arguments.get("query") or ""),
                tags=_string_list(arguments.get("tags"), "tags"),
                time_end=_optional_string(arguments.get("time_end")),
            )
        else:
            raise ValueError(f"Unknown tool: {name}")
        return {
            "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
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
