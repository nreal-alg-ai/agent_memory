"""Agent Memory 测试 App 的本地 HTTP 后端。

对外提供与 android_runtime.py 同名同签名的 Kotlin 桥接函数，保证 Android 外壳零改动；
同时为定制前端提供 /api/agent-memory/* 与 /api/chat 等 HTTP 端点。
"""

from __future__ import annotations

import hmac
import json
import mimetypes
import os
import secrets
import sqlite3
import threading
import time
import urllib.parse
import uuid
import zipfile
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent_memory_runtime import AgentMemoryRuntime
from .audio_event_handler import (
    is_assistant_query,
    is_speaker_enrollment,
    preprocess_audio_event,
)
from .memory_admin import AgentMemoryAdmin
from .simple_llm_chat import ChatServiceError, chat_with_memory


_CONFIG_KEYS = (
    "app_home",
    "static_dir",
    "provider",
    "model",
    "base_url",
    "api_key",
    "owner_id",
)


def _log(message: str) -> None:
    try:
        log_dir = Path(os.environ.get("AI_GLASSES_APP_HOME", "/tmp")) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / "agent_memory_backend.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    except Exception:
        pass


class SpeakerStore:
    """声纹样本存储与分类（包装层自己的 JSON 文件，不含原始 PCM）。"""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        self._data: Dict[str, Dict[str, Any]] = {}
        if self._path.is_file():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._data = raw
            except (OSError, json.JSONDecodeError):
                pass

    def add_sample(
        self,
        user_id: str,
        embedding: List[float],
        model: str,
        session_id: str,
        sample_index: int,
        sample_total: int,
    ) -> Dict[str, Any]:
        if not embedding:
            return {"state": "error", "reason": "empty_embedding"}
        with self._lock:
            profile = self._data.setdefault(
                user_id,
                {"model": model, "samples": [], "enrolled": False},
            )
            profile["model"] = model
            profile["samples"].append([float(value) for value in embedding])
            profile["enrolled"] = len(profile["samples"]) >= max(1, int(sample_total or 3))
            self._save()
            return {
                "state": "completed" if profile["enrolled"] else "collecting",
                "sample_index": len(profile["samples"]),
                "sample_total": max(1, int(sample_total or 3)),
                "enrolled": profile["enrolled"],
            }

    def profile(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            profile = self._data.get(user_id)
            if not profile or not profile.get("samples"):
                return {"enrolled": False, "sample_count": 0, "model": "", "enrollment_status": "none"}
            return {
                "enrolled": bool(profile.get("enrolled")),
                "sample_count": len(profile["samples"]),
                "model": str(profile.get("model") or ""),
                "enrollment_status": "ready" if profile.get("enrolled") else "collecting",
            }

    def classify(self, user_id: str, embedding: List[float], model: str) -> Dict[str, Any]:
        if not embedding:
            return {"state": "unknown", "label": "", "similarity": 0.0, "model": model, "threshold": 0.5}
        with self._lock:
            profile = self._data.get(user_id)
            samples = profile.get("samples") if profile else None
            if not samples:
                return {"state": "unknown", "label": "", "similarity": 0.0, "model": model, "threshold": 0.5}
            import numpy as np

            query = np.asarray(embedding, dtype=np.float32)
            query_norm = float(np.linalg.norm(query)) or 1e-6
            best = 0.0
            for sample in samples:
                vector = np.asarray(sample, dtype=np.float32)
                norm = float(np.linalg.norm(vector)) or 1e-6
                similarity = float(np.dot(query, vector) / (query_norm * norm))
                best = max(best, similarity)
            threshold = 0.5
            return {
                "state": "matched" if best >= threshold else "unmatched",
                "label": "user" if best >= threshold else "",
                "similarity": round(best, 4),
                "model": model,
                "threshold": threshold,
            }

    def cancel_session(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            profile = self._data.get(user_id)
            if profile:
                profile["samples"] = []
                profile["enrolled"] = False
                self._save()
            return {"cancelled": True}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )


class ReplyQueue:
    """唤醒问句处理结果队列，供 Kotlin queue_status / wait_audio_event 轮询。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._events: Dict[str, Dict[str, Any]] = {}

    def mark_running(self, event_id: str) -> None:
        with self._lock:
            self._events[event_id] = {"event_id": event_id, "status": "running", "dispatch": None}

    def mark_completed(self, event_id: str, result: Dict[str, Any]) -> None:
        with self._lock:
            self._events[event_id] = {
                "event_id": event_id,
                "status": "completed",
                "dispatch": {"result": result},
            }
            self._prune()

    def mark_failed(self, event_id: str, error: str) -> None:
        with self._lock:
            self._events[event_id] = {
                "event_id": event_id,
                "status": "failed",
                "dispatch": {"result": {"error": str(error)[:500]}},
            }
            self._prune()

    def snapshot(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            events = sorted(self._events.values(), key=lambda item: item["event_id"], reverse=True)
            return events[: max(1, int(limit or 20))]

    def wait(self, event_id: str, timeout: float) -> Optional[Dict[str, Any]]:
        deadline = time.monotonic() + max(0.0, min(float(timeout), 30.0))
        while time.monotonic() < deadline:
            with self._lock:
                event = self._events.get(event_id)
                if event and event.get("status") in {"completed", "failed"}:
                    return event
            time.sleep(0.1)
        with self._lock:
            return self._events.get(event_id)

    def _prune(self) -> None:
        if len(self._events) <= 200:
            return
        oldest = sorted(self._events.items(), key=lambda item: item[0])[: len(self._events) - 200]
        for event_id, _ in oldest:
            self._events.pop(event_id, None)


class CaptureRegistry:
    """capture 会话注册表（不保存音频数据，原始 PCM 已由外壳丢弃）。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._captures: Dict[str, Dict[str, Any]] = {}

    def start(self, user_id: str) -> Dict[str, Any]:
        capture_id = f"cap_{uuid.uuid4().hex[:12]}"
        with self._lock:
            self._captures[capture_id] = {
                "capture_id": capture_id,
                "user_id": user_id,
                "status": "active",
                "started_at_ms": int(time.time() * 1000),
                "segment_count": 0,
            }
            return dict(self._captures[capture_id])

    def add_segment(self, capture_id: str) -> None:
        with self._lock:
            capture = self._captures.get(capture_id)
            if capture:
                capture["segment_count"] = int(capture.get("segment_count") or 0) + 1

    def stop(self, capture_id: str) -> Dict[str, Any]:
        with self._lock:
            capture = self._captures.get(capture_id)
            if capture is None:
                return {"capture_id": capture_id, "status": "not_found", "segment_count": 0}
            capture["status"] = "stopped"
            return dict(capture)

    def status(self, capture_id: str) -> Dict[str, Any]:
        with self._lock:
            capture = self._captures.get(capture_id)
            return dict(capture) if capture else {"capture_id": capture_id, "status": "none", "segment_count": 0}


class _BackendRuntime:
    def __init__(
        self,
        *,
        runtime: AgentMemoryRuntime,
        admin: AgentMemoryAdmin,
        speakers: SpeakerStore,
        replies: ReplyQueue,
        captures: CaptureRegistry,
        server: ThreadingHTTPServer,
        thread: threading.Thread,
        token: str,
        owner_id: str,
        platform: str,
        app_home: str,
        static_dir: str,
        llm_config: Dict[str, str],
    ) -> None:
        self.runtime = runtime
        self.admin = admin
        self.speakers = speakers
        self.replies = replies
        self.captures = captures
        self.server = server
        self.thread = thread
        self.token = token
        self.owner_id = owner_id
        self.platform = platform
        self.app_home = app_home
        self.static_dir = static_dir
        self.llm_config = llm_config


_lock = threading.RLock()
_runtime: Optional[_BackendRuntime] = None
_device_state: Dict[str, Any] = {}


def _require_runtime() -> _BackendRuntime:
    with _lock:
        if _runtime is None:
            raise RuntimeError("agent memory runtime is not running")
        return _runtime


def _runtime_for_owner(user_id: str) -> _BackendRuntime:
    runtime = _require_runtime()
    if str(user_id or "").strip() != runtime.owner_id:
        raise ValueError("user_id does not match this device owner")
    return runtime


def _parse_config(config_json: str) -> Dict[str, str]:
    try:
        raw = json.loads(config_json)
    except json.JSONDecodeError as exc:
        raise ValueError("agent memory runtime config must be valid JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("agent memory runtime config must be a JSON object")
    config = {key: str(raw.get(key) or "").strip() for key in _CONFIG_KEYS}
    config["platform"] = str(raw.get("platform") or "android").strip().lower()
    missing = [key for key, value in config.items() if not value]
    if missing:
        raise ValueError("agent memory runtime config missing: " + ", ".join(sorted(missing)))
    if not config["base_url"].startswith("https://"):
        raise ValueError("agent memory runtime base_url must use https")
    if config["platform"] not in {"android", "ios"}:
        raise ValueError("mobile runtime platform must be android or ios")
    return config


def _json_object(raw: str, label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _public_payload(runtime: _BackendRuntime) -> Dict[str, Any]:
    _, port = runtime.server.server_address
    return {
        "running": True,
        "base_url": f"http://127.0.0.1:{port}",
        "local_token": runtime.token,
        "owner_id": runtime.owner_id,
        "platform": runtime.platform,
    }


def start(config_json: str) -> str:
    """初始化 agent_memory runtime + 本地 HTTP server，返回 base_url + token。"""

    config = _parse_config(config_json)
    global _runtime
    with _lock:
        if _runtime is not None:
            return json.dumps(_public_payload(_runtime), ensure_ascii=False)
        app_home = Path(config["app_home"])
        data_dir = app_home / "data" / "agent_memory"
        data_dir.mkdir(parents=True, exist_ok=True)
        os.environ["AI_GLASSES_APP_HOME"] = str(app_home)

        llm_config = {
            "llm_name": config["model"],
            "llm_base_url": config["base_url"],
            "llm_api_key": config["api_key"],
            "llm_timeout": 120,
            "llm_json_mode": True,
        }
        embedding_config = {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "base_url": config["base_url"],
            "api_key": config["api_key"],
            "timeout": 3,
        }
        memory_manager_config = {"memory_enabled": True}
        memory_runtime_config = {
            "max_pending_interaction_turns": 1,
            "max_pending_interaction_chars": 2000,
            "transcript_episode_max_segments": 1,
            "transcript_episode_max_chars": 2000,
            "transcript_episode_max_gap_seconds": 60,
        }
        runtime = AgentMemoryRuntime(
            db_path=data_dir / "memory.db",
            llm_config=llm_config,
            embedding_config=embedding_config,
            memory_manager_config=memory_manager_config,
            memory_runtime_config=memory_runtime_config,
        )
        admin = AgentMemoryAdmin(db_path=str(data_dir / "memory.db"))
        speakers = SpeakerStore(path=str(data_dir / "speaker_profiles.json"))
        replies = ReplyQueue()
        captures = CaptureRegistry()
        token = secrets.token_urlsafe(32)
        handler = type(
            "AgentMemoryHandler",
            (_Handler,),
            {
                "local_auth_token": token,
                "static_dir": config["static_dir"],
                "platform": config["platform"],
                "owner_id": config["owner_id"],
            },
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, name="agent-memory-local-http", daemon=True)
        thread.start()
        _runtime = _BackendRuntime(
            runtime=runtime,
            admin=admin,
            speakers=speakers,
            replies=replies,
            captures=captures,
            server=server,
            thread=thread,
            token=token,
            owner_id=config["owner_id"],
            platform=config["platform"],
            app_home=str(app_home),
            static_dir=config["static_dir"],
            llm_config={
                "base_url": config["base_url"],
                "api_key": config["api_key"],
                "model": config["model"],
            },
        )
        _log(f"HTTP server bound 127.0.0.1:{server.server_address[1]} thread_alive={thread.is_alive()}")
        return json.dumps(_public_payload(_runtime), ensure_ascii=False)


def status() -> str:
    with _lock:
        payload: Dict[str, Any] = {"running": False}
        if _runtime is not None:
            payload = _public_payload(_runtime)
        return json.dumps(payload, ensure_ascii=False)


def stop() -> str:
    global _runtime
    with _lock:
        runtime = _runtime
        _runtime = None
    if runtime is None:
        return json.dumps({"running": False}, ensure_ascii=False)
    try:
        runtime.server.shutdown()
        runtime.server.server_close()
    except Exception:
        pass
    runtime.thread.join(timeout=5.0)
    runtime.admin.close()
    runtime.runtime.close()
    return json.dumps({"running": False}, ensure_ascii=False)


def start_capture(user_id: str) -> str:
    runtime = _runtime_for_owner(user_id)
    return json.dumps(runtime.captures.start(user_id), ensure_ascii=False)


def stop_capture(user_id: str, capture_id: str) -> str:
    runtime = _runtime_for_owner(user_id)
    return json.dumps(runtime.captures.stop(str(capture_id or "").strip()), ensure_ascii=False)


def capture_status(user_id: str, capture_id: str) -> str:
    runtime = _runtime_for_owner(user_id)
    return json.dumps(runtime.captures.status(str(capture_id or "").strip()), ensure_ascii=False)


def ingest_audio_event(
    user_id: str,
    capture_id: str,
    event_json: str,
    private_json: str = "{}",
    turn_context_json: str = "{}",
) -> str:
    runtime = _runtime_for_owner(user_id)
    event = _json_object(event_json, "audio event")
    private = _json_object(private_json, "private audio event")
    _json_object(turn_context_json, "audio turn context")
    event_id = str(event.get("event_id") or f"evt_{uuid.uuid4().hex[:12]}")

    if is_speaker_enrollment(event):
        return json.dumps(_handle_enrollment(runtime, event_id, private), ensure_ascii=False)
    if is_assistant_query(event):
        return json.dumps(_queue_assistant_query(runtime, event_id, event, private), ensure_ascii=False)
    return json.dumps(_ingest_ambient(runtime, event_id, str(capture_id or "").strip(), event, private), ensure_ascii=False)


def _handle_enrollment(runtime: _BackendRuntime, event_id: str, private: Dict[str, Any]) -> Dict[str, Any]:
    embedding = private.get("speaker_embedding") if isinstance(private.get("speaker_embedding"), list) else []
    model = str(private.get("speaker_embedding_model") or "sherpa_campplus").strip()
    session_id = str(private.get("enrollment_session_id") or "").strip()
    sample_index = int(private.get("sample_index") or 0)
    sample_total = int(private.get("sample_total") or 3)
    result = runtime.speakers.add_sample(
        user_id=runtime.owner_id,
        embedding=[float(value) for value in embedding],
        model=model,
        session_id=session_id,
        sample_index=sample_index,
        sample_total=sample_total,
    )
    runtime.replies.mark_completed(event_id, {"enrollment": result})
    return {
        "event_id": event_id,
        "status": "completed",
        "dispatch": {"result": {"enrollment": result}},
    }


def _queue_assistant_query(
    runtime: _BackendRuntime,
    event_id: str,
    event: Dict[str, Any],
    private: Dict[str, Any],
) -> Dict[str, Any]:
    preprocessed = preprocess_audio_event(event, private)
    query = (preprocessed or {}).get("text") or str(event.get("text") or "").strip()
    runtime.replies.mark_running(event_id)

    def _process() -> None:
        try:
            recall = runtime.runtime.recall(query=query, top_k=8, budget="mid")
            memory_context = str(recall.get("memory_context") or "")
            reply = chat_with_memory(
                user_message=query,
                memory_context=memory_context,
                llm_config=_runtime_llm_config(),
            )
            try:
                runtime.runtime.store_interaction(user_message=query, assistant_reply=reply)
            except Exception as exc:
                _log(f"store_interaction failed: {exc}")
            runtime.replies.mark_completed(
                event_id,
                {
                    "reply": reply,
                    "memory_context": memory_context,
                    "recall_path": recall.get("actual_recall_path"),
                    "recall_elapsed_ms": recall.get("elapsed_ms"),
                    "engine": "agent_memory",
                },
            )
        except Exception as exc:
            _log(f"assistant query failed: {exc}")
            runtime.replies.mark_failed(event_id, str(exc))

    threading.Thread(target=_process, name=f"assistant-query-{event_id[:8]}", daemon=True).start()
    return {"event_id": event_id, "status": "pending"}


def _ingest_ambient(
    runtime: _BackendRuntime,
    event_id: str,
    capture_id: str,
    event: Dict[str, Any],
    private: Dict[str, Any],
) -> Dict[str, Any]:
    preprocessed = preprocess_audio_event(event, private)
    if preprocessed is None:
        return {"event_id": event_id, "status": "completed", "stored": False, "reason": "filtered"}
    if capture_id:
        runtime.captures.add_segment(capture_id)
    try:
        result = runtime.runtime.ingest_ambient_segment(
            speaker=preprocessed["speaker"],
            text=preprocessed["text"],
            timestamp=preprocessed.get("timestamp"),
            source_type=preprocessed.get("source_type") or "allday_recording",
        )
        return {
            "event_id": event_id,
            "status": "completed",
            "stored": True,
            "result": result,
            "speaker": preprocessed["speaker"],
        }
    except Exception as exc:
        _log(f"ambient ingest failed: {exc}")
        return {"event_id": event_id, "status": "failed", "stored": False, "error": str(exc)[:300]}


def location_preflight(message: str) -> str:
    text = str(message or "").strip()
    location_markers = (
        "哪里",
        "在哪",
        "位置",
        "附近",
        "天气",
        "多远",
        "导航",
        "营业",
        "几点开门",
        "几点关门",
        "路线",
        "怎么去",
    )
    needed = any(marker in text for marker in location_markers)
    return json.dumps(
        {"needed": needed, "reason": "location_keyword" if needed else "no_location_marker"},
        ensure_ascii=False,
    )


def set_network_state(online: bool) -> str:
    runtime = _require_runtime()
    with _lock:
        _device_state["network_online"] = bool(online)
    return json.dumps({"online": bool(online)}, ensure_ascii=False)


_ALLOWED_DEVICE_KEYS = (
    "state",
    "running",
    "capture_id",
    "started_at_ms",
    "duration_seconds",
    "captured_samples",
    "audio_rms_dbfs",
    "audio_peak_dbfs",
    "audio_level_at_ms",
    "vad_segment_count",
    "ambient_final_count",
    "speech_rejected_count",
    "last_final_at_ms",
    "sample_rate",
    "channels",
    "encoding",
    "network_online",
    "latest_partial",
    "partial_sequence",
    "inference_queue_depth",
    "enrollment_state",
    "enrollment_session_id",
    "enrollment_sample_count",
    "enrollment_sample_total",
    "model_state",
    "model_version",
    "model_self_test",
    "transcription_ready",
    "input_device_type",
    "input_device_source",
    "input_device_name",
    "pss_kb",
    "last_error",
    "device_event_queue",
)


def set_device_state(state_json: str) -> str:
    state = _json_object(state_json, "android device state")
    public = {key: state[key] for key in _ALLOWED_DEVICE_KEYS if key in state}
    with _lock:
        _device_state.clear()
        _device_state.update(public)
    return json.dumps(public, ensure_ascii=False)


def queue_status(user_id: str) -> str:
    runtime = _runtime_for_owner(user_id)
    events = runtime.replies.snapshot(limit=20)
    summary = {
        "pending": sum(1 for item in events if item["status"] == "pending"),
        "running": sum(1 for item in events if item["status"] == "running"),
        "completed": sum(1 for item in events if item["status"] == "completed"),
        "failed": sum(1 for item in events if item["status"] == "failed"),
        "network_online": bool(_device_state.get("network_online", True)),
    }
    return json.dumps({"summary": summary, "events": events}, ensure_ascii=False)


def wait_audio_event(user_id: str, event_id: str, timeout: float = 10.0) -> str:
    runtime = _runtime_for_owner(user_id)
    event = runtime.replies.wait(str(event_id or "").strip(), max(0.0, min(float(timeout), 30.0)))
    return json.dumps(event or {}, ensure_ascii=False)


def speaker_profile(user_id: str) -> str:
    runtime = _runtime_for_owner(user_id)
    return json.dumps(runtime.speakers.profile(user_id), ensure_ascii=False)


def cancel_speaker_enrollment(user_id: str, enrollment_session_id: str = "") -> str:
    runtime = _runtime_for_owner(user_id)
    return json.dumps(runtime.speakers.cancel_session(user_id), ensure_ascii=False)


def classify_speaker(user_id: str, embedding_json: str, model_name: str) -> str:
    runtime = _runtime_for_owner(user_id)
    try:
        embedding = json.loads(embedding_json)
    except json.JSONDecodeError as exc:
        raise ValueError("speaker embedding must be valid JSON") from exc
    if not isinstance(embedding, list):
        raise ValueError("speaker embedding must be an array")
    return json.dumps(
        runtime.speakers.classify(
            user_id=user_id,
            embedding=[float(value) for value in embedding],
            model=str(model_name or "").strip(),
        ),
        ensure_ascii=False,
    )


def retry_failed(user_id: str, event_id: str = "") -> str:
    return json.dumps({"retried": 0, "note": "agent memory backend processes events synchronously"}, ensure_ascii=False)


def create_diagnostic_bundle(app_home: str, output_path: str, device_json: str = "{}") -> str:
    """导出诊断 ZIP：agent_memory DB + 后端日志 + 元数据；不含声纹与 API key。"""

    root = Path(str(app_home or "")).expanduser().resolve()
    destination = Path(str(output_path or "")).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("android app home does not exist")
    if not destination.is_absolute() or not destination.parent.is_dir():
        raise ValueError("diagnostic output directory does not exist")
    device = _json_object(device_json, "android diagnostic device state")
    temporary_zip = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    exported: List[str] = []
    skipped: Dict[str, str] = {}
    try:
        with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            db_path = root / "data" / "agent_memory" / "memory.db"
            if db_path.is_file():
                archive.write(db_path, "agent_memory/memory.db")
                exported.append("agent_memory/memory.db")
            else:
                skipped["agent_memory/memory.db"] = "missing"
            log_path = root / "logs" / "agent_memory_backend.log"
            if log_path.is_file():
                archive.write(log_path, "logs/agent_memory_backend.log")
                exported.append("logs/agent_memory_backend.log")
            archive.writestr(
                "diagnostics.json",
                json.dumps(
                    {
                        "schema": "agent_memory_test_diagnostic.v1",
                        "platform": "android",
                        "device": device,
                        "exported_files": exported,
                        "skipped_files": skipped,
                        "excludes": ["api_key", "speaker_embeddings", "raw_pcm"],
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
            )
            exported.append("diagnostics.json")
        os.replace(temporary_zip, destination)
        return json.dumps(
            {
                "schema": "agent_memory_test_diagnostic.v1",
                "files": exported,
                "skipped_files": skipped,
                "size_bytes": destination.stat().st_size,
            },
            ensure_ascii=False,
        )
    finally:
        temporary_zip.unlink(missing_ok=True)


def _runtime_llm_config() -> Dict[str, str]:
    return dict(_require_runtime().llm_config)


class _Handler(BaseHTTPRequestHandler):
    local_auth_token: str = ""
    static_dir: str = ""
    platform: str = "android"
    owner_id: str = ""

    def log_message(self, format: str, *args: Any) -> None:
        _log(f"HTTP {self.command} {self.path} -> {format % args}")

    def parse_request(self) -> bool:
        if not super().parse_request():
            return False
        if self._is_authorized():
            return True
        self.send_error(HTTPStatus.UNAUTHORIZED, "local runtime authorization required")
        return False

    def _is_authorized(self) -> bool:
        expected = str(type(self).local_auth_token or "")
        if not expected:
            return True
        header_token = str(self.headers.get("X-AI-Glasses-Local-Token") or "")
        if header_token and hmac.compare_digest(header_token, expected):
            return True
        cookie_header = str(self.headers.get("Cookie") or "")
        try:
            cookie = SimpleCookie(cookie_header)
        except CookieError:
            return False
        cookie_token = ""
        if "ai_glasses_local_token" in cookie:
            morsel = cookie["ai_glasses_local_token"]
            cookie_token = morsel.value if morsel is not None else ""
        return bool(cookie_token and hmac.compare_digest(cookie_token, expected))

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self._send_static("index.html")
            return
        if parsed.path.startswith("/static/"):
            relative = parsed.path[len("/static/"):]
            self._send_static(relative)
            return
        if parsed.path == "/health":
            self._send_json({"ok": True})
            return
        if parsed.path == "/api/runtime":
            self._send_json(self._runtime_info())
            return
        if parsed.path == "/api/audio/capabilities":
            self._send_json(self._audio_capabilities())
            return
        if parsed.path == "/api/speaker/profile":
            query = urllib.parse.parse_qs(parsed.query)
            user_id = query.get("user_id", [type(self).owner_id])[0]
            self._send_json(self._current().speakers.profile(user_id))
            return
        if parsed.path == "/api/agent-memory/stats":
            self._send_json(self._current().admin.stats())
            return
        if parsed.path == "/api/agent-memory/facts":
            self._send_list(self._current().admin.list_facts)
            return
        if parsed.path == "/api/agent-memory/states":
            self._send_list(self._current().admin.list_states)
            return
        if parsed.path == "/api/agent-memory/actionables":
            self._send_list(self._current().admin.list_actionables)
            return
        if parsed.path == "/api/agent-memory/episodes":
            self._send_list(self._current().admin.list_episodes)
            return
        self._send_json({"detail": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/chat":
            self._handle_chat()
            return
        if parsed.path == "/api/agent-memory/reflect":
            runtime = self._current()
            report = runtime.runtime.trigger_reflect()
            self._send_json({"reflect": report, "stats": runtime.admin.stats()})
            return
        if parsed.path == "/api/agent-memory/import":
            self._handle_import()
            return
        self._send_json({"detail": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        prefix = "/api/agent-memory/facts/"
        if parsed.path.startswith(prefix):
            fact_id = parsed.path[len(prefix):].strip("/")
            try:
                result = self._current().admin.delete_fact(int(fact_id))
            except (TypeError, ValueError):
                self._send_json({"detail": "invalid fact id"}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result)
            return
        self._send_json({"detail": "not found"}, status=HTTPStatus.NOT_FOUND)

    def _handle_chat(self) -> None:
        try:
            body = self._read_json()
        except ValueError as exc:
            self._send_json({"detail": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        runtime = self._current()
        message = str(body.get("message") or "").strip()
        if not message:
            self._send_json({"detail": "message is required"}, status=HTTPStatus.BAD_REQUEST)
            return
        try:
            recall = runtime.runtime.recall(query=message, top_k=8, budget="mid")
            memory_context = str(recall.get("memory_context") or "")
            reply = chat_with_memory(
                user_message=message,
                memory_context=memory_context,
                llm_config=_runtime_llm_config(),
            )
            runtime.runtime.store_interaction(user_message=message, assistant_reply=reply)
        except ChatServiceError as exc:
            reply = f"回复生成失败：{exc}"
            memory_context = ""
        except Exception as exc:
            _log(f"/api/chat failed: {exc}")
            self._send_json({"detail": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_json(
            {
                "reply": reply,
                "debug": {
                    "engine": "agent_memory",
                    "recall_context": memory_context,
                    "recall_path": recall.get("actual_recall_path") if isinstance(recall, dict) else None,
                    "recall_elapsed_ms": recall.get("elapsed_ms") if isinstance(recall, dict) else None,
                    "recall_status": recall.get("status") if isinstance(recall, dict) else None,
                },
            }
        )

    def _handle_import(self) -> None:
        try:
            body = self._read_json()
        except ValueError as exc:
            self._send_json({"detail": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        runtime = self._current()
        text = str(body.get("text") or "").strip()
        if not text:
            self._send_json({"detail": "text is required"}, status=HTTPStatus.BAD_REQUEST)
            return
        result = runtime.runtime.ingest_ambient_segment(
            speaker="user",
            text=text,
            source_type="manual_import",
        )
        self._send_json({"stored": True, "result": result, "stats": runtime.admin.stats()})

    def _runtime_info(self) -> Dict[str, Any]:
        with _lock:
            device = dict(_device_state)
        return {
            "routing_mode": "memory_first",
            "platform": type(self).platform,
            "audio_input_owner": "native",
            "network_state": "online" if device.get("network_online", True) else "offline",
            "owner_id": type(self).owner_id,
            "engine": "agent_memory",
            "device": device,
        }

    def _audio_capabilities(self) -> Dict[str, Any]:
        return {
            "audio_input_ready": True,
            "ambient_transcription_ready": True,
            "assistant_query_ready": True,
            "speaker_enrollment_ready": True,
            "engine": "agent_memory",
            "components": {},
        }

    def _send_list(self, method: Any) -> None:
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        q = query.get("q", [""])[0]
        try:
            limit = int(query.get("limit", ["100"])[0])
        except (TypeError, ValueError):
            limit = 100
        self._send_json({"items": method(q=q, limit=limit)})

    def _current(self) -> _BackendRuntime:
        return _require_runtime()

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("request body must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _send_static(self, relative: str) -> None:
        root = Path(type(self).static_dir).resolve()
        target = (root / relative).resolve()
        if not str(target).startswith(str(root) + os.sep) or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    """本地命令行启动（调试用）：python -m backend.server <config.json>"""

    import sys

    config_json = sys.argv[1] if len(sys.argv) > 1 else "{}"
    print(start(config_json))
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print(stop())


if __name__ == "__main__":
    main()
