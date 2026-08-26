#!/usr/bin/env python3
"""Run the agent-memory MCP server over stdio.

Example client configuration::

    {
      "mcpServers": {
        "agent-memory": {
          "command": "python",
          "args": [
            "/Users/zhouboyu/Documents/agent_memory/scripts/memory_mcp_server.py"
          ]
        }
      }
    }

The server writes MCP frames to stdout. Logs are sent to stderr or to the path
configured in ``config.yaml``, so they never corrupt the protocol stream.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from memory.config import split_memory_config
from memory.memory_runtime import MemoryRuntime
from mcp.mcp_server import MemoryMCPService, StdioMCPServer
from voice.voice_runtime import VoiceRuntime


_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


def _expand_env_refs(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env_refs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_refs(item) for item in value]
    if isinstance(value, str):
        return _ENV_REF_RE.sub(lambda match: os.getenv(match.group(1), ""), value)
    return value


def load_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load config.yaml") from exc
    loaded = yaml.safe_load(path.expanduser().resolve().read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("config.yaml must contain a mapping at the top level")
    return _expand_env_refs(loaded)


def build_logger(log_path: Path | None, level: str) -> logging.Logger:
    logger = logging.getLogger("memory.mcp")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(log_path, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    return logger


def _resolve_config_path(value: Any) -> Path:
    """Resolve the config path, including compatibility with local helpers."""
    if isinstance(value, (str, Path)):
        return Path(value).expanduser().resolve()
    legacy_config = getattr(value, "config", None)
    if legacy_config is not None:
        return Path(legacy_config).expanduser().resolve()
    return DEFAULT_CONFIG_PATH


def _resolve_configured_path(value: Any, config_path: Path) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _explicit_override(config_source: Any, key: str) -> Any:
    if isinstance(config_source, (str, Path)) or config_source is None:
        return None
    value = getattr(config_source, key, None)
    if value is None or not str(value).strip():
        return None
    return value


def build_service(config_source: Any = None) -> tuple[MemoryMCPService, logging.Logger]:
    config_path = _resolve_config_path(config_source)
    config = load_config(config_path)
    server_config = config.get("memory_mcp_server") or {}
    if not isinstance(server_config, dict):
        raise ValueError("memory_mcp_server in config.yaml must be a mapping")
    db_path = _resolve_configured_path(server_config.get("db_path"), config_path)
    db_override = _explicit_override(config_source, "db_path")
    if db_override is not None:
        db_path = _resolve_configured_path(db_override, config_path)
    if db_path is None:
        raise ValueError("memory_mcp_server.db_path must be configured")
    log_path = _resolve_configured_path(server_config.get("log_path"), config_path)
    log_override = _explicit_override(config_source, "log_path")
    if log_override is not None:
        log_path = _resolve_configured_path(log_override, config_path)
    asr_result_dir = _resolve_configured_path(
        server_config.get("asr_result_dir"),
        config_path,
    )
    log_level = str(
        _explicit_override(config_source, "log_level")
        or server_config.get("log_level")
        or "INFO"
    )
    queue_timeout = float(
        _explicit_override(config_source, "queue_timeout")
        if _explicit_override(config_source, "queue_timeout") is not None
        else server_config.get("queue_timeout", 30.0)
    )
    memory_runtime_config, memory_manager_config, voice_runtime_config = split_memory_config(
        config,
        include_voice_runtime=True,
    )
    logger = build_logger(log_path, log_level)
    logger.info(
        "Memory MCP service configured config_path=%s cwd=%s db_path=%s log_path=%s asr_result_dir=%s",
        config_path,
        Path.cwd(),
        db_path,
        log_path,
        asr_result_dir,
    )
    try:
        memory_runtime = MemoryRuntime(
            db_path=db_path,
            memory_runtime_config=memory_runtime_config,
            memory_manager_config=memory_manager_config,
            logger=logger,
        )
        voice_logger = logger.getChild("voice")

        def build_voice_runtime() -> VoiceRuntime:
            """Load VAD/ASR/speaker models on the first audio request."""
            logger.info("Loading voice runtime models on first audio request")
            runtime = VoiceRuntime(
                voice_runtime_config,
                logger=voice_logger,
            )
            logger.info("Voice runtime models loaded")
            return runtime

        return MemoryMCPService(
            memory_runtime,
            voice_runtime_factory=build_voice_runtime,
            queue_timeout=queue_timeout,
            asr_result_dir=asr_result_dir,
            logger=logger,
        ), logger
    except Exception:
        if "memory_runtime" in locals():
            memory_runtime.close()
        raise


def main() -> int:
    # Keep stdout exclusively for MCP JSON-RPC. Some ASR dependencies emit
    # progress/version text to stdout while loading or processing models.
    protocol_output = sys.stdout
    sys.stdout = sys.stderr
    service, logger = build_service()
    config = load_config(DEFAULT_CONFIG_PATH)
    server_config = config.get("memory_mcp_server") or {}
    db_path = _resolve_configured_path(server_config.get("db_path"), DEFAULT_CONFIG_PATH)
    logger.info("Memory MCP server started db_path=%s", db_path)
    StdioMCPServer(
        service,
        output_stream=protocol_output,
        logger=logger,
    ).serve_forever()
    logger.info("Memory MCP server stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
