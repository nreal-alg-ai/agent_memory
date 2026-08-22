#!/usr/bin/env python3
"""Run the agent-memory MCP server over stdio.

Example client configuration::

    {
      "mcpServers": {
        "agent-memory": {
          "command": "python",
          "args": [
            "/Users/zhouboyu/Documents/agent_memory/scripts/memory_mcp_server.py",
            "--db-path",
            "/Users/zhouboyu/Documents/agent_memory/tmp/mcp/memory.db"
          ]
        }
      }
    }

The server writes MCP frames to stdout. Logs are sent to stderr (or to the
path supplied by ``--log-path``), so they never corrupt the protocol stream.
"""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agent-memory as an MCP stdio server.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("memory.db"),
        help="SQLite memory database path. Defaults to ./memory.db.",
    )
    parser.add_argument("--log-path", type=Path, help="Optional memory log file path.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--queue-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for queued writes during graceful shutdown.",
    )
    return parser.parse_args()


def build_service(args: argparse.Namespace) -> tuple[MemoryMCPService, logging.Logger]:
    config = load_config(args.config)
    memory_runtime_config, memory_manager_config, voice_runtime_config = split_memory_config(
        config,
        include_voice_runtime=True,
    )
    logger = build_logger(args.log_path, args.log_level)
    try:
        memory_runtime = MemoryRuntime(
            db_path=args.db_path,
            memory_runtime_config=memory_runtime_config,
            memory_manager_config=memory_manager_config,
            logger=logger,
        )
        voice_logger = logger.getChild("voice")
        voice_runtime = VoiceRuntime(
            voice_runtime_config,
            logger=voice_logger,
        )
        return MemoryMCPService(
            memory_runtime,
            voice_runtime,
            queue_timeout=args.queue_timeout,
        ), logger
    except Exception:
        if "memory_runtime" in locals():
            memory_runtime.close()
        raise


def main() -> int:
    args = parse_args()
    service, logger = build_service(args)
    logger.info("Memory MCP server started db_path=%s", args.db_path.expanduser().resolve())
    StdioMCPServer(service, logger=logger).serve_forever()
    logger.info("Memory MCP server stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
