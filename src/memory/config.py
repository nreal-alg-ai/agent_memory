"""Configuration helpers for the memory manager and runtime."""

from __future__ import annotations

from typing import Any, Dict, Tuple


def split_memory_config(
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split project config into runtime and unified manager mappings.

    The project config keeps memory settings in two sections:
    ``memory_manager`` owns LLM, embedding, storage, and recall settings, while
    ``memory_runtime`` owns frontend batching settings. The returned manager
    mapping retains its nested ``llm`` and ``embedding`` mappings.
    """
    manager_section = config.get("memory_manager")
    runtime_section = config.get("memory_runtime")
    memory_manager_config = (
        dict(manager_section) if isinstance(manager_section, dict) else {}
    )
    memory_runtime_config = (
        dict(runtime_section) if isinstance(runtime_section, dict) else {}
    )

    for nested_key in ("llm", "embedding"):
        nested_value = memory_manager_config.get(nested_key)
        memory_manager_config[nested_key] = (
            dict(nested_value) if isinstance(nested_value, dict) else {}
        )

    return memory_runtime_config, memory_manager_config
