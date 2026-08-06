"""Configuration helpers for the memory manager and runtime."""

from __future__ import annotations

from typing import Any, Dict, Tuple


def split_memory_config(
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Split project config into runtime, manager, LLM, and embedding mappings.

    The project config keeps memory settings in two sections:
    ``memory_manager`` owns LLM, embedding, storage, and recall settings, while
    ``memory_runtime`` owns frontend batching settings.  ``memory_manager``
    owns storage, reflection, and recall settings, with separate nested
    ``llm`` and ``embedding`` mappings.
    """
    manager_section = config.get("memory_manager")
    runtime_section = config.get("memory_runtime")
    memory_manager_config = (
        dict(manager_section) if isinstance(manager_section, dict) else {}
    )
    memory_runtime_config = (
        dict(runtime_section) if isinstance(runtime_section, dict) else {}
    )

    embedding = memory_manager_config.pop("embedding", None)
    embedding_config = dict(embedding) if isinstance(embedding, dict) else {}

    llm = memory_manager_config.pop("llm", None)
    llm_config = dict(llm) if isinstance(llm, dict) else {}

    return (
        memory_runtime_config,
        memory_manager_config,
        llm_config,
        embedding_config,
    )
