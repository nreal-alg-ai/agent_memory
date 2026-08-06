#!/usr/bin/env python3
"""Small embedding client used by the unified memory prototype.

The production voice_recording project can call remote embedding APIs. For this
standalone memory prototype we keep the same class name but add a deterministic
local fallback so benchmark memory construction can run even when the embedding
provider is unavailable.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "provider": "local_hash",
    "model": "local-hash-embedding",
    "base_url": "",
    "api_key": "",
    "api_key_env": "",
    "dimensions": 384,
    "normalize": True,
    "timeout": 30,
}


def _load_project_config() -> Dict[str, Any]:
    try:
        import yaml

        config_path = Path(__file__).resolve().parents[2] / "config.yaml"
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        manager_config = loaded.get("memory_manager")
        if isinstance(manager_config, dict) and isinstance(manager_config.get("embedding"), dict):
            return dict(manager_config["embedding"])
    except Exception:
        pass
    return {}


def _resolve_env(value: Any) -> str:
    text = str(value or "").strip()
    match = re.fullmatch(r"\${([A-Za-z_][A-Za-z0-9_]*)}", text)
    if match:
        return os.environ.get(match.group(1), "").strip()
    return text


def _build_openai_embedding_url(base_url: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return "https://api.openai.com/v1/embeddings"
    if base.endswith("/embeddings"):
        return base
    if base.endswith("/v1") or re.search(r"/api/paas/v\d+$", base):
        return f"{base}/embeddings"
    return f"{base}/v1/embeddings"


class EmbeddingClient:
    """OpenAI-compatible embedding client with deterministic local fallback."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        merged = dict(DEFAULT_CONFIG)
        merged.update(_load_project_config())
        if config:
            merged.update(config)
        self._config = merged
        self._provider = str(merged.get("provider") or "local_hash").strip().lower()
        self._model = str(merged.get("model") or "local-hash-embedding")
        self._base_url = str(merged.get("base_url") or "")
        self._api_key = _resolve_env(merged.get("api_key"))
        api_key_env = str(merged.get("api_key_env") or "").strip()
        if not self._api_key and api_key_env:
            self._api_key = os.environ.get(api_key_env, "").strip()
        self._dim = max(8, int(merged.get("dimensions") or 384))
        self._normalize = bool(merged.get("normalize", True))
        self._timeout = int(merged.get("timeout") or 30)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        if self._provider == "openai" and self._api_key:
            remote = self._embed_openai(text)
            if remote is not None:
                return remote
        return self._embed_local_hash(text)

    def embed_batch(self, texts: Iterable[str]) -> List[np.ndarray]:
        return [self.embed_text(text) for text in texts]

    def _embed_openai(self, text: str) -> Optional[np.ndarray]:
        url = _build_openai_embedding_url(self._base_url)
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {"model": self._model, "input": text or " "}
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            vector = data.get("data", [{}])[0].get("embedding")
            if not vector:
                return None
            return self._as_vector(vector)
        except Exception as exc:
            logger.warning("Remote embedding failed; falling back to local hash: %s", exc)
            return None

    def _embed_local_hash(self, text: str) -> np.ndarray:
        vector = np.zeros((self._dim,), dtype=np.float32)
        terms = re.findall(r"[A-Za-z0-9_.$'-]+|[\u4e00-\u9fff]+", str(text or "").lower())
        if not terms:
            terms = ["empty"]
        for term in terms:
            digest = hashlib.blake2b(term.encode("utf-8"), digest_size=16).digest()
            idx = int.from_bytes(digest[:4], "little") % self._dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + min(2.0, math.log1p(len(term)) / 2.0)
            vector[idx] += sign * weight
        return self._as_vector(vector)

    def _as_vector(self, raw: Any) -> np.ndarray:
        vector = np.asarray(raw, dtype=np.float32).reshape(-1)
        if vector.size != self._dim:
            resized = np.zeros((self._dim,), dtype=np.float32)
            keep = min(self._dim, vector.size)
            resized[:keep] = vector[:keep]
            vector = resized
        if self._normalize:
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
        return vector.reshape(1, -1).astype(np.float32)
