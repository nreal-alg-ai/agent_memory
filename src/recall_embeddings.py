"""One lazy embedding batch pool spanning independently committed stores."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Sequence

from pwm_logging import as_pwm_logger


class RecallEmbeddingIndexer:
    def __init__(self, *, databases: Sequence[Any], client_provider: Callable,
                 enabled: bool = True, logger=None) -> None:
        self._databases = tuple(databases)
        self._client_provider = client_provider
        self._enabled = enabled
        self._logger = as_pwm_logger(logger or logging.getLogger(__name__))

    def initialize(self) -> bool:
        """Check each cache profile before the shared worker/readers start."""
        if not self._enabled:
            return False
        client = self._client_provider()
        for db in self._databases:
            changed = db.ensure_embedding_profile(client.model_signature)
            self._logger.info("Recall embedding profile database=%s changed=%s", db.db_path, changed)
        return any(db.recall_documents_needing_embeddings(limit=1) for db in self._databases)

    def generate(self, *, force: bool = False) -> dict:
        """Attempt each missing vector once per invocation, outside both write transactions."""
        if any(db._conn.in_transaction for db in self._databases):
            raise RuntimeError("Recall embeddings must be generated outside a write transaction")
        if not self._enabled:
            return {"status": "empty", "reason": "embedding_disabled", "rebuilt": 0}
        client = self._client_provider()
        batch_size = max(1, int(client.batch_size))
        cursors = {db: 0 for db in self._databases}
        requested = rebuilt = failed = 0
        started = time.monotonic()
        while True:
            batch = []
            for db in self._databases:
                rows = db.recall_documents_needing_embeddings(after_id=cursors[db], limit=batch_size - len(batch))
                batch.extend((db, row) for row in rows)
                if rows:
                    cursors[db] = rows[-1]["id"]
                if len(batch) == batch_size:
                    break
            if not batch or (not force and len(batch) < batch_size):
                break
            requested += len(batch)
            try:
                vectors = client.embed_batch([row["identity_text"] for _, row in batch])
                if len(vectors) != len(batch):
                    raise ValueError("Recall embedding batch result length mismatch")
            except Exception:
                self._logger.warning("Recall document batch embedding failed", exc_info=True)
                vectors = [None] * len(batch)
            failed += sum(vector is None for vector in vectors)
            by_database = {}
            for (db, row), vector in zip(batch, vectors):
                if vector is not None:
                    by_database.setdefault(db, []).append((row["id"], row["identity_text"], vector))
            for db, updates in by_database.items():
                try:
                    rebuilt += db.set_recall_document_embeddings(updates)
                except Exception:
                    failed += len(updates)
                    self._logger.warning("Recall vector writeback failed database=%s", db.db_path, exc_info=True)
        report = {
            "status": "partial" if failed else "ok" if requested else "empty",
            "force": force, "requested_count": requested, "rebuilt": rebuilt, "missing_count": failed,
            "has_pending": any(db.recall_documents_needing_embeddings(limit=1) for db in self._databases),
            "batch_size": batch_size, "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
        }
        self._logger.log_json_format("memory_recall_document_embeddings", "finish", report)
        return report
