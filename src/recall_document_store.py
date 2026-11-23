"""Neutral recall-document storage shared by memory and user-model databases."""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Optional, Sequence

import numpy as np

from storage_utils import json_dumps, json_loads, local_now_text, normalize_reference_timestamp

try:
    import jieba
except ImportError:  # pragma: no cover
    jieba = None

_LEXICAL_DATE_PATTERNS = (
    re.compile(r"(?<!\d)\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}(?:日)?(?!\d)"),
    re.compile(r"(?<!\d)\d{1,2}月\d{1,2}日?(?!\d)"),
)


def _lexical_index_text(identity_text: Any) -> str:
    """Convert display identity text into deterministic FTS search tokens.

    Recall documents keep the original ``identity_text`` for embeddings and
    display. FTS receives a separate token stream so Chinese text can be
    searched by words instead of relying on SQLite's default tokenizer.
    Dates are excluded here because fact time is filtered through structured
    time columns rather than lexical coincidence.
    """
    text_lines: list[str] = []
    for line in str(identity_text or "").splitlines():
        clean_line = line.strip()
        colon_positions = [
            position
            for position in (clean_line.find(":"), clean_line.find("："))
            if position >= 0
        ]
        if colon_positions:
            # identity_text is formatted as one ``field: value`` per line.
            # Index only the value so labels such as ``summary`` and
            # ``entities`` do not become searchable memory content.
            clean_line = clean_line[min(colon_positions) + 1 :].strip()
        if clean_line:
            text_lines.append(clean_line)
    text = "\n".join(text_lines)
    for pattern in _LEXICAL_DATE_PATTERNS:
        text = pattern.sub(" ", text)
    if not text.strip():
        return ""

    if jieba is not None:
        # Search mode keeps useful sub-tokens for Chinese compounds while the
        # regular cut preserves the complete domain phrase when available.
        raw_tokens = [
            *jieba.lcut(text, HMM=False),
            *jieba.cut_for_search(text, HMM=False),
        ]
    else:
        # Keep the database usable in minimal environments. This fallback is
        # deliberately simple; production installs should include jieba.
        raw_tokens = []
        for match in re.findall(
            r"[A-Za-z][A-Za-z0-9_.$'-]*|\d+(?:/\d+)?|[\u4e00-\u9fff]+",
            text,
        ):
            if re.fullmatch(r"[\u4e00-\u9fff]+", match):
                raw_tokens.append(match)
                raw_tokens.extend(match)
                raw_tokens.extend(
                    match[index : index + 2]
                    for index in range(len(match) - 1)
                )
            else:
                raw_tokens.append(match)

    tokens: list[str] = []
    seen: set[str] = set()
    for raw_token in raw_tokens:
        token = re.sub(r"\s+", "", str(raw_token or "")).strip()
        if not token or re.fullmatch(r"[^\w\u4e00-\u9fff]+", token):
            continue
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return " ".join(tokens)


def _embedding_to_blob(embedding: Optional[np.ndarray]) -> Optional[bytes]:
    if embedding is None:
        return None
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    return vector.tobytes()


def _blob_to_embedding(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        vector = np.frombuffer(bytes(value), dtype=np.float32)
    except Exception:
        return None
    return vector.reshape(1, -1).astype(np.float32)


class RecallDocumentStore:
    """SQLite mixin; subclasses own source rows, transactions and type filters."""

    RECALL_DOCUMENT_TABLE = "recall_documents"
    EMBEDDING_PROFILE_TABLE = "embedding_profile"
    EXTRA_EMBEDDING_COLUMNS: tuple[tuple[str, str], ...] = ()

    def _init_recall_document_schema(self) -> None:
        table = self.RECALL_DOCUMENT_TABLE
        self._conn.execute(f"""CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            object_type TEXT NOT NULL, object_id INTEGER NOT NULL,
            title TEXT NOT NULL DEFAULT '', summary TEXT NOT NULL DEFAULT '',
            identity_text TEXT NOT NULL DEFAULT '', identity_text_embedding BLOB,
            entity_ids TEXT NOT NULL DEFAULT '[]', topic_keys TEXT NOT NULL DEFAULT '[]',
            time_start TEXT NOT NULL DEFAULT '', time_end TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT '', confidence REAL NOT NULL DEFAULT 0.0,
            importance REAL NOT NULL DEFAULT 0.0, metadata TEXT NOT NULL DEFAULT '{{}}',
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            UNIQUE(object_type, object_id)
        )""")
        self._conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_pending_embeddings ON {table}(id) "
                           "WHERE identity_text_embedding IS NULL AND trim(identity_text)!=''")
        self._conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_type_status_time "
                           f"ON {table}(object_type,status,time_end,updated_at)")
        fts = f"{table}_identity_fts"
        self._conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {fts} USING fts5(lexical_index_text)")
        self._conn.execute(f"""CREATE TRIGGER IF NOT EXISTS delete_{table}_fts
            AFTER DELETE ON {table} BEGIN DELETE FROM {fts} WHERE rowid=OLD.id; END""")
        rows = self._conn.execute(f"SELECT id,identity_text FROM {table} "
                                  f"WHERE id NOT IN (SELECT rowid FROM {fts})").fetchall()
        self._conn.executemany(f"INSERT INTO {fts}(rowid,lexical_index_text) VALUES (?,?)",
                               [(row["id"], _lexical_index_text(row["identity_text"])) for row in rows])
        self._commit_if_needed()

    def upsert_recall_document(
        self, *, object_type: str, object_id: int, identity_text: str,
        identity_text_embedding: Optional[np.ndarray] = None,
        title: str = "", summary: str = "", entity_ids: Optional[Sequence[int]] = None,
        topic_keys: Optional[Sequence[str]] = None, time_start: str = "", time_end: str = "",
        status: str = "", confidence: float = 0.0, importance: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None, reference_timestamp: Any = None,
    ) -> int:
        """Keep the old vector exactly when the current identity text is unchanged."""
        kind = str(object_type or "").strip().lower()
        if not kind or int(object_id or 0) <= 0:
            raise ValueError("object_type and a positive object_id are required")
        now = local_now_text() if reference_timestamp is None else normalize_reference_timestamp(reference_timestamp)
        values = {
            "object_type": kind, "object_id": int(object_id), "title": str(title or ""),
            "summary": str(summary or ""), "identity_text": str(identity_text or ""),
            "identity_text_embedding": _embedding_to_blob(identity_text_embedding),
            "entity_ids": json_dumps(list(dict.fromkeys(int(v) for v in entity_ids or [] if str(v).strip().isdigit() and int(v) > 0))),
            "topic_keys": json_dumps(list(dict.fromkeys(str(v).strip() for v in topic_keys or [] if str(v).strip()))),
            "time_start": str(time_start or ""), "time_end": str(time_end or ""), "status": str(status or ""),
            "confidence": float(confidence or 0), "importance": float(importance or 0),
            "metadata": json_dumps(metadata or {}), "created_at": now, "updated_at": now,
        }
        table = self.RECALL_DOCUMENT_TABLE
        updates = [f"{key}=excluded.{key}" for key in values
                   if key not in {"object_type", "object_id", "created_at", "identity_text_embedding"}]
        updates.append(f"identity_text_embedding=CASE WHEN {table}.identity_text=excluded.identity_text "
                       f"THEN COALESCE(excluded.identity_text_embedding,{table}.identity_text_embedding) "
                       "ELSE excluded.identity_text_embedding END")
        with self.transaction():
            self._conn.execute(f"INSERT INTO {table}({','.join(values)}) VALUES ({','.join('?' for _ in values)}) "
                               f"ON CONFLICT(object_type,object_id) DO UPDATE SET {','.join(updates)}", list(values.values()))
            document_id = int(self._conn.execute(f"SELECT id FROM {table} WHERE object_type=? AND object_id=?",
                                                 (kind, int(object_id))).fetchone()["id"])
            fts = f"{table}_identity_fts"
            self._conn.execute(f"DELETE FROM {fts} WHERE rowid=?", (document_id,))
            self._conn.execute(f"INSERT INTO {fts}(rowid,lexical_index_text) VALUES (?,?)",
                               (document_id, _lexical_index_text(identity_text)))
        return document_id

    def recall_documents_needing_embeddings(self, *, after_id: int = 0, limit: int = 32):
        return [dict(row) for row in self._conn.execute(
            f"SELECT id,identity_text FROM {self.RECALL_DOCUMENT_TABLE} "
            "WHERE id>? AND identity_text_embedding IS NULL AND trim(identity_text)!='' ORDER BY id LIMIT ?",
            (after_id, limit),
        ).fetchall()]

    def set_recall_document_embeddings(self, embeddings) -> int:
        updated = 0
        with self.transaction():
            for document_id, text, vector in embeddings:
                updated += self._conn.execute(
                    f"UPDATE {self.RECALL_DOCUMENT_TABLE} SET identity_text_embedding=? "
                    "WHERE id=? AND identity_text=? AND identity_text_embedding IS NULL",
                    (_embedding_to_blob(vector), int(document_id), text),
                ).rowcount
        return updated

    def ensure_embedding_profile(self, signature: str) -> bool:
        """Invalidate vectors in this database only, without changing business time."""
        profile = self.EMBEDDING_PROFILE_TABLE
        with self.transaction():
            self._conn.execute(f"CREATE TABLE IF NOT EXISTS {profile}(id INTEGER PRIMARY KEY CHECK(id=1),signature TEXT NOT NULL)")
            old = self._conn.execute(f"SELECT signature FROM {profile} WHERE id=1").fetchone()
            if old is not None and old["signature"] == signature:
                return False
            self._conn.execute(f"UPDATE {self.RECALL_DOCUMENT_TABLE} SET identity_text_embedding=NULL")
            for table, column in self.EXTRA_EMBEDDING_COLUMNS:
                self._conn.execute(f"UPDATE {table} SET {column}=NULL")
            self._conn.execute(f"INSERT INTO {profile}(id,signature) VALUES(1,?) "
                               "ON CONFLICT(id) DO UPDATE SET signature=excluded.signature", (signature,))
        return True

    def open_reader(self):
        reader = object.__new__(type(self))
        reader.db_path = self.db_path
        reader._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        reader._conn.row_factory = sqlite3.Row
        reader._transaction_depth = 0
        reader._conn.execute("PRAGMA query_only=ON")
        reader._conn.execute("PRAGMA foreign_keys=ON")
        reader._conn.execute("PRAGMA busy_timeout=30000")
        return reader

    @contextmanager
    def reader_transaction(self):
        reader = self.open_reader()
        try:
            reader._conn.execute("BEGIN")
            yield reader
        finally:
            try:
                reader._conn.rollback()
            finally:
                reader.close()

    def _register_recall_timestamp(self, reference_timestamp: Optional[str]) -> None:
        reference = normalize_reference_timestamp(reference_timestamp or local_now_text())
        zone = datetime.fromisoformat(reference).tzinfo

        def timestamp(value):
            if not value:
                return None
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                return (parsed if parsed.tzinfo else parsed.replace(tzinfo=zone)).timestamp()
            except (ValueError, TypeError, OverflowError):
                return None

        self._conn.create_function("recall_timestamp", 1, timestamp, deterministic=True)

    def _query_recall_documents(self, *, source_table, clauses, params, terms=(), limit=40, embeddings_only=False):
        """Use the same typed lexical/vector query implementation in both stores."""
        table = self.RECALL_DOCUMENT_TABLE
        joined = f"{table} AS document JOIN {source_table} AS source ON source.id=document.object_id"
        if embeddings_only:
            clauses = [*clauses, "document.identity_text_embedding IS NOT NULL"]
            sql = f"SELECT document.* FROM {joined} WHERE {' AND '.join(clauses)} ORDER BY document.id"
        elif terms:
            fts = f"{table}_identity_fts"
            match = self._terms_to_fts_query(terms)
            if not match:
                return []
            sql = (f"SELECT document.* FROM {joined} JOIN {fts} ON {fts}.rowid=document.id "
                   f"WHERE {' AND '.join(clauses)} AND {fts} MATCH ? "
                   f"ORDER BY bm25({fts}),document.updated_at DESC,document.id DESC LIMIT ?")
            params = [*params, match, max(1, int(limit))]
        else:
            sql = (f"SELECT document.* FROM {joined} WHERE {' AND '.join(clauses)} "
                   "ORDER BY document.updated_at DESC,document.id DESC LIMIT ?")
            params = [*params, max(1, int(limit))]
        rows = []
        for row in self._conn.execute(sql, params).fetchall():
            item = dict(row)
            item["identity_text_embedding"] = _blob_to_embedding(item["identity_text_embedding"])
            for key in ("entity_ids", "topic_keys", "metadata"):
                item[key] = json_loads(item[key], {} if key == "metadata" else [])
            rows.append(item)
        return rows

    @staticmethod
    def _terms_to_fts_query(terms: Sequence[str]) -> str:
        clean = list(dict.fromkeys(re.sub(r"\s+", " ", str(t or "").strip()).lower()
                                   for t in terms or () if str(t or "").strip()))[:12]
        return " OR ".join('"' + term.replace('"', '""') + '"' for term in clean)
