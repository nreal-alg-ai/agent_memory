"""agent_memory 记忆管理：列表/搜索/统计/删除（包装层能力，不改第三方代码）。"""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any, Dict, List, Optional


_SEARCHABLE = {
    "memory_facts": (
        "summary", "keywords", "entities", "entity_ids", "fact_root_topic",
        "fact_aspect_topic", "fact_kind", "fact_subject", "embedding_text", "metadata",
    ),
    "memory_states": (
        "canonical_name", "summary", "time_line", "entity_ids", "entity_key",
        "state_scope", "state_type", "identity_text", "metadata",
    ),
    "memory_actionable_items": (
        "canonical_name", "summary", "item_type", "owner", "status",
        "due_at", "entity_ids", "embedding_text", "metadata",
    ),
}


class AgentMemoryAdmin:
    """独立连接读写 agent_memory 的 SQLite，供 HTTP API 使用。"""

    def __init__(self, db_path: str, *, database_lock: Optional[Any] = None) -> None:
        self._db_path = db_path
        self._lock = database_lock or threading.RLock()
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            timeout=30.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            counts = {
                "facts": self._count("memory_facts"),
                "states": self._count("memory_states"),
                "actionables": self._count("memory_actionable_items"),
                "episodes": self._count("memory_episodes"),
            }
            last_reflect = self._conn.execute(
                "SELECT MAX(updated_at) AS last FROM memory_states"
            ).fetchone()["last"]
            return {
                "counts": counts,
                "last_reflect_at": last_reflect or "",
            }

    def list_facts(self, q: str = "", limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            return self._list_with_search("memory_facts", q=q, limit=limit)

    def list_states(self, q: str = "", limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            return self._list_with_search("memory_states", q=q, limit=limit)

    def list_actionables(self, q: str = "", limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            return self._list_with_search("memory_actionable_items", q=q, limit=limit)

    def list_episodes(self, q: str = "", limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            sql = "SELECT * FROM memory_episodes"
            params: List[Any] = []
            keyword = str(q or "").strip()
            if keyword:
                like = f"%{keyword}%"
                sql += (
                    " WHERE title LIKE ? OR summary LIKE ? OR participants LIKE ?"
                    " OR canonical_topics LIKE ? OR source_type LIKE ? OR metadata LIKE ?"
                )
                params = [like] * 6
            sql += " ORDER BY id DESC LIMIT ?"
            params.append(max(1, min(int(limit or 200), 500)))
            return self._rows(sql, params)

    def delete_fact(self, fact_id: int) -> Dict[str, Any]:
        """删除一条事实及关联检索索引/FTS 行；States/Actionables 为派生数据不动。"""
        fact_id = int(fact_id)
        with self._lock:
            with self._conn:
                index_ids = [
                    int(row["id"])
                    for row in self._conn.execute(
                        "SELECT id FROM memory_index_entries"
                        " WHERE target_table = 'memory_facts' AND target_id = ?",
                        (fact_id,),
                    ).fetchall()
                ]
                for index_id in index_ids:
                    try:
                        self._conn.execute(
                            "DELETE FROM memory_index_entries_fts WHERE rowid = ?",
                            (index_id,),
                        )
                    except sqlite3.Error:
                        # 无 FTS5 的嵌入式 SQLite（如 Chaquopy）没有该表，忽略即可
                        pass
                deleted_indexes = self._conn.execute(
                    "DELETE FROM memory_index_entries"
                    " WHERE target_table = 'memory_facts' AND target_id = ?",
                    (fact_id,),
                ).rowcount
                cursor = self._conn.execute("DELETE FROM memory_facts WHERE id = ?", (fact_id,))
            return {
                "deleted": cursor.rowcount > 0,
                "fact_id": fact_id,
                "removed_index_rows": int(deleted_indexes),
            }

    def _list_with_search(self, table: str, *, q: str, limit: int) -> List[Dict[str, Any]]:
        fields = _SEARCHABLE.get(table)
        if fields is None:
            raise ValueError(f"unsupported memory table: {table}")
        sql = f"SELECT * FROM {table}"
        params: List[Any] = []
        keyword = str(q or "").strip()
        if keyword:
            like = f"%{keyword}%"
            placeholders = " OR ".join(f'"{field}" LIKE ?' for field in fields)
            sql += f" WHERE ({placeholders})"
            params = [like] * len(fields)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, min(int(limit or 200), 500)))
        return self._rows(sql, params)

    def _rows(self, sql: str, params: List[Any]) -> List[Dict[str, Any]]:
        rows = self._conn.execute(sql, params).fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for key in list(item.keys()):
                if isinstance(item[key], (bytes, bytearray)):
                    # 向量 BLOB 无法 JSON 序列化，面板展示不需要，直接剔除
                    del item[key]
            for key in (
                "entities",
                "entity_ids",
                "canonical_topics",
                "participants",
                "metadata",
                "evidence_fact_ids",
                "time_line",
            ):
                if key in item and isinstance(item[key], str):
                    try:
                        item[key] = json.loads(item[key])
                    except (TypeError, json.JSONDecodeError):
                        pass
            result.append(item)
        return result

    def _count(self, table: str) -> int:
        row = self._conn.execute(f'SELECT COUNT(*) AS n FROM "{table}"').fetchone()
        return int(row["n"]) if row else 0
