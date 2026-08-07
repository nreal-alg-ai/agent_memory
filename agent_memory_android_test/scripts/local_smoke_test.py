"""本地冒烟测试：ingest -> recall -> chat -> reflect -> list -> delete -> HTTP API。

运行方式（项目根目录）：
    PYTHONDONTWRITEBYTECODE=1 conda run -n hermes python scripts/local_smoke_test.py

使用无效的本地 base_url 让 LLM 快速失败，链路其余部分（存储/召回/反思/管理）全部走真实代码。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_MEMORY_SRC = PROJECT_ROOT.parent / "src"
for import_root in (str(PROJECT_ROOT), str(AGENT_MEMORY_SRC)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)


def _http_json(url: str, token: str, method: str = "GET", payload: dict | None = None) -> dict:
    data = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"X-AI-Glasses-Local-Token": token, "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {url}: {body[:300]}") from exc


def _http_text(url: str, token: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"X-AI-Glasses-Local-Token": token},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def main() -> int:
    import backend.server as server

    data_root = Path(tempfile.mkdtemp(prefix="agent_memory_test_"))
    app_home = data_root / "app_home"
    static_dir = PROJECT_ROOT / "static"
    app_home.mkdir(parents=True, exist_ok=True)

    config = {
        "app_home": str(app_home),
        "static_dir": str(static_dir),
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "base_url": "https://127.0.0.1:1",
        "api_key": "smoke-test-invalid-key",
        "owner_id": "smoke-user",
    }
    started = json.loads(server.start(json.dumps(config, ensure_ascii=False)))
    base_url = started["base_url"]
    token = started["local_token"]
    assert started["running"] is True, started
    print(f"[ok] start -> {base_url}")

    # 0) 静态资源与健康检查
    index_html = _http_text(f"{base_url}/", token)
    assert "Agent Memory 测试" in index_html, "index.html 未按定制前端返回"
    app_js = _http_text(f"{base_url}/static/app.js", token)
    assert "agent_memory" in app_js, "app.js 未按定制前端返回"
    health = _http_json(f"{base_url}/health", token)
    assert health.get("ok") is True, health
    print("[ok] static index/app.js + /health")

    # 1) ambient 事件（final、含说话人）应被存储
    ambient = server.ingest_audio_event(
        "smoke-user",
        "",
        json.dumps(
            {
                "schema_version": "audio_event.v1",
                "event_id": "ambient-1",
                "type": "transcript_final",
                "lane": "ambient",
                "source_type": "ambient_audio",
                "text": "我今天在虹桥机场喝了冰美式，然后坐高铁去北京。",
                "final": True,
                "speaker": {"state": "matched", "label": "user"},
                "start_ms": 1000,
                "end_ms": 5000,
            },
            ensure_ascii=False,
        ),
        "{}",
        "{}",
    )
    ambient_result = json.loads(ambient)
    assert ambient_result["status"] == "completed" and ambient_result["stored"] is True, ambient_result
    print("[ok] ambient ingest stored")

    # 2) 敏感事件应被过滤
    sensitive = json.loads(
        server.ingest_audio_event(
            "smoke-user",
            "",
            json.dumps(
                {
                    "event_id": "ambient-sensitive",
                    "type": "transcript_final",
                    "lane": "ambient",
                    "text": "验证码是 123456，请帮我记一下。",
                    "final": True,
                },
                ensure_ascii=False,
            ),
            "{}",
            "{}",
        )
    )
    assert sensitive["stored"] is False and sensitive["reason"] == "filtered", sensitive
    print("[ok] sensitive event filtered")

    # 3) 唤醒问句：应返回 pending，随后轮询 queue_status 到 completed
    query_event_id = "assistant-1"
    queued = json.loads(
        server.ingest_audio_event(
            "smoke-user",
            "",
            json.dumps(
                {
                    "event_id": query_event_id,
                    "type": "transcript_final",
                    "lane": "assistant",
                    "source_type": "wake_query",
                    "text": "我早上喝了什么？",
                    "final": True,
                    "speaker": {"state": "matched", "label": "user"},
                },
                ensure_ascii=False,
            ),
            "{}",
            "{}",
        )
    )
    assert queued["status"] == "pending", queued
    print("[ok] assistant query queued as pending")

    completed = None
    for _ in range(60):
        queue = json.loads(server.queue_status("smoke-user"))
        for event in queue["events"]:
            if event["event_id"] == query_event_id:
                if event["status"] in {"completed", "failed"}:
                    completed = event
                break
        if completed:
            break
        time.sleep(0.5)
    assert completed is not None, "assistant query never completed"
    assert completed["status"] in {"completed", "failed"}, completed
    print(f"[ok] assistant query finished status={completed['status']}")

    # 4) HTTP: /api/chat（带 token）
    chat = _http_json(
        f"{base_url}/api/chat",
        token,
        method="POST",
        payload={"message": "我今天早上喝了什么？", "user_id": "smoke-user"},
    )
    assert "reply" in chat and "debug" in chat, chat
    print(f"[ok] /api/chat reply_len={len(chat['reply'])} recall_chars={len(chat['debug'].get('recall_context') or '')}")

    # 5) 统计与列表
    stats = _http_json(f"{base_url}/api/agent-memory/stats", token)
    assert set(stats["counts"]) == {"facts", "states", "actionables", "episodes"}, stats
    episodes = _http_json(f"{base_url}/api/agent-memory/episodes?limit=20", token)
    assert isinstance(episodes["items"], list) and len(episodes["items"]) >= 1, episodes
    facts = _http_json(f"{base_url}/api/agent-memory/facts?limit=50", token)
    print(f"[ok] stats={stats['counts']} episodes={len(episodes['items'])} facts={len(facts['items'])}")

    # 6) 反思
    reflect = _http_json(f"{base_url}/api/agent-memory/reflect", token, method="POST", payload={"user_id": "smoke-user"})
    assert "reflect" in reflect, reflect
    print(f"[ok] reflect report keys={sorted(reflect['reflect'].keys())[:8]}")

    # 7) 删除一条事实（若存在）
    if facts["items"]:
        fact_id = int(facts["items"][0]["id"])
        deleted = _http_json(f"{base_url}/api/agent-memory/facts/{fact_id}", token, method="DELETE")
        assert deleted["deleted"] is True, deleted
        after = _http_json(f"{base_url}/api/agent-memory/facts?limit=50", token)
        assert all(int(item["id"]) != fact_id for item in after["items"]), "fact still listed after delete"
        print(f"[ok] deleted fact id={fact_id} removed_index_rows={deleted.get('removed_index_rows')}")
    else:
        print("[warn] no facts to delete (LLM 未配置，事实提取跳过，属预期)")

    # 8) 手动导入
    imported = _http_json(
        f"{base_url}/api/agent-memory/import",
        token,
        method="POST",
        payload={"text": "下周三是 Alex 的生日，记得准备礼物。", "user_id": "smoke-user"},
    )
    assert imported["stored"] is True, imported
    print("[ok] manual import stored")

    # 9) 声纹：样本入库 + 分类
    enrollment = json.loads(
        server.ingest_audio_event(
            "smoke-user",
            "",
            json.dumps(
                {
                    "event_id": "enroll-1",
                    "type": "speaker_update",
                    "lane": "enrollment",
                    "source_type": "speaker_enrollment",
                    "final": True,
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "speaker_embedding": [0.1] * 64,
                    "speaker_embedding_model": "sherpa_campplus",
                    "enrollment_session_id": "session-1",
                    "sample_index": 1,
                    "sample_total": 1,
                },
                ensure_ascii=False,
            ),
            "{}",
        )
    )
    assert enrollment["dispatch"]["result"]["enrollment"]["enrolled"] is True, enrollment
    enrollment_event = json.loads(server.wait_audio_event("smoke-user", "enroll-1", 2.0))
    assert enrollment_event.get("status") == "completed", enrollment_event
    assert enrollment_event["dispatch"]["result"]["enrollment"]["enrolled"] is True, enrollment_event
    classified = json.loads(
        server.classify_speaker("smoke-user", json.dumps([0.12] * 64), "sherpa_campplus")
    )
    assert classified["state"] == "matched", classified
    profile = json.loads(server.speaker_profile("smoke-user"))
    assert profile["enrolled"] is True and profile["sample_count"] == 1, profile
    print(f"[ok] speaker enrolled + classified {classified['state']} sim={classified['similarity']}")

    server.stop()
    print("[ok] stop")
    print(f"\nALL SMOKE TESTS PASSED (data_root={data_root})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
