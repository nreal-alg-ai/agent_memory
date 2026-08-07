"""基于 OpenAI 兼容 API 的简单聊天封装，仅使用标准库。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


SYSTEM_PROMPT = (
    "你是 Agent Memory 测试 App 的助手。你会看到从个人记忆系统召回的上下文，"
    "请优先依据记忆回答；记忆不足时明确说明。回答简洁、口语化，不要编造记忆中没有的事实。"
)


class ChatServiceError(RuntimeError):
    """LLM 聊天调用失败。"""


def chat_with_memory(
    user_message: str,
    memory_context: str,
    llm_config: Dict[str, Any],
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """拼接 system + 记忆上下文 + 历史 + 用户消息，调用 OpenAI 兼容接口。"""
    base_url = str(llm_config.get("base_url") or "").strip().rstrip("/")
    api_key = str(llm_config.get("api_key") or "").strip()
    model = str(llm_config.get("model") or "").strip()
    if not base_url or not api_key or not model:
        return "助手尚未配置模型，请在设置页填写 provider / model / base_url / api_key。"

    url = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
    system_content = SYSTEM_PROMPT
    if memory_context.strip():
        system_content += f"\n\n【记忆上下文】\n{memory_context.strip()}"

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_content}]
    for turn in chat_history or []:
        role = str(turn.get("role") or "")
        content = str(turn.get("content") or "")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})

    payload = {"model": model, "messages": messages, "temperature": 0.4, "stream": False}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise ChatServiceError(f"LLM HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ChatServiceError(f"LLM 网络错误: {exc}") from exc

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ChatServiceError(f"LLM 响应格式异常: {body}") from exc
    return str(content or "").strip()
