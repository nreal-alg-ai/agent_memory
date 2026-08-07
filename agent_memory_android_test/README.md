# Agent Memory 测试 App

独立 Android 测试 App：把 agent_memory（同事的记忆系统，本仓库 `src/`）作为唯一记忆引擎，
套用本系统 Android 外壳（音频采集 / VAD / KWS / ASR / 声纹 / WebView UI / TTS）。

本 App 代码位于 agent_memory 仓库（`/Users/huyaokai/Desktop/workspace/agent_memory`）内的
`agent_memory_android_test/` 子目录，开发分支为 `dev_ykhu`。

## 约束

- **不修改** `ai_glasses_memory_assistant/` 任何代码。
- **不修改** `ai_glasses_memory_assistant/` 任何代码。
- agent_memory 引擎源码就是本仓库的 `src/`，修改直接在 `dev_ykhu` 分支上进行。
- 两套系统是竞品关系：本 App 使用全新空 SQLite（`<app_home>/data/agent_memory/memory.db`），
  与原系统数据完全独立，不导入、不迁移。

## 目录

```
agent_memory_android_test/
├── android/      复制自本系统 android/，改包名 com.agentmemory.test + Python 模块引用
├── backend/      Python 薄后端（Kotlin 桥接 + HTTP API）
├── static/       定制前端（聊天 + agent_memory 四类记忆视图 + 召回调试）
├── scripts/      本地冒烟测试
└── ../src/       agent_memory 引擎源码（同一仓库，App 直接引用，无副本）
```

## agent_memory 来源与更新

- 据点仓库：`/Users/huyaokai/Desktop/workspace/agent_memory`（独立 git 仓库，`dev_ykhu` 分支，
  保留上游 `origin` remote）。App 与引擎在同一仓库：引擎是仓库根 `src/`，App 是子目录
  `agent_memory_android_test/`。
- 引擎引用：Chaquopy 打包 `../src/**/*.py`（`memory.*` 包），冒烟测试 PYTHONPATH 指向
  `PROJECT_ROOT.parent / "src"`——改引擎即改仓库 `src/`，无需同步副本。
- 原 `ai_glasses_memory_assistant/.external/agent_memory`（main）保持原样，属于原系统。

## 后端功能

- `backend/server.py`：与 `android_runtime.py` 同名同签名的 Kotlin 桥接函数
  （start / ingest_audio_event / queue_status / classify_speaker / stop 等），Android 外壳 Kotlin 零改动。
- HTTP API（WebView 前端调用）：
  - `POST /api/chat`：agent_memory recall -> LLM 回复 -> store_interaction，响应含 `debug.recall_context`
  - `GET /api/agent-memory/stats|facts|states|actionables|episodes`（q/limit）
  - `DELETE /api/agent-memory/facts/{id}`：删除事实及关联索引/FTS 行
  - `POST /api/agent-memory/reflect`、`POST /api/agent-memory/import`
  - 兼容端点：`/api/audio/capabilities`、`/api/speaker/profile`、`/api/runtime`、`/health`
- LLM/Embedding 配置由 Android 设置页动态注入；Embedding 失败自动降级本地 hash。

## 构建

```bash
cd agent_memory_android_test/android
JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
ANDROID_HOME=/opt/homebrew/share/android-commandlinetools \
./gradlew assembleDebug
```

APK：`android/app/build/outputs/apk/debug/app-debug.apk`

安装到真机（需先连接 USB 并信任）：

```bash
android/tools/build_and_install_debug.sh
```

## 本地验证（不依赖 Android）

```bash
cd agent_memory_android_test
PYTHONDONTWRITEBYTECODE=1 conda run -n hermes python scripts/local_smoke_test.py
```

覆盖：启动、静态资源、环境转写存储、敏感信息过滤、唤醒问句队列、
`/api/chat`、统计/列表、反思、事实删除、手动导入、声纹录入与分类。

## 真机验收（每场景 3 遍并记录）

1. 说话 -> Facts 出现且内容正确（需要先在设置页配置 provider/model/base_url/api_key）。
2. 间隔后提问 -> 回复引用记忆（`debug.recall_context` 非空）。
3. 记忆面板「触发反思」-> States/Actionables 新增条目。
4. Facts 删除 -> 列表消失且不再出现。
5. 蓝牙输入路由 + 说话人证据（配对/BLE 连接不视为通过）。
