# Agent Memory Android App

一个独立的 Android 测试应用：把本仓库 `src/` 的 agent_memory 记忆系统作为唯一记忆引擎，
套用 Android 语音外壳（音频采集 / VAD / 唤醒词检测 / ASR / 声纹 / WebView UI / TTS），
支持全天候收音待机、语音问答、记忆管理（事实 / 状态 / 待办 / 片段）与召回调试。

> 记忆数据保存在 App 私有目录（`<app_home>/data/agent_memory/memory.db`），独立使用，不导入、不迁移其他数据。

## 功能

- 全天待机：开启后持续后台收音，检测到唤醒词「你好小忆」后回复「我在，请说」并倾听提问。
- 语音问答：提问实时转写显示，回复以聊天气泡呈现并语音朗读。
- 记忆管理：事实 / 状态 / 待办 / 片段四类视图，支持搜索、手动导入、触发反思与删除。
- 召回调试：查看最近一次聊天召回的记忆上下文（recall path / 耗时）。
- 声纹：录入参考声纹并用于说话人分类。
- 独立配置：聊天 LLM 必填；Embedding 选填（填齐后独立注入，留空自动回退本地 hash 向量）。

## 环境要求

- 设备：Android 8.0+（minSdk 26），arm64 架构（当前 APK 只打包 arm64-v8a）。
- 构建（仅开发者需要）：JDK 17、Android SDK 35（build-tools 35.x）。
- 本地模型包：kws / vad / online_asr / ambient_asr / speaker，安装方式见「首次配置」。

## 构建与安装

### 方法一：安装预构建 APK

将 `android/app/build/outputs/apk/debug/app-debug.apk` 拷贝到手机直接安装（需允许安装未知来源），
或通过 adb 安装：

```bash
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
```

> 仅安装 APK 还无法使用语音功能，仍需安装本地模型包；建议直接使用下面的「一键安装」。

### 方法二：自行构建

```bash
cd android
JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
ANDROID_HOME=/opt/homebrew/share/android-commandlinetools \
./gradlew assembleDebug
```

APK 输出路径：`android/app/build/outputs/apk/debug/app-debug.apk`

安装并启动：

```bash
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.agentmemory.test/.MainActivity
```

### 一键安装（推荐：APK + 本地模型包）

模型包（约 286MB）不随 git 分发，请从以下链接下载：

```text
https://xreal.feishu.cn/wiki/TWeJwLMzzilrKTksKIVcyYVln1w
```

下载得到 `x4000-sherpa-1.13.4-v2.zip` 后解压到：

```text
android/model-pack/x4000-sherpa-1.13.4-v2/
├── manifest.json
├── vad/  kws/  online_asr/  sensevoice/  speaker/
```

连接手机并授权 USB 调试后，在仓库根目录运行：

```bash
bash scripts/setup_device.sh --serial <DEVICE_SERIAL>
```

`<DEVICE_SERIAL>` 是手机的 ADB 设备序列号。手机连接电脑并开启 USB 调试后，运行：

```bash
adb devices -l
```

输出第一列即为序列号，例如：

```text
R5CT3466WGW    device usb:1-1 product:r0qzcx model:SM_S9010 device:r0q
```

则命令为：

```bash
bash scripts/setup_device.sh --serial R5CT3466WGW
```

脚本依次：安装 APK（覆盖安装，保留数据）→ 校验并安装本地模型包 → 启动 App。

## 首次配置

1. 打开 App → 右上角设置 → 「模型与服务配置 → 高级设置」。
2. 填写聊天模型配置（必填）：服务提供商、模型名称、API 地址（必须 HTTPS）、API 密钥。
3. 填写 Embedding 配置（选填）：provider 必须为 `openai`，模型、API 地址（HTTPS）、密钥填齐后独立生效；
   留空则沿用聊天接口地址，远程失败时自动降级本地 hash 向量。
4. 安装本地模型：填写模型清单 URL（HTTPS）后点击「下载或更新本地模型」；
   或使用 `android/tools/install_local_model_pack.py` 通过 adb 本地安装。
5. 运行「模型自检」，五项全部通过后再开始使用。

## 使用说明

1. 主界面点击「开启全天待机」：进入持续后台收音，状态栏显示绿色呼吸圆点与实时音量条，VAD 语句数实时增长。
2. 说唤醒词「你好小忆」：听到「我在，请说」回应，聊天区同步显示该气泡。
3. 说出问题：聊天区实时显示转写文字；提问结束后生成用户气泡，回复到达后生成助手气泡并语音朗读。
4. 记忆管理：设置 → 记忆，查看事实 / 状态 / 待办 / 片段；支持搜索、手动导入、触发反思、删除事实。
5. 召回调试：设置 → Debug，查看最近一次聊天的 `recall_context`、`recall_path`、耗时。
6. Embedding 验证（可选）：`GET /api/embedding/status` 返回 `remote` / `hash_fallback` / `not_configured`。

## 本地验证（无需手机）

```bash
cd agent_memory_android_test
PYTHONDONTWRITEBYTECODE=1 conda run -n hermes python scripts/local_smoke_test.py
```

覆盖：启动、静态资源、环境转写存储、敏感信息过滤、唤醒问句队列、/api/chat、统计 / 列表、反思、
事实删除、手动导入、声纹录入与分类、Embedding 配置分支。

## 真机验收（每场景 3 遍并记录）

0. 设置页填齐聊天四项（Embedding 可填可不填）；若填了 Embedding，`/api/embedding/status` 必须 `remote_ok=true`。
1. 开启全天待机 → 绿色呼吸点出现，说话时音量条跳动，VAD 语句数增长。
2. 说「你好小忆」→ 听到「我在，请说」，聊天区出现对应气泡；随后提问实时显示转写，回复以气泡呈现并朗读。
3. 说话 → Facts 出现且内容正确。
4. 间隔后提问 → 回复引用记忆（`debug.recall_context` 非空）。
5. 记忆面板「触发反思」→ States/Actionables 新增条目。
6. Facts 删除 → 列表消失且不再出现。
7. 蓝牙输入路由 + 说话人证据（配对/BLE 连接不视为通过）。

## 目录结构

```
agent_memory_android_test/
├── android/      Android 外壳（Kotlin + Chaquopy 打包 Python 引擎）
├── backend/      本地 HTTP 后端（Python）
├── static/       前端（聊天 + 记忆视图 + 召回调试）
├── scripts/      本地冒烟测试
└── ../src/       agent_memory 记忆引擎源码（同一仓库，直接引用）
```

## HTTP API 概览

- `POST /api/chat`：文字聊天，recall → LLM 回复 → store_interaction，响应含 `debug.recall_context`。
- `GET /api/embedding/status`：探测当前 embedding 模式。
- `GET /api/agent-memory/stats|facts|states|actionables|episodes`：记忆统计与列表（q/limit）。
- `DELETE /api/agent-memory/facts/{id}`：删除事实及关联索引。
- `POST /api/agent-memory/reflect`、`POST /api/agent-memory/import`：触发反思、手动导入。
- `GET /api/runtime`、`GET /api/audio/capabilities`、`GET /api/speaker/profile`、`GET /health`。

## 常见问题

- 语音问答没有回复：检查聊天模型配置、网络状态与五项模型自检是否通过。
- 找不到「高级设置」：该入口仅原生 Android 环境显示；浏览器模式只支持文字聊天与记忆面板。
- 唤醒词不能识别：当前固定为「你好小忆」，需先完成模型安装与自检，并确认收音输入路由正常。
