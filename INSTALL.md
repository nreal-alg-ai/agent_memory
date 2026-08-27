# Agent Memory 安装与运行

本文介绍如何在 macOS 或 Linux 上安装并运行 Agent Memory。以下命令默认在工程根目录执行。

## 1. 准备系统环境

建议使用 Python 3.11。音频处理会加载 ASR、VAD 和说话人识别模型，首次运行需要联网下载模型，并预留足够的磁盘和内存空间。

macOS 还可以先安装基础工具：

```bash
brew install curl git
```

## 2. 安装 micromamba

使用 micromamba 官方安装脚本：

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

安装完成后重新加载 shell 配置。zsh 使用：

```bash
source ~/.zshrc
micromamba --version
```

如果当前 shell 尚未加载 micromamba，也可以临时启用：

```bash
eval "$(micromamba shell hook --shell zsh)"
```

bash 用户将上面的 `zsh` 替换为 `bash`，并按需执行 `source ~/.bashrc`。

## 3. 创建 Python 环境

```bash
micromamba create -n agent-memory python=3.11 -c https://prefix.dev/conda-forge \
  --override-channels
micromamba activate agent-memory
python --version
```

升级 pip 基础工具并安装工程依赖：

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

依赖包括 PyTorch、torchaudio、Transformers、Qwen ASR、FunASR、ModelScope、Silero VAD、jieba、SQLite FTS 查询所需的上层代码以及音频处理库。安装过程较长属于正常情况。

## 4. 配置 MCP 客户端

MCP 客户端应使用当前环境中的 Python 和绝对脚本路径，例如：

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "/path/to/micromamba/envs/agent-memory/bin/python",
      "args": [
        "/path/to/agent_memory/scripts/memory_mcp_server.py"
      ]
    }
  }
}
```

可以用下面的命令查看环境路径：

```bash
micromamba env list
which python
```

当前 MCP 服务主要接口为：

- `process_audio_files`：提交一个或多个音频文件的异步处理任务
- `get_audio_processing_job`：查询音频任务状态和结果
- `trigger_memory_recall`：执行记忆召回

## 5. 运行端到端测试

使用默认测试音频目录运行：

```bash
python scripts/test_voice_memory_runtime.py \
  --config config.yaml \
  --max-files 1 \
  --max-duration-s 30 \
  --override
```

其中：

- `--max-files` 限制本次处理的音频文件数量
- `--max-duration-s` 限制单个音频文件的最长处理时长
- `--override` 删除配置路径下已有的数据库、日志和报告后重新测试

也可以直接测试 ASR：

```bash
python scripts/voice/test_nonstreaming_vad_asr.py \
  --config config.yaml \
  --max-files 1
```

## 6. 常见问题

### 模型首次加载很慢或 MCP 握手超时

第一次运行可能同时下载 Qwen ASR、VAD 和 CAM++ 说话人模型。先单独运行一次端到端测试完成模型缓存，再启动 MCP 客户端。音频处理接口已经是异步任务模式，提交后应通过 `get_audio_processing_job` 查询进度。

### `database is locked`

确认同一个数据库没有被多个 Agent Memory 进程同时使用。不同测试任务应配置不同的 `memory_mcp_server.result_dir`，不要并行写入同一个 `memory.db`。

### `no such module: fts5`

当前 Python 使用的 SQLite 没有编译 FTS5。需要更换为带 FTS5 的 Python/SQLite 环境；仅重新安装 Python 包通常不能解决系统 SQLite 缺少编译选项的问题。

### 远程 embedding 超时

检查 `GLM_API_KEY`、网络和 `memory_manager.embedding` 中的 `base_url`。调试时可以暂时切换到本地 hash provider，但其语义检索效果会明显下降。

### ASR 没有输出

确认 `voice_runtime.asr.backend` 与对应的 `backends` 配置一致，并查看 `tmp/mcp/memory.log` 和 `tmp/mcp/asr_result/`。如果使用 Qwen ASR，首次运行需要等待模型下载完成。

## 7. 停止服务

前台运行 MCP 服务时，使用 `Ctrl-C` 停止。服务会尝试清空 MemoryRuntime 中尚未提交的 pending 数据，并关闭后台任务队列。
