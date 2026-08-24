# Agent Memory MCP Server

`memory_mcp_server.py` exposes the existing `MemoryRuntime` through the MCP
stdio transport. The server owns one `SessionDB`, `MemoryNodeManager`, and
`MemoryRuntime` for its entire process lifetime.

## Start

```bash
/Applications/miniconda3/envs/python3_11/bin/python \
  /Users/zhouboyu/Documents/agent_memory/scripts/memory_mcp_server.py
```

The MCP protocol uses stdout. Memory logs go to stderr by default, or to
`memory_mcp_server.log_path` when supplied. Database path, log level, queue
timeout, runtime/model settings, and environment references such as
`${GLM_API_KEY}` are all read from `config.yaml` during startup.

Example client configuration:

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "/Applications/miniconda3/envs/python3_11/bin/python",
      "args": [
        "/Users/zhouboyu/Documents/agent_memory/scripts/memory_mcp_server.py"
      ]
    }
  }
}
```

## Tools

The MCP server exposes only two tools:

- `process_audio_files`: accepts a non-empty `files` array. Each item contains
  an `audio_path` and may override `source_type`, `session_start`, and `tags`.
  The server runs VAD, ASR, and speaker identification, submits the resulting
  transcript segments to memory, flushes the store queue, and runs the
  configured reflection workflow for each file. The response contains one
  report per file, including transcription and memory-store results.
- `trigger_memory_recall`: searches the latest committed memory snapshot using
  a `query`, with optional `tags` and `time_end`. The response contains the
  assembled `memory_context`, recall-path metadata, and timing information.

`process_audio_file`, transcript ingestion, reflection, and pending-queue
operations are internal implementation steps of `process_audio_files`; they are
not exposed as separate MCP tools.

The server does not expose internal database methods or the manager's private
fact/state extraction functions. This keeps pending buffering and write
ordering in `MemoryRuntime`.
