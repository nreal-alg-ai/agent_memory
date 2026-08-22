# Agent Memory MCP Server

`memory_mcp_server.py` exposes the existing `MemoryRuntime` through the MCP
stdio transport. The server owns one `SessionDB`, `MemoryNodeManager`, and
`MemoryRuntime` for its entire process lifetime.

## Start

```bash
/Applications/miniconda3/envs/python3_11/bin/python \
  /Users/zhouboyu/Documents/agent_memory/scripts/memory_mcp_server.py \
  --config /Users/zhouboyu/Documents/agent_memory/config.yaml \
  --db-path /Users/zhouboyu/Documents/agent_memory/tmp/mcp/memory.db \
  --log-path /Users/zhouboyu/Documents/agent_memory/tmp/mcp/memory.log
```

The MCP protocol uses stdout. Memory logs go to stderr by default, or to
`--log-path` when supplied. Environment references such as `${GLM_API_KEY}` in
`config.yaml` are expanded during startup.

Example client configuration:

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "/Applications/miniconda3/envs/python3_11/bin/python",
      "args": [
        "/Users/zhouboyu/Documents/agent_memory/scripts/memory_mcp_server.py",
        "--config",
        "/Users/zhouboyu/Documents/agent_memory/config.yaml",
        "--db-path",
        "/Users/zhouboyu/Documents/agent_memory/tmp/mcp/memory.db"
      ]
    }
  }
}
```

## Tools

- `accept_single_interaction_turn`: accepts one `user_message` and optional
  `assistant_response`, tags, and `turn_timestamp`. It returns a queue report;
  the runtime may keep the turn pending until segmentation decides to flush it.
- `accept_single_transcript_segment`: accepts one transcript `segment`, with
  optional `source_type` and tags. Compatible segments are buffered by the
  runtime.
- `trigger_memory_reflect`: flushes pending interaction/transcript input and
  queues state/actionable-item reflection. Use `limit` and
  `reflect_timestamp` to override the configured defaults.
- `trigger_memory_recall`: immediately searches the latest committed database
  snapshot. It returns `memory_context`, `actual_recall_path`, and recall
  timing/diagnostic fields. `recall_path` accepts `stage1`, `stage2`, or
  `normal`.

The server does not expose internal database methods or the manager's private
fact/state extraction functions. This keeps pending buffering and write
ordering in `MemoryRuntime`.
