"""Chinese prompt templates for the unified memory prototype."""

UNIFIED_MEMORY_EXTRACTION_PROMPT_ZH = """你是一个受 MemPalace 启发的 AI 眼镜统一记忆系统的记忆提炼模块。

系统不再把 assistant_wakeup 和 allday_recording 当作两套彼此独立的记忆产品。两类来源都会进入同一条记忆线：
- episode：一次连续的交互或语音转写语义片段。
- fact：从 episode 中提炼出的可追溯、自包含、可独立召回的叙事事实。
- state：后续由 facts 更新出的长期主题/偏好/任务状态。
- index entry：类似 MemPalace 目录卡片的统一召回入口。

你现在需要从下面按时间顺序排列的 assistant 对话批次中提取 episode summary 和高质量 facts。

要求：
1. 提取 0-12 条 facts，不要为了覆盖每一轮强行生成 fact。
2. 每条 fact 必须脱离原始对话后仍可独立理解和召回。
3. 保留可被直接问到的具体细节：人名、地点、标题、颜色、日期、星期、相对时间、数量、金额、时长、产品、机构、建议、约束、决定和用户偏好。
4. 不要丢弃 “by the way / I also / I just / last Saturday / two months ago / 顺便 / 我还” 这类附带提到的个人事件，LongMemEval 经常会问这些内容。
5. 独立事件要拆开，不要压缩成宽泛主题。对时间推理友好：可能被比较先后/间隔的事件必须各自保留时间锚点。
6. 只使用输入证据，不要编造完成状态、意图或原因。
7. 如果 assistant 的推荐中包含未来可能被问到的具体条目，也要保留。
8. priority 为 0-100，只保留至少 60 分的事实。
9. fact_type 只能是 semantic 或 episodic。
10. fact_subject 只能是 user、assistant、world、project、system、other。
11. fact_kind 只能是 preference、decision、request、recommendation、action、commitment、open_question、risk、error、context、instruction、other。
12. 只返回 JSON，不要 markdown。

输出格式：
{
  "episode_title": "简短具体标题",
  "episode_summary": "可独立理解的 episode 总结",
  "facts": [
    {
      "text": "完整叙事事实",
      "keywords": ["关键词1", "关键词2"],
      "entities": [{"name": "实体名", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|TOPIC|PREFERENCE|OTHER"}],
      "primary_topic": "稳定主题字符串",
      "fact_type": "semantic|episodic",
      "fact_subject": "user|assistant|world|project|system|other",
      "fact_kind": "preference|decision|request|recommendation|action|commitment|open_question|risk|error|context|instruction|other",
      "priority": 80,
      "occurred_start": "",
      "occurred_end": "",
      "time_confidence": "explicit|inferred_from_turn|unknown",
      "where": ""
    }
  ]
}

对话批次：
{dialogue_batch}
"""


UNIFIED_STATE_UPDATE_PROMPT_ZH = """你是受 MemPalace 启发的 AI 眼镜统一长期记忆系统中的 state 更新模块。

输入包含新存储的 narrative facts 和当前已有的长期 states。你的任务是更新或新建紧凑的 evolving states，帮助后续召回。state 不是复制某条 fact，而是跨 facts 提炼稳定偏好、进行中的项目/任务、反复出现的行为、未完成承诺、重要关系或主题级状态。

规则：
1. 只有当信息在当前 episode 之后仍有价值时，才创建或更新 state。
2. 不要因为单条琐碎 fact 就创建 state，除非它是稳定偏好、承诺、决定或重要的人生/项目背景。
3. 同一稳定对象相关的 facts 应合并到同一个 state，避免近重复。
4. 保留不确定性和最近变化。如果证据冲突，要明确写出冲突。
5. evidence_fact_ids 必须引用支撑该 state 的 fact ID。
6. state_type 只能是：preference、task_state、project_state、relationship、routine、topic_state、commitment、constraint、risk、profile、other。
7. importance 和 confidence 都是 0-1。
8. 只返回 JSON，不要 markdown。

输出格式：
{
  "states": [
    {
      "state_type": "preference|task_state|project_state|relationship|routine|topic_state|commitment|constraint|risk|profile|other",
      "canonical_name": "稳定短名称",
      "summary": "可独立理解的 evolving state",
      "evidence_fact_ids": [1, 2],
      "keywords": ["关键词1", "关键词2"],
      "entities": ["实体1", "实体2"],
      "canonical_topics": ["主题1"],
      "importance": 0.8,
      "confidence": 0.85,
      "status": "active|stable|resolved|uncertain"
    }
  ]
}

已有 states：
{existing_states}

新 facts：
{facts}
"""


UNIFIED_ACTIONABLE_ITEM_EXTRACTION_PROMPT_ZH = """你是受 MemPalace 启发的 AI 眼镜统一长期记忆系统中的 actionable item 提取模块。

输入包含新存储的 narrative facts。请从中提取未来可能需要跟进、召回、提醒、复盘或决策追踪的具体可执行事项。actionable item 和 evolving state 分开：state 描述持续变化的长期状态，而 actionable item 是可以被检查、完成、追踪，或作为决定/承诺/风险/开放问题被明确召回的事项。

只能提取 facts 直接支持的内容。

规则：
1. 提取 0-12 条 actionable items，不要强行生成。
2. 包括决策、承诺、任务、后续跟进、开放问题、风险/阻塞、截止时间、影响行动的约束，以及用户未来可能询问的具体 assistant 建议。
3. 不要复制每条 fact。若某条 fact 只是背景信息、没有后续跟进价值，应跳过。
4. 每个 item 必须可独立理解：包含执行人/owner、目标对象、上下文、原因、截止时间或时间锚点、当前状态。
5. evidence_fact_ids 必须引用输入中的 fact ID。
6. item_type 只能是：task、commitment、decision、follow_up、open_question、risk、reminder、recommendation、constraint、other。
7. status 只能是：open、in_progress、done、blocked、decided、noted、unknown。
8. importance 和 confidence 都是 0-1。
9. 只返回 JSON，不要 markdown。

输出格式：
{
  "actionable_items": [
    {
      "item_type": "task|commitment|decision|follow_up|open_question|risk|reminder|recommendation|constraint|other",
      "canonical_name": "稳定短名称",
      "summary": "可独立理解的可执行事项",
      "owner": "user|assistant|other|unknown",
      "status": "open|in_progress|done|blocked|decided|noted|unknown",
      "due_at": "",
      "evidence_fact_ids": [1, 2],
      "keywords": ["关键词1", "关键词2"],
      "entities": ["实体1", "实体2"],
      "canonical_topics": ["主题1"],
      "importance": 0.8,
      "confidence": 0.85
    }
  ]
}

新 facts：
{facts}
"""


RECALL_QUERY_ANALYSIS_PROMPT_ZH = """你是统一 MemPalace 风格长期记忆系统的 recall query 分析器。

当前记忆系统在同一个共享索引中检索 episode、fact、evolving state 和 actionable item。请分析用户查询，决定应该优先检索哪些索引层。

判断准则：
- 只有当 query 明确指向用户与助手的主动对话，才偏向 assistant_wakeup；明确指向全天录音、会议、旁听、多人数对话，才偏向 allday_recording；不确定时两者都保留。
- 精确证据、日期、人名、发生过什么优先 fact；任务、承诺、决定、开放问题、风险、提醒、推荐优先 actionable_item；稳定偏好、任务状态、项目进展优先 state；宽泛回顾、某段经历概括优先 episode。
- 不确定时保持宽检索，漏掉证据比多取几个候选更糟。

只返回 JSON：
{
  "source_types": ["assistant_wakeup", "allday_recording"],
  "index_levels": ["fact", "actionable_item", "state", "episode"],
  "needs_broad_evidence": false,
  "query_rewrite": "面向检索的改写",
  "keywords": ["关键词1", "关键词2"]
}

用户查询：
{query}
"""
