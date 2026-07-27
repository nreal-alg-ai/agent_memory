"""Chinese prompt templates for the unified memory prototype."""

UNIFIED_MEMORY_EXTRACTION_PROMPT_ZH = """你是一个受 MemPalace 启发的 AI 眼镜统一记忆系统的记忆提炼模块。

系统不再把 assistant_wakeup 和 allday_recording 当作两套彼此独立的记忆产品。两类来源都会进入同一条记忆线：
- episode：一次连续的交互或语音转写语义片段。
- fact：从 episode 中提炼出的可追溯、自包含、可独立召回的叙事事实。
- state：后续由 facts 更新出的长期主题/偏好/任务状态。
- index entry：类似 MemPalace 目录卡片的统一召回入口。

你现在需要从下面按时间顺序排列的对话/转写证据批次中提取 episode summary、episode canonical_topics 和 Hindsight 风格的高质量 narrative facts。证据可能来自用户与 assistant 的主动对话，也可能来自多人环境语音转写；请根据 speaker、role 和 time 判断参与者与语义流动。

canonical_topics 生成规则：
- canonical_topics 是 episode 级别的稳定主题名称，按相关性从高到低排序，输出 1-5 个。
- 如果本次 episode 与已有长期 topic 候选中的某个主题属于同一稳定对象或同一长期议题，必须复用候选里的 canonical_topic 原文，不要改写同义词。
- 只有当输入证据明确出现了新的稳定对象、项目、产品、人物关系或长期议题，才创建新 canonical topic。
- 不要输出过泛的 topic，例如“方案确定”“产品设计讨论”“部门协作”“问题讨论”“用户咨询”。topic 应尽量包含具体对象或稳定领域，例如“AI眼镜语音记忆系统”“手机推广策略”。
- 如果证据不足以确定具体对象，优先复用最相关的已有长期 topic；仍无法确定时输出更保守的上位主题，但不要新建碎片化短词。

Hindsight 风格 narrative fact 的核心要求：
- 每条 fact 应覆盖一次完整 exchange 或一个清晰议题片段，而不是单个 utterance。不要把“用户提出问题”“助手给出建议”“用户否定/接受建议”机械拆成多条碎片；如果它们围绕同一问题相互回应，应优先合并成一条 narrative fact。
- 每条 fact 必须能在不阅读原始对话的情况下独立理解，并保留对话的 pragmatic flow：用户为什么提出这个问题，助手给了什么方案，用户如何回应，最后形成了什么倾向、决定、约束、未解决问题或下一步。
- 每条 fact 必须在 text 中自然体现五维度信息：what（完整事件/议题/方案/结论）、when（对话时间或明确时间锚点）、where（地点/场景/平台/项目范围；未提及时说明未提及具体地点/场景）、who（用户、助手和其他关键人物/组织及其角色）、why（明确原因、动机、担忧、分歧、约束、影响、结论或后续安排）。
- 对一个 5 轮左右的对话批次或一段多人转写片段，通常输出 1-3 条 facts；只有当批次中确实存在多个互不相关的事件/议题时才拆开。绝大多数情况下不要超过 5 条。

提取规则：
1. 提取 0-5 条 facts，不要为了覆盖每一轮强行生成 fact。
2. 每条 fact 必须是一段完整叙事，至少包含“议题背景 + 用户/助手的观点或动作流动 + 理由/分歧/约束/结论/下一步”中的关键要素。
3. 保留可被直接问到的具体细节：人名、地点、标题、颜色、日期、星期、相对时间、数量、金额、时长、产品、机构、建议、约束、决定和用户偏好。
4. 不要丢弃 “by the way / I also / I just / last Saturday / two months ago / 顺便 / 我还” 这类附带提到的个人事件；但如果它们属于同一 exchange 的上下文，应合并进同一条 narrative fact，而不是拆成无背景短句。
5. 只有真正互不相关的事件才拆开；时间推理需要比较先后/间隔的事件可以拆成多条，但每条仍必须保留完整背景和时间锚点。
6. 只使用输入证据，不要编造完成状态、意图或原因。
7. 如果 assistant 的推荐中包含未来可能被问到的具体条目，要放入相关 exchange 的 narrative fact，并写清用户是否接受、拒绝、犹豫或提出约束。
8. priority 为 0-100，只保留至少 60 分的事实。
9. fact_type 只能是 semantic 或 episodic。
10. fact_subject 只能是 user、assistant、world、project、system、other。
11. fact_kind 只能是 preference、decision、request、recommendation、action、commitment、open_question、risk、error、context、instruction、other。
12. 不要输出只有“用户说了 X”“助手建议 Y”的短 fact；如果删除议题背景、理由、分歧或结论后会变成泛泛短句，必须补回这些信息；若对话没有足够信息支撑，则不要输出该 fact。
13. 不要把 assistant 的寒暄、礼貌收尾、泛化鼓励或无具体信息的回复单独作为 fact，例如“希望这个方法能帮到您”“有其他问题可以继续沟通”“好的”“不客气”等；除非它明确改变了用户决定、承诺或下一步。
14. keywords 只能包含用于检索的短实体、主题、症状、方案、约束或决定，通常每个关键词 2-8 个汉字或一个短英文短语；不要把完整句子、寒暄、礼貌话、语气词、泛化表达或“希望这个方法能帮到您”这类文本放入 keywords。
15. 只返回 JSON，不要 markdown。

输出格式：
{
  "episode_title": "简短具体标题",
  "episode_summary": "可独立理解的 episode 总结",
  "canonical_topics": ["按相关性排序的稳定主题1", "稳定主题2"],
  "facts": [
    {
      "text": "覆盖完整 exchange 的自包含 narrative fact，在一段文本中体现 what/when/where/who/why，包含背景、用户/助手观点或动作流动、理由/分歧/约束/结论/下一步",
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

已有长期 canonical topics 候选：
{existing_long_term_topics}

对话/转写证据批次：
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

输入包含新存储的 narrative facts。请从中提取未来真正需要跟进、提醒、执行、复盘或决策追踪的具体可执行事项。actionable item 和 evolving state 分开：state 描述持续变化的长期状态、偏好、约束和背景；actionable item 必须是可以被检查、完成、追踪，或明确作为决定/承诺/风险/开放问题被召回的事项。

只能提取 facts 直接支持的内容。

规则：
1. 提取 0-4 条 actionable items。多数普通对话可以输出空列表，不要为了覆盖 facts 强行生成。
2. 只提取强 actionable：用户明确要求提醒/跟进/记录/安排/执行，用户或 assistant 明确承诺未来会做，用户做出可追踪的决定，存在必须后续解决的开放问题，或存在会阻塞行动的高价值风险。
3. 不要把“用户愿意试试/可以试一试/听起来不错/可能会考虑”单独提取为 actionable item；这类弱尝试意愿默认只属于 fact 或 state。只有当它同时包含明确提醒需求、截止时间、具体后续检查、强承诺或可验证执行计划时才提取。
4. 不要把 assistant 的普通建议单独提取为 actionable item。只有当用户明确采纳、要求后续提醒/跟进，或该建议已经变成用户的任务/承诺/决定时才提取。
5. 普通约束、偏好、背景信息应留给 state，不要作为 constraint item；只有当约束正在阻塞一个明确行动或决策时才提取。
6. 不要复制每条 fact。若多条 facts 指向同一件事，只保留一条最具体、最可追踪的 item。
7. 每个 item 必须可独立理解：包含执行人/owner、目标对象、上下文、原因、截止时间或时间锚点（如果存在）、当前状态。
8. evidence_fact_ids 必须引用输入中的 fact ID。
9. item_type 只能是：task、commitment、decision、follow_up、open_question、risk、reminder、recommendation、constraint、other。
10. status 只能是：open、in_progress、done、blocked、decided、noted、unknown。
11. importance 和 confidence 都是 0-1。弱尝试意愿的 confidence 不应提高来绕过规则。
12. 只返回 JSON，不要 markdown。

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
