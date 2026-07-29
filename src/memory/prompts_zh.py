"""Chinese prompt templates for the unified memory prototype."""

UNIFIED_MEMORY_EXTRACTION_PROMPT_ZH = """你是 AI 眼镜长期记忆系统的记忆提炼模块。

系统不再把 assistant_wakeup 和 allday_recording 当作两套彼此独立的记忆产品。两类来源都会进入同一条记忆线：
- episode：一次连续的交互或语音转写语义片段。
- fact：从 episode 中提炼出的可追溯、自包含、可独立召回的叙事事实。
- state：后续由 facts 更新出的长期主题/偏好/约束/风险等演化状态。
- index entry：类似 MemPalace 目录卡片的统一召回入口。

你现在需要从下面按时间顺序排列的对话/转写证据批次中提取 episode summary、episode canonical_topics 和 Hindsight 风格的高质量 narrative facts。证据可能来自用户与 assistant 的主动对话，也可能来自多人环境语音转写；请根据 speaker、role 和 time 判断参与者与语义流动。

canonical_topics 生成规则：
- canonical_topics 是 episode 级别的稳定主题名称，按相关性从高到低排序，输出 1-5 个。
- 如果本次 episode 与已有长期 topic 候选中的某个主题属于同一稳定对象或同一长期议题，必须复用候选里的 canonical_topic 原文，不要改写同义词。
- 只有当输入证据明确出现了新的稳定对象、项目、产品、人物关系或长期议题，才创建新 canonical topic。
- 不要输出过泛的 topic，例如“方案确定”“产品设计讨论”“部门协作”“问题讨论”“用户咨询”。topic 应尽量包含具体对象或稳定领域，例如“AI眼镜语音记忆系统”“手机推广策略”。
- 如果证据不足以确定具体对象，不要仅凭领域相关性强行复用已有长期 topic；只有严格确认是同一对象或议题时才复用，否则输出基于当前证据的保守上位主题，但不要生成碎片化短词。
- 复用已有 topic 需要严格判断：当前 episode/fact 的核心对象或具体议题、讨论目标和语义范围，必须与该 topic 的 canonical_name 基本一致。仅仅属于同一大领域、共享某个实体、使用相似词，或前后 episode 时间接近，都不足以复用。
- 如果一个 fact 只是涉及已有 topic 的背景，但它实际讨论的是另一个对象或议题，必须为该 fact 选择更准确的 primary_topic，不能为了保持 topic 数量而强行归入已有 topic。

已有长期 memory_states 参考：
{existing_memory_states}

memory_states 使用规则：
- state_scope=topic_state 且 state_type=topic 的状态，是 episode canonical_topics 和 fact primary_topic 的命名参考；如果当前证据表达的是同一长期对象或议题，优先原样复用 canonical_name。
- state_scope=entity_state 的状态，只用于理解实体的长期属性、偏好、约束、风险或关系，不能直接把 entity_state 的 canonical_name 当作 episode topic 或 fact primary_topic。
- 这些 states 只是历史背景，不是当前 episode 的事实证据。当前对话没有明确支持的内容不能写入 fact；当前对话与历史 state 冲突时，以当前对话为准。
- fact 的 primary_topic 必须描述该 fact 的主要议题；它可以复用 topic_state 的 canonical_name，但不能因为 entity_state 的名称而改变为实体属性标题。
- fact 的 primary_topic 只有在该 fact 的核心对象、讨论目标和语义范围都与 topic_state 的 canonical_name 相近时，才可以原样复用该名称。
- 不要仅因为 fact 与某个 topic_state 同属“健康”“产品”“团队”等宽泛领域，或 fact 中出现了该 topic 的相关背景，就复用该名称。若当前证据不能支持这种严格对应，应使用当前证据中的更具体主题，必要时输出新的保守 topic。
- 不能因为 entity_state 的名称、summary 或历史 timeline 而改变 fact 的 primary_topic；entity_state 只能辅助理解，不得作为 topic 候选直接复用。
- 每条 fact 必须额外输出一个 `primary_entity`，表示这条 fact 主要描述、影响或归属的唯一实体；它必须是一个实体对象，而不是数组。
- `primary_entity` 必须来自当前 fact 的 `entities`，不能因为实体只是被提及、提供建议、作为地点/工具或背景，就把它选为主要实体。多人互动时，选择该 fact 主要描述或影响的主体；如果 fact 主要描述用户自身的偏好、习惯、约束或风险，选择用户。
- `entities` 仍然保留所有与 fact 直接相关的实体，用于完整召回；后续 entity_state 匹配只使用 `primary_entity`，一条 fact 不得同时归属多个实体。

Hindsight 风格 narrative fact 的核心要求：
- 每条 fact 应覆盖一次完整 exchange 或一个清晰议题片段，而不是单个 utterance。不要把“用户提出问题”“助手给出建议”“用户否定/接受建议”机械拆成多条碎片；如果它们围绕同一问题相互回应，应优先合并成一条 narrative fact。
- 每条 fact 必须能在不阅读原始对话的情况下独立理解，并保留对话的 pragmatic flow：用户为什么提出这个问题，助手给了什么方案，用户如何回应，最后形成了什么倾向、决定、约束、未解决问题或下一步。
- 每条 fact 必须在 text 中自然体现五维度信息：what（完整事件/议题/方案/结论）、when（对话时间或明确时间锚点）、where（地点/场景/平台/项目范围；未提及时说明未提及具体地点/场景）、who（用户、助手和其他关键人物/组织及其角色）、why（明确原因、动机、担忧、分歧、约束、影响、结论或后续安排）。
- 对一个 5 轮左右的对话批次或一段多人转写片段，通常输出 1-3 条 facts；只有当批次中确实存在多个互不相关的事件/议题时才拆开。绝大多数情况下不要超过 5 条。

时间保真要求：
- 必须保留影响语义的顺序词和先后关系：first、first time、second、previous、next、later、earlier、before、after、once、again、subsequent、prior、last、most recent，以及“第一次/首次/第二次/之前/之后/此前/随后/后来/更早/最近一次/上一次”等。不要把“first service on March 15”弱化成“service experience”，而应保留“3月15日第一次保养/首次 service”这样的可比较时间锚。
- 必须在 text 和 keywords 中保留相对时间表达：yesterday、last Saturday、previous week、two months ago、about a month ago、mid-February、recently、shortly after，以及“昨天/上周六/前一周/两个月前/约一个月前/二月中旬/最近/不久后”等。如果它能根据 Conversation timestamp 无歧义换算，应在 occurred_start/occurred_end 写入补全后的日期或保守日期范围。
- 如果未来问题的答案依赖事件先后、间隔、第一次/上一次或“哪个更早”，fact text 必须同时包含事件对象和时间锚/顺序词，不能只存主题名。
- 如果同一批证据中多个事件可能被未来问题比较先后，应在一条 narrative fact 中明确写出相对顺序，或拆成多条各自带完整背景和时间锚的 facts；不要只保留比较中的一方。
- 带时间锚或顺序词的个人经历即使只是顺带提到，也应认真保留，例如购买、保养/维修、修理、预约、参加活动、旅行、会议、测试、失败、决定等。

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
14. keywords 只能包含用于检索的短实体、主题、症状、方案、约束、决定和关键时间/顺序锚，通常每个关键词 2-8 个汉字或一个短英文短语；对带时间锚的事件，必须加入原始或补全后的时间词，例如“March 15 2023”“first service”“3/22”“last Saturday”“two months ago”“上周六”“两个月前”。不要把完整句子、寒暄、礼貌话、语气词、泛化表达或“希望这个方法能帮到您”这类文本放入 keywords。
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
      "primary_entity": {"name": "该 fact 的唯一主要实体", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|TOPIC|PREFERENCE|OTHER"},
      "primary_topic": "稳定主题字符串；优先复用相关 topic_state 的 canonical_name，不要使用 entity_state 名称",
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

对话/转写证据批次：
{dialogue_batch}
"""


UNIFIED_STATE_UPDATE_PROMPT_ZH = """你是 AI 眼镜长期记忆系统中的 state 更新模块。

输入包含新存储的 narrative facts 和当前已有的长期 states。你的任务是更新或新建紧凑的 evolving states，帮助后续召回。state 不是复制某条 fact，而是跨 facts 提炼稳定偏好、反复出现的行为、长期约束、持续风险、重要关系或主题级状态。

边界说明：
- 主题、项目、议题的演化进展统一归入 topic_state，不再单独创建项目类 state。
- 具体任务、决定、承诺、提醒、开放问题应归入 actionable_item，不再创建任务类或承诺类 state。
- state_scope 只能是 topic_state 或 entity_state；topic_state 的 state_type 固定为 topic。
- 本模块只输出真正需要长期保留的 state；如果事实只表示一次性任务或承诺，不要输出 state。

规则：
1. 只有当信息在当前 episode 之后仍有价值时，才创建或更新 state。
2. 不要因为单条琐碎 fact 就创建 state，除非它是稳定偏好、长期约束、持续风险或重要的人生/项目背景。
3. 同一稳定对象相关的 facts 应合并到同一个 state，避免近重复。
4. 保留不确定性和最近变化。如果证据冲突，要明确写出冲突。
5. evidence_fact_ids 必须引用支撑该 state 的 fact ID。
6. state_type 只能是：topic、preference、profile、routine、relationship、constraint、risk。
7. importance 和 confidence 都是 0-1。
8. 只返回 JSON，不要 markdown。

输出格式：
{
  "states": [
    {
      "state_scope": "topic_state|entity_state",
      "state_type": "topic|preference|profile|routine|relationship|constraint|risk",
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


UNIFIED_TOPIC_STATE_UPDATE_PROMPT_ZH = """你是 AI 眼镜长期记忆系统中的 topic_state 更新模块。

输入已经完成主题解析：系统已经判断这批 facts 应归入某个长期 topic，或应创建一个新的长期 topic。你的任务是更新这个 topic_state 的 summary，而不是重新决定主题归属。

规则：
1. 只围绕给定 canonical_topic 更新，不要把无关 facts 合并进来。
2. summary 需要体现这个主题的长期状态：背景、最近变化、关键参与者、已经形成的决定/偏好/约束、仍未解决的问题和下一步。
3. 如果 existing_topic_state 已有内容，要增量融合，不要简单拼接，不要丢失仍然有效的长期信息。
4. 不要把单句 fact 改写成另一句 fact；topic_state 必须比 fact 更抽象、更稳定。
5. summary 必须是简短的当前状态快照，最多 1-2 句话，建议不超过 120 个中文字符；不要把历史 timeline 拼接进 summary。
6. time_line_updates 只记录本次 facts 带来的状态变化，输出 0-3 条；每条包含发生时间、变化类型、简短变化说明和 fact_ids。不要重复输出已有 timeline，也不要把没有变化的内容写入 timeline。
7. evidence_fact_ids 必须引用输入 facts 中支撑本次更新的 fact ID。
8. 只返回 JSON，不要 markdown。

输出格式：
{
  "update_needed": true,
  "canonical_name": "稳定主题名",
  "summary": "简短的当前长期 topic_state 快照",
  "time_line_updates": [
    {
      "occurred_at": "",
      "change_type": "confirmed|changed|rejected|resolved|updated",
      "summary": "本次状态变化",
      "fact_ids": [1]
    }
  ],
  "keywords": ["关键词1", "关键词2"],
  "entities": ["实体1", "实体2"],
  "canonical_topics": ["主题1"],
  "evidence_fact_ids": [1, 2],
  "importance": 0.8,
  "confidence": 0.85,
  "status": "active|stable|resolved|uncertain"
}

canonical_topic：
{canonical_topic}

已有 topic_state：
{existing_topic_state}

新 facts：
{facts}
"""


UNIFIED_ENTITY_STATE_UPDATE_PROMPT_ZH = """你是 AI 眼镜长期记忆系统中的 entity-scoped state 更新模块。

输入已经完成实体解析和属性主题初步分组：系统已经判断这批 facts 可能更新某个实体的某类长期状态。你的任务是更新这个实体的某个具体属性，而不是重新决定实体归属。

entity-scoped state 的目标：
- preference：某个实体稳定偏好、选择倾向、反复表达的喜好/厌恶。
- relationship：某个实体与他人/组织/项目之间的关系状态。
- profile：某个实体稳定画像、身份、背景、长期职责或重要上下文。
- routine：某个实体反复出现的习惯、流程、节奏。
- constraint：某个实体长期或当前持续影响行动的限制条件。
- risk：某个实体持续存在、会影响后续判断或行动的风险。

规则：
1. 只围绕给定 entity、state_type 和 attribute_name 更新，不要写成主题进展总结。
2. 如果 facts 只说明某个议题的进展，应留给 topic_state；这里只保留对 entity 本身长期有用的具体属性。
3. 如果 existing_entity_state 与当前属性不是同一件事，返回 update_needed=false，不要强行合并。
4. 如果 existing_entity_state 已有内容，要增量融合，不要简单拼接。
5. canonical_name 只能是简短、具体的属性或主题标题，例如“灵活健身方式偏好”或“健康管理”；不要包含实体名、state_type、斜杠、连字符或完整句子。实体由输入的 entity 单独表示，state_type 由单独字段表示。如果已有 entity_state 与当前属性相同，复用其不含实体和 state_type 的 canonical_name。
6. 如果输入只支持一次性事件、单次建议、临时请求或礼貌回应，返回 update_needed=false。
7. summary 必须是简短的当前状态快照，最多 1-2 句话，建议不超过 120 个中文字符；不要把历史 timeline 拼接进 summary。
8. time_line_updates 只记录本次 facts 带来的状态变化，输出 0-3 条；每条包含发生时间、变化类型、简短变化说明和 fact_ids。不要重复输出已有 timeline，也不要把没有变化的内容写入 timeline。
9. summary 必须能回答：“关于这个实体的这个属性，我们长期应该记住什么？”
10. evidence_fact_ids 必须引用输入 facts 中支撑本次更新的 fact ID。
11. 只返回 JSON，不要 markdown。

输出格式：
{
  "update_needed": true,
  "canonical_name": "属性或主题短标题，不包含实体名和 state_type",
  "summary": "简短的当前 entity-scoped state 快照",
  "time_line_updates": [
    {
      "occurred_at": "",
      "change_type": "confirmed|changed|rejected|resolved|updated",
      "summary": "本次状态变化",
      "fact_ids": [1]
    }
  ],
  "keywords": ["关键词1", "关键词2"],
  "entities": ["实体1", "实体2"],
  "canonical_topics": ["主题1"],
  "evidence_fact_ids": [1, 2],
  "importance": 0.8,
  "confidence": 0.85,
  "status": "active|stable|resolved|uncertain"
}

entity_state_target：
{entity_state_target}

已有 entity_state：
{existing_entity_state}

新 facts：
{facts}
"""


UNIFIED_ACTIONABLE_ITEM_EXTRACTION_PROMPT_ZH = """你是 AI 眼镜长期记忆系统中的 actionable item 提取模块。

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


RECALL_QUERY_ANALYSIS_PROMPT_ZH = """你是 AI 眼镜长期记忆系统中用来处理用户 recall query 分析器。

当前记忆系统在同一个共享索引中检索 episode、fact、evolving state 和 actionable item。请分析用户查询，决定应该优先检索哪些索引层。

判断准则：
- 只有当 query 明确指向用户与助手的主动对话，才偏向 assistant_wakeup；明确指向全天录音、会议、旁听、多人数对话，才偏向 allday_recording；不确定时两者都保留。
- 精确证据、日期、人名、发生过什么优先 fact；任务、承诺、决定、开放问题、风险、提醒、推荐优先 actionable_item；稳定偏好、长期约束、习惯/流程、关系画像、主题/项目/议题演化优先 state；宽泛回顾、某段经历概括优先 episode。
- 不确定时保持宽检索，漏掉证据比多取几个候选更糟。
- keywords 输出 2-8 个短检索词，优先保留具体人物、组织、产品、项目、主题、动作、结果、约束和时间锚点；不要输出完整句子、寒暄或泛化词。
- entities 输出对语义检索有帮助的实体名称及类型。实体可以是人物、组织、地点、产品、项目、技术或具体概念；普通的“今天/昨天/上周”等时间表达不要作为实体。

只返回 JSON：
{
  "source_types": ["assistant_wakeup", "allday_recording"],
  "index_levels": ["fact", "actionable_item", "state", "episode"],
  "needs_broad_evidence": false,
  "query_rewrite": "面向检索的改写",
  "keywords": ["关键词1", "关键词2"],
  "entities": [{"name": "实体名", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|OTHER"}]
}

用户查询：
{query}
"""
