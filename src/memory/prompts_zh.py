"""Chinese prompt templates for the unified memory prototype."""

ENTITY_EXTRACTION_GUIDANCE_ZH = """实体提取规则:

实体不是只限传统 NER。这里的实体指后续可以跨 facts 聚合、检索、建图的语义锚点。
优先抽取对长期记忆有复用价值的名词或短名词短语，而不是只抽取专有名词。

实体类型(type 可选值):
- PERSON(人): 对话中提到的具体人名、称呼、角色或 speaker
- ORGANIZATION(组织): 公司、团队、机构
- LOCATION(地点): 地理位置、场所
- PRODUCT(产品): 产品名、服务名
- PROJECT(项目): 项目名、产品名或长期工作事项
- TECHNOLOGY(技术): 技术栈、框架、库、工具、API 或系统
- CONCEPT(概念): 抽象概念、方法论、理论或可复用想法
- TOPIC(主题): 讨论的话题领域
- PREFERENCE(偏好): 用户的偏好、喜好、习惯或厌恶
- OTHER(其他): 明确提到但不适合上述类型的实体

应该抽取的实体包括：
- 对话主体或角色：用户、助手、speaker_1、speaker_2、妻子、孩子、团队、客户等
- 用户长期相关的领域、问题、任务、状态或场景：健康管理、身体状态、工作、商务活动、应酬、家庭教育、夫妻沟通、疲劳感等
- 可复用的方案、方法、工具、活动或对象：健康饮食、家庭会议、统一规则、野餐、瑜伽垫等
- 明确影响用户选择的约束对象或条件：经济负担、固定作息、时间不足、工作压力等

不要抽取普通时间表达作为实体，例如：今天、昨天、上周、最近三天、2026-05-07、10:30、三个月。
时间应作为 fact 的时间元数据处理，不进入 entity graph。
只有有语义身份的命名时间概念才可作为实体，例如：春节、Q3 财报季、Sprint 42。

不要抽取纯属性、纯形容词短语、孤立程度词或泛化标签作为实体；它们应保留在 fact text、keywords、topic 或 state 中。
例如：低场地依赖、低强度、高优先级、低成本、强隐私、轻量级。
但如果短语中包含可复用的核心对象或场景，应抽取核心对象，例如：
- “长期高强度工作带来的疲劳感”可抽取“工作”“疲劳感”
- “高频商务活动”可抽取“商务活动”
- “经济负担太重”可抽取“经济负担”

实体应来自对话中明确出现或由角色/事实主体直接确定的内容，不要过度推断。
每条长期记忆 fact 通常至少包含主体实体（如 用户/助手/speaker）和 1-4 个核心语义锚点。"""


EPISODE_SUMMARY_PROMPT_ZH = """你是长期记忆系统的 episode 摘要模块。请根据以下按时间顺序排列的对话/转写片段，只生成一个忠实、可追溯、自包含的 episode title 和 episode summary。

摘要要求：
1. title 必须简短具体，能区分同主题下的不同 episode，不要只写“健康管理”“家庭沟通”等宽泛主题。
2. summary 必须保留当前 episode 中明确出现的关键对象、参与者、动作、建议、拒绝/接受、约束、原因、结论或未解决问题。
3. summary 是 episode 级别的忠实压缩，不要提前归纳成长期偏好、画像、风险或跨 episode 观察。
4. 只使用输入证据，不要补充历史 state、外部知识或未明说的完成状态。
5. 如果对话包含多个相关信息点，用一段完整复句表达；不要写成空泛的“讨论了某主题”。
6. 忽略寒暄、礼貌收尾和无复用价值的重复内容。
7. 只返回 JSON，不要 markdown。

输出格式：
{
  "title": "简短具体标题",
  "summary": "可独立理解的 episode 摘要"
}

对话/转写片段：
{dialogue_batch}
"""


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
- 如果一个 fact 只是涉及已有 topic 的背景，但它实际讨论的是另一个对象或议题，必须在 `fact_root_topic` 中选择更准确的根主题，不能为了保持 topic 数量而强行归入已有 topic。
- 每条 fact 还要并列输出 `fact_root_topic` 和 `fact_aspect_topic`，把细粒度议题放入一个更稳定的根主题下。`fact_root_topic` 应是产品、项目或长期议题，`fact_aspect_topic` 应是当前 fact 讨论的具体方面；如果无法确认根主题，使用当前证据支持的保守根主题；如果无法进一步区分具体方面，可以让 `fact_aspect_topic` 与 `fact_root_topic` 相同，不要猜测无证据的上层主题。
- `fact_root_topic` 必须参考已有 `memory_states` 中 `state_scope=topic_state` 的 `canonical_name`；如果当前 fact 与某个已有 topic_state 的核心对象、讨论目标和语义范围一致，优先原样复用该 `canonical_name`。仅共享实体、领域或相近关键词时不要强行复用；无法严格匹配时使用当前证据支持的保守根主题。

已有长期 memory_states 参考：
{existing_memory_states}

memory_states 使用规则：
- state_scope=topic_state 且 state_type=topic 的状态，是 episode canonical_topics 和 fact `fact_root_topic` 的命名参考；如果当前证据表达的是同一长期对象或议题，优先原样复用 canonical_name。
- state_scope=entity_state 的状态，只用于理解实体的长期属性、偏好、约束、风险或关系，不能直接把 entity_state 的 canonical_name 当作 episode topic 或 fact 的 fact_root_topic。
- 这些 states 只是历史背景，不是当前 episode 的事实证据。当前对话没有明确支持的内容不能写入 fact；当前对话与历史 state 冲突时，以当前对话为准。
- fact 的 `fact_root_topic` 必须描述该 fact 的主要稳定议题；它可以复用 topic_state 的 canonical_name，但不能因为 entity_state 的名称而改变为实体属性标题。
- `fact_root_topic` 只有在该 fact 的核心对象、讨论目标和语义范围都与 topic_state 的 canonical_name 相近时，才可以原样复用该名称；`fact_aspect_topic` 应保留该 fact 在根主题下的具体讨论方面。
- 不要仅因为 fact 与某个 topic_state 同属“健康”“产品”“团队”等宽泛领域，或 fact 中出现了该 topic 的相关背景，就复用该名称。若当前证据不能支持这种严格对应，应使用当前证据中的更具体主题，必要时输出新的保守 topic。
- 不能因为 entity_state 的名称、summary 或历史 timeline 而改变 fact 的 root_topic；entity_state 只能辅助理解，不得作为 topic 候选直接复用。
- 每条 fact 必须额外输出一个 `primary_entity`，表示这条 fact 主要描述、影响或归属的唯一实体；它必须是一个实体对象，而不是数组。
- `primary_entity` 必须来自当前 fact 的 `entities`，不能因为实体只是被提及、提供建议、作为地点/工具或背景，就把它选为主要实体。多人互动时，选择该 fact 主要描述或影响的主体；如果 fact 主要描述用户自身的偏好、习惯、约束或风险，选择用户。
- `entities` 仍然保留所有与 fact 直接相关的实体，用于完整召回；后续 entity_state 匹配只使用 `primary_entity`，一条 fact 不得同时归属多个实体。

""" + ENTITY_EXTRACTION_GUIDANCE_ZH + """

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

state_aspects 输出规则：
- `fact_kind` 仍然是这条 fact 的主语义类型；`state_aspects` 是这条 fact 对 entity_state 的多个长期状态投影切片。
- 只有当该 fact 对用户或关键实体的长期状态确实有复用价值时才输出 state_aspects；普通一次性事件、临时建议、寒暄、低价值背景输出空数组。
- 每条 fact 最多输出 3 个 state_aspects，只保留最明确、最有价值的侧面。
- state_type 只能是 preference、profile、routine、relationship、constraint、risk。
- aspect_summary 必须只描述当前 state_type 的贡献点，不能复述整条 fact。比如 risk 必须说明可能影响什么或造成什么后果；constraint 说明限制条件；routine 说明反复行为/节奏；preference 说明偏好/拒绝/选择倾向。
- attribute_name 必须比 episode topic 更具体，例如“碎片化健身方式偏好”“健康管理执行限制”“减重计划持续性风险”，不要直接使用“健康管理”这类宽泛主题，除非证据只能支持宽泛属性。
- evidence_basis 必须引用当前 fact 中支持该 aspect 的具体证据，不要引入历史 state 或外部推断。

actionable_aspects 输出规则：
- `actionable_aspects` 是这条 fact 对后续 actionable_item 提取的候选投影，用于降低后续 LLM 成本和噪声。
- 只有当该 fact 明确包含未来需要提醒、跟进、执行、复盘、决策追踪、开放问题处理，或阻塞行动的高价值风险时才输出；普通偏好、背景、一次性建议、弱尝试意愿输出空数组。
- 每条 fact 最多输出 2 个 actionable_aspects，只保留最具体、最可追踪的事项线索。
- item_type 只能是 task、commitment、decision、follow_up、open_question、risk、reminder、recommendation、constraint。
- 不要把“愿意试试/可以考虑/听起来不错”提取为 actionable_aspect，除非同时有明确提醒、截止时间、具体后续检查、强承诺或可验证执行计划。
- action_summary 必须说明需要被跟进/执行/追踪的事项，不能只复述整条 fact。
- trigger_basis 必须引用当前 fact 中支持该 actionable 线索的具体证据。
- due_at 只在证据明确包含时间、截止或提醒时间时填写，否则为空字符串。

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
      "fact_root_topic": "稳定的产品/项目/长期议题根主题",
      "fact_aspect_topic": "当前 fact 讨论的具体方面",
      "fact_type": "semantic|episodic",
      "fact_subject": "user|assistant|world|project|system|other",
      "fact_kind": "preference|decision|request|recommendation|action|commitment|open_question|risk|error|context|instruction|other",
      "priority": 80,
      "occurred_start": "",
      "occurred_end": "",
      "time_confidence": "explicit|inferred_from_turn|unknown",
      "where": "",
      "state_aspects": [
        {
          "state_type": "preference|profile|routine|relationship|constraint|risk",
          "attribute_name": "具体属性名",
          "aspect_summary": "只描述该 state_type 侧面的长期状态贡献点",
          "evidence_basis": "当前 fact 中支持该 aspect 的具体证据",
          "confidence": 0.8
        }
      ],
      "actionable_aspects": [
        {
          "item_type": "task|commitment|decision|follow_up|open_question|risk|reminder|recommendation|constraint",
          "action_summary": "需要后续提醒、跟进、执行、复盘或决策追踪的具体事项",
          "owner": "user|assistant|other|unknown",
          "status": "open|in_progress|done|blocked|decided|noted|unknown",
          "due_at": "",
          "trigger_basis": "当前 fact 中支持该 actionable 线索的具体证据",
          "confidence": 0.8
        }
      ]
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
1. 给定的 canonical_topic 是根主题，只围绕这个根主题更新，不要把无关 facts 合并进来。
2. summary 需要体现根主题的长期状态：背景、最近变化、关键参与者、已经形成的决定/偏好/约束、仍未解决的问题和下一步。
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

输入包含新存储的候选 narrative facts，部分 fact 会带有 actionable_aspects 作为前置筛选出的行动线索。请从中提取未来真正需要跟进、提醒、执行、复盘或决策追踪的具体可执行事项。actionable item 和 evolving state 分开：state 描述持续变化的长期状态、偏好、约束和背景；actionable item 必须是可以被检查、完成、追踪，或明确作为决定/承诺/风险/开放问题被召回的事项。

优先参考 actionable_aspects，但最终只能提取 fact summary 和 actionable_aspects 直接支持的内容。

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

候选 facts：
{facts}
"""


RECALL_QUERY_ANALYSIS_PROMPT_ZH = """你是 AI 眼镜长期记忆系统中的 recall query 分析器。

请先理解当前记忆结构，再分析用户查询应该优先检索哪些记忆层。当前默认 recall 路径不使用共享的 memory_index_entries 表作为检索入口，而是直接检索以下三类原始记忆表；因此不要把“index”理解为一个额外的统一文档层。

记忆结构：
1. `memory_facts` / fact：从一次 episode 的对话或全天候转写中提炼出的、可追溯且自包含的 narrative fact。它保留具体发生了什么、谁参与、时间、地点/场景、原因、观点变化、建议、接受/拒绝、约束、结论和未解决问题等证据。fact 可能是一次事件、一次讨论结论，也可能是用户明确表达的偏好、习惯、画像、风险或约束；但它仍然是当前对话证据，不等于跨多次对话融合后的长期状态。fact 通常带有 `fact_type`、`fact_kind`、`fact_subject`、`summary`、`keywords`、`entities`、`fact_root_topic`、`fact_aspect_topic` 和 `time_key`。
2. `memory_states` / state：由多个 facts 反思更新出的长期演化状态，不是原始对话引用。它包含两类投影：
   - `topic_state`：某个项目、产品、主题或长期议题的根状态，保存整体背景、进展、决定、约束、风险和未解决问题；细粒度的 aspect（例如“直播平台选择”“赠品方案”）作为根状态的上下文和检索别名，不一定单独形成 state。
   - `entity_state`：某个实体的长期属性，包括 preference（偏好）、routine（习惯/流程）、profile（画像/背景）、relationship（关系）、constraint（约束）和 risk（风险）。
   state 适合回答“长期是什么状态、通常怎样、对某人/某项目的稳定认识是什么”，但不能替代具体 fact 证据。
3. `memory_actionable_items` / actionable_item：从 facts 中提炼出的需要未来执行、跟进、提醒、复盘或决策追踪的事项。包括 task、commitment、decision、follow_up、open_question、risk、reminder、recommendation 和被明确行动阻塞的 constraint。每个 item 通常带有 `canonical_name`、`summary`、`owner`、`status`、`due_at` 和 `evidence_fact_ids`。普通偏好、背景、一次性描述或没有明确后续动作的建议不属于 actionable_item。

episode 是原始对话/转写批次的存储容器，包含 title、summary、参与者和时间范围；当前默认 recall 不把 episode 作为独立可选择的检索层。需要回顾一段经历时，优先选择 `fact`；需要长期概括时，同时考虑 `state`。states 和 actionable_items 都可以通过 `evidence_fact_ids` 追溯到 facts。

判断准则：
- 只有当 query 明确指向用户与助手的主动对话，才偏向 `assistant_wakeup`；明确指向全天录音、会议、旁听、多人数对话，才偏向 `allday_recording`；不确定时两者都保留。
- 具体发生了什么、日期、地点、人名、原话语义、事件先后和可追溯证据，优先 `fact`。
- 稳定偏好、长期约束、习惯/流程、关系画像、个人背景，以及主题/项目/议题的长期演化，优先 `state`。
- 任务、承诺、决定、开放问题、风险、提醒、推荐和明确下一步，优先 `actionable_item`；如果用户同时询问事项的背景或来源，可以同时选择 `fact`。
- 查询涉及“目前进展、长期状态和下一步”时，通常同时选择 `state`、`actionable_item` 和 `fact`。
- 不确定时保持宽检索，漏掉证据比多取几个候选更糟，但不要无差别默认选择所有层。
- `layer_preference` 输出 1-3 个最相关的层，值只能是 `fact`、`state`、`actionable_item`；它表示需要优先加强的召回层，不是新的数据库表。
- `keywords` 输出 2-8 个短检索词，优先保留具体人物、组织、产品、项目、主题、动作、结果、约束和时间锚点；不要输出完整句子、寒暄或泛化词。
- `entities` 输出对语义检索有帮助的实体名称及类型。实体可以是人物、组织、地点、产品、项目、技术或具体概念；普通的“今天/昨天/上周”等时间表达不要作为实体。

只返回 JSON：
{
  "source_types": ["assistant_wakeup", "allday_recording"],
  "layer_preference": ["fact", "actionable_item", "state"],
  "needs_broad_evidence": false,
  "query_rewrite": "面向原始记忆表检索的改写",
  "keywords": ["关键词1", "关键词2"],
  "entities": [{"name": "实体名", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|OTHER"}]
}

用户查询：
{query}
"""
