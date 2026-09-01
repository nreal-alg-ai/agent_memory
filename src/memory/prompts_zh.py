"""Chinese prompt templates for the unified memory prototype."""

MEMORY_RETRIEVED_FORMAT_PROMPT_ZH = """[统一记忆]
系统说明：记忆按语义角色分组。state 和 actionable item 提供紧凑摘要；fact 提供可追溯证据。
时间字段说明：对于 fact，dialogue_time 表示对话/转写讨论该 fact 的时间；event_time 表示 fact 描述的现实事件发生时间。二者含义不同；event_time 未知时，不要用 dialogue_time 推断它。
{memory_sections}"""

MEMORY_RETRIEVED_SECTION_SPECS_ZH = (
    (
        "[检索事实]",
        "这些是直接从 memory_facts 检索出的、按相关性排序的叙事事实。",
        "fact",
    ),
    (
        "[长期状态]",
        "这些是根据 memory facts 反思得到的演化状态，应作为摘要上下文理解，不是用户的直接原话。",
        "state",
    ),
    (
        "[行动事项]",
        "这些是可能需要后续跟进的决定、任务、承诺、风险或开放问题。",
        "actionable_item",
    ),
)

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


EPISODE_SUMMARY_PROMPT_ZH = """你是长期记忆系统的 episode 聚合模块。输入是同一个连续时间区间内，已经由 fact extraction 模块提炼完成、按时间顺序排列的 narrative facts。你的任务不是重新提炼 facts，而是把这些 facts 组织成一个更高层的、可独立理解的事件摘要。

概念边界：
- fact 是可独立召回的证据单元，保留一个具体议题或事件的细节。
- episode 是一段连续交互/转写中多个相关 facts 的上层叙事，应体现这段经历整体发生了什么。
- episode 不是长期 state、用户画像、长期偏好、风险评估或 actionable item；不要把一次 episode 推广成跨 episode 结论。

聚合步骤：
1. 先判断 facts 是否属于同一个连续事件或共同议题。对同一对象、同一目标、前后回应或因果链上的 facts 进行合并；不要按 fact 数量机械拼接。
2. 按时间恢复事件进展：背景/问题 → 讨论或方案 → 用户态度（接受、拒绝、犹豫）→ 约束与理由 → 决定、结果或未解决事项。输入没有支持的环节直接省略。
3. 对互不相关但确实处于同一连续 episode 的 facts，使用一段有层次的概括串联它们；不要制造事实之间不存在的因果关系，也不要为了追求单一主题而删除高价值事实。
4. 保留能改变未来回答的具体信息：对象、人、地点/场景、时间锚、数量、方案、选择、拒绝、限制、承诺、结果和开放问题。普通寒暄、重复表达、泛化解释和礼貌收尾应忽略。
5. 保留事实之间的先后、转折和条件关系。不能把“建议但未接受”写成“已决定”，不能把“计划/可能”写成“已完成”，不能把“待确认”写成“已确认”。
6. 如果 facts 之间存在观点冲突或状态变化，明确写出变化过程或当前结论；不要擅自裁决冲突，也不要用较新的 fact 覆盖仍有价值的早期背景。
7. summary 必须是 1 段自包含叙事，读者不看 facts 也能理解 episode 的核心对象、发生过程和结论/未决点；不要只写“讨论了某主题”。
8. title 是该 episode 的检索标题：简短、具体、可区分同主题的其他 episode，优先使用“对象 + 核心事件/决定/问题”，不要只写“健康管理”“家庭沟通”“项目讨论”等泛主题。
9. canonical_topics 只输出 1-3 个稳定主题。优先从 facts 的 `fact_root_topic` 中归并复用；可以合并同义方面，但不要把 `fact_aspect_topic`、动作、结论或零散关键词当作主题。没有充分证据时宁可少输出，不要编造上位主题。
10. 只使用输入 facts 作为证据，不要补充原始对话、历史 state、外部知识、推测的责任人/截止时间或未明说的完成状态。
11. 只返回符合格式的 JSON，不要 markdown、解释文字或额外字段。

输出格式：
{
  "title": "简短具体标题",
  "summary": "一段自包含、按时间和因果组织的 episode 摘要",
  "canonical_topics": ["稳定主题1", "稳定主题2"]
}

已提炼 facts：
{facts}
"""


UNIFIED_MEMORY_EXTRACTION_PROMPT_ZH = """你是 AI 眼镜长期记忆系统的记忆提炼模块。

当前记忆结构：
- fact：从一批连续证据中提炼出的可追溯、自包含、可独立召回的 narrative fact，是记忆提炼的基本证据单元。
- episode：由独立模块根据一段连续时间内新增的、已经生成的 facts 汇总出的更高层事件；它不是每个输入批次或每条 fact 的简单副本。
- state：reflection 根据 facts 形成的演化投影，包括 `topic_state`（长期主题进展）和 `entity_state`（实体的偏好、画像、习惯、关系、约束或风险）。state 不是当前对话的直接证据。
- actionable_item：从 facts 中筛选出的可能需要后续跟进的任务、承诺、决定、开放问题或风险。

你现在需要从下面按时间顺序排列的对话/转写证据批次中提取 Hindsight 风格的高质量 narrative facts。episode summary、episode canonical_topics 将由独立模块根据已生成 facts 负责，不要在本 prompt 中输出 episode 级字段。

memory_states 使用规则：
- state_scope=topic_state 且 state_type=topic 的状态，只作为 fact `fact_root_topic` 的命名参考；如果当前证据表达的是同一长期对象或议题，优先原样复用 canonical_name。
- state_scope=entity_state 的状态，只用于理解实体的长期属性、偏好、约束、风险或关系，不能直接把 entity_state 的 canonical_name 当作 episode topic 或 fact 的 fact_root_topic。
- 这些 states 只是历史背景，不是当前 episode 的事实证据。当前对话没有明确支持的内容不能写入 fact；当前对话与历史 state 冲突时，以当前对话为准。
- fact 的 `fact_root_topic` 必须描述该 fact 的主要稳定议题；它可以复用 topic_state 的 canonical_name，但不能因为 entity_state 的名称而改变为实体属性标题。
- `fact_root_topic` 只有在该 fact 的核心对象、讨论目标和语义范围都与 topic_state 的 canonical_name 相近时，才可以原样复用该名称；`fact_aspect_topic` 应保留该 fact 在根主题下的具体讨论方面。
- 不要仅因为 fact 与某个 topic_state 同属“健康”“产品”“团队”等宽泛领域，或 fact 中出现了该 topic 的相关背景，就复用该名称。若当前证据不能支持这种严格对应，应使用当前证据中的更具体主题，必要时输出新的保守 topic。
- 不能因为 entity_state 的名称、summary 或历史 timeline 而改变 fact 的 root_topic；entity_state 只能辅助理解，不得作为 topic 候选直接复用。
- `entities` 保留所有与 fact 直接相关的实体，用于完整召回；`primary_entity` 表示这条 fact 主要描述、影响或归属的单一实体，必须来自 `entities`。对于用户自己的偏好、习惯、约束或风险，优先将“用户”作为 primary_entity；对于助手自己的动作或建议，优先将“助手”作为 primary_entity。

""" + ENTITY_EXTRACTION_GUIDANCE_ZH + """

Hindsight 风格 narrative fact 的核心要求：
- 每条 fact 应覆盖一次完整 exchange 或一个清晰议题片段，而不是单个 utterance。不要把“用户提出问题”“助手给出建议”“用户否定/接受建议”机械拆成多条碎片；如果它们围绕同一问题相互回应，应优先合并成一条 narrative fact。
- 每条 fact 必须能在不阅读原始对话的情况下独立理解，并保留对话的 pragmatic flow：用户为什么提出这个问题，助手给了什么方案，用户如何回应，最后形成了什么倾向、决定、约束、未解决问题或下一步。
- 每条 fact 应在 text 中优先体现 what（完整事件/议题/方案/结论）；when、where、who、why 只有在输入证据明确出现且有助于理解时才加入。缺失的信息直接省略，不要写“未提及具体地点/场景”“没有说明原因”等无信息量的占位句。
- 压缩解释过程，不压缩事实答案；删除无关细节，但不要删除理解事实所需的主体、对象、时间、关键动作、用户态度、结果、决定或约束。
- 对一个 5 轮左右的对话批次或一段多人转写片段，通常输出 1-3 条 facts；只有当批次中确实存在多个互不相关的事件/议题时才拆开。绝大多数情况下不要超过 5 条。

fact_type 判别规则：
- `semantic` 表示不依赖某一次具体经历也能复用的稳定知识或长期信息，例如项目结构、概念定义、系统约定、常识、用户长期偏好、长期指令或长期约束。它描述“通常是什么/长期怎样”，重点是跨多次对话仍成立的稳定认识。
- `episodic` 表示某次具体发生过的经历或事件，例如用户在某轮提出请求、助手执行修改或测试、一次失败或通过、某个时间点的决定、状态变化或情绪反应。它描述“某次发生了什么”，即使事件涉及一个长期项目，也仍然可以是 episodic。
- 判断核心是该 fact 是否依赖一次具体经历才能成立，而不是主题是否长期存在、内容是否重要，或是否可能影响未来。一次性的请求、建议、修改、测试结果、决定或风险事件默认标为 `episodic`；只有证据明确支持跨场景、跨时间可复用的稳定知识或长期模式时才标为 `semantic`。
- 不要因为 fact 使用了“偏好”“风险”“决定”等 fact_kind 就自动标为 `semantic`：一次具体场景中的偏好表达、临时风险、单次决定仍应标为 `episodic`；反复出现或明确声明长期有效的偏好、约束、指令才可以标为 `semantic`。

时间保真要求：
- 必须保留影响语义的顺序词和先后关系：first、first time、second、previous、next、later、earlier、before、after、once、again、subsequent、prior、last、most recent，以及“第一次/首次/第二次/之前/之后/此前/随后/后来/更早/最近一次/上一次”等。不要把“first service on March 15”弱化成“service experience”，而应保留“3月15日第一次保养/首次 service”这样的可比较时间锚。
- 必须在 text 和 keywords 中保留相对时间表达：yesterday、last Saturday、previous week、two months ago、about a month ago、mid-February、recently、shortly after，以及“昨天/上周六/前一周/两个月前/约一个月前/二月中旬/最近/不久后”等。如果能根据 Conversation timestamp 无歧义换算，直接将解析后的实际事件时间写入 `event_time_key`。
- 每个片段中的 `Time` 是该片段发生的对话/转写时间，只能作为推导相对事件时间的参考锚点，不是 fact 的默认事件时间。必须结合 fact 所描述的具体事件和原文时间表达，单独推导 `event_time_key`；不要因为 fact 在某个时间被讨论，就把该对话时间直接复制为事件时间。
- `event_time_key` 表示 fact 所描述事件最有代表性的现实发生时间或时间锚点，不表示对话时间、LLM 提炼时间或当前系统时间。它是单一时间字段，不要输出事件结束时间或额外的起止时间字段。
- 时间推导优先级为：原文明确的绝对日期/时间 > 结合片段 `Time` 可以无歧义换算的相对时间 > 明确表示事件就在当前对话中发生的时间。比如对话时间为 2023-05-30，`last month (around April 2023)` 的 `event_time_key` 应为 2023-04 附近的代表性日期，而不是 2023-05-30；`last weekend (May 27-28)` 应使用 2023-05-27 附近的代表性日期；只有“今天决定/刚刚完成”这类明确发生在当前对话中的事件，才使用 2023-05-30。
- 如果 fact 同时描述当前对话行为和更早发生的背景事件，应以该 fact 主要描述的事件为准；必要时拆成多条 facts，不能用对话时间覆盖更早事件。若只能确认月份、周末或相对时间范围，保留原始时间表达在 text/keywords 中，并在 `event_time_key` 中填写保守的代表性时间锚点。
- 优先输出证据中明确给出的日期、时间、星期或相对时间。相对时间只有在结合当前片段的 `Time` 可以无歧义换算时才转换为绝对时间；无法判断时不要猜测，`event_time_key` 留空并将 `time_confidence` 设为 `unknown`。不要用当前对话时间、当前系统时间或 LLM 提炼时间补造事件时间。
- 如果同一 fact 包含多个时间不同的事件，按语义拆分 facts，避免用一个事件时间掩盖互不相关的事件。
- 如果未来问题的答案依赖事件先后、间隔、第一次/上一次或“哪个更早”，fact text 必须同时包含事件对象和时间锚/顺序词，不能只存主题名。
- 如果同一批证据中多个事件可能被未来问题比较先后，应在一条 narrative fact 中明确写出相对顺序，或拆成多条各自带完整背景和时间锚的 facts；不要只保留比较中的一方。
- 带时间锚或顺序词的个人经历即使只是顺带提到，也应认真保留，例如购买、保养/维修、修理、预约、参加活动、旅行、会议、测试、失败、决定等。

提取规则：
1. 提取 0-5 条 facts，不要为了覆盖每一轮强行生成 fact。
2. 每条 fact 必须是一段完整叙事，至少包含“议题背景 + 用户/助手的观点或动作流动 + 理由/分歧/约束/结论/下一步”中的关键要素。
3. 保留可被直接问到的具体细节：人名、地点、标题、颜色、日期、星期、相对时间、数量、金额、时长、产品、机构、建议、约束、决定和用户偏好。
4. 压缩助手的解释、推导和泛化建议，但保留未来可能直接成为答案的事实细节，以及用户明确接受、拒绝、选择或形成的决定。
5. 不要丢弃 “by the way / I also / I just / last Saturday / two months ago / 顺便 / 我还” 这类附带提到的个人事件；但如果它们属于同一 exchange 的上下文，应合并进同一条 narrative fact，而不是拆成无背景短句。
6. 只有真正互不相关的事件才拆开；时间推理需要比较先后/间隔的事件可以拆成多条，但每条仍必须保留完整背景和时间锚点。
7. 只使用输入证据，不要编造完成状态、意图或原因。
8. 如果助手的推荐中包含未来可能被问到的具体条目，要放入相关 exchange 的 narrative fact，并写清用户是否接受、拒绝、犹豫或提出约束。
9. priority 为 0-100，只保留至少 60 分的事实。
10. fact_type 只能是 semantic 或 episodic，并严格按照上面的稳定知识/长期信息与单次事件边界判断。
11. fact_kind 只能是 preference、decision、request、recommendation、action、commitment、open_question、risk、error、context、instruction、other。
12. 不要输出只有“用户说了 X”“助手建议 Y”的短 fact；如果删除议题背景、理由、分歧或结论后会变成泛泛短句，必须补回这些信息；若对话没有足够信息支撑，则不要输出该 fact。
13. 不要把助手的寒暄、礼貌收尾、泛化鼓励或无具体信息的回复单独作为 fact，例如“希望这个方法能帮到您”“有其他问题可以继续沟通”“好的”“不客气”等；除非它明确改变了用户决定、承诺或下一步。
14. keywords 只能包含用于检索的短实体、主题、症状、方案、约束、决定和关键时间/顺序锚，通常每个关键词 2-8 个汉字或一个短英文短语；对带时间锚的事件，必须加入原始或补全后的时间词，例如“March 15 2023”“first service”“3/22”“last Saturday”“two months ago”“上周六”“两个月前”。不要把完整句子、寒暄、礼貌话、语气词、泛化表达或“希望这个方法能帮到您”这类文本放入 keywords。
15. 只返回 JSON，不要 markdown。

entity_state_signal 输出规则：
- `entity_state_signal` 只是提示当前 fact 可能对用户或关键实体的长期状态有贡献，不是最终的 entity_state 更新。
- 只有证据明确表达偏好、画像、习惯、关系、约束或风险，且具有跨场景复用价值时才输出；普通一次性事件、临时建议、寒暄和低价值背景输出空数组。
- 每条 fact 最多输出 3 个 signal。每个 signal 只保留 `state_type`、`attribute_name`、`evidence_basis`、`confidence`，以及证据明确时的 `entity`；不要生成 aspect_summary，不要引用历史 state。
- `state_type` 只能是 preference、profile、routine、relationship、constraint、risk。
- `attribute_name` 应具体描述可能受影响的属性；`evidence_basis` 必须引用当前 fact 中的证据。后续 reflection 会结合已有 entity_state 决定是否创建、更新或忽略该信号。

action_signal 输出规则：
- `action_signal` 只是候选线索，不是最终的 actionable_item。它允许有少量误报，后续 actionable 提取模块会重新核验。
- 只有当前 fact 明显涉及未来行动、执行承诺、未完成决定或明确提醒/跟进时才输出；没有未来导向时输出空数组。不要根据上下文推断责任人、截止时间、完成状态或最终结论。
- `action_strength` 只能是 `assigned`、`committed`、`pending_decision` 或 `follow_up`，表示粗粒度候选类型，不代表最终判断；`item_type=decision` 仅可搭配 `pending_decision`。
- 每条 fact 最多输出 1 个 signal。每个 signal 只保留 `item_type`、`action_strength`、`evidence_basis` 和 `confidence`，可选填写证据明确的 `due_at`；不要生成 action_summary、owner 或 status。
- `evidence_basis` 必须来自当前 fact 的直接证据，不要拼接多个事实或补充未出现的责任人、时间和结论。
- `item_type` 只能是 task、commitment、decision、follow_up、open_question、risk、reminder、recommendation、constraint。

输出格式：
{
  "facts": [
    {
      "text": "覆盖完整 exchange 的自包含 narrative fact；优先写清 what，when/where/who/why 仅在证据明确且有助于理解时写入，缺失信息直接省略",
      "keywords": ["关键词1", "关键词2"],
      "entities": [{"name": "实体名", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|TOPIC|PREFERENCE|OTHER"}],
      "primary_entity": {"name": "这条 fact 主要描述、影响或归属的单一实体", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|TOPIC|PREFERENCE|OTHER"},
      "fact_root_topic": "稳定的产品/项目/长期议题根主题",
      "fact_aspect_topic": "当前 fact 讨论的具体方面",
      "fact_type": "semantic|episodic；semantic=可跨多次对话复用的稳定知识或长期信息，episodic=依赖某次具体经历的事件或状态变化",
      "fact_kind": "preference|decision|request|recommendation|action|commitment|open_question|risk|error|context|instruction|other",
      "priority": 80,
      "event_time_key": "根据对话时间锚点和 fact 内容推导出的事件实际发生时间或代表性时间锚点；无法判断时为空字符串",
      "time_confidence": "explicit|inferred_from_turn|unknown；分别表示原文明确给出、结合当前片段 Time 和相对表达推断、无法判断",
      "where": "明确出现的地点、场景、平台或项目范围；没有明确证据时保持为空字符串，不要填写‘未提及’或类似说明",
      "entity_state_signal": [
        {
          "state_type": "preference|profile|routine|relationship|constraint|risk",
          "attribute_name": "具体属性名",
          "entity": {"name": "明确受影响的实体", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|TOPIC|PREFERENCE|OTHER"},
          "evidence_basis": "当前 fact 中支持该 signal 的具体证据",
          "confidence": 0.8
        }
      ],
      "action_signal": [
        {
          "item_type": "task|commitment|decision|follow_up|open_question|risk|reminder|recommendation|constraint",
          "action_strength": "assigned|committed|pending_decision|follow_up",
          "due_at": "",
          "evidence_basis": "当前 fact 中支持该 signal 的具体证据",
          "confidence": 0.8
        }
      ]
    }
  ]
}

已有长期 memory_states 参考：
{existing_memory_states}

对话/转写证据批次：
{dialogue_batch}
"""

UNIFIED_TOPIC_STATE_UPDATE_PROMPT_ZH = """你是 AI 眼镜长期记忆系统中的 topic_state 更新模块。

输入已经完成主题解析：系统已经判断当前候选证据应归入某个长期根主题，或应创建一个新的长期根主题。你的任务是基于 candidate_topic_state 和已有 topic_state 更新根 topic_state 的 summary，而不是重新决定主题归属。

规则：
1. `candidate_topic_state.root_topic_name` 是待更新的根主题，只围绕它更新，不要把其他主题或仅仅共享实体的内容合并进来。
2. `aspect_topics` 是该根主题下的具体方面，只用于理解局部进展和组织状态，不要为每个 aspect 创建独立 topic_state。
3. `parent_topics` 是 episode 或上层语境提供的辅助主题，只能作为背景参考；如果与根主题无关，不要写入 summary。
4. `identity_text`、`keywords`、`context_entities` 和 `fact_summaries` 是候选主题的聚合检索与证据信息。优先使用其中具体、重复或已形成结论的内容，但不要机械复制 identity_text，也不要把所有关键词和实体堆进 summary。
5. `fact_ids` 是可引用的证据 ID。只能引用 candidate 中确实支撑本次变化的 ID，不要根据 ID 猜测事实内容。
6. summary 需要体现根主题的长期状态：背景、最近变化、关键参与者、已经形成且仍有效的决定/偏好/约束、仍未解决的问题和下一步。
7. 如果 existing_topic_state 已有内容，要增量融合，不要简单拼接，不要丢失仍然有效的长期信息。
8. 不要把单个 fact 改写成另一句 fact；topic_state 必须比单个事实更抽象、更稳定。
9. summary 必须是简短的当前状态快照，最多 1-2 句话，建议不超过 120 个中文字符；不要把历史 timeline 拼接进 summary。
10. time_line_updates 只记录 candidate_topic_state 中本次证据带来的状态变化，输出 0-3 条；每条包含发生时间、变化类型、简短变化说明和 fact_ids。不要重复已有 timeline，也不要把没有变化的内容写入 timeline。
11. evidence_fact_ids 必须来自 candidate_topic_state.fact_ids，并且只能引用真正支撑本次更新的 fact ID。
12. 只返回 JSON，不要 markdown。

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

candidate_topic_state：
{candidate_topic_state}

candidate_topic_state 字段说明与使用方式：
- `root_topic_name`：候选的稳定根主题名称，是本次 topic_state 的核心归属。不要把 aspect 或无关 episode 主题改写成新的根主题。
- `topic_key`：根主题的稳定内部键，只用于确认主题身份，不要写入 summary、canonical_name 或 timeline。
- `identity_text`：由根主题、方面、关键词、实体和事实摘要构成的候选身份文本，用于整体理解和区分主题；不要原样复制为 state summary。
- `aspect_topics`：当前候选 facts 涉及的具体方面，例如平台选择、预算限制或方案进展。它们是根主题下的上下文，不是必须单独创建的 state。
- `parent_topics`：候选 facts 所在 episode 的上层主题，只在能帮助理解根主题时使用，不能覆盖更准确的 root_topic_name。
- `keywords`：从相关 facts 聚合出的检索关键词，用于识别具体对象、动作、结果和约束；不要把泛化词逐个列入 summary。
- `context_entities`：相关 facts 中的语义实体，用于确认参与者、产品、项目或对象；只有对根主题状态有意义时才写入 summary。
- `fact_summaries`：相关 facts 的压缩摘要，是本次候选状态变化的主要证据内容。只提炼其中与根主题直接相关且有长期价值的部分。
- `fact_ids`：上述摘要对应的 fact 标识，只用于 evidence_fact_ids 和 time_line_updates[].fact_ids 的证据引用。
- `source_type`：候选来源类型，仅作为来源背景，不要把它当作主题内容。

已有 topic_state：
{existing_topic_state}
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

entity_state_target 字段说明与使用方式：
- `entity`：本次更新所描述的实体名称。所有 summary、canonical_name 和 time_line_updates 都必须围绕这个实体展开，不要把其他被提及但不是主要对象的实体写成当前状态的归属者。
- `entity_key`：该实体的稳定内部标识，用于确认实体身份。它不是自然语言内容，不要把它写入 summary、canonical_name 或 timeline；只需用它确认本次 candidate 与已有 entity_state 是否属于同一个实体。
- `state_type`：本次允许更新的 entity_state 类型，只能围绕这个类型提炼信息。不要因为候选内容同时涉及其他方面，就擅自改成 preference、profile、routine、relationship、constraint 或 risk 中的另一类。
- `attribute_name`：本次候选状态的具体属性名称，是更新的主要语义边界。summary 应该说明这个属性对该实体的长期含义，而不是泛泛总结整段对话或 topic_state 的进展。
- `attribute_key`：属性的稳定内部键，用于辅助确认属性身份。它不是需要展示给用户的内容，不要直接复制到 summary 或 canonical_name。
- `attribute_name_aliases`：该属性可能出现的同义名称或历史名称。判断已有 entity_state 是否描述同一属性时可以参考这些别名，但不要把所有别名机械拼接进输出；如果已有状态表达的是不同属性，应返回 `update_needed=false`。
- `state_signal_evidence`：当前候选 facts 中支持该 entity_state 信号的证据。信号不是最终状态结论；请结合完整 fact summary 和 existing_entity_state 判断是否真的需要更新。
- `state_signal_evidence[].fact_id`：支持该信号的 fact 标识，只用于在输出的 `evidence_fact_ids` 和 `time_line_updates[].fact_ids` 中引用。
- `state_signal_evidence[].confidence`：信号提炼置信度，用于保守判断；低置信度或证据不足时不要扩展出新的长期结论。

处理原则：先使用 `entity` 和 `entity_key` 确认状态归属，再使用 `state_type`、`attribute_name` 和属性别名确认更新边界，最后综合当前 facts、`state_signal_evidence` 与 existing_entity_state 生成当前状态。只有当证据对该实体属性确实具有长期价值时才更新；一次性事件、单次建议、临时请求或仅属于 topic_state 的进展都应返回 `update_needed=false`。如果 existing_entity_state 与 candidate 是同一实体和同一属性，则在保留已有长期结论的基础上增量融合；如果属性不同，不要强行合并。

已有 entity_state：
{existing_entity_state}
"""


UNIFIED_ACTIONABLE_ITEM_EXTRACTION_PROMPT_ZH = """你是 AI 眼镜长期记忆系统中的 actionable item 提取模块。

输入包含一个 topic candidate 及其新存储的 narrative facts，部分 fact 会带有 action_signal 作为前置筛选出的行动线索。请在这个 topic 范围内提取未来真正需要跟进、提醒、执行、复盘或决策追踪的具体可执行事项。actionable item 和 evolving state 分开：state 描述持续变化的长期状态、偏好、约束和背景；actionable item 必须是可以被检查、完成、追踪，或明确作为决定/承诺/风险/开放问题被召回的事项。

`action_signal` 只是候选线索，可能包含误报，不能直接转换为 actionable_item。必须重新核验完整 fact summary 和 signal 的直接证据；无法确认时输出空列表。

规则：
1. 提取 0-4 条 actionable items。多数普通对话可以输出空列表，不要为了覆盖 facts 强行生成。
2. 每个 item 都必须有明确的 fact 证据和可核验结果；action_signal 是必要的候选线索，但不是充分条件。建议、计划、考虑、倾向、讨论或泛化的“后续探讨”不能单独生成 item。
3. `pending_decision` 必须同时满足：决策对象明确；当前仍未决；存在具体选项、确认/商定动作或截止时间。缺少任意一项则不要生成 decision/open_question。
4. “已经决定 X，但执行细节/落实进度尚未确定”不属于 `pending_decision`；只有明确责任人、动作和交付物时，才可生成 task/commitment。
5. `assigned` 必须有明确责任主体、具体动作和交付物或完成标准；`committed` 必须是明确承诺执行；`follow_up` 必须是明确提醒、跟进、复盘、下次确认或汇报。仅“有人提出/建议/需要/各部门回去落实”不够。
6. 已经确定且没有剩余动作的决定、普通约束、偏好和背景信息属于 fact/state，不要作为 actionable item。弱尝试意愿和助手普通建议也默认输出空列表。
7. 不要复制每条 fact。若多条 facts 指向同一件事，只保留一条最具体、最可追踪的 item。
8. 以下情况必须输出空列表：
   - “团队决定同时上线多种赠品形式，但执行细节尚未确定。”
   - “建议多方面了解后进一步探讨。”
   - “有人提出需要与银行谈判争取较低利率。”
   以下情况可以生成 `pending_decision`：
   - “尚未决定选择抖音还是小红书作为主促销平台。”
   - “团队存在分歧，需要在下周二前确认产品颜色。”
9. `evidence_fact_ids` 必须引用输入中的 fact ID；每个 item 应能对应到一个直接支持它的 fact signal。
10. 每个 item 必须可独立理解：`summary` 必须明确写出“谁负责 + 在什么时间/截止条件前（证据有才写）+ 做什么/针对什么对象”，并包含必要上下文、原因和当前状态。没有明确时间时不要编造日期；责任主体无法判断时可以使用“责任主体未明确”，但仍需写清待执行的动作。
11. `canonical_name` 是用于识别、检索和合并同一待办的稳定短标题，不是完整句子或详细 summary。应包含最核心的动作/事项和目标对象，例如“提交报销材料”“跟进直播平台选择”；不要加入 owner、状态、截止日期、原因、多个无关事实或泛化标题。更新已有 item 时，如果仍是同一件事，必须复用已有的 `canonical_name`，只有事项本身发生变化时才修改。
12. `owner` 表示这条待办属于谁、由谁负责完成、跟进或作出决定，不是待办涉及的对象，也不是所有被提及的实体。依据 supporting facts 中的明确指派关系判断；`primary_entity` 只能在它同时明确表示执行或决策责任时作为辅助依据。更新已有 item 时优先保持已有 item 的 owner，只有新 fact 明确显示责任归属发生变化时才修改。无法判断时使用“未知”。
13. owner 必须是证据支持的实际责任主体名称，可以是“用户”“助手”“N_SPK8013”“小王”“团队”或“市场部”等；不要因为某人发言、被提及或提出要求就自动把待办归给该人，也不要输出固定类别“其他”。
14. evidence_fact_ids 必须引用输入中的 fact ID。
15. item_type 只能是：task、commitment、decision、follow_up、open_question、risk、reminder、recommendation、constraint、other。
16. status 只能是：open、in_progress、done、blocked、decided、noted、unknown。
17. importance 和 confidence 都是 0-1。弱尝试意愿的 confidence 不应提高来绕过规则。
18. 如果已有 actionable_items 与当前 topic 相关，优先判断新 facts 是否明确改变其状态、截止时间、责任人或当前描述。只有新 facts 明确支持变化时才输出 `operation="update"`；没有新证据表明完成时，不要擅自标记为 done。
19. 更新已有 item 时必须使用已有 item 的 `id`，并保持其 `canonical_name` 和 `item_type` 稳定；新 item 使用 `operation="create"` 和 `existing_item_id=0`。
20. 只返回 JSON，不要 markdown。

输出格式：
{
  "actionable_items": [
    {
      "operation": "create|update",
      "existing_item_id": 0,
      "item_type": "task|commitment|decision|follow_up|open_question|risk|reminder|recommendation|constraint|other",
      "canonical_name": "用于识别和合并待办的稳定短标题，例如：提交报销材料",
      "summary": "自包含可执行事项，明确责任主体、时间/截止条件（如有）和具体动作",
      "owner": "具体责任主体名称，例如：用户、小王、项目经理；无法判断时：未知",
      "status": "open|in_progress|done|blocked|decided|noted|unknown",
      "due_at": "",
      "evidence_fact_ids": [1, 2],
      "keywords": ["关键词1", "关键词2"],
      "canonical_topics": ["主题1"],
      "importance": 0.8,
      "confidence": 0.85
    }
  ]
}

topic_candidate（包含 topic 元信息和详细 supporting facts）：
{topic_candidate}

已有 topic_state（如果存在）：
{existing_topic_state}

当前 topic 已有的 actionable_items：
{existing_actionable_items}

"""


RECALL_QUERY_ANALYSIS_PROMPT_ZH = """你是 AI 眼镜长期记忆系统中的 recall query 分析器。

请先理解当前记忆结构，再分析用户查询应该优先检索哪些记忆层。

记忆结构：
1. `memory_facts` / fact：从一次 episode 的对话或全天候转写中提炼出的、可追溯且自包含的 narrative fact。它保留具体发生了什么、谁参与、时间、地点/场景、原因、观点变化、建议、接受/拒绝、约束、结论和未解决问题等证据。fact 可能是一次事件、一次讨论结论，也可能是用户明确表达的偏好、习惯、画像、风险或约束；但它仍然是当前对话证据，不等于跨多次对话融合后的长期状态。fact 通常带有 `fact_type`、`fact_kind`、`primary_entity`、`summary`、`keywords`、`entities`、`fact_root_topic`、`fact_aspect_topic`、`event_time_key` 和 `dialogue_time_key`。
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
- `temporal_mode` 表示时间范围应该匹配哪一种 fact 时间：`event_time` 表示事实描述的现实事件时间，`dialogue_time` 表示对话/转写发生时间，`both` 表示任一时间命中即可，`none` 表示不做时间硬过滤。询问“做了什么/发生了什么/买过什么”优先使用 `event_time`；询问“讨论了什么/提到过什么/问过什么”优先使用 `dialogue_time`；无法判断时使用 `none`。

只返回 JSON：
{
  "source_types": ["assistant_wakeup", "allday_recording"],
  "layer_preference": ["fact", "actionable_item", "state"],
  "needs_broad_evidence": false,
  "query_rewrite": "面向原始记忆表检索的改写",
  "keywords": ["关键词1", "关键词2"],
  "entities": [{"name": "实体名", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|OTHER"}],
  "temporal_mode": "event_time|dialogue_time|both|none"
}

用户查询：
{query}
"""
