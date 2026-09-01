"""English prompt templates for the unified memory prototype."""

MEMORY_RETRIEVED_FORMAT_PROMPT_EN = """[Unified Memory]
System note: Memories are grouped by semantic role. States and actionable items provide compact summaries; facts provide traceable evidence.
System note: For facts, dialogue_time is when the conversation/transcript discussed the fact, while event_time is when the real-world event described by the fact occurred. They are different fields; an unknown event_time must not be inferred from dialogue_time.
{memory_sections}"""

MEMORY_RETRIEVED_SECTION_SPECS_EN = (
    (
        "[Retrieved Facts]",
        "These are ranked narrative facts retrieved directly from memory_facts.",
        "fact",
    ),
    (
        "[Long-term States]",
        "These are evolving state projections derived from memory facts. Treat them as summarized context, not direct user quotations.",
        "state",
    ),
    (
        "[Actionable Items]",
        "These are decisions, tasks, commitments, risks, or open questions that may require follow-up.",
        "actionable_item",
    ),
)

ENTITY_EXTRACTION_GUIDANCE_EN = """Entity extraction rules:

Entities are not limited to traditional named entities. In this memory system, an entity is a semantic anchor that can be reused across facts for clustering, retrieval, and graph construction.
Prefer nouns or short noun phrases with long-term memory value instead of only proper names.

Allowed entity type values:
- PERSON: specific people, names, roles, or speakers mentioned in the conversation
- ORGANIZATION: companies, teams, institutions, or groups
- LOCATION: geographic locations or venues
- PRODUCT: products or services
- PROJECT: projects, product names, or long-running work items
- TECHNOLOGY: technology stacks, frameworks, libraries, tools, APIs, or systems
- CONCEPT: abstract concepts, methods, theories, or reusable ideas
- TOPIC: discussed domains or subject areas
- PREFERENCE: user preferences, likes, habits, or dislikes
- OTHER: explicitly mentioned entities that do not fit the above types

Entities that should be extracted include:
- Conversation subjects or roles, such as user, assistant, speaker_1, speaker_2, spouse, child, team, or client
- User-relevant domains, problems, tasks, states, or scenarios, such as health management, physical condition, work, business events, family education, communication, or fatigue
- Reusable plans, methods, tools, activities, or objects, such as healthy eating, family meetings, shared rules, picnics, or yoga mats
- Constraint objects or conditions that affect user choices, such as financial burden, fixed schedule, time shortage, or work pressure

Do not extract ordinary time expressions as entities, such as today, yesterday, last week, the past three days, 2026-05-07, 10:30, or three months.
Time should be stored as fact time metadata, not in the entity graph.
Only named time concepts with semantic identity may be entities, such as Spring Festival, Q3 earnings season, or Sprint 42.

Do not extract pure attributes, adjective-only phrases, isolated degree words, or generic labels as entities; keep them in fact text, keywords, topics, or states instead.
For example: low venue dependence, low intensity, high priority, low cost, strong privacy, lightweight.
If a phrase contains a reusable object or scenario, extract the core object or scenario:
- "fatigue caused by long-term high-intensity work" may yield "work" and "fatigue"
- "high-frequency business activities" may yield "business activities"
- "the financial burden is too heavy" may yield "financial burden"

Entities must come from explicit conversation content or be directly determined by a role/fact subject. Do not over-infer.
Each retained memory fact should usually include a subject entity, such as user/assistant/speaker, plus 1-4 core semantic anchors."""


EPISODE_SUMMARY_PROMPT_EN = """You are the episode summarization module for a long-term memory system. Generate exactly one faithful, traceable, self-contained episode title and episode summary from the chronological dialogue/transcript segments below.

Summary requirements:
1. title must be short and specific enough to distinguish this episode from other episodes on the same broad topic. Avoid vague titles such as "health management" or "family communication".
2. summary must preserve explicit key objects, participants, actions, recommendations, acceptance/refusal, constraints, reasons, conclusions, or unresolved issues from this episode.
3. This is an episode-level faithful compression. Do not turn it into a long-term preference, profile, risk, or cross-episode observation.
4. Use only input evidence. Do not import prior state, outside knowledge, or unsupported completion status.
5. If the episode contains several related high-value points, express them in one complete summary rather than a vague topic statement.
6. Ignore greetings, courtesy closings, and repeated low-value content.
7. Return JSON only. No markdown.

Output schema:
{
  "title": "short specific title",
  "summary": "self-contained episode summary"
}

Dialogue/transcript segments:
{dialogue_batch}
"""


UNIFIED_MEMORY_EXTRACTION_PROMPT_EN = """You are the memory extraction module for a unified AI-glasses memory system inspired by MemPalace.

The system no longer treats assistant_wakeup and allday_recording as two separate memory products. Both sources enter one memory line:
- episode: one coherent interaction or transcript episode.
- fact: traceable, self-contained narrative evidence extracted from the episode.
- state: evolving long-term topic/preference/constraint/risk state derived later from facts.
- index entry: a MemPalace-style directory card used for unified recall.

Your task now is to extract the episode summary, episode canonical_topics, and Hindsight-style high-quality narrative facts from the chronological dialogue/transcript evidence batch below. The evidence may come from an active user-assistant exchange or a multi-speaker ambient transcript; use speaker, role, and time to infer participants and pragmatic flow.

Existing long-term memory states for context:
{existing_memory_states}

canonical_topics rules:
- canonical_topics are stable episode-level topic names, ordered from most relevant to least relevant. Return 1-5 topics.
- If this episode belongs to the same stable object or long-running topic as an existing candidate, reuse the candidate's canonical_topic exactly instead of rewriting it.
- Create a new canonical topic only when the evidence clearly introduces a new stable object, project, product, relationship, or long-running issue.
- Avoid over-generic topics such as "solution finalized", "product design discussion", "team collaboration", "problem discussion", or "user consultation". Prefer a concrete object or stable domain, e.g. "AI glasses voice memory system" or "phone promotion strategy".
- If the evidence is too thin to identify the concrete object, do not reuse an existing topic merely because it is broadly related. Reuse it only when the same object or issue is still clear; otherwise output a conservative topic grounded in the current evidence rather than a fragmented short phrase.
- Reuse an existing topic only after a strict semantic check: the current episode/fact must have substantially the same core object or concrete issue, discussion goal, and semantic scope as that topic's canonical_name. Sharing a broad domain, one entity, similar words, or nearby timestamps is not sufficient.
- If a fact mentions an existing topic only as background but actually discusses a different object or issue, choose a more accurate `fact_root_topic` for that fact instead of forcing it into the existing topic to reduce the topic count.
- Every fact must also output `fact_root_topic` and `fact_aspect_topic` as sibling fields to place the fine-grained issue under a stable root topic. `fact_root_topic` should be a product, project, or long-running issue; `fact_aspect_topic` should be the specific aspect discussed by the fact. If the root cannot be supported, use a conservative root topic grounded in the current evidence; if no finer aspect is supported, set fact_aspect_topic equal to fact_root_topic rather than inventing an unsupported parent.
- `fact_root_topic` must reference the existing `memory_states` entries with `state_scope=topic_state` and their `canonical_name` values. When the fact has the same core object, discussion goal, and semantic scope as an existing topic state, reuse that `canonical_name` exactly. Do not force reuse based only on a shared entity, broad domain, or similar keywords; otherwise use a conservative root topic supported by the current evidence.
- The system reuses this fact's `entities` as the root-topic context entities; do not generate a separate context-entity field.

memory_states usage rules:
- A state with state_scope=topic_state and state_type=topic is a naming reference for episode canonical_topics and the fact's `fact_root_topic`. If the current evidence refers to the same durable object or issue, reuse its canonical_name exactly.
- A state with state_scope=entity_state is only background for understanding durable entity attributes, preferences, constraints, risks, or relationships. Do not directly use an entity_state canonical_name as an episode topic or the fact's fact_root_topic.
- These states are historical context, not evidence for the current episode. Do not write unsupported historical details into a fact; when current dialogue conflicts with a prior state, current dialogue wins.
- fact's `fact_root_topic` must describe the main durable topic of that fact. It may reuse a relevant topic_state canonical_name, but must not become an entity-attribute title merely because an entity_state was provided.
- `fact_root_topic` may reuse a topic_state canonical_name only when the fact's core object, discussion goal, and semantic scope are all substantially aligned with that name; `fact_aspect_topic` should retain the specific aspect discussed under that root.
- Do not reuse a topic_state merely because the fact belongs to a broad domain such as health, products, or teams, or because it contains related background. When the evidence does not support a strict match, use a more specific topic grounded in the current evidence, or create a conservative new topic.
- Never change a fact's root_topic because of an entity_state name, summary, or historical timeline. An entity_state may help interpretation, but it is not a topic candidate to reuse directly.
- Every fact must also output one `primary_entity`, the single entity that the fact mainly describes, affects, or belongs to; it must be one object, not an array.
- `primary_entity` must come from the fact's `entities`. Do not choose an entity merely because it is mentioned, provides a recommendation, or is a location, tool, or background context. For a multi-person exchange, choose the person or entity mainly described or affected by the fact; for a fact about the user's own preference, habit, constraint, or risk, choose the user.
- Keep `entities` for all directly relevant entities so retrieval preserves participants and context; downstream entity-state matching uses only `primary_entity`, so one fact must not be assigned to multiple entities.

""" + ENTITY_EXTRACTION_GUIDANCE_EN + """

Core Hindsight-style narrative fact requirements:
- Each fact should cover a complete exchange or a clear topic segment, not a single utterance. Do not mechanically split "the user raised a problem", "the assistant suggested a solution", and "the user accepted/rejected it" into separate fragments; if they respond to the same issue, merge them into one narrative fact.
- Each fact must be understandable without reading the original dialogue and preserve the pragmatic flow of the interaction: why the user raised the issue, what the assistant suggested, how the user responded, and what preference, decision, constraint, unresolved question, or next step emerged.
- Each fact must naturally include the five dimensions in its text: what (complete event/topic/plan/conclusion), when (conversation timestamp or explicit time anchor), where (location/setting/platform/project scope; if absent, say no specific location/setting was mentioned), who (user, assistant, and other key people/organizations with their roles), and why (explicit reason, motivation, concern, disagreement, constraint, implication, conclusion, or follow-up).
- For a roughly five-turn dialogue batch or a coherent multi-speaker transcript segment, usually produce 1-3 facts. Only split when the batch truly contains multiple unrelated events/topics. In most cases, do not exceed 5 facts.

fact_type classification:
- `semantic` is reusable stable knowledge or long-term information that does not depend on one particular experience, such as project structure, concept definitions, system conventions, common knowledge, a user's long-term preference, a persistent instruction, or a durable constraint. It describes what is generally true or remains valid across conversations.
- `episodic` is a concrete experience or event that happened at a particular time, such as the user making a request in one turn, the assistant modifying or testing something, one failure or success, a decision made at a point in time, a state change, or an emotional reaction. It describes what happened on that occasion; it can still be episodic even when it concerns a long-running project.
- The key test is whether the fact depends on one particular experience to be true, not whether its topic is long-running, important, or potentially useful later. One-off requests, recommendations, modifications, test results, decisions, and risk events are `episodic` by default. Use `semantic` only when the evidence supports knowledge or a pattern that is stable and reusable across contexts and time.
- Do not label a fact `semantic` merely because its fact_kind is preference, risk, or decision. A preference expressed in one situation, a temporary risk, or a single decision remains `episodic`; a repeatedly observed or explicitly long-term preference, constraint, or instruction may be `semantic`.

Temporal fidelity requirements:
- Preserve sequence and ordering expressions exactly when they affect meaning: first, first time, second, previous, next, later, earlier, before, after, once, again, subsequent, prior, last, most recent, and similar wording. Do not paraphrase away order. For example, keep "serviced for the first time on March 15" rather than reducing it to "had a good service experience".
- Preserve relative time expressions in text and keywords: yesterday, last Saturday, previous week, two months ago, about a month ago, mid-February, recently, shortly after, and similar phrases. If the expression can be resolved unambiguously from the Conversation timestamp, write the resolved real-world event time directly into `event_time_key`.
- The `Time` shown for each segment is the dialogue/transcript timestamp. It is only the reference anchor for resolving relative event times, not the default event time for the fact. Derive `event_time_key` from the specific event described by the fact and the temporal evidence in the text; do not copy the dialogue timestamp merely because the event was discussed at that time.
- `event_time_key` is the most representative real-world occurrence time or temporal anchor for the event described by the fact. It is not the dialogue time, extraction time, or current system time. It is a single time field; do not output an event end time, interval, or additional start/end time fields.
- Use this priority when deriving time: explicit absolute date/time in the evidence > a relative expression that can be resolved unambiguously from the segment `Time` > an event explicitly described as happening during the current conversation. For example, with dialogue time 2023-05-30, `last month (around April 2023)` should resolve to a representative time around 2023-04 rather than 2023-05-30; `last weekend (May 27-28)` should use a representative time around 2023-05-27; only an event explicitly described as decided/completed today should use 2023-05-30.
- If a fact describes both a current conversational act and an earlier background event, use the time of the event the fact primarily describes; split the fact when necessary instead of allowing the dialogue timestamp to overwrite the earlier event time. If only a month, weekend, or relative period is supported, preserve the original expression in text/keywords and use a conservative representative time anchor in `event_time_key`.
- Prefer explicit dates, times, weekdays, and relative time expressions from the evidence. Resolve a relative expression to an absolute time only when the current segment `Time` makes the resolution unambiguous; otherwise do not guess, leave `event_time_key` empty, and set `time_confidence` to `unknown`. Never fabricate an event time from the dialogue timestamp, current time, or extraction time.
- If one fact contains multiple events with different times, split them into separate facts instead of using one event time to hide unrelated events.
- If an event's answerability depends on temporal order, the fact text must include both the event object and the time anchor or order marker. Do not store only the topic name.
- If multiple events in the batch may later be compared by before/after/first/which happened earlier, either keep them in one narrative fact that explicitly states their relative order, or split them into separate complete facts with their own time anchors. Avoid keeping only one side of a comparison.
- Personal events mentioned as side context remain important when they include time anchors or ordering words, such as purchases, service/maintenance, repairs, appointments, attendance, travel, meetings, tests, failures, and decisions.

Rules:
1. Extract 0-5 facts. Do not force a fact for every turn.
2. Each fact must be a complete narrative preserving the essential topic background plus the flow of user/assistant views or actions and an explicit reason, disagreement, constraint, conclusion, or next step.
3. Preserve concrete answerable details: names, places, titles, colors, dates, weekdays, relative times, numbers, amounts, durations, products, organizations, recommendations, constraints, decisions, and user preferences.
4. Do not drop personal events mentioned as asides, e.g. "by the way", "I also", "I just", "last Saturday", "two months ago"; but if they are context inside the same exchange, merge them into the same narrative fact instead of emitting context-free short notes.
5. Split only truly unrelated events. Events that must be compared for temporal reasoning may be split, but every split fact must still preserve its own background and time anchor.
6. Use only the dialogue evidence. Do not invent completion, intent, or reasons.
7. Keep assistant recommendations that contain concrete future-answerable items inside the relevant exchange narrative, and include whether the user accepted, rejected, hesitated, or added constraints when supported.
8. priority is 0-100. Keep only facts worth at least 60.
9. fact_type must be semantic or episodic, using the stable-knowledge/long-term-information versus one-specific-event boundary above.
10. fact_kind must be preference, decision, request, recommendation, action, commitment, open_question, risk, error, context, instruction, or other.
11. Do not output short facts like "the user said X" or "the assistant suggested Y". If deleting the topic background, reason, disagreement, or conclusion would make the text a vague short note, add those details back; if the dialogue does not support them, omit the fact.
12. Do not store assistant pleasantries, generic closings, or low-information encouragement as standalone facts, e.g. "hope this helps", "let me know if you have other questions", "okay", or "you're welcome", unless they explicitly change a decision, commitment, or next step.
13. keywords must be short retrieval terms: entities, topics, symptoms, plans, constraints, decisions, and important time/order anchors. For time-sensitive facts, include the original or resolved time phrase such as "March 15 2023", "first service", "3/22", "last Saturday", or "two months ago". Do not put full sentences, pleasantries, filler, generic encouragement, or phrases like "hope this method helps you" into keywords.
14. Return JSON only. No markdown.

entity_state_signal rules:
- `entity_state_signal` only flags that the fact may contribute to a durable state of the user or another key entity; it is not the final entity_state update.
- Output it only when the evidence explicitly expresses a reusable preference, profile, routine, relationship, constraint, or risk. Use an empty array for one-off events, temporary suggestions, pleasantries, and low-value background.
- Return at most 3 signals per fact. Each signal contains only `state_type`, `attribute_name`, `evidence_basis`, and `confidence`, plus an evidence-supported `entity` when needed; do not generate an aspect summary or use prior state. Later reflection decides whether to create, update, or ignore it.
- state_type must be one of: preference, profile, routine, relationship, constraint, risk.
- attribute_name should name the specific potentially affected attribute, and evidence_basis must cite evidence from the current fact.

action_signal rules:
- `action_signal` is only a candidate hint, not the final actionable_item. It may contain a small number of false positives and will be re-checked by the actionable extraction module.
- Output it only when the fact clearly involves a future action, execution commitment, unresolved decision, or explicit reminder/follow-up. Otherwise output an empty array. Do not infer an owner, deadline, completion state, or final conclusion from context.
- `action_strength` must be one of `assigned`, `committed`, `pending_decision`, or `follow_up`; it is a coarse candidate type, not a final judgment. `item_type=decision` is allowed only with `pending_decision`.
- Return at most 1 signal per fact. Each signal contains only `item_type`, `action_strength`, `evidence_basis`, and `confidence`, with optional evidence-supported `due_at`; do not generate action_summary, owner, or status.
- `evidence_basis` must come from direct evidence in the current fact; do not combine facts or invent an owner, time, or conclusion.
- item_type must be one of: task, commitment, decision, follow_up, open_question, risk, reminder, recommendation, constraint.

Output schema:
{
  "episode_title": "short concrete title",
  "episode_summary": "self-contained summary of the interaction episode",
  "canonical_topics": ["stable topic 1 ordered by relevance", "stable topic 2"],
  "facts": [
    {
      "text": "self-contained narrative fact covering a complete exchange and expressing what/when/where/who/why, with background, user/assistant view or action flow, and reason/disagreement/constraint/conclusion/next step",
      "keywords": ["keyword1", "keyword2"],
      "entities": [{"name": "entity name", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|TOPIC|PREFERENCE|OTHER"}],
      "primary_entity": {"name": "the single primary entity of this fact", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|TOPIC|PREFERENCE|OTHER"},
      "fact_root_topic": "stable product/project/long-running issue root topic",
      "fact_aspect_topic": "specific aspect discussed by this fact",
      "fact_type": "semantic|episodic; semantic=reusable stable knowledge or long-term information, episodic=an event or state change tied to a specific experience",
      "fact_kind": "preference|decision|request|recommendation|action|commitment|open_question|risk|error|context|instruction|other",
      "priority": 80,
      "event_time_key": "real-world event occurrence time or representative temporal anchor derived from the dialogue time anchor and fact content; empty when it cannot be determined",
      "time_confidence": "explicit|inferred_from_turn|unknown; explicit evidence, resolved from the segment Time and a relative expression, or undetermined",
      "where": "",
      "entity_state_signal": [
        {
          "state_type": "preference|profile|routine|relationship|constraint|risk",
          "attribute_name": "specific attribute name",
          "entity": {"name": "explicitly affected entity", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|TOPIC|PREFERENCE|OTHER"},
          "evidence_basis": "specific evidence from this fact supporting the signal",
          "confidence": 0.8
        }
      ],
      "action_signal": [
        {
          "item_type": "task|commitment|decision|follow_up|open_question|risk|reminder|recommendation|constraint",
          "action_strength": "assigned|committed|pending_decision|follow_up",
          "due_at": "",
          "evidence_basis": "specific evidence from this fact supporting the signal",
          "confidence": 0.8
        }
      ]
    }
  ]
}

Dialogue/transcript evidence batch:
{dialogue_batch}
"""

UNIFIED_TOPIC_STATE_UPDATE_PROMPT_EN = """You are the topic_state update module for a unified AI-glasses long-term memory system inspired by MemPalace.

Topic resolution has already been completed: the system decided that the candidate evidence belongs to one durable root topic, or that a new root topic should be created. Use candidate_topic_state and the existing topic_state to update the root topic_state summary; do not re-route the topic.

Rules:
1. `candidate_topic_state.root_topic_name` is the stable root topic. Update only that root topic; do not merge content that merely shares an entity or broad domain.
2. `aspect_topics` are concrete aspects under the root topic. Use them to understand local progress, but do not create a separate topic_state for every aspect.
3. `parent_topics` are auxiliary episode-level or higher-level context. Use them only when relevant to the root topic; they must not override a more accurate root_topic_name.
4. `identity_text`, `keywords`, `context_entities`, and `fact_summaries` are aggregated identity and evidence fields. Use specific, repeated, or concluded information from them, but do not copy identity_text verbatim or dump every keyword and entity into the summary.
5. `fact_ids` are the available evidence IDs. Cite only IDs that are actually supported by the candidate content; never infer facts from an ID.
6. The summary should describe the durable root state: background, recent changes, key participants, decisions/preferences/constraints that still matter, unresolved questions, and next steps.
7. If existing_topic_state is present, merge incrementally instead of concatenating. Preserve durable information that is still valid.
8. Do not rewrite one fact into another fact. A topic_state must be more abstract and stable than an individual fact.
9. summary must be a concise current-state snapshot: at most 1-2 sentences and preferably no more than 80 English words. Do not append the historical timeline or every aspect to summary.
10. time_line_updates must contain only changes supported by candidate_topic_state, with 0-3 events. Each event must include time, change_type, a short change summary, and fact_ids. Do not repeat existing timeline events or record unchanged information.
11. evidence_fact_ids must come from candidate_topic_state.fact_ids and cite only facts that support this update.
12. Return JSON only. No markdown.

Output schema:
{
  "update_needed": true,
  "canonical_name": "stable topic name",
  "summary": "concise current long-term topic_state snapshot",
  "aspects": [
    {
      "name": "specific aspect",
      "summary": "current progress for this aspect",
      "status": "active|stable|resolved|uncertain"
    }
  ],
  "time_line_updates": [
    {
      "occurred_at": "",
      "change_type": "confirmed|changed|rejected|resolved|updated",
      "summary": "the state change in this update",
      "fact_ids": [1]
    }
  ],
  "keywords": ["keyword1", "keyword2"],
  "entities": ["entity1", "entity2"],
  "canonical_topics": ["topic1"],
  "evidence_fact_ids": [1, 2],
  "importance": 0.8,
  "confidence": 0.85,
  "status": "active|stable|resolved|uncertain"
}

candidate_topic_state:
{candidate_topic_state}

candidate_topic_state field descriptions and usage:
- `root_topic_name`: the stable root topic this state belongs to. Do not turn an aspect or unrelated episode topic into a new root topic.
- `topic_key`: the stable internal root-topic key. Use it only to confirm identity; do not write it into summary, canonical_name, or timeline.
- `identity_text`: an aggregated identity text containing the root topic, aspects, keywords, entities, and fact summaries. Use it for holistic topic understanding; do not copy it verbatim as the state summary.
- `aspect_topics`: concrete aspects such as platform selection, budget constraints, or plan progress. They are context under the root topic, not necessarily independent states.
- `parent_topics`: higher-level episode context. Use it only when it helps explain the root topic and never let it override root_topic_name.
- `keywords`: aggregated retrieval anchors from the related facts. Use them to identify concrete objects, actions, outcomes, and constraints; do not list generic words in the summary.
- `context_entities`: semantic entities from the related facts. Use them to confirm people, products, projects, or objects only when they matter to the root state.
- `fact_summaries`: compressed summaries of the related facts and the main evidence for the candidate update. Extract only content directly relevant and durable for the root topic.
- `fact_ids`: IDs corresponding to the summaries. Use them only for evidence_fact_ids and time_line_updates[].fact_ids.
- `source_type`: provenance of the candidate; treat it as context, not topic content.

existing_topic_state:
{existing_topic_state}
"""


UNIFIED_ENTITY_STATE_UPDATE_PROMPT_EN = """You are the entity-scoped state update module for a unified AI-glasses long-term memory system inspired by MemPalace.

The input has already passed entity and preliminary attribute-topic resolution: the system has decided that these facts may update one durable state type for one entity and one attribute. Your task is to update that specific entity attribute, not to re-route the entity.

entity-scoped state targets:
- preference: stable preferences, selection tendencies, likes/dislikes.
- relationship: durable relationship state between this entity and other people, organizations, or projects.
- profile: stable identity, background, responsibility, role, or important context.
- routine: repeated habits, processes, cadence, or workflow.
- constraint: durable or currently persistent limitations that affect action.
- risk: a persistent risk about an entity that can affect future decisions or actions.

Rules:
1. Update only the given entity, state_type, and attribute_name. Do not write a topic progress summary.
2. If the facts only describe topic progress, leave that to topic_state. Keep only durable information about this specific entity attribute.
3. If existing_entity_state is about a different attribute, return update_needed=false instead of forcing a merge.
4. If existing_entity_state is present and about the same attribute, merge incrementally instead of concatenating.
5. canonical_name must be only a short, concrete attribute or topic title, such as "flexible workout preference" or "health management". Do not include the entity name, state_type, slashes, hyphens, or a full sentence. The entity is provided separately and state_type is a separate field. If existing_entity_state describes the same attribute, reuse its canonical_name without entity or state_type decorations.
6. If the input supports only a one-off event, single recommendation, temporary request, or courtesy response, return update_needed=false.
7. summary must be a concise current-state snapshot: at most 1-2 sentences and preferably no more than 60 English words. Do not append the historical timeline to summary.
8. time_line_updates must contain only changes supported by the new facts, with 0-3 events. Each event must include time, change_type, a short change summary, and fact_ids. Do not repeat existing timeline events or record unchanged information.
9. The summary must answer: "What should we remember long-term about this entity attribute?"
10. evidence_fact_ids must cite supporting fact IDs from the input facts.
11. Return JSON only. No markdown.

Output schema:
{
  "update_needed": true,
  "canonical_name": "short attribute or topic title without entity or state_type",
  "summary": "concise current entity-scoped state snapshot",
  "time_line_updates": [
    {
      "occurred_at": "",
      "change_type": "confirmed|changed|rejected|resolved|updated",
      "summary": "the state change in this update",
      "fact_ids": [1]
    }
  ],
  "keywords": ["keyword1", "keyword2"],
  "entities": ["entity1", "entity2"],
  "evidence_fact_ids": [1, 2],
  "importance": 0.8,
  "confidence": 0.85,
  "status": "active|stable|resolved|uncertain"
}

entity_state_target:
{entity_state_target}

entity_state_target field descriptions and usage:
- `entity`: the entity described by this update. All summary, canonical_name, and time_line_updates content must be about this entity. Do not assign the state to another entity that is only mentioned as background or context.
- `entity_key`: the stable internal identity key for the entity. Use it only to confirm that the candidate and existing_entity_state refer to the same entity. It is not natural-language content and must not be copied into summary, canonical_name, or the timeline.
- `state_type`: the entity_state type that this update is allowed to modify. Stay within this type; do not switch to preference, profile, routine, relationship, constraint, or risk merely because the candidate also touches another aspect.
- `attribute_name`: the specific attribute represented by this candidate and the main semantic boundary of the update. The summary must explain what this attribute means long-term for the entity, rather than summarizing the whole dialogue or a topic_state.
- `attribute_key`: the stable internal key for the attribute. Use it to help confirm attribute identity, but do not copy it into summary or canonical_name.
- `attribute_name_aliases`: alternative or historical names for the same attribute. Use them when deciding whether existing_entity_state describes the same attribute, but do not mechanically concatenate all aliases into the output. If the existing state describes a different attribute, return `update_needed=false`.
- `state_signal_evidence`: evidence in the candidate facts supporting this entity_state signal. The signal is not a final state conclusion; use the full fact summaries and existing_entity_state to decide whether an update is warranted.
- `state_signal_evidence[].fact_id`: the fact identifier supporting the signal. Use it only to cite evidence in `evidence_fact_ids` and `time_line_updates[].fact_ids`.
- `state_signal_evidence[].confidence`: the extraction confidence for the signal. Use it to stay conservative; low-confidence or weakly supported signals must not be expanded into new long-term conclusions.

Processing principle: first use `entity` and `entity_key` to confirm the state owner, then use `state_type`, `attribute_name`, and the attribute aliases to establish the update boundary, and finally use the current facts, `state_signal_evidence`, and existing_entity_state to produce the current state. Update only when the evidence has durable value for this entity attribute. For one-off events, single recommendations, temporary requests, or topic-only progress, return `update_needed=false`. When existing_entity_state refers to the same entity and attribute, merge the candidate incrementally while preserving the existing long-term conclusion. If the attribute is different, do not force a merge.

existing_entity_state:
{existing_entity_state}
"""


UNIFIED_ACTIONABLE_ITEM_EXTRACTION_PROMPT_EN = """You are the actionable-item extraction module for a unified AI-glasses long-term memory system inspired by MemPalace.

The input contains one topic candidate and its newly stored narrative facts. Some facts include action_signal as pre-filtered action signals. Within this topic, extract only concrete actionable items that truly require future follow-up, reminder, execution, review, or decision tracking. Actionable items are separate from evolving states: states summarize durable situations, preferences, constraints, and background; actionable items must be checkable, completable, trackable, or explicitly recalled as decisions, commitments, risks, or open questions.

`action_signal` is only a candidate hint and may contain false positives. Do not convert it directly into an actionable item. Re-check the complete fact summary and direct signal evidence; when uncertain, return an empty list.

Rules:
1. Extract 0-4 actionable items. Many ordinary dialogue batches should return an empty list. Do not force coverage of facts.
2. Every item must have direct fact evidence and a verifiable outcome. action_signal is necessary candidate evidence but not sufficient; suggestions, plans, considerations, discussions, or vague "discuss further" language do not create an item by themselves.
3. `pending_decision` requires all three: a concrete decision object, an explicitly unresolved state, and a concrete option, confirmation/selection action, or deadline. If any condition is missing, do not create a decision/open_question.
4. "Decision X is complete but implementation details/progress remain" is not `pending_decision`; create a task/commitment only when an owner, action, and deliverable are explicit.
5. `assigned` requires an explicit responsible party, concrete action, and deliverable or completion criterion; `committed` requires an explicit commitment to execute; `follow_up` requires an explicit reminder, follow-up, review, next confirmation, or report request. "Someone proposed", "suggested", "needs", or "departments should follow up" is insufficient.
6. Completed decisions without remaining action, ordinary constraints, preferences, and background belong in facts/states. Weak willingness and ordinary assistant recommendations also default to an empty list.
7. Do not copy every fact. If multiple facts point to the same matter, keep only the most specific and trackable item.
8. Always return an empty list for: "The team decided to launch several gift forms, but implementation details are not settled"; "Suggest learning more and discussing later"; "Someone proposed negotiating with the bank." These may be `pending_decision`: "It is not yet decided whether to use TikTok or Xiaohongshu as the main platform"; "The team disagrees and must confirm the product color by next Tuesday".
9. `evidence_fact_ids` must cite fact IDs from the input; each item should map to a directly supporting fact signal.
10. Keep each item self-contained: `summary` must state "who is responsible + by what time or deadline (only when supported by evidence) + what action or target is involved", with the necessary context, reason, and current status. Do not invent a date when no time is supported; when the responsible party is unclear, you may say "responsible party not established" while still stating the action.
11. `owner` identifies who the item belongs to and who is responsible for completing, following up, or deciding it. It is not the target object and not a list of mentioned entities. Use explicit assignment language in the facts; use `primary_entity` only when it also clearly represents execution or decision responsibility. When updating an existing item, preserve its owner unless new facts clearly show that responsibility changed. Use `unknown` when the evidence is insufficient.
12. owner must be the name of an evidence-supported responsible entity, such as "user", "assistant", "N_SPK8013", "Xiao Wang", "the team", or "the marketing department". Do not assign ownership merely because someone spoke or was mentioned, and do not output the generic category "other". Use `unknown` only when the responsible party is not established.
13. evidence_fact_ids must cite supporting fact IDs from the input.
14. item_type must be one of: task, commitment, decision, follow_up, open_question, risk, reminder, recommendation, constraint, other.
15. status must be one of: open, in_progress, done, blocked, decided, noted, unknown.
16. importance and confidence are 0-1. Do not inflate confidence to bypass the weak-willingness rule.
17. When existing actionable_items are provided for this topic, first check whether the new facts explicitly change their status, deadline, owner, or current description. Emit `operation="update"` only when new evidence supports a change; do not mark an item done without evidence of completion.
18. For an existing item, use its `id` in `existing_item_id` and preserve its `canonical_name` and `item_type`; for a new item use `operation="create"` and `existing_item_id=0`.
19. Return JSON only. No markdown.

Output schema:
{
  "actionable_items": [
    {
      "operation": "create|update",
      "existing_item_id": 0,
      "item_type": "task|commitment|decision|follow_up|open_question|risk|reminder|recommendation|constraint|other",
      "canonical_name": "stable short name",
      "summary": "self-contained actionable item stating the responsible party, time/deadline when supported, and concrete action",
      "owner": "concrete responsible entity name, such as user, Xiao Wang, or the project manager; use unknown when unclear",
      "status": "open|in_progress|done|blocked|decided|noted|unknown",
      "due_at": "",
      "evidence_fact_ids": [1, 2],
      "keywords": ["keyword1", "keyword2"],
      "canonical_topics": ["topic1"],
      "importance": 0.8,
      "confidence": 0.85
    }
  ]
}

Topic candidate, including topic metadata and detailed supporting facts:
{topic_candidate}

Existing topic_state, if any:
{existing_topic_state}

Existing actionable_items for this topic:
{existing_actionable_items}

"""


RECALL_QUERY_ANALYSIS_PROMPT_EN = """You are the recall query analyzer for the AI-glasses long-term memory system.

Understand the memory structure before analyzing the query. 

Memory structure:
1. `memory_facts` / fact: traceable, self-contained narrative facts extracted from one conversation episode or all-day transcript. They preserve what happened, participants, time, place or scene, reasons, viewpoint changes, suggestions, acceptance or rejection, constraints, conclusions, and unresolved questions. A fact may contain an explicitly stated preference, routine, profile detail, risk, or constraint, but it remains current conversational evidence rather than a cross-episode long-term summary. Facts usually include `fact_type`, `fact_kind`, `primary_entity`, `summary`, `keywords`, `entities`, `fact_root_topic`, `fact_aspect_topic`, `event_time_key`, and `dialogue_time_key`.
2. `memory_states` / state: durable evolving projections updated from multiple facts, not raw dialogue quotations. It contains:
   - `topic_state`: the root state for a project, product, topic, or long-running issue, containing overall background, progress, decisions, constraints, risks, and unresolved issues; fine-grained aspects such as "livestream platform selection" or "gift plan" are stored as root-state context and retrieval aliases and do not necessarily become separate states.
   - `entity_state`: durable properties of an entity, including preference, routine, profile, relationship, constraint, and risk.
   Use state for long-term patterns and snapshots, but do not treat it as a replacement for concrete fact evidence.
3. `memory_actionable_items` / actionable_item: concrete items extracted from facts that need future execution, follow-up, reminder, review, or decision tracking. They include tasks, commitments, decisions, follow-ups, open questions, risks, reminders, recommendations, and constraints that block a specific action. Items usually include `canonical_name`, `summary`, `owner`, `status`, `due_at`, and `evidence_fact_ids`. Ordinary preferences, background, one-off descriptions, and suggestions without a concrete next action are not actionable items.

An episode is the storage container for a conversation or transcript batch with a title, summary, participants, and time range. The default recall path does not retrieve episodes as an independent selectable layer. For recalling an experience, prefer `fact`; for a durable overview, consider `state` as well. States and actionable items can be traced back to facts through `evidence_fact_ids`.

Guidance:
- Use `source_types` only when the query clearly points to assistant_wakeup interactions or allday_recording transcripts. Otherwise use both.
- Prefer `fact` for what happened, dates, places, people, exact evidence, event order, and traceable details.
- Prefer `state` for stable preferences, durable constraints, routines, relationships, profiles, and topic/project/issue evolution.
- Prefer `actionable_item` for tasks, commitments, decisions, open questions, risks, reminders, recommendations, and explicit next steps. If the user also asks for background or evidence, include `fact` too.
- For queries about current progress, durable status, and next steps, usually include `state`, `actionable_item`, and `fact`.
- Keep the plan broad when unsure, but do not select every layer by default. Missing evidence is worse than retrieving a few extra candidates.
- Output 1-3 values in `layer_preference`, chosen from `fact`, `state`, and `actionable_item`. It identifies layers to prioritize; it is not a new database table.
- Extract 2-8 short retrieval keywords, prioritizing concrete people, organizations, products, projects, topics, actions, outcomes, constraints, and time anchors. Do not output full sentences, pleasantries, or generic words.
- Extract useful semantic entities with names and types. Entities may be people, organizations, locations, products, projects, technologies, or concrete concepts; ordinary time expressions such as today, yesterday, or last week are not entities.
- `temporal_mode` selects which fact timestamp should be used for a time range: `event_time` means the real-world event time described by the fact, `dialogue_time` means when the conversation/transcript occurred, `both` means either timestamp may match, and `none` means no hard time filter. Prefer `event_time` for queries asking what happened, was done, bought, or visited; prefer `dialogue_time` for queries asking what was discussed, mentioned, or asked; use `none` when the temporal intent is unclear.

Return JSON only:
{
  "source_types": ["assistant_wakeup", "allday_recording"],
  "layer_preference": ["fact", "actionable_item", "state"],
  "needs_broad_evidence": false,
  "query_rewrite": "retrieval-focused rewrite over raw memory tables",
  "keywords": ["keyword1", "keyword2"],
  "entities": [{"name": "entity name", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|OTHER"}],
  "temporal_mode": "event_time|dialogue_time|both|none"
}

User query:
{query}
"""
