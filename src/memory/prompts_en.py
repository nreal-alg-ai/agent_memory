"""English prompt templates for the unified memory prototype."""

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
- Preserve relative time expressions in text and keywords: yesterday, last Saturday, previous week, two months ago, about a month ago, mid-February, recently, shortly after, and similar phrases. If the expression is tied to a known Conversation timestamp, also write the resolved date or conservative date range into occurred_start/occurred_end.
- `occurred_start` is the real-world start time or occurrence time of the event described by the fact; `occurred_end` is the real-world end time or upper bound of the event interval. They are not the dialogue-turn time, extraction time, or current system time.
- For a point event, one-off action, or event whose duration cannot be established, fill only `occurred_start` and leave `occurred_end` as an empty string. Fill `occurred_end` only when the evidence clearly indicates a duration, ending, or upper bound.
- Prefer explicit dates, times, weekdays, and relative time expressions from the evidence. Resolve a relative expression to an absolute date only when the Conversation timestamp makes the resolution unambiguous; otherwise do not guess, leave the time field empty, and set `time_confidence` to `unknown`. Never fabricate an event time from the current time.
- If one fact contains multiple events with different time ranges, split them into separate facts instead of using one interval to hide unrelated events. Use both fields only for the start and end of the same event.
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

state_aspects rules:
- `fact_kind` remains the primary semantic type of the fact. `state_aspects` are projection slices showing how this fact can contribute to multiple entity_state types.
- Output state_aspects only when the fact has durable long-term value for the user or another key entity. For one-off events, temporary suggestions, pleasantries, or low-value background, use an empty array.
- Return at most 3 state_aspects per fact, keeping only the clearest and most useful aspects.
- state_type must be one of: preference, profile, routine, relationship, constraint, risk.
- aspect_summary must describe only the contribution for the current state_type, not restate the whole fact. For example, risk must say what may be affected or what negative outcome may happen; constraint states the limiting condition; routine states repeated behavior or cadence; preference states likes, dislikes, refusal, or selection tendency.
- attribute_name must be more specific than the episode topic, such as "flexible workout preference", "health management execution constraint", or "weight-loss plan continuity risk". Avoid broad names like "health management" unless the evidence supports only a broad attribute.
- evidence_basis must cite the specific evidence in the current fact that supports this aspect. Do not import prior state or outside inference.

actionable_aspects rules:
- `actionable_aspects` are candidate projections showing how this fact may later become an actionable_item, used to reduce downstream LLM cost and noise.
- Output actionable_aspects only when the fact explicitly contains something that needs future reminder, follow-up, execution, review, decision tracking, open-question resolution, or a high-value risk that blocks action.
- Return at most 2 actionable_aspects per fact, keeping only the most concrete and trackable action signals.
- item_type must be one of: task, commitment, decision, follow_up, open_question, risk, reminder, recommendation, constraint.
- Do not extract weak willingness such as "might try", "sounds good", or "may consider" as an actionable_aspect unless it also includes an explicit reminder, deadline, follow-up check, strong commitment, or verifiable execution plan.
- action_summary must describe the specific item to follow up, execute, remind, review, or track. Do not restate the whole fact.
- trigger_basis must cite the specific evidence in the current fact that supports this actionable signal.
- due_at should be filled only when the evidence contains an explicit time, deadline, or reminder time; otherwise use an empty string.

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
      "occurred_start": "real-world event start time or point-event time; empty when it cannot be determined",
      "occurred_end": "real-world event end time or interval upper bound; empty for point events",
      "time_confidence": "explicit|inferred_from_turn|unknown; explicit evidence, inferred from the current conversation time, or undetermined",
      "where": "",
      "state_aspects": [
        {
          "state_type": "preference|profile|routine|relationship|constraint|risk",
          "attribute_name": "specific attribute name",
          "aspect_summary": "durable contribution for this state_type only",
          "evidence_basis": "specific evidence from this fact supporting the aspect",
          "confidence": 0.8
        }
      ],
      "actionable_aspects": [
        {
          "item_type": "task|commitment|decision|follow_up|open_question|risk|reminder|recommendation|constraint",
          "action_summary": "specific item that needs future reminder, follow-up, execution, review, or decision tracking",
          "owner": "user|assistant|other|unknown",
          "status": "open|in_progress|done|blocked|decided|noted|unknown",
          "due_at": "",
          "trigger_basis": "specific evidence from this fact supporting the actionable signal",
          "confidence": 0.8
        }
      ]
    }
  ]
}

Dialogue/transcript evidence batch:
{dialogue_batch}
"""


UNIFIED_STATE_UPDATE_PROMPT_EN = """You are the state update module for a unified AI-glasses long-term memory system inspired by MemPalace.

The input contains newly stored narrative facts and the current long-term states. Your task is to update or create compact evolving states that help future recall. A state is not a copy of one fact. It should summarize stable preferences, recurring behavior, durable constraints, persistent risks, important relationships, or topic-level situations across multiple facts.

Boundary rules:
- Topic, project, and issue progress belong in topic_state. Do not create a separate project-style state.
- Concrete tasks, decisions, commitments, reminders, and open questions belong in actionable_item. Do not create task-style or commitment-style states.
- state_scope must be topic_state or entity_state; topic_state always uses state_type topic.
- Output a state only when the information should remain useful as durable memory. Do not output a state for a one-off task or commitment.

Rules:
1. Create or update only states that are useful beyond the current episode.
2. Do not create a state from a single trivial fact unless it is a durable preference, long-running constraint, persistent risk, or important life/project context.
3. Merge facts about the same durable subject into one state instead of creating near-duplicates.
4. Preserve uncertainty and recent changes. If evidence conflicts, explicitly state the conflict.
5. Use evidence_fact_ids to cite the fact IDs that support the state.
6. state_type must be one of: topic, preference, profile, routine, relationship, constraint, risk.
7. importance is 0-1. confidence is 0-1.
8. Return JSON only. No markdown.

Output schema:
{
  "states": [
    {
      "state_scope": "topic_state|entity_state",
      "state_type": "topic|preference|profile|routine|relationship|constraint|risk",
      "canonical_name": "stable short name",
      "summary": "self-contained evolving state",
      "evidence_fact_ids": [1, 2],
      "keywords": ["keyword1", "keyword2"],
      "entities": ["entity1", "entity2"],
      "canonical_topics": ["topic1"],
      "importance": 0.8,
      "confidence": 0.85,
      "status": "active|stable|resolved|uncertain"
    }
  ]
}

Existing states:
{existing_states}

New facts:
{facts}
"""


UNIFIED_TOPIC_STATE_UPDATE_PROMPT_EN = """You are the topic_state update module for a unified AI-glasses long-term memory system inspired by MemPalace.

The input has already passed topic resolution: the system has decided that these facts belong to a long-term topic, or that a new long-term topic should be created. Your task is to update that topic_state summary, not to re-route the topic.

Rules:
1. The given canonical_topic is the root topic. Update only that root topic and do not merge unrelated facts.
2. Do not create a separate topic_state for each aspect; return concrete aspects as local progress under the root topic.
3. The summary should describe the durable root state: background, recent changes, key participants, decisions/preferences/constraints that still matter, unresolved questions, and next steps.
4. If existing_topic_state is present, merge incrementally instead of concatenating. Preserve durable information that is still valid.
5. Do not rewrite one fact into another fact. A topic_state must be more abstract and stable than individual facts.
6. summary must be a concise current-state snapshot: at most 1-2 sentences and preferably no more than 80 English words. Do not append the historical timeline or every aspect to summary.
7. Return only evidence-supported aspects from the input. Each aspect should include a name, current progress summary, and status.
8. time_line_updates must contain only changes supported by the new facts, with 0-3 events. Each event must include time, change_type, a short change summary, and fact_ids. Do not repeat existing timeline events or record unchanged information.
9. evidence_fact_ids must cite supporting fact IDs from the input facts.
10. Return JSON only. No markdown.

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

canonical_topic:
{canonical_topic}

existing_topic_state:
{existing_topic_state}

new facts:
{facts}
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

existing_entity_state:
{existing_entity_state}

new facts:
{facts}
"""


UNIFIED_ACTIONABLE_ITEM_EXTRACTION_PROMPT_EN = """You are the actionable-item extraction module for a unified AI-glasses long-term memory system inspired by MemPalace.

The input contains newly stored candidate narrative facts. Some facts include actionable_aspects as pre-filtered action signals. Extract only concrete actionable items that truly require future follow-up, reminder, execution, review, or decision tracking. Actionable items are separate from evolving states: states summarize durable situations, preferences, constraints, and background; actionable items must be checkable, completable, trackable, or explicitly recalled as decisions, commitments, risks, or open questions.

Prefer actionable_aspects when present, but extract only items directly supported by the fact summary and actionable_aspects.

Rules:
1. Extract 0-4 actionable items. Many ordinary dialogue batches should return an empty list. Do not force coverage of facts.
2. Extract only strong actionable items: the user explicitly asks for a reminder/follow-up/record/arrangement/execution, the user or assistant clearly commits to future action, the user makes a trackable decision, a specific open issue must be resolved later, or a high-value risk is blocking an action.
3. Do not extract weak willingness such as "the user is willing to try", "might try", "sounds good", or "may consider" as standalone actionable items. These belong in facts or states. Extract them only when they include an explicit reminder request, deadline, follow-up check, strong commitment, or verifiable execution plan.
4. Do not extract ordinary assistant recommendations as actionable items. Extract them only when the user explicitly accepts them, asks for follow-up/reminders, or the recommendation becomes the user's task, commitment, or decision.
5. Ordinary constraints, preferences, and background belong in states, not constraint items. Extract a constraint item only when the constraint is actively blocking a concrete action or decision.
6. Do not copy every fact. If multiple facts point to the same matter, keep only the most specific and trackable item.
7. Keep each item self-contained: include actor/owner, target object, context, reason, deadline/time if known, and current status.
8. evidence_fact_ids must cite supporting fact IDs from the input.
9. item_type must be one of: task, commitment, decision, follow_up, open_question, risk, reminder, recommendation, constraint, other.
10. status must be one of: open, in_progress, done, blocked, decided, noted, unknown.
11. importance and confidence are 0-1. Do not inflate confidence to bypass the weak-willingness rule.
12. Return JSON only. No markdown.

Output schema:
{
  "actionable_items": [
    {
      "item_type": "task|commitment|decision|follow_up|open_question|risk|reminder|recommendation|constraint|other",
      "canonical_name": "stable short name",
      "summary": "self-contained actionable item",
      "owner": "user|assistant|other|unknown",
      "status": "open|in_progress|done|blocked|decided|noted|unknown",
      "due_at": "",
      "evidence_fact_ids": [1, 2],
      "keywords": ["keyword1", "keyword2"],
      "entities": ["entity1", "entity2"],
      "canonical_topics": ["topic1"],
      "importance": 0.8,
      "confidence": 0.85
    }
  ]
}

Candidate facts:
{facts}
"""


RECALL_QUERY_ANALYSIS_PROMPT_EN = """You are the recall query analyzer for the AI-glasses long-term memory system.

Understand the memory structure before analyzing the query. The default recall path does not use the shared `memory_index_entries` table as its retrieval entry point. It searches the following raw memory tables directly, so do not treat "index" as an additional unified document layer.

Memory structure:
1. `memory_facts` / fact: traceable, self-contained narrative facts extracted from one conversation episode or all-day transcript. They preserve what happened, participants, time, place or scene, reasons, viewpoint changes, suggestions, acceptance or rejection, constraints, conclusions, and unresolved questions. A fact may contain an explicitly stated preference, routine, profile detail, risk, or constraint, but it remains current conversational evidence rather than a cross-episode long-term summary. Facts usually include `fact_type`, `fact_kind`, `primary_entity`, `summary`, `keywords`, `entities`, `fact_root_topic`, `fact_aspect_topic`, and `time_key`.
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

Return JSON only:
{
  "source_types": ["assistant_wakeup", "allday_recording"],
  "layer_preference": ["fact", "actionable_item", "state"],
  "needs_broad_evidence": false,
  "query_rewrite": "retrieval-focused rewrite over raw memory tables",
  "keywords": ["keyword1", "keyword2"],
  "entities": [{"name": "entity name", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|OTHER"}]
}

User query:
{query}
"""
