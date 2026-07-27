"""English prompt templates for the unified memory prototype."""

UNIFIED_MEMORY_EXTRACTION_PROMPT_EN = """You are the memory extraction module for a unified AI-glasses memory system inspired by MemPalace.

The system no longer treats assistant_wakeup and allday_recording as two separate memory products. Both sources enter one memory line:
- episode: one coherent interaction or transcript episode.
- fact: traceable, self-contained narrative evidence extracted from the episode.
- state: evolving long-term topic/preference/task state derived later from facts.
- index entry: a MemPalace-style directory card used for unified recall.

Your task now is to extract the episode summary and Hindsight-style high-quality narrative facts from the chronological assistant dialogue batch below.

Core Hindsight-style narrative fact requirements:
- Each fact should cover a complete exchange or a clear topic segment, not a single utterance. Do not mechanically split "the user raised a problem", "the assistant suggested a solution", and "the user accepted/rejected it" into separate fragments; if they respond to the same issue, merge them into one narrative fact.
- Each fact must be understandable without reading the original dialogue and preserve the pragmatic flow of the interaction: why the user raised the issue, what the assistant suggested, how the user responded, and what preference, decision, constraint, unresolved question, or next step emerged.
- Each fact must naturally include the five dimensions in its text: what (complete event/topic/plan/conclusion), when (conversation timestamp or explicit time anchor), where (location/setting/platform/project scope; if absent, say no specific location/setting was mentioned), who (user, assistant, and other key people/organizations with their roles), and why (explicit reason, motivation, concern, disagreement, constraint, implication, conclusion, or follow-up).
- For a roughly five-turn dialogue batch, usually produce 1-3 facts. Only split when the batch truly contains multiple unrelated events/topics. In most cases, do not exceed 5 facts.

Rules:
1. Extract 0-5 facts. Do not force a fact for every turn.
2. Each fact must be a complete narrative preserving the essential topic background plus the flow of user/assistant views or actions and an explicit reason, disagreement, constraint, conclusion, or next step.
3. Preserve concrete answerable details: names, places, titles, colors, dates, weekdays, relative times, numbers, amounts, durations, products, organizations, recommendations, constraints, decisions, and user preferences.
4. Do not drop personal events mentioned as asides, e.g. "by the way", "I also", "I just", "last Saturday", "two months ago"; but if they are context inside the same exchange, merge them into the same narrative fact instead of emitting context-free short notes.
5. Split only truly unrelated events. Events that must be compared for temporal reasoning may be split, but every split fact must still preserve its own background and time anchor.
6. Use only the dialogue evidence. Do not invent completion, intent, or reasons.
7. Keep assistant recommendations that contain concrete future-answerable items inside the relevant exchange narrative, and include whether the user accepted, rejected, hesitated, or added constraints when supported.
8. priority is 0-100. Keep only facts worth at least 60.
9. fact_type must be semantic or episodic.
10. fact_subject must be user, assistant, world, project, system, or other.
11. fact_kind must be preference, decision, request, recommendation, action, commitment, open_question, risk, error, context, instruction, or other.
12. Do not output short facts like "the user said X" or "the assistant suggested Y". If deleting the topic background, reason, disagreement, or conclusion would make the text a vague short note, add those details back; if the dialogue does not support them, omit the fact.
13. Return JSON only. No markdown.

Output schema:
{
  "episode_title": "short concrete title",
  "episode_summary": "self-contained summary of the interaction episode",
  "facts": [
    {
      "text": "self-contained narrative fact covering a complete exchange and expressing what/when/where/who/why, with background, user/assistant view or action flow, and reason/disagreement/constraint/conclusion/next step",
      "keywords": ["keyword1", "keyword2"],
      "entities": [{"name": "entity name", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|TOPIC|PREFERENCE|OTHER"}],
      "primary_topic": "stable topic string",
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

Dialogue batch:
{dialogue_batch}
"""


UNIFIED_STATE_UPDATE_PROMPT_EN = """You are the state update module for a unified AI-glasses long-term memory system inspired by MemPalace.

The input contains newly stored narrative facts and the current long-term states. Your task is to update or create compact evolving states that help future recall. A state is not a copy of one fact. It should summarize a stable preference, ongoing project/task, recurring behavior, open commitment, important relationship, or topic-level situation across multiple facts.

Rules:
1. Create or update only states that are useful beyond the current episode.
2. Do not create a state from a single trivial fact unless it is a durable preference, commitment, decision, or important life/project context.
3. Merge facts about the same durable subject into one state instead of creating near-duplicates.
4. Preserve uncertainty and recent changes. If evidence conflicts, explicitly state the conflict.
5. Use evidence_fact_ids to cite the fact IDs that support the state.
6. state_type must be one of: preference, task_state, project_state, relationship, routine, topic_state, commitment, constraint, risk, profile, other.
7. importance is 0-1. confidence is 0-1.
8. Return JSON only. No markdown.

Output schema:
{
  "states": [
    {
      "state_type": "preference|task_state|project_state|relationship|routine|topic_state|commitment|constraint|risk|profile|other",
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


UNIFIED_ACTIONABLE_ITEM_EXTRACTION_PROMPT_EN = """You are the actionable-item extraction module for a unified AI-glasses long-term memory system inspired by MemPalace.

The input contains newly stored narrative facts. Extract concrete actionable items that may require future follow-up, recall, reminder, review, or decision tracking. Actionable items are separate from evolving states: a state summarizes an ongoing situation, while an actionable item is something that can be checked, completed, tracked, or explicitly recalled as a decision/commitment/risk/open question.

Extract only items directly supported by the facts.

Rules:
1. Extract 0-12 actionable items. Do not force an item.
2. Include decisions, commitments, tasks, follow-ups, open questions, risks/blockers, deadlines, constraints that affect action, and concrete assistant recommendations the user may later ask about.
3. Do not copy every fact. If a fact is merely background with no follow-up value, skip it.
4. Keep each item self-contained: include actor/owner, target object, context, reason, deadline/time if known, and current status.
5. evidence_fact_ids must cite supporting fact IDs from the input.
6. item_type must be one of: task, commitment, decision, follow_up, open_question, risk, reminder, recommendation, constraint, other.
7. status must be one of: open, in_progress, done, blocked, decided, noted, unknown.
8. importance and confidence are 0-1.
9. Return JSON only. No markdown.

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

New facts:
{facts}
"""


RECALL_QUERY_ANALYSIS_PROMPT_EN = """You are the recall query analyzer for a unified MemPalace-style memory system.

The memory system has one shared index over episodes, facts, evolving states, and actionable items. Analyze the user query and decide which parts of the index are most useful.

Guidance:
- Use source_types only when the query clearly points to assistant_wakeup interactions or allday_recording transcripts. Otherwise use both.
- Use index_levels to prefer fact for exact evidence, actionable_item for tasks/commitments/decisions/open questions/risks/reminders/recommendations, state for stable preferences/tasks/project status, and episode for broad summaries.
- Keep the plan broad when unsure. Missing evidence is worse than retrieving a few extra candidates.

Return JSON only:
{
  "source_types": ["assistant_wakeup", "allday_recording"],
  "index_levels": ["fact", "actionable_item", "state", "episode"],
  "needs_broad_evidence": false,
  "query_rewrite": "retrieval-focused rewrite",
  "keywords": ["keyword1", "keyword2"]
}

User query:
{query}
"""
