"""English prompt templates for the unified memory prototype."""

MEMORY_RETRIEVED_FORMAT_PROMPT_EN = """[Unified Memory]
System note: Recall may contain traceable fact evidence, user_claims about the wearer, and active goals, plans, or work items. user_claims and future-oriented objects are derived from and supported by facts; do not present them as unsupported new facts.
System note: For facts, dialogue_time is when the conversation/transcript discussed the fact, while event_time is when the real-world event described by the fact occurred. They are different fields; an unknown event_time must not be inferred from dialogue_time.
{memory_sections}"""

MEMORY_RETRIEVED_SECTION_SPECS_EN = (
    (
        "[Retrieved Facts]",
        "These are ranked narrative facts retrieved directly from memory_facts.",
        "fact",
    ),
    (
        "[User Claims]",
        "These are active user_claims about user self. Interpret them with their supporting facts, origin, status, and validity time.",
        "user_claim",
    ),
    (
        "[Active Goals]",
        "These are active goals; their current status and target time take precedence over older related facts.",
        "goal",
    ),
    (
        "[Active Plans]",
        "These are active plans. A plan must not be treated as already completed.",
        "plan",
    ),
    (
        "[Open Work Items]",
        "These are not-closed work items. Respect their responsible party, due time, and status.",
        "work_item",
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
- Conversation subjects or roles, such as user self, assistant, speaker_1, speaker_2, spouse, child, team, or client. `user self` is a reserved identity for the verified source speaker `speaker_用户本人`; a speaker label is not the output entity name.
- Users in a product, market, or service domain: use a specific label such as product user, target consumer, or target customer, never bare user. These are external domain roles, not the AI-glasses wearer, and normally use CONCEPT or OTHER rather than PERSON.
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
Each retained memory fact should usually include a subject entity, such as user self/assistant/speaker, plus 1-4 core semantic anchors."""


USER_ACTIVITY_EXTRACTION_PROMPT_EN = """You are the situational-observation module of a personal world model. Given chronological dialogue or ambient-transcript units and extracted facts, describe the user's activity or situation observable during the capture window. An activity is what the user was doing then, not what they reported doing earlier, usually do, or intend to do later.

First identify the interaction scene from unit types and chronology, then summarize its main discussion topics. Raw source_units are primary evidence that an activity occurred; evidence_facts only help identify topics and check details. Do not generate one activity per fact. The number of activities should reflect distinct observable scenes, not the number of facts.

Rules:
1. For interaction units, an ongoing conversation with the assistant is itself the observable activity. Usually summarize a continuous conversation as one activity such as "The user talked with the assistant, mainly about ...", with user_role participant. Mention a few representative topics, but do not turn each topic, fact, or behavior mentioned in the conversation into a separate activity. Split only when the raw units show a distinct activity, a new day or long interruption, or a clear scene change.
2. For transcript units, describe the discussion, meeting, or other situation actually taking place during recording. Use participant when the user demonstrably speaks or participates; use actor only when raw units directly establish that the user actually performed a non-conversational action during capture; use contextual when the recording only establishes exposure to a situation, not personal speech or action. Being mentioned by others does not prove participation or action.
3. Strictly distinguish observed activity from its subject matter. The activity time comes from source_units capture times, not a fact's event_time_key. If the user says "I exercised this morning", "I've been learning photography", or "I plan to visit a mountain tomorrow", the observable activity is talking about those matters, not exercising, taking photographs, or traveling during this window. Past events, habits, preferences, hypotheticals, advice, and future plans may be discussion topics but not separate actor activities. A clearly contemporaneous statement such as "I am cooking right now" can support an additional action if evidence is sufficient.
4. Each activity_text must first state the activity or situation occurring in the window, then concisely name its main topics. Preserve the distinction between discussing, recalling, planning, and actually doing. Do not invent motives, emotions, duties, outcomes, or locations. Wake words, greetings, templated assistant replies, and fragments without an identifiable scene may yield an empty list.
5. evidence_fact_ids may reference only supplied facts supporting the scene or its topics; a fact alone cannot establish that a mentioned behavior happened during the capture window. Raw units may independently support the observation, so the list may be empty. confidence is between 0 and 1.
6. Return JSON only.

For an interaction in which the user says they exercised this morning, saw a movie last night, and might visit a mountain tomorrow, output one activity such as: {"user_activities": [{"activity_text": "The user talked with the assistant about exercise, a movie, and plans for a mountain trip", "user_role": "participant", "confidence": 0.9, "evidence_fact_ids": []}]}. Do not output separate activities saying the user is exercising, watching a movie, or traveling.

Raw source_units:
{source_units}

Extracted evidence_facts:
{evidence_facts}
"""


UNIFIED_MEMORY_EXTRACTION_PROMPT_EN = """You are the memory extraction module for a unified AI-glasses memory system inspired by MemPalace.

Current memory structure:
- fact: a traceable, self-contained, independently retrievable narrative evidence unit extracted from a continuous evidence batch.
- user_activity: a grounded observation of user activity or surrounding context generated from a longer window of source units and facts by a separate module; do not output it here.
- memory_topic_items: a topic registry populated from facts. It records fact-topic mappings, not a topic summary or state; this prompt does not read existing topic names.
- user_claim: a durable proposition about user self produced by the user-claim extraction pipeline and traceable to supporting facts; relation objects may be other entities. It is not direct evidence from the current dialogue and does not replace facts.
- goal / plan / work_item: future desired outcomes, explicit arrangements, and closed-loop responsibilities extracted by a separate Intent & Execution task after facts are stored. Do not output them here.

Your task now is to extract Hindsight-style high-quality narrative facts from the chronological dialogue/transcript evidence-unit batch below. Each unit's `Unit type` only identifies its source: `interaction` is one ordered user-and-assistant exchange, while `transcript` is ambient-recording ASR text. Longer-window situational observations are generated separately; do not output user_activity here.

Identity boundary:
- `user self` refers only to the AI-glasses wearer. `speaker_用户本人` is a verified wearer speaker; confirmed names listed below may also refer to the same person when the current fact's context resolves that mention unambiguously. In Chinese output use `用户本人`; `speaker_助手` corresponds to the assistant. Never merge a namesake with the wearer.
- Known aliases help identity resolution but are not evidence of a new fact. Propose a new alias only from the wearer's self-identification, a direct address followed by the wearer's clear response, or a third-party mention marked as an unconfirmed candidate. Never treat generic user, product users, or speaker labels as names.
- For unverified ambient speaker labels such as N_SPKxxx or speaker_1, never infer that a speaker is user self from words such as I, we, or user. Keep speakers, team members, and external objects under their actual names or speaker labels.
- In product, market, or service discussions, user, customer, consumer, and audience are external domain roles. Prefer product user, target consumer, or target customer; never emit bare user, and never classify it as PERSON, user self, or a future-commitment owner.
- A short reply from `speaker_用户本人` can confirm or correct another speaker's specific earlier statement in this same evidence batch. Preserve both the earlier statement and the wearer's actual response in the same fact. A mere acknowledgment such as "uh-huh" or "got it" does not establish agreement or responsibility.

Preceding-context rules:
- The preceding context below is a short tail from the already processed batch immediately before the current batch. Use it only to resolve references, ellipsis, continuity, and the starting point of the current topic.
- Only the current dialogue/transcript evidence batch may directly support a new fact. Do not generate, complete, or confirm a fact from preceding context alone, and do not repeat a fact solely because it appeared there.
- When the current batch conflicts with preceding context, prefer the current batch. If the current batch still cannot disambiguate or verify a statement, omit the fact.

Topic and entity rules:
- `fact_topic` names the specific aspect discussed by this fact and must be grounded in the current evidence. Use a short, specific, searchable topic, not a broad domain, full factual sentence, action, one-off conclusion, emotion, or isolated keyword. Name it conservatively when evidence is limited.
- Every fact must also output one `primary_entity`, the single entity that the fact mainly describes, affects, or belongs to; it must be one object, not an array.
- `primary_entity` must come from the fact's `entities`. Do not choose an entity merely because it is mentioned, provides a recommendation, or is a location, tool, or background context. For a multi-person exchange, choose the person or entity mainly described or affected by the fact; for a fact about the wearer's own preference, habit, constraint, or risk, choose user self only when the current evidence identifies `speaker_用户本人` or clearly refers to a confirmed wearer alias.
- Keep `entities` for all directly relevant entities. When a previously confirmed alias or a newly confirmed alias clearly refers to the wearer in this fact, include both the mentioned name and user self in `entities`; do not do this for a namesake. A third-party statement must remain attributed to its speaker, not rewritten as the wearer's own statement.

""" + ENTITY_EXTRACTION_GUIDANCE_EN + """

Core Hindsight-style narrative fact requirements:
- Each fact should cover a complete exchange or a clear topic segment, not a single utterance. Do not mechanically split "the user raised a problem", "the assistant suggested a solution", and "the user accepted/rejected it" into separate fragments; if they respond to the same issue, merge them into one narrative fact.
- Each fact must be understandable without reading the original dialogue and preserve the pragmatic flow of the interaction: why the user raised the issue, what the assistant suggested, how the user responded, and what preference, decision, constraint, unresolved question, or next step emerged.
- Each fact must naturally include the five dimensions in its text: what (complete event/topic/plan/conclusion), when (conversation timestamp or explicit time anchor), where (location/setting/platform/project scope; if absent, say no specific location/setting was mentioned), who (user, assistant, and other key people/organizations with their roles), and why (explicit reason, motivation, concern, disagreement, constraint, implication, conclusion, or follow-up).
- For a roughly five-turn dialogue batch or a coherent multi-speaker transcript segment, usually produce 1-3 facts. Only split when the batch truly contains multiple unrelated events/topics. In most cases, do not exceed 5 facts.

fact_type classification:
- `fact_type` describes the fact's primary semantic role. It must be one of preference, decision, request, recommendation, action, commitment, open_question, risk, error, context, instruction, or other.
- Choose the one category that best represents the fact's memory value. Do not use `open_question` merely because the assistant did not answer or the user's wording is vague.

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
9. fact_type must be preference, decision, request, recommendation, action, commitment, open_question, risk, error, context, instruction, or other.
10. Do not output short facts like "the user said X" or "the assistant suggested Y". If deleting the topic background, reason, disagreement, or conclusion would make the text a vague short note, add those details back; if the dialogue does not support them, omit the fact.
11. Do not store assistant pleasantries, generic closings, or low-information encouragement as standalone facts, e.g. "hope this helps", "let me know if you have other questions", "okay", or "you're welcome", unless they explicitly change a decision, commitment, or next step.
12. keywords must be short retrieval terms: entities, topics, symptoms, plans, constraints, decisions, and important time/order anchors. For time-sensitive facts, include the original or resolved time phrase such as "March 15 2023", "first service", "3/22", "last Saturday", or "two months ago". Do not put full sentences, pleasantries, filler, generic encouragement, or phrases like "hope this method helps you" into keywords.
13. Return JSON only. No markdown.

user_claim_signals rules:
- `user_claim_signals` are evidence-routing hints only for explicit user-claim extraction. They are not final claims and must not decide relations to existing claims; they are not used to induce claims from repeated events. Claims about other people, organizations, product users, or ambient speakers must not be routed here.
- Emit a signal only when (1) `speaker_用户本人` directly states the proposition, (2) the wearer clearly confirms or corrects another speaker's specific proposition, or (3) a third party explicitly describes the wearer through a confirmed alias. The third case is a `third_party_report`, not a wearer assertion; a namesake, unrelated wearer utterance, or vague acknowledgment is insufficient.
- `owner_evidence_type` is `self_statement|confirmed_response|third_party_report`. The first two require an exact wearer quote in `owner_evidence_text`; the response case also requires an earlier other-speaker quote. A third-party report has empty `owner_evidence_text` and an exact other-speaker quote containing the confirmed alias in `other_evidence_text`. Never invent quotes or quote preceding context.
- `subject_entity` must always be `user self`, never the other end of a relationship. A relationship involving user self may be signalled; its other person, organization, or project remains a fact entity and may later be the claim's object_entity.
- Emit `user_claim_signals` only when this fact explicitly states or clearly confirms a proposition about user self; do not place observations intended for cross-event induction here. A specific standalone action belongs in `user_action_signals`, not duplicated as a user-claim signal.
- `claim_type_hint` must be one of: identity_profile, affiliation, relationship, preference, constraint, behavior_pattern. Use `behavior_pattern` only when the wearer explicitly states or confirms a routine or regular behavior, such as "I run every Wednesday"; never infer it from a sequence of separate observed actions.
- `claim_anchor` is a short label identifying the subject of an explicit claim, such as "quiet travel preference" or "Wednesday running routine". It is not a key for accumulating behavioral samples, a full sentence, or the final user_claim.
- Output signals only when this fact has concrete evidence for an explicit claim about user self. Use an empty array for one-off background, claims about third parties themselves, temporary suggestions, assistant speculation, patterns inferred only from repeated actions, pleasantries, and low-value content.
- Signal-to-fact grounding is mandatory: the signal's subject, proposition, and necessary qualifiers must be fully supported by the same fact's `text`. A response-based signal must preserve both the earlier statement and wearer response there. Do not borrow information omitted from this fact from another fact or preceding context. `evidence_basis` must explain how the wearer's utterance supports the final proposition; otherwise omit the signal.
- Return at most 3 signals per fact. Every signal must contain subject_entity, claim_type_hint, claim_anchor, owner_evidence_type, owner_evidence_text, other_evidence_text, evidence_basis, and confidence.

user_action_signals rules:
- `user_action_signals` identify a specific completed or ongoing action, real choice, or substantive participation by user self in this fact. They are not habits, preferences, or observed patterns. Discussing an activity is not performing it; the separate user_activity module summarizes the capture-window situation.
- `subject_entity` is always `user self` and must be in this fact's `entities`. Never substitute a product user, another person, or an unverified ambient speaker for the wearer.
- `action_kind` is `action|choice|participation`: concrete performance, a choice among real options, or actual participation in a meeting/activity. `action_text` must specify what the wearer did and any supported time/object; this same fact's `text` must directly support it. Do not borrow content from other facts, preceding context, or general knowledge.
- A wearer report of a past action, a clear confirmation that the wearer did something, or a third-party report of a completed wearer action through a confirmed alias may qualify. Use `owner_evidence_type=self_statement|confirmed_response|third_party_report` and the same exact-quote rules as user_claim_signals. Do not turn a third-party report into a wearer assertion.
- Return an empty array for future plans, suggestions, hypotheticals, merely asking or discussing, and assistant/third-party actions. "I run every Wednesday" is a reported routine, not multiple observed actions; emit a signal only if the text separately establishes a specific occurrence. Ordinary wearer–assistant conversation itself is not an action signal.
- Return at most 3 signals per fact. Each has subject_entity, action_kind, action_text, owner_evidence_type, owner_evidence_text, other_evidence_text, evidence_basis, and confidence. evidence_basis must identify the supporting content in this fact. The signal inherits the fact's event_time_key; do not output another time field.

user_intent_signals rules:
- `user_intent_signals` are structured hints that this fact is evidence about the wearer's future world. They are not final goals, plans, or work items, and must not directly create, update, or override an existing object.
- Record only user self's future-facing information: `subject_entity` must always be exactly `user self` (use `用户本人` in Chinese output). Never use another person, team, organization, product user, or ambient-recording speaker as the subject.
- Output a signal when the wearer directly expresses or clearly confirms a future goal, arrangement, responsibility, or lifecycle change. A third party may also explicitly report an arrangement or responsibility concerning the wearer through a confirmed alias; mark this `third_party_report` and never turn an assignment, guess, or suggestion into the wearer's own commitment. Preserve speaker attribution in this fact.
- `assertion_source` also identifies the evidence source: `self_statement|confirmed_response|third_party_report|observed_event`. Wearer statements and confirmations require an exact wearer quote in `owner_evidence_text`; confirmations also need the earlier other-speaker quote. Third-party reports have empty `owner_evidence_text` and an exact quote containing the confirmed alias in `other_evidence_text`. Use `observed_event` only for an objective event explicitly recorded in the same fact, with both quote fields empty.
- `evidence_kind` must be one of goal, plan, responsibility, or lifecycle_update. `candidate_object_types` may contain only goal, plan, and work_item. A third-party report about a goal or responsibility is only a hint, not proof that the wearer set the goal or accepted a commitment.
- `user_role` must be owner, participant, or responsible: user self respectively owns the goal, participates in the arrangement, or bears the responsibility. Do not output a signal for a product user, customer, consumer, or an ambient speaker merely mentioned in the evidence.
- `explicitness` must be direct, reported, or tentative. For a third-party report use `assertion_source=third_party_report` and `explicitness=reported`. When the wearer accepts another person's proposal, use `assertion_source=confirmed_response`. Assistant suggestions, open hypotheticals, and speculation do not themselves create signals.
- `user_intent_anchor` is a short, stable grouping label for the goal, event, or responsibility, such as "Tianjin client meeting", "half-marathon training goal", or "client report delivery". It is not a full sentence or the final object description.
- Signal-to-fact grounding is mandatory: the future goal, arrangement, responsibility, or lifecycle change itself must be explicitly retained in the same fact's `text`. Do not attach information found elsewhere in the raw unit batch to a fact that does not preserve it; output an empty array when the fact text cannot independently support the signal.
- Return at most 2 signals per fact. Every signal must include subject_entity, evidence_kind, candidate_object_types, user_role, user_intent_anchor, assertion_source, explicitness, owner_evidence_text, other_evidence_text, evidence_basis, and confidence. evidence_basis must be concrete evidence directly locatable in this fact's `text`.

Output schema:
{
  "facts": [
    {
      "text": "self-contained narrative fact covering a complete exchange and expressing what/when/where/who/why, with background, user/assistant view or action flow, and reason/disagreement/constraint/conclusion/next step",
      "keywords": ["keyword1", "keyword2"],
      "entities": [{"name": "entity name", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|TOPIC|PREFERENCE|OTHER"}],
      "primary_entity": {"name": "the single primary entity of this fact", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|TOPIC|PREFERENCE|OTHER"},
      "fact_topic": "specific aspect discussed by this fact",
      "fact_type": "preference|decision|request|recommendation|action|commitment|open_question|risk|error|context|instruction|other",
      "priority": 80,
      "event_time_key": "real-world event occurrence time or representative temporal anchor derived from the dialogue time anchor and fact content; empty when it cannot be determined",
      "time_confidence": "explicit|inferred_from_turn|unknown; explicit evidence, resolved from the segment Time and a relative expression, or undetermined",
      "where": "",
      "world_owner_alias_candidates": [{
        "alias_name": "name in this fact's entities",
        "evidence_type": "self_identification|direct_address|third_party_reference",
        "evidence_text": "exact current-batch quote containing the name",
        "owner_response_text": "exact subsequent wearer reply for direct_address; otherwise empty",
        "confidence": 0.8
      }],
      "user_claim_signals": [
        {
          "subject_entity": "user self",
          "claim_type_hint": "identity_profile|affiliation|relationship|preference|constraint|behavior_pattern",
          "claim_anchor": "short label identifying the explicit claim subject",
          "owner_evidence_type": "self_statement|confirmed_response|third_party_report",
          "owner_evidence_text": "exact wearer quote for self/response; empty for third-party report",
          "other_evidence_text": "exact other-speaker quote for response/report; otherwise empty",
          "evidence_basis": "specific evidence from this fact supporting the signal",
          "confidence": 0.8
        }
      ],
      "user_action_signals": [
        {
          "subject_entity": "user self",
          "action_kind": "action|choice|participation",
          "action_text": "one concrete completed or ongoing wearer action, choice, or participation",
          "owner_evidence_type": "self_statement|confirmed_response|third_party_report",
          "owner_evidence_text": "exact wearer quote for self/response; empty for third-party report",
          "other_evidence_text": "exact other-speaker quote for response/report; otherwise empty",
          "evidence_basis": "specific evidence in this fact supporting this occurrence",
          "confidence": 0.8
        }
      ],
      "user_intent_signals": [
        {
          "subject_entity": "always `user self`; use `用户本人` in Chinese output",
          "evidence_kind": "goal|plan|responsibility|lifecycle_update",
          "candidate_object_types": ["goal|plan|work_item"],
          "user_role": "owner|participant|responsible",
          "user_intent_anchor": "a stable label for grouping the goal, arrangement, or responsibility",
          "assertion_source": "self_statement|confirmed_response|third_party_report|observed_event",
          "explicitness": "direct|reported|tentative",
          "owner_evidence_text": "exact wearer quote for self/response; empty for third-party report",
          "other_evidence_text": "exact other-speaker quote for response/report; otherwise empty",
          "evidence_basis": "concrete evidence from this fact supporting the signal",
          "confidence": 0.8
        }
      ]
    }
  ]
}

Confirmed user-self aliases (identity resolution only, not current fact evidence):
{known_world_owner_aliases}

Preceding context (understanding aid only; not independent evidence for this fact task):
{preceding_context}

Dialogue/transcript evidence batch:
{dialogue_batch}
"""

INTENT_EXTRACTION_PROMPT_EN = """You extract explicit user_intents for a personal world model.

1. Objective
Review each input `user_intent_signal` together with its own fact. Determine whether the fact directly supports one future-facing goal, established arrangement, or bounded responsibility of user self. Return one `intent_candidate` when supported, or null when it is not. Do not infer an intent by combining multiple signals or decide whether a new candidate matches an existing object.

2. Meaning of user_intent
A `user_intent` is a trackable, updateable future object, not a verbatim restatement of the fact. The fact preserves the statement and its context. Only these object types are allowed:
- `goal`: an explicit, enduring desired outcome spanning multiple actions; a one-off wish or single action is not a goal.
- `plan`: an established future arrangement, trip, meeting, activity, or event involving user self, supported by a time/window or explicit arranged, agreed, or registered evidence; it is not a to-do.
- `work_item`: a bounded action for which user self is responsible, with at least one completion anchor beyond the action: a deliverable, deadline, explicit commitment, or assignment explicitly accepted by user self. It may be personal_action, commitment, assigned, or external_commitment.
Never derive a work item automatically from a goal or plan. A corresponding signal may yield a work item only when the fact independently supports a next action, deliverable, commitment, or assignment.

3. Meaning of the input user_intent_signal
Each item has its own `signal_id`, one `user_intent_signal`, and its associated fact. The same fact may appear under multiple signals, but remains one piece of evidence. The signal is a lead from fact extraction, neither an established user_intent nor new evidence beyond the fact: `evidence_kind` hints at goal, plan, responsibility, or lifecycle_update; `candidate_object_types` limits permitted output types; `subject_entity` and `user_role` hint at user self's role; `user_intent_anchor` locates the object; `assertion_source`, `explicitness`, and `evidence_basis` help inspect provenance. Its `confidence` is only a prior estimate. Verify the signal against the current fact's `summary`; do not copy its conclusion, use another item's evidence, or treat a repeated fact as independent evidence. Reference time only helps interpret relative times already present in the fact; it is not evidence of an arrangement or deadline.

4. Output rules
1. Return exactly one result for every input `signal_id`, with one candidate object or null. Return null when the signal and fact disagree, the fact is insufficient, or no qualifying future object exists. The candidate's `object_type` must be in that signal's `candidate_object_types`.
2. Use only direct evidence from the current fact. Assistant suggestions, speculation, behavior patterns, predictions, and unresolved hypotheticals such as maybe, if, should we, or consider cannot create an object. Another person's request does not assign responsibility to user self; the fact must preserve user self's clear acceptance, commitment, or correction. A mere acknowledgment or "got it" is insufficient.
3. `assertion_source=third_party_report` is a verifiable lead, not proof that user self set a goal or accepted a work item. A low-confidence plan is possible only when the fact clearly describes an established future arrangement; qualify its summary with the third-party source.
4. Make `summary` self-contained, including the object, user role, and material time or conditions. `evidence_fact_ids` must be exactly `[fact.id]` for the current item. Judge final confidence from the fact; do not copy the signal's confidence.
5. `lifecycle_evidence` describes only a change directly observed in the fact, not a database operation. A plan that happened is `occurred`; a finished work item is `completed`. A `lifecycle_update` may describe a change to an existing object, but later reconciliation must not create a new object if no matching existing object is found.
6. `primary_domain` must be one of `general|health|work|life|information|finance|family|learning|social|interest|travel`. Prefer the most specific user-life domain; reserve `general` for genuinely cross-domain objects, not uncertainty. Leave unsupported times and details empty. Return JSON only.

Reference time: {reference_timestamp}
Independent signal items (each has an integer `signal_id`, `fact`, and one `user_intent_signal`):
{signal_items}

Return JSON only:
{
  "results": [
    {
      "signal_id": 1,
      "intent_candidate": {
        "object_type": "goal|plan|work_item",
        "summary": "complete, user-facing object description",
        "canonical_key": "short stable identity key for matching objects of the same type",
        "primary_domain": "general|health|work|life|information|finance|family|learning|social|interest|travel",
        "subject_entity": "core subject: goal owner, plan actor, or work-item responsible party",
        "time_start": "start/available time, or empty when unsupported",
        "time_end": "target/end/due time, or empty when unsupported",
        "lifecycle_evidence": "type-specific lifecycle evidence value",
        "confidence": 0.0,
        "evidence_fact_ids": [1],
        "details": {}
      }
    }
  ]
}

Put type-specific structured fields only in `details`; do not duplicate them as top-level fields:
- goal: `desired_outcome` (required), `success_criteria`, `target_at`. `time_end` may also represent target_at.
- plan: `event_or_activity` (required), `time_precision`, `location`, `participants`.
- work_item: `action_text` (required), `deliverable`, `responsibility_type`, `priority`, `beneficiary_entities`, `delegator_entities`, `collaborator_entities`.

Lifecycle values: goal: `none|confirmed|completed|cancelled|blocked`; plan: `none|confirmed|occurred|cancelled|rescheduled|blocked`; work_item: `none|confirmed|completed|cancelled|blocked`. For an unqualified item, still return its signal_id with `intent_candidate: null`. Return JSON only."""


INTENT_RECONCILIATION_PROMPT_EN = """Decide only the relationship between new Intent & Execution candidates and existing objects. Do not invent facts or modify object fields.

For each candidate, choose a matching existing object only when it represents the same goal, plan, or work item with compatible subject, core activity/deliverable, and time. Return create when no reliable match exists. Plan `occurred` means the event happened; work_item `completed` means the responsibility was fulfilled.

Return JSON only:
{
  "decisions": [
    {
      "candidate_index": 0,
      "operation": "create|confirm|update|complete|cancel|reschedule|block",
      "target_object_type": "goal|plan|work_item",
      "target_object_id": 0,
      "reason": "brief evidence-grounded reason"
    }
  ]
}

candidates:
{candidates}

existing objects:
{existing_objects}
"""


EXPLICIT_USER_CLAIM_EXTRACTION_PROMPT_EN = """You extract explicit user_claims for a personal world model.

1. Objective
Review each input `user_claim_signal` together with its corresponding fact. Determine whether that fact directly supports an explicit proposition about user self. Generate a `user_claim` when it does; otherwise return empty `claims`. Do not infer patterns from observed behavior, generate future goals or to-dos, or decide how new claims relate to existing claims.

2. Meaning of user_claim
A `user_claim` is a complete proposition about user self, explicitly stated or clearly confirmed by the user and traceable to a fact. It may describe the user's identity, affiliation, other relationships, preferences, constraints, or a continuing behavior the user has described. It is not a paraphrase of the fact: the fact preserves the original event and conversational context, while the claim extracts a reusable proposition that can be compared with later evidence. Each claim must express one central assertion, including its subject, attribute or relationship, concrete content, and any necessary object, time, or condition. Reusable does not mean permanently true; do not turn a conditional or temporary statement into an unconditional permanent attribute.
Allowed `claim_type` values are:
- `identity_profile`: an identity, background, or role explicitly described by the user;
- `affiliation`: the user's membership or affiliation with an organization, team, project, or place;
- `relationship`: another explicit relationship between the user and a person, organization, or project;
- `preference`: a clearly expressed liking, dislike, or priority;
- `constraint`: a clearly expressed restriction, prohibition, capability boundary, or continuing condition;
- `behavior_pattern`: a continuing behavior or routine explicitly stated or confirmed by the user, such as "I run every Wednesday".
A single event or action is not a continuing-behavior claim. Patterns inferred from repeated observed actions belong to `user_observed_pattern`. Future trips, plans, commitments, and to-dos belong to facts or user_intents. Assistant guesses, suggestions, and unconfirmed third-party statements are not explicit user_claims.

3. Meaning of the input user_claim_signal
Each input item has its own `signal_id`, one `user_claim_signal`, and its source fact. The same fact may appear in several items, but it remains one piece of factual evidence. The signal is a candidate hint produced during fact extraction, not a final claim or new evidence beyond the fact: `claim_type_hint` suggests a possible type, `claim_anchor` points to the topic to verify, `evidence_basis` helps locate the evidence, and the signal's `confidence` is only the earlier stage's estimate. Verify the signal against the current item's fact `summary`. Do not copy the signal's judgment, borrow facts from another item, or count a repeated fact as multiple independent observations.

4. Output rules
1. Return exactly one result for every input `signal_id`; a signal may yield zero or several distinct claims. Return empty `claims`, rather than omitting the result, when evidence is insufficient, the signal conflicts with its fact, or the fact contains no qualifying explicit proposition.
2. An explicit proposition must come from the user's own direct statement or their clear confirmation, denial, or correction of another person's specific statement. A response-based fact must preserve both the earlier statement and the user's final position. A third-party report alone, vague acknowledgment, one-off action, or assistant speculation cannot establish a claim.
3. Each claim's `subject_entity` must equal the non-empty `world_owner_entity` (user self) in the current fact. Never use another person, product user, or ambient speaker as the subject. A non-empty `object_entity` must occur in the same fact's `entities`. Return empty `claims` if the user's identity is not explicit in the fact.
4. Every claim must be directly supported by the current fact's `summary`, and its `evidence_fact_ids` must be exactly `[fact_id]` from that item. Do not cite another fact, even if its content is similar.
5. `claim_text` must be a complete, self-contained statement that preserves any object, scope, time, or condition that changes the meaning; do not combine unrelated propositions. Use short, stable lowercase English predicates such as `has_role`, `member_of`, `prefers`, `dislikes`, `requires`, `cannot`, or `has_routine`. Express negation in both predicate and claim_text; do not add a separate polarity field.
6. `primary_domain` must be one of `general|health|work|life|information|finance|family|learning|social|interest|travel`; choose the most specific domain. Fill `valid_from`/`valid_to` only when the fact explicitly establishes the validity period; do not guess. Judge the claim's `confidence` independently from how well the current fact supports the final proposition; do not copy the signal's confidence.
7. Review each item independently even when several items share a fact. Different signals may output the same proposition; the system will deduplicate it later. Return JSON only, without markdown.

Output:
{
  "results": [{
    "signal_id": 1,
    "claims": [{
      "subject_entity": "",
      "claim_type": "identity_profile|affiliation|relationship|preference|constraint|behavior_pattern",
      "primary_domain": "general|health|work|life|information|finance|family|learning|social|interest|travel",
      "predicate": "",
      "object_entity": "",
      "claim_text": "complete proposition with subject and meaning",
      "valid_from": "",
      "valid_to": "",
      "evidence_fact_ids": [1],
      "confidence": 0.85
    }]
  }]
}

signal_items:
{signal_items}
"""


USER_OBSERVED_PATTERN_EXTRACTION_PROMPT_EN = """You conservatively extract user_observed_patterns about the user for a personal world model.

Input is a set of semantically similar user_action_signals, including new and historical signals. Each signal includes its fact, event time, and evidence source. Retrieval only established action similarity; it did not prove that each signal describes a separate event. Decide whether the evidence supports a behavioral pattern and its demonstrated conditions.

A user_observed_pattern describes a behavioral pattern inferred from concrete action evidence. It is not an explicit claim by the user and does not imply endorsement. A habit explicitly stated by the user belongs in explicit user_claims.

Hard rules:
1. A pattern requires at least three distinct support_signal_ids, each directly supporting it, including at least one new_signal_id.
2. Repeated mentions of one event in different facts do not count as repeated behavior. A single event, plan, task, or assistant suggestion is insufficient. If repeated behavior is not established, return an empty array.
3. Do not infer preferences from isolated behavior or substitute conversation or recording time for event time.
4. subject_entity must equal the given user-self subject and `world_owner_entity`.
5. pattern_text is a complete reader-facing behavioral pattern description with its subject.
6. `primary_domain` must be one of `general|health|work|life|information|finance|family|learning|social|interest|travel`; choose the most specific user-life domain, and use `general` only for genuinely cross-domain patterns.
7. Day-of-week, time-of-day, place, and other conditions require support from multiple relevant signals. Do not invent a periodic schedule when event times are insufficient. Evidence IDs must be input IDs. An empty list is correct. Return JSON only.

Output:
{
  "patterns": [{
    "subject_entity": "",
    "primary_domain": "general|health|work|life|information|finance|family|learning|social|interest|travel",
    "pattern_text": "complete behavioral pattern description with its subject",
    "condition_text": "",
    "support_signal_ids": [1, 2, 3],
    "confidence": 0.8
  }]
}

Pattern extraction target and new signals:
{pattern_target}

Similar action signals and their facts:
{action_signals}
"""


USER_OBSERVED_PATTERN_RECONCILIATION_PROMPT_EN = """You classify the semantic relationship between a new user_observed_pattern and existing patterns.

Compare the complete pattern_text, condition_text, and supporting evidence. Pay close attention to weekday, time of day, location, conditions, and scope. Similar actions are not necessarily the same pattern.

semantic_relation must be:
- duplicate: the same behavioral pattern under the same conditions despite different wording;
- supports: new evidence supports the existing pattern without changing its conditions;
- refines: the new pattern adds a supported condition or narrows the existing scope;
- contradicts: the patterns cannot both hold under the same conditions;
- unrelated: different patterns or insufficient evidence to determine a relation.

The candidate may relate to more than one existing pattern. A shared action or signal is not enough for duplicate. Return an empty relations list when no clear relation exists. Return JSON only.

Output:
{
  "relations": [{
    "existing_pattern_id": 1,
    "semantic_relation": "duplicate|supports|refines|contradicts|unrelated",
    "confidence": 0.8,
    "reason": "brief explanation"
  }]
}

New pattern:
{candidate_pattern}

Existing patterns:
{existing_patterns}
"""


USER_OBSERVED_PATTERN_SIGNAL_RELATION_PROMPT_EN = """Determine how each new user_action_signal relates to each retrieved user_observed_pattern. Similarity retrieval provides candidates, not proof. Each pattern includes representative historical action evidence and event times.

Classify a signal-pattern pair as:
- supports: a new behavior instance under the pattern's actual conditions. Similar action words alone, or a retelling of the same event, are not new support.
- refines: evidence suggests the pattern's scope or conditions may need narrowing. A single exception rarely overturns a general pattern.
- contradicts: evidence clearly conflicts under the same conditions. Different times, places, or circumstances need not conflict; one isolated action usually does not refute a habit.
- unrelated: different behavior, insufficient evidence, or a duplicate account of one event.

Use only the supplied evidence. Never infer event time from conversation time. One signal may support multiple distinct patterns. Include only pairs with a meaningful relation; an empty list is valid. Return JSON only.

Output:
{
  "decisions": [{
    "signal_id": 1,
    "existing_pattern_id": 2,
    "semantic_relation": "supports|refines|contradicts|unrelated",
    "confidence": 0.8,
    "reason": "brief explanation"
  }]
}

New action signals and their facts:
{new_signals}

Existing patterns and representative evidence:
{existing_patterns}
"""


DERIVED_USER_CLAIM_EXTRACTION_PROMPT_EN = """You produce directly derived user_claims about user self for a personal world model.

The input has two sections. `changed_explicit_claims` are explicit user_claims for the current subject that changed after this incoming fact batch. `related_active_explicit_claims` are historical active explicit user_claims loaded from the database because they concern this subject or its directly linked entities. Produce only conclusions that follow directly and necessarily from those claims.

A derived user_claim is not pattern induction, common-sense completion, or a plausible guess. It must be explainable as “because premise A (and premise B), conclusion C,” while keeping user self as the subject of both premises and conclusion. If the available user_claim premises do not entail a new user-self claim, return an empty list. The system assigns `claim_origin=derived`; do not include that field in the output.

Rules:
1. Use only the supplied premise_claim_ids; do not use external knowledge or unstated background.
2. Every candidate must cite at least one premise from `changed_explicit_claims`, and every cited ID must be in the input.
3. Only affiliation, relationship, and constraint claim types are allowed. Never create identity_profile, preference, behavior_pattern, goals, plans, work items, personality labels, or risk judgments.
4. subject_entity_id must be the user-self entity ID appearing in the input; never use a third party as subject. object_entity_id (0 means no object), when present, must be an entity ID appearing in the input. Never create an entity.
5. Use concise stable lowercase predicates and a complete, self-contained reader-facing claim_text.
6. Do not repeat or merely paraphrase a premise. Return an empty list when no strictly entailed new conclusion exists.
7. `primary_domain` must be one of `general|health|work|life|information|finance|family|learning|social|interest|travel`; choose the most specific user-life domain, and use `general` only for genuinely cross-domain conclusions.
8. confidence is confidence that the conclusion follows given the premises, not confidence that the premises are true. Return JSON only.

Output:
{
  "claims": [{
    "subject_entity_id": 0,
    "claim_type": "affiliation|relationship|constraint",
    "primary_domain": "general|health|work|life|information|finance|family|learning|social|interest|travel",
    "predicate": "",
    "object_entity_id": 0,
    "claim_text": "complete self-contained derived conclusion",
    "premise_claim_ids": [1, 2],
    "confidence": 0.8
  }]
}

changed explicit user_claims:
{changed_claims}

related active explicit user_claims:
{related_claims}
"""


USER_CLAIM_RECONCILIATION_PROMPT_EN = """You classify semantic relationships between user_claims in a personal world model.

The input contains only the natural-language text of new candidate user_claims and existing user_claims. Determine their semantic relationship from text alone. Do not infer source reliability, claim origin, confidence, time, database status, or any write strategy.

All user_claim subjects are user self; relation objects may be other entities. Never mistake an object entity for the claim subject.

semantic_relation is one of:
- duplicate: the texts state the same proposition;
- supports: the candidate directly supports or repeats the existing claim without identical wording;
- contradicts: the texts cannot both be true, but the candidate does not explicitly replace or update the existing claim;
- refines: the candidate adds a condition, scope, exception, or more specific form;
- supersedes: the candidate text explicitly says that the old proposition changed, was negated, corrected, or replaced;
- unrelated: both can be true or no clear relation exists.

Different preferences, roles, or relationships are not inherently contradictory: liking painting and liking piano are normally unrelated. One candidate can relate to multiple existing claims; list every clearly related existing claim. Return an empty relations array when none is related. Return JSON only.

Output:
{
  "decisions": [{
    "candidate_claim_index": 0,
    "relations": [{
      "existing_claim_id": 12,
      "semantic_relation": "duplicate|supports|contradicts|refines|supersedes",
      "confidence": 0.9,
      "reason": "short explanation"
    }]
  }]
}

candidate_claims:
{candidate_claims}

existing_claims:
{existing_claims}
"""


USER_CLAIM_STATE_RESOLUTION_PROMPT_EN = """Jointly propose old-claim status decisions and user-state updates from changed user claims. Creating a state does not require an existing state or a claim relation event.

Inputs: changed_claims are direct evidence for this update; related_claims are existing claims reached through relation events; supporting_claims provide additional evidence for existing states. existing_states and state_claim_evidence describe current states and their support. The relations in relation_events are already classified; do not classify them again. An empty history list does not prohibit state creation.

Reason in this order, then return one joint result:
1. Independently assess new states. Read every changed claim and combine evidence concerning the same state dimension. Ask whether it supports a durable preference, stable constraint, or behavioral pattern useful for future prediction or service decisions. Do this even when existing_states=[] and relation_events=[]; emit create when evidence suffices rather than returning empty operations automatically. One direct, unambiguous explicit claim with durable meaning may establish an enduring_state, such as "I have always preferred morning meetings" establishing a time preference; multiple claims are not mandatory. Related concrete preferences may be aggregated, but do not mix unrelated subjects merely to create an aggregate. State text should express a reusable judgment, not paste or closely paraphrase claims.
2. Distinguish durable tendencies from repetition. enduring_state covers preferences, styles, values, or constraints that broadly persist across time or situations. An explicit stable preference is not a one-off event merely because it was said once. recurring_state requires an explicitly stated habit/rule or multiple observations supporting periodic, frequent, repeated, or conditional behavior. Liking an activity, planning it once, or remembering an experience does not prove repetition. A one-time goal, plan, trip, recent emotion, or concern about ability does not automatically become a durable state. Create only enduring_state or recurring_state from claims, not situational_state.
3. Align with existing states. If an existing state with the same state_domain, state_type, state_kind, and state_topic already expresses the conclusion, reinforce or revise it rather than creating a duplicate. If new evidence challenges a state, examine state_claim_evidence, state_pattern_evidence, supporting_claims, and supporting_patterns before weakening, expiring, or partially revising it. Losing one support claim does not necessarily invalidate the state. Do not weaken or expire a recurring_state solely because a claim changed while an observed pattern still supports it. Do not abandon a justified create decision merely because no old state is affected.
4. Decide old-claim status. Only a contradicts or supersedes event with an old target claim and a changed trigger claim may justify an old-claim update. The relation alone is not a status decision: consider text, origin (explicit is usually more direct than derived), time, and conditions. Contradicts may at most weaken; only an unambiguous supersedes may supersede. Claims valid under different conditions or times may coexist. Do not alter changed claims or invalidate an old claim merely because an aggregate state fails. If there is no eligible relation, claim_updates=[]; this does not prevent state operations.
5. Check joint consistency. A claim proposed as superseded cannot remain effective support for a state. Omit individual state candidates lacking evidence, but do not discard other supported states in the same update.

State operations are create|reinforce|revise|weaken|expire. Existing-state updates must use its id; create uses target_state_id=0. Controlled enduring_state kinds: communication_style|decision_style|value_priority|risk_preference|privacy_preference|time_preference|work_style|lifestyle_preference|long_term_constraint. Controlled recurring_state kinds: daily_routine|weekly_routine|work_routine|health_routine|social_routine|conditional_behavior. state_kind is a category, not a unique state identity; distinct states in one category may coexist. On create, also output a short, stable state_topic for the specific subject, such as weekday_gym or evening_walk. If new evidence or a paraphrase concerns an existing state, update its target_state_id instead of creating another topic. Preserve an existing state's state_type, state_kind, and state_topic; stay within state_domain.

Each state operation cites input claims only, with evidence_type=user_claim, evidence_id, role=change_trigger|support|counterexample|context, and weight. Cite at least one changed claim as the trigger. A claim proposed as superseded cannot remain support in this update. Counterexamples must not be mislabeled as support. Include concise reasons and confidence in [0,1]. Return an empty operations array only after assessing every changed claim and finding no justified operation. Return JSON only:
{
  "claim_updates": [{"claim_id": 1, "new_status": "weakened|superseded", "relation_event_id": 1, "reason": "why this evidence changes the old claim"}],
  "operations": [{
    "operation": "create|reinforce|revise|weaken|expire",
    "target_state_id": 0,
    "state_type": "enduring_state|recurring_state",
    "state_kind": "controlled option",
    "state_topic": "stable topic for this specific state",
    "state_text": "complete aggregate user-state statement",
    "conditions": {}, "valid_from": "", "valid_to": "", "confidence": 0.8,
    "evidence": [{"evidence_type": "user_claim", "evidence_id": 1, "role": "change_trigger|support|counterexample|context", "weight": 1.0}],
    "reason": "why the state changes"
  }]
}

reference_time: {reference_time}
state_domain: {state_domain}
changed_claims: {changed_claims}
related_claims: {related_claims}
supporting_claims: {supporting_claims}
relation_events: {relation_events}
existing_states: {existing_states}
state_claim_evidence: {state_claim_evidence}
state_pattern_evidence: {state_pattern_evidence}
supporting_patterns: {supporting_patterns}
"""


USER_ACTIVITY_STATE_UPDATE_PROMPT_EN = """Incrementally update situational_state from newly generated user activity observations.

This update is triggered by new activity. An activity describes what the user actually participated in during a continuous time window. Facts provide concrete evidence from that window. Related intents are currently active goals/plans/work items with direct evidence links to those facts; they are supplemental context only. For `user_role`, `actor` means evidence supports that the user is doing the activity, `participant` means the user is involved, and `contextual` is background only and must not be used to infer that the user is doing it. Do not copy an activity, fact, or intent into a state, and do not assume that mentioning a topic means the user is currently working on it.

Create or update only `situational_state`; do not create enduring_state or recurring_state here. Emit an operation only when evidence indicates that the user is in an unfinished phase, is continuing to handle an issue, is preparing for something, is subject to a current constraint, or has a time-bounded current concern/availability. Chit-chat, one-off Q&A, reminiscence, preference statements, completed activities, and one-time itineraries normally do not warrant a situational state. Return an empty operations list when evidence is insufficient.

Every operation must cite at least one `user_activity` item from this input. Facts may only be cited from `linked_facts`; goals/plans/work items may only be cited from `related_active_intents`. Related intents were prefiltered by the system because they share fact evidence with the activity; still decide whether they describe the same ongoing situation. Never cite an ID not supplied here. Activity is the trigger and situation anchor; do not infer that a completed activity is still current.

Choose `state_domain` only from `allowed_domains`. When updating an existing state, use its ID and preserve its state_domain, state_type, state_kind, and state_topic. `state_kind` must be one of: `current_focus`, `current_constraint`, `current_availability`, `current_energy`, `current_stress`, `current_social_engagement`, `current_preparation`. It is a category; different ongoing situations in one category may coexist. On create, output a short, stable `state_topic` for the specific situation. New evidence for an existing situation must update its target_state_id rather than create a duplicate. `state_text` must be a self-contained description of the current state, not a historical summary. Provide a plausible `valid_to` or ending condition when evidence supports it; do not invent a deadline. Allowed operations are create, reinforce, revise, weaken, expire; `target_state_id` is 0 for create.

Return JSON only:
{
  "operations": [
    {
      "operation": "create|reinforce|revise|weaken|expire",
      "target_state_id": 0,
      "state_domain": "one allowed_domains value",
      "state_type": "situational_state",
      "state_kind": "one controlled option",
      "state_topic": "stable topic for this specific situation",
      "state_text": "complete and explicit current-state statement",
      "conditions": {},
      "valid_from": "",
      "valid_to": "",
      "confidence": 0.8,
      "evidence": [
        {"evidence_type": "user_activity|fact|goal|plan|work_item", "evidence_id": 1, "role": "change_trigger|support|counterexample|context", "weight": 1.0}
      ],
      "reason": "brief reason why this situation supports the operation"
    }
  ]
}

reference_time:
{reference_time}

allowed_domains:
{allowed_domains}

runtime_context:
{runtime_context}

new_activities:
{activities}

linked_facts:
{facts}

related_active_intents:
{related_intents}

existing_situational_states:
{existing_states}
"""


USER_OBSERVED_PATTERN_STATE_UPDATE_PROMPT_EN = """Incrementally update recurring_state from the user's newly created or newly supported observed patterns.

A user_observed_pattern is inferred from independent observed actions, not self-report. supporting_actions are representative raw evidence. changed_patterns were created or gained support in this task; affected_patterns are older patterns touched by relation events. The relation events classify support, refinement, or contradiction. An existing state may already express the same routine based on an explicit user_claim: update that state instead of creating another state for a different evidence source.

Only update recurring_state in the given state_domain. Do not invent a weekday, frequency, time, or condition that the evidence does not establish. state_text should express a reusable behavioral tendency, not copy a pattern. Use target_state_id for an equivalent existing state; otherwise create a stable, specific state_topic. state_kind must be one of daily_routine|weekly_routine|work_routine|health_routine|social_routine|conditional_behavior.

New patterns or new support may create, reinforce, or revise a state. Weaken only when a relation event and concrete counterevidence justify it; a single exception rarely negates a repeated behavior. Do not expire states in this first version. Return an empty operations list when no reliable change is supported. Every operation must cite at least one changed_pattern or trigger_signal. Durable support should cite a user_observed_pattern ID; a raw counterexample may cite a user_action_signal ID. Counterexamples must not be labeled support. Cite only supplied IDs.

Return JSON only:
{
  "operations": [{
    "operation": "create|reinforce|revise|weaken", "target_state_id": 0,
    "state_type": "recurring_state", "state_kind": "health_routine",
    "state_topic": "stable identifier for the specific pattern",
    "state_text": "complete behavioral state", "conditions": {},
    "valid_from": "", "valid_to": "", "confidence": 0.8,
    "evidence": [{"evidence_type": "user_observed_pattern|user_action_signal", "evidence_id": 1, "role": "change_trigger|support|counterexample|context", "weight": 1.0}],
    "reason": "why this evidence changes the state"
  }]
}

reference_time: {reference_time}
state_domain: {state_domain}
changed_patterns: {changed_patterns}
affected_patterns: {affected_patterns}
relation_events: {relation_events}
trigger_signals: {trigger_signals}
existing_states: {existing_states}
state_evidence: {state_evidence}
supporting_claims: {supporting_claims}
"""


USER_PREDICTION_NEAR_TERM_PREPARATION_PROMPT_EN = """You are the prediction module of a personal world model. Given user goals, plans, or work items with a time anchor in the specified forecast window, forecast possible user preparation before that anchor. Recurring states are optional supporting evidence.

Predictions are testable hypotheses, not observed facts. Cite one input intent by object_type and id. Cite recurring_state IDs only when genuinely relevant; state_ids may be empty. Do not merely restate the intent, invent precise items, places, or times, or predict the assistant's actions. A deadline alone does not guarantee preparation. Return an empty list when evidence is insufficient. Each forecast window must be after the reference time and no later than that intent's anchor_at. At most 5 predictions.

Return JSON only:
{{"predictions":[{{"intent_type":"plan|work_item|goal","intent_id":1,"state_ids":[],"behavior_text":"The user may confirm the travel arrangement","window_start":"ISO datetime","window_end":"ISO datetime","conditions":["If the arrangement still holds"],"likelihood_band":"low|medium|high","uncertainties":["The arrangement may change"],"rationale":"How the intent and optional recurring states support this forecast"}}]}}

Reference time: {as_of}
Forecast window: {horizon_start} to {horizon_end}
Near-term intents: {intents}
Active recurring states: {states}
"""


RECALL_QUERY_ANALYSIS_PROMPT_EN = """You are the recall query analyzer for the AI-glasses long-term memory system.

Understand the memory structure before analyzing the query. 

Memory structure:
1. `fact`: traceable, self-contained narrative evidence extracted from dialogue or ambient transcripts. It preserves what happened, participants, time, place or scene, reasons, viewpoint changes, suggestions, acceptance or rejection, constraints, conclusions, and unresolved questions. Facts usually include `summary`, `keywords`, `entities`, `fact_topic`, `event_time_key`, and `dialogue_time_key`.
2. `user_claim`: a durable proposition about user self with status, confidence, validity time, and traceable supporting facts. Its origin is explicit (directly stated or confirmed by reliable records) or derived (logically entailed by existing claims). Repeated behavior observations are stored separately as `user_observed_pattern`. Relation objects may be other entities. Claims do not replace one-time event facts.
3. `goal`, `plan`, and `work_item`: future-oriented objects extracted from facts, respectively representing a desired outcome, an explicit arrangement, and a closed-loop responsibility. Their status and target/start/due times are meaningful.

Guidance:
- `recall_object_types` lists object types eligible for direct retrieval. It must always include `fact`; add `user_claim` only when the query asks about the user's durable claims or patterns; add the relevant `goal`, `plan`, or `work_item` only for future goals, arrangements, responsibilities, deadlines, unfinished work, or their lifecycle. user_activity is not retrieved in this version.
- Prefer `fact` for what happened, dates, places, people, exact evidence, event order, and traceable details.
- For stable preferences, durable constraints, routines, relationships, or profiles, add `user_claim` alongside the supporting facts.
- For tasks, commitments, future arrangements, goals, deadlines, or unfinished work, add the relevant prospective object alongside the supporting facts.
- Keep the plan broad when unsure. Missing evidence is worse than retrieving a few extra candidates.
- Extract 2-8 short retrieval keywords, prioritizing concrete people, organizations, products, projects, topics, actions, outcomes, and constraints. Do not output full sentences, pleasantries, generic words, or ordinary time expressions.
- Extract useful semantic entities with names and types. Entities may be people, organizations, locations, products, projects, technologies, or concrete concepts; ordinary time expressions such as today, yesterday, or last week are not entities.
- `temporal_mode` selects which fact timestamp should be used for a time range: `event_time` means the real-world event time described by the fact, `dialogue_time` means when the conversation/transcript occurred, `both` means either timestamp may match, and `none` means no hard time filter. Prefer `event_time` for queries asking what happened, was done, bought, or visited; prefer `dialogue_time` for queries asking what was discussed, mentioned, or asked; use `none` when the temporal intent is unclear.
- Parse `temporal_bounds` from the original query using the reference time. Use `YYYY-MM-DD HH:MM:SS` or `null` for `start` / `end`, and provide at least one bound. `end` is exclusive. Output `null` when no time constraint applies.
- Set `is_prospective_time_slot_query` to true only when the user asks for the set or status of their own current/future arrangements, plans, to-dos, or unfinished responsibilities in a time window. It is false for external information, advice, a single-fact confirmation, a prediction, or an immediate command such as creating a reminder.

Return JSON only:
{
  "recall_object_types": ["fact"],
  "needs_broad_evidence": false,
  "query_rewrite": "retrieval-focused rewrite over the unified memory projection",
  "keywords": ["keyword1", "keyword2"],
  "entities": [{"name": "entity name", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|OTHER"}],
  "temporal_bounds": {"start": "YYYY-MM-DD HH:MM:SS|null", "end": "YYYY-MM-DD HH:MM:SS|null"},
  "temporal_mode": "event_time|dialogue_time|both|none",
  "is_prospective_time_slot_query": false
}

Original user query:
{query}

Reference time for resolving relative time expressions:
{reference_time}
"""
