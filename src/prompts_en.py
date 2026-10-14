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
- `user_claim_signals` may contain zero or more structured evidence hints for user_claims whose subject is user self. They are not final claims and must not decide relations to existing claims. Every signal must concern user self as its subject; claims about other people, organizations, product users, or ambient speakers must not be routed here.
- Emit a signal only when this fact's entities contain user self and either (1) `speaker_用户本人` directly states the proposition, (2) the wearer clearly confirms or corrects another speaker's specific proposition, or (3) a third party explicitly describes the wearer through a confirmed alias. The third case is a `third_party_report`, not a wearer assertion; a namesake, unrelated wearer utterance, or vague acknowledgment is insufficient.
- `owner_evidence_type` is `self_statement|confirmed_response|third_party_report`. The first two require an exact wearer quote in `owner_evidence_text`; the response case also requires an earlier other-speaker quote. A third-party report has empty `owner_evidence_text` and an exact other-speaker quote containing the confirmed alias in `other_evidence_text`. Never invent quotes or quote preceding context.
- The signal's `entity` must always identify user self, never the other end of a relationship. A relationship involving user self may be signalled; its other person, organization, or project remains a fact entity and may later be the claim's object_entity.
- `signal_kind` must be one of: `explicit_assertion` (a durable proposition directly stated by an entity or confirmed by a reliable record), or `pattern_observation` (an observation that may later support one directionally specific pattern together with evidence from other time windows).
- `claim_type_hint` must be one of: identity_profile, affiliation, relationship, preference, constraint, behavior_pattern. A single action must not become an explicit_assertion for behavior_pattern; at most it is a pattern_observation.
- `claim_anchor` is a short, stable grouping label that can collect facts from different time windows about one possible claim or pattern, such as "quiet travel preference" or "swimming routine". It is neither a sentence nor the final user_claim.
- Output signals only when this fact has concrete evidence value for a claim about user self or later pattern induction. Use an empty array for one-off background, claims about third parties themselves, temporary suggestions, assistant speculation, pleasantries, and low-value content.
- Each signal's `entity.name` must exactly reuse one entity name from the same fact's `entities`, and its `entity.type` must match that entity's type. Do not add, rewrite, or guess an entity that is not listed by the fact; omit the signal when no such entity exists.
- Signal-to-fact grounding is mandatory: the signal's entity, proposition, and necessary qualifiers must be fully supported by the same fact's `text`. A response-based signal must preserve both the earlier statement and wearer response there. Do not borrow information omitted from this fact from another fact or preceding context. `evidence_basis` must explain how the wearer's utterance supports the final proposition; otherwise omit the signal.
- Return at most 3 signals per fact. Every signal must contain entity, signal_kind, claim_type_hint, claim_anchor, owner_evidence_type, owner_evidence_text, other_evidence_text, evidence_basis, and confidence.

user_intent_signals rules:
- `user_intent_signals` are structured hints that this fact is evidence about the wearer's future world. They are not final goals, plans, or work items, and must not directly create, update, or override an existing object.
- Record only user self's future-facing information: `subject_entity` must always be exactly `user self` (use `用户本人` in Chinese output). Never use another person, team, organization, product user, or ambient-recording speaker as the subject.
- Output a signal when the wearer directly expresses or clearly confirms a future goal, arrangement, responsibility, or lifecycle change. A third party may also explicitly report an arrangement or responsibility concerning the wearer through a confirmed alias; mark this `third_party_report` and never turn an assignment, guess, or suggestion into the wearer's own commitment. Preserve speaker attribution in this fact.
- `owner_evidence_type` is `self_statement|confirmed_response|third_party_report`. The first two require an exact wearer quote in `owner_evidence_text`; response-based signals also need an earlier other-speaker quote. Third-party reports have empty `owner_evidence_text` and an exact quote containing the confirmed alias in `other_evidence_text`.
- `evidence_kind` must be one of goal, plan, responsibility, or lifecycle_update. `candidate_object_types` may contain only goal, plan, and work_item. A third-party report about a goal or responsibility is only a hint, not proof that the wearer set the goal or accepted a commitment.
- `user_role` must be owner, participant, or responsible: user self respectively owns the goal, participates in the arrangement, or bears the responsibility. Do not output a signal for a product user, customer, consumer, or an ambient speaker merely mentioned in the evidence.
- `assertion_source` must be self_statement, third_party_report, or observed_event; `explicitness` must be direct, reported, or tentative. For a third-party report use `assertion_source=third_party_report` and `explicitness=reported`. When the wearer accepts another person's proposal, use self_statement. Assistant suggestions, open hypotheticals, and speculation do not themselves create signals.
- `user_intent_anchor` is a short, stable grouping label for the goal, event, or responsibility, such as "Tianjin client meeting", "half-marathon training goal", or "client report delivery". It is not a full sentence or the final object description.
- Signal-to-fact grounding is mandatory: the future goal, arrangement, responsibility, or lifecycle change itself must be explicitly retained in the same fact's `text`. Do not attach information found elsewhere in the raw unit batch to a fact that does not preserve it; output an empty array when the fact text cannot independently support the signal.
- Return at most 2 signals per fact. Every signal must include subject_entity, evidence_kind, candidate_object_types, user_role, user_intent_anchor, assertion_source, explicitness, owner_evidence_type, owner_evidence_text, other_evidence_text, evidence_basis, and confidence. evidence_basis must be concrete evidence directly locatable in this fact's `text`.

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
          "entity": {"name": "an entity name copied exactly from this fact's entities", "type": "the type of that same entity in this fact's entities"},
          "signal_kind": "explicit_assertion|pattern_observation",
          "claim_type_hint": "identity_profile|affiliation|relationship|preference|constraint|behavior_pattern",
          "claim_anchor": "specific claim or pattern grouping label",
          "owner_evidence_type": "self_statement|confirmed_response|third_party_report",
          "owner_evidence_text": "exact wearer quote for self/response; empty for third-party report",
          "other_evidence_text": "exact other-speaker quote for response/report; otherwise empty",
          "evidence_basis": "specific evidence from this fact supporting the signal",
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
          "assertion_source": "self_statement|third_party_report|observed_event",
          "explicitness": "direct|reported|tentative",
          "owner_evidence_type": "self_statement|confirmed_response|third_party_report",
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

INTENT_EXTRACTION_PROMPT_EN = """Extract Intent & Execution objects from stored, traceable narrative facts for a personal world model. The input is a scheduling batch of independent signal items, not an evidence group for one intent.

If a signal has `assertion_source=third_party_report`, treat it as a verifiable third-party lead, not a wearer commitment. Do not create a goal or work item solely from it. A plan may be emitted only when the fact clearly describes an established future arrangement; keep confidence low and preserve the source qualification in the summary.

Only output:
- goal: an explicit, durable desired outcome that spans more than one action;
- plan: an explicit future arrangement, event, activity, trip, meeting, or appointment with a time/window or explicit arranged, agreed, or registered evidence;
- work_item: a bounded responsibility with a clear responsible party, action, and at least one completion anchor: a deliverable, deadline, explicit commitment, or an assignment explicitly accepted by the wearer.

Use only direct fact evidence. Do not store assistant suggestions, speculation, behavior patterns, or predictions. Do not create an object for weak hypotheticals such as maybe, if, should we, or consider. A goal or plan never automatically creates a work item. Use `occurred` only for a plan whose event happened, and `completed` only for a completed work item. Each candidate has exactly one `object_type`; do not fan out one fact into multiple object types without independent evidence. `lifecycle_evidence` describes only an observed fact-level change, never a database operation; reconciliation decides create, update, or completion against existing objects. Every candidate must output one `primary_domain` from `general|health|work|life|information|finance|family|learning|social|interest|travel`. It is the primary user-life domain affected by the object: prefer the most specific domain, and use `general` only for genuinely cross-domain objects, never as an uncertainty fallback.

Another person's request or assignment does not by itself make it the wearer's responsibility. For a response-based intent, the fact must preserve the wearer's subsequent clear acceptance, commitment, or correction. A mere acknowledgment or "got it" is insufficient.

Judge each `signal_id` independently using only its own fact and user_intent_signal; never combine evidence across items. Return exactly one result per signal, with either one `intent_candidate` object or null. A signal is only a routing hint, not a guaranteed object. `evidence_fact_ids` must contain exactly that item's `fact.id`. A lifecycle update may yield a candidate describing a change to an existing object; this prompt does not know whether that object exists in the database. Later reconciliation performs the match and must not create a new object when no match exists.

World owner: {world_owner_name}
Reference time: {reference_timestamp}
Independent signal items (each has `signal_id`, `fact`, and one `user_intent_signal`):
{signal_items}

Return JSON only:
{
  "results": [
    {
      "signal_id": "copy the input signal_id exactly",
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

The input contains stored, traceable narrative facts. Each fact may include zero or more `user_claim_signals` with candidate claim types, signal kinds, grouping labels, and evidence pointers. Inspect all signals, but treat them only as candidate hints, never as additional facts or final conclusions. Every emitted claim must be directly supported by the corresponding fact summary and cite its factual basis in `evidence_fact_ids`. An `explicit_assertion` may point to a direct-claim candidate; a `pattern_observation` is only an observational hint and cannot by itself establish a preference, habit, or behavioral rule. A proposition directly stated by the wearer, or clearly confirmed or corrected by the wearer in response to another person's specific statement, can support an explicit claim. Another person's statement alone, a vague acknowledgment, a one-off action, a recommendation, or assistant speculation cannot. A response-based fact must preserve both the earlier statement and the wearer's final position.

The system assigns `claim_origin=explicit` to claims from this process; do not include that field in the output.

`user_claim` is a durable proposition about user self that remains traceable to facts; it is not a retelling of a one-time event and does not replace the underlying evidence. Every claim's `subject_entity` must be the non-empty `world_owner_entity` explicitly marked in its evidence fact (user self). Never create a user_claim whose subject is another person, organization, product user, or ambient speaker. Other entities may be the `object_entity` of a relation involving user self. If no evidence fact explicitly identifies `world_owner_entity`, return no claim. Allowed explicit claim_type values are identity_profile, affiliation, relationship, preference, and constraint. Behavioral regularities (`behavior_pattern`) are not extracted by this prompt; they are handled by induction.

Rules:
1. subject_entity must exactly equal `world_owner_entity` on every cited evidence fact; never use a third party as subject. Any object_entity must occur in input fact entities.
2. evidence_fact_ids must cite only input fact IDs and each claim needs direct evidence.
3. Use concise stable lowercase predicates, such as has_role, works_with, member_of, located_in, prefers, dislikes, requires, cannot, has_constraint.
4. Use object_entity for relational claims. For non-relational claims, express the attribute, value, and relevant qualification completely in claim_text.
5. claim_text is a complete, self-contained, reader-facing proposition with its subject, predicate, and value.
6. Encode negation in predicate and claim_text, such as dislikes, cannot, or is_not_member_of; do not output a standalone positive/negative field. Return no claim without direct evidence.
7. Do not output a one-time trip, task, plan, recommendation, or open question. An empty list is correct. Return JSON only.
8. `primary_domain` must be one of `general|health|work|life|information|finance|family|learning|social|interest|travel`; choose the most specific user-life domain, and use `general` only for genuinely cross-domain claims.

Output:
{
  "claims": [{
    "subject_entity": "",
    "claim_type": "identity_profile|affiliation|relationship|preference|constraint",
    "primary_domain": "general|health|work|life|information|finance|family|learning|social|interest|travel",
    "predicate": "",
    "object_entity": "",
    "claim_text": "complete proposition with subject and meaning",
    "valid_from": "",
    "valid_to": "",
    "evidence_fact_ids": [1],
    "confidence": 0.85
  }]
}

facts:
{facts}
"""


INDUCTIVE_USER_CLAIM_EXTRACTION_PROMPT_EN = """You conservatively extract inductive user_claims about user self for a personal world model.

Input facts are traceable observations for one entity and one claim_anchor, drawn from different time windows. The claim_anchor only gathers candidate evidence; it is not a conclusion. Decide whether the evidence supports a pattern; do not summarize facts.

An inductive user_claim expresses a preference or behavioral pattern supported by repeated evidence across facts and time windows. Only preference and behavior_pattern claims are allowed; the system sets `claim_origin` to `inductive`. It is not a one-time event fact and does not replace its supporting facts.

Hard rules:
1. A claim requires at least three distinct support_fact_ids, and every cited ID must directly support the proposed pattern.
2. A single statement, event, plan, task, or assistant recommendation is not a pattern.
3. Do not turn an isolated behavior into a preference. Cite only facts that directly support the proposed pattern in support_fact_ids; other facts need not be labeled as counterexamples.
4. subject_entity must equal the given user-self subject and `world_owner_entity`; never produce a pattern about another entity. predicate is one of prefers, dislikes, usually_does, avoids, has_routine.
5. claim_text is a complete reader-facing pattern proposition with its subject, property, and value.
6. `primary_domain` must be one of `general|health|work|life|information|finance|family|learning|social|interest|travel`; choose the most specific user-life domain, and use `general` only for genuinely cross-domain patterns.
7. Evidence IDs must be input IDs. An empty list is correct. Return JSON only.

Output:
{
  "claims": [{
    "subject_entity": "",
    "claim_type": "preference|behavior_pattern",
    "primary_domain": "general|health|work|life|information|finance|family|learning|social|interest|travel",
    "predicate": "prefers|dislikes|usually_does|avoids|has_routine",
    "claim_text": "complete pattern proposition with subject and meaning",
    "condition_text": "",
    "support_fact_ids": [1, 2, 3],
    "confidence": 0.8
  }]
}

candidate entity and claim grouping hint:
{induction_target}

evidence facts:
{facts}
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
3. Align with existing states. If an existing state in the same domain, class, and key already expresses the conclusion, reinforce or revise it rather than creating a duplicate. If new evidence challenges a state, examine all state_claim_evidence and supporting_claims before weakening, expiring, or partially revising it. Losing one support claim does not necessarily invalidate the state. Do not abandon a justified create decision merely because no old state is affected.
4. Decide old-claim status. Only a contradicts or supersedes event with an old target claim and a changed trigger claim may justify an old-claim update. The relation alone is not a status decision: consider text, origin (explicit is usually more direct than inductive or derived), time, and conditions. Contradicts may at most weaken; only an unambiguous supersedes may supersede. Claims valid under different conditions or times may coexist. Do not alter changed claims or invalidate an old claim merely because an aggregate state fails. If there is no eligible relation, claim_updates=[]; this does not prevent state operations.
5. Check joint consistency. A claim proposed as superseded cannot remain effective support for a state. Omit individual state candidates lacking evidence, but do not discard other supported states in the same update.

State operations are create|reinforce|revise|weaken|expire. Existing-state updates must use its id; create uses target_state_id=0. Controlled enduring_state keys: communication_style|decision_style|value_priority|risk_preference|privacy_preference|time_preference|work_style|lifestyle_preference|long_term_constraint. Controlled recurring_state keys: daily_routine|weekly_routine|work_routine|health_routine|social_routine|conditional_behavior. Preserve an existing state's class and key; stay within state_domain.

Each state operation cites input claims only, with evidence_type=user_claim, evidence_id, role=change_trigger|support|counterexample|context, and weight. Cite at least one changed claim as the trigger. A claim proposed as superseded cannot remain support in this update. Counterexamples must not be mislabeled as support. Include concise reasons and confidence in [0,1]. Return an empty operations array only after assessing every changed claim and finding no justified operation. Return JSON only:
{
  "claim_updates": [{"claim_id": 1, "new_status": "weakened|superseded", "relation_event_id": 1, "reason": "why this evidence changes the old claim"}],
  "operations": [{
    "operation": "create|reinforce|revise|weaken|expire",
    "target_state_id": 0,
    "state_class": "enduring_state|recurring_state",
    "state_key": "controlled option",
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
"""


USER_ACTIVITY_STATE_UPDATE_PROMPT_EN = """Incrementally update situational_state from newly generated user activity observations.

This update is triggered by new activity. An activity describes what the user actually participated in during a continuous time window. Facts provide concrete evidence from that window. Related intents are currently active goals/plans/work items with direct evidence links to those facts; they are supplemental context only. For `user_role`, `actor` means evidence supports that the user is doing the activity, `participant` means the user is involved, and `contextual` is background only and must not be used to infer that the user is doing it. Do not copy an activity, fact, or intent into a state, and do not assume that mentioning a topic means the user is currently working on it.

Create or update only `situational_state`; do not create enduring_state or recurring_state here. Emit an operation only when evidence indicates that the user is in an unfinished phase, is continuing to handle an issue, is preparing for something, is subject to a current constraint, or has a time-bounded current concern/availability. Chit-chat, one-off Q&A, reminiscence, preference statements, completed activities, and one-time itineraries normally do not warrant a situational state. Return an empty operations list when evidence is insufficient.

Every operation must cite at least one `user_activity` item from this input. Facts may only be cited from `linked_facts`; goals/plans/work items may only be cited from `related_active_intents`. Related intents were prefiltered by the system because they share fact evidence with the activity; still decide whether they describe the same ongoing situation. Never cite an ID not supplied here. Activity is the trigger and situation anchor; do not infer that a completed activity is still current.

Choose `state_domain` only from `allowed_domains`. When updating an existing state, use the ID of an existing state with the same domain, class, and key. `state_key` must be one of: `current_focus`, `current_constraint`, `current_availability`, `current_energy`, `current_stress`, `current_social_engagement`, `current_preparation`. `state_text` must be a self-contained description of the current state, not a historical summary. Provide a plausible `valid_to` or ending condition when evidence supports it; do not invent a deadline. Allowed operations are create, reinforce, revise, weaken, expire; `target_state_id` is 0 for create.

Return JSON only:
{
  "operations": [
    {
      "operation": "create|reinforce|revise|weaken|expire",
      "target_state_id": 0,
      "state_domain": "one allowed_domains value",
      "state_class": "situational_state",
      "state_key": "one controlled option",
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


RECALL_QUERY_ANALYSIS_PROMPT_EN = """You are the recall query analyzer for the AI-glasses long-term memory system.

Understand the memory structure before analyzing the query. 

Memory structure:
1. `fact`: traceable, self-contained narrative evidence extracted from dialogue or ambient transcripts. It preserves what happened, participants, time, place or scene, reasons, viewpoint changes, suggestions, acceptance or rejection, constraints, conclusions, and unresolved questions. Facts usually include `summary`, `keywords`, `entities`, `fact_topic`, `event_time_key`, and `dialogue_time_key`.
2. `user_claim`: a durable proposition about user self with status, confidence, validity time, and traceable supporting facts. Its origin is explicit (directly stated or confirmed by reliable records), inductive (a preference or behavior pattern inferred across facts and time windows), or derived (logically entailed by existing claims). Relation objects may be other entities. It does not replace one-time event facts.
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
