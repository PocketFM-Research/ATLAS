"""
Reflection-based quality scoring prompt (Appendix C.3).

After each extraction attempt, an LLM scores the result 0–10 on three axes:
  - Accuracy:    valid triples, schema adherence, no malformed structures
  - Consistency: stable naming/typing, no referential ambiguity
  - Redundancy:  no low-value or repetitive entities/relations

Score < 7 → re-extract using the reflection feedback as guidance.
Critical errors (schema violations, self-referential relations) force score to 0.
"""

SYSTEM_PROMPT = """You are a quality-control annotator for narrative knowledge graphs.
Your output must be valid JSON only — no prose, no markdown fences, no explanation."""

ACCEPTANCE_THRESHOLD = 7
MAX_RETRIES = 3


def build_event_reflection_prompt(scene_text: str, extracted_events: list) -> str:
    import json
    events_json = json.dumps(extracted_events, ensure_ascii=False, indent=2)
    return f"""Score the quality of the following event extraction from a movie screenplay scene.

SCREENPLAY TEXT:
\"\"\"
{scene_text[:2000]}
\"\"\"

EXTRACTED EVENTS:
{events_json}

SCORING CRITERIA — rate 0–10 on each axis, then give an overall score:

1. Accuracy (0–10):
   - Are events grounded in the text? No invented events?
   - Are event names and descriptions factually accurate?
   - Are participant names correctly identified?

2. Consistency (0–10):
   - Are entity names stable and unambiguous?
   - Are scope labels (local/global) appropriate?

3. Redundancy (0–10, higher = less redundant):
   - Are events narratively distinct? No duplicates or near-duplicates?
   - Do events represent meaningful story beats, not trivial stage directions?

CRITICAL FAILURES (force overall score to 0):
   - An event that has no grounding in the screenplay text
   - Self-referential events (event refers to itself as participant)

Return ONLY a JSON object:
{{
  "accuracy": <0-10>,
  "consistency": <0-10>,
  "redundancy": <0-10>,
  "overall": <0-10>,
  "feedback": "<one sentence describing the most important issue to fix on retry>"
}}
"""


def build_entity_reflection_prompt(scene_text: str, extracted_entities: list) -> str:
    import json
    ents_json = json.dumps(extracted_entities, ensure_ascii=False, indent=2)
    return f"""Score the quality of the following entity extraction from a movie screenplay scene.

SCREENPLAY TEXT:
\"\"\"
{scene_text[:2000]}
\"\"\"

EXTRACTED ENTITIES:
{ents_json}

SCORING CRITERIA — rate 0–10 on each axis, then give an overall score:

1. Accuracy (0–10):
   - Are all entities actually present in the text?
   - Are entity types correct (Character/Location/TimePoint/Object/Concept)?
   - Are canonical names accurate?

2. Consistency (0–10):
   - Are surface forms and canonical names internally consistent?
   - No ambiguous or conflicting type assignments?

3. Redundancy (0–10, higher = less redundant):
   - No duplicate entities (same entity listed twice under different names)?
   - All entities are narratively relevant?

CRITICAL FAILURES (force overall score to 0):
   - Entity typed incorrectly (e.g. a location listed as a Character)
   - Entity that does not appear anywhere in the screenplay text
   - Two entities with identical canonical names but different types

Return ONLY a JSON object:
{{
  "accuracy": <0-10>,
  "consistency": <0-10>,
  "redundancy": <0-10>,
  "overall": <0-10>,
  "feedback": "<one sentence describing the most important issue to fix on retry>"
}}
"""


def build_relation_reflection_prompt(
    scene_text: str,
    extracted_relations: list,
    events: list,
    entities: list,
) -> str:
    import json

    # Slim node lists for the prompt
    node_index = {
        e.get("id", e.get("temp_id", "")): f"{e.get('type','')}:{e.get('name', e.get('canonical_name',''))}"
        for e in events + entities
    }
    node_index_json = json.dumps(node_index, ensure_ascii=False)
    rels_json = json.dumps(extracted_relations, ensure_ascii=False, indent=2)

    return f"""Score the quality of the following relation extraction from a movie screenplay scene.

NODE INDEX (id -> type:name):
{node_index_json}

EXTRACTED RELATIONS:
{rels_json}

SCREENPLAY TEXT:
\"\"\"
{scene_text[:1500]}
\"\"\"

VALID RELATION TYPES:
performs, undergoes, experiences,
kinship_with, affinity_with, hostility_with, affiliated_with,
precedes, causes, contrasts_with, references,
occurs_at, occurs_on, located_at, present_on,
possesses, uses, is_a, part_of

SCORING CRITERIA — rate 0–10, then overall:

1. Accuracy (0–10):
   - Is every relation grounded in the screenplay text?
   - Are relation types correct and schema-valid?
   - Do source and target IDs refer to real nodes in the index?

2. Consistency (0–10):
   - No contradictory relations (e.g. A precedes B and B precedes A)?
   - Confident assignments where evidence is clear?

3. Redundancy (0–10, higher = less redundant):
   - No duplicate (source, relation, target) triples?

CRITICAL FAILURES (force overall score to 0):
   - Relation type not in the valid list above
   - Self-referential relation (source == target)
   - Source or target ID that does not exist in the node index

Return ONLY a JSON object:
{{
  "accuracy": <0-10>,
  "consistency": <0-10>,
  "redundancy": <0-10>,
  "overall": <0-10>,
  "feedback": "<one sentence describing the most important issue to fix on retry>"
}}
"""
