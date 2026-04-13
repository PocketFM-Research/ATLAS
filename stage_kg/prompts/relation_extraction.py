"""
Prompt templates for Relation Extraction (Pass 3).

Relations are typed edges grounded in the screenplay evidence.
Schema constraints restrict valid (source_type, relation, target_type) triples.
"""

SYSTEM_PROMPT = """You are a precise relation extractor for movie screenplay knowledge graphs.
Your output must be valid JSON only — no prose, no markdown fences, no explanation."""

RELATION_SCHEMA = """
Allowed relations by category (exactly as defined in the STAGE paper schema):

EVENT-ROLE:
  performs    : Character|Object|Concept|Vehicle -performs-> Event   (actively executes/initiates)
  undergoes   : Character|Object|Vehicle -undergoes-> Event          (acted upon or targeted)
  experiences : Character -experiences-> Event               (internal mental/emotional event)

SOCIAL:
  kinship_with  : Character -kinship_with-> Character        (kinship or marital)
  affinity_with : Character -affinity_with-> Character       (friends, allies, collaborators)
  hostility_with: Character -hostility_with-> Character      (enemies, rivals)
  affiliated_with: Character|Concept -affiliated_with-> Concept  (belongs to org/faction)

INTER-EVENT:
  precedes      : Event -precedes-> Event      (earlier in time/narrative order)
  causes        : Event -causes-> Event        (directly causes or triggers)
  contrasts_with: Event -contrasts_with-> Event (contrast or parallel in outcome)
  references    : Event -references-> Event    (refers to, recalls, describes another event)

SPATIOTEMPORAL:
  occurs_at  : Event -occurs_at-> Location|Vehicle      (event occurs at a location or aboard a vehicle)
  occurs_on  : Event -occurs_on-> TimePoint              (event occurs at a time point)
  located_at : Character|Object|Concept|Vehicle -located_at-> Location|Vehicle
  present_on : Character|Object|Concept|Vehicle -present_on-> TimePoint

OBJECT-RELATED:
  possesses : Character|Concept -possesses-> Object|Vehicle   (owns or holds)
  uses      : Character|Object -uses-> Object|Vehicle         (operates another object or vehicle)

SEMANTIC:
  is_a    : Character|Object|Concept|Vehicle -is_a-> Concept     (type-instance or subclass)
  part_of : Object|Location|Character|Concept|Vehicle -part_of-> Object|Location|Concept|Vehicle  (part-whole)
"""


def build_prompt(
    scene_id: str,
    scene_title: str,
    scene_text: str,
    events: list,
    entities: list,
    chunk_id: str = "",
    movie_title: str = "",
) -> str:
    import json

    events_json = json.dumps(
        [{"id": e["temp_id"], "name": e["name"], "type": "Event"} for e in events],
        ensure_ascii=False,
    )
    entities_json = json.dumps(
        [{"id": e["temp_id"], "name": e["canonical_name"], "type": e["type"]} for e in entities],
        ensure_ascii=False,
    )
    movie_block = f"\nMovie: {movie_title}" if movie_title else ""

    return f"""Extract typed relations between entities and events in the following screenplay scene.
{movie_block}
Scene ID: {scene_id}
Scene heading: {scene_title}
Chunk: {chunk_id or scene_id}

EVENTS (use the ids exactly as provided):
{events_json}

ENTITIES (use the ids exactly as provided):
{entities_json}

SCREENPLAY TEXT:
\"\"\"
{scene_text}
\"\"\"

SCHEMA:
{RELATION_SCHEMA}

INSTRUCTIONS:
1. Extract relations between the nodes listed above using ONLY the allowed relation types.
2. Check schema validity: every (source_type, relation, target_type) triple must be listed in the schema.
3. For each relation provide:
   - "temp_id": a short unique slug (e.g. "rel_001")
   - "source_id": exact id of the source node from the lists above
   - "source_type": type of the source node (Character, Event, etc.)
   - "relation": the relation label (exactly as listed in schema)
   - "target_id": exact id of the target node from the lists above
   - "target_type": type of the target node
   - "scene_id": "{scene_id}"
   - "chunk_id": "{chunk_id or scene_id}"
   - "evidence": list of text spans supporting this relation
   - "confidence": float 0.0–1.0 (your confidence this relation is correct)
4. Use only ids from this chunk. Do not invent ids and do not reuse ids from a different chunk.
5. For performs/undergoes/experiences, only link an entity to an event when that entity is clearly the participant in the quoted evidence.
6. Do not assign an event to the wrong speaker or nearby character just because they appear in the same chunk.
7. Camera/stage directions may support occurs_at/located_at, but should not be the sole evidence for an event-role relation.
8. REJECT any triple not valid under the schema — do not include it.
9. MANDATORY EVENT-ROLE EDGES: For EVERY event listed above, produce at least one performs, undergoes, or experiences relation linking a Character (or Object/Vehicle) to that event. Use the character who performed or was affected. If unclear, use the most prominent character in the scene.
10. COVERAGE REQUIREMENT: Every entity listed above MUST appear as source or target in at least one relation. If no meaningful relation exists, use located_at, is_a, or present_on as a fallback.
11. Every event MUST also have at least one occurs_at relation linking it to a Location or Vehicle when a location is mentioned. This includes fight/match events occurring in a ring, gym, or arena — always connect them to the venue Location node.
12. Do NOT hallucinate relations not grounded in the text.

Return ONLY a JSON array of relation objects. No other text.

Example element shape:
{{
  "temp_id": "rel_001",
  "source_id": "ent_001",
  "source_type": "Character",
  "relation": "performs",
  "target_id": "ev_001",
  "target_type": "Event",
  "scene_id": "{scene_id}",
  "chunk_id": "{chunk_id or scene_id}",
  "evidence": ["CAPTAIN: Everyone out, now!"],
  "confidence": 0.95
}}
"""
