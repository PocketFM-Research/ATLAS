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
  performs    : Character|Object|Concept -performs-> Event   (actively executes/initiates)
  undergoes   : Character|Object -undergoes-> Event          (acted upon or targeted)
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
  occurs_at  : Event -occurs_at-> Location          (event occurs at a location)
  occurs_on  : Event -occurs_on-> TimePoint         (event occurs at a time point)
  located_at : Character|Object|Concept -located_at-> Location
  present_on : Character|Object|Concept -present_on-> TimePoint

OBJECT-RELATED:
  possesses : Character|Concept -possesses-> Object   (owns or holds)
  uses      : Character|Object -uses-> Object         (operates another object)

SEMANTIC:
  is_a    : Character|Object|Concept -is_a-> Concept     (type-instance or subclass)
  part_of : Object|Location|Character|Concept -part_of-> Object|Location|Concept  (part-whole)
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

EVENTS (use temp_id to reference):
{events_json}

ENTITIES (use temp_id to reference):
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
   - "source_id": temp_id of the source node
   - "source_type": type of the source node (Character, Event, etc.)
   - "relation": the relation label (exactly as listed in schema)
   - "target_id": temp_id of the target node
   - "target_type": type of the target node
   - "scene_id": "{scene_id}"
   - "chunk_id": "{chunk_id or scene_id}"
   - "evidence": list of text spans supporting this relation
   - "confidence": float 0.0–1.0 (your confidence this relation is correct)
4. REJECT any triple not valid under the schema — do not include it.
5. Include at minimum: one occurs_at per event with a location, one performs/undergoes per major event.
6. Do NOT hallucinate relations not grounded in the text.

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
  "evidence": ["SAAVIK: Plot an intercept course for the Kobayashi Maru."],
  "confidence": 0.95
}}
"""
