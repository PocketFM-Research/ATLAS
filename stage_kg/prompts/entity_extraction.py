"""
Prompt templates for Entity Extraction (Pass 2).

Entities are extracted in relation to the event inventory.
Node types: Character, Location, TimePoint, Object, Concept.
"""

SYSTEM_PROMPT = """You are a precise named-entity annotator for movie screenplays.
Your output must be valid JSON only — no prose, no markdown fences, no explanation."""

NODE_TYPES = ["Character", "Location", "TimePoint", "Object", "Concept"]


def build_prompt(
    scene_id: str,
    scene_title: str,
    scene_text: str,
    events: list,
    chunk_id: str = "",
    movie_title: str = "",
) -> str:
    import json
    events_json = json.dumps(events, ensure_ascii=False, indent=2)
    movie_block = f"\nMovie: {movie_title}" if movie_title else ""

    return f"""Extract all entities from the following screenplay scene that are connected to the events listed below.
{movie_block}
Scene ID: {scene_id}
Scene heading: {scene_title}
Chunk: {chunk_id or scene_id}

EVENTS ALREADY EXTRACTED (use these as anchors):
{events_json}

SCREENPLAY TEXT:
\"\"\"
{scene_text}
\"\"\"

INSTRUCTIONS:
1. Entity types allowed: {', '.join(NODE_TYPES)}
2. Extract every named entity (character, place, time, object, or concept) that:
   a. Appears in the screenplay text, AND
   b. Is involved in at least one of the listed events, OR is clearly important to the scene.
3. For each entity provide:
   - "temp_id": a short unique slug (e.g. "ent_001")
   - "type": one of {NODE_TYPES}
   - "surface_forms": list of all surface name variants found in this scene
   - "canonical_name": the best single canonical name for this entity
   - "description": one sentence describing the entity in context
   - "scene_id": "{scene_id}"
   - "chunk_id": "{chunk_id or scene_id}"
   - "evidence": list of short text spans supporting this entity
   - "linked_event_ids": list of temp_ids of events this entity is involved in
4. Do NOT invent entities not present in the text.
5. Characters: include named crew, named civilians, named antagonists. Exclude unnamed extras.
6. Locations: include named places, ships, rooms, planets, regions.
7. TimePoints: include specific dates, times, or named time periods explicitly mentioned.
8. Objects: include named props, weapons, vehicles, technology that are plot-relevant.
9. Concepts: include named organizations, groups, ideologies, or abstract constructs explicitly named.

Return ONLY a JSON array of entity objects. No other text.

Example element shape:
{{
  "temp_id": "ent_001",
  "type": "Character",
  "surface_forms": ["Captain Reyes", "Reyes", "Captain"],
  "canonical_name": "Captain Reyes",
  "description": "The ship's commanding officer directing the evacuation.",
  "scene_id": "{scene_id}",
  "chunk_id": "{chunk_id or scene_id}",
  "evidence": ["CAPTAIN REYES: Everyone out, now!", "Captain Reyes grabs the radio."],
  "linked_event_ids": ["ev_001", "ev_002"]
}}
"""
