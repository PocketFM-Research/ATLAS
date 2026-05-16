"""
Prompt templates for Event Extraction (Pass 1).

Events anchor the narrative structure and are extracted first.
Each event represents an action, interaction, or state transition
that is narratively salient in the scene.
"""

SYSTEM_PROMPT = """You are a precise narrative analyst extracting events from movie screenplays.
Your output must be valid JSON only — no prose, no markdown fences, no explanation."""


def build_prompt(
    scene_id: str,
    scene_title: str,
    scene_text: str,
    scene_summary: str = "",
    chunk_id: str = "",
    movie_title: str = "",
) -> str:
    summary_block = f"\nScene summary: {scene_summary}" if scene_summary else ""
    movie_block = f"\nMovie: {movie_title}" if movie_title else ""

    return f"""Extract all narratively salient events from the following screenplay scene.
{movie_block}
Scene ID: {scene_id}
Scene heading: {scene_title}{summary_block}
Chunk: {chunk_id or scene_id}

SCREENPLAY TEXT:
\"\"\"
{scene_text}
\"\"\"

INSTRUCTIONS:
1. Extract every action, interaction, or state transition that advances the story or reveals character.
2. Ignore trivial stage directions (camera angles, lighting cues) unless they carry narrative meaning.
3. For each event provide:
   - "temp_id": a short unique slug (e.g. "ev_001")
   - "name": a concise, canonical event name (verb phrase, max 12 words)
   - "description": one-sentence description grounding the event in the scene
   - "scene_id": "{scene_id}"
   - "chunk_id": "{chunk_id or scene_id}"
   - "evidence": the exact text span(s) from the screenplay supporting this event (list of strings)
   - "participants": list of character/entity names obviously involved (may be empty)
   - "location_hint": location name if clearly stated (or null)
   - "time_hint": time reference if clearly stated (or null)
   - "scope": "local" if the event matters only in this scene, "global" if it has broader narrative significance
4. Prefer precision over exhaustive recall. Extract 3–15 events per scene.
5. Order events chronologically within the scene.

Return ONLY a JSON array of event objects. No other text.

Example element shape:
{{
  "temp_id": "ev_001",
  "name": "captain orders evacuation",
  "description": "The captain orders everyone to evacuate the building immediately.",
  "scene_id": "{scene_id}",
  "chunk_id": "{chunk_id or scene_id}",
  "evidence": ["CAPTAIN: Everyone out, now!"],
  "participants": ["Captain Reyes", "crew"],
  "location_hint": "control room",
  "time_hint": null,
  "scope": "global"
}}
"""
