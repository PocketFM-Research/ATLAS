"""
Prompt templates for atomic claim extraction.

Claims are lightweight, text-grounded facts extracted from screenplay chunks
using the same chunk-level workflow as the STAGE graph extractors.
"""

from ..schema import RELATION_TYPES

SYSTEM_PROMPT = """You are a precise factual claim extractor for movie screenplays.
Your output must be valid JSON only — no prose, no markdown fences, no explanation."""


def build_prompt(
    scene_id: str,
    scene_title: str,
    scene_text: str,
    chunk_id: str = "",
    movie_title: str = "",
    scene_summary: str = "",
) -> str:
    movie_block = f"\nMovie: {movie_title}" if movie_title else ""
    summary_block = f"\nScene summary: {scene_summary}" if scene_summary else ""

    return f"""Extract grounded atomic claims from the following screenplay scene chunk.
{movie_block}
Scene ID: {scene_id}
Scene heading: {scene_title}{summary_block}
Chunk: {chunk_id or scene_id}

SCREENPLAY TEXT:
\"\"\"
{scene_text}
\"\"\"

INSTRUCTIONS:
1. Extract grounded atomic claims only.
2. A claim is atomic if it states one smallest verifiable fact. If a claim can be split into two facts, split it.
3. Each claim must have exactly one subject, one predicate, and one object.
4. Prefer STAGE schema predicates when possible: {', '.join(RELATION_TYPES)}.
5. Keep subjects and objects short, canonical, and text-grounded.
6. Do not invent hidden motives, implications, or off-screen facts.
7. Extract underlying scene facts, not transcript-style speech wrappers.
8. You may combine nearby sentences only when they jointly support one single atomic fact.
9. You must split one sentence into multiple claims if it contains multiple separable facts.
10. Convert dialogue into the simplest verifiable fact unless the speech act itself is the salient event.
11. Prefer graph-aligned minimal claims:
    - actions -> Character/Event
    - social relations -> Character/Character
    - object, location, concept claims only when explicitly supported
12. Keep event objects short and action-centered.
13. Exclude explicit quoted subclaims when they are only dialogue wording.
14. Skip filler greetings, weak banter, and vague claims.
15. General examples:
    - GOOD split: "A enters the room and hugs B" -> "A enters the room." + "A hugs B."
    - GOOD rewrite: "A says B should leave" -> "A orders B to leave." if that is the stable event.
    - BAD: "A says 'you should leave now.'"
    - BAD: "A has a complicated conversation with B about the problem."
16. For each claim provide:
   - "temp_id": a short unique slug (e.g. "cl_001")
   - "subject": canonical subject string
   - "predicate": relation label
   - "object": canonical object string
   - "claim_text": one-sentence paraphrase of the fact
   - "scene_id": "{scene_id}"
   - "chunk_id": "{chunk_id or scene_id}"
   - "evidence": list of exact supporting text spans
   - "confidence": float 0.0–1.0
17. `claim_text` should be clean plain English and semantically equivalent to the atomic fact.
18. Keep evidence compact: include 1-2 short supporting spans per claim.
19. Return 0-8 claims depending on how much grounded content exists.
20. If a candidate depends mainly on quote wording rather than a stable fact, omit it.

Return ONLY a JSON array of claim objects. No other text.

Example element shape:
{{
  "temp_id": "cl_001",
  "subject": "captain",
  "predicate": "performs",
  "object": "orders the crew to evacuate",
  "claim_text": "The captain orders the crew to evacuate.",
  "scene_id": "{scene_id}",
  "chunk_id": "{chunk_id or scene_id}",
  "evidence": ["CAPTAIN: Everyone out, now!"],
  "confidence": 0.95
}}
"""
