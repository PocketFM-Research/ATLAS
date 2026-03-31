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
1. Extract every grounded atomic claim supported by the text.
2. Each claim must contain exactly one subject, one predicate, and one object.
3. Prefer STAGE schema predicates when possible: {', '.join(RELATION_TYPES)}.
4. Keep subject and object short, canonical, and text-grounded.
5. Do not invent hidden motivations, unsupported implications, or off-screen facts.
6. Skip weak filler dialogue, greetings, or claims that are too vague to verify.
7. For each claim provide:
   - "temp_id": a short unique slug (e.g. "cl_001")
   - "subject": canonical subject string
   - "predicate": relation label
   - "object": canonical object string
   - "claim_text": one-sentence paraphrase of the fact
   - "scene_id": "{scene_id}"
   - "chunk_id": "{chunk_id or scene_id}"
   - "evidence": list of exact supporting text spans
   - "confidence": float 0.0–1.0
8. Return 0-8 claims depending on how much grounded content exists.
9. Keep evidence compact: include 1-2 short supporting spans per claim, not long passages.

Return ONLY a JSON array of claim objects. No other text.

Example element shape:
{{
  "temp_id": "cl_001",
  "subject": "saavik",
  "predicate": "performs",
  "object": "orders intercept course to kobayashi maru",
  "claim_text": "Saavik orders an intercept course to the Kobayashi Maru.",
  "scene_id": "{scene_id}",
  "chunk_id": "{chunk_id or scene_id}",
  "evidence": ["SAAVIK: Plot an intercept course for the Kobayashi Maru."],
  "confidence": 0.95
}}
"""
