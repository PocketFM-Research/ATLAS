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
1. Extract explicit atomic factual claims made by the screenplay text, whether or not they seem plausible, grounded, realistic, or consistent with the story world.
2. A claim is atomic if it states one smallest verifiable fact. If a claim can be split into two facts, split it.
3. Each claim must have exactly one subject, one predicate, and one object.
4. Prefer STAGE schema predicates when possible: {', '.join(RELATION_TYPES)}.
5. Keep subjects and objects short, canonical, and text-grounded.
6. Do not invent hidden motives, implications, or off-screen facts.
7. Do not filter out strange, impossible, surreal, unsupported, or out-of-world details. If the text explicitly states them, extract them as claims so a later verifier can judge whether they are grounded.
8. You may combine nearby sentences only when they jointly support one single atomic fact.
9. You must split one sentence into multiple claims if it contains multiple separable facts.
10. Convert dialogue into the simplest verifiable factual claim when a character states a concrete fact about identity, relationship, possession, location, time, legal status, ability, prior action, or current state.
    - GOOD: "A: I have never owned a business." -> "A has never owned a business."
    - GOOD: "A: B has never represented me." -> "B has never represented A."
    - GOOD: "A: I never attended that school." -> "A never attended that school."
    - BAD: only extracting "A says something" when the spoken content is itself a concrete factual claim.
    - BAD: omitting a concrete dialogue assertion because it may be false, unsupported, or inconsistent.
11. If one sentence contains a normal action plus a bizarre or impossible detail, preserve the bizarre detail as its own claim.
    - GOOD: "An official pauses a proceeding so a group can decide by an impossible method." -> "The official pauses the proceeding." + "The group decides by an impossible method."
    - GOOD: "A trained bird lands on a character's shoulder and drops a legal document." -> "The trained bird lands on the character's shoulder." + "The trained bird drops a legal document."
    - BAD: extracting only the normal part and dropping the unusual object, location, method, or procedure.
12. Prefer graph-aligned minimal claims:
    - actions -> Character/Event
    - social relations -> Character/Character
    - object, location, concept claims only when explicitly supported
13. Keep event objects short and action-centered, but include enough detail to preserve the factual content being tested.
14. Exclude explicit quoted subclaims only when they are merely greetings, jokes, insults, vague banter, or wording with no stable factual content.
15. Skip filler greetings, weak banter, and vague claims.
16. General examples:
    - GOOD split: "A enters the room and hugs B" -> "A enters the room." + "A hugs B."
    - GOOD rewrite: "A says B should leave" -> "A orders B to leave." if that is the stable event.
    - GOOD factual dialogue rewrite: "A says she is divorced from B" -> "A is divorced from B."
    - BAD: "A says 'you should leave now.'" when the only content is a command and the speech act itself is not important.
    - BAD: "A has a complicated conversation with B about the problem."
17. For each claim provide:
   - "temp_id": a short unique slug (e.g. "cl_001")
   - "subject": canonical subject string
   - "predicate": relation label
   - "object": canonical object string
   - "claim_text": one-sentence paraphrase of the fact
   - "scene_id": "{scene_id}"
   - "chunk_id": "{chunk_id or scene_id}"
   - "evidence": list of exact supporting text spans
   - "confidence": float 0.0–1.0
18. `claim_text` should be clean plain English and semantically equivalent to the atomic fact.
19. Keep evidence compact: include 1-2 short supporting spans per claim.
20. Return 0-25 claims depending on how much explicit factual content exists.
21. If a candidate depends mainly on quote wording rather than a stable fact, omit it.

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
