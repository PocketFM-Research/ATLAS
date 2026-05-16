"""
Prompt templates for Merge Adjudication (Normalization Pass).

When string-similarity or embedding clustering suggests two nodes may refer
to the same real-world entity, this prompt asks an LLM to adjudicate.
"""

SYSTEM_PROMPT = """You are a careful knowledge graph curator adjudicating entity merge decisions.
Your output must be valid JSON only — no prose, no markdown fences, no explanation."""


def build_entity_merge_prompt(
    node_a: dict,
    node_b: dict,
    movie_title: str = "",
) -> str:
    import json

    movie_block = f"Movie: {movie_title}\n" if movie_title else ""

    return f"""{movie_block}Two entity nodes in a movie knowledge graph may refer to the same real-world entity.
Decide whether they should be merged.

NODE A:
{json.dumps(node_a, ensure_ascii=False, indent=2)}

NODE B:
{json.dumps(node_b, ensure_ascii=False, indent=2)}

INSTRUCTIONS:
1. Examine the name, aliases, description, type, and evidence for both nodes.
2. If they clearly refer to the same entity (same character, same place, same object), merge them.
3. If it is ambiguous or they clearly differ, do NOT merge.
4. Return a JSON object with:
   - "merge": true or false
   - "canonical_name": the best canonical name if merging (pick the most complete/formal form)
   - "reason": one-sentence explanation
   - "confidence": float 0.0–1.0

Return ONLY the JSON object. No other text.

Example:
{{"merge": true, "canonical_name": "Captain Reyes", "reason": "Both nodes refer to the same character, with one node using a shortened surface form and the other the full name.", "confidence": 0.97}}
"""


def build_event_merge_prompt(
    event_a: dict,
    event_b: dict,
    movie_title: str = "",
) -> str:
    import json

    movie_block = f"Movie: {movie_title}\n" if movie_title else ""

    return f"""{movie_block}Two event nodes in a movie knowledge graph may refer to the same narrative event.
Decide whether they should be merged.

EVENT A:
{json.dumps(event_a, ensure_ascii=False, indent=2)}

EVENT B:
{json.dumps(event_b, ensure_ascii=False, indent=2)}

INSTRUCTIONS:
1. Examine name, description, evidence, scene_id, and participants for both events.
2. Merge only if they clearly describe the same action in the same scene (or closely adjacent scenes).
3. Do NOT merge events that are merely similar in type but occur at different points in the story.
4. Return a JSON object with:
   - "merge": true or false
   - "canonical_name": best name if merging
   - "reason": one-sentence explanation
   - "confidence": float 0.0–1.0

Return ONLY the JSON object. No other text.
"""
