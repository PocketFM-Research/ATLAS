"""Run the Gemini-backed atomic claim extractor on a selected movie scene."""

import csv
import json
from pathlib import Path

from stage_kg.evaluation.config import EvaluationConfig
from stage_kg.evaluation.new_claim_extractor import ClaimExtractor


DEFAULT_MOVIE_ID = "en04052c0f20834cf1bac19927d8f758e0"
DEFAULT_INPUT_DIR = Path("/Users/mobiletest4/Downloads/Github/STAGE-Evaluation-Pipeline")
DEFAULT_SCENE_ID = "1"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "test_claim_extractor"
DEFAULT_GEMINI_KEY_PATH = DEFAULT_INPUT_DIR / "gemini.txt"


def load_scene(script_path: Path, scene_id: str) -> dict:
    with script_path.open("r", encoding="utf-8") as f:
        scenes = json.load(f)

    for scene in scenes:
        if str(scene.get("_id", -1) + 1) == str(scene_id):
            return scene

    raise ValueError(f"Scene {scene_id} not found in {script_path}")


def main():
    api_key = DEFAULT_GEMINI_KEY_PATH.read_text(encoding="utf-8").strip() if DEFAULT_GEMINI_KEY_PATH.exists() else ""
    if not api_key:
        raise EnvironmentError(
            f"Put your Gemini API key into {DEFAULT_GEMINI_KEY_PATH} with no quotes or extra text."
        )

    script_path = DEFAULT_INPUT_DIR / "English" / DEFAULT_MOVIE_ID / "script.json"
    if not script_path.exists():
        raise FileNotFoundError(f"script.json not found: {script_path}")

    scene = load_scene(script_path, DEFAULT_SCENE_ID)
    scene_id = str(scene["_id"] + 1)
    scene_title = scene.get("title", "")
    scene_text = scene.get("content", "")

    extractor = ClaimExtractor.for_gemini(api_key=api_key)
    config = EvaluationConfig()
    claims = extractor.extract_claims(
        text=scene_text,
        scene_id=scene_id,
        scene_title=scene_title,
        movie_id=DEFAULT_MOVIE_ID,
    )

    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_text_path = output_dir / f"scene_{scene_id}_text.txt"
    scene_text_path.write_text(scene_text, encoding="utf-8")

    csv_path = output_dir / f"scene_{scene_id}_claims.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "claim_id",
                "subject",
                "predicate",
                "object",
                "claim_text",
                "relation_group",
                "scene_id",
                "chunk_id",
                "sentence_idx",
                "confidence",
                "evidence",
            ],
        )
        writer.writeheader()
        for claim in claims:
            writer.writerow(
                {
                    "claim_id": claim.claim_id,
                    "subject": claim.subject,
                    "predicate": claim.predicate,
                    "object": claim.object,
                    "claim_text": claim.claim_text,
                    "relation_group": config.get_relation_group(claim.predicate),
                    "scene_id": claim.scene_id,
                    "chunk_id": claim.chunk_id,
                    "sentence_idx": claim.sentence_idx,
                    "confidence": claim.confidence,
                    "evidence": " || ".join(claim.evidence or []),
                }
            )

    print(f"Extracted {len(claims)} claims")
    print(f"Saved text to: {scene_text_path}")
    print(f"Saved claims to: {csv_path}")


if __name__ == "__main__":
    main()
