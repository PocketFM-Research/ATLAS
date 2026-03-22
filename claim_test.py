"""Run claim extraction on a selected scene from the Star Trek movie."""

import csv
import json
from pathlib import Path

from stage_kg.evaluation.new_claim_extractor import ClaimExtractor
from stage_kg.evaluation.config import EvaluationConfig


DEFAULT_MOVIE_ID = "en04052c0f20834cf1bac19927d8f758e0"
DEFAULT_INPUT_DIR = Path("/Users/mobiletest4/Downloads/Github/STAGE-Evaluation-Pipeline")
DEFAULT_SCENE_ID = "1"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "test_claim_extractor"


def load_scene(script_path: Path, scene_id: str) -> dict:
    with script_path.open("r", encoding="utf-8") as f:
        scenes = json.load(f)

    for scene in scenes:
        if str(scene.get("_id", -1) + 1) == str(scene_id):
            return scene

    raise ValueError(f"Scene {scene_id} not found in {script_path}")


def main():
    script_path = DEFAULT_INPUT_DIR / "English" / DEFAULT_MOVIE_ID / "script.json"
    if not script_path.exists():
        raise FileNotFoundError(f"script.json not found: {script_path}")

    scene = load_scene(script_path, DEFAULT_SCENE_ID)
    scene_id = str(scene["_id"] + 1)
    scene_title = scene.get("title", "")
    scene_text = scene.get("content", "")

    extractor = ClaimExtractor()
    config = EvaluationConfig()
    claims = extractor.extract_claims(scene_text, scene_id)

    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_text_path = output_dir / "scene_1_text.txt"
    scene_text_path.write_text(scene_text, encoding="utf-8")

    csv_path = output_dir / "test_claim_extractor.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject",
                "predicate",
                "object",
                "relation",
                "relation_group",
                "sentence_idx",
            ],
        )
        writer.writeheader()
        for claim in claims:
            relation_group = config.get_relation_group(claim.predicate)
            writer.writerow(
                {
                    "subject": claim.subject,
                    "predicate": claim.predicate,
                    "object": claim.object,
                    "relation": claim.predicate,
                    "relation_group": relation_group,
                    "sentence_idx": claim.sentence_idx,
                }
            )
if __name__ == "__main__":
    main()
