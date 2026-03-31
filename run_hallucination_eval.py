"""Run claim-centric hallucination evaluation against a STAGE final graph."""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

from stage_kg.evaluation.claim_verifier import KnowledgeGraphVerifier
from stage_kg.evaluation.config import EvaluationConfig
from stage_kg.evaluation.hallucination_eval import HallucinationEvaluator
from stage_kg.evaluation.new_claim_extractor import Claim, ClaimExtractor
from stage_kg.ingest.loader import load_movie


ROOT = Path("/Users/mobiletest4/Downloads/Github/STAGE-Evaluation-Pipeline")
DEFAULT_GRAPH_PATH = ROOT / "output_stage_fixed_gemini_small" / "en04052c0f20834cf1bac19927d8f758e0" / "final_graph.json"
DEFAULT_OUTPUT_DIR = ROOT / "hallucination_eval_output"
DEFAULT_GEMINI_KEY_PATH = ROOT / "gemini.txt"


def read_gemini_api_key(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Gemini key file not found: {path}")
    key = path.read_text(encoding="utf-8").strip()
    if not key or key == "PASTE_GEMINI_API_KEY_HERE":
        raise ValueError(f"Put your Gemini API key into {path} with no quotes or extra text.")
    return key


def infer_language_and_movie_dir(graph_path: Path) -> Tuple[str, Path, str]:
    movie_id = graph_path.parent.name
    language = "zh" if movie_id.lower().startswith("ch") else "en"
    language_dir = "Chinese" if language == "zh" else "English"
    movie_dir = ROOT / language_dir / movie_id
    return movie_id, movie_dir, language


def save_claims_csv(claims, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "claim_id",
                "scene_id",
                "chunk_id",
                "sentence_idx",
                "subject",
                "predicate",
                "object",
                "claim_text",
                "confidence",
                "evidence",
            ],
        )
        writer.writeheader()
        for claim in claims:
            writer.writerow(
                {
                    "claim_id": claim.claim_id,
                    "scene_id": claim.scene_id,
                    "chunk_id": claim.chunk_id,
                    "sentence_idx": claim.sentence_idx,
                    "subject": claim.subject,
                    "predicate": claim.predicate,
                    "object": claim.object,
                    "claim_text": claim.claim_text,
                    "confidence": claim.confidence,
                    "evidence": " || ".join(claim.evidence or []),
                }
            )


def load_claims_csv(path: Path) -> List[Claim]:
    claims: List[Claim] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            evidence = [item.strip() for item in row.get("evidence", "").split("||") if item.strip()]
            claims.append(
                Claim(
                    subject=row.get("subject", ""),
                    predicate=row.get("predicate", ""),
                    object=row.get("object", ""),
                    claim_text=row.get("claim_text", ""),
                    scene_id=row.get("scene_id", ""),
                    sentence_idx=int(row.get("sentence_idx", 0) or 0),
                    chunk_id=row.get("chunk_id", ""),
                    evidence=evidence,
                    confidence=float(row.get("confidence", 0.8) or 0.8),
                    claim_id=row.get("claim_id", row.get("id", "")),
                )
            )
    return claims


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hallucination evaluation on screenplay scenes.")
    parser.add_argument("--graph-path", default=str(DEFAULT_GRAPH_PATH), help="Path to final_graph.json")
    parser.add_argument("--max-scenes", type=int, default=1, help="How many scenes to evaluate")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to save outputs")
    parser.add_argument("--gemini-key-file", default=str(DEFAULT_GEMINI_KEY_PATH), help="Path to gemini.txt")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    parser.add_argument("--claims-csv", default="", help="Optional existing claims CSV to verify instead of extracting new claims")
    args = parser.parse_args()

    graph_path = Path(args.graph_path)
    output_dir = Path(args.output_dir)
    key_path = Path(args.gemini_key_file)

    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}")

    movie_id, movie_dir, language = infer_language_and_movie_dir(graph_path)
    if not movie_dir.exists():
        raise FileNotFoundError(f"Movie directory not found for graph {graph_path}: {movie_dir}")

    config = EvaluationConfig()
    verifier = KnowledgeGraphVerifier(str(graph_path), EvaluationConfig())
    claims_csv = Path(args.claims_csv) if args.claims_csv else None

    if claims_csv:
        if not claims_csv.exists():
            raise FileNotFoundError(f"Claims CSV not found: {claims_csv}")
        all_claims = load_claims_csv(claims_csv)
        if not all_claims:
            raise ValueError(f"No claims loaded from {claims_csv}")
        evaluator = HallucinationEvaluator(claim_extractor=None, config=config)
        all_results = [verifier.verify_claim(claim) for claim in all_claims]
        summary_metrics = evaluator._compute_metrics(all_results, [], [])
        scene_ids = sorted({claim.scene_id for claim in all_claims})
    else:
        api_key = read_gemini_api_key(key_path)
        movie = load_movie(movie_dir=movie_dir, movie_id=movie_id, language=language)
        scenes = movie.scenes[: max(0, args.max_scenes)]
        if not scenes:
            raise ValueError("No scenes selected for evaluation.")

        claim_extractor = ClaimExtractor.for_gemini(api_key=api_key, model=args.model)
        evaluator = HallucinationEvaluator(
            claim_extractor=claim_extractor,
            config=config,
        )

        all_results = []
        all_claims = []

        for scene in scenes:
            print(f"Evaluating scene {scene.scene_id}: {scene.title}")
            claims = claim_extractor.extract_claims(
                text=scene.content,
                scene_id=scene.scene_id,
                scene_title=scene.title,
                scene_summary=scene.summary,
                movie_id=movie_id,
                movie_title=movie.title,
            )
            all_claims.extend(claims)
            metrics, results, _, _ = evaluator.evaluate_scene(scene.content, scene.scene_id, verifier)
            print(
                f"  claims={len(claims)} grounded={metrics.grounded_claims} "
                f"multihop={metrics.grounded_multihop_claims} hallucinated={metrics.hallucinated_claims}"
            )
            all_results.extend(results)

        summary_metrics = evaluator._compute_metrics(all_results, [], [])
        scene_ids = [scene.scene_id for scene in scenes]

    output_dir.mkdir(parents=True, exist_ok=True)
    save_claims_csv(all_claims, output_dir / "claims.csv")
    evaluator.save_results(all_results, str(output_dir / "hallucination_details.csv"))
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "graph_path": str(graph_path),
                "movie_id": movie_id,
                "max_scenes": args.max_scenes,
                "scene_ids": scene_ids,
                "claims_csv": str(claims_csv) if claims_csv else None,
                "metrics": asdict(summary_metrics),
            },
            f,
            indent=2,
        )

    print(f"Saved summary to {output_dir / 'summary.json'}")
    print(f"Saved claims to {output_dir / 'claims.csv'}")
    print(f"Saved details to {output_dir / 'hallucination_details.csv'}")


if __name__ == "__main__":
    main()
