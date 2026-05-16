"""Unified evaluation runner.

`EvaluationRunner` is the programmatic entry point for evaluating generated
scene text against a knowledge graph. Pass it a `final_graph.json` path and the
scene text; it runs hallucination, consistency, and repetition evaluation and
writes complete JSON/CSV outputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from stage_kg.evaluation.claim_verifier import KnowledgeGraphVerifier
from stage_kg.evaluation.config import EvaluationConfig
from stage_kg.evaluation.hallucination_eval import HallucinationEvaluator
from stage_kg.evaluation.llm_claim_extractor import ClaimExtractor
from stage_kg.llm import get_llm
from stage_kg.utils.cache import Cache

SceneTextInput = Union[str, Sequence[str], Dict[str, Union[str, Dict[str, str]]]]

SCENE_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"(?:scene|sc)\s*[\w.-]+(?:\s*[-:]\s*.+)?"
    r"|(?:int|ext|int/ext|i/e)\.\s+.+"
    r"|\d+\s*[\).:：、-]\s*.+"
    r")\s*$",
    re.IGNORECASE,
)


def object_to_dict(value: Any) -> Any:
    """Convert evaluator dataclasses into JSON-serializable dictionaries."""
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: object_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [object_to_dict(item) for item in value]
    return value


def load_scene_text(path: Union[str, Path]) -> SceneTextInput:
    """Load scene text from .txt or .json."""
    text_path = Path(path)
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")
    if text_path.suffix.lower() == ".json":
        return json.loads(text_path.read_text(encoding="utf-8"))
    return text_path.read_text(encoding="utf-8")


def normalize_scene_text(scene_text: SceneTextInput) -> Dict[str, str]:
    """Normalize supported text inputs to {scene_id: full_scene_text}."""
    if isinstance(scene_text, dict):
        normalized: Dict[str, str] = {}
        for scene_id, value in scene_text.items():
            if isinstance(value, dict):
                parts = [str(text).strip() for text in value.values() if str(text).strip()]
                if parts:
                    normalized[str(scene_id)] = "\n\n".join(parts)
            elif isinstance(value, str) and value.strip():
                normalized[str(scene_id)] = value.strip()
        return normalized

    if isinstance(scene_text, str):
        return dict(split_story_into_scenes(scene_text))

    if isinstance(scene_text, Sequence):
        return {
            f"scene_{index:03d}": text.strip()
            for index, text in enumerate(scene_text, start=1)
            if isinstance(text, str) and text.strip()
        }

    raise TypeError("scene_text must be text, list[str], {scene_id: text}, or {scene_id: {character_id: text}}.")


def split_story_into_scenes(story_text: str) -> List[Tuple[str, str]]:
    """Split a single story string on scene headings."""
    scenes: List[Tuple[str, str]] = []
    current_id = ""
    current_lines: List[str] = []
    heading_count = 0

    def flush() -> None:
        if current_id and current_lines:
            text = "\n".join(current_lines).strip()
            if text:
                scenes.append((current_id, text))

    for raw_line in story_text.splitlines():
        line = raw_line.strip()
        if line and SCENE_HEADING_RE.match(line):
            flush()
            heading_count += 1
            current_id = scene_id_from_heading(line, heading_count)
            current_lines = [line]
            continue
        if current_id:
            current_lines.append(raw_line)

    flush()
    stripped = story_text.strip()
    return scenes or ([("scene_001", stripped)] if stripped else [])


def scene_id_from_heading(heading: str, index: int) -> str:
    normalized = re.sub(r"\s+", "_", heading.strip().lower())
    normalized = re.sub(r"[^a-z0-9_.-]+", "", normalized).strip("._-")
    return normalized or f"scene_{index:03d}"


class EvaluationRunner:
    """Run the full eval suite for one KG and one scene-text input."""

    def __init__(
        self,
        graph_path: Union[str, Path],
        output_dir: Union[str, Path],
        config: Optional[EvaluationConfig] = None,
        provider: str = "openai",
        model: str = "gpt-5.4-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        claim_extractor: Optional[ClaimExtractor] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.graph_path = Path(graph_path)
        self.output_dir = Path(output_dir)
        self.config = config or EvaluationConfig()

        if not self.graph_path.exists():
            raise FileNotFoundError(f"Graph not found: {self.graph_path}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(cache_dir) if cache_dir else self.output_dir / ".cache" / "claim_extraction"
        self.claim_cache = Cache(self.cache_dir)
        self.verifier = KnowledgeGraphVerifier(str(self.graph_path), self.config)

        self.claim_extractor = claim_extractor or self._build_claim_extractor(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        self.hallucination_eval = HallucinationEvaluator(
            self.claim_extractor,
            self.config,
            cache=self.claim_cache,
        )

        from stage_kg.evaluation.consistency_eval import ConsistencyEvaluator
        from stage_kg.evaluation.repetition_eval import RepetitionEvaluator

        self.consistency_eval = ConsistencyEvaluator(
            self.config,
            llm=getattr(self.claim_extractor, "llm", None),
        )
        self.repetition_eval = RepetitionEvaluator(self.config)

    def run(self, scene_text: SceneTextInput, save_outputs: bool = True) -> Dict[str, Any]:
        """Run hallucination, consistency, and repetition evaluation."""
        scenes = normalize_scene_text(scene_text)
        if not scenes:
            raise ValueError("No scene text found after normalization.")
        self.hallucination_eval.cache_namespace = self._cache_namespace(scenes)

        hallucination_metrics, hallucination_details = self._run_hallucination(scenes)
        consistency_claims = [
            result.claim
            for result in hallucination_details["_raw_results"]
        ]
        consistency_metrics, consistency_details = self._run_consistency(
            scenes,
            consistency_claims,
        )
        repetition_metrics, repetition_details = self._run_repetition(scenes)

        output = {
            "metadata": {
                "graph_path": str(self.graph_path),
                "scene_count": len(scenes),
                "claim_cache_dir": str(self.cache_dir),
                "claim_cache_namespace": self.hallucination_eval.cache_namespace,
            },
            "input": {
                "scene_texts": scenes,
            },
            "metrics": {
                "hallucination": hallucination_metrics,
                "consistency": consistency_metrics,
                "repetition": repetition_metrics,
            },
            "details": {
                "hallucination": hallucination_details,
                "consistency": consistency_details,
                "repetition": repetition_details,
            },
        }

        if save_outputs:
            self.save_outputs(output)

        return self._json_safe_output(output)

    def save_outputs(self, output: Dict[str, Any]) -> None:
        """Write summary/detail JSON and detailed CSV files."""
        summary = {
            "metadata": output["metadata"],
            "metrics": output["metrics"],
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (self.output_dir / "details.json").write_text(
            json.dumps(self._json_safe_output(output), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        hallucination = output["details"]["hallucination"]
        self.hallucination_eval.save_results(
            hallucination["_raw_results"],
            str(self.output_dir / "hallucination_details.csv"),
            abstentions=hallucination["_raw_abstentions"],
            abstentions_output_path=str(self.output_dir / "hallucination_abstentions.csv"),
            low_confidence_claims=hallucination["_raw_low_confidence_claims"],
            low_confidence_output_path=str(self.output_dir / "hallucination_low_confidence.csv"),
        )

        consistency = output["details"]["consistency"]
        self.consistency_eval.save_violations(
            consistency["_raw_violations"],
            str(self.output_dir / "consistency_violations.csv"),
        )

        repetition = output["details"]["repetition"]
        self.repetition_eval.save_instances(
            repetition["_raw_instances"],
            str(self.output_dir / "repetition_instances.csv"),
        )
        self._save_internal_repetition_csv(
            repetition["internal_repetitions"],
            self.output_dir / "internal_repetition_instances.csv",
        )

    def _build_claim_extractor(
        self,
        provider: str,
        model: str,
        api_key: Optional[str],
        base_url: Optional[str],
    ) -> ClaimExtractor:
        resolved_api_key = api_key or self._api_key_from_environment(provider)
        if not resolved_api_key:
            raise ValueError(
                "Hallucination evaluation requires an API key. "
                "Pass --api_key or set the provider environment variable."
            )

        llm = get_llm(
            provider=provider,
            model=model,
            api_key=resolved_api_key,
            base_url=base_url,
        )
        return ClaimExtractor(llm=llm)

    def _api_key_from_environment(self, provider: str) -> Optional[str]:
        env_var = {
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }.get(provider)
        return os.environ.get(env_var) if env_var else None

    def _cache_namespace(self, scenes: Dict[str, str]) -> str:
        fingerprint = hashlib.sha256()
        fingerprint.update(str(self.graph_path.resolve()).encode("utf-8"))
        for scene_id, text in sorted(scenes.items()):
            fingerprint.update(scene_id.encode("utf-8"))
            fingerprint.update(b"\0")
            fingerprint.update(text.encode("utf-8"))
            fingerprint.update(b"\0")
        return f"{self.graph_path.stem}_{fingerprint.hexdigest()[:16]}"

    def _run_hallucination(self, scenes: Dict[str, str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        all_results = []
        all_abstentions = []
        all_low_confidence_claims = []

        for scene_id, text in scenes.items():
            _, results, abstentions, low_confidence_claims = self.hallucination_eval.evaluate_scene(
                text,
                scene_id,
                self.verifier,
            )
            all_results.extend(results)
            all_abstentions.extend(abstentions)
            all_low_confidence_claims.extend(low_confidence_claims)

        metrics = self.hallucination_eval._compute_metrics(
            all_results,
            all_abstentions,
            all_low_confidence_claims,
        )
        return object_to_dict(metrics), {
            "claims": [object_to_dict(result.claim) for result in all_results],
            "verification_results": [object_to_dict(result) for result in all_results],
            "abstentions": [object_to_dict(item) for item in all_abstentions],
            "low_confidence_claims": [object_to_dict(item) for item in all_low_confidence_claims],
            "_raw_results": all_results,
            "_raw_abstentions": all_abstentions,
            "_raw_low_confidence_claims": all_low_confidence_claims,
        }

    def _run_consistency(
        self,
        scenes: Dict[str, str],
        claims: Sequence[Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        metrics, violations = self.consistency_eval.evaluate_consistency(scenes, claims=claims)
        return object_to_dict(metrics), {
            "violations": [object_to_dict(violation) for violation in violations],
            "_raw_violations": violations,
        }

    def _run_repetition(self, scenes: Dict[str, str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        metrics, instances = self.repetition_eval.evaluate_repetition(scenes)
        internal_repetitions = self.repetition_eval.evaluate_internal_repetition(scenes)
        internal_metrics = self._internal_repetition_metrics(internal_repetitions, len(scenes))
        repetition_metrics = object_to_dict(metrics)
        repetition_metrics["internal"] = internal_metrics
        internal_instances = self._internal_repetition_instances(internal_repetitions)
        return repetition_metrics, {
            "instances": [object_to_dict(instance) for instance in instances],
            "detected_repetitions": [
                object_to_dict(instance)
                for instance in instances
                if instance.is_repetitive
            ],
            "internal_metrics": internal_metrics,
            "internal_repetitions": [object_to_dict(instance) for instance in internal_repetitions],
            "internal_repetition_instances": internal_instances,
            "_raw_instances": instances,
        }

    def _internal_repetition_metrics(
        self,
        internal_repetitions: List[Any],
        scene_count: int,
    ) -> Dict[str, Any]:
        scores = [
            max(
                float(getattr(item, "internal_sentence_repetition", 0.0)),
                float(getattr(item, "internal_ngram_repetition", 0.0)),
            )
            for item in internal_repetitions
        ]
        return {
            "total_scenes": scene_count,
            "repetitive_scenes": len(internal_repetitions),
            "explicit_repetition_instances": sum(
                len(getattr(item, "repeated_sentence_instances", []))
                + len(getattr(item, "repeated_phrase_instances", []))
                for item in internal_repetitions
            ),
            "internal_repetition_rate": len(internal_repetitions) / max(1, scene_count),
            "avg_internal_repetition_score": sum(scores) / len(scores) if scores else 0.0,
            "max_internal_repetition_score": max(scores) if scores else 0.0,
        }

    def _internal_repetition_instances(self, internal_repetitions: List[Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for item in internal_repetitions:
            score = max(
                float(getattr(item, "internal_sentence_repetition", 0.0)),
                float(getattr(item, "internal_ngram_repetition", 0.0)),
            )
            for sentence in getattr(item, "repeated_sentence_instances", []):
                rows.append({
                    "scene_id": item.scene_id,
                    "instance_type": "sentence",
                    "repeated_text": sentence.get("text", ""),
                    "count": sentence.get("count", 0),
                    "positions": sentence.get("positions", []),
                    "internal_repetition_score": score,
                    "sample_text": item.sample_text,
                })
            for phrase in getattr(item, "repeated_phrase_instances", []):
                rows.append({
                    "scene_id": item.scene_id,
                    "instance_type": "phrase",
                    "repeated_text": phrase.get("text", ""),
                    "count": phrase.get("count", 0),
                    "positions": phrase.get("positions", []),
                    "internal_repetition_score": score,
                    "sample_text": item.sample_text,
                })
        return rows

    def _save_internal_repetition_csv(
        self,
        internal_repetitions: List[Dict[str, Any]],
        output_path: Path,
    ) -> None:
        with output_path.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "scene_id",
                    "instance_type",
                    "repeated_text",
                    "count",
                    "positions",
                    "source",
                    "internal_repetition_score",
                    "sample_text",
                ],
            )
            writer.writeheader()
            for item in internal_repetitions:
                score = max(
                    item.get("internal_sentence_repetition", 0.0),
                    item.get("internal_ngram_repetition", 0.0),
                )
                for sentence in item.get("repeated_sentence_instances", []):
                    writer.writerow({
                        "scene_id": item.get("scene_id", ""),
                        "instance_type": "sentence",
                        "repeated_text": sentence.get("text", ""),
                        "count": sentence.get("count", 0),
                        "positions": ";".join(str(pos) for pos in sentence.get("positions", [])),
                        "source": sentence.get("source", ""),
                        "internal_repetition_score": score,
                        "sample_text": item.get("sample_text", ""),
                    })
                for phrase in item.get("repeated_phrase_instances", []):
                    writer.writerow({
                        "scene_id": item.get("scene_id", ""),
                        "instance_type": "phrase",
                        "repeated_text": phrase.get("text", ""),
                        "count": phrase.get("count", 0),
                        "positions": ";".join(str(pos) for pos in phrase.get("positions", [])),
                        "source": phrase.get("source", ""),
                        "internal_repetition_score": score,
                        "sample_text": item.get("sample_text", ""),
                    })

    def _json_safe_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        safe_output = object_to_dict(output)
        for detail in safe_output.get("details", {}).values():
            for key in list(detail.keys()):
                if key.startswith("_raw_"):
                    detail.pop(key)
        return safe_output


UnifiedEvaluationRunner = EvaluationRunner
