"""Self-consistency evaluation (Metric 2)."""

import csv
import re
from typing import Dict, List, Sequence, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

from stage_kg.evaluation.claim_extractor import ClaimExtractor, Claim
from stage_kg.evaluation.config import EvaluationConfig


# Matches the same scene-heading patterns as repetition_eval.py
SCENE_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"(?:scene|sc)\s*[\w.-]+(?:\s*[-:]\s*.+)?"
    r"|(?:int|ext|int/ext|i/e)\.\s+.+"
    r"|\d+\s*[\).:-]\s+.+"
    r")\s*$",
    re.IGNORECASE,
)

SceneInput = Union[str, Sequence[str], Dict[str, str]]


@dataclass
class ConsistencyMetrics:
    """Self-consistency metrics."""
    total_scene_pairs: int
    total_character_pairs: int
    total_possible_comparisons: int
    contradictions: int
    state_changes_without_events: int
    contradiction_rate: float

    by_character: Dict[str, Dict] = None  # character_id -> {contradictions, drift_instances}

    def to_dict(self):
        return {
            "total_scene_pairs": self.total_scene_pairs,
            "total_character_pairs": self.total_character_pairs,
            "total_possible_comparisons": self.total_possible_comparisons,
            "contradictions": self.contradictions,
            "state_changes_without_events": self.state_changes_without_events,
            "contradiction_rate": self.contradiction_rate,
            "by_character": self.by_character or {},
        }


@dataclass
class ConsistencyViolation:
    """A detected consistency violation."""
    character_id: str
    scene_t: str
    scene_tk: str
    claim_t: str
    claim_tk: str
    violation_type: str  # "contradiction", "state_change_without_event", "attribute_mismatch"
    severity: float  # 0-1
    explanation: str


class ConsistencyEvaluator:
    """Evaluate self-consistency across scenes.

    Accepts full story text, a list of scene texts, or a flat
    {scene_id: scene_text} dict — the same SceneInput contract as
    RepetitionEvaluator.  Nested {scene_id: {char_id: text}} dicts raise
    TypeError so callers get a clear error instead of silent wrong results.
    """

    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.claim_extractor = ClaimExtractor()

    # ── public API ────────────────────────────────────────────────────────────

    def evaluate_consistency(
        self,
        scenes: SceneInput,
    ) -> "Tuple[ConsistencyMetrics, List[ConsistencyViolation]]":
        """Evaluate consistency across scene texts.

        Args:
            scenes: full story text (str), a list of scene texts, or a flat
                    {scene_id: scene_text} dict.

        Returns:
            (ConsistencyMetrics, list of ConsistencyViolation)
        """
        scene_map = self._parse_scene_input(scenes)
        scene_ids = list(scene_map.keys())

        # Extract claims per scene (each full scene treated as one unit)
        claims_by_scene: Dict[str, List[Claim]] = {}
        for scene_id, text in scene_map.items():
            claims_by_scene[scene_id] = self.claim_extractor.extract_claims(
                text, scene_id, "scene"
            )

        violations: List[ConsistencyViolation] = []
        for i, scene_t in enumerate(scene_ids):
            for scene_tk in scene_ids[i + 1:]:
                pair_violations = self._check_contradiction(
                    "scene", scene_t, scene_tk,
                    claims_by_scene.get(scene_t, []),
                    claims_by_scene.get(scene_tk, []),
                )
                violations.extend(pair_violations)

        total_scene_pairs = len(scene_ids) * (len(scene_ids) - 1) // 2 if scene_ids else 0
        metrics = ConsistencyMetrics(
            total_scene_pairs=total_scene_pairs,
            total_character_pairs=total_scene_pairs,
            total_possible_comparisons=total_scene_pairs,
            contradictions=len(violations),
            state_changes_without_events=sum(
                1 for v in violations if v.violation_type == "state_change_without_event"
            ),
            contradiction_rate=len(violations) / max(1, total_scene_pairs),
            by_character={"scene": {
                "contradictions": len(violations),
                "drift_instances": sum(
                    1 for v in violations
                    if v.violation_type == "state_change_without_event"
                ),
            }},
        )

        return metrics, violations

    def save_violations(
        self,
        violations: List[ConsistencyViolation],
        output_path: str,
    ):
        """Save violations to CSV."""
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "character_id", "scene_t", "scene_tk", "claim_t", "claim_tk",
                "violation_type", "severity", "explanation",
            ])
            writer.writeheader()
            for v in violations:
                writer.writerow({
                    "character_id": v.character_id,
                    "scene_t": v.scene_t,
                    "scene_tk": v.scene_tk,
                    "claim_t": v.claim_t,
                    "claim_tk": v.claim_tk,
                    "violation_type": v.violation_type,
                    "severity": v.severity,
                    "explanation": v.explanation,
                })

    # ── input parsing ─────────────────────────────────────────────────────────

    def _parse_scene_input(self, scenes: SceneInput) -> Dict[str, str]:
        """Return an ordered {scene_id: scene_text} dict."""
        if isinstance(scenes, str):
            return dict(self._split_story_into_scenes(scenes))

        if isinstance(scenes, dict):
            if any(isinstance(v, dict) for v in scenes.values()):
                raise TypeError(
                    "ConsistencyEvaluator now accepts full scene text only. "
                    "Pass {scene_id: scene_text}, not {scene_id: {character_id: text}}."
                )
            if not all(isinstance(v, str) for v in scenes.values()):
                raise TypeError("Scene dictionaries must be {scene_id: full_scene_text}.")
            return {
                str(sid): text.strip()
                for sid, text in scenes.items()
                if isinstance(text, str) and text.strip()
            }

        if isinstance(scenes, Sequence):
            return {
                f"scene_{idx:03d}": text.strip()
                for idx, text in enumerate(scenes, start=1)
                if isinstance(text, str) and text.strip()
            }

        raise TypeError("scenes must be a string, a sequence of strings, or {scene_id: text}.")

    def _split_story_into_scenes(self, story_text: str) -> List[Tuple[str, str]]:
        scenes: List[Tuple[str, str]] = []
        current_id = ""
        current_lines: List[str] = []
        heading_count = 0

        def flush():
            if current_id and current_lines:
                text = "\n".join(current_lines).strip()
                if text:
                    scenes.append((current_id, text))

        for raw_line in story_text.splitlines():
            line = raw_line.strip()
            if line and SCENE_HEADING_RE.match(line):
                flush()
                heading_count += 1
                current_id = self._scene_id_from_heading(line, heading_count)
                current_lines = [line]
                continue
            if current_id:
                current_lines.append(raw_line)

        flush()
        stripped = story_text.strip()
        return scenes or ([("scene_001", stripped)] if stripped else [])

    def _scene_id_from_heading(self, heading: str, index: int) -> str:
        normalized = re.sub(r"\s+", "_", heading.strip().lower())
        normalized = re.sub(r"[^a-z0-9_.-]+", "", normalized).strip("._-")
        return normalized or f"scene_{index:03d}"

    # ── contradiction detection ───────────────────────────────────────────────

    def _check_contradiction(
        self,
        char_id: str,
        scene_t: str,
        scene_tk: str,
        claims_t: List[Claim],
        claims_tk: List[Claim],
    ) -> "List[ConsistencyViolation]":
        violations = []
        for claim_t in claims_t:
            for claim_tk in claims_tk:
                if self._are_contradicting(claim_t, claim_tk):
                    violations.append(ConsistencyViolation(
                        character_id=char_id,
                        scene_t=scene_t,
                        scene_tk=scene_tk,
                        claim_t=claim_t.claim_text,
                        claim_tk=claim_tk.claim_text,
                        violation_type="contradiction",
                        severity=0.8,
                        explanation=(
                            f"Contradicting predicates: "
                            f"'{claim_t.predicate}' vs '{claim_tk.predicate}'"
                        ),
                    ))
        return violations

    def _are_contradicting(self, claim1: Claim, claim2: Claim) -> bool:
        """Return True if two claims carry directly opposing state or predicate."""
        if claim1.subject != claim2.subject:
            return False

        opposites = {
            "alive": "dead",
            "present": "absent",
            "together": "separated",
            "hostile": "friendly",
        }

        pred1 = claim1.predicate.lower()
        pred2 = claim2.predicate.lower()
        obj1 = claim1.object.lower()
        obj2 = claim2.object.lower()

        for key, val in opposites.items():
            # predicate carries the state word (original path)
            if (
                (key in pred1 and val in pred2)
                or (val in pred1 and key in pred2)
            ) and claim1.object == claim2.object:
                return True

            # object carries the state word — e.g. "is alive" vs "is dead"
            # (claim extractor puts the adjective in .object, predicate="experiences")
            if (key in obj1 and val in obj2) or (val in obj1 and key in obj2):
                return True

        return False
