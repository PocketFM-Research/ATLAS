"""Self-consistency evaluation (Metric 2)."""

import csv
import json
import re
from typing import Any, Dict, List, Sequence, Tuple, Union
from dataclasses import dataclass

from stage_kg.evaluation.config import EvaluationConfig
from stage_kg.llm.base import BaseLLM
from stage_kg.utils.json_repair import parse_llm_json


# Matches the same scene-heading patterns as repetition_eval.py
SCENE_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"(?:scene|sc)\s*[\w.-]+(?:\s*[-:]\s*.+)?"
    r"|(?:int|ext|int/ext|i/e)\.\s+.+"
    r"|\d+\s*[\).:：、-]\s*.+"
    r")\s*$",
    re.IGNORECASE,
)
NEGATION_RE = re.compile(r"\b(?:no|not|never|none|nothing|cannot|can't|isn't|aren't|wasn't|weren't|won't|don't|doesn't|didn't)\b")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
WORD_RE = re.compile(r"[a-z0-9']+")

SceneInput = Union[str, Sequence[str], Dict[str, str]]

TRANSITION_TERMS = {
    "become",
    "became",
    "change",
    "changed",
    "changes",
    "dead",
    "death",
    "die",
    "dies",
    "died",
    "divorce",
    "divorced",
    "divorces",
    "engaged",
    "healed",
    "injured",
    "kills",
    "killed",
    "leave",
    "leaves",
    "left",
    "married",
    "marries",
    "marry",
    "moves",
    "moved",
    "recovers",
    "recovered",
    "reconcile",
    "reconciles",
    "reconciled",
    "remarries",
    "remarry",
    "returns",
    "returned",
    "separate",
    "separated",
    "separates",
    "survives",
    "survived",
    "wounded",
}


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

    def __init__(
        self,
        config: EvaluationConfig = None,
        llm: BaseLLM = None,
        max_candidate_pairs: int = 250,
    ):
        self.config = config or EvaluationConfig()
        self.llm = llm
        self.max_candidate_pairs = max_candidate_pairs

    # ── public API ────────────────────────────────────────────────────────────

    def evaluate_consistency(
        self,
        scenes: SceneInput,
        claims: Sequence[Any] = None,
    ) -> "Tuple[ConsistencyMetrics, List[ConsistencyViolation]]":
        """Evaluate consistency across atomic claims.

        Args:
            scenes: full story text, list of scene texts, or {scene_id: scene_text}.
                    Used for scene ordering and metrics denominator.
            claims: Atomic claims from the hallucination claim-extraction pass.
                    When omitted, the evaluator cannot run LLM claim-pair
                    adjudication and falls back to an empty claim set.

        Returns:
            (ConsistencyMetrics, list of ConsistencyViolation)
        """
        scene_map = self._parse_scene_input(scenes)
        scene_ids = list(scene_map.keys())
        normalized_claims = self._normalize_claims(claims or [], scene_ids)
        candidate_pairs = self._candidate_pairs(normalized_claims)
        violations = self._adjudicate_candidate_pairs(candidate_pairs, normalized_claims)

        cross_scene_pairs = len(scene_ids) * (len(scene_ids) - 1) // 2 if scene_ids else 0
        total_scene_pairs = cross_scene_pairs + len(scene_ids)
        metrics = ConsistencyMetrics(
            total_scene_pairs=total_scene_pairs,
            total_character_pairs=len(candidate_pairs),
            total_possible_comparisons=len(candidate_pairs),
            contradictions=len(violations),
            state_changes_without_events=sum(
                1 for v in violations if v.violation_type == "state_change_without_event"
            ),
            contradiction_rate=len(violations) / max(1, len(candidate_pairs)),
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

    def _normalize_claims(self, claims: Sequence[Any], scene_ids: Sequence[str]) -> List[Dict[str, Any]]:
        scene_order = {scene_id: index for index, scene_id in enumerate(scene_ids)}
        normalized: List[Dict[str, Any]] = []
        seen = set()

        for index, claim in enumerate(claims):
            item = self._claim_to_dict(claim)
            subject = self._normalize_claim_part(item.get("subject", ""))
            predicate = self._normalize_claim_part(item.get("predicate", ""))
            obj = self._normalize_claim_part(item.get("object", ""))
            text = str(item.get("claim_text", "") or "").strip()
            scene_id = str(item.get("scene_id", "") or "")
            if not subject or not predicate or not obj or not text or not scene_id:
                continue

            key = (subject, predicate, obj, scene_id, int(item.get("sentence_idx", 0) or 0))
            if key in seen:
                continue
            seen.add(key)
            normalized.append({
                "id": str(item.get("id") or item.get("claim_id") or f"claim_{index:05d}"),
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "claim_text": text,
                "scene_id": scene_id,
                "sentence_idx": int(item.get("sentence_idx", 0) or 0),
                "scene_order": scene_order.get(scene_id, len(scene_order)),
                "confidence": float(item.get("confidence", 0.0) or 0.0),
            })

        return sorted(normalized, key=lambda item: (item["scene_order"], item["sentence_idx"], item["id"]))

    def _claim_to_dict(self, claim: Any) -> Dict[str, Any]:
        if isinstance(claim, dict):
            return claim
        if hasattr(claim, "to_dict"):
            return claim.to_dict()
        return {
            "id": getattr(claim, "claim_id", ""),
            "subject": getattr(claim, "subject", ""),
            "predicate": getattr(claim, "predicate", ""),
            "object": getattr(claim, "object", ""),
            "claim_text": getattr(claim, "claim_text", ""),
            "scene_id": getattr(claim, "scene_id", ""),
            "sentence_idx": getattr(claim, "sentence_idx", 0),
            "confidence": getattr(claim, "confidence", 0.0),
        }

    def _candidate_pairs(self, claims: Sequence[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for index, claim_i in enumerate(claims):
            for claim_j in claims[index + 1:]:
                if not self._should_compare_claims(claim_i, claim_j):
                    continue
                pairs.append((claim_i, claim_j))
                if len(pairs) >= self.max_candidate_pairs:
                    return pairs
        return pairs

    def _should_compare_claims(self, claim_i: Dict[str, Any], claim_j: Dict[str, Any]) -> bool:
        same_subject = claim_i["subject"] == claim_j["subject"]
        shared_subject_words = self._shared_content_words(claim_i["subject"], claim_j["subject"])
        shared_claim_words = self._shared_content_words(
            f"{claim_i['subject']} {claim_i['object']}",
            f"{claim_j['subject']} {claim_j['object']}",
        )
        if not same_subject and shared_subject_words == 0 and shared_claim_words < 2:
            return False

        same_relation = claim_i["predicate"] == claim_j["predicate"]
        related_relation = self._predicate_family(claim_i["predicate"]) == self._predicate_family(claim_j["predicate"])
        object_overlap = self._shared_content_words(claim_i["object"], claim_j["object"])
        possible_rule_conflict = self._rule_contradiction(claim_i, claim_j)
        return same_relation or related_relation or object_overlap > 0 or possible_rule_conflict

    def _predicate_family(self, predicate: str) -> str:
        if predicate in {"located_at", "occurs_at"}:
            return "location"
        if predicate in {"possesses", "owns", "has_attribute"}:
            return "attribute"
        if predicate in {"relationship", "interacts_with"}:
            return "relationship"
        return predicate

    def _adjudicate_candidate_pairs(
        self,
        candidate_pairs: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
        claims: Sequence[Dict[str, Any]],
    ) -> List[ConsistencyViolation]:
        if not candidate_pairs:
            return []

        if self.llm is None:
            return [
                self._violation_from_pair(claim_i, claim_j, "rule_based_contradiction", "Rule-based opposite/negation conflict.")
                for claim_i, claim_j in candidate_pairs
                if self._rule_contradiction(claim_i, claim_j)
            ]

        return self._llm_adjudicate_pairs(candidate_pairs, claims)

    def _llm_adjudicate_pairs(
        self,
        candidate_pairs: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
        claims: Sequence[Dict[str, Any]],
    ) -> List[ConsistencyViolation]:
        violations: List[ConsistencyViolation] = []
        batch_size = 20
        for batch_start in range(0, len(candidate_pairs), batch_size):
            batch = candidate_pairs[batch_start: batch_start + batch_size]
            prompt = self._build_llm_prompt(batch, claims)
            raw = self.llm.complete(
                prompt,
                system=(
                    "You are a strict story self-consistency judge. "
                    "Only mark contradiction=true when the two claims cannot both be true "
                    "in the same story world after considering scene order and the provided timeline context. "
                    "Do not mark justified state changes as contradictions when intervening context plausibly "
                    "explains the change, such as divorce, remarriage, relocation, injury, recovery, death, "
                    "reconciliation, or another transition event. Ignore mere tension, uncertainty, repeated wording, "
                    "and claims that could be true at different times unless the chronology makes them impossible. "
                    "Return JSON only."
                ),
                temperature=0.0,
                max_tokens=4096,
            )
            parsed = parse_llm_json(raw, schema_hint="consistency_pairs")
            decisions = parsed.get("decisions", parsed) if isinstance(parsed, dict) else parsed
            if not isinstance(decisions, list):
                continue
            pair_lookup = {f"pair_{idx}": pair for idx, pair in enumerate(batch)}
            for decision in decisions:
                if not isinstance(decision, dict) or not decision.get("contradiction"):
                    continue
                pair_id = str(decision.get("pair_id", ""))
                if pair_id not in pair_lookup:
                    continue
                confidence = float(decision.get("confidence", 0.0) or 0.0)
                if confidence < 0.65:
                    continue
                claim_i, claim_j = pair_lookup[pair_id]
                violations.append(self._violation_from_pair(
                    claim_i,
                    claim_j,
                    "llm_claim_contradiction",
                    str(decision.get("reason", "LLM judged the claims mutually inconsistent.")),
                    severity=confidence,
                ))
        return violations

    def _build_llm_prompt(
        self,
        batch: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
        claims: Sequence[Dict[str, Any]],
    ) -> str:
        pairs = []
        for index, (claim_i, claim_j) in enumerate(batch):
            pairs.append({
                "pair_id": f"pair_{index}",
                "claim_a": self._claim_for_prompt(claim_i),
                "claim_b": self._claim_for_prompt(claim_j),
                "timeline_context": [
                    self._claim_for_prompt(claim)
                    for claim in self._timeline_context_claims(claim_i, claim_j, claims)
                ],
            })
        return (
            "Judge whether each pair of atomic story claims is mutually inconsistent.\n"
            "A contradiction means both claims cannot be true in the same story world, "
            "given their scene order, wording, and the timeline_context claims between or near them. "
            "Use timeline_context to decide whether an apparent conflict is a justified state change. "
            "If context plausibly explains the change, return contradiction=false. "
            "Return JSON in this shape:\n"
            "{\"decisions\":[{\"pair_id\":\"pair_0\",\"contradiction\":false,"
            "\"confidence\":0.0,\"reason\":\"brief reason\"}]}\n\n"
            f"Claim pairs:\n{json.dumps(pairs, indent=2, ensure_ascii=False)}"
        )

    def _timeline_context_claims(
        self,
        claim_i: Dict[str, Any],
        claim_j: Dict[str, Any],
        claims: Sequence[Dict[str, Any]],
        max_context: int = 12,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant claims that may justify an apparent state change."""
        start = min(claim_i["scene_order"], claim_j["scene_order"])
        end = max(claim_i["scene_order"], claim_j["scene_order"])
        pair_ids = {claim_i["id"], claim_j["id"]}
        pair_subject_words = (
            self._content_word_set(claim_i["subject"])
            | self._content_word_set(claim_j["subject"])
        )
        pair_object_words = (
            self._content_word_set(claim_i["object"])
            | self._content_word_set(claim_j["object"])
        )
        pair_text_words = (
            self._content_word_set(claim_i["claim_text"])
            | self._content_word_set(claim_j["claim_text"])
        )
        pair_family = {
            self._predicate_family(claim_i["predicate"]),
            self._predicate_family(claim_j["predicate"]),
        }

        scored_context: List[Tuple[int, int, Dict[str, Any]]] = []
        for claim in claims:
            if claim["id"] in pair_ids:
                continue
            scene_order = claim["scene_order"]
            if scene_order < start or scene_order > end:
                continue

            subject_overlap = len(pair_subject_words & self._content_word_set(claim["subject"]))
            object_overlap = len(pair_object_words & self._content_word_set(claim["object"]))
            text_words = self._content_word_set(claim["claim_text"])
            text_overlap = len(pair_text_words & text_words)
            same_family = self._predicate_family(claim["predicate"]) in pair_family
            transition = bool(text_words & TRANSITION_TERMS)

            score = (
                subject_overlap * 4
                + object_overlap * 3
                + text_overlap
                + (3 if same_family else 0)
                + (4 if transition else 0)
            )
            if score <= 0:
                continue
            scored_context.append((score, abs(scene_order - start), claim))

        scored_context.sort(key=lambda item: (-item[0], item[1], item[2]["scene_order"], item[2]["sentence_idx"]))
        selected = [claim for _, _, claim in scored_context[:max_context]]
        return sorted(selected, key=lambda item: (item["scene_order"], item["sentence_idx"], item["id"]))

    def _claim_for_prompt(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": claim["id"],
            "scene_id": claim["scene_id"],
            "scene_order": claim["scene_order"],
            "sentence_idx": claim["sentence_idx"],
            "subject": claim["subject"],
            "predicate": claim["predicate"],
            "object": claim["object"],
            "text": claim["claim_text"],
        }

    def _violation_from_pair(
        self,
        claim_i: Dict[str, Any],
        claim_j: Dict[str, Any],
        violation_type: str,
        explanation: str,
        severity: float = 0.8,
    ) -> ConsistencyViolation:
        return ConsistencyViolation(
            character_id=claim_i["subject"] or "scene",
            scene_t=claim_i["scene_id"],
            scene_tk=claim_j["scene_id"],
            claim_t=claim_i["claim_text"],
            claim_tk=claim_j["claim_text"],
            violation_type=violation_type,
            severity=severity,
            explanation=explanation,
        )

    def _rule_contradiction(self, claim1: Dict[str, Any], claim2: Dict[str, Any]) -> bool:
        if claim1["subject"] != claim2["subject"]:
            return False

        opposites = (
            ("alive", "dead"),
            ("present", "absent"),
            ("together", "separated"),
            ("hostile", "friendly"),
            ("married", "divorced"),
            ("married", "unmarried"),
            ("signed", "unsigned"),
            ("open", "closed"),
            ("locked", "unlocked"),
            ("inside", "outside"),
            ("rich", "poor"),
            ("empty", "full"),
            ("silent", "speaking"),
            ("sober", "high"),
            ("sober", "drunk"),
        )

        pred1 = claim1["predicate"]
        pred2 = claim2["predicate"]
        obj1 = claim1["object"]
        obj2 = claim2["object"]
        text1 = self._normalize_claim_part(claim1["claim_text"])
        text2 = self._normalize_claim_part(claim2["claim_text"])

        shared_context = self._shared_content_words(text1, text2)

        for key, val in opposites:
            # predicate carries the state word (original path)
            if (
                (key in pred1 and val in pred2)
                or (val in pred1 and key in pred2)
            ) and claim1["object"] == claim2["object"]:
                return True

            if ((key in obj1 and val in obj2) or (val in obj1 and key in obj2)) and shared_context >= 1:
                return True

            if ((key in text1 and val in text2) or (val in text1 and key in text2)) and shared_context >= 2:
                return True

        if self._negation_conflict(text1, text2):
            return True

        if self._number_conflict(text1, text2):
            return True

        return False

    def _normalize_claim_part(self, value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9']+", str(value).lower()))

    def _negation_conflict(self, text1: str, text2: str) -> bool:
        neg1 = bool(NEGATION_RE.search(text1))
        neg2 = bool(NEGATION_RE.search(text2))
        if neg1 == neg2:
            return False
        words1 = self._content_word_set(text1)
        words2 = self._content_word_set(text2)
        return len(words1 & words2) >= 2

    def _number_conflict(self, text1: str, text2: str) -> bool:
        nums1 = set(NUMBER_RE.findall(text1))
        nums2 = set(NUMBER_RE.findall(text2))
        if not nums1 or not nums2 or nums1 == nums2:
            return False
        words1 = self._content_word_set(text1)
        words2 = self._content_word_set(text2)
        return len(words1 & words2) >= 2

    def _shared_content_words(self, text1: str, text2: str) -> int:
        return len(self._content_word_set(text1) & self._content_word_set(text2))

    def _content_word_set(self, text: str) -> set:
        stopwords = {
            "about", "after", "again", "because", "before", "being", "could",
            "does", "doing", "from", "have", "into", "just", "like", "more",
            "never", "only", "over", "same", "scene", "says", "that", "their",
            "there", "they", "this", "through", "with", "without", "would",
        }
        return {
            word
            for word in text.split()
            if len(word) > 3 and word not in stopwords and not NEGATION_RE.fullmatch(word)
        }
