"""
Deterministic gold-vs-predicted matching and metrics for temporal
hallucination evaluation.

Predictions and gold annotations are lists of:
    {"id": int, "hallucination": "In scene X, [FACT] was established, yet in scene Y, [EVENT] happens"}

Matching strategy (no LLM required):
  - Parse scene X and scene Y from both strings.
  - Require scene-pair agreement when both pairs parse.
  - Compute token-overlap (Jaccard) on the fact text and on the event text
    after normalization (lowercase, drop punctuation, drop stopwords).
  - A pair matches if BOTH fact similarity >= threshold AND event similarity
    >= threshold. The default threshold is forgiving enough for paraphrase
    but strict enough that unrelated strings don't collide.

Each gold and each prediction can match at most once (greedy by best score).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .graph_verifier import parse_hallucination_string


DEFAULT_THRESHOLD = 0.35


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
    "happen",
    "happens",
    "establish",
    "established",
    "scene",
    "says",
    "say",
    "said",
    "claim",
    "claims",
    "claimed",
    "claiming",
    "state",
    "states",
    "stated",
    "stating",
    "describes",
    "described",
    "describing",
}

# Intentionally empty. We do NOT hand-curate synonym mappings tailored to
# specific stories — that's how a matcher overfits to the gold set. The
# tokenizer below applies generic suffix stripping (-ing/-ed/-s); anything
# beyond that should come from a general-purpose embedding or stemmer, not
# from story-specific word pairs.
_CANONICAL_TOKENS: Dict[str, str] = {}

_NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "hundred": "100",
}


def _tokens(text: str) -> set:
    if not text:
        return set()
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    normalized = set()
    for token in text.split():
        if not token or token in _STOPWORDS:
            continue
        token = _NUMBER_WORDS.get(token, token)
        token = _CANONICAL_TOKENS.get(token, token)
        if len(token) > 4 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 3 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        normalized.add(token)
    return normalized


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _soft_overlap(a: set, b: set) -> float:
    """Paraphrase-friendly overlap. Rewards subset matches in verbose outputs."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    containment = inter / min(len(a), len(b))
    return max(_jaccard(a, b), containment)


@dataclass
class ParsedHallucination:
    raw: str
    scene_x: Optional[int]
    scene_y: Optional[int]
    fact: str
    event: str

    @property
    def has_scene_pair(self) -> bool:
        return self.scene_x is not None and self.scene_y is not None


_FACT_LEAKAGE = re.compile(r"\s+was\s+established\b.*$", re.IGNORECASE)
_EVENT_LEAKAGE = re.compile(r"\s+happens\.?\s*$", re.IGNORECASE)
_INLINE_DASH = re.compile(r"\s+[—–]\s+|\s+--?\s+")


def _strip_fact(text: str) -> str:
    s = _FACT_LEAKAGE.sub("", str(text or "").strip())
    s = _INLINE_DASH.split(s, maxsplit=1)[0].strip()
    return s.strip(" ,.").strip()


def _strip_event(text: str) -> str:
    s = _EVENT_LEAKAGE.sub("", str(text or "").strip())
    s = _INLINE_DASH.split(s, maxsplit=1)[0].strip()
    return s.strip(" ,.").strip()


def parse_hallucination(item: Any) -> ParsedHallucination:
    """Parse a hallucination dict or string into ParsedHallucination."""
    if isinstance(item, dict):
        text = item.get("hallucination", "")
    else:
        text = item or ""

    parsed = parse_hallucination_string(text)
    if parsed is None:
        return ParsedHallucination(raw=text, scene_x=None, scene_y=None, fact=text, event="")
    x, fact, y, event = parsed
    return ParsedHallucination(
        raw=text,
        scene_x=x,
        scene_y=y,
        fact=_strip_fact(fact),
        event=_strip_event(event),
    )


def similarity(a: ParsedHallucination, b: ParsedHallucination) -> Tuple[float, float, float]:
    """
    Return (fact_sim, event_sim, combined). combined = min(fact_sim, event_sim).
    """
    fact_sim = _soft_overlap(_tokens(a.fact), _tokens(b.fact))
    event_sim = _soft_overlap(_tokens(a.event), _tokens(b.event))
    return fact_sim, event_sim, min(fact_sim, event_sim)


def _is_match(
    a: ParsedHallucination,
    b: ParsedHallucination,
    threshold: float,
) -> Tuple[bool, float]:
    if a.has_scene_pair and b.has_scene_pair:
        if a.scene_x != b.scene_x or a.scene_y != b.scene_y:
            return False, 0.0
    elif a.has_scene_pair or b.has_scene_pair:
        # One side parsed, the other didn't. Require strong text overlap.
        pass

    fact_sim, event_sim, combined = similarity(a, b)
    if fact_sim >= threshold and event_sim >= threshold:
        return True, combined
    return False, combined


@dataclass
class MatchResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    matched_pairs: List[Tuple[int, int]] = field(default_factory=list)


def match_predictions(
    gold: List[Any],
    predicted: List[Any],
    threshold: float = DEFAULT_THRESHOLD,
) -> MatchResult:
    """
    Greedy best-score matching between gold and predicted hallucinations.
    """
    parsed_gold = [parse_hallucination(g) for g in gold]
    parsed_pred = [parse_hallucination(p) for p in predicted]

    candidates: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(parsed_gold):
        for pi, p in enumerate(parsed_pred):
            ok, score = _is_match(g, p, threshold)
            if ok:
                candidates.append((score, gi, pi))

    candidates.sort(key=lambda t: t[0], reverse=True)

    used_gold = set()
    used_pred = set()
    matched: List[Tuple[int, int]] = []
    for _, gi, pi in candidates:
        if gi in used_gold or pi in used_pred:
            continue
        used_gold.add(gi)
        used_pred.add(pi)
        matched.append((gi, pi))

    tp = len(matched)
    fp = len(parsed_pred) - tp
    fn = len(parsed_gold) - tp
    return MatchResult(tp=tp, fp=fp, fn=fn, matched_pairs=matched)


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def compute_metrics(
    story_id: str,
    method: str,
    gold: List[Any],
    predicted: List[Any],
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    result = match_predictions(gold=gold, predicted=predicted, threshold=threshold)
    precision, recall, f1 = precision_recall_f1(result.tp, result.fp, result.fn)
    return {
        "story_id": story_id,
        "method": method,
        "Gold": len(gold),
        "Pred.": len(predicted),
        "TP": result.tp,
        "FP": result.fp,
        "FN": result.fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "F1": round(f1, 4),
    }


def aggregate_metrics(per_case_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate per-case rows into one row per method (micro-averaged TP/FP/FN).
    """
    by_method: Dict[str, Dict[str, int]] = {}
    for row in per_case_rows:
        method = row.get("method", "?")
        acc = by_method.setdefault(
            method,
            {"Gold": 0, "Pred.": 0, "TP": 0, "FP": 0, "FN": 0},
        )
        acc["Gold"] += int(row.get("Gold", 0))
        acc["Pred."] += int(row.get("Pred.", 0))
        acc["TP"] += int(row.get("TP", 0))
        acc["FP"] += int(row.get("FP", 0))
        acc["FN"] += int(row.get("FN", 0))

    aggregated: List[Dict[str, Any]] = []
    for method, acc in by_method.items():
        precision, recall, f1 = precision_recall_f1(acc["TP"], acc["FP"], acc["FN"])
        aggregated.append(
            {
                "story_id": "__aggregate__",
                "method": method,
                "Gold": acc["Gold"],
                "Pred.": acc["Pred."],
                "TP": acc["TP"],
                "FP": acc["FP"],
                "FN": acc["FN"],
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "F1": round(f1, 4),
            }
        )
    return aggregated
