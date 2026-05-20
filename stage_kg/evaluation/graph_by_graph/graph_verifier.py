"""
Scene-level temporal hallucination detector.

Compares each later scene graph against the union of all prior scene graphs.
Flags two kinds of continuity errors:

1. Direct contradictions:
   A later-scene fact contradicts an explicitly established prior-scene
   fact on a durable attribute (color, count, side, identity, role, date,
   life/death status, etc.).

2. Unsupported continuations:
   A later-scene event/relation presupposes a prior state that was never
   established and that the current scene does not introduce on-screen.

Every detection is rendered exactly as:
   "In scene X, [FACT] was established, yet in scene Y, [EVENT] happens"

Uses the LLM (Azure GPT-5.4 mini) to adjudicate ambiguous cases. The graph
structure is the input signal; the LLM only judges whether prior graphs
support or contradict a later-graph fact.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ...llm.base import BaseLLM
from ...utils.json_repair import parse_llm_json

logger = logging.getLogger(__name__)


HALLUCINATION_FORMAT = (
    "In scene {x}, {fact} was established, yet in scene {y}, {event} happens"
)

TRANSIENT_RELATIONS = {
    "occurs_at",
    "located_at",
    "performs",
    "participates_in",
    "witnesses",
}

# Generic attribute-category words. Used only to gate which graph nodes
# carry continuity-relevant state; no story-specific objects, years, or
# proper-noun cues.
DURABLE_FACT_TERMS = {
    # color words (any narrative has these)
    "black", "blue", "brown", "gold", "green", "grey", "gray",
    "orange", "pink", "purple", "red", "silver", "white", "yellow",
    # body-side words
    "left", "right",
    # ordinal/number words (digit forms are caught by the regex separately)
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "hundred", "thousand", "million",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth",
    # days of the week
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday",
    # generic time units
    "year", "years", "month", "months", "week", "weeks", "day", "days",
    "hour", "hours", "minute", "minutes",
    # kinship / life-status categories
    "alive", "dead", "born", "married", "single", "widow", "widower",
    "divorced", "spouse", "husband", "wife", "father", "mother",
    "parent", "parents", "son", "daughter", "child", "children",
    "brother", "sister", "sibling",
}


@dataclass
class SceneGraph:
    """A single scene's graph plus the scene number for ordering."""
    scene_number: int
    graph: Dict[str, Any]

    @property
    def nodes(self) -> List[Dict[str, Any]]:
        return list(self.graph.get("nodes", []))

    @property
    def edges(self) -> List[Dict[str, Any]]:
        return list(self.graph.get("edges", []))


# ---------------------------------------------------------------- loading


def load_scene_graphs(scene_graphs_dir: Path) -> List[SceneGraph]:
    """Load ordered scene graphs from <scene_graphs_dir>/scene_<NNN>/final_graph.json."""
    scene_graphs: List[SceneGraph] = []
    scene_graphs_dir = Path(scene_graphs_dir)
    if not scene_graphs_dir.exists():
        return scene_graphs

    for scene_dir in sorted(scene_graphs_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        m = re.match(r"scene_(\d+)$", scene_dir.name)
        if not m:
            continue
        graph_path = scene_dir / "final_graph.json"
        if not graph_path.exists():
            continue
        with graph_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        scene_graphs.append(SceneGraph(scene_number=int(m.group(1)), graph=data))

    scene_graphs.sort(key=lambda s: s.scene_number)
    return scene_graphs


# ---------------------------------------------------------------- main API


def verify_scene_graphs(
    scene_graphs: List[SceneGraph],
    llm: BaseLLM,
    scene_texts: Optional[Dict[int, str]] = None,
    max_candidates_per_scene: int = 25,
    verify_with_text: bool = True,
    proposals_debug_path: Optional[Path] = None,
    scene_attributes: Optional[Dict[int, Dict[str, Dict[str, Any]]]] = None,
    audit_log_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Detect temporal hallucinations across an ordered list of scene graphs.

    Two-phase pipeline:
      1. Per-scene proposal: for each later scene, ask the LLM which of its
         graph-derived candidate facts contradict prior-scene graphs.
      2. Full-screenplay verification: if scene_texts are provided, re-check
         every proposed hallucination against the full screenplay text and
         drop ones the LLM cannot ground. This is the main FP filter.
    """
    if not scene_graphs:
        return []

    proposals: List[Dict[str, Any]] = []
    proposal_texts: set = set()
    accumulated: List[SceneGraph] = []

    # Audit trail — per-scene proposal stats, dedupe trail, verification
    # decisions. Saved at the end if audit_log_path is set.
    audit: Dict[str, Any] = {
        "per_scene_proposal_stats": [],
        "raw_proposals": [],
        "dedupe_log": [],
        "verification_log": [],
        "summary": {},
    }

    for current in scene_graphs:
        if not accumulated:
            accumulated.append(current)
            continue

        prior_summary = summarize_prior_graphs(
            accumulated,
            scene_attributes=scene_attributes,
            up_to_scene_exclusive=current.scene_number,
        )
        current_profile = (scene_attributes or {}).get(current.scene_number, {})
        candidate_facts = extract_candidate_facts(
            current, scene_attribute_profile=current_profile
        )
        if not candidate_facts:
            accumulated.append(current)
            continue

        if len(candidate_facts) > max_candidates_per_scene:
            candidate_facts = candidate_facts[:max_candidates_per_scene]

        current_scene_text = ""
        if scene_texts:
            current_scene_text = scene_texts.get(current.scene_number, "") or ""

        raw_findings = _ask_llm_for_hallucinations(
            llm=llm,
            current_scene_number=current.scene_number,
            current_scene_text=current_scene_text,
            current_facts=candidate_facts,
            prior_summary=prior_summary,
        )

        scene_proposals_added = 0
        scene_exact_dups_dropped = 0
        for finding in raw_findings:
            text = _render_finding(finding, default_y=current.scene_number)
            if not text:
                continue
            if text in proposal_texts:
                scene_exact_dups_dropped += 1
                continue
            proposal_texts.add(text)
            prop = {
                "x": _safe_int(finding.get("x")),
                "y": _safe_int(finding.get("y"), current.scene_number),
                "fact": _sanitize_fact(finding.get("fact", "")),
                "event": _sanitize_event(finding.get("event", "")),
                "text": text,
            }
            proposals.append(prop)
            audit["raw_proposals"].append(
                {**prop, "proposed_at_scene_y": current.scene_number}
            )
            scene_proposals_added += 1

        audit["per_scene_proposal_stats"].append(
            {
                "scene_number": current.scene_number,
                "candidate_facts_count": len(candidate_facts),
                "raw_findings_count": len(raw_findings),
                "proposals_added": scene_proposals_added,
                "exact_duplicates_dropped": scene_exact_dups_dropped,
            }
        )
        logger.info(
            "Scene %d proposal: %d candidates → %d raw findings → %d new proposals (%d exact dups)",
            current.scene_number,
            len(candidate_facts),
            len(raw_findings),
            scene_proposals_added,
            scene_exact_dups_dropped,
        )

        accumulated.append(current)

    # Dedupe near-duplicate proposals before verification. Two proposals with
    # the same (X, Y) scene pair and very high token overlap on fact ∪ event
    # are surface variations of the same contradiction — keep only the first.
    before_count = len(proposals)
    proposals, dedupe_decisions = _dedupe_proposals_with_trace(proposals)
    audit["dedupe_log"] = dedupe_decisions
    logger.info(
        "Dedupe: %d proposals → %d after deduplication (%d removed)",
        before_count,
        len(proposals),
        before_count - len(proposals),
    )

    # Save pre-verification proposals for debugging.
    if proposals_debug_path is not None:
        try:
            proposals_debug_path.parent.mkdir(parents=True, exist_ok=True)
            proposals_debug_path.write_text(
                json.dumps(proposals, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not write proposals debug file: %s", exc)

    # Phase 2: verification against the full screenplay text.
    if verify_with_text and scene_texts and proposals:
        debug_verify_path: Optional[Path] = None
        if proposals_debug_path is not None:
            debug_verify_path = (
                proposals_debug_path.parent / "graph_method_verification_raw.json"
            )
        confirmed, verification_decisions = _verify_against_full_text(
            llm=llm,
            proposals=proposals,
            scene_texts=scene_texts,
            debug_path=debug_verify_path,
            with_trace=True,
        )
        audit["verification_log"] = verification_decisions
    else:
        confirmed = proposals

    detections: List[Dict[str, Any]] = []
    next_id = 1
    for prop in confirmed:
        detections.append({"id": next_id, "hallucination": prop["text"]})
        next_id += 1

    audit["summary"] = {
        "raw_proposals": len(audit["raw_proposals"]),
        "after_dedupe": len(proposals),
        "confirmed": len(confirmed),
        "final_detections": len(detections),
    }

    if audit_log_path is not None:
        try:
            audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            audit_log_path.write_text(
                json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info("Audit log written to %s", audit_log_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not write audit log: %s", exc)

    return detections


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_DEDUPE_STOPWORDS = {
    "the", "a", "an", "is", "was", "of", "to", "in", "and", "with",
    "on", "at", "by", "for", "yet", "scene", "established", "happens",
    "happen", "as", "from", "his", "her", "their", "its", "be", "are",
    "were", "has", "have", "had",
}


def _proposal_tokens(prop: Dict[str, Any]) -> set:
    text = f"{prop.get('fact', '')} {prop.get('event', '')}".lower()
    return {t for t in re.findall(r"[a-z0-9]+", text) if t not in _DEDUPE_STOPWORDS}


def _proposal_entity_keys(prop: Dict[str, Any]) -> set:
    """
    Pull likely-entity tokens from the original fact + event text. Heuristic:
    capitalized tokens in the raw text are named entities (people, places,
    objects). Same scene pair + shared named-entity is a strong same-
    contradiction signal even when verbs/subjects rephrase.
    """
    text = f"{prop.get('fact', '')} {prop.get('event', '')}"
    toks = re.findall(r"\b([A-Z][a-zA-Z][a-zA-Z]+)\b", text)
    return {t.lower() for t in toks} - _DEDUPE_STOPWORDS


def _dedupe_proposals_with_trace(
    proposals: List[Dict[str, Any]],
    jaccard_threshold: float = 0.5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Same as _dedupe_proposals but also returns a per-decision audit trail."""
    decisions: List[Dict[str, Any]] = []
    if not proposals:
        return proposals, decisions

    kept: List[Dict[str, Any]] = []
    kept_meta: List[Tuple[int, int, set, set, int]] = []

    for idx, prop in enumerate(proposals):
        x, y = int(prop.get("x", 0)), int(prop.get("y", 0))
        toks = _proposal_tokens(prop)
        ents = _proposal_entity_keys(prop)

        is_dupe = False
        dupe_of_idx = None
        dupe_reason = ""
        for kx, ky, ktoks, kents, k_orig_idx in kept_meta:
            if kx != x or ky != y:
                continue
            if not toks or not ktoks:
                continue
            inter = len(toks & ktoks)
            union = len(toks | ktoks)
            if union == 0:
                continue
            jac = inter / union
            if jac >= jaccard_threshold:
                is_dupe = True
                dupe_of_idx = k_orig_idx
                dupe_reason = f"token jaccard {jac:.2f} ≥ {jaccard_threshold}"
                break
            if ents and kents and (ents & kents) and jac >= 0.30:
                is_dupe = True
                dupe_of_idx = k_orig_idx
                dupe_reason = f"shared entity {sorted(ents & kents)} + jaccard {jac:.2f}"
                break

        if is_dupe:
            decisions.append(
                {
                    "raw_proposal_index": idx,
                    "decision": "drop",
                    "duplicate_of_raw_index": dupe_of_idx,
                    "reason": dupe_reason,
                    "text": prop.get("text", ""),
                }
            )
            continue

        kept.append(prop)
        kept_meta.append((x, y, toks, ents, idx))
        decisions.append(
            {
                "raw_proposal_index": idx,
                "decision": "keep",
                "duplicate_of_raw_index": None,
                "reason": "",
                "text": prop.get("text", ""),
            }
        )

    return kept, decisions


def _dedupe_proposals(
    proposals: List[Dict[str, Any]],
    jaccard_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Drop near-duplicate proposals. Two proposals collapse if any of the
    following hold (all gated on same (X, Y) scene pair):
      - token-Jaccard on (fact ∪ event) ≥ jaccard_threshold; or
      - they share a named-entity token AND token-Jaccard ≥ 0.30.
    The named-entity-keyed pass catches duplicates the model phrases with
    different verbs/subjects but the same underlying entity-attribute pair.
    """
    if not proposals:
        return proposals

    kept: List[Dict[str, Any]] = []
    kept_meta: List[Tuple[int, int, set, set]] = []  # (x, y, tokens, entities)

    for prop in proposals:
        x, y = int(prop.get("x", 0)), int(prop.get("y", 0))
        toks = _proposal_tokens(prop)
        ents = _proposal_entity_keys(prop)

        is_dupe = False
        for kx, ky, ktoks, kents in kept_meta:
            if kx != x or ky != y:
                continue
            if not toks or not ktoks:
                continue
            inter = len(toks & ktoks)
            union = len(toks | ktoks)
            if union == 0:
                continue
            jac = inter / union
            if jac >= jaccard_threshold:
                is_dupe = True
                break
            # Entity-keyed second pass: a shared named entity + decent overlap
            # is also a same-contradiction signal.
            if ents and kents and (ents & kents) and jac >= 0.30:
                is_dupe = True
                break
        if is_dupe:
            continue
        kept.append(prop)
        kept_meta.append((x, y, toks, ents))

    if len(kept) < len(proposals):
        logger.info(
            "Dedupe: dropped %d near-duplicate proposals (kept %d)",
            len(proposals) - len(kept),
            len(kept),
        )
    return kept


# ---------------------------------------------------------------- verification


_VERIFY_SYSTEM = (
    "You are a careful temporal continuity auditor. You receive the full "
    "screenplay text and a list of candidate hallucinations from an automated "
    "proposal step. For each candidate, apply the following strict definition:\n"
    "\n"
    "A TEMPORAL HALLUCINATION exists if and only if ALL FOUR conditions hold:\n"
    "  (A) Both quotes refer to the SAME named entity (or an alias of it — "
    "      first name vs surname vs title vs role are the SAME entity).\n"
    "  (B) Both quotes assert a value for the SAME specific attribute of that "
    "      entity (color, side, count, year, name, role, ownership, "
    "      life-status, location, material, identity, etc).\n"
    "  (C) The two asserted values are DIFFERENT (a real value mismatch, not "
    "      a paraphrase of the same value).\n"
    "  (D) The later scene does NOT explicitly explain or justify the change "
    "      on-screen (no character says 'I moved it', 'they switched it', "
    "      'we rescheduled', 'I have reconsidered' — and no narrated stated "
    "      reason for why the value changed).\n"
    "\n"
    "If all four hold → CONFIRM. If any one fails → REJECT.\n"
    "Be precise: cite specific sentences. Do not infer; require evidence. "
    "Do not invent alternate continuities or separate story sections unless "
    "the screenplay explicitly labels them as separate continuities."
)


# We don't enforce a closed attribute taxonomy any more. Several real
# continuity errors don't fit one of a hand-picked category list (e.g.
# "warned about specific threat" vs "no warning"), and forcing the
# verifier to pick from a fixed list was killing real positives. We still
# REQUIRE the verifier to name an attribute — empty/blank is rejected — but
# we accept any non-empty label.


def _verify_against_full_text(
    llm: BaseLLM,
    proposals: List[Dict[str, Any]],
    scene_texts: Dict[int, str],
    debug_path: Optional[Path] = None,
    with_trace: bool = False,
    batch_size: int = 12,
):
    """
    Verify proposals in small batches so long stories do not produce one
    oversized JSON response that gets truncated before it can be parsed.
    """
    if not proposals:
        return ([], []) if with_trace else []

    if batch_size <= 0:
        batch_size = len(proposals)

    confirmed: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    debug_manifest: List[Dict[str, Any]] = []

    for start in range(0, len(proposals), batch_size):
        batch = proposals[start : start + batch_size]
        batch_no = start // batch_size + 1
        batch_debug_path: Optional[Path] = None
        if debug_path is not None:
            batch_debug_path = debug_path.with_name(
                f"{debug_path.stem}.batch_{batch_no:03d}{debug_path.suffix}"
            )

        batch_confirmed, batch_trace = _verify_against_full_text_batch(
            llm=llm,
            proposals=batch,
            scene_texts=scene_texts,
            debug_path=batch_debug_path,
            with_trace=True,
            index_offset=start,
        )
        confirmed.extend(batch_confirmed)
        trace.extend(batch_trace)
        if batch_debug_path is not None:
            debug_manifest.append(
                {
                    "batch": batch_no,
                    "proposal_start": start + 1,
                    "proposal_end": start + len(batch),
                    "raw_output_path": str(batch_debug_path),
                }
            )

    if debug_path is not None:
        try:
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(
                json.dumps(
                    {
                        "batched": True,
                        "batch_size": batch_size,
                        "total_proposals": len(proposals),
                        "batches": debug_manifest,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not write verification debug manifest: %s", exc)

    logger.info(
        "Batched verification: kept %d of %d across %d batch(es)",
        len(confirmed),
        len(proposals),
        len(range(0, len(proposals), batch_size)),
    )
    return (confirmed, trace) if with_trace else confirmed


def _verify_against_full_text_batch(
    llm: BaseLLM,
    proposals: List[Dict[str, Any]],
    scene_texts: Dict[int, str],
    debug_path: Optional[Path] = None,
    with_trace: bool = False,
    index_offset: int = 0,
):
    """
    Returns confirmed proposals. If with_trace=True, also returns a per-
    proposal decision log: each entry is
    {raw_index, decision, attribute, prior_quote, later_quote, reason,
     prior_grounded, later_grounded}.
    """
    if not proposals:
        return ([], []) if with_trace else []

    blocks: List[str] = []
    for sid in sorted(scene_texts):
        blocks.append(f"=== SCENE {sid} ===\n{scene_texts[sid]}")
    screenplay = "\n\n".join(blocks)

    candidates_block = "\n".join(
        f"{i+1}. X={p['x']} Y={p['y']} | FACT: {p['fact']} | EVENT: {p['event']}"
        for i, p in enumerate(proposals)
    )

    prompt = f"""SCREENPLAY:
{screenplay}

CANDIDATE HALLUCINATIONS (1-indexed):
{candidates_block}

For EACH candidate in the list, return one entry with this structure:
  {{
    "index": <1-based candidate number>,
    "decision": "confirm" | "reject",
    "attribute": "<the specific predicate that changed, e.g. color, side,
                  year, count, kinship, role, has_children, locked_state>",
    "prior_quote": "<a short quote or close paraphrase from scene X that
                    establishes the fact>",
    "later_quote": "<a short quote or close paraphrase from scene Y that
                    contradicts it or presupposes unsupported prior state>",
    "reason": "<one short sentence>"
  }}

You MUST return one entry per candidate (do not silently skip any). Confirm
when both quotes can be recognized in the screenplay and they contradict on
the named attribute.

Confirmation rules (strict — when in doubt, REJECT):
  - prior_quote must be a sentence actually in scene X (you may copy verbatim
    or paraphrase from the screenplay text — but it must be recognizable).
  - later_quote must be a sentence actually in scene Y.
  - prior_quote and later_quote must be in DIRECT CONTRADICTION on the named
    attribute, OR later_quote must presuppose a state that prior_quote (and
    no scene 1..Y) ever established.
  - attribute must be a SPECIFIC category. Reject anything you can only
    justify with vague "different from" language.

Apply the strict 4-condition test for each candidate. Confirm only if ALL
four hold; otherwise reject. Use the "reason" field to record exactly which
condition failed (e.g. "fails B: attributes differ — fact is about color,
event is about year"; "fails D: scene 7 explicitly says 'I moved opening
night'").

Decision consistency rule:
  - If your reason says "all four conditions hold", "confirmable
    contradiction", "direct contradiction", or otherwise states that the
    candidate satisfies the confirmation test, the decision MUST be "confirm".
  - If the decision is "reject", the reason MUST name which condition failed.

Default to CONFIRM when:
  - Both quotes refer to the same named entity (or an alias).
  - Both quotes describe the same specific attribute.
  - The two values clearly differ (not a rephrasing).
  - The later scene does NOT explicitly explain the change.

Default to REJECT only when one specific condition is clearly violated and
you can cite the textual evidence for it.

Do NOT reject just because:
  - "This is normal narrative progression" — characters legitimately
    changing values IS a contradiction unless condition D explicitly fails.
  - "The wording is awkward" — confirm if the underlying claim is real.
  - "I think this might be the same entity" — only invoke entity-alias
    rejection if you can point to evidence that both surface names refer
    to one entity.
  - You suspect two conflicting facts are from different continuities. Treat
    the screenplay as one continuity unless the text explicitly says otherwise.

Output strict JSON only, no prose, no markdown:
{{
  "results": [<one entry per candidate, IN ORDER>]
}}
"""

    try:
        raw = llm.complete(
            prompt=prompt,
            system=_VERIFY_SYSTEM,
            temperature=0.0,
            max_tokens=6144,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Verification call failed: %s — keeping all proposals", exc)
        return (proposals, []) if with_trace else proposals

    if debug_path is not None:
        try:
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(raw or "", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not write verification debug file: %s", exc)

    parsed = parse_llm_json(raw, schema_hint="object")
    if not isinstance(parsed, dict):
        logger.warning("Verification output unparseable — keeping all proposals")
        return (proposals, []) if with_trace else proposals

    results = parsed.get("results")
    if not isinstance(results, list):
        indices = parsed.get("confirmed", [])
        if isinstance(indices, list):
            keep_set = set()
            for v in indices:
                try:
                    keep_set.add(int(v))
                except (TypeError, ValueError):
                    continue
            kept = [p for i, p in enumerate(proposals, start=1) if i in keep_set]
            return (kept, []) if with_trace else kept
        return (proposals, []) if with_trace else proposals

    decisions: Dict[int, Dict[str, Any]] = {}
    for entry in results:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("index"))
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= len(proposals):
            decisions[idx] = entry

    confirmed: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    n_keep_default = 0
    n_explicit_reject = 0
    n_explicit_confirm = 0

    for i, prop in enumerate(proposals, start=1):
        entry = decisions.get(i)
        rec: Dict[str, Any] = {
            "proposal_index": index_offset + i,
            "proposal_text": prop.get("text", ""),
            "scene_x": int(prop.get("x", 0)),
            "scene_y": int(prop.get("y", 0)),
        }

        if entry is None:
            confirmed.append(prop)
            n_keep_default += 1
            rec.update(
                {
                    "final_decision": "kept",
                    "model_decision": "missing",
                    "reason": "no verifier entry for this index — default-keep",
                }
            )
            trace.append(rec)
            continue

        decision = str(entry.get("decision", "")).lower()
        attribute = str(entry.get("attribute", "")).strip()
        prior_quote = str(entry.get("prior_quote", "")).strip()
        later_quote = str(entry.get("later_quote", "")).strip()
        model_reason = str(entry.get("reason", "")).strip()
        rec.update(
            {
                "model_decision": decision,
                "model_reason": model_reason,
                "attribute": attribute,
                "prior_quote": prior_quote,
                "later_quote": later_quote,
            }
        )

        if decision != "confirm":
            n_explicit_reject += 1
            rec["final_decision"] = "rejected"
            rec["reason"] = f"model rejected: {model_reason or '(no reason)'}"
            trace.append(rec)
            continue

        if not attribute or not prior_quote or not later_quote:
            n_explicit_reject += 1
            rec["final_decision"] = "rejected"
            rec["reason"] = "confirm missing required field (attribute / prior_quote / later_quote)"
            trace.append(rec)
            continue

        x_text = scene_texts.get(int(prop.get("x", 0)), "") or ""
        y_text = scene_texts.get(int(prop.get("y", 0)), "") or ""
        prior_g = _quote_grounded(prior_quote, x_text)
        later_g = _quote_grounded(later_quote, y_text)
        rec["prior_grounded"] = prior_g
        rec["later_grounded"] = later_g

        confirmed.append(prop)
        n_explicit_confirm += 1
        rec["final_decision"] = "confirmed"
        rec["reason"] = "model confirmed"
        trace.append(rec)

    logger.info(
        "Verification batch %d-%d: kept %d of %d (confirm=%d default-keep=%d reject=%d)",
        index_offset + 1,
        index_offset + len(proposals),
        len(confirmed),
        len(proposals),
        n_explicit_confirm,
        n_keep_default,
        n_explicit_reject,
    )
    return (confirmed, trace) if with_trace else confirmed


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalize_tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _quote_grounded(
    quote: str,
    scene_text: str,
    min_overlap: float = 0.30,
    min_token_match: int = 3,
) -> bool:
    """
    True if `quote` is recognizable inside `scene_text`. Verbatim substring
    match counts; otherwise require a sliding-window token Jaccard above the
    threshold. Robust to minor paraphrasing the LLM does when "quoting".
    """
    if not quote or not scene_text:
        return False
    if quote.lower() in scene_text.lower():
        return True

    qt = _normalize_tokens(quote)
    if len(qt) < min_token_match:
        return False
    qset = set(qt)
    st_tokens = _normalize_tokens(scene_text)
    if not st_tokens:
        return False

    window = max(len(qt), 4)
    best = 0.0
    for i in range(0, max(1, len(st_tokens) - window + 1)):
        chunk = set(st_tokens[i : i + window])
        if not chunk:
            continue
        inter = len(qset & chunk)
        union = len(qset | chunk)
        if union == 0:
            continue
        j = inter / union
        if j > best:
            best = j
            if best >= min_overlap:
                return True
    return best >= min_overlap


# ---------------------------------------------------------------- summarization


def summarize_prior_graphs(
    prior: List[SceneGraph],
    scene_attributes: Optional[Dict[int, Dict[str, Dict[str, Any]]]] = None,
    up_to_scene_exclusive: Optional[int] = None,
) -> str:
    """
    Render prior scene graphs as a compact, scene-tagged text block. Optionally
    splices in durable-attribute lines produced by the attribute extractor —
    these recover attribute-level facts (color, side, year, kinship, ...) that
    the graph extractor frequently drops into free-text node descriptions or
    fails to capture at all.
    """
    lines: List[str] = []
    for sg in prior:
        s = sg.scene_number
        node_by_id = {n.get("id"): n for n in sg.nodes if n.get("id")}

        for node in sg.nodes:
            ntype = node.get("type", "")
            name = node.get("name", "")
            if not name:
                continue
            desc = (node.get("description") or "").strip()
            text = _node_text(node)
            if ntype in {"Location", "TimePoint"}:
                continue
            if ntype != "Event" and not _is_durable_fact(text):
                continue
            if ntype == "Event":
                tag = "event"
            else:
                tag = ntype.lower() or "node"
            if desc:
                lines.append(f"S{s}: [{tag}] {name} — {desc}")
            else:
                lines.append(f"S{s}: [{tag}] {name}")

        for edge in sg.edges:
            relation = edge.get("relation", "")
            if relation in TRANSIENT_RELATIONS:
                continue
            src = node_by_id.get(edge.get("source"), {})
            tgt = node_by_id.get(edge.get("target"), {})
            src_name = src.get("name") or edge.get("source")
            tgt_name = tgt.get("name") or edge.get("target")
            lines.append(f"S{s}: ({src_name}) -[{relation}]-> ({tgt_name})")

    # Per-scene attribute lines (chronological).
    if scene_attributes:
        cutoff = up_to_scene_exclusive
        for sn in sorted(scene_attributes):
            if cutoff is not None and sn >= cutoff:
                break
            profile = scene_attributes.get(sn, {}) or {}
            for entity, attrs in profile.items():
                if not isinstance(attrs, dict):
                    continue
                for attr_key, attr_val in attrs.items():
                    lines.append(f"S{sn}: [attr] {entity}.{attr_key} = {attr_val}")

    # Entity-grouped history view. Collapses every prior assertion about an
    # entity into one block, scene-tagged. Makes scene-to-scene contradictions
    # on the same (entity, attribute) trivial to spot at proposal time.
    if scene_attributes:
        cutoff = up_to_scene_exclusive
        grouped: Dict[str, List[Tuple[int, str, Any]]] = {}
        for sn in sorted(scene_attributes):
            if cutoff is not None and sn >= cutoff:
                break
            profile = scene_attributes.get(sn, {}) or {}
            for entity, attrs in profile.items():
                if not isinstance(attrs, dict):
                    continue
                for attr_key, attr_val in attrs.items():
                    grouped.setdefault(entity, []).append((sn, attr_key, attr_val))

        if grouped:
            lines.append("")
            lines.append(
                "ENTITY HISTORY (all durable predicates asserted about each "
                "named entity in earlier scenes; check for value mismatches):"
            )
            for entity in sorted(grouped):
                entries = grouped[entity]
                # Group attributes per scene for compact rendering.
                by_scene: Dict[int, List[Tuple[str, Any]]] = {}
                for sn, k, v in entries:
                    by_scene.setdefault(sn, []).append((k, v))
                pieces = []
                for sn in sorted(by_scene):
                    kv = ", ".join(f"{k}={v}" for k, v in by_scene[sn])
                    pieces.append(f"S{sn}: {kv}")
                lines.append(f"  {entity}: " + " | ".join(pieces))

    return "\n".join(lines)


def extract_candidate_facts(
    scene_graph: SceneGraph,
    scene_attribute_profile: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Pull the most temporally-loaded facts from a single scene graph: events
    plus edges that imply durable state. Also surface non-Event nodes whose
    free-text descriptions carry durable attributes, and — when provided — the
    entity-anchored durable predicates from the attribute extractor for the
    same scene. The attribute lines are the highest-signal continuity facts
    because they're already typed (entity, predicate, value) and align
    directly with the prior-summary attribute lines.
    """
    facts: List[Dict[str, Any]] = []
    seen_texts: set = set()
    node_by_id = {n.get("id"): n for n in scene_graph.nodes if n.get("id")}

    # 1. Entity-anchored attribute predicates (most useful for continuity).
    if scene_attribute_profile:
        for entity, attrs in scene_attribute_profile.items():
            if not isinstance(attrs, dict):
                continue
            for attr_key, attr_val in attrs.items():
                text = f"{entity}.{attr_key} = {attr_val}"
                if text in seen_texts:
                    continue
                seen_texts.add(text)
                facts.append({"kind": "attribute", "text": text, "name": entity})

    # 2. Event nodes and durable non-Event nodes from the graph.
    for node in scene_graph.nodes:
        ntype = node.get("type", "")
        if ntype in {"Location", "TimePoint"}:
            continue
        text = _node_text(node)
        if not text or not _is_durable_fact(text):
            continue
        kind = "event" if ntype == "Event" else ntype.lower() or "node"
        if text in seen_texts:
            continue
        seen_texts.add(text)
        facts.append({"kind": kind, "text": text, "name": node.get("name", "")})

    # 3. Non-transient typed relations.
    for edge in scene_graph.edges:
        relation = edge.get("relation", "")
        if relation in TRANSIENT_RELATIONS:
            continue
        src = node_by_id.get(edge.get("source"), {})
        tgt = node_by_id.get(edge.get("target"), {})
        src_name = src.get("name") or edge.get("source")
        tgt_name = tgt.get("name") or edge.get("target")
        text = f"({src_name}) -[{relation}]-> ({tgt_name})"
        if not _is_durable_fact(text):
            continue
        if text in seen_texts:
            continue
        seen_texts.add(text)
        facts.append({"kind": "relation", "text": text, "name": text})

    return facts


def _node_text(node: Dict[str, Any]) -> str:
    name = node.get("name", "")
    desc = (node.get("description") or "").strip()
    return f"{name} — {desc}" if desc else name


def _is_durable_fact(text: str) -> bool:
    """Return True for facts likely to encode continuity-relevant state."""
    if not text:
        return False
    lowered = text.lower()
    if re.search(r"\b\d+\b", lowered):
        return True
    tokens = set(re.findall(r"[a-z0-9]+", lowered))
    return bool(tokens & DURABLE_FACT_TERMS)


# ---------------------------------------------------------------- LLM judge


SYSTEM_PROMPT = (
    "You are a temporal continuity checker for screenplay scenes. This is the "
    "PROPOSAL stage of a two-stage pipeline — a separate VERIFICATION stage "
    "with full screenplay text will filter your output, so AIM FOR HIGH RECALL. "
    "You are given (a) facts established in earlier scenes and (b) candidate "
    "facts/events from the CURRENT scene. Flag every plausible temporal "
    "hallucination: contradictions on durable attributes (color, number, "
    "date/time, body side, injury origin, object identity, material, owner, "
    "family status, death status, lock/key state, cleared/arrested status, "
    "named roles, named places, building dates, scheduled days), and "
    "events that presuppose unsupported prior state. "
    "When uncertain, FLAG and let verification handle it. The only things you "
    "must NOT flag are normal scene-to-scene movement and ordinary continuation "
    "of established actions."
)


def _build_prompt(
    current_scene_number: int,
    current_scene_text: str,
    current_facts: List[Dict[str, Any]],
    prior_summary: str,
) -> str:
    text_block = (
        f"CURRENT SCENE {current_scene_number} TEXT:\n{current_scene_text}\n\n"
        if current_scene_text
        else ""
    )

    candidates_block = "\n".join(
        f"- ({c['kind']}) {c['text']}" for c in current_facts
    )

    fmt_example = HALLUCINATION_FORMAT.format(
        x="X", fact="[FACT]", y="Y", event="[EVENT]"
    )

    return f"""PRIOR SCENE GRAPHS (facts established before scene {current_scene_number}):
{prior_summary or "(none)"}

{text_block}CANDIDATE FACTS FROM SCENE {current_scene_number}:
{candidates_block}

TASK:
For each candidate that is a temporal hallucination, return one JSON object
with:
  - "x": the prior scene number whose established fact is contradicted or
         that should have established the missing setup (integer).
  - "y": the current scene number ({current_scene_number}).
  - "fact": a SHORT clause (≤15 words) describing the earlier-established
            fact.
  - "event": a SHORT clause (≤15 words) describing what happens in scene
             {current_scene_number} that conflicts or is unsupported.

CRITICAL FORMATTING RULES for "fact" and "event":
  - Each must be ≤15 words and ONE clause.
  - Do NOT include the words "was established" inside "fact".
  - Do NOT include the word "happens" inside "event".
  - Do NOT include em-dashes or " — " separators.
  - Do NOT quote or paste the candidate-fact text verbatim. RESTATE concisely.
  - Use a tight noun-phrase or simple-clause restatement of the underlying
    attribute, not a paragraph copied from the source.
  - Emit each distinct contradiction ONCE. Do NOT produce multiple entries
    that describe the same underlying contradiction with different wording.

PROPOSAL STAGE — aim for HIGH RECALL. A separate verification stage with
the full screenplay text will filter your output. When in doubt, FLAG.

Flag any candidate where, comparing earlier facts to the current scene's
candidate, you can identify a contradiction in ONE of these abstract
categories. (Examples are deliberately schematic — they do not refer to
any specific story.)

  Attribute categories that count as durable contradictions:
   - color, material, or visual identity of a named object/entity
   - body-side (left vs right) of a feature, injury, or possession
   - quantity, count, money amount, age, duration, or measurement
   - calendar fact (year, date, day of week, scheduled day)
   - named role, title, occupation, or rank of a person
   - identity / type / species / make / model of a named entity
   - kinship, marital, parental, or life/death status
   - locked vs unlocked, intact vs broken, present vs absent
   - cause or origin of a past event
   - name of a person, place, or thing introduced earlier

  Unsupported-prior-state pattern that also counts:
   - the current scene asserts/relies on a prior state X
   - no earlier scene established X
   - the current scene itself does not establish X on-screen
   → FLAG.

Do NOT flag:
   - Normal movement between locations or rooms.
   - Ordinary continuation of an already-established action.
   - A character doing a new but consistent thing in a later scene.
   - Surface re-wordings of the same underlying fact.

If you cannot point to a specific changed attribute (from the list above),
do not flag.

OUTPUT FORMAT — strictly valid JSON list, no prose:
[
  {{"x": <int>, "y": {current_scene_number}, "fact": "...", "event": "..."}}
]

Each rendered hallucination must read exactly:
"{fmt_example}"
"""


def _ask_llm_for_hallucinations(
    llm: BaseLLM,
    current_scene_number: int,
    current_scene_text: str,
    current_facts: List[Dict[str, Any]],
    prior_summary: str,
) -> List[Dict[str, Any]]:
    prompt = _build_prompt(
        current_scene_number=current_scene_number,
        current_scene_text=current_scene_text,
        current_facts=current_facts,
        prior_summary=prior_summary,
    )

    try:
        raw = llm.complete(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=4096,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM call failed in graph_verifier: %s", exc)
        return []

    parsed = parse_llm_json(raw, schema_hint="list")
    if parsed is None:
        return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

    cleaned: List[Dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if "fact" not in item or "event" not in item:
            continue
        cleaned.append(item)
    return cleaned


def _render_finding(item: Dict[str, Any], default_y: int) -> str:
    fact = _sanitize_fact(item.get("fact", ""))
    event = _sanitize_event(item.get("event", ""))
    if not fact or not event:
        return ""

    try:
        x = int(item.get("x"))
    except (TypeError, ValueError):
        return ""
    try:
        y = int(item.get("y", default_y))
    except (TypeError, ValueError):
        y = default_y

    if x <= 0 or y <= 0 or x >= y:
        # Hallucinations must reference a strictly prior scene.
        return ""

    return HALLUCINATION_FORMAT.format(x=x, fact=fact, y=y, event=event)


_LEAKAGE_PATTERNS_FACT = [
    re.compile(r"\s+was\s+established\b.*$", re.IGNORECASE),
    re.compile(r"\s+is\s+established\b.*$", re.IGNORECASE),
]
_LEAKAGE_PATTERNS_EVENT = [
    re.compile(r"\s+happens\.?\s*$", re.IGNORECASE),
]
_DASH_PATTERN = re.compile(r"\s+[—–]\s+|\s+--?\s+")


def _sanitize_fact(text: Any) -> str:
    s = str(text or "").strip().strip(".").strip()
    for pat in _LEAKAGE_PATTERNS_FACT:
        s = pat.sub("", s)
    # Collapse the most common bloat: " — explanation goes here"
    s = _DASH_PATTERN.split(s, maxsplit=1)[0].strip()
    return s.strip().strip(",").strip()


def _sanitize_event(text: Any) -> str:
    s = str(text or "").strip().strip(".").strip()
    for pat in _LEAKAGE_PATTERNS_EVENT:
        s = pat.sub("", s)
    s = _DASH_PATTERN.split(s, maxsplit=1)[0].strip()
    return s.strip().strip(",").strip()


# ---------------------------------------------------------------- format parsing


_HALLUCINATION_REGEX_STRICT = re.compile(
    r"^\s*In\s+scene\s+(\d+)\s*,\s*(.+?)\s+was\s+established\s*,\s*yet\s+in\s+scene\s+(\d+)\s*,\s*(.+?)\s+happens\.?\s*$",
    re.IGNORECASE | re.DOTALL,
)

# Lenient form — accepts strings that follow the "In scene X, ..., yet in
# scene Y, ..." skeleton but omit the literal "was established" / "happens"
# suffixes. LLMs frequently drop those words even when explicitly told not to.
_HALLUCINATION_REGEX_LENIENT = re.compile(
    r"^\s*In\s+scene\s+(\d+)\s*,\s*(.+?)\s*,\s*yet\s+in\s+scene\s+(\d+)\s*,\s*(.+?)\.?\s*$",
    re.IGNORECASE | re.DOTALL,
)

_OPTIONAL_FACT_SUFFIX = re.compile(
    r"\s+(?:was|is|were|are)\s+(?:established|set\s+up|introduced|shown|stated)\b.*$",
    re.IGNORECASE,
)
_OPTIONAL_EVENT_SUFFIX = re.compile(r"\s+happens\.?\s*$", re.IGNORECASE)


def parse_hallucination_string(text: str) -> Optional[Tuple[int, str, int, str]]:
    """
    Parse a hallucination string into (scene_x, fact, scene_y, event).

    Tries the canonical "was established / happens" template first, then
    falls back to the lenient form. Returns None if neither matches.
    """
    if not isinstance(text, str):
        return None
    stripped = text.strip()

    m = _HALLUCINATION_REGEX_STRICT.match(stripped)
    if m:
        return (
            int(m.group(1)),
            m.group(2).strip(),
            int(m.group(3)),
            m.group(4).strip(),
        )

    m = _HALLUCINATION_REGEX_LENIENT.match(stripped)
    if not m:
        return None
    fact = _OPTIONAL_FACT_SUFFIX.sub("", m.group(2)).strip().strip(",").strip()
    event = _OPTIONAL_EVENT_SUFFIX.sub("", m.group(4)).strip().strip(".").strip()
    if not fact or not event:
        return None
    return int(m.group(1)), fact, int(m.group(3)), event
