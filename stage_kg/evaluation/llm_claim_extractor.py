"""Hallucination-focused atomic claim extraction built on the STAGE extractor stack."""

import logging
import re
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ..extraction.event_extractor import MAX_CHUNK_CHARS
from ..ingest.loader import SceneRecord
from ..llm.base import BaseLLM
from ..llm.gemini_client import GeminiLLM
from ..prompts import claim_extraction as cp
from ..schema import RELATION_TYPES
from ..utils.cache import Cache
from ..utils.json_repair import parse_llm_json, validate_claim_list
from ..utils.logging_utils import PromptLogger


logger = logging.getLogger(__name__)

ALLOWED_PREDICATES = set(RELATION_TYPES)
CLAIM_MAX_CHUNK_CHARS = min(900, MAX_CHUNK_CHARS)
REPORTING_VERB_RE = re.compile(
    r"\b(says?|said|asks?|asked|states?|stated|tells?|told|exclaims?|shouts?|yells?|compares?|calls?|references?)\b"
)
GENERIC_SUBJECTS = {
    "i",
    "me",
    "we",
    "she",
    "he",
    "they",
    "you",
    "someone",
    "somebody",
    "it",
    "speaker",
    "the speaker",
    "person addressed",
    "the person addressed",
    "guys",
    "cops",
    "officers",
    "all eyes",
    "our fighters",
    "their plan",
    "the town",
    "scene",
    "the scene",
    "this movie",
    "the movie",
    "filming",
    "the filming",
    "speaker",
    "the speaker",
    "character",
    "a character",
    "the person addressed",
    "others",
    "some of the sisters",
    "guys",
}
GENERIC_SUBJECT_PREFIXES = (
    "a character",
    "character ",
    "the character ",
    "scene ",
    "the scene ",
    "speaker ",
    "the speaker ",
    "person addressed",
    "the person addressed",
)
GENERIC_OBJECT_TOKENS = {"something", "someone", "somebody", "thing"}
SOUND_CLAIM_HINTS = ("siren", "barking", "audible sound", "bwoop")
META_REFERENCE_HINTS = ("green draft", "8 28 09", "48a")
INTERPRETIVE_STATE_HINTS = (
    "better days",
    "need to ",
    "about to ",
    "looks scared",
    "agitated",
    "depressed",
    "tries not to cry",
    "being awake",
    " is wild",
)
WEAK_ACTION_HINTS = (
    "double take",
    "clawing",
    "studying ",
    "leaping backwards",
    "handling something",
    "admits",
    "appraises",
    "bangs",
    "dons",
    "gestures to",
    "hands",
    "invites",
    "lays",
    "orders",
    "puts",
    "rubs",
    "shrugs",
    "swallows",
    "laughs",
    "smiles",
    "rises",
    "stands",
    "sits",
    "watches",
    "looks",
    "nods",
    "repeats the word",
    "recites the maxim",
    "quotes",
)
SCENE_META_HINTS = (
    "scene occurs",
    "scene is located",
    "scene takes place",
    "this movie occurs",
    "this movie is present",
    "the filming references",
)
LOW_VALUE_ATTRIBUTE_RE = re.compile(
    r"\b(years old|year old|lbs|pounds|audible sound|redhead|police uniform|uniform)\b"
)
MICRO_ACTION_RE = re.compile(
    r"\b("
    r"instruct(?:ing)?|asks?|asked|tells?|told|states?|stated|compares?|compared|"
    r"threatens?|threatened|yells?|yelled|shouts?|shouted|sitting|sits?\b|staring|stares?|"
    r"talking|talks?|jogging|lands?|trails?|hugs?|grabs?|tags?|swings?|nails?|pulls?|"
    r"tries? to pull|fights? back|struggl(?:es?|ing)|throws? herself|shoves?|cups?|"
    r"walks? past|walks? up|walks? back|walks? toward|walks? away|walks? quickly|pulls? up|"
    r"shouting|uppercut|hook|turning and looking|tries? to keep up|leaves?(?:\b| .*)|"
    r"getting out|supervis(?:e|ing)|holds? the mits|holds? his arms up|running down the street|"
    r"runs? in|runs? across|runs? to the edge|continues to walk|takes off|looks? out|"
    r"looks? at|looks? to|rolls? his eyes|closes? his eyes|made it right|roofs? with|"
    r"fall right back into it|stands? rubbing|standing around|watch(?:es|ing)|makes? out|"
    r"holding the white dog|holding|chattering|gives? .* look|shakes? (?:her|his) head|"
    r"jumps? up|starts? for the door|rushes? toward|kicking|crying out for help|"
    r"drinking coffee|smokes?|smoking|pops? out|waits?\b|nods?\b|come flying at|"
    r"screeches? up outside|pile out of the car|rubbing (?:her|his|their) shoulders|"
    r"limps? toward|slows? down|hanging out front|leans? back|takes? cash|mumbles?|"
    r"cruises?|toots? the horn|arresting|cannot find his keys|starts? the car|"
    r"exit the theater|standing in the doorway|says goodbye|calls? dicky s name|"
    r"calls? .* name|sees?\b|starts? to run|runs? into the scene|win a fight"
    r")\b"
)
STRUCTURAL_LOCATION_RE = re.compile(
    r"\b(lower class neighborhood|alice george ward s house|back of the kitchen|"
    r"2nd floor|second floor|foxwoods resort|outside the gym|open window|limousine|"
    r"roof of house)\b"
)
PROP_ATTRIBUTE_RE = re.compile(r"\b(lucky strike|three newspapers|wild)\b")
PROCESS_STATE_RE = re.compile(
    r"\b(thrashing|handcuffed|escorted by a guard|instructed to|absent for approximately one week)\b"
)
INTERPRETIVE_EXPERIENCE_RE = re.compile(r"\b(sadness|unsure)\b")
SCENE_PRESENCE_RE = re.compile(r"\b(convicts|corner jacket)\b")
EXPLANATORY_CAUSE_RE = re.compile(r"\b(believes?|states?)\b")
ORDERING_SCAFFOLD_RE = re.compile(r"\b(precedes?|before the fight)\b")
TIME_SCAFFOLD_RE = re.compile(r"\b(months later)\b")
SCENE_DESCRIPTION_RE = re.compile(r"\b(crowded with convicts)\b")
WEAK_AFFILIATION_RE = re.compile(r"\b(affiliated with crack)\b")
VAGUE_REFERENCE_RE = re.compile(
    r"\b(references?|mentions?|addresses|refers to|quotes?)\b"
)
LOW_VALUE_DIALOGUE_RE = re.compile(
    r"\b(says?|states?|asks?|tells?)\b.*\b("
    r"this|that|something|anything|proposal|deal|challenge|happens|"
    r"beautiful|delusional|legend|possible|sure|ready|okay|ok|"
    r"busy|thrilled|softie|sitting duck|no go|love|trust|fear|shame"
    r")\b"
)
LOW_VALUE_DESCRIPTION_RE = re.compile(
    r"\b(is|are|was|were|appears?|seems?)\b.*\b("
    r"beautiful|sexy|successful|appealing|boyish|bespectacled|"
    r"crowded|prepared|ready|content|attractive|charismatic|depressing|"
    r"modestly dressed|unoffending|dispassionate|sufficiently|softie|"
    r"called|fourteen|twelve|participant|sure|curious|hungry"
    r")\b"
)
LOW_VALUE_EVENT_RE = re.compile(
    r"\b("
    r"look(?:s|ed)?|smiles?|laughs?|nods?|shrugs?|rises?|stands?|sits?|"
    r"rubs?|swallows?|orders?|invites?|admits?|appraises?|gestures?|"
    r"hands?|lays?|puts?|dons?|bangs?|watches?|pauses?"
    r")\b"
)
NON_ATOMIC_MICRO_ACTION_RE = re.compile(
    r"\b("
    r"talking on the phone|staring|stares|studying|double take|gives? .* look|"
    r"jogging a little curve|pulls? fellow cops away|fights? back|struggl(?:es?|ing)|"
    r"grabs? a cop|break hands|throws? herself|shoves? cops|walks? past|"
    r"supervis(?:e|es|ing)|pulls? up in a roofing truck|watches? the fight|"
    r"short right uppercut|\bhook\b|jumps? up|starts? for the door|takes? off|"
    r"holds? the mits|about to throw|trails? alice|hugs? alice'?s shoulders|"
    r"lou gold and espn made it right|compares? him to one of the mtv girls|"
    r"\bbooboo leaves\b|exit the theater|states? her support|\bkaren shouts\b"
    r")\b"
)
GENERIC_OBJECTS = {
    "that guy",
    "the guy",
    "someone",
    "somebody",
    "something",
    "the scene",
    "this movie",
}


@dataclass
class Claim:
    """Atomic claim extracted for hallucination verification."""

    subject: str
    predicate: str
    object: str
    claim_text: str
    scene_id: str
    sentence_idx: int
    chunk_id: str = ""
    evidence: Optional[List[str]] = None
    confidence: float = 0.8
    claim_id: str = ""

    def to_dict(self) -> Dict:
        return {
            "id": self.claim_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "claim_text": self.claim_text,
            "scene_id": self.scene_id,
            "sentence_idx": self.sentence_idx,
            "chunk_id": self.chunk_id,
            "evidence": self.evidence or [],
            "confidence": self.confidence,
        }


@dataclass
class ExtractionAbstention:
    """Claim candidate held out from scoring because it is not graph-faithfulness-worthy."""

    text: str
    scene_id: str
    sentence_idx: int
    reason: str
    speaker: str = ""
    context_subject: str = ""


@dataclass
class LowConfidenceClaim:
    """Borderline claim candidate held out from scoring."""

    subject: str
    predicate: str
    object: str
    claim_text: str
    scene_id: str
    sentence_idx: int
    score: float
    reasons: List[str]


def extract_claims_for_scene(
    scene: SceneRecord,
    llm: BaseLLM,
    movie_id: str = "hallucination",
    movie_title: str = "",
    cache: Optional[Cache] = None,
    prompt_logger: Optional[PromptLogger] = None,
    max_chunk_chars: int = CLAIM_MAX_CHUNK_CHARS,
) -> List[Dict]:
    """Extract atomic claims for one scene record using the STAGE chunk/reflection flow."""
    return _extract_claim_dicts(
        scene_id=scene.scene_id,
        scene_title=scene.title,
        scene_summary=scene.summary,
        scene_text=scene.content,
        chunks=scene.chunks,
        llm=llm,
        movie_id=movie_id,
        movie_title=movie_title,
        max_chunk_chars=max_chunk_chars,
        cache=cache,
        prompt_logger=prompt_logger,
    )


def extract_claims_from_text(
    text: str,
    scene_id: str,
    llm: BaseLLM,
    scene_title: str = "",
    scene_summary: str = "",
    movie_id: str = "hallucination",
    movie_title: str = "",
    cache: Optional[Cache] = None,
    prompt_logger: Optional[PromptLogger] = None,
    max_chunk_chars: int = CLAIM_MAX_CHUNK_CHARS,
) -> List[Dict]:
    """Extract atomic claims from arbitrary generated text for hallucination evaluation."""
    chunks = _build_chunks(text, scene_id, max_chunk_chars)
    return _extract_claim_dicts(
        scene_id=scene_id,
        scene_title=scene_title,
        scene_summary=scene_summary,
        scene_text=text,
        chunks=chunks,
        llm=llm,
        movie_id=movie_id,
        movie_title=movie_title,
        max_chunk_chars=max_chunk_chars,
        cache=cache,
        prompt_logger=prompt_logger,
    )


class ClaimExtractor:
    """Gemini-compatible wrapper around the STAGE-style atomic claim extraction flow."""

    def __init__(
        self,
        llm: BaseLLM,
        prompt_logger: Optional[PromptLogger] = None,
        max_chunk_chars: int = CLAIM_MAX_CHUNK_CHARS,
    ):
        self.llm = llm
        self.prompt_logger = prompt_logger
        self.max_chunk_chars = max_chunk_chars

    @classmethod
    def for_gemini(
        cls,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        prompt_logger: Optional[PromptLogger] = None,
        max_chunk_chars: int = CLAIM_MAX_CHUNK_CHARS,
    ) -> "ClaimExtractor":
        llm = GeminiLLM(model=model, api_key=api_key)
        return cls(
            llm=llm,
            prompt_logger=prompt_logger,
            max_chunk_chars=max_chunk_chars,
        )

    def extract_claim_dicts(
        self,
        text: str,
        scene_id: str,
        scene_title: str = "",
        scene_summary: str = "",
        movie_id: str = "hallucination",
        movie_title: str = "",
        cache: Optional[Cache] = None,
    ) -> List[Dict]:
        return extract_claims_from_text(
            text=text,
            scene_id=scene_id,
            llm=self.llm,
            scene_title=scene_title,
            scene_summary=scene_summary,
            movie_id=movie_id,
            movie_title=movie_title,
            cache=cache,
            prompt_logger=self.prompt_logger,
            max_chunk_chars=self.max_chunk_chars,
        )

    def extract_claims(
        self,
        text: str,
        scene_id: str,
        scene_title: str = "",
        scene_summary: str = "",
        movie_id: str = "hallucination",
        movie_title: str = "",
        cache: Optional[Cache] = None,
    ) -> List[Claim]:
        return [
            _claim_from_dict(claim_dict)
            for claim_dict in self.extract_claim_dicts(
                text=text,
                scene_id=scene_id,
                scene_title=scene_title,
                scene_summary=scene_summary,
                movie_id=movie_id,
                movie_title=movie_title,
                cache=cache,
            )
        ]


def _extract_claim_dicts(
    scene_id: str,
    scene_title: str,
    scene_summary: str,
    scene_text: str,
    chunks: List[Dict],
    llm: BaseLLM,
    movie_id: str,
    movie_title: str,
    max_chunk_chars: int,
    cache: Optional[Cache],
    prompt_logger: Optional[PromptLogger],
) -> List[Dict]:
    sentence_map = _build_sentence_map(scene_text)
    scene_claims: List[Dict] = []
    seen: Set[Tuple[str, str, str, int]] = set()

    for chunk in chunks:
        chunk_id = chunk.get("id", f"{scene_id}_chunk_0")
        cache_key = Cache.make_key(movie_id, "claims", scene_id, chunk_id) if cache else None

        if cache_key and cache is not None:
            cached = cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit: claims for %s/%s", scene_id, chunk_id)
                scene_claims.extend(cached)
                continue

        chunk_text = str(chunk.get("content", "") or "")
        if not chunk_text.strip():
            continue
        if len(chunk_text) > max_chunk_chars:
            chunk_text = chunk_text[:max_chunk_chars]

        def extract_once():
            prompt = cp.build_prompt(
                scene_id=scene_id,
                scene_title=scene_title,
                scene_text=chunk_text,
                chunk_id=chunk_id,
                movie_title=movie_title,
                scene_summary=scene_summary,
            )
            raw = llm.complete(
                prompt,
                system=cp.SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=8192,
            )
            if prompt_logger:
                prompt_logger.log("claim_extraction", scene_id, prompt, raw, llm.model_id)
            result = parse_llm_json(raw, schema_hint="claim_list")
            if result is not None and not isinstance(result, list):
                result = [result]
            return result, prompt

        raw_claims, _ = extract_once()

        if not raw_claims:
            logger.warning("Claim extraction yielded nothing for %s/%s", scene_id, chunk_id)
            if cache_key and cache is not None:
                cache.set(cache_key, [])
            continue

        cleaned_claims = _postprocess_claims(
            raw_claims=raw_claims,
            scene_id=scene_id,
            chunk_id=chunk_id,
            movie_id=movie_id,
            sentence_map=sentence_map,
        )

        if not validate_claim_list(cleaned_claims):
            logger.warning("Claim list failed validation for %s/%s — keeping filtered claims anyway", scene_id, chunk_id)

        deduped_chunk_claims: List[Dict] = []
        for claim in cleaned_claims:
            key = (
                claim.get("subject", ""),
                claim.get("predicate", ""),
                claim.get("object", ""),
                int(claim.get("sentence_idx", 0)),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped_chunk_claims.append(claim)

        if cache_key and cache is not None:
            cache.set(cache_key, deduped_chunk_claims)
        scene_claims.extend(deduped_chunk_claims)

    return scene_claims


def _postprocess_claims(
    raw_claims: List[Dict],
    scene_id: str,
    chunk_id: str,
    movie_id: str,
    sentence_map: List[str],
) -> List[Dict]:
    cleaned: List[Dict] = []
    seen: Set[Tuple[str, str, str, int]] = set()

    for item in raw_claims:
        if not isinstance(item, dict):
            continue

        subject = _normalize_text(item.get("subject", ""))
        predicate = _normalize_predicate(item.get("predicate", ""))
        obj = _normalize_text(item.get("object", ""))
        claim_text = _clean_preserve_case(item.get("claim_text", ""))
        evidence = _normalize_evidence_list(item.get("evidence", []))
        confidence = _normalize_confidence(item.get("confidence", 0.8))

        if not subject or not predicate or not obj:
            continue
        if predicate not in ALLOWED_PREDICATES:
            continue
        if not evidence:
            continue

        sentence_idx = _resolve_sentence_idx(claim_text, evidence, sentence_map)
        key = (subject, predicate, obj, sentence_idx)
        if key in seen:
            continue
        seen.add(key)

        temp_id = str(item.get("temp_id", "") or "")
        cleaned.append(
            {
                "id": _build_claim_id(movie_id, scene_id, chunk_id, temp_id),
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "claim_text": claim_text or f"{subject} {predicate} {obj}",
                "scene_id": scene_id,
                "chunk_id": chunk_id,
                "sentence_idx": sentence_idx,
                "evidence": evidence,
                "confidence": confidence,
            }
        )

    return cleaned


def _build_chunks(text: str, scene_id: str, max_chunk_chars: int) -> List[Dict]:
    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped) <= max_chunk_chars:
        return [{"id": f"{scene_id}_chunk_0", "content": stripped}]

    chunks: List[Dict] = []
    cursor = 0
    chunk_idx = 0
    while cursor < len(stripped):
        end = min(len(stripped), cursor + max_chunk_chars)
        if end < len(stripped):
            split_at = stripped.rfind("\n", cursor, end)
            if split_at <= cursor:
                split_at = stripped.rfind(" ", cursor, end)
            if split_at > cursor:
                end = split_at
        chunk_text = stripped[cursor:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "id": f"{scene_id}_chunk_{chunk_idx}",
                    "content": chunk_text,
                }
            )
            chunk_idx += 1
        cursor = max(end + 1, cursor + 1)
    return chunks


def _build_sentence_map(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [_normalize_text(part) for part in parts if part.strip()]


def _resolve_sentence_idx(claim_text: str, evidence: List[str], sentence_map: List[str]) -> int:
    candidates = [_normalize_text(claim_text)] + [_normalize_text(item) for item in evidence]
    for idx, sentence in enumerate(sentence_map):
        if any(candidate and (candidate in sentence or sentence in candidate) for candidate in candidates):
            return idx
    return 0


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _clean_preserve_case(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_predicate(value: str) -> str:
    predicate = _normalize_text(value).replace(" ", "_")
    aliases = {
        "located_in": "located_at",
        "happens_at": "occurs_at",
        "happens_on": "occurs_on",
        "has": "possesses",
    }
    return aliases.get(predicate, predicate)


def _normalize_evidence_list(value) -> List[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    cleaned: List[str] = []
    seen = set()
    for item in items:
        text = _clean_preserve_case(item)
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def _normalize_confidence(value) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.8
    return max(0.0, min(1.0, score))


def _build_claim_id(movie_id: str, scene_id: str, chunk_id: str, temp_id: str) -> str:
    chunk_slug = re.sub(r"[^A-Za-z0-9]+", "_", str(chunk_id or "chunk")).strip("_") or "chunk"
    suffix = temp_id or uuid.uuid4().hex[:6]
    return f"cl_{movie_id[:8]}_{scene_id}_{chunk_slug}_{suffix}"


def _claim_from_dict(data: Dict) -> Claim:
    return Claim(
        subject=data.get("subject", ""),
        predicate=data.get("predicate", ""),
        object=data.get("object", ""),
        claim_text=data.get("claim_text", ""),
        scene_id=data.get("scene_id", ""),
        sentence_idx=int(data.get("sentence_idx", 0)),
        chunk_id=data.get("chunk_id", ""),
        evidence=list(data.get("evidence", []) or []),
        confidence=float(data.get("confidence", 0.8)),
        claim_id=data.get("id", ""),
    )


def assess_claim_quality(claim: Claim) -> Tuple[str, float, List[str]]:
    """
    Classify a claim as keep / low_confidence / abstain.

    We abstain only on explicit quoted dialogue-style subclaims that should not
    be scored as atomic factual claims.
    """
    reasons: List[str] = []
    claim_text = _normalize_text(claim.claim_text)
    subject = _normalize_text(claim.subject)
    obj = _normalize_text(claim.object)
    predicate = _normalize_predicate(claim.predicate)
    if predicate in {"performs", "references", "causes", "precedes"} and _has_quoted_span(claim.claim_text):
        reasons.append("quoted_dialogue")

    if not subject or subject in GENERIC_SUBJECTS or any(subject.startswith(prefix) for prefix in GENERIC_SUBJECT_PREFIXES):
        reasons.append("generic_subject")

    if obj in GENERIC_OBJECTS and predicate in {"references", "causes", "precedes", "affiliated_with"}:
        reasons.append("generic_object")

    if any(hint in claim_text for hint in META_REFERENCE_HINTS):
        reasons.append("draft_artifact")

    if any(hint in claim_text for hint in SOUND_CLAIM_HINTS):
        reasons.append("sound_or_media_meta")

    if any(hint in claim_text for hint in SCENE_META_HINTS):
        reasons.append("scene_meta")

    if predicate in {"occurs_at", "occurs_on", "located_at", "present_on"} and (
        subject in {"scene", "the scene", "this movie", "the movie", "filming", "the filming"}
        or " scene " in f" {claim_text} "
        or claim_text.startswith("scene ")
        or claim_text.startswith("the scene ")
        or claim_text.startswith("this movie ")
    ):
        reasons.append("scene_meta")

    if LOW_VALUE_ATTRIBUTE_RE.search(claim_text):
        reasons.append("low_value_attribute")

    if reasons:
        return "abstain", 0.0, reasons

    low_confidence_reasons: List[str] = []
    if predicate == "performs" and MICRO_ACTION_RE.search(claim_text):
        low_confidence_reasons.append("micro_action_granularity")
    if predicate == "located_at" and STRUCTURAL_LOCATION_RE.search(claim_text):
        low_confidence_reasons.append("structural_location_scaffold")
    if predicate == "possesses" and PROP_ATTRIBUTE_RE.search(claim_text):
        low_confidence_reasons.append("prop_attribute_detail")
    if predicate == "undergoes" and PROCESS_STATE_RE.search(claim_text):
        low_confidence_reasons.append("process_state_fragment")
    if predicate == "experiences" and INTERPRETIVE_EXPERIENCE_RE.search(claim_text):
        low_confidence_reasons.append("interpretive_state")
    if predicate == "present_on" and SCENE_PRESENCE_RE.search(claim_text):
        low_confidence_reasons.append("scene_presence_scaffold")
    if predicate == "causes" and EXPLANATORY_CAUSE_RE.search(claim_text):
        low_confidence_reasons.append("explanatory_causality")
    if predicate == "precedes" and ORDERING_SCAFFOLD_RE.search(claim_text):
        low_confidence_reasons.append("ordering_scaffold")
    if predicate == "occurs_at" and TIME_SCAFFOLD_RE.search(claim_text):
        low_confidence_reasons.append("time_scaffold")
    if predicate == "is_a" and SCENE_DESCRIPTION_RE.search(claim_text):
        low_confidence_reasons.append("scene_description")
    if predicate == "affiliated_with" and WEAK_AFFILIATION_RE.search(claim_text):
        low_confidence_reasons.append("weak_affiliation")
    if predicate in {"experiences", "undergoes"} and any(hint in claim_text for hint in INTERPRETIVE_STATE_HINTS):
        low_confidence_reasons.append("interpretive_state")

    if predicate == "performs" and any(hint in claim_text for hint in WEAK_ACTION_HINTS):
        low_confidence_reasons.append("weak_action_paraphrase")
    if predicate in {"references", "precedes", "causes"} and VAGUE_REFERENCE_RE.search(claim_text):
        low_confidence_reasons.append("vague_reference")
    if predicate in {"performs", "references", "affiliated_with"} and LOW_VALUE_DIALOGUE_RE.search(claim_text):
        low_confidence_reasons.append("low_value_dialogue")
    if predicate in {"is_a", "possesses", "experiences"} and LOW_VALUE_DESCRIPTION_RE.search(claim_text):
        low_confidence_reasons.append("low_value_description")
    if predicate in {"performs", "experiences", "undergoes"} and LOW_VALUE_EVENT_RE.search(claim_text):
        low_confidence_reasons.append("low_value_event")

    if "about to " in claim_text:
        low_confidence_reasons.append("irrealis_future_action")

    if low_confidence_reasons:
        return "low_confidence", 0.4, low_confidence_reasons

    return "keep", 0.8, []


def should_abstain_low_confidence_claim(claim: Claim, reasons: List[str]) -> bool:
    """Return True when a low-confidence claim is too weak to score in CSV re-verification."""
    reason_set = set(reasons)
    if reason_set & {
        "ordering_scaffold",
        "explanatory_causality",
        "scene_presence_scaffold",
        "weak_affiliation",
        "time_scaffold",
        "scene_description",
        "vague_reference",
        "low_value_dialogue",
        "low_value_description",
        "low_value_event",
    }:
        return True
    if "prop_attribute_detail" in reason_set:
        return True
    if "micro_action_granularity" in reason_set and NON_ATOMIC_MICRO_ACTION_RE.search(_normalize_text(claim.claim_text)):
        return True
    return False


def _has_quoted_span(text: str) -> bool:
    if '"' in text:
        return True
    return bool(re.search(r"(^|[\s(])'[^']{2,}'(?=[$\s).,!?:;])", text))
