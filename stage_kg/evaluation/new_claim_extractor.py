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
