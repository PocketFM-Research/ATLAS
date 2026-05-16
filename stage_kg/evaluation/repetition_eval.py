"""Scene-level repetition evaluation (Metric 3)."""

import csv
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from stage_kg.evaluation.config import EvaluationConfig

try:
    import spacy
except ImportError:
    spacy = None


SceneInput = Union[str, Sequence[str], Dict[str, str]]
LLMAdjudicator = Callable[[str, str, Dict[str, float]], Dict[str, object]]

SCENE_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"(?:scene|sc)\s*[\w.-]+(?:\s*[-:]\s*.+)?"
    r"|(?:int|ext|int/ext|i/e)\.\s+.+"
    r"|\d+\s*[\).:：、-]\s*.+"
    r")\s*$",
    re.IGNORECASE,
)
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
PUNCT_RE = re.compile(r"[,:;.!?]")
QUOTE_RE = re.compile(r"[\"“”']([^\"“”']{2,120})[\"“”']")

STOPWORDS = {
    "a", "about", "again", "all", "already", "also", "an", "and", "any",
    "are", "as", "at", "be", "been", "before", "but", "by", "could", "did",
    "didn", "do", "does", "don", "even", "for", "from", "had", "hadn", "has",
    "have", "he", "her", "him", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "just", "like", "maybe", "no", "not", "of", "on", "or",
    "s", "she", "so", "still", "t", "that", "the", "there", "they", "this", "though",
    "time", "to", "too", "ve", "was", "wasn", "what", "when", "where", "while",
    "why", "with", "would", "you",
}
COMMON_EVENT_VERBS = {
    "arrive", "ask", "attack", "call", "change", "check", "decide", "enter",
    "fight", "find", "give", "go", "hide", "leave", "open", "place", "run",
    "take", "tell", "unlock", "walk",
    "found", "gave", "hid", "left", "ran", "told", "took", "went",
}
STATE_WORDS = {
    "afraid", "angry", "alone", "alive", "anxious", "ashamed", "calm", "dead",
    "determined", "excited", "frightened", "guilty", "happy", "hurt", "injured",
    "nervous", "ready", "sad", "safe", "sick", "tired", "trapped", "upset",
    "worried", "wounded",
}

PAIR_SCORE_FIELDS = [
    "semantic_similarity",
    "ngram_overlap",
    "sentence_overlap",
    "event_overlap",
    "state_overlap",
    "structure_similarity",
    "novelty_score",
    "motif_overlap",
    "beat_overlap",
    "internal_repetition_score",
]
CSV_FIELDS = [
    "scene_i_id",
    "scene_j_id",
    *PAIR_SCORE_FIELDS,
    "repetition_types",
    "is_repetitive",
    "llm_used",
    "llm_reason",
    "text_i_sample",
    "text_j_sample",
]


def _avg(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


@dataclass
class RepetitionMetrics:
    """Aggregate scene-level repetition metrics."""

    total_scene_pairs: int
    repetitive_pairs: int
    avg_semantic_similarity: float
    avg_ngram_overlap: float
    repetition_rate: float
    avg_sentence_overlap: float = 0.0
    avg_event_overlap: float = 0.0
    avg_state_overlap: float = 0.0
    avg_structure_similarity: float = 0.0
    avg_novelty_score: float = 0.0
    avg_motif_overlap: float = 0.0
    avg_beat_overlap: float = 0.0
    avg_internal_repetition_score: float = 0.0

    def to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def from_instances(cls, instances: List["RepetitionInstance"]):
        repetitive_pairs = sum(instance.is_repetitive for instance in instances)
        averages = {
            f"avg_{field}": _avg(getattr(inst, field) for inst in instances)
            for field in PAIR_SCORE_FIELDS
        }
        return cls(
            total_scene_pairs=len(instances),
            repetitive_pairs=repetitive_pairs,
            repetition_rate=(repetitive_pairs / len(instances)) if instances else 0.0,
            **averages,
        )


@dataclass
class RepetitionInstance:
    """A detected or evaluated scene-pair repetition instance."""

    scene_i_id: str
    scene_j_id: str
    semantic_similarity: float
    ngram_overlap: float
    sentence_overlap: float
    event_overlap: float
    state_overlap: float
    structure_similarity: float
    novelty_score: float
    motif_overlap: float
    beat_overlap: float
    internal_repetition_score: float
    repetition_types: List[str]
    is_repetitive: bool
    sample_text_i: str
    sample_text_j: str
    llm_used: bool = False
    llm_reason: str = ""


@dataclass
class InternalRepetitionInstance:
    """Repetition found inside one scene."""

    scene_id: str
    internal_sentence_repetition: float
    internal_ngram_repetition: float
    repeated_sentences: List[str]
    repeated_phrases: List[str]
    sample_text: str
    repeated_sentence_instances: List[Dict[str, object]] = field(default_factory=list)
    repeated_phrase_instances: List[Dict[str, object]] = field(default_factory=list)

    @property
    def internal_repetition_score(self) -> float:
        return max(self.internal_sentence_repetition, self.internal_ngram_repetition)


@dataclass
class SceneProfile:
    """Local NLP profile for one scene."""

    scene_id: str
    text: str
    sentences: List[str]
    sentence_keys: Set[str]
    event_signatures: Set[str]
    state_signatures: Set[str]
    info_signatures: Set[str]
    motif_signatures: Set[str]
    beat_signatures: Set[str]
    quotes: Set[str]
    pos_patterns: Set[str]
    structure_signature: List[float]
    internal_sentence_repetition: float = 0.0
    internal_ngram_repetition: float = 0.0
    repeated_sentences: List[str] = field(default_factory=list)
    repeated_phrases: List[str] = field(default_factory=list)
    repeated_sentence_instances: List[Dict[str, object]] = field(default_factory=list)
    repeated_phrase_instances: List[Dict[str, object]] = field(default_factory=list)


class RepetitionEvaluator:
    """Evaluate repetition, novelty, and story progression across full scene text."""

    def __init__(
        self,
        config: EvaluationConfig = None,
        llm_adjudicator: LLMAdjudicator = None,
        max_llm_pairs: int = 10,
    ):
        self.config = config or EvaluationConfig()
        self.llm_adjudicator = llm_adjudicator
        self.max_llm_pairs = max_llm_pairs
        self.llm_calls = 0
        self.embedding_model = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._semantic_chunk_cache: Dict[str, List[str]] = {}
        self.nlp = self._load_spacy()

    def evaluate_repetition(
        self,
        scenes: SceneInput,
    ) -> Tuple[RepetitionMetrics, List[RepetitionInstance]]:
        """Evaluate repetition across full scene texts."""
        profiles = self._profile_scenes(scenes)
        prior_info = self._previous_info_signatures(profiles)
        instances = []
        for index, scene_i in enumerate(profiles):
            for scene_j in profiles[index + 1:]:
                instances.append(self._build_instance(
                    scene_i,
                    scene_j,
                    prior_info[scene_j.scene_id],
                ))
        return RepetitionMetrics.from_instances(instances), instances

    def evaluate_internal_repetition(
        self,
        scenes: SceneInput,
        threshold: float = 0.15,
    ) -> List[InternalRepetitionInstance]:
        """Evaluate repeated wording inside each individual scene."""
        profiles = self._profile_scenes(scenes)
        results = []
        for profile in profiles:
            score = max(profile.internal_sentence_repetition, profile.internal_ngram_repetition)
            explicit_instances = (
                profile.repeated_sentence_instances
                + profile.repeated_phrase_instances
            )
            if (
                not explicit_instances
                and score < threshold
            ):
                continue
            if not explicit_instances:
                continue
            results.append(InternalRepetitionInstance(
                scene_id=profile.scene_id,
                internal_sentence_repetition=profile.internal_sentence_repetition,
                internal_ngram_repetition=profile.internal_ngram_repetition,
                repeated_sentences=profile.repeated_sentences or [],
                repeated_phrases=profile.repeated_phrases or [],
                sample_text=profile.text[:220],
                repeated_sentence_instances=profile.repeated_sentence_instances or [],
                repeated_phrase_instances=profile.repeated_phrase_instances or [],
            ))
        return results

    def _profile_scenes(self, scenes: SceneInput) -> List[SceneProfile]:
        return [
            self._profile_scene(scene_id, text)
            for scene_id, text in self._normalize_scene_units(scenes)
        ]

    def save_instances(self, instances: List[RepetitionInstance], output_path: str):
        """Save repetition instances to CSV."""
        with open(output_path, "w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for instance in instances:
                row = {
                    "scene_i_id": instance.scene_i_id,
                    "scene_j_id": instance.scene_j_id,
                    "repetition_types": ";".join(instance.repetition_types),
                    "is_repetitive": instance.is_repetitive,
                    "llm_used": instance.llm_used,
                    "llm_reason": instance.llm_reason,
                    "text_i_sample": instance.sample_text_i,
                    "text_j_sample": instance.sample_text_j,
                }
                row.update({
                    field: f"{getattr(instance, field):.3f}"
                    for field in PAIR_SCORE_FIELDS
                })
                writer.writerow(row)

    def _normalize_scene_units(self, scenes: SceneInput) -> List[Tuple[str, str]]:
        if isinstance(scenes, str):
            return self._split_story_into_scenes(scenes)

        if isinstance(scenes, dict):
            if any(isinstance(value, dict) for value in scenes.values()):
                raise TypeError(
                    "RepetitionEvaluator now accepts full scene text only. "
                    "Pass {scene_id: scene_text}, not {scene_id: {character_id: text}}."
                )
            if not all(isinstance(value, str) for value in scenes.values()):
                raise TypeError("Scene dictionaries must be {scene_id: full_scene_text}.")
            return [
                (str(scene_id), text.strip())
                for scene_id, text in scenes.items()
                if text.strip()
            ]

        if isinstance(scenes, Sequence):
            return [
                (f"scene_{index:03d}", text.strip())
                for index, text in enumerate(scenes, start=1)
                if isinstance(text, str) and text.strip()
            ]

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

    def _build_instance(
        self,
        scene_i: SceneProfile,
        scene_j: SceneProfile,
        prior_info_for_j: Set[str],
    ) -> RepetitionInstance:
        scores = self._score_scene_pair(scene_i, scene_j, prior_info_for_j)
        repetition_types = self._repetition_types(scores)
        is_repetitive = bool(repetition_types)
        llm_used = False
        llm_reason = ""

        if self._should_use_llm(scores, is_repetitive):
            result = self.llm_adjudicator(scene_i.text, scene_j.text, scores)
            self.llm_calls += 1
            llm_used = True
            llm_reason = str(result.get("reason", ""))
            is_repetitive = bool(result.get("is_repetitive", is_repetitive))
            if is_repetitive and "llm_story_progression" not in repetition_types:
                repetition_types.append("llm_story_progression")
            if not is_repetitive:
                repetition_types = []

        return RepetitionInstance(
            scene_i_id=scene_i.scene_id,
            scene_j_id=scene_j.scene_id,
            semantic_similarity=scores["semantic_similarity"],
            ngram_overlap=scores["ngram_overlap"],
            sentence_overlap=scores["sentence_overlap"],
            event_overlap=scores["event_overlap"],
            state_overlap=scores["state_overlap"],
            structure_similarity=scores["structure_similarity"],
            novelty_score=scores["novelty_score"],
            motif_overlap=scores["motif_overlap"],
            beat_overlap=scores["beat_overlap"],
            internal_repetition_score=scores["internal_repetition_score"],
            repetition_types=repetition_types,
            is_repetitive=is_repetitive,
            sample_text_i=scene_i.text[:160],
            sample_text_j=scene_j.text[:160],
            llm_used=llm_used,
            llm_reason=llm_reason,
        )

    def _score_scene_pair(
        self,
        scene_i: SceneProfile,
        scene_j: SceneProfile,
        prior_info_for_j: Set[str],
    ) -> Dict[str, float]:
        whole_scene_similarity = self._semantic_similarity(scene_i.text, scene_j.text)
        local_scene_similarity = self._max_chunk_similarity(scene_i, scene_j)
        return {
            "semantic_similarity": max(whole_scene_similarity, local_scene_similarity),
            "whole_scene_similarity": whole_scene_similarity,
            "local_scene_similarity": local_scene_similarity,
            "ngram_overlap": self._ngram_overlap(scene_i.text, scene_j.text),
            "sentence_overlap": self._sentence_overlap(scene_i, scene_j),
            "event_overlap": self._signature_overlap(
                scene_i.event_signatures, scene_j.event_signatures
            ),
            "state_overlap": self._set_overlap(scene_i.state_signatures, scene_j.state_signatures),
            "structure_similarity": self._structure_similarity(scene_i, scene_j),
            "novelty_score": self._novelty_score(scene_j.info_signatures, prior_info_for_j),
            "motif_overlap": self._coverage_overlap(
                scene_i.motif_signatures,
                scene_j.motif_signatures,
            ),
            "beat_overlap": self._coverage_overlap(
                scene_i.beat_signatures,
                scene_j.beat_signatures,
            ),
            "internal_repetition_score": max(
                scene_i.internal_sentence_repetition,
                scene_i.internal_ngram_repetition,
                scene_j.internal_sentence_repetition,
                scene_j.internal_ngram_repetition,
            ),
        }

    def _max_chunk_similarity(self, scene_i: SceneProfile, scene_j: SceneProfile) -> float:
        chunks_i = self._semantic_chunks(scene_i)
        chunks_j = self._semantic_chunks(scene_j)
        if not chunks_i or not chunks_j:
            return 0.0
        return max(
            self._semantic_similarity(chunk_i, chunk_j)
            for chunk_i in chunks_i
            for chunk_j in chunks_j
        )

    def _semantic_chunks(self, scene: SceneProfile) -> List[str]:
        cached = self._semantic_chunk_cache.get(scene.scene_id)
        if cached is not None:
            return cached

        sentences = [
            sentence
            for sentence in self._line_sentences(scene.text)
            if len(self._words(sentence)) >= 4
        ]
        if len(sentences) <= 4:
            chunks = [" ".join(sentences)] if sentences else [scene.text]
        else:
            chunks = []
            for window_size in (4, 6):
                step = max(1, window_size // 2)
                for index in range(0, len(sentences), step):
                    window = sentences[index:index + window_size]
                    if len(window) >= 3:
                        chunks.append(" ".join(window))
            chunks.append(" ".join(sentences))

        self._semantic_chunk_cache[scene.scene_id] = chunks
        return chunks

    def _repetition_types(self, scores: Dict[str, float]) -> List[str]:
        repetition_types = []
        semantic_similarity = scores["semantic_similarity"]
        whole_scene_similarity = scores.get("whole_scene_similarity", semantic_similarity)
        local_scene_similarity = scores.get("local_scene_similarity", semantic_similarity)
        ngram_overlap = scores["ngram_overlap"]
        sentence_overlap = scores["sentence_overlap"]
        event_overlap = scores["event_overlap"]
        state_overlap = scores["state_overlap"]
        structure_similarity = scores["structure_similarity"]
        novelty_score = scores["novelty_score"]
        motif_overlap = scores["motif_overlap"]
        beat_overlap = scores["beat_overlap"]

        strong_scene_reuse = (
            semantic_similarity >= 0.90
            and motif_overlap >= 0.70
            and (sentence_overlap >= 0.12 or ngram_overlap >= 0.12)
            and novelty_score <= 0.30
        )
        moderate_scene_reuse = (
            semantic_similarity >= 0.82
            and motif_overlap >= 0.78
            and beat_overlap >= 0.70
            and (sentence_overlap >= 0.10 or ngram_overlap >= 0.10)
            and novelty_score <= 0.35
        )

        if strong_scene_reuse or (
            semantic_similarity >= self.config.similarity_threshold
            and ngram_overlap >= 0.20
            and sentence_overlap >= 0.18
            and motif_overlap >= 0.50
        ):
            repetition_types.append("scene_semantic_repetition")
        if moderate_scene_reuse:
            repetition_types.append("semantic_low_novelty")
        if whole_scene_similarity >= 0.82 and motif_overlap >= 0.70 and novelty_score <= 0.35:
            repetition_types.append("whole_scene_pattern_repetition")
        if local_scene_similarity >= 0.72 and (
            whole_scene_similarity >= 0.70
            and motif_overlap >= 0.70
            and (sentence_overlap >= 0.10 or ngram_overlap >= 0.10)
        ) and novelty_score <= 0.86:
            repetition_types.append("local_scene_pattern_repetition")
        if semantic_similarity >= getattr(self.config, "cross_scene_embedding_threshold", 0.78) and (
            sentence_overlap >= 0.12
            and motif_overlap >= 0.70
            and novelty_score <= 0.35
        ):
            repetition_types.append("embedding_scene_pattern_repetition")
        if sentence_overlap >= 0.5 or (
            sentence_overlap >= 0.25 and (event_overlap >= 0.25 or motif_overlap >= 0.50)
        ):
            repetition_types.append("cross_scene_sentence_reuse")
        if semantic_similarity >= 0.82 and motif_overlap >= 0.70 and (
            event_overlap >= 0.18 or sentence_overlap >= 0.12 or ngram_overlap >= 0.12
        ):
            repetition_types.append("narrative_repetition")
        if (
            semantic_similarity >= 0.82
            and motif_overlap >= 0.70
            and event_overlap >= 0.18
            and novelty_score <= 0.35
        ):
            repetition_types.append("repeated_story_beat")
        if event_overlap >= 0.65 and novelty_score <= 0.35:
            repetition_types.append("repeated_events_low_novelty")
        if state_overlap >= 0.65 and novelty_score <= 0.35:
            repetition_types.append("repeated_character_state")
        if structure_similarity >= 0.9 and novelty_score <= 0.35:
            repetition_types.append("low_creativity_structure")
        if motif_overlap >= 0.70 and semantic_similarity >= 0.82 and novelty_score <= 0.35:
            repetition_types.append("motif_repetition")
        if beat_overlap >= 0.70 and semantic_similarity >= 0.82 and novelty_score <= 0.35:
            repetition_types.append("loop_repetition")
        return repetition_types

    def _should_use_llm(self, scores: Dict[str, float], is_repetitive: bool) -> bool:
        if not self.llm_adjudicator or self.llm_calls >= self.max_llm_pairs or is_repetitive:
            return False

        semantic_similarity = scores["semantic_similarity"]
        event_overlap = scores["event_overlap"]
        novelty_score = scores["novelty_score"]
        structure_similarity = scores["structure_similarity"]

        return any((
            0.82 <= semantic_similarity < 0.92 and 0.35 < novelty_score < 0.65,
            event_overlap >= 0.5 and 0.35 < novelty_score < 0.65,
            structure_similarity >= 0.85 and 0.35 < novelty_score < 0.65,
        ))

    def _profile_scene(self, scene_id: str, text: str) -> SceneProfile:
        doc = self.nlp(text) if self.nlp else None
        sentences = self._sentences(text, doc)
        sentence_keys = {
            normalized
            for sentence in sentences
            for normalized in [self._normalize_text(sentence)]
            if normalized
        }
        quotes = self._quotes(text)
        event_signatures = self._event_signatures(text, doc)
        state_signatures = self._state_signatures(text, doc)
        phrase_signatures = self._phrase_signatures(text, doc)
        motif_signatures = self._motif_signatures(text, doc, quotes)
        beat_signatures = self._beat_signatures(sentences, doc)
        repeated_sentences = self._repeated_sentences(sentences)
        repeated_phrases = self._repeated_phrases(text)
        repeated_sentence_instances = self._repeated_sentence_instances(sentences)
        repeated_phrase_instances = self._repeated_phrase_instances(text)

        return SceneProfile(
            scene_id=scene_id,
            text=text,
            sentences=sentences,
            sentence_keys=sentence_keys,
            event_signatures=event_signatures,
            state_signatures=state_signatures,
            info_signatures=(
                event_signatures
                | state_signatures
                | phrase_signatures
                | motif_signatures
                | beat_signatures
            ),
            motif_signatures=motif_signatures,
            beat_signatures=beat_signatures,
            quotes=quotes,
            pos_patterns=self._pos_patterns(doc),
            structure_signature=self._structure_signature(sentences),
            internal_sentence_repetition=self._internal_sentence_repetition(sentences),
            internal_ngram_repetition=self._internal_ngram_repetition(text),
            repeated_sentences=repeated_sentences,
            repeated_phrases=repeated_phrases,
            repeated_sentence_instances=repeated_sentence_instances,
            repeated_phrase_instances=repeated_phrase_instances,
        )

    def _load_spacy(self):
        if spacy is None:
            return None
        try:
            return spacy.load("en_core_web_sm")
        except Exception:
            return None

    def _sentences(self, text: str, doc) -> List[str]:
        line_sentences = self._line_sentences(text)
        if doc is not None:
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            if sentences:
                return self._merge_sentence_candidates(sentences, line_sentences)
        return line_sentences or [
            match.group(0).strip()
            for match in SENTENCE_RE.finditer(text)
            if match.group(0).strip()
        ]

    def _line_sentences(self, text: str) -> List[str]:
        sentences = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = [
                match.group(0).strip()
                for match in SENTENCE_RE.finditer(line)
                if match.group(0).strip()
            ]
            sentences.extend(parts or [line])
        return sentences

    def _merge_sentence_candidates(self, *groups: Sequence[str]) -> List[str]:
        merged = []
        seen = set()
        for group in groups:
            for sentence in group:
                normalized = self._normalize_text(sentence)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                merged.append(sentence)
        return merged

    def _quotes(self, text: str) -> Set[str]:
        return {
            normalized
            for quote in QUOTE_RE.findall(text)
            for normalized in [self._normalize_text(quote)]
            if normalized
        }

    def _phrase_signatures(self, text: str, doc) -> Set[str]:
        if doc is not None:
            return {
                "noun:" + normalized
                for chunk in doc.noun_chunks
                for normalized in [self._normalize_text(chunk.text)]
                if normalized
            }
        tokens = self._content_words(text)
        return {
            "noun:" + " ".join(tokens[index:index + 2])
            for index in range(len(tokens) - 1)
        }

    def _motif_signatures(self, text: str, doc, quotes: Set[str]) -> Set[str]:
        tokens = self._content_words(text, doc)
        motifs = {f"motif:{token}" for token in tokens}
        motifs.update(
            f"motif:{'_'.join(window)}"
            for size in (2, 3)
            for window in self._windows(tokens, size)
            if len(set(window)) > 1
        )
        motifs.update(f"quote:{quote}" for quote in quotes)
        return motifs

    def _beat_signatures(self, sentences: List[str], doc) -> Set[str]:
        beats = set()
        for sentence in sentences:
            words = self._content_words(sentence)
            if not words:
                continue
            beats.add("beat:sentence_topic:" + "_".join(sorted(set(words))[:6]))
            beats.update(
                "beat:phrase:" + "_".join(window)
                for size in (2, 3)
                for window in self._windows(words, size)
                if len(set(window)) > 1
            )

        if doc is not None:
            for sent in doc.sents:
                verbs = sorted({
                    token.lemma_.lower()
                    for token in sent
                    if token.pos_ == "VERB" and not token.is_stop
                })
                nouns = sorted({
                    token.lemma_.lower()
                    for token in sent
                    if token.pos_ in {"NOUN", "PROPN", "PRON"} and not token.is_stop
                })
                if verbs and nouns:
                    beats.add(
                        "beat:action:" + "_".join(verbs[:2]) + ":" + "_".join(nouns[:4])
                    )
        return beats

    def _event_signatures(self, text: str, doc) -> Set[str]:
        if doc is not None:
            signatures = set()
            for token in doc:
                if token.pos_ != "VERB":
                    continue
                context = [
                    neighbor.lemma_.lower()
                    for neighbor in doc[max(0, token.i - 3): token.i + 4]
                    if neighbor.pos_ in {"NOUN", "PROPN", "PRON", "ADJ"}
                ]
                if context:
                    signatures.add("event:" + " ".join([token.lemma_.lower()] + context[:4]))
            return signatures

        tokens = self._words(text)
        return {
            "event:" + " ".join(window)
            for index, token in enumerate(tokens)
            if self._looks_like_event_verb(token)
            for window in [tokens[max(0, index - 3): index + 5]]
            if window
        }

    def _state_signatures(self, text: str, doc) -> Set[str]:
        if doc is not None:
            signatures = {f"entity:{ent.text.lower()}" for ent in doc.ents}
            for token in doc:
                if token.lemma_.lower() in STATE_WORDS or token.pos_ == "ADJ":
                    signatures.add(f"state:{self._nearest_subject(token)}:{token.lemma_.lower()}")
            return signatures

        tokens = self._words(text)
        signatures = {
            f"entity:{match.lower()}"
            for match in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        }
        signatures.update(
            f"state:{tokens[index - 1] if index else ''}:{token}"
            for index, token in enumerate(tokens)
            if token in STATE_WORDS
        )
        return signatures

    def _structure_signature(self, sentences: List[str]) -> List[float]:
        if not sentences:
            return [0.0, 0.0, 0.0, 0.0]

        lengths = [len(self._words(sentence)) for sentence in sentences]
        punct_counts = [len(PUNCT_RE.findall(sentence)) for sentence in sentences]
        avg_len = sum(lengths) / len(lengths)
        variance = sum((length - avg_len) ** 2 for length in lengths) / len(lengths)
        return [len(sentences), avg_len, variance ** 0.5, sum(punct_counts) / len(punct_counts)]

    def _pos_patterns(self, doc) -> Set[str]:
        if doc is None:
            return set()
        return {
            pattern
            for sent in doc.sents
            for pattern in [
                " ".join(
                    token.pos_
                    for token in sent
                    if not token.is_punct and not token.is_space
                )
            ]
            if pattern
        }

    def _sentence_overlap(self, scene_i: SceneProfile, scene_j: SceneProfile) -> float:
        return max(
            self._set_overlap(scene_i.sentence_keys, scene_j.sentence_keys),
            self._set_overlap(scene_i.quotes, scene_j.quotes),
            self._semantic_sentence_overlap(scene_i.sentences, scene_j.sentences),
        )

    def _semantic_sentence_overlap(self, sentences_i: List[str], sentences_j: List[str]) -> float:
        filtered_i = [sentence for sentence in sentences_i if len(self._words(sentence)) >= 4]
        filtered_j = [sentence for sentence in sentences_j if len(self._words(sentence)) >= 4]
        if not filtered_i or not filtered_j:
            return 0.0

        matches = sum(
            1
            for sentence_i in filtered_i
            if max(self._semantic_similarity(sentence_i, sentence_j) for sentence_j in filtered_j) >= 0.82
        )
        return matches / max(len(filtered_i), len(filtered_j))

    def _structure_similarity(self, scene_i: SceneProfile, scene_j: SceneProfile) -> float:
        vector_similarity = self._vector_cosine(
            scene_i.structure_signature,
            scene_j.structure_signature,
        )
        if scene_i.pos_patterns or scene_j.pos_patterns:
            return (vector_similarity + self._set_overlap(scene_i.pos_patterns, scene_j.pos_patterns)) / 2
        return vector_similarity

    def _internal_sentence_repetition(self, sentences: List[str]) -> float:
        filtered = [
            sentence.strip()
            for sentence in sentences
            if len(self._words(sentence)) >= 2
            and not self._is_screenplay_boilerplate(sentence)
        ]
        if len(filtered) < 2:
            return 0.0
        instance_count = len(self._repeated_sentence_instances(filtered))
        return min(1.0, instance_count / max(1, len(filtered) * 0.2))

    def _internal_ngram_repetition(self, text: str, n: int = 3) -> float:
        words = self._words(text)
        if len(words) < n * 2:
            return 0.0
        ngrams = list(self._windows(words, n))
        if not ngrams:
            return 0.0
        repeated = len(ngrams) - len(set(ngrams))
        return min(1.0, repeated / max(1, len(ngrams) * 0.12))

    def _repeated_sentences(self, sentences: List[str]) -> List[str]:
        return [item["text"] for item in self._repeated_sentence_instances(sentences)]

    def _repeated_phrases(self, text: str, n: int = 3) -> List[str]:
        return [item["text"] for item in self._repeated_phrase_instances(text, n=n)]

    def _repeated_sentence_instances(self, sentences: List[str]) -> List[Dict[str, object]]:
        filtered: List[Tuple[int, str]] = []
        for index, sentence in enumerate(sentences, start=1):
            stripped = sentence.strip()
            if len(self._words(stripped)) < 2:
                continue
            if self._is_screenplay_boilerplate(stripped):
                continue
            if self._is_non_content_sentence(stripped):
                continue
            filtered.append((index, stripped))
        if len(filtered) < 2:
            return []

        instances: List[Dict[str, object]] = []
        consumed: Set[int] = set()
        local_window = int(getattr(self.config, "internal_repetition_sentence_window", 4))
        exact_groups: Dict[str, List[Tuple[int, str]]] = {}
        for position, sentence in filtered:
            exact_groups.setdefault(self._normalize_text(sentence), []).append((position, sentence))
        for normalized, group in exact_groups.items():
            if len(group) < 2 or not normalized:
                continue
            for cluster in self._local_clusters(group, local_window):
                if len(cluster) < 2:
                    continue
                consumed.update(position for position, _ in cluster)
                instances.append({
                    "text": cluster[0][1],
                    "count": len(cluster),
                    "positions": [position for position, _ in cluster],
                    "source": "exact_sentence_duplicate",
                    "matched_texts": [text for _, text in cluster],
                })

        threshold = float(getattr(self.config, "internal_sentence_similarity_threshold", 0.98))
        for local_i, (position_i, sentence_i) in enumerate(filtered):
            if position_i in consumed:
                continue
            group = [(position_i, sentence_i)]
            for position_j, sentence_j in filtered[local_i + 1:]:
                if position_j in consumed:
                    continue
                if position_j - position_i > local_window:
                    break
                if self._semantic_similarity(sentence_i, sentence_j) >= threshold:
                    group.append((position_j, sentence_j))
            if len(group) < 2:
                continue
            consumed.update(position for position, _ in group)
            instances.append({
                "text": group[0][1],
                "count": len(group),
                "positions": [position for position, _ in group],
                "source": "sentence_embedding_similarity",
                "matched_texts": [text for _, text in group],
            })
        return instances

    def _local_clusters(
        self,
        group: Sequence[Tuple[int, str]],
        local_window: int,
    ) -> List[List[Tuple[int, str]]]:
        clusters: List[List[Tuple[int, str]]] = []
        current: List[Tuple[int, str]] = []
        for item in sorted(group, key=lambda value: value[0]):
            if not current or item[0] - current[-1][0] <= local_window:
                current.append(item)
                continue
            clusters.append(current)
            current = [item]
        if current:
            clusters.append(current)
        return clusters

    def _is_non_content_sentence(self, sentence: str) -> bool:
        stripped = sentence.strip()
        words = self._words(stripped)
        if not words:
            return True
        letters = [char for char in stripped if char.isalpha()]
        if letters and stripped.upper() == stripped and len(words) <= 4:
            return True
        if stripped.startswith("(") and stripped.endswith(")"):
            return True
        if stripped.endswith("-"):
            return True
        content_words = [
            word
            for word in words
            if word not in STOPWORDS and len(word) > 2
        ]
        if len(content_words) < 2:
            return True
        if len(words) <= 2 and any(word in STOPWORDS for word in words):
            return True
        return False

    def _repeated_phrase_instances(self, text: str, n: int = 4) -> List[Dict[str, object]]:
        words = self._words(text)
        if len(words) < n * 2:
            return []

        phrase_positions: Dict[Tuple[str, ...], List[int]] = {}
        for index, window in enumerate(self._windows(words, n), start=1):
            if not any(word not in STOPWORDS for word in window):
                continue
            phrase_positions.setdefault(window, []).append(index)

        local_window = int(getattr(self.config, "internal_repetition_phrase_window", 20))
        instances: List[Dict[str, object]] = []
        for phrase, positions in phrase_positions.items():
            if len(positions) < 2:
                continue
            clusters = self._local_position_clusters(positions, local_window)
            for cluster in clusters:
                if len(cluster) < 2:
                    continue
                instances.append({
                    "text": " ".join(phrase),
                    "count": len(cluster),
                    "positions": cluster,
                    "source": f"{n}_gram_duplicate",
                    "matched_texts": [" ".join(phrase)] * len(cluster),
                })
        return instances

    def _local_position_clusters(
        self,
        positions: Sequence[int],
        local_window: int,
    ) -> List[List[int]]:
        clusters: List[List[int]] = []
        current: List[int] = []
        for position in sorted(positions):
            if not current or position - current[-1] <= local_window:
                current.append(position)
                continue
            clusters.append(current)
            current = [position]
        if current:
            clusters.append(current)
        return clusters

    def _is_screenplay_boilerplate(self, text: str) -> bool:
        return bool(re.search(
            r"\b(?:continued|cont'd|green draft|page|fade in|fade out|cut to|title|scene)\b",
            text,
            re.IGNORECASE,
        ))

    def _previous_info_signatures(self, profiles: List[SceneProfile]) -> Dict[str, Set[str]]:
        prior: Set[str] = set()
        result: Dict[str, Set[str]] = {}
        for profile in profiles:
            result[profile.scene_id] = set(prior)
            prior.update(profile.info_signatures)
        return result

    def _novelty_score(self, current_info: Set[str], prior_info: Set[str]) -> float:
        if not current_info:
            return 0.0
        return len(current_info - prior_info) / len(current_info)

    def _signature_overlap(self, values_i: Set[str], values_j: Set[str]) -> float:
        lexical_overlap = self._set_overlap(values_i, values_j)
        if lexical_overlap >= 0.75 or not values_i or not values_j:
            return lexical_overlap

        matches = sum(
            1
            for value_i in values_i
            if max(self._semantic_similarity(value_i, value_j) for value_j in values_j) >= 0.82
        )
        return max(lexical_overlap, matches / len(values_i))

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        return self._vector_cosine(self._embedding(text1), self._embedding(text2))

    def _embedding(self, text: str):
        key = text.strip()
        if key not in self._embedding_cache:
            if self.embedding_model is None:
                self.embedding_model = SentenceTransformer(
                    getattr(self.config, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
                )
            self._embedding_cache[key] = self.embedding_model.encode(
                key,
                convert_to_tensor=False,
            )
        return self._embedding_cache[key]

    def _ngram_overlap(self, text1: str, text2: str, n: int = 2) -> float:
        ngrams_1 = set(self._windows(self._words(text1), n))
        ngrams_2 = set(self._windows(self._words(text2), n))
        return self._set_overlap(ngrams_1, ngrams_2)

    def _set_overlap(self, values_i: Set[str], values_j: Set[str]) -> float:
        if not values_i or not values_j:
            return 0.0
        return len(values_i & values_j) / len(values_i | values_j)

    def _coverage_overlap(self, values_i: Set[str], values_j: Set[str]) -> float:
        if not values_i or not values_j:
            return 0.0
        return len(values_i & values_j) / min(len(values_i), len(values_j))

    def _vector_cosine(self, vector_i, vector_j) -> float:
        arr_i = np.asarray(vector_i, dtype=float)
        arr_j = np.asarray(vector_j, dtype=float)
        norm_product = np.linalg.norm(arr_i) * np.linalg.norm(arr_j)
        if norm_product == 0:
            return 0.0
        return float(np.dot(arr_i, arr_j) / norm_product)

    def _nearest_subject(self, token) -> str:
        for node in [token, *token.ancestors]:
            for child in node.children:
                if child.dep_ in {"nsubj", "nsubjpass"}:
                    return child.lemma_.lower()
        return ""

    def _looks_like_event_verb(self, token: str) -> bool:
        stem = re.sub(r"(ing|ed|es|s)$", "", token)
        return token in COMMON_EVENT_VERBS or stem in COMMON_EVENT_VERBS

    def _content_words(self, text: str, doc=None) -> List[str]:
        if doc is not None:
            return [
                token.lemma_.lower()
                for token in doc
                if token.pos_ in {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
                and not token.is_stop
                and not token.is_punct
                and len(token.text) > 2
            ]
        return [word for word in self._words(text) if len(word) > 2 and word not in STOPWORDS]

    def _words(self, text: str) -> List[str]:
        return [word.lower() for word in WORD_RE.findall(text)]

    def _normalize_text(self, text: str) -> str:
        return " ".join(self._words(text))

    def _windows(self, values: Sequence[str], size: int) -> Iterable[Tuple[str, ...]]:
        for index in range(len(values) - size + 1):
            yield tuple(values[index:index + size])
