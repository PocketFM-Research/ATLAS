"""High-precision claim extraction for graph-grounded evaluation."""

import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import spacy


LOCATION_PREPS = {"in", "at", "into", "onto", "inside", "outside", "near", "on", "within"}
LOCATION_NOUN_HINTS = {
    "bridge",
    "room",
    "corridor",
    "console",
    "station",
    "dock",
    "bay",
    "zone",
    "ship",
    "enterprise",
    "lab",
    "elevator",
    "planet",
    "space",
    "home",
    "deck",
    "helm",
    "post",
    "simulator",
    "orbit",
    "castle",
    "transporter",
}
TIME_WORDS = {
    "day",
    "night",
    "morning",
    "evening",
    "afternoon",
    "today",
    "tomorrow",
    "yesterday",
    "stardate",
    "year",
    "minute",
}
STATE_ADJ_HINTS = {
    "new",
    "unexpected",
    "young",
    "beautiful",
    "fair",
    "angry",
    "sad",
    "worried",
    "nervous",
    "ready",
    "curious",
    "afraid",
    "troubled",
    "dead",
    "alive",
    "hurt",
    "injured",
    "laconic",
}
BE_FORMS = {"be", "am", "is", "are", "was", "were"}
POSSESS_FORMS = {"have", "has", "had"}
SELF_REFERENTIAL_PRONOUNS = {"i", "me", "myself"}
ANAPHORIC_PRONOUNS = {"he", "she", "him", "her"}
UNRESOLVABLE_PRONOUNS = {
    "it", "this", "that", "these", "those", "you", "we", "us", "they", "them",
    "mine", "yours", "ours", "theirs", "hers", "his",
}
SUBJECT_NAME_DEPS = {"compound", "flat", "fixed", "amod", "nmod", "appos"}
NON_INFORMATIVE_OBJECTS = {"scene", "scenes", "something", "anything"}
TITLE_TOKENS = {"the", "captain", "commander", "lt", "lt.", "mr", "mr.", "mister", "dr", "dr.", "doctor"}
GENERIC_SUBJECTS = {"captain", "the captain", "commander", "doctor", "dr", "dr.", "mister", "mr", "mr.", "sir"}
NON_FACTUAL_ROOTS = {
    "say",
    "tell",
    "ask",
    "reply",
    "continue",
    "damn",
    "hello",
    "hi",
    "thanks",
    "thank",
    "remind",
}
WEAK_FACTUAL_VERBS = {
    "assume",
    "suppose",
    "guess",
    "wonder",
    "hope",
    "wish",
    "mean",
    "try",
}
WEAK_ACTION_PHRASES = {
    "do his best",
    "do her best",
    "do my best",
    "do our best",
    "do their best",
}
STATE_CHANGE_VERBS = {"die", "fall", "hit", "hurt", "injure"}
EXPERIENCE_VERBS = {"feel", "worry", "fear", "smile", "grin"}
SPEAKER_PREFIX_RE = re.compile(r"^(?P<label>[A-Z][A-Z0-9 .'\-]{0,40}):\s*(?P<content>.*)$")
UPPERCASE_SPEAKER_RE = re.compile(
    r"^(?P<label>[A-Z][A-Z0-9.'\-]*(?:\s+[A-Z][A-Z0-9.'\-]*){0,3})\s+(?P<content>.+)$"
)
SIMPLE_NAME_ACTION_RE = re.compile(
    r"^(?P<subject>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s+"
    r"(?P<verb1>[a-z]+s)(?:\s+and\s+(?P<verb2>[a-z]+s))?\.?$"
)
NAME_PHRASE_RE = re.compile(r"^[A-Z][A-Za-z.'\-]*(?:\s+[A-Z][A-Za-z.'\-]*){0,4}$")
LEADING_IS_SUBJECT_RE = re.compile(
    r"^.*?\bis\s+(?P<subject>[A-Z][A-Z.'\-]+(?:\s+[A-Z][A-Z.'\-]+){0,4}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s*,\s*(?P<rest>.+)$"
)
LEADING_STAGE_DIRECTION_RE = re.compile(r"^(?:\([^)]+\)\s*)+")
LEADING_VOCATIVE_RE = re.compile(
    r"^(?:Captain|Commander|Doctor|Dr\.?|Mister|Mr\.?|Lieutenant|Lt\.?|Jim|Spock|Bones|Saavik|Sulu|Uhura|Kirk)"
    r"(?:\s+[A-Z][a-z]+)?(?:\s*\.\.\.|\s*,)\s+",
    re.IGNORECASE,
)
TRAILING_ADDRESS_TOKENS = {
    "jim",
    "spock",
    "captain",
    "mister",
    "sir",
    "doctor",
    "bones",
    "sulu",
    "uhura",
    "kirk",
    "saavik",
}
POSSESSIVE_PRONOUNS = {"his", "her", "their", "its", "my", "our", "your"}


@dataclass
class Claim:
    """Atomic extracted claim."""

    subject: str
    predicate: str
    object: str
    claim_text: str
    scene_id: str
    character_id: str
    sentence_idx: int

    def to_dict(self):
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "claim_text": self.claim_text,
            "scene_id": self.scene_id,
            "character_id": self.character_id,
            "sentence_idx": self.sentence_idx,
        }


@dataclass
class ExtractionAbstention:
    """Sentence-level abstention for ambiguous or low-confidence text."""

    text: str
    scene_id: str
    character_id: str
    sentence_idx: int
    reason: str
    speaker: str = ""
    context_subject: str = ""

    def to_dict(self):
        return {
            "text": self.text,
            "scene_id": self.scene_id,
            "character_id": self.character_id,
            "sentence_idx": self.sentence_idx,
            "reason": self.reason,
            "speaker": self.speaker,
            "context_subject": self.context_subject,
        }


@dataclass
class LowConfidenceClaim:
    """Claim candidate extracted from text but held out from scoring."""

    subject: str
    predicate: str
    object: str
    claim_text: str
    scene_id: str
    character_id: str
    sentence_idx: int
    score: float
    reasons: List[str]

    def to_dict(self):
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "claim_text": self.claim_text,
            "scene_id": self.scene_id,
            "character_id": self.character_id,
            "sentence_idx": self.sentence_idx,
            "score": self.score,
            "reasons": ";".join(self.reasons),
        }


@dataclass
class PreparedLine:
    """Normalized screenplay line with optional speaker context."""

    text: str
    raw_text: str
    speaker_subject: str = ""


class ClaimExtractor:
    """Extract atomic claims while abstaining on weak screenplay lines."""

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

    def extract_claims(
        self,
        text: str,
        scene_id: str,
        character_id: str,
    ) -> List[Claim]:
        claims, _ = self.extract_claims_with_abstentions(text, scene_id, character_id)
        return claims

    def extract_claims_with_quality(
        self,
        text: str,
        scene_id: str,
        character_id: str,
    ) -> Tuple[List[Claim], List[ExtractionAbstention], List[LowConfidenceClaim]]:
        """Extract kept claims plus abstained and low-confidence candidates."""
        if not self.nlp:
            return self._extract_claims_regex(text, scene_id, character_id), [], []

        claims: List[Claim] = []
        abstentions: List[ExtractionAbstention] = []
        low_confidence_claims: List[LowConfidenceClaim] = []
        seen: Set[Tuple[str, str, str, int]] = set()
        seen_low_confidence: Set[Tuple[str, str, str, int]] = set()
        sentence_idx = 0

        for line in self._prepare_lines(text):
            if not line.text:
                continue
            antecedent_subject = ""

            doc = self.nlp(line.text)
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if not sent_text:
                    continue

                sent_claims, abstain_reason = self._extract_from_sent(
                    sent=sent,
                    sent_idx=sentence_idx,
                    scene_id=scene_id,
                    character_id=character_id,
                    speaker_subject=line.speaker_subject,
                    antecedent_subject=antecedent_subject,
                )

                kept_any = False
                low_confidence_any = False
                for claim in sent_claims:
                    if self._should_skip_claim(claim):
                        continue

                    decision, score, reasons = self._assess_claim_quality(
                        claim=claim,
                        speaker_subject=line.speaker_subject,
                        sentence_text=sent_text,
                    )
                    if decision == "abstain":
                        continue
                    if decision == "low_confidence":
                        key = (claim.subject, claim.predicate, claim.object, claim.sentence_idx)
                        if key in seen_low_confidence:
                            continue
                        seen_low_confidence.add(key)
                        low_confidence_claims.append(
                            LowConfidenceClaim(
                                subject=claim.subject,
                                predicate=claim.predicate,
                                object=claim.object,
                                claim_text=claim.claim_text,
                                scene_id=claim.scene_id,
                                character_id=claim.character_id,
                                sentence_idx=claim.sentence_idx,
                                score=score,
                                reasons=reasons,
                            )
                        )
                        low_confidence_any = True
                        continue

                    key = (claim.subject, claim.predicate, claim.object)
                    if key in seen:
                        continue

                    seen.add(key)
                    claims.append(claim)
                    kept_any = True

                    if claim.subject and not claim.subject.startswith("ent_"):
                        antecedent_subject = claim.subject
                    elif claim.subject == character_id and line.speaker_subject:
                        antecedent_subject = line.speaker_subject

                if not kept_any:
                    reason = abstain_reason or "no_high_confidence_claim"
                    if low_confidence_any:
                        reason = "only_low_confidence_claims"
                    abstentions.append(
                        ExtractionAbstention(
                            text=sent_text,
                            scene_id=scene_id,
                            character_id=character_id,
                            sentence_idx=sentence_idx,
                            reason=reason,
                            speaker=line.speaker_subject,
                            context_subject=antecedent_subject,
                        )
                    )

                sentence_idx += 1

        return claims, abstentions, low_confidence_claims

    def extract_claims_with_abstentions(
        self,
        text: str,
        scene_id: str,
        character_id: str,
    ) -> Tuple[List[Claim], List[ExtractionAbstention]]:
        """Extract high-confidence claims and explicitly abstain on weak sentences."""
        claims, abstentions, _ = self.extract_claims_with_quality(text, scene_id, character_id)
        return claims, abstentions

    def _prepare_lines(self, text: str) -> List[PreparedLine]:
        lines: List[PreparedLine] = []

        for raw_line in text.replace("\r", "\n").split("\n"):
            line = raw_line.strip()
            if not line:
                continue

            speaker, content = self._split_speaker_prefix(line)
            content = LEADING_STAGE_DIRECTION_RE.sub("", content).strip()
            content = self._strip_leading_vocative(content)
            content = re.sub(r"\s+", " ", content).strip()

            if not content:
                continue

            lines.append(
                PreparedLine(
                    text=content,
                    raw_text=line,
                    speaker_subject=self._normalize_speaker_subject(speaker) if speaker else "",
                )
            )

        return lines

    def _split_speaker_prefix(self, line: str) -> Tuple[str, str]:
        colon_match = SPEAKER_PREFIX_RE.match(line)
        if colon_match:
            return colon_match.group("label").strip(), colon_match.group("content").strip()

        upper_match = UPPERCASE_SPEAKER_RE.match(line)
        if upper_match:
            label = upper_match.group("label").strip()
            content = upper_match.group("content").strip()
            if label == label.upper() and any(char.islower() for char in content):
                label_tokens = label.split()
                if label_tokens and label_tokens[-1] in {"I", "WE", "YOU"}:
                    moved = label_tokens.pop()
                    label = " ".join(label_tokens)
                    content = f"{moved} {content}".strip()
                return label, content

        return "", line

    def _strip_leading_vocative(self, line: str) -> str:
        if not LEADING_VOCATIVE_RE.match(line):
            return line

        stripped = LEADING_VOCATIVE_RE.sub("", line, count=1).strip()
        return stripped or line

    def _extract_from_sent(
        self,
        sent,
        sent_idx: int,
        scene_id: str,
        character_id: str,
        speaker_subject: str,
        antecedent_subject: str,
    ) -> Tuple[List[Claim], Optional[str]]:
        claims: List[Claim] = []
        text = sent.text.strip()
        if not text:
            return claims, "empty_sentence"
        if not any(char.isalpha() for char in text):
            return claims, "non_text_fragment"
        if text.endswith("?"):
            return claims, "question_or_prompt"

        root = sent.root
        if self._is_speaker_location_fragment(text, speaker_subject):
            obj = text
            if text.split()[0].lower() in LOCATION_PREPS and self._looks_like_location(text):
                obj = " ".join(text.split()[1:]).lower()
                claims.append(
                    Claim(
                        subject=speaker_subject,
                        predicate="located_at",
                        object=obj,
                        claim_text=text,
                        scene_id=scene_id,
                        character_id=character_id,
                        sentence_idx=sent_idx,
                    )
                )
                return claims, None

        if root.lemma_.lower() in NON_FACTUAL_ROOTS:
            return claims, "dialogue_act_or_interjection"
        if self._looks_like_imperative(sent, speaker_subject):
            return claims, "imperative_or_command"

        candidate_predicates = self._candidate_predicates(sent)
        for predicate in candidate_predicates:
            predicate_lemma = predicate.lemma_.lower()

            if predicate_lemma in WEAK_FACTUAL_VERBS:
                continue

            subjects = self._extract_subjects(
                predicate=predicate,
                sent=sent,
                character_id=character_id,
                speaker_subject=speaker_subject,
                antecedent_subject=antecedent_subject,
            )

            has_explicit_subject = any(
                token.head == predicate and token.dep_ in {"nsubj", "nsubjpass", "expl", "csubj"}
                for token in sent
            )
            if not subjects and predicate_lemma in BE_FORMS and speaker_subject and not has_explicit_subject:
                subjects = [speaker_subject]

            claims.extend(
                self._extract_prep_claims(
                    predicate=predicate,
                    sent=sent,
                    sent_idx=sent_idx,
                    scene_id=scene_id,
                    character_id=character_id,
                    subjects=subjects,
                )
            )

            if predicate_lemma in BE_FORMS:
                claims.extend(
                    self._extract_be_claims(
                        predicate=predicate,
                        sent=sent,
                        sent_idx=sent_idx,
                        scene_id=scene_id,
                        character_id=character_id,
                        subjects=subjects,
                    )
                )
            elif predicate_lemma in POSSESS_FORMS:
                claims.extend(
                    self._extract_possess_claims(
                        predicate=predicate,
                        sent=sent,
                        sent_idx=sent_idx,
                        scene_id=scene_id,
                        character_id=character_id,
                        subjects=subjects,
                    )
                )
            elif subjects:
                claims.extend(
                    self._extract_action_claims(
                        predicate=predicate,
                        sent=sent,
                        sent_idx=sent_idx,
                        scene_id=scene_id,
                        character_id=character_id,
                        subjects=subjects,
                    )
                )

        if claims:
            descriptor_claims = self._extract_intro_descriptor_fallback(
                text=text,
                sent_idx=sent_idx,
                scene_id=scene_id,
                character_id=character_id,
                speaker_subject=speaker_subject,
                antecedent_subject=antecedent_subject,
            )
            if descriptor_claims:
                claims.extend(descriptor_claims)
            return claims, None

        speaker_fragment_claims = self._extract_speaker_fragment_fallback(
            sent=sent,
            sent_idx=sent_idx,
            scene_id=scene_id,
            character_id=character_id,
            speaker_subject=speaker_subject,
        )
        if speaker_fragment_claims:
            return speaker_fragment_claims, None

        fallback_claims = self._extract_simple_name_action_fallback(
            text=text,
            sent_idx=sent_idx,
            scene_id=scene_id,
            character_id=character_id,
            speaker_subject=speaker_subject,
            antecedent_subject=antecedent_subject,
        )
        if fallback_claims:
            return fallback_claims, None

        descriptor_claims = self._extract_intro_descriptor_fallback(
            text=text,
            sent_idx=sent_idx,
            scene_id=scene_id,
            character_id=character_id,
            speaker_subject=speaker_subject,
            antecedent_subject=antecedent_subject,
        )
        if descriptor_claims:
            return descriptor_claims, None

        if root.pos_ in {"PROPN", "NOUN", "ADP"}:
            fragment_claims = self._extract_fragment_claims(
                root=root,
                sent=sent,
                sent_idx=sent_idx,
                scene_id=scene_id,
                character_id=character_id,
                speaker_subject=speaker_subject,
                antecedent_subject=antecedent_subject,
            )
            if fragment_claims:
                return fragment_claims, None

        return claims, "no_high_confidence_claim"

    def _extract_speaker_fragment_fallback(
        self,
        sent,
        sent_idx: int,
        scene_id: str,
        character_id: str,
        speaker_subject: str,
    ) -> List[Claim]:
        if not speaker_subject:
            return []

        root = sent.root
        text = sent.text.strip()
        claims: List[Claim] = []

        nominal_subjects = [
            token for token in sent
            if token.head == root and token.dep_ in {"nsubj", "nsubjpass"} and token.pos_ in {"NOUN", "PROPN"}
        ]

        if root.pos_ == "VERB" and nominal_subjects:
            subject_phrase = self._get_token_phrase(nominal_subjects[0])
            if subject_phrase:
                relation = "undergoes" if root.tag_ == "VBN" else "performs"
                claims.append(
                    Claim(
                        subject=speaker_subject,
                        predicate=relation,
                        object=f"{root.lemma_.lower()} {subject_phrase}",
                        claim_text=text,
                        scene_id=scene_id,
                        character_id=character_id,
                        sentence_idx=sent_idx,
                    )
                )
                return claims

        if root.pos_ == "NOUN" and text.lower().startswith("no "):
            cleaned = self._strip_trailing_address(text.strip(" .!?")).lower().strip(" ,")
            claims.append(
                Claim(
                    subject=speaker_subject,
                    predicate="experiences",
                    object=cleaned,
                    claim_text=text,
                    scene_id=scene_id,
                    character_id=character_id,
                    sentence_idx=sent_idx,
                )
            )
            return claims

        if root.lemma_.lower() == "none":
            be_token = next(
                (token for token in sent if token.lemma_.lower() == "be" and token.pos_ == "AUX"),
                None,
            )
            acomp_token = next(
                (token for token in sent if token.head == be_token and token.dep_ in {"acomp", "attr"})
                if be_token is not None
                else None,
                None,
            )
            if acomp_token is not None:
                acomp_phrase = self._get_token_phrase(acomp_token)
                if acomp_phrase:
                    claims.append(
                        Claim(
                            subject=speaker_subject,
                            predicate="experiences",
                            object=acomp_phrase,
                            claim_text=text,
                            scene_id=scene_id,
                            character_id=character_id,
                            sentence_idx=sent_idx,
                        )
                    )
            return claims

        return claims

    def _extract_simple_name_action_fallback(
        self,
        text: str,
        sent_idx: int,
        scene_id: str,
        character_id: str,
        speaker_subject: str,
        antecedent_subject: str,
    ) -> List[Claim]:
        match = SIMPLE_NAME_ACTION_RE.match(text.strip())
        if not match:
            return []

        subject = self._normalize_subject(
            match.group("subject"),
            character_id=character_id,
            speaker_subject=speaker_subject,
            antecedent_subject=antecedent_subject,
        )
        if not subject:
            return []

        verbs = [match.group("verb1")]
        if match.group("verb2"):
            verbs.append(match.group("verb2"))

        claims: List[Claim] = []
        for verb in verbs:
            lemma = self._heuristic_lemmatize_present(verb)
            if not lemma or lemma in NON_FACTUAL_ROOTS or lemma in WEAK_FACTUAL_VERBS:
                continue
            claims.append(
                Claim(
                    subject=subject,
                    predicate="performs",
                    object=lemma,
                    claim_text=text.strip(),
                    scene_id=scene_id,
                    character_id=character_id,
                    sentence_idx=sent_idx,
                )
            )

        return claims

    def _extract_intro_descriptor_fallback(
        self,
        text: str,
        sent_idx: int,
        scene_id: str,
        character_id: str,
        speaker_subject: str,
        antecedent_subject: str,
    ) -> List[Claim]:
        stripped = text.strip()
        if not stripped:
            return []

        subject_text = ""
        remainder = ""

        leading_is_match = LEADING_IS_SUBJECT_RE.match(stripped)
        if leading_is_match:
            subject_text = leading_is_match.group("subject")
            remainder = leading_is_match.group("rest")
        elif "," in stripped:
            possible_subject, possible_rest = stripped.split(",", 1)
            if (
                self._looks_like_name_phrase(possible_subject.strip())
                and any(char.isalpha() for char in possible_rest)
            ):
                subject_text = possible_subject
                remainder = possible_rest

        if not subject_text or not remainder:
            return []

        subject = self._normalize_subject(
            subject_text,
            character_id=character_id,
            speaker_subject=speaker_subject,
            antecedent_subject=antecedent_subject,
        )
        if not subject:
            return []

        descriptors: List[str] = []
        candidate_chunks = re.split(r"\s*,\s*|\s+but\s+|\s+and\s+", remainder)
        for chunk in candidate_chunks:
            descriptor = self._clean_descriptor(chunk)
            if not descriptor:
                continue
            descriptors.append(descriptor)

        claims: List[Claim] = []
        seen = set()
        for descriptor in descriptors:
            if descriptor in seen:
                continue
            seen.add(descriptor)
            claims.append(
                Claim(
                    subject=subject,
                    predicate="experiences",
                    object=descriptor,
                    claim_text=stripped,
                    scene_id=scene_id,
                    character_id=character_id,
                    sentence_idx=sent_idx,
                )
            )

        return claims

    def _candidate_predicates(self, sent) -> List:
        predicates = []
        for token in sent:
            if token.pos_ not in {"VERB", "AUX"}:
                continue
            if token.dep_ not in {"ROOT", "conj"}:
                continue
            predicates.append(token)

        if sent.root.pos_ in {"VERB", "AUX"} and sent.root not in predicates:
            predicates.insert(0, sent.root)

        predicates = sorted({token.i: token for token in predicates}.values(), key=lambda token: token.i)
        return predicates

    def _extract_fragment_claims(
        self,
        root,
        sent,
        sent_idx: int,
        scene_id: str,
        character_id: str,
        speaker_subject: str,
        antecedent_subject: str,
    ) -> List[Claim]:
        claims: List[Claim] = []
        text = sent.text.strip()

        if speaker_subject and text.split()[0].lower() in LOCATION_PREPS and self._looks_like_location(text):
            obj = text.lower()
            if text.split()[0].lower() in LOCATION_PREPS:
                obj = " ".join(text.split()[1:]).lower()
            claims.append(
                Claim(
                    subject=speaker_subject,
                    predicate="located_at",
                    object=obj,
                    claim_text=text,
                    scene_id=scene_id,
                    character_id=character_id,
                    sentence_idx=sent_idx,
                )
            )
            return claims

        subject = self._normalize_subject(
            self._get_subject_phrase(root),
            character_id=character_id,
            speaker_subject=speaker_subject,
            antecedent_subject=antecedent_subject,
        )
        if not subject:
            return claims

        for token in sent:
            if token.head != root or token.dep_ != "prep":
                continue
            obj_token = next((child for child in token.children if child.dep_ == "pobj"), None)
            if obj_token is None:
                continue
            obj_phrase = self._get_token_phrase(obj_token)
            if not obj_phrase:
                continue
            prep = token.text.lower()
            if prep == "on" and self._looks_like_time(obj_phrase):
                predicate = "present_on"
            elif prep in LOCATION_PREPS and self._looks_like_location(obj_phrase):
                predicate = "located_at"
            else:
                continue
            claims.append(
                Claim(
                    subject=subject,
                    predicate=predicate,
                    object=obj_phrase,
                    claim_text=text,
                    scene_id=scene_id,
                    character_id=character_id,
                    sentence_idx=sent_idx,
                )
            )

        return claims

    def _extract_subjects(
        self,
        predicate,
        sent,
        character_id: str,
        speaker_subject: str,
        antecedent_subject: str,
    ) -> List[str]:
        subject_tokens = [
            token for token in sent
            if token.head == predicate and token.dep_ in {"nsubj", "nsubjpass", "expl"}
        ]

        if not subject_tokens and predicate.dep_ in {"conj", "xcomp", "advcl", "ccomp"} and predicate.head != predicate:
            subject_tokens = [
                token for token in sent
                if token.head == predicate.head and token.dep_ in {"nsubj", "nsubjpass", "expl"}
            ]

        subjects = []
        for token in subject_tokens:
            subject = self._normalize_subject(
                self._get_subject_phrase(token),
                character_id=character_id,
                speaker_subject=speaker_subject,
                antecedent_subject=antecedent_subject,
            )
            if subject:
                subjects.append(subject)

        if not subjects and speaker_subject and any(token.text.lower() in SELF_REFERENTIAL_PRONOUNS for token in sent):
            subjects.append(speaker_subject)
        if not subjects and speaker_subject and any(token.text.lower() in {"we", "us"} for token in sent):
            if predicate.lemma_.lower() not in BE_FORMS:
                subjects.append(speaker_subject)
        if not subjects and speaker_subject and predicate.dep_ == "ROOT" and predicate.tag_ == "VBG":
            subjects.append(speaker_subject)

        return list(dict.fromkeys(subjects))

    def _extract_be_claims(
        self,
        predicate,
        sent,
        sent_idx: int,
        scene_id: str,
        character_id: str,
        subjects: List[str],
    ) -> List[Claim]:
        claims: List[Claim] = []
        attrs = []
        for token in sent:
            if token.head == predicate and token.dep_ in {"attr", "acomp", "oprd"}:
                for attr_token, phrase in self._expand_attribute_phrases(token):
                    if phrase:
                        attrs.append((attr_token, phrase))

        for subj in subjects:
            for attr_token, attr in attrs:
                claims.append(
                    Claim(
                        subject=subj,
                        predicate=self._classify_copula_relation(attr_token, attr),
                        object=attr,
                        claim_text=sent.text.strip(),
                        scene_id=scene_id,
                        character_id=character_id,
                        sentence_idx=sent_idx,
                    )
                )

        return claims

    def _extract_possess_claims(
        self,
        predicate,
        sent,
        sent_idx: int,
        scene_id: str,
        character_id: str,
        subjects: List[str],
    ) -> List[Claim]:
        claims: List[Claim] = []
        objects = [
            self._get_token_phrase(token)
            for token in sent
            if token.head == predicate and token.dep_ in {"dobj", "obj", "attr"}
        ]
        objects = [obj for obj in objects if obj]

        for subj in subjects:
            for obj in objects:
                claims.append(
                    Claim(
                        subject=subj,
                        predicate="possesses",
                        object=obj,
                        claim_text=sent.text.strip(),
                        scene_id=scene_id,
                        character_id=character_id,
                        sentence_idx=sent_idx,
                    )
                )

        return claims

    def _extract_action_claims(
        self,
        predicate,
        sent,
        sent_idx: int,
        scene_id: str,
        character_id: str,
        subjects: List[str],
    ) -> List[Claim]:
        claims: List[Claim] = []
        action_phrase = self._build_action_phrase(predicate)
        if not action_phrase or action_phrase in NON_INFORMATIVE_OBJECTS:
            return claims
        if action_phrase in WEAK_ACTION_PHRASES:
            return claims

        is_passive = any(token.head == predicate and token.dep_ in {"nsubjpass", "auxpass"} for token in sent)
        if predicate.lemma_.lower() in STATE_CHANGE_VERBS:
            relation = "undergoes"
        elif predicate.lemma_.lower() in EXPERIENCE_VERBS:
            relation = "experiences"
        elif is_passive:
            relation = "undergoes"
        else:
            relation = "performs"

        for subj in subjects:
            claims.append(
                Claim(
                    subject=subj,
                    predicate=relation,
                    object=action_phrase,
                    claim_text=sent.text.strip(),
                    scene_id=scene_id,
                    character_id=character_id,
                    sentence_idx=sent_idx,
                )
            )

        return claims

    def _extract_prep_claims(
        self,
        predicate,
        sent,
        sent_idx: int,
        scene_id: str,
        character_id: str,
        subjects: List[str],
    ) -> List[Claim]:
        claims: List[Claim] = []

        for token in sent:
            if token.head != predicate or token.dep_ != "prep":
                continue

            prep = token.text.lower()
            obj_token = next((child for child in token.children if child.dep_ == "pobj"), None)
            if obj_token is None:
                continue

            obj_phrase = self._get_token_phrase(obj_token)
            if not obj_phrase or obj_phrase in NON_INFORMATIVE_OBJECTS:
                continue

            if prep == "on" and self._looks_like_time(obj_phrase):
                relation = "present_on"
            elif prep in LOCATION_PREPS and self._looks_like_location(obj_phrase):
                relation = "located_at"
            else:
                continue

            for subj in subjects:
                claims.append(
                    Claim(
                        subject=subj,
                        predicate=relation,
                        object=obj_phrase,
                        claim_text=sent.text.strip(),
                        scene_id=scene_id,
                        character_id=character_id,
                        sentence_idx=sent_idx,
                    )
                )

        return claims

    def _build_action_phrase(self, predicate) -> str:
        particle = next(
            (child.text.lower() for child in predicate.children if child.dep_ == "prt"),
            "",
        )
        verb_phrase = predicate.lemma_.lower()
        if particle:
            verb_phrase = f"{verb_phrase} {particle}"

        pieces = [verb_phrase]

        direct_objects = []
        for child in predicate.children:
            if child.dep_ not in {"dobj", "obj", "attr", "oprd"}:
                continue
            phrase = self._get_token_phrase(child)
            if phrase:
                direct_objects.append(phrase)

        if direct_objects:
            pieces.append(direct_objects[0])

        for child in predicate.children:
            if child.dep_ != "prep":
                continue
            obj_token = next((grand for grand in child.children if grand.dep_ == "pobj"), None)
            if obj_token is None:
                continue
            obj_phrase = self._get_token_phrase(obj_token)
            if not obj_phrase:
                continue
            if child.text.lower() in LOCATION_PREPS and self._looks_like_location(obj_phrase):
                continue
            pieces.append(f"{child.text.lower()} {obj_phrase}")
            break

        for child in predicate.children:
            if child.dep_ not in {"xcomp", "ccomp"} or child.pos_ != "VERB":
                continue
            xcomp_phrase = child.lemma_.lower()
            for xcomp_child in child.children:
                if xcomp_child.dep_ in {"dobj", "obj"}:
                    obj_phrase = self._get_token_phrase(xcomp_child)
                    if obj_phrase:
                        xcomp_phrase = f"{xcomp_phrase} {obj_phrase}"
                        break
            pieces.append(f"to {xcomp_phrase}")
            break

        phrase = " ".join(part for part in pieces if part).strip()
        phrase = re.sub(r"\s+", " ", phrase)
        return phrase

    def _get_token_phrase(self, token) -> str:
        tokens = [
            tok for tok in token.subtree
            if not tok.is_space and not tok.is_punct and tok.dep_ != "punct"
        ]
        if not tokens:
            return ""

        tokens = sorted(tokens, key=lambda tok: tok.i)
        phrase = " ".join(tok.text for tok in tokens)
        phrase = re.sub(r"\s+", " ", phrase).strip(" -,:;.!?")
        phrase = self._strip_trailing_address(phrase)
        return phrase.lower()

    def _get_subject_phrase(self, token) -> str:
        if token.pos_ == "PRON":
            return token.text.lower()

        phrase_tokens = [token]
        for child in token.lefts:
            if child.dep_ in SUBJECT_NAME_DEPS and child.pos_ in {"PROPN", "NOUN", "ADJ"}:
                phrase_tokens.append(child)
                for desc in child.subtree:
                    if desc.dep_ != "punct" and desc.i != token.i:
                        phrase_tokens.append(desc)

        phrase_tokens = sorted({tok.i: tok for tok in phrase_tokens}.values(), key=lambda tok: tok.i)
        phrase = " ".join(
            tok.text
            for tok in phrase_tokens
            if not tok.is_space and not tok.is_punct and tok.dep_ != "punct"
        )
        phrase = re.sub(r"\s+", " ", phrase).strip(" -,:;.!?")
        return phrase.lower()

    def _expand_attribute_phrases(self, token):
        phrases = []
        seen = set()
        primary_phrase = self._get_token_phrase(token)
        if primary_phrase:
            seen.add(primary_phrase)
            phrases.append((token, primary_phrase))

        if primary_phrase and (" and " in primary_phrase or " or " in primary_phrase):
            return phrases

        for candidate in token.children:
            if candidate.dep_ != "conj":
                continue
            phrase = self._get_token_phrase(candidate)
            if not phrase or phrase in seen:
                continue
            seen.add(phrase)
            phrases.append((candidate, phrase))

        return phrases

    def _normalize_subject(
        self,
        subject: str,
        character_id: str,
        speaker_subject: str,
        antecedent_subject: str,
    ) -> str:
        subject = subject.strip().lower()
        if not subject:
            return ""
        if subject in SELF_REFERENTIAL_PRONOUNS:
            return speaker_subject or character_id
        if subject in ANAPHORIC_PRONOUNS:
            return antecedent_subject if antecedent_subject else ""
        if subject in UNRESOLVABLE_PRONOUNS:
            return ""
        subject_tokens = [token for token in subject.split() if token]
        while subject_tokens and subject_tokens[0] in TITLE_TOKENS:
            subject_tokens.pop(0)
        normalized = " ".join(subject_tokens).strip() or subject
        if normalized in GENERIC_SUBJECTS:
            return ""
        return normalized

    def _looks_like_time(self, text: str) -> bool:
        lowered = text.lower()
        return any(word in lowered for word in TIME_WORDS)

    def _looks_like_location(self, text: str) -> bool:
        lowered = text.lower()
        return any(hint in lowered for hint in LOCATION_NOUN_HINTS)

    def _classify_copula_relation(self, attr_token, attr_phrase: str) -> str:
        if self._looks_like_time(attr_phrase):
            return "present_on"
        if self._looks_like_location(attr_phrase):
            return "located_at"
        if self._is_state_description(attr_token, attr_phrase):
            return "experiences"
        return "is_a"

    def _is_state_description(self, attr_token, attr_phrase: str) -> bool:
        if attr_token.pos_ == "ADJ":
            return True
        lowered = attr_phrase.lower()
        return any(hint in lowered for hint in STATE_ADJ_HINTS)

    def _looks_like_imperative(self, sent, speaker_subject: str) -> bool:
        if not speaker_subject:
            return False
        root = sent.root
        has_subject = any(token.head == root and token.dep_ in {"nsubj", "nsubjpass"} for token in sent)
        if has_subject:
            return False
        if sent.text.strip().endswith("?"):
            return True
        if root.pos_ in {"NOUN", "PROPN"} and sent.text.strip().endswith("!"):
            return True
        if root.pos_ == "VERB" and root.tag_ == "VB":
            return True
        return False

    def _is_speaker_location_fragment(self, text: str, speaker_subject: str) -> bool:
        if not speaker_subject:
            return False
        if not text:
            return False
        first = text.split()[0].lower()
        return first in LOCATION_PREPS and self._looks_like_location(text)

    def _should_skip_claim(self, claim: Claim) -> bool:
        if not claim.subject or not claim.object:
            return True

        subject = claim.subject.lower()
        obj = claim.object.lower()
        if subject in {"none", "someone", "somebody"}:
            return True
        if "voice" in subject.split():
            return True
        if obj in NON_INFORMATIVE_OBJECTS:
            return True
        if len(subject.split()) > 4 or len(obj.split()) > 10:
            return True
        if not any(char.isalpha() for char in subject):
            return True
        if claim.predicate == "located_at" and not self._looks_like_location(obj):
            return True
        return False

    def _assess_claim_quality(
        self,
        claim: Claim,
        speaker_subject: str,
        sentence_text: str,
    ) -> Tuple[str, float, List[str]]:
        """Classify an extracted claim as keep, low_confidence, or abstain."""
        score = 0.6
        reasons: List[str] = []

        subject = claim.subject.lower().strip()
        obj = claim.object.lower().strip()
        subject_tokens = subject.split()
        object_tokens = obj.split()

        if speaker_subject and subject == speaker_subject:
            score += 0.1
            reasons.append("speaker_aligned")

        if len(subject_tokens) == 1:
            score += 0.05
        elif len(subject_tokens) > 4:
            score -= 0.2
            reasons.append("long_subject")

        if len(object_tokens) > 8:
            score -= 0.15
            reasons.append("long_object")
        elif 1 <= len(object_tokens) <= 5:
            score += 0.05

        if any(token in POSSESSIVE_PRONOUNS for token in object_tokens[:2]):
            score -= 0.2
            reasons.append("possessive_object")

        if any(token in TRAILING_ADDRESS_TOKENS for token in object_tokens[-1:]):
            score -= 0.15
            reasons.append("trailing_address")

        if "," in obj or ";" in obj:
            score -= 0.1
            reasons.append("clausal_object")

        if claim.predicate == "located_at":
            if self._looks_like_location(obj):
                score += 0.15
            else:
                score -= 0.3
                reasons.append("weak_location")
            if " and " in sentence_text.lower() and any(token in POSSESSIVE_PRONOUNS for token in object_tokens[:2]):
                score -= 0.25
                reasons.append("coordinated_possessive_location")
        elif claim.predicate == "experiences":
            if obj.startswith("no ") or self._is_state_phrase(obj):
                score += 0.1
            else:
                score -= 0.05
                reasons.append("weak_state_phrase")
        elif claim.predicate == "is_a":
            if len(object_tokens) > 6:
                score -= 0.15
                reasons.append("long_is_a_phrase")
        elif claim.predicate in {"performs", "undergoes"}:
            if object_tokens and object_tokens[0] in WEAK_FACTUAL_VERBS:
                score -= 0.25
                reasons.append("weak_action_head")
            if len(object_tokens) == 1 and object_tokens[0] in {"project", "activate"}:
                score += 0.05

        if subject in {"data", "message", "signal"}:
            score -= 0.2
            reasons.append("underspecified_subject")

        if subject in GENERIC_SUBJECTS or not any(char.isalpha() for char in subject):
            return "abstain", 0.0, reasons + ["invalid_subject"]

        if score >= 0.55:
            return "keep", score, reasons
        if score >= 0.35:
            return "low_confidence", score, reasons or ["borderline_quality"]
        return "abstain", score, reasons or ["low_quality_claim"]

    def _extract_claims_regex(
        self,
        text: str,
        scene_id: str,
        character_id: str,
    ) -> List[Claim]:
        claims: List[Claim] = []
        sentences = re.split(r"[.!?]+", text)

        for sent_idx, sent in enumerate(sentences):
            sent = sent.strip()
            if not sent:
                continue

            words = sent.split()
            if len(words) < 2:
                continue

            claims.append(
                Claim(
                    subject=words[0].lower(),
                    predicate="performs",
                    object=" ".join(words[1:]).lower(),
                    claim_text=sent,
                    scene_id=scene_id,
                    character_id=character_id,
                    sentence_idx=sent_idx,
                )
            )

        return claims

    def _normalize_text(self, text: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
        return re.sub(r"\s+", " ", normalized).strip()

    def _normalize_speaker_subject(self, speaker: str) -> str:
        normalized = self._normalize_text(speaker)
        tokens = [token for token in normalized.split() if token and token not in TITLE_TOKENS]
        return " ".join(tokens).strip() or normalized

    def _clean_descriptor(self, text: str) -> str:
        descriptor = text.strip(" .!?,-").lower()
        descriptor = re.sub(r"^(but\s+)?(still|somewhat|very)\s+", "", descriptor)
        descriptor = re.sub(r"^but\s+", "", descriptor)
        if not descriptor:
            return ""
        if descriptor.startswith("about ") and any(char.isdigit() for char in descriptor):
            return ""
        if descriptor in GENERIC_SUBJECTS:
            return ""
        if len(descriptor.split()) > 8:
            return ""
        return descriptor

    def _looks_like_name_phrase(self, text: str) -> bool:
        return bool(NAME_PHRASE_RE.match(text.strip()))

    def _is_state_phrase(self, text: str) -> bool:
        if any(hint in text for hint in STATE_ADJ_HINTS):
            return True
        return any(word in text.split() for word in {"aware", "curious", "boyish", "laconic", "beautiful", "young"})

    def _heuristic_lemmatize_present(self, verb: str) -> str:
        lower = verb.lower()
        if lower.endswith("ies") and len(lower) > 3:
            return lower[:-3] + "y"
        if lower.endswith("ves") and len(lower) > 3:
            return lower[:-3] + "ve"
        if lower.endswith("oes") and len(lower) > 3:
            return lower[:-2]
        if lower.endswith(("ches", "shes", "sses", "zzes", "xes")) and len(lower) > 4:
            return lower[:-2]
        if lower.endswith("ses") and len(lower) > 3:
            if lower[-4:-2] == "ss":
                return lower[:-2]
            return lower[:-1]
        if lower.endswith("zes") and len(lower) > 3:
            return lower[:-2]
        if lower.endswith("es") and len(lower) > 3:
            return lower[:-1]
        if lower.endswith("s") and len(lower) > 2:
            return lower[:-1]
        return lower

    def _strip_trailing_address(self, phrase: str) -> str:
        parts = phrase.split()
        if len(parts) > 2 and parts[-1].lower() in TRAILING_ADDRESS_TOKENS:
            return " ".join(parts[:-1])
        return phrase
