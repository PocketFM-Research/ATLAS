import re
from dataclasses import dataclass
from typing import List, Set, Tuple

import spacy


LOCATION_PREPS = {"in", "at", "into", "onto", "inside", "outside", "near", "on"}
NON_LOCATION_OBJECTS = {"use", "appearance"}
LOCATION_NOUN_HINTS = {
    "bridge", "room", "corridor", "console", "station", "dock", "bay", "zone",
    "ship", "enterprise", "lab", "elevator", "planet", "space", "home", "deck",
}
STATE_ADJ_HINTS = {
    "new", "unexpected", "young", "beautiful", "fair", "angry", "sad",
    "worried", "nervous", "ready", "curious", "afraid", "troubled",
}
TIME_WORDS = {
    "day", "night", "morning", "evening", "afternoon",
    "today", "tomorrow", "yesterday", "stardate",
}
BE_FORMS = {"be", "am", "is", "are", "was", "were"}
POSSESS_FORMS = {"have", "has", "had"}


@dataclass
class Claim:
    """Atomic extracted claim."""

    subject: str
    predicate: str
    object: str
    claim_text: str
    scene_id: str
    sentence_idx: int

    def to_dict(self):
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "claim_text": self.claim_text,
            "scene_id": self.scene_id,
            "sentence_idx": self.sentence_idx,
        }


class ClaimExtractor:
    """Extract atomic claims and normalize predicates to known schema relations."""

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

    def extract_claims(self, text: str, scene_id: str) -> List[Claim]:
        if not self.nlp:
            return self._extract_claims_regex(text, scene_id)

        doc = self.nlp(text)
        claims: List[Claim] = []
        seen: Set[Tuple[str, str, str, int]] = set()

        for sent_idx, sent in enumerate(doc.sents):
            sent_claims = self._extract_from_sent(sent, sent_idx, scene_id)
            for claim in sent_claims:
                key = (claim.subject, claim.predicate, claim.object, claim.sentence_idx)
                if key not in seen:
                    seen.add(key)
                    claims.append(claim)
        return claims

    def _extract_from_sent(self, sent, sent_idx: int, scene_id: str) -> List[Claim]:
        claims: List[Claim] = []
        root = sent.root
        root_lemma = root.lemma_.lower()

        subjects = self._extract_subjects(root, sent)
        if not subjects:
            return claims

        location_claims = self._extract_prep_claims(
            root, sent, scene_id, sent_idx, subjects
        )
        claims.extend(location_claims)

        if root_lemma in BE_FORMS:
            claims.extend(
                self._extract_be_claims(root, sent, scene_id, sent_idx, subjects)
            )
        elif root_lemma in POSSESS_FORMS:
            claims.extend(
                self._extract_possess_claims(root, sent, scene_id, sent_idx, subjects)
            )
        else:
            claims.extend(
                self._extract_action_claims(root, sent, scene_id, sent_idx, subjects)
            )

        return claims

    def _extract_subjects(self, root, sent) -> List[str]:
        subjects = []
        for token in sent:
            if token.head == root and token.dep_ in {"nsubj", "nsubjpass", "expl"}:
                phrase = self._get_token_phrase(token)
                if phrase:
                    subjects.append(phrase)
        return subjects

    def _extract_be_claims(self, root, sent, scene_id, sent_idx, subjects) -> List[Claim]:
        claims: List[Claim] = []
        attrs = []
        for token in sent:
            if token.head == root and token.dep_ in {"attr", "acomp", "oprd"}:
                phrase = self._get_token_phrase(token)
                if phrase:
                    attrs.append((token, phrase))

        for subj in subjects:
            if not attrs:
                claims.append(
                    Claim(
                        subj,
                        "is_a",
                        self._fallback_object(sent, root),
                        sent.text.strip(),
                        scene_id,
                        sent_idx,
                    )
                )
                continue

            for attr_token, attr in attrs:
                pred = self._classify_copula_relation(attr_token, attr)
                claims.append(
                    Claim(subj, pred, attr, sent.text.strip(), scene_id, sent_idx)
                )
        return claims

    def _extract_possess_claims(self, root, sent, scene_id, sent_idx, subjects) -> List[Claim]:
        claims: List[Claim] = []
        objects = []
        for token in sent:
            if token.head == root and token.dep_ in {"dobj", "obj", "attr"}:
                phrase = self._get_token_phrase(token)
                if phrase:
                    objects.append(phrase)

        for subj in subjects:
            if not objects:
                claims.append(
                    Claim(
                        subj,
                        "possesses",
                        self._fallback_object(sent, root),
                        sent.text.strip(),
                        scene_id,
                        sent_idx,
                    )
                )
                continue

            for obj in objects:
                claims.append(
                    Claim(subj, "possesses", obj, sent.text.strip(), scene_id, sent_idx)
                )
        return claims

    def _extract_action_claims(self, root, sent, scene_id, sent_idx, subjects) -> List[Claim]:
        claims: List[Claim] = []
        action_phrase = self._build_action_phrase(root)
        if not action_phrase:
            action_phrase = self._fallback_object(sent, root)

        for subj in subjects:
            claims.append(
                Claim(subj, "performs", action_phrase, sent.text.strip(), scene_id, sent_idx)
            )
        return claims

    def _extract_prep_claims(self, root, sent, scene_id, sent_idx, subjects) -> List[Claim]:
        claims: List[Claim] = []
        for token in sent:
            if token.head != root or token.dep_ != "prep":
                continue
            prep = token.text.lower()
            obj_token = next((child for child in token.children if child.dep_ == "pobj"), None)
            if obj_token is None:
                continue
            obj_phrase = self._get_token_phrase(obj_token)
            if not obj_phrase:
                continue
            if obj_phrase in NON_LOCATION_OBJECTS:
                continue

            if prep in {"on"} and self._looks_like_time(obj_phrase):
                predicate = "occurs_on"
            elif prep in LOCATION_PREPS and self._looks_like_location(obj_phrase):
                predicate = "located_at"
            else:
                continue

            for subj in subjects:
                claims.append(
                    Claim(subj, predicate, obj_phrase, sent.text.strip(), scene_id, sent_idx)
                )
        if not claims:
            for subj in subjects:
                claims.append(
                    Claim(
                        subj,
                        "located_at",
                        self._fallback_object(sent, root),
                        sent.text.strip(),
                        scene_id,
                        sent_idx,
                    )
                )
        return claims

    def _build_action_phrase(self, root) -> str:
        pieces = [root.lemma_.lower()]

        direct_objs = [
            self._get_token_phrase(child)
            for child in root.children
            if child.dep_ in {"dobj", "obj", "attr"} and self._get_token_phrase(child)
        ]
        if direct_objs:
            pieces.append(direct_objs[0])

        prt = next((child.text.lower() for child in root.children if child.dep_ == "prt"), "")
        if prt:
            pieces.append(prt)

        prep_phrases = []
        for child in root.children:
            if child.dep_ != "prep":
                continue
            obj_token = next((grand for grand in child.children if grand.dep_ == "pobj"), None)
            if obj_token is None:
                continue
            obj_phrase = self._get_token_phrase(obj_token)
            if obj_phrase:
                prep_phrases.append(f"{child.text.lower()} {obj_phrase}")

        if prep_phrases:
            pieces.append(prep_phrases[0])

        phrase = " ".join(piece for piece in pieces if piece).strip()
        return re.sub(r"\s+", " ", phrase)

    def _get_token_phrase(self, token) -> str:
        tokens = [
            tok
            for tok in token.subtree
            if not tok.is_space and tok.dep_ != "punct"
        ]
        if not tokens:
            return ""

        phrase = " ".join(tok.text for tok in tokens)
        phrase = re.sub(r"\s+", " ", phrase).strip(" -,:;")
        if not phrase:
            return ""

        # Filter out all-caps screenplay speaker labels.
        if phrase.isupper() and len(phrase.split()) <= 3:
            return ""

        return phrase.lower()

    def _looks_like_time(self, text: str) -> bool:
        lowered = text.lower()
        return any(word in lowered for word in TIME_WORDS)

    def _looks_like_location(self, text: str) -> bool:
        lowered = text.lower()
        return any(hint in lowered for hint in LOCATION_NOUN_HINTS)

    def _classify_copula_relation(self, attr_token, attr_phrase: str) -> str:
        if self._looks_like_time(attr_phrase):
            return "occurs_on"

        lower = attr_phrase.lower()
        if any(hint in lower for hint in LOCATION_NOUN_HINTS):
            return "located_at"

        if self._is_state_description(attr_token, lower):
            return "experiences"

        return "is_a"

    def _is_state_description(self, attr_token, lower_phrase: str) -> bool:
        if attr_token.pos_ == "ADJ":
            return True

        tokens = [tok for tok in attr_token.subtree if not tok.is_space and tok.dep_ != "punct"]
        if not tokens:
            return False

        pos_set = {tok.pos_ for tok in tokens}
        if pos_set <= {"ADJ", "ADV", "CCONJ", "DET", "PART"}:
            return True

        return any(hint in lower_phrase for hint in STATE_ADJ_HINTS)

    def _fallback_object(self, sent, root) -> str:
        text = sent.text.strip()
        if not text:
            return "unspecified"

        root_text = root.text.lower().strip()
        if root_text:
            idx = text.lower().find(root_text)
            if idx != -1:
                tail = text[idx + len(root_text):].strip(" \n\t-,:;.!?")
                tail = re.sub(r"\s+", " ", tail).strip()
                if tail:
                    return tail.lower()

        cleaned = re.sub(r"[\r\n]+", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -,:;")
        return cleaned.lower() if cleaned else "unspecified"

    def _extract_claims_regex(self, text: str, scene_id: str) -> List[Claim]:
        claims: List[Claim] = []
        sentences = re.split(r"[.!?]+", text)

        for sent_idx, sent in enumerate(sentences):
            sent = sent.strip()
            if not sent:
                continue
            words = sent.split()
            if len(words) < 2:
                continue
            subject = words[0].lower()
            obj = " ".join(words[1:]).lower()
            claims.append(
                Claim(subject, "performs", obj, sent, scene_id, sent_idx)
            )
        return claims
