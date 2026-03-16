"""FActScore-style claim extraction from generated text."""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass
import spacy

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


class ClaimExtractor:
    """Extract atomic subject-predicate-object claims from text."""
    
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
        character_id: str
    ) -> List[Claim]:
        """
        Extract atomic claims from generated text.
        
        Args:
            text: Generated description text
            scene_id: Scene identifier
            character_id: Character being described
            
        Returns:
            List of Claim objects
        """
        if not self.nlp:
            return self._extract_claims_regex(text, scene_id, character_id)
        
        claims = []
        doc = self.nlp(text)
        
        for sent_idx, sent in enumerate(doc.sents):
            # Extract SVO triples via dependency parsing
            sent_claims = self._extract_from_sent(sent, sent_idx, scene_id, character_id)
            claims.extend(sent_claims)
        
        return claims
    
    def _extract_from_sent(
        self,
        sent,
        sent_idx: int,
        scene_id: str,
        character_id: str
    ) -> List[Claim]:
        """Extract SVO triples from a single sentence using dependency parsing."""
        claims = []
        
        # Find root verb
        root = None
        for token in sent:
            if token.dep_ == "ROOT" and token.pos_ in ["VERB", "AUX"]:
                root = token
                break
        
        if not root:
            return claims
        
        # Extract subjects
        subjects = []
        for token in sent:
            if token.head == root and token.dep_ in ["nsubj", "nsubjpass"]:
                subjects.append(self._get_noun_phrase(token))
        
        # Extract objects
        objects = []
        for token in sent:
            if token.head == root and token.dep_ in ["dobj", "attr", "prep"]:
                objects.append(self._get_noun_phrase(token))
        
        # Create claims for all subject-verb-object combinations
        for subj in subjects:
            for obj in objects:
                claims.append(Claim(
                    subject=subj,
                    predicate=root.text,
                    object=obj,
                    claim_text=sent.text,
                    scene_id=scene_id,
                    character_id=character_id,
                    sentence_idx=sent_idx,
                ))
        
        # If no objects found, still create subject-verb claim
        if subjects and not objects:
            for subj in subjects:
                claims.append(Claim(
                    subject=subj,
                    predicate=root.text,
                    object="",
                    claim_text=sent.text,
                    scene_id=scene_id,
                    character_id=character_id,
                    sentence_idx=sent_idx,
                ))
        
        return claims
    
    def _get_noun_phrase(self, token) -> str:
        """Extract noun phrase from a token (includes modifiers)."""
        phrase = token.text
        
        # Add modifiers
        for child in token.children:
            if child.dep_ in ["amod", "det", "nmod", "compound"]:
                phrase = f"{child.text} {phrase}"
        
        # Add children (for compound nouns)
        for child in token.children:
            if child.dep_ in ["punc"]:
                continue
            if child.pos_ in ["NOUN", "PROPN", "ADJ"]:
                phrase = f"{phrase} {child.text}"
        
        return phrase.strip().lower()
    
    def _extract_claims_regex(
        self,
        text: str,
        scene_id: str,
        character_id: str
    ) -> List[Claim]:
        """Fallback regex-based claim extraction (no spaCy)."""
        claims = []
        sentences = re.split(r'[.!?]+', text)
        
        for sent_idx, sent in enumerate(sentences):
            sent = sent.strip()
            if not sent:
                continue
            
            # Simple pattern: find capitalized nouns (subjects) and verbs
            words = sent.split()
            
            # Very basic: assume first capitalized word is subject
            subject = ""
            predicate = ""
            obj = ""
            
            for word in words:
                if not subject and word[0].isupper():
                    subject = word.lower()
                elif subject and not predicate:
                    predicate = word.lower()
                elif subject and predicate and not obj:
                    obj = word.lower()
            
            if subject and predicate:
                claims.append(Claim(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    claim_text=sent,
                    scene_id=scene_id,
                    character_id=character_id,
                    sentence_idx=sent_idx,
                ))
        
        return claims