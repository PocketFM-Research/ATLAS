"""Build evaluation text directly from screenplay scenes and graph evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from stage_kg.ingest.loader import MovieData


GENERIC_ALIASES = {
    "captain",
    "the captain",
    "mister",
    "mr",
    "mr.",
    "doctor",
    "dr",
    "dr.",
    "voice",
    "computer",
    "subject vessel",
    "crew",
}
TITLE_TOKENS = {
    "the",
    "captain",
    "commander",
    "lt",
    "lt.",
    "mr",
    "mr.",
    "mister",
    "dr",
    "dr.",
    "doctor",
}
SPEAKER_LABEL_RE = re.compile(r"^[A-Z0-9][A-Z0-9 .'\-:]+(?:\([A-Z0-9 .'\-]+\))?$")
CLAUSE_SPLIT_PATTERNS = (
    re.compile(r"\s*;\s*"),
    re.compile(
        r",\s+(?=(?:and\s+)?(?:[A-Z][A-Z0-9.'\-]+|[A-Z][a-z]+)(?:\s+(?:[A-Z][A-Z0-9.'\-]+|[A-Z][a-z]+)){0,3}\b)"
    ),
    re.compile(r"\s+but\s+(?=[A-Z])"),
)


@dataclass
class SceneTextBuildStats:
    """Coverage stats for screenplay-backed evaluation text."""

    total_scenes: int = 0
    total_character_slots: int = 0
    populated_character_slots: int = 0
    empty_character_slots: int = 0
    low_evidence_slots: int = 0
    script_snippet_hits: int = 0
    graph_evidence_hits: int = 0

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class LowEvidenceCharacterScene:
    """A character/scene pair with insufficient grounded evidence for scoring."""

    scene_id: str
    character_id: str
    character_name: str
    message: str
    reason: str
    script_snippet_hits: int = 0
    graph_evidence_hits: int = 0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class ScreenplaySceneTextBuilder:
    """Construct per-scene, per-character text from script content."""

    def __init__(self, max_snippets: int = 8, max_chars: int = 1800):
        self.max_snippets = max_snippets
        self.max_chars = max_chars

    def build(
        self,
        movie: MovieData,
        graph_data: Dict,
    ) -> Tuple[
        Dict[str, Dict[str, str]],
        SceneTextBuildStats,
        List[LowEvidenceCharacterScene],
    ]:
        """Build `{scene_id: {character_id: text}}` from screenplay scenes."""
        scene_descriptions: Dict[str, Dict[str, str]] = {}
        stats = SceneTextBuildStats(total_scenes=len(movie.scenes))
        low_evidence_entries: List[LowEvidenceCharacterScene] = []

        scenes_by_id = {scene.scene_id: scene for scene in movie.scenes}
        characters = [
            node for node in graph_data.get("nodes", []) if node.get("type") == "Character"
        ]
        incident_edges = self._build_incident_edge_index(graph_data.get("edges", []))

        for character in characters:
            char_id = character["id"]
            scene_refs = sorted(
                {str(scene_id) for scene_id in character.get("scene_refs", [])},
                key=self._scene_sort_key,
            )

            for scene_id in scene_refs:
                stats.total_character_slots += 1
                scene = scenes_by_id.get(scene_id)
                if scene is None:
                    stats.empty_character_slots += 1
                    continue

                text, script_hits, evidence_hits, reason = self._build_character_scene_text(
                    scene_id=scene_id,
                    scene_text=scene.content,
                    character=character,
                    incident_edges=incident_edges.get((char_id, scene_id), []),
                )

                stats.script_snippet_hits += script_hits
                stats.graph_evidence_hits += evidence_hits

                if not text:
                    stats.low_evidence_slots += 1
                    low_evidence_entries.append(
                        LowEvidenceCharacterScene(
                            scene_id=scene_id,
                            character_id=char_id,
                            character_name=character.get("name", char_id),
                            message=(
                                f"{character.get('name', char_id)} is present in scene {scene_id} "
                                "but has no grounded actions or descriptive evidence."
                            ),
                            reason=reason or "no_grounded_scene_evidence",
                            script_snippet_hits=script_hits,
                            graph_evidence_hits=evidence_hits,
                        )
                    )
                    continue

                scene_descriptions.setdefault(scene_id, {})[char_id] = text
                stats.populated_character_slots += 1

        stats.empty_character_slots = max(
            0, stats.total_character_slots - stats.populated_character_slots - stats.low_evidence_slots
        )

        return scene_descriptions, stats, low_evidence_entries

    def _build_character_scene_text(
        self,
        scene_id: str,
        scene_text: str,
        character: Dict,
        incident_edges: Sequence[Dict],
    ) -> Tuple[str, int, int, str]:
        aliases = self._build_aliases(character)
        script_snippets = self._match_scene_snippets(scene_text, aliases)

        evidence_snippets: List[str] = []
        for edge in incident_edges:
            evidence_snippets.extend(edge.get("evidence", []) or [])

        focused_evidence_snippets: List[str] = []
        for snippet in evidence_snippets:
            focused_evidence_snippets.extend(self._extract_character_focused_snippets(snippet, aliases))

        informative_script_snippets = [
            snippet for snippet in script_snippets if self._is_informative_snippet(snippet, aliases)
        ]
        informative_evidence_snippets = [
            snippet for snippet in focused_evidence_snippets if self._is_informative_snippet(snippet, aliases)
        ]

        combined: List[str] = []
        seen: Set[str] = set()

        for snippet in informative_script_snippets:
            key = self._normalize_text(snippet)
            if key and key not in seen:
                seen.add(key)
                combined.append(snippet.strip())

        for snippet in informative_evidence_snippets:
            key = self._normalize_text(snippet)
            if key and key not in seen:
                seen.add(key)
                combined.append(snippet.strip())

        if not combined:
            reason = self._low_evidence_reason(
                aliases=aliases,
                script_hits=len(script_snippets),
                evidence_hits=len(evidence_snippets),
                scene_id=scene_id,
            )
            return "", len(script_snippets), len(evidence_snippets), reason

        selected: List[str] = []
        total_chars = 0
        for snippet in combined[: self.max_snippets]:
            if not snippet:
                continue
            candidate = snippet.strip()
            if not candidate:
                continue
            projected = total_chars + len(candidate) + (1 if selected else 0)
            if projected > self.max_chars:
                break
            selected.append(candidate)
            total_chars = projected

        return "\n".join(selected), len(script_snippets), len(evidence_snippets), ""

    def _build_incident_edge_index(
        self,
        edges: Iterable[Dict],
    ) -> Dict[Tuple[str, str], List[Dict]]:
        index: Dict[Tuple[str, str], List[Dict]] = {}
        for edge in edges:
            scene_refs = [str(scene_id) for scene_id in edge.get("scene_refs", [])]
            for scene_id in scene_refs:
                index.setdefault((edge.get("source", ""), scene_id), []).append(edge)
                index.setdefault((edge.get("target", ""), scene_id), []).append(edge)
        return index

    def _build_aliases(self, character: Dict) -> Set[str]:
        aliases: Set[str] = set()
        canonical_name = self._normalize_text(character.get("name", ""))
        canonical_tokens = {
            token for token in canonical_name.split()
            if token and token not in TITLE_TOKENS
        }

        raw_aliases = [character.get("name", "")]
        raw_aliases.extend(character.get("aliases", []) or [])

        for alias in raw_aliases:
            normalized = self._normalize_text(alias)
            if not normalized or normalized in GENERIC_ALIASES:
                continue

            alias_tokens = [
                token for token in normalized.split()
                if token and token not in TITLE_TOKENS
            ]
            if normalized != canonical_name and canonical_tokens:
                if not set(alias_tokens) & canonical_tokens:
                    continue

            aliases.add(normalized)

            tokens = alias_tokens or normalized.split()
            if len(tokens) > 1:
                surname = tokens[-1]
                if surname not in GENERIC_ALIASES and len(surname) > 2:
                    aliases.add(surname)

        return aliases

    def _match_scene_snippets(self, scene_text: str, aliases: Set[str]) -> List[str]:
        if not scene_text or not aliases:
            return []

        snippets = self._split_scene_text(scene_text)
        matched: List[str] = []

        for snippet in snippets:
            focused_snippets = self._extract_character_focused_snippets(snippet, aliases)
            matched.extend(focused_snippets)

        return matched

    def _extract_character_focused_snippets(self, snippet: str, aliases: Set[str]) -> List[str]:
        normalized = self._normalize_text(snippet)
        if not normalized:
            return []
        if not any(alias in normalized for alias in aliases):
            return []

        clauses = [snippet.strip()]
        for pattern in CLAUSE_SPLIT_PATTERNS:
            split_clauses: List[str] = []
            for clause in clauses:
                parts = [part.strip(" ,-") for part in pattern.split(clause) if part.strip(" ,-")]
                split_clauses.extend(parts or [clause])
            clauses = split_clauses

        focused: List[str] = []
        seen: Set[str] = set()
        for clause in clauses:
            normalized_clause = self._normalize_text(clause)
            if not normalized_clause:
                continue
            if not any(alias in normalized_clause for alias in aliases):
                continue
            if normalized_clause in seen:
                continue
            seen.add(normalized_clause)
            focused.append(clause.strip())

        return focused or [snippet.strip()]

    def _is_informative_snippet(self, snippet: str, aliases: Set[str]) -> bool:
        normalized = self._normalize_text(snippet)
        if not normalized:
            return False

        if normalized in aliases:
            return False

        if len(normalized.split()) <= 2 and normalized.isalpha():
            return False

        return True

    def _low_evidence_reason(
        self,
        aliases: Set[str],
        script_hits: int,
        evidence_hits: int,
        scene_id: str,
    ) -> str:
        if script_hits == 0 and evidence_hits == 0:
            return "scene_ref_without_matching_script_or_graph_evidence"
        if script_hits > 0 and evidence_hits == 0:
            return "only_name_mentions_without_grounded_graph_edges"
        if script_hits == 0 and evidence_hits > 0:
            return "graph_evidence_present_but_not_scene_specific"
        return "insufficient_informative_scene_evidence"

    def _split_scene_text(self, scene_text: str) -> List[str]:
        lines = [line.strip() for line in scene_text.replace("\r", "\n").split("\n")]

        blocks: List[str] = []
        current_block: List[str] = []
        current_kind = ""

        for line in lines:
            if not line:
                if current_block:
                    blocks.append(self._finalize_screenplay_block(current_block, current_kind))
                    current_block = []
                    current_kind = ""
                continue

            if self._is_speaker_label(line):
                if current_block:
                    blocks.append(self._finalize_screenplay_block(current_block, current_kind))
                current_block = [line]
                current_kind = "dialogue"
                continue

            if current_kind == "dialogue":
                current_block.append(line)
                continue

            if current_kind != "action":
                current_block = []
                current_kind = "action"
            current_block.append(line)

        if current_block:
            blocks.append(self._finalize_screenplay_block(current_block, current_kind))

        segments: List[str] = []
        for block in blocks:
            if not block:
                continue
            parts = re.split(r"(?<=[.!?])\s+", block)
            segments.extend(part.strip() for part in parts if part.strip())

        return segments

    def _finalize_screenplay_block(self, lines: List[str], kind: str) -> str:
        if not lines:
            return ""

        if kind == "dialogue":
            speaker = lines[0].strip()
            content_lines = list(lines[1:])
            while content_lines and re.fullmatch(r"\([^)]*\)", content_lines[0].strip()):
                content_lines.pop(0)
            if not content_lines:
                return speaker

            combined = content_lines[0]
            for line in content_lines[1:]:
                if combined.endswith("-"):
                    combined = combined[:-1] + line
                else:
                    combined = f"{combined} {line}"
            combined = re.sub(r"\s+", " ", combined).strip()
            return f"{speaker}: {combined}"

        combined = lines[0]
        for line in lines[1:]:
            if combined.endswith("-"):
                combined = combined[:-1] + line
            else:
                combined = f"{combined} {line}"

        combined = re.sub(r"\s+", " ", combined).strip()
        return combined

    def _is_speaker_label(self, line: str) -> bool:
        if len(line) > 40:
            return False
        if not any(char.isalpha() for char in line):
            return False
        if line != line.upper():
            return False
        return bool(SPEAKER_LABEL_RE.match(line))

    def _normalize_text(self, text: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
        return re.sub(r"\s+", " ", normalized).strip()

    def _scene_sort_key(self, value: str) -> Tuple[int, str]:
        if value.isdigit():
            return int(value), value
        return 10**9, value
