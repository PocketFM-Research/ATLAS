"""
Screenplay data ingestion from the STAGE dataset.

Supports:
- script.json  (scene-segmented screenplay)
- doc2chunks.json (pre-computed chunking)
- extraction_results.json (reference annotations; optional)
- rename_map.json (entity normalization hints)
- CSV metadata files
"""

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

MIN_SCENE_CHUNK_MERGE_CHARS = 1200


@dataclass
class SceneRecord:
    """A single scene from script.json."""
    scene_id: str           # e.g. "1", "2" (from _id + part)
    order: int              # 0-based position
    title: str              # scene heading e.g. "1、INT. ENTERPRISE BRIDGE"
    subtitle: str
    content: str            # full scene text
    scene_category: str     # INT / EXT / other
    region: str             # high-level location name
    main_location: str
    sub_location: str
    summary: str            # from doc2chunks metadata if available
    chunks: List[Dict]      # from doc2chunks.json
    source_doc_id: str      # doc_0, doc_1, ...


@dataclass
class MovieData:
    """All ingested data for one movie."""
    movie_id: str
    title: str
    language: str           # "en" or "zh"
    data_dir: Path

    scenes: List[SceneRecord] = field(default_factory=list)
    rename_map: Dict[str, str] = field(default_factory=dict)

    # Reference annotations (not used in extraction, useful for eval)
    extraction_results: Dict[str, Any] = field(default_factory=dict)
    episodes: List[Dict] = field(default_factory=list)
    episode_relations: List[Dict] = field(default_factory=list)


def load_movie(movie_dir: Path, movie_id: str, title: str = "", language: str = "en") -> MovieData:
    """
    Load all available data for one movie from its dataset directory.

    Args:
        movie_dir: Path to the movie's directory (contains script.json etc.)
        movie_id: Unique movie identifier.
        title: Human-readable title (from CSV metadata).
        language: 'en' or 'zh'.

    Returns:
        Populated MovieData instance.
    """
    movie = MovieData(
        movie_id=movie_id,
        title=title,
        language=language,
        data_dir=movie_dir,
    )

    # Load scene data
    script_path = movie_dir / "script.json"
    doc2chunks_path = movie_dir / "doc2chunks.json"

    if not script_path.exists():
        logger.error("script.json not found in %s", movie_dir)
        return movie

    with script_path.open("r", encoding="utf-8") as f:
        raw_scenes = json.load(f)

    # Build doc2chunks index for metadata/chunks
    doc2chunks: Dict[str, Any] = {}
    if doc2chunks_path.exists():
        with doc2chunks_path.open("r", encoding="utf-8") as f:
            doc2chunks = json.load(f)

    # Build scene records
    for raw in raw_scenes:
        scene_id = str(raw["_id"] + 1)  # 1-based string id matching "scene_1_part_1"
        doc_key = f"scene_{scene_id}_part_1"
        d2c = doc2chunks.get(doc_key, {})
        meta = d2c.get("document_metadata", {})

        scene = SceneRecord(
            scene_id=scene_id,
            order=raw["_id"],
            title=raw.get("title", ""),
            subtitle=raw.get("subtitle", ""),
            content=raw.get("content", ""),
            scene_category=meta.get("scene_category", _infer_category(raw.get("title", ""))),
            region=meta.get("region", ""),
            main_location=meta.get("main_location", ""),
            sub_location=meta.get("sub_location", "") or "",
            summary=meta.get("summary", ""),
            chunks=d2c.get("chunks", []),
            source_doc_id=meta.get("source_doc_id", f"doc_{raw['_id']}"),
        )

        scene.chunks = _prepare_scene_chunks(
            raw_scene_text=raw.get("content", ""),
            raw_scene_id=raw["_id"],
            source_doc_id=scene.source_doc_id,
            chunks=scene.chunks,
        )

        movie.scenes.append(scene)

    logger.info(
        "Loaded %d scenes for movie '%s' (%s)", len(movie.scenes), title or movie_id, movie_id
    )

    # Load rename_map
    rename_path = movie_dir / "rename_map.json"
    if rename_path.exists():
        with rename_path.open("r", encoding="utf-8") as f:
            movie.rename_map = json.load(f)
        logger.debug("Loaded %d rename mappings", len(movie.rename_map))

    # Load reference extraction results (optional)
    ext_path = movie_dir / "extraction_results.json"
    if ext_path.exists():
        with ext_path.open("r", encoding="utf-8") as f:
            movie.extraction_results = json.load(f)

    # Load episodes
    ep_path = movie_dir / "episodes.json"
    if ep_path.exists():
        with ep_path.open("r", encoding="utf-8") as f:
            movie.episodes = json.load(f)

    # Load episode relations (filename has typo in dataset)
    for rel_name in ("episde_relations.json", "episode_relations.json"):
        rel_path = movie_dir / rel_name
        if rel_path.exists():
            with rel_path.open("r", encoding="utf-8") as f:
                movie.episode_relations = json.load(f)
            break

    return movie


def load_all_movies(
    dataset_dir: Path,
    language: str = "en",
    movie_ids: Optional[List[str]] = None,
) -> List[MovieData]:
    """
    Load all (or selected) movies from the dataset directory.

    Args:
        dataset_dir: Root dataset directory containing English/ and Chinese/.
        language: 'en' or 'zh'.
        movie_ids: If provided, only load these movie IDs.

    Returns:
        List of MovieData objects.
    """
    lang_dir_map = {"en": "English", "zh": "Chinese"}
    csv_map = {"en": "english_movie_info.csv", "zh": "chinese_movie_info.csv"}

    lang_dir = dataset_dir / lang_dir_map.get(language, "English")
    csv_path = dataset_dir / csv_map.get(language, "english_movie_info.csv")

    # Read metadata CSV
    metadata: Dict[str, Dict] = {}
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata[row["movie_id"]] = row

    movies = []
    if not lang_dir.exists():
        logger.error("Language directory not found: %s", lang_dir)
        return movies

    dirs = sorted(lang_dir.iterdir())
    for movie_dir in dirs:
        if not movie_dir.is_dir():
            continue
        mid = movie_dir.name
        if movie_ids and mid not in movie_ids:
            continue
        meta = metadata.get(mid, {})
        movie = load_movie(
            movie_dir=movie_dir,
            movie_id=mid,
            title=meta.get("title", ""),
            language=language,
        )
        movies.append(movie)

    logger.info("Loaded %d movies (language=%s)", len(movies), language)
    return movies


def _infer_category(title: str) -> str:
    """Infer INT/EXT from scene heading title."""
    upper = title.upper()
    if "INT." in upper:
        return "INT"
    if "EXT." in upper:
        return "EXT"
    return "UNKNOWN"


def _prepare_scene_chunks(
    raw_scene_text: str,
    raw_scene_id: int,
    source_doc_id: str,
    chunks: List[Dict],
) -> List[Dict]:
    """Collapse very short scenes into a single chunk to avoid over-segmentation."""
    if not chunks:
        return [_single_chunk(raw_scene_text, raw_scene_id, source_doc_id)]

    scene_text = (raw_scene_text or "").strip()
    if not scene_text:
        return chunks

    if len(scene_text) <= MIN_SCENE_CHUNK_MERGE_CHARS and len(chunks) > 1:
        return [_single_chunk(raw_scene_text, raw_scene_id, source_doc_id)]

    return chunks


def _single_chunk(raw_scene_text: str, raw_scene_id: int, source_doc_id: str) -> Dict[str, Any]:
    return {
        "id": f"doc_{raw_scene_id}_chunk_0",
        "content": raw_scene_text,
        "source_doc_id": source_doc_id,
        "start_pos": 0,
        "end_pos": len(raw_scene_text),
    }
