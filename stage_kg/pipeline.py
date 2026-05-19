"""
STAGE Knowledge Graph Pipeline Orchestrator.

Runs the full extraction pipeline for one movie:
  Pass 1: Event extraction (per scene/chunk)
  Pass 2: Entity extraction (anchored to events)
  Pass 3: Relation extraction (schema-constrained)
  Pass 4: Normalization (string clustering + LLM adjudication)
  Pass 5: Graph consolidation and serialization

Intermediate outputs are cached to disk for resumability.
"""

import copy
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .ingest.loader import MovieData, SceneRecord
from .extraction.event_extractor import extract_events_for_movie
from .extraction.entity_extractor import extract_entities_for_movie
from .extraction.relation_extractor import extract_relations_for_movie
from .normalization.normalizer import normalize_nodes
from .graph.builder import build_graph_from_extractions, KnowledgeGraph
from .evaluation.stats import compute_stats, print_stats, compute_pre_post_merge_stats
from .llm.base import BaseLLM
from .utils.cache import Cache
from .utils.logging_utils import PromptLogger

logger = logging.getLogger(__name__)

ENTITY_TYPES = ["Character", "Location", "TimePoint", "Object", "Vehicle", "Concept"]


def run_pipeline(
    movie: MovieData,
    llm: BaseLLM,
    output_dir: Path,
    skip_normalization: bool = False,
    api_key: Optional[str] = None,
) -> KnowledgeGraph:
    """
    Run the full KG extraction pipeline for one movie.

    Args:
        movie: Ingested MovieData object.
        llm: Configured LLM backend.
        output_dir: Root output directory. Artifacts go to output_dir/<movie_id>/.
        skip_normalization: Skip LLM-based merge adjudication (faster, lower quality).

    Returns:
        Final KnowledgeGraph for the movie.
    """
    movie_out = Path(output_dir) / movie.movie_id
    movie_out.mkdir(parents=True, exist_ok=True)

    cache = Cache(movie_out / ".cache")
    prompt_logger = PromptLogger(movie_out / "logs", movie.movie_id)
    embedding_api_key = api_key if llm.model_id.startswith("gemini/") else None

    logger.info(
        "=" * 60 + f"\nStarting pipeline for: {movie.title or movie.movie_id}\n"
        + f"Scenes: {len(movie.scenes)} | Model: {llm.model_id}\n" + "=" * 60
    )

    # ------------------------------------------------------------------ Pass 1: Events
    logger.info("[Pass 1] Event extraction ...")
    all_events = extract_events_for_movie(
        scenes=movie.scenes,
        llm=llm,
        movie_id=movie.movie_id,
        movie_title=movie.title,
        cache=cache,
        prompt_logger=prompt_logger,
    )
    _save_intermediate(all_events, movie_out / "raw_events.json")

    # ------------------------------------------------------------------ Pass 2: Entities
    logger.info("[Pass 2] Entity extraction ...")
    all_entities = extract_entities_for_movie(
        scenes=movie.scenes,
        scene_events=all_events,
        llm=llm,
        movie_id=movie.movie_id,
        movie_title=movie.title,
        cache=cache,
        prompt_logger=prompt_logger,
    )
    _save_intermediate(all_entities, movie_out / "raw_entities.json")

    # ------------------------------------------------------------------ Pass 3: Relations
    logger.info("[Pass 3] Relation extraction ...")
    all_relations = extract_relations_for_movie(
        scenes=movie.scenes,
        scene_events=all_events,
        scene_entities=all_entities,
        llm=llm,
        movie_id=movie.movie_id,
        movie_title=movie.title,
        cache=cache,
        prompt_logger=prompt_logger,
    )
    _save_intermediate(all_relations, movie_out / "raw_relations.json")

    # ------------------------------------------------------------------ Pass 4: Normalization
    logger.info("[Pass 4] Normalization ...")

    # Flatten raw events and entities
    flat_events = [ev for evs in all_events.values() for ev in evs]
    flat_entities_by_type: Dict[str, List[Dict]] = {t: [] for t in ENTITY_TYPES}
    for scene_ents in all_entities.values():
        for ent in scene_ents:
            t = ent.get("type", "")
            if t in flat_entities_by_type:
                flat_entities_by_type[t].append(ent)

    merge_log_all = []

    if skip_normalization:
        logger.info("Skipping LLM adjudication (--skip_normalization)")
        normalized_events = flat_events
        normalized_entities = flat_entities_by_type
    else:
        # Normalize events
        normalized_events, ev_log = normalize_nodes(
            nodes=flat_events,
            node_type="Event",
            rename_map={},  # events don't use rename_map
            llm=llm,
            movie_id=movie.movie_id,
            movie_title=movie.title,
            cache=cache,
            prompt_logger=prompt_logger,
            api_key=embedding_api_key,
        )
        merge_log_all.extend(ev_log)

        # Normalize entities by type
        normalized_entities = {}
        for etype in ENTITY_TYPES:
            raw_ents = flat_entities_by_type.get(etype, [])
            if not raw_ents:
                normalized_entities[etype] = []
                continue
            norm_ents, ent_log = normalize_nodes(
                nodes=raw_ents,
                node_type=etype,
                rename_map=movie.rename_map,
                llm=llm,
                movie_id=movie.movie_id,
                movie_title=movie.title,
                cache=cache,
                prompt_logger=prompt_logger,
                api_key=embedding_api_key,
            )
            normalized_entities[etype] = norm_ents
            merge_log_all.extend(ent_log)

        # Pre/post merge stats
        for etype in ENTITY_TYPES:
            raw = flat_entities_by_type.get(etype, [])
            norm = normalized_entities.get(etype, [])
            if raw:
                ppm = compute_pre_post_merge_stats(raw, norm)
                logger.info(
                    "[%s] %s: %d raw -> %d merged (dup rate %.1f%%)",
                    movie.movie_id, etype, ppm["raw_count"], ppm["merged_count"],
                    ppm["duplicate_rate"] * 100,
                )

    _save_intermediate(merge_log_all, movie_out / "merge_log.json")

    # ------------------------------------------------------------------ Pass 5: Graph build
    logger.info("[Pass 5] Building movie-level graph ...")

    graph = build_graph_from_extractions(
        movie_id=movie.movie_id,
        title=movie.title,
        all_events=all_events,
        all_entities=all_entities,
        all_relations=all_relations,
        normalized_events=normalized_events,
        normalized_entities=normalized_entities,
    )

    # Save final graph
    final_graph_path = movie_out / "final_graph.json"
    graph.save(final_graph_path)

    # Save scene segmentation
    _save_scene_segmentation(movie, movie_out / "scene_segmentation.json")

    # Evaluation stats
    stats = compute_stats(graph.to_dict())
    print_stats(stats)
    with (movie_out / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("Pipeline complete. Output: %s", movie_out)
    return graph


# ------------------------------------------------------------------ helpers

def _save_intermediate(data, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.debug("Saved intermediate: %s", path)


def run_pipeline_per_scene(
    movie: MovieData,
    llm: BaseLLM,
    output_dir: Path,
    skip_normalization: bool = False,
    api_key: Optional[str] = None,
) -> Dict[str, "KnowledgeGraph"]:
    """
    Build one independent KG per scene, reusing the existing pipeline.

    Outputs:
        <output_dir>/<movie_id>/scene_graphs/scene_<NNN>/final_graph.json

    Returns:
        Mapping scene_id -> KnowledgeGraph.
    """
    movie_out = Path(output_dir) / movie.movie_id
    scene_graphs_root = movie_out / "scene_graphs"
    scene_graphs_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Building per-scene graphs for %s (%d scenes)",
        movie.title or movie.movie_id,
        len(movie.scenes),
    )

    results: Dict[str, "KnowledgeGraph"] = {}

    for scene in movie.scenes:
        scene_dir_name = _scene_dir_name(scene)
        scene_out_root = scene_graphs_root / scene_dir_name

        # Build a one-scene MovieData copy. Use a synthetic per-scene movie_id
        # so the existing pipeline's caching/output rooting works in isolation.
        scene_movie_id = f"{movie.movie_id}__scene_{_scene_num(scene)}"
        scene_movie = MovieData(
            movie_id=scene_movie_id,
            title=movie.title,
            language=movie.language,
            data_dir=movie.data_dir,
            scenes=[copy.deepcopy(scene)],
            rename_map=dict(movie.rename_map),
        )

        logger.info(
            "[per_scene] %s scene_id=%s title=%r",
            movie.movie_id,
            scene.scene_id,
            scene.title,
        )

        graph = run_pipeline(
            movie=scene_movie,
            llm=llm,
            output_dir=scene_graphs_root,
            skip_normalization=skip_normalization,
            api_key=api_key,
        )

        # run_pipeline wrote to scene_graphs_root/<scene_movie_id>/final_graph.json.
        # Move/rename that directory to the canonical scene_<NNN> name.
        produced_dir = scene_graphs_root / scene_movie_id
        if produced_dir != scene_out_root:
            if scene_out_root.exists():
                # Clean prior output to avoid leftover artifacts mixing in.
                import shutil

                shutil.rmtree(scene_out_root)
            produced_dir.rename(scene_out_root)

        results[scene.scene_id] = graph

    logger.info(
        "Per-scene graphs complete: %d scene graphs written under %s",
        len(results),
        scene_graphs_root,
    )
    return results


def _scene_num(scene: SceneRecord) -> int:
    try:
        return int(scene.scene_id)
    except (TypeError, ValueError):
        return scene.order + 1


def _scene_dir_name(scene: SceneRecord) -> str:
    return f"scene_{_scene_num(scene):03d}"


def _save_scene_segmentation(movie: MovieData, path: Path) -> None:
    scenes = [
        {
            "scene_id": s.scene_id,
            "order": s.order,
            "title": s.title,
            "category": s.scene_category,
            "region": s.region,
            "main_location": s.main_location,
            "summary": s.summary,
            "content_length": len(s.content),
            "num_chunks": len(s.chunks),
        }
        for s in movie.scenes
    ]
    _save_intermediate(scenes, path)
