# STAGE Dataset

**STAGE: A Benchmark for Knowledge Graph Construction, Question Answering, and In-Script Role-Playing over Movie Screenplays**

A bilingual (Chinese and English) benchmark for narrative understanding over movie screenplays.

## Dataset Overview

STAGE contains structured annotations for understanding complex narratives in both Chinese and English films. The dataset includes:

- **Chinese**: 42 Chinese movie scripts with full annotations
- **English**: 109 English movie scripts with full annotations
- **Metadata**: `chinese_movie_info.csv` and `english_movie_info.csv` with movie-level information (title, word count, number of scenes, genres)

## Directory Structure

Each movie directory contains:

| File / Directory | Description |
|---|---|
| `script.json` | Scene-by-scene segmented movie script |
| `episodes.json` | Episode-level narrative units with descriptions and related events |
| `episde_relations.json` | Pairwise temporal/causal relations between episodes with confidence scores |
| `episde_causality_graph.json` | Weighted causality graph over episodes |
| `extraction_results.json` | Per-scene entity and event extraction results |
| `doc2chunks.json` | Document-to-chunk mapping with full text and metadata |
| `rename_map.json` | Entity name normalization/alias map |
| `question_pairs.csv` | Multi-hop question-answer pairs with evidence and QA type annotations |
| `ICRP/` | In-Character Role-Playing data (per-character subdirectories) |

### ICRP (In-Character Role-Playing)

Each movie's `ICRP/` directory contains one subfolder per character (typically 3 characters per movie), with:

| File | Description |
|---|---|
| `persona_card.json` | Character traits, speaking style, behavioral constraints, and dialogue exemplars |
| `key_relations.json` | Key relationships with other characters |
| `icrp_qa.json` | Role-playing QA pairs with supporting/contradicting facts |
| `actions.csv` | Character's physical actions per scene |
| `dialogues.csv` | Character's dialogue lines per scene |
| `facts.csv` | Factual statements about the character per scene |

## Data Format

### script.json
```json
[
  {
    "_id": 1,
    "title": "Scene Title",
    "subtitle": "Scene Subtitle",
    "content": "Scene content with dialogue and actions..."
  }
]
```

### episodes.json
```json
[
  {
    "id": "ep_4f8f0b6d18e73a34",
    "name": "Episode Title",
    "description": "Narrative description of the episode",
    "source_documents": ["scene_7_part_1"],
    "related_events": ["event description 1"],
    "related_occasions": ["occasion description"]
  }
]
```

### episde_relations.json
```json
[
  {
    "id": "rel_ep_ep_8f935bc28619",
    "subject_id": "ep_...",
    "object_id": "ep_...",
    "relation_type": "precedes",
    "reason": "Natural language justification",
    "source_documents": ["scene_11_part_1"],
    "confidence": 0.95
  }
]
```

### episde_causality_graph.json
```json
[
  {
    "subject_id": "ep_...",
    "object_id": "ep_...",
    "relation_type": "CAUSAL_LINK",
    "original_relation_type": "precedes",
    "weight": 0.7,
    "effective_weight": 0.595
  }
]
```

### question_pairs.csv

| Column | Description |
|---|---|
| `id` | Question ID |
| `scene` | Related scene |
| `question` | Question text |
| `answer` | Answer text |
| `evidence` | Supporting evidence from the script |
| `qa_type` | QA type (e.g., `character states`, `causal/relational queries`, `dialogue/beliefs`, `detailed description`, `temporal references`) |

## Publication Status

**Published as arXiv preprint**

## Citation

If you use this dataset in your research, please cite our paper:

```bibtex
@article{tian2026stage,
  title={STAGE: A Benchmark for Knowledge Graph Construction, Question Answering, and In-Script Role-Playing over Movie Screenplays},
  author={Tian, Qiuyu and Li, Yiding and Chen, Fengyi and Liu, Zequn and Kong, Youyong and Guo, Fan and Li, Yuyao and Shen, Jinjing and Xie, Zhijing and Luo, Yiyun and others},
  journal={arXiv preprint arXiv:2601.08510},
  year={2026}
}
```

## Contact

For questions or issues regarding the dataset, please open an issue in this repository.
