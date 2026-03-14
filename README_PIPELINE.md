# STAGE Knowledge Graph Extraction Pipeline

Research-faithful reproduction of the screenplay-to-knowledge-graph pipeline from the paper:

> **STAGE: A Benchmark for Knowledge Graph Construction, Question Answering, and In-Script Role-Playing over Movie Screenplays**
> Tian et al., arXiv:2601.08510

## What this pipeline does

Processes movie screenplays from the STAGE dataset and produces structured knowledge graphs:

```
screenplay text → events → entities → relations → normalized graph
```

It does **not** implement story generation, QA, or role-playing.

---

## Project structure

```
stage_kg/
├── schema.py                   # Node types and relation schema
├── pipeline.py                 # Main orchestrator
├── ingest/
│   └── loader.py               # Load script.json, doc2chunks.json, rename_map.json
├── extraction/
│   ├── event_extractor.py      # Pass 1: events
│   ├── entity_extractor.py     # Pass 2: entities (anchored to events)
│   └── relation_extractor.py   # Pass 3: typed relations (schema-constrained)
├── normalization/
│   └── normalizer.py           # Dedup + LLM merge adjudication
├── graph/
│   └── builder.py              # Movie-level graph consolidation + serialization
├── evaluation/
│   └── stats.py                # Intrinsic statistics
├── prompts/
│   ├── event_extraction.py
│   ├── entity_extraction.py
│   ├── relation_extraction.py
│   └── merge_adjudication.py
├── llm/
│   ├── base.py
│   ├── openai_client.py        # OpenAI backend
│   ├── anthropic_client.py     # Anthropic backend
│   └── vllm_client.py          # Local vLLM endpoint
└── utils/
    ├── cache.py                # Disk cache (enables resumable runs)
    ├── json_repair.py          # LLM JSON validation + repair
    └── logging_utils.py        # Logging + prompt logging

build_graph.py                  # CLI entry point
scripts/run_example.py          # Quick example run
requirements.txt
```

---

## Setup

```bash
pip install anthropic openai tqdm
```

---

## Running the pipeline

### Process one movie (Star Trek II, first 5 scenes):

```bash
ANTHROPIC_API_KEY=sk-ant-... python build_graph.py \
    --input_dir . \
    --output_dir ./output \
    --model anthropic \
    --movie_ids en04052c0f20834cf1bac19927d8f758e0 \
    --max_scenes 5
```

### Process one movie with OpenAI:

```bash
OPENAI_API_KEY=sk-... python build_graph.py \
    --input_dir . \
    --output_dir ./output \
    --model openai \
    --model_name gpt-4o \
    --movie_ids en04052c0f20834cf1bac19927d8f758e0
```

### Process all English movies:

```bash
ANTHROPIC_API_KEY=sk-ant-... python build_graph.py \
    --input_dir . \
    --output_dir ./output \
    --model anthropic \
    --language en
```

### Use local vLLM:

```bash
python build_graph.py \
    --input_dir . \
    --output_dir ./output \
    --model vllm \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --base_url http://localhost:8000/v1
```

### Run example script:

```bash
ANTHROPIC_API_KEY=sk-ant-... python scripts/run_example.py --max_scenes 3
```

---

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--input_dir` | `.` | STAGE dataset root |
| `--output_dir` | `./output` | Output directory |
| `--model` | `anthropic` | LLM provider: `openai`, `anthropic`, `vllm` |
| `--model_name` | provider default | Model name/ID |
| `--language` | `en` | `en` or `zh` |
| `--movie_ids` | all | Process only these movie IDs |
| `--max_scenes` | all | Limit scenes per movie (for testing) |
| `--skip_normalization` | false | Skip LLM merge adjudication |
| `--verbose` | false | Debug logging |

---

## Output files (per movie)

```
output/<movie_id>/
├── final_graph.json          # Final merged knowledge graph
├── raw_events.json           # Events before normalization {scene_id -> [events]}
├── raw_entities.json         # Entities before normalization {scene_id -> [entities]}
├── raw_relations.json        # Relations before normalization {scene_id -> [relations]}
├── scene_segmentation.json   # Scene metadata and ordering
├── merge_log.json            # Merge adjudication decisions
├── stats.json                # Intrinsic graph statistics
└── logs/
    ├── run_*.log             # Run log
    └── prompts_*.jsonl       # All prompts + raw LLM responses (for reproducibility)
```

---

## Final graph format

```json
{
  "movie_id": "en04052c0f20834cf1bac19927d8f758e0",
  "title": "Star Trek II: The Wrath of Khan",
  "nodes": [
    {
      "id": "ev_en04052c_1_ev_001",
      "type": "Event",
      "name": "Saavik assumes command of the bridge",
      "aliases": ["Saavik assumes command of the bridge"],
      "description": "Saavik takes command as acting captain.",
      "scene_refs": ["1"],
      "evidence": ["Lieutenant Saavik, commanding"]
    },
    {
      "id": "ent_en04052c_1_ent_001",
      "type": "Character",
      "name": "Saavik",
      "aliases": ["Saavik", "Captain Saavik", "Lieutenant Saavik"],
      "description": "A young half-Vulcan officer.",
      "scene_refs": ["1"],
      "evidence": ["Lieutenant Saavik, commanding"]
    }
  ],
  "edges": [
    {
      "id": "e_7ea000a87d25",
      "source": "ent_en04052c_1_ent_001",
      "relation": "performs",
      "target": "ev_en04052c_1_ev_001",
      "scene_refs": ["1"],
      "evidence": ["Lieutenant Saavik, commanding"],
      "confidence": 0.95
    }
  ]
}
```

---

## Graph schema

**Node types:** `Character`, `Event`, `Location`, `TimePoint`, `Object`, `Concept`

**Relation categories and labels:**

| Category | Relations |
|----------|-----------|
| Event-Role | `performs`, `undergoes`, `experiences` |
| Social | `kinship_with`, `affinity_with`, `hostility_with`, `affiliated_with` |
| Inter-Event | `before`, `after`, `causes`, `precedes`, `elaborates`, `contrasts_with` |
| Spatiotemporal | `occurs_at`, `located_at` |
| Object-related | `owns`, `uses`, `part_of` |
| Semantic | `instance_of`, `related_to`, `refers_to` |

All edges are schema-constrained: invalid `(source_type, relation, target_type)` triples are rejected.

---

## Pipeline architecture

The pipeline follows the STAGE paper's extraction order:

```
1. Event extraction   → LLM extracts events from each scene/chunk
2. Entity extraction  → LLM extracts entities anchored to event inventory
3. Relation extraction → LLM extracts typed edges (schema-constrained)
4. Normalization      → String clustering + LLM merge adjudication
5. Graph build        → Deduplicate + cross-scene merge + serialize
```

**Caching:** All LLM calls are cached to `output/<movie_id>/.cache/`. Interrupted runs resume automatically.

**Reproducibility:** Every prompt and raw LLM response is logged to `output/<movie_id>/logs/prompts_*.jsonl`.

---

## Intrinsic evaluation

The pipeline automatically computes and saves `stats.json` for each movie:

- Node count by type
- Edge count by relation
- Evidence coverage (% nodes/edges with grounding text)
- Schema violation count (should be 0)
- Average degree
- Disconnected component count
- Duplicate rate before vs after normalization
- Average edge confidence

---

## TODO / underspecified by paper

- `TODO [paper underspecified]` The paper does not detail the exact chunking strategy for very long scenes beyond basic segmentation — current implementation uses dataset's pre-computed `doc2chunks.json` when available, otherwise falls back to the full scene content.
- `TODO [paper underspecified]` The embedding model used for similarity clustering in normalization is not specified. Current implementation uses string edit-distance (difflib). Swap `_similarity()` in `normalizer.py` for a sentence-embedding cosine similarity for better results.
- `TODO [paper underspecified]` The threshold for auto-merge vs LLM-adjudication is set empirically (0.80 / 0.95). Paper does not specify exact values.
- `TODO [eval]` Paper QA, event summarization scoring, and role-play evaluation are not implemented.
