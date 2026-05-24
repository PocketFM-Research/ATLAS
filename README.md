# ATLAS

Temporal-hallucination detection and evaluation for screenplays. Builds one knowledge graph per scene, then checks every later scene against the union of earlier scene graphs for continuity errors. Two detection methods run in parallel: a graph-grounded verifier and an LLM-as-judge baseline.

Two modes:

- **Detect.** Point it at any screenplay (`script.json`) and get back a list of suspected continuity errors from each method. No gold annotations required.
- **Evaluate.** If you also supply a gold annotations file, the pipeline scores each method's predictions against gold and reports precision / recall / F1.

## What it produces

For each story (always):

- **Per-scene knowledge graphs** under `<output>/<story_id>/scene_graphs/scene_NNN/final_graph.json`
- **Graph-verifier predictions** — `graph_method_hallucinations.json`
- **LLM-judge baseline predictions** — `llm_judge_hallucinations.json`
- **Audit trails** — `graph_method_audit.json`, `graph_method_proposals.json`, `graph_method_verification_raw.json`, `llm_judge_raw_response.txt`

Only when gold annotations are supplied:

- **Per-case metrics** — `<output>/<story_id>/comparison_metrics.json`
- **Aggregate metrics** — `<output>/comparison_metrics.csv` and `<output>/comparison_metrics.json`

Without gold, P/R/F1 columns are present but degenerate (every prediction counts as an FP since there's nothing to match against). The useful artifact in that mode is `*_hallucinations.json` itself.

Each detected hallucination is rendered as one canonical string:

```
In scene X, [FACT] was established, yet in scene Y, [EVENT] happens
```

## The pipeline

```
script.json
   │
   ▼
ingest.loader            ─►  ordered SceneRecord list
   │
   ▼
ATLAS.pipeline.run_pipeline_per_scene
   ├─ event extraction       (LLM)
   ├─ entity extraction      (LLM, anchored to events)
   ├─ relation extraction    (LLM, schema-constrained)
   ├─ normalization          (TF-IDF clustering + LLM merge adjudication)
   └─ graph build            ─►  scene_NNN/final_graph.json
   │
   ▼
ATLAS.evaluation.attribute_extractor   ─►  per-scene entity.attribute = value lines
ATLAS.evaluation.graph_verifier        ─►  scene-by-scene proposal + full-text verification
ATLAS.evaluation.llm_judge_hallucination ─►  whole-screenplay LLM judge (baseline)
   │
   ▼
ATLAS.evaluation.hallucination_metrics  ─►  match predictions against gold, score
```

The graph verifier is two phases:

1. **Proposal.** For each later scene, the LLM is given the union of prior scene graphs plus the current scene's candidate facts and emits suspected contradictions or unsupported-state-continuations.
2. **Verification.** Every proposal is rechecked against the full screenplay text in small batches. Each candidate must satisfy four conditions (same entity, same attribute, mismatched values, no on-screen justification) to be confirmed. Unconfirmed proposals are dropped.

The LLM judge baseline reads the full screenplay and emits hallucinations in one shot, without graph grounding. Both methods use the same LLM (Azure GPT-5.4 mini in the default config) so comparisons aren't confounded by model choice.

## Setup

```bash
pip install -r requirements.txt
```

Put your Azure OpenAI key in `openai.txt` and the deployment target URI in `base_url.txt` (one line each). These two files are gitignored.

The default config expects:

| Setting | Value |
|---|---|
| `--model` | `azure_openai` |
| `--model_name` | `gpt-5.4-mini` |
| `--api_version` | from `--api_version` flag, or extracted from the target URL |

### Supported LLM backends

`--model` accepts any of:

| Provider | Notes |
|---|---|
| `azure_openai` | Default. Auto-detects "Target URI" vs plain Azure-resource endpoints. |
| `openai` | Public OpenAI API. |
| `anthropic` | Claude models via `anthropic` SDK. |
| `gemini` | Google Gemini via `google-genai` SDK. |
| `vllm` | Local OpenAI-compatible endpoint (e.g. self-hosted vLLM). Pass `--base_url http://localhost:8000/v1`. |

The same `BaseLLM` instance is used for graph construction, attribute extraction, the graph verifier, and the LLM judge — comparisons aren't confounded by model swaps mid-run.

## Running

### Detect on a screenplay (no gold)

```bash
python3 run_scene_graph_hallucination_eval.py \
    --movie_dir English/my_screenplay \
    --output_dir Results/run_001 \
    --api_key_file openai.txt \
    --base_url_file base_url.txt
```

This runs both methods and writes predictions to `Results/run_001/my_screenplay/graph_method_hallucinations.json` and `llm_judge_hallucinations.json`. No scoring step.

### Evaluate against gold

Add `--annotations`:

```bash
python3 run_scene_graph_hallucination_eval.py \
    --movie_dir English/synthetic_hallucination_001 \
    --annotations English/synthetic_hallucination_001/hallucinations.json \
    --output_dir Results/run_001 \
    --api_key_file openai.txt \
    --base_url_file base_url.txt
```

`--story_id` defaults to the basename of `--movie_dir`.

### Batch via manifest

```bash
python3 run_scene_graph_hallucination_eval.py \
    --manifest manifests/eval_set.json \
    --output_dir Results/eval_set \
    --api_key_file openai.txt \
    --base_url_file base_url.txt
```

Manifest format:

```json
[
  {"story_id": "case_001", "movie_dir": "English/case_001", "annotations": "English/case_001/hallucinations.json"},
  {"story_id": "case_002", "movie_dir": "English/case_002"}
]
```

`annotations` is optional per case — omit it for stories you want to detect on without scoring. Relative paths are resolved against the manifest file's directory.

### Reuse already-built scene graphs

If you've built scene graphs in a prior run and only want to rerun the verifiers / scoring:

```bash
python3 run_scene_graph_hallucination_eval.py \
    --manifest manifests/eval_set.json \
    --output_dir Results/eval_set \
    --reuse_existing_graphs \
    --api_key_file openai.txt --base_url_file base_url.txt
```

### Re-verify proposals only

If you want to keep the existing graph-verifier proposals and only rerun the full-text verification pass (faster, cheaper iteration on prompt tuning):

```bash
python3 run_scene_graph_hallucination_eval.py \
    --manifest manifests/eval_set.json \
    --output_dir Results/eval_set \
    --verify_existing_proposals \
    --api_key_file openai.txt --base_url_file base_url.txt
```

### Skip one method

```bash
--skip_graph_method   # only run LLM-judge baseline
--skip_llm_judge      # only run graph verifier
```

### Build graphs only (no eval)

`build_graph.py` is the lower-level entry point if you want graphs without evaluation:

```bash
python3 build_graph.py \
    --input_dir . \
    --output_dir Results/graphs \
    --model azure_openai --model_name gpt-5.4-mini \
    --api_key_file openai.txt --base_url_file base_url.txt \
    --movie_ids en04052c0f20834cf1bac19927d8f758e0 \
    --per_scene
```

`--per_scene` writes one graph per scene under `scene_graphs/scene_NNN/`. Omit it to get a single movie-level graph in `final_graph.json`.

## CLI flags

### `run_scene_graph_hallucination_eval.py`

| Flag | Description |
|---|---|
| `--manifest` | JSON list of cases (alternative to single-case flags) |
| `--movie_dir` | Single case: path to a movie directory |
| `--annotations` | Single case: gold annotations JSON (optional — omit for detect-only) |
| `--story_id` | Single case: story id (defaults to movie_dir name) |
| `--output_dir` | Output root (required) |
| `--model` | LLM provider; default `azure_openai` |
| `--model_name` | Model name; default `gpt-5.4-mini` |
| `--api_key` / `--api_key_file` | API key (string or file) |
| `--base_url` / `--base_url_file` | Azure deployment target URL |
| `--api_version` | Azure API version (auto-detected from URL if omitted) |
| `--max_scenes` | Cap scenes per movie (for quick test runs) |
| `--skip_normalization` | Skip the LLM merge-adjudication step in graph build |
| `--reuse_existing_graphs` | Don't rebuild scene graphs if they're already present |
| `--verify_existing_proposals` | Rerun only the full-text verifier on existing proposals |
| `--skip_graph_method` | Don't run the graph verifier |
| `--skip_llm_judge` | Don't run the LLM-judge baseline |
| `--match_threshold` | Jaccard threshold for gold-vs-prediction matching |
| `--language` | `en` or `zh` |
| `--verbose` | Debug logging |

### `build_graph.py`

| Flag | Description |
|---|---|
| `--input_dir` | Dataset root |
| `--output_dir` | Output root |
| `--model` / `--model_name` | LLM provider and model |
| `--api_key` / `--api_key_file` | API key |
| `--base_url` / `--base_url_file` | Endpoint URL (Azure or local vLLM) |
| `--api_version` | Azure API version |
| `--language` | `en` or `zh` |
| `--movie_ids` | Restrict to specific movie IDs |
| `--max_scenes` | Cap scenes per movie |
| `--per_scene` | Write per-scene graphs instead of one movie-level graph |
| `--skip_normalization` | Skip LLM merge adjudication |
| `--verbose` | Debug logging |

## Input format

Each movie directory needs at minimum:

```
<movie_dir>/
├── script.json           # scene-segmented screenplay (required)
├── doc2chunks.json       # pre-computed chunking (optional)
└── rename_map.json       # entity alias hints (optional)
```

`script.json` is a list of scenes:

```json
[
  {"_id": 1, "title": "1、INT. ENTERPRISE BRIDGE - DAY", "subtitle": "", "content": "..."},
  {"_id": 2, "title": "2、EXT. SHIP - NIGHT", "subtitle": "", "content": "..."}
]
```

Gold annotations are optional. When supplied, they're a list of hallucination strings in the canonical template:

```json
[
  {"id": 1, "hallucination": "In scene 2, the gun in Marcus's left hand was established, yet in scene 7, Marcus draws the gun with his right hand happens"},
  {"id": 2, "hallucination": "In scene 4, Sarah being a widow was established, yet in scene 11, Sarah's husband visits her at home happens"}
]
```

## Output layout

```
<output_dir>/
├── comparison_metrics.csv          # one row per (story, method)
├── comparison_metrics.json
└── <story_id>/
    ├── scene_graphs/
    │   └── scene_NNN/final_graph.json
    ├── attributes/
    │   └── scene_NNN_attributes.json     # per-scene durable-attribute profiles
    ├── graph_method_hallucinations.json  # final graph-verifier predictions
    ├── graph_method_proposals.json       # pre-verification proposals
    ├── graph_method_verification_raw.json
    ├── graph_method_audit.json           # full decision trail
    ├── llm_judge_hallucinations.json     # final LLM-judge predictions
    ├── llm_judge_raw_response.txt
    └── comparison_metrics.json           # per-case scores
```

## Graph schema

**Node types:** `Character`, `Event`, `Location`, `TimePoint`, `Object`, `Concept`

**Relation categories:**

| Category | Relations |
|---|---|
| Event-Role | `performs`, `undergoes`, `experiences` |
| Social | `kinship_with`, `affinity_with`, `hostility_with`, `affiliated_with` |
| Inter-Event | `before`, `after`, `causes`, `precedes`, `elaborates`, `contrasts_with` |
| Spatiotemporal | `occurs_at`, `located_at` |
| Object | `owns`, `uses`, `part_of` |
| Semantic | `instance_of`, `related_to`, `refers_to` |

All edges are schema-checked; invalid `(source_type, relation, target_type)` triples are rejected at build time.

## Repository layout

```
ATLAS/
├── ingest/loader.py              # screenplay loading
├── pipeline.py                   # run_pipeline / run_pipeline_per_scene
├── extraction/                   # event, entity, relation passes
├── normalization/                # TF-IDF + spectral clustering + LLM adjudication
├── graph/builder.py              # movie-level KG consolidation
├── evaluation/
│   ├── attribute_extractor.py    # per-scene entity-anchored attributes
│   ├── graph_verifier.py         # proposal + full-text verification
│   ├── llm_judge_hallucination.py # baseline
│   ├── hallucination_metrics.py  # match + P/R/F1
│   └── stats.py                  # intrinsic graph stats
├── llm/                          # openai, azure_openai, anthropic, gemini, vllm backends
├── prompts/                      # extraction prompt templates
├── utils/                        # cache, json repair, logging
└── schema.py                     # node + relation types, valid-triple table

run_scene_graph_hallucination_eval.py   # eval entry point
build_graph.py                          # graph-build-only entry point
visualize_graph.py                      # interactive HTML graph viewer
```

## Scoring

`ATLAS/evaluation/hallucination_metrics.py` matches predictions against gold without calling the LLM. For each (gold, prediction) pair:

1. Parse the scene numbers X (prior) and Y (later) out of both strings using the canonical template.
2. Require the (X, Y) pair to match. Predictions on different scene pairs cannot collapse into the same gold.
3. Compute token-Jaccard similarity on the *fact* clause and on the *event* clause separately, after lowercasing and dropping stopwords.
4. A prediction matches a gold if **both** Jaccard scores meet `--match_threshold` (default tuned for paraphrase tolerance without false collisions).

Matching is greedy: each gold can be claimed by at most one prediction. Unmatched golds become false negatives; unmatched predictions become false positives. Per-case rows include:

| Column | Meaning |
|---|---|
| `story_id` | Case identifier |
| `method` | `graph_verifier` or `llm_judge` |
| `Gold` | Number of gold annotations |
| `Pred.` | Number of predictions emitted |
| `TP` / `FP` / `FN` | Match counts |
| `precision`, `recall`, `F1` | Standard formulas; zero when both sides are empty |

An aggregate row per method micro-averages TP/FP/FN across stories.

## Debugging a run

If a run produces a surprising score, the audit files give a full decision trail:

- `graph_method_audit.json` — `per_scene_proposal_stats` (how many candidates the LLM saw per scene, how many it flagged), `raw_proposals` (every flagged item before dedupe), `dedupe_log` (which proposals collapsed and why), `verification_log` (the LLM's confirm/reject decision plus reason on every proposal), `summary` (overall counts).
- `graph_method_proposals.json` — proposals after dedupe but before full-text verification.
- `graph_method_verification_raw.json` — raw LLM JSON from each verification batch.
- `llm_judge_raw_response.txt` — raw LLM output before parsing / template-matching.
- `attributes/scene_NNN_attributes.json` — what the attribute extractor recovered for each scene (entity → attribute → value). The graph verifier folds these into the prior-scene summary.

The metrics file shows P/R/F1; the audit file shows *why*.

## Visualization

`visualize_graph.py` renders a knowledge graph as interactive HTML using D3.js force-directed layout. One page per scene plus a full-graph page:

```bash
python3 visualize_graph.py --graph Results/run_001/case_001/scene_graphs/scene_007/final_graph.json
```

Output goes to `<graph_dir>/viz/`. Useful for spot-checking what the extractor pulled out of a given scene.

## Caching and resumability

All LLM calls are cached to `<output>/<story_id>/.cache/` keyed by content + a `SCHEMA_VERSION` field. Interrupted runs resume automatically. Bump `SCHEMA_VERSION` in `ATLAS/utils/cache.py` when changing prompts or schema if you want stale entries to invalidate.

Every prompt and raw LLM response is logged to `<output>/<story_id>/logs/prompts_*.jsonl` for audit / replay.

## Hallucination string template

Every gold and predicted hallucination follows one canonical template:

```
In scene X, [FACT] was established, yet in scene Y, [EVENT] happens
```

- `X` and `Y` are integers, both > 0, and `X < Y` (the contradicting event must come after the established fact).
- `[FACT]` is a short clause (≤15 words) describing the earlier-established attribute or state.
- `[EVENT]` is a short clause (≤15 words) describing what happens in the later scene that contradicts or presupposes that state.

The parser also accepts a lenient form (without the literal "was established" / "happens" suffixes) since LLMs frequently drop those words even when prompted not to. Strings that don't fit either form are silently dropped from the predictions list.

## License

This project is licensed under the MIT License.

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)