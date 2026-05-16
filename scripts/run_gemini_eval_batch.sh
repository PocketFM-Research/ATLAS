#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

API_KEY_FILE="${API_KEY_FILE:-gemini.txt}"
MODEL_PROVIDER="${MODEL_PROVIDER:-gemini}"
MODEL_NAME="${MODEL_NAME:-gemini-2.5-flash}"
GRAPH_DIR="${GRAPH_DIR:-output_gemini_batch}"
MOVIE_ROOT="${MOVIE_ROOT:-movie_results}"
EVAL_ROOT="${EVAL_ROOT:-$MOVIE_ROOT}"
TEXT_SOURCE="${TEXT_SOURCE:-scene_script}"
DATASET_DIR="${DATASET_DIR:-.}"
MAX_SCENES="${MAX_SCENES:-}"
SKIP_NORMALIZATION="${SKIP_NORMALIZATION:-0}"
BUILD_RETRIES="${BUILD_RETRIES:-20}"
EVAL_RETRIES="${EVAL_RETRIES:-3}"
RETRY_SLEEP="${RETRY_SLEEP:-15}"
STATUS_FILE="${STATUS_FILE:-$GRAPH_DIR/.batch_status}"
EVAL_ENV_PREFIX=(
  OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1
  VECLIB_MAXIMUM_THREADS=1
  KMP_DUPLICATE_LIB_OK=TRUE
  KMP_INIT_AT_FORK=FALSE
)

if [[ ! -f "$API_KEY_FILE" ]]; then
  echo "Missing API key file: $API_KEY_FILE" >&2
  exit 1
fi

mkdir -p "$GRAPH_DIR" "$EVAL_ROOT"

write_status() {
  local movie_id="${1:-}"
  local title="${2:-}"
  local phase="${3:-idle}"
  local attempt="${4:-0}"
  local graph_path="${5:-}"
  local eval_dir="${6:-}"
  local note="${7:-}"
  {
    printf 'CURRENT_MOVIE_ID=%q\n' "$movie_id"
    printf 'CURRENT_TITLE=%q\n' "$title"
    printf 'CURRENT_PHASE=%q\n' "$phase"
    printf 'CURRENT_ATTEMPT=%q\n' "$attempt"
    printf 'CURRENT_GRAPH_PATH=%q\n' "$graph_path"
    printf 'CURRENT_EVAL_DIR=%q\n' "$eval_dir"
    printf 'CURRENT_NOTE=%q\n' "$note"
  } > "$STATUS_FILE"
}

on_exit() {
  local exit_code=$?
  if [[ "$exit_code" -ne 0 ]]; then
    write_status "${movie_id:-}" "${title:-}" "failed" "${CURRENT_ATTEMPT:-0}" "${graph_path:-}" "${eval_dir:-}" "Batch exited with code ${exit_code}"
  fi
}

trap on_exit EXIT

write_status "" "" "starting" "0" "" "" "Batch starting"

sync_graph_artifacts() {
  local movie_id="$1"
  local graph_src_dir="$GRAPH_DIR/$movie_id"
  local graph_dst_dir="$MOVIE_ROOT/$movie_id/graph"
  mkdir -p "$graph_dst_dir"
  find "$graph_src_dir" -maxdepth 1 -type f ! -name '*.log' -exec cp {} "$graph_dst_dir/" \;
}

MOVIES=(
  "en6277019821914a26ba881ef5d254969f|The Fighter"
  "en3b3b6db8683b4f509171b4b097837dbd|Platinum Blonde"
  "enbbd7083882674c1fa0c668d4d7331e22|Alien: Resurrection"
  "en44fd740e98fd44c186677b99ad139c57|Fargo"
  "en8dc9b5e23f324c86a3593ba801f672ac|Intolerable Cruelty"
)

for item in "${MOVIES[@]}"; do
  movie_id="${item%%|*}"
  title="${item#*|}"

  echo
  echo "============================================================"
  echo "Movie: $title ($movie_id)"
  echo "============================================================"

  graph_path="$GRAPH_DIR/$movie_id/final_graph.json"
  movie_root="$MOVIE_ROOT/$movie_id"
  eval_dir="$movie_root/eval"
  graph_export_dir="$movie_root/graph"
  summary_path="$eval_dir/summary.json"

  if [[ ! -f "$graph_path" ]]; then
    build_cmd=(
      python3 build_graph.py
      --input_dir "$DATASET_DIR"
      --output_dir "$GRAPH_DIR"
      --model "$MODEL_PROVIDER"
      --model_name "$MODEL_NAME"
      --api_key_file "$API_KEY_FILE"
      --movie_ids "$movie_id"
    )

    if [[ -n "$MAX_SCENES" ]]; then
      build_cmd+=(--max_scenes "$MAX_SCENES")
    fi

    if [[ "$SKIP_NORMALIZATION" == "1" ]]; then
      build_cmd+=(--skip_normalization)
    fi

    build_attempt=1
    while [[ ! -f "$graph_path" && "$build_attempt" -le "$BUILD_RETRIES" ]]; do
      write_status "$movie_id" "$title" "graph" "$build_attempt" "$graph_path" "$eval_dir" "Building graph"
      echo "Building graph... attempt $build_attempt/$BUILD_RETRIES"
      CURRENT_ATTEMPT="$build_attempt"
      "${build_cmd[@]}"

      if [[ -f "$graph_path" ]]; then
        sync_graph_artifacts "$movie_id"
        write_status "$movie_id" "$title" "graph_done" "$build_attempt" "$graph_path" "$eval_dir" "Graph build completed"
        echo "Graph build completed: $graph_path"
        break
      fi

      echo "Graph build did not produce final_graph.json for $title ($movie_id)." >&2
      if [[ "$build_attempt" -ge "$BUILD_RETRIES" ]]; then
        echo "Exhausted build retries for $title ($movie_id)." >&2
        exit 1
      fi

      echo "Sleeping ${RETRY_SLEEP}s before retry..." >&2
      write_status "$movie_id" "$title" "graph_retry_wait" "$build_attempt" "$graph_path" "$eval_dir" "Graph missing after build attempt"
      sleep "$RETRY_SLEEP"
      build_attempt=$((build_attempt + 1))
    done
  else
    sync_graph_artifacts "$movie_id"
    write_status "$movie_id" "$title" "graph_done" "0" "$graph_path" "$eval_dir" "Graph already exists"
    echo "Graph already exists at $graph_path"
  fi

  if [[ -f "$graph_path" ]]; then
    mkdir -p "$eval_dir" "$graph_export_dir"
    eval_attempt=1
    while [[ ! -f "$summary_path" && "$eval_attempt" -le "$EVAL_RETRIES" ]]; do
      write_status "$movie_id" "$title" "eval" "$eval_attempt" "$graph_path" "$eval_dir" "Running evaluation"
      echo "Running evaluation... attempt $eval_attempt/$EVAL_RETRIES"
      CURRENT_ATTEMPT="$eval_attempt"
      env "${EVAL_ENV_PREFIX[@]}" python3 eval_experiments.py \
        --movie_ids "$movie_id" \
        --dataset_dir "$DATASET_DIR" \
        --graph_dir "$GRAPH_DIR" \
        --output_dir "$eval_dir" \
        --text_source "$TEXT_SOURCE" \
        --refresh_descriptions

      if [[ -f "$summary_path" ]]; then
        write_status "$movie_id" "$title" "eval_done" "$eval_attempt" "$graph_path" "$eval_dir" "Evaluation completed"
        echo "Evaluation completed: $summary_path"
        break
      fi

      echo "Evaluation did not produce summary.json for $title ($movie_id)." >&2
      if [[ "$eval_attempt" -ge "$EVAL_RETRIES" ]]; then
        echo "Exhausted evaluation retries for $title ($movie_id)." >&2
        exit 1
      fi

      echo "Sleeping ${RETRY_SLEEP}s before retry..." >&2
      write_status "$movie_id" "$title" "eval_retry_wait" "$eval_attempt" "$graph_path" "$eval_dir" "Evaluation missing summary.json"
      sleep "$RETRY_SLEEP"
      eval_attempt=$((eval_attempt + 1))
    done
  else
    write_status "$movie_id" "$title" "failed" "0" "$graph_path" "$eval_dir" "Graph missing before evaluation"
    echo "Skipping evaluation because graph is missing: $graph_path" >&2
    exit 1
  fi

  write_status "$movie_id" "$title" "movie_done" "0" "$graph_path" "$eval_dir" "Movie graph + evaluation complete"
done

write_status "" "" "finished" "0" "" "" "Batch finished successfully"
trap - EXIT
echo
echo "Batch finished."
echo "Graphs: $GRAPH_DIR"
echo "Evaluations: $EVAL_ROOT"
