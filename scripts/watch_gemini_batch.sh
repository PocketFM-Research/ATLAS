#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BATCH_DIR="${BATCH_DIR:-output_gemini_batch}"
MOVIE_ROOT="${MOVIE_ROOT:-movie_results}"
EVAL_ROOT="${EVAL_ROOT:-$MOVIE_ROOT}"
INTERVAL="${INTERVAL:-5}"
STALE_AFTER="${STALE_AFTER:-120}"
STATUS_FILE="${STATUS_FILE:-$BATCH_DIR/.batch_status}"

movies=(
  "en6277019821914a26ba881ef5d254969f|The Fighter"
  "en3b3b6db8683b4f509171b4b097837dbd|Platinum Blonde"
  "enbbd7083882674c1fa0c668d4d7331e22|Alien: Resurrection"
  "en44fd740e98fd44c186677b99ad139c57|Fargo"
  "en8dc9b5e23f324c86a3593ba801f672ac|Intolerable Cruelty"
)

latest_log() {
  ls -t "$BATCH_DIR"/logs/run_*.log 2>/dev/null | head -n 1
}

latest_eval_log() {
  local eval_dir="$1"
  ls -t "$eval_dir"/eval_run_*.log 2>/dev/null | head -n 1
}

load_status() {
  CURRENT_MOVIE_ID=""
  CURRENT_TITLE=""
  CURRENT_PHASE=""
  CURRENT_ATTEMPT=""
  CURRENT_GRAPH_PATH=""
  CURRENT_EVAL_DIR=""
  CURRENT_NOTE=""
  if [[ -f "$STATUS_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$STATUS_FILE"
  fi
}

file_stage() {
  local movie_id="$1"
  local movie_dir="$MOVIE_ROOT/$movie_id/graph"
  if [[ ! -d "$movie_dir" ]]; then
    movie_dir="$BATCH_DIR/$movie_id"
  fi
  if [[ -f "$movie_dir/final_graph.json" ]]; then
    echo "done"
  elif [[ -f "$movie_dir/merge_log.json" ]]; then
    echo "normalizing"
  elif [[ -f "$movie_dir/raw_relations.json" ]]; then
    echo "relations_done"
  elif [[ -f "$movie_dir/raw_entities.json" ]]; then
    echo "entity_done"
  elif [[ -f "$movie_dir/raw_events.json" ]]; then
    echo "event_done"
  elif [[ -d "$movie_dir/.cache" ]]; then
    echo "started"
  else
    echo "not_started"
  fi
}

pretty_stage() {
  case "$1" in
    done) echo "final_graph.json present" ;;
    normalizing) echo "normalization/build stage" ;;
    relations_done) echo "raw_relations.json present" ;;
    entity_done) echo "raw_entities.json present" ;;
    event_done) echo "raw_events.json present" ;;
    started) echo ".cache exists only" ;;
    not_started) echo "not started" ;;
    *) echo "$1" ;;
  esac
}

eval_stage() {
  local movie_id="$1"
  local eval_dir="$EVAL_ROOT/$movie_id/eval"
  if [[ ! -d "$eval_dir" ]]; then
    eval_dir="$EVAL_ROOT/$movie_id"
  fi
  if [[ -f "$eval_dir/summary.json" ]]; then
    echo "summary.json present"
  elif [[ -f "$eval_dir/hallucination_details.csv" || -f "$eval_dir/consistency_violations.csv" || -f "$eval_dir/repetition_instances.csv" ]]; then
    echo "evaluation outputs partial"
  elif [[ -f "$eval_dir/${movie_id}_scene_script_descriptions.json" ]]; then
    echo "descriptions only (eval incomplete)"
  elif [[ -d "$eval_dir" ]]; then
    echo "eval dir exists"
  else
    echo "not started"
  fi
}

while true; do
  clear
  load_status
  echo "Gemini Batch Watch"
  echo "Root: $ROOT_DIR"
  echo "Batch dir: $BATCH_DIR"
  echo "Movie root: $MOVIE_ROOT"
  echo "Eval root: $EVAL_ROOT"
  echo

  log_file=""
  if [[ "${CURRENT_PHASE:-}" == "eval" || "${CURRENT_PHASE:-}" == "eval_retry_wait" || "${CURRENT_PHASE:-}" == "eval_done" ]]; then
    if [[ -n "${CURRENT_EVAL_DIR:-}" ]]; then
      log_file="$(latest_eval_log "$CURRENT_EVAL_DIR" || true)"
    fi
  fi
  if [[ -z "${log_file:-}" ]]; then
    log_file="$(latest_log || true)"
  fi

  echo "Current movie: ${CURRENT_TITLE:-none} ${CURRENT_MOVIE_ID:+($CURRENT_MOVIE_ID)}"
  echo "Current phase: ${CURRENT_PHASE:-unknown}"
  echo "Current note: ${CURRENT_NOTE:-}"
  if [[ -n "${CURRENT_ATTEMPT:-}" && "${CURRENT_ATTEMPT:-0}" != "0" ]]; then
    echo "Current attempt: ${CURRENT_ATTEMPT}"
  fi
  echo

  if [[ -n "${log_file:-}" && -f "$log_file" ]]; then
    now_epoch="$(date +%s)"
    if stat_out="$(stat -f '%m' "$log_file" 2>/dev/null)"; then
      log_mtime="$stat_out"
    else
      log_mtime="$(python3 - <<'PY' "$log_file"
from pathlib import Path
import sys
print(int(Path(sys.argv[1]).stat().st_mtime))
PY
)"
    fi
    age=$((now_epoch - log_mtime))
    echo "Latest log: $log_file"
    echo "Last update: ${age}s ago"
    if (( age > STALE_AFTER )); then
      echo "Status: STALE / likely stopped"
    else
      echo "Status: active recently"
    fi
    echo
    echo "Recent log lines:"
    tail -n 8 "$log_file" 2>/dev/null || true
  else
    echo "Latest log: none"
    echo "Status: no run logs found"
  fi

  echo
  echo "Movie stages:"
  for item in "${movies[@]}"; do
    movie_id="${item%%|*}"
    title="${item#*|}"
    graph_stage="$(file_stage "$movie_id")"
    movie_eval_stage="$(eval_stage "$movie_id")"
    printf '  %-36s  %-24s  graph: %-32s  eval: %s\n' \
      "$movie_id" "$title" "$(pretty_stage "$graph_stage")" "$movie_eval_stage"
  done

  echo
  echo "Tip: if Last update keeps climbing past ${STALE_AFTER}s and no new stage files appear, the run likely died."
  sleep "$INTERVAL"
done
