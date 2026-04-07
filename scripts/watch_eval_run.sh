#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EVAL_DIR="${1:-${EVAL_DIR:-eval_results}}"
INTERVAL="${INTERVAL:-5}"
STALE_AFTER="${STALE_AFTER:-120}"

latest_log() {
  ls -t "$EVAL_DIR"/eval_run_*.log 2>/dev/null | head -n 1
}

eval_stage() {
  if [[ -f "$EVAL_DIR/summary.json" ]]; then
    echo "summary.json present"
  elif [[ -f "$EVAL_DIR/hallucination_details.csv" || -f "$EVAL_DIR/consistency_violations.csv" || -f "$EVAL_DIR/repetition_instances.csv" ]]; then
    echo "evaluation outputs partial"
  elif find "$EVAL_DIR" -maxdepth 1 -type f -name '*_descriptions.json' | grep -q .; then
    echo "descriptions only (eval incomplete)"
  elif [[ -d "$EVAL_DIR" ]]; then
    echo "eval dir exists"
  else
    echo "not started"
  fi
}

while true; do
  clear
  echo "Eval Run Watch"
  echo "Root: $ROOT_DIR"
  echo "Eval dir: $EVAL_DIR"
  echo

  log_file="$(latest_log || true)"
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
    tail -n 12 "$log_file" 2>/dev/null || true
  else
    echo "Latest log: none"
    echo "Status: no eval log found"
  fi

  echo
  echo "Eval stage: $(eval_stage)"
  echo
  echo "Top-level files:"
  find "$EVAL_DIR" -maxdepth 1 -type f | sort

  echo
  echo "Tip: if Last update keeps climbing past ${STALE_AFTER}s and summary.json does not appear, the eval likely died."
  sleep "$INTERVAL"
done
