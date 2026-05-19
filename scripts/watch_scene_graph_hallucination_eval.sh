#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${1:-${OUTPUT_DIR:-movie_results/synthetic_hallucination_batch}}"
MANIFEST="${MANIFEST:-synthetic_hallucination_cases.json}"
INTERVAL="${INTERVAL:-5}"
STALE_AFTER="${STALE_AFTER:-180}"

latest_log() {
  ls -t "$OUTPUT_DIR"/logs/run_*.log 2>/dev/null | head -n 1
}

mtime_epoch() {
  local path="$1"
  if stat -f '%m' "$path" >/dev/null 2>&1; then
    stat -f '%m' "$path"
  else
    python3 - <<'PY' "$path"
from pathlib import Path
import sys
print(int(Path(sys.argv[1]).stat().st_mtime))
PY
  fi
}

manifest_rows() {
  python3 - <<'PY' "$MANIFEST"
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
if not manifest.exists():
    raise SystemExit(0)

for entry in json.loads(manifest.read_text(encoding="utf-8")):
    story_id = entry["story_id"]
    movie_dir = Path(entry["movie_dir"])
    script = movie_dir / "script.json"
    expected = 0
    if script.exists():
        try:
            expected = len(json.loads(script.read_text(encoding="utf-8")))
        except Exception:
            expected = 0
    print(f"{story_id}\t{expected}")
PY
}

case_status() {
  local story_id="$1"
  local expected="$2"
  local case_dir="$OUTPUT_DIR/$story_id"
  local scene_dir="$case_dir/scene_graphs"
  local graph_count=0

  if [[ -d "$scene_dir" ]]; then
    graph_count="$(find "$scene_dir" -path '*/final_graph.json' -type f | wc -l | tr -d ' ')"
  fi

  local graph_status="not started"
  if [[ "$expected" != "0" && "$graph_count" -ge "$expected" ]]; then
    graph_status="scene graphs done ($graph_count/$expected)"
  elif [[ "$graph_count" -gt 0 ]]; then
    graph_status="scene graphs building ($graph_count/$expected)"
  elif [[ -d "$scene_dir" ]]; then
    graph_status="scene_graphs dir exists"
  fi

  local verifier_status="not started"
  if [[ -f "$case_dir/graph_method_hallucinations.json" ]]; then
    verifier_status="graph verifier done"
  fi

  local judge_status="not started"
  if [[ -f "$case_dir/llm_judge_hallucinations.json" ]]; then
    judge_status="llm judge done"
  fi

  local metrics_status="not started"
  if [[ -f "$case_dir/comparison_metrics.json" ]]; then
    metrics_status="metrics done"
  fi

  printf '  %-32s graph: %-30s verifier: %-20s judge: %-16s metrics: %s\n' \
    "$story_id" "$graph_status" "$verifier_status" "$judge_status" "$metrics_status"
}

while true; do
  clear
  echo "Scene Graph Hallucination Eval Watch"
  echo "Root: $ROOT_DIR"
  echo "Output dir: $OUTPUT_DIR"
  echo "Manifest: $MANIFEST"
  echo

  log_file="$(latest_log || true)"
  if [[ -n "${log_file:-}" && -f "$log_file" ]]; then
    now_epoch="$(date +%s)"
    log_mtime="$(mtime_epoch "$log_file")"
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
    echo "Status: waiting for run to start"
  fi

  echo
  echo "Case stages:"
  if [[ -f "$MANIFEST" ]]; then
    while IFS=$'\t' read -r story_id expected; do
      [[ -z "${story_id:-}" ]] && continue
      case_status "$story_id" "$expected"
    done < <(manifest_rows)
  else
    echo "  manifest not found"
  fi

  echo
  if [[ -f "$OUTPUT_DIR/comparison_metrics.csv" ]]; then
    echo "Aggregate metrics: $OUTPUT_DIR/comparison_metrics.csv present"
  else
    echo "Aggregate metrics: not ready"
  fi
  echo
  echo "Tip: if Last update climbs past ${STALE_AFTER}s and case stages stop changing, the run probably died."
  sleep "$INTERVAL"
done
