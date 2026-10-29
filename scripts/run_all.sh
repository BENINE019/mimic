#!/usr/bin/env bash
# Run naming, tracking and table mapping on every clips/cNN.mp4.
# Usage: bash scripts/run_all.sh 9.5        (your measured marker size in cm)
# Safe to rerun: existing names and cached SAM results are reused.
set -uo pipefail
MARKER_CM="${1:?give the marker size in cm, e.g. bash scripts/run_all.sh 9.5}"
: "${MODEL_API_KEY:?export MODEL_API_KEY first}"

for clip in clips/c[0-9][0-9].mp4; do
  name=$(basename "$clip" .mp4)
  out="out/$name"
  echo "================ $name ================"
  if [ ! -f "$out/names.json" ]; then
    python scripts/name_objects.py "$clip" --output "$out" || { echo "naming failed for $name"; continue; }
  fi
  python -m mimic.tracking "$clip" --names "$out/names.json" || { echo "tracking failed for $name"; continue; }
  python -m mimic.table_map "$clip" --marker-cm "$MARKER_CM" || echo "table mapping failed for $name"
done
echo "Done. Check out/cNN/topdown.png for each clip."
