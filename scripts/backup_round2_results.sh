#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
DEST="$ROOT_DIR/artifacts/round2_strong_n_100e_$STAMP"
mkdir -p "$DEST"

RESULTS_CSV="$(find "$ROOT_DIR" -type f -path '*/strong_n_100e/results.csv' | head -n 1 || true)"
ARGS_YAML="$(find "$ROOT_DIR" -type f -path '*/strong_n_100e/args.yaml' | head -n 1 || true)"
METRICS_SUMMARY="$(find "$ROOT_DIR" -type f -name 'metrics_summary_round1_round2.csv' | head -n 1 || true)"
VAL_DIR="$(find "$ROOT_DIR" -type d -name 'strong_n_100e_val' | head -n 1 || true)"
TEST_DIR="$(find "$ROOT_DIR" -type d -name 'strong_n_100e_test' | head -n 1 || true)"

for f in "$RESULTS_CSV" "$ARGS_YAML" "$METRICS_SUMMARY"; do
  if [[ -n "$f" && -f "$f" ]]; then
    cp "$f" "$DEST/"
  fi
done

if [[ -n "$VAL_DIR" && -d "$VAL_DIR" ]]; then
  cp -R "$VAL_DIR" "$DEST/"
fi
if [[ -n "$TEST_DIR" && -d "$TEST_DIR" ]]; then
  cp -R "$TEST_DIR" "$DEST/"
fi

if [[ -d "$ROOT_DIR/data/processed/fpic_component/split_manifests_8_1_1" ]]; then
  mkdir -p "$DEST/split_manifests_8_1_1"
  cp "$ROOT_DIR/data/processed/fpic_component/split_manifests_8_1_1/"*.txt "$DEST/split_manifests_8_1_1/"
fi

echo "Backup created: $DEST"
ls -la "$DEST"
