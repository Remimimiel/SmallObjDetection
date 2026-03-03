#!/usr/bin/env bash
set -euo pipefail

# Round 2 = strong_n_100e
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
PROJECT_DIR="$ROOT_DIR/result/segment"

echo "[0/6] Prepare dataset from local zip if needed..."
bash scripts/prepare_fpic_from_zip.sh

echo "[1/6] Re-splitting dataset to 5008/626/626..."
python scripts/re_split_fpic_8_1_1.py --root data/processed/fpic_component --seed 42

echo "[2/6] Train strong_n_100e..."
yolo segment train \
  model=yolov8n-seg.pt \
  data=data/processed/fpic_component/data.yaml \
  epochs=100 imgsz=800 batch=2 device=0 seed=42 cos_lr=True amp=False \
  project="$PROJECT_DIR" name=strong_n_100e exist_ok=True

BEST_PT="$PROJECT_DIR/strong_n_100e/weights/best.pt"
if [[ ! -f "$BEST_PT" ]]; then
  BEST_PT="$(find "$ROOT_DIR" -type f -path '*/strong_n_100e/weights/best.pt' | head -n 1 || true)"
fi
if [[ -z "${BEST_PT:-}" || ! -f "$BEST_PT" ]]; then
  echo "ERROR: best.pt not found after strong_n_100e training."
  exit 1
fi

echo "[3/6] Evaluate strong_n_100e on val..."
yolo segment val \
  model="$BEST_PT" \
  data=data/processed/fpic_component/data.yaml split=val device=0 \
  project="$PROJECT_DIR" name=strong_n_100e_val exist_ok=True

echo "[4/6] Evaluate strong_n_100e on test..."
yolo segment val \
  model="$BEST_PT" \
  data=data/processed/fpic_component/data.yaml split=test device=0 \
  project="$PROJECT_DIR" name=strong_n_100e_test exist_ok=True

echo "[5/6] Summarize metrics (round1 + round2)..."
python scripts/collect_metrics.py \
  --project "$PROJECT_DIR" \
  --experiments baseline_n_50e strong_n_100e \
  --out "$PROJECT_DIR/metrics_summary_round1_round2.csv"

echo "[6/6] Round2 done."
echo "Keep these files for analysis:"
echo "  - $PROJECT_DIR/strong_n_100e/results.csv"
echo "  - $PROJECT_DIR/strong_n_100e/args.yaml"
echo "  - $PROJECT_DIR/strong_n_100e/weights/best.pt"
echo "  - $PROJECT_DIR/strong_n_100e_val/* and $PROJECT_DIR/strong_n_100e_test/*"
echo "  - $PROJECT_DIR/metrics_summary_round1_round2.csv"
