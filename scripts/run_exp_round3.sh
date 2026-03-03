#!/usr/bin/env bash
set -euo pipefail

# Round 3 = strong_s_80e
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
PROJECT_DIR="$ROOT_DIR/result/segment"

echo "[0/6] Prepare dataset from local zip if needed..."
bash scripts/prepare_fpic_from_zip.sh

echo "[1/6] Re-splitting dataset to 5008/626/626..."
python scripts/re_split_fpic_8_1_1.py --root data/processed/fpic_component --seed 42

echo "[2/6] Train strong_s_80e..."
yolo segment train \
  model=yolov8s-seg.pt \
  data=data/processed/fpic_component/data.yaml \
  epochs=80 imgsz=640 batch=4 device=0 seed=42 cos_lr=True amp=False \
  project="$PROJECT_DIR" name=strong_s_80e exist_ok=True

BEST_PT="$PROJECT_DIR/strong_s_80e/weights/best.pt"
if [[ ! -f "$BEST_PT" ]]; then
  BEST_PT="$(find "$ROOT_DIR" -type f -path '*/strong_s_80e/weights/best.pt' | head -n 1 || true)"
fi
if [[ -z "${BEST_PT:-}" || ! -f "$BEST_PT" ]]; then
  echo "ERROR: best.pt not found after strong_s_80e training."
  exit 1
fi

echo "[3/6] Evaluate strong_s_80e on val..."
yolo segment val \
  model="$BEST_PT" \
  data=data/processed/fpic_component/data.yaml split=val device=0 \
  project="$PROJECT_DIR" name=strong_s_80e_val exist_ok=True

echo "[4/6] Evaluate strong_s_80e on test..."
yolo segment val \
  model="$BEST_PT" \
  data=data/processed/fpic_component/data.yaml split=test device=0 \
  project="$PROJECT_DIR" name=strong_s_80e_test exist_ok=True

echo "[5/6] Summarize metrics (round2 + round3)..."
python scripts/collect_metrics.py \
  --project "$PROJECT_DIR" \
  --experiments strong_n_100e strong_s_80e \
  --out "$PROJECT_DIR/metrics_summary_round2_round3.csv"

echo "[6/6] Round3 done."
echo "Keep these files for analysis:"
echo "  - $PROJECT_DIR/strong_s_80e/results.csv"
echo "  - $PROJECT_DIR/strong_s_80e/args.yaml"
echo "  - $PROJECT_DIR/strong_s_80e/weights/best.pt"
echo "  - $PROJECT_DIR/strong_s_80e_val/* and $PROJECT_DIR/strong_s_80e_test/*"
echo "  - $PROJECT_DIR/metrics_summary_round2_round3.csv"
