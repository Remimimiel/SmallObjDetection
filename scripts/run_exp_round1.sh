#!/usr/bin/env bash
set -euo pipefail

# Round 1 = baseline only (fastest complete loop)
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
PROJECT_DIR="$ROOT_DIR/result/segment"

echo "[0/6] Prepare dataset from local zip if needed..."
# default zip path: ../pcbsegclassnet.zip (parallel to repo directory)
bash scripts/prepare_fpic_from_zip.sh

echo "[1/6] Re-splitting dataset to 5008/626/626..."
python scripts/re_split_fpic_8_1_1.py --root data/processed/fpic_component --seed 42

echo "[2/6] Sanity train (5 epochs)..."
yolo segment train \
  model=yolov8n-seg.pt \
  data=data/processed/fpic_component/data.yaml \
  epochs=5 imgsz=640 batch=4 device=0 seed=42 cos_lr=True amp=False \
  project="$PROJECT_DIR" name=sanity_n_5e exist_ok=True

echo "[3/6] Baseline train (50 epochs)..."
yolo segment train \
  model=yolov8n-seg.pt \
  data=data/processed/fpic_component/data.yaml \
  epochs=50 imgsz=640 batch=4 device=0 seed=42 cos_lr=True amp=False \
  project="$PROJECT_DIR" name=baseline_n_50e exist_ok=True

BEST_PT="$PROJECT_DIR/baseline_n_50e/weights/best.pt"
if [[ ! -f "$BEST_PT" ]]; then
  BEST_PT="$(find "$ROOT_DIR" -type f -path "*/baseline_n_50e/weights/best.pt" | head -n 1 || true)"
fi
if [[ -z "${BEST_PT:-}" || ! -f "$BEST_PT" ]]; then
  echo "ERROR: best.pt not found after baseline training."
  exit 1
fi

echo "[4/6] Evaluate baseline on val/test..."
yolo segment val \
  model="$BEST_PT" \
  data=data/processed/fpic_component/data.yaml split=val device=0 \
  project="$PROJECT_DIR" name=baseline_n_50e_val exist_ok=True

yolo segment val \
  model="$BEST_PT" \
  data=data/processed/fpic_component/data.yaml split=test device=0 \
  project="$PROJECT_DIR" name=baseline_n_50e_test exist_ok=True

echo "[5/6] Summarize metrics..."
python scripts/collect_metrics.py --project "$PROJECT_DIR" --experiments baseline_n_50e --out "$PROJECT_DIR/metrics_summary_round1.csv"

echo "[6/6] Round1 done."
echo "Keep these files for analysis:"
echo "  - data/processed/fpic_component/split_manifests_8_1_1/{train,val,test}.txt"
echo "  - $PROJECT_DIR/baseline_n_50e/results.csv"
echo "  - $PROJECT_DIR/baseline_n_50e/args.yaml"
echo "  - $PROJECT_DIR/baseline_n_50e/weights/best.pt"
echo "  - $PROJECT_DIR/baseline_n_50e_val/* and $PROJECT_DIR/baseline_n_50e_test/*"
echo "  - $PROJECT_DIR/metrics_summary_round1.csv"
