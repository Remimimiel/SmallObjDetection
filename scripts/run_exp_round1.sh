#!/usr/bin/env bash
set -euo pipefail

# Round 1 = baseline only (fastest complete loop)
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/5] Re-splitting dataset to 5008/626/626..."
python scripts/re_split_fpic_8_1_1.py --root data/processed/fpic_component --seed 42

echo "[2/5] Sanity train (5 epochs)..."
yolo segment train \
  model=yolov8n-seg.pt \
  data=data/processed/fpic_component/data.yaml \
  epochs=5 imgsz=640 batch=4 device=0 seed=42 cos_lr=True \
  project=result/segment name=sanity_n_5e exist_ok=True

echo "[3/5] Baseline train (50 epochs)..."
yolo segment train \
  model=yolov8n-seg.pt \
  data=data/processed/fpic_component/data.yaml \
  epochs=50 imgsz=640 batch=4 device=0 seed=42 cos_lr=True \
  project=result/segment name=baseline_n_50e exist_ok=True

echo "[4/5] Evaluate baseline on val/test..."
yolo segment val \
  model=result/segment/baseline_n_50e/weights/best.pt \
  data=data/processed/fpic_component/data.yaml split=val device=0 \
  project=result/segment name=baseline_n_50e_val exist_ok=True

yolo segment val \
  model=result/segment/baseline_n_50e/weights/best.pt \
  data=data/processed/fpic_component/data.yaml split=test device=0 \
  project=result/segment name=baseline_n_50e_test exist_ok=True

echo "[5/5] Summarize metrics..."
python scripts/collect_metrics.py --project result/segment --experiments baseline_n_50e --out result/segment/metrics_summary_round1.csv

echo "Round1 done."
echo "Keep these files for analysis:"
echo "  - result/segment/baseline_n_50e/results.csv"
echo "  - result/segment/baseline_n_50e/args.yaml"
echo "  - result/segment/baseline_n_50e/weights/best.pt"
echo "  - result/segment/baseline_n_50e_val/* and baseline_n_50e_test/*"
echo "  - result/segment/metrics_summary_round1.csv"
