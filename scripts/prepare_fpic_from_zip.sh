#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="$ROOT_DIR/data/processed/fpic_component"
ZIP_DEFAULT="$ROOT_DIR/../pcbsegclassnet.zip"
ZIP_PATH="${1:-$ZIP_DEFAULT}"

has_processed_dataset() {
  local train_count val_count test_count
  train_count=$(find "$DATA_ROOT/images/train" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) 2>/dev/null | wc -l | tr -d ' ')
  val_count=$(find "$DATA_ROOT/images/val" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) 2>/dev/null | wc -l | tr -d ' ')
  test_count=$(find "$DATA_ROOT/images/test" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) 2>/dev/null | wc -l | tr -d ' ')
  [[ "$train_count" -gt 0 && "$val_count" -gt 0 ]]
}

if has_processed_dataset; then
  echo "[prepare] Found processed dataset under $DATA_ROOT, skip unzip/convert."
  exit 0
fi

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "[prepare] ERROR: zip not found: $ZIP_PATH"
  echo "[prepare] Usage: bash scripts/prepare_fpic_from_zip.sh /abs/path/to/pcbsegclassnet.zip"
  exit 1
fi

RAW_BASE="$ROOT_DIR/data/raw/fpic_component"
mkdir -p "$RAW_BASE"

echo "[prepare] Unzipping: $ZIP_PATH"
unzip -oq "$ZIP_PATH" -d "$RAW_BASE"

# Try to locate segmentation root expected by convert_data.py
SEG_ROOT=""
for cand in \
  "$RAW_BASE/PCBSegClassNet/data/segmentation" \
  "$RAW_BASE/pcbsegclassnet/PCBSegClassNet/data/segmentation" \
  "$RAW_BASE/data/segmentation"
do
  if [[ -d "$cand" ]]; then
    SEG_ROOT="$cand"
    break
  fi
done

if [[ -z "$SEG_ROOT" ]]; then
  SEG_ROOT=$(find "$RAW_BASE" -type d -path '*/data/segmentation' | head -n 1 || true)
fi

if [[ -z "$SEG_ROOT" || ! -d "$SEG_ROOT" ]]; then
  echo "[prepare] ERROR: cannot find segmentation directory after unzip."
  echo "[prepare] Please inspect: $RAW_BASE"
  exit 1
fi

echo "[prepare] Converting masks -> YOLO-seg from: $SEG_ROOT"
cd "$ROOT_DIR"
python convert_data.py --src "$SEG_ROOT" --out data/processed/fpic_component

echo "[prepare] Dataset prepared under data/processed/fpic_component"
