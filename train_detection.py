#!/usr/bin/env python3
"""
Train YOLOv8n-seg on the prepared dataset (MPS), validate, and run inference.
Produces confusion matrix and PR curve in runs/segment/train and saves
an example prediction as detection_result.png.
"""

from __future__ import annotations

import sys
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO


ROOT = Path(".")
DATASET_ROOT = ROOT / "data/processed/fpic_component"
DATA_YAML = DATASET_ROOT / "data.yaml"
IMAGES_VAL = DATASET_ROOT / "images/val"
LABELS_VAL = DATASET_ROOT / "labels/val"


def ensure_dataset() -> None:
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Missing dataset yaml: {DATA_YAML}")
    if not IMAGES_VAL.exists() or not LABELS_VAL.exists():
        raise FileNotFoundError("Dataset structure incomplete under data/processed/fpic_component")


def main() -> int:
    # Ensure real-time logs in VS Code terminals
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    ensure_dataset()

    model = YOLO("yolov8n-seg.pt")

    project_dir = (ROOT / "runs/segment").resolve()

    print("Starting YOLOv8 training...", flush=True)
    # Auto-select device: cuda > mps > cpu
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=4,
        device=device,
        amp=False,
        project=str(project_dir),
        name="train",
        exist_ok=True,
        cache="disk",
        workers=2,
        cos_lr=True,
        patience=50,
        verbose=True,
    )

    model.val(
        data=str(DATA_YAML),
        imgsz=640,
        device=device,
        plots=True,
        project=str(project_dir),
        name="train",
    )

    # Inference on one validation image
    # Pick a validation image with the most labels to improve visibility of boxes
    val_labels = sorted([p for p in LABELS_VAL.iterdir() if p.suffix.lower() == ".txt"])
    if not val_labels:
        raise FileNotFoundError(f"No validation labels found in {LABELS_VAL}")
    label_counts = []
    for lbl in val_labels:
        n = len([ln for ln in lbl.read_text().splitlines() if ln.strip()])
        label_counts.append((n, lbl))
    label_counts.sort(reverse=True)
    best_lbl = label_counts[0][1]
    test_img = IMAGES_VAL / f"{best_lbl.stem}.png"
    if not test_img.exists():
        # fallback to any image
        val_images = sorted([p for p in IMAGES_VAL.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
        if not val_images:
            raise FileNotFoundError(f"No validation images found in {IMAGES_VAL}")
        test_img = val_images[0]

    results = model.predict(
        source=str(test_img),
        device=device,
        save=False,
        conf=0.01,
    )
    if results:
        plotted = results[0].plot(boxes=True, masks=True)
        cv2.imwrite("detection_result.png", plotted)
        print("Saved: detection_result.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
