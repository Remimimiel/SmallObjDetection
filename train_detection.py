#!/usr/bin/env python3
"""
Train YOLOv8s-seg on the prepared dataset (MPS), validate, and run inference.
Produces confusion matrix and PR curve in runs/segment/train and saves
an example prediction as detection_result.png.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
import cv2
import yaml
from ultralytics import YOLO


ROOT = Path(".")
RAW_IMAGES = ROOT / "data/raw/pcbdataset/images"
RAW_ANN = ROOT / "data/raw/pcbdataset/annfiles"
LABELS_FLAT = ROOT / "data/processed/labels"
PROCESSED = ROOT / "data/processed"
SEG_ROOT = PROCESSED / "seg"
DATA_YAML = SEG_ROOT / "data.yaml"
IMAGES_TRAIN = SEG_ROOT / "images/train"
IMAGES_VAL = SEG_ROOT / "images/val"
LABELS_TRAIN = SEG_ROOT / "labels/train"
LABELS_VAL = SEG_ROOT / "labels/val"
CLASSES_TXT = PROCESSED / "classes.txt"


def ensure_dirs() -> None:
    IMAGES_TRAIN.mkdir(parents=True, exist_ok=True)
    IMAGES_VAL.mkdir(parents=True, exist_ok=True)
    LABELS_TRAIN.mkdir(parents=True, exist_ok=True)
    LABELS_VAL.mkdir(parents=True, exist_ok=True)


def symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        # Fallback to copy if symlink not permitted
        dst.write_bytes(src.read_bytes())


def clear_dir(dir_path: Path) -> None:
    if not dir_path.exists():
        return
    for p in dir_path.iterdir():
        if p.is_file() or p.is_symlink():
            p.unlink()


def prepare_split(val_ratio: float = 0.2, seed: int = 42) -> None:
    if not RAW_IMAGES.exists() or not LABELS_FLAT.exists():
        raise FileNotFoundError("Missing images or labels. Run convert_data.py first.")
    ensure_dirs()
    clear_dir(IMAGES_TRAIN)
    clear_dir(IMAGES_VAL)
    clear_dir(LABELS_TRAIN)
    clear_dir(LABELS_VAL)

    images = sorted([p for p in RAW_IMAGES.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not images:
        raise FileNotFoundError(f"No images found in {RAW_IMAGES}")

    random.seed(seed)
    random.shuffle(images)
    val_count = max(1, int(len(images) * val_ratio))
    val_set = set(images[:val_count])

    for img in images:
        target_img_dir = IMAGES_VAL if img in val_set else IMAGES_TRAIN
        target_lbl_dir = LABELS_VAL if img in val_set else LABELS_TRAIN
        symlink_or_copy(img, target_img_dir / img.name)

        lbl = LABELS_FLAT / f"{img.stem}.txt"
        if lbl.exists():
            symlink_or_copy(lbl, target_lbl_dir / lbl.name)


def ensure_data_yaml() -> None:
    if DATA_YAML.exists():
        return
    if not CLASSES_TXT.exists():
        raise FileNotFoundError("Missing classes.txt. Run convert_data.py first.")

    names = [ln.strip() for ln in CLASSES_TXT.read_text().splitlines() if ln.strip()]
    data = {
        "path": str(SEG_ROOT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    DATA_YAML.write_text(yaml.safe_dump(data, sort_keys=False))


def main() -> int:
    # Ensure real-time logs in VS Code terminals
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    prepare_split()
    ensure_data_yaml()
    # Remove stale cache to avoid broken symlink references
    for cache in [
        SEG_ROOT / "labels/train.cache",
        SEG_ROOT / "labels/val.cache",
    ]:
        if cache.exists():
            cache.unlink()

    # More stable segmentation model for better curves
    model = YOLO("yolov8s-seg.pt")

    project_dir = (ROOT / "runs/segment").resolve()

    print("Starting YOLOv8 training...", flush=True)
    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=4,
        device="mps",
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
        device="mps",
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
        device="mps",
        save=False,
        conf=0.01,
    )
    if results:
        plotted = results[0].plot()
        cv2.imwrite("detection_result.png", plotted)
        print("Saved: detection_result.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
