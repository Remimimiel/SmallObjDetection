#!/usr/bin/env python3
"""
Generate augmentation demo figure (1x4) using Albumentations.
Saves to figures/chapter3/augmentation_demo.png
"""

from __future__ import annotations

import random
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt


def load_image_with_many_targets(images_dir: Path, ann_dir: Path, min_targets: int = 5) -> Path:
    ann_files = sorted(ann_dir.glob("*.txt"))
    if not ann_files:
        raise FileNotFoundError(f"No annotation files found in {ann_dir}")

    candidates = []
    for ann in ann_files:
        lines = [ln for ln in ann.read_text().splitlines() if ln.strip()]
        if len(lines) >= min_targets:
            candidates.append(ann)

    if not candidates:
        # Fallback to any annotation file
        candidates = ann_files

    ann_path = random.choice(candidates)
    stem = ann_path.stem
    for ext in (".png", ".jpg", ".jpeg"):
        img_path = images_dir / f"{stem}{ext}"
        if img_path.exists():
            return img_path
    raise FileNotFoundError(f"No matching image found for {ann_path.name} in {images_dir}")


def main() -> int:
    images_dir = Path("data/raw/pcbdataset/images")
    ann_dir = Path("data/raw/pcbdataset/annfiles")
    out_path = Path("figures/chapter3/augmentation_demo.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img_path = load_image_with_many_targets(images_dir, ann_dir, min_targets=5)
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Enforce odd kernel size within [15, 30]
    k = random.choice(list(range(15, 31, 2)))
    motion_blur = A.MotionBlur(blur_limit=(k, k), p=1.0)
    # Albumentations GaussNoise expects std_range in recent versions
    gaussian_noise = A.GaussNoise(std_range=(0.2, 0.44), mean_range=(0.0, 0.0), p=1.0)
    lighting_jitter = A.RandomBrightnessContrast(
        brightness_limit=0.25, contrast_limit=0.25, p=1.0
    )

    aug_motion = motion_blur(image=img_rgb)["image"]
    aug_noise = gaussian_noise(image=img_rgb)["image"]
    aug_light = lighting_jitter(image=img_rgb)["image"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[1].imshow(aug_motion)
    axes[1].set_title("Motion Blur")
    axes[2].imshow(aug_noise)
    axes[2].set_title("Gaussian Noise")
    axes[3].imshow(aug_light)
    axes[3].set_title("Lighting Jitter")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
