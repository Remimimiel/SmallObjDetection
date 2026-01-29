#!/usr/bin/env python3
"""
Convert FPIC-Component (PCBSegClassNet) semantic masks to YOLOv8 instance segmentation format.

Assumption: each unique non-black color in the mask is a class. Each connected component
of the same color is treated as a separate instance.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/raw/fpic_component/PCBSegClassNet/data/segmentation")
    p.add_argument("--out", default="data/processed/fpic_component", help="Output YOLO directory")
    args = p.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out)
    out_img = out_dir / "images"
    out_lbl = out_dir / "labels"
    for split in ["train", "val", "test"]:
        (out_img / split).mkdir(parents=True, exist_ok=True)
        (out_lbl / split).mkdir(parents=True, exist_ok=True)

    splits = []
    for split in ["train", "val", "test"]:
        img_dir = src_dir / split / "images"
        msk_dir = src_dir / split / "masks"
        if img_dir.exists() and msk_dir.exists():
            splits.append((split, img_dir, msk_dir))

    if not splits:
        raise FileNotFoundError(f"No images/masks found under {src_dir}")

    colors: Dict[Tuple[int, int, int], int] = {}
    for _, _, msk_dir in splits:
        for msk_path in msk_dir.glob("*.png"):
            mask = cv2.imread(str(msk_path), cv2.IMREAD_COLOR)
            if mask is None:
                continue
            uniq = np.unique(mask.reshape(-1, 3), axis=0)
            for bgr in uniq:
                bgr_t = tuple(int(x) for x in bgr.tolist())
                if bgr_t == (0, 0, 0):
                    continue
                if bgr_t not in colors:
                    colors[bgr_t] = len(colors)
        if len(colors) >= 25:
            break

    if not colors:
        raise RuntimeError("No class colors found in masks.")

    colors = {k: i for i, k in enumerate(sorted(colors.keys()))}

    def mask_to_instances(mask: np.ndarray, color: Tuple[int, int, int]) -> List[np.ndarray]:
        target = np.all(mask == np.array(color, dtype=np.uint8), axis=2).astype(np.uint8) * 255
        contours, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    for split, img_dir, msk_dir in splits:
        for img_path in img_dir.glob("*.png"):
            mask_path = msk_dir / img_path.name
            if not mask_path.exists():
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
            if img is None or mask is None:
                continue
            h, w = img.shape[:2]

            lines: List[str] = []
            for color, cid in colors.items():
                contours = mask_to_instances(mask, color)
                for cnt in contours:
                    if cnt.shape[0] < 3:
                        continue
                    pts = cnt.squeeze(1).astype(float)
                    if pts.ndim != 2 or pts.shape[0] < 3:
                        continue
                    pts[:, 0] /= w
                    pts[:, 1] /= h
                    pts = np.clip(pts, 0.0, 1.0)
                    coords = " ".join([f"{p[0]:.6f} {p[1]:.6f}" for p in pts])
                    lines.append(f"{cid} {coords}")

            (out_lbl / split / f"{img_path.stem}.txt").write_text("\n".join(lines))
            out_img_path = out_img / split / img_path.name
            if not out_img_path.exists():
                cv2.imwrite(str(out_img_path), img)

    names = [f"class_{i}" for i in range(len(colors))]
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"nc: {len(names)}",
                f"names: {names}",
            ]
        )
        + "\n"
    )

    print(f"YOLO dataset ready: {out_dir}")
    print(f"Data YAML: {data_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
