#!/usr/bin/env python3
"""
Convert FPIC-Component (PCBSegClassNet) semantic masks to YOLOv8 instance segmentation format.

Assumption: each unique non-black color in the mask is a class. Each connected component
of the same color is treated as a separate instance.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from multiprocessing import Pool


_COLORS: Dict[Tuple[int, int, int], int] = {}
_OUT_IMG: Path | None = None
_OUT_LBL: Path | None = None


def _init_worker(colors: Dict[Tuple[int, int, int], int], out_img: str, out_lbl: str) -> None:
    global _COLORS, _OUT_IMG, _OUT_LBL
    _COLORS = colors
    _OUT_IMG = Path(out_img)
    _OUT_LBL = Path(out_lbl)


def _mask_to_instances(mask: np.ndarray, color: Tuple[int, int, int]) -> List[np.ndarray]:
    target = np.all(mask == np.array(color, dtype=np.uint8), axis=2).astype(np.uint8) * 255
    contours, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _process_one(args: Tuple[str, str, str]) -> int:
    split, img_path_s, mask_path_s = args
    img_path = Path(img_path_s)
    mask_path = Path(mask_path_s)

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if img is None or mask is None:
        return 0

    h, w = img.shape[:2]
    lines: List[str] = []
    for color, cid in _COLORS.items():
        contours = _mask_to_instances(mask, color)
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

    assert _OUT_LBL is not None and _OUT_IMG is not None
    (_OUT_LBL / split / f"{img_path.stem}.txt").write_text("\n".join(lines))
    out_img_path = _OUT_IMG / split / img_path.name
    if not out_img_path.exists():
        cv2.imwrite(str(out_img_path), img)
    return 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/raw/fpic_component/PCBSegClassNet/data/segmentation")
    p.add_argument("--out", default="data/processed/fpic_component", help="Output YOLO directory")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--print-every", type=int, default=200)
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

    print("Scanning masks to build color map...", flush=True)
    colors: Dict[Tuple[int, int, int], int] = {}
    scanned = 0
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
            scanned += 1
            if scanned % 200 == 0:
                print(f"  scanned {scanned} masks, classes so far: {len(colors)}", flush=True)
        if len(colors) >= 25:
            break

    if not colors:
        raise RuntimeError("No class colors found in masks.")

    colors = {k: i for i, k in enumerate(sorted(colors.keys()))}

    print("Building image-mask pair list...", flush=True)
    items: List[Tuple[str, str, str]] = []
    for split, img_dir, msk_dir in splits:
        for img_path in img_dir.glob("*.png"):
            mask_path = msk_dir / img_path.name
            if mask_path.exists():
                items.append((split, str(img_path), str(mask_path)))
        print(f"  {split}: {len(items)} pairs so far", flush=True)

    total = len(items)
    if total == 0:
        raise RuntimeError("No image-mask pairs found.")

    print(f"Converting {total} samples with {args.workers} workers...", flush=True)
    start = time.time()
    done = 0

    with Pool(processes=args.workers, initializer=_init_worker, initargs=(colors, str(out_img), str(out_lbl))) as pool:
        for _ in pool.imap_unordered(_process_one, items, chunksize=16):
            done += 1
            if done % args.print_every == 0 or done == total:
                elapsed = time.time() - start
                rate = done / max(elapsed, 1e-6)
                print(f"{done}/{total} ({done/total:.1%}) - {rate:.2f} items/s", flush=True)

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
