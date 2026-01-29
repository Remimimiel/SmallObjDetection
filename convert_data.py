#!/usr/bin/env python3
"""
Convert polygon annotations to YOLOv8 segmentation TXT format.

Each output line: class_id x1 y1 x2 y2 ... (normalized polygon coords)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_line(line: str) -> Tuple[List[float], str]:
    """
    Parse a single annotation line and return (coords, class_name).
    The line is expected to contain polygon coords followed by class name,
    optionally followed by a numeric flag (e.g., difficulty).
    """
    parts = line.strip().split()
    if len(parts) < 7:
        raise ValueError("Too few tokens")

    # Find the last non-numeric token as class name
    class_idx = None
    for i in range(len(parts) - 1, -1, -1):
        if not is_float(parts[i]):
            class_idx = i
            break
    if class_idx is None:
        raise ValueError("No class token found")

    class_name = parts[class_idx]
    coord_tokens = parts[:class_idx]
    if len(coord_tokens) % 2 != 0 or len(coord_tokens) < 6:
        raise ValueError("Invalid coordinate count")

    coords = [float(x) for x in coord_tokens]
    return coords, class_name


def normalize_coords(coords: List[float], w: int, h: int) -> List[float]:
    norm = []
    for i, v in enumerate(coords):
        if i % 2 == 0:
            norm.append(v / w)
        else:
            norm.append(v / h)
    return norm


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="data/raw/pcbdataset/images", help="Images directory")
    p.add_argument("--ann", default="data/raw/pcbdataset/annfiles", help="Annotation directory")
    p.add_argument("--out", default="data/processed/labels", help="Output labels directory")
    p.add_argument("--classes-out", default="data/processed/classes.txt", help="Class list output")
    args = p.parse_args()

    images_dir = Path(args.images)
    ann_dir = Path(args.ann)
    out_dir = Path(args.out)
    classes_out = Path(args.classes_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    classes_out.parent.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(ann_dir.glob("*.txt"))
    if not ann_files:
        print(f"No annotation files found in {ann_dir}")
        return 1

    # Collect class names
    class_names: Dict[str, int] = {}
    parsed_cache: Dict[Path, List[Tuple[List[float], str]]] = {}
    for ann in ann_files:
        lines = ann.read_text().splitlines()
        parsed = []
        for line in lines:
            if not line.strip():
                continue
            try:
                coords, cname = parse_line(line)
            except ValueError:
                continue
            if cname not in class_names:
                class_names[cname] = len(class_names)
            parsed.append((coords, cname))
        parsed_cache[ann] = parsed

    # Write class list
    classes_out.write_text("\n".join([k for k, _ in sorted(class_names.items(), key=lambda x: x[1])]))

    # Convert annotations
    converted = 0
    for ann, items in parsed_cache.items():
        if not items:
            continue
        stem = ann.stem
        # Find corresponding image (png/jpg/jpeg)
        img_path = None
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            cand = images_dir / f"{stem}{ext}"
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            continue

        with Image.open(img_path) as im:
            w, h = im.size

        lines_out: List[str] = []
        for coords, cname in items:
            norm = normalize_coords(coords, w, h)
            # Clamp to [0,1] for safety
            norm = [min(1.0, max(0.0, v)) for v in norm]
            cid = class_names[cname]
            line = " ".join([str(cid)] + [f"{v:.6f}" for v in norm])
            lines_out.append(line)

        out_path = out_dir / f"{stem}.txt"
        out_path.write_text("\n".join(lines_out))
        converted += 1

    print(f"Converted {converted} files. Classes: {len(class_names)}")
    print(f"Labels: {out_dir}")
    print(f"Classes: {classes_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
