#!/usr/bin/env python3
"""
Deterministically re-split FPIC processed dataset to 8:1:1 using existing
train/val folders where train stays unchanged and val is split into val/test.

Target counts:
- train: 5008
- val: 626
- test: 626
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def list_images(d: Path):
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def count_instances(label_dir: Path) -> int:
    total = 0
    for p in sorted(label_dir.glob("*.txt")):
        total += sum(1 for ln in p.read_text().splitlines() if ln.strip())
    return total


def write_manifest(path: Path, stems: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(stems) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed/fpic_component", help="Processed dataset root")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    img_train = root / "images" / "train"
    img_val = root / "images" / "val"
    img_test = root / "images" / "test"
    lbl_train = root / "labels" / "train"
    lbl_val = root / "labels" / "val"
    lbl_test = root / "labels" / "test"

    for d in [img_train, img_val, img_test, lbl_train, lbl_val, lbl_test]:
        if not d.exists():
            raise FileNotFoundError(f"Missing directory: {d}")

    train_imgs = list_images(img_train)
    val_imgs = list_images(img_val)
    test_imgs = list_images(img_test)

    if len(train_imgs) != 5008:
        raise RuntimeError(f"Expected 5008 train images, got {len(train_imgs)}")

    # Normalize from current state: move existing test back into val, then re-split.
    if test_imgs:
        print(f"Found existing test images: {len(test_imgs)}. Moving back to val before re-split...")
        for p in test_imgs:
            stem = p.stem
            lp = lbl_test / f"{stem}.txt"
            if not lp.exists():
                raise FileNotFoundError(f"Missing test label: {lp}")
            if not args.dry_run:
                shutil.move(str(p), str(img_val / p.name))
                shutil.move(str(lp), str(lbl_val / lp.name))
                npy = img_test / f"{stem}.npy"
                if npy.exists():
                    shutil.move(str(npy), str(img_val / npy.name))

    val_imgs = list_images(img_val)
    if len(val_imgs) != 1252:
        raise RuntimeError(f"Expected 1252 val images before split, got {len(val_imgs)}")

    random.seed(args.seed)
    shuffled = val_imgs[:]
    random.shuffle(shuffled)
    new_test = shuffled[:626]

    print(f"Re-splitting with seed={args.seed}: val->test {len(new_test)} files")

    for p in new_test:
        stem = p.stem
        lp = lbl_val / f"{stem}.txt"
        if not lp.exists():
            raise FileNotFoundError(f"Missing val label: {lp}")
        if not args.dry_run:
            shutil.move(str(p), str(img_test / p.name))
            shutil.move(str(lp), str(lbl_test / lp.name))
            npy = img_val / f"{stem}.npy"
            if npy.exists():
                shutil.move(str(npy), str(img_test / npy.name))

    train_imgs = list_images(img_train)
    val_imgs = list_images(img_val)
    test_imgs = list_images(img_test)

    if (len(train_imgs), len(val_imgs), len(test_imgs)) != (5008, 626, 626):
        raise RuntimeError(
            f"Split mismatch after operation: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}"
        )

    if len(list(lbl_train.glob("*.txt"))) != 5008:
        raise RuntimeError("train labels mismatch")
    if len(list(lbl_val.glob("*.txt"))) != 626:
        raise RuntimeError("val labels mismatch")
    if len(list(lbl_test.glob("*.txt"))) != 626:
        raise RuntimeError("test labels mismatch")

    split_dir = root / "split_manifests_8_1_1"
    if not args.dry_run:
        write_manifest(split_dir / "train.txt", [p.stem for p in train_imgs])
        write_manifest(split_dir / "val.txt", [p.stem for p in val_imgs])
        write_manifest(split_dir / "test.txt", [p.stem for p in test_imgs])

    print("Final split counts")
    print(f"  train: images={len(train_imgs)}, labels={len(list(lbl_train.glob('*.txt')))}, instances={count_instances(lbl_train)}")
    print(f"  val:   images={len(val_imgs)}, labels={len(list(lbl_val.glob('*.txt')))}, instances={count_instances(lbl_val)}")
    print(f"  test:  images={len(test_imgs)}, labels={len(list(lbl_test.glob('*.txt')))}, instances={count_instances(lbl_test)}")
    print(f"Saved split manifests under: {split_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
