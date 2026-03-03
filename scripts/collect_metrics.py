#!/usr/bin/env python3
"""Collect key YOLO segmentation metrics from experiment directories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_best(results_csv: Path, key: str = "metrics/mAP50-95(M)") -> dict:
    with results_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty csv: {results_csv}")
    if key not in rows[0]:
        raise KeyError(f"Column {key} not found in {results_csv}")

    best = max(rows, key=lambda r: float(r[key]))
    return {
        "best_epoch": int(float(best["epoch"])),
        "precision_B": float(best["metrics/precision(B)"]),
        "recall_B": float(best["metrics/recall(B)"]),
        "map50_B": float(best["metrics/mAP50(B)"]),
        "map5095_B": float(best["metrics/mAP50-95(B)"]),
        "precision_M": float(best["metrics/precision(M)"]),
        "recall_M": float(best["metrics/recall(M)"]),
        "map50_M": float(best["metrics/mAP50(M)"]),
        "map5095_M": float(best["metrics/mAP50-95(M)"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="result/segment", help="YOLO project directory")
    ap.add_argument("--experiments", nargs="+", required=True, help="Experiment names under project")
    ap.add_argument("--out", default="result/segment/metrics_summary.csv")
    args = ap.parse_args()

    project = Path(args.project)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment",
        "best_epoch",
        "precision_B",
        "recall_B",
        "map50_B",
        "map5095_B",
        "precision_M",
        "recall_M",
        "map50_M",
        "map5095_M",
    ]

    rows = []
    for name in args.experiments:
        csv_path = project / name / "results.csv"
        if not csv_path.exists():
            print(f"[skip] missing {csv_path}")
            continue
        rec = parse_best(csv_path)
        rec["experiment"] = name
        rows.append(rec)

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Saved summary: {out}")
    for r in rows:
        print(
            f"{r['experiment']}: epoch={r['best_epoch']}, "
            f"mAP50-95(B)={r['map5095_B']:.5f}, mAP50-95(M)={r['map5095_M']:.5f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
