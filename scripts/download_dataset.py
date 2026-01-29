#!/usr/bin/env python3
"""
Download a small metal fastener dataset from Kaggle or OpenML.

Examples:
  python scripts/download_dataset.py --source kaggle
  python scripts/download_dataset.py --source kaggle --kaggle yartinz/npu-bolt
  python scripts/download_dataset.py --source openml --openml-id 12345
  python scripts/download_dataset.py --source auto
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import requests


DEFAULT_KAGGLE_DATASETS = [
    # Dense small metal fastener datasets
    "yartinz/npu-bolt",
    "ruruamour/screw-dataset",
]

OPENML_LIST_URL = "https://www.openml.org/api/v1/json/data/list"
OPENML_DOWNLOAD_URL = "https://www.openml.org/data/v1/download/{did}"


def _run(cmd: List[str]) -> int:
    return subprocess.call(cmd)


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def kaggle_download(slug: str, out_dir: Path) -> bool:
    if _which("kaggle") is None:
        print("ERROR: kaggle CLI not found. Install with `pip install kaggle`.", file=sys.stderr)
        return False
    ensure_dir(out_dir)
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(out_dir), "--unzip"]
    print(f"Running: {' '.join(cmd)}")
    return _run(cmd) == 0


def openml_search(query: str) -> List[Tuple[str, str]]:
    params = {"search": query}
    resp = requests.get(OPENML_LIST_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    datasets = data.get("data", {}).get("dataset", [])
    results = []
    for d in datasets:
        did = str(d.get("did", "")).strip()
        name = str(d.get("name", "")).strip()
        if did and name:
            results.append((did, name))
    return results


def openml_download(did: str, out_dir: Path) -> bool:
    ensure_dir(out_dir)
    url = OPENML_DOWNLOAD_URL.format(did=did)
    out_path = out_dir / f"openml_{did}.data"
    print(f"Downloading OpenML dataset {did} -> {out_path}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    meta_path = out_dir / f"openml_{did}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"openml_id": did, "download_url": url}, f, indent=2)
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download a metal fastener dataset.")
    p.add_argument("--source", choices=["kaggle", "openml", "auto"], default="auto")
    p.add_argument("--kaggle", help="Kaggle dataset slug, e.g. yartinz/npu-bolt")
    p.add_argument("--openml-id", help="OpenML dataset id (did)")
    p.add_argument("--openml-query", default="screw bolt nut fastener")
    p.add_argument("--out", default="data/raw", help="Output directory")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)

    if args.source in ("kaggle", "auto"):
        slugs = [args.kaggle] if args.kaggle else list(DEFAULT_KAGGLE_DATASETS)
        for slug in slugs:
            if not slug:
                continue
            print(f"Trying Kaggle dataset: {slug}")
            if kaggle_download(slug, out_dir):
                print(f"Downloaded Kaggle dataset: {slug}")
                return 0
            print(f"Kaggle download failed: {slug}", file=sys.stderr)

        if args.source == "kaggle":
            return 2

    # OpenML path
    if args.source in ("openml", "auto"):
        did = args.openml_id
        if not did:
            print("Searching OpenML datasets...")
            results = openml_search(args.openml_query)
            if not results:
                print("No OpenML datasets found for query.", file=sys.stderr)
                return 3
            print("Found OpenML datasets:")
            for did, name in results[:10]:
                print(f"  {did}: {name}")
            print("Re-run with --openml-id <did> to download.")
            return 4

        if openml_download(did, out_dir):
            print(f"Downloaded OpenML dataset: {did}")
            return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
