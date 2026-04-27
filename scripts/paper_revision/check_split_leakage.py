#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check whether the validation files reported by best_val_predictions.csv overlap with inferred training files.

Run:
    python scripts_paper_revision/check_split_leakage.py --repo-root .
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--run", type=str, default="attention_res1d_model")
    parser.add_argument("--out", type=Path, default=Path("results/paper_revision_audit/split_leakage_report.json"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    meta_path = root / "data/processed/dataset_distill_meta.csv"
    pred_path = root / "results/models/runs" / args.run / "best_val_predictions.csv"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    meta = pd.read_csv(meta_path)
    pred = pd.read_csv(pred_path)

    all_files = set(meta["Filename"].astype(str))
    val_files = set(pred["Filename"].astype(str))
    inferred_train_files = all_files - val_files
    overlap = inferred_train_files & val_files

    report = {
        "run": args.run,
        "n_all_files": len(all_files),
        "n_val_files": len(val_files),
        "n_inferred_train_files": len(inferred_train_files),
        "overlap_count": len(overlap),
        "overlap_files": sorted(overlap),
        "note": "This checks leakage based on available output files. For complete proof, save train_files and val_files during training."
    }

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
