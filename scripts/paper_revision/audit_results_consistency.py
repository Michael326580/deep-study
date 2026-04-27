#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit repository-traceable numerical results for the MST manuscript.

Run from the deep-study repository root:
    python scripts_paper_revision/audit_results_consistency.py --repo-root .
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_NUMBERS = {
    "single_student_mae_mm": 0.0264,
    "ensemble_mae_mm": 0.02195,
    "max_error_after_cleaning_mm": 0.0426,
    "student_latency_ms": 0.04,
}


def f4(x: float) -> str:
    return f"{float(x):.6g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("results/paper_revision_audit"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    out_dir = (root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_path = root / "data/processed/fixed_results.csv"
    meta_path = root / "data/processed/dataset_distill_meta.csv"
    val_path = root / "data/processed/grouped_val_comparison.csv"
    table_path = root / "results/paper_outputs/paper_outputs/table01_main_quantitative_comparison.csv"
    attn_summary_path = root / "results/models/runs/attention_res1d_model/train_summary.json"
    final_summary_path = root / "results/models/runs/final_model/train_summary.json"

    missing = [p for p in [fixed_path, meta_path, val_path, table_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(str(p) for p in missing))

    fixed = pd.read_csv(fixed_path)
    meta = pd.read_csv(meta_path)
    val = pd.read_csv(val_path)
    table = pd.read_csv(table_path)

    report = {
        "fixed_results": {
            "n": int(len(fixed)),
            "position_range_mm": [float(fixed["Position_mm"].min()), float(fixed["Position_mm"].max())],
            "width_range_pixel": [float(fixed["FFT_Width"].min()), float(fixed["FFT_Width"].max())],
            "width_mean_pixel": float(fixed["FFT_Width"].mean()),
        },
        "dataset_distill_meta": {
            "n_segments": int(len(meta)),
            "position_range_mm": [float(meta["Position_mm"].min()), float(meta["Position_mm"].max())],
            "teacher_width_range_pixel": [float(meta["Teacher_Width"].min()), float(meta["Teacher_Width"].max())],
            "teacher_width_mean_pixel": float(meta["Teacher_Width"].mean()),
        },
        "grouped_val_comparison": {
            "n_files": int(len(val)),
            "teacher_mae_mm": float(val["Teacher_AbsErr_mm"].mean()),
            "teacher_maxae_mm": float(val["Teacher_AbsErr_mm"].max()),
            "student_mae_mm": float(val["Student_AbsErr_mm"].mean()),
            "student_maxae_mm": float(val["Student_AbsErr_mm"].max()),
            "teacher_3sigma_n": int(val["Teacher_Keep_3Sigma"].sum()) if "Teacher_Keep_3Sigma" in val.columns else None,
            "student_3sigma_n": int(val["Student_Keep_3Sigma"].sum()) if "Student_Keep_3Sigma" in val.columns else None,
        },
        "paper_table01": table.to_dict(orient="records"),
        "target_number_conflicts": {},
    }

    if attn_summary_path.exists():
        report["attention_res1d_model"] = json.loads(attn_summary_path.read_text(encoding="utf-8"))
    if final_summary_path.exists():
        report["final_model"] = json.loads(final_summary_path.read_text(encoding="utf-8"))

    current_student_mae = report["grouped_val_comparison"]["student_mae_mm"]
    report["target_number_conflicts"]["single_student_mae_mm"] = {
        "target": TARGET_NUMBERS["single_student_mae_mm"],
        "current_grouped_val": current_student_mae,
        "status": "conflict" if abs(current_student_mae - TARGET_NUMBERS["single_student_mae_mm"]) > 1e-6 else "matched",
    }
    report["target_number_conflicts"]["latency_ms"] = {
        "target": TARGET_NUMBERS["student_latency_ms"],
        "current_traceable": "not available in table01 for Student",
        "status": "missing timing log",
    }

    (out_dir / "audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# Repository numerical audit",
        "",
        f"- fixed_results.csv: N={report['fixed_results']['n']}, position={report['fixed_results']['position_range_mm'][0]}--{report['fixed_results']['position_range_mm'][1]} mm, width={f4(report['fixed_results']['width_range_pixel'][0])}--{f4(report['fixed_results']['width_range_pixel'][1])} pixel",
        f"- dataset_distill_meta.csv: N={report['dataset_distill_meta']['n_segments']} segments",
        f"- Teacher MAE: {f4(report['grouped_val_comparison']['teacher_mae_mm'])} mm",
        f"- Student MAE: {f4(report['grouped_val_comparison']['student_mae_mm'])} mm",
        "",
        "## Conflict note",
        "",
        f"- Target single-student MAE 0.0264 mm conflicts with current grouped validation MAE {f4(current_student_mae)} mm.",
        "- Student latency is not traceable unless a timing script and raw timing log are added.",
    ]
    (out_dir / "audit_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved: {out_dir / 'audit_report.json'}")


if __name__ == "__main__":
    main()
