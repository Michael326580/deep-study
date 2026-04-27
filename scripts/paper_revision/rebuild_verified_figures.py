#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebuild MST/IOP-ready figures from repository-traceable data.

Run from repository root:
    python scripts_paper_revision/rebuild_verified_figures.py --repo-root .

Outputs:
    results/paper_revision_figures/*.pdf
    results/paper_revision_figures/*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def hyperbola(x, a, b, c):
    return 1.0 / (a * x + b) + c


def savefig(out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}.pdf")
    plt.savefig(out_dir / f"{name}.png", dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("results/paper_revision_figures"))
    parser.add_argument("--waveform-file", type=str, default="-257.csv")
    parser.add_argument("--waveform-start", type=int, default=None)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    out_dir = (root / args.out).resolve()

    fixed = pd.read_csv(root / "data/processed/fixed_results.csv")
    val = pd.read_csv(root / "data/processed/grouped_val_comparison.csv")
    hist = pd.read_csv(root / "results/models/runs/attention_res1d_model/train_history.csv")
    table01 = pd.read_csv(root / "results/paper_outputs/paper_outputs/table01_main_quantitative_comparison.csv")

    # Fig. 8a: hyperbolic fit
    x = fixed["Position_mm"].to_numpy(float)
    y = fixed["FFT_Width"].to_numpy(float)
    popt, _ = curve_fit(hyperbola, x, y, p0=[0.001, -0.1, 10.0], maxfev=50000)
    yfit = hyperbola(x, *popt)
    residual = y - yfit
    r2 = 1.0 - float(np.sum((y - yfit) ** 2)) / float(np.sum((y - np.mean(y)) ** 2))

    plt.figure(figsize=(7.0, 4.2))
    plt.scatter(x, y, s=9, label="Teacher width")
    xx = np.linspace(x.min(), x.max(), 600)
    plt.plot(xx, hyperbola(xx, *popt), linewidth=1.4, label=f"Hyperbolic fit, R$^2$={r2:.6f}")
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Fringe width (pixel)")
    plt.legend(frameon=False)
    savefig(out_dir, "fig08a_hyperbolic_fit_verified")

    # Fig. 8b: residuals
    plt.figure(figsize=(7.0, 3.2))
    plt.axhline(0, linewidth=0.9)
    plt.scatter(x, residual, s=9)
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Fit residual (pixel)")
    savefig(out_dir, "fig08b_hyperbolic_residuals_verified")

    # Position error curves
    val = val.sort_values("Position_mm")
    plt.figure(figsize=(7.0, 3.6))
    plt.plot(val["Position_mm"], val["Teacher_Error_mm"], linewidth=1.1, label="Teacher")
    plt.plot(val["Position_mm"], val["Student_Error_mm"], linewidth=1.1, label="Student")
    plt.axhline(0, linewidth=0.8)
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Position error (mm)")
    plt.legend(frameon=False)
    savefig(out_dir, "fig10_position_error_teacher_student_verified")

    # Error CDF
    plt.figure(figsize=(6.5, 3.8))
    for col, label in [("Teacher_AbsErr_mm", "Teacher"), ("Student_AbsErr_mm", "Student")]:
        values = np.sort(val[col].dropna().to_numpy(float))
        cdf = np.arange(1, len(values) + 1) / len(values)
        plt.plot(values, cdf, linewidth=1.5, label=label)
    plt.xlabel("Absolute position error (mm)")
    plt.ylabel("Empirical CDF")
    plt.legend(frameon=False)
    savefig(out_dir, "fig11_error_cdf_verified")

    # Training and validation loss
    plt.figure(figsize=(7.0, 4.0))
    plt.plot(hist["epoch"], hist["train_loss"], linewidth=1.1, label="Train loss")
    plt.plot(hist["epoch"], hist["val_loss"], linewidth=1.1, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend(frameon=False)
    savefig(out_dir, "fig05_training_loss_verified")

    # File-level MAE
    plt.figure(figsize=(7.0, 4.0))
    plt.plot(hist["epoch"], hist["file_mae"], linewidth=1.1, label="File-level MAE")
    idx = hist["file_mae"].idxmin()
    plt.scatter([hist.loc[idx, "epoch"]], [hist.loc[idx, "file_mae"]], s=25, label="Best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("File-level width MAE (pixel)")
    plt.legend(frameon=False)
    savefig(out_dir, "fig05b_validation_file_mae_verified")

    # Accuracy summary
    plt.figure(figsize=(6.8, 4.0))
    methods = table01["Method"].astype(str).tolist()
    mae = table01["Position MAE (mm)"].to_numpy(float)
    plt.bar(methods, mae)
    plt.ylabel("Position MAE (mm)")
    plt.xlabel("Method")
    for i, v in enumerate(mae):
        plt.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    savefig(out_dir, "fig09_accuracy_summary_verified")

    # Representative waveform from raw CSV
    raw_path = root / "data/raw/1.22" / args.waveform_file
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        if args.waveform_start is None:
            start = max(0, len(raw) // 2 - 1024)
        else:
            start = int(args.waveform_start)
        seg = raw.iloc[start:start + 2048]
        ch1 = seg.iloc[:, 0].astype(float).to_numpy()
        ch2 = seg.iloc[:, 1].astype(float).to_numpy()
        ch1 = (ch1 - ch1.mean()) / (ch1.std() + 1e-12)
        ch2 = (ch2 - ch2.mean()) / (ch2.std() + 1e-12)
        diff = ch1 - ch2
        t = np.arange(len(ch1))
        position = fixed.loc[fixed["Filename"] == args.waveform_file, "Position_mm"]
        plt.figure(figsize=(7.0, 3.8))
        plt.plot(t, ch1, linewidth=0.8, label="Channel 1")
        plt.plot(t, ch2, linewidth=0.8, label="Channel 2")
        plt.plot(t, diff, linewidth=0.8, label="Differential")
        plt.xlabel("Sample index (pixel)")
        plt.ylabel("Normalized amplitude")
        if len(position):
            plt.title(f"Representative waveform, {float(position.iloc[0]):.1f} mm")
        plt.legend(frameon=False)
        savefig(out_dir, "fig07_waveform_verified")
    else:
        print(f"Warning: raw waveform file not found: {raw_path}")

    summary = {
        "hyperbola": {"A": float(popt[0]), "B": float(popt[1]), "C": float(popt[2]), "R2": float(r2)},
        "residual_mae_pixel": float(np.mean(np.abs(residual))),
        "residual_maxae_pixel": float(np.max(np.abs(residual))),
        "output_dir": str(out_dir),
    }
    (out_dir / "figure_generation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
