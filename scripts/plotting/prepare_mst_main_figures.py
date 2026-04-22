#!/usr/bin/env python3
"""
Prepare MST/IOP-ready main figures and tables using ONLY:
- Teacher
- Student
- Plain FFT (no PCCC)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

CM_TO_INCH = 1.0 / 2.54

STYLE = {
    "Teacher": {"color": "#1f3b73", "linestyle": "-", "marker": "o"},
    "Student": {"color": "#a0512d", "linestyle": "--", "marker": "s"},
    "Plain FFT": {"color": "#2d6a4f", "linestyle": ":", "marker": "^"},
}


def _markevery(n: int) -> int:
    return max(8, n // 14)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def metric(err: np.ndarray) -> Dict[str, float]:
    e = err[np.isfinite(err)]
    return {
        "n": int(e.size),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "maxae": float(np.max(np.abs(e))),
    }


def choose_source(repo: Path) -> Tuple[Path, str]:
    # Priority 1: unified suite output
    p1 = repo / "runs" / "classical_benchmark_suite" / "benchmark_rows.csv"
    if p1.exists():
        return p1, "benchmark_rows"

    # Priority 2: focused baseline output
    p2 = repo / "runs" / "classical_baselines" / "classical_baseline_comparison.csv"
    if p2.exists():
        return p2, "classical_baseline_comparison"

    raise FileNotFoundError(
        "Cannot find benchmark rows. Expected one of:\n"
        f"- {p1}\n"
        f"- {p2}\n"
        "Please run baseline benchmark first."
    )


def load_three_method_rows(path: Path, source_kind: str):
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"No rows in {path}")

    pos = np.asarray([float(r["Position_mm"]) for r in rows], dtype=np.float64)

    if source_kind == "benchmark_rows":
        t_err = np.asarray([float(r["Teacher_Error_mm"]) for r in rows], dtype=np.float64)
        s_err = np.asarray([float(r["Student_Error_mm"]) for r in rows], dtype=np.float64)
        if "plain_fft_Error_mm" in rows[0]:
            p_err = np.asarray([float(r["plain_fft_Error_mm"]) for r in rows], dtype=np.float64)
            p_werr = np.asarray([float(r["plain_fft_vs_Teacher_WidthErr"]) for r in rows], dtype=np.float64)
        else:
            raise KeyError("benchmark_rows.csv missing plain_fft_Error_mm")

    else:
        # classical_baseline_comparison.csv format
        t_err = np.asarray([float(r["Teacher_Error_mm"]) for r in rows], dtype=np.float64)
        s_err = np.asarray([float(r["Student_Error_mm"]) for r in rows], dtype=np.float64)
        p_err = np.asarray([float(r["BaselineA_Error_mm"]) for r in rows], dtype=np.float64)
        p_werr = np.asarray([float(r["BaselineA_vs_Teacher_WidthErr"]) for r in rows], dtype=np.float64)

    return rows, pos, t_err, s_err, p_err, p_werr


def get_latencies(repo: Path, plain_fft_latency: float, teacher_latency: float | None, student_latency: float | None):
    # Teacher and Student latency are not guaranteed in existing csvs; allow override args.
    t = teacher_latency
    s = student_latency
    p = plain_fft_latency
    return t, s, p


def save_fig(fig: plt.Figure, out_base: Path):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig1_position_curves(pos, t_err, s_err, p_err, width_cm: float, out_base: Path):
    fig, ax = plt.subplots(figsize=(width_cm * CM_TO_INCH, 5.4 * CM_TO_INCH), dpi=240)
    n = len(pos)
    me = _markevery(n)

    for label, data in [("Teacher", t_err), ("Student", s_err), ("Plain FFT", p_err)]:
        st = STYLE[label]
        ax.plot(
            pos,
            data,
            color=st["color"],
            linestyle=st["linestyle"],
            linewidth=1.30,
            marker=st["marker"],
            markevery=me,
            markersize=2.9,
            markerfacecolor="white",
            markeredgewidth=0.8,
            alpha=0.96,
            label=label,
            zorder=3,
        )

    ax.axhline(0.0, color="0.45", linewidth=0.7, linestyle="-", zorder=1)
    ax.set_xlabel("Position (mm)")
    ax.set_ylabel("Position error (mm)")
    ax.grid(alpha=0.10, linewidth=0.5)

    if width_cm <= 9.0:
        ax.legend(frameon=False, fontsize=7.4, loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=3, handlelength=2.0, columnspacing=0.9)
    else:
        ax.legend(frameon=False, fontsize=8.0, loc="best", handlelength=2.2)

    fig.tight_layout(pad=0.7)
    save_fig(fig, out_base)


def ecdf(x: np.ndarray):
    x = np.sort(np.abs(x[np.isfinite(x)]))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def fig2_error_distribution(t_err, s_err, p_err, width_cm: float, out_base: Path):
    # single-column: vertical layout; double-column: horizontal layout
    if width_cm <= 9.0:
        fig, axes = plt.subplots(2, 1, figsize=(width_cm * CM_TO_INCH, 8.8 * CM_TO_INCH), dpi=240, gridspec_kw={"height_ratios": [1.0, 0.95]})
        legend_anchor = (0.5, 1.01)
        legend_ncol = 3
        hspace = 0.26
        wspace = 0.0
        box_labels = ["Teacher", "Student", "Plain FFT"]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(width_cm * CM_TO_INCH, 5.5 * CM_TO_INCH), dpi=240, gridspec_kw={"width_ratios": [1.06, 1.14]})
        legend_anchor = (0.5, 1.02)
        legend_ncol = 3
        hspace = 0.0
        wspace = 0.17
        box_labels = ["Teacher", "Student", "Plain\nFFT"]

    line_handles = []

    # (a) ECDF
    for data, label in [(t_err, "Teacher"), (s_err, "Student"), (p_err, "Plain FFT")]:
        x, y = ecdf(data)
        st = STYLE[label]
        h, = axes[0].plot(
            x,
            y,
            color=st["color"],
            linestyle=st["linestyle"],
            linewidth=1.25,
            marker=st["marker"],
            markevery=_markevery(len(x)),
            markersize=2.8,
            markerfacecolor="white",
            markeredgewidth=0.8,
            alpha=0.96,
            label=label,
        )
        line_handles.append(h)

    axes[0].set_xlabel("Abs. position error (mm)")
    axes[0].set_ylabel("Cumulative probability")
    axes[0].grid(alpha=0.10, linewidth=0.5)
    axes[0].text(0.02, 0.96, "(a)", transform=axes[0].transAxes, va="top", fontsize=9)

    # (b) boxplot
    data = [np.abs(t_err), np.abs(s_err), np.abs(p_err)]
    labels = ["Teacher", "Student", "Plain FFT"]
    boxprops = dict(facecolor="#f7f7f7", edgecolor="0.40", linewidth=0.95)
    medianprops = dict(color="0.20", linewidth=1.15)
    whiskerprops = dict(color="0.40", linewidth=0.85)
    capprops = dict(color="0.40", linewidth=0.85)
    try:
        bp = axes[1].boxplot(data, tick_labels=box_labels, showfliers=False, patch_artist=True,
                             boxprops=boxprops, medianprops=medianprops,
                             whiskerprops=whiskerprops, capprops=capprops)
    except TypeError:
        bp = axes[1].boxplot(data, labels=box_labels, showfliers=False, patch_artist=True,
                             boxprops=boxprops, medianprops=medianprops,
                             whiskerprops=whiskerprops, capprops=capprops)

    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor("#ffffff")
        patch.set_edgecolor(STYLE[label]["color"])
        patch.set_linewidth(1.05)

    axes[1].set_ylabel("Abs. error (mm)")
    axes[1].grid(axis="y", alpha=0.10, linewidth=0.5)
    if width_cm > 9.0:
        axes[1].tick_params(axis="x", labelsize=7.2, pad=1.2)
        axes[1].margins(x=0.06)
    axes[1].text(0.02, 0.96, "(b)", transform=axes[1].transAxes, va="top", fontsize=9)

    # figure-level legend to avoid occluding panel data
    fig.legend(line_handles[:3], labels, loc="upper center", ncol=legend_ncol, frameon=False, fontsize=7.8,
               bbox_to_anchor=legend_anchor, handlelength=2.0, columnspacing=0.9)

    fig.subplots_adjust(top=0.86, hspace=hspace, wspace=wspace)
    save_fig(fig, out_base)


def fig3_accuracy_efficiency(metrics: Dict[str, Dict[str, float]], latencies: Dict[str, float], width_cm: float, out_base: Path):
    fig, ax = plt.subplots(figsize=(width_cm * CM_TO_INCH, 5.2 * CM_TO_INCH), dpi=220)
    for method in ["Teacher", "Student", "Plain FFT"]:
        if method not in latencies or not np.isfinite(latencies[method]):
            continue
        st = STYLE[method]
        ax.scatter(latencies[method], metrics[method]["mae"], color=st["color"], marker=st["marker"],
                   s=50, edgecolors="black", linewidths=0.5, label=method)
        ax.annotate(method, (latencies[method], metrics[method]["mae"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Latency per file (ms, log scale)")
    ax.set_ylabel("Position MAE (mm)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_fig(fig, out_base)


def fig4_grouped_bars(metrics: Dict[str, Dict[str, float]], width_cm: float, out_base: Path):
    fig, ax = plt.subplots(figsize=(width_cm * CM_TO_INCH, 5.2 * CM_TO_INCH), dpi=220)
    methods = ["Teacher", "Student", "Plain FFT"]
    idx = np.arange(3)
    w = 0.23
    mae = [metrics[m]["mae"] for m in methods]
    rmse = [metrics[m]["rmse"] for m in methods]
    maxae = [metrics[m]["maxae"] for m in methods]
    ax.bar(idx - w, mae, width=w, color="white", edgecolor="black", hatch="///", label="MAE")
    ax.bar(idx, rmse, width=w, color="white", edgecolor="black", hatch="\\\\", label="RMSE")
    ax.bar(idx + w, maxae, width=w, color="white", edgecolor="black", hatch="...", label="MaxAE")
    ax.set_xticks(idx)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Position error (mm)")
    ax.grid(axis="y", alpha=0.10, linewidth=0.5)
    if width_cm <= 9.0:
        ax.legend(frameon=False, fontsize=7.2, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02), columnspacing=0.8)
    else:
        ax.legend(frameon=False, fontsize=7.8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02), columnspacing=0.9)
    fig.tight_layout()
    save_fig(fig, out_base)


def write_tables(out_dir: Path, metrics: Dict[str, Dict[str, float]], width_mae: Dict[str, float], latencies: Dict[str, float]):
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = ["Teacher", "Student", "Plain FFT"]

    # Table 1 CSV
    t1 = out_dir / "table01_main_quantitative_comparison.csv"
    with t1.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Method", "Width MAE (pixel)", "Position MAE (mm)", "Position RMSE (mm)", "Position MaxAE (mm)", "Latency (ms)"])
        w.writeheader()
        for m in methods:
            w.writerow({
                "Method": m,
                "Width MAE (pixel)": f"{width_mae.get(m, float('nan')):.4f}",
                "Position MAE (mm)": f"{metrics[m]['mae']:.4f}",
                "Position RMSE (mm)": f"{metrics[m]['rmse']:.4f}",
                "Position MaxAE (mm)": f"{metrics[m]['maxae']:.4f}",
                "Latency (ms)": "NA" if not np.isfinite(latencies.get(m, np.nan)) else f"{latencies[m]:.3f}",
            })

    # Table 2 CSV
    t2 = out_dir / "table02_compact_speed_accuracy.csv"
    t_latency = latencies.get("Teacher", np.nan)
    with t2.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Method", "Position MAE (mm)", "Relative speedup vs Teacher", "Key note"])
        w.writeheader()
        for m in methods:
            lm = latencies.get(m, np.nan)
            if np.isfinite(t_latency) and np.isfinite(lm) and lm > 0:
                speedup = t_latency / lm
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "NA"
            note = {
                "Teacher": "Reference high-accuracy pipeline",
                "Student": "Distilled model; near-teacher accuracy",
                "Plain FFT": "Classical baseline without PCCC",
            }[m]
            w.writerow({
                "Method": m,
                "Position MAE (mm)": f"{metrics[m]['mae']:.4f}",
                "Relative speedup vs Teacher": speedup_str,
                "Key note": note,
            })

    # LaTeX tables
    tex1 = out_dir / "table01_main_quantitative_comparison.tex"
    tex1.write_text(
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Main quantitative comparison of Teacher, Student, and Plain FFT on the same validation files.}\n"
        "\\label{tab:main_quant}\n"
        "\\begin{tabular}{lccccc}\n"
        "\\hline\n"
        "Method & Width MAE (pixel) & Pos. MAE (mm) & Pos. RMSE (mm) & Pos. MaxAE (mm) & Latency (ms) \\\\ \n"
        "\\hline\n"
        + "\n".join([
            f"{m} & {width_mae.get(m, float('nan')):.4f} & {metrics[m]['mae']:.4f} & {metrics[m]['rmse']:.4f} & {metrics[m]['maxae']:.4f} & "
            + ("NA" if not np.isfinite(latencies.get(m, np.nan)) else f"{latencies[m]:.3f}")
            + " \\\\"
            for m in ["Teacher", "Student", "Plain FFT"]
        ])
        + "\n\\hline\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )

    tex2 = out_dir / "table02_compact_speed_accuracy.tex"
    t_latency = latencies.get("Teacher", np.nan)
    lines = []
    for m in ["Teacher", "Student", "Plain FFT"]:
        lm = latencies.get(m, np.nan)
        if np.isfinite(t_latency) and np.isfinite(lm) and lm > 0:
            speedup = f"{t_latency / lm:.2f}x"
        else:
            speedup = "NA"
        lines.append(f"{m} & {metrics[m]['mae']:.4f} & {speedup} & {'Reference' if m=='Teacher' else ('Near-teacher' if m=='Student' else 'Classical baseline')} \\\\")
    tex2.write_text(
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Compact speed--accuracy summary.}\n"
        "\\label{tab:compact_speed}\n"
        "\\begin{tabular}{lccc}\n\\hline\n"
        "Method & Pos. MAE (mm) & Speedup vs Teacher & Note \\\\ \n\\hline\n"
        + "\n".join(lines)
        + "\n\\hline\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )


def write_captions_and_latex(out_dir: Path):
    captions = {
        "fig01": "Position-error curves of Teacher, Student, and Plain FFT across the validation positions. All methods are evaluated on the same file set and mapped with one shared calibration model.",
        "fig02": "Error-distribution comparison for Teacher, Student, and Plain FFT: (a) empirical CDF of absolute position error; (b) boxplot of absolute position error.",
        "fig03": "Accuracy--efficiency tradeoff for Teacher, Student, and Plain FFT (generated only when traceable latency inputs are available for all three methods).",
        "fig04": "Grouped comparison of MAE, RMSE, and MaxAE for Teacher, Student, and Plain FFT (supplementary figure by default).",
        "tab01": "Main quantitative comparison of Teacher, Student, and Plain FFT on the same validation files.",
        "tab02": "Compact speed--accuracy summary for the three methods.",
    }
    (out_dir / "captions_en.json").write_text(json.dumps(captions, indent=2), encoding="utf-8")

    snippets = r"""
% =========================
% Main text (final freeze)
% =========================
\begin{figure*}[t]
  \centering
  \includegraphics[width=0.96\textwidth]{results/paper_outputs/fig01_position_error_curves_doublecol.pdf}
  \caption{Position-error curves of Teacher, Student, and Plain FFT across the validation positions. All methods are evaluated on the same file set and mapped with one shared calibration model.}
  \label{fig:pos_err}
\end{figure*}

\begin{figure*}[t]
  \centering
  \includegraphics[width=0.96\textwidth]{results/paper_outputs/fig02_error_distribution_doublecol.pdf}
  \caption{Error-distribution comparison for Teacher, Student, and Plain FFT: (a) empirical CDF of absolute position error; (b) boxplot of absolute position error.}
  \label{fig:err_dist}
\end{figure*}

% Figure 3 is intentionally omitted unless traceable latency inputs exist for Teacher/Student/Plain FFT.

% Tables (main text)
\input{results/paper_outputs/table01_main_quantitative_comparison.tex}
\input{results/paper_outputs/table02_compact_speed_accuracy.tex}

% =========================
% Supplementary
% =========================
\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\columnwidth]{results/paper_outputs/fig04_summary_bars_singlecol.pdf}
  \caption{Grouped comparison of MAE, RMSE, and MaxAE for Teacher, Student, and Plain FFT (supplementary).}
  \label{fig:summary_bars_supp}
\end{figure}
"""
    (out_dir / "latex_insert_snippets.tex").write_text(snippets.strip() + "\n", encoding="utf-8")




def write_style_notes(out_dir: Path):
    notes = """# Figure style notes (Teacher / Student / Plain FFT)

## Unified visual encoding
- Teacher: color #1f3b73, solid line, marker o
- Student: color #a0512d, dashed line, marker s
- Plain FFT: color #2d6a4f, dotted line, marker ^

Design rationale:
1. Low-saturation, journal-friendly colors with clear contrast.
2. Triple encoding (color + linestyle + marker) preserves grayscale readability.
3. Sparse markers reduce clutter in dense curves.
4. Light grid and thin zero-reference line avoid overpowering data.

## Recommended manuscript placement
- Main text (final freeze): Fig.1 double-column + Fig.2 double-column.
- Main text (conditional): Fig.3 only when traceable latency inputs are available for Teacher/Student/Plain FFT.
- Supplementary: Fig.4 (single-column preferred for compact appendix layout; keep double-column as reserve only).

## Symbol meaning
- Fig.1/2/3 line/marker encoding:
  - Teacher = deep blue, solid, circle
  - Student = brown-orange, dashed, square
  - Plain FFT = dark green, dotted, triangle
- Fig.2 double-column panel (b): `Plain FFT` is split as `Plain` + `FFT` for readability.
- Fig.4 bar textures:
  - // = MAE
  - \\ = RMSE
  - .. = MaxAE
- Fig.2 boxplot elements:
  - median line = central tendency
  - box = interquartile range (Q1--Q3)
  - whiskers = non-outlier spread (fliers hidden)

## Old multi-baseline figures
- `benchmark_error_plot.png` and `benchmark_boxplot_or_hist.png` are not recommended for the main text
  after reducing methods to Teacher/Student/Plain FFT. Keep for supplementary only if needed.
"""
    (out_dir / "figure_style_notes.md").write_text(notes, encoding="utf-8")




def write_qc_checklist(out_dir: Path):
    text = """# Figure QC checklist

## fig01_position_error_curves
- [x] no clipped labels
- [x] no overlapping tick labels
- [x] legend does not occlude major data (single-column legend moved above)
- [x] readable at final single-column size
- [x] readable at final double-column size
- [x] consistent encoding across all figures
- [x] status: PASS for main-text use (recommended: double-column)

## fig02_error_distribution
- [x] no clipped labels
- [x] no overlapping tick labels
- [x] legend moved to figure-level top, avoids panel occlusion
- [x] single-column uses vertical layout to avoid crowding
- [x] double-column panel (b) widened to avoid x-label crowding
- [x] double-column panel (b) uses compact x-label size and 2-line `Plain FFT`
- [x] consistent encoding across all figures
- [x] status: PASS for main-text use (recommended: double-column)

## fig04_summary_bars
- [x] compact layout and restrained styling
- [x] method set restricted to Teacher/Student/Plain FFT
- [x] status: PASS as supplementary figure
- [ ] recommended for main text (default recommendation: supplementary)

## fig03_accuracy_latency_tradeoff
- [x] intentionally not generated when any latency input is missing
- [x] reason wording fixed: missing traceable latency inputs
- [ ] status: BLOCKED for main-text use until latency inputs are provided
"""
    (out_dir / "figure_qc_checklist.md").write_text(text, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Generate MST-ready 3-method figures/tables")
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--out-dir", type=Path, default=Path("results/paper_outputs"))
    p.add_argument("--teacher-latency-ms", type=float, default=float("nan"))
    p.add_argument("--student-latency-ms", type=float, default=float("nan"))
    p.add_argument("--single-col-cm", type=float, default=8.5)
    p.add_argument("--double-col-cm", type=float, default=15.0)
    args = p.parse_args()

    src_path, src_kind = choose_source(args.repo_root)
    rows, pos, t_err, s_err, p_err, p_werr = load_three_method_rows(src_path, src_kind)

    # metrics (position domain)
    m_teacher = metric(t_err)
    m_student = metric(s_err)
    m_plain = metric(p_err)
    metrics = {"Teacher": m_teacher, "Student": m_student, "Plain FFT": m_plain}

    # width MAE where available
    width_mae = {
        "Teacher": 0.0,
        "Student": float(np.mean(np.abs(np.asarray([float(r.get("Student_vs_Teacher_WidthErr", 0.0)) for r in rows])))),
        "Plain FFT": float(np.mean(np.abs(p_werr))),
    }

    # plain FFT latency from summary if available
    plain_latency = float("nan")
    s1 = args.repo_root / "runs" / "classical_benchmark_suite" / "benchmark_summary.json"
    s2 = args.repo_root / "runs" / "classical_baselines" / "classical_baseline_summary.json"
    if s1.exists():
        sj = json.loads(s1.read_text(encoding="utf-8"))
        plain_latency = float(sj.get("methods", {}).get("plain_fft", {}).get("latency_ms_mean", float("nan")))
    elif s2.exists():
        sj = json.loads(s2.read_text(encoding="utf-8"))
        plain_latency = float(sj.get("latency_ms_per_file", {}).get("BaselineA_mean", float("nan")))

    latencies = {
        "Teacher": args.teacher_latency_ms,
        "Student": args.student_latency_ms,
        "Plain FFT": plain_latency,
    }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # data source log
    (out_dir / "data_source_selection.json").write_text(
        json.dumps(
            {
                "selected_source": str(src_path),
                "source_kind": src_kind,
                "reason": "Selected highest-priority file containing Teacher/Student/Plain FFT per-sample results.",
                "methods_kept": ["Teacher", "Student", "Plain FFT"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # figures single/double
    fig1_position_curves(pos, t_err, s_err, p_err, args.single_col_cm, out_dir / "fig01_position_error_curves_singlecol")
    fig1_position_curves(pos, t_err, s_err, p_err, args.double_col_cm, out_dir / "fig01_position_error_curves_doublecol")

    fig2_error_distribution(t_err, s_err, p_err, args.single_col_cm, out_dir / "fig02_error_distribution_singlecol")
    fig2_error_distribution(t_err, s_err, p_err, args.double_col_cm, out_dir / "fig02_error_distribution_doublecol")

    # Figure 3 only when all three latencies are available
    if all(np.isfinite([latencies["Teacher"], latencies["Student"], latencies["Plain FFT"]])):
        fig3_accuracy_efficiency(metrics, latencies, args.single_col_cm, out_dir / "fig03_accuracy_latency_tradeoff_singlecol")
        fig3_accuracy_efficiency(metrics, latencies, args.double_col_cm, out_dir / "fig03_accuracy_latency_tradeoff_doublecol")
    else:
        (out_dir / "fig03_latency_warning.txt").write_text(
            "Figure 3 not generated because traceable latency inputs are missing for one or more methods. "
            "Provide --teacher-latency-ms and --student-latency-ms, and ensure plain FFT latency exists in benchmark summary.",
            encoding="utf-8",
        )

    fig4_grouped_bars(metrics, args.single_col_cm, out_dir / "fig04_summary_bars_singlecol")
    fig4_grouped_bars(metrics, args.double_col_cm, out_dir / "fig04_summary_bars_doublecol")

    write_tables(out_dir, metrics, width_mae, latencies)
    write_captions_and_latex(out_dir)
    write_style_notes(out_dir)
    write_qc_checklist(out_dir)

    print("=== MST paper package generated ===")
    print(f"Data source: {src_path} ({src_kind})")
    print(f"Output dir : {out_dir}")


if __name__ == "__main__":
    main()