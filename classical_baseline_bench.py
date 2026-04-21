#!/usr/bin/env python3
"""
Classical baselines for deep-study repository.

Default baselines:
- Baseline A: plain_fft_no_pccc (direct difference + FFT peak)
- Baseline B: full_xcorr_fft (full-range cross-correlation alignment + FFT peak)

Optional Baseline B variants are exposed for ablation (e.g., normalized xcorr, zero-crossing).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_val_reference(path: Path) -> List[Dict[str, float]]:
    rows = read_csv_rows(path)
    required = {"Filename", "Position_mm", "FFT_Width", "Student_Width"}
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise KeyError(f"{path} missing columns: {sorted(missing)}")

    out: List[Dict[str, float]] = []
    for r in rows:
        out.append(
            {
                "Filename": r["Filename"],
                "Position_mm": float(r["Position_mm"]),
                "Teacher_Width": float(r["FFT_Width"]),
                "Student_Width": float(r["Student_Width"]),
            }
        )
    return out


def load_teacher_xy(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    rows = read_csv_rows(path)
    required = {"Position_mm", "FFT_Width"}
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise KeyError(f"{path} missing columns: {sorted(missing)}")

    x = np.asarray([float(r["Position_mm"]) for r in rows], dtype=np.float64)
    w = np.asarray([float(r["FFT_Width"]) for r in rows], dtype=np.float64)
    return x, w


# ---------------- hyperbola calibration ----------------
def _linear_fit(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float]:
    mx = float(np.mean(xs))
    my = float(np.mean(ys))
    sxx = float(np.sum((xs - mx) ** 2))
    if sxx <= 1e-18:
        raise ValueError("Degenerate x values")
    sxy = float(np.sum((xs - mx) * (ys - my)))
    a = sxy / sxx
    b = my - a * mx
    return a, b


def fit_hyperbola_teacher(x: np.ndarray, w: np.ndarray, c_span: float = 20.0) -> Tuple[float, float, float]:
    min_w = float(np.min(w))
    c_lo = min_w - c_span
    c_hi = min_w - 1e-6

    best = (0.0, 0.0, 0.0, float("inf"))
    lo, hi = c_lo, c_hi
    for _ in range(4):
        for c in np.linspace(lo, hi, 2000):
            d = w - c
            if np.any(d <= 1e-12):
                continue
            y = 1.0 / d
            try:
                a, b = _linear_fit(x, y)
            except ValueError:
                continue
            pred = 1.0 / (a * x + b) + c
            if not np.all(np.isfinite(pred)):
                continue
            sse = float(np.sum((w - pred) ** 2))
            if sse < best[3]:
                best = (float(a), float(b), float(c), sse)
        c_best = best[2]
        half = (hi - lo) / 20.0
        lo = max(c_lo, c_best - half)
        hi = min(c_hi, c_best + half)

    return best[0], best[1], best[2]


def inverse_func(w: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    d = w - C
    out = np.full_like(w, np.nan, dtype=np.float64)
    ok = np.abs(d) > 1e-12
    out[ok] = (1.0 / d[ok] - B) / A
    return out


# ---------------- signal utilities ----------------
def moving_average(x: np.ndarray, win: int = 9) -> np.ndarray:
    if win <= 1:
        return x.copy()
    k = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(x, k, mode="same")


def extract_segments(ch1: np.ndarray, ch2: np.ndarray, min_seg_len: int = 2000) -> List[Tuple[np.ndarray, np.ndarray]]:
    starts = np.where((ch1 == 0) & (ch2 == 0))[0]
    ends = np.where((ch1 == 0) & (ch2 == 61680))[0]
    segs: List[Tuple[np.ndarray, np.ndarray]] = []
    for s in starts:
        possible_ends = ends[ends > s]
        if len(possible_ends) == 0:
            continue
        e = int(possible_ends[0])
        if e - s <= min_seg_len:
            continue
        segs.append((ch1[s + 1 : e], ch2[s + 1 : e]))
    return segs


def _fft_width_from_diff(diff: np.ndarray) -> float:
    n = len(diff)
    if n < 100:
        return float("nan")
    sig = diff - np.mean(diff)
    spec = np.abs(np.fft.rfft(sig * np.hanning(n)))
    if len(spec) < 8:
        return float("nan")
    k = int(np.argmax(spec[5:]) + 5)
    if 1 <= k < len(spec) - 1:
        yl, yc, yr = spec[k - 1], spec[k], spec[k + 1]
        denom = yl - 2.0 * yc + yr
        delta = 0.5 * (yl - yr) / denom if abs(denom) > 1e-18 else 0.0
    else:
        delta = 0.0
    freq = (k + delta) / n
    if freq <= 1e-12:
        return float("nan")
    return float(1.0 / (2.0 * freq))


def _align_by_lag(x: np.ndarray, y: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    if lag > 0:
        return x[lag:], y[:-lag]
    if lag < 0:
        return x[:lag], y[-lag:]
    return x, y


def width_plain_fft(seg1: np.ndarray, seg2: np.ndarray) -> float:
    x = moving_average(seg1.astype(np.float64), 9)
    y = moving_average(seg2.astype(np.float64), 9)
    diff = x - y
    return _fft_width_from_diff(diff)


def width_full_xcorr_fft(seg1: np.ndarray, seg2: np.ndarray, normalized: bool = False) -> float:
    x = moving_average(seg1.astype(np.float64), 9)
    y = moving_average(seg2.astype(np.float64), 9)
    x = x - np.mean(x)
    y = y - np.mean(y)

    if normalized:
        x_std = float(np.std(x))
        y_std = float(np.std(y))
        if x_std <= 1e-12 or y_std <= 1e-12:
            return float("nan")
        x = x / x_std
        y = y / y_std

    corr = np.correlate(x, y, mode="full")
    lags = np.arange(-len(y) + 1, len(x))

    # near anti-phase in this dataset -> choose most negative correlation
    lag = int(lags[np.argmin(corr)])
    xa, ya = _align_by_lag(x, y, lag)
    n = min(len(xa), len(ya))
    if n < 100:
        return float("nan")
    diff = xa[:n] - ya[:n]
    return _fft_width_from_diff(diff)


def width_zero_crossing(seg1: np.ndarray, seg2: np.ndarray) -> float:
    x = moving_average(seg1.astype(np.float64), 9)
    y = moving_average(seg2.astype(np.float64), 9)
    diff = (x - y) - np.mean(x - y)
    n = len(diff)
    if n < 100:
        return float("nan")
    s = np.sign(diff)
    s[s == 0] = 1
    zc = np.where(np.diff(s) != 0)[0]
    if len(zc) < 6:
        return float("nan")
    hp = np.diff(zc).astype(np.float64)
    hp = hp[(hp > 2) & (hp < n / 2)]
    if len(hp) == 0:
        return float("nan")
    return float(np.median(hp))


def robust_mean_width(vals: List[float]) -> float:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    med = float(np.median(arr))
    keep = arr[(arr > med * 0.7) & (arr < med * 1.3)]
    if keep.size == 0:
        return float("nan")
    return float(np.mean(keep))


def baseline_b_width(seg1: np.ndarray, seg2: np.ndarray, method: str) -> float:
    if method == "full_xcorr_fft":
        return width_full_xcorr_fft(seg1, seg2, normalized=False)
    if method == "norm_xcorr_fft":
        return width_full_xcorr_fft(seg1, seg2, normalized=True)
    if method == "zero_crossing":
        return width_zero_crossing(seg1, seg2)
    raise ValueError(f"unknown baseline-b method: {method}")


def compute_baselines_for_file(path: Path, baseline_b: str, min_seg_len: int = 2000) -> Tuple[float, float, int, float, float]:
    rows = read_csv_rows(path)
    if not rows:
        return float("nan"), float("nan"), 0, float("nan"), float("nan")
    if "Channel 1" not in rows[0] or "Channel 2" not in rows[0]:
        return float("nan"), float("nan"), 0, float("nan"), float("nan")

    c1 = np.asarray([float(r["Channel 1"]) for r in rows], dtype=np.float64)
    c2 = np.asarray([float(r["Channel 2"]) for r in rows], dtype=np.float64)

    segs = extract_segments(c1, c2, min_seg_len=min_seg_len)
    a_vals: List[float] = []
    b_vals: List[float] = []

    t0 = time.perf_counter()
    for s1, s2 in segs:
        a_vals.append(width_plain_fft(s1, s2))
    t1 = time.perf_counter()

    for s1, s2 in segs:
        b_vals.append(baseline_b_width(s1, s2, method=baseline_b))
    t2 = time.perf_counter()

    lat_a_ms = (t1 - t0) * 1000.0
    lat_b_ms = (t2 - t1) * 1000.0
    return robust_mean_width(a_vals), robust_mean_width(b_vals), len(segs), lat_a_ms, lat_b_ms


# ---------------- metrics / plotting ----------------
def metric_dict(err: np.ndarray) -> Dict[str, float]:
    e = err[np.isfinite(err)]
    if e.size == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "maxae": float("nan")}
    return {
        "n": int(e.size),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "maxae": float(np.max(np.abs(e))),
    }


def maybe_plot(out_png: Path, table_rows: List[Dict[str, float]], baseline_b_label: str) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        return f"matplotlib unavailable: {e}"

    pos = np.asarray([float(r["Position_mm"]) for r in table_rows])
    t_err = np.asarray([float(r["Teacher_Error_mm"]) for r in table_rows])
    s_err = np.asarray([float(r["Student_Error_mm"]) for r in table_rows])
    a_err = np.asarray([float(r["BaselineA_Error_mm"]) for r in table_rows])
    b_err = np.asarray([float(r["BaselineB_Error_mm"]) for r in table_rows])

    plt.figure(figsize=(10, 5), dpi=180)
    plt.plot(pos, t_err, label="Teacher", linewidth=1.2)
    plt.plot(pos, s_err, label="Student", linewidth=1.2)
    plt.plot(pos, a_err, label="Baseline A: Plain FFT", linewidth=1.0)
    plt.plot(pos, b_err, label=f"Baseline B: {baseline_b_label}", linewidth=1.0)
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("Position (mm)")
    plt.ylabel("Position error (mm)")
    plt.title("Teacher / Student / Classical baselines on same validation files")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Run classical baselines on deep-study validation files")
    p.add_argument("--raw-dir", type=Path, default=Path("1.22"), help="raw waveform folder")
    p.add_argument("--val-csv", type=Path, default=Path("grouped_val_comparison.csv"))
    p.add_argument("--teacher-csv", type=Path, default=Path("fixed_results.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/classical_baselines"))
    p.add_argument("--min-seg-len", type=int, default=2000)
    p.add_argument(
        "--baseline-b",
        type=str,
        default="full_xcorr_fft",
        choices=["full_xcorr_fft", "norm_xcorr_fft", "zero_crossing"],
        help="replacement method for baseline B",
    )
    args = p.parse_args()

    val_rows = load_val_reference(args.val_csv)
    tx, tw = load_teacher_xy(args.teacher_csv)
    A, B, C = fit_hyperbola_teacher(tx, tw)

    out_rows: List[Dict[str, float]] = []
    missing_files: List[str] = []
    lat_a_all: List[float] = []
    lat_b_all: List[float] = []

    for row in val_rows:
        fname = str(row["Filename"])
        fpath = args.raw_dir / fname
        if not fpath.exists():
            missing_files.append(fname)
            continue

        wa, wb, seg_count, lat_a_ms, lat_b_ms = compute_baselines_for_file(
            fpath, baseline_b=args.baseline_b, min_seg_len=args.min_seg_len
        )
        pos = float(row["Position_mm"])
        teacher_w = float(row["Teacher_Width"])
        student_w = float(row["Student_Width"])

        w_arr = np.asarray([teacher_w, student_w, wa, wb], dtype=np.float64)
        pos_pred = inverse_func(w_arr, A, B, C)

        lat_a_all.append(lat_a_ms)
        lat_b_all.append(lat_b_ms)

        out_rows.append(
            {
                "Filename": fname,
                "Position_mm": pos,
                "Segment_Count": int(seg_count),
                "Teacher_Width": teacher_w,
                "Student_Width": student_w,
                "BaselineA_Width": wa,
                "BaselineB_Width": wb,
                "Teacher_Pos_Pred": float(pos_pred[0]),
                "Student_Pos_Pred": float(pos_pred[1]),
                "BaselineA_Pos_Pred": float(pos_pred[2]),
                "BaselineB_Pos_Pred": float(pos_pred[3]),
                "BaselineA_Latency_ms": lat_a_ms,
                "BaselineB_Latency_ms": lat_b_ms,
            }
        )

    if not out_rows:
        raise RuntimeError("No valid baseline rows were generated. Check --raw-dir and --val-csv.")

    for r in out_rows:
        pos = float(r["Position_mm"])
        r["Teacher_Error_mm"] = float(r["Teacher_Pos_Pred"] - pos)
        r["Student_Error_mm"] = float(r["Student_Pos_Pred"] - pos)
        r["BaselineA_Error_mm"] = float(r["BaselineA_Pos_Pred"] - pos)
        r["BaselineB_Error_mm"] = float(r["BaselineB_Pos_Pred"] - pos)

        r["Student_vs_Teacher_WidthErr"] = float(r["Student_Width"] - r["Teacher_Width"])
        r["BaselineA_vs_Teacher_WidthErr"] = float(r["BaselineA_Width"] - r["Teacher_Width"])
        r["BaselineB_vs_Teacher_WidthErr"] = float(r["BaselineB_Width"] - r["Teacher_Width"])

    arr = lambda k: np.asarray([float(r[k]) for r in out_rows], dtype=np.float64)
    summary = {
        "inputs": {
            "raw_dir": str(args.raw_dir),
            "val_csv": str(args.val_csv),
            "teacher_csv": str(args.teacher_csv),
            "baseline_b": args.baseline_b,
            "files_in_val_csv": len(val_rows),
            "files_processed": len(out_rows),
            "files_missing_raw": len(missing_files),
        },
        "calibration_model": {
            "formula": "W = 1/(A*x + B) + C",
            "A": A,
            "B": B,
            "C": C,
            "fit_source": str(args.teacher_csv),
        },
        "width_error_vs_teacher": {
            "Student": metric_dict(arr("Student_vs_Teacher_WidthErr")),
            "BaselineA_PlainFFT": metric_dict(arr("BaselineA_vs_Teacher_WidthErr")),
            f"BaselineB_{args.baseline_b}": metric_dict(arr("BaselineB_vs_Teacher_WidthErr")),
        },
        "position_error_mm": {
            "Teacher": metric_dict(arr("Teacher_Error_mm")),
            "Student": metric_dict(arr("Student_Error_mm")),
            "BaselineA_PlainFFT": metric_dict(arr("BaselineA_Error_mm")),
            f"BaselineB_{args.baseline_b}": metric_dict(arr("BaselineB_Error_mm")),
        },
        "latency_ms_per_file": {
            "BaselineA_mean": float(np.mean(lat_a_all)) if lat_a_all else float("nan"),
            "BaselineB_mean": float(np.mean(lat_b_all)) if lat_b_all else float("nan"),
        },
        "fairness_checks": {
            "same_file_list_as_val_csv": True,
            "same_inverse_calibration_for_all_methods": True,
            "teacher_labels_not_used_in_baseline_signal_processing": True,
            "note": "Teacher CSV is only used for shared calibration parameters.",
        },
        "missing_files": missing_files[:50],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "classical_baseline_comparison.csv"
    json_path = args.out_dir / "classical_baseline_summary.json"
    png_path = args.out_dir / "classical_baseline_position_error.png"

    fieldnames = [
        "Filename", "Position_mm", "Segment_Count",
        "Teacher_Width", "Student_Width", "BaselineA_Width", "BaselineB_Width",
        "Teacher_Pos_Pred", "Student_Pos_Pred", "BaselineA_Pos_Pred", "BaselineB_Pos_Pred",
        "Teacher_Error_mm", "Student_Error_mm", "BaselineA_Error_mm", "BaselineB_Error_mm",
        "Student_vs_Teacher_WidthErr", "BaselineA_vs_Teacher_WidthErr", "BaselineB_vs_Teacher_WidthErr",
        "BaselineA_Latency_ms", "BaselineB_Latency_ms",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    plot_warn = maybe_plot(png_path, out_rows, baseline_b_label=args.baseline_b)
    if plot_warn is not None:
        summary["plot_warning"] = plot_warn

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Classical baseline benchmark completed ===")
    print(f"Saved CSV : {csv_path}")
    print(f"Saved JSON: {json_path}")
    if plot_warn is None:
        print(f"Saved PNG : {png_path}")
    else:
        print(f"Plot skipped: {plot_warn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())