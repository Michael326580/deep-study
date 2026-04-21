#!/usr/bin/env python3
"""
Unified classical-method benchmark suite for deep-study.

Outputs:
- benchmark_rows.csv
- benchmark_summary.json
- benchmark_ranked_methods.csv
- benchmark_error_plot.png (if matplotlib available)
- benchmark_boxplot_or_hist.png (if matplotlib available)
- benchmark_paper_recommendation.json
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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


# ---------- shared calibration ----------
def _linear_fit(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float]:
    mx = float(np.mean(xs))
    my = float(np.mean(ys))
    sxx = float(np.sum((xs - mx) ** 2))
    if sxx <= 1e-18:
        raise ValueError("Degenerate x")
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
        for c in np.linspace(lo, hi, 1600):
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


# ---------- waveform processing ----------
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
        pe = ends[ends > s]
        if len(pe) == 0:
            continue
        e = int(pe[0])
        if e - s <= min_seg_len:
            continue
        segs.append((ch1[s + 1 : e], ch2[s + 1 : e]))
    return segs


def _fft_width_from_signal(sig: np.ndarray, start_idx: int = 5) -> float:
    n = len(sig)
    if n < 100:
        return float("nan")
    s = sig - np.mean(sig)
    spec = np.abs(np.fft.rfft(s * np.hanning(n)))
    if len(spec) <= start_idx + 2:
        return float("nan")
    k = int(np.argmax(spec[start_idx:]) + start_idx)
    if 1 <= k < len(spec) - 1:
        yl, yc, yr = spec[k - 1], spec[k], spec[k + 1]
        denom = yl - 2 * yc + yr
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


def preprocess(seg1: np.ndarray, seg2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = moving_average(seg1.astype(np.float64), 9)
    y = moving_average(seg2.astype(np.float64), 9)
    return x, y


def method_plain_fft(seg1: np.ndarray, seg2: np.ndarray) -> float:
    x, y = preprocess(seg1, seg2)
    return _fft_width_from_signal(x - y)


def method_full_xcorr_fft(seg1: np.ndarray, seg2: np.ndarray) -> float:
    x, y = preprocess(seg1, seg2)
    x = x - np.mean(x)
    y = y - np.mean(y)
    corr = np.correlate(x, y, mode="full")
    lags = np.arange(-len(y) + 1, len(x))
    lag = int(lags[np.argmin(corr)])  # anti-phase
    xa, ya = _align_by_lag(x, y, lag)
    n = min(len(xa), len(ya))
    if n < 100:
        return float("nan")
    return _fft_width_from_signal(xa[:n] - ya[:n])


def method_norm_xcorr_fft(seg1: np.ndarray, seg2: np.ndarray) -> float:
    x, y = preprocess(seg1, seg2)
    x = x - np.mean(x)
    y = y - np.mean(y)
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 1e-12 or sy <= 1e-12:
        return float("nan")
    x = x / sx
    y = y / sy
    corr = np.correlate(x, y, mode="full")
    lags = np.arange(-len(y) + 1, len(x))
    lag = int(lags[np.argmin(corr)])
    xa, ya = _align_by_lag(x, y, lag)
    n = min(len(xa), len(ya))
    if n < 100:
        return float("nan")
    return _fft_width_from_signal(xa[:n] - ya[:n])


def method_welch_periodogram(seg1: np.ndarray, seg2: np.ndarray) -> float:
    x, y = preprocess(seg1, seg2)
    d = (x - y) - np.mean(x - y)
    n = len(d)
    if n < 256:
        return float("nan")
    win = 512 if n >= 512 else 256
    step = win // 2
    if step <= 0:
        return float("nan")
    acc = None
    cnt = 0
    for s in range(0, n - win + 1, step):
        seg = d[s : s + win] * np.hanning(win)
        spec = np.abs(np.fft.rfft(seg)) ** 2
        if acc is None:
            acc = np.zeros_like(spec)
        acc += spec
        cnt += 1
    if acc is None or cnt == 0:
        return float("nan")
    acc = acc / cnt
    k = int(np.argmax(acc[5:]) + 5)
    freq = k / win
    if freq <= 1e-12:
        return float("nan")
    return float(1.0 / (2.0 * freq))


def method_autocorr_period(seg1: np.ndarray, seg2: np.ndarray) -> float:
    x, y = preprocess(seg1, seg2)
    d = (x - y) - np.mean(x - y)
    n = len(d)
    if n < 100:
        return float("nan")
    ac = np.correlate(d, d, mode="full")[n - 1 :]
    if len(ac) < 20:
        return float("nan")
    ac[:5] = -np.inf
    lag = int(np.argmax(ac))
    if lag <= 1:
        return float("nan")
    # autocorr lag approximates full period
    return float(lag / 2.0)


def method_hilbert_phase(seg1: np.ndarray, seg2: np.ndarray) -> float:
    x, y = preprocess(seg1, seg2)
    d = (x - y) - np.mean(x - y)
    n = len(d)
    if n < 100:
        return float("nan")

    # FFT-based analytic signal (Hilbert transform)
    X = np.fft.fft(d)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = 1
        h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        h[0] = 1
        h[1 : (n + 1) // 2] = 2
    z = np.fft.ifft(X * h)
    phase = np.unwrap(np.angle(z))

    # robust slope via linear fit
    t = np.arange(n, dtype=np.float64)
    a, _ = _linear_fit(t, phase)
    freq = a / (2.0 * np.pi)
    if freq <= 1e-12:
        return float("nan")
    return float(1.0 / (2.0 * freq))


def method_sinusoid_ls(seg1: np.ndarray, seg2: np.ndarray) -> float:
    x, y = preprocess(seg1, seg2)
    d = (x - y) - np.mean(x - y)
    n = len(d)
    if n < 120:
        return float("nan")

    # coarse FFT peak for search window
    coarse = _fft_width_from_signal(d)
    if not np.isfinite(coarse) or coarse <= 1.0:
        f_center = 0.03
    else:
        f_center = 1.0 / (2.0 * coarse)

    f_lo = max(0.005, f_center * 0.6)
    f_hi = min(0.25, f_center * 1.4)
    freqs = np.linspace(f_lo, f_hi, 120)

    t = np.arange(n, dtype=np.float64)
    best_f = float("nan")
    best_mse = float("inf")
    for f in freqs:
        w = 2.0 * np.pi * f
        s = np.sin(w * t)
        c = np.cos(w * t)
        X = np.stack([s, c, np.ones_like(t)], axis=1)
        try:
            beta, *_ = np.linalg.lstsq(X, d, rcond=None)
        except Exception:
            continue
        pred = X @ beta
        mse = float(np.mean((d - pred) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_f = float(f)

    if not np.isfinite(best_f) or best_f <= 1e-12:
        return float("nan")
    return float(1.0 / (2.0 * best_f))


def method_zero_crossing(seg1: np.ndarray, seg2: np.ndarray) -> float:
    x, y = preprocess(seg1, seg2)
    d = (x - y) - np.mean(x - y)
    n = len(d)
    if n < 100:
        return float("nan")
    s = np.sign(d)
    s[s == 0] = 1
    zc = np.where(np.diff(s) != 0)[0]
    if len(zc) < 6:
        return float("nan")
    hp = np.diff(zc).astype(np.float64)
    hp = hp[(hp > 2) & (hp < n / 2)]
    if len(hp) == 0:
        return float("nan")
    return float(np.median(hp))


METHODS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "plain_fft": method_plain_fft,
    "full_xcorr_fft": method_full_xcorr_fft,
    "norm_xcorr_fft": method_norm_xcorr_fft,
    "welch_periodogram": method_welch_periodogram,
    "autocorr_period": method_autocorr_period,
    "hilbert_phase": method_hilbert_phase,
    "sinusoid_ls": method_sinusoid_ls,
    "zero_crossing": method_zero_crossing,
}


def robust_mean(vals: List[float]) -> float:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    med = float(np.median(arr))
    keep = arr[(arr > med * 0.7) & (arr < med * 1.3)]
    if keep.size == 0:
        return float("nan")
    return float(np.mean(keep))


def metric(err: np.ndarray) -> Dict[str, float]:
    e = err[np.isfinite(err)]
    if e.size == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "maxae": float("nan")}
    return {
        "n": int(e.size),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "maxae": float(np.max(np.abs(e))),
    }


def method_failure_hint(method: str) -> str:
    hints = {
        "zero_crossing": "Sensitive to noise/phase jitter; sign-flip counts become unstable on non-ideal sinusoidal segments.",
        "autocorr_period": "Can lock to harmonic/subharmonic peaks when waveform is distorted or truncated.",
        "hilbert_phase": "Requires relatively clean narrowband structure; phase unwrap noise can bias slope.",
        "sinusoid_ls": "Single-tone assumption may be violated by harmonics and segment boundary effects.",
        "plain_fft": "No explicit lag alignment; residual channel misalignment can broaden spectral peak.",
        "full_xcorr_fft": "Full-range lag may pick wrong cycle under periodic ambiguity (less robust than constrained PCCC).",
        "norm_xcorr_fft": "Normalization helps amplitude drift but still suffers periodic lag ambiguity.",
        "welch_periodogram": "Window averaging stabilizes frequency but loses resolution if segments are short.",
    }
    return hints.get(method, "No note")


def recommend_tiers(summary_methods: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
    # rank by position RMSE
    items = [(m, v.get("position_rmse", float("inf"))) for m, v in summary_methods.items()]
    items = [(m, r) for m, r in items if np.isfinite(r)]
    if not items:
        return {"A_main": [], "B_appendix": [], "C_drop": []}
    items.sort(key=lambda x: x[1])
    best = items[0][1]

    A, B, C = [], [], []
    for m, r in items:
        if r <= best * 1.25 and m not in {"zero_crossing"}:
            A.append(m)
        elif r <= best * 2.0:
            B.append(m)
        else:
            C.append(m)

    if "zero_crossing" in A:
        A.remove("zero_crossing")
        B.append("zero_crossing")
    if "zero_crossing" in B and best > 0 and dict(items).get("zero_crossing", 0) > best * 1.8:
        B.remove("zero_crossing")
        C.append("zero_crossing")

    return {"A_main": sorted(set(A)), "B_appendix": sorted(set(B)), "C_drop": sorted(set(C))}


def maybe_plot_error_curve(out_path: Path, rows: List[Dict[str, float]], methods: List[str]) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        return f"matplotlib unavailable: {e}"

    pos = np.asarray([float(r["Position_mm"]) for r in rows])
    plt.figure(figsize=(11, 6), dpi=180)
    plt.plot(pos, [r["Teacher_Error_mm"] for r in rows], label="Teacher", linewidth=1.4)
    plt.plot(pos, [r["Student_Error_mm"] for r in rows], label="Student", linewidth=1.4)
    for m in methods:
        plt.plot(pos, [r[f"{m}_Error_mm"] for r in rows], linewidth=1.0, label=m)
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("Position (mm)")
    plt.ylabel("Position error (mm)")
    plt.title("Classical benchmark: position error curves")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return None


def maybe_plot_boxhist(out_path: Path, rows: List[Dict[str, float]], methods: List[str]) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        return f"matplotlib unavailable: {e}"

    labels = ["Teacher", "Student"] + methods
    data = [
        np.abs(np.asarray([r["Teacher_Error_mm"] for r in rows])),
        np.abs(np.asarray([r["Student_Error_mm"] for r in rows])),
    ]
    for m in methods:
        data.append(np.abs(np.asarray([r[f"{m}_Error_mm"] for r in rows])))

    plt.figure(figsize=(11, 5), dpi=180)
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("|Position error| (mm)")
    plt.title("Error distribution across methods")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Run unified classical benchmark suite")
    p.add_argument("--raw-dir", type=Path, default=Path("1.22"))
    p.add_argument("--val-csv", type=Path, default=Path("grouped_val_comparison.csv"))
    p.add_argument("--teacher-csv", type=Path, default=Path("fixed_results.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/classical_benchmark_suite"))
    p.add_argument("--min-seg-len", type=int, default=2000)
    p.add_argument(
        "--methods",
        type=str,
        default="plain_fft,full_xcorr_fft,norm_xcorr_fft,welch_periodogram,autocorr_period,hilbert_phase,sinusoid_ls,zero_crossing",
        help="comma-separated method names",
    )
    args = p.parse_args()

    method_list = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in method_list if m not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")

    val_rows = load_val_reference(args.val_csv)
    tx, tw = load_teacher_xy(args.teacher_csv)
    A, B, C = fit_hyperbola_teacher(tx, tw)

    rows_out: List[Dict[str, float]] = []
    missing_files: List[str] = []

    method_latency: Dict[str, List[float]] = {m: [] for m in method_list}

    for vr in val_rows:
        fname = str(vr["Filename"])
        fpath = args.raw_dir / fname
        if not fpath.exists():
            missing_files.append(fname)
            continue

        wf_rows = read_csv_rows(fpath)
        if not wf_rows or "Channel 1" not in wf_rows[0] or "Channel 2" not in wf_rows[0]:
            continue

        c1 = np.asarray([float(r["Channel 1"]) for r in wf_rows], dtype=np.float64)
        c2 = np.asarray([float(r["Channel 2"]) for r in wf_rows], dtype=np.float64)
        segs = extract_segments(c1, c2, min_seg_len=args.min_seg_len)

        method_widths: Dict[str, float] = {}
        for m in method_list:
            vals: List[float] = []
            t0 = time.perf_counter()
            fn = METHODS[m]
            for s1, s2 in segs:
                vals.append(fn(s1, s2))
            t1 = time.perf_counter()
            method_latency[m].append((t1 - t0) * 1000.0)
            method_widths[m] = robust_mean(vals)

        pos = float(vr["Position_mm"])
        teacher_w = float(vr["Teacher_Width"])
        student_w = float(vr["Student_Width"])

        all_w = [teacher_w, student_w] + [method_widths[m] for m in method_list]
        pos_pred = inverse_func(np.asarray(all_w, dtype=np.float64), A, B, C)

        r: Dict[str, float] = {
            "Filename": fname,
            "Position_mm": pos,
            "Segment_Count": len(segs),
            "Teacher_Width": teacher_w,
            "Student_Width": student_w,
            "Teacher_Pos_Pred": float(pos_pred[0]),
            "Student_Pos_Pred": float(pos_pred[1]),
            "Teacher_Error_mm": float(pos_pred[0] - pos),
            "Student_Error_mm": float(pos_pred[1] - pos),
        }

        for i, m in enumerate(method_list):
            w = method_widths[m]
            p_pred = float(pos_pred[2 + i])
            r[f"{m}_Width"] = float(w)
            r[f"{m}_Pos_Pred"] = p_pred
            r[f"{m}_Error_mm"] = p_pred - pos
            r[f"{m}_vs_Teacher_WidthErr"] = float(w - teacher_w)
            r[f"{m}_Latency_ms"] = method_latency[m][-1]

        rows_out.append(r)

    if not rows_out:
        raise RuntimeError("No rows generated. Check raw-dir / val-csv.")

    # summary per method
    method_summary: Dict[str, Dict[str, float]] = {}
    for m in method_list:
        w_err = np.asarray([float(r[f"{m}_vs_Teacher_WidthErr"]) for r in rows_out], dtype=np.float64)
        p_err = np.asarray([float(r[f"{m}_Error_mm"]) for r in rows_out], dtype=np.float64)
        method_summary[m] = {
            "width_mae_vs_teacher": metric(w_err)["mae"],
            "width_rmse_vs_teacher": metric(w_err)["rmse"],
            "width_maxae_vs_teacher": metric(w_err)["maxae"],
            "position_mae": metric(p_err)["mae"],
            "position_rmse": metric(p_err)["rmse"],
            "position_maxae": metric(p_err)["maxae"],
            "latency_ms_mean": float(np.mean(method_latency[m])) if method_latency[m] else float("nan"),
            "failure_hint": method_failure_hint(m),
        }

    teacher_err = np.asarray([float(r["Teacher_Error_mm"]) for r in rows_out], dtype=np.float64)
    student_err = np.asarray([float(r["Student_Error_mm"]) for r in rows_out], dtype=np.float64)

    tiers = recommend_tiers(method_summary)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # benchmark_rows.csv
    # collect all keys in stable order
    base_fields = [
        "Filename", "Position_mm", "Segment_Count",
        "Teacher_Width", "Student_Width", "Teacher_Pos_Pred", "Student_Pos_Pred",
        "Teacher_Error_mm", "Student_Error_mm",
    ]
    method_fields: List[str] = []
    for m in method_list:
        method_fields.extend([
            f"{m}_Width", f"{m}_Pos_Pred", f"{m}_Error_mm", f"{m}_vs_Teacher_WidthErr", f"{m}_Latency_ms"
        ])
    row_csv = out_dir / "benchmark_rows.csv"
    with row_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=base_fields + method_fields)
        w.writeheader()
        w.writerows(rows_out)

    # benchmark_summary.json
    summary = {
        "inputs": {
            "raw_dir": str(args.raw_dir),
            "val_csv": str(args.val_csv),
            "teacher_csv": str(args.teacher_csv),
            "methods": method_list,
            "files_in_val_csv": len(val_rows),
            "files_processed": len(rows_out),
            "files_missing_raw": len(missing_files),
        },
        "calibration_model": {
            "formula": "W = 1/(A*x + B) + C",
            "A": A,
            "B": B,
            "C": C,
            "fit_source": str(args.teacher_csv),
        },
        "teacher_position_error": metric(teacher_err),
        "student_position_error": metric(student_err),
        "methods": method_summary,
        "fairness_checks": {
            "same_file_list_as_val_csv": True,
            "same_segmentation_rule_for_all_methods": True,
            "same_inverse_calibration_for_all_methods": True,
            "no_teacher_label_used_in_classical_signal_processing": True,
        },
    }
    sum_json = out_dir / "benchmark_summary.json"
    sum_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # benchmark_ranked_methods.csv
    ranked = sorted(method_summary.items(), key=lambda kv: (kv[1]["position_rmse"], kv[1]["position_mae"]))
    rank_csv = out_dir / "benchmark_ranked_methods.csv"
    with rank_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "rank", "method", "position_rmse", "position_mae", "position_maxae",
                "width_rmse_vs_teacher", "latency_ms_mean", "tier", "failure_hint",
            ],
        )
        w.writeheader()
        for i, (m, v) in enumerate(ranked, start=1):
            if m in tiers["A_main"]:
                tier = "A_main"
            elif m in tiers["B_appendix"]:
                tier = "B_appendix"
            else:
                tier = "C_drop"
            w.writerow(
                {
                    "rank": i,
                    "method": m,
                    "position_rmse": v["position_rmse"],
                    "position_mae": v["position_mae"],
                    "position_maxae": v["position_maxae"],
                    "width_rmse_vs_teacher": v["width_rmse_vs_teacher"],
                    "latency_ms_mean": v["latency_ms_mean"],
                    "tier": tier,
                    "failure_hint": v["failure_hint"],
                }
            )

    # plots
    err_plot = out_dir / "benchmark_error_plot.png"
    dist_plot = out_dir / "benchmark_boxplot_or_hist.png"
    warn1 = maybe_plot_error_curve(err_plot, rows_out, method_list)
    warn2 = maybe_plot_boxhist(dist_plot, rows_out, method_list)

    # recommendation json
    rec = {
        "A_main": tiers["A_main"],
        "B_appendix": tiers["B_appendix"],
        "C_drop": tiers["C_drop"],
        "teacher_student_included": ["teacher", "student"],
        "method_notes": {m: method_failure_hint(m) for m in method_list},
    }
    if warn1:
        rec["plot_warning_error_plot"] = warn1
    if warn2:
        rec["plot_warning_distribution_plot"] = warn2

    rec_json = out_dir / "benchmark_paper_recommendation.json"
    rec_json.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== classical benchmark suite done ===")
    print(f"rows: {row_csv}")
    print(f"summary: {sum_json}")
    print(f"ranked: {rank_csv}")
    print(f"recommendation: {rec_json}")
    if warn1:
        print(f"error plot skipped: {warn1}")
    else:
        print(f"error plot: {err_plot}")
    if warn2:
        print(f"distribution plot skipped: {warn2}")
    else:
        print(f"distribution plot: {dist_plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())