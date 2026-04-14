# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ==============================================================================
# ✅ 你只改这里（直接运行，不需要命令行）
# ==============================================================================
POS_MM = 185.5
DATA_FOLDER = r"D:\桌面\new learning\069date\1.22"
DATASET_TYPE = "NEW"          # "NEW" or "OLD"
SEG_MODE = "middle"           # "first" / "middle" / "last"
MIN_LEN = 50
RESAMPLE_TO = 2048            # 0=不重采样；2048=论文展示统一长度
NORMALIZE = "none"            # none|demean|zscore|minmax
SINGLE_COL = True             # True=单栏3.5in；False=双栏7.16in
EXPORT_EPS = False
OUT_PREFIX = ""               # 留空自动命名

# ==============================================================================
# IEEE-like plotting defaults
# ==============================================================================
def set_ieee_rcparams(font_size=8):
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def pos_to_filename(target_pos_mm: float, dataset_type: str) -> str:
    steps = int(round((target_pos_mm - 180.0) * 10))  # 0.1 mm step
    if dataset_type.upper() == "OLD":
        file_num = -300 + steps
    elif dataset_type.upper() == "NEW":
        file_num = -449 + steps
    else:
        raise ValueError("DATASET_TYPE must be 'NEW' or 'OLD'.")
    return f"{file_num}.csv"

def extract_segments(c1: np.ndarray, c2: np.ndarray, min_len: int = 50):
    c1 = np.asarray(c1)
    c2 = np.asarray(c2)
    starts = np.where((c1 == 0) & (c2 == 0))[0]
    ends = np.where((c1 == 0) & (c2 == 61680))[0]
    segs = []
    if len(starts) == 0 or len(ends) == 0:
        return segs
    for s in starts:
        e_candidates = ends[ends > s]
        if len(e_candidates) == 0:
            continue
        e = int(e_candidates[0])
        if (e - s - 1) >= min_len:
            segs.append((int(s), int(e)))
    return segs

def pick_segment(segments, mode="middle"):
    if not segments:
        return None
    if mode == "first":
        return segments[0]
    if mode == "last":
        return segments[-1]
    return segments[len(segments)//2]

def resample_1d(y: np.ndarray, L: int):
    if L <= 0 or len(y) == L:
        return y
    x_old = np.linspace(0.0, 1.0, num=len(y), endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=L, endpoint=True)
    return np.interp(x_new, x_old, y)

def plot_dual_channel_segment(c1_seg, c2_seg, out_prefix,
                             single_col=True, resample_to=0,
                             normalize="none", export_eps=False):
    set_ieee_rcparams(font_size=8)

    c1 = c1_seg.astype(float)
    c2 = c2_seg.astype(float)

    if resample_to and resample_to > 0:
        c1 = resample_1d(c1, resample_to)
        c2 = resample_1d(c2, resample_to)

    norm = normalize.lower()
    if norm == "demean":
        c1 = c1 - np.mean(c1)
        c2 = c2 - np.mean(c2)
    elif norm == "zscore":
        c1 = (c1 - np.mean(c1)) / (np.std(c1) + 1e-12)
        c2 = (c2 - np.mean(c2)) / (np.std(c2) + 1e-12)
    elif norm == "minmax":
        c1 = (c1 - np.min(c1)) / (np.max(c1) - np.min(c1) + 1e-12)
        c2 = (c2 - np.min(c2)) / (np.max(c2) - np.min(c2) + 1e-12)
    elif norm != "none":
        raise ValueError("NORMALIZE must be none|demean|zscore|minmax")

    fig_w = 3.5 if single_col else 7.16
    fig_h = 2.2 if single_col else 2.4

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(fig_w, fig_h),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08}
    )

    x = np.arange(len(c1))
    ax1.plot(x, c1, color="0.0", linestyle="-", label="Channel 1")
    ax2.plot(x, c2, color="0.0", linestyle="--", label="Channel 2")

    ax1.set_ylabel("Intensity (a.u.)")
    ax2.set_ylabel("Intensity (a.u.)")
    ax2.set_xlabel("Sample index")

    for ax in (ax1, ax2):
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
        ax.legend(loc="upper right", frameon=True, framealpha=0.95)

    pdf_path = f"{out_prefix}.pdf"
    png_path = f"{out_prefix}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=600)
    if export_eps:
        fig.savefig(f"{out_prefix}.eps")

    plt.close(fig)
    return {"pdf": pdf_path, "png": png_path, "N": len(c1)}

def main():
    fname = pos_to_filename(POS_MM, DATASET_TYPE)
    fpath = os.path.join(DATA_FOLDER, fname)
    if not os.path.exists(fpath):
        if os.path.exists(fpath + ".csv"):
            fpath = fpath + ".csv"
            fname = fname + ".csv"
        else:
            raise FileNotFoundError(f"Cannot find file: {fpath}")

    df = pd.read_csv(fpath)
    if "Channel 1" not in df.columns or "Channel 2" not in df.columns:
        raise ValueError("CSV must contain 'Channel 1' and 'Channel 2'.")

    c1 = pd.to_numeric(df["Channel 1"], errors="coerce").fillna(0).to_numpy()
    c2 = pd.to_numeric(df["Channel 2"], errors="coerce").fillna(0).to_numpy()

    segs = extract_segments(c1, c2, min_len=MIN_LEN)
    if not segs:
        raise RuntimeError("No valid segments found. Check markers (0,0) and (0,61680).")

    s, e = pick_segment(segs, mode=SEG_MODE)
    c1_seg = c1[s+1:e]
    c2_seg = c2[s+1:e]

    if OUT_PREFIX.strip():
        out_prefix = OUT_PREFIX.strip()
    else:
        coltag = "singlecol" if SINGLE_COL else "twocol"
        out_prefix = f"Fig_RawWaveformSeg_{POS_MM:.1f}mm_{coltag}"

    info = plot_dual_channel_segment(
        c1_seg, c2_seg,
        out_prefix=out_prefix,
        single_col=SINGLE_COL,
        resample_to=RESAMPLE_TO,
        normalize=NORMALIZE,
        export_eps=EXPORT_EPS
    )

    print("[Saved]")
    print(f"Position(mm): {POS_MM:.1f}, file: {fname}")
    print(f"Segments found: {len(segs)}, plotted: {SEG_MODE} (start={s}, end={e}, len={e-s-1})")
    print(info)

if __name__ == "__main__":
    main()
