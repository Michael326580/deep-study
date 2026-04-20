# =========================================================
# [脚本说明]
# 用途：IEEE TIM 原始波形图脚本（无标题、图例避挡版，命令行可调）。
# 输入：1. 解析参数并设置 IEEE 风格 rcParams。
# 输出：2. 位置映射到文件名，鲁棒读取 CSV。
# =========================================================

# =========================================================
# [????]
# ???IEEE TIM ???????????????????
# ????????????????
# ???Fig_RawWaveform_*.pdf/.png?
# =========================================================

# TIM_waveform_plot.py
# Paper-ready waveform figure for IEEE TIM (no title, no corner annotation, extra y-headroom for legend)

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# =========================
# Defaults (edit if needed)
# =========================
DEFAULT_POS = 192.4
DEFAULT_FOLDER = r"D:\桌面\new learning\069date\1.22"
DEFAULT_DATASET = "NEW"   # "NEW": 180.0mm -> -449,  "OLD": 180.0mm -> -300

DEFAULT_PAD = 2
DEFAULT_INTERP = True

# Increase this to create more top blank space for legends
DEFAULT_HEADROOM = 1.28   # <-- “再提高一点”就调大，比如 1.30 / 1.35


# =========================
# IEEE-like plotting style
# =========================
def set_ieee_rcparams():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


# =========================
# Helpers
# =========================
def pos_to_filename(target_pos: float, dataset_type: str) -> str:
    steps = int(round((target_pos - 180.0) * 10))  # 0.1 mm step
    ds = dataset_type.upper()
    if ds == "OLD":
        file_num = -300 + steps
    elif ds == "NEW":
        file_num = -449 + steps
    else:
        raise ValueError("dataset_type must be 'NEW' or 'OLD'")
    return f"{file_num}.csv"


def robust_read_csv(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.read_csv(file_path, engine="python")


def remove_markers_for_plot(c1, c2, pad=2, do_interpolate=True):
    """
    Plot-only cleaning:
      marker points: (c1==0 & c2==61680) OR (c1==0 & c2==0)
      remove +/- pad samples around markers
      fill by linear interpolation if do_interpolate=True
    """
    c1 = pd.Series(pd.to_numeric(c1, errors="coerce").astype(float))
    c2 = pd.Series(pd.to_numeric(c2, errors="coerce").astype(float))

    marker_mask = ((c1 == 0) & (c2 == 61680)) | ((c1 == 0) & (c2 == 0))

    if pad > 0:
        idx = np.where(marker_mask.to_numpy())[0]
        dil = np.zeros_like(marker_mask.to_numpy(), dtype=bool)
        n = len(dil)
        for i in idx:
            lo = max(0, i - pad)
            hi = min(n, i + pad + 1)
            dil[lo:hi] = True
        marker_mask = pd.Series(dil)

    c1_clean = c1.mask(marker_mask, np.nan)
    c2_clean = c2.mask(marker_mask, np.nan)

    if do_interpolate:
        c1_clean = c1_clean.interpolate(method="linear", limit_direction="both")
        c2_clean = c2_clean.interpolate(method="linear", limit_direction="both")

    removed = int(np.sum(marker_mask.to_numpy()))
    return c1_clean.to_numpy(), c2_clean.to_numpy(), removed


def nice_ylim(y, headroom=1.25):
    """Set y-limits with additional top space for legend (headroom > 1)."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0, 1.0
    ymin = float(np.min(y))
    ymax = float(np.max(y))
    # keep a small bottom margin + headroom on top
    span = max(ymax - ymin, 1.0)
    ylo = ymin - 0.06 * span
    yhi = ymax * headroom if ymax > 0 else ymax / headroom
    # if headroom doesn't increase enough (rare), enforce a minimum
    yhi = max(yhi, ymax + 0.10 * span)
    return ylo, yhi


# =========================
# Main plot function
# =========================
def plot_waveform_figure(pos, folder, dataset, pad, do_interpolate, headroom,
                         twocol=False, out_prefix=None, make_eps=False, show=False):
    set_ieee_rcparams()

    filename = pos_to_filename(pos, dataset)
    file_path = os.path.join(folder, filename)

    if not os.path.exists(file_path):
        # tolerate accidental double ".csv"
        if os.path.exists(file_path + ".csv"):
            file_path += ".csv"
            filename += ".csv"
        else:
            raise FileNotFoundError(f"Cannot find file: {file_path}")

    df = robust_read_csv(file_path)
    if ("Channel 1" not in df.columns) or ("Channel 2" not in df.columns):
        raise ValueError("CSV must contain columns: 'Channel 1', 'Channel 2'")

    c1_raw = pd.to_numeric(df["Channel 1"], errors="coerce")
    c2_raw = pd.to_numeric(df["Channel 2"], errors="coerce")

    c1_plot, c2_plot, removed = remove_markers_for_plot(
        c1_raw, c2_raw, pad=pad, do_interpolate=do_interpolate
    )

    # Figure size: IEEE single-column ~3.5 in; two-column ~7.16 in
    if twocol:
        fig_w, fig_h = 7.16, 3.6
    else:
        fig_w, fig_h = 3.5, 3.6

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(fig_w, fig_h), sharex=True,
        gridspec_kw={"hspace": 0.08}
    )

    x = np.arange(len(c1_plot), dtype=float)

    # Top subplot (Channel 1)
    ax1.plot(x, c1_plot, color="#b30000", label="Channel 1")
    ax1.set_ylabel("Intensity (a.u.)")
    ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)

    ylo1, yhi1 = nice_ylim(c1_plot, headroom=headroom)
    ax1.set_ylim(ylo1, yhi1)

    # Legend (will not cover due to raised ylim)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.95,
               borderaxespad=0.4, handlelength=2.4)

    # Bottom subplot (Channel 2)
    ax2.plot(x, c2_plot, color="#0000b3", label="Channel 2")
    ax2.set_ylabel("Intensity (a.u.)")
    ax2.set_xlabel("Sample Index")
    ax2.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)

    ylo2, yhi2 = nice_ylim(c2_plot, headroom=headroom)
    ax2.set_ylim(ylo2, yhi2)

    ax2.legend(loc="upper right", frameon=True, framealpha=0.95,
               borderaxespad=0.4, handlelength=2.4)

    # No title / no corner annotation (paper figure: caption in LaTeX)
    # Tight layout
    fig.tight_layout(pad=0.15)

    if out_prefix is None:
        out_prefix = f"Fig_RawWaveform_{pos:.1f}mm_{'twocol' if twocol else 'singlecol'}"

    pdf_path = f"{out_prefix}.pdf"
    png_path = f"{out_prefix}.png"
    fig.savefig(pdf_path)              # vector
    fig.savefig(png_path, dpi=600)     # high-res raster for quick view
    if make_eps:
        fig.savefig(f"{out_prefix}.eps")

    if show:
        plt.show()
    plt.close(fig)

    return {"pdf": pdf_path, "png": png_path, "file": filename, "removed": removed}


# =========================
# Entry
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Generate IEEE TIM-ready raw waveform figure (2 subplots)."
    )
    parser.add_argument("--pos", type=float, default=DEFAULT_POS, help="Target displacement (mm)")
    parser.add_argument("--folder", type=str, default=DEFAULT_FOLDER, help="Folder containing CSV files")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, choices=["NEW", "OLD"],
                        help="Filename mapping rule")
    parser.add_argument("--pad", type=int, default=DEFAULT_PAD, help="Marker dilation samples on each side")
    parser.add_argument("--no_interp", action="store_true", help="Do not interpolate removed markers")
    parser.add_argument("--headroom", type=float, default=DEFAULT_HEADROOM,
                        help="Y-axis headroom factor (>1). Increase to avoid legend overlap.")
    parser.add_argument("--twocol", action="store_true", help="Two-column width figure (7.16 in)")
    parser.add_argument("--eps", action="store_true", help="Also export EPS")
    parser.add_argument("--out", type=str, default=None, help="Output prefix (without extension)")
    parser.add_argument("--show", action="store_true", help="Display figure window")

    args = parser.parse_args()

    info = plot_waveform_figure(
        pos=args.pos,
        folder=args.folder,
        dataset=args.dataset,
        pad=args.pad,
        do_interpolate=(not args.no_interp),
        headroom=args.headroom,
        twocol=args.twocol,
        out_prefix=args.out,
        make_eps=args.eps,
        show=args.show
    )

    print("[Saved]")
    print({k: info[k] for k in ["pdf", "png", "file"]})
    print(f"Removed/filled samples (plot-only): {info['removed']} (pad={args.pad}, interpolate={not args.no_interp})")


if __name__ == "__main__":
    main()
