#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import MaxNLocator

# ----------------------------
# Model: W = 1/(A x + B) + C
# ----------------------------
def hyperbola(x, A, B, C):
    return 1.0 / (A * x + B) + C

def r2_score(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

def rmse(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def iqr_inlier_mask(residual, fence=2.5):
    q1 = np.percentile(residual, 25)
    q3 = np.percentile(residual, 75)
    iqr = q3 - q1
    lo = q1 - fence * iqr
    hi = q3 + fence * iqr
    return (residual >= lo) & (residual <= hi)

def robust_curve_fit(x, y, p0):
    """
    Robust fitting via curve_fit(method='trf') -> least_squares backend,
    with soft_l1 loss to reduce the effect of outliers.
    """
    popt, pcov = curve_fit(
        hyperbola, x, y,
        p0=p0,
        method="trf",
        loss="soft_l1",
        f_scale=0.5,
        maxfev=30000
    )
    return popt, pcov

# ----------------------------
# IEEE-like plotting defaults
# ----------------------------
def set_ieee_rcparams(font_size=8):
    mpl.rcParams.update({
        # Fonts (Times-like)
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": font_size,

        # Axes / ticks / lines
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,

        # PDF font embedding (important for IEEE PDF check)
        "pdf.fonttype": 42,   # TrueType
        "ps.fonttype": 42,

        # Saving
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

def make_ieee_fit_figure(
    x, y,
    out_prefix="Fig_HyperbolaFit_singlecol",
    fence=2.5,
    single_col=True,
    color=False,
    show_outliers=False,
    show_params=True
):
    """
    single_col=True  -> 3.5 in width (IEEE single column)
    single_col=False -> 7.16 in width (IEEE two columns)
    """
    set_ieee_rcparams(font_size=8)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Sort by x for nice curve
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Initial guess (close to your calibration scale)
    p0 = [4e-4, -3e-2, 5e-1]

    # 1) robust fit on all points
    popt0, _ = robust_curve_fit(x, y, p0=p0)
    yhat0 = hyperbola(x, *popt0)
    res0 = y - yhat0

    # 2) IQR inlier selection
    in_mask = iqr_inlier_mask(res0, fence=fence)
    x_in, y_in = x[in_mask], y[in_mask]
    x_out, y_out = x[~in_mask], y[~in_mask]

    # 3) final robust fit on inliers
    popt, _ = robust_curve_fit(x_in, y_in, p0=popt0)
    yhat_in = hyperbola(x_in, *popt)
    res_in = y_in - yhat_in

    # Metrics
    R2 = r2_score(y_in, yhat_in)
    RMSE = rmse(y_in, yhat_in)
    n_all = len(x)
    n_in = len(x_in)
    removed_pct = (n_all - n_in) / max(n_all, 1) * 100.0
    A, B, C = popt

    # Figure size
    fig_w = 3.5 if single_col else 7.16
    fig_h = 2.6 if single_col else 2.8  # keep compact for IEEE
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [3.0, 1.2], "hspace": 0.08},
        sharex=True
    )

    # Styles (grayscale-safe by default)
    if color:
        # Color online, but still distinguishable in grayscale via markers/line width
        pts_color = "tab:blue"
        fit_color = "tab:red"
        out_color = "tab:orange"
    else:
        pts_color = "0.35"
        fit_color = "0.0"
        out_color = "0.15"

    # Top: data + fit
    ax1.scatter(
        x_in, y_in,
        s=10, marker="o",
        c=pts_color, alpha=0.75,
        edgecolors="none",
        label="Inliers (used for fit)"
    )
    if show_outliers and len(x_out) > 0:
        ax1.scatter(
            x_out, y_out,
            s=18, marker="x",
            c=out_color, alpha=0.9,
            label="Outliers (IQR)"
        )

    x_dense = np.linspace(np.min(x), np.max(x), 600)
    ax1.plot(
        x_dense, hyperbola(x_dense, *popt),
        color=fit_color, linewidth=1.6,
        label="Hyperbolic fit"
    )

    ax1.set_ylabel("Half-fringe width (pixel)")
    ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # Legend: put at upper right usually safe
    ax1.legend(loc="upper right", frameon=True, framealpha=0.95)

    # Inset text (keep concise for IEEE)
    # Recommend: show model + R2 + RMSE; optionally show A,B,C
    lines = [r"$W=\frac{1}{Ax+B}+C$"]
    if show_params:
        lines.append(f"$A$={A:.4g}, $B$={B:.4g}, $C$={C:.4g}")
    lines.append(f"$R^2$={R2:.6f}, RMSE={RMSE:.4f} px")
    lines.append(f"$N$={n_in}/{n_all} (removed {removed_pct:.2f}%)")
    text = "\n".join(lines)

    ax1.text(
        0.02, 0.06, text,
        transform=ax1.transAxes,
        ha="left", va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.92)
    )

    # Bottom: residuals
    ax2.axhline(0.0, color="0.0", linewidth=1.0)
    ax2.scatter(
        x_in, res_in,
        s=10, marker="o",
        c=pts_color, alpha=0.75,
        edgecolors="none"
    )
    ax2.set_xlabel(r"Displacement $x$ (mm)")
    ax2.set_ylabel("Residual\n(px)")
    ax2.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=6))

    # Save: vector PDF + high-res PNG
    pdf_path = f"{out_prefix}.pdf"
    png_path = f"{out_prefix}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)

    return {
        "pdf": pdf_path,
        "png": png_path,
        "A": A, "B": B, "C": C,
        "R2": R2, "RMSE_px": RMSE,
        "N_all": n_all, "N_inliers": n_in,
        "Removed_percent": removed_pct
    }

def main():
    parser = argparse.ArgumentParser(description="IEEE-style hyperbolic fit figure generator (T-IM ready).")
    parser.add_argument("--csv", type=str, default="fft_results_robust.csv")
    parser.add_argument("--xcol", type=str, default="Position_mm")
    parser.add_argument("--ycol", type=str, default="FFT_Width")
    parser.add_argument("--fence", type=float, default=2.5, help="IQR fence factor")
    parser.add_argument("--twocol", action="store_true", help="generate 2-column width figure (7.16 in)")
    parser.add_argument("--color", action="store_true", help="use color (still grayscale-distinguishable)")
    parser.add_argument("--show_outliers", action="store_true", help="plot IQR outliers")
    parser.add_argument("--hide_params", action="store_true", help="do not print A,B,C in the inset box")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.xcol not in df.columns or args.ycol not in df.columns:
        raise ValueError(f"CSV must contain columns: {args.xcol}, {args.ycol}")

    x = df[args.xcol].to_numpy(dtype=float)
    y = df[args.ycol].to_numpy(dtype=float)

    if args.twocol:
        out = make_ieee_fit_figure(
            x, y,
            out_prefix="Fig_HyperbolaFit_twocol",
            fence=args.fence,
            single_col=False,
            color=args.color,
            show_outliers=args.show_outliers,
            show_params=(not args.hide_params),
        )
    else:
        out = make_ieee_fit_figure(
            x, y,
            out_prefix="Fig_HyperbolaFit_singlecol",
            fence=args.fence,
            single_col=True,
            color=args.color,
            show_outliers=args.show_outliers,
            show_params=(not args.hide_params),
        )

    print("[Saved]")
    for k, v in out.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
