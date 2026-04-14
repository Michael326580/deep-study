import io
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# =========================================================
# Configuration
# Put this script, "new learning.zip", and "deep study.zip"
# in the same folder by default.
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

NEW_ZIP = BASE_DIR / "new learning.zip"
DEEP_ZIP = BASE_DIR / "deep study.zip"

OUT_CSV = BASE_DIR / "pccc_vs_nn_optimized_comparison.csv"
OUT_PNG_PIXEL = BASE_DIR / "pccc_vs_nn_optimized_width_pixel.png"
OUT_PNG_MM = BASE_DIR / "pccc_vs_nn_optimized_position_error_mm.png"

DETECTOR_PITCH_UM_PER_PIXEL = 14.0


# =========================================================
# Utility functions
# =========================================================
def numeric_stem_key(name: str):
    stem = Path(name).stem
    try:
        return int(stem)
    except Exception:
        return stem


def width_func(x, A, B, C):
    x = np.asarray(x, dtype=float)
    return 1.0 / (A * x + B) + C


def inverse_func(W, A, B, C):
    W = np.asarray(W, dtype=float)
    denom = W - C
    denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
    return (1.0 / denom - B) / A


def compute_metrics(arr):
    arr = np.asarray(arr, dtype=float)
    mae = float(np.mean(np.abs(arr)))
    rmse = float(np.sqrt(np.mean(arr ** 2)))
    maxae = float(np.max(np.abs(arr)))
    return mae, rmse, maxae


def compute_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return np.nan
    return 1.0 - ss_res / ss_tot


# =========================================================
# ResLightWaveNet definition
# Must match the training code that saved best_model_optimized.pth
# =========================================================
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class ResLightWaveNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(32, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def _make_layer(self, in_c, out_c, blocks, stride):
        layers = [ResBlock(in_c, out_c, stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock(out_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


# =========================================================
# Data loading
# =========================================================
def check_files():
    print("Current working directory:", Path.cwd())
    print("Script directory:", BASE_DIR)
    print("NEW_ZIP:", NEW_ZIP)
    print("DEEP_ZIP:", DEEP_ZIP)

    if not NEW_ZIP.exists():
        raise FileNotFoundError(f"Missing file: {NEW_ZIP}")
    if not DEEP_ZIP.exists():
        raise FileNotFoundError(f"Missing file: {DEEP_ZIP}")


def load_teacher_results():
    with zipfile.ZipFile(NEW_ZIP, "r") as z:
        target = "new learning/fft_results_robust.csv"
        if target not in z.namelist():
            raise FileNotFoundError(f"Cannot find {target} in {NEW_ZIP.name}")

        with z.open(target) as f:
            teacher_df = pd.read_csv(f)

    required_cols = ["Filename", "Position_mm", "FFT_Width"]
    for col in required_cols:
        if col not in teacher_df.columns:
            raise KeyError(f"Teacher CSV missing required column: {col}")

    return teacher_df


def load_distill_label_range():
    with zipfile.ZipFile(DEEP_ZIP, "r") as z:
        target = "deep study/dataset_distill.npz"
        if target not in z.namelist():
            raise FileNotFoundError(f"Cannot find {target} in {DEEP_ZIP.name}")

        with z.open(target) as f:
            npz_bytes = io.BytesIO(f.read())
            distill = np.load(npz_bytes)

            if "y" not in distill:
                raise KeyError("dataset_distill.npz does not contain 'y'")

            y = distill["y"].astype(np.float32)
            y_min = float(y.min())
            y_max = float(y.max())
            y_range = y_max - y_min

    return y_min, y_max, y_range


def extract_model_to_temp():
    tmp_model = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
    tmp_model_path = tmp_model.name
    tmp_model.close()

    with zipfile.ZipFile(DEEP_ZIP, "r") as z:
        target = "deep study/best_model_optimized.pth"
        if target not in z.namelist():
            raise FileNotFoundError(f"Cannot find {target} in {DEEP_ZIP.name}")

        with z.open(target) as src, open(tmp_model_path, "wb") as dst:
            dst.write(src.read())

    return tmp_model_path


def build_segment_samples_from_zip():
    all_samples = []
    all_meta = []

    with zipfile.ZipFile(DEEP_ZIP, "r") as z:
        csv_names = [
            n for n in z.namelist()
            if n.startswith("deep study/1.22/") and n.endswith(".csv")
        ]
        csv_names = sorted(csv_names, key=lambda n: numeric_stem_key(Path(n).name))

        if len(csv_names) == 0:
            raise RuntimeError("No CSV files found under deep study/1.22/")

        for member in csv_names:
            with z.open(member) as f:
                df = pd.read_csv(f)

            if "Channel 1" not in df.columns or "Channel 2" not in df.columns:
                continue

            c1 = df["Channel 1"].to_numpy()
            c2 = df["Channel 2"].to_numpy()

            starts = np.where((c1 == 0) & (c2 == 0))[0]
            ends = np.where((c1 == 0) & (c2 == 61680))[0]

            for s_idx in starts:
                possible_ends = ends[ends > s_idx]
                if len(possible_ends) == 0:
                    continue

                e_idx = int(possible_ends[0])
                seg_len = e_idx - s_idx
                if seg_len <= 2000:
                    continue

                seg_c1 = c1[s_idx + 1:e_idx]
                seg_c2 = c2[s_idx + 1:e_idx]

                sample = np.zeros((2048, 2), dtype=np.float32)
                curr_len = min(len(seg_c1), 2048)
                sample[:curr_len, 0] = seg_c1[:curr_len]
                sample[:curr_len, 1] = seg_c2[:curr_len]

                # Match training-time normalization
                sample /= 16383.0

                all_samples.append(sample)
                all_meta.append(Path(member).name)

    if len(all_samples) == 0:
        raise RuntimeError("No valid segments were parsed from deep study/1.22/")

    return np.stack(all_samples), all_meta


# =========================================================
# Inference
# =========================================================
def infer_student_widths(samples, y_min, y_range, model_path):
    model = ResLightWaveNet()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    X = torch.from_numpy(samples).float().permute(0, 2, 1)

    preds_norm = []
    batch_size = 256

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            pred = model(X[i:i + batch_size]).squeeze(1).cpu().numpy()
            preds_norm.append(pred)

    preds_norm = np.concatenate(preds_norm)
    preds_width = preds_norm * y_range + y_min
    return preds_width


# =========================================================
# Plotting
# =========================================================
def make_pixel_plot(merged, A, B, C):
    teacher_res = merged["Teacher_Width_Residual_pixel"].to_numpy(dtype=float)
    student_res = merged["Student_Width_Residual_pixel"].to_numpy(dtype=float)

    teacher_mae_pixel, teacher_rmse_pixel, teacher_maxae_pixel = compute_metrics(teacher_res)
    student_mae_pixel, student_rmse_pixel, student_maxae_pixel = compute_metrics(student_res)

    plt.figure(figsize=(10, 7), dpi=200)

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(
        merged["Position_mm"],
        merged["FFT_Width"],
        linewidth=1.2,
        label="Traditional PCCC width"
    )
    ax1.plot(
        merged["Position_mm"],
        merged["Student_Width"],
        linewidth=1.2,
        label="Neural network width"
    )

    x_fit = np.linspace(merged["Position_mm"].min(), merged["Position_mm"].max(), 600)
    ax1.plot(
        x_fit,
        width_func(x_fit, A, B, C),
        "--",
        linewidth=1.0,
        label="PCCC hyperbolic fit"
    )
    ax1.set_xlabel("Position (mm)")
    ax1.set_ylabel("Fringe width (pixel)")
    ax1.set_title("PCCC vs optimized neural-network width reconstruction")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(
        merged["Position_mm"],
        merged["Teacher_Width_Residual_pixel"],
        linewidth=1.0,
        label=f"PCCC residual  MAE={teacher_mae_pixel:.4f} pixel"
    )
    ax2.plot(
        merged["Position_mm"],
        merged["Student_Width_Residual_pixel"],
        linewidth=1.0,
        label=f"NN residual  MAE={student_mae_pixel:.4f} pixel"
    )
    ax2.axhline(0.0, linestyle="--", linewidth=0.8)
    ax2.set_xlabel("True position (mm)")
    ax2.set_ylabel("Width residual (pixel)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUT_PNG_PIXEL, bbox_inches="tight")
    plt.close()


def make_mm_position_error_plot(merged):
    teacher_err = merged["Teacher_Error_mm"].to_numpy(dtype=float)
    student_err = merged["Student_Error_mm"].to_numpy(dtype=float)

    teacher_mae_mm, teacher_rmse_mm, teacher_maxae_mm = compute_metrics(teacher_err)
    student_mae_mm, student_rmse_mm, student_maxae_mm = compute_metrics(student_err)

    plt.figure(figsize=(10, 5), dpi=200)

    plt.plot(
        merged["Position_mm"],
        merged["Teacher_Error_mm"],
        linewidth=1.0,
        label=f"PCCC error  MAE={teacher_mae_mm:.4f} mm"
    )
    plt.plot(
        merged["Position_mm"],
        merged["Student_Error_mm"],
        linewidth=1.0,
        label=f"NN error  MAE={student_mae_mm:.4f} mm"
    )
    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.xlabel("True position (mm)")
    plt.ylabel("Position error (mm)")
    plt.title("Position reconstruction error (reference only)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(OUT_PNG_MM, bbox_inches="tight")
    plt.close()


# =========================================================
# Main
# =========================================================
def main():
    check_files()

    print("\n[1/8] Loading traditional PCCC results...")
    teacher_df = load_teacher_results()
    print(f"Teacher rows: {len(teacher_df)}")

    print("\n[2/8] Loading distillation label range...")
    y_min, y_max, y_range = load_distill_label_range()
    print(f"Label range: min={y_min:.6f}, max={y_max:.6f}")

    print("\n[3/8] Extracting optimized model checkpoint...")
    model_path = extract_model_to_temp()
    print(f"Temp model path: {model_path}")

    print("\n[4/8] Building waveform segments from deep study/1.22/ ...")
    samples, meta = build_segment_samples_from_zip()
    print(f"Valid segments: {len(samples)}")

    print("\n[5/8] Running optimized neural-network inference...")
    preds_width = infer_student_widths(samples, y_min, y_range, model_path)

    student_seg_df = pd.DataFrame({
        "Filename": meta,
        "Student_Width": preds_width
    })

    student_file_df = (
        student_seg_df.groupby("Filename", as_index=False)
        .agg(
            Student_Width=("Student_Width", "mean"),
            Segment_Count=("Student_Width", "size")
        )
    )
    print(f"Files with NN predictions: {len(student_file_df)}")

    print("\n[6/8] Merging teacher and student results...")
    merged = teacher_df.merge(student_file_df, on="Filename", how="inner").copy()
    merged = merged.sort_values("Position_mm").reset_index(drop=True)

    if len(merged) == 0:
        raise RuntimeError("Merged result is empty. Please check Filename consistency.")

    print(f"Merged rows: {len(merged)}")

    x_true = merged["Position_mm"].to_numpy(dtype=float)
    w_teacher = merged["FFT_Width"].to_numpy(dtype=float)
    w_student = merged["Student_Width"].to_numpy(dtype=float)

    print("\n[7/8] Fitting hyperbolic width-position mapping using PCCC widths...")
    popt, _ = curve_fit(
        width_func,
        x_true,
        w_teacher,
        p0=[0.001, 0.01, 0.0],
        method="trf",
        loss="soft_l1",
        maxfev=10000
    )
    A, B, C = popt
    print(f"Fitted parameters: A={A:.10f}, B={B:.10f}, C={C:.10f}")

    merged["Fit_Width"] = width_func(merged["Position_mm"].to_numpy(dtype=float), A, B, C)

    # Pixel-domain residuals
    merged["Teacher_Width_Residual_pixel"] = merged["FFT_Width"] - merged["Fit_Width"]
    merged["Student_Width_Residual_pixel"] = merged["Student_Width"] - merged["Fit_Width"]

    # Position-domain inverse mapping for reference
    merged["Teacher_Pos_Pred"] = inverse_func(w_teacher, A, B, C)
    merged["Student_Pos_Pred"] = inverse_func(w_student, A, B, C)

    merged["Teacher_Error_mm"] = merged["Teacher_Pos_Pred"] - merged["Position_mm"]
    merged["Student_Error_mm"] = merged["Student_Pos_Pred"] - merged["Position_mm"]

    merged["Teacher_AbsErr_mm"] = np.abs(merged["Teacher_Error_mm"])
    merged["Student_AbsErr_mm"] = np.abs(merged["Student_Error_mm"])

    # 3-sigma reference masks
    for prefix in ["Teacher", "Student"]:
        err = merged[f"{prefix}_Error_mm"].to_numpy(dtype=float)
        mu = err.mean()
        sigma = err.std(ddof=0)
        merged[f"{prefix}_Keep_3Sigma"] = np.abs(err - mu) <= 3 * sigma

    teacher_width_mae, teacher_width_rmse, teacher_width_maxae = compute_metrics(
        merged["Teacher_Width_Residual_pixel"]
    )
    student_width_mae, student_width_rmse, student_width_maxae = compute_metrics(
        merged["Student_Width_Residual_pixel"]
    )

    teacher_pos_mae, teacher_pos_rmse, teacher_pos_maxae = compute_metrics(
        merged["Teacher_Error_mm"]
    )
    student_pos_mae, student_pos_rmse, student_pos_maxae = compute_metrics(
        merged["Student_Error_mm"]
    )

    avg_width_diff = float(np.mean(np.abs(merged["Student_Width"] - merged["FFT_Width"])))
    r2_fit = compute_r2(merged["FFT_Width"], merged["Fit_Width"])
    corr_teacher_student = float(np.corrcoef(merged["FFT_Width"], merged["Student_Width"])[0, 1])

    print("\n[8/8] Saving CSV and figures...")
    merged.to_csv(OUT_CSV, index=False)

    make_pixel_plot(merged, A, B, C)
    make_mm_position_error_plot(merged)

    print("\n==================== Summary ====================")
    print("Main output CSV:")
    print(f"  {OUT_CSV}")

    print("\nMain figure (both y-axes in pixel):")
    print(f"  {OUT_PNG_PIXEL}")

    print("\nReference figure (position error in mm):")
    print(f"  {OUT_PNG_MM}")

    print("\n----- Width-domain metrics (pixel) -----")
    print(f"PCCC residual   -> MAE={teacher_width_mae:.6f} pixel, RMSE={teacher_width_rmse:.6f} pixel, MaxAE={teacher_width_maxae:.6f} pixel")
    print(f"NN residual     -> MAE={student_width_mae:.6f} pixel, RMSE={student_width_rmse:.6f} pixel, MaxAE={student_width_maxae:.6f} pixel")

    print("\n----- Position-domain metrics (mm, reference only) -----")
    print(f"PCCC position   -> MAE={teacher_pos_mae:.6f} mm, RMSE={teacher_pos_rmse:.6f} mm, MaxAE={teacher_pos_maxae:.6f} mm")
    print(f"NN position     -> MAE={student_pos_mae:.6f} mm, RMSE={student_pos_rmse:.6f} mm, MaxAE={student_pos_maxae:.6f} mm")

    print("\n----- Additional checks -----")
    print(f"Average |NN width - PCCC width| = {avg_width_diff:.6f} pixel")
    print(f"R^2 of PCCC hyperbolic fit      = {r2_fit:.8f}")
    print(f"Corr(PCCC width, NN width)      = {corr_teacher_student:.8f}")
    print(f"Detector pitch                  = {DETECTOR_PITCH_UM_PER_PIXEL:.3f} um/pixel")
    print("Note: detector pitch is only a sensor sampling property, not a direct displacement conversion factor.")

    print("\nDone.")


if __name__ == "__main__":
    main()