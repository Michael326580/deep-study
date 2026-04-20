# =========================================================
# [脚本说明]
# 用途：工程化训练主干脚本（支持本地文件和 zip，按文件级切分训练）。
# 输入：1. 自动发现教师 CSV 与原始数据来源（文件夹或 zip）。
# 输出：2. `build_grouped_dataset` 从原始 CSV 切段并构建 `X/y`，同时记录每段文件名与位置。
# =========================================================

# =========================================================
# [????]
# ????????????????????????
# ??????? + ???????? zip??
# ???best_model_grouped_resnet.pth ????????
# =========================================================

import io
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# =========================================================
# Basic config
# This script supports either:
# 1) extracted files in the same folder
# 2) or the two zip files in the same folder
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

SEED = 42
VAL_RATIO = 0.20

BATCH_SIZE = 32
LEARNING_RATE = 0.002
EPOCHS = 120
WEIGHT_DECAY = 1e-4
NOISE_STD = 0.005

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = BASE_DIR / "best_model_grouped_resnet.pth"
TRAIN_CURVE_PATH = BASE_DIR / "training_curve_grouped.png"
VAL_CSV_PATH = BASE_DIR / "grouped_val_comparison.csv"
VAL_PNG_PIXEL = BASE_DIR / "grouped_val_width_pixel.png"
VAL_PNG_MM = BASE_DIR / "grouped_val_position_error_mm.png"
META_CSV_PATH = BASE_DIR / "grouped_dataset_metadata.csv"


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Utility
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
# File locating
# =========================================================
def find_teacher_csv_path():
    candidates = [
        BASE_DIR / "fft_results_robust.csv",
        BASE_DIR / "new learning" / "fft_results_robust.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def find_raw_folder_path():
    candidates = [
        BASE_DIR / "1.22",
        BASE_DIR / "deep study" / "1.22",
        BASE_DIR / "new learning" / "069date" / "1.22",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def find_new_learning_zip():
    p = BASE_DIR / "new learning.zip"
    return p if p.exists() else None


def find_deep_study_zip():
    p = BASE_DIR / "deep study.zip"
    return p if p.exists() else None


# =========================================================
# Load teacher results
# =========================================================
def load_teacher_results():
    teacher_csv = find_teacher_csv_path()
    if teacher_csv is not None:
        print(f"Using local teacher CSV: {teacher_csv}")
        teacher_df = pd.read_csv(teacher_csv)
        return teacher_df

    new_zip = find_new_learning_zip()
    if new_zip is not None:
        print(f"Using teacher CSV from zip: {new_zip}")
        with zipfile.ZipFile(new_zip, "r") as z:
            target = "new learning/fft_results_robust.csv"
            if target not in z.namelist():
                raise FileNotFoundError(f"Cannot find {target} inside {new_zip}")
            with z.open(target) as f:
                teacher_df = pd.read_csv(f)
        return teacher_df

    raise FileNotFoundError(
        "Cannot find fft_results_robust.csv.\n"
        "Place extracted fft_results_robust.csv next to this script,\n"
        "or place new learning.zip next to this script."
    )


# =========================================================
# Iterate raw waveform csvs
# =========================================================
def iter_raw_csvs():
    raw_folder = find_raw_folder_path()
    if raw_folder is not None:
        print(f"Using local raw folder: {raw_folder}")
        files = sorted(raw_folder.glob("*.csv"), key=lambda p: numeric_stem_key(p.name))
        for path in files:
            try:
                df = pd.read_csv(path)
            except Exception:
                df = pd.read_csv(path, engine="python")
            yield path.name, df
        return

    deep_zip = find_deep_study_zip()
    if deep_zip is not None:
        print(f"Using raw csvs from zip: {deep_zip}")
        with zipfile.ZipFile(deep_zip, "r") as z:
            names = [
                n for n in z.namelist()
                if n.startswith("deep study/1.22/") and n.endswith(".csv")
            ]
            names = sorted(names, key=lambda n: numeric_stem_key(Path(n).name))
            for member in names:
                with z.open(member) as f:
                    try:
                        df = pd.read_csv(f)
                    except Exception:
                        f.seek(0)
                        df = pd.read_csv(f, engine="python")
                yield Path(member).name, df
        return

    raise FileNotFoundError(
        "Cannot find raw waveform csvs.\n"
        "Place extracted 1.22 folder next to this script,\n"
        "or place deep study.zip next to this script."
    )


# =========================================================
# Build grouped dataset from raw waveforms + teacher labels
# =========================================================
def build_grouped_dataset():
    teacher_df = load_teacher_results()

    required_cols = ["Filename", "Position_mm", "FFT_Width"]
    for col in required_cols:
        if col not in teacher_df.columns:
            raise KeyError(f"Teacher CSV missing required column: {col}")

    label_map = dict(zip(teacher_df["Filename"], teacher_df["FFT_Width"]))
    pos_map = dict(zip(teacher_df["Filename"], teacher_df["Position_mm"]))

    X_list = []
    y_list = []
    filename_list = []
    position_list = []

    total_files = 0
    used_files = 0

    for fname, df in iter_raw_csvs():
        total_files += 1

        if fname not in label_map:
            continue

        if "Channel 1" not in df.columns or "Channel 2" not in df.columns:
            continue

        target_width = float(label_map[fname])
        target_pos = float(pos_map[fname])

        c1 = df["Channel 1"].to_numpy()
        c2 = df["Channel 2"].to_numpy()

        starts = np.where((c1 == 0) & (c2 == 0))[0]
        ends = np.where((c1 == 0) & (c2 == 61680))[0]

        valid_segments_this_file = 0

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
            sample /= 16383.0

            X_list.append(sample)
            y_list.append(target_width)
            filename_list.append(fname)
            position_list.append(target_pos)

            valid_segments_this_file += 1

        if valid_segments_this_file > 0:
            used_files += 1

        if total_files % 100 == 0:
            print(f"Processed raw files: {total_files}")

    if len(X_list) == 0:
        raise RuntimeError("No valid samples were built from raw data.")

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    filenames = np.asarray(filename_list)
    positions = np.asarray(position_list, dtype=np.float32)

    meta_df = pd.DataFrame({
        "Filename": filenames,
        "Position_mm": positions,
        "Teacher_Width": y
    })
    meta_df.to_csv(META_CSV_PATH, index=False)

    print("\nDataset rebuilt from raw waveforms.")
    print(f"Raw files scanned: {total_files}")
    print(f"Files with valid segments: {used_files}")
    print(f"Total segment samples: {len(X)}")
    print(f"Unique files: {len(np.unique(filenames))}")

    return teacher_df, X, y, filenames, positions


# =========================================================
# Group split by filename
# =========================================================
def split_by_filename(filenames, val_ratio=0.2, seed=42):
    unique_files = np.array(sorted(np.unique(filenames), key=numeric_stem_key))
    rng = np.random.default_rng(seed)
    shuffled = unique_files.copy()
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(unique_files) * val_ratio)))
    val_files = set(shuffled[:n_val].tolist())
    train_files = set(shuffled[n_val:].tolist())

    train_idx = np.where(np.isin(filenames, list(train_files)))[0]
    val_idx = np.where(np.isin(filenames, list(val_files)))[0]

    return train_idx, val_idx, train_files, val_files


# =========================================================
# Dataset
# =========================================================
class WaveformDataset(Dataset):
    def __init__(self, X, y, filenames, y_min, y_range, is_train=False, noise_std=0.005):
        self.X = torch.from_numpy(X).float().permute(0, 2, 1)
        self.filenames = list(filenames)
        self.raw_y = y.astype(np.float32)

        self.y_min = float(y_min)
        self.y_range = float(y_range)
        self.y_norm = ((self.raw_y - self.y_min) / self.y_range).astype(np.float32)
        self.y = torch.from_numpy(self.y_norm).float().unsqueeze(1)

        self.is_train = is_train
        self.noise_std = noise_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        fname = self.filenames[idx]

        if self.is_train:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        return x, y, fname

    def denormalize(self, y_pred_norm):
        return y_pred_norm * self.y_range + self.y_min


# =========================================================
# Model
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
# Validation helper
# =========================================================
def evaluate_model(model, loader, denormalize_fn, criterion):
    model.eval()

    total_loss = 0.0
    total_n = 0

    pred_list = []
    true_list = []
    fname_list = []

    with torch.no_grad():
        for x, y, fnames in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)
            loss = criterion(pred, y)

            batch_n = x.size(0)
            total_loss += loss.item() * batch_n
            total_n += batch_n

            pred_denorm = denormalize_fn(pred.detach().cpu().numpy()).reshape(-1)
            true_denorm = denormalize_fn(y.detach().cpu().numpy()).reshape(-1)

            pred_list.extend(pred_denorm.tolist())
            true_list.extend(true_denorm.tolist())
            fname_list.extend(list(fnames))

    val_loss = total_loss / total_n

    seg_abs = np.abs(np.asarray(pred_list) - np.asarray(true_list))
    seg_mae = float(np.mean(seg_abs))

    seg_df = pd.DataFrame({
        "Filename": fname_list,
        "Pred_Width": pred_list,
        "True_Width": true_list
    })
    file_df = (
        seg_df.groupby("Filename", as_index=False)
        .agg(
            Pred_Width=("Pred_Width", "mean"),
            True_Width=("True_Width", "first"),
            Segment_Count=("Pred_Width", "size")
        )
    )
    file_mae = float(np.mean(np.abs(file_df["Pred_Width"] - file_df["True_Width"])))

    return val_loss, seg_mae, file_mae, file_df


# =========================================================
# Final validation report
# =========================================================
def make_val_report(model, val_loader, denormalize_fn, teacher_df, val_files):
    model.eval()

    pred_list = []
    true_list = []
    fname_list = []

    with torch.no_grad():
        for x, y, fnames in val_loader:
            x = x.to(DEVICE)
            pred = model(x)

            pred_denorm = denormalize_fn(pred.detach().cpu().numpy()).reshape(-1)
            true_denorm = denormalize_fn(y.numpy()).reshape(-1)

            pred_list.extend(pred_denorm.tolist())
            true_list.extend(true_denorm.tolist())
            fname_list.extend(list(fnames))

    seg_df = pd.DataFrame({
        "Filename": fname_list,
        "Pred_Width": pred_list,
        "True_Width": true_list
    })

    pred_file_df = (
        seg_df.groupby("Filename", as_index=False)
        .agg(
            Student_Width=("Pred_Width", "mean"),
            Teacher_Label_Width=("True_Width", "first"),
            Segment_Count=("Pred_Width", "size")
        )
    )

    val_teacher_df = teacher_df[teacher_df["Filename"].isin(list(val_files))].copy()
    val_teacher_df = val_teacher_df.sort_values("Position_mm").reset_index(drop=True)

    merged = val_teacher_df.merge(pred_file_df, on="Filename", how="inner")
    merged = merged.sort_values("Position_mm").reset_index(drop=True)

    # Fit hyperbolic calibration using all teacher results
    x_all = teacher_df["Position_mm"].to_numpy(dtype=float)
    w_all = teacher_df["FFT_Width"].to_numpy(dtype=float)

    popt, _ = curve_fit(
        width_func,
        x_all,
        w_all,
        p0=[0.001, 0.01, 0.0],
        method="trf",
        loss="soft_l1",
        maxfev=10000
    )
    A, B, C = popt

    merged["Fit_Width"] = width_func(merged["Position_mm"].to_numpy(dtype=float), A, B, C)

    merged["Teacher_Width_Residual_pixel"] = merged["FFT_Width"] - merged["Fit_Width"]
    merged["Student_Width_Residual_pixel"] = merged["Student_Width"] - merged["Fit_Width"]

    merged["Teacher_Pos_Pred"] = inverse_func(merged["FFT_Width"], A, B, C)
    merged["Student_Pos_Pred"] = inverse_func(merged["Student_Width"], A, B, C)

    merged["Teacher_Error_mm"] = merged["Teacher_Pos_Pred"] - merged["Position_mm"]
    merged["Student_Error_mm"] = merged["Student_Pos_Pred"] - merged["Position_mm"]

    merged["Teacher_AbsErr_mm"] = np.abs(merged["Teacher_Error_mm"])
    merged["Student_AbsErr_mm"] = np.abs(merged["Student_Error_mm"])

    # 3-sigma masks
    for prefix in ["Teacher", "Student"]:
        err = merged[f"{prefix}_Error_mm"].to_numpy(dtype=float)
        mu = err.mean()
        sigma = err.std(ddof=0)
        merged[f"{prefix}_Keep_3Sigma"] = np.abs(err - mu) <= 3 * sigma

    merged.to_csv(VAL_CSV_PATH, index=False)

    # Metrics
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

    student_mask = merged["Student_Keep_3Sigma"].to_numpy(dtype=bool)
    student_pos_mae_3s, student_pos_rmse_3s, student_pos_maxae_3s = compute_metrics(
        merged.loc[student_mask, "Student_Error_mm"]
    )

    avg_width_diff = float(np.mean(np.abs(merged["Student_Width"] - merged["FFT_Width"])))
    corr_teacher_student = float(np.corrcoef(merged["FFT_Width"], merged["Student_Width"])[0, 1])
    r2_fit = compute_r2(teacher_df["FFT_Width"], width_func(teacher_df["Position_mm"], A, B, C))

    # Pixel plot
    plt.figure(figsize=(10, 7), dpi=200)

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(merged["Position_mm"], merged["FFT_Width"], linewidth=1.2, label="Traditional PCCC width")
    ax1.plot(merged["Position_mm"], merged["Student_Width"], linewidth=1.2, label="Grouped NN width")
    x_fit = np.linspace(merged["Position_mm"].min(), merged["Position_mm"].max(), 600)
    ax1.plot(x_fit, width_func(x_fit, A, B, C), "--", linewidth=1.0, label="PCCC hyperbolic fit")
    ax1.set_xlabel("Position (mm)")
    ax1.set_ylabel("Fringe width (pixel)")
    ax1.set_title("Validation files: PCCC vs grouped neural-network width reconstruction")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(
        merged["Position_mm"],
        merged["Teacher_Width_Residual_pixel"],
        linewidth=1.0,
        label=f"PCCC residual  MAE={teacher_width_mae:.4f} pixel"
    )
    ax2.plot(
        merged["Position_mm"],
        merged["Student_Width_Residual_pixel"],
        linewidth=1.0,
        label=f"Grouped NN residual  MAE={student_width_mae:.4f} pixel"
    )
    ax2.axhline(0.0, linestyle="--", linewidth=0.8)
    ax2.set_xlabel("True position (mm)")
    ax2.set_ylabel("Width residual (pixel)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(VAL_PNG_PIXEL, bbox_inches="tight")
    plt.close()

    # mm plot
    plt.figure(figsize=(10, 5), dpi=200)
    plt.plot(
        merged["Position_mm"],
        merged["Teacher_Error_mm"],
        linewidth=1.0,
        label=f"PCCC error  MAE={teacher_pos_mae:.4f} mm"
    )
    plt.plot(
        merged["Position_mm"],
        merged["Student_Error_mm"],
        linewidth=1.0,
        label=f"Grouped NN error  MAE={student_pos_mae:.4f} mm"
    )
    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.xlabel("True position (mm)")
    plt.ylabel("Position error (mm)")
    plt.title("Validation files: position reconstruction error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(VAL_PNG_MM, bbox_inches="tight")
    plt.close()

    print("\n================ Validation report ================")
    print(f"Saved CSV: {VAL_CSV_PATH}")
    print(f"Saved pixel plot: {VAL_PNG_PIXEL}")
    print(f"Saved mm plot: {VAL_PNG_MM}")

    print("\n----- Width-domain metrics on hold-out files (pixel) -----")
    print(f"PCCC residual   -> MAE={teacher_width_mae:.6f}, RMSE={teacher_width_rmse:.6f}, MaxAE={teacher_width_maxae:.6f}")
    print(f"Grouped NN      -> MAE={student_width_mae:.6f}, RMSE={student_width_rmse:.6f}, MaxAE={student_width_maxae:.6f}")

    print("\n----- Position-domain metrics on hold-out files (mm) -----")
    print(f"PCCC position   -> MAE={teacher_pos_mae:.6f}, RMSE={teacher_pos_rmse:.6f}, MaxAE={teacher_pos_maxae:.6f}")
    print(f"Grouped NN      -> MAE={student_pos_mae:.6f}, RMSE={student_pos_rmse:.6f}, MaxAE={student_pos_maxae:.6f}")
    print(f"Grouped NN 3σ   -> MAE={student_pos_mae_3s:.6f}, RMSE={student_pos_rmse_3s:.6f}, MaxAE={student_pos_maxae_3s:.6f}")

    print("\n----- Additional checks -----")
    print(f"Average |NN width - PCCC width| = {avg_width_diff:.6f} pixel")
    print(f"Corr(PCCC width, NN width)      = {corr_teacher_student:.8f}")
    print(f"R^2 of hyperbolic fit           = {r2_fit:.8f}")


# =========================================================
# Training
# =========================================================
def train():
    set_seed(SEED)
    print(f"Device: {DEVICE}")

    teacher_df, X, y, filenames, positions = build_grouped_dataset()

    train_idx, val_idx, train_files, val_files = split_by_filename(
        filenames, val_ratio=VAL_RATIO, seed=SEED
    )

    print("\nGroup split by filename")
    print(f"Train files: {len(train_files)} | Val files: {len(val_files)}")
    print(f"Train segments: {len(train_idx)} | Val segments: {len(val_idx)}")

    X_train = X[train_idx]
    y_train = y[train_idx]
    fn_train = filenames[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]
    fn_val = filenames[val_idx]

    y_min = float(y_train.min())
    y_max = float(y_train.max())
    y_range = float(y_max - y_min)

    train_set = WaveformDataset(
        X_train, y_train, fn_train,
        y_min=y_min, y_range=y_range,
        is_train=True, noise_std=NOISE_STD
    )
    val_set = WaveformDataset(
        X_val, y_val, fn_val,
        y_min=y_min, y_range=y_range,
        is_train=False, noise_std=NOISE_STD
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    model = ResLightWaveNet().to(DEVICE)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_seg_mae": [],
        "val_file_mae": []
    }

    best_file_mae = float("inf")

    def denorm_fn(arr):
        return arr * y_range + y_min

    print("\nStart grouped training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for x, y_norm, _ in train_loader:
            x = x.to(DEVICE)
            y_norm = y_norm.to(DEVICE)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y_norm)
            loss.backward()
            optimizer.step()

            batch_n = x.size(0)
            train_loss_sum += loss.item() * batch_n
            train_n += batch_n

        train_loss = train_loss_sum / train_n

        val_loss, val_seg_mae, val_file_mae, _ = evaluate_model(
            model, val_loader, denormalize_fn=denorm_fn, criterion=criterion
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_seg_mae"].append(val_seg_mae)
        history["val_file_mae"].append(val_file_mae)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch+1:03d}/{EPOCHS}] "
            f"TrainLoss={train_loss:.6f} | "
            f"ValLoss={val_loss:.6f} | "
            f"ValSegMAE={val_seg_mae:.6f} px | "
            f"ValFileMAE={val_file_mae:.6f} px | "
            f"LR={current_lr:.7f}"
        )

        if val_file_mae < best_file_mae:
            best_file_mae = val_file_mae
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "y_min": y_min,
                    "y_max": y_max,
                    "y_range": y_range,
                    "train_files": sorted(list(train_files), key=numeric_stem_key),
                    "val_files": sorted(list(val_files), key=numeric_stem_key),
                    "seed": SEED,
                    "config": {
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "epochs": EPOCHS,
                        "weight_decay": WEIGHT_DECAY,
                        "noise_std": NOISE_STD,
                        "val_ratio": VAL_RATIO
                    }
                },
                CHECKPOINT_PATH
            )

    print(f"\nBest grouped checkpoint saved to: {CHECKPOINT_PATH}")
    print(f"Best validation file-level MAE: {best_file_mae:.6f} pixel")

    # Training curve
    plt.figure(figsize=(10, 5), dpi=180)
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Val loss")
    plt.plot(history["val_file_mae"], label="Val file MAE (pixel)")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Metric (log scale)")
    plt.title("Grouped training curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(TRAIN_CURVE_PATH, bbox_inches="tight")
    plt.close()

    # Final evaluation with best checkpoint
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    make_val_report(
        model=model,
        val_loader=val_loader,
        denormalize_fn=lambda arr: arr * ckpt["y_range"] + ckpt["y_min"],
        teacher_df=teacher_df,
        val_files=set(ckpt["val_files"])
    )


if __name__ == "__main__":
    train()