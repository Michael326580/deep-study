#!/usr/bin/env python3
# =========================================================
# [脚本说明]
# 用途：高性能主训练脚本（深度残差 1D-CNN + SE 注意力），同时支持推理。
# 输入：1. 设置随机种子与基础评估函数（MAE/RMSE/MaxAE）。
# 输出：2. `build_file_level_split` 依据 `Filename` 做文件级切分。
# =========================================================

# =========================================================
# [????]
# ???????????????? 1D-CNN + SE ????
# ???dataset_distill.npz + grouped_dataset_metadata.csv?
# ?????/???????????? CSV?
# =========================================================

"""
High-performance 1D-CNN student training for deep-study.

Key features:
- File-level grouped split (no segment leakage)
- Deep residual 1D-CNN with SE attention
- AdamW + CosineAnnealingWarmRestarts
- Mixed SmoothL1 + Huber regression loss
- Signal augmentations: gaussian noise, random shift, amplitude scaling
- Early stopping on file-level MAE
- Reports segment/file metrics: MAE, RMSE, MaxAE

Usage examples:
  python train_attention_res1d_student.py train \
      --npz data/processed/dataset_distill.npz \
      --meta data/processed/grouped_dataset_metadata.csv \
      --out-dir results/models/runs/attn_res1d

  python train_attention_res1d_student.py infer \
      --checkpoint results/models/runs/attn_res1d/best_attention_res1d.pth \
      --npz data/processed/dataset_distill.npz \
      --meta data/processed/grouped_dataset_metadata.csv \
      --save-csv results/models/runs/attn_res1d/infer_all.csv
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Utility metrics
# =========================================================
def mae(arr: np.ndarray) -> float:
    return float(np.mean(np.abs(arr)))


def rmse(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(arr))))


def maxae(arr: np.ndarray) -> float:
    return float(np.max(np.abs(arr)))


@dataclass
class EvalResult:
    seg_mae: float
    seg_rmse: float
    seg_maxae: float
    file_mae: float
    file_rmse: float
    file_maxae: float
    loss: float
    pred_df: pd.DataFrame


# =========================================================
# Data split and dataset
# =========================================================
def numeric_stem_key(name: str):
    stem = Path(str(name)).stem
    try:
        return int(stem)
    except ValueError:
        return stem


def build_file_level_split(
    filenames: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """File-level grouped split to avoid leakage."""
    unique_files = np.array(sorted(np.unique(filenames), key=numeric_stem_key), dtype=object)
    rng = np.random.default_rng(seed)
    shuffled = unique_files.copy()
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(unique_files) * val_ratio)))
    val_files = shuffled[:n_val].tolist()
    train_files = shuffled[n_val:].tolist()

    train_idx = np.where(np.isin(filenames, train_files))[0]
    val_idx = np.where(np.isin(filenames, val_files))[0]
    return train_idx, val_idx, train_files, val_files


class DistillDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        filenames: np.ndarray,
        y_min: float,
        y_range: float,
        train_mode: bool,
        noise_std: float,
        shift_max: int,
        amp_scale: float,
    ):
        self.X = torch.from_numpy(X).float().permute(0, 2, 1)  # [N, C=2, L]
        self.filenames = filenames.astype(str)

        self.y_min = float(y_min)
        self.y_range = float(y_range)
        self.y_raw = y.astype(np.float32)
        self.y_norm = ((self.y_raw - self.y_min) / self.y_range).astype(np.float32)
        self.y = torch.from_numpy(self.y_norm).float().unsqueeze(1)

        self.train_mode = bool(train_mode)
        self.noise_std = float(noise_std)
        self.shift_max = int(shift_max)
        self.amp_scale = float(amp_scale)

    def __len__(self) -> int:
        return len(self.X)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        if self.shift_max > 0:
            shift = int(torch.randint(low=-self.shift_max, high=self.shift_max + 1, size=(1,)).item())
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=-1)

        if self.amp_scale > 0:
            scale = 1.0 + float(torch.empty(1).uniform_(-self.amp_scale, self.amp_scale).item())
            x = x * scale

        return x

    def __getitem__(self, idx: int):
        x = self.X[idx].clone()
        if self.train_mode:
            x = self._augment(x)
        y = self.y[idx]
        fname = self.filenames[idx]
        return x, y, fname


# =========================================================
# Model: Deep Residual 1D CNN + SE Attention
# =========================================================
class SEAttention1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))
        return x * w


class ResidualAttnBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEAttention1D(out_ch, reduction=8)
        self.drop = nn.Dropout(dropout)

        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.short = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.short(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.drop(out)
        out = out + residual
        out = self.act(out)
        return out


class DeepAttnRes1DRegressor(nn.Module):
    def __init__(self, in_channels: int = 2, base_width: int = 32, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_width),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_stage(base_width, base_width, blocks=2, stride=1, dropout=dropout)
        self.layer2 = self._make_stage(base_width, base_width * 2, blocks=2, stride=2, dropout=dropout)
        self.layer3 = self._make_stage(base_width * 2, base_width * 4, blocks=2, stride=2, dropout=dropout)
        self.layer4 = self._make_stage(base_width * 4, base_width * 8, blocks=2, stride=2, dropout=dropout)

        feat_ch = base_width * 8
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(feat_ch, feat_ch // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_ch // 2, 1),
        )

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, blocks: int, stride: int, dropout: float) -> nn.Sequential:
        layers = [ResidualAttnBlock(in_ch, out_ch, stride=stride, dropout=dropout)]
        for _ in range(1, blocks):
            layers.append(ResidualAttnBlock(out_ch, out_ch, stride=1, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


# =========================================================
# Mixed robust loss
# =========================================================
class MixedRobustLoss(nn.Module):
    def __init__(self, smooth_beta: float = 1.0, huber_delta: float = 1.0, alpha: float = 0.5):
        super().__init__()
        self.smooth = nn.SmoothL1Loss(beta=smooth_beta)
        self.huber = nn.HuberLoss(delta=huber_delta)
        self.alpha = float(alpha)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.smooth(pred, target) + (1.0 - self.alpha) * self.huber(pred, target)


# =========================================================
# Evaluation
# =========================================================
def denorm(y_norm: np.ndarray, y_min: float, y_range: float) -> np.ndarray:
    return y_norm * y_range + y_min


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    y_min: float,
    y_range: float,
) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total_n = 0

    all_pred: List[float] = []
    all_true: List[float] = []
    all_fname: List[str] = []

    with torch.no_grad():
        for x, y, fnames in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            batch_n = x.size(0)
            total_loss += float(loss.item()) * batch_n
            total_n += batch_n

            pred_raw = denorm(pred.detach().cpu().numpy().reshape(-1), y_min, y_range)
            true_raw = denorm(y.detach().cpu().numpy().reshape(-1), y_min, y_range)

            all_pred.extend(pred_raw.tolist())
            all_true.extend(true_raw.tolist())
            all_fname.extend(list(fnames))

    pred_arr = np.asarray(all_pred, dtype=float)
    true_arr = np.asarray(all_true, dtype=float)
    seg_err = pred_arr - true_arr

    seg_df = pd.DataFrame({
        "Filename": all_fname,
        "Pred_Width": pred_arr,
        "True_Width": true_arr,
    })

    file_df = (
        seg_df.groupby("Filename", as_index=False)
        .agg(
            Pred_Width=("Pred_Width", "mean"),
            True_Width=("True_Width", "first"),
            Segment_Count=("Pred_Width", "size"),
        )
    )
    file_err = file_df["Pred_Width"].to_numpy(dtype=float) - file_df["True_Width"].to_numpy(dtype=float)

    return EvalResult(
        seg_mae=mae(seg_err),
        seg_rmse=rmse(seg_err),
        seg_maxae=maxae(seg_err),
        file_mae=mae(file_err),
        file_rmse=rmse(file_err),
        file_maxae=maxae(file_err),
        loss=total_loss / max(total_n, 1),
        pred_df=file_df,
    )


# =========================================================
# Training
# =========================================================
def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    npz_path = Path(args.npz)
    meta_path = Path(args.meta)
    if not meta_path.exists():
        alt_meta = Path("data/processed/dataset_distill_meta.csv")
        if str(args.meta) in {"grouped_dataset_metadata.csv", "data/processed/grouped_dataset_metadata.csv"} and alt_meta.exists():
            print(
                "[INFO] Metadata CSV not found at grouped_dataset_metadata.csv; "
                "fallback to dataset_distill_meta.csv"
            )
            meta_path = alt_meta
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found: {meta_path}. "
            "File-level split requires per-segment filename metadata."
        )

    distill = np.load(npz_path)
    if "X" not in distill or "y" not in distill:
        raise KeyError("dataset_distill.npz must contain keys: X, y")

    X = distill["X"].astype(np.float32)
    y = distill["y"].astype(np.float32)

    meta_df = pd.read_csv(meta_path)
    if "Filename" not in meta_df.columns:
        raise KeyError("Metadata CSV must contain 'Filename' column")

    if len(meta_df) != len(X):
        raise ValueError(
            "Metadata length does not match NPZ sample count. "
            f"meta={len(meta_df)}, npz={len(X)}"
        )

    filenames = meta_df["Filename"].astype(str).to_numpy()

    train_idx, val_idx, train_files, val_files = build_file_level_split(
        filenames=filenames,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    X_train, y_train, fn_train = X[train_idx], y[train_idx], filenames[train_idx]
    X_val, y_val, fn_val = X[val_idx], y[val_idx], filenames[val_idx]

    y_min = float(np.min(y_train))
    y_max = float(np.max(y_train))
    y_range = float(y_max - y_min)
    if y_range <= 1e-12:
        raise ValueError("y_range is too small; cannot normalize labels safely.")

    train_set = DistillDataset(
        X=X_train,
        y=y_train,
        filenames=fn_train,
        y_min=y_min,
        y_range=y_range,
        train_mode=True,
        noise_std=args.noise_std,
        shift_max=args.shift_max,
        amp_scale=args.amp_scale,
    )
    val_set = DistillDataset(
        X=X_val,
        y=y_val,
        filenames=fn_val,
        y_min=y_min,
        y_range=y_range,
        train_mode=False,
        noise_std=0.0,
        shift_max=0,
        amp_scale=0.0,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = DeepAttnRes1DRegressor(base_width=args.base_width, dropout=args.dropout).to(device)
    criterion = MixedRobustLoss(
        smooth_beta=args.smooth_beta,
        huber_delta=args.huber_delta,
        alpha=args.loss_alpha,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.t0,
        T_mult=args.t_mult,
        eta_min=args.min_lr,
    )

    best_file_mae = float("inf")
    best_epoch = -1
    patience_count = 0

    history_rows: List[Dict[str, float]] = []
    ckpt_path = out_dir / "best_attention_res1d.pth"
    last_path = out_dir / "last_attention_res1d.pth"
    hist_csv = out_dir / "train_history.csv"
    val_pred_csv = out_dir / "best_val_predictions.csv"

    print("=" * 80)
    print("Train DeepAttnRes1D student")
    print(f"Device: {device}")
    print(f"Train files: {len(train_files)} | Val files: {len(val_files)}")
    print(f"Train segments: {len(train_set)} | Val segments: {len(val_set)}")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for step, (x, y_norm, _) in enumerate(train_loader):
            x = x.to(device)
            y_norm = y_norm.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y_norm)
            loss.backward()

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            global_step = (epoch - 1) + step / max(len(train_loader), 1)
            scheduler.step(global_step)

            bs = x.size(0)
            train_loss_sum += float(loss.item()) * bs
            train_n += bs

        train_loss = train_loss_sum / max(train_n, 1)

        val_res = evaluate(model, val_loader, criterion, device, y_min, y_range)

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_res.loss,
            "seg_mae": val_res.seg_mae,
            "seg_rmse": val_res.seg_rmse,
            "seg_maxae": val_res.seg_maxae,
            "file_mae": val_res.file_mae,
            "file_rmse": val_res.file_rmse,
            "file_maxae": val_res.file_maxae,
        }
        history_rows.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"TrainLoss={train_loss:.6f} ValLoss={val_res.loss:.6f} | "
            f"SegMAE={val_res.seg_mae:.6f} SegRMSE={val_res.seg_rmse:.6f} SegMaxAE={val_res.seg_maxae:.6f} | "
            f"FileMAE={val_res.file_mae:.6f} FileRMSE={val_res.file_rmse:.6f} FileMaxAE={val_res.file_maxae:.6f}"
        )

        improved = val_res.file_mae < (best_file_mae - args.min_delta)
        if improved:
            best_file_mae = val_res.file_mae
            best_epoch = epoch
            patience_count = 0

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "y_min": y_min,
                    "y_max": y_max,
                    "y_range": y_range,
                    "train_files": sorted(train_files, key=numeric_stem_key),
                    "val_files": sorted(val_files, key=numeric_stem_key),
                    "seed": args.seed,
                    "model": {
                        "name": "DeepAttnRes1DRegressor",
                        "base_width": args.base_width,
                        "dropout": args.dropout,
                    },
                    "train_args": vars(args),
                    "best_epoch": best_epoch,
                    "best_file_mae": best_file_mae,
                },
                ckpt_path,
            )
            val_res.pred_df.to_csv(val_pred_csv, index=False)
        else:
            patience_count += 1

        if patience_count >= args.early_stop_patience:
            print(
                f"Early stopping triggered at epoch={epoch}. "
                f"Best epoch={best_epoch}, best file MAE={best_file_mae:.6f}."
            )
            break

    # Save last checkpoint and training history
    torch.save(
        {
            "state_dict": model.state_dict(),
            "y_min": y_min,
            "y_max": y_max,
            "y_range": y_range,
            "train_files": sorted(train_files, key=numeric_stem_key),
            "val_files": sorted(val_files, key=numeric_stem_key),
            "seed": args.seed,
            "model": {
                "name": "DeepAttnRes1DRegressor",
                "base_width": args.base_width,
                "dropout": args.dropout,
            },
            "train_args": vars(args),
        },
        last_path,
    )

    pd.DataFrame(history_rows).to_csv(hist_csv, index=False)

    summary = {
        "best_epoch": best_epoch,
        "best_file_mae": best_file_mae,
        "checkpoint": str(ckpt_path),
        "last_checkpoint": str(last_path),
        "history_csv": str(hist_csv),
        "best_val_pred_csv": str(val_pred_csv),
    }
    with open(out_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nTraining finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# =========================================================
# Inference helper
# =========================================================
def infer(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.checkpoint)
    npz_path = Path(args.npz)
    meta_path = Path(args.meta)
    if not meta_path.exists():
        alt_meta = Path("data/processed/dataset_distill_meta.csv")
        if str(args.meta) in {"grouped_dataset_metadata.csv", "data/processed/grouped_dataset_metadata.csv"} and alt_meta.exists():
            print(
                "[INFO] Metadata CSV not found at grouped_dataset_metadata.csv; "
                "fallback to dataset_distill_meta.csv"
            )
            meta_path = alt_meta

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {meta_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    base_width = int(ckpt.get("model", {}).get("base_width", 32))
    dropout = float(ckpt.get("model", {}).get("dropout", 0.1))

    model = DeepAttnRes1DRegressor(base_width=base_width, dropout=dropout)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    distill = np.load(npz_path)
    X = distill["X"].astype(np.float32)

    meta_df = pd.read_csv(meta_path)
    if "Filename" not in meta_df.columns:
        raise KeyError("Metadata CSV must contain 'Filename' column")
    if len(meta_df) != len(X):
        raise ValueError(f"Metadata length mismatch: meta={len(meta_df)} npz={len(X)}")

    filenames = meta_df["Filename"].astype(str).to_numpy()

    y_min = float(ckpt["y_min"])
    y_range = float(ckpt["y_range"])

    X_t = torch.from_numpy(X).float().permute(0, 2, 1)
    bs = int(args.batch_size)

    preds: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X_t), bs):
            pred_norm = model(X_t[i:i + bs]).squeeze(1).numpy()
            pred_raw = denorm(pred_norm, y_min, y_range)
            preds.append(pred_raw)

    pred_all = np.concatenate(preds)

    seg_df = pd.DataFrame({
        "Filename": filenames,
        "Pred_Width": pred_all,
    })
    file_df = (
        seg_df.groupby("Filename", as_index=False)
        .agg(
            Pred_Width=("Pred_Width", "mean"),
            Segment_Count=("Pred_Width", "size"),
        )
        .sort_values("Filename", key=lambda s: s.map(numeric_stem_key))
    )

    if args.save_csv:
        save_csv = Path(args.save_csv)
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        file_df.to_csv(save_csv, index=False)
        print(f"Saved inference CSV: {save_csv}")

    print(file_df.head(10).to_string(index=False))


# =========================================================
# CLI
# =========================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or infer with DeepAttnRes1D student model.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train model")
    p_train.add_argument("--npz", type=str, default="data/processed/dataset_distill.npz")
    p_train.add_argument("--meta", type=str, default="dataset_distill_meta.csv")
    p_train.add_argument("--out-dir", type=str, default="results/models/runs/attention_res1d")

    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--val-ratio", type=float, default=0.2)

    p_train.add_argument("--epochs", type=int, default=180)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--num-workers", type=int, default=0)

    p_train.add_argument("--lr", type=float, default=2e-3)
    p_train.add_argument("--min-lr", type=float, default=1e-6)
    p_train.add_argument("--weight-decay", type=float, default=1e-4)
    p_train.add_argument("--grad-clip", type=float, default=1.0)

    # Warm restarts scheduler
    p_train.add_argument("--t0", type=int, default=10)
    p_train.add_argument("--t-mult", type=int, default=2)

    # Model
    p_train.add_argument("--base-width", type=int, default=32)
    p_train.add_argument("--dropout", type=float, default=0.15)

    # Loss
    p_train.add_argument("--smooth-beta", type=float, default=1.0)
    p_train.add_argument("--huber-delta", type=float, default=1.0)
    p_train.add_argument("--loss-alpha", type=float, default=0.5)

    # Augmentations
    p_train.add_argument("--noise-std", type=float, default=0.005)
    p_train.add_argument("--shift-max", type=int, default=12)
    p_train.add_argument("--amp-scale", type=float, default=0.08)

    # Early stopping
    p_train.add_argument("--early-stop-patience", type=int, default=30)
    p_train.add_argument("--min-delta", type=float, default=1e-5)

    p_infer = sub.add_parser("infer", help="Run inference")
    p_infer.add_argument("--checkpoint", type=str, required=True)
    p_infer.add_argument("--npz", type=str, default="data/processed/dataset_distill.npz")
    p_infer.add_argument("--meta", type=str, default="dataset_distill_meta.csv")
    p_infer.add_argument("--batch-size", type=int, default=256)
    p_infer.add_argument("--save-csv", type=str, default="")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "train":
        train(args)
    elif args.cmd == "infer":
        infer(args)
    else:
        raise ValueError(f"Unsupported command: {args.cmd}")


if __name__ == "__main__":
    main()
