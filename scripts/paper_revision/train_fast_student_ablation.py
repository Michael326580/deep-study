#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train and evaluate lightweight student candidates for the deep-study fringe-width regression paper.

Purpose
-------
This script follows the same file-level split philosophy as train_attention_res1d_student.py, but
trains much smaller candidates whose latency was screened by benchmark_candidate_fast_students.py.
It reports both width-level and calibrated displacement-level metrics, and benchmarks the trained
checkpoint on input shape [1, 2, 2048].

Run from repository root, for example:

    python scripts/paper_revision/train_fast_student_ablation.py \
        --models plain_cnn_w8 plain_cnn_w16 tiny_dw_w8_d3 \
        --epochs 160 --batch-size 64 --device cpu --threads 1

Outputs
-------
    results/models/runs/fast_student_ablation/<model_name>/
        best_<model_name>.pth
        train_history.csv
        best_val_predictions.csv
        train_summary.json

    results/paper_revision_latency/fast_student_ablation_summary.csv
    results/paper_revision_latency/fast_student_ablation_summary.md

Notes
-----
- Accuracy evidence only becomes valid after training finishes.
- Speed evidence is CPU/PyTorch-specific unless you additionally export to ONNX/RKNN/OpenVINO.
- The paper should compare a trained lightweight student, not the untrained latency-screening model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def find_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "data").exists() and (parent / "results").exists():
            return parent
    return Path.cwd()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def numeric_stem_key(name: str):
    stem = Path(str(name)).stem
    try:
        return int(stem)
    except Exception:
        return stem


def mae(err: np.ndarray) -> float:
    return float(np.mean(np.abs(err)))


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(err))))


def maxae(err: np.ndarray) -> float:
    return float(np.max(np.abs(err)))


def ensure_x_shape(X: np.ndarray) -> np.ndarray:
    """Return X as [N, 2, 2048]-like float32 tensor array."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected X with 3 dimensions, got shape {X.shape}")
    # common dataset_distill.npz shape: [N, 2048, 2]
    if X.shape[1] == 2048 and X.shape[2] == 2:
        X = np.transpose(X, (0, 2, 1))
    elif X.shape[1] == 2:
        pass
    else:
        raise ValueError(f"Cannot infer channel/length dimensions from X shape {X.shape}")
    return X.astype(np.float32)


def build_file_level_split(filenames: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    unique_files = np.array(sorted(np.unique(filenames.astype(str)), key=numeric_stem_key), dtype=object)
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
    def __init__(self, X: np.ndarray, y: np.ndarray, filenames: np.ndarray, positions: np.ndarray,
                 y_min: float, y_range: float, train: bool,
                 noise_std: float, shift_max: int, amp_scale: float):
        self.X = torch.from_numpy(X).float()
        self.y_raw = np.asarray(y, dtype=np.float32)
        self.y = torch.from_numpy(((self.y_raw - y_min) / y_range).astype(np.float32)).view(-1, 1)
        self.filenames = filenames.astype(str)
        self.positions = np.asarray(positions, dtype=np.float32)
        self.train = bool(train)
        self.noise_std = float(noise_std)
        self.shift_max = int(shift_max)
        self.amp_scale = float(amp_scale)

    def __len__(self) -> int:
        return len(self.X)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        if self.shift_max > 0:
            shift = int(torch.randint(-self.shift_max, self.shift_max + 1, (1,)).item())
            if shift:
                x = torch.roll(x, shifts=shift, dims=-1)
        if self.amp_scale > 0:
            scale = 1.0 + float(torch.empty(1).uniform_(-self.amp_scale, self.amp_scale).item())
            x = x * scale
        return x

    def __getitem__(self, idx: int):
        x = self.X[idx].clone()
        if self.train:
            x = self._augment(x)
        return x, self.y[idx], self.filenames[idx], float(self.positions[idx])


class DSConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, k: int = 7):
        super().__init__()
        pad = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=k, stride=stride, padding=pad, groups=in_ch, bias=False),
            nn.BatchNorm1d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyDWStudent(nn.Module):
    def __init__(self, width: int = 8, depth: int = 4, in_channels: int = 2):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv1d(in_channels, width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        ]
        ch = width
        for i in range(depth):
            out = width * min(2 ** (i // 2), 4)
            layers.append(DSConvBlock(ch, out, stride=2 if i in {0, 1, 2} else 1, k=7))
            ch = out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(ch, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class SmallPlainCNN(nn.Module):
    def __init__(self, width: int = 16, in_channels: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, width, 9, stride=2, padding=4, bias=False), nn.BatchNorm1d(width), nn.ReLU(inplace=True),
            nn.Conv1d(width, width * 2, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(width * 2), nn.ReLU(inplace=True),
            nn.Conv1d(width * 2, width * 4, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(width * 4), nn.ReLU(inplace=True),
            nn.Conv1d(width * 4, width * 4, 3, stride=2, padding=1, bias=False), nn.BatchNorm1d(width * 4), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(width * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_model(name: str) -> nn.Module:
    if name == "tiny_dw_w8_d3":
        return TinyDWStudent(width=8, depth=3)
    if name == "tiny_dw_w8_d4":
        return TinyDWStudent(width=8, depth=4)
    if name == "tiny_dw_w16_d4":
        return TinyDWStudent(width=16, depth=4)
    if name == "plain_cnn_w8":
        return SmallPlainCNN(width=8)
    if name == "plain_cnn_w16":
        return SmallPlainCNN(width=16)
    if name == "plain_cnn_w24":
        return SmallPlainCNN(width=24)
    if name == "plain_cnn_w32":
        return SmallPlainCNN(width=32)
    raise ValueError(f"Unsupported model name: {name}")


def count_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))


def estimate_macs(model: nn.Module, x: torch.Tensor) -> int:
    macs = 0
    hooks = []

    def hconv(m, inp, out):
        nonlocal macs
        b, cout, lout = out.shape
        macs += int(b * cout * lout * (m.in_channels // m.groups) * m.kernel_size[0])

    def hlin(m, inp, out):
        nonlocal macs
        b = out.shape[0] if getattr(out, 'ndim', 0) else 1
        macs += int(b * m.in_features * m.out_features)

    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            hooks.append(module.register_forward_hook(hconv))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(hlin))
    model.eval()
    with torch.inference_mode():
        model(x)
    for h in hooks:
        h.remove()
    return int(macs)


def fit_width_to_position(fixed_csv: Path) -> Tuple[Callable[[np.ndarray], np.ndarray], str]:
    df = pd.read_csv(fixed_csv)
    w = df["FFT_Width"].to_numpy(dtype=float)
    x = df["Position_mm"].to_numpy(dtype=float)
    try:
        from scipy.optimize import curve_fit  # type: ignore

        def inv_hyperbola(width, a, b, c):
            return a / (width - b) + c

        p0 = [1000.0, float(np.min(w) - 1.0), 150.0]
        popt, _ = curve_fit(inv_hyperbola, w, x, p0=p0, maxfev=20000)

        def mapper(width_arr: np.ndarray) -> np.ndarray:
            return inv_hyperbola(np.asarray(width_arr, dtype=float), *popt)

        desc = f"inverse_hyperbola: x = a/(W-b)+c, a={popt[0]:.9g}, b={popt[1]:.9g}, c={popt[2]:.9g}"
        return mapper, desc
    except Exception as e:
        # Fallback: a local polynomial is less physical but avoids blocking the ablation script.
        coef = np.polyfit(w, x, deg=3)

        def mapper(width_arr: np.ndarray) -> np.ndarray:
            return np.polyval(coef, np.asarray(width_arr, dtype=float))

        desc = "fallback_poly3_width_to_position: " + ",".join(f"{c:.9g}" for c in coef) + f"; scipy_fit_error={repr(e)}"
        return mapper, desc


@dataclass
class EvalResult:
    loss: float
    seg_width_mae: float
    file_width_mae: float
    pos_mae: float
    pos_rmse: float
    pos_maxae: float
    pred_df: pd.DataFrame


def denorm(y_norm: np.ndarray, y_min: float, y_range: float) -> np.ndarray:
    return y_norm * y_range + y_min


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
             y_min: float, y_range: float, width_to_pos: Callable[[np.ndarray], np.ndarray]) -> EvalResult:
    model.eval()
    losses = 0.0
    total = 0
    rows = []
    with torch.inference_mode():
        for xb, yb, fnames, pos in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            n = xb.shape[0]
            losses += float(loss.item()) * n
            total += n
            pred_w = denorm(pred.detach().cpu().numpy().reshape(-1), y_min, y_range)
            true_w = denorm(yb.detach().cpu().numpy().reshape(-1), y_min, y_range)
            pos_np = np.asarray(pos, dtype=float)
            for f, pw, tw, pp in zip(list(fnames), pred_w, true_w, pos_np):
                rows.append((str(f), float(pw), float(tw), float(pp)))
    seg_df = pd.DataFrame(rows, columns=["Filename", "Pred_Width", "True_Width", "Position_mm"])
    seg_err = seg_df["Pred_Width"].to_numpy(float) - seg_df["True_Width"].to_numpy(float)
    file_df = seg_df.groupby("Filename", as_index=False).agg(
        Pred_Width=("Pred_Width", "mean"),
        True_Width=("True_Width", "first"),
        Position_mm=("Position_mm", "first"),
        Segment_Count=("Pred_Width", "size"),
    )
    pred_pos = width_to_pos(file_df["Pred_Width"].to_numpy(float))
    file_df["Pred_Position_mm"] = pred_pos
    file_df["Position_Error_mm"] = file_df["Pred_Position_mm"].to_numpy(float) - file_df["Position_mm"].to_numpy(float)
    pos_err = file_df["Position_Error_mm"].to_numpy(float)
    file_w_err = file_df["Pred_Width"].to_numpy(float) - file_df["True_Width"].to_numpy(float)
    return EvalResult(
        loss=losses / max(total, 1),
        seg_width_mae=mae(seg_err),
        file_width_mae=mae(file_w_err),
        pos_mae=mae(pos_err),
        pos_rmse=rmse(pos_err),
        pos_maxae=maxae(pos_err),
        pred_df=file_df,
    )


def benchmark_model(model: nn.Module, device: torch.device, length: int, warmup: int, repeat: int) -> Dict[str, float]:
    x = torch.randn(1, 2, length, device=device)
    model = model.to(device).eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter_ns()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1e6)
    arr = np.asarray(times, dtype=float)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def train_one(model_name: str, args: argparse.Namespace, root: Path, device: torch.device,
              X: np.ndarray, y: np.ndarray, filenames: np.ndarray, positions: np.ndarray,
              train_idx: np.ndarray, val_idx: np.ndarray, train_files: List[str], val_files: List[str],
              width_to_pos: Callable[[np.ndarray], np.ndarray], calib_desc: str) -> Dict[str, Any]:
    out_dir = root / args.out_base / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    y_train = y[train_idx]
    y_min = float(np.min(y_train))
    y_max = float(np.max(y_train))
    y_range = y_max - y_min
    if y_range <= 1e-12:
        raise ValueError("Label range is too small.")

    train_set = DistillDataset(X[train_idx], y[train_idx], filenames[train_idx], positions[train_idx],
                               y_min, y_range, True, args.noise_std, args.shift_max, args.amp_scale)
    val_set = DistillDataset(X[val_idx], y[val_idx], filenames[val_idx], positions[val_idx],
                             y_min, y_range, False, 0.0, 0, 0.0)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = make_model(model_name).to(device)
    params = count_params(model)
    macs = estimate_macs(model.cpu(), torch.randn(1, 2, args.length))
    model = model.to(device)

    criterion = nn.SmoothL1Loss(beta=args.smooth_beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    best = None
    best_epoch = -1
    patience = 0
    hist = []
    ckpt_path = out_dir / f"best_{model_name}.pth"

    print("=" * 80)
    print(f"Training {model_name} | params={params} | MACs={macs/1e6:.3f}M | device={device}")
    print(f"Train files={len(train_files)}, val files={len(val_files)}, train segments={len(train_set)}, val segments={len(val_set)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_sum = 0
        for xb, yb, _, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            bs = xb.shape[0]
            loss_sum += float(loss.item()) * bs
            n_sum += bs
        scheduler.step()
        train_loss = loss_sum / max(n_sum, 1)
        val = evaluate(model, val_loader, criterion, device, y_min, y_range, width_to_pos)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val.loss,
            "seg_width_mae": val.seg_width_mae,
            "file_width_mae": val.file_width_mae,
            "pos_mae": val.pos_mae,
            "pos_rmse": val.pos_rmse,
            "pos_maxae": val.pos_maxae,
        }
        hist.append(row)
        print(f"{model_name} epoch {epoch:03d}/{args.epochs} | loss={train_loss:.6f} | fileW_MAE={val.file_width_mae:.6f} px | posMAE={val.pos_mae:.6f} mm | posMaxAE={val.pos_maxae:.6f} mm")
        improved = best is None or val.pos_mae < (best["pos_mae"] - args.min_delta)
        if improved:
            best = row.copy()
            best_epoch = epoch
            patience = 0
            torch.save({
                "state_dict": model.state_dict(),
                "model_name": model_name,
                "y_min": y_min,
                "y_max": y_max,
                "y_range": y_range,
                "train_files": sorted(train_files, key=numeric_stem_key),
                "val_files": sorted(val_files, key=numeric_stem_key),
                "seed": args.seed,
                "calibration": calib_desc,
                "args": vars(args),
            }, ckpt_path)
            val.pred_df.to_csv(out_dir / "best_val_predictions.csv", index=False)
        else:
            patience += 1
        if patience >= args.early_stop_patience:
            print(f"Early stop {model_name}: best_epoch={best_epoch}")
            break

    pd.DataFrame(hist).to_csv(out_dir / "train_history.csv", index=False)

    # Reload best for latency, to avoid benchmarking a worse late epoch.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = make_model(model_name).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    latency = benchmark_model(model, device=device, length=args.length, warmup=args.warmup, repeat=args.repeat)

    summary = {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "params": params,
        "macs": macs,
        "macs_million": macs / 1e6,
        **(best or {}),
        **latency,
        "checkpoint": str(ckpt_path),
        "out_dir": str(out_dir),
    }
    with open(out_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/processed/dataset_distill.npz")
    ap.add_argument("--meta", default="data/processed/dataset_distill_meta.csv")
    ap.add_argument("--fixed-csv", default="data/processed/fixed_results.csv")
    ap.add_argument("--out-base", default="results/models/runs/fast_student_ablation")
    ap.add_argument("--report-dir", default="results/paper_revision_latency")
    ap.add_argument("--models", nargs="+", default=["plain_cnn_w8", "plain_cnn_w16", "tiny_dw_w8_d3"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--smooth-beta", type=float, default=0.5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--noise-std", type=float, default=0.003)
    ap.add_argument("--shift-max", type=int, default=6)
    ap.add_argument("--amp-scale", type=float, default=0.05)
    ap.add_argument("--early-stop-patience", type=int, default=35)
    ap.add_argument("--min-delta", type=float, default=1e-5)
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--repeat", type=int, default=1000)
    args = ap.parse_args()

    root = find_root()
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, args.threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    npz_path = root / args.npz
    meta_path = root / args.meta
    fixed_csv = root / args.fixed_csv
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    if not fixed_csv.exists():
        raise FileNotFoundError(fixed_csv)

    distill = np.load(npz_path)
    if "X" not in distill:
        raise KeyError("dataset_distill.npz must contain X")
    X = ensure_x_shape(distill["X"])
    meta_df = pd.read_csv(meta_path)
    if "Filename" not in meta_df.columns:
        raise KeyError("metadata must contain Filename")
    filenames = meta_df["Filename"].astype(str).to_numpy()
    if "Teacher_Width" in meta_df.columns:
        y = meta_df["Teacher_Width"].to_numpy(dtype=np.float32)
    elif "y" in distill:
        y = distill["y"].astype(np.float32)
    else:
        raise KeyError("Need Teacher_Width in metadata or y in npz")
    if "Position_mm" in meta_df.columns:
        positions = meta_df["Position_mm"].to_numpy(dtype=np.float32)
    else:
        raise KeyError("metadata must contain Position_mm")
    if len(X) != len(meta_df):
        raise ValueError(f"X/meta length mismatch: X={len(X)}, meta={len(meta_df)}")

    train_idx, val_idx, train_files, val_files = build_file_level_split(filenames, args.val_ratio, args.seed)
    width_to_pos, calib_desc = fit_width_to_position(fixed_csv)
    print(f"Calibration: {calib_desc}")

    summaries = []
    for name in args.models:
        summaries.append(train_one(name, args, root, device, X, y, filenames, positions, train_idx, val_idx, train_files, val_files, width_to_pos, calib_desc))

    report_dir = root / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summaries).sort_values(["pos_mae", "mean_ms"], ascending=[True, True])
    df.to_csv(report_dir / "fast_student_ablation_summary.csv", index=False)

    md = [
        "# Fast student ablation summary",
        "",
        f"- Device: `{device}`",
        f"- Input: `[1, 2, {args.length}]`",
        f"- Calibration: `{calib_desc}`",
        "",
        "| Model | Params | MACs(M) | Pos MAE (mm) | Pos RMSE (mm) | Pos MaxAE (mm) | Mean ms | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        md.append(f"| {r['model_name']} | {int(r['params'])} | {r['macs_million']:.3f} | {r['pos_mae']:.6f} | {r['pos_rmse']:.6f} | {r['pos_maxae']:.6f} | {r['mean_ms']:.4f} | {r['p95_ms']:.4f} |")
    md.extend([
        "",
        "## Environment",
        "",
        f"- Python: `{sys.version.split()[0]}`",
        f"- Platform: `{platform.platform()}`",
        f"- PyTorch: `{torch.__version__}`",
        f"- CPU threads: `{torch.get_num_threads()}`",
    ])
    (report_dir / "fast_student_ablation_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Saved summary: {report_dir / 'fast_student_ablation_summary.csv'}")
    print(f"Saved summary: {report_dir / 'fast_student_ablation_summary.md'}")


if __name__ == "__main__":
    main()
