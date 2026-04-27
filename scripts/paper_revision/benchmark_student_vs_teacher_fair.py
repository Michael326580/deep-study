#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fair latency benchmark for the deep-study conoscopy project.

Purpose
-------
1) Benchmark the real attention_res1d_model with the same input length used in the paper/data package: [1, 2, 2048].
2) Avoid misleading timing caused by wrong input length, too few warm-up iterations, time.time(), or missing hardware metadata.
3) Optionally benchmark the FFT/PCCC teacher core and file-level teacher pipeline when the raw CSV files are present.
4) Save a traceable JSON + Markdown report that can be cited in a paper only after inspection.

Recommended command from repository root:
    python scripts/paper_revision/benchmark_student_vs_teacher_fair.py --device cpu --threads 1 --repeat 1000 --warmup 100

Optional CUDA command, if available:
    python scripts/paper_revision/benchmark_student_vs_teacher_fair.py --device cuda --repeat 2000 --warmup 300

Notes
-----
- This script reports latency; it does not prove accuracy.
- Do not compare CPU teacher vs GPU student unless the paper explicitly states the hardware settings.
- If your trained checkpoint was trained on 1024-length segments, the script will still test 2048 because the current paper/data package says 2048 x 2. If the model cannot accept 2048, the paper and data pipeline must be reconciled.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Model definition copied from scripts/training/train_attention_res1d_student.py
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def repo_root_from_script() -> Path:
    # Expected local path: <repo>/scripts/paper_revision/this_script.py
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "results").exists() and (parent / "data").exists():
            return parent
    # fallback for the recommended layout
    return p.parents[2]


def quantiles_ms(times_ms: List[float]) -> Dict[str, float]:
    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median_ms": float(np.median(arr)),
        "p05_ms": float(np.percentile(arr, 5)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def estimate_macs_conv_linear(model: nn.Module, example: torch.Tensor) -> int:
    """Rough MAC count for Conv1d and Linear layers by forward hooks."""
    macs = 0
    hooks = []

    def hook_conv(module: nn.Conv1d, inputs: Tuple[torch.Tensor], output: torch.Tensor):
        nonlocal macs
        # output: [B, Cout, Lout]
        b, cout, lout = output.shape
        cin = module.in_channels
        k = module.kernel_size[0]
        groups = module.groups
        macs += int(b * cout * lout * (cin // groups) * k)

    def hook_linear(module: nn.Linear, inputs: Tuple[torch.Tensor], output: torch.Tensor):
        nonlocal macs
        batch = int(output.shape[0]) if output.ndim >= 1 else 1
        macs += int(batch * module.in_features * module.out_features)

    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(hook_conv))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(hook_linear))
    with torch.inference_mode():
        model(example)
    for h in hooks:
        h.remove()
    return int(macs)


def load_dataset_example(root: Path, length: int, channels: int) -> Tuple[torch.Tensor, str]:
    """Prefer a real dataset sample. Fallback to random data only if NPZ is absent."""
    npz_candidates = [
        root / "data/processed/dataset_distill.npz",
        root / "processed/dataset_distill.npz",
    ]
    for p in npz_candidates:
        if p.exists():
            data = np.load(p)
            if "X" not in data:
                continue
            x_np = data["X"][0].astype(np.float32)
            # Usually [L, C]
            if x_np.ndim != 2:
                raise ValueError(f"Unexpected X[0] shape in {p}: {x_np.shape}")
            if x_np.shape[0] == length and x_np.shape[1] == channels:
                x = torch.from_numpy(x_np).permute(1, 0).unsqueeze(0).contiguous()
                return x, f"real sample from {p}"
            if x_np.shape[0] == channels and x_np.shape[1] == length:
                x = torch.from_numpy(x_np).unsqueeze(0).contiguous()
                return x, f"real sample from {p}"
            raise ValueError(
                f"Dataset sample shape {x_np.shape} does not match required length={length}, channels={channels}. "
                "This must be fixed before latency can be cited."
            )
    x = torch.randn(1, channels, length, dtype=torch.float32)
    return x, "random fallback; dataset_distill.npz not found"


def load_model(root: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any], Path]:
    ckpt_path = root / "results/models/runs/attention_res1d_model/best_attention_res1d.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = ckpt.get("model", {"base_width": 32, "dropout": 0.1})
    model = DeepAttnRes1DRegressor(
        base_width=int(model_cfg.get("base_width", 32)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model, model_cfg, ckpt_path


def bench_torch_model(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeat: int,
    batch_size: int,
) -> Dict[str, Any]:
    if batch_size > 1:
        x = x.repeat(batch_size, 1, 1).contiguous()
    x = x.to(device)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
        synchronize_if_needed(device)

        times_ms: List[float] = []
        for _ in range(repeat):
            t0 = time.perf_counter_ns()
            _ = model(x)
            synchronize_if_needed(device)
            t1 = time.perf_counter_ns()
            times_ms.append((t1 - t0) / 1e6)

    stats = quantiles_ms(times_ms)
    stats.update({
        "batch_size": int(batch_size),
        "latency_per_sample_mean_ms": float(stats["mean_ms"] / batch_size),
        "latency_per_sample_median_ms": float(stats["median_ms"] / batch_size),
        "throughput_samples_per_s_mean": float(1000.0 * batch_size / stats["mean_ms"]),
        "repeat": int(repeat),
        "warmup": int(warmup),
        "input_shape": list(x.shape),
    })
    return stats


def try_torchscript(model: nn.Module, x: torch.Tensor, device: torch.device) -> Optional[nn.Module]:
    try:
        traced = torch.jit.trace(model, x.to(device))
        traced = torch.jit.freeze(traced.eval())
        return traced
    except Exception as exc:
        print(f"[WARN] TorchScript trace/freeze failed: {exc}")
        return None


def load_teacher_module(root: Path):
    module_path = root / "scripts/data_processing/pccc_fft_teacher.py"
    if not module_path.exists():
        return None, f"Teacher script not found: {module_path}"
    spec = importlib.util.spec_from_file_location("pccc_fft_teacher", module_path)
    if spec is None or spec.loader is None:
        return None, f"Cannot import teacher script: {module_path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, None


def bench_teacher_core_on_segment(root: Path, length: int, warmup: int, repeat: int) -> Dict[str, Any]:
    teacher, err = load_teacher_module(root)
    if teacher is None:
        return {"available": False, "reason": err}

    # Use real dataset sample if available; otherwise synthetic sinusoid.
    npz = root / "data/processed/dataset_distill.npz"
    if npz.exists():
        data = np.load(npz)
        x_np = data["X"][0].astype(np.float64)
        if x_np.shape[0] != length or x_np.shape[1] != 2:
            return {"available": False, "reason": f"dataset X shape {x_np.shape} does not match ({length}, 2)"}
        ch1 = x_np[:, 0]
        ch2 = x_np[:, 1]
        source = str(npz)
    else:
        t = np.arange(length, dtype=np.float64)
        ch1 = np.sin(2 * np.pi * t / 40.0)
        ch2 = -np.sin(2 * np.pi * t / 40.0)
        source = "synthetic sinusoid fallback"

    # pccc function expects pandas Series because it calls .values
    try:
        import pandas as pd
        s1 = pd.Series(ch1)
        s2 = pd.Series(ch2)
        for _ in range(warmup):
            teacher.calculate_width_constrained(s1, s2)
        times_ms = []
        for _ in range(repeat):
            t0 = time.perf_counter_ns()
            teacher.calculate_width_constrained(s1, s2)
            t1 = time.perf_counter_ns()
            times_ms.append((t1 - t0) / 1e6)
        stats = quantiles_ms(times_ms)
        stats.update({"available": True, "source": source, "repeat": repeat, "warmup": warmup})
        return stats
    except Exception as exc:
        return {"available": False, "reason": repr(exc)}


def bench_teacher_file_level(root: Path, warmup_files: int, repeat_files: int) -> Dict[str, Any]:
    teacher, err = load_teacher_module(root)
    if teacher is None:
        return {"available": False, "reason": err}
    raw_dir = root / "data/raw/1.22"
    if not raw_dir.exists():
        return {"available": False, "reason": f"Raw directory not found: {raw_dir}"}
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        return {"available": False, "reason": f"No CSV files found in {raw_dir}"}
    random.seed(0)
    sample_files = files[: max(1, min(repeat_files, len(files)))]
    try:
        for fp in sample_files[: max(0, min(warmup_files, len(sample_files)))]:
            teacher.process_single_file_mlv(fp)
        times_ms = []
        for fp in sample_files:
            t0 = time.perf_counter_ns()
            teacher.process_single_file_mlv(fp)
            t1 = time.perf_counter_ns()
            times_ms.append((t1 - t0) / 1e6)
        stats = quantiles_ms(times_ms)
        stats.update({"available": True, "n_files": len(sample_files), "raw_dir": str(raw_dir)})
        return stats
    except Exception as exc:
        return {"available": False, "reason": repr(exc)}


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = []
    lines.append("# Fair latency benchmark report\n")
    lines.append("## Environment\n")
    env = report["environment"]
    for k, v in env.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("\n## Student model\n")
    lines.append(f"- Checkpoint: `{report['checkpoint']}`")
    lines.append(f"- Model config: `{report['model_config']}`")
    lines.append(f"- Parameters: `{report['student_model']['num_parameters']}`")
    lines.append(f"- Estimated Conv/Linear MACs for batch=1: `{report['student_model']['estimated_macs_batch1']}`")
    lines.append(f"- Input source: `{report['input_source']}`")
    for mode, rows in report["student_latency"].items():
        lines.append(f"\n### {mode}\n")
        for row in rows:
            lines.append(
                f"- batch={row['batch_size']}, input={row['input_shape']}, "
                f"mean={row['mean_ms']:.4f} ms/batch, median={row['median_ms']:.4f} ms/batch, "
                f"mean/sample={row['latency_per_sample_mean_ms']:.4f} ms, "
                f"throughput={row['throughput_samples_per_s_mean']:.2f} samples/s"
            )
    lines.append("\n## Teacher/PCCC timing\n")
    lines.append("- Segment-level core: `" + json.dumps(report["teacher_core_latency"], ensure_ascii=False) + "`")
    lines.append("- File-level full teacher: `" + json.dumps(report["teacher_file_latency"], ensure_ascii=False) + "`")
    lines.append("\n## Interpretation rule\n")
    lines.append("- Cite only comparisons obtained under the same hardware, same input length, same precision, and clearly specified batch size.")
    lines.append("- If the student is slower on CPU, the paper should not claim speed superiority. It can still claim a deployment-oriented surrogate model only after optimized inference is demonstrated.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Benchmark device.")
    parser.add_argument("--length", type=int, default=2048, help="Required waveform length. Paper/data package uses 2048.")
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1, help="CPU intra-op threads. Use 1 for reproducible single-sample latency.")
    parser.add_argument("--batch-sizes", default="1,8,32", help="Comma-separated batch sizes.")
    parser.add_argument("--out-dir", default="results/paper_revision_latency")
    parser.add_argument("--teacher-repeat", type=int, default=200)
    args = parser.parse_args()

    root = repo_root_from_script()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)

    if device.type == "cpu":
        torch.set_num_threads(max(1, int(args.threads)))
        torch.set_num_interop_threads(1)

    model, model_cfg, ckpt_path = load_model(root, device)
    x_cpu, source = load_dataset_example(root, args.length, args.channels)
    x_device = x_cpu.to(device)

    # Check that model actually accepts the claimed input shape.
    with torch.inference_mode():
        y = model(x_device)
    if list(y.shape) != [1, 1]:
        raise RuntimeError(f"Unexpected model output shape: {list(y.shape)}")

    estimated_macs = estimate_macs_conv_linear(model, x_device)
    num_params = count_parameters(model)

    batch_sizes = [int(s.strip()) for s in args.batch_sizes.split(",") if s.strip()]
    student_latency: Dict[str, List[Dict[str, Any]]] = {"pytorch_eager": []}
    for bs in batch_sizes:
        student_latency["pytorch_eager"].append(
            bench_torch_model(model, x_cpu, device, args.warmup, args.repeat, bs)
        )

    traced = try_torchscript(model, x_cpu, device)
    if traced is not None:
        student_latency["torchscript_frozen"] = []
        for bs in batch_sizes:
            student_latency["torchscript_frozen"].append(
                bench_torch_model(traced, x_cpu, device, args.warmup, args.repeat, bs)
            )

    teacher_core = bench_teacher_core_on_segment(root, args.length, max(10, args.warmup // 5), args.teacher_repeat)
    teacher_file = bench_teacher_file_level(root, warmup_files=2, repeat_files=20)

    env = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NA",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "precision": "FP32",
    }

    report: Dict[str, Any] = {
        "status": "traceable_benchmark_v2",
        "repository_root": str(root),
        "checkpoint": str(ckpt_path),
        "model_config": model_cfg,
        "input_source": source,
        "required_input_shape_batch1": [1, args.channels, args.length],
        "environment": env,
        "student_model": {
            "num_parameters": num_params,
            "estimated_macs_batch1": estimated_macs,
            "estimated_macs_batch1_million": estimated_macs / 1e6,
        },
        "student_latency": student_latency,
        "teacher_core_latency": teacher_core,
        "teacher_file_latency": teacher_file,
        "warnings": [
            "Do not cite speedup unless teacher and student are measured on the same hardware and same input/data protocol.",
            "Do not cite microsecond-level latency unless the measured report actually supports it.",
            "The paper/data package requires 2 x 2048 input; [1,2,1024] timing is not valid for the current manuscript.",
        ],
    }

    json_path = out_dir / "fair_latency_report_v2.json"
    md_path = out_dir / "fair_latency_report_v2.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, md_path)

    print("\nSaved:")
    print(json_path)
    print(md_path)
    print("\nKey student result:")
    first = student_latency["pytorch_eager"][0]
    print(f"input={first['input_shape']}, batch={first['batch_size']}, mean/sample={first['latency_per_sample_mean_ms']:.4f} ms")
    if teacher_core.get("available"):
        print(f"teacher core mean={teacher_core['mean_ms']:.4f} ms/segment")
    else:
        print(f"teacher core unavailable: {teacher_core.get('reason')}")


if __name__ == "__main__":
    main()
