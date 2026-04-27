#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recover train_summary.json and summary table after train_fast_student_ablation.py
crashes at torch.load under PyTorch >= 2.6.

Run from repository root:
python scripts/paper_revision/recover_fast_student_summary.py --models plain_cnn_w16 --device cpu --threads 1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch


def find_root() -> Path:
    p = Path.cwd().resolve()
    for parent in [p, *p.parents]:
        if (parent / 'data').exists() and (parent / 'results').exists():
            return parent
    return p


def load_training_module(root):
    import sys
    import importlib.util

    path = root / "scripts" / "paper_revision" / "train_fast_student_ablation.py"
    module_name = "train_fast_student_ablation_recover"

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load training module from {path}")

    mod = importlib.util.module_from_spec(spec)

    # 关键修复：dataclass 在 Python 3.13 下需要先注册到 sys.modules
    sys.modules[module_name] = mod

    spec.loader.exec_module(mod)
    return mod


def read_best_metrics(out_dir: Path) -> Dict[str, Any]:
    hist_path = out_dir / 'train_history.csv'
    if not hist_path.exists():
        raise FileNotFoundError(f'Missing {hist_path}')
    hist = pd.read_csv(hist_path)
    best_row = hist.loc[hist['pos_mae'].astype(float).idxmin()].to_dict()
    return {
        'best_epoch': int(best_row.get('epoch', -1)),
        'train_loss': float(best_row.get('train_loss', np.nan)),
        'val_loss': float(best_row.get('val_loss', np.nan)),
        'seg_width_mae': float(best_row.get('seg_width_mae', np.nan)),
        'file_width_mae': float(best_row.get('file_width_mae', np.nan)),
        'pos_mae': float(best_row.get('pos_mae', np.nan)),
        'pos_rmse': float(best_row.get('pos_rmse', np.nan)),
        'pos_maxae': float(best_row.get('pos_maxae', np.nan)),
    }


def recover_one(root: Path, mod, model_name: str, args) -> Dict[str, Any]:
    out_dir = root / args.out_base / model_name
    ckpt_path = out_dir / f'best_{model_name}.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Missing checkpoint: {ckpt_path}')

    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model = mod.make_model(model_name).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    params = mod.count_params(model)
    macs = mod.estimate_macs(model.cpu(), torch.randn(1, 2, args.length))
    model = model.to(args.device)
    latency = mod.benchmark_model(model, device=torch.device(args.device), length=args.length,
                                  warmup=args.warmup, repeat=args.repeat)
    best = read_best_metrics(out_dir)
    summary = {
        'model_name': model_name,
        'params': int(params),
        'macs': int(macs),
        'macs_million': float(macs / 1e6),
        **best,
        **latency,
        'checkpoint': str(ckpt_path),
        'out_dir': str(out_dir),
        'recovered_after_torch26_load_error': True,
        'torch_version': torch.__version__,
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'device': args.device,
        'threads': args.threads,
        'input_shape': [1, 2, args.length],
        'notes': 'Recovered with torch.load(..., weights_only=False) from a locally generated checkpoint.',
    }
    with open(out_dir / 'train_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def write_reports(root: Path, summaries: List[Dict[str, Any]], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summaries)
    csv_path = report_dir / 'fast_student_ablation_summary.csv'
    md_path = report_dir / 'fast_student_ablation_summary.md'
    df.to_csv(csv_path, index=False)

    lines = ['# Fast student ablation summary', '']
    lines.append('| model | params | MACs(M) | pos MAE(mm) | pos RMSE(mm) | max AE(mm) | mean latency(ms) |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|')
    for s in summaries:
        lines.append(
            f"| {s['model_name']} | {s['params']} | {s['macs_million']:.3f} | "
            f"{s['pos_mae']:.6f} | {s['pos_rmse']:.6f} | {s['pos_maxae']:.6f} | {s['mean_ms']:.6f} |"
        )
    lines.append('')
    lines.append('Recovered after PyTorch >=2.6 weights_only loading error if applicable.')
    md_path.write_text('\n'.join(lines), encoding='utf-8')
    print('Saved:', csv_path)
    print('Saved:', md_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['plain_cnn_w16'])
    ap.add_argument('--out-base', default='results/models/runs/fast_student_ablation')
    ap.add_argument('--report-dir', default='results/paper_revision_latency')
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--length', type=int, default=2048)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--repeat', type=int, default=1000)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    root = find_root()
    mod = load_training_module(root)
    summaries = [recover_one(root, mod, m, args) for m in args.models]
    write_reports(root, summaries, root / args.report_dir)
    for s in summaries:
        print(s['model_name'], 'pos_mae=', s['pos_mae'], 'mean_ms=', s['mean_ms'])


if __name__ == '__main__':
    main()
