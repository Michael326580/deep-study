#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark candidate lightweight student networks on the same input protocol as the manuscript.
This script is for latency screening only; it does not train or report accuracy.

Run from repo root:
    python scripts/paper_revision/benchmark_candidate_fast_students.py --device cpu --threads 1 --repeat 1000 --warmup 100
"""
from __future__ import annotations
import argparse, json, platform, sys, time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn


def find_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "data").exists() and (parent / "results").exists():
            return parent
    return p.parents[2]

class DSConvBlock(nn.Module):
    def __init__(self, in_ch:int, out_ch:int, stride:int=1, k:int=7):
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
    def forward(self, x):
        return self.net(x)

class TinyDWStudent(nn.Module):
    """Very small depthwise-separable 1D-CNN for deployment-oriented ablation."""
    def __init__(self, width:int=8, depth:int=4, in_channels:int=2):
        super().__init__()
        layers = [nn.Conv1d(in_channels, width, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm1d(width), nn.ReLU(inplace=True)]
        ch = width
        for i in range(depth):
            out = width * min(2 ** (i // 2), 4)
            layers.append(DSConvBlock(ch, out, stride=2 if i in {0, 1, 2} else 1, k=7))
            ch = out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(ch, 32), nn.ReLU(inplace=True), nn.Linear(32, 1))
    def forward(self, x):
        return self.head(self.features(x))

class SmallPlainCNN(nn.Module):
    """Plain strided Conv1D without SE attention/residual overhead."""
    def __init__(self, width:int=16, in_channels:int=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, width, 9, stride=2, padding=4, bias=False), nn.BatchNorm1d(width), nn.ReLU(inplace=True),
            nn.Conv1d(width, width*2, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(width*2), nn.ReLU(inplace=True),
            nn.Conv1d(width*2, width*4, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(width*4), nn.ReLU(inplace=True),
            nn.Conv1d(width*4, width*4, 3, stride=2, padding=1, bias=False), nn.BatchNorm1d(width*4), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(width*4, 1)
        )
    def forward(self, x):
        return self.net(x)


def count_params(m):
    return int(sum(p.numel() for p in m.parameters()))

def estimate_macs(model, x):
    macs=0; hooks=[]
    def hconv(m, inp, out):
        nonlocal macs
        b, cout, lout = out.shape
        macs += int(b * cout * lout * (m.in_channels // m.groups) * m.kernel_size[0])
    def hlin(m, inp, out):
        nonlocal macs
        b = out.shape[0] if out.ndim else 1
        macs += int(b * m.in_features * m.out_features)
    for m in model.modules():
        if isinstance(m, nn.Conv1d): hooks.append(m.register_forward_hook(hconv))
        elif isinstance(m, nn.Linear): hooks.append(m.register_forward_hook(hlin))
    with torch.inference_mode(): model(x)
    for h in hooks: h.remove()
    return int(macs)

def bench(model, x, warmup, repeat, device):
    model.eval().to(device); x=x.to(device)
    with torch.inference_mode():
        for _ in range(warmup): model(x)
        if device.type == 'cuda': torch.cuda.synchronize()
        times=[]
        for _ in range(repeat):
            t0=time.perf_counter_ns(); model(x)
            if device.type == 'cuda': torch.cuda.synchronize()
            t1=time.perf_counter_ns(); times.append((t1-t0)/1e6)
    arr=np.asarray(times)
    return dict(mean_ms=float(arr.mean()), median_ms=float(np.median(arr)), p95_ms=float(np.percentile(arr,95)), min_ms=float(arr.min()), max_ms=float(arr.max()))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--length', type=int, default=2048)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--repeat', type=int, default=1000)
    ap.add_argument('--out-dir', default='results/paper_revision_latency')
    args=ap.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    device=torch.device(args.device)
    if device.type=='cpu':
        torch.set_num_threads(max(1,args.threads)); torch.set_num_interop_threads(1)
    root=find_root(); out_dir=root/args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    x=torch.randn(1,2,args.length)
    candidates={
        'tiny_dw_w8_d3': TinyDWStudent(width=8, depth=3),
        'tiny_dw_w8_d4': TinyDWStudent(width=8, depth=4),
        'tiny_dw_w16_d4': TinyDWStudent(width=16, depth=4),
        'plain_cnn_w8': SmallPlainCNN(width=8),
        'plain_cnn_w16': SmallPlainCNN(width=16),
    }
    rows=[]
    for name,m in candidates.items():
        macs=estimate_macs(m, x)
        stats=bench(m, x, args.warmup, args.repeat, device)
        rows.append({'name':name,'params':count_params(m),'macs':macs,'macs_million':macs/1e6, **stats})
        print(name, rows[-1])
    report={'purpose':'latency screening for lightweight student candidates; not accuracy evidence', 'input_shape':[1,2,args.length], 'env':{'python':sys.version, 'platform':platform.platform(), 'torch':torch.__version__, 'device':str(device), 'threads':torch.get_num_threads()}, 'rows':rows}
    (out_dir/'candidate_fast_student_latency.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    md=['# Candidate fast student latency screening','',f'- input: `[1, 2, {args.length}]`','- These are untrained models. Use this only to select architectures for retraining.','']
    md.append('| Candidate | Params | MACs(M) | Mean ms | Median ms | P95 ms |')
    md.append('|---|---:|---:|---:|---:|---:|')
    for r in rows:
        md.append(f"| {r['name']} | {r['params']} | {r['macs_million']:.3f} | {r['mean_ms']:.4f} | {r['median_ms']:.4f} | {r['p95_ms']:.4f} |")
    (out_dir/'candidate_fast_student_latency.md').write_text('\n'.join(md),encoding='utf-8')
if __name__=='__main__': main()
