# -*- coding: utf-8 -*-
"""
check_distill.py
用于校验 dataset_distill.npz 与 dataset_distill_meta.csv 的完整性与一致性。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate distillation dataset files.")
    parser.add_argument("--npz", type=Path, default=Path("dataset_distill.npz"), help="NPZ 文件路径")
    parser.add_argument("--meta", type=Path, default=Path("dataset_distill_meta.csv"), help="META CSV 文件路径")
    parser.add_argument("--seq-len", type=int, default=2048, help="期望序列长度")
    parser.add_argument("--channels", type=int, default=2, help="期望通道数")
    parser.add_argument("--x-min-warn", type=float, default=-0.1, help="X 最小值告警阈值")
    parser.add_argument("--x-max-warn", type=float, default=5.0, help="X 最大值告警阈值")
    return parser.parse_args()


def _load_npz_robust(npz_path: Path):
    """优先禁用 pickle；若遇到 object array 再回退允许 pickle。"""

    def _materialize(npz_obj):
        # np.load 返回的是懒加载对象，访问具体键时才可能触发 object-array 错误。
        # 这里主动读取全部键，确保在本函数内完成异常处理。
        data = {}
        for k in npz_obj.files:
            data[k] = npz_obj[k]
        return data

    try:
        with np.load(npz_path, allow_pickle=False) as npz_obj:
            data = _materialize(npz_obj)
        return data, False
    except ValueError as e:
        msg = str(e)
        if "Object arrays cannot be loaded" not in msg:
            raise
        with np.load(npz_path, allow_pickle=True) as npz_obj:
            data = _materialize(npz_obj)
        return data, True


def main() -> int:
    args = parse_args()

    npz_path = args.npz
    meta_path = args.meta

    errors = []
    warnings = []

    print("==== Distill Dataset Full Check ====")
    print(f"[INFO] NPZ path : {npz_path}")
    print(f"[INFO] META path: {meta_path}")

    if not npz_path.exists():
        errors.append(f"缺少文件: {npz_path}")
    if not meta_path.exists():
        errors.append(f"缺少文件: {meta_path}")

    if errors:
        print("\n❌ FAIL")
        for e in errors:
            print(" -", e)
        return 1

    try:
        distill, used_pickle_fallback = _load_npz_robust(npz_path)
    except Exception as e:
        print("\n❌ FAIL")
        print(f" - 无法读取 NPZ: {e}")
        return 1

    if used_pickle_fallback:
        warnings.append(
            "NPZ 包含 object 数组（通常是 filenames），已回退为 allow_pickle=True 读取。"
            "建议重新生成 NPZ 并将 filenames 存为固定字符串 dtype。"
        )

    required_npz_keys = {"X", "y"}
    missing_npz_keys = required_npz_keys - set(distill.keys())
    if missing_npz_keys:
        errors.append(f"NPZ 缺少键: {sorted(missing_npz_keys)}；当前键: {list(distill.keys())}")

    if errors:
        print("\n❌ FAIL")
        for e in errors:
            print(" -", e)
        return 1

    X = distill["X"]
    y = distill["y"]

    print(f"[INFO] NPZ keys: {list(distill.keys())}")
    print(f"[INFO] X shape={X.shape}, dtype={X.dtype}")
    print(f"[INFO] y shape={y.shape}, dtype={y.dtype}")

    if X.ndim != 3:
        errors.append(f"X 维度应为 3，当前为 {X.ndim}")
    else:
        if X.shape[1] != args.seq_len or X.shape[2] != args.channels:
            warnings.append(f"X 形状非预期 (N,{args.seq_len},{args.channels})，当前 {X.shape}")

    if y.ndim != 1:
        errors.append(f"y 维度应为 1，当前为 {y.ndim}")

    if len(X) != len(y):
        errors.append(f"X 与 y 样本数不一致: len(X)={len(X)}, len(y)={len(y)}")

    if not np.isfinite(X).all():
        errors.append("X 存在 NaN/Inf")
    if not np.isfinite(y).all():
        errors.append("y 存在 NaN/Inf")

    if len(y) > 0:
        y_min, y_max = float(np.min(y)), float(np.max(y))
        print(f"[INFO] y range=[{y_min:.6f}, {y_max:.6f}]")
    else:
        errors.append("y 为空")

    if len(X) > 0:
        x_min, x_max = float(np.min(X)), float(np.max(X))
        print(f"[INFO] X range=[{x_min:.6f}, {x_max:.6f}]")
        if x_min < args.x_min_warn or x_max > args.x_max_warn:
            warnings.append(
                f"X 数值范围可能异常: [{x_min:.6f}, {x_max:.6f}] "
                f"(告警阈值 [{args.x_min_warn}, {args.x_max_warn}])"
            )

        zero_sample_ratio = float((np.abs(X).sum(axis=(1, 2)) == 0).mean())
        print(f"[INFO] 全零样本占比: {zero_sample_ratio:.4%}")
        if zero_sample_ratio > 0.01:
            warnings.append(f"全零样本占比偏高: {zero_sample_ratio:.4%}")

    if "segment_lens" in distill:
        segment_lens = distill["segment_lens"]
        print(
            f"[INFO] segment_lens shape={segment_lens.shape}, "
            f"min={int(segment_lens.min())}, max={int(segment_lens.max())}"
        )
        if len(segment_lens) != len(X):
            errors.append(f"segment_lens 长度与 X 不一致: {len(segment_lens)} vs {len(X)}")
        if (segment_lens <= 0).any() or (segment_lens > args.seq_len).any():
            errors.append(f"segment_lens 存在非法值（应在 1..{args.seq_len}）")
    else:
        warnings.append("NPZ 中没有 segment_lens（不是错误，但建议保留）")

    if "filenames" in distill:
        filenames = distill["filenames"]
        print(f"[INFO] filenames shape={filenames.shape}")
        if len(filenames) != len(X):
            errors.append(f"filenames 长度与 X 不一致: {len(filenames)} vs {len(X)}")
        else:
            unique_file_count = len(set(map(str, filenames.tolist())))
            print(f"[INFO] NPZ 中唯一文件数: {unique_file_count}")
    else:
        warnings.append("NPZ 中没有 filenames（不是错误，但建议保留）")

    try:
        meta_df = pd.read_csv(meta_path)
    except Exception as e:
        errors.append(f"META 无法读取: {e}")
        meta_df = None

    if meta_df is not None:
        print(f"[INFO] META rows={len(meta_df)}, cols={list(meta_df.columns)}")

        required_meta_cols = {"Filename", "Teacher_Width", "SegmentStart", "SegmentEnd", "SegmentLen"}
        missing_meta_cols = required_meta_cols - set(meta_df.columns)
        if missing_meta_cols:
            errors.append(f"META 缺少列: {sorted(missing_meta_cols)}")

        if len(meta_df) != len(X):
            errors.append(f"META 行数与 X 样本数不一致: {len(meta_df)} vs {len(X)}")

        if "SegmentLen" in meta_df.columns:
            seglen = pd.to_numeric(meta_df["SegmentLen"], errors="coerce")
            bad_seglen = ((seglen <= 0) | (seglen > args.seq_len) | (~np.isfinite(seglen))).sum()
            if bad_seglen > 0:
                errors.append(f"META 中 SegmentLen 非法行数: {int(bad_seglen)}")

        if "Teacher_Width" in meta_df.columns:
            tw = pd.to_numeric(meta_df["Teacher_Width"], errors="coerce")
            if not np.isfinite(tw.to_numpy()).all():
                errors.append("META.Teacher_Width 存在 NaN/Inf 或非数值")

        if "SegmentStart" in meta_df.columns and "SegmentEnd" in meta_df.columns:
            ss = pd.to_numeric(meta_df["SegmentStart"], errors="coerce")
            se = pd.to_numeric(meta_df["SegmentEnd"], errors="coerce")
            bad_order = (se <= ss).sum()
            bad_nan = ((~np.isfinite(ss)) | (~np.isfinite(se))).sum()
            if bad_order > 0:
                errors.append(f"META 中 SegmentEnd <= SegmentStart 的行数: {int(bad_order)}")
            if bad_nan > 0:
                errors.append(f"META 中 SegmentStart/SegmentEnd 非法行数: {int(bad_nan)}")

        if "Filename" in meta_df.columns:
            uniq_meta_files = meta_df["Filename"].astype(str).nunique()
            print(f"[INFO] META 中唯一文件数: {uniq_meta_files}")

    print("\n==== Result ====")
    if errors:
        print("❌ FAIL")
        for e in errors:
            print(" -", e)
    else:
        print("✅ PASS（关键结构与数值检查通过）")

    if warnings:
        print("\n⚠️ WARNINGS")
        for w in warnings:
            print(" -", w)

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())