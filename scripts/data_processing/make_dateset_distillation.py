import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# 读取老师数据去做数据集：对应 fixed_results.csv 里的高精度结果，做成 dataset_distill.npz
# ================= 默认配置（可由命令行参数覆盖） =================
DEFAULT_RAW_DATA_FOLDER = "data/raw/1.22"
DEFAULT_LABEL_CSV = "data/processed/fixed_results.csv"
DEFAULT_SAVE_PATH = "data/processed/dataset_distill.npz"
DEFAULT_META_PATH = "data/processed/dataset_distill_meta.csv"
DEFAULT_SEQ_LEN = 2048
DEFAULT_MIN_SEG_LEN = 2000
DEFAULT_SCALE = 16383.0
DEFAULT_WAVEFORM_REGEX = r"^-?\d+\.csv$"
# =============================================================


def _load_teacher_table(label_csv: Path) -> pd.DataFrame:
    """加载 teacher 结果并做基本校验。"""
    if not label_csv.exists():
        raise FileNotFoundError(f"找不到标签文件: {label_csv}")

    teacher_df = pd.read_csv(label_csv)
    required_cols = {"Filename", "FFT_Width"}
    missing_cols = required_cols.difference(teacher_df.columns)
    if missing_cols:
        raise KeyError(
            f"标签文件缺少必要字段: {sorted(missing_cols)}; "
            f"当前字段: {teacher_df.columns.tolist()}"
        )

    # 保留 Filename + FFT_Width + Position_mm(若存在)，并统一类型。
    keep_cols = ["Filename", "FFT_Width"]
    if "Position_mm" in teacher_df.columns:
        keep_cols.append("Position_mm")
    teacher_df = teacher_df[keep_cols].copy()

    teacher_df["Filename"] = teacher_df["Filename"].astype(str)
    teacher_df["FFT_Width"] = pd.to_numeric(teacher_df["FFT_Width"], errors="coerce")

    before = len(teacher_df)
    teacher_df = teacher_df[np.isfinite(teacher_df["FFT_Width"])].copy()
    dropped_bad = before - len(teacher_df)

    # 若同一 Filename 重复，保留最后一条（常见于“修正后覆盖”场景）。
    dup_count = int(teacher_df["Filename"].duplicated(keep=False).sum())
    if dup_count > 0:
        teacher_df = teacher_df.drop_duplicates(subset=["Filename"], keep="last").copy()

    print(f"📖 已加载 teacher 标签: {label_csv}")
    print(f"   - 记录数(去重后): {len(teacher_df)}")
    print(f"   - 丢弃无效 FFT_Width 行: {dropped_bad}")
    print(f"   - 去重涉及行数: {dup_count}")

    return teacher_df


def _load_waveform_csv(path: Path) -> pd.DataFrame:
    """兼容不同编码/解析器读取原始波形 csv。"""
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine="python")


def make_distilled_dataset(
    raw_data_folder: Path,
    label_csv: Path,
    save_path: Path,
    meta_path: Path,
    seq_len: int,
    min_seg_len: int,
    scale: float,
    waveform_regex: str,
) -> None:
    # 1) 加载“老师”答案（distillation target）
    teacher_df = _load_teacher_table(label_csv)
    label_map = dict(zip(teacher_df["Filename"], teacher_df["FFT_Width"]))
    pos_map = (
        dict(zip(teacher_df["Filename"], teacher_df["Position_mm"]))
        if "Position_mm" in teacher_df.columns
        else {}
    )

    if not raw_data_folder.exists():
        raise FileNotFoundError(f"原始波形目录不存在: {raw_data_folder}")

    # 2) 扫描原始波形并切段（优先使用“teacher 文件名交集”，避免扫到无关 CSV）
    filename_pattern = re.compile(waveform_regex) if waveform_regex else None
    excluded_names = {label_csv.name, save_path.name, meta_path.name}

    all_csv_files = sorted(
        [
            f
            for f in os.listdir(raw_data_folder)
            if f.endswith(".csv") and f not in excluded_names
        ]
    )

    all_csv_set = set(all_csv_files)
    teacher_files_exist = sorted([fname for fname in label_map.keys() if fname in all_csv_set])

    fallback_mode = "none"
    if filename_pattern is not None:
        files = [f for f in teacher_files_exist if filename_pattern.match(f) is not None]
        if len(files) == 0 and len(teacher_files_exist) > 0:
            # 正则太严格时，回退到 teacher 交集（保证仍只扫有标签的文件）。
            files = list(teacher_files_exist)
            fallback_mode = "teacher_intersection_without_regex"
    else:
        files = list(teacher_files_exist)

    if len(files) == 0 and len(all_csv_files) > 0 and len(teacher_files_exist) == 0:
        sample_teacher = list(label_map.keys())[:5]
        sample_csv = all_csv_files[:5]
        raise RuntimeError(
            "当前目录里没有任何与标签表 Filename 对应的原始波形文件。\n"
            f"- 标签文件示例 Filename: {sample_teacher}\n"
            f"- 目录内 CSV 示例: {sample_csv}\n"
            "请把原始波形 CSV 放到 --raw-data-folder 指定目录，或检查 Filename 是否一致。"
        )

    X_list, y_list = [], []
    meta_rows = []
    skipped_no_label = 0
    skipped_bad_wave = 0
    diagnose = defaultdict(int)
    diagnose_examples = defaultdict(list)

    def _add_example(key: str, value: str, limit: int = 5) -> None:
        if len(diagnose_examples[key]) < limit:
            diagnose_examples[key].append(value)

    print("🚀 开始制作蒸馏数据集...")
    print(f"   - 原始目录: {raw_data_folder}")
    print(f"   - 目录内 csv 数: {len(all_csv_files)}")
    print(f"   - 波形文件数: {len(files)}")
    print(f"   - 文件名过滤: {waveform_regex}")
    if fallback_mode != "none":
        print(f"   - ⚠️ 已启用回退模式: {fallback_mode}（你的 --waveform-regex 可能过严）")
    print(f"   - seq_len={seq_len}, min_seg_len={min_seg_len}, scale={scale}")

    for i, fname in enumerate(files):
        diagnose["files_scanned"] += 1

        if fname not in label_map:
            skipped_no_label += 1
            diagnose["not_in_label_map"] += 1
            _add_example("not_in_label_map", fname)
            continue

        target_width = float(label_map[fname])
        if not np.isfinite(target_width):
            skipped_no_label += 1
            diagnose["non_finite_label"] += 1
            _add_example("non_finite_label", fname)
            continue

        path = raw_data_folder / fname
        try:
            df = _load_waveform_csv(path)
            if "Channel 1" not in df.columns or "Channel 2" not in df.columns:
                skipped_bad_wave += 1
                diagnose["missing_channel_cols"] += 1
                _add_example("missing_channel_cols", fname)
                continue

            c1 = df["Channel 1"].to_numpy()
            c2 = df["Channel 2"].to_numpy()

            starts = np.where((c1 == 0) & (c2 == 0))[0]
            ends = np.where((c1 == 0) & (c2 == 61680))[0]

            if len(starts) == 0:
                diagnose["no_start_marker"] += 1
                _add_example("no_start_marker", fname)
                continue
            if len(ends) == 0:
                diagnose["no_end_marker"] += 1
                _add_example("no_end_marker", fname)
                continue

            file_has_valid_segment = False
            no_end_after_start_count = 0
            short_seg_count = 0

            for s_idx in starts:
                possible_ends = ends[ends > s_idx]
                if len(possible_ends) == 0:
                    no_end_after_start_count += 1
                    continue

                e_idx = int(possible_ends[0])
                seg_len = e_idx - s_idx
                if seg_len <= min_seg_len:
                    short_seg_count += 1
                    continue

                seg_c1 = c1[s_idx + 1 : e_idx]
                seg_c2 = c2[s_idx + 1 : e_idx]

                # 固定长度: 截断 + 右侧零填充，再按 ADC 满量程做缩放
                sample = np.zeros((seq_len, 2), dtype=np.float32)
                curr_len = min(len(seg_c1), seq_len)
                sample[:curr_len, 0] = seg_c1[:curr_len]
                sample[:curr_len, 1] = seg_c2[:curr_len]
                sample /= np.float32(scale)

                X_list.append(sample)
                y_list.append(np.float32(target_width))
                file_has_valid_segment = True
                meta_rows.append(
                    {
                        "Filename": fname,
                        "Teacher_Width": np.float32(target_width),
                        "Position_mm": pos_map.get(fname, np.nan),
                        "SegmentStart": int(s_idx),
                        "SegmentEnd": int(e_idx),
                        "SegmentLen": int(curr_len),
                    }
                )

            if file_has_valid_segment:
                diagnose["file_has_valid_segment"] += 1
            else:
                if no_end_after_start_count > 0:
                    diagnose["no_end_after_start"] += 1
                    _add_example("no_end_after_start", fname)
                if short_seg_count > 0:
                    diagnose["all_segments_too_short"] += 1
                    _add_example("all_segments_too_short", fname)
                if no_end_after_start_count == 0 and short_seg_count == 0:
                    diagnose["markers_found_but_no_valid_segment"] += 1
                    _add_example("markers_found_but_no_valid_segment", fname)

        except Exception as e:
            skipped_bad_wave += 1
            diagnose["read_or_parse_error"] += 1
            _add_example("read_or_parse_error", f"{fname}: {e}")
            print(f"⚠️ 处理文件失败 {fname}: {e}")

        if (i + 1) % 100 == 0:
            print(f"进度: {i + 1}/{len(files)}")

    if len(X_list) == 0:
        diagnose_msg = [
            "没有生成任何有效样本，关键诊断信息如下：",
            f"- 扫描文件数: {diagnose['files_scanned']}",
            f"- 不在标签表中的文件: {diagnose['not_in_label_map']}",
            f"- 标签非数值/非有限: {diagnose['non_finite_label']}",
            f"- 缺少 Channel 列: {diagnose['missing_channel_cols']}",
            f"- 无起始标记(0,0): {diagnose['no_start_marker']}",
            f"- 无结束标记(0,61680): {diagnose['no_end_marker']}",
            f"- 起始后找不到结束标记: {diagnose['no_end_after_start']}",
            f"- 全部片段过短(<= min_seg_len): {diagnose['all_segments_too_short']}",
            f"- 有标记但仍无有效片段: {diagnose['markers_found_but_no_valid_segment']}",
            f"- 读取/解析异常: {diagnose['read_or_parse_error']}",
            "",
            "示例文件：",
        ]
        for key in [
            "not_in_label_map",
            "missing_channel_cols",
            "no_start_marker",
            "no_end_marker",
            "no_end_after_start",
            "all_segments_too_short",
            "read_or_parse_error",
        ]:
            examples = diagnose_examples.get(key, [])
            if examples:
                diagnose_msg.append(f"  * {key}: {examples}")
        raise RuntimeError("\n".join(diagnose_msg))

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    meta_df = pd.DataFrame(meta_rows)

    np.savez(
        save_path,
        X=X,
        y=y,
        filenames=meta_df["Filename"].astype(str).to_numpy(),
        segment_lens=meta_df["SegmentLen"].to_numpy(dtype=np.int32),
    )
    meta_df.to_csv(meta_path, index=False)

    print("\n✅ 蒸馏数据集制作完成")
    print(f"   - 样本数: {len(X)}")
    print(f"   - X shape: {X.shape}, dtype={X.dtype}")
    print(f"   - y shape: {y.shape}, range=[{y.min():.6f}, {y.max():.6f}]")
    print(f"   - 跳过(无标签): {skipped_no_label}")
    print(f"   - 跳过(坏波形/异常): {skipped_bad_wave}")
    print(f"   - 含有效片段的文件数: {diagnose['file_has_valid_segment']}")
    print(f"💾 NPZ:  {save_path}")
    print(f"🧾 META: {meta_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build distillation dataset from teacher CSV + raw waveform CSVs")
    parser.add_argument("--raw-data-folder", type=Path, default=Path(DEFAULT_RAW_DATA_FOLDER))
    parser.add_argument("--label-csv", type=Path, default=Path(DEFAULT_LABEL_CSV))
    parser.add_argument("--save-path", type=Path, default=Path(DEFAULT_SAVE_PATH))
    parser.add_argument("--meta-path", type=Path, default=Path(DEFAULT_META_PATH))
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--min-seg-len", type=int, default=DEFAULT_MIN_SEG_LEN)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument(
        "--waveform-regex",
        type=str,
        default=DEFAULT_WAVEFORM_REGEX,
        help="仅处理匹配该正则的 csv 文件名（默认匹配 -449.csv / 123.csv 这类原始波形文件）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_distilled_dataset(
        raw_data_folder=args.raw_data_folder,
        label_csv=args.label_csv,
        save_path=args.save_path,
        meta_path=args.meta_path,
        seq_len=args.seq_len,
        min_seg_len=args.min_seg_len,
        scale=args.scale,
        waveform_regex=args.waveform_regex,
    )