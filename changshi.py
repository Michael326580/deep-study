# =========================================================
# [脚本说明]
# 用途：按指定位置查看原始双通道波形（绘图前去除标记尖刺）。
# 输入：1. `pos_to_filename` 将物理位置映射成文件名。
# 输出：2. `remove_markers_for_plot` 去除 `(0,0)` 和 `(0,61680)` 及邻域，可选插值。
# =========================================================

# =========================================================
# [????]
# ????????????????????????
# ?????????????????????
# ???????????????
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# 1) ⚙️ 用户设置区域（只改这里）
# ==============================================================================
TARGET_POS = 192.4

# DATA_FOLDER = r'D:\桌面\learning-notes\069date\069date3'  # 旧数据集
DATA_FOLDER = r'D:\桌面\new learning\069date\1.22'        # 新数据集

# "OLD" = 旧数据集 (180.0mm -> -300)
# "NEW" = 新数据集 (180.0mm -> -449)
DATASET_TYPE = "NEW"

# 绘图清洗参数（去除触发标记尖刺）
PAD = 2                 # 标记点左右扩展去除点数（建议 2~5）
DO_INTERPOLATE = True   # True: 插值填回，曲线更顺；False: 断开显示更真实

# 饱和值阈值（你图里 Channel2 有 61680，说明可能是 16bit 上限标记；仅做提示）
SAT_THRESHOLD = 16300   # 你原脚本用 16300 作为饱和提示阈值，可保持不变


# ==============================================================================
# 2) 全局显示设置（中文）
# ==============================================================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 3) 工具函数：位置 -> 文件名
# ==============================================================================
def pos_to_filename(target_pos, dataset_type):
    steps = int(round((target_pos - 180.0) * 10))  # 0.1mm step
    if dataset_type == "OLD":
        file_num = -300 + steps
    elif dataset_type == "NEW":
        file_num = -449 + steps
    else:
        raise ValueError("DATASET_TYPE 必须是 'OLD' 或 'NEW'")
    return f"{file_num}.csv"


# ==============================================================================
# 4) 工具函数：仅用于绘图的“去标记点/平滑”
# ==============================================================================
def remove_markers_for_plot(c1, c2, pad=2, do_interpolate=True):
    """
    仅用于【绘图展示】的清洗：
    - 标记点：(c1==0 & c2==61680) 或 (c1==0 & c2==0)
    - 对标记点左右扩展 pad 个点一起去掉
    - 用 NaN 断开曲线；可选用线性插值恢复连续观感
    """
    c1 = pd.Series(pd.to_numeric(c1, errors="coerce").astype(float))
    c2 = pd.Series(pd.to_numeric(c2, errors="coerce").astype(float))

    marker_mask = ((c1 == 0) & (c2 == 61680)) | ((c1 == 0) & (c2 == 0))

    if pad > 0:
        idx = np.where(marker_mask.to_numpy())[0]
        dil = np.zeros_like(marker_mask.to_numpy(), dtype=bool)
        n = len(dil)
        for i in idx:
            lo = max(0, i - pad)
            hi = min(n, i + pad + 1)
            dil[lo:hi] = True
        marker_mask = pd.Series(dil)

    c1_clean = c1.mask(marker_mask, np.nan)
    c2_clean = c2.mask(marker_mask, np.nan)

    if do_interpolate:
        # 线性插值，避免出现竖直尖刺
        c1_clean = c1_clean.interpolate(method="linear", limit_direction="both")
        c2_clean = c2_clean.interpolate(method="linear", limit_direction="both")

    return c1_clean.to_numpy(), c2_clean.to_numpy(), marker_mask.to_numpy()


# ==============================================================================
# 5) 主功能：查看某一位置原始波形（已去除 0 / 61680 标记尖刺）
# ==============================================================================
def view_raw_waveform():
    print(f"🔍 正在查找位置: {TARGET_POS:.1f} mm ...")

    filename = pos_to_filename(TARGET_POS, DATASET_TYPE)
    file_path = os.path.join(DATA_FOLDER, filename)

    # 文件存在性检查（兼容可能的双重 .csv）
    if not os.path.exists(file_path):
        if os.path.exists(file_path + ".csv"):
            file_path += ".csv"
            filename += ".csv"
        else:
            print("❌ 错误：找不到对应的文件！")
            print(f"   推算文件名: {filename}")
            print(f"   搜索路径: {file_path}")
            return

    print(f"✅ 找到文件: {filename}")

    # 读取数据
    try:
        try:
            df = pd.read_csv(file_path)
        except:
            df = pd.read_csv(file_path, engine="python")

        if ("Channel 1" not in df.columns) or ("Channel 2" not in df.columns):
            print("❌ 错误：文件中找不到 'Channel 1' 或 'Channel 2' 列")
            return

        c1_raw = pd.to_numeric(df["Channel 1"], errors="coerce")
        c2_raw = pd.to_numeric(df["Channel 2"], errors="coerce")

    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # 仅用于绘图：去掉触发标记尖刺
    c1_plot, c2_plot, marker_mask = remove_markers_for_plot(
        c1_raw, c2_raw, pad=PAD, do_interpolate=DO_INTERPOLATE
    )

    # 统计信息
    max_c1 = np.nanmax(c1_raw.to_numpy(dtype=float))
    max_c2 = np.nanmax(c2_raw.to_numpy(dtype=float))
    n_mark = int(np.sum(marker_mask))

    sat_text1 = " (疑似饱和!)" if max_c1 >= SAT_THRESHOLD else ""
    sat_text2 = " (疑似饱和!)" if max_c2 >= SAT_THRESHOLD else ""

    print(f"🧹 标记点清理：共去除/替换 {n_mark} 个采样点（含 pad={PAD} 扩展），插值={DO_INTERPOLATE}")

    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    fig.suptitle(
        f"原始波形查看（已去除触发标记 0/61680）: {TARGET_POS:.1f} mm\n文件: {filename}",
        fontsize=14, fontweight="bold"
    )

    # Channel 1
    ax1.plot(c1_plot, color="#d62728", linewidth=1.1, label="Channel 1（清理后用于展示）")
    ax1.set_ylabel("光强 (Intensity)", fontsize=12)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.text(
        0.02, 0.90,
        f"Raw Max: {max_c1:.0f}{sat_text1}",
        transform=ax1.transAxes,
        color="red" if max_c1 >= SAT_THRESHOLD else "black",
        fontweight="bold"
    )

    # Channel 2
    ax2.plot(c2_plot, color="#1f77b4", linewidth=1.1, label="Channel 2（清理后用于展示）")
    ax2.set_ylabel("光强 (Intensity)", fontsize=12)
    ax2.set_xlabel("采样点 (Index)", fontsize=12)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.text(
        0.02, 0.90,
        f"Raw Max: {max_c2:.0f}{sat_text2}",
        transform=ax2.transAxes,
        color="red" if max_c2 >= SAT_THRESHOLD else "black",
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    view_raw_waveform()
