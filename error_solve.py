# =========================================================
# [脚本说明]
# 用途：对失败/离群文件做二次“智能修复”，并合并回结果表。
# 输入：1. `calculate_width` 复用互相关 + FFT 的宽度算法。
# 输出：2. `process_single_file_smart_filter` 对单文件分段并按段长中位数做筛选。
# =========================================================

# =========================================================
# [????]
# ??????????????????????
# ???failed_report.csv / outliers_report.csv + ???????
# ??????? fixed_results.csv?
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft
import os
import re

# ==============================================================================
# 0. 参数设置
# ==============================================================================
# 新数据集路径
DATA_FOLDER = r'D:\桌面\new learning\069date\1.22'

# 长度容差：如果某一段的长度偏离中位数超过 20%，就认为是坏的
LENGTH_TOLERANCE = 0.2 

# 输出文件名
OUTPUT_FIXED_CSV = 'fixed_results.csv'

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 1. 核心计算函数 (复用之前的逻辑)
# ==============================================================================
def estimate_period(sig):
    n = len(sig)
    if n == 0: return 50
    yf = np.abs(fft(sig)[:n//2])
    idx = np.argmax(yf[5:]) + 5
    if idx <= 0: return 50 
    return int(n / idx)

def calculate_width(ch1, ch2):
    n = len(ch1)
    if n < 100: return np.nan
    
    # 预处理
    win = 10
    ch1_s = np.convolve(ch1.values, np.ones(win)/win, mode='same')
    ch2_s = np.convolve(ch2.values, np.ones(win)/win, mode='same')
    ch1_c = ch1_s - np.mean(ch1_s)
    ch2_c = ch2_s - np.mean(ch2_s)
    
    # 对齐
    period = estimate_period(ch1_c)
    corr = signal.correlate(ch1_c, ch2_c, mode='full')
    lags = signal.correlation_lags(len(ch1_c), len(ch2_c), mode='full')
    search_radius = int(period * 0.6)
    mask = (lags >= -search_radius) & (lags <= search_radius)
    
    if np.any(mask):
        shift = lags[mask][np.argmin(corr[mask])]
        if shift > 0: sig = ch1_c[shift:] - ch2_c[:-shift]
        elif shift < 0: sig = ch1_c[:shift] - ch2_c[-shift:]
        else: sig = ch1_c - ch2_c
    else:
        sig = ch1_c - ch2_c

    # FFT
    N = len(sig)
    window = np.hanning(N)
    yf_mag = np.abs(fft(sig * window)[:N//2])
    
    start_idx = 5
    if len(yf_mag) <= start_idx: return np.nan
    peak_idx = np.argmax(yf_mag[start_idx:]) + start_idx
    
    # 插值优化
    if 0 < peak_idx < len(yf_mag) - 1:
        y_L, y_C, y_R = yf_mag[peak_idx-1], yf_mag[peak_idx], yf_mag[peak_idx+1]
        denom = y_L - 2*y_C + y_R
        delta = 0.5 * (y_L - y_R) / denom if denom != 0 else 0
        freq = (peak_idx + delta) / N
    else:
        freq = peak_idx / N
        
    return 1.0 / (2.0 * freq) if freq > 0 else np.nan

# ==============================================================================
# 2. 智能分段与清洗逻辑
# ==============================================================================
def process_single_file_smart_filter(file_path):
    try:
        try: df = pd.read_csv(file_path)
        except: df = pd.read_csv(file_path, engine='python')
        
        c1 = pd.to_numeric(df['Channel 1'], errors='coerce').fillna(0)
        c2 = pd.to_numeric(df['Channel 2'], errors='coerce').fillna(0)
        
        # 1. 寻找所有波形段的起止点
        starts = df.index[(c1 == 0) & (c2 == 0)].tolist()
        ends = df.index[(c1 == 0) & (c2 == 61680)].tolist()
        
        segments = []
        
        for s in starts:
            # 找这个起点之后最近的终点
            possible_ends = [e for e in ends if e > s]
            if possible_ends:
                e = possible_ends[0]
                length = e - s
                if length > 50: # 忽略极短的噪点
                    segments.append({
                        'start': s, 'end': e, 'length': length,
                        'c1_data': c1.iloc[s+1:e],
                        'c2_data': c2.iloc[s+1:e]
                    })
        
        if not segments: return np.nan, "无有效段"

        # 2. 【核心】计算中位数长度，剔除异常段
        lengths = [seg['length'] for seg in segments]
        median_len = np.median(lengths)
        
        valid_widths = []
        dropped_count = 0
        
        for seg in segments:
            # 判断长度是否偏差过大 (比如长了20%或者短了20%)
            deviation = abs(seg['length'] - median_len) / median_len
            
            if deviation <= LENGTH_TOLERANCE:
                # 长度正常，进行计算
                w = calculate_width(seg['c1_data'], seg['c2_data'])
                if not np.isnan(w):
                    valid_widths.append(w)
            else:
                dropped_count += 1
                
        # 3. 返回清洗后的平均值
        if not valid_widths: return np.nan, "所有段均无效"
        
        # 再次利用鲁棒平均剔除计算值的离群点
        final_vals = np.array(valid_widths)
        med_val = np.median(final_vals)
        # 只取中位数附近的值，防止某一正常长度的段算出离谱结果
        final_valid = final_vals[(final_vals > med_val * 0.7) & (final_vals < med_val * 1.3)]
        
        if len(final_valid) == 0: return np.nan, "计算值离散度过大"
        
        return np.mean(final_valid), f"保留 {len(final_valid)}/{len(segments)} 段 (剔除 {dropped_count} 个畸形)"

    except Exception as e:
        return np.nan, str(e)

# ==============================================================================
# 3. 主程序
# ==============================================================================
def fix_dataset2_errors():
    print("🚀 开始执行【畸形波形剔除与修复】...")
    
    # 1. 收集所有需要修复的文件
    tasks = []
    
    # 读取之前生成的报告
    if os.path.exists('failed_report.csv'):
        tasks.append(pd.read_csv('failed_report.csv'))
    if os.path.exists('outliers_report.csv'):
        tasks.append(pd.read_csv('outliers_report.csv'))
        
    if not tasks:
        print("❌ 未找到错误报告 (failed_report.csv 或 outliers_report.csv)。")
        print("   请先运行 main_analysis.py 生成报告。")
        return

    todo_df = pd.concat(tasks, ignore_index=True).drop_duplicates(subset=['Filename'])
    print(f"📋 待修复文件数: {len(todo_df)}")
    
    fixed_results = []
    
    for i, row in todo_df.iterrows():
        fname = str(row['Filename'])
        pos = row['Position_mm']
        f_path = os.path.join(DATA_FOLDER, fname)
        
        if not os.path.exists(f_path):
            print(f"⚠️ 文件丢失: {fname}")
            continue
            
        # 执行智能修复
        new_width, msg = process_single_file_smart_filter(f_path)
        
        status = "✅ 修复成功" if not np.isnan(new_width) else "❌ 仍无效"
        print(f"[{i+1}/{len(todo_df)}] {fname} ({pos}mm): {status} | {msg}")
        
        if not np.isnan(new_width):
            fixed_results.append({
                'Position_mm': pos,
                'FFT_Width': new_width,
                'Filename': fname
            })

    # 2. 合并结果
    if fixed_results:
        # 读取原始成功结果
        if os.path.exists('fft_results.csv'):
            original_df = pd.read_csv('fft_results.csv')
            # 从原始结果中删掉我们刚刚修复的这些（避免重复，用新的覆盖旧的）
            fixed_filenames = [x['Filename'] for x in fixed_results]
            original_df = original_df[~original_df['Filename'].isin(fixed_filenames)]
            
            # 合并
            final_df = pd.concat([original_df, pd.DataFrame(fixed_results)], ignore_index=True)
        else:
            final_df = pd.DataFrame(fixed_results)
            
        final_df = final_df.sort_values(by='Position_mm')
        final_df.to_csv(OUTPUT_FIXED_CSV, index=False)
        
        print(f"\n🎉 修复完成！")
        print(f"   共修复了 {len(fixed_results)} 个坏文件。")
        print(f"   最终完整结果已保存至: {OUTPUT_FIXED_CSV}")
        print("   现在可以用 plot_raw_scatter_new.py 读取这个新文件画图了！")
    else:
        print("\n⚠️ 没有成功修复任何文件。")

if __name__ == '__main__':
    fix_dataset2_errors()