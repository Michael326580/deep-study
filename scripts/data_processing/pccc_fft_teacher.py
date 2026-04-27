# =========================================================
# [脚本说明]
# 用途：主流程脚本之一，完成全量鲁棒提取 + 双曲线拟合 + 离群诊断。
# 输入：平滑去均值，估周期，互相关在 ±0.6T 内找位移，FFT 主峰抛物线插值。
# 输出：按标记切段后做两级清洗。
# =========================================================

# =========================================================
# [????]
# ??????????????????? + ????? + ???
# ???????? CSV ???
# ???fft_results_robust.csv?failed/outliers ???????
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft
from scipy.optimize import curve_fit
import os
import re
from pathlib import Path

# ==============================================================================
# 0. 全局设置
# ==============================================================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ★★★ 智能清洗参数 ★★★
LENGTH_TOLERANCE = 0.05  # 长度容差：偏离中位数 20% 的波形段被视为畸形
VALUE_TOLERANCE = 0.15   # 数值容差：计算结果偏离中位数 30% 的被视为计算错误

# ==============================================================================
# 第一部分：核心算法 (FFT & 智能清洗)
# ==============================================================================
def estimate_period(sig):
    """预估信号周期"""
    n = len(sig)
    if n == 0: return 50
    yf = np.abs(fft(sig)[:n//2])
    # 忽略直流和极低频 (前5个点)
    idx = np.argmax(yf[5:]) + 5
    if idx <= 0: return 50 
    return int(n / idx)

def calculate_width_constrained(ch1, ch2):
    """
    计算半波长宽度（带周期约束）
    保留了您原始的高精度逻辑
    """
    n = len(ch1)
    if n < 100: return np.nan # 稍微放宽一点限制，适应短波形
    
    # 1. 简单预处理 (去直流)
    win = 10
    ch1_s = np.convolve(ch1.values, np.ones(win)/win, mode='same')
    ch2_s = np.convolve(ch2.values, np.ones(win)/win, mode='same')
    ch1_c = ch1_s - np.mean(ch1_s)
    ch2_c = ch2_s - np.mean(ch2_s)
    
    # 2. 预估周期
    period_est = estimate_period(ch1_c)
    if period_est < 10: period_est = 50
    
    # 3. 互相关 (带范围锁死)
    corr = signal.correlate(ch1_c, ch2_c, mode='full')
    lags = signal.correlation_lags(len(ch1_c), len(ch2_c), mode='full')
    
    # 锁死在 ±0.6 个周期内，防止跳变到下一个波峰
    search_radius = int(period_est * 0.6) 
    mask = (lags >= -search_radius) & (lags <= search_radius)
    
    shift = 0
    if np.any(mask):
        best_idx = np.argmin(corr[mask]) # 找最负 (反相点)
        shift = lags[mask][best_idx]
    
    # 4. 切片对齐
    if shift > 0:
        sig = ch1_c[shift:] - ch2_c[:-shift]
    elif shift < 0:
        sig = ch1_c[:shift] - ch2_c[-shift:]
    else:
        sig = ch1_c - ch2_c
        
    # 5. FFT 分析
    N = len(sig)
    if N < 50: return np.nan
    
    window = np.hanning(N)
    sig_w = sig * window
    
    yf = fft(sig_w)
    yf_mag = np.abs(yf[:N//2])
    
    # 6. 找主频 & 抛物线插值优化
    start_idx = 5
    if len(yf_mag) <= start_idx: return np.nan
    peak_idx = np.argmax(yf_mag[start_idx:]) + start_idx
    
    if 0 < peak_idx < len(yf_mag) - 1:
        y_L = yf_mag[peak_idx - 1]
        y_C = yf_mag[peak_idx]
        y_R = yf_mag[peak_idx + 1]
        denom = y_L - 2*y_C + y_R
        if denom != 0:
            delta = 0.5 * (y_L - y_R) / denom
            true_peak_idx = peak_idx + delta
        else:
            true_peak_idx = peak_idx
    else:
        true_peak_idx = peak_idx
        
    frequency = true_peak_idx / N
    if frequency <= 0: return np.nan
    
    # 半波长宽度 = 1 / (2 * 频率)
    return 1.0 / (2.0 * frequency)

def process_single_file_mlv(file_path):
    """
    【智能清洗核心】
    1. 读取文件
    2. 提取所有波形段
    3. MLV (基于中位数长度投票) 剔除畸形段
    4. 对有效段计算宽度
    5. 数值清洗，返回最终均值
    """
    try:
        # 1. 鲁棒读取
        try: df = pd.read_csv(file_path)
        except: df = pd.read_csv(file_path, engine='python')
        
        if 'Channel 1' not in df.columns: return np.nan, "列名错误"

        c1 = pd.to_numeric(df['Channel 1'], errors='coerce').fillna(0)
        c2 = pd.to_numeric(df['Channel 2'], errors='coerce').fillna(0)
        
        # 2. 寻找波形段起止点
        starts = df.index[(c1 == 0) & (c2 == 0)].tolist()
        ends = df.index[(c1 == 0) & (c2 == 61680)].tolist()
        
        segments = []
        for s in starts:
            possible_ends = [e for e in ends if e > s]
            if possible_ends:
                e = possible_ends[0]
                length = e - s
                # 忽略极短的噪点 (小于50点肯定不是有效波形)
                if length > 50: 
                    segments.append({
                        'length': length,
                        'c1': c1.iloc[s+1:e],
                        'c2': c2.iloc[s+1:e]
                    })
        
        if not segments: return np.nan, "无有效波形段"

        # 3. 【清洗层级一】基于中位数长度的“民主投票”
        lengths = [seg['length'] for seg in segments]
        median_len = np.median(lengths)
        
        valid_widths = []
        
        for seg in segments:
            # 计算长度偏差率
            deviation = abs(seg['length'] - median_len) / median_len
            
            if deviation <= LENGTH_TOLERANCE:
                # 长度合格，才进行计算 (调用上面的核心算法)
                w = calculate_width_constrained(seg['c1'], seg['c2'])
                if not np.isnan(w):
                    valid_widths.append(w)
                
        if not valid_widths: 
            return np.nan, f"所有段长度均异常 (中位数:{median_len})"
        
        # 4. 【清洗层级二】基于计算结果的离群值剔除
        final_vals = np.array(valid_widths)
        med_val = np.median(final_vals)
        
        # 只保留中位数附近的数值
        lower_bound = med_val * (1 - VALUE_TOLERANCE)
        upper_bound = med_val * (1 + VALUE_TOLERANCE)
        final_valid = final_vals[(final_vals > lower_bound) & (final_vals < upper_bound)]
        
        if len(final_valid) == 0: 
            return np.nan, "计算结果离散度过大"
        
        # 计算最终均值
        result = np.mean(final_valid)
        info = f"保留 {len(final_valid)}/{len(segments)} 段"
        return result, info

    except Exception as e:
        return np.nan, str(e)

# ==============================================================================
# 第二部分：批量处理流程 (应用全量清洗)
# ==============================================================================
def process_batch_all(folder_path):
    print(f"\nSTEP 1: 执行全量智能预处理 (MLV + FFT)")
    print(f"📂 数据源: {folder_path}")
    print(f"⚙️  策略: 长度容差 ±{int(LENGTH_TOLERANCE*100)}%, 数值容差 ±{int(VALUE_TOLERANCE*100)}%")
    
    if not os.path.exists(folder_path):
        print(f"❌ 错误：找不到文件夹 {folder_path}")
        return

    all_files = os.listdir(folder_path)
    tasks = []
    
    # --- 1.1 文件名解析 ---
    for f in all_files:
        name_body = f[:-4] if f.lower().endswith('.csv') else f
        
        if re.match(r'^-?\d+$', name_body):
            file_num = int(name_body)
            # 【映射规则】: -449 对应 180.0mm
            position = 180.0 + (file_num - (-449)) * 0.1
            tasks.append({'filename': f, 'file_num': file_num, 'pos': position})
    
    tasks.sort(key=lambda x: x['file_num'])
    print(f"📋 识别到 {len(tasks)} 个文件，开始清洗计算...")
    
    results = []
    failures = []

    # --- 1.2 循环处理 ---
    total = len(tasks)
    for i, task in enumerate(tasks):
        f_path = os.path.join(folder_path, task['filename'])
        
        # ★★★ 调用智能清洗函数 ★★★
        width, msg = process_single_file_mlv(f_path)
        
        if not np.isnan(width):
            results.append({
                'Position_mm': round(task['pos'], 1), 
                'FFT_Width': width, 
                'Filename': task['filename']
            })
        else:
            failures.append({
                'Filename': task['filename'], 
                'Position_mm': round(task['pos'], 1), 
                'Reason': msg
            })

        if (i + 1) % 50 == 0: 
            print(f"⏳ [{i + 1}/{total}] 处理中...")

    # --- 1.3 保存结果 ---
    if results:
        res_df = pd.DataFrame(results).dropna()
        Path('data/processed').mkdir(parents=True, exist_ok=True)
        res_df.to_csv('data/processed/fft_results_robust.csv', index=False)
        print(f"\n✅ 成功结果已保存: data/processed/fft_results_robust.csv (共 {len(res_df)} 条)")
    
    if failures:
        fail_df = pd.DataFrame(failures)
        fail_df.to_csv('data/processed/failed_report.csv', index=False)
        print(f"⚠️ {len(fail_df)} 个文件处理失败/被过滤，详情见 failed_report.csv")
    else:
        print("🎉 完美！所有文件均通过清洗。")

# ==============================================================================
# 第三部分：双曲线拟合与诊断 (您的原版绘图逻辑)
# ==============================================================================
def fit_hyperbola_diagnostic(csv_file):
    print(f"\nSTEP 2: 执行双曲线拟合诊断 (含绘图)")
    if not os.path.exists(csv_file):
        print(f"❌ 错误：找不到文件 {csv_file}")
        return
        
    df = pd.read_csv(csv_file)
    if df.empty: return

    # 定义双曲线模型: w = 1 / (Ax + B) + C
    def hyperbola_func(x, A, B, C):
        return 1.0 / (A * x + B) + C

    # 1. 初始拟合
    p0 = [0.001, 0.01, 0]
    try:
        popt_init, _ = curve_fit(hyperbola_func, df['Position_mm'], df['FFT_Width'], 
                                 p0=p0, method='trf', loss='soft_l1', f_scale=0.5, maxfev=20000)
    except:
        def hyperbola_simple(x, A, B): return 1.0 / (A * x + B)
        popt_init, _ = curve_fit(hyperbola_simple, df['Position_mm'], df['FFT_Width'], maxfev=20000)
        popt_init = np.append(popt_init, 0)

    # 2. IQR 异常剔除
    y_pred_init = hyperbola_func(df['Position_mm'], *popt_init)
    residuals_init = df['FFT_Width'] - y_pred_init
    
    Q1 = np.percentile(residuals_init, 25)
    Q3 = np.percentile(residuals_init, 75)
    IQR = Q3 - Q1
    fence_factor = 2.5 
    mask = (residuals_init >= Q1 - fence_factor*IQR) & (residuals_init <= Q3 + fence_factor*IQR)
    
    clean_df = df[mask].copy()
    outliers = df[~mask].copy()
    
    if not outliers.empty:
        outliers.to_csv('data/processed/outliers_report.csv', index=False)
        print(f"⚠️ 拟合后筛选出 {len(outliers)} 个离群点")
    
    # 3. 最终拟合
    try:
        popt_final, _ = curve_fit(hyperbola_func, clean_df['Position_mm'], clean_df['FFT_Width'], 
                                  p0=popt_init, method='trf', loss='soft_l1', maxfev=20000)
    except:
        popt_final = popt_init

    # 4. 生成诊断图 (High Quality)
    A, B, C = popt_final
    equation = f"W = 1 / ({A:.6f}x + {B:.6f}) + {C:.4f}"
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # 上图：拟合效果
    ax1.scatter(df['Position_mm'], df['FFT_Width'], c='gray', s=10, alpha=0.3, label='原始数据 (清洗后)')
    ax1.scatter(clean_df['Position_mm'], clean_df['FFT_Width'], c='#1f77b4', s=20, label='拟合有效数据')
    if not outliers.empty:
        ax1.scatter(outliers['Position_mm'], outliers['FFT_Width'], c='red', marker='x', s=50, label='离群点')
    
    x_smooth = np.linspace(df['Position_mm'].min(), df['Position_mm'].max(), 500)
    ax1.plot(x_smooth, hyperbola_func(x_smooth, *popt_final), 'r-', linewidth=2, label='拟合曲线')
    
    ax1.set_title(f"诊断结果: {equation}", fontsize=12)
    ax1.set_ylabel("半波长宽度 (Pixel)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 下图：残差
    res = clean_df['FFT_Width'] - hyperbola_func(clean_df['Position_mm'], *popt_final)
    ax2.axhline(0, color='black', alpha=0.7)
    ax2.scatter(clean_df['Position_mm'], res, c='purple', s=15, alpha=0.6, label='残差')
    
    ax2.set_xlabel("物理位置 (mm)")
    ax2.set_ylabel("偏差 (Pixel)")
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    Path('results/diagnostics').mkdir(parents=True, exist_ok=True)
    plt.savefig('results/diagnostics/final_diagnostic_plot.png', dpi=150)
    print("📸 诊断图表已生成: final_diagnostic_plot.png")
    # plt.show() 

# ==============================================================================
# 主入口
# ==============================================================================
if __name__ == '__main__':
    # ⚠️ 请确认您的数据路径
    data_folder = 'data/raw/1.22'
    
    # 1. 运行全量智能清洗与计算
    process_batch_all(data_folder)
    
    # 2. 运行双曲线拟合诊断与绘图
    fit_hyperbola_diagnostic('data/processed/fft_results_robust.csv')