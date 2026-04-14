import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
import os
import re

# ==============================================================================
# 0. 全局设置
# ==============================================================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def process_and_linear_fit_annotated(folder_path):
    print(f"🚀 开始执行【185-195mm 局部线性拟合】(带标注版)...")
    print(f"📂 数据源: {folder_path}")
    
    # --- 1. 扫描并筛选文件 ---
    if not os.path.exists(folder_path):
        print(f"❌ 错误：找不到文件夹 {folder_path}")
        return

    all_files = os.listdir(folder_path)
    valid_tasks = []
    
    TARGET_MIN = 185.0
    TARGET_MAX = 195.0
    
    print(f"🎯 筛选范围: {TARGET_MIN}mm - {TARGET_MAX}mm")
    
    for f in all_files:
        name_body = f[:-4] if f.lower().endswith('.csv') else f
        if re.match(r'^-?\d+$', name_body):
            file_num = int(name_body)
            # 映射公式: -449 对应 180.0mm，步长 0.1
            position = 180.0 + (file_num - (-449)) * 0.1
            
            if TARGET_MIN <= position <= TARGET_MAX:
                valid_tasks.append({'filename': f, 'file_num': file_num, 'pos': position})
    
    valid_tasks.sort(key=lambda x: x['file_num'])
    print(f"✅ 共筛选出 {len(valid_tasks)} 个有效文件。")
    
    if len(valid_tasks) == 0:
        print("❌ 错误：未找到符合条件的文件。")
        return

    # --- 2. 核心算法 (周期约束版) ---
    def estimate_period(sig):
        n = len(sig)
        if n == 0: return 50
        yf = np.abs(fft(sig)[:n//2])
        idx = np.argmax(yf[5:]) + 5
        return int(n / idx) if idx > 0 else 50

    def calculate_width(ch1, ch2):
        n = len(ch1)
        if n < 200: return np.nan
        
        # 预处理
        win = 10
        ch1_s = np.convolve(ch1.values, np.ones(win)/win, mode='same')
        ch2_s = np.convolve(ch2.values, np.ones(win)/win, mode='same')
        ch1_c = ch1_s - np.mean(ch1_s)
        ch2_c = ch2_s - np.mean(ch2_s)
        
        # 周期约束
        p_est = estimate_period(ch1_c)
        r = int(p_est * 0.6)
        
        corr = signal.correlate(ch1_c, ch2_c, mode='full')
        lags = signal.correlation_lags(len(ch1_c), len(ch2_c), mode='full')
        mask = (lags >= -r) & (lags <= r)
        
        shift = 0
        if np.any(mask):
            shift = lags[mask][np.argmin(corr[mask])]
        
        # 切片
        if shift > 0: sig = ch1_c[shift:] - ch2_c[:-shift]
        elif shift < 0: sig = ch1_c[:shift] - ch2_c[-shift:]
        else: sig = ch1_c - ch2_c
            
        # FFT
        N = len(sig)
        if N < 100: return np.nan
        yf_mag = np.abs(fft(sig * np.hanning(N))[:N//2])
        peak_idx = np.argmax(yf_mag[5:]) + 5
        
        # 插值
        if 0 < peak_idx < len(yf_mag) - 1:
            yL, yC, yR = yf_mag[peak_idx-1], yf_mag[peak_idx], yf_mag[peak_idx+1]
            denom = yL - 2*yC + yR
            if denom != 0: peak_idx += 0.5 * (yL - yR) / denom
            
        freq = peak_idx / N
        return 1.0 / (2.0 * freq) if freq > 0 else np.nan

    def get_robust_mean(data):
        d = np.array(data)
        d = d[~np.isnan(d)]
        if len(d) == 0: return np.nan
        m = np.median(d)
        v = d[(d > m*0.5) & (d < m*1.5)]
        return np.mean(v) if len(v) > 0 else np.nan

    # --- 3. 批量处理 ---
    results = []
    print("\n🔽 正在处理数据...")
    
    for i, task in enumerate(valid_tasks):
        try:
            f_path = os.path.join(folder_path, task['filename'])
            df = pd.read_csv(f_path)
            
            df['Channel 1'] = pd.to_numeric(df['Channel 1'], errors='coerce').fillna(-1)
            df['Channel 2'] = pd.to_numeric(df['Channel 2'], errors='coerce').fillna(-1)
            
            starts = df.index[(df['Channel 1'] == 0) & (df['Channel 2'] == 0)].tolist()
            ends = df.index[(df['Channel 1'] == 0) & (df['Channel 2'] == 61680)].tolist()
            
            file_widths = []
            for s_idx in starts:
                poss_ends = [e for e in ends if e > s_idx]
                if poss_ends:
                    e_idx = poss_ends[0]
                    if e_idx - s_idx > 100:
                        w = calculate_width(df.iloc[s_idx+1 : e_idx]['Channel 1'], 
                                          df.iloc[s_idx+1 : e_idx]['Channel 2'])
                        if not np.isnan(w):
                            file_widths.append(w)
            
            avg_w = get_robust_mean(file_widths)
            if not np.isnan(avg_w):
                results.append({'Position_mm': round(task['pos'], 1), 'FFT_Width': avg_w})
            
            if (i+1) % 25 == 0: print(f"   进度: {i+1}/{len(valid_tasks)}...")
                
        except Exception as e:
            print(f"⚠️ 处理 {task['filename']} 时出错: {e}")

    # --- 4. 拟合与绘图 ---
    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("❌ 错误：没有成功提取到任何数据点。")
        return
        
    # 清洗异常
    Q1, Q3 = df_res['FFT_Width'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df_clean = df_res[(df_res['FFT_Width'] >= Q1 - 1.5*IQR) & (df_res['FFT_Width'] <= Q3 + 1.5*IQR)]
    
    # 拟合 y = kx + b
    def linear_f(x, k, b): return k * x + b
    popt, _ = curve_fit(linear_f, df_clean['Position_mm'], df_clean['FFT_Width'])
    
    # R2 计算
    y_pred = linear_f(df_clean['Position_mm'], *popt)
    ss_res = np.sum((df_clean['FFT_Width'] - y_pred)**2)
    ss_tot = np.sum((df_clean['FFT_Width'] - df_clean['FFT_Width'].mean())**2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"\n✅ 拟合结果: y = {popt[0]:.4f}x + {popt[1]:.4f}")
    print(f"✅ 线性度 R²: {r2:.6f}")

    # --- 绘图 (带标注) ---
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 散点
    ax.scatter(df_res['Position_mm'], df_res['FFT_Width'], c='gray', alpha=0.3, label='原始点 (被清洗)')
    ax.scatter(df_clean['Position_mm'], df_clean['FFT_Width'], c='#1f77b4', s=30, label='拟合点 (有效)')
    
    # 拟合线
    x_range = np.linspace(TARGET_MIN, TARGET_MAX, 100)
    ax.plot(x_range, linear_f(x_range, *popt), 'r--', linewidth=2, label='线性拟合')
    
    # 【关键修改】在左下角添加文本框
    # transform=ax.transAxes 表示使用坐标轴比例坐标 (0,0)左下, (1,1)右上
    text_str = f"拟合方程:\n$y = {popt[0]:.4f}x + {popt[1]:.4f}$\n\n线性度:\n$R^2 = {r2:.6f}$"
    ax.text(0.03, 0.03, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

    ax.set_title("185-195mm 局部线性拟合分析", fontsize=14, fontweight='bold')
    ax.set_xlabel("物理位置 (mm)", fontsize=12)
    ax.set_ylabel("半波长宽度 (Pixel)", fontsize=12)
    ax.legend(loc='upper right') # 图例放右上角，避开文本框
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_name = 'linear_fit_final_annotated.png'
    plt.savefig(save_name, dpi=300)
    print(f"\n🖼️ 结果图已保存: {save_name}")
    plt.show()

if __name__ == '__main__':
    data_folder = r'D:\桌面\new learning\069date\1.22'
    process_and_linear_fit_annotated(data_folder)