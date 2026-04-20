# =========================================================
# [脚本说明]
# 用途：对指定位置输出“五联图”全链路诊断（原始、对齐、差分、局部、频谱）。
# 输入：`-449 -> 180.0mm`）。
# 输出：2. 提取第一个有效段。
# =========================================================

# =========================================================
# [????]
# ????????????????
# ????????? + ???????
# ???diagnose_5plots_*.png?
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import os
import re

# ==============================================================================
# 全局设置
# ==============================================================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def inspect_constrained_5plots(folder_path, target_positions):
    print(f"🕵️‍♂️ 开始执行【周期约束版】五图全链路诊断...")
    print("数据映射：文件 -449 -> 180mm, 步长 0.1mm")
    
    if not os.path.exists(folder_path):
        print(f"❌ 错误：找不到文件夹 {folder_path}")
        return

    # 1. 建立文件索引 (关键修改：适应新文件名规则)
    all_files = os.listdir(folder_path)
    pos_map = {}
    for f in all_files:
        name_body = f[:-4] if f.lower().endswith('.csv') else f
        if re.match(r'^-?\d+$', name_body):
            file_num = int(name_body)
            # 【核心修改】：起始文件 -449 对应 180.0mm
            pos = round(180.0 + (file_num - (-449)) * 0.1, 1)
            pos_map[pos] = f
            
    # 2. 逐个位置处理
    for target_pos in target_positions:
        target_pos = round(target_pos, 1)
        if target_pos not in pos_map:
            print(f"⚠️ 找不到位置 {target_pos}mm 的文件")
            continue
            
        filename = pos_map[target_pos]
        filepath = os.path.join(folder_path, filename)
        print(f"\n🔬 正在诊断位置: {target_pos} mm (文件: {filename})")
        
        try:
            # --- 数据读取 ---
            try: df = pd.read_csv(filepath)
            except: df = pd.read_csv(filepath, engine='python')
            
            if 'Channel 1' not in df.columns: continue
            df['Channel 1'] = pd.to_numeric(df['Channel 1'], errors='coerce').fillna(-1)
            df['Channel 2'] = pd.to_numeric(df['Channel 2'], errors='coerce').fillna(-1)
            
            # 截取有效段
            starts = df.index[(df['Channel 1'] == 0) & (df['Channel 2'] == 0)].tolist()
            ends = df.index[(df['Channel 1'] == 0) & (df['Channel 2'] == 61680)].tolist()
            
            valid_seg = None
            for s in starts:
                poss_ends = [e for e in ends if e > s]
                if poss_ends and (poss_ends[0] - s > 100):
                    valid_seg = df.iloc[s+1 : poss_ends[0]]
                    break
            
            if valid_seg is None:
                print("❌ 波形无效")
                continue

            # 获取原始数据 (Raw)
            ch1_raw = valid_seg['Channel 1'].values
            ch2_raw = valid_seg['Channel 2'].values
            
            # --- 1. 预处理 (Smoothing + Detrending) ---
            win = 10
            ch1_s = np.convolve(ch1_raw, np.ones(win)/win, mode='same')
            ch2_s = np.convolve(ch2_raw, np.ones(win)/win, mode='same')
            ch1_c = ch1_s - np.mean(ch1_s)
            ch2_c = ch2_s - np.mean(ch2_s)
            
            # --- 2. 周期约束对齐 (Constrained Alignment) ---
            # 2.1 预估周期
            n_fft = len(ch1_c)
            yf_temp = np.abs(fft(ch1_c)[:n_fft//2])
            idx_temp = np.argmax(yf_temp[5:]) + 5
            period_est = int(n_fft / idx_temp) if idx_temp > 0 else 50
            
            # 2.2 锁死搜索范围 (±0.6 T)
            search_radius = int(period_est * 0.6)
            
            corr = signal.correlate(ch1_c, ch2_c, mode='full')
            lags = signal.correlation_lags(len(ch1_c), len(ch2_c), mode='full')
            
            mask = (lags >= -search_radius) & (lags <= search_radius)
            
            shift = 0
            if np.any(mask):
                best_idx = np.argmin(corr[mask]) # 找最负点 (反相)
                shift = lags[mask][best_idx]
            
            print(f"👉 预估周期: {period_est} pixel")
            print(f"👉 搜索范围: [{-search_radius}, {search_radius}]")
            print(f"👉 计算位移: {shift}")

            # 3. 切片对齐
            if shift > 0:
                ch1_aligned = ch1_c[shift:]
                ch2_aligned = ch2_c[:-shift]
            elif shift < 0:
                ch1_aligned = ch1_c[:shift]
                ch2_aligned = ch2_c[-shift:]
            else:
                ch1_aligned = ch1_c
                ch2_aligned = ch2_c
                
            sig_aligned = ch1_aligned - ch2_aligned

            # --- 4. 准备绘图数据 (Zoom & FFT) ---
            
            # 局部放大区域 (取中间 200 个点)
            mid_idx = len(ch1_aligned) // 2
            zoom_radius = 100
            z_start = max(0, mid_idx - zoom_radius)
            z_end = min(len(ch1_aligned), mid_idx + zoom_radius)
            z_x = np.arange(z_start, z_end)
            
            z_ch1 = ch1_aligned[z_start:z_end]
            z_ch2 = ch2_aligned[z_start:z_end]
            z_diff = sig_aligned[z_start:z_end]
            
            # FFT 计算
            N = len(sig_aligned)
            window = np.hanning(N)
            yf = fft(sig_aligned * window)
            yf_mag = np.abs(yf[:N//2])
            xf = fftfreq(N, 1)[:N//2]
            
            peak_idx = np.argmax(yf_mag[5:]) + 5
            peak_freq = xf[peak_idx]
            width_calc = 1.0 / (2.0 * peak_freq) if peak_freq > 0 else 0

            # --- 5. 绘制五图 (5 Subplots) ---
            fig = plt.figure(figsize=(12, 20)) # 画布拉长
            plt.suptitle(f"全链路诊断报告: 位置 {target_pos}mm (Shift={shift})", fontsize=16, fontweight='bold')
            
            # 图1: 原始采集波形 (Raw)
            ax1 = plt.subplot(5, 1, 1)
            ax1.plot(ch1_raw, color='orange', alpha=0.7, label='Ch1 Raw')
            ax1.plot(ch2_raw, color='green', alpha=0.7, label='Ch2 Raw')
            ax1.set_title("1. 原始采集波形 (Original Raw Waveforms)")
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            
            # 图2: 对齐后全景 (Aligned Panorama)
            ax2 = plt.subplot(5, 1, 2)
            ax2.plot(ch1_aligned, color='orange', alpha=0.7, label='Ch1 Aligned')
            ax2.plot(ch2_aligned, color='green', alpha=0.7, label='Ch2 Aligned')
            ax2.set_title("2. 对齐后全景 (Aligned Panorama - De-trended)")
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            
            # 图3: 算法处理后信号 (Differential)
            ax3 = plt.subplot(5, 1, 3)
            ax3.plot(sig_aligned, color='#1f77b4', label='Diff (Ch1 - Ch2)')
            ax3.set_title("3. 差分处理结果 (Differential Signal)")
            ax3.legend(loc='upper right')
            ax3.grid(True, alpha=0.3)
            
            # 图4: 局部显微 (Zoom: Aligned + Diff)
            ax4 = plt.subplot(5, 1, 4)
            # 画 Ch1 和 Ch2
            ax4.plot(z_x, z_ch1, '.-', color='orange', alpha=0.6, label='Ch1')
            ax4.plot(z_x, z_ch2, '.-', color='green', alpha=0.6, label='Ch2')
            # 画差分 (加粗显示)
            ax4.plot(z_x, z_diff, '.-', color='#1f77b4', linewidth=2, label='Diff Signal')
            
            ax4.set_title(f"4. 局部显微 (Zoom {z_start}-{z_end}) [Ch1, Ch2 & Diff]")
            
            # 画辅助线 (标记 Ch1 峰值)
            local_peaks, _ = signal.find_peaks(z_ch1, distance=10)
            for p in local_peaks:
                ax4.axvline(z_x[p], color='red', linestyle='--', alpha=0.3)
                
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
            
            # 图5: FFT 频谱 (Spectrum)
            ax5 = plt.subplot(5, 1, 5)
            ax5.plot(xf, yf_mag, color='purple', label='Magnitude')
            ax5.plot(xf[peak_idx], yf_mag[peak_idx], 'ro')
            ax5.annotate(f'Peak: {peak_freq:.4f}\nWidth: {width_calc:.2f}', 
                         xy=(xf[peak_idx], yf_mag[peak_idx]), 
                         xytext=(xf[peak_idx]+0.02, yf_mag[peak_idx]),
                         arrowprops=dict(facecolor='black', shrink=0.05))
            
            ax5.set_title("5. FFT 频谱分析 (Frequency Spectrum)")
            ax5.set_xlim(0, 0.1) # 聚焦低频
            ax5.set_xlabel("Frequency (Cycles/Pixel)")
            ax5.grid(True, alpha=0.3)
            
            plt.tight_layout()
            out_name = f"diagnose_5plots_{target_pos}mm.png"
            plt.savefig(out_name, dpi=150)
            print(f"✅ 五图诊断已生成: {out_name}")
            plt.close()

        except Exception as e:
            print(f"❌ 出错: {e}")

# =========================================================
if __name__ == '__main__':
    # ⚠️ 这里的路径已更新为新的数据集路径
    data_folder = r'D:\桌面\new learning\069date\1.22'
    
    # 输入您想检查的位置 (例如 190.0)
    targets = [181.5] 
    
    inspect_constrained_5plots(data_folder, targets)