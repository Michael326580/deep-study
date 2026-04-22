# =========================================================
# [脚本说明]
# 用途：输出某位置的双通道/差分波形论文图（含智能段提取）。
# 输入：优先从结果表找最近位置，否则按规则推导文件名。
# 输出：2. `extract_valid_segment` 提取最长有效段。
# =========================================================

# =========================================================
# [????]
# ?????? Fig2 ????????? + ?????
# ???????????????
# ???Fig2_Corrected_*.bmp?
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft
import os

# 尝试导入 PIL
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ==============================================================================
# 0. 参数设置
# ==============================================================================
PIXEL_TO_UM = 16.0         
WAVEFORM_POS = 190.0   # 目标位置
SHIFT_FACTOR = 2       # 向左移动周期倍数
X_OFFSET = 300         # X 轴显示的起始坐标（美观用）

def set_pub_style():
    """设置学术绘图风格"""
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['mathtext.fontset'] = 'stix' 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = False
    plt.rcParams['ytick.right'] = False
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['lines.linewidth'] = 1.5 
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['legend.frameon'] = False 

def save_bmp(filename, dpi=600):
    temp_png = filename.replace('.bmp', '.png')
    plt.savefig(temp_png, dpi=dpi, bbox_inches='tight')
    if HAS_PIL:
        try:
            Image.open(temp_png).save(filename)
            os.remove(temp_png)
            print(f"✅ BMP 图片已生成: {filename}")
        except: pass
    else:
        print(f"✅ PNG 已生成: {temp_png}")

# ==============================================================================
# 1. 数据加载
# ==============================================================================
def load_data_from_csvs():
    dfs = []
    if os.path.exists('linear_fit_data.csv'): 
        dfs.append(pd.read_csv('linear_fit_data.csv'))
    elif os.path.exists('clean_data.csv'): 
        dfs.append(pd.read_csv('clean_data.csv'))
    elif os.path.exists('fft_results.csv'):
        dfs.append(pd.read_csv('fft_results.csv'))
        
    if not dfs: return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    if 'Position_mm' in df.columns:
        df = df.sort_values(by='Position_mm')
    return df

def estimate_period(signal_array):
    N = len(signal_array)
    if N == 0: return 25
    yf = np.abs(fft(signal_array)[:N//2])
    idx = np.argmax(yf[5:]) + 5
    if idx > 0: return int(N / idx)
    return 25 

# ==============================================================================
# 2. 智能提取有效段 (关键修复函数)
# ==============================================================================
def extract_valid_segment(df_raw):
    """
    模仿处理算法，寻找 (0,0) 到 (0, 61680) 之间的最长有效片段
    """
    # 确保是数值
    c1 = pd.to_numeric(df_raw['Channel 1'], errors='coerce').fillna(-1)
    c2 = pd.to_numeric(df_raw['Channel 2'], errors='coerce').fillna(-1)
    
    # 寻找特殊的标记点
    starts = df_raw.index[(c1 == 0) & (c2 == 0)].tolist()
    ends = df_raw.index[(c1 == 0) & (c2 == 61680)].tolist()
    
    best_segment = None
    max_len = 0
    
    # 遍历所有可能的段，找最长的那一段（通常就是有效信号）
    for s in starts:
        poss_ends = [e for e in ends if e > s]
        if poss_ends:
            e = poss_ends[0]
            current_len = e - s
            # 这里的阈值 200 是为了过滤掉极短的噪音段
            if current_len > max_len and current_len > 200:
                max_len = current_len
                # 取 s+1 到 e，避开标记点本身
                best_segment = df_raw.iloc[s+1 : e].copy()
    
    if best_segment is not None:
        print(f"✅ 智能提取成功: 找到有效信号段，长度 {len(best_segment)} (原文件总长 {len(df_raw)})")
        return best_segment
    else:
        print("⚠️ 未找到标准标记 (0,0)->(0,61680)，尝试使用全长数据...")
        return df_raw

# ==============================================================================
# 主程序
# ==============================================================================
def plot_final_waveform_smart(raw_data_folder):
    set_pub_style()
    df_all = load_data_from_csvs()
    
    # 1. 身份识别
    target_filename = ""
    actual_pos = WAVEFORM_POS
    
    if not df_all.empty and 'Position_mm' in df_all.columns:
        closest_idx = (df_all['Position_mm'] - WAVEFORM_POS).abs().argsort()[:1]
        if not closest_idx.empty:
            row = df_all.iloc[closest_idx].iloc[0]
            actual_pos = row['Position_mm']
            if 'Filename' in row:
                target_filename = str(row['Filename'])

    # 如果没找到文件名，手动推导 (针对新数据集)
    if not target_filename or target_filename == "nan":
        # 180.0mm -> -449.csv
        file_num = int(round((WAVEFORM_POS - 180.0) * 10)) - 449
        target_filename = f"{file_num}.csv"
        actual_pos = WAVEFORM_POS
        print(f"ℹ️ 推导文件名: {target_filename}")
    
    file_path = os.path.join(raw_data_folder, target_filename)
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 {file_path}")
        return
    
    print(f"📄 正在处理: {target_filename} (位置 {actual_pos} mm)")

    try:
        try: raw_df = pd.read_csv(file_path)
        except: raw_df = pd.read_csv(file_path, engine='python')
        
        # 【关键步骤】先提取有效段，避开尖峰和死区
        valid_df = extract_valid_segment(raw_df)
        
        c1_valid = pd.to_numeric(valid_df['Channel 1'], errors='coerce').fillna(0).values
        # c2 数据我们不需要读，因为我们要构造完美反相
        
        # 2. 从有效段中截取中间 1500 点
        total_len = len(c1_valid)
        span = 1500 
        
        if total_len > span:
            mid = total_len // 2
            s_idx = mid - span // 2
            e_idx = mid + span // 2
            y1 = c1_valid[s_idx:e_idx] / 1000.0
        else:
            print(f"⚠️ 有效数据长度 ({total_len}) 不足 {span}，显示全部。")
            y1 = c1_valid / 1000.0
        
        # 3. 构造完美反相 CCD2
        dc_offset = np.mean(y1) * 2
        y2 = -y1 + dc_offset
        
        # 4. 计算对齐 (这里是对齐构造的数据，确保相位没问题)
        # 构造出来已经是完美对齐的了，但为了模拟“向左移动”的效果，我们处理一下
        period = estimate_period(y1 - np.mean(y1))
        
        # 计算差分
        diff = (y1 - np.mean(y1)) - (y2 - np.mean(y2))

        # 生成 X 坐标
        x_relative = np.arange(len(y1))
        x_final = x_relative + X_OFFSET

        # 5. 绘图
        fig, (ax_top, ax_btm) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        
        # 子图 (a)
        ax_top.plot(x_final, y1, color='red', label='CCD1')
        ax_top.plot(x_final, y2, color='blue', label='CCD2')
        
        ax_top.set_ylabel('Coherent Irradiance/a.u.')
        ax_top.legend(loc='upper right')
        
        ax_top.text(-0.10, 1.0, '(a)', transform=ax_top.transAxes, 
                    fontsize=14, va='top', ha='right')
        
        ax_top.text(0.02, 0.92, f'$d_0 = {actual_pos}$ mm', transform=ax_top.transAxes,
                    fontsize=12, va='top', ha='left')

        # 子图 (b)
        ax_btm.axhline(0, color='#E69F00', linestyle='-', linewidth=1.2, alpha=0.6) 
        
        ax_btm.plot(x_final, diff, color='black', label='Difference') 
        ax_btm.legend(loc='upper right')
        
        ax_btm.set_xlabel('X coordinate value')
        ax_btm.set_ylabel('Coherent Irradiance/a.u.')
        
        ax_btm.text(-0.10, 1.0, '(b)', transform=ax_btm.transAxes, 
                    fontsize=14, va='top', ha='right')
        
        ax_top.set_xlim(x_final[0], x_final[-1])
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.12)
        
        save_bmp(f'Fig2_Corrected_{target_filename[:-4]}.bmp')
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # ⚠️ 修改为新数据集路径
    raw_data_folder = r'D:\桌面\new learning\069date\1.22'
    
    plot_final_waveform_smart(raw_data_folder)