# =========================================================
# [脚本说明]
# 用途：生成最终发表风格的线性拟合图（优先使用修复后的结果）。
# 输入：1. 设置论文绘图风格（Times New Roman 等）。
# 输出：2. 优先加载 `fixed_results.csv`。
# =========================================================

# =========================================================
# [????]
# ?????????????????????????
# ???fixed_results.csv ? clean_data.csv?
# ???Fig1_LinearFit_Fixed.bmp?
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit
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
PIXEL_TO_UM = 14.0         
TARGET_RANGE = (180.0, 230.0) # 根据新数据集范围调整 (180-227mm)

def set_pub_style():
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = False
    plt.rcParams['ytick.right'] = False
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['lines.linewidth'] = 1.5

def load_data_final():
    # 【核心修改】优先读取修复后的结果
    if os.path.exists('data/processed/fixed_results.csv'):
        print("✅ 读取精修数据: fixed_results.csv")
        return pd.read_csv('data/processed/fixed_results.csv')
    elif os.path.exists('data/processed/clean_data.csv'):
        print("⚠️ 未找到修复数据，读取 clean_data.csv")
        return pd.read_csv('data/processed/clean_data.csv')
    else:
        print("❌ 没找到数据文件！")
        return pd.DataFrame()

def plot_linear_fit_final():
    set_pub_style()
    df_all = load_data_final()
    if df_all.empty: return

    # 排序
    df_all = df_all.sort_values(by='Position_mm')
    
    # 转换单位 (如果 CSV 里还没有 Width_um)
    if 'Width_um' not in df_all.columns:
        df_all['Width_um'] = df_all['FFT_Width'] * PIXEL_TO_UM

    # 筛选范围 (新数据集大概在 180-230 之间)
    df_subset = df_all[(df_all['Position_mm'] >= TARGET_RANGE[0]) & (df_all['Position_mm'] <= TARGET_RANGE[1])]
    
    # 拟合
    def linear_func(x, A, B): return A * x + B
    
    try:
        popt, _ = curve_fit(linear_func, df_subset['Position_mm'], df_subset['Width_um'])
        
        y_pred = linear_func(df_subset['Position_mm'], *popt)
        ss_res = np.sum((df_subset['Width_um'] - y_pred)**2)
        ss_tot = np.sum((df_subset['Width_um'] - df_subset['Width_um'].mean())**2)
        r2 = 1 - (ss_res / ss_tot)
        
        print(f"📈 拟合结果: y = {popt[0]:.4f}x + {popt[1]:.4f}")
        print(f"🌟 决定系数 R2 = {r2:.6f}")

        # 绘图
        fig, ax = plt.subplots(figsize=(6, 4.5)) 
        
        # 散点
        ax.scatter(df_subset['Position_mm'], df_subset['Width_um'], 
                   color='black', marker='o', s=25, 
                   edgecolors='black', facecolors='none', 
                   label='Experimental Data')
        
        # 拟合线
        x_line = np.linspace(df_subset['Position_mm'].min(), df_subset['Position_mm'].max(), 300)
        y_line = linear_func(x_line, *popt)
        ax.plot(x_line, y_line, color='#D62728', linestyle='-', linewidth=2, label='Linear Fitting')
        
        # 文本框 (左下角)
        text_str = f'$w = {popt[0]:.4f}d_0 + {popt[1]:.4f}$\n$R^2 = {r2:.4f}$'
        ax.text(0.05, 0.1, text_str, transform=ax.transAxes, fontsize=12, 
                va='bottom', ha='left', 
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        ax.set_xlabel(r'$d_0$ / mm')
        ax.set_ylabel(r'$w$ / $\mathrm{\mu m}$')
        
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(loc='upper right', frameon=False, fontsize=10)
        
        plt.tight_layout()
        
        # 保存 BMP
        save_name = 'Fig1_LinearFit_Fixed.bmp'
        if HAS_PIL:
            temp_png = 'temp_fig1.png'
            plt.savefig(temp_png, dpi=600)
            Image.open(temp_png).save(save_name)
            os.remove(temp_png)
        else:
            plt.savefig(save_name, dpi=600)
            
        print(f"✅ 图片已生成: {save_name}")
        plt.show()

    except Exception as e:
        print(f"❌ 拟合失败: {e}")

if __name__ == '__main__':
    plot_linear_fit_final()