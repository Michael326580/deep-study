import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit
import os

# ==============================================================================
# 0. 参数与全局设置
# ==============================================================================
# 这里的比例尺请根据实际情况确认
PIXEL_TO_UM = 14.0         
TARGET_RANGE = (185.0, 195.0)

def set_pub_style():
    """设置符合出版要求的图表风格 (Times New Roman)"""
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 11
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = False
    plt.rcParams['ytick.right'] = False
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['lines.linewidth'] = 1.5

# ==============================================================================
# 1. 数据加载函数 (已修复)
# ==============================================================================
def load_data_from_csvs():
    print("📥 Loading data...")
    dfs = []
    
    # 优先读取清洗后的数据
    if os.path.exists('clean_data.csv'):
        dfs.append(pd.read_csv('clean_data.csv'))
    
    # 初始化 df_all
    if not dfs:
        # 如果没有 clean_data，尝试读取 fft_results.csv (未经清洗的总表)
        if os.path.exists('fft_results.csv'):
             print("⚠️ Warning: Using 'fft_results.csv' (uncleaned data).")
             df_all = pd.read_csv('fft_results.csv')
        else:
            print("❌ Error: No data found! Please run the analysis script first.")
            return pd.DataFrame()
    else:
        df_all = pd.concat(dfs, ignore_index=True)

    # 排序
    df_all = df_all.sort_values(by='Position_mm')
    
    # 【修复关键点】: 无论读哪个文件，都要执行这一步单位换算！
    # 转换单位：Pixel -> um
    if 'Width_um' not in df_all.columns:
        if 'FFT_Width' in df_all.columns:
            df_all['Width_um'] = df_all['FFT_Width'] * PIXEL_TO_UM
        else:
            print("❌ Error: Column 'FFT_Width' missing in csv.")
            return pd.DataFrame()
        
    return df_all

# ==============================================================================
# 2. 绘图主程序 (只画图 1)
# ==============================================================================
def plot_linear_fit_figure():
    set_pub_style()
    
    # 1. 加载数据
    df_all = load_data_from_csvs()
    if df_all.empty: return

    # 2. 筛选 185-195mm
    df_subset = df_all[(df_all['Position_mm'] >= TARGET_RANGE[0]) & (df_all['Position_mm'] <= TARGET_RANGE[1])]
    
    if df_subset.empty:
        print(f"❌ Error: No data found in range {TARGET_RANGE[0]}-{TARGET_RANGE[1]} mm.")
        return
    
    print(f"✅ Data filtered: {len(df_subset)} points in range {TARGET_RANGE}.")

    # 3. 线性拟合 y = Ax + B
    def linear_func(x, A, B): return A * x + B
    
    try:
        # 执行拟合
        popt, _ = curve_fit(linear_func, df_subset['Position_mm'], df_subset['Width_um'])
        
        # 计算 R²
        y_pred = linear_func(df_subset['Position_mm'], *popt)
        ss_res = np.sum((df_subset['Width_um'] - y_pred)**2)
        ss_tot = np.sum((df_subset['Width_um'] - df_subset['Width_um'].mean())**2)
        r2 = 1 - (ss_res / ss_tot)
        
        print(f"✅ Fitting result: y = {popt[0]:.4f}x + {popt[1]:.4f}, R2 = {r2:.6f}")

        # ======================================================================
        # 开始绘图
        # ======================================================================
        fig, ax = plt.subplots(figsize=(5.5, 4)) 
        
        # 绘制散点
        ax.scatter(df_subset['Position_mm'], df_subset['Width_um'], 
                   color='black', marker='o', s=25, 
                   edgecolors='black', facecolors='none', 
                   label='Raw Data')
        
        # 绘制拟合线
        x_line = np.linspace(TARGET_RANGE[0], TARGET_RANGE[1], 300)
        y_line = linear_func(x_line, *popt)
        ax.plot(x_line, y_line, color='#D62728', linestyle='-', linewidth=2, label='Linear Fitting')
        
        # 添加文本框 (左下角)
        text_str = f'$w = {popt[0]:.4f}d_0 + {popt[1]:.4f}$\n$R^2 = {r2:.4f}$'
        
        ax.text(0.05, 0.05, text_str, transform=ax.transAxes, fontsize=11, 
                va='bottom', ha='left', 
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        # 设置轴标签
        ax.set_xlabel(r'$d_0$ / mm')
        ax.set_ylabel(r'$w$ / $\mathrm{\mu m}$')
        
        # 设置范围和刻度
        margin = 0.5 
        ax.set_xlim(TARGET_RANGE[0] - margin, TARGET_RANGE[1] + margin)
        
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        # 图例
        ax.legend(loc='upper right', frameon=False, fontsize=10)
        
        plt.tight_layout()
        save_name = 'Fig1_LinearFit_Only.png'
        plt.savefig(save_name, dpi=600)
        print(f"✅ Generated: {save_name}")
        plt.show()

    except Exception as e:
        print(f"❌ Fitting failed: {e}")

if __name__ == '__main__':
    plot_linear_fit_figure()