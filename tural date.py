# =========================================================
# [脚本说明]
# 用途：绘制新数据集“原始散点图”（不拟合不修饰）。
# 输入：1. 读取并按位置排序。
# 输出：2. 仅绘制 `Position_mm` vs `FFT_Width` 黑色散点。
# =========================================================

# =========================================================
# [????]
# ????????????????????
# ???fft_results.csv?
# ???raw_scatter_new.png?
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# 0. 全局设置
# ==============================================================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['font.size'] = 12  # 字号设置

def plot_raw_scatter_new():
    print("🚀 开始绘制【新数据集 - 原始纯净散点图】...")
    
    # --- 1. 读取数据 ---
    # 直接读取 fft_results.csv，这是最原始的计算结果总表
    # 它包含了所有算出来的点（含离群点），但不含计算失败（NaN）的点
    file_path = 'fft_results.csv'
    
    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到 {file_path}")
        print("   请确保您已经运行了主程序 (main_analysis.py) 对新数据集进行处理。")
        return

    df = pd.read_csv(file_path)
    
    # 按位置排序，画图顺序更顺畅
    if 'Position_mm' in df.columns:
        df = df.sort_values(by='Position_mm')
    
    print(f"📋 数据加载完成，共计 {len(df)} 个有效数据点。")
    print("   (已自动忽略计算失败的点，只展示有数值的点)")

    # --- 2. 绘图 (单图结构) ---
    plt.figure(figsize=(12, 8))
    
    # 核心绘图逻辑：
    # c='black': 黑色，代表原始未处理
    # alpha=0.6: 设置透明度，观察是否有数据点重叠
    plt.scatter(df['Position_mm'], df['FFT_Width'], 
                c='black', marker='o', s=25, alpha=0.6, label='原始测量数据')

    # --- 3. 装饰与美化 ---
    plt.title("新数据集：原始测量数据分布 (Raw Data Scatter)", fontsize=16, fontweight='bold')
    plt.xlabel("物理位置 (mm)", fontsize=14)
    plt.ylabel("半波长宽度 (Pixel)", fontsize=14)
    
    # 加上网格
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 自动调整 Y 轴范围，留出一点边距
    if not df.empty:
        y_vals = df['FFT_Width']
        y_min, y_max = y_vals.min(), y_vals.max()
        margin = (y_max - y_min) * 0.05 if y_max != y_min else 1.0
        plt.ylim(y_min - margin, y_max + margin)
        
        # 自动调整 X 轴范围
        x_vals = df['Position_mm']
        x_min, x_max = x_vals.min(), x_vals.max()
        x_margin = (x_max - x_min) * 0.02
        plt.xlim(x_min - x_margin, x_max + x_margin)

    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # 保存
    save_name = 'raw_scatter_new.png'
    plt.savefig(save_name, dpi=300)
    print(f"\n✅ 图片已生成: {save_name}")
    print("   (这是一张没有任何修饰、没有拟合线、没有叉号的纯净数据图)")
    # plt.show() # 如果在服务器运行请注释此行

if __name__ == '__main__':
    plot_raw_scatter_new()