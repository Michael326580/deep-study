# =========================================================
# [脚本说明]
# 用途：最简线性拟合绘图脚本（`sklearn` 版本）。
# 输入：1. 读取 CSV 并取 `Position_mm` 与 `FFT_Width`。
# 输出：2. `LinearRegression.fit` 得到斜率截距。
# =========================================================

# =========================================================
# [????]
# ??????????????sklearn ????
# ???fft_results_robust.csv?
# ???final_linear_fitting.png?
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ==============================================================================
# 0. 参数设置
# ==============================================================================
# 输入文件 (对应刚才生成的清洗后的文件)
INPUT_CSV = 'fft_results_robust.csv' 
# INPUT_CSV = 'fixed_results.csv' # 如果你用的是修复脚本生成的，解开这行注释

# 图片保存路径
OUTPUT_IMG = 'final_linear_fitting.png'

# 绘图设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.style.use('seaborn-v0_8-paper') # 使用科研风格底图 (如果没有这个style，可以注释掉)

# ==============================================================================
# 1. 读取数据
# ==============================================================================
try:
    df = pd.read_csv(INPUT_CSV)
    print(f"成功读取文件: {INPUT_CSV}, 共 {len(df)} 行数据。")
except FileNotFoundError:
    print(f"❌ 找不到文件: {INPUT_CSV}")
    print("请先运行之前的【全量智能预处理脚本】生成这个文件！")
    exit()

# 提取 X 和 Y
X = df['Position_mm'].values.reshape(-1, 1) # 物理位置
y = df['FFT_Width'].values                  # 算法算出的宽度

# ==============================================================================
# 2. 线性拟合
# ==============================================================================
reg = LinearRegression()
reg.fit(X, y)

k = reg.coef_[0]      # 斜率
b = reg.intercept_    # 截距
y_pred = reg.predict(X)
r2 = r2_score(y, y_pred) # R平方

equation_text = f"$y = {k:.4f}x + {b:.4f}$"
r2_text = f"$R^2 = {r2:.6f}$"

print(f"拟合结果: {equation_text}, {r2_text}")

# ==============================================================================
# 3. 绘制图像
# ==============================================================================
plt.figure(figsize=(10, 6), dpi=150)

# 1. 画散点 (原始数据)
plt.scatter(X, y, color='#1f77b4', alpha=0.6, s=15, label='测量数据 (已清洗)')

# 2. 画拟合线
plt.plot(X, y_pred, color='#d62728', linewidth=2, linestyle='--', label='线性拟合')

# 3. 装饰图表
plt.title('大焦距线性锥光全息测量系统 - 线性度分析', fontsize=14, pad=15)
plt.xlabel('物理位置 / Physical Position (mm)', fontsize=12)
plt.ylabel('条纹半波长宽度 / Half-wavelength Width (Pixel)', fontsize=12)

# 4. 添加网格
plt.grid(True, linestyle='--', alpha=0.5)

# 5. 添加图例框 (显示公式)
from matplotlib.offsetbox import AnchoredText
legend_text = f"拟合方程:\n{equation_text}\n\n线性度:\n{r2_text}"
at = AnchoredText(legend_text, prop=dict(size=11), frameon=True, loc='lower left')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.gca().add_artist(at)

plt.legend(loc='upper right', fontsize=11)

# ==============================================================================
# 4. 保存与显示
# ==============================================================================
plt.tight_layout()
plt.savefig(OUTPUT_IMG)
print(f"🎉 图片已保存至: {OUTPUT_IMG}")
plt.show()