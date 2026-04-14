import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置 =================
INPUT_CSV = 'fft_results_robust.csv' 
# =======================================
#反向推算模型（拿尺子量）
#输入：算法算出来的条纹宽度 W (例如 26.5 像素)。
#输出：反推出来的物理位置 x (例如 173.3mm)。
#数学逻辑：它其实就是上面那个 width_func 的反函数。
# 1. 定义物理模型 (双曲线)
def width_func(x, A, B, C):
    return 1.0 / (A * x + B) + C

# 2. 定义反演模型 (宽度 -> 位置)
def inverse_func(W, A, B, C):
    # 避免除零风险
    if np.any(np.abs(W - C) < 1e-9):
        return np.full_like(W, np.nan)
    return (1.0 / (W - C) - B) / A

def main():
    print("🚀 开始执行【全量精度反演验证】...")
    
    # 1. 加载数据
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"📋 已加载 {len(df)} 条测试数据。")
    except:
        print(f"❌ 错误：找不到 {INPUT_CSV}。")
        return

    x_true = df['Position_mm'].values
    w_meas = df['FFT_Width'].values
    
    # 2. 标定模型 (拟合 A, B, C)
    print("⚙️  正在标定反演模型参数...")
    p0 = [0.001, 0.01, 0] 
    try:
        popt, _ = curve_fit(width_func, x_true, w_meas, p0=p0, method='trf', loss='soft_l1', maxfev=10000)
        A, B, C = popt
    except Exception as e:
        print(f"❌ 拟合失败: {e}")
        return
        
    print(f"   标定参数: A={A:.6e}, B={B:.6e}, C={C:.6e}")
    
    # 3. 批量反演 (核心步骤)
    # 用测量出来的宽度 w_meas，反推位置 x_pred
    x_pred = inverse_func(w_meas, A, B, C)
    
    # 4. 计算误差
    # Error = 预测位置 - 真实位置
    errors = x_pred - x_true
    abs_errors = np.abs(errors)
    
    # 5. 生成统计报告
    mae = np.mean(abs_errors)       # 平均绝对误差
    rmse = np.sqrt(np.mean(errors**2)) # 均方根误差
    max_err = np.max(abs_errors)    # 最大误差 (最坏情况)
    p99_err = np.percentile(abs_errors, 99) # 99%的情况下误差小于多少
    
    print("\n" + "="*40)
    print("📊 绝对精度测试报告 (Absolute Accuracy Report)")
    print("="*40)
    print(f"✅ 测试样本数: {len(df)}")
    print(f"✅ 平均误差 (MAE)  : {mae:.5f} mm")
    print(f"✅ 均方根误差 (RMSE): {rmse:.5f} mm")
    print(f"⚠️ 最大误差 (Max)   : {max_err:.5f} mm")
    print(f"🛡️ 99%置信误差     : < {p99_err:.5f} mm")
    print("="*40)
    
    if max_err < 0.05:
        print("🏆 结论：系统精度极高，满足精密加工需求！")
    elif max_err < 0.1:
        print("✨ 结论：系统精度良好，符合一般工业标准。")
    else:
        print("⚠️ 结论：存在个别大误差，建议检查离群点。")

    # 6. 绘图分析
    plt.figure(figsize=(12, 10), dpi=120)
    
    # 子图1: 真实 vs 预测
    plt.subplot(2, 1, 1)
    plt.scatter(x_true, x_pred, s=5, alpha=0.5, c='blue', label='Test Points')
    plt.plot([x_true.min(), x_true.max()], [x_true.min(), x_true.max()], 'r--', lw=2, label='Ideal Line')
    plt.title('True Position vs. Inverted Position', fontsize=12)
    plt.xlabel('True Position (mm)')
    plt.ylabel('Calculated Position (mm)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 子图2: 误差分布 (Error vs Position)
    plt.subplot(2, 1, 2)
    # 画误差散点
    plt.scatter(x_true, errors, s=10, alpha=0.6, c=errors, cmap='coolwarm')
    # 画 ±0.05mm 容差线
    plt.axhline(0.05, color='r', linestyle='--', alpha=0.5, label='±0.05mm Tolerance')
    plt.axhline(-0.05, color='r', linestyle='--', alpha=0.5)
    plt.axhline(0, color='k', alpha=0.8)
    
    plt.title(f'Position Error Distribution (Max: {max_err:.4f} mm)', fontsize=12)
    plt.xlabel('Physical Position (mm)')
    plt.ylabel('Error (mm)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('accuracy_test_report.png')
    print("\n📸 精度分布图已保存至: accuracy_test_report.png")
    plt.show()

if __name__ == '__main__':
    main()