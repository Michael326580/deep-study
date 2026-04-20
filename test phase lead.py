# =========================================================
# [脚本说明]
# 用途：批量统计哪个通道领先（相位先后关系检查）。
# 输入：1. 遍历数值文件并映射位置。
# 输出：2. 选取中间强信号区间并去均值。
# =========================================================

# =========================================================
# [????]
# ??????????????????
# ??????? CSV ???
# ???phase_check_results.csv ? phase_check_report.png?
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import re

# ==============================================================================
# 全局设置
# ==============================================================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def batch_check_phase_lead(folder_path):
    print(f"🕵️‍♂️ 开始批量扫描相位关系: {folder_path}")
    print("目标：统计是 Channel 1 跑在前面，还是 Channel 2 跑在前面？")
    
    all_files = os.listdir(folder_path)
    results = []
    
    count = 0
    for f in all_files:
        # 1. 筛选合法文件
        name_body = f[:-4] if f.lower().endswith('.csv') else f
        if not re.match(r'^-?\d+$', name_body):
            continue
            
        file_num = int(name_body)
        pos = round(180.0 + (file_num - (-300)) * 0.1, 1)
        filepath = os.path.join(folder_path, f)
        
        try:
            # 2. 读取数据
            try: df = pd.read_csv(filepath)
            except: df = pd.read_csv(filepath, engine='python')
            
            if 'Channel 1' not in df.columns: continue
            
            # 3. 截取有效段 (防止首尾0值的干扰)
            df['Channel 1'] = pd.to_numeric(df['Channel 1'], errors='coerce').fillna(0)
            df['Channel 2'] = pd.to_numeric(df['Channel 2'], errors='coerce').fillna(0)
            
            # 找非零区域
            mask = (df['Channel 1'] > 0) & (df['Channel 2'] > 0)
            if not mask.any(): continue
            
            # 只取中间一段最强的信号来分析，避免边缘噪声误导
            valid_idx = df.index[mask]
            center = (valid_idx[0] + valid_idx[-1]) // 2
            span = 500 # 取中间500个点，足够看清相位关系了
            start = max(valid_idx[0], center - span//2)
            end = min(valid_idx[-1], center + span//2)
            
            if end - start < 100: continue
            
            ch1 = df.iloc[start:end]['Channel 1'].values
            ch2 = df.iloc[start:end]['Channel 2'].values
            
            # 4. 预处理 (去直流)
            ch1 = ch1 - np.mean(ch1)
            ch2 = ch2 - np.mean(ch2)
            
            # 5. 互相关计算位移
            # 限制搜索范围在 ±50 像素内，防止周期跳变干扰判断
            # 我们假设物理安装误差不会超过 50 个像素
            limit = 100 
            corr = signal.correlate(ch1, ch2, mode='full')
            lags = signal.correlation_lags(len(ch1), len(ch2), mode='full')
            
            mask_lag = (lags >= -limit) & (lags <= limit)
            
            # 寻找“最负”的点 (因为它们是反相的：峰对谷)
            best_idx = np.argmin(corr[mask_lag])
            shift = lags[mask_lag][best_idx]
            
            # 6. 判定结论
            # Shift > 0: Ch2 滞后 (Ch1 在前)
            # Shift < 0: Ch2 超前 (Ch2 在前)
            if shift > 0:
                status = "Ch1_Leads"
            elif shift < 0:
                status = "Ch2_Leads"
            else:
                status = "Aligned"
                
            results.append({
                'Filename': f,
                'Position_mm': pos,
                'Shift': shift,
                'Status': status
            })
            
            count += 1
            if count % 100 == 0: print(f"已扫描 {count} 个文件...")
            
        except Exception as e:
            continue

    # --- 统计与绘图 ---
    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("❌ 没有提取到有效数据")
        return

    # 1. 打印统计摘要
    print("\n" + "="*40)
    print("📊 相位关系统计报告")
    print("="*40)
    summary = res_df['Status'].value_counts()
    print(summary)
    
    ch1_lead_count = summary.get('Ch1_Leads', 0)
    total = len(res_df)
    ratio = ch1_lead_count / total * 100
    
    print(f"\n👉 结论: {ratio:.1f}% 的文件中，Channel 1 波形都在 Channel 2 前面。")
    print(f"👉 平均位移量: {res_df['Shift'].mean():.2f} 像素")
    print("="*40)
    
    # 2. 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(res_df['Shift'], bins=30, color='#1f77b4', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='0 (完全对齐)')
    plt.title(f"通道相位偏移分布 (Shift Distribution)\nShift > 0 代表 Channel 1 超前", fontsize=14)
    plt.xlabel("位移量 (Pixel)", fontsize=12)
    plt.ylabel("文件数量", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phase_check_report.png', dpi=150)
    print("\n✅ 分布图已生成: phase_check_report.png")
    
    # 保存CSV以便查看个别特例
    res_df.to_csv('phase_check_results.csv', index=False)

if __name__ == '__main__':
    # ⚠️ 修改为你的路径
    data_folder = r'D:\桌面\learning-notes\069date\069date3'
    batch_check_phase_lead(data_folder)