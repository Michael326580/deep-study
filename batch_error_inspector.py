import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# 0. 全局设置
#生成所有的异常图像波形
# ==============================================================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def batch_inspect_errors(data_folder):
    print("🚀 开始执行【批量错误波形诊断】...")
    print(f"📂 数据源: {data_folder}")
    
    tasks = []
    
    # 1. 读取异常点 (Outliers) - 由 fit_hyperbola_diagnostic 生成
    if os.path.exists('outliers_report.csv'):
        df_out = pd.read_csv('outliers_report.csv')
        df_out['Type'] = '异常点 (Outlier)'
        tasks.append(df_out[['Filename', 'Position_mm', 'Type']])
        print(f"📋 载入 IQR 异常名单: {len(df_out)} 个")
    
    # 2. 读取失败点 (Failed) - 由修改后的主程序生成
    if os.path.exists('failed_report.csv'):
        df_fail = pd.read_csv('failed_report.csv')
        df_fail['Type'] = '计算失败 (Failed)'
        tasks.append(df_fail[['Filename', 'Position_mm', 'Type']])
        print(f"📋 载入计算失败名单: {len(df_fail)} 个")
    
    if not tasks:
        print("✅ 太棒了！没有发现任何异常或失败报告（或者您还没运行主程序的诊断步骤）。")
        return

    all_tasks = pd.concat(tasks, ignore_index=True)
    
    # 创建保存目录
    save_dir = 'Error_Waveforms_Dataset2' # 改个名字区分一下
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"🔍 准备生成 {len(all_tasks)} 张诊断图，保存至 {save_dir}/ ...\n")

    count = 0
    for idx, row in all_tasks.iterrows():
        filename = str(row['Filename'])
        pos = row['Position_mm']
        err_type = row['Type']
        
        file_path = os.path.join(data_folder, filename)
        
        if not os.path.exists(file_path):
            print(f"⚠️ 文件丢失: {filename}")
            continue
            
        try:
            # 读取数据
            try: df = pd.read_csv(file_path)
            except: df = pd.read_csv(file_path, engine='python')
            
            c1 = pd.to_numeric(df['Channel 1'], errors='coerce')
            c2 = pd.to_numeric(df['Channel 2'], errors='coerce')
            
            # 绘图
            fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            fig.suptitle(f"{err_type}\n文件: {filename} (位置: {pos} mm)", fontsize=14, color='red', fontweight='bold')
            
            ax[0].plot(c1, color='orange', label='Channel 1')
            ax[0].set_ylabel('Intensity')
            ax[0].legend(loc='upper right')
            ax[0].grid(True, alpha=0.3)
            ax[0].text(0.02, 0.85, f"Max: {c1.max()}", transform=ax[0].transAxes, color='red')
            
            ax[1].plot(c2, color='royalblue', label='Channel 2')
            ax[1].set_ylabel('Intensity')
            ax[1].set_xlabel('Sample Index')
            ax[1].legend(loc='upper right')
            ax[1].grid(True, alpha=0.3)
            ax[1].text(0.02, 0.85, f"Max: {c2.max()}", transform=ax[1].transAxes, color='red')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{pos}mm_{filename[:-4]}.png"))
            plt.close()
            count += 1
            
        except Exception as e:
            print(f"❌ 画图失败 {filename}: {e}")

    print(f"\n✅ 完成！已生成 {count} 张图片。")

if __name__ == '__main__':
    # ⚠️ 适配您的第二个数据集路径
    data_folder_new = r'D:\桌面\new learning\069date\1.22'
    
    batch_inspect_errors(data_folder_new)