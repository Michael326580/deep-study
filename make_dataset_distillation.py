# =========================================================
# [脚本说明]
# 用途：从教师结果和原始波形生成蒸馏训练集 `dataset_distill.npz`。
# 输入：1. 读取教师 CSV，建立 `Filename -> FFT_Width` 映射。
# 输出：2. 遍历原始 CSV，仅处理有教师标签的文件。
# =========================================================

# =========================================================
# [????]
# ?????????????
# ???fft_results_robust.csv + ???????
# ???dataset_distill.npz?
# =========================================================

import pandas as pd
import numpy as np
import os
#读取老师数据去做数据集：对应fft_results_robust.csv里的高精度结果，做成dataset_distill.npz
# ================= 配置 =================
RAW_DATA_FOLDER = r'D:\桌面\new learning\069date\1.22'  # 原始波形文件夹
LABEL_CSV = 'fft_results_robust.csv'                 # PCCC 算出的高精度结果
SAVE_PATH = 'dataset_distill.npz'                    # 新数据集名称
SEQ_LEN = 2048
# =======================================

def make_distilled_dataset():
    # 1. 加载“老师”的答案 (PCCC High Precision Results)
    if not os.path.exists(LABEL_CSV):
        print(f"❌ 错误：找不到 {LABEL_CSV}。请先运行之前的 robust 算法脚本生成它！")
        return

    teacher_df = pd.read_csv(LABEL_CSV)
    # 建立一个字典映射: Filename -> High_Precision_Width
    # 这样读取波形时，能直接查到它对应的精确宽度
    label_map = dict(zip(teacher_df['Filename'], teacher_df['FFT_Width']))
    
    print(f"📖 已加载“老师”的答案，共 {len(label_map)} 个文件的精确指标。")

    X_list = []
    y_list = []
    
    files = [f for f in os.listdir(RAW_DATA_FOLDER) if f.endswith('.csv')]
    
    print("🚀 开始制作蒸馏数据集...")
    
    for i, fname in enumerate(files):
        # 如果这个文件之前的算法都算挂了（没在csv里），那 AI 也不要学它（自动清洗）
        if fname not in label_map:
            continue
            
        target_width = label_map[fname] # 获取高精度浮点标签
        
        try:
            path = os.path.join(RAW_DATA_FOLDER, fname)
            try: df = pd.read_csv(path)
            except: df = pd.read_csv(path, engine='python')
            
            c1 = df['Channel 1'].values
            c2 = df['Channel 2'].values
            
            # 切分波形 (逻辑同前)
            starts = np.where((c1 == 0) & (c2 == 0))[0]
            ends = np.where((c1 == 0) & (c2 == 61680))[0]
            
            for s_idx in starts:
                possible_ends = ends[ends > s_idx]
                if len(possible_ends) > 0:
                    e_idx = possible_ends[0]
                    length = e_idx - s_idx
                    
                    if length > 2000: # 稍微严格一点
                        seg_c1 = c1[s_idx+1 : e_idx]
                        seg_c2 = c2[s_idx+1 : e_idx]
                        
                        # 标准化
                        sample = np.zeros((SEQ_LEN, 2))
                        curr_len = min(len(seg_c1), SEQ_LEN)
                        sample[:curr_len, 0] = seg_c1[:curr_len]
                        sample[:curr_len, 1] = seg_c2[:curr_len]
                        sample = sample / 16383.0 
                        
                        X_list.append(sample)
                        y_list.append(target_width) # ★★★ 这里存的是精确宽度，不是文件名
            
        except Exception as e:
            print(f"Error in {fname}: {e}")

        if (i+1) % 100 == 0: print(f"进度: {i+1}/{len(files)}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    print(f"\n✅ 蒸馏数据集制作完成！")
    print(f"样本数: {len(X)}")
    print(f"标签示例 (Width): {y[:5]}") # 看看是不是浮点数
    
    np.savez(SAVE_PATH, X=X, y=y)
    print(f"已保存至 {SAVE_PATH}")

if __name__ == '__main__':
    make_distilled_dataset()