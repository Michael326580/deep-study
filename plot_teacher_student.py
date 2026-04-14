import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
#做图反映最终结果的脚本
# ================= 配置 =================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = 'dataset_distill.npz'
MODEL_PATH = 'best_model_v2.pth' # 确保这是您刚才训练出的那个最优模型
BATCH_SIZE = 64
# =======================================

# 1. 模型定义 (保持不变)
class LightWaveNet(nn.Module):
    def __init__(self):
        super(LightWaveNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1) 
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.regressor(self.features(x))

class WaveformDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = torch.from_numpy(data['X']).float().permute(0, 2, 1)
        raw_y = data['y']
        
        self.y_min = raw_y.min()
        self.y_max = raw_y.max()
        self.y_range = self.y_max - self.y_min
        
        y_norm = (raw_y - self.y_min) / self.y_range
        self.y = torch.from_numpy(y_norm).float().unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def denormalize(self, y_pred_norm):
        return y_pred_norm * self.y_range + self.y_min

def plot_paper_figures():
    print("🚀 正在生成最终论文插图...")
    
    # 加载数据和模型
    dataset = WaveformDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = LightWaveNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except:
        # 如果是 weights_only 问题，尝试加参数
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            
            real_preds = dataset.denormalize(outputs.cpu())
            real_targets = dataset.denormalize(y_batch.cpu())
            
            y_pred.extend(real_preds.numpy().flatten())
            y_true.extend(real_targets.numpy().flatten())
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    errors = y_pred - y_true
    
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(errors))

    # --- 图1: 师徒相关性 (Correlation Plot) ---
    plt.figure(figsize=(7, 6), dpi=300)
    plt.scatter(y_true, y_pred, alpha=0.5, s=10, color='#1f77b4', edgecolors='none', label='Test Samples')
    
    # 画理想对角线
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Identity')
    
    plt.title(f'Prediction Consistency\n$R^2 = {r2:.5f}, MAE = {mae:.4f}$', fontsize=14)
    plt.xlabel('PCCC Calculated Width (Teacher)', fontsize=12)
    plt.ylabel('LightWaveNet Predicted Width (Student)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('Fig_Correlation.png')
    print("✅ 图1已保存: Fig_Correlation.png")

    # --- 图2: 误差分布直方图 (Error Histogram) ---
    plt.figure(figsize=(7, 4), dpi=300)
    # 画直方图
    plt.hist(errors, bins=50, color='#2ca02c', alpha=0.7, rwidth=0.9, density=True)
    
    # 标注统计信息
    mu = np.mean(errors)
    sigma = np.std(errors)
    plt.title(f'Error Distribution ($\mu={mu:.4f}, \sigma={sigma:.4f}$)', fontsize=14)
    plt.xlabel('Prediction Error (Pixel)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('Fig_ErrorDist.png')
    print("✅ 图2已保存: Fig_ErrorDist.png")

if __name__ == '__main__':
    plot_paper_figures()