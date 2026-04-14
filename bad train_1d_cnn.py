import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
#📊 优化后 MAE: 记录是 0.0264
# ================= 配置 =================
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100  # 增加轮数，让它学透
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = 'dataset_distill.npz'
# =======================================

class WaveformDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = torch.from_numpy(data['X']).float().permute(0, 2, 1)
        raw_y = data['y']
        
        # ★★★ 核心修改：标签归一化 ★★★
        self.y_min = raw_y.min()  
        self.y_max = raw_y.max()
        self.y_range = self.y_max - self.y_min
        
        # 把 y 压缩到 [0, 1]
        y_norm = (raw_y - self.y_min) / self.y_range
        
        self.y = torch.from_numpy(y_norm).float().unsqueeze(1)
        print(f"标签已归一化: Min={self.y_min:.2f}, Max={self.y_max:.2f}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    # 辅助函数：把预测出的 0-1 变回 mm
    def denormalize(self, y_pred_norm):
        return y_pred_norm * self.y_range + self.y_min

# 模型保持不变 (LightWaveNet)
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
            nn.Dropout(0.1), # 稍微降低 Dropout
            nn.Linear(64, 1) # 输出 0-1 之间的值
        )

    def forward(self, x):
        return self.regressor(self.features(x))

def train():
    print(f"🚀 使用设备: {DEVICE}")
    dataset = WaveformDataset(DATA_PATH)
    
    total_len = len(dataset)
    train_len = int(total_len * 0.8)
    val_len = total_len - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    
    model = LightWaveNet().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ★★★ 新增：学习率衰减 (如果 Loss 不降了，自动把学习率除以 10) ★★★
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    print("\nStarting Training (V2 - Normalized)...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= train_len
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= val_len
        
        # 更新学习率
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_v2.pth')
            
    # 评估真实误差 (还原回 mm)
    model.load_state_dict(torch.load('best_model_v2.pth'))
    model.eval()
    errors = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            
            # 反归一化：把 0-1 变回 mm
            real_preds = dataset.denormalize(outputs.cpu())
            real_targets = dataset.denormalize(y_batch.cpu())
            
            diff = torch.abs(real_preds - real_targets).numpy()
            errors.extend(diff)
    
    mae = np.mean(errors)
    print(f"\n📊 最终测试集平均误差 (MAE): {mae:.4f} mm")
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.yscale('log') # 用对数坐标看 Loss，更清晰
    plt.title('Training Convergence (Normalized)')
    plt.savefig('training_curve_v2.png')

if __name__ == '__main__':
    train()