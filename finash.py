import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
#📊 优化后 MAE: 0.026678 (原单位)
# ================= 激进配置 =================
BATCH_SIZE = 64          # 加大 Batch Size，梯度更稳
LEARNING_RATE = 0.002    # 稍微调大初始学习率，配合余弦退火
EPOCHS = 150             # 多跑一会，反正模型小
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = 'dataset_distill.npz'
# ===========================================

# 1. 数据集 (新增：数据增强)
class WaveformDataset(Dataset):
    def __init__(self, npz_path, is_train=False):
        data = np.load(npz_path)
        self.X = torch.from_numpy(data['X']).float().permute(0, 2, 1)
        raw_y = data['y']
        
        self.y_min = raw_y.min()
        self.y_max = raw_y.max()
        self.y_range = self.y_max - self.y_min
        y_norm = (raw_y - self.y_min) / self.y_range
        self.y = torch.from_numpy(y_norm).float().unsqueeze(1)
        
        self.is_train = is_train # 标记是否是训练集

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        # ★★★ 核心优化：在线数据增强 ★★★
        # 只有训练时才加噪声，强迫模型学波形结构
        if self.is_train:
            # 加入随机高斯噪声 (模拟传感器底噪)
            noise = torch.randn_like(x) * 0.005 
            x = x + noise
            
        return x, y
    
    def denormalize(self, y_pred_norm):
        return y_pred_norm * self.y_range + self.y_min

# 2. 模型升级：ResBlock 结构
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 如果输入输出维度不匹配，用 1x1 卷积调整维度 (Skip Connection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # ★★★ 残差连接：x + F(x)
        out = self.relu(out)
        return out

class ResLightWaveNet(nn.Module):
    def __init__(self):
        super(ResLightWaveNet, self).__init__()
        # 初始层
        self.pre_layer = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1)
        )
        
        # 残差堆叠层 (类似于 ResNet18 的简化版)
        self.layer1 = self._make_layer(32, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def _make_layer(self, in_c, out_c, blocks, stride):
        layers = []
        layers.append(ResBlock(in_c, out_c, stride))
        for _ in range(1, blocks):
            layers.append(ResBlock(out_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

def train():
    print(f"🚀 使用设备: {DEVICE} | 模式: ResNet + AdamW + SmoothL1")
    
    # 分别创建 dataset，训练集开启增强
    full_dataset = WaveformDataset(DATA_PATH, is_train=False) 
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # 这里的 trick 是手动 split，让 train set 开启 is_train=True
    # 为了简单，我们先 split index，再实例化
    indices = torch.randperm(len(full_dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_set = torch.utils.data.Subset(WaveformDataset(DATA_PATH, is_train=True), train_indices)
    val_set = torch.utils.data.Subset(WaveformDataset(DATA_PATH, is_train=False), val_indices)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    
    model = ResLightWaveNet().to(DEVICE)
    
    # ★★★ 核心优化：Loss 和 Optimizer ★★★
    criterion = nn.SmoothL1Loss() # 比 MSE 更稳
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # 更好的正则化
    
    # 使用余弦退火学习率 (Cosine Annealing) - 跑很多轮时效果最好
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    print("\nStarting Optimized Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_set)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_set)
        
        scheduler.step() # 更新学习率
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if (epoch+1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{EPOCHS}] Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.6f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_optimized.pth')

    # --- 最终评估 ---
    print("\n🏁 正在评估最优模型...")
    model.load_state_dict(torch.load('best_model_optimized.pth'))
    model.eval()
    errors = []
    
    # 用验证集算 MAE
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            
            # 反归一化
            real_pred = full_dataset.denormalize(pred.cpu())
            real_y = full_dataset.denormalize(y.cpu())
            
            diff = torch.abs(real_pred - real_y).numpy()
            errors.extend(diff)
            
    mae = np.mean(errors)
    print(f"📊 优化后 MAE: {mae:.6f} (原单位)")
    print(f"   (之前的记录是 0.0264，看看这次能不能破纪录！)")
    
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.yscale('log')
    plt.title(f'Optimized Training (Final MAE: {mae:.5f})')
    plt.savefig('training_curve_opt.png')

if __name__ == '__main__':
    train()