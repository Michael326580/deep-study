# deep-study 项目代码分析报告

## 1. 项目总体结论

该仓库是一个围绕“条纹波形测量与位移反演”的实验型研究代码库，主线流程可以概括为：

1. 从原始双通道波形 CSV（`Channel 1/2`）中按标记点切段；
2. 使用互相关 + FFT 提取半波长宽度（`FFT_Width`）作为“教师标签”；
3. 将段样本蒸馏为 `dataset_distill.npz`；
4. 用 1D-CNN/ResNet 风格学生网络拟合教师标签；
5. 通过双曲线标定函数实现宽度到位移的反演，并输出误差图与对比图。

仓库中存在明显的“迭代脚本堆叠”特征：同类逻辑在多个脚本中重复实现（如 `xxx.py`、`zippppp.py`、`finash.py`、`bad train_1d_cnn.py`），文件命名也带有实验草稿属性，说明代码主要服务于快速试验，而非工程化产品。

## 2. 目录与模块结构梳理

### 2.1 数据与产物

- 原始数据目录：`1.22/`（大量按数值命名的 CSV，含负号文件名）。
- 训练与中间产物：`dataset_distill.npz`、`fft_results_robust.csv`、`grouped_dataset_metadata.csv`。
- 训练权重：`best_model_v2.pth`、`best_model_optimized.pth`、`best_model_grouped_resnet.pth`。
- 图表产物：大量 `*.png/*.pdf/*.bmp`，包含线性拟合图、诊断图、训练曲线、误差图。

### 2.2 核心脚本分层（按职责）

- **数据清洗与教师标签生成**：`finaish_end.py`、`error_solve.py`、`185-195mmnihe.py`。
- **数据集蒸馏**：`make_dataset_distillation.py`。
- **模型训练（学生网络）**：`bad train_1d_cnn.py`（早期）、`finash.py`（改进）、`xxx.py`（分组验证版）。
- **模型评估与对比**：`verify_accuracy.py`、`zippppp.py`、`plot_teacher_student.py`。
- **论文/展示绘图**：`TIM-picture1.py`、`TIM-picture2.py`、`xxxxxtest.py`、`picture*.py`、`fixed date picture.py`。
- **问题排查工具**：`outlier_inspector.py`、`batch_error_inspector.py`、`test phase lead.py`、`changshi.py`。

## 3. 核心流程与主要 Python 文件逻辑

## 3.1 教师标签提取链路

### `finaish_end.py`

- 使用 `estimate_period` 预估主周期，再在 `±0.6T` 内做互相关寻优，避免相位跳周期误匹配。
- `process_single_file_mlv` 对每个文件做两级清洗：
  - 基于段长度中位数的容差过滤（`LENGTH_TOLERANCE`）；
  - 基于宽度中位数的数值容差过滤（`VALUE_TOLERANCE`）。
- `process_batch_all` 全量跑数据并输出 `fft_results_robust.csv` 与 `failed_report.csv`。
- `fit_hyperbola_diagnostic` 做双曲线稳健拟合、IQR 剔离群并产出 `outliers_report.csv` 与诊断图。

**评价**：这是仓库中最完整、最接近“主流程入口”的教师标签生产脚本。

### `error_solve.py`

- 面向失败/离群样本二次修复：读取 `failed_report.csv` + `outliers_report.csv`，针对问题文件重算宽度。
- 使用“长度容差 + 中位数近邻值”机制做局部恢复，并尝试与旧结果合并成 `fixed_results.csv`。

**评价**：思路实用，但与主流程耦合强，且依赖外部报告文件存在，缺少统一的 pipeline 编排。

### `185-195mmnihe.py`

- 专注 185–195 mm 局部范围筛选、宽度提取与线性拟合，输出带方程/`R²` 标注图。
- 适合做局部线性区间分析，不是全量处理脚本。

## 3.2 数据集构建与学生网络训练

### `make_dataset_distillation.py`

- 根据 `fft_results_robust.csv` 的 `Filename -> FFT_Width` 映射，遍历原始 CSV 切段。
- 将段对齐/截断到 `2048x2` 并按 `16383` 归一化，保存成 `dataset_distill.npz`。

**评价**：实现直接、可用；但路径强绑定到本地 Windows 绝对路径，不利复现。

### `bad train_1d_cnn.py`

- 使用较浅 `LightWaveNet`，标签归一化到 `[0,1]`。
- `random_split` 做训练/验证，`ReduceLROnPlateau` 控学习率。

**评价**：能跑通，但样本切分按 segment 随机，存在同文件泄漏到验证集的风险。

### `finash.py`

- 模型升级为残差 1D-CNN（`ResLightWaveNet`），训练期加入高斯噪声增强。
- `AdamW + SmoothL1 + CosineAnnealingLR` 组合较合理。

**评价**：相比早期脚本有明显训练策略改进，但仍沿用随机切分，评估可能偏乐观。

### `xxx.py`（当前最工程化训练脚本）

- 可从解压目录或 zip 自动发现教师 CSV 与原始波形源。
- 重建分段数据集并保存 `grouped_dataset_metadata.csv`。
- 关键改进：`split_by_filename` 按文件级分组切分 train/val，避免 segment 泄漏。
- 训练中同时追踪 segment 与 file 级 MAE；最佳模型按 file-level MAE 保存。
- 训练后执行双曲线拟合与宽度/位移误差对比图输出。

**评价**：这是仓库中最推荐保留并继续演进的训练主干。

## 3.3 评估与可视化

### `verify_accuracy.py`

- 基于 `fft_results_robust.csv` 拟合 `W=1/(Ax+B)+C`，再做反函数位移反演，输出 MAE/RMSE/P99 等。

**评价**：报告维度清晰，但异常处理较粗糙（例如除零直接全数组 NaN）。

### `zippppp.py`

- 在 zip 场景下完成“教师 vs 学生”全流程对比（读取、推理、拟合、绘图、统计）。
- 复用了 `ResLightWaveNet`，输出对比 CSV 与像素/位移误差图。

**评价**：可离线复现实验，但与 `xxx.py` 重复逻辑较多，维护成本高。

### `TIM-picture1.py` / `TIM-picture2.py` / `xxxxxtest.py`

- 主要服务论文图的版式标准（单栏/双栏、字体、残差子图、灰度友好样式等）。

**评价**：图形质量意识较好，但与核心算法混在同目录，建议后续归档到 `scripts/figures/`。

## 4. 代码质量评估

## 4.1 优点

- **算法思路清晰**：互相关对齐 + FFT 主频 + 峰值插值是一条合理的宽度提取路径。
- **鲁棒意识较强**：多处使用 IQR、soft_l1、中位数容差过滤来抑制异常值。
- **实验闭环完整**：从原始数据到模型训练、验证、图表输出链路齐全。

## 4.2 主要问题

1. **命名与结构不规范**：大量文件名不可读（如 `xxx.py`、`finash.py`、`zippppp.py`）。
2. **重复代码严重**：模型结构、指标函数、宽度/反函数逻辑在多个文件重复。
3. **配置硬编码**：存在大量 `D:\桌面\...` 路径与脚本内常量，不可移植。
4. **异常捕获过宽**：大量裸 `except:`，会吞掉关键错误上下文。
5. **工程化不足**：无 `requirements.txt/pyproject.toml`、无统一 CLI、无单元测试。
6. **数据与代码混仓**：大模型权重和大量图片直接进仓库，版本管理成本高。

## 5. 潜在 Bug / 风险点

1. **位置映射规则不一致风险**
   - 多数新数据脚本采用 `-449 -> 180.0mm`，但 `test phase lead.py` 使用 `-300` 映射，可能导致位置标注错误。

2. **标签归一化分母为 0 风险**
   - 多脚本用 `y_range = y_max - y_min`，若标签近乎常数会触发除零（如 `finash.py`、`bad train_1d_cnn.py`）。

3. **`inverse_func` 数值稳定性**
   - 当 `W≈C` 时反演发散或 NaN；部分脚本处理策略不一致，可能导致统计波动。

4. **训练/验证信息泄漏**
   - 旧脚本按 segment 随机拆分（`random_split`），同一文件段可能同时出现在 train 和 val。

5. **重复读取与内存压力**
   - 多脚本全量读入并构建 `X_list`，对更大数据规模会迅速涨内存。

6. **模型加载兼容性**
   - 不同脚本有的保存 `state_dict`，有的保存完整 dict，加载逻辑不统一，易出现 key mismatch。

## 6. 优化建议（按优先级）

## P0（必须先做）

1. **确定唯一主流程脚本**
   - 建议以 `xxx.py` 为基线，拆成模块：
     - `core/signal.py`（切段、对齐、FFT）
     - `core/calibration.py`（双曲线拟合与反演）
     - `core/model.py`（网络结构）
     - `train.py` / `evaluate.py` / `make_dataset.py`

2. **统一配置管理**
   - 把路径、阈值、超参迁移到 `yaml` 或 argparse 参数，禁止硬编码绝对路径。

3. **消除重复实现**
   - 抽公共函数（`width_func/inverse_func/compute_metrics/ResLightWaveNet`）到单一模块，所有脚本 import。

4. **修正映射常量**
   - 将 `FILE_ZERO_INDEX`（如 `-449`）集中定义，避免各脚本散落不同版本。

## P1（重要）

1. **构建最小测试集与自动化测试**
   - 单测目标：
     - 标记点切段是否正确；
     - `calculate_width` 对合成信号误差是否在阈值内；
     - `inverse_func(width_func(x))≈x`。

2. **改进异常处理**
   - 裸 `except:` 替换为 `except Exception as e:` 并记录上下文（文件名、段号、参数）。

3. **提升可复现性**
   - 增加 `requirements.txt`、固定随机种子、记录 git commit + config 到训练输出目录。

4. **规范日志**
   - 用 `logging` 替代 `print`，支持 `INFO/WARNING/ERROR` 级别与文件落盘。

## P2（可持续优化）

1. **性能提升**
   - FFT 与相关计算可批处理或并行（`joblib` / `multiprocessing`）；
   - 对频繁重复读取的 CSV 增加缓存中间结果（如 parquet/npz）。

2. **仓库瘦身**
   - 模型权重与大图迁移到 Release 或对象存储，仓库仅保留轻量样例与脚本。

3. **代码风格统一**
   - 引入 `ruff + black + isort`，并配置 pre-commit。

## 7. 推荐的重构后项目骨架

```text
deep-study/
  pyproject.toml
  requirements.txt
  README.md
  src/deep_study/
    config.py
    io.py
    signal_processing.py
    calibration.py
    dataset.py
    models.py
    train.py
    evaluate.py
    figures.py
  scripts/
    make_dataset.py
    run_train.py
    run_eval.py
    make_figures.py
  tests/
    test_signal_processing.py
    test_calibration.py
    test_dataset_split.py
```

## 8. 结语

这个仓库在“科研可行性验证”层面已经很扎实：算法链路完整、结果产物充分、对异常数据有实际处理经验。下一步的关键不是继续堆脚本，而是把已验证的方法沉淀为模块化、可测试、可复现的工程代码。若完成上述 P0/P1 改造，项目会从“个人实验仓库”跃迁为“可协作的研究工程仓库”。
