# deep-study

本仓库用于双通道干涉波形的教师算法（FFT/稳健拟合）、学生网络蒸馏训练、误差诊断与论文图表产出。

## 目录结构（已标准化）

```text
.
├── src/                       # 可复用源码（当前预留）
├── scripts/
│   ├── data_processing/       # 数据预处理/教师标签生成
│   ├── training/              # 训练与推理脚本
│   ├── evaluation/            # 误差排查、异常修复、诊断
│   ├── plotting/              # 论文风格图表与主图生成
│   ├── legacy/                # 历史脚本（保留可复现）
│   └── utils/                 # 预留
├── configs/                   # 配置文件（预留）
├── data/
│   ├── raw/                   # 原始采集数据
│   ├── interim/               # 中间数据（预留）
│   └── processed/             # 处理后数据集与标签
├── results/
│   ├── models/                # 训练权重与训练日志
│   ├── figures/               # 常规图像输出
│   ├── diagnostics/           # 诊断图/异常波形图
│   ├── tables/                # 预留
│   └── paper_outputs/         # 论文图表与latex片段
├── docs/                      # 文档
├── tests/                     # 测试（预留）
└── legacy/                    # 顶层历史目录（预留）
```

## 快速开始

> 建议所有脚本均在仓库根目录执行，以保证相对路径一致。

### 1) 教师标签与鲁棒结果

```bash
python scripts/data_processing/pccc_fft_teacher.py
```

默认读取 `data/raw/1.22`，输出：
- `data/processed/fft_results_robust.csv`
- `data/processed/failed_report.csv`
- `data/processed/outliers_report.csv`
- `results/diagnostics/final_diagnostic_plot.png`

### 2) 异常修复

```bash
python scripts/evaluation/error_solve.py
```

读取 `failed/outliers report`，输出 `data/processed/fixed_results.csv`。

### 3) 蒸馏数据集构建

```bash
python scripts/data_processing/make_dataset_distillation.py
```

默认输出：
- `data/processed/dataset_distill.npz`
- `data/processed/dataset_distill_meta.csv`

### 4) 学生网络训练

```bash
python scripts/training/train_attention_res1d_student.py train
```

默认读取 `data/processed/dataset_distill.npz` 与 `data/processed/grouped_dataset_metadata.csv`，输出到 `results/models/runs/attention_res1d`。

### 5) 主图/论文图

```bash
python scripts/plotting/prepare_mst_main_figures.py
```

输出到 `results/paper_outputs`。

## 脚本清单

详见：`docs/SCRIPT_AUDIT_AND_DEPENDENCY_CN.md`（逐脚本功能、输入输出、关系与分类）。