# 脚本全量审计与依赖关系（基于仓库真实代码）

> 审计范围：`scripts/**/*.py` 共 13 个脚本。

## 1. 分类总览

| 脚本 | 归类 | 说明 |
|---|---|---|
| `scripts/data_processing/pccc_fft_teacher.py` | 数据处理 + 评估 | 主教师流程：批处理宽度提取 + 双曲线拟合 + 离群/失败报告 |
| `scripts/data_processing/make_dataset_distillation.py` | 数据处理 | 将原始波形与 teacher 标签打包为蒸馏训练集 `.npz` |
| `scripts/training/train_attention_res1d_student.py` | 训练 + 推理 | 1D-CNN + SE 注意力学生网络，支持 `train/infer` 子命令 |
| `scripts/evaluation/error_solve.py` | 评估 | 对失败/离群文件做二次修复并合并结果 |
| `scripts/evaluation/error_inspector.py` | 评估 + 画图 | 批量输出异常波形诊断图 |
| `scripts/evaluation/outlier_inspector.py` | 评估 + 画图 | 生成 5 子图全链路诊断 |
| `scripts/plotting/prepare_mst_main_figures.py` | 画图 + 评估汇总 | 生成论文主图、表格、latex 插入片段 |
| `scripts/plotting/picture4-produce.py` | 画图 | IEEE 风格双曲线拟合图 |
| `scripts/plotting/picture5-produce.py` | 画图 | IEEE 风格原始双通道波形段图 |
| `scripts/plotting/picture6-produce.py` | 画图 | 最终线性拟合图（优先 fixed results） |
| `scripts/legacy/picture1-produce.py` | 历史遗留（画图） | 早期线性拟合图 |
| `scripts/legacy/picture2-produce.py` | 历史遗留（画图） | 早期波形图生成脚本 |
| `scripts/legacy/picture3-produce.py` | 历史遗留（评估/画图） | sklearn 版线性拟合 |
| `scripts/legacy/185-195mmnihe.py` | 历史遗留（评估/画图） | 185-195mm 区间局部拟合 |

## 2. 逐脚本说明（功能 / 输入 / 输出 / 关系）

## 2.1 `scripts/data_processing/pccc_fft_teacher.py`
- **功能**：
  - 扫描原始 CSV；按 `(0,0)` 到 `(0,61680)` 标记切段。
  - 采用“平滑 + 去均值 + 周期约束互相关 + FFT 峰值插值”计算宽度。
  - 进行两级清洗（长度中位数容差、数值中位数容差）。
  - 输出 robust 结果与失败报告，并做双曲线拟合、IQR 离群分析与诊断图。
- **输入**：`data/raw/1.22/*.csv`。
- **输出**：
  - `data/processed/fft_results_robust.csv`
  - `data/processed/failed_report.csv`
  - `data/processed/outliers_report.csv`
  - `results/diagnostics/final_diagnostic_plot.png`
- **关系**：
  - 被 `error_solve.py` / `error_inspector.py` / `outlier_inspector.py` 消费其报告与结果。

## 2.2 `scripts/evaluation/error_solve.py`
- **功能**：
  - 读取 `failed_report` + `outliers_report`，对对应文件重算宽度并做段级过滤。
  - 合并原始成功结果，生成修复后的 `fixed_results.csv`。
- **输入**：
  - `data/processed/failed_report.csv`
  - `data/processed/outliers_report.csv`
  - `data/raw/1.22/*.csv`
  - （可选）`data/processed/fft_results.csv`（原脚本兼容路径）
- **输出**：`data/processed/fixed_results.csv`
- **关系**：
  - 其输出可被 `make_dataset_distillation.py`、`picture6-produce.py`、legacy 画图脚本使用。

## 2.3 `scripts/data_processing/make_dataset_distillation.py`
- **功能**：
  - 将 teacher 标签（`Filename`,`FFT_Width`）与原始波形做交集匹配。
  - 逐文件提取有效段，构造成定长 `(seq_len,2)` 样本，生成蒸馏数据。
- **输入**：
  - `data/raw/1.22/*.csv`
  - `data/processed/fixed_results.csv`
- **输出**：
  - `data/processed/dataset_distill.npz`
  - `data/processed/dataset_distill_meta.csv`
- **关系**：
  - 被训练脚本 `train_attention_res1d_student.py` 消费。

## 2.4 `scripts/training/train_attention_res1d_student.py`
- **功能**：
  - 子命令 `train`：文件级分组切分、训练 SE-Res1D 学生网络、保存 ckpt/history/summary。
  - 子命令 `infer`：加载 checkpoint 并对数据集推理导出 CSV。
- **输入**：
  - `data/processed/dataset_distill.npz`
  - `data/processed/grouped_dataset_metadata.csv`（或 fallback 到 `dataset_distill_meta.csv`）
- **输出**（默认）：`results/models/runs/attention_res1d/*`
  - `best_attention_res1d.pth`, `last_attention_res1d.pth`
  - `train_history.csv`, `train_summary.json`, `best_val_predictions.csv`
- **关系**：
  - 下游可被 `prepare_mst_main_figures.py` 等统计脚本用于比较。

## 2.5 `scripts/evaluation/error_inspector.py`
- **功能**：批量将失败/离群文件画成双通道波形图，辅助人工排查。
- **输入**：
  - `outliers_report.csv` / `failed_report.csv`（当前脚本为工作目录相对路径）
  - 原始 CSV 目录（函数参数）
- **输出**：`results/diagnostics/Error_Waveforms_Dataset2/*.png`（建议）
- **关系**：依赖 `pccc_fft_teacher.py` 产出的报告。

## 2.6 `scripts/evaluation/outlier_inspector.py`
- **功能**：针对指定位置输出 5 子图诊断（原始、对齐、差分、zoom、FFT）。
- **输入**：原始 CSV 目录 + 目标位置列表。
- **输出**：`diagnose_5plots_*.png`。
- **关系**：用于解释 `pccc_fft_teacher.py` 或 `error_solve.py` 中异常点。

## 2.7 `scripts/plotting/prepare_mst_main_figures.py`
- **功能**：
  - 从 benchmark CSV 选择数据源（优先 `runs/classical_benchmark_suite/...`）。
  - 生成主图（误差曲线/误差分布/汇总条形图）和表格（CSV+TeX）与 latex 插片段。
- **输入**：benchmark 结果 CSV。
- **输出**：默认 `results/paper_outputs/*`。
- **关系**：论文最终产物聚合器。

## 2.8 `scripts/plotting/picture4-produce.py`
- **功能**：双曲线拟合（稳健拟合 + IQR）+ 残差图，IEEE 单栏/双栏输出。
- **输入**：`--csv`（默认 `data/processed/fft_results_robust.csv`）。
- **输出**：`Fig_HyperbolaFit_*.pdf/.png`。
- **关系**：读取教师结果，属于论文图子集。

## 2.9 `scripts/plotting/picture5-produce.py`
- **功能**：提取并绘制单文件双通道波形段（IEEE 风格）。
- **输入**：原始数据目录 + 位置映射参数。
- **输出**：`*.pdf/*.png`。
- **关系**：原始波形可视化，不依赖训练流程。

## 2.10 `scripts/plotting/picture6-produce.py`
- **功能**：最终线性拟合图；优先读 `fixed_results`，缺省回退 `clean_data`。
- **输入**：`data/processed/fixed_results.csv` 或 `data/processed/clean_data.csv`。
- **输出**：`Fig1_LinearFit_Fixed.bmp`。
- **关系**：依赖修复后结果，属于最终展示图。

## 2.11~2.14 legacy 脚本
- `picture1/2/3-produce.py`、`185-195mmnihe.py`：
  - **归类**：历史遗留（仍可复现早期流程）。
  - **主要风险**：路径硬编码、与新目录结构耦合较弱、与新主流程重复。
  - **建议**：不删除，放入 `scripts/legacy/`，仅做结果追溯。

## 3. 当前主要流水线（建议）

1) `scripts/data_processing/pccc_fft_teacher.py` 生成 robust 结果与异常报告。  
2) `scripts/evaluation/error_solve.py` 生成 `fixed_results.csv`。  
3) `scripts/data_processing/make_dataset_distillation.py` 生成 `dataset_distill.npz`。  
4) `scripts/training/train_attention_res1d_student.py train` 训练学生模型。  
5) `scripts/plotting/prepare_mst_main_figures.py` 聚合论文图表。  

## 4. 工程化改造落地情况

已完成：
- 按功能目录迁移脚本：`data_processing/training/evaluation/plotting/legacy`。
- 原始/处理后数据迁移到 `data/raw` 与 `data/processed`。
- 结果与论文产物迁移到 `results/*`。
- 主 README 与本审计文档建立。
- 部分主脚本默认路径已改为新目录。

待进一步落地（建议后续 PR）：
- 将 legacy 脚本统一改成 argparse 参数化。
- 抽取公共算法到 `src/deep_study/`（宽度计算、切段、拟合、绘图样式）。
- 增加 `tests/`（最少 smoke test + I/O contract test）。
- 增加 `configs/*.yaml`，统一管理数据路径和超参。