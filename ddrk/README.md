# DRKG-COVID: 复现DRKG用于COVID-19药物重新定位

本项目基于论文《Analysis of Drug Repurposing Knowledge Graphs for Covid-19》，实现了使用知识图谱嵌入（KGE）和链路预测来发现COVID-19潜在治疗药物的方法。

## 项目结构

```
ddrk/
├── data/                    # 数据目录
│   ├── drkg.tsv             # DRKG知识图谱数据
│   ├── entity_idx_map.txt   # 实体映射文件
│   ├── relation_idx_map.txt # 关系映射文件
│   └── processed/           # 处理后的数据
├── results/                 # 结果目录
│   ├── models/              # 存储训练好的模型
│   ├── predictions/         # 预测结果
│   └── figures/             # 可视化图形
├── src/                     # 源代码
│   ├── utils/               # 工具函数
│   ├── drkg_with_dgl/       # DGL相关代码
│   ├── drug_repurpose/      # 药物重新定位代码
│   ├── embedding_analysis/  # 嵌入分析代码
│   └── raw_graph_analysis/  # 原始图分析代码
├── notebooks/               # Jupyter笔记本
├── setup.sh                 # 环境设置脚本
└── run_drkg.sh              # 运行流程脚本
```

## 环境需求

- 操作系统: Linux (首选) 或 macOS
- Python 3.7+
- CUDA (推荐使用，用于GPU加速)
- 硬件要求:
  - CPU: 8+ 核心推荐
  - RAM: 16GB+ (32GB更佳)
  - GPU: 
    - 最低: 任何CUDA兼容GPU，8GB+ 显存
    - 推荐: NVIDIA RTX 30系列或更新，16GB+ 显存
    - 优化: NVIDIA RTX 4090 (24GB)，已针对此GPU做特殊优化

## RTX 4090优化特性

如果检测到NVIDIA RTX 4090 GPU，本实现提供以下优化:

- **自动参数调整**: 自动设置最优批次大小、嵌入维度和负样本数，以充分利用24GB显存
- **更高维度嵌入**: 支持高达800维的嵌入向量，提高模型表现力
- **混合CPU-GPU训练**: 对于特大模型自动启用混合CPU-GPU模式
- **高级并行处理**: 优化的批处理预测算法，比串行处理快5-10倍
- **集成预测**: 自动训练多个模型(TransE和RotatE)并集成结果
- **高级可视化**: 包括网络视图和增强的药物候选图表
- **训练时间监控**: 详细记录训练进度和性能指标

## 快速开始

### 1. 环境设置

首先，运行环境设置脚本：

```bash
chmod +x setup.sh
./setup.sh
```

这将：
- 安装所需的系统依赖
- 创建Conda环境
- 下载DRKG数据集
- 设置项目目录结构
- 检测GPU并安装适当版本的PyTorch和DGL

### 2. 运行完整流程

运行完整的药物重新定位流程：

```bash
chmod +x run_drkg.sh
./run_drkg.sh
```

该脚本会自动运行：
1. 数据处理
2. 模型训练
   - 如果检测到RTX 4090，会训练TransE和RotatE两种模型
   - 针对GPU性能自动调整参数
3. 药物重新定位预测
4. 结果可视化（包括高级网络视图）
5. 集成多模型结果（如可用）
6. 生成详细性能报告

### 3. 单独运行各个步骤

如果需要单独运行某个步骤：

#### 数据处理

```bash
conda activate drkg
python -m src.utils.data_utils
```

#### 模型训练

```bash
conda activate drkg
# 对于RTX 4090，使用高级参数
python -m src.drkg_with_dgl.train_embeddings --model TransE --dim 800 --batch_size 8192 --neg_samples 512 --gpu 0
```

支持的模型：
- TransE: 适合大多数情况，训练速度快
- RotatE: 可以捕获更复杂的关系模式，但训练较慢
- DistMult: 简单有效的模型
- ComplEx: 可以处理非对称关系

#### 药物预测

```bash
conda activate drkg
python -m src.drug_repurpose.covid_drug_repurposing --model_dir "results/models/你的模型目录" --plot --top_k 200
```

#### 模型集成预测

```bash
conda activate drkg
python -m src.drug_repurpose.ensemble_predictions --pred_files "预测文件1.csv,预测文件2.csv" --output "ensemble_results.csv" --plot
```

## 参数说明

### 模型训练参数

- `--model`: 知识图谱嵌入模型类型 (TransE, RotatE, DistMult, ComplEx)
- `--dim`: 嵌入维度，RTX 4090可使用高达800维
- `--batch_size`: 批次大小，RTX 4090推荐值为4096-8192
- `--neg_samples`: 负采样数量，RTX 4090推荐值为256-512
- `--lr`: 学习率，默认0.001，可使用0.0005获得更稳定训练
- `--epochs`: 训练轮数，以DRKG数据集规模，200轮通常足够
- `--gpu`: GPU ID，0表示第一个GPU
- `--mix_cpu_gpu`: 是否使用混合CPU-GPU模式（对于大模型推荐）

### 药物预测参数

- `--model_dir`: 模型目录
- `--entity_map`: 实体映射文件
- `--disease`: 疾病实体名称
- `--relation`: 关系名称
- `--top_k`: 返回前k个候选药物
- `--output`: 输出文件路径
- `--plot`: 是否绘制图表

## 性能指标

使用RTX 4090 GPU的典型性能指标:

| 模型    | 维度  | 批次大小 | 训练时间(200轮) | 内存使用  | GPU使用率 |
|--------|------|---------|--------------|---------|----------|
| TransE | 400  | 8192    | ~1.5小时      | ~12GB   | ~90%     |
| TransE | 800  | 4096    | ~3小时        | ~20GB   | ~95%     |
| RotatE | 400  | 4096    | ~4小时        | ~16GB   | ~85%     |
| DistMult | 400 | 8192   | ~1小时        | ~10GB   | ~80%     |

## 结果示例

运行完成后，您可以在以下位置找到结果：

- 预测的候选药物：`results/predictions/covid_drug_predictions.csv`（和.xlsx格式）
- 可视化结果：
  - 柱状图：`results/predictions/covid_drug_predictions.png`
  - 网络图：`results/predictions/covid_drug_predictions_network.png`
- 训练好的模型：`results/models/模型名称_维度_时间戳/`
- 性能报告：`results/performance_report_时间戳.txt`
- 集成预测结果：`results/predictions/ensemble_时间戳/`

## 注意事项

1. 显存要求：
   - TransE 400维: 推荐至少12GB显存
   - RotatE 400维: 推荐至少16GB显存
   - TransE/RotatE 800维: 推荐至少20GB显存
   - RTX 4090 (24GB)可以舒适运行所有配置

2. 训练时间：
   - 使用RTX 4090，训练200轮通常需要1-4小时，取决于模型
   - CPU训练可能需要数天时间

3. 系统内存：
   - 处理完整DRKG数据集需要至少16GB RAM
   - 推荐32GB或更高内存获得最佳性能

## 参考文献

```
@misc{drkg2020,
  author = {Ioannidis, Vassilis N. and Song, Xiang and Manchanda, Saurav and Li, Mufei and Pan, Xiaoqin
            and Zheng, Da and Ning, Xia and Zeng, Xiangxiang and Karypis, George},
  title = {DRKG - Drug Repurposing Knowledge Graph for Covid-19},
  howpublished = "\url{https://github.com/gnn4dr/DRKG/}",
  year = {2020}
}
``` 