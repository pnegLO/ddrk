#!/bin/bash

# 设置错误处理
set -e
echo "启动DRKG COVID-19药物重新定位流程..."

# 检查环境是否已设置
if [ ! -d "src" ] || [ ! -d "data" ]; then
    echo "环境未设置，请先运行 setup.sh"
    exit 1
fi

# 激活Conda环境
source activate drkg || conda activate drkg || echo "请手动激活drkg环境"

# RTX 4090配置 - 24GB显存
BATCH_SIZE=8192
EMBEDDING_DIM=800
NEG_SAMPLES=512
MAX_EPOCHS=200

# 步骤1: 数据处理
echo "====================="
echo "步骤1: 数据处理"
echo "====================="
python -m src.utils.data_utils

# 检查数据处理是否成功
if [ ! -d "data/processed" ]; then
    echo "数据处理失败，请检查日志"
    exit 1
fi

# 步骤2: 模型训练
echo "====================="
echo "步骤2: 模型训练"
echo "====================="

# 检查GPU信息
if command -v nvidia-smi &> /dev/null; then
    echo "检测到GPU："
    nvidia-smi
    
    # 获取GPU信息
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)
    
    echo "使用GPU: $GPU_NAME ($GPU_MEMORY)"
    
    # 如果检测到RTX 4090，使用优化配置
    if [[ "$GPU_NAME" == *"4090"* ]]; then
        echo "检测到RTX 4090 GPU，使用优化配置"
        GPU_FLAG="--gpu 0 --batch_size $BATCH_SIZE --dim $EMBEDDING_DIM --neg_samples $NEG_SAMPLES --epochs $MAX_EPOCHS --lr 0.0005"
    else
        echo "使用标准GPU配置"
        GPU_FLAG="--gpu 0"
    fi
else
    echo "未检测到GPU，使用CPU训练"
    GPU_FLAG="--gpu -1"
fi

# 训练不同的模型
echo "训练TransE模型..."
python -m src.drkg_with_dgl.train_embeddings --model TransE $GPU_FLAG

echo "训练RotatE模型..."
python -m src.drkg_with_dgl.train_embeddings --model RotatE $GPU_FLAG

# 选择最新的模型目录
LATEST_TRANSE=$(ls -t results/models | grep "TransE" | head -n 1)
LATEST_ROTATE=$(ls -t results/models | grep "RotatE" | head -n 1)

if [ -z "$LATEST_TRANSE" ] && [ -z "$LATEST_ROTATE" ]; then
    echo "找不到训练好的模型，请检查训练日志"
    exit 1
fi

# 使用最新的TransE模型，如果没有则使用RotatE
if [ -n "$LATEST_TRANSE" ]; then
    MODEL_DIR="results/models/$LATEST_TRANSE"
    MODEL_TYPE="TransE"
else
    MODEL_DIR="results/models/$LATEST_ROTATE"
    MODEL_TYPE="RotatE"
fi

# 步骤3: 药物重新定位
echo "====================="
echo "步骤3: 药物重新定位"
echo "====================="
echo "使用模型: $MODEL_DIR ($MODEL_TYPE)"

# 运行药物重新定位
python -m src.drug_repurpose.covid_drug_repurposing --model_dir "$MODEL_DIR" --plot --top_k 200

# 如果同时有TransE和RotatE模型，进行集成预测
if [ -n "$LATEST_TRANSE" ] && [ -n "$LATEST_ROTATE" ]; then
    echo "====================="
    echo "步骤4: 模型集成预测"
    echo "====================="
    echo "集成TransE和RotatE模型进行预测..."
    
    # 创建一个新的集成结果目录
    ENSEMBLE_DIR="results/predictions/ensemble_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$ENSEMBLE_DIR"
    
    # 为每个模型运行预测并保存结果
    python -m src.drug_repurpose.covid_drug_repurposing --model_dir "results/models/$LATEST_TRANSE" --output "$ENSEMBLE_DIR/transe_predictions.csv"
    python -m src.drug_repurpose.covid_drug_repurposing --model_dir "results/models/$LATEST_ROTATE" --output "$ENSEMBLE_DIR/rotate_predictions.csv"
    
    # 运行集成脚本（假设我们已经创建了这个脚本）
    python -m src.drug_repurpose.ensemble_predictions \
        --pred_files "$ENSEMBLE_DIR/transe_predictions.csv,$ENSEMBLE_DIR/rotate_predictions.csv" \
        --output "$ENSEMBLE_DIR/ensemble_predictions.csv" \
        --plot
        
    echo "集成预测结果保存在: $ENSEMBLE_DIR/ensemble_predictions.csv"
    echo "集成可视化结果保存在: $ENSEMBLE_DIR/ensemble_predictions.png"
fi

echo "====================="
echo "流程完成！"
echo "====================="
echo "预测结果保存在: results/predictions/covid_drug_predictions.csv"
echo "可视化结果保存在: results/predictions/covid_drug_predictions.png"
echo "模型保存在: $MODEL_DIR"

# 生成训练和预测的摘要报告
REPORT_FILE="results/performance_report_$(date +%Y%m%d_%H%M%S).txt"
echo "生成性能报告: $REPORT_FILE"

{
    echo "=== DRKG COVID-19药物重新定位性能报告 ==="
    echo "日期: $(date)"
    echo ""
    echo "=== 硬件信息 ==="
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
        echo "GPU内存: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)"
        echo "CUDA版本: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
    else
        echo "未检测到GPU，使用CPU训练"
    fi
    echo ""
    echo "=== 模型信息 ==="
    echo "模型类型: $MODEL_TYPE"
    echo "嵌入维度: $EMBEDDING_DIM"
    echo "批次大小: $BATCH_SIZE"
    echo "负采样数量: $NEG_SAMPLES"
    echo "训练轮数: $MAX_EPOCHS"
    echo ""
    echo "=== 预测结果摘要 ==="
    echo "Top-10 候选药物:"
    head -n 11 "results/predictions/covid_drug_predictions.csv"
} > "$REPORT_FILE"

echo "性能报告已保存到: $REPORT_FILE" 