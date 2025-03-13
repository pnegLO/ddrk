#!/bin/bash

# 设置错误处理
set -e
echo "开始设置DRKG (Drug Repurposing Knowledge Graph)环境..."

# 创建工作目录
WORK_DIR="$(pwd)"
echo "工作目录: $WORK_DIR"

# 创建项目结构
mkdir -p data
mkdir -p src/utils
mkdir -p src/drkg_with_dgl
mkdir -p src/drug_repurpose
mkdir -p src/embedding_analysis
mkdir -p src/raw_graph_analysis
mkdir -p notebooks
mkdir -p results/models
mkdir -p results/predictions
mkdir -p results/figures

# 安装系统依赖
echo "安装系统依赖..."
if [ "$(uname)" == "Linux" ]; then
    sudo apt update
    sudo apt install -y build-essential git wget curl python3-dev python3-pip openjdk-11-jdk unzip
elif [ "$(uname)" == "Darwin" ]; then
    echo "Mac OS X 检测到，跳过系统依赖安装"
    # 确保有基本工具
    if ! command -v brew &> /dev/null; then
        echo "推荐安装Homebrew: https://brew.sh/"
    fi
else
    echo "不支持的操作系统。请确保手动安装所需的系统依赖。"
fi

# 检查Conda是否已安装
if ! command -v conda &> /dev/null; then
    echo "Conda未安装，正在安装Miniconda..."
    if [ "$(uname)" == "Linux" ]; then
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    elif [ "$(uname)" == "Darwin" ]; then
        # 检查是否是Apple Silicon
        if [ "$(uname -m)" == "arm64" ]; then
            wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -O miniconda.sh
        else
            wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
        fi
    fi
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    export PATH="$HOME/miniconda/bin:$PATH"
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.zshrc
    source ~/.bashrc 2>/dev/null || source ~/.zshrc 2>/dev/null || true
else
    echo "Conda已安装，继续执行..."
fi

# 创建并激活Conda环境
echo "创建DRKG的Conda环境..."
conda create -n drkg python=3.7 -y
source activate drkg || conda activate drkg

# 检测GPU
echo "检测GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU 检测到，获取详细信息："
    nvidia-smi
    
    # 获取GPU型号
    GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    echo "GPU型号: $GPU_MODEL"
    
    # 检查是否是RTX 4090
    if [[ "$GPU_MODEL" == *"4090"* ]]; then
        echo "检测到RTX 4090，安装适用于高端GPU的PyTorch和DGL版本"
        # 使用CUDA 11.8（兼容RTX 4090）
        pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
        pip install dgl-cu118==1.1.2 dglke==0.1.2
    else
        # 使用标准CUDA版本
        pip install torch==1.7.1 torchvision==0.8.2
        pip install dgl==0.6.1 dglke==0.1.2
    fi
else
    echo "未检测到NVIDIA GPU，安装CPU版本PyTorch和DGL"
    pip install torch==1.7.1 torchvision==0.8.2 --extra-index-url https://download.pytorch.org/whl/cpu
    pip install dgl==0.6.1 dglke==0.1.2
fi

# 安装Python依赖
echo "安装Python依赖..."
pip install pandas numpy matplotlib seaborn scikit-learn
pip install jupyter notebook tqdm
pip install networkx psutil 
pip install openpyxl plotly
pip install tensorboard

# RTX 4090优化包
if [[ "$GPU_MODEL" == *"4090"* ]]; then
    pip install cupy-cuda11x  # 用于GPU加速数组操作
    pip install numba         # JIT编译加速
    
    # 提示用户关于NVIDIA Apex的可选安装
    echo "提示: 对于更快的训练速度，可以安装NVIDIA Apex（需要从源码编译）"
    echo "更多信息: https://github.com/NVIDIA/apex"
fi

# 克隆DRKG代码库
echo "克隆DRKG代码库..."
if [ ! -d "DRKG_repo" ]; then
    git clone https://github.com/gnn4dr/DRKG.git DRKG_repo
else
    echo "DRKG_repo目录已存在，跳过克隆步骤。"
fi

# 下载DRKG数据集
echo "下载DRKG数据集..."
cd data

if [ ! -f "drkg.tsv" ]; then
    wget https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz
    tar -xzvf drkg.tar.gz
    rm drkg.tar.gz
else
    echo "DRKG数据集已存在，跳过下载步骤。"
fi

if [ ! -f "entity_idx_map.txt" ]; then
    wget https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/entity_idx_map.txt
else
    echo "实体映射文件已存在，跳过下载步骤。"
fi

if [ ! -f "relation_idx_map.txt" ]; then
    wget https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/relation_idx_map.txt
else
    echo "关系映射文件已存在，跳过下载步骤。"
fi

cd ..

# 拷贝所需的工具函数
echo "拷贝工具函数..."
cp -r DRKG_repo/utils/* src/utils/

# 创建启动脚本
echo "创建Jupyter启动脚本..."
cat > start_jupyter.sh << 'EOF'
#!/bin/bash
source activate drkg || conda activate drkg
jupyter notebook
EOF
chmod +x start_jupyter.sh

# 设置run_drkg.sh和setup.sh的执行权限
chmod +x run_drkg.sh

echo "====================="
echo "DRKG环境设置完成！"
echo "系统信息："
uname -a
echo "Python版本："
python --version
if command -v nvidia-smi &> /dev/null; then
    echo "GPU信息："
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
fi
echo "====================="
echo "使用以下命令启动Jupyter Notebook："
echo "./start_jupyter.sh"
echo "使用以下命令运行完整流程："
echo "./run_drkg.sh"
echo "=====================" 