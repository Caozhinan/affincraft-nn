#!/bin/bash

# 设置脚本在遇到错误时立即退出，并防止使用未定义的变量
set -euo pipefail

# --- AffinCraft模型4卡分布式预训练脚本 ---
# 硬件配置: 4x NVIDIA A100-SXM4-40GB (单节点)

# 1. 分布式训练配置
GPUS_PER_NODE=1         # 使用4张A100卡
MASTER_ADDR="localhost" # 单节点训练，主节点地址为本地
MASTER_PORT=29500       # 主节点端口
NNODES=1                # 节点总数
NODE_RANK=0             # 当前节点排名 (从0开始) 单机训练时 → 永远是 0

# 2. 检查输入参数
if [ "$#" -ne 2 ]; then
    echo "错误: 需要提供两个参数。"
    echo "用法: bash $0 <训练PKL文件路径> <验证PKL文件路径>"
    exit 1
fi

# 3. 路径和目录配置
TRAIN_PKL_FILE="$1"
VALID_PKL_FILE="$2"
SAVE_DIR="./affincraft_pretrain_ckpts"
USER_DIR="/xcfhome/zncao02/affincraft-nn/graphormer" # Graphormer自定义模块的路径

# 创建检查点保存目录，-p选项确保在目录已存在时不报错
mkdir -p "$SAVE_DIR"

# 4. 打印训练信息
echo "======================================================"
echo "开始4卡分布式预训练 AffinCraft 模型..."
echo "======================================================"
echo "硬件配置: ${GPUS_PER_NODE}x NVIDIA A100-SXM4-40GB"
echo "训练数据: $TRAIN_PKL_FILE"
echo "验证数据: $VALID_PKL_FILE"
echo "检查点保存到: $SAVE_DIR"
echo "------------------------------------------------------"

# 5. 启动分布式训练
# 使用 torchrun (torch.distributed.run) 启动，这是PyTorch官方推荐的最新方式
# 使用 torchrun 替代 python -m torch.distributed.launch
torchrun \  # 使用 PyTorch 分布式启动器
    --nproc_per_node $GPUS_PER_NODE \   # 每个节点 GPU 数
    --nnodes $NNODES \                  # 节点数
    --node_rank $NODE_RANK \            # 当前节点 rank
    --master_addr $MASTER_ADDR \        # 主节点地址
    --master_port $MASTER_PORT \        # 主节点端口
    "$(which fairseq-train)" \          # 调用 fairseq-train
    --user-dir "$USER_DIR" \            # 自定义模块所在目录
    --num-workers 8 \                  # DataLoader worker 数
    --ddp-backend=nccl \                # 分布式后端
    --dataset-source affincraft \       # 数据集类型
    --train-pkl-pattern "$TRAIN_PKL_FILE" \  # 训练数据路径模式
    --valid-pkl-pattern "$VALID_PKL_FILE" \  # 验证数据路径模式

    --task graph_prediction \           # 任务：图预测
    --criterion l2_loss_rmsd_with_flag \     # 损失函数：L2/RMSD
    --arch graphormer_large \           # 模型结构：Graphormer-large
    --num-classes 1 \                   # 输出类别数（回归=1）

    --max-nodes 512 \                   # 最大图节点数
    --attention-dropout 0.1 \           # 注意力 dropout
    --act-dropout 0.1 \                 # 激活函数 dropout
    --dropout 0.1 \                     # 全局 dropout

    --optimizer adam \                  # 优化器 Adam
    --adam-betas '(0.9, 0.999)' \       # Adam 参数
    --adam-eps 1e-8 \                   # Adam epsilon
    --clip-norm 5.0 \                   # 梯度裁剪
    --weight-decay 0.01 \               # 权重衰减

    --lr-scheduler polynomial_decay \   # 学习率策略：多项式衰减
    --power 1 \                         # 多项式幂次（1=线性）
    --warmup-updates 20000 \            # 预热步数
    --total-num-update 500000 \         # 总更新步数
    --lr 1e-4 \                         # 初始学习率
    --end-learning-rate 1e-9 \          # 最小学习率

    --batch-size 32 \                   # 单 GPU batch size
    --update-freq 1 \                   # 梯度累积步数
    --fp16 \                            # 半精度训练
    --data-buffer-size 20 \             # 数据缓存大小

    --encoder-layers 24 \               # Transformer 层数
    --encoder-embed-dim 1024 \          # 嵌入维度
    --encoder-ffn-embed-dim 1024 \      # FFN 维度
    --encoder-attention-heads 16 \      # Attention 头数

    --max-epoch 200 \                   # 最大训练 epoch 数
    --save-dir "$SAVE_DIR" \            # checkpoint 保存目录
    --log-interval 50 \                 # 日志打印间隔（step）
    
    # === 修改的部分 ===
    --save-interval 1 \                 # 每个 epoch 保存一次 ckpt
    --validate-interval 1 \             # 每个 epoch 验证一次
    --keep-last-epochs 20 \             # 保留最近的 20 个 epoch ckpt
    --seed 42                           # 随机种子，保证可复现

echo "======================================================"
echo "预训练完成。检查点已保存到: $SAVE_DIR"
echo "======================================================"