#!/bin/bash

# AffinCraft 模型 - 多GPU分布式预训练脚本 (使用 FLAG 对抗性训练)
#
# 功能:
#   - 使用 torchrun 启动 PyTorch 分布式训练。
#   - 适配 Graphormer 结合 FLAG 对抗性训练的需求。
#   - 提供清晰的参数配置区域，易于修改和维护。
#   - 支持可选索引文件(.idx)，增强数据加载的灵活性。
#
# 用法:
#   bash your_script_name.sh <训练PKL文件> <验证PKL文件>

export PYTHONWARNINGS="ignore::UserWarning:pkg_resources, ignore::FutureWarning:dgl.backend.pytorch.sparse, ignore::FutureWarning, ignore::UserWarning"
export PYTHONPATH=/data/run01/scw6f3q/zncao/affincraft/lib/python3.9/site-packages
# --- 脚本安全设置 ---
set -euo pipefail  # 遇错立即退出，禁止未定义变量，管道出错即退出
# export LC_ALL=C.UTF-8

# --- 分布式训练配置 ---
GPUS_PER_NODE=8                 # 单节点使用的GPU数量
NNODES=1                        # 总节点数 (单机设为1)
NODE_RANK=0                     # 当前节点序号 (单机设为0)
MASTER_ADDR="localhost"         # 主节点地址
MASTER_PORT=29500               # 主节点通信端口


# --- 路径配置 ---
USER_DIR="/data/run01/scw6f3q/zncao/affincraft-nn/graphormer"   # Graphormer 自定义模块路径
SAVE_DIR="./affincraft_pretrain_ckpts_flag_multi_gpu"  # 检查点和日志保存目录

# 【可选】索引文件路径 (留空则不使用)
TRAIN_PKL_INDEX="/data/run01/scw6f3q/zncao/data_pkl/train.idx"
VALID_PKL_INDEX="/data/run01/scw6f3q/zncao/data_pkl/valid.idx"

# --- 核心训练参数 ---
LR=1e-4                         # 学习率
BATCH_SIZE_PER_GPU=16           # 单GPU批次大小
UPDATE_FREQ=1                   # 梯度累积步数
WARMUP_UPDATES=75000
TOTAL_UPDATES=1875000
SEED=42                         # 随机种子

# ====================================================================================
# 2. 参数检查与准备
# ====================================================================================

if [ "$#" -ne 2 ]; then
    echo "错误: 需要提供两个必需的参数。"
    echo "用法: bash $0 <训练PKL文件> <验证PKL文件>"
    exit 1
fi

TRAIN_PKL_FILE="$1"
VALID_PKL_FILE="$2"

# 创建保存目录
mkdir -p "$SAVE_DIR"

# 动态构建可选参数
OPTIONAL_ARGS=""
if [ -n "$TRAIN_PKL_INDEX" ]; then
    OPTIONAL_ARGS+=" --train-pkl-index $TRAIN_PKL_INDEX"
    echo "信息: 使用训练索引文件: $TRAIN_PKL_INDEX"
fi
if [ -n "$VALID_PKL_INDEX" ]; then
    OPTIONAL_ARGS+=" --valid-pkl-index $VALID_PKL_INDEX"
    echo "信息: 使用验证索引文件: $VALID_PKL_INDEX"
fi

# --- 打印训练信息 ---
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * GPUS_PER_NODE * UPDATE_FREQ))
echo "==================================================================="
echo "          AffinCraft - 多GPU分布式预训练 (FLAG Enabled)          "
echo "==================================================================="
echo "硬件配置:           ${GPUS_PER_NODE} GPUs × ${NNODES} node(s)"
echo "检查点保存目录:     ${SAVE_DIR}"
echo "梯度累积步数:       ${UPDATE_FREQ}"
echo "全局有效批次大小:   ${EFFECTIVE_BATCH_SIZE}"
echo "==================================================================="

# ====================================================================================
# 3. 启动分布式训练
# ====================================================================================

torchrun \
    --nproc_per_node "$GPUS_PER_NODE" \
    --nnodes "$NNODES" \
    --node_rank "$NODE_RANK" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    "$(which fairseq-train)" \
    --save-dir "$SAVE_DIR" \
    --user-dir "$USER_DIR" \
    \
    --ddp-backend=legacy_ddp \
    --num-workers 8 \
    --dataset-source affincraft \
    --train-pkl-pattern "$TRAIN_PKL_FILE" \
    --valid-pkl-pattern "$VALID_PKL_FILE" \
    $OPTIONAL_ARGS \
    --data-buffer-size 20 \
    \
    --task graph_prediction \
    --criterion l2_loss_rmsd \
    --arch graphormer_large \
    --num-classes 1 \
    --max-nodes 512 \
    --optimizer adam \
    --adam-betas '(0.9, 0.999)' \
    --adam-eps 1e-8 \
    --clip-norm 5.0 \
    --weight-decay 0.01 \
    --lr-scheduler polynomial_decay \
    --power 1 \
    --max-epoch 150 \
    --max-update "$TOTAL_UPDATES" \
    --warmup-updates "$WARMUP_UPDATES" \
    --total-num-update "$TOTAL_UPDATES" \
    --lr "$LR" \
    --end-learning-rate 1e-9 \
    --batch-size "$BATCH_SIZE_PER_GPU" \
    --update-freq "$UPDATE_FREQ" \
    --fp16 \
    --fp16-scale-window 128 \
    --fp16-init-scale 1 \
    --encoder-layers 24 \
    --encoder-embed-dim 1024 \
    --encoder-ffn-embed-dim 1024 \
    --encoder-attention-heads 16 \
    --attention-dropout 0.1 \
    --act-dropout 0.1 \
    --dropout 0.1 \
    \
    --log-interval 50 \
    --save-interval 1 \
    --validate-interval 1 \
    --keep-last-epochs 20 \
    --seed "$SEED"

# ====================================================================================
# 4. 结束信息
# ====================================================================================
echo "==================================================================="
echo "多GPU预训练完成！最终检查点已保存到: $SAVE_DIR"
