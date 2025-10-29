#!/bin/bash

# AffinCraft 模型 - 多GPU分布式预训练脚本 (使用 LMDB 数据格式)
#
# 功能:
#   - 使用 torchrun 启动 PyTorch 分布式训练。
#   - 适配 Graphormer 结合 FLAG 对抗性训练的需求。
#   - 使用 LMDB 格式数据,提升数据加载效率。
#
# 用法:
#   bash md_multi_gpu.sh
export PYTHONWARNINGS="ignore::UserWarning:pkg_resources, ignore::FutureWarning:dgl.backend.pytorch.sparse, ignore::FutureWarning, ignore::UserWarning"
export PYTHONPATH=/data/run01/scw6f3q/zncao/affincraft/lib/python3.9/site-packages
#export CUDA_LAUNCH_BLOCKING=0
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
#export NCCL_ASYNC_ERROR_HANDLING=1
#export NCCL_SOCKET_NTHREADS=8
#export NCCL_NSOCKS_PERTHREAD=8
# --- 脚本安全设置 ---
set -euo pipefail  # 遇错立即退出,禁止未定义变量,管道出错即退出

# --- 分布式训练配置 ---
GPUS_PER_NODE=8                 # 单节点使用的GPU数量
NNODES=1                        # 总节点数 (单机设为1)
NODE_RANK=0                     # 当前节点序号 (单机设为0)
MASTER_ADDR="localhost"         # 主节点地址
MASTER_PORT=29500               # 主节点通信端口

# --- 路径配置 ---
USER_DIR="/data/run01/scw6f3q/zncao/affincraft-nn/graphormer"   # Graphormer 自定义模块路径
SAVE_DIR="./affincraft_pretrain_ckpts_lmdb_multi_gpu"           # 检查点和日志保存目录

# 【修改】LMDB 数据路径 (不再需要索引文件)
TRAIN_LMDB="/ssd/home/scw6f3q/train_lmdb"
VALID_LMDB="/ssd/home/scw6f3q/lmdb/valid.lmdb"

# --- 核心训练参数 ---
LR=1e-4                         # 学习率
BATCH_SIZE_PER_GPU=16          # 单GPU批次大小
UPDATE_FREQ=1                   # 梯度累积步数
WARMUP_UPDATES=75000
TOTAL_UPDATES=1875000
SEED=42                         # 随机种子
NUM_WORKERS=8                   # 每个GPU的DataLoader worker数量 (匹配6个CPU核)

# ====================================================================================
# 2. 参数检查与准备
# ====================================================================================

# 检查LMDB文件是否存在
if [ ! -d "$TRAIN_LMDB" ]; then
    echo "错误: 训练LMDB目录不存在: $TRAIN_LMDB"
    exit 1
fi

if [ ! -d "$VALID_LMDB" ]; then
    echo "错误: 验证LMDB目录不存在: $VALID_LMDB"
    exit 1
fi

# 创建保存目录
mkdir -p "$SAVE_DIR"

# --- 打印训练信息 ---
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * GPUS_PER_NODE * UPDATE_FREQ))
echo "==================================================================="
echo "          AffinCraft - 多GPU分布式预训练 (LMDB格式)          "
echo "==================================================================="
echo "硬件配置:           ${GPUS_PER_NODE} GPUs × ${NNODES} node(s)"
echo "DataLoader workers: ${NUM_WORKERS} per GPU"
echo "训练数据:           ${TRAIN_LMDB}"
echo "验证数据:           ${VALID_LMDB}"
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
    --required-batch-size-multiple 1 \
    \
    --ddp-backend=c10d \
    --find-unused-parameters \
    --num-workers "$NUM_WORKERS" \
    --dataset-source affincraft \
    --train-pkl-pattern "$TRAIN_LMDB" \
    --valid-pkl-pattern "$VALID_LMDB" \
    --data-buffer-size 100 \
    \
    --task graph_prediction \
    --criterion l2_loss_rmsd \
    --arch graphormer_large \
    --num-classes 1 \
    --max-nodes 464 \
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
    --encoder-layers 18 \
    --encoder-embed-dim 896 \
    --encoder-ffn-embed-dim 896 \
    --encoder-attention-heads 32 \
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
