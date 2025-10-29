#!/bin/bash

# AffinCraft 模型 - 单GPU预训练脚本 (使用 LMDB 数据格式)
# 功能:
#   - 使用 fairseq-train 启动单 GPU 训练。
#   - 适配 Graphormer 结合 FLAG 对抗性训练。
#   - 读取 LMDB 格式数据，提升数据加载效率。

set -euo pipefail

# --- 环境变量 ---
export PYTHONWARNINGS="ignore::UserWarning:pkg_resources, ignore::FutureWarning:dgl.backend.pytorch.sparse, ignore::FutureWarning, ignore::UserWarning"
export PYTHONPATH="/data/run01/scw6f3q/zncao/affincraft/lib/python3.9/site-packages"

# --- 基础配置 ---
GPUS_PER_NODE=1
SEED=42

# --- 路径配置 ---
USER_DIR="/data/run01/scw6f3q/zncao/affincraft-nn/graphormer"
SAVE_DIR="./affincraft_pretrain_ckpts_lmdb_single_gpu"

# --- LMDB 数据路径 ---
TRAIN_LMDB="/data/run01/scw6f3q/zncao/lmdb_affincraft/train.lmdb"
VALID_LMDB="/data/run01/scw6f3q/zncao/lmdb_affincraft/valid.lmdb"

# --- 核心训练参数 --- bs 16
# LR=1e-4
# BATCH_SIZE_PER_GPU=20
# UPDATE_FREQ=1
# WARMUP_UPDATES=75000
# TOTAL_UPDATES=1875000
# NUM_WORKERS=6

LR=1e-4
BATCH_SIZE_PER_GPU=20
UPDATE_FREQ=1
WARMUP_UPDATES=45000
TOTAL_UPDATES=1400000
NUM_WORKERS=6

# ====================================================================================
# 2. 参数检查与准备
# ====================================================================================

if [ ! -d "$TRAIN_LMDB" ]; then
    echo "❌ 错误: 训练 LMDB 目录不存在: $TRAIN_LMDB"
    exit 1
fi

if [ ! -d "$VALID_LMDB" ]; then
    echo "❌ 错误: 验证 LMDB 目录不存在: $VALID_LMDB"
    exit 1
fi

mkdir -p "$SAVE_DIR"

EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * GPUS_PER_NODE * UPDATE_FREQ))

echo "==================================================================="
echo "          AffinCraft - 单GPU预训练 (LMDB格式)          "
echo "==================================================================="
echo "硬件配置:           ${GPUS_PER_NODE} GPU"
echo "DataLoader workers: ${NUM_WORKERS}"
echo "训练数据:           ${TRAIN_LMDB}"
echo "验证数据:           ${VALID_LMDB}"
echo "检查点保存目录:     ${SAVE_DIR}"
echo "梯度累积步数:       ${UPDATE_FREQ}"
echo "全局有效批次大小:   ${EFFECTIVE_BATCH_SIZE}"
echo "==================================================================="

# ====================================================================================
# 3. 启动训练 (单GPU)
# ====================================================================================

fairseq-train \
    --save-dir "$SAVE_DIR" \
    --user-dir "$USER_DIR" \
    \
    --num-workers "$NUM_WORKERS" \
    --dataset-source affincraft \
    --train-pkl-pattern "$TRAIN_LMDB" \
    --valid-pkl-pattern "$VALID_LMDB" \
    --data-buffer-size 100 \
    \
    --task graph_prediction \
    --criterion l2_loss_rmsd \
    --arch graphormer_large \
    --encoder-layers 18 \
    --encoder-embed-dim 896 \
    --encoder-ffn-embed-dim 896 \
    --encoder-attention-heads 32 \
    --num-classes 1 \
    --max-nodes 512 \
    --attention-dropout 0.1 \
    --act-dropout 0.1 \
    --dropout 0.1 \
    \
    --optimizer adam \
    --adam-betas "(0.9, 0.999)" \
    --adam-eps 1e-8 \
    --clip-norm 5.0 \
    --weight-decay 0.01 \
    \
    --lr-scheduler polynomial_decay \
    --power 1 \
    --max-epoch 150 \
    --max-update "$TOTAL_UPDATES" \
    --warmup-updates "$WARMUP_UPDATES" \
    --total-num-update "$TOTAL_UPDATES" \
    --lr "$LR" \
    --end-learning-rate 1e-9 \
    \
    --batch-size "$BATCH_SIZE_PER_GPU" \
    --update-freq "$UPDATE_FREQ" \
    --fp16 \
    --fp16-scale-window 128 \
    --fp16-init-scale 1 \
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
echo "✅ 训练完成！最终检查点已保存到: $SAVE_DIR"
echo "模型配置: 18层, 896维, 32个注意力头"
echo "==================================================================="