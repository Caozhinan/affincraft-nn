#!/bin/bash
# ====================================================================================
# AffinCraft 模型 - 单卡预训练脚本（无 FLAG）
#
# 功能:
#   - 在单张 GPU 上进行 Graphormer 模型预训练。
#   - 与多卡参数完全对应，方便直接对比。
# ====================================================================================

export PYTHONWARNINGS="ignore::UserWarning:pkg_resources, \
ignore::FutureWarning, \
ignore::UserWarning:dgl.backend.pytorch.sparse"
export PYTHONPATH=/data/run01/scw6f3q/zncao/affincraft/lib/python3.9/site-packages

set -euo pipefail  # 遇错立即退出，禁止使用未定义变量

# ====================================================================================
# 1. 基础配置
# ====================================================================================

USER_DIR="/data/run01/scw6f3q/zncao/affincraft-nn/graphormer"
SAVE_DIR="./affincraft_pretrain_ckpts_single_gpu"   # 检查点保存目录

# 【可选】索引文件路径 (留空则不使用)
TRAIN_PKL_INDEX="/data/run01/scw6f3q/zncao/data_pkl/train.idx"
VALID_PKL_INDEX="/data/run01/scw6f3q/zncao/data_pkl/valid.idx"

# --- 核心训练超参，与多卡一致 ---
LR=1e-4
BATCH_SIZE=16
UPDATE_FREQ=1
WARMUP_UPDATES=75000
TOTAL_UPDATES=1875000
SEED=42

# ====================================================================================
# 2. 参数检查
# ====================================================================================

if [ "$#" -ne 2 ]; then
    echo "错误: 需要提供两个必需的参数。"
    echo "用法: bash $0 <训练PKL文件> <验证PKL文件>"
    exit 1
fi

TRAIN_PKL_FILE="$1"
VALID_PKL_FILE="$2"

mkdir -p "$SAVE_DIR"

OPTIONAL_ARGS=""
if [ -n "$TRAIN_PKL_INDEX" ]; then
    OPTIONAL_ARGS+=" --train-pkl-index $TRAIN_PKL_INDEX"
    echo "信息: 使用训练索引文件: $TRAIN_PKL_INDEX"
fi
if [ -n "$VALID_PKL_INDEX" ]; then
    OPTIONAL_ARGS+=" --valid-pkl-index $VALID_PKL_INDEX"
    echo "信息: 使用验证索引文件: $VALID_PKL_INDEX"
fi

# ====================================================================================
# 3. 打印训练信息
# ====================================================================================
echo "==================================================================="
echo "          AffinCraft - 单卡 Graphormer 预训练（无 FLAG）           "
echo "==================================================================="
echo "硬件配置:           单GPU"
echo "学习率:             ${LR}"
echo "批次大小:           ${BATCH_SIZE}"
echo "梯度累积步数:       ${UPDATE_FREQ}"
echo "全局有效批次大小:   $((BATCH_SIZE * UPDATE_FREQ))"
echo "检查点目录:         ${SAVE_DIR}"
echo "==================================================================="

# ====================================================================================
# 4. 启动训练
# ====================================================================================

"$(which fairseq-train)" \
  --save-dir "$SAVE_DIR" \
  --user-dir "$USER_DIR" \
  \
  --ddp-backend legacy_ddp \
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
  \
  --optimizer adam \
  --adam-betas '(0.9,0.999)' \
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
  \
  --batch-size "$BATCH_SIZE" \
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
# 5. 结束提示
# ====================================================================================
echo "==================================================================="
echo "✅ 单卡预训练完成！最终检查点已保存到: $SAVE_DIR"
echo "==================================================================="