#!/bin/bash
# AffinCraft 模型 - 多GPU分布式预训练脚本 (使用 LMDB 数据格式)

export PYTHONWARNINGS="ignore::UserWarning:pkg_resources, ignore::FutureWarning:dgl.backend.pytorch.sparse, ignore::FutureWarning, ignore::UserWarning"
export PYTHONPATH=/data/run01/scw6f3q/zncao/affincraft/lib/python3.9/site-packages:$PYTHONPATH
export PYTHONPATH=/data/run01/scw6f3q/zncao/affincraft-nn/fairseq:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export TORCH_NAN=1
export CUDA_LAUNCH_BLOCKING=1
export TF_CPP_MIN_LOG_LEVEL=2

export OMP_NUM_THREADS=1
set -euo pipefail

# --- 分布式训练配置 ---
GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=29500

# --- 路径配置 ---
USER_DIR="/data/run01/scw6f3q/zncao/affincraft-nn/graphormer"
SAVE_DIR="/data/run01/scw6f3q/zncao/affincraft-nn/ckpt_pretrian"

TRAIN_LMDB="/ssd/home/scw6f3q/train_lmdb"
VALID_LMDB="/ssd/home/scw6f3q/valid_lmdb"

# --- 训练参数 ---
LR=5e-5
BATCH_SIZE_PER_GPU=8
UPDATE_FREQ=1
SEED=42
NUM_WORKERS=2

# ====================================================================================
# 1. 根据1.46M样本、8GPU×8batch、100epoch计算步数
# ====================================================================================
MAX_EPOCH=100
UPDATES_PER_EPOCH=22770
TOTAL_UPDATES=$((MAX_EPOCH * UPDATES_PER_EPOCH)) # 2,277,000
WARMUP_UPDATES=91000  # 保持约4%比例

# ====================================================================================
# 2. 检查与打印
# ====================================================================================

if [ ! -d "$TRAIN_LMDB" ]; then
    echo "错误: 训练LMDB目录不存在: $TRAIN_LMDB"
    exit 1
fi

if [ ! -d "$VALID_LMDB" ]; then
    echo "错误: 验证LMDB目录不存在: $VALID_LMDB"
    exit 1
fi

mkdir -p "$SAVE_DIR"

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
echo "-------------------------------------------------------------------"
echo "目标训练轮数:       ${MAX_EPOCH}"
echo "估算总更新步数:     ${TOTAL_UPDATES}"
echo "学习率预热步数:     ${WARMUP_UPDATES}"
echo "==================================================================="

# ====================================================================================
# 3. 启动训练
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
    --data-buffer-size 20 \
    \
    --task graph_prediction \
    --criterion l2_loss_rmsd_fp32 \
    --arch graphormer_large \
    --num-classes 1 \
    --max-nodes 460 \
    --optimizer adam \
    --adam-betas '(0.9, 0.999)' \
    --adam-eps 1e-8 \
    --clip-norm 5.0 \
    --weight-decay 0.01 \
    --lr-scheduler polynomial_decay \
    --power 1 \
    \
    --max-epoch "$MAX_EPOCH" \
    --max-update "$TOTAL_UPDATES" \
    --warmup-updates "$WARMUP_UPDATES" \
    --total-num-update "$TOTAL_UPDATES" \
    \
    --lr "$LR" \
    --end-learning-rate 1e-9 \
    --batch-size "$BATCH_SIZE_PER_GPU" \
    --update-freq "$UPDATE_FREQ" \
    \
    --encoder-layers 18 \
    --encoder-embed-dim 896 \
    --encoder-ffn-embed-dim 896 \
    --encoder-attention-heads 32 \
    --attention-dropout 0.05 \
    --act-dropout 0.05 \
    --dropout 0.05 \
    \
    --log-interval 200 \
    --save-interval 1 \
    --validate-interval 1 \
    --keep-last-epochs 20 \
    --seed "$SEED"

echo "==================================================================="
echo "多GPU预训练完成！最终检查点已保存到: $SAVE_DIR"
echo "==================================================================="