#!/bin/bash

# 设置脚本在遇到错误时立即退出，并防止使用未定义的变量
# set -euo pipefail

# --- AffinCraft模型单卡预训练脚本 ---
# 硬件配置: 1x NVIDIA A100-SXM4-40GB

# 1. 检查输入参数
if [ "$#" -ne 2 ]; then
    echo "错误: 需要提供两个参数。"
    echo "用法: bash $0 <训练PKL文件路径> <验证PKL文件路径>"
    exit 1
fi

# 2. 路径和目录配置
TRAIN_PKL_FILE="$1"
VALID_PKL_FILE="$2"
SAVE_DIR="./affincraft_pretrain_ckpts"
USER_DIR="/xcfhome/zncao02/affincraft-nn/graphormer" # Graphormer自定义模块的路径

# 创建检查点保存目录
mkdir -p "$SAVE_DIR"

# 3. 打印训练信息
echo "======================================================"
echo "开始单卡预训练 AffinCraft 模型..."
echo "======================================================"
echo "硬件配置: 1x NVIDIA A100-SXM4-40GB"
echo "训练数据: $TRAIN_PKL_FILE"
echo "验证数据: $VALID_PKL_FILE"
echo "检查点保存到: $SAVE_DIR"
echo "------------------------------------------------------"

# 4. 启动训练
# 单卡训练直接调用 fairseq-train，不需要 torch.distributed
"$(which fairseq-train)" \
    --user-dir "$USER_DIR" \
    --num-workers 8 \
    --dataset-source affincraft \
    --train-pkl-pattern "$TRAIN_PKL_FILE" \
    --valid-pkl-pattern "$VALID_PKL_FILE" \
    --task graph_prediction \
    --criterion l2_loss_rmsd_with_flag \
    --arch graphormer_large \
    --num-classes 1 \
    --max-nodes 512 \
    --attention-dropout 0.1 \
    --act-dropout 0.1 \
    --dropout 0.1 \
    --optimizer adam \
    --adam-betas '(0.9, 0.999)' \
    --adam-eps 1e-8 \
    --clip-norm 5.0 \
    --weight-decay 0.01 \
    --lr-scheduler polynomial_decay \
    --power 1 \
    --warmup-updates 20000 \
    --total-num-update 500000 \
    --lr 1e-4 \
    --end-learning-rate 1e-9 \
    --batch-size 32 \
    --update-freq 1 \
    --fp16 \
    --data-buffer-size 20 \
    --encoder-layers 24 \
    --encoder-embed-dim 1024 \
    --encoder-ffn-embed-dim 1024 \
    --encoder-attention-heads 16 \
    --max-epoch 200 \
    --save-dir "$SAVE_DIR" \
    --log-interval 50 \
    --save-interval 1 \
    --validate-interval 1 \
    --keep-last-epochs 20 \
    --seed 42

echo "======================================================"
echo "单卡预训练完成。检查点已保存到: $SAVE_DIR"
echo "======================================================"