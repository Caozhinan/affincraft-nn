#!/bin/bash  
  
# 训练脚本：使用 graphormer_large 架构和动态参数设置进行 AffinCraft 模型训练  
# 用法: bash train_affincraft_large_dynamic.sh <train_pkl_file> <valid_pkl_file>  
  
# 检查参数数量  
if [ "$#" -ne 2 ]; then  
    echo "用法: bash $0 <训练PKL文件路径> <验证PKL文件路径>"  
    exit 1  
fi  
  
TRAIN_PKL_FILE=$1  
VALID_PKL_FILE=$2  
SAVE_DIR="../affincraft_ckpts"  
USER_DIR="../../graphormer"  
  
# 训练参数配置  
max_epoch=1000  
batch_size=8  # graphormer_large 需要更小的 batch size  
n_gpu=1  
  
# 动态参数计算  
[ -z "${update_freq}" ] && update_freq=1  
[ -z "${total_steps}" ] && total_steps=$((320000*(max_epoch+1)/batch_size/n_gpu/update_freq))  
[ -z "${warmup_steps}" ] && warmup_steps=$((total_steps*10/100))  
  
# 创建检查点保存目录  
mkdir -p $SAVE_DIR  
  
echo "开始训练 AffinCraft 模型 (graphormer_large)..."  
echo "训练数据: $TRAIN_PKL_FILE"  
echo "验证数据: $VALID_PKL_FILE"  
echo "检查点将保存到: $SAVE_DIR"  
echo "动态参数设置:"  
echo "  - max_epoch: $max_epoch"  
echo "  - batch_size: $batch_size"  
echo "  - n_gpu: $n_gpu"  
echo "  - update_freq: $update_freq"  
echo "  - total_steps: $total_steps"  
echo "  - warmup_steps: $warmup_steps"  
  
# fairseq-train 训练命令  
CUDA_VISIBLE_DEVICES=0 fairseq-train \  
    --user-dir $USER_DIR \  
    --num-workers 8 \  
    --ddp-backend=legacy_ddp \  
    --dataset-source affincraft \  
    --train-pkl-pattern "$TRAIN_PKL_FILE" \  
    --valid-pkl-pattern "$VALID_PKL_FILE" \  
    --task graph_prediction \  
    --criterion l1_loss \  
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
    --warmup-updates $warmup_steps \  
    --total-num-update $total_steps \  
    --lr 2e-4 \  
    --end-learning-rate 1e-9 \  
    --batch-size $batch_size \  
    --update-freq $update_freq \  
    --fp16 \  
    --data-buffer-size 20 \  
    --max-epoch $max_epoch \  
    --save-dir $SAVE_DIR \  
    --log-interval 100 \  
    --save-interval-updates 5000 \  
    --validate-interval-updates 2500 \  
    --keep-interval-updates 10 \  
    --no-epoch-checkpoints \  
    --seed 42  
  
echo "训练完成。检查点已保存到: $SAVE_DIR"  
echo "最终参数总结:"  
echo "  - 使用架构: graphormer_large"  
echo "  - 总训练步数: $total_steps"  
echo "  - 预热步数: $warmup_steps"  
echo "  - 批次大小: $batch_size"  
echo "  - 更新频率: $update_freq"