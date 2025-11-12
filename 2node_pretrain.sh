#!/bin/bash
#SBATCH -p gpu_4090
#SBATCH -N 2
#SBATCH --gres=gpu:8
#SBATCH --qos=gpugpu
#SBATCH -J AffinCraft-LMDB-2nodes
#SBATCH -o /data/run01/scw6f3q/zncao/affincraft-nn/logs/pretrain_2nodes.out
#SBATCH -e /data/run01/scw6f3q/zncao/affincraft-nn/logs/pretrain_2nodes.err

source /etc/profile.d/modules.sh
module load singularity/3.10.0
module load cuda/12.4
module load gcc/12.2

CONTAINER=~/run/12.4.1-devel-ubuntu20.04
ENV_PATH=/data/run01/scw6f3q/zncao/affincraft
CONDA_SH=/data/apps_4090/miniforge3/24.1.2/etc/profile.d/conda.sh
TRAIN_SCRIPT=/data/run01/scw6f3q/zncao/affincraft-nn/graphormer/train_finetune/pretrain_2node.sh
mkdir -p /data/run01/scw6f3q/zncao/affincraft-nn/logs

# 获取主节点 IPv4 地址
MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_IP=$(getent ahostsv4 "$MASTER_HOST" | awk '{print $1}' | head -n 1)
export MASTER_ADDR=$MASTER_IP
export MASTER_PORT=23456
export PYTORCH_USE_IPV6=0
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
echo "=========================================="
echo " 🎯 双节点训练"
echo " 节点列表: $SLURM_JOB_NODELIST"
echo " MASTER_ADDR=$MASTER_ADDR"
echo " MASTER_PORT=$MASTER_PORT"
echo "=========================================="

# 重要：不使用 --net/--network=host
srun --export=ALL --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
    singularity exec --nv \
    --bind /data/run01/scw6f3q:/data/run01/scw6f3q \
    --bind /data/apps_4090:/data/apps_4090 \
    --bind /ssd/home/scw6f3q:/ssd/home/scw6f3q \
    "$CONTAINER" \
    bash -c "
        set -e
        source $CONDA_SH
        conda activate $ENV_PATH
        export PYTHONUNBUFFERED=1
        export PYTORCH_USE_IPV6=0
        echo '[INFO] 当前节点:' \$(hostname)
        echo '[INFO] MASTER_ADDR:' \$MASTER_ADDR
        echo '[INFO] MASTER_PORT:' \$MASTER_PORT
        echo '[INFO] Python:' \$(which python)
        python -c 'import torch; print(\"[INFO] Torch version:\", torch.__version__); print(\"[INFO] CUDA available:\", torch.cuda.is_available()); print(\"[INFO] GPU count:\", torch.cuda.device_count())'
        echo '[INFO] 启动训练脚本: $TRAIN_SCRIPT'
        bash $TRAIN_SCRIPT
    "

echo "✅ 作业完成于 $(date)"