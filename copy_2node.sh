#!/bin/bash
#SBATCH -p gpu_4090
#SBATCH -N 2
#SBATCH --gres=gpu:8
#SBATCH --qos=gpugpu  
#SBATCH -J AffinCraft-LMDB-2nodes
#SBATCH -o /data/run01/scw6f3q/zncao/affincraft-nn/logs/pretrain_2nodes-%j.out
#SBATCH -e /data/run01/scw6f3q/zncao/affincraft-nn/logs/pretrain_2nodes-%j.err

########################################
#   环境与路径设置
########################################

echo "[INFO] Job starting at $(date)"

# 加载系统模块
source /etc/profile.d/modules.sh
module load singularity/3.10.0
module load cuda/12.4
module load gcc/12.2

# 容器镜像路径
CONTAINER=~/run/12.4.1-devel-ubuntu20.04

# Conda 环境路径及初始化脚本
ENV_PATH=/data/run01/scw6f3q/zncao/affincraft
CONDA_SH=/data/apps_4090/miniforge3/24.1.2/etc/profile.d/conda.sh

# 实际训练脚本路径
TRAIN_SCRIPT=/data/run01/scw6f3q/zncao/affincraft-nn/graphormer/train_finetune/md_multi_gpu_2nodes.sh

# 日志目录
mkdir -p /data/run01/scw6f3q/zncao/affincraft-nn/logs

########################################
#   打印任务信息
########################################
echo "=========================================="
echo " 🎯 AffinCraft 双节点GPU训练 (LMDB格式)"
echo " 节点数:        $SLURM_NNODES"
echo " 节点列表:      $SLURM_JOB_NODELIST"
echo " 作业ID:        $SLURM_JOB_ID"
echo " 总GPU数:       16 (8×2)"
echo " 启动时间:      $(date)"
echo "=========================================="

########################################
#   分布式容器执行逻辑
########################################
# srun 会确保在两个节点上各启动一个进程
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
    singularity exec --nv \
    --bind /data/run01/scw6f3q:/data/run01/scw6f3q \
    --bind /data/apps_4090:/data/apps_4090 \
    --bind /ssd/home/scw6f3q:/ssd/home/scw6f3q \
    $CONTAINER \
    bash -c "
        set -euo pipefail
        source $CONDA_SH
        conda activate $ENV_PATH
        export PYTHONUNBUFFERED=1
        export PYTHONWARNINGS='ignore::UserWarning:pkg_resources, ignore::FutureWarning:dgl.backend.pytorch.sparse, ignore::FutureWarning, ignore::UserWarning'

        echo '[INFO] 当前节点:' \$(hostname)
        echo '[INFO] Python:' \$(which python)
        python -c 'import torch; print(\"[INFO] Torch version:\", torch.__version__); print(\"[INFO] CUDA available:\", torch.cuda.is_available()); print(\"[INFO] GPU count:\", torch.cuda.device_count())'

        echo '[INFO] 开始执行训练脚本: $TRAIN_SCRIPT'
        bash $TRAIN_SCRIPT
    "

########################################
#   收尾信息输出
########################################
echo "=========================================="
echo " ✅ 任务完成于: $(date)"
echo "=========================================="
