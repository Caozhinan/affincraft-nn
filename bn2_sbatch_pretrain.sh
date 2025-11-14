#!/bin/bash
#SBATCH -p gpu_4090
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH -J AffinCraft-LMDB-pretrain
#SBATCH -o /data/run01/scw6f3q/zncao/affincraft-nn/logs/bn2_pretrain.out
#SBATCH -e /data/run01/scw6f3q/zncao/affincraft-nn/logs/bn2_pretrain.err

## 环境加载
export PYTHONWARNINGS="ignore::UserWarning:pkg_resources, ignore::FutureWarning:dgl.backend.pytorch.sparse, ignore::FutureWarning, ignore::UserWarning"
echo "[INFO] Job starting at $(date)"
source /etc/profile.d/modules.sh
module load singularity/3.10.0
module load cuda/12.4
module load gcc/12.2

# ----------------【路径设置】----------------
CONTAINER=~/run/12.4.1-devel-ubuntu20.04

# Conda 环境路径与初始化脚本
ENV_PATH=/data/run01/scw6f3q/zncao/affincraft
CONDA_SH=/data/apps_4090/miniforge3/24.1.2/etc/profile.d/conda.sh

# 训练脚本路径
TRAIN_SCRIPT='/data/run01/scw6f3q/zncao/affincraft-nn/graphormer/train_finetune/md_bak.sh'

# 创建日志目录
mkdir -p logs

# ----------------【任务信息输出】----------------
echo "=========================================="
echo " 🎯 AffinCraft GPU 任务启动 (LMDB格式)"
echo " 节点:        $(hostname)"
echo " 作业ID:      $SLURM_JOB_ID"
echo " GPUs:        $CUDA_VISIBLE_DEVICES"
echo " 启动时间:    $(date)"
echo "=========================================="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# ----------------【容器执行逻辑】----------------
singularity exec --nv \
    --bind /data/run01/scw6f3q:/data/run01/scw6f3q \
    --bind /data/apps_4090:/data/apps_4090 \
    --bind /ssd/home/scw6f3q:/ssd/home/scw6f3q \
    $CONTAINER \
    bash -c "
        set -e
        source $CONDA_SH
        conda activate $ENV_PATH
        export PYTHONUNBUFFERED=1
        echo '[INFO] Python path:' \$(which python)
        python -c 'import torch; print(\"[INFO] Torch version:\", torch.__version__); print(\"[INFO] CUDA available:\", torch.cuda.is_available())'

        bash $TRAIN_SCRIPT
    "

# ----------------【收尾信息】----------------
echo "=========================================="
echo " ✅ 任务完成于: $(date)"
echo "=========================================="
