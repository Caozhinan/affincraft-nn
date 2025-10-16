#!/bin/bash
#SBATCH -p gpu_4090                 
#SBATCH --nodes=1                   
#SBATCH --gpus=8                    
#SBATCH -J AffinCraft-pretrain      
#SBATCH -o logs/pretrain-1.out     
#SBATCH -e logs/pretrain-1.err     


## 环境加载
export PYTHONWARNINGS="ignore::UserWarning:pkg_resources, ignore::FutureWarning:dgl.backend.pytorch.sparse, ignore::FutureWarning, ignore::UserWarning"
echo "[INFO] Job starting at $(date)"
module load singularity/3.10.0
module load cuda/12.4
module load gcc/12.2

# ----------------【路径设置】----------------
CONTAINER=~/run/12.4.1-devel-ubuntu20.04

# Conda 环境路径与初始化脚本
ENV_PATH=/data/run01/scw6f3q/zncao/affincraft
CONDA_SH=/data/apps_4090/miniforge3/24.1.2/etc/profile.d/conda.sh

# 训练脚本和数据
TRAIN_SCRIPT=/data/run01/scw6f3q/zncao/affincraft-nn/graphormer/train_finetune/md_train.sh
TRAIN_PKL=/data/run01/scw6f3q/zncao/data_pkl/train.pkl
VALID_PKL=/data/run01/scw6f3q/zncao/data_pkl/valid.pkl

# 创建日志目录
mkdir -p logs

# ----------------【任务信息输出】----------------
echo "=========================================="
echo " 🎯 AffinCraft GPU 任务启动"
echo " 节点:        $(hostname)"
echo " 作业ID:      $SLURM_JOB_ID"
echo " GPUs:        $CUDA_VISIBLE_DEVICES"
echo " 启动时间:    $(date)"
echo "=========================================="

# ----------------【容器执行逻辑】----------------
singularity exec --nv \
    --bind /data/run01/scw6f3q:/data/run01/scw6f3q \
    --bind /data/apps_4090:/data/apps_4090 \
    $CONTAINER \
    bash -c "
        set -e
        source $CONDA_SH
        conda activate $ENV_PATH

        echo '[INFO] Python path:' \$(which python)
        python -c 'import torch; print(\"[INFO] Torch version:\", torch.__version__); print(\"[INFO] CUDA available:\", torch.cuda.is_available())'

        bash $TRAIN_SCRIPT $TRAIN_PKL $VALID_PKL
    "

# ----------------【收尾信息】----------------
echo "=========================================="
echo " ✅ 任务完成于: $(date)"
echo "=========================================="