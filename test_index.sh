#!/bin/bash

#SBATCH -p gpu_4090
#SBATCH -J pkl_index
#SBATCH --gpus=1
#SBATCH -o build_index_%j.log
#SBATCH -e build_index_%j.err

# 加载您的conda环境
module load miniforge/24.1.2
conda activate /data/run01/scw6f3q/zncao/affincraft

# 执行Python脚本
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "---"

python index_pkl.py /data/run01/scw6f3q/zncao/data_pkl/valid.pkl /data/run01/scw6f3q/zncao/data_pkl/valid.idx

echo "---"
echo "Job finished on $(date)"