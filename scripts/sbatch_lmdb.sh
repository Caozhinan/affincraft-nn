#!/bin/bash

#SBATCH -p gpu_4090
#SBATCH -J lmdb_index
#SBATCH --gpus=1
#SBATCH -o build_index_%j.log
#SBATCH -e build_index_%j.err

# 加载您的conda环境
source /data/apps_4090/miniforge3/24.1.2/etc/profile.d/conda.sh
module load miniforge/24.1.2
conda activate /data/run01/scw6f3q/zncao/affincraft

# 执行Python脚本
# echo "Job started on $(date)"
# echo "Running on node: $(hostname)"
# echo "---"

# python /data/run01/scw6f3q/zncao/affincraft-nn/scripts/convert_affincraft_pkl_to_lmdb.py /data/run01/scw6f3q/zncao/data_pkl/valid.pkl /ssd/home/scw6f3q/lmdb/valid.lmdb
# echo "---"
# echo "Job finished on $(date)"

python check_lmdb.py /ssd/home/scw6f3q/lmdb/valid.lmdb