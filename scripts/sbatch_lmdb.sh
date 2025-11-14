#!/bin/bash

#SBATCH -p gpu_4090
#SBATCH -J lmdb_index
#SBATCH --gpus=1
#SBATCH -o build_index_%j.log
#SBATCH -e build_index_%j.err

# 加载您的conda环境
source /etc/profile.d/modules.sh
module load singularity/3.10.0
module load cuda/12.4
module load gcc/12.2
module load miniforge/24.1.2

source /data/apps_4090/miniforge3/24.1.2/etc/profile.d/conda.sh

conda activate /data/run01/scw6f3q/zncao/affincraft


singularity exec --nv \
    --bind /data/run01/scw6f3q:/data/run01/scw6f3q \
    --bind /data/apps_4090:/data/apps_4090 \
    --bind /ssd/home/scw6f3q:/ssd/home/scw6f3q \
    ~/run/12.4.1-devel-ubuntu20.04 \
    bash -c "
        source /data/apps_4090/miniforge3/24.1.2/etc/profile.d/conda.sh
        conda activate /data/run01/scw6f3q/zncao/affincraft
        python clean_lmdb.py /ssd/home/scw6f3q/train_lmdb /ssd/home/scw6f3q/new_train_lmdb
    "


