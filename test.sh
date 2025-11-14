#!/bin/bash
#SBATCH -p gpu_4090                         
#SBATCH --gpus=1                   
#SBATCH -J test   
#SBATCH -o logs/test-2.out     
#SBATCH -e logs/test-2.err     


source /etc/profile.d/modules.sh  
module load singularity/3.10.0
module load cuda/12.4
module load gcc/12.2

singularity exec --nv \
    --bind /data/run01/scw6f3q:/data/run01/scw6f3q \
    --bind /data/apps_4090:/data/apps_4090 \
    --bind /ssd/home/scw6f3q:/ssd/home/scw6f3q \
    ~/run/12.4.1-devel-ubuntu20.04 \
    bash -c "  
        source /data/apps_4090/miniforge3/24.1.2/etc/profile.d/conda.sh  
        conda activate /data/run01/scw6f3q/zncao/affincraft  
        python /data/run01/scw6f3q/zncao/affincraft-nn/diagnose_dataloader.py  
    "