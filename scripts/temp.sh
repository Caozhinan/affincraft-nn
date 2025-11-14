# # #!/bin/bash


module load singularity/3.10.0
singularity exec --nv \
    --bind /data/run01/scw6f3q:/data/run01/scw6f3q \
    --bind /data/apps_4090:/data/apps_4090 \
    --bind /ssd/home/scw6f3q:/ssd/home/scw6f3q \
    ~/run/12.4.1-devel-ubuntu20.04 \
    bash -c "
        source /data/apps_4090/miniforge3/24.1.2/etc/profile.d/conda.sh
        conda activate /data/run01/scw6f3q/zncao/affincraft
        python check_lmdb.py /ssd/home/scw6f3q/valid_lmdb
    "





# # #SBATCH -p gpu_4090
# # #SBATCH -J lmdb_index
# # #SBATCH --gpus=1
# # #SBATCH -o build_index_%j.log
# # #SBATCH -e build_index_%j.err

# # # 加载您的conda环境
# # source /data/apps_4090/miniforge3/24.1.2/etc/profile.d/conda.sh
# # module load miniforge/24.1.2
# # conda activate /data/run01/scw6f3q/zncao/affincraft


# # python split_lmdb.py \
# #     /ssd/home/scw6f3q/lmdb/valid.lmdb \
# #     /ssd/home/scw6f3q/valid_lmdb \
# #     /ssd/home/scw6f3q/test_lmdb \
#     42