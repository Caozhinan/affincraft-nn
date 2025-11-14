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

# ================================
# 🔧 网络与分布式训练配置
# ================================

# 获取主节点 IPv4 地址（rank0）
MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_IP=$(getent ahostsv4 "$MASTER_HOST" | awk '{print $1}' | head -n 1)
export MASTER_ADDR=$MASTER_IP
export MASTER_PORT=23456

# 关闭 IPv6，强制使用 IPv4
export PYTORCH_USE_IPV6=0

# 指定正确网卡接口 (根据 ip addr show 得知是 bond0)
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0

# 非 InfiniBand 网络需禁用 RDMA
export NCCL_IB_DISABLE=1

# 强制 NCCL 输出初始化与拓扑调试日志
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=GRAPH
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# 防止部分节点上 NCCL 共享内存冲突（某些 HPC 环境常见）
export NCCL_SHM_DISABLE=1

# 碰到通信问题时，提前触发 NCCL 异常检测
export NCCL_ASYNC_ERROR_HANDLING=1

echo "=========================================="
echo " 🎯 双节点训练启动"
echo " 节点列表: $SLURM_JOB_NODELIST"
echo " MASTER_ADDR=$MASTER_ADDR"
echo " MASTER_PORT=$MASTER_PORT"
echo " NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo "=========================================="

# ================================
# 🚀 启动容器与训练
# ================================
# 保留原有结构，仅修改必要环境变量
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
        export NCCL_SOCKET_IFNAME=bond0
        export GLOO_SOCKET_IFNAME=bond0
        export NCCL_IB_DISABLE=1
        export NCCL_DEBUG=INFO
        export NCCL_DEBUG_SUBSYS=GRAPH
        export TORCH_DISTRIBUTED_DEBUG=DETAIL
        export NCCL_ASYNC_ERROR_HANDLING=1
        export NCCL_SHM_DISABLE=1

        echo '------------------------------------------'
        echo '[INFO] 当前节点:' \$(hostname)
        echo '[INFO] MASTER_ADDR:' \$MASTER_ADDR
        echo '[INFO] MASTER_PORT:' \$MASTER_PORT
        echo '[INFO] Python:' \$(which python)
        python -c 'import torch; print(\"[INFO] Torch version:\", torch.__version__); print(\"[INFO] CUDA available:\", torch.cuda.is_available()); print(\"[INFO] GPU count:\", torch.cuda.device_count())'
        echo '[INFO] 启动训练脚本: $TRAIN_SCRIPT'
        bash $TRAIN_SCRIPT
        echo '[INFO] 训练脚本完成'
    "

echo "✅ 作业完成于 $(date)"