#!/bin/bash  
  
# 训练脚本：使用包含多个复合物的PKL文件进行AffinCraft模型分布式训练  
# 用法: bash train_affincraft_distributed.sh <train_pkl_file> <valid_pkl_file> [可选: <test_pkl_file>]  
  
# 分布式训练配置  
# 每个节点上的GPU数量  
GPUS_PER_NODE=8 # 根据你的节点实际GPU数量修改  
# 主节点地址 (多节点训练时，设置为第一个节点的IP地址)  
MASTER_ADDR="localhost" # 单节点训练时为localhost，多节点时设置为实际IP  
# 主节点端口 (确保端口未被占用)  
MASTER_PORT=29501  
# 节点数量  
NNODES=1 # 单节点训练时为1，多节点时设置为实际节点数量  
# 当前节点在所有节点中的排名 (0到NNODES-1)  
NODE_RANK=0 # 每个节点运行时需要设置不同的NODE_RANK  
  
# 检查参数数量  
if [ "$#" -lt 2 ]; then  
    echo "用法: bash $0 <训练PKL文件路径> <验证PKL文件路径> [可选: <测试PKL文件路径>]"  
    exit 1  
fi  
  
TRAIN_PKL_FILE=$1  
VALID_PKL_FILE=$2  
TEST_PKL_FILE=${3:-} # 可选的测试PKL文件  
  
SAVE_DIR="./affincraft_distributed_ckpts"  
USER_DIR="../../graphormer" # 确保这个路径指向你的graphormer目录  
  
# 创建检查点保存目录  
mkdir -p $SAVE_DIR  
  
echo "开始分布式训练 AffinCraft 模型..."  
echo "训练数据: $TRAIN_PKL_FILE"  
echo "验证数据: $VALID_PKL_FILE"  
if [ -n "$TEST_PKL_FILE" ]; then  
    echo "测试数据: $TEST_PKL_FILE"  
fi  
echo "检查点将保存到: $SAVE_DIR"  
echo "分布式配置: GPUS_PER_NODE=$GPUS_PER_NODE, NNODES=$NNODES, NODE_RANK=$NODE_RANK, MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"  
  
# 构建分布式参数  
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"  
  
# fairseq-train 训练命令  
python -m torch.distributed.launch $DISTRIBUTED_ARGS \  
  $(which fairseq-train) \  
    --user-dir $USER_DIR \  
    --num-workers 8 \  
    --ddp-backend=legacy_ddp \  
    --dataset-source affincraft \  
    --train-pkl-pattern "$TRAIN_PKL_FILE" \  
    --valid-pkl-pattern "$VALID_PKL_FILE" \  
    $( [ -n "$TEST_PKL_FILE" ] && echo "--test-pkl-pattern \"$TEST_PKL_FILE\"" ) \  
    --task graph_prediction \  
    --criterion l1_loss \  
    --arch graphormer_base \  
    --num-classes 1 \  
    --max-nodes 512 \  
    --attention-dropout 0.1 \  
    --act-dropout 0.1 \  
    --dropout 0.1 \  
    --optimizer adam \  
    --adam-betas '(0.9, 0.999)' \  
    --adam-eps 1e-8 \  
    --clip-norm 5.0 \  
    --weight-decay 0.01 \  
    --lr-scheduler polynomial_decay \  
    --power 1 \  
    --warmup-updates 10000 \  
    --total-num-update 100000 \  
    --lr 2e-4 \  
    --end-learning-rate 1e-9 \  
    --batch-size 16 \  
    --fp16 \  
    --data-buffer-size 20 \  
    --encoder-layers 12 \  
    --encoder-embed-dim 768 \  
    --encoder-ffn-embed-dim 768 \  
    --encoder-attention-heads 32 \  
    --max-epoch 1000 \  
    --save-dir $SAVE_DIR \  
    --log-interval 100 \  
    --save-interval-updates 5000 \  
    --validate-interval-updates 2500 \  
    --keep-interval-updates 10 \  
    --no-epoch-checkpoints \  
    --seed 42  
  
echo "训练完成。检查点已保存到: $SAVE_DIR"