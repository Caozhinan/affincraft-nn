SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

save_dir=$1
[ -z "${save_dir}" ] && save_dir="$(realpath $SCRIPT_DIR/../../../checkpoint)"
dataset=$2
[ -z "${dataset}" ] && dataset="pdbbind:set_name=refined-set-2019-coreset-2016,cutoffs=5-5-5,seed=0"
data_path=$3
[ -z "${data_path}" ] && data_path="$(realpath $SCRIPT_DIR/../../../data)"
suffix=$4
[ -z "${suffix}" ] && suffix="_CASF2016"
echo Running $SCRIPT_DIR/../../dynaformer/evaluate/evaluate.py
echo Finding checkpoint files in: $save_dir
echo Using dataset: $dataset
echo Save downloaded data to $data_path
echo Save csv with suffix: $suffix

layers=4
num_head=32
hidden_size=512
ffn_size=512
dist_head=gbf3d
num_dist_head_kernel=256
num_edge_types=16384

python $SCRIPT_DIR/../../dynaformer/evaluate/evaluate.py \
  --split "test" --suffix "$suffix" \
  --user-dir "$(realpath $SCRIPT_DIR/../../dynaformer)" \
  --num-workers 16 --ddp-backend=legacy_ddp \
  --dataset-name $dataset \
  --dataset-source pyg --data-path $data_path \
  --batch-size 1 --data-buffer-size 20 \
  --task graph_prediction_with_flag --criterion l2_loss_with_flag --arch graphormer_base --num-classes 1 \
  --encoder-layers $layers --encoder-attention-heads $num_head \
  --encoder-embed-dim $hidden_size --encoder-ffn-embed-dim $ffn_size \
  --fp16 --save-dir "$save_dir" \
  --max-nodes 600 --dist-head $dist_head  \
  --num-dist-head-kernel $num_dist_head_kernel --num-edge-types $num_edge_types --fingerprint




