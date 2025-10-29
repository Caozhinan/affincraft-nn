seed=42
batch_size=32
#模型原始参数
ckpt_path="/xcfhome/zncao02/model_bap/Dynafomer/checkpoint/model2.pt"
#新的ckpt存储路径
save_dir="/xcfhome/zncao02/model_bap/Dynafomer/bindingnet/ckpt/high_dp_0"
#pkl文件的路径
dataset_name=custom:train_path=/xcfhome/zncao02/model_bap/Dynafomer/bindingnet/high/train.pkl,valid_path=/xcfhome/zncao02/model_bap/Dynafomer/bindingnet/high/valid.pkl,test_path=/xcfhome/zncao02/model_bap/Dynafomer/bindingnet/high/test.pkl
data_path=/xcfhome/zncao02/model_bap/Dynafomer/bindingnet/high/
lr=1e-5

torchrun --nproc_per_node=1 \
  $(which fairseq-train) \
  --user-dir "/xcfhome/zncao02/model_bap/Dynafomer/Dynaformer/dynaformer" \
  --num-workers 8 --ddp-backend=legacy_ddp \
  --finetune-from-model $ckpt_path \
  --dataset-name "$dataset_name" \
  --dataset-source pyg --data-path "$data_path" \
  --batch-size $batch_size --data-buffer-size 40 \
  --task graph_prediction_with_flag --criterion l1_loss_with_flag --arch graphormer_base --num-classes 1 \
  --lr $lr --end-learning-rate 1e-9 --lr-scheduler polynomial_decay --power 1 \
  --warmup-updates 68200 --total-num-update  1164060 \
  --update-freq 1 \
  --encoder-layers 4 --encoder-attention-heads 32 \
  --encoder-embed-dim 512 --encoder-ffn-embed-dim 512 \
  --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 --weight-decay 1e-5 \
  --optimizer adam --adam-betas "(0.9,0.999)" --adam-eps 1e-8 --flag-m 1 --flag-step-size 0.0001 --flag-mag 0.001 --clip-norm 5 \
  --fp16 --save-dir "$save_dir" --seed $seed --fingerprint \
  --max-nodes 600 --dist-head gbf3d \
  --num-dist-head-kernel 256 --num-edge-types 16384 \
  --max-epoch 60