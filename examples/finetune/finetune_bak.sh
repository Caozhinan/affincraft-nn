seed=42
batch_size=16
ckpt_path="/xcfhome/zncao02/model_bap/Dynafomer/checkpoint/model2.pt"
save_dir="/xcfhome/zncao02/model_bap/Dynafomer/bindingnet/ckpt/all/"
dataset_name=custom:train_path=/xcfhome/zncao02/model_bap/Dynafomer/bindingnet/all/train.pkl,valid_path=/xcfhome/zncao02/model_bap/Dynafomer/bindingnet/all/valid.pkl,test_path=/xcfhome/zncao02/model_bap/Dynafomer/bindingnet/all/test.pkl
data_path=/xcfhome/zncao02/model_bap/Dynafomer/bindingnet/all/
lr=1e-6

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
  --warmup-updates 58500 --total-num-update 975000 \
  --update-freq 4 \
  --encoder-layers 4 --encoder-attention-heads 32 \
  --encoder-embed-dim 512 --encoder-ffn-embed-dim 512 \
  --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.1 --weight-decay 1e-5 \
  --optimizer adam --adam-betas "(0.9,0.999)" --adam-eps 1e-8 --flag-m 1 --flag-step-size 0.0001 --flag-mag 0.001 --clip-norm 5 \
  --fp16 --save-dir "$save_dir" --seed $seed --fingerprint \
  --max-nodes 600 --dist-head gbf3d  \
  --num-dist-head-kernel 256 --num-edge-types 16384 \
  --max-epoch 60