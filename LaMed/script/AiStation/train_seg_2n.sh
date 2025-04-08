#!/bin/bash

# run "accelerate config" first!
#export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=1800  # 1800秒 = 30分钟
accelerate launch --config_file /107556523204/haoqin/code/default6_config.yaml /107556523204/haoqin/code/R1Seg-3D/train_R1Seg3DSAM.py \
    --sam_bert_path /107556523204/pretrain/clip-vit-base-patch32/ \
    --pretrained_model /107556523204/pretrain/SegVol/vit_pretrain.ckpt \
    --num_clicks 1 \
    --version v0 \
    --local_loss False \
    --gather_loss True \
    --bf16 True \
    --output_dir /107556523204/output/R1Seg-3D/R1Seg-3DSAM-step1 \
    --num_train_epochs 100 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.01 \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True\
    --dataloader_num_workers 10 \
    --report_to tensorboard \
    --data_root /defaultShare/M3D_Data/ \
    --seg_data_path /defaultShare/M3D_Data/M3D-Seg/M3D_Seg_npy \
    --refseg_data_path /defaultShare/M3D_Data/M3D-RefSeg/M3D_RefSeg_npy \
    --refseg_data_train_path /defaultShare/M3D_Data/M3D-RefSeg/M3D_RefSeg_cap_train.csv \
    --refseg_data_test_path /defaultShare/M3D_Data/M3D-RefSeg/M3D_RefSeg_cap_test.csv
