#!/bin/bash
# -*- coding: utf-8 -*-
# 获取传入的参数
num=$1

# 使用参数
echo "Running A.sh with num=$num"

# run "accelerate config" first!
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1
# accelerate launch  --config_file /107556523204/haoqin/code/default4_config.yaml  
CUDA_VISIBLE_DEVICES=0 python /107556523204/haoqin/code/val_R1Seg-3D/eval_Lamed_RSeg.py \
    --model_name_or_path /107556523204/output/R1Seg-3D/output_model/LaMed-finetune-rseg-Lora-llama3-8B-step4-ep12 \
    --num_clicks 2 \
    --model_type llama3 \
    --version v0 \
    --test_mode True \
    --bf16 True \
    --description True \
    --dataset_code $num \
    --output_dir /107556523204/haoqin/val_R1Seg-3D/eval_llama3_ep12/ \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.01 \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True\
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --data_root /defaultShare/M3D_Data/ \
    --seg_data_path /defaultShare/M3D_Data/M3D-Seg/M3D_Seg_npy \
    --refseg_data_path /defaultShare/M3D_Data/M3D-RefSeg/M3D_RefSeg_npy \
    --refseg_data_train_path /defaultShare/M3D_Data/M3D-RefSeg/M3D_RefSeg_cap_train.csv \
    --refseg_data_test_path /defaultShare/M3D_Data/M3D-RefSeg/M3D_RefSeg_cap_test.csv
