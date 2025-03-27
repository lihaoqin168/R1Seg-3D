#!/bin/bash

# run "accelerate config" first!
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1

accelerate launch  --config_file /107556523204/haoqin/code/default6_config.yaml /107556523204/haoqin/code/R1Seg-3D/train_RSeg.py \
    --version v0 \
    --model_name_or_path /107556523204/pretrain/Qwen2.5-7B \
    --model_type Qwen2.5 \
    --lora_enable False \
    --seg_enable False \
    --pretrain_vision_model /107556523204/output/R1Seg-3D/LaMed/output/M3DSAM-preVit-A40-X256-cntMmask2/checkpoint-126600/model.safetensors \
    --tune_mm_mlp_adapter True \
    --bf16 True \
    --output_dir /107556523204/output/R1Seg-3D/LaMed/output/LaMed-qwen-7B-mmproj-X256-step3 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 1000000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True\
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    --data_root /defaultShare/M3D_Data/ \
    --seg_data_path /defaultShare/M3D_Data/M3D-Seg/M3D_Seg_npy \
    --refseg_data_path /defaultShare/M3D_Data/M3D-RefSeg/M3D_RefSeg_npy \
    --refseg_data_train_path /defaultShare/M3D_Data/M3D-RefSeg/M3D_RefSeg_cap_train.csv \
    --refseg_data_test_path /defaultShare/M3D_Data/M3D-RefSeg/M3D_RefSeg_cap_test.csv

