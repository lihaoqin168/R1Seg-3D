#!/bin/bash
#Qwen2.5-7B
# run "accelerate config" first!
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1

accelerate launch  --config_file /107556523204/haoqin/code/default6_config.yaml /107556523204/haoqin/code/R1Seg-3D/train_R1Seg3D.py \
    --version v0 \
    --num_clicks 2 \
    --model_name_or_path /107556523204/pretrain/Qwen2.5-7B \
    --pretrain_mm_mlp_adapter /107556523204/output/R1Seg-3D/LaMed-mmproj-qwen-7B-step2/mm_projector.bin \
    --model_type Qwen2.5 \
    --lora_enable True \
    --seg_enable True \
    --tune_vision_module False \
    --pretrain_vision_model /107556523204/output/R1Seg-3D/R1Seg-3DSAM-step1/r1seg_3dsam.bin \
    --tune_mm_mlp_adapter False \
    --bf16 True \
    --output_dir /107556523204/output/R1Seg-3D/LaMed-Lora-Qwen25-7B-step3/ \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
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

