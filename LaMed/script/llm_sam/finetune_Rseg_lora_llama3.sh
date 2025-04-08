#!/bin/bash
#Phi-3-mini-4k-instruct
# run "accelerate config" first!
#export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=1800

accelerate launch  --config_file /107556523204/haoqin/code/default6_config.yaml /107556523204/haoqin/code/R1Seg-3D/train_LLM_Seg3D.py \
    --version v0 \
    --model_name_or_path /107556523204/pretrain/Llama-3.1-8B \
    --pretrain_mllm /107556523204/output/LLM_Seg3D/LaMed-Lora-Llama-8B-step3/model_with_lora.bin \
    --model_type llama3 \
    --lora_enable True \
    --tune_vision_module True \
    --seg_enable True \
    --tune_mm_mlp_adapter false \
    --pretrain_vision_model /107556523204/output/R1Seg-3D/R1Seg-3DSAM-step1/r1seg_3dsam.bin \
    --bf16 True \
    --output_dir /107556523204/output/LLM_Seg3D/LaMed-finetune-rseg-Lora-Llama-8B-step4 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.5 \
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
    --dataloader_num_workers 16 \
    --report_to tensorboard \
    --data_root /defaultShare/M3D_Data/ \
    --seg_data_path /defaultShare/M3D_Data/M3D-Seg/M3D_Seg_npy \
    --refseg_data_path /defaultShare/M3D_Data/M3D-RefSeg/M3D_RefSeg_npy \
    --refseg_data_train_path /defaultShare/M3D_Data/M3D-RefSeg/M3D_RefSeg_cap_train.csv \
    --refseg_data_test_path /defaultShare/M3D_Data/M3D-RefSeg/M3D_RefSeg_cap_test.csv



