#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python /107556523204/haoqin/code/R1Seg-3D/merge_lora_weights_and_save_hf_model.py \
    --version v0 \
    --model_type llama3 \
    --model_name_or_path /107556523204/pretrain/Llama-3.1-8B \
    --pretrain_vision_model /107556523204/output/R1Seg-3D/R1Seg-3DSAM-step1/r1seg_3dsam.bin \
    --model_with_lora /107556523204/output/R1Seg-3D/LaMed-finetune-rseg-Lora-llama3-8B-step4-ep12/model_with_lora.bin \
    --output_dir /107556523204/output/R1Seg-3D/output_model/LaMed-finetune-rseg-Lora-llama3-8B-step4-ep12/
