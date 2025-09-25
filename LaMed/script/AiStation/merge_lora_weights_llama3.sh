#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python /code/R1Seg-3D/merge_lora_weights_and_save_hf_model.py \
    --version v0 \
    --model_type llama3 \
    --model_name_or_path /pretrain/Llama-3.1-8B \
    --pretrain_vision_model /output/R1Seg-3D/R1Seg-3DSAM-step1/r1seg_3dsam.bin \
    --model_with_lora /output/R1Seg-3D/LaMed-finetune-rseg-Lora-llama3-8B-step4-ep12/model_with_lora.bin \
    --output_dir /output/R1Seg-3D/output_model/LaMed-finetune-rseg-Lora-llama3-8B-step4-ep12/
