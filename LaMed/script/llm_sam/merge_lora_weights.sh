#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python /107556523204/haoqin/code/R1Seg-3D/merge_lora_weights_and_save_hf_model.py \
    --version v0 \
    --model_type phi3 \
    --model_name_or_path /107556523204/pretrain/Phi-3-mini-4k-instruct \
    --pretrain_vision_model /107556523204/output/R1Seg-3D/R1Seg-3DSAM-step1/r1seg_3dsam.bin \
    --model_with_lora /107556523204/output/LLM_Seg3D/LaMed-Lora-Phi3-4B-step3/model_with_lora.bin \
    --output_dir /107556523204/output/LLM_Seg3D/output_model/LaMed-Lora-Phi3-4B-step3/
