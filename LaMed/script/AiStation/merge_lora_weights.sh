#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python /code/R1Seg-3D/merge_lora_weights_and_save_hf_model.py \
    --version v0 \
    --model_type phi3 \
    --model_name_or_path /pretrain/Phi-3-mini-4k-instruct \
    --pretrain_vision_model /output/R1Seg-3D/R1Seg-3DSAM-step1/r1seg_3dsam.bin \
    --model_with_lora /output/LLM_Seg3D/LaMed-Lora-Phi3-4B-step3-ep3/model_with_lora.bin \
    --output_dir /output/LLM_Seg3D/output_model/LaMed-Lora-Phi3-4B-step3-ep3/
