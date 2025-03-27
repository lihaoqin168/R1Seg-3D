#!/bin/bash
# -*- coding: utf-8 -*-
CUDA_VISIBLE_DEVICES=0 python /107556523204/haoqin/code/R1Seg-3D/infer_R1Seg3D.py \
    --model_name_or_path /107556523204/output/M3D-ViT-RSeg/LaMed/output_model/LaMed-Phi3-4B-X256-ep10step4-2nkSegcnt \
    --num_clicks 2 \
    --model_type phi3 \
    --version v0 \
    --bf16 True \
    --description True \
    --output_dir /107556523204/haoqin/infer_R1Seg3D/ \
    --image_path /defaultShare/M3D_Data/M3D-Seg/M3D_Seg_npy/0011/s0733/image.npy \
    --seg_path  /defaultShare/M3D_Data/M3D-Seg/M3D_Seg_npy/0011/s0733/masks/mask_45.npy
