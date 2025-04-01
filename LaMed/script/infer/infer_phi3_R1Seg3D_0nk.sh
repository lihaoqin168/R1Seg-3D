#!/bin/bash
# -*- coding: utf-8 -*-
# 获取传入的参数
num=$1
lab=$2

# 使用参数
echo "Running A.sh with num=$num"
CUDA_VISIBLE_DEVICES=0 python /107556523204/haoqin/code/R1Seg-3D/infer_R1Seg3D.py \
    --model_name_or_path /107556523204/output/M3D-ViT-RSeg/LaMed/output_model/LaMed-Phi3-4B-X256-ep10step4-2nkSegcnt \
    --num_clicks 0 \
    --model_type phi3 \
    --version v0 \
    --bf16 True \
    --description True \
    --output_dir /107556523204/haoqin/infer_phi3_R1Seg3D_0nk/ \
    --output_nii ${num}_${lab}.nii.gz \
    --image_path /defaultShare/M3D_Data/M3D-Seg/M3D_Seg_npy/0011/${num}/image.npy \
    --seg_path  /defaultShare/M3D_Data/M3D-Seg/M3D_Seg_npy/0011/${num}/masks/mask_${lab}.npy
