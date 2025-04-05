#!/bin/bash
# -*- coding: utf-8 -*-
# 获取传入的参数
num=$1
lab=$2

# 使用参数
echo "Running A.sh with num=$num"
CUDA_VISIBLE_DEVICES=0 python /107556523204/haoqin/code/R1Seg-3D/infer_R1Seg3D_SAM.py \
    --sam_bert_path /107556523204/pretrain/clip-vit-base-patch32/ \
    --pretrained_model /107556523204/output/R1Seg-3D/R1Seg-3DSAM-step1/r1seg_3dsam.bin \
    --num_clicks 0 \
    --version v0 \
    --bf16 True \
    --output_dir /107556523204/haoqin/infer_R1Seg-3DSAM-step1_0nk/ \
    --output_nii ${num}_${lab}.nii.gz \
    --image_path /defaultShare/M3D_Data/M3D-Seg/M3D_Seg_npy/0011/$num/image.npy \
    --seg_path  /defaultShare/M3D_Data/M3D-Seg/M3D_Seg_npy/0011/$num/masks/mask_${lab}.npy
