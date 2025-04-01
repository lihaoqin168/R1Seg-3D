#!/bin/bash
# -*- coding: utf-8 -*-

# 定义循环次数或参数范围
array=('s0030' 's0393' 's0399' 's0639' 's0733' 's0776' 's0830' 's1094' 's1106' 's1136' 's1171' 's1340')
lab="$1"  # 双引号包裹以处理可能的特殊字符

# 循环调用sh，并传入参数 array
for item in "${array[@]}"; do
    echo "Calling infer_llama3_R1Seg3D_2nk.sh with item=${item} lab=${lab} "
    bash infer_llama3_R1Seg3D_2nk.sh "${item}" "${lab}"
done

echo "All calls to infer_llama3_R1Seg3D_2nk.sh completed. Elements processed: ${array[*]}"
