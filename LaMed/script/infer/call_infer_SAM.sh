#!/bin/bash
# -*- coding: utf-8 -*-

# 定义数组参数范围
array=('s0030' 's0393' 's0399' 's0639' 's0733' 's0776' 's0830' 's1094' 's1106' 's1136' 's1171' 's1340')
lab="$1"  # 双引号包裹以处理可能的特殊字符

# 循环调用脚本并传递参数
for item in "${array[@]}"; do
    echo "Calling inferR1Seg3D_SAM_0nk.sh with item=${item} lab=${lab} "
    bash inferR1Seg3D_SAM_0nk.sh "${item}" "${lab}"
done

echo "All calls to inferR1Seg3D_SAM_0nk.sh completed. Elements processed: ${array[*]}"