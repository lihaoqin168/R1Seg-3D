#!/bin/bash
# -*- coding: utf-8 -*-

# 定义循环次数或参数范围
start_num=11
end_num=24

# 循环调用sh，并传入参数 num

num=$start_num
while [ $num -le $end_num ]; do
    expanded_num=$(printf "%04d" "$num")
    echo "Calling predict_llm_RSeg3D.sh with num=$expanded_num"
#    bash predict_llm_RSeg3D.sh $expanded_num
    bash predict_llm_RSeg3D_step3.sh $expanded_num
    num=$((num + 1))
done

echo "All calls to predict_llm_RSeg3D.sh completed. $start_num -- $end_num"
