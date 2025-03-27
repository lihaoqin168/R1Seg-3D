#!/bin/bash

# 定义循环次数或参数范围
start_num=13
end_num=24

# 循环调用sh，并传入参数 num

num=$start_num
while [ $num -le $end_num ]; do
    expanded_num=$(printf "%04d" "$num")
    echo "Calling predict_Lamed_RSeg_x256_qwen.sh with num=$expanded_num"
    bash predict_Lamed_RSeg_x256_qwen.sh $expanded_num
    num=$((num + 1))
done

echo "All calls to predict_Lamed_RSeg_x256_qwen.sh completed. $start_num -- $end_num"