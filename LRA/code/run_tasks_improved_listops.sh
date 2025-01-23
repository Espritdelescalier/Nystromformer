#!/bin/bash

select_list=("sum" "random" "abs" "step" "embed")
opt=$2

for select_type in "${select_list[@]}" ; do
  for _ in {1..5} ; do
    CUDA_VISIBLE_DEVICES=$1 python run_tasks_improved.py --model curformer --task listops --select_type $select_type $opt
  done
done

