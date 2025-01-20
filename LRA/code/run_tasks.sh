#!/bin/bash

task_list=("listops" "text" "retrieval" "image" "pathfinder32-curv_contour_length_14")
model=$2
opt=$3


for task in "${task_list[@]}" ; do
  for _ in {1..5} ; do
    CUDA_VISIBLE_DEVICES=$1 python run_tasks.py --model $model --task $task $opt
  done
done

