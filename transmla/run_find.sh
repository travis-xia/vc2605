#!/bin/bash

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
model_path=/inspire/hdd/project/crosstemporalknowledge/xiacheng-240108120111/my/TransMLA-cvpr/VideoChat_Flash_Qwen2_5_7B_1M_res224
save_results=search_results/VideoChat-Flash-Qwen2_5-7B-1M_res224_rank_search
eval_batch_size=4

python find_best_ranks.py \
    --model-path $model_path \
    --save-results $save_results \
    --freqfold 4 \
    --ppl-eval-batch-size $eval_batch_size
