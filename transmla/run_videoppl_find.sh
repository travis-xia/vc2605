export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
model_path=/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/vc2605/VideoChat-Flash-Qwen2_5-2B_res448
save_results=search_results_videoppl/VideoChat-Flash-Qwen2_5-2B_res448_rank_search

python find_best_ranks_videoppl.py \
    --model-path $model_path \
    --save-results $save_results --qk-mqa-dim 64
