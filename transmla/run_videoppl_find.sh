export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
model_path=/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/vc2605/VideoChat_Flash_Qwen2_5_7B_1M_res224
save_results=search_results_videoppl/VideoChat-Flash-Qwen2_5-7B-1M_res224_rank_search

device=cuda:0

python find_best_ranks_videoppl.py \
    --model-path $model_path \
    --save-results $save_results \
    --device $device \
    --qk-mqa-dim 64
