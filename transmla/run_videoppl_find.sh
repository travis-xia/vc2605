export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
model_path=/inspire/hdd/project/crosstemporalknowledge/xiacheng-240108120111/my/TransMLA-cvpr/VideoChat_Flash_Qwen2_5_7B_1M_res224
save_results=search_results_videoppl

python find_best_ranks_videoppl.py \
    --model-path $model_path \
    --save-results $save_results
