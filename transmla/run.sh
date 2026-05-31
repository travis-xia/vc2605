export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
model_path=/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/vc2605/VideoChat-Flash-Qwen2_5-2B_res448
save_path=outputs/VideoChat-Flash-Qwen2_5-2B_res448_mla_2605

python converter_mllm.py \
    --model-path $model_path \
    --save-path $save_path --qk-mqa-dim 64