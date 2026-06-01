export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
model_path=/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/vc2605/VideoChat_Flash_Qwen2_5_7B_1M_res224
save_path=outputs/videoppl/VideoChat-Flash-Qwen2_5-7B-1M_res224_mla_2605

device=cuda:0

python converter_mllm_videoppl.py \
    --model-path $model_path \
    --save-path $save_path \
    --device $device \
    --qk-mqa-dim 64
