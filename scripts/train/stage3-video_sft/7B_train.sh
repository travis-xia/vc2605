export OMP_NUM_THREADS=1
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1

export ACCELERATE_CPU_AFFINITY=1

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export AV_LOG_LEVEL=quiet  # 抑制 decord/pyav 的 [h264 @ ...] mmco 等 FFmpeg 刷屏


DATA_VERSION="data/stage3_short-long_mix_sft.yaml"
DATA_VERSION_CLEAN=$(basename "$DATA_VERSION")

VISION_MODEL_VERSION="umt-large"
VISION_MODEL_VERSION_CLEAN="umt-large"

LLM_VERSION="/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/VideoChat_Flash_Qwen2_5_7B_1M_res224"
LLM_VERSION_CLEAN="Qwen2_7B"

mm_projector_type=tome16_mlp_hd64
PROMPT_VERSION="qwen_2"

MID_RUN_NAME=$(basename "$LLM_VERSION")_$(date +"%Y%m%d_%H%M%S")
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

mkdir -p ./checkpoints/stage3-video_sft ./output_logs/stage3-video_sft

# --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \

NUM_GPUS=8
torchrun --nproc_per_node=${NUM_GPUS} \
    llava/train/train_mem.py \
    --deepspeed scripts/zero1.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_VERSION} \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --mm_vision_select_layer -2 \
    --mm_projector_type ${mm_projector_type} \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_nopad \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_nopad \
    --mm_newline_position nothing \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir ./checkpoints/stage3-video_sft/${MID_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 20 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 12 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --dataloader_drop_last True \
    --frames_upbound 512 \
    --frames_lowbound 64 \
    --time_msg short \
    --local_num_frames 4 \
    --vision_encode_type video_image \
    --sample_type dynamic_fps1 \
    --mm_local_num_frames 4 \
    --verbose_logging True 2>&1 | tee -a ./output_logs/stage3-video_sft/${MID_RUN_NAME}.log

# --attn_implementation sdpa \
    # --torch_compile True \
    # --torch_compile_backend "inductor" \