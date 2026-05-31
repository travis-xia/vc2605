import torch
import torch.nn as nn
from typing import Optional, Tuple

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb_interleave

from utils import pca_calc, get_qkv_calibrate_outputs, evaluate_video_ppl, statistics_qkv_rmsnorm

from lora_qkv import LoraQKV  # 复用已有实现


def low_rank_qkv_videoppl(model, tokenizer, train_loader, test_loader, **kwargs):

    message = "Calibrating rope-removed model's qkv outputs"
    rm_rope_qkv_outputs = get_qkv_calibrate_outputs(model, train_loader, message)

    kv_lora_rank_list = kwargs.get("kv_lora_rank_list")
    default_kv_rank = kwargs.get("kv_lora_rank")

    for layer_idx, layer in enumerate(model.model.layers):
        layer_kv_rank = None
        if kv_lora_rank_list is not None:
            layer_kv_rank = kv_lora_rank_list[layer_idx]
        elif default_kv_rank is not None:
            layer_kv_rank = default_kv_rank
        else:
            layer_kv_rank = 512

        setattr(layer, "self_attn", LoraQKV(
            layer.self_attn,
            rm_rope_qkv_outputs["query"][layer_idx],
            rm_rope_qkv_outputs["key"][layer_idx],
            rm_rope_qkv_outputs["value"][layer_idx],
            q_lora_rank=kwargs["q_lora_rank"],
            qk_mqa_dim=kwargs["qk_mqa_dim"],
            collapse=kwargs["collapse"],
            kv_lora_rank=layer_kv_rank,
            use_qkv_norm=kwargs["use_qkv_norm"],
            balance_kv_ratio=kwargs["balance_kv_ratio"],
            rms_norm_eps=model.config.rms_norm_eps,
        ))

    if kwargs["use_qkv_norm"]:
        lora_qkv_outputs = get_qkv_calibrate_outputs(model, train_loader)
        for layer_idx, layer in enumerate(model.model.layers):
            statistics_qkv_rmsnorm(
                layer.self_attn,
                lora_qkv_outputs["q_a_proj"][layer_idx] if len(lora_qkv_outputs["q_a_proj"]) > layer_idx else None,
                lora_qkv_outputs["kv_a_proj"][layer_idx]
            )

    if test_loader:
        message = "Evaluating lora-qkv model's ppl"
        dataset_ppl = evaluate_video_ppl(model, tokenizer, test_loader, message)
        print(f'Low rank approximate QKV ppl: {dataset_ppl:.4f}')

    return model


