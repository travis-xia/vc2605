import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional, Tuple
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from utils import get_qkv_calibrate_outputs, evaluate_video_ppl

from partial_rope import PartialRope  # 复用已有实现


def partial_rope_videoppl(model, tokenizer, train_loader, test_loader, **kwargs):

    freqfold = kwargs["freqfold"]
    collapse = kwargs["collapse"]

    message = "Calibrating original model's qkv outputs"
    ori_qkv_outputs = get_qkv_calibrate_outputs(model, train_loader, message)

    def partial_rope_freqfold(model, ori_qkv_outputs, test_loader, freqfold: int, collapse):
        for layer_idx, layer in enumerate(model.model.layers):
            setattr(layer, "self_attn", PartialRope(
                layer.self_attn,
                ori_qkv_outputs["key"][layer_idx],
                freqfold=freqfold,
                collapse=collapse,
            ))

        # if test_loader:
        #     message = f"Evaluating partial-rope model's ppl, freqfold={freqfold}"
        #     dataset_ppl = evaluate_video_ppl(model, tokenizer, test_loader, message)
        #     print(f'Partial RoPE ppl, freqfold={freqfold}: {dataset_ppl:.4f}')
        #     return model, dataset_ppl
        # else:
        #     return model, None
        return model, None

    if freqfold != "auto":
        freqfold = int(freqfold)
        return partial_rope_freqfold(model, ori_qkv_outputs, test_loader, freqfold, collapse)[0]
    else:
        assert test_loader is not None, "test_loader is required for auto freqfold detection"
        device = model.device
        model_original = model.to("cpu")

        print(f"Auto freqfold detection...")

        best_freqfold = freqfold = collapse
        best_ppl = float("inf")
        while freqfold <= model_original.config.head_dim // 2:
            model = deepcopy(model_original)
            model = model.to(device)
            model, ppl = partial_rope_freqfold(model, ori_qkv_outputs, test_loader, freqfold, collapse)
            if ppl < best_ppl:
                best_ppl = ppl
                best_freqfold = freqfold
                freqfold *= 2
            else:
                break

        model = deepcopy(model_original)
        model = model.to(device)
        model, _ = partial_rope_freqfold(model, ori_qkv_outputs, None, best_freqfold, collapse)

        print(f"Best freqfold: {best_freqfold}")

        return model, best_freqfold


