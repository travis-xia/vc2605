# coding=utf-8
# Borrows some implementations from https://github.com/Cooperx521/PyramidDrop, thanks!
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen2 model with VideoChat-Flash token compression."""
import math
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from .constants import IGNORE_INDEX


logger = logging.get_logger(__name__)


def _ensure_layer_types(config: Qwen2Config) -> None:
    if getattr(config, "layer_types", None):
        return
    layer_types = ["full_attention"] * config.num_hidden_layers
    if getattr(config, "use_sliding_window", False) and getattr(config, "sliding_window", None) is not None:
        max_window_layers = getattr(config, "max_window_layers", 0)
        for idx in range(config.num_hidden_layers):
            if idx >= max_window_layers:
                layer_types[idx] = "sliding_attention"
    config.layer_types = layer_types


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


@use_kernel_forward_from_hub("RMSNorm")
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # PartialRope / LoraQKV 等 transmla 模块仍使用 past_key_value
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen2DecoderLayer,
        "attentions": Qwen2Attention,
    }


class Qwen2RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Qwen2Config, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen2Model_Flash(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        _ensure_layer_types(config)
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in config.layer_types
        self._attn_implementation = config._attn_implementation

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _build_causal_mask_mapping(
        self,
        attention_mask: Optional[torch.Tensor],
        inputs_embeds: torch.Tensor,
        cache_position: torch.LongTensor,
        past_key_values: Optional[Cache],
        position_ids: torch.LongTensor,
    ) -> dict:
        if self._attn_implementation == "flash_attention_2" and attention_mask is not None and past_key_values is not None:
            is_padding_right = attention_mask[:, -1].sum().item() != inputs_embeds.shape[0]
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        return causal_mask_mapping

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, Tuple[BaseModelOutputWithPast, Optional[torch.Tensor]]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            causal_mask_mapping = self._build_causal_mask_mapping(
                attention_mask, inputs_embeds, cache_position, past_key_values, position_ids
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            rank_layer = layer_idx + 1
            if rank_layer in getattr(self, "llm_compress_layer_list", []):
                if hidden_states.shape[1] != 1:
                    stage = self.llm_compress_layer_list.index(rank_layer)
                    position_ids, attention_mask, hidden_states, labels = self.video_level_compress(
                        cur_num=stage,
                        rank_layer=rank_layer,
                        features=hidden_states,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
                    position_embeddings = self.rotary_emb(hidden_states, position_ids)
                    causal_mask_mapping = self._build_causal_mask_mapping(
                        attention_mask, hidden_states, cache_position, past_key_values, position_ids
                    )
                else:
                    stage = self.llm_compress_layer_list.index(rank_layer)
                    cur_visual_length = [
                        int(cur_image_token * self.llm_image_token_ratio_list[stage])
                        for cur_image_token in self.num_image_token_lens
                    ]
                    next_visual_length = [
                        int(cur_image_token * self.llm_image_token_ratio_list[stage + 1])
                        for cur_image_token in self.num_image_token_lens
                    ]
                    new_position_ids = []
                    for idx, cur_position_ids in enumerate(position_ids):
                        cur_position_ids = cur_position_ids - (cur_visual_length[idx] - next_visual_length[idx])
                        new_position_ids.append(cur_position_ids)
                    assert idx == 0, idx
                    position_ids = torch.tensor(new_position_ids, dtype=torch.long).unsqueeze(0)
                    position_embeddings = self.rotary_emb(hidden_states, position_ids)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            outputs = tuple(v for v in [hidden_states, past_key_values if use_cache else None, all_hidden_states, all_self_attns] if v is not None)
            return outputs, labels

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        ), labels

    def video_level_compress(
        self, cur_num, rank_layer, features, position_ids, attention_mask, labels
    ):
        if self.llm_compress_type == "uniform0_attention":
            if cur_num == 0:
                llm_compress_type = "uniform"
            else:
                llm_compress_type = "attention"
        else:
            llm_compress_type = self.llm_compress_type

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if position_ids is None:
            position_ids = torch.arange(0, features.shape[1], dtype=torch.long, device=features.device).unsqueeze(0)

        if getattr(self.config, "tokenizer_padding_side", "right") == "right":
            batch_size = features.shape[0]
            image_tokens = [
                int(cur_image_token * self.llm_image_token_ratio_list[cur_num])
                for cur_image_token in self.num_image_token_lens
            ]
            keep_length = [
                int(cur_image_token * self.llm_image_token_ratio_list[cur_num + 1])
                for cur_image_token in self.num_image_token_lens
            ]

            features_list = []
            attention_mask_list = []
            labels_list = []

            if attention_mask is None:
                attention_mask = torch.ones((batch_size, features.shape[1]), dtype=torch.bool, device=features.device)
            else:
                attention_mask = attention_mask.bool()
            if labels is None:
                labels = torch.full((batch_size, features.shape[1]), IGNORE_INDEX, device=features.device)

            if "attention" in llm_compress_type:
                hidden_states = features.clone().detach()
                self_attn = self.layers[rank_layer].self_attn
                hidden_states = self.layers[rank_layer].input_layernorm(hidden_states)

                num_heads = self.config.num_attention_heads
                num_key_value_heads = self.config.num_key_value_heads
                head_dim = self_attn.head_dim

                bsz, q_len, _ = hidden_states.size()

                query_states = self_attn.q_proj(hidden_states)
                key_states = self_attn.k_proj(hidden_states)

                query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

                cos, sin = self.rotary_emb(hidden_states, position_ids)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                key_states = repeat_kv(key_states, self_attn.num_key_value_groups)

                eager_attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, q_len), hidden_states, past_key_values_length=0
                ).to(device=query_states.device)

            features = [cur_features[cur_attention_mask] for cur_features, cur_attention_mask in zip(features, attention_mask)]
            labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
            attention_mask = [
                cur_attention_mask[cur_attention_mask]
                for cur_attention_mask, cur_attention_mask in zip(attention_mask, attention_mask)
            ]

            for i in range(batch_size):
                image_index = self.first_image_token_position[i]
                if image_index == -1:
                    features_list.append(features[i])
                    attention_mask_list.append(attention_mask[i])
                    labels_list.append(labels[i])
                    continue

                if "attention" in llm_compress_type:
                    cur_key_states = key_states[i]
                    cur_query_states = query_states[i]
                    cur_eager_attention_mask = eager_attention_mask[i]

                    if self.training:
                        answer_index = torch.where(labels[i] != -100)[0].tolist()
                        index_before_answer = []
                        for index in answer_index:
                            if labels[i][index - 1] == -100:
                                index_before_answer.append(index - 1)
                        if index_before_answer == []:
                            features_list.append(features[i])
                            attention_mask_list.append(attention_mask[i])
                            labels_list.append(labels[i])
                            continue

                        index_before_answer = torch.tensor(index_before_answer, device=labels[0].device)
                        text_query_states = cur_query_states[:, index_before_answer, :]
                        text_eager_attention_mask = cur_eager_attention_mask[:, index_before_answer, :]
                    else:
                        prompt_total_len = self.text_prompt_lens[i] + image_tokens[i]
                        text_query_states = cur_query_states[:, prompt_total_len - 1, :].unsqueeze(1)
                        text_eager_attention_mask = cur_eager_attention_mask[:, prompt_total_len - 1, :].unsqueeze(1)

                    attn_weights = torch.matmul(text_query_states, cur_key_states.transpose(1, 2)) / math.sqrt(head_dim)
                    attn_weights = attn_weights + text_eager_attention_mask
                    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

                    attention_avg_head = torch.mean(attn_weights, dim=0)
                    attention_avg_head = attention_avg_head[:, image_index : image_index + image_tokens[i]]
                    attention_avg_text = torch.mean(attention_avg_head, dim=0)

                    if llm_compress_type == "attention":
                        top_rank_index = attention_avg_text.topk(keep_length[i]).indices
                    else:
                        raise NotImplementedError(llm_compress_type)

                elif llm_compress_type == "uniform":
                    top_rank_index = torch.linspace(0, image_tokens[i] - 1, keep_length[i], dtype=torch.long)
                else:
                    raise NotImplementedError(llm_compress_type)

                top_rank_index = top_rank_index + image_index
                top_rank_index = top_rank_index.sort().values

                start_index = image_index + image_tokens[i]
                new_input_embeds = torch.cat(
                    [features[i][:image_index, :], features[i][top_rank_index, :], features[i][start_index:, :]], dim=0
                )
                new_labels = torch.cat([labels[i][:image_index], labels[i][top_rank_index], labels[i][start_index:]], dim=0)
                new_attention_mask = torch.cat(
                    [attention_mask[i][:image_index], attention_mask[i][top_rank_index], attention_mask[i][start_index:]], dim=0
                )

                features_list.append(new_input_embeds)
                attention_mask_list.append(new_attention_mask)
                labels_list.append(new_labels)

            tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
            if tokenizer_model_max_length is not None:
                new_input_embeds = [x[:tokenizer_model_max_length] for x in features_list]
                new_attention_mask = [x[:tokenizer_model_max_length] for x in attention_mask_list]
                new_labels = [x[:tokenizer_model_max_length] for x in labels_list]
            else:
                new_input_embeds = features_list
                new_attention_mask = attention_mask_list
                new_labels = labels_list

            max_len = max(x.shape[0] for x in new_input_embeds)

            embeds_padded = []
            labels_paded = []
            attention_mask_padded = []
            position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
            for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
                cur_len_emb = cur_new_embed.shape[0]
                dif = max_len - cur_len_emb

                cur_new_embed = torch.cat(
                    [cur_new_embed, torch.zeros((dif, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)],
                    dim=0,
                )
                cur_new_labels = torch.cat(
                    [cur_new_labels, torch.full((dif,), IGNORE_INDEX, dtype=cur_new_labels.dtype, device=cur_new_labels.device)],
                    dim=0,
                )
                cur_attention_mask = new_attention_mask[i]
                cur_attention_mask = torch.cat(
                    [cur_attention_mask, torch.full((dif,), False, dtype=cur_attention_mask.dtype, device=cur_attention_mask.device)],
                    dim=0,
                )

                embeds_padded.append(cur_new_embed)
                labels_paded.append(cur_new_labels)
                attention_mask_padded.append(cur_attention_mask)

                cur_len = new_attention_mask[i].sum().item()
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

            new_input_embeds = torch.stack(embeds_padded, dim=0).to(features[0].dtype)
            new_attention_mask = torch.stack(attention_mask_padded, dim=0)
            new_labels = torch.stack(labels_paded, dim=0)

            if _position_ids is None:
                position_ids = None
            if _labels is None:
                new_labels = None
            if _attention_mask is None:
                new_attention_mask = None
            else:
                new_attention_mask = new_attention_mask.to(dtype=_attention_mask.dtype)

            return position_ids, new_attention_mask, new_input_embeds, new_labels

        raise ValueError(f"Unexpected tokenizer_padding_side: {self.config.tokenizer_padding_side}")


class Qwen2ForCausalLM_Flash(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model_Flash(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs, labels = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            labels=labels,
            **kwargs,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        model_inputs["labels"] = kwargs.get("labels")
        return model_inputs


__all__ = [
    "Qwen2PreTrainedModel",
    "Qwen2Model_Flash",
    "Qwen2ForCausalLM_Flash",
    "Qwen2RMSNorm",
]
