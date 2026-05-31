import torch
import torch.nn as nn
from typing import Optional, Tuple
import copy

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb_interleave

from utils import pca_calc, get_qkv_calibrate_outputs, evaluate_ppl, statistics_qkv_rmsnorm

# =============================
# 全局配置：每层要测试的 kv_lora_rank 列表
# 基准 rank=512 会自动添加，不需要在这里包含
# =============================
TEST_RANKS = [640, 448, 384, 256]
BASELINE_RANK = 512

 
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class LoraQKV(nn.Module):
    def __init__(
        self, 
        self_attn, 
        query_outputs, 
        key_outputs, 
        value_outputs, 
        q_lora_rank=None,
        qk_mqa_dim=64, 
        collapse=1,
        kv_lora_rank=896,
        use_qkv_norm=False, 
        balance_kv_ratio=None, 
        rms_norm_eps=1e-6,
    ):
        super().__init__()
        assert qk_mqa_dim * collapse == self_attn.head_dim

        self.config = self_attn.config
        self.dtype = self_attn.q_proj.weight.dtype
        self.layer_idx = self_attn.layer_idx
        self.num_attention_heads = self_attn.num_attention_heads
        self.head_dim = self_attn.head_dim
        self.qk_mqa_dim = qk_mqa_dim
        self.collapse = collapse
        self.latent_dim = self_attn.latent_dim
        self.attention_dropout = self_attn.attention_dropout
        self.hidden_size = self_attn.hidden_size
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        assert self.kv_lora_rank <= 2 * self.latent_dim - self.qk_mqa_dim, f"kv_lora_rank ({self.kv_lora_rank}) must be less than 2 * latent_dim ({self.latent_dim}) - qk_mqa_dim ({self.qk_mqa_dim})"

        self.attention_function = ALL_ATTENTION_FUNCTIONS["sdpa"]
        self.scaling = (self.head_dim + self.qk_mqa_dim)**(-0.5)

        # -----------------Attributes for the bias-----------------
        q_bias = self_attn.q_proj.bias is not None
        k_bias = self_attn.k_proj.bias is not None
        v_bias = self_attn.v_proj.bias is not None
        assert q_bias == k_bias == v_bias, f"q_bias ({q_bias}), k_bias ({k_bias}), v_bias ({v_bias}) must be the same"
        self.attention_bias = q_bias

        # -----------------module definitions-----------------
        # q_a_proj & q_b_proj
        if q_lora_rank is not None:
            self.q_a_proj = nn.Linear(
                self.hidden_size, 
                q_lora_rank, 
                bias=False,
                device=self_attn.q_proj.weight.device,
                dtype=self.dtype,
            )
            if use_qkv_norm:
                self.q_a_layernorm = nn.RMSNorm(q_lora_rank, device=self_attn.q_proj.weight.device, dtype=self.dtype, eps=rms_norm_eps)
            self.q_b_proj = nn.Linear(
                q_lora_rank,
                self.num_attention_heads * (self.qk_mqa_dim + self.head_dim), 
                bias=self.attention_bias,
                device=self_attn.q_proj.weight.device,
                dtype=self.dtype,
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size, 
                self.num_attention_heads * (self.qk_mqa_dim + self.head_dim), 
                bias=self.attention_bias,
                device=self_attn.q_proj.weight.device,
                dtype=self.dtype,
            )
        # kv_a_proj & kv_b_proj
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            kv_lora_rank + qk_mqa_dim,
            bias=self.attention_bias,
            device=self_attn.k_proj.weight.device,
            dtype=self.dtype,
        )
        if use_qkv_norm:
            self.kv_a_layernorm = nn.RMSNorm(kv_lora_rank, device=self_attn.k_proj.weight.device, dtype=self.dtype, eps=rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            self.num_attention_heads * self.head_dim * 2,
            bias=False,
            device=self_attn.k_proj.weight.device,
            dtype=self.dtype,
        )
        # nothing else to do for o_proj
        self.o_proj = self_attn.o_proj

        # -----------------apply bkv on the key and value outputs-----------------
        if balance_kv_ratio is not None:
            k_outputs_norm = torch.cat([key.reshape(-1, self.latent_dim)[:,self.qk_mqa_dim:] for key in key_outputs]).norm(p=2,dim=0).mean()
            v_outputs_norm = torch.cat([value.reshape(-1, self.latent_dim)[:,self.qk_mqa_dim:] for value in value_outputs]).norm(p=2,dim=0).mean()
            ratio = k_outputs_norm / (v_outputs_norm * balance_kv_ratio)
            self_attn.k_proj.weight.data[self.qk_mqa_dim:] /= ratio
            if self.attention_bias:
                self_attn.k_proj.bias.data[self.qk_mqa_dim:] /= ratio
            self_attn.k_up_proj.weight.data[:, self.qk_mqa_dim:] *= ratio
        else:
            ratio = 1
        kv_outputs = [torch.cat([key_outputs[i][:,:,qk_mqa_dim:] / ratio, value_outputs[i]], dim=-1) for i in range(len(key_outputs))]

        # -----------------apply pca on the query and key/value outputs-----------------
        if self.q_lora_rank is not None:
            R_q = pca_calc(query_outputs, self_attn.q_proj.weight.device)
        else:
            R_q = None
        R_kv = pca_calc(kv_outputs, self_attn.k_proj.weight.device)

        # -----------------initialize the weights / bias-----------------
        self._init_weights(self_attn, R_q, R_kv)
        
    def _init_weights(self, self_attn, R_q, R_kv):
        # 0. Split the weights of k_proj and v_proj into rope / nope parts.
        k_a_rope_weight, k_a_nope_weight = self_attn.k_proj.weight.data.split([self.qk_mqa_dim, self.latent_dim - self.qk_mqa_dim],dim=0)
        k_b_rope_weight, k_b_nope_weight = self_attn.k_up_proj.weight.data.split([self.qk_mqa_dim, self.latent_dim - self.qk_mqa_dim], dim=1)
        k_b_rope_weight = k_b_rope_weight.view(self.num_attention_heads, self.head_dim, self.qk_mqa_dim)
        k_b_nope_weight = k_b_nope_weight.view(self.num_attention_heads, self.head_dim, self.latent_dim-self.qk_mqa_dim)
        
        v_a_nope_weight  = self_attn.v_proj.weight.data
        v_b_nope_weight = self_attn.v_up_proj.weight.data
        v_b_nope_weight = v_b_nope_weight.view(self.num_attention_heads, self.head_dim, self.latent_dim)

        if self.attention_bias:
            q_bias = self_attn.q_proj.bias.data
            v_bias = self_attn.v_proj.bias.data
            k_bias_rope, k_bias_nope = self_attn.k_proj.bias.data.split([self.qk_mqa_dim, self.latent_dim - self.qk_mqa_dim], dim=0)


        # 1. Initialize q_a_proj / q_b_proj if q_lora_rank is not None (revised by xiaojuan based on bias...)
        # 1.1 Initialize q_a_proj
        original_scaling = getattr(self.config, "query_pre_attn_scalar", self.head_dim)**-0.5
        scaling = original_scaling / self.scaling
        if self.q_lora_rank is not None:
            q_weight = self_attn.q_proj.weight.data.to(torch.float64)

            q_a_weight = (R_q.T @ q_weight)[: self.q_lora_rank].to(self.dtype)
            self.q_a_proj.weight.data = q_a_weight.contiguous()
            
            q_b_weight = R_q[:, :self.q_lora_rank].to(self.dtype)
            q_b_weight = q_b_weight.view(self.num_attention_heads, self.head_dim, self.q_lora_rank)
            # Absorb the rope part of k_b_proj into q_b_proj
            q_b_rope_weight = torch.einsum("hdq,hdk->hkq", q_b_weight, k_b_rope_weight)
            q_b_with_mqa_weight = torch.cat([q_b_weight, q_b_rope_weight], dim=1).reshape(
                self.num_attention_heads * (self.head_dim + self.qk_mqa_dim), self.q_lora_rank
            )

            # Scale the weight before initializing the q_b_projAdd commentMore actions
            # In the original GQA, attention scores are divided by sqrt(head_dim).
            # However, in the transformed MLA, the attention scores are divided by sqrt(head_dim + qk_mqa_dim).
            self.q_b_proj.weight.data = q_b_with_mqa_weight.contiguous() * scaling

        else:
            q_weight = self_attn.q_proj.weight.data.view(self.num_attention_heads, self.head_dim, self.hidden_size)
            q_rope_weight = torch.einsum("hdD,hdk->hkD", q_weight, k_b_rope_weight) 
            q_with_mqa_weight = torch.cat([q_weight, q_rope_weight], dim=1).reshape(
                self.num_attention_heads * (self.head_dim + self.qk_mqa_dim), self.hidden_size
            )
            
            self.q_proj.weight.data = q_with_mqa_weight.contiguous() * scaling

        if self.attention_bias:
            q_bias = q_bias.reshape(self.num_attention_heads, self.head_dim)
            q_rope_bias = torch.einsum("hd,hdk->hk", q_bias.to(torch.float64), k_b_rope_weight.to(torch.float64)).to(self.dtype)
            q_bias = torch.cat([q_bias, q_rope_bias], dim=1).flatten().contiguous() * scaling
            if self.q_lora_rank is not None:
                self.q_b_proj.bias.data = q_bias
            else:
                self.q_proj.bias.data = q_bias
            
        
        # 2. Low-rank decomposing k_proj and v_proj
        # 2.1 Concatenate the nope parts of k_proj and v_proj
        kv_a_nope_weight = torch.cat([k_a_nope_weight, v_a_nope_weight], dim=0).to(torch.float64)
        if self.attention_bias:
            kv_a_nope_bias = torch.cat([k_bias_nope, v_bias]).unsqueeze(-1).to(torch.float64)
            kv_a_nope_weight = torch.cat([kv_a_nope_weight, kv_a_nope_bias], dim=-1)
        kv_b_nope_weight = torch.cat(
            [
                torch.cat([k_b_nope_weight, torch.zeros_like(v_b_nope_weight)], dim=-1),
                torch.cat([torch.zeros_like(k_b_nope_weight), v_b_nope_weight], dim=-1)
            ], 
            dim=1
        ).reshape(2 * self.num_attention_heads * self.head_dim, 2 * self.latent_dim - self.qk_mqa_dim).to(torch.float64)
        

        # 2.2 Low-rank decomposing kv_a_nope_weight and kv_b_nope_weight
        kv_a_nope_weight = (R_kv.T @ kv_a_nope_weight)[: self.kv_lora_rank].to(self.dtype)
        if self.attention_bias:
            kv_a_nope_weight, kv_a_nope_bias = torch.split(kv_a_nope_weight, [self.hidden_size, 1], dim=-1)
            kv_a_nope_bias = kv_a_nope_bias.flatten().to(self.dtype)
        kv_b_nope_weight = (kv_b_nope_weight @ R_kv)[:, :self.kv_lora_rank].to(self.dtype)
        self.kv_b_proj.weight.data = kv_b_nope_weight.contiguous()
        kv_a_proj_with_mqa_weight = torch.cat([kv_a_nope_weight, k_a_rope_weight], dim=0)
        self.kv_a_proj_with_mqa.weight.data = kv_a_proj_with_mqa_weight.contiguous()
        if self.attention_bias:
            kv_a_proj_with_mqa_bias = torch.cat([kv_a_nope_bias, k_bias_rope])
            self.kv_a_proj_with_mqa.bias.data = kv_a_proj_with_mqa_bias.contiguous()


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # query
        if self.q_lora_rank is not None:
            query_states = self.q_a_proj(hidden_states)
            if hasattr(self, "q_a_layernorm"):
                query_states = self.q_a_layernorm(query_states)
            query_states = self.q_b_proj(query_states)
        else:
            query_states = self.q_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, -1).transpose(1,2)
        q_nope, q_rope = query_states.split([self.head_dim, self.qk_mqa_dim], dim=-1)

        # key and value
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_nope, k_rope = compressed_kv.split([self.kv_lora_rank, self.qk_mqa_dim], dim=-1)
        kv_nope = kv_nope.view(bsz, 1, q_len, self.kv_lora_rank)
        k_rope = k_rope.view(bsz, 1, q_len, self.qk_mqa_dim)

        cos, sin = position_embeddings
        q_rope, k_rope = apply_rotary_pos_emb_interleave(q_rope, k_rope, cos[ :, :, : : self.collapse], sin[ :, :, : : self.collapse])
        query_states = torch.cat([q_nope, q_rope], dim=-1)

        if hasattr(self, "kv_a_layernorm"):
            kv_nope = self.kv_a_layernorm(kv_nope)
        kv_nope = self.kv_b_proj(kv_nope).view(bsz, q_len, self.num_attention_heads, self.head_dim * 2).transpose(1, 2)
        k_nope, value_states = kv_nope.split([self.head_dim, self.head_dim],dim=-1)
        key_states = torch.cat([k_nope, repeat_kv(k_rope, self.num_attention_heads)], dim=-1)

        attn_output, attn_weights = self.attention_function(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            softcap=getattr(self.config, "attn_logit_softcapping", None)
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


def low_rank_qkv_find(model, tokenizer, train_loader, test_loader, ppl_tolerance=0.1, **kwargs):
    """
    遍历每一层的不同 kv_lora_rank 设置，测试对 ppl 的影响
    测试的 rank 值：使用全局配置 TEST_RANKS + BASELINE_RANK
    优化：基准只测试一次，然后每层只测试 TEST_RANKS 中的值
    目标：找到在可接受 PPL 损失范围内（默认 0.1）的最小 rank
    """
    
    print("\n" + "="*80)
    print("开始搜索每层 kv_lora_rank 对 PPL 的影响".center(80))
    print(f"PPL 容忍度: {ppl_tolerance:.2f}".center(80))
    print(f"测试 rank 值: {TEST_RANKS} + 基准 {BASELINE_RANK}".center(80))
    print("="*80 + "\n")
    
    # 获取 qkv 输出用于校准
    message = "Calibrating rope-removed model's qkv outputs"
    rm_rope_qkv_outputs = get_qkv_calibrate_outputs(model, train_loader, message)
    
    num_layers = len(model.model.layers)
    test_ranks = TEST_RANKS.copy()  # 每层需要测试的 rank 值（不包括基准）
    
    # 存储结果
    results = {}
    
    # 保存原始模型状态，用于每次测试后恢复
    original_model = copy.deepcopy(model)
    
    # 首先测试基准配置（所有层都使用 BASELINE_RANK）
    print(f"\n{'='*60}")
    print(f"测试基准配置（所有层 kv_lora_rank={BASELINE_RANK}）".center(60))
    print(f"{'='*60}")
    
    # 恢复到原始模型状态
    model = copy.deepcopy(original_model)
    
    # 构建基准的 kv_lora_rank_list（所有层都是 BASELINE_RANK）
    baseline_kv_lora_rank_list = [BASELINE_RANK] * num_layers
    
    # 应用 LoraQKV 到所有层
    for idx, layer in enumerate(model.model.layers):
        setattr(layer, "self_attn", LoraQKV(
            layer.self_attn,
            rm_rope_qkv_outputs["query"][idx], 
            rm_rope_qkv_outputs["key"][idx], 
            rm_rope_qkv_outputs["value"][idx], 
            q_lora_rank=kwargs["q_lora_rank"], 
            qk_mqa_dim=kwargs["qk_mqa_dim"], 
            collapse=kwargs["collapse"],
            kv_lora_rank=BASELINE_RANK,
            use_qkv_norm=kwargs["use_qkv_norm"],
            balance_kv_ratio=kwargs["balance_kv_ratio"],
            rms_norm_eps=model.config.rms_norm_eps,
        ))
    
    # 如果使用 qkv norm，需要重新校准
    if kwargs["use_qkv_norm"]:
        lora_qkv_outputs = get_qkv_calibrate_outputs(model, train_loader)
        for idx, layer in enumerate(model.model.layers):
            statistics_qkv_rmsnorm(
                layer.self_attn, 
                lora_qkv_outputs["q_a_proj"][idx] if len(lora_qkv_outputs["q_a_proj"]) > idx else None, 
                lora_qkv_outputs["kv_a_proj"][idx]
            )
    
    # 评估基准 ppl
    baseline_ppl = None
    if test_loader:
        message = f"Evaluating baseline configuration (all layers kv_lora_rank={BASELINE_RANK})"
        baseline_ppl = evaluate_ppl(model, tokenizer.pad_token_id, test_loader, message)
        print(f"基准配置 PPL: {baseline_ppl:.4f}")
    else:
        print("基准配置: 无测试数据")
    
    # 为每层初始化结果，都包含基准值
    for layer_idx in range(num_layers):
        results[layer_idx] = {BASELINE_RANK: baseline_ppl}
    
    # 然后测试每一层的不同 rank
    for layer_idx in range(num_layers):
        print(f"\n{'='*60}")
        print(f"测试第 {layer_idx} 层的不同 kv_lora_rank".center(60))
        print(f"{'='*60}")
        
        for rank in test_ranks:
            print(f"\n  测试 Layer {layer_idx} 使用 kv_lora_rank = {rank}")
            
            # 恢复到原始模型状态
            model = copy.deepcopy(original_model)
            
            # 构建当前测试的 kv_lora_rank_list
            # 除了当前层使用测试 rank，其他层都使用基准 BASELINE_RANK
            kv_lora_rank_list = [BASELINE_RANK] * num_layers
            kv_lora_rank_list[layer_idx] = rank
            
            # 应用 LoraQKV 到所有层
            for idx, layer in enumerate(model.model.layers):
                layer_kv_rank = kv_lora_rank_list[idx]
                
                setattr(layer, "self_attn", LoraQKV(
                    layer.self_attn,
                    rm_rope_qkv_outputs["query"][idx], 
                    rm_rope_qkv_outputs["key"][idx], 
                    rm_rope_qkv_outputs["value"][idx], 
                    q_lora_rank=kwargs["q_lora_rank"], 
                    qk_mqa_dim=kwargs["qk_mqa_dim"], 
                    collapse=kwargs["collapse"],
                    kv_lora_rank=layer_kv_rank,
                    use_qkv_norm=kwargs["use_qkv_norm"],
                    balance_kv_ratio=kwargs["balance_kv_ratio"],
                    rms_norm_eps=model.config.rms_norm_eps,
                ))
            
            # 如果使用 qkv norm，需要重新校准
            if kwargs["use_qkv_norm"]:
                lora_qkv_outputs = get_qkv_calibrate_outputs(model, train_loader)
                for idx, layer in enumerate(model.model.layers):
                    statistics_qkv_rmsnorm(
                        layer.self_attn, 
                        lora_qkv_outputs["q_a_proj"][idx] if len(lora_qkv_outputs["q_a_proj"]) > idx else None, 
                        lora_qkv_outputs["kv_a_proj"][idx]
                    )
            
            # 评估 ppl
            if test_loader:
                message = f"Evaluating Layer {layer_idx} with kv_lora_rank={rank}"
                dataset_ppl = evaluate_ppl(model, tokenizer.pad_token_id, test_loader, message)
                results[layer_idx][rank] = dataset_ppl
                print(f"    Layer {layer_idx}, kv_lora_rank={rank}: PPL = {dataset_ppl:.4f}")
            else:
                results[layer_idx][rank] = None
                print(f"    Layer {layer_idx}, kv_lora_rank={rank}: 无测试数据")
    
    # 打印汇总结果
    print("\n" + "="*80)
    print("搜索结果汇总".center(80))
    print("="*80)
    baseline_ppl_str = f"{baseline_ppl:.4f}" if baseline_ppl is not None else "N/A"
    print(f"基准配置 PPL (所有层 rank={BASELINE_RANK}): {baseline_ppl_str}")
    print(f"PPL 容忍度: +{ppl_tolerance:.2f}")
    print("-" * 80)
    
    # 动态生成表头
    header_parts = ['Layer']
    for rank in sorted(TEST_RANKS, reverse=True):  # 从大到小排列
        header_parts.append(f'Rank={rank}')
    header_parts.extend(['推荐Rank', 'PPL损失'])
    
    header_line = " ".join(f"{part:<12}" for part in header_parts)
    print(header_line)
    print("-" * len(header_line))
    
    recommended_ranks = []
    
    for layer_idx in range(num_layers):
        layer_results = results[layer_idx]
        if baseline_ppl is not None and all(ppl is not None for rank, ppl in layer_results.items() if rank != BASELINE_RANK):
            # 找到在容忍度内的最小 rank
            all_ranks = sorted(TEST_RANKS + [BASELINE_RANK])  # 从小到大检查
            acceptable_ranks = []
            for rank in all_ranks:
                if rank in layer_results:
                    ppl_diff = layer_results[rank] - baseline_ppl
                    if ppl_diff <= ppl_tolerance:
                        acceptable_ranks.append((rank, layer_results[rank], ppl_diff))
            
            if acceptable_ranks:
                # 选择最小的可接受 rank
                recommended_rank, recommended_ppl, ppl_diff = acceptable_ranks[0]
                recommended_ranks.append((layer_idx, recommended_rank))
                
                if ppl_diff <= 0:
                    ppl_loss_str = f"{ppl_diff:.4f}"
                else:
                    ppl_loss_str = f"+{ppl_diff:.4f}"
            else:
                # 如果都不在容忍度内，选择基准
                recommended_rank, recommended_ppl = BASELINE_RANK, baseline_ppl
                ppl_loss_str = "基准"
                recommended_ranks.append((layer_idx, BASELINE_RANK))
            
            # 动态生成行数据
            row_parts = [f"{layer_idx}"]
            for rank in sorted(TEST_RANKS, reverse=True):
                ppl_val = layer_results.get(rank)
                if ppl_val is not None:
                    row_parts.append(f"{ppl_val:.4f}")
                else:
                    row_parts.append("N/A")
            
            recommended_str = f"{recommended_rank}({recommended_ppl:.4f})"
            row_parts.extend([recommended_str, ppl_loss_str])
            
            row_line = " ".join(f"{part:<12}" for part in row_parts)
            print(row_line)
        else:
            # 生成 N/A 行
            row_parts = [f"{layer_idx}"]
            for rank in sorted(TEST_RANKS, reverse=True):
                row_parts.append("N/A")
            row_parts.extend(["N/A", "N/A"])
            
            row_line = " ".join(f"{part:<12}" for part in row_parts)
            print(row_line)
            recommended_ranks.append((layer_idx, BASELINE_RANK))  # 默认使用基准
    
    print("="*80)
    
    # 统计推荐配置
    if baseline_ppl is not None:
        # 动态统计所有可能的 rank
        all_possible_ranks = sorted(TEST_RANKS + [BASELINE_RANK])
        rank_counts = {rank: 0 for rank in all_possible_ranks}
        total_savings = 0
        
        for layer_idx, recommended_rank in recommended_ranks:
            if recommended_rank in rank_counts:
                rank_counts[recommended_rank] += 1
                if recommended_rank < BASELINE_RANK:
                    total_savings += (BASELINE_RANK - recommended_rank)
        
        print(f"\n推荐配置总结:")
        print(f"- 基准配置 PPL: {baseline_ppl:.4f}")
        print(f"- PPL 容忍度: +{ppl_tolerance:.2f}")
        print(f"- 推荐 rank 分布:")
        for rank in sorted(all_possible_ranks, reverse=True):
            print(f"  * Rank {rank}: {rank_counts[rank]} 层")
        print(f"- 总参数节省: {total_savings} × kv_lora_rank 维度")
        
        # 生成推荐的 KV_LORA_RANKS 列表
        recommended_list = [rank for _, rank in sorted(recommended_ranks)]
        print(f"\n推荐的 KV_LORA_RANKS 配置:")
        print(f"KV_LORA_RANKS = {recommended_list}")
    
    print("="*80)
    
    # 恢复到原始模型
    model = original_model
    return model, results


def low_rank_qkv(model, tokenizer, train_loader, test_loader, **kwargs):
    """
    原始的 low_rank_qkv 函数，保持兼容性
    """
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
            # fallback to a reasonable default if neither is provided
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
        dataset_ppl = evaluate_ppl(model, tokenizer.pad_token_id, test_loader, message)
        print(f'Low rank approximate QKV ppl: {dataset_ppl:.4f}')
    
    return model
