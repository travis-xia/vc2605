from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

class Qwen2MLAConfig(Qwen2Config):

    def __init__(
        self, 
        *args, 
        kv_lora_rank=512,
        kv_lora_rank_list=None,
        q_lora_rank=None,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        qk_latent_layernorm=True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.kv_lora_rank = kv_lora_rank
        self.kv_lora_rank_list = kv_lora_rank_list
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_rope_head_dim + qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_latent_layernorm = qk_latent_layernorm