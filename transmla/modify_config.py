import json
import transformers.models as models
import shutil
import os




def modify_config(model, config_path: str, args):
    import json

    with open(config_path, "r") as f:
        config = json.load(f)

    if args.deepseek_style:
        settings = {
            "deepseek_v3": {
                "auto_map": {
                    "AutoConfig": "configuration_deepseek_v3.DeepseekV3Config",
                    "AutoModel": "modeling_deepseek_v3.DeepseekV3Model",
                    "AutoModelForCausalLM": "modeling_deepseek_v3.DeepseekV3ForCausalLM"
                },
                "architectures": ["DeepseekV3ForCausalLM"],
            }
        }
        for key, value in settings["deepseek_v3"].items():
            config[key] = value
    
    config["num_key_value_heads"] = config["num_attention_heads"]
    config["attention_bias"] = model.model.layers[0].self_attn.attention_bias
    config["qk_rope_head_dim"] = config["head_dim"] = args.qk_mqa_dim
    config["qk_nope_head_dim"] = config["v_head_dim"] = model.model.layers[0].self_attn.head_dim
    config["q_lora_rank"] = args.q_lora_rank
    # 写入每层的 kv_lora_rank 列表，若不存在则回退为单值
    kv_ranks = []
    for layer in model.model.layers:
        kv_ranks.append(getattr(layer.self_attn, "kv_lora_rank", None))
    if all(r is not None for r in kv_ranks) and len(kv_ranks) > 0:
        config["kv_lora_rank_list"] = kv_ranks
    else:
        # 兼容旧逻辑
        config["kv_lora_rank"] = getattr(model.model.layers[0].self_attn, "kv_lora_rank", None)

    config["qk_latent_layernorm"] = hasattr(model.model.layers[0].self_attn, "kv_a_layernorm")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # copy transformers files to the saving path
    transformers_dir = "./configuration/"
    for item in os.listdir(transformers_dir):
        source_path = os.path.join(transformers_dir, item)
        shutil.copy(source_path, args.save_path)