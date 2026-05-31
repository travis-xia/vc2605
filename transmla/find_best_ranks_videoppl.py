import argparse
import os
from transformers import AutoTokenizer, AutoModel
import torch

from modify_config import modify_config
from utils import get_dataset, prepare_dataloader, prepare_video_dataloader, evaluate_video_ppl
from partial_rope_videoppl import partial_rope_videoppl
from lora_qkv_videoppl_find import low_rank_qkv_videoppl_find


def load_model_and_tokenizer(args):
    # 使用AutoModel而不是AutoModelForCausalLM来加载VideoChat-Flash（VideoChat-Flash官方代码中的设置）
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if args.dtype == "fp16" else torch.bfloat16 if args.dtype == "bf16" else torch.float32,
        device_map=args.device,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 设置mm_llm_compress为False（VideoChat-Flash官方代码中的设置）
    model.config.mm_llm_compress = False
    # 由于不使用llm_compress，设置为空列表或默认值
    model.model.llm_compress_type = "attention"
    model.model.llm_compress_layer_list = []
    model.model.llm_image_token_ratio_list = []
    model.model.first_image_token_position = []
    model.model.text_prompt_lens = []
    model.model.num_image_token_lens = []

    return model, tokenizer


def get_dataset_loader(tokenizer: AutoTokenizer, **kwargs):
    dataset = get_dataset(kwargs["cal_dataset"])
    train_loader = prepare_dataloader(
        dataset=dataset["train"],
        tokenizer=tokenizer,
        max_seqlen=kwargs["cal_max_seqlen"],
        batch_size=kwargs["cal_batch_size"],
        nsamples=kwargs["cal_nsamples"],
        seed=kwargs["seed"],
    )
    
    # 使用视频数据进行评估
    if kwargs.get("video_eval_path") and kwargs["ppl_eval_batch_size"] > 0:
        test_loader = prepare_video_dataloader(
            jsonl_path=kwargs["video_eval_path"],
            tokenizer=tokenizer,
            video_base_path=kwargs.get("video_base_path"),
            max_num_frames=kwargs.get("max_num_frames", 512),
            batch_size=1,  # 视频评估建议batch_size=1
            max_samples=kwargs.get("video_max_samples")
        )
    else:
        test_loader = None
    
    return train_loader, test_loader

    
def main(args):

    ##############################
    #       original model       #
    ##############################
    print("\n" + "="*60)
    print("Original Model".center(60))
    print("="*60 + "\n")

    # get model, tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    # get dataset
    train_loader, test_loader = get_dataset_loader(tokenizer, **vars(args))

    if test_loader:
        message = "Evaluating original model's video ppl"
        dataset_ppl = evaluate_video_ppl(model, tokenizer, test_loader, message)
        print(f'Original video ppl: {dataset_ppl:.4f}')

    ##############################
    #        partial rope        #
    ##############################
    print("\n" + "="*60)
    print("Partial RoPE Model".center(60))
    print("="*60 + "\n")

    if args.collapse == "auto":
        head_dim = model.config.head_dim if hasattr(model.config, "head_dim") and model.config.head_dim is not None else model.config.hidden_size // model.config.num_attention_heads
        model.config.head_dim = head_dim
        args.collapse = head_dim // args.qk_mqa_dim
        print(f"Auto collapse: {args.collapse} (head_dim={head_dim} / qk_mqa_dim={args.qk_mqa_dim})")
    else:
        args.collapse = int(args.collapse)

    model = partial_rope_videoppl(model, tokenizer, train_loader, test_loader, **vars(args))
    if args.freqfold == "auto":
        args.freqfold = model[1]
        model = model[0]

    ##############################
    #   搜索最佳 kv_lora_rank    #
    ##############################
    print("\n" + "="*60)
    print("搜索最佳 kv_lora_rank (VideoChat-Flash)".center(60))
    print("="*60 + "\n")

    model, search_results = low_rank_qkv_videoppl_find(model, tokenizer, train_loader, test_loader, **vars(args))

    # 保存搜索结果到文件
    if args.save_results:
        import json
        results_file = os.path.join(args.save_results, "kv_rank_search_results_videoppl.json")
        os.makedirs(args.save_results, exist_ok=True)
        
        # 转换结果为可序列化格式
        serializable_results = {}
        for layer_idx, layer_results in search_results.items():
            serializable_results[str(layer_idx)] = {str(rank): ppl for rank, ppl in layer_results.items()}
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\n搜索结果已保存到: {results_file}")

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="VideoChat-Flash-Qwen2_5-7B-1M_res224", help="Model to load")
    parser.add_argument("--save-results", type=str, default="search_results_videoppl", help="Directory to save search results.")
    parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", type=str, help="Device to use.", default="auto")
    parser.add_argument("--cal-dataset", type=str, help="Dataset to calibrate and calculate perplexity on.", choices=["wikitext2", "ptb", "c4", "alpaca"], default="wikitext2")
    parser.add_argument("--cal-nsamples", type=int, help="Number of samples of the calibration data to load.", default=128)
    parser.add_argument("--cal-batch-size", type=int, default=16, help="Batch size for loading the calibration data.")
    parser.add_argument("--cal-max-seqlen", type=int, default=1024, help="Maximum sequence length for the calibration data.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    parser.add_argument("--ppl-eval-batch-size", type=int, default=1, help="Batch size for evaluating the perplexity.")
    parser.add_argument("--freqfold", type=str, default="4", help="Freqfold for removing RoPE, int or auto")
    parser.add_argument("--collapse", type=str, default="auto", help="Collapse for removing RoPE, int or auto")
    parser.add_argument("--qk-mqa-dim", type=int, default=128, help="")
    parser.add_argument("--q-lora-rank", type=int, help="")
    parser.add_argument("--balance-kv-ratio", type=float, default=1, help="")
    parser.add_argument("--use-qkv-norm", action='store_true', default=False, help="")
    parser.add_argument("--ppl-tolerance", type=float, default=0.05, help="PPL tolerance for accepting lower ranks (default: 0.05)")
    
    # 视频评估相关参数
    parser.add_argument("--video-eval-path", type=str, 
                       default="/inspire/hdd/project/crosstemporalknowledge/xiacheng-240108120111/myflash0730/annotations/video/llava-video_2_3_m_academic_v0_1_cap_processed_3124_with_duration.jsonl",
                       help="Path to video evaluation JSONL file")
    parser.add_argument("--video-base-path", type=str, 
                        default="/inspire/hdd/project/crosstemporalknowledge/xiacheng-240108120111/dataset/LLaVA-Video-178K",
                       help="Base path for video files")
    parser.add_argument("--max-num-frames", type=int, default=512,
                       help="Maximum number of frames to extract from video")
    parser.add_argument("--video-max-samples", type=int, default=16,
                       help="Maximum number of video samples to evaluate")
    
    args = parser.parse_args()

    main(args)
