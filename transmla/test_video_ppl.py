#!/usr/bin/env python3
"""
测试视频PPL评估功能的简单脚本
"""

import json
import os
import sys

def create_dummy_video_jsonl(output_path, num_samples=3):
    """创建虚拟的视频数据JSONL文件用于测试"""
    dummy_data = []
    
    for i in range(num_samples):
        item = {
            "id": f"test_video_{i}",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nDescribe what you see in this video."
                },
                {
                    "from": "gpt", 
                    "value": f"This is a test video description for sample {i}. The video shows various activities and scenes that demonstrate the model's understanding capabilities."
                }
            ],
            "video": f"test_video_{i}.mp4",
            "duration": 30.0 + i * 10
        }
        dummy_data.append(item)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dummy_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Created dummy video JSONL with {num_samples} samples at: {output_path}")

def test_video_dataset_loading():
    """测试视频数据集加载功能"""
    try:
        from utils import load_video_dataset
        
        # 创建测试数据
        test_jsonl = "test_video_data.jsonl"
        create_dummy_video_jsonl(test_jsonl, 3)
        
        # 测试加载
        data = load_video_dataset(test_jsonl)
        print(f"Successfully loaded {len(data)} video samples")
        
        # 打印第一个样本
        if data:
            print("First sample:")
            print(f"  ID: {data[0]['id']}")
            print(f"  Video path: {data[0]['video_path']}")
            print(f"  Conversations: {len(data[0]['conversations'])} turns")
            print(f"  Duration: {data[0]['duration']}")
        
        # 清理测试文件
        os.remove(test_jsonl)
        print("✓ Video dataset loading test passed")
        return True
        
    except Exception as e:
        print(f"✗ Video dataset loading test failed: {e}")
        return False

def test_imports():
    """测试必要的导入"""
    try:
        from utils import load_video_dataset, prepare_video_dataloader, evaluate_video_ppl
        print("✓ Video processing functions imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def main():
    print("Testing Video PPL Evaluation Setup")
    print("=" * 50)
    
    # 测试导入
    if not test_imports():
        print("Import test failed, stopping...")
        return
    
    # 测试数据加载
    if not test_video_dataset_loading():
        print("Dataset loading test failed, stopping...")
        return
    
    print("\n" + "=" * 50)
    print("All basic tests passed! ✓")
    print("\nTo run the full video PPL evaluation, use:")
    print("python converter_mllm_videoppl.py --model-path VideoChat-Flash-Qwen2_5-7B-1M_res224")

if __name__ == "__main__":
    main()
