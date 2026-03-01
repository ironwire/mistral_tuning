#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Inference and Evaluation with Mistral-7B-Instruct + LoRA adapter (QLoRA).

Example - Single sample inference by ID:
CUDA_VISIBLE_DEVICES=0 python infer_with_adapter.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --adapter outputs/ft_finance \
  --eval_file finance_eval_100.jsonl \
  --sample_id A_LANG_000013

Example - Batch evaluation:
CUDA_VISIBLE_DEVICES=0 python infer_with_adapter.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --adapter outputs/ft_finance \
  --eval_file finance_eval_100.jsonl \
  --output_file results.txt
"""
import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
from typing import Optional, List, Dict


def build_prompt(instruction: str, user_input: str) -> str:
    """
    和训练阶段保持一致的 prompt 模板
    """
    instruction = instruction.strip()
    user_input = user_input.strip()
    
    if user_input:
        content = f"{instruction}\n\n{user_input}"
    else:
        content = instruction
    
    return f"<s>[INST] {content} [/INST]"


def load_model_and_tokenizer(model_name: str, adapter_path: Optional[str] = None):
    """
    加载模型和tokenizer
    """
    # 4-bit 量化配置（必须和训练一致）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    print("🔹 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 挂载 LoRA adapter（如果提供）
    if adapter_path:
        print(f"🔹 Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.eval()
    return model, tokenizer


def infer_single(
    model, 
    tokenizer, 
    instruction: str, 
    user_input: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """
    单条推理
    """
    prompt = build_prompt(instruction, user_input)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0.0),
            top_p=0.95 if temperature > 0.0 else 1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取 [/INST] 之后的内容
    if "[/INST]" in decoded:
        response = decoded.split("[/INST]")[-1].strip()
    else:
        response = decoded.strip()
    
    return response


def load_eval_data(file_path: str) -> List[Dict]:
    """
    加载评测数据（JSONL格式）
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def find_sample_by_id(eval_data: List[Dict], sample_id: str) -> Optional[Dict]:
    """
    根据ID查找样本
    """
    for sample in eval_data:
        if sample.get('id') == sample_id:
            return sample
    return None


def format_result(result: Dict, include_expected: bool = True) -> str:
    """
    格式化单个结果为易读文本
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"Sample ID: {result['id']}")
    lines.append(f"Subtask: {result['subtask']}")
    lines.append("-" * 80)
    lines.append("Instruction:")
    lines.append(result['instruction'])
    lines.append("")
    
    if result['input']:
        lines.append("Input:")
        lines.append(result['input'])
        lines.append("")
    
    if include_expected and result.get('expected'):
        lines.append("Expected Output:")
        lines.append(result['expected'])
        lines.append("")
    
    lines.append("Model Response:")
    lines.append(result['prediction'])
    lines.append("=" * 80)
    lines.append("")
    
    return "\n".join(lines)


def save_results_text(results: List[Dict], output_file: str, include_expected: bool = True):
    """
    保存评测结果到文本文件
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入标题
        f.write("=" * 80 + "\n")
        f.write(f"EVALUATION RESULTS - Total Samples: {len(results)}\n")
        f.write("=" * 80 + "\n\n")
        
        # 写入每个样本的结果
        for result in results:
            f.write(format_result(result, include_expected))
    
    print(f"\n✅ Results saved to {output_file}")


def save_results_jsonl(results: List[Dict], output_file: str):
    """
    保存评测结果到JSONL文件（备用格式）
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✅ Results (JSONL) saved to {output_file}")


def calculate_metrics(results: List[Dict]) -> Dict:
    """
    计算基本的评测指标
    """
    total = len(results)
    
    # 简单的匹配度计算
    exact_matches = 0
    for item in results:
        if item['prediction'].strip() == item['expected'].strip():
            exact_matches += 1
    
    metrics = {
        'total_samples': total,
        'exact_match': exact_matches,
        'exact_match_rate': exact_matches / total if total > 0 else 0,
    }
    
    # 按subtask统计
    subtask_stats = {}
    for item in results:
        subtask = item.get('subtask', 'unknown')
        if subtask not in subtask_stats:
            subtask_stats[subtask] = {'total': 0, 'exact_match': 0}
        
        subtask_stats[subtask]['total'] += 1
        if item['prediction'].strip() == item['expected'].strip():
            subtask_stats[subtask]['exact_match'] += 1
    
    # 计算每个subtask的准确率
    for subtask, stats in subtask_stats.items():
        stats['exact_match_rate'] = stats['exact_match'] / stats['total']
    
    metrics['subtask_performance'] = subtask_stats
    
    return metrics


def save_metrics_text(metrics: Dict, output_file: str):
    """
    保存指标到文本文件
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION METRICS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Samples: {metrics['total_samples']}\n")
        f.write(f"Exact Match: {metrics['exact_match']}\n")
        f.write(f"Exact Match Rate: {metrics['exact_match_rate']:.2%}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("Per-Subtask Performance:\n")
        f.write("-" * 80 + "\n")
        
        for subtask, stats in metrics['subtask_performance'].items():
            f.write(f"\n{subtask}:\n")
            f.write(f"  Total: {stats['total']}\n")
            f.write(f"  Exact Match: {stats['exact_match']}\n")
            f.write(f"  Exact Match Rate: {stats['exact_match_rate']:.2%}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"✅ Metrics saved to {output_file}")


def batch_evaluate(
    model,
    tokenizer,
    eval_data: List[Dict],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> List[Dict]:
    """
    批量评测
    """
    results = []
    
    for item in tqdm(eval_data, desc="Evaluating"):
        try:
            instruction = item['instruction']
            user_input = item.get('input', '')
            expected_output = item.get('output', '')
            
            prediction = infer_single(
                model, 
                tokenizer, 
                instruction, 
                user_input,
                max_new_tokens,
                temperature,
            )
            
            result = {
                'id': item.get('id', ''),
                'subtask': item.get('subtask', ''),
                'instruction': instruction,
                'input': user_input,
                'expected': expected_output,
                'prediction': prediction,
            }
            results.append(result)
            
        except Exception as e:
            print(f"\n❌ Error processing item {item.get('id', 'unknown')}: {e}")
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--model", required=True, help="Base model name or path")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter (optional)")
    
    # Data arguments
    parser.add_argument("--eval_file", required=True, help="Path to evaluation JSONL file")
    
    # Single sample inference arguments
    parser.add_argument("--sample_id", default=None, help="Sample ID for single inference (e.g., A_LANG_000013)")
    parser.add_argument("--sample_index", type=int, default=None, help="Sample index for single inference (0-based)")
    
    # Batch evaluation arguments
    parser.add_argument("--output_file", default="results.txt", help="Output file for batch results")
    parser.add_argument("--no_expected", action="store_true", help="Don't include expected output in results file")
    parser.add_argument("--save_jsonl", action="store_true", help="Also save results in JSONL format")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    
    args = parser.parse_args()
    
    # 加载评测数据
    print(f"🔹 Loading evaluation data from {args.eval_file}")
    eval_data = load_eval_data(args.eval_file)
    print(f"Loaded {len(eval_data)} samples")
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(args.model, args.adapter)
    
    # 单样本推理模式（通过ID）
    if args.sample_id:
        print("\n📝 Single Sample Inference Mode (by ID)")
        print("=" * 80)
        
        sample = find_sample_by_id(eval_data, args.sample_id)
        if sample is None:
            print(f"❌ Sample with ID '{args.sample_id}' not found!")
            return
        
        instruction = sample['instruction']
        user_input = sample.get('input', '')
        expected_output = sample.get('output', '')
        
        response = infer_single(
            model,
            tokenizer,
            instruction,
            user_input,
            args.max_new_tokens,
            args.temperature,
        )
        
        # 打印结果
        result = {
            'id': sample.get('id'),
            'subtask': sample.get('subtask'),
            'instruction': instruction,
            'input': user_input,
            'expected': expected_output,
            'prediction': response,
        }
        
        print("\n" + format_result(result, include_expected=True))
    
    # 单样本推理模式（通过索引）
    elif args.sample_index is not None:
        print("\n📝 Single Sample Inference Mode (by Index)")
        print("=" * 80)
        
        if args.sample_index < 0 or args.sample_index >= len(eval_data):
            print(f"❌ Invalid sample index {args.sample_index}! Valid range: 0-{len(eval_data)-1}")
            return
        
        sample = eval_data[args.sample_index]
        instruction = sample['instruction']
        user_input = sample.get('input', '')
        expected_output = sample.get('output', '')
        
        response = infer_single(
            model,
            tokenizer,
            instruction,
            user_input,
            args.max_new_tokens,
            args.temperature,
        )
        
        # 打印结果
        result = {
            'id': sample.get('id'),
            'subtask': sample.get('subtask'),
            'instruction': instruction,
            'input': user_input,
            'expected': expected_output,
            'prediction': response,
        }
        
        print("\n" + format_result(result, include_expected=True))
    
    # 批量评测模式
    else:
        print("\n📊 Batch Evaluation Mode")
        print("=" * 80)
        
        # 执行批量评测
        results = batch_evaluate(
            model,
            tokenizer,
            eval_data,
            args.max_new_tokens,
            args.temperature,
        )
        
        # 保存文本格式结果
        save_results_text(results, args.output_file, include_expected=not args.no_expected)
        
        # 可选：保存JSONL格式
        if args.save_jsonl:
            jsonl_file = args.output_file.replace('.txt', '.jsonl')
            save_results_jsonl(results, jsonl_file)
        
        # 计算并显示指标
        metrics = calculate_metrics(results)
        print("\n📈 Evaluation Metrics:")
        print("=" * 80)
        print(f"Total Samples: {metrics['total_samples']}")
        print(f"Exact Match: {metrics['exact_match']}")
        print(f"Exact Match Rate: {metrics['exact_match_rate']:.2%}")
        print("\nPer-Subtask Performance:")
        for subtask, stats in metrics['subtask_performance'].items():
            print(f"  {subtask:20s}: {stats['exact_match']}/{stats['total']} ({stats['exact_match_rate']:.2%})")
        print("=" * 80)
        
        # 保存指标到文本文件
        metrics_file = args.output_file.replace('.txt', '_metrics.txt')
        save_metrics_text(metrics, metrics_file)
        
        # 也保存JSON格式的指标（方便程序读取）
        metrics_json = args.output_file.replace('.txt', '_metrics.json')
        with open(metrics_json, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"✅ Metrics (JSON) saved to {metrics_json}")


if __name__ == "__main__":
    main()
    