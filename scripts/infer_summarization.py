#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_summarization.py

针对 summarization 任务的批量推理脚本。
读取 eval_all_240.jsonl，对三个模型（Base / FT-A / FT-ABC）分别跑推理，
输出与原有 txt 文件完全一致的格式。

用法示例：

# Base 模型（无 adapter）
CUDA_VISIBLE_DEVICES=0 python infer_summarization.py \
    --model  mistralai/Mistral-7B-Instruct-v0.2 \
    --eval   eval_all_240.jsonl \
    --output results_base_240.txt

# FT-A 模型
CUDA_VISIBLE_DEVICES=0 python infer_summarization.py \
    --model   mistralai/Mistral-7B-Instruct-v0.2 \
    --adapter outputs/ft_A \
    --eval    eval_all_240.jsonl \
    --output  results_ftA_240.txt

# FT-ABC 模型
CUDA_VISIBLE_DEVICES=0 python infer_summarization.py \
    --model   mistralai/Mistral-7B-Instruct-v0.2 \
    --adapter outputs/ft_ABC \
    --eval    eval_all_240.jsonl \
    --output  results_ftABC_240.txt
"""

import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

SEP = "=" * 60


# ─────────────────────────────────────────
# Prompt builder  （与训练阶段保持一致）
# ─────────────────────────────────────────
def build_prompt(instruction: str, user_input: str) -> str:
    instruction = (instruction or "").strip()
    user_input  = (user_input  or "").strip()
    content = f"{instruction}\n\n{user_input}" if user_input else instruction
    return f"<s>[INST] {content} [/INST]"


# ─────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────
def load_model_and_tokenizer(model_name: str, adapter_path: Optional[str] = None):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("🔹 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path:
        print(f"🔹 Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────
# Single inference
# ─────────────────────────────────────────
@torch.no_grad()
def infer_one(
    model,
    tokenizer,
    instruction: str,
    user_input: str,
    max_new_tokens: int = 512,
) -> str:
    prompt = build_prompt(instruction, user_input)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]", 1)[-1]
    return decoded.strip()


# ─────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────
def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ─────────────────────────────────────────
# Output formatting  （与原 txt 格式完全一致）
# ─────────────────────────────────────────
def format_sample(sample_id: str, instruction: str, user_input: str, response: str) -> str:
    lines = [
        SEP,
        f"ID: {sample_id}",
        SEP,
        "",
        "INSTRUCTION:",
        instruction,
        "",
        "INPUT:",
        user_input,
        "",
        "RESPONSE:",
        response,
        "",
        "",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",          required=True,  help="Base model name or path")
    ap.add_argument("--adapter",        default=None,   help="LoRA adapter directory (omit for base model)")
    ap.add_argument("--eval",           required=True,  help="Evaluation JSONL file (e.g. eval_all_240.jsonl)")
    ap.add_argument("--output",         required=True,  help="Output txt file (e.g. results_ftA_240.txt)")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--limit",          type=int, default=0, help="Only run first N samples (0 = all, for testing)")
    args = ap.parse_args()

    # 加载数据
    print(f"📖 Loading eval data: {args.eval}")
    records = load_jsonl(args.eval)
    if args.limit > 0:
        records = records[:args.limit]
        print(f"   (limited to first {args.limit} samples)")
    print(f"   {len(records)} samples loaded")

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(args.model, args.adapter)
    tag = "BASE" if not args.adapter else f"ADAPTER({args.adapter})"
    print(f"✅ Model ready: {tag}")

    # 推理 + 写文件
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_err = 0

    with open(out_path, "w", encoding="utf-8") as f_out:
        for record in tqdm(records, desc="Inferring"):
            sample_id   = record.get("id", "UNKNOWN")
            instruction = record.get("instruction", "")
            user_input  = record.get("input", "")

            try:
                response = infer_one(
                    model, tokenizer,
                    instruction, user_input,
                    args.max_new_tokens,
                )
                n_ok += 1
            except Exception as e:
                response = f"[ERROR: {e}]"
                n_err += 1
                print(f"\n❌ Error on {sample_id}: {e}")

            f_out.write(format_sample(sample_id, instruction, user_input, response))

    print(f"\n✅ Done. {n_ok} succeeded, {n_err} errors.")
    print(f"📄 Output → {out_path}")


if __name__ == "__main__":
    main()
