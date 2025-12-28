#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference with Mistral-7B-Instruct + LoRA adapter (QLoRA).

Example:
CUDA_VISIBLE_DEVICES=0 python infer_with_adapter.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --adapter outputs/ft_C_only \
  --instruction "Compute the gross margin."
  --input "Income Statement (USD, million)\nRevenue: 1200\nCOGS: 720"
"""

import argparse
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter (optional)")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--input", default="")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    # 4-bit 量化配置（必须和训练一致）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("🔹 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 挂载 LoRA adapter（如果提供）
    if args.adapter:
        print(f"🔹 Loading LoRA adapter from {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()

    prompt = build_prompt(args.instruction, args.input)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n===== PROMPT =====")
    print(prompt)
    print("\n===== MODEL OUTPUT =====")
    print(decoded.split("[/INST]")[-1].strip())


if __name__ == "__main__":
    main()
