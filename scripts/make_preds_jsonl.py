#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch inference for evaluation.

Input jsonl format (each line):
{
  "id": "...",
  "instruction": "...",
  "input": "...",
  "output": "44.0%"
}

Output jsonl format:
{
  "id": "...",
  "gold": "44.0%",
  "pred": "44.0%"
}
"""

import argparse
import json
import torch
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


def build_prompt(instruction, user_input):
    instruction = instruction.strip()
    user_input = (user_input or "").strip()
    if user_input:
        content = f"{instruction}\n\n{user_input}"
    else:
        content = instruction
    return f"<s>[INST] {content} [/INST]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("🔹 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.adapter:
        print(f"🔹 Loading LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()

    fout = open(args.output_jsonl, "w", encoding="utf-8")

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            obj = json.loads(line)
            prompt = build_prompt(obj["instruction"], obj.get("input", ""))

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred = decoded.split("[/INST]")[-1].strip()

            out = {
                "id": obj.get("id"),
                "gold": obj.get("output"),
                "pred": pred,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    fout.close()
    print(f"\n✅ Predictions saved to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
