#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import torch
import argparse
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


# -------------------------------------------------
# Prompt construction (Mistral-style)
# -------------------------------------------------
def build_prompt(instruction: str, user_input: str) -> str:
    instruction = instruction.strip()
    user_input = user_input.strip()
    if user_input:
        content = f"{instruction}\n\n{user_input}"
    else:
        content = instruction
    return f"<s>[INST] {content} [/INST]"


# -------------------------------------------------
# Load evaluation samples
# -------------------------------------------------
def load_samples(filepath):
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


# -------------------------------------------------
# Inference loop
# -------------------------------------------------
def run_inference(
    model,
    tokenizer,
    samples,
    output_file,
    max_new_tokens=256,
    temperature=0.0,
):
    with open(output_file, "w", encoding="utf-8") as out:
        for i, item in enumerate(samples, 1):
            print(f"  Processing {i}/{len(samples)}: {item['id']}")

            prompt = build_prompt(item["instruction"], item["input"])
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0.0),
                    eos_token_id=tokenizer.eos_token_id,
                )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = decoded.split("[/INST]")[-1].strip()

            out.write(f"\n{'=' * 60}\n")
            out.write(f"ID: {item['id']}\n")
            out.write(f"{'=' * 60}\n")
            out.write(f"INSTRUCTION:\n{item['instruction']}\n")
            out.write(f"\nINPUT:\n{item['input']}\n")
            out.write(f"\nRESPONSE:\n{response}\n")

            out.flush()


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch evaluation for finance language tasks")

    parser.add_argument("--eval_file", required=True, help="Evaluation jsonl file")
    parser.add_argument("--out_dir", default="results", help="Output directory")

    parser.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument(
        "--model_type",
        choices=["base", "ft"],
        default="base",
        help="Evaluate base model or fine-tuned model",
    )
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter (required for ft)")

    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prefix", required=True, help="Output file prefix")

    args = parser.parse_args()

    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Load evaluation samples
    # -------------------------------------------------
    print(f"📂 Loading evaluation samples from: {args.eval_file}")
    samples = load_samples(args.eval_file)
    print(f"   Loaded {len(samples)} samples\n")

    # -------------------------------------------------
    # Load tokenizer (once)
    # -------------------------------------------------
    print("🔹 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------------------------------
    # Quantization config (4-bit QLoRA style)
    # -------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # -------------------------------------------------
    # Load model (single-model mode)
    # -------------------------------------------------
    print(f"🔹 Loading model type: {args.model_type.upper()}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    if args.model_type == "ft":
        assert args.adapter is not None, "FT mode requires --adapter"
        print(f"🔹 Loading LoRA adapter from: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()

    # -------------------------------------------------
    # Output file (single, unambiguous)
    # -------------------------------------------------
    output_file = OUT_DIR / f"{args.model_type}_language_{args.prefix}.txt"

    print(f"▶ Running {args.model_type.upper()} inference → {output_file}\n")

    run_inference(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        output_file=output_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print("\n🎉 Evaluation completed successfully.")
    print(f"📄 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
