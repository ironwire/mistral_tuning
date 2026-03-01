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

def build_prompt(instruction: str, user_input: str) -> str:
    instruction = instruction.strip()
    user_input = user_input.strip()
    if user_input:
        content = f"{instruction}\n\n{user_input}"
    else:
        content = instruction
    return f"<s>[INST] {content} [/INST]"

def load_samples(filepath):
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

def run_inference(model, tokenizer, samples, output_file, max_new_tokens, temperature):
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
                    do_sample=(temperature > 0),
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            #response = decoded.split("[/INST]")[-1].strip()
            response = decoded.split("</s>")[-1].strip()

            # out.write(f"# Model: {args.base_model}\n")
            # out.write(f"# Adapter: {args.adapter or 'None (Base)'}\n")
            # out.write(f"# Temperature: {args.temperature}\n")
            # out.write(f"# Max new tokens: {args.max_new_tokens}\n\n")

            out.write(f"\n{'='*60}\n")
            out.write(f"ID: {item['id']}\n")
            out.write(f"{'='*60}\n")
            out.write(f"INSTRUCTION: {item['instruction']}\n")
            out.write(f"\nINPUT:\n{item['input']}\n")
            out.write(f"\nRESPONSE:\n{response}\n")
            out.flush()

def main():
    parser = argparse.ArgumentParser(description="高效批量评估")
    parser.add_argument("--eval_file", required=True, help="评估样本文件路径")
    parser.add_argument("--model_type", choices=["base", "ft"], default="base", help="Evaluate base model or Fine-tuned model")
    parser.add_argument("--out_dir", default="results", help="输出目录")
    parser.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--adapter", default=None, help="Adapter路径（可选）")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prefix", default="eval", help="输出文件前缀")
    args = parser.parse_args()
    
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(exist_ok=True)
    
    # 加载样本
    print(f"📂 Loading evaluation samples from: {args.eval_file}")
    samples = load_samples(args.eval_file)
    print("🔹 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"   Loaded {len(samples)} samples\n")
    
    # 4-bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # ===== 1. Base Model 评估 =====
    # print("🔹 Loading BASE model...")
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     args.base_model,
    #     quantization_config=bnb_config,
    #     device_map="auto",
    #     torch_dtype=torch.float16,
    # )
    # tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    
    # base_model.eval()
    print(f"🔹 Loading model type: {args.model_type.upper()}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    if args.model_type == "ft":
        assert args.adapter is not None, "FT mode requires --adapter"
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()


    #base_output = OUT_DIR / f"base_language_{args.prefix}.txt"
    output_file = OUT_DIR / f"{args.model_type}_language_{args.prefix}.txt"
    #print(f"▶ Running BASE model inference → {base_output}")
    run_inference(model, tokenizer, samples, output_file, 
                  args.max_new_tokens, args.temperature)
    print("✅ Base model done.\n")
    
    #del base_model
    torch.cuda.empty_cache()
    
    # ===== 2. Fine-tuned Model 评估（如果提供adapter）=====
    if args.adapter:
        print(f"🔹 Loading BASE model + Adapter: {args.adapter}")
        ft_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        ft_model = PeftModel.from_pretrained(ft_model, args.adapter)
        ft_model.eval()
        
        #ft_output = OUT_DIR / f"ft_{args.prefix}.txt"
        output_file = OUT_DIR / f"{args.model_type}_language_{args.prefix}.txt"
        #print(f"▶ Running FINE-TUNED model inference → {ft_output}")
        print(f"▶ Running {args.model_type.upper()} inference → {output_file}")
        run_inference(
            model,
            tokenizer,
            samples,
            output_file,
            args.max_new_tokens,
            args.temperature
        )
        #run_inference(ft_model, tokenizer, samples, ft_output,
        #              args.max_new_tokens, args.temperature)
        
        print("✅ Fine-tuned model done.\n")
    
    print("🎉 All evaluations completed!")
    print(f"   Results saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()