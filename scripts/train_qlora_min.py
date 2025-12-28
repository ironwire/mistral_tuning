#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal QLoRA fine-tuning for Mistral-7B-Instruct on a single 12GB GPU (e.g., RTX A3000).

Data format (JSONL), each line like:
{
  "instruction": "...",
  "input": "...",
  "output": "...",
  "task": "...", "id": "...", "meta": {...}  // optional
}

Run example:
  CUDA_VISIBLE_DEVICES=0 python train_qlora_min.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --train_jsonl data/train_C_syn.jsonl \
    --out_dir outputs/ft_C_only \
    --max_len 1024 \
    --epochs 1
"""

import os
import json
import argparse
from typing import Dict, List

import torch
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# -----------------------------
# 1) 把一条 jsonl 样本拼成 “指令微调”常用的训练文本
# -----------------------------
def build_prompt(instruction: str, user_input: str, answer: str) -> str:
    """
    这里我们采用一种非常朴素的 prompt 模板：
    - [INST] ... [/INST] 是 Mistral Instruct 常见的对话格式之一
    - 训练目标：让模型在看到指令+输入后，生成 answer
    """
    instruction = (instruction or "").strip()
    user_input = (user_input or "").strip()
    answer = (answer or "").strip()

    # 如果 input 为空，就只用 instruction；否则 instruction + input 拼接
    if user_input:
        user_part = f"{instruction}\n\n{user_input}"
    else:
        user_part = instruction

    # 注意：训练文本里要包含“正确答案”，Trainer 才能做 next-token prediction 学习
    # 末尾加 tokenizer.eos_token 有助于模型学会“在这里结束”
    return f"<s>[INST] {user_part} [/INST] {answer}</s>"


# -----------------------------
# 2) 读取 jsonl，构造 HuggingFace Dataset
# -----------------------------
def load_jsonl(path: str) -> Dataset:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(obj)
    return Dataset.from_list(rows)


# -----------------------------
# 3) 主函数：QLoRA 训练流程
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    # 12GB 显存：建议 max_len=1024 起步，2048 容易爆显存/变慢
    parser.add_argument("--max_len", type=int, default=1024)

    # 训练轮数：论文实验通常 2~3，先最小跑通用 1
    parser.add_argument("--epochs", type=float, default=1.0)

    # 12GB 显存：per_device_batch_size 通常设 1，然后靠梯度累积做大 batch
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)

    # 学习率：QLoRA 常见 1e-4 ~ 2e-4，先用 2e-4
    parser.add_argument("--lr", type=float, default=2e-4)

    # LoRA 的 rank：r=8 是“稳妥论文参数”；r 越大越吃显存
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 让实验可复现：固定随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # -----------------------------
    # A) 准备 tokenizer
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # 有的 LLM 没有 pad_token（Mistral 常见情况），我们把 pad_token 设成 eos_token
    # 这样在 batch padding 时不会报错
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------
    # B) QLoRA 的 “Q”：4-bit 量化配置
    # -----------------------------
    # 关键点：
    # - load_in_4bit=True：用 4-bit 载入基座权重，显存大幅降低
    # - nf4：QLoRA 论文中常用的量化类型
    # - compute_dtype=float16：A3000 上 fp16 稳定
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # -----------------------------
    # C) 加载量化后的基座模型
    # -----------------------------
    # device_map="auto"：让 transformers 自动把模型放到 GPU（单卡就会放满一张卡）
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # 训练前的常规准备：
    # - prepare_model_for_kbit_training：为 k-bit（4-bit）训练做兼容处理
    #   常包含：启用 gradient checkpointing 前的设置、输入输出 dtype 适配等
    model = prepare_model_for_kbit_training(model)

    # -----------------------------
    # D) QLoRA 的 “LoRA”：在注意力层挂可训练 Adapter
    # -----------------------------
    # target_modules 是关键：
    # 对 Mistral / Llama 系列，通常对 q/k/v/o 四个投影加 LoRA，性价比最好
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # 把 LoRA “挂”到模型上：此时只有 LoRA 参数会参与训练
    model = get_peft_model(model, lora_config)

    # 打印可训练参数占比（你写论文 Methods/Appendix 会用到）
    model.print_trainable_parameters()

    # -----------------------------
    # E) 准备训练数据：jsonl -> prompt -> tokenize
    # -----------------------------
    ds = load_jsonl(args.train_jsonl)

    def to_text(example: Dict) -> Dict:
        txt = build_prompt(
            instruction=example.get("instruction", ""),
            user_input=example.get("input", ""),
            answer=example.get("output", ""),
        )
        return {"text": txt}

    ds = ds.map(to_text, remove_columns=ds.column_names)

    def tokenize(batch: Dict) -> Dict:
        # truncation=True：超过 max_len 截断（12GB 必须）
        # padding="max_length"：固定长度 padding，训练更稳定
        tok = tokenizer(
            batch["text"],
            max_length=args.max_len,
            truncation=True,
            padding="max_length",
        )
        # Causal LM 的 labels 就是 input_ids 自身（让模型预测下一个 token）
        tok["labels"] = tok["input_ids"].copy()
        return tok

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])

    # -----------------------------
    # F) Data collator：这里我们已经自己做了 labels，不需要 MLM
    # -----------------------------
    # DataCollatorForLanguageModeling(mlm=False) 是最省事的选择
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # -----------------------------
    # G) TrainingArguments：按 12GB 单卡的“稳妥配置”
    # -----------------------------
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,

        # fp16 对 A3000 是最稳的；bf16 在部分卡上可能不支持或不稳定
        fp16=True,
        bf16=False,

        # 日志与保存：先简单
        logging_steps=10,
        save_steps=200,                 # 视数据量调整；小数据可以更频繁保存
        save_total_limit=2,

        # 关键：不要频繁评测/复杂调度，先跑通最小实验
        eval_strategy="no",

        # 12GB 常用策略：关闭过度并行，避免 CPU 抢资源
        dataloader_num_workers=2,

        # 让 Trainer 少一些“自动行为”，更可控
        report_to=[],

        # 对 12GB 来说，有时 gradient checkpointing 能救命（更省显存但更慢）
        # 你如果 OOM，可以把下面这行打开：
        # gradient_checkpointing=True,
    )

    # -----------------------------
    # H) Trainer：启动训练
    # -----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # -----------------------------
    # I) 保存 LoRA adapter（注意：QLoRA 通常只保存 adapter，而不是整模型）
    # -----------------------------
    # 这会输出 adapter_config.json + adapter_model.safetensors（或 bin）
    trainer.model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    print(f"\n✅ Done. LoRA adapter saved to: {args.out_dir}")
    print("Tip: For inference, load base model + this adapter.")


if __name__ == "__main__":
    main()
