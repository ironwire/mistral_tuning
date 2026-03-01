#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import math
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# -----------------------------
# Prompt builder (Mistral Instruct)
# -----------------------------
def build_prompt(instruction: str, user_input: str) -> str:
    instruction = (instruction or "").strip()
    user_input = (user_input or "").strip()
    if user_input:
        content = f"{instruction}\n\n{user_input}"
    else:
        content = instruction
    return f"<s>[INST] {content} [/INST]"


# -----------------------------
# Numeric parsing utilities
# -----------------------------
NUM_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")

def parse_first_number(text: str) -> Optional[float]:
    """
    Extract the first number from text.
    Handles commas and trailing '%' by stripping them.
    Returns float or None if no number.
    """
    if not text:
        return None
    m = NUM_RE.search(text)
    if not m:
        return None
    s = m.group(0).replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def is_percent_string(s: str) -> bool:
    return "%" in (s or "")


def parse_reference_value(ref_text: str) -> Optional[float]:
    """
    Parse the reference numeric value. If reference contains '%', treat reference as percentage number.
    Example: "41.7%" -> 41.7
    """
    return parse_first_number(ref_text)


def parse_prediction_value(pred_text: str, ref_text: str) -> Optional[float]:
    """
    Parse prediction numeric value. If reference is percentage and prediction seems like a fraction (0~1),
    you MAY want to scale it by 100. We keep it conservative:
    - If ref contains '%' and pred is in [0, 1.5], we assume pred is fraction and scale by 100.
    """
    v = parse_first_number(pred_text)
    if v is None:
        return None

    if is_percent_string(ref_text):
        if 0.0 <= v <= 1.5:
            return v * 100.0
    return v


def exact_match(pred: float, ref: float, eps: float = 1e-9) -> bool:
    return abs(pred - ref) <= eps


def within_tol_1pct(pred: float, ref: float) -> bool:
    """
    Relative error <= 1%. For near-zero refs, use absolute tolerance.
    """
    denom = max(abs(ref), 1e-9)
    rel = abs(pred - ref) / denom
    return rel <= 0.01


# -----------------------------
# Data loading
# -----------------------------
def load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


# -----------------------------
# Model loading
# -----------------------------
def load_model_and_tokenizer(model_name: str, adapter_path: Optional[str] = None):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    if adapter_path:
        # adapter_path must be a directory containing adapter_config.json
        base = PeftModel.from_pretrained(base, adapter_path)

    base.eval()
    return base, tok


# -----------------------------
# Inference (greedy, deterministic)
# -----------------------------
@torch.no_grad()
def generate_one(model, tok, instruction: str, user_input: str, max_new_tokens: int = 16) -> str:
    prompt = build_prompt(instruction, user_input)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )

    decoded = tok.decode(out[0], skip_special_tokens=True)
    # keep only assistant part
    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]", 1)[-1]
    return decoded.strip()


# -----------------------------
# Evaluation loop
# -----------------------------
def eval_numeracy(rows: List[Dict[str, Any]], model, tok, max_new_tokens: int, save_preds: Optional[str] = None):
    n = 0
    n_parsed = 0
    n_em = 0
    n_tol = 0

    pred_records = []

    for item in rows:
        inst = item.get("instruction", "")
        inp = item.get("input", "")
        ref_text = str(item.get("output", "")).strip()
        sid = item.get("id", item.get("meta", {}).get("id", ""))

        pred_text = generate_one(model, tok, inst, inp, max_new_tokens=max_new_tokens)

        ref_val = parse_reference_value(ref_text)
        pred_val = parse_prediction_value(pred_text, ref_text)

        parsed_ok = (ref_val is not None) and (pred_val is not None)
        if parsed_ok:
            n_parsed += 1
            if exact_match(pred_val, ref_val):
                n_em += 1
            if within_tol_1pct(pred_val, ref_val):
                n_tol += 1

        pred_records.append({
            "id": sid,
            "ref_text": ref_text,
            "pred_text": pred_text,
            "ref_val": ref_val,
            "pred_val": pred_val,
            "parsed_ok": parsed_ok,
        })

        n += 1

    if save_preds:
        Path(save_preds).parent.mkdir(parents=True, exist_ok=True)
        with open(save_preds, "w", encoding="utf-8") as f:
            for r in pred_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Metrics: report on ALL samples and also on PARSED subset
    parsed_rate = (n_parsed / n) if n else 0.0
    em_all = (n_em / n) if n else 0.0
    tol_all = (n_tol / n) if n else 0.0

    em_parsed = (n_em / n_parsed) if n_parsed else 0.0
    tol_parsed = (n_tol / n_parsed) if n_parsed else 0.0

    return {
        "n": n,
        "parsed_n": n_parsed,
        "parsed_rate": parsed_rate,
        "em_all": em_all,
        "tol1_all": tol_all,
        "em_parsed": em_parsed,
        "tol1_parsed": tol_parsed,
    }


def main():
    ap = argparse.ArgumentParser("Minimal numeracy evaluation (EM / <=1%)")
    ap.add_argument("--eval_jsonl", required=True, help="JSONL file with instruction/input/output per line")
    ap.add_argument("--model", required=True, help="Base model name, e.g., mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--adapter", default=None, help="LoRA adapter directory (optional)")
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="Limit number of samples (0 = all)")
    ap.add_argument("--save_preds", default=None, help="Optional path to save per-sample predictions as JSONL")
    args = ap.parse_args()

    rows = load_jsonl(args.eval_jsonl, limit=(args.limit or None))
    print(f"Loaded {len(rows)} samples from {args.eval_jsonl}")

    print("Loading model...")
    model, tok = load_model_and_tokenizer(args.model, args.adapter)

    metrics = eval_numeracy(rows, model, tok, args.max_new_tokens, save_preds=args.save_preds)

    tag = "BASE" if not args.adapter else f"ADAPTER({args.adapter})"
    print("\n=== Numeracy Eval Results ===")
    print(f"Model: {tag}")
    print(f"n = {metrics['n']}")
    print(f"parsed = {metrics['parsed_n']} ({metrics['parsed_rate']*100:.1f}%)")
    print(f"Exact Match (all): {metrics['em_all']*100:.1f}%")
    print(f"<=1% Tolerance (all): {metrics['tol1_all']*100:.1f}%")
    print(f"Exact Match (parsed only): {metrics['em_parsed']*100:.1f}%")
    print(f"<=1% Tolerance (parsed only): {metrics['tol1_parsed']*100:.1f}%")

    if args.save_preds:
        print(f"Saved per-sample outputs to: {args.save_preds}")


if __name__ == "__main__":
    main()
