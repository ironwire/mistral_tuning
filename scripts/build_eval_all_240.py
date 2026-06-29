#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_eval_all_240.py

将160个synthetic样本和80个real-world样本合并，
生成最终推理输入文件 eval_all_240.jsonl。

用法:
    python build_eval_all_240.py \
        --synthetic eval_synthetic_160.jsonl \
        --realworld eval_realworld_80.jsonl \
        --out        eval_all_240.jsonl
"""

import json
import argparse
from pathlib import Path


def load_jsonl(path: str) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def validate(records: list):
    errors = []

    # 1. ID唯一性
    ids = [r["id"] for r in records]
    duplicates = [i for i in set(ids) if ids.count(i) > 1]
    if duplicates:
        errors.append(f"Duplicate IDs: {duplicates}")

    # 2. 必要字段
    required = {"id", "group", "task", "instruction", "input"}
    for r in records:
        missing = required - set(r.keys())
        if missing:
            errors.append(f"{r['id']}: missing fields {missing}")

    # 3. 无空input
    empty = [r["id"] for r in records if not str(r.get("input", "")).strip()]
    if empty:
        errors.append(f"Empty input: {empty}")

    # 4. 无response字段
    has_response = [r["id"] for r in records if "response" in r or "output" in r]
    if has_response:
        errors.append(f"Unexpected response/output field in: {has_response[:5]}")

    return errors


def print_summary(syn: list, rw: list):
    syn_s0 = sum(1 for r in syn if "S0" in r["group"])
    syn_s1 = sum(1 for r in syn if "S1" in r["group"])
    rw_s0  = sum(1 for r in rw  if "S0" in r["group"])
    rw_s1  = sum(1 for r in rw  if "S1" in r["group"])

    print("\n📊 Dataset Summary")
    print(f"{'':30s} {'S0':>6} {'S1':>6} {'Total':>6}")
    print("-" * 48)
    print(f"{'Synthetic':30s} {syn_s0:>6} {syn_s1:>6} {len(syn):>6}")
    print(f"{'Real-world':30s} {rw_s0:>6} {rw_s1:>6} {len(rw):>6}")
    print(f"{'  └ old (original 20)':30s} {'10':>6} {'10':>6} {'20':>6}")
    print(f"{'  └ new (expanded 60)':30s} {rw_s0-10:>6} {rw_s1-10:>6} {'60':>6}")
    print("-" * 48)
    print(f"{'Total':30s} {syn_s0+rw_s0:>6} {syn_s1+rw_s1:>6} {len(syn)+len(rw):>6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", default="eval_synthetic_160.jsonl")
    ap.add_argument("--realworld", default="eval_realworld_80.jsonl")
    ap.add_argument("--out",       default="eval_all_240.jsonl")
    args = ap.parse_args()

    print(f"📖 Loading synthetic: {args.synthetic}")
    syn = load_jsonl(args.synthetic)
    print(f"   {len(syn)} records")

    print(f"📖 Loading real-world: {args.realworld}")
    rw = load_jsonl(args.realworld)
    print(f"   {len(rw)} records")

    merged = syn + rw

    print("\n🔍 Validating...")
    errors = validate(merged)
    if errors:
        print("❌ Validation failed:")
        for e in errors:
            print(f"   {e}")
        return
    print("✅ All checks passed")

    print_summary(syn, rw)

    out_path = Path(args.out)
    write_jsonl(out_path, merged)
    print(f"\n✅ Written → {out_path}  ({len(merged)} samples total)")


if __name__ == "__main__":
    main()
