#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_synthetic_160.py

从原有的模型输出txt文件中提取前80个synthetic样本的Input，
转成jsonl格式，再与新生成的80个合并，得到160个样本的模板文件。

用法:
    python merge_synthetic_160.py \
        --txt  ft_language_summary_80_ftA_new.txt \
        --new  eval_synthetic_new80.jsonl \
        --out  eval_synthetic_160.jsonl
"""

import json
import re
import argparse
from pathlib import Path

DEFAULT_INSTRUCTION = (
    "Summarize the following financial text in a concise and formal manner. "
    "Focus on key financial facts and avoid speculative language."
)

def extract_from_txt(txt_path: str, n: int = 80) -> list:
    """
    从txt文件中提取前n个synthetic样本（ID格式 SUM_S0_00XX / SUM_S1_00XX）。
    real-world样本（ID含四位以上前缀数字，如SUM_S1_1038）自动排除。
    """
    with open(txt_path, encoding="utf-8") as f:
        content = f.read()

    sep = "=" * 60
    blocks = content.split(sep)

    records = []
    i = 1  # blocks[0]是空白，从1开始；ID块和内容块交替出现
    while i < len(blocks) - 1:
        id_block = blocks[i].strip()
        content_block = blocks[i + 1]
        i += 2

        # 提取ID
        id_match = re.search(r"ID:\s*(SUM_[S01]+_\d+)", id_block)
        if not id_match:
            continue
        sample_id = id_match.group(1)

        # 只保留synthetic样本：ID第三段是4位数字（0001-0080）
        # real-world样本的ID如 SUM_S1_1038，第三段首位不为0
        id_suffix = sample_id.split("_")[-1]
        if not re.match(r"^0\d{3}$", id_suffix):
            continue  # 跳过real-world样本

        # 提取INPUT（在RESPONSE之前的部分）
        input_match = re.search(
            r"INPUT:\s*\n(.*?)(?=\nRESPONSE:|\Z)", content_block, re.DOTALL
        )
        if not input_match:
            continue
        input_text = input_match.group(1).strip()

        # 判断group
        group = "S0_no_absolute" if "_S0_" in sample_id else "S1_with_absolute"

        records.append({
            "id": sample_id,
            "group": group,
            "task": "summarization",
            "instruction": DEFAULT_INSTRUCTION,
            "input": input_text,
        })

        if len(records) >= n:
            break

    return records


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", required=True,
                    help="原有模型输出txt文件（含80 synthetic + 20 real-world）")
    ap.add_argument("--new", required=True,
                    help="新生成的80个synthetic样本jsonl文件")
    ap.add_argument("--out", default="eval_synthetic_160.jsonl",
                    help="输出：160个synthetic样本模板文件")
    args = ap.parse_args()

    # 1. 从txt提取原有80个synthetic的Input
    print(f"📖 Reading txt: {args.txt}")
    old_records = extract_from_txt(args.txt, n=80)
    print(f"   Extracted: {len(old_records)} synthetic samples")

    s0_old = sum(1 for r in old_records if "S0" in r["id"])
    s1_old = sum(1 for r in old_records if "S1" in r["id"])
    print(f"   S0: {s0_old}, S1: {s1_old}")

    # 2. 读取新80个
    print(f"📖 Reading new jsonl: {args.new}")
    new_records = load_jsonl(args.new)
    print(f"   Loaded: {len(new_records)} new samples")

    s0_new = sum(1 for r in new_records if "S0" in r["id"])
    s1_new = sum(1 for r in new_records if "S1" in r["id"])
    print(f"   S0: {s0_new}, S1: {s1_new}")

    # 3. 检查ID不重复
    old_ids = {r["id"] for r in old_records}
    new_ids = {r["id"] for r in new_records}
    overlap = old_ids & new_ids
    if overlap:
        print(f"⚠️  Warning: {len(overlap)} overlapping IDs: {overlap}")
    else:
        print("✅ No ID overlap between old and new samples")

    # 4. 合并
    merged = old_records + new_records
    print(f"\n📊 Merged total: {len(merged)} samples")
    print(f"   S0: {sum(1 for r in merged if 'S0' in r['id'])}")
    print(f"   S1: {sum(1 for r in merged if 'S1' in r['id'])}")

    # 5. 写出
    out_path = Path(args.out)
    write_jsonl(out_path, merged)
    print(f"\n✅ Written → {out_path}")


if __name__ == "__main__":
    main()
