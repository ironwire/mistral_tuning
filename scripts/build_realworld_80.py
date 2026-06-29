#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_realworld_80.py

从Excel文件提取60个新增real-world样本的Input，
再从原有txt文件提取20个旧real-world样本的Input，
合并成eval_realworld_80.jsonl。

用法:
    python build_realworld_80.py \
        --xlsx  RealWorld样本采集记录表.xlsx \
        --txt   ft_language_summary_80_ftA_new.txt \
        --out   eval_realworld_80.jsonl
"""

import json
import re
import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INSTRUCTION = (
    "Summarize the following financial text in a concise and formal manner. "
    "Focus on key financial facts and avoid speculative language."
)


def extract_new_from_excel(xlsx_path: str) -> list:
    """从Excel提取60个新增real-world样本。"""
    df = pd.read_excel(xlsx_path, sheet_name="样本记录表", header=2)

    col_id    = "Sample ID\n(必填)"
    col_group = "Grounding\n条件"
    col_input = "Input段落\n原文"

    records = []
    for _, row in df.iterrows():
        sample_id  = str(row[col_id]).strip()
        grounding  = str(row[col_group]).strip()
        input_text = str(row[col_input]).strip()

        # 跳过空行或无效行
        if not sample_id or sample_id == "nan":
            continue
        if not input_text or input_text == "nan":
            print(f"  ⚠️  Skipping {sample_id}: empty input")
            continue

        group = "S0_no_absolute" if grounding == "S0" else "S1_with_absolute"

        records.append({
            "id":          sample_id,
            "group":       group,
            "task":        "summarization",
            "instruction": DEFAULT_INSTRUCTION,
            "input":       input_text,
        })

    return records


def extract_old_from_txt(txt_path: str) -> list:
    """从原有txt文件提取20个旧real-world样本（ID含前缀数字，如SUM_S1_1038）。"""
    with open(txt_path, encoding="utf-8") as f:
        content = f.read()

    sep = "=" * 60
    blocks = content.split(sep)

    records = []
    i = 1
    while i < len(blocks) - 1:
        id_block      = blocks[i].strip()
        content_block = blocks[i + 1]
        i += 2

        id_match = re.search(r"ID:\s*(SUM_[S01]+_\d+)", id_block)
        if not id_match:
            continue
        sample_id = id_match.group(1)

        # 只保留real-world样本：ID第三段首位不为0（如1038、2025）
        id_suffix = sample_id.split("_")[-1]
        if re.match(r"^0\d{3}$", id_suffix):
            continue  # 跳过synthetic样本

        input_match = re.search(
            r"INPUT:\s*\n(.*?)(?=\nRESPONSE:|\Z)", content_block, re.DOTALL
        )
        if not input_match:
            continue
        input_text = input_match.group(1).strip()

        group = "S0_no_absolute" if "_S0_" in sample_id else "S1_with_absolute"

        records.append({
            "id":          sample_id,
            "group":       group,
            "task":        "summarization",
            "instruction": DEFAULT_INSTRUCTION,
            "input":       input_text,
        })

    return records


def write_jsonl(path: Path, records: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Excel样本记录表（60个新增real-world）")
    ap.add_argument("--txt",  required=True, help="原有模型输出txt文件（含20个旧real-world）")
    ap.add_argument("--out",  default="eval_realworld_80.jsonl", help="输出文件")
    args = ap.parse_args()

    # 1. 提取60个新增样本
    print(f"📖 Reading Excel: {args.xlsx}")
    new_records = extract_new_from_excel(args.xlsx)
    s0_new = sum(1 for r in new_records if "S0" in r["group"])
    s1_new = sum(1 for r in new_records if "S1" in r["group"])
    print(f"   Extracted: {len(new_records)} new real-world samples  (S0={s0_new}, S1={s1_new})")

    # 2. 提取20个旧样本
    print(f"📖 Reading txt: {args.txt}")
    old_records = extract_old_from_txt(args.txt)
    s0_old = sum(1 for r in old_records if "S0" in r["group"])
    s1_old = sum(1 for r in old_records if "S1" in r["group"])
    print(f"   Extracted: {len(old_records)} old real-world samples  (S0={s0_old}, S1={s1_old})")

    # 3. 检查ID不重复
    new_ids = {r["id"] for r in new_records}
    old_ids = {r["id"] for r in old_records}
    overlap = new_ids & old_ids
    if overlap:
        print(f"⚠️  Warning: {len(overlap)} overlapping IDs: {overlap}")
    else:
        print("✅ No ID overlap between old and new samples")

    # 4. 合并：旧20在前，新60在后
    merged = old_records + new_records
    s0_total = sum(1 for r in merged if "S0" in r["group"])
    s1_total = sum(1 for r in merged if "S1" in r["group"])
    print(f"\n📊 Merged total: {len(merged)} real-world samples  (S0={s0_total}, S1={s1_total})")

    # 5. 写出
    out_path = Path(args.out)
    write_jsonl(out_path, merged)
    print(f"✅ Written → {out_path}")


if __name__ == "__main__":
    main()
