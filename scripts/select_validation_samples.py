#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_validation_samples.py

从 annotations_240.jsonl 和推理结果txt文件中，
按策略选出50条样本用于人工验证（Human Validation）。

选取策略：优先选边界案例和有争议的样本，而不是随机抽取。
- S0_Synth 12条：FT-A和FT-A×3.5 L2/L3结论不同的
- S0_Real  10条：Base L3被自动标为Yes的（可疑artefact）
- S1_Synth 15条：L2 grounding verification边界案例
- S1_Real  13条：Base format-matching artefact集中区

用法:
    python select_validation_samples.py \
        --annotations  results/annotations_240.jsonl \
        --ablation     results/annotations_ablation.jsonl \
        --eval         data/eval_all_240.jsonl \
        --base_txt     results/results_base_240.txt \
        --fta_txt      results/results_ftA_240.txt \
        --fta35_txt    results/results_ftA3500_240.txt \
        --ftabc_txt    results/results_ftABC_240.txt \
        --out          results/validation_50_samples.jsonl \
        --seed         42
"""

import re
import json
import random
import argparse
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# 文件读取
# ─────────────────────────────────────────────────────────────

def load_jsonl(path):
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records[r["id"]] = r
    return records


def load_results_txt(path):
    """读取推理结果txt，返回 {sample_id: response_text}"""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    sep = "=" * 60
    blocks = content.split(sep)
    results = {}
    i = 1
    while i < len(blocks) - 1:
        id_block = blocks[i].strip()
        cb = blocks[i + 1]
        i += 2
        if not id_block.startswith("ID:"):
            continue
        sample_id = id_block.replace("ID:", "").strip()
        resp_match = re.search(r"RESPONSE:\s*\n(.*)", cb, re.DOTALL)
        response = resp_match.group(1).strip() if resp_match else ""
        results[sample_id] = response
    return results


def load_eval_jsonl(path):
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records[r["id"]] = r
    return records


# ─────────────────────────────────────────────────────────────
# 选样策略
# ─────────────────────────────────────────────────────────────

def get_subset(ann):
    grounding = ann.get("grounding", "")
    synthetic = ann.get("synthetic", True)
    if grounding == "S0" and synthetic:
        return "S0_Synth"
    elif grounding == "S0" and not synthetic:
        return "S0_Real"
    elif grounding == "S1" and synthetic:
        return "S1_Synth"
    else:
        return "S1_Real"


def score_sample(ann, ablation_ann):
    """
    给每个样本打一个"争议分"，越高越值得人工验证。
    分数来源：
    1. L2/L3自动标注中有Yes的（边界案例）
    2. FT-A和FT-A×3.5标注不一致（volume effect边界）
    3. Base模型有L3=Yes（可疑artefact）
    """
    score = 0

    # L2/L3有标注的样本更值得验证
    for model in ["Base", "FT-A", "FT-ABC"]:
        m = ann.get(model, {})
        if m.get("L2"):
            score += 2
        if m.get("L3"):
            score += 2

    # Base L3=Yes 是经典artefact，重点验证
    if ann.get("Base", {}).get("L3"):
        score += 3

    # FT-A和FT-A×3.5结论不一致（volume effect边界）
    if ablation_ann:
        fta_l1   = ann.get("FT-A", {}).get("L1", False)
        fta35_l1 = ablation_ann.get("FT-A-3500", {}).get("L1", False)
        fta_l2   = ann.get("FT-A", {}).get("L2", False)
        fta35_l2 = ablation_ann.get("FT-A-3500", {}).get("L2", False)
        if fta_l1 != fta35_l1:
            score += 3
        if fta_l2 != fta35_l2:
            score += 2

    # S1条件下Base有L1（format-matching artefact区）
    if ann.get("grounding") == "S1" and ann.get("Base", {}).get("L1"):
        score += 2

    return score


def select_samples(annotations, ablation_annotations, targets):
    """
    按子集分组，按争议分排序，取前N条。
    targets: {"S0_Synth": 12, "S0_Real": 10, "S1_Synth": 15, "S1_Real": 13}
    """
    groups = {k: [] for k in targets}

    for sid, ann in annotations.items():
        subset = get_subset(ann)
        if subset not in groups:
            continue
        abl = ablation_annotations.get(sid)
        score = score_sample(ann, abl)
        groups[subset].append((score, sid, ann))

    selected = {}
    for subset, items in groups.items():
        n = targets[subset]
        # 按争议分降序排，取前N条
        items.sort(key=lambda x: -x[0])
        top = items[:n]
        for _, sid, ann in top:
            selected[sid] = ann

    return selected


# ─────────────────────────────────────────────────────────────
# 输出50条样本的详细jsonl
# ─────────────────────────────────────────────────────────────

def build_output_record(sid, ann, eval_data,
                        base_txt, fta_txt, fta35_txt, ftabc_txt):
    eval_rec = eval_data.get(sid, {})
    return {
        "id": sid,
        "subset": get_subset(ann),
        "grounding": ann.get("grounding", ""),
        "synthetic": ann.get("synthetic", True),
        "input": eval_rec.get("input", ""),
        "instruction": eval_rec.get("instruction", ""),
        "responses": {
            "Base":       base_txt.get(sid, ""),
            "FT-A":       fta_txt.get(sid, ""),
            "FT-A-3500":  fta35_txt.get(sid, ""),
            "FT-A+B+C":   ftabc_txt.get(sid, ""),
        },
        "auto_annotations": {
            "Base":       ann.get("Base", {}),
            "FT-A":       ann.get("FT-A", {}),
            "FT-ABC":     ann.get("FT-ABC", {}),
        }
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations",  required=True)
    ap.add_argument("--ablation",     required=True)
    ap.add_argument("--eval",         required=True)
    ap.add_argument("--base_txt",     required=True)
    ap.add_argument("--fta_txt",      required=True)
    ap.add_argument("--fta35_txt",    required=True)
    ap.add_argument("--ftabc_txt",    required=True)
    ap.add_argument("--out",          default="results/validation_50_samples.jsonl")
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    print("📖 Loading files...")
    annotations      = load_jsonl(args.annotations)
    ablation_ann     = load_jsonl(args.ablation)
    eval_data        = load_eval_jsonl(args.eval)
    base_txt         = load_results_txt(args.base_txt)
    fta_txt          = load_results_txt(args.fta_txt)
    fta35_txt        = load_results_txt(args.fta35_txt)
    ftabc_txt        = load_results_txt(args.ftabc_txt)

    print(f"   Annotations: {len(annotations)}, Ablation: {len(ablation_ann)}")

    targets = {
        "S0_Synth": 12,
        "S0_Real":  10,
        "S1_Synth": 15,
        "S1_Real":  13,
    }

    print("🎯 Selecting samples...")
    selected = select_samples(annotations, ablation_ann, targets)
    print(f"   Selected: {len(selected)} samples")

    # 子集分布报告
    from collections import Counter
    subset_counts = Counter(get_subset(ann) for ann in selected.values())
    for subset, count in sorted(subset_counts.items()):
        print(f"   {subset}: {count}")

    # 构建输出记录
    out_records = []
    for sid, ann in selected.items():
        rec = build_output_record(
            sid, ann, eval_data,
            base_txt, fta_txt, fta35_txt, ftabc_txt
        )
        out_records.append(rec)

    # 按subset排序，方便标注者阅读
    subset_order = {"S0_Synth": 0, "S0_Real": 1, "S1_Synth": 2, "S1_Real": 3}
    out_records.sort(key=lambda x: (subset_order.get(x["subset"], 9), x["id"]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Saved → {out_path}")


if __name__ == "__main__":
    main()
