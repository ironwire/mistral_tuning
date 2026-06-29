#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
annotate_hallucination.py

对三个模型（Base / FT-A / FT-ABC）的推理结果进行L1/L2/L3/Template标注。
标注逻辑严格对齐论文中的三级可检测性分类体系。

用法:
    python annotate_hallucination.py \
        --base    results_base_240.txt \
        --fta     results_ftA_240.txt \
        --ftabc   results_ftABC_240.txt \
        --eval    eval_all_240.jsonl \
        --out     annotations_240.jsonl \
        --report  hallucination_report.txt
"""

import re
import json
import argparse
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# 1. 已知Template模式（论文Table C.18 + 新数据高频模板）
# ─────────────────────────────────────────────────────────────
TEMPLATE_PATTERNS = [
    r"operating cash flow of USD",
    r"USD\s+3\.3\s+billion",
    r"USD\s+1\.3\s+billion",
    r"USD\s+1\.8\s+billion",
    r"USD\s+1011\s+million",
    r"USD\s+1033\s+million",
    r"USD\s+1038\s+million",
    r"USD\s+138\s+million",
    r"USD\s+1111\s+million",
    r"growth of 11%",
    r"growth of 13%",
    r"maintained stable net debt",
    r"continued investment in digital",
    r"decline in operating income",
]
TEMPLATE_RE = re.compile("|".join(TEMPLATE_PATTERNS), re.IGNORECASE)

# ─────────────────────────────────────────────────────────────
# 2. L1: 带货币符号的绝对数值
# ─────────────────────────────────────────────────────────────
L1_RE = re.compile(
    r"USD\s+[\d,\.]+\s*(billion|million|trillion|thousand)"
    r"|[\$€£¥]\s*[\d,\.]+\s*(billion|million|trillion|thousand)?"
    r"|\b[\d,\.]+\s*(billion|million|trillion)\s+USD",
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────
# 3. L2: 不带货币符号的专业数字
# ─────────────────────────────────────────────────────────────
L2_NUM_RE = re.compile(
    r"\b([\d,\.]+)\s*(billion|million|trillion|thousand)\b",
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────
# 4. L3: 隐含量化声明
# ─────────────────────────────────────────────────────────────
L3_PATTERNS = [
    r"\brevenue\s+(increased|decreased|grew|declined|rose|fell|surged|dropped)\b",
    r"\b(operating\s+income|net\s+income|earnings|profit)\s+(increased|decreased|grew|declined|rose|fell|improved|declined)\b",
    r"\bmaintained\s+stable\b",
    r"\bremained\s+stable\b",
    r"\bnet\s+debt\s+(remained|declined|increased|decreased|improved)\b",
    r"\bcash\s+flow\s+(improved|declined|increased|decreased|remained)\b",
    r"\bmargin\s+(improved|compressed|expanded|declined|increased|decreased)\b",
    r"\bcapital\s+expenditure[s]?\s+(increased|decreased|rose|fell|remained)\b",
    r"\bstrong\s+(growth|performance|results|revenue|earnings)\b",
    r"\bsignificant\s+(increase|decrease|growth|decline|improvement)\b",
    r"\bsubstantial\s+(growth|increase|decrease|decline|improvement)\b",
    r"\b(outperformed|underperformed|exceeded|missed)\b",
]
L3_RE = re.compile("|".join(L3_PATTERNS), re.IGNORECASE)

# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def detect_input_unit(input_text: str) -> float:
    """
    检测input文本的隐含单位倍数。
    S1 synthetic样本的表头是 '(USD, million)'，数字如3827实际是3827 million。
    如果检测到这种表头，返回1e6（即数字需要×1e6才能和response比较）。
    """
    if re.search(r'\(USD[,\s]+million\)', input_text, re.IGNORECASE):
        return 1e6
    if re.search(r'\(USD[,\s]+billion\)', input_text, re.IGNORECASE):
        return 1e9
    return 1.0  # 无隐含单位，数字即绝对值


def extract_input_numbers(input_text: str) -> set:
    """
    从input文本中提取所有数值，统一换算为实际值（考虑隐含单位）。
    """
    unit_multiplier = detect_input_unit(input_text)
    nums = set()

    # 带单位的数值（如 "$3.2 billion"）
    for m in re.finditer(r"([\d,\.]+)\s*(billion|million|trillion|thousand)", input_text, re.IGNORECASE):
        try:
            val = float(m.group(1).replace(",", ""))
            mult = {"billion": 1e9, "million": 1e6, "trillion": 1e12, "thousand": 1e3}
            nums.add(val * mult[m.group(2).lower()])
        except ValueError:
            pass

    # 纯数字（如表格中的 "3827"）— 乘以隐含单位
    for m in re.finditer(r"\b(\d{2,})\b", input_text):
        try:
            val = float(m.group(1).replace(",", ""))
            nums.add(val * unit_multiplier)
        except ValueError:
            pass

    return nums


def extract_response_l1_values(response: str) -> list:
    """
    从response中提取L1数值（带货币符号），统一换算为实际值。
    返回 [(原始匹配串, 实际值), ...]
    """
    results = []
    for m in L1_RE.finditer(response):
        raw = m.group(0)
        num_m = re.search(r"[\d,\.]+", raw)
        if not num_m:
            continue
        try:
            val = float(num_m.group(0).replace(",", ""))
        except ValueError:
            continue
        unit_m = re.search(r"billion|million|trillion|thousand", raw, re.IGNORECASE)
        if unit_m:
            mult = {"billion": 1e9, "million": 1e6, "trillion": 1e12, "thousand": 1e3}
            val *= mult.get(unit_m.group(0).lower(), 1)
        results.append((raw, val))
    return results


def is_grounded(value: float, input_nums: set, tolerance: float = 0.02) -> bool:
    """检查数值是否在input中有溯源（允许2%误差）。"""
    for iv in input_nums:
        if iv == 0:
            continue
        if abs(value - iv) / max(abs(iv), 1e-9) <= tolerance:
            return True
    return False


# ─────────────────────────────────────────────────────────────
# 核心标注函数
# ─────────────────────────────────────────────────────────────

def annotate_sample(sample_id: str, grounding: str, input_text: str, response: str) -> dict:
    if not response:
        return {"L1": False, "L2": False, "L3": False, "Template": False}

    input_nums = extract_input_numbers(input_text)

    # ── L1 ──
    l1_values = extract_response_l1_values(response)
    if grounding == "S0":
        l1 = len(l1_values) > 0
    else:
        # S1: 有L1匹配，但需检查是否有未溯源的数值
        l1 = any(not is_grounded(val, input_nums) for _, val in l1_values)

    # ── L2 ──
    l2 = False
    if grounding == "S0":
        # S0: response中出现任何量级数字即为L2
        l2 = bool(L2_NUM_RE.search(response))
        # 但已被L1覆盖的（带货币符号）不重复计
        if l2 and l1:
            # L2检测不带货币符号的部分
            l2_only = re.sub(
                r"USD\s+[\d,\.]+\s*(billion|million|trillion|thousand)"
                r"|[\$€£¥]\s*[\d,\.]+\s*(billion|million|trillion|thousand)?",
                "", response, flags=re.IGNORECASE
            )
            l2 = bool(L2_NUM_RE.search(l2_only))
    else:
        # S1: 检查不带货币符号的数值是否未溯源
        for m in L2_NUM_RE.finditer(response):
            raw_num = m.group(1) or m.group(3) or m.group(4)
            if not raw_num:
                continue
            try:
                val = float(raw_num.replace(",", ""))
            except ValueError:
                l2 = True
                break
            unit_m = re.search(r"billion|million|trillion|thousand", m.group(0), re.IGNORECASE)
            if unit_m:
                mult = {"billion": 1e9, "million": 1e6, "trillion": 1e12, "thousand": 1e3}
                val *= mult.get(unit_m.group(0).lower(), 1)
            if not is_grounded(val, input_nums):
                l2 = True
                break

    # ── L3 ──
    # L3只判断response中出现、但input中完全没有依据的量化声明。
    # 检测方法：对每个L3模式，检查input中是否有语义相近的关键词。
    # 如果input里有对应的动词/形容词支撑，则不判L3。
    l3 = False
    for m in L3_RE.finditer(response):
        matched_phrase = m.group(0).lower()
        # 提取L3模式中的核心动词/状态词
        core_words = re.findall(
            r'increased|decreased|grew|declined|rose|fell|surged|dropped|'
            r'stable|improved|compressed|expanded|strong|significant|substantial|'
            r'outperformed|underperformed|exceeded|missed|remained|maintained',
            matched_phrase, re.IGNORECASE
        )
        # 检查input中是否有这些核心词（允许同义词匹配）
        grounded = False
        for word in core_words:
            if re.search(r'\b' + word + r'\b', input_text, re.IGNORECASE):
                grounded = True
                break
        if not grounded:
            l3 = True
            break

    # ── Template ──
    template = bool(TEMPLATE_RE.search(response))

    return {"L1": l1, "L2": l2, "L3": l3, "Template": template}


# ─────────────────────────────────────────────────────────────
# 文件读取
# ─────────────────────────────────────────────────────────────

def load_results_txt(path: str) -> dict:
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


def load_eval_jsonl(path: str) -> dict:
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records[r["id"]] = {"group": r.get("group", ""), "input": r.get("input", "")}
    return records


def get_grounding(sample_id: str, group: str) -> str:
    if "S0" in group or "S0" in sample_id:
        return "S0"
    return "S1"


def is_synthetic(sample_id: str) -> bool:
    suffix = sample_id.split("_")[-1]
    return bool(re.match(r"^0\d{3}$", suffix))


# ─────────────────────────────────────────────────────────────
# 统计与报告
# ─────────────────────────────────────────────────────────────

def compute_stats(annotations: list, model: str) -> dict:
    subsets = {
        "S0_Synth": lambda a: a["grounding"] == "S0" and a["synthetic"],
        "S0_Real":  lambda a: a["grounding"] == "S0" and not a["synthetic"],
        "S1_Synth": lambda a: a["grounding"] == "S1" and a["synthetic"],
        "S1_Real":  lambda a: a["grounding"] == "S1" and not a["synthetic"],
        "Overall":  lambda a: True,
    }
    stats = {}
    for label, condition in subsets.items():
        subset = [a for a in annotations if condition(a)]
        n = len(subset)
        stats[label] = {"n": n}
        for level in ["L1", "L2", "L3", "Template"]:
            count = sum(1 for a in subset if a[model][level])
            stats[label][level] = {"count": count, "rate": round(count / n * 100, 1) if n else 0}
    return stats


def format_report(all_stats: dict, annotations: list) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("HALLUCINATION ANNOTATION REPORT")
    lines.append(f"Total samples annotated: {len(annotations)}")
    lines.append("=" * 72)

    models = ["Base", "FT-A", "FT-ABC"]
    subsets = ["S0_Synth", "S0_Real", "S1_Synth", "S1_Real", "Overall"]

    for model in models:
        lines.append(f"\n{'─'*72}")
        lines.append(f"Model: {model}")
        lines.append(f"{'─'*72}")
        lines.append(f"{'Subset':<14} {'n':>4}  {'L1':>14}  {'L2':>14}  {'L3':>14}  {'Template':>16}")
        lines.append("─" * 72)
        for subset in subsets:
            s = all_stats[model].get(subset, {})
            n = s.get("n", 0)
            if n == 0:
                lines.append(f"{subset:<14} {0:>4}")
                continue
            l1 = f"{s['L1']['count']}/{n} ({s['L1']['rate']}%)"
            l2 = f"{s['L2']['count']}/{n} ({s['L2']['rate']}%)"
            l3 = f"{s['L3']['count']}/{n} ({s['L3']['rate']}%)"
            tm = f"{s['Template']['count']}/{n} ({s['Template']['rate']}%)"
            lines.append(f"{subset:<14} {n:>4}  {l1:>14}  {l2:>14}  {l3:>14}  {tm:>16}")

    # Cross-validation（仅对原始100个样本，即S0_Synth 40 + S0_Real 10 + S1_Synth 40 + S1_Real 10）
    lines.append(f"\n{'─'*72}")
    lines.append("CROSS-VALIDATION vs. Original Paper (n=100, L1 only)")
    lines.append(f"{'─'*72}")
    paper = {
        "Base":   {"S0_Synth": 0.0, "S0_Real": 0.0, "S1_Synth": 0.0, "S1_Real": 10.0, "Overall": 1.0},
        "FT-A":   {"S0_Synth": 100.0, "S0_Real": 100.0, "S1_Synth": 50.0, "S1_Real": 100.0, "Overall": 80.0},
        "FT-ABC": {"S0_Synth": 100.0, "S0_Real": 90.0, "S1_Synth": 97.5, "S1_Real": 100.0, "Overall": 98.0},
    }
    lines.append(f"{'Model':<10} {'Subset':<14} {'Paper%':>8}  {'New%':>8}  {'Diff':>6}  {'Status':>8}")
    lines.append("─" * 60)
    for model in models:
        for subset in subsets:
            pv = paper[model].get(subset, None)
            s = all_stats[model].get(subset, {})
            nv = s.get("L1", {}).get("rate", None)
            if pv is None or nv is None:
                continue
            diff = abs(pv - nv)
            status = "✅ OK" if diff <= 5.0 else f"⚠️ {diff:.1f}pp"
            lines.append(f"{model:<10} {subset:<14} {pv:>8.1f}  {nv:>8.1f}  {diff:>6.1f}  {status:>8}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",   required=True)
    ap.add_argument("--fta",    required=True)
    ap.add_argument("--ftabc",  required=True)
    ap.add_argument("--eval",   required=True)
    ap.add_argument("--out",    default="annotations_240.jsonl")
    ap.add_argument("--report", default="hallucination_report.txt")
    args = ap.parse_args()

    print("📖 Loading eval data...")
    eval_data = load_eval_jsonl(args.eval)
    print(f"   {len(eval_data)} samples")

    print("📖 Loading model results...")
    base_results  = load_results_txt(args.base)
    fta_results   = load_results_txt(args.fta)
    ftabc_results = load_results_txt(args.ftabc)
    print(f"   Base: {len(base_results)}, FT-A: {len(fta_results)}, FT-ABC: {len(ftabc_results)}")

    print("🏷️  Annotating...")
    annotations = []
    missing = []

    for sample_id, meta in eval_data.items():
        grounding  = get_grounding(sample_id, meta["group"])
        input_text = meta["input"]
        base_resp  = base_results.get(sample_id, "")
        fta_resp   = fta_results.get(sample_id, "")
        ftabc_resp = ftabc_results.get(sample_id, "")

        if not base_resp and not fta_resp and not ftabc_resp:
            missing.append(sample_id)
            continue

        record = {
            "id":        sample_id,
            "group":     meta["group"],
            "grounding": grounding,
            "synthetic": is_synthetic(sample_id),
            "Base":   annotate_sample(sample_id, grounding, input_text, base_resp),
            "FT-A":   annotate_sample(sample_id, grounding, input_text, fta_resp),
            "FT-ABC": annotate_sample(sample_id, grounding, input_text, ftabc_resp),
        }
        annotations.append(record)

    print(f"   Annotated: {len(annotations)}, Missing: {len(missing)}")

    print("📊 Computing statistics...")
    all_stats = {}
    for model in ["Base", "FT-A", "FT-ABC"]:
        all_stats[model] = compute_stats(annotations, model)

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for a in annotations:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"✅ Annotations → {out_path}")

    report = format_report(all_stats, annotations)
    report_path = Path(args.report)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✅ Report → {report_path}")

    print("\n📈 Quick summary (Overall):")
    print(f"{'Model':<10} {'L1':>8}  {'L2':>8}  {'L3':>8}  {'Template':>10}")
    print("─" * 50)
    for model in ["Base", "FT-A", "FT-ABC"]:
        s = all_stats[model]["Overall"]
        print(f"{model:<10} {s['L1']['rate']:>7.1f}%  {s['L2']['rate']:>7.1f}%  {s['L3']['rate']:>7.1f}%  {s['Template']['rate']:>9.1f}%")


if __name__ == "__main__":
    main()
