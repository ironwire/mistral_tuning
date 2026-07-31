#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
annotate_ablation.py

对 FT-A×3.5 ablation 模型的推理结果进行 L1/L2/L3/Template 标注。
标注逻辑与 annotate_hallucination.py 完全一致，
额外输出与原有三模型结果的对比表（volume confound ablation 专用）。

用法:
    python annotate_ablation.py \
        --fta35   results/results_ftA3500_240.txt \
        --eval    data/eval_all_240.jsonl \
        --orig    results/annotations_240.jsonl \
        --out     results/annotations_ablation.jsonl \
        --report  results/ablation_report.txt
"""

import re
import json
import argparse
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# 与 annotate_hallucination.py 完全一致的检测规则
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

L1_RE = re.compile(
    r"USD\s+[\d,\.]+\s*(billion|million|trillion|thousand)"
    r"|[\$€£¥]\s*[\d,\.]+\s*(billion|million|trillion|thousand)?"
    r"|\b[\d,\.]+\s*(billion|million|trillion)\s+USD",
    re.IGNORECASE
)

L2_NUM_RE = re.compile(
    r"\b([\d,\.]+)\s*(billion|million|trillion|thousand)\b",
    re.IGNORECASE
)

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
# 工具函数（与原脚本完全一致）
# ─────────────────────────────────────────────────────────────

def detect_input_unit(input_text):
    if re.search(r'\(USD[,\s]+million\)', input_text, re.IGNORECASE):
        return 1e6
    if re.search(r'\(USD[,\s]+billion\)', input_text, re.IGNORECASE):
        return 1e9
    return 1.0


def extract_input_numbers(input_text):
    unit_multiplier = detect_input_unit(input_text)
    nums = set()
    for m in re.finditer(r"([\d,\.]+)\s*(billion|million|trillion|thousand)", input_text, re.IGNORECASE):
        try:
            val = float(m.group(1).replace(",", ""))
            mult = {"billion": 1e9, "million": 1e6, "trillion": 1e12, "thousand": 1e3}
            nums.add(val * mult[m.group(2).lower()])
        except ValueError:
            pass
    for m in re.finditer(r"\b(\d{2,})\b", input_text):
        try:
            nums.add(float(m.group(1).replace(",", "")) * unit_multiplier)
        except ValueError:
            pass
    return nums


def extract_response_l1_values(response):
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


def is_grounded(value, input_nums, tolerance=0.02):
    for iv in input_nums:
        if iv == 0:
            continue
        if abs(value - iv) / max(abs(iv), 1e-9) <= tolerance:
            return True
    return False


def annotate_sample(sample_id, grounding, input_text, response):
    if not response:
        return {"L1": False, "L2": False, "L3": False, "Template": False}

    input_nums = extract_input_numbers(input_text)

    # L1
    l1_values = extract_response_l1_values(response)
    if grounding == "S0":
        l1 = len(l1_values) > 0
    else:
        l1 = any(not is_grounded(val, input_nums) for _, val in l1_values)

    # L2
    l2 = False
    if grounding == "S0":
        l2 = bool(L2_NUM_RE.search(response))
        if l2 and l1:
            l2_only = re.sub(
                r"USD\s+[\d,\.]+\s*(billion|million|trillion|thousand)"
                r"|[\$€£¥]\s*[\d,\.]+\s*(billion|million|trillion|thousand)?",
                "", response, flags=re.IGNORECASE
            )
            l2 = bool(L2_NUM_RE.search(l2_only))
    else:
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

    # L3
    l3 = False
    for m in L3_RE.finditer(response):
        matched_phrase = m.group(0).lower()
        core_words = re.findall(
            r'increased|decreased|grew|declined|rose|fell|surged|dropped|'
            r'stable|improved|compressed|expanded|strong|significant|substantial|'
            r'outperformed|underperformed|exceeded|missed|remained|maintained',
            matched_phrase, re.IGNORECASE
        )
        grounded = False
        for word in core_words:
            if re.search(r'\b' + word + r'\b', input_text, re.IGNORECASE):
                grounded = True
                break
        if not grounded:
            l3 = True
            break

    # Template
    template = bool(TEMPLATE_RE.search(response))

    return {"L1": l1, "L2": l2, "L3": l3, "Template": template}


# ─────────────────────────────────────────────────────────────
# 文件读取
# ─────────────────────────────────────────────────────────────

def load_results_txt(path):
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
            records[r["id"]] = {"group": r.get("group", ""), "input": r.get("input", "")}
    return records


def load_orig_annotations(path):
    """加载原有三模型的标注结果，用于对比"""
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records[r["id"]] = r
    return records


def get_grounding(sample_id, group):
    if "S0" in group or "S0" in sample_id:
        return "S0"
    return "S1"


def is_synthetic(sample_id):
    suffix = sample_id.split("_")[-1]
    return bool(re.match(r"^0\d{3}$", suffix))


# ─────────────────────────────────────────────────────────────
# 统计
# ─────────────────────────────────────────────────────────────

def compute_stats(annotations, model_key):
    subsets = {
        "S0_Synth": lambda a: a["grounding"] == "S0" and a["synthetic"],
        "S0_Real":  lambda a: a["grounding"] == "S0" and not a["synthetic"],
        "S1_Synth": lambda a: a["grounding"] == "S1" and a["synthetic"],
        "S1_Real":  lambda a: a["grounding"] == "S1" and not a["synthetic"],
        "Overall":  lambda a: True,
    }
    stats = {}
    for label, cond in subsets.items():
        subset = [a for a in annotations if cond(a)]
        n = len(subset)
        stats[label] = {"n": n}
        for level in ["L1", "L2", "L3", "Template"]:
            count = sum(1 for a in subset if a[model_key][level])
            stats[label][level] = {
                "count": count,
                "rate": round(count / n * 100, 1) if n else 0
            }
    return stats


# ─────────────────────────────────────────────────────────────
# 报告生成
# ─────────────────────────────────────────────────────────────

def format_report(annotations, orig_annotations):
    lines = []
    lines.append("=" * 72)
    lines.append("VOLUME CONFOUND ABLATION REPORT")
    lines.append("Comparing FT-A (1000 samples) vs FT-A×3.5 (3500 samples, Category A only)")
    lines.append(f"Total samples: {len(annotations)}")
    lines.append("=" * 72)

    # FT-A×3.5 stats
    stats_new = compute_stats(annotations, "FT-A-3500")

    lines.append("\n── FT-A×3.5 Results (3500 Category A samples) ──")
    lines.append(f"{'Subset':<14} {'n':>4}  {'L1':>14}  {'L2':>14}  {'L3':>14}  {'Template':>16}")
    lines.append("─" * 72)
    for subset in ["S0_Synth", "S0_Real", "S1_Synth", "S1_Real", "Overall"]:
        s = stats_new.get(subset, {})
        n = s.get("n", 0)
        if n == 0:
            lines.append(f"{subset:<14} {0:>4}")
            continue
        l1 = f"{s['L1']['count']}/{n} ({s['L1']['rate']}%)"
        l2 = f"{s['L2']['count']}/{n} ({s['L2']['rate']}%)"
        l3 = f"{s['L3']['count']}/{n} ({s['L3']['rate']}%)"
        tm = f"{s['Template']['count']}/{n} ({s['Template']['rate']}%)"
        lines.append(f"{subset:<14} {n:>4}  {l1:>14}  {l2:>14}  {l3:>14}  {tm:>16}")

    # 关键对比表（ablation的核心）
    lines.append("\n")
    lines.append("=" * 72)
    lines.append("KEY COMPARISON: Volume Effect vs. Content Effect")
    lines.append("=" * 72)
    lines.append(f"{'Model':<20} {'Data':>10}  {'L1 Overall':>12}  {'L2 Overall':>12}  {'L3 Overall':>12}")
    lines.append("─" * 72)

    # 从orig annotations提取FT-A和FT-ABC的Overall数值
    if orig_annotations:
        orig_list = list(orig_annotations.values())

        def get_rate(orig_list, model_key, level):
            n = len(orig_list)
            count = sum(1 for a in orig_list if a.get(model_key, {}).get(level, False))
            return round(count / n * 100, 1) if n else 0

        fta_l1   = get_rate(orig_list, "FT-A",   "L1")
        fta_l2   = get_rate(orig_list, "FT-A",   "L2")
        fta_l3   = get_rate(orig_list, "FT-A",   "L3")
        ftabc_l1 = get_rate(orig_list, "FT-ABC", "L1")
        ftabc_l2 = get_rate(orig_list, "FT-ABC", "L2")
        ftabc_l3 = get_rate(orig_list, "FT-ABC", "L3")

        lines.append(f"{'FT-A':<20} {'1000xA':>10}  {fta_l1:>11.1f}%  {fta_l2:>11.1f}%  {fta_l3:>11.1f}%")

        new_l1 = stats_new["Overall"]["L1"]["rate"]
        new_l2 = stats_new["Overall"]["L2"]["rate"]
        new_l3 = stats_new["Overall"]["L3"]["rate"]
        lines.append(f"{'FT-A×3.5 (ablation)':<20} {'3500xA':>10}  {new_l1:>11.1f}%  {new_l2:>11.1f}%  {new_l3:>11.1f}%")
        lines.append(f"{'FT-A+B+C':<20} {'3500xABC':>10}  {ftabc_l1:>11.1f}%  {ftabc_l2:>11.1f}%  {ftabc_l3:>11.1f}%")

        lines.append("")
        lines.append("── Interpretation ──")

        vol_effect_l1 = new_l1 - fta_l1
        content_effect_l1 = ftabc_l1 - new_l1

        lines.append(f"Volume effect  (FT-A → FT-A×3.5):    L1 {vol_effect_l1:+.1f}pp")
        lines.append(f"Content effect (FT-A×3.5 → FT-A+B+C): L1 {content_effect_l1:+.1f}pp")
        lines.append("")

        if abs(vol_effect_l1) <= 5.0 and content_effect_l1 >= 5.0:
            lines.append("✅ RESULT: Volume effect is minimal; content type (B+C numeracy data) is the")
            lines.append("   primary driver of hallucination increase. Original paper conclusion SUPPORTED.")
        elif abs(vol_effect_l1) > 10.0:
            lines.append("⚠️  RESULT: Volume effect is substantial. Conclusion requires qualification.")
            lines.append("   Consider adding a stronger caveat in Section 3.2.4 and 5.4.")
        else:
            lines.append(f"⚠️  RESULT: Mixed. Volume effect ({vol_effect_l1:+.1f}pp) is non-trivial.")
            lines.append("   Recommend reporting both effects explicitly in Section 3.2.4.")
    else:
        lines.append("(Original annotations not provided — comparison skipped)")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fta35",  required=True,  help="FT-A×3.5 推理结果 txt 文件")
    ap.add_argument("--eval",   required=True,  help="eval_all_240.jsonl")
    ap.add_argument("--orig",   default=None,   help="原有 annotations_240.jsonl（用于对比，可选）")
    ap.add_argument("--out",    default="results/annotations_ablation.jsonl")
    ap.add_argument("--report", default="results/ablation_report.txt")
    args = ap.parse_args()

    print("📖 Loading eval data...")
    eval_data = load_eval_jsonl(args.eval)
    print(f"   {len(eval_data)} samples")

    print("📖 Loading FT-A×3.5 results...")
    fta35_results = load_results_txt(args.fta35)
    print(f"   {len(fta35_results)} samples loaded")

    orig_annotations = {}
    if args.orig:
        print("📖 Loading original annotations for comparison...")
        orig_annotations = load_orig_annotations(args.orig)
        print(f"   {len(orig_annotations)} original annotations loaded")

    print("🏷️  Annotating FT-A×3.5...")
    annotations = []
    missing = []

    for sample_id, meta in eval_data.items():
        grounding  = get_grounding(sample_id, meta["group"])
        input_text = meta["input"]
        resp       = fta35_results.get(sample_id, "")

        if not resp:
            missing.append(sample_id)
            continue

        record = {
            "id":          sample_id,
            "group":       meta["group"],
            "grounding":   grounding,
            "synthetic":   is_synthetic(sample_id),
            "FT-A-3500":  annotate_sample(sample_id, grounding, input_text, resp),
        }
        annotations.append(record)

    print(f"   Annotated: {len(annotations)}, Missing: {len(missing)}")

    # 保存标注结果
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for a in annotations:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"✅ Annotations → {out_path}")

    # 生成报告
    report = format_report(annotations, orig_annotations)
    report_path = Path(args.report)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✅ Report → {report_path}")

    # 快速摘要
    stats = compute_stats(annotations, "FT-A-3500")
    print("\n📈 FT-A×3.5 Quick Summary (Overall):")
    print(f"  L1: {stats['Overall']['L1']['rate']}%")
    print(f"  L2: {stats['Overall']['L2']['rate']}%")
    print(f"  L3: {stats['Overall']['L3']['rate']}%")
    print(f"  Template: {stats['Overall']['Template']['rate']}%")


if __name__ == "__main__":
    main()
SCRIPTEOF