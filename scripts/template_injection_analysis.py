#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
template_injection_analysis.py

Analyzes template injection frequency in model outputs.
Identifies recurring fabricated numerical patterns across unrelated inputs,
as described in Section 4.5 and Table 8 of:
  "When Financial Fine-tuning Fails: A Three-Level Detectability Analysis
   of Numerical Hallucination in Domain-Adapted Language Models"

Template injection is defined as the insertion of memorized canonical values
regardless of input content. A pattern is considered a template if it:
  1. Appears in the model output
  2. Is NOT grounded in the source input

Usage:
    # Analyze one model
    python template_injection_analysis.py \
        --results results_ftA_240.txt \
        --eval    eval_all_240.jsonl \
        --model   FT-A \
        --out     template_analysis_ftA.jsonl

    # Compare all three models and produce Table 8
    python template_injection_analysis.py \
        --results results_base_240.txt results_ftA_240.txt results_ftABC_240.txt \
        --eval    eval_all_240.jsonl \
        --models  Base FT-A FT-ABC \
        --table8
"""

import re
import json
import argparse
from pathlib import Path
from collections import Counter
from typing import Optional

# ─────────────────────────────────────────────────────────────
# Known template patterns (Table 8 / Appendix D)
# ─────────────────────────────────────────────────────────────
TEMPLATE_PATTERNS = {
    "operating cash flow of USD":    r"operating cash flow of USD",
    "USD 3.3 billion":               r"USD\s+3\.3\s+billion",
    "USD 1.3 billion":               r"USD\s+1\.3\s+billion",
    "USD 1.8 billion":               r"USD\s+1\.8\s+billion",
    "USD 1011 million":              r"USD\s+1011\s+million",
    "USD 1033 million":              r"USD\s+1033\s+million",
    "USD 1038 million":              r"USD\s+1038\s+million",
    "USD 138 million":               r"USD\s+138\s+million",
    "USD 1111 million":              r"USD\s+1111\s+million",
    "growth of 11%":                 r"growth of 11\%",
    "growth of 13%":                 r"growth of 13\%",
    "maintained stable net debt":    r"maintained stable net debt",
    "continued investment in digital": r"continued investment in digital",
    "decline in operating income":   r"decline in operating income",
}

COMPILED_PATTERNS = {
    label: re.compile(pattern, re.IGNORECASE)
    for label, pattern in TEMPLATE_PATTERNS.items()
}

# Combined pattern for any template presence
ANY_TEMPLATE_RE = re.compile(
    "|".join(TEMPLATE_PATTERNS.values()), re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────
# Core detection functions
# ─────────────────────────────────────────────────────────────
def detect_templates(response: str) -> dict:
    """
    Detect all template patterns in a response.

    Returns:
        dict with keys:
            - any_template (bool): Whether any template was found
            - patterns_found (list): Names of matched patterns
            - pattern_matches (dict): {pattern_name: matched_text}
    """
    if not response:
        return {"any_template": False, "patterns_found": [], "pattern_matches": {}}

    patterns_found = []
    pattern_matches = {}

    for label, regex in COMPILED_PATTERNS.items():
        m = regex.search(response)
        if m:
            patterns_found.append(label)
            pattern_matches[label] = m.group(0)

    return {
        "any_template": len(patterns_found) > 0,
        "patterns_found": patterns_found,
        "pattern_matches": pattern_matches,
    }


def analyze_novel_patterns(responses: list, top_n: int = 20) -> list:
    """
    Discover novel high-frequency USD patterns not in the known template list.
    Useful for identifying new templates in expanded datasets.

    Returns list of (pattern_string, count) sorted by frequency.
    """
    usd_pattern = re.compile(
        r"USD\s+[\d,\.]+\s*(billion|million|trillion)",
        re.IGNORECASE
    )
    counter = Counter()
    for resp in responses:
        for m in usd_pattern.finditer(resp):
            # Normalize: remove internal spaces
            normalized = re.sub(r"\s+", " ", m.group(0).strip())
            counter[normalized] += 1

    # Filter out patterns already in known list
    known_values = set()
    for label in TEMPLATE_PATTERNS:
        known_values.add(label.lower())

    novel = [
        (pat, cnt) for pat, cnt in counter.most_common(top_n * 2)
        if pat.lower() not in known_values and cnt >= 2
    ]
    return novel[:top_n]


# ─────────────────────────────────────────────────────────────
# File I/O utilities
# ─────────────────────────────────────────────────────────────
def load_eval_jsonl(path: str) -> dict:
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records[r["id"]] = {
                "group": r.get("group", ""),
                "input": r.get("input", ""),
            }
    return records


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


# ─────────────────────────────────────────────────────────────
# Table 8 generation
# ─────────────────────────────────────────────────────────────
def generate_table8(model_responses: dict, n: int):
    """
    Generate Table 8: Template injection frequency by model.

    Args:
        model_responses: {model_name: {sample_id: response}}
        n: total sample count
    """
    print(f"\n{'='*70}")
    print(f"TABLE 8: Template Injection Frequency (n={n} samples each)")
    print(f"{'='*70}")
    models = list(model_responses.keys())
    header = f"{'Pattern':<38}" + "".join(f"{m:>10}" for m in models)
    print(header)
    print("─" * 70)

    for label in TEMPLATE_PATTERNS:
        regex = COMPILED_PATTERNS[label]
        counts = {}
        for model, responses in model_responses.items():
            cnt = sum(1 for r in responses.values() if regex.search(r))
            counts[model] = cnt

        # Only print if at least one model has >0
        if any(c > 0 for c in counts.values()):
            row = f'"{label}"'[:37].ljust(38)
            for model in models:
                pct = counts[model] / n * 100
                row += f"{pct:>9.0f}%"
            print(row)

    print("─" * 70)

    # Overall template presence
    print("\nOverall template presence (any pattern):")
    for model, responses in model_responses.items():
        cnt = sum(1 for r in responses.values()
                  if ANY_TEMPLATE_RE.search(r))
        print(f"  {model}: {cnt}/{n} ({cnt/n*100:.1f}%)")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Template injection frequency analysis"
    )
    ap.add_argument("--results", nargs="+", required=True,
                    help="One or more inference result txt files")
    ap.add_argument("--eval",    required=True,
                    help="eval_all_240.jsonl")
    ap.add_argument("--models",  nargs="+", default=None,
                    help="Model labels (must match order of --results)")
    ap.add_argument("--out",     default="template_analysis.jsonl",
                    help="Output jsonl file")
    ap.add_argument("--table8",  action="store_true",
                    help="Print Table 8 format comparing all models")
    ap.add_argument("--novel",   action="store_true",
                    help="Discover novel high-frequency patterns")
    args = ap.parse_args()

    # Default model labels
    if not args.models:
        args.models = [Path(r).stem for r in args.results]

    if len(args.models) != len(args.results):
        print("❌ Number of --models must match number of --results files")
        return

    # Load eval data
    print(f"📖 Loading eval data: {args.eval}")
    eval_data = load_eval_jsonl(args.eval)
    n = len(eval_data)
    print(f"   {n} samples")

    # Load all model responses
    all_responses = {}
    for model, results_path in zip(args.models, args.results):
        print(f"📖 Loading results [{model}]: {results_path}")
        all_responses[model] = load_results_txt(results_path)
        print(f"   {len(all_responses[model])} responses")

    # ── Table 8 mode ──
    if args.table8:
        generate_table8(all_responses, n)

    # ── Novel pattern discovery ──
    if args.novel:
        print(f"\n{'='*60}")
        print("Novel high-frequency USD patterns (not in known list):")
        print("─" * 60)
        for model, responses in all_responses.items():
            resp_list = list(responses.values())
            novel = analyze_novel_patterns(resp_list)
            print(f"\n{model}:")
            if novel:
                for pat, cnt in novel[:10]:
                    print(f"  {pat:<40} {cnt:>4}x ({cnt/n*100:.1f}%)")
            else:
                print("  No novel patterns found above threshold.")

    # ── Per-sample annotation ──
    all_records = []
    for sample_id, meta in eval_data.items():
        record = {
            "id":    sample_id,
            "group": meta["group"],
        }
        for model in args.models:
            response = all_responses[model].get(sample_id, "")
            detection = detect_templates(response)
            record[model] = {
                "any_template":    detection["any_template"],
                "patterns_found":  detection["patterns_found"],
            }
        all_records.append(record)

    # Write output
    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n✅ Per-sample annotations → {out_path}")


if __name__ == "__main__":
    main()
