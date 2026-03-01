#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import random
import argparse
from pathlib import Path

DEFAULT_INSTRUCTION = (
    "Summarize the following financial text in a concise and formal manner. "
    "Focus on key financial facts and avoid speculative language."
)

def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def gen_S0(idx: int) -> dict:
    # No absolute figures; only relative/qualitative signals
    growth = random.choice([3, 5, 8, 10, 12, 15])
    debt_state = random.choice([
        "remained stable",
        "declined slightly",
        "increased modestly",
    ])
    capex = random.choice(["increased", "decreased", "remained stable"])
    margin = random.choice(["improved", "compressed", "remained stable"])
    headwinds = random.choice(["FX headwinds", "higher input costs", "supply constraints", "demand softness"])
    investments = random.choice(["facility upgrades", "automation initiatives", "IT modernization", "capacity expansion"])

    text = (
        f"In fiscal year 2023, the company achieved year-over-year revenue growth of {growth}%. "
        f"Operating margins {margin} due to {headwinds}. "
        f"Net debt {debt_state}. "
        f"Capital expenditures {capex} driven by {investments}. "
        f"Management emphasized disciplined cost control and cautious guidance."
    )

    return {
        "id": f"SUM_S0_{idx:04d}",
        "group": "S0_no_absolute",
        "task": "summarization",
        "instruction": DEFAULT_INSTRUCTION,
        "input": text
    }

def gen_S1(idx: int) -> dict:
    # With absolute figures explicitly present
    rev = random.randint(800, 8000)  # USD million
    cogs = random.randint(int(rev * 0.35), int(rev * 0.85))
    ocf = random.randint(50, 2000)   # USD million
    growth = random.choice([3, 5, 8, 10, 12, 15])
    debt = random.randint(100, 5000) # USD million
    capex = random.randint(10, 1200) # USD million

    text = (
        "Income Statement and Cash Flow Summary (USD, million)\n"
        f"Revenue: {rev}\n"
        f"COGS: {cogs}\n"
        f"Operating Cash Flow: {ocf}\n"
        f"Net Debt: {debt}\n"
        f"Capital Expenditures: {capex}\n\n"
        f"In fiscal year 2023, the company reported revenue growth of {growth}% year-over-year."
    )

    return {
        "id": f"SUM_S1_{idx:04d}",
        "group": "S1_with_absolute",
        "task": "summarization",
        "instruction": DEFAULT_INSTRUCTION,
        "input": text
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n0", type=int, default=40, help="Number of S0 samples (no absolute)")
    ap.add_argument("--n1", type=int, default=40, help="Number of S1 samples (with absolute)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default="data/eval/eval_summary_80.jsonl")
    args = ap.parse_args()

    random.seed(args.seed)

    records = []
    for i in range(args.n0):
        records.append(gen_S0(i + 1))
    for i in range(args.n1):
        records.append(gen_S1(i + 1))

    random.shuffle(records)

    out_path = Path(args.output)
    write_jsonl(out_path, records)
    print(f"✅ Wrote {len(records)} samples → {out_path} (S0={args.n0}, S1={args.n1}, seed={args.seed})")

if __name__ == "__main__":
    main()