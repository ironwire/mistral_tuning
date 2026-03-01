#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hallucination evaluation for financial summarization.

Two modes:
- strict:
    Only counts currency-formatted absolute numbers (e.g., USD 1,234).
- semantic:
    Counts any absolute financial quantity (e.g., 3.3 billion, 1033 million)
    that is not grounded in the input.

Grouping:
- SUM_S0_xxxx: no absolute numbers allowed
- SUM_S1_xxxx: absolute numbers allowed only if grounded
"""

import json
import re
import argparse
import random
from typing import Dict, List, Tuple


# -----------------------------
# Regex patterns
# -----------------------------
STRICT_ABS_RE = re.compile(
    r"\b(?:USD|\$)\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?\b",
    re.IGNORECASE,
)

SEMANTIC_ABS_RE = re.compile(
    r"""
    (?:USD|\$)?\s*
    \d+(?:\.\d+)?\s*
    (?:million|billion|thousand)?
    """,
    re.IGNORECASE | re.VERBOSE,
)

NUM_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?")


# -----------------------------
# Bootstrap CI
# -----------------------------
def bootstrap_ci(
    flags: List[int],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    if not flags:
        return 0.0, 0.0

    rng = random.Random(seed)
    means = []

    for _ in range(n_boot):
        sample = [rng.choice(flags) for _ in range(len(flags))]
        means.append(sum(sample) / len(sample))

    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return lo, hi


# -----------------------------
# Hallucination detection
# -----------------------------
def has_hallucination(
    eval_item: Dict,
    response: str,
    mode: str,
) -> bool:
    sample_id = eval_item["id"]
    inp = eval_item["input"]

    if mode == "strict":
        abs_re = STRICT_ABS_RE
    else:
        abs_re = SEMANTIC_ABS_RE

    out_nums = abs_re.findall(response)
    if not out_nums:
        return False

    # S0: any absolute number is hallucination
    if sample_id.startswith("SUM_S0"):
        return True

    # S1: must be grounded in input
    in_nums = set(NUM_TOKEN_RE.findall(inp))
    out_flat = set(NUM_TOKEN_RE.findall(" ".join(out_nums)))

    new_nums = out_flat - in_nums
    return len(new_nums) > 0


# -----------------------------
# Parse model output txt
# -----------------------------
def parse_outputs_txt(path: str) -> Dict[str, str]:
    outputs = {}
    cur_id = None
    collecting = False
    buf = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()

            if line.startswith("ID:"):
                cur_id = line.replace("ID:", "").strip()
                buf = []
                collecting = False

            elif line.startswith("RESPONSE:"):
                collecting = True

            elif collecting:
                if line.startswith("="):
                    continue
                buf.append(line)

            if cur_id and collecting and line.strip() == "":
                outputs[cur_id] = "\n".join(buf).strip()
                collecting = False

        if cur_id and collecting:
            outputs[cur_id] = "\n".join(buf).strip()

    return outputs


# -----------------------------
# Load eval samples
# -----------------------------
def load_eval_jsonl(path: str) -> Dict[str, Dict]:
    items = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            items[obj["id"]] = obj
    return items


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser("Evaluate hallucination rate (strict vs semantic)")
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--pred_txt", required=True)
    parser.add_argument("--mode", choices=["strict", "semantic"], required=True)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    eval_items = load_eval_jsonl(args.eval_jsonl)
    preds = parse_outputs_txt(args.pred_txt)

    rows = []

    for sid, item in eval_items.items():
        if sid not in preds:
            continue
        h = has_hallucination(item, preds[sid], args.mode)
        rows.append((sid, int(h)))

    def compute(prefix=None):
        flags = [h for sid, h in rows if prefix is None or sid.startswith(prefix)]
        n = len(flags)
        rate = (sum(flags) / n) if n > 0 else 0.0
        lo, hi = bootstrap_ci(flags, args.n_boot, seed=args.seed)
        return n, rate, lo, hi

    n_all, r_all, lo_all, hi_all = compute()
    n_s0, r_s0, lo_s0, hi_s0 = compute("SUM_S0")
    n_s1, r_s1, lo_s1, hi_s1 = compute("SUM_S1")

    print(f"Mode: {args.mode}")
    print(f"Overall: n={n_all}, rate={r_all*100:.1f}%, 95% CI [{lo_all*100:.1f}%, {hi_all*100:.1f}%]")
    print(f"S0 (no absolute): n={n_s0}, rate={r_s0*100:.1f}%, 95% CI [{lo_s0*100:.1f}%, {hi_s0*100:.1f}%]")
    print(f"S1 (with absolute): n={n_s1}, rate={r_s1*100:.1f}%, 95% CI [{lo_s1*100:.1f}%, {hi_s1*100:.1f}%]")


if __name__ == "__main__":
    main()
