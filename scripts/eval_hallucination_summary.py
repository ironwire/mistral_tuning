#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate hallucination rate for financial summarization outputs.

Definition:
- S0 samples (SUM_S0_xxxx): no absolute numbers in input
  → ANY absolute number in output is hallucination
- S1 samples (SUM_S1_xxxx): absolute numbers allowed only if present in input
  → hallucination if new absolute numbers are introduced
"""

import json
import re
import argparse
import random
from typing import Dict, List, Tuple


# -----------------------------
# Regex patterns
# -----------------------------
NUM_TOKEN_RE = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b")
ABS_NUM_RE = re.compile(r"\b(?:USD|\$)?\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?\b", re.IGNORECASE)


# -----------------------------
# Bootstrap CI
# -----------------------------
def bootstrap_ci(
    flags: List[int],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    if len(flags) == 0:
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
def has_fabricated_absolute(eval_item: Dict, response: str) -> bool:
    """
    Determine whether the response fabricates absolute numerical values.
    """
    sample_id = eval_item["id"]
    inp = eval_item["input"]

    # If no absolute numbers appear in output → no hallucination
    if not ABS_NUM_RE.search(response):
        return False

    # S0: no absolute numbers allowed at all
    if sample_id.startswith("SUM_S0"):
        return True

    # S1: absolute numbers must be grounded in input
    in_nums = set(t.replace(",", "") for t in NUM_TOKEN_RE.findall(inp))
    out_nums = set(t.replace(",", "") for t in NUM_TOKEN_RE.findall(response))
    new_nums = out_nums - in_nums

    return len(new_nums) > 0


# -----------------------------
# Parse model output txt
# -----------------------------
def parse_outputs_txt(path: str) -> Dict[str, str]:
    """
    Parse eval_language_batch.py output file into {id: response}.
    """
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
    parser = argparse.ArgumentParser("Evaluate hallucination rate for summaries")
    parser.add_argument("--eval_jsonl", required=True, help="Evaluation jsonl file")
    parser.add_argument("--pred_txt", required=True, help="Model output txt file")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    eval_items = load_eval_jsonl(args.eval_jsonl)
    preds = parse_outputs_txt(args.pred_txt)

    rows = []

    for sid, item in eval_items.items():
        if sid not in preds:
            continue
        hallucinated = has_fabricated_absolute(item, preds[sid])
        rows.append((sid, int(hallucinated)))

    def compute(prefix: str = None):
        flags = [h for sid, h in rows if prefix is None or sid.startswith(prefix)]
        n = len(flags)
        rate = (sum(flags) / n) if n > 0 else 0.0
        lo, hi = bootstrap_ci(flags, n_boot=args.n_boot, seed=args.seed)
        return n, rate, lo, hi

    n_all, r_all, lo_all, hi_all = compute()
    n_s0, r_s0, lo_s0, hi_s0 = compute("SUM_S0")
    n_s1, r_s1, lo_s1, hi_s1 = compute("SUM_S1")

    print(f"Overall: n={n_all}, rate={r_all*100:.1f}%, 95% CI [{lo_all*100:.1f}%, {hi_all*100:.1f}%]")
    print(f"S0 (no absolute): n={n_s0}, rate={r_s0*100:.1f}%, 95% CI [{lo_s0*100:.1f}%, {hi_s0*100:.1f}%]")
    print(f"S1 (with absolute): n={n_s1}, rate={r_s1*100:.1f}%, 95% CI [{lo_s1*100:.1f}%, {hi_s1*100:.1f}%]")


if __name__ == "__main__":
    main()