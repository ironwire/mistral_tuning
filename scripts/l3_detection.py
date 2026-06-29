#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
l3_detection.py

L3 (Covert-Implicit) hallucination detection via semantic pattern matching
combined with input grounding verification.

Implements the detection protocol described in Appendix D of:
  "When Financial Fine-tuning Fails: A Three-Level Detectability Analysis
   of Numerical Hallucination in Domain-Adapted Language Models"

Definition:
  L3 hallucination = a quantitative claim in the model output that implies
  a numerical relationship (e.g., "revenue increased", "maintained stable
  net debt") WITHOUT corresponding grounding evidence in the source input.

Detection Logic:
  1. Apply semantic pattern list to model response → candidate L3 matches
  2. For each match, extract core state word (verb/adjective)
  3. Search source input for that core word (case-insensitive)
  4. If NOT found in input → confirmed L3 hallucination
  5. If found in input → grounded, not flagged

Usage:
    # Single sample
    python l3_detection.py \
        --input  "Operating margins remained stable due to demand softness." \
        --response "The company maintained stable margins and revenue increased significantly."

    # Batch mode (jsonl)
    python l3_detection.py \
        --eval   eval_all_240.jsonl \
        --results results_ftA_240.txt \
        --model  FT-A \
        --out    l3_detections_ftA.jsonl
"""

import re
import json
import argparse
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────
# L3 Semantic Pattern List (Appendix D.2)
# ─────────────────────────────────────────────────────────────
L3_PATTERNS = [
    r"\brevenue\s+(increased|decreased|grew|declined|rose|fell|surged|dropped)\b",
    r"\b(operating\s+income|net\s+income|earnings|profit)\s+"
    r"(increased|decreased|grew|declined|rose|fell|improved|declined)\b",
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

# Core state words to check for grounding
CORE_WORDS_RE = re.compile(
    r"\b(increased|decreased|grew|declined|rose|fell|surged|dropped|"
    r"stable|improved|compressed|expanded|strong|significant|substantial|"
    r"outperformed|underperformed|exceeded|missed|remained|maintained)\b",
    re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────
# Core detection function
# ─────────────────────────────────────────────────────────────
def detect_l3(input_text: str, response: str, grounding: str = "S0") -> dict:
    """
    Detect L3 hallucination in a single response.

    Args:
        input_text: Source input text
        response:   Model-generated response
        grounding:  "S0" (weak) or "S1" (strong)

    Returns:
        dict with keys:
            - l3 (bool): Whether L3 hallucination is detected
            - matches (list): Matched phrases in response
            - ungrounded (list): Matches not found in input
            - grounded (list): Matches found in input (not flagged)
    """
    if not response:
        return {"l3": False, "matches": [], "ungrounded": [], "grounded": []}

    matches = []
    ungrounded = []
    grounded_matches = []

    for m in L3_RE.finditer(response):
        phrase = m.group(0)
        matches.append(phrase)

        # Extract core state words from matched phrase
        core_words = CORE_WORDS_RE.findall(phrase)

        # Check if any core word appears in input
        is_grounded = False
        for word in core_words:
            if re.search(r"\b" + re.escape(word) + r"\b", input_text, re.IGNORECASE):
                is_grounded = True
                break

        if is_grounded:
            grounded_matches.append(phrase)
        else:
            ungrounded.append(phrase)

    l3 = len(ungrounded) > 0

    return {
        "l3": l3,
        "matches": matches,
        "ungrounded": ungrounded,
        "grounded": grounded_matches,
    }


# ─────────────────────────────────────────────────────────────
# File I/O utilities
# ─────────────────────────────────────────────────────────────
def load_eval_jsonl(path: str) -> dict:
    """Load eval jsonl → {sample_id: {group, input}}"""
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
    """Load inference results txt → {sample_id: response}"""
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


def get_grounding(sample_id: str, group: str) -> str:
    return "S0" if ("S0" in group or "S0" in sample_id) else "S1"


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="L3 hallucination detection via semantic pattern matching + grounding verification"
    )

    # Single sample mode
    ap.add_argument("--input",    default=None, help="Source input text (single sample mode)")
    ap.add_argument("--response", default=None, help="Model response (single sample mode)")
    ap.add_argument("--grounding", default="S0", choices=["S0", "S1"],
                    help="Grounding condition (default: S0)")

    # Batch mode
    ap.add_argument("--eval",    default=None, help="eval_all_240.jsonl")
    ap.add_argument("--results", default=None, help="Inference results txt file")
    ap.add_argument("--model",   default="model", help="Model label (e.g. FT-A)")
    ap.add_argument("--out",     default="l3_detections.jsonl",
                    help="Output jsonl for batch mode")

    args = ap.parse_args()

    # ── Single sample mode ──
    if args.input and args.response:
        print(f"\n{'─'*60}")
        print(f"Input:    {args.input[:120]}")
        print(f"Response: {args.response[:120]}")
        print(f"Grounding: {args.grounding}")
        print(f"{'─'*60}")

        result = detect_l3(args.input, args.response, args.grounding)

        print(f"L3 Hallucination: {'YES' if result['l3'] else 'NO'}")
        if result["matches"]:
            print(f"\nAll L3 pattern matches ({len(result['matches'])}):")
            for m in result["matches"]:
                status = "UNGROUNDED ⚠️" if m in result["ungrounded"] else "grounded ✅"
                print(f"  [{status}] \"{m}\"")
        else:
            print("No L3 patterns detected.")
        return

    # ── Batch mode ──
    if not args.eval or not args.results:
        ap.print_help()
        return

    print(f"📖 Loading eval data: {args.eval}")
    eval_data = load_eval_jsonl(args.eval)
    print(f"   {len(eval_data)} samples")

    print(f"📖 Loading results: {args.results}")
    results = load_results_txt(args.results)
    print(f"   {len(results)} responses")

    records = []
    n_l3 = 0

    for sample_id, meta in eval_data.items():
        grounding  = get_grounding(sample_id, meta["group"])
        input_text = meta["input"]
        response   = results.get(sample_id, "")

        detection = detect_l3(input_text, response, grounding)

        record = {
            "id":          sample_id,
            "group":       meta["group"],
            "grounding":   grounding,
            "model":       args.model,
            "l3":          detection["l3"],
            "matches":     detection["matches"],
            "ungrounded":  detection["ungrounded"],
            "grounded":    detection["grounded"],
        }
        records.append(record)
        if detection["l3"]:
            n_l3 += 1

    # Write output
    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    n = len(records)
    print(f"\n📊 L3 Detection Summary — Model: {args.model}")
    print(f"   Total samples: {n}")
    print(f"   L3 detected:   {n_l3}/{n} ({n_l3/n*100:.1f}%)")

    # Breakdown by grounding condition
    for gc in ["S0", "S1"]:
        subset = [r for r in records if r["grounding"] == gc]
        k = sum(1 for r in subset if r["l3"])
        print(f"   {gc}: {k}/{len(subset)} ({k/len(subset)*100:.1f}%)")

    print(f"\n✅ Detections written → {out_path}")


if __name__ == "__main__":
    main()
