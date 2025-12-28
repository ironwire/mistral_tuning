import json
import random
import argparse

def gen_sample(idx):
    growth = random.choice([5, 8, 10, 12])
    capex_change = random.choice(["increased", "decreased"])
    debt_state = random.choice(["remained stable", "declined slightly"])

    input_text = (
        f"In fiscal year 2023, the company achieved year-over-year revenue growth of {growth}%. "
        f"Net debt {debt_state}. "
        f"Capital expenditures {capex_change} due to operational investments."
    )

    output_text = (
        f"In fiscal year 2023, the company reported year-over-year revenue growth of {growth}%. "
        f"Net debt {debt_state}, and capital expenditures {capex_change}."
    )

    return {
        "id": f"C_NOHALL_{idx:06d}",
        "task": "finance_numeracy_constraint",
        "instruction": (
            "Summarize the text.\n\n"
            "IMPORTANT:\n"
            "- Do NOT introduce any numbers that are not explicitly present in the input.\n"
            "- If no absolute financial figures are provided, do NOT fabricate them.\n"
            "- Use only the information given."
        ),
        "input": input_text,
        "output": output_text
    }

def main(args):
    records = [gen_sample(i) for i in range(args.n)]
    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Generated {len(records)} no-hallucination numeracy samples → {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--output", type=str, default="data/C_numeracy/train_C_aux_nohall.jsonl")
    args = parser.parse_args()
    main(args)
