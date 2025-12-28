import json
import random
import argparse
from pathlib import Path

random.seed(42)

# -----------------------------
# Templates
# -----------------------------

SUMMARY_INSTRUCTION = (
    "Summarize the following financial text in a concise and formal manner. "
    "Focus on key financial facts and avoid speculative language."
)

KEYPOINT_INSTRUCTION = (
    "Extract the key financial points from the following text. "
    "Return the result as concise bullet points."
)

RISK_INSTRUCTION = (
    "Identify the main risk factors mentioned in the following financial disclosure. "
    "Return a concise list of risks without additional explanation."
)

# -----------------------------
# Synthetic content pools
# -----------------------------

COMPANY_EVENTS = [
    lambda rev, growth, ocf: f"reported total revenue of USD {rev} billion",
    lambda rev, growth, ocf: f"achieved year-over-year revenue growth of {growth}%",
    lambda rev, growth, ocf: "experienced a decline in operating income due to higher costs",
    lambda rev, growth, ocf: "maintained stable net debt levels compared to the prior year",
    lambda rev, growth, ocf: "continued investment in digital infrastructure",
    lambda rev, growth, ocf: "increased capital expenditures related to facility upgrades",
    lambda rev, growth, ocf: f"generated operating cash flow of USD {ocf} million",
]

RISK_EVENTS = [
    "exposure to foreign exchange volatility",
    "potential supply chain disruptions",
    "regulatory changes increasing compliance costs",
    "macroeconomic uncertainty affecting demand",
    "rising interest rates impacting financing costs",
]

# -----------------------------
# Generators
# -----------------------------
def gen_summary(idx):
    rev = round(random.uniform(1.0, 8.0), 1)
    growth = random.randint(2, 15)
    ocf = random.randint(200, 1200)

    sentence_funcs = random.sample(COMPANY_EVENTS, k=3)
    sentences = [fn(rev, growth, ocf) for fn in sentence_funcs]

    text = (
        f"In fiscal year 2023, the company {sentences[0]}. "
        f"It {sentences[1]}. "
        f"Additionally, the company {sentences[2]}."
    )

    summary = (
        f"In fiscal year 2023, the company reported revenue of USD {rev} billion "
        f"and achieved year-over-year growth of {growth}%. "
        f"The company generated operating cash flow of USD {ocf} million."
    )

    return build_record(idx, "summary", SUMMARY_INSTRUCTION, text, summary)


def gen_key_points(idx):
    ocf = random.randint(300, 1500)
    capex = random.randint(100, 600)

    text = (
        f"The company generated operating cash flow of USD {ocf} million during the period. "
        f"Capital expenditures increased to USD {capex} million due to infrastructure investments. "
        f"Net debt remained stable year over year."
    )

    output = (
        f"- Operating cash flow amounted to USD {ocf} million.\n"
        f"- Capital expenditures increased to USD {capex} million.\n"
        f"- Net debt levels remained stable year over year."
    )

    return build_record(idx, "key_points", KEYPOINT_INSTRUCTION, text, output)


def gen_risk(idx):
    risks = random.sample(RISK_EVENTS, k=3)

    text = (
        "The company faces several risks, including "
        + ", ".join(risks[:-1])
        + f", and {risks[-1]}."
    )

    output = "\n".join(f"- {r.capitalize()}." for r in risks)

    return build_record(idx, "risk_extraction", RISK_INSTRUCTION, text, output)


def build_record(idx, subtask, instruction, inp, out):
    return {
        "id": f"A_LANG_{idx:06d}",
        "task": "finance_language",
        "subtask": subtask,
        "instruction": instruction,
        "input": inp,
        "output": out,
        "meta": {
            "source": "synthetic",
            "domain": "finance",
            "style": "formal"
        }
    }

# -----------------------------
# Main
# -----------------------------

def main(args):
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    idx = 0

    for _ in range(args.n):
        p = random.random()
        if p < 0.4:
            records.append(gen_summary(idx))
        elif p < 0.75:
            records.append(gen_key_points(idx))
        else:
            records.append(gen_risk(idx))
        idx += 1

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Generated {len(records)} finance language samples → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="Number of samples")
    parser.add_argument("--output", type=str, default="data/train_A_finance_language.jsonl")
    args = parser.parse_args()
    main(args)
