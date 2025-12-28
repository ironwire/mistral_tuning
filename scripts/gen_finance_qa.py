import json
import random
import argparse

FINANCE_QA_BANK = [
    {
        "q": "What is gross margin?",
        "a": "Gross margin is calculated as (Revenue minus Cost of Goods Sold) divided by Revenue. It measures a company's profitability after direct production costs."
    },
    {
        "q": "What is the difference between gross margin and net margin?",
        "a": "Gross margin measures profitability after direct costs, while net margin accounts for all expenses, including operating costs, interest, and taxes."
    },
    {
        "q": "What does operating cash flow indicate?",
        "a": "Operating cash flow reflects the cash generated from a company's core business operations, excluding financing and investing activities."
    },
    {
        "q": "How is revenue growth typically measured?",
        "a": "Revenue growth is typically measured as the percentage increase in revenue compared to the same period in the previous year."
    },
    {
        "q": "What does stable net debt imply?",
        "a": "Stable net debt implies that a company's total debt minus cash holdings has remained relatively unchanged over a given period."
    }
]

def gen_qa(idx):
    item = random.choice(FINANCE_QA_BANK)
    return {
        "id": f"B_QA_{idx:06d}",
        "task": "finance_qa",
        "instruction": "Answer the following financial question accurately and concisely.",
        "input": item["q"],
        "output": item["a"]
    }

def main(args):
    records = [gen_qa(i) for i in range(args.n)]
    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Generated {len(records)} finance QA samples → {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--output", type=str, default="data/B_qa/train_B_finance_qa.jsonl")
    args = parser.parse_args()
    main(args)