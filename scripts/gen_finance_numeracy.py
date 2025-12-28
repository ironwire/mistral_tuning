import json, random, uuid
from decimal import Decimal, ROUND_HALF_UP

def r(x, d=1):
    return float(Decimal(x).quantize(Decimal(f"1.{''.join(['0']*d)}"), rounding=ROUND_HALF_UP))

def gross_margin():
    rev = random.randint(500, 5000)
    cogs = random.randint(int(rev*0.3), int(rev*0.9))
    ans = r((rev - cogs) / rev * 100, 1)
    return {
        "instruction": """Compute the requested financial metric.

            IMPORTANT:
            - Return ONLY the final numeric answer.
            - Do NOT include any explanation, reasoning steps, or text.
            - Do NOT output intermediate values.
            - Use the exact format shown in the examples.
            """,
        "input": (
            f"Income Statement (USD, million)\n"
            f"Revenue: {rev}\n"
            f"COGS: {cogs}\n\n"
            f"Metric: Gross Margin (%)"
        ),
        "output": f"{ans}%",
        "meta": {"type": "gross_margin",
                 "unit": "percent",
                "rounding": "1_decimal"
        }
    }
def yoy_growth():
    prev = random.randint(200, 2000)
    curr = random.randint(int(prev*0.7), int(prev*1.5))
    ans = r((curr - prev) / prev * 100, 1)
    return {
        "instruction": "Compute the year-over-year growth rate. Return only the final numeric answer.",
        "input": f"Key Metric (USD, million)\n2023: {prev}\n2024: {curr}\n\nQuestion: Compute YoY growth (%).",
        "output": f"{ans}%",
        "meta": {"type": "yoy_growth"}
    }

GENERATORS = [gross_margin, yoy_growth]

def main(n=300, out="data/train_C_strong_300.jsonl"):
    with open(out, "w") as f:
        for _ in range(n):
            g = random.choice(GENERATORS)()
            record = {
                "id": f"C_SYN_{uuid.uuid4().hex[:8]}",
                "task": "finance_numeracy",
                **g
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
