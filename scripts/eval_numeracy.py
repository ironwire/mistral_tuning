import json, re, math

def extract_number(s):
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group()) if m else None

def eval_file(pred_file, tol=0.01):
    em, tol_ok, total = 0, 0, 0
    for line in open(pred_file):
        obj = json.loads(line)
        gold = extract_number(obj["gold"])
        pred = extract_number(obj["pred"])
        if gold is None or pred is None:
            continue
        total += 1
        if abs(pred - gold) < 1e-6:
            em += 1
        if abs(pred - gold) / max(abs(gold), 1e-6) <= tol:
            tol_ok += 1
    return {
        "EM": em / total,
        "Tol@1%": tol_ok / total,
        "Total": total
    }

if __name__ == "__main__":
    print(eval_file("outputs/ft_A_B_C/numeracy_preds.jsonl"))
