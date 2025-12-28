import json
import subprocess
from pathlib import Path

#EVAL_FILE = "data/train_A_24.jsonl"
EVAL_FILE = "data/evaluation_samples_30.jsonl"
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

BASE_OUT = OUT_DIR / "base_A_eval_30.txt"
FTA_OUT = OUT_DIR / "ft_A_eval_30.txt"

BASE_CMD = (
    'CUDA_VISIBLE_DEVICES=0 python scripts/infer_with_adapter.py '
    '--model mistralai/Mistral-7B-Instruct-v0.2 '
    '--instruction "{instruction}" '
    '--input "{input}" '
    '--max_new_tokens 256'
)

FTA_CMD = (
    'CUDA_VISIBLE_DEVICES=0 python scripts/infer_with_adapter.py '
    '--model mistralai/Mistral-7B-Instruct-v0.2 '
    '--adapter outputs/ft_A '
    '--instruction "{instruction}" '
    '--input "{input}" '
    '--max_new_tokens 256'
)

def run(cmd_template, outfile):
    with open(EVAL_FILE, "r", encoding="utf-8") as f, open(outfile, "w", encoding="utf-8") as out:
        for line in f:
            item = json.loads(line)

            instruction = item["instruction"].replace('"', '\\"')
            input_text = item["input"].replace('"', '\\"')

            out.write(f"\n===== {item['id']} =====\n")
            out.flush()

            cmd = cmd_template.format(
                instruction=instruction,
                input=input_text
            )

            subprocess.run(cmd, shell=True, stdout=out, stderr=out)

if __name__ == "__main__":
    print("▶ Running Base model evaluation...")
    run(BASE_CMD, BASE_OUT)

    print("▶ Running FT-A model evaluation...")
    run(FTA_CMD, FTA_OUT)

    print("✅ Evaluation completed.")