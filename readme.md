## Financial Domain Fine-Tuning: Language, Numeracy, and Hallucination Analysis

This repository contains the experimental code, data generation scripts, and evaluation pipeline for the paper:

```
    What Does Financial Fine-Tuning Really Improve?
    A Controlled Study on Language Quality, Numerical Accuracy, and Hallucination in Financial AI Systems
```

The project investigates how different types of financial fine-tuning affect large language model behavior, with a particular focus on language quality, numerical reasoning, and hallucination in financial applications.

### Overview

Large language models are increasingly used in financial engineering applications such as report summarization and financial analysis.
This repository provides a controlled and reproducible experimental framework to study:

- Financial language adaptation (terminology, fluency, structure)
- Financial numerical reasoning (ratio calculation, growth computation)
- Hallucination behavior, especially fabricated financial figures

The experiments are designed to disentangle the effects of:

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Financial Domain Fine-Tuning: Language, Numeracy, and Hallucination Analysis](#financial-domain-fine-tuning-language-numeracy-and-hallucination-analysis)
  - [Overview](#overview)
  - [Repository Structure](#repository-structure)
  - [Environment Setup](#environment-setup)

<!-- /code_chunk_output -->

- Finance-language fine-tuning (A)
- Finance QA fine-tuning (B)
- Financial numeracy fine-tuning (C)

### Repository Structure

```text

finance-llm-tuning/
├── data/
    │   ├── evaluation_samples_30.jsonl  # Evaluation samples A, B, C 10 samples each type
│   ├── train_A_24.jsonl          # Finance language data (A), summary, key points, risk extraction, 10, 8,6 for each type respectively
│   ├── train_A_large.jsonl          # Finance language data large number of samples (A)
│   ├── train_A_small.jsonl          # Financial langugage data small number of samples(A)
│   ├── train_A_B_C_large.jsonl      # Merged training data (A+B+C 1000 samples, 500, 300, 200 for each type)
│   ├── train_B_finance_qa.jsonl          # Financial QA data
│   ├── train_C_finance_aux_nohall.jsonl          # Grounded Numeracy Constraint Samples
│   ├── train_C_strong_300.jsonl          # Financial numeracy eval set 300 strong supervisionsamples
│   └── train_C_strong_small.jsonl          # Financial numeracy eval set 300 strong supervisionsamples

│
├── scripts/
│   ├── eval_language_batch.py     # Batch inference and qualitative evaluation for finance language tasks (Base vs fine-tuned models).
│   ├── eval_numeracy.py           # Automatic evaluation of numerical reasoning accuracy (exact match and tolerance-based metrics).
│   ├── gen_finance_language.py    # Generation of finance-language training data for summarization and structured text tasks (Dataset A).
│   ├── gen_finance_numeracy.py    # Generation of synthetic financial numeracy training samples with ground-truth calculations (Dataset C).
│   ├── gen_finance_qa.py          # Generation of finance-domain question answering data covering financial concepts and definitions (Dataset B).
│   ├── gen_nohall_numeracy.py     # Generation of numeracy samples without hallucination-prone patterns for controlled analysis.
│   ├── infer_with_adapter.py      # Single-sample inference using the base model with optional LoRA adapters.
│   ├── make_preds_jsonl.py        # Utility script to convert model outputs into JSONL format for downstream evaluation.
│   ├── run_eval_A.py              # Evaluation pipeline for finance-language tasks with human assessment support.
│   └── train_qlora_min.py         # Minimal QLoRA-based fine-tuning script for controlled and reproducible experiments.
├── outputs/
│   ├── base/                    # Base model (no fine-tuning)
│   ├── ft_A/                    # Fine-tuned model (A)
│   ├── ft_A_B/                  # Fine-tuned model (A+B)
│   ├── ft_A_B_C/                # Fine-tuned model (A+B+C)
│   ├── ft_C_small/              # Fine-tuned model with weak supervision Template  (C)
│   ├── ft_C_strong/             # Fine-tuned │model with strong supervision Template  (C)
│   └── ft_C_string_300/         # Fine-tuned model with 300 strong Supervision Template samples (C)
│
├── results/
│   ├── base_A_eval.txt               # Base model financial language outputs
│   ├── base_A_eval_30.txt            # Base model financial language 30 samples outputs
│   ├── base_language_A_eval_30.txt   # Base model financial language 30 samples outputs
│   ├── ft_A_eval.txt              # Tuned model financial language outputs
│   └── ft_A_eval_30.txt              # Tuned model financial language 30 samples with A+B+C sample type 10 pieces each type   outputs
└── README.md

```

### Environment Setup

**Recommended Environment**

- OS: Linux
- GPU: ≥ 12GB VRAM (tested on RTX A3000 12GB)
- Python: 3.9 – 3.10
- CUDA: 11.8 or newer
- Install Dependencies

**Install Dependencies**

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install torch transformers peft bitsandbytes datasets accelerate matplotlib
```

🚀 **Training (QLoRA Fine-Tuning)**
Fine-tuning is performed using QLoRA for memory efficiency.

**Example: Train with A+B+C Data**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_qlora_min.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --train_jsonl data/train_A_B_C_large.jsonl \
  --out_dir outputs/ft_A_B_C \
  --epochs 1 \
  --max_len 1024 \
  --batch_size 1 \
  --grad_accum 16
```

The resulting LoRA adapters will be saved under outputs/.

🔎 **Inference and Evaluation**
Single-Sample Inference

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_with_adapter.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --adapter outputs/ft_A_B_C \
  --instruction "<instruction text>" \
  --input "<input text>"
```

Batch Inference and Evaluation(Base vs Fine-Tuned)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_language_batch.py \
  --eval_file data/eval_language_24.jsonl \
  --adapter outputs/ft_A_B_C \
  --prefix abc_eval
```

Outputs will be saved under results/ for manual or automated evaluation.

📊 **Evaluation Metrics**
The evaluation follows the methodology described in the paper:
**- Language Quality:** Human evaluation (fluency, professionalism, faithfulness)
**- Numerical Accuracy:**

- Exact Match
- Tolerance Accuracy (±1%)
- Unit / scale error rate

**- Hallucination Rate:**

- Fabricated revenue or cash flow
- Unit and scale errors
- Narrative-completion hallucination

📈 **Visualization**

Figure generation scripts (e.g., hallucination type distribution) are implemented using matplotlib.
Generated figures are saved in both .pdf and .png formats for direct inclusion in LaTeX manuscripts.

📄 **Reproducibility**

All experiments were conducted with:

- Fixed random seeds
- Identical inference settings across models
- Publicly available base models

The repository is intended to fully support reproducibility of the reported results.

---

📜 **License**

This project is released for research and academic use only.
Please check the license of the underlying base models (e.g., Mistral) before any commercial use.

---

📬 **Contact**

For questions, issues, or collaboration inquiries, please open an issue on GitHub or contact the authors via the email provided in the paper.

⭐ **Citation**

If you find this work useful, please consider citing our paper:

```bibtex
@article{your_paper_reference,
  title={What Does Financial Fine-Tuning Really Improve? A Controlled Study on Language Quality, Numerical Accuracy, and Hallucination in Financial AI Systems},
  author={Your Name},
  journal={Conference Name},
  year={2024}
}
```
