# Financial LLM Hallucination Study



This repository contains the code and data for reproducing the experiments in:

> **When Financial Fine-tuning Fails: A Three-Level Detectability Analysis of Numerical Hallucination in Domain-Adapted Language Models**
>
> *Expert Systems with Applications* (Under Review)

## Key Findings

- Domain-specific fine-tuning **increases** numerical hallucination rates from 2% to 98%
- We propose a **three-level detectability taxonomy** (L1/L2/L3) for numerical hallucination
- Fine-tuned models exhibit **"template injection"** by memorizing training patterns
- L1-only evaluation underestimates hallucination rates by an order of magnitude

## Table of Contents

- [Repository Structure](#repository-structure)
- [Data](#data)
- [Scripts](#scripts)
- [Results](#results)
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Hardware Requirements](#hardware-requirements)
- [Citation](#citation)
- [License](#license)

## Repository Structure

```
finance-llm-tuning/
│
├── data/                  # Training and evaluation datasets (JSONL)
│   └── eval/              # Evaluation splits
├── scripts/               # Training, generation, and evaluation scripts
├── results/               # Generated outputs and evaluation logs
├── outputs/               # Fine-tuned model checkpoints
├── requirements.txt
└── README.md
```

## Data

All datasets are in JSONL format following an `instruction–input–output` structure.

### Training Data

| Category | Files | Description |
|----------|-------|-------------|
| **A: Financial Language** | `train_A_large.jsonl`, `train_A_small.jsonl` | Summarization, key points extraction, risk factors |
| **B: Financial Knowledge** | `train_B_finance_qa.jsonl` | QA on financial concepts, accounting, terminology |
| **C: Financial Numeracy** | `train_C_aux_nohall.jsonl`, `train_C_strong_*.jsonl`, `train_C_syn*.jsonl` | Arithmetic and numeric transformation tasks |
| **Combined** | `train_A_B_C_large.jsonl`, `train_A_B_C_new.jsonl` | All categories combined |

### Evaluation Data

| Type | Files | Description |
|------|-------|-------------|
| Summarization | `eval_summary_80_with_10k.jsonl`, `evaluation_samples_30.jsonl` | Open-ended summarization evaluation |
| Language | `eval_language_30_A_from.jsonl`, `eval_language_100_from.jsonl` | Structured extraction evaluation |

## Scripts

### Training

| Script | Description |
|--------|-------------|
| `train_qlora_min.py` | QLoRA fine-tuning (Base→FT-A, FT-A→FT-ABC, or FT-C) |

### Generation

| Script | Description |
|--------|-------------|
| `gen_eval_summary_80.py` | Generate summarization outputs |
| `gen_finance_language.py` | Generate structured extractions |
| `gen_finance_numeracy.py` | Generate numeric predictions |
| `gen_nohall_numeracy.py` | Generate constrained numeric outputs |

### Inference

| Script | Description |
|--------|-------------|
| `infer_with_adapter.py` | Load base model with LoRA adapter |
| `infer_with_adapter_without_ins.py` | Inference without instruction prefix |

### Evaluation

| Script | Description |
|--------|-------------|
| `eval_hallucination_summary.py` | Detect unsupported numeric outputs (L1/L2/L3) |
| `eval_hallucination_summary_new.py` | Updated hallucination detection |
| `eval_language_batch.py` | Batch evaluation for extraction tasks |
| `eval_numeracy.py` | Numeric prediction correctness |
| `make_preds_jsonl.py` | Convert logs to structured JSONL |

## Results

### Output Files

| Type | Examples |
|------|----------|
| Summarization | `base_language_summary80.txt`, `ft_language_summary80_ftA.txt`, `ft_language_summary80_ftABC.txt` |
| Metrics | `base_table1_30_eval_metrics.json`, `fta_table1_30_eval_metrics.json`, `ftabc_table1_30_eval_metrics.json` |
| Numeracy | `numeracy_base_preds_new.jsonl`, `numeracy_ftc_preds_new.jsonl` |
| Logs | `base_A_eval.txt`, `ft_A_eval.txt`, `ft_language_ABC.txt` |

## Quick Start

### Step 1: Train Model

```bash
cd scripts
python train_qlora_min.py \
    --train_file ../data/train_A_large.jsonl \
    --output_dir ../outputs/ft_A
```

### Step 2: Generate Outputs

```bash
python gen_eval_summary_80.py \
    --model_path ../outputs/ft_A \
    --eval_file ../data/eval_summary_80_with_10k.jsonl \
    --output_file ../results/ft_language_summary80_ftA.txt
```

### Step 3: Evaluate Hallucination

```bash
python eval_hallucination_summary.py \
    --pred_file ../results/ft_language_summary80_ftA.txt \
    --input_file ../data/eval_summary_80_with_10k.jsonl
```

## Environment Setup

### Requirements

- Python 3.10+
- PyTorch
- HuggingFace Transformers
- PEFT
- bitsandbytes (for QLoRA)

### Installation

```bash
pip install -r requirements.txt
```

## Hardware Requirements

| Component | Specification |
|-----------|---------------|
| GPU | Single RTX A3000 (12GB) or equivalent |
| Quantization | 4-bit (QLoRA) |

Training is feasible on consumer-grade GPUs with 4-bit quantization.

## Citation

If you find this work useful, please cite:

```bibtex
@article{author2026financial,
  title={When Financial Fine-tuning Fails: A Three-Level Detectability Analysis of Numerical Hallucination in Domain-Adapted Language Models},
  author={[Your Name]},
  journal={Expert Systems with Applications},
  year={2026},
  note={Under Review}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open a [GitHub Issue](../../issues).

---

**Note**: Deterministic decoding (`temperature=0.0`) is recommended for reproducible evaluation results.