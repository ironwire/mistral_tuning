# When Financial Fine-tuning Fails
## A Three-Level Detectability Analysis of Numerical Hallucination in Domain-Adapted Language Models

[![Paper](https://img.shields.io/badge/Paper-Neurocomputing%20(Under%20Review)-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-preprint-red)]()
[![ORCID](https://img.shields.io/badge/ORCID-0009--0004--8591--5556-green)](https://orcid.org/0009-0004-8591-5556)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Xiaodong Li¹², Peiwei Liu¹**

¹ School of Computer Science, Guangzhou University of Applied Science and Technology  
² IT Department, COFCO Corporation, Beijing

---

## Overview

This repository contains the complete implementation for our study of numerical hallucination in financial summarization under domain-specific fine-tuning. We introduce a **three-level detectability taxonomy** (L1/L2/L3), identify **template injection** as a primary hallucination mechanism, and conduct a **volume-matched ablation experiment** that isolates the effect of training data type from data quantity.

### Key Findings

| Finding | Result |
|---------|--------|
| Base model L1 hallucination | 5.4% (near-zero — numerical restraint maintained) |
| FT-A L1 hallucination | 82.5% (domain fine-tuning substantially degrades restraint) |
| FT-A×3.5 L1 hallucination | 64.6% (volume effect: −17.9pp — data quantity is not the driver) |
| FT-A+B+C L1 hallucination | 90.8% (content effect: +26.2pp — numeracy supervision amplifies hallucination) |
| Template injection (FT-A) | "operating cash flow of USD" in 97% of outputs |
| Template injection (FT-A+B+C) | "USD 1011 million" in 50% of outputs |
| Inter-annotator agreement (L1) | Cohen's κ = 0.883 (Almost Perfect) |
| Inter-annotator agreement (L2) | Cohen's κ = 0.657 (Substantial) |
| Inter-annotator agreement (L3) | Cohen's κ = 0.451 (Moderate) |

---

## Repository Structure

```
mistral_tuning/
├── data/
│   ├── gen_finance_language.py        # Generate Category A training data
│   ├── build_realworld_80.py          # Extract 80 real-world 10-K excerpts
│   └── build_eval_all_240.py          # Assemble final 240-sample eval dataset
├── scripts/
│   ├── train_qlora_min.py             # QLoRA fine-tuning (FT-A, FT-A×3.5, FT-A+B+C, FT-C)
│   ├── infer_summarization.py         # Batch summarization inference
│   ├── annotate_hallucination.py      # L1/L2/L3 annotation pipeline (Base/FT-A/FT-ABC)
│   ├── annotate_ablation.py           # L1/L2/L3 annotation for FT-A×3.5 ablation model
│   ├── l3_detection.py                # L3 semantic pattern matching + grounding verification
│   ├── template_injection_analysis.py # Template frequency analysis
│   ├── select_validation_samples.py   # Stratified sample selection for human validation
│   ├── gen_annotation_excel.py        # Generate bilingual annotation Excel for human validators
│   ├── compute_kappa.py               # Cohen's κ calculation and FP/FN analysis
│   ├── compute_mcnemar.py             # McNemar's paired test for model comparisons
│   └── generate_figures.py            # Generate all figures at 300 DPI
└── README.md
```

---

## Dataset

The evaluation dataset comprises **240 samples** balanced across grounding condition and data source:

| Category | S0 (Weak) | S1 (Strong) | Total |
|----------|-----------|-------------|-------|
| Synthetic | 80 | 80 | 160 |
| Real-world (SEC EDGAR 10-K) | 40 | 40 | 80 |
| **Total** | **120** | **120** | **240** |

**Real-world samples** are extracted from 10-K filings on [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar). Due to licensing constraints, raw documents are not redistributed. All excerpts are identified by CIK in `data/realworld_index.xlsx` for independent replication.

**Grounding conditions:**
- **S0 (Weak):** Input contains no absolute numerical quantities (no monetary values or counts)
- **S1 (Strong):** Input contains at least one explicit absolute numerical value

---

## Models

All experiments use **Mistral-7B-Instruct-v0.2** with **QLoRA** fine-tuning.

| Model | Training Data | Volume | Description |
|-------|---------------|--------|-------------|
| Base | — | — | Original instruction-tuned model, no domain adaptation |
| FT-A | Category A | ~1,000 samples | Financial domain language fine-tuning |
| FT-A×3.5 | Category A | ~3,500 samples | Volume-matched ablation: same content as FT-A, same volume as FT-A+B+C |
| FT-A+B+C | Categories A+B+C | ~3,500 samples | Full fine-tuning with numeracy supervision |
| FT-C | Category C | ~2,000 samples | Numeracy-only (used for competence baseline) |

**Training hardware:** Single NVIDIA RTX A3000 (12 GB)  
**Training duration:** 3–5 hours per run (single epoch, QLoRA 4-bit NF4)

---

## Hallucination Taxonomy

| Level | Type | Definition | Detection |
|-------|------|------------|-----------|
| L1 | Overt | Fabricated values in currency-denominated form (e.g., "USD 1.3 billion") | Regex: currency symbol + number |
| L2 | Covert-Explicit | Fabricated absolute magnitudes without currency marker (e.g., "3.3 billion", "1011 million") | Number extraction + source grounding check |
| L3 | Covert-Implicit | Ungrounded quantitative state claims (e.g., "revenue increased") | Semantic pattern matching + grounding verification |

**Note:** L2 detection targets absolute magnitude numbers only. Percentage-based quantities (e.g., "growth of 11%") are classified as L3 when ungrounded, as they represent relative rather than absolute numerical claims.

---

## Three-Way Failure Mode Taxonomy

Our experiments reveal three distinct failure modes in fine-tuned financial summarisation models:

| Failure Mode | Model | Mechanism |
|-------------|-------|-----------|
| **Template Injection** | FT-A, FT-A+B+C | Memorised canonical values inserted regardless of input content, producing fluent but fabricated numerical statements |
| **Repetition-Induced Omission** | FT-A×3.5 | Syntactically uniform training data causes repetition until the output budget is exhausted, omitting input content. Low apparent hallucination reflects omission, not restraint |
| **Format-Matching Artefact** | Base (S1_Real) | Faithful reproduction of input values using currency expressions triggers automatic L1 detection despite being fully grounded |

---

## Volume-Matched Ablation

To isolate the effect of training data type from data quantity, we trained FT-A×3.5 using 3,500 Category A samples — identical in content type to FT-A but matched in volume to FT-A+B+C.

| Model | Data | Volume | L1 | L2 | L3 |
|-------|------|--------|----|----|-----|
| FT-A | Category A | 1,000 | 82.5% | 34.6% | 3.8% |
| FT-A×3.5 | Category A | 3,500 | 64.6% | 16.2% | 9.6% |
| FT-A+B+C | Categories A+B+C | 3,500 | 90.8% | 45.8% | 13.3% |

**Volume effect** (FT-A → FT-A×3.5): L1 **−17.9pp** — additional Category A training *reduces* hallucination  
**Content effect** (FT-A×3.5 → FT-A+B+C): L1 **+26.2pp** — adding numeracy content sharply increases hallucination

The opposing directions confirm that **data type, not data volume, is the primary driver** of hallucination amplification.

---

## Reproduction

### 1. Environment Setup

```bash
pip install torch transformers peft datasets accelerate bitsandbytes
pip install scipy numpy pandas matplotlib openpyxl
```

### 2. Data Construction

```bash
# Generate Category A training data (1000 samples for FT-A)
python data/gen_finance_language.py --n 1000 --output data/train_A_1000.jsonl

# Generate Category A training data (3500 samples for FT-A×3.5 ablation)
python data/gen_finance_language.py --n 3500 --output data/train_A_3500.jsonl

# Build real-world 10-K excerpts (requires EDGAR access)
python data/build_realworld_80.py --index data/realworld_index.xlsx

# Assemble full 240-sample evaluation dataset
python data/build_eval_all_240.py --output data/eval_all_240.jsonl
```

### 3. Fine-tuning

```bash
# Fine-tune FT-A (financial language, 1000 samples)
CUDA_VISIBLE_DEVICES=0 python scripts/train_qlora_min.py \
  --model /path/to/Mistral-7B-Instruct-v0.2 \
  --train_jsonl data/train_A_1000.jsonl \
  --out_dir outputs/ft_A \
  --epochs 1 --lr 2e-4 --lora_r 8 --seed 42

# Fine-tune FT-A×3.5 (volume-matched ablation, 3500 Category A samples)
CUDA_VISIBLE_DEVICES=0 python scripts/train_qlora_min.py \
  --model /path/to/Mistral-7B-Instruct-v0.2 \
  --train_jsonl data/train_A_3500.jsonl \
  --out_dir outputs/ft_A_3500 \
  --epochs 1 --lr 2e-4 --lora_r 8 --seed 42

# Fine-tune FT-A+B+C (full mixture, 3500 samples)
CUDA_VISIBLE_DEVICES=0 python scripts/train_qlora_min.py \
  --model /path/to/Mistral-7B-Instruct-v0.2 \
  --train_jsonl data/train_ABC_3500.jsonl \
  --out_dir outputs/ft_A_B_C \
  --epochs 1 --lr 2e-4 --lora_r 8 --seed 42
```

### 4. Inference

```bash
# Base model (no adapter)
CUDA_VISIBLE_DEVICES=0 python scripts/infer_summarization.py \
  --model /path/to/Mistral-7B-Instruct-v0.2 \
  --eval data/eval_all_240.jsonl \
  --output results/results_base_240.txt

# FT-A
CUDA_VISIBLE_DEVICES=0 python scripts/infer_summarization.py \
  --model /path/to/Mistral-7B-Instruct-v0.2 \
  --adapter outputs/ft_A \
  --eval data/eval_all_240.jsonl \
  --output results/results_ftA_240.txt

# FT-A×3.5 (ablation)
CUDA_VISIBLE_DEVICES=0 python scripts/infer_summarization.py \
  --model /path/to/Mistral-7B-Instruct-v0.2 \
  --adapter outputs/ft_A_3500 \
  --eval data/eval_all_240.jsonl \
  --output results/results_ftA3500_240.txt

# FT-A+B+C
CUDA_VISIBLE_DEVICES=0 python scripts/infer_summarization.py \
  --model /path/to/Mistral-7B-Instruct-v0.2 \
  --adapter outputs/ft_A_B_C \
  --eval data/eval_all_240.jsonl \
  --output results/results_ftABC_240.txt
```

### 5. Hallucination Annotation

```bash
# Annotate Base / FT-A / FT-A+B+C
python scripts/annotate_hallucination.py \
  --base   results/results_base_240.txt \
  --fta    results/results_ftA_240.txt \
  --ftabc  results/results_ftABC_240.txt \
  --eval   data/eval_all_240.jsonl \
  --out    results/annotations_240.jsonl \
  --report results/hallucination_report.txt

# Annotate FT-A×3.5 ablation model (with comparison to original three models)
python scripts/annotate_ablation.py \
  --fta35  results/results_ftA3500_240.txt \
  --eval   data/eval_all_240.jsonl \
  --orig   results/annotations_240.jsonl \
  --out    results/annotations_ablation.jsonl \
  --report results/ablation_report.txt
```

### 6. Statistical Tests

```bash
# McNemar's paired test for all pairwise model comparisons
python scripts/compute_mcnemar.py \
  --annotations results/annotations_240.jsonl \
  --out         results/mcnemar_report.txt
```

### 7. Human Validation

```bash
# Step 1: Select 50 stratified samples for human annotation
python scripts/select_validation_samples.py \
  --annotations  results/annotations_240.jsonl \
  --ablation     results/annotations_ablation.jsonl \
  --eval         data/eval_all_240.jsonl \
  --base_txt     results/results_base_240.txt \
  --fta_txt      results/results_ftA_240.txt \
  --fta35_txt    results/results_ftA3500_240.txt \
  --ftabc_txt    results/results_ftABC_240.txt \
  --out          results/validation_50_samples.jsonl

# Step 2: Generate annotation Excel forms (bilingual CN/EN)
python scripts/gen_annotation_excel.py \
  --samples results/validation_50_samples.jsonl \
  --annotator A \
  --out results/annotation_form_annotatorA.xlsx

python scripts/gen_annotation_excel.py \
  --samples results/validation_50_samples.jsonl \
  --annotator B \
  --out results/annotation_form_annotatorB.xlsx

# Step 3: Compute Cohen's kappa after annotation is complete
python scripts/compute_kappa.py \
  --annotator_a results/annotation_form_annotatorA.xlsx \
  --annotator_b results/annotation_form_annotatorB.xlsx \
  --out         results/kappa_report.txt
```

### 8. Template Injection Analysis

```bash
python scripts/template_injection_analysis.py \
  --results ./results/ \
  --models Base FT-A FT-A+B+C
```

---

## L3 Detection Protocol

The complete L3 detection protocol is implemented in `scripts/l3_detection.py`. Key patterns:

```python
L3_PATTERNS = [
    r"revenue (increased|decreased|grew|declined|rose|fell|surged|dropped)",
    r"(operating income|net income|earnings|profit) (increased|decreased|grew|declined|rose|fell|improved)",
    r"maintained stable",
    r"remained stable",
    r"net debt (remained|declined|increased|decreased|improved)",
    r"cash flow (improved|declined|increased|decreased|remained)",
    r"margin (improved|compressed|expanded|declined|increased|decreased)",
    r"capital expenditures (increased|decreased|rose|fell|remained)",
    r"strong (growth|performance|results|revenue|earnings)",
    r"significant (increase|decrease|growth|decline|improvement)",
    r"substantial (growth|increase|decrease|decline|improvement)",
    r"(outperformed|underperformed|exceeded|missed)",
]
```

A candidate match is confirmed as hallucination only if the core state verb/adjective is **absent** from the source input (case-insensitive exact match). See Appendix D of the paper for full details.

---

## Grounding Condition Classifier

Reference implementation (Algorithm 1 in the paper):

```python
import re

def classify_grounding(text: str) -> str:
    """Classify input as S0 (weak) or S1 (strong) grounding."""
    pattern = r'\$[\d,]+|\d+\s*(million|billion|USD)'
    if re.search(pattern, text, re.IGNORECASE):
        return 'S1'  # Strong grounding
    return 'S0'      # Weak grounding
```

---

## Results Summary

Full results with 95% Wilson confidence intervals, Fisher's exact tests, and McNemar's paired tests are reported in the paper.

### Overall Hallucination Rates

| Model | L1 Overt | L2 Covert-Explicit | L3 Covert-Implicit |
|-------|----------|--------------------|--------------------|
| Base | 5.4% [3.1–9.2%] | 2.9% [1.4–5.9%] | 17.9%† [13.6–23.3%] |
| FT-A | 82.5% [77.1–86.9%] | 34.6% [28.9–40.8%] | 3.8% [2.0–7.0%] |
| FT-A×3.5 ‡ | 64.6% [58.3–70.4%] | 16.2% [11.9–21.7%] | 9.6% [6.4–14.0%] |
| FT-A+B+C | 90.8% [86.5–93.9%] | 45.8% [39.6–52.2%] | 13.3% [9.6–18.2%] |

† Base L3 rate is inflated by faithful paraphrase detection artefacts; actual ungrounded L3 ≈ 0%. Confirmed by human validation (consensus L3 ≈ 0%). See paper Appendix D.5.

‡ FT-A×3.5 exhibits repetitive degeneration on 100% of synthetic samples (27/27), which artificially reduces its L1 rate on synthetic inputs. See paper Section 4.7.

### Inter-Annotator Agreement (Human Validation, n=50 samples)

| Level | Cohen's κ | Interpretation |
|-------|-----------|----------------|
| L1 (Overt) | 0.883 | Almost Perfect |
| L2 (Covert-Explicit) | 0.657 | Substantial |
| L3 (Covert-Implicit) | 0.451 | Moderate |

S0-condition agreement: κ = 1.000 (perfect) for both S0_Synth and S0_Real.

### McNemar's Test Results (L1, paired, n=240)

| Comparison | Discordant Pairs | χ² | p-value |
|-----------|-----------------|-----|---------|
| Base vs. FT-A | 189 | 179.13 | < 0.001 *** |
| Base vs. FT-A+B+C | 209 | 199.12 | < 0.001 *** |
| FT-A vs. FT-A+B+C | 46 | 7.85 | 0.005 ** |

---

## Citation

```bibtex
@article{li2026financial,
  title={When Financial Fine-tuning Fails: A Three-Level Detectability Analysis of 
         Numerical Hallucination in Domain-Adapted Language Models},
  author={Li, Xiaodong and Liu, Peiwei},
  journal={Neurocomputing},
  year={2026},
  note={Under review}
}
```

---

## License

Code: MIT License. See [LICENSE](LICENSE).  
Data: Real-world 10-K excerpts are subject to SEC EDGAR terms of use and are not redistributed.

---

## Contact

**Xiaodong Li** · lixd@cofco.com · [ORCID 0009-0004-8591-5556](https://orcid.org/0009-0004-8591-5556)  
School of Computer Science, Guangzhou University of Applied Science and Technology
