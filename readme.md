# When Financial Fine-tuning Fails
## A Three-Level Detectability Analysis of Numerical Hallucination in Domain-Adapted Language Models


---

## Overview

This repository contains the complete implementation for our study of numerical hallucination in financial summarization under domain-specific fine-tuning. We introduce a **three-level detectability taxonomy** (L1/L2/L3) and identify **template injection** as a primary hallucination mechanism.

### Key Findings

| Finding | Result |
|---------|--------|
| Base model L1 hallucination | 6.2% (near-zero restraint) |
| FT-A L1 hallucination | 82.9% (domain fine-tuning degrades restraint) |
| FT-A+B+C L1 hallucination | 90.8% (numeracy supervision amplifies the problem) |
| Template injection (FT-A) | "operating cash flow of USD" in 97% of outputs |
| Template injection (FT-A+B+C) | "USD 1011 million" in 50% of outputs |

---

## Repository Structure

```
mistral_tuning/
├── scripts/
│   ├── gen_eval_summary_160.py        # Generate 160 synthetic eval samples
│   ├── merge_synthetic_160.py         # Merge synthetic S0/S1 splits
│   ├── build_realworld_80.py          # Extract 80 real-world 10-K excerpts
│   └── build_eval_all_240.py          # Assemble final 240-sample dataset
│   └── train_qlora_min.py              # QLoRA fine-tuning (FT-A, FT-A+B+C, FT-C)
│   └── infer_summarization.py         # Run summarization inference on eval set
│   ├── annotate_hallucination.py      # L1/L2/L3 annotation pipeline
│   ├── l3_detection.py                # L3 semantic pattern matching + grounding verification
│   └── template_injection_analysis.py # Template frequency analysis
│   └── generate_figures.py          # generate 5 figure diagrams (300 DPI)
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

| Model | Training Data | Description |
|-------|---------------|-------------|
| Base | — | Original instruction-tuned model, no domain adaptation |
| FT-A | Category A (~1,000 samples) | Financial domain language fine-tuning |
| FT-A+B+C | A + B + C (~3,500 samples) | Full fine-tuning with numeracy supervision |
| FT-C | Category C (~2,000 samples) | Numeracy-only (used for competence baseline) |

**Training hardware:** Single NVIDIA RTX A3000 (12 GB)  
**Training duration:** Several hours per run (single epoch, QLoRA 4-bit NF4)

---

## Hallucination Taxonomy

| Level | Type | Definition | Detection |
|-------|------|------------|-----------|
| L1 | Overt | Fabricated values in currency-denominated form (e.g., "USD 1.3 billion") | Regex: currency symbol + number |
| L2 | Covert-Explicit | Fabricated absolute magnitudes without currency marker (e.g., "3.3 billion") | Number extraction + source grounding check |
| L3 | Covert-Implicit | Ungrounded quantitative state claims (e.g., "revenue increased") | Semantic pattern matching + grounding verification |

**Note:** L2 detection targets absolute magnitude numbers only. Percentage-based quantities (e.g., "growth of 11%") are classified as L3 when ungrounded.

---

## Reproduction

### 1. Environment Setup

```bash
pip install torch transformers peft datasets accelerate bitsandbytes
pip install scipy numpy pandas matplotlib
```

### 2. Data Construction

```bash
# Generate synthetic evaluation data
python scripts/gen_eval_summary_160.py

# Build real-world 10-K excerpts (requires EDGAR access)
python scripts/build_realworld_80.py --index data/realworld_index.csv

# Assemble full 240-sample dataset
python scripts/build_eval_all_240.py
```

### 3. Fine-tuning

```bash
# Fine-tune FT-A (financial language only)
python scripts/train_qlora_min.py --variant FT-A --data_category A

# Fine-tune FT-A+B+C (full mixture)
python scripts/train_qlora_min.py --variant FT-A+B+C --data_category A+B+C
```

### 4. Inference

```bash
python scripts/infer_summarization.py \
  --model_path ./checkpoints/FT-A \
  --eval_data ./data/eval_240.jsonl \
  --output_dir ./outputs/FT-A/
```

### 5. Hallucination Annotation

```bash
python scripts/annotate_hallucination.py \
  --predictions ./outputs/FT-A/ \
  --sources ./data/eval_240.jsonl \
  --output ./results/FT-A_hallucination.csv
```

### 6. Template Injection Analysis

```bash
python scripts/template_injection_analysis.py \
  --results ./results/ \
  --models Base FT-A FT-A+B+C
```

---

## L3 Detection Protocol

The complete L3 detection protocol (regex patterns + grounding verification) is implemented in `evaluation/l3_detection.py`. Key patterns:

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

A candidate match is confirmed as hallucination only if the core state verb/adjective is absent from the source input (case-insensitive exact match). See Appendix D of the paper for full details.

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

Full results with 95% Wilson confidence intervals and Fisher's exact tests are reported in the paper. Key table:

| Model | L1 Overt | L2 Covert-Explicit | L3 Covert-Implicit |
|-------|----------|--------------------|--------------------|
| Base | 6.2% [3.8–10.1%] | 3.8% [2.0–7.0%] | 17.9%* [13.6–23.3%] |
| FT-A | 82.9% [77.6–87.2%] | 34.6% [28.9–40.8%] | 3.8% [2.0–7.0%] |
| FT-A+B+C | 90.8% [86.5–93.9%] | 46.2% [39.6–52.2%] | 13.3% [9.6–18.2%] |

*Base L3 rate is inflated by faithful paraphrase detection artefacts; actual ungrounded L3 ≈ 0%. See paper Appendix D.5.

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
