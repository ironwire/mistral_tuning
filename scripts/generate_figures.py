#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_figures.py

Generates all five figures for:
  "When Financial Fine-tuning Fails: A Three-Level Detectability Analysis
   of Numerical Hallucination in Domain-Adapted Language Models"

Figures generated:
  Figure 1 — Hallucination rates by detectability level (grouped bar chart)
  Figure 2 — Hallucination rates heatmap by model, level, and grounding condition
  Figure 3 — Template injection frequency (FT-A vs FT-A+B+C bar charts)
  Figure 4 — Comparative outputs demonstrating template injection (text figure,
              generated as a formatted text file, not a plot)
  Figure 5 — The restraint gap: fine-tuning trajectory (scatter plot)

Usage:
    # Generate all figures
    python generate_figures.py --out figures/

    # Generate specific figures
    python generate_figures.py --out figures/ --figs 1 2 5

Requirements:
    pip install matplotlib numpy
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Data (n=240 dataset)
# ─────────────────────────────────────────────────────────────

# Figure 1 & Table 5: Overall hallucination rates
OVERALL = {
    "Base":     {"L1": 6.2,  "L2": 3.8,  "L3": 17.9},
    "FT-A":     {"L1": 82.9, "L2": 34.6, "L3": 3.8},
    "FT-A+B+C": {"L1": 90.8, "L2": 46.2, "L3": 13.3},
}

# Figure 2: Hallucination rates by model, level, grounding condition
# Columns: L1-S0, L1-S1, L2-S0, L2-S1, L3-S0, L3-S1
HEATMAP_DATA = np.array([
    [0.8,  11.7,  0.0,  7.5,  30.8,  5.0],   # Base
    [96.7, 69.2,  0.0,  69.2,  3.3,  4.2],   # FT-A
    [90.0, 91.7,  0.0,  91.7,  7.5, 19.2],   # FT-A+B+C
])
HEATMAP_ROWS = ["Base", "FT-A", "FT-A+B+C"]
HEATMAP_COLS = ["L1-S0", "L1-S1", "L2-S0", "L2-S1", "L3-S0", "L3-S1"]

# Figure 3: Template injection frequency
FTA_TEMPLATES = [
    ("operating cash flow of USD", 97),
    ("USD 3.3 billion",            37),
    ("USD 1033 million",           28),
    ("USD 138 million",            18),
    ("USD 1.3 billion",            14),
    ("USD 1038 million",           14),
    ("decline in operating income",10),
    ("growth of 11%",               9),
    ("growth of 13%",               7),
    ("maintained stable net debt",  1),
]
FTABC_TEMPLATES = [
    ("operating cash flow of USD",   80),
    ("USD 1011 million",             50),
    ("USD 138 million",              27),
    ("USD 1.3 billion",              24),
    ("growth of 11%",                20),
    ("USD 1.8 billion",              17),
    ("USD 1111 million",             12),
    ("maintained stable net debt",    7),
    ("continued investment in digital",5),
    ("USD 3.3 billion",               2),
]


# ─────────────────────────────────────────────────────────────
# Figure 1: Grouped bar chart
# ─────────────────────────────────────────────────────────────
def generate_figure1(out_dir: Path):
    models = list(OVERALL.keys())
    l1 = [OVERALL[m]["L1"] for m in models]
    l2 = [OVERALL[m]["L2"] for m in models]
    l3 = [OVERALL[m]["L3"] for m in models]

    x = np.arange(len(models))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    color_l1 = "#E05A4E"
    color_l2 = "#F5A623"
    color_l3 = "#5B9BD5"

    bars1 = ax.bar(x - width, l1, width, label="L1 (Overt)",           color=color_l1, zorder=3)
    bars2 = ax.bar(x,          l2, width, label="L2 (Covert-Explicit)", color=color_l2, zorder=3)
    bars3 = ax.bar(x + width,  l3, width, label="L3 (Covert-Implicit)", color=color_l3, zorder=3)

    def add_labels(bars, vals):
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.8,
                        f"{val}%",
                        ha="center", va="bottom", fontsize=9, color="#333")

    add_labels(bars1, l1)
    add_labels(bars2, l2)
    add_labels(bars3, l3)

    ax.set_ylabel("Hallucination Rate (%)", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_title("Hallucination Rates by Detectability Level", fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="both", labelsize=10)
    ax.yaxis.grid(True, color="#e0e0e0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#ccc")
    ax.spines["bottom"].set_color("#ccc")
    ax.legend(fontsize=10, frameon=True, framealpha=0.9,
              loc="upper left", bbox_to_anchor=(0.01, 0.99))

    plt.tight_layout()
    out_path = out_dir / "figure1_hallucination_rates.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ Figure 1 → {out_path}")


# ─────────────────────────────────────────────────────────────
# Figure 2: Heatmap
# ─────────────────────────────────────────────────────────────
def generate_figure2(out_dir: Path):
    colors_list = [
        (0.18, 0.80, 0.44),
        (0.98, 0.80, 0.08),
        (0.95, 0.29, 0.29),
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list("gryr", colors_list, N=256)

    fig, ax = plt.subplots(figsize=(9, 3.2))
    fig.patch.set_facecolor("white")

    im = ax.imshow(HEATMAP_DATA / 100.0, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(HEATMAP_COLS)))
    ax.set_xticklabels(HEATMAP_COLS, fontsize=11)
    ax.set_yticks(range(len(HEATMAP_ROWS)))
    ax.set_yticklabels(HEATMAP_ROWS, fontsize=11)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    for i in range(HEATMAP_DATA.shape[0]):
        for j in range(HEATMAP_DATA.shape[1]):
            val = HEATMAP_DATA[i, j]
            text_color = "white" if val > 45 else "#1a1a1a"
            ax.text(j, i, f"{val:.1f}%",
                    ha="center", va="center",
                    fontsize=11, fontweight="500", color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Hallucination Rate (%)", fontsize=9, labelpad=8)

    ax.set_title(
        "Numerical Hallucination Rates by Model, Detectability Level, and Grounding Condition",
        fontsize=11, fontweight="normal", pad=14
    )
    ax.set_xlabel("Detectability Level – Grounding Condition", fontsize=10, labelpad=8)
    ax.xaxis.set_label_position("bottom")

    for x in np.arange(-0.5, len(HEATMAP_COLS), 1):
        ax.axvline(x, color="white", linewidth=1.5)
    for y in np.arange(-0.5, len(HEATMAP_ROWS), 1):
        ax.axhline(y, color="white", linewidth=1.5)

    ax.set_xlim(-0.5, len(HEATMAP_COLS) - 0.5)
    ax.set_ylim(len(HEATMAP_ROWS) - 0.5, -0.5)

    plt.tight_layout()
    out_path = out_dir / "figure2_hallucination_heatmap.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ Figure 2 → {out_path}")


# ─────────────────────────────────────────────────────────────
# Figure 3: Template injection bar charts
# ─────────────────────────────────────────────────────────────
def generate_figure3(out_dir: Path):
    color_fta   = "#378ADD"
    color_ftabc = "#7F77DD"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.patch.set_facecolor("white")

    def draw_bars(ax, items, color, title):
        labels_r = [x[0] for x in items][::-1]
        vals_r   = [x[1] for x in items][::-1]
        y = np.arange(len(labels_r))

        bars = ax.barh(y, vals_r, color=color, height=0.6, zorder=2)

        for bar, v in zip(bars, vals_r):
            if v >= 10:
                ax.text(v - 1.5, bar.get_y() + bar.get_height() / 2,
                        f"{v}%", va="center", ha="right",
                        fontsize=9.5, fontweight="500", color="white", zorder=3)
            else:
                ax.text(v + 1.5, bar.get_y() + bar.get_height() / 2,
                        f"{v}%", va="center", ha="left",
                        fontsize=9.5, fontweight="500", color="#444", zorder=3)

        ax.set_yticks(y)
        ax.set_yticklabels(labels_r, fontsize=9.5)
        ax.set_xlim(0, 110)
        ax.set_xlabel("Frequency (%)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="normal", pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("white")
        ax.xaxis.grid(True, color="#e0e0e0", linewidth=0.6, zorder=1)
        ax.set_axisbelow(True)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis="x", labelsize=9)

    draw_bars(ax1, FTA_TEMPLATES,   color_fta,   "FT-A Template Injection Patterns")
    draw_bars(ax2, FTABC_TEMPLATES, color_ftabc, "FT-A+B+C Template Injection Patterns")

    plt.tight_layout(pad=2.0)
    out_path = out_dir / "figure3_template_injection.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ Figure 3 → {out_path}")


# ─────────────────────────────────────────────────────────────
# Figure 5: Restraint gap scatter plot
# ─────────────────────────────────────────────────────────────
def generate_figure5(out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Zone shading
    safe = mpatches.FancyBboxPatch(
        (0.0, 0.55), 0.55, 0.45,
        boxstyle="round,pad=0.01", linewidth=0,
        facecolor="#d4f7e0", alpha=0.7, zorder=1
    )
    ax.add_patch(safe)
    ax.text(0.05, 0.93, "SAFE ZONE", fontsize=9, color="#2e7d52",
            fontweight="bold", va="top", zorder=3)

    danger = mpatches.FancyBboxPatch(
        (0.42, 0.0), 0.58, 0.50,
        boxstyle="round,pad=0.01", linewidth=0,
        facecolor="#fde8e8", alpha=0.7, zorder=1
    )
    ax.add_patch(danger)
    ax.text(0.97, 0.48, "DANGER ZONE\n(High competence,\nno discipline)",
            fontsize=8, color="#b71c1c", fontweight="bold",
            ha="right", va="top", zorder=3)

    # Trajectory arrow
    ax.annotate("", xy=(0.83, 0.13), xytext=(0.25, 0.78),
                arrowprops=dict(
                    arrowstyle="->", color="#666", lw=1.6,
                    connectionstyle="arc3,rad=0.10"
                ), zorder=4)
    ax.text(0.50, 0.52, "fine-tuning\ndirection",
            fontsize=8, color="#777", rotation=-46,
            ha="center", va="center", zorder=4)

    # Diagonal reference
    ax.plot([0, 1], [1, 0], "--", color="#ccc", linewidth=1.0, alpha=0.7, zorder=2)

    # Model points
    ax.scatter([0.25], [0.78], s=180, color="#2ecc71",
               zorder=5, edgecolors="white", linewidths=1.5)
    ax.text(0.25, 0.85, "Base", ha="center", va="bottom",
            fontsize=10, color="#1a7a42", fontweight="bold", zorder=5)
    ax.text(0.25, 0.68, "L1: 6.2%", ha="center", va="top",
            fontsize=8.5, color="#1a7a42", zorder=5)

    ax.scatter([0.60], [0.20], s=220, color="#e74c3c",
               zorder=5, edgecolors="white", linewidths=1.5)
    ax.text(0.60, 0.27, "FT-A", ha="center", va="bottom",
            fontsize=10, color="#922b21", fontweight="bold", zorder=5)
    ax.text(0.60, 0.11, "L1: 82.9%", ha="center", va="top",
            fontsize=8.5, color="#922b21", zorder=5)

    ax.scatter([0.87], [0.10], s=260, color="#8e44ad",
               zorder=5, edgecolors="white", linewidths=1.5)
    ax.text(0.74, 0.20, "FT-A+B+C", ha="center", va="bottom",
            fontsize=10, color="#6c3483", fontweight="bold", zorder=5)
    ax.text(0.87, 0.19, "L1: 90.8%", ha="center", va="bottom",
            fontsize=8.5, color="#6c3483", zorder=5)

    # Axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Numerical Competence →", fontsize=11, labelpad=8)
    ax.set_ylabel("Numerical Discipline ↑", fontsize=11, labelpad=8)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["None\nLow", "Medium", "High"], fontsize=9)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["None\nLow", "Medium", "High"], fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#ccc")
    ax.spines["bottom"].set_color("#ccc")
    ax.tick_params(colors="#888")
    ax.set_title("The Restraint Gap: Fine-tuning Trajectory",
                 fontsize=12, fontweight="normal", pad=14)

    plt.tight_layout()
    out_path = out_dir / "figure5_restraint_gap.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ Figure 5 → {out_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Generate all paper figures (n=240 dataset)"
    )
    ap.add_argument("--out", default="figures",
                    help="Output directory (default: figures/)")
    ap.add_argument("--figs", nargs="+", type=int,
                    choices=[1, 2, 3, 5], default=[1, 2, 3, 5],
                    help="Which figures to generate (default: all)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating figures → {out_dir}/")
    generators = {
        1: generate_figure1,
        2: generate_figure2,
        3: generate_figure3,
        5: generate_figure5,
    }

    for fig_num in sorted(args.figs):
        generators[fig_num](out_dir)

    print(f"\nDone. {len(args.figs)} figure(s) saved to {out_dir}/")


if __name__ == "__main__":
    main()
