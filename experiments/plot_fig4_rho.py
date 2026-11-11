# -*- coding: utf-8 -*-
"""Figure 4 (fig:rho): improvement rate vs bottleneck sharpness (synthetic).

Data source : results/verify_q4.csv (rho_bin groups, plain vs bottleneck)
Output      : paper/figures/fig_rho.pdf

Group means are computed from the CSV rows (26 / 4 / 0 instances per bin).
The empty bin (rho >= 0.15) is drawn with a "no samples" note.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "verify_q4.csv")
OUT = os.path.join(ROOT, "paper", "figures", "fig_rho.pdf")

# Publication style: Times-compatible fonts and a restrained gray palette
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.linewidth"] = 0.6
plt.rcParams["axes.edgecolor"] = "#4d4d4d"
plt.rcParams["xtick.color"] = "#4d4d4d"
plt.rcParams["ytick.color"] = "#4d4d4d"

BINS = ["rho<0.05", "0.05<=rho<0.15", "rho>=0.15"]
BIN_LABELS = [r"$\rho<0.05$", r"$0.05\leq\rho<0.15$", r"$\rho\geq0.15$"]
SEARCHERS = ["plain", "bottleneck"]


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
    means = {}
    counts = {}
    for b in BINS:
        sub = [r for r in rows if r["rho_bin"] == b]
        counts[b] = len(sub)
        for se in SEARCHERS:
            col = f"improvement_{se}_pct"
            vals = [float(r[col]) for r in sub]
            means[(b, se)] = sum(vals) / len(vals) if vals else 0.0

    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    x = list(range(len(BINS)))
    w = 0.36
    colors = {"plain": "#e8e8e8", "bottleneck": "#404040"}
    hatches = {"plain": "", "bottleneck": "///"}
    EDGE = "#2b2b2b"
    for bi, b in enumerate(BINS):
        for si, se in enumerate(SEARCHERS):
            v = means[(b, se)]
            label = f"{v:.2f}" if counts[b] else ""
            ax.bar(bi + (si - 0.5) * w, v, width=w,
                   color=colors[se], edgecolor=EDGE, hatch=hatches[se],
                   linewidth=0.7, label=se if bi == 0 else None)
            if counts[b]:
                ax.text(bi + (si - 0.5) * w, v + 0.28, label,
                        ha="center", va="bottom", fontsize=8)
    # instance counts above each bin
    for bi, b in enumerate(BINS):
        if counts[b]:
            note = f"$n={counts[b]}$"
            ax.text(bi, ax.get_ylim()[1], note, ha="center", va="top",
                    fontsize=8.5)
        else:
            ax.text(bi, 12.4, "no samples", ha="center", va="bottom",
                    fontsize=8, style="italic", color="#808080")

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, fontsize=9)
    ax.set_ylabel("Improvement (%)", fontsize=9)
    ax.set_ylim(0, 15.5)
    ax.tick_params(labelsize=8, length=2.5)
    ax.grid(axis="y", linestyle=(0, (1, 3)), linewidth=0.5,
            color="#c8c8c8", alpha=0.8)

    handles = [
        Patch(facecolor=colors["plain"], edgecolor=EDGE, label="plain"),
        Patch(facecolor=colors["bottleneck"], edgecolor=EDGE,
              hatch="///", label="bottleneck"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper left",
              frameon=False, borderaxespad=0.2, handlelength=1.6,
              handleheight=0.9)

    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")
    print("bin means:", {k: round(v, 2) for k, v in means.items()})
    print("bin counts:", counts)


if __name__ == "__main__":
    main()
