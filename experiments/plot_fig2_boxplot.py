# -*- coding: utf-8 -*-
"""Figure 2 (fig:boxplot): improvement-rate boxplots per size x init x searcher.

Data source : results/real_main.csv (224 cells, seed=42)
Output      : paper/figures/fig_boxplot.pdf

Box order per size cluster:
  [plain/min_min, bottleneck/min_min, plain/round_robin, bottleneck/round_robin]
Colour by initial solution (light gray = min_min, dark gray = round_robin);
bottleneck boxes carry a hatch for black-and-white printing.
Y axis is log scale because the two initial solutions differ by ~10x.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "real_main.csv")
OUT = os.path.join(ROOT, "paper", "figures", "fig_boxplot.pdf")

# Publication style: Times-compatible fonts and a restrained gray palette
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.linewidth"] = 0.6
plt.rcParams["axes.edgecolor"] = "#4d4d4d"
plt.rcParams["xtick.color"] = "#4d4d4d"
plt.rcParams["ytick.color"] = "#4d4d4d"

SIZES = ["512x16", "1024x32", "2048x64"]
INITS = ["min_min", "round_robin"]
SEARCHERS = ["plain", "bottleneck"]


def size_of(r):
    return f"{r['n']}x{r['m']}"


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
    data = {}
    for size in SIZES:
        for init in INITS:
            for se in SEARCHERS:
                data[(size, init, se)] = [
                    float(r["improvement_pct"])
                    for r in rows
                    if size_of(r) == size
                    and r["init_method"] == init
                    and r["searcher"] == se
                ]

    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    cluster_gap = 1.0          # extra gap between size clusters
    within_gap = 0.35          # gap between the two searcher pairs
    local = [0.0, 1.0, 1.0 + within_gap + 1.0, 2.0 + within_gap + 1.0]
    width = 4.0                # span of one cluster
    step = width + cluster_gap

    FACE = {"min_min": "#e8e8e8", "round_robin": "#404040"}
    EDGE = "#2b2b2b"
    box_pos = []
    centers = []
    for si, size in enumerate(SIZES):
        base = si * step
        pos = [base + l for l in local]
        box_pos.append(pos)
        centers.append(base + (local[-1] - local[0]) / 2)
        # subtle vertical separator between size clusters
        if si < len(SIZES) - 1:
            sep_x = base + width / 2 + cluster_gap / 2
            ax.axvline(sep_x, color="#b0b0b0", linestyle=(0, (1, 3)),
                       linewidth=0.6, zorder=0)
        for pi, (init, se) in enumerate(
            [(i, s) for i in INITS for s in SEARCHERS]
        ):
            facecolor = FACE[init]
            hatch = "///" if se == "bottleneck" else ""
            bp = ax.boxplot(
                data[(size, init, se)],
                positions=[pos[pi]],
                widths=0.75,
                patch_artist=True,
                boxprops=dict(facecolor=facecolor, edgecolor=EDGE, hatch=hatch,
                              linewidth=0.8),
                medianprops=dict(color="white" if init == "round_robin"
                                 else EDGE, linewidth=1.5),
                whiskerprops=dict(color=EDGE, linewidth=0.8),
                capprops=dict(color=EDGE, linewidth=0.8),
                flierprops=dict(marker=".", markersize=3,
                                markeredgecolor=EDGE,
                                markerfacecolor=EDGE),
                showfliers=True,
            )
            for patch in bp["boxes"]:
                patch.set_linewidth(0.8)

    ax.set_yscale("log")
    ax.set_xticks(centers)
    ax.set_xticklabels(
        [r"$512\times16$", r"$1024\times32$", r"$2048\times64$"],
        fontsize=10)
    ax.set_ylabel("Improvement (%)", fontsize=9)
    ax.set_ylim(0.3, 150)
    ax.tick_params(labelsize=8, length=2.5)
    ax.grid(axis="y", which="major", linestyle=(0, (1, 3)), linewidth=0.5,
            color="#c8c8c8", alpha=0.8)

    legend_handles = [
        Patch(facecolor="#e8e8e8", edgecolor=EDGE, label="plain, Min-Min"),
        Patch(facecolor="#e8e8e8", edgecolor=EDGE, hatch="///",
              label="bottleneck, Min-Min"),
        Patch(facecolor="#404040", edgecolor=EDGE,
              label="plain, round-robin"),
        Patch(facecolor="#404040", edgecolor=EDGE, hatch="///",
              label="bottleneck, round-robin"),
    ]
    ax.legend(handles=legend_handles, fontsize=7.5, ncol=2,
              loc="upper left", frameon=False, borderaxespad=0.2,
              handlelength=1.6, handleheight=0.9)

    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
