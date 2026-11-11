# -*- coding: utf-8 -*-
"""Figure 3 (fig:traj): budget-quality trajectory at group-mean level.

real_main.csv stores only the two budget points (5000 / 10000) per cell;
per-evaluation history is not saved in the CSV, so the figure shows the
two-budget mean trajectory (documented honestly in the caption; the
per-evaluation curves are available in the online supplement).

Data source : results/real_main.csv (group means of gap_final per
              (size, init, searcher, budget))
Output      : paper/figures/fig_traj.pdf
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "real_main.csv")
OUT = os.path.join(ROOT, "paper", "figures", "fig_traj.pdf")

# Publication style: Times-compatible fonts and a restrained gray palette
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.linewidth"] = 0.6
plt.rcParams["axes.edgecolor"] = "#4d4d4d"
plt.rcParams["xtick.color"] = "#4d4d4d"
plt.rcParams["ytick.color"] = "#4d4d4d"

SIZES = ["512x16", "1024x32", "2048x64"]
BUDGETS = ["5000", "10000"]
INITS = ["min_min", "round_robin"]
SEARCHERS = ["plain", "bottleneck"]

# (size, searcher) -> (linestyle, marker); plain = dashed gray,
# bottleneck = solid dark (stronger visual weight for the proposed method)
STYLES = {
    ("512x16", "plain"):      dict(ls=(0, (4, 2)), marker="o", color="#6e6e6e"),
    ("1024x32", "plain"):     dict(ls=(0, (4, 2)), marker="s", color="#6e6e6e"),
    ("2048x64", "plain"):     dict(ls=(0, (4, 2)), marker="^", color="#6e6e6e"),
    ("512x16", "bottleneck"): dict(ls="-",  marker="o", color="#1a1a1a"),
    ("1024x32", "bottleneck"):dict(ls="-",  marker="s", color="#1a1a1a"),
    ("2048x64", "bottleneck"):dict(ls="-",  marker="^", color="#1a1a1a"),
}


def size_of(r):
    return f"{r['n']}x{r['m']}"


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
    means = {}
    for size in SIZES:
        for init in INITS:
            for se in SEARCHERS:
                for b in BUDGETS:
                    vals = [
                        float(r["gap_final_pct"])
                        for r in rows
                        if size_of(r) == size
                        and r["init_method"] == init
                        and r["searcher"] == se
                        and r["max_evals"] == b
                    ]
                    means[(size, init, se, b)] = sum(vals) / len(vals)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharex=True)
    xs = [0, 1]
    for ai, init in enumerate(INITS):
        ax = axes[ai]
        for size in SIZES:
            for se in SEARCHERS:
                ys = [means[(size, init, se, b)] for b in BUDGETS]
                lw = 1.6 if se == "bottleneck" else 1.1
                ax.plot(xs, ys, **STYLES[(size, se)], markersize=4.2,
                        linewidth=lw, markeredgewidth=0.6,
                        markeredgecolor="white", zorder=3)
        ax.set_yscale("log")
        ax.set_title({"min_min": "Min-Min",
                      "round_robin": "round-robin"}[init], fontsize=9.5,
                     pad=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(["5000", "10000"], fontsize=8.5)
        ax.set_xlabel("Evaluation budget", fontsize=9)
        ax.set_xlim(-0.12, 1.12)
        ax.tick_params(labelsize=8, length=2.5)
        ax.grid(axis="y", which="major", linestyle=(0, (1, 3)),
                linewidth=0.5, color="#c8c8c8", alpha=0.8)
    axes[0].set_ylabel(r"Final gap (\%, log scale)", fontsize=9)

    legend_handles = []
    for size in SIZES:
        for se in SEARCHERS:
            lw = 1.6 if se == "bottleneck" else 1.1
            line, = axes[0].plot([], [], **STYLES[(size, se)], linewidth=lw)
            legend_handles.append((line, f"{se}, {size}"))
    axes[1].legend([h for h, _ in legend_handles],
                   [l for _, l in legend_handles],
                   fontsize=7, ncol=2, loc="lower left", frameon=False,
                   borderaxespad=0.2, handlelength=1.6, columnspacing=0.8)

    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
