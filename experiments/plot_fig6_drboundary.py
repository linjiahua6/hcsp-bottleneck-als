# -*- coding: utf-8 -*-
"""Figure 5 (fig:drboundary): mean B-A improvement difference (pp) of the
optional destroy-repair (DR) extension, grouped by consistency class and
initial solution. Data source: results/real_q6.csv (64 paired cells).

Negative values mean DR hurts; under Min-Min initials B-A is exactly 0.00pp
in every group (DR returns to the same local optimum).
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "real_q6.csv")
OUT = os.path.join(ROOT, "paper", "figures", "fig_drboundary.pdf")

# Publication style: Times-compatible fonts and a restrained gray palette
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.linewidth"] = 0.6
plt.rcParams["axes.edgecolor"] = "#4d4d4d"
plt.rcParams["xtick.color"] = "#4d4d4d"
plt.rcParams["ytick.color"] = "#4d4d4d"

CATS = ["u_c", "u_i", "u_s"]
CAT_LABELS = [r"$u_c$", r"$u_i$", r"$u_s$"]


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
    pairs = {}
    for r in rows:
        k = (r["filepath"], r["init_method"], r["max_evals"])
        pairs.setdefault(k, {})[r["searcher"]] = r

    # B-A per (cat, init, budget)
    vals = {}
    for cat in CATS:
        for init in ["min_min", "round_robin"]:
            for b in ["5000", "10000"]:
                ba = []
                for k, d in pairs.items():
                    if k[1] == init and k[2] == b and "A" in d and "B" in d:
                        if d["A"]["category"] == cat:
                            ba.append(float(d["B"]["improvement_pct"])
                                      - float(d["A"]["improvement_pct"]))
                vals[(cat, init, b)] = (sum(ba) / len(ba)) if ba else 0.0

    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    x = list(range(len(CATS)))
    w = 0.26
    EDGE = "#2b2b2b"
    series = [
        ("min_min", "5000", "#e8e8e8", ""),
        ("min_min", "10000", "#e8e8e8", ""),
        ("round_robin", "5000", "#8c8c8c", ""),
        ("round_robin", "10000", "#404040", "///"),
    ]
    for si, (init, b, fc, hatch) in enumerate(series):
        ys = [vals[(cat, init, b)] for cat in CATS]
        ax.bar([xi + (si - 1.5) * w for xi in x], ys, width=w,
               color=fc, edgecolor=EDGE, hatch=hatch,
               linewidth=0.7)
        for xi, v in zip(x, ys):
            if v != 0.0:
                ax.text(xi + (si - 1.5) * w, v - 0.9,
                        f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}",
                        ha="center", va="top", fontsize=6.8)
            else:
                ax.text(xi + (si - 1.5) * w, 0.6, "0.0",
                        ha="center", va="bottom", fontsize=6.8)

    ax.axhline(0.0, color="#1a1a1a", linestyle=(0, (4, 2)), linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(CAT_LABELS, fontsize=9.5)
    ax.set_xlabel("Consistency class", fontsize=9)
    ax.set_ylabel(r"Mean $B-A$ (pp)", fontsize=9)
    ax.set_ylim(-42, 3)
    ax.tick_params(labelsize=8, length=2.5)
    ax.grid(axis="y", linestyle=(0, (1, 3)), linewidth=0.5,
            color="#c8c8c8", alpha=0.8, zorder=0)

    # Min-Min both budgets are 0.00: merge into one legend entry
    handles = [
        Patch(facecolor="#e8e8e8", edgecolor=EDGE,
              label="Min-Min (both budgets: 0.00)"),
        Patch(facecolor="#8c8c8c", edgecolor=EDGE, label="round-robin, 5000"),
        Patch(facecolor="#404040", edgecolor=EDGE, hatch="///",
              label="round-robin, 10000"),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="lower left",
              frameon=False, borderaxespad=0.2, handlelength=1.6,
              handleheight=0.9)

    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)
    print("series values:", {k: round(v, 2) for k, v in vals.items()})


if __name__ == "__main__":
    main()
