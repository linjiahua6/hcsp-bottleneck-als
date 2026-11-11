# -*- coding: utf-8 -*-
"""Generate fig_paramheat.pdf (parameter-sensitivity heatmap) and print the
exact group means for the new equal-time table (tab:eqtime) and the sign-test
table (tab:signif). All numbers computed from the CSV files, nothing altered.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAM_CSV = os.path.join(ROOT, "results", "real_param_sensitivity.csv")
EQTIME_CSV = os.path.join(ROOT, "results", "real_equal_time.csv")
OUT = os.path.join(ROOT, "paper", "figures", "fig_paramheat.pdf")

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.linewidth"] = 0.6
plt.rcParams["axes.edgecolor"] = "#4d4d4d"
plt.rcParams["xtick.color"] = "#4d4d4d"
plt.rcParams["ytick.color"] = "#4d4d4d"

# ---------------------------------------------------------------- heatmap
rows = list(csv.DictReader(open(PARAM_CSV, encoding="utf-8-sig")))
KS = [4, 8, 16]
RS = [2, 4]
SS = [4, 8]

def mean_impr(init, K, R, S):
    sub = [r for r in rows
           if r["init_method"] == init
           and int(r["K"]) == K and int(r["R"]) == R and int(r["S"]) == S]
    vals = [float(r["improvement_pct"]) for r in sub]
    return sum(vals) / len(vals) if vals else float("nan")

fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.5),
                         gridspec_kw={"width_ratios": [1, 1.06]},
                         constrained_layout=True)
initials = ["min_min", "round_robin"]
titles = {"min_min": "Min-Min initials", "round_robin": "round-robin initials"}

for ax, init in zip(axes, initials):
    M = np.array([[mean_impr(init, K, R, S) for K in KS] for (R, S) in
                  [(2, 4), (2, 8), (4, 4), (4, 8)]])
    vmin, vmax = np.nanmin(M), np.nanmax(M)
    im = ax.imshow(M, cmap="Greys", vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(4):
        for j in range(3):
            v = M[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=7.5,
                    color="white" if v > (vmin + vmax) / 2 else "black")
            # thin light separators between cells
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="white",
                                       linewidth=1.2))
    # highlight the default (K,R,S)=(4,2,4) cell: top-left cell
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, fill=False,
                               edgecolor="#1a1a1a", linewidth=2.0))
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"$K={k}$" for k in KS], fontsize=9)
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"$(R,S)=({r},{s})$" for (r, s) in
                        [(2, 4), (2, 8), (4, 4), (4, 8)]], fontsize=8.5)
    ax.set_title(titles[init], fontsize=9.5, pad=4)
    ax.tick_params(length=0)
    ax.grid(False)

cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
cbar.ax.tick_params(labelsize=7.5, length=2)
lo = int(np.floor(vmin)); hi = int(np.ceil(vmax))
cbar.set_ticks([lo, round((lo + hi) / 2), hi])
cbar.set_label("Improvement (%)", fontsize=8.5)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)
print("wrote", OUT)
print("heat values round_robin:", [[f"{mean_impr('round_robin', k, r, s):.2f}" for k in KS] for (r, s) in [(2, 4), (2, 8), (4, 4), (4, 8)]])
print("heat values min_min    :", [[f"{mean_impr('min_min', k, r, s):.2f}" for k in KS] for (r, s) in [(2, 4), (2, 8), (4, 4), (4, 8)]])

# ------------------------------------------------------- equal-time table
eq = list(csv.DictReader(open(EQTIME_CSV, encoding="utf-8-sig")))
print("\n=== equal-time group means (c_final, gap_final, evals_actual) ===")
for size in ["512x16", "1024x32"]:
    for init in ["min_min", "round_robin"]:
        for se in ["plain", "bottleneck"]:
            line = []
            for tb in ["1.0", "3.0", "5.0"]:
                sub = [r for r in eq
                       if r["filepath"].split("\\")[0].endswith(size)
                       and r["init_method"] == init
                       and r["searcher"] == se
                       and abs(float(r["time_budget_sec"]) - float(tb)) < 1e-9]
                if not sub:
                    line.append("NA")
                    continue
                c = sum(float(r["c_final"]) for r in sub) / len(sub)
                g = sum(float(r["gap_final_pct"]) for r in sub) / len(sub)
                e = sum(float(r["evals_actual"]) for r in sub) / len(sub)
                line.append(f"{c:,.0f} / {g:.2f}% / {e:,.0f}")
            print(f"{size:9s} {init:11s} {se:10s} : " + " | ".join(line))
