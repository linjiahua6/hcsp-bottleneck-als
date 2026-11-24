# -*- coding: utf-8 -*-
"""Poster-specific figures, drawn at the exact on-poster render size.

Each PDF is produced 1:1 (millimetres) so the font sizes set here are
what the audience sees on the A0 sheet: ticks 15-18 pt, labels 17-18 pt,
readable from 1-2 m. The original paper figures are never touched.

Outputs (into _poster_template/figures/, *_poster names so the current
poster build stays intact until the layout is switched):
  motivation_poster.pdf   275 x 136 mm  improvement rate vs rho (verify_q4.csv)
  results_poster.pdf      500 x 116 mm  improvement boxplots + final-gap
                                        trajectories (real_main.csv)
  qualitative_poster.pdf  270 x 143 mm  DR boundary B-A (real_q6.csv)
  paramheat_poster.pdf    194 x 86 mm   parameter sensitivity heatmap
                                        (real_param_sensitivity.csv)

Palette (unified with the poster):
  plain = light gray #6e6e6e (light series #c9c9c9),
  BLS   = poster purple #741B5B (light series #c96aae).
Legend order: plain first, BLS second; every panel self-contained.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "_poster_template", "figures")

MM = 25.4  # millimetres per inch

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.linewidth"] = 1.1
plt.rcParams["axes.edgecolor"] = "#4d4d4d"
plt.rcParams["xtick.color"] = "#4d4d4d"
plt.rcParams["ytick.color"] = "#4d4d4d"

PLAIN = "#6e6e6e"        # plain local search (trajectories / dark gray boxes)
PLAIN_LIGHT = "#c9c9c9"  # plain, light series (bars / Min-Min boxes)
BLS = "#741B5B"          # proposed method (poster purple)
BLS_LIGHT = "#c96aae"    # BLS, light series
MID_PURPLE = "#a05b8f"   # intermediate series (round-robin 5000)
EDGE = "#2b2b2b"
GRID = "#c8c8c8"


def _save(fig, name, w_mm, h_mm):
    fig.set_size_inches(w_mm / MM, h_mm / MM)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path)  # no bbox_inches="tight": keep the exact 1:1 mm size
    plt.close(fig)
    print(f"wrote {path}  ({w_mm} x {h_mm} mm)")


# ----------------------------------------------------------------------
# 1) Motivation: improvement rate vs bottleneck sharpness rho
# ----------------------------------------------------------------------
def fig_motivation():
    rows = list(csv.DictReader(
        open(os.path.join(RES, "verify_q4.csv"), encoding="utf-8-sig")))
    BINS = ["rho<0.05", "0.05<=rho<0.15", "rho>=0.15"]
    LABELS = [r"$\rho<0.05$", r"$0.05\leq\rho<0.15$", r"$\rho\geq0.15$"]
    SERIES = ["plain", "bottleneck"]

    means, counts = {}, {}
    for b in BINS:
        sub = [r for r in rows if r["rho_bin"] == b]
        counts[b] = len(sub)
        for se in SERIES:
            vals = [float(r[f"improvement_{se}_pct"]) for r in sub]
            means[(b, se)] = sum(vals) / len(vals) if vals else 0.0

    fig = plt.figure(figsize=(275 / MM, 136 / MM))
    ax = fig.add_axes([0.215, 0.085, 0.765, 0.80])
    x = list(range(len(BINS)))
    w = 0.36
    for bi, b in enumerate(BINS):
        for si, se in enumerate(SERIES):
            v = means[(b, se)]
            color = PLAIN_LIGHT if se == "plain" else BLS
            ax.bar(bi + (si - 0.5) * w, v, width=w, color=color,
                   edgecolor=EDGE, linewidth=1.2)
            if counts[b]:
                ax.text(bi + (si - 0.5) * w, v + 0.35, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=15)
        if counts[b]:
            ax.text(bi, 15.5, f"$n={counts[b]}$", ha="center", va="top",
                    fontsize=15)
        else:
            ax.text(bi, 12.4, "no samples", ha="center", va="bottom",
                    fontsize=15, style="italic", color="#808080")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=15)
    ax.set_ylabel("Improvement (%)", fontsize=17, labelpad=6)
    ax.set_ylim(0, 15.5)
    ax.tick_params(labelsize=15, length=4, width=1.1)
    ax.grid(axis="y", linestyle=(0, (1, 3)), linewidth=0.8, color=GRID,
            alpha=0.8)
    handles = [
        Patch(facecolor=PLAIN_LIGHT, edgecolor=EDGE, label="plain"),
        Patch(facecolor=BLS, edgecolor=EDGE, label="BLS"),
    ]
    ax.legend(handles=handles, fontsize=15, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), frameon=False,
              handlelength=1.8, handleheight=1.2, columnspacing=1.2)
    _save(fig, "motivation_poster.pdf", 275, 136)


# ----------------------------------------------------------------------
# 2) Results: improvement boxplots + final-gap trajectories
# ----------------------------------------------------------------------
def fig_results():
    rows = list(csv.DictReader(
        open(os.path.join(RES, "real_main.csv"), encoding="utf-8-sig")))
    SIZES = ["512x16", "1024x32", "2048x64"]
    INITS = ["min_min", "round_robin"]
    SE = ["plain", "bottleneck"]

    def size_of(r):
        return f"{r['n']}x{r['m']}"

    data = {(s, i, se): [float(r["improvement_pct"]) for r in rows
                         if size_of(r) == s and r["init_method"] == i
                         and r["searcher"] == se]
            for s in SIZES for i in INITS for se in SE}
    means = {}
    for s in SIZES:
        for i in INITS:
            for se in SE:
                for b in ["5000", "10000"]:
                    vals = [float(r["gap_final_pct"]) for r in rows
                            if size_of(r) == s and r["init_method"] == i
                            and r["searcher"] == se and r["max_evals"] == b]
                    means[(s, i, se, b)] = sum(vals) / len(vals)

    fig = plt.figure(figsize=(500 / MM, 140 / MM))
    gs = fig.add_gridspec(1, 3, left=0.115, right=0.992, top=0.93,
                          bottom=0.26, wspace=0.53,
                          width_ratios=[1.45, 1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # -- panel 0: boxplots ------------------------------------------------
    FACE = {("min_min", "plain"): PLAIN_LIGHT,
            ("min_min", "bottleneck"): BLS_LIGHT,
            ("round_robin", "plain"): PLAIN,
            ("round_robin", "bottleneck"): BLS}
    local = [0.0, 1.0, 1.35, 2.35]
    cluster_gap = 1.0
    width, step = 4.0, 5.0
    centers = []
    for si, s in enumerate(SIZES):
        base = si * step
        pos = [base + l for l in local]
        centers.append(base + (local[-1] - local[0]) / 2)
        for pi, (i, se) in enumerate([(i_, s_) for i_ in INITS for s_ in SE]):
            ax0.boxplot(
                data[(s, i, se)], positions=[pos[pi]], widths=0.75,
                patch_artist=True,
                boxprops=dict(facecolor=FACE[(i, se)], edgecolor=EDGE,
                              linewidth=1.2),
                medianprops=dict(
                    color="white" if i == "round_robin" else EDGE,
                    linewidth=2.0),
                whiskerprops=dict(color=EDGE, linewidth=1.2),
                capprops=dict(color=EDGE, linewidth=1.2),
                flierprops=dict(marker=".", markersize=5,
                                markeredgecolor=EDGE, markerfacecolor=EDGE))
    ax0.set_yscale("log")
    ax0.set_xticks(centers)
    ax0.set_xticklabels(
        [r"$512\times16$", r"$1024\times32$", r"$2048\times64$"],
        fontsize=16)
    ax0.set_ylabel("Improvement (%) (log scale)", fontsize=18, labelpad=6)
    ax0.set_ylim(0.3, 2000)  # headroom at the top for the legend
    ax0.tick_params(labelsize=16, length=4, width=1.1)
    ax0.grid(axis="y", which="major", linestyle=(0, (1, 3)), linewidth=0.8,
             color=GRID, alpha=0.8)
    legend0 = [
        Patch(facecolor=PLAIN_LIGHT, edgecolor=EDGE, label="plain, Min-Min"),
        Patch(facecolor=PLAIN, edgecolor=EDGE, label="plain, round-robin"),
        Patch(facecolor=BLS_LIGHT, edgecolor=EDGE, label="BLS, Min-Min"),
        Patch(facecolor=BLS, edgecolor=EDGE, label="BLS, round-robin"),
    ]
    ax0.legend(handles=legend0, fontsize=15, ncol=2, loc="upper left",
               frameon=False, handlelength=1.8, handleheight=1.2,
               columnspacing=1.0, borderaxespad=0.2)

    # -- panels 1/2: group-mean final-gap trajectories --------------------
    STYLES = {
        ("512x16", "plain"):      dict(ls=(0, (4, 2)), marker="o", color=PLAIN),
        ("1024x32", "plain"):     dict(ls=(0, (4, 2)), marker="s", color=PLAIN),
        ("2048x64", "plain"):     dict(ls=(0, (4, 2)), marker="^", color=PLAIN),
        ("512x16", "bottleneck"): dict(ls="-", marker="o", color=BLS),
        ("1024x32", "bottleneck"): dict(ls="-", marker="s", color=BLS),
        ("2048x64", "bottleneck"): dict(ls="-", marker="^", color=BLS),
    }
    xs = [0, 1]
    for ai, init in enumerate(INITS):
        ax = [ax1, ax2][ai]
        for s in SIZES:
            for se in SE:
                ys = [means[(s, init, se, b)] for b in ["5000", "10000"]]
                lw = 2.4 if se == "bottleneck" else 1.6
                ax.plot(xs, ys, **STYLES[(s, se)], markersize=7,
                        linewidth=lw, markeredgewidth=1.0,
                        markeredgecolor="white", zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(5, 40) if ai == 0 else ax.set_ylim(80, 50000)
        ax.set_title({"min_min": "Min-Min",
                      "round_robin": "round-robin"}[init],
                     fontsize=17, pad=5)
        ax.set_xticks(xs)
        ax.set_xticklabels(["5000", "10000"], fontsize=16)
        ax.set_xlabel("Evaluation budget", fontsize=17, labelpad=3)
        ax.set_xlim(-0.12, 1.12)
        ax.tick_params(labelsize=16, length=4, width=1.1)
        ax.grid(axis="y", which="major", linestyle=(0, (1, 3)), linewidth=0.8,
                color=GRID, alpha=0.8)
    ax1.set_ylabel("Final gap (%) (log scale)", fontsize=18, labelpad=6)

    # figure-level legend for the trajectories (one row, bottom centre)
    handles, labels = [], []
    for s in SIZES:
        for se in SE:
            lw = 2.4 if se == "bottleneck" else 1.6
            line, = ax2.plot([], [], **STYLES[(s, se)], linewidth=lw)
            handles.append(line)
            labels.append(f"{se}, {s}")
    fig.legend(handles, labels, fontsize=15, ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, 0.02), frameon=False,
               handlelength=2.2, columnspacing=1.2)
    _save(fig, "results_poster.pdf", 500, 140)


# ----------------------------------------------------------------------
# 3) Qualitative: DR boundary, mean B-A improvement difference (pp)
# ----------------------------------------------------------------------
def fig_qualitative():
    rows = list(csv.DictReader(
        open(os.path.join(RES, "real_q6.csv"), encoding="utf-8-sig")))
    CATS = ["u_c", "u_i", "u_s"]
    CAT_LABELS = [r"$u_c$", r"$u_i$", r"$u_s$"]

    pairs = {}
    for r in rows:
        k = (r["filepath"], r["init_method"], r["max_evals"])
        pairs.setdefault(k, {})[r["searcher"]] = r
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

    fig = plt.figure(figsize=(270 / MM, 143 / MM))
    ax = fig.add_axes([0.165, 0.155, 0.815, 0.745])
    x = list(range(len(CATS)))
    w = 0.22
    series = [
        ("min_min", "5000", "#c9c9c9"),
        ("min_min", "10000", "#c9c9c9"),
        ("round_robin", "5000", MID_PURPLE),
        ("round_robin", "10000", BLS),
    ]
    for si, (init, b, color) in enumerate(series):
        ys = [vals[(cat, init, b)] for cat in CATS]
        ax.bar([xi + (si - 1.5) * w for xi in x], ys, width=w,
               color=color, edgecolor=EDGE, linewidth=1.1)
        if init == "round_robin":
            for xi, v in zip(x, ys):
                ax.text(xi + (si - 1.5) * w, v - 1.6,
                        f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}",
                        ha="center", va="top", fontsize=13)
    ax.axhline(0.0, color="#1a1a1a", linestyle=(0, (4, 2)), linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(CAT_LABELS, fontsize=18)
    ax.set_xlabel("Consistency class", fontsize=18, labelpad=9)
    ax.set_ylabel(r"Mean $B-A$ (pp)", fontsize=18, labelpad=6)
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(-46, 8)
    ax.tick_params(labelsize=18, length=4, width=1.1)
    ax.grid(axis="y", linestyle=(0, (1, 3)), linewidth=0.8, color=GRID,
            alpha=0.8, zorder=0)
    handles = [
        Patch(facecolor="#c9c9c9", edgecolor=EDGE,
              label="Min-Min: 0.00"),
        Patch(facecolor=MID_PURPLE, edgecolor=EDGE, label="round-robin, 5000"),
        Patch(facecolor=BLS, edgecolor=EDGE, label="round-robin, 10000"),
    ]
    ax.legend(handles=handles, fontsize=16, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.0), frameon=False,
              handlelength=1.8, handleheight=1.2, columnspacing=1.0)
    _save(fig, "qualitative_poster.pdf", 270, 143)


# ----------------------------------------------------------------------
# 4) Parameter sensitivity heatmap
# ----------------------------------------------------------------------
def fig_paramheat():
    rows = list(csv.DictReader(
        open(os.path.join(RES, "real_param_sensitivity.csv"),
             encoding="utf-8-sig")))
    KS = [4, 8, 16]
    RS_SS = [(2, 4), (2, 8), (4, 4), (4, 8)]

    def mean_impr(init, K, R, S):
        sub = [r for r in rows
               if r["init_method"] == init
               and int(r["K"]) == K and int(r["R"]) == R and int(r["S"]) == S]
        vals = [float(r["improvement_pct"]) for r in sub]
        return sum(vals) / len(vals) if vals else float("nan")

    fig = plt.figure(figsize=(194 / MM, 86 / MM))
    gs = fig.add_gridspec(1, 2, left=0.245, right=0.88, top=0.86,
                          bottom=0.10, wspace=0.05)
    initials = ["min_min", "round_robin"]
    titles = {"min_min": "Min-Min initials", "round_robin": "round-robin initials"}
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

    im = None
    vmin = vmax = None
    for ax, init in zip(axes, initials):
        M = np.array([[mean_impr(init, K, R, S) for K in KS]
                      for (R, S) in RS_SS])
        vmin, vmax = np.nanmin(M), np.nanmax(M)
        im = ax.imshow(M, cmap="Greys", vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(4):
            for j in range(3):
                v = M[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=16,
                        color="white" if v > (vmin + vmax) / 2 else "black")
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, edgecolor="white",
                                           linewidth=1.6))
        ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, fill=False,
                                   edgecolor=BLS, linewidth=3.0))
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"$K={k}$" for k in KS], fontsize=16)
        ax.set_yticks(range(4))
        if init == "min_min":
            ax.set_yticklabels([f"$(R,S)=({r},{s})$" for (r, s) in RS_SS],
                               fontsize=16)
        else:
            ax.set_yticklabels([])  # row labels shared with the left panel
        ax.tick_params(length=0)
        ax.grid(False)
        pos = gs[0, 0].get_position(fig) if init == "min_min" \
            else gs[0, 1].get_position(fig)
        fig.text(pos.x0 + pos.width / 2, 0.915, titles[init],
                 ha="center", va="center", fontsize=17)

    # colorbar + label
    cax = fig.add_axes([0.905, 0.12, 0.035, 0.66])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=14, length=3, width=1.0)
    lo = int(np.floor(vmin)); hi = int(np.ceil(vmax))
    cb.set_ticks([lo, round((lo + hi) / 2), hi])
    fig.text(0.888, 0.45, "Improvement (%)",
             rotation=90, ha="center", va="center", fontsize=14)
    _save(fig, "paramheat_poster.pdf", 194, 86)


if __name__ == "__main__":
    fig_motivation()
    fig_results()
    fig_qualitative()
    fig_paramheat()
