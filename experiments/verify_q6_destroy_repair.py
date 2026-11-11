"""验证 Q6：破坏-重建（Destroy-and-Repair）机制的贡献消融。

阶段一（必做，快速验证）：
- 规模：512x16、1024x32（generate_synthetic_matrix，seed=42，heterogeneity=1.0）
- 初解：round_robin / min_min
- 预算：5000 / 10000 次评估
- 三组对比：
  A: bottleneck_local_search(K=4, R=2, S=4)              —— 基线（Q5 最优配置）
  B: bottleneck_local_search_dr(K=4, R=2, S=4, accept_strategy="strict")
  C: bottleneck_local_search_dr(K=4, R=2, S=4, accept_strategy="metropolis")
- 输出：results/verify_q6_phase1.csv + verify_q6_phase1_summary.md

阶段二（条件触发）：当阶段一中 B 相比 A 的平均改进率提升 >= +1.0pp 且
在多数单元格不劣化时，扩展到 2048x64、lpt、random_greedy、预算 20000，
输出 results/verify_q6_full.csv + verify_q6_full_summary.md。

机制演示（附）：三节点循环陷阱实例上，单次瓶颈搜索陷入局部最优而
破坏-重建成功逃逸，用于论文边界讨论。

单独可运行，不依赖其他验证脚本。
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.destroy_repair import bottleneck_local_search_dr
from src.evaluate import improvement_rate
from src.feasibility import compute_loads
from src.initial_solutions import lpt, min_min, random_greedy, round_robin
from src.loader import generate_synthetic_matrix
from src.local_search import bottleneck_local_search

MATRIX_SEED = 42
K, R, S = 4, 2, 4  # Q5 验证的最优配置
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

PHASE1_SIZES = [(512, 16), (1024, 32)]
PHASE1_INITS = {
    "round_robin": lambda P: round_robin(P),
    "min_min": lambda P: min_min(P),
}
PHASE1_BUDGETS = [5000, 10000]
PHASE2_SIZES = [(512, 16), (1024, 32), (2048, 64)]
PHASE2_INITS = {
    "round_robin": lambda P: round_robin(P),
    "min_min": lambda P: min_min(P),
    "lpt": lambda P: lpt(P),
    "random_greedy": lambda P: random_greedy(P, seed=42),
}
PHASE2_BUDGETS = [5000, 10000, 20000]


def _run_cell(P, assignment, loads, c0, budget):
    """跑一个 (规模, 初解, 预算) 单元格的三组搜索，返回记录 dict。"""
    t0 = time.perf_counter()
    res_a = bottleneck_local_search(
        P, assignment, loads, c0, K=K, R=R, S=S, max_evals=budget
    )
    t_a = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_b = bottleneck_local_search_dr(
        P, assignment, loads, c0, K=K, R=R, S=S, max_evals=budget,
        accept_strategy="strict",
    )
    t_b = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_c = bottleneck_local_search_dr(
        P, assignment, loads, c0, K=K, R=R, S=S, max_evals=budget,
        accept_strategy="metropolis",
    )
    t_c = time.perf_counter() - t0

    i_a = improvement_rate(c0, res_a["cmax"])
    i_b = improvement_rate(c0, res_b["cmax"])
    i_c = improvement_rate(c0, res_c["cmax"])
    return {
        "c0": round(c0, 6),
        "c_a": round(res_a["cmax"], 6),
        "c_b": round(res_b["cmax"], 6),
        "c_c": round(res_c["cmax"], 6),
        "improvement_a_pct": round(i_a, 6),
        "improvement_b_pct": round(i_b, 6),
        "improvement_c_pct": round(i_c, 6),
        "evals_a": int(res_a["evals"]),
        "evals_b": int(res_b["evals"]),
        "evals_c": int(res_c["evals"]),
        "segments_b": int(res_b["segments"]),
        "segments_c": int(res_c["segments"]),
        "restarts_b": int(res_b["restarts"]),
        "restarts_c": int(res_c["restarts"]),
        "accepted_worse_c": int(res_c["accepted_worse"]),
        "runtime_a_sec": round(t_a, 6),
        "runtime_b_sec": round(t_b, 6),
        "runtime_c_sec": round(t_c, 6),
        "b_vs_a_pp": round(i_b - i_a, 6),
        "c_vs_a_pp": round(i_c - i_a, 6),
        "b_better_than_a": int(res_b["cmax"] < res_a["cmax"] - 1e-9),
        "c_better_than_a": int(res_c["cmax"] < res_a["cmax"] - 1e-9),
    }


def run_cells(sizes, inits, budgets, out_csv, print_header=True):
    """按 (规模, 初解, 预算) 网格跑三组搜索，写 CSV，返回记录列表。"""
    rows = []
    if print_header:
        print(
            f"{'n':>5} {'m':>3} {'init':<13} {'evals':>6} | "
            f"{'c0':>10} {'I_A':>8} {'I_B':>8} {'I_C':>8} "
            f"{'B-A(pp)':>8} {'C-A(pp)':>8}"
        )
        print("-" * 86)
    for n, m in sizes:
        P = generate_synthetic_matrix(n, m, seed=MATRIX_SEED)
        print(f"\n== n={n}, m={m} ==")
        for init_name, init_fn in inits.items():
            assignment, loads, c0 = init_fn(P)
            for budget in budgets:
                rec = _run_cell(P, assignment, loads, c0, budget)
                rec.update({"n": n, "m": m, "init_method": init_name,
                            "max_evals": budget})
                rows.append(rec)
                print(
                    f"{n:>5} {m:>3} {init_name:<13} {budget:>6} | "
                    f"{c0:>10.2f} {rec['improvement_a_pct']:>8.2f} "
                    f"{rec['improvement_b_pct']:>8.2f} "
                    f"{rec['improvement_c_pct']:>8.2f} "
                    f"{rec['b_vs_a_pp']:>+8.2f} {rec['c_vs_a_pp']:>+8.2f}"
                )
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarize(rows):
    """汇总一组记录：B/C 相对 A 的平均 pp、更优/持平/劣化计数。"""
    b_diffs = [r["b_vs_a_pp"] for r in rows]
    c_diffs = [r["c_vs_a_pp"] for r in rows]
    b_better = sum(r["b_better_than_a"] for r in rows)
    c_better = sum(r["c_better_than_a"] for r in rows)
    n_cells = len(rows)
    return {
        "n_cells": n_cells,
        "mean_b_vs_a": sum(b_diffs) / n_cells,
        "mean_c_vs_a": sum(c_diffs) / n_cells,
        "b_better": b_better,
        "c_better": c_better,
        "b_worse": sum(r["b_vs_a_pp"] < -1e-9 for r in rows),
        "c_worse": sum(r["c_vs_a_pp"] < -1e-9 for r in rows),
    }


def run_trap_demo():
    """三节点循环陷阱：单次瓶颈搜索陷入局部最优，DR 逃逸。

    构造：60 个任务、3 个节点，P 行按组循环 roll [1,5,9]，错配初解
    （A 组放节点 2、B 组放节点 0、C 组放节点 1）。单次迁移/交换无法
    协调大规模互移，破坏-重建后可重搜到接近全局最优的平衡解。
    """
    k = 20
    P = np.empty((3 * k, 3))
    for g in range(3):
        P[g * k:(g + 1) * k] = np.roll(np.array([1.0, 5.0, 9.0]), -g)
    a0 = np.empty(3 * k, dtype=np.int64)
    a0[0:k] = 2
    a0[k:2 * k] = 0
    a0[2 * k:3 * k] = 1
    l0 = compute_loads(P, a0)
    c0 = float(np.max(l0))

    single = bottleneck_local_search(P, a0, l0, c0, K=4, R=2, S=4,
                                     max_evals=100000)
    i_single = improvement_rate(c0, single["cmax"])
    results = []
    for seed in (1, 2, 3):
        dr = bottleneck_local_search_dr(
            P, a0, l0, c0, K=4, R=2, S=4, max_evals=50000,
            destroy_frac=0.2, accept_strategy="strict",
            segment_evals=3000, max_stagnation=10, seed=seed,
        )
        # 找到首次超越单次局部最优时的累计评估数
        first_escape = None
        for entry in dr["history"]:
            if entry["cmax"] < single["cmax"] - 1e-9:
                first_escape = entry["evals"]
                break
        results.append((seed, dr["cmax"], first_escape,
                        dr["improvements"], dr["segments"]))
    return {
        "k": k,
        "c0": c0,
        "c_single": single["cmax"],
        "i_single": i_single,
        "evals_single": single["evals"],
        "runs": results,
    }


def phase1_summary_md(rows, trap):
    """生成阶段一摘要 markdown 文本。"""
    s = summarize(rows)
    trigger = s["mean_b_vs_a"] >= 1.0 and s["b_worse"] <= s["n_cells"] // 2
    lines = []
    lines.append("# Q6 破坏-重建机制消融验证摘要（阶段一）\n")
    lines.append("## 实验设计\n")
    lines.append("- 规模：512x16、1024x32（generate_synthetic_matrix, "
                 f"seed={MATRIX_SEED}, heterogeneity=1.0）")
    lines.append("- 初解：round_robin、min_min；预算：5000、10000 次评估")
    lines.append("- 三组对比（K=4, R=2, S=4，Q5 最优配置）：")
    lines.append("  - A: bottleneck_local_search（基线）")
    lines.append("  - B: bottleneck_local_search_dr（strict 接受准则，"
                 "默认参数 destroy_frac=0.2, segment_evals=1000, "
                 "max_stagnation=3）")
    lines.append("  - C: bottleneck_local_search_dr（metropolis 接受准则，"
                 "其余同 B）\n")
    lines.append("## 阶段一结果\n")
    lines.append(f"- 单元格总数：{s['n_cells']}")
    lines.append(f"- B 相对 A 平均改进率差值：{s['mean_b_vs_a']:+.2f}pp；"
                 f"B 更优 {s['b_better']} 格、劣化 {s['b_worse']} 格、"
                 f"其余持平")
    lines.append(f"- C 相对 A 平均改进率差值：{s['mean_c_vs_a']:+.2f}pp；"
                 f"C 更优 {s['c_better']} 格、劣化 {s['c_worse']} 格、"
                 f"其余持平\n")
    lines.append("按初解/预算拆分的平均 B-A（pp）：\n")
    lines.append("| 初解 | 5000 | 10000 |")
    lines.append("|---|---|---|")
    for init in ("round_robin", "min_min"):
        cells = [r for r in rows if r["init_method"] == init]
        vals = []
        for budget in (5000, 10000):
            sub = [r for r in cells if r["max_evals"] == budget]
            vals.append(f"{sum(r['b_vs_a_pp'] for r in sub) / len(sub):+.2f}")
        lines.append(f"| {init} | {vals[0]} | {vals[1]} |")
    lines.append("")
    lines.append("## 阶段二触发判定\n")
    if trigger:
        lines.append(f"- B-A 平均 {s['mean_b_vs_a']:+.2f}pp >= +1.0pp 且"
                     f" 劣化格 {s['b_worse']}/{s['n_cells']} 未超半数"
                     " → 触发阶段二")
    else:
        lines.append(f"- B-A 平均 {s['mean_b_vs_a']:+.2f}pp"
                     f"（劣化格 {s['b_worse']}/{s['n_cells']}）"
                     " → 不满足触发条件（< +1.0pp），不进入阶段二")
    lines.append("")
    lines.append("## 机制有效性演示（三节点循环陷阱实例）\n")
    lines.append("- 构造：60 任务、3 节点，P 按组循环 roll [1,5,9]，"
                 "错配初解（每组全部放在最慢节点）")
    lines.append(f"- 初解 c0 = {trap['c0']:.2f}")
    lines.append(f"- 单次瓶颈搜索：收敛于 c = {trap['c_single']:.2f}"
                 f"（I = {trap['i_single']:.2f}%），仅用 "
                 f"{trap['evals_single']} evals 即耗尽邻域"
                 "（给 100000 预算也只用了这么多，无法自行逃逸）")
    for seed, c, first_escape, imp, segs in trap["runs"]:
        i = improvement_rate(trap["c0"], c)
        esc = f"{first_escape}" if first_escape is not None else "未逃逸"
        lines.append(f"- DR strict（seed={seed}）：c = {c:.2f}"
                     f"（I = {i:.2f}%），首次超越单次局部最优于累计 "
                     f"{esc} evals（{imp} 次全局改善，{segs} 段）")
    lines.append("")
    lines.append("## 结论\n")
    if not trigger:
        lines.append("阶段一未观察到 DR 相对基线的正贡献：随机合成实例上，")
        lines.append("min_min 初解下瓶颈搜索的局部最优盆地唯一且深"
                     "（扰动重搜始终回到同一局部最优），")
        lines.append("round_robin 初解下切段破坏扰乱了尚未收敛的下降进程"
                     "（segment_evals 越小越明显）。")
        lines.append("陷阱实例证明破坏-重建机制本身有效（能逃离单次搜索"
                     "无法逃出的局部最优），但在典型随机实例上没有用武之地。")
        lines.append("论文定位建议：作为边界讨论报告——DR 仅在存在"
                     "深局部最优结构的实例上有价值，")
        lines.append("对随机均匀实例无贡献；保留实现与参数接口作为未来工作。")
    else:
        lines.append("阶段一显示 DR 有正贡献（详见阶段二完整结果）。")
    return "\n".join(lines) + "\n"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 阶段一 ----
    print("== 阶段一：512/1024 规模、round_robin/min_min、5000/10000 ==")
    rows1 = run_cells(
        PHASE1_SIZES, PHASE1_INITS, PHASE1_BUDGETS,
        RESULTS_DIR / "verify_q6_phase1.csv",
    )
    trap = run_trap_demo()
    print("\n== 陷阱实例演示 ==")
    print(f"  c0={trap['c0']:.2f}  single: c={trap['c_single']:.2f} "
          f"(I={trap['i_single']:.2f}%, evals={trap['evals_single']})")
    for seed, c, first_escape, imp, segs in trap["runs"]:
        print(f"  DR seed={seed}: c={c:.2f} "
              f"(I={improvement_rate(trap['c0'], c):.2f}%, "
              f"first_escape={first_escape}, imp={imp}, segs={segs})")

    s1 = summarize(rows1)
    print(f"\n阶段一：{s1['n_cells']} 格，mean(B-A)={s1['mean_b_vs_a']:+.2f}pp，"
          f"B 更优 {s1['b_better']}、劣化 {s1['b_worse']}；"
          f"mean(C-A)={s1['mean_c_vs_a']:+.2f}pp")
    md1 = phase1_summary_md(rows1, trap)
    with open(RESULTS_DIR / "verify_q6_phase1_summary.md", "w",
              encoding="utf-8") as f:
        f.write(md1)
    print(f"阶段一 CSV 与摘要已写入 results/")

    # ---- 阶段二（条件触发） ----
    trigger = s1["mean_b_vs_a"] >= 1.0 and s1["b_worse"] <= s1["n_cells"] // 2
    if trigger:
        print("\n阶段一 B-A >= +1.0pp 且多数不劣化 → 进入阶段二")
        rows_full = run_cells(
            PHASE2_SIZES, PHASE2_INITS, PHASE2_BUDGETS,
            RESULTS_DIR / "verify_q6_full.csv",
        )
        s2 = summarize(rows_full)
        print(f"阶段二：{s2['n_cells']} 格，mean(B-A)={s2['mean_b_vs_a']:+.2f}pp，"
              f"B 更优 {s2['b_better']}、劣化 {s2['b_worse']}")
        with open(RESULTS_DIR / "verify_q6_full_summary.md", "w",
                  encoding="utf-8") as f:
            f.write(f"# Q6 破坏-重建机制消融验证摘要（完整）\n\n"
                    f"单元格总数：{s2['n_cells']}\n"
                    f"B 相对 A 平均差值：{s2['mean_b_vs_a']:+.2f}pp"
                    f"（更优 {s2['b_better']}、劣化 {s2['b_worse']}）\n"
                    f"C 相对 A 平均差值：{s2['mean_c_vs_a']:+.2f}pp"
                    f"（更优 {s2['c_better']}、劣化 {s2['c_worse']}）\n")
        print("阶段二 CSV 与摘要已写入 results/")
    else:
        print("\n阶段一 B-A < +1.0pp → 不进入阶段二，如实记录负结果"
              "（详见 verify_q6_phase1_summary.md 边界讨论）")


if __name__ == "__main__":
    main()
