"""验证 Q2：等时间预算下 bottleneck 对 plain 的优势是否保持。

设计：
- 与 Q1 相同实例（generate_synthetic_matrix，seed=42）与初解
- 时间预算：1s / 3s / 5s（每个搜索独立计时）
- 实现：plain_local_search 与 bottleneck_local_search 新增可选参数
  time_budget（默认 None，行为不变）；本脚本用 max_evals=10^9 配合
  time_budget，在每轮边界检查时间，超过预算即停止
- 记录 c0、c_final、improvement、实际 evals、accepted、实际耗时
- 输出 CSV：results/verify_q2.csv
- 控制台打印每个 (规模, 初解, 时间预算) 下 plain 与 bottleneck 的结果

单独可运行，不依赖其他验证脚本。
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import improvement_rate
from src.initial_solutions import lpt, min_min, random_greedy, round_robin
from src.loader import generate_synthetic_matrix
from src.local_search import bottleneck_local_search, plain_local_search

SIZES = [(512, 16), (1024, 32), (2048, 64)]
TIME_BUDGETS = [1.0, 3.0, 5.0]
INIT_FNS = {
    "round_robin": lambda P: round_robin(P),
    "min_min": lambda P: min_min(P),
    "lpt": lambda P: lpt(P),
    "random_greedy": lambda P: random_greedy(P, seed=42),
}
MATRIX_SEED = 42
OUT_CSV = Path(__file__).resolve().parents[1] / "results" / "verify_q2.csv"
K, R, S = 8, 4, 8
HUGE_EVALS = 10**9  # 让时间预算成为实际停止条件


def run_one(P, assignment, loads, c0, searcher, time_budget):
    """运行一次搜索，返回 (c_final, improvement, evals, accepted, runtime)。"""
    t0 = time.perf_counter()
    if searcher == "plain":
        res = plain_local_search(P, assignment, loads, c0, HUGE_EVALS,
                                 time_budget=time_budget)
    else:
        res = bottleneck_local_search(P, assignment, loads, c0,
                                      K=K, R=R, S=S, max_evals=HUGE_EVALS,
                                      time_budget=time_budget)
    runtime = time.perf_counter() - t0
    c_final = float(res["cmax"])
    return (
        c_final,
        improvement_rate(c0, c_final),
        int(res["evals"]),
        int(res["accepted_moves"]),
        runtime,
    )


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print(
        f"{'n':>5} {'m':>3} {'init':<12} {'T(s)':>4} | "
        f"{'c0':>10} {'c_plain':>10} {'c_bottle':>10} {'diff(pp)':>9} "
        f"{'evals_p':>8} {'evals_b':>8} better?"
    )
    print("-" * 112)

    for n, m in SIZES:
        P = generate_synthetic_matrix(n, m, seed=MATRIX_SEED)
        print(f"\n== n={n}, m={m} ==")
        for init_name, init_fn in INIT_FNS.items():
            assignment, loads, c0 = init_fn(P)
            for budget in TIME_BUDGETS:
                c_p, i_p, ev_p, ac_p, rt_p = run_one(
                    P, assignment, loads, c0, "plain", budget
                )
                c_b, i_b, ev_b, ac_b, rt_b = run_one(
                    P, assignment, loads, c0, "bottleneck", budget
                )
                diff = i_b - i_p
                better = c_b < c_p - 1e-9
                rows.append(
                    {
                        "n": n,
                        "m": m,
                        "init_method": init_name,
                        "time_budget": budget,
                        "c0": round(c0, 6),
                        "c_plain": round(c_p, 6),
                        "c_bottleneck": round(c_b, 6),
                        "improvement_plain_pct": round(i_p, 6),
                        "improvement_bottleneck_pct": round(i_b, 6),
                        "evals_plain": ev_p,
                        "evals_bottleneck": ev_b,
                        "accepted_plain": ac_p,
                        "accepted_bottleneck": ac_b,
                        "runtime_plain_sec": round(rt_p, 6),
                        "runtime_bottleneck_sec": round(rt_b, 6),
                        "bottleneck_better": int(better),
                    }
                )
                print(
                    f"{n:>5} {m:>3} {init_name:<12} {budget:>4.0f} | "
                    f"{c0:>10.2f} {c_p:>10.2f} {c_b:>10.2f} {diff:>+9.2f} "
                    f"{ev_p:>8} {ev_b:>8} {'yes' if better else 'no':>7}"
                )

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_cells = len(rows)
    n_better = sum(r["bottleneck_better"] for r in rows)
    diffs = [r["improvement_bottleneck_pct"] - r["improvement_plain_pct"] for r in rows]
    print(f"\n总计 {n_cells} 个单元格，bottleneck 更优 {n_better} 个 "
          f"({100 * n_better / n_cells:.1f}%)，平均差值 {sum(diffs) / len(diffs):+.2f}pp")
    print(f"CSV 已写入: {OUT_CSV}")


if __name__ == "__main__":
    main()
