"""验证 Q1（最关键）：等评估次数下 bottleneck 是否优于 plain。

设计：
- 规模：512x16、1024x32、2048x64（generate_synthetic_matrix，seed=42）
- 初解：round_robin / min_min / lpt / random_greedy(seed=42)
- 搜索：plain_local_search 与 bottleneck_local_search(K=8, R=4, S=8)
- 预算：2000 / 5000 / 10000 / 20000 次评估
- 每个 (规模, 初解, 预算) 组合跑一次 plain 与 bottleneck，记录
  c0、c_plain、c_bottleneck、evals、accepted、runtime
- 输出 CSV：results/verify_q1.csv
- 控制台打印紧凑表格：每行显示 c0、两解、差值（pp）、bottleneck 是否更优

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
BUDGETS = [2000, 5000, 10000, 20000]
INIT_FNS = {
    "round_robin": lambda P: round_robin(P),
    "min_min": lambda P: min_min(P),
    "lpt": lambda P: lpt(P),
    "random_greedy": lambda P: random_greedy(P, seed=42),
}
MATRIX_SEED = 42
OUT_CSV = Path(__file__).resolve().parents[1] / "results" / "verify_q1.csv"
K, R, S = 8, 4, 8


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print(
        f"{'n':>5} {'m':>3} {'init':<12} {'evals':>6} | "
        f"{'c0':>10} {'plain':>10} {'bottle':>10} {'diff(pp)':>9} better?"
    )
    print("-" * 92)

    for n, m in SIZES:
        P = generate_synthetic_matrix(n, m, seed=MATRIX_SEED)
        print(f"\n== n={n}, m={m} ==")
        for init_name, init_fn in INIT_FNS.items():
            t0 = time.perf_counter()
            assignment, loads, c0 = init_fn(P)
            init_time = time.perf_counter() - t0
            for budget in BUDGETS:
                t0 = time.perf_counter()
                res_p = plain_local_search(P, assignment, loads, c0, budget)
                t_plain = time.perf_counter() - t0

                t0 = time.perf_counter()
                res_b = bottleneck_local_search(
                    P, assignment, loads, c0, K=K, R=R, S=S, max_evals=budget
                )
                t_bottle = time.perf_counter() - t0

                c_plain = float(res_p["cmax"])
                c_bottle = float(res_b["cmax"])
                i_plain = improvement_rate(c0, c_plain)
                i_bottle = improvement_rate(c0, c_bottle)
                diff = i_bottle - i_plain
                better = c_bottle < c_plain - 1e-9
                rows.append(
                    {
                        "n": n,
                        "m": m,
                        "init_method": init_name,
                        "max_evals": budget,
                        "c0": round(c0, 6),
                        "c_plain": round(c_plain, 6),
                        "c_bottleneck": round(c_bottle, 6),
                        "improvement_plain_pct": round(i_plain, 6),
                        "improvement_bottleneck_pct": round(i_bottle, 6),
                        "evals_plain": int(res_p["evals"]),
                        "evals_bottleneck": int(res_b["evals"]),
                        "accepted_plain": int(res_p["accepted_moves"]),
                        "accepted_bottleneck": int(res_b["accepted_moves"]),
                        "runtime_plain_sec": round(t_plain, 6),
                        "runtime_bottleneck_sec": round(t_bottle, 6),
                        "bottleneck_better": int(better),
                    }
                )
                print(
                    f"{n:>5} {m:>3} {init_name:<12} {budget:>6} | "
                    f"{c0:>10.2f} {c_plain:>10.2f} {c_bottle:>10.2f} "
                    f"{diff:>+9.2f} {'yes' if better else 'no':>7}"
                )
            # 初解耗时仅提示，不计入行
            _ = init_time

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
