"""验证 Q3：初解稳健性——不同初解下 bottleneck 搜索的改进率分布。

设计：
- 规模：512x16（generate_synthetic_matrix，seed=42）
- 初解：round_robin / min_min / lpt / random_greedy(seed=1..5)
- 搜索：bottleneck_local_search(K=8, R=4, S=8)，预算 5000
- 记录每个 (初解, 种子) 的 c0、c_final、改进率、evals、accepted、runtime
- 输出 CSV：results/verify_q3.csv
- 控制台打印每初解的改进率分布（均值/标准差/最小/最大）与改进实例比例

说明：round_robin / min_min / lpt 是确定性算法，多种子下结果相同
（如实记录 std=0）；种子只对 random_greedy 生效。

单独可运行，不依赖其他验证脚本。
"""

from __future__ import annotations

import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import improvement_rate
from src.initial_solutions import lpt, min_min, random_greedy, round_robin
from src.loader import generate_synthetic_matrix
from src.local_search import bottleneck_local_search

N, M = 512, 16
SEEDS = [1, 2, 3, 4, 5]
MATRIX_SEED = 42
MAX_EVALS = 5000
K, R, S = 8, 4, 8
OUT_CSV = Path(__file__).resolve().parents[1] / "results" / "verify_q3.csv"


def make_init(P, method, seed):
    if method == "round_robin":
        return round_robin(P)
    if method == "min_min":
        return min_min(P)
    if method == "lpt":
        return lpt(P)
    if method == "random_greedy":
        return random_greedy(P, seed=seed)
    raise ValueError(f"未知初解方法: {method}")


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    P = generate_synthetic_matrix(N, M, seed=MATRIX_SEED)
    methods = ["round_robin", "min_min", "lpt", "random_greedy"]
    rows = []

    print(f"== n={N}, m={M}, max_evals={MAX_EVALS} ==")
    print(f"{'init':<12} {'seed':>4} | {'c0':>9} {'c_final':>9} "
          f"{'I(%)':>8} {'evals':>6} {'acc':>5} {'t(s)':>7}")
    print("-" * 78)

    by_init: dict[str, list[float]] = {mname: [] for mname in methods}
    for mname in methods:
        for seed in SEEDS:
            assignment, loads, c0 = make_init(P, mname, seed)
            t0 = time.perf_counter()
            res = bottleneck_local_search(
                P, assignment, loads, c0, K=K, R=R, S=S, max_evals=MAX_EVALS
            )
            runtime = time.perf_counter() - t0
            c_final = float(res["cmax"])
            imp = improvement_rate(c0, c_final)
            by_init[mname].append(imp)
            rows.append(
                {
                    "n": N,
                    "m": M,
                    "init_method": mname,
                    "seed": seed,
                    "c0": round(c0, 6),
                    "c_final": round(c_final, 6),
                    "improvement_pct": round(imp, 6),
                    "evals": int(res["evals"]),
                    "accepted_moves": int(res["accepted_moves"]),
                    "runtime_sec": round(runtime, 6),
                }
            )
            print(
                f"{mname:<12} {seed:>4} | {c0:>9.2f} {c_final:>9.2f} "
                f"{imp:>8.2f} {res['evals']:>6} {res['accepted_moves']:>5} "
                f"{runtime:>7.3f}"
            )

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n== 每初解的改进率分布（%）==")
    print(f"{'init':<12} {'mean':>7} {'std':>7} {'min':>7} {'max':>7} "
          f"{'improved':>9}")
    for mname in methods:
        vals = by_init[mname]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        n_pos = sum(1 for v in vals if v > 1e-9)
        print(f"{mname:<12} {mean:>7.2f} {std:>7.2f} {min(vals):>7.2f} "
              f"{max(vals):>7.2f} {n_pos:>5}/{len(vals)}")

    all_vals = [v for vals in by_init.values() for v in vals]
    mean_all = statistics.mean(all_vals)
    std_all = statistics.stdev(all_vals)
    print(f"\n全部 20 次运行：均值 {mean_all:.2f}%，标准差 {std_all:.2f}%")
    print(f"CSV 已写入: {OUT_CSV}")


if __name__ == "__main__":
    main()
