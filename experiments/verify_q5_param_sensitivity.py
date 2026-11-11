"""验证 Q5：参数敏感性——(K, R, S) 网格下 bottleneck 搜索的改进率。

设计：
- 规模：512x16、1024x32（generate_synthetic_matrix，seed=42）
- 初解：min_min（主）、round_robin（辅）
- 预算：5000
- 参数网格：K in {4,8,16}，R in {2,4,8}，S in {4,8,16}，共 27 组
- 每组跑一次 bottleneck_local_search，记录 c_final、evals、accepted、runtime
- 输出 CSV：results/verify_q5.csv
- 控制台打印：
  1. 每个 K（固定 R=4, S=8）的平均改进率
  2. 每个 R（固定 K=8, S=8）的平均改进率
  3. 每个 S（固定 K=8, R=4）的平均改进率
  4. 最优 3 组与固定 (8,4,8) 的差距

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
from src.initial_solutions import min_min, round_robin
from src.loader import generate_synthetic_matrix
from src.local_search import bottleneck_local_search

SIZES = [(512, 16), (1024, 32)]
INIT_NAMES = ["min_min", "round_robin"]
MAX_EVALS = 5000
KS = [4, 8, 16]
RS = [2, 4, 8]
SS = [4, 8, 16]
MATRIX_SEED = 42
OUT_CSV = Path(__file__).resolve().parents[1] / "results" / "verify_q5.csv"


def make_init(P, method):
    if method == "min_min":
        return min_min(P)
    return round_robin(P)


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    grid = [(k, r, s) for k in KS for r in RS for s in SS]
    rows = []
    print(f"== max_evals={MAX_EVALS}，参数网格 {len(grid)} 组 ==")

    for n, m in SIZES:
        P = generate_synthetic_matrix(n, m, seed=MATRIX_SEED)
        for init_name in INIT_NAMES:
            assignment, loads, c0 = make_init(P, init_name)
            print(f"\n-- n={n}, m={m}, init={init_name}, c0={c0:.2f} --")
            print(f"{'K':>3} {'R':>2} {'S':>3} | {'c_final':>9} {'I(%)':>8} "
                  f"{'evals':>6} {'acc':>5} {'t(s)':>7}")
            for k, r, s in grid:
                t0 = time.perf_counter()
                res = bottleneck_local_search(
                    P, assignment, loads, c0, K=k, R=r, S=s, max_evals=MAX_EVALS
                )
                runtime = time.perf_counter() - t0
                c_final = float(res["cmax"])
                imp = improvement_rate(c0, c_final)
                rows.append(
                    {
                        "n": n,
                        "m": m,
                        "init_method": init_name,
                        "K": k,
                        "R": r,
                        "S": s,
                        "c0": round(c0, 6),
                        "c_final": round(c_final, 6),
                        "improvement_pct": round(imp, 6),
                        "evals": int(res["evals"]),
                        "accepted_moves": int(res["accepted_moves"]),
                        "runtime_sec": round(runtime, 6),
                    }
                )
                print(f"{k:>3} {r:>2} {s:>3} | {c_final:>9.2f} {imp:>8.2f} "
                      f"{res['evals']:>6} {res['accepted_moves']:>5} "
                      f"{runtime:>7.3f}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 聚合工具：按条件取行的改进率
    def mean_imp(pred):
        vals = [r["improvement_pct"] for r in rows if pred(r)]
        return statistics.mean(vals)

    print("\n== 每个 K（固定 R=4, S=8）的平均改进率（%，4 个规模-初解组合）==")
    for k in KS:
        print(f"  K={k:>2}: {mean_imp(lambda r: r['K'] == k and r['R'] == 4 and r['S'] == 8):>7.2f}")

    print("\n== 每个 R（固定 K=8, S=8）的平均改进率 ==")
    for r in RS:
        print(f"  R={r:>2}: {mean_imp(lambda x: x['K'] == 8 and x['R'] == r and x['S'] == 8):>7.2f}")

    print("\n== 每个 S（固定 K=8, R=4）的平均改进率 ==")
    for s in SS:
        print(f"  S={s:>2}: {mean_imp(lambda x: x['K'] == 8 and x['R'] == 4 and x['S'] == s):>7.2f}")

    # 按 (K,R,S) 聚合平均改进率，排序取最优 3 组
    combos = {}
    for r in rows:
        key = (r["K"], r["R"], r["S"])
        combos.setdefault(key, []).append(r["improvement_pct"])
    ranked = sorted(
        ((statistics.mean(v), k) for k, v in combos.items()), reverse=True
    )
    fixed_key = (8, 4, 8)
    fixed_mean = statistics.mean(combos[fixed_key])
    print("\n== 最优 3 组与固定 (8,4,8) 的对比 ==")
    for i, (mean, key) in enumerate(ranked[:3]):
        gap = mean - fixed_mean
        print(f"  #{i + 1} (K,R,S)={key}: mean I={mean:.2f}% "
              f"（与 (8,4,8) 差 {gap:+.2f}pp）")
    print(f"  固定 (8,4,8): mean I={fixed_mean:.2f}%")
    print(f"  网格内最差: (K,R,S)={ranked[-1][1]}, mean I={ranked[-1][0]:.2f}%")

    print(f"\nCSV 已写入: {OUT_CSV}")


if __name__ == "__main__":
    main()
