"""验证 Q4：实例结构稳健性——按瓶颈尖锐度 rho 分档的改进率对比。

设计：
- 三种结构（512x16，各 10 个矩阵种子 100..109）：
  1. balanced：heterogeneity=0.3（均衡型）
  2. medium：heterogeneity=1.0（中等异质）
  3. high：heterogeneity=2.0（强异质）
- 对每个实例计算 Min-Min 初解的 bottleneck_sharpness(loads)，分成三档：
  rho < 0.05 / 0.05 <= rho < 0.15 / rho >= 0.15
- 每档下跑 plain 与 bottleneck（预算 5000），记录改进率与 rho
- 输出 CSV：results/verify_q4.csv
- 控制台打印按 rho 分档的改进率对比

说明：三种结构各只有一个 rho 值的话分档无统计意义，因此每种结构
取 10 个矩阵种子以获得足够样本（与"不要每个实验都上最大规模"的
精神一致，512x16 很快）。

单独可运行，不依赖其他验证脚本。
"""

from __future__ import annotations

import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bottleneck import bottleneck_sharpness
from src.evaluate import improvement_rate
from src.initial_solutions import min_min
from src.loader import generate_synthetic_matrix
from src.local_search import bottleneck_local_search, plain_local_search

N, M = 512, 16
STRUCTURES = [
    ("balanced", 0.3),
    ("medium", 1.0),
    ("high", 2.0),
]
MATRIX_SEEDS = list(range(100, 110))
MAX_EVALS = 5000
K, R, S = 8, 4, 8
BINS = [(0.0, 0.05, "rho<0.05"), (0.05, 0.15, "0.05<=rho<0.15"), (0.15, 1e9, "rho>=0.15")]
OUT_CSV = Path(__file__).resolve().parents[1] / "results" / "verify_q4.csv"


def bin_of(rho: float) -> str:
    for lo, hi, name in BINS:
        if lo <= rho < hi:
            return name
    return "rho>=0.15"


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"== n={N}, m={M}, max_evals={MAX_EVALS} ==")
    print(f"{'structure':<9} {'seed':>4} | {'rho':>8} {'bin':<14} "
          f"{'c0':>9} {'I_plain':>8} {'I_bottle':>8} diff(pp)")
    print("-" * 82)

    for sname, hetero in STRUCTURES:
        print(f"\n-- {sname} (heterogeneity={hetero}) --")
        for seed in MATRIX_SEEDS:
            P = generate_synthetic_matrix(N, M, seed=seed, heterogeneity=hetero)
            assignment, loads, c0 = min_min(P)
            rho = float(bottleneck_sharpness(loads))
            rho_bin = bin_of(rho)

            t0 = time.perf_counter()
            res_p = plain_local_search(P, assignment, loads, c0, MAX_EVALS)
            rt_p = time.perf_counter() - t0
            t0 = time.perf_counter()
            res_b = bottleneck_local_search(
                P, assignment, loads, c0, K=K, R=R, S=S, max_evals=MAX_EVALS
            )
            rt_b = time.perf_counter() - t0

            c_p = float(res_p["cmax"])
            c_b = float(res_b["cmax"])
            i_p = improvement_rate(c0, c_p)
            i_b = improvement_rate(c0, c_b)
            diff = i_b - i_p
            rows.append(
                {
                    "n": N,
                    "m": M,
                    "structure": sname,
                    "heterogeneity": hetero,
                    "seed": seed,
                    "rho": round(rho, 6),
                    "rho_bin": rho_bin,
                    "c0": round(c0, 6),
                    "c_plain": round(c_p, 6),
                    "c_bottleneck": round(c_b, 6),
                    "improvement_plain_pct": round(i_p, 6),
                    "improvement_bottleneck_pct": round(i_b, 6),
                    "evals_plain": int(res_p["evals"]),
                    "evals_bottleneck": int(res_b["evals"]),
                    "runtime_plain_sec": round(rt_p, 6),
                    "runtime_bottleneck_sec": round(rt_b, 6),
                }
            )
            print(
                f"{sname:<9} {seed:>4} | {rho:>8.4f} {rho_bin:<14} "
                f"{c0:>9.2f} {i_p:>8.2f} {i_b:>8.2f} {diff:>+7.2f}"
            )

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 按 rho 分档聚合
    print("\n== 按 rho 分档的改进率对比（%）==")
    print(f"{'rho_bin':<14} {'n':>3} {'mean_rho':>9} "
          f"{'I_plain':>8} {'I_bottle':>8} {'diff(pp)':>9}")
    by_bin: dict[str, list[dict]] = {}
    for r in rows:
        by_bin.setdefault(r["rho_bin"], []).append(r)
    for lo, hi, name in BINS:
        if name not in by_bin:
            print(f"{name:<14} {'0':>3} {'-':>9} {'-':>8} {'-':>8} {'-':>9}")
            continue
        grp = by_bin[name]
        mean_rho = statistics.mean(r["rho"] for r in grp)
        mean_p = statistics.mean(r["improvement_plain_pct"] for r in grp)
        mean_b = statistics.mean(r["improvement_bottleneck_pct"] for r in grp)
        print(f"{name:<14} {len(grp):>3} {mean_rho:>9.4f} "
              f"{mean_p:>8.2f} {mean_b:>8.2f} {mean_b - mean_p:>+9.2f}")

    print(f"\nCSV 已写入: {OUT_CSV}")


if __name__ == "__main__":
    main()
