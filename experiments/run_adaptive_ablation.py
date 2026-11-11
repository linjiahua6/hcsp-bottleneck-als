"""自适应消融实验：plain / 固定规模 bottleneck / 自适应 bottleneck 三搜索对比。

运行方式（项目根目录）：
    # 快速模式（100x8，重复 1 次）
    .venv\\Scripts\\python.exe experiments\\run_adaptive_ablation.py --fast

    # 正式实验（512x16、1024x32、2048x64，重复 5 次，预算 2000/5000/10000）
    .venv\\Scripts\\python.exe experiments\\run_adaptive_ablation.py

    # 自定义
    .venv\\Scripts\\python.exe experiments\\run_adaptive_ablation.py ^
        --sizes 512x16 1024x32 --max-evals 2000 5000 --repeats 3

输出：
    results/adaptive_ablation.csv             逐次运行明细
    results/adaptive_ablation_summary.json    按 (n, m, init_method, searcher,
                                              max_evals) 分组的改进率均值/标准差

设计约定：
- 合成矩阵 seed 固定（同规模实例一致）；重复 5 次通过改变 initial_fn
  的 seed 实现（只对 random_greedy 生效，其余初解为确定性方法）。
- 计时用 time.perf_counter，同一进程同一线程，每个搜索单独计时。
- 改进率 improvement_pct 使用 src.evaluate.improvement_rate（单位 %）。
- 不隐藏任何结果：若 adaptive 在部分单元差于 fixed，如实写入 CSV/汇总。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adaptive_search import adaptive_bottleneck_search
from src.evaluate import improvement_rate
from src.initial_solutions import lpt, min_min, random_greedy, round_robin
from src.loader import generate_synthetic_matrix
from src.local_search import bottleneck_local_search, plain_local_search

INITIALS = {
    "round_robin": round_robin,
    "min_min": min_min,
    "lpt": lpt,
    "random_greedy": random_greedy,
}

SEARCHERS = {
    "plain": plain_local_search,
    "fixed": bottleneck_local_search,
    "adaptive": adaptive_bottleneck_search,
}

DEFAULT_SIZES = [(512, 16), (1024, 32), (2048, 64)]
FAST_SIZES = [(100, 8)]
DEFAULT_BUDGETS = [2000, 5000, 10000]
MATRIX_SEED = 2026
REPEAT_SEEDS = [100, 101, 102, 103, 104]

CSV_FIELDS = [
    "n", "m", "init_method", "searcher", "max_evals", "seed",
    "c0", "c_final", "improvement_pct", "evals", "accepted_moves", "runtime_sec",
]


def parse_size(token: str) -> tuple[int, int]:
    """把 '512x16' 解析为 (512, 16)。"""
    try:
        n_str, m_str = token.lower().split("x")
        return int(n_str), int(m_str)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"无法解析规模 '{token}'，应为形如 512x16 的字符串") from exc


def run_cell(
    P: np.ndarray,
    init_name: str,
    init_fn,
    searcher_name: str,
    searcher_fn,
    budget: int,
    seed: int,
) -> dict:
    """运行一次（初解 + 单个搜索），返回 CSV 一行。"""
    if init_fn is random_greedy:
        a0, l0, c0 = init_fn(P, seed=seed)
    else:
        a0, l0, c0 = init_fn(P)

    t0 = time.perf_counter()
    res = searcher_fn(P, a0, l0, c0, max_evals=budget)
    elapsed = time.perf_counter() - t0

    return {
        "n": P.shape[0],
        "m": P.shape[1],
        "init_method": init_name,
        "searcher": searcher_name,
        "max_evals": budget,
        "seed": seed,
        "c0": round(float(c0), 6),
        "c_final": round(float(res["cmax"]), 6),
        "improvement_pct": round(improvement_rate(c0, res["cmax"]), 4),
        "evals": res["evals"],
        "accepted_moves": res["accepted_moves"],
        "runtime_sec": round(elapsed, 6),
    }


def summarize(rows: list[dict]) -> list[dict]:
    """按 (n, m, init_method, searcher, max_evals) 分组，计算改进率均值与标准差。"""
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["n"], row["m"], row["init_method"],
            row["searcher"], row["max_evals"],
        )
        groups[key].append(row["improvement_pct"])

    summary = []
    for key in sorted(groups.keys()):
        vals = groups[key]
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary.append({
            "n": key[0],
            "m": key[1],
            "init_method": key[2],
            "searcher": key[3],
            "max_evals": key[4],
            "count": len(vals),
            "mean_improvement_pct": round(statistics.mean(vals), 4),
            "std_improvement_pct": round(std, 4),
        })
    return summary


def print_group_table(summary: list[dict]) -> None:
    """打印分组均值表，便于快速比较 fixed vs adaptive。"""
    print("\n=== 分组改进率均值（%）===")
    print(f"{'n':>6} {'m':>4} {'init':>12} {'budget':>7} | "
          f"{'plain':>8} {'fixed':>8} {'adaptive':>8} | {'adv-fix':>8}")
    key_order = ["plain", "fixed", "adaptive"]
    by_cell: dict[tuple, dict[str, float]] = defaultdict(dict)
    for g in summary:
        cell = (g["n"], g["m"], g["init_method"], g["max_evals"])
        by_cell[cell][g["searcher"]] = g["mean_improvement_pct"]
    for cell in sorted(by_cell.keys()):
        vals = by_cell[cell]
        row_vals = [vals.get(k, float("nan")) for k in key_order]
        diff = vals.get("adaptive", float("nan")) - vals.get("fixed", float("nan"))
        print(
            f"{cell[0]:>6} {cell[1]:>4} {cell[2]:>12} {cell[3]:>7} | "
            f"{row_vals[0]:>8.3f} {row_vals[1]:>8.3f} {row_vals[2]:>8.3f} | "
            f"{diff:>+8.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="自适应候选规模消融实验")
    parser.add_argument(
        "--sizes", nargs="+", default=None,
        help="实例规模列表，如 512x16 1024x32（缺省为 512x16 1024x32 2048x64）",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="快速模式：100x8、重复 1 次（用于快速验证脚本）",
    )
    parser.add_argument("--repeats", type=int, default=5, help="重复次数（缺省 5）")
    parser.add_argument(
        "--max-evals", type=int, nargs="+", default=DEFAULT_BUDGETS,
        help="每个搜索的评估预算（缺省 2000 5000 10000）",
    )
    parser.add_argument(
        "--out-csv", default="results/adaptive_ablation.csv",
        help="明细 CSV 输出路径（缺省 results/adaptive_ablation.csv）",
    )
    parser.add_argument(
        "--summary-json", default="results/adaptive_ablation_summary.json",
        help="汇总 JSON 输出路径（缺省 results/adaptive_ablation_summary.json）",
    )
    args = parser.parse_args()

    if args.fast:
        sizes = FAST_SIZES
        repeats = 1
    else:
        sizes = DEFAULT_SIZES if args.sizes is None else [parse_size(s) for s in args.sizes]
        repeats = args.repeats

    seeds = REPEAT_SEEDS[: max(repeats, 1)]
    rows: list[dict] = []
    total_runs = len(sizes) * len(INITIALS) * len(args.max_evals) * len(seeds) * len(SEARCHERS)
    print(f"规模: {sizes}, 预算: {args.max_evals}, 重复: {len(seeds)}, "
          f"共 {total_runs} 次搜索")

    t_start = time.perf_counter()
    done = 0
    for (n, m) in sizes:
        P = generate_synthetic_matrix(n, m, seed=MATRIX_SEED)
        print(f"[{n}x{m}] 矩阵已生成（seed={MATRIX_SEED}）")
        for init_name, init_fn in INITIALS.items():
            for budget in args.max_evals:
                for seed in seeds:
                    for searcher_name, searcher_fn in SEARCHERS.items():
                        row = run_cell(
                            P, init_name, init_fn,
                            searcher_name, searcher_fn,
                            budget, seed,
                        )
                        rows.append(row)
                        done += 1
                        if done % 50 == 0 or done == total_runs:
                            print(f"  进度 {done}/{total_runs} "
                                  f"({time.perf_counter() - t_start:.1f}s)")
    total_time = time.perf_counter() - t_start
    print(f"实验完成，总耗时 {total_time:.1f}s")

    out_csv = Path(args.out_csv)
    out_json = Path(args.summary_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV 明细已写入: {out_csv}（{len(rows)} 行）")

    summary = summarize(rows)
    payload = {
        "sizes": [[n, m] for (n, m) in sizes],
        "matrix_seed": MATRIX_SEED,
        "budgets": args.max_evals,
        "repeat_seeds": seeds,
        "repeats": len(seeds),
        "total_runs": total_runs,
        "total_runtime_sec": round(total_time, 3),
        "groups": summary,
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"JSON 汇总已写入: {out_json}（{len(summary)} 个分组）")

    print_group_table(summary)


if __name__ == "__main__":
    main()
