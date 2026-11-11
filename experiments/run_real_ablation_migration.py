"""V1.5 消融实验：BLS-M（仅迁移）在真实 HCSP 512x16 数据上的运行。

设计（严格复用 V1.3 已审计的主实验设置，仅新增 BLS-M 单元格）：
- 数据：512x16 全部 12 个矩阵（与 run_real_main.py --full 的 512 部分
  完全相同的文件集合）；
- 初解：min_min、round_robin；每个 (矩阵, 初解) 生成一次
  (a0, l0, c0)，两个预算共享（与 run_real_main.py 结构一致）；
- 搜索：仅 BLS-M = bottleneck_migration_search(K=4, R=2)（S 恒 0）；
- 预算：5000、10000 次候选评估；
- 0 值处理：与主实验一致，0 -> 1e-6；
- c_opt：官方 .sol 解中 makespan 最小者在实验矩阵上重算；
- 输出：results/real_ablation_blsm.csv（新增独立文件，不覆盖任何
  现有结果）+ results/real_ablation_blsm_summary.md。

CSV 列与 real_main.csv 一致，另加 termination 列：
- local_optimum：预算内无严格改善动作而停止；
- budget_exhausted：评估数达到预算上限。

用法：
    python experiments/run_real_ablation_migration.py
"""

from __future__ import annotations

import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import improvement_rate
from src.initial_solutions import min_min, round_robin
from src.loader import load_hcsp_matrix, load_hcsp_solution
from src.local_search_migration_only import bottleneck_migration_search

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "hcsp"
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_CSV = RESULTS_DIR / "real_ablation_blsm.csv"
OUT_SUMMARY = RESULTS_DIR / "real_ablation_blsm_summary.md"

K, R = 4, 2  # S 恒为 0（消融变体）
BUDGETS = [5000, 10000]
ZERO_EPS = 1e-6
SEED = 42  # 与主实验抽样/记录同源约定

INIT_FNS = {
    "min_min": lambda P: min_min(P),
    "round_robin": lambda P: round_robin(P),
}


def best_known_cmax(P: np.ndarray, mat: Path) -> float:
    """官方 .sol 解中 makespan 最小者在矩阵 P 上的值（best known）。

    与 run_real_main.py 同名函数一致：在实验矩阵（0 -> eps）上重算。
    """
    sols = sorted(mat.parent.glob(f"{mat.name}_*.sol"))
    best = float("inf")
    for sol in sols:
        a = load_hcsp_solution(str(sol))
        loads = np.array([P[a == j, j].sum() for j in range(P.shape[1])])
        best = min(best, float(loads.max()))
    return best


def select_datasets() -> list[Path]:
    """512x16 全部 12 个矩阵（排序，确定性）。"""
    sdir = DATA_ROOT / "512x16"
    return sorted(f for f in sdir.iterdir() if not f.name.endswith(".sol"))


def termination_of(res: dict, budget: int) -> str:
    """由返回结果推断终止原因。"""
    if res["evals"] >= budget:
        return "budget_exhausted"
    return "local_optimum"


def run_experiment(datasets: list[Path]) -> list[dict]:
    """对每个 (矩阵, 初解, 预算) 运行 BLS-M，返回 CSV 行。"""
    rows: list[dict] = []
    n_cells = len(datasets) * len(INIT_FNS) * len(BUDGETS)
    cell_no = 0
    for mat in datasets:
        P = load_hcsp_matrix(str(mat))
        n_zeros = int(np.count_nonzero(P == 0))
        P_exp = np.where(P > 0, P, ZERO_EPS)
        n, m = P_exp.shape
        c_opt = best_known_cmax(P_exp, mat)
        print(f"\n== {mat.relative_to(DATA_ROOT)} (n={n}, m={m}, "
              f"n_zeros={n_zeros}, c_opt={c_opt:.2f}) ==")
        for init_name, init_fn in INIT_FNS.items():
            # 与主实验一致：每个 (矩阵, 初解) 只生成一次初解，预算共享
            a0, l0, c0 = init_fn(P_exp)
            for budget in BUDGETS:
                t0 = time.perf_counter()
                res = bottleneck_migration_search(
                    P_exp, a0, l0, c0, K=K, R=R, max_evals=budget)
                rt = time.perf_counter() - t0
                c_final = float(res["cmax"])
                imp = improvement_rate(float(c0), c_final)
                gap0 = (float(c0) - c_opt) / max(c_opt, 1e-9) * 100.0
                gapf = (c_final - c_opt) / max(c_opt, 1e-9) * 100.0
                cell_no += 1
                rows.append({
                    "filepath": str(mat.relative_to(DATA_ROOT)),
                    "n": n,
                    "m": m,
                    "init_method": init_name,
                    "searcher": "bls_migration",
                    "max_evals": budget,
                    "c0": round(float(c0), 6),
                    "c_final": round(c_final, 6),
                    "c_opt": round(c_opt, 6),
                    "improvement_pct": round(imp, 6),
                    "gap_initial_pct": round(gap0, 6),
                    "gap_final_pct": round(gapf, 6),
                    "evals": int(res["evals"]),
                    "accepted_moves": int(res["accepted_moves"]),
                    "runtime_sec": round(rt, 6),
                    "seed": SEED,
                    "termination": termination_of(res, budget),
                })
                print(f"  [{cell_no}/{n_cells}] "
                      f"{init_name:<11} bls_migration b={budget:>5} | "
                      f"c0={float(c0):>12.2f} -> {c_final:>12.2f} | "
                      f"imp={imp:>6.2f}% gap0={gap0:>7.2f}% "
                      f"gapf={gapf:>7.2f}% | evals={res['evals']} "
                      f"term={termination_of(res, budget)} | {rt:>6.2f}s")
    return rows


def summarize(rows: list[dict], started: str) -> str:
    """生成 real_ablation_blsm_summary.md 内容。"""
    lines: list[str] = []
    lines.append("# BLS-M（仅迁移消融）运行汇总\n")
    lines.append(f"- 生成时间: {started}")
    lines.append(f"- 规模: 512x16 全部 12 个矩阵；初解: min_min、round_robin；"
                 f"预算: {BUDGETS}")
    lines.append(f"- 变体: bottleneck_migration_search(K={K}, R={R}, S=0)；"
                 f"seed={SEED}")
    lines.append(f"- 单元格数: {len(rows)}（12 x 2 x 2）\n")

    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["init_method"], r["max_evals"])
        groups.setdefault(key, []).append(r)

    lines.append("## 分组统计（init x budget）\n")
    lines.append("| init | budget | 单元格数 | 改进率均值±std(%) "
                 "| gap_final 均值±std(%) | evals 中位数 | "
                 "termination 分布 |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for key in sorted(groups):
        init, budget = key
        g = groups[key]
        imps = [r["improvement_pct"] for r in g]
        gaps = [r["gap_final_pct"] for r in g]
        evals = sorted(r["evals"] for r in g)
        terms = {}
        for r in g:
            terms[r["termination"]] = terms.get(r["termination"], 0) + 1
        lines.append(
            f"| {init} | {budget} | {len(g)} | "
            f"{statistics.mean(imps):.2f}±{statistics.pstdev(imps):.2f} | "
            f"{statistics.mean(gaps):.2f}±{statistics.pstdev(gaps):.2f} | "
            f"{evals[len(evals)//2]} | "
            f"{', '.join(f'{k}={v}' for k, v in sorted(terms.items()))} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    if not DATA_ROOT.is_dir():
        print(f"[FAIL] 数据目录不存在: {DATA_ROOT}")
        return 1

    datasets = select_datasets()
    if len(datasets) != 12:
        print(f"[FAIL] 期望 12 个 512x16 矩阵，实际: {len(datasets)}")
        return 1
    print(f"数据选取（512x16 全部，{len(datasets)} 个）:")
    for f in datasets:
        print(f"  - {f.relative_to(DATA_ROOT)}")

    rows = run_experiment(datasets)
    if len(rows) != 48:
        print(f"[FAIL] 期望 48 行，实际: {len(rows)}")
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV 已写入: {OUT_CSV.relative_to(PROJECT_ROOT)} "
          f"（{len(rows)} 行）")

    OUT_SUMMARY.write_text(
        summarize(rows, started), encoding="utf-8")
    print(f"汇总已写入: {OUT_SUMMARY.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
