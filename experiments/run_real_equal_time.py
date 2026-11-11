"""实验 A：真实 HCSP 数据上的等时间预算对比（plain vs bottleneck）。

设计：
- 数据：512x16 全部 12 个 + 1024x32 抽 4 个（u_c/u_i/u_s 各 >= 1、
  hihi/lolo 各 >= 1；固定 seed 可复现）。
- 初解：min_min、round_robin。
- 搜索：plain_local_search vs bottleneck_local_search(K=4, R=2, S=4)。
- 时间预算：1s、3s、5s。
- 实现：两个搜索都传 max_evals=10**9 与 time_budget=预算秒数
  （bottleneck 不获得任何额外时间；以先到者为准）。
- timed_out 判定（近似，如实说明）：runtime_sec >= time_budget_sec
  视为时间耗尽；若搜索在预算内因无改善动作提前收敛则 timed_out=False。
- 关键指标：bottleneck 更优配对比例 + plain/bottleneck 的实际评估次数
  （直接回应"优势是否来自更多评估"）。
- 输出：results/real_equal_time.csv 与 results/real_equal_time_summary.md。

0 值处理与 run_real_main.py 一致：官方 lolo/lohi 类矩阵含 0 值执行
时间，实验层替换为 1e-6（如实报告）。

用法：
    python experiments/run_real_equal_time.py --fast  # 512x16 抽 4 个
    python experiments/run_real_equal_time.py --full  # 正式实验
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import improvement_rate
from src.initial_solutions import min_min, round_robin
from src.loader import load_hcsp_matrix, load_hcsp_solution
from src.local_search import bottleneck_local_search, plain_local_search

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "hcsp"
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_CSV = RESULTS_DIR / "real_equal_time.csv"
OUT_SUMMARY = RESULTS_DIR / "real_equal_time_summary.md"
SYNTH_CSV = RESULTS_DIR / "verify_q2.csv"  # 合成数据 Q2（对比参照）

K, R, S = 4, 2, 4
TIME_BUDGETS = [1.0, 3.0, 5.0]
MAX_EVALS = 10**9  # 评估预算设极大，实际由 time_budget 决定停止
ZERO_EPS = 1e-6

INIT_FNS = {
    "min_min": lambda P: min_min(P),
    "round_robin": lambda P: round_robin(P),
}

SEARCHERS = {
    "plain": lambda P, a, l, c, tb: plain_local_search(
        P, a, l, c, max_evals=MAX_EVALS, time_budget=tb),
    "bottleneck": lambda P, a, l, c, tb: bottleneck_local_search(
        P, a, l, c, K=K, R=R, S=S, max_evals=MAX_EVALS, time_budget=tb),
}


def parse_category(name: str) -> tuple[str, str, str | None]:
    """解析矩阵文件名类别：返回 (u 类, 异质性, A/B 前缀或 None)。"""
    if name.startswith(("A.", "B.")):
        prefix, rest = name.split(".", 1)
    else:
        prefix, rest = None, name.rsplit(".", 1)[0]
    parts = rest.split("_")
    return "_".join(parts[:2]), parts[2], prefix


def sample_matrices(
    mats: list[Path], n_select: int, min_per_u: int, rng: np.random.Generator,
) -> list[Path]:
    """按类别约束抽样（与 run_real_main.py 相同实现，seed 可复现）。"""
    cats: dict[tuple[str, str], list[Path]] = {}
    for f in mats:
        u, het, _ = parse_category(f.name)
        cats.setdefault((u, het), []).append(f)
    classes = sorted(cats.keys())
    picked: list[tuple[str, str]] = []
    for _ in range(500):
        rng.shuffle(classes)
        picked = classes[:n_select]
        u_counts = Counter(c[0] for c in picked)
        het_counts = Counter(c[1] for c in picked)
        if (all(u_counts[u] >= min_per_u for u in ("u_c", "u_i", "u_s"))
                and het_counts["hihi"] >= 1 and het_counts["lolo"] >= 1):
            break
    else:
        raise RuntimeError("抽样约束无法满足（数据类别不完整？）")
    chosen: list[Path] = []
    for cls in picked:
        variants = cats[cls]
        chosen.append(variants[int(rng.integers(len(variants)))])
    return sorted(chosen)


def best_known_cmax(P: np.ndarray, mat: Path) -> float:
    """官方 .sol 解中 makespan 最小者在实验矩阵 P 上的值（best known）。"""
    sols = sorted(mat.parent.glob(f"{mat.name}_*.sol"))
    best = float("inf")
    for sol in sols:
        a = load_hcsp_solution(str(sol))
        loads = np.array([P[a == j, j].sum() for j in range(P.shape[1])])
        best = min(best, float(loads.max()))
    return best


def select_datasets(mode: str, rng: np.random.Generator) -> list[Path]:
    """按模式选择实验数据集，返回矩阵文件列表。"""
    selected: list[Path] = []
    if mode == "fast":
        mats = sorted(f for f in (DATA_ROOT / "512x16").iterdir()
                      if not f.name.endswith(".sol"))
        selected = sample_matrices(mats, n_select=4, min_per_u=1, rng=rng)
    else:
        sdir = DATA_ROOT / "512x16"
        selected += sorted(f for f in sdir.iterdir()
                           if not f.name.endswith(".sol"))
        mats = sorted(f for f in (DATA_ROOT / "1024x32").iterdir()
                      if not f.name.endswith(".sol"))
        selected += sample_matrices(mats, n_select=4, min_per_u=1, rng=rng)
    return selected


def run_experiment(datasets: list[Path], seed: int) -> list[dict]:
    """对每个 (矩阵, 初解, 搜索, 时间预算) 单元格运行实验，返回 CSV 行。"""
    rows: list[dict] = []
    n_cells_total = len(datasets) * len(INIT_FNS) * len(SEARCHERS) \
        * len(TIME_BUDGETS)
    cell_no = 0
    for mat in datasets:
        try:
            P = load_hcsp_matrix(str(mat))
        except Exception as exc:  # noqa: BLE001 —— 加载失败不静默
            print(f"[FAIL] 矩阵加载失败，跳过并计数: {mat.name} ({exc})")
            continue
        n_zeros = int(np.count_nonzero(P == 0))
        P_exp = np.where(P > 0, P, ZERO_EPS)
        n, m = P_exp.shape
        try:
            c_opt = best_known_cmax(P_exp, mat)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {mat.name}: 最优解参考值解析失败 ({exc})，跳过")
            continue
        rel = mat.relative_to(DATA_ROOT)
        print(f"\n== {rel} (n={n}, m={m}, n_zeros={n_zeros}, "
              f"c_opt={c_opt:.2f}) ==")
        for init_name, init_fn in INIT_FNS.items():
            a0, l0, c0 = init_fn(P_exp)
            for tb in TIME_BUDGETS:
                for searcher_name, searcher_fn in SEARCHERS.items():
                    t0 = time.perf_counter()
                    res = searcher_fn(P_exp, a0, l0, float(c0), tb)
                    rt = time.perf_counter() - t0
                    c_final = float(res["cmax"])
                    imp = improvement_rate(float(c0), c_final)
                    gap0 = (float(c0) - c_opt) / max(c_opt, 1e-9) * 100.0
                    gapf = (c_final - c_opt) / max(c_opt, 1e-9) * 100.0
                    evals = int(res["evals"])
                    # 近似判定：运行时长达到时间预算即视为时间耗尽；
                    # 预算内收敛（无改善动作）提前停止则 timed_out=False。
                    timed_out = rt >= tb
                    cell_no += 1
                    rows.append({
                        "filepath": str(rel),
                        "n": n,
                        "m": m,
                        "init_method": init_name,
                        "searcher": searcher_name,
                        "time_budget_sec": tb,
                        "c0": round(float(c0), 6),
                        "c_final": round(c_final, 6),
                        "c_opt": round(c_opt, 6),
                        "improvement_pct": round(imp, 6),
                        "gap_initial_pct": round(gap0, 6),
                        "gap_final_pct": round(gapf, 6),
                        "evals_actual": evals,
                        "accepted_moves": int(res["accepted_moves"]),
                        "runtime_sec": round(rt, 6),
                        "timed_out": 1 if timed_out else 0,
                        "seed": seed,
                    })
                    print(f"  [{cell_no}/{n_cells_total}] "
                          f"{init_name:<11} {searcher_name:<10} "
                          f"tb={tb:>3}g | c0={float(c0):>12.2f} -> "
                          f"{c_final:>12.2f} | imp={imp:>6.2f}% "
                          f"gapf={gapf:>7.2f}% | evals={evals:>9} "
                          f"| {rt:>6.2f}s{' (timeout)' if timed_out else ''}")
    return rows


def summarize(rows: list[dict], datasets: list[Path],
              mode: str, seed: int, started: str) -> str:
    """生成 real_equal_time_summary.md 内容。"""
    if not rows:
        return "# 真实 HCSP 数据等时间预算对比汇总\n\n（无有效数据行）\n"

    lines: list[str] = []
    lines.append("# 实验 A：真实 HCSP 数据等时间预算对比汇总\n")
    lines.append(f"- 生成时间: {started}")
    lines.append(f"- 模式: {mode}（seed={seed}）")
    lines.append(f"- 初解: min_min、round_robin；"
                 f"搜索: plain vs bottleneck(K={K},R={R},S={S})；"
                 f"时间预算: {TIME_BUDGETS} 秒")
    lines.append(f"- 实现: 两搜索均传 max_evals=1e9 与 time_budget"
                 f"（bottleneck 无额外时间；先到者为准）")
    lines.append(f"- timed_out 判定（近似）: runtime_sec >= time_budget_sec "
                 f"视为时间耗尽；预算内因无改善动作提前收敛则记为 0")
    lines.append(f"- 0 值处理: 官方 lolo/lohi 类矩阵含 0 值执行时间，"
                 f"实验层替换为 {ZERO_EPS:g}（如实报告，不改变排序关系）\n")
    lines.append(f"- CSV 行数: {len(rows)}\n")

    lines.append("## 数据选取\n")
    by_size: dict[str, list[str]] = {}
    for f in datasets:
        by_size.setdefault(f.parent.name, []).append(f.name)
    for size in sorted(by_size):
        lines.append(f"- {size}（{len(by_size[size])} 个）: "
                     f"{', '.join(by_size[size])}")
    lines.append("")

    # ---- 分组统计 ----
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["n"], r["m"], r["init_method"], r["searcher"],
               r["time_budget_sec"])
        groups.setdefault(key, []).append(r)

    lines.append("## 分组统计（size x init x searcher x time_budget）\n")
    lines.append("| size | init | searcher | tb(s) | 单元格数 | 改进率均值±std(%) "
                 "| gap_final 均值±std(%) | 平均 evals_actual | timed_out 比例 |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    for key in sorted(groups):
        n, m, init, searcher, tb = key
        g = groups[key]
        imps = [r["improvement_pct"] for r in g]
        gaps = [r["gap_final_pct"] for r in g]
        evals = [r["evals_actual"] for r in g]
        to_ratio = sum(r["timed_out"] for r in g) / len(g)
        lines.append(
            f"| {n}x{m} | {init} | {searcher} | {tb:g} | {len(g)} | "
            f"{statistics.mean(imps):.2f}±{statistics.pstdev(imps):.2f} | "
            f"{statistics.mean(gaps):.2f}±{statistics.pstdev(gaps):.2f} | "
            f"{statistics.mean(evals):.0f} | {to_ratio:.0%} |"
        )
    lines.append("")

    # ---- bottleneck 优于 plain 配对比例 ----
    lines.append("## bottleneck 优于 plain 的配对比例\n")
    lines.append("（同 (filepath, init, tb) 配对，c_final 严格更小"
                 "（1e-9 容差）计为更优）\n")
    lines.append("| size | init | tb(s) | 配对数 | bottleneck 更优比例 |")
    lines.append("|---|---|---:|---:|---:|")
    pair_keys = sorted({(r["n"], r["m"], r["init_method"],
                         r["time_budget_sec"]) for r in rows})
    for n, m, init, tb in pair_keys:
        pairs: dict[str, dict[str, float]] = {}
        for r in rows:
            if (r["n"], r["m"], r["init_method"], r["time_budget_sec"]) != \
                    (n, m, init, tb):
                continue
            pairs.setdefault(r["filepath"], {})[r["searcher"]] = r["c_final"]
        n_pairs = 0
        n_better = 0
        for cpair in pairs.values():
            if "plain" not in cpair or "bottleneck" not in cpair:
                continue
            n_pairs += 1
            if cpair["bottleneck"] < cpair["plain"] - 1e-9:
                n_better += 1
        if n_pairs:
            lines.append(
                f"| {n}x{m} | {init} | {tb:g} | {n_pairs} | "
                f"{n_better}/{n_pairs} ({100 * n_better / n_pairs:.1f}%) |"
            )
    lines.append("")

    # ---- evals 对比（关键指标）----
    lines.append("## 评估次数对比（关键指标：优势是否来自更多评估）\n")
    lines.append("| size | init | tb(s) | plain 平均 evals | bottleneck 平均 evals "
                 "| 倍数(plain/bn) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for n, m, init, tb in pair_keys:
        p_ev = [r["evals_actual"] for r in rows
                if (r["n"], r["m"], r["init_method"], r["time_budget_sec"])
                == (n, m, init, tb) and r["searcher"] == "plain"]
        b_ev = [r["evals_actual"] for r in rows
                if (r["n"], r["m"], r["init_method"], r["time_budget_sec"])
                == (n, m, init, tb) and r["searcher"] == "bottleneck"]
        if p_ev and b_ev:
            p_mean = statistics.mean(p_ev)
            b_mean = statistics.mean(b_ev)
            ratio = p_mean / max(b_mean, 1.0)
            lines.append(f"| {n}x{m} | {init} | {tb:g} | {p_mean:.0f} | "
                         f"{b_mean:.0f} | {ratio:.1f}x |")
    lines.append("")
    lines.append("- 解读：等时间预算下 plain 的单次评估更快、总评估次数更多；"
                 "若 bottleneck 仍更优，说明优势来自候选筛选质量而非评估数量。")
    lines.append("")

    # ---- 与合成 Q2 对比 ----
    lines.append("## 与合成数据（verify_q2）对比\n")
    lines.append("注意：verify_q2 用 bottleneck(K=8,R=4,S=8)；本实验用 "
                 f"(K={K},R={R},S={S})。仅对比双方共同覆盖的 512x16 单元格，"
                 "量级参考。\n")
    if SYNTH_CSV.is_file():
        with SYNTH_CSV.open(newline="", encoding="utf-8") as f:
            synth = list(csv.DictReader(f))
        lines.append("| 数据 | init | tb(s) | plain 改进率均值(%) "
                     "| bottleneck 改进率均值(%) |")
        lines.append("|---|---|---:|---:|---:|")
        for init in ("min_min", "round_robin"):
            for tb in TIME_BUDGETS:
                s_rows = [r for r in synth
                          if int(r["n"]) == 512 and int(r["m"]) == 16
                          and r["init_method"] == init
                          and abs(float(r["time_budget"]) - tb) < 1e-9]
                r_rows = [r for r in rows
                          if r["n"] == 512 and r["m"] == 16
                          and r["init_method"] == init
                          and abs(r["time_budget_sec"] - tb) < 1e-9]
                if not s_rows:
                    continue
                s_p = statistics.mean(
                    float(r["improvement_plain_pct"]) for r in s_rows)
                s_b = statistics.mean(
                    float(r["improvement_bottleneck_pct"]) for r in s_rows)
                r_p = statistics.mean(r["improvement_pct"] for r in r_rows
                                      if r["searcher"] == "plain")
                r_b = statistics.mean(r["improvement_pct"] for r in r_rows
                                      if r["searcher"] == "bottleneck")
                lines.append(f"| 合成(seed=42) | {init} | {tb:g} | "
                             f"{s_p:.2f} | {s_b:.2f} |")
                lines.append(f"| 真实 | {init} | {tb:g} | "
                             f"{r_p:.2f} | {r_b:.2f} |")
        lines.append("")
    else:
        lines.append("（未找到 results/verify_q2.csv，跳过对比）\n")

    # ---- 提前停止如实记录 ----
    early = [r for r in rows if not r["timed_out"]]
    lines.append("## 提前停止（无改善动作收敛）情况\n")
    if early:
        by_pair: dict[tuple, int] = {}
        for r in early:
            key = (r["n"], r["m"], r["init_method"], r["searcher"],
                   r["time_budget_sec"])
            by_pair[key] = by_pair.get(key, 0) + 1
        lines.append("- 预算内提前收敛的单元格按 (size, init, searcher, tb) 统计：")
        for key in sorted(by_pair):
            n, m, init, searcher, tb = key
            lines.append(f"  - {n}x{m} / {init} / {searcher} / {tb:g}s: "
                         f"{by_pair[key]} 个")
    else:
        lines.append("- 所有单元格均耗尽时间预算（无提前收敛）。")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="实验 A：真实 HCSP 数据等时间预算对比")
    parser.add_argument("--fast", action="store_true",
                        help="只用 512x16 的 4 个矩阵（快速验证流程）")
    parser.add_argument("--full", action="store_true",
                        help="正式实验：512x16 全部 + 1024x32 抽 4")
    parser.add_argument("--seed", type=int, default=42,
                        help="抽样随机种子（默认 42，可复现）")
    args = parser.parse_args()
    if args.fast == args.full:
        parser.error("必须指定 --fast 或 --full 之一")

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    rng = np.random.default_rng(args.seed)
    mode = "fast" if args.fast else "full"

    if not DATA_ROOT.is_dir():
        print(f"[FAIL] 数据目录不存在: {DATA_ROOT}")
        return 1

    datasets = select_datasets(mode, rng)
    print(f"数据选取（mode={mode}, seed={args.seed}）:")
    for f in datasets:
        print(f"  - {f.relative_to(DATA_ROOT)}")
    n_cells = len(datasets) * len(INIT_FNS) * len(SEARCHERS) * len(TIME_BUDGETS)
    print(f"共 {len(datasets)} 个矩阵，{n_cells} 个单元格")

    rows = run_experiment(datasets, args.seed)
    if not rows:
        print("[FAIL] 没有产生任何有效实验行。")
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV 已写入: {OUT_CSV.relative_to(PROJECT_ROOT)} "
          f"（{len(rows)} 行）")

    summary = summarize(rows, datasets, mode, args.seed, started)
    OUT_SUMMARY.write_text(summary, encoding="utf-8")
    print(f"汇总已写入: {OUT_SUMMARY.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
