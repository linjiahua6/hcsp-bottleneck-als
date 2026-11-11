"""实验 B：真实 HCSP 数据上的参数敏感性（K/R/S 网格）。

设计：
- 数据：512x16 抽 4 个（u_c/u_i/u_s 各 >= 1、hihi/lolo 各 >= 1；
  固定 seed 可复现），控制总时长。
- 初解：min_min、round_robin。
- 参数网格：K ∈ {4, 8, 16}，R ∈ {2, 4}，S ∈ {4, 8}，共 12 组。
- 预算：10000 次评估。
- 输出：results/real_param_sensitivity.csv 与
  results/real_param_sensitivity_summary.md。
- 研究问题：合成数据 Q5 上 (4,2,4) 最优；真实数据上最优参数是否
  仍为 (4,2,4)？如实报告。

0 值处理与 run_real_main.py 一致：官方 lolo/lohi 类矩阵含 0 值执行
时间，实验层替换为 1e-6（如实报告）。

用法：
    python experiments/run_real_param_sensitivity.py [--seed 42]
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
from src.local_search import bottleneck_local_search

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "hcsp"
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_CSV = RESULTS_DIR / "real_param_sensitivity.csv"
OUT_SUMMARY = RESULTS_DIR / "real_param_sensitivity_summary.md"
SYNTH_CSV = RESULTS_DIR / "verify_q5.csv"  # 合成数据 Q5（对比参照）

BUDGET = 10000
ZERO_EPS = 1e-6
GRID = [(K, R, S) for K in (4, 8, 16) for R in (2, 4) for S in (4, 8)]
BASE = (4, 2, 4)  # 合成 Q5 最优、真实主实验采用的基准

INIT_FNS = {
    "min_min": lambda P: min_min(P),
    "round_robin": lambda P: round_robin(P),
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


def select_datasets(rng: np.random.Generator) -> list[Path]:
    """512x16 抽 4 个（u_c/u_i/u_s 各 >= 1、hihi/lolo 各 >= 1）。"""
    mats = sorted(f for f in (DATA_ROOT / "512x16").iterdir()
                  if not f.name.endswith(".sol"))
    return sample_matrices(mats, n_select=4, min_per_u=1, rng=rng)


def run_experiment(datasets: list[Path], seed: int) -> list[dict]:
    """对每个 (矩阵, 初解, K, R, S) 单元格运行实验，返回 CSV 行。"""
    rows: list[dict] = []
    n_cells_total = len(datasets) * len(INIT_FNS) * len(GRID)
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
        rel = mat.relative_to(DATA_ROOT)
        print(f"\n== {rel} (n={n}, m={m}, n_zeros={n_zeros}) ==")
        for init_name, init_fn in INIT_FNS.items():
            a0, l0, c0 = init_fn(P_exp)
            for K, R, S in GRID:
                t0 = time.perf_counter()
                res = bottleneck_local_search(
                    P_exp, a0, l0, float(c0), K=K, R=R, S=S,
                    max_evals=BUDGET)
                rt = time.perf_counter() - t0
                c_final = float(res["cmax"])
                imp = improvement_rate(float(c0), c_final)
                cell_no += 1
                rows.append({
                    "filepath": str(rel),
                    "n": n,
                    "m": m,
                    "init_method": init_name,
                    "K": K,
                    "R": R,
                    "S": S,
                    "c0": round(float(c0), 6),
                    "c_final": round(c_final, 6),
                    "improvement_pct": round(imp, 6),
                    "evals": int(res["evals"]),
                    "runtime_sec": round(rt, 6),
                    "seed": seed,
                })
                print(f"  [{cell_no}/{n_cells_total}] "
                      f"{init_name:<11} K={K:>2} R={R} S={S} | "
                      f"c0={float(c0):>12.2f} -> {c_final:>12.2f} | "
                      f"imp={imp:>6.2f}% | evals={res['evals']:>6} | "
                      f"{rt:>6.2f}s")
    return rows


def _group_imp(rows: list[dict], init: str, key_fn) -> dict[tuple, float]:
    """按 key_fn 分组，返回 {key: 改进率均值}（只统计给定初解）。"""
    groups: dict[tuple, list[float]] = {}
    for r in rows:
        if r["init_method"] != init:
            continue
        groups.setdefault(key_fn(r), []).append(r["improvement_pct"])
    return {k: statistics.mean(v) for k, v in groups.items()}


def _imp_table(rows: list[dict], init: str, keys: list[tuple]) -> str:
    """输出给定初解在参数子集上的改进率均值表。"""
    imp = _group_imp(rows, init, lambda r: (r["K"], r["R"], r["S"]))
    header = "| K | R | S | 改进率均值(%) |"
    sep = "|---|---|---:|---:|"
    body = [f"| {K} | {R} | {S} | {imp.get((K, R, S), float('nan')):.2f} |"
            for (K, R, S) in keys if (K, R, S) in imp]
    return "\n".join([header, sep] + body)


def summarize(rows: list[dict], datasets: list[Path],
              seed: int, started: str) -> str:
    """生成 real_param_sensitivity_summary.md 内容。"""
    if not rows:
        return "# 真实 HCSP 数据参数敏感性汇总\n\n（无有效数据行）\n"

    lines: list[str] = []
    lines.append("# 实验 B：真实 HCSP 数据参数敏感性汇总\n")
    lines.append(f"- 生成时间: {started}")
    lines.append(f"- 数据: 512x16 抽 {len(datasets)} 个（seed={seed}）")
    lines.append(f"- 初解: min_min、round_robin；预算: {BUDGET}；"
                 f"参数网格: K∈{{4,8,16}} × R∈{{2,4}} × S∈{{4,8}}（{len(GRID)} 组）")
    lines.append(f"- 0 值处理: 官方 lolo/lohi 类矩阵含 0 值执行时间，"
                 f"实验层替换为 {ZERO_EPS:g}（如实报告，不改变排序关系）\n")
    lines.append(f"- CSV 行数: {len(rows)}\n")

    lines.append("## 数据选取\n")
    lines.append(f"- 512x16（{len(datasets)} 个）: "
                 f"{', '.join(f.name for f in datasets)}\n")

    # ---- 全网格总表 ----
    lines.append("## 全参数网格改进率均值（size=512x16）\n")
    lines.append("| init | K | R | S | 改进率均值(%) | 相对(4,2,4)差(pp) |")
    lines.append("|---|---|---|---:|---:|---:|")
    for init in ("min_min", "round_robin"):
        imp = _group_imp(rows, init, lambda r: (r["K"], r["R"], r["S"]))
        base = imp.get(BASE, float("nan"))
        for (K, R, S) in GRID:
            v = imp.get((K, R, S), float("nan"))
            diff = v - base
            lines.append(f"| {init} | {K} | {R} | {S} | {v:.2f} | "
                         f"{diff:+.2f} |")
    lines.append("")

    # ---- 分四部分：K / R / S / 最优组合 ----
    lines.append("## 1. 固定 R=2, S=4，比较 K\n")
    for init in ("min_min", "round_robin"):
        lines.append(f"### {init}\n")
        lines.append(_imp_table(rows, init,
                                [(K, 2, 4) for K in (4, 8, 16)]))
        lines.append("")
    lines.append("## 2. 固定 K=4, S=4，比较 R\n")
    for init in ("min_min", "round_robin"):
        lines.append(f"### {init}\n")
        lines.append(_imp_table(rows, init, [(4, R, 4) for R in (2, 4)]))
        lines.append("")
    lines.append("## 3. 固定 K=4, R=2，比较 S\n")
    for init in ("min_min", "round_robin"):
        lines.append(f"### {init}\n")
        lines.append(_imp_table(rows, init, [(4, 2, S) for S in (4, 8)]))
        lines.append("")

    # ---- 最优组合与 (4,2,4) 的差距 ----
    lines.append("## 4. 最优组合与 (4,2,4) 的差距\n")
    for init in ("min_min", "round_robin"):
        imp = _group_imp(rows, init, lambda r: (r["K"], r["R"], r["S"]))
        best = max(imp.items(), key=lambda kv: kv[1])
        base = imp[BASE]
        lines.append(f"- {init}: 最优组合 {best[0]}（改进率均值 {best[1]:.2f}%），"
                     f"(4,2,4) 为 {base:.2f}%，差距 {best[1] - base:+.2f}pp")
    overall = {}
    for r in rows:
        overall.setdefault((r["K"], r["R"], r["S"]), []).append(
            r["improvement_pct"])
    best_all = max(overall.items(), key=lambda kv: statistics.mean(kv[1]))
    base_all = statistics.mean(overall[BASE])
    lines.append(f"- 合并初解: 最优组合 {best_all[0]}（均值 "
                 f"{statistics.mean(best_all[1]):.2f}%），(4,2,4) 为 "
                 f"{base_all:.2f}%，差距 "
                 f"{statistics.mean(best_all[1]) - base_all:+.2f}pp")
    if best_all[0] == BASE or abs(statistics.mean(best_all[1]) - base_all) < 0.5:
        lines.append("- 判定: (4,2,4) 在真实数据上**仍为最优或接近最优**"
                     "（与合成 Q5 一致，如实报告）")
    else:
        lines.append(f"- 判定: 真实数据上最优为 {best_all[0]}，与合成 Q5 的 "
                     "(4,2,4) 不同（如实报告）")
    lines.append("")

    # ---- 与合成 Q5 对比 ----
    lines.append("## 5. 与合成数据 Q5（verify_q5）对比\n")
    if SYNTH_CSV.is_file():
        with SYNTH_CSV.open(newline="", encoding="utf-8") as f:
            synth = list(csv.DictReader(f))
        lines.append("| 数据 | init | K | R | S | 改进率均值(%) |")
        lines.append("|---|---|---|---|---:|---:|")
        synth_groups: dict[tuple, list[float]] = {}
        for r in synth:
            key = (r["init_method"], int(r["K"]), int(r["R"]), int(r["S"]))
            if key[1:] in GRID:
                synth_groups.setdefault(key, []).append(
                    float(r["improvement_pct"]))
        for init in ("min_min", "round_robin"):
            real_imp = _group_imp(rows, init, lambda r: (r["K"], r["R"], r["S"]))
            for (K, R, S) in GRID:
                if (init, K, R, S) not in synth_groups:
                    continue
                s_mean = statistics.mean(synth_groups[(init, K, R, S)])
                r_mean = real_imp.get((K, R, S), float("nan"))
                lines.append(f"| 合成(seed=42) | {init} | {K} | {R} | {S} | "
                             f"{s_mean:.2f} |")
                lines.append(f"| 真实 | {init} | {K} | {R} | {S} | "
                             f"{r_mean:.2f} |")
        lines.append("")
        lines.append("- 注：verify_q5 的初解与实例规模与本实验不完全一致，"
                     "仅按 (init, K, R, S) 对齐作量级参考。")
        lines.append("")
    else:
        lines.append("（未找到 results/verify_q5.csv，跳过对比）\n")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="实验 B：真实 HCSP 数据参数敏感性")
    parser.add_argument("--seed", type=int, default=42,
                        help="抽样随机种子（默认 42，可复现）")
    args = parser.parse_args()

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    rng = np.random.default_rng(args.seed)

    if not DATA_ROOT.is_dir():
        print(f"[FAIL] 数据目录不存在: {DATA_ROOT}")
        return 1

    datasets = select_datasets(rng)
    print(f"数据选取（seed={args.seed}）:")
    for f in datasets:
        print(f"  - {f.relative_to(DATA_ROOT)}")
    n_cells = len(datasets) * len(INIT_FNS) * len(GRID)
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

    summary = summarize(rows, datasets, args.seed, started)
    OUT_SUMMARY.write_text(summary, encoding="utf-8")
    print(f"汇总已写入: {OUT_SUMMARY.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
