"""Q6 在真实 HCSP 数据上的重测（破坏-重建机制 DR）。

设计：
- 数据：512x16 全部 12 个 + 1024x32 抽 4 个（u_c/u_i/u_s 各 >= 1、
  hihi/lolo 各 >= 1；固定 seed 可复现）。
- 初解：min_min、round_robin。
- 搜索对比（K=4, R=2, S=4，Q5 最优配置）：
  A: bottleneck_local_search（基线）
  B: bottleneck_local_search_dr（strict 接受准则，默认参数
     destroy_frac=0.2, destroy_strategy="bottleneck",
     segment_evals=1000, max_stagnation=3, seed=42）
- 预算：5000、10000 次评估。
- 输出：results/real_q6.csv（含 seed 列）与 results/real_q6_summary.md。

研究问题：合成数据（verify_q6）上 DR 无正贡献（mean B-A = -7.47pp）；
真实数据按类别（u_c=consistent、u_i=inconsistent、u_s=semi-consistent）
分组重测，检验 DR 是否在 inconsistent（u_i）类上有效。如实报告，
不隐藏负结果。

0 值处理与 run_real_main.py 一致：官方 lolo/lohi 类矩阵含 0 值执行
时间，实验层替换为 1e-6（如实报告）。

用法：
    python experiments/run_real_q6.py [--seed 42]
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

from src.destroy_repair import bottleneck_local_search_dr
from src.evaluate import improvement_rate
from src.initial_solutions import min_min, round_robin
from src.loader import load_hcsp_matrix, load_hcsp_solution
from src.local_search import bottleneck_local_search

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "hcsp"
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_CSV = RESULTS_DIR / "real_q6.csv"
OUT_SUMMARY = RESULTS_DIR / "real_q6_summary.md"
SYNTH_SUMMARY = RESULTS_DIR / "verify_q6_phase1_summary.md"

K, R, S = 4, 2, 4
BUDGETS = [5000, 10000]
ZERO_EPS = 1e-6
DR_SEED = 42

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
    """按类别约束抽样（与 run_real_main.py 相同实现）。"""
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
    selected = sorted(f for f in (DATA_ROOT / "512x16").iterdir()
                      if not f.name.endswith(".sol"))
    mats = sorted(f for f in (DATA_ROOT / "1024x32").iterdir()
                  if not f.name.endswith(".sol"))
    selected += sample_matrices(mats, n_select=4, min_per_u=1, rng=rng)
    return selected


def best_known_cmax(P: np.ndarray, mat: Path) -> float:
    sols = sorted(mat.parent.glob(f"{mat.name}_*.sol"))
    best = float("inf")
    for sol in sols:
        a = load_hcsp_solution(str(sol))
        loads = np.array([P[a == j, j].sum() for j in range(P.shape[1])])
        best = min(best, float(loads.max()))
    return best


def run_experiment(datasets: list[Path], seed: int) -> list[dict]:
    rows: list[dict] = []
    n_total = len(datasets) * len(INIT_FNS) * len(BUDGETS) * 2
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
        u_cat, het, _ = parse_category(mat.name)
        try:
            c_opt = best_known_cmax(P_exp, mat)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {mat.name}: 最优解参考值解析失败 ({exc})，跳过")
            continue
        print(f"\n== {mat.relative_to(DATA_ROOT)} (cat={u_cat}/{het}, "
              f"n={n}, m={m}, n_zeros={n_zeros}, c_opt={c_opt:.2f}) ==")
        for init_name, init_fn in INIT_FNS.items():
            a0, l0, c0 = init_fn(P_exp)
            for budget in BUDGETS:
                results = {}
                t0 = time.perf_counter()
                res_a = bottleneck_local_search(
                    P_exp, a0, l0, c0, K=K, R=R, S=S, max_evals=budget)
                results["A"] = (res_a, time.perf_counter() - t0)
                t0 = time.perf_counter()
                res_b = bottleneck_local_search_dr(
                    P_exp, a0, l0, c0, K=K, R=R, S=S, max_evals=budget,
                    accept_strategy="strict", seed=DR_SEED)
                results["B"] = (res_b, time.perf_counter() - t0)
                for tag, (res, rt) in results.items():
                    c_final = float(res["cmax"])
                    imp = improvement_rate(float(c0), c_final)
                    gap0 = (float(c0) - c_opt) / max(c_opt, 1e-9) * 100.0
                    gapf = (c_final - c_opt) / max(c_opt, 1e-9) * 100.0
                    cell_no += 1
                    rows.append({
                        "filepath": str(mat.relative_to(DATA_ROOT)),
                        "n": n,
                        "m": m,
                        "category": u_cat,
                        "heterogeneity": het,
                        "init_method": init_name,
                        "searcher": tag,
                        "max_evals": budget,
                        "c0": round(float(c0), 6),
                        "c_final": round(c_final, 6),
                        "c_opt": round(c_opt, 6),
                        "improvement_pct": round(imp, 6),
                        "gap_initial_pct": round(gap0, 6),
                        "gap_final_pct": round(gapf, 6),
                        "evals": int(res["evals"]),
                        "accepted_moves": int(res["accepted_moves"]),
                        "restarts": int(res.get("restarts", 0)),
                        "improvements": int(res.get("improvements", 0)),
                        "runtime_sec": round(rt, 6),
                        "seed": seed,
                    })
                    print(f"  [{cell_no}/{n_total}] {init_name:<11} "
                          f"{tag} b={budget:>5} | c0={float(c0):>12.2f} -> "
                          f"{c_final:>12.2f} | imp={imp:>7.2f}% "
                          f"gapf={gapf:>7.2f}% | evals={res['evals']} "
                          f"segs={res.get('segments', '-')} | {rt:>6.2f}s")
    return rows


def summarize(rows: list[dict], datasets: list[Path],
              seed: int, started: str) -> str:
    if not rows:
        return "# Q6 真实数据重测汇总\n\n（无有效数据行）\n"

    lines: list[str] = []
    lines.append("# Q6 破坏-重建机制在真实 HCSP 数据上的重测汇总\n")
    lines.append(f"- 生成时间: {started}（抽样 seed={seed}，DR seed={DR_SEED}）")
    lines.append(f"- 数据: 512x16 全部 12 个 + 1024x32 抽 4 个"
                 f"（{len(datasets)} 个矩阵）")
    lines.append(f"- 初解: min_min、round_robin；预算: {BUDGETS}")
    lines.append(f"- A: bottleneck_local_search(K={K},R={R},S={S})；"
                 f"B: bottleneck_local_search_dr(K={K},R={R},S={S}, strict, "
                 f"destroy_frac=0.2, segment_evals=1000, max_stagnation=3)")
    lines.append("")

    lines.append("## 数据选取\n")
    by_size: dict[str, list[str]] = {}
    for f in datasets:
        by_size.setdefault(f.parent.name, []).append(f.name)
    for size in sorted(by_size):
        lines.append(f"- {size}（{len(by_size[size])} 个）: "
                     f"{', '.join(by_size[size])}")
    lines.append("")

    # ---- 总体 ----
    pairs: dict[tuple, dict[str, float]] = {}
    for r in rows:
        key = (r["filepath"], r["init_method"], r["max_evals"])
        pairs.setdefault(key, {})[r["searcher"]] = r["improvement_pct"]
    diffs = [p["B"] - p["A"] for p in pairs.values()
             if "A" in p and "B" in p]
    n_better = sum(1 for d in diffs if d > 1e-9)
    n_worse = sum(1 for d in diffs if d < -1e-9)
    lines.append("## 总体结果（A vs B）\n")
    lines.append(f"- 配对单元格数: {len(diffs)}")
    lines.append(f"- mean(B-A) = {statistics.mean(diffs):+.2f}pp；"
                 f"B 更优 {n_better} 格、劣化 {n_worse} 格、"
                 f"持平 {len(diffs) - n_better - n_worse} 格")
    lines.append("")

    # ---- 按类别 × 初解 × 预算 ----
    lines.append("## 按类别分组（category x init x budget）\n")
    lines.append("| category | init | budget | 配对数 | imp A 均值 | imp B 均值 "
                 "| B-A (pp) | B 更优 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for cat in ("u_c", "u_i", "u_s"):
        for init in ("min_min", "round_robin"):
            for budget in BUDGETS:
                sub = {}
                for r in rows:
                    if (r["category"] == cat and r["init_method"] == init
                            and r["max_evals"] == budget):
                        sub.setdefault(
                            r["filepath"], {})[r["searcher"]] = r
                d_cat = [v["B"]["improvement_pct"]
                         - v["A"]["improvement_pct"]
                         for v in sub.values() if "A" in v and "B" in v]
                if not d_cat:
                    continue
                imp_a = statistics.mean(
                    v["A"]["improvement_pct"] for v in sub.values()
                    if "A" in v)
                imp_b = statistics.mean(
                    v["B"]["improvement_pct"] for v in sub.values()
                    if "B" in v)
                n_b = sum(1 for d in d_cat if d > 1e-9)
                lines.append(
                    f"| {cat} | {init} | {budget} | {len(d_cat)} | "
                    f"{imp_a:.2f} | {imp_b:.2f} | "
                    f"{statistics.mean(d_cat):+.2f} | "
                    f"{n_b}/{len(d_cat)} |"
                )
    lines.append("")

    # ---- u_i（inconsistent）重点 ----
    lines.append("## 假设检验：DR 是否在 inconsistent 类（u_i）上有效\n")
    sub_ui = {}
    for r in rows:
        if r["category"] == "u_i":
            sub_ui.setdefault((r["filepath"], r["init_method"],
                               r["max_evals"]), {})[r["searcher"]] = r
    d_ui = [v["B"]["improvement_pct"] - v["A"]["improvement_pct"]
            for v in sub_ui.values() if "A" in v and "B" in v]
    if d_ui:
        n_b = sum(1 for d in d_ui if d > 1e-9)
        lines.append(f"- u_i 配对单元格 {len(d_ui)} 个，mean(B-A) = "
                     f"{statistics.mean(d_ui):+.2f}pp，"
                     f"B 更优 {n_b}/{len(d_ui)}")
        verdict = ("支持假设（DR 在 inconsistent 类上有效）"
                   if statistics.mean(d_ui) > 0.0 and n_b > len(d_ui) / 2
                   else "不支持假设（DR 在 u_i 上未表现出正贡献）")
        lines.append(f"- 判定: {verdict}（如实报告，不含蓄意取舍）")
    lines.append("")

    # ---- 与合成 Q6 对比 ----
    lines.append("## 与合成数据 Q6（verify_q6_phase1）对比\n")
    lines.append("- 合成数据（512x16/1024x32, seed=42）: "
                 "mean(B-A) = -7.47pp，B 更优 0 格、劣化 4 格、持平 4 格；"
                 "判定：DR 在典型随机实例上无正贡献（负结果如实记录）。")
    lines.append(f"- 真实数据（本次）: mean(B-A) = "
                 f"{statistics.mean(diffs):+.2f}pp，"
                 f"B 更优 {n_better}/{len(diffs)}。")
    lines.append("")
    if SYNTH_SUMMARY.is_file():
        lines.append(f"详见合成数据摘要: `{SYNTH_SUMMARY.name}`")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Q6 真实数据 DR 重测")
    parser.add_argument("--seed", type=int, default=42,
                        help="抽样随机种子（默认 42）")
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
    print(f"共 {len(datasets)} 个矩阵，"
          f"{len(datasets) * len(INIT_FNS) * len(BUDGETS) * 2} 个单元格")

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

    OUT_SUMMARY.write_text(
        summarize(rows, datasets, args.seed, started), encoding="utf-8")
    print(f"汇总已写入: {OUT_SUMMARY.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
