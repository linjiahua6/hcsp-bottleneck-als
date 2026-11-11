"""真实 HCSP 数据上 HEFT 初始解与 bottleneck 搜索的对比实验。

设计：
- 数据：512x16 全部 12 个；1024x32 抽 4 个（u_c/u_i/u_s 各 >= 1、
  hihi/lolo 各 >= 1）；2048x64 抽 2 个（hihi/lolo 各 >= 1）。
  抽样固定 seed，输出选中文件名列表。
- 方法（预算 10000 单档，控制实验量）：
  A: min_min（已有基线初解）
  B: heft（新增初解，独立任务退化版 HEFT）
  C: min_min + bottleneck_local_search(K=4, R=2, S=4)
  D: heft   + bottleneck_local_search(K=4, R=2, S=4)
- 关键结论：C vs D。若 C 优于 D → bottleneck 搜索比 HEFT 初解更强；
  若接近 → HEFT 初解质量已接近瓶颈搜索能达到的水平。
- 输出：results/real_heft.csv 与 results/real_heft_summary.md。

0 值处理与 run_real_main.py 一致：官方 lolo/lohi 类矩阵含 0 值执行
时间，实验层替换为 1e-6（如实报告）。

用法：
    python experiments/run_real_heft.py --fast   # 512x16 抽 4 个，冒烟
    python experiments/run_real_heft.py --full   # 正式实验
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
from src.heft import heft
from src.initial_solutions import min_min
from src.loader import load_hcsp_matrix, load_hcsp_solution
from src.local_search import bottleneck_local_search

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "hcsp"
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_CSV = RESULTS_DIR / "real_heft.csv"
OUT_SUMMARY = RESULTS_DIR / "real_heft_summary.md"

K, R, S = 4, 2, 4
BUDGET = 10000
ZERO_EPS = 1e-6  # 0 值执行时间替换为极小正数（算法模块要求 p>0）


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
        mats = sorted(f for f in (DATA_ROOT / "2048x64").iterdir()
                      if not f.name.endswith(".sol"))
        # 抽 2 个：无法覆盖三类 u，只约束 hihi/lolo 各至少 1
        selected += sample_matrices(mats, n_select=2, min_per_u=0, rng=rng)
    return selected


def run_experiment(datasets: list[Path], seed: int) -> list[dict]:
    """对每个 (矩阵, 方法) 单元格运行实验，返回 CSV 行。"""
    rows: list[dict] = []
    n_cells_total = len(datasets) * 4
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
        # A/B：两种初解；C/D：各自 + bottleneck 搜索
        init_fns = {"min_min": min_min, "heft": heft}
        for init_name, init_fn in init_fns.items():
            a0, l0, c0 = init_fn(P_exp)
            for with_search in (False, True):
                method = f"{init_name}+bn" if with_search else init_name
                t0 = time.perf_counter()
                if with_search:
                    res = bottleneck_local_search(
                        P_exp, a0, l0, float(c0), K=K, R=R, S=S,
                        max_evals=BUDGET)
                    c_final = float(res["cmax"])
                    evals = int(res["evals"])
                else:
                    c_final = float(c0)
                    evals = 0
                rt = time.perf_counter() - t0
                imp = improvement_rate(float(c0), c_final)
                gap0 = (float(c0) - c_opt) / max(c_opt, 1e-9) * 100.0
                gapf = (c_final - c_opt) / max(c_opt, 1e-9) * 100.0
                cell_no += 1
                rows.append({
                    "filepath": str(rel),
                    "n": n,
                    "m": m,
                    "method": method,
                    "c0": round(float(c0), 6),
                    "c_final": round(c_final, 6),
                    "c_opt": round(c_opt, 6),
                    "improvement_pct": round(imp, 6),
                    "gap_initial_pct": round(gap0, 6),
                    "gap_final_pct": round(gapf, 6),
                    "evals": evals,
                    "runtime_sec": round(rt, 6),
                    "seed": seed,
                })
                print(f"  [{cell_no}/{n_cells_total}] {method:<12} | "
                      f"c0={float(c0):>12.2f} -> {c_final:>12.2f} | "
                      f"imp={imp:>6.2f}% gap0={gap0:>7.2f}% "
                      f"gapf={gapf:>7.2f}% | evals={evals:>6} | {rt:>6.2f}s")
    return rows


def summarize(rows: list[dict], datasets: list[Path],
              mode: str, seed: int, started: str) -> str:
    """生成 real_heft_summary.md 内容。"""
    if not rows:
        return "# 真实 HCSP 数据 HEFT 对比汇总\n\n（无有效数据行）\n"

    # Windows 下 filepath 为反斜杠（如 512x16\\u_c_hihi.0），
    # 配对逻辑统一规范化为正斜杠，保证 split("/") 与 startswith 正确。
    rows = [dict(r, filepath=r["filepath"].replace("\\", "/"))
            for r in rows]

    def by_key(key_fn):
        groups: dict[tuple, list[dict]] = {}
        for r in rows:
            groups.setdefault(key_fn(r), []).append(r)
        return groups

    lines: list[str] = []
    lines.append("# 真实 HCSP 数据 HEFT 对比实验汇总\n")
    lines.append(f"- 生成时间: {started}")
    lines.append(f"- 模式: {mode}（seed={seed}）")
    lines.append(f"- 方法: A=min_min、B=heft（独立任务退化版）、"
                 f"C=min_min+bn、D=heft+bn；"
                 f"搜索 bottleneck(K={K},R={R},S={S})、预算 {BUDGET}")
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

    # ---- 分组统计（size x method）----
    lines.append("## 分组统计（size x method）\n")
    lines.append("| size | method | 单元格数 | c0 均值 | gap0 均值(%) | "
                 "c_final 均值 | gap_final 均值(%) | 改进率均值±std(%) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for key in sorted(by_key(lambda r: (r["n"], r["m"], r["method"]))):
        n, m, method = key
        g = by_key(lambda r: (r["n"], r["m"], r["method"]))[key]
        c0s = [r["c0"] for r in g]
        g0s = [r["gap_initial_pct"] for r in g]
        cfs = [r["c_final"] for r in g]
        gfs = [r["gap_final_pct"] for r in g]
        imps = [r["improvement_pct"] for r in g]
        lines.append(
            f"| {n}x{m} | {method} | {len(g)} | {statistics.mean(c0s):.2f} | "
            f"{statistics.mean(g0s):.2f} | {statistics.mean(cfs):.2f} | "
            f"{statistics.mean(gfs):.2f} | "
            f"{statistics.mean(imps):.2f}±{statistics.pstdev(imps):.2f} |"
        )
    lines.append("")

    # ---- 初始解质量：min_min vs heft（配对）----
    lines.append("## 初始解质量对比（A=min_min vs B=heft）\n")
    lines.append("（同矩阵配对，heft 的 gap0 严格更小（1e-9 容差）计为更优）\n")
    lines.append("| size | 配对数 | heft 更优比例 | gap0 差均值(heft-min_min, pp) |")
    lines.append("|---|---:|---:|---:|")
    for size in sorted({r["filepath"].split("/")[0] for r in rows}):
        pairs: dict[str, dict[str, float]] = {}
        for r in rows:
            if not r["filepath"].startswith(size + "/"):
                continue
            pairs.setdefault(r["filepath"], {})[r["method"]] = r["gap_initial_pct"]
        n_better = 0
        diffs = []
        for cpair in pairs.values():
            if "min_min" not in cpair or "heft" not in cpair:
                continue
            d = cpair["min_min"] - cpair["heft"]
            diffs.append(d)
            if d > 1e-9:
                n_better += 1
        if diffs:
            lines.append(f"| {size} | {len(diffs)} | {n_better}/{len(diffs)} "
                         f"({100 * n_better / len(diffs):.1f}%) | "
                         f"{statistics.mean(diffs):.2f} |")
    lines.append("")

    # ---- 后处理质量：C vs D（配对）----
    lines.append("## 后处理质量对比（C=min_min+bn vs D=heft+bn）\n")
    lines.append("（同矩阵配对，D 的 gap_final 严格更小（1e-9 容差）计为更优）\n")
    lines.append("| size | 配对数 | D 更优比例 | gap_final 差均值(D-C, pp) |")
    lines.append("|---|---:|---:|---:|")
    for size in sorted({r["filepath"].split("/")[0] for r in rows}):
        pairs: dict[str, dict[str, float]] = {}
        for r in rows:
            if not r["filepath"].startswith(size + "/"):
                continue
            pairs.setdefault(r["filepath"], {})[r["method"]] = r["gap_final_pct"]
        n_better = 0
        diffs = []
        for cpair in pairs.values():
            if "min_min+bn" not in cpair or "heft+bn" not in cpair:
                continue
            d = cpair["min_min+bn"] - cpair["heft+bn"]
            diffs.append(d)
            if d > 1e-9:
                n_better += 1
        if diffs:
            lines.append(f"| {size} | {len(diffs)} | {n_better}/{len(diffs)} "
                         f"({100 * n_better / len(diffs):.1f}%) | "
                         f"{statistics.mean(diffs):.2f} |")
    lines.append("")

    # ---- 关键结论：Min-Min+bn vs HEFT+bn ----
    c_rows = [r for r in rows if r["method"] == "min_min+bn"]
    d_rows = [r for r in rows if r["method"] == "heft+bn"]
    lines.append("## 关键结论：Min-Min+bottleneck vs HEFT+bottleneck\n")
    if c_rows and d_rows:
        c_gap = statistics.mean(r["gap_final_pct"] for r in c_rows)
        d_gap = statistics.mean(r["gap_final_pct"] for r in d_rows)
        pairs: dict[str, dict[str, float]] = {}
        for r in rows:
            if r["method"] in ("min_min+bn", "heft+bn"):
                pairs.setdefault(r["filepath"], {})[r["method"]] = \
                    r["gap_final_pct"]
        n_d_better = 0
        n_ties = 0
        for cpair in pairs.values():
            if "min_min+bn" not in cpair or "heft+bn" not in cpair:
                continue
            d = cpair["min_min+bn"] - cpair["heft+bn"]
            if d > 1e-9:
                n_d_better += 1
            elif abs(d) <= 1e-9:
                n_ties += 1
        lines.append(f"- min_min+bn 的 gap_final 均值: {c_gap:.2f}%")
        lines.append(f"- heft+bn 的 gap_final 均值: {d_gap:.2f}%")
        lines.append(f"- D（heft+bn）更优 {n_d_better}/{len(pairs)}，"
                     f"持平 {n_ties}，劣化 {len(pairs) - n_d_better - n_ties}")
        if c_gap < d_gap - 1e-9:
            lines.append("- 判定: **Min-Min+bottleneck 更优** → bottleneck "
                         "搜索相对 HEFT 初解具有独立价值（HEFT 初解未能替代）")
        elif d_gap < c_gap - 1e-9:
            lines.append("- 判定: **HEFT+bottleneck 更优** → HEFT 初解与瓶颈 "
                         "搜索互补（如实报告）")
        else:
            lines.append("- 判定: 两者接近（差值 < 1e-9 pp 级）→ HEFT 初解质量 "
                         "已接近瓶颈搜索能达到的水平（如实报告）")
    else:
        lines.append("（数据不足，无法比较）")
    lines.append("")

    # ---- HEFT 退化说明 ----
    lines.append("## HEFT 退化说明\n")
    lines.append("- 本实验的 HEFT 是独立任务场景下的退化实现：无依赖 → 无 upward "
                 "rank，任务优先级 = 平均执行时间（与 LPT 排序相同）；"
                 "分配决策为 EFT（loads[j] + P[i, j] 最小），区别于 LPT 的"
                 "负载最小决策。论文中需明确说明，避免被误读为完整 DAG HEFT。")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="真实 HCSP 数据 HEFT 对比实验")
    parser.add_argument("--fast", action="store_true",
                        help="只用 512x16 的 4 个矩阵（快速验证流程）")
    parser.add_argument("--full", action="store_true",
                        help="正式实验：512x16 全部 + 1024x32 抽 4 + "
                             "2048x64 抽 2")
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
    print(f"共 {len(datasets)} 个矩阵，{len(datasets) * 4} 个单元格")

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
