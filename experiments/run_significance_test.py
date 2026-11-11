"""实验 C：真实主实验（real_main.csv）的配对显著性检验。

设计：
- 读取 results/real_main.csv。
- 对每个 (规模, 初解, 预算) 分组，取 plain 与 bottleneck 的 c_final
  配对（同 filepath）。
- 检验 H0：plain 与 bottleneck 无差异。
  - 优先 scipy.stats.wilcoxon（双尾，配对样本）；
  - scipy 未安装时退化为符号检验（binomial test）：
    n_better = bottleneck 严格更优的配对数（1e-9 容差）；
    n_pairs = 参与检验的非平局配对数；
    p = 2 * min(P(X <= n_better), P(X >= n_better)),
        X ~ Binomial(n_pairs, 0.5)。
- 输出 results/significance_test.md：分组表格 + 整体 72 对结论段。

不引入新依赖：scipy 可选，未安装即用符号检验（如实注明）。

用法：
    python experiments/run_significance_test.py [--csv results/real_main.csv]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_MD = RESULTS_DIR / "significance_test.md"

TIE_TOL = 1e-9  # c_final 差值的绝对容差：|diff| <= tol 视为平局


def _binom_tail_p(n: int, k: int) -> float:
    """二项分布 X ~ Bin(n, 0.5) 的 P(X <= k)。n 为 0 时返回 1.0。"""
    if n <= 0:
        return 1.0
    total = 0.0
    for i in range(k + 1):
        total += math.comb(n, i)
    return total * 0.5 ** n


def sign_test(n_better: int, n_worse: int) -> float:
    """符号检验（双尾）。

    X ~ Binomial(n_better + n_worse, 0.5)；
    p = 2 * min(P(X <= n_better), P(X >= n_better))。
    """
    n = n_better + n_worse
    if n == 0:
        return 1.0
    lower = _binom_tail_p(n, min(n_better, n_worse))
    return min(1.0, 2.0 * lower)


def wilcoxon_test(diffs: list[float]) -> float:
    """scipy 配对 Wilcoxon 符号秩检验（双尾）；无 scipy 返回 None。"""
    try:
        from scipy.stats import wilcoxon  # 延迟导入：scipy 可选
    except ImportError:
        return None
    if not diffs:
        return 1.0
    try:
        res = wilcoxon(diffs, alternative="two-sided")
        return float(res.pvalue)
    except Exception:  # noqa: BLE001 —— 零差等边界情况回退符号检验
        return None


def load_pairs(csv_path: Path) -> list[dict]:
    """读取 real_main.csv，返回按 (n, m, init, budget) 分组的配对 diff。

    每行 dict: {key: (n,m,init,budget), filepath, c_plain, c_bottleneck}。
    diff = c_plain - c_bottleneck（> 0 表示 bottleneck 更优）。
    """
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cells: dict[tuple, dict[str, dict[str, float]]] = {}
    for r in rows:
        key = (int(r["n"]), int(r["m"]), r["init_method"], int(r["max_evals"]))
        cells.setdefault(key, {}).setdefault(r["filepath"], {})[
            r["searcher"]] = float(r["c_final"])
    pairs: list[dict] = []
    for key in sorted(cells):
        for fp in sorted(cells[key]):
            cpair = cells[key][fp]
            if "plain" not in cpair or "bottleneck" not in cpair:
                continue
            pairs.append({
                "key": key,
                "filepath": fp,
                "c_plain": cpair["plain"],
                "c_bottleneck": cpair["bottleneck"],
            })
    return pairs


def analyze(pairs: list[dict]) -> tuple[list[dict], dict]:
    """分组检验 + 整体检验。返回 (分组行, 整体行)。"""
    groups: dict[tuple, list[dict]] = {}
    for p in pairs:
        groups.setdefault(p["key"], []).append(p)

    group_rows: list[dict] = []
    for key in sorted(groups):
        g = groups[key]
        n_pairs = len(g)
        diffs = [p["c_plain"] - p["c_bottleneck"] for p in g]
        n_better = sum(1 for d in diffs if d > TIE_TOL)
        n_worse = sum(1 for d in diffs if d < -TIE_TOL)
        n_ties = n_pairs - n_better - n_worse
        p_w = wilcoxon_test([d for d in diffs if abs(d) > TIE_TOL])
        p_sign = sign_test(n_better, n_worse)
        use_scipy = p_w is not None
        p_value = p_w if use_scipy else p_sign
        n, m, init, budget = key
        group_rows.append({
            "size": f"{n}x{m}",
            "init": init,
            "budget": budget,
            "n_pairs": n_pairs,
            "n_bottleneck_better": n_better,
            "n_ties": n_ties,
            "p_value": p_value,
            "method": "wilcoxon" if use_scipy else "sign",
            "significant_at_0.05": p_value < 0.05,
        })

    n_pairs = len(pairs)
    diffs = [p["c_plain"] - p["c_bottleneck"] for p in pairs]
    n_better = sum(1 for d in diffs if d > TIE_TOL)
    n_worse = sum(1 for d in diffs if d < -TIE_TOL)
    n_ties = n_pairs - n_better - n_worse
    p_w = wilcoxon_test([d for d in diffs if abs(d) > TIE_TOL])
    p_sign = sign_test(n_better, n_worse)
    use_scipy = p_w is not None
    overall = {
        "size": "整体",
        "init": "全部",
        "budget": "全部",
        "n_pairs": n_pairs,
        "n_bottleneck_better": n_better,
        "n_ties": n_ties,
        "p_value": p_w if use_scipy else p_sign,
        "method": "wilcoxon" if use_scipy else "sign",
        "significant_at_0.05": (p_w if use_scipy else p_sign) < 0.05,
    }
    return group_rows, overall


def render(rows: list[dict], overall: dict) -> str:
    """渲染 significance_test.md 内容。"""
    method = overall["method"]
    lines: list[str] = []
    lines.append("# 实验 C：真实主实验配对显著性检验\n")
    lines.append(f"- 数据来源: results/real_main.csv（{overall['n_pairs']} 个配对）")
    lines.append(f"- 检验方法: {'scipy.stats.wilcoxon（双尾，配对样本）' if method == 'wilcoxon' else '符号检验（binomial test，scipy 未安装）'}")
    lines.append("- H0: plain 与 bottleneck 的 c_final 无差异；")
    lines.append("- 配对定义: 同 (filepath, 规模, 初解, 预算) 的 plain/bottleneck "
                 f"c_final；|diff| <= {TIE_TOL:g} 计为平局；")
    lines.append("- 符号检验: X ~ Binomial(n_pairs, 0.5)，"
                 "p = 2 * min(P(X <= n_better), P(X >= n_better))；"
                 "n_pairs 为非平局配对数。\n")

    lines.append("## 分组检验结果\n")
    lines.append("| size | init | budget | n_pairs | n_bottleneck_better | "
                 "n_ties | p_value | significant_at_0.05 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['size']} | {r['init']} | {r['budget']} | {r['n_pairs']} | "
            f"{r['n_bottleneck_better']} | {r['n_ties']} | "
            f"{r['p_value']:.6g} | {'是' if r['significant_at_0.05'] else '否'} |"
        )
    lines.append("")

    lines.append("## 整体检验（全部配对）\n")
    lines.append(f"- 配对总数: {overall['n_pairs']}；"
                 f"bottleneck 严格更优: {overall['n_bottleneck_better']}；"
                 f"平局: {overall['n_ties']}；"
                 f"劣化: {overall['n_pairs'] - overall['n_bottleneck_better'] - overall['n_ties']}")
    lines.append(f"- p_value = {overall['p_value']:.6g}"
                 f"（{overall['method']}）")
    if overall["significant_at_0.05"]:
        lines.append(f"- 结论: p < 0.05，**拒绝 H0**——整体 "
                     f"{overall['n_bottleneck_better']}/{overall['n_pairs']} "
                     "的瓶颈优先优势在统计上显著。")
    else:
        lines.append("- 结论: p >= 0.05，不能拒绝 H0（如实报告）。")
    lines.append("")

    lines.append("## 说明\n")
    lines.append("- 符号检验是 Wilcoxon 的无分布替代（对差异幅度不敏感，"
                 "只利用方向）；两者结论一致时更稳健。")
    n_2048 = max((r["n_pairs"] for r in rows if r["size"] == "2048x64"),
                 default=0)
    if n_2048:
        lines.append(f"- 分组检验中 2048x64 每组 {n_2048} 对；"
                     "显著性解读以整体检验为主。")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="实验 C：真实主实验配对显著性检验")
    parser.add_argument("--csv", type=str,
                        default=str(RESULTS_DIR / "real_main.csv"),
                        help="主实验 CSV 路径（默认 results/real_main.csv）")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"[FAIL] CSV 不存在: {csv_path}")
        return 1

    pairs = load_pairs(csv_path)
    if not pairs:
        print("[FAIL] CSV 中没有可配对的 plain/bottleneck 单元格。")
        return 1

    rows, overall = analyze(pairs)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render(rows, overall), encoding="utf-8")
    print(f"检验完成（{len(pairs)} 个配对），报告已写入: "
          f"{OUT_MD.relative_to(PROJECT_ROOT)}")
    print(f"整体: p_value = {overall['p_value']:.6g}"
          f"（{overall['method']}），"
          f"显著: {'是' if overall['significant_at_0.05'] else '否'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
