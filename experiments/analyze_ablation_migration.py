"""V1.5 消融分析：BLS-M vs PLS、BLS vs BLS-M、BLS vs PLS（复现核对）。

数据来源：
- results/real_main.csv：主实验 plain + bottleneck 的 512x16 部分
  （96 行 = 48 配对条件），复用不重跑；
- results/real_ablation_blsm.csv：新增 BLS-M 48 行。

匹配规则：按 (filepath, init_method, max_evals) 精确匹配，禁止按行序
拼接。核验项：
- 新 CSV 48 行、无重复、n=512/m=16、seed=42；
- BLS-M 的 c0 与 c_opt 必须与主实验对应行完全一致（同一初解、同一
  目标函数，取四舍五入到 6 位小数的值比较）。

输出：results/ablation_blsm_analysis.md（条件级胜平负 + 矩阵级稳健性
+ 统计检验 + 依赖结构说明）。
"""

from __future__ import annotations

import csv
import statistics
import sys
from math import comb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
MAIN_CSV = RESULTS_DIR / "real_main.csv"
ABL_CSV = RESULTS_DIR / "real_ablation_blsm.csv"
OUT_MD = RESULTS_DIR / "ablation_blsm_analysis.md"

TOL = 1e-9


def binom_two_tail_p(k: int, n: int) -> float:
    """精确二项符号检验（双尾）：p = 2 * min(P(X<=k), P(X>=k))。"""
    if n <= 0:
        return float("nan")
    p_lo = sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    p_hi = sum(comb(n, i) for i in range(k, n + 1)) * (0.5 ** n)
    return min(1.0, 2 * min(p_lo, p_hi))


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["max_evals"] = int(r["max_evals"])
        r["n"] = int(r["n"])
        r["m"] = int(r["m"])
        for col in ("c0", "c_final", "c_opt", "improvement_pct",
                    "gap_initial_pct", "gap_final_pct", "runtime_sec"):
            r[col] = float(r[col])
    return rows


def verify_ablation_csv(main_rows: list[dict], abl_rows: list[dict]) -> list[str]:
    """校验新 CSV 的完整性、无重复、与主实验的初解/目标一致性。"""
    problems: list[str] = []
    if len(abl_rows) != 48:
        problems.append(f"BLS-M CSV 行数应为 48，实际 {len(abl_rows)}")
    keys = [(r["filepath"], r["init_method"], r["max_evals"])
            for r in abl_rows]
    if len(set(keys)) != len(keys):
        problems.append("BLS-M CSV 存在重复 (filepath, init, budget)")
    for r in abl_rows:
        if r["n"] != 512 or r["m"] != 16:
            problems.append(f"{r['filepath']}: 规模应为 512x16")
        if r["seed"] != "42":
            problems.append(f"{r['filepath']}: seed 应为 42")
    # 与主实验对应行的 c0 / c_opt 一致性（同一初解、同一目标函数）
    main_map = {(r["filepath"], r["init_method"], r["max_evals"]): r
                for r in main_rows}
    for r in abl_rows:
        key = (r["filepath"], r["init_method"], r["max_evals"])
        if key not in main_map:
            problems.append(f"{key}: 主实验缺少对应条件，无法复用比对")
            continue
        m = main_map[key]
        if abs(r["c0"] - m["c0"]) > 1e-5:
            problems.append(f"{key}: c0 不一致 (blsm={r['c0']} "
                            f"main={m['c0']})")
        if abs(r["c_opt"] - m["c_opt"]) > 1e-5:
            problems.append(f"{key}: c_opt 不一致 (blsm={r['c_opt']} "
                            f"main={m['c_opt']})")
    return problems


def paired_table(method_rows: list[dict], rival_rows: list[dict],
                 method_name: str, rival_name: str) -> dict:
    """按 (filepath, init, budget) 精确配对，返回比较统计。"""
    rival_map = {(r["filepath"], r["init_method"], r["max_evals"]): r
                 for r in rival_rows}
    pairs = []
    for r in method_rows:
        key = (r["filepath"], r["init_method"], r["max_evals"])
        if key not in rival_map:
            continue
        v = rival_map[key]
        diff_c = v["c_final"] - r["c_final"]  # >0 方法更优
        if abs(diff_c) <= TOL:
            outcome = "tie"
        elif diff_c > 0:
            outcome = "win"
        else:
            outcome = "loss"
        pairs.append({
            **r, "rival": v, "outcome": outcome,
            "d_gap": v["gap_final_pct"] - r["gap_final_pct"],
            "d_imp": r["improvement_pct"] - v["improvement_pct"],
            "d_c": diff_c,
        })
    return {"method": method_name, "rival": rival_name, "pairs": pairs}


def summarize_comparison(comp: dict) -> list[str]:
    """生成一组比较的报告行（条件级 + 分初始解 + 矩阵级 + 检验）。"""
    pairs = comp["pairs"]
    L: list[str] = []
    L.append(f"### {comp['method']} vs {comp['rival']} "
             f"（{len(pairs)} 配对条件）")

    wins = sum(p["outcome"] == "win" for p in pairs)
    ties = sum(p["outcome"] == "tie" for p in pairs)
    losses = sum(p["outcome"] == "loss" for p in pairs)
    L.append(f"条件级：胜 {wins} / 平 {ties} / 负 {losses}")

    # 条件级符号检验（剔除平局；注明依赖结构）
    n_nontie = wins + losses
    p_cond = binom_two_tail_p(wins, n_nontie) if n_nontie else float("nan")
    L.append(f"条件级符号检验：n={n_nontie}（剔除平局），k={wins}，"
             f"p = {p_cond:.4g}（注意：48 条件非独立，同一矩阵贡献 4 个"
             f"条件，此 p 值仅供量级参考）")

    # 差值统计：分初始解
    for init in ("min_min", "round_robin"):
        sub = [p for p in pairs if p["init_method"] == init]
        if not sub:
            continue
        dg = sorted(p["d_gap"] for p in sub)
        di = sorted(p["d_imp"] for p in sub)
        med_dg = dg[len(dg) // 2]
        med_di = di[len(di) // 2]
        w = sum(p["outcome"] == "win" for p in sub)
        t = sum(p["outcome"] == "tie" for p in sub)
        lo = sum(p["outcome"] == "loss" for p in sub)
        L.append(f"- {init}: 胜 {w} / 平 {t} / 负 {lo}；"
                 f"gap_final 差（{comp['rival']} − {comp['method']}，pp）"
                 f"中位数 {med_dg:+.2f}；改进率差（{comp['method']} − "
                 f"{comp['rival']}，pp）中位数 {med_di:+.2f}")

    # 评估数与运行时间
    m_ev = sorted(p["evals"] for p in pairs)
    r_ev = sorted(p["rival"]["evals"] for p in pairs)
    m_rt = sorted(p["runtime_sec"] for p in pairs)
    r_rt = sorted(p["rival"]["runtime_sec"] for p in pairs)
    L.append(f"- evals 中位数：{comp['method']} = {m_ev[len(m_ev)//2]}，"
             f"{comp['rival']} = {r_ev[len(r_ev)//2]}；"
             f"runtime 中位数：{comp['method']} = "
             f"{m_rt[len(m_rt)//2]:.4f}s，{comp['rival']} = "
             f"{r_rt[len(r_rt)//2]:.4f}s")

    # 矩阵级：12 矩阵 × 4 条件
    per_mat: dict[str, dict] = {}
    for p in pairs:
        m = per_mat.setdefault(
            p["filepath"],
            {"win": 0, "tie": 0, "loss": 0})
        m[p["outcome"]] += 1
    mats = sorted(per_mat)
    all4 = sum(per_mat[m]["win"] == 4 for m in mats)
    # 矩阵级判定：4 条件多数（全平记 tie）
    mat_outcomes = []
    for m in mats:
        st = per_mat[m]
        if st["win"] > st["loss"]:
            mat_outcomes.append("win")
        elif st["loss"] > st["win"]:
            mat_outcomes.append("loss")
        else:
            mat_outcomes.append("tie")
    m_wins = mat_outcomes.count("win")
    m_ties = mat_outcomes.count("tie")
    m_losses = mat_outcomes.count("loss")
    n_mat_nontie = m_wins + m_losses
    p_mat = binom_two_tail_p(m_wins, n_mat_nontie) if n_mat_nontie else float("nan")
    L.append(f"矩阵级（12 矩阵 × 4 条件/矩阵）：4/4 全胜矩阵 {all4}/12；"
             f"矩阵级多数判定：胜 {m_wins} / 平 {m_ties} / 负 {m_losses}；"
             f"矩阵级符号检验 n={n_mat_nontie}（剔除平局），k={m_wins}，"
             f"p = {p_mat:.4g}")

    # 各矩阵明细
    L.append("| 矩阵 | win | tie | loss |")
    L.append("|---|---:|---:|---:|")
    for m in mats:
        st = per_mat[m]
        L.append(f"| {m} | {st['win']} | {st['tie']} | {st['loss']} |")
    L.append("")
    return L


def main() -> int:
    main_rows = [r for r in load_rows(MAIN_CSV) if r["n"] == 512]
    abl_rows = load_rows(ABL_CSV)

    problems = verify_ablation_csv(main_rows, abl_rows)
    if problems:
        print("[FAIL] 数据校验未通过：")
        for p in problems:
            print("  -", p)
        return 1
    print("数据校验通过：BLS-M 48 行完整、无重复；c0/c_opt 与主实验一致。")

    plain = [r for r in main_rows if r["searcher"] == "plain"]
    bottleneck = [r for r in main_rows if r["searcher"] == "bottleneck"]
    blsm = abl_rows
    print(f"配对规模核对：plain={len(plain)}, bottleneck={len(bottleneck)}, "
          f"bls_migration={len(blsm)}")
    if not (len(plain) == len(bottleneck) == len(blsm) == 48):
        print("[FAIL] 三方行数应均为 48")
        return 1

    comps = [
        paired_table(blsm, plain, "BLS-M", "PLS"),
        paired_table(bottleneck, blsm, "BLS", "BLS-M"),
        paired_table(bottleneck, plain, "BLS", "PLS"),
    ]

    lines: list[str] = []
    lines.append("# V1.5 消融分析：BLS-M vs PLS / BLS vs BLS-M / BLS vs PLS\n")
    lines.append("## 数据与匹配\n")
    lines.append("- BLS-M：`results/real_ablation_blsm.csv`（48 行，新增）；")
    lines.append("- BLS/PLS：`results/real_main.csv` 的 512x16 部分"
                 "（96 行，复用，不重跑）；")
    lines.append("- 匹配：按 (filepath, init_method, max_evals) 精确配对；"
                 "BLS-M 的 c0 与 c_opt 已核验与主实验对应行一致。")
    lines.append("- 差值口径：gap_final 差 = 对手 − 方法（pp，正值表示方法"
                 "更优）；改进率差 = 方法 − 对手（pp）。")
    lines.append("- 依赖结构：48 条件非独立——同一矩阵贡献 4 个条件"
                 "（2 初解 × 2 预算）；矩阵级（12 配对）是更保守的读数；"
                 "子组（初解 × 预算）视为探索性，未做多重比较校正。\n")

    lines.append("## 比较结果\n")
    for comp in comps:
        lines.extend(summarize_comparison(comp))

    lines.append("## 复现核对\n")
    b_vs_p = comps[2]
    w = sum(p["outcome"] == "win" for p in b_vs_p["pairs"])
    t = sum(p["outcome"] == "tie" for p in b_vs_p["pairs"])
    lo = sum(p["outcome"] == "loss" for p in b_vs_p["pairs"])
    lines.append(f"BLS vs PLS（512x16 部分，48 配对）：胜 {w} / 平 {t} / "
                 f"负 {lo}。主实验 512x16 部分应为 48/48 BLS 更优；"
                 f"若不一致说明复用或匹配存在问题。")

    lines.append("\n## 谨慎解释\n")
    lines.append("- BLS-M 与 PLS 的差异仅来自瓶颈候选/目标筛选（两者都只做"
                 "迁移），但其邻域大小（每轮 ≤KR vs n(m−1)）、每轮轮数与"
                 "搜索路径不同，BLS-M 优于 PLS 不能直接等同于'瓶颈筛选"
                 "的纯因果作用已被证明'。")
    lines.append("- BLS 与 BLS-M 的差异来自 swap 的增量（两者迁移分支为"
                 "同一代码路径），但加入 swap 也改变了每轮评估数与轮数，"
                 "增量解释同样受搜索路径影响。")
    lines.append("- 上述比较均在等评估预算（5000/10000）下进行；公平性"
                 "约束与主实验一致（同矩阵、同初解、同目标函数、同容差、"
                 "确定性平局规则）。")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"分析报告已写入: {OUT_MD.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
