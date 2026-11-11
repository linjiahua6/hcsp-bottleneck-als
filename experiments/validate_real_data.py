"""真实 HCSP 数据加载与锚定验证。

遍历 data/hcsp/ 下所有矩阵与 .sol 解文件，逐项核对：
1. 矩阵：加载成功、形状 == 头部声明 (n, m)、记录 min/max；
2. 解文件：长度 == n、编号 0-based 且 max < m；
3. 锚定：用 .sol 分配计算 makespan，与文件名解析的最优值对比；
4. optimality gap 参考值（验证阶段应为 0）。

任一 (矩阵, 解文件) 对的锚定判定：文件名中的最优值只保留
1 位小数（数据集命名惯例，即 cmax 的一位小数版本），因此
|cmax_from_sol - optimal_from_name| 的理论上界为 0.05（舍入误差）
加浮点累加误差；diff > 0.051 才判定为真实不匹配 [FAIL]，
脚本退出码非 0（供流程检查阻断）。接近舍入边界（diff > 0.049）
的案例以 [WARN] 如实标出。

输出 results/real_data_validation.csv：
    filepath, n, m, min_p, max_p, sol_len, sol_max,
    cmax_from_sol, optimal_from_name, diff
（diff 为真实绝对差 |cmax - optimal|，如实记录）

注意：官方 benchmark 部分实例（lolo/lohi 类）含 0 值执行时间，
属数据事实；锚定计算在原始矩阵（含 0）上进行——官方 .sol 正是
基于含 0 矩阵的最优解，因此 diff 应精确为 0。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.loader import (
    load_hcsp_matrix,
    load_hcsp_solution,
    parse_optimal_from_filename,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "hcsp"
RESULTS_DIR = PROJECT_ROOT / "results"
CSV_OUT = RESULTS_DIR / "real_data_validation.csv"

DIFF_TOL = 1e-3  # 浮点容差（用户要求：比较用 1e-3 级绝对容差）
ROUND_BOUND = 0.05 + DIFF_TOL  # 文件名 1 位小数舍入误差上界 + 浮点容差
WARN_BOUND = 0.049  # 接近舍入边界，标记为 [WARN]（如实报告）


def _read_declared_shape(path: Path) -> tuple[int, int]:
    """从文件头部读取声明的 (n, m)。"""
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip():
            tokens = line.split()
            return int(tokens[0]), int(tokens[1])
    raise ValueError(f"矩阵文件没有头部: {path}")


def main() -> int:
    if not DATA_ROOT.is_dir():
        print(f"[FAIL] 数据目录不存在: {DATA_ROOT}")
        return 1

    rows: list[dict] = []
    failures: list[str] = []
    n_mats_ok = 0
    n_mats_fail = 0
    n_sols_ok = 0
    n_sols_fail = 0
    max_diff = 0.0
    n_zero_value_mats = 0

    for sdir in sorted(DATA_ROOT.iterdir()):
        if not sdir.is_dir() or sdir.name.startswith("."):
            continue  # 跳过 .git 等隐藏目录
        mats = sorted(f for f in sdir.iterdir()
                      if not f.name.endswith(".sol")
                      and not f.name.startswith("."))
        for mat in mats:
            # ---- 1. 矩阵加载 ----
            try:
                P = load_hcsp_matrix(str(mat))
                n_decl, m_decl = _read_declared_shape(mat)
            except Exception as exc:  # noqa: BLE001 —— 加载失败必须统计，不静默
                failures.append(f"{mat.relative_to(DATA_ROOT)}: 矩阵加载失败 ({exc})")
                n_mats_fail += 1
                continue
            if P.shape != (n_decl, m_decl):
                failures.append(
                    f"{mat.relative_to(DATA_ROOT)}: 形状 {P.shape} "
                    f"!= 头部声明 ({n_decl}, {m_decl})"
                )
                n_mats_fail += 1
                continue
            n_mats_ok += 1
            if np.any(P == 0):
                n_zero_value_mats += 1
            min_p = float(P.min())
            max_p = float(P.max())

            # ---- 2/3. 解文件与锚定 ----
            sols = sorted(sdir.glob(f"{mat.name}_*.sol"))
            if not sols:
                failures.append(
                    f"{mat.relative_to(DATA_ROOT)}: 未找到任何 .sol 解文件"
                )
                n_mats_fail += 1
                continue
            for sol in sols:
                rel = sol.relative_to(DATA_ROOT)
                target = parse_optimal_from_filename(str(sol))
                if target is None:
                    failures.append(f"{rel}: 文件名无法解析最优值")
                    n_sols_fail += 1
                    continue
                try:
                    a = load_hcsp_solution(str(sol))
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{rel}: 解文件解析失败 ({exc})")
                    n_sols_fail += 1
                    continue
                if len(a) != P.shape[0]:
                    failures.append(
                        f"{rel}: 解长度 {len(a)} != 任务数 {P.shape[0]}"
                    )
                    n_sols_fail += 1
                    continue
                if a.min() < 0 or a.max() >= P.shape[1]:
                    failures.append(
                        f"{rel}: 编号越界 (min={a.min()}, max={a.max()}, "
                        f"m={P.shape[1]})"
                    )
                    n_sols_fail += 1
                    continue
                loads = np.array([P[a == j, j].sum() for j in range(P.shape[1])])
                cmax = float(loads.max())
                diff = abs(cmax - target)
                max_diff = max(max_diff, diff)
                ok = diff <= ROUND_BOUND
                if ok:
                    n_sols_ok += 1
                    if diff > WARN_BOUND:
                        failures.append(
                            f"{rel}: [WARN] 接近 1 位小数舍入边界 "
                            f"(diff={diff:g}，官方命名规则在边界处可能与标准舍入不同)"
                        )
                else:
                    n_sols_fail += 1
                    failures.append(
                        f"{rel}: makespan {cmax!r} 与文件名最优值 {target!r} "
                        f"不符（diff={diff:g}，超出 1 位小数舍入范围）"
                    )
                rows.append({
                    "filepath": str(rel),
                    "n": P.shape[0],
                    "m": P.shape[1],
                    "min_p": f"{min_p:g}",
                    "max_p": f"{max_p:g}",
                    "sol_len": len(a),
                    "sol_max": int(a.max()),
                    "cmax_from_sol": f"{cmax:.6f}",
                    "optimal_from_name": f"{target:.6f}",
                    "diff": f"{diff:.6e}",
                })

    # ---- 输出 CSV ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filepath", "n", "m", "min_p", "max_p", "sol_len", "sol_max",
            "cmax_from_sol", "optimal_from_name", "diff",
        ])
        writer.writeheader()
        writer.writerows(rows)

    # ---- 控制台汇总 ----
    print("=" * 70)
    print("真实 HCSP 数据验证汇总")
    print(f"  矩阵加载成功: {n_mats_ok}   矩阵加载失败: {n_mats_fail}")
    print(f"  解文件验证 [OK]: {n_sols_ok}   解文件 [FAIL]: {n_sols_fail}")
    print(f"  含 0 值执行时间的矩阵数: {n_zero_value_mats}")
    print(f"  最大 |cmax - optimal|: {max_diff:.3e}"
          f"（文件名最优值为 1 位小数舍入值，理论上界 {ROUND_BOUND:g}）")
    n_warn = sum(1 for msg in failures if "[WARN]" in msg)
    if n_warn:
        print(f"  舍入边界案例 [WARN]: {n_warn} 个（详见下方列表，不影响通过）")
    print(f"  CSV 输出: {CSV_OUT.relative_to(PROJECT_ROOT)}")
    if failures:
        print("-" * 70)
        print(f"异常文件列表（{len(failures)} 项）:")
        for msg in failures:
            tag = "[WARN]" if "[WARN]" in msg else "[FAIL]"
            print(f"  {tag} {msg.replace('[WARN] ', '')}")
    print("=" * 70)

    if failures or n_sols_fail or n_mats_fail:
        has_real_fail = any("[WARN]" not in msg for msg in failures)
        if has_real_fail or n_sols_fail or n_mats_fail:
            print("[FAIL] 存在验证失败项，请先修复 loader 再进入主实验。")
            return 1
    print("[OK] 全部验证通过，可进入真实数据主实验。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
