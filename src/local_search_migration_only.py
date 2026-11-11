"""BLS 的仅迁移消融变体（BLS-M，ablation）。

V1.5 消融实验专用变体：沿用 BLS 的瓶颈识别、候选任务筛选（K）与迁移
目标筛选（R），但关闭交换候选（S 恒为 0）。每轮至多评估 KR 次单任务
迁移，best-improvement 接受、严格改善条件、预算/终止逻辑与 BLS 一致。

实现约定（零改动原则）：
- 本模块不修改 src/local_search.py 的任何现有行为；
- 轮内枚举直接复用 src/local_search._evaluate_round（与
  bottleneck_local_search、adaptive_bottleneck_search 同一函数），
  传入 S=0 使交换候选列表为空、交换循环零次执行——迁移分支与 BLS
  字面为同一代码路径（候选排序、平局处理、评估计数完全一致）；
- 每轮仍会执行一次 sorted_tasks_of 预计算（S=0 时结果未被使用），
  仅增加少量运行时间，不改变任何评估结果；运行时间如实记录。
"""

from __future__ import annotations

import time

import numpy as np

from .feasibility import compute_loads
from .local_search import MOVE, SWAP, _check_input, _evaluate_round, _make_result


def bottleneck_migration_search(
    P: np.ndarray,
    assignment: np.ndarray,
    loads: np.ndarray,
    cmax: float,
    K: int = 4,
    R: int = 2,
    max_evals: int = 10000,
    tol: float = 1e-9,
    time_budget: float | None = None,
) -> dict:
    """瓶颈导向局部搜索的仅迁移变体（BLS-M）。

    每轮：
    1. find_bottleneck 定位瓶颈节点 b；
    2. select_candidate_tasks 从 b 上选 K 个候选任务（按 P[i, b] 降序）；
    3. 对每个候选任务 i，按迁移后目标负载 L_j + P[i, j] 升序选前 R 个
       目标节点（跳过当前节点 a[i]），评估单任务迁移；
    4. 交换候选关闭（S=0）：不评估任何两任务交换；
    5. 只保留严格改善（C'_max < C_max - tol）的候选，取 C'_max 最小者
       执行；若无改善则停止；
    6. 负载与分配增量更新，累计 evals，达到 max_evals 停止。

    与 bottleneck_local_search 的唯一区别是关闭交换；迁移分支、接受
    条件、终止条件与评估计数完全一致。

    Args:
        P: 执行时间矩阵 (n, m)。
        assignment: 初始分配 (n,)。
        loads: 初始负载 (m,)。
        cmax: 初始 C_max。
        K: 每轮瓶颈节点候选任务数。
        R: 每个候选任务的目标节点数。
        max_evals: 候选评估预算（>= 0）。
        tol: 浮点比较容差。
        time_budget: 可选时间预算（秒）。None 表示不限制；给定后，
            在每轮边界检查已用时间，超过预算即停止（与 max_evals
            先到者为准）。默认 None。

    Returns:
        标准结果 dict（字段与 local_search 模块一致）：
        assignment / loads / cmax / evals / accepted_moves / improved /
        history。
    """
    if K <= 0:
        raise ValueError(f"K 必须为正整数，实际: {K}")
    if R <= 0:
        raise ValueError(f"R 必须为正整数，实际: {R}")

    P_arr = np.asarray(P, dtype=np.float64)
    n, m = P_arr.shape
    a = np.asarray(assignment, dtype=np.int64).copy()
    loads = np.asarray(loads, dtype=np.float64).copy()
    c0 = float(cmax)
    cur = c0
    _check_input(P_arr, a, loads, cur, tol)

    evals = 0
    accepted = 0
    history: list[dict] = []
    t_start = time.perf_counter()

    while evals < max_evals and (time_budget is None or time.perf_counter() - t_start < time_budget):
        round_no = len(history) + 1
        # S=0：交换候选列表为空，仅评估迁移；迁移分支与 BLS 同一函数
        best_action, best_c, evals = _evaluate_round(
            P_arr, a, loads, cur, K, R, 0, max_evals, evals, tol
        )

        if best_action is None:
            history.append(
                {"round": round_no, "cmax": cur, "evals": evals, "accepted": 0}
            )
            break  # 无严格改善（或瓶颈节点无任务/预算耗尽），停止

        # 执行动作（增量更新 assignment 与 loads）。
        # S=0 时 best_action 必为 MOVE；SWAP 分支保留作防御性一致。
        if best_action[0] == MOVE:
            _, i, dst = best_action
            src = int(a[i])
            a[i] = dst
            loads[src] -= P_arr[i, src]
            loads[dst] += P_arr[i, dst]
        else:
            _, i, dst, h = best_action
            src = int(a[i])
            a[i] = dst
            a[h] = src
            loads[src] += P_arr[h, src] - P_arr[i, src]
            loads[dst] += P_arr[i, dst] - P_arr[h, dst]

        cur = best_c
        accepted += 1
        history.append(
            {"round": round_no, "cmax": cur, "evals": evals, "accepted": 1}
        )

    loads = compute_loads(P_arr, a)
    cur = float(np.max(loads))
    return _make_result(a, loads, cur, c0, evals, accepted, history, tol)
