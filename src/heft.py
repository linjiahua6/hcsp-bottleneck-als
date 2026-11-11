"""HEFT（Heterogeneous Earliest Finish Time）初始解 —— 独立任务退化版。

重要说明（论文中务必明确，避免审稿人误解）：
    经典 HEFT 是为 DAG 工作流调度设计的列表调度算法，其核心包括：
    （a）upward rank 计算（依赖 DAG 拓扑）；
    （b）任务优先级排序；
    （c）插入式（insertion-based）EFT 分配。

    本项目研究的是**独立任务**场景（无依赖），因此 HEFT 在此退化为：
        1. 计算每个任务在所有节点上的平均执行时间 avg_i = mean_j P[i, j]
           （独立任务无 upward rank，平均执行时间是其唯一合理优先级代理）；
        2. 按 avg_i 降序排列任务（与 LPT 的排序完全一致，因为无依赖）；
        3. 依次将每个任务分配到使其**完成时间**（loads[j] + P[i, j]）
           最小的节点（EFT 决策，而非 LPT 的"负载最小"决策）；
        4. 平局取节点编号小者（确定性）。
    复杂度 O(n log n + n m)。

    与 LPT 的区别仅在分配决策：LPT 选负载最小节点，HEFT 选完成时间
    最小节点。与工作流 HEFT 的区别：无依赖、无 upward rank、无插入。
"""

from __future__ import annotations

import numpy as np

from .feasibility import compute_loads


def _validate_matrix(P: np.ndarray) -> np.ndarray:
    """校验并规范化执行时间矩阵（与 initial_solutions 一致：p_ij > 0）。"""
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError(f"执行时间矩阵必须是二维数组，实际维度: {P.ndim}")
    n, m = P.shape
    if n == 0:
        raise ValueError("任务数必须大于 0")
    if m == 0:
        raise ValueError("节点数必须大于 0")
    if not np.all(np.isfinite(P)) or not np.all(P > 0):
        raise ValueError("执行时间矩阵中所有值必须是正的有限数值")
    return P


def heft(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """HEFT 在独立任务场景下的退化实现（列表调度 + EFT 分配）。

    步骤：
    1. avg_i = mean_j P[i, j]（独立任务无依赖，平均执行时间替代 upward rank）；
    2. 按 avg_i 降序排列任务（stable sort，与 LPT 排序一致）；
    3. 依次将任务分配到完成时间 loads[j] + P[i, j] 最小的节点；
    4. 平局取节点编号小者（np.argmin 自然行为，确定性）。

    Args:
        P: 执行时间矩阵 (n, m)，所有值必须 > 0 且有限。

    Returns:
        (assignment, loads, cmax)：与现有初解接口一致的三元组；
        assignment 为形状 (n,) 的 int64 节点编号数组。
    """
    P = _validate_matrix(P)
    n, m = P.shape
    avg_times = P.mean(axis=1)
    order = np.argsort(-avg_times, kind="stable")  # 降序，与 LPT 相同
    assignment = np.empty(n, dtype=np.int64)
    loads = np.zeros(m, dtype=np.float64)
    for i in order:
        efts = loads + P[i]  # 任务 i 在各节点上的完成时间
        j = int(np.argmin(efts))  # 平局取节点编号小者
        assignment[i] = j
        loads[j] += P[i, j]
    # 增量累加与 bincount 的浮点顺序不同，为与下游核验
    # （verify_solution, tol=1e-9）严格一致，返回前用统一入口重算。
    loads = compute_loads(P, assignment)
    cmax = float(np.max(loads))
    return assignment, loads, cmax
