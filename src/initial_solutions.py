"""异构独立任务调度的初始解生成方法。

每个函数接受执行时间矩阵 P（形状 (n, m)），返回三元组：
    assignment: np.ndarray，形状 (n,)，assignment[i] 为任务 i 分配的节点索引；
    loads:      np.ndarray，形状 (m,)，节点 j 的负载 L_j；
    cmax:       float，最大完工时间 max_j L_j。
"""

from __future__ import annotations

import numpy as np

from .feasibility import compute_loads


def _validate_matrix(P: np.ndarray) -> np.ndarray:
    """校验并规范化执行时间矩阵。"""
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


def _finalize(
    P: np.ndarray, assignment: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """由分配计算负载与 C_max，返回标准三元组 (assignment, loads, cmax)。"""
    loads = compute_loads(P, assignment)
    cmax = float(np.max(loads))
    return assignment, loads, cmax


def round_robin(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """轮询（Round-Robin）初始解。

    按任务索引顺序将任务轮流分配到节点 0, 1, ..., m-1, 0, 1, ...
    复杂度 O(n)。

    Args:
        P: 执行时间矩阵 (n, m)。

    Returns:
        (assignment, loads, cmax)。
    """
    P = _validate_matrix(P)
    n, m = P.shape
    assignment = np.arange(n, dtype=np.int64) % m
    return _finalize(P, assignment)


def min_min(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """经典 Min-Min 启发式。

    每次迭代：
    1. 对每个未分配任务 i，计算其在各节点上的完成时间 loads[j] + P[i, j]，
       取其最小值 mct_i（对应节点 j_i）；
    2. 选择全局最小 mct_i 的任务-节点对进行分配；
    3. 更新该节点负载，重复直至所有任务分配完毕。

    平局时优先选择任务索引较小的任务，保证结果确定性。
    复杂度 O(n^2 m)。

    Args:
        P: 执行时间矩阵 (n, m)。

    Returns:
        (assignment, loads, cmax)。
    """
    P = _validate_matrix(P)
    n, m = P.shape
    assignment = np.full(n, -1, dtype=np.int64)
    loads = np.zeros(m, dtype=np.float64)
    unassigned = list(range(n))

    while unassigned:
        best_task = -1
        best_machine = -1
        best_ct = np.inf
        for i in unassigned:
            cts = loads + P[i]  # 任务 i 在各节点上的完成时间
            j = int(np.argmin(cts))
            ct = float(cts[j])
            if ct < best_ct - 1e-12 or (abs(ct - best_ct) <= 1e-12 and i < best_task):
                best_ct = ct
                best_machine = j
                best_task = i
        assignment[best_task] = best_machine
        loads[best_machine] += P[best_task, best_machine]
        unassigned.remove(best_task)

    # 增量累加与 bincount 的浮点顺序不同，为与下游核验
    # （verify_solution, tol=1e-9）严格一致，返回前用统一入口重算。
    return _finalize(P, assignment)


def lpt(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """LPT（最长处理时间优先）初始解。

    按任务在所有节点上的平均执行时间降序排列，依次将任务分配到当前
    负载最小的节点。复杂度 O(n log n + n m)。

    Args:
        P: 执行时间矩阵 (n, m)。

    Returns:
        (assignment, loads, cmax)。
    """
    P = _validate_matrix(P)
    n, m = P.shape
    mean_times = P.mean(axis=1)
    order = np.argsort(-mean_times, kind="stable")  # 降序
    assignment = np.empty(n, dtype=np.int64)
    loads = np.zeros(m, dtype=np.float64)
    for i in order:
        j = int(np.argmin(loads))
        assignment[i] = j
        loads[j] += P[i, j]
    return _finalize(P, assignment)


def random_greedy(
    P: np.ndarray, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, float]:
    """随机贪心初始解。

    以固定随机种子打乱任务顺序，然后依次将任务分配到当前负载最小的
    节点。复杂度 O(n m)。

    Args:
        P: 执行时间矩阵 (n, m)。
        seed: 随机种子（固定种子保证结果可复现）。

    Returns:
        (assignment, loads, cmax)。
    """
    P = _validate_matrix(P)
    n, m = P.shape
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    assignment = np.empty(n, dtype=np.int64)
    loads = np.zeros(m, dtype=np.float64)
    for i in order:
        j = int(np.argmin(loads))
        assignment[i] = j
        loads[j] += P[i, j]
    return _finalize(P, assignment)
