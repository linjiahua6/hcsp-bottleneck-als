"""解的可行性检查与目标函数计算。

分配方案用长度为 n 的整数数组 assignment 表示，assignment[i] 为任务 i
分配到的节点索引。节点 j 的负载 L_j = sum_{i: assignment[i]=j} P[i, j]，
最大完工时间 C_max = max_j L_j。
"""

from __future__ import annotations

import numpy as np


def check_feasibility(P: np.ndarray, assignment: np.ndarray) -> bool:
    """检查一个分配是否可行。

    条件：
    1. P 为二维数组，且所有执行时间为正的有限数值；
    2. assignment 为长度 n 的一维整数数组；
    3. 每个元素在 [0, m-1] 范围内。

    （assignment 中每个任务恰好对应一个位置，任务唯一分配自动满足。）

    Args:
        P: 执行时间矩阵 (n, m)。
        assignment: 分配向量 (n,)。

    Returns:
        可行返回 True，否则 False。
    """
    P_arr = np.asarray(P)
    if P_arr.ndim != 2:
        return False
    n, m = P_arr.shape
    if n == 0 or m == 0:
        return False
    if not (np.all(np.isfinite(P_arr)) and np.all(P_arr > 0)):
        return False

    a = np.asarray(assignment)
    if a.ndim != 1 or a.shape[0] != n:
        return False
    if not np.issubdtype(a.dtype, np.integer):
        return False
    if np.any(a < 0) or np.any(a >= m):
        return False
    return True


def compute_loads(P: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    """根据分配计算每个节点的负载。

    Args:
        P: 执行时间矩阵 (n, m)。
        assignment: 分配向量 (n,)，元素为节点索引。

    Returns:
        形状 (m,) 的 float64 数组，第 j 个元素为节点 j 的负载 L_j。

    Raises:
        ValueError: P 非二维、assignment 形状不符、类型非整数或元素越界。
    """
    P_arr = np.asarray(P, dtype=np.float64)
    if P_arr.ndim != 2:
        raise ValueError(f"执行时间矩阵必须是二维数组，实际维度: {P_arr.ndim}")
    n, m = P_arr.shape
    if m == 0:
        raise ValueError("节点数必须大于 0")

    a = np.asarray(assignment)
    if a.ndim != 1 or a.shape[0] != n:
        raise ValueError(f"assignment 形状应为 ({n},)，实际为 {a.shape}")
    if not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"assignment 必须是整数数组，实际 dtype: {a.dtype}")
    if np.any(a < 0) or np.any(a >= m):
        raise ValueError(f"assignment 中存在越界节点索引，应在 [0, {m - 1}] 内")

    # 每个任务的执行时间取 P[i, assignment[i]]，按节点累加
    task_times = P_arr[np.arange(n), a]
    loads = np.bincount(a, weights=task_times, minlength=m)
    return np.asarray(loads, dtype=np.float64)


def compute_cmax(P: np.ndarray, assignment: np.ndarray) -> float:
    """计算最大完工时间 C_max = max_j L_j。

    Args:
        P: 执行时间矩阵 (n, m)。
        assignment: 分配向量 (n,)。

    Returns:
        C_max 值。
    """
    loads = compute_loads(P, assignment)
    return float(np.max(loads))


def verify_solution(
    P: np.ndarray,
    assignment: np.ndarray,
    loads: np.ndarray,
    cmax: float,
    tol: float = 1e-9,
) -> bool:
    """综合核验解的一致性。

    检查：
    1. assignment 可行（长度、范围、P 为正）；
    2. loads 与由 assignment 重新计算的负载一致（容差 tol）；
    3. cmax 与 loads 的最大值一致（容差 tol）。

    Args:
        P: 执行时间矩阵 (n, m)。
        assignment: 分配向量 (n,)。
        loads: 待核验的节点负载 (m,)。
        cmax: 待核验的最大完工时间。
        tol: 比较容差（绝对与相对容差均为 tol）。

    Returns:
        全部一致返回 True，否则 False。
    """
    if not check_feasibility(P, assignment):
        return False

    expected_loads = compute_loads(P, assignment)
    loads_arr = np.asarray(loads, dtype=np.float64)
    if loads_arr.ndim != 1 or loads_arr.shape != expected_loads.shape:
        return False
    if not np.allclose(loads_arr, expected_loads, rtol=tol, atol=tol):
        return False

    try:
        cmax_val = float(cmax)
    except (TypeError, ValueError):
        return False
    if abs(cmax_val - float(np.max(expected_loads))) > tol:
        return False
    return True
