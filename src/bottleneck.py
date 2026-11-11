"""瓶颈分析与候选任务选择。

瓶颈（bottleneck）指当前负载最大的节点。瓶颈导向搜索的核心思想是：
每次迭代只针对瓶颈节点上的任务生成移动/交换候选，从而把评估预算
集中在最可能带来 C_max 改善的地方。

设计选择说明：
- find_bottleneck 在并列瓶颈时返回编号最小的节点（确定性，保证可复现）；
- bottleneck_sharpness 中的"第二大负载"按降序排序后取第 2 个元素，
  若多个节点并列最大则 rho = 0；
- select_candidate_tasks 使用稳定排序（np.argsort kind="stable"），
  并列时保持任务编号升序。
"""

from __future__ import annotations

import numpy as np


def find_bottleneck(loads: np.ndarray) -> int:
    """返回负载最大的节点索引；并列时返回编号最小的节点。

    Args:
        loads: 形状 (m,) 的节点负载数组。

    Returns:
        瓶颈节点索引。

    Raises:
        ValueError: loads 为空。
    """
    loads = np.asarray(loads, dtype=np.float64)
    if loads.size == 0:
        raise ValueError("loads 不能为空")
    # np.argmax 在并列时返回第一个最大值的索引，即编号最小的节点
    return int(np.argmax(loads))


def bottleneck_sharpness(loads: np.ndarray) -> float:
    """计算瓶颈尖锐度 rho = (L_max - L_second) / L_max。

    L_max 为最大负载，L_second 为第二大负载（允许与 L_max 并列）。

    Args:
        loads: 形状 (m,) 的节点负载数组。

    Returns:
        尖锐度 rho（范围 [0, 1]）；若 m <= 1 或 L_max <= 0，返回 0.0。
    """
    loads = np.asarray(loads, dtype=np.float64)
    m = loads.size
    if m <= 1:
        return 0.0
    l_max = float(np.max(loads))
    if l_max <= 0:
        return 0.0
    sorted_desc = np.sort(loads)[::-1]
    l_second = float(sorted_desc[1])
    return (l_max - l_second) / l_max


def select_candidate_tasks(
    P: np.ndarray, assignment: np.ndarray, node: int, K: int
) -> np.ndarray:
    """从节点 node 上的任务中按 P[i, node] 降序选前 K 个。

    任务在瓶颈节点上的执行时间越大，移走它对降低瓶颈负载越有利。

    Args:
        P: 执行时间矩阵 (n, m)。
        assignment: 分配向量 (n,)。
        node: 目标节点索引。
        K: 最多选取的任务数（K <= 0 时返回空数组）。

    Returns:
        形状 (k,) 的 int64 任务索引数组，k = min(K, 该节点任务数)，
        按 P[i, node] 降序排列；若该节点没有任务，返回空数组。

    Raises:
        ValueError: node 越界。
    """
    P_arr = np.asarray(P, dtype=np.float64)
    a = np.asarray(assignment)
    m = P_arr.shape[1]
    if node < 0 or node >= m:
        raise ValueError(f"节点索引 node 应在 [0, {m - 1}] 内，实际: {node}")
    if K <= 0:
        return np.empty(0, dtype=np.int64)

    tasks = np.nonzero(a == node)[0]
    if tasks.size == 0:
        return np.empty(0, dtype=np.int64)

    # 稳定降序排序：P 值并列时保持任务编号升序（可复现）
    order = np.argsort(-P_arr[tasks, node], kind="stable")
    ranked = tasks[order]
    return ranked[:K].astype(np.int64)
