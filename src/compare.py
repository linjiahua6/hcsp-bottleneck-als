"""批量实验对比工具：同一实例上运行普通、瓶颈与自适应局部搜索。"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .adaptive import AdaptiveConfig, DEFAULT_CONFIG
from .adaptive_search import adaptive_bottleneck_search
from .initial_solutions import random_greedy
from .local_search import bottleneck_local_search, plain_local_search


def run_comparison(
    P: np.ndarray,
    initial_fn: Callable,
    max_evals: int,
    K: int = 8,
    R: int = 4,
    S: int = 8,
    seed: int = 42,
) -> dict:
    """对同一实例运行普通局部搜索与瓶颈局部搜索并返回结果。

    两个搜索共用同一个初解，各自独立使用 max_evals 评估预算。

    Args:
        P: 执行时间矩阵 (n, m)。
        initial_fn: 初始解函数（如 src.initial_solutions.min_min 或
            random_greedy），返回 (assignment, loads, cmax)。
        max_evals: 每个搜索的评估预算（两个搜索各用一份）。
        K: bottleneck 搜索每轮瓶颈节点候选任务数。
        R: 每个候选任务的目标节点数。
        S: 每个目标节点的交换候选任务数。
        seed: 随机种子（仅当 initial_fn 为 random_greedy 时使用）。

    Returns:
        dict，包含三个子 dict：
        - "initial": {"assignment", "loads", "cmax"}
        - "plain": plain_local_search 的完整结果
        - "bottleneck": bottleneck_local_search 的完整结果
    """
    if initial_fn is random_greedy:
        assignment, loads, cmax = initial_fn(P, seed=seed)
    else:
        assignment, loads, cmax = initial_fn(P)

    plain_res = plain_local_search(
        P, assignment, loads, cmax, max_evals=max_evals
    )
    bottle_res = bottleneck_local_search(
        P, assignment, loads, cmax, K=K, R=R, S=S, max_evals=max_evals
    )

    return {
        "initial": {
            "assignment": np.asarray(assignment, dtype=np.int64),
            "loads": np.asarray(loads, dtype=np.float64),
            "cmax": float(cmax),
        },
        "plain": plain_res,
        "bottleneck": bottle_res,
    }


def run_adaptive_comparison(
    P: np.ndarray,
    initial_fn: Callable,
    max_evals: int,
    config: AdaptiveConfig | None = None,
    seed: int = 42,
) -> dict:
    """对同一实例运行 plain、固定规模 bottleneck、自适应 bottleneck 三个搜索。

    三个搜索共用同一个初解，各自独立使用 max_evals 评估预算。
    固定规模搜索使用默认 K=8, R=4, S=8；自适应搜索使用给定 config。

    Args:
        P: 执行时间矩阵 (n, m)。
        initial_fn: 初始解函数（如 src.initial_solutions.min_min 或
            random_greedy），返回 (assignment, loads, cmax)。
        max_evals: 每个搜索的评估预算（三个搜索各用一份）。
        config: 自适应搜索配置（None 时使用 DEFAULT_CONFIG）。
        seed: 随机种子（仅当 initial_fn 为 random_greedy 时使用）。

    Returns:
        dict，包含四个子 dict：
        - "initial": {"assignment", "loads", "cmax"}
        - "plain": plain_local_search 的完整结果
        - "fixed": bottleneck_local_search 的完整结果（K=8, R=4, S=8）
        - "adaptive": adaptive_bottleneck_search 的完整结果
    """
    if initial_fn is random_greedy:
        assignment, loads, cmax = initial_fn(P, seed=seed)
    else:
        assignment, loads, cmax = initial_fn(P)

    if config is None:
        config = DEFAULT_CONFIG

    plain_res = plain_local_search(
        P, assignment, loads, cmax, max_evals=max_evals
    )
    fixed_res = bottleneck_local_search(
        P, assignment, loads, cmax, K=8, R=4, S=8, max_evals=max_evals
    )
    adaptive_res = adaptive_bottleneck_search(
        P, assignment, loads, cmax, config=config, max_evals=max_evals
    )

    return {
        "initial": {
            "assignment": np.asarray(assignment, dtype=np.int64),
            "loads": np.asarray(loads, dtype=np.float64),
            "cmax": float(cmax),
        },
        "plain": plain_res,
        "fixed": fixed_res,
        "adaptive": adaptive_res,
    }
