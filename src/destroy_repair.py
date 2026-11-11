"""破坏—重建（Destroy-and-Repair）增强的瓶颈邻域搜索。

模块包含两部分：
1. `destroy_repair`：对当前解执行一次破坏（选一批任务移出分配）与重建
   （将这批任务随机重新分配），返回可行的新解三元组 (assignment, loads, cmax)。
2. `bottleneck_local_search_dr`：迭代局部搜索（ILS）框架——反复执行
   "瓶颈局部搜索 → 破坏重建 → 瓶颈局部搜索"，帮助跳出局部最优。

设计选择说明（与本项目既有约定保持一致）：
- 破坏阶段不消耗评估预算（evals 只统计局部搜索的候选评估）；
- 每段局部搜索用 `segment_evals` 预算（截断到剩余总预算），段内仍为
  best-improvement、严格改善接受、无改善即停；
- 严格模式（accept_strategy="strict"，默认，经典 ILS）：只有找到优于
  全局最优的解才更新，扰动始终从全局最优出发；
- 模拟退火模式（accept_strategy="metropolis"）：按 Metropolis 准则接受
  变差解（exp(-Δ/T)，T = 0.05 * c0，每段按 cooling_rate 退火），扰动从
  当前解出发；
- destroy_strategy="bottleneck"（默认）：按节点负载降序、节点内按
  P[i, j] 降序选破坏任务，集中破坏瓶颈区域；"random"：随机选任务；
- 重建为均匀随机分配，且**不允许任务回到原节点**（保证扰动非平凡；
  m = 1 时无替代节点，保持原分配）；
- 连续 max_stagnation 段未改善全局最优时提前停止；
- 输入核验、增量更新、返回标准字段与 bottleneck_local_search 一致，
  额外增加 restarts / improvements / accepted_worse / segments 四个统计字段。

历史记录 history 为各段局部搜索 history 的拼接，round 全局递增。
"""

from __future__ import annotations

import math
import time

import numpy as np

from .feasibility import compute_loads
from .local_search import _check_input, _make_result, bottleneck_local_search


def destroy_repair(
    P: np.ndarray,
    assignment: np.ndarray,
    loads: np.ndarray,
    cmax: float,
    destroy_frac: float = 0.2,
    strategy: str = "bottleneck",
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """破坏—重建：移出一批任务并随机重新分配，返回新解三元组。

    破坏集大小 q = max(1, round(destroy_frac * n))（不超过 n）。

    - strategy="bottleneck"（默认）：按节点负载降序遍历节点，每个节点内
      按 P[i, j] 降序收集任务，直到凑满 q 个（破坏集中于高负载/瓶颈区域）；
    - strategy="random"：均匀随机选 q 个任务。

    重建：q 个任务各自均匀随机分配到除原节点外的任一节点
    （不允许回到原节点；m = 1 时保持原分配）。

    Args:
        P: 执行时间矩阵 (n, m)。
        assignment: 当前分配 (n,)。
        loads: 当前负载 (m,)。
        cmax: 当前 C_max。
        destroy_frac: 破坏比例，须在 (0, 1]。
        strategy: 破坏策略，取值 "bottleneck" 或 "random"。
        seed: 随机种子（rng 为 None 时生效）。
        rng: 可选随机数生成器（给定则忽略 seed，供调用方复用同一生成器）。

    Returns:
        (new_assignment, new_loads, new_cmax)，与输入一致的可行解。
    """
    if not 0 < destroy_frac <= 1:
        raise ValueError(f"destroy_frac 必须在 (0, 1]，实际: {destroy_frac}")
    if strategy not in ("bottleneck", "random"):
        raise ValueError(f"strategy 必须为 'bottleneck' 或 'random'，实际: {strategy!r}")

    P_arr = np.asarray(P, dtype=np.float64)
    n, m = P_arr.shape
    a = np.asarray(assignment, dtype=np.int64).copy()
    loads_arr = np.asarray(loads, dtype=np.float64).copy()
    cur = float(cmax)
    _check_input(P_arr, a, loads_arr, cur, tol=1e-9)

    if rng is None:
        rng = np.random.default_rng(seed)

    q = max(1, int(round(destroy_frac * n)))
    q = min(q, n)

    if strategy == "random":
        picked = rng.choice(n, size=q, replace=False)
    else:
        # 瓶颈导向：负载降序遍历节点，节点内按执行时间降序取任务
        picked_list: list[int] = []
        node_order = np.argsort(-loads_arr, kind="stable")
        for j in node_order:
            if len(picked_list) >= q:
                break
            tasks_j = np.nonzero(a == j)[0]
            if tasks_j.size == 0:
                continue
            p_order = np.argsort(-P_arr[tasks_j, j], kind="stable")
            for t in tasks_j[p_order]:
                if len(picked_list) >= q:
                    break
                picked_list.append(int(t))
        picked = np.asarray(picked_list, dtype=np.int64)

    a_new = a.copy()
    if m > 1:
        for idx, t in enumerate(picked):
            # 从除原节点外的节点中均匀随机选一个
            choices = np.arange(m)[np.arange(m) != a[t]]
            a_new[t] = int(rng.choice(choices))
    loads_new = compute_loads(P_arr, a_new)
    cmax_new = float(loads_new.max())
    return a_new, loads_new, cmax_new


def bottleneck_local_search_dr(
    P: np.ndarray,
    assignment: np.ndarray,
    loads: np.ndarray,
    cmax: float,
    K: int = 4,
    R: int = 2,
    S: int = 4,
    max_evals: int = 10000,
    tol: float = 1e-9,
    destroy_frac: float = 0.2,
    destroy_strategy: str = "bottleneck",
    accept_strategy: str = "strict",
    segment_evals: int = 1000,
    max_stagnation: int = 3,
    cooling_rate: float = 0.8,
    seed: int = 42,
    time_budget: float | None = None,
) -> dict:
    """破坏—重建增强的瓶颈邻域搜索（ILS 框架）。

    流程：对当前解运行瓶颈局部搜索（每段 segment_evals 预算）→
    若优于全局最优则更新并重置停滞计数 → 破坏重建生成扰动解 →
    重复，直到总预算耗尽或连续 max_stagnation 段未改善全局最优。

    默认 K=4, R=2, S=4（验证阶段 Q5 显示该小邻域配置在固定评估预算下
    优于 K=8, R=4, S=8）。

    Args:
        P: 执行时间矩阵 (n, m)。
        assignment: 初始分配 (n,)。
        loads: 初始负载 (m,)。
        cmax: 初始 C_max。
        K, R, S: 瓶颈邻域参数（每段局部搜索使用）。
        max_evals: 总评估预算（>= 0，破坏重建不消耗预算）。
        tol: 浮点比较容差。
        destroy_frac: 每次破坏的任务比例 (0, 1]。
        destroy_strategy: "bottleneck"（默认）或 "random"。
        accept_strategy: "strict"（默认，经典 ILS）或 "metropolis"。
        segment_evals: 每段局部搜索的评估预算（> 0）。
        max_stagnation: 连续未改善全局最优的最大段数（> 0）。
        cooling_rate: Metropolis 温度退火率 (0, 1]。
        seed: 随机种子（破坏重建用，保证可复现）。
        time_budget: 可选时间预算（秒），段边界检查，与 max_evals 先到者为准。

    Returns:
        标准结果 dict（assignment/loads/cmax/evals/accepted_moves/improved/history），
        额外增加：
        - "restarts": 破坏重建次数；
        - "improvements": 破坏后找到更优全局最优的段数；
        - "accepted_worse": Metropolis 接受变差解的次数（strict 恒为 0）；
        - "segments": 局部搜索段数。
    """
    if destroy_frac <= 0 or destroy_frac > 1:
        raise ValueError(f"destroy_frac 必须在 (0, 1]，实际: {destroy_frac}")
    if destroy_strategy not in ("bottleneck", "random"):
        raise ValueError(
            f"destroy_strategy 必须为 'bottleneck' 或 'random'，实际: {destroy_strategy!r}"
        )
    if accept_strategy not in ("strict", "metropolis"):
        raise ValueError(
            f"accept_strategy 必须为 'strict' 或 'metropolis'，实际: {accept_strategy!r}"
        )
    if segment_evals <= 0:
        raise ValueError(f"segment_evals 必须为正整数，实际: {segment_evals}")
    if max_stagnation <= 0:
        raise ValueError(f"max_stagnation 必须为正整数，实际: {max_stagnation}")
    if not 0 < cooling_rate <= 1:
        raise ValueError(f"cooling_rate 必须在 (0, 1]，实际: {cooling_rate}")

    P_arr = np.asarray(P, dtype=np.float64)
    a = np.asarray(assignment, dtype=np.int64).copy()
    loads_arr = np.asarray(loads, dtype=np.float64).copy()
    c0 = float(cmax)
    _check_input(P_arr, a, loads_arr, c0, tol)

    evals = 0
    accepted_total = 0
    history: list[dict] = []
    restarts = 0
    improvements = 0
    accepted_worse = 0
    segments = 0

    best_a = a.copy()
    best_loads = loads_arr.copy()
    best_c = c0
    base_a = a.copy()
    base_loads = loads_arr.copy()
    base_c = c0

    rng = np.random.default_rng(seed)
    T = 0.05 * c0 if accept_strategy == "metropolis" else None
    stagnation = 0
    t_start = time.perf_counter()

    while evals < max_evals:
        if time_budget is not None and time.perf_counter() - t_start >= time_budget:
            break

        seg_budget = min(segment_evals, max_evals - evals)
        if seg_budget <= 0:
            break

        res = bottleneck_local_search(
            P_arr, base_a, base_loads, base_c,
            K=K, R=R, S=S, max_evals=seg_budget, tol=tol,
        )
        segments += 1
        evals += int(res["evals"])
        accepted_total += int(res["accepted_moves"])
        for entry in res["history"]:
            history.append({**entry, "round": len(history) + 1})

        seg_a = np.asarray(res["assignment"], dtype=np.int64)
        seg_loads = np.asarray(res["loads"], dtype=np.float64)
        seg_c = float(res["cmax"])

        if seg_c < best_c - tol:
            best_a = seg_a.copy()
            best_loads = seg_loads.copy()
            best_c = seg_c
            improvements += 1
            stagnation = 0
            if accept_strategy == "metropolis":
                # 当前解跟随段结果（后续从它扰动）
                base_a = seg_a.copy()
                base_loads = seg_loads.copy()
                base_c = seg_c
        else:
            stagnation += 1
            if accept_strategy == "metropolis" and T is not None:
                delta = seg_c - base_c
                accept = delta <= 0
                if not accept and T > 0:
                    accept = rng.random() < math.exp(-delta / T)
                if accept:
                    base_a = seg_a.copy()
                    base_loads = seg_loads.copy()
                    base_c = seg_c
                    if delta > 0:
                        accepted_worse += 1

        if stagnation >= max_stagnation:
            break
        if evals >= max_evals:
            break

        # 破坏重建：strict 从全局最优扰动；metropolis 从当前解扰动
        if accept_strategy == "strict":
            pert_a, pert_loads, pert_c = best_a, best_loads, best_c
        else:
            pert_a, pert_loads, pert_c = base_a, base_loads, base_c
        base_a, base_loads, base_c = destroy_repair(
            P_arr,
            pert_a,
            pert_loads,
            pert_c,
            destroy_frac=destroy_frac,
            strategy=destroy_strategy,
            rng=rng,
        )
        restarts += 1
        if T is not None:
            T = max(T * cooling_rate, 1e-12)

    out = _make_result(best_a, best_loads, best_c, c0, evals, accepted_total, history, tol)
    out["restarts"] = restarts
    out["improvements"] = improvements
    out["accepted_worse"] = accepted_worse
    out["segments"] = segments
    return out
