"""局部搜索：普通局部搜索（基线）与瓶颈导向局部搜索。

两个搜索函数接口统一，均返回 dict：
    {
        "assignment":      最终分配 (n,)，
        "loads":           最终负载 (m,)，
        "cmax":            最终 C_max，
        "evals":           实际候选评估次数，
        "accepted_moves":  接受动作数，
        "improved":        相对输入是否严格改善（bool），
        "history":         [{"round", "cmax", "evals", "accepted"}, ...]，
    }

设计选择说明：
- 两搜索均为"每轮 best-improvement"：一轮内枚举候选并记录 C'_max 最小者，
  轮末执行；轮内负载不变，因此评估用缓冲数组 + 增量修改两个位置，
  不重算全部节点负载。
- 接受条件：C'_max < C_max - tol（严格改善）。
- 平局动作取枚举顺序中先出现者（确定性）。
- history 中 "accepted" 为该轮接受的动作数（两搜索每轮至多 1）。
- 输入 (assignment, loads, cmax) 在函数开头用 verify_solution 核验，
  不一致时抛 ValueError；搜索过程不再重复调用（由测试核验最终解）。
- 瓶颈搜索的轮内枚举抽取为私有函数 _evaluate_round，供
  src/adaptive_search.py 的自适应搜索复用（行为一致，不重复实现）。
"""

from __future__ import annotations

import time

import numpy as np

from .bottleneck import find_bottleneck, select_candidate_tasks
from .feasibility import compute_loads, verify_solution

MOVE = "move"
SWAP = "swap"


def _check_input(P: np.ndarray, assignment: np.ndarray, loads: np.ndarray, cmax: float, tol: float) -> None:
    """核验输入的 (assignment, loads, cmax) 三者一致。"""
    if not verify_solution(P, assignment, loads, cmax, tol=tol):
        raise ValueError("输入初解不一致：assignment、loads 与 cmax 不匹配或不可行")


def _make_result(
    assignment: np.ndarray,
    loads: np.ndarray,
    cmax: float,
    c0: float,
    evals: int,
    accepted: int,
    history: list,
    tol: float,
) -> dict:
    """组装标准返回字典。"""
    return {
        "assignment": assignment,
        "loads": loads,
        "cmax": float(cmax),
        "evals": evals,
        "accepted_moves": accepted,
        "improved": bool(cmax < c0 - tol),
        "history": history,
    }


def plain_local_search(
    P: np.ndarray,
    assignment: np.ndarray,
    loads: np.ndarray,
    cmax: float,
    max_evals: int,
    tol: float = 1e-9,
    time_budget: float | None = None,
) -> dict:
    """普通局部搜索（基线）：按固定顺序枚举单任务迁移。

    每轮固定顺序枚举：外层任务 0..n-1，内层目标节点 0..m-1（跳过当前
    节点）。对每个任务-节点对评估单任务迁移（增量计算 C'_max）。一轮
    枚举完成后执行该轮 C'_max 最小的严格改善动作；若整轮无严格改善则
    停止。达到 max_evals 时停止评估。

    Args:
        P: 执行时间矩阵 (n, m)。
        assignment: 初始分配 (n,)。
        loads: 初始负载 (m,)。
        cmax: 初始 C_max。
        max_evals: 候选评估预算（>= 0）。
        tol: 浮点比较容差。
        time_budget: 可选时间预算（秒）。None 表示不限制；给定后，
            在每轮边界检查已用时间，超过预算即停止（与 max_evals
            先到者为准）。默认 None，行为与旧版本一致。

    Returns:
        标准结果 dict（见模块文档字符串）。
    """
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
        buf = loads.copy()  # 每轮枚举期间 loads 不变，缓冲数组只改两个位置

        best_c = np.inf
        best_i = -1
        best_j = -1

        for i in range(n):
            if evals >= max_evals:
                break
            src = int(a[i])
            p_src = P_arr[i, src]
            for j in range(m):
                if j == src:
                    continue
                if evals >= max_evals:
                    break
                # 增量评估迁移：只改 src 与 j 两个位置的负载
                buf[src] = loads[src] - p_src
                buf[j] = loads[j] + P_arr[i, j]
                cand_cmax = float(buf.max())
                buf[src] = loads[src]
                buf[j] = loads[j]
                evals += 1

                if cand_cmax < cur - tol and cand_cmax < best_c - tol:
                    best_c = cand_cmax
                    best_i = i
                    best_j = j

        if best_i >= 0:
            # 执行本轮最优动作（严格改善）
            src = int(a[best_i])
            a[best_i] = best_j
            loads[src] -= P_arr[best_i, src]
            loads[best_j] += P_arr[best_i, best_j]
            cur = best_c
            accepted += 1
            history.append(
                {"round": round_no, "cmax": cur, "evals": evals, "accepted": 1}
            )
        else:
            history.append(
                {"round": round_no, "cmax": cur, "evals": evals, "accepted": 0}
            )
            break  # 无严格改善，停止

    # 增量更新与 bincount 的浮点顺序不同（真实数据大值域下差异可达
    # 1e-9 量级），返回前重算保证与下游核验（tol=1e-9）严格一致。
    loads = compute_loads(P_arr, a)
    cur = float(np.max(loads))
    return _make_result(a, loads, cur, c0, evals, accepted, history, tol)


def _evaluate_round(
    P_arr: np.ndarray,
    a: np.ndarray,
    loads: np.ndarray,
    cur: float,
    K: int,
    R: int,
    S: int,
    max_evals: int,
    evals: int,
    tol: float,
) -> tuple:
    """枚举一轮候选（迁移 + 交换），不修改当前解。

    与 bottleneck_local_search 的轮内逻辑一致：
    - find_bottleneck 定位瓶颈节点 b；
    - select_candidate_tasks 从 b 上选 K 个候选任务（按 P[i, b] 降序）；
    - 每任务按 L_j + P[i, j] 升序选 R 个目标节点评估迁移；
    - 每目标节点按 P[h, j] 降序选 S 个任务评估交换；
    - 只保留严格改善（C' < cur - tol）且 C' 最小者；
    - evals >= max_evals 时停止枚举。

    Returns:
        (best_action, best_c, evals)：
        - best_action: (MOVE, i, dst) / (SWAP, i, dst, h) / None；
        - best_c: 最优动作对应的 C'_max（无动作时为 inf）；
        - evals: 本轮枚举结束后的累计评估数。
    """
    m = P_arr.shape[1]
    buf = loads.copy()  # 每轮枚举期间 loads 不变，缓冲数组只改两个位置

    b = find_bottleneck(loads)
    cand_tasks = select_candidate_tasks(P_arr, a, b, K)
    if cand_tasks.size == 0:
        return None, np.inf, evals

    # 每轮预计算各节点上的任务（按 P[·, j] 降序），供交换候选复用
    sorted_tasks_of: list[np.ndarray] = []
    for j in range(m):
        tasks_j = np.nonzero(a == j)[0]
        if tasks_j.size > 0:
            order = np.argsort(-P_arr[tasks_j, j], kind="stable")
            sorted_tasks_of.append(tasks_j[order])
        else:
            sorted_tasks_of.append(np.empty(0, dtype=np.int64))

    best_c = np.inf
    best_action: tuple | None = None  # (MOVE, i, dst) 或 (SWAP, i, dst, h)

    for i in cand_tasks:
        if evals >= max_evals:
            break
        src = int(a[i])
        p_i_src = P_arr[i, src]

        # 目标节点：按迁移后目标负载 L_j + P[i, j] 升序，跳过 src，取前 R
        dst_order = np.argsort(loads + P_arr[i], kind="stable")
        dst_list = [int(j) for j in dst_order if j != src][:R]

        for dst in dst_list:
            if evals >= max_evals:
                break

            # --- 迁移评估：i 从 src 移到 dst ---
            buf[src] = loads[src] - p_i_src
            buf[dst] = loads[dst] + P_arr[i, dst]
            cand_cmax = float(buf.max())
            buf[src] = loads[src]
            buf[dst] = loads[dst]
            evals += 1
            if cand_cmax < cur - tol and cand_cmax < best_c - tol:
                best_c = cand_cmax
                best_action = (MOVE, int(i), dst)

            if evals >= max_evals:
                break

            # --- 交换评估：i 移到 dst，h 从 dst 移到 src ---
            swap_cands = sorted_tasks_of[dst][:S]
            for h in swap_cands:
                if evals >= max_evals:
                    break
                h = int(h)
                if h == i:  # 理论上不会发生（i 在 src，h 在 dst，src != dst）
                    continue
                buf[src] = loads[src] - p_i_src + P_arr[h, src]
                buf[dst] = loads[dst] - P_arr[h, dst] + P_arr[i, dst]
                cand_cmax = float(buf.max())
                buf[src] = loads[src]
                buf[dst] = loads[dst]
                evals += 1
                if cand_cmax < cur - tol and cand_cmax < best_c - tol:
                    best_c = cand_cmax
                    best_action = (SWAP, int(i), dst, h)

    return best_action, best_c, evals


def bottleneck_local_search(
    P: np.ndarray,
    assignment: np.ndarray,
    loads: np.ndarray,
    cmax: float,
    K: int = 8,
    R: int = 4,
    S: int = 8,
    max_evals: int = 10000,
    tol: float = 1e-9,
    time_budget: float | None = None,
) -> dict:
    """瓶颈导向局部搜索。

    每轮：
    1. find_bottleneck 定位瓶颈节点 b；
    2. select_candidate_tasks 从 b 上选 K 个候选任务（按 P[i, b] 降序）；
    3. 对每个候选任务 i，按迁移后目标负载 L_j + P[i, j] 升序选前 R 个
       目标节点（跳过当前节点 a[i]），评估单任务迁移；
    4. 对每个目标节点 b'，在其任务中按 P[h, b'] 降序选前 S 个任务 h，
       评估交换（i 到 b'，h 到 a[i]）；
    5. 只保留严格改善（C'_max < C_max - tol）的候选，取 C'_max 最小者执行；
       若无改善则停止；
    6. 负载与分配增量更新，累计 evals，达到 max_evals 停止。

    Args:
        P: 执行时间矩阵 (n, m)。
        assignment: 初始分配 (n,)。
        loads: 初始负载 (m,)。
        cmax: 初始 C_max。
        K: 每轮瓶颈节点候选任务数。
        R: 每个候选任务的目标节点数。
        S: 每个目标节点的交换候选任务数。
        max_evals: 候选评估预算（>= 0）。
        tol: 浮点比较容差。
        time_budget: 可选时间预算（秒）。None 表示不限制；给定后，
            在每轮边界检查已用时间，超过预算即停止（与 max_evals
            先到者为准）。默认 None，行为与旧版本一致。

    Returns:
        标准结果 dict（见模块文档字符串）。
    """
    if K <= 0:
        raise ValueError(f"K 必须为正整数，实际: {K}")
    if R <= 0:
        raise ValueError(f"R 必须为正整数，实际: {R}")
    if S <= 0:
        raise ValueError(f"S 必须为正整数，实际: {S}")

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
        best_action, best_c, evals = _evaluate_round(
            P_arr, a, loads, cur, K, R, S, max_evals, evals, tol
        )

        if best_action is None:
            history.append(
                {"round": round_no, "cmax": cur, "evals": evals, "accepted": 0}
            )
            break  # 无严格改善（或瓶颈节点无任务/预算耗尽），停止

        # 执行动作（增量更新 assignment 与 loads）
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
