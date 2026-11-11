"""自适应瓶颈邻域搜索：每轮按 rho、eta 动态调节候选规模 (K, R, S)。

与 bottleneck_local_search 结构一致（每轮 best-improvement、严格改善接受、
增量负载更新、入口核验），区别仅在于每轮规模由自适应调节器给出：

    rho = bottleneck_sharpness(loads)
    eta = compute_progress(history, config.window)
    K, R, S = adaptive_sizes(rho, eta, ..., m=P.shape[1])

返回 dict 在标准字段基础上额外增加：
    - "config":        config 的 asdict()（便于实验记录）；
    - "final_sizes":   最后一轮使用的 (K, R, S)；无任何轮执行时为 None。
history 每轮一条：
    {"round", "cmax", "evals", "accepted", "rho", "eta", "K", "R", "S"}

设计选择说明：
- 复用 src/local_search.py 的 _evaluate_round（与 bottleneck_local_search
  共享核心枚举，行为一致），不复制迁移/交换评估代码；
- rho、eta 基于执行本轮动作之前的 loads 与 history 计算（history 含
  上一轮的记录），与规范一致；
- config 为不可变 dataclass，可直接作为默认参数；
- 其他行为（接受条件、增量更新、输入核验、返回字段）与
  bottleneck_local_search 保持一致。
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .adaptive import AdaptiveConfig, DEFAULT_CONFIG, adaptive_sizes, compute_progress
from .bottleneck import bottleneck_sharpness
from .local_search import MOVE, SWAP, _check_input, _evaluate_round, _make_result


def adaptive_bottleneck_search(
    P: np.ndarray,
    assignment: np.ndarray,
    loads: np.ndarray,
    cmax: float,
    config: AdaptiveConfig = DEFAULT_CONFIG,
    max_evals: int = 10000,
    tol: float = 1e-9,
) -> dict:
    """自适应瓶颈邻域搜索。

    每轮：
    1. 定位瓶颈节点 b；
    2. rho = bottleneck_sharpness(loads)；
    3. eta = compute_progress(history, config.window)；
    4. (K, R, S) = adaptive_sizes(rho, eta, ..., m)；
    5. 按 (K, R, S) 枚举迁移与交换候选（复用 _evaluate_round）；
    6. 执行本轮最优严格改善动作；无改善则停止；
    7. history 追加 {"round", "cmax", "evals", "accepted",
       "rho", "eta", "K", "R", "S"}。

    Args:
        P: 执行时间矩阵 (n, m)。
        assignment: 初始分配 (n,)。
        loads: 初始负载 (m,)。
        cmax: 初始 C_max。
        config: 自适应配置（src.adaptive.AdaptiveConfig）。
        max_evals: 候选评估预算（>= 0）。
        tol: 浮点比较容差。

    Returns:
        dict：标准字段 + "config"（asdict）+ "final_sizes"（最后一轮 (K, R, S)）。
    """
    P_arr = np.asarray(P, dtype=np.float64)
    _, m = P_arr.shape
    a = np.asarray(assignment, dtype=np.int64).copy()
    loads_arr = np.asarray(loads, dtype=np.float64).copy()
    c0 = float(cmax)
    cur = c0
    _check_input(P_arr, a, loads_arr, cur, tol)

    evals = 0
    accepted = 0
    history: list[dict] = []
    final_sizes: tuple[int, int, int] | None = None

    while evals < max_evals:
        round_no = len(history) + 1

        rho = bottleneck_sharpness(loads_arr)
        eta = compute_progress(history, config.window)
        K, R, S = adaptive_sizes(
            rho,
            eta,
            K0=config.K0,
            R0=config.R0,
            S0=config.S0,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            K_min=config.K_min,
            K_max=config.K_max,
            R_min=config.R_min,
            R_max=config.R_max,
            S_min=config.S_min,
            S_max=config.S_max,
            m=m,
        )
        final_sizes = (K, R, S)

        best_action, best_c, evals = _evaluate_round(
            P_arr, a, loads_arr, cur, K, R, S, max_evals, evals, tol
        )

        if best_action is None:
            history.append(
                {
                    "round": round_no,
                    "cmax": cur,
                    "evals": evals,
                    "accepted": 0,
                    "rho": rho,
                    "eta": eta,
                    "K": K,
                    "R": R,
                    "S": S,
                }
            )
            break  # 无严格改善（或瓶颈节点无任务/预算耗尽），停止

        # 执行动作（增量更新 assignment 与 loads）
        if best_action[0] == MOVE:
            _, i, dst = best_action
            src = int(a[i])
            a[i] = dst
            loads_arr[src] -= P_arr[i, src]
            loads_arr[dst] += P_arr[i, dst]
        else:
            _, i, dst, h = best_action
            src = int(a[i])
            a[i] = dst
            a[h] = src
            loads_arr[src] += P_arr[h, src] - P_arr[i, src]
            loads_arr[dst] += P_arr[i, dst] - P_arr[h, dst]

        cur = best_c
        accepted += 1
        history.append(
            {
                "round": round_no,
                "cmax": cur,
                "evals": evals,
                "accepted": 1,
                "rho": rho,
                "eta": eta,
                "K": K,
                "R": R,
                "S": S,
            }
        )

    res = _make_result(a, loads_arr, cur, c0, evals, accepted, history, tol)
    res["config"] = asdict(config)
    res["final_sizes"] = final_sizes
    return res
