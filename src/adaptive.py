"""自适应候选规模调节器。

第三阶段核心：根据两个反馈信号动态调节每轮候选规模 (K, R, S)：

- rho（瓶颈尖锐度，来自 bottleneck_sharpness）：rho 越大说明瓶颈越突出，
  搜索应集中火力于瓶颈 → 缩小 K；
- eta（近期接受率，来自 compute_progress）：eta 越大说明邻域仍有潜力，
  应扩大探索 → 增大 R、S。

设计选择说明：
- compute_progress 只统计最近 window 轮中 accepted > 0 的轮占比；
  history 为空时返回 0.0；window 必须为正整数。
- adaptive_sizes 先把 rho、eta 防御性截断到 [0, 1]，再套用公式：
      K = clip(K0 * (1 + alpha * (1 - rho)), K_min, K_max)
      R = clip(R0 * (1 + beta * eta),  R_min, min(R_max, m))   # m 提供时
      S = clip(S0 * (1 + gamma * eta), S_min, S_max)
  最后按"四舍五入（round-half-up）"取整，保证返回值始终落在各自区间内。
- R_max 默认 None：m 提供时上界取 m；两者都缺省时无上界，仅施加下界。
"""

from __future__ import annotations

from dataclasses import dataclass


def compute_progress(history: list, window: int = 5) -> float:
    """计算近期接受率 eta。

    eta =（最近 window 轮中 accepted > 0 的轮数）/（轮数）。
    轮数不足 window 时取全部；history 为空时返回 0.0。

    Args:
        history: 搜索历史，每轮一条 dict，须含 "accepted" 字段。
        window: 观察窗口大小（正整数）。

    Returns:
        eta（范围 [0, 1]）。

    Raises:
        ValueError: window <= 0。
    """
    if window <= 0:
        raise ValueError(f"window 必须为正整数，实际: {window}")
    if not history:
        return 0.0
    recent = history[-window:] if len(history) > window else history
    accepted_rounds = sum(1 for entry in recent if int(entry["accepted"]) > 0)
    return accepted_rounds / len(recent)


def _clip_and_round(x: float, lo: int, hi: int | None) -> int:
    """先按 [lo, hi] 截断，再四舍五入（round-half-up）取整。

    hi 为 None 时仅施加下界 lo。
    """
    if hi is None:
        return int(max(x, float(lo)) + 0.5)
    hi = max(hi, lo)
    xc = min(max(x, float(lo)), float(hi))
    return int(xc + 0.5)


def adaptive_sizes(
    rho: float,
    eta: float,
    K0: int = 8,
    R0: int = 4,
    S0: int = 8,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.5,
    K_min: int = 2,
    K_max: int = 32,
    R_min: int = 1,
    R_max: int | None = None,
    S_min: int = 2,
    S_max: int = 32,
    m: int | None = None,
) -> tuple[int, int, int]:
    """按反馈信号计算本轮候选规模 (K, R, S)。

    规则：
        K = clip(K0 * (1 + alpha * (1 - rho)), K_min, K_max)
        R = clip(R0 * (1 + beta * eta),  R_min, R_max_eff)
        S = clip(S0 * (1 + gamma * eta), S_min, S_max)
    其中 R_max_eff = min(R_max, m)（m 提供时；R_max 缺省则取 m）。

    含义：
    - rho 越大 → K 越小（瓶颈清晰，集中火力）；
    - eta 越大 → R、S 越大（邻域仍有潜力，扩大探索）。

    Args:
        rho: 瓶颈尖锐度（越界时截断到 [0, 1]）。
        eta: 近期接受率（越界时截断到 [0, 1]）。
        K0: K 的基准值。
        R0: R 的基准值。
        S0: S 的基准值。
        alpha: rho 对 K 的调节系数。
        beta: eta 对 R 的调节系数。
        gamma: eta 对 S 的调节系数。
        K_min: K 的下界。
        K_max: K 的上界。
        R_min: R 的下界。
        R_max: R 的上界（None 时若提供 m 则取 m）。
        S_min: S 的下界。
        S_max: S 的上界。
        m: 节点数；提供时 R 的上界不超过 m。

    Returns:
        整数三元组 (K, R, S)，全部经四舍五入且落在各自区间内。

    Raises:
        ValueError: K0、R0 或 S0 非正。
    """
    if K0 <= 0:
        raise ValueError(f"K0 必须为正整数，实际: {K0}")
    if R0 <= 0:
        raise ValueError(f"R0 必须为正整数，实际: {R0}")
    if S0 <= 0:
        raise ValueError(f"S0 必须为正整数，实际: {S0}")

    rho_c = min(max(float(rho), 0.0), 1.0)
    eta_c = min(max(float(eta), 0.0), 1.0)

    K_raw = K0 * (1.0 + alpha * (1.0 - rho_c))
    K = _clip_and_round(K_raw, K_min, K_max)

    # 若 m 提供，R 上界取 min(R_max 或 m, m)；否则仅用 R_max
    if m is not None:
        base = R_max if R_max is not None else m
        r_max_eff = min(base, m)
    else:
        r_max_eff = R_max
    R_raw = R0 * (1.0 + beta * eta_c)
    R = _clip_and_round(R_raw, R_min, r_max_eff)

    S_raw = S0 * (1.0 + gamma * eta_c)
    S = _clip_and_round(S_raw, S_min, S_max)

    return (K, R, S)


@dataclass(frozen=True)
class AdaptiveConfig:
    """自适应调节器配置（不可变，便于作为默认参数与实验记录）。

    字段默认值为"均值匹配 fixed 规模"的推荐配置（经快速消融诊断后选定）：
    - K ∈ [K0, K0*(1+alpha)] = [6, 9]，围绕 fixed 的 K=8 摆动；
    - R ∈ [R0, R0*(1+beta)] = [3, 4]（round 后），围绕 fixed 的 R=4；
    - S ∈ [S0, S0*(1+gamma)] = [6, 8]（round 后），围绕 fixed 的 S=8。
    平均每轮评估数 ~220，与固定规模（8×4×9=288）相当，保证两者在相同
    max_evals 预算下轮数可比，自适应机制不靠"邻域整体缩小/放大"获利。

    注意：adaptive_sizes 的签名默认值（K0=8, R0=4, S0=8, alpha=beta=gamma=0.5）
    为规范示例值，与本节字段默认值不同；adaptive_bottleneck_search 使用
    本配置调用 adaptive_sizes（显式传参），两者互不影响。

    Attributes:
        K0, R0, S0: 基准候选规模。
        alpha, beta, gamma: 调节系数。
        K_min, K_max, R_min, R_max, S_min, S_max: 各规模的截断区间。
        window: compute_progress 的观察窗口。
    """

    K0: int = 6
    R0: int = 3
    S0: int = 6
    alpha: float = 0.5
    beta: float = 0.33
    gamma: float = 0.33
    K_min: int = 2
    K_max: int = 32
    R_min: int = 1
    R_max: int | None = None
    S_min: int = 2
    S_max: int = 8
    window: int = 5


DEFAULT_CONFIG = AdaptiveConfig()
