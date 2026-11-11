"""实验评估辅助工具：改进率计算与函数计时。"""

from __future__ import annotations

import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def improvement_rate(c0: float, c_final: float) -> float:
    """计算改进率（百分比）。

    improvement = (c0 - c_final) / c0 * 100

    Args:
        c0: 初始目标值（如初始解的 C_max）。
        c_final: 最终目标值（如邻域搜索后的 C_max）。

    Returns:
        改进率（百分比）；若 c0 <= 0 返回 0.0。
    """
    if c0 <= 0:
        return 0.0
    return (c0 - c_final) / c0 * 100.0


def run_with_timer(func: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T, float]:
    """执行 func 并计时。

    Args:
        func: 待执行的函数。
        *args, **kwargs: 传给 func 的位置/关键字参数。

    Returns:
        (func 的返回值, 运行时间（秒）)，时间基于 time.perf_counter。
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed
