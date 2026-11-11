"""HEFT 初始解（独立任务退化版）的单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from src.feasibility import verify_solution
from src.heft import heft
from src.initial_solutions import lpt, round_robin
from src.loader import generate_synthetic_matrix


def test_heft_shape_and_range() -> None:
    """输出形状与取值范围正确。"""
    P = generate_synthetic_matrix(n=30, m=5, seed=7)
    assignment, loads, cmax = heft(P)
    assert assignment.shape == (30,)
    assert assignment.dtype == np.int64
    assert np.all(assignment >= 0) and np.all(assignment < 5)
    assert loads.shape == (5,)
    assert cmax > 0


def test_heft_loads_consistent() -> None:
    """loads 与 assignment 一致，且通过可行性核验。"""
    P = generate_synthetic_matrix(n=64, m=8, seed=11)
    assignment, loads, cmax = heft(P)
    assert verify_solution(P, assignment, loads, cmax, tol=1e-9)
    expected = np.array([P[assignment == j, j].sum() for j in range(8)])
    assert np.allclose(loads, expected)
    assert cmax == pytest.approx(float(loads.max()), abs=1e-12)


def test_heft_deterministic() -> None:
    """同输入两次调用结果完全一致。"""
    P = generate_synthetic_matrix(n=100, m=8, seed=42)
    a1, l1, c1 = heft(P)
    a2, l2, c2 = heft(P)
    assert np.array_equal(a1, a2)
    assert np.allclose(l1, l2)
    assert c1 == c2


def test_heft_beats_round_robin_on_skewed_instance() -> None:
    """节点速度差异大的倾斜实例上，HEFT 严格优于轮询。

    节点 j 的执行时间为 base * (j + 1)：节点 0 最快、节点 3 最慢。
    轮询把任务均匀分摊到慢节点导致大瓶颈，HEFT 优先堆快节点。
    """
    n, m = 20, 4
    base = np.arange(1, n + 1, dtype=np.float64)  # 任务规模 1..20
    P = np.column_stack([base * (j + 1) for j in range(m)])
    _, _, c_rr = round_robin(P)
    _, _, c_heft = heft(P)
    assert c_heft < c_rr


def test_heft_not_worse_than_lpt_on_random() -> None:
    """随机实例上 HEFT 不差于 LPT（独立任务 EFT 贪心，允许 1e-9 浮点容差）。"""
    for seed in (42, 2026):
        P = generate_synthetic_matrix(n=100, m=8, seed=seed)
        _, _, c_lpt = lpt(P)
        _, _, c_heft = heft(P)
        assert c_heft <= c_lpt + 1e-9, (
            f"seed={seed}: heft={c_heft} > lpt={c_lpt} + 1e-9"
        )


def test_heft_invalid_matrix_raises() -> None:
    """非法矩阵（负值/零值/非有限/非二维/空）抛 ValueError。"""
    with pytest.raises(ValueError):
        heft(np.array([[-1.0, 2.0], [3.0, 4.0]]))  # 负值
    with pytest.raises(ValueError):
        heft(np.array([[0.0, 2.0], [3.0, 4.0]]))  # 零值（算法模块要求 > 0）
    with pytest.raises(ValueError):
        heft(np.array([[np.nan, 2.0], [3.0, 4.0]]))  # NaN
    with pytest.raises(ValueError):
        heft(np.array([[np.inf, 2.0], [3.0, 4.0]]))  # 无穷
    with pytest.raises(ValueError):
        heft(np.array([1.0, 2.0, 3.0]))  # 一维
    with pytest.raises(ValueError):
        heft(np.empty((0, 4)))  # 零任务
    with pytest.raises(ValueError):
        heft(np.empty((4, 0)))  # 零节点
