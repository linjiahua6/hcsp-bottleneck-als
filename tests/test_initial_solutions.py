"""initial_solutions 模块的单元测试。"""

import numpy as np
import pytest

from src.feasibility import compute_cmax, compute_loads, verify_solution
from src.initial_solutions import lpt, min_min, random_greedy, round_robin

SOLVERS = [round_robin, min_min, lpt, random_greedy]


def make_matrix() -> np.ndarray:
    """固定的小规模执行时间矩阵：10 任务 × 3 节点。"""
    return np.array(
        [
            [2.0, 3.0, 4.0],
            [4.0, 1.0, 5.0],
            [1.0, 6.0, 2.0],
            [3.0, 3.0, 3.0],
            [5.0, 2.0, 1.0],
            [2.0, 2.0, 4.0],
            [1.0, 1.0, 1.0],
            [6.0, 4.0, 2.0],
            [3.0, 5.0, 1.0],
            [2.0, 1.0, 3.0],
        ],
        dtype=np.float64,
    )


def solve(solver, P: np.ndarray):
    """以统一接口调用各初始解函数（random_greedy 需要 seed）。"""
    if solver is random_greedy:
        return solver(P, seed=42)
    return solver(P)


@pytest.mark.parametrize("solver", SOLVERS, ids=[f.__name__ for f in SOLVERS])
def test_shape_and_range(solver):
    P = make_matrix()
    assignment, loads, cmax = solve(solver, P)
    n, m = P.shape
    assert assignment.shape == (n,)
    assert np.issubdtype(assignment.dtype, np.integer)
    assert np.all((assignment >= 0) & (assignment < m))
    assert loads.shape == (m,)
    assert isinstance(cmax, float)


@pytest.mark.parametrize("solver", SOLVERS, ids=[f.__name__ for f in SOLVERS])
def test_loads_and_cmax_consistent(solver):
    P = make_matrix()
    assignment, loads, cmax = solve(solver, P)
    expected_loads = compute_loads(P, assignment)
    assert np.allclose(loads, expected_loads)
    assert np.isclose(cmax, compute_cmax(P, assignment))
    assert verify_solution(P, assignment, loads, cmax)


def test_round_robin_pattern():
    P = make_matrix()  # n=10, m=3
    assignment, loads, cmax = round_robin(P)
    expected_assignment = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    assert np.array_equal(assignment, expected_assignment)
    assert np.isclose(cmax, float(np.max(loads)))


def test_min_min_beats_round_robin():
    """节点 0 对所有任务都快：Min-Min 应把所有任务压在节点 0，C_max 更小。"""
    n, m = 9, 3
    P = np.full((n, m), 100.0)
    P[:, 0] = 1.0
    _, _, cmax_rr = round_robin(P)
    _, _, cmax_mm = min_min(P)
    assert cmax_mm < cmax_rr


def test_lpt_assigns_to_least_loaded():
    P = np.array(
        [
            [10.0, 11.0],  # 均值 10.5，最大，最先分配
            [1.0, 1.0],  # 均值 1.0，最小，最后分配
            [5.0, 5.0],  # 均值 5.0，居中
        ]
    )
    assignment, loads, cmax = lpt(P)
    assert assignment[0] == 0  # 最先 → 节点 0
    assert assignment[2] == 1  # 其次 → 节点 1（负载 5 < 10）
    assert assignment[1] == 1  # 最后 → 节点 1（负载 6 < 10）
    assert np.isclose(cmax, 10.0)
    assert np.allclose(loads, [10.0, 6.0])


def test_random_greedy_reproducible():
    P = make_matrix()
    a1, l1, c1 = random_greedy(P, seed=42)
    a2, l2, c2 = random_greedy(P, seed=42)
    assert np.array_equal(a1, a2)
    assert np.allclose(l1, l2)
    assert np.isclose(c1, c2)


def test_random_greedy_different_seeds_differ():
    P = np.random.default_rng(0).uniform(1.0, 100.0, size=(20, 4))
    a1, _, _ = random_greedy(P, seed=1)
    a2, _, _ = random_greedy(P, seed=2)
    assert not np.array_equal(a1, a2)


@pytest.mark.parametrize("solver", SOLVERS, ids=[f.__name__ for f in SOLVERS])
def test_invalid_matrix_raises(solver):
    P = np.array([[0.0, 1.0], [1.0, 2.0]])  # 存在非正值
    with pytest.raises(ValueError):
        solve(solver, P)
