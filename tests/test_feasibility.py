"""feasibility 模块的单元测试。"""

import numpy as np
import pytest

from src.feasibility import (
    check_feasibility,
    compute_cmax,
    compute_loads,
    verify_solution,
)


@pytest.fixture
def P() -> np.ndarray:
    """4 任务 × 3 节点的执行时间矩阵。"""
    return np.array(
        [
            [2.0, 3.0, 4.0],
            [4.0, 1.0, 5.0],
            [1.0, 6.0, 2.0],
            [3.0, 3.0, 3.0],
        ]
    )


@pytest.fixture
def assignment() -> np.ndarray:
    """任务 0→节点 0，任务 1→节点 2，任务 2→节点 1，任务 3→节点 0。"""
    return np.array([0, 2, 1, 0])


class TestCheckFeasibility:
    def test_valid(self, P, assignment):
        assert check_feasibility(P, assignment)

    def test_wrong_length(self, P):
        assert not check_feasibility(P, np.array([0, 1]))
        assert not check_feasibility(P, np.array([0, 1, 2, 0, 1]))

    def test_out_of_range(self, P):
        assert not check_feasibility(P, np.array([0, 3, 1, 0]))
        assert not check_feasibility(P, np.array([0, -1, 1, 0]))

    def test_non_integer_dtype(self, P):
        assert not check_feasibility(P, np.array([0.0, 1.0, 2.0, 0.0]))

    def test_two_dimensional_assignment(self, P):
        assert not check_feasibility(P, np.array([[0, 1, 2, 0]]))

    def test_bad_matrix(self):
        assert not check_feasibility(
            np.array([[1.0, 0.0], [2.0, 3.0]]), np.array([0, 1])
        )
        assert not check_feasibility(
            np.array([[np.nan, 1.0], [2.0, 3.0]]), np.array([0, 1])
        )
        assert not check_feasibility(np.ones((2, 2, 2)), np.array([0, 1]))
        assert not check_feasibility(np.zeros((0, 2)), np.array([], dtype=int))


class TestComputeLoads:
    def test_loads(self, P, assignment):
        loads = compute_loads(P, assignment)
        expected = np.array([5.0, 6.0, 5.0])  # 节点0: 2+3, 节点1: 6, 节点2: 5
        assert np.allclose(loads, expected)
        assert loads.dtype == np.float64

    def test_invalid_assignment_raises(self, P):
        with pytest.raises(ValueError):
            compute_loads(P, np.array([0, 1, 2]))  # 长度不符
        with pytest.raises(ValueError):
            compute_loads(P, np.array([0, 5, 2, 0]))  # 越界
        with pytest.raises(ValueError):
            compute_loads(P, np.array([0.0, 1.0, 2.0, 0.0]))  # 非整数


class TestComputeCmax:
    def test_cmax(self, P, assignment):
        assert np.isclose(compute_cmax(P, assignment), 6.0)


class TestVerifySolution:
    def test_consistent(self, P, assignment):
        loads = compute_loads(P, assignment)
        cmax = compute_cmax(P, assignment)
        assert verify_solution(P, assignment, loads, cmax)

    def test_wrong_loads(self, P, assignment):
        loads = compute_loads(P, assignment)
        cmax = compute_cmax(P, assignment)
        bad_loads = loads.copy()
        bad_loads[0] += 0.5
        assert not verify_solution(P, assignment, bad_loads, cmax)

    def test_wrong_loads_shape(self, P, assignment):
        cmax = compute_cmax(P, assignment)
        assert not verify_solution(P, assignment, np.zeros(2), cmax)

    def test_wrong_cmax(self, P, assignment):
        loads = compute_loads(P, assignment)
        assert not verify_solution(P, assignment, loads, 5.9)

    def test_infeasible_assignment(self, P):
        assert not verify_solution(P, np.array([0, 99, 1, 0]), np.zeros(3), 0.0)

    def test_tolerance(self, P, assignment):
        loads = compute_loads(P, assignment)
        cmax = compute_cmax(P, assignment)
        assert verify_solution(P, assignment, loads, cmax + 1e-12, tol=1e-9)
        assert not verify_solution(P, assignment, loads, cmax + 1e-3, tol=1e-9)
