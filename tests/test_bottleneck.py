"""bottleneck 模块的单元测试。"""

import numpy as np
import pytest

from src.bottleneck import (
    bottleneck_sharpness,
    find_bottleneck,
    select_candidate_tasks,
)


class TestFindBottleneck:
    def test_simple(self):
        loads = np.array([1.0, 5.0, 3.0])
        assert find_bottleneck(loads) == 1

    def test_tie_returns_smallest_index(self):
        loads = np.array([2.0, 5.0, 5.0, 1.0])
        assert find_bottleneck(loads) == 1
        loads_all_equal = np.array([4.0, 4.0, 4.0])
        assert find_bottleneck(loads_all_equal) == 0

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            find_bottleneck(np.array([]))


class TestBottleneckSharpness:
    def test_normal(self):
        loads = np.array([10.0, 6.0, 4.0])
        rho = bottleneck_sharpness(loads)
        assert np.isclose(rho, (10.0 - 6.0) / 10.0)

    def test_single_machine(self):
        assert bottleneck_sharpness(np.array([5.0])) == 0.0

    def test_all_zero(self):
        assert bottleneck_sharpness(np.zeros(3)) == 0.0

    def test_tie_max_gives_zero(self):
        # 两个节点并列最大 → L_second == L_max → rho = 0
        loads = np.array([7.0, 7.0, 2.0])
        assert np.isclose(bottleneck_sharpness(loads), 0.0)

    def test_range(self):
        rng = np.random.default_rng(0)
        loads = rng.uniform(1.0, 100.0, size=5)
        rho = bottleneck_sharpness(loads)
        assert 0.0 <= rho <= 1.0


class TestSelectCandidateTasks:
    def make_fixture(self):
        P = np.array(
            [
                [5.0, 1.0, 1.0],
                [3.0, 9.0, 1.0],
                [4.0, 2.0, 8.0],
                [1.0, 7.0, 6.0],
                [2.0, 5.0, 3.0],
            ]
        )
        assignment = np.array([1, 0, 1, 2, 1])
        return P, assignment

    def test_descending_and_truncate(self):
        P, assignment = self.make_fixture()
        # 节点 1 上任务 0, 2, 4；P[·,1] = 1, 2, 5 → 降序 [4, 2, 0]
        tasks = select_candidate_tasks(P, assignment, node=1, K=2)
        assert np.array_equal(tasks, np.array([4, 2]))
        assert tasks.dtype == np.int64

    def test_k_larger_than_task_count_returns_all(self):
        P, assignment = self.make_fixture()
        all_tasks = select_candidate_tasks(P, assignment, node=1, K=10)
        assert np.array_equal(all_tasks, np.array([4, 2, 0]))

    def test_empty_node(self):
        P = np.array([[1.0, 2.0], [3.0, 4.0]])
        assignment = np.array([0, 0])
        tasks = select_candidate_tasks(P, assignment, node=1, K=3)
        assert tasks.size == 0

    def test_k_nonpositive_returns_empty(self):
        P, assignment = self.make_fixture()
        assert select_candidate_tasks(P, assignment, node=1, K=0).size == 0
        assert select_candidate_tasks(P, assignment, node=1, K=-1).size == 0

    def test_stable_order_on_ties(self):
        # 节点 0 上三个任务执行时间并列，应保持任务编号升序
        P = np.array([[3.0, 9.0], [3.0, 8.0], [3.0, 7.0]])
        assignment = np.array([0, 0, 0])
        tasks = select_candidate_tasks(P, assignment, node=0, K=3)
        assert np.array_equal(tasks, np.array([0, 1, 2]))

    def test_invalid_node_raises(self):
        P, assignment = self.make_fixture()
        with pytest.raises(ValueError):
            select_candidate_tasks(P, assignment, node=3, K=2)
        with pytest.raises(ValueError):
            select_candidate_tasks(P, assignment, node=-1, K=2)
