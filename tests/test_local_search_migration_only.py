"""BLS-M（仅迁移消融变体）行为测试。

验证两个核心性质：
1. swap 确已关闭：在"仅交换能改善"的实例上，BLS 改善而 BLS-M 不改善；
2. 迁移分支与 BLS 一致：在"交换无帮助"的实例上，BLS-M 与 BLS 执行
   相同的迁移序列，最终解与接受动作数一致（评估数因 BLS 额外评估
   交换而更多）。
"""

import numpy as np

from src.local_search import bottleneck_local_search
from src.local_search_migration_only import bottleneck_migration_search


def test_swap_only_improvement_is_disabled_in_blsm():
    """仅交换能改善的实例：BLS 改善，BLS-M 不改善。

    P = [[10, 0.1], [5, 9.9]]，a=[0,1]，loads=[10, 9.9]：
    - 迁移 i->1：L1' = 9.9 + 0.1 = 10.0，不严格改善；
    - 交换 i<->h：makespan = max(5, 0.1) = 5 < 10，改善。
    因此 BLS（含 swap）改善，BLS-M（仅迁移）应停在原解。
    """
    P = np.array([[10.0, 0.1], [5.0, 9.9]])
    a = np.array([0, 1], dtype=np.int64)
    loads = np.array([10.0, 9.9])
    c0 = 10.0

    blsm = bottleneck_migration_search(P, a, loads, c0, K=4, R=2,
                                       max_evals=10000)
    bls = bottleneck_local_search(P, a, loads, c0, K=4, R=2, S=4,
                                  max_evals=10000)

    assert not blsm["improved"], "BLS-M 不应通过交换改善"
    assert np.isclose(blsm["cmax"], 10.0)
    assert bls["improved"], "BLS 应通过交换改善"
    assert np.isclose(bls["cmax"], 5.0)


def test_migration_branch_matches_bls_when_swap_never_helps():
    """交换无帮助的实例：迁移序列、最终解与接受数一致。

    P = [[10, 0.5], [1000, 1]]，a=[0,1]，loads=[10, 1]：
    - 迁移 i->1：makespan = max(0, 1.5) = 1.5，改善；
    - 任何交换把 h（p_h0=1000）带回机器 0，均不改善。
    两搜索应执行同一迁移并停在同一局部最优（1.5）；
    BLS 每轮额外评估交换，评估总数更大。
    """
    P = np.array([[10.0, 0.5], [1000.0, 1.0]])
    a = np.array([0, 1], dtype=np.int64)
    loads = np.array([10.0, 1.0])
    c0 = 10.0

    blsm = bottleneck_migration_search(P, a, loads, c0, K=4, R=2,
                                       max_evals=10000)
    bls = bottleneck_local_search(P, a, loads, c0, K=4, R=2, S=4,
                                  max_evals=10000)

    assert np.isclose(blsm["cmax"], 1.5), blsm
    assert np.isclose(bls["cmax"], blsm["cmax"]), \
        "迁移分支不一致：BLS 与 BLS-M 最终解应相同"
    assert bls["accepted_moves"] == blsm["accepted_moves"] == 1
    assert bls["evals"] > blsm["evals"], \
        "BLS 应比 BLS-M 多评估交换候选"
    assert blsm["evals"] == 3, blsm  # 1 次改善迁移 + 2 次失败迁移


def test_budget_stop_matches_bls_on_first_evaluation():
    """预算在第一次评估后耗尽：两搜索行为一致。

    max_evals=1 时，两搜索都在第一轮评估 1 次迁移（i->1 改善 1.5）后
    预算耗尽，轮末执行已找到的改善动作；BLS 在该边界上同样不会继续
    评估交换。断言两搜索最终解、评估数与改善状态完全一致。
    """
    P = np.array([[10.0, 0.5], [1000.0, 1.0]])
    a = np.array([0, 1], dtype=np.int64)
    loads = np.array([10.0, 1.0])
    c0 = 10.0

    blsm = bottleneck_migration_search(P, a, loads, c0, K=4, R=2,
                                       max_evals=1)
    bls = bottleneck_local_search(P, a, loads, c0, K=4, R=2, S=4,
                                  max_evals=1)

    assert blsm["evals"] == 1
    assert bls["evals"] == 1
    assert blsm["improved"] == bls["improved"]
    assert np.isclose(blsm["cmax"], bls["cmax"])
    assert np.isclose(blsm["cmax"], 1.5)
    assert np.isclose(bls["cmax"], 1.5)
