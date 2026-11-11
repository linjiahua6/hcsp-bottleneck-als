"""local_search 模块的单元测试。"""

import numpy as np
import pytest

from src.feasibility import compute_cmax, compute_loads, verify_solution
from src.initial_solutions import min_min
from src.local_search import bottleneck_local_search, plain_local_search

REQUIRED_KEYS = {
    "assignment",
    "loads",
    "cmax",
    "evals",
    "accepted_moves",
    "improved",
    "history",
}
HISTORY_KEYS = {"round", "cmax", "evals", "accepted"}

SEARCHERS = [plain_local_search, bottleneck_local_search]
SEARCHER_IDS = ["plain", "bottleneck"]


def make_random_instance() -> np.ndarray:
    """20 任务 × 4 节点的随机实例（固定种子，确定性）。"""
    return np.random.default_rng(7).uniform(1.0, 20.0, size=(20, 4))


def check_result_structure(res: dict, P: np.ndarray, c0: float) -> None:
    """检查结果字典的结构与一致性。"""
    assert REQUIRED_KEYS <= set(res.keys())
    a, loads, cmax = res["assignment"], res["loads"], res["cmax"]
    assert verify_solution(P, a, loads, cmax)
    assert cmax <= c0 + 1e-9  # 不劣化
    assert isinstance(res["evals"], int) and res["evals"] >= 0
    assert isinstance(res["accepted_moves"], int) and res["accepted_moves"] >= 0
    assert isinstance(res["improved"], bool)
    assert res["improved"] == (cmax < c0 - 1e-9)
    assert isinstance(res["history"], list)
    for entry in res["history"]:
        assert set(entry.keys()) == HISTORY_KEYS
        assert isinstance(entry["round"], int) and entry["round"] >= 1
        assert isinstance(entry["accepted"], int) and entry["accepted"] >= 0


@pytest.mark.parametrize("searcher", SEARCHERS, ids=SEARCHER_IDS)
def test_basic_properties(searcher):
    P = make_random_instance()
    a0, l0, c0 = min_min(P)
    res = searcher(P, a0, l0, c0, max_evals=800)
    assert res["evals"] <= 800
    check_result_structure(res, P, c0)
    # history 中 cmax 单调不增
    cmaxes = [entry["cmax"] for entry in res["history"]]
    for k in range(1, len(cmaxes)):
        assert cmaxes[k] <= cmaxes[k - 1] + 1e-9
    # 无接受时 cmax 保持初值
    if res["accepted_moves"] == 0:
        assert res["cmax"] == pytest.approx(c0)


@pytest.mark.parametrize("searcher", SEARCHERS, ids=SEARCHER_IDS)
def test_zero_budget_returns_initial(searcher):
    P = make_random_instance()
    a0, l0, c0 = min_min(P)
    res = searcher(P, a0, l0, c0, max_evals=0)
    assert res["evals"] == 0
    assert res["accepted_moves"] == 0
    assert res["cmax"] == pytest.approx(c0)
    assert np.array_equal(res["assignment"], a0)
    assert res["history"] == []
    assert res["improved"] is False


def test_both_improve_minmin_suboptimal():
    """Min-Min 次优实例（经暴力枚举实测：c0=35.414，最优=30.5663）。

    实例由 seed=1 的 8×2 均匀矩阵生成；此断言基于真实计算而非估计。
    """
    P = np.random.default_rng(1).uniform(1.0, 20.0, size=(8, 2))
    a0, l0, c0 = min_min(P)
    assert c0 > 30.5663 + 1e-6  # 确认初解确实次优（最优值经暴力枚举）
    for searcher in SEARCHERS:
        res = searcher(P, a0, l0, c0, max_evals=2000)
        assert res["improved"]
        assert res["cmax"] < c0 - 1e-9
        check_result_structure(res, P, c0)


def test_bottleneck_not_worse_than_plain_on_bottleneck_instance():
    """"任务-节点错配"实例：只有交换能改善，迁移无法改善。

    任务 0-2 在节点 0 快（1）、节点 1 慢（10）；任务 3-5 相反。
    初解把任务 0-2 放节点 0、任务 3-5 放节点 1，loads=[30, 30]。
    - 任何单任务迁移都会使目标节点负载超过 30 → plain 无法改善；
    - 交换（0↔3 等）可使负载降至 21 → bottleneck 可逐轮改善到 3。
    """
    n, m = 6, 2
    P = np.full((n, m), 1.0)
    P[:3, 0] = 10.0
    P[:3, 1] = 1.0
    P[3:, 0] = 1.0
    P[3:, 1] = 10.0
    a0 = np.array([0, 0, 0, 1, 1, 1])
    l0 = compute_loads(P, a0)
    c0 = compute_cmax(P, a0)
    assert np.isclose(c0, 30.0)

    res_plain = plain_local_search(P, a0, l0, c0, max_evals=200)
    res_bottle = bottleneck_local_search(
        P, a0, l0, c0, K=6, R=1, S=6, max_evals=200
    )

    assert res_plain["cmax"] == pytest.approx(c0)  # 迁移无法改善
    assert res_bottle["cmax"] < c0 - 1e-9  # 交换可改善
    assert res_bottle["cmax"] <= res_plain["cmax"] + 1e-9  # 不差于 plain
    check_result_structure(res_bottle, P, c0)
    # 交换动作确实生效：最终分配应与初解不同
    assert not np.array_equal(res_bottle["assignment"], a0)


@pytest.mark.parametrize("searcher", SEARCHERS, ids=SEARCHER_IDS)
def test_tol_equal_cmax_not_accepted(searcher):
    """C'_max 与 C_max 相等（不严格小于）时不接受。"""
    P = np.array([[1.0, 2.0], [1.0, 2.0]])
    a0 = np.array([0, 1])
    l0 = compute_loads(P, a0)  # [1, 2]
    c0 = compute_cmax(P, a0)  # 2.0
    res = searcher(P, a0, l0, c0, max_evals=100)
    assert res["cmax"] == pytest.approx(c0)
    assert res["accepted_moves"] == 0
    assert res["improved"] is False


@pytest.mark.parametrize("searcher", SEARCHERS, ids=SEARCHER_IDS)
def test_deterministic(searcher):
    P = make_random_instance()
    a0, l0, c0 = min_min(P)
    r1 = searcher(P, a0, l0, c0, max_evals=500)
    r2 = searcher(P, a0, l0, c0, max_evals=500)
    assert r1["cmax"] == pytest.approx(r2["cmax"])
    assert np.array_equal(r1["assignment"], r2["assignment"])
    assert r1["evals"] == r2["evals"]
    assert r1["accepted_moves"] == r2["accepted_moves"]


@pytest.mark.parametrize("searcher", SEARCHERS, ids=SEARCHER_IDS)
def test_invalid_input_raises(searcher):
    P = make_random_instance()
    a0, l0, c0 = min_min(P)
    bad_loads = l0.copy()
    bad_loads[0] += 1.0  # 与 assignment 不一致
    with pytest.raises(ValueError):
        searcher(P, a0, bad_loads, c0, max_evals=100)
    with pytest.raises(ValueError):
        searcher(P, a0, l0, c0 + 1.0, max_evals=100)  # cmax 与 loads 不一致


def test_bottleneck_invalid_params_raise():
    P = make_random_instance()
    a0, l0, c0 = min_min(P)
    with pytest.raises(ValueError):
        bottleneck_local_search(P, a0, l0, c0, K=0, max_evals=100)
    with pytest.raises(ValueError):
        bottleneck_local_search(P, a0, l0, c0, R=0, max_evals=100)
    with pytest.raises(ValueError):
        bottleneck_local_search(P, a0, l0, c0, S=0, max_evals=100)
