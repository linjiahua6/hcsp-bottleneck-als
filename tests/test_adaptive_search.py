"""adaptive_search 模块的单元测试。"""

import numpy as np
import pytest

from src.adaptive import AdaptiveConfig
from src.adaptive_search import adaptive_bottleneck_search
from src.feasibility import compute_cmax, compute_loads, verify_solution
from src.initial_solutions import min_min, round_robin
from src.local_search import bottleneck_local_search, plain_local_search

STANDARD_KEYS = {
    "assignment",
    "loads",
    "cmax",
    "evals",
    "accepted_moves",
    "improved",
    "history",
}
ADAPTIVE_EXTRA_KEYS = {"config", "final_sizes"}
ADAPTIVE_HISTORY_KEYS = {
    "round", "cmax", "evals", "accepted", "rho", "eta", "K", "R", "S",
}


def test_adaptive_search_basic():
    """20×4 小实例：输出结构、不劣化、history 字段齐全。"""
    P = np.random.default_rng(7).uniform(1.0, 20.0, size=(20, 4))
    a0, l0, c0 = min_min(P)
    res = adaptive_bottleneck_search(P, a0, l0, c0, max_evals=800)

    assert STANDARD_KEYS | ADAPTIVE_EXTRA_KEYS <= set(res.keys())
    assert res["evals"] <= 800
    assert verify_solution(P, res["assignment"], res["loads"], res["cmax"])
    assert res["cmax"] <= c0 + 1e-9  # 不劣化
    assert res["improved"] == (res["cmax"] < c0 - 1e-9)
    assert isinstance(res["config"], dict) and res["config"]["K0"] == 6
    if res["history"]:
        assert isinstance(res["final_sizes"], tuple) and len(res["final_sizes"]) == 3
    else:
        assert res["final_sizes"] is None
    # history 中 cmax 单调不增
    cmaxes = [entry["cmax"] for entry in res["history"]]
    for k in range(1, len(cmaxes)):
        assert cmaxes[k] <= cmaxes[k - 1] + 1e-9


def test_adaptive_search_history_has_new_fields():
    """Min-Min 次优实例（seed=1，已知可改进）：每轮都有 rho、eta、K、R、S 且取值合理。"""
    P = np.random.default_rng(1).uniform(1.0, 20.0, size=(8, 2))
    a0, l0, c0 = min_min(P)
    res = adaptive_bottleneck_search(P, a0, l0, c0, max_evals=2000)
    assert res["history"]  # 实例可改进，history 非空

    cfg = AdaptiveConfig()
    for entry in res["history"]:
        assert set(entry.keys()) == ADAPTIVE_HISTORY_KEYS
        assert 0.0 <= entry["rho"] <= 1.0
        assert 0.0 <= entry["eta"] <= 1.0
        assert cfg.K_min <= entry["K"] <= cfg.K_max
        assert cfg.R_min <= entry["R"] <= P.shape[1]
        assert cfg.S_min <= entry["S"] <= cfg.S_max


def test_adaptive_search_zero_budget():
    """max_evals=0：直接返回初解，history 为空，final_sizes 为 None。"""
    P = np.random.default_rng(7).uniform(1.0, 20.0, size=(20, 4))
    a0, l0, c0 = min_min(P)
    res = adaptive_bottleneck_search(P, a0, l0, c0, max_evals=0)
    assert res["evals"] == 0
    assert res["accepted_moves"] == 0
    assert res["history"] == []
    assert res["cmax"] == pytest.approx(c0)
    assert np.array_equal(res["assignment"], a0)
    assert res["improved"] is False
    assert res["final_sizes"] is None


def test_adaptive_search_not_worse_than_fixed_on_smooth_instance():
    """瓶颈平坦实例（执行时间近均匀 + 轮询初解）：自适应不劣于固定规模。"""
    P = np.random.default_rng(11).uniform(9.0, 11.0, size=(24, 3))
    a0, l0, c0 = round_robin(P)
    res_fixed = bottleneck_local_search(P, a0, l0, c0, max_evals=3000)
    res_adapt = adaptive_bottleneck_search(P, a0, l0, c0, max_evals=3000)
    assert res_adapt["cmax"] <= res_fixed["cmax"] + 1e-9
    assert verify_solution(
        P, res_adapt["assignment"], res_adapt["loads"], res_adapt["cmax"]
    )


def test_adaptive_search_not_worse_than_plain_on_bottleneck_instance():
    """复用第二步的 6×2 错配实例：只有交换能改善，自适应应与 fixed 一样找到。"""
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
    res_adapt = adaptive_bottleneck_search(P, a0, l0, c0, max_evals=200)

    assert res_plain["cmax"] == pytest.approx(c0)  # 迁移无法改善
    assert res_adapt["cmax"] < c0 - 1e-9  # 交换可改善
    assert res_adapt["cmax"] <= res_plain["cmax"] + 1e-9
    assert not np.array_equal(res_adapt["assignment"], a0)


def test_adaptive_search_deterministic():
    """同种子两次运行结果一致。"""
    P = np.random.default_rng(7).uniform(1.0, 20.0, size=(20, 4))
    a0, l0, c0 = min_min(P)
    r1 = adaptive_bottleneck_search(P, a0, l0, c0, max_evals=500)
    r2 = adaptive_bottleneck_search(P, a0, l0, c0, max_evals=500)
    assert r1["cmax"] == pytest.approx(r2["cmax"])
    assert np.array_equal(r1["assignment"], r2["assignment"])
    assert r1["evals"] == r2["evals"]
    assert r1["accepted_moves"] == r2["accepted_moves"]
    assert r1["final_sizes"] == r2["final_sizes"]


def test_adaptive_search_invalid_input_raises():
    """loads / cmax 与 assignment 不一致时抛 ValueError。"""
    P = np.random.default_rng(7).uniform(1.0, 20.0, size=(20, 4))
    a0, l0, c0 = min_min(P)
    bad_loads = l0.copy()
    bad_loads[0] += 1.0
    with pytest.raises(ValueError):
        adaptive_bottleneck_search(P, a0, bad_loads, c0, max_evals=100)
    with pytest.raises(ValueError):
        adaptive_bottleneck_search(P, a0, l0, c0 + 1.0, max_evals=100)
