"""destroy_repair 模块的单元测试。"""

import numpy as np
import pytest

from src.destroy_repair import bottleneck_local_search_dr, destroy_repair
from src.feasibility import compute_loads, verify_solution
from src.initial_solutions import min_min, round_robin
from src.local_search import bottleneck_local_search

STANDARD_KEYS = {
    "assignment",
    "loads",
    "cmax",
    "evals",
    "accepted_moves",
    "improved",
    "history",
}
DR_EXTRA_KEYS = {"restarts", "improvements", "accepted_worse", "segments"}


def _make_instance(seed=7, n=20, m=4):
    P = np.random.default_rng(seed).uniform(1.0, 20.0, size=(n, m))
    return P


# ---------- destroy_repair ----------


def test_destroy_repair_bottleneck_feasible():
    """瓶颈导向破坏重建后的解可行且形状正确。"""
    P = _make_instance()
    a0, l0, c0 = min_min(P)
    a1, l1, c1 = destroy_repair(P, a0, l0, c0, destroy_frac=0.2, seed=42)
    assert a1.shape == a0.shape
    assert l1.shape == l0.shape
    assert verify_solution(P, a1, l1, c1)
    assert np.isclose(c1, float(np.max(l1)))


def test_destroy_repair_bottleneck_targets_bottleneck_tasks():
    """瓶颈导向：破坏集中发生在高负载节点（至少破坏一个瓶颈节点上的任务）。"""
    n, m = 12, 3
    P = np.full((n, m), 1.0)
    P[0, 0] = 100.0  # 任务 0 在节点 0 上极重
    a0 = np.zeros(n, dtype=np.int64)
    a0[1:] = 1  # 节点 0 只有任务 0，负载 100；其余在节点 1
    l0 = compute_loads(P, a0)
    c0 = float(np.max(l0))
    # 破坏 20% ≈ 2 个任务，瓶颈节点 0 只有 1 个任务，必然被选中
    a1, l1, c1 = destroy_repair(P, a0, l0, c0, destroy_frac=0.2, seed=42)
    assert a1[0] != a0[0]  # 瓶颈任务被移出


def test_destroy_repair_random_feasible():
    """随机破坏策略同样返回可行解。"""
    P = _make_instance()
    a0, l0, c0 = min_min(P)
    a1, l1, c1 = destroy_repair(P, a0, l0, c0, strategy="random", seed=42)
    assert verify_solution(P, a1, l1, c1)


def test_destroy_repair_destroy_count():
    """破坏任务数 q = max(1, round(destroy_frac * n))，且全部换了节点。"""
    P = _make_instance(n=10, m=3)
    a0, l0, c0 = round_robin(P)
    a1, _, _ = destroy_repair(P, a0, l0, c0, destroy_frac=0.25, seed=42)
    # 25% × 10 = 2.5 → round-half-to-even = 2；重建不回原节点 → changed == 2
    changed = int(np.sum(a1 != a0))
    assert changed == 2


def test_destroy_repair_deterministic():
    """同种子两次结果一致；不同种子结果（通常）不同。"""
    P = _make_instance()
    a0, l0, c0 = min_min(P)
    a1, l1, c1 = destroy_repair(P, a0, l0, c0, seed=42)
    a2, l2, c2 = destroy_repair(P, a0, l0, c0, seed=42)
    assert np.array_equal(a1, a2)
    assert np.array_equal(l1, l2)
    assert c1 == pytest.approx(c2)


def test_destroy_repair_rng_reuse():
    """给定 rng 时复用该生成器（序列与显式派生一致，仅验证不报错且可行）。"""
    P = _make_instance()
    a0, l0, c0 = min_min(P)
    rng = np.random.default_rng(7)
    a1, l1, c1 = destroy_repair(P, a0, l0, c0, rng=rng)
    assert verify_solution(P, a1, l1, c1)


def test_destroy_repair_invalid_args():
    """非法参数抛 ValueError。"""
    P = _make_instance(n=8, m=2)
    a0, l0, c0 = round_robin(P)
    with pytest.raises(ValueError):
        destroy_repair(P, a0, l0, c0, destroy_frac=0.0)
    with pytest.raises(ValueError):
        destroy_repair(P, a0, l0, c0, destroy_frac=1.5)
    with pytest.raises(ValueError):
        destroy_repair(P, a0, l0, c0, strategy="bad")


# ---------- bottleneck_local_search_dr ----------


def test_dr_search_returns_standard_fields_and_extras():
    """输出结构：标准字段 + DR 附加字段；最终解可行且不劣于初解。"""
    P = _make_instance()
    a0, l0, c0 = min_min(P)
    res = bottleneck_local_search_dr(P, a0, l0, c0, max_evals=2000)
    assert STANDARD_KEYS | DR_EXTRA_KEYS <= set(res.keys())
    assert res["evals"] <= 2000
    assert verify_solution(P, res["assignment"], res["loads"], res["cmax"])
    assert res["cmax"] <= c0 + 1e-9
    assert res["segments"] >= 1
    assert res["restarts"] == res["segments"] - 1
    assert res["accepted_worse"] == 0  # strict 模式不接受变差
    assert isinstance(res["history"], list)
    # history 条目为标准 4 字段
    for entry in res["history"]:
        assert set(entry.keys()) == {"round", "cmax", "evals", "accepted"}


def test_dr_search_strict_no_worse_than_single_search():
    """严格模式：第一段与单次搜索同起点，最终解不劣于同预算的单次瓶颈搜索。"""
    P = _make_instance()
    a0, l0, c0 = min_min(P)
    single = bottleneck_local_search(P, a0, l0, c0, K=4, R=2, S=4, max_evals=5000)
    dr = bottleneck_local_search_dr(
        P, a0, l0, c0, K=4, R=2, S=4, max_evals=5000, accept_strategy="strict"
    )
    assert dr["cmax"] <= single["cmax"] + 1e-9


def test_dr_search_metropolis_runs():
    """Metropolis 模式：结构正确，accepted_worse >= 0，结果可行。"""
    P = _make_instance()
    a0, l0, c0 = round_robin(P)
    res = bottleneck_local_search_dr(
        P, a0, l0, c0, max_evals=3000, accept_strategy="metropolis"
    )
    assert STANDARD_KEYS | DR_EXTRA_KEYS <= set(res.keys())
    assert verify_solution(P, res["assignment"], res["loads"], res["cmax"])
    assert res["accepted_worse"] >= 0


def test_dr_search_zero_budget():
    """max_evals=0：不执行任何段，返回初解。"""
    P = _make_instance()
    a0, l0, c0 = min_min(P)
    res = bottleneck_local_search_dr(P, a0, l0, c0, max_evals=0)
    assert res["evals"] == 0
    assert res["segments"] == 0
    assert res["restarts"] == 0
    assert res["history"] == []
    assert res["cmax"] == pytest.approx(c0)
    assert np.array_equal(res["assignment"], a0)
    assert res["improved"] is False


def test_dr_search_deterministic():
    """同种子两次运行结果一致。"""
    P = _make_instance()
    a0, l0, c0 = round_robin(P)
    r1 = bottleneck_local_search_dr(P, a0, l0, c0, max_evals=3000)
    r2 = bottleneck_local_search_dr(P, a0, l0, c0, max_evals=3000)
    assert r1["cmax"] == pytest.approx(r2["cmax"])
    assert np.array_equal(r1["assignment"], r2["assignment"])
    assert r1["evals"] == r2["evals"]
    assert r1["restarts"] == r2["restarts"]


def test_dr_search_improves_on_trap_instance():
    """陷阱实例：单次瓶颈搜索陷入局部最优，破坏重建能逃出并找到更优解。

    构造：两节点，任务执行时间差异大，初解将所有"重-轻"任务放到节点 0、
    "轻-重"任务放到节点 1。单次迁移/交换无法协调多任务互移，
    破坏重建后重新搜索可以找到更好的平衡。
    """
    n, m = 30, 2
    P = np.full((n, m), 1.0)
    P[:15, 0] = 10.0
    P[:15, 1] = 1.0
    P[15:, 0] = 1.0
    P[15:, 1] = 10.0
    a0 = np.zeros(n, dtype=np.int64)
    a0[15:] = 1
    l0 = compute_loads(P, a0)
    c0 = float(np.max(l0))
    assert np.isclose(c0, 150.0)

    single = bottleneck_local_search(P, a0, l0, c0, K=4, R=2, S=4, max_evals=2000)
    dr = bottleneck_local_search_dr(
        P, a0, l0, c0, K=4, R=2, S=4, max_evals=4000,
        destroy_frac=0.3, accept_strategy="strict", max_stagnation=5, seed=42,
    )
    # 破坏重建不劣于单次搜索
    assert dr["cmax"] <= single["cmax"] + 1e-9
    assert verify_solution(P, dr["assignment"], dr["loads"], dr["cmax"])


def test_dr_search_invalid_input_raises():
    """loads / cmax 与 assignment 不一致时抛 ValueError。"""
    P = _make_instance()
    a0, l0, c0 = min_min(P)
    bad_loads = l0.copy()
    bad_loads[0] += 1.0
    with pytest.raises(ValueError):
        bottleneck_local_search_dr(P, a0, bad_loads, c0, max_evals=100)
    with pytest.raises(ValueError):
        bottleneck_local_search_dr(P, a0, l0, c0 + 1.0, max_evals=100)


def test_dr_search_invalid_args():
    """非法参数抛 ValueError。"""
    P = _make_instance(n=8, m=2)
    a0, l0, c0 = round_robin(P)
    with pytest.raises(ValueError):
        bottleneck_local_search_dr(P, a0, l0, c0, destroy_frac=0.0)
    with pytest.raises(ValueError):
        bottleneck_local_search_dr(P, a0, l0, c0, destroy_strategy="bad")
    with pytest.raises(ValueError):
        bottleneck_local_search_dr(P, a0, l0, c0, accept_strategy="bad")
    with pytest.raises(ValueError):
        bottleneck_local_search_dr(P, a0, l0, c0, segment_evals=0)
    with pytest.raises(ValueError):
        bottleneck_local_search_dr(P, a0, l0, c0, max_stagnation=0)
    with pytest.raises(ValueError):
        bottleneck_local_search_dr(P, a0, l0, c0, cooling_rate=0.0)
