"""adaptive 模块（compute_progress / adaptive_sizes / AdaptiveConfig）的单元测试。"""

from dataclasses import asdict

import pytest

from src.adaptive import DEFAULT_CONFIG, AdaptiveConfig, adaptive_sizes, compute_progress


def _hist(accepted_flags) -> list[dict]:
    """构造 history：每轮一条 {"round", "cmax", "evals", "accepted"}。"""
    return [
        {
            "round": i + 1,
            "cmax": 100.0 - i,
            "evals": 10 * (i + 1),
            "accepted": flag,
        }
        for i, flag in enumerate(accepted_flags)
    ]


def test_compute_progress_empty_history():
    assert compute_progress([], window=5) == 0.0


def test_compute_progress_all_accepted():
    h = _hist([1, 1, 1, 1, 1])
    assert compute_progress(h, window=5) == 1.0


def test_compute_progress_none_accepted():
    h = _hist([0, 0, 0, 0, 0])
    assert compute_progress(h, window=5) == 0.0


def test_compute_progress_window_truncation():
    # history 长度 8 > window 5：只取最近 5 轮 [0, 1, 0, 1, 0] → 2/5
    h = _hist([1, 1, 1, 0, 1, 0, 1, 0])
    assert compute_progress(h, window=5) == pytest.approx(2 / 5)
    # 不足 window 时取全部：[1, 0, 1] → 2/3
    h2 = _hist([1, 0, 1])
    assert compute_progress(h2, window=5) == pytest.approx(2 / 3)


def test_compute_progress_window_invalid():
    with pytest.raises(ValueError):
        compute_progress(_hist([1]), window=0)


def test_adaptive_sizes_sharp_bottleneck_small_K():
    # rho=1.0 → K = K0 = 8（公式下端点，集中火力）；比 rho=0 时的 K 小
    K_sharp, _, _ = adaptive_sizes(rho=1.0, eta=0.0)
    assert K_sharp == 8
    K_flat, _, _ = adaptive_sizes(rho=0.0, eta=0.0)
    assert K_sharp < K_flat


def test_adaptive_sizes_flat_bottleneck_large_K():
    # rho=0.0 → K = clip(8 * (1 + 0.5)) = 12；极端 alpha 时截断到 K_max
    K, _, _ = adaptive_sizes(rho=0.0, eta=0.0)
    assert K == 12
    K2, _, _ = adaptive_sizes(rho=0.0, eta=0.0, alpha=10.0)
    assert K2 == 32  # K_max


def test_adaptive_sizes_high_eta_large_R_S():
    _, R, S = adaptive_sizes(rho=0.0, eta=1.0)
    assert R == 6  # clip(4 * 1.5) = 6
    assert S == 12  # clip(8 * 1.5) = 12
    assert R > 4 and S > 8


def test_adaptive_sizes_low_eta_small_R_S():
    _, R, S = adaptive_sizes(rho=0.0, eta=0.0)
    assert R == 4 and S == 8


def test_adaptive_sizes_clip_bounds():
    # 极端 rho/eta（含越界值）均不越界：R 缺省上界时受 m 限制
    for rho, eta in [(2.5, -1.5), (1.0, 10.0), (0.0, 0.0), (-3.0, 5.0)]:
        K, R, S = adaptive_sizes(rho, eta, m=16)
        assert 2 <= K <= 32
        assert 1 <= R <= 16
        assert 2 <= S <= 32
    # 极端 K0 下 K 被截断到 K_max
    K, _, _ = adaptive_sizes(0.0, 0.0, K0=100, alpha=1.0)
    assert K == 32


def test_adaptive_sizes_respect_m():
    # m=3：R = clip(6, 1, 3) = 3
    _, R1, _ = adaptive_sizes(rho=0.0, eta=1.0, m=3)
    assert R1 == 3
    # R_max=5, m=3 → min(5, 3) = 3
    _, R2, _ = adaptive_sizes(rho=0.0, eta=1.0, R_max=5, m=3)
    assert R2 == 3
    # R_max=2, m=16 → min(2, 16) = 2
    _, R3, _ = adaptive_sizes(rho=0.0, eta=1.0, R_max=2, m=16)
    assert R3 == 2


def test_adaptive_sizes_deterministic():
    r1 = adaptive_sizes(0.37, 0.63, m=16)
    r2 = adaptive_sizes(0.37, 0.63, m=16)
    assert r1 == r2
    assert all(isinstance(x, int) for x in r1)


def test_adaptive_sizes_invalid_bases():
    with pytest.raises(ValueError):
        adaptive_sizes(0.0, 0.0, K0=0)
    with pytest.raises(ValueError):
        adaptive_sizes(0.0, 0.0, R0=0)
    with pytest.raises(ValueError):
        adaptive_sizes(0.0, 0.0, S0=0)


def test_adaptive_config_defaults():
    assert DEFAULT_CONFIG == AdaptiveConfig()
    assert DEFAULT_CONFIG.window == 5
    # 推荐配置：均值匹配 fixed 规模（K=8, R=4, S=8）
    assert DEFAULT_CONFIG.K0 == 6
    assert DEFAULT_CONFIG.R0 == 3
    assert DEFAULT_CONFIG.S0 == 6
    assert DEFAULT_CONFIG.S_max == 8
    assert DEFAULT_CONFIG.R_max is None
    d = asdict(DEFAULT_CONFIG)
    assert set(d.keys()) == {
        "K0", "R0", "S0", "alpha", "beta", "gamma",
        "K_min", "K_max", "R_min", "R_max", "S_min", "S_max", "window",
    }
