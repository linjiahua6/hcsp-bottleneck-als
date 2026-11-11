"""HCSP 瓶颈导向自适应邻域搜索 —— 基础模块包。

包含以下子模块：
- loader:             HCSP 矩阵加载与合成数据生成；
- initial_solutions:  轮询、Min-Min、LPT、随机贪心四种初始解；
- heft:               HEFT 初始解（独立任务退化版：列表调度 + EFT 分配）；
- feasibility:        可行性检查与目标函数（负载、C_max）；
- evaluate:           改进率计算与计时工具；
- bottleneck:         瓶颈定位、尖锐度与候选任务选择；
- local_search:       普通局部搜索与瓶颈导向局部搜索；
- adaptive:           自适应候选规模调节器（rho/eta → K/R/S）；
- adaptive_search:    自适应瓶颈邻域搜索；
- destroy_repair:     破坏—重建机制与迭代局部搜索（逃离局部最优）；
- compare:            批量实验对比工具。
"""

from .adaptive import DEFAULT_CONFIG, AdaptiveConfig, adaptive_sizes, compute_progress
from .adaptive_search import adaptive_bottleneck_search
from .bottleneck import bottleneck_sharpness, find_bottleneck, select_candidate_tasks
from .compare import run_adaptive_comparison, run_comparison
from .destroy_repair import bottleneck_local_search_dr, destroy_repair
from .evaluate import improvement_rate, run_with_timer
from .feasibility import (
    check_feasibility,
    compute_cmax,
    compute_loads,
    verify_solution,
)
from .heft import heft
from .initial_solutions import lpt, min_min, random_greedy, round_robin
from .loader import (
    generate_synthetic_matrix,
    load_hcsp_matrix,
    load_hcsp_solution,
    parse_optimal_from_filename,
)
from .local_search import bottleneck_local_search, plain_local_search

__all__ = [
    "load_hcsp_matrix",
    "load_hcsp_solution",
    "parse_optimal_from_filename",
    "generate_synthetic_matrix",
    "round_robin",
    "min_min",
    "lpt",
    "random_greedy",
    "heft",
    "check_feasibility",
    "compute_loads",
    "compute_cmax",
    "verify_solution",
    "improvement_rate",
    "run_with_timer",
    "find_bottleneck",
    "bottleneck_sharpness",
    "select_candidate_tasks",
    "plain_local_search",
    "bottleneck_local_search",
    "run_comparison",
    "compute_progress",
    "adaptive_sizes",
    "AdaptiveConfig",
    "DEFAULT_CONFIG",
    "adaptive_bottleneck_search",
    "run_adaptive_comparison",
    "destroy_repair",
    "bottleneck_local_search_dr",
]
