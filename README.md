# hcsp-bottleneck-als

面向异构独立任务调度问题（HCSP, Heterogeneous Computing Scheduling Problem）的
“初解无关的瓶颈导向自适应邻域搜索”（Bottleneck-oriented Adaptive Neighborhood
Search）实验项目。

## Citation

This repository accompanies the paper:

> J. Lin and Y. Li, “BLS: A Bottleneck-Oriented Post-Processing Operator
> for Heterogeneous Task Scheduling,” in *Proc. PDCAT 2026* (to appear).

If you use this code, please cite the paper above.

**问题描述**：给定 n 个独立任务、m 个异构计算节点，以及执行时间矩阵 P（形状
(n, m)，其中 P[i, j] > 0 表示任务 i 在节点 j 上的执行时间）。求一个可行分配
`assignment`（形状 (n,)，`assignment[i]` 为任务 i 分配的节点），使最大完工时间
C_max = max_j L_j 最小，其中 L_j = Σ_{i: assignment[i]=j} P[i, j]。

本仓库当前进度：
- **第一步：基础模块**（数据加载、初始解、可行性检查、评估工具）与单元测试；
- **第二步：局部搜索**（瓶颈分析与候选选择、普通/瓶颈导向局部搜索、对比工具）；
- **第三步：自适应候选规模**（rho/eta 反馈调节 K/R/S、自适应搜索、消融实验）；
- **第四步：验证阶段**（五个核心问题的低成本验证实验与汇总）；
- **第五步：破坏-重建机制**（destroy-repair 逃离局部最优，消融为负结果，见边界讨论）；
- **第六步：真实 HCSP 数据**（官方 benchmark 适配、加载验证、主实验与 Q6 重测）。

论文级正式实验（更大规模、重复次数）将在验证结论支持的方法定位基础上进行。

## 目录结构

```
hcsp-bottleneck-als/
├── data/
│   └── hcsp/                # 官方 HCSP benchmark（512x16/1024x32/2048x64，含 .sol 最优解）
├── src/
│   ├── __init__.py          # 包入口，导出公共 API
│   ├── loader.py            # 加载 HCSP 矩阵（自动检测官方/通用格式）/ 解文件 / 生成合成矩阵
│   ├── initial_solutions.py # 轮询、Min-Min、LPT、随机贪心
│   ├── feasibility.py       # 可行性检查与目标函数（负载、C_max）
│   ├── evaluate.py          # 改进率计算与计时
│   ├── bottleneck.py        # 瓶颈定位、尖锐度、候选任务选择
│   ├── local_search.py      # 普通局部搜索 / 瓶颈导向局部搜索
│   ├── adaptive.py          # 自适应候选规模调节器（rho/eta → K/R/S）
│   ├── adaptive_search.py   # 自适应瓶颈邻域搜索
│   ├── destroy_repair.py    # 破坏-重建算子与迭代局部搜索（ILS）
│   └── compare.py           # 批量实验对比工具
├── experiments/
│   ├── run_adaptive_ablation.py   # 固定 vs 自适应消融实验脚本
│   ├── verify_q1_equal_evals.py   # Q1：等评估次数对比（最关键）
│   ├── verify_q2_equal_time.py    # Q2：等时间预算对比
│   ├── verify_q3_init_robustness.py   # Q3：初解稳健性
│   ├── verify_q4_instance_structure.py # Q4：实例结构（rho 分档）
│   ├── verify_q5_param_sensitivity.py  # Q5：参数敏感性网格
│   ├── verify_q6_destroy_repair.py     # Q6：破坏-重建消融（阶段一/二）
│   ├── validate_real_data.py      # 真实数据加载与最优解锚定验证
│   ├── run_real_main.py           # 真实数据主实验（--fast/--full）
│   ├── run_real_q6.py             # 真实数据 Q6 DR 重测
│   └── verify_summary.py          # 汇总脚本（读 Q1–Q5 CSV → Markdown）
├── results/                 # 消融与验证输出（CSV / JSON / Markdown 汇总）
├── tests/
│   ├── conftest.py          # pytest 配置（自动把项目根目录加入 sys.path）
│   ├── test_loader.py
│   ├── test_loader_real.py  # 真实数据格式（自动检测/.sol/文件名解析）
│   ├── test_initial_solutions.py
│   ├── test_feasibility.py
│   ├── test_bottleneck.py
│   ├── test_local_search.py
│   ├── test_adaptive.py
│   ├── test_adaptive_search.py
│   └── test_destroy_repair.py
├── PROJECT_MEMORY.md        # 项目记忆文档（供后续会话快速恢复上下文）
└── README.md
```

## 环境要求与安装

- Python 3.10+（推荐 3.12/3.13；注：本机默认 Python 3.14.0a5 为 alpha 版，与 numpy 轮子不兼容）
- numpy
- matplotlib
- pytest

Dependencies: see requirements.txt

本项目已附带 `.venv` 虚拟环境（Python 3.13.15，已安装 numpy 2.5.3、pytest 9.1.1），可直接使用：

```
.venv\Scripts\python.exe -m pytest tests -v
```

若需自行创建环境：

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 运行测试

在项目根目录下运行：

```
cd hcsp-bottleneck-als
pytest -v
```

或使用项目虚拟环境：

```
.venv\Scripts\python.exe -m pytest tests -v
```

（`tests/conftest.py` 会自动把项目根目录加入模块搜索路径，无需安装包即可运行测试。）

## 加载 HCSP 数据

`load_hcsp_matrix` 自动检测两种格式，将矩阵文件放入 `data/hcsp/` 后直接加载：

```python
from src.loader import load_hcsp_matrix, load_hcsp_solution, parse_optimal_from_filename

P = load_hcsp_matrix("data/hcsp/your_matrix.txt")    # P 形状 (n, m)
a = load_hcsp_solution("data/hcsp/your_matrix.sol")  # 长度 n 的 0-based 节点编号（int64）
c = parse_optimal_from_filename("data/hcsp/your_matrix_7360953,7.sol")  # 7360953.7
print(P.shape, a.shape, c)
```

**格式一：通用格式**（每行一个任务）：

- 每行 m 个数值，共 n 行，空格或制表符分隔；
- 所有执行时间必须 > 0；
- 空行会被跳过。

**格式二：官方 HCSP 格式**（自动检测，无需手工处理）：

- 首行 `n m [id]`（2–3 个整数 token），随后 n*m 行每行一个数值，按 C 序
  （行主序）reshape 为 (n, m)；
- 检测条件：首行恰为 2/3 个整数 token 且后续非空行数 == n*m，否则回退到
  通用格式（宁可回退不抛异常）；
- 官方 benchmark 部分实例（lolo/lohi 类）含 0 值执行时间，按数据事实原样
  保留（仅拒绝负值；实验脚本会把 0 替换为 1e-6 以适配算法模块的 p > 0 要求）。

`.sol` 解文件兼容两种写法：每行一个节点编号（CPR）或单行空格分隔全部编号
（MS/bare），均为 0-based。文件名中的最优 makespan 为 **1 位小数舍入值**
（舍入误差 ≤ 0.05），`parse_optimal_from_filename` 无法解析时返回 `None`。

## 初始解方法

| 函数 | 说明 | 复杂度 |
| --- | --- | --- |
| `round_robin` | 任务按顺序轮流分配到各节点 | O(n) |
| `min_min` | 经典 Min-Min | O(n²m) |
| `lpt` | 按平均执行时间降序，放入负载最小的节点 | O(n log n + n m) |
| `random_greedy` | 随机顺序 + 最小负载贪心（固定种子） | O(n m) |

所有函数返回三元组 `(assignment, loads, cmax)`：
`assignment` 形状 (n,) 的节点索引数组，`loads` 形状 (m,) 的节点负载，
`cmax` 为最大完工时间。

## 最小示例

```python
from src.loader import generate_synthetic_matrix
from src.initial_solutions import min_min

P = generate_synthetic_matrix(n=100, m=8, seed=42)
assignment, loads, cmax = min_min(P)
print("C_max =", cmax)
```

在项目根目录下可一行运行：

```
python -c "from src.loader import generate_synthetic_matrix; from src.initial_solutions import min_min; P = generate_synthetic_matrix(100, 8, seed=42); a, loads, cmax = min_min(P); print('C_max =', round(cmax, 3))"
```

## 第二步：局部搜索

### 瓶颈分析（`src/bottleneck.py`）

| 函数 | 说明 |
| --- | --- |
| `find_bottleneck(loads)` | 返回负载最大的节点索引；并列时取编号最小（确定性） |
| `bottleneck_sharpness(loads)` | rho = (L_max - L_second) / L_max；m <= 1 或 L_max <= 0 时返回 0 |
| `select_candidate_tasks(P, assignment, node, K)` | 从 node 上的任务中按 P[i, node] 降序选前 K 个 |

### 局部搜索（`src/local_search.py`）

两个搜索接口统一，返回 dict：

```python
{
    "assignment": ...,      # 最终分配 (n,)
    "loads": ...,           # 最终负载 (m,)
    "cmax": ...,            # 最终 C_max
    "evals": ...,           # 实际候选评估次数
    "accepted_moves": ...,  # 接受动作数
    "improved": ...,        # 相对输入是否严格改善（bool）
    "history": [            # 每轮一条：{"round", "cmax", "evals", "accepted"}
        {"round": 1, "cmax": ..., "evals": ..., "accepted": 1},
        ...
    ],
}
```

- `plain_local_search(P, assignment, loads, cmax, max_evals, tol=1e-9)`：
  基线。固定顺序枚举单任务迁移（外层任务 0..n-1，内层节点 0..m-1），
  每轮执行 C'_max 最小的严格改善动作（best-improvement），无改善即停。
- `bottleneck_local_search(P, assignment, loads, cmax, K=8, R=4, S=8, max_evals=10000, tol=1e-9, time_budget=None)`：
  瓶颈优先。每轮定位瓶颈节点，从其任务中选 K 个候选（按执行时间降序），
  对每个候选按 L_j + P[i, j] 升序选 R 个目标节点评估迁移，再对每个目标
  节点按 P[h, j] 降序选 S 个任务评估交换；取 C'_max 最小的严格改善动作
  执行，无改善即停。
- `time_budget`（两搜索共有的可选参数，默认 None）：给定秒数后，在每轮
  边界检查已用时间，超过预算即停止（与 max_evals 先到者为准）。用于
  等时间预算实验（Q2）。默认行为与旧版本一致。

设计约定：
- 接受条件：C'_max < C_max - tol（严格改善）；
- 迁移与交换的负载更新均为增量计算（只改源/目标两个位置），不重算全部节点；
- 平局动作取枚举顺序中先出现者（确定性，可复现）；
- 输入 (assignment, loads, cmax) 在入口处用 `verify_solution` 核验，不一致抛 `ValueError`。

### 对比工具（`src/compare.py`）

```python
from src.compare import run_comparison

res = run_comparison(P, initial_fn=min_min, max_evals=10000, K=8, R=4, S=8)
print(res["initial"]["cmax"])      # 初解 C_max
print(res["plain"]["cmax"])        # 普通局部搜索
print(res["bottleneck"]["cmax"])   # 瓶颈导向局部搜索
```

### 最小示例（合成 100×8，Min-Min 初解 + 瓶颈局部搜索）

```python
from src.loader import generate_synthetic_matrix
from src.initial_solutions import min_min
from src.local_search import bottleneck_local_search
from src.evaluate import improvement_rate

P = generate_synthetic_matrix(n=100, m=8, seed=42)
assignment, loads, c0 = min_min(P)
res = bottleneck_local_search(P, assignment, loads, c0, max_evals=20000)
print("C0 =", c0)
print("C_final =", res["cmax"])
print("improved =", res["improved"])
print("improvement (%):", improvement_rate(c0, res["cmax"]))
```

## 第三步：自适应候选规模

### 反馈信号（`src/adaptive.py`）

- `rho = bottleneck_sharpness(loads)`：瓶颈尖锐度 = (L_max - L_second) / L_max。
  rho 越大说明瓶颈越突出，搜索应集中火力于瓶颈 → 缩小 K。
- `eta = compute_progress(history, window)`：近期接受率 = 最近 window 轮中
  accepted > 0 的轮占比。eta 越大说明邻域仍有潜力 → 扩大 R、S。

### 调节规则（`adaptive_sizes`）

```
K = clip(K0 * (1 + alpha * (1 - rho)), K_min, K_max)
R = clip(R0 * (1 + beta * eta),  R_min, min(R_max 或 m, m))   # m 提供时
S = clip(S0 * (1 + gamma * eta), S_min, S_max)
```

- rho、eta 先防御性截断到 [0, 1]，结果四舍五入（round-half-up）取整；
- `AdaptiveConfig` 数据类集中存放全部参数；
- **默认配置（均值匹配）**：K∈[6,9]、R∈[3,4]、S∈[6,8]（围绕固定规模
  K=8/R=4/S=8 摆动），平均每轮评估数 ~220 与固定规模（288）相当。
  诊断发现：在 best-improvement 框架下每轮预算越大轮数越少、改进越慢，
  因此默认配置不选择让邻域整体膨胀的参数（如 K∈[8,12]、S∈[8,12]）。
- 注意：`adaptive_sizes` 的签名默认值（8/4/8、0.5/0.5/0.5）是规范示例值，
  与 `AdaptiveConfig` 的字段默认值不同，搜索内部显式传参、互不影响。

### 自适应搜索（`src/adaptive_search.py`）

```python
adaptive_bottleneck_search(P, assignment, loads, cmax,
                           config=AdaptiveConfig(), max_evals=10000, tol=1e-9)
```

每轮：定位瓶颈 → rho/eta → adaptive_sizes 得到本轮 (K, R, S) →
枚举迁移/交换（与 `bottleneck_local_search` 共享 `_evaluate_round`）→
执行最优严格改善动作 → history 追加记录。

返回 dict 在标准字段基础上额外增加：

```python
{
    # ...标准字段（assignment/loads/cmax/evals/accepted_moves/improved）,
    "history": [
        {"round", "cmax", "evals", "accepted",  # 与第二步一致
         "rho", "eta", "K", "R", "S"},          # 第三步新增（每轮都有）
        ...
    ],
    "config": {...},          # 本次运行使用的配置（asdict）
    "final_sizes": (K, R, S), # 最后一轮规模；无轮次时为 None
}
```

### 三搜索对比（`src/compare.py`）

```python
from src.compare import run_adaptive_comparison

res = run_adaptive_comparison(P, initial_fn=min_min, max_evals=10000)
print(res["plain"]["cmax"])     # 基线（普通局部搜索）
print(res["fixed"]["cmax"])     # 固定规模 bottleneck（K=8, R=4, S=8）
print(res["adaptive"]["cmax"])  # 自适应 bottleneck
```

### 消融实验（`experiments/run_adaptive_ablation.py`）

```
# 快速模式：100x8、重复 1 次（验证脚本可用）
.venv\Scripts\python.exe experiments\run_adaptive_ablation.py --fast

# 正式实验：512x16、1024x32、2048x64 × 4 初解 × 3 预算 × 3 搜索 × 重复 5 次
.venv\Scripts\python.exe experiments\run_adaptive_ablation.py

# 自定义规模 / 预算 / 重复次数
.venv\Scripts\python.exe experiments\run_adaptive_ablation.py --sizes 512x16 1024x32 --max-evals 2000 5000 --repeats 3
```

输出：
- `results/adaptive_ablation.csv`：逐次运行明细
  （n, m, init_method, searcher, max_evals, seed, c0, c_final,
  improvement_pct, evals, accepted_moves, runtime_sec）；
- `results/adaptive_ablation_summary.json`：按 (n, m, init_method, searcher,
  max_evals) 分组的改进率均值与标准差。

实验设计约定：合成矩阵 seed 固定（2026）；重复通过改变 initial_fn 的 seed
实现（只对 random_greedy 生效，其余初解为确定性方法）；计时用
time.perf_counter。若 adaptive 在部分单元上差于 fixed，CSV/汇总会如实记录。

### 最小示例（合成 100×8，Min-Min 初解，fixed vs adaptive）

```python
from src.loader import generate_synthetic_matrix
from src.initial_solutions import min_min
from src.local_search import bottleneck_local_search
from src.adaptive_search import adaptive_bottleneck_search
from src.evaluate import improvement_rate

P = generate_synthetic_matrix(n=100, m=8, seed=42)
a0, l0, c0 = min_min(P)

res_fixed = bottleneck_local_search(P, a0, l0, c0, max_evals=10000)
res_adapt = adaptive_bottleneck_search(P, a0, l0, c0, max_evals=10000)

print("C0 =", round(c0, 3))
print("fixed    : C =", round(res_fixed["cmax"], 3),
      " improvement =", round(improvement_rate(c0, res_fixed["cmax"]), 3), "%")
print("adaptive : C =", round(res_adapt["cmax"], 3),
      " improvement =", round(improvement_rate(c0, res_adapt["cmax"]), 3), "%")
print("final sizes (K, R, S) =", res_adapt["final_sizes"])
```

实测输出（合成 100×8、seed=42、Min-Min 初解、max_evals=10000）：

```
C0 = 207.571
fixed    : C = 169.303  improvement = 18.436 %
adaptive : C = 169.303  improvement = 18.436 %
final sizes (K, R, S) = (9, 4, 8)
```

## 验证阶段（第四步）

验证阶段用最小成本回答 5 个核心问题，判断方法的核心贡献是否成立。
每个脚本单独可运行、互不依赖；结果写入 `results/` 下对应 CSV；
最后用 `verify_summary.py` 汇总为 `results/verify_summary.md`。

### 5 个验证问题的动机

| 问题 | 动机 | 判定标准（用户给出） |
| --- | --- | --- |
| Q1 等评估次数：bottleneck vs plain | 核心假设：瓶颈优先的候选筛选是否真的更有效 | 更优比例 > 60% 且平均差值 > 0.5pp → 成立 |
| Q2 等时间预算 | 优势是否只来自"每轮评估量少、轮数多" | 若等时间下优势消失 → 论文需重新定位贡献 |
| Q3 初解稳健性 | 初解无关性是否成立 | 四种初解下改进率均在合理范围 |
| Q4 实例结构（rho 分档） | 方法适用范围边界 | 尖锐实例优势明显、平坦实例优势消失 → 边界清晰 |
| Q5 参数敏感性 | 默认 (8,4,8) 是否够好 | 与最优组合差距 < 0.5pp → 参数不敏感、易用 |

### 运行命令

```
# Q1（最关键）：3 规模 × 4 初解 × 4 预算 × 2 搜索，约 1–2 分钟
.venv\Scripts\python.exe experiments\verify_q1_equal_evals.py

# Q2：同上实例与初解，时间预算 1/3/5s，约 3–5 分钟
.venv\Scripts\python.exe experiments\verify_q2_equal_time.py

# Q3：512x16 × 4 初解 × 5 种子，秒级
.venv\Scripts\python.exe experiments\verify_q3_init_robustness.py

# Q4：3 种结构 × 10 矩阵种子，秒级
.venv\Scripts\python.exe experiments\verify_q4_instance_structure.py

# Q5：2 规模 × 2 初解 × 27 参数组，秒级
.venv\Scripts\python.exe experiments\verify_q5_param_sensitivity.py

# 汇总：读 Q1–Q5 CSV，生成 results/verify_summary.md
.venv\Scripts\python.exe experiments\verify_summary.py
```

### 输出文件

| 文件 | 内容 |
| --- | --- |
| `results/verify_q1.csv` | 等评估次数对比（每单元格一行，含 c0/c_plain/c_bottleneck/evals/accepted/runtime/bottleneck_better） |
| `results/verify_q2.csv` | 等时间预算对比（字段同上，另含实际耗时） |
| `results/verify_q3.csv` | 每 (初解, 种子) 一次运行，含改进率 |
| `results/verify_q4.csv` | 每实例一行，含 rho、rho_bin、plain/bottleneck 改进率 |
| `results/verify_q5.csv` | 每 (规模, 初解, K, R, S) 一次运行 |
| `results/verify_summary.md` | 汇总：Q1–Q5 统计、核心假设判定、贡献定位建议 |

### 汇总文件解读

`verify_summary.md` 由脚本按规则生成，不预填数字：

1. **Q1 节**：bottleneck 更优单元格比例、平均差值、最好/最差单元格；
2. **Q2 节**：等时间预算下同样统计 + 每档时间预算的分档统计；
3. **Q3 节**：每初解改进率分布表与稳健性判定；
4. **Q4 节**：按 rho 分档的改进率表（注意：本数据集中 Min-Min 初解的
   rho 很少达到 0.15，高尖锐档可能无样本）；
5. **Q5 节**：最优 3 组参数、固定 (8,4,8) 与其差距、敏感性判定；
6. **总结节**：核心假设是否成立 + 方法贡献定位建议（Q2 决定贡献
   是否来自候选筛选本身）。

## 第五步：破坏-重建机制（destroy-and-repair）

针对 best-improvement 框架下局部搜索“轮数受限、易陷入局部最优”的问题，
实现了破坏-重建增强（`src/destroy_repair.py`）：迭代局部搜索（ILS）框架，
局部搜索收敛后破坏一部分任务并随机重建，再继续搜索，以逃离局部最优。

### 破坏-重建算子（`destroy_repair`）

```python
destroy_repair(P, assignment, loads, cmax, destroy_frac=0.2,
               strategy="bottleneck", seed=None, rng=None)
    -> (assignment_new, loads_new, cmax_new)
```

- `strategy="bottleneck"`（默认）：按节点负载降序，从瓶颈节点开始、
  节点内按执行时间降序选取 `q = max(1, round(destroy_frac * n))` 个任务破坏；
- `strategy="random"`：随机选取 q 个任务破坏；
- 重建：每个被破坏任务随机分配到**除原节点外**的任一节点（保证扰动非平凡；
  m=1 时保持原分配）。

### 迭代局部搜索（`bottleneck_local_search_dr`）

```python
bottleneck_local_search_dr(P, assignment, loads, cmax, K=4, R=2, S=4,
                           max_evals=10000, tol=1e-9, destroy_frac=0.2,
                           destroy_strategy="bottleneck",
                           accept_strategy="strict", segment_evals=1000,
                           max_stagnation=3, cooling_rate=0.8,
                           seed=42, time_budget=None)
```

主循环：`bottleneck_local_search` 分段执行（每段预算 = min(segment_evals,
剩余预算)）→ 段末按接受准则决定扰动起点 → 破坏-重建 → 下一段，直到预算
耗尽或连续 `max_stagnation` 段无全局改善。破坏-重建本身不消耗评估次数。

- `accept_strategy="strict"`（默认）：经典 ILS，只从全局最优解扰动；
- `accept_strategy="metropolis"`：从当前解扰动，以 T=0.05·c0 的
  Metropolis 准则接受变差解，每段按 `cooling_rate` 退火。

返回 dict 在标准字段基础上额外增加：`restarts`（破坏-重建次数）、
`improvements`（全局最优改善次数）、`accepted_worse`（接受变差次数）、
`segments`（已执行段数）；history 条目与第二步标准一致（4 字段）。

### Q6 消融实验（`experiments/verify_q6_destroy_repair.py`）

```
# 阶段一（必做）：512/1024 规模 × round_robin/min_min × 5000/10000 预算
#   A: bottleneck_local_search(K=4,R=2,S=4) 基线
#   B: DR strict    C: DR metropolis
.venv\Scripts\python.exe experiments\verify_q6_destroy_repair.py
```

输出：
- `results/verify_q6_phase1.csv`、`results/verify_q6_phase1_summary.md`；
- 阶段二（2048×64、lpt、random_greedy、20000，输出 `verify_q6_full.*`）
  仅在阶段一 B 相对 A 平均改进率提升 >= +1.0pp 且多数单元格不劣化时触发。

**实测结论（负结果，如实记录）**：阶段一 8 个单元格中 B 平均比 A 低
**-7.47pp**（更优 0 格、持平 4 格、劣化 4 格），未触发阶段二。原因：

1. **min_min 初解**：瓶颈搜索 440–640 evals 即收敛，其局部最优盆地在
   随机合成矩阵上唯一且深——即使破坏 50% 任务，重搜仍回到同一局部最优；
2. **round_robin 初解**：(4,2,4) 邻域需数千评估才收敛，切段破坏扰乱了
   尚未完成的下降进程（segment_evals 越小劣化越大）。

**机制有效性演示（陷阱实例）**：60 任务 × 3 节点、执行时间按组循环
roll [1,5,9] 的错配实例上，单次瓶颈搜索收敛于 c=70（I=61.11%）后停滞
（760 evals 耗尽邻域，再给预算也无法改进）；DR strict 在 3 个种子下分别
逃逸至 c=20/24/24（I=86.7%–88.9%），首次超越单次局部最优仅需累计
200–280 evals。说明破坏-重建机制本身有效，只是在随机均匀实例的
landscape 上没有用武之地。

论文定位建议：作为**边界讨论**报告——DR 仅在存在深局部最优结构的实例上
有价值，对随机均匀实例无贡献；实现与参数接口（destroy_strategy /
accept_strategy / segment_evals 等）保留为未来工作接口。

## 第六步：真实 HCSP 数据

在官方 HCSP benchmark（来源：chgogos/hcsp GitHub 仓库，sparse-checkout 至
`data/hcsp/`）上完成加载适配、锚定验证与主实验。

### 数据概览

- 60 个矩阵 + 72 个解文件：512×16 全部 12 个（u_c/u_i/u_s × hihi/hilo/lohi/lolo）；
  1024×32 与 2048×64 各 A/B 两组 × 12 类 = 24 个矩阵。
- 512/1024 头部为 `n m id`（id 与类别相关），2048 头部为 `n m`；
  矩阵体每行一个数值（C 序）。
- 解文件：CPR 后缀 = 每行一个编号；MS 后缀 = 单行空格分隔；均 0-based。
- 文件名末尾含官方最优 makespan（1 位小数，如 `_CPR_7360953,7.sol`）。

### 加载验证（必须先跑通）

```
.venv\Scripts\python.exe experiments\validate_real_data.py
```

遍历全部矩阵与解文件：核对形状/取值、.sol 长度与编号范围、用 .sol 分配重算
makespan 并与文件名最优值对比（判据 0.051 = 1 位小数舍入上界 0.05 + 1e-3）。
输出 `results/real_data_validation.csv`；任何文件失败会打印 `[FAIL]` 且退出码非 0。
实测：60 矩阵 + 72 解全部 `[OK]`，退出码 0。

### 主实验

```
# 快速冒烟：512×16 中的 4 个矩阵（约 1–2 分钟）
.venv\Scripts\python.exe experiments\run_real_main.py --fast

# 正式实验：28 矩阵（512×16 全部 12 + 1024×32 抽 8 + 2048×64 抽 8，
# 抽样 seed=42，可用 --seed 覆盖；选中文件名打印在汇总中）
# × 2 初解 × 2 搜索 × 2 预算 = 224 单元格（2048×64 扩样：2026-10 补丁 4→8，
# 约束 u_c/u_i/u_s 各 ≥2、hihi/lolo 各 ≥2）
.venv\Scripts\python.exe experiments\run_real_main.py --full
```

输出：
- `results/real_main.csv`：每单元格一行（c0/c_final/c_opt/改进率/gap/evals/耗时/seed 等；
  seed 列回溯抽样与记录使用的随机种子）；
- `results/real_main_summary.md`：按 (规模, 初解, 搜索, 预算) 分组统计、
  bottleneck 优于 plain 的配对比例、与合成数据（verify_q1）的对比。

### Q6 破坏-重建重测

```
.venv\Scripts\python.exe experiments\run_real_q6.py
# 可用 --seed 覆盖抽样种子（默认 42）；CSV 含 seed 列
```

16 矩阵（512×16 全部 12 + 1024×32 抽 4）× 2 初解 × 2 搜索
（A: bottleneck(4,2,4) vs B: DR strict）× 2 预算，输出 `results/real_q6.csv`
与 `real_q6_summary.md`（按 u_c/u_i/u_s 分组对比 A/B）。

### 实测结论（如实记录）

- **瓶颈优先在真实数据上依然有效**：bottleneck 优于 plain 的配对比例 **112/112（100%）**。
- **gap_final 取决于初解**：min_min 初解下 bottleneck 的 gap_final 为 7.10%（512×16）、
  9.51%（1024×32）、11.10%（2048×64），plain 为 10.3–20.0%；round_robin 初解质量极差
  （gap 数百~数千%），bottleneck 可大幅缩小但 plain 几乎无效。
- **显著性（配对符号检验，scipy 未安装用 binomial 替代）**：整体 112 配对
  p = 3.85e-34；2048×64 扩样后每组 8 对全部 8/8 更优（p = 0.0078 < 0.05），
  各规模 × 初解 × 预算分组全部显著。
- **破坏-重建（DR）真实数据重测仍为负**：mean(B-A) = -8.50pp；u_i（inconsistent）
  类上 -14.52pp 且 0/24 更优 → 不支持“DR 在 inconsistent 类上有效”的假设，
  与合成 Q6（-7.47pp）构成双重负结果 → 降级为未来工作。
- **合成 vs 真实**：真实数据上改进空间显著更大（512 round_robin、预算 10000：
  bottleneck 改进率真实 68.99% vs 合成 35.83%），合成结果可作为保守下界。

## 下一步

- 真实数据主实验已跑通（28 矩阵、224 单元格、112 配对全部显著），论文级实验可增加重复次数；
- gap 报告建议按初解分别呈现（min_min 7–12% vs round_robin 数百%）；
- 破坏-重建（DR）经合成与真实双重负结果，降级为未来工作（保留接口）；
- 若需进一步提升：小邻域多轮（缩小 K/R/S 基准）或非贪心接受准则（如模拟退火式）。
