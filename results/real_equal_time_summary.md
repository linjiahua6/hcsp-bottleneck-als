# 实验 A：真实 HCSP 数据等时间预算对比汇总

- 生成时间: 2026-10-04 12:03:48
- 模式: full（seed=42）
- 初解: min_min、round_robin；搜索: plain vs bottleneck(K=4,R=2,S=4)；时间预算: [1.0, 3.0, 5.0] 秒
- 实现: 两搜索均传 max_evals=1e9 与 time_budget（bottleneck 无额外时间；先到者为准）
- timed_out 判定（近似）: runtime_sec >= time_budget_sec 视为时间耗尽；预算内因无改善动作提前收敛则记为 0
- 0 值处理: 官方 lolo/lohi 类矩阵含 0 值执行时间，实验层替换为 1e-06（如实报告，不改变排序关系）

- CSV 行数: 192

## 数据选取

- 1024x32（4 个）: A.u_s_hilo, B.u_c_hihi, B.u_i_lohi, B.u_i_lolo
- 512x16（12 个）: u_c_hihi.0, u_c_hilo.0, u_c_lohi.0, u_c_lolo.0, u_i_hihi.0, u_i_hilo.0, u_i_lohi.0, u_i_lolo.0, u_s_hihi.0, u_s_hilo.0, u_s_lohi.0, u_s_lolo.0

## 分组统计（size x init x searcher x time_budget）

| size | init | searcher | tb(s) | 单元格数 | 改进率均值±std(%) | gap_final 均值±std(%) | 平均 evals_actual | timed_out 比例 |
|---|---|---|---|---:|---:|---:|---:|---:|
| 512x16 | min_min | bottleneck | 1 | 12 | 5.68±2.80 | 7.10±2.70 | 353 | 0% |
| 512x16 | min_min | bottleneck | 3 | 12 | 5.68±2.80 | 7.10±2.70 | 353 | 0% |
| 512x16 | min_min | bottleneck | 5 | 12 | 5.68±2.80 | 7.10±2.70 | 353 | 0% |
| 512x16 | min_min | plain | 1 | 12 | 6.87±3.04 | 5.74±2.67 | 261760 | 0% |
| 512x16 | min_min | plain | 3 | 12 | 6.87±3.04 | 5.74±2.67 | 261760 | 0% |
| 512x16 | min_min | plain | 5 | 12 | 6.87±3.04 | 5.74±2.67 | 261760 | 0% |
| 512x16 | round_robin | bottleneck | 1 | 12 | 72.97±7.65 | 55.64±21.35 | 11733 | 0% |
| 512x16 | round_robin | bottleneck | 3 | 12 | 72.97±7.65 | 55.64±21.35 | 11733 | 0% |
| 512x16 | round_robin | bottleneck | 5 | 12 | 72.97±7.65 | 55.64±21.35 | 11733 | 0% |
| 512x16 | round_robin | plain | 1 | 12 | 47.58±16.44 | 248.05±200.54 | 1450240 | 75% |
| 512x16 | round_robin | plain | 3 | 12 | 50.34±15.99 | 223.63±177.23 | 2689920 | 33% |
| 512x16 | round_robin | plain | 5 | 12 | 51.57±15.86 | 210.65±160.11 | 3687040 | 33% |
| 1024x32 | min_min | bottleneck | 1 | 4 | 5.64±1.93 | 10.12±1.64 | 540 | 0% |
| 1024x32 | min_min | bottleneck | 3 | 4 | 5.64±1.93 | 10.12±1.64 | 540 | 0% |
| 1024x32 | min_min | bottleneck | 5 | 4 | 5.64±1.93 | 10.12±1.64 | 540 | 0% |
| 1024x32 | min_min | plain | 1 | 4 | 7.02±2.23 | 8.48±0.81 | 1523712 | 100% |
| 1024x32 | min_min | plain | 3 | 4 | 7.56±2.14 | 7.86±0.74 | 2745856 | 0% |
| 1024x32 | min_min | plain | 5 | 4 | 7.56±2.14 | 7.86±0.74 | 2745856 | 0% |
| 1024x32 | round_robin | bottleneck | 1 | 4 | 81.02±10.60 | 216.08±234.54 | 24610 | 0% |
| 1024x32 | round_robin | bottleneck | 3 | 4 | 81.02±10.60 | 216.08±234.54 | 24610 | 0% |
| 1024x32 | round_robin | bottleneck | 5 | 4 | 81.02±10.60 | 216.08±234.54 | 24610 | 0% |
| 1024x32 | round_robin | plain | 1 | 4 | 28.61±11.09 | 1119.16±515.23 | 1523712 | 100% |
| 1024x32 | round_robin | plain | 3 | 4 | 38.52±15.17 | 978.59±533.76 | 4571136 | 100% |
| 1024x32 | round_robin | plain | 5 | 4 | 43.40±17.76 | 913.27±548.54 | 7578880 | 100% |

## bottleneck 优于 plain 的配对比例

（同 (filepath, init, tb) 配对，c_final 严格更小（1e-9 容差）计为更优）

| size | init | tb(s) | 配对数 | bottleneck 更优比例 |
|---|---|---:|---:|---:|
| 512x16 | min_min | 1 | 12 | 0/12 (0.0%) |
| 512x16 | min_min | 3 | 12 | 0/12 (0.0%) |
| 512x16 | min_min | 5 | 12 | 0/12 (0.0%) |
| 512x16 | round_robin | 1 | 12 | 12/12 (100.0%) |
| 512x16 | round_robin | 3 | 12 | 12/12 (100.0%) |
| 512x16 | round_robin | 5 | 12 | 12/12 (100.0%) |
| 1024x32 | min_min | 1 | 4 | 0/4 (0.0%) |
| 1024x32 | min_min | 3 | 4 | 0/4 (0.0%) |
| 1024x32 | min_min | 5 | 4 | 0/4 (0.0%) |
| 1024x32 | round_robin | 1 | 4 | 4/4 (100.0%) |
| 1024x32 | round_robin | 3 | 4 | 4/4 (100.0%) |
| 1024x32 | round_robin | 5 | 4 | 4/4 (100.0%) |

## 评估次数对比（关键指标：优势是否来自更多评估）

| size | init | tb(s) | plain 平均 evals | bottleneck 平均 evals | 倍数(plain/bn) |
|---|---|---:|---:|---:|---:|
| 512x16 | min_min | 1 | 261760 | 353 | 740.8x |
| 512x16 | min_min | 3 | 261760 | 353 | 740.8x |
| 512x16 | min_min | 5 | 261760 | 353 | 740.8x |
| 512x16 | round_robin | 1 | 1450240 | 11733 | 123.6x |
| 512x16 | round_robin | 3 | 2689920 | 11733 | 229.3x |
| 512x16 | round_robin | 5 | 3687040 | 11733 | 314.2x |
| 1024x32 | min_min | 1 | 1523712 | 540 | 2821.7x |
| 1024x32 | min_min | 3 | 2745856 | 540 | 5084.9x |
| 1024x32 | min_min | 5 | 2745856 | 540 | 5084.9x |
| 1024x32 | round_robin | 1 | 1523712 | 24610 | 61.9x |
| 1024x32 | round_robin | 3 | 4571136 | 24610 | 185.7x |
| 1024x32 | round_robin | 5 | 7578880 | 24610 | 308.0x |

- 解读：等时间预算下 plain 的单次评估更快、总评估次数更多；若 bottleneck 仍更优，说明优势来自候选筛选质量而非评估数量。

## 与合成数据（verify_q2）对比

注意：verify_q2 用 bottleneck(K=8,R=4,S=8)；本实验用 (K=4,R=2,S=4)。仅对比双方共同覆盖的 512x16 单元格，量级参考。

| 数据 | init | tb(s) | plain 改进率均值(%) | bottleneck 改进率均值(%) |
|---|---|---:|---:|---:|
| 合成(seed=42) | min_min | 1 | 3.32 | 3.12 |
| 真实 | min_min | 1 | 6.87 | 5.68 |
| 合成(seed=42) | min_min | 3 | 3.32 | 3.12 |
| 真实 | min_min | 3 | 6.87 | 5.68 |
| 合成(seed=42) | min_min | 5 | 3.32 | 3.12 |
| 真实 | min_min | 5 | 6.87 | 5.68 |
| 合成(seed=42) | round_robin | 1 | 39.39 | 49.88 |
| 真实 | round_robin | 1 | 47.58 | 72.97 |
| 合成(seed=42) | round_robin | 3 | 42.43 | 49.88 |
| 真实 | round_robin | 3 | 50.34 | 72.97 |
| 合成(seed=42) | round_robin | 5 | 42.43 | 49.88 |
| 真实 | round_robin | 5 | 51.57 | 72.97 |

## 提前停止（无改善动作收敛）情况

- 预算内提前收敛的单元格按 (size, init, searcher, tb) 统计：
  - 512x16 / min_min / bottleneck / 1s: 12 个
  - 512x16 / min_min / bottleneck / 3s: 12 个
  - 512x16 / min_min / bottleneck / 5s: 12 个
  - 512x16 / min_min / plain / 1s: 12 个
  - 512x16 / min_min / plain / 3s: 12 个
  - 512x16 / min_min / plain / 5s: 12 个
  - 512x16 / round_robin / bottleneck / 1s: 12 个
  - 512x16 / round_robin / bottleneck / 3s: 12 个
  - 512x16 / round_robin / bottleneck / 5s: 12 个
  - 512x16 / round_robin / plain / 1s: 3 个
  - 512x16 / round_robin / plain / 3s: 8 个
  - 512x16 / round_robin / plain / 5s: 8 个
  - 1024x32 / min_min / bottleneck / 1s: 4 个
  - 1024x32 / min_min / bottleneck / 3s: 4 个
  - 1024x32 / min_min / bottleneck / 5s: 4 个
  - 1024x32 / min_min / plain / 3s: 4 个
  - 1024x32 / min_min / plain / 5s: 4 个
  - 1024x32 / round_robin / bottleneck / 1s: 4 个
  - 1024x32 / round_robin / bottleneck / 3s: 4 个
  - 1024x32 / round_robin / bottleneck / 5s: 4 个
