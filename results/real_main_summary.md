# 真实 HCSP 数据主实验汇总（块 1 + 块 2）

- 生成时间: 2026-10-04 12:39:42
- 模式: full（seed=42）
- 初解: min_min、round_robin；搜索: plain vs bottleneck(K=4,R=2,S=4)；预算: [5000, 10000]
- 0 值处理: 官方 lolo/lohi 类矩阵含 0 值执行时间，实验层替换为 1e-06（如实报告，不改变排序关系）

## 数据选取

- 1024x32（8 个）: A.u_c_lohi, A.u_c_lolo, A.u_s_hilo, B.u_c_hihi, B.u_i_hilo, B.u_i_lohi, B.u_i_lolo, B.u_s_lolo
- 2048x64（8 个）: A.u_c_lolo, A.u_i_lolo, A.u_s_hilo, B.u_c_hihi, B.u_c_lohi, B.u_i_hihi, B.u_i_hilo, B.u_s_hihi
- 512x16（12 个）: u_c_hihi.0, u_c_hilo.0, u_c_lohi.0, u_c_lolo.0, u_i_hihi.0, u_i_hilo.0, u_i_lohi.0, u_i_lolo.0, u_s_hihi.0, u_s_hilo.0, u_s_lohi.0, u_s_lolo.0

## 分组统计（size x init x searcher x budget）

| size | init | searcher | budget | 单元格数 | 改进率均值±std(%) | gap_final 均值±std(%) |
|---|---|---|---|---:|---:|---:|
| 512x16 | min_min | bottleneck | 5000 | 12 | 5.68±2.80 | 7.10±2.70 |
| 512x16 | min_min | bottleneck | 10000 | 12 | 5.68±2.80 | 7.10±2.70 |
| 512x16 | min_min | plain | 5000 | 12 | 2.23±2.64 | 11.07±4.48 |
| 512x16 | min_min | plain | 10000 | 12 | 2.91±2.61 | 10.30±4.31 |
| 512x16 | round_robin | bottleneck | 5000 | 12 | 59.67±8.99 | 159.07±118.59 |
| 512x16 | round_robin | bottleneck | 10000 | 12 | 68.99±6.20 | 88.90±56.54 |
| 512x16 | round_robin | plain | 5000 | 12 | 5.31±2.17 | 519.44±266.90 |
| 512x16 | round_robin | plain | 10000 | 12 | 8.94±3.70 | 495.08±253.44 |
| 1024x32 | min_min | bottleneck | 5000 | 8 | 5.50±1.70 | 9.51±1.63 |
| 1024x32 | min_min | bottleneck | 10000 | 8 | 5.50±1.70 | 9.51±1.63 |
| 1024x32 | min_min | plain | 5000 | 8 | 0.56±0.60 | 15.29±3.31 |
| 1024x32 | min_min | plain | 10000 | 8 | 0.56±0.60 | 15.29±3.31 |
| 1024x32 | round_robin | bottleneck | 5000 | 8 | 54.92±9.58 | 622.24±389.23 |
| 1024x32 | round_robin | bottleneck | 10000 | 8 | 68.75±8.39 | 413.97±303.67 |
| 1024x32 | round_robin | plain | 5000 | 8 | 4.31±1.73 | 1352.13±521.77 |
| 1024x32 | round_robin | plain | 10000 | 8 | 4.34±1.74 | 1351.84±521.81 |
| 2048x64 | min_min | bottleneck | 5000 | 8 | 7.69±4.66 | 11.10±2.22 |
| 2048x64 | min_min | bottleneck | 10000 | 8 | 7.69±4.66 | 11.10±2.22 |
| 2048x64 | min_min | plain | 5000 | 8 | 0.57±0.95 | 20.00±6.94 |
| 2048x64 | min_min | plain | 10000 | 8 | 1.11±0.92 | 19.35±6.94 |
| 2048x64 | round_robin | bottleneck | 5000 | 8 | 43.06±6.62 | 1381.07±1017.84 |
| 2048x64 | round_robin | bottleneck | 10000 | 8 | 52.14±9.48 | 1184.98±955.73 |
| 2048x64 | round_robin | plain | 5000 | 8 | 3.06±1.24 | 2284.97±1413.15 |
| 2048x64 | round_robin | plain | 10000 | 8 | 3.46±1.34 | 2280.17±1417.03 |

## bottleneck 优于 plain 的单元格比例

（同 (filepath, init, budget) 配对，c_final 严格更小（1e-9 容差）计为更优）

| size | init | budget | 配对数 | bottleneck 更优比例 |
|---|---|---:|---:|---:|
| 512x16 | min_min | 5000 | 12 | 12/12 (100.0%) |
| 512x16 | min_min | 10000 | 12 | 12/12 (100.0%) |
| 512x16 | round_robin | 5000 | 12 | 12/12 (100.0%) |
| 512x16 | round_robin | 10000 | 12 | 12/12 (100.0%) |
| 1024x32 | min_min | 5000 | 8 | 8/8 (100.0%) |
| 1024x32 | min_min | 10000 | 8 | 8/8 (100.0%) |
| 1024x32 | round_robin | 5000 | 8 | 8/8 (100.0%) |
| 1024x32 | round_robin | 10000 | 8 | 8/8 (100.0%) |
| 2048x64 | min_min | 5000 | 8 | 8/8 (100.0%) |
| 2048x64 | min_min | 10000 | 8 | 8/8 (100.0%) |
| 2048x64 | round_robin | 5000 | 8 | 8/8 (100.0%) |
| 2048x64 | round_robin | 10000 | 8 | 8/8 (100.0%) |

## 与合成数据（verify_q1）对比

注意：verify_q1 用 bottleneck(K=8,R=4,S=8)、预算 2000-20000、初解含 lpt/random_greedy；本实验用 (K=4,R=2,S=4)、预算 [5000, 10000]。以下仅对比双方共同覆盖的单元格类型（按 init 匹配），量级参考。

| 数据 | size | init | budget | plain 改进率均值(%) | bottleneck 改进率均值(%) |
|---|---|---|---:|---:|---:|
| 合成(seed=42) | 512x16 | min_min | 5000 | 0.31 | 3.03 |
| 真实 | 512x16 | min_min | 5000 | 2.23 | 5.68 |
| 合成(seed=42) | 512x16 | min_min | 10000 | 0.45 | 3.12 |
| 真实 | 512x16 | min_min | 10000 | 2.91 | 5.68 |
| 合成(seed=42) | 512x16 | round_robin | 5000 | 6.48 | 26.93 |
| 真实 | 512x16 | round_robin | 5000 | 5.31 | 59.67 |
| 合成(seed=42) | 512x16 | round_robin | 10000 | 10.70 | 35.83 |
| 真实 | 512x16 | round_robin | 10000 | 8.94 | 68.99 |

## 最优性 gap 说明

- gap = (C - c_opt) / c_opt * 100%，c_opt 为官方 .sol 解（CPR/MS 中 makespan 最小者）在实验矩阵上重算的值；
- c_opt 的原始文件名精度为 1 位小数（舍入误差 <= 0.05），重算值精度更高，gap 计算不受其影响。
