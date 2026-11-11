# Q6 破坏-重建机制在真实 HCSP 数据上的重测汇总

- 生成时间: 2026-10-04 12:43:31（抽样 seed=42，DR seed=42）
- 数据: 512x16 全部 12 个 + 1024x32 抽 4 个（16 个矩阵）
- 初解: min_min、round_robin；预算: [5000, 10000]
- A: bottleneck_local_search(K=4,R=2,S=4)；B: bottleneck_local_search_dr(K=4,R=2,S=4, strict, destroy_frac=0.2, segment_evals=1000, max_stagnation=3)

## 数据选取

- 1024x32（4 个）: A.u_s_hilo, B.u_c_hihi, B.u_i_lohi, B.u_i_lolo
- 512x16（12 个）: u_c_hihi.0, u_c_hilo.0, u_c_lohi.0, u_c_lolo.0, u_i_hihi.0, u_i_hilo.0, u_i_lohi.0, u_i_lolo.0, u_s_hihi.0, u_s_hilo.0, u_s_lohi.0, u_s_lolo.0

## 总体结果（A vs B）

- 配对单元格数: 64
- mean(B-A) = -8.50pp；B 更优 1 格、劣化 31 格、持平 32 格

## 按类别分组（category x init x budget）

| category | init | budget | 配对数 | imp A 均值 | imp B 均值 | B-A (pp) | B 更优 |
|---|---|---:|---:|---:|---:|---:|---:|
| u_c | min_min | 5000 | 5 | 3.59 | 3.59 | +0.00 | 0/5 |
| u_c | min_min | 10000 | 5 | 3.59 | 3.59 | +0.00 | 0/5 |
| u_c | round_robin | 5000 | 5 | 66.00 | 64.01 | -1.99 | 1/5 |
| u_c | round_robin | 10000 | 5 | 70.28 | 65.86 | -4.43 | 0/5 |
| u_i | min_min | 5000 | 6 | 7.05 | 7.05 | +0.00 | 0/6 |
| u_i | min_min | 10000 | 6 | 7.05 | 7.05 | +0.00 | 0/6 |
| u_i | round_robin | 5000 | 6 | 47.34 | 25.74 | -21.60 | 0/6 |
| u_i | round_robin | 10000 | 6 | 63.67 | 27.18 | -36.49 | 0/6 |
| u_s | min_min | 5000 | 5 | 6.09 | 6.09 | +0.00 | 0/5 |
| u_s | min_min | 10000 | 5 | 6.09 | 6.09 | +0.00 | 0/5 |
| u_s | round_robin | 5000 | 5 | 62.53 | 49.71 | -12.82 | 0/5 |
| u_s | round_robin | 10000 | 5 | 72.05 | 52.21 | -19.84 | 0/5 |

## 假设检验：DR 是否在 inconsistent 类（u_i）上有效

- u_i 配对单元格 24 个，mean(B-A) = -14.52pp，B 更优 0/24
- 判定: 不支持假设（DR 在 u_i 上未表现出正贡献）（如实报告，不含蓄意取舍）

## 与合成数据 Q6（verify_q6_phase1）对比

- 合成数据（512x16/1024x32, seed=42）: mean(B-A) = -7.47pp，B 更优 0 格、劣化 4 格、持平 4 格；判定：DR 在典型随机实例上无正贡献（负结果如实记录）。
- 真实数据（本次）: mean(B-A) = -8.50pp，B 更优 1/64。

详见合成数据摘要: `verify_q6_phase1_summary.md`
