# BLS-M（仅迁移消融）运行汇总

- 生成时间: 2026-10-10 21:16:28
- 规模: 512x16 全部 12 个矩阵；初解: min_min、round_robin；预算: [5000, 10000]
- 变体: bottleneck_migration_search(K=4, R=2, S=0)；seed=42
- 单元格数: 48（12 x 2 x 2）

## 分组统计（init x budget）

| init | budget | 单元格数 | 改进率均值±std(%) | gap_final 均值±std(%) | evals 中位数 | termination 分布 |
|---|---|---:|---:|---:|---:|---|
| min_min | 5000 | 12 | 4.50±3.69 | 8.39±2.37 | 32 | local_optimum=12 |
| min_min | 10000 | 12 | 4.50±3.69 | 8.39±2.37 | 32 | local_optimum=12 |
| round_robin | 5000 | 12 | 67.38±8.22 | 91.86±37.61 | 1800 | local_optimum=12 |
| round_robin | 10000 | 12 | 67.38±8.22 | 91.86±37.61 | 1800 | local_optimum=12 |
