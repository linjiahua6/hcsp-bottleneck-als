# 证据链审计报告

- 审计时间: 2026-10-04 12:45:11
- 检查项总数: 36；[PASS] 35、[WARN] 1、[FAIL] 0

| # | 检查项 | 检查对象 | 状态 | 说明 |
|---|---|---|---|---|
| 1 | real_equal_time.csv | row_count | [PASS] | CSV 192 行 vs summary 声明 192 行 |
| 2 | real_heft.csv | row_count | [PASS] | CSV 72 行 vs summary 声明 72 行 |
| 3 | real_main.csv | row_count | [PASS] | CSV 224 行 vs summary 声明 224 行 |
| 4 | real_param_sensitivity.csv | row_count | [PASS] | CSV 96 行 vs summary 声明 96 行 |
| 5 | real_q6.csv | row_count | [PASS] | CSV 128 行 vs summary 声明 128 行 |
| 6 | 未配置行数检查的 CSV | row_count | [WARN] | adaptive_ablation.csv、real_data_validation.csv、verify_q1.csv、verify_q2.csv、verify_q3.csv、verify_q4.csv、verify_q5.csv、verify_q6_phase1.csv 无对应行数声明配置，跳过 |
| 7 | real_main.csv | improvement_monotone | [PASS] | 所有行 c_final <= c0 + 1e-9 |
| 8 | real_q6.csv | improvement_monotone | [PASS] | 所有行 c_final <= c0 + 1e-9 |
| 9 | real_heft.csv | improvement_monotone | [PASS] | 所有行 c_final <= c0 + 1e-9 |
| 10 | real_equal_time.csv | improvement_monotone | [PASS] | 所有行 c_final <= c0 + 1e-9 |
| 11 | real_param_sensitivity.csv | improvement_monotone | [PASS] | 所有行 c_final <= c0 + 1e-9 |
| 12 | real_main.csv | gap_consistency | [PASS] | 24 个分组 gap_final 均值与 CSV 重算一致 |
| 13 | real_main.csv | seed_column | [PASS] | 含 seed 列 |
| 14 | real_q6.csv | seed_column | [PASS] | 含 seed 列 |
| 15 | real_heft.csv | seed_column | [PASS] | 含 seed 列 |
| 16 | real_equal_time.csv | seed_column | [PASS] | 含 seed 列 |
| 17 | real_param_sensitivity.csv | seed_column | [PASS] | 含 seed 列 |
| 18 | real_main.csv/1024x32 | sampling_reproducible | [PASS] | CSV 抽样与 seed=42 顺序重放一致 |
| 19 | real_main.csv/2048x64 | sampling_reproducible | [PASS] | CSV 抽样与 seed=42 顺序重放一致 |
| 20 | real_q6.csv/1024x32 | sampling_reproducible | [PASS] | CSV 抽样与 seed=42 顺序重放一致 |
| 21 | real_heft.csv/1024x32 | sampling_reproducible | [PASS] | CSV 抽样与 seed=42 顺序重放一致 |
| 22 | real_heft.csv/2048x64 | sampling_reproducible | [PASS] | CSV 抽样与 seed=42 顺序重放一致 |
| 23 | real_equal_time.csv/1024x32 | sampling_reproducible | [PASS] | CSV 抽样与 seed=42 顺序重放一致 |
| 24 | real_param_sensitivity.csv/512x16 | sampling_reproducible | [PASS] | CSV 抽样与 seed=42 顺序重放一致 |
| 25 | run_real_main.py | runner_seed_default | [PASS] | --seed 默认 42 |
| 26 | run_real_q6.py | runner_seed_default | [PASS] | --seed 默认 42 |
| 27 | run_real_heft.py | runner_seed_default | [PASS] | --seed 默认 42 |
| 28 | run_real_equal_time.py | runner_seed_default | [PASS] | --seed 默认 42 |
| 29 | run_real_param_sensitivity.py | runner_seed_default | [PASS] | --seed 默认 42 |
| 30 | real_data_validation.csv | filepath_exists | [PASS] | 所有 filepath 指向的文件均存在 |
| 31 | real_equal_time.csv | filepath_exists | [PASS] | 所有 filepath 指向的文件均存在 |
| 32 | real_heft.csv | filepath_exists | [PASS] | 所有 filepath 指向的文件均存在 |
| 33 | real_main.csv | filepath_exists | [PASS] | 所有 filepath 指向的文件均存在 |
| 34 | real_param_sensitivity.csv | filepath_exists | [PASS] | 所有 filepath 指向的文件均存在 |
| 35 | real_q6.csv | filepath_exists | [PASS] | 所有 filepath 指向的文件均存在 |
| 36 | 全部 | sol_mates | [PASS] | 72 个 .sol 均有同名矩阵文件 |

## 结论：无 [FAIL]。警告项（[WARN]）如实保留，建议人工确认。
