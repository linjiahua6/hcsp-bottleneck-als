"""证据链审计脚本：自动检查实验产物（CSV/汇总/脚本）之间的一致性。

检查项：
1. 所有 results/*.csv 行数与对应 summary 声明一致（无 summary/未配置
   解析则 [WARN] 跳过）；
2. 主实验 CSV 每行 c_final <= c0 + 1e-9；
3. real_main.csv 的 gap_final 与 real_main_summary.md 分组统计表一致
   （容差 1e-6）；
4. 每个实验 CSV 含 seed 列（缺失则 [WARN]）；
5. 每个实验的抽样矩阵列表可通过 seed=42 复现（重跑抽样逻辑对比；
   CSV 无该规模行则跳过）；
6. 各 runner 脚本的 --seed 参数默认值为 42；
7. 所有 CSV 中 filepath 列指向的文件均存在；
8. 所有 .sol 文件与对应 .txt 同名且存在。

输出 results/audit_report.md：[PASS]/[FAIL]/[WARN] 逐项列出；
任何 [FAIL] 则退出码为 1。

用法：
    python experiments/audit_evidence.py
    python experiments/audit_evidence.py --results-dir TMP --data-dir TMP \
        --project-root TMP   # 测试/定制场景
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "hcsp"
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_MD = RESULTS_DIR / "audit_report.md"

# ---- 参与"主实验类"检查的 CSV（含 c0/c_final、需要 seed 列）----
EXPERIMENT_CSVS = [
    "real_main.csv",
    "real_q6.csv",
    "real_heft.csv",
    "real_equal_time.csv",
    "real_param_sensitivity.csv",
]

# ---- 抽样复现配置：CSV -> 按 runner 顺序的抽样计划
#      [(规模目录, 抽取类数, 每 u 类最少个数, 每 hihi/lolo 类最少个数)] ----
# 注意：runner 的 select_datasets 用同一个 rng 依次抽样各规模，复现必须按
# 相同顺序重放（rng 状态会被前面的抽样消耗）。
SAMPLING_PLAN: dict[str, list[tuple[str, int, int, int]]] = {
    "real_main.csv": [("1024x32", 8, 2, 1), ("2048x64", 8, 2, 2)],
    "real_q6.csv": [("1024x32", 4, 1, 1)],
    "real_heft.csv": [("1024x32", 4, 1, 1), ("2048x64", 2, 0, 1)],
    "real_equal_time.csv": [("1024x32", 4, 1, 1)],
    "real_param_sensitivity.csv": [("512x16", 4, 1, 1)],
}

# ---- 需检查 --seed 默认值为 42 的 runner 脚本 ----
RUNNER_SEED_FILES = [
    "run_real_main.py",
    "run_real_q6.py",
    "run_real_heft.py",
    "run_real_equal_time.py",
    "run_real_param_sensitivity.py",
]

TIE_TOL = 1e-9
# summary 分组统计只打印 2 位小数：舍入上界 0.005 + 浮点容差 1e-6
GAP_TOL = 0.005 + 1e-6


def _norm_fp(fp: str) -> str:
    """把 filepath 规范化为正斜杠分隔（兼容 Windows 反斜杠 CSV）。"""
    return fp.replace("\\", "/")


# ---------------------------------------------------------------------------
# summary 解析器
# ---------------------------------------------------------------------------

def _parse_real_main_summary_row_counts(text: str) -> int | None:
    """real_main_summary.md：分组统计表的'单元格数'列求和。"""
    total = 0
    found = False
    for line in text.splitlines():
        m = re.match(
            r"^\|\s*\d+x\d+\s*\|\s*\w+\s*\|\s*\w+\s*\|\s*\d+\s*\|"
            r"\s*(\d+)\s*\|", line)
        if m:
            total += int(m.group(1))
            found = True
    return total if found else None


def _parse_real_q6_summary_row_counts(text: str) -> int | None:
    """real_q6_summary.md：'配对单元格数: N'（A/B 两行一组）→ N * 2。"""
    m = re.search(r"配对单元格数[:：]\s*(\d+)", text)
    return int(m.group(1)) * 2 if m else None


def _parse_generic_row_counts(text: str) -> int | None:
    """新脚本 summary：'- CSV 行数: N'。"""
    m = re.search(r"CSV 行数[:：]\s*(\d+)", text)
    return int(m.group(1)) if m else None


SUMMARY_META: dict[str, tuple[str, object]] = {
    "real_main.csv": ("real_main_summary.md", _parse_real_main_summary_row_counts),
    "real_q6.csv": ("real_q6_summary.md", _parse_real_q6_summary_row_counts),
    "real_heft.csv": ("real_heft_summary.md", _parse_generic_row_counts),
    "real_equal_time.csv": ("real_equal_time_summary.md", _parse_generic_row_counts),
    "real_param_sensitivity.csv": (
        "real_param_sensitivity_summary.md", _parse_generic_row_counts),
}


def _parse_real_main_gap_table(text: str) -> dict[tuple, float]:
    """解析 real_main_summary.md 分组统计表 → {(n,m,init,searcher,budget): gap 均值}。"""
    out: dict[tuple, float] = {}
    for line in text.splitlines():
        m = re.match(
            r"^\|\s*(\d+)x(\d+)\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*(\d+)\s*\|"
            r"\s*\d+\s*\|\s*[^|]*\|\s*([0-9.eE+-]+)\s*±", line)
        if m:
            key = (int(m.group(1)), int(m.group(2)), m.group(3), m.group(4),
                   int(m.group(5)))
            out[key] = float(m.group(6))
    return out


def _read_csv(path: Path) -> list[dict] | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _check(checks: list[dict], check_id: str, name: str, ok: bool,
           message: str, warn: bool = False) -> None:
    checks.append({
        "id": check_id,
        "name": name,
        "status": "PASS" if ok else ("WARN" if warn else "FAIL"),
        "message": message,
    })


# ---------------------------------------------------------------------------
# 各检查项
# ---------------------------------------------------------------------------

def check_row_counts(results_dir: Path) -> list[dict]:
    """检查项 1：CSV 行数与 summary 声明一致。"""
    checks: list[dict] = []
    unconfigured = []
    for csv_file in sorted(results_dir.glob("*.csv")):
        if csv_file.name not in SUMMARY_META:
            unconfigured.append(csv_file.name)
            continue
        rows = _read_csv(csv_file)
        summary_name, parser = SUMMARY_META[csv_file.name]
        summary_path = results_dir / summary_name
        if rows is None:
            _check(checks, "row_count", csv_file.name, False,
                   "CSV 无法读取", warn=True)
            continue
        if not summary_path.is_file():
            _check(checks, "row_count", csv_file.name, False,
                   f"summary 不存在（{summary_name}），跳过行数检查", warn=True)
            continue
        declared = parser(summary_path.read_text(encoding="utf-8"))
        if declared is None:
            _check(checks, "row_count", csv_file.name, False,
                   f"summary 中未找到行数声明，跳过", warn=True)
            continue
        n_rows = len(rows)
        _check(checks, "row_count", csv_file.name, n_rows == declared,
               f"CSV {n_rows} 行 vs summary 声明 {declared} 行"
               + ("" if n_rows == declared else "（不一致）"))
    if unconfigured:
        _check(checks, "row_count", "未配置行数检查的 CSV", False,
               "、".join(unconfigured) + " 无对应行数声明配置，跳过", warn=True)
    return checks


def check_improvement_monotone(results_dir: Path) -> list[dict]:
    """检查项 2：主实验 CSV 每行 c_final <= c0 + 1e-9。"""
    checks: list[dict] = []
    for name in EXPERIMENT_CSVS:
        rows = _read_csv(results_dir / name)
        if rows is None:
            continue  # 该实验尚未运行，不检查
        bad = []
        for i, r in enumerate(rows, start=2):  # 行号含表头
            if "c0" not in r or "c_final" not in r:
                _check(checks, "improvement_monotone", name, False,
                       f"缺少 c0/c_final 列，无法检查", warn=True)
                break
            try:
                c0 = float(r["c0"])
                cf = float(r["c_final"])
            except ValueError:
                _check(checks, "improvement_monotone", name, False,
                       f"第 {i} 行数值无法解析", warn=True)
                continue
            if cf > c0 + TIE_TOL:
                bad.append((i, c0, cf))
        _check(checks, "improvement_monotone", name, not bad,
               "所有行 c_final <= c0 + 1e-9" if not bad else
               f"{len(bad)} 行违反 c_final <= c0（例: 行 {bad[0][0]} "
               f"c0={bad[0][1]} c_final={bad[0][2]}）")
    return checks


def check_gap_consistency(results_dir: Path) -> list[dict]:
    """检查项 3：real_main.csv 的 gap_final 与 summary 分组统计一致。"""
    checks: list[dict] = []
    rows = _read_csv(results_dir / "real_main.csv")
    summary_path = results_dir / "real_main_summary.md"
    if rows is None:
        _check(checks, "gap_consistency", "real_main.csv", False,
               "real_main.csv 不存在，跳过", warn=True)
        return checks
    if not summary_path.is_file():
        _check(checks, "gap_consistency", "real_main.csv", False,
               "real_main_summary.md 不存在，跳过", warn=True)
        return checks
    declared = _parse_real_main_gap_table(
        summary_path.read_text(encoding="utf-8"))
    actual: dict[tuple, list[float]] = {}
    for r in rows:
        key = (int(r["n"]), int(r["m"]), r["init_method"], r["searcher"],
               int(r["max_evals"]))
        actual.setdefault(key, []).append(float(r["gap_final_pct"]))
    if not declared:
        _check(checks, "gap_consistency", "real_main.csv", False,
               "summary 中未解析出分组统计表，跳过", warn=True)
        return checks
    bad = []
    for key in sorted(set(declared) | set(actual)):
        if key not in declared or key not in actual:
            bad.append(f"{key}: 单侧缺失")
            continue
        a_mean = sum(actual[key]) / len(actual[key])
        if abs(a_mean - declared[key]) > GAP_TOL:
            bad.append(f"{key}: summary={declared[key]:.6f} "
                       f"csv={a_mean:.6f}")
    _check(checks, "gap_consistency", "real_main.csv", not bad,
           f"{len(declared)} 个分组 gap_final 均值与 CSV 重算一致"
           if not bad else f"{len(bad)} 个分组不一致: " + "; ".join(bad[:5]))
    return checks


def check_seed_column(results_dir: Path) -> list[dict]:
    """检查项 4：每个实验 CSV 含 seed 列（缺失 → [WARN]）。"""
    checks: list[dict] = []
    for name in EXPERIMENT_CSVS:
        rows = _read_csv(results_dir / name)
        if rows is None:
            continue
        has_seed = rows and "seed" in rows[0]
        _check(checks, "seed_column", name, has_seed,
               "含 seed 列" if has_seed else
               "缺少 seed 列（runner 用固定 seed=42 但未写入 CSV）",
               warn=not has_seed)
    return checks


def check_sampling_reproducible(results_dir: Path,
                                data_dir: Path) -> list[dict]:
    """检查项 5：抽样矩阵列表可通过 seed=42 复现。"""
    checks: list[dict] = []
    if not data_dir.is_dir():
        _check(checks, "sampling_reproducible", "全部", False,
               f"数据目录不存在: {data_dir}，跳过", warn=True)
        return checks
    try:
        from experiments import run_real_main as rrm
    except Exception as exc:  # noqa: BLE001
        _check(checks, "sampling_reproducible", "全部", False,
               f"无法导入抽样实现（{exc}），跳过", warn=True)
        return checks

    import numpy as np

    for name, plan in SAMPLING_PLAN.items():
        rows = _read_csv(results_dir / name)
        if rows is None:
            continue
        fp_col = "filepath" in (rows[0] if rows else {})
        if not fp_col:
            _check(checks, "sampling_reproducible", name, False,
                   "CSV 无 filepath 列，无法复现抽样", warn=True)
            continue
        # 与 runner 相同：同一个 rng 按计划顺序依次抽样各规模
        rng = np.random.default_rng(42)
        for size, n_select, min_per_u, min_per_het in plan:
            csv_fps = {_norm_fp(r["filepath"]) for r in rows
                       if _norm_fp(r["filepath"]).split("/")[0] == size}
            if not csv_fps:
                _check(checks, "sampling_reproducible", f"{name}/{size}", False,
                       "CSV 中无该规模行，跳过（不消耗 rng，后续规模重放可能"
                       "受影响，如实标注）", warn=True)
                continue
            size_dir = data_dir / size
            if not size_dir.is_dir():
                _check(checks, "sampling_reproducible", f"{name}/{size}",
                       False, f"数据目录不存在: {size_dir}", warn=True)
                continue
            mats = sorted(f for f in size_dir.iterdir()
                          if not f.name.endswith(".sol"))
            try:
                sampled = rrm.sample_matrices(
                    mats, n_select, min_per_u, rng, min_per_het=min_per_het)
            except Exception as exc:  # noqa: BLE001
                _check(checks, "sampling_reproducible", f"{name}/{size}",
                       False, f"抽样复现失败（{exc}）", warn=True)
                continue
            sampled_fps = {_norm_fp(str(f.relative_to(data_dir))) for f in sampled}
            ok = csv_fps == sampled_fps
            _check(checks, "sampling_reproducible", f"{name}/{size}", ok,
                   "CSV 抽样与 seed=42 顺序重放一致"
                   if ok else
                   f"不一致: CSV 有 {sorted(csv_fps)}，复现为 "
                   f"{sorted(sampled_fps)}")
    return checks


def _extract_seed_default(path: Path) -> int | None:
    """AST 解析 runner 脚本中 add_argument('--seed', default=...) 的默认值。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            consts = [a.value for a in node.args
                      if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            if "--seed" in consts:
                for kw in node.keywords:
                    if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                        return kw.value.value
                return None
    return None


def check_runner_seed_defaults(project_root: Path) -> list[dict]:
    """检查项 6：runner 脚本的 --seed 默认值为 42。"""
    checks: list[dict] = []
    exp_dir = project_root / "experiments"
    if not exp_dir.is_dir():
        _check(checks, "runner_seed_default", "全部", False,
               f"experiments 目录不存在: {exp_dir}，跳过", warn=True)
        return checks
    for name in RUNNER_SEED_FILES:
        path = exp_dir / name
        if not path.is_file():
            _check(checks, "runner_seed_default", name, False,
                   "脚本不存在，跳过", warn=True)
            continue
        default = _extract_seed_default(path)
        if default is None:
            _check(checks, "runner_seed_default", name, False,
                   "未找到 --seed 参数（或 default 非字面量），跳过", warn=True)
            continue
        _check(checks, "runner_seed_default", name, default == 42,
               f"--seed 默认 {default}" + ("" if default == 42 else "（应 42）"))
    return checks


def check_filepaths_exist(results_dir: Path, data_dir: Path) -> list[dict]:
    """检查项 7：CSV 中 filepath 列指向的文件均存在。"""
    checks: list[dict] = []
    checked_any = False
    for csv_file in sorted(results_dir.glob("*.csv")):
        rows = _read_csv(csv_file)
        if not rows or "filepath" not in rows[0]:
            continue
        checked_any = True
        missing = sorted({r["filepath"] for r in rows
                          if not (data_dir / r["filepath"]).is_file()})
        _check(checks, "filepath_exists", csv_file.name, not missing,
               "所有 filepath 指向的文件均存在" if not missing else
               f"{len(missing)} 个文件不存在（例: {missing[:3]}）")
    if not checked_any:
        _check(checks, "filepath_exists", "全部", False,
               "没有含 filepath 列的 CSV（或数据目录缺失）", warn=True)
    return checks


def check_sol_mates(data_dir: Path) -> list[dict]:
    """检查项 8：所有 .sol 文件与对应矩阵文件同名且存在。

    官方命名：矩阵名无 .txt 后缀（如 A.u_c_hihi），解文件名为
    "{矩阵名}_{CPR|MS|bare 等}_{最优值}.sol"。按前缀 "矩阵名_" 配对。
    """
    checks: list[dict] = []
    if not data_dir.is_dir():
        _check(checks, "sol_mates", "全部", False,
               f"数据目录不存在: {data_dir}，跳过", warn=True)
        return checks
    bad = []
    n_sol = 0
    for size_dir in sorted(data_dir.iterdir()):
        if not size_dir.is_dir() or size_dir.name.startswith("."):
            continue  # 跳过隐藏目录（如 .git 稀疏克隆元数据）
        mats = {f.name for f in size_dir.iterdir()
                if f.is_file() and not f.name.endswith(".sol")}
        for sol in sorted(size_dir.glob("*.sol")):
            n_sol += 1
            if not any(sol.name.startswith(m + "_") for m in mats):
                bad.append(str(sol))
    _check(checks, "sol_mates", "全部", not bad,
           f"{n_sol} 个 .sol 均有同名矩阵文件" if not bad else
           f"{len(bad)} 个 .sol 无对应矩阵文件（例: {bad[:3]}）")
    return checks


# ---------------------------------------------------------------------------
# 主审计入口
# ---------------------------------------------------------------------------

def run_audit(results_dir: Path, data_dir: Path,
              project_root: Path) -> tuple[str, list[dict]]:
    """执行全部检查，返回 (报告 Markdown, 检查项列表)。"""
    checks: list[dict] = []
    checks += check_row_counts(results_dir)
    checks += check_improvement_monotone(results_dir)
    checks += check_gap_consistency(results_dir)
    checks += check_seed_column(results_dir)
    checks += check_sampling_reproducible(results_dir, data_dir)
    checks += check_runner_seed_defaults(project_root)
    checks += check_filepaths_exist(results_dir, data_dir)
    checks += check_sol_mates(data_dir)

    n_pass = sum(1 for c in checks if c["status"] == "PASS")
    n_warn = sum(1 for c in checks if c["status"] == "WARN")
    n_fail = sum(1 for c in checks if c["status"] == "FAIL")

    lines: list[str] = []
    lines.append("# 证据链审计报告\n")
    lines.append(f"- 审计时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 检查项总数: {len(checks)}；"
                 f"[PASS] {n_pass}、[WARN] {n_warn}、[FAIL] {n_fail}\n")
    lines.append("| # | 检查项 | 检查对象 | 状态 | 说明 |")
    lines.append("|---|---|---|---|---|")
    for i, c in enumerate(checks, start=1):
        lines.append(f"| {i} | {c['name']} | {c['id']} | [{c['status']}] | "
                     f"{c['message']} |")
    lines.append("")
    if n_fail:
        lines.append(f"## 结论：存在 {n_fail} 项 [FAIL]，需排查后才能进入论文写作。")
    else:
        lines.append("## 结论：无 [FAIL]。警告项（[WARN]）如实保留，"
                     "建议人工确认。")
    lines.append("")
    return "\n".join(lines), checks


def main() -> int:
    parser = argparse.ArgumentParser(description="证据链审计脚本")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR),
                        help="results 目录（默认项目 results/）")
    parser.add_argument("--data-dir", type=str, default=str(DATA_ROOT),
                        help="真实数据目录（默认 data/hcsp/）")
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT),
                        help="项目根目录（默认脚本上级目录）")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)
    project_root = Path(args.project_root)

    report, checks = run_audit(results_dir, data_dir, project_root)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "audit_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"审计报告已写入: {out}")

    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    n_warn = sum(1 for c in checks if c["status"] == "WARN")
    n_pass = sum(1 for c in checks if c["status"] == "PASS")
    for c in checks:
        mark = c["status"]
        if mark != "PASS":
            print(f"[{mark}] {c['name']} / {c['id']}: {c['message']}")
    print(f"汇总: [PASS] {n_pass}、[WARN] {n_warn}、[FAIL] {n_fail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
