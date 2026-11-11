"""证据链审计脚本（experiments/audit_evidence.py）的单元测试。

测试通过构造临时 results/ 与 data/ 目录，驱动 run_audit 的检查逻辑。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments import audit_evidence as audit

HEADER = ("filepath,n,m,init_method,searcher,max_evals,c0,c_final,c_opt,"
          "improvement_pct,gap_initial_pct,gap_final_pct,evals,"
          "accepted_moves,runtime_sec,seed")

ROW_PLAIN = ("512x16/u_c_hihi.0,512,16,min_min,plain,5000,100.0,95.0,80.0,"
             "5.0,25.0,18.75,100,1,0.1,42")
ROW_BN = ("512x16/u_c_hihi.0,512,16,min_min,bottleneck,5000,100.0,90.0,"
          "80.0,10.0,25.0,12.5,50,1,0.05,42")


def _write_real_main(results_dir: Path, rows: list[str],
                     header: str = HEADER) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "real_main.csv").write_text(
        header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def _make_data(data_dir: Path, relpaths: list[str]) -> None:
    for rel in relpaths:
        p = data_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("1 2\n3 4\n", encoding="utf-8")


def _status_of(checks: list[dict], check_id: str, target: str) -> str | None:
    for c in checks:
        if c["id"] == check_id and target in c["name"]:
            return c["status"]
    return None


def test_audit_passes_on_clean_results(tmp_path: Path) -> None:
    """干净环境（行数无 summary、数据完整、含 seed）不产生任何 [FAIL]。"""
    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    _write_real_main(results_dir, [ROW_PLAIN, ROW_BN])
    _make_data(data_dir, ["512x16/u_c_hihi.0"])

    _, checks = audit.run_audit(results_dir, data_dir, tmp_path)
    fails = [c for c in checks if c["status"] == "FAIL"]
    assert not fails, [c["message"] for c in fails]
    # 行数检查因无 summary 记 [WARN]（跳过而非失败）
    assert _status_of(checks, "row_count", "real_main.csv") == "WARN"


def test_audit_fails_on_inconsistent_row_count(tmp_path: Path) -> None:
    """summary 声明行数与 CSV 实际行数不一致 → [FAIL]。"""
    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    _write_real_main(results_dir, [ROW_PLAIN, ROW_BN])  # 实际 2 行
    _make_data(data_dir, ["512x16/u_c_hihi.0"])
    # summary 分组统计表宣称 3 个单元格（gap 均值与 CSV 一致，避免干扰）
    (results_dir / "real_main_summary.md").write_text(
        "# 真实 HCSP 数据主实验汇总（块 1 + 块 2）\n\n"
        "## 分组统计（size x init x searcher x budget）\n\n"
        "| size | init | searcher | budget | 单元格数 | 改进率均值±std(%) "
        "| gap_final 均值±std(%) |\n"
        "|---|---|---|---|---:|---:|---:|\n"
        "| 512x16 | min_min | plain | 5000 | 3 | 5.00±0.00 | 18.75±0.00 |\n",
        encoding="utf-8")

    _, checks = audit.run_audit(results_dir, data_dir, tmp_path)
    assert _status_of(checks, "row_count", "real_main.csv") == "FAIL"


def test_audit_fails_on_improvement_violation(tmp_path: Path) -> None:
    """存在 c_final > c0 的行 → [FAIL]。"""
    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    bad_row = ("512x16/u_c_hihi.0,512,16,min_min,plain,5000,100.0,105.0,"
               "80.0,-5.0,25.0,31.25,100,1,0.1,42")
    _write_real_main(results_dir, [bad_row, ROW_BN])
    _make_data(data_dir, ["512x16/u_c_hihi.0"])

    _, checks = audit.run_audit(results_dir, data_dir, tmp_path)
    assert _status_of(checks, "improvement_monotone", "real_main.csv") == "FAIL"


def test_audit_warns_on_missing_seed(tmp_path: Path) -> None:
    """CSV 缺少 seed 列 → [WARN]（不是 FAIL）。"""
    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    header_no_seed = HEADER.rsplit(",", 1)[0]
    _write_real_main(
        results_dir,
        [ROW_PLAIN.rsplit(",", 1)[0], ROW_BN.rsplit(",", 1)[0]],
        header=header_no_seed)
    _make_data(data_dir, ["512x16/u_c_hihi.0"])

    _, checks = audit.run_audit(results_dir, data_dir, tmp_path)
    assert _status_of(checks, "seed_column", "real_main.csv") == "WARN"


def test_audit_detects_missing_filepath(tmp_path: Path) -> None:
    """CSV 的 filepath 指向不存在的文件 → [FAIL]。"""
    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    _write_real_main(results_dir, [ROW_PLAIN, ROW_BN])
    # 数据目录存在但缺该矩阵文件
    (data_dir / "512x16").mkdir(parents=True)

    _, checks = audit.run_audit(results_dir, data_dir, tmp_path)
    assert _status_of(checks, "filepath_exists", "real_main.csv") == "FAIL"
