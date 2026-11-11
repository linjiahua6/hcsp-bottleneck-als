"""真实 HCSP 数据格式适配的单元测试。

覆盖：官方格式（头部 n m / n m id + 每行一个数值）自动检测、
C 序 reshape、通用格式回退、.sol 解文件解析、文件名最优值解析、
以及"解文件 makespan == 文件名最优值"的端到端锚定。

不依赖 data/ 的测试全部用 tmp_path 构造等价合成格式文件；
依赖真实文件的测试在 data/hcsp 不存在时自动 skip。
"""

from pathlib import Path

import numpy as np
import pytest

from src.loader import (
    load_hcsp_matrix,
    load_hcsp_solution,
    parse_optimal_from_filename,
)

REAL_DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "hcsp"


def _write_official_matrix(path: Path, n: int, m: int, values,
                           with_id: bool = True):
    """按官方格式写矩阵：头部 + 每行一个数值（行主序）。"""
    lines = [f"{n} {m} 0" if with_id else f"{n} {m}"]
    for v in np.asarray(values, dtype=float).reshape(-1):
        lines.append(f"{v:g}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------- 官方格式自动检测 ----------


def test_real_format_512():
    """512x16 真实文件：头部 3 token（n m 0），每行一个数值。"""
    f = REAL_DATA_ROOT / "512x16" / "u_c_hihi.0"
    if not f.is_file():
        pytest.skip("真实 HCSP 数据未下载")
    P = load_hcsp_matrix(str(f))
    assert P.shape == (512, 16)
    assert P.dtype == np.float64
    assert np.all(np.isfinite(P)) and np.all(P > 0)


def test_real_format_1024():
    """1024x32 真实文件：头部 3 token（n m id，id=0/1/2 均能解析）。"""
    f = REAL_DATA_ROOT / "1024x32" / "A.u_c_hihi"
    if not f.is_file():
        pytest.skip("真实 HCSP 数据未下载")
    P = load_hcsp_matrix(str(f))
    assert P.shape == (1024, 32)
    assert np.all(np.isfinite(P)) and np.all(P > 0)


def test_real_format_2048():
    """2048x64 真实文件：头部仅 2 token（n m，无 id）。"""
    f = REAL_DATA_ROOT / "2048x64" / "A.u_c_hihi"
    if not f.is_file():
        pytest.skip("真实 HCSP 数据未下载")
    P = load_hcsp_matrix(str(f))
    assert P.shape == (2048, 64)
    assert np.all(np.isfinite(P)) and np.all(P > 0)


def test_real_reshape_c_order(tmp_path):
    """小矩阵验证官方格式按 C 序（行主序）reshape。"""
    path = tmp_path / "m.txt"
    _write_official_matrix(path, 2, 3,
                           [[1, 2, 3], [4, 5, 6]], with_id=True)
    P = load_hcsp_matrix(str(path))
    assert P.shape == (2, 3)
    assert np.allclose(P, [[1, 2, 3], [4, 5, 6]])


def test_real_format_allows_zero(tmp_path):
    """官方 benchmark 数据含 0 值执行时间（lolo/lohi 类），原样保留。"""
    path = tmp_path / "z.txt"
    _write_official_matrix(path, 2, 2, [[0, 2], [3, 4]], with_id=True)
    P = load_hcsp_matrix(str(path))
    assert P.shape == (2, 2)
    assert np.allclose(P, [[0, 2], [3, 4]])
    assert np.any(P == 0)


def test_auto_detect_falls_back(tmp_path):
    """通用格式（每行 m 个数值）不触发官方分支，仍按原逻辑解析。"""
    # 首行为整数 token 但数据行数 != n*m → 回退通用格式
    path = tmp_path / "general.txt"
    path.write_text("1 2\n3.0 4.0\n5.0 6.0\n", encoding="utf-8")
    P = load_hcsp_matrix(str(path))
    assert P.shape == (3, 2)
    assert np.allclose(P, [[1, 2], [3, 4], [5, 6]])

    # 首行含小数 token → 不触发头部解析
    path2 = tmp_path / "general2.txt"
    path2.write_text("1.0 2.5 3.0\n4.0 5.0 6.5\n", encoding="utf-8")
    P2 = load_hcsp_matrix(str(path2))
    assert P2.shape == (2, 3)
    assert np.allclose(P2, [[1.0, 2.5, 3.0], [4.0, 5.0, 6.5]])


def test_shape_mismatch_raises(tmp_path):
    """头部声明的 n*m 与实际数据行数不符 → 回退通用格式 → 行宽不一致报错。"""
    path = tmp_path / "mismatch.txt"
    path.write_text("2 3\n1\n2\n3\n4\n5\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_hcsp_matrix(str(path))


# ---------- .sol 解文件 ----------


def test_solution_0based(tmp_path):
    """解文件解析为 0-based 的 int64 机器编号数组。"""
    path = tmp_path / "x.sol"
    path.write_text("6\n0\n13\n0\n", encoding="utf-8")
    sol = load_hcsp_solution(str(path))
    assert sol.dtype == np.int64
    assert np.array_equal(sol, [6, 0, 13, 0])


def test_solution_single_line_ms_style(tmp_path):
    """MS 风格解文件：单行空格分隔全部编号，同样按 0-based 解析。"""
    path = tmp_path / "ms.sol"
    path.write_text("6 0 0 0 0 0 0 3 1 2", encoding="utf-8")
    sol = load_hcsp_solution(str(path))
    assert sol.dtype == np.int64
    assert np.array_equal(sol, [6, 0, 0, 0, 0, 0, 0, 3, 1, 2])


def test_solution_skips_blank_lines(tmp_path):
    path = tmp_path / "y.sol"
    path.write_text("0\n\n1\n\n2\n", encoding="utf-8")
    sol = load_hcsp_solution(str(path))
    assert np.array_equal(sol, [0, 1, 2])


def test_solution_invalid_raises(tmp_path):
    path = tmp_path / "bad.sol"
    path.write_text("0\nx\n2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_hcsp_solution(str(path))

    path2 = tmp_path / "neg.sol"
    path2.write_text("0\n-1\n2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_hcsp_solution(str(path2))

    path3 = tmp_path / "empty.sol"
    path3.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_hcsp_solution(str(path3))


def test_solution_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_hcsp_solution(str(tmp_path / "not_exist.sol"))


# ---------- 文件名最优值解析 ----------


def test_parse_optimal_comma_decimal():
    """7360953,7 → 7360953.7（逗号小数点）；无方法标签的变体。"""
    assert parse_optimal_from_filename("u_c_hihi.0_CPR_7360953,7.sol") == \
        pytest.approx(7360953.7)
    assert parse_optimal_from_filename("u_s_hilo.0_95553,3.sol") == \
        pytest.approx(95553.3)
    assert parse_optimal_from_filename("A.u_c_hihi_CPR_19517000,9.sol") == \
        pytest.approx(19517000.9)


def test_parse_optimal_invalid():
    """无法解析出数字时返回 None。"""
    assert parse_optimal_from_filename("no_number.sol") is None
    assert parse_optimal_from_filename("matrix_name.sol") is None
    assert parse_optimal_from_filename("abc_def_.sol") is None


# ---------- 端到端锚定：解文件 makespan == 文件名最优值 ----------


def test_solution_makespan_matches_filename(tmp_path):
    """小实例端到端：加载矩阵 + 解文件，计算 makespan 必须等于文件名值。"""
    mat_path = tmp_path / "matrix.txt"
    _write_official_matrix(mat_path, 3, 2,
                           [[1, 9], [9, 1], [5, 5]], with_id=True)
    sol_path = tmp_path / "m_6.sol"  # 文件名含最优值 6
    sol_path.write_text("0\n1\n0\n", encoding="utf-8")

    P = load_hcsp_matrix(str(mat_path))
    sol = load_hcsp_solution(str(sol_path))
    loads = np.array([P[sol == j, j].sum() for j in range(P.shape[1])])
    makespan = float(loads.max())
    assert makespan == pytest.approx(6.0)  # 节点0: 1+5=6；节点1: 1
    assert parse_optimal_from_filename(str(sol_path)) == pytest.approx(6.0)
    assert makespan == pytest.approx(
        parse_optimal_from_filename(str(sol_path)), abs=1e-6)


def test_real_sol_anchor_on_downloaded_data():
    """真实数据抽查锚定：随机 3 个矩阵的解文件 makespan 与文件名一致。"""
    f = REAL_DATA_ROOT / "512x16" / "u_c_hihi.0"
    if not f.is_file():
        pytest.skip("真实 HCSP 数据未下载")
    checked = 0
    for size, mat_name in [
        ("512x16", "u_c_hihi.0"),
        ("1024x32", "A.u_i_hihi"),
        ("2048x64", "A.u_s_lolo"),
    ]:
        sdir = REAL_DATA_ROOT / size
        mat = sdir / mat_name
        P = load_hcsp_matrix(str(mat))
        sols = sorted(sdir.glob(f"{mat_name}_*.sol"))
        assert sols, f"未找到 {mat_name} 的解文件"
        for sol in sols:
            target = parse_optimal_from_filename(str(sol))
            assert target is not None
            a = load_hcsp_solution(str(sol))
            loads = np.array([P[a == j, j].sum() for j in range(P.shape[1])])
            assert float(loads.max()) == pytest.approx(target, abs=0.1)
            checked += 1
    assert checked >= 3
