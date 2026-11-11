"""loader 模块的单元测试。"""

from pathlib import Path

import numpy as np
import pytest

from src.loader import (
    generate_synthetic_matrix,
    load_hcsp_matrix,
    load_hcsp_solution,
    parse_optimal_from_filename,
)


class TestGenerateSyntheticMatrix:
    def test_shape_and_dtype(self):
        P = generate_synthetic_matrix(n=10, m=3)
        assert P.shape == (10, 3)
        assert P.dtype == np.float64

    def test_all_positive(self):
        P = generate_synthetic_matrix(n=20, m=5, seed=7)
        assert np.all(P > 0)
        assert np.all(np.isfinite(P))

    def test_reproducible_with_same_seed(self):
        P1 = generate_synthetic_matrix(n=8, m=4, seed=123)
        P2 = generate_synthetic_matrix(n=8, m=4, seed=123)
        assert np.array_equal(P1, P2)

    def test_different_seeds_differ(self):
        P1 = generate_synthetic_matrix(n=8, m=4, seed=1)
        P2 = generate_synthetic_matrix(n=8, m=4, seed=2)
        assert not np.array_equal(P1, P2)

    def test_heterogeneity_increases_spread(self):
        P_low = generate_synthetic_matrix(n=200, m=8, seed=42, heterogeneity=0.0)
        P_high = generate_synthetic_matrix(n=200, m=8, seed=42, heterogeneity=2.0)
        spread_low = float(np.ptp(P_low) / np.median(P_low))
        spread_high = float(np.ptp(P_high) / np.median(P_high))
        assert spread_high > spread_low

    @pytest.mark.parametrize("n,m,h", [(0, 3, 1.0), (5, 0, 1.0), (5, 3, -1.0)])
    def test_invalid_args(self, n, m, h):
        with pytest.raises(ValueError):
            generate_synthetic_matrix(n=n, m=m, heterogeneity=h)


class TestLoadHcspMatrix:
    @pytest.fixture
    def sample_file(self, tmp_path: Path) -> Path:
        content = "1.0 2.5 3.0\n4.0 5.0 6.5\n7.0 8.0 9.0\n"
        path = tmp_path / "hcsp_sample.txt"
        path.write_text(content, encoding="utf-8")
        return path

    def test_load_values(self, sample_file):
        P = load_hcsp_matrix(str(sample_file))
        expected = np.array(
            [[1.0, 2.5, 3.0], [4.0, 5.0, 6.5], [7.0, 8.0, 9.0]],
            dtype=np.float64,
        )
        assert P.shape == (3, 3)
        assert P.dtype == np.float64
        assert np.allclose(P, expected)

    def test_load_tab_separated(self, tmp_path):
        path = tmp_path / "tabs.txt"
        path.write_text("1\t2\t3\n4\t5\t6\n", encoding="utf-8")
        P = load_hcsp_matrix(str(path))
        assert P.shape == (2, 3)
        assert np.allclose(P, [[1, 2, 3], [4, 5, 6]])

    def test_load_skips_blank_lines(self, tmp_path):
        path = tmp_path / "blank.txt"
        path.write_text("1 2\n\n3 4\n", encoding="utf-8")
        P = load_hcsp_matrix(str(path))
        assert P.shape == (2, 2)
        assert np.allclose(P, [[1, 2], [3, 4]])

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_hcsp_matrix(str(tmp_path / "not_exist.txt"))

    def test_ragged_rows_raise(self, tmp_path):
        path = tmp_path / "ragged.txt"
        path.write_text("1 2 3\n4 5\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_hcsp_matrix(str(path))

    def test_non_numeric_raises(self, tmp_path):
        path = tmp_path / "bad.txt"
        path.write_text("1 2 3\n4 x 6\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_hcsp_matrix(str(path))

    def test_non_positive_raises(self, tmp_path):
        path = tmp_path / "neg.txt"
        path.write_text("1 2 3\n4 -5 6\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_hcsp_matrix(str(path))

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.txt"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError):
            load_hcsp_matrix(str(path))

    # ---------- 官方 HCSP 格式（自动检测） ----------

    def test_official_format_with_id(self, tmp_path):
        """首行 n m id + 每行一个数值：按行主序展开。"""
        path = tmp_path / "official_id.txt"
        path.write_text("2 3 0\n1.5\n2.5\n3.5\n4.5\n5.5\n6.5\n",
                        encoding="utf-8")
        P = load_hcsp_matrix(str(path))
        assert P.shape == (2, 3)
        assert np.allclose(P, [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])

    def test_official_format_without_id(self, tmp_path):
        """首行仅 n m（2048x64 规模的真实文件即如此）。"""
        path = tmp_path / "official_no_id.txt"
        path.write_text("2 3\n1\n2\n3\n4\n5\n6\n", encoding="utf-8")
        P = load_hcsp_matrix(str(path))
        assert P.shape == (2, 3)
        assert np.allclose(P, [[1, 2, 3], [4, 5, 6]])

    def test_official_format_row_count_mismatch_falls_back_and_raises(self,
                                                                     tmp_path):
        """首行像头部但数据行数 != n*m → 回退通用格式 → 行宽不一致报错。"""
        path = tmp_path / "mismatch.txt"
        path.write_text("2 3\n1\n2\n3\n4\n5\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_hcsp_matrix(str(path))

    def test_official_format_non_positive_raises(self, tmp_path):
        path = tmp_path / "official_neg.txt"
        path.write_text("2 2\n1\n-2\n3\n4\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_hcsp_matrix(str(path))

    def test_official_format_zero_allowed(self, tmp_path):
        """官方 benchmark 数据含 0 值执行时间（lolo/lohi 类），原样保留。"""
        path = tmp_path / "official_zero.txt"
        path.write_text("2 2\n0\n2\n3\n4\n", encoding="utf-8")
        P = load_hcsp_matrix(str(path))
        assert P.shape == (2, 2)
        assert np.allclose(P, [[0, 2], [3, 4]])
        assert np.any(P == 0)

    def test_official_format_non_numeric_raises(self, tmp_path):
        path = tmp_path / "official_bad.txt"
        path.write_text("2 2\n1\nx\n3\n4\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_hcsp_matrix(str(path))

    def test_official_format_float_header_falls_back(self, tmp_path):
        """头部含小数 → 不触发官方格式 → 按通用格式解析。"""
        path = tmp_path / "float_header.txt"
        path.write_text("2.5 3.5\n1 2\n3 4\n", encoding="utf-8")
        P = load_hcsp_matrix(str(path))
        assert P.shape == (3, 2)

    # ---------- 真实数据锚定（data/hcsp 存在时） ----------

    def test_real_hcsp_official_format_and_sol_anchor(self):
        """用 .sol 最优解文件锚定官方格式解析：计算 makespan 必须等于
        文件名中的最优值；同时核对头部声明的 (n, m) 与矩阵形状。
        解文件兼容 CPR（每行一个编号）与 MS（单行空格分隔）两种写法。
        """
        root = Path(__file__).resolve().parents[1] / "data" / "hcsp"
        if not (root / "512x16" / "u_c_hihi.0").is_file():
            pytest.skip("真实 HCSP 数据未下载")

        checked = 0
        for size in ("512x16", "1024x32", "2048x64"):
            sdir = root / size
            mats = sorted(f for f in sdir.iterdir()
                          if not f.name.endswith(".sol"))
            # 512x16 全部 12 个，其余规模抽前 4 个
            subsample = mats if size == "512x16" else mats[:4]
            for f in subsample:
                P = load_hcsp_matrix(str(f))
                n, m = (int(x) for x in
                        f.read_text().splitlines()[0].split()[:2])
                assert P.shape == (n, m), f"{f.name}: 形状 {P.shape} != ({n}, {m})"
                sols = sorted(sdir.glob(f"{f.name}_*.sol"))
                assert sols, f"未找到 {f.name} 的解文件"
                for sol in sols:
                    target = parse_optimal_from_filename(sol.name)
                    a = load_hcsp_solution(str(sol))
                    loads = np.array([P[a == j, j].sum()
                                      for j in range(P.shape[1])])
                    got = float(loads.max())
                    assert got == pytest.approx(target, rel=1e-6, abs=0.1), (
                        f"{f.name}/{sol.name}: 计算 makespan {got} "
                        f"!= 文件名最优值 {target}"
                    )
                    checked += 1
        assert checked >= 20
