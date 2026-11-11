"""HCSP 执行时间矩阵的加载与合成数据生成。

本模块提供两个入口：
- load_hcsp_matrix:         从纯文本文件加载 HCSP 执行时间矩阵；
- generate_synthetic_matrix: 生成用于实验/测试的合成矩阵。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_hcsp_matrix(filepath: str) -> np.ndarray:
    """从纯文本文件加载 HCSP 执行时间矩阵。

    支持两种格式（自动检测，公开接口不变）：

    1. 官方 HCSP 格式：首行为头部 ``n m``（部分文件含第三个整数
       instance_id），其后每行一个数值，共 n*m 行，按行主序展开为
       形状 (n, m) 的矩阵。官方 benchmark 数据中部分实例（低异质性
       lolo/lohi 类）含 0 值执行时间，属数据事实，加载时原样保留
       （要求非负：p_ij >= 0）；
    2. 通用格式（详见 README）：每行对应一个任务，共 n 行；每行 m 个
       数值（空格或制表符分隔），要求全部为正（p_ij > 0）。

    检测规则：首行恰好 2 或 3 个 token 且全部为整数、其余非空行数等于
    n*m 时，按官方格式解析；否则回退到通用格式。

    两格式共同要求：所有执行时间必须有限（非 NaN/inf）；空行会被跳过。

    Args:
        filepath: HCSP 矩阵文件路径。

    Returns:
        形状为 (n, m) 的 float64 数组 P，其中 P[i, j] 为任务 i 在节点 j 上的
        执行时间。

    Raises:
        FileNotFoundError: 文件不存在。
        OSError: 文件存在但读取失败。
        ValueError: 文件为空、各行列数不一致、存在无法解析的数值、
            非正数或非有限值（NaN/inf）。
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"HCSP 矩阵文件不存在: {filepath}")

    # 读取所有非空行，同时记录原始行号（用于错误信息）
    lines: list[tuple[int, str]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:  # 跳过空行
                    continue
                lines.append((line_no, line))
    except OSError as exc:
        raise OSError(f"读取 HCSP 矩阵文件失败: {filepath}") from exc

    if not lines:
        raise ValueError(f"文件中没有有效数据: {filepath}")

    # ---- 官方 HCSP 格式自动检测 ----
    # 首行 2-3 个整数 token，且其余非空行数 == n*m → 每行一个数值，行主序
    header = lines[0][1].split()
    n = m = None
    if 2 <= len(header) <= 3:
        try:
            n, m = int(header[0]), int(header[1])
        except ValueError:
            n = m = None
    if n is not None and n > 0 and m > 0 and len(lines) - 1 == n * m:
        values: list[float] = []
        official_ok = True
        for line_no, line in lines[1:]:
            try:
                values.append(float(line))
            except ValueError:
                official_ok = False  # 不抛异常，回退到通用格式
                break
        if official_ok:
            matrix = np.asarray(values, dtype=np.float64).reshape(n, m)
            if not np.all(np.isfinite(matrix)):
                raise ValueError("矩阵中包含 NaN 或无穷值，所有执行时间必须是有限数值。")
            if np.any(matrix < 0):
                raise ValueError("矩阵中存在负执行时间，要求所有 p_ij >= 0。")
            # 官方 benchmark 数据含 0 值执行时间（lolo/lohi 类实例），
            # 按数据事实原样保留；0 值仅表示该任务在该节点上执行时间为零。
            return matrix
        # 否则回退通用格式（宁可回退也不误判）

    # ---- 通用格式：每行一个任务，每行 m 个数值 ----
    rows: list[list[float]] = []
    for line_no, line in lines:
        tokens = line.split()
        try:
            rows.append([float(tok) for tok in tokens])
        except ValueError as exc:
            raise ValueError(
                f"第 {line_no} 行存在无法解析为数值的内容: {line!r}"
            ) from exc

    n_cols = len(rows[0])
    for i, row in enumerate(rows, start=1):
        if len(row) != n_cols:
            raise ValueError(
                f"第 {i} 行有 {len(row)} 个数值，与第 1 行的 {n_cols} 个不一致，"
                "请检查是否每行对应一个任务、每列对应一个节点。"
            )

    matrix = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("矩阵中包含 NaN 或无穷值，所有执行时间必须是有限数值。")
    if not np.all(matrix > 0):
        raise ValueError("矩阵中存在非正执行时间，要求所有 p_ij > 0。")

    return matrix


def load_hcsp_solution(filepath: str) -> np.ndarray:
    """从 .sol 文件加载分配方案（0-based 机器编号数组）。

    真实数据存在两种写法，统一按空白分隔解析整个文件：
    - CPR 解：每行一个编号，共 n 行；
    - MS 解：单行（或数行）以空白分隔的 n 个编号。

    Args:
        filepath: .sol 解文件路径。

    Returns:
        形状为 (n,) 的 int64 数组，第 i 个元素为任务 i 分配的机器编号
        （0-based，与项目 assignment 约定一致）。

    Raises:
        FileNotFoundError: 文件不存在。
        OSError: 文件存在但读取失败。
        ValueError: 文件为空、存在无法解析为整数的内容或负数编号。
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"HCSP 解文件不存在: {filepath}")

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"读取 HCSP 解文件失败: {filepath}") from exc

    tokens = text.split()
    if not tokens:
        raise ValueError(f"解文件中没有有效数据: {filepath}")

    ids: list[int] = []
    for tok in tokens:
        try:
            ids.append(int(tok))
        except ValueError as exc:
            raise ValueError(
                f"解文件中存在无法解析为整数的内容: {tok!r}"
            ) from exc

    solution = np.asarray(ids, dtype=np.int64)
    if np.any(solution < 0):
        raise ValueError("解文件中存在负数机器编号，要求 0-based 非负编号。")
    return solution


def parse_optimal_from_filename(filepath: str) -> float | None:
    """从 .sol 文件名解析最优 makespan。

    约定文件名形如 ``u_c_hihi.0_CPR_7360953,7.sol``（方法标签可选，
    如 ``u_s_hilo.0_95553,3.sol``），最后一个下划线之后的数字即最优值，
    小数分隔符可能为逗号。

    Args:
        filepath: .sol 文件路径。

    Returns:
        解析出的最优 makespan（float）；无法解析时返回 None。
    """
    stem = Path(filepath).name
    if stem.endswith(".sol"):
        stem = stem[:-4]
    tail = stem.rsplit("_", 1)[-1]
    try:
        return float(tail.replace(",", "."))
    except ValueError:
        return None


def generate_synthetic_matrix(
    n: int,
    m: int,
    seed: int = 42,
    heterogeneity: float = 1.0,
) -> np.ndarray:
    """生成合成 HCSP 执行时间矩阵（用于测试与实验）。

    构造方式：
    1. 基础执行时间 base_ij 服从对数正态分布（几何均值 20，噪声 0.2）；
    2. 任务因子 task_i 与节点速度 speed_j 均服从对数正态分布，其标准差
       sigma = 0.4 * heterogeneity；
    3. P[i, j] = base_ij * task_i / speed_j。

    因此 heterogeneity = 0 时任务间/节点间几乎没有系统性差异；
    heterogeneity 越大，矩阵中执行时间的差异越大。

    Args:
        n: 任务数，必须 > 0。
        m: 节点数，必须 > 0。
        seed: 随机种子（同一 seed 结果可复现）。
        heterogeneity: 异质性系数，必须 >= 0。

    Returns:
        形状为 (n, m) 的正 float64 矩阵。

    Raises:
        ValueError: n、m 或 heterogeneity 取值非法。
    """
    if n <= 0:
        raise ValueError(f"任务数 n 必须为正整数，实际: {n}")
    if m <= 0:
        raise ValueError(f"节点数 m 必须为正整数，实际: {m}")
    if heterogeneity < 0:
        raise ValueError(f"heterogeneity 必须 >= 0，实际: {heterogeneity}")

    rng = np.random.default_rng(seed)
    sigma = 0.4 * float(heterogeneity)

    base = rng.lognormal(mean=np.log(20.0), sigma=0.2, size=(n, m))
    task_factor = rng.lognormal(mean=0.0, sigma=sigma, size=n)
    node_speed = rng.lognormal(mean=0.0, sigma=sigma, size=m)

    matrix = base * task_factor[:, np.newaxis] / node_speed[np.newaxis, :]
    matrix = np.clip(matrix, 1e-6, None)
    return np.asarray(matrix, dtype=np.float64)
