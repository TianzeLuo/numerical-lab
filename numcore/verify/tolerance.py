"""机器精度、容差与"不会撒谎"的对数运算。

容差纪律（POLICY §3，B 类断言）：容差**必须**能说清来源。本模块提供三种来源：

1. ``unit_roundoff``：IEEE-754 单位舍入误差 ``u``；
2. ``scaled_tol``：由"求和次数 n × u × 量级 scale"推出的浮点误差界（Wilkinson 口径）；
3. 调用方显式给出的解析容差（例如收敛判据、本征值界），此时应把来源写进 ``detail``。

本模块还提供 ``safe_log`` / ``safe_divide``：把定义域外的取值显式映射为 ``-inf``/``nan``，
而不是让 numpy 抛 ``RuntimeWarning``（本仓库 pytest 把 RuntimeWarning 当错误）。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "MACHINE_EPS",
    "UNIT_ROUNDOFF",
    "machine_eps",
    "unit_roundoff",
    "ulp",
    "scaled_tol",
    "machine_tol",
    "relative_error",
    "max_abs",
    "frobenius",
    "safe_log",
    "safe_log10",
    "safe_divide",
    "safe_sqrt",
    "is_all_finite",
    "count_significant_digits",
    "cancellation_digits",
]

#: binary64 能可靠表达的十进制有效位数上限（= log10(2^53) ≈ 15.95）。
_MAX_DIGITS: float = float(np.log10(2.0**53))

#: IEEE-754 binary64 的机器 epsilon（1 与下一个可表示数之差）。
MACHINE_EPS: float = float(np.finfo(np.float64).eps)
#: 单位舍入误差 u = eps/2。
UNIT_ROUNDOFF: float = MACHINE_EPS / 2.0


def machine_eps(dtype: object = np.float64) -> float:
    """返回 ``dtype`` 的机器 epsilon（1 与下一个可表示浮点数之差）。"""
    return float(np.finfo(np.dtype(dtype)).eps)


def unit_roundoff(dtype: object = np.float64) -> float:
    """返回 ``dtype`` 的单位舍入误差 ``u = eps/2``。

    在"每次算术运算的相对误差不超过 u"的模型下（Higham, *Accuracy and Stability*,
    §2.2），长度为 n 的求和误差界为 ``n·u·Σ|a_i| + O(u²)``。
    """
    return machine_eps(dtype) / 2.0


def ulp(x: float, dtype: object = np.float64) -> float:
    """返回 ``x`` 处的 ulp（相邻浮点数间距）。``0`` 处返回最小次正规数。"""
    x = float(x)
    info = np.finfo(np.dtype(dtype))
    if x == 0.0:
        return float(info.smallest_subnormal)
    return float(np.spacing(abs(x)))


def scaled_tol(scale: float, n_ops: int = 1, factor: float = 8.0, dtype: object = np.float64) -> float:
    """浮点误差界 ``factor · n_ops · u · scale``。

    参数
    ----
    scale
        问题量级（如 ``max|A|``、``‖b‖``、解的量级）。必须由调用方给出，不允许"猜"。
    n_ops
        误差累积所涉及的浮点运算次数（长度 n 的求和一般取 n）。
    factor
        安全系数，默认 8：覆盖误差常数的未知因子，同时远小于"病态放大"的量级。
    """
    u = unit_roundoff(dtype)
    s = abs(float(scale))
    if s == 0.0:
        s = float(np.finfo(np.dtype(dtype)).tiny)
    return float(factor * max(int(n_ops), 1) * u * s)


def machine_tol(scale: float, n_ops: int = 1, factor: float = 8.0) -> float:
    """:func:`scaled_tol` 的别名（语义更直白：机器精度量级的容差）。"""
    return scaled_tol(scale, n_ops=n_ops, factor=factor)


def relative_error(computed: np.ndarray | float, exact: np.ndarray | float) -> float:
    """相对误差 ``‖computed − exact‖_∞ / ‖exact‖_∞``（分母为 0 时退化为绝对误差）。"""
    c = np.asarray(computed, dtype=float)
    e = np.asarray(exact, dtype=float)
    num = max_abs(c - e)
    den = max_abs(e)
    return float(num if den == 0.0 else num / den)


def max_abs(a: np.ndarray | float) -> float:
    """``‖a‖_∞``（最大绝对值；空数组返回 0）。"""
    arr = np.asarray(a, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))


def frobenius(a: np.ndarray) -> float:
    """Frobenius 范数 ``‖A‖_F = sqrt(Σ a_ij²)``（用缩放求和避免上溢）。"""
    arr = np.asarray(a, dtype=float)
    if arr.size == 0:
        return 0.0
    m = max_abs(arr)
    if m == 0.0:
        return 0.0
    return float(m * np.sqrt(np.sum((arr / m) ** 2)))


def safe_log(x: np.ndarray | float, *, floor: float | None = None) -> np.ndarray:
    """``log(x)``：``x ≤ 0`` 显式映射为 ``-inf`` / ``nan``，不产生 RuntimeWarning。

    ``floor`` 非空时先把 ``x`` 截断到 ``≥ floor``（用于"对数色标"这类显示场景）。
    """
    arr = np.asarray(x, dtype=float)
    if floor is not None:
        arr = np.maximum(arr, float(floor))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(arr)
    return out


def safe_log10(x: np.ndarray | float, *, floor: float | None = None) -> np.ndarray:
    """``log10(x)``，语义同 :func:`safe_log`。"""
    arr = np.asarray(x, dtype=float)
    if floor is not None:
        arr = np.maximum(arr, float(floor))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log10(arr)
    return out


def safe_divide(num: np.ndarray | float, den: np.ndarray | float, *, fill: float = 0.0) -> np.ndarray:
    """逐元素除法：``den == 0`` 处填 ``fill``（默认 0），不产生 RuntimeWarning。"""
    n = np.asarray(num, dtype=float)
    d = np.asarray(den, dtype=float)
    out = np.full(np.broadcast_shapes(n.shape, d.shape), float(fill), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(n, d, out=out, where=(d != 0))
    return out


def safe_sqrt(x: np.ndarray | float) -> np.ndarray:
    """``sqrt(max(x, 0))``：负值（应为舍入噪声）截断为 0。"""
    return np.sqrt(np.maximum(np.asarray(x, dtype=float), 0.0))


def is_all_finite(x: np.ndarray | float) -> bool:
    """全部元素有限（无 nan/inf）。"""
    return bool(np.all(np.isfinite(np.asarray(x, dtype=float))))


def count_significant_digits(value: float, reference: float) -> float:
    """``value`` 与 ``reference`` 在前多少位十进制数字上一致。

    ``= −log10(|value − reference| / |reference|)``，并按 binary64 的极限封顶到 15.95 位
    （超过这个位数已经不可信了——"剩 17 位有效数字"这种话本身就是谎言）。
    完全相等时返回封顶值 15.95。
    """
    v = float(value)
    r = float(reference)
    if r == 0.0 or v == r or not np.isfinite(v) or not np.isfinite(r):
        return _MAX_DIGITS if v == r else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        d = -np.log10(abs(v - r) / abs(r))
    return float(min(max(d, 0.0), _MAX_DIGITS))


def cancellation_digits(result: float, operand_magnitude: float, dtype: object = np.float64) -> float:
    """灾难性抵消后**还剩**多少位有效数字（1.1 章核心量）。

    机理：``a − b`` 中 ``a ≈ b`` 时，结果的绝对误差仍是 ``~u·max(|a|,|b|)``，
    于是相对误差被放大为 ``u·M/|a−b|``，剩余位数

        ``d ≈ log10(|a − b| / (u·M)) = log10(|a−b|/M) + log10(1/u)``

    其中 ``M = operand_magnitude`` 是两个操作数的量级。当 ``|a−b| = M``（无抵消）时
    ``d ≈ 15.95``（binary64 的十进制极限），当 ``|a−b| = 10⁻⁸·M`` 时只剩约 8 位。
    """
    res = abs(float(result))
    m = abs(float(operand_magnitude))
    u = unit_roundoff(dtype)
    if m == 0.0:
        return _MAX_DIGITS
    if res == 0.0:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.log10(res / m) - np.log10(u)
    return float(min(max(d, 0.0), _MAX_DIGITS))
