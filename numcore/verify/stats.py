"""轻量统计工具（UQ / 蒙特卡洛章节与测试共用）。

刻意不依赖 scipy：正态分位数用 Acklam 有理逼近（|误差| < 1.2e-9），
Student-t 分位数用"小自由度查表 + 大自由度正态近似"，并**在 docstring 里写明这张表的边界**——
统计近似绝不能假装自己是精确的。
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "norm_ppf",
    "norm_cdf",
    "t_ppf",
    "mean_ci",
    "std_err",
    "effective_sample_size",
    "pearson",
    "r_squared",
    "autocorrelation",
]

# Acklam 逆正态 CDF 系数（相对误差 < 1.15e-9）
_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
_P_LOW = 0.02425
_P_HIGH = 1.0 - _P_LOW

#: 常用置信水平的 Student-t 分位数表（df = 1..30），来源：标准 t 分布表。
_T_TABLE: dict[float, tuple[float, ...]] = {
    0.90: (3.078, 1.886, 1.638, 1.533, 1.476, 1.440, 1.415, 1.397, 1.383, 1.372, 1.363, 1.356, 1.350,
           1.345, 1.341, 1.337, 1.333, 1.330, 1.328, 1.325, 1.323, 1.321, 1.319, 1.318, 1.316, 1.315,
           1.314, 1.313, 1.311, 1.310),
    0.95: (6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812, 1.796, 1.782, 1.771,
           1.761, 1.753, 1.746, 1.740, 1.734, 1.729, 1.725, 1.721, 1.717, 1.714, 1.711, 1.708, 1.706,
           1.703, 1.701, 1.699, 1.697),
    0.99: (31.821, 6.965, 4.541, 3.747, 3.365, 3.143, 2.998, 2.896, 2.821, 2.764, 2.718, 2.681, 2.650,
           2.624, 2.602, 2.583, 2.567, 2.552, 2.539, 2.528, 2.518, 2.508, 2.500, 2.492, 2.485, 2.479,
           2.473, 2.467, 2.462, 2.457),
}


def norm_cdf(x: Any) -> np.ndarray:
    """标准正态 CDF ``Φ(x) = ½(1 + erf(x/√2))``（用 ``math.erf`` 的向量化包装，精确到机器精度）。"""
    from math import erf, sqrt

    arr = np.asarray(x, dtype=float)
    flat = np.array([0.5 * (1.0 + erf(float(v) / sqrt(2.0))) for v in arr.ravel()])
    return flat.reshape(arr.shape)


def norm_ppf(p: Any) -> np.ndarray:
    """标准正态分位数 ``Φ⁻¹(p)``（Acklam 逼近，|绝对误差| < 1.2e-9）。"""
    q = np.asarray(p, dtype=float)
    out = np.empty_like(q)
    flat_q = q.ravel()
    flat_o = out.ravel()
    for i, pv in enumerate(flat_q):
        if not np.isfinite(pv) or pv < 0.0 or pv > 1.0:
            flat_o[i] = np.nan
            continue
        if pv == 0.0:
            flat_o[i] = -np.inf
            continue
        if pv == 1.0:
            flat_o[i] = np.inf
            continue
        if pv < _P_LOW:
            t = np.sqrt(-2.0 * np.log(pv))
            num = ((((_C[0] * t + _C[1]) * t + _C[2]) * t + _C[3]) * t + _C[4]) * t + _C[5]
            den = (((_D[0] * t + _D[1]) * t + _D[2]) * t + _D[3]) * t + 1.0
            flat_o[i] = num / den
        elif pv <= _P_HIGH:
            t = pv - 0.5
            r = t * t
            num = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * t
            den = ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
            flat_o[i] = num / den
        else:
            t = np.sqrt(-2.0 * np.log(1.0 - pv))
            num = ((((_C[0] * t + _C[1]) * t + _C[2]) * t + _C[3]) * t + _C[4]) * t + _C[5]
            den = (((_D[0] * t + _D[1]) * t + _D[2]) * t + _D[3]) * t + 1.0
            flat_o[i] = -num / den
    return out


def t_ppf(p: float, df: int) -> float:
    """Student-t 分位数。

    限制（必须诚实）：``df ≤ 30`` 时用查表 + 线性插值；``df > 30`` 时用正态近似
    ``t ≈ z``（df=30 处误差已 < 0.5%）。仅支持 ``p ∈ {0.90, 0.95, 0.99}`` 的**双侧**常用值，
    其他 ``p`` 抛 ``ValueError``——宁可报错，也不悄悄给一个不准确的数。
    """
    df = int(df)
    if df < 1:
        raise ValueError("df 必须 ≥ 1")
    table = None
    for key in _T_TABLE:
        if abs(p - key) < 1e-12:
            table = _T_TABLE[key]
            break
    if table is None:
        raise ValueError(f"t_ppf 仅支持 p ∈ {sorted(_T_TABLE)}，收到 p={p}")
    if df <= 30:
        return float(table[df - 1])
    return float(norm_ppf(0.5 + p / 2.0))  # 正态近似


def std_err(samples: Any) -> float:
    """样本均值的标准误 ``s/√n``（``s`` 为无偏样本标准差）。"""
    x = np.asarray(samples, dtype=float).ravel()
    n = x.size
    if n < 2:
        raise ValueError("至少需要 2 个样本")
    return float(np.std(x, ddof=1) / np.sqrt(n))


def mean_ci(samples: Any, *, level: float = 0.95) -> tuple[float, float]:
    """样本均值的置信区间 ``(均值, 半宽)``。

    ``n < 2`` 抛错；``n`` 小时用 Student-t（查表），大样本退化为正态。
    """
    x = np.asarray(samples, dtype=float).ravel()
    n = x.size
    if n < 2:
        raise ValueError("至少需要 2 个样本")
    se = std_err(x)
    if n - 1 <= 30:
        crit = t_ppf(level, n - 1)
    else:
        crit = float(norm_ppf(0.5 + level / 2.0))
    return float(np.mean(x)), float(crit * se)


def autocorrelation(x: Any, *, max_lag: int | None = None) -> np.ndarray:
    """自相关函数 ``ρ_k``（除以零滞后方差；用 ``np.correlate`` 之外的手写累加，便于控制）。"""
    a = np.asarray(x, dtype=float).ravel()
    n = a.size
    if n < 2:
        raise ValueError("样本太少")
    a = a - float(np.mean(a))
    denom = float(np.dot(a, a))
    if denom == 0.0:
        return np.zeros(1)
    m = n - 1 if max_lag is None else min(int(max_lag), n - 1)
    out = np.empty(m + 1)
    for k in range(m + 1):
        out[k] = float(np.dot(a[: n - k], a[k:])) / denom
    return out


def effective_sample_size(x: Any, *, max_lag: int | None = None) -> float:
    """有效样本量 ``N_eff = N / (1 + 2Σρ_k)``（截断在 ρ 首次转负处）。

    用于蒙特卡洛误差估计：``误差 ≈ σ/√N_eff``，比"假装样本独立"诚实得多。
    """
    a = np.asarray(x, dtype=float).ravel()
    n = a.size
    rho = autocorrelation(a, max_lag=max_lag)
    s = 0.0
    for k in range(1, rho.size):
        if rho[k] <= 0.0:
            break
        s += float(rho[k])
    return float(n / (1.0 + 2.0 * s))


def pearson(x: Any, y: Any) -> float:
    """Pearson 相关系数。"""
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    if a.size != b.size or a.size < 2:
        raise ValueError("x 与 y 长度必须相同且 ≥2")
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    den = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    return 0.0 if den == 0.0 else float(np.dot(a, b) / den)


def r_squared(y_true: Any, y_pred: Any) -> float:
    """决定系数 ``R² = 1 − SS_res/SS_tot``。"""
    t = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    if t.size != p.size:
        raise ValueError("长度不一致")
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - float(np.mean(t))) ** 2))
    return 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
