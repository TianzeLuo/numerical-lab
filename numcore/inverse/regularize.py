r"""``numcore.inverse.regularize`` —— 5.2 章内核（一）：谱滤波、迭代正则化与全变差。

数学对象
--------
离散线性反问题 ``A x = b``（``A`` 为 ``m x n``、``m >= n``），其中 ``b = A x* + e``：
``e`` 是噪声、``x*`` 是**已知**的真解（这一章所有问题的真解都是造题时给定的，
所以"误差"是可以直接量的，不需要再找一个参考解来自证）。

全部方法都建立在 1.3 章的**同一个 SVD 口径**上（:func:`numcore.linalg.svd.jacobi_svd`，
thin 形式 ``A = U diag(s) V^T``、``s_1 >= ... >= s_n >= 0``、列符号钉死）：

==============================  ==================================================
Tikhonov ``min ||Ax-b||^2 + lam^2 ||x||^2``    滤波因子 ``f_i = s_i^2 / (s_i^2 + lam^2)``
TSVD ``min ||x||  s.t. 秩 <= k``                滤波因子 ``f_i = 1{i <= k}``（理想低通）
Landweber ``x_{k+1} = x_k + omega A^T (b - A x_k)``  滤波因子 ``f_i = 1 - (1 - omega s_i^2)^k``
TV ``min 0.5||Ax-b||^2 + lam TV(x)``            **没有**滤波因子（非线性、非对角）
==============================  ==================================================

三条纪律（这一层的"口径"，不是风格建议）

1. **``lam`` 是平方参数**：Tikhonov 写成 ``+lam^2 ||x||^2``，于是滤波因子是
   ``s_i^2/(s_i^2+lam^2)``。写成 ``+lam||x||^2`` 会让"λ 比 σ 大还是小"这句话差一个平方，
   两章之间对不上号。TSVD/图像侧把 ``lam`` 当成**截断阈值**（``s_i > lam`` 保留）。
2. **Landweber 与矩阵无关的写法必须一致**：:func:`landweber` 用谱公式、
   :func:`landweber_operator` 用真正的矩阵迭代；两者在同一问题上必须给出同一个解
   （tests 里对拍），否则"滤波因子"只是画出来的形状，不是这个方法的行为。
3. **TV 不是谱滤波**：它没有滤波因子，也不对角化。:func:`tv_filter_probe` 用
   ``b = u_i`` 逐个探测出来的"有效滤波因子"**依赖输入**——探针换一个（``b = sum u_i``）
   数就变。谁把它画成一条与 Tikhonov 并列的滤波曲线，谁就在撒谎。

POLICY 说明（P2/P3/P4/P6）：SVD / 解线性系统走 :mod:`numcore.linalg`；随机数走
:func:`numcore.utils.rng`；无 IO、无 ``print``、无全局可变状态、无缓存。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from numcore.linalg.lstsq import prescribed_spectrum, random_orthogonal
from numcore.linalg.svd import (
    SVDFit,
    golub_kahan_svd,
    jacobi_svd,
    kogbetliantz_svd,
    numerical_rank,
    spectral_condition,
)
from numcore.utils import rng

__all__ = [
    # 容器
    "SpectralProblem",
    "RegularizedSolution",
    "LandweberRun",
    "TVRun",
    "PicardDiagnostics",
    "BiasVariance",
    # 问题构造与分解
    "PICARD_DECAY",
    "picard_class",
    "factorize",
    "test_problem",
    "picard_diagnostics",
    # 空间算子（图像 / 简化 CT）
    "phantom_image",
    "spatial_blur_matrix",
    "separable_blur_forward",
    "separable_blur_adjoint",
    "separable_svd",
    "separable_filter_factors",
    "separable_tikhonov",
    "separable_tsvd",
    "separable_apply_factors",
    "radon_matrix",
    # 滤波因子
    "tikhonov_factors",
    "tsvd_factors",
    "landweber_factors",
    "effective_dof",
    "filter_summary",
    # 解
    "filtered_solution",
    "tikhonov",
    "tsvd",
    "landweber",
    "landweber_operator",
    "tv_solve",
    "tv_denoise",
    "tv_objective",
    "tv_filter_probe",
    # 误差 / 偏差-方差 / 量级
    "vector_norm",
    "relative_error_norm",
    "bias_variance_analytic",
    "bias_variance_montecarlo",
    "unregularized_amplification",
    "smallest_resolvable_sigma",
    "noise_dominated_count",
]

_U = float(np.finfo(float).eps) / 2.0  # 单位舍入 u = 2^-53

#: 离散 Picard 条件的三档：真解系数 ``alpha_i = s_i^decay``。
#: ``decay > 1`` 时 ``|u_i^T b|/s_i`` 递减（Picard 成立）；``decay <= 1`` 时不成立。
PICARD_DECAY: dict[str, float] = {
    "satisfied": 2.0,
    "borderline": 1.0,
    "violated": 0.5,
}


# --------------------------------------------------------------------------- #
# 容器
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SpectralProblem:
    """一个"真解已知、噪声水平可控"的离散不适定问题。

    字段里的 ``U`` / ``s`` / ``V`` 是 :func:`factorize` 的实测结果（不是抄参数），
    ``b_exact = A x*`` 与 ``b = b_exact + e`` 分开保存——**偏差-方差分解必须能拿到
    无噪的 b_exact**，否则偏差里混着噪声，三条曲线加起来不等于总误差。
    """

    A: np.ndarray
    b: np.ndarray
    b_exact: np.ndarray
    x_true: np.ndarray
    U: np.ndarray
    s: np.ndarray
    V: np.ndarray
    noise_abs: float = 0.0
    noise_rel: float = 0.0
    spectrum: str = "decay"
    decay: float = 2.0
    seed: int = 20261018
    detail: Mapping[str, float] = field(default_factory=dict)

    @property
    def m(self) -> int:
        return int(self.A.shape[0])

    @property
    def n(self) -> int:
        return int(self.A.shape[1])

    @property
    def b_norm(self) -> float:
        return vector_norm(self.b)

    @property
    def b_exact_norm(self) -> float:
        return vector_norm(self.b_exact)

    @property
    def x_norm(self) -> float:
        return vector_norm(self.x_true)

    @property
    def condition(self) -> float:
        return spectral_condition(self.s)

    @property
    def rank(self) -> int:
        return numerical_rank(self.s, self.m, self.n)

    def coefficients(self) -> np.ndarray:
        """``u_i^T b``（**带噪**右端项的 Picard 系数）。"""
        return np.asarray(self.U, dtype=float).T @ np.asarray(self.b, dtype=float)

    def exact_coefficients(self) -> np.ndarray:
        """``u_i^T b_exact = s_i * (v_i^T x*)``。"""
        return np.asarray(self.U, dtype=float).T @ np.asarray(self.b_exact, dtype=float)


@dataclass(frozen=True)
class RegularizedSolution:
    """一次正则化求解的结果（解 + 两个 L 曲线坐标 + 滤波因子）。"""

    x: np.ndarray
    lam: float
    method: str
    factors: np.ndarray
    residual_norm: float
    residual_rel: float
    x_norm: float
    n_iter: int = 0
    error_rel: float = float("nan")
    detail: Mapping[str, float] = field(default_factory=dict)

    @property
    def dof(self) -> float:
        """有效自由度 ``sum f_i``（TSVD 时就是截断阶）。"""
        return float(np.sum(np.asarray(self.factors, dtype=float)))


@dataclass(frozen=True)
class LandweberRun:
    """Landweber 迭代的逐步历史（半收敛要看得见，就必须逐步记录）。

    ``error_rel`` 只在给了真解时才有意义；``stop`` 说明迭代为什么停
    （``'n_iter'`` / ``'tol'`` / ``'diverged'``——``omega`` 超过 ``2/s_1^2`` 时真的会发散）。
    """

    x: np.ndarray
    n_iter: int
    omega: float
    x_norm: np.ndarray
    residual_rel: np.ndarray
    error_rel: np.ndarray
    stop: str = "n_iter"
    objective: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))


@dataclass(frozen=True)
class TVRun:
    """全变差问题的原始-对偶迭代结果（``x``、目标函数历史、可选真误差历史）。"""

    x: np.ndarray
    lam: float
    n_iter: int
    objective: np.ndarray
    error_rel: np.ndarray
    tau: float
    sigma: float
    converged_bound: float = float("nan")


@dataclass(frozen=True)
class PicardDiagnostics:
    """离散 Picard 条件的实测诊断（同图衰减对照的定量版）。

    ``slope_coeff`` 在**整段**上拟合：加了噪声之后小奇异值那一端的系数被噪声接管，
    整段斜率一定比理论值平（实测 κ=1e6、噪声 1e-3 时是 0.564 而不是 2）——
    这正是"Picard 条件只在噪声之上那一段谈得上成立"的定量版本。
    ``slope_coeff_signal`` 只拟合 ``|u_i^T b| > noise_abs`` 的信号段（理论值 = ``decay``）。
    """

    coeff: np.ndarray
    ratio: np.ndarray
    slope_coeff: float
    slope_ratio: float
    correlation: float
    n_below_noise: int
    first_below_noise: int
    slope_coeff_signal: float = float("nan")
    n_signal: int = 0
    slope_ratio_signal: float = float("nan")


@dataclass(frozen=True)
class BiasVariance:
    """偏差-方差分解（解析式与蒙特卡洛各一份，两者必须对得上）。"""

    lam: np.ndarray
    bias2: np.ndarray
    variance: np.ndarray
    mse: np.ndarray
    dof: np.ndarray
    lam_argmin: float
    mse_min: float
    bias2_mc: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    variance_mc: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    mse_mc: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    mse_mc_std: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))


# --------------------------------------------------------------------------- #
# 基本量
# --------------------------------------------------------------------------- #
def vector_norm(x: Any) -> float:
    """2 范数（自研：``sqrt(dot(x,x))``；不用任何现成范数例程）。"""
    v = np.asarray(x, dtype=float).ravel()
    return float(np.sqrt(float(np.dot(v, v))))


def relative_error_norm(x: Any, x_ref: Any) -> float:
    """相对前向误差 ``||x - x*|| / ||x*||``。"""
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(x_ref, dtype=float).ravel()
    denom = vector_norm(b)
    return vector_norm(a - b) / denom if denom > 0.0 else float("nan")


def factorize(A: Any, *, tol: float | None = None, method: str = "auto") -> SVDFit:
    """1.3 口径的 thin SVD（``A = U diag(s) V^T``、``s`` 降序、符号钉死）。

    ``method="auto"`` 先试单边 Jacobi（相对精度最好），**但带两道体检**：
    ``converged`` 标记，加上重建误差与 ``V^T V`` 的正交性实测；任一不合格就回退到
    双边（Kogbetliantz）、再不行回退到 Golub--Kahan，并把回退写进 ``detail``。

    为什么必须体检（实测，不是防御性编程）：单边 Jacobi 在**重奇异值**的对称循环矩阵上
    会停滞——24 x 24 的高斯模糊矩阵跑满 60 次扫描后 ``off_diagonal`` 仍有 ``0.939``、
    ``converged=False``，返回的奇异值全是 ``0.375947``（列范数），而真值是
    ``[1.0, 0.873, 0.873, ...]``：相对误差 **2221 倍**。若把这份 SVD 直接拿去画 L 曲线，
    整章都是假的，而图上没有任何东西会提醒你。双边 Jacobi 在同一矩阵上 11 次扫描收敛，
    与 LAPACK 的相对差 ``1.9e-13``。
    """
    first = str(method).lower()
    if first != "auto":
        if first == "jacobi":
            return jacobi_svd(A, tol=tol)
        if first in ("kogbetliantz", "kog"):
            return kogbetliantz_svd(A, tol=tol)
        if first in ("golub_kahan", "gk"):
            return golub_kahan_svd(A, tol=tol)
        raise ValueError(f"未知的 method {method!r}（auto/jacobi/kogbetliantz/golub_kahan）")
    fit = jacobi_svd(A, tol=tol)
    if _svd_ok(A, fit):
        return fit
    alt = kogbetliantz_svd(A, tol=tol)
    if _svd_ok(A, alt):
        return _mark_fallback(alt, "kogbetliantz")
    gk = golub_kahan_svd(A, tol=tol)
    return _mark_fallback(gk, "golub_kahan")


def _mark_fallback(fit: SVDFit, name: str) -> SVDFit:
    """把"这份 SVD 是回退来的"写进 ``detail``（帧与断言都要能读出这件事）。"""
    detail = dict(fit.detail)
    detail["fallback"] = 1.0
    detail["fallback_method_" + name] = 1.0
    return replace(fit, detail=detail)


def _svd_ok(A: Any, fit: SVDFit, *, rtol: float = 1e-10) -> bool:
    """SVD 体检：``converged`` + 重建误差 + ``V^T V`` 正交性。

    两条数值判据都按 ``u * ||A||_max`` 的相对口径量，避免"矩阵整体放大 1e6 倍就通不过"
    这种与算法无关的失败。
    """
    M = np.asarray(A, dtype=float)
    if not bool(fit.converged):
        return False
    scale = float(np.max(np.abs(M))) if M.size else 1.0
    scale = scale if scale > 0.0 else 1.0
    recon = float(np.max(np.abs(M - fit.reconstruct()))) / scale
    V = np.asarray(fit.V, dtype=float)
    orth = float(np.max(np.abs(V.T @ V - np.eye(V.shape[1])))) if V.size else 0.0
    U = np.asarray(fit.U, dtype=float)
    orth_u = float(np.max(np.abs(U.T @ U - np.eye(U.shape[1])))) if U.size else 0.0
    return bool(recon <= float(rtol) and orth <= float(rtol) and orth_u <= float(rtol))


def picard_class(decay: float) -> str:
    """由衰减指数给 Picard 档位名（``>1`` 成立、``=1`` 临界、``<1`` 不成立）。"""
    d = float(decay)
    if d > 1.0 + 1e-12:
        return "satisfied"
    if d < 1.0 - 1e-12:
        return "violated"
    return "borderline"


# --------------------------------------------------------------------------- #
# 造题
# --------------------------------------------------------------------------- #
def test_problem(
    m: int,
    n: int,
    kappa: float,
    *,
    spectrum: str = "decay",
    decay: float = 2.0,
    picard: str | None = None,
    noise_rel: float = 0.0,
    seed: int = 20261018,
) -> SpectralProblem:
    """构造"真解已知、条件数恰好为 ``kappa``"的离散不适定问题。

    ``A = U diag(s) V^T``：``U``（``m x n``）、``V``（``n x n``）列正交，``s`` 由
    :func:`numcore.linalg.lstsq.prescribed_spectrum` 给出（``s_1/s_n = kappa``）。
    真解系数取 ``v_i^T x* = s_i^(decay-1)``，于是 ``u_i^T b_exact = s_i^decay``：

    * ``decay = 2``（``picard='satisfied'``）：``|u_i^T b|`` 与 ``s_i^2`` 同步衰减，
      ``|u_i^T b|/s_i`` 随 ``i`` 递减——**离散 Picard 条件成立**，问题是"可正则化的"；
    * ``decay = 1``（``'borderline'``）：比值 ``|u_i^T b|/s_i`` 持平，截断与否都差不多；
    * ``decay = 0.5``（``'violated'``）：比值随 ``i`` **增长**，最小二乘解在噪声下
      必然被小奇异值方向吞掉，且正则化只能"少错一点"，不能恢复真解。

    **噪声口径**（与 1.3 的 ``synthetic_truncation_problem`` 不同，这里写清楚）：
    ``noise_rel`` 是**实际的相对噪声范数** ``||e||/||b_exact||``——噪声向量先归一化再乘。
    1.3 那边 ``noise_rel`` 是逐分量幅度，实际相对噪声是 ``noise_rel * sqrt(m)``；
    两边混用会把"噪声水平"读错 ``sqrt(m)`` 倍（实测 m=32 时 5.66 倍）。
    """
    if picard is not None:
        if picard not in PICARD_DECAY:
            raise ValueError(f"未知的 picard 档位 {picard!r}（可选 {sorted(PICARD_DECAY)}）")
        decay = PICARD_DECAY[picard]
    U = random_orthogonal(int(m), int(n), seed=int(seed))
    V = random_orthogonal(int(n), int(n), seed=int(seed) + 977)
    s = prescribed_spectrum(int(n), float(kappa), spectrum)
    alpha = s ** float(decay)
    b_exact = U @ alpha
    x_true = V @ (alpha / s)
    A = (U * s) @ V.T
    noise_rel = float(max(noise_rel, 0.0))
    if noise_rel > 0.0:
        g = rng(int(seed) + 4241).standard_normal(int(m))
        e = float(noise_rel) * vector_norm(b_exact) * (g / vector_norm(g))
    else:
        e = np.zeros(int(m), dtype=float)
    b = b_exact + e
    return SpectralProblem(
        A=A, b=b, b_exact=b_exact, x_true=x_true, U=U, s=s, V=V,
        noise_abs=vector_norm(e), noise_rel=noise_rel, spectrum=spectrum,
        decay=float(decay), seed=int(seed),
        detail={
            "kappa_given": float(kappa),
            "kappa_measured": spectral_condition(s),
            "b_exact_norm": vector_norm(b_exact),
            "x_true_norm": vector_norm(x_true),
        },
    )


def picard_diagnostics(
    U: Any,
    s: Any,
    b: Any,
    *,
    noise_abs: float | None = None,
) -> PicardDiagnostics:
    """实测离散 Picard 条件：``|u_i^T b|`` 的衰减斜率、比值斜率、相关系数。

    * ``slope_coeff``：``log|u_i^T b|`` 对 ``log s_i`` 的最小二乘斜率（理论值 = ``decay``）；
    * ``slope_ratio``：``log(|u_i^T b|/s_i)`` 对 ``log s_i`` 的斜率（``= slope_coeff - 1``）；
      Picard 条件成立当且仅当它 **< 0**；
    * ``correlation``：``log s_i`` 与 ``log|u_i^T b|`` 的皮尔逊相关系数（"同图衰减"的定量版）；
    * ``n_below_noise`` / ``first_below_noise``：有多少个（第一个）系数已经落到噪声水平
      ``noise_abs`` 之下——从那里开始，``u_i^T b`` 里剩下的全是噪声，
      "解"的那些分量没有任何信息，只有放大后的噪声。
    """
    Uu = np.asarray(U, dtype=float)
    sv = np.asarray(s, dtype=float).ravel()
    bb = np.asarray(b, dtype=float).ravel()
    coeff = np.abs(Uu.T @ bb)
    safe = np.where(sv > 0.0, sv, np.inf)
    ratio = coeff / safe
    ls = np.log(np.maximum(sv, np.finfo(float).tiny))
    lc = np.log(np.maximum(coeff, np.finfo(float).tiny))
    slope_c = _ols_slope(ls, lc)
    slope_r = _ols_slope(ls, np.log(np.maximum(ratio, np.finfo(float).tiny)))
    corr = _pearson(ls, lc)
    if noise_abs is None or float(noise_abs) <= 0.0:
        below = np.zeros(coeff.size, dtype=bool)
        n_below = 0
        first = -1
        signal = np.ones(coeff.size, dtype=bool)
    else:
        below = coeff < float(noise_abs)
        n_below = int(np.count_nonzero(below))
        first = int(np.argmax(below)) if n_below > 0 else -1
        signal = ~below
    n_sig = int(np.count_nonzero(signal))
    if n_sig >= 3:
        slope_sig = _ols_slope(ls[signal], lc[signal])
        slope_ratio_sig = _ols_slope(ls[signal], np.log(np.maximum(ratio[signal], np.finfo(float).tiny)))
    else:
        slope_sig = float("nan")
        slope_ratio_sig = float("nan")
    return PicardDiagnostics(
        coeff=coeff, ratio=ratio, slope_coeff=slope_c, slope_ratio=slope_r,
        correlation=corr, n_below_noise=n_below, first_below_noise=first,
        slope_coeff_signal=slope_sig, n_signal=n_sig, slope_ratio_signal=slope_ratio_sig,
    )


def _ols_slope(x: Any, y: Any) -> float:
    """``y = a + b x`` 的最小二乘斜率（自研；P3 禁用现成相关系数/协方差）。"""
    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    good = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[good], yv[good]
    if xv.size < 2:
        return float("nan")
    xm = float(np.mean(xv))
    ym = float(np.mean(yv))
    sxx = float(np.sum((xv - xm) ** 2))
    if sxx == 0.0:
        return float("nan")
    return float(np.sum((xv - xm) * (yv - ym)) / sxx)


def _pearson(x: Any, y: Any) -> float:
    """皮尔逊相关系数（自研，避免 ``np.corrcoef``；P3）。"""
    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    xm = xv - float(np.mean(xv))
    ym = yv - float(np.mean(yv))
    denom = float(np.sqrt(float(np.dot(xm, xm)) * float(np.dot(ym, ym))))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(xm, ym) / denom)


# --------------------------------------------------------------------------- #
# 滤波因子
# --------------------------------------------------------------------------- #
def tikhonov_factors(s: Any, lam: float) -> np.ndarray:
    """``f_i = s_i^2/(s_i^2 + lam^2)``（``lam = 0`` 时全 1，即最小二乘解）。"""
    sv = np.asarray(s, dtype=float).ravel()
    l2 = float(lam) * float(lam)
    return (sv * sv) / (sv * sv + l2)


def tsvd_factors(s: Any, k: int) -> np.ndarray:
    """``f_i = 1{i <= k}``：理想低通——要么全留、要么全扔，没有过渡带。"""
    sv = np.asarray(s, dtype=float).ravel()
    kk = int(np.clip(int(k), 0, sv.size))
    f = np.zeros(sv.size, dtype=float)
    f[:kk] = 1.0
    return f


def landweber_factors(s: Any, n_iter: int, omega: float | None = None) -> np.ndarray:
    """``f_i = 1 - (1 - omega s_i^2)^k``；缺省 ``omega = 1/s_1^2``（收敛最快的那一档）。

    ``omega >= 2/s_1^2`` 时 ``|1 - omega s_1^2| > 1``：**迭代发散**，滤波因子不再落在
    ``[0,1]`` 里——这不是"滤波形状不同"，是方法失效（帧与断言都要把这条线画出来）。
    """
    sv = np.asarray(s, dtype=float).ravel()
    if omega is None:
        omega = 1.0 / float(sv[0] ** 2) if sv.size and sv[0] > 0.0 else 0.0
    r = 1.0 - float(omega) * sv * sv
    return 1.0 - r ** int(n_iter)


def effective_dof(factors: Any) -> float:
    """有效自由度 ``sum_i f_i``（TSVD 的 ``k`` 是它的特例）。"""
    return float(np.sum(np.asarray(factors, dtype=float).ravel()))


def filter_summary(s: Any, lam: float, *, k_tsvd: int | None = None, n_iter: int = 30, omega: float | None = None) -> dict[str, float]:
    """三种方法的滤波因子摘要（自由度、最大/最小因子、半功率点下标）。

    "半功率点"（``f_i = 1/2``）是**同一个尺度上**比较三种曲线的地方：
    Tikhonov 在 ``s_i = lam`` 处正好 1/2，TSVD 在截断处从 1 跳到 0（没有 1/2 这个点），
    Landweber 的过渡带宽度 ~ ``1/sqrt(k)``。
    """
    sv = np.asarray(s, dtype=float).ravel()
    ft = tikhonov_factors(sv, lam)
    kk = int(np.count_nonzero(sv > float(lam))) if k_tsvd is None else int(k_tsvd)
    fs = tsvd_factors(sv, kk)
    fl = landweber_factors(sv, n_iter, omega)
    half_t = int(np.argmin(np.abs(ft - 0.5))) if ft.size else -1
    half_l = int(np.argmin(np.abs(fl - 0.5))) if fl.size else -1
    return {
        "lam": float(lam),
        "dof_tikhonov": effective_dof(ft),
        "dof_tsvd": effective_dof(fs),
        "dof_landweber": effective_dof(fl),
        "k_tsvd": float(kk),
        "half_power_index_tikhonov": float(half_t),
        "half_power_index_landweber": float(half_l),
        "landweber_max_factor": float(np.max(fl)) if fl.size else float("nan"),
    }


# --------------------------------------------------------------------------- #
# 解（谱滤波）
# --------------------------------------------------------------------------- #
def filtered_solution(
    U: Any,
    s: Any,
    V: Any,
    b: Any,
    factors: Any,
    *,
    method: str = "",
    lam: float = float("nan"),
    n_iter: int = 0,
    x_true: Any | None = None,
    b_norm: float | None = None,
) -> RegularizedSolution:
    """由滤波因子组装解 ``x = sum_i f_i (u_i^T b / s_i) v_i``。

    ``s_i = 0`` 的方向按"截断"处理（``1/s_i`` 不存在，滤波因子再大也没有意义）。
    """
    Uu = np.asarray(U, dtype=float)
    sv = np.asarray(s, dtype=float).ravel()
    Vv = np.asarray(V, dtype=float)
    bb = np.asarray(b, dtype=float).ravel()
    f = np.asarray(factors, dtype=float).ravel()
    if f.size != sv.size:
        raise ValueError(f"滤波因子长度 {f.size} 与奇异值个数 {sv.size} 不一致")
    coeff = Uu.T @ bb
    safe = np.where(sv > 0.0, sv, 1.0)
    scale = np.where(sv > 0.0, f * coeff / safe, 0.0)
    x = Vv @ scale
    # 残差用 A x = U (s * scale) 直接算（不显式形成 A）：m x n 的 A 只在小问题上才存得下
    r = bb - Uu @ (sv * scale)
    rn = vector_norm(r)
    bn = vector_norm(bb) if b_norm is None else float(b_norm)
    err = relative_error_norm(x, x_true) if x_true is not None else float("nan")
    return RegularizedSolution(
        x=x, lam=float(lam), method=method or "filtered", factors=f.copy(),
        residual_norm=rn, residual_rel=(rn / bn if bn > 0.0 else float("nan")),
        x_norm=vector_norm(x), n_iter=int(n_iter), error_rel=err,
    )


def _A_from_factors(U: np.ndarray, s: np.ndarray, V: np.ndarray) -> np.ndarray:
    """由 SVD 因子还原 ``A``（只在需要显式 ``A`` 的地方用；不参与求解路径）。"""
    return (U * s) @ V.T


def tikhonov(U: Any, s: Any, V: Any, b: Any, lam: float, *, x_true: Any | None = None) -> RegularizedSolution:
    """Tikhonov 解（``f_i = s_i^2/(s_i^2+lam^2)``）。"""
    f = tikhonov_factors(s, lam)
    return filtered_solution(U, s, V, b, f, method="tikhonov", lam=float(lam), x_true=x_true)


def tsvd(U: Any, s: Any, V: Any, b: Any, k: int, *, x_true: Any | None = None) -> RegularizedSolution:
    """截断 SVD 解（保留前 ``k`` 个奇异方向；``k = 0`` 给零向量）。"""
    f = tsvd_factors(s, k)
    return filtered_solution(U, s, V, b, f, method="tsvd", lam=float(k), n_iter=int(k), x_true=x_true)


def landweber(
    U: Any,
    s: Any,
    V: Any,
    b: Any,
    *,
    n_iter: int = 30,
    omega: float | None = None,
    x_true: Any | None = None,
) -> RegularizedSolution:
    """谱形式的 Landweber 解（与 :func:`landweber_operator` 同解，见 tests 对拍）。"""
    f = landweber_factors(s, n_iter, omega)
    om = float(1.0 / float(np.asarray(s, dtype=float).ravel()[0] ** 2)) if omega is None else float(omega)
    return filtered_solution(
        U, s, V, b, f, method="landweber", lam=om, n_iter=int(n_iter), x_true=x_true
    )


# --------------------------------------------------------------------------- #
# 迭代正则化（与矩阵无关的写法）
# --------------------------------------------------------------------------- #
def landweber_operator(
    apply_A: Callable[[np.ndarray], np.ndarray],
    apply_AT: Callable[[np.ndarray], np.ndarray],
    b: Any,
    x0: Any,
    *,
    n_iter: int,
    omega: float,
    x_true: Any | None = None,
    record_every: int = 1,
    tol: float = 0.0,
) -> LandweberRun:
    """矩阵无关的 Landweber 迭代 ``x_{k+1} = x_k + omega A^T(b - A x_k)``。

    逐步记录 ``||x_k||``、相对残差与（可选）相对真误差：**半收敛**——误差先降后升——
    只有逐步历史才看得见，而它正是"迭代正则化必须早停"的全部证据。

    ``omega`` 必须落在 ``(0, 2/s_1^2)``（``s_1`` 是 ``A`` 的最大奇异值）内才收敛；
    超出时迭代值会指数增长，``stop`` 记成 ``'diverged'``（判据：出现非有限值，
    或 ``||x_k||`` 超过初始值的 1e12 倍）。
    """
    bb = np.asarray(b, dtype=float)
    x = np.array(x0, dtype=float)
    step = int(max(1, record_every))
    x_norms = [vector_norm(x)]
    res = [vector_norm(bb - apply_A(x)) / (vector_norm(bb) or 1.0)]
    errs = [relative_error_norm(x, x_true) if x_true is not None else float("nan")]
    stop = "n_iter"
    # 发散判据的标尺：以初始迭代（或 1，取大者）为基准放大 1e12 倍。
    # 不能用 ||x0|| 单独当基准——x0 = 0 时基准是 0，第一步就会被误判成发散
    # （这个 bug 在开发时真的出现过：landweber 迭代 1 步就报 diverged，误差 0.41）。
    limit = 1e12 * max(vector_norm(x), 1.0)
    for k in range(1, int(n_iter) + 1):
        r = bb - apply_A(x)
        x = x + float(omega) * apply_AT(r)
        xk = vector_norm(x)
        if not np.isfinite(xk) or xk > limit:
            stop = "diverged"
            x_norms.append(float(min(xk, limit)) if np.isfinite(xk) else float(limit))
            res.append(res[-1])
            errs.append(errs[-1])
            break
        if k % step == 0 or k == int(n_iter):
            x_norms.append(xk)
            res.append(vector_norm(bb - apply_A(x)) / (vector_norm(bb) or 1.0))
            errs.append(relative_error_norm(x, x_true) if x_true is not None else float("nan"))
        if tol > 0.0 and res[-1] <= tol:
            stop = "tol"
            break
    return LandweberRun(
        x=x, n_iter=int(n_iter), omega=float(omega),
        x_norm=np.asarray(x_norms, dtype=float),
        residual_rel=np.asarray(res, dtype=float),
        error_rel=np.asarray(errs, dtype=float),
        stop=stop,
    )


# --------------------------------------------------------------------------- #
# 全变差：原始-对偶（Chambolle-Pock），不做任何"谱滤波"假设
# --------------------------------------------------------------------------- #
def _forward_diff(x: np.ndarray) -> np.ndarray:
    """前向差分（Neumann 边界：最后一个位置差分为 0）；1 维给 ``(1, n)``、2 维给 ``(2, h, w)``。"""
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        d = np.zeros((1, a.size), dtype=float)
        if a.size > 1:
            d[0, :-1] = a[1:] - a[:-1]
        return d
    if a.ndim == 2:
        h, w = a.shape
        d = np.zeros((2, h, w), dtype=float)
        if h > 1:
            d[0, :-1, :] = a[1:, :] - a[:-1, :]
        if w > 1:
            d[1, :, :-1] = a[:, 1:] - a[:, :-1]
        return d
    raise ValueError(f"只支持 1 维/2 维信号，得到 {a.ndim} 维")


def _divergence(p: np.ndarray) -> np.ndarray:
    """``grad^T p``（``grad`` 的**精确**伴随，不是它的相反数）。

    ``<grad x, p> = <x, grad^T p>`` 必须逐位成立，否则原始-对偶迭代在解一个
    与目标函数不同的对偶问题——收敛曲线很好看，解是错的。
    """
    q = np.asarray(p, dtype=float)
    if q.ndim == 2:  # 1 维信号：p 形状 (1, n)
        row = q[0]
        out = np.zeros(row.size, dtype=float)
        out[:-1] -= row[:-1]
        out[1:] += row[:-1]
        return out
    if q.ndim == 3:  # 2 维信号：p 形状 (2, h, w)
        h, w = q.shape[1], q.shape[2]
        out = np.zeros((h, w), dtype=float)
        p0 = q[0]
        out[:-1, :] -= p0[:-1, :]
        out[1:, :] += p0[:-1, :]
        p1 = q[1]
        out[:, :-1] -= p1[:, :-1]
        out[:, 1:] += p1[:, :-1]
        return out
    raise ValueError(f"只支持 1 维/2 维对偶变量，得到 {q.ndim} 维")


def _tv_value(x: np.ndarray) -> float:
    """各向同性全变差 ``sum ||(grad x)_p||_2``。"""
    d = _forward_diff(x)
    if d.ndim == 2:
        return float(np.sum(np.abs(d[0])))
    return float(np.sum(np.sqrt(d[0] ** 2 + d[1] ** 2)))


def tv_solve(
    apply_A: Callable[[np.ndarray], np.ndarray],
    apply_AT: Callable[[np.ndarray], np.ndarray],
    b: Any,
    x0: Any,
    lam: float,
    *,
    n_iter: int = 200,
    norm_A: float | None = None,
    tau: float | None = None,
    sigma: float | None = None,
    x_true: Any | None = None,
    record: bool = True,
) -> TVRun:
    """``min_x 0.5||Ax-b||^2 + lam TV(x)`` 的原始-对偶解（Chambolle-Pock，自研）。

    对偶变量 ``p1``（数据项）、``p2``（全变差项），``K = [A; grad]``：

    * ``p1 <- (p1 + sigma (A xbar - b)) / (1 + sigma)``
    * ``p2 <- clip(p2 + sigma grad(xbar), -lam, lam)``
    * ``x <- x - tau (A^T p1 + grad^T p2)``、``xbar <- 2x_new - x``

    步长取 ``tau = sigma = 0.99/sqrt(||A||^2 + ||grad||^2)``（``||grad||^2 <= 8`` 二维、
    ``<= 4`` 一维）。``lam = 0`` 时退化为纯最小二乘（全变差项不起作用），
    这一点被单测钉住——否则"TV 的参数拖到 0"会变成另一个算法。
    """
    bb = np.asarray(b, dtype=float)
    x = np.array(x0, dtype=float)
    lam = float(max(lam, 0.0))
    norm_a = 1.0 if norm_A is None else float(norm_A)
    grad_sq = 8.0 if x.ndim == 2 else 4.0
    big_l = float(np.sqrt(norm_a * norm_a + grad_sq))
    if tau is None or sigma is None:
        step = 0.99 / big_l
        tau = step if tau is None else float(tau)
        sigma = step if sigma is None else float(sigma)
    else:
        tau, sigma = float(tau), float(sigma)
    p1 = np.zeros_like(bb)
    p2 = np.zeros_like(_forward_diff(x))
    obj = np.zeros(1 if record else 0, dtype=float)
    err = np.zeros(1 if record else 0, dtype=float)
    if record:
        obj[0] = tv_objective(apply_A, x, bb, lam)
        err[0] = relative_error_norm(x, x_true) if x_true is not None else float("nan")
    xbar = x.copy()
    hist: list[float] = []
    ehist: list[float] = []
    for _ in range(int(n_iter)):
        p1 = (p1 + sigma * (apply_A(xbar) - bb)) / (1.0 + sigma)
        p2 = np.clip(p2 + sigma * _forward_diff(xbar), -lam, lam)
        x_new = x - tau * (apply_AT(p1) + _divergence(p2))
        xbar = 2.0 * x_new - x
        x = x_new
        if record:
            hist.append(tv_objective(apply_A, x, bb, lam))
            ehist.append(relative_error_norm(x, x_true) if x_true is not None else float("nan"))
    if record:
        obj = np.concatenate([obj, np.asarray(hist, dtype=float)])
        err = np.concatenate([err, np.asarray(ehist, dtype=float)])
    return TVRun(x=x, lam=lam, n_iter=int(n_iter), objective=obj, error_rel=err, tau=tau, sigma=sigma)


def tv_objective(apply_A: Callable[[np.ndarray], np.ndarray], x: np.ndarray, b: np.ndarray, lam: float) -> float:
    """``0.5||Ax-b||^2 + lam TV(x)``（原始目标函数；对偶间隙的另一半在 tests 里单独量）。"""
    r = apply_A(x) - b
    return 0.5 * float(np.dot(np.asarray(r, dtype=float).ravel(), np.asarray(r, dtype=float).ravel())) + lam * _tv_value(x)


def tv_denoise(b: Any, lam: float, *, n_iter: int = 200) -> TVRun:
    """``A = I`` 的 TV 去噪（``min 0.5||x-b||^2 + lam TV(x)``）。

    一维两点信号有**闭式解**（差值的软阈值）：``x = (m - v, m + v)``，
    ``m = (b_1+b_2)/2``、``v = soft((b_2-b_1)/2, lam)``——单测拿它当 A 类靶子。
    """
    bb = np.asarray(b, dtype=float)
    return tv_solve(lambda z: z, lambda z: z, bb, bb, lam, n_iter=n_iter, norm_A=1.0)


def tv_filter_probe(
    A: Any,
    U: Any,
    s: Any,
    V: Any,
    lam: float,
    *,
    probe_index: Sequence[int] | None = None,
    n_iter: int = 200,
) -> dict[str, Any]:
    """用 ``b = u_i`` 逐个探测 TV 的"有效滤波因子"（**并证明它不是滤波因子**）。

    对 ``b = u_i``（单位数据）探测：``f_i^TV = s_i <v_i, x>``——若 TV 与 Tikhonov 一样
    是谱滤波，这个数只该由 ``i`` 决定、且非对角项为 0。实测恰好相反：探针换一个
    （``b = u_1 + u_n``）同一个 ``i`` 上的读数就变，非对角泄漏也不为 0
    （见 ``leakage_*``）。所以 TV 只能画成"某一次探测下的一条曲线"，不能当成滤波器。
    """
    M = np.asarray(A, dtype=float)
    Uu = np.asarray(U, dtype=float)
    sv = np.asarray(s, dtype=float).ravel()
    Vv = np.asarray(V, dtype=float)
    idx = list(range(sv.size)) if probe_index is None else [int(i) for i in probe_index]
    diag = np.full(sv.size, np.nan, dtype=float)
    leakage = np.full(sv.size, np.nan, dtype=float)
    for i in idx:
        b = Uu[:, i]
        run = tv_solve(lambda z: M @ z, lambda z: M.T @ z, b, np.zeros(M.shape[1]), lam, n_iter=n_iter, norm_A=float(sv[0]))
        coeff = Vv.T @ run.x
        diag[i] = float(sv[i] * coeff[i])
        off = np.abs(coeff) * sv
        off[i] = 0.0
        leakage[i] = float(np.max(off))
    # 混合探针：b = (u_1 + u_n)/sqrt(2)，同一批 i 上的读数与单探针的差就是"依赖性"的证据
    mix = (Uu[:, 0] + Uu[:, -1]) / float(np.sqrt(2.0))
    run_mix = tv_solve(lambda z: M @ z, lambda z: M.T @ z, mix, np.zeros(M.shape[1]), lam, n_iter=n_iter, norm_A=float(sv[0]))
    coeff_mix = Vv.T @ run_mix.x
    diag_mix = sv * coeff_mix
    return {
        "probe_index": np.asarray(idx, dtype=float),
        "filter_diag": diag,
        "filter_diag_mixed": diag_mix,
        "leakage": leakage,
        "max_probe_dependence": float(np.max(np.abs(diag_mix[idx] - diag[idx]))) if len(idx) else 0.0,
    }


# --------------------------------------------------------------------------- #
# 偏差-方差分解
# --------------------------------------------------------------------------- #
def _noise_component_variance(noise_abs: float, m: int) -> float:
    """白噪声总范数为 ``noise_abs`` 时**每个分量**的方差 ``noise_abs^2/m``。"""
    return float(noise_abs) ** 2 / float(m)


def bias_variance_analytic(
    U: Any,
    s: Any,
    V: Any,
    b_exact: Any,
    x_true: Any,
    noise_abs: float,
    lam: np.ndarray,
) -> BiasVariance:
    """解析的偏差-方差分解（对 Tikhonov 族，逐位可验）。

    * 偏差：``E[x_lam] - x*``，其中 ``E[x_lam] = V diag(f_i/s_i) U^T b_exact``；
      偏差**只与无噪右端项有关**——把带噪的 ``b`` 代进去，量到的"偏差"里混了噪声；
    * 方差：``sigma_e^2 sum_i (f_i/s_i)^2``（``sigma_e^2 = noise_abs^2/m``）；
    * 总和 ``bias^2 + variance = E||x_lam - x*||^2``，即**期望总误差**。
    """
    Uu = np.asarray(U, dtype=float)
    sv = np.asarray(s, dtype=float).ravel()
    Vv = np.asarray(V, dtype=float)
    be = np.asarray(b_exact, dtype=float).ravel()
    xt = np.asarray(x_true, dtype=float).ravel()
    m = int(Uu.shape[0])
    lam_arr = np.asarray(lam, dtype=float).ravel()
    alpha = Uu.T @ be
    xt_coef = Vv.T @ xt
    var_e = _noise_component_variance(float(noise_abs), m)
    bias2 = np.zeros(lam_arr.size, dtype=float)
    variance = np.zeros(lam_arr.size, dtype=float)
    dof = np.zeros(lam_arr.size, dtype=float)
    for j, l in enumerate(lam_arr):
        f = tikhonov_factors(sv, l)
        coef = f * alpha / np.where(sv > 0.0, sv, 1.0)
        bias2[j] = float(np.sum((coef - xt_coef) ** 2))
        variance[j] = var_e * float(np.sum((f / np.where(sv > 0.0, sv, 1.0)) ** 2))
        dof[j] = float(np.sum(f))
    mse = bias2 + variance
    best = int(np.argmin(mse))
    return BiasVariance(
        lam=lam_arr, bias2=bias2, variance=variance, mse=mse, dof=dof,
        lam_argmin=float(lam_arr[best]), mse_min=float(mse[best]),
    )


def bias_variance_montecarlo(
    U: Any,
    s: Any,
    V: Any,
    b_exact: Any,
    x_true: Any,
    noise_abs: float,
    lam: np.ndarray,
    *,
    n_trials: int = 64,
    seed: int = 20261018,
) -> BiasVariance:
    """蒙特卡洛版的三条曲线：**实测**偏差²、方差、总误差（与解析式对拍）。

    噪声按 ``N(0, noise_abs^2/m)`` 逐分量抽（P4：走 :func:`numcore.utils.rng`）。
    同一批 ``n_trials`` 条噪声同时喂给整张 ``lam`` 网格——解在 SVD 坐标里是
    ``V diag(f_i/s_i) U^T b``，所以 ``n_trials`` 次试验只要一次矩阵乘法，
    这正是"知道 SVD 就能把实验做便宜"的地方。
    """
    Uu = np.asarray(U, dtype=float)
    sv = np.asarray(s, dtype=float).ravel()
    Vv = np.asarray(V, dtype=float)
    be = np.asarray(b_exact, dtype=float).ravel()
    xt = np.asarray(x_true, dtype=float).ravel()
    m = int(Uu.shape[0])
    lam_arr = np.asarray(lam, dtype=float).ravel()
    n_lam = lam_arr.size
    sd = float(np.sqrt(_noise_component_variance(float(noise_abs), m)))
    # 先算 E[x_lam] 的实测版（第一遍），再绕它算方差（第二遍，同一批种子重放）
    mean = np.zeros((xt.size, n_lam), dtype=float)
    gen = rng(int(seed))
    for _ in range(int(n_trials)):
        e = sd * gen.standard_normal(m)
        coef = Uu.T @ (be + e)
        for j, l in enumerate(lam_arr):
            f = tikhonov_factors(sv, l)
            mean[:, j] += Vv @ (f * coef / np.where(sv > 0.0, sv, 1.0))
    mean /= float(n_trials)
    per_trial = np.zeros((int(n_trials), n_lam), dtype=float)
    sum_sq_dev = np.zeros(n_lam, dtype=float)
    gen = rng(int(seed))
    for t in range(int(n_trials)):
        e = sd * gen.standard_normal(m)
        coef = Uu.T @ (be + e)
        for j, l in enumerate(lam_arr):
            f = tikhonov_factors(sv, l)
            x = Vv @ (f * coef / np.where(sv > 0.0, sv, 1.0))
            d = x - mean[:, j]
            sum_sq_dev[j] += float(np.dot(d, d))
            per_trial[t, j] = float(np.dot(x - xt, x - xt))
    n = float(n_trials)
    var_mc = sum_sq_dev / n
    mse_mc = np.mean(per_trial, axis=0)
    bias2_mc = np.sum((mean - xt[:, None]) ** 2, axis=0)
    std = np.std(per_trial, axis=0, ddof=1) / np.sqrt(n) if n_trials > 1 else np.zeros(n_lam)
    best = int(np.argmin(mse_mc))
    return BiasVariance(
        lam=lam_arr, bias2=bias2_mc, variance=var_mc, mse=mse_mc,
        dof=np.asarray([effective_dof(tikhonov_factors(sv, l)) for l in lam_arr], dtype=float),
        lam_argmin=float(lam_arr[best]), mse_min=float(mse_mc[best]),
        bias2_mc=bias2_mc, variance_mc=var_mc, mse_mc=mse_mc, mse_mc_std=std,
    )


# --------------------------------------------------------------------------- #
# 量级：噪声被放大多少倍
# --------------------------------------------------------------------------- #
def unregularized_amplification(
    s: Any,
    noise_abs: float,
    *,
    m: int | None = None,
    x_norm: float = 1.0,
    worst_case: bool = True,
) -> float:
    """不做正则化时噪声被放大多少倍（相对解误差）。

    * ``worst_case=True``：``||A^+ e|| <= ||e|| / s_n``，误差上界 ``noise_abs/(s_n ||x*||)``
      ——噪声全落在最小奇异方向上时取到；
    * ``worst_case=False``：白噪声的均方根误差
      ``sigma_e sqrt(sum_i 1/s_i^2) / ||x*||``，``sigma_e^2 = noise_abs^2/m``。

    两个数差一个 ``sqrt(m)`` 量级的因子；"κ=1e6、噪声 1e-3"到底放大多少倍，
    取决于问的是哪一个——所以帧里两个都报，并写清各是哪一种。
    """
    sv = np.asarray(s, dtype=float).ravel()
    inv_sq = float(np.sum(1.0 / np.maximum(sv, np.finfo(float).tiny) ** 2))
    denom = float(x_norm) if float(x_norm) > 0.0 else 1.0
    if worst_case:
        return float(noise_abs) * float(np.sqrt(inv_sq)) / denom
    mm = int(m) if m else int(sv.size)
    return float(noise_abs) / float(np.sqrt(mm)) * float(np.sqrt(inv_sq)) / denom


def smallest_resolvable_sigma(s: Any, noise_abs: float, *, b_norm: float) -> float:
    """``sigma_res = noise_abs / ||b||``：噪声水平换算到奇异值尺度上的"可分辨下界"。

    小于它的那些方向，``|u_i^T b|`` 已经和噪声同量级，Tikhonov 的"半功率点"落在那里
    就不再是在滤噪，而是在滤信号。
    """
    return float(noise_abs) / (float(b_norm) if float(b_norm) > 0.0 else 1.0)


def noise_dominated_count(s: Any, noise_abs: float, *, b_norm: float) -> int:
    """有多少个奇异方向已经"被噪声接管"（``sigma_i < noise_abs/||b||``）。"""
    sv = np.asarray(s, dtype=float).ravel()
    thr = smallest_resolvable_sigma(sv, noise_abs, b_norm=b_norm)
    return int(np.count_nonzero(sv < thr))


# --------------------------------------------------------------------------- #
# 空间算子：图像反卷积与简化 CT
# --------------------------------------------------------------------------- #
def phantom_image(n: int) -> np.ndarray:
    """确定性小幻影：一个圆盘 + 一个方块 + 两条细棒（边缘是反问题的难点所在）。

    ``n`` 取 20--28 用于反卷积、10--12 用于简化 CT（``A`` 显式存下来，
    尺寸决定了每帧的代价）。像素值只有 0 / 0.5 / 0.75 三档：解模糊之后
    **边缘是否被恢复**是可以目测的，而灰度渐变图不行。
    """
    nn = int(n)
    y, x = np.mgrid[0:nn, 0:nn].astype(float)
    img = np.zeros((nn, nn), dtype=float)
    cy, cx, r = 0.42 * nn, 0.40 * nn, 0.24 * nn
    disk = (y - cy) ** 2 + (x - cx) ** 2 <= r * r
    img[disk] = 1.0
    sq = (y > 0.20 * nn) & (y < 0.44 * nn) & (x > 0.62 * nn) & (x < 0.86 * nn)
    img[sq] = 0.5
    bar1 = (np.abs(y - 0.78 * nn) < max(1.0, 0.045 * nn)) & (x > 0.12 * nn) & (x < 0.88 * nn)
    bar2 = (np.abs(y - 0.66 * nn) < max(1.0, 0.045 * nn)) & (x > 0.12 * nn) & (x < 0.88 * nn)
    img[bar1 | bar2] = 0.75
    return img


def spatial_blur_matrix(n: int, sigma: float, *, radius: int | None = None) -> np.ndarray:
    """一维周期（环绕）高斯模糊矩阵 ``B``（``n x n``）：``B[i, j] = g((i-j) mod n)``。

    周期边界让模糊算子是**循环**的、从而可对角化到 Fourier 基——但这里**不用** FFT
    （P2/P3 禁），而是老老实实形成矩阵再走 SVD。行归一化保证常向量是特征向量
    （``B 1 = 1``）：否则"模糊"会顺手改变图像的平均亮度，把反问题的误差与亮度偏移混在一起。
    """
    nn = int(n)
    sig = float(sigma)
    rr = int(radius) if radius is not None else max(1, int(np.ceil(3.0 * sig)))
    offs = np.arange(-rr, rr + 1, dtype=float)
    w = np.exp(-0.5 * (offs / sig) ** 2)
    w = w / float(np.sum(w))
    B = np.zeros((nn, nn), dtype=float)
    for k, o in enumerate(offs.astype(int)):
        B[np.arange(nn), (np.arange(nn) + o) % nn] = w[k]
    return B


def separable_blur_forward(X: Any, B: np.ndarray) -> np.ndarray:
    """二维可分离模糊 ``B X B^T``（行、列各模糊一次；不做任何卷积例程调用）。"""
    M = np.asarray(X, dtype=float)
    Bm = np.asarray(B, dtype=float)
    return Bm @ M @ Bm.T


def separable_blur_adjoint(Z: Any, B: np.ndarray) -> np.ndarray:
    """``(B X B^T)`` 的伴随 ``B^T Z B``。"""
    M = np.asarray(Z, dtype=float)
    Bm = np.asarray(B, dtype=float)
    return Bm.T @ M @ Bm


def separable_svd(B: np.ndarray) -> SVDFit:
    """一维模糊矩阵的 SVD（1.3 口径）：二维算子的奇异值是 ``s_i s_j``（可分离的代价与红利）。"""
    return factorize(B)


def separable_filter_factors(sh: Any, lam: float, *, kind: str = "tikhonov") -> np.ndarray:
    """二维可分离算子的滤波因子矩阵 ``f_ij``（``kind`` 取 ``tikhonov`` / ``tsvd``）。

    ``sh`` 是一维方向的奇异值；二维奇异值是外积 ``sh_i sh_j``。Tikhonov 用
    ``f_ij = (sh_i sh_j)^2/((sh_i sh_j)^2 + lam^2)``，TSVD 用
    ``f_ij = 1{sh_i sh_j > lam}``（**同一个 lam 当阈值**，两种方法才能在同一条
    参数轴上比较——这是本章"同样自由度谁更准"那一节的公平性前提）。
    """
    sv = np.asarray(sh, dtype=float).ravel()
    prod = np.outer(sv, sv)
    if kind == "tikhonov":
        return (prod * prod) / (prod * prod + float(lam) ** 2)
    if kind == "tsvd":
        return (prod > float(lam)).astype(float)
    raise ValueError(f"未知的 kind {kind!r}（tikhonov / tsvd）")


def separable_tikhonov(Y: Any, Uh: np.ndarray, sh: Any, Vh: np.ndarray, lam: float) -> np.ndarray:
    """可分离 Tikhonov 的二维解 ``X = Vh diag(f) (Uh^T Y Uh) Vh^T``（逐元素滤波）。"""
    return separable_apply_factors(Y, Uh, sh, Vh, separable_filter_factors(sh, lam, kind="tikhonov"))


def separable_tsvd(Y: Any, Uh: np.ndarray, sh: Any, Vh: np.ndarray, lam: float) -> np.ndarray:
    """可分离 TSVD（阈值同为 ``lam``）：保留 ``sh_i sh_j > lam`` 的二维模态。"""
    return separable_apply_factors(Y, Uh, sh, Vh, separable_filter_factors(sh, lam, kind="tsvd"))


def separable_apply_factors(Y: Any, Uh: np.ndarray, sh: Any, Vh: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """给定**滤波因子矩阵** ``factors`` 直接组装二维解（可分离算子的通用入口）。

    ``X = Vh (factors * (Uh^T Y Uh) / (sh_i sh_j)) Vh^T``（``sh_i sh_j = 0`` 处截断）。
    有了它，除 Tikhonov/TSVD 之外的任何"逐模态"方案（例如"只保留前 k 个一维模态的
    张量积"这种截断族）都不需要再抄一遍这条公式。
    """
    M = np.asarray(Y, dtype=float)
    sv = np.asarray(sh, dtype=float).ravel()
    prod = np.outer(sv, sv)
    safe = np.where(prod > 0.0, prod, 1.0)
    coef = np.asarray(Uh, dtype=float).T @ M @ np.asarray(Uh, dtype=float)
    scale = np.where(prod > 0.0, factors * coef / safe, 0.0)
    return np.asarray(Vh, dtype=float) @ scale @ np.asarray(Vh, dtype=float).T


def radon_matrix(n: int, n_angles: int, *, n_bins: int | None = None) -> np.ndarray:
    """简化平行束 CT 的显式投影矩阵（``(n_angles*n_bins) x (n*n)``，自研射线驱动模型）。

    每条射线上按**双线性**权重把像素累加到探测器（最近邻会让投影在像素边界上跳变，
    "重建出方格"这种伪影会被误读成正则化的问题）。角度在 ``[0, pi)`` 上均匀取，
    探测器宽度取像素宽度的 ``n_bins/n`` 倍（``n_bins >= n`` 时略有超采样，
    离散化误差小于反问题本身的病态程度）。
    """
    nn = int(n)
    na = int(n_angles)
    nb = int(n_bins) if n_bins is not None else nn
    A = np.zeros((na * nb, nn * nn), dtype=float)
    yy, xx = np.mgrid[0:nn, 0:nn].astype(float)
    cx = cy = 0.5 * (nn - 1)
    scale = float(nb) / float(nn)
    for a in range(na):
        th = np.pi * (a + 0.5) / na
        diry, dirx = float(np.sin(th)), float(np.cos(th))
        nory, norx = -dirx, diry  # 垂直方向 = 探测器轴
        chord = nn * (abs(dirx) + abs(diry))  # 常数图像在线上的积分长度（像素单位）
        for b in range(nb):
            s = (b + 0.5 - 0.5 * nb) / scale  # 探测器坐标（像素单位）
            dist = (xx - cx) * norx + (yy - cy) * nory - s
            w = np.maximum(1.0 - np.abs(dist), 0.0)
            tot = float(np.sum(w))
            if tot > 0.0:
                # 行归一化到"线积分"口径：常图像 1 的投影 = 弦长（不是 1、也不是像素个数）
                A[a * nb + b, :] = (w / tot * chord).ravel()
    return A
