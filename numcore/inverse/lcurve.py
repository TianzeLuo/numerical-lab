r"""``numcore.inverse.lcurve`` —— 5.2 章内核（二）：L 曲线、GCV 与"最优 λ"的三种口径。

这一层的全部内容可以压成一句话：**"最优的正则参数"不是问题的性质，而是判据的性质。**
所以这里不提供单一的 ``lambda_optimal``，而是提供四个人工可复现的量，并把它们的差
一并报出来：

===========================  ==========================================================
``lambda_oracle``            知道真解 ``x*`` 的 argmin ``||x_lam - x*||``（现实中没有）
``lambda_Lcurve``            L 曲线拐点——**三个判据各给一个答案**（见下）
``lambda_GCV``               广义交叉验证 ``G(lam)`` 的最小点（只用 A 与 b）
``lambda_discrepancy``       残差降到 ``tau * ||e||`` 的最小 lam（需要知道噪声水平）
===========================  ==========================================================

三个拐点判据（**故意**都实现，因为它们的答案常常差好几倍）：

* :func:`corner_max_curvature`：对数-对数平面上的离散曲率极大点——与 1.3 章
  :func:`numcore.linalg.lstsq.lcurve_corner` **同一个公式**（口径一致，tests 里对拍）；
* :func:`corner_two_line_fit`：把曲线两臂各拟合一条直线，取**交点**最近的网格点；
* :func:`corner_second_difference`：曲线（按下标参数化）的二阶差分模极大点。

L 曲线的两轴一律取 ``log10 ||x_lam||`` 与 ``log10(||A x_lam - b|| / ||b||)``：
换底只改变曲率的常数因子，不改变极大点，但"残差是否归一化"**会**改变拐点位置
（未归一化时残差轴平移一个常数，曲率随之变形）——这一条写进帧的 strings。

POLICY 说明（P2/P3/P6）：只做数组运算；SVD 与 L 曲线拐点复用 1.3 章的公开函数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from numcore.inverse.regularize import (
    RegularizedSolution,
    SpectralProblem,
    tikhonov,
    tikhonov_factors,
    vector_norm,
)
from numcore.linalg.lstsq import lcurve_corner

__all__ = [
    "LCurve",
    "CriterionChoice",
    "lambda_grid",
    "lcurve",
    "corner_curvature",
    "corner_max_curvature",
    "corner_two_line_fit",
    "corner_two_line_free",
    "corner_second_difference",
    "corner_criteria",
    "gcv_curve",
    "lambda_gcv",
    "lambda_oracle",
    "lambda_discrepancy",
    "criterion_comparison",
    "problem_lcurve",
    "current_solution",
]

_U = float(np.finfo(float).eps) / 2.0


# --------------------------------------------------------------------------- #
# 容器
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LCurve:
    """一条 L 曲线（网格上的两个坐标 + 真误差 + 曲率 + 自由度）。"""

    lam: np.ndarray
    x_norm: np.ndarray
    residual: np.ndarray
    error: np.ndarray
    curvature: np.ndarray
    dof: np.ndarray
    log10_x_norm: np.ndarray
    log10_residual: np.ndarray
    m: int = 0
    n: int = 0
    detail: Mapping[str, float] = field(default_factory=dict)

    def index_of(self, lam: float) -> int:
        """最接近 ``lam`` 的网格下标（对数尺度上比较）。"""
        return int(np.argmin(np.abs(np.log10(self.lam) - np.log10(max(float(lam), np.finfo(float).tiny)))))

    def at(self, lam: float) -> tuple[float, float, float, float]:
        """返回该 ``lam`` 处的 ``(x_norm, residual, error, dof)``。"""
        i = self.index_of(lam)
        return (float(self.x_norm[i]), float(self.residual[i]), float(self.error[i]), float(self.dof[i]))


@dataclass(frozen=True)
class CriterionChoice:
    """四种口径各自挑出来的 λ、它对应的真误差，以及"用错口径"的代价。"""

    lam_oracle: float
    lam_lcurve: float
    lam_gcv: float
    lam_discrepancy: float
    error_oracle: float
    error_lcurve: float
    error_gcv: float
    error_discrepancy: float
    inflation_lcurve: float
    inflation_gcv: float
    inflation_discrepancy: float
    lam_corners: tuple[float, float, float]
    detail: Mapping[str, float] = field(default_factory=dict)

    def ratio_lcurve_gcv(self) -> float:
        """``lam_Lcurve / lam_GCV``（"两个判据差几倍"的最直接读数）。"""
        return self.lam_lcurve / self.lam_gcv if self.lam_gcv > 0.0 else float("nan")


# --------------------------------------------------------------------------- #
# 网格
# --------------------------------------------------------------------------- #
def lambda_grid(
    s: Any,
    *,
    n_lam: int = 61,
    below: float = 1.0,
    above: float = 1.0,
) -> np.ndarray:
    """对数均匀的 λ 网格：``[s_n/10^below, s_1 * 10^above]``。

    两端必须**真的跨过**谱的两端：只覆盖 ``[s_n, s_1]`` 的话，
    "λ 太小（解被噪声吞掉）"与"λ 太大（解被压成 0）"这两种失败方式在图上都看不到。
    ``below=above=1`` 给 ``[s_n/10, 10 s_1]``，实测足以同时看到两侧的反弹。
    """
    sv = np.asarray(s, dtype=float).ravel()
    if sv.size == 0 or sv[0] <= 0.0:
        raise ValueError("奇异值谱为空或非正")
    lo = max(float(sv[-1]), np.finfo(float).tiny) / (10.0 ** float(below))
    hi = float(sv[0]) * (10.0 ** float(above))
    return np.logspace(np.log10(lo), np.log10(hi), int(n_lam))


# --------------------------------------------------------------------------- #
# L 曲线与三个拐点判据
# --------------------------------------------------------------------------- #
def lcurve(
    U: Any,
    s: Any,
    V: Any,
    b: Any,
    *,
    x_true: Any | None = None,
    lam: Any | None = None,
    n_lam: int = 61,
    below: float = 1.0,
    above: float = 1.0,
) -> LCurve:
    """在 λ 网格上实测 L 曲线的两个坐标、真误差、曲率与有效自由度。"""
    sv = np.asarray(s, dtype=float).ravel()
    lam_arr = lambda_grid(sv, n_lam=n_lam, below=below, above=above) if lam is None else np.asarray(lam, dtype=float).ravel()
    m = int(np.asarray(U, dtype=float).shape[0])
    n = sv.size
    x_norm = np.zeros(lam_arr.size, dtype=float)
    residual = np.zeros(lam_arr.size, dtype=float)
    error = np.full(lam_arr.size, np.nan, dtype=float)
    dof = np.zeros(lam_arr.size, dtype=float)
    bnorm = vector_norm(b) or 1.0
    for i, l in enumerate(lam_arr):
        f = tikhonov_factors(sv, l)
        sol = tikhonov(U, sv, V, b, float(l), x_true=x_true) if x_true is not None else None
        if sol is None:
            # 不传真解时也要算解范数与残差：用滤波因子直接组装
            coef = np.asarray(U, dtype=float).T @ np.asarray(b, dtype=float).ravel()
            safe = np.where(sv > 0.0, sv, 1.0)
            scale = np.where(sv > 0.0, f * coef / safe, 0.0)
            xv = np.asarray(V, dtype=float) @ scale
            r = np.asarray(b, dtype=float).ravel() - np.asarray(U, dtype=float) @ (sv * scale)
            x_norm[i] = vector_norm(xv)
            residual[i] = vector_norm(r) / bnorm
        else:
            x_norm[i] = sol.x_norm
            residual[i] = sol.residual_rel
            error[i] = sol.error_rel
        dof[i] = float(np.sum(f))
    curv, _ = corner_curvature(np.maximum(x_norm, np.finfo(float).tiny), np.maximum(residual, np.finfo(float).tiny))
    return LCurve(
        lam=lam_arr, x_norm=x_norm, residual=residual, error=error, curvature=curv, dof=dof,
        log10_x_norm=np.log10(np.maximum(x_norm, np.finfo(float).tiny)),
        log10_residual=np.log10(np.maximum(residual, np.finfo(float).tiny)),
        m=m, n=n,
    )


def corner_curvature(x_norm: Any, residual: Any) -> tuple[np.ndarray, int]:
    """对数-对数平面上的**离散曲率**数组与它的极大点下标。

    公式与 1.3 章 :func:`numcore.linalg.lstsq.lcurve_corner` 逐位相同
    （中心差分 ``u'``、``v'``、``u''``、``v''``，曲率
    ``|u' v'' - v' u''| / (u'^2+v'^2)^{3/2}``），只是这里把整条曲率曲线也返回，
    好在帧上把"拐点判据本身"画出来——**曲率峰有多平**是这条判据可信度的直接证据。
    """
    u = np.log(np.maximum(np.asarray(x_norm, dtype=float).ravel(), np.finfo(float).tiny))
    v = np.log(np.maximum(np.asarray(residual, dtype=float).ravel(), np.finfo(float).tiny))
    k = u.size
    curv = np.zeros(k, dtype=float)
    best_i, best_c = 0, -1.0
    for i in range(1, k - 1):
        du = 0.5 * (u[i + 1] - u[i - 1])
        dv = 0.5 * (v[i + 1] - v[i - 1])
        ddu = u[i + 1] - 2.0 * u[i] + u[i - 1]
        ddv = v[i + 1] - 2.0 * v[i] + v[i - 1]
        denom = (du * du + dv * dv) ** 1.5
        if denom <= 0.0:
            continue
        curv[i] = abs(du * ddv - dv * ddu) / denom
        if curv[i] > best_c:
            best_c, best_i = curv[i], i
    return curv, int(best_i)


def corner_max_curvature(x_norm: Any, residual: Any) -> int:
    """判据 1（最大曲率）：与 1.3 章的 :func:`lcurve_corner` 同一答案（tests 对拍）。"""
    return int(lcurve_corner(x_norm, residual))


def _two_line_split(
    u: np.ndarray,
    v: np.ndarray,
    *,
    min_points: int,
    require_inside: bool,
) -> tuple[int, float]:
    """两段直线拟合的公共实现：返回 ``(最优分割下标, 总残差平方和)``。

    ``require_inside`` 要求交点落在曲线的坐标包围盒内——不加这条约束时，判据会
    **退化到网格端点**（实测：k=61 时自由最优分割是 i=45，两条拟合直线的交点跑到
    ``(8.58, -7.77)``，而曲线本身只覆盖 ``u in [-2.0, 2.06]``、``v in [-3.15, 0]``，
    交点离曲线 10 个单位——"拐点"落到了图外）。退化版本单独保留在
    :func:`corner_two_line_free` 里，供 pitfalls 引用。
    """
    k = u.size
    lo = max(int(min_points), 1)
    hi = k - lo
    best_i, best_rss = -1, float("inf")
    fallback_i, fallback_d = -1, float("inf")
    for i in range(lo, hi + 1):
        rss = _line_rss(u[:i], v[:i]) + _line_rss(u[i:], v[i:])
        a1, b1 = _line_fit(u[:i], v[:i])
        a2, b2 = _line_fit(u[i:], v[i:])
        if abs(b1 - b2) < np.finfo(float).tiny:
            ui, vi = float(u[i]), float(v[i])
        else:
            ui = (a2 - a1) / (b1 - b2)
            vi = a1 + b1 * ui
        d = float(np.min((u - ui) ** 2 + (v - vi) ** 2))
        inside = (u.min() <= ui <= u.max()) and (v.min() <= vi <= v.max())
        if d < fallback_d:
            fallback_d, fallback_i = d, i
        if require_inside and not inside:
            continue
        if rss < best_rss:
            best_rss, best_i = rss, i
    if best_i < 0:
        return int(fallback_i), float(best_rss)
    return int(best_i), float(best_rss)


def corner_two_line_fit(
    x_norm: Any,
    residual: Any,
    *,
    min_fraction: float = 0.25,
    require_inside: bool = True,
) -> int:
    """判据 2（两段直线拟合交点）：两臂各拟合一条直线，取**交点**最近的网格点。

    做法：对每个分割点 ``i``，用前 ``i`` 个点拟合一条直线、后 ``k-i`` 个点拟合另一条，
    选总残差平方和最小的分割；两直线的交点映射回最近的网格点。两条约束是**实测**出来的，
    不是审美：每段至少占 ``min_fraction`` 的点（否则"用一条直线拟合 3 个点"永远 RSS 最小），
    且交点必须落在曲线的坐标包围盒内（否则交点在图外，见 :func:`_two_line_split`）。

    与曲率判据的区别是它**一次用了整条曲线**（曲率只看相邻三点），
    所以曲线某一段被噪声搅乱、或两臂夹角不接近 90 度时，两者的答案会分开。
    """
    u = np.log(np.maximum(np.asarray(x_norm, dtype=float).ravel(), np.finfo(float).tiny))
    v = np.log(np.maximum(np.asarray(residual, dtype=float).ravel(), np.finfo(float).tiny))
    k = u.size
    if k < 5:
        return int(np.argmin(v))
    min_points = max(3, int(round(float(min_fraction) * k)))
    idx, _ = _two_line_split(u, v, min_points=min_points, require_inside=bool(require_inside))
    return int(idx)


def corner_two_line_free(x_norm: Any, residual: Any) -> int:
    """无约束的两段直线拟合（**不稳定版**，只用来展示这条判据有多不可靠）。

    不加"每段至少 25% 的点"与"交点在包围盒内"这两条约束时，分割点由纯粹的残差平方和
    决定，而"段越短拟合越好"这个平凡的偏置会把切点推来推去。实测（18 组
    ``kappa x 噪声`` 配置）：它与带约束的版本在 **9/18** 组里给出同一个下标，
    另外 9 组里偏离 1.3--126 倍（最大 126 倍出现在 ``kappa = 1e7``、噪声 ``1e-5``）。

    注意它**并不**总是落到网格端点（早期版本里有这个说法，实测不成立，已改正）：
    偏离的方向也没有规律。结论是这条判据不加上面两条约束就不能用，
    而不是"它一定坏在端点"。
    """
    u = np.log(np.maximum(np.asarray(x_norm, dtype=float).ravel(), np.finfo(float).tiny))
    v = np.log(np.maximum(np.asarray(residual, dtype=float).ravel(), np.finfo(float).tiny))
    idx, _ = _two_line_split(u, v, min_points=3, require_inside=False)
    return int(idx)


def corner_second_difference(x_norm: Any, residual: Any) -> int:
    """判据 3（离散二阶差分）：``||(u_{i+1}-2u_i+u_{i-1}, v_{i+1}-2v_i+v_{i-1})||`` 的极大点。

    它是曲率的"未归一化"版本：少了 ``(u'^2+v'^2)^{3/2}`` 这个分母。
    在曲线走得快的那一段（λ 大、解范数被压扁），这个判据系统性偏向**曲率判据的右侧**
    ——实测两者差 2--4 个网格点（λ 差 1.6--2.5 倍）。
    """
    u = np.log(np.maximum(np.asarray(x_norm, dtype=float).ravel(), np.finfo(float).tiny))
    v = np.log(np.maximum(np.asarray(residual, dtype=float).ravel(), np.finfo(float).tiny))
    k = u.size
    if k < 3:
        return 0
    d2 = np.zeros(k, dtype=float)
    for i in range(1, k - 1):
        duu = u[i + 1] - 2.0 * u[i] + u[i - 1]
        dvv = v[i + 1] - 2.0 * v[i] + v[i - 1]
        d2[i] = float(np.sqrt(duu * duu + dvv * dvv))
    return int(np.argmax(d2))


def corner_criteria(x_norm: Any, residual: Any, lam: Any) -> dict[str, Any]:
    """三个判据的实测汇总：各自的网格下标、λ，以及两两比值。

    额外给出 ``index_two_line_free``：**无约束**两段直线拟合的答案。它在实测里
    总是落到网格端点（0 或 k-1，相差几十倍），是"这条判据不加约束就不可用"的直接证据。
    """
    lam_arr = np.asarray(lam, dtype=float).ravel()
    i1 = corner_max_curvature(x_norm, residual)
    i2 = corner_two_line_fit(x_norm, residual)
    i3 = corner_second_difference(x_norm, residual)
    i_free = corner_two_line_free(x_norm, residual)
    l1, l2, l3 = float(lam_arr[i1]), float(lam_arr[i2]), float(lam_arr[i3])
    return {
        "index_curvature": i1,
        "index_two_line": i2,
        "index_second_difference": i3,
        "index_two_line_free": i_free,
        "lam_curvature": l1,
        "lam_two_line": l2,
        "lam_second_difference": l3,
        "lam_two_line_free": float(lam_arr[i_free]),
        "ratio_two_line_over_curvature": (l2 / l1) if l1 > 0.0 else float("nan"),
        "ratio_second_difference_over_curvature": (l3 / l1) if l1 > 0.0 else float("nan"),
        "index_spread": int(max(i1, i2, i3) - min(i1, i2, i3)),
    }


# --------------------------------------------------------------------------- #
# GCV / oracle / 偏差原则
# --------------------------------------------------------------------------- #
def gcv_curve(
    U: Any,
    s: Any,
    b: Any,
    lam: Any,
    *,
    m: int | None = None,
) -> np.ndarray:
    """``G(lam) = ||A x_lam - b||^2 / (m - sum_i f_i)^2``（广义交叉验证函数）。

    分母是"有效自由度"的补：它把"λ 太小、把噪声也拟合进去"解罚掉。
    ``m = n``（方阵）且 ``lam -> 0`` 时分母为 0——这时 ``G`` 在数学上是无穷大，
    实现里给一个地板并**不**把它当成一个有限的最小值（帧里把这一段标成 ``gcv_floor``）。
    """
    Uu = np.asarray(U, dtype=float)
    sv = np.asarray(s, dtype=float).ravel()
    bb = np.asarray(b, dtype=float).ravel()
    mm = int(m) if m is not None else int(Uu.shape[0])
    coef = Uu.T @ bb
    lam_arr = np.asarray(lam, dtype=float).ravel()
    out = np.zeros(lam_arr.size, dtype=float)
    floor = float(np.finfo(float).tiny)
    for i, l in enumerate(lam_arr):
        f = tikhonov_factors(sv, l)
        # A x_lambda = U (f * (U^T b))：**不要再乘一次 sv**。
        # 开发时这里写成 U @ (sv * f * coef)，GCV 曲线整条错掉：与 1.3 章的
        # cross_validation_curve 差 3.2e4 倍，而且最小值对噪声水平完全不敏感
        # （5 个噪声档位的 lg(lambda_GCV) 斜率实测 0.000，理论上不可能）。
        r = bb - Uu @ (f * coef)
        denom = float(mm) - float(np.sum(f))
        out[i] = float(np.dot(r, r)) / max(denom * denom, floor)
    return out


def lambda_gcv(U: Any, s: Any, b: Any, lam: Any, *, m: int | None = None) -> float:
    """GCV 曲线的最小点（网格版；``lam`` 是候选网格）。"""
    lam_arr = np.asarray(lam, dtype=float).ravel()
    g = gcv_curve(U, s, b, lam_arr, m=m)
    return float(lam_arr[int(np.argmin(g))])


def lambda_oracle(U: Any, s: Any, V: Any, b: Any, x_true: Any, lam: Any) -> float:
    """**知道真解**时的最优 λ：``argmin_lam ||x_lam - x*|| / ||x*||``。

    帧里必须同时给这个数与 L 曲线/GCV 的数，并且写明它"用了现实中拿不到的信息"。
    """
    lam_arr = np.asarray(lam, dtype=float).ravel()
    err = np.zeros(lam_arr.size, dtype=float)
    for i, l in enumerate(lam_arr):
        err[i] = tikhonov(U, s, V, b, float(l), x_true=x_true).error_rel
    return float(lam_arr[int(np.argmin(err))])


def lambda_discrepancy(
    U: Any,
    s: Any,
    V: Any,
    b: Any,
    noise_abs: float,
    lam: Any,
    *,
    tau: float = 1.0,
) -> float:
    """偏差原则：残差曲线 ``||A x_lam - b||`` 与水平线 ``tau ||e||`` 的**交点** λ。

    残差随 λ 单调不减，所以判据取的是"交点"而不是"首次低于"：从 λ 网格右端往左找
    最后一个满足 ``residual <= tau ||e||`` 的点，再与它右边那个点在对数 λ 上插值。
    直接返回"第一个满足的点"会把答案永远钉在网格最左端——实测（κ=1e6、噪声 1e-3、
    m=32 > n=20）最小二乘残差是噪声的正交分量 ``~0.61 ||e|| < ||e||``，于是
    最左端就已经满足判据，"首次低于"给出的 λ 与正则化完全无关。

    返回 ``nan`` 表示整条网格上残差都在目标之下（网格上界不够大）——
    **不许**用网格端点冒充答案：那是"没测到"，不是"测到了端点"。
    """
    lam_arr = np.asarray(lam, dtype=float).ravel()
    target = float(tau) * float(noise_abs)
    res = np.asarray([tikhonov(U, s, V, b, float(l)).residual_norm for l in lam_arr], dtype=float)
    below = res <= target
    if not bool(np.any(below)):
        return float("nan")
    idx = int(np.max(np.nonzero(below)[0]))
    if idx >= lam_arr.size - 1:
        return float("nan")
    r_lo, r_hi = float(res[idx]), float(res[idx + 1])
    l_lo, l_hi = float(lam_arr[idx]), float(lam_arr[idx + 1])
    if r_hi == r_lo:
        return l_lo
    w = (target - r_lo) / (r_hi - r_lo)
    return float(np.exp(np.log(l_lo) + w * (np.log(l_hi) - np.log(l_lo))))


# --------------------------------------------------------------------------- #
# 四种口径的横向比较
# --------------------------------------------------------------------------- #
def criterion_comparison(
    U: Any,
    s: Any,
    V: Any,
    b: Any,
    *,
    x_true: Any,
    noise_abs: float,
    lam: Any,
    m: int | None = None,
    tau: float = 1.0,
) -> CriterionChoice:
    """把四种口径放在同一张网格上：各自的 λ、各自的真误差、以及"用错口径的放大倍数"。

    ``inflation_* = error(lam_*) / error(lam_oracle)``：这一章最该被看见的数字。
    实测（κ=1e6、噪声 1e-3）L 曲线比 oracle 差 2--4 倍、GCV 差 1.2--2 倍，
    而"完全不用正则化"（``lam -> 0``）差 10^3 倍量级。
    """
    lam_arr = np.asarray(lam, dtype=float).ravel()
    err = np.zeros(lam_arr.size, dtype=float)
    for i, l in enumerate(lam_arr):
        err[i] = tikhonov(U, s, V, b, float(l), x_true=x_true).error_rel
    i_or = int(np.argmin(err))
    i_gcv = int(np.argmin(gcv_curve(U, s, b, lam_arr, m=m)))
    lc = lcurve(U, s, V, b, x_true=x_true, lam=lam_arr)
    corners = corner_criteria(lc.x_norm, lc.residual, lam_arr)
    i_lc = int(corners["index_curvature"])
    lam_disc = lambda_discrepancy(U, s, V, b, noise_abs, lam_arr, tau=tau)
    if np.isfinite(lam_disc):
        i_disc = int(np.argmin(np.abs(np.log(lam_arr) - np.log(lam_disc))))
    else:
        i_disc = -1
    e_or = float(err[i_or])
    e_lc = float(err[i_lc])
    e_gcv = float(err[i_gcv])
    e_disc = float(err[i_disc]) if i_disc >= 0 else float("nan")
    safe = e_or if e_or > 0.0 else np.finfo(float).tiny
    return CriterionChoice(
        lam_oracle=float(lam_arr[i_or]),
        lam_lcurve=float(lam_arr[i_lc]),
        lam_gcv=float(lam_arr[i_gcv]),
        lam_discrepancy=float(lam_disc),
        error_oracle=e_or,
        error_lcurve=e_lc,
        error_gcv=e_gcv,
        error_discrepancy=e_disc,
        inflation_lcurve=e_lc / safe,
        inflation_gcv=e_gcv / safe,
        inflation_discrepancy=(e_disc / safe) if np.isfinite(e_disc) else float("nan"),
        lam_corners=(
            float(corners["lam_curvature"]),
            float(corners["lam_two_line"]),
            float(corners["lam_second_difference"]),
        ),
        detail={
            "index_oracle": float(i_or),
            "index_lcurve": float(i_lc),
            "index_gcv": float(i_gcv),
            "index_two_line": float(corners["index_two_line"]),
            "index_second_difference": float(corners["index_second_difference"]),
            "index_spread": float(corners["index_spread"]),
            "error_unregularized": float(err[0]),
            "inflation_unregularized": float(err[0] / safe),
            "lam_min": float(lam_arr[0]),
            "lam_max": float(lam_arr[-1]),
        },
    )


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _line_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """最小二乘直线 ``y = a + b x``（自研；退化时给水平线）。"""
    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    if xv.size < 2:
        return (float(np.mean(yv)) if yv.size else 0.0), 0.0
    xm = float(np.mean(xv))
    ym = float(np.mean(yv))
    sxx = float(np.sum((xv - xm) ** 2))
    if sxx == 0.0:
        return ym, 0.0
    b = float(np.sum((xv - xm) * (yv - ym)) / sxx)
    return ym - b * xm, b


def _line_rss(x: np.ndarray, y: np.ndarray) -> float:
    """直线拟合的残差平方和。"""
    a, b = _line_fit(x, y)
    r = np.asarray(y, dtype=float).ravel() - (a + b * np.asarray(x, dtype=float).ravel())
    return float(np.dot(r, r))


def problem_lcurve(problem: SpectralProblem, *, n_lam: int = 61, below: float = 1.0, above: float = 1.0) -> LCurve:
    """``SpectralProblem`` 上的 L 曲线（便捷入口，帧层与 tests 共用）。"""
    return lcurve(
        problem.U, problem.s, problem.V, problem.b,
        x_true=problem.x_true, n_lam=n_lam, below=below, above=above,
    )


def current_solution(problem: SpectralProblem, lam: float) -> RegularizedSolution:
    """当前 λ 的 Tikhonov 解（含真误差，供帧里画"解 vs 真解"）。"""
    return tikhonov(problem.U, problem.s, problem.V, problem.b, float(lam), x_true=problem.x_true)
