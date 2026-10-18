r"""2.2 章数值核心：三次样条、B 样条与分段逼近。

数学对象
--------
给定节点 :math:`x_0 < \dots < x_n` 与数据 :math:`y_i`，三次样条是分段三次多项式
:math:`S \in C^2[x_0,x_n]` 且 :math:`S(x_i) = y_i`。记 :math:`M_i = S''(x_i)`、
:math:`h_i = x_{i+1}-x_i`、:math:`\Delta_i = (y_{i+1}-y_i)/h_i`。在每个区间上由
四个条件 :math:`(y_i, y_{i+1}, M_i, M_{i+1})` 唯一确定的三次 Hermite 多项式给出

.. math::
    S(x) = M_i\frac{(x_{i+1}-x)^3}{6h_i} + M_{i+1}\frac{(x-x_i)^3}{6h_i}
         + \Big(y_i - \frac{M_i h_i^2}{6}\Big)\frac{x_{i+1}-x}{h_i}
         + \Big(y_{i+1} - \frac{M_{i+1} h_i^2}{6}\Big)\frac{x-x_i}{h_i},

它在节点处的**二阶导自动连续**（两侧都取 :math:`M_i`），剩下的自由度由
**一阶导连续**（:math:`C^1`）条件定下来，而那个条件恰好是一个三对角方程：

.. math::
    \mu_i M_{i-1} + 2 M_i + \lambda_i M_{i+1} = d_i, \qquad
    \lambda_i = \frac{h_i}{h_{i-1}+h_i},\quad \mu_i = 1-\lambda_i,\quad
    d_i = \frac{6}{h_{i-1}+h_i}\Big(\Delta_i - \Delta_{i-1}\Big).

边界条件决定第 0 行与第 :math:`n-1` 行：``natural``（:math:`M_0=M_n=0`）、
``clamped``（给定 :math:`S'(x_0),S'(x_n)`）、``not_a_knot``（在 :math:`x_1,x_{n-1}`
处三阶导连续，等价于"首末两个节点不是真正的结点"）。三个边界都只用
**手写追赶法**（:func:`tridiagonal_apply`）解 —— 本章要教的就是这个三对角结构，
所以连 :math:`n=3` 的特例都不许换成通用稠密求解。

B 样条
------
次数 :math:`k`、节点向量 :math:`t` 的 B 样条由 Cox--de Boor 递推定义

.. math::
    B_{i,0}(x) = [t_i \le x < t_{i+1}], \qquad
    B_{i,k}(x) = \frac{x-t_i}{t_{i+k}-t_i}B_{i,k-1}(x)
               + \frac{t_{i+k+1}-x}{t_{i+k+1}-t_{i+1}}B_{i+1,k-1}(x),

满足单位分解 :math:`\sum_i B_{i,k} \equiv 1`、非负性与局部支撑
:math:`\mathrm{supp}\,B_{i,k} = [t_i, t_{i+k+1}]`。曲线 :math:`C(x)=\sum_i c_i B_{i,k}(x)`
用 de Boor 算法求值，控制点修改只影响 :math:`k+1` 个区间；Boehm 节点插入不改变曲线。

自带的最小多项式实现
--------------------
2.1 章拥有 ``numcore/interp/poly.py``（Lagrange/Newton/重心/Runge 现象）。
本章的 ``spline_vs_poly`` 需要"同一组数据上的高阶多项式"作对照，因此在**本文件内**
自带一份最小重心公式实现（:func:`lagrange_value` 等）：这样 2.2 的断言与帧不依赖
2.1 的进度，也不会因为对 2.1 内部 API 的假设而耦合。两份实现各管各的章节。

实现约束
--------
只用 ``numpy`` 的容器与逐元素原语（POLICY P2/P3/P4/P6）：没有任何 ``numpy.linalg`` /
``numpy.fft`` / 现成插值器调用，三对角与带状系统一律手写；随机数据经
:func:`numcore.utils.rng` 取得。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np

__all__ = [
    # 三对角（手写追赶法）
    "tridiagonal_apply",
    "tridiagonal_matvec",
    "tridiagonal_residual",
    "tridiagonal_factor",
    "thomas_flops",
    # 三对角系统与样条
    "SplineSystem",
    "spline_system",
    "spline_moments",
    "PiecewiseCubic",
    "cubic_spline",
    "cubic_from_moments",
    "cubic_hermite",
    "pchip",
    "polyline",
    "fritsch_carlson_slopes",
    "hermite_monotone_ok",
    "hermite_monotone_margin",
    "monotonicity_report",
    # 曲率
    "curvature_of",
    "curvature_comb",
    "curvature_energy",
    # B 样条
    "clamped_knots",
    "uniform_clamped_knots",
    "knot_span",
    "bspline_basis",
    "bspline_basis_matrix",
    "de_boor",
    "bspline_derivative",
    "knot_insert",
    "insertion_sequence",
    "basis_support",
    "bspline_uniform_natural_ctrl",
    "convex_hull_gap",
    # 对照用多项式（本章自带的最小实现）
    "barycentric_weights",
    "lagrange_value",
    "lagrange_deriv2",
    "lebesgue_curve",
    "lebesgue_constant",
    "gauss_legendre",
    "interpolating_polynomial_energy",
    # 数据
    "TEST_FUNCTIONS",
    "sample_function",
    "uniform_nodes",
    "chebyshev_nodes",
]

Boundary = Literal["natural", "clamped", "not_a_knot"]

#: 单位舍入 u = 2^-53（用于把"精确 0"抬到对数轴画得出来的地板）
UNIT_ROUNDOFF = float(np.finfo(float).eps) / 2.0


# --------------------------------------------------------------------------- #
# 0. 小工具
# --------------------------------------------------------------------------- #
def _as_nodes(x: Any) -> np.ndarray:
    """把输入变成严格递增的一维节点数组（含边界校验）。"""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < 2:
        raise ValueError(f"节点至少 2 个，得到 {arr.size} 个")
    if not np.all(np.isfinite(arr)):
        raise ValueError("节点必须是有限数")
    d = np.diff(arr)
    if np.any(d <= 0.0):
        raise ValueError("节点必须严格递增（重复或逆序都不允许）")
    return arr


def _as_values(y: Any, n: int) -> np.ndarray:
    arr = np.asarray(y, dtype=float).ravel()
    if arr.size != n:
        raise ValueError(f"数据长度 {arr.size} 与节点数 {n} 不一致")
    if not np.all(np.isfinite(arr)):
        raise ValueError("数据必须是有限数")
    return arr


def _locate(nodes: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """返回 :math:`x_q` 所在的区间下标 :math:`i`：``nodes[i] <= xq < nodes[i+1]``（右端夹住）。

    这里用最朴素的线性扫描而不是二分：区间数在本章不超过几十，代码能一眼看出对错
    比省几次比较重要（也用不到任何现成的查找原语）。
    """
    m = nodes.size - 1
    idx = np.zeros(xq.size, dtype=np.int64)
    for j in range(1, m):
        idx = np.where(xq >= nodes[j], j, idx)
    return idx


# --------------------------------------------------------------------------- #
# 1. 三对角系统：手写追赶法（Thomas）
# --------------------------------------------------------------------------- #
def _check_tridiag(lower: Any, diag: Any, upper: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = np.asarray(lower, dtype=float).ravel()
    dg = np.asarray(diag, dtype=float).ravel()
    up = np.asarray(upper, dtype=float).ravel()
    n = dg.size
    if n == 0:
        raise ValueError("三对角系统为空")
    if lo.size != n or up.size != n:
        raise ValueError(f"三对角三条对角线长度必须一致：lower={lo.size}, diag={n}, upper={up.size}")
    return lo, dg, up


def tridiagonal_matvec(lower: Any, diag: Any, upper: Any, x: Any) -> np.ndarray:
    r"""三对角矩阵乘向量 :math:`A x`，其中 :math:`A = \mathrm{tridiag}(\ell, d, u)`。

    ``lower[0]`` 与 ``upper[-1]`` 不参与运算（它们是矩阵外的位置），约定为 0。
    """
    lo, dg, up = _check_tridiag(lower, diag, upper)
    v = np.asarray(x, dtype=float).ravel()
    if v.size != dg.size:
        raise ValueError(f"向量长度 {v.size} 与阶数 {dg.size} 不一致")
    out = dg * v
    out[1:] += lo[1:] * v[:-1]
    out[:-1] += up[:-1] * v[1:]
    return out


def tridiagonal_apply(lower: Any, diag: Any, upper: Any, rhs: Any) -> np.ndarray:
    r"""追赶法（Thomas 算法）：解三对角系统 :math:`A x = f`。

    消元--回代两步，共 :math:`8n-7` 次浮点运算、无填充：

    .. math::
        c'_0 = u_0/d_0,\quad
        c'_i = \frac{u_i}{d_i - \ell_i c'_{i-1}},\quad
        f'_i = \frac{f_i - \ell_i f'_{i-1}}{d_i - \ell_i c'_{i-1}},\qquad
        x_n = f'_n,\quad x_i = f'_i - c'_i x_{i+1}.

    样条系统的矩阵是**严格对角占优**的（:math:`|\ell_i| + |u_i| < |d_i|`，因为
    :math:`\mu_i + \lambda_i = 1 < 2`，两端行更明显），所以不需要选主元；
    真的遇到零主元（只有人为构造的病态输入才会）就抛 :class:`ZeroDivisionError`，
    绝不静默返回垃圾。
    """
    lo, dg, up = _check_tridiag(lower, diag, upper)
    f = np.asarray(rhs, dtype=float).ravel()
    n = dg.size
    if f.size != n:
        raise ValueError(f"右端项长度 {f.size} 与阶数 {n} 不一致")
    cp = np.zeros(n, dtype=float)
    fp = np.zeros(n, dtype=float)
    piv = dg[0]
    if piv == 0.0:
        raise ZeroDivisionError("追赶法遇到零主元（第 0 行）")
    cp[0] = up[0] / piv
    fp[0] = f[0] / piv
    for i in range(1, n):
        piv = dg[i] - lo[i] * cp[i - 1]
        if piv == 0.0:
            raise ZeroDivisionError(f"追赶法遇到零主元（第 {i} 行）")
        cp[i] = up[i] / piv
        fp[i] = (f[i] - lo[i] * fp[i - 1]) / piv
    out = np.empty(n, dtype=float)
    out[-1] = fp[-1]
    for i in range(n - 2, -1, -1):
        out[i] = fp[i] - cp[i] * out[i + 1]
    return out


def tridiagonal_factor(lower: Any, diag: Any, upper: Any) -> tuple[np.ndarray, np.ndarray]:
    r"""追赶法的因子：返回 :math:`(\ell, u)` 使 :math:`A = LU`（无填充）。

    约定 :math:`U` 的对角元是返回值 ``piv``、**超对角元就是原样的** ``upper[:-1]``；
    :math:`L` 是单位下三角，其次对角元是 :math:`\ell_i = a_i/u_{i-1}`（``lower[0]=0``）。
    注意 :math:`\ell \ne` ``lower``：原次对角元要除以主元才是乘子，
    把这两者弄混是"重建对不上"的经典原因（断言里就是这么抓出来的）。
    """
    lo, dg, up = _check_tridiag(lower, diag, upper)
    n = dg.size
    piv = np.zeros(n, dtype=float)
    mult = np.zeros(n, dtype=float)
    piv[0] = dg[0]
    if piv[0] == 0.0:
        raise ZeroDivisionError("追赶法遇到零主元（第 0 行）")
    for i in range(1, n):
        if piv[i - 1] == 0.0:
            raise ZeroDivisionError(f"追赶法遇到零主元（第 {i - 1} 行）")
        mult[i] = lo[i] / piv[i - 1]
        piv[i] = dg[i] - mult[i] * up[i - 1]
    return mult, piv


def tridiagonal_residual(lower: Any, diag: Any, upper: Any, x: Any, rhs: Any) -> float:
    r"""残差 :math:`\|Ax-f\|_\infty / (\|A\|_\infty \|x\|_\infty + \|f\|_\infty)`。

    分母里带 :math:`\|A\|\|x\|` 是"范数型后向误差"的标准写法：它无量纲，
    规模放大 1e6 倍也不会变（与 1.2 章 ``backward_error`` 同一口径）。
    """
    lo, dg, up = _check_tridiag(lower, diag, upper)
    v = np.asarray(x, dtype=float).ravel()
    f = np.asarray(rhs, dtype=float).ravel()
    r = tridiagonal_matvec(lo, dg, up, v) - f
    num = float(np.max(np.abs(r))) if r.size else 0.0
    row_sum = np.abs(dg) + np.abs(lo) + np.abs(up)
    denom = float(np.max(row_sum)) * float(np.max(np.abs(v)) if v.size else 0.0) + float(
        np.max(np.abs(f)) if f.size else 0.0
    )
    return num / denom if denom > 0.0 else num


def thomas_flops(n: int) -> int:
    r"""追赶法的浮点运算量 :math:`8n-7`（:math:`n\ge1`）。

    与稠密消元的 :math:`\sim\frac{2}{3}n^3` 相比，这正是"看到三对角就别用稠密"的量化理由。
    """
    if n < 1:
        raise ValueError("阶数至少 1")
    return 8 * int(n) - 7


# --------------------------------------------------------------------------- #
# 2. 三次样条：C² 条件 → 三对角系统
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SplineSystem:
    r"""决定样条二阶导 :math:`M_1..M_{n-1}` 的三对角系统（本文件的核心结构）。

    ``lower/diag/upper/rhs`` 是**降阶后**的系统（未知量只有内点弯矩
    :math:`M_1,\dots,M_{n-1}`）：端点弯矩由边界条件解析地消去，所以三种边界条件
    都落在同一套三对角存储里。``lam``/``mu``/``d`` 给出教科书里的
    :math:`\lambda_i,\mu_i,d_i`，便于断言逐项核对。
    """

    lower: np.ndarray
    diag: np.ndarray
    upper: np.ndarray
    rhs: np.ndarray
    lam: np.ndarray
    mu: np.ndarray
    d: np.ndarray
    bc: str
    d_left: float
    d_right: float

    @property
    def n_interior(self) -> int:
        return int(self.diag.size)


def spline_system(
    x: Any,
    y: Any,
    bc: str = "natural",
    *,
    d_left: float | None = None,
    d_right: float | None = None,
) -> SplineSystem:
    r"""三次样条的 C¹ 条件导出的三对角系统（未知量 = 内点弯矩）。

    区间 :math:`i` 上取 Hermite 形式后，节点 :math:`x_i` 处一阶导连续给出

    .. math::
        \mu_i M_{i-1} + 2M_i + \lambda_i M_{i+1} = d_i,\qquad i=1,\dots,n-1 .

    三种边界把它化成同一个三对角系统的**首末行修改**：

    * ``natural``：:math:`M_0=M_n=0`，首末行不动；
    * ``clamped``：由 :math:`2M_0+M_1=\frac{6}{h_0}(\Delta_0-d_0)` 解出
      :math:`M_0`，代回 :math:`i=1` 行得对角元 :math:`2-\mu_1/2`、
      右端 :math:`d_1-\frac{3(\Delta_0-d_0)}{h_0+h_1}`；右端对称；
    * ``not_a_knot``：由 :math:`(M_1-M_0)/h_0=(M_2-M_1)/h_1` 解出 :math:`M_0`，
      代回得 :math:`\big(2+\frac{h_0}{h_1}\big)M_1+\frac{h_1-h_0}{h_1}M_2=d_1`；右端对称。

    ``not_a_knot`` 至少需要 3 个区间（4 个节点）才谈得上，否则抛 :class:`ValueError`。
    """
    xs = _as_nodes(x)
    ys = _as_values(y, xs.size)
    n = xs.size - 1  # 区间数
    h = np.diff(xs)
    delta = np.diff(ys) / h
    if bc not in ("natural", "clamped", "not_a_knot"):
        raise ValueError(f"未知边界条件 {bc!r}（可选 natural / clamped / not_a_knot）")
    if n < 2:
        raise ValueError("样条至少需要 3 个节点（2 个区间）")

    lam = np.zeros(n + 1, dtype=float)
    mu = np.zeros(n + 1, dtype=float)
    d = np.zeros(n + 1, dtype=float)
    for i in range(1, n):
        lam[i] = h[i] / (h[i - 1] + h[i])
        mu[i] = h[i - 1] / (h[i - 1] + h[i])
        d[i] = 6.0 * (delta[i] - delta[i - 1]) / (h[i - 1] + h[i])

    dl = float(delta[0] if d_left is None else d_left)
    dr = float(delta[-1] if d_right is None else d_right)

    size = n - 1
    lower = np.zeros(size, dtype=float)
    diag = np.full(size, 2.0, dtype=float)
    upper = np.zeros(size, dtype=float)
    rhs = np.zeros(size, dtype=float)
    for j in range(size):
        i = j + 1
        lower[j] = mu[i]
        upper[j] = lam[i]
        rhs[j] = d[i]
    if bc == "clamped":
        lower[0] = 0.0
        diag[0] = 2.0 - 0.5 * mu[1]
        rhs[0] = d[1] - 3.0 * (delta[0] - dl) / (h[0] + h[1])
        upper[-1] = 0.0
        diag[-1] = 2.0 - 0.5 * lam[n - 1]
        rhs[-1] = d[n - 1] - 3.0 * (dr - delta[-1]) / (h[n - 2] + h[n - 1])
    elif bc == "not_a_knot":
        if n < 3:
            raise ValueError("not-a-knot 边界至少需要 4 个节点（3 个区间）")
        lower[0] = 0.0
        diag[0] = 2.0 + h[0] / h[1]
        upper[0] = (h[1] - h[0]) / h[1]
        rhs[0] = d[1]
        upper[-1] = 0.0
        diag[-1] = 2.0 + h[n - 1] / h[n - 2]
        lower[-1] = (h[n - 2] - h[n - 1]) / h[n - 2]
        rhs[-1] = d[n - 1]
    return SplineSystem(
        lower=lower, diag=diag, upper=upper, rhs=rhs, lam=lam, mu=mu, d=d, bc=bc, d_left=dl, d_right=dr
    )


def spline_moments(
    x: Any,
    y: Any,
    bc: str = "natural",
    *,
    d_left: float | None = None,
    d_right: float | None = None,
) -> np.ndarray:
    r"""用追赶法解出全部弯矩 :math:`M_0..M_n`（含由边界条件还原的两个端点值）。

    返回长度为 ``len(x)`` 的数组。``natural`` 时端点为 0；``clamped`` 时由
    :math:`2M_0+M_1=\frac{6}{h_0}(\Delta_0-d_0)` 还原；``not_a_knot`` 时由三阶导连续还原。
    """
    xs = _as_nodes(x)
    ys = _as_values(y, xs.size)
    n = xs.size - 1
    if bc == "natural" and n == 1:
        return np.zeros(2, dtype=float)
    sysm = spline_system(xs, ys, bc, d_left=d_left, d_right=d_right)
    interior = tridiagonal_apply(sysm.lower, sysm.diag, sysm.upper, sysm.rhs)
    m = np.zeros(n + 1, dtype=float)
    m[1:n] = interior
    h = np.diff(xs)
    delta = np.diff(ys) / h
    if bc == "clamped":
        m[0] = 0.5 * (6.0 * (delta[0] - sysm.d_left) / h[0] - m[1])
        m[n] = 0.5 * (6.0 * (sysm.d_right - delta[-1]) / h[-1] - m[n - 1])
    elif bc == "not_a_knot":
        m[0] = m[1] + h[0] * (m[1] - m[2]) / h[1]
        m[n] = m[n - 1] + h[n - 1] * (m[n - 1] - m[n - 2]) / h[n - 2]
    return m


@dataclass(frozen=True)
class PiecewiseCubic:
    r"""分段三次多项式 :math:`S`（每个区间用局部幂基 :math:`s=x-x_i` 存储）。

    ``coeffs[i] = (c_0,c_1,c_2,c_3)`` 表示
    :math:`S(x_i+s) = c_0 + c_1 s + c_2 s^2 + c_3 s^3`。
    ``d1_left/d1_right``、``d2_left/d2_right`` 是每个区间**两端的一侧极限**——
    把它们分开存是本章的关键：只有分开才画得出节点处的**跳变**
    （用一根平滑曲线连过去等于把跳变抹掉，见 ``chapters/2.2/pitfalls.md``）。
    """

    x: np.ndarray
    y: np.ndarray
    coeffs: np.ndarray
    d1_left: np.ndarray
    d1_right: np.ndarray
    d2_left: np.ndarray
    d2_right: np.ndarray
    label: str = ""
    bc: str = ""
    moments: np.ndarray | None = field(default=None)
    slopes: np.ndarray | None = field(default=None)

    @property
    def n_intervals(self) -> int:
        return int(self.x.size - 1)

    # -- 求值 ------------------------------------------------------------ #
    def evaluate(self, xq: Any, deriv: int = 0) -> np.ndarray:
        """在 ``xq`` 处求 :math:`S^{(deriv)}`（``deriv = 0..3``，区间外按端点区间外推）。"""
        if deriv not in (0, 1, 2, 3):
            raise ValueError(f"只支持 0..3 阶导，得到 {deriv}")
        pts = np.atleast_1d(np.asarray(xq, dtype=float))
        i = _locate(self.x, pts)
        s = pts - self.x[i]
        c = self.coeffs[i]
        if deriv == 0:
            return c[:, 0] + s * (c[:, 1] + s * (c[:, 2] + s * c[:, 3]))
        if deriv == 1:
            return c[:, 1] + s * (2.0 * c[:, 2] + 3.0 * s * c[:, 3])
        if deriv == 2:
            return 2.0 * c[:, 2] + 6.0 * s * c[:, 3]
        return 6.0 * c[:, 3]  # 三阶导在每个区间上是常数，但**各区间不同**

    def derivative_sides(self, order: int = 2) -> tuple[np.ndarray, np.ndarray]:
        r"""**节点处**两侧的一侧极限 :math:`(S^{(r)}(x_i^-), S^{(r)}(x_i^+))`。

        返回两个长度 ``n+1`` 的数组：``left[i]`` 是左区间在 :math:`x_i` 右端点的值，
        ``right[i]`` 是右区间在 :math:`x_i` 左端点的值。端点只有一侧，另一侧用 ``nan``
        标记（画图时自然断开，不会用一根假线把跳变抹平）。

        这是本章"防误导"的核心 API：**只有把两侧分开存，节点处的跳变才画得出来**。
        """
        if order == 1:
            lft_all, rgt_all = self.d1_left, self.d1_right
        elif order == 2:
            lft_all, rgt_all = self.d2_left, self.d2_right
        else:
            raise ValueError("只支持一阶/二阶导的两侧极限（三阶导在区间内是常数）")
        m = self.n_intervals
        left = np.full(m + 1, np.nan, dtype=float)
        right = np.full(m + 1, np.nan, dtype=float)
        left[1:] = rgt_all  # x_i^- 取左区间右端点
        right[:-1] = lft_all  # x_i^+ 取右区间左端点
        return left, right

    def node_grid(self, eps: float) -> np.ndarray:
        r"""为了画出跳变而用的采样横坐标：每个节点处**左右各取一点** (:math:`x_i\pm\epsilon`)。

        默认的等距采样会以 :math:`O(h)` 的概率正好踩在节点上，于是二阶导的跳变
        被"相邻两点连线"抹成一段斜坡——图上看不出任何异常。这个函数就是那条对策：
        把 :math:`\epsilon` 取得足够小，让跳变以竖直段的形式**显式**画出来。
        """
        nodes = self.x
        pts: list[float] = [float(nodes[0])]
        for i in range(1, nodes.size - 1):
            pts.append(float(nodes[i]) - float(eps))
            pts.append(float(nodes[i]) + float(eps))
        pts.append(float(nodes[-1]))
        return np.asarray(pts, dtype=float)


    # -- 结构性量 -------------------------------------------------------- #
    def jumps(self, which: str = "d2") -> np.ndarray:
        r"""内点处的跳变 :math:`S^{(r)}(x_i^+)-S^{(r)}(x_i^-)`（长度 ``n-1``）。

        ``which="d2"`` 是本章的主角：``C^2`` 样条它是 :math:`O(u)`，
        ``C^1`` 的 PCHIP 它是 :math:`O(h)` 级别的真跳变，折线（``C^0``）的二阶导
        两侧都是 0 —— 所以**跳变本身也要用一阶导的跳变来对照**，见 :meth:`jump_report`。
        """
        if which == "d2":
            lft, rgt = self.d2_left[1:], self.d2_right[:-1]
        elif which == "d1":
            lft, rgt = self.d1_left[1:], self.d1_right[:-1]
        elif which == "d3":
            lft, rgt = self.d3_left()[1:], self.d3_right()[:-1]
        else:
            raise ValueError(f"未知的导数阶 {which!r}")
        return lft - rgt

    def d3_left(self) -> np.ndarray:
        """每区间左端的三阶导（常数）。"""
        return 6.0 * self.coeffs[:, 3]

    def d3_right(self) -> np.ndarray:
        """每区间右端的三阶导（常数，与左端相同）。"""
        return 6.0 * self.coeffs[:, 3]

    def jump_report(self) -> dict[str, float]:
        """节点处一阶导/二阶导跳变的最大绝对值（``C^1``/``C^2`` 的量化判据）。"""
        j1 = np.abs(self.jumps("d1"))
        j2 = np.abs(self.jumps("d2"))
        gap1 = j1 / max(float(np.max(np.abs(self.d1_left))), 1e-300)
        gap2 = j2 / max(float(np.max(np.abs(self.d2_left))), 1e-300)
        return {
            "max_d1_jump": float(np.max(j1)) if j1.size else 0.0,
            "max_d2_jump": float(np.max(j2)) if j2.size else 0.0,
            "rel_d1_jump": float(np.max(gap1)) if gap1.size else 0.0,
            "rel_d2_jump": float(np.max(gap2)) if gap2.size else 0.0,
        }

    def energy(self) -> float:
        r"""曲率能量 :math:`\int_{x_0}^{x_n} (S'')^2\,\mathrm{d}x`（**解析精确**）。

        每个区间的 :math:`S''` 是一次函数，故积分闭式：

        .. math::
            \int_0^{h}(2c_2+6c_3s)^2\,\mathrm{d}s
            = 4c_2^2h + 12c_2c_3h^2 + 12c_3^2h^3
            = \frac{h}{3}\big(M_i^2 + M_iM_{i+1} + M_{i+1}^2\big),

        两条表达式在断言里互相对拍（它们是同一个积分的两种写法）。
        """
        h = np.diff(self.x)
        c2 = self.coeffs[:, 2]
        c3 = self.coeffs[:, 3]
        per = 4.0 * c2 * c2 * h + 12.0 * c2 * c3 * h * h + 12.0 * c3 * c3 * h * h * h
        return float(np.sum(per))

    def energy_from_moments(self) -> float:
        r""":math:`\sum_i \frac{h_i}{3}(M_i^2+M_iM_{i+1}+M_{i+1}^2)` —— 与 :meth:`energy` 同值。"""
        m = self.d2_left
        m_next = self.d2_right
        h = np.diff(self.x)
        return float(np.sum(h * (m * m + m * m_next + m_next * m_next) / 3.0))

    def arc_length(self, n_grid: int = 400) -> float:
        """弧长（复合梯形，采样足够密时误差 :math:`O(n^{-2})`）。"""
        pts = np.linspace(self.x[0], self.x[-1], int(n_grid) + 1)
        v = self.evaluate(pts, 1)
        dx = np.diff(pts)
        seg = np.sqrt(1.0 + v * v)
        return float(np.sum(0.5 * (seg[:-1] + seg[1:]) * dx))

    def max_slope(self, n_grid: int = 400) -> float:
        pts = np.linspace(self.x[0], self.x[-1], int(n_grid) + 1)
        return float(np.max(np.abs(self.evaluate(pts, 1))))


def cubic_hermite(x: Any, y: Any, slopes: Any, *, label: str = "hermite") -> PiecewiseCubic:
    r"""给定节点斜率的三次 Hermite 插值 :math:`H`（:math:`H(x_i)=y_i`, :math:`H'(x_i)=m_i`）。

    区间 :math:`[x_i,x_{i+1}]` 上 :math:`H` 由 :math:`(y_i,y_{i+1},m_i,m_{i+1})` 唯一确定：

    .. math::
        H(x_i+s) = y_i + m_i s + \frac{3\Delta_i - 2m_i - m_{i+1}}{h_i}s^2
                 + \frac{m_i + m_{i+1} - 2\Delta_i}{h_i^2}s^3 .

    PCHIP 就是这个族里挑了一组不让曲线越界的斜率 :math:`m_i`（见 :func:`pchip`）。
    """
    xs = _as_nodes(x)
    ys = _as_values(y, xs.size)
    m = np.asarray(slopes, dtype=float).ravel()
    if m.size != xs.size:
        raise ValueError(f"斜率长度 {m.size} 与节点数 {xs.size} 不一致")
    h = np.diff(xs)
    delta = np.diff(ys) / h
    c0 = ys[:-1].copy()
    c1 = m[:-1].copy()
    c2 = (3.0 * delta - 2.0 * m[:-1] - m[1:]) / h
    c3 = (m[:-1] + m[1:] - 2.0 * delta) / (h * h)
    coeffs = np.stack([c0, c1, c2, c3], axis=1)
    d1l = c1
    d1r = c1 + 2.0 * c2 * h + 3.0 * c3 * h * h
    d2l = 2.0 * c2
    d2r = 2.0 * c2 + 6.0 * c3 * h
    return PiecewiseCubic(
        x=xs, y=ys, coeffs=coeffs, d1_left=d1l, d1_right=d1r, d2_left=d2l, d2_right=d2r,
        label=label, bc="hermite", moments=None, slopes=m,
    )


def cubic_from_moments(x: Any, y: Any, moments: Any, *, label: str = "", bc: str = "") -> PiecewiseCubic:
    r"""由节点值 :math:`y_i` 与节点二阶导 :math:`M_i` 造分段三次 Hermite 插值。

    区间 :math:`[x_i,x_{i+1}]` 上（:math:`s = x-x_i`）

    .. math::
        S = y_i + \Big(\Delta_i - \frac{h_i(2M_i+M_{i+1})}{6}\Big)s
          + \frac{M_i}{2}s^2 + \frac{M_{i+1}-M_i}{6h_i}s^3 ,

    它**自动**过所有节点、且 :math:`S''` 在节点处两侧都等于 :math:`M_i`（连续）。
    :math:`M` 取成三对角系统的解时 :math:`S'` 也连续，于是它就是三次样条；
    人为把 :math:`M` 整体乘 :math:`(1-\tau)` 就得到一条"更平"的曲线：仍然过所有数据点、
    但 :math:`S'` 在节点处**跳变**——这就是本章 ``control_points`` 的张力参数。
    """
    xs = _as_nodes(x)
    ys = _as_values(y, xs.size)
    m = np.asarray(moments, dtype=float).ravel()
    if m.size != xs.size:
        raise ValueError(f"弯矩个数 {m.size} 与节点数 {xs.size} 不一致")
    h = np.diff(xs)
    delta = np.diff(ys) / h
    c0 = ys[:-1].copy()
    c1 = delta - h * (2.0 * m[:-1] + m[1:]) / 6.0
    c2 = 0.5 * m[:-1]
    c3 = (m[1:] - m[:-1]) / (6.0 * h)
    coeffs = np.stack([c0, c1, c2, c3], axis=1)
    return PiecewiseCubic(
        x=xs, y=ys, coeffs=coeffs,
        d1_left=c1, d1_right=delta + h * (m[:-1] + 2.0 * m[1:]) / 6.0,
        d2_left=m[:-1].copy(), d2_right=m[1:].copy(),
        label=label, bc=bc, moments=m, slopes=None,
    )


def cubic_spline(
    x: Any,
    y: Any,
    bc: str = "natural",
    *,
    d_left: float | None = None,
    d_right: float | None = None,
) -> PiecewiseCubic:
    r"""三次样条 :math:`S\in C^2`，:math:`S(x_i)=y_i`，边界条件由 ``bc`` 指定。

    ``clamped`` 时 :math:`d_0,d_1` 缺省取**单侧割线斜率** :math:`\Delta_0,\Delta_{n-1}`
    （"端点先按直线走"），也可以由调用方给出。得到的 :class:`PiecewiseCubic` 里
    ``moments`` 是全文的 :math:`M_i`，``jump_report`` 会告诉你它到底有多 :math:`C^2`。
    """
    xs = _as_nodes(x)
    ys = _as_values(y, xs.size)
    n = xs.size - 1
    if n == 1:
        # 两个节点：任何边界条件下的"三次样条"都退化成那条直线
        m = np.zeros(2, dtype=float)
        return cubic_from_moments(xs, ys, m, label=f"spline[{bc}]", bc=bc)
    m = spline_moments(xs, ys, bc, d_left=d_left, d_right=d_right)
    return cubic_from_moments(xs, ys, m, label=f"spline[{bc}]", bc=bc)


def polyline(x: Any, y: Any) -> PiecewiseCubic:
    r"""分段线性插值（"C^0 参照物"）。

    ``d2`` 两侧都是 0：它的二阶导图**看起来最光滑**（一条 0 线），可它连
    :math:`C^1` 都不是——一阶导在每个内点跳变。这正是本章最阴的陷阱之一。
    """
    xs = _as_nodes(x)
    ys = _as_values(y, xs.size)
    h = np.diff(xs)
    delta = np.diff(ys) / h
    z = np.zeros(xs.size - 1, dtype=float)
    coeffs = np.stack([ys[:-1].copy(), delta.copy(), z.copy(), z.copy()], axis=1)
    return PiecewiseCubic(
        x=xs, y=ys, coeffs=coeffs, d1_left=delta.copy(), d1_right=delta.copy(),
        d2_left=z.copy(), d2_right=z.copy(), label="polyline", bc="linear", moments=None,
        slopes=delta.copy(),
    )


def hermite_monotone_ok(d0: float, d1: float, delta: float) -> bool:
    r"""区间上三次 Hermite 单调的**充分条件**（Fritsch--Carlson）。

    令 :math:`\alpha = d_0/\Delta`、:math:`\beta = d_1/\Delta`（:math:`\Delta` 是割线斜率），
    则 :math:`\alpha\ge0,\ \beta\ge0,\ \alpha^2+\beta^2\le 9` 时该区间上的三次
    Hermite 单调。这是充分而非必要条件（真正的单调区域边界由
    :math:`\alpha^2+\beta^2=9` 的椭圆弧与两条直线拼成），本章用它作为**判据**，
    再用密采样实测单调性兜底。
    """
    if delta == 0.0:
        return d0 == 0.0 and d1 == 0.0
    a = d0 / delta
    b = d1 / delta
    return a >= 0.0 and b >= 0.0 and a * a + b * b <= 9.0


def hermite_monotone_margin(d0: float, d1: float, delta: float) -> float:
    r"""距"充分单调区域"的余量 :math:`9-(\alpha^2+\beta^2)`（负数 = 判据失效）。

    非负表示充分条件成立；负值只说明判据不保证，**不代表一定不单调**——
    这个区别本身就是本章"形状保持不能只看判据"的那条陷阱。
    """
    if delta == 0.0:
        return 9.0 if (d0 == 0.0 and d1 == 0.0) else -9.0
    a = d0 / delta
    b = d1 / delta
    return 9.0 - (a * a + b * b)


def fritsch_carlson_slopes(x: Any, y: Any) -> np.ndarray:
    r"""Fritsch--Carlson 保单调斜率（PCHIP 的 :math:`m_i`）。

    * 内点：两侧割线同号时取**加权调和平均**
      :math:`m_i = \frac{w_1+w_2}{w_1/\Delta_{i-1}+w_2/\Delta_i}`，
      :math:`w_1 = 2h_i+h_{i-1}`, :math:`w_2 = h_i+2h_{i-1}`；异号（或为零）时取 0
      —— 极值点必须把斜率压成 0，否则曲线会冲出去再回来；
    * 端点：三点单侧公式，再按
      :math:`\mathrm{sign}` 与 :math:`|m_0|\le3|\Delta_0|` 截断（这是"端点不许更陡"的
      标准做法，缺了它端点区间仍可能越界）。
    """
    xs = _as_nodes(x)
    ys = _as_values(y, xs.size)
    n = xs.size
    if n < 3:
        raise ValueError("PCHIP 至少需要 3 个节点")
    h = np.diff(xs)
    delta = np.diff(ys) / h
    m = np.zeros(n, dtype=float)
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] > 0.0:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
        else:
            m[i] = 0.0
    m[0] = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if m[0] * delta[0] <= 0.0:
        m[0] = 0.0
    elif delta[0] * delta[1] < 0.0 and abs(m[0]) > 3.0 * abs(delta[0]):
        m[0] = 3.0 * delta[0]
    m[-1] = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if m[-1] * delta[-1] <= 0.0:
        m[-1] = 0.0
    elif delta[-1] * delta[-2] < 0.0 and abs(m[-1]) > 3.0 * abs(delta[-1]):
        m[-1] = 3.0 * delta[-1]
    return m


def pchip(x: Any, y: Any) -> PiecewiseCubic:
    r"""保单调三次 Hermite 插值（PCHIP）：节点处 :math:`C^1`，但 :math:`S''` 逐点跳变。

    它是"看起来和样条一样光滑、其实只是 :math:`C^1`"的活教材：
    一阶导连续（肉眼几乎看不出），二阶导在节点处是**真跳变**（曲率梳与误差估计会立刻发现）。
    """
    xs = _as_nodes(x)
    ys = _as_values(y, xs.size)
    return cubic_hermite(xs, ys, fritsch_carlson_slopes(xs, ys), label="pchip")


def monotonicity_report(piece: PiecewiseCubic, *, n_grid: int = 4001) -> dict[str, float]:
    """密采样实测分段函数是否单调（返回最小/最大斜率与"违反了几个采样点"）。"""
    pts = np.linspace(piece.x[0], piece.x[-1], int(n_grid))
    v = piece.evaluate(pts, 1)
    d = np.diff(piece.evaluate(pts, 0))
    sign_ref = 1.0 if float(np.sum(d)) >= 0.0 else -1.0
    bad = int(np.count_nonzero(sign_ref * v < -1e-12))
    return {
        "min_slope": float(np.min(v)),
        "max_slope": float(np.max(v)),
        "violations": float(bad),
        "monotone": 1.0 if bad == 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
# 3. 曲率与曲率梳
# --------------------------------------------------------------------------- #
def curvature_of(piece: PiecewiseCubic, xq: Any) -> np.ndarray:
    r"""平面曲线 :math:`(x, S(x))` 的曲率 :math:`\kappa = \frac{S''}{(1+S'^2)^{3/2}}`。

    注意它是**带符号**的：凸向上为正。曲率梳把 :math:`\kappa` 画成法向的"梳齿"，
    所以符号决定了梳齿朝哪一侧——朝下时不要误读成"曲率为零"。
    """
    pts = np.atleast_1d(np.asarray(xq, dtype=float))
    d1 = piece.evaluate(pts, 1)
    d2 = piece.evaluate(pts, 2)
    return d2 / np.power(1.0 + d1 * d1, 1.5)


def curvature_comb(
    piece: PiecewiseCubic,
    xq: Any,
    *,
    scale: str = "global",
    length: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    r"""曲率梳的几何：返回 ``(base, tip)``，两个都是 ``(n, 2)``。

    ``base = (x, S(x))``，``tip = base + L\,\kappa\,n``，其中
    :math:`n = (-S', 1)/\sqrt{1+S'^2}` 是单位法向。

    ``scale="global"``：:math:`L` 取成"最大曲率处梳齿长 = ``length``"，
    因此**不同点之间的梳齿长度可比**（这是唯一诚实的画法）。
    ``scale="local"``：每个采样点的梳齿都被归一化成 ``length``——它只告诉你曲率的
    **方向**，长度信息已经完全丢失，拿它比较"哪里更弯"必然被骗（见 pitfalls）。
    """
    pts = np.atleast_1d(np.asarray(xq, dtype=float))
    vals = piece.evaluate(pts, 0)
    d1 = piece.evaluate(pts, 1)
    kap = curvature_of(piece, pts)
    base = np.stack([pts, vals], axis=1)
    if scale == "local":
        spine = length * np.sign(kap)
    elif scale == "global":
        peak = float(np.max(np.abs(kap))) if kap.size else 0.0
        spine = (length / peak) * kap if peak > 0.0 else np.zeros_like(kap)
    else:
        raise ValueError(f"未知的曲率梳尺度 {scale!r}（可选 global / local）")
    norm_v = np.stack([-d1, np.ones_like(d1)], axis=1) / np.sqrt(1.0 + d1 * d1)[:, None]
    tip = base + norm_v * spine[:, None]
    return base, tip


def curvature_energy(piece: PiecewiseCubic) -> float:
    """曲率能量 :math:`\\int (S'')^2`（:meth:`PiecewiseCubic.energy` 的函数形式）。"""
    return piece.energy()


# --------------------------------------------------------------------------- #
# 4. B 样条：基函数、de Boor、节点插入
# --------------------------------------------------------------------------- #
def clamped_knots(x: Any, degree: int) -> np.ndarray:
    r"""把节点序列 :math:`x_0<\dots<x_m` 变成**夹紧**（clamped）的节点向量。

    两端各重复 ``degree+1`` 次（曲线过端点、端点导数为零阶连续），内部结点保持单重：
    :math:`t = (x_0,\dots,x_0, x_1,\dots,x_{m-1}, x_m,\dots,x_m)`。
    """
    xs = _as_nodes(x)
    k = int(degree)
    if k < 1:
        raise ValueError("次数至少 1")
    if xs.size < k + 1:
        raise ValueError(f"次数 {k} 至少需要 {k + 1} 个节点")
    return np.concatenate([np.full(k + 1, xs[0]), xs[1:-1], np.full(k + 1, xs[-1])])


def uniform_clamped_knots(n_ctrl: int, degree: int, *, t0: float = 0.0, t1: float = 1.0) -> np.ndarray:
    """参数域 ``[t0, t1]`` 上的均匀夹紧节点向量（``n_ctrl`` 个控制点 → ``n_ctrl+degree+1`` 个节点）。"""
    k = int(degree)
    n = int(n_ctrl)
    if k < 1:
        raise ValueError("次数至少 1")
    if n < k + 1:
        raise ValueError(f"次数 {k} 至少需要 {k + 1} 个控制点，得到 {n}")
    n_int = n - k - 1
    inner = np.linspace(t0, t1, n_int + 2)[1:-1] if n_int > 0 else np.zeros(0, dtype=float)
    return np.concatenate([np.full(k + 1, float(t0)), inner, np.full(k + 1, float(t1))])


def knot_span(knots: Any, degree: int, xq: Any) -> np.ndarray:
    r"""de Boor 算法要的区间下标 :math:`\mu`：``t[mu] <= x < t[mu+1]``。

    约定：``x`` 等于最右端点时取最后一个非空区间（:math:`\mu = n_{\rm basis}-1`），
    否则在 :math:`x=t_{\rm last}` 处所有基函数都为 0，分区单位性会假性失败。
    """
    t = np.asarray(knots, dtype=float).ravel()
    k = int(degree)
    n_basis = t.size - k - 1
    pts = np.atleast_1d(np.asarray(xq, dtype=float))
    lo = float(t[0])
    hi = float(t[-1])
    # 允许 1e-9·区间长度 的容差并夹住：有限差分探针会略微越过端点，
    # 而"略微越界"与"真的落在参数域外"必须区别对待。
    tol = 1e-9 * max(hi - lo, 1.0)
    if np.any(pts < lo - tol) or np.any(pts > hi + tol):
        raise ValueError(f"参数 x 落在节点向量之外 [{lo}, {hi}]")
    pts = np.clip(pts, lo, hi)
    span = np.full(pts.size, n_basis - 1, dtype=np.int64)
    for mu in range(k, n_basis):
        span = np.where(pts >= t[mu], mu, span)
    return np.minimum(span, n_basis - 1)


def bspline_basis(i: int, degree: int, knots: Any, xq: Any) -> np.ndarray:
    r"""Cox--de Boor 递推给出的单个 B 样条 :math:`B_{i,k}(x)`（向量化在 ``xq`` 上）。

    约定最右端取闭（:math:`x=t_{\rm last}` 时最后一个非空基函数算 1），
    这样单位分解在闭区间上处处成立。
    """
    t = np.asarray(knots, dtype=float).ravel()
    k = int(degree)
    n_basis = t.size - k - 1
    if not 0 <= i < n_basis:
        raise IndexError(f"基函数下标 {i} 越界（共 {n_basis} 个）")
    pts = np.atleast_1d(np.asarray(xq, dtype=float))
    right_end = pts >= t[-1]
    b = (((pts >= t[i]) & (pts < t[i + 1])) | (right_end & (t[i] < t[i + 1]) & (t[i + 1] >= t[-1]))).astype(float)
    for d in range(1, k + 1):
        b_prev = b
        b_next = _bspline_basis_raw(i + 1, d - 1, t, pts)
        left_den = t[i + d] - t[i]
        right_den = t[i + d + 1] - t[i + 1]
        term_l = (pts - t[i]) / left_den * b_prev if left_den > 0.0 else np.zeros_like(pts)
        term_r = (t[i + d + 1] - pts) / right_den * b_next if right_den > 0.0 else np.zeros_like(pts)
        b = term_l + term_r
    return b


def _bspline_basis_raw(i: int, degree: int, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """:func:`bspline_basis` 的内部递归（不做下标校验，允许 :math:`i=n_{\\rm basis}` 这种边界）。"""
    n_basis = t.size - degree - 1
    if i >= n_basis or i < 0:
        return np.zeros_like(pts)
    right_end = pts >= t[-1]
    b = (((pts >= t[i]) & (pts < t[i + 1])) | (right_end & (t[i] < t[i + 1]) & (t[i + 1] >= t[-1]))).astype(float)
    for d in range(1, degree + 1):
        b_prev = b
        b_next = _bspline_basis_raw(i + 1, d - 1, t, pts)
        left_den = t[i + d] - t[i]
        right_den = t[i + d + 1] - t[i + 1]
        term_l = (pts - t[i]) / left_den * b_prev if left_den > 0.0 else np.zeros_like(pts)
        term_r = (t[i + d + 1] - pts) / right_den * b_next if right_den > 0.0 else np.zeros_like(pts)
        b = term_l + term_r
    return b


def bspline_basis_matrix(knots: Any, degree: int, xq: Any) -> np.ndarray:
    r"""全部基函数在 ``xq`` 上的值，形状 ``(n_basis, n_q)``。

    它是"局部性"最直接的证据：第 :math:`i` 行只在 :math:`[t_i, t_{i+k+1}]` 内非零。
    """
    t = np.asarray(knots, dtype=float).ravel()
    k = int(degree)
    n_basis = t.size - k - 1
    pts = np.atleast_1d(np.asarray(xq, dtype=float))
    out = np.zeros((n_basis, pts.size), dtype=float)
    for i in range(n_basis):
        out[i] = bspline_basis(i, k, t, pts)
    return out


def basis_support(knots: Any, i: int, degree: int) -> tuple[float, float]:
    r"""基函数 :math:`B_{i,k}` 的支撑区间 :math:`[t_i, t_{i+k+1}]`。"""
    t = np.asarray(knots, dtype=float).ravel()
    k = int(degree)
    return float(t[i]), float(t[i + k + 1])


def de_boor(knots: Any, ctrl: Any, degree: int, xq: Any) -> np.ndarray:
    r"""de Boor 算法求 B 样条曲线 :math:`C(x)=\sum_i c_i B_{i,k}(x)` 的值。

    .. math::
        d_j^{[0]} = c_{\mu-k+j},\qquad
        d_j^{[r]} = (1-\alpha_j^{[r]})d_{j-1}^{[r-1]} + \alpha_j^{[r]} d_j^{[r-1]},\quad
        \alpha_j^{[r]} = \frac{x - t_{\mu-k+j}}{t_{\mu+1+j-r}-t_{\mu-k+j}},

    输出 :math:`d_k^{[k]}`。全程只用 :math:`k+1` 个控制点（这就是局部性），
    复杂度 :math:`O(k^2)` 与节点总数无关。
    """
    t = np.asarray(knots, dtype=float).ravel()
    c = np.asarray(ctrl, dtype=float).ravel()
    k = int(degree)
    n_basis = t.size - k - 1
    if c.size != n_basis:
        raise ValueError(f"控制点个数 {c.size} 与节点向量给出的 {n_basis} 不一致")
    pts = np.atleast_1d(np.asarray(xq, dtype=float))
    span = knot_span(t, k, pts)
    d = np.empty((pts.size, k + 1), dtype=float)
    for j in range(k + 1):
        d[:, j] = c[span - k + j]
    for r in range(1, k + 1):
        for j in range(k, r - 1, -1):
            left = t[span - k + j]
            right = t[span + 1 + j - r]
            den = right - left
            alpha = np.where(den > 0.0, (pts - left) / np.where(den > 0.0, den, 1.0), 0.0)
            d[:, j] = (1.0 - alpha) * d[:, j - 1] + alpha * d[:, j]
    return d[:, k]


def bspline_derivative(knots: Any, ctrl: Any, degree: int, order: int = 1) -> tuple[np.ndarray, np.ndarray]:
    r"""B 样条曲线的 :math:`r` 阶导仍是一条 B 样条（**降次**），返回 ``(knots', ctrl')``。

    用标准差分公式

    .. math::
        c_i^{(r)} = \frac{k!}{(k-r)!}\cdot\frac{c_{i+1}^{(r-1)} - c_i^{(r-1)}}{t_{i+k+1-r}-t_{i+1}},

    节点向量去掉两端各一个。求值仍走 :func:`de_boor`，因此导数与函数值是同一套算法。
    """
    t = np.asarray(knots, dtype=float).ravel().copy()
    c = np.asarray(ctrl, dtype=float).ravel().copy()
    k = int(degree)
    r = int(order)
    if r < 0 or r > k:
        raise ValueError(f"求导阶 {r} 必须落在 0..{k}")
    for step in range(1, r + 1):
        deg = k - step + 1  # 本步要微分的次数（差分公式里的 p）
        new_c = np.zeros(c.size - 1, dtype=float)
        for i in range(c.size - 1):
            den = t[i + deg + 1] - t[i + 1]
            new_c[i] = deg * (c[i + 1] - c[i]) / den if den > 0.0 else 0.0
        c = new_c
        t = t[1:-1]
    return t, c


def knot_insert(knots: Any, ctrl: Any, degree: int, u: float) -> tuple[np.ndarray, np.ndarray]:
    r"""Boehm 节点插入：在参数 ``u`` 处插入一个节点，曲线**逐点不变**。

    .. math::
        c'_i = (1-\alpha_i)c_{i-1} + \alpha_i c_i,\qquad
        \alpha_i = \frac{u - t_i}{t_{i+k}-t_i}\quad (i = \mu-k+1,\dots,\mu),

    :math:`\mu` 是 :math:`u` 所在的区间，其余控制点原样保留。插入一个节点只动
    :math:`k` 个控制点、只影响 :math:`k+1` 个区间——这就是"局部修改的局部性"。
    """
    t = np.asarray(knots, dtype=float).ravel()
    c = np.asarray(ctrl, dtype=float).ravel()
    k = int(degree)
    u = float(u)
    if not (t[k] <= u <= t[-k - 1]):
        raise ValueError(f"插入点 {u} 落在参数域 [{t[k]}, {t[-k - 1]}] 之外")
    mu = int(knot_span(t, k, np.array([min(u, t[-k - 1])]))[0])
    new_t = np.concatenate([t[: mu + 1], np.array([u]), t[mu + 1 :]])
    new_c = np.empty(c.size + 1, dtype=float)
    for i in range(new_c.size):
        if i <= mu - k:
            new_c[i] = c[i]
        elif i >= mu + 1:
            new_c[i] = c[i - 1]
        else:
            den = t[i + k] - t[i]
            alpha = (u - t[i]) / den if den > 0.0 else 0.0
            new_c[i] = (1.0 - alpha) * c[i - 1] + alpha * c[i]
    return new_t, new_c


def insertion_sequence(knots: Any, ctrl: Any, degree: int, u: float, times: int) -> list[tuple[np.ndarray, np.ndarray]]:
    r"""连续插入同一个参数 ``u`` ``times`` 次，返回每一步的 ``(knots, ctrl)``（含第 0 步）。

    给"节点插入动画"用：每一步曲线都不变（差 :math:`O(u)`），变的是**表示**——
    控制多边形越来越贴近曲线，这就是"插入节点 = 加密控制网"。
    """
    if int(times) < 0:
        raise ValueError("插入次数不能为负")
    out: list[tuple[np.ndarray, np.ndarray]] = [(np.asarray(knots, dtype=float).copy(), np.asarray(ctrl, dtype=float).copy())]
    t = np.asarray(knots, dtype=float).copy()
    c = np.asarray(ctrl, dtype=float).copy()
    for _ in range(int(times)):
        t, c = knot_insert(t, c, degree, u)
        out.append((t.copy(), c.copy()))
    return out


def bspline_uniform_natural_ctrl(y: Any) -> tuple[np.ndarray, np.ndarray]:
    r"""均匀节点 :math:`0,1,\dots,m` 上**自然三次样条**的 B 样条控制点（三对角系统！）。

    取基函数 :math:`N_j` 为以 :math:`j` 为中心的均匀三次 B 样条
    （:math:`N(0)=4/6`, :math:`N(\pm1)=1/6`，支撑 :math:`[j-2,j+2]`），
    控制点 :math:`c_{-1},\dots,c_{m+1}`。由两个恒等式

    .. math::
        S(x_i) = \tfrac16 c_{i-1} + \tfrac46 c_i + \tfrac16 c_{i+1} = y_i,\qquad
        M_i = S''(x_i) = c_{i-1} - 2c_i + c_{i+1}

    与自然边界 :math:`M_0=M_m=0` 得 :math:`c_0=y_0`、:math:`c_m=y_m`（端点方程与插值
    方程联立后 :math:`c_{\mp1}` 被消掉），剩下的是**严格三对角**系统

    .. math::
        \tfrac16 c_{i-1} + \tfrac46 c_i + \tfrac16 c_{i+1} = y_i,\quad i=1,\dots,m-1,

    用追赶法求解；:math:`c_{-1}=2c_0-c_1`、:math:`c_{m+1}=2c_m-c_{m-1}` 事后还原。
    于是"解三对角系统"与"de Boor 求值"给出**同一条曲线**——这是本章最硬的一条交叉验证。
    """
    ys = np.asarray(y, dtype=float).ravel()
    if ys.size < 2:
        raise ValueError("至少 2 个数据点")
    if not np.all(np.isfinite(ys)):
        raise ValueError("数据必须是有限数")
    m = ys.size - 1
    ctrl = np.zeros(m + 3, dtype=float)
    ctrl[1] = ys[0]
    ctrl[m + 1] = ys[m]
    if m - 1 > 0:
        size = m - 1
        lower = np.full(size, 1.0 / 6.0, dtype=float)
        diag = np.full(size, 4.0 / 6.0, dtype=float)
        upper = np.full(size, 1.0 / 6.0, dtype=float)
        rhs = ys[1:m].copy()
        rhs[0] -= ys[0] / 6.0
        rhs[-1] -= ys[m] / 6.0
        ctrl[2 : m + 1] = tridiagonal_apply(lower, diag, upper, rhs)
    ctrl[0] = 2.0 * ctrl[1] - ctrl[2]
    ctrl[m + 2] = 2.0 * ctrl[m + 1] - ctrl[m]
    knots = np.arange(-3.0, m + 4.0, dtype=float)
    return knots, ctrl


def convex_hull_gap(knots: Any, ctrl: Any, degree: int, xq: Any) -> float:
    r"""曲线越出控制点凸包的最大距离（应当**恒为 0**，这是 B 样条的凸包性质）。

    一维"凸包"就是 :math:`[\min_i c_i, \max_i c_i]`，所以这里量的是
    :math:`\max(0,\ C(x)-\max c,\ \min c - C(x))` 的最大值。
    """
    c = np.asarray(ctrl, dtype=float).ravel()
    vals = de_boor(knots, c, degree, xq)
    lo = float(np.min(c))
    hi = float(np.max(c))
    gap = np.maximum(0.0, np.maximum(vals - hi, lo - vals))
    return float(np.max(gap)) if gap.size else 0.0


# --------------------------------------------------------------------------- #
# 5. 对照用高阶多项式（本章自带的最小实现，见模块 docstring）
# --------------------------------------------------------------------------- #
def barycentric_weights(x: Any) -> np.ndarray:
    r"""重心插值权 :math:`w_j = 1/\prod_{k\ne j}(x_j-x_k)`。

    先在缩放变量 :math:`z=(x-a)/(b-a)` 上算：缩放把所有权同乘一个常数
    :math:`(b-a)^{-(n-1)}`，而重心公式对权的**公共因子不变**，所以这样既不改变结果，
    又避免了直接连乘的溢出/下溢（等距节点上 :math:`w_j` 能差几十个数量级）。
    """
    xs = _as_nodes(x)
    n = xs.size
    lo = float(xs[0])
    hi = float(xs[-1])
    z = (xs - lo) / (hi - lo)
    w = np.ones(n, dtype=float)
    for j in range(n):
        prod = 1.0
        for k in range(n):
            if k != j:
                prod *= z[j] - z[k]
        w[j] = 1.0 / prod
    return w


def lagrange_value(x: Any, y: Any, xq: Any, *, weights: Any = None) -> np.ndarray:
    r"""过 :math:`(x_i,y_i)` 的 Lagrange 插值多项式在 ``xq`` 处的值（重心公式）。

    .. math::
        p(x) = \frac{\sum_j \frac{w_j y_j}{x-x_j}}{\sum_j \frac{w_j}{x-x_j}},

    :math:`O(n)` 求值、数值上远比解 Vandermonde 系统稳（但**不是**无条件稳：
    :math:`\sum_j|w_j/(x-x_j)|` 的规模正是 Lebesgue 常数，节点越差它越大）。
    """
    xs = _as_nodes(x)
    ys = _as_values(y, xs.size)
    w = barycentric_weights(xs) if weights is None else np.asarray(weights, dtype=float).ravel()
    pts = np.atleast_1d(np.asarray(xq, dtype=float))
    out = np.empty(pts.size, dtype=float)
    for m in range(pts.size):
        d = pts[m] - xs
        hit = np.flatnonzero(d == 0.0)
        if hit.size:
            out[m] = ys[int(hit[0])]
            continue
        t = w / d
        out[m] = float(np.sum(t * ys) / np.sum(t))
    return out


def lagrange_deriv2(x: Any, y: Any, xq: Any, *, weights: Any = None) -> np.ndarray:
    r"""Lagrange 插值多项式的二阶导 :math:`p''(x_q)`（重心公式，:math:`O(n)`）。

    由 :math:`p=N/D`、:math:`N=\sum_j w_jy_j/(x-x_j)`、:math:`D=\sum_j w_j/(x-x_j)` 得

    .. math::
        p'' = \frac{N'' - 2p'D' - pD''}{D},\qquad
        N'' = 2\sum_j\frac{w_jy_j}{(x-x_j)^3},\quad D'' = 2\sum_j\frac{w_j}{(x-x_j)^3}.

    落在节点上时按 :math:`10^{-9}` 倍区间长度侧移（重心公式在节点处是 0/0）。
    """
    xs = _as_nodes(x)
    ys = _as_values(y, xs.size)
    w = barycentric_weights(xs) if weights is None else np.asarray(weights, dtype=float).ravel()
    pts = np.atleast_1d(np.asarray(xq, dtype=float)).copy()
    span = float(xs[-1] - xs[0])
    for m in range(pts.size):
        if np.any(pts[m] == xs):
            pts[m] = pts[m] + 1e-9 * span
    out = np.empty(pts.size, dtype=float)
    for m in range(pts.size):
        d = pts[m] - xs
        t = w / d
        den = float(np.sum(t))
        num = float(np.sum(t * ys))
        p = num / den
        dp = float(np.sum(-t / d * ys) - p * np.sum(-t / d)) / den
        d2n = float(np.sum(2.0 * t / (d * d) * ys))
        d2d = float(np.sum(2.0 * t / (d * d)))
        out[m] = (d2n - 2.0 * dp * float(np.sum(-t / d)) - p * d2d) / den
    return out


def lebesgue_curve(nodes: Any, xs: Any) -> np.ndarray:
    r"""Lebesgue 函数 :math:`\Lambda(x)=\sum_j|\ell_j(x)|` 在 ``xs`` 上的值。

    它把"插值误差"与"数据误差"分开：:math:`\|p-f\|_\infty\le(1+\Lambda_n)\|p^*-f\|_\infty`。
    实现上直接用 :func:`lagrange_value` 取 :math:`y=e_j` 再取绝对值求和——
    定义怎么写就怎么算，不引入额外的公式。
    """
    nd = np.asarray(nodes, dtype=float).ravel()
    pts = np.atleast_1d(np.asarray(xs, dtype=float))
    w = barycentric_weights(nd)
    total = np.zeros(pts.size, dtype=float)
    for j in range(nd.size):
        e = np.zeros(nd.size, dtype=float)
        e[j] = 1.0
        total += np.abs(lagrange_value(nd, e, pts, weights=w))
    return total


def lebesgue_constant(nodes: Any, *, n_grid: int = 2001) -> float:
    """Lebesgue 常数 :math:`\\Lambda_n=\\max_x\\sum_j|\\ell_j(x)|`（网格实测）。"""
    nd = np.asarray(nodes, dtype=float).ravel()
    xs = np.linspace(nd[0], nd[-1], int(n_grid))
    return float(np.max(lebesgue_curve(nd, xs)))


def gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    r"""``n`` 点 Gauss--Legendre 节点与权重（牛顿迭代 + Legendre 三项递推）。

    .. math::
        P_{j+1}(x) = \frac{(2j+1)xP_j(x)-jP_{j-1}(x)}{j+1},\qquad
        P_n'(x) = \frac{n\,(xP_n-P_{n-1})}{x^2-1},\qquad
        w_k = \frac{2}{(1-x_k^2)P_n'(x_k)^2}.

    本章只需要它做**多项式能量**的精确积分（:math:`(p'')^2` 的次数不超过
    :math:`2n-6`，:math:`n` 点 Gauss 对次数 :math:`2n-1` 精确，故一次即准）。
    """
    m = int(n)
    if m < 1:
        raise ValueError("Gauss 点数至少 1")
    k = np.arange(1, m + 1, dtype=float)
    x = np.cos(np.pi * (k - 0.25) / (m + 0.5))
    for _ in range(100):
        p0 = np.ones_like(x)
        p1 = x.copy()
        for j in range(2, m + 1):
            p2 = ((2.0 * j - 1.0) * x * p1 - (j - 1.0) * p0) / j
            p0, p1 = p1, p2
        dp = m * (x * p1 - p0) / (x * x - 1.0)
        dx = -p1 / dp
        x = x + dx
        if float(np.max(np.abs(dx))) < 1e-15:
            break
    w = 2.0 / ((1.0 - x * x) * dp * dp)
    order = np.argsort(x)
    return x[order], w[order]


def interpolating_polynomial_energy(x: Any, y: Any, *, n_quad: int | None = None) -> float:
    r"""插值多项式的曲率能量 :math:`\int_a^b (p'')^2\,\mathrm{d}x`（Gauss 精确）。

    :math:`(p'')^2` 的次数不超过 :math:`2(n-3)`（:math:`n` = 数据点个数），而 :math:`m` 点
    Gauss--Legendre 对次数 :math:`2m-1` 精确——所以只要 :math:`m\ge n-2`（缺省
    ``max(n+2, 16)`` 满足）这里的积分就**不是数值近似**，而是精确值（直到舍入）。
    它给"样条能量最小"这条变分性质提供了可比较的对手。
    """
    xs = _as_nodes(x)
    ys = _as_values(y, xs.size)
    if xs.size < 4:
        raise ValueError("至少 4 个点才能谈插值多项式的能量")
    m = int(n_quad) if n_quad is not None else max(xs.size + 2, 16)
    nodes, weights = gauss_legendre(m)
    a = float(xs[0])
    b = float(xs[-1])
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    pts = mid + half * nodes
    d2 = lagrange_deriv2(xs, ys, pts)
    return float(half * np.sum(weights * d2 * d2))


# --------------------------------------------------------------------------- #
# 6. 常用测试数据
# --------------------------------------------------------------------------- #
def _f_runge(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + 25.0 * x * x)


def _f_sine(x: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x)


def _f_bump(x: np.ndarray) -> np.ndarray:
    return np.exp(-16.0 * x * x)


def _f_cusp(x: np.ndarray) -> np.ndarray:
    return np.abs(x - 0.25) + 0.2 * x


def _f_monotone(x: np.ndarray) -> np.ndarray:
    return x + 0.4 * np.sin(0.5 * np.pi * x)


#: 本章用到的解析测试函数（定义域一律 ``[-1, 1]``，除 ``cusp`` 外都解析）。
TEST_FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "runge": _f_runge,
    "sine": _f_sine,
    "bump": _f_bump,
    "cusp": _f_cusp,
    "monotone": _f_monotone,
}


def sample_function(name: str, x: Any) -> np.ndarray:
    """在 ``x`` 上采样 :data:`TEST_FUNCTIONS` 里的解析函数。"""
    if name not in TEST_FUNCTIONS:
        raise KeyError(f"未知测试函数 {name!r}（可选 {sorted(TEST_FUNCTIONS)}）")
    pts = np.asarray(x, dtype=float)
    return TEST_FUNCTIONS[name](pts)


def uniform_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """``[a,b]`` 上的 ``n`` 个等距节点（Runge 现象与样条都要用同一套节点才可比较）。"""
    m = int(n)
    if m < 2:
        raise ValueError("至少 2 个节点")
    return np.linspace(float(a), float(b), m)


def chebyshev_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    r"""``[a,b]`` 上的 ``n`` 个 Chebyshev--Lobatto 节点（**递增**）：

    .. math::
        x_k = \frac{a+b}{2} + \frac{b-a}{2}\Big(-\cos\frac{k\pi}{n-1}\Big),\quad k=0,\dots,n-1 .

    端点必取到，所以它能和等距节点在**同一批端点**上直接对比（Runge 现象的公平对照）。
    """
    m = int(n)
    if m < 2:
        raise ValueError("至少 2 个节点")
    k = np.arange(m, dtype=float)
    z = -np.cos(np.pi * k / (m - 1))
    return 0.5 * (a + b) + 0.5 * (b - a) * z
