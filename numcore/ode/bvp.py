r"""3.4 章：两点边值问题与 Chebyshev 谱方法的数值内核。

数学对象
--------

本章处理**两点边值问题**（BVP）

.. math::
    u''+p(x)u'+q(x)u=r(x),\qquad x\in[a,b],\qquad
    \mathcal B_a u=\alpha,\quad \mathcal B_b u=\beta,

其中 :math:`\mathcal B` 是 Dirichlet（:math:`u`）或 Neumann（:math:`u'`）算子；另有一个
非线性算例 :math:`u''=f(x,u)`（Bratu 型）。三条互不相同的数值路线都在这里：

1. **打靶法**（:func:`shooting`）：把 BVP 换成初值问题，用 Newton 迭代求初始斜率
   :math:`s=u'(a)` 使 :math:`u(b;s)=\beta`。Jacobian 由**变分方程**
   :math:`z'=J(x,y)z,\ z(a)=(0,1)` 积分得到（不是差分近似）。初值问题的单步推进
   **复用 3.1 章的** :func:`numcore.ode.rk.rk_step`（显式 RK，表驱动）。
2. **有限差分**（:func:`fd_bvp`）：内部二阶中心差商；**边界条件的处理单独可调**
   （``bc_order=1/2``）——本章最容易被读错的数字就在这里，见 :func:`bc_residuals`。
3. **Chebyshev 谱方法**：配置法（:func:`cheb_collocation`，把微分矩阵的行**搬进**方程组）
   与 tau 法（:func:`cheb_tau`，在系数空间里投影，边界行单独替换）。

Chebyshev 侧的真源是微分矩阵 :math:`D_N`（:func:`cheb_diff_matrix`，Trefethen 的
显式公式）与系数空间的一阶/二阶微分算子（:func:`coeff_first_matrix` /
:func:`coeff_second_matrix`）：

.. math::
    (u')_k=\frac{2}{c_k}\sum_{\substack{j>k\\ j+k\ \text{奇}}}j\,a_j,\qquad
    (u'')_k=\frac{1}{c_k}\sum_{\substack{j>k+1\\ j+k\ \text{偶}}}j(j^2-k^2)\,a_j,

:math:`c_0=c_N=2`、其余 :math:`c_k=1`。两条公式都由 ``tests/3.4`` 与
:math:`D_N`、:math:`D_N^2` 逐点对拍（不是抄来的信任）。

三条不许撒谎的约定
------------------
* **非有限值不出内核**：发散一律在 :data:`BLOWUP` 处显式停止并置掩码；显示上限
  :data:`DISPLAY_CEIL` 只用于"看着超出量程"的占位，且随值给出 ``*_clip`` 掩码。
* **稠密矩阵是现实**：:math:`D_N` 是稠密的、:math:`\|D_N\|_\infty=N^2`、去零空间后的
  条件数按 :math:`O(N^2)` 增长（:func:`diff_matrix_diagnostics` 实测）。因此**小的**
  :math:`N` 才用谱方法直接解方程组；:func:`fd_matrix` 走另一条路（稀疏结构 + 自研 LU）。
  线性系统一律调用 :mod:`numcore.linalg.lu`（P2/P3：不碰任何现成求解器）。
* **误差与残差分开报**：配置法在节点上的方程残差**恒为** 0（那就是它的定义），
  真正的精度要看节点之间（:func:`spectral_residual` 在细网格上给的是这个）。
  有限差分的离散残差是 :math:`\sim u` 级，而解的误差可以是 :math:`10^{13}` 倍——
  「残差小」与「解准」是两件事，本章用实测数字把这句话钉死。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from numcore.linalg.cond import norm_inf as _norm_inf
from numcore.linalg.lu import lu_decompose, solve_lu
from numcore.linalg.svd import jacobi_svd
from numcore.ode.rk import rk_step

__all__ = [
    "BLOWUP",
    "DISPLAY_CEIL",
    "UNIT_ROUNDOFF",
    "BVPProblem",
    "PROBLEM_LABELS",
    "problem",
    "problem_names",
    "cheb_nodes",
    "cheb_nodes_on",
    "domain_scale",
    "to_reference",
    "lobatto_scale",
    "cheb_diff_matrix",
    "cheb_second_matrix",
    "barycentric_weights",
    "barycentric_interpolate",
    "piecewise_linear",
    "sample_solution",
    "clenshaw",
    "cheb_transform",
    "clenshaw_curtis",
    "coeff_first_matrix",
    "coeff_second_matrix",
    "cheb_derivative_coeffs",
    "cheb_series_values",
    "node_spacing",
    "node_density",
    "cluster_ratio",
    "outer_node_count",
    "ShootingResult",
    "shooting",
    "shooting_scan",
    "shooting_sensitivity",
    "shooting_endpoint",
    "FDResult",
    "fd_matrix",
    "fd_bvp",
    "SpectralResult",
    "collocation_matrix",
    "cheb_collocation",
    "tau_matrix",
    "cheb_tau",
    "bvp_operator",
    "spectral_residual",
    "max_error",
    "interpolation_error",
    "default_grid",
    "interior_error",
    "bc_residuals",
    "operator_diagnostics",
    "diff_matrix_diagnostics",
    "convergence_study",
    "fit_exponential",
    "fit_power",
    "roundoff_floor",
    "amplification_digits",
]

#: 初值问题的爆炸上限：超过它就停止推进并如实标注（:math:`10^{150}` 远早于 double 的
#: :math:`1.8\times10^{308}` 溢出，所以"停下"总是发生在 ``inf`` 出现**之前**）。
#: 显示用的截断另有其值（:data:`DISPLAY_CEIL`）——"算不动了"与"画不下"是两件事。
BLOWUP = 1.0e150

#: 帧里"看着超出量程"的占位值（配 ``*_clip`` 掩码使用；不是真值）。
DISPLAY_CEIL = 1.0e6

#: 单位舍入 :math:`u=\varepsilon/2`（与 1.2 / 3.2 章同口径）。
UNIT_ROUNDOFF = float(np.finfo(float).eps) / 2.0

#: 有限差分内部格式固定二阶；边界格式由 ``bc_order`` 选择（1 = 单侧一阶，2 = 单侧二阶）。
FD_INTERIOR_ORDER = 2


# --------------------------------------------------------------------------- #
# 1. Chebyshev 机器：节点、微分矩阵、系数空间算子
# --------------------------------------------------------------------------- #
def cheb_nodes(n: int) -> np.ndarray:
    r"""Chebyshev--Lobatto（极值）节点 :math:`x_j=\cos(j\pi/N)`，**降序** :math:`x_0=1,\ x_N=-1`。

    降序是 Trefethen 的约定，也是本章配置法矩阵的约定：下标 0 是右端点、下标 ``N`` 是左端点。
    这一点在边界行替换时最容易写反（``tests/3.4`` 有一条断言专门钉端点归属）。
    """
    if n < 1:
        raise ValueError(f"Chebyshev 至少需要 N=1，得到 {n}")
    return np.cos(np.pi * np.arange(n + 1, dtype=float) / float(n))


def domain_scale(problem: "BVPProblem") -> float:
    r"""区间 :math:`[a,b]` 到参考区间 :math:`[-1,1]` 的伸缩因子 :math:`s=2/(b-a)`。

    :math:`D_N` 是**参考区间**上的微分矩阵：物理坐标下的导数要乘 :math:`s`
    （一阶乘 :math:`s`、二阶乘 :math:`s^2`，链式法则）。本章最经典的静默错误就是
    忘了这个因子——解在 :math:`[0,1]` 上时会整体差 4 倍。
    """
    return 2.0 / (problem.b - problem.a)


def to_reference(problem: "BVPProblem", x: Any) -> np.ndarray:
    r"""物理坐标 :math:`x\in[a,b]` 映到参考坐标 :math:`\xi\in[-1,1]`。"""
    a, b = problem.x_span
    xx = np.asarray(x, dtype=float)
    return (2.0 * xx - (a + b)) / (b - a)


def cheb_nodes_on(problem: "BVPProblem", n: int) -> np.ndarray:
    r"""问题区间上的 Chebyshev--Lobatto 节点（降序：``x[0]=b``、``x[n]=a``）。"""
    a, b = problem.x_span
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    return mid + half * cheb_nodes(n)


def cheb_diff_matrix(n: int) -> np.ndarray:
    r""":math:`(N+1)\times(N+1)` Chebyshev 配置微分矩阵 :math:`D_N`（:math:`u'\approx D_Nu`）。

    用显式公式（Trefethen, *Spectral Methods in MATLAB*, Ch. 6）：非对角元

    .. math::
        D_{ij}=\frac{c_i}{c_j}\frac{(-1)^{i+j}}{x_i-x_j},\qquad c_0=c_N=2,\ c_k=1,

    对角元取 :math:`D_{ii}=-\sum_{j\ne i}D_{ij}`——这一行把**常数在零空间里**这件事
    写死成 :math:`D_N\mathbf 1=0`（所以 :math:`D_N` 奇异，谈"条件数"必须先挖掉零空间）。
    对角元**不是** 0：由行和为零反解出 :math:`D_{00}=(2N^2+1)/6`、:math:`D_{NN}=-(2N^2+1)/6`
    （本章实测 N=8 时 21.5、N=16 时 85.5，与闭式逐位吻合）。
    """
    if n < 1:
        raise ValueError(f"Chebyshev 至少需要 N=1，得到 {n}")
    x = cheb_nodes(n)
    c = np.ones(n + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1.0) ** np.arange(n + 1))
    dx = x[:, None] - x[None, :]
    D = np.outer(c, 1.0 / c) / (dx + np.eye(n + 1))
    return D - np.diag(D.sum(axis=1))


def cheb_second_matrix(n: int) -> np.ndarray:
    r"""二阶微分矩阵 :math:`D_N^2`（配置法里 :math:`u''\approx D_N^2u`）。

    注意它是 :math:`D_N` 的**乘积**而不是独立的二阶公式：两者在配置意义下等价，
    但乘积会把 :math:`D_N` 的舍入误差放大 :math:`\|D_N\|_\infty=N^2` 倍，这就是
    "谱方法撞舍入地板"的机制之一。
    """
    D = cheb_diff_matrix(n)
    return D @ D


def barycentric_weights(n: int) -> np.ndarray:
    r"""Chebyshev--Lobatto 节点的重心插值权 :math:`w_j=(-1)^j\delta_j`（两端各折半）。

    重心公式让"把节点上的值插到任意点"变成 :math:`O(N)`——本章所有"细网格上的误差/残差"
    都靠它把离散解延拓成函数，而不是靠眼睛连折线。
    """
    w = (-1.0) ** np.arange(n + 1, dtype=float)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def barycentric_interpolate(x_nodes: Any, values: Any, xq: Any) -> np.ndarray:
    r"""重心插值：由 :math:`(x_j,v_j)` 求 :math:`x_q` 处的值（向量化，命中节点时取原值）。"""
    xn = np.asarray(x_nodes, dtype=float).reshape(-1)
    vn = np.asarray(values, dtype=float).reshape(-1)
    if xn.size != vn.size:
        raise ValueError(f"节点与值的长度不一致：{xn.size} vs {vn.size}")
    n = xn.size - 1
    w = barycentric_weights(n) if _is_lobatto(xn) else _generic_weights(xn)
    q = np.asarray(xq, dtype=float).reshape(-1)
    num = np.zeros_like(q)
    den = np.zeros_like(q)
    for j in range(xn.size):
        d = q - xn[j]
        hit = np.abs(d) <= 0.0
        d = np.where(hit, 1.0, d)
        t = w[j] / d
        num = num + t * vn[j]
        den = den + t
    out = num / den
    for j in range(xn.size):
        hit = q == xn[j]
        if np.any(hit):
            out = np.where(hit, vn[j], out)
    return out


def piecewise_linear(x_nodes: Any, values: Any, xq: Any) -> np.ndarray:
    r"""分段线性采样：**等距网格**上把离散解搬到细网格时用它，而不是全局多项式插值。

    理由是一条实测出来的硬事实：:math:`n` 个等距节点上的多项式插值，Lebesgue 常数按
    :math:`\sim2^n/n` 增长；:math:`n=50` 时连**直线**都插不准（误差 :math:`1.5\times10^{-3}`）。
    重心公式本身是稳定的，不稳定的是"等距节点上的多项式插值"这个问题本身。
    """
    xn = np.asarray(x_nodes, dtype=float).reshape(-1)
    vn = np.asarray(values, dtype=float).reshape(-1)
    q = np.asarray(xq, dtype=float).reshape(-1)
    if xn.size != vn.size:
        raise ValueError(f"节点与值长度不一致：{xn.size} vs {vn.size}")
    if xn.size < 2:
        return np.full_like(q, float(vn[0]) if vn.size else 0.0)
    asc = xn[1] > xn[0]
    xs = xn if asc else xn[::-1]
    vs = vn if asc else vn[::-1]
    idx = np.clip(np.searchsorted(xs, q) - 1, 0, xs.size - 2)
    x0 = xs[idx]
    x1 = xs[idx + 1]
    t = (q - x0) / (x1 - x0)
    return vs[idx] * (1.0 - t) + vs[idx + 1] * t


def sample_solution(x_nodes: Any, values: Any, xq: Any, *, kind: str = "auto") -> np.ndarray:
    r"""把离散解采到细网格：Chebyshev 网格走重心插值，其它网格走分段线性（``kind`` 可强制）。"""
    xn = np.asarray(x_nodes, dtype=float).reshape(-1)
    if kind == "linear":
        return piecewise_linear(xn, values, xq)
    if kind == "barycentric":
        return barycentric_interpolate(xn, values, xq)
    if kind == "auto":
        return barycentric_interpolate(xn, values, xq) if lobatto_scale(xn) is not None else piecewise_linear(xn, values, xq)
    raise ValueError(f"未知采样方式 {kind!r}")


def _is_lobatto(xn: np.ndarray) -> bool:
    n = xn.size - 1
    if n < 1:
        return False
    return bool(np.allclose(xn, cheb_nodes(n), rtol=0.0, atol=1e-14))


def lobatto_scale(xn: Any) -> float | None:
    r"""若 ``xn`` 是某个区间上的 Chebyshev--Lobatto 网格，返回伸缩因子 :math:`s`，否则 ``None``。

    认网格这件事必须"允许平移与缩放"：本章的问题既在 :math:`[-1,1]`（``smooth``/``abs3``）
    也在 :math:`[0,1]` 与 :math:`[0,L]` 上。只比较 :math:`|x_j|` 与 :math:`\cos(j\pi/N)`
    会把 :math:`[0,1]` 上的谱网格误判成"普通网格"，于是边界残差悄悄退化成
    "用 Chebyshev 间距当 h 的单侧差商"，算出 1.0 这种荒唐数字（本章真踩过）。
    """
    x = np.asarray(xn, dtype=float).reshape(-1)
    n = x.size - 1
    if n < 1 or x[0] == x[-1]:
        return None
    mid = 0.5 * (x[0] + x[-1])
    half = 0.5 * (x[0] - x[-1])
    xi = (x - mid) / half
    if np.allclose(xi, cheb_nodes(n), rtol=0.0, atol=1e-12):
        return float(2.0 / abs(x[0] - x[-1]))
    return None


def _generic_weights(xn: np.ndarray) -> np.ndarray:
    r"""一般节点的重心权（本章只用于等距等非 Lobatto 网格）。

    直接算 :math:`\prod_{i\ne j}(x_j-x_i)` 再取倒数会**下溢成 0 再除零**：
    801 个等距节点的乘积是 :math:`h^{800}` 量级，double 里就是 0（本章实测当场
    ``ZeroDivisionError``）。所以走对数：:math:`\log|w_j|=-\sum_{i\ne j}\log|x_j-x_i|`
    再整体平移最大值（重心公式对公共因子不变），符号由节点排序给出。
    """
    n = xn.size
    logs = np.zeros(n)
    for j in range(n):
        d = np.abs(xn[j] - np.delete(xn, j))
        logs[j] = -float(np.sum(np.log(d)))
    logs -= float(np.max(logs))
    sign = (-1.0) ** np.arange(n, dtype=float)
    if xn.size > 1 and xn[1] < xn[0]:
        sign = sign[::-1].copy()
    return sign * np.exp(logs)


def clenshaw(coeffs: Any, x: Any) -> np.ndarray:
    r"""Clenshaw 递推求 :math:`\sum_k a_kT_k(x)`（不构造单项式基，避免系数病态放大）。"""
    a = np.asarray(coeffs, dtype=float).reshape(-1)
    q = np.asarray(x, dtype=float)
    b1 = np.zeros_like(q)
    b2 = np.zeros_like(q)
    for k in range(a.size - 1, 0, -1):
        b1, b2 = 2.0 * q * b1 - b2 + a[k], b1
    return q * b1 - b2 + a[0]


def cheb_transform(values: Any) -> np.ndarray:
    r"""极值网格上的离散 Chebyshev 变换（DCT-I）：节点值 :math:`\to` 系数 :math:`a_k`。

    .. math::
        a_k=\frac{2}{Nc_k}\sum_{j=0}^{N}\frac{1}{c_j}v_j\cos\frac{\pi jk}{N},

    :math:`c_0=c_N=2`、其余 1。与 :func:`clenshaw` 互逆（``tests/3.4`` 逐系数对拍）。
    这里**不调用** FFT（P2），:math:`O(N^2)` 的显式求和足够本章规模。
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    mm = v.size - 1
    if mm < 1:
        raise ValueError(f"至少需要两个节点值，得到 {v.size}")
    c = np.ones(mm + 1)
    c[0] = 2.0
    c[-1] = 2.0
    j = np.arange(mm + 1, dtype=float)
    theta = np.pi * np.outer(j, j) / float(mm)
    a = (2.0 / (mm * c)) * (np.cos(theta) @ (v / c))
    return a


def clenshaw_curtis(n: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Clenshaw--Curtis 求积：返回 :math:`(x_j,w_j)`，对 :math:`\deg\le N` 的多项式精确。

    公式（Trefethen, *Spectral Methods in MATLAB*, p. 128）里的两个端点权单独处理；
    ``tests/3.4`` 用 :math:`\int x^m` 的闭式值逐阶对拍到 :math:`m=N`。
    """
    if n < 1:
        raise ValueError(f"至少需要 N=1，得到 {n}")
    x = cheb_nodes(n)
    theta = np.pi * np.arange(n + 1, dtype=float) / float(n)
    w = np.zeros(n + 1)
    if n == 1:
        return x, np.ones(2)
    v = np.ones(n - 1)
    if n % 2 == 0:
        w[0] = 1.0 / (n * n - 1.0)
        w[n] = w[0]
        for k in range(1, n // 2):
            v = v - 2.0 * np.cos(2.0 * k * theta[1:n]) / (4.0 * k * k - 1.0)
        v = v - np.cos(n * theta[1:n]) / (n * n - 1.0)
    else:
        w[0] = 1.0 / (n * n)
        w[n] = w[0]
        for k in range(1, (n + 1) // 2):
            v = v - 2.0 * np.cos(2.0 * k * theta[1:n]) / (4.0 * k * k - 1.0)
    w[1:n] = 2.0 * v / n
    return x, w


def coeff_first_matrix(n: int) -> np.ndarray:
    r"""系数空间的一阶微分算子 :math:`M_1`：:math:`(u')_k=\frac{2}{c_k}\sum_{j>k,\,j+k\ \text{奇}}j\,a_j`。"""
    if n < 1:
        raise ValueError(f"至少需要 N=1，得到 {n}")
    M = np.zeros((n + 1, n + 1))
    for k in range(n + 1):
        ck = 2.0 if (k == 0 or k == n) else 1.0
        for j in range(k + 1, n + 1):
            if (j + k) % 2 == 1:
                M[k, j] = 2.0 * j / ck
    return M


def coeff_second_matrix(n: int) -> np.ndarray:
    r"""系数空间的二阶微分算子 :math:`M_2`：:math:`(u'')_k=\frac{1}{c_k}\sum_{j>k+1,\,j+k\ \text{偶}}j(j^2-k^2)a_j`。

    这条公式在 :math:`k=0` 处的二分之一因子（:math:`c_0=2`）是最经典的抄错点：
    漏掉它时 :math:`M_2a` 与 :math:`D_N^2u` 的偏差可以大到 :math:`10^2` 量级
    （``tests/3.4`` 把两支都算出来做负对照）。
    """
    if n < 1:
        raise ValueError(f"至少需要 N=1，得到 {n}")
    M = np.zeros((n + 1, n + 1))
    for k in range(n + 1):
        ck = 2.0 if (k == 0 or k == n) else 1.0
        for j in range(k + 2, n + 1):
            if (j + k) % 2 == 0:
                M[k, j] = j * (j * j - k * k) / ck
    return M


def cheb_derivative_coeffs(coeffs: Any, order: int = 1) -> np.ndarray:
    """系数空间求导（``order`` 只能取 1 或 2；更高的阶请连续调用）。"""
    a = np.asarray(coeffs, dtype=float).reshape(-1)
    n = a.size - 1
    if order == 1:
        return coeff_first_matrix(n) @ a
    if order == 2:
        return coeff_second_matrix(n) @ a
    raise ValueError(f"只支持一阶/二阶，得到 order={order}")


def cheb_series_values(coeffs: Any, xq: Any, order: int = 0) -> np.ndarray:
    """把系数表示的 Chebyshev 级数（可先求 ``order`` 阶导）在 ``xq`` 上求值。"""
    a = np.asarray(coeffs, dtype=float).reshape(-1)
    for _ in range(int(order)):
        a = cheb_derivative_coeffs(a, 1)
    return clenshaw(a, xq)


# --------------------------------------------------------------------------- #
# 2. 节点聚集的几何量（viz 3 的真源）
# --------------------------------------------------------------------------- #
def node_spacing(n: int) -> np.ndarray:
    r"""相邻节点间距 :math:`|x_{j+1}-x_j|`（降序网格，长度 :math:`N`）。"""
    x = cheb_nodes(n)
    return np.abs(np.diff(x))


def node_density(n: int, xq: Any) -> np.ndarray:
    r"""节点密度 :math:`\rho(x)=\frac{N}{\pi\sqrt{1-x^2}}`（单位长度的节点数）。

    它是 :math:`x=\cos\theta`、:math:`\theta` 均匀时的分布密度：把 :math:`N+1` 个点按
    :math:`\arccos` 均匀铺开，投影回区间就得到端点处的 :math:`1/\sqrt{1-x^2}` 聚集。
    """
    q = np.asarray(xq, dtype=float)
    s = np.sqrt(np.maximum(1.0 - q * q, 0.0))
    return float(n) / (np.pi * np.maximum(s, 1.0 / float(n)))


def cluster_ratio(n: int) -> float:
    r"""最大间距 / 最小间距：一维"聚集程度"的标量度量（等距网格恒为 1）。

    Chebyshev 网格上它随 :math:`N` **线性**增长（本章实测 N=8..128：5.03 → 81.5，
    即 :math:`\approx0.6365N`）；等距网格恒为 1。
    """
    g = node_spacing(n)
    return float(np.max(g) / np.min(g))


def outer_node_count(n: int, frac: float = 0.1) -> int:
    """落在区间两端各 ``frac`` 宽度内的节点数（不含中点侧的节点重复计数）。"""
    if not 0.0 < frac < 0.5:
        raise ValueError(f"frac 必须在 (0, 0.5) 内，得到 {frac}")
    x = cheb_nodes(n)
    return int(np.count_nonzero(np.abs(x) >= 1.0 - frac))


# --------------------------------------------------------------------------- #
# 3. 问题库
# --------------------------------------------------------------------------- #
ArrayFun = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class BVPProblem:
    r"""一个两点边值问题的全部数据：:math:`u''+pu'+qu=r`（或 :math:`u''=f(x,u)`）。

    线性问题走 ``p/q/r`` 三个系数函数；非线性问题走 ``nonlinear``/``jac_u``
    （Bratu 型 :math:`u''=f(x,u)`，不含 :math:`u'`）。``exact`` 一律给出**闭式解**——
    本章每条误差曲线都对它取差，没有"参考实现"当挡箭牌。
    """

    name: str
    label: str
    x_span: tuple[float, float]
    alpha: float
    beta: float
    bc_left: str
    bc_right: str
    p: ArrayFun
    q: ArrayFun
    r: ArrayFun
    exact: ArrayFun
    exact_prime: ArrayFun | None
    analytic: bool
    smoothness: str
    note: str
    params: tuple[tuple[str, float], ...] = ()
    nonlinear: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    jac_u: ArrayFun | None = None
    sensitivity_theory: float | None = None

    # -- 便利访问 ---------------------------------------------------------- #
    @property
    def a(self) -> float:
        return float(self.x_span[0])

    @property
    def b(self) -> float:
        return float(self.x_span[1])

    @property
    def length(self) -> float:
        return self.b - self.a

    def param(self, key: str, default: float = float("nan")) -> float:
        for k, v in self.params:
            if k == key:
                return float(v)
        return float(default)

    def coefficients(self, x: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xx = np.asarray(x, dtype=float)
        return self.p(xx), self.q(xx), self.r(xx)

    def rhs_system(self, x: float, y: np.ndarray) -> np.ndarray:
        r"""一阶系统 :math:`y=(u,u')` 的右端（打靶法用）：

        .. math::
            y'=\begin{pmatrix}u'\\ r-pu'-qu\end{pmatrix}
            \quad\text{或}\quad
            y'=\begin{pmatrix}u'\\ f(x,u)\end{pmatrix}.
        """
        u, up = float(y[0]), float(y[1])
        if self.nonlinear is not None:
            return np.array([up, float(self.nonlinear(np.array([x]), np.array([u]))[0])])
        p, q, r = self.coefficients(np.array([x]))
        return np.array([up, float(r[0]) - float(p[0]) * up - float(q[0]) * u])

    def jacobian_system(self, x: float, y: np.ndarray) -> np.ndarray:
        r"""变分方程的 Jacobian :math:`\partial(\text{rhs})/\partial y`（2x2）。"""
        u = float(y[0])
        if self.nonlinear is not None:
            fu = float(self.jac_u(np.array([u]))[0]) if self.jac_u is not None else 0.0
            return np.array([[0.0, 1.0], [fu, 0.0]])
        p, q, _ = self.coefficients(np.array([x]))
        return np.array([[0.0, 1.0], [-float(q[0]), -float(p[0])]])

    def derivative(self, x: Any) -> np.ndarray:
        r"""闭式一阶导数 :math:`u'(x)`（"谱微分准不准"要拿它对拍）。

        ``exact_prime`` 缺省为 ``None`` 时抛 :class:`ValueError`，**绝不**退化成
        "用差商假装解析解"——那样比较出来的误差是差商自己的截断误差（见 pitfalls.md）。
        """
        if self.exact_prime is None:
            raise ValueError(f"算例 {self.name!r} 没有提供闭式导数 exact_prime")
        return np.asarray(self.exact_prime(np.asarray(x, dtype=float)), dtype=float)

    def boundary_values(self) -> tuple[float, float]:
        """两端的目标值 :math:`(\\alpha,\\beta)`（Neumann 时是导数值）。"""
        return float(self.alpha), float(self.beta)


def _const(value: float) -> ArrayFun:
    return lambda x: np.full(np.shape(x), float(value), dtype=float)


def _zero(x: np.ndarray) -> np.ndarray:
    return np.zeros(np.shape(x), dtype=float)


PROBLEM_LABELS: dict[str, str] = {
    "smooth": "解析解 e^x sin(pi x)（谱方法指数收敛）",
    "abs3": "折点解 |x|^3（只 C2，退化成代数收敛）",
    "abs5": "折点解 |x|^5（只 C4，代数阶更高）",
    "tanh": "陡峭但解析的 tanh(k x)（要 N 跟得上 k）",
    "layer": "增长模态的边界层（边界条件在增长的一侧）",
    "decay": "衰减模态的边界层（边界条件在衰减的一侧：打靶法在这里失效）",
    "neumann": "混合边界：u(0)=0 与 u'(1)=0（测边界格式的阶）",
    "bratu": "非线性 Bratu 型 u''=1.5u^2（Newton 化）",
}

_PROBLEM_NAMES = ("smooth", "abs3", "abs5", "tanh", "layer", "decay", "neumann", "bratu")


def problem_names() -> tuple[str, ...]:
    """问题库里的全部名字（顺序固定，便于帧与断言复现）。"""
    return _PROBLEM_NAMES


def problem(name: str, **params: float) -> BVPProblem:
    r"""构造问题（纯函数：同名同参数给出逐位相同的 :class:`BVPProblem`）。

    可用参数：``tanh`` 用 ``k``（默认 10）；``layer`` 用 ``k``（默认 20）与 ``L``（默认 2）。
    """
    key = str(name)
    if key == "smooth":
        # u = e^x sin(pi x)：u(±1)=0；u'' = e^x((1-pi^2) sin + 2pi cos)
        def exact(x: np.ndarray) -> np.ndarray:
            return np.exp(x) * np.sin(np.pi * x)

        def exact_prime(x: np.ndarray) -> np.ndarray:
            return np.exp(x) * (np.sin(np.pi * x) + np.pi * np.cos(np.pi * x))

        def d2(x: np.ndarray) -> np.ndarray:
            return np.exp(x) * ((1.0 - np.pi**2) * np.sin(np.pi * x) + 2.0 * np.pi * np.cos(np.pi * x))

        return BVPProblem(
            name=key,
            label=PROBLEM_LABELS[key],
            x_span=(-1.0, 1.0),
            alpha=0.0,
            beta=0.0,
            bc_left="dirichlet",
            bc_right="dirichlet",
            p=_zero,
            q=_const(-1.0),
            r=lambda x: d2(x) - exact(x),
            exact=exact,
            exact_prime=exact_prime,
            analytic=True,
            smoothness="解析（最近奇点在无穷远）",
            note="方程 u''-u=r：解析解，谱方法应当指数收敛到舍入地板",
        )
    if key == "abs3":
        # u = |x|^3：u'' = 6|x|；|x|^3 是偶函数，两端目标值都是 +1
        return BVPProblem(
            name=key,
            label=PROBLEM_LABELS[key],
            x_span=(-1.0, 1.0),
            alpha=1.0,
            beta=1.0,
            bc_left="dirichlet",
            bc_right="dirichlet",
            p=_zero,
            q=_zero,
            r=lambda x: 6.0 * np.abs(x),
            exact=lambda x: np.abs(x) ** 3,
            exact_prime=lambda x: 3.0 * x * np.abs(x),
            analytic=False,
            smoothness="C2（u''' 在 x=0 跳变）",
            note="u=|x|^3：插值误差 ~N^-3，但配置解只 ~N^-2（微分吃掉一阶）",
        )
    if key == "abs5":
        # u = |x|^5：u'' = 20|x|^3；同样是偶函数，两端都是 +1
        return BVPProblem(
            name=key,
            label=PROBLEM_LABELS[key],
            x_span=(-1.0, 1.0),
            alpha=1.0,
            beta=1.0,
            bc_left="dirichlet",
            bc_right="dirichlet",
            p=_zero,
            q=_zero,
            r=lambda x: 20.0 * np.abs(x) ** 3,
            exact=lambda x: np.abs(x) ** 5,
            exact_prime=lambda x: 5.0 * np.sign(x) * np.abs(x) ** 4,
            analytic=False,
            smoothness="C4（u^(5) 在 x=0 跳变）",
            note="u=|x|^5：代数收敛的阶随光滑度上升（实测 p 随 m 增加）",
        )
    if key == "tanh":
        k = float(params.get("k", 10.0))
        if k <= 0.0:
            raise ValueError(f"tanh 的陡度 k 必须为正，得到 {k}")

        def exact(x: np.ndarray) -> np.ndarray:
            return np.tanh(k * np.asarray(x, dtype=float))

        def exact_prime(x: np.ndarray) -> np.ndarray:
            return k / np.cosh(k * np.asarray(x, dtype=float)) ** 2

        def d2(x: np.ndarray) -> np.ndarray:
            t = np.tanh(k * np.asarray(x, dtype=float))
            return -2.0 * k * k * t * (1.0 - t * t)

        return BVPProblem(
            name=key,
            label=PROBLEM_LABELS[key],
            x_span=(-1.0, 1.0),
            alpha=float(np.tanh(-k)),
            beta=float(np.tanh(k)),
            bc_left="dirichlet",
            bc_right="dirichlet",
            p=_zero,
            q=_zero,
            r=d2,
            exact=exact,
            exact_prime=exact_prime,
            analytic=True,
            smoothness=f"解析，但奇点在 x=±i*pi/(2k)（k={k:g}）",
            note="陡峭但解析：仍然指数收敛，只是斜率变平（需要 N ~ k 才到地板）",
            params=(("k", k),),
        )
    if key == "layer":
        k = float(params.get("k", 20.0))
        length = float(params.get("L", 2.0))
        if k <= 0.0 or length <= 0.0:
            raise ValueError(f"layer 需要 k>0 且 L>0，得到 k={k}, L={length}")

        def exact(x: np.ndarray) -> np.ndarray:
            # sinh(kx)/sinh(kL) 的稳定写法：左端是指数小的量，直接写 sinh 会溢出
            xx = np.asarray(x, dtype=float)
            num = np.exp(-k * (length - xx)) * (1.0 - np.exp(-2.0 * k * xx))
            den = 1.0 - math.exp(-2.0 * k * length)
            return num / den

        def exact_prime(x: np.ndarray) -> np.ndarray:
            return k * np.cosh(k * np.asarray(x, dtype=float)) / np.sinh(k * length)

        return BVPProblem(
            name=key,
            label=PROBLEM_LABELS[key],
            x_span=(0.0, length),
            alpha=0.0,
            beta=1.0,
            bc_left="dirichlet",
            bc_right="dirichlet",
            p=_zero,
            q=_const(-(k * k)),
            r=_zero,
            exact=exact,
            exact_prime=exact_prime,
            analytic=True,
            smoothness=f"解析（边界层宽度 1/k，k={k:g}）",
            note="u''=k^2u：单打靶的灵敏度 = sinh(kL)/k，指数增长",
            params=(("k", k), ("L", length)),
            sensitivity_theory=float(np.sinh(k * length) / k) if k * length < 700.0 else float("inf"),
        )
    if key == "decay":
        k = float(params.get("k", 20.0))
        length = float(params.get("L", 2.0))
        if k <= 0.0 or length <= 0.0:
            raise ValueError(f"decay 需要 k>0 且 L>0，得到 k={k}, L={length}")

        def exact(x: np.ndarray) -> np.ndarray:
            # sinh(k(L-x))/sinh(kL)：右端指数小，直接写 sinh 相除会溢出
            xx = np.asarray(x, dtype=float)
            num = np.exp(-k * xx) * (1.0 - np.exp(-2.0 * k * (length - xx)))
            den = 1.0 - math.exp(-2.0 * k * length)
            return num / den

        def exact_prime(x: np.ndarray) -> np.ndarray:
            # d/dx sinh(k(L-x))/sinh(kL) = -k cosh(k(L-x))/sinh(kL)
            # 用 exact 的稳定形式表示：cosh(k(L-x))/sinh(kL) = u(x)·(1+e^{-2k(L-x)})/(1-e^{-2k(L-x)})
            xx = np.asarray(x, dtype=float)
            e = np.exp(-2.0 * k * (length - xx))
            return -k * exact(xx) * (1.0 + e) / (1.0 - e)

        return BVPProblem(
            name=key,
            label=PROBLEM_LABELS[key],
            x_span=(0.0, length),
            alpha=1.0,
            beta=0.0,
            bc_left="dirichlet",
            bc_right="dirichlet",
            p=_zero,
            q=_const(-(k * k)),
            r=_zero,
            exact=exact,
            exact_prime=exact_prime,
            analytic=True,
            smoothness=f"解析（边界层宽度 1/k，k={k:g}）",
            note="u''=k^2u、u(0)=1、u(L)=0：物理解是**衰减**模态，要在指数小的分量上抵消",
            params=(("k", k), ("L", length)),
            sensitivity_theory=float(np.sinh(k * length) / k) if k * length < 700.0 else float("inf"),
        )
    if key == "neumann":
        # u = sin(pi x / 2)：u(0)=0、u'(1)=0；u'' = -(pi/2)^2 u
        w = np.pi / 2.0

        def exact(x: np.ndarray) -> np.ndarray:
            return np.sin(w * np.asarray(x, dtype=float))

        return BVPProblem(
            name=key,
            label=PROBLEM_LABELS[key],
            x_span=(0.0, 1.0),
            alpha=0.0,
            beta=0.0,  # Neumann 端的导数值
            bc_left="dirichlet",
            bc_right="neumann",
            p=_zero,
            q=_zero,
            r=lambda x: -(w * w) * exact(x),
            exact=exact,
            exact_prime=lambda x: w * np.cos(w * np.asarray(x, dtype=float)),
            analytic=True,
            smoothness="解析",
            note="边界格式一阶/二阶的差别就在这里：整体阶 1 vs 2（内部恒为二阶）",
        )
    if key == "bratu":
        # u'' = 1.5 u^2，u(0)=4、u(1)=1，精确解 u = 4/(1+x)^2
        return BVPProblem(
            name=key,
            label=PROBLEM_LABELS[key],
            x_span=(0.0, 1.0),
            alpha=4.0,
            beta=1.0,
            bc_left="dirichlet",
            bc_right="dirichlet",
            p=_zero,
            q=_zero,
            r=_zero,
            exact=lambda x: 4.0 / (1.0 + np.asarray(x, dtype=float)) ** 2,
            exact_prime=lambda x: -8.0 / (1.0 + np.asarray(x, dtype=float)) ** 3,
            analytic=True,
            smoothness="解析",
            note="非线性：打靶法/差分/谱方法三条路都要 Newton 化（解不唯一，初值要对）",
            nonlinear=lambda x, u: 1.5 * u * u,
            jac_u=lambda u: 3.0 * u,
        )
    raise KeyError(f"未知问题 {name!r}；可用：{list(_PROBLEM_NAMES)}")


# --------------------------------------------------------------------------- #
# 4. 打靶法
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ShootingResult:
    """单打靶的结果（含"它靠不靠得住"的那几个数）。"""

    slope: float
    slope_theory: float
    u_end: float
    residual: float
    iterations: int
    converged: bool
    sensitivity: float
    x: np.ndarray
    u: np.ndarray
    history: tuple[float, ...]
    n_steps: int
    method: str
    error_max: float
    error_mid_rel: float
    valid: bool
    stagnated: bool = False

    @property
    def digits_lost(self) -> float:
        r"""灵敏度吃掉的十进制位数 :math:`\log_{10}` 灵敏度（灵敏度 > 1 才有意义）。"""
        return amplification_digits(self.sensitivity)


def amplification_digits(sensitivity: float) -> float:
    r"""放大因子对应的十进制位数：:math:`\log_{10}S`（:math:`S\le1` 时返回 0）。"""
    s = float(sensitivity)
    if not math.isfinite(s) or s <= 1.0:
        return 0.0
    return math.log10(s)


def _initial_state(problem: BVPProblem, slope: float) -> np.ndarray:
    r"""左端初值 :math:`y(a)`：Dirichlet 端钉住 :math:`u`、Neumann 端钉住 :math:`u'`，
    另一分量就是打靶参数 ``slope``（本章的问题库里左端恒为 Dirichlet）。"""
    if problem.bc_left == "dirichlet":
        return np.array([problem.alpha, float(slope)])
    return np.array([float(slope), problem.alpha])


def _match_index(problem: BVPProblem) -> int:
    """右端要匹配的分量：Dirichlet 匹配 :math:`u`（0），Neumann 匹配 :math:`u'`（1）。"""
    return 0 if problem.bc_right == "dirichlet" else 1


def _integrate_ivp(
    problem: BVPProblem,
    slope: float,
    *,
    n_steps: int,
    method: str = "rk4",
    with_variational: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    r"""从左端初值出发推进到 :math:`b`（打靶法的"一枪"）。

    返回 ``(x, u, z, ok)``：``z`` 是变分方程 :math:`z'=Jz,\ z(a)=\partial y(a)/\partial s`
    的解（``z[0]`` 就是 :math:`\partial u/\partial s`）。飞出 :data:`BLOWUP` 时立即停下
    并 ``ok=False``，绝不让 ``inf``/``nan`` 流进结果。
    """
    ns = max(2, int(n_steps))
    a, b = problem.x_span
    h = (b - a) / ns
    x = a + h * np.arange(ns + 1)
    y = _initial_state(problem, slope)
    z = np.array([0.0, 1.0]) if problem.bc_left == "dirichlet" else np.array([1.0, 0.0])
    us = np.empty(ns + 1)
    us[0] = y[0]
    ok = True
    for i in range(ns):
        y_old = y
        step = rk_step(method, problem.rhs_system, x[i], y, h)
        y = step.y
        if with_variational:
            # Jacobian 要取**半步中点**的状态（两个端点状态的平均）。
            # 用步末的 y 会把变分方程降到一阶，Newton 随之退化成线性收敛
            # （本章实测：取步末状态时 |F| 每步只降 1.3e-3 倍，取中点后立刻恢复二次）。
            jac = problem.jacobian_system(x[i] + 0.5 * h, 0.5 * (y_old + y))
            step_z = rk_step("rk4", lambda t, zz, J=jac: J @ zz, x[i], z, h)
            z = step_z.y
        us[i + 1] = y[0]
        if not np.all(np.isfinite(y)) or float(np.max(np.abs(y))) > BLOWUP:
            us[i + 1 :] = np.nan
            ok = False
            break
    if not ok:
        us = np.where(np.isfinite(us), us, 0.0)
    if not with_variational:
        z = np.array([float("nan"), float("nan")])
    return x, us, z, ok


def shooting_endpoint(
    problem: BVPProblem, slope: float, *, n_steps: int = 400, method: str = "rk4"
) -> tuple[float, float]:
    r"""给定初始斜率，返回 :math:`(F(s),\ F'(s))`：右端匹配量与它对 :math:`s` 的导数。"""
    x, us, z, ok = _integrate_ivp(problem, slope, n_steps=n_steps, method=method)
    if not ok:
        return float("nan"), float("nan")
    k = _match_index(problem)
    if k == 0:
        return float(us[-1]), float(z[0])
    # Neumann：右端匹配 u'(b)，而 us 里存的是 u；用变分量的第二个分量
    return float(_endpoint_derivative(problem, slope, n_steps=n_steps, method=method)), float(z[1])


def _endpoint_derivative(problem: BVPProblem, slope: float, *, n_steps: int, method: str) -> float:
    """右端点的 :math:`u'(b)`（把一枪打到头，取状态第二个分量）。"""
    ns = max(2, int(n_steps))
    a, b = problem.x_span
    h = (b - a) / ns
    y = _initial_state(problem, slope)
    for i in range(ns):
        y = rk_step(method, problem.rhs_system, a + i * h, y, h).y
        if not np.all(np.isfinite(y)) or float(np.max(np.abs(y))) > BLOWUP:
            return float("nan")
    return float(y[1])


def shooting_sensitivity(
    problem: BVPProblem, slope: float | None = None, *, n_steps: int = 400, method: str = "rk4"
) -> float:
    r"""实测打靶灵敏度 :math:`S=\left|\partial (\text{右端匹配量})/\partial s\right|`。

    变分方程积分给出（不是差分近似）。对 :math:`u''=k^2u`、:math:`u(a)=0` 的算例，
    闭式是 :math:`S=\sinh(kL)/k`（``tests/3.4`` 用它做 A 类对拍）。
    """
    s = float(problem.alpha + 1.0) if slope is None else float(slope)
    _, duds = shooting_endpoint(problem, s, n_steps=n_steps, method=method)
    return abs(float(duds))


def shooting(
    problem: BVPProblem,
    *,
    slope0: float | None = None,
    n_steps: int = 400,
    max_iter: int = 60,
    tol: float = 1e-12,
    method: str = "rk4",
) -> ShootingResult:
    r"""单打靶 + Newton：求 :math:`s` 使 :math:`F(s)=0`（:math:`F` 是右端匹配量与目标之差）。

    Newton 步是 :math:`s\leftarrow s-F(s)/F'(s)`，其中 :math:`F'` 由变分方程给出
    （线性问题由叠加原理**一步**收敛，非线性问题二次收敛——两个现象都由 ``tests/3.4``
    记录实测迭代次数）。|F| 停滞或飞出 :data:`BLOWUP` 时如实返回 ``converged=False``。
    """
    a, b = problem.x_span
    s = float((problem.beta - problem.alpha) / (b - a)) if slope0 is None else float(slope0)
    history: list[float] = []
    converged = False
    stagnated = False
    it = 0
    best = math.inf
    for it in range(1, int(max_iter) + 1):
        matched, duds = shooting_endpoint(problem, s, n_steps=n_steps, method=method)
        f_val = matched - problem.beta
        history.append(float(f_val))
        if not math.isfinite(f_val):
            break
        if abs(f_val) <= tol:
            converged = True
            break
        if not math.isfinite(duds) or duds == 0.0:
            break
        if it > 1 and abs(f_val) >= 0.5 * best:
            # |F| 不再下降：这是打靶函数的**可分辨下限**（舍入被灵敏度放大），
            # 再迭代只是在噪声里打转——如实记为停滞，不冒充成"还在收敛"。
            stagnated = True
            break
        best = min(best, abs(f_val))
        s_new = s - f_val / duds
        if not math.isfinite(s_new):
            break
        s = s_new
    x, us, z, ok = _integrate_ivp(problem, s, n_steps=n_steps, method=method)
    exact = problem.exact(x)
    err = np.abs(us - exact)
    mid = us.size // 2
    rel_mid = float(abs(us[mid] - exact[mid]) / max(abs(exact[mid]), 1e-300))
    sens = abs(float(z[_match_index(problem)])) if ok else float("inf")
    matched = _matched_value(problem, s, n_steps=n_steps, method=method)
    return ShootingResult(
        slope=float(s),
        slope_theory=_theory_slope(problem, n_steps=n_steps, method=method),
        u_end=matched,
        residual=float(matched - problem.beta),
        iterations=int(it),
        converged=bool(converged),
        sensitivity=sens,
        x=x,
        u=us,
        history=tuple(history),
        n_steps=int(n_steps),
        method=method,
        error_max=float(np.max(err)) if ok else float("inf"),
        error_mid_rel=rel_mid if ok else float("inf"),
        valid=bool(ok),
        stagnated=bool(stagnated),
    )


def _matched_value(problem: BVPProblem, slope: float, *, n_steps: int, method: str) -> float:
    """右端匹配量 :math:`u(b)` 或 :math:`u'(b)`（Dirichlet / Neumann）。"""
    if _match_index(problem) == 0:
        _, us, _, ok = _integrate_ivp(problem, slope, n_steps=n_steps, method=method, with_variational=False)
        return float(us[-1]) if ok else float("nan")
    return _endpoint_derivative(problem, slope, n_steps=n_steps, method=method)


def _theory_slope(problem: BVPProblem, *, n_steps: int, method: str) -> float:
    r"""参考斜率：从 :math:`s=0` 打一枪，用**线性叠加**反推（:math:`F` 对 :math:`s` 是仿射的）。

    它只是对照量（非线性问题下无意义），不许当成"实测值"写进断言。
    """
    f0, duds = shooting_endpoint(problem, 0.0, n_steps=n_steps, method=method)
    if not math.isfinite(duds) or duds == 0.0:
        return float("nan")
    return float(-f0 / duds)


def shooting_scan(
    problem: BVPProblem,
    slopes: Any,
    *,
    n_steps: int = 400,
    method: str = "rk4",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""斜率扫描：返回 ``(x, family, matched, ok)``。

    ``family[k]`` 是第 ``k`` 个初始斜率对应的整条轨迹（``(m, n_steps+1)``），
    ``matched[k]`` 是这一枪在右端的匹配量（打靶函数的值），``ok[k]`` 标记有没有飞出
    :data:`BLOWUP`——飞出去的那几枪在帧里带掩码，不冒充成"很小的值"。
    """
    ss = np.asarray(slopes, dtype=float).reshape(-1)
    xs: np.ndarray | None = None
    rows: list[np.ndarray] = []
    ends = np.zeros(ss.size)
    ok = np.zeros(ss.size, dtype=bool)
    for k, s in enumerate(ss):
        x, us, _, good = _integrate_ivp(problem, float(s), n_steps=n_steps, method=method, with_variational=False)
        if xs is None:
            xs = x
        rows.append(us)
        ends[k] = _matched_value(problem, float(s), n_steps=n_steps, method=method) if good else float("nan")
        ok[k] = good
    family = np.vstack(rows)
    if xs is None:
        xs = np.linspace(problem.a, problem.b, int(n_steps) + 1)
    return xs, family, ends, ok


# --------------------------------------------------------------------------- #
# 5. 有限差分
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FDResult:
    """有限差分解与它的诊断量（离散残差、真实边界残差、整体/内部误差）。"""

    x: np.ndarray
    u: np.ndarray
    n: int
    bc_order: int
    linear_residual: float
    error_max: float
    error_interior: float
    bc_discrete: float
    bc_true: float
    iterations: int
    converged: bool
    nnz_per_row: float
    cond_inf: float


def _neumann_row_left(order: int, h: float, n: int) -> tuple[np.ndarray, np.ndarray, float]:
    r"""左端 :math:`u'(a)=\alpha` 的单侧差分行：返回 ``(cols, coefs, scale)``。

    ``order=1``：:math:`(u_1-u_0)/h`；``order=2``：:math:`(-3u_0+4u_1-u_2)/(2h)`。
    """
    if order == 1:
        return np.array([0, 1]), np.array([-1.0, 1.0]), 1.0 / h
    if order == 2:
        return np.array([0, 1, 2]), np.array([-3.0, 4.0, -1.0]), 1.0 / (2.0 * h)
    raise ValueError(f"bc_order 只能是 1 或 2，得到 {order}")


def _neumann_row_right(order: int, h: float, n: int) -> tuple[np.ndarray, np.ndarray, float]:
    r"""右端 :math:`u'(b)=\beta` 的单侧差分行（下标 ``n`` 是右端点）。"""
    if order == 1:
        return np.array([n, n - 1]), np.array([1.0, -1.0]), 1.0 / h
    if order == 2:
        return np.array([n, n - 1, n - 2]), np.array([3.0, -4.0, 1.0]), 1.0 / (2.0 * h)
    raise ValueError(f"bc_order 只能是 1 或 2，得到 {order}")


def fd_matrix(problem: BVPProblem, n: int, *, bc_order: int = 2) -> tuple[np.ndarray, np.ndarray]:
    r"""组装有限差分方程组 :math:`Au=d`（内部二阶中心差商，边界按 ``bc_order``）。

    网格是等距的 :math:`x_i=a+ih`；内部行

    .. math::
        \frac{u_{i+1}-2u_i+u_{i-1}}{h^2}
        +p_i\frac{u_{i+1}-u_{i-1}}{2h}+q_iu_i=r_i .

    边界行就是边界条件本身（Dirichlet 取单位行；Neumann 取单侧差商行）——
    **这正是"边界格式的阶"进入整体精度的入口**。
    """
    if n < 3:
        raise ValueError(f"有限差分至少需要 n=3 个区间，得到 {n}")
    if bc_order not in (1, 2):
        raise ValueError(f"bc_order 只能是 1 或 2，得到 {bc_order}")
    a, b = problem.x_span
    h = (b - a) / float(n)
    x = a + h * np.arange(n + 1, dtype=float)
    p, q, r = problem.coefficients(x)
    A = np.zeros((n + 1, n + 1))
    d = np.zeros(n + 1)
    if problem.nonlinear is None:
        for i in range(1, n):
            A[i, i - 1] = 1.0 / h**2 - p[i] / (2.0 * h)
            A[i, i] = -2.0 / h**2 + q[i]
            A[i, i + 1] = 1.0 / h**2 + p[i] / (2.0 * h)
            d[i] = r[i]
    else:
        for i in range(1, n):
            A[i, i - 1] = 1.0 / h**2
            A[i, i] = -2.0 / h**2
            A[i, i + 1] = 1.0 / h**2
            d[i] = 0.0
    _fill_bc_rows(problem, A, d, x, h, n, bc_order)
    return A, d


def _fill_bc_rows(
    problem: BVPProblem,
    A: np.ndarray,
    d: np.ndarray,
    x: np.ndarray,
    h: float,
    n: int,
    bc_order: int,
) -> None:
    """把两端边界行写进 ``A``/``d``（内部与非线性两条路共用）。"""
    if problem.bc_left == "dirichlet":
        A[0, :] = 0.0
        A[0, 0] = 1.0
        d[0] = problem.alpha
    elif problem.bc_left == "neumann":
        cols, coefs, scale = _neumann_row_left(bc_order, h, n)
        A[0, :] = 0.0
        A[0, cols] = coefs * scale
        d[0] = problem.alpha * scale
    else:
        raise ValueError(f"未知左边界类型 {problem.bc_left!r}")
    if problem.bc_right == "dirichlet":
        A[n, :] = 0.0
        A[n, n] = 1.0
        d[n] = problem.beta
    elif problem.bc_right == "neumann":
        cols, coefs, scale = _neumann_row_right(bc_order, h, n)
        A[n, :] = 0.0
        A[n, cols] = coefs * scale
        d[n] = problem.beta * scale
    else:
        raise ValueError(f"未知右边界类型 {problem.bc_right!r}")


def fd_bvp(
    problem: BVPProblem,
    n: int,
    *,
    bc_order: int = 2,
    max_iter: int = 60,
    newton_tol: float = 1e-13,
) -> FDResult:
    r"""有限差分求解；非线性问题走 Newton（Jacobian 是三对角 + 边界行，同样走自研 LU）。

    ``linear_residual`` 是 :math:`\|Au-d\|_\infty`：它**永远**是 :math:`u` 量级，
    而解的误差由截断误差与条件数决定——这两个数在本章被一起展示（见 ``pitfalls.md``）。
    """
    if n < 3:
        raise ValueError(f"有限差分至少需要 n=3 个区间，得到 {n}")
    a, b = problem.x_span
    h = (b - a) / float(n)
    x = a + h * np.arange(n + 1, dtype=float)
    A, d = fd_matrix(problem, n, bc_order=bc_order)
    if problem.nonlinear is None:
        u = solve_lu(lu_decompose(A, pivot="partial"), d)
        it = 1
        converged = True
        res_sys = float(np.max(np.abs(A @ u - d)))
    else:
        u = np.linspace(problem.alpha, problem.beta, n + 1)
        # 先满足边界行（Newton 只在内部残差上迭代）
        _apply_bc_to_vector(problem, u, bc_order, h, n)
        it = 0
        converged = False
        for it in range(1, int(max_iter) + 1):
            res = A @ u
            res[0] = u[0] - (problem.alpha if problem.bc_left == "dirichlet" else 0.0)
            res[n] = u[n] - (problem.beta if problem.bc_right == "dirichlet" else 0.0)
            for i in range(1, n):
                res[i] = A[i] @ u - float(problem.nonlinear(x[i : i + 1], u[i : i + 1])[0])
            J = A.copy()
            if problem.jac_u is not None:
                for i in range(1, n):
                    J[i, i] -= float(problem.jac_u(u[i : i + 1])[0])
            delta = solve_lu(lu_decompose(J, pivot="partial"), -res)
            u = u + delta
            # 残差的绝对地板 ~ eps/h^2（差商把舍入放大 1/h^2 倍），
            # 所以判据必须看**相对步长**，而不是"残差 < 1e-13"。
            if float(np.max(np.abs(delta))) <= newton_tol * (1.0 + float(np.max(np.abs(u)))):
                converged = True
                break
        it = max(1, it)
        res_sys = float(np.max(np.abs(res)))
    exact = problem.exact(x)
    err = np.abs(u - exact)
    m = max(3, n // 8)
    bc_disc, bc_true = bc_residuals(problem, x, u, method="fd", bc_order=bc_order)
    return FDResult(
        x=x,
        u=u,
        n=int(n),
        bc_order=int(bc_order),
        linear_residual=res_sys,
        error_max=float(np.max(err)),
        error_interior=float(np.max(err[m : n - m + 1])),
        bc_discrete=bc_disc,
        bc_true=bc_true,
        iterations=int(it),
        converged=bool(converged),
        nnz_per_row=float(np.count_nonzero(np.abs(A) > 0.0) / (n + 1)),
        cond_inf=float(_cond_inf_safe(A)),
    )


def _apply_bc_to_vector(problem: BVPProblem, u: np.ndarray, bc_order: int, h: float, n: int) -> None:
    """把初值向量的两个端点调到满足边界条件（Newton 的起点）。"""
    if problem.bc_left == "dirichlet":
        u[0] = problem.alpha
    if problem.bc_right == "dirichlet":
        u[n] = problem.beta


# --------------------------------------------------------------------------- #
# 6. Chebyshev 谱方法：配置法与 tau 法
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SpectralResult:
    """谱方法解与它的诊断量（矩阵条件数、节点残差、边界残差、Newton 迭代数）。"""

    x: np.ndarray
    u: np.ndarray
    coeffs: np.ndarray
    n: int
    method: str
    matrix: np.ndarray
    linear_residual: float
    error_max: float
    error_interior: float
    bc_discrete: float
    bc_true: float
    iterations: int
    converged: bool
    cond_inf: float
    density: float


def _bc_row(problem: BVPProblem, side: str, ind: int, D: np.ndarray, n: int) -> tuple[np.ndarray, float]:
    """配置法的边界行：Dirichlet 取单位行，Neumann 直接搬 :math:`D_N` 的一行。"""
    kind = problem.bc_left if side == "left" else problem.bc_right
    value = problem.alpha if side == "left" else problem.beta
    row = np.zeros(n + 1)
    if kind == "dirichlet":
        row[ind] = 1.0
    elif kind == "neumann":
        row = D[ind, :].copy()
    else:
        raise ValueError(f"未知边界类型 {kind!r}")
    return row, float(value)


def collocation_matrix(problem: BVPProblem, n: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Chebyshev 配置法方程组 :math:`Lu=f`：内部行是 :math:`D_N^2+pD_N+qI`，两端换成边界行。

    下标约定：``0`` 是 :math:`x=+1`、``N`` 是 :math:`x=-1`（降序网格）。
    """
    if n < 2:
        raise ValueError(f"配置法至少需要 N=2，得到 {n}")
    x = cheb_nodes_on(problem, n)
    D = cheb_diff_matrix(n)
    sc = domain_scale(problem)
    p, q, r = problem.coefficients(x)
    L = (sc * sc) * (D @ D) + (sc * p)[:, None] * D + np.diag(q)
    f = r.copy()
    row, val = _bc_row(problem, "right", 0, sc * D, n)
    L[0, :] = row
    f[0] = val
    row, val = _bc_row(problem, "left", n, sc * D, n)
    L[n, :] = row
    f[n] = val
    return L, f


def tau_matrix(problem: BVPProblem, n: int) -> tuple[np.ndarray, np.ndarray]:
    r"""tau 法方程组：在**系数空间**里投影（前 :math:`N-1` 个模态），最后两行换成边界行。

    系数空间里 :math:`u''\to M_2a`（上三角）、:math:`u'\to M_1a`、:math:`u\to a`，
    :math:`r` 的系数由 :func:`cheb_transform` 给出。与配置法的差别是"方程在哪一点成立"：
    tau 要求残差的前 :math:`N-1` 个 Chebyshev 系数为零（投影意义），配置法要求残差在
    :math:`N+1` 个节点处为零。两者都是 :math:`(N+1)\times(N+1)` 的稠密系统，
    条件数同量级（实测见 :func:`convergence_study` 的 ``cond`` 输出）。
    """
    if n < 2:
        raise ValueError(f"tau 法至少需要 N=2，得到 {n}")
    if problem.bc_left != "dirichlet" or problem.bc_right != "dirichlet":
        raise ValueError("tau 法在本章只实现了两端 Dirichlet（Neumann 行要另写，见配置法）")
    x = cheb_nodes_on(problem, n)
    p, q, r = problem.coefficients(x)
    if not (_is_constant(p) and _is_constant(q)):
        raise ValueError("tau 法在本章只实现了常系数 p/q（变系数请走配置法）")
    sc = domain_scale(problem)
    M2 = coeff_second_matrix(n)
    M1 = coeff_first_matrix(n)
    A = (sc * sc) * M2 + (sc * float(p[0])) * M1 + float(q[0]) * np.eye(n + 1)
    f = cheb_transform(r)
    # 丢掉最后两行（它们被边界条件替换），前面的模态保留
    A[n - 1, :] = (-1.0) ** np.arange(n + 1)  # u(-1)
    f[n - 1] = problem.alpha
    A[n, :] = 1.0  # u(+1)
    f[n] = problem.beta
    return A, f


def _is_constant(v: np.ndarray) -> bool:
    return bool(np.max(np.abs(v - v[0])) <= 1e-14 * max(1.0, float(np.max(np.abs(v)))))


def cheb_collocation(
    problem: BVPProblem, n: int, *, max_iter: int = 60, newton_tol: float = 1e-13
) -> SpectralResult:
    r"""Chebyshev 配置法求解（非线性问题按 Newton 迭代，Jacobian :math:`D_N^2-\mathrm{diag}(f_u)`）。"""
    if n < 2:
        raise ValueError(f"配置法至少需要 N=2，得到 {n}")
    x = cheb_nodes_on(problem, n)
    D = cheb_diff_matrix(n)
    sc = domain_scale(problem)
    L, f = collocation_matrix(problem, n)
    if problem.nonlinear is None:
        u = solve_lu(lu_decompose(L, pivot="partial"), f)
        it, converged = 1, True
        res_sys = float(np.max(np.abs(L @ u - f)))
    else:
        u = np.linspace(problem.beta, problem.alpha, n + 1)  # 下标 0 是 x=b（降序网格）
        it, converged = 0, False
        for it in range(1, int(max_iter) + 1):
            F = (sc * sc) * (D @ (D @ u)) - np.asarray(problem.nonlinear(x, u), dtype=float)
            F[0] = u[0] - (problem.beta if problem.bc_right == "dirichlet" else 0.0)
            F[n] = u[n] - (problem.alpha if problem.bc_left == "dirichlet" else 0.0)
            J = (sc * sc) * (D @ D) - np.diag(np.asarray(problem.jac_u(u), dtype=float))
            J[0, :] = 0.0
            J[0, 0] = 1.0
            J[n, :] = 0.0
            J[n, n] = 1.0
            delta = solve_lu(lu_decompose(J, pivot="partial"), -F)
            u = u + delta
            # 收敛判据用**相对步长**：残差的绝对值下限由 eps*(s^2 N^4) 决定，
            # 拿"残差 < 1e-13"当判据会让 Newton 在舍入噪声里空转（实测过）。
            if float(np.max(np.abs(delta))) <= newton_tol * (1.0 + float(np.max(np.abs(u)))):
                converged = True
                break
        it = max(1, it)
        res_sys = float(np.max(np.abs(F)))
    coeffs = cheb_transform(u)
    exact = problem.exact(x)
    err = np.abs(u - exact)
    m = max(2, n // 8)
    bc_disc, bc_true = bc_residuals(problem, x, u, method="cheb")
    return SpectralResult(
        x=x,
        u=u,
        coeffs=coeffs,
        n=int(n),
        method="collocation",
        matrix=L,
        linear_residual=res_sys,
        error_max=float(np.max(err)),
        error_interior=float(np.max(err[m : n - m + 1])),
        bc_discrete=bc_disc,
        bc_true=bc_true,
        iterations=int(it),
        converged=bool(converged),
        cond_inf=float(_cond_inf_safe(L)),
        density=float(np.count_nonzero(np.abs(L) > 0.0) / L.size),
    )


def cheb_tau(problem: BVPProblem, n: int) -> SpectralResult:
    r"""tau 法求解（常系数线性问题）：解出系数后回代节点，诊断量与配置法同口径。"""
    if n < 2:
        raise ValueError(f"tau 法至少需要 N=2，得到 {n}")
    if problem.nonlinear is not None:
        raise ValueError("tau 法在本章只用于线性问题")
    A, f = tau_matrix(problem, n)
    coeffs = solve_lu(lu_decompose(A, pivot="partial"), f)
    x = cheb_nodes_on(problem, n)
    u = clenshaw(coeffs, cheb_nodes(n))
    exact = problem.exact(x)
    err = np.abs(u - exact)
    m = max(2, n // 8)
    bc_disc, bc_true = bc_residuals(problem, x, u, method="cheb")
    return SpectralResult(
        x=x,
        u=u,
        coeffs=coeffs,
        n=int(n),
        method="tau",
        matrix=A,
        linear_residual=float(np.max(np.abs(A @ coeffs - f))),
        error_max=float(np.max(err)),
        error_interior=float(np.max(err[m : n - m + 1])),
        bc_discrete=bc_disc,
        bc_true=bc_true,
        iterations=1,
        converged=True,
        cond_inf=float(_cond_inf_safe(A)),
        density=float(np.count_nonzero(np.abs(A) > 0.0) / A.size),
    )


def bvp_operator(problem: BVPProblem, n: int, *, method: str = "collocation") -> np.ndarray:
    """按方法名给出 :math:`(N+1)\\times(N+1)` 的离散算子（配置法/tau/有限差分小规模版）。"""
    if method == "collocation":
        return collocation_matrix(problem, n)[0]
    if method == "tau":
        return tau_matrix(problem, n)[0]
    if method == "fd":
        return fd_matrix(problem, n)[0]
    raise ValueError(f"未知方法 {method!r}（可用 collocation / tau / fd）")


def spectral_residual(problem: BVPProblem, coeffs: Any, xq: Any) -> np.ndarray:
    r"""把谱解的**连续**残差 :math:`u_N''+pu_N'+qu_N-r` 在 ``xq`` 上算出来。

    这是本章最会骗人的一个量：在配置节点上它恒为 0（那是方程组的定义），
    在节点之间它才是真正的"方程差多少"。用系数空间求导（不是 :math:`D_N` 乘积），
    因为这里要的是多项式插值本身的导数。
    """
    a = np.asarray(coeffs, dtype=float).reshape(-1)
    q = np.asarray(xq, dtype=float)
    xi = to_reference(problem, q)
    sc = domain_scale(problem)
    u = cheb_series_values(a, xi, 0)
    up = sc * cheb_series_values(a, xi, 1)
    upp = (sc * sc) * cheb_series_values(a, xi, 2)
    if problem.nonlinear is not None:
        return upp - np.asarray(problem.nonlinear(q, u), dtype=float)
    p, qq, r = problem.coefficients(q)
    return upp + p * up + qq * u - r


# --------------------------------------------------------------------------- #
# 7. 诊断：误差、边界残差、条件数、收敛研究
# --------------------------------------------------------------------------- #
def max_error(problem: BVPProblem, x: Any, u: Any) -> float:
    """最大模范数下的误差 :math:`\\|u_{\\rm num}-u_{\\rm ex}\\|_\\infty`。"""
    xx = np.asarray(x, dtype=float)
    uu = np.asarray(u, dtype=float)
    return float(np.max(np.abs(uu - problem.exact(xx))))


def interpolation_error(problem: BVPProblem, n: int, xq: Any = None) -> float:
    r"""**插值**误差：把精确解在 :math:`N+1` 个 Chebyshev 节点上插值，再与精确解在细网格上比差。

    它与"谱配置**解**的误差"**不是一回事**：对只 :math:`C^s` 的解，插值误差按
    :math:`N^{-(s+1)}` 衰减，而配置解只有 :math:`N^{-s}`（微分把光滑度吃掉一阶）。
    把前者当成后者写进结论，就是本章最典型的假证据（``tests/3.4`` 拿它做负对照）。
    """
    x = cheb_nodes_on(problem, n)
    coeffs = cheb_transform(problem.exact(x))
    q = default_grid(problem) if xq is None else np.asarray(xq, dtype=float)
    return float(np.max(np.abs(clenshaw(coeffs, to_reference(problem, q)) - problem.exact(q))))


def default_grid(problem: BVPProblem, m: int = 801) -> np.ndarray:
    r"""区间内部的细网格（**避开端点**：端点上解被边界条件钉死，会把误差看小）。"""
    a, b = problem.x_span
    return a + (b - a) * (np.arange(m, dtype=float) + 0.5) / float(m)


def interior_error(problem: BVPProblem, x: Any, u: Any, *, frac: float = 0.25) -> float:
    """**内部**误差（掐掉两端各 ``frac`` 比例的区间）——与整体误差并列报告。"""
    xx = np.asarray(x, dtype=float)
    uu = np.asarray(u, dtype=float)
    lo = xx.min()
    hi = xx.max()
    span = hi - lo
    mask = (xx >= lo + frac * span) & (xx <= hi - frac * span)
    if not np.any(mask):
        mask = np.ones_like(xx, dtype=bool)
    return float(np.max(np.abs(uu[mask] - problem.exact(xx[mask]))))


def bc_residuals(
    problem: BVPProblem,
    x: Any,
    u: Any,
    *,
    method: str = "cheb",
    bc_order: int = 2,
) -> tuple[float, float]:
    r"""返回 ``(离散边界残差, 真实边界残差)``——**两个不同的量**。

    * 离散残差：数值格式自己那一行方程的残差。Dirichlet 是 :math:`u_{\rm num}(x_b)-g`；
      Neumann 是**该格式用的那个差商**减目标值。它恒为 :math:`O(u)`——因为方程就是这么解的。
    * 真实残差：把数值解当真函数看，边界条件被违反了多少。Dirichlet 时两者相同；
      Neumann 时真实残差要用**更高阶**的导数估计，一阶边界格式下它是 :math:`O(h)`——
      这就是"整体阶被边界格式拉到一阶"的入口。

    Chebyshev 侧两个残差都会用谱导数（插值多项式的精确导数）：配置法把边界行写成了
    :math:`D_N` 的一行，所以它的"离散"与"真实"本来就是同一个东西——这正好是两种方法
    在处理边界上的差别。把这两个数混为一谈，是本章最容易造出的假证据：
    截图里"边界残差 1e-16"看着完美，实际上格式只是一阶的。
    """
    xx = np.asarray(x, dtype=float)
    uu = np.asarray(u, dtype=float)
    n = xx.size - 1
    h = float(xx[1] - xx[0]) if n >= 1 else 1.0
    spec = None
    sc_lob = lobatto_scale(xx) if method == "cheb" else None
    if sc_lob is not None and n >= 1:
        # 参考区间上的 D_N 给出 d/d(xi)，物理导数还要乘 s=2/(b-a)
        spec = sc_lob * (cheb_diff_matrix(n) @ uu)
    disc = 0.0
    true = 0.0
    # 端点下标的朝向：有限差分网格升序（0 是左端），Chebyshev 网格降序（0 是右端）。
    # 这里写反过一次：Neumann 条件在 x=b 上，却被拿去比 x=a 的导数，量出 pi/2 这种
    # "看起来像误差、其实是另一个点的真值"的数（本章实测 1.571）。
    if xx[0] <= xx[-1]:
        left_ind, right_ind = 0, n
    else:
        left_ind, right_ind = n, 0
    for side, ind in (("left", left_ind), ("right", right_ind)):
        kind = problem.bc_left if side == "left" else problem.bc_right
        target = problem.alpha if side == "left" else problem.beta
        if kind == "dirichlet":
            d_val = float(uu[ind] - target)
            disc = max(disc, abs(d_val))
            true = max(true, abs(d_val))
        elif kind == "neumann":
            if spec is not None:
                d_val = float(spec[ind] - target)
                true = max(true, abs(d_val))
                disc = max(disc, abs(d_val))
            else:
                order = bc_order if method == "fd" else 2
                d_val = _one_sided_derivative(uu, h, side, order)
                disc = max(disc, abs(float(d_val - target)))
                dv4 = _one_sided_derivative(uu, h, side, 4)
                true = max(true, abs(float(dv4 - target)))
        else:
            raise ValueError(f"未知边界类型 {kind!r}")
    return float(disc), float(true)


def _one_sided_derivative(u: np.ndarray, h: float, side: str, order: int) -> float:
    r"""单侧差商 :math:`u'(a)` 或 :math:`u'(b)`（``order`` = 1/2/4）。"""
    n = u.size - 1
    if side == "left":
        c = {1: [-1.0, 1.0], 2: [-1.5, 2.0, -0.5], 4: [-25.0 / 12.0, 4.0, -3.0, 4.0 / 3.0, -0.25]}[order]
        vals = [u[j] for j in range(len(c))]
    else:
        c = {1: [-1.0, 1.0], 2: [0.5, -2.0, 1.5], 4: [0.25, -4.0 / 3.0, 3.0, -4.0, 25.0 / 12.0]}[order]
        vals = [u[n - (len(c) - 1) + j] for j in range(len(c))]
    return float(np.dot(np.array(c), np.array(vals)) / h)


def _cond_inf_safe(A: np.ndarray) -> float:
    r"""1-范数意义下的条件数估计（走自研 LU 求逆，见了零主元返回 ``inf``）。

    ``inf`` 只在内核里出现，进帧前一律换哨兵（帧里不许有非有限值）。
    """
    try:
        res = lu_decompose(A, pivot="partial")
        n = A.shape[0]
        cols = []
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1.0
            cols.append(solve_lu(res, e))
        inv = np.column_stack(cols)
        return float(_norm_inf(A) * _norm_inf(inv))
    except (ZeroDivisionError, ValueError):
        return float("inf")


def operator_diagnostics(A: Any) -> dict[str, float]:
    r"""矩阵的"结构 + 条件"体检：:math:`\|A\|_\infty`、非零密度、1-范数条件数。"""
    M = np.asarray(A, dtype=float)
    nnz = int(np.count_nonzero(M))
    return {
        "norm_inf": float(_norm_inf(M)),
        "density": float(nnz / M.size),
        "n_nonzero": float(nnz),
        "cond_inf": _cond_inf_safe(M),
    }


def diff_matrix_diagnostics(n: int, *, svd: bool = True) -> dict[str, float]:
    r"""微分矩阵 :math:`D_N` 的体检——本章"它到底是什么"的全部数字都在这里。

    * :math:`\|D_N\|_\infty=N^2`（实测逐位吻合，见 ``tests/3.4``）；
    * :math:`D_N` **奇异**（:math:`D_N\mathbf 1=0`），所以 :math:`\kappa=\sigma_{\max}/\sigma_{\min}`
      会报出 :math:`10^{17}` 这种**毫无意义**的数——那是零空间里的舍入噪声；
    * 正确口径是挖掉零空间后的 :math:`\kappa^+=\sigma_{\max}/\sigma_{\min}^+`，
      它按 :math:`O(N^2)` 增长（本章实测指数见断言）；
    * 矩阵是**稠密**的（密度 1），而且 :math:`D_N^2` 更糟：:math:`\|D_N^2\|_\infty\sim N^4`。
    """
    D = cheb_diff_matrix(n)
    row_res = float(np.max(np.abs(D.sum(axis=1))))
    diag: dict[str, float] = {
        "norm_inf": float(_norm_inf(D)),
        "rowsum_residual": row_res,
        "density": float(np.count_nonzero(D) / D.size),
        "n": float(n),
    }
    if svd:
        fit = jacobi_svd(D)
        s = np.asarray(fit.s, dtype=float)
        smax = float(s[0])
        smin = float(s[-1])
        smin_nz = float(s[-2]) if s.size > 1 else float("nan")
        diag["sigma_max"] = smax
        diag["sigma_min"] = smin
        diag["sigma_min_nonzero"] = smin_nz
        diag["kappa_naive"] = smax / smin if smin > 0.0 else float("inf")
        diag["kappa_nonzero"] = smax / smin_nz if smin_nz > 0.0 else float("inf")
        diag["D2_norm_inf"] = float(_norm_inf(cheb_second_matrix(n)))
    return diag


def convergence_study(
    problem: BVPProblem,
    ns: Sequence[int],
    *,
    method: str = "collocation",
    bc_order: int = 2,
) -> dict[str, np.ndarray]:
    r"""对一串 :math:`N` 跑同一个方法，收集误差/条件数/边界残差。

    返回的数组逐个对应输入 ``ns``（**升序**，与 :func:`fit_exponential` /
    :func:`fit_power` 的"按 h 递减"约定相反——拟合函数内部会自己排好）。
    """
    n_list = [int(v) for v in ns]
    errs = np.zeros(len(n_list))
    errs_int = np.zeros(len(n_list))
    conds = np.zeros(len(n_list))
    bcs = np.zeros(len(n_list))
    disc = np.zeros(len(n_list))
    lin = np.zeros(len(n_list))
    for k, n in enumerate(n_list):
        if method == "collocation":
            r = cheb_collocation(problem, n)
        elif method == "tau":
            r = cheb_tau(problem, n)
        elif method == "fd":
            r = fd_bvp(problem, n, bc_order=bc_order)
        else:
            raise ValueError(f"未知方法 {method!r}")
        errs[k] = r.error_max
        errs_int[k] = r.error_interior
        conds[k] = r.cond_inf
        bcs[k] = r.bc_true
        disc[k] = r.bc_discrete
        lin[k] = r.linear_residual
    return {
        "n": np.array(n_list, dtype=float),
        "error": errs,
        "error_interior": errs_int,
        "cond_inf": conds,
        "bc_true": bcs,
        "bc_discrete": disc,
        "linear_residual": lin,
    }


def _r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0


def _line_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    r"""最小二乘直线 :math:`y=c_0+c_1x`（自写正规方程两点式，不用任何成品拟合器）。

    只用均值/协方差/方差三个统计量：:math:`c_1=\mathrm{cov}(x,y)/\mathrm{var}(x)`、
    :math:`c_0=\bar y-c_1\bar x`。它比"套一个 polyfit"更不容易在这一章里说错话——
    阶数与斜率的口径必须与 :mod:`numcore.verify` 的拟合器逐位可比（``tests/3.4`` 对拍）。
    """
    xb = float(np.mean(x))
    yb = float(np.mean(y))
    dx = x - xb
    var = float(np.sum(dx * dx))
    if var <= 0.0:
        return 0.0, yb
    c1 = float(np.sum(dx * (y - yb)) / var)
    return c1, yb - c1 * xb


def fit_exponential(ns: Any, errs: Any, *, head: int | None = None) -> dict[str, float]:
    r"""半对数最小二乘拟合 :math:`\log e\approx c-\lambda N`（谱方法的指数收敛）。

    返回 ``slope``（:math:`-\lambda`，负值）、``intercept`` 与 ``r2``。``head`` 保留**粗端**
    （小 :math:`N`），因为细端一旦触到舍入地板，斜率会被拉平——窗口是结论的一部分。
    """
    n = np.asarray(ns, dtype=float).reshape(-1)
    e = np.asarray(errs, dtype=float).reshape(-1)
    if n.size != e.size:
        raise ValueError(f"ns 与 errs 长度不一致：{n.size} vs {e.size}")
    if head is not None:
        n = n[: int(head)]
        e = e[: int(head)]
    good = e > 0.0
    if np.count_nonzero(good) < 3:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "n_points": float(np.count_nonzero(good))}
    y = np.log(e[good])
    nn = n[good]
    c1, c0 = _line_fit(nn, y)
    return {
        "slope": c1,
        "intercept": c0,
        "r2": float(_r_squared(y, c0 + c1 * nn)),
        "n_points": float(nn.size),
    }


def fit_power(xs: Any, errs: Any, *, tail: int | None = None) -> dict[str, float]:
    r"""双对数最小二乘拟合 :math:`e\approx C x^{-p}`（代数收敛），返回 ``p``、``r2``。

    ``tail`` 保留**细端**（大 :math:`x`）——与 :func:`fit_exponential` 的 ``head`` 方向相反，
    这个"方向别记反"的坑在 ``pitfalls.md`` 里有实测。
    """
    x = np.asarray(xs, dtype=float).reshape(-1)
    e = np.asarray(errs, dtype=float).reshape(-1)
    if x.size != e.size:
        raise ValueError(f"xs 与 errs 长度不一致：{x.size} vs {e.size}")
    if tail is not None:
        x = x[-int(tail) :]
        e = e[-int(tail) :]
    good = (e > 0.0) & (x > 0.0)
    if np.count_nonzero(good) < 3:
        return {"p": float("nan"), "intercept": float("nan"), "r2": float("nan"), "n_points": float(np.count_nonzero(good))}
    y = np.log(e[good])
    xx = np.log(x[good])
    c1, c0 = _line_fit(xx, y)
    return {
        "p": float(-c1),
        "intercept": c0,
        "r2": float(_r_squared(y, c0 + c1 * xx)),
        "n_points": float(xx.size),
    }


def roundoff_floor(errs: Any, ns: Any = None) -> dict[str, float]:
    r"""舍入地板与"撞地板的 N"：返回 ``floor``、``n_floor``、``n_at_min``。

    定义（写进 ``theory``，不许事后改口径）：

    * ``floor`` = 整条曲线的最小误差（那就是能到的最低点）；
    * ``n_at_min`` = 取到最小值的 N；
    * ``n_floor`` = 第一个满足 :math:`e(N)\le 10\\,\\text{floor}` 的 N——
      从那以后误差不再随 N 下降，继续加密只是在浪费自由度。
    """
    e = np.asarray(errs, dtype=float).reshape(-1)
    if ns is None:
        n = np.arange(e.size, dtype=float)
    else:
        n = np.asarray(ns, dtype=float).reshape(-1)
    if e.size == 0 or n.size != e.size:
        raise ValueError("errs 与 ns 长度必须一致且非空")
    k = int(np.argmin(e))
    floor = float(e[k])
    thresh = 10.0 * floor
    hit = np.nonzero(e <= thresh)[0]
    n_floor = float(n[hit[0]]) if hit.size else float(n[k])
    return {"floor": floor, "n_floor": n_floor, "n_at_min": float(n[k])}
