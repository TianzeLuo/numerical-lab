r"""形函数、参考单元与等参映射。

数学定义
--------
每个单元 :math:`K` 配一个**参考单元** :math:`\hat K`（一维 :math:`[0,1]`、
三角形 :math:`\{(\xi,\eta):\xi\ge0,\eta\ge0,\xi+\eta\le1\}`、四边形 :math:`[-1,1]^2`）
与一个可逆映射 :math:`F_K:\hat K\to K`。基函数定义在参考单元上，
物理单元上的值是**拉回**来的：

.. math::
    \varphi_i(x) = \hat\varphi_i\bigl(F_K^{-1}(x)\bigr),\qquad
    \nabla\varphi_i = J_K^{-\mathsf T}\,\hat\nabla\hat\varphi_i,\qquad
    J_K = \frac{\partial F_K}{\partial\hat x}.

**等参**（isoparametric）的意思是：同一个映射 :math:`F_K` 既用来变换几何、也用来变换基函数，
于是四边形单元上的 :math:`J_K` 一般是**逐点变化**的——只有平行四边形才退化成常数。
本模块因此只提供"在给定参考点上逐点算 :math:`J_K`"的接口，
把"取单元中心的 :math:`J` 当常数"这种近似留给 :mod:`~numcore.pde.fem.assemble`
做成一个可对照的错误路线（实测它破坏什么）。

形函数族与 :mod:`~numcore.pde.fem.mesh` 的 family 一一对应：

* 拉格朗日 :math:`P_1/P_2`（一维，节点可以是任意点集，用**重心公式**求值）；
* Chebyshev--Lobatto 节点上的任意次拉格朗日基（p-加密）；
* Hermite 三次基（含一阶导自由度，梁方程用；需要 :math:`N''`）；
* 双线性四边形 :math:`Q_1`；
* 单纯形 :math:`P_1`。

求积规则全部自研（Gauss--Legendre 用 Newton 迭代求 Legendre 零点，
单纯形规则用经典对称表）——本章不该借 :mod:`numpy` 的任何成品数值工具。
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "gauss_legendre",
    "gauss_on_interval",
    "lagrange_values",
    "lagrange_values_and_derivs",
    "hermite_values",
    "hermite_second_derivs",
    "tri_p1_values",
    "quad_q1_values",
    "tri_rule",
    "quad_rule",
    "map_tri",
    "tri_jacobian",
    "map_quad",
    "quad_jacobian",
    "quad_jacobian_det",
    "simplex_measure",
    "isoparametric_ratio",
]


# --------------------------------------------------------------------------- #
# 一维 Gauss--Legendre（自研：Newton 迭代 + 三项递推）
# --------------------------------------------------------------------------- #
def gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    r"""``n`` 点 Gauss--Legendre 节点与权重（区间 :math:`[-1,1]`）。

    节点是 :math:`P_n` 的零点，用 Newton 迭代

    .. math::
        x \leftarrow x - \frac{P_n(x)}{P_n'(x)},\qquad
        P_n'(x) = \frac{n}{x^2-1}\bigl(xP_n(x)-P_{n-1}(x)\bigr)

    从 :math:`\cos\bigl(\pi(i-\tfrac14)/(n+\tfrac12)\bigr)` 出发（标准初值，几步就收敛）；
    权重 :math:`w_i=\frac{2}{(1-x_i^2)P_n'(x_i)^2}`，恒满足 :math:`\sum w_i=2`。
    精度判据是"对 :math:`x^k\ (k\le 2n-1)` 精确"，本章只用它做单元积分。
    """
    nn = int(n)
    if nn < 1:
        raise ValueError(f"点数至少 1，得到 {n!r}")
    if nn == 1:
        return np.zeros(1, dtype=float), np.asarray([2.0])
    i = np.arange(1, nn + 1, dtype=float)
    x = np.cos(np.pi * (i - 0.25) / (nn + 0.5))
    for _ in range(100):
        p0 = np.ones_like(x)
        p1 = x.copy()
        for k in range(2, nn + 1):
            p0, p1 = p1, ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k
        # 现在 p1 = P_n(x)，p0 = P_{n-1}(x)
        dp = nn * (x * p1 - p0) / (x * x - 1.0)
        step = p1 / dp
        x = x - step
        if float(np.max(np.abs(step))) < 4.0 * float(np.finfo(float).eps):
            break
    p0 = np.ones_like(x)
    p1 = x.copy()
    for k in range(2, nn + 1):
        p0, p1 = p1, ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k
    dp = nn * (x * p1 - p0) / (x * x - 1.0)
    w = 2.0 / ((1.0 - x * x) * dp * dp)
    order = np.argsort(x)
    return x[order], w[order]


def gauss_on_interval(n: int, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    r"""仿射映射到 :math:`[a,b]` 的 ``n`` 点 Gauss 规则（权重乘 :math:`(b-a)/2`）。"""
    x, w = gauss_legendre(n)
    half = 0.5 * (float(b) - float(a))
    mid = 0.5 * (float(a) + float(b))
    return mid + half * x, half * w


# --------------------------------------------------------------------------- #
# 一维拉格朗日基（任意节点集，重心公式）
# --------------------------------------------------------------------------- #
def _barycentric_weights(nodes: np.ndarray) -> np.ndarray:
    r""":math:`w_k = 1/\prod_{j\ne k}(x_k-x_j)`（重心权重）。"""
    n = nodes.size
    w = np.ones(n, dtype=float)
    for k in range(n):
        prod = 1.0
        for j in range(n):
            if j != k:
                prod *= float(nodes[k] - nodes[j])
        w[k] = 1.0 / prod
    return w


def lagrange_values(nodes: Any, x: Any) -> np.ndarray:
    r"""节点集 ``nodes`` 上的拉格朗日基在 ``x`` 处的值，返回 ``(n_node, len(x))``。

    用重心公式 :math:`\ell_k(x)=\frac{w_k/(x-x_k)}{\sum_j w_j/(x-x_j)}`
    （第二重心公式在浮点下比"逐项乘 (x-x_j)"稳定得多，节点数到十几也不掉精度）；
    ``x`` 恰好落在某个节点上时用 :math:`\ell_k(x_i)=\delta_{ki}` 的精确恒等式，
    避免 0/0。
    """
    xn = np.asarray(nodes, dtype=float).reshape(-1)
    xv = np.atleast_1d(np.asarray(x, dtype=float))
    w = _barycentric_weights(xn)
    V = np.empty((xn.size, xv.size), dtype=float)
    for j, xj in enumerate(xv):
        diff = xj - xn
        hit = np.flatnonzero(diff == 0.0)
        if hit.size:
            V[:, j] = 0.0
            V[int(hit[0]), j] = 1.0
            continue
        t = w / diff
        V[:, j] = t / float(np.sum(t))
    return V


def lagrange_values_and_derivs(nodes: Any, x: Any) -> tuple[np.ndarray, np.ndarray]:
    r"""拉格朗日基的值与一阶导，返回 ``(V, D)``，形状都是 ``(n_node, len(x))``。

    导数用第二重心公式的微分形式

    .. math::
        \ell_k'(x)=\ell_k(x)\Bigl(T(x)-\frac{1}{x-x_k}\Bigr),\qquad
        T(x)=\frac{\sum_j w_j/(x-x_j)^2}{\sum_j w_j/(x-x_j)},

    节点上则用精确恒等式 :math:`\ell_k'(x_i)=\frac{w_k}{w_i(x_i-x_k)}\ (i\ne k)`、
    :math:`\ell_k'(x_k)=-\sum_{i\ne k}\ell_i'(x_k)`（每行和为零）。
    """
    xn = np.asarray(nodes, dtype=float).reshape(-1)
    xv = np.atleast_1d(np.asarray(x, dtype=float))
    w = _barycentric_weights(xn)
    n = xn.size
    V = np.empty((n, xv.size), dtype=float)
    D = np.empty((n, xv.size), dtype=float)
    Dmat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for k in range(n):
            if i != k:
                Dmat[i, k] = w[k] / (w[i] * (xn[i] - xn[k]))
        Dmat[i, i] = -float(np.sum(Dmat[i]))
    for j, xj in enumerate(xv):
        diff = xj - xn
        hit = np.flatnonzero(diff == 0.0)
        if hit.size:
            V[:, j] = 0.0
            V[int(hit[0]), j] = 1.0
            D[:, j] = Dmat[int(hit[0])]
            continue
        t = w / diff
        s = float(np.sum(t))
        V[:, j] = t / s
        tt = float(np.sum(w / (diff * diff)))
        D[:, j] = V[:, j] * (tt / s - 1.0 / diff)
    return V, D


# --------------------------------------------------------------------------- #
# Hermite 三次基（参考单元 [0,1]，长度 h）
# --------------------------------------------------------------------------- #
def hermite_values(xi: Any, h: float) -> np.ndarray:
    r"""Hermite 三次基在参考点 :math:`\xi\in[0,1]` 上的值，形状 ``(4, len(xi))``。

    顺序是 :math:`(N_1,N_2,N_3,N_4)=(w_i,\ w_i',\ w_{i+1},\ w_{i+1}')` 对应的基：

    .. math::
        N_1=1-3\xi^2+2\xi^3,\quad N_2=h(\xi-2\xi^2+\xi^3),\quad
        N_3=3\xi^2-2\xi^3,\quad N_4=h(-\xi^2+\xi^3).
    """
    x = np.atleast_1d(np.asarray(xi, dtype=float))
    hh = float(h)
    return np.vstack(
        [
            1.0 - 3.0 * x**2 + 2.0 * x**3,
            hh * (x - 2.0 * x**2 + x**3),
            3.0 * x**2 - 2.0 * x**3,
            hh * (-x**2 + x**3),
        ]
    )


def hermite_second_derivs(xi: Any, h: float) -> tuple[np.ndarray, np.ndarray]:
    r"""Hermite 基对**物理坐标**的一阶、二阶导 ``(dN/dx, d2N/dx2)``，形状各 ``(4, len(xi))``。

    梁的刚度阵是 :math:`\int EI\,N_i''N_j''\,dx`，所以二阶导不是可选项。
    """
    x = np.atleast_1d(np.asarray(xi, dtype=float))
    hh = float(h)
    d1 = np.vstack(
        [
            -6.0 * x + 6.0 * x**2,
            hh * (1.0 - 4.0 * x + 3.0 * x**2),
            6.0 * x - 6.0 * x**2,
            hh * (-2.0 * x + 3.0 * x**2),
        ]
    ) / hh
    d2 = np.vstack(
        [
            -6.0 + 12.0 * x,
            hh * (-4.0 + 6.0 * x),
            6.0 - 12.0 * x,
            hh * (-2.0 + 6.0 * x),
        ]
    ) / (hh * hh)
    return d1, d2


# --------------------------------------------------------------------------- #
# 二维形函数
# --------------------------------------------------------------------------- #
def tri_p1_values(xi: Any, eta: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""参考三角形上的 :math:`P_1` 基：:math:`(1-\xi-\eta,\ \xi,\ \eta)`，返回 ``(N, dN/dξ, dN/dη)``。"""
    x = np.atleast_1d(np.asarray(xi, dtype=float))
    y = np.atleast_1d(np.asarray(eta, dtype=float))
    zero = np.zeros_like(x)
    one = np.ones_like(x)
    N = np.vstack([1.0 - x - y, x, y])
    dxi = np.vstack([-one, one, zero])
    deta = np.vstack([-one, zero, one])
    return N, dxi, deta


def quad_q1_values(xi: Any, eta: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""参考正方形 :math:`[-1,1]^2` 上的双线性基，顶点顺序 :math:`(-1,-1),(1,-1),(1,1),(-1,1)`。

    返回 ``(N, dN/dξ, dN/dη)``，形状各 ``(4, len(xi))``。
    """
    x = np.atleast_1d(np.asarray(xi, dtype=float))
    y = np.atleast_1d(np.asarray(eta, dtype=float))
    sx = np.vstack([-np.ones_like(x), np.ones_like(x), np.ones_like(x), -np.ones_like(x)])
    sy = np.vstack([-np.ones_like(y), -np.ones_like(y), np.ones_like(y), np.ones_like(y)])
    N = 0.25 * (1.0 + sx * x) * (1.0 + sy * y)
    dxi = 0.25 * sx * (1.0 + sy * y)
    deta = 0.25 * sy * (1.0 + sx * x)
    return N, dxi, deta


# --------------------------------------------------------------------------- #
# 求积规则（参考单元）
# --------------------------------------------------------------------------- #
def quad_rule(n: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""张量积 Gauss 规则（参考正方形），返回 ``(xi, eta, w)``。

    ``n`` 点一维 Gauss 的张量积对 :math:`\xi,\eta` 各自 :math:`2n-1` 次多项式精确。
    :math:`Q_1` 刚度阵的被积函数（仿射单元）是多项式，2 点就够；
    畸变四边形上它是**有理函数**，任何点数都不精确——这一点在章节里实测。
    """
    g, w1 = gauss_legendre(int(n))
    xi, eta = np.meshgrid(g, g, indexing="ij")
    ww = np.outer(w1, w1)
    return xi.reshape(-1), eta.reshape(-1), ww.reshape(-1)


#: 单纯形规则表：``{点数: (重心坐标, 权重)}``（权重和为 1/2，即参考三角形面积）。
_TRI_RULES: dict[int, tuple[np.ndarray, np.ndarray]] = {
    1: (np.asarray([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]), np.asarray([0.5])),
    3: (
        np.asarray([[2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                    [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]]),
        np.asarray([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]),
    ),
    6: (
        np.asarray(
            [
                [0.816847572980459, 0.091576213509771, 0.091576213509771],
                [0.091576213509771, 0.816847572980459, 0.091576213509771],
                [0.091576213509771, 0.091576213509771, 0.816847572980459],
                [0.108103018168070, 0.445948490915965, 0.445948490915965],
                [0.445948490915965, 0.108103018168070, 0.445948490915965],
                [0.445948490915965, 0.445948490915965, 0.108103018168070],
            ]
        ),
        np.asarray(
            [0.109951743655322, 0.109951743655322, 0.109951743655322,
             0.223381589678011, 0.223381589678011, 0.223381589678011]
        ),
    ),
}


def tri_rule(n: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""参考三角形上的对称求积规则，返回 ``(xi, eta, w)``。

    可选 ``n=1``（形心，1 次精确）、``n=3``（2 次精确）、``n=6``（4 次精确）。
    :math:`P_1` 的刚度阵是常数、质量阵是 2 次多项式，3 点规则就精确。
    """
    if int(n) not in _TRI_RULES:
        raise ValueError(f"没有 {n} 点的单纯形规则（可选 {sorted(_TRI_RULES)}）")
    bary, w = _TRI_RULES[int(n)]
    return bary[:, 1].copy(), bary[:, 2].copy(), w.copy()


# --------------------------------------------------------------------------- #
# 等参映射
# --------------------------------------------------------------------------- #
def map_tri(xi: Any, eta: Any, corners: Any) -> np.ndarray:
    r"""三角形上的**仿射**映射 :math:`F_K(\xi,\eta)=p_0+\xi(p_1-p_0)+\eta(p_2-p_0)`。"""
    p = np.asarray(corners, dtype=float)
    x = np.atleast_1d(np.asarray(xi, dtype=float))
    y = np.atleast_1d(np.asarray(eta, dtype=float))
    return p[0][None, :] + x[:, None] * (p[1] - p[0])[None, :] + y[:, None] * (p[2] - p[0])[None, :]


def tri_jacobian(corners: Any) -> np.ndarray:
    r"""三角形映射的 Jacobi 矩阵（常数，与参考点无关）。"""
    p = np.asarray(corners, dtype=float)
    return np.column_stack([p[1] - p[0], p[2] - p[0]])


def map_quad(xi: Any, eta: Any, corners: Any) -> np.ndarray:
    r"""双线性等参映射：:math:`F_K=\sum_i N_i(\xi,\eta)\,p_i`（:math:`N_i` 是 :func:`quad_q1_values`）。"""
    p = np.asarray(corners, dtype=float)
    N, _, _ = quad_q1_values(xi, eta)
    return N.T @ p


def quad_jacobian(xi: Any, eta: Any, corners: Any) -> np.ndarray:
    r"""双线性映射的 Jacobi 矩阵，返回形状 ``(len(xi), 2, 2)``（**逐点**，不是常数）。

    .. math::
        J(\xi,\eta)=\sum_i p_i\otimes\nabla_{\!\hat x}N_i(\xi,\eta)\in\mathbb R^{2\times2}.
    """
    p = np.asarray(corners, dtype=float)
    _, dxi, deta = quad_q1_values(xi, eta)
    # J[a, b] = sum_i p[i, a] * dN_i/dxhat_b
    J = np.einsum("ia,ikb->kab", p, np.stack([dxi, deta], axis=-1))
    return J


def quad_jacobian_det(xi: Any, eta: Any, corners: Any) -> np.ndarray:
    r"""双线性映射的 :math:`\det J`（逐点）。四边形畸变到自交时它会变号，网格就此失效。"""
    J = quad_jacobian(xi, eta, corners)
    return J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]


def simplex_measure(corners: Any) -> float:
    r"""参考单纯形映到物理三角形后的面积 :math:`|\det J|/2`。"""
    J = tri_jacobian(corners)
    return 0.5 * abs(float(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]))


def isoparametric_ratio(xi: Any, eta: Any, corners: Any) -> float:
    r"""单元内 :math:`\max\det J/\min\det J`：等参映射"有多不仿射"的度量。

    仿射单元（三角形、平行四边形）恒等于 1；畸变四边形上它大于 1，
    这个比值正是"取中心 J 当常数"会带来的相对误差的量级。
    """
    d = quad_jacobian_det(xi, eta, corners)
    lo = float(np.min(d))
    hi = float(np.max(d))
    if lo <= 0.0:
        return float("inf")
    return hi / lo
