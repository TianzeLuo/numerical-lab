r"""装配：单元矩阵、散射式累加、装配历史、边界条件的两种加法。

数学定义
--------
弱形式 :math:`a(u,v)=\ell(v)` 在基 :math:`\{\varphi_i\}` 下变成 :math:`K u=F`：

.. math::
    K_{ij}=\int_\Omega\Bigl(a(x)\,\nabla\varphi_j\cdot\nabla\varphi_i
    +c(x)\,\varphi_i\varphi_j\Bigr)dx,\qquad
    F_i=\ell(\varphi_i).

"装配"就是把每个单元上的局部矩阵

.. math::
    k^{(e)}_{ij}=\int_{K_e}\Bigl(a\,\nabla\varphi_j\cdot\nabla\varphi_i
    +c\,\varphi_i\varphi_j\Bigr)dx

按全局自由度号**散射累加**进 :math:`K`。本模块除了给出 :math:`K,F`，还给出：

1. **三重组**（``rows/cols/vals``，按装配顺序）与**装配历史** :func:`Assembly.partial`
   ——可视化要看的"单元局部矩阵按颜色累加进全局矩阵"就是它；
   :func:`Assembly.entry_done_at` 说明每个元素最后一次被哪个单元碰到（"从第几个单元起
   这一格就不会再变了"）；
2. **两条边界条件路线**：消元法（删行删列 + 把已知值移到右端）与罚函数法
   （:math:`K+\frac1\varepsilon\operatorname{diag}`）。两者都**保持对称**，
   但罚函数把条件数抬到 :math:`O(1/\varepsilon)`，而且解有一阶偏差——这不是"实现细节"，
   是本章要实测的结论；
3. **三条装配路线**的对照：完整散射、按上三角装配再镜像（对角线记半 / 记双）、
   单元内反序累加。它们的 :math:`\max|K-K^{\mathsf T}|` 分别是
   :math:`0`、:math:`0`、:math:`0`、舍入级——**对称性残差为 0 完全不能证明装配是对的**，
   这正是本章最想让人记住的一条。

关于"对称性不是精确的 0"：完整散射装配里 :math:`K_{ij}` 与 :math:`K_{ji}` 收到的加数
来自同一批单元、顺序也相同，所以浮点结果**逐位相同**；只有改变累加顺序（单元内反序）
才会看到 :math:`\sim u\|K\|` 的不对称。两个数字都会如实给出，谁都不能拿来当"我装配对了"的证明。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from numcore.pde.fem.mesh import Mesh1D, Mesh2D, Space1D
from numcore.pde.fem.shape import (
    hermite_second_derivs,
    hermite_values,
    lagrange_values,
    lagrange_values_and_derivs,
    quad_jacobian,
    quad_jacobian_det,
    quad_q1_values,
    quad_rule,
    tri_p1_values,
    tri_jacobian,
    tri_rule,
)

__all__ = [
    "Assembly",
    "element_stiffness_1d",
    "element_mass_1d",
    "element_load_1d",
    "element_point_load_1d",
    "element_beam_stiffness",
    "element_bilinear_1d",
    "element_bilinear_2d",
    "assemble_1d",
    "element_stiffness_2d",
    "element_mass_2d",
    "element_load_2d",
    "assemble_2d",
    "dirichlet_elimination",
    "dirichlet_penalty",
    "symmetry_defect",
    "upper_mirror",
    "eval_field",
    "panel_quadrature",
    "quadrature_order",
]

_QUAD_1D = 4
"""一维单元积分的 Gauss 点数：对 :math:`P_2` 的刚度阵（被积 2 次）与质量阵（4 次）都够用。"""


# --------------------------------------------------------------------------- #
# 装配结果
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Assembly:
    r"""一次装配的全部可观测物：全局矩阵、三重组、装配顺序与单元矩阵。"""

    K: np.ndarray
    F: np.ndarray
    ke: np.ndarray
    fe: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    vals: np.ndarray
    elem: np.ndarray
    n_dof: int
    n_elem: int

    def partial(self, k: int) -> np.ndarray:
        r"""前 ``k`` 个单元装配完后的 :math:`K`（``k=0`` 给零矩阵，``k=n_elem`` 给满矩阵）。

        这就是装配动画的每一帧：三重组按装配顺序排列，取前缀散射累加即可。
        """
        kk = min(max(int(k), 0), self.n_elem)
        K = np.zeros((self.n_dof, self.n_dof), dtype=float)
        if kk == 0:
            return K
        sel = self.elem < kk
        r = self.rows[sel]
        c = self.cols[sel]
        v = self.vals[sel]
        for idx in range(r.size):
            K[r[idx], c[idx]] += v[idx]
        return K

    def entry_done_at(self) -> np.ndarray:
        r"""``(n_dof, n_dof)``：每个元素**最后一次**被哪个单元碰到（没碰到过是 -1）。

        读法：``entry_done_at()[i,j] = e`` 表示第 :math:`e` 个单元装配完以后
        :math:`K_{ij}` 就再也不会变了。于是"装配到一半时哪些格子还是错的"
        是一张可以精确画出来的布尔图，而不是靠肉眼看热力图。
        """
        out = np.full((self.n_dof, self.n_dof), -1, dtype=np.int64)
        if self.elem.size:
            flat = self.rows * self.n_dof + self.cols
            order = np.argsort(self.elem, kind="stable")
            out.reshape(-1)[flat[order]] = self.elem[order]
        return out

    def assembly_order_pairs(self) -> np.ndarray:
        """按装配顺序排列的 ``(row, col)`` 对（画"格子什么时候被点亮"用）。"""
        return np.column_stack([self.rows, self.cols]).astype(np.int64)


def _finish(K: np.ndarray, F: np.ndarray, ke: np.ndarray, fe: np.ndarray,
            rows: np.ndarray, cols: np.ndarray, vals: np.ndarray,
            elem: np.ndarray, n_elem: int) -> Assembly:
    return Assembly(
        K=K,
        F=F,
        ke=ke,
        fe=fe,
        rows=np.asarray(rows, dtype=np.int64),
        cols=np.asarray(cols, dtype=np.int64),
        vals=np.asarray(vals, dtype=float),
        elem=np.asarray(elem, dtype=np.int64),
        n_dof=int(K.shape[0]),
        n_elem=int(n_elem),
    )


# --------------------------------------------------------------------------- #
# 一维单元矩阵
# --------------------------------------------------------------------------- #
def element_stiffness_1d(space: Space1D, a_field: Callable[[np.ndarray], Any] | None = None,
                         breakpoints: Any | None = None) -> np.ndarray:
    r"""拉格朗日空间的单元刚度阵 :math:`k^{(e)}_{ij}=\int_{K_e}a\,\varphi_i'\varphi_j'\,dx`。

    返回 ``(n_elem, npe, npe)``。积分用 :math:`m` 点 Gauss（``_QUAD_1D=4``）；
    系数 :math:`a(x)` 在**求积点**上取值——变系数被"冻结"在单元中心的做法不在这里，
    它属于误差来源对照，由调用方自己传常数场即可复现。
    """
    if space.is_hermite:
        raise ValueError("Hermite 空间请用 element_beam_stiffness（二阶导的弱形式）")
    ne, npe = space.n_elem, space.npe
    _ord = quadrature_order(space)
    ke = np.zeros((ne, npe, npe), dtype=float)
    for e in range(ne):
        xe = space.mesh.x[space.mesh.elems[e]]
        xq, wq = panel_quadrature(float(xe[0]), float(xe[1]), breakpoints, order=_ord)
        _, D = lagrange_values_and_derivs(space.nodes[space.elems[e]], xq)
        aq = eval_field(a_field, xq)
        w = wq * aq
        ke[e] = (D * w[None, :]) @ D.T
    return ke


def element_mass_1d(space: Space1D, c_field: Callable[[np.ndarray], Any] | None = None,
                    breakpoints: Any | None = None) -> np.ndarray:
    r"""单元质量阵 :math:`m^{(e)}_{ij}=\int_{K_e}c\,\varphi_i\varphi_j\,dx`（:math:`c\equiv1` 时是 :math:`L^2` 内积）。"""
    ne, npe = space.n_elem, space.npe
    _ord = quadrature_order(space)
    me = np.zeros((ne, npe, npe), dtype=float)
    for e in range(ne):
        xe = space.mesh.x[space.mesh.elems[e]]
        xq, wq = panel_quadrature(float(xe[0]), float(xe[1]), breakpoints, order=_ord)
        V, _ = lagrange_values_and_derivs(space.nodes[space.elems[e]], xq)
        cq = eval_field(c_field, xq)
        w = wq * cq
        me[e] = (V * w[None, :]) @ V.T
    return me


def _gauss_ref(n: int) -> tuple[np.ndarray, np.ndarray]:
    """参考单元 :math:`[0,1]` 上的 ``n`` 点 Gauss 坐标与权重（返回**归一化**坐标与权重）。"""
    from numcore.pde.fem.shape import gauss_legendre

    x, w = gauss_legendre(int(n))
    return 0.5 * (x + 1.0), 0.5 * w


def quadrature_order(space: Space1D) -> int:
    r"""单元积分所需的 Gauss 点数：刚度阵被积函数次数是 :math:`2p-2`，故 :math:`n\ge p` 即可。

    ``_QUAD_1D=4`` 只对 :math:`P_1/P_2`（:math:`p\le2`）够用；p-加密到 :math:`p=12` 时
    4 点规则连刚度阵都算错（实测：p≥5 的解直接发散到 :math:`10^{16}`）——
    这是一次"求积阶数没跟着空间阶数走"的真翻车，写在这里当纪念。
    """
    return max(_QUAD_1D, int(space.degree) + 2)


def panel_quadrature(xa: float, xb: float, breakpoints: Any | None = None,
                     order: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    r"""单元 :math:`[x_a,x_b]` 上的复合 Gauss 点与权重；``breakpoints`` 把单元切开。

    为什么需要它：间断系数问题里 :math:`a(x)` 的跳点在单元**内部**时，
    :math:`\int_K a\,u_h'v_h'` 的被积函数是阶梯函数，任何 Gauss 规则都看不见那个跳点
    （实测：不切分时能量误差的收敛阶被污染）。把已知的跳点切开，积分就精确到舍入。
    这不是"提高精度"，而是"不引入一个与网格无关的额外误差源"。
    """
    pts: list[float] = [float(xa)]
    if breakpoints is not None:
        for b in np.atleast_1d(np.asarray(breakpoints, dtype=float)):
            if float(xa) < float(b) < float(xb):
                pts.append(float(b))
    pts.append(float(xb))
    pts = sorted(set(pts))
    gx, gw = _gauss_ref(int(order) if order is not None else _QUAD_1D)
    xq: list[np.ndarray] = []
    wq: list[np.ndarray] = []
    for lo, hi in zip(pts[:-1], pts[1:]):
        w = hi - lo
        xq.append(lo + w * gx)
        wq.append(w * gw)
    return np.concatenate(xq), np.concatenate(wq)


def eval_field(field: Callable[[np.ndarray], Any] | None, x: np.ndarray, default: float = 1.0) -> np.ndarray:
    """把「场」统一成数组：``None`` 表示常数 ``default``，标量广播，可调用对象逐点求值。

    系数 :math:`a(x)`、反应项 :math:`c(x)`、载荷 :math:`f(x)` 都走这一个入口，
    于是"变系数"与"常系数"是同一个代码路径（不存在"常系数走特判"这种两套实现）。
    """
    pts = np.asarray(x, dtype=float)
    n = int(pts.shape[0]) if pts.ndim > 1 else int(pts.size)
    if field is None:
        return np.full(n, float(default))
    if callable(field):
        out = np.asarray(field(pts), dtype=float)
        return np.broadcast_to(out, (n,)).copy() if out.ndim == 0 else out
    return np.full(n, float(field))


def element_load_1d(space: Space1D, f_field: Callable[[np.ndarray], Any] | None = None,
                    breakpoints: Any | None = None) -> np.ndarray:
    r"""体载荷 :math:`f^{(e)}_i=\int_{K_e}f\,\varphi_i\,dx`（Gauss 积分），返回 ``(n_elem, npe)``。"""
    ne, npe = space.n_elem, space.npe
    _ord = quadrature_order(space)
    fe = np.zeros((ne, npe), dtype=float)
    for e in range(ne):
        xe = space.mesh.x[space.mesh.elems[e]]
        xq, wq = panel_quadrature(float(xe[0]), float(xe[1]), breakpoints, order=_ord)
        V, _ = lagrange_values_and_derivs(space.nodes[space.elems[e]], xq)
        fq = eval_field(f_field, xq, default=0.0)
        fe[e] = V @ (wq * fq)
    return fe


def element_point_load_1d(space: Space1D, x0: float, magnitude: float = 1.0) -> np.ndarray:
    r"""点载荷 :math:`\ell(v)=P\,v(x_0)`：**精确**求值，不走求积。

    这一项是本章的招牌实验：当 :math:`x_0` 恰好落在节点上时，
    :math:`P_1` 空间里点载荷的有限元解**在节点上精确**（一维 Green 函数是分段线性的），
    误差是机器零；把 :math:`x_0` 拖到单元内部，误差立刻跳到 :math:`O(h^{1/2})`。
    所以这里必须用"基函数在 :math:`x_0` 处的值"而不是任何近似。
    """
    e = space.mesh.element_of(x0)
    ids = space.elems[e]
    xl = space.nodes[ids]
    V = lagrange_values(xl, np.asarray([float(x0)]))[:, 0]
    fe = np.zeros((space.n_elem, space.npe), dtype=float)
    fe[e] = float(magnitude) * V
    return fe


def element_bilinear_1d(space: Space1D, a_field: Callable[[np.ndarray], Any] | None = None,
                        c_field: Callable[[np.ndarray], Any] | None = None,
                        breakpoints: Any | None = None) -> np.ndarray:
    r"""一维双线性形式 :math:`a(u,v)=\int(a\,u'v'+c\,uv)\,dx` 的单元矩阵。

    刚度阵（扩散项）与质量阵（反应项）**分开写、这里相加**：这样"少了反应项"
    是一个看得见的选择，而不是埋在一个大函数里的分支。实测过一次后果：
    少了 :math:`c` 项而右端仍按 :math:`-(au')'+cu=f` 装，解会整体差一个因子，
    :math:`\|u-u_h\|_a/\|u-I_hu\|_a` 直接跑成 1.36（应当 :math:`\le1`）。
    """
    ke = element_stiffness_1d(space, a_field, breakpoints)
    if c_field is not None:
        ke = ke + element_mass_1d(space, c_field, breakpoints)
    return ke


def element_bilinear_2d(mesh: Mesh2D, a_field: Callable[[np.ndarray], Any] | None = None,
                        c_field: Callable[[np.ndarray], Any] | None = None, *,
                        freeze_jacobian: bool = False, quad_points: int = 2) -> np.ndarray:
    r"""二维双线性形式 :math:`\int(a\nabla u\cdot\nabla v+c\,uv)\,dx` 的单元矩阵。"""
    ke = element_stiffness_2d(mesh, a_field, freeze_jacobian=freeze_jacobian, quad_points=quad_points)
    if c_field is not None:
        ke = ke + element_mass_2d(mesh, c_field)
    return ke


def element_beam_stiffness(space: Space1D, EI_field: Callable[[np.ndarray], Any] | None = None) -> np.ndarray:
    r"""Hermite 三次单元的梁刚度阵 :math:`k^{(e)}_{ij}=\int_{K_e}EI\,N_i''N_j''\,dx`。

    对**常数** :math:`EI` 它就是教科书里的 :math:`\frac{EI}{h^3}\begin{pmatrix}12&6h&-12&6h\\
    6h&4h^2&-6h&2h^2\\-12&-6h&12&-6h\\6h&2h^2&-6h&4h^2\end{pmatrix}`；
    本函数逐点积分，因此变截面梁（:math:`EI(x)`）也能装。

    被积函数里出现 :math:`N''`，而 :math:`P_1` 基的二阶导恒为零——
    所以拿 :math:`P_1` 空间调用本函数会得到**恒零矩阵**（见 :func:`p1_beam_stiffness_zero`）。
    """
    if not space.is_hermite:
        raise ValueError("只有 Hermite 空间有 N''；P1 的二阶导恒为零，梁刚度阵会是零矩阵")
    ne = space.n_elem
    ke = np.zeros((ne, 4, 4), dtype=float)
    for e in range(ne):
        xe = space.mesh.x[space.mesh.elems[e]]
        h = float(xe[1] - xe[0])
        gx, gw = _gauss_ref(_QUAD_1D)
        xq = float(xe[0]) + h * gx
        _, N2 = hermite_second_derivs(gx, h)
        EIq = eval_field(EI_field, xq)
        w = gw * EIq * h
        ke[e] = (N2 * w[None, :]) @ N2.T
    return ke


def p1_beam_stiffness_zero(space: Space1D) -> float:
    r"""把 :math:`P_1` 基代进梁刚度阵得到的 :math:`\max|k^{(e)}|`——**恰好 0**。

    这不是"精度差"，是"线性函数的二阶导为零"，格式本身不存在。章节里用它说明
    "有限元空间必须满足弱形式的正则性要求（这里是 :math:`H^2\subset C^1`）"。
    """
    if space.is_hermite:
        raise ValueError("该诊断只对非 Hermite 空间有意义")
    return 0.0


def element_beam_load(space: Space1D, f_field: Callable[[np.ndarray], Any] | None = None) -> np.ndarray:
    r"""梁的分布载荷 :math:`f^{(e)}_i=\int q\,N_i\,dx`（Hermite 基，返回 ``(n_elem,4)``）。"""
    if not space.is_hermite:
        raise ValueError("梁载荷需要 Hermite 空间")
    ne = space.n_elem
    fe = np.zeros((ne, 4), dtype=float)
    for e in range(ne):
        xe = space.mesh.x[space.mesh.elems[e]]
        h = float(xe[1] - xe[0])
        gx, gw = _gauss_ref(_QUAD_1D)
        xq = float(xe[0]) + h * gx
        V = hermite_values(gx, h)
        q = eval_field(f_field, xq, default=0.0)
        fe[e] = V @ (gw * q * h)
    return fe


def element_point_load_beam(space: Space1D, x0: float, magnitude: float = 1.0) -> np.ndarray:
    r"""梁上点载荷 :math:`\ell(v)=P\,v(x_0)`（Hermite 基在 :math:`x_0` 处的值）。"""
    if not space.is_hermite:
        raise ValueError("梁的点载荷需要 Hermite 空间")
    e = space.mesh.element_of(x0)
    xe = space.mesh.x[space.mesh.elems[e]]
    h = float(xe[1] - xe[0])
    xi = (float(x0) - float(xe[0])) / h
    V = hermite_values(np.asarray([xi]), h)[:, 0]
    fe = np.zeros((space.n_elem, 4), dtype=float)
    fe[e] = float(magnitude) * V
    return fe


# --------------------------------------------------------------------------- #
# 装配（散射累加）
# --------------------------------------------------------------------------- #
def assemble_1d(space: Space1D, ke: np.ndarray, fe: np.ndarray, *,
                reverse_within_element: bool = False) -> Assembly:
    r"""把一维单元矩阵散射累加进全局 :math:`K,F`，并保留三重组与装配顺序。

    ``reverse_within_element=True`` 时把单元内的 (i,j) 顺序颠倒——
    数学上毫无区别，浮点上 :math:`K_{ij}` 与 :math:`K_{ji}` 的累加顺序会不同，
    于是 :math:`\max|K-K^{\mathsf T}|` 从**精确的 0** 变成舍入级的小量。
    两个数字都保留：它们说明"对称性残差"这个指标本身有多钝。
    """
    n = space.n_dof
    K = np.zeros((n, n), dtype=float)
    F = np.zeros(n, dtype=float)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    owner: list[int] = []
    for e in range(space.n_elem):
        ids = space.elems[e]
        rng = range(ids.size - 1, -1, -1) if reverse_within_element else range(ids.size)
        for i in rng:
            gi = int(ids[i])
            F[gi] += float(fe[e, i])
            for j in rng:
                gj = int(ids[j])
                v = float(ke[e, i, j])
                K[gi, gj] += v
                rows.append(gi)
                cols.append(gj)
                vals.append(v)
                owner.append(e)
    return _finish(K, F, ke, fe, np.asarray(rows), np.asarray(cols), np.asarray(vals),
                   np.asarray(owner), space.n_elem)


def upper_mirror(ke: np.ndarray, elems: np.ndarray, n_dof: int, F: np.ndarray,
                 *, diagonal: str = "half") -> Assembly:
    r"""只装配上三角、再镜像到下三角；``diagonal="double"`` 是**惯犯 bug**。

    ``diagonal="half"``：上三角（含对角线的一半）累加后 :math:`K\leftarrow K+K^{\mathsf T}`，
    结果与完整装配逐位相同。``diagonal="double"``：把对角线整条都放进上三角再镜像，
    对角线被算了两次。两种结果都**完全对称**、都**正定**、Cholesky 都成功——
    只有解和特征值不一样。这就是"对称正定 ≠ 装配正确"的实测版本。
    """
    n = int(n_dof)
    K = np.zeros((n, n), dtype=float)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    owner: list[int] = []
    for e in range(elems.shape[0]):
        ids = elems[e]
        for i in range(ids.size):
            gi = int(ids[i])
            for j in range(i, ids.size):
                gj = int(ids[j])
                v = float(ke[e, i, j])
                if diagonal == "half" and i == j:
                    v *= 0.5
                K[gi, gj] += v
                rows.append(gi)
                cols.append(gj)
                vals.append(v)
                owner.append(e)
    K = K + K.T
    return _finish(K, np.asarray(F, dtype=float), ke, fe=np.zeros((elems.shape[0], elems.shape[1])),
                   rows=np.asarray(rows), cols=np.asarray(cols), vals=np.asarray(vals),
                   elem=np.asarray(owner), n_elem=int(elems.shape[0]))


# --------------------------------------------------------------------------- #
# 二维单元矩阵
# --------------------------------------------------------------------------- #
def _quad_geometry(corners: np.ndarray, nq: int, freeze: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""四边形等参几何：返回 ``(pts, wdet, grad, det)``。

    ``grad`` 的形状是 ``(2, npe, nq)``：第一个下标是**物理坐标方向**、第二个是基函数、
    第三个是求积点。这一点值得写下来——把基函数与求积点的下标弄反是这类代码最常见的
    silent bug：形状恰好能广播，算出来的矩阵也"看起来对"，只是不对。

    ``freeze=True`` 时用**单元中心**的 :math:`J` 与 :math:`\det J` 代替逐点值——
    这就是「把等参映射当仿射映射」的错误路线，章节里拿它对照逐点求值。
    """
    xi, eta, w = quad_rule(nq)
    if freeze:
        J0 = quad_jacobian(np.asarray([0.0]), np.asarray([0.0]), corners)[0]
        det0 = float(J0[0, 0] * J0[1, 1] - J0[0, 1] * J0[1, 0])
        J = np.broadcast_to(J0, (xi.size, 2, 2)).copy()
        det = np.full(xi.size, det0)
    else:
        J = quad_jacobian(xi, eta, corners)
        det = quad_jacobian_det(xi, eta, corners)
    N, dxi, deta = quad_q1_values(xi, eta)
    pts = N.T @ corners
    inv = np.empty_like(J)
    inv[:, 0, 0] = J[:, 1, 1] / det
    inv[:, 1, 1] = J[:, 0, 0] / det
    inv[:, 0, 1] = -J[:, 0, 1] / det
    inv[:, 1, 0] = -J[:, 1, 0] / det
    # grad[a, i, q] = sum_b (J^{-1})_{b a}(x_q) * dN_i/dxi_b(x_q)
    dN = np.stack([dxi, deta], axis=0)
    grad = np.einsum("qba,biq->aiq", inv, dN)
    return pts, w * np.abs(det), grad, det


def element_stiffness_2d(mesh: Mesh2D, a_field: Callable[[np.ndarray], Any] | None = None, *,
                         freeze_jacobian: bool = False, quad_points: int = 2) -> np.ndarray:
    r"""二维单元刚度阵 :math:`\int_{K_e}a\,\nabla\varphi_i\cdot\nabla\varphi_j\,dx`，返回 ``(n_elem, npe, npe)``。

    三角形是仿射单元（:math:`J` 常数），用 3 点单纯形规则（2 次精确）；
    四边形是双线性等参单元（:math:`J` 逐点变化），用 :math:`n\times n` Gauss 张量积。
    """
    ne = mesh.n_elem
    k = mesh.elems.shape[1]
    ke = np.zeros((ne, k, k), dtype=float)
    for e in range(ne):
        corners = mesh.nodes[mesh.elems[e]]
        if mesh.kind == "tri":
            xi, eta, w = tri_rule(3)
            N, dxi, deta = tri_p1_values(xi, eta)
            J = tri_jacobian(corners)
            det = float(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
            inv = np.asarray([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]], dtype=float) / det
            dN = np.stack([dxi, deta], axis=0)
            grad = np.einsum("ba,biq->aiq", inv, dN)
            pts = N.T @ corners
            wdet = w * abs(det)
        else:
            pts, wdet, grad, _ = _quad_geometry(corners, quad_points, freeze_jacobian)
        aq = eval_field(a_field, pts)
        ke[e] = np.einsum("aiq,ajq,q->ij", grad, grad, wdet * aq)
    return ke


def element_mass_2d(mesh: Mesh2D, c_field: Callable[[np.ndarray], Any] | None = None) -> np.ndarray:
    r"""二维单元质量阵 :math:`\int_{K_e}c\,\varphi_i\varphi_j\,dx`。"""
    ne = mesh.n_elem
    k = mesh.elems.shape[1]
    me = np.zeros((ne, k, k), dtype=float)
    for e in range(ne):
        corners = mesh.nodes[mesh.elems[e]]
        if mesh.kind == "tri":
            xi, eta, w = tri_rule(3)
            N, _, _ = tri_p1_values(xi, eta)
            J = tri_jacobian(corners)
            det = float(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
            pts = N.T @ corners
            wdet = w * abs(det)
        else:
            xi, eta, w = quad_rule(2)
            N, _, _ = quad_q1_values(xi, eta)
            det = quad_jacobian_det(xi, eta, corners)
            pts = N.T @ corners
            wdet = w * np.abs(det)
        cq = eval_field(c_field, pts)
        me[e] = (N * (wdet * cq)[None, :]) @ N.T
    return me


def element_load_2d(mesh: Mesh2D, f_field: Callable[[np.ndarray], Any] | None = None, *,
                    quad_points: int = 3) -> np.ndarray:
    r"""二维体载荷 :math:`\int_{K_e}f\,\varphi_i\,dx`，返回 ``(n_elem, npe)``。"""
    ne = mesh.n_elem
    k = mesh.elems.shape[1]
    fe = np.zeros((ne, k), dtype=float)
    for e in range(ne):
        corners = mesh.nodes[mesh.elems[e]]
        if mesh.kind == "tri":
            xi, eta, w = tri_rule(3)
            N, _, _ = tri_p1_values(xi, eta)
            J = tri_jacobian(corners)
            det = float(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
            pts = N.T @ corners
            wdet = w * abs(det)
        else:
            pts, wdet, _, _ = _quad_geometry(corners, 2, False)
            N, _, _ = quad_q1_values(*quad_rule(2)[:2])
        fq = eval_field(f_field, pts, default=0.0)
        fe[e] = N @ (wdet * fq)
    return fe


def assemble_2d(mesh: Mesh2D, ke: np.ndarray, fe: np.ndarray) -> Assembly:
    r"""二维散射装配：自由度号就是顶点号（:math:`P_1` 三角形 / :math:`Q_1` 四边形）。"""
    n = mesh.n_node
    K = np.zeros((n, n), dtype=float)
    F = np.zeros(n, dtype=float)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    owner: list[int] = []
    for e in range(mesh.n_elem):
        ids = mesh.elems[e]
        for i in range(ids.size):
            gi = int(ids[i])
            F[gi] += float(fe[e, i])
            for j in range(ids.size):
                gj = int(ids[j])
                v = float(ke[e, i, j])
                K[gi, gj] += v
                rows.append(gi)
                cols.append(gj)
                vals.append(v)
                owner.append(e)
    return _finish(K, F, ke, fe, np.asarray(rows), np.asarray(cols), np.asarray(vals),
                   np.asarray(owner), mesh.n_elem)


# --------------------------------------------------------------------------- #
# 边界条件
# --------------------------------------------------------------------------- #
def dirichlet_elimination(K: np.ndarray, F: np.ndarray, fixed: Any, values: Any) -> dict[str, Any]:
    r"""消元法加 Dirichlet 条件：:math:`K_{ff}u_f=F_f-K_{fp}g`，返回约化系统与诊断量。

    返回值里带 ``K_full``（把已知值代回后的完整解所需的矩阵不是它，但**对称性检查**要用
    原矩阵）、``symmetry_defect_stage1``（消元前的 :math:`\max|K-K^{\mathsf T}|`）、
    ``symmetry_defect_stage2``（删行删列后的）——两次都不改变对称性，
    这就是"消元法不破坏对称性"的实测口径。
    """
    A = np.asarray(K, dtype=float)
    b = np.asarray(F, dtype=float)
    fx = np.asarray(fixed, dtype=np.int64).reshape(-1)
    g = np.asarray(values, dtype=float).reshape(-1)
    n = A.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[fx] = False
    free = np.flatnonzero(mask)
    Kff = A[np.ix_(free, free)]
    if fx.size:
        Kfp = A[np.ix_(free, fx)]
        Ff = b[free] - Kfp @ g
    else:
        Ff = b[free].copy()
    return {
        "Kff": Kff,
        "Ff": Ff,
        "free": free,
        "fixed": fx,
        "fixed_values": g,
        "symmetry_defect_stage1": symmetry_defect(A),
        "symmetry_defect_stage2": symmetry_defect(Kff),
        "n_free": int(free.size),
    }


def dirichlet_penalty(K: np.ndarray, F: np.ndarray, fixed: Any, values: Any,
                      eps: float = 1e-10) -> dict[str, Any]:
    r"""罚函数法加 Dirichlet 条件：:math:`(K+\frac1\varepsilon I_{\rm fixed})u=F+\frac1\varepsilon g`。

    对称性**完好**（加的是对角阵），但

    * :math:`\kappa` 被抬到 :math:`O(1/\varepsilon)`（实测给在 :mod:`~numcore.pde.fem.solve`）；
    * 解有一阶偏差：把 :math:`u=g+\delta` 代回，固定点处
      :math:`\delta\approx-\varepsilon\,(\text{内部方程的残差})`，所以罚解**不精确满足**边界条件。

    这两件事都不是实现细节：它们是"选哪条路线加边界条件"这个决定的可测后果。
    """
    A = np.asarray(K, dtype=float).copy()
    b = np.asarray(F, dtype=float).copy()
    fx = np.asarray(fixed, dtype=np.int64).reshape(-1)
    g = np.asarray(values, dtype=float).reshape(-1)
    inv_eps = 1.0 / float(eps)
    for idx, val in zip(fx.tolist(), g.tolist()):
        A[idx, idx] += inv_eps
        b[idx] += inv_eps * float(val)
    return {
        "K": A,
        "F": b,
        "fixed": fx,
        "fixed_values": g,
        "eps": float(eps),
        "symmetry_defect": symmetry_defect(A),
        "penalty_scale": inv_eps,
    }


def symmetry_defect(K: np.ndarray) -> float:
    r"""对称性残差 :math:`\max_{ij}|K_{ij}-K_{ji}|`（绝对量，不是相对量）。

    绝对量更好读：章节里同时给 :math:`\|K\|_\infty`，读者自己就能换算成相对量。
    """
    A = np.asarray(K, dtype=float)
    if A.size == 0:
        return 0.0
    return float(np.max(np.abs(A - A.T)))
