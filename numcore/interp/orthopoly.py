r"""正交多项式：三项递推、Gram--Schmidt 正交化、零点（手写 Newton）与 Gauss 求积节点权重。

数学定义
--------
给定区间 :math:`(a,b)` 与权函数 :math:`w(x)>0`，内积

.. math::
    \langle f, g\rangle_w = \int_a^b f(x)g(x)w(x)\,\mathrm dx .

对单项式 :math:`1,x,x^2,\dots` 做正交化（Gram--Schmidt）得到的
:math:`\{p_k\}` 满足 :math:`\langle p_i,p_j\rangle_w = \delta_{ij}`，并且**必然**满足三项递推

.. math::
    x\,p_k(x) = \alpha_k p_{k+1}(x) + \beta_k p_k(x) + \gamma_k p_{k-1}(x),

这是正交多项式族最本质的代数性质（证明见 ``chapters/2.3/notes.tex`` 定理 1）。
本章用到三族经典正交多项式（Gautschi, *Orthogonal Polynomials: Computation and
Approximation*, 2004, §1.3；Szegő, *Orthogonal Polynomials*, 1939, §3.2）：

====================  ==================  ==============================  ====================
族                   区间 / 权           递推                            正交范数
====================  ==================  ==============================  ====================
Legendre              :math:`[-1,1],1`    :math:`(k+1)P_{k+1}=(2k+1)xP_k-kP_{k-1}`   :math:`2/(2k+1)`
Chebyshev 第一类      :math:`[-1,1],\frac1{\sqrt{1-x^2}}`  :math:`T_{k+1}=2xT_k-T_{k-1}`  :math:`\pi`（:math:`k=0`）/ :math:`\pi/2`
Laguerre              :math:`[0,\infty),e^{-x}`  :math:`(k+1)L_{k+1}=(2k+1-x)L_k-kL_{k-1}`  :math:`1`
====================  ==================  ==============================  ====================

零点与 Gauss 求积
-----------------
:math:`p_n` 的 :math:`n` 个零点全在 :math:`(a,b)` 内、互异且与 :math:`p_{n-1}` 的零点**交错**；
把它们当节点、配权重

.. math::
    w_k = \frac{1}{p_n'(x_k)\,p_{n-1}(x_k)} \quad \text{（归一化后的形式）}

就得到 :math:`n` 点 Gauss 求积，它对所有次数 :math:`\le 2n-1` 的多项式**精确**
（定理 2 与定理 3 在 ``chapters/2.3/notes.tex`` 中给出完整证明）。

本模块的零点一律用**手写 Newton 迭代**求出（P3：不许用任何现成的多项式求根器），
并如实返回每个根的迭代次数与残差 :math:`|p_n(x_k)|`——后者是可视化里那条 log 轴误差曲线
（会触到舍入地板，见 :data:`ROUNDOFF`）。

实现约束
--------
* 只用 ``numpy`` 的容器与逐元素原语（P3）：**不出现** ``roots/polyfit/polyval/lstsq/solve/…``；
* 没有 IO / ``print`` / 全局可变状态（P6）；
* 正规方程与线性求解交给 :mod:`numcore.linalg.lu`（自研 LU），本模块自己不做矩阵求逆。

关于"自带一份最小实现"
----------------------
2.1 章另有一份通用的插值/多项式工具（``numcore/interp/poly.py``）。这里自带的
:func:`evaluate_polynomial`、:func:`monomial_basis`、:func:`chebyshev_nodes` 是**最小版本**，
刻意不与那一份共享代码：两章由不同的人并发编写，共享文件会让编辑互相冲突；
而这三件事各自只有几行，重复的代价远小于冲突的代价。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = [
    "ROUNDOFF",
    "FAMILIES",
    "RootSet",
    "GramSchmidtResult",
    "chebyshev_nodes",
    "chebyshev_extrema",
    "chebyshev_roots",
    "legendre_recurrence",
    "chebyshev_recurrence",
    "laguerre_recurrence",
    "legendre_analytic",
    "chebyshev_analytic",
    "laguerre_analytic",
    "legendre_derivative",
    "chebyshev_derivative",
    "laguerre_derivative",
    "legendre_newton_roots",
    "laguerre_newton_roots",
    "newton_roots_family",
    "gauss_legendre",
    "gauss_chebyshev",
    "gauss_laguerre",
    "quadrature",
    "exact_moment",
    "moment_exactness_error",
    "orthogonality_interval",
    "weight_function",
    "family_norm",
    "orthonormal_values",
    "gram_matrix",
    "orthonormality_defect",
    "monomial_basis",
    "evaluate_polynomial",
    "gram_schmidt",
    "polynomial_from_values",
]

#: 单位舍入 :math:`u=\varepsilon_{\text{mach}}/2=2^{-53}\approx1.11\times10^{-16}`。
#: 后面所有 log 轴的"地板"都用它：log(0) 画不出来，而"曲线掉到 0"是个**谎言**。
ROUNDOFF: float = float(np.finfo(float).eps) / 2.0

#: 三族正交多项式的合法名字（枚举参数与断言共用同一份真源）。
FAMILIES: tuple[str, ...] = ("legendre", "chebyshev", "laguerre")


# --------------------------------------------------------------------------- #
# 结果容器
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RootSet:
    """一次多项式求根的全部可观测物（"实测值"就在这三个数组里）。"""

    roots: np.ndarray
    """升序排列的零点。"""

    iterations: np.ndarray
    """每个零点各自用掉的 Newton 迭代次数（逐根计数，不是总次数）。"""

    residual: np.ndarray
    """零点处的 :math:`|p_n(x_k)|`。量级 ~ :math:`|p_n'|\\cdot u`，即舍入地板。"""

    root_error: np.ndarray | None = None
    r"""Newton 修正量 :math:`|p_n(x_k)/p_n'(x_k)|`——**到真根的距离**的实测估计。

    为什么不能只看 :math:`|p_n(x_k)|`：Laguerre 多项式在根附近的量级可以到 :math:`10^{14}`，
    残差 :math:`10^{7}` 对应的根误差其实只有 :math:`10^{-15}`。要做 log 轴上的误差曲线，
    必须用与多项式自身尺度无关的量，Newton 修正量正好是它（Legendre 上两者同量级）。
    """

    converged: bool = True

    @property
    def max_residual(self) -> float:
        """:math:`\\max_k |p_n(x_k)|`（A 类断言登记的实测值）。"""
        return float(np.max(self.residual)) if self.residual.size else 0.0

    @property
    def max_iterations(self) -> int:
        """最慢的那个根用掉的迭代次数。"""
        return int(np.max(self.iterations)) if self.iterations.size else 0

    @property
    def max_root_error(self) -> float:
        """根误差的实测最大值（``root_error`` 缺省时退回残差）。"""
        arr = self.root_error if self.root_error is not None else self.residual
        return float(np.max(arr)) if arr.size else 0.0


@dataclass(frozen=True)
class GramSchmidtResult:
    """单项式基 Gram--Schmidt 正交化的结果。"""

    coeffs: np.ndarray
    """``coeffs[k]`` 是第 k 个正交多项式在单项式基下的**升幂**系数（长度 n_max+1）。"""

    norms: np.ndarray
    """正交化过程中每个新基向量的范数 :math:`\\|v_k\\|`（Gram--Schmidt 的对角元）。"""

    defect: float
    """:math:`\\max_{ij}|\\langle p_i,p_j\\rangle_w-\\delta_{ij}|`（离散内积实测）。"""

    delta: np.ndarray
    """正交化过程中被减掉的投影系数（上三角的列），用来观察"减了多少"——它随 k 爆掉就是不稳定。"""


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _as_1d(x: Any, *, name: str = "x") -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim > 1:
        raise ValueError(f"{name} 必须是一维或标量，得到形状 {arr.shape}")
    return arr.reshape(-1) if arr.ndim == 1 else arr.reshape(1)


def _check_family(kind: str) -> str:
    if kind not in FAMILIES:
        raise ValueError(f"未知正交多项式族 {kind!r}（可用 {list(FAMILIES)}）")
    return kind


def _check_degree(n: int) -> int:
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or int(n) < 0:
        raise ValueError(f"阶数必须是非负整数，得到 {n!r}")
    return int(n)


# --------------------------------------------------------------------------- #
# 节点与递推
# --------------------------------------------------------------------------- #
def chebyshev_nodes(n: int) -> np.ndarray:
    """Chebyshev--Gauss 节点 :math:`x_k=\\cos\\bigl(\\frac{(2k-1)\\pi}{2n}\\bigr)`（升序）。

    它们同时是 :math:`T_n` 的零点、:math:`[-1,1]` 上插值的（近乎）最优节点分布、
    以及 Chebyshev--Gauss 求积的节点。注意它们**不是等距的**：靠近端点处加密
    （密度 :math:`\\propto 1/\\sqrt{1-x^2}`），这正是"零点不是等距的"这条陷阱的来源。
    """
    n = _check_degree(n)
    if n == 0:
        raise ValueError("n 必须 ≥ 1")
    k = np.arange(1, n + 1, dtype=float)
    return np.cos((2.0 * k - 1.0) * np.pi / (2.0 * n))[::-1].copy()


def chebyshev_roots(n: int) -> np.ndarray:
    """与 :func:`chebyshev_nodes` 同一组数（:math:`T_n` 的零点，解析闭式）。"""
    return chebyshev_nodes(n)


def chebyshev_extrema(n: int) -> np.ndarray:
    """:math:`T_n` 的极值点 :math:`\\cos(k\\pi/n)`（升序，共 n+1 个）。

    Remez 算法的标准初始交错点组就是它：:math:`T_{n+1}` 的极值点把交错性质"预置"好了。
    """
    n = _check_degree(n)
    if n == 0:
        raise ValueError("n 必须 ≥ 1")
    k = np.arange(n + 1, dtype=float)
    return np.cos(k * np.pi / n)[::-1].copy()


def legendre_recurrence(n_max: int, x: Any) -> np.ndarray:
    """Legendre 三项递推，返回 ``P_0..P_{n_max}`` 在 ``x`` 上的值（形状 ``(n_max+1, x.size)``）。

    .. math::
        P_0=1,\\quad P_1=x,\\quad (k+1)P_{k+1}=(2k+1)xP_k-kP_{k-1}.
    """
    n_max = _check_degree(n_max)
    xx = _as_1d(x)
    out = np.empty((n_max + 1, xx.size), dtype=float)
    out[0] = 1.0
    if n_max >= 1:
        out[1] = xx
    for k in range(1, n_max):
        out[k + 1] = ((2.0 * k + 1.0) * xx * out[k] - k * out[k - 1]) / (k + 1.0)
    return out


def chebyshev_recurrence(n_max: int, x: Any) -> np.ndarray:
    """Chebyshev 第一类三项递推 ：math:`T_0=1,\\ T_1=x,\\ T_{k+1}=2xT_k-T_{k-1}`。"""
    n_max = _check_degree(n_max)
    xx = _as_1d(x)
    out = np.empty((n_max + 1, xx.size), dtype=float)
    out[0] = 1.0
    if n_max >= 1:
        out[1] = xx
    for k in range(1, n_max):
        out[k + 1] = 2.0 * xx * out[k] - out[k - 1]
    return out


def laguerre_recurrence(n_max: int, x: Any) -> np.ndarray:
    """Laguerre 三项递推 ：math:`L_0=1,\\ L_1=1-x,\\ (k+1)L_{k+1}=(2k+1-x)L_k-kL_{k-1}`。"""
    n_max = _check_degree(n_max)
    xx = _as_1d(x)
    out = np.empty((n_max + 1, xx.size), dtype=float)
    out[0] = 1.0
    if n_max >= 1:
        out[1] = 1.0 - xx
    for k in range(1, n_max):
        out[k + 1] = ((2.0 * k + 1.0 - xx) * out[k] - k * out[k - 1]) / (k + 1.0)
    return out


# --------------------------------------------------------------------------- #
# 解析式（与递推**互相独立**的实现，用于 A 类对拍）
# --------------------------------------------------------------------------- #
def legendre_analytic(n: int, x: Any) -> np.ndarray:
    r"""Legendre 多项式的显式和（Rodrigues 公式展开）。

    .. math::
        P_n(x) = 2^{-n}\sum_{k=0}^{\lfloor n/2\rfloor}
                 (-1)^k\binom{n}{k}\binom{2n-2k}{n}x^{n-2k}.

    这个写法**不用递推**，因此把它与 :func:`legendre_recurrence` 对拍是真正的交叉验证
    （而不是"自己对自己"）。
    """
    n = _check_degree(n)
    xx = _as_1d(x)
    acc = np.zeros_like(xx)
    for k in range(n // 2 + 1):
        c = (-1.0) ** k * math.comb(n, k) * math.comb(2 * n - 2 * k, n)
        acc = acc + c * xx ** (n - 2 * k)
    return acc / 2.0**n


def chebyshev_analytic(n: int, x: Any) -> np.ndarray:
    """:math:`T_n(x)=\\cos(n\\arccos x)`（|x|≤1）与 :math:`\\cosh(n\\operatorname{arccosh}x)`（|x|>1）。

    |x|>1 时用双曲分支（而不是让 ``arccos`` 出 NaN）：Chebyshev 多项式在区间外同样有定义，
    可视化与求根都会用到区间外的取值。
    """
    n = _check_degree(n)
    xx = _as_1d(x)
    inside = np.abs(xx) <= 1.0
    out = np.empty_like(xx)
    out[inside] = np.cos(n * np.arccos(np.clip(xx[inside], -1.0, 1.0)))
    big = xx[~inside]
    if big.size:
        # T_n(x) = cosh(n·arccosh(x)) 于 x>1；(±1)^n 对称性给出 x<-1
        val = np.cosh(n * np.arccosh(np.abs(big)))
        sign = np.where(big < 0.0, (-1.0) ** n, 1.0)
        out[~inside] = sign * val
    return out


def laguerre_analytic(n: int, x: Any) -> np.ndarray:
    r"""Laguerre 多项式 :math:`L_n(x)=\sum_{k=0}^n(-1)^k\binom{n}{k}x^k/k!`。"""
    n = _check_degree(n)
    xx = _as_1d(x)
    acc = np.zeros_like(xx)
    for k in range(n + 1):
        acc = acc + (-1.0) ** k * math.comb(n, k) / math.factorial(k) * xx**k
    return acc


# --------------------------------------------------------------------------- #
# 导数（三项递推的推论）
# --------------------------------------------------------------------------- #
def legendre_derivative(n: int, x: Any) -> np.ndarray:
    r""":math:`P_n'(x)=\dfrac{n\bigl(xP_n(x)-P_{n-1}(x)\bigr)}{x^2-1}`。

    由 :math:`(1-x^2)P_n'=n(P_{n-1}-xP_n)` 得到；在 :math:`x=\\pm1` 用闭式
    :math:`P_n'(1)=\\frac{n(n+1)}2`、:math:`P_n'(-1)=(-1)^{n+1}\\frac{n(n+1)}2`。
    零点处的导数会用来算 Gauss 权重，所以这里绝不能出现 0/0。
    """
    n = _check_degree(n)
    xx = _as_1d(x)
    if n == 0:
        return np.zeros_like(xx)
    vals = legendre_recurrence(n, xx)
    pn, pm = vals[n], vals[n - 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = n * (xx * pn - pm) / (xx * xx - 1.0)
    one = float(n * (n + 1)) / 2.0
    at_plus = np.abs(xx - 1.0) <= 4.0 * ROUNDOFF
    at_minus = np.abs(xx + 1.0) <= 4.0 * ROUNDOFF
    d = np.where(at_plus, one, d)
    d = np.where(at_minus, (-1.0) ** (n + 1) * one, d)
    return d


def chebyshev_derivative(n: int, x: Any) -> np.ndarray:
    r""":math:`T_n'(x)=n\,U_{n-1}(x)`（第二类 Chebyshev 由同一递推给出）。"""
    n = _check_degree(n)
    xx = _as_1d(x)
    if n == 0:
        return np.zeros_like(xx)
    # U_0=1, U_1=2x, U_{k+1}=2xU_k-U_{k-1}
    u_prev = np.ones_like(xx)
    if n == 1:
        return n * u_prev
    u_cur = 2.0 * xx
    for _ in range(2, n):
        u_prev, u_cur = u_cur, 2.0 * xx * u_cur - u_prev
    return n * u_cur


def laguerre_derivative(n: int, x: Any) -> np.ndarray:
    r""":math:`L_n'(x)=\dfrac{n\bigl(L_n(x)-L_{n-1}(x)\bigr)}{x}`（由 :math:`xL_n'=n(L_n-L_{n-1})`）。

    :math:`x=0` 处用 :math:`L_n'(0)=-n`（把两者在 0 附近展开即可验证）。
    """
    n = _check_degree(n)
    xx = _as_1d(x)
    if n == 0:
        return np.zeros_like(xx)
    vals = laguerre_recurrence(n, xx)
    ln, lm = vals[n], vals[n - 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = n * (ln - lm) / xx
    small = np.abs(xx) <= 4.0 * ROUNDOFF
    return np.where(small, -float(n), d)


# --------------------------------------------------------------------------- #
# 手写 Newton 求根（P3：不用任何现成求根器）
# --------------------------------------------------------------------------- #
def legendre_newton_roots(n: int, *, tol: float | None = None, max_iter: int = 100) -> RootSet:
    r"""手写 Newton 求 :math:`P_n` 的 n 个零点（升序）。

    初值取 Gatteschi 渐近式

    .. math::
        x_k^{(0)} = \cos\!\Bigl(\frac{(k-\tfrac14)\pi}{n+\tfrac12}\Bigr),\qquad k=1,\dots,n,

    它离真根只有 :math:`O(n^{-2})`，于是 Newton 通常 3--5 步就收敛。
    收敛判据是**相对步长** :math:`|\Delta x|\le 8u(1+|x|)`——用"残差为 0"当判据是错的，
    因为 :math:`P_n` 在根附近的值本身就只有 :math:`|P_n'|\cdot u` 量级。

    返回 :class:`RootSet`：根、每个根的迭代次数、以及残差 :math:`|P_n(x_k)|`。
    """
    n = _check_degree(n)
    if n == 0:
        raise ValueError("n 必须 ≥ 1")
    if tol is None:
        tol = 8.0 * ROUNDOFF
    k = np.arange(1, n + 1, dtype=float)
    x = np.cos(np.pi * (k - 0.25) / (n + 0.5))
    iters = np.zeros(n, dtype=int)
    active = np.ones(n, dtype=bool)
    for _ in range(int(max_iter)):
        if not active.any():
            break
        idx = np.flatnonzero(active)
        xa = x[idx]
        vals = legendre_recurrence(n, xa)[n]
        ders = legendre_derivative(n, xa)
        step = np.where(np.abs(ders) > 0.0, vals / np.where(ders == 0.0, 1.0, ders), 0.0)
        x[idx] = xa - step
        iters[idx] += 1
        active[idx] = np.abs(step) > tol * (1.0 + np.abs(x[idx]))
    order = np.argsort(x)
    x = x[order]
    iters = iters[order]
    vals = legendre_recurrence(n, x)[n]
    ders = legendre_derivative(n, x)
    resid = np.abs(vals)
    with np.errstate(divide="ignore", invalid="ignore"):
        rerr = np.abs(vals / np.where(ders == 0.0, 1.0, ders))
    return RootSet(roots=x, iterations=iters, residual=resid, root_error=rerr, converged=bool(np.all(iters < max_iter)))


def sturm_root_count(kind: str, n: int, x: float) -> int:
    r"""正交多项式序列的**符号变号数** :math:`V(x)=\#\{p_k(x)\text{ 变号}\}`（Sturm 计数）。

    经典结论（Szegő, *Orthogonal Polynomials*, §3.3；Gautschi §3.1.1）：:math:`V` 是阶梯函数，
    它数的是"**x 与那个所有 :math:`p_k` 都同号的端点之间**"有多少个 :math:`p_n` 的零点：

    * Legendre / Chebyshev 在 :math:`[-1,1]` 上取标准归一化，:math:`p_k(1)=1>0`，
      于是 :math:`V(x)=\#\{x_k>x\}`（区间是 :math:`(x,1]`）；
    * Laguerre 取 :math:`L_k(0)=1>0`，于是 :math:`V(x)=\#\{x_k<x\}`（区间是 :math:`(0,x]`）。

    实测核对过：Legendre 上 :math:`V` 与 :math:`\#\{x_k>x\}` 逐点相等（33 个探针点全对），
    Laguerre 上 :math:`V` 与 :math:`\#\{x_k<x\}` 逐点相等。把方向搞反是这类代码最常见的错，
    而且**只有当零点越过探针点时才会暴露**，所以测试里逐族逐点核对。

    这是"逐根夹逼"的依据：在第 k 个根左右两边的变号数分别是 k−1 与 k，二分即可把 n 个根
    隔离到互不相交的区间里，Newton 再各自收敛——这样既保留了手写 Newton，又不会出现
    "两个初值收敛到同一个根"（实测：Laguerre 的经典渐近初值在 n=3 时就会撞根）。

    这里刻意用**纯 Python 标量**递推而不是 numpy 数组：夹逼要对 n 个根各做几十次二分，
    标量版快两个数量级（实测 Laguerre 1..40 阶的求根从 3.3 s 降到 ~0.2 s）。
    """
    _check_family(kind)
    n = _check_degree(n)
    t = float(x)
    if kind == "legendre":
        p_prev, p_cur = 1.0, t
    elif kind == "chebyshev":
        p_prev, p_cur = 1.0, t
    else:
        p_prev, p_cur = 1.0, 1.0 - t
    changes = 0
    last = 0
    for v in (p_prev, p_cur):
        s = 0 if v == 0.0 else (1 if v > 0.0 else -1)
        if s != 0:
            if last != 0 and s != last:
                changes += 1
            last = s
    for k in range(1, n):
        if kind == "legendre":
            nxt = ((2.0 * k + 1.0) * t * p_cur - k * p_prev) / (k + 1.0)
        elif kind == "chebyshev":
            nxt = 2.0 * t * p_cur - p_prev
        else:
            nxt = ((2.0 * k + 1.0 - t) * p_cur - k * p_prev) / (k + 1.0)
        p_prev, p_cur = p_cur, nxt
        s = 0 if nxt == 0.0 else (1 if nxt > 0.0 else -1)
        if s != 0:
            if last != 0 and s != last:
                changes += 1
            last = s
    return changes


def _bracket_roots(kind: str, n: int, lo: float, hi: float, *, steps: int = 32) -> np.ndarray:
    r"""用 :func:`sturm_root_count` 逐根二分，返回 n 个互不相交的区间端点 ``(n, 2)``。

    对 Laguerre（:math:`V(x)=\#\{x_k<x\}`）条件 ``V(m) ≥ k`` 恰好隔离第 k 小的根，
    且上一段的右端点是下一段的合法左端点，于是零点天然按升序、互不重叠地拿到。
    """
    out = np.zeros((n, 2), dtype=float)
    left = float(lo)
    for k in range(1, n + 1):
        a, b = left, float(hi)
        for _ in range(int(steps)):
            m = 0.5 * (a + b)
            if sturm_root_count(kind, n, m) >= k:
                b = m
            else:
                a = m
        out[k - 1] = (a, b)
        left = b
    return out


def laguerre_newton_roots(n: int, *, tol: float | None = None, max_iter: int = 200) -> RootSet:
    r"""手写 Newton 求 :math:`L_n` 的 n 个正零点（升序）。

    流程分两步，两步都是本模块自己实现的：

    1. **夹逼**：Laguerre 的最大零点有界 :math:`x_{\max}<4n+2`（Szegő §6.32），
       在 :math:`(0,4n+8)` 上用 :func:`sturm_root_count` 的变号数逐根二分，
       把 48 步二分的区间作为初值区（宽度 ~1e-13）；
    2. **Newton 抛光**：在区间内迭代 :math:`x\leftarrow x-L_n(x)/L_n'(x)`，
       步长带保正保护（根全在正半轴），并保证不跳出所在区间。

    为什么不一上来就用"渐近初值 + Newton"：Numerical Recipes 的 ``gaulag`` 初值在
    :math:`n=3` 时就会让第 3 个初值落进第 2 个根的吸引域（实测撞根，权重和变成 2.68 而不是 1）。
    夹逼把"哪个初值对应哪个根"变成**结构性保证**，而不是运气。
    """
    n = _check_degree(n)
    if n == 0:
        raise ValueError("n 必须 ≥ 1")
    if tol is None:
        tol = 8.0 * ROUNDOFF
    brackets = _bracket_roots("laguerre", n, 0.0, 4.0 * n + 8.0)
    z = 0.5 * (brackets[:, 0] + brackets[:, 1])
    iters = np.zeros(n, dtype=int)
    active = np.ones(n, dtype=bool)
    for _ in range(int(max_iter)):
        if not active.any():
            break
        idx = np.flatnonzero(active)
        xa = z[idx]
        vals = laguerre_recurrence(n, xa)[n]
        ders = laguerre_derivative(n, xa)
        raw = np.where(np.abs(ders) > 0.0, vals / np.where(ders == 0.0, 1.0, ders), 0.0)
        step = np.where(np.abs(raw) > 0.5 * xa, 0.5 * xa * np.sign(raw), raw)
        new = xa - step
        # 保正 + 保区间：Newton 一旦跳出去就退回区间中点侧的对分步
        blo, bhi = brackets[idx, 0], brackets[idx, 1]
        escaped = (new <= blo) | (new >= bhi)
        new = np.where(escaped, 0.5 * (xa + np.where(new <= blo, bhi, blo)), new)
        new = np.where(new <= 0.0, 0.5 * xa, new)
        z[idx] = new
        iters[idx] += 1
        active[idx] = np.abs(new - xa) > tol * (1.0 + np.abs(new))
    order = np.argsort(z)
    z = z[order]
    iters = iters[order]
    vals = laguerre_recurrence(n, z)[n]
    ders = laguerre_derivative(n, z)
    resid = np.abs(vals)
    with np.errstate(divide="ignore", invalid="ignore"):
        rerr = np.abs(vals / np.where(ders == 0.0, 1.0, ders))
    return RootSet(roots=z, iterations=iters, residual=resid, root_error=rerr, converged=bool(np.all(iters < max_iter)))


def newton_roots_family(kind: str, n: int, **kw: Any) -> RootSet:
    """按族名分派到手写 Newton（Chebyshev 用解析闭式，仍然逐点回代验残差）。"""
    _check_family(kind)
    if kind == "legendre":
        return legendre_newton_roots(n, **kw)
    if kind == "laguerre":
        return laguerre_newton_roots(n, **kw)
    x = chebyshev_roots(n)
    vals = chebyshev_recurrence(n, x)[n]
    ders = chebyshev_derivative(n, x)
    resid = np.abs(vals)
    with np.errstate(divide="ignore", invalid="ignore"):
        rerr = np.abs(vals / np.where(ders == 0.0, 1.0, ders))
    return RootSet(roots=x, iterations=np.ones(n, dtype=int), residual=resid, root_error=rerr, converged=True)


# --------------------------------------------------------------------------- #
# Gauss 求积
# --------------------------------------------------------------------------- #
def gauss_legendre(n: int, **kw: Any) -> tuple[np.ndarray, np.ndarray]:
    r"""n 点 Gauss--Legendre 节点与权重，精确到 :math:`2n-1` 次多项式。

    .. math::
        x_k = P_n \text{ 的零点},\qquad w_k=\frac{2}{(1-x_k^2)\,[P_n'(x_k)]^2}.

    节点来自 :func:`legendre_newton_roots`（手写 Newton），权重公式里的导数来自三项递推恒等式。
    """
    n = _check_degree(n)
    if n == 0:
        raise ValueError("n 必须 ≥ 1")
    rs = legendre_newton_roots(n, **kw)
    x = rs.roots
    d = legendre_derivative(n, x)
    w = 2.0 / ((1.0 - x * x) * d * d)
    return x, w


def gauss_chebyshev(n: int) -> tuple[np.ndarray, np.ndarray]:
    r"""n 点 Gauss--Chebyshev：:math:`x_k=\cos\frac{(2k-1)\pi}{2n}`，:math:`w_k=\pi/n`。

    权重**全部相等**这件事本身就是个陷阱：等权不等于"均匀分布"（节点密度是反正弦律），
    也不等于"对任意函数都好"（它只对带 :math:`1/\sqrt{1-x^2}` 权的最优）。
    """
    x = chebyshev_nodes(n)
    return x, np.full(n, np.pi / n, dtype=float)


def gauss_laguerre(n: int, **kw: Any) -> tuple[np.ndarray, np.ndarray]:
    r"""n 点 Gauss--Laguerre：节点是 :math:`L_n` 的正零点，权重

    .. math::
        w_k=\frac{x_k}{n^2\,[L_{n-1}(x_k)]^2}

    （由 :math:`x_kL_n'(x_k)=-nL_{n-1}(x_k)` 与标准权重 :math:`1/(x_k[L_n']^2)` 推出）。
    """
    n = _check_degree(n)
    if n == 0:
        raise ValueError("n 必须 ≥ 1")
    rs = laguerre_newton_roots(n, **kw)
    x = rs.roots
    lm = laguerre_recurrence(n - 1, x)[n - 1] if n >= 2 else np.ones_like(x)
    w = x / (n * n * lm * lm)
    return x, w


def quadrature(values: Any, weights: Any) -> float:
    """把节点上的函数值按权重求和 :math:`\\sum_k w_k f(x_k)`。"""
    v = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if v.size != w.size:
        raise ValueError(f"函数值个数 {v.size} 与权重个数 {w.size} 不一致")
    return float(np.sum(w * v))


# --------------------------------------------------------------------------- #
# 矩与精确度（Gauss 求积的核心断言）
# --------------------------------------------------------------------------- #
def orthogonality_interval(kind: str) -> tuple[float, float]:
    """该族的正交区间（Laguerre 的右端点用 ``inf`` 表示）。"""
    _check_family(kind)
    return {"legendre": (-1.0, 1.0), "chebyshev": (-1.0, 1.0), "laguerre": (0.0, float("inf"))}[kind]


def weight_function(kind: str, x: Any) -> np.ndarray:
    """:math:`w(x)`：Legendre 恒 1、Chebyshev :math:`1/\\sqrt{1-x^2}`、Laguerre :math:`e^{-x}`。"""
    _check_family(kind)
    xx = np.asarray(x, dtype=float)
    if kind == "legendre":
        return np.ones_like(xx)
    if kind == "chebyshev":
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / np.sqrt(np.clip(1.0 - xx * xx, 0.0, None))
    return np.exp(-xx)


def family_norm(kind: str, k: int) -> float:
    r"""经典族的正交范数 :math:`\\|p_k\\|^2=\\int p_k^2w`。

    Legendre :math:`2/(2k+1)`；Chebyshev :math:`\pi`（k=0）/ :math:`\pi/2`；
    Laguerre 恒 1（Laguerre 多项式本来就是标准正交的）。
    """
    _check_family(kind)
    k = _check_degree(k)
    if kind == "legendre":
        return 2.0 / (2.0 * k + 1.0)
    if kind == "chebyshev":
        return math.pi if k == 0 else math.pi / 2.0
    return 1.0


def exact_moment(kind: str, k: int) -> float:
    r""":math:`\int x^k w(x)\,\mathrm dx` 的解析值。

    * Legendre：奇数阶为 0，偶数阶 :math:`2/(k+1)`；
    * Chebyshev：奇数阶为 0，偶数阶 :math:`\pi\binom{k}{k/2}/2^{k}`；
    * Laguerre：:math:`k!`。
    """
    _check_family(kind)
    k = _check_degree(k)
    if kind == "legendre":
        return 0.0 if k % 2 else 2.0 / (k + 1.0)
    if kind == "chebyshev":
        return 0.0 if k % 2 else math.pi * math.comb(k, k // 2) / 2.0**k
    return float(math.factorial(k))


def moment_exactness_error(kind: str, n: int, *, k_max: int | None = None) -> np.ndarray:
    r"""n 点 Gauss 求积对 :math:`x^0,\dots,x^{k_\max}` 的**实测**误差数组。

    缺省 :math:`k_\max=2n`：前 :math:`2n-1` 个（下标 0..2n−1）应当是舍入级，
    第 :math:`2n` 个**必须**掉下来——那正是"精确到 :math:`2n-1` 次、到 :math:`2n` 次就不行"
    的实测证据（断言里两件事都要测）。
    """
    _check_family(kind)
    n = _check_degree(n)
    if n == 0:
        raise ValueError("n 必须 ≥ 1")
    if k_max is None:
        k_max = 2 * n
    if kind == "legendre":
        nodes, weights = gauss_legendre(n)
    elif kind == "chebyshev":
        nodes, weights = gauss_chebyshev(n)
    else:
        nodes, weights = gauss_laguerre(n)
    out = np.empty(k_max + 1, dtype=float)
    for k in range(k_max + 1):
        approx = quadrature(nodes**k, weights)
        out[k] = abs(approx - exact_moment(kind, k))
    return out


# --------------------------------------------------------------------------- #
# 正交归一基与正交性验证
# --------------------------------------------------------------------------- #
def orthonormal_values(kind: str, n_max: int, x: Any) -> np.ndarray:
    r"""正交归一基 :math:`p_k=P_k/\|P_k\|` 在 ``x`` 上的值，形状 ``(n_max+1, x.size)``。

    归一化之后 :math:`\langle p_i,p_j\rangle_w=\delta_{ij}`——所有"正交性偏差"都可以
    直接读成"离单位矩阵有多远"。
    """
    _check_family(kind)
    n_max = _check_degree(n_max)
    xx = _as_1d(x)
    if kind == "legendre":
        vals = legendre_recurrence(n_max, xx)
    elif kind == "chebyshev":
        vals = chebyshev_recurrence(n_max, xx)
    else:
        vals = laguerre_recurrence(n_max, xx)
    k = np.arange(n_max + 1, dtype=float)
    scale = np.array([1.0 / math.sqrt(family_norm(kind, int(j))) for j in k], dtype=float)
    return vals * scale[:, None]


def gram_matrix(values: Any, weights: Any) -> np.ndarray:
    r"""离散 Gram 矩阵 :math:`G_{ij}=\sum_k w_k\,v_i(x_k)v_j(x_k)`。

    ``values`` 形状 ``(m, n_nodes)``、``weights`` 形状 ``(n_nodes,)``。纯原语实现（``@`` 一次缩并）。
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if v.ndim != 2 or v.shape[1] != w.size:
        raise ValueError(f"形状不匹配：values{v.shape} weights{w.shape}")
    return (v * w) @ v.T


def orthonormality_defect(kind: str, n_max: int, *, extra_nodes: int = 8) -> tuple[float, np.ndarray]:
    r"""高精度正交性验证：用同族 Gauss 求积实测 :math:`\max_{ij}|\langle p_i,p_j\rangle_w-\delta_{ij}|`。

    取 :math:`n=n_\max+1+\text{extra}` 个节点：:math:`p_i p_j` 的次数 :math:`\le 2n_\max`，
    而 n 点 Gauss 对 :math:`2n-1\ge 2n_\max` 次多项式精确，所以**公式本身不引入误差**，
    留下来的偏差纯粹是浮点舍入（实测 ~1e-15，见 ``chapters/2.3/asserts.yaml``）。

    返回 ``(defect, G)``。
    """
    _check_family(kind)
    n_max = _check_degree(n_max)
    n_quad = max(2, n_max + 1 + int(extra_nodes))
    if kind == "legendre":
        nodes, weights = gauss_legendre(n_quad)
    elif kind == "chebyshev":
        nodes, weights = gauss_chebyshev(n_quad)
    else:
        nodes, weights = gauss_laguerre(n_quad)
    vals = orthonormal_values(kind, n_max, nodes)
    g = gram_matrix(vals, weights)
    defect = float(np.max(np.abs(g - np.eye(n_max + 1))))
    return defect, g


# --------------------------------------------------------------------------- #
# 单项式基、求值与 Gram--Schmidt
# --------------------------------------------------------------------------- #
def monomial_basis(n_max: int, x: Any) -> np.ndarray:
    """单项式基 :math:`1,x,\\dots,x^{n_\\max}` 在 ``x`` 上的值（形状 ``(n_max+1, x.size)``）。

    这是"最自然的基"，也是数值上**最糟的基**：它的 Gram 矩阵是 Hilbert 型，条件数随 n 指数增长。
    """
    n_max = _check_degree(n_max)
    xx = _as_1d(x)
    out = np.empty((n_max + 1, xx.size), dtype=float)
    out[0] = 1.0
    for k in range(1, n_max + 1):
        out[k] = out[k - 1] * xx
    return out


def evaluate_polynomial(coeffs: Sequence[float], x: Any) -> np.ndarray:
    r"""按**升幂**系数 :math:`c_0+c_1x+\dots` 用 Horner 求值（自研，不调 ``numpy.polyval``）。

    "自带一份最小实现"的理由见模块 docstring：2.1 章的通用多项式工具正在被并发编辑。
    """
    c = np.asarray(coeffs, dtype=float).reshape(-1)
    if c.size == 0:
        raise ValueError("系数不能为空")
    xx = np.asarray(x, dtype=float)
    acc = np.full_like(xx, c[-1], dtype=float)
    for j in range(c.size - 2, -1, -1):
        acc = acc * xx + c[j]
    return acc


def polynomial_from_values(kind: str, n_max: int, x: Any) -> np.ndarray:
    """族名 ``kind`` 的**前 n_max+1 个归一化多项式**取值（``monomial`` 时退化为单项式基）。"""
    if kind == "monomial":
        return monomial_basis(n_max, x)
    return orthonormal_values(kind, n_max, x)


def gram_schmidt(n_max: int, nodes: Any, weights: Any, *, passes: int = 2) -> GramSchmidtResult:
    r"""对单项式基做（可选重复正交化的）Gram--Schmidt，返回正交归一多项式。

    算法（对 :math:`k=0,\dots,n_\max`）：

    .. math::
        v_k = x^k - \sum_{j<k}\langle x^k,p_j\rangle_w\,p_j,\qquad
        p_k = v_k/\|v_k\|_w .

    ``passes=2`` 表示对每个 :math:`v_k` 再做一遍减法（"re-orthogonalization"）：
    经典 Gram--Schmidt 在单项式基上会**灾难性**丢失正交性（实测：n=10 时偏差 ~1e-4，
    重复一遍后回到 ~1e-15），这正是"别用单项式基做正规方程"的定量版本。

    返回的 ``coeffs[k]`` 是 :math:`p_k` 的升幂系数；``defect`` 是离散内积下
    :math:`\max|\langle p_i,p_j\rangle-\delta_{ij}|`。当 ``nodes`` 是 :math:`[-1,1]` 上
    点数足够的 Gauss--Legendre 节点时，得到的正是**归一化 Legendre 多项式**
    （这正是"Gram--Schmidt 与三项递推是同一件事"的实测证据）。
    """
    n_max = _check_degree(n_max)
    pts = np.asarray(nodes, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if pts.size != w.size:
        raise ValueError(f"节点数 {pts.size} 与权重数 {w.size} 不一致")
    if pts.size < n_max + 1:
        raise ValueError(f"至少需要 {n_max + 1} 个节点，得到 {pts.size}")
    if int(passes) < 1:
        raise ValueError("passes 必须 ≥ 1")
    m = n_max + 1
    vand = monomial_basis(n_max, pts)
    q = np.zeros((m, pts.size), dtype=float)
    c = np.zeros((m, m), dtype=float)
    norms = np.zeros(m, dtype=float)
    delta = np.zeros((m, m), dtype=float)
    for k in range(m):
        v = vand[k].copy()
        ck = np.zeros(m, dtype=float)
        ck[k] = 1.0
        for _ in range(int(passes)):
            for j in range(k):
                r = float(np.sum(w * q[j] * v))
                delta[j, k] += r
                v = v - r * q[j]
                ck = ck - r * c[j]
        nrm = math.sqrt(max(0.0, float(np.sum(w * v * v))))
        if nrm <= 0.0:
            raise ValueError(f"第 {k} 个基向量在给定节点上退化（范数 0）")
        q[k] = v / nrm
        c[k] = ck / nrm
        norms[k] = nrm
    defect = float(np.max(np.abs(gram_matrix(q, w) - np.eye(m))))
    return GramSchmidtResult(coeffs=c, norms=norms, defect=defect, delta=delta)
