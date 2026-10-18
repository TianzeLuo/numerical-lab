"""2.1 章（插值与多项式逼近）的数值内核。

数学对象
--------
给定 :math:`n+1` 个互异节点 :math:`x_0<\\dots<x_n` 与函数值 :math:`y_i=f(x_i)`，
次数不超过 :math:`n` 的插值多项式 :math:`p_n` 由 :math:`p_n(x_i)=y_i` 唯一确定
（存在唯一性：把系数写成线性方程组，Vandermonde 矩阵非奇异）。本模块给出它的
**四种等价写法**与**两种节点分布**，因为"公式等价"不等于"数值等价"：

===================  ==========================================================================
Lagrange 形式        :math:`p_n(x)=\\sum_i y_i\\ell_i(x)`，
                     :math:`\\ell_i(x)=\\prod_{j\\ne i}(x-x_j)/(x_i-x_j)`（教科书形式，:math:`O(n^2)` 每点）
Newton 差商          :math:`p_n(x)=\\sum_k f[x_0\\dots x_k]\\prod_{j<k}(x-x_j)`，
                     :math:`f[x_i\\dots x_{i+k}]=\\bigl(f[x_{i+1}\\dots x_{i+k}]-f[x_i\\dots x_{i+k-1}]\\bigr)/(x_{i+k}-x_i)`
重心形式             :math:`p_n(x)=\\sum_i \\frac{w_i}{x-x_i}y_i \\Big/ \\sum_i \\frac{w_i}{x-x_i}`，
                     :math:`w_i=1/\\prod_{j\\ne i}(x_i-x_j)`
Vandermonde 直接解   :math:`Vc=y`，:math:`V_{ij}=x_i^{\\,n-j}`（"直接解法"对照项，用 LU 解）
===================  ==========================================================================

两条中心结论（本章的断言与可视化都围着它们转）：

* **插值误差公式** 若 :math:`f\\in C^{n+1}[a,b]`，则对每个 :math:`x` 存在
  :math:`\\xi=\\xi(x)\\in(a,b)` 使
  :math:`f(x)-p_n(x)=\\frac{f^{(n+1)}(\\xi)}{(n+1)!}\\prod_i(x-x_i)`。
  见 :func:`error_envelope` / :func:`runge_derivative_over_factorial`。
* **Lebesgue 不等式** :math:`\\|f-p_n\\|_\\infty\\le(1+\\Lambda_n)\\,\\|f-p_n^*\\|_\\infty`，
  :math:`\\Lambda_n=\\max_x\\sum_i|\\ell_i(x)|` 只由节点决定：**插值算子的范数**。
  见 :func:`lebesgue_function` / :func:`lebesgue_constant`。
  等距节点的 :math:`\\Lambda_n\\approx 2^{n+1}/(e\\,n\\ln n)` 指数爆炸，
  Chebyshev 节点的 :math:`\\Lambda_n\\approx(2/\\pi)\\ln n+O(1)` 对数增长——这就是
  "换节点"能治好 Runge 现象的全部原因。

实现约束
--------
* 纯函数：无 IO、无 ``print``、无全局可变状态（POLICY P6）。
* 只用 numpy 的容器与逐元素原语；**不调用** ``numpy.linalg`` / ``numpy.fft``（P2），
  也不用 ``polyfit / polyval / interp / roots`` 这类成品算法（P3）——差商、重心权重、
  Remez 交换全部手写。
* 解 Vandermonde 系统走自研 LU（:func:`numcore.linalg.lu.lu_decompose` +
  :func:`~numcore.linalg.lu.solve_lu`），这正是本章第 4 个可视化要对照的那条路。
* 对数轴上的 0 与非有限值统一经 :func:`clamp_for_log` 处理，并把夹掉的个数如实报出来。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from numcore.linalg.lu import condition_estimate_1, exact_solve, lu_decompose, solve_lu

__all__ = [
    # 节点分布
    "NODE_KINDS",
    "equispaced_nodes",
    "chebyshev_nodes",
    "chebyshev_gauss_nodes",
    "morphed_nodes",
    "node_family",
    # 多项式（单项式基，系数按降幂）
    "poly_eval",
    "poly_mul",
    "poly_add",
    "poly_scale",
    "poly_derivative",
    "poly_from_roots",
    "poly_compose",
    # Chebyshev
    "chebyshev_t",
    "chebyshev_trig",
    "chebyshev_poly_coeffs",
    "chebyshev_monic_coeffs",
    "chebyshev_series_eval",
    "chebyshev_design_matrix",
    "chebyshev_to_monomial",
    "chebyshev_to_monomial_in_x",
    "chebyshev_monic_sup_norm",
    # Lagrange
    "lagrange_basis",
    "lagrange_interpolate",
    # Newton
    "NewtonForm",
    "divided_difference_table",
    "divided_differences",
    "newton_form",
    "newton_interpolate",
    # 重心
    "barycentric_weights",
    "barycentric_weights_chebyshev",
    "barycentric_interpolate",
    "barycentric_partition_of_unity",
    "barycentric_weight_identity",
    # Vandermonde
    "vandermonde_matrix",
    "vandermonde_solve",
    "vandermonde_condition",
    # Lebesgue
    "lebesgue_function",
    "lebesgue_max",
    "lebesgue_constant",
    "lebesgue_asymptotic_equispaced",
    "lebesgue_asymptotic_chebyshev",
    "lebesgue_growth_table",
    # 误差、Runge 与测试函数
    "runge",
    "runge_complex",
    "TEST_FUNCTIONS",
    "test_function_values",
    "derivative_over_factorial",
    "sup_derivative_over_factorial",
    "runge_derivative_over_factorial",
    "derivative_over_factorial_cauchy",
    "sup_abs_over_interval",
    "monic_product",
    "error_envelope",
    "pointwise_error",
    "max_abs_error",
    "runge_error_table",
    "interpolation_method_table",
    # Remez
    "RemezResult",
    "remez",
    "remez_error_curve",
    "alternating_extrema",
    # 对数轴安全与斜率
    "UNIT_ROUNDOFF",
    "ROUNDOFF_FLOOR",
    "clamp_for_log",
    "local_semilog_slope",
    "local_loglog_slope",
    "fit_loglog_slope",
]

#: 单位舍入 ``u = 2^{-53} ≈ 1.11e-16``（与 1.2 章同一口径：``eps/2``）。
UNIT_ROUNDOFF: float = float(np.finfo(float).eps) / 2.0

#: 对数轴地板：与 1.2 章 ``lu_heatmap`` 的残差地板同一取值——对数轴画不了 0，
#: 而插值误差在节点上**恰好**是 0。
ROUNDOFF_FLOOR: float = UNIT_ROUNDOFF

#: 节点分布的名字（manifest 的 enum 用它，不要在别处硬编码字符串）。
NODE_KINDS: tuple[str, ...] = ("equispaced", "chebyshev")


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _nodes(nodes: Any, *, name: str = "nodes") -> np.ndarray:
    """把节点数组规范化成升序 1 维 float 数组，并检查互异。**

    互异性是插值问题的前提：重复节点会让 Vandermonde 矩阵奇异、让乘积形式的
    重心权重变成除零。宁可当场报错，也不要静默返回 NaN。
    """
    arr = np.asarray(nodes, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} 不能为空")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 必须是有限数")
    if np.unique(arr).size != arr.size:
        raise ValueError(f"{name} 必须互不相同（重复节点会让插值问题奇异）")
    return arr


def _check_degree(n: int, *, name: str = "n") -> int:
    m = int(n)
    if m < 0:
        raise ValueError(f"{name} 必须非负，得到 {n!r}")
    return m


# --------------------------------------------------------------------------- #
# 节点分布
# --------------------------------------------------------------------------- #
def equispaced_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """等距节点 :math:`x_i=a+i(b-a)/n`，共 :math:`n+1` 个（含端点、升序）。

    等距节点是最自然的直觉，也是 Runge 现象的根源：它对应的 Lebesgue 常数
    :math:`\\Lambda_n\\approx2^{n+1}/(e\\,n\\ln n)` 指数爆炸（见
    :func:`lebesgue_asymptotic_equispaced`）。
    """
    m = _check_degree(n)
    if not (b > a):
        raise ValueError(f"需要 a < b，得到 [{a}, {b}]")
    if m == 0:
        raise ValueError("n 至少为 1（一个点无法插值）")
    return np.linspace(float(a), float(b), m + 1)


def chebyshev_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """Chebyshev–Lobatto 节点 :math:`x_k=\\cos(k\\pi/n)` 仿射映射到 :math:`[a,b]`（升序，含端点）。

    它是本章的"正确节点"：让 :math:`\\Lambda_n` 降到对数增长，也让
    :math:`\\max_x|\\prod_i(x-x_i)|` 在所有首一 :math:`n+1` 次多项式中最小
    （见 :func:`chebyshev_monic_coeffs` 与 notes.tex 的定理）。
    """
    m = _check_degree(n)
    if not (b > a):
        raise ValueError(f"需要 a < b，得到 [{a}, {b}]")
    if m == 0:
        raise ValueError("n 至少为 1")
    k = np.arange(m + 1, dtype=float)
    z = np.cos(np.pi * k / m)[::-1]  # 升序
    return 0.5 * (a + b) + 0.5 * (b - a) * z


def chebyshev_gauss_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """第一类 Chebyshev 节点（:math:`T_{n+1}` 的零点）:math:`x_k=\\cos\\frac{(2k+1)\\pi}{2n+2}`。

    与 Lobatto 节点一样给出对数增长的 :math:`\\Lambda_n`，但不含端点；
    2.3 章的 Gauss 求积会用同一族点，这里提供它是为了让"节点分布"这件事有第二种选择。
    """
    m = _check_degree(n)
    if not (b > a):
        raise ValueError(f"需要 a < b，得到 [{a}, {b}]")
    k = np.arange(m + 1, dtype=float)
    z = np.cos((2.0 * k + 1.0) * np.pi / (2.0 * (m + 1)))[::-1]
    return 0.5 * (a + b) + 0.5 * (b - a) * z


def morphed_nodes(n: int, t: float, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """等距 → Chebyshev 的**连续变形**：:math:`x_i(t)=(1-t)x_i^{\\rm eq}+t\\,x_i^{\\rm ch}`。

    ``t=0`` 是等距、``t=1`` 是 Chebyshev–Lobatto，中间没有定理只有实测——
    这正是"误差曲线"可视化里那条时间轴要展示的连续过渡（端点聚集是慢慢发生的，
    不是一下跳到最优）。
    """
    tt = float(t)
    if not (0.0 <= tt <= 1.0) or not math.isfinite(tt):
        raise ValueError(f"变形参数 t 必须落在 [0, 1]，得到 {t!r}")
    xeq = equispaced_nodes(n, a, b)
    xch = chebyshev_nodes(n, a, b)
    return (1.0 - tt) * xeq + tt * xch


def node_family(kind: str, n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """按名字取节点族（``"equispaced"`` / ``"chebyshev"``）。"""
    if kind == "equispaced":
        return equispaced_nodes(n, a, b)
    if kind == "chebyshev":
        return chebyshev_nodes(n, a, b)
    raise ValueError(f"未知节点分布 {kind!r}（可用 {NODE_KINDS}）")


# --------------------------------------------------------------------------- #
# 多项式：单项式基，系数**按降幂**排列（c[0] x^m + ... + c[m]）
# --------------------------------------------------------------------------- #
def poly_eval(coef: Any, x: Any) -> np.ndarray:
    """Horner 法则求 :math:`\\sum_k c_k x^{m-k}`（自研，**不用** ``np.polyval``）。

    标量进标量出、数组进数组出；返回类型统一为 ``ndarray``（0 维即标量）。
    """
    c = np.asarray(coef, dtype=float).ravel()
    if c.size == 0:
        raise ValueError("系数不能为空")
    xs = np.asarray(x, dtype=float)
    acc = np.full(xs.shape, c[0], dtype=float)
    for k in range(1, c.size):
        acc = acc * xs + c[k]
    return acc


def poly_mul(p: Any, q: Any) -> np.ndarray:
    """多项式乘法（朴素卷积，``np.convolve`` 是 P3 禁用的成品算法）。"""
    a = np.asarray(p, dtype=float).ravel()
    b = np.asarray(q, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        raise ValueError("多项式系数不能为空")
    out = np.zeros(a.size + b.size - 1, dtype=float)
    for i in range(a.size):
        if a[i] != 0.0:
            out[i : i + b.size] += a[i] * b
    return out


def poly_add(p: Any, q: Any) -> np.ndarray:
    """多项式加法（按降幂对齐，长度取长者）。"""
    a = np.asarray(p, dtype=float).ravel()
    b = np.asarray(q, dtype=float).ravel()
    m = max(a.size, b.size)
    out = np.zeros(m, dtype=float)
    out[m - a.size :] += a
    out[m - b.size :] += b
    return out


def poly_scale(p: Any, s: float) -> np.ndarray:
    """数乘。"""
    return np.asarray(p, dtype=float).ravel() * float(s)


def poly_derivative(p: Any) -> np.ndarray:
    """导数多项式的系数（降幂）；常数多项式的导数记为 ``[0.]``。"""
    c = np.asarray(p, dtype=float).ravel()
    if c.size <= 1:
        return np.zeros(1, dtype=float)
    k = np.arange(c.size - 1, 0, -1, dtype=float)
    return c[: c.size - 1] * k


def poly_from_roots(nodes: Any) -> np.ndarray:
    """首一多项式 :math:`\\prod_i(x-x_i)` 的系数（降幂），用于误差公式里的节点乘积项。"""
    nd = _nodes(nodes)
    out = np.array([1.0])
    for r in nd:
        out = poly_mul(out, np.array([1.0, -float(r)]))
    return out


def poly_compose(p: Any, q: Any) -> np.ndarray:
    """复合多项式 :math:`p(q(x))`（Horner 形式逐层乘 ``q``，结果**按真实次数**裁掉前导零）。

    用途：把 Chebyshev 级数 :math:`\\sum_j c_jT_j(t)` 在仿射变量
    :math:`t=\\alpha x+\\beta` 下展开成 :math:`x` 的单项式系数。

    裁前导零不是洁癖：调用方要用 ``len(结果)`` 判断**次数**
    （Remez 的偶次降维就靠它把 :math:`q(u)` 映回 :math:`p(x)=q(x^2)`）；
    留着前导零会让次数虚高一倍，把降维算出的多项式搬到错误的幂次上。
    """
    a = np.asarray(p, dtype=float).ravel()
    b = np.asarray(q, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        raise ValueError("多项式系数不能为空")
    nz = np.flatnonzero(a != 0.0)
    if nz.size == 0:
        return np.zeros(1, dtype=float)
    a = a[nz[0] :]
    out = np.array([float(a[0])], dtype=float)
    for coef in a[1:]:
        out = poly_mul(out, b)
        out[-1] += float(coef)
    return out


# --------------------------------------------------------------------------- #
# Chebyshev 多项式
# --------------------------------------------------------------------------- #
def chebyshev_t(n: int, x: Any) -> np.ndarray:
    """递推定义 :math:`T_0=1,\\ T_1=x,\\ T_{k+1}=2xT_k-T_{k-1}`。

    递推在 :math:`|x|>1` 时也成立（这也是它与三角定义分道扬镳的地方：
    三角定义只对 :math:`|x|\\le1` 有意义）。
    """
    m = _check_degree(n)
    xs = np.asarray(x, dtype=float)
    if m == 0:
        return np.ones_like(xs)
    if m == 1:
        return xs.copy()
    t0 = np.ones_like(xs)
    t1 = xs.copy()
    for _ in range(1, m):
        t0, t1 = t1, 2.0 * xs * t1 - t0
    return t1


def chebyshev_trig(n: int, x: Any) -> np.ndarray:
    """:math:`T_n(x)=\\cos(n\\arccos x)`（:math:`|x|\\le1`；超出范围先夹到端点）。

    与 :func:`chebyshev_t` 的递推定义**必须**一致（B 类断言），
    它们的差别只在 :math:`|x|>1`：那里三角定义需要复数。
    """
    m = _check_degree(n)
    xs = np.clip(np.asarray(x, dtype=float), -1.0, 1.0)
    return np.cos(m * np.arccos(xs))


def chebyshev_poly_coeffs(n: int) -> np.ndarray:
    """:math:`T_n` 的单项式系数（降幂）。首项系数为 :math:`2^{n-1}`（:math:`n\\ge1`）。

    例：:math:`T_2=2x^2-1`、:math:`T_3=4x^3-3x`、:math:`T_4=8x^4-8x^2+1`。
    递推 :math:`T_{k+1}=2xT_k-T_{k-1}` 在系数空间里就是"右移一位再乘 2"减去上一行
    ——注意那个**减号**：写成加法会得到 :math:`2x^2+1` 这种似是而非的东西，
    而它在 :math:`x=0` 处依然等于 :math:`\\pm1`，很容易蒙混过关。
    """
    m = _check_degree(n)
    if m == 0:
        return np.array([1.0])
    t0 = np.array([1.0])
    t1 = np.array([1.0, 0.0])
    for _ in range(1, m):
        # 2x T_k - T_{k-1}；系数按降幂，「乘 x」是在**末尾**补一个零
        t2 = poly_add(poly_scale(poly_mul(t1, np.array([1.0, 0.0])), 2.0), poly_scale(t0, -1.0))
        t0, t1 = t1, t2
    return t1


def chebyshev_monic_coeffs(n: int) -> np.ndarray:
    """:math:`T_n/2^{n-1}`（:math:`n\\ge1`）：**首一**且无穷范数最小 :math:`=2^{1-n}`。

    这就是 Chebyshev 最小最大性质的代数形式：在所有首一 :math:`n` 次多项式中，
    它的 :math:`\\|\\cdot\\|_\\infty` 最小。误差公式里的 :math:`\\prod_i(x-x_i)`
    正是首一多项式，所以"选 Chebyshev 节点"与"用最小最大多项式"是同一件事。
    """
    m = _check_degree(n)
    if m == 0:
        raise ValueError("n 至少为 1（0 次首一多项式恒为 1，没有最小最大性质可言）")
    return chebyshev_poly_coeffs(m) / float(2.0 ** (m - 1))


def chebyshev_monic_sup_norm(n: int) -> float:
    """首一 Chebyshev 多项式的 :math:`\\|\\cdot\\|_\\infty=2^{1-n}`（理论值，供参考线用）。"""
    m = _check_degree(n)
    if m == 0:
        raise ValueError("n 至少为 1")
    return float(2.0 ** (1 - m))


def chebyshev_series_eval(coef: Any, x: Any) -> np.ndarray:
    """求 :math:`\\sum_j c_jT_j(x)`（用 Clenshaw 递推，数值上等价于逐项递推但更省）。"""
    c = np.asarray(coef, dtype=float).ravel()
    if c.size == 0:
        raise ValueError("系数不能为空")
    xs = np.asarray(x, dtype=float)
    b1 = np.zeros_like(xs)
    b2 = np.zeros_like(xs)
    for j in range(c.size - 1, 0, -1):
        b1, b2 = 2.0 * xs * b1 - b2 + c[j], b1
    return xs * b1 - b2 + c[0]


def chebyshev_design_matrix(x: Any, degree: int) -> np.ndarray:
    """设计矩阵 :math:`A_{ij}=T_j(x_i)`（``j=0..degree``），Remez 的线性系统用它。"""
    m = _check_degree(degree)
    xs = np.asarray(x, dtype=float).ravel()
    A = np.empty((xs.size, m + 1), dtype=float)
    t0 = np.ones_like(xs)
    A[:, 0] = t0
    if m >= 1:
        t1 = xs.copy()
        A[:, 1] = t1
        for j in range(2, m + 1):
            t0, t1 = t1, 2.0 * xs * t1 - t0
            A[:, j] = t1
    return A


def chebyshev_to_monomial(coef: Any) -> np.ndarray:
    """:math:`\\sum_j c_jT_j` 的单项式系数（降幂）。"""
    c = np.asarray(coef, dtype=float).ravel()
    out = np.zeros(1, dtype=float)
    for j, cj in enumerate(c):
        if cj != 0.0:
            out = poly_add(out, poly_scale(chebyshev_poly_coeffs(j), float(cj)))
    return out


# --------------------------------------------------------------------------- #
# Lagrange 形式
# --------------------------------------------------------------------------- #
def lagrange_basis(nodes: Any, x: float) -> np.ndarray:
    """一组基函数值 :math:`\\ell_i(x)=\\prod_{j\\ne i}(x-x_j)/(x_i-x_j)`（朴素双循环）。"""
    nd = _nodes(nodes)
    xv = float(x)
    m = nd.size
    out = np.ones(m, dtype=float)
    for i in range(m):
        for j in range(m):
            if i != j:
                out[i] *= (xv - nd[j]) / (nd[i] - nd[j])
    return out


def lagrange_interpolate(nodes: Any, values: Any, xq: Any) -> np.ndarray:
    """教科书 Lagrange 求和 :math:`p_n(x)=\\sum_i y_i\\ell_i(x)`。

    这是"照定义算"的版本：每点 :math:`O(n^2)`，且**不**做任何稳定性处理——
    第 4 个可视化里与重心公式对照的"慢而脆"的一端。
    """
    nd = _nodes(nodes)
    vals = np.asarray(values, dtype=float).ravel()
    if vals.size != nd.size:
        raise ValueError(f"values 长度 {vals.size} 与节点数 {nd.size} 不符")
    xs = np.asarray(xq, dtype=float)
    flat = xs.ravel()
    out = np.empty(flat.size, dtype=float)
    for k, x in enumerate(flat):
        out[k] = float(lagrange_basis(nd, float(x)) @ vals)
    return out.reshape(xs.shape)


# --------------------------------------------------------------------------- #
# Newton 差商形式
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NewtonForm:
    """Newton 插值形式 :math:`\\sum_k a_k\\prod_{j<k}(x-x_j)`，``a_k=f[x_0\\dots x_k]``。"""

    nodes: np.ndarray
    coeffs: np.ndarray

    @property
    def degree(self) -> int:
        return int(self.coeffs.size - 1)

    def eval(self, xq: Any) -> np.ndarray:
        """嵌套乘法（Horner 的 Newton 版）：从最后一层往前乘 :math:`(x-x_k)` 再加系数。"""
        xs = np.asarray(xq, dtype=float)
        acc = np.full(xs.shape, float(self.coeffs[-1]), dtype=float)
        for k in range(self.coeffs.size - 2, -1, -1):
            acc = acc * (xs - float(self.nodes[k])) + float(self.coeffs[k])
        return acc

    def to_monomial(self) -> np.ndarray:
        """展开成单项式系数（降幂）；只在小 :math:`n` 上用（展开会放大舍入）。"""
        out = np.zeros(1, dtype=float)
        basis = np.array([1.0])
        for k in range(self.coeffs.size):
            out = poly_add(out, poly_scale(basis, float(self.coeffs[k])))
            basis = poly_mul(basis, np.array([1.0, -float(self.nodes[k])]))
        return out


def divided_difference_table(nodes: Any, values: Any) -> np.ndarray:
    """差商表：``T[i, j] = f[x_{i-j} ... x_i]``（下三角有值，第 0 列是函数值）。

    递推 :math:`f[x_i\\dots x_{i+k}]=\\frac{f[x_{i+1}\\dots x_{i+k}]-f[x_i\\dots x_{i+k-1}]}{x_{i+k}-x_i}`
    直接向量化成一行一次减法一次除法——表的内容就是"逐步加密节点时差商怎么变"。
    """
    nd = _nodes(nodes)
    vals = np.asarray(values, dtype=float).ravel()
    if vals.size != nd.size:
        raise ValueError(f"values 长度 {vals.size} 与节点数 {nd.size} 不符")
    m = nd.size
    T = np.zeros((m, m), dtype=float)
    T[:, 0] = vals
    for j in range(1, m):
        T[j:, j] = (T[j:, j - 1] - T[j - 1 : m - 1, j - 1]) / (nd[j:] - nd[: m - j])
    return T


def divided_differences(nodes: Any, values: Any) -> np.ndarray:
    """Newton 系数 :math:`a_k=f[x_0\\dots x_k]`（差商表的对角线）。"""
    T = divided_difference_table(nodes, values)
    return np.array(np.diag(T), dtype=float)


def newton_form(nodes: Any, values: Any) -> NewtonForm:
    """构造 :class:`NewtonForm`（节点 + 差商系数）。"""
    nd = _nodes(nodes)
    return NewtonForm(nodes=nd, coeffs=divided_differences(nd, values))


def newton_interpolate(nodes: Any, values: Any, xq: Any) -> np.ndarray:
    """Newton 形式求值（与 Lagrange / 重心应当给出同一个多项式）。"""
    return newton_form(nodes, values).eval(xq)


# --------------------------------------------------------------------------- #
# 重心（barycentric）形式
# --------------------------------------------------------------------------- #
def barycentric_weights(nodes: Any) -> np.ndarray:
    """:math:`w_i=1/\\prod_{j\\ne i}(x_i-x_j)`，**不做归一化**。

    不归一化是有意的：只有乘积形式的权重才满足**恒等式**
    :math:`\\sum_i w_i/(x-x_i)\\equiv1`（重心公式的分母）。
    需要抗上溢的版本请用 :func:`barycentric_weights_chebyshev` 或调用方自行缩放
    （整体常数在重心公式里会约掉）。
    """
    nd = _nodes(nodes)
    m = nd.size
    w = np.ones(m, dtype=float)
    for i in range(m):
        diff = np.delete(nd, i) - nd[i]
        if np.any(diff == 0.0):
            raise ValueError("节点必须互不相同")
        w[i] = 1.0 / float(np.prod(diff))
    return w


def barycentric_weights_chebyshev(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """Chebyshev–Lobatto 节点的**闭式**权重：:math:`w_j\\propto(-1)^j\\delta_j`。

    :math:`\\delta_0=\\delta_n=1/2`，其余 :math:`\\delta_j=1`。闭式与乘积形式只差一个
    与 :math:`j` 无关的常数（在重心公式里约掉），但**不用算** :math:`O(n^2)` 个乘积，
    也不会在 :math:`n` 大时上溢。
    """
    m = _check_degree(n)
    if m == 0:
        raise ValueError("n 至少为 1")
    sign = np.where(np.arange(m + 1) % 2 == 0, 1.0, -1.0)
    delta = np.ones(m + 1, dtype=float)
    delta[0] = 0.5
    delta[m] = 0.5
    return sign * delta


def barycentric_interpolate(
    nodes: Any,
    values: Any,
    xq: Any,
    weights: Any | None = None,
) -> np.ndarray:
    """重心插值公式（第二形式）：:math:`p_n(x)=\\sum_i\\frac{w_i}{x-x_i}y_i\\big/\\sum_i\\frac{w_i}{x-x_i}`。

    * 每点 :math:`O(n)`（权重预计算 :math:`O(n^2)` 一次），比 Lagrange 求和便宜一个量级；
    * **数值稳定**：权重只依赖节点，求值只是加权平均，误差量级 :math:`u\\,\\Lambda_n`
      （见 notes.tex 的定理与 ``test_barycentric_error_is_u_level``）；
    * ``x`` 恰好落在节点上时直接返回 :math:`y_i`（避免 0/0），这也是"插值多项式在节点上
      取值精确"这条不变量的落点。
    """
    nd = _nodes(nodes)
    vals = np.asarray(values, dtype=float).ravel()
    if vals.size != nd.size:
        raise ValueError(f"values 长度 {vals.size} 与节点数 {nd.size} 不符")
    w = barycentric_weights(nd) if weights is None else np.asarray(weights, dtype=float).ravel()
    if w.size != nd.size:
        raise ValueError(f"weights 长度 {w.size} 与节点数 {nd.size} 不符")
    xs = np.asarray(xq, dtype=float)
    flat = xs.ravel()
    diff = flat[:, None] - nd[None, :]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = w[None, :] / diff
        num = np.sum(ratio * vals[None, :], axis=1)
        den = np.sum(ratio, axis=1)
        out = num / den
    hit = diff == 0.0
    rows, cols = np.nonzero(hit)
    if rows.size:
        out[rows] = vals[cols]
    return out.reshape(xs.shape)


def barycentric_partition_of_unity(nodes: Any, xs: Any, weights: Any | None = None) -> np.ndarray:
    r""":math:`\sum_i\ell_i(x)` 的实测值——恒等于 1（重心形式的"单位分解"）。

    这条恒等式**不是**装饰：重心公式的分母之所以能把 :math:`p_n` 表示成加权平均，
    靠的就是它。它也解释了为什么整体缩放权重不影响结果（分子分母同阶齐次）。

    注意与 :func:`lebesgue_function` 的区别：这里**先**归一化再求和（得到 1），
    而 Lebesgue 函数是 :math:`\sum_i|\ell_i(x)|`——绝对值放在归一化**里面**。
    两者写混（先取绝对值再除以绝对值和）会得到一个"看起来像 2 的数"，
    那个数其实是 :math:`\Lambda_n(x)`：实测 Chebyshev n=6 时它到 2.08，
    而真正的 :math:`\sum_i\ell_i-1` 只有 2.2e-16。
    """
    nd = _nodes(nodes)
    w = _scaled_weights(nd, weights)
    xa = np.asarray(xs, dtype=float)
    flat = xa.ravel()
    diff = flat[:, None] - nd[None, :]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = w[None, :] / diff
        den = np.sum(ratio, axis=1)
        ell = ratio / den[:, None]
        total = np.sum(ell, axis=1)
    hit = diff == 0.0
    rows, _ = np.nonzero(hit)
    if rows.size:
        total[rows] = 1.0
    return total.reshape(xa.shape)


def barycentric_weight_identity(nodes: Any, xs: Any, weights: Any | None = None) -> np.ndarray:
    r""":math:`\bigl(\sum_i\frac{w_i}{x-x_i}\bigr)\prod_j(x-x_j)` 的实测值——恒等于 1。

    这是乘积形式权重的"代数恒等式"。写成 :math:`\sum_iw_i/(x-x_i)=1` 是**错的**：
    未归一化的和等于 :math:`1/\prod_j(x-x_j)`（随 :math:`x` 变化）；
    只有先除以同一个和（即做归一化）才得到 1。断言与 pitfalls 都按正确版本写。
    """
    nd = _nodes(nodes)
    w = np.asarray(barycentric_weights(nd) if weights is None else weights, dtype=float).ravel()
    if w.size != nd.size:
        raise ValueError(f"weights 长度 {w.size} 与节点数 {nd.size} 不符")
    xa = np.asarray(xs, dtype=float)
    flat = xa.ravel()
    diff = flat[:, None] - nd[None, :]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        total = np.sum(w[None, :] / diff, axis=1)
    hit = diff == 0.0
    rows, _ = np.nonzero(hit)
    if rows.size:
        total[rows] = np.nan  # 节点上无定义：如实给 NaN，不假装有值
    return total * np.prod(diff, axis=1)


# --------------------------------------------------------------------------- #
# Vandermonde 直接解法（对照项）
# --------------------------------------------------------------------------- #
def vandermonde_matrix(nodes: Any, degree: int | None = None) -> np.ndarray:
    """Vandermonde 矩阵 :math:`V_{ij}=x_i^{\\,m-j}`（最高次在前，与 :func:`poly_eval` 同序）。

    ``degree`` 缺省取 :math:`n`（节点数减一），此时 :math:`V` 方阵、方程组
    :math:`Vc=y` 的解就是插值多项式的系数。
    """
    nd = _nodes(nodes)
    m = nd.size - 1 if degree is None else _check_degree(degree)
    if m < 0:
        raise ValueError("degree 不能为负")
    V = np.empty((nd.size, m + 1), dtype=float)
    for j in range(m + 1):
        V[:, j] = nd ** float(m - j)
    return V


def vandermonde_solve(nodes: Any, values: Any) -> np.ndarray:
    """直接解 :math:`Vc=y` 得插值系数（**用自研 LU**，不是 ``numpy.linalg.solve``）。

    这一步在数值上是**后向稳定**的（残差 :math:`\\|Vc-y\\|\\approx u\\|V\\|\\|c\\|`），
    但系数误差被 :math:`\\kappa(V)` 放大——"公式不稳定"与"问题病态"在这里又一次分了家，
    见 ``chapters/2.1/pitfalls.md``。
    """
    nd = _nodes(nodes)
    vals = np.asarray(values, dtype=float).ravel()
    if vals.size != nd.size:
        raise ValueError(f"values 长度 {vals.size} 与节点数 {nd.size} 不符")
    V = vandermonde_matrix(nd)
    res = lu_decompose(V, pivot="partial")
    return solve_lu(res, vals)


def vandermonde_condition(nodes: Any) -> float:
    """:math:`\\kappa_1(V)`（自研 LU 求逆列的 1-范数估计，与 1.2 章同一口径）。

    :math:`n` 大时会返回 ``inf``：那不是数值溢出，而是"这个矩阵在 float64 里
    已经没有可用的逆"——**诚实报告 inf，比返回一个好看的小数要安全**。
    """
    return float(condition_estimate_1(vandermonde_matrix(nodes)))


# --------------------------------------------------------------------------- #
# Lebesgue 函数与常数
# --------------------------------------------------------------------------- #
def _scaled_weights(nodes: np.ndarray, weights: Any | None) -> np.ndarray:
    """取重心权重并做整体缩放（只影响溢出，不影响任何比值）。"""
    w = barycentric_weights(nodes) if weights is None else np.asarray(weights, dtype=float).ravel()
    if w.size != nodes.size:
        raise ValueError(f"weights 长度 {w.size} 与节点数 {nodes.size} 不符")
    scale = float(np.max(np.abs(w)))
    if scale == 0.0 or not math.isfinite(scale):
        raise ValueError("重心权重全零或非有限")
    return w / scale


def lebesgue_function(nodes: Any, xs: Any, weights: Any | None = None) -> np.ndarray:
    r"""Lebesgue 函数 :math:`\Lambda_n(x)=\sum_i|\ell_i(x)|`。

    用重心形式算每个基函数 :math:`\ell_i(x)=\frac{w_i/(x-x_i)}{\sum_j w_j/(x-x_j)}`，
    于是每点 :math:`O(n)` 而不是 :math:`O(n^2)`；节点上按定义取 :math:`\Lambda_n(x_i)=1`。
    """
    nd = _nodes(nodes)
    w = _scaled_weights(nd, weights)
    xs_arr = np.asarray(xs, dtype=float)
    flat = xs_arr.ravel()
    diff = flat[:, None] - nd[None, :]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = w[None, :] / diff
        num = np.sum(np.abs(ratio), axis=1)
        den = np.abs(np.sum(ratio, axis=1))
        out = num / den
    hit = diff == 0.0
    rows, _ = np.nonzero(hit)
    if rows.size:
        out[rows] = 1.0  # 节点上只有一个基函数为 1，其余为 0
    return out.reshape(xs_arr.shape)


def lebesgue_max(
    nodes: Any,
    a: float = -1.0,
    b: float = 1.0,
    *,
    n_grid: int = 2001,
    refine_rounds: int = 4,
    refine_points: int = 201,
    weights: Any | None = None,
) -> tuple[float, float]:
    r""":math:`(\Lambda_n, x^*)`：网格最大值 + 逐轮缩小的局部细化。

    为什么必须细化：等距节点的 :math:`\Lambda_n(x)` 峰值挤在离端点极近的地方
    （宽度 :math:`O(n^{-2})`），只撒均匀网格会**系统性低估** :math:`\Lambda_n`——
    这正是"实测值"最容易骗人的地方，所以把细化写进函数而不是留给调用方。
    """
    nd = _nodes(nodes)
    w = _scaled_weights(nd, weights)
    lo, hi = float(a), float(b)
    if not (hi > lo):
        raise ValueError(f"需要 a < b，得到 [{a}, {b}]")
    grid = max(9, int(n_grid))
    xs = np.linspace(lo, hi, grid)
    vals = lebesgue_function(nd, xs, weights=w)
    k = int(np.argmax(vals))
    best = float(vals[k])
    xbest = float(xs[k])
    width = (hi - lo) / float(grid - 1) * 4.0
    for _ in range(max(0, int(refine_rounds))):
        left = max(lo, xbest - width)
        right = min(hi, xbest + width)
        if not (right > left):
            break
        xs2 = np.linspace(left, right, max(5, int(refine_points)))
        v2 = lebesgue_function(nd, xs2, weights=w)
        k2 = int(np.argmax(v2))
        if float(v2[k2]) > best:
            best = float(v2[k2])
            xbest = float(xs2[k2])
        width *= 0.1
    return best, xbest


def lebesgue_constant(
    nodes: Any,
    a: float = -1.0,
    b: float = 1.0,
    *,
    n_grid: int = 2001,
    refine_rounds: int = 4,
) -> float:
    r""":math:`\Lambda_n=\max_{x\in[a,b]}\sum_i|\ell_i(x)|`（= 插值算子关于 :math:`\|\cdot\|_\infty` 的范数）。"""
    return lebesgue_max(nodes, a, b, n_grid=n_grid, refine_rounds=refine_rounds)[0]


def lebesgue_asymptotic_equispaced(n: int) -> float:
    r"""等距节点的经典渐近式 :math:`\Lambda_n\approx\frac{2^{n+1}}{e\,n\ln n}`（:math:`n\ge2`）。

    指数增长 :math:`\approx2^n` 的来源是 :math:`\prod_{j\ne i}|x_i-x_j|` 在端点附近
    小到 :math:`O(n^{-n})`，于是权重 :math:`w_i` 大到 :math:`O(n^{n})`。
    """
    m = _check_degree(n)
    if m <= 1:
        return 1.0
    return float(2.0 ** (m + 1) / (math.e * m * math.log(m)))


def lebesgue_asymptotic_chebyshev(n: int) -> float:
    r"""Chebyshev–Lobatto 节点的渐近式 :math:`\Lambda_n\approx\frac{2}{\pi}\ln n+0.9625`。

    对数增长（而不是指数）就是 Chebyshev 节点"值得"的全部理由；常数 0.9625 是
    第二类 Chebyshev 点的标准渐近常数，本章把它当参考线，并用实测值报告偏差。
    """
    m = _check_degree(n)
    if m <= 1:
        return 1.0
    return float(2.0 / math.pi * math.log(m) + 0.9625)


def lebesgue_growth_table(
    n_max: int,
    *,
    a: float = -1.0,
    b: float = 1.0,
    n_grid: int = 2001,
    refine_rounds: int = 4,
) -> dict[str, np.ndarray]:
    """两个节点族的 Lebesgue 常数表 + 渐近参考线 + 实测/理论斜率（规则 2 的原料）。

    返回的键（全部是等长数组，``n = 1..n_max``）：

    ``n``、``lambda_equispaced``、``lambda_chebyshev``（实测，含局部细化）、
    ``theory_equispaced``、``theory_chebyshev``（渐近式）、
    ``log10_*``（同名的十进制对数，给对数轴用）、
    ``slope_measured_*`` / ``slope_theory_*``（双对数局部斜率 :math:`d\\log_{10}\\Lambda/d\\log_{10}n`）、
    ``error_asymptotic_*``（:math:`|\\Lambda_{\\rm meas}/\\Lambda_{\\rm asym}-1|`，即"解析式的相对偏差"）。
    """
    mm = _check_degree(n_max)
    if mm < 2:
        raise ValueError("n_max 至少为 2")
    ns = np.arange(1, mm + 1)
    lam_eq = np.zeros(mm, dtype=float)
    lam_ch = np.zeros(mm, dtype=float)
    for i, n in enumerate(ns):
        lam_eq[i] = lebesgue_constant(
            equispaced_nodes(int(n), a, b), a, b, n_grid=n_grid, refine_rounds=refine_rounds
        )
        lam_ch[i] = lebesgue_constant(
            chebyshev_nodes(int(n), a, b), a, b, n_grid=n_grid, refine_rounds=refine_rounds
        )
    th_eq = np.array([lebesgue_asymptotic_equispaced(int(n)) for n in ns], dtype=float)
    th_ch = np.array([lebesgue_asymptotic_chebyshev(int(n)) for n in ns], dtype=float)
    log_n = np.log10(ns.astype(float))
    log_eq = np.log10(lam_eq)
    log_ch = np.log10(lam_ch)
    log_th_eq = np.log10(th_eq)
    log_th_ch = np.log10(th_ch)
    return {
        "n": ns.astype(float),
        "lambda_equispaced": lam_eq,
        "lambda_chebyshev": lam_ch,
        "theory_equispaced": th_eq,
        "theory_chebyshev": th_ch,
        "log10_n": log_n,
        "log10_lambda_equispaced": log_eq,
        "log10_lambda_chebyshev": log_ch,
        "log10_theory_equispaced": log_th_eq,
        "log10_theory_chebyshev": log_th_ch,
        # 斜率一律用**原始值**算（函数内部取 log10）：传 log10 值进去会得到二重对数
        "slope_measured_equispaced": local_loglog_slope(ns, lam_eq),
        "slope_measured_chebyshev": local_loglog_slope(ns, lam_ch),
        "slope_theory_equispaced": local_loglog_slope(ns, th_eq),
        "slope_theory_chebyshev": local_loglog_slope(ns, th_ch),
        "error_asymptotic_equispaced": np.abs(lam_eq / th_eq - 1.0),
        "error_asymptotic_chebyshev": np.abs(lam_ch / th_ch - 1.0),
    }


# --------------------------------------------------------------------------- #
# 误差公式与 Runge 例子
# --------------------------------------------------------------------------- #
def runge(x: Any, a: float = 25.0) -> np.ndarray:
    r"""Runge 函数 :math:`f(x)=1/(1+ax^2)`（经典取 :math:`a=25`，奇点在 :math:`x=\pm i/\sqrt a`）。

    :math:`a=25` 时奇点到实轴的距离只有 :math:`1/5`（对应的 Bernstein 椭圆参数
    :math:`\rho=1/\sqrt a+\sqrt{1+1/a}\approx1.2198`），所以等距插值很快就发散。
    """
    aa = float(a)
    if not (aa > 0.0):
        raise ValueError(f"参数 a 必须为正，得到 {a!r}")
    xs = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + aa * xs * xs)


def runge_complex(z: Any, a: float = 25.0) -> np.ndarray:
    r"""复数域上的 Runge 函数 :math:`1/(1+az^2)`——Cauchy 积分公式**必须**用它。

    :func:`runge` 会把输入转成 ``float``（它的定义域是实轴），复数输入会被静默丢掉虚部，
    于是 :func:`derivative_over_factorial_cauchy` 拿到的是实轴上的采样值，围道积分毫无意义。
    这里显式转 ``complex``：两条独立的验证路径（闭式部分分式 vs 围道积分）才真的可比。
    """
    aa = float(a)
    if not (aa > 0.0):
        raise ValueError(f"参数 a 必须为正，得到 {a!r}")
    zz = np.asarray(z, dtype=complex)
    return 1.0 / (1.0 + aa * zz * zz)


#: 本章用作"被测函数"的两支：Runge（有奇点、会发散）与 exp（整函数、包络紧）。
TEST_FUNCTIONS: tuple[str, ...] = ("exp", "runge")


def test_function_values(name: str, xs: Any, a: float = 25.0) -> np.ndarray:
    """取被测函数在 ``xs`` 上的**精确值**（对拍的真值来源）。"""
    xa = np.asarray(xs, dtype=float)
    if name == "exp":
        return np.exp(xa)
    if name == "runge":
        return runge(xa, a)
    raise ValueError(f"未知测试函数 {name!r}（可用 {TEST_FUNCTIONS}）")


def derivative_over_factorial(name: str, xs: Any, k: int, a: float = 25.0) -> np.ndarray:
    r""":math:`f^{(k)}(x)/k!`（``"exp"`` 走 :math:`e^x/k!`，``"runge"`` 走闭式部分分式）。"""
    xa = np.asarray(xs, dtype=float)
    if name == "exp":
        return np.exp(xa) / float(math.factorial(int(k)))
    if name == "runge":
        return runge_derivative_over_factorial(xa, k, a)
    raise ValueError(f"未知测试函数 {name!r}（可用 {TEST_FUNCTIONS}）")


def sup_derivative_over_factorial(
    name: str,
    k: int,
    a_int: float = -1.0,
    b_int: float = 1.0,
    *,
    a: float = 25.0,
    n_grid: int = 4001,
) -> float:
    r""":math:`\sup_{[a,b]}|f^{(k)}|/k!`——误差包络里的那个常数。

    ``"exp"`` 有闭式（:math:`e^{b}/k!`），``"runge"`` 用密网格 + 局部细化实测
    （它的上确界在 :math:`x=0` 附近，偶数阶时恰好取到 :math:`(\sqrt a)^k`）。
    """
    if name == "exp":
        return float(math.exp(float(b_int)) / float(math.factorial(int(k))))
    if name == "runge":
        return sup_abs_over_interval(
            lambda x: derivative_over_factorial("runge", x, k, a), a_int, b_int, n_grid=n_grid
        )
    raise ValueError(f"未知测试函数 {name!r}（可用 {TEST_FUNCTIONS}）")


def runge_derivative_over_factorial(x: Any, k: int, a: float = 25.0) -> np.ndarray:
    r""":math:`f^{(k)}(x)/k!` 的**闭式**（:math:`f=1/(1+ax^2)`，部分分式 + 复极点）。

    由 :math:`1+ax^2=(1+i\sqrt a\,x)(1-i\sqrt a\,x)` 得

    .. math::
        \frac{f^{(k)}(x)}{k!}=\frac12\left[
        \frac{(-i\sqrt a)^k}{(1+i\sqrt a\,x)^{k+1}}+\frac{(i\sqrt a)^k}{(1-i\sqrt a\,x)^{k+1}}\right],

    对实 :math:`x` 两项共轭，结果取实部即可。**除以阶乘是刻意的**：误差公式里出现的正是
    :math:`f^{(n+1)}/(n+1)!`，直接按这个量计算就不会遇到 :math:`(n+1)!` 上溢
    （:math:`n=30` 时 :math:`31!\\approx8\\times10^{33}` 已经开始吃掉有效位）。
    """
    m = _check_degree(k)
    aa = float(a)
    if not (aa > 0.0):
        raise ValueError(f"参数 a 必须为正，得到 {a!r}")
    xs = np.asarray(x, dtype=float)
    beta = math.sqrt(aa)
    p = 1j * beta
    den1 = (1.0 + p * xs) ** (m + 1)
    den2 = (1.0 - p * xs) ** (m + 1)
    val = 0.5 * ((-p) ** m / den1 + p**m / den2)
    return np.real(val)


def derivative_over_factorial_cauchy(
    zfun: Callable[[Any], Any],
    x: Any,
    k: int,
    *,
    radius: float = 1.0,
    n_theta: int = 512,
) -> np.ndarray:
    r"""Cauchy 积分公式给出的 :math:`f^{(k)}(x)/k!=\frac1{2\pi i}\oint\frac{f(z)}{(z-x)^{k+1}}dz`。

    离散化（:math:`z=x+re^{i\theta}`，梯形法）：

    .. math::
        \frac{f^{(k)}(x)}{k!}\approx\frac{1}{N\,r^k}\sum_{j=0}^{N-1}f(x+re^{i\theta_j})e^{-ik\theta_j}.

    这是本章唯一的"通用求导"工具（不需要知道 :math:`f` 的解析式），与
    :func:`runge_derivative_over_factorial` 的闭式互为独立实现——两条路给出同一个数，
    才说明"理论上界"不是自己编的。``zfun`` 必须能吃复数数组（见 :func:`runge_complex`）、
    ``radius`` 要小于奇点到 :math:`x` 的距离（:math:`a=25` 的 Runge 函数请用 :math:`r<0.2`）。

    .. warning::
       **半径选大了不会报错，只会给你一个完全错误的数。** 实测（:math:`a=25`、奇点在
       :math:`\pm0.2i`）：:math:`r=0.15/0.19` 时与闭式的相对偏差 1e-16，
       :math:`r=0.25/1.0` 时偏差 **100%**——轮廓把极点圈进来了，积分值被残差项主导。
       调用者必须自己保证"轮廓内解析"，这个函数查不了。
    """
    m = _check_degree(k)
    r = float(radius)
    if not (r > 0.0):
        raise ValueError(f"radius 必须为正，得到 {radius!r}")
    N = int(n_theta)
    if N < 8:
        raise ValueError("n_theta 至少为 8")
    xs = np.asarray(x, dtype=float)
    theta = 2.0 * np.pi * np.arange(N, dtype=float) / N
    z = xs.reshape(-1, 1) + r * np.exp(1j * theta).reshape(1, N)
    vals = np.asarray(zfun(z), dtype=complex)
    weights = np.exp(-1j * m * theta).reshape(1, N)
    out = np.sum(vals * weights, axis=1) / (N * r**m)
    return np.real(out).reshape(xs.shape)


def sup_abs_over_interval(
    fun: Callable[[Any], Any],
    a: float,
    b: float,
    *,
    n_grid: int = 2001,
    refine_rounds: int = 3,
) -> float:
    r"""网格上的 :math:`\max|g|` + 局部细化（把"上确界"尽量算准）。

    误差包络里要用 :math:`\sup_{[a,b]}|f^{(n+1)}|/(n+1)!`；用密网格 + 缩小区间细化，
    并在返回前乘一个 :math:`(1+10^{-12})` 的安全因子：包络是**上界**，
    宁可放宽一点点也不能因为采样点没落上而谎报"误差越界"。
    """
    lo, hi = float(a), float(b)
    if not (hi > lo):
        raise ValueError(f"需要 a < b，得到 [{a}, {b}]")
    grid = max(9, int(n_grid))
    xs = np.linspace(lo, hi, grid)
    vals = np.abs(np.asarray(fun(xs), dtype=float))
    k = int(np.argmax(vals))
    best = float(vals[k])
    xbest = float(xs[k])
    width = (hi - lo) / float(grid - 1) * 4.0
    for _ in range(max(0, int(refine_rounds))):
        left = max(lo, xbest - width)
        right = min(hi, xbest + width)
        if not (right > left):
            break
        xs2 = np.linspace(left, right, 201)
        v2 = np.abs(np.asarray(fun(xs2), dtype=float))
        k2 = int(np.argmax(v2))
        if float(v2[k2]) > best:
            best = float(v2[k2])
            xbest = float(xs2[k2])
        width *= 0.2
    return best * (1.0 + 1e-12)


def monic_product(nodes: Any, xs: Any) -> np.ndarray:
    r""":math:`\prod_i(x-x_i)` 的**带符号**值（误差公式里的节点乘积项）。"""
    nd = _nodes(nodes)
    xa = np.asarray(xs, dtype=float)
    prod = np.ones_like(xa)
    for r in nd:
        prod = prod * (xa - float(r))
    return prod


def error_envelope(nodes: Any, xs: Any, sup_derivative_over_factorial: float) -> np.ndarray:
    r"""插值误差的**理论上界包络** :math:`\frac{\sup|f^{(n+1)}|}{(n+1)!}\prod_i|x-x_i|`。

    逐点插值误差恰好等于 :math:`\frac{f^{(n+1)}(\xi(x))}{(n+1)!}\prod_i(x-x_i)`
    （:math:`\xi(x)` 未知），所以这条包络是**严格**上界：图上误差曲线永远在它下面，
    比值 :math:`|f-p|/\\text{envelope}` 的实测分布就是"这个界有多松"。
    """
    sup_val = float(sup_derivative_over_factorial)
    if sup_val < 0.0 or not math.isfinite(sup_val):
        raise ValueError(f"导数的上确界必须是有限非负数，得到 {sup_derivative_over_factorial!r}")
    return sup_val * np.abs(monic_product(nodes, xs))


def pointwise_error(f_values: Any, p_values: Any) -> np.ndarray:
    r"""逐点绝对误差 :math:`|f(x)-p_n(x)|`。"""
    return np.abs(np.asarray(f_values, dtype=float) - np.asarray(p_values, dtype=float))


def max_abs_error(f_values: Any, p_values: Any) -> float:
    r""":math:`\max_x|f(x)-p_n(x)|`（图上"看起来还行"时唯一能救你的数）。"""
    return float(np.max(pointwise_error(f_values, p_values)))


def runge_error_table(
    n_max: int,
    *,
    a: float = 25.0,
    a_int: float = -1.0,
    b_int: float = 1.0,
    n_grid: int = 1201,
) -> dict[str, np.ndarray]:
    """两个节点族在 Runge 函数上的**最大误差随 n** 的实测曲线。

    返回 ``n``、``error_equispaced``、``error_chebyshev``、``log10_*``、
    ``lambda_*``（同 n 的 Lebesgue 常数，用来解释误差为什么这么走）。
    等距那条先降后升（Runge 现象），Chebyshev 那条单调快速下降——两条曲线的分岔
    就是本章第 1 个可视化的全部内容。
    """
    mm = _check_degree(n_max)
    if mm < 2:
        raise ValueError("n_max 至少为 2")
    xs = np.linspace(a_int, b_int, max(101, int(n_grid)))
    f = runge(xs, a)
    ns = np.arange(1, mm + 1)
    err_eq = np.zeros(mm, dtype=float)
    err_ch = np.zeros(mm, dtype=float)
    lam_eq = np.zeros(mm, dtype=float)
    lam_ch = np.zeros(mm, dtype=float)
    for i, n in enumerate(ns):
        nd_eq = equispaced_nodes(int(n), a_int, b_int)
        nd_ch = chebyshev_nodes(int(n), a_int, b_int)
        err_eq[i] = max_abs_error(f, barycentric_interpolate(nd_eq, runge(nd_eq, a), xs))
        err_ch[i] = max_abs_error(f, barycentric_interpolate(nd_ch, runge(nd_ch, a), xs))
        lam_eq[i] = lebesgue_constant(nd_eq, a_int, b_int, n_grid=1001, refine_rounds=2)
        lam_ch[i] = lebesgue_constant(nd_ch, a_int, b_int, n_grid=1001, refine_rounds=2)
    return {
        "n": ns.astype(float),
        "error_equispaced": err_eq,
        "error_chebyshev": err_ch,
        "log10_error_equispaced": np.log10(err_eq),
        "log10_error_chebyshev": np.log10(err_ch),
        "lambda_equispaced": lam_eq,
        "lambda_chebyshev": lam_ch,
    }


def interpolation_method_table(
    n_max: int,
    *,
    kind: str = "equispaced",
    function: str = "exp",
    a: float = 25.0,
    a_int: float = -1.0,
    b_int: float = 1.0,
    n_grid: int = 801,
) -> dict[str, np.ndarray]:
    r"""重心公式 vs Vandermonde 直接解：三条**误差通道** + :math:`\kappa(V)` 随 n 的实测表。

    两个方法用**同一批节点、同一个函数、同一个评估网格**，所以曲线之间的差距
    只可能来自算法与基的选择。三条通道分别是：

    1. ``error_barycentric`` —— 重心公式的**函数值**误差（实测 :math:`\approx u\,\Lambda_n`）；
    2. ``error_vandermonde`` —— 解 Vandermonde 再 Horner 求值的**函数值**误差；
    3. ``error_coefficients_vandermonde`` —— Vandermonde 解的**系数**误差，真值来自
       :func:`numcore.linalg.lu.exact_solve` 的**有理数精确解**（同一组 float 输入，
       与被测的浮点 LU 完全独立）。

    再加上 ``kappa_bound_vandermonde = κ₁(V)·u``（最坏情况系数误差尺度）与
    ``residual_vandermonde``（后向误差，一直在 :math:`u` 量级）。
    实测的结论有点反直觉，也正是第 4 个可视化要讲的：**函数值通道两条曲线几乎重合
    （n=36 时 1.0e-7 vs 3.5e-8），系数通道却从 1e-13 涨到 1e0 以上**——
    系数是单项式基这个病态表示下的中间产物，而重心公式根本不产生系数。

    ``function`` 缺省取 ``"exp"`` 而不是 Runge：Runge 函数本身插值就极难
    （n=30 时误差 2.4e3），会盖住一切算法差异；整函数的插值误差掉到舍入底以下，
    才看得清算法通道的区别。见 ``chapters/2.1/pitfalls.md``。
    """
    mm = _check_degree(n_max)
    if mm < 2:
        raise ValueError("n_max 至少为 2")
    xs = np.linspace(a_int, b_int, max(101, int(n_grid)))
    f = test_function_values(function, xs, a)
    ns = np.arange(1, mm + 1)
    err_b = np.zeros(mm, dtype=float)
    err_v = np.zeros(mm, dtype=float)
    cerr = np.zeros(mm, dtype=float)
    kappa = np.zeros(mm, dtype=float)
    resid = np.zeros(mm, dtype=float)
    for i, n in enumerate(ns):
        nd = node_family(kind, int(n), a_int, b_int)
        vals = test_function_values(function, nd, a)
        p_b = barycentric_interpolate(nd, vals, xs)
        err_b[i] = max_abs_error(f, p_b)
        V = vandermonde_matrix(nd)
        c = vandermonde_solve(nd, vals)
        with np.errstate(over="ignore", invalid="ignore"):
            p_v = poly_eval(c, xs)
        bad = ~np.isfinite(p_v)
        if np.any(bad):
            p_v = np.where(bad, np.inf, p_v)
        err_v[i] = max_abs_error(f, p_v)
        kappa[i] = float(condition_estimate_1(V))
        # 系数误差：真值 = 有理数精确解（自研 Fraction 高斯消元，不用 numpy.linalg）
        c_exact = exact_solve(V, vals)
        den = float(np.max(np.abs(c_exact)))
        cerr[i] = float(np.max(np.abs(c - c_exact))) / den if den > 0.0 else float("inf")
        with np.errstate(over="ignore", invalid="ignore"):
            scale = float(np.max(np.sum(np.abs(V), axis=1))) * float(np.max(np.abs(c))) + float(
                np.max(np.abs(vals))
            )
            r = float(np.max(np.abs(V @ c - vals)))
        resid[i] = r / scale if scale > 0.0 and math.isfinite(scale) else float("inf")
    return {
        "n": ns.astype(float),
        "error_barycentric": err_b,
        "error_vandermonde": err_v,
        "error_coefficients_vandermonde": cerr,
        "kappa_vandermonde": kappa,
        "kappa_bound_vandermonde": kappa * UNIT_ROUNDOFF,
        "residual_vandermonde": resid,
        "log10_error_barycentric": np.log10(np.maximum(err_b, ROUNDOFF_FLOOR)),
        "log10_error_vandermonde": np.log10(np.maximum(err_v, ROUNDOFF_FLOOR)),
        "log10_error_coefficients_vandermonde": np.log10(np.maximum(cerr, ROUNDOFF_FLOOR)),
        "log10_kappa_vandermonde": np.log10(kappa),
        "unit_roundoff": np.full(mm, UNIT_ROUNDOFF),
    }


# --------------------------------------------------------------------------- #
# Remez 交换算法：最佳一致逼近
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RemezResult:
    """Remez 交换算法的输出（含**交错证书**——它才是"最优"的证据）。"""

    degree: int
    a: float
    b: float
    coeffs_chebyshev: np.ndarray
    coeffs_monomial: np.ndarray
    error_level: float
    reference_x: np.ndarray
    reference_err: np.ndarray
    n_iter: int
    converged: bool
    max_error: float
    max_error_x: float
    n_extrema: int
    alternation_ok: bool
    equioscillation_spread: float = float("nan")
    """等幅振荡点上 :math:`\\max|e|/\\min|e|`（真等幅振荡 = 1）。"""

    def eval(self, xq: Any) -> np.ndarray:
        """用单项式系数求值（Horner）。"""
        return poly_eval(self.coeffs_monomial, xq)

    def error(self, xq: Any) -> np.ndarray:
        """带符号误差 :math:`f(x)-p(x)` 需要调用方给 ``f``；这里只给多项式值之外的部分。"""
        return self.eval(xq)


def _parabolic_peak(xs: np.ndarray, vals: np.ndarray, i: int) -> float:
    """三点抛物线顶点细化：把网格上的极值点挪到真正的极值位置。

    对 :math:`|e|` 在 ``i`` 附近做二次拟合（三点等距），顶点偏移

    .. math::
        \frac{h}{2}\frac{|e_{i-1}|-|e_{i+1}|}{|e_{i-1}|-2|e_i|+|e_{i+1}|}

    （分母为 0 或非有限时退回网格点）。不做这一步，Remez 的收敛会慢一个量级。
    """
    if i <= 0 or i >= xs.size - 1:
        return float(xs[i])
    y0 = abs(float(vals[i - 1]))
    y1 = abs(float(vals[i]))
    y2 = abs(float(vals[i + 1]))
    den = y0 - 2.0 * y1 + y2
    h = float(xs[i + 1] - xs[i])
    if den == 0.0 or not math.isfinite(den):
        return float(xs[i])
    shift = 0.5 * h * (y0 - y2) / den
    if not math.isfinite(shift) or abs(shift) > h:
        return float(xs[i])
    return float(xs[i] + shift)


def _grid_peaks(xs: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """``|vals|`` 在网格上的局部极大点下标（含端点）。"""
    mag = np.abs(vals)
    peak = np.zeros(mag.size, dtype=bool)
    peak[1:-1] = (mag[1:-1] >= mag[:-2]) & (mag[1:-1] >= mag[2:])
    peak[0] = mag[0] >= mag[1]
    peak[-1] = mag[-1] >= mag[-2]
    return np.flatnonzero(peak)


def _alternating_subsequence(xs: np.ndarray, es: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """从按 x 升序的极值点里贪心取符号严格交错的子列（同号相邻时保留 |e| 大者）。"""
    keep: list[int] = []
    for t in range(xs.size):
        if not keep:
            keep.append(t)
            continue
        last = keep[-1]
        if np.sign(es[t]) == np.sign(es[last]):
            if abs(float(es[t])) > abs(float(es[last])):
                keep[-1] = t
        else:
            keep.append(t)
    sel = np.array(keep, dtype=int)
    return xs[sel], es[sel]


def alternating_extrema(
    err_fun: Callable[[Any], Any],
    count: int,
    a: float,
    b: float,
    *,
    n_grid: int = 2001,
) -> tuple[np.ndarray, np.ndarray, int]:
    r"""在 :math:`[a,b]` 上找 ``count`` 个**符号交错**的极值点（Remez 的交换步）。

    做法：密网格上取 :math:`|e|` 的局部极大 → 抛物线细化 → 贪心保留符号交错序列
    → 若候选多于 ``count``，反复丢掉 :math:`|e|` 最小的那个（丢掉序列中的任何一点
    都不会破坏交错性）。

    返回 ``(x, e, n_candidates)``：``x`` 升序、``e`` 是对应带符号误差，
    ``n_candidates`` 是交错候选总数。
    """
    m = _check_degree(count)
    if m < 2:
        raise ValueError("count 至少为 2")
    grid = max(9, int(n_grid))
    xs = np.linspace(float(a), float(b), grid)
    e = np.asarray(err_fun(xs), dtype=float).ravel()
    idx = _grid_peaks(xs, e)
    if idx.size == 0:  # pragma: no cover - 防御：常数误差
        idx = np.array([int(np.argmax(np.abs(e)))])
    cand_x = np.empty(idx.size, dtype=float)
    cand_e = np.empty(idx.size, dtype=float)
    for t, i in enumerate(idx):
        px = _parabolic_peak(xs, e, int(i))
        cand_x[t] = px
        # 细化后的点直接用误差函数取值（不用 np.interp：它是 P3 禁用的成品算法）
        cand_e[t] = float(np.asarray(err_fun(np.array([px], dtype=float)), dtype=float).ravel()[0])
    sway_x, sway_e = _alternating_subsequence(cand_x, cand_e)
    n_cand = int(sway_x.size)
    keep = list(range(sway_x.size))
    while len(keep) > m:
        worst = min(range(len(keep)), key=lambda t: abs(float(sway_e[keep[t]])))
        keep.pop(worst)
    sel = np.array(keep, dtype=int)
    return sway_x[sel], sway_e[sel], n_cand


def _equioscillation_certificate(
    err_fun: Callable[[Any], Any],
    a: float,
    b: float,
    *,
    n_grid: int = 4001,
    rel: float = 1e-3,
    min_count: int = 2,
) -> tuple[np.ndarray, np.ndarray, int, float, bool]:
    r"""交错证书：所有满足 :math:`|e|\ge(1-	ext{rel})\max|e|` 且符号交错的极值点。

    返回 ``(x, e, n, spread, ok)``：``n`` 是等幅振荡点数，``spread`` 是这些点上
    :math:`\max|e|/\min|e|`（真正的等幅振荡应当 = 1），``ok`` 表示
    "点数不少于 ``min_count`` 且符号严格交错"。

    为什么不用"参考集"当证书：参考集是算法的**输入**，用它自己证明自己等于循环论证；
    这里从稠密网格重新测量，偶函数 + 偶次数的情形还会自然地多出一个点
    （对称性把 :math:`x=0` 也变成极值点），实测值因此更诚实。
    """
    grid = max(101, int(n_grid))
    xs = np.linspace(float(a), float(b), grid)
    e = np.asarray(err_fun(xs), dtype=float).ravel()
    mag = np.abs(e)
    top = float(np.max(mag))
    if top == 0.0:
        return np.zeros(0), np.zeros(0), 0, float("nan"), False
    idx = _grid_peaks(xs, e)
    pick = [int(i) for i in idx if mag[int(i)] >= top * (1.0 - float(rel))]
    if not pick:
        pick = [int(np.argmax(mag))]
    px = np.empty(len(pick), dtype=float)
    pe = np.empty(len(pick), dtype=float)
    for t, i in enumerate(pick):
        px[t] = _parabolic_peak(xs, e, int(i))
        pe[t] = float(np.asarray(err_fun(np.array([px[t]])), dtype=float).ravel()[0])
    order = np.argsort(px)
    px, pe = px[order], pe[order]
    sx, se = _alternating_subsequence(px, pe)
    n = int(sx.size)
    spread = float(np.max(np.abs(se)) / np.min(np.abs(se))) if n and np.all(np.abs(se) > 0.0) else float("nan")
    ok = bool(n >= int(min_count) and np.all(np.sign(se[1:]) * np.sign(se[:-1]) < 0.0))
    return sx, se, n, spread, ok


def _remez_error_fun(
    fun: Callable[[float], float],
    coef: np.ndarray,
    alpha: float,
    beta: float,
) -> Callable[[Any], np.ndarray]:
    """在 :math:`t` 变量上构造误差函数 :math:`f(x(t))-\\sum_jc_jT_j(t)`，:math:`x=(t-\\beta)/\\alpha`。

    ``alpha,beta`` 是**正向**映射 :math:`t=\\alpha x+\\beta` 的系数（:func:`_remez_core`
    用它在 :math:`[a,b]` 与 :math:`[-1,1]` 之间换坐标）。这里要的是逆映射：
    写成 :math:`x=\\alpha t+\\beta` 是最容易犯又最难发现的错——当区间就是
    :math:`[-1,1]` 时 :math:`\\alpha=1,\\beta=0`，两种写法**完全一样**，
    只有在非对称区间（比如偶次降维用的 :math:`u\\in[0,1]`）上才会暴露。
    """

    def err(t: Any) -> np.ndarray:
        tt = np.asarray(t, dtype=float)
        xx = (tt - beta) / alpha
        fv = np.array([float(fun(float(v))) for v in np.ravel(xx)], dtype=float).reshape(tt.shape)
        return fv - chebyshev_series_eval(coef, tt)

    return err


def _initial_reference(degree: int, *, skew: float = 0.99) -> np.ndarray:
    r"""初始参考集：:math:`T_{m+1}` 的 :math:`m+2` 个极值点，内点乘 ``skew`` 打破退化。

    为什么需要 ``skew``：在这些点上 :math:`T_{m+1}(t_i)=(-1)^i`，于是交错方程组的
    最后一列与 :math:`T_{m+1}` 那一列**重合**；这时解出来的 :math:`E` 恰好是
    :math:`T_{m+1}` 的系数。奇特的后果：偶函数配对称参考集时那个系数必为 0，
    算法会"收敛"到 :math:`E=0` 的假答案（实测 runge6：``E=-6e-17`` 而真实误差 0.45）。
    把内点收缩 1% 即可避开这个重合（偶函数 + 偶次数另有专门处理，见 :func:`remez`）。
    """
    m = _check_degree(degree)
    t = np.cos(np.pi * np.arange(m + 2, dtype=float) / (m + 1))[::-1].copy()
    if m >= 1:
        # 非对称收缩：均匀乘一个常数**不破坏对称性**，对称参考集 + 奇偶函数仍会退化
        # （实测 atan 在 m=5 时第一轮就假收敛到 E=-2e-17）。按序号线性收缩才是真的破对称。
        taper = 1.0 - (1.0 - float(skew)) * np.arange(1, m + 1, dtype=float) / (m + 1.0)
        t[1:-1] = t[1:-1] * taper
    return t


def _is_even_function(fun: Callable[[float], float], half: float, *, n_grid: int = 17, tol: float = 1e-12) -> bool:
    """数值判定 :math:`f(-x)=f(x)`（只在对称区间 :math:`[-h,h]` 上问）。"""
    xs = np.linspace(-half, half, max(5, int(n_grid)))
    f1 = np.array([float(fun(float(x))) for x in xs], dtype=float)
    f2 = np.array([float(fun(float(-x))) for x in xs], dtype=float)
    scale = max(float(np.max(np.abs(f1))), 1e-300)
    return bool(np.max(np.abs(f1 - f2)) <= float(tol) * scale)


def _remez_core(
    fun: Callable[[float], float],
    degree: int,
    a: float,
    b: float,
    *,
    n_grid: int,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float, int, int, bool]:
    """标准 Remez 交换。

    返回 ``(cheb 系数, 参考点 x, E, alpha, beta, 迭代数, 交替候选数, 是否收敛)``。
    """
    m = _check_degree(degree)
    aa, bb = float(a), float(b)
    alpha = 2.0 / (bb - aa)
    beta = -(aa + bb) / (bb - aa)
    to_x = lambda t: (np.asarray(t, dtype=float) - beta) / alpha  # noqa: E731

    ref_t = _initial_reference(m)
    prev_e = float("nan")
    coef = np.zeros(m + 1, dtype=float)
    E = 0.0
    n_iter = 0
    converged = False
    n_cand = 0
    for it in range(1, max(1, int(max_iter)) + 1):
        n_iter = it
        A = np.empty((m + 2, m + 2), dtype=float)
        A[:, : m + 1] = chebyshev_design_matrix(ref_t, m)
        A[:, m + 1] = np.where(np.arange(m + 2) % 2 == 0, 1.0, -1.0)
        rhs = np.array([float(fun(float(v))) for v in to_x(ref_t)], dtype=float)
        sol = solve_lu(lu_decompose(A, pivot="partial"), rhs)
        coef = np.array(sol[: m + 1], dtype=float)
        E = float(sol[m + 1])
        err = _remez_error_fun(fun, coef, alpha, beta)
        new_t, _new_e, n_cand = alternating_extrema(err, m + 2, -1.0, 1.0, n_grid=n_grid)
        if new_t.size != m + 2:  # pragma: no cover - 防御
            break
        moved = float(np.max(np.abs(new_t - ref_t)))
        ref_t = new_t
        if math.isfinite(prev_e) and abs(abs(E) - abs(prev_e)) <= tol * max(abs(E), 1e-300):
            converged = True
            break
        if moved == 0.0:
            converged = True
            break
        prev_e = E
    return coef, to_x(ref_t), E, alpha, beta, n_iter, n_cand, converged


def _monomial_coeffs_cheb_in_t(coef: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    r""":math:`\sum_jc_jT_j(\alpha x+\beta)` 的单项式系数（降幂）。"""
    return chebyshev_to_monomial_in_x(coef, alpha, beta)


def _remez_even_reduction(
    fun: Callable[[float], float],
    degree: int,
    b: float,
    *,
    n_grid: int,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, float, int, bool]:
    r"""偶函数 + 偶次数的降维求解：在 :math:`u=x^2\in[0,b^2]` 上求 :math:`F(u)=f(\sqrt u)`。

    为什么必须降维：偶函数配对称参考集时交错方程组**必然奇异**（对称性把奇次项与
    交错符号列凑成同一个零空间），实测 runge6 的 :math:`E` 会算成 :math:`-6\times10^{-17}`。
    数学上正确的做法是先证明"偶函数的最佳逼近必为偶多项式"
    （若不偶，则它的偶部同样好且次数不升，由 Haar 空间的唯一性得证），
    于是令 :math:`p(x)=q(x^2)`，在 :math:`[0,b^2]` 上对 :math:`F` 做 :math:`m/2` 次 Remez。
    实测（与 LP 最优对照）：:math:`m=2` 时 :math:`E=0.3231146`、等幅振荡点 5 个，
    :math:`m=6` 时 :math:`E=0.1459473`、9 个——正好是 :math:`[0,b^2]` 上的
    :math:`m/2+2` 个点映回来（:math:`u=0` 只映到 :math:`x=0`，其余成对）。
    """
    m = _check_degree(degree)
    half = m // 2
    F = lambda u: float(fun(math.sqrt(max(0.0, float(u)))))  # noqa: E731
    coef_u, _ref_u, E, alpha_u, beta_u, n_iter, _nc, converged = _remez_core(
        F, half, 0.0, float(b) * float(b), n_grid=n_grid, max_iter=max_iter, tol=tol
    )
    # q(u) 的单项式系数（降幂） → p(x) = q(x^2)
    q = _monomial_coeffs_cheb_in_t(coef_u, alpha_u, beta_u)
    deg_q = q.size - 1
    p = np.zeros(2 * deg_q + 1, dtype=float)
    for k, ck in enumerate(q):  # q[k] 对应 u^{deg_q-k} = x^{2(deg_q-k)}，降幂排在第 2k 位
        p[2 * k] = ck
    return p, float(E), n_iter, bool(converged)


def remez(
    fun: Callable[[float], float],
    degree: int,
    a: float = -1.0,
    b: float = 1.0,
    *,
    n_grid: int = 4001,
    max_iter: int = 60,
    tol: float = 1e-9,
) -> RemezResult:
    r"""Remez 交换算法：求 :math:`[a,b]` 上 :math:`f` 的 :math:`m` 次**最佳一致逼近**多项式。

    算法（每一步都可被交错定理检验）：

    1. 取 :math:`m+2` 个 Chebyshev 极值点（内点收缩 1%）作为初始参考集；
    2. 解 :math:`(m+2)	imes(m+2)` 线性系统
       :math:`\sum_jc_jT_j(t_i)+(-1)^iE=f(t_i)`（Chebyshev 基 + 交错符号列，
       用自研 LU 解，:math:`m\le20` 时条件数完全可控）；
    3. 在密网格上找误差的 :math:`m+2` 个交错极值（:func:`alternating_extrema`），换掉参考集；
    4. :math:`E` 不再变化时停止。

    两个**实测逼出来的**细节：

    * 初始参考集必须打破 :math:`T_{m+1}(t_i)=(-1)^i` 的重合，否则偶函数会假收敛到 :math:`E=0`；
    * **偶函数 + 偶次数**（区间对称）走降维分支 :func:`_remez_even_reduction`：
      在 :math:`u=x^2` 上求 :math:`m/2` 次最佳逼近。这类问题在 :math:`x` 空间的
      等幅振荡点数是 :math:`m+3`（不是 :math:`m+2`）——对称性把 :math:`x=0` 也变成极值点。

    收敛后 :math:`|f-p|` 在等幅振荡点上取同一个值——**交错定理**说明这等价于最优性
    （这正是断言要测的性质，而不是"我说它最优"）。

    ``tol`` 是**实用**收敛阈值：判据是 :math:`|E_{k+1}|-|E_k|\le\text{tol}\cdot|E_k|`。
    取机器精度级（1e-13）反而永远不收敛——参考集被抛物线细化推着在 :math:`10^{-13}`
    附近抖动，实测 :math:`\Delta E/|E|` 的地板就在 :math:`10^{-11}\sim10^{-15}`。
    缺省 1e-9 给出的 :math:`E` 与把上限定到 200 次迭代时**逐位相同**。
    """
    m = _check_degree(degree)
    aa, bb = float(a), float(b)
    if not (bb > aa):
        raise ValueError(f"需要 a < b，得到 [{a}, {b}]")
    if m % 2 == 0 and abs(aa + bb) <= 1e-12 * max(1.0, abs(aa), abs(bb)) and _is_even_function(fun, bb):
        p, E, n_iter, converged = _remez_even_reduction(
            fun, m, bb, n_grid=n_grid, max_iter=max_iter, tol=tol
        )
        coef_cheb = np.zeros(m + 1, dtype=float)  # 偶解在 Chebyshev 基下没有闭式，留空
        err = lambda x: np.array(  # noqa: E731
            [float(fun(float(v))) for v in np.ravel(np.asarray(x, dtype=float))], dtype=float
        ).reshape(np.asarray(x, dtype=float).shape) - poly_eval(p, x)
    else:
        coef_cheb, _ref_x, E, alpha, beta, n_iter, _nc, converged = _remez_core(
            fun, m, aa, bb, n_grid=n_grid, max_iter=max_iter, tol=tol
        )
        p = _monomial_coeffs_cheb_in_t(coef_cheb, alpha, beta)
        err_t = _remez_error_fun(fun, coef_cheb, alpha, beta)

        def err(x: Any) -> np.ndarray:  # noqa: E731 - 把 t 变量上的误差换成 x 变量
            r"""证书与 max_error 都在 **x** 变量上量；t 只活在 :func:`_remez_core` 内部。

            不换回来的话，非对称区间会静默错位：:math:`[0,1]` 上
            :math:`t=\alpha x+\beta=2x-1`，直接拿 ``err(x)`` 只在
            :math:`x\in[0.5,1]`（即 :math:`t\in[0,1]`）上量误差，
            证书点数会从 6 掉到 3——而 :math:`[-1,1]` 上 :math:`\alpha=1,\beta=0`，
            两种写法完全一样，所以这个错在对称区间上根本不会暴露。
            """
            xx = np.asarray(x, dtype=float)
            return np.asarray(err_t(alpha * xx + beta), dtype=float)
    xs = np.linspace(aa, bb, max(2001, int(n_grid)))
    e_vals = np.asarray(err(xs), dtype=float)
    kbest = int(np.argmax(np.abs(e_vals)))
    max_err = float(np.abs(e_vals[kbest]))
    cert_x, cert_e, n_extrema, spread, alt_ok = _equioscillation_certificate(
        err, aa, bb, n_grid=max(2001, int(n_grid)), rel=1e-3, min_count=m + 2
    )
    return RemezResult(
        degree=m,
        a=aa,
        b=bb,
        coeffs_chebyshev=coef_cheb,
        coeffs_monomial=p,
        error_level=float(E),
        reference_x=cert_x,
        reference_err=cert_e,
        n_iter=n_iter,
        converged=bool(converged),
        max_error=max_err,
        max_error_x=float(xs[kbest]),
        n_extrema=n_extrema,
        alternation_ok=alt_ok,
        equioscillation_spread=float(spread),
    )



def chebyshev_to_monomial_in_x(coef: Any, alpha: float, beta: float) -> np.ndarray:
    r"""把 :math:`\sum_jc_jT_j(t)`（:math:`t=\alpha x+\beta`）展开成 :math:`x` 的单项式系数。"""
    c = np.asarray(coef, dtype=float).ravel()
    out = np.zeros(1, dtype=float)
    for j, cj in enumerate(c):
        if cj != 0.0:
            out = poly_add(out, poly_scale(poly_compose(chebyshev_poly_coeffs(j), np.array([alpha, beta])), float(cj)))
    return out


def remez_error_curve(res: RemezResult, fun: Callable[[float], float], xq: Any) -> np.ndarray:
    r"""带符号误差 :math:`f(x)-p(x)`（交错图上正负交替就是这条曲线在验证最优性）。"""
    xs = np.asarray(xq, dtype=float)
    fv = np.array([float(fun(float(v))) for v in np.ravel(xs)], dtype=float).reshape(xs.shape)
    return fv - res.eval(xs)


# --------------------------------------------------------------------------- #
# 对数轴安全与斜率
# --------------------------------------------------------------------------- #
def clamp_for_log(
    values: Any,
    floor: float | None = None,
    ceiling: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    r"""把数组夹到对数轴能画的范围，并**如实报告夹了多少个**。

    三件事一起做：

    * ``0``（插值误差在节点上恰好是 0！）→ ``floor``，1.2 章的口径是 :math:`u/2`；
    * ``inf/nan`` → ``ceiling``（或 ``floor``），数量记在 ``n_nonfinite``；
    * 负数 → 取绝对值？**不**。负数在对数轴上是没有定义的，直接报错，
      免得把"符号错误"悄悄画成"很小"。

    返回 ``(clamped, info)``，``info`` 里的键会原样进帧的 ``meta``——
    图上的地板不是秘密。
    """
    arr = np.asarray(values, dtype=float)
    lo = ROUNDOFF_FLOOR if floor is None else float(floor)
    hi = float(ceiling) if ceiling is not None else max(lo * 10.0, float(np.max(arr[np.isfinite(arr)])) if np.any(np.isfinite(arr)) else lo * 10.0)
    if not (hi > lo):
        hi = lo * 10.0
    if np.any(arr[np.isfinite(arr)] < 0.0):
        raise ValueError("clamp_for_log 不接受负值（对数轴没有定义）")
    finite = np.isfinite(arr)
    n_nonfinite = int(arr.size - np.count_nonzero(finite))
    filled = np.where(finite, arr, hi)
    below = filled < lo
    above = filled > hi
    n_floor = int(np.count_nonzero(below))
    n_ceiling = int(np.count_nonzero(above)) + n_nonfinite
    out = np.clip(filled, lo, hi)
    info: dict[str, float] = {
        "floor": float(lo),
        "ceiling": float(hi),
        "n_floor": float(n_floor),
        "n_ceiling": float(n_ceiling),
        "n_nonfinite": float(n_nonfinite),
    }
    return out, info


def _central_diff(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """自研中心差分（``np.gradient`` 是 P3 禁用的成品算法）。端点用单侧差分。"""
    out = np.empty_like(y)
    if y.size == 2:
        out[:] = (y[1] - y[0]) / (x[1] - x[0])
        return out
    out[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    out[0] = (y[1] - y[0]) / (x[1] - x[0])
    out[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return out


def local_semilog_slope(x: Any, y: Any) -> np.ndarray:
    r""":math:`d\log_{10}y/dx` 的中心差分（端点单侧差分）。

    用于"log 纵轴 + 线性横轴"的图：等距节点的 :math:`\log_{10}\Lambda_n` 斜率
    在 :math:`n` 大时趋于 :math:`\log_{10}2\approx0.30103`，
    Chebyshev 的那条趋于 0——两条参考线的斜率就该这么算。
    """
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    if xa.size != ya.size or xa.size < 2:
        raise ValueError("x 与 y 必须等长且至少 2 个点")
    if np.any(ya <= 0.0):
        raise ValueError("local_semilog_slope 需要正的 y")
    return _central_diff(np.log10(ya), xa)


def local_loglog_slope(x: Any, y: Any) -> np.ndarray:
    r""":math:`d\log_{10}y/d\log_{10}x` 的中心差分（双对数图的局部斜率）。

    规则 2 要求双对数图同时画出"实测拟合斜率"与"理论斜率"：两者都必须由**同一个
    算子**作用在各自的曲线上得到，否则比较的就不是斜率而是两种算法。
    """
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    if xa.size != ya.size or xa.size < 2:
        raise ValueError("x 与 y 必须等长且至少 2 个点")
    if np.any(xa <= 0.0) or np.any(ya <= 0.0):
        raise ValueError("local_loglog_slope 需要正的 x 与 y")
    return _central_diff(np.log10(ya), np.log10(xa))


def fit_loglog_slope(x: Any, y: Any, i0: int = 0, i1: int | None = None) -> float:
    r"""窗口 ``[i0, i1)`` 上的双对数最小二乘斜率（实测拟合斜率，画参考线用）。"""
    xa = np.log10(np.asarray(x, dtype=float).ravel())
    ya = np.log10(np.asarray(y, dtype=float).ravel())
    if xa.size != ya.size:
        raise ValueError("x 与 y 必须等长")
    j1 = xa.size if i1 is None else int(i1)
    xw = xa[int(i0) : j1]
    yw = ya[int(i0) : j1]
    if xw.size < 2:
        raise ValueError("拟合窗口至少要有 2 个点")
    xm = float(np.mean(xw))
    ym = float(np.mean(yw))
    den = float(np.sum((xw - xm) ** 2))
    if den == 0.0:
        raise ValueError("拟合窗口内 x 没有变化")
    return float(np.sum((xw - xm) * (yw - ym)) / den)
