r"""函数逼近：最小二乘（L²）投影、最佳一致逼近（Remez）与误差衰减率。

数学定义
--------
设 :math:`f\in C[a,b]`，:math:`\Phi_n=\operatorname{span}\{\varphi_0,\dots,\varphi_n\}`。

* **最小二乘投影** 是 :math:`\Phi_n` 中在 :math:`L^2_w` 范数下离 :math:`f` 最近的那个元素：

  .. math::
      P_n f=\sum_{k\le n}c_k\varphi_k,\qquad
      G c=b,\quad G_{ij}=\langle\varphi_i,\varphi_j\rangle_w,\quad b_i=\langle f,\varphi_i\rangle_w,

  也就是把 :math:`f` 正交投影到有限维子空间（正规方程 :math:`Gc=b`）。
  本章用**手写正规方程 + 自研 LU** 解它（P3：不许 ``lstsq``）。

* **最佳一致逼近** 是 :math:`\Phi_n` 中最小化 :math:`\|f-p\|_\infty` 的那个元素；
  它的特征是**交错定理**：误差在 :math:`n+2` 个点上取到符号交替的等幅极值
  （Chebyshev, 1854；证明见 ``chapters/2.3/notes.tex`` 定理 4）。Remez 交换算法就是在
  迭代地寻找这组交错点。

* **误差衰减率** 由光滑性决定（Jackson 定理的思路）：若 :math:`f` 的前
  :math:`\nu` 阶导数连续、:math:`f^{(\nu)}` 有界变差，则系数 :math:`c_k=O(k^{-\nu-1})`，
  于是 :math:`\|f-P_nf\|=O(n^{-\nu-1/2})`；若 :math:`f` 在 Bernstein 椭圆
  :math:`E_\rho` 内解析，则 :math:`\|f-P_nf\|=O(\rho^{-n})`（指数/谱收敛）。
  两条率都是**可以实测**的：前者在双对数图上是一条直线，后者在半对数图上是一条直线。

实现约束
--------
* 只用 ``numpy`` 容器与逐元素原语（P3）：没有 ``lstsq/solve/inv/polyfit/polyval``；
* 线性方程组一律经 :func:`numcore.linalg.lu` 的自研 LU；
* 没有 IO / ``print`` / 全局可变状态（P6）。
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from numcore.interp.orthopoly import (
    ROUNDOFF,
    gauss_legendre,
    gram_matrix,
    monomial_basis,
    orthonormal_values,
)
from numcore.linalg.lu import condition_estimate_1, lu_decompose, solve_lu

__all__ = [
    "ProjectionResult",
    "RemezResult",
    "BASES",
    "FUNCTION_NAMES",
    "basis_values",
    "weighted_norm",
    "solve_linear_system",
    "gram_condition_estimate",
    "composite_gauss_legendre",
    "project_onto_basis",
    "l2_projection",
    "projection_error_curve",
    "projector_matrix",
    "idempotence_defect",
    "clenshaw",
    "clenshaw_basis",
    "chebyshev_to_monomial",
    "remez",
    "refine_extrema",
    "alternation_ratio",
    "fit_loglog_slope",
    "fit_semilog_slope",
    "bernstein_rho",
    "theory_algebraic_rate",
    "theory_exponential_rate",
    "test_function",
    "smoothness_index",
    "smoothness_table",
]

#: 允许的基底名。
BASES: tuple[str, ...] = ("legendre", "chebyshev", "monomial")

#: 演示函数名（收敛率对照用）。
FUNCTION_NAMES: tuple[str, ...] = ("entire", "pole", "cusp", "jump")


# --------------------------------------------------------------------------- #
# 结果容器
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProjectionResult:
    """一次最小二乘投影的全部可观测物。"""

    coeffs: np.ndarray
    """在所选基底下的系数 :math:`c`（升幂或按基函数编号）。"""

    values: np.ndarray
    """投影在求积节点上的取值 :math:`(P_nf)(x_i)`。"""

    nodes: np.ndarray
    """求积节点（= 做离散内积的地方）。"""

    weights: np.ndarray
    """求积权重（含区间缩放）。"""

    l2_error: float
    r"""带权 :math:`L^2` 误差 :math:`\|f-P_nf\|_{L^2_w}`（**实测**，由复合 Gauss 求积给出）。"""

    linf_error: float
    r""":math:`\max_i|f(x_i)-(P_nf)(x_i)|`（节点上的离散 :math:`L^\infty` 误差）。"""

    gram: np.ndarray
    """Gram 矩阵 :math:`G`（正规方程的左端）。"""

    gram_cond: float
    r""":math:`\kappa_1(G)` 的**自研**估计（LU 求逆列）——正交基下 ≈1，单项式基下会爆掉。"""

    normal_residual: float
    """正规方程本身的相对残差 :math:`\\|Gc-b\\|_\\infty/\\|b\\|_\\infty`。"""

    n_terms: int
    """维数 :math:`n+1`。"""


@dataclass(frozen=True)
class RemezResult:
    """一次 Remez 交换算法的结果。"""

    coeffs: np.ndarray
    """Chebyshev 基下的系数 :math:`c_0,\\dots,c_n`（用 Chebyshev 基做交换方程，条件数远好于单项式基）。"""

    x_grid: np.ndarray
    """误差曲线的采样网格。"""

    error: np.ndarray
    """最终误差曲线 :math:`f-p`（带符号：交错性质看的就是它的符号）。"""

    alt_x: np.ndarray
    """最终交错点（应恰好 :math:`n+2` 个）。"""

    alt_error: np.ndarray
    """交错点上的误差（符号交替）。"""

    max_error: float
    r""":math:`\|f-p\|_\infty`（网格上的实测最大值）。"""

    iterations: int
    """实际迭代次数。"""

    converged: bool
    """交错点集合是否稳定（不变）。"""

    history: tuple[np.ndarray, ...]
    """每次迭代的误差曲线（时间轴动画用）。"""

    alt_history: tuple[np.ndarray, ...]
    """每次迭代使用的交错点（动画里的闪烁标记）。"""

    coeff_history: tuple[np.ndarray, ...] = ()
    """每次迭代的多项式系数（与 ``history`` 一一对应，供动画逐帧复原误差曲线）。"""

    @property
    def n_alt(self) -> int:
        """交错点个数（理论值 :math:`n+2`）。"""
        return int(self.alt_x.size)

    @property
    def amplitude_ratio(self) -> float:
        r"""交错点上的振幅比 :math:`\max|e|/\min|e|`（收敛时 → 1）。"""
        a = np.abs(self.alt_error)
        if a.size == 0 or float(np.min(a)) == 0.0:
            return float("inf")
        return float(np.max(a) / np.min(a))

    @property
    def sign_alternates(self) -> bool:
        """交错点上误差符号是否严格交替。"""
        s = np.sign(self.alt_error)
        return bool(s.size >= 2 and np.all(s[1:] * s[:-1] < 0.0))

    def evaluate(self, x: Any) -> np.ndarray:
        """用 Clenshaw 算法在 ``x`` 上求值（Chebyshev 系数）。"""
        return clenshaw(self.coeffs, x)

    def monomial_coeffs(self) -> np.ndarray:
        """转成单项式（升幂）系数——Deg ≤ 16 时转换本身是良态的，再高就不该转了。"""
        return chebyshev_to_monomial(self.coeffs)


# --------------------------------------------------------------------------- #
# 基底与线性代数
# --------------------------------------------------------------------------- #
def basis_values(basis: str, n_terms: int, x: Any) -> np.ndarray:
    """基底前 ``n_terms`` 个函数在 ``x`` 上的值，形状 ``(n_terms, x.size)``。

    ``legendre`` / ``chebyshev`` 给的是**正交归一**版本（在各自权重下）；
    ``monomial`` 给单项式 :math:`1,x,x^2,\\dots`。
    注意：本章的投影用的是 **Lebesgue 测度**（权恒为 1），所以只有 Legendre 基是正交的；
    Chebyshev 基与单项式基都只是"另一组基"——它们张成同一个子空间，
    投影结果（作为函数）相同，**但正规方程的条件数完全不同**，这正是可视化 1 要展示的。
    """
    n_terms = int(n_terms)
    if n_terms < 1:
        raise ValueError(f"n_terms 必须 ≥ 1，得到 {n_terms}")
    if basis == "monomial":
        return monomial_basis(n_terms - 1, x)
    if basis in ("legendre", "chebyshev"):
        return orthonormal_values(basis, n_terms - 1, x)
    raise ValueError(f"未知基底 {basis!r}（可用 {list(BASES)}）")


def weighted_norm(values: Any, weights: Any) -> float:
    r""":math:`\sqrt{\sum_i w_i v_i^2}`（离散带权 :math:`L^2` 范数）。"""
    v = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if v.size != w.size:
        raise ValueError(f"函数值个数 {v.size} 与权重个数 {w.size} 不一致")
    return math.sqrt(max(0.0, float(np.sum(w * v * v))))


def solve_linear_system(A: Any, b: Any) -> np.ndarray:
    """自研 LU + 回代解 :math:`Ax=b`（**不调用** ``numpy.linalg.solve``，P2/P3）。"""
    arr = np.asarray(A, dtype=float)
    rhs = np.asarray(b, dtype=float).reshape(-1)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"需要方阵，得到形状 {arr.shape}")
    if arr.shape[0] != rhs.size:
        raise ValueError(f"形状不匹配：A{arr.shape} b{rhs.shape}")
    res = lu_decompose(arr, pivot="partial")
    if res.singular:
        raise ValueError("矩阵奇异（LU 遇到零主元），无法求解")
    return solve_lu(res, rhs)


def gram_condition_estimate(gram: Any) -> float:
    r""":math:`\kappa_1(G)` 的自研估计（LU 求逆列；奇异时返回 ``inf``）。

    为什么用 1-范数而不是 2-范数：2-范数要 SVD（P2 禁止），而 1-范数的估计只需要解
    :math:`G y_j=e_j`——正好是本章已经自研的那条路。代价是它**本身**在病态时不可靠，
    所以它只用来展示"条件数被平方放大"的量级，不当真值。
    """
    try:
        return float(condition_estimate_1(np.asarray(gram, dtype=float)))
    except (ZeroDivisionError, ValueError):
        return float("inf")


def composite_gauss_legendre(
    a: float,
    b: float,
    *,
    panels: int = 1,
    nodes_per_panel: int = 40,
    breakpoints: Sequence[float] = (),
) -> tuple[np.ndarray, np.ndarray]:
    r"""复合 Gauss--Legendre 求积：把 :math:`[a,b]` 按断点切段，每段一套 Gauss 节点。

    为什么需要它：Gauss 求积的高精度建立在"被积函数光滑"上。对 :math:`|x-a|` 或
    :math:`\operatorname{sign}(x-a)` 这种被积函数，**不在折点处切段**就会让求积误差
    （~1e-3 量级）盖过我们想测的逼近误差（~1e-3 到 1e-6 量级）——那时"误差曲线"
    画的是求积的错，不是逼近的错。断点切段后每一段都光滑，Gauss 重新恢复谱精度。

    返回 ``(nodes, weights)``，权重已含每段的 Jacobian（于是 :math:`\sum w = b-a`）。
    """
    lo, hi = float(a), float(b)
    if not lo < hi:
        raise ValueError(f"需要 a < b，得到 [{lo}, {hi}]")
    if int(panels) < 1 or int(nodes_per_panel) < 1:
        raise ValueError("panels 与 nodes_per_panel 都必须 ≥ 1")
    cuts = [lo]
    for bp in sorted(float(t) for t in breakpoints):
        if lo < bp < hi:
            cuts.append(bp)
    cuts.append(hi)
    ref_nodes, ref_weights = gauss_legendre(int(nodes_per_panel))
    nodes: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for c0, c1 in zip(cuts[:-1], cuts[1:]):
        # 每个断点段再等分成 panels 个子面板（光滑段用 1 个就够；断点附近的振荡靠它压住）
        edges = np.linspace(c0, c1, int(panels) + 1)
        for e0, e1 in zip(edges[:-1], edges[1:]):
            half = 0.5 * (e1 - e0)
            mid = 0.5 * (e1 + e0)
            nodes.append(mid + half * ref_nodes)
            weights.append(half * ref_weights)
    return np.concatenate(nodes), np.concatenate(weights)


# --------------------------------------------------------------------------- #
# 最小二乘投影
# --------------------------------------------------------------------------- #
def project_onto_basis(
    f_vals: Any,
    basis_mat: Any,
    weights: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    r"""解正规方程 :math:`Gc=b`，返回 ``(coeffs, projection, gram, normal_residual)``。

    .. math::
        G_{ij}=\sum_k w_k\varphi_i(x_k)\varphi_j(x_k),\qquad
        b_i=\sum_k w_k f(x_k)\varphi_i(x_k).

    用的是**手写正规方程 + 自研 LU**；正规方程的经典缺陷（条件数是基底条件数的平方）
    由返回值里的 ``gram`` 如实暴露，调用方可自行估计 :math:`\kappa(G)`。
    """
    f = np.asarray(f_vals, dtype=float).reshape(-1)
    bmat = np.asarray(basis_mat, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if bmat.ndim != 2 or bmat.shape[1] != f.size or w.size != f.size:
        raise ValueError(f"形状不匹配：basis{bmat.shape} f{f.shape} w{w.shape}")
    gram = gram_matrix(bmat, w)
    rhs = bmat @ (w * f)
    coeffs = solve_linear_system(gram, rhs)
    proj = coeffs @ bmat
    scale = float(np.max(np.abs(rhs))) if rhs.size else 0.0
    res = float(np.max(np.abs(gram @ coeffs - rhs)))
    return coeffs, proj, gram, (res / scale if scale > 0.0 else 0.0)


def l2_projection(
    f: Callable[[np.ndarray], np.ndarray],
    n_terms: int,
    *,
    basis: str = "legendre",
    a: float = -1.0,
    b: float = 1.0,
    n_quad: int | None = None,
    n_panels: int = 1,
    breakpoints: Sequence[float] = (),
) -> ProjectionResult:
    r"""把 :math:`f` 在 Lebesgue 测度下投影到 :math:`\Phi_{n-1}`（维数 ``n_terms``）。

    ``breakpoints`` 给被积函数的折点/跳点（例如 :math:`|x-x_0|` 的 :math:`x_0`）：
    求积会在那里切段（见 :func:`composite_gauss_legendre`）。
    """
    n_terms = int(n_terms)
    if n_terms < 1:
        raise ValueError(f"n_terms 必须 ≥ 1，得到 {n_terms}")
    if n_quad is None:
        # 光滑被积函数的 Gauss 求积误差随节点数指数下降：4n+16 个节点已经远低于舍入地板。
        n_quad = int(4 * n_terms + 16)
    nodes, weights = composite_gauss_legendre(
        a, b, panels=int(n_panels), nodes_per_panel=int(n_quad), breakpoints=breakpoints
    )
    bmat = basis_values(basis, n_terms, nodes)
    f_vals = np.asarray(f(nodes), dtype=float).reshape(-1)
    coeffs, proj, gram, nres = project_onto_basis(f_vals, bmat, weights)
    return ProjectionResult(
        coeffs=coeffs,
        values=proj,
        nodes=nodes,
        weights=weights,
        l2_error=weighted_norm(f_vals - proj, weights),
        linf_error=float(np.max(np.abs(f_vals - proj))),
        gram=gram,
        gram_cond=gram_condition_estimate(gram),
        normal_residual=nres,
        n_terms=n_terms,
    )


def projection_error_curve(
    f: Callable[[np.ndarray], np.ndarray],
    ns: Sequence[int],
    *,
    basis: str = "legendre",
    a: float = -1.0,
    b: float = 1.0,
    n_quad: int | None = None,
    breakpoints: Sequence[float] = (),
) -> np.ndarray:
    r"""维数 :math:`n=1,2,\dots` 上的 :math:`L^2` 投影误差曲线（实测，逐点真算）。

    ``n_quad`` 显式给一个**偏小**的求积阶数时，可以亲手看到"不切段 + 低阶求积"如何把
    误差曲线压成一条假的水平线（那时量到的是求积误差，不是逼近误差）。
    """
    out = np.empty(len(ns), dtype=float)
    for i, n in enumerate(ns):
        res = l2_projection(f, int(n), basis=basis, a=a, b=b, n_quad=n_quad, breakpoints=breakpoints)
        out[i] = res.l2_error
    return out


def projector_matrix(basis_mat: Any, weights: Any) -> np.ndarray:
    r"""离散投影算子 :math:`P=B^{\mathsf T}G^{-1}B\,W`（**行**=基函数，:math:`G=BWB^{\mathsf T}`）。

    :math:`G^{-1}B^{\mathsf T}W` 用自研 LU 逐列回代求出（等价于求逆，但只解必要的右端项）。
    这个矩阵是"投影"这件事的代数化身，B 类断言直接查 :math:`P^2=P`。
    """
    bmat = np.asarray(basis_mat, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if bmat.ndim != 2 or bmat.shape[1] != w.size:
        raise ValueError(f"形状不匹配：basis{bmat.shape} w{w.shape}")
    gram = gram_matrix(bmat, w)
    m = gram.shape[0]
    rhs = bmat * w  # (m, n_nodes) = Bᵀ W 的转置形式
    # 逐列解 G y_j = rhs[:, j]
    y = np.empty((m, bmat.shape[1]), dtype=float)
    lu = lu_decompose(gram, pivot="partial")
    if lu.singular:
        raise ValueError("Gram 矩阵奇异：投影算子不存在")
    for j in range(bmat.shape[1]):
        y[:, j] = solve_lu(lu, rhs[:, j])
    return bmat.T @ y


def idempotence_defect(projector: Any) -> float:
    r""":math:`\max|P^2-P|`（投影算子幂等性的实测偏差）。"""
    p = np.asarray(projector, dtype=float)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError(f"投影算子必须是方阵，得到 {p.shape}")
    return float(np.max(np.abs(p @ p - p)))


# --------------------------------------------------------------------------- #
# Chebyshev 求值与基变换
# --------------------------------------------------------------------------- #
def clenshaw(coeffs: Sequence[float], x: Any) -> np.ndarray:
    r"""Clenshaw 算法求 :math:`\sum_k c_kT_k(x)`（Chebyshev 基的 Horner）。

    .. math::
        b_{n+1}=b_{n+2}=0,\qquad b_k=2xb_{k+1}-b_{k+2}+c_k,\qquad p=b_0-c_2\cdot 0-xb_1 .
    """
    c = np.asarray(coeffs, dtype=float).reshape(-1)
    if c.size == 0:
        raise ValueError("系数不能为空")
    xx = np.asarray(x, dtype=float)
    b1 = np.zeros_like(xx, dtype=float)
    b2 = np.zeros_like(xx, dtype=float)
    for k in range(c.size - 1, 0, -1):
        b1, b2 = 2.0 * xx * b1 - b2 + c[k], b1
    return xx * b1 - b2 + c[0]


def chebyshev_to_monomial(coeffs: Sequence[float]) -> np.ndarray:
    r"""Chebyshev 系数 → 单项式（升幂）系数，用 :math:`T_{k+1}=2xT_k-T_{k-1}` 精确递推。

    :math:`T_k` 的系数是整数，递推是**精确**的（浮点只承担 2 的幂与加减），
    所以这个转换在 Deg ≤ 20 时不引入可观误差；但"用单项式系数表示高次多项式"本身
    在求值时仍会失去条件数（见 ``chapters/2.3/pitfalls.md``）。
    """
    c = np.asarray(coeffs, dtype=float).reshape(-1)
    if c.size == 0:
        raise ValueError("系数不能为空")
    n = c.size - 1
    # t_prev = T_{k-1}, t_cur = T_k（升幂系数向量）
    t_prev = np.zeros(n + 1, dtype=float)
    t_cur = np.zeros(n + 1, dtype=float)
    t_prev[0] = 1.0
    if n >= 1:
        t_cur[1] = 1.0
    out = c[0] * t_prev.copy()
    if n >= 1:
        out += c[1] * t_cur
    for k in range(2, n + 1):
        nxt = np.zeros(n + 1, dtype=float)
        nxt[1:] += 2.0 * t_cur[:-1]
        nxt -= t_prev
        t_prev, t_cur = t_cur, nxt
        out += c[k] * t_cur
    return out


# --------------------------------------------------------------------------- #
# Remez 交换算法
# --------------------------------------------------------------------------- #
def _refine_one(x0: float, x1: float, x2: float, e0: float, e1: float, e2: float) -> tuple[float, float]:
    """三点抛物线插值找极值（返回 ``(x*, e*)``）；退化为端点时返回中间点。"""
    denom = e0 - 2.0 * e1 + e2
    if denom == 0.0:
        return x1, e1
    shift = 0.5 * (e0 - e2) / denom  # 以格距为单位
    if not -1.0 < shift < 1.0:
        return x1, e1
    h = x1 - x0 if x1 != x0 else (x2 - x1)
    xstar = x1 + shift * h
    estar = e1 - 0.25 * (e0 - e2) * shift
    return xstar, estar


def _golden_max(
    err_at: Callable[[np.ndarray], np.ndarray],
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    iters: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    r"""**向量化**黄金分割：在每个区间 :math:`[lo_i,hi_i]` 上极大化 :math:`|e(x)|`。

    为什么抛物线插值不够：抛物线在**折点**（:math:`|x-a|` 的尖点）与**跳点**
    （:math:`\operatorname{sign}(x-a)`）附近会给出错误的极值位置与偏小的幅度，
    于是 de la Vallée Poussin 下界永远追不上实测上界，Remez 会**空转到 max_iter**。
    黄金分割只用 :math:`|e|` 的取值，对折点/跳点都成立；把所有候选点一起算，
    几十次迭代在这里只是一次向量化调用。
    """
    a = np.asarray(lo, dtype=float).copy()
    b = np.asarray(hi, dtype=float).copy()
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc = err_at(c)
    fd = err_at(d)
    for _ in range(int(iters)):
        left = np.abs(fc) >= np.abs(fd)
        a = np.where(left, a, c)
        b = np.where(left, d, b)
        c = b - invphi * (b - a)
        d = a + invphi * (b - a)
        fc = err_at(c)
        fd = err_at(d)
    take_c = np.abs(fc) >= np.abs(fd)
    return np.where(take_c, c, d), np.where(take_c, fc, fd)


def refine_extrema(
    x: Any,
    e: Any,
    *,
    err: Callable[[np.ndarray], np.ndarray] | None = None,
    golden_iters: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """找出 ``e`` 在网格 ``x`` 上的全部局部极值并细化，返回 ``(x*, e*)``（按 x 升序）。

    网格上的"局部极值点"只用于定位；真正的极值位置必须细化，否则 Remez 的收敛速度会退化成线性。
    给了 ``err``（在任意点求误差的可调用对象）时用 :func:`_golden_max`（对折点/跳点稳健），
    否则退回抛物线插值 :func:`_refine_one`（只对光滑函数正确）。
    """
    xx = np.asarray(x, dtype=float).reshape(-1)
    ee = np.asarray(e, dtype=float).reshape(-1)
    if xx.size != ee.size or xx.size < 3:
        raise ValueError("x 与 e 必须等长且至少 3 个点")
    idx: list[int] = []
    for i in range(1, xx.size - 1):
        left = ee[i] - ee[i - 1]
        right = ee[i + 1] - ee[i]
        if left == 0.0 and right == 0.0:
            continue
        if left * right <= 0.0:
            idx.append(i)
    picks: list[tuple[float, float]] = [(float(xx[0]), float(ee[0]))]
    if idx and err is not None:
        ii = np.array(idx, dtype=int)
        xs, es = _golden_max(err, xx[ii - 1], xx[ii + 1], iters=int(golden_iters))
        picks.extend((float(u), float(v)) for u, v in zip(xs, es))
    else:
        for i in idx:
            picks.append(
                _refine_one(
                    float(xx[i - 1]), float(xx[i]), float(xx[i + 1]),
                    float(ee[i - 1]), float(ee[i]), float(ee[i + 1]),
                )
            )
    picks.append((float(xx[-1]), float(ee[-1])))
    xs_arr = np.array([p[0] for p in picks], dtype=float)
    es_arr = np.array([p[1] for p in picks], dtype=float)
    order = np.argsort(xs_arr)
    return xs_arr[order], es_arr[order]


def _alternating_subset(xs: np.ndarray, es: np.ndarray, count: int) -> np.ndarray:
    """从候选极值里挑出 ``count`` 个符号交替的点（返回**索引**数组）。

    这是 Remez 的"换点"规则，也是最容易写错的一步。这里用的是
    **de la Vallée Poussin 极大化最小振幅**的精确解：

    1. 先做同号剪枝（相邻同号只留 |e| 大的），得到一个符号严格交替的序列——它的任意
       子序列仍然交替，于是"选 count 个交替点"变成"在交替序列里选 count 个"；
    2. 用动态规划选出使 :math:`\\min|e|` 最大的那 count 个：
       :math:`\\mathrm{best}[i][t]=\\max_{j<i}\\min(\\mathrm{best}[j][t-1],|e_i|)`。

    为什么不能"反复丢掉 |e| 最小的那个"：丢掉一个内部点会让它的左右邻居变成同号相邻，
    符号交替被破坏，交错方程随即**奇异**（实测：E 掉到 1e-16、误差曲线乱跳、算法假收敛）。
    候选不足 ``count`` 个时返回空数组，让调用方保留旧集合。
    """
    m = xs.size
    if m < count:
        return np.zeros(0, dtype=int)
    keep: list[int] = []
    for i in range(m):
        s = float(np.sign(es[i]))
        if s == 0.0:  # 恰好为 0 的极值：按"与上一个相反"处理，避免破坏交替
            s = -float(np.sign(es[keep[-1]])) if keep else 1.0
        if keep and s == float(np.sign(es[keep[-1]])):
            if abs(es[i]) > abs(es[keep[-1]]):
                keep[-1] = i
        else:
            keep.append(i)
    if len(keep) < count:
        return np.zeros(0, dtype=int)
    if len(keep) > 3 * count:  # 控制 DP 规模：交替序列的任意子序列依然交替，可以安全截断
        order = sorted(range(len(keep)), key=lambda j: -abs(es[keep[j]]))[: 3 * count]
        keep = sorted(keep[j] for j in order)
    n = len(keep)
    y = np.abs(es[keep])
    neg = float("-inf")
    best = np.full((n, count), neg, dtype=float)
    par = np.full((n, count), -1, dtype=int)
    for i in range(n):
        best[i, 0] = y[i]
    for t in range(1, count):
        for i in range(1, n):
            arg = -1
            bestv = neg
            for j in range(i):
                if best[j, t - 1] == neg:
                    continue
                v = min(best[j, t - 1], y[i])
                if v > bestv:
                    bestv, arg = v, j
            best[i, t] = bestv
            par[i, t] = arg
    end = int(np.argmax(best[:, count - 1]))
    if best[end, count - 1] == neg:
        return np.zeros(0, dtype=int)
    chain: list[int] = []
    i, t = end, count - 1
    while t >= 0 and i >= 0:
        chain.append(keep[i])
        i = int(par[i, t])
        t -= 1
    if len(chain) != count:
        return np.zeros(0, dtype=int)
    return np.array(sorted(chain), dtype=int)


def alternation_ratio(err_at_points: Any) -> float:
    r""":math:`\max|e|/\min|e|`（交错点上的振幅比，收敛时 → 1）。"""
    a = np.abs(np.asarray(err_at_points, dtype=float).reshape(-1))
    if a.size == 0 or float(np.min(a)) == 0.0:
        return float("inf")
    return float(np.max(a) / np.min(a))


def remez(
    f: Callable[[np.ndarray], np.ndarray],
    deg: int,
    *,
    a: float = -1.0,
    b: float = 1.0,
    n_grid: int = 2001,
    max_iter: int = 40,
    certificate_tol: float = 1e-9,
) -> RemezResult:
    r"""Remez 交换算法：求 :math:`\Phi_n` 中 :math:`\|f-p\|_\infty` 最小的多项式。

    每一轮解**交错方程**

    .. math::
        \sum_{k=0}^{n}c_kT_k(x_j)+\sigma_jE=f(x_j),\qquad \sigma_j=(-1)^j,\ j=0,\dots,n+1,

    用 Chebyshev 基（而不是单项式基）写这组方程：交错点接近 Chebyshev 极点，
    此时 Chebyshev 基的系统条件数几乎与 n 无关，而单项式基的 Vandermonde 会立刻病态。
    解出 :math:`c,E` 后在细网格上找 :math:`f-p` 的极值，用
    :func:`_alternating_subset` 换点。

    **停机判据不是"点不动了"**，而是 de la Vallée Poussin 的**最优性证书**：

    .. math::
        h:=\min_{x_j\in X}|f(x_j)-p(x_j)|\ \ge\ (1-\varepsilon)\max_{[a,b]}|f-p|
        \quad\Longrightarrow\quad \|f-p\|_\infty\le\frac{h}{1-\varepsilon}\le\frac{\|f-p^*\|_\infty}{1-\varepsilon},

    即"交错点上的最小振幅"追上了"实测最大误差"时，当前多项式就是最优的。
    只测"点集不变"会在浮点噪声里**假收敛**：实测中交换点会在两三个集合之间循环，
    其中既有最优的那个、也有误差大一倍的（见 ``chapters/2.3/pitfalls.md``）。
    因此这里同时记录**历史最优迭代**，循环结束时返回误差最小的那一个。

    初值取 :math:`T_{n+1}` 的极值点 :math:`\cos(j\pi/(n+1))`：它们天然符号交替，
    是 Remez 的标准起点；**起点选得不好会不收敛**。
    """
    deg = int(deg)
    if deg < 0:
        raise ValueError(f"次数必须 ≥ 0，得到 {deg}")
    lo, hi = float(a), float(b)
    if not lo < hi:
        raise ValueError(f"需要 a < b，得到 [{lo}, {hi}]")
    n_alt = deg + 2
    x_grid = np.linspace(lo, hi, int(n_grid))
    f_grid = np.asarray(f(x_grid), dtype=float).reshape(-1)
    # 误差求值的舍入尺度：证书的绝对判据要用它（见下面的注释）
    fscale = max(1.0, float(np.max(np.abs(f_grid))))

    j = np.arange(n_alt, dtype=float)
    x_alt = 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos(j * np.pi / (deg + 1))
    x_alt = np.sort(x_alt)
    signs = np.where(np.arange(n_alt) % 2 == 0, 1.0, -1.0)

    history: list[np.ndarray] = []
    alt_history: list[np.ndarray] = []
    coeff_history: list[np.ndarray] = []
    converged = False
    iterations = 0
    best: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None = None
    for it in range(1, int(max_iter) + 1):
        iterations = it
        design = np.empty((n_alt, deg + 2), dtype=float)
        design[:, : deg + 1] = clenshaw_basis(deg, x_alt)
        design[:, deg + 1] = signs
        rhs = np.asarray(f(x_alt), dtype=float).reshape(-1)
        sol = solve_linear_system(design, rhs)
        coeffs = sol[: deg + 1]
        err = f_grid - clenshaw(coeffs, x_grid)
        history.append(err.copy())
        alt_history.append(x_alt.copy())
        coeff_history.append(coeffs.copy())
        e_alt = rhs - clenshaw(coeffs, x_alt)
        h = float(np.min(np.abs(e_alt)))
        max_e = float(np.max(np.abs(err)))
        if best is None or max_e < best[4]:
            best = (coeffs.copy(), err.copy(), x_alt.copy(), e_alt.copy(), max_e)
        # 最优性证书。两条判据是同一件事的两个刻度：
        #   (1) 相对形式：下界追平实测上界（误差远高于求值舍入时用它）；
        #   (2) 绝对形式：两者之差已经在**误差求值的舍入水平** 8·u·‖f‖∞ 之内。
        # 缺了 (2)，误差掉到 1e-11 以下的整函数（如 eˣ, n≥10）永远拿不到证书，
        # 算法会空转到 max_iter——那时"没收敛"其实只是"已经算不动了"。
        rel_ok = h >= (1.0 - float(certificate_tol)) * max_e
        abs_ok = (max_e - h) <= 8.0 * ROUNDOFF * fscale
        if rel_ok or abs_ok:
            converged = True
            break
        cand_x, cand_e = refine_extrema(x_grid, err, err=lambda t: np.asarray(f(t), dtype=float).reshape(-1) - clenshaw(coeffs, t))
        pick = _alternating_subset(cand_x, cand_e, n_alt)
        if pick.size != n_alt:
            # 候选极值不足 n+2 个（例如误差曲线太平）：这一轮不算收敛，保留旧集合
            converged = False
            break
        new_x = np.sort(cand_x[pick])
        x_alt = new_x

    if best is None:  # pragma: no cover - max_iter ≥ 1 时不可能发生
        raise ValueError("Remez 一次迭代都没有执行（max_iter 必须 ≥ 1）")
    b_coeffs, b_err, b_alt, b_e_alt, b_max = best
    if not history or float(np.max(np.abs(history[-1] - b_err))) > 0.0:
        history.append(b_err.copy())
        alt_history.append(b_alt.copy())
        coeff_history.append(b_coeffs.copy())
    return RemezResult(
        coeffs=b_coeffs,
        x_grid=x_grid,
        error=b_err,
        alt_x=b_alt,
        alt_error=b_e_alt,
        max_error=b_max,
        iterations=iterations,
        converged=bool(converged),
        history=tuple(history),
        alt_history=tuple(alt_history),
        coeff_history=tuple(coeff_history),
    )


def clenshaw_basis(deg: int, x: Any) -> np.ndarray:
    """``T_0..T_deg`` 在 ``x`` 上的值，形状 ``(x.size, deg+1)``（Remez 方程的设计矩阵）。"""
    deg = int(deg)
    if deg < 0:
        raise ValueError("deg 必须 ≥ 0")
    xx = np.asarray(x, dtype=float).reshape(-1)
    out = np.empty((xx.size, deg + 1), dtype=float)
    out[:, 0] = 1.0
    if deg >= 1:
        out[:, 1] = xx
    for k in range(1, deg):
        out[:, k + 1] = 2.0 * xx * out[:, k] - out[:, k - 1]
    return out


# --------------------------------------------------------------------------- #
# 衰减率：实测拟合与理论值
# --------------------------------------------------------------------------- #
def _least_squares_slope(u: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """一元线性最小二乘 ``v = s·u + t``，返回 ``(s, R²)``（2×2 正规方程手写）。"""
    mu = float(np.mean(u))
    mv = float(np.mean(v))
    du = u - mu
    dv = v - mv
    denom = float(np.sum(du * du))
    if denom == 0.0:
        return 0.0, 0.0
    slope = float(np.sum(du * dv) / denom)
    inter = mv - slope * mu
    resid = dv - slope * du
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum(dv * dv))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return slope, r2


def fit_loglog_slope(ns: Any, errs: Any, *, skip: int = 0) -> tuple[float, float]:
    r"""双对数斜率 :math:`\mathrm d\log e/\mathrm d\log n` 与 :math:`R^2`。

    :math:`e\sim Cn^{-\alpha}` 时它就是 :math:`-\alpha`。``skip`` 丢掉前若干个点
    （粗端还没进入渐近区时用；**细端**的舍入地板不能用 skip 处理，得靠调用方截断）。
    """
    n = np.asarray(ns, dtype=float).reshape(-1)[int(skip):]
    e = np.asarray(errs, dtype=float).reshape(-1)[int(skip):]
    if n.size < 2:
        raise ValueError("至少需要 2 个点")
    if np.any(n <= 0.0) or np.any(e <= 0.0):
        raise ValueError("log-log 拟合要求 n>0 且 e>0（0 要先按地板截断）")
    return _least_squares_slope(np.log(n), np.log(e))


def fit_semilog_slope(ns: Any, errs: Any, *, skip: int = 0) -> tuple[float, float]:
    r"""半对数斜率 :math:`\mathrm d\ln e/\mathrm d n` 与 :math:`R^2`。

    :math:`e\sim C\rho^{-n}`（解析函数、Bernstein 椭圆参数 :math:`\rho`）时它是 :math:`-\ln\rho`；
    若"看起来线性"但 :math:`R^2` 明显小于 1，那多半是**超几何**（整函数）收敛，
    不是指数收敛——这正是"log 轴上看起来直"这条陷阱的判据。
    """
    n = np.asarray(ns, dtype=float).reshape(-1)[int(skip):]
    e = np.asarray(errs, dtype=float).reshape(-1)[int(skip):]
    if n.size < 2:
        raise ValueError("至少需要 2 个点")
    if np.any(e <= 0.0):
        raise ValueError("半对数拟合要求 e>0（0 要先按地板截断）")
    return _least_squares_slope(n, np.log(e))


def bernstein_rho(z: complex) -> float:
    r"""Bernstein 椭圆参数 :math:`\rho=|z+\sqrt{z^2-1}|`（:math:`z` 是最近的奇点）。

    :math:`f` 在 :math:`E_\rho` 内解析时系数按 :math:`\rho^{-k}` 衰减。
    实轴上区间 :math:`[-1,1]`：若奇点在 :math:`z=iy`，则 :math:`\rho=|y|+\sqrt{1+y^2}`。
    """
    zz = complex(z)
    root = cmath.sqrt(zz * zz - 1.0)
    return abs(zz + root)


def theory_algebraic_rate(nu: float) -> float:
    r"""代数率的理论指数 :math:`\alpha=\nu+\tfrac12`。

    :math:`\nu` 的定义：:math:`f` 有 :math:`\nu` 阶连续导数，且
    :math:`f^{(\nu)}` 有界变差（于是 :math:`c_k=O(k^{-\nu-1})`，
    而 :math:`\|f-P_nf\|^2=\sum_{k>n}c_k^2=O(n^{-2\nu-1})`）。
    折点 :math:`|x-a|`：:math:`\nu=1\Rightarrow\alpha=3/2`；
    跳点 :math:`\operatorname{sign}(x-a)`：:math:`\nu=0\Rightarrow\alpha=1/2`。
    """
    v = float(nu)
    if v < 0.0:
        raise ValueError(f"nu 必须 ≥ 0，得到 {v}")
    return v + 0.5


def theory_exponential_rate(z: complex | None) -> float:
    r"""指数率的理论斜率 :math:`-\ln\rho`（:math:`z=\\text{None}` 表示整函数，返回 ``-inf``）。"""
    if z is None:
        return float("-inf")
    rho = bernstein_rho(z)
    if rho <= 1.0:
        raise ValueError(f"奇点在区间内（rho={rho}），没有收敛可言")
    return -math.log(rho)


# --------------------------------------------------------------------------- #
# 演示函数与光滑性对照表
# --------------------------------------------------------------------------- #
def test_function(name: str, *, x0: float = 0.0, scale: float = 25.0) -> Callable[[np.ndarray], np.ndarray]:
    """演示函数工厂（收敛率可视化与断言共用同一份真源）。

    ==========  ==========================================  ==========================
    名字        表达式                                      光滑性
    ==========  ==========================================  ==========================
    ``entire``  :math:`e^{x}`                                整函数（比任何指数都快）
    ``pole``    :math:`1/(1+s x^2)`                          解析，最近奇点 :math:`\\pm i/\\sqrt{s}`
    ``cusp``    :math:`|x-x_0|`                              :math:`f'` 有跳（:math:`\\nu=1`）
    ``jump``    :math:`\\operatorname{sign}(x-x_0)`          本身有跳（:math:`\\nu=0`）
    ==========  ==========================================  ==========================
    """
    if name == "entire":
        return lambda x: np.exp(np.asarray(x, dtype=float))
    if name == "pole":
        s = float(scale)
        return lambda x: 1.0 / (1.0 + s * np.asarray(x, dtype=float) ** 2)
    if name == "cusp":
        xa = float(x0)
        return lambda x: np.abs(np.asarray(x, dtype=float) - xa)
    if name == "jump":
        xa = float(x0)
        return lambda x: np.where(np.asarray(x, dtype=float) < xa, -1.0, 1.0)
    raise ValueError(f"未知演示函数 {name!r}（可用 {list(FUNCTION_NAMES)}）")


def smoothness_index(name: str) -> float:
    r""":math:`\nu`：第一个**不**连续的导数的阶（``cusp``→1、``jump``→0，解析/整函数 → ``inf``）。"""
    if name in ("cusp",):
        return 1.0
    if name in ("jump",):
        return 0.0
    if name in ("pole", "entire"):
        return float("inf")
    raise ValueError(f"未知演示函数 {name!r}")


def smoothness_table(*, scale: float = 25.0) -> list[dict[str, Any]]:
    """光滑性与理论衰减率的对照表（笔记与帧共用同一份真源）。"""
    s = float(scale)
    rho = bernstein_rho(complex(0.0, 1.0 / math.sqrt(s)))
    return [
        {"name": "entire", "nu": float("inf"), "kind": "super-geometric", "alpha": float("nan"), "rate": float("-inf"), "note": "整函数：比任何 rho^{-n} 都快，半对数图上并不直"},
        {"name": "pole", "nu": float("inf"), "kind": "geometric", "alpha": float("nan"), "rate": -math.log(rho), "note": f"Bernstein 椭圆 rho={rho:.6f}，误差 ~ rho^(-n)"},
        {"name": "cusp", "nu": 1.0, "kind": "algebraic", "alpha": theory_algebraic_rate(1.0), "rate": float("nan"), "note": "f' 跳变：系数 O(k^-2)，L2 误差 O(n^-3/2)"},
        {"name": "jump", "nu": 0.0, "kind": "algebraic", "alpha": theory_algebraic_rate(0.0), "rate": float("nan"), "note": "本身跳变：系数 O(k^-1)，L2 误差 O(n^-1/2)"},
    ]
