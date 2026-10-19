r"""多项式混沌展开（PCE）：正交基、投影与回归、谱衰减、**留出集**误差与外推。

这一层的四句话
--------------
1. **基是概率测度的一部分。** 均匀输入配 Legendre、正态输入配 Hermite；
   在 :math:`[-1,1]` 上对正态输入用 Legendre 基，谱收敛会退化成"看起来还行"的代数收敛。
   本模块的 :func:`legendre_values` / :func:`hermite_values` 都是**正交归一**的
   （:math:`\langle\phi_k,\phi_l\rangle=\delta_{kl}` 对相应测度），于是
   :func:`pce_moments` 里 :math:`\mathrm{Var}=\sum_{k>0}c_k^2` 这条恒等式才成立。
2. **系数有两条路**：投影（用 Gauss 求积，永远不过拟合）与回归（用最小二乘，
   样本不够时**会**过拟合）。:func:`pce_projection` 与 :func:`pce_regression` 并存，
   ``tests/5.4`` 在多項式上对拍两者——它们必须给出同一组系数。
3. **误差必须报留出集（outsample）。** 训练误差随阶数单调下降（最后到机器零），
   它**不能**用来说明代理模型好不好；:func:`error_vs_order` 同时给两条曲线，
   并让过拟合区（训练误差→0、留出误差→上升）显式出现在同一张图上。
4. **外推会失效，而且失效得很干脆。** :func:`extrapolation_error` 在训练区间之外
   报误差；对多项式混沌来说，"略微外推一下"没有任何数学保证——它不是插值问题，
   是基的支撑集问题。

自研的边界：Gauss–Legendre / Gauss–Hermite 节点由 Newton 迭代在递推式上求出
（P2/P3 禁止 ``numpy.linalg`` 与现成求积；本层也不 import :mod:`numcore.quad`，
依赖方向只允许 ``inverse → linalg/utils``），``tests/5.4`` 拿 2.4 章的
:func:`numcore.quad.gauss.gauss_legendre` 与本实现逐位对拍。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from numcore.linalg.eigen import hermitian_jacobi_eigh
from numcore.linalg.qr import qr_least_squares
from numcore.linalg.svd import jacobi_svd, truncated_solution
from numcore.utils import rng as make_rng

__all__ = [
    "MAX_ORDER",
    "multi_index",
    "n_basis",
    "legendre_values",
    "hermite_values",
    "gauss_legendre_rule",
    "gauss_hermite_rule",
    "basis_matrix",
    "design_samples",
    "pce_projection",
    "pce_regression",
    "PCEFit",
    "pce_evaluate",
    "pce_moments",
    "pce_sobol_indices",
    "relative_l2_error",
    "error_vs_order",
    "extrapolation_error",
    "coefficient_spectrum",
    "TARGETS",
    "target_names",
    "target_function",
    "target_label",
    "fit_target",
]

#: 帧里允许的最大阶数（(d+1)^p 的基规模随 p 增长，超过这个数就没有交互意义了）
MAX_ORDER = 14


# --------------------------------------------------------------------------- #
# 多指标与基
# --------------------------------------------------------------------------- #
def multi_index(dim: int, order: int) -> np.ndarray:
    """总阶数 ≤ ``order`` 的多指标集合，形状 ``(n_basis, dim)``，按总阶数→字典序排列。

    阶数 :math:`|\\alpha|=\\sum_j\\alpha_j\\le p`、维数 :math:`d` 时的基规模是
    :math:`\\binom{p+d}{p}`——**它随 p 与 d 一起爆炸**，这是 PCE 在高维失效的第一个原因。
    """
    d = int(dim)
    p = int(order)
    if d < 1 or p < 0:
        raise ValueError("dim >= 1，order >= 0")
    out: list[tuple[int, ...]] = []

    def rec(prefix: tuple[int, ...], remaining: int, slots: int) -> None:
        if slots == 1:
            for k in range(remaining + 1):  # 总阶数 <= p：最后一维取遍 0..remaining
                out.append(prefix + (k,))
            return
        for k in range(remaining + 1):
            rec(prefix + (k,), remaining - k, slots - 1)

    rec((), p, d)
    arr = np.array(out, dtype=np.int64)
    keys = np.lexsort(tuple(arr[:, j] for j in range(d - 1, -1, -1)))
    return arr[keys]


def n_basis(dim: int, order: int) -> int:
    """基规模 :math:`\\binom{p+d}{p}`（与 :func:`multi_index` 的行数一致）。"""
    d = int(dim)
    p = int(order)
    return int(math.comb(p + d, d))


def legendre_values(x: Any, order: int) -> np.ndarray:
    r"""正交归一的 Legendre 基 :math:`\phi_k(x)=\sqrt{2k+1}P_k(x)`，``x ∈ [-1, 1]``。

    归一化常数取 :math:`\sqrt{2k+1}` 是因为均匀测度 :math:`\\mathrm dx/2` 下
    :math:`\\int_{-1}^1P_k^2\\,\\mathrm dx/2=\\frac1{2k+1}`。
    """
    t = np.asarray(x, dtype=np.float64).ravel()
    p = int(order)
    if p < 0:
        raise ValueError("order >= 0")
    # 先按标准三项递推算**未归一**的 P_k，最后整体乘 sqrt(2k+1)：
    # 直接对 phi_k 写递推要同时跟踪两个不同的归一化因子，第一版就是在这里把
    # P_{k-2} 的因子写错，导致 k >= 3 的基完全不是 Legendre 多项式
    # （节点/权重看起来仍然"像那么回事"，正交性缺陷 1.3e+16）。
    P = np.empty((t.size, p + 1), dtype=np.float64)
    P[:, 0] = 1.0
    if p >= 1:
        P[:, 1] = t
    for k in range(2, p + 1):
        P[:, k] = ((2 * k - 1) * t * P[:, k - 1] - (k - 1) * P[:, k - 2]) / k
    scale = np.sqrt(2.0 * np.arange(p + 1, dtype=np.float64) + 1.0)
    return P * scale[None, :]


def hermite_values(x: Any, order: int) -> np.ndarray:
    r"""正交归一的 Hermite 基（概率论约定）:math:`\phi_k(x)=\mathrm{He}_k(x)/\sqrt{k!}`。

    正交性对标准正态测度成立：:math:`\int \mathrm{He}_k\mathrm{He}_l\,\varphi(x)\,\mathrm dx=k!\,\delta_{kl}`，
    于是 :math:`\phi_k` 是正交归一的。递推（对归一化函数直接写，避免阶乘溢出）：

    .. math::
        \phi_{k+1}=\frac{x\,\phi_k}{\sqrt{k+1}}-\sqrt{\frac{k}{k+1}}\,\phi_{k-1}.
    """
    t = np.asarray(x, dtype=np.float64).ravel()
    p = int(order)
    if p < 0:
        raise ValueError("order >= 0")
    out = np.empty((t.size, p + 1), dtype=np.float64)
    out[:, 0] = 1.0
    if p >= 1:
        out[:, 1] = t
    for k in range(1, p):
        out[:, k + 1] = t * out[:, k] / math.sqrt(k + 1.0) - math.sqrt(k / (k + 1.0)) * out[:, k - 1]
    return out


def basis_matrix(x: Any, mi: np.ndarray, *, kind: str = "legendre") -> np.ndarray:
    """张量积基矩阵 ``(n, n_basis)``：``Psi_{i,alpha} = prod_j phi_{alpha_j}(x_ij)``。"""
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim == 1:
        xx = xx[:, None]
    m = np.asarray(mi, dtype=np.int64)
    n, d = xx.shape
    if m.shape[1] != d:
        raise ValueError(f"多指标维数 {m.shape[1]} 与样本维数 {d} 不符")
    order = int(np.max(m)) if m.size else 0
    if kind not in ("legendre", "hermite"):
        raise ValueError(f"未知基 {kind!r}（可用 legendre / hermite）")
    one_d = legendre_values if kind == "legendre" else hermite_values
    vals = [one_d(xx[:, j], order) for j in range(d)]
    out = np.ones((n, m.shape[0]), dtype=np.float64)
    for k in range(m.shape[0]):
        for j in range(d):
            out[:, k] *= vals[j][:, m[k, j]]
    return out


def design_samples(dim: int, n: int, *, kind: str = "legendre", seed: Any = None) -> np.ndarray:
    """训练/留出样本：Legendre 用 ``U(-1,1)``，Hermite 用标准正态（Box–Muller 自研）。"""
    d = int(dim)
    g = make_rng(seed)
    if kind == "legendre":
        return 2.0 * g.random((int(n), d)) - 1.0
    if kind == "hermite":
        out = np.empty((int(n), d), dtype=np.float64)
        for j in range(d):
            u1 = g.random(int(n))
            u2 = g.random(int(n))
            r = np.sqrt(-2.0 * np.log(np.maximum(u1, np.finfo(float).tiny)))
            out[:, j] = r * np.cos(2.0 * np.pi * u2)
        return out
    raise ValueError(f"未知基 {kind!r}（可用 legendre / hermite）")


# --------------------------------------------------------------------------- #
# Gauss 求积（自研 Newton 迭代）
# --------------------------------------------------------------------------- #
def _gauss_from_jacobi(b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Golub–Welsch：由三项递推系数 :math:`b_k` 直接给出 Gauss 节点与权重。

    正交归一多项式满足 :math:`x\phi_k=b_k\phi_{k+1}+b_{k-1}\phi_{k-1}`（对称测度下
    :math:`a_k=0`），于是对称三对角 Jacobi 矩阵 :math:`J_n` 的特征值就是 Gauss 节点，
    第 :math:`i` 个特征向量的**第一个分量平方**就是权重
    （概率测度下 :math:`w_i=v_{i,1}^2`，因为测度总质量为 1）。

    用 :func:`numcore.linalg.eigen.hermitian_jacobi_eigh` 而不是"牛顿迭代 + 权重公式"：
    后者每换一个族就要重推一次归一化常数，本模块第一版正是这样把 Legendre 的权重
    少乘了测度总质量（和只有 1 而不是 2 的等价物），而节点看起来完全正确。
    """
    n = int(b.size) + 1
    J = np.zeros((n, n), dtype=np.float64)
    for k in range(n - 1):
        J[k, k + 1] = b[k]
        J[k + 1, k] = b[k]
    w, V = hermitian_jacobi_eigh(J)
    # Jacobi 矩阵实对称，特征值/特征向量在舍入意义下是实的（该例程返回 complex 容器）
    return np.asarray(w).real, np.asarray(V[0, :]).real ** 2


def gauss_legendre_rule(nq: int) -> tuple[np.ndarray, np.ndarray]:
    r"""``nq`` 点 Gauss–Legendre 节点与权重，**对均匀概率测度** :math:`\mathrm dx/2` 归一（权重和为 1）。

    递推系数 :math:`b_k=\frac{k+1}{\sqrt{(2k+1)(2k+3)}}`。
    ``tests/5.4`` 与 2.4 章的 :func:`numcore.quad.gauss.gauss_legendre` 对拍
    （那一支对 :math:`\mathrm dx` 归一，权重和 2，比值必须逐位等于 2）。
    """
    n = int(nq)
    if n < 1:
        raise ValueError("nq >= 1")
    k = np.arange(n - 1, dtype=np.float64)
    b = (k + 1.0) / np.sqrt((2.0 * k + 1.0) * (2.0 * k + 3.0))
    return _gauss_from_jacobi(b)


def gauss_hermite_rule(nq: int) -> tuple[np.ndarray, np.ndarray]:
    r"""``nq`` 点 Gauss–Hermite（概率论约定）：对标准正态测度归一（权重和为 1）。

    递推系数 :math:`b_k=\sqrt{k+1}`。三点的权重因此是 :math:`(1/6,2/3,1/6)`—— 
    ``tests/5.4`` 用这个闭式钉住归一化（本模块第一版在这里给的是 :math:`(1/12,\dots)`，
    只差一个因子 2，但它会让"正交性"和"方差恒等式"两条断言一起错）。
    """
    n = int(nq)
    if n < 1:
        raise ValueError("nq >= 1")
    k = np.arange(n - 1, dtype=np.float64)
    return _gauss_from_jacobi(np.sqrt(k + 1.0))


# --------------------------------------------------------------------------- #
# 系数：投影与回归
# --------------------------------------------------------------------------- #
def _tensor_nodes(dim: int, nq: int, *, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """张量积求积节点与权重（``nq^dim`` 个）。"""
    if kind == "legendre":
        nodes, weights = gauss_legendre_rule(nq)
    elif kind == "hermite":
        nodes, weights = gauss_hermite_rule(nq)
    else:
        raise ValueError(f"未知基 {kind!r}")
    mesh = np.meshgrid(*([nodes] * int(dim)), indexing="ij")
    wmesh = np.meshgrid(*([weights] * int(dim)), indexing="ij")
    pts = np.stack([m.ravel() for m in mesh], axis=1)
    w = np.ones(pts.shape[0], dtype=np.float64)
    for wm in wmesh:
        w = w * wm.ravel()
    return pts, w


@dataclass(frozen=True)
class PCEFit:
    """一次 PCE 拟合的结果（系数 + 拟合方式 + 规模 + 是否欠定）。"""

    coeff: np.ndarray
    mi: np.ndarray
    kind: str
    method: str
    n_train: int
    n_basis: int
    underdetermined: bool
    residual_train: float


def pce_projection(
    func: Callable[[np.ndarray], np.ndarray],
    *,
    dim: int = 1,
    order: int = 4,
    kind: str = "legendre",
    nq: int | None = None,
) -> PCEFit:
    r"""投影（伪谱）系数 :math:`c_\alpha=\frac{E[f\phi_\alpha]}{E[\phi_\alpha^2]}=E[f\phi_\alpha]`。

    节点的选择不是随便定的：被积函数是 :math:`f\,\phi_\alpha`，其中 :math:`\phi_\alpha`
    的次数是 :math:`|\alpha|\le p`，所以只要 :math:`\deg f\le p` 就需要
    :math:`2n_q-1\ge 2p`，即 :math:`n_q\ge p+1`。本实现取 :math:`n_q=p+1`（缺省），
    于是**任何次数 ≤ p 的多项式都被精确投影**。
    第一版取的是 :math:`\lceil(p+1)/2\rceil`（"积分 :math:`f\phi_\alpha` 只需
    :math:`2n_q-1\ge p`"的想当然写法），p=4 时只有 3 个节点，
    :math:`x^4` 的系数直接算错 0.34——而曲线看上去只是"收敛得慢一点"。

    投影永远不过拟合——它不需要"训练集"。
    """
    d = int(dim)
    p = int(order)
    nq_use = int(nq) if nq is not None else max(2, p + 1)
    pts, w = _tensor_nodes(d, nq_use, kind=kind)
    mi = multi_index(d, p)
    psi = basis_matrix(pts, mi, kind=kind)
    fv = np.asarray(func(pts), dtype=np.float64).ravel()
    coeff = psi.T @ (w * fv)
    approx = psi @ coeff
    return PCEFit(
        coeff=coeff,
        mi=mi,
        kind=kind,
        method="projection",
        n_train=int(pts.shape[0]),
        n_basis=int(mi.shape[0]),
        underdetermined=False,
        residual_train=float(np.sqrt(np.mean((approx - fv) ** 2))),
    )


def pce_regression(
    func: Callable[[np.ndarray], np.ndarray],
    x_train: np.ndarray,
    *,
    order: int = 4,
    kind: str = "legendre",
) -> PCEFit:
    r"""回归（最小二乘）系数：``min_c ||Psi c - f||_2``。

    * ``n_basis <= n_train``：列主元 QR（1.3 章的 :func:`numcore.linalg.qr.qr_least_squares`）；
    * ``n_basis > n_train``：**欠定**，改用 Jacobi SVD 的截断解（最小范数解）。
      这不是"补一个兜底分支"：过拟合区正是从这里开始，而最小范数解是那一段里
      唯一有定义的答案。返回的 ``underdetermined`` 让上层能如实标注。
    """
    xx = np.asarray(x_train, dtype=np.float64)
    if xx.ndim == 1:
        xx = xx[:, None]
    d = xx.shape[1]
    mi = multi_index(d, int(order))
    psi = basis_matrix(xx, mi, kind=kind)
    fv = np.asarray(func(xx), dtype=np.float64).ravel()
    n, nb = psi.shape
    if nb <= n:
        coeff = qr_least_squares(psi, fv)
        under = False
    else:
        fit = jacobi_svd(psi)
        coeff = truncated_solution(fit.U, fit.s, fit.V, fv, int(fit.s.size))
        under = True
    resid = float(np.sqrt(np.mean((psi @ coeff - fv) ** 2)))
    return PCEFit(
        coeff=coeff,
        mi=mi,
        kind=kind,
        method="regression",
        n_train=int(n),
        n_basis=int(nb),
        underdetermined=under,
        residual_train=resid,
    )


def pce_evaluate(fit: PCEFit, x: Any) -> np.ndarray:
    """代理模型在 ``x`` 上的取值 ``\\hat f(x)=\\sum_\\alpha c_\\alpha\\Psi_\\alpha(x)``。"""
    return basis_matrix(x, fit.mi, kind=fit.kind) @ fit.coeff


def pce_moments(fit: PCEFit) -> tuple[float, float]:
    r"""解析均值与方差：``E[f]=c_0``、``Var[f]=\sum_{\alpha\ne0}c_\alpha^2``。

    第二条**只在正交归一基 + 系数是投影/无偏回归时**成立；这是 PCE 最有用的一条恒等式，
    也是"代理模型的方差不用再抽样"的根据。``tests/5.4`` 拿它与蒙特卡洛方差对拍。
    """
    c = np.asarray(fit.coeff, dtype=np.float64).ravel()
    zero = np.all(fit.mi == 0, axis=1)
    var = float(np.sum(c[~zero] ** 2))
    return float(c[np.flatnonzero(zero)[0]]), var


def pce_sobol_indices(fit: PCEFit) -> tuple[np.ndarray, np.ndarray]:
    r"""由系数直接读出 Sobol' 指数（解析，不用抽样）：

    .. math::
        S_i=\frac{\sum_{\alpha\in\mathcal A_i}c_\alpha^2}{V},\quad
        S_{Ti}=\frac{\sum_{\alpha:\alpha_i>0}c_\alpha^2}{V},\qquad
        \mathcal A_i=\{\alpha\ne0:\alpha_i>0,\ \alpha_j=0\ (j\ne i)\}.
    """
    c = np.asarray(fit.coeff, dtype=np.float64).ravel()
    mi = np.asarray(fit.mi, dtype=np.int64)
    _, var = pce_moments(fit)
    d = mi.shape[1]
    first = np.zeros(d, dtype=np.float64)
    total = np.zeros(d, dtype=np.float64)
    if var <= 0.0:
        return first, total
    for j in range(d):
        others = np.delete(np.arange(d), j)
        only_j = (mi[:, j] > 0) & np.all(mi[:, others] == 0, axis=1) if d > 1 else (mi[:, j] > 0)
        any_j = mi[:, j] > 0
        first[j] = float(np.sum(c[only_j] ** 2)) / var
        total[j] = float(np.sum(c[any_j] ** 2)) / var
    return first, total


def relative_l2_error(fit: PCEFit, func: Callable[[np.ndarray], np.ndarray], x: Any) -> float:
    r"""相对 :math:`L_2` 误差 :math:`\|\hat f-f\|/\|f\|`（在**给定的**点集上；留出集由调用方保证）。"""
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim == 1:
        xx = xx[:, None]
    fv = np.asarray(func(xx), dtype=np.float64).ravel()
    pv = pce_evaluate(fit, xx)
    denom = float(np.sqrt(np.mean(fv * fv)))
    if denom <= 0.0:
        return float(np.sqrt(np.mean((pv - fv) ** 2)))
    return float(np.sqrt(np.mean((pv - fv) ** 2)) / denom)


# --------------------------------------------------------------------------- #
# 阶数-误差曲线（含过拟合区）与外推
# --------------------------------------------------------------------------- #
def error_vs_order(
    func: Callable[[np.ndarray], np.ndarray],
    *,
    dim: int = 1,
    orders: Sequence[int],
    n_train: int = 64,
    kind: str = "legendre",
    method: str = "regression",
    n_test: int = 2000,
    seed: Any = None,
) -> dict[str, np.ndarray]:
    r"""截断阶数 vs 误差：返回 ``{"orders", "train", "outsample", "n_basis", "underdetermined"}``。

    * ``train`` 在**训练集**上算（回归会一路降到机器零，欠定时精确过点）；
    * ``outsample`` 在**独立留出集**上算——它才是"代理模型好不好"的答案，
      并且会出现 U 形：阶数不够是偏差，阶数过高（基规模超过样本数）是方差。

    ``method="projection"`` 时 ``train`` 与 ``outsample`` 只是同一误差在两个点集上的取值
    （投影不用训练集，两条曲线必然几乎重合）——这正好说明**过拟合是回归独有的病**。
    """
    d = int(dim)
    xs = np.asarray(list(orders), dtype=np.int64)
    tr = np.empty(xs.size, dtype=np.float64)
    os_ = np.empty(xs.size, dtype=np.float64)
    nb = np.empty(xs.size, dtype=np.int64)
    under = np.zeros(xs.size, dtype=np.int64)
    xtr = design_samples(d, int(n_train), kind=kind, seed=20261018 if seed is None else seed)
    xte = design_samples(d, int(n_test), kind=kind, seed=(20261018 if seed is None else int(seed)) + 777)
    for i, p in enumerate(xs):
        if method == "projection":
            fit = pce_projection(func, dim=d, order=int(p), kind=kind)
            tr[i] = relative_l2_error(fit, func, xtr)
        else:
            fit = pce_regression(func, xtr, order=int(p), kind=kind)
            tr[i] = relative_l2_error(fit, func, xtr)
            under[i] = 1 if fit.underdetermined else 0
        os_[i] = relative_l2_error(fit, func, xte)
        nb[i] = int(fit.n_basis)
    return {"orders": xs, "train": tr, "outsample": os_, "n_basis": nb, "underdetermined": under}


def extrapolation_error(
    func: Callable[[np.ndarray], np.ndarray],
    *,
    dim: int = 1,
    order: int = 6,
    n_train: int = 64,
    kind: str = "legendre",
    method: str = "regression",
    factor: float = 1.2,
    n_test: int = 1000,
    seed: Any = None,
) -> dict[str, float]:
    r"""外推误差：在训练区间 :math:`[-1,1]`（或标准正态的 :math:`\pm3\sigma`）**之外**评估。

    ``factor`` 是"往外推多远"的倍率：Legendre 的评估区间取 :math:`[1, 1+\text{factor}-1]` 之外
    的半环——实现上取 :math:`|x|\in[1,\ \text{factor}]`（Hermite 取 :math:`[3, 3\,\text{factor}]`）。
    返回 ``{"insample", "outsample", "extrap", "ratio"}``：``extrap/outsample`` 就是
    "外推把误差放大了几倍"。
    """
    d = int(dim)
    f = float(factor)
    if f <= 1.0:
        raise ValueError("factor 必须 > 1（否则不叫外推）")
    xin = design_samples(d, int(n_test), kind=kind, seed=20261018 if seed is None else seed)
    xout = design_samples(d, int(n_test), kind=kind, seed=(20261018 if seed is None else int(seed)) + 31337)
    g = make_rng((20261018 if seed is None else int(seed)) + 4242)
    if kind == "legendre":
        inner = 1.0 + (f - 1.0) * g.random((int(n_test), d))
    else:
        inner = 3.0 + 3.0 * (f - 1.0) * g.random((int(n_test), d))
    xext = np.where(g.random((int(n_test), d)) < 0.5, -inner, inner)
    xtr = design_samples(d, int(n_train), kind=kind, seed=20261018 if seed is None else seed)
    if method == "projection":
        fit = pce_projection(func, dim=d, order=int(order), kind=kind)
    else:
        fit = pce_regression(func, xtr, order=int(order), kind=kind)
    ins = relative_l2_error(fit, func, xin)
    outs = relative_l2_error(fit, func, xout)
    ext = relative_l2_error(fit, func, xext)
    return {"insample": ins, "outsample": outs, "extrap": ext, "ratio": ext / outs if outs > 0 else float("inf")}


def coefficient_spectrum(fit: PCEFit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""谱：返回 ``(总阶数 levels, 每阶的 RMS 系数, 每阶的系数个数)``。

    逐阶 RMS（而不是"最大系数"）是更稳的衰减度量：单个大系数可能只是某个多指标的偶然，
    而 :math:`\\sqrt{\\frac1{\\#\\mathcal A_k}\\sum_{\\alpha\\in\\mathcal A_k}c_\\alpha^2}`
    直接对应 :math:`L_2` 误差的分布。解析函数上它按指数衰减
    （:math:`\\sim e^{-\\gamma k}`），有折点/间断的函数上只按幂律衰减
    （:math:`\\sim k^{-\\beta}`）——后者是本节"谱收敛不是免费的"那条。
    """
    c = np.asarray(fit.coeff, dtype=np.float64).ravel()
    mi = np.asarray(fit.mi, dtype=np.int64)
    levels = np.sum(mi, axis=1)
    kmax = int(np.max(levels)) if levels.size else 0
    rms = np.zeros(kmax + 1, dtype=np.float64)
    cnt = np.zeros(kmax + 1, dtype=np.int64)
    for k in range(kmax + 1):
        sel = levels == k
        cnt[k] = int(np.count_nonzero(sel))
        rms[k] = float(np.sqrt(np.mean(c[sel] ** 2))) if cnt[k] else 0.0
    return np.arange(kmax + 1, dtype=np.float64), rms, cnt


# --------------------------------------------------------------------------- #
# 目标函数（1 维，x 在训练区间内；每个都注明光滑性）
# --------------------------------------------------------------------------- #
def _t_poly4(x: np.ndarray) -> np.ndarray:
    return x[:, 0] ** 4 - 2.0 * x[:, 0] ** 2 + x[:, 0]


def _t_exp(x: np.ndarray) -> np.ndarray:
    return np.exp(1.5 * x[:, 0])


def _t_runge(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + 25.0 * x[:, 0] ** 2)


def _t_abs(x: np.ndarray) -> np.ndarray:
    return np.abs(x[:, 0])


TARGETS: dict[str, tuple[str, str, Callable[[np.ndarray], np.ndarray]]] = {
    "poly4": (
        "四次多项式 x^4 - 2x^2 + x",
        "4 阶截断就**精确**（其余系数精确为 0），p >= 4 时误差到机器零",
        _t_poly4,
    ),
    "exp": ("指数 exp(1.5 x)", "整函数：谱按指数衰减（实测斜率就是衰减率）", _t_exp),
    "runge": ("Runge 1/(1+25x^2)", "极点离区间不远：仍是指数衰减，但速率小得多", _t_runge),
    "abs": ("绝对值 |x|", "折点：谱只按幂律衰减，截断误差 ~ p^(-3/2)——谱收敛失效", _t_abs),
}


def target_names() -> tuple[str, ...]:
    """可用目标函数名（顺序稳定）。"""
    return tuple(TARGETS)


def target_label(name: str) -> str:
    """目标函数的中文标签。"""
    key = str(name)
    if key not in TARGETS:
        raise KeyError(f"未知目标函数 {name!r}（可用：{sorted(TARGETS)}）")
    return TARGETS[key][0]


def target_function(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """目标函数本身。"""
    key = str(name)
    if key not in TARGETS:
        raise KeyError(f"未知目标函数 {name!r}（可用：{sorted(TARGETS)}）")
    return TARGETS[key][2]


def fit_target(
    name: str,
    *,
    order: int = 6,
    kind: str = "legendre",
    method: str = "regression",
    n_train: int = 64,
    seed: Any = None,
) -> PCEFit:
    """按名字拟合一个 PCE（帧与断言共用的入口，保证两边同一套参数）。"""
    func = target_function(name)
    if method == "projection":
        return pce_projection(func, dim=1, order=int(order), kind=kind)
    xtr = design_samples(1, int(n_train), kind=kind, seed=20261018 if seed is None else seed)
    return pce_regression(func, xtr, order=int(order), kind=kind)
