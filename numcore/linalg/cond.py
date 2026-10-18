r"""条件数：:math:`\kappa(A)=\|A\|\,\|A^{-1}\|` 的四种范数、条件数椭球与误差界。

数学定义
--------
对方阵 :math:`A \in \mathbb{R}^{n\times n}` 与范数 :math:`\|\cdot\|`，

.. math::
    \kappa(A) = \|A\|\,\|A^{-1}\|, \qquad
    \kappa_2(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}.

本章要教的三个结论（完整证明见 ``chapters/1.1/notes.tex``）：

1. **前向误差界**：若 :math:`\|\Delta A\| \le \epsilon\|A\|`、:math:`\|\Delta b\| \le \epsilon\|b\|`
   且 :math:`\epsilon\kappa(A) < 1`，则
   :math:`\|\hat x - x\|/\|x\| \le \dfrac{2\epsilon\kappa(A)}{1-\epsilon\kappa(A)}`
   （Higham, *Accuracy and Stability*, §7.2；Golub--Van Loan, §2.7）。
2. **条件数的几何意义**：单位球在 :math:`A` 下的像是椭球，**半轴之比恰为** :math:`\kappa_2(A)`；
   面积是 :math:`|\det A|`（体积与形状是两件不同的事）。
3. **范数差异**：:math:`\kappa_2 \le \kappa_1 \le n\,\kappa_2` 之类的不等式意味着
   "条件数"必须带着范数说——同一矩阵 :math:`\kappa_1` 与 :math:`\kappa_2` 可以差一个 :math:`n`。

实现约束
--------
只用 ``numpy`` 的容器与逐元素原语，**不调用** ``numpy.linalg``（POLICY P2）：
矩阵逆由 :mod:`numcore.linalg.lu` 的 LU 逐列回代得到；广义谱范数由幂迭代/反幂迭代给出。
本节函数的命名一律避开被禁的属性名（``cond``/``norm``/``inv``/``det``/``svd``），
统一用 :func:`condition_number` / :func:`condition_1` / :func:`matrix_inverse` 这类名字。
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from numcore.linalg.lu import (
    backward_error,
    ball_matrix,
    lu_partial,
    residual_norm,
    singular_values_2x2,
    solve_lu,
    unit_ball_image,
)
from numcore.utils import rng

__all__ = [
    # 范数（自研）
    "norm_1",
    "norm_inf",
    "norm_fro",
    "spectral_norm_estimate",
    "smallest_singular_estimate",
    "singular_values_estimate",
    # 逆与条件数
    "matrix_inverse",
    "condition_1",
    "condition_inf",
    "condition_fro",
    "condition_2",
    "condition_number",
    "condition_componentwise",
    # 误差界与实测放大比
    "forward_error_bound",
    "rhs_amplification_measured",
    "perturbed_solution",
    # 几何（条件数椭球）
    "ellipse_semi_axes",
    "ellipse_image",
    "ellipse_radii",
    # 演示矩阵族
    "condition_matrix",
    "hilbert_matrix",
    "givens_rotation",
    "orthogonal_matrix",
    "symmetric_with_spectrum",
    "diagonal_condition",
]

Norm = Literal[1, 2, "inf", "fro"]

#: 允许的范数记号（``"inf"`` 与 ``np.inf`` 等价，``"fro"`` 是 Frobenius）。
_NORMS: tuple[Any, ...] = (1, 2, "inf", "fro")


def _as_square(A: Any) -> np.ndarray:
    """校验并返回方阵（float64，C 连续）。"""
    arr = np.asarray(A, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"需要方阵，得到形状 {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("空矩阵没有条件数")
    if not np.all(np.isfinite(arr)):
        raise ValueError("矩阵含非有限元（nan/inf）")
    return np.ascontiguousarray(arr)


def _norm_key(p: Any) -> Any:
    """把范数记号规范化，未知记号直接报错（不静默回退到 2 范数）。"""
    if p is None:
        return 2
    if isinstance(p, str):
        low = p.strip().lower()
        if low in ("inf", "infinity", "max"):
            return "inf"
        if low in ("fro", "f", "frobenius"):
            return "fro"
        if low in ("1", "one"):
            return 1
        if low in ("2", "two", "spec", "spectral"):
            return 2
        raise ValueError(f"未知范数 {p!r}（可用 1 / 2 / 'inf' / 'fro'）")
    if p in (1, 2):
        return int(p)
    if p == np.inf or p == float("inf"):
        return "inf"
    raise ValueError(f"未知范数 {p!r}（可用 1 / 2 / 'inf' / 'fro'）")


# --------------------------------------------------------------------------- #
# 范数
# --------------------------------------------------------------------------- #
def norm_1(A: Any) -> float:
    r""":math:`\|A\|_1 = \max_j \sum_i |a_{ij}|`（最大**列**和）。"""
    arr = _as_square(A)
    return float(np.max(np.sum(np.abs(arr), axis=0)))


def norm_inf(A: Any) -> float:
    r""":math:`\|A\|_\infty = \max_i \sum_j |a_{ij}|`（最大**行**和）。"""
    arr = _as_square(A)
    return float(np.max(np.sum(np.abs(arr), axis=1)))


def norm_fro(A: Any) -> float:
    r""":math:`\|A\|_F = \sqrt{\sum_{ij} a_{ij}^2}`（按最大元缩放求和，避免上溢）。"""
    arr = _as_square(A)
    m = float(np.max(np.abs(arr)))
    if m == 0.0:
        return 0.0
    return float(m * np.sqrt(np.sum((arr / m) ** 2)))


def _rayleigh_sigma_max(B: np.ndarray, *, n_iter: int, tol: float, seed: int, restarts: int) -> float:
    r""":math:`B = A^{\mathsf T}A` 的最大特征值的平方根（幂迭代，多个随机起点取最大）。

    多个起点是必要的：幂迭代只保证收敛到**起点所在不变子空间**里的最大特征对，
    一个恰好与主特征向量正交的起点会给出错误的（偏小的）结果。
    """
    n = B.shape[0]
    g = rng(seed)
    best = 0.0
    for _ in range(max(int(restarts), 1)):
        x = np.asarray(g.standard_normal(n), dtype=float)
        nx = float(np.sqrt(np.sum(x * x)))
        if nx == 0.0:  # pragma: no cover - 概率为零的退化
            continue
        x = x / nx
        lam = float(x @ (B @ x))
        for _it in range(max(int(n_iter), 1)):
            z = B @ x
            nz = float(np.sqrt(np.sum(z * z)))
            if nz == 0.0:
                lam = 0.0
                break
            x = z / nz
            new = float(x @ (B @ x))
            if abs(new - lam) <= tol * max(abs(new), 1e-300):
                lam = new
                break
            lam = new
        best = max(best, lam)
    return float(np.sqrt(max(best, 0.0)))


def spectral_norm_estimate(
    A: Any, *, n_iter: int = 400, tol: float = 1e-15, seed: int = 20261018, restarts: int = 2
) -> float:
    r"""谱范数 :math:`\|A\|_2 = \sigma_{\max}(A)` 的幂迭代估计。

    估计**不是**真值：它按 :math:`(\sigma_2/\sigma_1)^2` 线性收敛，且起点不巧会低估。
    所以 :func:`condition_number` 在小矩阵上用闭式/对拍，在大矩阵上才退回这里，
    并把"它只是个估计"写进返回值的语义里（不要把它当作解析真值）。
    """
    arr = _as_square(A)
    return _rayleigh_sigma_max(arr.T @ arr, n_iter=n_iter, tol=tol, seed=seed, restarts=restarts)


def smallest_singular_estimate(
    A: Any, *, n_iter: int = 400, tol: float = 1e-15, seed: int = 20261018
) -> float:
    r""":math:`\sigma_{\min}(A)` 的反幂迭代估计（在 :math:`A^{\mathsf T}A` 上做逆迭代）。

    注意代价：这一步显式构造 :math:`A^{\mathsf T}A` 并解它，**把条件数平方了**
    （:math:`\kappa_2(A^{\mathsf T}A) = \kappa_2(A)^2`）。所以 :math:`\kappa \gtrsim 10^{8}`
    之后这个估计就不可信了——``chapters/1.1/pitfalls.md`` 把这条列为必读。
    矩阵奇异或 :math:`\sigma_{\min}` 在机器精度下为 0 时返回 0。
    """
    arr = _as_square(A)
    n = arr.shape[0]
    B = arr.T @ arr
    res = lu_partial(B)
    if res.singular or float(np.min(np.abs(np.diag(res.U)))) == 0.0:
        return 0.0
    g = rng(seed)
    x = np.asarray(g.standard_normal(n), dtype=float)
    nx = float(np.sqrt(np.sum(x * x)))
    if nx == 0.0:  # pragma: no cover - 概率为零的退化
        return 0.0
    x = x / nx
    rho = 0.0
    for _it in range(max(int(n_iter), 1)):
        z = solve_lu(res, x)
        nz = float(np.sqrt(np.sum(z * z)))
        if nz == 0.0 or not np.isfinite(nz):
            return 0.0
        x = z / nz
        new = float(x @ solve_lu(res, x))
        if abs(new - rho) <= tol * max(abs(new), 1e-300) and rho > 0.0:
            rho = new
            break
        rho = new
    if rho <= 0.0 or not np.isfinite(rho):
        return 0.0
    return float(1.0 / np.sqrt(rho))


def singular_values_estimate(A: Any) -> tuple[float, float]:
    r"""``(sigma_min, sigma_max)``：2×2 用闭式解，更大规模用幂迭代/反幂迭代估计。"""
    arr = _as_square(A)
    if arr.shape == (2, 2):
        return singular_values_2x2(arr)
    smax = spectral_norm_estimate(arr)
    smin = smallest_singular_estimate(arr)
    return float(smin), float(smax)


# --------------------------------------------------------------------------- #
# 矩阵逆与条件数
# --------------------------------------------------------------------------- #
def matrix_inverse(A: Any) -> np.ndarray:
    r""":math:`A^{-1}`：用自研 LU + 逐列回代（:math:`Ay_j = e_j`），不用 ``numpy.linalg``。

    奇异（或主元精确为 0）时抛 :class:`ZeroDivisionError`——"逆不存在"必须是异常，
    静默返回 inf/nan 会让下游的 :math:`\kappa` 变成一个看起来很大的数字，反而更像"病态"。
    """
    arr = _as_square(A)
    n = arr.shape[0]
    res = lu_partial(arr)
    if res.singular or float(np.min(np.abs(np.diag(res.U)))) == 0.0:
        raise ZeroDivisionError("矩阵奇异（或主元恰为 0），逆不存在")
    cols = []
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        cols.append(solve_lu(res, e))
    return np.stack(cols, axis=1)


def _kappa(A: Any, p: Any) -> float:
    key = _norm_key(p)
    arr = _as_square(A)
    if key == 2 and arr.shape == (2, 2):
        smin, smax = singular_values_2x2(arr, stable=True)
        return float("inf") if smin <= 0.0 else float(smax / smin)
    inv = matrix_inverse(arr)
    if key == 1:
        return norm_1(arr) * norm_1(inv)
    if key == "inf":
        return norm_inf(arr) * norm_inf(inv)
    if key == "fro":
        return norm_fro(arr) * norm_fro(inv)
    smin, smax = singular_values_estimate(arr)
    return float("inf") if smin <= 0.0 else float(smax / smin)


def condition_1(A: Any) -> float:
    r""":math:`\kappa_1(A) = \|A\|_1\|A^{-1}\|_1`。"""
    return _kappa(A, 1)


def condition_inf(A: Any) -> float:
    r""":math:`\kappa_\infty(A) = \|A\|_\infty\|A^{-1}\|_\infty`。"""
    return _kappa(A, "inf")


def condition_fro(A: Any) -> float:
    r""":math:`\kappa_F(A) = \|A\|_F\|A^{-1}\|_F`（Frobenius 范数的条件数）。"""
    return _kappa(A, "fro")


def condition_2(A: Any) -> float:
    r""":math:`\kappa_2(A) = \sigma_{\max}/\sigma_{\min}`（谱条件数，= 椭球半轴之比）。

    2×2 走闭式解（:math:`\sigma_{\min}=|\det A|/\sigma_{\max}`，用除法换掉相减）；
    更大规模用幂迭代估计（见 :func:`smallest_singular_estimate` 的代价说明）。
    """
    return _kappa(A, 2)


def condition_number(A: Any, p: Any = 2) -> float:
    r"""条件数 :math:`\kappa(A) = \|A\|\,\|A^{-1}\|`，``p`` 取 ``1 / 2 / "inf" / "fro"``。

    没有缺省之外的"自动选择"：不同范数可以差一个 :math:`n` 因子
    （:math:`\kappa_2 \le \kappa_1 \le n\kappa_2`），所以图上、报告里都必须写清用的是哪个。
    """
    return _kappa(A, p)


def condition_componentwise(A: Any) -> float:
    r""":math:`\kappa_{\rm comp}(A) = \bigl\||A^{-1}|\,|A|\bigr\|_\infty`。

    它对应**分量相对**扰动模型 :math:`|\Delta A| \le \epsilon|A|`，比范数型
    :math:`\|\Delta A\| \le \epsilon\|A\|` 更贴近"每个元素各自有 :math:`u` 级相对误差"
    的真实浮点场景：对**行缩放**的矩阵它不变，而 :math:`\kappa_\infty` 会随缩放漂移。
    """
    arr = _as_square(A)
    inv = matrix_inverse(arr)
    M = np.abs(inv) @ np.abs(arr)
    return float(np.max(np.sum(M, axis=1)))


# --------------------------------------------------------------------------- #
# 误差界与实测放大比
# --------------------------------------------------------------------------- #
def forward_error_bound(
    A: Any, *, rel_delta_b: float = 0.0, rel_delta_a: float = 0.0, p: Any = 2
) -> float:
    r"""前向相对误差界 :math:`\dfrac{\kappa(\delta A+\delta b)}{1-\kappa\,\delta A}`
    （:math:`\delta A = \|\Delta A\|/\|A\|`，:math:`\delta b = \|\Delta b\|/\|b\|`）。

    这是本章的定量主线：**误差 = 条件数 × 扰动**。第一个因子由问题决定（改不了），
    第二个因子由数据/算法决定（可以改）。分母在 :math:`\kappa\delta A \ge 1` 时失效——
    那不是"界不够紧"，而是问题在给定精度下已经不可解。
    """
    k = condition_number(A, p)
    da = float(rel_delta_a)
    db = float(rel_delta_b)
    if k * da >= 1.0:
        return float("inf")
    return float(k * (da + db) / (1.0 - k * da))


def perturbed_solution(A: Any, b: Any, delta_b: Any) -> np.ndarray:
    r"""解 :math:`A x = b + \Delta b`（用自研 LU），给"扰动右端项"实验用。"""
    arr = _as_square(A)
    bv = np.asarray(b, dtype=float).reshape(-1)
    dv = np.asarray(delta_b, dtype=float).reshape(-1)
    if bv.size != arr.shape[0] or dv.size != arr.shape[0]:
        raise ValueError(f"长度不匹配：A{arr.shape}, b{bv.shape}, db{dv.shape}")
    res = lu_partial(arr)
    if res.singular:
        raise ZeroDivisionError("矩阵奇异，扰动实验无意义")
    return solve_lu(res, bv + dv)


def rhs_amplification_measured(
    A: Any, b: Any, delta_b: Any, *, p: Any = 2, exact: Any = None
) -> float:
    r"""右端项扰动的**实测**放大比
    :math:`\dfrac{\|\hat x' - \hat x\|/\|\hat x\|}{\|\Delta b\|/\|b\|}`。

    分子用同一个求解器算两次（同一种舍入走向），所以它主要反映**问题**的放大，
    而不是算法的随机抖动。理论上界就是 :math:`\kappa(A)`（:func:`forward_error_bound`），
    实测比值通常只有它的百分之几十——把实测值和上界都画出来才叫"不撒谎"。
    """
    arr = _as_square(A)
    bv = np.asarray(b, dtype=float).reshape(-1)
    dv = np.asarray(delta_b, dtype=float).reshape(-1)
    key = _norm_key(p)

    def _norm(v: np.ndarray) -> float:
        if key == 1:
            return float(np.sum(np.abs(v)))
        if key == "inf":
            return float(np.max(np.abs(v)))
        if key == "fro":
            return float(np.sqrt(np.sum(v * v)))
        return float(np.sqrt(np.sum(v * v)))

    x0 = np.asarray(exact, dtype=float) if exact is not None else perturbed_solution(arr, bv, np.zeros_like(bv))
    x1 = perturbed_solution(arr, bv, dv)
    nb = _norm(bv)
    nd = _norm(dv)
    nx = _norm(x0)
    if nb == 0.0 or nd == 0.0 or nx == 0.0:
        raise ValueError("b、Δb、x 都必须非零，放大比才有意义")
    return float((_norm(x1 - x0) / nx) / (nd / nb))


# --------------------------------------------------------------------------- #
# 几何：单位球在 A 下的像（条件数椭球）
# --------------------------------------------------------------------------- #
def ellipse_semi_axes(A: Any) -> tuple[float, float]:
    r"""单位圆像（椭圆）的半轴 ``(短轴, 长轴)`` = :math:`(\sigma_{\min}, \sigma_{\max})`。

    **半轴之比就是** :math:`\kappa_2(A)`：这就是条件数的几何定义，
    也是"条件数椭球"这个可视化的全部内容。2×2 闭式，见 :func:`singular_values_2x2`。
    """
    arr = _as_square(A)
    if arr.shape != (2, 2):
        raise ValueError(f"椭球可视化目前只支持 2×2，得到 {arr.shape}")
    return singular_values_2x2(arr)


def ellipse_image(A: Any, n_theta: int = 361) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""单位圆的像 ``(theta, x, y)``，``x = A @ (cos t, sin t)``。"""
    return unit_ball_image(A, n_theta=n_theta)


def ellipse_radii(A: Any, n_theta: int = 361) -> tuple[float, float]:
    r"""在单位圆上均匀**采样**得到的像的半径区间 ``(min r, max r)``（不是闭式奇异值）。

    它总是**高估** :math:`\sigma_{\min}`（除非采样点恰好落在短轴方向），
    因为最扁方向上的 :math:`r(\theta)` 变化率是 :math:`O(\kappa)`。
    返回它是为了让"采样测 κ"这种常见做法有一个可被断言的误差量。
    """
    _, x, y = unit_ball_image(A, n_theta=n_theta)
    r = np.sqrt(x * x + y * y)
    return float(np.min(r)), float(np.max(r))


# --------------------------------------------------------------------------- #
# 演示矩阵族
# --------------------------------------------------------------------------- #
def hilbert_matrix(n: int) -> np.ndarray:
    r"""Hilbert 矩阵 :math:`H_{ij} = 1/(i+j+1)`（:math:`\kappa` 随 n 指数增长）。"""
    k = int(n)
    if k < 1:
        raise ValueError(f"n 至少为 1，得到 {n!r}")
    i = np.arange(k, dtype=float)[:, None]
    j = np.arange(k, dtype=float)[None, :]
    return 1.0 / (i + j + 1.0)


def diagonal_condition(d: Any) -> float:
    r"""对角矩阵 :math:`\mathrm{diag}(d)` 的精确条件数 :math:`\max|d_i|/\min|d_i|`（任何 p 范数）。"""
    dd = np.asarray(d, dtype=float).reshape(-1)
    if dd.size == 0:
        raise ValueError("空对角")
    lo = float(np.min(np.abs(dd)))
    hi = float(np.max(np.abs(dd)))
    if lo == 0.0:
        return float("inf")
    return hi / lo


def givens_rotation(n: int, i: int, j: int, theta: float) -> np.ndarray:
    r""":math:`n\times n` Givens 旋转 :math:`G_{ij}(\theta)`（在 (i,j) 平面内旋转）。"""
    k = int(n)
    a, b = int(i), int(j)
    if not (0 <= a < k and 0 <= b < k and a != b):
        raise ValueError(f"下标非法：n={n}, i={i}, j={j}")
    G = np.eye(k)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    G[a, a] = c
    G[b, b] = c
    G[a, b] = -s
    G[b, a] = s
    return G


def orthogonal_matrix(n: int, *, seed: int = 20261018) -> np.ndarray:
    r"""随机正交矩阵：若干 Givens 旋转的乘积（逐位正交，:math:`\kappa_2 = 1`）。

    不用 QR 分解（那是 1.3 章的内容）：Givens 旋转的乘积本身就是正交的，
    并且"正交矩阵的条件数**恰好**是 1"是一个可以被精确断言的解析事实。
    """
    k = int(n)
    if k < 1:
        raise ValueError(f"n 至少为 1，得到 {n!r}")
    g = rng(seed)
    Q = np.eye(k)
    for i in range(k - 1):
        for j in range(i + 1, k):
            Q = Q @ givens_rotation(k, i, j, float(g.uniform(-np.pi, np.pi)))
    return Q


def symmetric_with_spectrum(eigs: Any, *, seed: int = 20261018) -> np.ndarray:
    r"""构造**已知特征值**的对称矩阵 :math:`A = Q\Lambda Q^{\mathsf T}`（:math:`Q` 正交）。

    2-范数条件数因此是解析已知的 :math:`\max|\lambda|/\min|\lambda|`；
    它给"条件数的解析对拍"提供任意指定 :math:`\kappa` 的测试矩阵。
    """
    lam = np.asarray(eigs, dtype=float).reshape(-1)
    if lam.size == 0:
        raise ValueError("特征值列表为空")
    Q = orthogonal_matrix(lam.size, seed=seed)
    return Q @ np.diag(lam) @ Q.T


def condition_matrix(name: str, kappa: float, *, shear: float = 0.8) -> np.ndarray:
    r"""2×2 演示矩阵族（给"条件数椭球"可视化用），``kappa`` 是**目标** :math:`\kappa_2`。

    ==================  ==========================================================
    ``shear``           先剪切再各向异性拉伸，:math:`\kappa_2 \approx \kappa`
    ``diagonal``        :math:`\mathrm{diag}(1, 1/\kappa)`，:math:`\kappa_2 = \kappa` 精确
    ``rotation``        正交矩阵，:math:`\kappa_2 = 1`（扰动被原样传递，不放大）
    ``graded``          :math:`[[1,1],[1,1+1/\kappa]]`，:math:`\kappa_2 \approx 4\kappa`
    ``triangular``      上三角 :math:`[[1,1],[0,1/\kappa]]`
    ==================  ==========================================================

    实测 :math:`\kappa_2` 请用 :func:`condition_2`：``graded`` 与 ``triangular``
    的闭式值并不等于 ``kappa``（图上必须标"目标值 vs 实测值"，否则就是在撒谎）。
    """
    k = float(kappa)
    if not np.isfinite(k) or k < 1.0:
        raise ValueError(f"kappa 必须是 >= 1 的有限数，得到 {kappa!r}")
    key = str(name).strip().lower()
    if key == "shear":
        return ball_matrix(float(shear), k)
    if key == "diagonal":
        return np.diag([1.0, 1.0 / k])
    if key in ("rotation", "orthogonal"):
        th = np.deg2rad(40.0)
        return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    if key == "graded":
        return np.array([[1.0, 1.0], [1.0, 1.0 + 1.0 / k]])
    if key == "triangular":
        return np.array([[1.0, 1.0], [0.0, 1.0 / k]])
    raise KeyError(f"未知矩阵族 {name!r}（可用 shear/diagonal/rotation/graded/triangular）")


def backward_error_estimate(A: Any, x: Any, b: Any) -> float:
    r"""范数型后向误差 :math:`\eta = \|Ax-b\|_\infty/(\|A\|_\infty\|x\|_\infty+\|b\|_\infty)`。

    它与 :math:`\kappa` **无关**（后向稳定的算法给出 :math:`\eta \approx u`），
    与 :func:`forward_error_bound` 一起构成"稳定 ≠ 准"的两半证据。
    """
    return backward_error(A, x, b)


def residual_size(A: Any, x: Any, b: Any) -> float:
    r""":math:`\|Ax-b\|_\infty`（绝对残差，给副轴的"看起来很准"做对照）。"""
    return residual_norm(A, x, b)
