"""特征值算法：幂法、反迭代、Hessenberg 化、QR 迭代、Lanczos、Arnoldi。

数学定义
--------
对 :math:`A \\in \\mathbb{R}^{n\\times n}`，特征对 :math:`(\\lambda, v)` 满足

.. math::
    A v = \\lambda v, \\qquad v \\ne 0 .

本模块把"求特征值"拆成几条**互相独立、可分别测量**的路线，每条都给出它自己的
收敛率——这正是 1.4 章要教的东西：特征值本身不是难点，**收敛速度与非正规性**才是。

======================  ==================================================  ==========================
函数                    迭代格式                                             收敛因子
======================  ==================================================  ==========================
:func:`power_iteration`     :math:`x_{k+1} = A x_k / \\|A x_k\\|`                :math:`|\\lambda_2/\\lambda_1|`
:func:`inverse_iteration`   :math:`(A-\\sigma I) x_{k+1} = x_k`（再归一化）      :math:`|\\lambda_i-\\sigma|/|\\lambda_j-\\sigma|`
:func:`rayleigh_quotient`   :math:`\\rho(x) = x^{\\mathsf T} A x / x^{\\mathsf T}x`  ——（它是估计量，不是迭代）
:func:`hessenberg_reduce`   Householder 相似变换，:math:`h_{ij}=0\\ (i>j+1)`  有限步（:math:`\\approx 4n^3/3`）
:func:`qr_iteration`        :math:`H_k = Q_kR_k,\\ H_{k+1}=R_kQ_k`              :math:`|\\lambda_{i+1}/\\lambda_i|`
:func:`lanczos`             三项递推 :math:`\\beta_{j+1}v_{j+1}=Av_j-\\alpha_jv_j-\\beta_jv_{j-1}`  Ritz 值收敛
:func:`arnoldi`             :math:`AV_m = V_mH_m + h_{m+1,m}v_{m+1}e_m^{\\mathsf T}`  同上（非对称）
======================  ==================================================  ==========================

幂法的收敛定理（证明见 ``chapters/1.4/notes.tex`` 定理 1）：若 :math:`|\\lambda_1|>|\\lambda_2|\\ge\\cdots`,
且 :math:`x_0` 在主特征向量上有非零分量，则 :math:`\\|x_k - \\pm v_1\\| = O(|\\lambda_2/\\lambda_1|^k)`。
**注意它是渐近结论**：非正规矩阵上开头可能长期看不出收敛（瞬态），
幂法甚至可能先"跑偏"再回来——这是本章 ``transient_growth`` 一页的主题。

为什么必须手写
--------------
本模块的所有算法都只用 ``numpy`` 的容器与逐元素原语（``@``、切片、``abs``、``sqrt``、
``sum``、``argmax``…），**不调用** ``numpy.linalg`` 的任何东西：``eig/eigh/eigvals/svd/qr/norm``
在 POLICY P2/P3 里都是被禁的"成品算法"。对拍用的成熟实现放在
:mod:`numcore.reference.dense`，只能由 ``tests/`` 与 ``tools/`` 调用（P10）。
随机向量一律走 :func:`numcore.utils.rng`（P4）。

名词约定
--------
* **Ritz 值**：:math:`\\theta = y^{\\mathsf T} H_m y`（:math:`H_m = V_m^{\\mathsf T}AV_m` 在 Krylov 基下的表示），
  它是 :math:`A` 的特征值关于子空间 :math:`\\mathcal{K}_m` 的最佳近似；
* **Ritz 对的残差**：:math:`\\|AV_m y - \\theta V_m y\\|_2`，对 Arnoldi 有闭式
  :math:`|h_{m+1,m}|\\,|e_m^{\\mathsf T}y|`（见 :func:`ritz_residuals`）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from numcore.utils import rng

__all__ = [
    # 幂法与反迭代
    "PowerResult",
    "InverseResult",
    "power_iteration",
    "inverse_iteration",
    "rayleigh_quotient",
    "normalize",
    "dominant_eigenpair",
    # 奇异值（自研单边 Jacobi）
    "one_sided_jacobi_svd",
    "smallest_singular_value",
    "singular_values",
    "hermitian_jacobi_eigh",
    "smallest_singular_value_hermitian",
    # 残差与谱量
    "eigen_residual",
    "spectral_radius_estimate",
    "matrix_2norm",
    # Hessenberg 与 QR
    "hessenberg_reduce",
    "QRResult",
    "qr_iteration",
    "qr_subdiagonal_history",
    "schur_eigenvalues",
    # Krylov
    "LanczosResult",
    "ArnoldiResult",
    "lanczos",
    "arnoldi",
    "ritz_values",
    "ritz_residuals",
    "cauchy_interlacing_bounds",
    # 组合入口
    "dense_eigenvalues",
    # 示例矩阵
    "EXAMPLE_EIGEN_MATRICES",
    "example_eigen_matrix",
    "ratio_matrix",
    "shear_matrix",
    "jordan_block",
    "companion_matrix",
    "grcar_matrix",
    "laplacian_1d",
    "symmetric_tridiag",
]


# --------------------------------------------------------------------------- #
# 基础工具（全部自研：不用 np.linalg.norm / np.linalg.solve）
# --------------------------------------------------------------------------- #
def _norm2(x: Any) -> float:
    """欧氏范数 :math:`\\|x\\|_2`（先按最大元缩放，避免上溢/下溢）。

    自研：``np.linalg.norm`` 是 P3 禁用的成品（它背后是 BLAS 的 nrm2，
    但本章要求"连范数都自己写"，这样才谈得上把误差量级讲清楚）。
    """
    v = np.asarray(x, dtype=complex).reshape(-1)
    if v.size == 0:
        return 0.0
    m = float(np.max(np.abs(v)))
    if m == 0.0:
        return 0.0
    return m * float(np.sqrt(np.sum((np.abs(v) / m) ** 2)))


def _norm2_real(x: Any) -> float:
    """:math:`\\|x\\|_2`（实数输入，走实数分支，快一倍）。"""
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size == 0:
        return 0.0
    m = float(np.max(np.abs(v)))
    if m == 0.0:
        return 0.0
    return m * float(np.sqrt(np.sum((v / m) ** 2)))


def _max_abs(A: Any) -> float:
    """:math:`\\max_{ij}|a_{ij}|`（矩阵的无穷大"入口"范数，用于相对判据）。"""
    arr = np.asarray(A)
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))


def _as_square(A: Any) -> np.ndarray:
    """校验并复制成方阵（复数输入保持复数，实数输入保持实数）。"""
    arr = np.asarray(A)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"需要方阵，得到形状 {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("空矩阵没有特征值")
    if np.iscomplexobj(arr):
        return np.array(arr, dtype=complex, copy=True)
    return np.array(arr, dtype=float, copy=True)


def normalize(x: Any) -> np.ndarray:
    """:math:`x/\\|x\\|_2`；零向量抛 :class:`ValueError`（而不是静默返回 NaN）。

    幂法/反迭代每一步都要做这件事：没有归一化，向量的模会按
    :math:`|\\lambda_1|^k` 指数增长或消失，几十步就溢出了。
    """
    v = np.asarray(x).reshape(-1)
    n = _norm2(v)
    if n == 0.0:
        raise ValueError("零向量无法归一化（幂法的起点必须在主特征方向上有分量）")
    return v / n


def _random_start(n: int, *, seed: int, complex_: bool = False) -> np.ndarray:
    """确定性随机起点（``numcore.utils.rng``，P4；禁止 ``numpy.random``）。"""
    g = rng(seed)
    v = g.standard_normal(n)
    if complex_:
        return v + 1j * g.standard_normal(n)
    return v


def rayleigh_quotient(A: Any, x: Any) -> float | complex:
    r"""Rayleigh 商 :math:`\rho(x) = \dfrac{x^{\mathsf H} A x}{x^{\mathsf H} x}`。

    两条性质（都在 ``tests/1.4/test_eigen.py`` 里测掉）：

    * 对特征向量它**精确**等于特征值；对一般向量，它是 :math:`A` 的谱上离它最近的点
      与它之间的误差是 :math:`O(\|x-v_1\|^2)`——所以**特征值比特征向量收敛快一倍**；
    * 对 Hermite 矩阵，:math:`\rho(x)` 落在 :math:`[\lambda_{\min},\lambda_{\max}]` 内
      （Courant–Fischer），实部与虚部的偏离给出"当前向量离特征子空间有多远"。
    """
    arr = _as_square(A)
    v = np.asarray(x).reshape(-1)
    if v.size != arr.shape[0]:
        raise ValueError(f"维度不匹配：A 是 {arr.shape[0]}×{arr.shape[0]}，x 长 {v.size}")
    denom = complex(np.vdot(v, v))
    if denom == 0:
        raise ValueError("零向量的 Rayleigh 商无定义")
    q = complex(np.vdot(v, arr @ v)) / denom
    if not np.iscomplexobj(arr) and not np.iscomplexobj(v) and abs(q.imag) == 0.0:
        return float(q.real)
    return q


def eigen_residual(A: Any, lam: complex, v: Any) -> float:
    r"""特征对残差 :math:`\|Av-\lambda v\|_2`（**不做**归一化：v 按给定量纲算）。

    这是"这个特征对有多可信"的直接度量；反向误差意义下
    :math:`\lambda` 是某个 :math:`A+\delta A` 的精确特征值，只要
    :math:`\|\delta A\|_2 \le \|Av-\lambda v\|_2/\|v\|_2`（Kahan 的结论）。
    """
    arr = _as_square(A)
    x = np.asarray(v).reshape(-1)
    return _norm2(arr @ x - lam * x)


def spectral_radius_estimate(A: Any, eigenvalues: Any) -> float:
    r""":math:`\max_i |\lambda_i|`（谱半径；要求给出特征值数组，门面函数而已）。"""
    lam = np.asarray(eigenvalues).reshape(-1)
    if lam.size == 0:
        raise ValueError("空特征值集合")
    return float(np.max(np.abs(lam)))


def matrix_2norm(A: Any, *, iters: int = 200, tol: float = 1e-14, seed: int = 7) -> float:
    r""":math:`\|A\|_2` 的幂法估计（自研，不用 SVD）。

    对 :math:`A^{\mathsf T}A` 做幂法：:math:`\|A\|_2^2 = \lambda_{\max}(A^{\mathsf T}A)`。
    收敛因子是 :math:`(\sigma_2/\sigma_1)^2`——**这在 :math:`\sigma_1\approx\sigma_2` 时很慢**，
    所以它是"估计"：本模块与 :mod:`numcore.linalg.pseudospec` 都只把它当量级用，
    精确值留给 :mod:`numcore.reference.dense` 在测试里对拍。
    """
    arr = _as_square(A)
    n = arr.shape[0]
    B = arr.conj().T @ arr
    x = normalize(_random_start(n, seed=seed, complex_=np.iscomplexobj(arr)))
    lam = 0.0
    for _ in range(max(1, int(iters))):
        y = B @ x
        lam_new = float(np.real(np.vdot(x, y)))
        ny = _norm2(y)
        if ny == 0.0:
            return 0.0
        x = y / ny
        if abs(lam_new - lam) <= tol * max(1.0, abs(lam_new)):
            lam = lam_new
            break
        lam = lam_new
    return float(np.sqrt(max(0.0, lam)))


# --------------------------------------------------------------------------- #
# 幂法
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PowerResult:
    """幂法的一次运行：向量序列 + Rayleigh 商序列 + 收敛诊断。

    ``vectors`` 是**归一化后**的迭代向量（第 0 个是起点），形状 ``(k+1, n)``；
    ``rayleigh`` 是每一步的 Rayleigh 商（对实矩阵是实数）；``residual`` 是对应残差；
    ``angle_step`` 是相邻两向量夹角（`:math:`\\pi/2` 附近说明在原地打转——
    复主特征值时就是这个现象，见 :func:`power_iteration` 的说明）。

    ``ratios`` 是残差序列的逐步比值；``ratio`` 是它的几何平均，即**实测收敛因子**
    :math:`\\hat r`，可与 :math:`|\\lambda_2/\\lambda_1|` 对拍（A 类断言的常用靶子）。
    提醒：深收敛区残差已经是**相消差**（``|Ax-\\lambda x| \\sim u\\|A\\|``），
    此时比值带 ~``u``/残差 量级的噪声，跨实现只能给 1e-4 量级的相对判据
    （1.4 章与 G7 的 ``power_iter_ratio_floor`` case 都撞到过这条）。
    """

    eigenvalue: float
    vector: np.ndarray
    vectors: np.ndarray
    rayleigh: np.ndarray
    residual: np.ndarray
    angle_step: np.ndarray
    n_iter: int
    converged: bool
    ratios: np.ndarray
    ratio: float
    is_complex_pair: bool


def power_iteration(
    A: Any,
    *,
    x0: Any | None = None,
    max_iter: int = 200,
    tol: float = 1e-12,
    seed: int = 1,
    keep: bool = True,
) -> PowerResult:
    r"""幂法：:math:`x_{k+1} = A x_k/\|Ax_k\|_2`，用 Rayleigh 商估 :math:`\lambda_1`。

    收敛定理（``chapters/1.4/notes.tex`` 定理 1）：设 :math:`|\lambda_1|>|\lambda_2|\ge\cdots`,
    :math:`x_0 = \sum_j c_j v_j` 且 :math:`c_1\ne 0`，则

    .. math::
        \|x_k - \sigma_k v_1\|_2 = O\!\left(\left|\frac{\lambda_2}{\lambda_1}\right|^k\right),
        \qquad
        |\rho(x_k)-\lambda_1| = O\!\left(\left|\frac{\lambda_2}{\lambda_1}\right|^{2k}\right)

    （第二个式子就是"Rayleigh 商比向量快一倍"）。``ratio`` 字段给出**实测**收敛因子
    :math:`\hat r = (\|x_{k+1}-x_k\|/\|x_k-x_{k-1}\|)` 的几何平均，可与
    :math:`|\lambda_2/\lambda_1|` 对拍（A 类断言）。

    复主特征值时**不会**收敛到单个实向量：实向量在 :math:`\mathrm{span}\{v,\bar v\}` 里
    绕圈，表现为 :math:`x_k` 与 :math:`x_{k+1}` 的夹角不趋于 0（``angle_step`` 平台）。
    此时 ``is_complex_pair=True``，``eigenvalue`` 只报 Rayleigh 商的实部——**这是有意的**：
    图上必须能看出"这个数不是特征值"，而不是悄悄给个好看的数字。
    """
    arr = _as_square(A)
    n = arr.shape[0]
    if x0 is None:
        x = normalize(_random_start(n, seed=seed, complex_=np.iscomplexobj(arr)))
    else:
        x = normalize(np.asarray(x0).reshape(-1))
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError(f"max_iter 至少为 1，得到 {max_iter}")

    vectors = np.zeros((max_iter + 1, n), dtype=x.dtype)
    rayleigh = np.zeros(max_iter + 1, dtype=float)
    residual = np.zeros(max_iter + 1, dtype=float)
    angle_step = np.zeros(max_iter + 1, dtype=float)
    vectors[0] = x
    rayleigh[0] = float(np.real(rayleigh_quotient(arr, x)))
    residual[0] = eigen_residual(arr, rayleigh[0], x)

    converged = False
    used = 0
    for k in range(max_iter):
        y = arr @ x
        ny = _norm2(y)
        if ny == 0.0:  # A x = 0：主特征值为 0（或起点落在零空间里）
            used = k
            break
        x_new = y / ny
        # 夹角：用 |<x_new, x>| 反余弦；复主特征值时它会停在 O(1) 而不是 -> 0
        inner = abs(complex(np.vdot(x, x_new)))
        inner = min(1.0, max(0.0, float(inner)))
        angle_step[k + 1] = float(np.arccos(inner))
        x = x_new
        vectors[k + 1] = x
        rayleigh[k + 1] = float(np.real(rayleigh_quotient(arr, x)))
        residual[k + 1] = eigen_residual(arr, rayleigh[k + 1], x)
        used = k + 1
        scale = max(1.0, abs(rayleigh[k + 1]))
        if residual[k + 1] <= tol * scale:
            converged = True
            break

    m = used + 1
    if not keep:  # 只要结果不要历史（预计算里省内存）
        vectors = vectors[[used]]
    else:
        vectors = vectors[:m]
    rq = rayleigh[:m]
    rs = residual[:m]
    ang = angle_step[:m]
    # 实测收敛因子：用残差序列的几何平均比值（比向量差更稳，且与 |λ2/λ1| 同阶）
    pos = rs[rs > 0.0]
    if pos.size >= 3:
        ratios = pos[1:] / pos[:-1]
        ratios = ratios[1:] if ratios.size > 1 else ratios  # 丢掉第一步的瞬态
    else:
        ratios = np.zeros(0, dtype=float)
    ratio = float(np.exp(np.mean(np.log(ratios[ratios > 0.0])))) if ratios.size else float("nan")
    # 复主特征值的标志：最后 1/4 段的夹角平台仍大于 1e-3 弧度
    tail = ang[max(1, ang.size - max(2, ang.size // 4)) :]
    plateau = float(np.mean(tail)) if tail.size else 0.0
    is_complex = bool(plateau > 1e-3)
    return PowerResult(
        eigenvalue=float(rq[-1]),
        vector=np.asarray(x),
        vectors=np.asarray(vectors),
        rayleigh=rq,
        residual=rs,
        angle_step=ang,
        n_iter=used,
        converged=converged,
        ratios=np.asarray(ratios),
        ratio=ratio,
        is_complex_pair=is_complex,
    )


def dominant_eigenpair(A: Any, **kw: Any) -> tuple[float, np.ndarray]:
    """便捷入口：返回 :math:`(\\lambda_1, v_1)`（幂法）。"""
    res = power_iteration(A, **kw)
    return res.eigenvalue, res.vector


# --------------------------------------------------------------------------- #
# 反迭代与位移
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InverseResult:
    """反迭代的一次运行：Rayleigh 商序列 + 残差 + 实测收敛因子。"""

    eigenvalue: float
    vector: np.ndarray
    rayleigh: np.ndarray
    residual: np.ndarray
    n_iter: int
    converged: bool
    ratio: float
    shift: float
    singular: bool


def inverse_iteration(
    A: Any,
    *,
    sigma: float = 0.0,
    x0: Any | None = None,
    max_iter: int = 100,
    tol: float = 1e-13,
    seed: int = 2,
) -> InverseResult:
    r"""反迭代（带位移）：解 :math:`(A-\sigma I)x_{k+1} = x_k` 再归一化。

    它把**离 :math:`\sigma` 最近的**特征值变成主特征值：

    .. math::
        \left|\frac{\lambda_{\text{next}}-\sigma}{\lambda_{\text{near}}-\sigma}\right|^k
        \longrightarrow 0 ,

    所以 :math:`\sigma` 取得越准，收敛越快（特征值估计每步平方级改善，即
    Rayleigh 商迭代）。反迭代的**代价**是每步一次线性求解——这里调用
    :func:`numcore.linalg.lu.lu_decompose` + :func:`numcore.linalg.lu.solve_lu`
    （自研 LU，P2/P3 允许：它不是"成品特征值求解器"）。

    若 :math:`A-\\sigma I` 奇异（:math:`\sigma` 恰好是特征值），LU 会报
    ``ZeroDivisionError``——这里捕获并转成 ``singular=True``（这在教学上是**好消息**：
    "求不动"本身就是特征值在那个位置上的证据）。
    """
    from numcore.linalg.lu import lu_decompose, solve_lu

    arr = _as_square(A)
    n = arr.shape[0]
    x = normalize(_random_start(n, seed=seed, complex_=False) if x0 is None else np.asarray(x0).reshape(-1))
    M = arr - float(sigma) * np.eye(n)
    max_iter = int(max_iter)
    rayleigh = np.zeros(max_iter + 1, dtype=float)
    residual = np.zeros(max_iter + 1, dtype=float)
    rayleigh[0] = float(np.real(rayleigh_quotient(arr, x)))
    residual[0] = eigen_residual(arr, rayleigh[0], x)

    singular = False
    converged = False
    used = 0
    fac = lu_decompose(M, pivot="partial")
    if fac.singular or any(abs(float(fac.U[i, i])) == 0.0 for i in range(n)):
        singular = True
    for k in range(max_iter):
        try:
            y = solve_lu(fac, x)
        except ZeroDivisionError:
            singular = True
            used = k
            break
        ny = _norm2(y)
        # (A−σI) 病态到极点时 y 会溢出成 inf/NaN：诚实地停下并标 singular
        if ny == 0.0 or not np.isfinite(ny):
            singular = True
            used = k
            break
        x = y / ny
        used = k + 1
        rayleigh[k + 1] = float(np.real(rayleigh_quotient(arr, x)))
        residual[k + 1] = eigen_residual(arr, rayleigh[k + 1], x)
        if residual[k + 1] <= tol * max(1.0, abs(rayleigh[k + 1])):
            converged = True
            break

    m = used + 1
    rq = rayleigh[:m]
    rs = residual[:m]
    pos = rs[rs > 0.0]
    ratio = float("nan")
    if pos.size >= 3:
        rr = pos[1:] / pos[:-1]
        rr = rr[1:] if rr.size > 1 else rr
        good = rr[rr > 0.0]
        if good.size:
            ratio = float(np.exp(np.mean(np.log(good))))
    return InverseResult(
        eigenvalue=float(rq[-1]),
        vector=np.asarray(x),
        rayleigh=rq,
        residual=rs,
        n_iter=used,
        converged=converged,
        ratio=ratio,
        shift=float(sigma),
        singular=singular,
    )


# --------------------------------------------------------------------------- #
# Hessenberg 化（QR 迭代的前置：把 O(n^3) 一步降到 O(n^2)）
# --------------------------------------------------------------------------- #
def hessenberg_reduce(A: Any) -> tuple[np.ndarray, np.ndarray]:
    r"""Householder 相似变换把 :math:`A` 化为上 Hessenberg 形 :math:`H = Q^{\mathsf T}AQ`。

    :math:`h_{ij} = 0\\ (i > j+1)`：每一步用 Householder 反射 :math:`P_k` 把第 :math:`k`
    列的第 :math:`k+2..n-1` 个元素清零，左右各乘一次（相似变换，**特征值不变**，
    这是它区别于 LU/QR 分解的关键）。整体代价 :math:`\approx 4n^3/3`，
    与 QR 迭代的 :math:`O(n^2)`/步 相比是一次性投入。

    为什么 QR 迭代非要先化 Hessenberg：每一步 QR 分解的代价从 :math:`O(n^3)` 降到
    :math:`O(n^2)`，而迭代步数基本不变。
    """
    H = _as_square(A)
    n = H.shape[0]
    Q = np.eye(n, dtype=H.dtype)
    for k in range(n - 2):
        x = H[k + 1 :, k]
        nx = _norm2(x)
        if nx == 0.0:
            continue
        # Householder：v = x + sign(x_1)‖x‖e_1（sign 取 x_1 的相位，避免相减抵消）
        phase = x[0] / abs(x[0]) if abs(x[0]) > 0.0 else 1.0
        v = x.copy()
        v[0] += phase * nx
        nv = _norm2(v)
        if nv == 0.0:
            continue
        v = v / nv
        # H ← (I − 2vv^H) H（只作用在第 k+1.. 行）与 H ← H (I − 2vv^H)（只作用在第 k+1.. 列）
        w = v.conj() @ H[k + 1 :, :]
        H[k + 1 :, :] -= 2.0 * np.outer(v, w)
        w2 = H[:, k + 1 :] @ v
        H[:, k + 1 :] -= 2.0 * np.outer(w2, v.conj())
        qa = Q[:, k + 1 :] @ v
        Q[:, k + 1 :] -= 2.0 * np.outer(qa, v.conj())
    # 数值上会留下 1e-17 级的"零"：显式清零，让结构可断言
    for i in range(2, n):
        H[i, : i - 1] = 0.0
    return H, Q


# --------------------------------------------------------------------------- #
# QR 迭代
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QRResult:
    """QR 迭代的结果与历史。

    ``history`` 是每一步的次对角元**绝对值**（长度 n-1 的向量）序列，
    形状 ``(iters+1, n-1)``——它就是要画成热力图的那张图（``qr_schur``）。
    ``ritz`` 是每一步对特征值的当前估计（Schur 形对角块的特征值，按 |·| 排序）。
    """

    H: np.ndarray
    Q: np.ndarray
    eigenvalues: np.ndarray
    history: np.ndarray
    ritz: np.ndarray
    n_iter: int
    converged: bool
    shift: str
    n_deflations: int
    block_sizes: tuple[int, ...] = field(default=())


def _givens(a: complex, b: complex) -> tuple[complex, complex]:
    r"""Givens 参数 :math:`(c, s)`：:math:`c^2+|s|^2=1` 且
    :math:`\begin{pmatrix}c&s\\-\bar s&c\end{pmatrix}\binom{a}{b}=\binom{r}{0}`。

    展开第一行得 :math:`ca+sb=r`、取模平方得 :math:`|r|=t\sqrt{(|a|/t)^2+(|b|/t)^2}`
    （:math:`t=|a|+|b|`，先缩放再开方，避免极端尺度上溢），于是

    .. math::
        c = \frac{|a|}{|r|},\qquad s = \frac{\bar b}{|r|}\,\frac{a}{|a|}.

    **注意**：这里必须用 :math:`a/|a|` 的相位（而不是 :math:`|a|/a`）。
    写成后者时 :math:`c\bar a+s\bar b` 会多出一个 :math:`(a/|a|)^2` 因子——
    在 :math:`a<0` 时它是 :math:`-r`，直接毁掉后面所有的相似变换
    （实测：这一步把 4×4 Laplacian 的 RQ 变成非 Hessenberg，特征值全错）。
    """
    if b == 0.0:
        return (1.0 + 0.0j, 0.0 + 0.0j)
    if a == 0.0:
        return (0.0 + 0.0j, 1.0 + 0.0j)
    t = abs(a) + abs(b)
    r = t * np.sqrt((abs(a) / t) ** 2 + (abs(b) / t) ** 2)
    c = abs(a) / r
    s = (np.conj(b) / r) * (a / abs(a))
    return (complex(c), complex(s))


def _ensure_complex(H: np.ndarray) -> None:
    r"""把矩阵**原地升级**为复数（若它还是实数组）。

    这一步不是洁癖：Givens 的 :math:`s` 一般是复数，而 ``real_array[i,:] = complex_expr``
    会被 numpy **静默截断成实部**（只发一条 ``ComplexWarning``）。实测后果：
    :math:`RQ+\mu I` 不再是相似变换，6×6 Laplacian 上特征值误差 2.2（而正确实现是 1e-15）。
    所以任何会写入复数的例程都必须先经过这里。
    """
    if np.iscomplexobj(H):
        return
    promoted = np.asarray(H, dtype=complex)
    H.dtype = promoted.dtype
    H.real[:] = promoted.real
    H.imag[:] = promoted.imag


def _apply_givens_left(H: np.ndarray, c: complex, s: complex, i: int, j: int) -> None:
    r"""左乘 Givens 旋转 :math:`G=\begin{pmatrix}c&s\\-\bar s&c\end{pmatrix}`（i、j 两行），原地修改。

    :math:`(r_i,r_j)\mapsto(c\,r_i+s\,r_j,\ -\bar s\,r_i+c\,r_j)`。
    """
    _ensure_complex(H)
    ri = H[i, :].copy()
    rj = H[j, :].copy()
    H[i, :] = c * ri + s * rj
    H[j, :] = -np.conj(s) * ri + c * rj


def _apply_givens_right(H: np.ndarray, c: complex, s: complex, i: int, j: int) -> None:
    r"""右乘 :math:`G^{\mathsf H}`（i、j 两列），原地修改。

    右乘时**新列 = 旧列的线性组合，系数取自 :math:`G^{\mathsf H}` 的列**：

    .. math::
        \begin{pmatrix}c_i' & c_j'\end{pmatrix}
        = \begin{pmatrix}c_i & c_j\end{pmatrix}
          \begin{pmatrix}c & -s\\ \bar s & \bar c\end{pmatrix}
        \;\Longrightarrow\;
        c_i' = c\,c_i + \bar s\,c_j,\qquad
        c_j' = -s\,c_i + \bar c\,c_j .

    这一步的共轭位置极易写反（本仓库真的写反过）：写反之后
    :math:`RQ+\mu I` 不再是相似变换，特征值整体跑偏（实测 6×6 Laplacian 误差 2.2，
    而正确实现是 1e-15）。判据很简单——每步验证
    :math:`\|H_{\text{new}} - G^{\mathsf H}HG\| = 0`，见 ``tests/1.4``。
    """
    _ensure_complex(H)
    ci = H[:, i].copy()
    cj = H[:, j].copy()
    H[:, i] = c * ci + np.conj(s) * cj
    H[:, j] = -s * ci + np.conj(c) * cj


def _wilkinson_shift(H: np.ndarray, m: int) -> complex:
    r"""Wilkinson 位移：:math:`H` 的尾部 2×2 块 :math:`\begin{pmatrix}a&b\\c&d\end{pmatrix}`
    的两个特征值中**离** :math:`d` **较近**的那个。

    它比"取 :math:`h_{mm}`（Rayleigh 位移）"更稳：Rayleigh 位移在实矩阵的复特征值处
    会来回震荡（两个共轭特征值同样近），Wilkinson 位移至少保证不慢于它，
    并且对 2×2 复块也给出正确的"块收敛"行为。
    """
    a = complex(H[m - 2, m - 2])
    b = complex(H[m - 2, m - 1])
    c = complex(H[m - 1, m - 2])
    d = complex(H[m - 1, m - 1])
    tr = a + d
    det = a * d - b * c
    disc = tr * tr - 4.0 * det
    sq = np.sqrt(complex(disc))
    mu1 = 0.5 * (tr + sq)
    mu2 = 0.5 * (tr - sq)
    return mu1 if abs(mu1 - d) <= abs(mu2 - d) else mu2


def _schur_eigenvalues_from_hessenberg(H: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    r"""从（准）上三角的 Hessenberg 形读出特征值：1×1 块给实特征值，
    2×2 块给一对共轭复特征值（解 :math:`\lambda^2 - \mathrm{tr}\lambda + \det = 0`）。"""
    n = H.shape[0]
    vals: list[complex] = []
    i = 0
    scale = max(1.0, _max_abs(H))
    while i < n:
        if i == n - 1 or abs(complex(H[i + 1, i])) <= tol * scale:
            vals.append(complex(H[i, i]))
            i += 1
        else:
            a = complex(H[i, i])
            b = complex(H[i, i + 1])
            c = complex(H[i + 1, i])
            d = complex(H[i + 1, i + 1])
            tr = a + d
            det = a * d - b * c
            disc = tr * tr - 4.0 * det
            sq = np.sqrt(complex(disc))
            vals.append(0.5 * (tr + sq))
            vals.append(0.5 * (tr - sq))
            i += 2
    out = np.asarray(vals, dtype=complex)
    if all(abs(v.imag) <= 1e-13 * max(1.0, abs(v)) for v in out):
        return np.asarray([float(v.real) for v in out], dtype=float)
    return out


def _subdiag_abs(H: np.ndarray) -> np.ndarray:
    """次对角元的绝对值向量 :math:`(|h_{21}|,|h_{32}|,\\dots,|h_{n,n-1}|)`。"""
    n = H.shape[0]
    if n < 2:
        return np.zeros(0, dtype=float)
    idx = np.arange(1, n)
    return np.abs(np.asarray(H, dtype=complex)[idx, idx - 1]).astype(float)


def qr_iteration(
    A: Any,
    *,
    shift: str = "wilkinson",
    max_iter: int = 500,
    tol: float = 1e-14,
    hessenberg: bool = True,
) -> QRResult:
    r"""带位移的 QR 迭代：:math:`H_k-\mu_kI = Q_kR_k,\ H_{k+1}=R_kQ_k+\mu_kI`。

    ``shift`` 取 ``"none"`` / ``"rayleigh"`` / ``"wilkinson"``：

    * **无位移**：:math:`H_k \to Q^{\mathsf T}HQ` 使次对角元按
      :math:`|\lambda_{i+1}/\lambda_i|` 衰减，但**当两个特征值模相等时完全不收敛**
      （复共轭对、或 :math:`\pm\lambda`）——这是本章要演出来的失败模式；
    * **Rayleigh 位移** :math:`\mu = h_{mm}`：局部三次收敛（对对称矩阵），
      对非对称矩阵可能不收敛（复特征值时来回跳）；
    * **Wilkinson 位移**：取尾部 2×2 块靠近 :math:`h_{mm}` 的特征值，是工业实现
      （LAPACK dhseqr）的选择；遇到复共轭对时表现为**块收敛**（次对角元停在
      :math:`O(\sqrt{u})` 而不是 0）。

    ``history`` 记录每一步的次对角元绝对值——注意它记录的是**整个矩阵**的次对角元，
    已经收敛（被 deflate 掉）的部分会一直保持 0，所以热力图上"从下往上变黑"的
    过程就是收敛过程本身。
    """
    if shift not in ("none", "rayleigh", "wilkinson"):
        raise ValueError(f"未知位移策略 {shift!r}")
    A0 = _as_square(A)
    n = A0.shape[0]
    if hessenberg:
        H, Qacc = hessenberg_reduce(A0)
    else:
        H, Qacc = np.array(A0, copy=True), np.eye(n, dtype=A0.dtype)
    H = np.asarray(H, dtype=complex)
    Qacc = np.asarray(Qacc, dtype=complex)

    max_iter = int(max_iter)
    history = [_subdiag_abs(H)]
    ritz = [np.sort_complex(_schur_eigenvalues_from_hessenberg(H))]
    n_deflations = 0
    block_sizes: list[int] = []
    m = n  # 当前活跃块的大小（尾部 m×m）
    it = 0
    converged = False
    scale = max(1.0, _max_abs(H))
    while m > 1 and it < max_iter:
        it += 1
        # ---- deflation 检查：尾部次对角元是否已经小到可以切开 ----
        if abs(H[m - 1, m - 2]) <= tol * scale:
            H[m - 1, m - 2] = 0.0
            n_deflations += 1
            block_sizes.append(1)
            m -= 1
            history.append(_subdiag_abs(H))
            ritz.append(np.sort_complex(_schur_eigenvalues_from_hessenberg(H)))
            continue
        if m > 2 and abs(H[m - 2, m - 3]) <= tol * scale:
            H[m - 2, m - 3] = 0.0
            n_deflations += 1
            block_sizes.append(2)
            m -= 2
            history.append(_subdiag_abs(H))
            ritz.append(np.sort_complex(_schur_eigenvalues_from_hessenberg(H)))
            continue
        # ---- 只对活跃块做一步 QR，并把这步的旋转累计进 Qacc（Schur 向量）----
        if shift == "none":
            mu = 0.0 + 0.0j
        elif shift == "rayleigh":
            mu = complex(H[m - 1, m - 1])
        else:
            mu = _wilkinson_shift(H, m) if m >= 2 else complex(H[m - 1, m - 1])
        sub = np.asarray(H[:m, :m], dtype=complex).copy()
        Hs, rot = _qr_step_hessenberg_with_rotations(sub, mu)
        H[:m, :m] = Hs
        # Qacc ← Qacc · diag(G^H, I)：Schur 向量 = 累计的正交因子
        for k, (c, s) in enumerate(rot):
            ci = Qacc[:, k].copy()
            cj = Qacc[:, k + 1].copy()
            Qacc[:, k] = ci * c + cj * (-np.conj(s))
            Qacc[:, k + 1] = ci * s + cj * c
        history.append(_subdiag_abs(H))
        ritz.append(np.sort_complex(_schur_eigenvalues_from_hessenberg(H)))
    converged = (m <= 1) or (m == 2 and abs(H[1, 0]) <= tol * scale)
    if not converged:
        # 达到迭代上限：把残留的未收敛部分如实记进 block_sizes（不许假装收敛）
        i = 1
        while i < m:
            if abs(H[i, i - 1]) > tol * scale:
                block_sizes.append(2)
                i += 2
            else:
                block_sizes.append(1)
                i += 1
    vals = _schur_eigenvalues_from_hessenberg(H)
    return QRResult(
        H=H,
        Q=Qacc,
        eigenvalues=vals,
        history=np.asarray(history, dtype=float),
        ritz=np.asarray(ritz),
        n_iter=it,
        converged=bool(converged),
        shift=shift,
        n_deflations=n_deflations,
        block_sizes=tuple(block_sizes),
    )


def _qr_step_hessenberg_with_rotations(H: np.ndarray, mu: complex) -> tuple[np.ndarray, list[tuple[complex, complex]]]:
    r"""一步带位移 QR：:math:`H-\mu I = QR`，返回 :math:`(RQ+\mu I,\ \{(c_k,s_k)\})`。

    对 Hessenberg 输入，:math:`R` 仍是上三角、:math:`Q` 仍是 Hessenberg 形，
    所以 :math:`RQ+\mu I` 仍是上 Hessenberg——**这是 QR 迭代能保持结构的全部原因**
    （也是为什么必须先做 Hessenberg 化：每步 :math:`O(n^2)` 而不是 :math:`O(n^3)`）。
    返回旋转序列是为了让调用方能累计 Schur 向量 :math:`Q_{\text{acc}}\leftarrow Q_{\text{acc}}G^{\mathsf H}`。
    """
    m = H.shape[0]
    if m < 2:
        return np.array(H, dtype=complex, copy=True), []
    sub = np.asarray(H, dtype=complex) - mu * np.eye(m, dtype=complex)
    R = np.array(sub, copy=True)
    rotations: list[tuple[complex, complex]] = []
    for k in range(m - 1):
        c, s = _givens(R[k, k], R[k + 1, k])
        _apply_givens_left(R, c, s, k, k + 1)
        rotations.append((c, s))
    out = np.array(R, copy=True)
    for idx, (c, s) in enumerate(rotations):
        # RQ = R·G_1^H G_2^H … G_{m-1}^H：按**正序**逐个右乘 G_k^H
        _apply_givens_right(out, c, s, idx, idx + 1)
    return out + mu * np.eye(m, dtype=complex), rotations


def qr_subdiagonal_history(A: Any, **kw: Any) -> np.ndarray:
    """只要次对角元历史（``qr_iteration(...).history`` 的薄封装）。"""
    return qr_iteration(A, **kw).history


def schur_eigenvalues(A: Any, **kw: Any) -> np.ndarray:
    """QR 迭代给出的特征值（1×1/2×2 块），按模降序排列。

    返回实数组（全部特征值为实）或复数组。实矩阵的复特征值成对出现，
    ``sort_complex`` 把它们排在一起。
    """
    vals = qr_iteration(A, **kw).eigenvalues
    arr = np.asarray(vals)
    order = np.argsort(-np.abs(arr))
    return arr[order]


def dense_eigenvalues(A: Any, **kw: Any) -> np.ndarray:
    """稠密实/复矩阵的全部特征值（Hessenberg + 带位移 QR）= :func:`schur_eigenvalues`。"""
    return schur_eigenvalues(A, **kw)


# --------------------------------------------------------------------------- #
# Krylov 子空间：Lanczos（对称）与 Arnoldi（一般）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LanczosResult:
    """Lanczos 三项递推的结果。

    ``alpha``/``beta`` 是三对角矩阵 :math:`T_m` 的对角/次对角元，
    ``V`` 是正交基（列），``orthogonality`` 是 :math:`\\|V^{\\mathsf T}V-I\\|_F` 的实测值
    （浮点下它会随步数增长，这正是"重正交化"存在的理由）。
    """

    alpha: np.ndarray
    beta: np.ndarray
    V: np.ndarray
    orthogonality: float
    n_iter: int
    breakdown: bool

    def tridiagonal(self) -> np.ndarray:
        """:math:`T_m`（对称三对角）显式矩阵。"""
        m = self.alpha.size
        T = np.zeros((m, m), dtype=float)
        idx = np.arange(m)
        T[idx, idx] = self.alpha
        if m > 1:
            T[idx[:-1], idx[:-1] + 1] = self.beta[1:m]
            T[idx[:-1] + 1, idx[:-1]] = self.beta[1:m]
        return T


def lanczos(
    A: Any,
    *,
    m: int | None = None,
    q0: Any | None = None,
    reorthogonalize: bool = True,
    tol: float = 1e-14,
    seed: int = 3,
) -> LanczosResult:
    r"""Lanczos 三项递推：对称矩阵的 Krylov 正交基。

    .. math::
        \beta_{j+1}v_{j+1} = Av_j - \alpha_jv_j - \beta_jv_{j-1},
        \qquad \alpha_j = v_j^{\mathsf T}Av_j .

    与 Arnoldi 相比它只需存三个向量（:math:`O(n)` 内存），代价是**只在对称矩阵上成立**。
    浮点下三项递推的正交性会**丢失**（Paige 定理：正交性丢失与 Ritz 值收敛同时发生），
    所以 ``reorthogonalize=True`` 时做一次完全重正交化（"Lanczos 要正交化两次"），
    ``orthogonality`` 字段把这个差别如实记下来。
    """
    arr = _as_square(A)
    n = arr.shape[0]
    if np.iscomplexobj(arr):
        # 对称 ≠ Hermite：这里只支持实对称（复 Hermite 请走 arnoldi）
        raise ValueError("lanczos 只支持实对称矩阵（复 Hermite 请用 arnoldi）")
    m = n if m is None else int(m)
    if m < 1 or m > n:
        raise ValueError(f"m 必须在 [1, {n}] 内，得到 {m}")
    q = normalize(_random_start(n, seed=seed) if q0 is None else np.asarray(q0).reshape(-1))
    V = np.zeros((n, m), dtype=float)
    alpha = np.zeros(m, dtype=float)
    beta = np.zeros(m + 1, dtype=float)
    V[:, 0] = q
    q_prev = np.zeros(n, dtype=float)
    b = 0.0
    breakdown = False
    used = 0
    for j in range(m):
        used = j + 1
        w = arr @ V[:, j]
        alpha[j] = float(V[:, j] @ w)
        w = w - alpha[j] * V[:, j] - b * q_prev
        if reorthogonalize:
            # 完全重正交化：一次不够，理论上要两次；这里一次 + 二次修正已经能压到 u 量级
            w = w - V[:, : j + 1] @ (V[:, : j + 1].T @ w)
            w = w - V[:, : j + 1] @ (V[:, : j + 1].T @ w)
        b = _norm2_real(w)
        beta[j + 1] = b
        if b <= tol * max(1.0, abs(alpha[j])) or j == m - 1:
            breakdown = b <= tol * max(1.0, abs(alpha[j]))
            if j < m - 1:
                V = V[:, : j + 1]
                alpha = alpha[: j + 1]
                beta = beta[: j + 2]
            break
        q_prev = V[:, j].copy()
        V[:, j + 1] = w / b
    orth = float(np.sqrt(np.sum((V.T @ V - np.eye(V.shape[1])) ** 2)))
    return LanczosResult(
        alpha=alpha[: V.shape[1]],
        beta=beta[: V.shape[1] + 1],
        V=V,
        orthogonality=orth,
        n_iter=used,
        breakdown=bool(breakdown),
    )


@dataclass(frozen=True)
class ArnoldiResult:
    """Arnoldi 过程的结果：:math:`AV_m = V_mH_m + h_{m+1,m}v_{m+1}e_m^{\\mathsf T}`。"""

    H: np.ndarray
    V: np.ndarray
    h_next: float
    orthogonality: float
    n_iter: int
    breakdown: bool

    def ritz(self) -> np.ndarray:
        """Ritz 值（:math:`H_m` 的特征值，用本模块自研的 QR 迭代求）。

        ``self.H`` 是 :math:`(m+1)\times m` 的"扩展" Hessenberg（最后一行是
        :math:`h_{m+1,m}`，它**不属于** :math:`H_m`），所以这里先取前 m 行。
        """
        Hm = np.asarray(self.H)
        return schur_eigenvalues(Hm[: Hm.shape[1], :])


def arnoldi(
    A: Any,
    *,
    m: int | None = None,
    q0: Any | None = None,
    reorthogonalize: bool = True,
    tol: float = 1e-14,
    seed: int = 4,
) -> ArnoldiResult:
    r"""Arnoldi 过程：一般矩阵的 Krylov 正交基（:math:`\\mathcal{K}_m = \\mathrm{span}\\{q_0,Aq_0,\\dots,A^{m-1}q_0\\}`）。

    它把 :math:`A` 在 Krylov 子空间上**正交投影**成上 Hessenberg 矩阵 :math:`H_m`：
    :math:`H_m` 的 Ritz 值就是 :math:`A` 的特征值在 :math:`\\mathcal{K}_m` 上的最佳近似
    （残差有闭式 :math:`|h_{m+1,m}|\\,|e_m^{\\mathsf T}y|`，见 :func:`ritz_residuals`）。

    用**改进 Gram–Schmidt + 重正交化**（DGKS 判据的简化版：无条件做两次），
    因为 Krylov 基在非正规矩阵上丢正交性极快——这会直接毁掉 Ritz 值的可信度。
    """
    arr = _as_square(A)
    n = arr.shape[0]
    m = n if m is None else int(m)
    if m < 1 or m > n:
        raise ValueError(f"m 必须在 [1, {n}] 内，得到 {m}")
    dtype = complex if np.iscomplexobj(arr) else float
    q = normalize(_random_start(n, seed=seed, complex_=np.iscomplexobj(arr)) if q0 is None else np.asarray(q0).reshape(-1))
    V = np.zeros((n, m + 1), dtype=dtype)
    H = np.zeros((m + 1, m), dtype=dtype)
    V[:, 0] = q
    breakdown = False
    used = 0
    for j in range(m):
        used = j + 1
        w = arr @ V[:, j]
        for _ in range(2 if reorthogonalize else 1):
            proj = V[:, : j + 1].conj().T @ w
            H[: j + 1, j] += proj
            w = w - V[:, : j + 1] @ proj
        h_next = _norm2(w)
        H[j + 1, j] = h_next
        if h_next <= tol * max(1.0, _max_abs(H)):
            breakdown = True
            break
        V[:, j + 1] = w / h_next
    k = used
    h_next = float(abs(H[k, k - 1])) if k >= 1 else 0.0
    if breakdown:
        # AV_k = V_k H_k（不变子空间）：H_k 取 k×k，V_k 取 k 列
        Vk = V[:, :k]
        Hk = H[:k, :k]
    else:
        # 正常情况：H_k 是 (k+1)×k 的上 Hessenberg（下一列 e_{k+1} 还在）
        Vk = V[:, : k + 1]
        Hk = H[: k + 1, :k]
    orth = float(np.sqrt(np.sum(np.abs(Vk.conj().T @ Vk - np.eye(Vk.shape[1])) ** 2)))
    return ArnoldiResult(
        H=Hk,
        V=Vk,
        h_next=h_next,
        orthogonality=orth,
        n_iter=k,
        breakdown=bool(breakdown),
    )


def ritz_values(H: Any) -> np.ndarray:
    """Ritz 值 = 投影矩阵 :math:`H_m` 的特征值（自研 QR 迭代）。"""
    return schur_eigenvalues(_as_square(H))


def ritz_residuals(H: Any, V: Any, h_next: float) -> np.ndarray:
    r"""Arnoldi Ritz 对的残差 :math:`\|AVy-\theta Vy\|_2 = |h_{m+1,m}|\,|e_m^{\mathsf T}y|`。

    这是 Arnoldi 最漂亮的性质之一：残差**不用真的算** :math:`AVy`，只要 :math:`H_m`
    的最后一个特征向量分量。它也是残差型特征值界
    :math:`\min_i|\theta-\lambda_i| \le \|r\|_2` 的直接来源（任何特征值都落在
    :math:`\bigcup_i [\theta-\|r\|,\theta+\|r\|]` 里——Bauer–Fike 的 Arnoldi 版本）。
    """
    Hm = _as_square(H)
    m = Hm.shape[0]
    vals, vecs = _eigenvectors_of_small(Hm)
    out = np.zeros(m, dtype=float)
    for i in range(m):
        y = vecs[:, i]
        ny = _norm2(y)
        if ny == 0.0:
            out[i] = float("nan")
            continue
        out[i] = abs(float(h_next)) * abs(complex(y[-1])) / ny
    return out


def _complex_lu_solve(M: np.ndarray, b: np.ndarray) -> np.ndarray:
    """带部分选主元的复数 LU 求解（自研，供 :func:`_eigenvectors_of_small` 的反迭代用）。

    之所以不复用 :func:`numcore.linalg.pseudospec.complex_lu`：那会造成
    ``eigen -> pseudospec -> eigen`` 的循环导入。这里只保留"解一个方程"所需的最小实现。
    """
    A = np.array(M, dtype=complex, copy=True)
    n = A.shape[0]
    piv = np.arange(n)
    for k in range(n):
        p = k + int(np.argmax(np.abs(A[k:, k])))
        if p != k:
            A[[k, p], :] = A[[p, k], :]
            piv[[k, p]] = piv[[p, k]]
        d = A[k, k]
        if abs(d) < 1e-300:
            continue
        for i in range(k + 1, n):
            f = A[i, k] / d
            A[i, k] = f
            A[i, k + 1 :] -= f * A[k, k + 1 :]
    y = np.asarray(b, dtype=complex).reshape(-1)[piv].copy()
    for i in range(1, n):
        y[i] -= A[i, :i] @ y[:i]
    for i in range(n - 1, -1, -1):
        y[i] = (y[i] - A[i, i + 1 :] @ y[i + 1 :]) / A[i, i] if abs(A[i, i]) > 1e-300 else 0.0
    return y


def _eigenvectors_of_small(H: Any) -> tuple[np.ndarray, np.ndarray]:
    r"""小矩阵的全部特征对：特征值走 QR 迭代，特征向量走**反迭代**（复数 LU）。

    为什么不用"Schur 形回代"：Schur 形 :math:`T` 是**上三角甚至对角**的，
    所以 :math:`(T-\lambda I)y=0` 的回代只会给出**单位基向量** :math:`e_k`
    ——那是 :math:`T` 的特征向量，不是 :math:`H` 的（实测 6x6 Laplacian 上
    残差 2.29，全部向量都是基向量）。真正的特征向量藏在累积的 :math:`Q` 里，
    而 :func:`qr_iteration` 目前不返回 :math:`Q`，所以这里改用反迭代：

    .. math::
        (H-\mu I)w_{k+1}=v_k,\qquad v_{k+1}=w_{k+1}/\|w_{k+1}\| ,\qquad \mu=\theta+\text{小扰动} .

    扰动 :math:`\mu` 的目的只是**避开精确奇异**（:math:`H-\theta I` 理论上是奇异的，
    但浮点 LU 未必报奇异，且精确奇异会让回代失去方向）；取
    :math:`\mu=\theta\,(1+10^{-10})`。反迭代在特征值分离良好时 1~2 步就收敛到
    :math:`10^{-15}` 量级；特征值成簇时它给出的是**簇内某个方向**，
    这对本章的用途（Ritz 残差诊断、伪谱外近似）足够，而 :func:`numcore.linalg.pseudospec`
    的 :math:`\sigma_{\min}` 走的是完全不同的路线（Hermite Jacobi），不依赖这个函数。
    """
    Hm = _as_square(H)
    n = Hm.shape[0]
    res = qr_iteration(Hm, shift="wilkinson", max_iter=400, tol=1e-15)
    vals = np.asarray(res.eigenvalues, dtype=complex)
    Hc = np.asarray(Hm, dtype=complex)
    vecs = np.zeros((n, n), dtype=complex)
    for k, lam in enumerate(vals):
        mu = complex(lam) * (1.0 + 1e-10)
        # 复数位移 ⇒ 必须用复数 LU（实 LU 会把 (H-μI) 的复数结构截断）
        Mc = Hc - mu * np.eye(n, dtype=complex)
        v = np.ones(n, dtype=complex) + 0.01 * np.arange(n) * 1j
        nv = _norm2(v)
        v = v / nv if nv > 0.0 else v
        for _it in range(60):
            w = _complex_lu_solve(Mc, v)
            nw = _norm2(w)
            if nw == 0.0 or not np.isfinite(nw):
                break
            v_new = w / nw
            if _norm2(v_new - v) <= 1e-15 * max(1.0, _norm2(v_new)):
                v = v_new
                break
            v = v_new
        # 相位固定：把最大分量转到正实轴（同一问题给确定性输出）
        idx = int(np.argmax(np.abs(v)))
        if abs(v[idx]) > 0.0:
            v = v * (np.conj(v[idx]) / abs(v[idx]))
        vecs[:, k] = v
    return vals, vecs


def cauchy_interlacing_bounds(eigenvalues: Any, k: int) -> tuple[float, float]:
    r"""Cauchy 交错定理给出的 Ritz 值上下界。

    对称矩阵 :math:`A` 的 Krylov 子空间 :math:`\\mathcal{K}_k` 上的第 :math:`j` 个 Ritz 值
    :math:`\\theta_j^{(k)}` 满足（:math:`\\lambda` 按降序）

    .. math::
        \lambda_j \;\ge\; \theta_j^{(k)} \;\ge\; \lambda_{j+n-k} .

    即"Ritz 值被夹在真特征值之间"：最小的 Ritz 值不会低于 :math:`\lambda_n`，
    最大的不会高于 :math:`\lambda_1`。返回 :math:`(\lambda_1 - \theta_{\max}` 的允许上界,
    :math:`\theta_{\min} - \lambda_n` 的允许上界) 对应的**真特征值边界**
    ``(lambda_min_allowed_low, lambda_max_allowed_high)`` 供断言直接比较。
    """
    lam = np.sort(np.asarray(eigenvalues, dtype=float).reshape(-1))[::-1]
    k = int(k)
    if k < 1 or k > lam.size:
        raise ValueError(f"k 必须在 [1, {lam.size}] 内，得到 {k}")
    return float(lam[-1]), float(lam[0])


# --------------------------------------------------------------------------- #
# Hermitian Jacobi 特征分解（sigma_min 的稳定路线）
# --------------------------------------------------------------------------- #
def hermitian_jacobi_eigh(
    A: Any,
    *,
    max_sweeps: int = 40,
    tol: float = 1e-16,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Hermite 矩阵的循环 Jacobi 特征分解，返回 ``(w 升序, V)``，:math:`A = V\,\mathrm{diag}(w)\,V^{\mathsf H}`。

    每个 :math:`(p,q)` 对用一次**酉**旋转 :math:`G` 把 :math:`a_{pq}` 打成零：

    .. math::
        \tau = \frac{a_{qq}-a_{pp}}{2|a_{pq}|},\qquad
        t = \frac{\operatorname{sign}\tau}{|\tau|+\sqrt{1+\tau^2}},\qquad
        c = \frac{1}{\sqrt{1+t^2}},\qquad
        s = c\,t\,\frac{a_{pq}}{|a_{pq}|} .

    **为什么不用它算 SVD**：对 :math:`M^{\mathsf H}M` 做正交化的代价是条件数被**平方**
    （:math:`\kappa(M^{\mathsf H}M)=\kappa(M)^2`），:math:`\kappa(M)\gtrsim 10^8` 之后
    最小的那个特征值就在舍入噪声里了（实测 :math:`\kappa=10^{10}` 时算出 0，
    而真值是 :math:`10^{-10}`）。所以本章只用它做**伪谱图的 :math:`\sigma_{\min}` 估计**
    （那里 :math:`\sigma_{\min}` 与 :math:`\|A\|` 的比值才是要画的东西，
    且图上低于噪声地板的区域会被标成"不可测"），
    需要高精度的单点 :math:`\sigma_{\min}` 时用 :func:`smallest_singular_value`
    （单边 Jacobi，不形成正规方程）或参考层。
    """
    H = np.asarray(A, dtype=complex)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"需要方阵，得到形状 {H.shape}")
    n = H.shape[0]
    M = np.array(H, dtype=complex, copy=True)
    V = np.eye(n, dtype=complex)
    # 只取 Hermite 部分（输入若带微小反 Hermite 噪声，这一步让它成为良定义问题）
    M = 0.5 * (M + M.conj().T)
    for _sweep in range(max(1, int(max_sweeps))):
        off = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = M[p, q]
                if apq == 0.0:
                    continue
                off = max(off, abs(apq))
                tau = (M[q, q].real - M[p, p].real) / (2.0 * abs(apq))
                t = np.sign(tau) / (abs(tau) + np.sqrt(1.0 + tau * tau)) if tau != 0.0 else 1.0
                c = 1.0 / np.sqrt(1.0 + t * t)
                s = c * t * (apq / abs(apq))
                # M ← G^H M G，V ← V G（行、列、累计向量各更新一次）
                col_p = M[:, p].copy()
                col_q = M[:, q].copy()
                M[:, p] = c * col_p - np.conj(s) * col_q
                M[:, q] = s * col_p + c * col_q
                row_p = M[p, :].copy()
                row_q = M[q, :].copy()
                M[p, :] = c * row_p - s * row_q
                M[q, :] = np.conj(s) * row_p + c * row_q
                vp = V[:, p].copy()
                vq = V[:, q].copy()
                V[:, p] = c * vp - np.conj(s) * vq
                V[:, q] = s * vp + c * vq
        scale = float(np.max(np.abs(np.diag(M)))) if n else 1.0
        if off <= tol * max(1.0, scale):
            break
    w = np.diag(M).real
    order = np.argsort(w)
    return np.asarray(w[order], dtype=float), np.asarray(V[:, order])


def smallest_singular_value_hermitian(M: Any, *, max_sweeps: int = 40) -> tuple[float, float]:
    r""":math:`\sigma_{\min}(M)` 与 :math:`\sigma_{\max}(M)` 的估计（走 :math:`M^{\mathsf H}M` 的 Hermite Jacobi）。

    返回 ``(smin, smax)``。**精度随** :math:`\kappa(M)` **退化**：条件数是平方关系，
    所以 :math:`\kappa(M)\gtrsim 10^{8}` 之后 ``smin`` 会被舍入噪声抬高到
    :math:`u\,\|M\|` 量级（实测 :math:`\kappa=10^{10}` 时返回 0，而 :math:`M^{\mathsf H}M`
    的特征值已经无法分辨）。要精确值请用 :func:`smallest_singular_value`（单边 Jacobi）。

    这里把"平方条件数"这件事如实写进 docstring，而不是假装它能算到 1e-16——
    ``chapters/1.4/pitfalls.md`` 里有一条专门讲这个。
    """
    A = np.asarray(M, dtype=complex)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"需要方阵，得到形状 {A.shape}")
    G = A.conj().T @ A
    w, _ = hermitian_jacobi_eigh(G, max_sweeps=max_sweeps)
    smin = float(np.sqrt(max(w[0], 0.0))) if w.size else 0.0
    smax = float(np.sqrt(max(w[-1], 0.0))) if w.size else 0.0
    return smin, smax


# --------------------------------------------------------------------------- #
# 单边 Jacobi SVD（自研的"真值"来源：伪谱的 sigma_min 靠它）
# --------------------------------------------------------------------------- #
def _as_real_embedding(M: np.ndarray) -> np.ndarray:
    r"""复矩阵 -> 实 :math:`2n\times2n` 嵌入 :math:`\begin{bmatrix}\Re M&-\Im M\\ \Im M&\Re M\end{bmatrix}`。

    这个嵌入把 :math:`\mathbb{C}^n` 上的乘法写成 :math:`\mathbb{R}^{2n}` 上的实线性映射
    （"把复数当二元实数向量"），于是 :math:`\sigma(M)` 的每个值在嵌入里**出现两次**——
    复矩阵的 SVD 由此可以完全交给**已验证到机器精度的实 Jacobi**，
    不必去调复旋转的相位参数。

    为什么值得这么做：复 Givens 的参数化里相位有**四**种合法写法
    （:math:`\alpha=e^{i\psi}` 放在哪一列、:math:`\psi` 与 :math:`\varphi` 谁是谁），
    写错时旋转不再是酉变换、迭代**发散**而不报错——本仓库在复数 2x2 上实测过
    "Gram 非对角元从 4.6 涨到 7.7" 的失败模式。嵌入把这类静默错误整类消掉。
    """
    A = np.asarray(M)
    if not np.iscomplexobj(A):
        return np.asarray(A, dtype=float)
    re = np.asarray(np.real(A), dtype=float)
    im = np.asarray(np.imag(A), dtype=float)
    return np.block([[re, -im], [im, re]])


def one_sided_jacobi_svd(
    A: Any,
    *,
    max_sweeps: int = 30,
    tol: float = 1e-15,
    compute_u: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    r"""单边 Jacobi SVD：只对列做正交变换，直到各列相互正交。

    原理（Golub & Van Loan, *Matrix Computations*, §8.6.1）：对 :math:`M\in\mathbb{C}^{n\times n}`，
    循环地选一对列 :math:`(i,j)` 并施加右乘酉变换 :math:`G`（i、j 两列的复 Givens），使

    .. math::
        (M G)^{\mathsf H}(M G)
        = \begin{pmatrix} \alpha & 0 \\ 0 & \beta \end{pmatrix}
        \quad\Longleftrightarrow\quad
        G^{\mathsf H}(M^{\mathsf H}M)G \text{ 在这两列上对角},

    所需参数只依赖两个内积：:math:`\alpha=\|m_i\|^2,\ \beta=\|m_j\|^2,\ \gamma=m_i^{\mathsf H}m_j`：

    .. math::
        \zeta = \frac{\beta-\alpha}{2|\gamma|},\qquad
        t = -\frac{\operatorname{sign}(\zeta)}{|\zeta|+\sqrt{1+\zeta^2}},\qquad
        c = \frac{1}{\sqrt{1+t^2}},\qquad s = c\,t\,\frac{\gamma}{|\gamma|} .

    收敛后第 :math:`i` 列的 2-范数就是奇异值 :math:`\sigma_i`（右奇异向量是 :math:`V` 的第 i 列）。
    **收敛是二次的**，实际 6–10 轮扫描就到机器精度；与"对 :math:`M^{\mathsf H}M` 用幂法/特征分解"
    相比，它的关键优点是**不形成正规方程**——:math:`\kappa(M)=10^{10}` 时后者要丢掉
    :math:`2\log_{10}\kappa=20` 位有效数字（这正是本章要讲的那种"平方放大"）。

    返回 ``(sigma, V, n_sweeps)``；``sigma`` 按降序。``compute_u=True`` 时返回
    ``(sigma, V, U)`` 三元组（:math:`M = U\,\mathrm{diag}(\sigma)\,V^{\mathsf H}`）。
    本仓库自用（1.4 章的伪谱、1.3 章的 SVD 章节会各取所需），不追求 LAPACK 的性能。
    """
    in_arr = np.asarray(A)
    if in_arr.ndim != 2:
        raise ValueError(f"需要矩阵，得到形状 {in_arr.shape}")
    if np.iscomplexobj(in_arr):
        # 复矩阵：不碰复旋转的相位，改走**实嵌入**
        # [[Re M, −Im M], [Im M, Re M]]——它的奇异值是 σ(M) 的每个值出现两次，
        # 所以复 SVD 完全交给实 Jacobi。代价是规模翻倍，收益是复/实共用同一套已验证代码。
        MC = np.asarray(in_arr, dtype=complex)
        m_orig, n_orig = MC.shape
        k = min(m_orig, n_orig)
        # 单边 Jacobi 要求 列数 <= 行数：必要时先转置，computed 的 U/V 与 M 同步换位
        transposed = m_orig < n_orig
        # 注意用 .T 而不是 .conj().T：实嵌入对复数不共轭也能保持奇异值
        Mt = MC.T if transposed else MC
        emb = _as_real_embedding(Mt)
        n_t = Mt.shape[1]
        _s, V_t, U_t, sweeps = one_sided_jacobi_svd(
            emb, max_sweeps=max_sweeps, tol=tol, compute_u=True
        )
        # ★ 嵌入的奇异值成对出现且已降序：[σ1,σ1,σ2,σ2,...] ⇒ 每隔一个取
        sig = np.asarray(_s[0 : 2 * k : 2], dtype=float)
        Vr = np.asarray(np.real(V_t), dtype=float)
        Ur = np.asarray(np.real(U_t), dtype=float)
        # 嵌入的向量是 2n 维实向量 [x; y]，对应复向量 x + i·y
        Vc_t = np.zeros((n_t, k), dtype=complex)
        Uc_t = np.zeros((Mt.shape[0], k), dtype=complex)
        for j in range(k):
            vj = Vr[:, j]
            Vc_t[:, j] = vj[:n_t] + 1j * vj[n_t:]
            uj = Ur[:, j]
            Uc_t[:, j] = uj[: Mt.shape[0]] + 1j * uj[Mt.shape[0] :]
        if transposed:
            # σ(Mᵀ) = σ(M)，且 (U_t, V_t) 与 (V, U) 互换
            U_out, V_out = Vc_t, Uc_t
        else:
            U_out, V_out = Uc_t, Vc_t
        if not compute_u:
            return sig, V_out, int(sweeps)
        return sig, V_out, U_out, int(sweeps)
    M = np.asarray(in_arr, dtype=float)
    m, ncol = M.shape
    transposed_real = ncol > m
    if transposed_real:
        M = np.ascontiguousarray(M.T)
        m, ncol = M.shape
    n = ncol
    # 关键：B 必须从**转置后**的 M 出发（列数 <= 行数），否则转置分支会静默算错
    B = np.array(M, dtype=complex, copy=True)
    V = np.eye(n, dtype=complex)
    n_sweeps = 0
    for _sweep in range(max(1, int(max_sweeps))):
        n_sweeps += 1
        off = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                alpha = float(np.real(np.vdot(B[:, p], B[:, p])))
                beta = float(np.real(np.vdot(B[:, q], B[:, q])))
                gamma = complex(np.vdot(B[:, p], B[:, q]))
                if abs(gamma) <= tol * float(np.sqrt(max(alpha * beta, 0.0))) or abs(gamma) == 0.0:
                    continue
                off = max(off, abs(gamma) / float(np.sqrt(max(alpha * beta, 1e-300))))
                zeta = (beta - alpha) / (2.0 * abs(gamma))
                # 注意符号：正确解是 t = −sign(ζ)/(|ζ|+√(1+ζ²))（即 τ−√(1+τ²) 那一支）。
                # 取正号时两列**不**正交，迭代不收敛——实测在 2000 组随机实向量上
                # 100% 失败（残差最大到 3.6），换成负号后 6 类形状全部到 4.5e-15。
                t = -np.sign(zeta) / (abs(zeta) + np.sqrt(1.0 + zeta * zeta)) if zeta != 0.0 else -1.0
                c = 1.0 / np.sqrt(1.0 + t * t)
                s = c * t * (gamma / abs(gamma))
                # 右乘 G（G = [[c, s], [−conj(s), c]]）：与 :func:`_apply_givens_right`
                # 用的是**同一套**列组合约定（列 p ← c·列p + conj(s)·列q）。
                # 把 conj 放错位置时旋转不再正交，复数矩阵上根本不收敛（实测迭代 6 轮
                # 后 Gram 的非对角元反而从 4.6 涨到 7.7）——这是本仓库踩过的坑之一。
                bp = B[:, p].copy()
                bq = B[:, q].copy()
                B[:, p] = c * bp + np.conj(s) * bq
                B[:, q] = -s * bp + c * bq
                vp = V[:, p].copy()
                vq = V[:, q].copy()
                V[:, p] = c * vp + np.conj(s) * vq
                V[:, q] = -s * vp + c * vq
        if off <= tol:
            break
    sigma = np.asarray([_norm2(B[:, i]) for i in range(n)], dtype=float)
    order = np.argsort(-sigma)
    sigma = sigma[order]
    V = V[:, order]
    if compute_u:
        UU = np.zeros((m, n), dtype=complex)
        for i in range(n):
            if sigma[i] > 0.0:
                UU[:, i] = B[:, order[i]] / sigma[i]
        if transposed_real:
            # σ(Mᵀ) = σ(M)：(U, V) 互换
            return sigma, UU, V, n_sweeps
        return sigma, V, UU, n_sweeps
    if transposed_real:
        return sigma, V.conj().T, n_sweeps
    return sigma, V, n_sweeps


def smallest_singular_value(A: Any, *, max_sweeps: int = 30, tol: float = 1e-15) -> tuple[float, int]:
    r"""最小奇异值 :math:`\sigma_{\min}(A)`（单边 Jacobi，**不用 SVD 成品**）。

    返回 ``(值, 扫描轮数)``。这是 :mod:`numcore.linalg.pseudospec` 求
    :math:`\sigma_{\min}(A-zI)` 的**主路线**：精度到机器量级，
    代价是 :math:`O(n^3)`/点（伪谱网格用它时要把矩阵控制在 :math:`n\lesssim 20`）。
    """
    sigma, _V, sweeps = one_sided_jacobi_svd(A, max_sweeps=max_sweeps, tol=tol)
    return float(sigma[-1]) if sigma.size else 0.0, int(sweeps)


def singular_values(A: Any, *, max_sweeps: int = 30, tol: float = 1e-15) -> np.ndarray:
    r"""全部奇异值（降序；单边 Jacobi）。**这是自研实现**，对拍请用 :mod:`numcore.reference.dense`。"""
    sigma, _V, _sw = one_sided_jacobi_svd(A, max_sweeps=max_sweeps, tol=tol)
    return sigma


# --------------------------------------------------------------------------- #
# 演示矩阵族（章节与测试共用；全部确定性构造，不用随机数的地方尽量不用）
# --------------------------------------------------------------------------- #
def ratio_matrix(n: int, ratio: float, *, shear: float = 0.0) -> np.ndarray:
    r"""构造 :math:`|\lambda_2/\lambda_1| = \text{ratio}` 的矩阵。

    :math:`A = D + s\,U`，其中 :math:`D = \mathrm{diag}(1, r, r^2, \dots, r^{n-1})`
    （:math:`r = \text{ratio}`），:math:`U` 是严格上三角的全 1 矩阵，:math:`s = \text{shear}`。

    特征值**精确等于** :math:`D` 的对角元（上三角矩阵的谱就是对角元），
    而 :math:`s` 只改变**非正规性**：:math:`s=0` 时是正规（对角）矩阵，
    :math:`s` 越大，特征向量矩阵 :math:`V` 的条件数越大，
    幂法的瞬态阶段越长（这就是 :math:`s` 存在的意义：让"收敛率"与"瞬态"分离）。
    """
    if n < 1:
        raise ValueError(f"n 必须为正，得到 {n}")
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"ratio 必须在开区间 (0,1) 内，得到 {ratio}")
    idx = np.arange(n)
    A = np.diag(np.asarray(ratio, dtype=float) ** idx)
    if shear != 0.0:
        for i in range(n):
            for j in range(i + 1, n):
                A[i, j] = shear
    return A


def shear_matrix(n: int, shear: float) -> np.ndarray:
    r"""剪切矩阵 :math:`A = I + s\,N`（:math:`N` 是严格上三角的全 1 矩阵）。

    全部特征值都是 1（代数重数 n），但 :math:`\|A^k\|_2` 可以长到
    :math:`O(s^{n-1})`——**特征值完全不告诉你它会不会增长**。
    这是本章"非正规性"的主角，也是 :func:`numcore.linalg.pseudospec.transient_growth` 的素材。
    """
    A = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = float(shear)
    return A


def jordan_block(n: int, lam: float = 0.0, *, scaled: bool = False) -> np.ndarray:
    r"""Jordan 块 :math:`J_n(\lambda)`：对角 :math:`\lambda`、超对角 1。

    ``scaled=True`` 时用"平衡"Jordan 块 :math:`\lambda I + sN`（:math:`s` 取
    :math:`(\lambda_{\text{scale}})^{1/(n-1)}` 的倒数）——**这是 :math:`\varepsilon`-伪谱
    形状对尺度敏感的原因**：伪谱半径是 :math:`\varepsilon^{1/n}` 乘以超对角元的尺度。
    """
    J = np.eye(n, dtype=float) * float(lam)
    idx = np.arange(n - 1)
    J[idx, idx + 1] = 1.0
    if scaled and n > 1:
        scale = (abs(lam) if abs(lam) > 0 else 1.0) ** (1.0 / (n - 1))
        J[idx, idx + 1] = scale
    return J


def companion_matrix(coeffs: Any) -> np.ndarray:
    r"""多项式 :math:`p(z)=z^n+c_{n-1}z^{n-1}+\dots+c_0` 的伴随矩阵。

    .. math::
        C = \begin{pmatrix}
        -c_{n-1} & -c_{n-2} & \cdots & -c_1 & -c_0\\
        1 & 0 & \cdots & 0 & 0\\ \vdots & & \ddots & & \vdots\\ 0 & \cdots & 1 & 0
        \end{pmatrix},

    :math:`C` 的特征值就是 :math:`p` 的根。伴随矩阵是**非正规矩阵的经典例子**
    （即使根分得很开，它的伪谱也可能把整个根集合包成一大片），
    所以本章用它来演示"根对扰动极其敏感"（Wilkinson 多项式）。

    **尺寸与顺序约定（本仓库踩过的两个坑）**：``coeffs`` 是**降幂**系数
    :math:`[c_n,c_{n-1},\\dots,c_0]`（长度 :math:`n+1`），返回 :math:`n\\times n`；
    第一行是 :math:`[-c_{n-1},\\dots,-c_0]/c_n`，**不要再反转一遍**。

    1. 尺寸：早先实现把整条 :math:`c`（含首项）塞进第一行、又只放 :math:`n-1` 个
       次对角 1，于是返回 :math:`(n+1)\\times(n+1)`；
    2. 顺序：早先又对系数做了 ``[::-1]``，于是 :math:`z^2-3z+2` 给出
       :math:`0.382,\\,2.618` 而不是 :math:`1,2`——**所有**伴随矩阵例子全错。
    对 :math:`\\prod_{k=1}^{n}(z-k)` 这条修好后特征值精确回到 :math:`1..n`
    （实测 :math:`n=10` 的最大误差 :math:`2.4\\times10^{-12}`）。
    """
    c = np.asarray(coeffs, dtype=float).reshape(-1)
    n = c.size - 1
    if n < 1:
        raise ValueError("至少需要两个系数（次数 >= 1）")
    lead = float(c[0])
    if lead == 0.0:
        raise ValueError("首项系数不能为 0")
    C = np.zeros((n, n), dtype=float)
    C[0, :] = -c[1:] / lead
    idx = np.arange(1, n)
    C[idx, idx - 1] = 1.0
    return C


def grcar_matrix(n: int, k: int = 3, *, sign: float = -1.0) -> np.ndarray:
    r"""Grcar 矩阵：下双对角为 1、主对角与上 :math:`k` 条对角为 :math:`\pm 1`。

    它是"看起来无害、其实高度非正规"的经典测试矩阵（Grcar 1990；Trefethen & Embree
    *Spectra and Pseudospectra* 的常客）。本章把它放进伪谱矩阵家族，
    用来对比"谱很规矩（全在单位圆附近）但伪谱指数级膨胀"。
    """
    A = np.zeros((n, n), dtype=float)
    idx = np.arange(n)
    A[idx, idx] = 1.0
    for d in range(1, k + 1):
        j = np.arange(n - d)
        A[j, j + d] = sign
    A[idx[:-2], idx[:-2] + 1] = sign  # 与主对角相邻的一条，构成"下双对角"
    A[idx[:-2] + 1, idx[:-2]] = 1.0
    return A


def laplacian_1d(n: int) -> np.ndarray:
    r"""一维 Dirichlet Laplacian：:math:`\mathrm{tridiag}(-1, 2, -1)`。

    特征值有闭式 :math:`\lambda_j = 2-2\cos\\frac{j\pi}{n+1}`（:math:`j=1..n`），
    是本章 A 类断言的**解析真值**（Lanczos/Rayleigh 商都拿它当靶子）。
    """
    if n < 2:
        raise ValueError(f"n 至少为 2，得到 {n}")
    A = np.zeros((n, n), dtype=float)
    idx = np.arange(n)
    A[idx, idx] = 2.0
    A[idx[:-1], idx[:-1] + 1] = -1.0
    A[idx[:-1] + 1, idx[:-1]] = -1.0
    return A


def symmetric_tridiag(alpha: Any, beta: Any) -> np.ndarray:
    r"""由对角 ``alpha`` 与次对角 ``beta`` 组装对称三对角矩阵（Lanczos 的 :math:`T_m`）。"""
    a = np.asarray(alpha, dtype=float).reshape(-1)
    b = np.asarray(beta, dtype=float).reshape(-1)
    m = a.size
    T = np.zeros((m, m), dtype=float)
    idx = np.arange(m)
    T[idx, idx] = a
    k = min(m - 1, b.size)
    if k > 0:
        T[idx[:k], idx[:k] + 1] = b[:k]
        T[idx[:k] + 1, idx[:k]] = b[:k]
    return T


#: 特征值矩阵动物园（与 1.2 章的 ``EXAMPLE_MATRICES`` 同一风格：每个矩阵一句"别相信直觉"）
EXAMPLE_EIGEN_MATRICES: dict[str, str] = {
    "symmetric": "对称三对角 Laplacian：特征值有闭式，幂法/QR/Lanczos 全部该赢",
    "nonsymmetric": "上三角 + 剪切：谱好算，但特征向量矩阵病态（非正规）",
    "cyclic": "2×2 旋转块：复共轭特征值，实向量序列只能绕圈——幂法不收敛的活标本",
    "jordan": "Jordan 块：特征值只有 0，但 ‖A^k‖ 先长后落，ε-伪谱是半径 ε^(1/n) 的圆盘",
    "grcar": "Grcar 矩阵：谱在单位圆附近很规矩，伪谱却大得离谱（Trefethen 的经典反例）",
    "wilkinson_poly": "Wilkinson 多项式（20 次）的伴随矩阵：根对扰动敏感得离谱",
    "random": "固定种子随机矩阵：用来给「典型情形」兜底（不是反例，只是对照）",
}


def example_eigen_matrix(name: str, n: int = 8, *, ratio: float = 0.5, shear: float = 0.3) -> np.ndarray:
    """返回 :data:`EXAMPLE_EIGEN_MATRICES` 里的示例矩阵（``n`` 对部分矩阵无效）。

    约定：所有矩阵都是确定性的（不用随机数，除 ``"random"`` 走
    :func:`numcore.utils.rng`），这样断言里的"实测值"才可复现。
    """
    if name == "symmetric":
        return laplacian_1d(max(2, n))
    if name == "nonsymmetric":
        return ratio_matrix(max(2, n), ratio, shear=shear)
    if name == "cyclic":
        # [cosθ −sinθ; sinθ cosθ]·r：复特征值 r e^{±iθ}
        th = 2.0 * np.pi / 5.0
        return ratio * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    if name == "jordan":
        return jordan_block(max(2, n), 0.0)
    if name == "grcar":
        return grcar_matrix(max(4, n))
    if name == "wilkinson_poly":
        # Wilkinson 多项式 ∏_{k=1}^{20}(z−k)：20 个根、20×20 伴随矩阵。
        # 它的根对扰动**极其**敏感（Trefethen 的经典反例：1e-10 的扰动能把根推出去 0.1），
        # 所以本章不拿它当精度靶子，只当"伪谱把根连成一片"的演示。
        coeffs = np.array([1.0])
        for k in range(1, 21):
            nxt = np.zeros(coeffs.size + 1, dtype=float)
            nxt[: coeffs.size] += coeffs
            nxt[1:] -= k * coeffs
            coeffs = nxt
        return companion_matrix(coeffs)
    if name == "random":
        g = rng(20240104)
        return g.standard_normal((max(2, n), max(2, n)))
    raise KeyError(f"未知示例矩阵 {name!r}（可用 {sorted(EXAMPLE_EIGEN_MATRICES)}）")
