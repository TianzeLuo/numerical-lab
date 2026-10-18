r"""Krylov 子空间法：CG（含预条件）与 GMRES(m)，全部手写（Arnoldi + Givens）。

数学定义
--------
给定 :math:`A\in\mathbb{R}^{n\times n}`、:math:`b`、:math:`x_0`，令 :math:`r_0=b-Ax_0`，

.. math::
    \mathcal{K}_k(A,r_0)=\operatorname{span}\{r_0,Ar_0,\dots,A^{k-1}r_0\}

是第 :math:`k` 个 **Krylov 子空间**。两类方法的区别只在"在子空间里挑哪个 :math:`x_k`"：

* **CG**（:math:`A` 对称正定）：:math:`x_k=\arg\min_{x\in x_0+\mathcal K_k}\|x-x^\*\|_A`，
  误差按 **A-范数** :math:`\|e\|_A=\sqrt{e^{\mathsf T}Ae}` 单调下降；等价刻画是
  :math:`r_k\perp\mathcal K_k`（残差两两正交）与 :math:`p_i^{\mathsf T}Ap_j=0\ (i\ne j)`。
  收敛界（最坏情形）
  .. math::
      \|x_k-x^\*\|_A\le 2\Bigl(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\Bigr)^k\|x_0-x^\*\|_A,
      \qquad \kappa=\kappa_2(A)=\lambda_{\max}/\lambda_{\min}.
* **GMRES**（任意非奇异 :math:`A`）：:math:`x_k=\arg\min_{x\in x_0+\mathcal K_k}\|b-Ax\|_2`。
  残差 **单调不增**（这是"最小残差"的直接推论），且
  :math:`r_k\perp A\mathcal K_k`。多项式视角：:math:`r_k=p_k(A)r_0`，
  :math:`p_k\in\mathcal P_k`、:math:`p_k(0)=1`——收敛快慢取决于"能否在谱上找到一个
  小范数且过原点的 :math:`k` 次多项式"。谱聚簇（而不是 :math:`\kappa` 小）才是 GMRES
  快的原因，这一点在 ``precond_effect`` 那一页会同时用谱散点与收敛曲线演示。

实现约束：只用 numpy 容器与逐元素原语（P3）、随机数走 :func:`numcore.utils.rng`（P4）、
纯函数（P6）；**不**调用 ``numpy.linalg``，小规模三角回代也用手写循环或自研 LU。
预条件一律取**左预条件**：解 :math:`M^{-1}Ax=M^{-1}b`；:math:`M` 以"能作用
:math:`M^{-1}r` 的对象"形式传入（见 :mod:`numcore.linalg.precond`）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from numcore.utils import rng

__all__ = [
    "matvec",
    "inner_product",
    "norm2",
    "cg_convergence_bound",
    "arnoldi",
    "ArnoldiResult",
    "lanczos",
    "LanczosResult",
    "tridiag_eigenvalues",
    "CGResult",
    "conjugate_gradient",
    "pcg",
    "GMRESResult",
    "gmres",
]


# --------------------------------------------------------------------------- #
# 线性算子与内积（稠密数组 / CSR / 预条件子都走同一套接口）
# --------------------------------------------------------------------------- #
def matvec(A: Any, x: Any) -> np.ndarray:
    """矩阵-向量乘：``A`` 可以是稠密 ``ndarray``，也可以是带 ``matvec`` 的容器（如 CSR）。"""
    xv = np.asarray(x, dtype=float).reshape(-1)
    if hasattr(A, "matvec"):
        return np.asarray(A.matvec(xv), dtype=float).reshape(-1)
    return np.asarray(A, dtype=float) @ xv


def norm2(x: Any) -> float:
    """欧氏范数 :math:`\\|x\\|_2`。"""
    xv = np.asarray(x, dtype=float).reshape(-1)
    return float(np.sqrt(float(xv @ xv)))


def inner_product(x: Any, y: Any) -> float:
    """欧氏内积 :math:`x^{\\mathsf T}y`。"""
    xv = np.asarray(x, dtype=float).reshape(-1)
    yv = np.asarray(y, dtype=float).reshape(-1)
    return float(xv @ yv)


def _precond_apply(M: Any, r: np.ndarray) -> np.ndarray:
    if M is None:
        return r
    if hasattr(M, "apply"):
        return np.asarray(M.apply(r), dtype=float).reshape(-1)
    raise TypeError("预条件子必须提供 apply(r) 方法（见 numcore.linalg.precond）")


def _as_rhs(b: Any, n: int) -> np.ndarray:
    rhs = np.asarray(b, dtype=float).reshape(-1)
    if rhs.size != n:
        raise ValueError(f"b 的长度应为 {n}，得到 {rhs.size}")
    return rhs


def _size(A: Any) -> int:
    if hasattr(A, "shape"):
        return int(A.shape[0])
    return int(np.asarray(A).shape[0])


# --------------------------------------------------------------------------- #
# CG
# --------------------------------------------------------------------------- #
def _matvec_M(precond: Any, x: np.ndarray) -> np.ndarray:
    """算 :math:`Mx`（**不是** :math:`M^{-1}x`）：预条件 Lanczos 的自伴内积要用它。

    ``precond=None``（恒等）时就是 ``x``。三种预条件子都提供 ``matvec_M``；
    没有该方法的对象退回 ``to_dense()``（只适合小规模）。
    """
    if precond is None:
        return x
    fn = getattr(precond, "matvec_M", None)
    if fn is not None:
        return np.asarray(fn(x), dtype=float)
    dense = getattr(precond, "to_dense", None)
    if dense is None:
        raise TypeError("预条件子必须提供 matvec_M(x) 或 to_dense()（预条件 Lanczos 需要能算 Mx）")
    return np.asarray(dense(), dtype=float) @ np.asarray(x, dtype=float)


def cg_convergence_bound(kappa: float, k: int) -> float:
    r"""CG 收敛界 :math:`2\bigl((\sqrt\kappa-1)/(\sqrt\kappa+1)\bigr)^k`（A-范数相对误差）。"""
    kk = float(kappa)
    if kk < 1.0:
        raise ValueError(f"条件数不能小于 1，得到 {kappa}")
    s = float(np.sqrt(kk))
    return float(2.0 * ((s - 1.0) / (s + 1.0)) ** int(k))


@dataclass(frozen=True)
class CGResult:
    """CG / PCG 的完整轨迹与正交性证据。"""

    x: np.ndarray
    iterates: np.ndarray
    residuals: np.ndarray
    abs_residuals: np.ndarray
    errors_2: np.ndarray
    errors_A: np.ndarray
    alphas: np.ndarray
    betas: np.ndarray
    residual_orthogonality: np.ndarray
    p_orthogonality: np.ndarray
    p_basis: np.ndarray
    r_basis: np.ndarray
    n_iter: int
    converged: bool
    breakdown: bool

    @property
    def bound(self) -> np.ndarray:
        return self.abs_residuals


def conjugate_gradient(
    A: Any,
    b: Any,
    *,
    x0: Any = None,
    tol: float = 1e-10,
    maxiter: int | None = None,
    precond: Any = None,
    x_exact: Any = None,
    track_basis: bool = False,
    track_orthogonality: bool = True,
) -> CGResult:
    r"""共轭梯度法（可选预条件）：:math:`A` 对称正定。

    迭代（PCG 形式，:math:`z=M^{-1}r`）：

    .. math::
        \alpha_k=\frac{r_k^{\mathsf T}z_k}{p_k^{\mathsf T}Ap_k},\quad
        x_{k+1}=x_k+\alpha_kp_k,\quad
        r_{k+1}=r_k-\alpha_kAp_k,\quad
        \beta_k=\frac{r_{k+1}^{\mathsf T}z_{k+1}}{r_k^{\mathsf T}z_k},\quad
        p_{k+1}=z_{k+1}+\beta_kp_k .

    停机判据是**真实相对残差** :math:`\|b-Ax_k\|_2/\|b\|_2\le\mathtt{tol}`
    （不是递推残差——递推残差会漂移，这一点在 GMRES 里也单独记了）。
    ``x_exact`` 给出时同时记录 2-范数误差与 A-范数误差（CG 单调下降的是后者）。
    """
    n = _size(A)
    rhs = _as_rhs(b, n)
    x = np.zeros(n, dtype=float) if x0 is None else np.asarray(x0, dtype=float).reshape(-1).copy()
    if x.size != n:
        raise ValueError(f"x0 的长度应为 {n}，得到 {x.size}")
    itmax = int(maxiter) if maxiter is not None else 4 * n + 10
    nb = norm2(rhs) or 1.0
    xs = None if x_exact is None else np.asarray(x_exact, dtype=float).reshape(-1)

    r = rhs - matvec(A, x)
    z = _precond_apply(precond, r)
    p = z.copy()
    rz = inner_product(r, z)

    iterates = [x.copy()]
    residuals = [norm2(r) / nb]
    abs_res = [norm2(r)]
    err2 = [norm2(x - xs) if xs is not None else float("nan")]
    errA = [_a_norm(A, x - xs) if xs is not None else float("nan")]
    alphas: list[float] = []
    betas: list[float] = []
    r_hist = [r.copy()] if track_orthogonality else []
    p_hist = [p.copy()] if track_orthogonality else []  # 第 i 行是方向 p_i（与 r_hist 对齐）

    converged = residuals[-1] <= tol
    breakdown = False
    k = 0
    while not converged and k < itmax:
        Ap = matvec(A, p)
        pAp = inner_product(p, Ap)
        if not np.isfinite(pAp) or pAp <= 0.0:
            breakdown = True  # A 不是正定 / p 退化：继续算下去没有意义
            break
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        z = _precond_apply(precond, r)
        rz_new = inner_product(r, z)
        nr = norm2(r)
        k += 1
        alphas.append(float(alpha))
        iterates.append(x.copy())
        abs_res.append(nr)
        residuals.append(nr / nb)
        err2.append(norm2(x - xs) if xs is not None else float("nan"))
        errA.append(_a_norm(A, x - xs) if xs is not None else float("nan"))
        if track_orthogonality:
            r_hist.append(r.copy())
        converged = residuals[-1] <= tol
        if converged or rz_new <= 0.0:
            break
        beta = rz_new / rz
        betas.append(float(beta))
        p = z + beta * p
        rz = rz_new
        if track_orthogonality:
            p_hist.append(p.copy())  # 新方向 p_{k+1}；提前收敛时序列会比 r_hist 短一个

    R = np.asarray(r_hist, dtype=float) if r_hist else np.zeros((0, n), dtype=float)
    P = np.asarray(p_hist, dtype=float) if p_hist else np.zeros((0, n), dtype=float)
    orth_r = _gram_matrix(R) if track_orthogonality else np.zeros((0, 0), dtype=float)
    orth_p = _gram_matrix(P, A) if (track_basis and track_orthogonality) else np.zeros((0, 0), dtype=float)
    return CGResult(
        x=x,
        iterates=np.asarray(iterates, dtype=float),
        residuals=np.asarray(residuals, dtype=float),
        abs_residuals=np.asarray(abs_res, dtype=float),
        errors_2=np.asarray(err2, dtype=float),
        errors_A=np.asarray(errA, dtype=float),
        alphas=np.asarray(alphas, dtype=float),
        betas=np.asarray(betas, dtype=float),
        residual_orthogonality=orth_r,
        p_orthogonality=orth_p,
        p_basis=P if track_basis else np.zeros((0, P.shape[1]), dtype=float),
        r_basis=R,
        n_iter=k,
        converged=bool(converged),
        breakdown=breakdown,
    )


def pcg(A: Any, b: Any, *, precond: Any = None, **kw: Any) -> CGResult:
    """预条件共轭梯度（PCG）；``precond=None`` 时退化为普通 CG。"""
    return conjugate_gradient(A, b, precond=precond, **kw)


def _a_norm(A: Any, e: np.ndarray) -> float:
    r"""A-范数 :math:`\|e\|_A=\sqrt{e^{\mathsf T}Ae}`（CG 真正单调下降的量）。"""
    v = float(np.asarray(e, dtype=float) @ matvec(A, e))
    return float(np.sqrt(max(v, 0.0)))


def _gram_matrix(B: np.ndarray, A: Any = None) -> np.ndarray:
    r"""归一化 Gram 矩阵：``A=None`` 时给 :math:`|r_i^{\mathsf T}r_j|/(\|r_i\|\|r_j\|)`，
    否则给 :math:`|p_i^{\mathsf T}Ap_j|` 的同型归一化值（供正交性断言与热力图使用）。"""
    m = B.shape[0]
    G = np.zeros((m, m), dtype=float)
    if m == 0:
        return G
    if A is None:
        for i in range(m):
            for j in range(i, m):
                v = abs(inner_product(B[i], B[j]))
                d = norm2(B[i]) * norm2(B[j])
                G[i, j] = G[j, i] = v / d if d > 0.0 else 0.0
    else:
        AB = np.asarray([matvec(A, B[i]) for i in range(m)], dtype=float)
        for i in range(m):
            for j in range(i, m):
                v = abs(inner_product(B[i], AB[j]))
                d = np.sqrt(abs(inner_product(B[i], AB[i])) * abs(inner_product(B[j], AB[j])))
                G[i, j] = G[j, i] = v / d if d > 0.0 else 0.0
    return G


# --------------------------------------------------------------------------- #
# Arnoldi / Lanczos
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArnoldiResult:
    """Arnoldi 分解 :math:`AV_m=V_{m+1}\\underline H_m` 的结果。"""

    V: np.ndarray
    H: np.ndarray
    n_steps: int
    orthogonality: float
    breakdown: bool
    happy: bool


def arnoldi(
    A: Any,
    v0: Any,
    m: int,
    *,
    precond: Any = None,
    reorth: bool = True,
) -> ArnoldiResult:
    r"""手写 Arnoldi 过程（修正 Gram--Schmidt + 可选二次正交化）。

    返回 ``V``（形状 ``(n, m+1)``，列正交归一）、``H``（``(m+1, m)`` 上 Hessenberg）。
    ``orthogonality`` 是 :math:`\max|V^{\mathsf T}V-I|` 的实测值：它必须落在
    :math:`O(u\kappa(V))` 量级，否则 Ritz 值/收敛曲线都是幻觉。
    左预条件时对 :math:`M^{-1}A` 做 Arnoldi（内积仍是欧氏内积）。
    """
    n = _size(A)
    v = np.asarray(v0, dtype=float).reshape(-1).copy()
    nv = norm2(v)
    if nv == 0.0:
        raise ValueError("起始向量不能是零向量")
    V = np.zeros((n, int(m) + 1), dtype=float)
    H = np.zeros((int(m) + 1, int(m)), dtype=float)
    V[:, 0] = v / nv
    happy = False
    breakdown = False
    steps = 0
    n_valid = 1  # breakdown 时 V[:, j+1] 没有被写出来，不能算进基里
    for j in range(int(m)):
        w = matvec(A, V[:, j])
        w = _precond_apply(precond, w)
        for _pass in range(2 if reorth else 1):
            for i in range(j + 1):
                h = inner_product(V[:, i], w)
                H[i, j] += h
                w = w - h * V[:, i]
        H[j + 1, j] = norm2(w)
        steps = j + 1
        if H[j + 1, j] <= 1e-14:
            # happy breakdown：Krylov 子空间已经不变，精确解就在里面
            breakdown = True
            happy = True
            break
        V[:, j + 1] = w / H[j + 1, j]
        n_valid = j + 2
    Vk = V[:, :n_valid]
    orth = float(np.max(np.abs(Vk.T @ Vk - np.eye(n_valid))))
    return ArnoldiResult(V=Vk, H=H[: steps + 1, :steps], n_steps=steps, orthogonality=orth, breakdown=breakdown, happy=happy)


@dataclass(frozen=True)
class LanczosResult:
    """对称（或预条件对称）Lanczos 的三对角化结果。"""

    diag: np.ndarray
    offdiag: np.ndarray
    V: np.ndarray
    orthogonality: float
    n_steps: int
    breakdown: bool

    @property
    def ritz(self) -> np.ndarray:
        """Ritz 值（三对角矩阵的全部特征值，手写 Sturm 二分求得）。"""
        return tridiag_eigenvalues(self.diag, self.offdiag)


def lanczos(
    A: Any,
    v0: Any,
    m: int,
    *,
    precond: Any = None,
    reorth: bool = True,
) -> LanczosResult:
    r"""对 :math:`M^{-1}A` 在 **M**-内积下做 Lanczos（预条件谱分析的引擎）。

    当 :math:`A,M` 都对称正定时，:math:`M^{-1}A` 关于内积
    :math:`\langle x,y\rangle_M=x^{\mathsf T}My` 自伴：
    :math:`\langle M^{-1}Ax,y\rangle_M=x^{\mathsf T}Ay=\langle x,M^{-1}Ay\rangle_M`。
    （**不是** :math:`M^{-1}`-内积——那个内积下 :math:`M^{-1}A` 一般不正自伴，
    实测拿它去算 SSOR 的条件数会偏 2\%～5\%：一维 :math:`n=32` 上给 38.82 而真值是 38.01。）
    于是三项递推成立、三对角矩阵 :math:`T_m` 的特征值（Ritz 值）全是实数。
    ``precond=None`` 时内积退化为欧氏内积，得到的就是 :math:`A` 自己的谱。

    注意 :math:`\alpha_j=\langle M^{-1}Av_j,v_j\rangle_M=v_j^{\mathsf T}Av_j`
    是 :math:`A` 的 Rayleigh 商，落在 :math:`[\lambda_{\min}(A),\lambda_{\max}(A)]` 内；
    而 Ritz 值逼近的是 :math:`M^{-1}A` 的谱（两者不是一回事）。
    """
    n = _size(A)

    def ip(x: np.ndarray, y: np.ndarray) -> float:
        return inner_product(x, _matvec_M(precond, y))

    def op(x: np.ndarray) -> np.ndarray:
        return _precond_apply(precond, matvec(A, x))

    v = np.asarray(v0, dtype=float).reshape(-1).copy()
    nv = float(np.sqrt(max(ip(v, v), 0.0)))
    if nv == 0.0:
        raise ValueError("起始向量不能是零向量（M-内积意义下）")
    V = np.zeros((n, int(m) + 1), dtype=float)
    V[:, 0] = v / nv
    diag = np.zeros(int(m), dtype=float)
    off = np.zeros(max(int(m) - 1, 0), dtype=float)
    beta_prev = 0.0
    steps = 0
    n_valid = 1  # V 里**真正有效**的列数：breakdown 时最后一列没被写出来
    breakdown = False
    for j in range(int(m)):
        w = op(V[:, j])
        alpha = ip(w, V[:, j])
        diag[j] = alpha
        w = w - alpha * V[:, j]
        if j > 0:
            w = w - beta_prev * V[:, j - 1]
        if reorth:  # 全正交化：小 m 下最稳，避免"幽灵特征值"
            for i in range(j + 1):
                c = ip(w, V[:, i])
                w = w - c * V[:, i]
        beta = float(np.sqrt(max(ip(w, w), 0.0)))
        steps = j + 1
        if beta <= 1e-14:
            breakdown = True
            break
        if j + 1 < int(m):
            off[j] = beta
        V[:, j + 1] = w / beta
        n_valid = j + 2
        beta_prev = beta
    Vk = V[:, :n_valid]
    G = np.array([[ip(Vk[:, i], Vk[:, j]) for j in range(n_valid)] for i in range(n_valid)], dtype=float)
    orth = float(np.max(np.abs(G - np.eye(n_valid))))
    return LanczosResult(
        diag=diag[:steps],
        offdiag=off[: max(steps - 1, 0)],
        V=Vk,
        orthogonality=orth,
        n_steps=steps,
        breakdown=breakdown,
    )


def tridiag_eigenvalues(diag: Any, offdiag: Any, *, n_bisect: int = 60) -> np.ndarray:
    r"""对称三对角矩阵的全部特征值：**Sturm 序列 + 二分**（手写，不用 ``numpy.linalg``）。

    原理：:math:`T-\sigma I` 的 LDL 分解中负主元个数等于小于 :math:`\sigma` 的特征值个数
    （Sturm 计数）。对第 :math:`k` 个特征值二分"计数 :math:`=k`"的位置，
    50--60 步二分就把区间压到机器精度（比 QL 迭代短、且没有收敛性假设）。
    """
    d = np.asarray(diag, dtype=float).ravel().copy()
    e = np.asarray(offdiag, dtype=float).ravel().copy()
    n = d.size
    if n == 0:
        return np.zeros(0, dtype=float)
    if e.size != n - 1:
        raise ValueError(f"次对角元长度应为 {n - 1}，得到 {e.size}")
    # Gershgorin 界（次对角元各贡献一次）
    pad = np.concatenate([e, np.zeros(1)])
    pad0 = np.concatenate([np.zeros(1), e])
    radius = np.abs(pad) + np.abs(pad0)
    lo = float(np.min(d - radius))
    hi = float(np.max(d + radius))
    span = max(hi - lo, 1e-300)
    lo -= 1e-12 * span
    hi += 1e-12 * span
    out = np.zeros(n, dtype=float)
    for k in range(n):
        a, b = lo, hi
        for _ in range(int(n_bisect)):
            mid = 0.5 * (a + b)
            if _sturm_count(d, e, mid) > k:
                b = mid
            else:
                a = mid
        out[k] = 0.5 * (a + b)
    return out


def _sturm_count(d: np.ndarray, e: np.ndarray, sigma: float) -> int:
    """:math:`T-\\sigma I` 的负主元个数 = 小于 :math:`\\sigma` 的特征值个数。"""
    n = d.size
    q = d[0] - sigma
    count = 1 if q < 0.0 else 0
    tiny = 1e-300
    for i in range(1, n):
        if q == 0.0:
            q = tiny
        q = (d[i] - sigma) - (e[i - 1] * e[i - 1]) / q
        if q < 0.0:
            count += 1
    return count


# --------------------------------------------------------------------------- #
# GMRES(m)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GMRESResult:
    """GMRES(m) 的轨迹：真实残差、Givens 递推残差、Arnoldi 基与正交性证据。"""

    x: np.ndarray
    iterates: np.ndarray
    residuals: np.ndarray
    est_residuals: np.ndarray
    abs_residuals: np.ndarray
    errors_2: np.ndarray
    basis: np.ndarray
    H: np.ndarray
    residual_vs_basis: np.ndarray
    """``|r_i^{\\mathsf T}(Av_j)|/(\\|r_i\\|\\|Av_j\\|)``：GMRES 最优性条件的实测矩阵（j<i 应为 0）。"""
    basis_orthogonality: float
    n_iter: int
    n_cycles: int
    converged: bool
    happy_breakdown: bool


def _givens(a: float, b: float) -> tuple[float, float]:
    """实 Givens 旋转 ``(c, s)``，使 ``[[c,s],[-s,c]] @ [a,b]^T = [r,0]^T``。"""
    if b == 0.0:
        return (1.0 if a >= 0.0 else -1.0), 0.0
    r = float(np.hypot(a, b))
    return a / r, b / r


def _solve_upper_triangular(R: np.ndarray, g: np.ndarray) -> np.ndarray:
    r"""上三角回代 :math:`Ry=g`（GMRES 的最小二乘解，手写；奇异时返回伪解）。"""
    m = g.size
    y = np.zeros(m, dtype=float)
    for i in range(m - 1, -1, -1):
        s = float(g[i])
        for j in range(i + 1, m):
            s -= R[i, j] * y[j]
        piv = float(R[i, i])
        if piv == 0.0:
            y[i] = 0.0
        else:
            y[i] = s / piv
    return y


def gmres(
    A: Any,
    b: Any,
    *,
    m: int = 30,
    tol: float = 1e-10,
    maxiter: int | None = None,
    x0: Any = None,
    precond: Any = None,
    x_exact: Any = None,
    track_orthogonality: bool = True,
) -> GMRESResult:
    r"""GMRES(m)：重启的广义最小残差法（Arnoldi + Givens，全部手写）。

    每个重启周期在 :math:`\mathcal K_m(M^{-1}A,M^{-1}r_0)` 上最小化
    :math:`\|M^{-1}(b-Ax)\|_2`；用 Givens 旋转把 Hessenberg 化为上三角，
    递推得到残差估计，最后回代得到修正量。**真实残差**与递推残差都记录：
    二者的差距随 :math:`\kappa(V)` 增长，是"看起来收敛了其实没有"的经典来源。

    残差单调性（同一周期内）是"最小残差"的直接推论；重启会破坏全局单调性
    ——这是 RESTART 的代价，可视化里能看到台阶。
    """
    n = _size(A)
    rhs = _as_rhs(b, n)
    x = np.zeros(n, dtype=float) if x0 is None else np.asarray(x0, dtype=float).reshape(-1).copy()
    if x.size != n:
        raise ValueError(f"x0 的长度应为 {n}，得到 {x.size}")
    xs = None if x_exact is None else np.asarray(x_exact, dtype=float).reshape(-1)
    nb = norm2(rhs) or 1.0
    mm = max(1, int(m))
    itmax = int(maxiter) if maxiter is not None else 10 * n

    iterates = [x.copy()]
    residuals = [norm2(rhs - matvec(A, x)) / nb]
    est_residuals = [residuals[0]]
    abs_res = [residuals[0] * nb]
    err2 = [norm2(x - xs) if xs is not None else float("nan")]
    basis_cols: list[np.ndarray] = []
    av_cols: list[np.ndarray] = []
    H_last = np.zeros((mm + 1, mm), dtype=float)
    orth_max = 0.0
    happy = False
    n_iter = 0
    n_cycles = 0
    converged = residuals[-1] <= tol

    while not converged and n_iter < itmax:
        r = rhs - matvec(A, x)
        z = _precond_apply(precond, r)  # 左预条件：对 M^{-1}r 建 Krylov 空间
        beta = norm2(z)
        if beta == 0.0:
            converged = True
            break
        V = np.zeros((n, mm + 1), dtype=float)
        V[:, 0] = z / beta
        H = np.zeros((mm + 1, mm), dtype=float)
        cs = np.zeros(mm, dtype=float)
        sn = np.zeros(mm, dtype=float)
        g = np.zeros(mm + 1, dtype=float)
        g[0] = beta
        steps = 0
        n_valid = 1
        for j in range(mm):
            av = matvec(A, V[:, j])
            av_cols.append(av.copy())
            w = _precond_apply(precond, av)
            for _pass in range(2):  # 二次正交化：Krylov 基一旦失真，残差曲线就是假的
                for i in range(j + 1):
                    h = inner_product(V[:, i], w)
                    H[i, j] += h
                    w = w - h * V[:, i]
            H[j + 1, j] = norm2(w)
            if H[j + 1, j] > 1e-14:
                V[:, j + 1] = w / H[j + 1, j]
                n_valid = j + 2
            for i in range(j):
                h0, h1 = H[i, j], H[i + 1, j]
                H[i, j] = cs[i] * h0 + sn[i] * h1
                H[i + 1, j] = -sn[i] * h0 + cs[i] * h1
            c, s = _givens(H[j, j], H[j + 1, j])
            cs[j], sn[j] = c, s
            h_next = float(H[j + 1, j])  # 旋转会把次对角元置零，先留住它做 happy 判据
            H[j, j] = c * H[j, j] + s * H[j + 1, j]
            H[j + 1, j] = 0.0
            g[j + 1] = -s * g[j]
            g[j] = c * g[j]
            steps = j + 1
            est = abs(g[j + 1]) / nb
            est_residuals.append(est)
            # 先把当前周期的解算出来（viz 要逐 Krylov 步的轨迹）
            y = _solve_upper_triangular(H[: steps, :steps], g[: steps])
            x_cur = x + V[:, :steps] @ y
            true_res = norm2(rhs - matvec(A, x_cur)) / nb
            iterates.append(x_cur.copy())
            residuals.append(true_res)
            abs_res.append(true_res * nb)
            err2.append(norm2(x_cur - xs) if xs is not None else float("nan"))
            n_iter += 1
            if h_next <= 1e-14:
                happy = True
            if true_res <= tol or est <= tol * 1e-3 or happy:
                break
        Vk = V[:, :n_valid]
        orth_max = max(orth_max, float(np.max(np.abs(Vk.T @ Vk - np.eye(n_valid)))))
        H_last = H[: steps + 1, :steps].copy()
        basis_cols.append(Vk)
        y = _solve_upper_triangular(H[: steps, :steps], g[: steps])
        x = x + V[:, : steps] @ y
        n_cycles += 1
        converged = residuals[-1] <= tol
        if happy:
            break

    # 最优性条件的实测：GMRES 里 r_k ⊥ A K_k(A,r_0)，即 |r_i^T (A v_j)| = 0 (j < i)。
    # 注意**不是** |r_i^T v_j| = 0——那是 CG 的性质（CG 的 r_k ⊥ K_k），两者别混。
    n_rec = len(residuals)
    rvb = np.zeros((n_rec, n_rec), dtype=float)
    if track_orthogonality:
        for i in range(n_rec):
            ri = rhs - matvec(A, iterates[i])
            nri = norm2(ri) or 1.0
            for j in range(min(i, len(av_cols))):
                nav = norm2(av_cols[j])
                if nav > 0.0:
                    rvb[i, j] = abs(inner_product(ri, av_cols[j])) / (nri * nav)
    return GMRESResult(
        x=x,
        iterates=np.asarray(iterates, dtype=float),
        residuals=np.asarray(residuals, dtype=float),
        est_residuals=np.asarray(est_residuals, dtype=float),
        abs_residuals=np.asarray(abs_res, dtype=float),
        errors_2=np.asarray(err2, dtype=float),
        basis=basis_cols[-1] if basis_cols else np.zeros((n, 0), dtype=float),
        H=H_last,
        residual_vs_basis=rvb,
        basis_orthogonality=orth_max,
        n_iter=n_iter,
        n_cycles=n_cycles,
        converged=bool(converged),
        happy_breakdown=happy,
    )
