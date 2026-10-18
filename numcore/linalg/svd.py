"""1.3 章 SVD 内核：两版自研奇异值分解 + 伪逆 + 截断。

数学对象
--------
``A = U diag(s) Vᵀ``（``U`` 列正交 ``m x r``、``V`` 列正交 ``n x r``、``r = min(m,n)``、
``s_1 >= ... >= s_r >= 0``）；``A⁺ = V diag(1/s_i) Uᵀ``（在 ``s_i > 0`` 的方向上）。

三套实现，互相独立、互为靶子：

* :func:`jacobi_svd` —— **单边 Jacobi**：只对 ``A`` 的列做正交旋转，让列两两正交，
  列范数就是奇异值。收敛是二次的，且小奇异值的**相对**精度极好
  （Demmel & Veselic 证过：相对精度可到机器精度量级）；代价是每对列都要做内积。
* :func:`kogbetliantz_svd` —— **双边旋转**：每次取 ``2 x 2`` 子块做一次**精确的**
  2x2 SVD（左、右各一个旋转），把该块对角化；整矩阵的非对角量按二次速度归零。
* :func:`golub_kahan_svd` —— **Golub--Kahan 双对角化 + 隐式 QR**：先用 Householder
  把 ``A`` 化成上双对角 ``B``，再对 ``B`` 跑带 Wilkinson 位移的隐式 QR
  （追 bulge 的 Givens 链）。这是 LAPACK 的路线，``O(mn²)`` 而非 Jacobi 的常数倍。

**为什么不许显式形成 ``AᵀA``**：``kappa(AᵀA) = kappa(A)^2``，在 double 下
``kappa(A) > 1/u`` 的方向会被直接抹平；上面三套都只对 ``A`` 本身做正交变换，
不平方条件数（见 ``chapters/1.3/pitfalls.md`` 第 1、6 条）。

出处：Golub & Van Loan, *Matrix Computations* 4th ed., §8.3--8.6；Demmel & Veselic,
"Jacobi's method is more accurate than QR", SIAM J. Matrix Anal. Appl. 13 (1992)。

POLICY 说明（P2/P3/P6）：全部自研，不调用 ``numpy.linalg``；无 IO、无随机、无全局状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from numcore.linalg.qr import givens_matrix, givens_rotation, householder_qr, householder_vector

__all__ = [
    "SVDFit",
    "BidiagonalFit",
    "jacobi_svd",
    "kogbetliantz_svd",
    "golub_kahan_svd",
    "bidiagonalize",
    "bidiagonal_svd",
    "sort_and_fix_signs",
    "numerical_rank",
    "pseudoinverse",
    "truncated_solution",
    "moore_penrose_defects",
    "singular_vector_residuals",
    "spectral_condition",
    "relative_singular_value_error",
]

_U = float(np.finfo(float).eps) / 2.0  # 单位舍入 u = 2^-53


# --------------------------------------------------------------------------- #
# 容器与工具
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SVDFit:
    """``A = U diag(s) Vᵀ``（thin）：``U`` 为 ``m x r``、``V`` 为 ``n x r``、``r = min(m,n)``。"""

    U: np.ndarray
    s: np.ndarray
    V: np.ndarray
    method: str = ""
    converged: bool = True
    iterations: int = 0
    detail: dict[str, float] = field(default_factory=dict)

    @property
    def m(self) -> int:
        return int(self.U.shape[0])

    @property
    def n(self) -> int:
        return int(self.V.shape[0])

    @property
    def rank(self) -> int:
        return numerical_rank(self.s, self.m, self.n)

    @property
    def condition(self) -> float:
        return spectral_condition(self.s)

    def reconstruct(self) -> np.ndarray:
        """由因子重建 ``A``（``U diag(s) Vᵀ``）。"""
        return (self.U * self.s) @ self.V.T

    def pinv(self, *, rtol: float | None = None) -> np.ndarray:
        return pseudoinverse(self.U, self.s, self.V, rtol=rtol)


@dataclass(frozen=True)
class BidiagonalFit:
    """双对角化结果 ``A = U1 B V1ᵀ``（``B`` 为 ``r x r``，``r = min(m,n)``）。

    ``alpha`` 是对角、``beta`` 是唯一那条次对角线；``orientation`` 说明它在**上方**
    （``m >= n``：``B`` 上双对角 ``A = U1 B V1ᵀ``）还是**下方**
    （``m < n``：宽矩阵的 A 只能写成下双对角 ``A = U1 B V1ᵀ``，``beta`` 落在次对角）。

    为什么宽矩阵要多一个字段：把 ``m < n`` 的矩阵化成 r×r 的**上**双对角是做不到的——
    最后一行的尾巴经反射后只能变成 ``[b, 0, ..., 0]``，那个 ``b`` 位于第 ``r`` 列，
    已经出了 r×r 的框。宽矩阵的规范形式是下双对角，所以这里转置过来算并显式标出朝向
    （``golub_kahan_svd`` 对宽矩阵本来就是转置处理的，两条路给出同一个 B）。
    """

    U1: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    V1: np.ndarray
    n_ops: int = 0
    orientation: str = "upper"

    @property
    def r(self) -> int:
        return int(self.alpha.size)

    @property
    def U1_thin(self) -> np.ndarray:
        """``m x r``：``A = U1_thin B V1_thinᵀ``。"""
        return self.U1[:, : self.r]

    @property
    def V1_thin(self) -> np.ndarray:
        """``n x r``。"""
        return self.V1[:, : self.r]

    def to_dense(self) -> np.ndarray:
        r = self.r
        B = np.diag(self.alpha)
        if r > 1:
            offset = 1 if self.orientation == "upper" else -1
            B = B + np.diag(self.beta, offset)
        return B


def _as_matrix(A: Any) -> np.ndarray:
    M = np.asarray(A, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"需要二维矩阵，得到 {M.ndim} 维")
    if M.size == 0:
        raise ValueError("空矩阵")
    return M


def sort_and_fix_signs(U: np.ndarray, s: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """奇异值降序排列，并把符号约定**钉死**：每列 ``U`` 的最大绝对值分量为正。

    为什么必须显式约定：奇异向量只在 ``+-1`` 的意义下唯一，
    "两版 SVD 的 ``U`` 相不相等"这种比较如果不先定符号，就只能比 ``|uᵀ v|``，
    很容易把"符号不同"误判成"算法错了"（pitfalls 第 5 条）。
    """
    s = np.asarray(s, dtype=float).ravel()
    idx = np.argsort(-s, kind="stable")
    s = s[idx]
    U = np.asarray(U, dtype=float)[:, idx]
    V = np.asarray(V, dtype=float)[:, idx]
    for j in range(s.size):
        if U.shape[0] == 0:
            break
        col = U[:, j]
        k = int(np.argmax(np.abs(col)))
        if col[k] < 0.0:
            U[:, j] = -col
            V[:, j] = -V[:, j]
    return U, s, V


def _complete_orthonormal(U: np.ndarray) -> np.ndarray:
    """把含零列的 ``U`` 补成列正交矩阵（零列对应 ``s = 0``，取值不影响重建）。"""
    m, r = U.shape
    out = np.array(U, dtype=float, copy=True)
    for j in range(r):
        nrm = float(np.sqrt(np.dot(out[:, j], out[:, j])))
        if nrm > 0.0:
            out[:, j] /= nrm
            continue
        for cand in range(m):
            e = np.zeros(m, dtype=float)
            e[cand] = 1.0
            e = e - out @ (out.T @ e)
            ne = float(np.sqrt(np.dot(e, e)))
            if ne > 0.5:  # 与已有列线性无关
                out[:, j] = e / ne
                break
    return out


# --------------------------------------------------------------------------- #
# 单边 Jacobi
# --------------------------------------------------------------------------- #
def jacobi_svd(A: Any, *, max_sweeps: int = 60, tol: float | None = None) -> SVDFit:
    """单边 Jacobi SVD：反复对**列对**做旋转使它们正交，列范数即奇异值。

    每对列的旋转角由 2x2 的 Gram 矩阵 ``[[alpha, gamma], [gamma, beta]]`` 给出：
    ``zeta = (beta - alpha)/(2 gamma)``、``t = sign(zeta)/(|zeta| + sqrt(1+zeta^2))``。
    旋转是正交的，所以**不做任何相减型消元**，也不需要 ``AᵀA``。

    参数
    ----
    max_sweeps
        最大扫描次数（每次扫描遍历全部 ``n(n-1)/2`` 个列对）。
    tol
        收敛判据 ``|gamma| <= tol sqrt(alpha beta)``，缺省 ``8u ≈ 8.9e-16``。

    ``m < n`` 时对 ``Aᵀ`` 做同一件事再交换 ``U``/``V``。
    """
    M = _as_matrix(A)
    m, n = M.shape
    if m < n:
        fit = jacobi_svd(M.T, max_sweeps=max_sweeps, tol=tol)
        return SVDFit(
            U=fit.V, s=fit.s, V=fit.U, method="jacobi", converged=fit.converged,
            iterations=fit.iterations, detail=fit.detail,
        )
    if tol is None:
        tol = 8.0 * _U
    W = np.array(M, dtype=float, copy=True)
    V = np.eye(n)
    converged = False
    sweeps = 0
    off = float("inf")
    for sweep in range(int(max_sweeps)):
        sweeps = sweep + 1
        off = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                alpha = float(np.dot(W[:, p], W[:, p]))
                beta = float(np.dot(W[:, q], W[:, q]))
                gamma = float(np.dot(W[:, p], W[:, q]))
                scale = float(np.sqrt(alpha * beta))
                if scale == 0.0:
                    continue
                rel = abs(gamma) / scale
                off = max(off, rel)
                if rel <= tol:
                    continue
                zeta = (beta - alpha) / (2.0 * gamma)
                t = np.sign(zeta) / (abs(zeta) + float(np.sqrt(1.0 + zeta * zeta)))
                c = 1.0 / float(np.sqrt(1.0 + t * t))
                s = c * t
                wp = W[:, p].copy()
                wq = W[:, q].copy()
                W[:, p] = c * wp - s * wq
                W[:, q] = s * wp + c * wq
                vp = V[:, p].copy()
                vq = V[:, q].copy()
                V[:, p] = c * vp - s * vq
                V[:, q] = s * vp + c * vq
        if off <= tol:
            converged = True
            break
    sigma = np.sqrt(np.sum(W * W, axis=0))
    U = W / np.where(sigma > 0.0, sigma, 1.0)
    U = _complete_orthonormal(U)
    U, sigma, V = sort_and_fix_signs(U, sigma, V)
    return SVDFit(
        U=U, s=sigma, V=V, method="jacobi", converged=converged, iterations=sweeps,
        detail={"off_diagonal": off, "tol": tol},
    )


# --------------------------------------------------------------------------- #
# 双边（Kogbetliantz）旋转
# --------------------------------------------------------------------------- #
def _svd_2x2(a: float, b: float, c: float, d: float, *, tol: float) -> tuple[float, float, float, float, float, float]:
    """``2 x 2`` 矩阵 ``[[a,b],[c,d]]`` 的精确 SVD：返回 ``(c1, s1, c2, s2, s1v, s2v)``。

    ``G2ᵀ B G1 = diag(s1v, s2v)``，其中 ``G1 = [[c1,s1],[-s1,c1]]``（右，``B <- B G1``）、
    ``G2 = [[c2,s2],[-s2,c2]]``（左，``B <- G2 B``）。做法是：先右旋转让两列正交
    （2x2 Gram 矩阵的特征向量），再左旋转把第一列转到 x 轴——
    此时第二列自动落在 y 轴上（列已正交），两个对角元就是奇异值（可带符号）。
    """
    alpha = a * a + c * c
    beta = b * b + d * d
    gamma = a * b + c * d
    c1, s1 = 1.0, 0.0
    if gamma != 0.0 and abs(gamma) > tol * float(np.sqrt(max(alpha * beta, 0.0))):
        zeta = (beta - alpha) / (2.0 * gamma)
        t = np.sign(zeta) / (abs(zeta) + float(np.sqrt(1.0 + zeta * zeta)))
        c1 = 1.0 / float(np.sqrt(1.0 + t * t))
        s1 = c1 * t
    # 右旋转后的两列：col1' = c1 col1 - s1 col2，col2' = s1 col1 + c1 col2
    a1 = c1 * a - s1 * b
    c1r = c1 * c - s1 * d
    b1 = s1 * a + c1 * b
    d1 = s1 * c + c1 * d
    c2, s2, r1 = givens_rotation(a1, c1r)
    # 第二列（已与第一列正交）经同一左旋转后落在 y 轴上
    r2 = -s2 * b1 + c2 * d1
    return c1, s1, c2, s2, r1, r2


def kogbetliantz_svd(A: Any, *, max_sweeps: int = 60, tol: float | None = None) -> SVDFit:
    """双边 Jacobi（Kogbetliantz）：每次对 ``2 x 2`` 子块做**精确** SVD 并左右同时旋转。

    与单边 Jacobi 的区别：单边只动列，双边还动行，所以对**方阵**收敛更快、
    也对"行列都稀疏"的结构更友好；代价是每个子块要多算一次行内积。

    双边的经典弱点是**重奇异值附近收敛慢**（同一对列之间来回旋转），
    本章 pitfalls 第 8 条记录了实测的扫描次数对比。

    ``m > n`` 时先用自研 Householder QR 归约到方阵 ``R``（正交变换不改变奇异值），
    再对 ``R`` 做双边旋转；``m < n`` 时转置。``detail["reduced_by_qr"]`` 记录这条分支。
    """
    M = _as_matrix(A)
    m, n = M.shape
    if m < n:
        fit = kogbetliantz_svd(M.T, max_sweeps=max_sweeps, tol=tol)
        return SVDFit(
            U=fit.V, s=fit.s, V=fit.U, method="kogbetliantz", converged=fit.converged,
            iterations=fit.iterations, detail=fit.detail,
        )
    if m > n:
        q = householder_qr(M)
        fit = kogbetliantz_svd(q.R, max_sweeps=max_sweeps, tol=tol)
        detail = dict(fit.detail)
        detail["reduced_by_qr"] = 1.0
        return SVDFit(
            U=q.Q @ fit.U, s=fit.s, V=fit.V, method="kogbetliantz", converged=fit.converged,
            iterations=fit.iterations, detail=detail,
        )
    if tol is None:
        tol = 8.0 * _U
    B = np.array(M, dtype=float, copy=True)
    U = np.eye(m)
    V = np.eye(n)
    converged = False
    sweeps = 0
    off = float("inf")
    scale0 = float(np.sqrt(np.sum(B * B))) or 1.0
    for sweep in range(int(max_sweeps)):
        sweeps = sweep + 1
        for p in range(n - 1):
            for q in range(p + 1, n):
                c1, s1, c2, s2, _r1, _r2 = _svd_2x2(
                    B[p, p], B[p, q], B[q, p], B[q, q], tol=tol
                )
                if (c1, s1) != (1.0, 0.0):
                    # B <- B G1：col_p' = c1 col_p - s1 col_q
                    B[:, [p, q]] = np.column_stack(
                        [c1 * B[:, p] - s1 * B[:, q], s1 * B[:, p] + c1 * B[:, q]]
                    )
                    V[:, [p, q]] = np.column_stack(
                        [c1 * V[:, p] - s1 * V[:, q], s1 * V[:, p] + c1 * V[:, q]]
                    )
                if (c2, s2) != (1.0, 0.0):
                    # B <- G2 B：row_p' = c2 row_p + s2 row_q；同时 U <- U G2ᵀ
                    B[[p, q], :] = np.vstack(
                        [c2 * B[p, :] + s2 * B[q, :], -s2 * B[p, :] + c2 * B[q, :]]
                    )
                    U[:, [p, q]] = np.column_stack(
                        [c2 * U[:, p] + s2 * U[:, q], -s2 * U[:, p] + c2 * U[:, q]]
                    )
        # 非对角量（相对）：**直接**取非对角元——用 ‖B‖²_F - Σd² 会灾难性抵消
        # （两个 O(n) 的量相减），在 1e-8 量级伪造出一个假的收敛停滞。
        off_mat = B - np.diag(np.diag(B))
        off = float(np.sqrt(np.sum(off_mat * off_mat))) / scale0
        if off <= tol:
            converged = True
            break
    diag = np.diag(B).copy()
    sigma = np.abs(diag)
    signs = np.where(diag < 0.0, -1.0, 1.0)
    U = U[:, :n] * signs
    V = V[:, :n]
    U = _complete_orthonormal(U)
    U, sigma, V = sort_and_fix_signs(U, sigma, V)
    return SVDFit(
        U=U, s=sigma, V=V, method="kogbetliantz", converged=converged, iterations=sweeps,
        detail={"off_diagonal": off, "tol": tol},
    )


# --------------------------------------------------------------------------- #
# Golub--Kahan：双对角化 + 隐式位移 QR
# --------------------------------------------------------------------------- #
def bidiagonalize(A: Any) -> BidiagonalFit:
    """Householder 双对角化：``A = U1 B V1ᵀ``，``B`` 只有对角与一条次对角线非零。

    左反射清零一列的对角以下，右反射清零一行"次对角以右"的元素；每步都用
    ``alpha = -sign(x_1)‖x‖`` 的稳定侧写（见 :mod:`numcore.linalg.qr`）。

    ``m < n`` 时走转置：把 ``Aᵀ``（高矩阵）化成的上双对角 ``B`` 转置回来，
    得到 ``A = U1 B V1ᵀ`` 中的**下**双对角 ``B``，并在 :class:`BidiagonalFit` 里
    用 ``orientation="lower"`` 标出朝向。这不是偷懒：宽矩阵**不存在** r×r 的上双对角形式
    （最后一行的尾巴经反射只能变成 ``[b, 0, ...]``，那个 ``b`` 落在第 r 列之外），
    硬按上双对角写会得到一个错的因子。
    """
    M0 = _as_matrix(A)
    m, n = M0.shape
    if m < n:
        fit = bidiagonalize(M0.T)
        return BidiagonalFit(
            U1=fit.V1, alpha=fit.alpha, beta=fit.beta, V1=fit.U1, n_ops=fit.n_ops, orientation="lower"
        )
    M = np.array(M0, dtype=float, copy=True)
    r = min(m, n)
    U1 = np.eye(m)
    V1 = np.eye(n)
    alpha = np.zeros(r, dtype=float)
    beta = np.zeros(max(r - 1, 0), dtype=float)
    ops = 0
    for j in range(r):
        v, a = householder_vector(M[j:, j])
        if np.any(v):
            M[j:, j:] -= 2.0 * np.outer(v, v @ M[j:, j:])
            U1[:, j:] -= 2.0 * np.outer(U1[:, j:] @ v, v)
            ops += 2 * (m - j) + 2 * (m - j) * (n - j)
        alpha[j] = a
        if j + 1 < n:
            w, b = householder_vector(M[j, j + 1 :])
            if np.any(w):
                M[j:, j + 1 :] -= 2.0 * np.outer(M[j:, j + 1 :] @ w, w)
                V1[:, j + 1 :] -= 2.0 * np.outer(V1[:, j + 1 :] @ w, w)
                ops += 2 * (n - j - 1) + 2 * (m - j) * (n - j - 1)
            if j < r - 1:
                beta[j] = b
            # 显式归零：把"应当精确为零"的位置钉死，避免舍入噪声破坏双对角结构
            M[j, j + 1 :] = 0.0
            if j < r - 1:
                M[j, j + 1] = b
    return BidiagonalFit(U1=U1, alpha=alpha, beta=beta, V1=V1, n_ops=ops, orientation="upper")


def _wilkinson_shift(t11: float, t12: float, t22: float) -> float:
    """2x2 对称阵 ``[[t11,t12],[t12,t22]]`` 中**靠近 t22** 的特征值（Wilkinson 位移）。

    用 ``mu = t22 - t12^2 / (delta + sign(delta) sqrt(delta^2 + t12^2))``
    （``delta = (t11 - t22)/2``），避免两个相近特征值相减的抵消。
    """
    if t12 == 0.0:
        return t22
    delta = 0.5 * (t11 - t22)
    root = float(np.sqrt(delta * delta + t12 * t12))
    if root == 0.0:
        return t22
    denom = delta + (root if delta >= 0.0 else -root)
    if denom == 0.0:
        return t22
    return t22 - (t12 * t12) / denom


def bidiagonal_svd(
    alpha: Any,
    beta: Any,
    *,
    tol: float | None = None,
    max_iter: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """上双对角阵的隐式位移 QR：返回 ``(s, U2, V2, info)``，``B = U2 diag(s) V2ᵀ``。

    算法（Golub & Van Loan, Alg. 8.6.1）：对活跃块 ``B[p:q, p:q]`` 取 ``BᵀB`` 尾 2x2 的
    Wilkinson 位移 ``mu``，令 ``y = alpha_p^2 - mu``、``z = alpha_p beta_p``，然后
    交替左右 Givens 旋转把 bulge 一路赶到右下角；超对角元可忽略时**收缩**（deflate）。

    关键点：位移只用 ``B`` 的元素算（``BᵀB`` 的两个元素恰好是 ``alpha``、``beta`` 的组合），
    **从不真正形成** ``BᵀB``——那会把条件数平方。
    """
    a = np.asarray(alpha, dtype=float).ravel().copy()
    b = np.asarray(beta, dtype=float).ravel().copy()
    n = a.size
    if b.size != max(n - 1, 0):
        raise ValueError(f"beta 长度应为 {max(n - 1, 0)}，得到 {b.size}")
    if tol is None:
        tol = 100.0 * _U
    if max_iter is None:
        max_iter = 200 * max(n, 1)
    B = np.diag(a)
    if n > 1:
        B = B + np.diag(b, 1)
    U2 = np.eye(n)
    V2 = np.eye(n)
    q = n
    it = 0
    while q > 1 and it < int(max_iter):
        # 1) 收缩：把可忽略的超对角元置零
        for i in range(q - 1, 0, -1):
            if abs(B[i - 1, i]) <= tol * (abs(B[i - 1, i - 1]) + abs(B[i, i])):
                B[i - 1, i] = 0.0
        # 2) 找活跃块 [p, q)
        p = q - 1
        while p > 0 and B[p - 1, p] != 0.0:
            p -= 1
        if p == q - 1:
            if B[q - 1, q - 1] < 0.0:
                B[q - 1, q - 1] = -B[q - 1, q - 1]
                U2[:, q - 1] = -U2[:, q - 1]
            q -= 1
            continue
        it += 1
        # 3) 位移与 chase
        aa = B[q - 2, q - 2]
        bb = B[q - 2, q - 1]
        cc = B[q - 1, q - 1]
        mu = _wilkinson_shift(aa * aa, aa * bb, bb * bb + cc * cc)
        y = B[p, p] * B[p, p] - mu
        z = B[p, p] * B[p, p + 1]
        for k in range(p, q - 1):
            c_r, s_r, _ = givens_rotation(y, z)
            # 右旋转取 (c, -s)：它对应 T = BᵀB 上隐式 QR 的第一步 G1ᵀ T G1
            c1, s1 = c_r, -s_r
            G1 = givens_matrix(n, k, k + 1, c1, s1)
            B = B @ G1
            V2 = V2 @ G1
            y = B[k, k]
            z = B[k + 1, k]
            c2, s2, _ = givens_rotation(y, z)
            G2 = givens_matrix(n, k, k + 1, c2, s2)
            B = G2 @ B
            U2 = U2 @ G2.T
            if k < q - 2:
                y = B[k, k + 1]
                z = B[k, k + 2]
    diag = np.diag(B).copy()
    sigma = np.abs(diag)
    signs = np.where(diag < 0.0, -1.0, 1.0)
    U2 = U2 * signs
    info = {
        "iterations": float(it),
        "max_iter": float(max_iter),
        "converged": 1.0 if q <= 1 else 0.0,
        "tol": float(tol),
        "off_diagonal": float(np.max(np.abs(B - np.diag(np.diag(B))))) if n else 0.0,
    }
    return sigma, U2, V2, info


def golub_kahan_svd(A: Any, *, tol: float | None = None, max_iter: int | None = None) -> SVDFit:
    """Golub--Kahan SVD：先 :func:`bidiagonalize`，再 :func:`bidiagonal_svd`，最后合因子。

    ``A = U1 B V1ᵀ``、``B = U2 diag(s) V2ᵀ``，于是 ``A = (U1 U2) diag(s) (V1 V2)ᵀ``。
    thin 形式由 ``r = min(m,n)`` 截断得到。
    """
    M = _as_matrix(A)
    m, n = M.shape
    r = min(m, n)
    if m < n:
        # 扁平矩阵：对转置做同一件事再交换因子（双对角化只对 m >= n 给出 r x r 的 B）
        fit = golub_kahan_svd(M.T, tol=tol, max_iter=max_iter)
        return SVDFit(
            U=fit.V, s=fit.s, V=fit.U, method="golub_kahan", converged=fit.converged,
            iterations=fit.iterations, detail=fit.detail,
        )
    bi = bidiagonalize(M)
    sigma, U2, V2, info = bidiagonal_svd(bi.alpha, bi.beta, tol=tol, max_iter=max_iter)
    U = bi.U1_thin @ U2
    V = bi.V1_thin @ V2
    U = U[:, :r]
    V = V[:, :r]
    U, sigma, V = sort_and_fix_signs(U, sigma[:r], V)
    detail = {k: float(v) for k, v in info.items()}
    detail["n_ops"] = float(bi.n_ops)
    return SVDFit(
        U=U, s=sigma, V=V, method="golub_kahan",
        converged=bool(info["converged"] > 0.5), iterations=int(info["iterations"]),
        detail=detail,
    )


# --------------------------------------------------------------------------- #
# 秩、伪逆、截断
# --------------------------------------------------------------------------- #
def numerical_rank(s: Any, m: int, n: int, *, rtol: float | None = None) -> int:
    """数值秩：``s_i > rtol * s_1`` 的个数，缺省 ``rtol = max(m,n) u``（LAPACK 口径）。

    这条判据里"容差"是本质的一半：同一矩阵在 ``rtol = 1e-10`` 下秩为 4、
    在 ``rtol = 1e-16`` 下秩为 5 是**同一件事实的两种说法**（pitfalls 第 4 条）。
    """
    sv = np.asarray(s, dtype=float).ravel()
    if sv.size == 0:
        return 0
    if rtol is None:
        rtol = max(int(m), int(n)) * _U
    cutoff = float(rtol) * float(sv[0])
    return int(np.count_nonzero(sv > cutoff))


def spectral_condition(s: Any) -> float:
    """``kappa_2 = s_1 / s_r``；``s_r = 0`` 时返回 ``inf``（矩阵奇异）。"""
    sv = np.asarray(s, dtype=float).ravel()
    if sv.size == 0:
        return float("nan")
    lo = float(sv[-1])
    if lo == 0.0:
        return float("inf")
    return float(sv[0]) / lo


def pseudoinverse(U: Any, s: Any, V: Any, *, rtol: float | None = None) -> np.ndarray:
    """由 SVD 因子算 Moore--Penrose 伪逆 ``A⁺ = V diag(1/s_i) Uᵀ``（截断掉 ``s_i <= rtol s_1``）。"""
    Uu = np.asarray(U, dtype=float)
    sv = np.asarray(s, dtype=float).ravel()
    Vv = np.asarray(V, dtype=float)
    if rtol is None:
        m, n = Uu.shape[0], Vv.shape[0]
        rtol = max(m, n) * _U
    cutoff = float(rtol) * (float(sv[0]) if sv.size else 0.0)
    inv = np.where(sv > cutoff, 1.0 / np.where(sv > cutoff, sv, 1.0), 0.0)
    return (Vv * inv) @ Uu.T


def truncated_solution(U: Any, s: Any, V: Any, b: Any, k: int) -> np.ndarray:
    """截断 SVD 解 ``x_k = sum_{i<=k} (u_iᵀb / s_i) v_i``（``k = 0`` 时给零向量）。"""
    Uu = np.asarray(U, dtype=float)
    sv = np.asarray(s, dtype=float).ravel()
    Vv = np.asarray(V, dtype=float)
    bb = np.asarray(b, dtype=float).ravel()
    kk = max(0, min(int(k), sv.size))
    coef = Uu[:, :kk].T @ bb
    scales = np.where(sv[:kk] != 0.0, coef / np.where(sv[:kk] != 0.0, sv[:kk], 1.0), 0.0)
    return Vv[:, :kk] @ scales


def moore_penrose_defects(A: Any, A_plus: Any) -> dict[str, float]:
    """四个 Moore--Penrose 条件的**实测**缺陷（都应当只有舍入量级）。

    ``A A⁺ A = A``、``A⁺ A A⁺ = A⁺``、``(A A⁺)ᵀ = A A⁺``、``(A⁺ A)ᵀ = A⁺ A``。
    """
    M = np.asarray(A, dtype=float)
    P = np.asarray(A_plus, dtype=float)
    scale_a = float(np.max(np.abs(M))) if M.size else 1.0
    scale_p = float(np.max(np.abs(P))) if P.size else 1.0
    aa = P @ M
    return {
        "A_A+_A_minus_A": float(np.max(np.abs(M @ P @ M - M))) / max(scale_a, 1e-300),
        "A+_A_A+_minus_A+": float(np.max(np.abs(P @ M @ P - P))) / max(scale_p, 1e-300),
        "A_A+_symmetry": float(np.max(np.abs(M @ P - (M @ P).T))) / max(scale_a * scale_p, 1e-300),
        "A+_A_symmetry": float(np.max(np.abs(aa - aa.T))) / max(scale_p * scale_a, 1e-300),
    }


def singular_vector_residuals(A: Any, U: Any, s: Any, V: Any) -> np.ndarray:
    """逐对残差 ``max(‖A v_i - s_i u_i‖, ‖Aᵀ u_i - s_i v_i‖)``（奇异向量的直接证据）。"""
    M = np.asarray(A, dtype=float)
    Uu = np.asarray(U, dtype=float)
    sv = np.asarray(s, dtype=float).ravel()
    Vv = np.asarray(V, dtype=float)
    out = np.zeros(sv.size, dtype=float)
    for i in range(sv.size):
        r1 = M @ Vv[:, i] - sv[i] * Uu[:, i]
        r2 = M.T @ Uu[:, i] - sv[i] * Vv[:, i]
        out[i] = max(float(np.max(np.abs(r1))), float(np.max(np.abs(r2))))
    return out


def relative_singular_value_error(s: Any, s_ref: Any) -> float:
    """``max_i |s_i - s_i^ref| / s_i^ref``（与参照物对拍用；``s`` 必须同序同长）。"""
    a = np.asarray(s, dtype=float).ravel()
    b = np.asarray(s_ref, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError(f"奇异值个数不一致：{a.size} vs {b.size}")
    safe = np.where(b > 0.0, b, 1.0)
    return float(np.max(np.abs(a - b) / safe))
