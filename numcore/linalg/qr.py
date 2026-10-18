"""1.3 章 QR 正交化内核：Householder 反射、Givens 旋转、列主元秩揭示、QR 最小二乘。

数学对象
--------
* **Householder 反射** ``H = I - 2 v vᵀ/(vᵀ v)``：取 ``v = x - alpha e_1``、
  ``alpha = -sign(x_1)‖x‖₂`` 时有 ``H x = alpha e_1``，把一列的下方元素**一次**清零；
* **QR 分解** ``A = Q R``（``Q`` 列正交、``R`` 上三角）；列主元形式 ``A P = Q R``，
  且 ``|r_11| >= |r_22| >= ...``——这是"秩揭示"的全部依据；
* **Givens 旋转** ``G(i,j,c,s)``：``c^2 + s^2 = 1``、``det G = 1``，一次清零一个元素
  （Householder 是"整列"，Givens 是"单个"，后者适合稀疏与逐步消元）；
* **正交性度量** ``‖Aᵀ r‖ / (‖A‖_F ‖r‖)``：最小二乘残差到底有多垂直。

出处：Golub & Van Loan, *Matrix Computations* 4th ed., §5.1--5.4；Higham,
*Accuracy and Stability of Numerical Algorithms* 2nd ed., §19--20。

POLICY 说明（P2/P3）：本模块不许调用 ``numpy.linalg`` 的任何东西，所有分解都是自研；
矩阵乘法只用 ``@``（BLAS 原语）与逐元素原语。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "QRResult",
    "PivotedQRResult",
    "GivensQRResult",
    "householder_vector",
    "householder_matrix",
    "householder_qr",
    "column_pivot_qr",
    "givens_rotation",
    "givens_matrix",
    "givens_qr",
    "qr_least_squares",
    "qr_residual",
    "back_substitute",
    "orthogonality_defect",
    "orthogonality_measure",
    "qr_reconstruction_error",
    "qr_rank",
    "qr_flops",
    "column_norms",
]


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _as_matrix(A: Any) -> np.ndarray:
    M = np.asarray(A, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"需要二维矩阵，得到 {M.ndim} 维")
    if M.size == 0:
        raise ValueError("空矩阵")
    return M


def _max_abs(A: np.ndarray) -> float:
    return float(np.max(np.abs(A))) if A.size else 0.0


def _vec_norm(x: np.ndarray) -> float:
    """欧几里得范数 ``sqrt(sum x_i^2)``（逐元素原语，自研）。"""
    return float(np.sqrt(np.dot(x, x)))


def _frobenius(A: np.ndarray) -> float:
    return float(np.sqrt(np.sum(A * A)))


def column_norms(A: Any) -> np.ndarray:
    """各列的欧几里得范数 ``‖A(:,j)‖₂``。"""
    M = _as_matrix(A)
    return np.sqrt(np.sum(M * M, axis=0))


# --------------------------------------------------------------------------- #
# Householder 反射
# --------------------------------------------------------------------------- #
def householder_vector(x: Any) -> tuple[np.ndarray, float]:
    """返回把 ``x`` 映到 ``alpha e_1`` 的反射向量 ``(v, alpha)``，``‖v‖₂ = 1``。

    取 ``alpha = -sign(x_1)‖x‖₂``（而不是 ``+‖x‖₂``）是为了**避免相减抵消**：
    若 ``x_1 > 0`` 而取正号，``v_1 = x_1 - ‖x‖`` 会在 ``x`` 接近 ``‖x‖e_1`` 时
    丢掉全部有效位——这正是本章 pitfalls 里"看起来很稳的公式其实不稳"的例子。
    """
    xv = np.asarray(x, dtype=float).ravel()
    nx = _vec_norm(xv)
    if nx == 0.0:
        return np.zeros_like(xv), 0.0
    alpha = -nx if xv[0] >= 0.0 else nx
    v = xv.copy()
    v[0] -= alpha
    nv = _vec_norm(v)
    if nv == 0.0:  # x 已经是 alpha e_1，无需反射
        return np.zeros_like(xv), alpha
    return v / nv, alpha


def householder_matrix(v: Any) -> np.ndarray:
    """由单位向量 ``v`` 构造 ``H = I - 2 v vᵀ``（对称、正交、对合）。"""
    vv = np.asarray(v, dtype=float).ravel()
    n = vv.size
    return np.eye(n) - 2.0 * np.outer(vv, vv)


@dataclass(frozen=True)
class QRResult:
    """``A = Q R``（thin）：``Q`` 是 ``m x k`` 列正交矩阵，``R`` 是 ``k x n`` 上三角，``k = min(m,n)``。"""

    Q: np.ndarray
    R: np.ndarray
    reflectors: tuple[np.ndarray, ...] = ()
    n_ops: int = 0
    m: int = 0
    n: int = 0

    @property
    def k(self) -> int:
        return int(self.Q.shape[1])

    def solve(self, b: Any) -> np.ndarray:
        """用已算好的 ``QR`` 解方阵方程 ``A x = b``（``R`` 需可逆）。"""
        return back_substitute(self.R, self.Q.T @ np.asarray(b, dtype=float))


def householder_qr(A: Any, *, collect_reflectors: bool = False) -> QRResult:
    """Householder QR：``A = Q R``，``Q`` 列正交、``R`` 上三角（``Q`` 取 thin 形式）。

    实现要点（决定后向稳定性的两条）：

    1. 反射向量**归一化**后再用：``H = I - 2 v vᵀ``，更新只做一次 rank-1 修正，
       不做任何相减型消元；
    2. 用 ``alpha = -sign(x_1)‖x‖`` 侧写反射，避免 ``v_1`` 的灾难性抵消。

    实测：``‖Q R - A‖_F / (u ‖A‖_F)`` 在 n <= 64 上不超过几个 n 的常数倍
    （见 ``tests/1.3/test_qr_svd.py`` 的 A 类断言）。
    """
    M = np.array(A, dtype=float, copy=True)
    m, n = M.shape
    k = min(m, n)
    Q = np.eye(m)
    reflectors: list[np.ndarray] = []
    ops = 0
    for j in range(k):
        v, _alpha = householder_vector(M[j:, j])
        if not np.any(v):
            continue
        # 右下块：M[j:, j:] <- M[j:, j:] - 2 v (vᵀ M[j:, j:])
        M[j:, j:] -= 2.0 * np.outer(v, v @ M[j:, j:])
        # 累积 Q：Q[:, j:] <- Q[:, j:] - 2 (Q[:, j:] v) vᵀ
        Q[:, j:] -= 2.0 * np.outer(Q[:, j:] @ v, v)
        if collect_reflectors:
            reflectors.append(v.copy())
        ops += 2 * (m - j) + 2 * (m - j) * (n - j) + 2 * m * (k - j)
    R = np.triu(M[:k, :])
    return QRResult(Q=Q[:, :k], R=R, reflectors=tuple(reflectors), n_ops=ops, m=m, n=n)


def qr_flops(m: int, n: int, *, accumulate_q: bool = True) -> int:
    """Householder QR 的**运算次数闭式**（与 :func:`householder_qr` 的计数口径一致）。

    每步（``j = 0..k-1``，``k = min(m,n)``）：

    * ``2(m-j)``：构造并归一化反射向量；
    * ``2(m-j)(n-j)``：右下块的 rank-1 更新；
    * ``2m(k-j)``：把反射累积进 ``Q``（``accumulate_q=False`` 时不计，
      对应"只做分解、稍后再形成 Q"的用法）。

    ``m = n`` 时主项为 ``(5/3)n^3``（含 Q 累积）/ ``(2/3)n^3``（不含），两者斜率都是 3。
    """
    k = min(int(m), int(n))
    total = 0
    for j in range(k):
        mj = int(m) - j
        nj = int(n) - j
        total += 2 * mj + 2 * mj * nj
        if accumulate_q:
            total += 2 * int(m) * (k - j)
    return total


# --------------------------------------------------------------------------- #
# 列主元 QR：秩揭示
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PivotedQRResult:
    """``A P = Q R``：列主元 QR，``|r_11| >= |r_22| >= ...``，``pivots`` 即 ``|r_jj|``。"""

    Q: np.ndarray
    R: np.ndarray
    perm: np.ndarray
    pivots: np.ndarray
    col_norms_remaining: np.ndarray
    rank: int = 0
    tol: float = 0.0
    m: int = 0
    n: int = 0
    n_ops: int = 0

    def permuted_A(self, A: Any) -> np.ndarray:
        return np.asarray(A, dtype=float)[:, self.perm]


def column_pivot_qr(A: Any, *, tol: float | None = None) -> PivotedQRResult:
    """列主元 Householder QR（秩揭示）：每步把**剩余列范数最大**的列换到主元位。

    判据与容差：``tol = max(m,n) * u * |r_11|``（LAPACK 的 rcond 口径）。
    数值秩取 ``|r_jj| > tol`` 的个数。这个判据是**启发式**，能证明的只有三条：

    * 主元序列单调不增（每步取的是最大剩余列范数）；
    * ``|r_11| >= sigma_1 / sqrt(n)``（``‖A‖_F <= sqrt(n)‖A‖_2`` 且最大列范数 ``>= ‖A‖_F/sqrt(n)``）；
    * 体积界 ``prod_{j<=k}|r_jj| = |det A_k| <= prod_{j<=k} sigma_j``（``A_k`` 是前 k 个主元列）。

    反过来的界**都不成立**：实测 ``|r_kk|·sqrt(n-k+1)/sigma_k`` 会低到 0.62、
    ``sigma_k/(|r_kk|·sqrt(n-k+1))`` 会低到 0.27，Kahan 矩阵更是经典反例——
    ``|r_nn|`` 可以远大于 ``sigma_n``。所以"秩"必须连同容差一起说（见 pitfalls）。
    """
    M = np.array(A, dtype=float, copy=True)
    m, n = M.shape
    k = min(m, n)
    Q = np.eye(m)
    perm = np.arange(n)
    sq = np.sum(M * M, axis=0)  # 列范数平方（增量更新）
    pivots = np.zeros(k, dtype=float)
    ops = 0
    for j in range(k):
        tail = sq[j:]
        p = int(np.argmax(tail)) + j
        if p != j:
            M[:, [j, p]] = M[:, [p, j]]
            perm[[j, p]] = perm[[p, j]]
            sq[[j, p]] = sq[[p, j]]
        v, _alpha = householder_vector(M[j:, j])
        if np.any(v):
            M[j:, j:] -= 2.0 * np.outer(v, v @ M[j:, j:])
            Q[:, j:] -= 2.0 * np.outer(Q[:, j:] @ v, v)
            ops += 2 * (m - j) + 2 * (m - j) * (n - j) + 2 * m * (k - j)
        pivots[j] = abs(M[j, j])
        if j + 1 < n:
            # 列范数下更新：新范数² = 旧范数² - 已消元分量²（截断到 0 防舍入负值）
            sq[j + 1 :] = np.maximum(sq[j + 1 :] - M[j, j + 1 :] ** 2, 0.0)
    if tol is None:
        u = float(np.finfo(float).eps) / 2.0
        tol = max(m, n) * u * (pivots[0] if k else 0.0)
    rank = int(np.count_nonzero(pivots > tol))
    R = np.triu(M[:k, :])
    return PivotedQRResult(
        Q=Q[:, :k],
        R=R,
        perm=perm,
        pivots=pivots,
        col_norms_remaining=np.sqrt(sq),
        rank=rank,
        tol=float(tol),
        m=m,
        n=n,
        n_ops=ops,
    )


def qr_rank(pivots: Any, *, tol: float | None = None) -> int:
    """由主元序列 ``|r_jj|`` 判定数值秩（缺省容差 ``max(m,n) u |r_11|`` 的退化版）。

    单独暴露出来是为了让"秩判据"可以被显式检验：容差取多少，秩就是多少——
    这正是 pitfalls 里"秩不是矩阵的性质，而是矩阵加容差的性质"那条。
    """
    p = np.asarray(pivots, dtype=float).ravel()
    if p.size == 0:
        return 0
    if tol is None:
        u = float(np.finfo(float).eps) / 2.0
        tol = p.size * u * float(p[0])
    return int(np.count_nonzero(p > tol))


# --------------------------------------------------------------------------- #
# Givens 旋转
# --------------------------------------------------------------------------- #
def givens_rotation(a: float, b: float) -> tuple[float, float, float]:
    """返回 ``(c, s, r)`` 使 ``[[c, s], [-s, c]] [a; b] = [r; 0]``，``c² + s² = 1``。

    ``r`` 取 ``a`` 的符号：``r = sign(a)·hypot(a, b)``（与 LAPACK dlartg 一致）——
    这样 ``c = a/r``、``s = b/r`` 都不会相减，``c·a + s·b = r`` 是加法。

    用 ``hypot`` 式的稳定写法（先比大小再缩放），避免 ``sqrt(a²+b²)`` 在
    ``a`` 或 ``b`` 极大/极小时上溢下溢——这是 Givens 相对 Householder 唯一需要小心的地方。
    """
    a = float(a)
    b = float(b)
    if b == 0.0:
        return 1.0, 0.0, a
    if a == 0.0:
        return 0.0, 1.0 if b > 0 else -1.0, abs(b)
    scale = max(abs(a), abs(b))
    r = scale * float(np.sqrt((a / scale) ** 2 + (b / scale) ** 2))
    c = a / r
    s = b / r
    if a < 0.0:  # 约定 r 与 a 同号（r = sign(a)·hypot，与 LAPACK dlartg 一致）
        c, s, r = -c, -s, -r
    return c, s, r


def givens_matrix(n: int, i: int, j: int, c: float, s: float) -> np.ndarray:
    """``n x n`` 的 Givens 旋转 ``G``：``G[i,i]=c, G[i,j]=s, G[j,i]=-s, G[j,j]=c``，其余同单位阵。

    ``det G = c² + s² = 1``（B 类断言之一）。
    """
    if not (0 <= i < n and 0 <= j < n) or i == j:
        raise ValueError(f"非法的旋转下标 (i, j) = ({i}, {j})，n = {n}")
    G = np.eye(n)
    G[i, i] = c
    G[j, j] = c
    G[i, j] = s
    G[j, i] = -s
    return G


@dataclass(frozen=True)
class GivensQRResult:
    """逐个清零得到的 QR：``Q`` 是 ``m x m`` 正交阵（旋转之积），``R`` 是 ``m x n`` 上三角。"""

    Q: np.ndarray
    R: np.ndarray
    rotations: tuple[tuple[int, int, float, float], ...] = field(default_factory=tuple)
    n_ops: int = 0

    @property
    def n_rotations(self) -> int:
        return len(self.rotations)


def givens_qr(A: Any) -> GivensQRResult:
    """用 Givens 旋转做 QR：``A = Q R``（``Q`` 为 ``m x m``）。

    清零顺序是按列自下而上（``(j-1, j), (j-2, j), ...``），每次只动两行——
    这是"稀疏 QR"与"逐步消元"动画的原型（Householder 一次动整列，画不出来中间态）。
    """
    M = np.array(A, dtype=float, copy=True)
    m, n = M.shape
    Q = np.eye(m)
    rotations: list[tuple[int, int, float, float]] = []
    ops = 0
    for col in range(min(m, n)):
        for row in range(m - 1, col, -1):
            a = M[row - 1, col]
            b = M[row, col]
            if b == 0.0:
                continue
            c, s, _r = givens_rotation(a, b)
            G = givens_matrix(m, row - 1, row, c, s)
            M = G @ M
            Q = Q @ G.T
            rotations.append((row - 1, row, c, s))
            ops += 6 * n + 6 * m
    return GivensQRResult(Q=Q, R=np.triu(M), rotations=tuple(rotations), n_ops=ops)


# --------------------------------------------------------------------------- #
# 三角回代与 QR 最小二乘
# --------------------------------------------------------------------------- #
def back_substitute(R: Any, y: Any) -> np.ndarray:
    """上三角回代 ``R x = y``；遇到零对角元抛 :class:`ValueError`（矩阵奇异）。"""
    Rt = np.asarray(R, dtype=float)
    b = np.asarray(y, dtype=float).ravel()
    n = Rt.shape[1]
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        if Rt[i, i] == 0.0:
            raise ValueError(f"R 的第 {i} 个对角元为 0，矩阵奇异（请改用列主元 QR 或 SVD）")
        acc = b[i] - float(np.dot(Rt[i, i + 1 :], x[i + 1 :]))
        x[i] = acc / Rt[i, i]
    return x


def qr_residual(A: Any, x: Any, b: Any) -> np.ndarray:
    """最小二乘残差 ``r = b - A x``（原样返回向量，范数由调用方决定）。"""
    return np.asarray(b, dtype=float).ravel() - np.asarray(A, dtype=float) @ np.asarray(x, dtype=float).ravel()


def qr_least_squares(A: Any, b: Any, *, tol: float | None = None) -> np.ndarray:
    """QR 解满秩最小二乘 ``min_x ‖A x - b‖₂``（``m >= n``）：``R x = (Qᵀ b)_{1:n}``。

    为什么不能"先算 ``AᵀA`` 再解"：``kappa(AᵀA) = kappa(A)^2``，正规方程把误差放大
    平方倍（1.3 章的主线，见 :mod:`numcore.linalg.lstsq`）。
    这里用列主元 QR：秩亏时按 ``|r_jj| <= tol`` 截断，给出一个基本解，并在
    ``rank < n`` 时由调用方决定是否改用 SVD（截断 SVD 才给最小范数解）。
    """
    M = _as_matrix(A)
    m, n = M.shape
    if m < n:
        raise ValueError(f"QR 最小二乘要求 m >= n（超定或方阵），得到 {m} x {n}")
    piv = column_pivot_qr(M, tol=tol)
    y = piv.Q.T @ np.asarray(b, dtype=float).ravel()
    if piv.rank < n:
        raise ValueError(
            f"矩阵数值秩 {piv.rank} < {n}（|r_nn| = {piv.pivots[-1]:.3e} <= tol = {piv.tol:.3e}）："
            "QR 不再是满秩最小二乘，请用截断 SVD（numcore.linalg.svd）"
        )
    x_perm = back_substitute(piv.R, y)
    x = np.empty(n, dtype=float)
    x[piv.perm] = x_perm
    return x


# --------------------------------------------------------------------------- #
# 度量
# --------------------------------------------------------------------------- #
def orthogonality_defect(Q: Any) -> float:
    """``‖Qᵀ Q - I‖_max``：列正交性的缺陷（Householder/Givens 的核心不变量）。"""
    M = np.asarray(Q, dtype=float)
    k = M.shape[1]
    G = M.T @ M - np.eye(k)
    return _max_abs(G)


def orthogonality_measure(A: Any, r: Any) -> float:
    """``‖Aᵀ r‖₂ / (‖A‖_F ‖r‖₂)``：残差与列空间的正交性度量（无量纲，0 为理论值）。

    为什么用 Frobenius 范数：它不需要 SVD 就能算，且有 ``‖A‖₂ <= ‖A‖_F <= sqrt(n)‖A‖₂``，
    所以这个量在 2-范数口径下最多差 ``sqrt(n)`` 倍——足够用来验证"垂直"。
    """
    M = np.asarray(A, dtype=float)
    rv = np.asarray(r, dtype=float).ravel()
    nr = _vec_norm(rv)
    nA = _frobenius(M)
    if nr == 0.0 or nA == 0.0:
        return 0.0
    return _vec_norm(M.T @ rv) / (nA * nr)


def qr_reconstruction_error(A: Any, Q: Any, R: Any) -> float:
    """``‖Q R - A‖_max``：分解的重建误差（后向稳定性的直接证据）。"""
    return _max_abs(np.asarray(Q, dtype=float) @ np.asarray(R, dtype=float) - np.asarray(A, dtype=float))
