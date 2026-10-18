r"""预条件子：Jacobi / SSOR / ILU(0)，以及预条件前后的谱分布与条件数变化。

数学定义
--------
预条件就是把方程 :math:`Ax=b` 换成"同样解、但更好解"的方程。本模块统一采用
**左预条件**：解 :math:`M^{-1}Ax=M^{-1}b`，:math:`M` 对称正定时

.. math::
    M = D \quad\text{(Jacobi)},\qquad
    M = \frac{1}{\omega(2-\omega)}(D+\omega L)D^{-1}(D+\omega U)\quad\text{(SSOR)},\qquad
    M = \tilde L\tilde U\quad\text{(ILU(0))},

其中 :math:`A=D+L+U`（:math:`L` 严格下三角、:math:`U` 严格上三角），
:math:`\tilde L\tilde U` 是**只在** :math:`A` 的稀疏结构上做消元得到的不完全分解
（不允许产生任何填充）。

关键结论（也是本章反复要演示的）：

* 对 CG 而言，真正决定迭代数的是 :math:`M^{-1}A` 的**谱分布**，不是
  :math:`\kappa(M^{-1}A)` 一个数：谱聚成几簇时，迭代数远小于
  :math:`\tfrac12\sqrt\kappa\ln(2/\varepsilon)` 这条最坏界。
* Jacobi 只改对角尺度，对 Laplacian 这类"对角本来就均匀"的问题几乎没用；
  SSOR/ILU(0) 才真正压缩谱宽。
* **实测 κ 的唯一诚实做法**是量谱（本模块用 :math:`M^{-1}`-内积下的手写 Lanczos +
  Sturm 二分），而不是把 :math:`\kappa` 的定义式抄一遍——谱的**内部** Ritz 值收敛很慢，
  所以"用少数几步 Lanczos 就宣称量到了整个谱"是这类图最常见的谎言。

实现约束：P3（只用 numpy 容器与原语）、P4（随机数走 :func:`numcore.utils.rng`）、
P6（纯函数）；不完全分解与三角求解全部手写在 :class:`numcore.linalg.sparse.CSR` 上，
不调用任何现成稀疏求解器。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from numcore.linalg.krylov import inner_product, lanczos, matvec, norm2, tridiag_eigenvalues
from numcore.linalg.sparse import CSR, to_csr_list
from numcore.utils import rng

__all__ = [
    "JacobiPrecond",
    "SSORPrecond",
    "ILU0Precond",
    "build_precond",
    "PrecondSpectrum",
    "precond_spectrum",
    "kappa_estimate",
    "kappa_from_factor",
]


def _as_square(A: Any) -> np.ndarray:
    arr = np.asarray(A, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"需要方阵，得到形状 {arr.shape}")
    return arr


# --------------------------------------------------------------------------- #
# Jacobi
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class JacobiPrecond:
    """对角（Jacobi）预条件 :math:`M=\\operatorname{diag}(A)`；``apply`` 给 :math:`M^{-1}r`。"""

    diag: np.ndarray

    @classmethod
    def from_matrix(cls, A: Any) -> "JacobiPrecond":
        arr = _as_square(A)
        d = np.diag(arr).copy()
        if np.any(d == 0.0):
            raise ValueError("对角含 0：Jacobi 预条件子不可用")
        return cls(diag=d)

    def apply(self, r: Any) -> np.ndarray:
        return np.asarray(r, dtype=float).reshape(-1) / self.diag

    def matvec_M(self, x: Any) -> np.ndarray:
        """:math:`Mx`（预条件 Lanczos 要用 **M**-内积，见 :func:`~numcore.linalg.krylov.lanczos`）。"""
        return self.diag * np.asarray(x, dtype=float).reshape(-1)

    def to_dense(self) -> np.ndarray:
        return np.diag(self.diag)

    @property
    def nnz(self) -> int:
        return int(self.diag.size)


# --------------------------------------------------------------------------- #
# SSOR
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SSORPrecond:
    r"""对称超松弛预条件 :math:`M=\frac{1}{\omega(2-\omega)}(D+\omega L)D^{-1}(D+\omega U)`。

    ``apply`` 用恒等式
    :math:`M^{-1}=\omega(2-\omega)(D+\omega U)^{-1}D(D+\omega L)^{-1}`
    （两次三角回代 + 一次对角缩放）。:math:`\omega=1` 时退化为对称 Gauss--Seidel（SSGS）。
    """

    omega: float
    diag: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def from_matrix(cls, A: Any, omega: float = 1.0) -> "SSORPrecond":
        arr = _as_square(A)
        w = float(omega)
        if not (0.0 < w < 2.0):
            raise ValueError(f"SSOR 需要 0 < omega < 2，得到 {omega}")
        return cls(omega=w, diag=np.diag(arr).copy(), lower=np.tril(arr, -1).copy(), upper=np.triu(arr, 1).copy())

    def apply(self, r: Any) -> np.ndarray:
        rhs = np.asarray(r, dtype=float).reshape(-1).copy()
        n = rhs.size
        y = np.zeros(n, dtype=float)
        for i in range(n):
            s = float(self.lower[i] @ y)  # lower[i,i]=0，无需减自身
            y[i] = (rhs[i] - self.omega * s) / self.diag[i]
        z = self.diag * y
        w = np.zeros(n, dtype=float)
        for i in range(n - 1, -1, -1):
            s = float(self.upper[i] @ w)
            w[i] = (z[i] - self.omega * s) / self.diag[i]
        return self.omega * (2.0 - self.omega) * w

    def matvec_M(self, x: Any) -> np.ndarray:
        r"""显式用 :math:`M=\frac{1}{\omega(2-\omega)}(D+\omega L)D^{-1}(D+\omega U)` 乘向量。

        只需要两次**三角乘法**（不是回代）：先 :math:`(D+\omega U)x`、再乘 :math:`D^{-1}`、
        最后乘 :math:`(D+\omega L)`。预条件 Lanczos 的自伴内积是 :math:`\langle x,y\rangle_M=x^{\mathsf T}My`
        （不是 :math:`M^{-1}`-内积），所以这里必须能算 :math:`My` 本身。
        """
        v = np.asarray(x, dtype=float).reshape(-1)
        w = self.omega * (self.upper @ v) + self.diag * v
        w = w / self.diag
        w = self.omega * (self.lower @ w) + self.diag * w
        return w / (self.omega * (2.0 - self.omega))

    def to_dense(self) -> np.ndarray:
        """显式组装 :math:`M`（只用于小规模对拍：验证 :math:`M^{-1}` 的实现）。"""
        D = np.diag(self.diag)
        Dl = D + self.omega * self.lower
        Du = D + self.omega * self.upper
        Dinv = np.diag(1.0 / self.diag)
        return (Dl @ Dinv @ Du) / (self.omega * (2.0 - self.omega))

    @property
    def nnz(self) -> int:
        return int(self.diag.size + 2 * int(np.count_nonzero(self.lower)))


# --------------------------------------------------------------------------- #
# ILU(0)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ILU0Precond:
    r"""零填充不完全 LU：:math:`M=\tilde L\tilde U`，结构 = :math:`A` 的结构（无填充）。

    因子以两个 :class:`~numcore.linalg.sparse.CSR` 存放（:math:`\tilde L` 单位下三角、
    :math:`\tilde U` 上三角），``apply`` 是 CSR 上的前代 + 回代。
    ``dropped`` 记录"因为不在结构里而被丢弃"的更新次数——它正是 ILU(0) 与精确 LU
    的全部差别；一维三对角矩阵上 ``dropped = 0``（ILU(0) 就是精确 LU、CG 一步收敛），
    二维五点差分上 ``dropped > 0``。
    """

    L: CSR
    U: CSR
    dropped: int
    omega_shift: float

    @classmethod
    def from_matrix(cls, A: Any, *, tol: float = 0.0, shift: float = 0.0) -> "ILU0Precond":
        arr = _as_square(A)
        n = arr.shape[0]
        if shift:
            arr = arr + shift * np.eye(n)
        pat = np.abs(arr) > tol
        np.fill_diagonal(pat, True)
        M = arr.copy()
        rows_l: list[int] = []
        cols_l: list[int] = []
        vals_l: list[float] = []
        rows_u: list[int] = []
        cols_u: list[int] = []
        vals_u: list[float] = []
        dropped = 0
        for k in range(n):
            piv = M[k, k]
            if piv == 0.0:
                raise ValueError(f"ILU(0) 在第 {k} 行遇到零主元（请先做置换或用 shift）")
            cols_k = np.nonzero(pat[k, k + 1 :])[0] + k + 1
            rows_k = np.nonzero(pat[k + 1 :, k])[0] + k + 1
            for i in rows_k.tolist():
                M[i, k] = M[i, k] / piv
                upd = cols_k[pat[i, cols_k]]
                if upd.size:
                    M[i, upd] = M[i, upd] - M[i, k] * M[k, upd]
                dropped += int(cols_k.size - upd.size)
        for i in range(n):
            cols = np.nonzero(pat[i])[0]
            for j in cols.tolist():
                if j < i:
                    rows_l.append(i)
                    cols_l.append(j)
                    vals_l.append(float(M[i, j]))
                elif j == i:
                    # 对角元同时属于两个因子：L 的对角是 1，U 的对角是主元
                    rows_l.append(i)
                    cols_l.append(i)
                    vals_l.append(1.0)
                    rows_u.append(i)
                    cols_u.append(i)
                    vals_u.append(float(M[i, i]))
                else:
                    rows_u.append(i)
                    cols_u.append(j)
                    vals_u.append(float(M[i, j]))
        L = to_csr_list(rows_l, cols_l, vals_l, n)
        U = to_csr_list(rows_u, cols_u, vals_u, n)
        return cls(L=L, U=U, dropped=int(dropped), omega_shift=float(shift))

    def apply(self, r: Any) -> np.ndarray:
        rhs = np.asarray(r, dtype=float).reshape(-1)
        n = rhs.size
        y = np.zeros(n, dtype=float)
        for i in range(n):
            cols, vals = self.L.row(i)  # 含对角（值 1）
            s = 0.0
            for c, v in zip(cols.tolist(), vals.tolist()):
                if c < i:
                    s += v * y[c]
                else:
                    piv = v
            y[i] = (rhs[i] - s) / piv
        x = np.zeros(n, dtype=float)
        for i in range(n - 1, -1, -1):
            cols, vals = self.U.row(i)
            s = 0.0
            piv = 1.0
            for c, v in zip(cols.tolist(), vals.tolist()):
                if c > i:
                    s += v * x[c]
                elif c == i:
                    piv = v
            if piv == 0.0:
                raise ValueError(f"ILU(0) 的 U 在第 {i} 行有零对角元：预条件子奇异")
            x[i] = (y[i] - s) / piv
        return x

    def matvec_M(self, x: Any) -> np.ndarray:
        """显式用 :math:`M=\\tilde L\\tilde U` 乘向量（两个 CSR 乘法，无回代）。

        预条件 Lanczos 的自伴内积是 :math:`\\langle x,y\\rangle_M=x^{\\mathsf T}My`
        （**不是** :math:`M^{-1}`-内积），所以这里要能算 :math:`My` 本身。
        """
        return self.L.matvec(self.U.matvec(np.asarray(x, dtype=float).reshape(-1)))

    def to_dense(self) -> np.ndarray:
        """显式组装 :math:`M=\\tilde L\\tilde U`（小规模对拍用）。"""
        return self.L.to_dense() @ self.U.to_dense()

    @property
    def nnz(self) -> int:
        return int(self.L.nnz + self.U.nnz)


def build_precond(name: str, A: Any, *, omega: float = 1.0, shift: float = 0.0) -> Any:
    """按名字构造预条件子：``"none"`` / ``"jacobi"`` / ``"ssor"`` / ``"ilu0"``。"""
    key = str(name).lower()
    if key == "none":
        return None
    if key == "jacobi":
        return JacobiPrecond.from_matrix(A)
    if key == "ssor":
        return SSORPrecond.from_matrix(A, omega=omega)
    if key == "ilu0":
        return ILU0Precond.from_matrix(A, shift=shift)
    raise ValueError(f"未知预条件子 {name!r}（可用：none/jacobi/ssor/ilu0）")


# --------------------------------------------------------------------------- #
# 谱分布与条件数
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PrecondSpectrum:
    """:math:`M^{-1}A` 的 Ritz 谱（手写 Lanczos + Sturm 二分）与条件数估计。"""

    ritz: np.ndarray
    diag: np.ndarray
    offdiag: np.ndarray
    V: np.ndarray
    orthogonality: float
    n_steps: int
    kappa_estimate: float

    @property
    def lambda_min(self) -> float:
        return float(self.ritz[0]) if self.ritz.size else float("nan")

    @property
    def lambda_max(self) -> float:
        return float(self.ritz[-1]) if self.ritz.size else float("nan")


def precond_spectrum(
    A: Any,
    *,
    precond: Any = None,
    m: int | None = None,
    v0: Any = None,
    seed: int = 20240105,
    n_bisect: int = 60,
) -> PrecondSpectrum:
    r"""实测 :math:`M^{-1}A` 的 Ritz 谱（``precond=None`` 时就是 :math:`A` 自己的谱）。

    :math:`A,M` 对称正定时 :math:`M^{-1}A` 关于 :math:`M^{-1}`-内积自伴，于是 Lanczos
    三项递推成立、三对角矩阵的特征值全为实数。``m`` 缺省取矩阵阶数（**精确**谱）；
    取小值时只有两端的 Ritz 值可信——内部值是出了名的收敛慢，别拿它当谱用。
    """
    n = int(A.shape[0]) if hasattr(A, "shape") else int(np.asarray(A).shape[0])
    steps = n if m is None else max(1, min(int(m), n))
    start = rng(seed).standard_normal(n) if v0 is None else np.asarray(v0, dtype=float).reshape(-1)
    res = lanczos(A, start, steps, precond=precond)
    ritz = tridiag_eigenvalues(res.diag, res.offdiag, n_bisect=n_bisect)
    lo = float(ritz[0]) if ritz.size else float("nan")
    hi = float(ritz[-1]) if ritz.size else float("nan")
    kappa = hi / lo if (ritz.size and lo > 0.0) else float("inf")
    return PrecondSpectrum(
        ritz=ritz,
        diag=res.diag,
        offdiag=res.offdiag,
        V=res.V,
        orthogonality=res.orthogonality,
        n_steps=res.n_steps,
        kappa_estimate=float(kappa),
    )


def kappa_estimate(A: Any, *, m: int | None = None, precond: Any = None) -> float:
    """数值条件数估计 :math:`\\lambda_{\\max}/\\lambda_{\\min}`（谱实测，不调用 ``numpy.linalg``）。"""
    return precond_spectrum(A, precond=precond, m=m).kappa_estimate


def kappa_from_factor(factor: float) -> float:
    r"""由**实测** CG 收敛因子反推有效条件数：:math:`\hat\kappa=\bigl(\frac{1+\hat\rho}{1-\hat\rho}\bigr)^2`。

    这是 CG 收敛界 :math:`\rho=(\sqrt\kappa-1)/(\sqrt\kappa+1)` 的逆。
    它给的是"与迭代数一致的那个等效 κ"：谱聚簇时 :math:`\hat\kappa` 会**远小于**真实 κ
    ——这正是"κ 下降但迭代数不按 :math:`\sqrt\kappa` 走"的根源（见
    ``chapters/1.5/pitfalls.md``）。
    """
    r = float(factor)
    if not (0.0 <= r < 1.0):
        raise ValueError(f"收敛因子应在 [0,1) 内，得到 {factor}")
    return float(((1.0 + r) / (1.0 - r)) ** 2)
