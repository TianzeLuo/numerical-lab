"""稀疏矩阵：CSR 容器、结构置换、以及**符号消元**（填充预测）。

为什么"稀疏 + 直接法"是一个矛盾
--------------------------------
LU 分解会**凭空造出非零元**：消元第 k 步把第 k 行的邻居变成两两相连的团（clique），
这个几何图像叫"消元即弦化"（elimination = chordal completion）：

    G(A) 的消元图：每一步 v 被消去时，把它的所有邻居连成一个团，然后删掉 v。

新连出来的边就是**填充**（fill-in）。填充与**数值无关，只与结构和消元顺序有关**，
所以可以在浮点运算之前用纯整数/布尔运算精确预测：

.. math::
    \\mathrm{nnz}(L+U) = 2n + 2\\sum_{k} |\\mathrm{later}(k)|,

其中 :math:`\\mathrm{later}(k)` 是"在 k 被消去时与 k 相邻、且排在 k 之后消去的节点"。
这个公式对**结构对称且不选主元**的消元是精确的（:data:`SymbolicLU` 就是这么算的），
并且可以用 :func:`exact_lu_nnz` 的**有理数精确消元**逐位验证——见 ``tests/1.2``。

本模块提供
----------
* :class:`CSR` —— 自研 CSR 容器（不依赖 ``scipy.sparse``）；含对称置换与矩阵向量乘。
* :func:`band_storage` / :func:`bandwidth` / :func:`profile` —— 带状存储与"带宽/剖面"两个
  经典排序目标。
* :func:`symbolic_lu` —— 符号消元，返回结构、填充边数与"消元过程中的最大团"。
* :func:`exact_lu_nnz` —— 用 :class:`fractions.Fraction` 做精确消元，返回**数值非零**个数；
  用来证明符号预测不是自说自话。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "CSR",
    "band_storage",
    "bandwidth",
    "profile",
    "structure_of",
    "spy_points",
    "SymbolicLU",
    "symbolic_lu",
    "exact_lu_nnz",
    "fill_edges",
    "permutation_matrix",
]


# --------------------------------------------------------------------------- #
# CSR 容器
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CSR:
    """压缩稀疏行（Compressed Sparse Row）容器。

    只存结构上"非零"的元素；``tol`` 用于把幅值小于阈值的元素当作零（结构判定，
    与浮点消元里的精确零不是一回事，见 :func:`structure_of`）。
    """

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    shape: tuple[int, int]

    # ---- 构造 ----
    @classmethod
    def from_dense(cls, A: Any, *, tol: float = 0.0) -> "CSR":
        arr = np.asarray(A, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"需要二维数组，得到形状 {arr.shape}")
        mask = np.abs(arr) > tol
        indptr = np.zeros(arr.shape[0] + 1, dtype=np.int64)
        rows, cols = np.nonzero(mask)
        indptr[1:] = np.cumsum(mask.sum(axis=1))
        return cls(
            indptr=indptr,
            indices=cols.astype(np.int64),
            data=arr[rows, cols].astype(float),
            shape=(int(arr.shape[0]), int(arr.shape[1])),
        )

    # ---- 基本属性 ----
    @property
    def nnz(self) -> int:
        return int(self.indices.size)

    @property
    def n_rows(self) -> int:
        return int(self.shape[0])

    @property
    def n_cols(self) -> int:
        return int(self.shape[1])

    @property
    def density(self) -> float:
        total = self.shape[0] * self.shape[1]
        return self.nnz / total if total else 0.0

    def to_dense(self) -> np.ndarray:
        out = np.zeros(self.shape, dtype=float)
        for i in range(self.shape[0]):
            lo, hi = int(self.indptr[i]), int(self.indptr[i + 1])
            out[i, self.indices[lo:hi]] = self.data[lo:hi]
        return out

    def row(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = int(self.indptr[i]), int(self.indptr[i + 1])
        return self.indices[lo:hi], self.data[lo:hi]

    def matvec(self, x: Any) -> np.ndarray:
        xv = np.asarray(x, dtype=float).reshape(-1)
        if xv.size != self.shape[1]:
            raise ValueError(f"x 的长度应为 {self.shape[1]}，得到 {xv.size}")
        out = np.zeros(self.shape[0], dtype=float)
        for i in range(self.shape[0]):
            cols, vals = self.row(i)
            if cols.size:
                out[i] = float(vals @ xv[cols])
        return out

    # ---- 结构变换 ----
    def permute_symmetric(self, perm: Sequence[int]) -> "CSR":
        """对称置换 :math:`P A P^{\\mathsf T}`（``perm[i]`` 是新矩阵第 i 行/列的原下标）。"""
        p = np.asarray(perm, dtype=int)
        n = self.shape[0]
        if p.size != n or sorted(p.tolist()) != list(range(n)):
            raise ValueError("perm 必须是 0..n-1 的排列")
        dense = self.to_dense()
        return CSR.from_dense(dense[np.ix_(p, p)])

    def transpose(self) -> "CSR":
        return CSR.from_dense(self.to_dense().T)

    def spy_points(self) -> tuple[np.ndarray, np.ndarray]:
        """非零元的 ``(行, 列)`` 坐标（0 基），供 spy plot 散点图层使用。"""
        rows = np.repeat(np.arange(self.shape[0], dtype=float), np.diff(self.indptr))
        return rows, self.indices.astype(float)


def permutation_matrix(perm: Sequence[int]) -> np.ndarray:
    """置换向量 → 置换矩阵（``perm[i]`` = 新第 i 行来自原第 ``perm[i]`` 行）。"""
    p = np.asarray(perm, dtype=int)
    n = p.size
    M = np.zeros((n, n), dtype=float)
    M[np.arange(n), p] = 1.0
    return M


# --------------------------------------------------------------------------- #
# 结构、带宽、剖面
# --------------------------------------------------------------------------- #
def structure_of(A: Any, *, tol: float = 0.0, add_diagonal: bool = False) -> np.ndarray:
    """结构布尔阵：``|a_ij| > tol``。``add_diagonal=True`` 时把对角补上。

    符号消元必须补对角：即使 :math:`a_{ii} = 0`，消元也会在那里产生主元位置
    （这正是"结构对称"假设的常见细节）。
    """
    arr = np.asarray(A, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"需要方阵，得到形状 {arr.shape}")
    S = np.abs(arr) > tol
    if add_diagonal:
        np.fill_diagonal(S, True)
    return S


def spy_points(A: Any, *, tol: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """稠密矩阵的非零坐标（行, 列），行下标 0 在**顶部**（spy plot 惯例）。"""
    S = np.abs(np.asarray(A, dtype=float)) > tol
    rows, cols = np.nonzero(S)
    return rows.astype(float), cols.astype(float)


def bandwidth(A: Any, perm: Sequence[int] | None = None, *, tol: float = 0.0) -> tuple[int, int]:
    """带宽 ``(下带宽, 上带宽)``：``max|i-j|``（可选先做对称置换 ``perm``）。"""
    arr = np.asarray(A, dtype=float)
    if perm is not None:
        p = np.asarray(perm, dtype=int)
        arr = arr[np.ix_(p, p)]
    S = np.abs(arr) > tol
    n = arr.shape[0]
    lower = 0
    upper = 0
    rows, cols = np.nonzero(S)
    for i, j in zip(rows.tolist(), cols.tolist()):
        d = i - j
        if d > lower:
            lower = d
        if -d > upper:
            upper = -d
    return min(lower, n - 1) if n else 0, min(upper, n - 1) if n else 0


def profile(A: Any, perm: Sequence[int] | None = None, *, tol: float = 0.0) -> int:
    """剖面（profile / skyline 度量）：:math:`\\sum_i (i - \\min\\{j : a_{ij} \\ne 0\\})`。

    对对称矩阵它就是"带状存储要存多少项"的一半；最小化剖面是 RCM 的目标。
    """
    arr = np.asarray(A, dtype=float)
    if perm is not None:
        p = np.asarray(perm, dtype=int)
        arr = arr[np.ix_(p, p)]
    S = np.abs(arr) > tol
    n = arr.shape[0]
    total = 0
    for i in range(n):
        cols = np.nonzero(S[i])[0]
        if cols.size:
            total += i - int(cols[0])
    return int(total)


def band_storage(A: Any, lower: int, upper: int) -> np.ndarray:
    """LAPACK 风格带状存储：``ab[upper + i - j, j] = a_ij``（只在 :math:`|i-j|` 带内）。

    返回形状 ``(lower + upper + 1, n)`` 的数组；带外元素为 0。
    带状存储的意义：LU 不产生带外填充，运算量从 :math:`O(n^3)` 降到 :math:`O(n b^2)`。
    """
    arr = np.asarray(A, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"需要方阵，得到形状 {arr.shape}")
    n = arr.shape[0]
    ab = np.zeros((lower + upper + 1, n), dtype=float)
    for j in range(n):
        i0 = max(0, j - upper)
        i1 = min(n, j + lower + 1)
        for i in range(i0, i1):
            ab[upper + i - j, j] = arr[i, j]
    return ab


# --------------------------------------------------------------------------- #
# 符号消元
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SymbolicLU:
    """符号 LU 的结果：结构（L、U 的非零位置）与统计量。"""

    nnz_lu: int
    nnz_l: int
    nnz_u: int
    fill_edges: int
    max_clique: int
    order: tuple[int, ...]
    l_rows: np.ndarray
    l_cols: np.ndarray
    u_rows: np.ndarray
    u_cols: np.ndarray
    fill_rows: np.ndarray
    fill_cols: np.ndarray

    @property
    def treewidth_upper_bound(self) -> int:
        """消元过程中的最大团大小减一（树宽的上界）。"""
        return max(0, self.max_clique - 1)

    def to_json(self) -> dict[str, Any]:
        return {
            "nnz_lu": self.nnz_lu,
            "nnz_l": self.nnz_l,
            "nnz_u": self.nnz_u,
            "fill_edges": self.fill_edges,
            "max_clique": self.max_clique,
            "treewidth_upper_bound": self.treewidth_upper_bound,
            "order": list(self.order),
        }


def symbolic_lu(
    A: Any,
    perm: Sequence[int] | None = None,
    *,
    tol: float = 0.0,
    keep_pattern: bool = False,
) -> SymbolicLU:
    """符号 LU：纯布尔运算预测 :math:`L+U` 的非零结构与填充边数。

    ``perm`` 是**消元顺序**（``perm[k]`` = 第 k 个被消去的节点）。缺省自然序。
    ``A`` 的结构若不对称，这里先按"并集"对称化——因为数值消元里的主元搜索会把
    行/列的结构混在一起（这也是"结构对称"假设在实战中必须显式声明的理由）。

    注意符号预测与数值现实的**方向**：符号消元给出的是非零结构的**超集**
    （它不知道"数值上恰好抵消"这种事），因此
    ``symbolic.nnz_lu >= exact_lu_nnz(A)`` 恒成立；两者相等当且仅当消元过程中
    没有发生精确抵消。反过来，**选主元**只可能让结构变大：
    ``symbolic_lu(A).nnz_lu <= symbolic_lu(A, perm).nnz_lu`` 并不成立（顺序才是主因），
    但"部分主元后的实际 nnz ≥ 无主元符号预测"是可靠的下界。
    """
    arr = np.asarray(A, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"需要方阵，得到形状 {arr.shape}")
    n = arr.shape[0]
    order = np.arange(n) if perm is None else np.asarray(perm, dtype=int)
    if order.size != n or sorted(order.tolist()) != list(range(n)):
        raise ValueError("perm 必须是 0..n-1 的排列")
    pos = np.empty(n, dtype=int)
    pos[order] = np.arange(n, dtype=int)  # 节点 → 消元顺序

    S = structure_of(arr, tol=tol, add_diagonal=True)
    S = S | S.T  # 结构对称化
    adj = S.copy()
    np.fill_diagonal(adj, False)  # 只留非对角邻接

    alive = np.ones(n, dtype=bool)
    l_rows: list[int] = []
    l_cols: list[int] = []
    u_rows: list[int] = []
    u_cols: list[int] = []
    fill_rows: list[int] = []
    fill_cols: list[int] = []
    fill_edges = 0
    max_clique = 1
    total_off = 0

    for step in range(n):
        k = int(order[step])
        nbrs = np.flatnonzero(adj[k] & alive)
        nbrs = nbrs[nbrs != k]
        later = np.array([v for v in nbrs.tolist() if pos[v] > step], dtype=int)
        # U 的第 k 行、L 的第 k 列都含有 later 中的节点（无主元时结构对称）
        for v in later.tolist():
            u_rows.append(k)
            u_cols.append(int(v))
            l_rows.append(int(v))
            l_cols.append(k)
        total_off += later.size
        max_clique = max(max_clique, later.size + 1)
        # 弦化：把邻居连成团
        for a_i in range(nbrs.size):
            for b_i in range(a_i + 1, nbrs.size):
                u, v = int(nbrs[a_i]), int(nbrs[b_i])
                if not adj[u, v]:
                    adj[u, v] = True
                    adj[v, u] = True
                    fill_edges += 1
                    if keep_pattern:
                        fill_rows.append(u)
                        fill_cols.append(v)
        alive[k] = False
        adj[k, :] = False
        adj[:, k] = False

    nnz_l = n + total_off
    nnz_u = n + total_off
    diag = np.arange(n, dtype=float)
    l_rows_a = np.concatenate([np.asarray(l_rows, dtype=float), diag]) if n else np.zeros(0)
    l_cols_a = np.concatenate([np.asarray(l_cols, dtype=float), diag]) if n else np.zeros(0)
    u_rows_a = np.concatenate([np.asarray(u_rows, dtype=float), diag]) if n else np.zeros(0)
    u_cols_a = np.concatenate([np.asarray(u_cols, dtype=float), diag]) if n else np.zeros(0)
    return SymbolicLU(
        nnz_lu=nnz_l + nnz_u,
        nnz_l=nnz_l,
        nnz_u=nnz_u,
        fill_edges=int(fill_edges),
        max_clique=int(max_clique),
        order=tuple(int(v) for v in order.tolist()),
        l_rows=l_rows_a,
        l_cols=l_cols_a,
        u_rows=u_rows_a,
        u_cols=u_cols_a,
        fill_rows=np.asarray(fill_rows, dtype=float),
        fill_cols=np.asarray(fill_cols, dtype=float),
    )


def fill_edges(A: Any, perm: Sequence[int] | None = None, *, tol: float = 0.0) -> int:
    """填充边数（= 符号消元新增的图边数）。"""
    return symbolic_lu(A, perm, tol=tol).fill_edges


def exact_lu_nnz(A: Any, perm: Sequence[int] | None = None) -> tuple[int, int, int]:
    """用**有理数精确消元**数一遍 :math:`L`、:math:`U` 的非零个数。

    这是符号预测的独立证人：浮点消元里"看起来非零"可能是舍入噪声，
    有理数运算则给出真正的零。返回 ``(nnz_L, nnz_U, nnz_L+nnz_U)``（都含对角）。

    只用于小矩阵（:math:`O(n^3)` 次 ``Fraction`` 运算），**不选主元**：
    主元搜索会改变非零结构，符号预测的公式本就假设了固定顺序。
    """
    arr = np.asarray(A, dtype=float)
    n = arr.shape[0]
    order = np.arange(n) if perm is None else np.asarray(perm, dtype=int)
    M: list[list[Fraction]] = [[Fraction(float(arr[i, j])) for j in range(n)] for i in range(n)]
    rank = np.zeros(n, dtype=int)  # 节点 → 消元步
    for step, k in enumerate(order.tolist()):
        rank[k] = step
    for step in range(n):
        k = int(order[step])
        piv = M[k][k]
        if piv == 0:
            continue
        for i in range(n):
            if rank[i] <= step or M[i][k] == 0:
                continue
            factor = M[i][k] / piv
            M[i][k] = factor  # L 的乘子写回原位置
            for j in range(n):
                if rank[j] > step:
                    M[i][j] -= factor * M[k][j]
    nnz_l = 0
    nnz_u = 0
    for i in range(n):
        for j in range(n):
            if M[i][j] == 0:
                continue
            if i > j:
                nnz_l += 1
            elif i == j:
                nnz_l += 1
                nnz_u += 1
            else:
                nnz_u += 1
    return nnz_l, nnz_u, nnz_l + nnz_u


def elimination_graph_snapshot(A: Any, steps: int, *, tol: float = 0.0) -> np.ndarray:
    """前 ``steps`` 步弦化之后的邻接布尔阵（可视化"填充边是怎么长出来的"）。"""
    S = structure_of(A, tol=tol, add_diagonal=True)
    adj = (S | S.T).copy()
    np.fill_diagonal(adj, False)
    alive = np.ones(adj.shape[0], dtype=bool)
    for k in range(min(steps, adj.shape[0])):
        nbrs = np.flatnonzero(adj[k] & alive)
        nbrs = nbrs[nbrs != k]
        for a_i in range(nbrs.size):
            for b_i in range(a_i + 1, nbrs.size):
                u, v = int(nbrs[a_i]), int(nbrs[b_i])
                adj[u, v] = True
                adj[v, u] = True
        alive[k] = False
        adj[k, :] = False
        adj[:, k] = False
    return adj


def to_csr_list(rows: Iterable[int], cols: Iterable[int], vals: Iterable[float], n: int) -> CSR:
    """由坐标三元组直接构造 CSR（给图/网格类稀疏矩阵用，避免先摊成稠密）。"""
    r = np.asarray(list(rows), dtype=np.int64)
    c = np.asarray(list(cols), dtype=np.int64)
    v = np.asarray(list(vals), dtype=float)
    order = np.argsort(r * n + c, kind="stable")
    r, c, v = r[order], c[order], v[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, r + 1, 1)
    indptr = np.cumsum(indptr)
    return CSR(indptr=indptr, indices=c, data=v, shape=(n, n))
