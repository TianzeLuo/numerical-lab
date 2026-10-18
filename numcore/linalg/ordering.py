"""稀疏排序：RCM 与最小度（消元顺序决定填充，填充决定一切）。

一句话动机
----------
对稀疏直接法来说，"换个未知量编号"就能把计算量从 :math:`O(n^3)` 压到近线性——这就是
排序的全部意义。本章只讲两个最经典、能手写清楚的：

* **RCM（Reverse Cuthill–McKee）**：BFS 分层 + 层内按度升序 + 整体逆序。
  它最小化的不是填充，而是**带宽/剖面**（bandwidth / profile）；
  对带状矩阵（有限差分、有限元的一维编号）几乎总是最优或接近最优。
* **最小度（minimum degree，AMD 的贪心简化版）**：每一步消去"当前图里度最小"的节点。
  它直接以**填充边数**为目标，是通用稀疏 LU 的默认策略（真实 AMD 用商图 + 近似度，
  这里用稠密布尔图实现"能看清每一步"的版本）。

两个都建立在同一个图论图像上：**消元 = 弦化**（见 :mod:`numcore.linalg.sparse`）。
排序的目标函数因此可以完全脱离数值：:func:`numcore.linalg.sparse.symbolic_lu` 给出的
``nnz_lu`` 就是"这个顺序会填出多少个非零元"的精确预测。

复杂度说明
----------
本模块的稠密布尔图实现是 :math:`O(n^3)`（每步要扫整张图），只适合 :math:`n \\lesssim 500`
的教学规模；生产实现（AMD/METIS）用的是商图与嵌套剖分。**这里是教材，不是求解器**，
但每个函数都给出了与精确消元对得上的预测（``tests/1.2``）。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .sparse import bandwidth, profile, structure_of, symbolic_lu

__all__ = [
    "adjacency",
    "degree",
    "cuthill_mckee",
    "reverse_cuthill_mckee",
    "rcm",
    "pseudo_peripheral_node",
    "minimum_degree",
    "natural_order",
    "ordering_report",
    "OrderingReport",
]


def adjacency(A: Any, *, tol: float = 0.0) -> np.ndarray:
    """对称邻接布尔阵（不含对角）：``adj[i,j] = |a_ij|>tol 或 |a_ji|>tol``。"""
    S = structure_of(A, tol=tol, add_diagonal=False)
    adj = S | S.T
    np.fill_diagonal(adj, False)
    return adj


def degree(adj: np.ndarray, alive: np.ndarray | None = None) -> np.ndarray:
    """当前图中每个节点的度（``alive`` 之外的节点不计）。"""
    if alive is None:
        return adj.sum(axis=1).astype(int)
    m = adj & alive.reshape(1, -1) & alive.reshape(-1, 1)
    return m.sum(axis=1).astype(int)


def natural_order(A: Any) -> np.ndarray:
    """自然序（用作所有排序的对照组）。"""
    n = np.asarray(A).shape[0] if np.asarray(A).ndim == 2 else 0
    return np.arange(n, dtype=int)


# --------------------------------------------------------------------------- #
# RCM
# --------------------------------------------------------------------------- #
def _bfs_order(adj: np.ndarray, start: int, *, degree_aware: bool = True) -> tuple[list[int], np.ndarray]:
    """从 ``start`` 做 BFS，返回访问顺序与层号。

    ``degree_aware=True`` 时层内按**当前度**升序（Cuthill–McKee 的原始提议）：
    度数小的节点先编号，能让同一层的"触角"更短。
    """
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    level = np.full(n, -1, dtype=int)
    order: list[int] = []
    seen[start] = True
    level[start] = 0
    queue: deque[int] = deque([start])
    while queue:
        v = queue.popleft()
        order.append(v)
        nbrs = np.flatnonzero(adj[v] & ~seen)
        if degree_aware and nbrs.size > 1:
            nbrs = nbrs[np.argsort(degree(adj)[nbrs], kind="stable")]
        for u in nbrs.tolist():
            seen[u] = True
            level[u] = level[v] + 1
            queue.append(u)
    if not seen.all():  # 不连通图：按编号顺序接着做
        for v in range(n):
            if not seen[v]:
                sub, _ = _bfs_order(adj, v, degree_aware=degree_aware)
                for u in sub:
                    if not seen[u]:
                        seen[u] = True
                        level[u] = 0
                        order.append(u)
    return order, level


def pseudo_peripheral_node(adj: np.ndarray, start: int = 0) -> tuple[int, int]:
    """伪外围节点与它的偏心率（两次 BFS 的启发式）。

    真正的"图外围"（最小偏心率）要算全源最短路；两次 BFS 是 Cuthill–McKee 时代的
    标准折中：从任意点出发取最远点 :math:`u`，再从 :math:`u` 出发取最远点 :math:`v`，
    返回 :math:`(v, \\mathrm{ecc}(u))`。带状矩阵上它通常给出带状结构的端点。
    """
    order, level = _bfs_order(adj, start, degree_aware=False)
    far = max(level.tolist())
    candidates = [i for i, lv in enumerate(level.tolist()) if lv == far]
    u = min(candidates) if candidates else start
    order2, level2 = _bfs_order(adj, u, degree_aware=False)
    far2 = max(level2.tolist())
    cand2 = [i for i, lv in enumerate(level2.tolist()) if lv == far2]
    v = min(cand2) if cand2 else u
    return v, int(far)


def cuthill_mckee(A: Any, *, start: int | None = None, tol: float = 0.0) -> np.ndarray:
    """Cuthill–McKee 排序（未逆序）。返回 ``order``：``order[i]`` = 新编号 i 对应的原节点。"""
    adj = adjacency(A, tol=tol)
    n = adj.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    if start is None:
        start, _ = pseudo_peripheral_node(adj)
    order, _ = _bfs_order(adj, int(start))
    return np.asarray(order, dtype=int)


def reverse_cuthill_mckee(A: Any, *, start: int | None = None, tol: float = 0.0) -> np.ndarray:
    """Reverse Cuthill–McKee：CM 顺序整体逆序。

    逆序不改变带宽，但**剖面减半**（:math:`\\sum_i (i-\\min j)` 里每行的"左端"变短），
    这就是 RCM 里那个 R 的全部意义。
    """
    return cuthill_mckee(A, start=start, tol=tol)[::-1].copy()


#: 简短别名
rcm = reverse_cuthill_mckee


# --------------------------------------------------------------------------- #
# 最小度
# --------------------------------------------------------------------------- #
def minimum_degree(A: Any, *, tol: float = 0.0, tie_break: str = "index") -> np.ndarray:
    """贪心最小度排序（简化 AMD）：每步消去当前度最小的节点，返回消元顺序。

    ``tie_break`` 取 ``"index"``（最小下标优先，可复现）或 ``"fill"``
    （并列时选"消去后新增填充边最少"的那个——真实 AMD 的近似度思想）。
    """
    adj = adjacency(A, tol=tol)
    n = adj.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    if tie_break not in ("index", "fill"):
        raise ValueError(f"未知 tie_break {tie_break!r}")
    work = adj.copy()
    alive = np.ones(n, dtype=bool)
    order: list[int] = []
    for _ in range(n):
        deg = degree(work, alive)
        deg_valid = np.where(alive, deg, np.iinfo(np.int64).max)
        best = int(np.min(deg_valid))
        cands = np.flatnonzero(alive & (deg == best))
        if cands.size == 1 or tie_break == "index":
            k = int(cands[0])
        else:
            best_fill = None
            k = int(cands[0])
            for c in cands.tolist():
                nbrs = np.flatnonzero(work[c] & alive)
                nbrs = nbrs[nbrs != c]
                new_edges = 0
                for a_i in range(nbrs.size):
                    for b_i in range(a_i + 1, nbrs.size):
                        if not work[int(nbrs[a_i]), int(nbrs[b_i])]:
                            new_edges += 1
                if best_fill is None or new_edges < best_fill:
                    best_fill = new_edges
                    k = int(c)
        nbrs = np.flatnonzero(work[k] & alive)
        nbrs = nbrs[nbrs != k]
        for a_i in range(nbrs.size):
            for b_i in range(a_i + 1, nbrs.size):
                u, v = int(nbrs[a_i]), int(nbrs[b_i])
                work[u, v] = True
                work[v, u] = True
        alive[k] = False
        work[k, :] = False
        work[:, k] = False
        order.append(k)
    return np.asarray(order, dtype=int)


# --------------------------------------------------------------------------- #
# 对比报告
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OrderingReport:
    """一种排序的全部可观测量（表格/柱状图/断言都用它）。"""

    name: str
    order: tuple[int, ...]
    nnz_lu: int
    fill_edges: int
    bandwidth: int
    profile: int
    max_clique: int
    nnz_original: int

    @property
    def fill_ratio(self) -> float:
        """填充倍数：``nnz(L+U) / nnz(A)``（1.0 表示完全没填）。"""
        return self.nnz_lu / self.nnz_original if self.nnz_original else float("inf")

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "nnz_lu": self.nnz_lu,
            "fill_edges": self.fill_edges,
            "bandwidth": self.bandwidth,
            "profile": self.profile,
            "max_clique": self.max_clique,
            "fill_ratio": self.fill_ratio,
            "order": list(self.order),
        }


def ordering_report(A: Any, order: Sequence[int], name: str, *, tol: float = 0.0) -> OrderingReport:
    """对给定的消元顺序算一份"填充/带宽/剖面"报告。"""
    sy = symbolic_lu(A, order, tol=tol)
    lo, hi = bandwidth(A, order, tol=tol)
    arr = np.asarray(A, dtype=float)
    return OrderingReport(
        name=name,
        order=tuple(int(v) for v in np.asarray(order, dtype=int).tolist()),
        nnz_lu=sy.nnz_lu,
        fill_edges=sy.fill_edges,
        bandwidth=int(max(lo, hi)),
        profile=profile(A, order, tol=tol),
        max_clique=sy.max_clique,
        nnz_original=int(np.count_nonzero(np.abs(arr) > tol)),
    )
