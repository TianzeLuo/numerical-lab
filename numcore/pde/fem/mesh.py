r"""网格与有限元空间：一维区间网格、二维三角/四边形网格、加密、自由度布局。

数学定义
--------
一维网格是一组**顶点** :math:`0=x_0<x_1<\dots<x_N=1` 与**单元** :math:`K_i=[x_i,x_{i+1}]`；
单元直径 :math:`h_i=x_{i+1}-x_i`。有限元空间 :math:`V_h` 由"单元 + 形函数"共同决定，
所以本模块把两件事分开：

* :class:`Mesh1D` / :class:`Mesh2D` 只说**几何**（顶点、连接、直径、面积）；
* :class:`Space1D` 说**自由度布局**：哪些节点是自由度、每个单元用到哪几个局部自由度、
  边界自由度是谁。同一个网格配不同的 family 就是不同的空间：

  ==============  ==============================  ==========================================
  family          单元形函数                      本章用途
  ==============  ==============================  ==========================================
  ``p1``          拉格朗日线性（2 节点）          二阶问题的基准空间，L2 阶 2 / 能量阶 1
  ``p2``          拉格朗日二次（顶点 + 中点）     二阶问题的高阶空间，L2 阶 3 / 能量阶 2
  ``lobatto``     单元素任意次拉格朗日             p-加密（Chebyshev--Lobatto 节点）
  ``hermite``     Hermite 三次（每个顶点 ``w``、``w'`` 两个自由度）  四阶梁方程
  ==============  ==============================  ==========================================

**为什么要单列 hermite**：梁方程 :math:`(EIw'')''=q` 的弱形式里出现 :math:`w''`，
试探空间必须是 :math:`H^2` 的子集，也就是 :math:`C^1` 的。用 :math:`P_1` 单元去解它，
单元刚度矩阵是**恒零矩阵**（线性函数的二阶导为零），整个 :math:`K` 直接奇异——
这不是"精度差一点"，是格式本身不存在。这一条在本模块里是可实测的
（:func:`~numcore.pde.fem.assemble.assemble` 会返回零矩阵，:mod:`~numcore.pde.fem.solve`
会如实报"Cholesky 在第 1 个主元失败"）。

二维网格用三角形（:math:`P_1`）或**双线性四边形**。四边形的等参映射不是仿射的，
所以本章能真的看到"在求积点上逐点算 Jacobi 行列式"与"取单元中心的 J 当常数"的差别。
:func:`distorted_quad_mesh` 给出一个边界保持不动的光滑畸变场，用来研究网格质量对误差的影响。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "Mesh1D",
    "Space1D",
    "Mesh2D",
    "uniform_mesh_1d",
    "mesh_from_nodes",
    "lagrange_space",
    "lobatto_space",
    "hermite_space",
    "uniform_tri_mesh",
    "uniform_quad_mesh",
    "distorted_quad_mesh",
    "chebyshev_lobatto_nodes",
    "neighbor_pairs",
]


# --------------------------------------------------------------------------- #
# 一维网格
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Mesh1D:
    r"""一维区间网格：顶点坐标 ``x``（升序）与单元连接 ``elems``（``(n_elem, 2)`` 顶点号）。

    只描述几何。想得到有限元空间请用 :func:`lagrange_space` /
    :func:`hermite_space` / :func:`lobatto_space`。
    """

    x: np.ndarray
    elems: np.ndarray

    def __post_init__(self) -> None:
        xs = np.asarray(self.x, dtype=float)
        el = np.asarray(self.elems, dtype=np.int64)
        if xs.ndim != 1 or xs.size < 2:
            raise ValueError(f"顶点数至少 2，得到 {xs.shape}")
        if el.ndim != 2 or el.shape[1] != 2:
            raise ValueError(f"一维单元连接必须是 (n_elem, 2)，得到 {el.shape}")
        if np.any(np.diff(xs) <= 0.0):
            raise ValueError("顶点必须严格递增（网格不能有零长或翻转的单元）")
        # 一维协调网格的构造不变量：单元从左到右首尾相接，覆盖全部顶点。
        # 这条检查抓到过一个真 bug（加密时一个单元只登记了一条连接，节点数涨了、单元数没涨）。
        if el.size:
            chain = np.all(el[1:, 0] == el[:-1, 1]) and int(el[0, 0]) == 0 and int(el[-1, 1]) == xs.size - 1
            if not chain:
                raise ValueError(
                    "单元连接必须首尾相接并覆盖全部顶点（一维协调网格的不变量）："
                    f"elems[0]={el[0].tolist()}, elems[-1]={el[-1].tolist()}, n_node={xs.size}"
                )
        object.__setattr__(self, "x", xs)
        object.__setattr__(self, "elems", el)

    # ---------------------------------------------------------------- 基本量
    @property
    def n_node(self) -> int:
        return int(self.x.size)

    @property
    def n_elem(self) -> int:
        return int(self.elems.shape[0])

    @property
    def h(self) -> np.ndarray:
        r"""每个单元的长度 :math:`h_i=x_{i+1}-x_i`（有限元的"单元直径"）。"""
        return self.x[self.elems[:, 1]] - self.x[self.elems[:, 0]]

    @property
    def centers(self) -> np.ndarray:
        """单元中点坐标（残差型指示子与热力图都用它当横坐标）。"""
        return 0.5 * (self.x[self.elems[:, 0]] + self.x[self.elems[:, 1]])

    @property
    def min_h(self) -> float:
        return float(np.min(self.h))

    @property
    def max_h(self) -> float:
        return float(np.max(self.h))

    def element_of(self, x: Any) -> int:
        """包含点 ``x`` 的单元编号（右端点归最后一个单元；点载荷的定位靠它）。"""
        xv = float(x)
        right = np.asarray(self.x[self.elems[:, 1]], dtype=float)
        idx = int(np.searchsorted(right, xv, side="left"))
        return min(max(idx, 0), self.n_elem - 1)

    def boundary_nodes(self) -> tuple[int, int]:
        """左右端点的**顶点号**（不是自由度号，自由度映射由 :class:`Space1D` 负责）。"""
        left = int(np.argmin(self.x))
        right = int(np.argmax(self.x))
        return left, right

    def neighbors(self, i: int) -> tuple[int, ...]:
        """与单元 ``i`` 共用一个顶点的单元（一维界面只有左右两个邻居）。"""
        out = []
        for j in range(self.n_elem):
            if j != i and set(self.elems[j].tolist()) & set(self.elems[i].tolist()):
                out.append(j)
        return tuple(out)

    # ---------------------------------------------------------------- 加密
    def refine_uniform(self) -> "Mesh1D":
        """一致加密：每个单元从中点一分为二（单元数翻倍，:math:`h` 减半）。"""
        new_x: list[float] = []
        new_el: list[tuple[int, int]] = []
        for k in range(self.n_elem):
            a, b = int(self.elems[k, 0]), int(self.elems[k, 1])
            m = len(new_x)
            new_x.extend([float(self.x[a]), 0.5 * (float(self.x[a]) + float(self.x[b]))])
            new_el.append((m, m + 1))
            new_el.append((m + 1, m + 2))
        new_x.append(float(self.x[-1]))
        return Mesh1D(np.asarray(new_x), np.asarray(new_el, dtype=np.int64))

    def refine_marked(self, marked: Any) -> "Mesh1D":
        r"""只加密 ``marked`` 为真的单元，并做**一层闭包**保证网格协调。

        一维里把一个单元二分会在邻居内部造出一个悬挂节点，邻居不跟着分就不是一个合法的
        :math:`H^1` 网格。所以真正的加密集合是 :math:`M\cup N(M)`（标记集及其直接邻居）。
        这个闭包不是"顺手补一下"：它让"自适应加密"的单元数比标记数多，
        章节里"同样自由度下的误差比"用的是**加密后的真实自由度**，不是标记数。
        """
        mask = np.asarray(marked, dtype=bool).reshape(-1)
        if mask.size != self.n_elem:
            raise ValueError(f"marked 长度必须是 n_elem={self.n_elem}，得到 {mask.size}")
        closed = mask.copy()
        for i in np.flatnonzero(mask):
            for j in self.neighbors(int(i)):
                closed[j] = True
        new_x: list[float] = []
        new_el: list[tuple[int, int]] = []
        for k in range(self.n_elem):
            a, b = int(self.elems[k, 0]), int(self.elems[k, 1])
            m = len(new_x)
            new_x.append(float(self.x[a]))
            if closed[k]:
                new_x.append(0.5 * (float(self.x[a]) + float(self.x[b])))
                new_el.append((m, m + 1))
                new_el.append((m + 1, m + 2))
            else:
                new_el.append((m, m + 1))
        new_x.append(float(self.x[-1]))
        return Mesh1D(np.asarray(new_x), np.asarray(new_el, dtype=np.int64))

    def refine_at(self, x0: float, levels: int = 1) -> "Mesh1D":
        """在 ``x0`` 附近做 ``levels`` 次局部二分（"把节点搬到奇点附近"的确定性版本）。"""
        mesh = self
        for _ in range(max(0, int(levels))):
            i = mesh.element_of(x0)
            mask = np.zeros(mesh.n_elem, dtype=bool)
            mask[i] = True
            mesh = mesh.refine_marked(mask)
        return mesh


def uniform_mesh_1d(n_elem: int, x0: float = 0.0, x1: float = 1.0) -> Mesh1D:
    """一致网格：``n_elem`` 个等长单元。"""
    ne = int(n_elem)
    if ne < 1:
        raise ValueError(f"单元数至少 1，得到 {n_elem!r}")
    if not float(x1) > float(x0):
        raise ValueError(f"需要 x1 > x0，得到 ({x0!r}, {x1!r})")
    xs = np.linspace(float(x0), float(x1), ne + 1)
    el = np.column_stack([np.arange(ne), np.arange(1, ne + 1)])
    return Mesh1D(xs, el.astype(np.int64))


def mesh_from_nodes(nodes: Any) -> Mesh1D:
    """由**任意**升序节点表造网格（每个相邻对是一个单元）：局部加密的显式构造入口。"""
    xs = np.asarray(nodes, dtype=float).reshape(-1)
    el = np.column_stack([np.arange(xs.size - 1), np.arange(1, xs.size)])
    return Mesh1D(xs, el.astype(np.int64))


def neighbor_pairs(mesh: Mesh1D) -> np.ndarray:
    """相邻单元对 ``(i, i+1)``：面跳跃项就在这些内部界面上取。"""
    return np.column_stack([np.arange(mesh.n_elem - 1), np.arange(1, mesh.n_elem)]).astype(np.int64)


def chebyshev_lobatto_nodes(n: int) -> np.ndarray:
    r"""区间 :math:`[-1,1]` 上的 :math:`n+1` 个 Chebyshev--Lobatto 点 :math:`\cos(k\pi/n)`。

    p-加密用它当节点集：它是**显式**的、端点必在，且 Lebesgue 常数只有
    :math:`O(\log n)`。真正的 Gauss--Lobatto 点（:math:`P_n'` 的零点）会让质量阵更对角，
    但节点集不是本章要教的东西——所以本模块明说用的是 Chebyshev--Lobatto，
    并且把"节点选择影响条件数"这件事放在 :func:`~numcore.pde.fem.solve.system_condition` 里实测。
    """
    nn = int(n)
    if nn < 1:
        raise ValueError(f"阶数至少 1，得到 {n!r}")
    k = np.arange(nn + 1)
    return np.cos(np.pi * k / nn)[::-1].copy()


# --------------------------------------------------------------------------- #
# 一维有限元空间
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Space1D:
    r"""一维有限元空间：一族形函数 + 一张网格。

    ``nodes`` 是**基函数节点**的坐标（:math:`P_2` 含中点、Hermite 每个顶点有两个自由度）、
    ``elems`` 是单元到**局部基函数编号**的连接（不是顶点编号）。两套编号分开是有意的：
    :math:`P_2` 与 Hermite 的"顶点"与"自由度"不是一一对应，混用是这类代码最常见的索引 bug。
    """

    family: str
    degree: int
    nodes: np.ndarray
    elems: np.ndarray
    mesh: Mesh1D
    dof_node: np.ndarray
    """每个自由度对应的顶点号（Hermite 的一个顶点有两个自由度，会重复出现）。"""

    dim: int | None = None
    """自由度的真实个数。Hermite 空间一个顶点两个自由度，``nodes.size`` 只是顶点数，
    所以这个字段不能由 ``nodes`` 推出来——写成推导式是这类代码最经典的索引 bug。"""

    @property
    def n_dof(self) -> int:
        return int(self.nodes.size if self.dim is None else self.dim)

    @property
    def n_elem(self) -> int:
        return int(self.elems.shape[0])

    @property
    def npe(self) -> int:
        return int(self.elems.shape[1])

    @property
    def is_hermite(self) -> bool:
        return self.family == "hermite"

    @property
    def is_cubic_hermite(self) -> bool:
        return self.family == "hermite"

    @property
    def element_h(self) -> np.ndarray:
        return self.mesh.h

    def boundary_dofs(self, kind: str = "dirichlet") -> np.ndarray:
        r"""边界自由度。

        * ``dirichlet``：所有落在 :math:`x\in\{0,1\}` 上的自由度（Hermite 是 ``w`` 与 ``w'``）；
        * ``clamped``：Hermite 梁两端固支 :math:`w=w'=0`；
        * ``simply_supported``：Hermite 梁两端简支 :math:`w=0`（:math:`w''=0` 是自然边界条件）；
        * ``neumann``：空集（自然边界条件不进矩阵）。
        """
        x0 = float(np.min(self.mesh.x))
        x1 = float(np.max(self.mesh.x))
        at_end = np.isclose(self.nodes, x0) | np.isclose(self.nodes, x1)
        if kind == "neumann":
            return np.flatnonzero(np.zeros(self.n_dof, dtype=bool))
        if kind == "dirichlet":
            return np.flatnonzero(at_end)
        if self.is_hermite:
            vert_at_end = at_end[self.dof_node]
            if kind == "clamped":
                return np.flatnonzero(vert_at_end)
            if kind == "simply_supported":
                # 只钉 w（偶数号自由度），w' 放开
                return np.flatnonzero(vert_at_end & (np.arange(self.n_dof) % 2 == 0))
        raise ValueError(f"未知的边界类型 {kind!r}（family={self.family!r}）")

    def sorted_dofs(self) -> np.ndarray:
        """按坐标排序的自由度号（画图用；:math:`P_2` 的自由度表不是按坐标排的）。"""
        return np.argsort(self.nodes, kind="stable")


def lagrange_space(mesh: Mesh1D, degree: int = 1) -> Space1D:
    r"""拉格朗日 :math:`P_1` / :math:`P_2` 空间。

    :math:`P_2` 的节点是"顶点 + 单元中点"，自由度按 ``[顶点..., 中点...]`` 编号；
    单元连接是 :math:`(v_i, v_{i+1}, m_i)`（局部基函数 2 是中点那个）。
    """
    if degree not in (1, 2):
        raise ValueError(f"只支持 P1/P2，得到 degree={degree!r}")
    nn = mesh.n_node
    if degree == 1:
        nodes = mesh.x.copy()
        elems = mesh.elems.copy()
    else:
        mids = 0.5 * (mesh.x[mesh.elems[:, 0]] + mesh.x[mesh.elems[:, 1]])
        nodes = np.concatenate([mesh.x, mids])
        mid_ids = nn + np.arange(mesh.n_elem)
        elems = np.column_stack([mesh.elems[:, 0], mesh.elems[:, 1], mid_ids]).astype(np.int64)
    dof_node = np.concatenate([np.arange(nn), np.arange(nn)]) if degree == 2 else np.arange(nn)
    return Space1D(
        family=f"p{degree}",
        degree=int(degree),
        nodes=nodes,
        elems=elems.astype(np.int64),
        mesh=mesh,
        dof_node=dof_node.astype(np.int64),
    )


def lobatto_space(degree: int, x0: float = 0.0, x1: float = 1.0) -> Space1D:
    r"""单元素 Chebyshev--Lobatto 空间（p-加密用）：一个单元、``degree`` 次拉格朗日基。

    自由度就是 :math:`p+1` 个节点上的值；Dirichlet 条件钉住两端，剩下 :math:`p-1` 个未知量。
    """
    p = int(degree)
    if p < 1:
        raise ValueError(f"阶数至少 1，得到 {degree!r}")
    xi = chebyshev_lobatto_nodes(p)
    nodes = 0.5 * (float(x1) - float(x0)) * (xi + 1.0) + float(x0)
    mesh = Mesh1D(np.asarray([float(x0), float(x1)]), np.asarray([[0, 1]], dtype=np.int64))
    elems = np.arange(p + 1).reshape(1, -1).astype(np.int64)
    return Space1D(
        family=f"lobatto{p}",
        degree=p,
        nodes=np.asarray(nodes, dtype=float),
        elems=elems,
        mesh=mesh,
        dof_node=np.zeros(p + 1, dtype=np.int64),
    )


def hermite_space(mesh: Mesh1D) -> Space1D:
    r"""Hermite 三次空间（:math:`C^1`）：每个顶点两个自由度 :math:`(w_i, w_i')`。

    单元 :math:`[x_i,x_{i+1}]` 上的四个基函数是三次 Hermite 插值基，自由度编号
    :math:`(2i,2i+1,2j,2j+1)`。这个空间是 :math:`H^2` 的子集，所以
    :math:`\int w''v''` 有定义——这正是梁方程需要的。
    """
    nn = mesh.n_node
    dof_node = np.repeat(np.arange(nn), 2)
    elems = np.column_stack(
        [2 * mesh.elems[:, 0], 2 * mesh.elems[:, 0] + 1, 2 * mesh.elems[:, 1], 2 * mesh.elems[:, 1] + 1]
    ).astype(np.int64)
    return Space1D(
        family="hermite",
        degree=3,
        nodes=mesh.x.copy(),
        elems=elems,
        mesh=mesh,
        dof_node=dof_node.astype(np.int64),
        dim=2 * nn,
    )


# --------------------------------------------------------------------------- #
# 二维网格
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Mesh2D:
    r"""二维网格：``nodes`` 是 :math:`(n_{node},2)` 顶点，``elems`` 是三角形或四边形连接。

    只描述几何；等参映射与 Jacobi 行列式在 :mod:`~numcore.pde.fem.shape` 里。
    """

    nodes: np.ndarray
    elems: np.ndarray
    kind: str = "tri"

    def __post_init__(self) -> None:
        pts = np.asarray(self.nodes, dtype=float)
        el = np.asarray(self.elems, dtype=np.int64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"二维顶点必须是 (n,2)，得到 {pts.shape}")
        want = 3 if self.kind == "tri" else 4
        if el.ndim != 2 or el.shape[1] != want:
            raise ValueError(f"{self.kind} 单元连接必须是 (n_elem,{want})，得到 {el.shape}")
        object.__setattr__(self, "nodes", pts)
        object.__setattr__(self, "elems", el)

    @property
    def n_node(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_elem(self) -> int:
        return int(self.elems.shape[0])

    @property
    def centroids(self) -> np.ndarray:
        return np.mean(self.nodes[self.elems], axis=1)

    @property
    def areas(self) -> np.ndarray:
        r"""单元面积（三角形用叉积、四边形拆两个三角形；恒正）。"""
        p = self.nodes[self.elems]
        if self.kind == "tri":
            a = p[:, 1] - p[:, 0]
            b = p[:, 2] - p[:, 0]
            return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        a = p[:, 2] - p[:, 0]
        b = p[:, 3] - p[:, 1]
        return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])

    @property
    def diameters(self) -> np.ndarray:
        """单元直径 = 最长边（先验误差估计里的 :math:`h_K`）。"""
        p = self.nodes[self.elems]
        k = p.shape[1]
        best = np.zeros(self.n_elem, dtype=float)
        for i in range(k):
            for j in range(i + 1, k):
                d = p[:, i] - p[:, j]
                best = np.maximum(best, np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2))
        return best

    @property
    def min_area(self) -> float:
        return float(np.min(self.areas))

    def boundary_mask(self) -> np.ndarray:
        """顶点是否落在边界上（用包围盒判定；本模块的网格都是正方形区域）。"""
        x0, y0 = np.min(self.nodes, axis=0)
        x1, y1 = np.max(self.nodes, axis=0)
        tol = 1e-12 * max(1.0, float(x1 - x0), float(y1 - y0))
        return (
            np.isclose(self.nodes[:, 0], x0, atol=tol)
            | np.isclose(self.nodes[:, 0], x1, atol=tol)
            | np.isclose(self.nodes[:, 1], y0, atol=tol)
            | np.isclose(self.nodes[:, 1], y1, atol=tol)
        )

    def refine_uniform(self) -> "Mesh2D":
        r"""一致加密：三角形一分为四、四边形一分为四（都靠边中点表去重）。

        新顶点按"先老点、后边中点、再单元中心"的顺序编号；父子关系只影响编号，不影响空间。
        """
        pts = [tuple(map(float, p)) for p in self.nodes]
        index = {p: i for i, p in enumerate(pts)}
        mids: dict[tuple[int, int], int] = {}

        def mid(i: int, j: int) -> int:
            key = (min(i, j), max(i, j))
            if key not in mids:
                q = 0.5 * (self.nodes[key[0]] + self.nodes[key[1]])
                pt = (float(q[0]), float(q[1]))
                if pt not in index:
                    index[pt] = len(pts)
                    pts.append(pt)
                mids[key] = index[pt]
            return mids[key]

        new_el: list[tuple[int, ...]] = []
        for e in self.elems:
            ids = [int(v) for v in e]
            if self.kind == "tri":
                a, b, c = ids
                ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
                new_el.extend([(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])
            else:
                a, b, c, d = ids
                ab, bc, cd, da = mid(a, b), mid(b, c), mid(c, d), mid(d, a)
                q = 0.25 * (self.nodes[a] + self.nodes[b] + self.nodes[c] + self.nodes[d])
                qt = (float(q[0]), float(q[1]))
                if qt not in index:
                    index[qt] = len(pts)
                    pts.append(qt)
                o = index[qt]
                new_el.extend([(a, ab, o, da), (ab, b, bc, o), (o, bc, c, cd), (da, o, cd, d)])
        return Mesh2D(np.asarray(pts, dtype=float), np.asarray(new_el, dtype=np.int64), self.kind)

    def to_triangles(self) -> "Mesh2D":
        """四边形网格拆成三角形（每个四边形沿较短对角线一分为二）：用于网格族对照。"""
        if self.kind == "tri":
            return self
        el = []
        for a, b, c, d in self.elems:
            pa, pb, pc, pd = self.nodes[[a, b, c, d]]
            d1 = float(np.sum((pa - pc) ** 2))
            d2 = float(np.sum((pb - pd) ** 2))
            el.append((int(a), int(b), int(c)) if d1 <= d2 else (int(a), int(b), int(d)))
            el.append((int(a), int(c), int(d)) if d1 <= d2 else (int(b), int(c), int(d)))
        return Mesh2D(self.nodes.copy(), np.asarray(el, dtype=np.int64), "tri")


def uniform_tri_mesh(nx: int, ny: int, x0: float = 0.0, x1: float = 1.0,
                     y0: float = 0.0, y1: float = 1.0) -> Mesh2D:
    r"""矩形上的一致三角形网格（每格两条直角边 + 一条反对角线）。"""
    nx, ny = int(nx), int(ny)
    if nx < 1 or ny < 1:
        raise ValueError(f"网格数至少 1，得到 nx={nx}, ny={ny}")
    xs = np.linspace(float(x0), float(x1), nx + 1)
    ys = np.linspace(float(y0), float(y1), ny + 1)
    pts = np.array([[a, b] for b in ys for a in xs], dtype=float)
    el = []
    for j in range(ny):
        for i in range(nx):
            p00 = j * (nx + 1) + i
            p10 = p00 + 1
            p01 = p00 + (nx + 1)
            p11 = p01 + 1
            el.append((p00, p10, p01))
            el.append((p10, p11, p01))
    return Mesh2D(pts, np.asarray(el, dtype=np.int64), "tri")


def _smooth_distortion(nodes: np.ndarray, amount: float) -> np.ndarray:
    r"""边界保持不动的光滑畸变场 :math:`(x,y)\mapsto (x,y)+d\sin(2\pi x)\sin(2\pi y)(1,1)`。

    场在 :math:`x\in\{0,1\}` 或 :math:`y\in\{0,1\}` 上恒为零，所以**边界顶点一个都没动**——
    这一点很重要：它保证 Dirichlet 边界仍然落在真实的边界上，
    误差的变化只能归因于单元形状，而不是"边界条件被挪了位置"。
    """
    x = nodes[:, 0]
    y = nodes[:, 1]
    d = float(amount) * np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
    return nodes + np.column_stack([d, -d])


def uniform_quad_mesh(nx: int, ny: int, distortion: float = 0.0) -> Mesh2D:
    r"""矩形上的一致**双线性四边形**网格，``distortion>0`` 时施加边界保持的光滑畸变。

    一致网格上每个四边形是平行四边形，等参映射退化成仿射映射（J 在单元内是常数）；
    畸变之后 J 在单元内逐点变化——"等参"这两个字这时候才真的开始起作用。
    """
    nx, ny = int(nx), int(ny)
    if nx < 1 or ny < 1:
        raise ValueError(f"网格数至少 1，得到 nx={nx}, ny={ny}")
    xs = np.linspace(0.0, 1.0, nx + 1)
    ys = np.linspace(0.0, 1.0, ny + 1)
    pts = np.array([[a, b] for b in ys for a in xs], dtype=float)
    pts = _smooth_distortion(pts, distortion)
    el = []
    for j in range(ny):
        for i in range(nx):
            p00 = j * (nx + 1) + i
            p10 = p00 + 1
            p01 = p00 + (nx + 1)
            p11 = p01 + 1
            el.append((p00, p10, p11, p01))
    return Mesh2D(pts, np.asarray(el, dtype=np.int64), "quad")


def distorted_quad_mesh(nx: int, ny: int, amount: float = 0.08) -> Mesh2D:
    """畸变四边形网格（:func:`uniform_quad_mesh` 的具名入口）。"""
    return uniform_quad_mesh(nx, ny, distortion=amount)
