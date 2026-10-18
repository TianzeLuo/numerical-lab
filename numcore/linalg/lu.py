"""LU 分解：无主元 / 部分主元 / 完全主元，以及增长因子与后向误差。

数学定义
--------
对 :math:`A \\in \\mathbb{R}^{n \\times n}`，LU 分解写作

.. math::
    P A = L U, \\qquad L \\text{ 单位下三角},\\ U \\text{ 上三角}.

消元的第 :math:`k` 步（:math:`0 \\le k < n-1`）先选主元，再做行变换

.. math::
    m_{ik} = \\frac{a_{ik}^{(k)}}{a_{kk}^{(k)}}, \\qquad
    a_{ij}^{(k+1)} = a_{ij}^{(k)} - m_{ik} a_{kj}^{(k)} \\quad (i,j > k),

乘子 :math:`m_{ik}` 直接写回下三角（LAPACK 的 in-place 布局），于是
:math:`L = I + \\mathrm{tril}(M, -1)`、:math:`U = \\mathrm{triu}(M)`。

三种主元策略的区别只在"第 :math:`k` 步选哪一行/列当主元"：

=================  ============================================  ====================
策略               选主元规则                                    增长因子上界
=================  ============================================  ====================
无主元             恒取 :math:`a_{kk}`（可能为 0 → 直接崩）       无界（构造性反例见下）
部分主元           第 k 列第 k..n-1 行中 :math:`|\\cdot|` 最大者     :math:`\\rho \\le 2^{n-1}`
完全主元           剩余子矩阵中 :math:`|\\cdot|` 最大者              :math:`\\rho \\le` 更小，但选主元贵
=================  ============================================  ====================

增长因子与后向误差
------------------
.. math::
    \\rho = \\frac{\\max_{ij} |u_{ij}|}{\\max_{ij} |a_{ij}|}, \\qquad
    \\eta = \\frac{\\|Ax - b\\|_\\infty}{\\|A\\|_\\infty \\|x\\|_\\infty + \\|b\\|_\\infty}.

Wilkinson 的经典结论：**部分主元是后向稳定的**，因为算出来的 :math:`\\hat L \\hat U`
满足 :math:`\\hat L \\hat U = A + \\delta A` 且 :math:`\\|\\delta A\\|_\\infty \\le c\\, n\\, \\rho\\, u \\|A\\|_\\infty`
（:math:`u` 是单位舍入）。也就是说"解不准"通常不是因为消元不稳，而是因为
:math:`\\kappa(A)` 大——把这两件事混为一谈是本章最常见的误读，见
``chapters/1.2/pitfalls.md``。

:func:`wilkinson_matrix` 给出的矩阵在**部分主元**下恰好取到上界 :math:`\\rho = 2^{n-1}`，
它同时是"部分主元也不能保证解准"的反例素材（:math:`\\rho` 大 ⇒ 后向误差界松）。

实现约束
--------
只用 ``numpy`` 的容器与逐元素原语：``np.outer`` + 切片做消元，``np.abs``/``max``
做范数，**不调用** ``np.linalg`` 的任何东西（POLICY P2/P3）。随机矩阵请走
:func:`numcore.utils.rng`（P4）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Literal

import numpy as np

Pivot = Literal["none", "partial", "complete"]

__all__ = [
    "LUResult",
    "lu_decompose",
    "lu_partial",
    "lu_no_pivot",
    "lu_complete",
    "lu_solve",
    "solve_lu",
    "det_from_lu",
    "permute_rows",
    "permute_cols",
    "residual",
    "residual_norm",
    "backward_error",
    "growth_factor",
    "leading_block_residuals",
    "ROUNDOFF_HINT",
    "elimination_flops",
    "wilkinson_matrix",
    "EXAMPLE_MATRICES",
    "example_matrix",
    "unit_ball_image",
    "unit_ball_radius",
    "exact_solve",
    "condition_estimate_1",
    "leading_principal_minors",
    "singular_values_2x2",
    "condition_2x2",
    "dominant_matrix",
    "ball_matrix",
]


# --------------------------------------------------------------------------- #
# 结果容器
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LUResult:
    """一次 LU 分解的全部可观测物。

    ``history[k]`` 是第 k 步消元**之后**的工作矩阵（乘子在 ``tril`` 里），
    ``history[0]`` 是原始矩阵的副本；``history[-1]`` 就是最终 :math:`L\\backslash U` 的
    紧凑布局。给"消元过程热力图"用，也让断言能逐步核对不变量。
    """

    L: np.ndarray
    U: np.ndarray
    perm: np.ndarray
    #: ``col_perm[j]``：当前第 j 列来自原始第几列（只有完全主元会动它；否则是 arange）
    col_perm: np.ndarray
    pivots: tuple[int, ...]
    row_swaps: tuple[tuple[int, int], ...]
    col_swaps: tuple[tuple[int, int], ...]
    rho: float
    n_pivots: int
    singular: bool
    ops: int
    history: tuple[np.ndarray, ...] = field(default=())
    #: 与 ``history`` 同步的**参照矩阵**：只跟着做同样的行/列交换、**不做消元**。
    #: 于是 ``ref_history[j][:m,:m]`` 正是"第 j 步之后那个前导块的原始样子"，
    #: 可以拿来算逐步残差 :math:`\|A_{\rm lead} - L_{\rm lead}U_{\rm lead}\|`。
    #: 为什么不能用最终的 ``P @ A @ Q``：第 j 步之后的**后续**交换还会改动位置 ≥ j 的行列，
    #: 因此最终的 P 并不是当时的 P——拿它算前导块残差会得到 O(1) 的假残差（真踩过）。
    ref_history: tuple[np.ndarray, ...] = field(default=())

    @property
    def n(self) -> int:
        return int(self.U.shape[0])

    def P(self) -> np.ndarray:
        """行置换矩阵 :math:`P`（``P @ A = L @ U``）。"""
        n = self.n
        p = np.zeros((n, n), dtype=float)
        p[np.arange(n), self.perm] = 1.0
        return p

    def Q(self) -> np.ndarray:
        r"""列置换矩阵 :math:`Q`（``P @ A @ Q = L @ U``）。

        完全主元会交换列；**行置换矩阵单独用是不够的**。踩过的坑：1.2 章的消元热力图
        用 ``P @ A`` 取前导块去重建 :math:`LU` 的残差，完全主元那一档于是得到 O(1) 的
        残差（正确的量级是 :math:`O(u)`）——因为列被换过而重建时没换回来。
        约定：``col_perm[j]`` 是"当前第 j 列来自原始第几列"。要让右乘真的换列，
        :math:`(AQ)[i,j]` 必须取 :math:`A[i,\text{col\_perm}[j]]`，即
        ``Q[col_perm[j], j] = 1``——**与 :meth:`P` 的写法互为转置**（行置换作用在左边、
        列置换作用在右边，同一个"位置 → 原索引"数组在两侧的矩阵写法正好相反）。
        第一版照抄 :meth:`P` 的写法，结果 Wilikinson 矩阵上
        :math:`\|PAQ-LU\|_\infty = 2`，一眼就露馅。无列交换时它是单位阵。
        """
        n = self.n
        q = np.zeros((n, n), dtype=float)
        q[self.col_perm, np.arange(n)] = 1.0
        return q

    def to_json(self) -> dict[str, Any]:
        """给测试/报告用的摘要（不含大数组）。"""
        return {
            "n": self.n,
            "rho": self.rho,
            "n_pivots": self.n_pivots,
            "singular": self.singular,
            "ops": self.ops,
            "pivots": list(self.pivots),
            "row_swaps": [list(s) for s in self.row_swaps],
            "col_swaps": [list(s) for s in self.col_swaps],
            "col_perm": [int(v) for v in self.col_perm],
        }


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _as_square(A: Any) -> np.ndarray:
    arr = np.asarray(A, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"需要方阵，得到形状 {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("空矩阵没有 LU 分解")
    return np.array(arr, dtype=float, copy=True)


def _max_abs(A: np.ndarray) -> float:
    if A.size == 0:
        return 0.0
    return float(np.max(np.abs(A)))


def permute_rows(A: Any, perm: Any) -> np.ndarray:
    """按行置换向量 ``perm`` 重排（``perm[i]`` 是新矩阵第 i 行原来的行号）。"""
    arr = np.asarray(A, dtype=float)
    return arr[np.asarray(perm, dtype=int), :]


def permute_cols(A: Any, perm: Any) -> np.ndarray:
    """按列置换向量重排（对称置换用 :func:`permute_rows` 与它各来一次）。"""
    arr = np.asarray(A, dtype=float)
    return arr[:, np.asarray(perm, dtype=int)]


def growth_factor(A: Any, U: Any) -> float:
    """增长因子 :math:`\\rho = \\max|u_{ij}| / \\max|a_{ij}|`。

    ``max|A| = 0`` 时按 0 处理（全零矩阵没有"增长"可言），不返回 NaN——
    NaN 会让"比较大小"的断言静默失效。
    """
    denom = _max_abs(np.asarray(A, dtype=float))
    num = _max_abs(np.asarray(U, dtype=float))
    if denom == 0.0:
        return 0.0
    return num / denom


def residual(A: Any, x: Any, b: Any) -> np.ndarray:
    """残差向量 :math:`r = b - Ax`（自研矩阵向量乘，不用 ``np.linalg``）。"""
    arr = np.asarray(A, dtype=float)
    xv = np.asarray(x, dtype=float).reshape(-1)
    bv = np.asarray(b, dtype=float).reshape(-1)
    return bv - arr @ xv


def residual_norm(A: Any, x: Any, b: Any) -> float:
    """:math:`\\|b - Ax\\|_\\infty`。"""
    return _max_abs(residual(A, x, b))


def backward_error(A: Any, x: Any, b: Any) -> float:
    """范数型后向误差 :math:`\\eta = \\|Ax-b\\|_\\infty / (\\|A\\|_\\infty\\|x\\|_\\infty + \\|b\\|_\\infty)`。

    这是"算出来的解究竟是不是某个邻近问题的精确解"的度量，量级应当在
    :math:`u \\approx 2.2\\times10^{-16}` 附近（与 :math:`\\kappa(A)` **无关**）。
    """
    arr = np.asarray(A, dtype=float)
    xv = np.asarray(x, dtype=float).reshape(-1)
    bv = np.asarray(b, dtype=float).reshape(-1)
    r = _max_abs(residual(arr, xv, bv))
    scale = _max_abs(arr.sum(axis=1)) if arr.size else 0.0  # ‖A‖_∞ = 最大行和
    denom = scale * _max_abs(xv) + _max_abs(bv)
    if denom == 0.0:
        return 0.0
    return r / denom


def elimination_flops(n: int) -> int:
    """稠密 LU（无主元 / 部分主元，主元搜索不计）的精确标量运算次数。

    第 :math:`k` 步要处理 :math:`n-k-1` 行、每行 :math:`1` 次除法 + :math:`n-k-1` 次
    乘加，于是

    .. math::
        N(n) = \\sum_{k=0}^{n-2} (n-k-1)\\,(2(n-k-1) + 1)
             = \\frac{n(n-1)(4n+1)}{6} \\sim \\frac{2}{3}n^3.
    """
    if n < 1:
        raise ValueError(f"n 必须为正，得到 {n}")
    j = np.arange(1, n, dtype=float)  # n-k-1 = 1..n-1
    return int(np.sum(j * (2.0 * j + 1.0)))


# --------------------------------------------------------------------------- #
# 三种主元策略
# --------------------------------------------------------------------------- #
def lu_decompose(
    A: Any,
    *,
    pivot: Pivot = "partial",
    history: bool = False,
    tol: float = 0.0,
) -> LUResult:
    """LU 分解的统一入口；``pivot`` 取 ``"none"`` / ``"partial"`` / ``"complete"``。

    ``history=True`` 时记录每一步之后的工作矩阵（热力图用，内存 :math:`O(n^3)`，只在小矩阵上开）。
    ``tol`` 是判定"主元为零"的阈值（相对尺度）；``tol=0`` 表示只有精确 0 才算奇异——
    这正是本章要展示的陷阱：**小主元不等于零主元**，但无主元消元遇到它会放大误差。
    """
    if pivot not in ("none", "partial", "complete"):
        raise ValueError(f"未知主元策略 {pivot!r}")
    M = _as_square(A)
    n = M.shape[0]
    perm = np.arange(n)
    col_perm = np.arange(n)
    pivots: list[int] = []
    row_swaps: list[tuple[int, int]] = []
    col_swaps: list[tuple[int, int]] = []
    ops = 0
    singular = False
    scale = _max_abs(M)
    thresh = tol * (scale if scale > 0.0 else 1.0)
    snapshots: list[np.ndarray] = []
    ref_snapshots: list[np.ndarray] = []
    ref = M.copy()  # 只做交换、不消元：前导块残差要用它
    if history:
        snapshots.append(M.copy())
        ref_snapshots.append(ref.copy())

    for k in range(n - 1):
        # ---- 选主元 ----
        if pivot == "partial":
            cand = np.abs(M[k:, k])
            piv = k + int(np.argmax(cand))
            # 并列时 argmax 取**最小行号**：可复现，也让 Wilkinson 矩阵恰好取到 2^{n-1}
            if piv != k:
                M[[k, piv], :] = M[[piv, k], :]
                ref[[k, piv], :] = ref[[piv, k], :]
                perm[[k, piv]] = perm[[piv, k]]
                row_swaps.append((k, piv))
            pivots.append(piv)
        elif pivot == "complete":
            block = np.abs(M[k:, k:])
            flat = int(np.argmax(block))
            dr, dc = divmod(flat, n - k)
            pr, pc = k + dr, k + dc
            if pr != k:
                M[[k, pr], :] = M[[pr, k], :]
                ref[[k, pr], :] = ref[[pr, k], :]
                perm[[k, pr]] = perm[[pr, k]]
                row_swaps.append((k, pr))
            if pc != k:
                M[:, [k, pc]] = M[:, [pc, k]]
                ref[:, [k, pc]] = ref[:, [pc, k]]
                col_perm[[k, pc]] = col_perm[[pc, k]]
                col_swaps.append((k, pc))
            pivots.append(pr)
        else:
            pivots.append(k)

        pivot_value = float(M[k, k])
        if abs(pivot_value) <= thresh and pivot_value == 0.0:
            # 精确零主元：这一步跳过（L 的该列乘子留 0），继续做后面的列
            singular = True
            if history:
                snapshots.append(M.copy())
                ref_snapshots.append(ref.copy())
            continue
        if abs(pivot_value) <= thresh and tol > 0.0:
            singular = True

        # ---- 消元（乘子写回下三角）----
        rows = n - k - 1
        m = M[k + 1 :, k] / pivot_value
        M[k + 1 :, k] = m
        M[k + 1 :, k + 1 :] -= np.outer(m, M[k, k + 1 :])
        ops += rows * (2 * rows + 1)
        if history:
            snapshots.append(M.copy())
            ref_snapshots.append(ref.copy())

    L = np.tril(M, -1) + np.eye(n)
    U = np.triu(M)
    rho = growth_factor(_as_square(A), U)
    return LUResult(
        L=L,
        U=U,
        perm=perm,
        pivots=tuple(pivots),
        col_perm=col_perm.copy(),
        row_swaps=tuple(row_swaps),
        col_swaps=tuple(col_swaps),
        rho=rho,
        n_pivots=len(pivots),
        singular=singular,
        ops=ops,
        history=tuple(snapshots),
        ref_history=tuple(ref_snapshots),
    )


#: 单位舍入 :math:`u = \varepsilon/2 = 2^{-53}` 的一半（:math:`2^{-54}`）。
#: 用途只有两个：给"恰好为 0"的量一个 log 轴上的显示地板，以及在断言里当
#: "舍入级"的判据。**它不是"误差上限"**——误差上限要看 :math:`n u \rho`。
ROUNDOFF_HINT = float(np.finfo(float).eps) / 2.0


def leading_block_residuals(res: LUResult, *, relative: bool = False) -> list[float]:
    r"""逐步的**前导块分解残差** :math:`\|A_{\rm lead} - L_{\rm lead}U_{\rm lead}\|_\infty`。

    为什么是"前导块"而不是整个矩阵：消元到第 j 步时，真正算完的只有
    :math:`A_{1:j+1,1:j+1}`；右下角还是"被更新过的原矩阵"，把它算进去只会得到假的大数。

    为什么要 :attr:`LUResult.ref_history` 而不是最终的 :math:`P`：第 j 步之后的后续交换
    还会改动位置 :math:`\ge j` 的行列，**最终的置换不是当时的置换**。用最终置换去取前导块
    会得到 O(1) 的残差（踩过两次：漏列置换、以及这里说的"最终 vs 当步"）。
    量级应当是 :math:`n u \rho`。

    需要 ``history=True`` 与（自动一起记录的）``ref_history``；否则抛 :class:`ValueError`。
    """
    if not res.history or not res.ref_history:
        raise ValueError("需要 history=True 才能算逐步残差（ref_history 与它同步记录）")
    if len(res.history) != len(res.ref_history):
        raise ValueError(
            f"history 与 ref_history 长度不一致：{len(res.history)} vs {len(res.ref_history)}"
        )
    n = res.n
    denom = _max_abs(res.ref_history[0]) if relative else 1.0
    out: list[float] = []
    for j, (snap, ref) in enumerate(zip(res.history, res.ref_history)):
        m = max(1, min(j + 1, n))
        L = np.tril(np.asarray(snap, dtype=float)[:m, :m], -1) + np.eye(m)
        U = np.triu(np.asarray(snap, dtype=float)[:m, :m])
        num = _max_abs(np.asarray(ref, dtype=float)[:m, :m] - L @ U)
        out.append(num / denom if denom > 0.0 else num)
    return out


def lu_no_pivot(A: Any, **kw: Any) -> LUResult:
    """无主元 LU：教材上的"朴素消元"。遇到零主元不报错，但结果不可用（``singular=True``）。"""
    return lu_decompose(A, pivot="none", **kw)


def lu_partial(A: Any, **kw: Any) -> LUResult:
    """部分主元（列主元）LU：``|m_{ik}| <= 1``，工业实现的标准选择。"""
    return lu_decompose(A, pivot="partial", **kw)


def lu_complete(A: Any, **kw: Any) -> LUResult:
    """完全主元 LU：主元取剩余子矩阵最大元，增长因子最小，代价是 :math:`O(n^2)` 次比较。"""
    return lu_decompose(A, pivot="complete", **kw)


# --------------------------------------------------------------------------- #
# 回代与行列式
# --------------------------------------------------------------------------- #
def solve_lu(res: LUResult, b: Any) -> np.ndarray:
    """用已有的 LU 分解解 :math:`Ax = b`：先 :math:`Ly = Pb`（前代），再 :math:`Ux = y`（回代）。

    完全主元时解向量还需要按 ``col_swaps`` 逆置换回去。
    """
    n = res.n
    bv = np.asarray(b, dtype=float).reshape(-1)
    if bv.size != n:
        raise ValueError(f"b 的长度应为 {n}，得到 {bv.size}")
    y = bv[res.perm].astype(float)  # P b
    # 前代：L 单位下三角
    for i in range(n):
        if i:
            y[i] -= float(res.L[i, :i] @ y[:i])
    # 回代：U 上三角
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        acc = y[i] - float(res.U[i, i + 1 :] @ x[i + 1 :])
        diag = float(res.U[i, i])
        if diag == 0.0:
            raise ZeroDivisionError(f"U 的第 {i} 个对角元为 0：矩阵奇异，回代失败")
        x[i] = acc / diag
    # 完全主元的列置换：x_original[col_perm[i]] = x[i]
    if res.col_swaps:
        out = np.empty(n, dtype=float)
        out[res.col_perm] = x
        return out
    return x


#: 兼容别名：旧名字更短，测试与章节代码都在用
lu_solve = solve_lu


def det_from_lu(res: LUResult) -> float:
    """由 LU 求行列式：:math:`\\det A = (-1)^{s} \\prod_i u_{ii}`（``s`` 是行交换次数）。"""
    sign = -1.0 if len(res.row_swaps) % 2 else 1.0
    if res.col_swaps:
        sign *= -1.0 if len(res.col_swaps) % 2 else 1.0
    prod = 1.0
    for i in range(res.n):
        prod *= float(res.U[i, i])
    return sign * prod


# --------------------------------------------------------------------------- #
# 反例矩阵动物园
# --------------------------------------------------------------------------- #
def wilkinson_matrix(n: int) -> np.ndarray:
    """Wilkinson 增长矩阵：**部分主元**下 :math:`\\rho = 2^{n-1}`（上界可达）。

    .. math::
        A_{ii} = 1,\\quad A_{ij} = -1\\ (j < i),\\quad A_{i,n-1} = 1.

    第 1 步第 0 列全是 :math:`\\pm 1`，并列时取最小行号 ⇒ 不换行，乘子全是 :math:`-1`；
    更新后最后一列变成 :math:`2`，子矩阵结构不变 ⇒ 每步翻倍，共 :math:`n-1` 步。
    """
    if n < 1:
        raise ValueError(f"n 必须为正，得到 {n}")
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i):
            A[i, j] = -1.0
        A[i, i] = 1.0
        A[i, n - 1] = 1.0
    return A


#: 反例动物园：每个矩阵对应一句"别相信直觉"的说明。
EXAMPLE_MATRICES: dict[str, str] = {
    "swap": "首元为 0：无主元消元直接除零（[[0,1],[1,0]]）",
    "tiny_pivot": "首元 1e-18：无主元时乘子 1e18，解的误差被放大到荒谬量级",
    "wilkinson": "部分主元的增长因子上界可达者：ρ = 2^{n-1}",
    "spd": "对称正定：Cholesky 无主元也稳定，ρ = 1",
    "hilbert": "Hilbert 矩阵：κ 指数增长，但**后向误差**依然 ~1e-16（别怪消元）",
    "diag_dominant": "严格对角占优：无主元也稳定（这是充分条件，不是必要条件）",
    "banded": "三对角：稀疏结构的极端例子，LU 不产生填充",
}


def example_matrix(name: str, n: int = 8) -> np.ndarray:
    """返回 :data:`EXAMPLE_MATRICES` 里的示例矩阵（``n`` 对部分矩阵无效）。"""
    if name == "swap":
        return np.array([[0.0, 1.0], [1.0, 0.0]])
    if name == "tiny_pivot":
        return np.array([[1e-18, 1.0], [1.0, 1.0]])
    if name == "wilkinson":
        return wilkinson_matrix(n)
    if name == "spd":
        # A = BᵀB + n I：构造性 SPD，不用随机数也能复现
        B = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                B[i, j] = 1.0 / (1.0 + ((i - j) % n))
        return B.T @ B + n * np.eye(n)
    if name == "hilbert":
        i = np.arange(1, n + 1, dtype=float).reshape(-1, 1)
        j = np.arange(1, n + 1, dtype=float).reshape(1, -1)
        return 1.0 / (i + j - 1.0)
    if name == "diag_dominant":
        A = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i != j:
                    A[i, j] = 1.0 / (1.0 + abs(i - j))
        A[np.arange(n), np.arange(n)] = np.sum(np.abs(A), axis=1) + 1.0
        return A
    if name == "banded":
        A = np.zeros((n, n), dtype=float)
        idx = np.arange(n)
        A[idx, idx] = 4.0
        A[idx[:-1], idx[:-1] + 1] = -1.0
        A[idx[:-1] + 1, idx[:-1]] = -1.0
        return A
    raise KeyError(f"未知示例矩阵 {name!r}（可用 {sorted(EXAMPLE_MATRICES)}）")


# --------------------------------------------------------------------------- #
# 可视化辅助：单位球的形变
# --------------------------------------------------------------------------- #
def unit_ball_image(A: Any, n_theta: int = 181) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """单位圆周在 :math:`A` 下的像（"单位球变成什么形状"）。

    返回 ``(theta, x, y)``：``x = A @ (cosθ, sinθ)``。对 2×2 矩阵而言，
    像是一个椭圆；面积 = :math:`|\\det A|`（体积就是行列式）。
    """
    arr = _as_square(A)
    if arr.shape[0] != 2:
        raise ValueError(f"单位球形变目前只画 2×2，得到 {arr.shape}")
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    pts = np.stack([np.cos(theta), np.sin(theta)], axis=0)
    img = arr @ pts
    return theta, img[0], img[1]


def unit_ball_radius(A: Any, n_theta: int = 361) -> tuple[float, float]:
    """像到原点的距离区间 ``(min r, max r)``。

    :math:`\\max r = \\sigma_{\\max}(A)`（最大奇异值），:math:`\\min r = \\sigma_{\\min}(A)`；
    两者之比就是谱条件数 :math:`\\kappa_2(A)`——**这里不调用 SVD**，纯粹靠采样圆周量出来，
    所以它是一条"实测值"，可以和参考层的 SVD 对拍（A 类断言）。
    """
    _, x, y = unit_ball_image(A, n_theta=n_theta)
    r = np.sqrt(x * x + y * y)
    return float(np.min(r)), float(np.max(r))


# --------------------------------------------------------------------------- #
# 精确解与条件数估计（都不调用 numpy.linalg）
# --------------------------------------------------------------------------- #
def exact_solve(A: Any, b: Any) -> np.ndarray:
    """用**有理数**高斯消元求 :math:`Ax = b` 的精确解，再转回浮点。

    这是 A 类断言的"真值"来源：解析的、与被测算法（LU + 浮点舍入）**完全独立**。
    输入按**浮点表示的精确值**进入有理数域（``Fraction(float)``，不做任何近似）：
    这才是"给定这组 double，方程组的真解是什么"的正确语义。

    .. note::
       早先这里用 ``Fraction(x).limit_denominator(10**15)`` 想把 ``1e-18`` "还原成整数比"，
       结果 1e-18 的最佳有理近似（分母 ≤ 1e15）就是 **0**——于是"精确解"把整个小主元项
       抹掉了，反例矩阵的前向误差凭空变成 0。精确解一旦有损，A 类断言就变成了自证。

    只适合小矩阵（:math:`O(n^3)` 次 ``Fraction`` 运算，n ≲ 20）。
    """
    arr = np.asarray(A, dtype=float)
    bv = np.asarray(b, dtype=float).reshape(-1)
    n = arr.shape[0]
    if arr.ndim != 2 or arr.shape[1] != n or bv.size != n:
        raise ValueError(f"形状不匹配：A{arr.shape} b{bv.shape}")
    M = [[Fraction(float(arr[i, j])) for j in range(n)] + [Fraction(float(bv[i]))] for i in range(n)]
    for k in range(n):
        piv = None
        for i in range(k, n):
            if M[i][k] != 0:
                piv = i
                break
        if piv is None:
            raise ZeroDivisionError(f"第 {k} 列全零：矩阵奇异，精确解不存在或不唯一")
        if piv != k:
            M[k], M[piv] = M[piv], M[k]
        pk = M[k][k]
        for i in range(k + 1, n):
            if M[i][k] == 0:
                continue
            f = M[i][k] / pk
            for j in range(k, n + 1):
                M[i][j] -= f * M[k][j]
    x = [Fraction(0)] * n
    for i in range(n - 1, -1, -1):
        acc = M[i][n]
        for j in range(i + 1, n):
            acc -= M[i][j] * x[j]
        x[i] = acc / M[i][i]
    return np.array([float(v) for v in x], dtype=float)


def condition_estimate_1(A: Any) -> float:
    r""":math:`\kappa_1(A) = \|A\|_1 \|A^{-1}\|_1` 的估计（自研 LU 求逆列）。

    ``numpy.linalg.cond`` 在本仓库里是被禁的成品算法（P2），所以这里用自己写的
    LU 解 :math:`A y_j = e_j` 得到 :math:`A^{-1}` 的各列再取 1-范数。
    注意这有个哲学问题：**用被测算法去度量被测算法的难度**。对本章的演示足够，
    但它本身在 :math:`\kappa` 极大时不可靠——这是 ``chapters/1.2/pitfalls.md`` 里
    明确写着的一条（要当真值请用 :func:`exact_solve` 或参考层）。
    """
    arr = np.asarray(A, dtype=float)
    n = arr.shape[0]
    norm_a = float(np.max(np.sum(np.abs(arr), axis=0)))  # ‖A‖_1 = 最大列和
    res = lu_partial(arr)
    inv_cols = []
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        inv_cols.append(solve_lu(res, e))
    inv = np.stack(inv_cols, axis=1)
    norm_inv = float(np.max(np.sum(np.abs(inv), axis=0)))
    return norm_a * norm_inv


def leading_principal_minors(A: Any) -> np.ndarray:
    r"""各阶顺序主子式的值 :math:`\det A_{1:k,1:k}`（由 LU 的对角元累乘，不需重算行列式）。

    数学用处：**无主元 LU 能顺利进行 ⟺ 所有顺序主子式非零**（充要条件）。
    所以这条曲线里一旦出现 0，无主元消元就会在对应那一步除零——这正是"选主元"存在的理由。
    这里不判断 Cholesky 那种"正定"，只关心"是不是零"。
    """
    res = lu_no_pivot(A)
    n = res.n
    out = np.zeros(n, dtype=float)
    prod = 1.0
    for k in range(n):
        prod *= float(res.U[k, k])
        out[k] = prod
    return out


def singular_values_2x2(A: Any, *, stable: bool = True) -> tuple[float, float]:
    r"""2×2 矩阵的奇异值 :math:`(\sigma_{\min}, \sigma_{\max})`，闭式解（不调用 SVD）。

    :math:`A^{\mathsf T}A` 的两个特征值是
    :math:`\lambda_\pm = E \pm \sqrt{E^2 - \det(A^{\mathsf T}A)}`（:math:`E = \|A\|_F^2/2`），
    于是 :math:`\sigma_\pm = \sqrt{\lambda_\pm}`。

    ``stable=True``（缺省）走**数值稳定**的等价写法：

    .. math::
        \sigma_{\max} = \sqrt{\lambda_+}, \qquad
        \sigma_{\min} = \frac{|\det A|}{\sigma_{\max}},

    因为减法的 :math:`\lambda_-` 在 :math:`\kappa` 大时会被相减抵消掉全部有效位——
    而 :math:`|\det A|/\sigma_{\max}` 不会。这条"用除法换减法"的技巧在 2×2 里能做到，
    也是 1.4 章讲奇异值分解时会再遇到的主题。

    返回 ``(smin, smax)``；:math:`\det A = 0` 时 :math:`\sigma_{\min} = 0`。
    """
    arr = np.asarray(A, dtype=float)
    if arr.shape != (2, 2):
        raise ValueError(f"只支持 2×2，得到 {arr.shape}")
    fro2 = float(np.sum(arr * arr))
    det = float(arr[0, 0] * arr[1, 1] - arr[0, 1] * arr[1, 0])
    e = 0.5 * fro2
    disc = max(0.0, e * e - det * det)
    lmax = e + np.sqrt(disc)
    smax = float(np.sqrt(max(0.0, lmax)))
    if stable:
        smin = abs(det) / smax if smax > 0.0 else 0.0
    else:
        lmin = e - np.sqrt(disc)
        smin = float(np.sqrt(max(0.0, lmin)))
    return smin, smax


def condition_2x2(A: Any, *, stable: bool = True) -> float:
    r"""2×2 的谱条件数 :math:`\kappa_2 = \sigma_{\max}/\sigma_{\min}`（闭式）。"""
    smin, smax = singular_values_2x2(A, stable=stable)
    if smin <= 0.0:
        return float("inf")
    return smax / smin


# --------------------------------------------------------------------------- #
# 两个演示矩阵族（章节与测试共用）
# --------------------------------------------------------------------------- #
def dominant_matrix(n: int, dominance: float, *, seed: int = 20240102) -> np.ndarray:
    r"""对角占优比**恰好**等于 ``dominance`` 的矩阵。

    .. math::
        |a_{ii}| = \text{dominance}\cdot\sum_{j\ne i}|a_{ij}| \quad (a_{ii} > 0)

    非对角元由固定种子生成（可复现）。:math:`\text{dominance} > 1` 即严格对角占优，
    此时无主元消元的乘子满足 :math:`|m_{ik}| \le 1`、:math:`\rho \le 2`——
    所以这个参数非常适合用来**实测**"充分条件"的边界。
    """
    from numcore.utils import rng  # 局部导入：避免 numcore 模块间循环依赖

    g = rng(seed + n)
    A = g.standard_normal((n, n))
    np.fill_diagonal(A, 0.0)
    off = np.sum(np.abs(A), axis=1)
    A[np.arange(n), np.arange(n)] = dominance * off + 1e-3
    return A


def ball_matrix(shear: float, kappa: float, *, angle_deg: float = 40.0) -> np.ndarray:
    r"""二维演示矩阵：先剪切、再各向异性拉伸（主轴旋转 ``angle_deg``）。

    .. math::
        A = R^{\mathsf T}\,\mathrm{diag}(1, 1/\kappa)\,R\,\begin{pmatrix}1 & s\\ 0 & 1\end{pmatrix}

    它的谱条件数大致按 :math:`\kappa` 增长（实测值由 :func:`singular_values_2x2` 给出），
    剪切因子 :math:`s` 决定几何形状偏得多厉害（:math:`\det A = 1/\kappa` 与 s 无关）。
    """
    th = np.deg2rad(angle_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    D = np.diag([1.0, 1.0 / kappa])
    S = np.array([[1.0, shear], [0.0, 1.0]])
    return R.T @ D @ R @ S
