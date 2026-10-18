r"""2.4 章 —— Richardson 外推与 Romberg 三角：把"低阶公式的误差"当资源用掉。

梯形法的误差有完整的 Euler--Maclaurin 展开（:math:`f` 足够光滑时）

.. math::
    T(h)=I+\tau_1h^2+\tau_2h^4+\tau_3h^6+\cdots,

其中 :math:`\tau_i` **与** :math:`h` **无关**。于是把两个步长的结果按

.. math::
    T_{k,j}=\frac{4^jT_{k,j-1}-T_{k-1,j-1}}{4^j-1},\qquad h_k=\frac{b-a}{2^k}

组合起来，就逐列消掉 :math:`h^{2},h^{4},\dots`：第 :math:`j` 列（从 0 数起）的误差是
:math:`O(h^{2j+2})`。第 0 列是梯形，第 1 列就是 Simpson。

这里有两个必须用实测数字说清的边界：

* **外推的放大是"有界"的，但确实不为 1**：每一列都是梯形值的线性组合，系数绝对值之和
  :math:`A_j=\sum|c_i|` 从 :math:`A_0=1` 升到 :math:`A_1=5/3`，然后**收敛到
  :math:`A_\infty=1.9692603\ldots`**（实测：j=4 时 1.9641387、j=14 时 1.9692603）。
  也就是说外推最多把输入的舍入噪声放大到约 2 倍——这才是"Romberg 稳定"这句话的量化版本
  （写成"每列涨 :math:`(4/3)^j`"是错的，本项目差点把它当结论写进笔记）。
  截断误差按 :math:`4^{-(j+1)}` 一路掉，舍入地板却停在 :math:`\approx2u|T|`：
  **两者相遇的那一行就是 Romberg 的实际精度极限**（本章实测：光滑函数上第 5 行、
  :math:`|T_{5,5}-I|\approx2.2\times10^{-16}`，再往上加行只会输出噪声）。
* **不光滑的函数上外推会骗人**：:math:`|x|` 或 :math:`\sqrt x` 的展开里
  有 :math:`h^{2\alpha+1}` 这类非整数幂，:math:`4^j` 组合消不掉它们，
  于是"每一列升高 2 阶"这条规律**当场失效**（实测见 :func:`column_orders`）。

外推的系数**不是**从教科书抄的：:func:`extrapolation_coefficients` 直接由递推
:math:`c^{(j)}=\big(4^jc^{(j-1)}-\text{shift}(c^{(j-1)})\big)/(4^j-1)` 算出来，
:math:`\sum|c^{(j)}|` 就是上面那个 :math:`A_j`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from numcore.quad.rates import fit_loglog

__all__ = [
    "RombergResult",
    "column_orders",
    "diagonal_errors",
    "extrapolation_coefficients",
    "extrapolation_gain",
    "richardson_step",
    "romberg_table",
    "trapezoid_levels",
]


def trapezoid_levels(
    f: Callable[[Any], Any],
    a: float,
    b: float,
    levels: int,
    *,
    with_counts: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    r"""逐次加倍步长的梯形序列 :math:`T_{k,0}`（:math:`k=0,\dots,\text{levels}-1`）。

    :math:`T_{k+1,0}=\tfrac12T_{k,0}+h_{k+1}\sum f(\text{新中点})`：每次加密**只算新点**，
    这正是 Romberg 划算的地方（第 k 层累计求值次数 :math:`2^k+1`）。
    ``with_counts=True`` 时同时返回累计求值次数。
    """
    m = int(levels)
    if m < 1:
        raise ValueError(f"levels 至少为 1，得到 {levels!r}")
    lo, hi = float(a), float(b)
    h = hi - lo
    f0, f1 = float(f(np.asarray([lo], dtype=float))[0]), float(f(np.asarray([hi], dtype=float))[0])
    vals = [0.5 * h * (f0 + f1)]
    counts = [2]
    for k in range(1, m):
        n_new = 2 ** (k - 1)
        h = (hi - lo) / (2**k)
        nodes = lo + h * (2.0 * np.arange(1, n_new + 1, dtype=float) - 1.0)
        vals.append(0.5 * vals[-1] + h * float(np.sum(f(nodes))))
        counts.append(counts[-1] + n_new)
    v = np.asarray(vals, dtype=float)
    c = np.asarray(counts, dtype=float)
    return (v, c) if with_counts else v


def richardson_step(t_coarse: float, t_fine: float, order: int = 2) -> float:
    r"""一步 Richardson 外推 :math:`(2^{q}T_{\text{fine}}-T_{\text{coarse}})/(2^{q}-1)`。

    ``order=2`` 时因子 :math:`2^2=4`（消掉 :math:`h^2` 项，得到 Simpson）。
    """
    q = int(order)
    if q < 1:
        raise ValueError(f"order 至少为 1，得到 {order!r}")
    factor = float(2**q)
    return (factor * float(t_fine) - float(t_coarse)) / (factor - 1.0)


def extrapolation_coefficients(column: int, row: int | None = None) -> np.ndarray:
    r"""第 ``column`` 列外推相对**:math:`T_{\cdot,0}`（梯形值）**的组合系数。

    返回长度 ``column+1`` 的向量 :math:`c`，使得
    :math:`T_{k,j}=\sum_{i=0}^{j}c_iT_{k-i,0}`（``row`` 只用于校验 :math:`k\ge j`）。
    :math:`\sum|c|` 就是该列把输入舍入噪声放大的倍数（有界，实测上确界 1.9692603…，
    且 :math:`\sum c_i=1` —— 组合必须能精确重现常数，否则连常函数都积不对）。
    """
    j = int(column)
    if j < 0:
        raise ValueError(f"column 不能为负，得到 {column!r}")
    if row is not None and int(row) < j:
        raise ValueError(f"第 {j} 列至少需要第 {j} 行，得到 row={row!r}")
    c = np.array([1.0])
    for s in range(1, j + 1):
        factor = float(2 ** (2 * s))
        nxt = np.zeros(c.size + 1, dtype=float)
        nxt[: c.size] += factor * c
        nxt[1:] -= c
        c = nxt / (factor - 1.0)
    return c


@dataclass(frozen=True)
class RombergResult:
    r"""Romberg 三角的完整快照（未用到的上三角是 ``nan``，不进帧）。

    ``table[k, j]`` 只在 :math:`j\le k` 有意义；取用请走 :meth:`column` / :meth:`diagonal`
    或 :meth:`long_form`（后者把 :math:`(j,k)` 拉直成一维，帧里只出现有限值）。
    """

    level: np.ndarray
    n_sub: np.ndarray
    evals: np.ndarray
    table: np.ndarray
    #: 已知解析真值时一并带上（``column_errors`` 需要它；没有就报错而不是编一个 0）。
    exact: float | None = None

    @property
    def size(self) -> int:
        return int(self.table.shape[0])

    def diagonal(self) -> np.ndarray:
        """对角元 :math:`T_{k,k}`（每一行能达到的最高阶估计）。"""
        m = self.size
        return np.asarray([self.table[k, k] for k in range(m)], dtype=float)

    def column(self, j: int) -> np.ndarray:
        """第 ``j`` 列的有效段 :math:`T_{k,j}`（``k=j..m-1``）。"""
        jj = int(j)
        if jj < 0 or jj >= self.size:
            raise ValueError(f"列号 {j} 超出 [0,{self.size - 1}]")
        return np.asarray([self.table[k, jj] for k in range(jj, self.size)], dtype=float)

    def long_form(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """拉直成 ``(j, k, value, error)`` 四个一维数组（只含有效三角，无 nan）。

        帧里画三角表只能用它：``nan`` 进不了 ``.nlf``（NLF 头用 ``allow_nan=False`` 编码），
        而且"未定义"与"算出来是 0"在图上必须是两件事。
        """
        js: list[int] = []
        ks: list[int] = []
        vs: list[float] = []
        m = self.size
        for k in range(m):
            for j in range(k + 1):
                js.append(j)
                ks.append(k)
                vs.append(float(self.table[k, j]))
        return (
            np.asarray(js, dtype=float),
            np.asarray(ks, dtype=float),
            np.asarray(vs, dtype=float),
            self.column_errors(),
        )

    def column_errors(self) -> np.ndarray:
        """与 :meth:`long_form` 同序的 :math:`|T_{k,j}-I|`（需要构造时给了 ``exact``）。"""
        if self.exact is None:
            raise ValueError("构造时没有给 exact，无法给出误差（不要用一个编出来的 0 顶替）")
        out = []
        m = self.size
        for k in range(m):
            for j in range(k + 1):
                out.append(abs(float(self.table[k, j]) - self.exact))
        return np.asarray(out, dtype=float)


def romberg_table(
    f: Callable[[Any], Any],
    a: float,
    b: float,
    levels: int,
    *,
    exact: float | None = None,
) -> RombergResult:
    r"""Romberg 三角：第 0 列是梯形序列，第 ``j`` 列消掉到 :math:`h^{2j}` 的项。"""
    m = int(levels)
    if m < 2:
        raise ValueError(f"Romberg 至少 2 层，得到 {levels!r}")
    vals, counts = trapezoid_levels(f, a, b, m, with_counts=True)
    table = np.full((m, m), np.nan, dtype=float)
    table[:, 0] = vals
    for j in range(1, m):
        factor = float(2 ** (2 * j))  # 4^j
        for k in range(j, m):
            table[k, j] = (factor * table[k, j - 1] - table[k - 1, j - 1]) / (factor - 1.0)
    return RombergResult(
        level=np.arange(m, dtype=float),
        n_sub=2.0 ** np.arange(m, dtype=float),
        evals=counts,
        table=table,
        exact=None if exact is None else float(exact),
    )


def diagonal_errors(result: RombergResult, exact: float) -> np.ndarray:
    r"""对角线误差 :math:`|T_{k,k}-I|`（第 k 行的最优估计）。"""
    return np.abs(result.diagonal() - float(exact))


def column_orders(
    result: RombergResult,
    exact: float,
    *,
    trim_tail: int = 0,
    floor: float | None = None,
) -> np.ndarray:
    r"""每一列**实测**的收敛阶（对该列的有效段做双对数拟合）。

    理论值：第 :math:`j` 列是 :math:`2j+2`（:math:`j=0` 梯形 2 阶、:math:`j=1` Simpson 4 阶…）。
    注意 :math:`h` **递减**、误差也递减，所以双对数斜率是**正的** :math:`+p`（不要取负号）。
    非光滑函数上这条规律失效，实测值会明显偏离——**偏离本身就是证据**，
    所以这个函数返回的是实测向量，而不是理论向量。
    """
    m = result.size
    out = np.full(m, np.nan, dtype=float)
    # 舍入地板的默认估计：u·max(1,|I|)。触到地板的行必须剔掉，否则拟合的是噪声：
    # 实测第 3 列在 10 层时不剔地板给出 6.96，剔掉后 7.97（理论 8）。
    if floor is None:
        floor = 2.0**-53 * max(1.0, abs(float(exact)))
    for j in range(m):
        k = np.arange(j, m)
        if k.size < 3:
            continue
        h = (result.n_sub[k]) ** -1.0
        # 该列的有效值按 h 递减排列（k 增大 → h 减小）
        err = np.abs(result.column(j) - float(exact))
        good = np.isfinite(err) & (err > 4.0 * float(floor))
        if np.count_nonzero(good) < 3:
            continue
        # 只保留"前 count 个仍在地板之上的行"：地板之上的行才是有效窗口
        n_good = int(np.count_nonzero(good))
        keep = np.zeros(err.size, dtype=bool)
        keep[np.flatnonzero(good)[:n_good]] = True
        good = keep
        # 裁尾不能把窗口裁到 3 个点以下：否则早期揭示（表还很小）时一列都算不出阶，
        # 右图会一直是空的——实测踩过这个坑，"阶"要么给出来要么如实说不够，
        # 不能因为窗口比裁剪量还短就整列消失。
        stop = max(3, err.size - int(trim_tail)) if trim_tail else err.size
        stop = min(stop, err.size)
        hh, ee = h[:stop][good[:stop]], err[:stop][good[:stop]]
        if hh.size < 3:
            continue
        out[j] = fit_loglog(hh, ee).slope
    return out


def extrapolation_gain(result: RombergResult, exact: float) -> np.ndarray:
    r"""同一行上相邻两列的**加速比** :math:`|T_{k,j}-I|/|T_{k,j+1}-I|`（长表，``(j,k)`` 序）。

    它回答的是"这一列外推值不值"：比值远大于 1 说明外推确实把误差压下去了；
    比值掉到 ~1 说明已经在外推的精度极限上（再外推只是放大噪声）。
    """
    m = result.size
    js: list[int] = []
    ks: list[int] = []
    gains: list[float] = []
    for k in range(m):
        for j in range(k):
            e0 = abs(float(result.table[k, j]) - float(exact))
            e1 = abs(float(result.table[k, j + 1]) - float(exact))
            with np.errstate(divide="ignore", invalid="ignore"):
                g = e0 / e1 if e1 > 0.0 else float("inf")
            js.append(j)
            ks.append(k)
            gains.append(g)
    return np.asarray(js, dtype=float), np.asarray(ks, dtype=float), np.asarray(gains, dtype=float)
