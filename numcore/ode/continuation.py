r"""3.5 章（下半）：**分岔与参数延拓**——周期轨、乘子、Feigenbaum 与 Hopf 的数值判定。

数学对象
--------

1. **周期轨与乘子**。:math:`p` 周期轨是 :math:`F(x)=g^p(x)-x=0` 的根，
   :math:`F'(x)=\prod_{i=0}^{p-1}g'(x_i)-1`；乘子 :math:`M=\prod g'(x_i)` 决定稳定性
   （:math:`|M|<1` 稳定）与分岔方向（:math:`M=-1` 周期倍增，:math:`M=+1` 切分岔）。
   本模块用 Newton 在 :math:`F` 上求根（:func:`cycle_newton`），**乘子是算出来的**，
   不是从"周期 p 出现了"推出来的——这两件事在窗口边缘差得很远。
2. **周期倍增点** :math:`r_n`：乘子穿过 :math:`-1` 的参数值，用二分实测
   （:func:`period_doubling_points`）。:math:`r_1=3` 与 :math:`r_2=1+\sqrt6` 有闭式，
   可以对拍；更高阶没有闭式，只用它们的**比值**（Feigenbaum）。
3. **Feigenbaum 常数** :math:`\delta=\lim\frac{R_n-R_{n-1}}{R_{n+1}-R_n}`（:func:`feigenbaum_delta`），
   其中 :math:`R_n` 是**超稳定参数**：:math:`2^{n-1}` 周期轨经过 :math:`x=1/2`
   （:func:`superstable_parameters`）。为什么用超稳定点而不是分岔点：超稳定点上乘子为 0，
   迭代收敛是二次的，数值上干净得多；而分岔点上乘子为 :math:`\pm1`，附近根本收不敛。
   :math:`R_2=1+\sqrt5` 也有闭式，可以对拍。
4. **Hopf 分岔**：平衡点的一对特征值穿过虚轴（:func:`branch_max_real_part` +
   :func:`hopf_threshold`）。Lorenz 的 :math:`C_\pm` 有闭式 :math:`\rho_H=470/19`，
   二分实测值与它对拍。

自研边界（P2/P3）
-----------------
只用逐元素运算与 :mod:`numcore.linalg.lu`；参数网格上的迭代一律**向量化**
（:func:`bifurcation_diagram`、:func:`detect_periods`），否则 700 个参数 × 2000 步暂态
在 Python 里要跑几千万次标量循环。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from numcore.ode import dynamics as dyn

__all__ = [
    "Cycle",
    "DoublingSpectrum",
    "analytic_two_cycle",
    "cycle_newton",
    "multiplier",
    "detect_period",
    "detect_periods",
    "bifurcation_diagram",
    "cycle_residuals",
    "cobweb",
    "period_doubling_points",
    "superstable_parameters",
    "feigenbaum_delta",
    "accumulation_point",
    "tangent_window_start",
    "branch_max_real_part",
    "hopf_threshold",
    "logistic",
]


def logistic(r: float) -> Callable[[float], float]:
    """标量 logistic 映射（分岔分析里最常用的那一维）。"""
    return lambda x: r * x * (1.0 - x)


# --------------------------------------------------------------------------- #
# 1. 周期轨：Newton 求根 + 乘子
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Cycle:
    """一条 :math:`p` 周期轨：轨道点、乘子、Newton 残差。"""

    points: np.ndarray
    period: int
    multiplier: float
    residual: float

    @property
    def stable(self) -> bool:
        return abs(self.multiplier) < 1.0


def multiplier(g: dyn.MapRHS, dg: dyn.MapRHS, points: Any) -> float:
    r"""乘子 :math:`M=\prod_i g'(x_i)`（沿给定顺序绕轨道一圈）。"""
    P = np.asarray(points, dtype=float)
    m = 1.0
    for x in P:
        m *= float(np.asarray(dg(np.asarray([x], dtype=float)), dtype=float).reshape(-1)[0])
    return m


def cycle_newton(
    g: dyn.MapRHS,
    dg: dyn.MapRHS,
    period: int,
    *,
    seed: float = 0.4,
    transient: int = 4000,
    tol: float = 1e-14,
    max_iter: int = 120,
) -> Cycle | None:
    r"""求 :math:`g^p(x)-x=0` 的一个根（Newton），返回整条轨道与乘子。

    种子先迭代 ``transient`` 次落到吸引子附近——直接拿 :math:`x_0=0.4` 做 Newton 常常
    收敛到**不动点**（它也是 :math:`g^p(x)=x` 的根），于是"周期 4"的轨道其实是 1 周期。
    这里用"迭代种子 + 拒绝不动点"两步把它排除：Newton 收敛后检查
    :math:`|g(x)-x|>10^{-8}`，是则判定为不动点、返回 ``None``。

    ``max_iter`` 用满仍不收敛也返回 ``None``——分岔点上 :math:`F'=0`，Newton 本来就该失败，
    这是要报告的状态（调用方据此改用二分/超稳定点）。
    """
    p = int(period)
    if p < 1:
        raise ValueError("周期至少为 1")
    x = float(seed)
    for _ in range(int(transient)):
        x = float(np.asarray(g(np.asarray([x], dtype=float)), dtype=float).reshape(-1)[0])
    for _ in range(int(max_iter)):
        # F(x) = g^p(x) − x 与 F'(x) = Π g'(x_i) − 1
        y = x
        prod = 1.0
        for _ in range(p):
            prod *= float(np.asarray(dg(np.asarray([y], dtype=float)), dtype=float).reshape(-1)[0])
            y = float(np.asarray(g(np.asarray([y], dtype=float)), dtype=float).reshape(-1)[0])
        F = y - x
        dF = prod - 1.0
        if abs(F) <= tol * max(1.0, abs(x)):
            break
        if dF == 0.0 or not math.isfinite(dF):
            return None
        step = -F / dF
        x = x + step
        if abs(step) <= 1e-16 * max(1.0, abs(x)):
            break
    # 轨道点与残差
    pts = []
    y = x
    for _ in range(p):
        pts.append(y)
        y = float(np.asarray(g(np.asarray([y], dtype=float)), dtype=float).reshape(-1)[0])
    P = np.asarray(pts, dtype=float)
    res = abs(y - x)
    if res > 1e-10:
        return None
    if p > 1:
        gp1 = float(np.asarray(g(np.asarray([x], dtype=float)), dtype=float).reshape(-1)[0])
        if abs(gp1 - x) <= 1e-8:
            return None  # 收敛到不动点，不是 p 周期
    return Cycle(points=P, period=p, multiplier=multiplier(g, dg, P), residual=res)


def analytic_two_cycle(r: float) -> tuple[np.ndarray, float]:
    r"""2 周期的闭式轨道点与乘子（:math:`3<r` 时两个实根）。

    :math:`x_{1,2}=\dfrac{r+1\pm\sqrt{(r-3)(r+1)}}{2r}`，乘子
    :math:`M=g'(x_1)g'(x_2)=4+2r-r^2`（推导见 ``notes.tex``）。
    """
    if r <= 3.0:
        raise ValueError("2 周期在 r>3 才出现")
    disc = (r - 3.0) * (r + 1.0)
    sq = math.sqrt(disc)
    x1 = (r + 1.0 + sq) / (2.0 * r)
    x2 = (r + 1.0 - sq) / (2.0 * r)
    return np.array([x1, x2]), 4.0 + 2.0 * r - r * r


# --------------------------------------------------------------------------- #
# 2. 参数网格上的周期检测与分岔图（向量化）
# --------------------------------------------------------------------------- #
def detect_period(
    g: dyn.MapRHS,
    x0: float,
    *,
    max_period: int = 32,
    transient: int = 4000,
    tol: float = 1e-9,
) -> tuple[int, float]:
    r"""轨道落在哪个周期上：返回 ``(最小周期 p, 闭合残差)``，``p=0`` 表示"≤ max_period 内不闭合"。

    判据是 :math:`|g^p(x)-x|\le\text{tol}`。**没有**用"乘子是否小于 1"当判据：
    窗口里同时存在稳定与不稳定周期轨，乘子判据会把不稳定轨也算成"吸引子"。
    对混沌轨道 ``p=0``，同时给出 :math:`\min_{p\le P}|g^p(x)-x|` 作为"最接近闭合"的残差——
    它不小（实测 1e-2 量级），这正是"混沌画不进周期框"的量化说法。
    """
    x = float(x0)
    for _ in range(int(transient)):
        x = float(np.asarray(g(np.asarray([x], dtype=float)), dtype=float).reshape(-1)[0])
    y = x
    best = float("inf")
    for p in range(1, int(max_period) + 1):
        y = float(np.asarray(g(np.asarray([y], dtype=float)), dtype=float).reshape(-1)[0])
        res = abs(y - x)
        best = min(best, res)
        if res <= tol:
            return p, res
    return 0, best


def detect_periods(
    r_grid: Any,
    *,
    max_period: int = 32,
    transient: int = 4000,
    x0: float = 0.4,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    r"""``detect_period`` 的**参数批**版本：一次推进整条 r 网格。

    返回 ``(周期数组, 闭合残差数组)``。位图式的分岔图能看出"这里有个窗口"，
    但窗口里到底是 3 周期还是 6 周期只能靠这个数组说话；两个数组一起给前端，
    才不至于让"周期窗口被画糊"变成一句空话。
    """
    R = np.asarray(r_grid, dtype=float).reshape(-1)
    x = np.full(R.size, float(x0))
    for _ in range(int(transient)):
        x = R * x * (1.0 - x)
    periods = np.zeros(R.size, dtype=int)
    resid = np.full(R.size, np.inf)
    y = x.copy()
    for p in range(1, int(max_period) + 1):
        y = R * y * (1.0 - y)
        res = np.abs(y - x)
        resid = np.minimum(resid, res)
        hit = (periods == 0) & (res <= tol)
        periods[hit] = p
    return periods, resid


def bifurcation_diagram(
    r_grid: Any,
    *,
    n_transient: int = 1000,
    n_keep: int = 64,
    x0: float = 0.4,
) -> np.ndarray:
    r"""logistic 分岔图的**点云**：形状 ``(n_keep, n_r)``（每列一个参数值）。

    ``n_transient`` 之后留下的 ``n_keep`` 个迭代值就是"吸引子在这个参数下的形状"。
    分辨率的两个旋钮都摆在参数里给用户拖：
    :math:`r` 方向的网格点数 ``n_r`` 与竖直方向每条轨道留下的点数 ``n_keep``。
    窗口窄到 :math:`10^{-3}` 量级（例如 3 周期窗口起点之后的倍周期级联）时，
    :math:`\Delta r` 不够小就会把整串窗口糊成一片黑——这是本章 ``pitfalls`` 的第一条。
    """
    R = np.asarray(r_grid, dtype=float).reshape(-1)
    x = np.full(R.size, float(x0))
    for _ in range(int(n_transient)):
        x = R * x * (1.0 - x)
    out = np.zeros((int(n_keep), R.size))
    for k in range(int(n_keep)):
        x = R * x * (1.0 - x)
        out[k] = x
    return out


def cycle_residuals(r_grid: Any, periods: Any, *, transient: int = 4000, x0: float = 0.4) -> np.ndarray:
    r"""逐参数给出"检测到的周期上轨道的闭合残差" :math:`|g^p(x)-x|`。

    这是帧里的 ``residual`` 数组：周期窗口里它是 :math:`10^{-15}` 量级，
    混沌区里它是 :math:`10^{-2}` 量级——**同一个图上的两类点，靠这个数区分**，
    不靠肉眼。
    """
    R = np.asarray(r_grid, dtype=float).reshape(-1)
    P = np.asarray(periods, dtype=int).reshape(-1)
    if P.size != R.size:
        raise ValueError("periods 与 r_grid 长度必须一致")
    x = np.full(R.size, float(x0))
    for _ in range(int(transient)):
        x = R * x * (1.0 - x)
    out = np.zeros(R.size)
    max_p = int(np.max(P)) if P.size else 0
    for p in range(1, max(1, max_p) + 1):
        y = x.copy()
        for _ in range(p):
            y = R * y * (1.0 - y)
        m = P == p
        out[m] = np.abs(y - x)[m]
    # 混沌参数（P=0）没有"周期"，给"最接近闭合"的残差：它对 p≤32 取最小。
    # 这一支必须无条件算，不能只在"全表都没有周期"时才算——否则混沌区的残差会
    # 留在 0.0 上，图上看起来"混沌也完美闭合"（本章初版真踩过这个坑）。
    m0 = P == 0
    if np.any(m0):
        y = x.copy()
        best = np.full(R.size, np.inf)
        for _ in range(max(32, max_p)):
            y = R * y * (1.0 - y)
            best = np.minimum(best, np.abs(y - x))
        out[m0] = best[m0]
    return out


def cobweb(g: dyn.MapRHS, x0: float, *, n_steps: int = 120, transient: int = 200) -> tuple[np.ndarray, np.ndarray]:
    r"""蛛网图（cobweb）折线：``(x0,0)→(x0,g(x0))→(g(x0),g(x0))→…``，长度 ``2n+2``。

    它是"迭代"这件事最直接的图形：不动点 = 折线落在对角线上，
    2 周期 = 一个方框，混沌 = 折线填满一段区间。
    """
    x = float(x0)
    for _ in range(int(transient)):
        x = float(np.asarray(g(np.asarray([x], dtype=float)), dtype=float).reshape(-1)[0])
    xs = [x, x]
    ys = [0.0, x]
    cur = x
    for _ in range(int(n_steps)):
        nxt = float(np.asarray(g(np.asarray([cur], dtype=float)), dtype=float).reshape(-1)[0])
        xs.extend([cur, nxt])
        ys.extend([nxt, nxt])
        cur = nxt
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


# --------------------------------------------------------------------------- #
# 3. 周期倍增点、超稳定参数、Feigenbaum
# --------------------------------------------------------------------------- #
def _sign_change_brackets(F: np.ndarray) -> np.ndarray:
    """找出 ``F`` 里所有**含根**的相邻格点对（返回左端点下标）。

    判据不能写成 :math:`F_iF_{i+1}<0`：网格节点**恰好落在根上**时（logistic 映射的
    不动点 :math:`x=1-1/r` 很容易正好落在等距节点上，例如 :math:`r=3.2` 给
    :math:`x=0.6875=13750/20000`），那个节点上的值是精确 0，两侧乘积都是 0 而不是负，
    严格不等式会**整条漏掉这个根**——本仓实测过：`cycle_multipliers(3.2, 1)` 因此返回空数组。
    这里把"端点正好是根"也当成一个括号，交给二分（二分在 ``f(lo)=0`` 时会一直收紧右端，
    最后收敛到那个精确根）。
    """
    z = np.asarray(F, dtype=float)
    left = z[:-1]
    right = z[1:]
    hit = (left * right < 0.0) | ((left == 0.0) & (right != 0.0)) | ((right == 0.0) & (left != 0.0))
    return np.nonzero(hit)[0]


def _bisect_vector(f: Callable[[np.ndarray], np.ndarray], a: np.ndarray, b: np.ndarray, steps: int = 70) -> np.ndarray:
    """对一批括号同时做二分（``f`` 必须能一次吃下整批）。"""
    lo = np.asarray(a, dtype=float).copy()
    hi = np.asarray(b, dtype=float).copy()
    flo = np.asarray(f(lo), dtype=float)
    for _ in range(int(steps)):
        mid = 0.5 * (lo + hi)
        fm = np.asarray(f(mid), dtype=float)
        neg = flo * fm <= 0.0
        hi = np.where(neg, mid, hi)
        lo = np.where(neg, lo, mid)
        flo = np.where(neg, flo, fm)
    return 0.5 * (lo + hi)


def cycle_multipliers(r: float, period: int, *, n_scan: int = 20001, tol: float = 1e-13) -> np.ndarray:
    r"""参数 ``r`` 处**所有恰好 p 周期**轨道的乘子（不依赖 Newton 的收敛域）。

    做法是把 :math:`F(x)=g^p(x)-x` 在 :math:`[0,1]` 上按细网格求值、找出全部符号变化，
    再用**向量化二分**把根压到机器精度；然后用 :math:`g^d(x)=x\ (d\mid p,\ d<p)`
    过滤掉低周期的根，最后逐根算 :math:`M=\prod g'(x_i)`。

    为什么不用 Newton：周期倍增点上 :math:`F'=0`，Newton 在那里本来就失效；
    而在"r 稍小于 :math:`r_n`"的一侧，吸引子虽然是 p 周期，但收敛速度是代数级的
    （乘子趋近 −1），固定次数的迭代根本落不到 1e-6 以内。
    网格法的代价是 :math:`O(p\cdot n_{scan})`，对 :math:`p\le8` 是毫秒级，
    并且给出**全部**周期轨（包括不稳定的），这正好是"周期窗口里到底有几条轨"要问的事。
    """
    p = int(period)
    if p < 1:
        raise ValueError("周期至少为 1")
    xs = np.linspace(0.0, 1.0, int(n_scan))
    y = xs.copy()
    for _ in range(p):
        y = r * y * (1.0 - y)
    F = y - xs
    idx = _sign_change_brackets(F)
    if idx.size == 0:
        return np.zeros(0)
    root = _bisect_vector(lambda z: _gpow(r, z, p) - z, xs[idx], xs[idx + 1])
    # 去重：根正好落在网格节点上时，它左右两个括号都会被判为"含根"，
    # 二分后得到两个相差 1e-16 的重复根（实测 r=3.2 的 p=1 会因此多出一个 −1.2）。
    if root.size > 1:
        rs = np.sort(root)
        keep_u = np.ones(rs.size, dtype=bool)
        keep_u[1:] = np.diff(rs) > 1e-9
        root = rs[keep_u]
    # 精确周期过滤：d < p 且 g^d(x) = x 的根属于更短的周期
    keep = np.ones(root.size, dtype=bool)
    for d in range(1, p):
        if p % d:
            continue
        keep &= np.abs(_gpow(r, root, d) - root) > 1e-9
    root = root[keep]
    if root.size == 0:
        return np.zeros(0)
    # 乘子
    m = np.ones(root.size)
    z = root.copy()
    for _ in range(p):
        m = m * (r * (1.0 - 2.0 * z))
        z = r * z * (1.0 - z)
    return m


def _gpow(r: float, x: Any, p: int) -> np.ndarray:
    """:math:`g^p(x)`（向量化；只用于 logistic 映射的分岔分析）。"""
    z = np.asarray(x, dtype=float).copy()
    for _ in range(int(p)):
        z = r * z * (1.0 - z)
    return z


def period_doubling_points(
    *,
    max_period: int = 8,
    r_lo: float = 2.9,
    r_hi: float = 4.0,
    scan: int = 3000,
    bisect: int = 60,
) -> np.ndarray:
    r"""逐级二分出周期倍增点 :math:`r_1=3,\ r_2=1+\sqrt6,\ r_3,\ldots`。

    谓词是"**存在**恰好 :math:`p` 周期的轨道且 :math:`|M|<1`"（:math:`p=1,2,4,\ldots`）：
    在倍增点左侧稳定轨还在（:math:`M>-1`），右侧它失稳、吸引子变成 :math:`2p` 周期。
    谓词只用 :func:`cycle_multipliers`（网格 + 二分，不依赖 Newton 的收敛域）。

    实测（本仓，``scan=3000``）：:math:`r_1=3.000000000000`、
    :math:`r_2=3.449489742783`、:math:`r_3=3.544090359552`、:math:`r_4=3.564407266095`
    ——前两个与 :math:`3`、:math:`1+\sqrt6` 的差在 :math:`10^{-13}` 内。

    ``max_period`` 建议不超过 8：:math:`g^p(x)-x` 在 :math:`[0,1]` 上有 :math:`2^p` 个根，
    :math:`p=16` 时 65536 个根已经密到网格分辨不出符号变化（那时该换超稳定参数口径）。
    """
    out: list[float] = []
    lo_bound = float(r_lo)
    period = 1
    while period <= int(max_period):
        grid = np.linspace(lo_bound, float(r_hi), int(scan))
        prev = bool(np.any(np.abs(cycle_multipliers(float(grid[0]), period)) < 1.0))
        found: tuple[float, float] | None = None
        for r in grid[1:]:
            cur = bool(np.any(np.abs(cycle_multipliers(float(r), period)) < 1.0))
            if prev and not cur:
                found = (float(r) - (grid[1] - grid[0]), float(r))
                break
            prev = cur
        if found is None:
            break
        a, b = found
        for _ in range(int(bisect)):
            mid = 0.5 * (a + b)
            if bool(np.any(np.abs(cycle_multipliers(mid, period)) < 1.0)):
                a = mid
            else:
                b = mid
        r_star = 0.5 * (a + b)
        out.append(r_star)
        lo_bound = r_star + 1e-9
        period *= 2
    return np.asarray(out, dtype=float)


def superstable_parameters(
    *,
    max_n: int = 6,
    r_hi: float = 4.0,
    scan: int = 200001,
    bisect: int = 90,
) -> np.ndarray:
    r"""超稳定参数 :math:`R_n`：:math:`g^{2^{n-1}}(1/2)=1/2` 且轨道**恰好**是 :math:`2^{n-1}` 周期。

    :math:`R_1=2`、:math:`R_2=1+\sqrt5` 有闭式；更高阶只能数值求。
    做法是**向量化**扫描 + 二分：整条 r 网格一次迭代 :math:`2^{n-1}` 次
    （:math:`h_p(r)=g^p_r(1/2)-1/2` 是逐元素映射，天然可批），再按"轨道恰好 p 周期"
    过滤掉低周期的根——**这一步不能省**：:math:`R_5`（16 周期）也是 :math:`h_{32}` 的根，
    不筛就会把 :math:`R_6` 读成 :math:`R_5`。

    实测（``scan=200001``）：:math:`R_2-R_2^{闭式}=0`、
    :math:`R_3=3.4985616993`、:math:`R_4=3.5546408628`、:math:`R_5=3.5666673799`、
    :math:`R_6=3.5692435316`——与文献值逐位一致。
    """
    out: list[float] = [2.0]
    for n in range(2, int(max_n) + 1):
        p = 2 ** (n - 1)
        lo = float(out[-1]) + 1e-12
        R = np.linspace(lo, float(r_hi), int(scan))
        h = _gpow_vec(R, 0.5, p) - 0.5
        half = _gpow_vec(R, 0.5, p // 2) - 0.5
        exact = np.abs(half) > 1e-9
        idx = _sign_change_brackets(h)
        idx = idx[exact[idx] & exact[idx + 1]]
        if idx.size == 0:
            raise ValueError(f"R_{n} 的扫描区间里没有变号（p={p}）：把 scan 调大或检查区间")
        a, b = float(R[idx[0]]), float(R[idx[0] + 1])
        for _ in range(int(bisect)):
            mid = 0.5 * (a + b)
            fm = float(_gpow_vec(np.asarray([mid]), 0.5, p)[0] - 0.5)
            if fm == 0.0:
                break
            if (_gpow_vec(np.asarray([a]), 0.5, p)[0] - 0.5) * fm <= 0.0:
                b = mid
            else:
                a = mid
        out.append(0.5 * (a + b))
    return np.asarray(out, dtype=float)


def _gpow_vec(R: Any, x0: float, p: int) -> np.ndarray:
    """参数批版迭代：``R`` 是参数向量，``x0`` 是统一初值（返回同形状）。"""
    Rv = np.asarray(R, dtype=float)
    x = np.full(Rv.shape, float(x0))
    for _ in range(int(p)):
        x = Rv * x * (1.0 - x)
    return x


def feigenbaum_delta(superstable: Any) -> np.ndarray:
    r"""相邻间隔之比 :math:`\delta_n=\dfrac{R_n-R_{n-1}}{R_{n+1}-R_n}`（趋向 4.6692016091…）。

    序列的**尾部**才有意义：前几项离极限很远（实测 :math:`\delta_2=2.16`、
    :math:`\delta_3=4.20`、:math:`\delta_4=4.60`、:math:`\delta_5=4.66`）。
    只报最后一个数而不报整条序列，是这一章最容易犯的"用收敛中段冒充极限"的错。
    """
    R = np.asarray(superstable, dtype=float).reshape(-1)
    if R.size < 3:
        raise ValueError("至少需要三个超稳定参数")
    d = np.diff(R)
    return d[:-1] / d[1:]


def accumulation_point(superstable: Any, *, delta: float = 4.669201609102990) -> float:
    r"""由超稳定参数外推聚点 :math:`r_\infty\approx R_n+\dfrac{R_n-R_{n-1}}{\delta-1}`。

    （等比级数求和：:math:`\sum_{k\ge0}\delta^{-k}=\delta/(\delta-1)`。）
    实测外推与文献值 :math:`3.569945672\ldots` 差在 :math:`10^{-7}` 量级。
    """
    R = np.asarray(superstable, dtype=float).reshape(-1)
    if R.size < 2:
        raise ValueError("至少需要两个超稳定参数")
    return float(R[-1]) + (float(R[-1]) - float(R[-2])) / (float(delta) - 1.0)


def tangent_window_start(
    *,
    period: int = 3,
    r_lo: float = 3.80,
    r_hi: float = 3.838,
    bisect: int = 60,
) -> float:
    r"""切分岔窗口起点：周期 ``period`` 的轨道第一次以 :math:`M=+1` 诞生（且稳定）。

    :math:`p=3` 的窗口起点有闭式 :math:`r=1+\sqrt8=3.8284271247\ldots`
    （:math:`g^3(x)-x` 出现二重根，判据"存在稳定轨道"由假变真）。
    实测：:math:`3.828427124746`，与闭式差 :math:`2\times10^{-13}`。

    ``r_hi`` 必须在**窗口内部**：3 周期窗口只到 :math:`r\approx3.8415`
    （那里 3 周期自己又倍周期成 6 周期），取 3.86 会让端点假设反过来——
    这是本函数初版真踩过的坑（报"端点假设不成立"）。
    """
    def inside(r: float) -> bool:
        m = cycle_multipliers(float(r), int(period), n_scan=20001)
        return bool(np.any(np.abs(m) < 1.0))

    a, b = float(r_lo), float(r_hi)
    if inside(a) or not inside(b):
        raise ValueError(
            "端点假设不成立：需要 inside(lo)=False、inside(hi)=True。"
            "常见原因：r_hi 取到了窗口外（3 周期窗口上界约 3.8415）"
        )
    for _ in range(int(bisect)):
        mid = 0.5 * (a + b)
        if inside(mid):
            b = mid
        else:
            a = mid
    return 0.5 * (a + b)


# --------------------------------------------------------------------------- #
# 4. 平衡点的参数延拓与 Hopf 阈值
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DoublingSpectrum:
    """周期倍增级联的实测汇总（帧与断言共用）。"""

    r_points: np.ndarray
    superstable: np.ndarray
    delta_ratios: np.ndarray
    r_infinity: float


def branch_max_real_part(
    make_system: Callable[[float], dyn.FlowSystem],
    values: Any,
    *,
    seed: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""沿参数延拓一条平衡点分支，返回 ``(参数, 平衡点, max Re λ)``。

    延拓（continuation）的意思是：每个参数值上的 Newton 都用**上一个参数值解出来的点**
    当初值——比每个点都从固定种子重来稳得多（Lorenz 的 :math:`C_\pm` 在
    :math:`\rho` 很大时离原点很远，固定种子的 Newton 会跑到另一支上去）。
    """
    vals = np.asarray(values, dtype=float).reshape(-1)
    pts = np.zeros((vals.size, int(np.asarray(seed, dtype=float).size)))
    mre = np.zeros(vals.size)
    cur = np.asarray(seed, dtype=float).reshape(-1)
    for i, v in enumerate(vals):
        sys_i = make_system(float(v))
        root = dyn.newton_root(sys_i.f, sys_i.jac_at, cur, tol=1e-13, max_iter=80)
        if root is None:
            pts[i] = cur
            mre[i] = float("nan")
            continue
        cur = root
        pts[i] = root
        mre[i] = float(np.max(dyn.eigenvalues(sys_i.jac_at(root)).real))
    return vals, pts, mre


def hopf_threshold(
    make_system: Callable[[float], dyn.FlowSystem],
    *,
    lo: float,
    hi: float,
    seed: Any,
    bisect: int = 60,
) -> float:
    r"""二分实测 Hopf 阈值：平衡点分支上 :math:`\max\operatorname{Re}\lambda` 由负变正的参数值。

    Lorenz 的 :math:`C_\pm` 在 :math:`\rho_H=\sigma(\sigma+\beta+3)/(\sigma-\beta-1)`
    发生 Hopf；本函数**不用**这个公式，只用"特征值实部穿过 0"这个事实，
    公式留给断言去对拍。
    """
    def crossing_at(v: float, *, cont: bool = True) -> float:
        vals = np.linspace(float(lo), float(v), 41) if cont else np.asarray([float(v)])
        _, _, mre = branch_max_real_part(make_system, vals, seed=seed)
        return float(mre[-1])

    a, b = float(lo), float(hi)
    fa, fb = crossing_at(a), crossing_at(b)
    if not (fa < 0.0 < fb):
        raise ValueError(f"端点假设不成立：max Re λ = {fa:.3e}（lo）与 {fb:.3e}（hi）")
    for _ in range(int(bisect)):
        mid = 0.5 * (a + b)
        fm = crossing_at(mid)
        if fm < 0.0:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)
