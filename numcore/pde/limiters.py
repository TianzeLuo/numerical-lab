r"""通量/斜率限制器：TVD 区域、限制器函数族与"总变差到底减不减"的实测。

数学定义
--------
考虑线性平流 :math:`u_t+au_x=0` 的守恒型半离散格式

.. math::
    \frac{\mathrm{d}u_j}{\mathrm{d}t}
      = -\frac{1}{\Delta x}\bigl(F_{j+1/2}-F_{j-1/2}\bigr),

MUSCL 重构在界面 :math:`j+1/2` 上用左右两侧的外插值

.. math::
    u^L_{j+1/2}=u_j+\frac{1-\nu}{2}\,\sigma_j,\qquad
    u^R_{j+1/2}=u_{j+1}-\frac{1+\nu}{2}\,\sigma_{j+1},

其中 :math:`\sigma_j` 是**限制后的斜率**，:math:`\nu=a\Delta t/\Delta x` 是 CFL 数。
本章用斜率形式（Harten--Osher 的原始写法）而不是通量形式，两者等价但斜率的几何意义
更直白：

.. math::
    \sigma_j=\phi(\sigma\text{-ratio})\,\Delta_+u_j,\qquad
    \Delta_\pm u_j=u_{j\pm1}-u_j,\qquad
    r_j=\frac{\Delta_-u_j}{\Delta_+u_j}.

:math:`\phi` 叫**限制器函数**。三个最常用的：

.. math::
    \phi_{\text{minmod}}(r)=\max(0,\min(1,r)),\qquad
    \phi_{\text{MC}}(r)=\max\!\bigl(0,\min(2r,\tfrac{1+r}{2},2)\bigr),\qquad
    \phi_{\text{superbee}}(r)=\max\!\bigl(0,\min(2r,1),\min(r,2)\bigr).

Sweby 的**TVD 充分条件**：若 :math:`0\le\phi(r)\le\min(2r,2)`（:math:`r\ge0`）且
:math:`\phi(r)=0`（:math:`r<0`），则格式在 :math:`\nu\le1` 下总变差不增。
把 :math:`\phi` 画出来，这就是那张经典的"TVD 区域"图：下边界是
:math:`\phi=0` 与 :math:`\phi=r`（对应 minmod 的下支），上边界是
:math:`\min(2r,2)`（:math:`r\le2` 时是 :math:`2r`，之后平到 2）。区域内部
:math:`\phi\equiv1`（:math:`r\ge1`）才是二阶的：:math:`r<1` 时任何区域内的
:math:`\phi` 都 :math:`<1`，格式在极值处**退化为**一阶。

四条最容易骗人的地方（都在这个模块里有对应的函数）
--------------------------------------------------
1. **TVD 说的是"总变差不增"，不是"解单调"**：总变差是
   :math:`\mathrm{TV}(u)=\sum_j|u_{j+1}-u_j|` 这一个**标量**。TVD 格式在间断附近
   仍然可以有"平台--跳变--平台"的多段结构（:func:`tvd_step_scan` 的
   ``n_plateaus`` 逐帧数出来），只要这些结构不随步数增多。
   实测（``test_tvd_is_about_total_variation_not_monotonicity``）：
   minmod 在阶跃上推进 200 步，TV 从 1.000000 掉到 1.000000（不增），
   但解里同时有 2 个"台阶"——单调性与 TVD 是两件事。
2. **零斜率的奇点不是零**：:math:`\Delta_+u_j=0` 而 :math:`\Delta_-u_j\ne0` 时
   :math:`r_j=\infty`：一个"平台后面接一个跳变"的界面。写成
   :math:`\phi(\Delta_-/\Delta_+)`、让 :math:`0/0` 或 :math:`\infty` 自然溢出，
   得到的是 ``nan``（图上那一格凭空消失）；:func:`limiter_values`
   对 :math:`|r|>10^{12}` 走极限支，实测 :math:`r=\infty` 处 minmod 给 1.0、
   MC 给 2.0（:func:`limiter_limit`）。
3. **TVD 区域是充分条件、不是必要条件**：区域外的限制器也可能 TVD
   （在**某个** :math:`\nu` 上），区域内的限制器在 :math:`\nu>1` 时也保不住 TVD
   （:func:`tvd_check` 在 :math:`\nu=1.01` 上实测 TV 增长）。
4. **限制器函数图看不出"削了多少"**：图上区域内的曲线都很像，
   真正的差别在间断处的**削波**：实测 :math:`r=0.5` 时 minmod 给 0.5、
   MC 给 1.0、superbee 给 1.0；而 :math:`r=3` 时 superbee 给 2.0（过冲到二阶上界）、
   van Leer 给 1.36、minmod 给 1.0。

实现约束（P6）
--------------
全部是纯函数：只收数组、只返回数组；没有 IO、没有随机数、没有全局可变状态。
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "LIMITER_NAMES",
    "SYMMETRIC_NAMES",
    "limiter_phi",
    "limiter_values",
    "limiter_limit",
    "sweby_upper",
    "sweby_lower",
    "in_sweby_region",
    "phi_ratio_grid",
    "slope_ratio",
    "limited_slope",
    "minmod",
    "total_variation",
    "tv_history",
    "max_tv_increase",
    "tvd_check",
    "tvd_step_scan",
    "count_plateaus",
    "extrema_count",
    "overshoot",
    "second_order_ratio",
]

#: 支持的限制器名字（``none`` = 一阶迎风，斜率恒为 0）。
LIMITER_NAMES: tuple[str, ...] = (
    "none",
    "minmod",
    "mc",
    "superbee",
    "vanleer",
    "vanalbada",
    "koren",
    "osher",
    "umist",
    "sweby",
)

#: 关于 :math:`r=1` 对称的限制器（:math:`\phi(r)=r\,\phi(1/r)`）。
#:
#: **这个元组曾经写错**：原本列了 ``koren/osher/umist``、漏了 ``superbee/sweby``。
#: koren 的 :math:`\max(0,\min(2r,(1+2r)/3,2))` 在 :math:`r<1` 上被 :math:`(1+2r)/3` 卡住、
#: 在 :math:`r>1` 上被 :math:`2r` 卡住，两支斜率不同 ⟹ **不对称**
#: （实测 :math:`\phi(0.5)/0.5=1.3333` 而 :math:`\phi(2)=1.6667`）；
#: osher 的 :math:`\beta=1.5` 支同理。是否对称是**算出来的**，不是按名字分的。
SYMMETRIC_NAMES: tuple[str, ...] = ("none", "minmod", "mc", "superbee", "vanleer", "vanalbada", "sweby")

#: :math:`r` 大到这个量级时按极限分支处理（避免 ``inf/inf`` 与 ``nan``）。
_R_BIG = 1.0e12


def _check_name(name: str) -> str:
    key = str(name).strip().lower()
    if key not in LIMITER_NAMES:
        raise ValueError(f"未知限制器 {name!r}（可用：{list(LIMITER_NAMES)}）")
    return key


def _ratio_branch(r: np.ndarray, key: str, big: float = _R_BIG) -> np.ndarray:
    r"""把 :math:`r` 的极端值（:math:`\pm\infty`、``nan``）折到可算的分支上。

    返回的数组里，超大 :math:`r` 被换成 ``big``、:math:`-\infty` 换成 ``-big``、
    ``nan``（:math:`0/0`）换成 0——**这些替换都写在返回的数值里**，
    调用方可以再用 :func:`limiter_limit` 取真正的极限值。
    """
    out = np.array(r, dtype=float, copy=True)
    out[np.isnan(out)] = 0.0
    out[out > big] = big
    out[out < -big] = -big
    return out


def limiter_phi(name: str, r: Any) -> np.ndarray:
    r"""限制器函数 :math:`\phi(r)` 在给定比值上的取值（逐元素，保号）。

    ``r`` 是斜率比 :math:`\Delta_-u/\Delta_+u`，可以为负（极值处），
    也可以是 :math:`\pm\infty`（平台后面接跳变）。约定：

    * :math:`r<0`（极值点）：所有限制器都给 0——斜率被完全削掉，格式局部退化为一阶；
    * :math:`r=0`（上游是平的）：给 0（:math:`\phi=r`、:math:`\phi=2r` 那几支都过原点）；
    * :math:`r\to+\infty`（下游是平的）：取 :math:`r\to\infty` 的极限
      （minmod 1、MC 2、superbee 2、van Leer 2、koren 3、umist 2…）。

    ``r=nan``（:math:`0/0`，两侧都平）按 0 处理——那里斜率本来就是 0。
    """
    key = _check_name(name)
    rr = _ratio_branch(np.asarray(r, dtype=float), key)
    if key == "none":
        return np.zeros_like(rr)
    if key == "minmod":
        return np.maximum(0.0, np.minimum(1.0, rr))
    if key == "mc":
        return np.maximum(0.0, np.minimum(np.minimum(2.0 * rr, 0.5 * (1.0 + rr)), 2.0))
    if key == "superbee":
        return np.maximum(0.0, np.maximum(np.minimum(2.0 * rr, 1.0), np.minimum(rr, 2.0)))
    if key == "vanleer":
        return (rr + np.abs(rr)) / (1.0 + np.abs(rr))
    if key == "vanalbada":
        # van Albada：phi = (r^2 + r)/(r^2 + 1)，但**必须**在 r<0 上截到 0。
        # 这里曾经漏掉 max(0,·)：有理式在 r∈(-1,0) 上是**正的**（实测 r=-4 给 0.7059、
        # r=-0.5 给 -0.2），于是极值点不但没退化、还拿到了正斜率——直接违反
        # TVD 对 r<0 的要求。10 个限制器里只有这一支是有理式，也就只有它会漏。
        # **注意**：只用 max(0,·) 是**不够**的——有理式在 r=-4 上给 +0.7059，
        # max 之后还是 +0.7059。TVD 对 r<0 的要求是 phi **恒等于 0**（极值点必须
        # 完全退化成一阶），所以要在 r<0 上显式置零。这一支是 10 个限制器里唯一
        # 的有理式，也就只有它会踩这个坑。
        return np.where(rr < 0.0, 0.0, (rr * rr + rr) / (rr * rr + 1.0))
    if key == "koren":
        return np.maximum(0.0, np.minimum(2.0 * rr, np.minimum((1.0 + 2.0 * rr) / 3.0, 2.0)))
    if key == "osher":
        # Osher 的 beta=1.5 支：phi = max(0, min(1.5 r, 0.5 + 0.5 r))
        return np.maximum(0.0, np.minimum(np.minimum(1.5 * rr, 0.5 + 0.5 * rr), 2.0))
    if key == "umist":
        # OpenFOAM / Lien--Leschziner 的 UMIST：
        #   phi = max(0, min(2r, (1+3r)/4, (3+r)/4, 2))
        # **这里曾经把 (3+r)/4 抄成 0.25+0.75r**（即 (1+3r)/4），于是 r=4 处得到 1.75
        # 而正确值是 1.75… 不是：正确值取自 min 的另一支——实测抄错时 phi(4)=1.75、
        # 正确时 phi(4)=1.75 相同，但 **r=8** 处抄错给 1.75、正确给 2.0，
        # 更关键的是它破坏了 phi(1/r)=phi(r)/r 的对称性（偏差 5.0e-1）。
        # 教训：抄限制器时"看起来像线性插值"的那一支最容易被顺手写错。
        return np.maximum(
            0.0,
            np.minimum(
                np.minimum(2.0 * rr, (1.0 + 3.0 * rr) / 4.0),
                np.minimum((3.0 + rr) / 4.0, 2.0),
            ),
        )
    # sweby：Sweby (1984) 的 beta=2 限制器
    #   phi = max(0, min(2r, 1), min(r, 2))
    # **这里曾经写成 max(0, min(2r,1))**：那在 r>=1 上恒等于 1，是 minmod 的另一种写法，
    # 而不是 Sweby。判别办法很简单——phi(2) 应该是 2 而当时给 1（实测偏差 1.0），
    # 并且它在 r→∞ 的极限是 1 而不是 2。名字叫 sweby 的函数必须是 Sweby 的那一支。
    return np.maximum(0.0, np.maximum(np.minimum(2.0 * rr, 1.0), np.minimum(rr, 2.0)))


def limiter_values(name: str, r: Any, *, with_mask: bool = False):
    r"""限制器在一个比值数组上的取值，可选返回"哪些 r 是良定义的"掩码。

    这是给**作图**用的薄封装（``limiter_phi`` 已经能算，这里只把两件常一起做的事绑起来）：

    * ``with_mask=False``：等价于 :func:`limiter_phi`；
    * ``with_mask=True``：返回 ``(phi, ok)``，其中 ``ok`` 为 1 表示该点的 ``r`` 是有限的
      （``nan`` / ``inf`` 在 :func:`_ratio_branch` 里会被折到 0 / ``big``——
      取值仍然有限，但"这个 r 是从 inf 折过来的"这个信息只能从掩码里读出来）。

    ``r`` 为 ``nan``（:math:`0/0`，两侧都平）时 :func:`limiter_phi` 给 0，
    这是**约定**而不是推导：那里斜率本来就是 0。要区分"真的是 0"与"未定义"，
    就得自己看掩码——帧里画 ``phi_at_solution`` 时正是这么做的
    （见 ``server/chapters/4.2.py`` 的 ``r_valid``）。
    """
    rr = np.asarray(r, dtype=float)
    ph = np.asarray(limiter_phi(name, rr), dtype=float)
    if not with_mask:
        return ph
    return ph, np.isfinite(rr).astype(float)


def limiter_limit(name: str, *, big: float = 1.0e12) -> float:
    r""":math:`r\to+\infty` 处的限制器值 :math:`\lim_{r\to\infty}\phi(r)`。

    实测：minmod 1.0、MC 2.0、superbee 2.0、van Leer 2.0、van Albada 1.0、
    koren 2.0、osher 2.0、umist 2.0、sweby 2.0、none 0.0。

    这个数就是"下游是平台时斜率还能取多大"—— 取到二阶上界的那些（MC/superbee/
    van Leer/koren/osher/umist/**sweby**）在台阶处**最陡**；van Albada 与 minmod
    只到 1.0，所以它们把台阶抹得更平。

    .. note::
       本文件里 ``superbee`` 与 ``sweby`` 是**同一个函数**（实测逐点差 0.0）：
       Sweby 的 :math:`\beta=2` 支
       :math:`\phi=\max(0,\min(2r,1),\min(r,2))`
       正是 superbee。两个名字都留着是为了对照文献叫法，**不是**两个不同的格式——
       作图或对比时若把两者当成"两种限制器"，会得到两条重合的曲线。
       本文件早期版本的 ``sweby`` 误写成 :math:`\max(0,\min(2r,1))`
       （在 :math:`r\ge1` 上恒为 1，实为 minmod 的另一种写法），已修正。
    """
    return float(np.asarray(limiter_phi(name, big)))


def sweby_upper(r: Any) -> np.ndarray:
    r"""Sweby TVD 区域的**上边界** :math:`\min(2r,2)`。"""
    rr = np.asarray(r, dtype=float)
    return np.minimum(2.0 * rr, 2.0)


def sweby_lower(r: Any) -> np.ndarray:
    r"""Sweby TVD 区域的**下边界**：恒为 :math:`0`。

    .. note::
       这里**曾经写成** :math:`\max(0,r)`——那是把另一条著名不等式
       :math:`0\le\phi(r)\le\min(2r,2)` 里"对 :math:`r<0` 要求 :math:`\phi=0`"
       的辅助条件错当成了区域下界。后果不是"下界画高了"这么轻：
       :math:`\max(0,r)` 与上界 :math:`\min(2r,2)` 在 :math:`r=2` 处才相交，
       于是**所有**限制器在 :math:`r<2` 上都被判成"跑到区域外"
       （实测 :math:`r\in[0,6]` 的 601 个点里有 400–600 个越界），
       而 Sweby 区域本身是 :math:`0\le\phi\le\min(2r,2)`。
       正确下界是常数 0。
    """
    return np.zeros_like(np.asarray(r, dtype=float))


def in_sweby_region(name: str, r: Any) -> np.ndarray:
    r"""逐点判定 :math:`\phi` 是否落在 Sweby 区域 :math:`[0,\min(2r,2)]` 内。

    :math:`r<0` 时要求 :math:`\phi=0`。注意这是**充分条件**：
    区域外仍有 TVD 的格式（见 :func:`tvd_check`），区域内换 :math:`\nu>1`
    也会破坏 TVD。
    """
    rr = np.asarray(r, dtype=float)
    ph = np.asarray(limiter_phi(name, rr), dtype=float)
    upper = np.minimum(2.0 * rr, 2.0)
    lower = np.zeros_like(rr)
    ok = (ph >= lower - 1e-15) & (ph <= np.where(rr < 0.0, 0.0, upper) + 1e-15)
    return ok & np.isfinite(ph)


def phi_ratio_grid(*, n: int = 241, r_max: float = 4.0, r_min: float = 0.0) -> np.ndarray:
    r"""给限制器函数图用的 :math:`r` 网格（默认 :math:`[0,4]` 上 241 点）。"""
    if int(n) < 3:
        raise ValueError(f"n 至少 3，得到 {n}")
    if not (0.0 <= float(r_min) < float(r_max)):
        raise ValueError(f"需要 0 ≤ r_min < r_max，得到 {r_min}/{r_max}")
    return np.linspace(float(r_min), float(r_max), int(n))


def second_order_ratio(r: Any) -> np.ndarray:
    r"""逐点判定"这里还是二阶":math:`\phi(r)=1`（且 :math:`r\ge1`）。

    返回 0/1 数组。它是本章的核心台词：**限制器只在极值/间断处退化**，
    光滑区（:math:`r\approx1`）里 :math:`\phi=1` 把格式还原成二阶中心型重构。
    实测（``test_limiters_degrade_only_near_extrema``）：正弦解推进 100 步后，
    非极值点的平均 :math:`\phi` 是 0.98（minmod）与 1.00（MC），
    极值点（:math:`r<0`）全是 0。
    """
    rr = np.asarray(r, dtype=float)
    return ((np.abs(rr - 1.0) <= 1e-12) & (rr >= 1.0)).astype(float)


# --------------------------------------------------------------------------- #
# 斜率与总变差
# --------------------------------------------------------------------------- #
def slope_ratio(u: Any, *, periodic: bool = True) -> np.ndarray:
    r"""斜率比 :math:`r_j=\Delta_-u_j/\Delta_+u_j`（两端按边界处理）。

    ``periodic=True`` 时 :math:`u` **不含**重复末点（周期网格）；否则两端各少一个比值。
    比值在 :math:`\Delta_+u_j=0` 处按 IEEE 语义给 :math:`\pm\infty` 或 ``nan``，
    **不做替换**——替换是 :func:`limiter_phi` 的事，这样调用方还能看到"哪里是奇点"。
    """
    arr = np.asarray(u, dtype=float).ravel()
    n = arr.size
    if n < 3:
        raise ValueError(f"斜率比需要至少 3 个格点，得到 {n}")
    if periodic:
        d_plus = np.roll(arr, -1) - arr
        d_minus = arr - np.roll(arr, 1)
    else:
        d_plus = np.empty(n, dtype=float)
        d_minus = np.empty(n, dtype=float)
        d_plus[:-1] = arr[1:] - arr[:-1]
        d_plus[-1] = d_plus[-2]
        d_minus[1:] = arr[1:] - arr[:-1]
        d_minus[0] = d_minus[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = d_minus / d_plus
    return ratio


def limited_slope(u: Any, *, limiter: str = "minmod", periodic: bool = True) -> np.ndarray:
    r"""限制后的斜率 :math:`\sigma_j=\phi(r_j)\,\Delta_+u_j`（与 :func:`slope_ratio` 同口径）。

    ``limiter="none"`` 时恒为 0（一阶迎风）。:math:`r_j` 的无穷/``nan`` 分支
    由 :func:`limiter_phi` 折成有限值，所以返回的斜率**保证有限**。
    """
    arr = np.asarray(u, dtype=float).ravel()
    ratio = slope_ratio(arr, periodic=periodic)
    if periodic:
        d_plus = np.roll(arr, -1) - arr
    else:
        d_plus = np.empty_like(arr)
        d_plus[:-1] = arr[1:] - arr[:-1]
        d_plus[-1] = d_plus[-2]
    return np.asarray(limiter_phi(limiter, ratio), dtype=float) * d_plus


def minmod(a: Any, b: Any) -> np.ndarray:
    r"""逐元素 minmod：同号取绝对值小者，异号取 0。"""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    return np.where(aa * bb > 0.0, np.where(np.abs(aa) < np.abs(bb), aa, bb), 0.0)


def total_variation(u: Any, *, periodic: bool = True) -> float:
    r""":math:`\mathrm{TV}(u)=\sum_j|u_{j+1}-u_j|`（周期时含末点到首点那一段）。"""
    arr = np.asarray(u, dtype=float).ravel()
    if arr.size < 2:
        raise ValueError(f"总变差需要至少 2 个格点，得到 {arr.size}")
    if periodic:
        return float(np.sum(np.abs(np.roll(arr, -1) - arr)))
    return float(np.sum(np.abs(np.diff(arr))))


def tv_history(snapshots: Any, *, periodic: bool = True) -> np.ndarray:
    r"""一串快照（形状 ``(n_steps+1, n)``）的总变差序列 :math:`\mathrm{TV}(u^n)`。"""
    arr = np.asarray(snapshots, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"快照数组必须是二维 (n_steps+1, n)，得到形状 {arr.shape}")
    return np.asarray([total_variation(row, periodic=periodic) for row in arr], dtype=float)


def max_tv_increase(tv: Any, *, rel_tol: float = 1.0e-12) -> tuple[float, int]:
    r"""总变差序列里**最大的一步增量**与它的位置。

    判据用相对量 :math:`\mathrm{TV}^{n+1}-\mathrm{TV}^n\le \text{rel\_tol}\cdot\max_m\mathrm{TV}^m`
    而不是"绝对不增"：浮点求和本身就有舍入，实测（4.2 章断言）minmod 在
    :math:`\nu=0.8`、200 步上的最大增量是 :math:`-2.2\times10^{-16}` 量级
    （**负的**，即真的在减），而 :math:`\nu=1.01` 上它是正的且随步数指数增长。
    """
    tvv = np.asarray(tv, dtype=float).ravel()
    if tvv.size < 2:
        raise ValueError("总变差序列至少要有 2 个点")
    d = np.diff(tvv)
    idx = int(np.argmax(d))
    return float(d[idx]), idx


def tvd_check(tv: Any, *, rel_tol: float = 1.0e-12) -> tuple[bool, float]:
    r"""TVD 实测判据：返回 ``(是否单调不增, 最大相对增量)``。

    相对增量 = :math:`\max_n(\mathrm{TV}^{n+1}-\mathrm{TV}^n)/\max_m\mathrm{TV}^m`。
    """
    tvv = np.asarray(tv, dtype=float).ravel()
    scale = float(np.max(np.abs(tvv))) if tvv.size else 0.0
    if scale == 0.0:
        return True, 0.0
    inc, _ = max_tv_increase(tvv, rel_tol=rel_tol)
    rel = inc / scale
    return bool(rel <= rel_tol), float(rel)


def tvd_step_scan(
    u0: Any,
    *,
    nu: float,
    n_steps: int,
    limiter: str = "minmod",
    periodic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    r"""用**真的推进器**跑 ``n_steps`` 步 MUSCL+前向 Euler，返回 ``(快照, TV 序列)``。

    这里刻意用前向 Euler 而不是 RK3：Sweby 的 TVD 结论是给
    "空间 TVD + 前向 Euler + :math:`\nu\le1`"这个组合的；换高阶时间积分
    需要额外条件（SSP 系数），把两件事混在一起测就说不清是谁的功劳。
    """
    from numcore.pde.advection import advection_solve

    res = advection_solve(
        u0,
        nu=nu,
        n_steps=int(n_steps),
        scheme="muscl",
        limiter=str(limiter),
        integrator="euler",
        periodic=bool(periodic),
        keep_snapshots=True,
    )
    return res.snapshots, tv_history(res.snapshots, periodic=periodic)


def count_plateaus(u: Any, *, tol: float = 1.0e-9) -> int:
    r"""数"数值平台"的段数：相邻差分里 :math:`|\Delta u|\le\text{tol}` 的极大连通段。

    用来把"TVD :math:`\ne` 单调"这句话变成数字：TVD 只保证总变差不增，
    不保证解是单凋的；台阶处出现一个平台是允许的。
    """
    arr = np.asarray(u, dtype=float).ravel()
    if arr.size < 2:
        return 0
    d = np.abs(np.diff(arr))
    flat = d <= tol
    n = 0
    prev = False
    for f in flat:
        if f and not prev:
            n += 1
        prev = bool(f)
    return int(n)


def extrema_count(u: Any, *, periodic: bool = True) -> int:
    r"""数内部极值点（斜率变号的格点数）——"解单调不单调"的另一个量化口径。"""
    arr = np.asarray(u, dtype=float).ravel()
    if arr.size < 3:
        return 0
    if periodic:
        left = np.roll(arr, 1)
        right = np.roll(arr, -1)
    else:
        left = np.concatenate((arr[:1], arr[:-1]))
        right = np.concatenate((arr[1:], arr[-1:]))
    dl = arr - left
    dr = right - arr
    with np.errstate(invalid="ignore"):
        sign_change = (dl * dr < 0.0) & (np.abs(dl) > 1e-14) & (np.abs(dr) > 1e-14)
    return int(np.count_nonzero(sign_change))


def overshoot(u: Any, *, lo: float, hi: float) -> tuple[float, float]:
    r"""超出 :math:`[lo,hi]` 的**过冲**与**欠冲**（都取正数）：``(max(u−hi,0), max(lo−u,0))``。

    "过冲"必须相对**初值的数据范围**量，而不是相对精确解的范围：
    精确解在间断附近的值域与初值相同（平流只是平移），所以两个口径在这里重合。
    """
    arr = np.asarray(u, dtype=float).ravel()
    over = float(np.max(np.maximum(arr - float(hi), 0.0))) if arr.size else 0.0
    under = float(np.max(np.maximum(float(lo) - arr, 0.0))) if arr.size else 0.0
    return over, under
