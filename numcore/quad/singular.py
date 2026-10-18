r"""2.4 章 —— 奇异与振荡被积函数：双指数（tanh--sinh）变换、幂换元、分段。

端点奇异性是"求积的阶"这句话失效最彻底的地方。三种处理方式，本章都要用实测数字对比：

1. **什么都不做**：:math:`(1+x)/\sqrt x` 在 :math:`x=0` 处是 ``inf``，等距规则**没有定义**
   （不是"误差大"，是公式里出现了 inf）；
2. **换元** :math:`x=t^p`：:math:`\int_0^1\frac{1+x}{\sqrt x}dx=\int_0^1(2+2t^2)dt`，
   被积函数变成多项式，等距规则**重新可用**——但代价是误差常数变了，
   而且你得先知道奇异指数 :math:`\alpha` 才能选 :math:`p`；
3. **双指数变换**（Takahashi--Mori）：:math:`x=\tanh\big(\frac\pi2\sinh t\big)`，

   .. math::
       w(t)=\frac{\frac\pi2\cosh t}{\cosh^2\big(\frac\pi2\sinh t\big)},

   节点在 :math:`t=\pm mh` 上取，:math:`h` 固定、:math:`m` 增大。它对**任意**
   :math:`\alpha>-1` 的端点奇异都收敛，而且收敛是**指数**的（半对数图上是直线），
   不需要知道 :math:`\alpha`。代价是双指数衰减的节点位置要算 :math:`\cosh` 两次。

两条必须如实标注的边界：

* :math:`h` **固定**时，误差不会无限下降：离散化误差停在地板
  :math:`\sim e^{-\pi^2/(2h)}` 附近，再增大 :math:`m` 只是让求值次数变多；
* :math:`t` 大到 :math:`\tanh(\cdot)` 在 double 里**舍入到 ±1** 时，那些节点的权重也下溢到 0。
  实测（:math:`h=2^{-4}`）：:math:`1-|x|<u` 从第 101 个节点（:math:`\pm mh\approx3.16`）开始出现。
  把这些点去掉、并把个数写进帧的 ``meta``/``strings``，才是"如实标注"；
  悄悄保留会让 :math:`f(\pm1)=\infty` 与 0 权重相乘得到 NaN。

振荡被积函数走的是另一条路：:math:`\sin(\omega x)` 光滑、阶没有被破坏，
被破坏的是**常数**（:math:`|f^{(p)}|\le\omega^p`）。全局规则要在
:math:`\omega` 增大时保持精度，节点数必须随 :math:`\omega` 增长；
分段（每面板不超过若干分之一周期）是唯一诚实的做法，:func:`oscillation_panels`
给的就是这个"必须付的成本"。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


__all__ = [
    "DoubleExponentialRule",
    "oscillation_panels",
    "panelled_gauss_for_oscillation",
    "saturation_abscissa",
    "tanh_sinh",
    "tanh_sinh_convergence",
    "tanh_sinh_rule",
]

#: 默认网格步长：:math:`2^{-4}`。理论上的离散化地板 :math:`e^{-\pi^2/(2h)}\approx e^{-78.9}`
#: 远在 double 精度之下，所以双精度下 :math:`h` 取这个量级是够的。
DEFAULT_H: float = 2.0**-4


@dataclass(frozen=True)
class DoubleExponentialRule:
    r"""一次 tanh--sinh 求积的节点、权重与"饱和"诊断。"""

    nodes: np.ndarray
    weights: np.ndarray
    h: float
    level: int
    saturated: np.ndarray
    t_max: float

    @property
    def n_saturated(self) -> int:
        """权重下溢/节点舍入到端点的个数（这些节点的贡献必须按 0 计，不能留 inf·0）。"""
        return int(np.count_nonzero(self.saturated))

    def integrate_values(self, values: Any) -> float:
        r"""按权重求和，并把**饱和节点**上的函数值先置 0。

        这一步不是洁癖：饱和节点的 :math:`x` 恰好是端点，奇异被积函数在那里是 ``inf``，
        而权重是 0——``0 * inf = nan`` 会让整帧炸掉。置 0 是数学上正确的处理
        （那些节点的权重已经下溢到 0），必须**先**置 0 再相乘，不能靠 ``np.where`` 事后挑选。
        """
        v = np.asarray(values, dtype=float).ravel()
        if v.size != self.weights.size:
            raise ValueError(f"函数值个数 {v.size} 与节点数 {self.weights.size} 不一致")
        safe = np.where(self.saturated, 0.0, v)
        return float(np.sum(self.weights * safe))

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.size)

    @property
    def sum_weights(self) -> float:
        return float(np.sum(self.weights))


def saturation_abscissa(*, u: float = 2.0**-53) -> float:
    r"""使 :math:`1-\tanh(z)<u` 的临界 :math:`z^\*`（解析解 :math:`z^\*=\mathrm{artanh}(1-u)`）。

    节点用 :math:`z=\frac\pi2\sinh t`，于是 :math:`t^\*=\mathrm{asinh}(2z^\*/\pi)`。
    超过它的节点在 double 里都映到同一个 :math:`x=\pm1`，权重下溢到 0——
    它们对积分没有任何贡献，只会把 :math:`f` 在端点的 ``inf`` 引进求和。
    """
    z = math.atanh(1.0 - float(u))
    return math.asinh(2.0 * z / math.pi)


def tanh_sinh_rule(
    level: int,
    *,
    h: float = DEFAULT_H,
    a: float = -1.0,
    b: float = 1.0,
) -> DoubleExponentialRule:
    r"""双指数变换的节点/权重：``level`` 是**每侧**的节点数，总节点数 :math:`2\,\text{level}+1`。

    .. math::
        x_k=\tanh\left(\frac\pi2\sinh t_k\right),\qquad
        w_k=h\,\frac{\frac\pi2\cosh t_k}{\cosh^2\left(\frac\pi2\sinh t_k\right)},\qquad
        t_k=kh,\ k=-m,\dots,m,

    再仿射映射到 :math:`[a,b]`（权重乘 :math:`\frac{b-a}{2}`）。
    权重里的因子 :math:`h` 不能漏：:math:`t` 方向用的是步长 :math:`h` 的梯形和
    :math:`h\sum_kw(t_k)f(x_k)`，漏掉它会让 :math:`\sum w_k` 变成 :math:`2/h` 而不是 2
    （实测：m=32、h=1/16 时漏掉 h 得 :math:`\sum w=31.9995`，补上就是
    :math:`31.9995/16=1.99997`）——这是本章实现期真踩过的一个 bug，靠"权重和"这条不变量抓出来的。
    ``|x|=1`` 的节点权重记 0 并计入 ``n_saturated``：那是"这个 :math:`h` 下再加点也没用"的实测标志。
    """
    m = int(level)
    if m < 1:
        raise ValueError(f"level 至少为 1，得到 {level!r}")
    hh = float(h)
    if not math.isfinite(hh) or hh <= 0.0:
        raise ValueError(f"h 必须是正有限数，得到 {h!r}")
    k = np.arange(-m, m + 1, dtype=float)
    t = k * hh
    half_pi = 0.5 * math.pi
    z = half_pi * np.sinh(t)
    with np.errstate(over="ignore", invalid="ignore"):
        xt = np.tanh(z)
        w = hh * half_pi * np.cosh(t) / np.cosh(z) ** 2
    # 饱和判据必须落在**变换后的规范区间 [−1,1]** 上：仿射映射到 [a,b] 之后
    # |x|>=1 这个条件就永远不成立了（[0,1] 上的节点全都 <=1），是实测抓出来的错。
    saturated = ~np.isfinite(w) | (np.abs(xt) >= 1.0)
    w = np.where(saturated, 0.0, w)
    xt = np.where(saturated, np.sign(xt), xt)
    half = 0.5 * (float(b) - float(a))
    mid = 0.5 * (float(a) + float(b))
    return DoubleExponentialRule(
        nodes=mid + half * xt,
        weights=half * w,
        h=hh,
        level=m,
        saturated=saturated,
        t_max=float(np.max(np.abs(t))),
    )


def tanh_sinh(
    f: Callable[[Any], Any],
    a: float,
    b: float,
    level: int,
    *,
    h: float = DEFAULT_H,
) -> float:
    r"""tanh--sinh 求积值 :math:`\sum_kw_kf(x_k)`（每侧 ``level`` 个节点）。"""
    rule = tanh_sinh_rule(level, h=h, a=a, b=b)
    return rule.integrate_values(f(rule.nodes))


def tanh_sinh_convergence(
    f: Callable[[Any], Any],
    a: float,
    b: float,
    levels: Any,
    *,
    exact: float,
    h: float = DEFAULT_H,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""tanh--sinh 的收敛过程：返回 ``(总节点数, 误差, 饱和节点数)``。

    "指数收敛"的判据是**半对数**图上的直线（:math:`\log_{10}\text{err}` 对节点数线性），
    不是双对数图上的直线——后者会给出一个没有含义的幂次。
    """
    ls = np.asarray(levels, dtype=int).ravel()
    errs = np.empty(ls.size, dtype=float)
    nodes = np.empty(ls.size, dtype=float)
    sat = np.empty(ls.size, dtype=float)
    for i, m in enumerate(ls):
        rule = tanh_sinh_rule(int(m), h=h, a=a, b=b)
        val = rule.integrate_values(f(rule.nodes))
        errs[i] = abs(val - float(exact))
        nodes[i] = float(rule.n_nodes)
        sat[i] = float(rule.n_saturated)
    return nodes, errs, sat


def oscillation_panels(
    omega: float,
    length: float = 1.0,
    *,
    periods_per_panel: float = 4.0,
) -> int:
    r"""让每个面板覆盖不超过 ``1/periods_per_panel`` 个周期（相位跨度 :math:`\le2\pi/`份额）。

    周期数是 :math:`\omega\cdot\text{length}/(2\pi)`，于是所需面板数
    :math:`\lceil \text{periods}\cdot\text{periods\_per\_panel}\rceil`。
    :math:`\omega` 翻倍、面板数翻倍——**这就是振荡问题的真实成本**，
    任何声称"节点数与 :math:`\omega` 无关"的方法都必须在别处付账（例如 Filon 的矩）。
    """
    w = float(omega)
    if not math.isfinite(w) or w <= 0.0:
        raise ValueError(f"omega 必须是正有限数，得到 {omega!r}")
    periods = w * abs(float(length)) / (2.0 * math.pi)
    return max(1, int(math.ceil(periods * float(periods_per_panel))))


def panelled_gauss_for_oscillation(
    f: Callable[[Any], Any],
    a: float,
    b: float,
    omega: float,
    *,
    nodes_per_panel: int = 5,
    periods_per_panel: float = 4.0,
) -> tuple[float, int]:
    r"""按 :func:`oscillation_panels` 分段做复合 Gauss，返回 ``(值, 面板数)``。

    代价被显式返回：调用方可以把它画进"求值次数 vs :math:`\omega`"里，
    而不是让读者以为分段是免费的。
    """
    from numcore.quad.gauss import composite_gauss

    panels = oscillation_panels(omega, abs(float(b) - float(a)), periods_per_panel=periods_per_panel)
    return composite_gauss(f, a, b, panels, nodes_per_panel=int(nodes_per_panel)), panels
