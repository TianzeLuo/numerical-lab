r"""2.4 章 —— Gauss--Legendre 求积：节点是 Legendre 零点，代数精度恰好 :math:`2n-1`。

:math:`n` 点 Gauss--Legendre 公式

.. math::
    \int_{-1}^{1}f(x)\,dx\approx\sum_{k=1}^{n}w_kf(x_k),\qquad
    x_k=P_n\ \text{的零点},\qquad w_k=\frac{2}{(1-x_k^2)[P_n'(x_k)]^2}

有两条**互相独立**的性质，本章两条都要用实测钉住：

1. **代数精度恰好是** :math:`2n-1`：对 :math:`k\le2n-1` 的单项式误差在舍入水平，
   而 :math:`k=2n` **必须当场失效**（:math:`\int x^{2n}` 与求积值的差是 O(1) 量级）。
   只测"很准"是没有上界的断言——2.3 章已经把这条写成"双向证据"的样板；
2. **权重全为正**、:math:`\sum w_k=b-a`，所以条件数 :math:`\Gamma=1`：
   函数值的相对误差**不会**被放大（:mod:`numcore.quad.condition`）。

节点与权重**不在这里重新实现**：它们来自 2.3 章的
:func:`numcore.interp.orthopoly.gauss_legendre`（Newton 求根 + 手写递推导数）。
2.3 章已经把"节点是 Legendre 零点、权重闭式、权重和"三条都断言过；
本模块只负责"把它当求积公式用"以及**复合**（分面板）版本。

复合 Gauss 的用处是把 Gauss 与等距规则放到**同一次函数求值的预算**下比较：
每面板 :math:`m` 点、共 :math:`p` 个面板时误差 :math:`O(h^{2m})`，
而复合 Simpson 用 3 倍的点只能给 :math:`O(h^4)`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from numcore.interp.orthopoly import gauss_legendre

__all__ = [
    "GaussRule",
    "composite_gauss",
    "exact_monomial",
    "gauss_legendre_rule",
    "gauss_rule_condition",
    "moment_errors",
    "moment_failure_factor",
    "exactness_limit",
]


@dataclass(frozen=True)
class GaussRule:
    r""":math:`n` 点 Gauss--Legendre 公式（映射到 :math:`[a,b]` 之后）。"""

    n: int
    a: float
    b: float
    nodes: np.ndarray
    weights: np.ndarray

    @property
    def degree(self) -> int:
        r"""代数精度 :math:`2n-1`。"""
        return 2 * self.n - 1

    @property
    def sum_weights(self) -> float:
        return float(np.sum(self.weights))

    def integrate_values(self, values: Any) -> float:
        """按权重求和 :math:`\\sum w_kf(x_k)`（调用方负责给对节点上的值）。"""
        v = np.asarray(values, dtype=float).ravel()
        if v.size != self.weights.size:
            raise ValueError(f"函数值个数 {v.size} 与节点数 {self.weights.size} 不一致")
        return float(np.sum(self.weights * v))


def gauss_legendre_rule(n: int, a: float = -1.0, b: float = 1.0) -> GaussRule:
    r""":math:`n` 点 Gauss--Legendre 公式，仿射映射到 :math:`[a,b]`。

    节点/权重取自 2.3 章的 :func:`numcore.interp.orthopoly.gauss_legendre`
    （那里用手写 Newton 求 :math:`P_n` 的零点、用递推恒等式给导数）。
    映射 :math:`x=\frac{a+b}{2}+\frac{b-a}{2}\xi` 把权重乘上 :math:`\frac{b-a}{2}`。
    """
    nn = int(n)
    if nn < 1:
        raise ValueError(f"n 至少为 1，得到 {n!r}")
    xi, wi = gauss_legendre(nn)
    half = 0.5 * (float(b) - float(a))
    mid = 0.5 * (float(a) + float(b))
    return GaussRule(
        n=nn,
        a=float(a),
        b=float(b),
        nodes=mid + half * np.asarray(xi, dtype=float),
        weights=half * np.asarray(wi, dtype=float),
    )


def composite_gauss(
    f: Callable[[Any], Any],
    a: float,
    b: float,
    panels: int,
    *,
    nodes_per_panel: int = 5,
) -> float:
    r"""复合 Gauss：把 :math:`[a,b]` 等分成 ``panels`` 个面板，每面板一个 :math:`m` 点 Gauss 公式。

    面板宽 :math:`h=(b-a)/p` 时误差 :math:`O(h^{2m})`——**没有** :math:`f'` 端点项，
    所以它比同阶的等距公式更"抗端点"；但面板数必须随振荡尺度增长，
    这是振荡被积函数那一节要付出的代价。
    """
    p = int(panels)
    if p < 1:
        raise ValueError(f"panels 至少为 1，得到 {panels!r}")
    m = int(nodes_per_panel)
    edges = np.linspace(float(a), float(b), p + 1)
    total = 0.0
    for i in range(p):
        rule = gauss_legendre_rule(m, float(edges[i]), float(edges[i + 1]))
        total += rule.integrate_values(f(rule.nodes))
    return float(total)


def exact_monomial(a: float, b: float, k: int) -> float:
    r""":math:`\int_a^b x^k\,dx=\frac{b^{k+1}-a^{k+1}}{k+1}` 的闭式（k ≥ 0）。"""
    kk = int(k)
    if kk < 0:
        raise ValueError(f"幂次必须 ≥ 0，得到 {k!r}")
    return (float(b) ** (kk + 1) - float(a) ** (kk + 1)) / (kk + 1)


def exactness_limit(n: int) -> int:
    r"""Gauss 公式的代数精度界 :math:`2n-1`（第 :math:`2n` 阶矩必须失效）。"""
    nn = int(n)
    if nn < 1:
        raise ValueError(f"n 至少为 1，得到 {n!r}")
    return 2 * nn - 1


def moment_errors(n: int, *, k_max: int | None = None, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    r"""逐阶矩的**实测**误差 :math:`\big|\sum_kw_kx_k^{j}-\int_a^bx^j\big|`，``j=0..k_max``。

    缺省 ``k_max = 2n``：前 :math:`2n` 个（下标 0..2n−1）应当是舍入级，
    第 :math:`2n` 个必须掉下来——两个方向都要测，缺一条就只剩"精度很好"这种没有上界的说法。
    """
    nn = int(n)
    rule = gauss_legendre_rule(nn, a, b)
    top = 2 * nn if k_max is None else int(k_max)
    if top < 0:
        raise ValueError(f"k_max 必须 ≥ 0，得到 {k_max!r}")
    out = np.empty(top + 1, dtype=float)
    for j in range(top + 1):
        got = rule.integrate_values(rule.nodes**j)
        out[j] = abs(got - exact_monomial(a, b, j))
    return out


def moment_failure_factor(n: int, *, a: float = -1.0, b: float = 1.0) -> float:
    r"""第 :math:`2n` 阶矩的失效倍数：:math:`\text{err}_{2n}/\max(\text{err}_{\le2n-1},\ u\cdot I)`。

    分母用**实测**的精确段误差（不是 0），否则这个比值会是 inf。
    实测（区间 :math:`[-1,1]`，:math:`n=2..12`）：分子在 :math:`10^{-1}` 量级、
    分母在 :math:`10^{-16}` 量级，比值 :math:`\ge10^{15}`——这才是"恰好 :math:`2n-1`"这句话的证据。
    """
    errs = moment_errors(n, k_max=2 * int(n), a=a, b=b)
    num = float(errs[-1])
    den = float(np.max(errs[:-1])) if errs.size > 1 else 0.0
    floor = float(2.0**-53 * abs(exact_monomial(a, b, 2 * int(n))))
    den = max(den, floor)
    return float(num / den)


def gauss_rule_condition(n: int, a: float = -1.0, b: float = 1.0) -> float:
    r"""Gauss 公式的条件数 :math:`\Gamma=\sum|w|/|\sum w|`（实测恒为 1，机器精度）。"""
    rule = gauss_legendre_rule(n, a, b)
    with np.errstate(invalid="ignore"):
        return float(np.sum(np.abs(rule.weights)) / abs(np.sum(rule.weights)))
