r"""2.4 章 —— Newton--Cotes 族：复合梯形 / Simpson / Simpson-3/8 与误差主项。

三个公式的节点都是**等距**的，于是它们的区别只剩权重。统一记子区间数为 :math:`n`、
步长 :math:`h=(b-a)/n`：

.. math::
    T_n &= h\left[\frac{f_0}{2}+f_1+\cdots+f_{n-1}+\frac{f_n}{2}\right],\\
    S_n &= \frac{h}{3}\left[f_0+4f_1+2f_2+4f_3+\cdots+4f_{n-1}+f_n\right]\quad(n\ \text{偶}),\\
    S^{(3/8)}_n &= \frac{3h}{8}\left[f_0+3f_1+3f_2+2f_3+\cdots+3f_{n-1}+f_n\right]\quad(3\mid n).

误差主项写成同一种形状（:math:`\xi\in(a,b)` 未知，但**常数与幂次是确定的**）：

.. math::
    E = C\,(b-a)\,h^{p}\,f^{(p)}(\xi),\qquad
    C_{\text{trap}}=-\tfrac1{12},\quad C_{\text{simp}}=-\tfrac1{180},\quad
    C_{3/8}=-\tfrac1{80},

其中 :math:`p=2`（梯形）或 :math:`p=4`（两种 Simpson）。这个形状有一个直接后果：
只要 :math:`f^{(p)}` 在区间上有上下界，:math:`E` 就被**夹在闭区间里**——
:func:`error_bracket` 给的就是这条双侧不等式，它比"误差是 O(h^p)"强得多
（后者对常数一无所知，也就不可能被证伪）。

等距带来的代价在**权重**上：闭 Newton--Cotes 的权重在 :math:`m=9` 个点处首次变负，
:math:`m\ge 11` 起 :math:`\sum|w_k|/\sum w_k` 随 :math:`m` 迅速膨胀
（:math:`m=19` 时实测 ≈175）。求积是加权和，这个比值就是把输入误差放大到结果上的因子，
详见 :mod:`numcore.quad.condition`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from numcore.interp.poly import poly_from_roots

__all__ = [
    "RULES",
    "ErrorLaw",
    "RuleEvaluation",
    "RuleLaw",
    "composite_eval",
    "composite_nodes",
    "composite_rule",
    "composite_weights",
    "error_bracket",
    "error_curve",
    "effective_sub_counts",
    "error_law",
    "fitted_constant",
    "newton_cotes_degree",
    "newton_cotes_weights",
    "rule_eval_count",
    "rule_law",
]

#: 本章实现的三个复合等距规则（名字与 manifest 的枚举一一对应）。
RULES: tuple[str, ...] = ("trapezoid", "simpson", "simpson38")


@dataclass(frozen=True)
class RuleLaw:
    """一个复合等距规则的"法律条文"：子区间约束、误差幂次、常数、代数精度。"""

    name: str
    label: str
    sub_multiple: int
    min_sub: int
    single_panel_degree: int
    error_power: int
    error_derivative: int
    #: 误差主项的系数 :math:`C`，约定 :math:`E:=Q-I=C(b-a)h^pf^{(p)}(\xi)`。
    #: 于是**凸**函数的梯形法高估（:math:`E>0`）：实测 ∫₀¹eˣ 用 1 个面板得
    #: Q−I=+0.1409，与 +(1/12)·1·e^ξ 一致（ξ≈0.54）。
    #: 常见教材写成 :math:`I-Q=-(1/12)(b-a)h^2f''(\xi)`，那是**另一个**符号约定；
    #: 本仓一律用 E=Q−I，所以三个常数都是正的（实现期这里踩过坑：
    #: 先写成负号，被"解析区间必须夹住实测误差"的断言当场抓住）。
    error_coefficient: float

    def needed_subintervals(self, n_sub: int) -> int:
        """把请求的子区间数**向上**取到合法值（Simpson 要偶数、3/8 要 3 的倍数）。"""
        n = int(n_sub)
        if n < self.min_sub:
            n = self.min_sub
        k = self.sub_multiple
        if k > 1 and n % k:
            n += k - (n % k)
        return n


_LAWS: dict[str, RuleLaw] = {
    "trapezoid": RuleLaw(
        name="trapezoid",
        label="复合梯形",
        sub_multiple=1,
        min_sub=1,
        single_panel_degree=1,
        error_power=2,
        error_derivative=2,
        error_coefficient=1.0 / 12.0,
    ),
    "simpson": RuleLaw(
        name="simpson",
        label="复合 Simpson（1/3）",
        sub_multiple=2,
        min_sub=2,
        single_panel_degree=3,
        error_power=4,
        error_derivative=4,
        error_coefficient=1.0 / 180.0,
    ),
    "simpson38": RuleLaw(
        name="simpson38",
        label="复合 Simpson-3/8",
        sub_multiple=3,
        min_sub=3,
        single_panel_degree=3,
        error_power=4,
        error_derivative=4,
        error_coefficient=1.0 / 80.0,
    ),
}


def rule_law(rule: str) -> RuleLaw:
    """取规则条文；未知名字直接抛 :class:`KeyError`（不静默退回梯形）。"""
    key = str(rule)
    if key not in _LAWS:
        raise KeyError(f"未知求积规则 {rule!r}（可用：{', '.join(RULES)}）")
    return _LAWS[key]


def check_subintervals(rule: str, n_sub: int) -> int:
    """把子区间数规范化到该规则的合法值（向上取整到 ``sub_multiple`` 的倍数）。"""
    return rule_law(rule).needed_subintervals(n_sub)


# --------------------------------------------------------------------------- #
# 单个闭 Newton–Cotes 公式的权重
# --------------------------------------------------------------------------- #
def newton_cotes_degree(m: int) -> int:
    """``m`` 个等距节点的闭 Newton--Cotes 公式的代数精度：``m-1``。"""
    mm = int(m)
    if mm < 2:
        raise ValueError(f"闭 Newton–Cotes 至少 2 个节点，得到 {m!r}")
    return mm - 1


def newton_cotes_weights(m: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    r"""``m`` 个等距节点的闭 Newton--Cotes 权重 :math:`w_j=\int_a^b L_j`（精确到舍入）。

    权重由 Lagrange 基的**精确积分**给出，而不是解 Vandermonde 方程组：
    :math:`L_j(x)=\prod_{k\ne j}(x-x_k)\big/\prod_{k\ne j}(x_j-x_k)` 的分子是多项式，
    它的积分逐项可算（用 2.1 章的 :func:`numcore.interp.poly.poly_from_roots` 造分子）。

    **先在标准区间上算、再仿射映射**。这不是数学上的改写而是数值上的改写：
    同一个 Lagrange 基展开在 :math:`[0,1]` 上条件极坏，实测 :math:`|\sum w-(b-a)|`
    在 m=19 时是 1.7e-3、m=24 时是 **21**（权重已经是垃圾），
    在 :math:`[-1,1]` 上算完再乘 :math:`(b-a)/2` 之后同一规模只有 6.1e-10 / 3.3e-6。
    这条"用最弱的不变量 :math:`\sum w=b-a` 抓最强的问题"的路子，
    与 2.3 章单项式基 vs Chebyshev 基是同一件事。

    权重**允许为负**（m ≥ 9 时必然出现），这不是 bug：负权重正是等距高阶公式
    条件变坏的根源（见 :mod:`numcore.quad.condition`）。
    """
    mm = int(m)
    aa, bb = float(a), float(b)
    # 标准区间 [-1,1] 上的等距节点：先在这里算权重（数值上远好于直接在任何 [a,b] 上展开）
    x = np.linspace(-1.0, 1.0, mm)
    w = np.zeros(mm, dtype=float)
    for j in range(mm):
        others = np.delete(x, j)
        coef = poly_from_roots(others)  # 降幂：c[0] x^deg + ... + c[deg]
        top = coef.size - 1
        total = 0.0
        for i, c in enumerate(coef):
            power = top - i
            total += float(c) * (1.0 - (-1.0) ** (power + 1)) / (power + 1)
        denom = float(np.prod(x[j] - others))
        w[j] = total / denom
    # 仿射映射 ξ∈[-1,1] → x=(a+b)/2 + (b-a)/2·ξ：权重整体乘 (b-a)/2
    return w * (0.5 * (bb - aa))


# --------------------------------------------------------------------------- #
# 复合规则的节点与权重
# --------------------------------------------------------------------------- #
def composite_nodes(rule: str, n_sub: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """复合规则的节点：``n+1`` 个等距点（三个规则用的是同一批点）。"""
    law = rule_law(rule)
    n = law.needed_subintervals(n_sub)
    return np.linspace(float(a), float(b), n + 1)


def composite_weights(rule: str, n_sub: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    r"""复合规则的**完整**权重向量（长度 ``n+1``，:math:`\sum w_k=b-a`）。

    共享节点按它在每个面板里出现的次数累加（例如复合 Simpson 的内部偶数节点权重是
    :math:`2h/3`）——这正是把公式当"加权和"看待时该有的权重，也是
    :func:`numcore.quad.condition.rule_condition` 要吃的东西。
    """
    law = rule_law(rule)
    n = law.needed_subintervals(n_sub)
    aa, bb = float(a), float(b)
    h = (bb - aa) / n
    w = np.zeros(n + 1, dtype=float)
    if law.name == "trapezoid":
        w[:] = h
        w[0] *= 0.5
        w[n] *= 0.5
    elif law.name == "simpson":
        w[:] = h / 3.0
        w[1 : n : 2] *= 4.0
        w[2 : n : 2] *= 2.0
    else:  # simpson38
        w[:] = 3.0 * h / 8.0
        w[1::3] *= 3.0
        w[2::3] *= 3.0
        w[3:n:3] *= 2.0
    return w


def effective_sub_counts(rule: str, sub_counts: Any) -> np.ndarray:
    """把请求的子区间序列映射成**实际生效**的序列（Simpson 取偶、3/8 取 3 的倍数）。

    做误差曲线拟合时必须用它，而不是用请求值：请求 ``n=8`` 的 3/8 规则实际跑的是
    ``n=9``，拿 8 去算 :math:`h` 会把阶拟合歪（实测：正确序列给 4.00，错误序列给 3.61）。
    想三条规则共用同一批点，直接取 6 的倍数即可。
    """
    ns = np.asarray(sub_counts, dtype=int).ravel()
    return np.asarray([rule_law(rule).needed_subintervals(int(n)) for n in ns], dtype=int)


def rule_eval_count(rule: str, n_sub: int) -> int:
    """该复合规则需要多少次函数求值（三个规则都是 ``n+1`` 个不同的等距节点）。"""
    return rule_law(rule).needed_subintervals(n_sub) + 1


@dataclass(frozen=True)
class RuleEvaluation:
    """一次复合求积的完整快照：节点、函数值、权重、结果，以及"值是否有限"。"""

    rule: str
    n_sub: int
    nodes: np.ndarray
    values: np.ndarray
    weights: np.ndarray
    value: float
    n_evals: int
    finite: bool


def composite_eval(
    f: Callable[[Any], Any],
    a: float,
    b: float,
    n_sub: int,
    *,
    rule: str = "trapezoid",
) -> RuleEvaluation:
    """复合等距规则求积，并把"函数值是否有限"如实记下来。

    奇异端点（如 :math:`(1+x)/\\sqrt x` 在 0 处）会让 ``values`` 出现 ``inf``，
    这时 ``finite=False``、``value`` 是 ``inf``：**这不是数值失败，而是公式在该函数上无定义**。
    帧里必须把这一点标出来，而不是悄悄把 inf 换成 0 或截断节点。
    """
    law = rule_law(rule)
    n = law.needed_subintervals(n_sub)
    x = composite_nodes(rule, n, a, b)
    w = composite_weights(rule, n, a, b)
    v = np.asarray(f(x), dtype=float)
    with np.errstate(invalid="ignore"):
        finite = bool(np.all(np.isfinite(v)))
        total = float(np.sum(w * v)) if finite else float("inf")
    return RuleEvaluation(
        rule=law.name,
        n_sub=n,
        nodes=x,
        values=v,
        weights=w,
        value=total,
        n_evals=int(x.size),
        finite=finite,
    )


def composite_rule(
    f: Callable[[Any], Any],
    a: float,
    b: float,
    n_sub: int,
    *,
    rule: str = "trapezoid",
) -> float:
    """复合等距规则的求积值（奇异端点时返回 ``inf``，用 :func:`composite_eval` 可拿到原因）。"""
    return composite_eval(f, a, b, n_sub, rule=rule).value


# --------------------------------------------------------------------------- #
# 误差主项：幂次、常数与严格括号
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ErrorLaw:
    r"""误差主项 :math:`E=C(b-a)h^p f^{(p)}(\xi)` 的三个常数与一条双侧不等式。"""

    rule: str
    power: int
    derivative: int
    coefficient: float

    def predicted(self, a: float, b: float, n_sub: int, dmin: float, dmax: float) -> tuple[float, float]:
        r"""由 :math:`f^{(p)}` 的界给出的 :math:`E=Q-I` 的**严格**闭区间。

        :math:`E=C(b-a)h^pf^{(p)}(\xi)`、:math:`\xi\in(a,b)`，于是
        :math:`E` 落在 :math:`C(b-a)h^p\cdot[d_{\min},d_{\max}]` 上
        （:math:`C` 与 :math:`f^{(p)}` 都取正号，区间随 :math:`f^{(p)}` 的正负自动翻向）。
        实测的**带符号**误差必须落在区间内部（端点对应 ξ 取到边界）。
        """
        law = rule_law(self.rule)
        n = law.needed_subintervals(n_sub)
        h = (float(b) - float(a)) / n
        scale = self.coefficient * (float(b) - float(a)) * h**self.power
        lo, hi = scale * float(dmin), scale * float(dmax)
        return (min(lo, hi), max(lo, hi))

    def h_power(self, a: float, b: float, n_sub: int) -> float:
        """该规模下的步长 :math:`h^p`（画理论参考线要用）。"""
        n = rule_law(self.rule).needed_subintervals(n_sub)
        return ((float(b) - float(a)) / n) ** self.power


def error_law(rule: str) -> ErrorLaw:
    """取误差主项的三个常数。"""
    law = rule_law(rule)
    return ErrorLaw(
        rule=law.name,
        power=law.error_power,
        derivative=law.error_derivative,
        coefficient=law.error_coefficient,
    )


def error_bracket(
    rule: str,
    a: float,
    b: float,
    n_sub: int,
    dmin: float,
    dmax: float,
) -> tuple[float, float]:
    """便捷入口：``error_law(rule).predicted(...)``。"""
    return error_law(rule).predicted(a, b, n_sub, dmin, dmax)


# --------------------------------------------------------------------------- #
# 实测误差曲线
# --------------------------------------------------------------------------- #
def error_curve(
    item_exact: float,
    f: Callable[[Any], Any],
    a: float,
    b: float,
    sub_counts: Any,
    *,
    rule: str = "trapezoid",
    relative: bool = False,
) -> np.ndarray:
    r"""实测误差 :math:`|Q_n-I|`（``relative=True`` 时除以 :math:`|I|`）。

    奇异被积函数在某个规模上会给出非有限值：这里把它映成 ``inf``——
    是**如实**的（该规则在该规模上无定义），而不是一个可以被 log 轴吃掉的洞。
    """
    ns = np.asarray(sub_counts, dtype=int).ravel()
    ref = float(item_exact)
    scale = abs(ref) if (relative and ref != 0.0) else 1.0
    out = np.empty(ns.size, dtype=float)
    for i, n in enumerate(ns):
        val = composite_rule(f, a, b, int(n), rule=rule)
        with np.errstate(invalid="ignore"):
            out[i] = abs(val - ref) / scale
    return out


def fitted_constant(
    errs: Any,
    a: float,
    b: float,
    sub_counts: Any,
    *,
    power: int,
) -> float:
    r"""由整条误差曲线反推误差常数 :math:`|C|\cdot|f^{(p)}(\xi)|`（取中位数，抗噪）。"""
    ns = np.asarray(sub_counts, dtype=float).ravel()
    e = np.asarray(errs, dtype=float).ravel()
    h = (float(b) - float(a)) / ns
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = e / ((float(b) - float(a)) * h**power)
    good = ratio[np.isfinite(ratio)]
    if good.size == 0:
        return float("nan")
    return float(np.median(good))
