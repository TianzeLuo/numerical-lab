"""2.4 章的**被积函数目录**：每一个都带解析原函数、光滑性标签与（可算的）导数界。

为什么要有这样一个目录
----------------------
"求积的误差阶"这句话只有配上**被积函数的光滑性**才有意义。同一套节点和权重，
在 :math:`e^x` 上量到 4 阶、在 :math:`\\sqrt x` 上量到 1.5 阶、在 :math:`|x|` 上
甚至量到"精确"（偶数个等距子区间时折点恰好落在节点上，梯形法逐段精确）。
把这三件事放进同一个目录、每个条目都带**解析真值**，才能让"量到的阶"有地方对账。

每个条目给出四样东西：

* ``exact``：:math:`\\int_a^b f` 的解析值（double 精度内可直接当参照物）；
* ``smoothness`` / ``order_cap``：:math:`f` 在端点附近的奇异指数 :math:`\\alpha`
  给出的**阶上限** :math:`\\min(p,\\alpha+1)`（:math:`p` 是方法的阶）——
  这是本章"非光滑函数上量到的不是求积的阶"那句话的可计算版本；
* ``derivative_bounds(q)``：:math:`f^{(q)}` 在 :math:`[a,b]` 上的**严格**上下界
  （能解析给出的才给，给不出就返回 ``None``）。有了它，Newton--Cotes 的误差主项
  :math:`E=C(b-a)h^p f^{(p)}(\\xi)` 就从"渐近说法"变成一条**双侧不等式**，
  断言可以卡住区间而不是只卡阶；
* ``substitution``：换元 :math:`x=\\varphi(t)` 与化简后的 :math:`g(t)=f(\\varphi(t))\\varphi'(t)`。
  :math:`g` 在端点**有限**——这正是"换元把奇异性变成光滑性"的可执行版本。

所有函数都只用 numpy 的逐元素原语，且对奇异端点显式使用 ``np.errstate`` 抑制
除零/无效值告警（本仓 pytest 把 ``RuntimeWarning`` 当错误：``filterwarnings = error``）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

__all__ = [
    "INTEGRAND_NAMES",
    "Integrand",
    "Substitution",
    "expected_order",
    "integrand",
    "integrand_names",
    "substituted_error_curve",
]


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Substitution:
    r"""换元 :math:`x=\varphi(t)`，以及化简后有限的被积函数 :math:`g(t)=f(\varphi(t))\varphi'(t)`。

    ``g`` **必须是解析化简过的**：直接算 ``f(phi(t))*jac(t)`` 在奇异端会得到
    ``inf * 0``，那是 NaN 的来源，而不是数学。
    """

    name: str
    domain: tuple[float, float]
    exact: float
    phi: Callable[[Any], Any]
    jac: Callable[[Any], Any]
    g: Callable[[Any], Any]
    smoothness: str


@dataclass(frozen=True)
class Integrand:
    """一个带解析真值的测试被积函数（区间、真值、光滑性、导数界、换元）。"""

    name: str
    label: str
    a: float
    b: float
    exact: float
    f: Callable[[Any], Any]
    smoothness: str
    order_cap: float
    breakpoints: tuple[float, ...] = ()
    singular_endpoints: tuple[float, ...] = ()
    derivative_bounds: Callable[[int], tuple[float, float] | None] | None = None
    substitution: Substitution | None = None
    note: str = ""
    params: dict[str, float] = field(default_factory=dict)
    #: :math:`\int_a^{x}` 的解析值：用来算"没被采样的尾巴"有多大（tanh--sinh 的地板就来自它）。
    tail: Callable[[float], float] | None = None

    def values(self, x: Any) -> np.ndarray:
        """在被积函数上逐点取值（返回 float 数组，可能是 ±inf——调用方负责判有限性）。"""
        return np.asarray(self.f(np.asarray(x, dtype=float)), dtype=float)

    def tail_integral(self, x_min: float) -> float | None:
        r"""从 :math:`a` 到 :math:`x_{\min}` 的解析积分（未采样尾巴），没有闭式就返回 ``None``。

        它的用处是给双指数求积的误差地板一个**可计算的解释**：变换后的节点
        最靠近奇异端点只能到 :math:`x_{\min}\approx u/2`（:math:`\tanh` 在 double 里舍入到 ±1），
        于是 :math:`\int_0^{x_{\min}}x^{-1/2}dx=2\sqrt{x_{\min}}\approx\sqrt{2u}\approx1.5\times10^{-8}`
        ——实测地板 1.17e-8 ~ 1.98e-8 正是这个量级，而不是 :math:`u`。
        """
        if self.tail is None:
            return None
        return float(self.tail(float(x_min)))


# --------------------------------------------------------------------------- #
# 各条目的被积函数（全部对数组友好）
# --------------------------------------------------------------------------- #
def _exp(x: Any) -> np.ndarray:
    return np.exp(np.asarray(x, dtype=float))


def _runge(x: Any) -> np.ndarray:
    xx = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + 25.0 * xx * xx)


def _cusp(x: Any) -> np.ndarray:
    return np.abs(np.asarray(x, dtype=float))


def _poly5(x: Any) -> np.ndarray:
    xx = np.asarray(x, dtype=float)
    return xx**5


def _sqrt_endpoint(x: Any) -> np.ndarray:
    xx = np.asarray(x, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.clip(xx, 0.0, None))


def _inv_sqrt(x: Any) -> np.ndarray:
    r""":math:`(1+x)/\sqrt x`：:math:`x\to0^+` 时无界，``f(0) = +inf`` 是**如实**返回。"""
    xx = np.asarray(x, dtype=float)
    safe = np.where(xx > 0.0, xx, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = (1.0 + xx) / np.sqrt(safe)
    return np.where(xx > 0.0, val, np.inf)


def _spike(x: Any, *, center: float = 0.3, sigma: float = 0.02) -> np.ndarray:
    r""":math:`e^{-((x-c)/\sigma)^2}`：宽度 :math:`\sigma` 的高斯峰，是"自适应求积会骗人"的标准反例。"""
    xx = (np.asarray(x, dtype=float) - center) / sigma
    return np.exp(-xx * xx)


def _log_endpoint(x: Any) -> np.ndarray:
    xx = np.asarray(x, dtype=float)
    safe = np.where(xx > 0.0, xx, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.log(safe)
    return np.where(xx > 0.0, val, -np.inf)


# --------------------------------------------------------------------------- #
# 各条目的导数界（严格上下界，用于把误差主项写成双侧不等式）
# --------------------------------------------------------------------------- #
def _bounds_exp(q: int) -> tuple[float, float]:
    """:math:`(e^x)^{(q)}=e^x` 在 [0,1] 上落在 [1, e]。"""
    return (1.0, math.e)


def _bounds_poly5(q: int) -> tuple[float, float] | None:
    r""":math:`(x^5)^{(q)} = \frac{5!}{(5-q)!}x^{5-q}` 在 [0,1] 上落在 [0, 5!/(5-q)!]。"""
    if q > 5:
        return (0.0, 0.0)
    if q < 0:
        return None
    return (0.0, math.factorial(5) / math.factorial(5 - q))


def _bounds_runge(q: int) -> tuple[float, float] | None:
    """Runge 函数的高阶导数界不解析给出（返回 None，让断言只用阶、不用常数）。"""
    return None


def _bounds_oscillation(omega: float) -> Callable[[int], tuple[float, float]]:
    r""":math:`|\frac{d^q}{dx^q}\sin(\omega x)|\le\omega^q`（对任意 x 成立）。"""

    def bounds(q: int) -> tuple[float, float]:
        return (-(omega**q), omega**q)

    return bounds


# --------------------------------------------------------------------------- #
# 目录
# --------------------------------------------------------------------------- #
_INTEGRANDS: dict[str, Integrand] = {
    "exp": Integrand(
        name="exp",
        label="e^x（解析，所有导数有界）",
        a=0.0,
        b=1.0,
        exact=math.e - 1.0,
        f=_exp,
        smoothness="C^∞：所有阶导数在 [0,1] 上有界",
        order_cap=math.inf,
        derivative_bounds=_bounds_exp,
        tail=lambda x: math.exp(x) - 1.0,
        note="光滑基准：复合规则的实测阶应当**等于**公式的阶，误差主项的常数也能卡住。",
    ),
    "poly5": Integrand(
        name="poly5",
        label="x^5（多项式，导数界是整数）",
        a=0.0,
        b=1.0,
        exact=1.0 / 6.0,
        f=_poly5,
        smoothness="C^∞（5 次多项式）",
        order_cap=math.inf,
        derivative_bounds=_bounds_poly5,
        tail=lambda x: x**6 / 6.0,
        note="常数的双侧卡点在这里最干净：f⁗(ξ)=120ξ 的界是 [0,120]。",
    ),
    "runge": Integrand(
        name="runge",
        label="1/(1+25x²)（光滑但导数很大）",
        a=-1.0,
        b=1.0,
        exact=0.4 * math.atan(5.0),
        f=_runge,
        smoothness="C^∞：导数界随阶数按 25^{q/2} 增长",
        order_cap=math.inf,
        tail=lambda x: math.atan(5.0 * x) / 5.0,
        note="渐近区来得晚：高阶方法的误差常数随阶数暴涨，p 阶只在足够小的 h 才显形。",
    ),
    "cusp": Integrand(
        name="cusp",
        label="|x|（折点，f′ 跳变）",
        a=-1.0,
        b=1.0,
        exact=1.0,
        f=_cusp,
        smoothness="C^0：f′ 在 0 处跳变，f″ 是 δ 函数",
        order_cap=2.0,
        breakpoints=(0.0,),
        tail=lambda x: 0.5 * x * x,
        note="陷阱：折点落在节点上时梯形法**逐段精确**，误差是 0（不是「阶很高」）。",
    ),
    "sqrt": Integrand(
        name="sqrt",
        label="√x（端点导数无界）",
        a=0.0,
        b=1.0,
        exact=2.0 / 3.0,
        f=_sqrt_endpoint,
        smoothness="C^0 ∩ C^∞(0,1]：f′(x)=1/(2√x) 在 0 处无界（α=1/2）",
        order_cap=1.5,
        singular_endpoints=(0.0,),
        substitution=Substitution(
            name="x = t²",
            domain=(0.0, 1.0),
            exact=2.0 / 3.0,
            phi=lambda t: np.asarray(t, dtype=float) ** 2,
            jac=lambda t: 2.0 * np.asarray(t, dtype=float),
            g=lambda t: 2.0 * np.asarray(t, dtype=float) ** 2,
            smoothness="换元后 g(t)=2t² 是多项式：导数无界的毛病被变量替换吃掉",
        ),
        tail=lambda x: (2.0 / 3.0) * x**1.5,
        note="阶上限 min(p, α+1) = 1.5：Simpson 的 4 阶在这里塌到 1.5，量到的不是公式的阶。",
    ),
    "inv_sqrt": Integrand(
        name="inv_sqrt",
        label="(1+x)/√x（端点本身无界）",
        a=0.0,
        b=1.0,
        exact=8.0 / 3.0,
        f=_inv_sqrt,
        smoothness="x→0⁺ 时 f 本身发散（α=−1/2）：等距规则在 x=0 处的函数值无定义",
        order_cap=0.5,
        singular_endpoints=(0.0,),
        substitution=Substitution(
            name="x = t²",
            domain=(0.0, 1.0),
            exact=8.0 / 3.0,
            phi=lambda t: np.asarray(t, dtype=float) ** 2,
            jac=lambda t: 2.0 * np.asarray(t, dtype=float),
            g=lambda t: 2.0 + 2.0 * np.asarray(t, dtype=float) ** 2,
            smoothness="换元后 g(t)=2+2t² 是多项式：等距规则重新可用（但仍是代数阶）",
        ),
        tail=lambda x: 2.0 * math.sqrt(x) + (2.0 / 3.0) * x**1.5,
        note="等距规则在 x=0 处取到 +inf，整条复合公式无定义——这一点本身就是证据。",
    ),
    "spike": Integrand(
        name="spike",
        label="exp(−((x−0.3)/0.02)²)（窄高斯峰）",
        a=0.0,
        b=1.0,
        exact=0.02 * math.sqrt(math.pi) * 0.5
        * (math.erf((1.0 - 0.3) / 0.02) + math.erf(0.3 / 0.02)),
        f=_spike,
        smoothness="C^∞：但特征宽度只有 0.02，初始采样点（0、0.5、1）全都落在峰外",
        order_cap=math.inf,
        tail=lambda x: 0.02 * math.sqrt(math.pi) * 0.5
        * (math.erf((x - 0.3) / 0.02) + math.erf(0.3 / 0.02)),
        note=(
            "自适应求积的反例：初始三点 Simpson 全落在峰外，得到 0，"
            "误差估计也恰好是 0 ——它当场宣布收敛，而真误差是 3.5e-2 量级。"
        ),
        params={"center": 0.3, "sigma": 0.02},
    ),
    "log": Integrand(
        name="log",
        label="ln x（对数端点奇异）",
        a=0.0,
        b=1.0,
        exact=-1.0,
        f=_log_endpoint,
        smoothness="x→0⁺ 时 f→−∞（对数奇异）：任何「阶」的说法都要重新算",
        order_cap=math.inf,
        singular_endpoints=(0.0,),
        substitution=Substitution(
            # 只用**有限区间**的换元：x=e^(−t) 那条半无穷路线（Gauss–Laguerre 的天然对象）
            # 在 [0,∞) 上没法用等距规则直接验证，写进帧里只会得到一串 NaN——
            # 实测踩过这个坑：换元域是 inf 时 composite_rule 返回 NaN，
            # 而 NaN 会被地板截断悄悄"修好"，图上看起来像一条正常的曲线。
            name="x = t²",
            domain=(0.0, 1.0),
            exact=-1.0,
            phi=lambda t: np.asarray(t, dtype=float) ** 2,
            jac=lambda t: 2.0 * np.asarray(t, dtype=float),
            # 4t·ln t：t→0 时 0·(−∞) 要按 0 处理（真实极限），
            # 且必须写成 log(where(t>0,t,1)) 才不会有 invalid value 警告。
            g=lambda t: 4.0
            * np.asarray(t, dtype=float)
            * np.log(np.where(np.asarray(t, dtype=float) > 0.0, np.asarray(t, dtype=float), 1.0)),
            smoothness="换元后 g(t)=4t·ln t 有界但仍带对数（导数在 0 处发散）：阶远高于原始被积函数、却到不了机器精度",
        ),
        tail=lambda x: x * (math.log(x) - 1.0) if x > 0 else 0.0,
        note="tanh–sinh 对对数奇异的收敛比幂奇异慢得多——这是两条曲线分叉的实测来源。",
    ),
    "oscillation": Integrand(
        name="oscillation",
        label="sin(ωx)（振荡）",
        a=0.0,
        b=1.0,
        exact=math.sin(10.0) / 10.0,
        f=lambda x: np.sin(10.0 * np.asarray(x, dtype=float)),
        smoothness="C^∞，但 |f^(q)| ≤ ω^q：ω 一大，全局规则的误差常数就跟着 ω^p 爆",
        order_cap=math.inf,
        derivative_bounds=_bounds_oscillation(10.0),
        tail=lambda x: (1.0 - math.cos(10.0 * x)) / 10.0,
        note="阶没变、常数变了：ω=40 时全局 Gauss 在同等节点数下比 ω=4 差 4^p 倍。",
        params={"omega": 10.0},
    ),
}

#: 目录里可用的名字（顺序固定，便于测试与帧里按序号取）。
INTEGRAND_NAMES: tuple[str, ...] = (
    "exp",
    "poly5",
    "runge",
    "cusp",
    "sqrt",
    "inv_sqrt",
    "log",
    "spike",
    "oscillation",
)


def integrand_names() -> tuple[str, ...]:
    """目录里所有被积函数的名字（固定顺序）。"""
    return INTEGRAND_NAMES


def integrand(name: str, *, omega: float = 10.0) -> Integrand:
    """取出目录里的条目；``oscillation`` 允许用 ``omega`` 参数化。

    参数化条目会**重建**一个 :class:`Integrand`（而不是改动目录里的常量），
    这样"目录"始终是不可变真源，帧里也不会出现两个 viz 互相污染的状态。
    """
    key = str(name)
    if key not in _INTEGRANDS:
        raise KeyError(f"未知被积函数 {name!r}（可用：{', '.join(INTEGRAND_NAMES)}）")
    base = _INTEGRANDS[key]
    if key != "oscillation":
        return base
    w = float(omega)
    if not math.isfinite(w) or w <= 0.0:
        raise ValueError(f"omega 必须是正有限数，得到 {omega!r}")
    return Integrand(
        name=base.name,
        label=f"sin({w:g}x)（振荡，ω={w:g}）",
        a=base.a,
        b=base.b,
        exact=math.sin(w) / w,
        f=lambda x: np.sin(w * np.asarray(x, dtype=float)),
        smoothness=base.smoothness,
        order_cap=base.order_cap,
        derivative_bounds=_bounds_oscillation(w),
        tail=lambda x: (1.0 - math.cos(w * x)) / w,
        note=base.note,
        params={"omega": w},
    )


def expected_order(rule_order: int, alpha: float) -> float:
    r"""非光滑被积函数的**实测阶预测**：:math:`\min(p,\alpha+1)`。

    端点附近 :math:`f\sim x^{\alpha}` 时，Euler--Maclaurin 展开被截断在
    :math:`n^{-(\alpha+1)}` 项上：方法阶 :math:`p` 再高也白搭。:math:`\alpha=1/2`
    给出 1.5（本章实测：梯形与 Simpson 都塌到 1.5），:math:`\alpha=-1/2`
    （函数本身发散）连定义都要另外处理。
    """
    return float(min(float(rule_order), float(alpha) + 1.0))


def substituted_error_curve(
    item: Integrand,
    sub_counts: Any,
    *,
    rule: str = "trapezoid",
) -> np.ndarray:
    """换元之后、用复合规则在 :math:`t` 上的误差曲线（真值取 ``substitution.exact``）。

    这条曲线是"换元把奇异性变成光滑性"的**实测**版本：同一个等距规则，
    只因为换了变量就重新收敛（代价是误差常数与端点行为都变了）。
    """
    from numcore.quad.newton_cotes import composite_rule

    if item.substitution is None:
        raise ValueError(f"被积函数 {item.name!r} 没有登记换元")
    sub = item.substitution
    lo, hi = sub.domain
    errs = []
    for n in np.asarray(sub_counts, dtype=int).ravel():
        val = composite_rule(sub.g, lo, hi, int(n), rule=rule)
        errs.append(abs(val - sub.exact))
    return np.asarray(errs, dtype=float)
