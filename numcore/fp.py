r"""浮点模型：舍入单位 :math:`u`、间距、有效位、补偿求和与误差传播。

数学定义
--------
设 :math:`\mathbb{F}` 是 IEEE-754 binary64（double）的数集，:math:`u = 2^{-53}` 是**单位舍入**
（round-to-nearest 下 :math:`u = \tfrac12\varepsilon_{\text{mach}}`）。本章用到的三个基本模型：

1. **表示模型**：:math:`\mathrm{fl}(x) = x(1+\delta)`，:math:`|\delta| \le u`
   （Higham, *Accuracy and Stability of Numerical Algorithms*, §2.2）。
2. **间距**：:math:`\mathrm{ulp}(x) = 2^{e-52}`（:math:`2^e \le |x| < 2^{e+1}`），
   即相邻两个可表示数之差；:math:`\mathrm{fl}(x)` 与 :math:`x` 的距离不超过
   :math:`\tfrac12\mathrm{ulp}(x)`。
3. **求和的 Wilkinson 界**：朴素左折叠求和 :math:`\hat s` 满足
   :math:`|\hat s - \sum x_i| \le (n-1)u\sum|x_i| + O(u^2)`；
   Kahan 补偿求和的界是 :math:`(2u + O(nu^2))\sum|x_i|`。

本模块只做"浮点本身"的事情：算间距、数有效位、做补偿求和、给误差区间。
它**不**做任何线代（那在 :mod:`numcore.linalg`），也不 import 任何 IO 模块（POLICY P6）；
随机性一律走 :func:`numcore.utils.rng`（P4）。

术语约定
--------
* **机器 epsilon** :math:`\varepsilon_{\text{mach}} = 2^{-52}` 是 1 与下一个可表示数之差；
* **单位舍入** :math:`u = 2^{-53}` 是它的一半——两者**不是**同一个东西，
  把它们混为一谈会让所有误差界差一个 2 倍（见 :func:`unit_roundoff` 的注释）。

:func:`float_grid` 与 :func:`neighborhood` 是给"数轴放大镜"用的：后者给出某个数附近
**全部**相邻可表示数，前者给出区间内的全部可表示数——浮点网格的不均匀性
（跨过 2 的幂时间距突然减半）就是在这条梯子上直接看出来的。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any, Iterable, Sequence

import numpy as np

from numcore.utils import rng

__all__ = [
    # 舍入单位与间距
    "unit_roundoff",
    "machine_epsilon",
    "decimal_digits_limit",
    "spacing",
    "spacings",
    "next_float",
    "ulp_distance",
    "signed_ulp_offset",
    "neighborhood",
    "float_grid",
    # 精确运算（两位和/积）
    "two_sum",
    "fast_two_sum",
    "two_product",
    "split",
    # 求和
    "naive_sum",
    "kahan_sum",
    "neumaier_sum",
    "pairwise_sum",
    "sorted_sum",
    "exact_sum",
    "sum_error",
    "sum_relative_error",
    "sum_error_bound",
    "kahan_error_bound",
    "kahan_sum_shuffled",
    "kahan_sum_parallel",
    "COMPENSATED_SCHEMES",
    "compensated_sum",
    "partial_sum_errors",
    "SumTrajectory",
    "sum_trajectory",
    # 抵消与有效位
    "cancellation_condition",
    "cancellation_interval",
    "remaining_digits",
    "digits_lost",
    "significant_digits",
    "relative_error",
    "sterbenz_exact",
    "exact_decimal",
    "residual_ulps",
    "subtraction_relative_error",
    "perturbation_amplification",
    "accumulation_values",
    "accumulation_residuals",
    # 区间传播
    "Interval",
    "interval_of_float",
    "sum_error_interval",
]

#: 位序变换用的掩码：把 float64 的位模式映成**单调**整数键（负号取反其余位）。
_SIGN_CLEAR: int = 0x7FFF_FFFF_FFFF_FFFF

#: binary64 能可靠表达的十进制有效位数 ``log10(2^53) = 15.95``。
_DECIMAL_DIGITS_LIMIT: float = float(np.log10(2.0**53))


# --------------------------------------------------------------------------- #
# 舍入单位与间距
# --------------------------------------------------------------------------- #
def unit_roundoff(dtype: object = np.float64) -> float:
    r"""单位舍入 :math:`u = \tfrac12\varepsilon_{\text{mach}}`：``fl(x)=x(1+\delta)`` 的 ``|\delta|`` 上界。

    binary64 上 :math:`u = 2^{-53} \approx 1.11\times10^{-16}`。注意它**不是**机器 epsilon：
    ``np.finfo(float).eps = 2^{-52} = 2u`` 是"1 与下一个可表示数之差"。
    两个名字在文献里都被叫过 "machine epsilon"，本章一律用**单位舍入** :math:`u`。
    """
    return float(np.finfo(np.dtype(dtype)).eps) / 2.0


def machine_epsilon(dtype: object = np.float64) -> float:
    r"""机器 epsilon :math:`\varepsilon_{\text{mach}} = 2u`：1 与下一个可表示数之差。"""
    return float(np.finfo(np.dtype(dtype)).eps)


def decimal_digits_limit() -> float:
    r"""binary64 的十进制有效位上限 :math:`\log_{10} 2^{53} \approx 15.95`。

    "还剩 17 位有效数字"这种说法在 binary64 上本身就是谎言，所以一切有效位计数
    都在这个值上封顶（与 :mod:`numcore.verify.tolerance` 同一口径）。
    """
    return _DECIMAL_DIGITS_LIMIT


def spacing(x: float) -> float:
    r"""``|x|`` 处的浮点间距 :math:`\mathrm{ulp}(x)`；``x = 0`` 给最小次正规数。

    网格**不均匀**：:math:`\mathrm{ulp}` 在每个二进制数量级（binade）边界上减半。
    例如 ``spacing(1.0) = 2.22e-16`` 而 ``spacing(nextafter(1.0, 0)) = 1.11e-16``。
    """
    v = abs(float(x))
    if not np.isfinite(v):
        raise ValueError(f"spacing 只对有限数有定义，得到 {x!r}")
    return float(np.spacing(v))


def spacings(x: Any) -> np.ndarray:
    r"""逐元素间距数组（:func:`spacing` 的向量版）——画"网格不均匀"曲线用。"""
    arr = np.abs(np.asarray(x, dtype=float))
    if np.any(~np.isfinite(arr)):
        raise ValueError("spacings 只对有限数有定义")
    return np.spacing(arr)


def _order_key(x: float) -> int:
    """把 double 的位模式映成单调整数键（IEEE-754 全序，手写位运算）。"""
    raw = int(np.array([float(x)], dtype=np.float64).view(np.int64)[0])
    return raw ^ ((raw >> 63) & _SIGN_CLEAR)


def _from_order_key(key: int) -> float:
    """:func:`_order_key` 的逆变换（该变换是对合，逆等于自身）。"""
    raw = int(key) ^ ((int(key) >> 63) & _SIGN_CLEAR)
    return float(np.array([raw], dtype=np.int64).view(np.float64)[0])


def next_float(x: float, *, direction: int = 1, steps: int = 1) -> float:
    r"""沿数轴走 ``steps`` 个可表示数（``direction=+1`` 向上，``-1`` 向下）。

    实现是**位运算**而不是逐次调 ``np.nextafter``：位模式（负数取反其余位）与实数序同构，
    所以"下一个浮点数"就是"整数键 +1"。这也顺带给出了 :func:`ulp_distance` 的定义。
    """
    if direction not in (-1, 1):
        raise ValueError(f"direction 只能是 ±1，得到 {direction!r}")
    n = int(steps)
    if n < 0:
        raise ValueError(f"steps 必须非负，得到 {steps!r}")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"next_float 只对有限数有定义，得到 {x!r}")
    if n == 0:
        return v
    # ±0 的键是相邻的，但数轴上它们是同一个点：natnextafter(0, -inf) = -5e-324 而不是 -0.0。
    # 从 +0 向下（或从 -0 向上）时必须先跳到另一个零的键上，否则会"走了 0 个 ulp"。
    if v == 0.0:
        key = _order_key(-0.0) if direction < 0 else _order_key(0.0)
    else:
        key = _order_key(v)
    return _from_order_key(key + direction * n)


def ulp_distance(a: float, b: float) -> int:
    r"""``a`` 与 ``b`` 之间隔了多少个可表示数（同一点为 0，相邻为 1）。

    :math:`\mathrm{ulp}(0.1+0.2)` 与 :math:`\mathrm{ulp}(0.3)` 的距离恰好是 1——
    这就是"``0.1+0.2 != 0.3``"在浮点网格上的几何含义。
    """
    if not (np.isfinite(float(a)) and np.isfinite(float(b))):
        raise ValueError("ulp_distance 只对有限数有定义")
    if float(a) == float(b):  # +0.0 与 -0.0 是同一个点，键却相邻
        return 0
    return abs(_order_key(float(a)) - _order_key(float(b)))


def signed_ulp_offset(center: float, x: float) -> int:
    r"""``x`` 相对 ``center`` 的**带符号** ulp 偏移（用于在对数尺度的数轴上定位）。

    放大到 :math:`u` 量级后，绝对坐标已经无法用 double 表示（相邻点相差 1e-16），
    唯一能画的是"离中心几个 ulp"——这个函数就是那把尺子。
    """
    d = ulp_distance(center, x)
    return int(d) if float(x) >= float(center) else -int(d)


def neighborhood(x: float, half: int = 32) -> np.ndarray:
    r"""``x`` 附近 **2·half+1** 个连续可表示数（升序）。

    这就是"放大到 u 量级"时屏幕上真正存在的东西：相邻点的间距在跨过 2 的幂时突然减半，
    所以梯子**不等距**。返回的是真实 double，画图时必须改用"距中心的 ulp 偏移"作横坐标，
    否则 double 自己会把这些点全部压成同一个像素（见 ``chapters/1.1/pitfalls.md``）。
    """
    c = float(x)
    if not np.isfinite(c):
        raise ValueError(f"neighborhood 只对有限数有定义，得到 {x!r}")
    k = int(half)
    if k < 0 or k > 4096:
        raise ValueError(f"half 必须在 0..4096，得到 {half!r}")
    lo = [c]
    hi: list[float] = []
    a = c
    b = c
    for _ in range(k):
        a = next_float(a, direction=-1)
        b = next_float(b, direction=1)
        lo.append(a)
        hi.append(b)
    return np.asarray(list(reversed(lo)) + hi, dtype=float)


def float_grid(start: float, stop: float, *, max_points: int = 4096) -> np.ndarray:
    r"""区间 ``[start, stop]`` 内**全部**可表示 double（升序）。

    点太多时抛 :class:`ValueError` 而不是默默截断：一个"看起来是网格、其实是采样"的图
    比没有图更危险。放大到 u 量级时区间里只有几十个点，正好可以逐点画出梯子。
    """
    a = float(start)
    b = float(stop)
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError("float_grid 只对有限端点有定义")
    if b < a:
        raise ValueError(f"需要 start <= stop，得到 [{a}, {b}]")
    cap = int(max_points)
    out = [a]
    x = a
    while x < b:
        x = next_float(x, direction=1)
        out.append(x)
        if len(out) > cap:
            raise ValueError(f"区间 [{a}, {b}] 内可表示数超过 {cap} 个；请缩小窗口或提高 max_points")
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# 精确运算：两位和 / 两位积
# --------------------------------------------------------------------------- #
def two_sum(a: float, b: float) -> tuple[float, float]:
    r"""Knuth 两位和：``s = fl(a+b)`` 与**精确**误差 ``e = (a+b) - s``，二者满足 ``a+b = s+e``。

    这是"浮点加法到底丢了多少"的**精确**答案：:math:`e` 不是估计值，而是可表示的那部分残差。
    ``0.1 + 0.2`` 的 ``e = 5.55e-18``，而 ``fl(0.1)+fl(0.2)-0.3 = 5.55e-17``——
    两个数相差 10 倍，混用就得到错误的结论（前者是加法的舍入，后者还含输入的表示误差）。
    """
    s = float(a) + float(b)
    bb = s - float(a)
    e = (float(a) - (s - bb)) + (float(b) - bb)
    return float(s), float(e)


def fast_two_sum(a: float, b: float) -> tuple[float, float]:
    r"""Fast2Sum 引理（要求 ``|a| >= |b|``）：``s = fl(a+b)``，``e = b - (s-a)`` 精确。

    这是 Kahan 补偿求和能工作的**全部原因**：只要当前和 :math:`|s|` 不小于新项 :math:`|y|`，
    这次加法的舍入误差就可以被精确地"捡回来"存进补偿量 :math:`c`。
    """
    av = float(a)
    bv = float(b)
    if abs(av) < abs(bv):
        raise ValueError(f"fast_two_sum 要求 |a| >= |b|，得到 a={a!r}, b={b!r}")
    s = av + bv
    e = bv - (s - av)
    return float(s), float(e)


def split(a: float) -> tuple[float, float]:
    r"""Dekker 拆分：``a = hi + lo``，``hi`` 只留高 26 位（给两位积用，不需要 FMA）。"""
    c = 134217729.0 * float(a)  # 2^27 + 1
    hi = c - (c - float(a))
    lo = float(a) - hi
    return float(hi), float(lo)


def two_product(a: float, b: float) -> tuple[float, float]:
    r"""Dekker 两位积：``p = fl(a*b)`` 与**精确**误差 ``e = a*b - p``（不用 FMA）。"""
    av = float(a)
    bv = float(b)
    p = av * bv
    ah, al = split(av)
    bh, bl = split(bv)
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return float(p), float(e)


# --------------------------------------------------------------------------- #
# 求和：朴素 / Kahan / Neumaier / 成对 / 排序
# --------------------------------------------------------------------------- #
def _as_float_list(values: Any) -> list[float]:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("求和需要至少一个元素")
    return [float(v) for v in arr]


def naive_sum(values: Any) -> float:
    r"""左折叠朴素求和：:math:`\hat s = \mathrm{fl}(\cdots\mathrm{fl}(\mathrm{fl}(x_1+x_2)+x_3)\cdots)`。

    误差界 :math:`|\hat s - \sum x_i| \le (n-1)u\sum|x_i|`——**随项数线性增长**。
    ``1e16 + 1 + 1 + ...`` 上每加一个 1 就丢一个 1，实测误差恰好等于 :math:`n-1`。
    """
    s = 0.0
    for x in _as_float_list(values):
        s += x
    return float(s)


def kahan_sum(values: Any) -> float:
    r"""Kahan 补偿求和：用补偿量 :math:`c` 把每次加法的舍入误差捡回来。

    误差界 :math:`|\hat s - \sum x_i| \le (2u + O(nu^2))\sum|x_i|`——**与 n 无关**（Higham §4.3）。
    代价是串行依赖：每一步都要等上一步的 :math:`c`，所以它**不可并行、不可乱序**——
    把项打乱（``shuffle``）会让补偿量白存，误差重新变回 :math:`n u` 量级。
    """
    s = 0.0
    c = 0.0
    for x in _as_float_list(values):
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return float(s)


def neumaier_sum(values: Any) -> float:
    r"""Neumaier 变体：按 :math:`|s|` 与 :math:`|x|` 的大小关系分支补误差，最后把补偿量加回。

    比 Kahan 多做一次最终修正（:math:`s + c`），因此在"和远小于各项绝对值"的抵消场景里
    也不会漏掉补偿量；误差界同样是 :math:`u` 量级。
    """
    s = 0.0
    c = 0.0
    for x in _as_float_list(values):
        t = s + x
        if abs(s) >= abs(x):
            c += (s - t) + x
        else:
            c += (x - t) + s
        s = t
    return float(s + c)


def pairwise_sum(values: Any) -> float:
    r"""成对（二分递归）求和：误差界 :math:`O(u\log_2 n)\sum|x_i|`（numpy 的缺省策略）。

    与 Kahan 的区别：成对求和**可以并行**、也不需要额外状态，代价是界里多一个
    :math:`\log_2 n`。把项按大小排序后再朴素求和（:func:`sorted_sum`）能达到类似效果，
    但那要求先看到全部数据。
    """
    vals = _as_float_list(values)
    stack = list(vals)
    while len(stack) > 1:
        nxt: list[float] = []
        for i in range(0, len(stack) - 1, 2):
            nxt.append(stack[i] + stack[i + 1])
        if len(stack) % 2:
            nxt.append(stack[-1])
        stack = nxt
    return float(stack[0])


def sorted_sum(values: Any) -> float:
    r"""按绝对值升序排序后再朴素求和：小项先累加，误差界退化为 :math:`O(u\log n)\sum|x|`。

    这不是"更好的算法"，而是"更好的数据顺序"：它需要先看到全部数据（不能流式），
    并且对**乱序到达**的场景完全无效。
    """
    vals = _as_float_list(values)
    vals.sort(key=abs)
    s = 0.0
    for x in vals:
        s += x
    return float(s)


def kahan_sum_shuffled(values: Any, *, seed: int = 20261018) -> float:
    r"""把项做一次**确定性置换**后再做 Kahan 补偿求和。

    用途是验证一句常被说错的话："乱序会让 Kahan 失效"。**纯置换不会**：
    Kahan 的误差界 :math:`(2u+O(nu^2))\sum|x_i|` 对任意顺序都成立，置换只是换了
    被补偿的次序（实测误差量级不变）。真正失效的是**并行归约**，
    见 :func:`kahan_sum_parallel`。
    """
    vals = _as_float_list(values)
    g = rng(seed)
    perm = g.permutation(len(vals))
    return kahan_sum([vals[int(i)] for i in perm])


def kahan_sum_parallel(values: Any, *, parts: int = 8) -> float:
    r"""分段补偿求和：段内 Kahan、段间**朴素**相加——即真实并行归约的做法。

    误差界退化为主项 :math:`\sim P\,u\,\sum|\hat s_p|`：补偿只在段内有效，
    "跨段"的那一次加法仍然会把小段的和整个丢掉。实测（``big=1e16, small=1e-3, n=4096``）
    串行 Kahan 误差 :math:`9.5\times10^{-2}`，8 段并行后变成 :math:`4.1`——回到朴素水平；
    而 ``small=1`` 时两者都是 1，**看不出差别**（段和恰好是可被大数吸收的偶数倍）。
    """
    vals = _as_float_list(values)
    p = max(1, int(parts))
    n = len(vals)
    if p == 1 or n < 2 * p:
        return kahan_sum(vals)
    seg = n // p
    partials = [kahan_sum(vals[i * seg : (i + 1) * seg]) for i in range(p)]
    if p * seg < n:
        partials.extend(vals[p * seg :])
    return naive_sum(partials)


#: 补偿求和的归约方式（给"求和顺序"可视化用）
COMPENSATED_SCHEMES: tuple[str, ...] = ("serial", "shuffled", "parallel8", "parallel64")


def compensated_sum(values: Any, *, scheme: str = "serial", seed: int = 20261018) -> float:
    r"""按 ``scheme`` 做补偿求和：``serial`` / ``shuffled`` / ``parallel8`` / ``parallel64``。"""
    key = str(scheme).strip().lower()
    if key == "serial":
        return kahan_sum(values)
    if key == "shuffled":
        return kahan_sum_shuffled(values, seed=seed)
    if key.startswith("parallel"):
        parts = int(key[len("parallel") :] or 8)
        return kahan_sum_parallel(values, parts=parts)
    raise KeyError(f"未知补偿求和方式 {scheme!r}（可用 {list(COMPENSATED_SCHEMES)}）")


#: 支持的求和方法名 → 实现
SUM_METHODS: dict[str, Any] = {
    "naive": naive_sum,
    "kahan": kahan_sum,
    "neumaier": neumaier_sum,
    "pairwise": pairwise_sum,
    "sorted": sorted_sum,
}


def exact_sum(values: Any) -> Fraction:
    r"""用有理数算**精确**和 :math:`\sum x_i`（每个 double 按它的精确值进入有理数域）。

    这是所有求和误差断言的"真值"来源：它不经过任何浮点运算，也不做任何近似——
    ``Fraction(0.1)`` 是 :math:`0.1` 那个 double 的精确值，不是数学上的 :math:`1/10`。
    """
    acc = Fraction(0)
    for x in _as_float_list(values):
        acc += Fraction(x)
    return acc


def sum_error(values: Any, method: str = "naive") -> float:
    r"""``|fl(Σx) - Σx|`` 的**实测**值（精确和由 :func:`exact_sum` 给出）。"""
    if method not in SUM_METHODS:
        raise KeyError(f"未知求和方法 {method!r}（可用 {sorted(SUM_METHODS)}）")
    vals = _as_float_list(values)
    exact = exact_sum(vals)
    got = SUM_METHODS[method](vals)
    return float(abs(Fraction(got) - exact))


def sum_relative_error(values: Any, method: str = "naive") -> float:
    r"""``|fl(Σx) - Σx| / |Σx|``；精确和为 0 时退化为绝对误差。"""
    vals = _as_float_list(values)
    exact = exact_sum(vals)
    got = SUM_METHODS[method](vals)
    err = abs(Fraction(got) - exact)
    den = abs(exact)
    return float(err / den) if den != 0 else float(err)


def sum_error_bound(values: Any, *, n_ops: int | None = None, dtype: object = np.float64) -> float:
    r"""朴素求和的 Wilkinson 前向误差界 :math:`n\,u\sum|x_i|`（:math:`n` 缺省取项数）。

    界里的 :math:`\sum|x_i|` 而不是 :math:`|\sum x_i|` 是关键：抵消（:math:`|\sum| \ll \sum|x|`）
    会让**相对**误差爆炸，但**绝对**误差始终被这个量管着。
    """
    vals = _as_float_list(values)
    n = len(vals) if n_ops is None else int(n_ops)
    return float(max(n, 1) * unit_roundoff(dtype) * sum(abs(v) for v in vals))


def kahan_error_bound(values: Any, *, n_ops: int | None = None, dtype: object = np.float64) -> float:
    r"""Kahan 的误差界 :math:`(2u + n u^2)\sum|x_i|`（与项数**无关**的主项）。"""
    vals = _as_float_list(values)
    n = len(vals) if n_ops is None else int(n_ops)
    u = unit_roundoff(dtype)
    return float((2.0 * u + max(n, 1) * u * u) * sum(abs(v) for v in vals))


@dataclass(frozen=True)
class SumTrajectory:
    r"""一次求和实验的**同步演化**轨迹：部分和长度 :math:`k` 处的各方法误差。

    ``error_*`` 是 :math:`|\hat s_k - \sum_{i\le k}x_i|` 的实测值（精确和逐项用有理数累加），
    ``error_bound_*`` 是对应的理论界。``floor`` 是给对数轴用的显示下限
    （误差**恰好为 0** 时对数轴画不出来，必须显式给地板而不是悄悄丢掉）。
    """

    index: np.ndarray
    exact: np.ndarray
    naive: np.ndarray
    kahan: np.ndarray
    neumaier: np.ndarray
    pairwise: np.ndarray
    sorted: np.ndarray
    error_naive: np.ndarray
    error_kahan: np.ndarray
    error_neumaier: np.ndarray
    error_pairwise: np.ndarray
    error_sorted: np.ndarray
    error_bound_naive: np.ndarray
    error_bound_kahan: np.ndarray
    floor: float
    n_exact_zero: int


def partial_sum_errors(values: Any, *, method: str = "kahan", checkpoints: Sequence[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    r"""在若干**前缀**上实测某种求和方法的绝对误差，返回 ``(index, error)``。

    与 :func:`sum_trajectory` 的分工：轨迹函数一次算完五种方法的曲线（流式，快），
    这个函数按需算**任意**方法（含 :func:`compensated_sum` 的并行变体）——每个采样点
    都要重扫一遍前缀，所以只用来补充那一两条曲线。精确和仍由有理数给出。
    """
    vals = _as_float_list(values)
    n = len(vals)
    if checkpoints is None:
        k = min(n, 33)
        pts = np.unique(np.rint(np.geomspace(1.0, float(n), num=k)).astype(int))
        pts = np.unique(np.concatenate([pts, np.array([1, n], dtype=int)]))
        checkpoints = [int(v) for v in pts]
    steps = sorted({int(c) for c in checkpoints if 1 <= int(c) <= n})
    if not steps:
        raise ValueError("checkpoints 必须至少有一个落在 1..n 内")
    idx = np.array([float(k) for k in steps], dtype=float)
    err = np.zeros(len(steps), dtype=float)
    for i, k in enumerate(steps):
        prefix = vals[:k]
        got = compensated_sum(prefix, scheme=method) if method in COMPENSATED_SCHEMES else SUM_METHODS[method](prefix)
        err[i] = float(abs(Fraction(got) - exact_sum(prefix)))
    return idx, err


def sum_trajectory(values: Any, *, checkpoints: Sequence[int] | None = None) -> SumTrajectory:
    r"""沿项数 :math:`k=1..n` 记录各方法的**部分和误差**（一趟扫码 + 有理数精确和）。

    ``checkpoints`` 缺省取 33 个对数/线性混合的采样点。返回的数组与 :class:`SumTrajectory`
    的字段一一对应；误差为 0 的点在数组里保留 0（**不**改写成地板值），
    地板只通过 :attr:`SumTrajectory.floor` 交给画图的人去用——数据与显示必须分开。
    """
    vals = _as_float_list(values)
    n = len(vals)
    if checkpoints is None:
        k = min(n, 33)
        pts = np.unique(np.rint(np.geomspace(1.0, float(n), num=k)).astype(int))
        pts = np.unique(np.concatenate([pts, np.array([1, n], dtype=int)]))
        checkpoints = [int(v) for v in pts]
    steps = sorted({int(c) for c in checkpoints if 1 <= int(c) <= n})
    if not steps:
        raise ValueError("checkpoints 必须至少有一个落在 1..n 内")

    s_naive = 0.0
    s_kahan = 0.0
    c_kahan = 0.0
    s_neu = 0.0
    c_neu = 0.0
    exact = Fraction(0)
    rows: list[tuple[float, ...]] = []
    idx = 0
    for i, x in enumerate(vals, start=1):
        s_naive += x
        y = x - c_kahan
        t = s_kahan + y
        c_kahan = (t - s_kahan) - y
        s_kahan = t
        t2 = s_neu + x
        if abs(s_neu) >= abs(x):
            c_neu += (s_neu - t2) + x
        else:
            c_neu += (x - t2) + s_neu
        s_neu = t2
        exact += Fraction(x)
        if idx < len(steps) and i == steps[idx]:
            idx += 1
            rows.append((float(i), float(exact), s_naive, s_kahan, s_neu))

    index = np.array([r[0] for r in rows], dtype=float)
    exact_arr = np.array([r[1] for r in rows], dtype=float)
    naive_arr = np.array([r[2] for r in rows], dtype=float)
    kahan_arr = np.array([r[3] for r in rows], dtype=float)
    neu_arr = np.array([r[4] for r in rows], dtype=float)

    # 成对 / 排序求和不是流式的：在采样点上对前缀重算（采样点很少，代价可接受）。
    pair_arr = np.array([pairwise_sum(vals[: int(k)]) for k in index], dtype=float)
    sort_arr = np.array([sorted_sum(vals[: int(k)]) for k in index], dtype=float)

    exact_frac = [exact_sum(vals[: int(k)]) for k in index]
    err = {
        "naive": np.array([abs(Fraction(v) - e) for v, e in zip(naive_arr, exact_frac)], dtype=float),
        "kahan": np.array([abs(Fraction(v) - e) for v, e in zip(kahan_arr, exact_frac)], dtype=float),
        "neumaier": np.array([abs(Fraction(v) - e) for v, e in zip(neu_arr, exact_frac)], dtype=float),
        "pairwise": np.array([abs(Fraction(v) - e) for v, e in zip(pair_arr, exact_frac)], dtype=float),
        "sorted": np.array([abs(Fraction(v) - e) for v, e in zip(sort_arr, exact_frac)], dtype=float),
    }
    abs_sum = np.array([float(sum(abs(v) for v in vals[: int(k)])) for k in index], dtype=float)
    u = unit_roundoff()
    bound_naive = index * u * abs_sum
    bound_kahan = (2.0 * u + index * u * u) * abs_sum
    scale = float(np.max(np.abs(exact_arr))) if exact_arr.size else 0.0
    floor = u * scale / 2.0
    if floor <= 0.0:
        floor = float(np.finfo(float).tiny)
    n_zero = int(sum(1 for key in ("naive", "kahan", "pairwise") for v in err[key] if v == 0.0))
    return SumTrajectory(
        index=index,
        exact=exact_arr,
        naive=naive_arr,
        kahan=kahan_arr,
        neumaier=neu_arr,
        pairwise=pair_arr,
        sorted=sort_arr,
        error_naive=err["naive"],
        error_kahan=err["kahan"],
        error_neumaier=err["neumaier"],
        error_pairwise=err["pairwise"],
        error_sorted=err["sorted"],
        error_bound_naive=bound_naive,
        error_bound_kahan=bound_kahan,
        floor=float(floor),
        n_exact_zero=n_zero,
    )


# --------------------------------------------------------------------------- #
# 抵消与有效位
# --------------------------------------------------------------------------- #
def cancellation_condition(a: float, b: float) -> float:
    r"""减法的**条件数** :math:`(|a|+|b|)/|a-b|`（Higham §1.7 的 :math:`\mathrm{cond}(a\pm b)`）。

    它只依赖输入与运算，不依赖算法：:math:`a \approx b` 时它可以是 :math:`10^{16}`，
    此时**任何**算法都救不回有效位。真正精确的是减法本身（见 :func:`sterbenz_exact`），
    被放大的是**输入自带的误差**。
    """
    av = abs(float(a))
    bv = abs(float(b))
    d = abs(float(a) - float(b))
    if d == 0.0:
        return float("inf")
    return float((av + bv) / d)


def sterbenz_exact(a: float, b: float) -> bool:
    r"""Sterbenz 引理：``b/2 <= a <= 2b``（同号）时 ``a-b`` 在浮点里**精确**。

    这条常常被"抵消会丢有效位"的说法盖住：减法本身没有舍入误差，
    ``0.1+0.2-0.3`` 的那 :math:`5.55\times10^{-17}` 全部来自**输入**的表示误差。
    图中若把它标成"算法出错"，是彻底的误读。
    """
    av = float(a)
    bv = float(b)
    if bv == 0.0 or av == 0.0 or (av > 0.0) != (bv > 0.0):
        return False
    ratio = av / bv
    return 0.5 <= ratio <= 2.0


def cancellation_interval(a: float, b: float, *, rel: float | None = None) -> Interval:
    r"""输入各带相对误差 :math:`\le u` 时，:math:`a-b` 的**区间**：:math:`(a-b) \pm u(|a|+|b|)`。

    这才是"抵消"的正确图像：结果的**绝对**误差仍然是 :math:`u(|a|+|b|)`（机器精度量级），
    但结果的**相对**误差是 :math:`u(|a|+|b|)/|a-b|`——抵消不制造误差，它把误差的相对尺度放大。
    """
    r = unit_roundoff() if rel is None else float(rel)
    d = float(a) - float(b)
    rad = r * (abs(float(a)) + abs(float(b)))
    return Interval(d - rad, d + rad)


def remaining_digits(
    result: float, operand_magnitude: float, *, u: float | None = None, dtype: object = np.float64
) -> float:
    r"""抵消后**还剩**多少位十进制有效数字：:math:`\log_{10}(|r|/(u M))`，封顶 15.95。

    :math:`M = \max(|a|,|b|)` 是操作数量级。:math:`|r|` 每小 10 倍就少一位有效数字：
    :math:`|r| = 10^{-8}M` 时只剩约 8 位——这就是"有效位一位一位消失"的定量版本。

    ``u`` 可显式给出（例如"输入只有 d 位十进制有效数字"对应的
    :math:`\delta_{\rm in} = \tfrac12 10^{1-d}`）；缺省用机器单位舍入 :math:`u`。
    """
    res = abs(float(result))
    m = abs(float(operand_magnitude))
    rel = unit_roundoff(dtype) if u is None else float(u)
    if m == 0.0:
        return _DECIMAL_DIGITS_LIMIT
    if res == 0.0:
        return 0.0
    d = float(np.log10(res / m) - np.log10(rel))
    return float(min(max(d, 0.0), _DECIMAL_DIGITS_LIMIT))


def digits_lost(a: float, b: float) -> float:
    r"""抵消**损失**的十进制位数 :math:`\log_{10}((|a|+|b|)/|a-b|)`（无抵消时为 0）。"""
    c = cancellation_condition(a, b)
    if not np.isfinite(c) or c <= 1.0:
        return 0.0
    return float(np.log10(c))


def significant_digits(value: float, reference: float) -> float:
    r"""``value`` 与 ``reference`` 一致的前导十进制位数 :math:`-\log_{10}(|v-r|/|r|)`。

    与 :func:`remaining_digits` 的分工：这个量比较**两个具体的数**（实测误差），
    后者给的是"抵消把误差放大到多少"的**先验**估计。断言里两者都要出现：
    先验说该剩 8 位，实测就得真剩 8 位。
    """
    v = float(value)
    r = float(reference)
    if v == r:
        return _DECIMAL_DIGITS_LIMIT
    if r == 0.0 or not np.isfinite(v) or not np.isfinite(r):
        return 0.0
    d = -float(np.log10(abs(v - r) / abs(r)))
    return float(min(max(d, 0.0), _DECIMAL_DIGITS_LIMIT))


def relative_error(computed: float, exact: float) -> float:
    r"""相对误差 :math:`|c-e|/|e|`（:math:`e=0` 时退化为绝对误差）。"""
    c = float(computed)
    e = float(exact)
    if e == 0.0:
        return abs(c - e)
    return float(abs(c - e) / abs(e))


def exact_decimal(x: float, *, digits: int = 25) -> str:
    r"""``x`` 作为 double 的**精确**十进制展开（截到 ``digits`` 位有效数字）。

    用来在 UI 上写清"你以为的 0.1 其实是 0.1000000000000000055511151231257827…"。
    它是纯字符串函数（不打印），字符串走帧的 ``strings`` 通道。
    """
    d = Decimal(float(x))
    with localcontext() as ctx:
        ctx.prec = max(1, int(digits))
        return format(+d, "f")


def residual_ulps(a: float, b: float) -> int:
    r"""两个 double 之间隔多少个 ulp（:func:`ulp_distance` 的语义化别名）。

    ``residual_ulps(0.1+0.2, 0.3) == 1``：所谓"0.1+0.2 不等于 0.3"，就是"差了 1 个 ulp"。
    """
    return ulp_distance(a, b)


def subtraction_relative_error(a: float, b: float) -> float:
    r"""``fl(a-b)`` 相对**精确差** :math:`a-b` 的相对误差（有理数域里算，不做近似）。

    结果常常**恰好是 0**：Sterbenz 引理保证 ``b/2 <= a <= 2b`` 时减法精确。
    所以"抵消"这个词描述的不是减法运算出错，而是**输入自带的误差**被放大
    （见 :func:`cancellation_condition` 与 :func:`perturbation_amplification`）。
    """
    av = float(a)
    bv = float(b)
    exact = Fraction(av) - Fraction(bv)
    got = Fraction(av - bv)
    if exact == 0:
        return 0.0 if got == 0 else float("inf")
    return float(abs(got - exact) / abs(exact))


def perturbation_amplification(a: float, b: float, *, ulps: int = 1) -> float:
    r"""两个输入各被推开 ``ulps`` 个 ulp 后，差 :math:`a-b` 的**相对**变化。

    这是"抵消"的实测放大比：把 ``a`` 向上、``b`` 向下各推 ``k`` 个 ulp（差最大的扰动方向），
    精确算出新的差，再除以原差。

    .. math::
        \text{amp} = \frac{k\,(\mathrm{ulp}(a)+\mathrm{ulp}(b))}{|a-b|}
        \approx \frac{2k\,u\max(|a|,|b|)}{|a-b|}

    它与 :func:`cancellation_condition` 只差一个 :math:`u` 因子：前者是"输入误差被放大多少倍"，
    后者是同一个放大倍数（无量纲化之后）。
    """
    av = float(a)
    bv = float(b)
    k = int(ulps)
    if k < 0:
        raise ValueError(f"ulps 必须非负，得到 {ulps!r}")
    if k == 0:
        return 0.0
    exact = Fraction(av) - Fraction(bv)
    if exact == 0:
        return float("inf")
    a_p = next_float(av, direction=1, steps=k)
    b_p = next_float(bv, direction=-1, steps=k)
    moved = Fraction(a_p) - Fraction(b_p)
    return float(abs(moved - exact) / abs(exact))


def accumulation_values(x: float, count: int) -> np.ndarray:
    r"""把 ``x`` 连续累加 :math:`k=0..\text{count}` 次的部分和 ``fl(fl(...(x+x)+x))``。

    注意这是**逐步累加**而不是 :math:`\mathrm{fl}(kx)`：``0.1`` 累加三次得到
    ``0.30000000000000004``，而 ``fl(3*0.1)`` 也是同一个数——但 k 更大时两条路会分岔，
    "先乘后加"的误差模型与"累加"的误差模型不是一回事。
    """
    xv = float(x)
    n = int(count)
    if n < 0:
        raise ValueError(f"count 必须非负，得到 {count!r}")
    out = np.zeros(n + 1, dtype=float)
    s = 0.0
    for k in range(n + 1):
        out[k] = s
        s += xv
    return out


def accumulation_residuals(
    x: float, count: int, *, reference: tuple[int, int] | None = None
) -> np.ndarray:
    r"""累加 :math:`k` 次 ``x`` 的部分和与**精确值**的绝对残差（``k=0..count``）。

    精确值用有理数算，所以这是"实测误差"而不是估计。``reference`` 决定"精确值"是谁：

    * ``None``（缺省）：:math:`k\cdot x`，其中 :math:`x` 是**那个 double** 的精确值——
      这量的是**纯求和误差**（输入精确已知，Wilkinson 模型）；
    * ``reference=(p, q)``：:math:`k\cdot p/q`——例如 ``(1, 10)`` 表示"``0.1`` 想代表的
      数学值 :math:`1/10`"，这量的是**表示误差 + 求和误差**，也就是
      ``0.1+0.2-0.3 = 5.55e-17`` 那类现象。

    两个残差不是一回事，图上必须分开画：把它们混起来，就会把"输入本身就不精确"
    误读成"求和算法不稳定"。
    """
    xv = float(x)
    n = int(count)
    if n < 0:
        raise ValueError(f"count 必须非负，得到 {count!r}")
    ref = Fraction(xv) if reference is None else Fraction(int(reference[0]), int(reference[1]))
    out = np.zeros(n + 1, dtype=float)
    s = 0.0
    for k in range(n + 1):
        out[k] = float(abs(Fraction(s) - k * ref))
        s += xv
    return out


# --------------------------------------------------------------------------- #
# 区间传播
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Interval:
    r"""闭区间 :math:`[lo, hi]`，用**向外舍入**的端点运算做误差传播。

    ``[a] + [b] = [fl_down(a.lo+b.lo), fl_up(a.hi+b.hi)]``：区间算术的每一步都只在
    端点做一次浮点运算并向外扩张一个 ulp，因此结果**保证**包含精确值。
    它是"误差传播"最保守也最可靠的图像——比"误差 ~ u"的估计更强，因为它从不低估。
    """

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if not (np.isfinite(self.lo) and np.isfinite(self.hi)):
            raise ValueError("区间端点必须是有限数")
        if self.lo > self.hi:
            raise ValueError(f"需要 lo <= hi，得到 [{self.lo}, {self.hi}]")

    # -- 基本量 ---------------------------------------------------------- #
    @property
    def width(self) -> float:
        r"""区间宽度 :math:`hi-lo`（向外扩张一个 ulp，保证不小于真宽度）。"""
        w = self.hi - self.lo
        if w <= 0.0:
            return 0.0
        return float(next_float(w, direction=1))

    @property
    def mid(self) -> float:
        """区间中点。"""
        return 0.5 * (self.lo + self.hi)

    @property
    def radius(self) -> float:
        """半径（半宽）。"""
        return 0.5 * (self.hi - self.lo)

    def contains(self, x: float) -> bool:
        """``x`` 是否落在区间内。"""
        return bool(self.lo <= float(x) <= self.hi)

    # -- 运算（端点向外一步） -------------------------------------------- #
    def __add__(self, other: "Interval") -> "Interval":
        lo = next_float(self.lo + other.lo, direction=-1)
        hi = next_float(self.hi + other.hi, direction=1)
        return Interval(lo, hi)

    def __sub__(self, other: "Interval") -> "Interval":
        lo = next_float(self.lo - other.hi, direction=-1)
        hi = next_float(self.hi - other.lo, direction=1)
        return Interval(lo, hi)

    def __mul__(self, other: "Interval") -> "Interval":
        cands = (
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        )
        lo = next_float(min(cands), direction=-1)
        hi = next_float(max(cands), direction=1)
        return Interval(lo, hi)

    def scaled(self, c: float) -> "Interval":
        """乘一个**精确已知**的常数 ``c``（不与 :meth:`__mul__` 的区间相乘混淆）。"""
        return self * Interval(float(c), float(c))


def interval_of_float(x: float, *, rel: float | None = None) -> Interval:
    r"""把一个浮点数看成"真值带相对误差 :math:`\le u`"的区间 :math:`[x(1-u), x(1+u)]`。"""
    r = unit_roundoff() if rel is None else float(rel)
    v = float(x)
    rad = r * abs(v)
    return Interval(v - rad, v + rad)


def sum_error_interval(values: Any, method: str = "naive", *, dtype: object = np.float64) -> Interval:
    r"""求和结果的误差区间：:math:`\hat s \pm b`，:math:`b` 取该方法的误差界。

    朴素取 :math:`n u \sum|x_i|`，Kahan 取 :math:`(2u+nu^2)\sum|x_i|`，成对取
    :math:`u\lceil\log_2 n\rceil\sum|x_i|`。区间**一定**包含精确和——这是可以直接断言的。
    """
    vals = _as_float_list(values)
    got = SUM_METHODS.get(method)
    if got is None:
        raise KeyError(f"未知求和方法 {method!r}（可用 {sorted(SUM_METHODS)}）")
    u = unit_roundoff(dtype)
    s_abs = sum(abs(v) for v in vals)
    if method == "naive":
        bound = len(vals) * u * s_abs
    elif method in ("kahan", "neumaier"):
        bound = (2.0 * u + len(vals) * u * u) * s_abs
    elif method == "pairwise":
        bound = u * float(np.ceil(np.log2(max(len(vals), 2)))) * s_abs
    else:  # sorted
        bound = (len(vals) - 1) * u * s_abs
    b = float(bound)
    c = float(got(vals))
    return Interval(next_float(c - b, direction=-1), next_float(c + b, direction=1))
