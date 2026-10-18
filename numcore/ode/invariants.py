r"""3.3 章的长期行为诊断：把「有界」与「漂移」变成**数字**。

为什么需要单独一个模块
----------------------
本章最容易犯的错不是算错，而是**量错**：

* 「能量误差看起来有界」不是证据。有限时间里任何东西看起来都有界；
  要么给出 :math:`\sup_k|H_k-H_0|`（带宽）与「总趋势 / 带宽」的比值，
  要么就是一张会骗人的图。而且**在一条有界振荡上做最小二乘，斜率必然非零**
  （采样混叠给出的假趋势），所以判据必须是「趋势 × 总时长 ≪ 带宽」，不是「斜率 = 0」。
* 「轨道闭合」也有两种读法：点的**集合**闭合（落在同一条曲线上）与**第 n 圈回到出发点**。
  辛方法的航向（相位）一直在漂，两者必须分开量：前者用映射的不变量
  :math:`H_h` 的相对变化，后者用每圈的最近回归距离。
* 「动量守恒到机器精度」**不能**证明辛性：任何 RK 方法都保线性不变量
  （:math:`\sum_ib_i=1`）。角动量（二次不变量）才是判别式。

本模块只做测量：输入时间序列/映射，输出带单位的数字。全部纯函数，无 IO、无随机
（P6）。幂律斜率复用 3.1 章的 :func:`numcore.ode.rk.loglog_slope`，与
:func:`numcore.verify.fit_order` 同口径——"图上画的斜率"与"断言里拟合的斜率"是同一个数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from numcore.ode.rk import loglog_slope

__all__ = [
    "Band",
    "PowerLaw",
    "RadiusStats",
    "ClosureStats",
    "QuadraticInvariant",
    "band",
    "relative_variation",
    "drift_slope",
    "half_split",
    "drift_over_band",
    "band_growth",
    "power_law",
    "radius_stats",
    "phase_error_turns",
    "wrapped_angle",
    "quadratic_invariant",
    "invariant_series",
    "invariant_level_curve",
    "return_distances",
    "ReturnStats",
    "closure_stats",
    "radial_deviation",
    "reference_closure",
    "curve_closure",
    "level_curve_radius",
    "momentum_drift",
    "angular_momentum_drift",
    "monotone_fraction",
]

#: 单位舍入（判定"机器精度级"的统一口径，与 1.2/3.2 章一致）。
U = float(np.finfo(float).eps) / 2.0


# --------------------------------------------------------------------------- #
# 1. 带宽与趋势
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Band:
    r"""一条序列的**带**：:math:`[\min,\max]`、带宽、中点，以及 :math:`\sup|v-v_0|`。

    ``amplitude`` 是 :math:`\max-\min`（带宽），``sup_from_start`` 是
    :math:`\sup_k|v_k-v_0|`。"能量误差有界"这句话在帧里必须落成 ``sup_from_start``
    这个数——它才是与 :math:`h` 有幂律关系的那个量。
    """

    lo: float
    hi: float
    amplitude: float
    midpoint: float
    sup_from_start: float
    n: int

    def as_dict(self) -> dict[str, float]:
        return {
            "band_lo": self.lo,
            "band_hi": self.hi,
            "band_amplitude": self.amplitude,
            "band_midpoint": self.midpoint,
            "sup_from_start": self.sup_from_start,
        }


def band(series: Any) -> Band:
    """把序列压成 :class:`Band`（空序列抛错——空数组没有"界"可言）。"""
    v = np.asarray(series, dtype=float).ravel()
    if v.size == 0:
        raise ValueError("空序列没有带宽可言")
    lo = float(np.min(v))
    hi = float(np.max(v))
    return Band(
        lo=lo,
        hi=hi,
        amplitude=hi - lo,
        midpoint=0.5 * (lo + hi),
        sup_from_start=float(np.max(np.abs(v - v[0]))),
        n=int(v.size),
    )


def relative_variation(series: Any) -> float:
    r""":math:`(\max v-\min v)/|v_0|`——"这条曲线变了多少"的无量纲版本。

    ``v_0`` 恰为 0 时退化为绝对带宽（并在调用方用 ``strings`` 说明）。
    """
    b = band(series)
    v0 = abs(float(np.asarray(series, dtype=float).ravel()[0]))
    if v0 == 0.0:
        # v₀=0 时相对量没有定义：按 docstring 的约定退化为**绝对带宽**，
        # 由调用方在 strings 里说明（原来这里直接 ZeroDivisionError，与 docstring 自相矛盾）。
        return float(b.amplitude)
    return float(b.amplitude / v0)


def drift_slope(t: Any, series: Any, *, relative_to: float | None = None) -> float:
    r"""最小二乘线性趋势 :math:`a`（``series ≈ a·t + b``，单位 1/时间）。

    **不**除以带宽、也**不**声称"非零就是漂移"：有界振荡上这个数也会非零
    （混叠），所以配套判据是 :func:`drift_over_band`。
    """
    tt = np.asarray(t, dtype=float).ravel()
    vv = np.asarray(series, dtype=float).ravel()
    if tt.size != vv.size:
        raise ValueError(f"t 与序列长度不同：{tt.size} vs {vv.size}")
    if tt.size < 3:
        raise ValueError("至少 3 个点才能谈趋势")
    tm = float(np.mean(tt))
    vm = float(np.mean(vv))
    sxx = float(np.sum((tt - tm) ** 2))
    if sxx == 0.0:
        raise ValueError("所有时刻相同，无法拟合趋势")
    a = float(np.sum((tt - tm) * (vv - vm)) / sxx)
    return a / float(relative_to) if relative_to else a


def half_split(t: Any, series: Any) -> tuple[float, float]:
    """前半段与后半段的趋势（1/时间）。**这才是"有没有长期漂移"的直接对照**：

    有界振荡的两半斜率量级相当且都不足以在观测时长内走出带宽；
    线性漂移的后半段斜率与前半段同号同量级，而"漂移/带宽"比随时间线性增长。
    """
    tt = np.asarray(t, dtype=float).ravel()
    vv = np.asarray(series, dtype=float).ravel()
    if tt.size < 6:
        raise ValueError("至少 6 个点才能前后对半")
    mid = tt.size // 2
    return drift_slope(tt[:mid], vv[:mid]), drift_slope(tt[mid:], vv[mid:])


def band_growth(series: Any) -> float:
    r"""后半段 :math:`\sup|v-v_0|` 与前半段之比——**"有界 vs 漂移"最稳的那个判据**。

    为什么不用"趋势/带宽"当主判据：最小二乘斜率给后面的时刻更大权重，**纯振荡**在
    长度 :math:`N` 个周期的窗口上也会给出 :math:`\approx 3/(\pi N)` 的比值
    （实测 sin 在 :math:`[0,4\pi]`（N=2）上是 0.477、:math:`[0,8\pi]`（N=4）上是 0.237，
    与 :math:`3/(\pi N)` 逐位吻合）；采样点与振荡周期不同步时还有混叠，也会造出假趋势。
    而这个比值对混叠免疫（两半受同样的影响），且直接对应理论：

    * **有界振荡**（辛方法）：两半的 :math:`\sup` 都是同一个带宽 ⟹ 比值 :math:`pprox 1`；
    * **线性漂移**（RK4、显式 Euler）：:math:`\sup` 随时间线性长 ⟹ 比值 :math:`pprox 2`
      （观测时长翻倍就是 2，再翻就是 2 的更高次——这正是"漂移"的定义）。
    """
    v = np.asarray(series, dtype=float).ravel()
    if v.size < 4:
        raise ValueError("至少 4 个点")
    v0 = float(v[0])
    half = v.size // 2
    first = float(np.max(np.abs(v[:half] - v0)))
    second = float(np.max(np.abs(v[half:] - v0)))
    if first == 0.0:
        return math.inf if second > 0.0 else 1.0
    return float(second / first)


def drift_over_band(t: Any, series: Any) -> tuple[float, float]:
    r"""返回 ``(总趋势, 趋势/带宽)``：总趋势 = :math:`|a|\cdot(t_{\max}-t_{\min})`。

    判据：**趋势/带宽 ≪ 1** 才能叫"有界振荡"。**前提是窗口里有很多个周期**——
    最小二乘斜率给后面的时刻更大的权重，窗口不对称时振荡本身就会贡献一个非零斜率：
    实测 sin 在恰好两个整周期 :math:`[0,4\pi]` 上趋势/带宽 = 0.47（不是漂移，是窗口的锅），
    而在对称窗口 :math:`[-4\pi,4\pi]` 上到舍入级。本章的窗口都是 10^3–10^6 步
    （10^2–10^3 个周期），所以这个数才读得成"有没有长期漂移"。

    实测（谐振子、h=0.05、10^6 步、4000 采样）：

    * 辛 Euler：带宽 2.56e-2、总趋势 4.6e-4 ⟹ 比值 1.8%（带宽由 :math:`h/(2-h)` 精确给出）；
    * Verlet：带宽 6.25e-4、总趋势 3.1e-6 ⟹ 0.5%（带宽 = :math:`h^2/4`）；
    * RK4：0.05 步长下能量在 10^6 步内**单调**掉了 2.17e-4（= :math:`h^5T/72`），
      趋势/带宽 = 100%——它是"漂移"，不是"振荡"。
    """
    tt = np.asarray(t, dtype=float).ravel()
    vv = np.asarray(series, dtype=float).ravel()
    a = drift_slope(tt, vv)
    horizon = float(tt[-1] - tt[0])
    trend = abs(a) * horizon
    amp = band(vv).amplitude
    return trend, float(trend / amp if amp > 0.0 else math.inf)


# --------------------------------------------------------------------------- #
# 2. 幂律
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PowerLaw:
    r""":math:`y\approx C h^p` 的两参数拟合：``exponent`` 是 :math:`p`，``constant`` 是 :math:`C`。

    ``log_residual`` 是双对数残差的最大绝对值——它比 :math:`R^2` 更能暴露
    "窗口没进渐近区"（对数尺度上 :math:`R^2` 几乎总是 0.999 以上）。
    """

    exponent: float
    constant: float
    log_residual: float
    n: int

    def predict(self, h: float) -> float:
        return float(self.constant * float(h) ** self.exponent)


def power_law(h: Any, y: Any, *, skip_tiny: float | None = None) -> PowerLaw:
    r"""拟合 :math:`y=Ch^p`：:math:`p` 用 :func:`numcore.ode.rk.loglog_slope`（同一口径），
    :math:`C` 取对数残差的指数均值。

    ``skip_tiny`` 给一个下限（例如机器精度量级），低于它的点先剔除——舍入地板上的
    点会把斜率拉平，这是 :mod:`numcore.verify` 的 8.4 节 的同一条纪律。
    """
    hh = np.asarray(h, dtype=float).ravel()
    yy = np.asarray(y, dtype=float).ravel()
    if hh.size != yy.size:
        raise ValueError(f"步长与误差长度不符：{hh.size} vs {yy.size}")
    if hh.size < 3:
        raise ValueError(f"拟合至少 3 个点，得到 {hh.size}")
    keep = np.isfinite(hh) & np.isfinite(yy) & (hh > 0.0) & (yy > 0.0)
    if skip_tiny is not None:
        keep &= yy > float(skip_tiny)
    hh, yy = hh[keep], yy[keep]
    p = loglog_slope(hh, yy)
    resid = np.log(yy) - (np.log(hh) * p)
    c = float(np.exp(np.mean(resid)))
    return PowerLaw(
        exponent=float(p),
        constant=c,
        log_residual=float(np.max(np.abs(resid - np.mean(resid)))),
        n=int(hh.size),
    )


# --------------------------------------------------------------------------- #
# 3. 相空间轨道的"闭合"到底怎么量
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RadiusStats:
    r"""相空间半径 :math:`|y|=\sqrt{|q|^2+|p|^2}` 的统计。

    ``drift`` 是末段均值减首段均值（相对 :math:`|y_0|`）：螺旋的指纹。
    ``band`` 是整段的 :math:`\max|y|-\min|y|`（相对）：**有界振荡**的指纹。
    两者必须一起看——只看 ``drift`` 会把"绕椭圆转"误判成"漂移"。
    """

    r0: float
    r_min: float
    r_max: float
    band: float
    drift: float
    n: int


def radius_stats(q: Any, p: Any, *, tail: float = 0.1) -> RadiusStats:
    """相空间半径的首尾漂移与带宽（``tail`` 是首/末段占整段的比例）。"""
    qq = np.asarray(q, dtype=float)
    pp = np.asarray(p, dtype=float)
    if qq.ndim == 1:
        qq = qq[:, None]
        pp = pp[:, None]
    r = np.sqrt(np.sum(qq * qq + pp * pp, axis=1))
    n = int(r.size)
    if n < 4:
        raise ValueError("至少 4 个点")
    r0 = float(r[0])
    k = max(1, int(round(tail * n)))
    drift = (float(np.mean(r[-k:])) - float(np.mean(r[:k]))) / r0
    return RadiusStats(
        r0=r0,
        r_min=float(np.min(r)),
        r_max=float(np.max(r)),
        band=float((np.max(r) - np.min(r)) / r0),
        drift=float(drift),
        n=n,
    )


def wrapped_angle(x: float) -> float:
    r"""把角度折到 :math:`(-\pi,\pi]`（相位差的唯一正确读法）。"""
    y = float(x) % (2.0 * math.pi)
    if y > math.pi:
        y -= 2.0 * math.pi
    return y


def phase_error_turns(q: Any, p: Any, t: Any, *, omega: float = 1.0, theta0: float = 0.0) -> float:
    r"""谐振子数值解的**相位误差**（单位：圈），:math:`\theta=\mathrm{atan2}(p,q)`。

    精确流以角速度 :math:`\omega` 匀速转（:math:`\theta_{exact}=\theta_0-\omega t`，
    本章的初值 :math:`(1,0)` 取 :math:`\theta_0=0`），数值流的角速度有 :math:`O(h^p)`
    偏差，于是相位误差随时间**线性累积**。这就是"轨道闭合"与"第 n 圈回原点"的分界：
    辛方法的点集闭合（不变量守恒），但相位照样漂。

    返回的是**累积**圈数：先把 :math:`\theta` 解开缠绕（:func:`np.unwrap`）再作差。
    不这么做就会把它折进半圈以内——0.25 圈的真误差与 −0.75、+1.25 圈再也分不出来。
    每步转过 :math:`|\omega|\Delta t\ge\pi` 时缠绕无法唯一确定，直接抛 ``ValueError``。
    """
    qq = np.asarray(q, dtype=float).ravel()
    pp = np.asarray(p, dtype=float).ravel()
    tt = np.asarray(t, dtype=float).ravel()
    if qq.size != tt.size or pp.size != tt.size:
        raise ValueError(f"q/p/t 长度必须一致：{qq.size}/{pp.size}/{tt.size}")
    if tt.size < 2:
        raise ValueError("至少两个采样点")
    step = float(np.max(np.abs(np.diff(tt))))
    if abs(omega) * step >= math.pi - 1e-9:
        raise ValueError(
            f"采样太稀：每步转过 {abs(omega) * step:.6f} rad ≥ π，相位缠绕无法唯一确定"
            f"（要么加密采样，要么只报首尾未缠绕的短窗口）"
        )
    th = np.arctan2(pp, qq)
    # **必须**先解开缠绕再作差：只取末点的 atan2 差值会把它折进 (−π,π]，
    # 于是 0.2503 圈的真误差与 −0.7497 圈、+1.2503 圈再也分不出来（实测踩过：
    # 反向流的 +6.366 圈被报成 +0.366 圈）。
    th_un = np.unwrap(th)
    turned = float(th_un[-1] - th_un[0])
    exact_turned = -omega * float(tt[-1] - tt[0]) + (theta0 - theta0)
    return float((turned - exact_turned) / (2.0 * math.pi))


@dataclass(frozen=True)
class ReturnStats:
    r"""逐圈的**回归距离**：``returns[k]`` = 第 k+1 个周期窗口内离**出发点**最近的距离（相对 :math:`|y_0|`）。

    **这是采样受限的量，不能单独当证据**：轨道是连续曲线而采样点离散，
    「最近距离」实际是「离出发点最近的那个采样点到出发点的距离」。
    实测（谐振子 h=0.1、600 圈、6000 个采样点）：Verlet 的这个数从前半段的 1.9e-2
    长到后半段的 3.4e-1，看起来像「轨道散开了」；换成 :func:`curve_closure`
    （到不变椭圆的径向偏差）同一组数据给出 1.2e-3 且前后半段不变。
    本章的「闭合」结论一律以 :func:`curve_closure` / :func:`reference_closure` 为准，
    这个函数保留下来专门用于说明那个**假象**是怎么来的。
    """

    returns: np.ndarray
    first: float
    last: float
    growth: float
    n_cycles: int
    period: float
    horizon: float


def return_distances(
    q: Any,
    p: Any,
    t: Any,
    period: float,
    *,
    window: float = 0.25,
) -> ReturnStats:
    """按周期把观测时长切窗，量每窗内到出发点的最近距离（相对 :math:`|y_0|`，采样受限）。"""
    qq = np.asarray(q, dtype=float)
    pp = np.asarray(p, dtype=float)
    if qq.ndim == 1:
        qq = qq[:, None]
        pp = pp[:, None]
    tt = np.asarray(t, dtype=float).ravel()
    y0 = np.concatenate([qq[0], pp[0]])
    scale = float(np.sqrt(np.sum(y0 * y0)))
    if not period > 0.0:
        raise ValueError(f"周期必须为正，得到 {period}")
    n_cycles = int(math.floor(float(tt[-1]) / period))
    if n_cycles < 1:
        raise ValueError(f"观测时长 {tt[-1]:g} 不足一个周期 {period:g}")
    out = np.zeros(n_cycles, dtype=float)
    for k in range(n_cycles):
        centre = (k + 1) * period
        lo = centre - window * period
        hi = centre + window * period
        sel = (tt >= lo) & (tt <= hi)
        if not bool(np.any(sel)):
            # 采样太稀：退回「离窗口中心最近的那个采样点」，并如实标出（值会偏大）
            j = int(np.argmin(np.abs(tt - centre)))
            sel = np.zeros(tt.size, dtype=bool)
            sel[j] = True
        d = np.sqrt(np.sum((qq[sel] - y0[0]) ** 2 + (pp[sel] - y0[1]) ** 2, axis=1))
        out[k] = float(np.min(d)) / scale
    first = float(out[0])
    last = float(out[-1])
    growth = float(last / first) if first > 0.0 else math.inf
    return ReturnStats(
        returns=out,
        first=first,
        last=last,
        growth=growth,
        n_cycles=n_cycles,
        period=float(period),
        horizon=float(tt[-1] - tt[0]),
    )


@dataclass(frozen=True)
class ClosureStats:
    r"""轨道到**参考曲线**的径向相对偏差 :math:`\delta_k=r_k/r_{\mathrm{ref}}(\theta_k)-1` 的汇总。

    * ``sup`` 是 :math:`\sup_k|\delta_k|`——"闭合误差"的那个数；
    * ``first_half`` / ``second_half`` 是前后半段的 :math:`\sup`，``growth`` 是比值。

    判据（与 :func:`curve_closure` 的 docstring 同一口径）：
    点集闭合时两半同量级（``growth`` 是 :math:`O(1)`，且 ``sup`` 已是舍入或 :math:`O(h^p)` 带宽量级）；
    螺旋时 ``growth`` 与观测时长同量级。**观测时长必须与 ``sup`` 一起报出**。
    """

    deviation: np.ndarray
    sup: float
    first_half: float
    second_half: float
    growth: float
    n: int
    at_roundoff: bool


def closure_stats(deviation: Any, *, roundoff: float | None = None) -> ClosureStats:
    r"""把一条径向偏差序列汇总成 :class:`ClosureStats`（``at_roundoff`` 标记舍入级）。"""
    dev = np.asarray(deviation, dtype=float).ravel()
    if dev.size < 4:
        raise ValueError("至少 4 个点")
    finite = dev[np.isfinite(dev)]
    if finite.size == 0:
        raise ValueError("偏差序列全是非有限值")
    sup = float(np.max(np.abs(finite)))
    half = dev.size // 2
    fh = float(np.max(np.abs(dev[:half]))) if half else float("nan")
    sh = float(np.max(np.abs(dev[half:]))) if dev.size - half else float("nan")
    thr = 1e-12 if roundoff is None else float(roundoff)
    growth = float(sh / fh) if fh > 0.0 else math.inf
    return ClosureStats(
        deviation=dev,
        sup=sup,
        first_half=fh,
        second_half=sh,
        growth=growth,
        n=int(dev.size),
        at_roundoff=bool(sup <= thr),
    )


def radial_deviation(q: Any, p: Any, r_ref: Any) -> np.ndarray:
    r""":math:`\delta_k=r_k/r_{\mathrm{ref},k}-1`（参考半径由调用方按同一极角给出）。"""
    qq = np.asarray(q, dtype=float)
    pp = np.asarray(p, dtype=float)
    if qq.ndim > 1:
        qq = np.sqrt(np.sum(qq * qq, axis=1))
        pp = np.sqrt(np.sum(pp * pp, axis=1))
        r = np.sqrt(qq * qq + pp * pp)
    else:
        r = np.sqrt(qq * qq + pp * pp)
    ref = np.asarray(r_ref, dtype=float).ravel()
    if ref.size != r.size:
        raise ValueError(f"参考半径长度不符：{ref.size} vs {r.size}")
    if bool(np.any(ref <= 0.0)):
        raise ValueError("参考半径必须为正")
    return r / ref - 1.0


def level_curve_radius(Q: Any, theta: Any, value: float) -> np.ndarray:
    r"""二次型等值线 :math:`\tfrac12[q,p]Q[q,p]^{\mathsf T}=\text{value}` 在极角 :math:`\theta` 处的半径。

    :math:`r(\theta)=\sqrt{2\,\text{value}\big/(Q_{00}\cos^2\theta+2Q_{01}\cos\theta\sin\theta+Q_{11}\sin^2\theta)}`。
    正定（星形）情形处处有定义；不定情形分母变号，返回 ``inf`` 由调用方处理。
    """
    Qm = np.asarray(Q, dtype=float)
    th = np.asarray(theta, dtype=float)
    ct = np.cos(th)
    st = np.sin(th)
    den = Qm[0, 0] * ct * ct + 2.0 * Qm[0, 1] * ct * st + Qm[1, 1] * st * st
    num = 2.0 * float(value)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.sqrt(num / den)
    return out


def curve_closure(Q: Any, q: Any, p: Any) -> ClosureStats:
    r"""轨道到**二次型等值线**的径向相对偏差——本章「轨道闭合」的正式判据。

    等值线取过出发点的水平集 :math:`H_h=\mathrm{const}`（:math:`Q` 由
    :func:`quadratic_invariant` 从 2×2 映射给出）。判据：

    * **点集闭合**（辛映射）：:math:`\sup_k|\delta_k|` 落在舍入级
      （``at_roundoff=True``，实测 1e-15…3e-14）——不变量沿线守恒，
      所以轨道**逐点**落在同一条曲线上；
    * **螺旋**（非辛映射）：:math:`\delta_k` 单调长大，``growth`` 与观测时长同量级
      （实测 RK4 在 600 圈里 :math:`\sup` 从前半段 1.3e-4 长到后半段 2.6e-4，比值 2.00）。

    注意这个量的**语义**：它证明的是「辛方法的轨道落在同一条闭合曲线上」，
    而**不是**「数值轨道离真实轨道很近」——后者的量级是 :math:`O(h^p)` 的带宽，
    要用 :func:`reference_closure` 与精确轨道比。
    """
    Qm = np.asarray(Q, dtype=float)
    qq = np.asarray(q, dtype=float).ravel()
    pp = np.asarray(p, dtype=float).ravel()
    if qq.size != pp.size:
        raise ValueError(f"q 与 p 长度不同：{qq.size} vs {pp.size}")
    val = 0.5 * float(Qm[0, 0] * qq[0] * qq[0] + 2.0 * Qm[0, 1] * qq[0] * pp[0] + Qm[1, 1] * pp[0] * pp[0])
    if val == 0.0:
        raise ValueError("出发点落在等值线 H_h=0 上，径向偏差没有定义")
    theta = np.arctan2(pp, qq)
    rc = level_curve_radius(Qm, theta, val)
    return closure_stats(radial_deviation(qq, pp, rc))


def reference_closure(q: Any, p: Any, r_ref: Any) -> ClosureStats:
    r"""轨道到**精确参考曲线**（解析给出的半径 :math:`r_{\mathrm{ref}}(	heta)`）的径向偏差。

    与 :func:`curve_closure` 的分工：那一个量的是「辛方法的不变量守恒」，
    这一个量的是「数值轨道离真实轨道多远」——辛方法是 :math:`O(h^p)` 的**有界**带宽
    （谐振子实测：辛 Euler h/(2−h)=2.6e-2、Verlet h²/8=1.25e-3、隐式中点 2e-14），
    非辛方法的这个偏差**随时间线性长大**（RK4 在 600 圈里 2.6e-4，前后半段翻倍）。
    """
    rr = np.asarray(r_ref, dtype=float).ravel()
    return closure_stats(radial_deviation(np.asarray(q, dtype=float).ravel(), np.asarray(p, dtype=float).ravel(), rr))


# --------------------------------------------------------------------------- #
# 4. 二次不变量（辛映射的指纹）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QuadraticInvariant:
    r"""2×2 映射 :math:`M` 的二次型不变量 :math:`H_h=\tfrac12[q,p]Q[q,p]^{\mathsf T}`。

    对 :math:`\det M=1` 的 2×2 矩阵，:math:`Q` 有**闭式**（相差一个标量）：

    .. math::
        Q=\begin{pmatrix} c & (d-a)/2\\ (d-a)/2 & -b\end{pmatrix},
        \qquad M=\begin{pmatrix}a&b\\c&d\end{pmatrix}.

    ``residual`` 是 :math:`\|M^{\mathsf T}QM-Q\|_\infty/\|Q\|_\infty`（相对量）：
    ``symplectic=True`` 时它是舍入级（实测 ≤ 8u），非辛时按 :math:`h` 的幂次长大——
    "不存在二次不变量"这件事因此有了**数字**，而不是一句断言。
    """

    Q: np.ndarray
    det: float
    symplectic: bool
    residual: float
    mixing: float
    normalized: np.ndarray

    def value(self, q: float, p: float) -> float:
        """不变量在 :math:`(q,p)` 处的值（含 1/2 因子）。"""
        return 0.5 * float(self.Q[0, 0] * q * q + 2.0 * self.Q[0, 1] * q * p + self.Q[1, 1] * p * p)


def quadratic_invariant(M: Any, *, tol: float | None = None) -> QuadraticInvariant:
    """由 2×2 映射给出二次不变量（闭式，不调用任何线性代数例程）。"""
    A = np.asarray(M, dtype=float)
    if A.shape != (2, 2):
        raise ValueError(f"需要 2×2 矩阵，得到 {A.shape}")
    a, b = float(A[0, 0]), float(A[0, 1])
    c, d = float(A[1, 0]), float(A[1, 1])
    det = a * d - b * c
    Q = np.array([[c, 0.5 * (d - a)], [0.5 * (d - a), -b]], dtype=float)
    QtQ = A.T @ Q @ A
    qn = float(np.max(np.abs(Q)))
    res = float(np.max(np.abs(QtQ - Q)) / qn) if qn > 0.0 else math.inf
    thr = 8.0 * U if tol is None else float(tol)
    q00 = float(Q[0, 0])
    norm = Q / q00 if q00 != 0.0 else Q / (-float(Q[1, 1]) if float(Q[1, 1]) != 0.0 else 1.0)
    return QuadraticInvariant(
        Q=Q,
        det=float(det),
        symplectic=bool(abs(det - 1.0) <= thr),
        residual=res,
        mixing=float(norm[0, 1]),
        normalized=norm,
    )


def invariant_series(Q: Any, q: Any, p: Any) -> np.ndarray:
    r"""沿轨迹求 :math:`H_h` 序列（辛映射上它应当逐点相等）。"""
    Qm = np.asarray(Q, dtype=float)
    qq = np.asarray(q, dtype=float).ravel()
    pp = np.asarray(p, dtype=float).ravel()
    if qq.size != pp.size:
        raise ValueError(f"q 与 p 长度不同：{qq.size} vs {pp.size}")
    return 0.5 * (Qm[0, 0] * qq * qq + 2.0 * Qm[0, 1] * qq * pp + Qm[1, 1] * pp * pp)


def invariant_level_curve(Q: Any, q0: float, p0: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    r"""过 :math:`(q_0,p_0)` 的二次型等值线（相空间图上的"闭合轨道"参考线）。

    用 :math:`H_h` 的水平集参数化：把二次型配方成
    :math:`\lambda_+u^2+\lambda_-v^2=\mathrm{const}`，再按椭圆参数采样。
    椭圆（:math:`\lambda` 同号）才有闭合曲线；双曲情形退回空数组，由调用方如实标注。
    """
    Qm = np.asarray(Q, dtype=float)
    a = float(Qm[0, 0])
    b = float(Qm[0, 1])
    c = float(Qm[1, 1])
    if a == 0.0:
        a = 1e-300
    # 二次型 (a q² + 2b qp + c p²) 的配方：迹 t=a+c、判别式 disc=sqrt((a-c)²+4b²)
    t = a + c
    disc = math.sqrt((a - c) ** 2 + 4.0 * b * b)
    lam_plus = 0.5 * (t + disc)
    lam_minus = 0.5 * (t - disc)
    val = 0.5 * (a * q0 * q0 + 2.0 * b * q0 * p0 + c * p0 * p0)
    if val == 0.0 or lam_plus * lam_minus <= 0.0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    th = np.linspace(0.0, 2.0 * math.pi, int(n))
    u = math.sqrt(2.0 * val / lam_plus) * np.cos(th)
    v = math.sqrt(2.0 * val / lam_minus) * np.sin(th)
    # 特征向量：λ 对应 (b, λ−a) 或 (λ−c, b) 中**范数更大**的那个
    # （两支都退化时说明 λ 与对角元重合，取坐标轴）。用 (b, λ−a) 会撞上
    # "两个几乎相等的数相减"：隐式中点的 Q 只有 5.6e-17 的非对角元，
    # 直接套公式会得到一根长度 1e-16 的"特征向量"，等值线随之完全画错。
    scale_q = max(abs(a), abs(c), 1e-300)
    if abs(b) <= 1e-12 * scale_q:
        if a >= c:
            e1 = np.array([1.0, 0.0])
            e2 = np.array([0.0, 1.0])
        else:
            e1 = np.array([0.0, 1.0])
            e2 = np.array([1.0, 0.0])
    else:
        e1 = np.array([b, lam_plus - a], dtype=float)
        if float(np.hypot(*e1)) < abs(lam_plus - c):
            e1 = np.array([lam_plus - c, b], dtype=float)
        e2 = np.array([b, lam_minus - a], dtype=float)
        if float(np.hypot(*e2)) < abs(lam_minus - c):
            e2 = np.array([lam_minus - c, b], dtype=float)
    e1 = e1 / math.sqrt(float(e1 @ e1))
    e2 = e2 / math.sqrt(float(e2 @ e2))
    qq = u * e1[0] + v * e2[0]
    pp = u * e1[1] + v * e2[1]
    return qq, pp


# --------------------------------------------------------------------------- #
# 5. 动量与角动量
# --------------------------------------------------------------------------- #
def momentum_drift(momentum: Any) -> float:
    r"""线性动量的最大相对漂移 :math:`\max_k|P_k-P_0|/\max(1,|P_0|)`。

    **警告**：任何 RK 方法（含 RK4 与显式 Euler）都精确保持线性不变量，
    所以这个数落在机器精度**不能**作为辛性的证据（本章 pitfalls 的第一条反例）。
    """
    v = np.asarray(momentum, dtype=float).ravel()
    if v.size == 0:
        raise ValueError("空序列")
    return float(np.max(np.abs(v - v[0])) / max(1.0, abs(float(v[0]))))


def angular_momentum_drift(angmom: Any) -> float:
    r"""角动量的最大**相对**漂移（:math:`L_0\\approx0` 时退化为绝对差）。

    与线性动量相反，角动量是二次不变量：径向 kick 的辛方法（Verlet/leapfrog）逐位保它，
    RK4 不保。实测（Kepler e=0.6、h=0.01、200 圈）：Verlet 1.9e-14、RK4 4.8e-7。
    """
    v = np.asarray(angmom, dtype=float).ravel()
    if v.size == 0:
        raise ValueError("空序列")
    scale = abs(float(v[0]))
    return float(np.max(np.abs(v - v[0])) / scale) if scale > 0.0 else float(np.max(np.abs(v - v[0])))


def monotone_fraction(series: Any) -> float:
    r""":math:`|\text{净变化}|/\text{总变差}`：1 表示**单调**（纯漂移），→0 表示来回振荡。

    这是"漂移 vs 振荡"的第二个独立判据（第一个是 :func:`drift_over_band`）：
    辛方法的能量误差来回走 ⟹ 该比值小；RK4 的能量单调掉 ⟹ 该比值 ≈ 1。
    """
    v = np.asarray(series, dtype=float).ravel()
    if v.size < 2:
        raise ValueError("至少 2 个点")
    total = float(np.sum(np.abs(np.diff(v))))
    if total == 0.0:
        return 0.0
    return float(abs(v[-1] - v[0]) / total)
