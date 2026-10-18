r"""3.1 章（下半）：**步长控制律**、误差估计可靠性与工作-精度前沿。

数学对象
--------

一步的误差估计 :math:`\hat e_n` 经加权 RMS 度量化成无量纲的 :math:`\epsilon_n`
（:math:`\epsilon_n\le1` 才接受）。控制律是一张「下一步走多大」的映射：

.. math::
    h_{n+1}=h_n\,\mathrm{fac},\qquad
    \mathrm{fac}=\mathrm{clip}\Bigl(\mathrm{safe}\cdot\epsilon_n^{-\alpha}
    \Bigl(\tfrac{\epsilon_{n-1}}{\epsilon_n}\Bigr)^{\beta},\;
    f_{\min},\;f_{\max}\Bigr)

* **I 控制器**（初等）:math:`\alpha=1/(\hat p+1)`、:math:`\beta=0`——只用当前误差；
* **PI 控制器**（Gustafsson）:math:`\alpha=\tfrac1{\hat p+1}-\tfrac34\beta`、
  :math:`\beta=0.04`（Hairer--Wanner 的 DOPRI5 缺省）——用上一步的误差**预期**
  下一步会变大还是变小，从而抑制步长序列的周期-2 振荡。

:math:`\hat p` 是**误差估计量**的阶（嵌入对里低阶解满足的阶条件条数），不是被推进
解的阶。把这两个 :math:`p` 用混，是本章最贵的一个错：指数写错一次，工作点就被
搬到别处（``tests/3.1`` 里对两种指数各测了一条曲线）。

**估计器可靠性**：:math:`r_n=\hat e_n/e_n`（估计 / 真局部误差）。
:math:`r_n\ll1` 表示估计器**乐观**（控制器以为很准，其实不然）；:math:`r_n\gg1` 表示悲观
（白花工作量）。:math:`r_n` 在所有步上都接近 1 才叫「估计器可信」，
而这只在阶条件匹配、解足够光滑、问题不刚的时候成立。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from numcore.ode import rk as rkmod

__all__ = [
    "ControlLaw",
    "AdaptiveRun",
    "WorkPrecision",
    "elementary_factor",
    "i_factor",
    "pi_factor",
    "controller_factor",
    "next_step",
    "integrate_adaptive",
    "estimator_ratio",
    "estimator_study",
    "true_local_errors",
    "work_precision",
    "work_at_accuracy",
    "step_ratio_oscillation",
]


# --------------------------------------------------------------------------- #
# 1. 控制律
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ControlLaw:
    r"""一个步长控制器：``name`` 取 ``"i"``（初等）或 ``"pi"``（Gustafsson PI）。

    ``safety < 1`` 是必须的：误差估计只是渐近意义下的近似，踩在 :math:`\epsilon=1`
    上走迟早会被拒绝。``fac_min``/``fac_max`` 是每步允许的放大/缩小倍数上限——
    没有它们，估计量偶尔穿零会让步长一步跳到无穷。
    """

    name: str = "pi"
    safety: float = 0.9
    fac_min: float = 0.2
    fac_max: float = 5.0
    beta: float = 0.04

    def exponent(self, order_low: int) -> float:
        r"""误差-步长幂次 :math:`\alpha`：I 控制器用 :math:`1/(\hat p+1)`。"""
        p = int(order_low)
        if p < 1:
            raise ValueError(f"误差估计量的阶至少为 1，得到 {order_low}")
        if self.name == "i":
            return 1.0 / (p + 1)
        if self.name == "pi":
            return 1.0 / (p + 1) - 0.75 * self.beta
        raise ValueError(f"未知控制器 {self.name!r}（可用：['i', 'pi']）")


def elementary_factor(
    error_measure: float,
    exponent: float,
    *,
    safety: float = 0.9,
    fac_min: float = 0.2,
    fac_max: float = 5.0,
) -> float:
    r""":math:`\mathrm{fac}=\mathrm{clip}(\mathrm{safe}\cdot\epsilon^{-\alpha})`（纯幂律 + 夹子）。

    :math:`\epsilon` 是**归一化**误差（:math:`\epsilon=1` 恰好等于容差）；非有限输入直接抛错，
    负输入也抛错（只可能是上游把符号弄反了）。:math:`\epsilon=0` 是**合法**的
    （例如一步就能精确积分的线性问题），此时因子无界，只能给上夹子——这是唯一允许静默裁剪的
    一支，而且方向是"把步长放大"而不是"原地不动"。
    """
    if not math.isfinite(error_measure):
        raise ValueError(f"误差度量必须是有限数，得到 {error_measure}")
    if error_measure < 0.0:
        raise ValueError(f"误差度量不能为负，得到 {error_measure}")
    if error_measure == 0.0:
        fac = float(fac_max)
    else:
        with np.errstate(over="ignore", under="ignore", divide="ignore"):
            fac = float(safety) * float(error_measure) ** (-float(exponent))
    return float(min(max(fac, float(fac_min)), float(fac_max)))


def i_factor(
    error_measure: float,
    order_low: int,
    *,
    safety: float = 0.9,
    fac_min: float = 0.2,
    fac_max: float = 5.0,
) -> float:
    r"""初等（I）控制器：:math:`h_{new}=h\,\mathrm{safe}\,(\mathrm{tol}/\mathrm{err})^{1/(\hat p+1)}`。

    这里 :math:`\epsilon=\mathrm{err}/\mathrm{tol}`，于是
    :math:`\mathrm{fac}=\mathrm{safe}\cdot\epsilon^{-1/(\hat p+1)}`——
    与教科书里那个「把误差推到容差上」的式子逐字相同。
    """
    return elementary_factor(
        error_measure, 1.0 / (int(order_low) + 1), safety=safety, fac_min=fac_min, fac_max=fac_max
    )


def pi_factor(
    error_measure: float,
    error_previous: float,
    order_low: int,
    *,
    safety: float = 0.9,
    beta: float = 0.04,
    fac_min: float = 0.2,
    fac_max: float = 5.0,
) -> float:
    r"""PI 控制器：:math:`\mathrm{fac}=\mathrm{safe}\,\epsilon_n^{-\alpha}(\epsilon_{n-1}/\epsilon_n)^{\beta'}`。

    缺省 :math:`\beta'=\tfrac34\beta=0.03`、:math:`\alpha=\tfrac1{\hat p+1}-\beta'`
    （Hairer--Wanner DOPRI5 的 expo1/expo2）。上一步误差偏大时这一项会**提前**放大步长，
    从而少几次「先拒绝再缩小」的来回。
    """
    p = int(order_low)
    b = 0.75 * float(beta)
    alpha = 1.0 / (p + 1) - b
    if not math.isfinite(error_measure) or not math.isfinite(error_previous):
        raise ValueError("误差度量必须是有限数")
    eps = float(error_measure)
    prev = float(error_previous)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        if eps <= 0.0:
            fac = float(fac_max)
        else:
            fac = float(safety) * eps ** (-alpha)
            if prev > 0.0:
                fac *= (prev / eps) ** b
    return float(min(max(fac, float(fac_min)), float(fac_max)))


def controller_factor(
    law: ControlLaw,
    error_measure: float,
    error_previous: float | None,
    order_low: int,
) -> float:
    """按控制器类型分派（``error_previous=None`` 时退化成初等律，第一步就该如此）。"""
    if law.name == "i" or error_previous is None:
        return elementary_factor(
            error_measure,
            law.exponent(order_low),
            safety=law.safety,
            fac_min=law.fac_min,
            fac_max=law.fac_max,
        )
    return pi_factor(
        error_measure,
        error_previous,
        order_low,
        safety=law.safety,
        beta=law.beta,
        fac_min=law.fac_min,
        fac_max=law.fac_max,
    )


def next_step(
    law: ControlLaw,
    h: float,
    error_measure: float,
    error_previous: float | None,
    order_low: int,
) -> float:
    r"""由当前步长与误差给出下一步长 :math:`h_{new}=h\cdot\mathrm{fac}`（已夹住）。

    :math:`h\le0` 或非有限直接抛错：控制器是**乘性**的，负步长会让积分倒着走而
    不报任何错——这类输入必须挡在入口。
    """
    hh = float(h)
    if not math.isfinite(hh) or hh <= 0.0:
        raise ValueError(f"步长必须是正的有限数，得到 {h}")
    fac = controller_factor(law, error_measure, error_previous, order_low)
    return hh * fac


# --------------------------------------------------------------------------- #
# 2. 自适应积分
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AdaptiveRun:
    r"""一次自适应积分的完整账本（**每一次试探**都记，不只是接受的那些）。

    为什么「拒绝也要留痕」：接受/拒绝比是控制器好坏的直接证据，
    而只保存接受点的实现会把「被拒绝的步」从历史里抹掉——那正是「看起来很稳」的来源。
    """

    method: str
    problem: str
    tol: float
    atol: float
    law: str
    local_extrapolation: bool
    t: np.ndarray
    y: np.ndarray
    t_attempt: np.ndarray
    h_attempt: np.ndarray
    error_attempt: np.ndarray
    accepted: np.ndarray
    error_true_attempt: np.ndarray
    max_error: float
    endpoint_error: float
    error_curve: np.ndarray
    n_accepted: int
    n_rejected: int
    n_f_evals: int
    n_extra_f_evals: int
    ok: bool
    blowup: bool
    t_blowup: float
    order_propagated: int
    order_estimate: int

    def accept_ratio(self) -> float:
        """接受步数 / 试探步数（1.0 表示一次都没拒绝过）。"""
        n = int(self.accepted.size)
        return float(np.count_nonzero(self.accepted)) / n if n else float("nan")

    def h_accepted(self) -> np.ndarray:
        return np.asarray(self.h_attempt, dtype=float)[np.asarray(self.accepted, dtype=bool)]

    def error_accepted(self) -> np.ndarray:
        return np.asarray(self.error_attempt, dtype=float)[np.asarray(self.accepted, dtype=bool)]

    def work_per_time(self) -> float:
        """:math:`f` 求值次数 / 时间跨度（含被拒绝步与额外求值）。"""
        span = float(self.t[-1] - self.t[0]) if self.t.size else 0.0
        return float(self.n_f_evals + self.n_extra_f_evals) / span if span > 0.0 else float("nan")

    def max_normalized_error(self) -> float:
        """接受步上 :math:`\\epsilon` 的最大值——控制器是否真的把误差压在容差内。"""
        acc = self.error_accepted()
        return float(np.max(acc)) if acc.size else float("nan")


def integrate_adaptive(
    method: "str | rkmod.Tableau",
    f: rkmod.RHS,
    t_span: tuple[float, float],
    y0: Any,
    tol: float,
    *,
    atol: float | None = None,
    h0: float | None = None,
    law: ControlLaw | None = None,
    local_extrapolation: bool = True,
    exact: Callable[[Any], np.ndarray] | None = None,
    label: str = "",
    max_steps: int = 200000,
    h_min: float = 1e-13,
    true_local_error: bool = False,
    n_sub: int = 8,
) -> AdaptiveRun:
    r"""嵌入对 + 控制律的自适应积分。

    四个约定（每一个都对应一条陷阱）：

    1. **``local_extrapolation`` 决定推进哪个解**：``True`` 推进高阶解 :math:`b`
       （现代求解器的做法），``False`` 推进低阶解 :math:`\hat b`。
       估计量的阶 :math:`\hat p+1` 与推进解的局部误差阶 :math:`p+1` 谁大谁小，
       决定了全局误差随容差的幂次是 :math:`\mathrm{tol}^{p/(\hat p+1)}` 还是
       :math:`\mathrm{tol}`——这不是「实现细节」，是**量级差别**。
    2. **FSAL 复用**：上一接受步末尾的 :math:`f(t+h,y_{n+1})` 直接当下一步第一级，
       省下的求值记在 ``n_f_evals`` 的账里（不是省在嘴上）。
    3. **爆炸显式停止**：与定步长同款，:math:`\max|y|>` :data:`numcore.ode.rk.BLOWUP`
       就收工并标注，绝不让非有限值流进帧。
    4. **``true_local_error=True`` 时**每一步都用 ``n_sub`` 个子步的参考解量真局部误差
       （代价记在 ``n_extra_f_evals``），用来对拍估计器——这是本章「估计器可信吗」
       唯一的实测依据。
    """
    tab = rkmod.as_tableau(method)
    if tab.b_hat is None:
        raise ValueError(f"{tab.name} 没有嵌入低阶权，无法做误差估计（可用：{list(rkmod.PAIR_METHODS)}）")
    if not tol > 0.0:
        raise ValueError(f"容差必须为正，得到 {tol}")
    law_eff = law or ControlLaw()
    atol_eff = float(atol) if atol is not None else float(tol)
    t0, t1 = float(t_span[0]), float(t_span[1])
    if not t1 > t0:
        raise ValueError(f"需要 t1 > t0，得到 {t_span}")
    yy = np.asarray(y0, dtype=float).reshape(-1)
    h = float(h0) if h0 is not None else (t1 - t0) * 1.0e-2
    order_low = int(tab.order_hat)
    order_prop = int(tab.order) if local_extrapolation else int(tab.order_hat)

    ts: list[float] = [t0]
    ys: list[np.ndarray] = [yy.copy()]
    t_att: list[float] = []
    h_att: list[float] = []
    err_att: list[float] = []
    acc_att: list[bool] = []
    tru_att: list[float] = []
    n_f = 0
    n_extra = 0
    err_prev: float | None = None
    t = t0
    blowup = False
    t_blow = float("nan")
    f_cache: np.ndarray | None = None
    n_attempt = 0
    while t < t1 - 1e-15 and n_attempt < int(max_steps):
        h = min(h, t1 - t)
        if h < h_min:
            break
        step = rkmod.rk_step(method, f, t, yy, h, f_start=f_cache)
        n_f += step.n_f_evals
        eps = rkmod.scaled_error_measure(step.error, step.y, yy, atol=atol_eff, rtol=tol)
        true_err = float("nan")
        if true_local_error:
            true_err = rkmod.step_error_reference(method, f, t, yy, h, n_sub=n_sub)
            n_extra += int(tab.stages) * (int(n_sub) + 1)
        accepted = bool(eps <= 1.0) or h <= h_min * 1.0000001
        t_att.append(t)
        h_att.append(h)
        err_att.append(eps)
        acc_att.append(accepted)
        tru_att.append(true_err)
        n_attempt += 1
        if accepted:
            y_prop = step.y if local_extrapolation else step.y_low
            yy = np.asarray(y_prop, dtype=float)
            t = t + h
            ts.append(t)
            ys.append(yy.copy())
            if step.fsal_ready and local_extrapolation:
                f_cache = step.stages[-1]
            else:
                f_cache = None
            # PI 的历史项用的是**上一步**的误差：先把当前值送进 next_step，再更新历史，
            # 否则 (eps/eps)^b ≡ 1，PI 退化成 I —— 一个不会报错的静默降级。
            h = next_step(law_eff, h, eps, err_prev, order_low)
            err_prev = eps
            if not np.all(np.isfinite(yy)) or float(np.max(np.abs(yy))) > rkmod.BLOWUP:
                blowup = True
                t_blow = t
                break
        else:
            h = h * elementary_factor(
                eps, 1.0 / (order_low + 1), safety=law_eff.safety, fac_min=law_eff.fac_min, fac_max=1.0
            )
            f_cache = None
    tt = np.asarray(ts, dtype=float)
    Y = np.asarray(ys, dtype=float)
    if exact is not None:
        curve = rkmod.pointwise_error(Y, np.asarray(exact(tt), dtype=float))
        max_err = float(np.max(curve))
        end_err = float(curve[-1])
    else:
        curve = np.zeros(tt.size, dtype=float)
        max_err = float("nan")
        end_err = float("nan")
    acc = np.asarray(acc_att, dtype=bool)
    return AdaptiveRun(
        method=method,
        problem=label,
        tol=float(tol),
        atol=atol_eff,
        law=law_eff.name,
        local_extrapolation=bool(local_extrapolation),
        t=tt,
        y=Y,
        t_attempt=np.asarray(t_att, dtype=float),
        h_attempt=np.asarray(h_att, dtype=float),
        error_attempt=np.asarray(err_att, dtype=float),
        accepted=acc,
        error_true_attempt=np.asarray(tru_att, dtype=float),
        max_error=max_err,
        endpoint_error=end_err,
        error_curve=curve,
        n_accepted=int(np.count_nonzero(acc)),
        n_rejected=int(np.count_nonzero(~acc)),
        n_f_evals=int(n_f),
        n_extra_f_evals=int(n_extra),
        ok=not blowup,
        blowup=blowup,
        t_blowup=t_blow,
        order_propagated=order_prop,
        order_estimate=order_low,
    )


# --------------------------------------------------------------------------- #
# 3. 估计器可靠性
# --------------------------------------------------------------------------- #
def estimator_ratio(
    estimated: Any,
    true_error: Any,
    *,
    floor: float = 1e-300,
) -> np.ndarray:
    r"""逐点比值 :math:`r=\hat e/e`（真误差触零时用 ``floor`` 兜底，避免 inf）。

    真误差为 0 的步（机器精度级）不携带信息，比值会被放大成假警报；
    调用方应当先按真误差的量级筛掉它们（``tests/3.1`` 里用相对量级筛选）。
    """
    e = np.asarray(estimated, dtype=float).ravel()
    t = np.asarray(true_error, dtype=float).ravel()
    if e.size != t.size:
        raise ValueError(f"长度不同：{e.size} vs {t.size}")
    denom = np.maximum(np.abs(t), float(floor))
    return e / denom


@dataclass(frozen=True)
class EstimatorStudy:
    r"""固定步长下「估计 vs 真局部误差」的对照（每一步都有）。"""

    method: str
    problem: str
    h: float
    normalized_error: np.ndarray
    true_error: np.ndarray
    ratio: np.ndarray
    ratio_median: float
    ratio_min: float
    ratio_max: float
    pessimistic: bool

    def summary(self) -> str:
        kind = "悲观（白花工作量）" if self.pessimistic else "乐观（可能超标）"
        return (
            f"{self.method}@{self.problem} h={self.h:g}：估计/真误差 中位数 {self.ratio_median:.3g}"
            f"（区间 [{self.ratio_min:.3g}, {self.ratio_max:.3g}]），{kind}"
        )


def estimator_study(
    problem_name: str,
    method: "str | rkmod.Tableau",
    h: float,
    *,
    n_sub: int = 8,
    t_end: float | None = None,
    floor_rel: float = 1e-8,
    max_steps: int = 20000,
) -> EstimatorStudy:
    r"""沿一条定步长轨道，逐点比较「嵌入对给出的估计」与「子步参考给出的真误差」。

    两条筛选纪律：

    * 真误差低于 ``floor_rel`` × 该轨道真误差最大值（或低于 **100 倍单位舍入**）的步被丢弃：
      那里真误差已到机器精度，比值没有意义；
    * **比值的中位数**才是「估计器偏乐观还是偏悲观」的判据——单步比值可以任意大，
      因为估计量可以在某一步恰好穿零。
    """
    prob = rkmod.problem(problem_name)
    t1 = float(prob.t_end if t_end is None else t_end)
    tab = rkmod.as_tableau(method)
    if tab.b_hat is None:
        raise ValueError(f"{tab.name} 没有嵌入对，无法研究误差估计器")
    y0 = prob.y0_array()
    eps: list[float] = []
    tru: list[float] = []
    t = 0.0
    yy = y0.copy()
    n_steps = 0
    while t < t1 - 1e-15 and n_steps < int(max_steps):
        hh = min(float(h), t1 - t)
        step = rkmod.rk_step(method, prob.rhs, t, yy, hh)
        # 两侧必须用**同一把**无权尺子：带 atol/rtol 的接受度量会把 1/atol 混进比值里
        # （第一版就是这么错的：比值中位数全是 1e14 量级，看起来像"估计器差 14 个数量级"，
        #  其实只是分母被 1e-14 除了一次）。
        eps.append(rkmod.plain_error_measure(step.error))
        tru.append(rkmod.plain_error_measure(rkmod.estimate_local_error(method, prob.rhs, t, yy, hh, n_sub=n_sub)))
        yy = step.y
        t += hh
        n_steps += 1
    e = np.asarray(eps, dtype=float)
    tr = np.asarray(tru, dtype=float)
    scale = float(np.max(tr)) if tr.size else 0.0
    keep = tr > max(float(floor_rel) * scale, 100.0 * float(np.finfo(float).eps))
    if not bool(np.any(keep)):
        raise ValueError("所有步的真误差都在舍入地板上，估计器研究无意义（请减小 h）")
    ratio = estimator_ratio(e[keep], tr[keep])
    med = float(np.median(ratio))
    return EstimatorStudy(
        method=tab.name,
        problem=problem_name,
        h=float(h),
        normalized_error=e,
        true_error=tr,
        ratio=ratio,
        ratio_median=med,
        ratio_min=float(np.min(ratio)),
        ratio_max=float(np.max(ratio)),
        pessimistic=bool(med > 1.0),
    )


def true_local_errors(run: AdaptiveRun) -> np.ndarray:
    """自适应轨道上**每一步**的真局部误差（要求 ``true_local_error=True``）。"""
    arr = np.asarray(run.error_true_attempt, dtype=float)
    if arr.size and not np.all(np.isfinite(arr)):
        raise ValueError("该 run 没有记录真局部误差（integrate_adaptive(true_local_error=True)）")
    return arr


# --------------------------------------------------------------------------- #
# 4. 工作-精度前沿
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WorkPrecision:
    r"""一条工作-精度曲线：一串容差上的（实测工作量, 实测误差）。

    工作量用 :math:`f` 求值次数每单位时间——它是**方法无关的**代价单位，
    比「步数」公平：四阶方法一步 4 次求值，七级 FSAL 一步 6 次。
    """

    method: str
    problem: str
    tol: np.ndarray
    work: np.ndarray
    error_max: np.ndarray
    error_endpoint: np.ndarray
    n_accepted: np.ndarray
    n_rejected: np.ndarray
    accept_ratio: np.ndarray
    order: int
    order_estimate: int

    def work_at(self, target_error: float) -> float:
        """达到 ``target_error`` 所需工作量的**对数-对数插值**（越界抛 :class:`ValueError`）。"""
        return work_at_accuracy(self.error_max, self.work, target_error)

    def achieved_tolerance_ratio(self) -> np.ndarray:
        """实测最大误差 / 请求容差（「容差 ≈ 精度」这个信念的检验量）。"""
        return self.error_max / self.tol


def work_precision(
    method: "str | rkmod.Tableau",
    problem_name: str,
    tols: Sequence[float],
    *,
    law: ControlLaw | None = None,
    local_extrapolation: bool = True,
    h0: float | None = None,
) -> WorkPrecision:
    r"""同一方法在一串容差上的工作-精度曲线（自适应积分实测，不是理论拟合）。

    ``tols`` 按调用方给的顺序使用（推荐从松到紧），返回数组与它一一对应。
    """
    prob = rkmod.problem(problem_name)
    tab = rkmod.as_tableau(method)
    grid = np.asarray(list(tols), dtype=float)
    n = grid.size
    work = np.zeros(n, dtype=float)
    emax = np.zeros(n, dtype=float)
    eend = np.zeros(n, dtype=float)
    nacc = np.zeros(n, dtype=float)
    nrej = np.zeros(n, dtype=float)
    ratio = np.zeros(n, dtype=float)
    for k, tol in enumerate(grid):
        run = integrate_adaptive(
            method,
            prob.rhs,
            (0.0, prob.t_end),
            prob.y0_array(),
            float(tol),
            h0=h0,
            law=law,
            local_extrapolation=local_extrapolation,
            exact=prob.exact,
            label=problem_name,
        )
        work[k] = run.work_per_time()
        emax[k] = run.max_error
        eend[k] = run.endpoint_error
        nacc[k] = float(run.n_accepted)
        nrej[k] = float(run.n_rejected)
        ratio[k] = run.accept_ratio()
    return WorkPrecision(
        method=tab.name,
        problem=problem_name,
        tol=grid,
        work=work,
        error_max=emax,
        error_endpoint=eend,
        n_accepted=nacc,
        n_rejected=nrej,
        accept_ratio=ratio,
        order=int(tab.order),
        order_estimate=int(tab.order_hat) + 1,
    )


def work_at_accuracy(error: Any, work: Any, target_error: float) -> float:
    r"""在 :math:`\log`--:math:`\log` 平面上插值出「达到目标误差所需工作量」。

    误差随工作量**单调下降**是这条曲线可用的前提：函数先检查两列都严格单调，
    不单调就抛错而不是给出一个好看但无意义的插值（「曲线交叉」意味着数据本身有问题）。
    """
    e = np.asarray(error, dtype=float).ravel()
    w = np.asarray(work, dtype=float).ravel()
    if e.size != w.size or e.size < 2:
        raise ValueError(f"需要长度相同的两组数据（≥2），得到 {e.size} 与 {w.size}")
    keep = np.isfinite(e) & np.isfinite(w) & (e > 0) & (w > 0)
    e = e[keep]
    w = w[keep]
    order = np.argsort(w)
    e = e[order]
    w = w[order]
    if not bool(np.all(np.diff(e) < 0)):
        raise ValueError("误差随工作量不是严格递减：曲线不能用于反查（先检查数据）")
    target = float(target_error)
    if target > e[0] or target < e[-1]:
        raise ValueError(
            f"目标误差 {target:g} 超出实测区间 [{e[-1]:g}, {e[0]:g}]："
            "反查只能插值，外推出来的工作量是编的"
        )
    xe = np.log(e)
    xw = np.log(w)
    # 自写对数-对数线性插值（np.interp 属 P3 黑名单：这是"现成算法"，
    # 而且这里的单调性前提必须由我们自己先验证）。
    lt = math.log(target)
    j = int(np.searchsorted(-xe, -lt, side="left"))
    j = min(max(j, 1), xe.size - 1)
    t = (lt - xe[j - 1]) / (xe[j] - xe[j - 1])
    return float(np.exp(xw[j - 1] + t * (xw[j] - xw[j - 1])))


def step_ratio_oscillation(h: Any) -> float:
    r"""步长序列的**振荡指标**：相邻比值的符号变化率 :math:`\\in[0,1]`。

    :math:`s_n=\\log(h_{n+1}/h_n)`；指标 = :math:`s` 的一阶差分的符号变化比例。
    接近 1 表示「放大-缩小-放大」的周期-2 来回（I 控制器的典型病），
    接近 0 表示步长平滑（PI 控制器的目标）。这是「控制器稳定性」的可执行判据。
    """
    hh = np.asarray(h, dtype=float).ravel()
    hh = hh[np.isfinite(hh) & (hh > 0)]
    if hh.size < 4:
        raise ValueError(f"至少需要 4 个正步长，得到 {hh.size}")
    s = np.log(hh[1:] / hh[:-1])
    d = np.diff(s)
    d = d[d != 0.0]
    if d.size < 2:
        return 0.0
    sign = np.sign(d)
    return float(np.count_nonzero(sign[1:] != sign[:-1])) / float(sign.size - 1)
