r"""2.4 章 —— 自适应求积：用**嵌套法则的差**当误差估计，递归只细分"贵"的区间。

自适应 Simpson 的全部机制就一行：

.. math::
    \hat E=\frac{S(a,m)+S(m,b)-S(a,b)}{15},

其中 :math:`S` 是单面板 Simpson。:math:`S(a,m)+S(m,b)` 的节点包含 :math:`S(a,b)` 的三个节点
（:math:`a,m,b`），所以这是一对**嵌套**法则：细分只多算两个中点。系数 15 来自
Simpson 误差主项的 :math:`h^4`：两半误差之和是整段的 :math:`1/16`，于是

.. math::
    S_{\text{fine}}-S_{\text{coarse}}=E_{\text{fine}}-E_{\text{coarse}}
    \approx E_{\text{coarse}}\left(\tfrac1{16}-1\right)=-\tfrac{15}{16}E_{\text{coarse}},

所以 :math:`\hat E\approx-E_{\text{coarse}}`，而 :math:`S_{\text{fine}}+\hat E` 是
:math:`S_{\text{fine}}` 的一次 Richardson 修正。

**这个估计器估的是它自己那个模型的误差，不是真误差**（本章第三条陷阱）：

* :math:`f` 光滑时 :math:`\hat E` 与真误差同量级（实测比值 1~2，符号相反）；
* :math:`f` 有折点/奇异时，:math:`h^4` 的假设在含奇异点的面板上不成立，
  :math:`\hat E` 可以比真误差小几个数量级——于是它宣布"已收敛"，而结果是错的；
* 到了舍入地板（:math:`\hat E\lesssim u\sum|f|`）它同样会宣布收敛：**地板不是 0**，
  它只是不再下降。

本模块把这三件事都变成可观测量：每个被接受的区间都留下它的
:math:`\hat E`、深度、求值次数；整条接受序列留下"累计估计误差"与"累计真误差"
（后者只有在调用方给出解析真值时才有——没有真值就写 ``None``，不编一个）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

__all__ = ["AdaptiveResult", "adaptive_simpson", "simpson_panel"]


def simpson_panel(fa: float, fm: float, fb: float, a: float, m: float, b: float) -> float:
    r"""单面板 Simpson：:math:`\frac{b-a}{6}\big(f(a)+4f(m)+f(b)\big)`。"""
    return (float(b) - float(a)) / 6.0 * (float(fa) + 4.0 * float(fm) + float(fb))


@dataclass(frozen=True)
class AdaptiveResult:
    r"""自适应细分的完整快照（每个被接受区间一行，按接受顺序排列）。

    数组都是**等长并按接受顺序对齐**的：``lo/hi`` 是区间端点、``depth`` 是细分深度、
    ``est`` 是该区间的局部误差估计、``evals`` 是为它花掉的求值次数、
    ``evals_cum`` 是到它为止的累计求值次数、``partial`` 是累计被接受值（收敛到积分值）、
    ``true_err_cum`` 是累计值与解析真值之差（没有真值时为 ``None``）。
    """

    value: float
    error_estimate: float
    n_evals: int
    n_intervals: int
    max_depth_used: int
    converged: bool
    tol: float
    lo: np.ndarray
    hi: np.ndarray
    depth: np.ndarray
    est: np.ndarray
    evals: np.ndarray
    evals_cum: np.ndarray
    partial: np.ndarray
    true_error: float | None
    true_err_cum: np.ndarray | None

    @property
    def widths(self) -> np.ndarray:
        """被接受区间的宽度（画细分树/直方图用）。"""
        return self.hi - self.lo

    def depth_counts(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 ``(深度, 该深度的区间数)``——"细分树"的横切面。"""
        d = self.depth.astype(int)
        levels = np.arange(0, int(d.max()) + 1) if d.size else np.zeros(0, dtype=int)
        counts = np.asarray([int(np.count_nonzero(d == k)) for k in levels], dtype=float)
        return levels.astype(float), counts

    def evals_by_depth(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 ``(深度, 该深度吃掉的求值次数)``——"谁吃掉了最多函数求值"的答案。"""
        d = self.depth.astype(int)
        levels = np.arange(0, int(d.max()) + 1) if d.size else np.zeros(0, dtype=int)
        tot = np.asarray([float(np.sum(self.evals[d == k])) for k in levels], dtype=float)
        return levels.astype(float), tot

    def worst_intervals(self, k: int = 5) -> np.ndarray:
        """按局部误差估计降序返回前 ``k`` 个区间的下标（"最多误差住在哪里"）。"""
        order = np.argsort(-np.abs(self.est))
        return order[: max(1, int(k))]


def adaptive_simpson(
    f: Callable[[Any], Any],
    a: float,
    b: float,
    tol: float,
    *,
    max_depth: int = 24,
    exact: float | None = None,
    max_intervals: int = 20000,
) -> AdaptiveResult:
    r"""自适应 Simpson：递归细分到局部估计误差之和 :math:`\le\text{tol}` 或深度上限。

    ``exact`` 给了解析真值时，结果里会多一条**累计真误差**曲线——
    把它与累计估计误差画在一起，就是"估计器估的是自己的误差"这句话的实测版。
    没有真值时 ``true_error`` / ``true_err_cum`` 都是 ``None``（如实标注，不填 0）。
    """
    lo, hi = float(a), float(b)
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        raise ValueError(f"区间必须有限且 b>a，得到 [{a}, {b}]")
    t = float(tol)
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError(f"tol 必须是正有限数，得到 {tol!r}")
    d_max = int(max_depth)
    if d_max < 0:
        raise ValueError(f"max_depth 不能为负，得到 {max_depth!r}")
    budget = int(max_intervals)
    if budget < 1:
        raise ValueError(f"max_intervals 至少为 1，得到 {max_intervals!r}")

    state = {"n_evals": 0}
    lo_list: list[float] = []
    hi_list: list[float] = []
    depth_list: list[int] = []
    est_list: list[float] = []
    evals_list: list[int] = []
    partial_list: list[float] = []
    running = 0.0

    m = 0.5 * (lo + hi)
    fa = float(f(np.asarray([lo], dtype=float))[0])
    fm = float(f(np.asarray([m], dtype=float))[0])
    fb = float(f(np.asarray([hi], dtype=float))[0])
    state["n_evals"] += 3

    def recurse(x0: float, x1: float, f0: float, f1: float, fmv: float, whole: float, depth: int) -> float:
        mid = 0.5 * (x0 + x1)
        lm = 0.5 * (x0 + mid)
        rm = 0.5 * (mid + x1)
        flm = float(f(np.asarray([lm], dtype=float))[0])
        frm = float(f(np.asarray([rm], dtype=float))[0])
        state["n_evals"] += 2
        left = simpson_panel(f0, flm, fmv, x0, lm, mid)
        right = simpson_panel(fmv, frm, f1, mid, rm, x1)
        fine = left + right
        est = (fine - whole) / 15.0
        local_tol = t * (x1 - x0) / (hi - lo)
        if depth >= d_max or abs(est) <= local_tol or len(lo_list) >= budget:
            nonlocal running
            lo_list.append(x0)
            hi_list.append(x1)
            depth_list.append(depth)
            est_list.append(est)
            evals_list.append(2.0)
            running += fine + est
            partial_list.append(running)
            return fine + est
        return recurse(x0, mid, f0, fmv, flm, left, depth + 1) + recurse(
            mid, x1, fmv, f1, frm, right, depth + 1
        )

    whole0 = simpson_panel(fa, fm, fb, lo, m, hi)
    value = recurse(lo, hi, fa, fb, fm, whole0, 0)

    lo_arr = np.asarray(lo_list, dtype=float)
    hi_arr = np.asarray(hi_list, dtype=float)
    est_arr = np.asarray(est_list, dtype=float)
    ev_arr = np.asarray(evals_list, dtype=float)
    cum = np.cumsum(ev_arr) + 3.0
    part = np.asarray(partial_list, dtype=float)
    true_err = None if exact is None else abs(float(value) - float(exact))
    true_cum = None if exact is None else np.abs(part - float(exact))
    total_est = float(np.sum(np.abs(est_arr))) if est_arr.size else 0.0
    return AdaptiveResult(
        value=float(value),
        error_estimate=total_est,
        n_evals=int(state["n_evals"]),
        n_intervals=int(lo_arr.size),
        max_depth_used=int(max(depth_list)) if depth_list else 0,
        converged=bool(total_est <= t and len(lo_list) < budget),
        tol=t,
        lo=lo_arr,
        hi=hi_arr,
        depth=np.asarray(depth_list, dtype=float),
        est=est_arr,
        evals=ev_arr,
        evals_cum=cum,
        partial=part,
        true_error=true_err,
        true_err_cum=true_cum,
    )
