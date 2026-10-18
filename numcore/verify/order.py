"""收敛阶的测量与判定（A 类断言的核心工具）。

数学背景
--------
若误差满足 ``e(h) = C·h^p + o(h^p)``，则在双对数坐标下 ``log e = log C + p·log h``
是一条直线，斜率即收敛阶 ``p``。因此：

* **全局拟合**（:func:`fit_order`）：对全部（或渐近区的）点做最小二乘；
* **局部阶**（:func:`consecutive_orders`）：相邻两点给出 ``p_i = log(e_i/e_{i+1}) / log(h_i/h_{i+1})``，
  用于检查"阶数是否稳定"，以及识别渐近区从哪一点开始。

判定口径（docs/CHAPTERS.md §2）：实测阶落在理论阶 ``±0.1`` 内。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import fmt_value
from .report import fail, record
from .tolerance import safe_log

__all__ = [
    "OrderFit",
    "fit_order",
    "consecutive_orders",
    "WindowDiagnosis",
    "diagnose_order",
    "asymptotic_window",
    "assert_order",
    "reference_line",
]

_DEFAULT_SLOPE_TOL = 0.1


@dataclass(frozen=True)
class OrderFit:
    """双对数最小二乘的拟合结果。"""

    slope: float
    intercept: float
    r2: float
    stderr: float
    n_used: int
    n_dropped: int
    h: np.ndarray
    err: np.ndarray

    @property
    def constant(self) -> float:
        """误差常数 ``C = exp(intercept)``。"""
        return float(np.exp(self.intercept))

    def predict(self, h: Any) -> np.ndarray:
        """用拟合直线预测误差 ``C·h^slope``。"""
        return self.constant * np.asarray(h, dtype=float) ** self.slope

    def __str__(self) -> str:  # pragma: no cover - 仅用于人读
        return (
            f"p = {self.slope:.4f} ± {self.stderr:.2e} (R²={self.r2:.6f}, "
            f"用点 {self.n_used}/{self.n_used + self.n_dropped})"
        )


def _sanitize(h: Any, err: Any) -> tuple[np.ndarray, np.ndarray]:
    hh = np.asarray(h, dtype=float).ravel()
    ee = np.asarray(err, dtype=float).ravel()
    if hh.shape != ee.shape:
        raise ValueError(f"h 与 err 形状不一致：{hh.shape} vs {ee.shape}")
    if hh.size == 0:
        raise ValueError("h/err 为空，无法拟合阶数")
    return hh, ee


def fit_order(
    h: Any,
    err: Any,
    *,
    skip: int = 0,
    head: int | None = None,
    tail: int | None = None,
    min_points: int = 3,
    drop_nonpositive: bool = True,
) -> OrderFit:
    """对 ``err ≈ C·h^slope`` 做双对数最小二乘。

    参数
    ----
    h, err
        步长与误差序列。约定按 ``h`` **递减**给出（从最粗网格到最细网格）。
    skip
        丢弃最前面的 ``skip`` 个点（尚未进入渐近区的大步长）。
    head
        只保留（skip 之后的）**最前面** ``head`` 个点，即**粗网格端**。
        这是对付"舍入地板"的正确工具：``err`` 在细网格端不再下降，
        拟合出的阶数会偏小，必须把细端切掉。
    tail
        只保留（skip/head 之后的）**最后** ``tail`` 个点，即**细网格端**。
        注意方向与 ``head`` 相反——保留细端恰好会保留被舍入地板污染的点，
        所以"尾部触地板"时应该用 ``head=`` 而不是 ``tail=``。
    drop_nonpositive
        丢弃 ``h ≤ 0`` 或 ``err ≤ 0`` 的点（0 误差意味着"精确到机器零"，
        不携带阶数信息）。
    """
    hh, ee = _sanitize(h, err)
    order = np.arange(hh.size)
    mask = np.ones(hh.size, dtype=bool)
    if skip:
        mask[: min(int(skip), hh.size)] = False
    if head is not None:
        idx = np.flatnonzero(mask)
        if int(head) < idx.size:
            mask[idx[int(head) :]] = False
    if tail is not None:
        idx = np.flatnonzero(mask)
        if int(tail) < idx.size:
            drop = idx[: idx.size - int(tail)]
            mask[drop] = False
    if drop_nonpositive:
        mask &= np.isfinite(hh) & np.isfinite(ee) & (hh > 0) & (ee > 0)
    else:
        mask &= np.isfinite(hh) & np.isfinite(ee)
    xs = safe_log(hh[mask])
    ys = safe_log(ee[mask])
    n = int(xs.size)
    if n < max(int(min_points), 2):
        raise ValueError(
            f"有效点不足：给定 {hh.size} 点，过滤后仅 {n} 点（需要 ≥{max(int(min_points), 2)}）。"
            f" 有效阶数={order[mask].tolist()}"
        )
    xbar = float(np.mean(xs))
    ybar = float(np.mean(ys))
    sxx = float(np.sum((xs - xbar) ** 2))
    if sxx == 0.0:
        raise ValueError("所有 h 相同，无法拟合阶数")
    sxy = float(np.sum((xs - xbar) * (ys - ybar)))
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    resid = ys - (intercept + slope * xs)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((ys - ybar) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    dof = max(n - 2, 1)
    stderr = float(np.sqrt(ss_res / dof / sxx))
    return OrderFit(
        slope=float(slope),
        intercept=float(intercept),
        r2=float(r2),
        stderr=stderr,
        n_used=n,
        n_dropped=int(hh.size - n),
        h=hh[mask],
        err=ee[mask],
    )


def consecutive_orders(h: Any, err: Any) -> np.ndarray:
    """相邻点局部阶 ``p_i = log(e_i/e_{i+1}) / log(h_i/h_{i+1})``（长度 n−1）。"""
    hh, ee = _sanitize(h, err)
    if hh.size < 2:
        raise ValueError("至少需要 2 个点才能计算局部阶")
    num = safe_log(ee[:-1] / ee[1:])
    den = safe_log(hh[:-1] / hh[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return out


@dataclass(frozen=True)
class WindowDiagnosis:
    """对"误差序列是否处于渐近幂律区"的**诊断**（不是自动决策）。

    设计立场：能不能用全部点拟合，是**测试作者**的判断，工具只负责把证据摆出来。
    自动挑窗口看起来聪明，实际上会在"舍入地板"与"预渐近区"两种污染下悄悄给出
    一个漂亮但错误的斜率——那正是本项目要消灭的东西。
    """

    local_orders: np.ndarray
    """相邻点局部阶（长度 n−1）。"""

    head_p: float
    """前 1/3 局部阶的中位数（大 h 端）。"""

    tail_p: float
    """后 1/3 局部阶的中位数（小 h 端，即渐近方向）。"""

    floored: bool
    """尾部被舍入地板淹没：``tail_p`` 明显小于 ``head_p`` 且已接近 0。"""

    noisy: bool
    """局部阶整体不服从单一幂律（首尾差超过 ``tol``）。"""

    suggest_skip: int
    """建议丢弃的前导点数（最长"局部阶一致"连续段之前的部分）。"""

    suggest_tail: int  #: 建议保留的点数（配合 suggest_skip 用作"保后缀"窗口）
    """建议保留的点数（从 ``suggest_skip`` 起的连续段长度）。"""

    def slice(self) -> slice:
        """建议窗口对应的切片。"""
        return slice(self.suggest_skip, self.suggest_skip + self.suggest_tail)

    def note(self) -> str:
        """一句话人读诊断（用于报告与 UI 提示）。"""
        if self.floored:
            return (
                f"序列尾部已触舍入地板：尾部局部阶 {self.tail_p:.2f} 远低于头部 {self.head_p:.2f}，"
                f"请用 head={max(2, self.suggest_skip + 2)} 只保留粗网格端（注意：tail= 保留的是细端，"
                "恰好会保留被污染的点），否则拟合出的阶数会偏小。"
            )
        if self.noisy:
            return (
                f"局部阶不服从单一幂律（头部 {self.head_p:.2f} → 尾部 {self.tail_p:.2f}），"
                f"建议只取连续一致段：skip={self.suggest_skip}, tail={self.suggest_tail}。"
            )
        return f"幂律区良好：局部阶中位数 {self.tail_p:.2f}，全程可用。"


def diagnose_order(
    h: Any,
    err: Any,
    *,
    tol: float = 0.1,
    floor_p: float = 0.25,
    min_run: int = 2,
) -> WindowDiagnosis:
    """诊断误差序列的幂律质量（见 :class:`WindowDiagnosis`）。"""
    hh, ee = _sanitize(h, err)
    orders = consecutive_orders(hh, ee)
    n = orders.size
    finite = np.isfinite(orders)
    if n == 0 or not np.any(finite):
        nan = float("nan")
        return WindowDiagnosis(orders, nan, nan, False, True, 0, hh.size)
    k = max(1, n // 3)
    head_p = float(np.median(orders[:k]))
    tail_p = float(np.median(orders[-k:]))
    # 最长"局部阶彼此一致"的连续段（作为 *建议*）
    best = (0, 0)
    for i in range(n):
        for j in range(i + 1, n + 1):
            seg = orders[i:j]
            if not np.all(np.isfinite(seg)):
                break
            med = float(np.median(seg))
            if np.all(np.abs(seg - med) <= tol) and (j - i) > best[1]:
                best = (i, j - i)
    start, length = best
    if length < min_run:
        start, length = 0, n
    floored = bool(tail_p < floor_p and head_p > tail_p + tol)
    noisy = bool(abs(head_p - tail_p) > tol and not floored)
    return WindowDiagnosis(
        local_orders=orders,
        head_p=head_p,
        tail_p=tail_p,
        floored=floored,
        noisy=noisy,
        suggest_skip=int(start),
        suggest_tail=int(min(length + 1, hh.size - start)),
    )


def asymptotic_window(h: Any, err: Any, *, tol: float = 0.1) -> int:
    """返回建议的前导丢弃点数（:func:`diagnose_order` 的便捷包装，仅作提示）。"""
    return int(diagnose_order(h, err, tol=tol).suggest_skip)


def assert_order(
    h: Any,
    err: Any,
    *,
    theory: float,
    tol: float = _DEFAULT_SLOPE_TOL,
    skip: int = 0,
    min_points: int = 3,
    label: str | None = None,
    detail: str = "",
    kind: str = "A",
) -> OrderFit:
    """断言实测收敛阶 ``= 理论阶 ± tol``（默认 ±0.1）。

    ``kind`` 决定这条记录算哪一类断言：缺省 ``"A"``（与解析阶对拍——这是本仓的多数用法），
    但"实测阶是否达到理论阶"严格说属于 C 类（收敛率）。声明为 ``@assertion("C")`` 的测试
    如果直接用缺省值，会得到"声明 C 却只登记 A"的类别不符失败（``numcore/verify/report.py``
    会当场报），这时显式传 ``kind="C"`` 即可。

    返回 :class:`OrderFit`，便于调用方继续使用拟合出的常数 ``C``、``R²`` 等。
    """
    name = label or f"收敛阶 = {theory:g}"
    fit = fit_order(h, err, skip=skip, min_points=min_points)
    local = consecutive_orders(fit.h, fit.err)
    ok = abs(fit.slope - theory) <= tol
    detail_full = (
        f"R²={fit.r2:.6f}，斜率标准误={fit.stderr:.2e}，用点 {fit.n_used} 个（丢弃 {fit.n_dropped} 个），"
        f"局部阶={np.array2string(local, precision=3, suppress_small=True)}"
        + (f"；{detail}" if detail else "")
    )
    if not ok:
        fail(
            label=name,
            theory=theory,
            measured=fit.slope,
            tol=f"±{tol:g}",
            detail=detail_full,
            kind=kind,
        )
    record(
        label=name,
        theory=theory,
        measured=fit.slope,
        tol=f"±{tol:g}",
        passed=True,
        detail=detail_full,
        kind=kind,
        extra={"r2": fit.r2, "stderr": fit.stderr, "n_used": fit.n_used},
    )
    return fit


def reference_line(
    h: Any,
    theory: float,
    *,
    anchor: tuple[float, float] | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """构造理论斜率参考线，供双对数图叠加（防误导规则 2）。

    ``anchor`` 给出参考线穿过的点 ``(h₀, e₀)``；缺省取 ``h`` 的最小值处的 ``scale``。
    """
    hh = np.asarray(h, dtype=float)
    if anchor is None:
        h0 = float(np.min(hh))
        e0 = float(scale)
    else:
        h0, e0 = float(anchor[0]), float(anchor[1])
    with np.errstate(over="ignore", invalid="ignore"):
        return e0 * (hh / h0) ** theory


def format_fit(fit: OrderFit) -> str:
    """人读摘要（报告与讲义共用）。"""
    return f"实测阶 {fmt_value(fit.slope)}（{fit}）"
