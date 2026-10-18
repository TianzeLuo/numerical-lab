"""不变量断言（B 类）：守恒量、正交性、对称性、单调性、有界漂移。

纪律（POLICY §3）：**容差必须说清来源**。本模块每个 helper 都把来源写进报告 ``detail``：

* ``machine:8·n·u·scale`` —— 浮点求和误差界（scale 取问题量级，n 取累积运算次数）；
* ``explicit`` —— 调用方给出的解析容差（此时 ``detail`` 必须解释为什么是这个数）。
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .errors import fmt_value
from .report import fail, record
from .tolerance import max_abs, scaled_tol

__all__ = [
    "assert_within",
    "assert_leq",
    "assert_all_finite",
    "assert_conserved",
    "assert_bounded_drift",
    "assert_linear_drift",
    "assert_monotone",
    "assert_orthogonal",
    "assert_symmetric",
]


def _tol_or_machine(
    tol: float | None,
    *,
    scale: float,
    n_ops: int,
    factor: float = 8.0,
) -> tuple[float, str]:
    """返回 ``(容差, 来源说明)``；``tol=None`` 时用机器精度误差界。"""
    if tol is None:
        return scaled_tol(scale, n_ops=n_ops, factor=factor), f"machine:{factor:g}·{max(n_ops,1)}·u·scale"
    return float(tol), "explicit"


def assert_within(
    measured: Any,
    theory: Any,
    *,
    rtol: float,
    atol: float = 0.0,
    label: str | None = None,
    detail: str = "",
) -> float:
    """断言 ``measured`` 落在 ``theory`` 的 ``±(rtol·|theory| + atol)`` 内。"""
    m = float(np.max(np.abs(np.asarray(measured, dtype=float))))
    t = float(theory)
    bound = rtol * abs(t) + atol
    name = label or "数值落在理论容差内"
    info = f"|实测−理论| = {fmt_value(abs(m - t))}，容差 rtol={rtol:g}, atol={atol:g}" + (f"；{detail}" if detail else "")
    if not abs(m - t) <= bound:
        fail(label=name, theory=t, measured=m, tol=f"±({rtol:g}·|t|+{atol:g})", detail=info, kind="B")
    record(label=name, theory=t, measured=m, tol=f"±({rtol:g}·|t|+{atol:g})", passed=True, detail=info, kind="B")
    return m


def assert_leq(
    measured: Any,
    bound: Any,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
    label: str | None = None,
    detail: str = "",
) -> float:
    """断言 ``measured ≤ bound``（允许 ``rtol·|bound| + atol`` 的舍入松弛）——用于"界"类结论。"""
    m = float(np.max(np.abs(np.asarray(measured, dtype=float))))
    b = float(bound)
    slack = rtol * abs(b) + atol
    name = label or "实测值不超过理论界"
    info = f"松弛 rtol={rtol:g}, atol={atol:g}，余量 = {fmt_value(b + slack - m)}" + (f"；{detail}" if detail else "")
    if not m <= b + slack:
        fail(label=name, theory=f"≤ {fmt_value(b)}", measured=m, tol=f"+{fmt_value(slack)}", detail=info, kind="B")
    record(label=name, theory=f"≤ {fmt_value(b)}", measured=m, tol=f"+{fmt_value(slack)}", passed=True, detail=info, kind="B")
    return m


def assert_all_finite(values: Any, *, label: str | None = None, detail: str = "") -> None:
    """断言全部元素有限（无 nan / ±inf）——数值格式崩溃的第一道警报。"""
    arr = np.asarray(values, dtype=float)
    n_bad = int(np.count_nonzero(~np.isfinite(arr)))
    name = label or "全部有限"
    info = f"共 {arr.size} 个元素，异常 {n_bad} 个" + (f"；{detail}" if detail else "")
    if n_bad:
        fail(label=name, theory="0 个非有限值", measured=n_bad, tol="0", detail=info, kind="B")
    record(label=name, theory="0 个非有限值", measured=0, tol="0", passed=True, detail=info, kind="B")


def assert_conserved(
    values: Any,
    *,
    quantity: str = "总量",
    rtol: float | None = None,
    atol: float | None = None,
    factor: float = 8.0,
    label: str | None = None,
    detail: str = "",
) -> float:
    """断言时间序列 ``values`` 上的守恒量不漂移。

    ``values`` 是守恒量随时间的取值序列（如总质量、总能量、总概率）。
    缺省容差为机器精度界 ``8·n·u·max|v|``（n = 序列长度），来源写进报告。
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        raise ValueError("空序列")
    scale = float(np.max(np.abs(v))) if v.size else 1.0
    if rtol is None and atol is None:
        tol, src = _tol_or_machine(None, scale=scale, n_ops=v.size, factor=factor)
        atol_eff = tol
    else:
        atol_eff = float(atol if atol is not None else 0.0)
        tol = atol_eff + float(rtol if rtol is not None else 0.0) * scale
        src = "explicit"
    drift = float(np.max(np.abs(v - v[0])))
    name = label or f"{quantity}守恒"
    info = (
        f"最大漂移 {fmt_value(drift)}（相对 {fmt_value(drift / scale if scale else 0.0)}），"
        f"{v.size} 步，量级 {fmt_value(scale)}，容差来源 {src}"
        + (f"；{detail}" if detail else "")
    )
    if not drift <= tol:
        fail(label=name, theory=f"漂移 ≤ {fmt_value(tol)}", measured=drift, tol=fmt_value(tol), detail=info, kind="B")
    record(label=name, theory=f"漂移 ≤ {fmt_value(tol)}", measured=drift, tol=fmt_value(tol), passed=True, detail=info, kind="B")
    return drift


def assert_bounded_drift(
    series: Any,
    *,
    tol: float | None = None,
    factor: float = 8.0,
    label: str | None = None,
    detail: str = "",
) -> float:
    """断言序列**有界振荡**（如辛方法的能量误差）而非单向漂移。

    判据：``max|v − v₀| ≤ tol``，且首尾之差不超过同样界。
    """
    v = np.asarray(series, dtype=float).ravel()
    if v.size < 2:
        raise ValueError("序列太短")
    scale = max(float(np.max(np.abs(v))), float(np.max(np.abs(v - v[0]))), 1e-300)
    if tol is None:
        t, src = _tol_or_machine(None, scale=scale, n_ops=v.size, factor=factor)
    else:
        t, src = float(tol), "explicit"
    amp = float(np.max(np.abs(v - v[0])))
    end = float(abs(v[-1] - v[0]))
    name = label or "有界振荡（无长期漂移）"
    info = f"振幅 {fmt_value(amp)}，首尾差 {fmt_value(end)}，{v.size} 步，容差来源 {src}" + (f"；{detail}" if detail else "")
    if not (amp <= t and end <= t):
        fail(label=name, theory=f"有界 ≤ {fmt_value(t)}", measured=max(amp, end), tol=fmt_value(t), detail=info, kind="B")
    record(label=name, theory=f"有界 ≤ {fmt_value(t)}", measured=max(amp, end), tol=fmt_value(t), passed=True, detail=info, kind="B")
    return amp


def assert_linear_drift(
    series: Any,
    *,
    min_slope: float | None = None,
    label: str | None = None,
    detail: str = "",
) -> float:
    """断言序列存在**线性漂移**（用于对照"非辛方法能量线性爬升"）。

    用最小二乘拟合 ``v_k ≈ a + b·k``，断言 ``b ≠ 0``（或 ``|b| ≥ min_slope``）。
    """
    v = np.asarray(series, dtype=float).ravel()
    if v.size < 3:
        raise ValueError("序列太短")
    k = np.arange(v.size, dtype=float)
    km, vm = float(np.mean(k)), float(np.mean(v))
    sxx = float(np.sum((k - km) ** 2))
    b = float(np.sum((k - km) * (v - vm)) / sxx)
    name = label or "线性漂移"
    info = f"斜率 {fmt_value(b)}/步，共 {v.size} 步" + (f"；{detail}" if detail else "")
    if min_slope is None:
        ok = abs(b) > 0.0
        theory: Any = "斜率 ≠ 0"
    else:
        ok = abs(b) >= min_slope
        theory = f"|斜率| ≥ {min_slope:g}"
    if not ok:
        fail(label=name, theory=theory, measured=b, tol=theory, detail=info, kind="B")
    record(label=name, theory=theory, measured=b, tol=theory, passed=True, detail=info, kind="B")
    return b


def assert_monotone(
    values: Any,
    *,
    direction: Literal["nonincreasing", "nondecreasing", "increasing", "decreasing"] = "nonincreasing",
    label: str | None = None,
    detail: str = "",
) -> None:
    """断言序列单调（总变差、误差范数、残差这类量的常见不变量）。"""
    v = np.asarray(values, dtype=float).ravel()
    if v.size < 2:
        raise ValueError("序列太短")
    d = np.diff(v)
    scale = max(float(np.max(np.abs(v))), 1e-300)
    slack = scaled_tol(scale, n_ops=v.size)
    if direction == "nonincreasing":
        bad = d > slack
    elif direction == "nondecreasing":
        bad = d < -slack
    elif direction == "increasing":
        bad = d <= 0
    else:
        bad = d >= 0
    worst = float(np.max(d)) if d.size else 0.0
    name = label or f"单调（{direction}）"
    info = f"最大相邻增量 {fmt_value(worst)}，松弛 {fmt_value(slack)}（machine:{v.size}·u·scale）" + (
        f"；{detail}" if detail else ""
    )
    if bool(np.any(bad)):
        idx = int(np.argmax(bad))
        fail(
            label=name,
            theory=direction,
            measured=f"第 {idx}→{idx + 1} 项违反（差 {fmt_value(d[idx])}）",
            tol=fmt_value(slack),
            detail=info,
            kind="B",
        )
    record(label=name, theory=direction, measured=f"违反 0 处", tol=fmt_value(slack), passed=True, detail=info, kind="B")


def assert_orthogonal(Q: Any, *, tol: float | None = None, label: str | None = None, detail: str = "") -> float:
    """断言 ``QᵀQ = I``（Householder/Givens 正交化、正交多项式的核心不变量）。"""
    A = np.asarray(Q, dtype=float)
    if A.ndim != 2:
        raise ValueError("Q 必须是二维矩阵")
    m, n = A.shape
    G = A.T @ A - np.eye(n)
    err = max_abs(G)
    scale = max(max_abs(A), 1.0)
    if tol is None:
        t, src = _tol_or_machine(None, scale=scale**2, n_ops=m)
    else:
        t, src = float(tol), "explicit"
    name = label or "正交性 QᵀQ = I"
    info = f"‖QᵀQ−I‖_∞ = {fmt_value(err)}，Q 为 {m}×{n}，容差来源 {src}" + (f"；{detail}" if detail else "")
    if not err <= t:
        fail(label=name, theory=f"≤ {fmt_value(t)}", measured=err, tol=fmt_value(t), detail=info, kind="B")
    record(label=name, theory=f"≤ {fmt_value(t)}", measured=err, tol=fmt_value(t), passed=True, detail=info, kind="B")
    return err


def assert_symmetric(A: Any, *, tol: float | None = None, label: str | None = None, detail: str = "") -> float:
    """断言矩阵对称：``‖A − Aᵀ‖_∞ ≤ tol``。"""
    M = np.asarray(A, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("A 必须是方阵")
    err = max_abs(M - M.T)
    if tol is None:
        t, src = _tol_or_machine(None, scale=max(max_abs(M), 1.0), n_ops=M.shape[0])
    else:
        t, src = float(tol), "explicit"
    name = label or "对称性 A = Aᵀ"
    info = f"‖A−Aᵀ‖_∞ = {fmt_value(err)}，n={M.shape[0]}，容差来源 {src}" + (f"；{detail}" if detail else "")
    if not err <= t:
        fail(label=name, theory=f"≤ {fmt_value(t)}", measured=err, tol=fmt_value(t), detail=info, kind="B")
    record(label=name, theory=f"≤ {fmt_value(t)}", measured=err, tol=fmt_value(t), passed=True, detail=info, kind="B")
    return err
