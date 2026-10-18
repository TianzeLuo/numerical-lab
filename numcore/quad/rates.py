r"""2.4 章 —— "实测斜率"的唯一出处：幂律 / 指数律拟合与逐段局部阶。

这一章几乎每个可视化都要回答同一个问题：**实测的收敛阶是多少**。于是拟合的口径必须
只有一份实现，而且必须能说清三件事：

1. **窗口**：从哪个点开始才算进入渐近区（``skip`` 丢掉粗端的预渐近段），
   到哪里为止（``trim_tail`` 丢掉细端的舍入地板段）。两端的理由完全不同——
   粗端是"h 还不够小"，细端是"h 已经小到只剩噪声"；
2. **方向**：:math:`y\approx Cx^{-p}` 的对数-对数斜率是 **负数** :math:`-p`。
   本模块一律返回**原始斜率**（负的），要"阶"自己取负号——这样就不会出现
   "两边都乘了 −1 于是断言永远通过"的假证据；
3. **只喂幂律关系**：对数型关系（例如"有效位数 vs 误差"）也能拟合出一条直线，
   但那个斜率没有任何含义（1.1 章实测 0.047）。:func:`fit_loglog` 因此要求
   ``x>0``、``y>0``，并在点数不足时抛错而不是返回一个数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "SlopeFit",
    "fit_loglog",
    "fit_semilog",
    "local_order",
    "local_ratio",
    "theory_line_exponential",
    "theory_line_power",
]


@dataclass(frozen=True)
class SlopeFit:
    """一次斜率拟合的结果：斜率、截距、:math:`R^2`、用点数与窗口。"""

    slope: float
    intercept: float
    r2: float
    n_used: int
    window_x: tuple[float, float]
    window_y: tuple[float, float]

    def power_line(self, x: Any) -> np.ndarray:
        r"""由拟合结果重建幂律参考线 :math:`\hat y=e^{b}x^{s}`。"""
        xx = np.asarray(x, dtype=float)
        return np.exp(self.intercept) * xx**self.slope

    def exponential_line(self, x: Any) -> np.ndarray:
        r"""由拟合结果重建指数律参考线 :math:`\hat y=e^{b}e^{sx}`。"""
        xx = np.asarray(x, dtype=float)
        return np.exp(self.intercept) * np.exp(self.slope * xx)


def _prepare(x: Any, y: Any, skip: int, trim_tail: int) -> tuple[np.ndarray, np.ndarray]:
    xx = np.asarray(x, dtype=float).ravel()
    yy = np.asarray(y, dtype=float).ravel()
    if xx.size != yy.size:
        raise ValueError(f"x 与 y 长度不一致：{xx.size} vs {yy.size}")
    s = max(0, int(skip))
    t = max(0, int(trim_tail))
    stop = xx.size - t if t else xx.size
    xx, yy = xx[s:stop], yy[s:stop]
    if xx.size < 3:
        raise ValueError(f"拟合至少需要 3 个点，窗口内只有 {xx.size} 个（skip={s}, trim_tail={t}）")
    return xx, yy


def _least_squares(u: np.ndarray, v: np.ndarray) -> tuple[float, float, float]:
    """最小二乘 ``v ≈ a + b u``：返回 ``(b, a, r2)``（手写，不用被禁的成品拟合）。"""
    um, vm = float(np.mean(u)), float(np.mean(v))
    du, dv = u - um, v - vm
    denom = float(np.sum(du * du))
    if denom == 0.0:
        raise ValueError("拟合自变量全相同——没有可拟合的斜率")
    b = float(np.sum(du * dv) / denom)
    a = vm - b * um
    resid = dv - b * du
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum(dv * dv))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return b, a, r2


def fit_loglog(x: Any, y: Any, *, skip: int = 0, trim_tail: int = 0) -> SlopeFit:
    r"""幂律拟合 :math:`y\approx Cx^{s}`（返回**原始**斜率 :math:`s`，通常为负）。

    :math:`y\approx Cx^{-p}` 时 ``s = -p``：要"阶 p"请取 ``-fit.slope``。
    """
    xx, yy = _prepare(x, y, skip, trim_tail)
    if np.any(xx <= 0.0) or np.any(yy <= 0.0):
        raise ValueError("对数-对数拟合要求 x>0 且 y>0（0 或负值说明该点不该进窗口）")
    b, a, r2 = _least_squares(np.log(xx), np.log(yy))
    return SlopeFit(slope=b, intercept=a, r2=r2, n_used=int(xx.size),
                    window_x=(float(xx[0]), float(xx[-1])), window_y=(float(yy[0]), float(yy[-1])))


def fit_semilog(x: Any, y: Any, *, skip: int = 0, trim_tail: int = 0) -> SlopeFit:
    r"""指数律拟合 :math:`y\approx Ce^{sx}`（半对数斜率，负值表示衰减）。

    双指数（tanh--sinh）收敛必须用这一条：它的误差是 :math:`e^{-cN}`，
    在双对数图上**不是**直线，硬用 :func:`fit_loglog` 会得到一个没有含义的幂次。
    """
    xx, yy = _prepare(x, y, skip, trim_tail)
    if np.any(yy <= 0.0):
        raise ValueError("半对数拟合要求 y>0")
    b, a, r2 = _least_squares(xx, np.log(yy))
    return SlopeFit(slope=b, intercept=a, r2=r2, n_used=int(xx.size),
                    window_x=(float(xx[0]), float(xx[-1])), window_y=(float(yy[0]), float(yy[-1])))


def local_order(x: Any, y: Any) -> np.ndarray:
    r"""逐段局部阶 :math:`p_i=\ln(y_{i+1}/y_i)\big/\ln(x_i/x_{i+1})`（恒非负，当 y 递减）。

    它比一个全局拟合更能说明"阶在哪里换了"：预渐近段、渐近段、舍入地板段
    在三段上的局部阶会明显不同，而那正是"窗口"这个词的实测含义。
    """
    xx = np.asarray(x, dtype=float).ravel()
    yy = np.asarray(y, dtype=float).ravel()
    if xx.size < 2 or xx.size != yy.size:
        raise ValueError("局部阶至少需要 2 个等长点")
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.log(yy[1:] / yy[:-1])
        den = np.log(xx[:-1] / xx[1:])
        return num / den


def local_ratio(y: Any) -> np.ndarray:
    """相邻误差比 :math:`y_{i+1}/y_i`（Romberg 的"加速比"就是它的倒数形式）。"""
    yy = np.asarray(y, dtype=float).ravel()
    if yy.size < 2:
        raise ValueError("相邻比至少需要 2 个点")
    return yy[1:] / yy[:-1]


def theory_line_power(x: Any, slope: float, x0: float, y0: float) -> np.ndarray:
    r"""锚在 :math:`(x_0,y_0)` 上的幂律理论线 :math:`y=y_0(x/x_0)^{s}`。

    常数项 :math:`C` 是未知的（它依赖 :math:`f^{(p)}(\xi)`），所以理论线只能锚在
    实测窗口的第一个点上——**被断言的是斜率，不是高度**。这一点必须在图注里写明，
    否则"理论与实测两条线叠在一起"会被读成"常数也对上了"。
    """
    xx = np.asarray(x, dtype=float)
    return float(y0) * (xx / float(x0)) ** float(slope)


def theory_line_exponential(x: Any, rate: float, x0: float, y0: float) -> np.ndarray:
    r"""锚在 :math:`(x_0,y_0)` 上的指数理论线 :math:`y=y_0e^{s(x-x_0)}`。"""
    xx = np.asarray(x, dtype=float)
    return float(y0) * np.exp(float(rate) * (xx - float(x0)))
