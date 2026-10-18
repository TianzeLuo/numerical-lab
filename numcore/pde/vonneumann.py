r"""von Neumann 稳定性分析：放大因子、稳定域、数值耗散与"相位"（符号翻转）。

数学定义
--------
把初值展开成 Fourier 模态 :math:`u_j^n = G^n e^{\mathrm{i}\varphi j}`
（:math:`\varphi = k\Delta x \in [0,\pi]` 是无量纲波数，:math:`\pi` 对应网格能表示的
最短波，即 Nyquist），代入一维热传导的 theta 族格式，得到**每个模态一个乘子**：

.. math::
    G_\theta(\varphi)
      = \frac{1 - 4(1-\theta)r\sin^2(\varphi/2)}{1 + 4\theta r\sin^2(\varphi/2)},
    \qquad r = \frac{\alpha\Delta t}{\Delta x^2}.

三个特例：:math:`\theta=0`（FTCS）:math:`G = 1 - 4r\sin^2(\varphi/2)`；
:math:`\theta=1`（BTCS）:math:`G = 1/(1+4r\sin^2(\varphi/2))`；
:math:`\theta=1/2`（Crank--Nicolson）:math:`G = (1-2r\sin^2(\varphi/2))/(1+2r\sin^2(\varphi/2))`。

稳定性判据是

.. math::
    \max_{\varphi\in[0,\pi]} |G_\theta(\varphi)| \le 1 .

因为分子关于 :math:`s^2=\sin^2(\varphi/2)` 单调减、分母单调增，极值只可能在
:math:`\varphi=0`（:math:`G=1`）或 :math:`\varphi=\pi`（:math:`G=1-4r`）取到，
于是**稳定性可以只用两个数判定**：:math:`r\le1/2`（FTCS）与"恒成立"（:math:`\theta\ge1/2`）。
配合 Lax 等价定理（相容 + 稳定 :math:`\iff` 收敛）就得到本章的主结论。

三条最容易骗人的地方（都在这个模块里有对应的函数/数字）
--------------------------------------------------------
1. **:math:`r\le1/2` 只对"纯 FTCS + Dirichlet + 常系数 + 一维"成立**：
   二维 :math:`r\le1/4`（:func:`critical_r_2d`），变系数要用 :math:`\max_j\lambda_j`
   （:func:`numcore.pde.heat.explicit_growth`），Neumann/周期在**有限网格**上的阈值
   恰好是 1/2 而 Dirichlet 略大于 1/2（:func:`critical_r_finite`）。
2. **无条件稳定 :math:`\ne` 精确**：BTCS 在 :math:`r=50` 时依然 :math:`|G|\le1`，
   但 :math:`G(\pi)=1/(1+200)\approx5\times10^{-3}`——一步把最高频抹掉 99.5\%，
   误差是 :math:`O(\Delta t)` 且常数巨大（见 :func:`symbol_expansion` 与本章断言）。
3. **"看起来光滑"不等于收敛**：:math:`G<0` 时模态每步**翻符号**，
   :math:`|G|` 又接近 1，画出来是"规则的细密振荡"，不像发散。
   翻转阈值由 :func:`sign_flip_wavenumber` 给出（:math:`r>1/4` 才存在）。

实现约束
--------
只有解析式与逐元素运算；"实测"一律调用 :mod:`numcore.pde.heat` 里真正的推进器
（周期网格用循环追赶法），不去复制一份公式假装是测量。不 import numcore.reference（P10）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from numcore.pde.heat import (
    discrete_sine_eigenvalue,
    exact_semidiscrete_sine,
    theta_step,
    uniform_grid,
)

__all__ = [
    "amplification_theta",
    "amplification_ftcs",
    "amplification_btcs",
    "amplification_cn",
    "amplification",
    "amplification_scan",
    "max_abs_amplification",
    "critical_r",
    "critical_r_finite",
    "critical_r_2d",
    "sign_flip_wavenumber",
    "amplification_phase",
    "phase_error",
    "mode_wavenumbers",
    "nyquist_wavenumber",
    "mode_decay",
    "mode_amplitudes",
    "mode_decay_measured",
    "mode_decay_resolved",
    "exact_decay",
    "dissipation_error",
    "dissipation_coefficient",
    "symbol_expansion",
    "numerical_amplification",
    "numerical_amplification_sine",
    "discrete_mode_chain",
    "semidiscrete_error_analytic",
    "time_error_leading",
    "amplification_2d_ftcs",
    "max_abs_amplification_2d",
    "adi_amplification_2d",
    "max_abs_adi_amplification_2d",
]


# --------------------------------------------------------------------------- #
# 放大因子：解析式
# --------------------------------------------------------------------------- #
def amplification_theta(phi: Any, r: float, theta: float = 0.5) -> Any:
    r"""theta 族的放大因子 :math:`G_\theta(\varphi) = \dfrac{1-4(1-\theta)r\sin^2(\varphi/2)}{1+4\theta r\sin^2(\varphi/2)}`。

    ``phi`` 可以是数组；``r`` 为负或 :math:`\theta\notin[0,1]` 时抛 :class:`ValueError`
    （反扩散与"外推格式"都会无条件不稳定，不该静默算出一个数）。
    """
    rr = float(r)
    th = float(theta)
    if rr < 0.0:
        raise ValueError(f"r 不能为负，得到 {r}")
    if not 0.0 <= th <= 1.0:
        raise ValueError(f"theta 必须在 [0,1] 内，得到 {theta}")
    p = np.asarray(phi, dtype=float)
    if np.any(p < 0.0) or np.any(p > np.pi + 1e-12):
        raise ValueError("波数 phi 必须落在 [0, pi] 内（pi 之外是网格无法表示的模态）")
    s2 = np.sin(p / 2.0) ** 2
    return (1.0 - 4.0 * (1.0 - th) * rr * s2) / (1.0 + 4.0 * th * rr * s2)


def amplification_ftcs(phi: Any, r: float) -> Any:
    r"""FTCS（显式）放大因子 :math:`G = 1 - 4r\sin^2(\varphi/2)`（:math:`|G|\le1\iff r\le1/2`）。"""
    return amplification_theta(phi, r, 0.0)


def amplification_btcs(phi: Any, r: float) -> Any:
    r"""BTCS（全隐式）放大因子 :math:`G = 1/(1+4r\sin^2(\varphi/2))`（无条件 :math:`0<G\le1`）。"""
    return amplification_theta(phi, r, 1.0)


def amplification_cn(phi: Any, r: float) -> Any:
    r"""Crank--Nicolson 放大因子 :math:`G = (1-2r\sin^2(\varphi/2))/(1+2r\sin^2(\varphi/2))`。

    :math:`r\to\infty` 时 :math:`G(\pi)\to-1`：**不增长，但每步翻符号**——
    "无条件稳定"在这里长得最像不稳定。
    """
    return amplification_theta(phi, r, 0.5)


THETA_BY_NAME = {"ftcs": 0.0, "btcs": 1.0, "cn": 0.5, "theta": None}


def amplification(scheme: str, phi: Any, r: float, *, theta: float | None = None) -> Any:
    """按格式名取放大因子（``ftcs`` / ``btcs`` / ``cn`` / ``theta``）。"""
    name = str(scheme).lower()
    if name not in THETA_BY_NAME:
        raise ValueError(f"未知格式 {scheme!r}（可用：{sorted(THETA_BY_NAME)}）")
    th = THETA_BY_NAME[name]
    if th is None:
        if theta is None:
            raise ValueError("scheme='theta' 需要显式给出 theta")
        th = float(theta)
    return amplification_theta(phi, r, th)


def amplification_scan(scheme: str, r: float, *, n_phi: int = 241, theta: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """在 :math:`\\varphi\\in[0,\\pi]` 上等距扫描放大因子，返回 ``(phis, g)``。"""
    if n_phi < 3:
        raise ValueError(f"n_phi 至少 3，得到 {n_phi}")
    phis = np.linspace(0.0, np.pi, int(n_phi))
    return phis, amplification(scheme, phis, r, theta=theta)


def max_abs_amplification(scheme: str, r: float, *, theta: float | None = None) -> float:
    r"""**解析**的 :math:`\max_\varphi|G|`：FTCS 是 :math:`\max(1,|1-4r|)`，隐式/CN 恒为 1。

    之所以能只看两端：分子关于 :math:`\sin^2(\varphi/2)` 单调减、分母单调增，
    复合后 :math:`G` 单调，故极值必在 :math:`\varphi\in\{0,\pi\}`。
    """
    if r < 0.0:
        raise ValueError(f"r 不能为负，得到 {r}")
    name = str(scheme).lower()
    if name not in THETA_BY_NAME:
        raise ValueError(f"未知格式 {scheme!r}（可用：{sorted(THETA_BY_NAME)}）")
    th = THETA_BY_NAME[name] if THETA_BY_NAME[name] is not None else (0.5 if theta is None else float(theta))
    if th == 0.0:
        return float(max(1.0, abs(1.0 - 4.0 * r)))
    return 1.0


def critical_r(scheme: str) -> float:
    r"""连续极限下的稳定阈值：``ftcs`` 是 :math:`1/2`，隐式与 CN 是 :math:`+\infty`（无条件稳定）。

    返回 ``inf`` 是**有意的**：它表示"没有有限阈值"。写进帧时必须转成 ``None``
    并在 ``strings`` 里说明，因为 JSON 没有 Infinity（PROTOCOLS §1.3）。
    """
    name = str(scheme).lower()
    if name not in THETA_BY_NAME:
        raise ValueError(f"未知格式 {scheme!r}（可用：{sorted(THETA_BY_NAME)}）")
    th = THETA_BY_NAME[name]
    if th is None:
        raise ValueError("scheme='theta' 没有单一阈值：请用 1/(2(1-2theta)) 自行判定，或用 critical_r_finite")
    if th < 0.5:
        return float(1.0 / (2.0 * (1.0 - 2.0 * th)))
    return float("inf")


def critical_r_finite(nx: int, *, bc: str = "dirichlet", theta: float = 0.0) -> float:
    r"""**有限网格**上的稳定阈值 :math:`r_* = 1/\bigl(2(1-2\theta)\sin^2(\varphi_{\max}/2)\bigr)`。

    :math:`\varphi_{\max}` 取网格能表示的**最大**波数：

    * ``dirichlet``：内部只有 ``nx-2`` 个模态，:math:`\varphi_{\max}=(nx-2)\pi/(nx-1)<\pi`，
      于是 :math:`r_*>1/2`，且随 :math:`\Delta x\to0` 单调降到 1/2；
    * ``neumann``：模态 :math:`\varphi_k=k\pi/(nx-1)`（:math:`k=0..nx-1`）**含** :math:`\pi`，
      所以 :math:`r_*` 恰好等于 1/2；
    * ``periodic``：模态 :math:`\varphi_m=2\pi m/nx`。**偶数网格**含 :math:`\pi`，
      阈值恰为 1/2；**奇数网格**最大的模态只到 :math:`\varphi_{\max}=\pi(nx-1)/nx<\pi`，
      阈值因此严格大于 1/2（实测差 3e-4 量级，见本章断言）。

    这就是"1/2 只在连续极限下才是一句完整的话"的实测版本：三种边界（以及网格奇偶性）
    给出的有限网格阈值互不相同，教科书那个 1/2 是它们的公共极限。
    """
    n = int(nx)
    if n < 3:
        raise ValueError(f"nx 至少 3，得到 {n}")
    name = str(bc).lower()
    if name not in ("dirichlet", "neumann", "periodic"):
        raise ValueError(f"未知边界条件 {bc!r}")
    th = float(theta)
    if not 0.0 <= th < 0.5:
        return float("inf")
    if name == "dirichlet":
        phi_max = (n - 2) * np.pi / (n - 1)
    elif name == "neumann":
        phi_max = np.pi
    else:
        # 奇数周期网格上 m 与 n-m 是同一个模态，最大的那个只到 2π⌊n/2⌋/n
        phi_max = 2.0 * np.pi * (n // 2) / n
    s = float(np.sin(phi_max / 2.0))
    if s == 0.0:  # pragma: no cover
        return float("inf")
    return float(1.0 / (2.0 * (1.0 - 2.0 * th) * s * s))


def critical_r_2d(nx: int, ny: int | None = None) -> float:
    r"""**有限网格**上二维 FTCS 的阈值 :math:`r_* = 1/\bigl(2(\sin^2\frac{\varphi_x}{2}+\sin^2\frac{\varphi_y}{2})\bigr)`。

    连续极限是 :math:`1/4`：两个方向各贡献一个 :math:`4r`，最坏情形 :math:`|1-8r|\le1`。
    "一维的 1/2 搬到二维"是最常见的维度错误。
    """
    if ny is None:
        ny = nx
    if nx < 3 or ny < 3:
        raise ValueError(f"nx、ny 至少 3，得到 {nx}/{ny}")
    tx = (nx - 2) * np.pi / (nx - 1)
    ty = (ny - 2) * np.pi / (ny - 1)
    s = float(np.sin(tx / 2.0)) ** 2 + float(np.sin(ty / 2.0)) ** 2
    return float(1.0 / (2.0 * s))


# --------------------------------------------------------------------------- #
# 相位：这一族格式里"相位误差"就是符号翻转
# --------------------------------------------------------------------------- #
def sign_flip_wavenumber(r: float, *, theta: float = 0.0) -> float | None:
    r"""放大因子由正变负的波数 :math:`\varphi_*`（不存在时返回 ``None``）。

    对 FTCS 解 :math:`1-4r\sin^2(\varphi/2)=0` 得
    :math:`\varphi_*=2\arcsin\frac{1}{2\sqrt r}`，要求 :math:`r\ge1/4`。
    :math:`\varphi>\varphi_*` 的模态每步**翻一次符号**：它们仍然在衰减
    （:math:`|G|<1`），但图像上是"一格一格的规则振荡"，肉眼与发散完全不像。
    CN 在 :math:`r>1/2` 时也会出现同样的事（:math:`\varphi_*=\pi` 处 :math:`G<0`）。
    """
    rr = float(r)
    if rr < 0.0:
        raise ValueError(f"r 不能为负，得到 {r}")
    th = float(theta)
    if th >= 0.5:
        # 分子 1 - 4(1-θ)r s² 在 s²=1 处取 1 - 4(1-θ)r；θ=1/2 时需 r > 1/2
        val = 1.0 - 4.0 * (1.0 - th) * rr
        return float(np.pi) if val < 0.0 else None
    if rr < 0.25:
        return None
    arg = 1.0 / (2.0 * float(np.sqrt(rr)))
    if arg > 1.0:  # pragma: no cover - rr>=1/4 时 arg<=1
        return None
    return float(2.0 * np.arcsin(arg))


def amplification_phase(scheme: str, phi: Any, r: float, *, theta: float | None = None) -> np.ndarray:
    r"""放大因子的辐角 :math:`\arg G\in\{0,\pi\}`（这一族格式的 :math:`G` 恒为实数）。

    热方程的 theta 族**没有色散**（:math:`G` 是实函数），所以"相位误差"在这里
    只剩一件事：:math:`G<0` 时模态每步翻转 :math:`\pi`。对双曲型方程才是连续的颜色散，
    见 4.2 章。
    """
    g = np.asarray(amplification(scheme, phi, r, theta=theta), dtype=float)
    return np.where(g < 0.0, np.pi, 0.0)


def phase_error(scheme: str, phi: Any, r: float, *, theta: float | None = None) -> np.ndarray:
    r"""相对精确解（:math:`G_{\text{exact}}=e^{-r\varphi^2}>0`）的相位误差，取值 :math:`0` 或 :math:`\pi`。"""
    return amplification_phase(scheme, phi, r, theta=theta)


# --------------------------------------------------------------------------- #
# 波数网格与模态衰减
# --------------------------------------------------------------------------- #
def nyquist_wavenumber() -> float:
    r"""Nyquist 波数 :math:`\varphi=\pi`：网格能表示的最短波（每格反号）。"""
    return float(np.pi)


def mode_wavenumbers(nx: int, *, bc: str = "dirichlet") -> np.ndarray:
    r"""网格上**可分辨**的波数集合 :math:`\varphi_k`。

    * ``dirichlet``：:math:`\varphi_k = k\pi/(nx-1)`，:math:`k=1,\dots,nx-2`
      （端点被钉住，常数模态与"每格反号"的幽灵模态都不存在）；
    * ``neumann``：:math:`k=0,\dots,nx-1`（含常数模态，其 :math:`G\equiv1`）；
    * ``periodic``：:math:`\varphi_m = 2\pi m/nx`，:math:`m=0,\dots,\lfloor nx/2\rfloor`。

    "可分辨"这三个字是要紧的：:math:`\varphi=\pi` 那个模态在有限差分网格上
    无法与它的负波数区分，Dirichlet 网格里它甚至不是边界条件下的容许模态，
    于是**有限网格的阈值并不等于 1/2**（:func:`critical_r_finite`）。
    """
    n = int(nx)
    if n < 3:
        raise ValueError(f"nx 至少 3，得到 {n}")
    name = str(bc).lower()
    if name == "dirichlet":
        return np.arange(1, n - 1, dtype=float) * np.pi / (n - 1)
    if name == "neumann":
        return np.arange(0, n, dtype=float) * np.pi / (n - 1)
    if name == "periodic":
        return np.arange(0, n // 2 + 1, dtype=float) * (2.0 * np.pi / n)
    raise ValueError(f"未知边界条件 {bc!r}")


def mode_decay(scheme: str, phi: Any, r: float, n_steps: int, *, theta: float | None = None) -> np.ndarray:
    r"""**数值**存活率 :math:`|G(\varphi)|^{n}`（每个模态各衰减多少）。"""
    if n_steps < 0:
        raise ValueError(f"n_steps 不能为负，得到 {n_steps}")
    g = np.asarray(amplification(scheme, phi, r, theta=theta), dtype=float)
    return np.abs(g) ** int(n_steps)


def exact_decay(phi: Any, r: float, n_steps: int) -> np.ndarray:
    r"""**精确**存活率 :math:`e^{-r\varphi^2 n}`。

    推导：连续解是 :math:`e^{-\alpha\beta^2 t}`，而
    :math:`\alpha\beta^2t = \alpha(\varphi/\Delta x)^2\,n\Delta t = r\varphi^2 n`——
    注意这里**没有** :math:`\Delta x`，整条曲线只由 :math:`r` 和 :math:`\varphi` 决定。
    """
    if n_steps < 0:
        raise ValueError(f"n_steps 不能为负，得到 {n_steps}")
    p = np.asarray(phi, dtype=float)
    return np.exp(-float(r) * p * p * int(n_steps))


def dissipation_error(scheme: str, phi: Any, r: float, n_steps: int, *, theta: float | None = None) -> np.ndarray:
    r"""数值耗散误差 :math:`|G|^{n} - e^{-r\varphi^2 n}`（正 = 衰减得比真解慢）。"""
    return mode_decay(scheme, phi, r, n_steps, theta=theta) - exact_decay(phi, r, n_steps)


def dissipation_coefficient(scheme: str, r: float, *, theta: float | None = None) -> float:
    r"""单步放大因子 :math:`\varphi^4` 项与精确值 :math:`r^2/2` 的差（leading 耗散误差系数）。

    把 :math:`G` 在 :math:`\varphi=0` 附近展开（:func:`symbol_expansion`）得
    :math:`G = 1 - r\varphi^2 + \bigl(\frac{r}{12} + \theta r^2\bigr)\varphi^4 + O(\varphi^6)`，
    而 :math:`e^{-r\varphi^2} = 1 - r\varphi^2 + \frac{r^2}{2}\varphi^4 + O(\varphi^6)`，于是

    .. math::
        c_4^{\text{num}} - c_4^{\text{exact}} = \frac{r}{12} + r^2\Bigl(\theta - \frac12\Bigr).

    对 CN（:math:`\theta=1/2`）它就是 :math:`r/12>0`：**数值解衰减得比真解慢**。
    对 FTCS（:math:`\theta=0`）是 :math:`r(\frac1{12}-\frac r2)`，在 :math:`r>1/6` 后变号——
    同一族格式里，"数值耗散"的方向可以随 :math:`r` 反转。
    """
    th = float(theta) if theta is not None else THETA_BY_NAME.get(str(scheme).lower())
    if th is None:
        raise ValueError("scheme='theta' 需要显式给出 theta")
    rr = float(r)
    return float(rr / 12.0 + rr * rr * (th - 0.5))


def symbol_expansion(scheme: str, r: float, *, theta: float | None = None) -> tuple[float, float, float]:
    r"""放大因子在 :math:`\varphi=0` 的展开系数 :math:`(c_0,c_2,c_4)`，:math:`G=c_0+c_2\varphi^2+c_4\varphi^4+O(\varphi^6)`。

    闭式：:math:`c_0=1`、:math:`c_2=-r`、:math:`c_4=\frac r{12}+\theta r^2`。
    这里给出的是**解析**系数；测试用 :math:`G(\varphi)` 在极小 :math:`\varphi` 上的取值
    做 Richardson 外推去核对它（两条独立路线）。
    """
    th = float(theta) if theta is not None else THETA_BY_NAME.get(str(scheme).lower())
    if th is None:
        raise ValueError("scheme='theta' 需要显式给出 theta")
    rr = float(r)
    return (1.0, -rr, rr / 12.0 + th * rr * rr)


# --------------------------------------------------------------------------- #
# 用真正的推进器做"实测"
# --------------------------------------------------------------------------- #
def _rayleigh_ratio(u_old: np.ndarray, u_new: np.ndarray) -> float:
    """单模态下的放大因子实测值 :math:`\\langle u_{new},u_{old}\\rangle/\\langle u_{old},u_{old}\\rangle`。

    为什么不直接取 ``u_new[j]/u_old[j]``：单点比值在节点接近零点时会被舍入放大，
    而 Rayleigh 商对特征向量是**精确**的，并且对我不小心混进来的其它模态分量不敏感
    （它给的是加权平均，只有混入量可忽略时才等于特征值——这一点在断言里也测了）。
    """
    num = float(np.dot(u_new, u_old))
    den = float(np.dot(u_old, u_old))
    if den == 0.0:
        raise ValueError("参考向量为零，无法测放大因子")
    return num / den


def numerical_amplification(
    scheme: str, *, m: int, nx: int, r: float, theta: float | None = None, n_steps: int = 1
) -> tuple[float, float]:
    r"""**实测**周期网格上第 ``m`` 个模态的放大因子，返回 ``(phi, G_measured)``。

    做法是把 :math:`u_j=\cos(\varphi_m j)`（:math:`\varphi_m=2\pi m/n_x`，周期网格上
    它是**精确**特征向量，不需要复数）真的喂给 :func:`numcore.pde.heat.theta_step`，
    推进 ``n_steps`` 步后用 Rayleigh 商取比值，再开 ``n_steps`` 次方。
    周期边界会走 Sherman--Morrison 循环追赶法，所以这条断言同时钉住了
    放大因子公式、循环求解器与时间推进三件事。
    """
    n = int(nx)
    if not 0 <= int(m) <= n // 2:
        raise ValueError(f"周期网格上的模态号 m 必须在 0..{n // 2} 之间，得到 {m}")
    if n_steps < 1:
        raise ValueError(f"n_steps 至少 1，得到 {n_steps}")
    phi = 2.0 * np.pi * int(m) / n
    grid = uniform_grid(n, periodic=True)
    dx = grid.dx
    u = np.cos(phi * np.arange(n, dtype=float))
    u0 = u.copy()
    th = THETA_BY_NAME.get(str(scheme).lower()) if theta is None else float(theta)
    if th is None:
        th = 0.5
    dt = 1.0
    for _ in range(int(n_steps)):
        u = theta_step(u, r=r, theta=th, dx=dx, dt=dt, bc="periodic")
    ratio = _rayleigh_ratio(u0, u)
    if ratio <= 0.0:
        # 翻符号的模态：取绝对值并保留符号信息由调用方从解析式判断
        ratio = abs(ratio)
    return float(phi), float(ratio ** (1.0 / int(n_steps)))


def numerical_amplification_sine(
    scheme: str, *, k: int, nx: int, r: float, theta: float | None = None, n_steps: int = 1
) -> tuple[float, float]:
    r"""**实测** Dirichlet 网格上第 ``k`` 个正弦模态的放大因子，返回 ``(\varphi_k, G)``。

    :math:`\sin(k\pi x_j/L)` 是零 Dirichlet 数据下离散 Laplacian 的**精确**特征向量，
    所以这条测量与 :func:`numerical_amplification`（周期 + 余弦）是两条独立路线，
    它们必须给出同一个 :math:`G(\varphi)` 公式。
    """
    n = int(nx)
    if not 1 <= int(k) <= n - 2:
        raise ValueError(f"模态号 k 必须在 1..{n - 2} 之间，得到 {k}")
    if n_steps < 1:
        raise ValueError(f"n_steps 至少 1，得到 {n_steps}")
    phi = int(k) * np.pi / (n - 1)
    grid = uniform_grid(n)
    u = np.sin(phi * np.arange(n, dtype=float))
    u0 = u.copy()
    th = THETA_BY_NAME.get(str(scheme).lower()) if theta is None else float(theta)
    if th is None:
        th = 0.5
    for _ in range(int(n_steps)):
        u = theta_step(u, r=r, theta=th, dx=grid.dx, dt=1.0, bc="dirichlet")
    ratio = _rayleigh_ratio(u0, u)
    if ratio <= 0.0:
        ratio = abs(ratio)
    return float(phi), float(ratio ** (1.0 / int(n_steps)))


def discrete_mode_chain(scheme: str, *, k: int, nx: int, r: float, n_steps: int, theta: float | None = None) -> float:
    r"""解析的模态放大链 :math:`G(\varphi_k)^n`（**不开绝对值**，保留翻符号信息）。

    与仿真对拍时用它当靶子：数值解逐点等于 :math:`G^n\sin(k\pi x/L)`，
    所以实测误差应当**逐位**落在这个解析值上。
    """
    phi = int(k) * np.pi / (nx - 1)
    th = THETA_BY_NAME.get(str(scheme).lower()) if theta is None else float(theta)
    if th is None:
        th = 0.5
    return float(np.asarray(amplification_theta(phi, r, th)) ** int(n_steps))


def semidiscrete_error_analytic(
    scheme: str, *, k: int, nx: int, r: float, n_steps: int, alpha: float = 1.0, length: float = 1.0, theta: float | None = None
) -> float:
    r"""单模态初值下，格式与**半离散精确解**之间的解析误差 :math:`|G^n - e^{-\lambda_k t}|`。

    这是测**时间阶**的正确靶子：靶子里已经含有离散特征值
    :math:`\lambda_k`（:func:`numcore.pde.heat.discrete_sine_eigenvalue`），
    空间误差被解析地扣掉。拿连续解当靶子会得到"CN 是 0 阶"这种荒唐结论
    （实测：固定 ``nx=41``、把 :math:`r` 从 0.4 减到 0.05，对连续解的实测阶是
    :math:`-0.0004`，对半离散解是 :math:`2.0000`）。
    """
    lam = discrete_sine_eigenvalue(nx, k, alpha=alpha, length=length)
    dx = length / (nx - 1)
    dt = r * dx * dx / alpha
    t = int(n_steps) * dt
    return float(abs(discrete_mode_chain(scheme, k=k, nx=nx, r=r, n_steps=n_steps, theta=theta) - float(np.exp(-lam * t))))


def time_error_leading(
    scheme: str, *, nx: int, r: float, n_steps: int, k: int = 1, alpha: float = 1.0, length: float = 1.0,
    theta: float | None = None
) -> float:
    r"""单模态时间误差的**首项闭式**（带符号）：:math:`u^n-u_{\text{sd}}(t)\approx c\sin(k\pi x/L)`。

    把 :math:`\log G` 在 :math:`z=\lambda\Delta t` 处展开、乘上 :math:`n`，再用
    :math:`n\Delta t=t`：

    * FTCS（:math:`G=1-z`）：:math:`\log G=-z-\frac{z^2}{2}-\cdots`
      :math:`\;\Longrightarrow\; c=-e^{-\lambda t}\dfrac{t\lambda^2\Delta t}{2}`；
    * BTCS（:math:`G=1/(1+z)`）：:math:`\log G=-z+\frac{z^2}{2}-\cdots`
      :math:`\;\Longrightarrow\; c=+e^{-\lambda t}\dfrac{t\lambda^2\Delta t}{2}`；
    * CN（:math:`G=\frac{1-z/2}{1+z/2}`）：:math:`\log G=-z-\frac{z^3}{12}-\cdots`
      :math:`\;\Longrightarrow\; c=-e^{-\lambda t}\dfrac{t\lambda^3\Delta t^2}{12}`。

    实测（``nx=41``，正弦模态，对半离散解）：FTCS :math:`r=0.4` 是
    :math:`-3.7189\times10^{-4}`（首项预测 :math:`-3.7139\times10^{-4}`，比 1.0013），
    BTCS 是 :math:`+3.7089\times10^{-4}`（预测 :math:`+3.7139\times10^{-4}`，比 0.9987），
    CN 是 :math:`-1.526486\times10^{-7}`（预测 :math:`-1.526485\times10^{-7}`，比 1.000001）。

    注意符号：**BTCS 衰减得比半离散解慢，FTCS/CN 衰减得更快**——
    这是"数值耗散"在时间方向上的同一件事（与 :func:`dissipation_coefficient` 的
    :math:`\varphi^4` 符号一致）。
    """
    dx = length / (nx - 1)
    dt = r * dx * dx / alpha
    t = int(n_steps) * dt
    lam = discrete_sine_eigenvalue(nx, k, alpha=alpha, length=length)
    th = float(theta) if theta is not None else THETA_BY_NAME.get(str(scheme).lower())
    if th is None:
        raise ValueError("scheme='theta' 需要显式给出 theta")
    amp = float(np.exp(-lam * t))
    if abs(th - 0.5) < 1e-15:
        return float(-amp * t * lam**3 * dt * dt / 12.0)
    # 符号：(2θ−1)。θ=0（FTCS）为负、θ=1（BTCS）为正——
    # 即"显式衰减得比半离散解快、隐式慢"。
    return float((2.0 * th - 1.0) * amp * t * lam * lam * dt / 2.0)


def semidiscrete_reference(
    x: Any, t: float, *, nx: int, k: int = 1, alpha: float = 1.0, length: float = 1.0
) -> np.ndarray:
    """半离散精确解（转发 :func:`numcore.pde.heat.exact_semidiscrete_sine`，方便一处 import）。"""
    return exact_semidiscrete_sine(x, t, nx=nx, k=k, alpha=alpha, length=length)


# --------------------------------------------------------------------------- #
# 二维：显式与 ADI 的放大因子
# --------------------------------------------------------------------------- #
def amplification_2d_ftcs(phi_x: Any, phi_y: Any, r: float) -> Any:
    r"""二维显式格式的放大因子 :math:`G = 1 - 4r(\sin^2\frac{\varphi_x}{2}+\sin^2\frac{\varphi_y}{2})`。"""
    px = np.asarray(phi_x, dtype=float)
    py = np.asarray(phi_y, dtype=float)
    return 1.0 - 4.0 * float(r) * (np.sin(px / 2.0) ** 2 + np.sin(py / 2.0) ** 2)


def max_abs_amplification_2d(r: float) -> float:
    r"""二维显式格式的 :math:`\max|G| = \max(1,|1-8r|)`（两向都取 Nyquist）。"""
    rr = float(r)
    if rr < 0.0:
        raise ValueError(f"r 不能为负，得到 {r}")
    return float(max(1.0, abs(1.0 - 8.0 * rr)))


def adi_amplification_2d(phi_x: Any, phi_y: Any, r: float) -> Any:
    r"""Peaceman--Rachford ADI 一步的放大因子。

    .. math::
        G = \frac{(1-\frac r2 a_x)(1-\frac r2 a_y)}{(1+\frac r2 a_x)(1+\frac r2 a_y)},
        \qquad a_x = 4\sin^2\frac{\varphi_x}{2},\ a_y = 4\sin^2\frac{\varphi_y}{2}.

    每个因子都是 :math:`\frac{1-z}{1+z}`（:math:`z\ge0`）的形状，模恒 :math:`\le1`，
    所以 ADI **无条件稳定**——而且这个证明只用了一步代数，不需要任何区域图。
    """
    px = np.asarray(phi_x, dtype=float)
    py = np.asarray(phi_y, dtype=float)
    ax = 4.0 * np.sin(px / 2.0) ** 2
    ay = 4.0 * np.sin(py / 2.0) ** 2
    half = 0.5 * float(r)
    return ((1.0 - half * ax) * (1.0 - half * ay)) / ((1.0 + half * ax) * (1.0 + half * ay))


def max_abs_adi_amplification_2d(r: float) -> float:
    r"""ADI 的 :math:`\max|G|`：恒等于 1（在 :math:`\varphi_x=\varphi_y=0` 取到）。"""
    if r < 0.0:
        raise ValueError(f"r 不能为负，得到 {r}")
    return 1.0

# --------------------------------------------------------------------------- #
# 模态分解：把"真的跑出来的解"按特征模态拆开
# --------------------------------------------------------------------------- #
def mode_amplitudes(
    u: Any, *, bc: str = "dirichlet", k_max: int | None = None, length: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    r"""把向量 :math:`u` 投影到**离散特征模态**上，返回 ``(phis, amps)``。

    离散正弦/余弦族在网格上是**正交**的，所以这里的投影是精确分解、不是近似拟合：

    * ``dirichlet``：:math:`\varphi_k=k\pi/(n-1)`，:math:`k=1..n-2`，
      :math:`a_k=\langle u,\sin\rangle/\langle\sin,\sin\rangle`；
    * ``neumann``：:math:`\varphi_k=k\pi/(n-1)`，:math:`k=0..n-1`（含常数模态）。
      **必须用梯形权重** :math:`w_0=w_{n-1}=\frac12`：网格含两个端点时
      :math:`\cos` 族对普通内积**不正交**，实测 :math:`\cos(\pi x)` 在
      :math:`\cos(3\pi x)` 上会漏出 :math:`1/21=0.0476` 的假分量（:math:`O(1/n)`，
      不是舍入）。带上端点半权重后才是 DCT-II 的正交关系；
    * ``periodic``：:math:`\varphi_m=2\pi m/n`，:math:`m=0..\lfloor n/2\rfloor`，
      返回余弦与正弦两支的**合成幅值** :math:`\sqrt{a_m^2+b_m^2}`
      （即复数 Fourier 系数的模，模态的"能量"）。

    范数用逐元素求和现算，不假设 :math:`n/2` 之类的闭式——奇偶网格、不同的
    :math:`k` 取值范围都不会算错。周期网格上的余弦/正弦是 :math:`\pm` 成对的，
    所以只取 :math:`m\le\lfloor n/2\rfloor` 就够（另一半是同一个物理模态）。
    """
    arr = np.asarray(u, dtype=float).ravel()
    n = arr.size
    if n < 3:
        raise ValueError(f"模态分解需要至少 3 个格点，得到 {n}")
    name = str(bc).lower()
    idx = np.arange(n, dtype=float)
    if name == "dirichlet":
        ks = np.arange(1, n - 1) if k_max is None else np.arange(1, min(int(k_max), n - 2) + 1)
        phis = ks * np.pi / (n - 1)
        amps = np.empty(ks.size, dtype=float)
        for i, k in enumerate(ks):
            basis = np.sin(k * np.pi * idx / (n - 1))
            amps[i] = float(np.dot(arr, basis) / np.dot(basis, basis))
        return phis, amps
    if name == "neumann":
        ks = np.arange(0, n) if k_max is None else np.arange(0, min(int(k_max), n - 1) + 1)
        phis = ks * np.pi / (n - 1)
        amps = np.empty(ks.size, dtype=float)
        w = np.ones(n, dtype=float)
        w[0] = 0.5
        w[-1] = 0.5
        for i, k in enumerate(ks):
            basis = np.cos(k * np.pi * idx / (n - 1))
            amps[i] = float(np.dot(w * arr, basis) / np.dot(w * basis, basis))
        return phis, amps
    if name == "periodic":
        ms = np.arange(0, n // 2 + 1) if k_max is None else np.arange(0, min(int(k_max), n // 2) + 1)
        phis = ms * (2.0 * np.pi / n)
        amps = np.empty(ms.size, dtype=float)
        for i, m in enumerate(ms):
            c = np.cos(2.0 * np.pi * m * idx / n)
            sn = np.sin(2.0 * np.pi * m * idx / n)
            nc = float(np.dot(c, c))
            ns = float(np.dot(sn, sn))
            # m=0 与 m=n/2（偶网格）时正弦支恒为 0，不能拿 0 做分母（那会得到 0/0 的 nan）
            ac = float(np.dot(arr, c) / nc) if nc > 0.0 else 0.0
            asn = float(np.dot(arr, sn) / ns) if ns > 0.0 else 0.0
            amps[i] = float(np.hypot(ac, asn))
        return phis, amps
    raise ValueError(f"未知边界条件 {bc!r}")


def mode_decay_resolved(
    u0: Any, u: Any, *, bc: str, k_max: int | None = None, floor_rel: float = 1e-8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""逐模态存活率 :math:`a_k(t)/a_k(0)` **加上一个"这个比值能不能信"的掩码**。

    两个判据缺一不可：

    1. **分母**：:math:`|a_k(0)|\le\text{floor\_rel}\cdot\max_k|a_k(0)|` —— 初值里根本没有这个模态
       （纯 :math:`\sin(\pi x)` 初值下 :math:`k\ge2` 的系数精确为 0，直接相除会得到 22.1 这种假放大率）；
    2. **分子**：:math:`|a_k(t)|\le\text{floor\_rel}\cdot\max_k|a_k(t)|` —— 解里这个模态已经被
       **别的模态的舍入**淹没了。

    第二条是本章实测出来的：FTCS 在 :math:`r=0.8`、:math:`t=0.05` 时最危险模态长到
    :math:`2.2^{100}\approx10^{34}`，其余模态的投影被 :math:`10^{34}\times10^{-16}=10^{18}`
    量级的舍入污染，"实测存活率"于是给出 :math:`1.5\times10^{2}` 的偏差——
    与公式不符，但错的不是公式，是这个比值本身已经不可测。
    噪声尺度必须取 :math:`\|u\|_\infty`（整个解的量级），不能取"投影出来这几个系数"的量级：
    后者会把噪声当信号（实测：只按投影系数定标时，8 个模态里有 7 个被误判为可测）。

    返回 ``(phis, ratio, alive)``，其中 ``ratio`` 在不可测处是 ``nan``。
    """
    phis, a0 = mode_amplitudes(u0, bc=bc, k_max=k_max)
    _, a1 = mode_amplitudes(u, bc=bc, k_max=k_max)
    ref0 = float(np.max(np.abs(a0))) if a0.size else 0.0
    # 分子的噪声尺度必须取**整个解**的量级，而不是"投影出来的这几个系数"的量级：
    # 解里只要有一个大幅度的别模态（例如 |G|>1 时被舍入激发出来的高频模态），
    # 每个系数上就叠着 eps·‖u‖∞ 的投影噪声。取 max|a1| 会把噪声当成信号。
    u_arr = np.asarray(u, dtype=float).ravel()
    ref1 = float(np.max(np.abs(u_arr))) if u_arr.size else 0.0
    if a1.size:
        ref1 = max(ref1, float(np.max(np.abs(a1))))
    ratio = np.full(a0.shape, np.nan, dtype=float)
    alive = np.zeros(a0.shape, dtype=bool)
    for i in range(a0.size):
        if ref0 > 0.0 and abs(a0[i]) <= floor_rel * ref0:
            continue
        if ref1 > 0.0 and abs(a1[i]) <= floor_rel * ref1:
            continue
        ratio[i] = a1[i] / a0[i]
        alive[i] = True
    return phis, ratio, alive


def mode_decay_measured(
    u0: Any, u: Any, *, bc: str, k_max: int | None = None, floor_rel: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """兼容入口：只要 ``(phis, ratio)``（不可测处是 ``nan``）。实现见 :func:`mode_decay_resolved`。"""
    phis, ratio, _ = mode_decay_resolved(u0, u, bc=bc, k_max=k_max, floor_rel=floor_rel)
    return phis, ratio
