"""解析参考问题：教材上能写出闭式解、或能化成已知特殊函数的"靶子"。

使用纪律：每个函数 docstring 第一行写清**它精确求解的是哪个方程/哪个量**。
本模块的函数只允许出现在测试与工具里（P10），不得被 ``numcore`` 其他部分调用——
否则"自研"就变成了"套公式"。
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np
import scipy.optimize as _sopt
import scipy.special as _sspec

__all__ = [
    # 插值与逼近
    "runge",
    "runge_d",
    "lebesgue_function",
    "lebesgue_constant",
    "interp_error_bound",
    "chebyshev_nodes",
    "equispaced_nodes",
    "gauss_legendre_nodes",
    # 变换
    "fourier_partial_sum",
    "gibbs_overshoot",
    # 积分
    "pendulum_period",
    "elliptic_k",
    # ODE / 动力系统
    "harmonic_oscillator",
    "kepler_solve",
    "kepler_orbit",
    "kepler_energy",
    "kepler_angular_momentum",
    "lorenz_reference_lyapunov",
    "logistic_period_doubling",
    "van_der_pol_limit_cycle_amplitude",
    # PDE
    "heat_sine_mode",
    "heat_dirichlet_series",
    "advection_periodic",
    "poisson_1d_sine",
]

#: Lorenz 系统标准参数下最大 Lyapunov 指数的公认值（Benettin 等，σ=10, ρ=28, β=8/3）。
lorenz_reference_lyapunov: float = 0.9056

#: logistic 映射 x→rx(1−x) 的周期倍增分岔点（前 5 个，解析值见注释）。
logistic_period_doubling: tuple[float, ...] = (
    3.0,  # 1→2：解析
    3.449489742783178,  # 2→4：1+√6
    3.5440903595519225,  # 4→8：数值（≈3.54409）
    3.5644072660954322,  # 8→16（≈3.5644）
    3.569945672704231,  # 混沌起始（≈3.56995，Feigenbaum 累积点）
)

#: van der Pol（μ=1）极限环的振幅（x 方向半幅）公认值 ≈ 2.0086。
van_der_pol_limit_cycle_amplitude: float = 2.008619860874843


# --------------------------------------------------------------------------- #
# 插值与逼近
# --------------------------------------------------------------------------- #
def runge(x: Any, a: float = 25.0) -> np.ndarray:
    """Runge 函数 ``f(x) = 1/(1 + a x²)``（经典取 ``a = 25``，区间 ``[-1, 1]``）。"""
    arr = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + a * arr**2)


def runge_d(x: Any, a: float = 25.0, n: int = 1) -> np.ndarray:
    """Runge 函数的 ``n`` 阶导数（用于插值误差界的解析上界）。"""
    arr = np.asarray(x, dtype=float)
    if n == 0:
        return runge(arr, a)
    if n == 1:
        return -2.0 * a * arr / (1.0 + a * arr**2) ** 2
    if n == 2:
        return (2.0 * a * (3.0 * a * arr**2 - 1.0)) / (1.0 + a * arr**2) ** 3
    raise NotImplementedError("只提供 0/1/2 阶导数")


def _lagrange_basis(nodes: np.ndarray, x: float) -> np.ndarray:
    """在 ``x`` 处取值的一组 Lagrange 基函数 ``ℓ_i(x)``。"""
    n = nodes.size
    out = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                out[i] *= (x - nodes[j]) / (nodes[i] - nodes[j])
    return out


def lebesgue_function(nodes: Any, xs: Any) -> np.ndarray:
    """Lebesgue 函数 ``Λ_n(x) = Σ_i |ℓ_i(x)|``（插值稳定性的度量）。"""
    nd = np.asarray(nodes, dtype=float)
    return np.array([float(np.sum(np.abs(_lagrange_basis(nd, float(x))))) for x in np.asarray(xs, dtype=float).ravel()])


def lebesgue_constant(nodes: Any, *, n_grid: int = 2001, a: float = -1.0, b: float = 1.0) -> float:
    """Lebesgue 常数 ``Λ_n = max_{x∈[a,b]} Λ_n(x)``（网格近似，网格足够密）。"""
    xs = np.linspace(a, b, int(n_grid))
    return float(np.max(lebesgue_function(nodes, xs)))


def interp_error_bound(f_sup: float, nodes: Any, x: float) -> float:
    """插值误差的解析上界 ``|f^{(n+1)}(ξ)|/(n+1)! · Π|x − x_i|``。

    ``f_sup`` 是 ``|f^{(n+1)}|`` 在区间上的上确界。
    """
    nd = np.asarray(nodes, dtype=float)
    n = nd.size
    prod = float(np.prod(np.abs(float(x) - nd)))
    fact = float(math.factorial(n))
    return f_sup / fact * prod


def chebyshev_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """Chebyshev–Lobatto 节点 ``x_k = cos(kπ/n)`` 映射到 ``[a, b]``（含端点，升序）。"""
    k = np.arange(int(n) + 1)
    z = np.cos(np.pi * k / int(n))[::-1]
    return 0.5 * (a + b) + 0.5 * (b - a) * z


def equispaced_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """等距节点（``n`` 等分 → ``n+1`` 个点）。"""
    return np.linspace(a, b, int(n) + 1)


def gauss_legendre_nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
    """``n`` 点 Gauss–Legendre 节点与权重（``numpy.polynomial.legendre.leggauss``，Newton 迭代到机器精度）。"""
    x, w = np.polynomial.legendre.leggauss(int(n))
    return np.asarray(x, dtype=float), np.asarray(w, dtype=float)


# --------------------------------------------------------------------------- #
# 变换
# --------------------------------------------------------------------------- #
def fourier_partial_sum(x: Any, terms: Sequence[float], harmonics: Sequence[int] | None = None) -> np.ndarray:
    """Fourier 部分和 ``Σ a_k sin(kπx)``（默认区间 ``[0,1]`` 上的正弦级数）。"""
    arr = np.asarray(x, dtype=float)
    ks = np.arange(1, len(terms) + 1) if harmonics is None else np.asarray(harmonics, dtype=int)
    out = np.zeros_like(arr)
    for a_k, k in zip(terms, ks):
        out += a_k * np.sin(k * np.pi * arr)
    return out


def gibbs_overshoot(*, n_terms: int = 4001) -> float:
    """±1 方波 Fourier 部分和在跳变处的**绝对过冲**。

    理论极限（Wilbraham–Gibbs 常数）：``(2/π)∫₀^π sin t / t dt − 1 ≈ 0.1789797``。
    注意两种常见表述：**相对于 2 的跳变高度**即 ``0.17898/2 ≈ 8.95%``——
    这正是"Gibbs 过冲约 9%"这个说法的来源（不要与"过冲 17.9%"混用，那是相对于半幅）。
    """
    xs = np.linspace(1e-9, 1.0, 200000)
    s = np.zeros_like(xs)
    for k in range(1, int(n_terms) + 1, 2):
        s += np.sin(2.0 * np.pi * k * xs) / k
    s *= 4.0 / np.pi
    return float(np.max(s) - 1.0)


# --------------------------------------------------------------------------- #
# 积分与特殊函数
# --------------------------------------------------------------------------- #
def elliptic_k(m: float) -> float:
    """第一类完全椭圆积分 ``K(m)``（参数约定：``m = k²``）。"""
    return float(_sspec.ellipk(float(m)))


def pendulum_period(theta0: float, length: float = 1.0, g: float = 9.81) -> float:
    """单摆（无阻尼、无小角近似）的精确周期 ``T = 4√(L/g)·K(sin²(θ₀/2))``。

    对照小角近似 ``T₀ = 2π√(L/g)``：``θ₀ = π/2`` 时精确周期比近似值大约 18%。
    """
    m = float(np.sin(theta0 / 2.0) ** 2)
    return 4.0 * float(np.sqrt(length / g)) * elliptic_k(m)


# --------------------------------------------------------------------------- #
# ODE 与动力系统
# --------------------------------------------------------------------------- #
def harmonic_oscillator(t: Any, omega: float = 1.0, x0: float = 1.0, v0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """``x'' + ω²x = 0`` 的精确解，返回 ``(x(t), v(t))``。"""
    tt = np.asarray(t, dtype=float)
    x = x0 * np.cos(omega * tt) + (v0 / omega) * np.sin(omega * tt)
    v = -x0 * omega * np.sin(omega * tt) + v0 * np.cos(omega * tt)
    return x, v


def kepler_solve(M: float, e: float, *, xtol: float = 1e-15) -> float:
    """解开普勒方程 ``E − e sin E = M``（``scipy.optimize.brentq`` 到 ``xtol``）。"""
    m = float(np.mod(M, 2.0 * np.pi))
    f = lambda E: E - e * np.sin(E) - m  # noqa: E731
    return float(_sopt.brentq(f, -np.pi, 2.0 * np.pi, xtol=xtol))


def kepler_orbit(t: Any, a: float = 1.0, e: float = 0.5, M0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """开普勒二体轨道（焦点在原点）：返回笛卡尔坐标 ``(x(t), y(t))``。

    取 ``μ = 1``、单位平均角速度 ``n = 1``（即 ``a = 1``），平近点角 ``M(t) = M0 + n t``。
    """
    tt = np.atleast_1d(np.asarray(t, dtype=float))
    xs = np.empty_like(tt)
    ys = np.empty_like(tt)
    for i, ti in enumerate(tt):
        E = kepler_solve(M0 + ti, e)
        xs[i] = a * (np.cos(E) - e)
        ys[i] = a * np.sqrt(1.0 - e**2) * np.sin(E)
    return xs, ys


def kepler_energy(x: Any, y: Any, vx: Any, vy: Any, mu: float = 1.0) -> np.ndarray:
    """二体问题的比能量 ``E = ½(vx²+vy²) − μ/r``（守恒量，用于 3.3 章）。"""
    r = np.sqrt(np.asarray(x, dtype=float) ** 2 + np.asarray(y, dtype=float) ** 2)
    return 0.5 * (np.asarray(vx, dtype=float) ** 2 + np.asarray(vy, dtype=float) ** 2) - mu / r


def kepler_angular_momentum(x: Any, y: Any, vx: Any, vy: Any) -> np.ndarray:
    """比角动量 ``L = x·vy − y·vx``（守恒量）。"""
    return np.asarray(x, dtype=float) * np.asarray(vy, dtype=float) - np.asarray(y, dtype=float) * np.asarray(vx, dtype=float)


# --------------------------------------------------------------------------- #
# PDE
# --------------------------------------------------------------------------- #
def heat_sine_mode(x: Any, t: float, *, alpha: float = 1.0, L: float = 1.0, k: int = 1) -> np.ndarray:
    """热方程 ``u_t = α u_xx``（``u(0)=u(L)=0``）的单模态精确解 ``sin(kπx/L)·exp(−α k²π² t/L²)``。"""
    arr = np.asarray(x, dtype=float)
    return np.sin(k * np.pi * arr / L) * float(np.exp(-alpha * (k * np.pi / L) ** 2 * t))


def heat_dirichlet_series(
    x: Any,
    t: float,
    *,
    f0: Callable[[float], float],
    alpha: float = 1.0,
    L: float = 1.0,
    modes: int = 60,
) -> np.ndarray:
    """热方程齐次 Dirichlet 问题的级数解（初值 ``f0``，系数用高精度求积算出）。

    ``u(x,t) = Σ_k b_k sin(kπx/L) exp(−α k²π²t/L²)``，
    ``b_k = (2/L)∫₀^L f0(ξ) sin(kπξ/L) dξ``（``scipy.integrate.quad``，``epsabs=1e-13``）。
    """
    import scipy.integrate as _sint

    arr = np.asarray(x, dtype=float)
    out = np.zeros_like(arr)
    for k in range(1, int(modes) + 1):
        bk = (2.0 / L) * _sint.quad(lambda xi: f0(float(xi)) * np.sin(k * np.pi * xi / L), 0.0, L, epsabs=1e-14, epsrel=1e-14, limit=200)[0]
        if abs(bk) < 1e-18:
            continue
        out += bk * np.sin(k * np.pi * arr / L) * float(np.exp(-alpha * (k * np.pi / L) ** 2 * t))
    return out


def advection_periodic(x: Any, t: float, *, c: float = 1.0, L: float = 1.0, f0: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """周期线性对流 ``u_t + c u_x = 0`` 的精确解 ``u(x,t) = f0((x − ct) mod L)``。"""
    arr = np.asarray(x, dtype=float)
    return f0(np.mod(arr - c * t, L))


def poisson_1d_sine(x: Any, *, k: int = 1, L: float = 1.0) -> np.ndarray:
    """``−u'' = sin(kπx/L)``、``u(0)=u(L)=0`` 的精确解 ``u = (L/(kπ))² sin(kπx/L)``。"""
    arr = np.asarray(x, dtype=float)
    return (L / (k * np.pi)) ** 2 * np.sin(k * np.pi * arr / L)
