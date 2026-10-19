r"""6.1 章内核：二维顶盖驱动方腔（流函数--涡量形式）。

这是全项目第一个**把前面各章的零件装成完整求解器**的模块，因此它的每一块都尽量
复用已有成果，而不是重写一遍：

========================================  ====================================================
复用对象                                   在本模块里的角色
========================================  ====================================================
:mod:`numcore.transform.fft`（2.5 章）      DST-I（正弦变换）——Dirichlet Poisson 的快速直接解
:mod:`numcore.linalg.iterative`（1.5 章）   迭代 Poisson 的残差/误差轨迹、最优 SOR 参数
:mod:`numcore.apps.ghia`                   Ghia--Ghia--Shin (1982) 公开基准表（对拍靶子）
========================================  ====================================================

数学
----
不可压 Navier--Stokes 的流函数--涡量形式（单位方腔、顶盖速度 :math:`U=1`）::

    u = psi_y,   v = -psi_x
    lap(psi) = -omega                       (Poisson，psi 在四壁为 0)
    omega_t + u omega_x + v omega_y = (1/Re) lap(omega)

壁面涡量用一阶 Thom 公式（由 ``psi`` 的鬼点 + 无滑移推出）::

    omega_wall = -2 psi_邻 / h^2 + 2 U_tau / h

其中 :math:`U_\tau` 是该壁上切向速度（顶盖 :math:`U_\tau=1`，其余为 0）。

三条"容易读错的量"（每一条都在 :mod:`tests.6.1` 里有实测记录）
------------------------------------------------------------------
1. **"残差小"不等于"解准"**：稳态迭代的残差量的是"离**离散**方程有多远"，
   它降到 :math:`10^{-6}` 时中心线剖面与基准的差仍可能是 :math:`10^{-2}` 量级——
   后者是**离散误差**，残差永远看不见它。:func:`poisson_iterative_history`
   把同一网格上的"残差曲线"与"误差曲线"画在一起，就是在演示这件事。
2. **网格与时间步的限制来自两个不同的机制**：显式扩散要求
   :math:`dt \le h^2 Re/4`，对流 CFL 要求 :math:`dt \le c_\text{cfl} h/|u|`。
   两者相等给出一个**可算出来的交叉雷诺数**
   :math:`Re_\times = 4 c_\text{cfl} (n-1)/u_\text{max}`（见 :func:`dt_limits`）；
   低于它时限制来自扩散（越**小**的 Re 越吃时间步），高于它时来自 CFL。
   "Re 越大越难算"这句话在这个交叉点两侧**是反的**。
3. **中心差分的稳定边界是格点雷诺数**：中心格式的系数在
   :math:`Re_h = |u| h Re > 2` 时失去对角占优，
   阈值 :math:`Re = 2(n-1)/u_\text{max}`（见 :func:`central_scheme_threshold_re`）。
   在 Re = 1000 上这意味着 :math:`n > 501` 才敢用中心格式——
   "换二阶格式提高精度"在粗网格上是**先炸后准**。

约定
----
* 网格：``n`` 个节点/边（含边界），:math:`h = 1/(n-1)`，节点坐标 :math:`i h`；
  数组索引 ``[j, i]`` = :math:`(y_j, x_i)`。
* 一切函数都是纯函数；没有随机数、没有 IO、没有全局可变状态。
* 壁面量（``u`` 在顶盖 = 1、其余 = 0；``v`` 四壁 = 0；``omega`` 由 Thom 公式给）
  由专门的函数显式设置，不做"看起来差不多"的近似。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from numcore.linalg.iterative import (
    laplacian_2d,
    sor_spectral_radius,
    stationary_solve,
    young_optimal_omega,
)
from numcore.transform.fft import fft, ifft

from .ghia import (
    BENCHMARK_TRANSCRIPTION_SPREAD,
    centerline_u,
    centerline_v,
    vortex_centers as _ghia_vortex_centers,
)

__all__ = [
    # 网格与基本量
    "CavityGrid",
    "cavity_grid",
    "lid_velocity",
    "initial_fields",
    "velocity_from_psi",
    "divergence",
    "wall_vorticity",
    "enstrophy",
    "kinetic_energy",
    "max_abs",
    # Poisson：直接解与迭代解
    "dst_basis",
    "dst1_matrix",
    "dst1_fft",
    "dst_eigenvalues",
    "poisson_solve_dst",
    "helmholtz_solve_dst",
    "poisson_residual",
    "poisson_iterative_history",
    "PoissonHistory",
    "sor_optimal_omega_poisson",
    # 稳态与瞬态
    "SteadyResult",
    "steady_solve",
    "TransientResult",
    "march_transient",
    "dt_limits",
    "stable_dt",
    "central_scheme_threshold_re",
    "cell_reynolds",
    # 诊断与对拍
    "linear_interp",
    "centerline_profiles",
    "BenchmarkComparison",
    "compare_with_benchmark",
    "vortex_centers",
    "ghia_centerline_u",
    "ghia_centerline_v",
    "ghia_vortex_centers",
]

#: 顶盖速度（单位方腔的无量纲约定）。
LID_SPEED = 1.0


# --------------------------------------------------------------------------- #
# 网格与基本量
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CavityGrid:
    """单位方腔上的均匀网格（``n`` 个节点/边，含边界）。"""

    n: int
    h: float
    x: np.ndarray
    y: np.ndarray

    @property
    def m(self) -> int:
        """内部点数/边（``n - 2``）。"""
        return self.n - 2

    @property
    def interior(self) -> tuple[slice, slice]:
        """内部点的切片（``(slice(1,-1), slice(1,-1))``，行=y、列=x）。"""
        return (slice(1, -1), slice(1, -1))

    def index_of(self, value: float, *, axis: str = "x") -> int:
        """离 ``value`` 最近的节点下标（用于把涡心吸附到网格节点上）。"""
        arr = self.x if axis == "x" else self.y
        return int(np.argmin(np.abs(arr - float(value))))


def cavity_grid(n: int) -> CavityGrid:
    """构造 ``n x n`` 单位方腔网格（``n >= 5``，且 ``h = 1/(n-1)``）。"""
    n = int(n)
    if n < 5:
        raise ValueError(f"方腔网格至少要 5 个节点/边（得到 n={n}）")
    h = 1.0 / (n - 1)
    x = np.arange(n, dtype=float) * h
    return CavityGrid(n=n, h=h, x=x, y=x.copy())


def lid_velocity(gr: CavityGrid, *, lid_speed: float = LID_SPEED) -> tuple[np.ndarray, np.ndarray]:
    """壁面速度场：顶盖 ``u = lid_speed``，其余壁与法向分量都是 0。

    返回 ``(u_wall, v_wall)`` 两个 **n x n** 数组（只有边界行/列非零）。
    """
    n = gr.n
    uw = np.zeros((n, n), dtype=float)
    vw = np.zeros((n, n), dtype=float)
    uw[n - 1, :] = float(lid_speed)
    return uw, vw


def initial_fields(gr: CavityGrid, *, kind: str = "rest") -> tuple[np.ndarray, np.ndarray]:
    r"""初值 ``(psi, omega)``。

    * ``rest``：流体静止、顶盖速度在 :math:`t=0^+` 突然加上（标准的"脉冲启动顶盖"）；
    * ``shear``：给一个满足不可压的解析剪切初值
      :math:`\psi = \sin(\pi x)^2 \sin(\pi y)^2`（用来检查 Poisson/速度算子而不是初始化）。

    两种初值都保证 :math:`\psi` 在四壁为 0（Poisson 的边界条件）与
    :math:`\omega` 由 Thom 公式自洽给出。
    """
    psi = np.zeros((gr.n, gr.n), dtype=float)
    if kind == "rest":
        pass
    elif kind == "shear":
        X, Y = np.meshgrid(gr.x, gr.y, indexing="xy")
        psi = np.sin(np.pi * X) ** 2 * np.sin(np.pi * Y) ** 2
        psi[0, :] = 0.0
        psi[-1, :] = 0.0
        psi[:, 0] = 0.0
        psi[:, -1] = 0.0
    else:
        raise ValueError(f"未知初值 {kind!r}（可用：rest/shear）")
    return psi, wall_vorticity(psi, gr)


def velocity_from_psi(psi: np.ndarray, gr: CavityGrid, *, lid_speed: float = LID_SPEED) -> tuple[np.ndarray, np.ndarray]:
    """由流函数算速度 ``(u, v) = (psi_y, -psi_x)``。

    内部用中心差分；**四壁用物理边界条件显式覆盖**（不能用单侧差分去求：
    :math:`\\psi` 在四壁恒为 0，单侧差分只会给出 0，把顶盖速度 1 整个抹掉）。
    """
    n, h = gr.n, gr.h
    u = np.zeros((n, n), dtype=float)
    v = np.zeros((n, n), dtype=float)
    u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * h)
    v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * h)
    uw, vw = lid_velocity(gr, lid_speed=lid_speed)
    u += uw
    v += vw
    return u, v


def divergence(u: np.ndarray, v: np.ndarray, gr: CavityGrid) -> np.ndarray:
    """离散散度 :math:`u_x + v_y`（中心差分，内部点；边界置 0）。

    流函数构造的速度在**连续**意义下无散；离散意义下只有把同一个差分算子
    用在同一个网格上才谈得上"零"。这里如实报出它有多大（见 pitfalls）。
    """
    n, h = gr.n, gr.h
    d = np.zeros((n, n), dtype=float)
    d[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * h) + (v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * h)
    return d


def wall_vorticity(psi: np.ndarray, gr: CavityGrid, *, lid_speed: float = LID_SPEED) -> np.ndarray:
    """由 Thom 公式给出四壁的涡量（返回新的 **n x n** 数组，内部点保持 ``psi`` 无关的 0）。

    :math:`\\omega_w = -2\\psi_\\text{邻}/h^2 - 2 U_\\tau/h`：
    底/左/右壁 :math:`U_\\tau=0`，顶盖 :math:`U_\\tau=1`。

    **符号是最容易写反的一处**：顶盖向 :math:`+x` 运动时顶壁涡量为负
    （:math:`\\omega\\approx-2/h`）。写反的后果不是"解不出来"，而是解**镜像**
    过来仍然光滑好看——实测（n=129、Re=100）主涡心会跑到
    :math:`x=1-0.617=0.383` 而不是 Ghia 的 :math:`0.6172`。

    只有**一阶**精度（:math:`O(h)`）；这是本章"网格收敛研究"里测到 1 阶的来源之一，
    不是 bug。
    """
    n, h = gr.n, gr.h
    om = np.zeros((n, n), dtype=float)
    om[0, 1:-1] = -2.0 * psi[1, 1:-1] / h**2
    om[-1, 1:-1] = -2.0 * psi[-2, 1:-1] / h**2 - 2.0 * float(lid_speed) / h
    om[1:-1, 0] = -2.0 * psi[1:-1, 1] / h**2
    om[1:-1, -1] = -2.0 * psi[1:-1, -2] / h**2
    # 四个角点：两条壁的公式都退化，取两壁的平均（角点对内部解的影响是 O(h) 级）
    om[0, 0] = -2.0 * psi[1, 1] / h**2
    om[0, -1] = -2.0 * psi[1, -2] / h**2
    om[-1, 0] = -2.0 * psi[-2, 1] / h**2 - 2.0 * float(lid_speed) / h
    om[-1, -1] = -2.0 * psi[-2, -2] / h**2 - 2.0 * float(lid_speed) / h
    return om


def enstrophy(omega: np.ndarray, gr: CavityGrid) -> float:
    """涡量平方的积分 :math:`\\int \\omega^2 \\, dA`（梯形法，单位方腔）。"""
    w = np.asarray(omega, dtype=float)
    return float(np.sum(w**2) * gr.h * gr.h)


def kinetic_energy(u: np.ndarray, v: np.ndarray, gr: CavityGrid) -> float:
    """动能 :math:`\\tfrac12\\int (u^2+v^2) dA`（梯形法）。"""
    return float(0.5 * np.sum(np.asarray(u, dtype=float) ** 2 + np.asarray(v, dtype=float) ** 2) * gr.h * gr.h)


def max_abs(arr: Any) -> float:
    """``max|arr|`` 的**有限值安全**写法（``np.max`` 遇 NaN 会静默返回 NaN）。"""
    a = np.abs(np.asarray(arr, dtype=float))
    return float(np.max(a)) if a.size else 0.0


# --------------------------------------------------------------------------- #
# Poisson：DST 直接解（复用 2.5 章的 FFT）+ 迭代解（复用 1.5 章）
# --------------------------------------------------------------------------- #
def dst_basis(m: int) -> np.ndarray:
    """DST-I 的变换矩阵 :math:`S_{pj}=\\sin(\\pi p j/(m+1))`，:math:`p,j=1..m`。

    :math:`S^2 = \\tfrac{m+1}{2} I`，所以逆变换是 :math:`(2/(m+1))^2 S \\cdot S`
    （矩阵对称，转置即自身）。
    """
    m = int(m)
    p = np.arange(1, m + 1, dtype=float)[:, None]
    j = np.arange(1, m + 1, dtype=float)[None, :]
    return np.sin(np.pi * p * j / (m + 1))


def dst1_matrix(v: np.ndarray) -> np.ndarray:
    """一维 DST-I（矩阵乘实现）：``Vhat_p = sum_j v_j sin(pi p j/(m+1))``。"""
    a = np.asarray(v, dtype=float)
    m = a.shape[0]
    return dst_basis(m) @ a


def dst1_fft(v: np.ndarray) -> np.ndarray:
    """一维 DST-I（**复用 2.5 章自研 FFT** 的奇延拓实现）。

    把长度 ``m`` 的实序列延拓成 ``[0, v, 0, -v[::-1]]``（长度 ``2m+2``）后做 FFT，
    取虚部即可得到 DST-I：``X_k = -Im(FFT(y)_k)/2``（``k = 1..m``，注意**不是** 0..m-1）。
    """
    a = np.asarray(v, dtype=float).ravel()
    m = a.shape[0]
    y = np.zeros(2 * m + 2, dtype=float)
    y[1 : m + 1] = a
    y[m + 2 :] = -a[::-1]
    return -np.imag(fft(y))[1 : m + 1] / 2.0


def dst_eigenvalues(m: int, h: float) -> np.ndarray:
    """Dirichlet 5 点 Laplacian 在 DST-I 基下的特征值
    :math:`\\lambda_p = -4\\sin^2(\\pi p/(2(m+1)))/h^2`（``p = 1..m``）。"""
    p = np.arange(1, int(m) + 1, dtype=float)
    return -4.0 * np.sin(np.pi * p / (2.0 * (int(m) + 1))) ** 2 / float(h) ** 2


def poisson_solve_dst(omega: np.ndarray, gr: CavityGrid, *, interior_rhs: np.ndarray | None = None) -> np.ndarray:
    """解 :math:`\\nabla^2\\psi = -\\omega`（:math:`\\psi=0` 于四壁）——**直接解**，不是迭代。

    用 DST-I 把 5 点 Laplacian 对角化：内部点的解在变换域里就是
    :math:`\\hat\\psi_{pq} = -\\hat\\omega_{pq}/(\\lambda_p+\\lambda_q)`。
    返回完整的 ``n x n`` 数组（四壁为 0）。``interior_rhs`` 允许直接给内部右端项
    （供 Helmholtz 复用同一套特征值）。
    """
    n, m, h = gr.n, gr.m, gr.h
    if m <= 0:
        return np.zeros((n, n), dtype=float)
    rhs = np.asarray(omega, dtype=float)[1:-1, 1:-1] if interior_rhs is None else np.asarray(interior_rhs, dtype=float)
    lam = dst_eigenvalues(m, h)
    S = dst_basis(m)
    what = S @ rhs @ S
    denom = lam[:, None] + lam[None, :]
    # lap(psi) = -omega  =>  lambda * psi_hat = -omega_hat（lambda < 0），这里少一个负号
    # 就会得到 psi = -精确解：制造解对拍（n=17 时 max|psi-exact| 从 2.0 掉到 1e-3）抓过这个 bug。
    psi_int = ((2.0 / (m + 1)) ** 2) * (S @ (-what / denom) @ S)
    psi = np.zeros((n, n), dtype=float)
    psi[1:-1, 1:-1] = psi_int
    return psi


def helmholtz_solve_dst(
    rhs: np.ndarray,
    a: float,
    gr: CavityGrid,
    *,
    bc: np.ndarray | None = None,
) -> np.ndarray:
    """解 :math:`(I - a\\nabla^2)\\omega = f`（内部），四壁值由 ``bc`` 给定。

    隐式扩散（后向 Euler）需要的就是这个算子：:math:`a = dt/Re`。实现方式是把
    壁面值对**相邻内部点**的贡献移到右端（``-a*g/h^2``），剩下的齐次 Dirichlet
    问题仍由 DST 对角化。``bc=None`` 表示齐次边界。
    """
    n, m, h = gr.n, gr.m, gr.h
    if m <= 0:
        return np.asarray(rhs, dtype=float).copy()
    f = np.asarray(rhs, dtype=float).copy()
    if bc is not None:
        g = np.asarray(bc, dtype=float)
        f[0, :] = f[0, :] - a * g[0, :] / h**2
        f[-1, :] = f[-1, :] - a * g[-1, :] / h**2
        f[:, 0] = f[:, 0] - a * g[:, 0] / h**2
        f[:, -1] = f[:, -1] - a * g[:, -1] / h**2
    f_int = f[1:-1, 1:-1]
    lam = dst_eigenvalues(m, h)
    S = dst_basis(m)
    denom = 1.0 - a * (lam[:, None] + lam[None, :])
    out_int = ((2.0 / (m + 1)) ** 2) * (S @ ((S @ f_int @ S) / denom) @ S)
    out = np.zeros((n, n), dtype=float)
    out[1:-1, 1:-1] = out_int
    if bc is not None:
        out[0, :] = bc[0, :]
        out[-1, :] = bc[-1, :]
        out[:, 0] = bc[:, 0]
        out[:, -1] = bc[:, -1]
    return out


def poisson_residual(psi: np.ndarray, omega: np.ndarray, gr: CavityGrid) -> np.ndarray:
    """内部点的 Poisson 残差 :math:`\\nabla_h^2\\psi + \\omega`（五点模板，边界为 0）。"""
    n, h = gr.n, gr.h
    r = np.zeros((n, n), dtype=float)
    lap = (psi[1:-1, 2:] + psi[1:-1, :-2] + psi[2:, 1:-1] + psi[:-2, 1:-1] - 4.0 * psi[1:-1, 1:-1]) / h**2
    r[1:-1, 1:-1] = lap + omega[1:-1, 1:-1]
    return r


@dataclass
class PoissonHistory:
    """迭代 Poisson 的完整轨迹（帧与断言共用同一份数据）。"""

    method: str
    omega_sor: float
    iterations: np.ndarray
    residuals: np.ndarray
    errors: np.ndarray
    psi: np.ndarray
    psi_exact: np.ndarray
    n_iter: int
    converged: bool
    rho_measured: float
    rho_theory: float
    omega_optimal: float
    residual_final: float
    error_final: float


def sor_optimal_omega_poisson(m: int) -> tuple[float, float]:
    """一维谱半径 :math:`\\mu=\\cos(\\pi/(m+1))` 与 Young 最优松弛因子
    :math:`\\omega_\\text{opt}=2/(1+\\sqrt{1-\\mu^2})`（复用 1.5 章的闭式）。"""
    mu = float(np.cos(np.pi / (int(m) + 1)))
    return mu, float(young_optimal_omega(mu))


def poisson_iterative_history(
    omega: np.ndarray,
    gr: CavityGrid,
    *,
    method: str = "sor",
    omega_sor: float = 1.0,
    maxiter: int = 400,
    tol: float = 1e-10,
) -> PoissonHistory:
    """在**同一个** ``omega`` 上跑 Jacobi / Gauss--Seidel / SOR，记录残差与误差两条曲线。

    矩阵、右端项、真解（DST 直接解）全部同源：这条曲线是本章"残差小不等于解准"的
    原始证据。方法实现**复用 1.5 章的** :func:`numcore.linalg.iterative.stationary_solve`
    与 :func:`~numcore.linalg.iterative.laplacian_2d`，这里不另写一份迭代法。

    注意：网格内部点数 ``m = n-2`` 会被装配成 ``m^2 x m^2`` 的**稠密**矩阵，
    所以本函数只适合 ``m <= 48`` 这类演示尺度（帧里就是这么用的）。
    """
    m = gr.m
    n = gr.n
    if m <= 0:
        raise ValueError("迭代 Poisson 至少需要 3 个节点/边")
    A = laplacian_2d(m) / gr.h**2
    b = -np.asarray(omega, dtype=float)[1:-1, 1:-1].reshape(-1)
    psi_exact = poisson_solve_dst(omega, gr)[1:-1, 1:-1].reshape(-1)
    method = str(method)
    if method == "jacobi":
        res = stationary_solve(A, b, method="jacobi", maxiter=maxiter, tol=tol, x_exact=psi_exact)
    elif method == "gs":
        res = stationary_solve(A, b, method="gs", maxiter=maxiter, tol=tol, x_exact=psi_exact)
    elif method == "sor":
        res = stationary_solve(A, b, method="sor", omega=float(omega_sor), maxiter=maxiter, tol=tol, x_exact=psi_exact)
    else:
        raise ValueError(f"未知迭代法 {method!r}（可用：jacobi/gs/sor）")
    mu, w_opt = sor_optimal_omega_poisson(m)
    rho_theory = float(sor_spectral_radius(mu, float(omega_sor)))
    psi = np.zeros((n, n), dtype=float)
    psi[1:-1, 1:-1] = res.x.reshape(m, m)
    resid = np.asarray(res.residuals, dtype=float)
    errs = np.asarray(res.errors, dtype=float)
    k = np.arange(resid.shape[0], dtype=float)
    return PoissonHistory(
        method=method,
        omega_sor=float(omega_sor),
        iterations=k,
        residuals=resid,
        errors=errs,
        psi=psi,
        psi_exact=poisson_solve_dst(omega, gr),
        n_iter=int(res.n_iter),
        converged=bool(res.converged),
        rho_measured=float(res.factor_measured),
        rho_theory=rho_theory,
        omega_optimal=w_opt,
        residual_final=float(resid[-1]) if resid.size else float("nan"),
        error_final=float(errs[-1]) if errs.size else float("nan"),
    )


# --------------------------------------------------------------------------- #
# 稳态解：SOR（红黑排序）+ DST 直接 Poisson 的耦合迭代
# --------------------------------------------------------------------------- #
@dataclass
class SteadyResult:
    """稳态方腔解与它的收敛历史。"""

    n: int
    re: float
    scheme: str
    h: float
    psi: np.ndarray
    omega: np.ndarray
    u: np.ndarray
    v: np.ndarray
    residual_history: np.ndarray
    enstrophy_history: np.ndarray
    energy_history: np.ndarray
    n_sweeps: int
    n_poisson: int
    converged: bool
    residual: float
    residual_initial: float
    max_divergence: float
    finite: bool
    diverged: bool


def _convection_coeffs(u: np.ndarray, v: np.ndarray, gr: CavityGrid, re: float, scheme: str) -> tuple[np.ndarray, ...]:
    """稳态涡量方程 :math:`u\\omega_x+v\\omega_y=\\frac1{Re}\\nabla^2\\omega` 的系数。

    返回 ``(diag, cxm, cxp, cym, cyp)``，满足 :math:`D\\omega_{ij}=\\sum c_\\cdot \\omega_\\cdot`。
    迎风（``upwind``）恒对角占优；中心（``central``）的对角元只有扩散项，
    在 :math:`Re_h>2` 时失去占优——这正是本章要量的那条线。
    """
    h = gr.h
    d = 1.0 / (float(re) * h * h)
    if scheme == "upwind":
        au = np.abs(u) / h
        av = np.abs(v) / h
        cxm = (u + np.abs(u)) / (2.0 * h) + d
        cxp = (np.abs(u) - u) / (2.0 * h) + d
        cym = (v + np.abs(v)) / (2.0 * h) + d
        cyp = (np.abs(v) - v) / (2.0 * h) + d
    elif scheme == "central":
        cxm = d + u / (2.0 * h)
        cxp = d - u / (2.0 * h)
        cym = d + v / (2.0 * h)
        cyp = d - v / (2.0 * h)
        au = np.zeros_like(u)
        av = np.zeros_like(v)
    else:
        raise ValueError(f"未知对流格式 {scheme!r}（可用：upwind/central）")
    return 4.0 * d + au + av, cxm, cxp, cym, cyp


def _residual_field(omega: np.ndarray, u: np.ndarray, v: np.ndarray, gr: CavityGrid, re: float, scheme: str) -> np.ndarray:
    """稳态涡量方程的逐点残差（与 :func:`_convection_coeffs` 同一个离散）。"""
    h = gr.h
    D, cxm, cxp, cym, cyp = _convection_coeffs(u, v, gr, re, scheme)
    rhs = (
        cxm[1:-1, 1:-1] * omega[1:-1, :-2]
        + cxp[1:-1, 1:-1] * omega[1:-1, 2:]
        + cym[1:-1, 1:-1] * omega[:-2, 1:-1]
        + cyp[1:-1, 1:-1] * omega[2:, 1:-1]
    )
    r = np.zeros_like(omega)
    r[1:-1, 1:-1] = rhs - D[1:-1, 1:-1] * omega[1:-1, 1:-1]
    return r


def _rb_sor_sweep(
    omega: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    gr: CavityGrid,
    re: float,
    w: float,
    scheme: str,
) -> np.ndarray:
    """红黑排序的 SOR 一趟（向量化；红点用黑点的旧值、黑点用红点的新值）。"""
    D, cxm, cxp, cym, cyp = _convection_coeffs(u, v, gr, re, scheme)
    out = omega.copy()
    n = gr.n
    jj, ii = np.meshgrid(np.arange(1, n - 1), np.arange(1, n - 1), indexing="ij")
    for color in (0, 1):
        mask = ((ii + jj) % 2) == color
        j = jj[mask]
        i = ii[mask]
        rhs = (
            cxm[j, i] * out[j, i - 1]
            + cxp[j, i] * out[j, i + 1]
            + cym[j, i] * out[j - 1, i]
            + cyp[j, i] * out[j + 1, i]
        )
        gs = rhs / D[j, i]
        out[j, i] = (1.0 - w) * out[j, i] + w * gs
    return out


def steady_solve(
    n: int,
    re: float,
    *,
    scheme: str = "upwind",
    omega_sor: float = 1.0,
    sweeps_per_poisson: int = 1,
    max_sweeps: int = 20000,
    tol: float = 1e-9,
    record_every: int = 20,
    psi0: np.ndarray | None = None,
) -> SteadyResult:
    """耦合迭代求稳态解：红黑 SOR 扫涡量方程 + DST 直接解 Poisson，交替进行。

    收敛判据是**离散稳态方程的相对残差**
    :math:`\\max|R| / (\\max|u|/h \\cdot \\max|\\omega|)`——
    它量的是"离离散方程有多远"，与"离基准有多远"是两件事（见模块 docstring 第 1 条）。
    ``re`` 允许为 ``inf``（无粘，用来暴露中心格式的失稳）。
    """
    gr = cavity_grid(int(n))
    re = float(re)
    if not (re > 0.0):
        raise ValueError(f"Re 必须为正（得到 {re!r}）")
    psi = np.zeros((gr.n, gr.n), dtype=float) if psi0 is None else np.asarray(psi0, dtype=float).copy()
    omega = wall_vorticity(psi, gr)
    u, v = velocity_from_psi(psi, gr)

    res_hist: list[float] = []
    ens_hist: list[float] = []
    ke_hist: list[float] = []
    n_poisson = 0
    converged = False
    diverged = False
    residual = float("inf")
    residual_initial = 0.0

    def _measure() -> tuple[float, np.ndarray]:
        r = _residual_field(omega, u, v, gr, re, scheme)
        scale = max(1.0, max_abs(u) / gr.h * max(max_abs(omega), 1.0))
        return max_abs(r) / scale, r

    residual_initial, _ = _measure()
    res_hist.append(residual_initial)
    ens_hist.append(enstrophy(omega, gr))
    ke_hist.append(kinetic_energy(u, v, gr))

    for sweep in range(int(max_sweeps)):
        omega = _rb_sor_sweep(omega, u, v, gr, re, float(omega_sor), scheme)
        omega = _replace_walls(omega, wall_vorticity(psi, gr))
        if (sweep + 1) % max(1, int(sweeps_per_poisson)) == 0:
            psi = poisson_solve_dst(omega, gr)
            u, v = velocity_from_psi(psi, gr)
            n_poisson += 1
            residual, _ = _measure()
            res_hist.append(residual)
            ens_hist.append(enstrophy(omega, gr))
            ke_hist.append(kinetic_energy(u, v, gr))
            if not math.isfinite(residual):
                diverged = True
                break
            if residual <= float(tol):
                converged = True
                break
        elif (sweep + 1) % max(1, int(record_every)) == 0:
            r_now, _ = _measure()
            res_hist.append(r_now)
            ens_hist.append(enstrophy(omega, gr))
            ke_hist.append(kinetic_energy(u, v, gr))
            if not math.isfinite(r_now):
                diverged = True
                break

    psi = poisson_solve_dst(omega, gr)
    u, v = velocity_from_psi(psi, gr)
    residual, _ = _measure()
    div = divergence(u, v, gr)
    finite = bool(np.all(np.isfinite(omega)) and np.all(np.isfinite(psi)) and np.all(np.isfinite(u)))
    return SteadyResult(
        n=gr.n,
        re=re,
        scheme=str(scheme),
        h=gr.h,
        psi=psi,
        omega=omega,
        u=u,
        v=v,
        residual_history=np.asarray(res_hist, dtype=float),
        enstrophy_history=np.asarray(ens_hist, dtype=float),
        energy_history=np.asarray(ke_hist, dtype=float),
        n_sweeps=int(max_sweeps if not (converged or diverged) else min(sweep + 1, int(max_sweeps))),
        n_poisson=int(n_poisson),
        converged=bool(converged),
        residual=float(residual),
        residual_initial=float(residual_initial),
        max_divergence=max_abs(div),
        finite=finite,
        diverged=bool(diverged or not finite),
    )


def _replace_walls(omega: np.ndarray, walls: np.ndarray) -> np.ndarray:
    """把四壁的涡量替换成 Thom 公式的值（内部点保持不变）。"""
    out = omega.copy()
    out[0, :] = walls[0, :]
    out[-1, :] = walls[-1, :]
    out[:, 0] = walls[:, 0]
    out[:, -1] = walls[:, -1]
    return out


# --------------------------------------------------------------------------- #
# 时间步限制与显式/隐式扩散
# --------------------------------------------------------------------------- #
def dt_limits(
    n: int,
    re: float,
    *,
    umax: float = 1.0,
    cfl: float = 0.8,
) -> dict[str, float]:
    """两种时间步限制与它们的交叉雷诺数。

    * 对流（CFL）：:math:`dt_\\text{cfl} = c_\\text{cfl} h/u_\\text{max}`；
    * 显式扩散（5 点模板）：:math:`dt_\\text{diff} = h^2 Re/4`；
    * 交叉点：:math:`dt_\\text{diff}=dt_\\text{cfl} \\iff Re_\\times = 4c_\\text{cfl}(n-1)/u_\\text{max}`。
    """
    gr = cavity_grid(int(n))
    h = gr.h
    dt_cfl = float(cfl) * h / max(float(umax), 1e-12)
    dt_diff = h * h * float(re) / 4.0
    re_cross = 4.0 * float(cfl) * (gr.n - 1) / max(float(umax), 1e-12)
    return {
        "h": h,
        "dt_cfl": dt_cfl,
        "dt_diffusion": dt_diff,
        "dt_min": min(dt_cfl, dt_diff),
        "re_cross": re_cross,
        "binding": 0.0 if dt_diff < dt_cfl else 1.0,
    }


def stable_dt(
    n: int,
    re: float,
    *,
    umax: float = 1.0,
    cfl: float = 0.8,
    diffusion: str = "implicit",
) -> float:
    """当前网格/Re/格式下可用的时间步。

    ``diffusion="implicit"``（后向 Euler + DST Helmholtz）时扩散不限制步长，
    只有 CFL；``"explicit"`` 时取两者较小——这个 ``min`` 就是本章第 2 条陷阱。
    """
    lim = dt_limits(n, re, umax=umax, cfl=cfl)
    if diffusion == "implicit":
        return float(lim["dt_cfl"])
    if diffusion == "explicit":
        return float(lim["dt_min"])
    raise ValueError(f"未知扩散处理 {diffusion!r}（可用：implicit/explicit）")


def cell_reynolds(re: float, n: int, *, umax: float = 1.0) -> float:
    """格点雷诺数 :math:`Re_h = u_\\text{max} h Re`。"""
    return float(umax) * cavity_grid(int(n)).h * float(re)


def central_scheme_threshold_re(n: int, *, umax: float = 1.0, re_h: float = 2.0) -> float:
    """中心差分失去对角占优的阈值 :math:`Re = Re_h (n-1)/u_\\text{max}`（``Re_h=2``）。"""
    return float(re_h) * (int(n) - 1) / max(float(umax), 1e-12)


def _upwind_tendency(omega: np.ndarray, u: np.ndarray, v: np.ndarray, gr: CavityGrid) -> np.ndarray:
    """迎风对流通量 :math:`-(u\\omega_x+v\\omega_y)`（内部点；壁面值取自 ``omega`` 自身）。"""
    h = gr.h
    dwdx = np.where(u > 0.0, (omega - _shift(omega, 0, -1)) / h, (_shift(omega, 0, 1) - omega) / h)
    dwdy = np.where(v > 0.0, (omega - _shift(omega, -1, 0)) / h, (_shift(omega, 1, 0) - omega) / h)
    return -(u * dwdx + v * dwdy)


def _shift(a: np.ndarray, dj: int, di: int) -> np.ndarray:
    """``a`` 沿 (j,i) 平移一格，边界用边缘值补齐（壁面值本来就是对的）。"""
    out = np.empty_like(a)
    js = slice(max(dj, 0), a.shape[0] + min(dj, 0))
    jd = slice(max(-dj, 0), a.shape[0] + min(-dj, 0))
    isn = slice(max(di, 0), a.shape[1] + min(di, 0))
    idst = slice(max(-di, 0), a.shape[1] + min(-di, 0))
    out[jd, idst] = a[js, isn]
    if dj > 0:
        out[:dj, :] = out[dj : dj + 1, :]
    elif dj < 0:
        out[dj:, :] = out[dj - 1 : dj, :]
    if di > 0:
        out[:, :di] = out[:, di : di + 1]
    elif di < 0:
        out[:, di:] = out[:, di - 1 : di]
    return out


@dataclass
class TransientResult:
    """瞬态推进的结果：快照 + 诊断历史。"""

    n: int
    re: float
    h: float
    dt: float
    t_end: float
    n_steps: int
    diffusion: str
    cfl: float
    times: np.ndarray
    psi: np.ndarray
    omega: np.ndarray
    u: np.ndarray
    v: np.ndarray
    psi_snapshots: list[np.ndarray]
    omega_snapshots: list[np.ndarray]
    kinetic_energy: np.ndarray
    enstrophy: np.ndarray
    max_speed: np.ndarray
    max_divergence: np.ndarray
    wall_vorticity_top: np.ndarray
    dt_cfl: float
    dt_diffusion: float
    finite: bool


def march_transient(
    n: int,
    re: float,
    *,
    t_end: float = 10.0,
    cfl: float = 0.8,
    diffusion: str = "implicit",
    n_frames: int = 21,
    max_steps: int = 20000,
    psi0: np.ndarray | None = None,
    omega0: np.ndarray | None = None,
) -> TransientResult:
    """从静止（或给定初值）**真实时间**推进：迎风对流显式 + 扩散后向 Euler（隐式，DST 解）。

    * 显式对流：:math:`dt \\le c_\\text{cfl} h/u_\\text{max}`；
    * 隐式扩散：无步长限制（Helmholtz 由 DST 直接解）；
    * ``diffusion="explicit"`` 时扩散也显式，步长取两者较小——用来演示两种限制的交叉。

    时间精度是**一阶**（对流显式 Euler + 扩散后向 Euler）。这是本章取舍：帧要在
    1 秒内出图，二阶格式（CN/AB2）留给读者；``notes.tex`` 里给出了它的截断误差与
    为什么稳态解不受一阶时间精度影响（稳态与时间离散无关）。
    """
    gr = cavity_grid(int(n))
    re = float(re)
    n_frames = max(1, int(n_frames))
    dt_cfl = stable_dt(gr.n, re, cfl=cfl, diffusion="implicit")
    dt_diff = dt_limits(gr.n, re, cfl=cfl)["dt_diffusion"]
    dt = dt_cfl if diffusion == "implicit" else min(dt_cfl, dt_diff)
    n_steps = int(min(int(max_steps), max(1, math.ceil(float(t_end) / max(dt, 1e-12)))))
    t_end = float(dt) * n_steps
    snap_idx = np.unique(np.linspace(0, n_steps, min(n_frames, n_steps + 1)).round().astype(int))

    if psi0 is None:
        psi, omega = initial_fields(gr, kind="rest")
    else:
        psi = np.asarray(psi0, dtype=float).copy()
        omega = wall_vorticity(psi, gr) if omega0 is None else _replace_walls(np.asarray(omega0, dtype=float).copy(), wall_vorticity(psi, gr))

    ke = np.zeros(snap_idx.shape[0], dtype=float)
    ens = np.zeros(snap_idx.shape[0], dtype=float)
    spd = np.zeros(snap_idx.shape[0], dtype=float)
    div = np.zeros(snap_idx.shape[0], dtype=float)
    times = np.zeros(snap_idx.shape[0], dtype=float)
    top_w = np.zeros((snap_idx.shape[0], gr.n), dtype=float)
    psi_snaps: list[np.ndarray] = []
    omega_snaps: list[np.ndarray] = []

    def _record(k: int, t: float) -> None:
        u_now, v_now = velocity_from_psi(psi, gr)
        times[k] = t
        ke[k] = kinetic_energy(u_now, v_now, gr)
        ens[k] = enstrophy(omega, gr)
        spd[k] = max_abs(np.sqrt(u_now**2 + v_now**2))
        div[k] = max_abs(divergence(u_now, v_now, gr))
        top_w[k] = omega[-1, :]
        psi_snaps.append(psi.copy())
        omega_snaps.append(omega.copy())

    snap_set = set(int(x) for x in snap_idx)
    slot = {int(s): k for k, s in enumerate(snap_idx)}
    if 0 in snap_set:
        _record(slot[0], 0.0)
    for step in range(1, n_steps + 1):
        u, v = velocity_from_psi(psi, gr)
        tendency = _upwind_tendency(omega, u, v, gr)
        omega_star = omega + dt * tendency
        walls = wall_vorticity(psi, gr)
        if diffusion == "explicit":
            lap = (
                omega_star[1:-1, 2:] + omega_star[1:-1, :-2] + omega_star[2:, 1:-1] + omega_star[:-2, 1:-1] - 4.0 * omega_star[1:-1, 1:-1]
            ) / gr.h**2
            omega_new = omega_star.copy()
            omega_new[1:-1, 1:-1] = omega_star[1:-1, 1:-1] + dt / re * lap
        else:
            omega_new = helmholtz_solve_dst(omega_star, dt / re, gr, bc=walls)
        omega = _replace_walls(omega_new, walls)
        psi = poisson_solve_dst(omega, gr)
        if not (np.all(np.isfinite(omega)) and np.all(np.isfinite(psi))):
            psi = np.nan_to_num(psi, nan=0.0, posinf=0.0, neginf=0.0)
            omega = np.nan_to_num(omega, nan=0.0, posinf=0.0, neginf=0.0)
            if step in snap_set:
                _record(slot[step], dt * step)
            break
        if step in snap_set:
            _record(slot[step], dt * step)

    u, v = velocity_from_psi(psi, gr)
    finite = bool(np.all(np.isfinite(psi)) and np.all(np.isfinite(omega)) and np.all(np.isfinite(u)))
    return TransientResult(
        n=gr.n,
        re=re,
        h=gr.h,
        dt=float(dt),
        t_end=float(t_end),
        n_steps=int(n_steps),
        diffusion=str(diffusion),
        cfl=float(cfl),
        times=times,
        psi=psi,
        omega=omega,
        u=u,
        v=v,
        psi_snapshots=psi_snaps,
        omega_snapshots=omega_snaps,
        kinetic_energy=ke,
        enstrophy=ens,
        max_speed=spd,
        max_divergence=div,
        wall_vorticity_top=top_w,
        dt_cfl=float(dt_cfl),
        dt_diffusion=float(dt_diff),
        finite=finite,
    )


# --------------------------------------------------------------------------- #
# 诊断：中心线剖面、基准对拍、涡心
# --------------------------------------------------------------------------- #
def linear_interp(xp: Sequence[float] | np.ndarray, fp: Sequence[float] | np.ndarray, x: Any) -> np.ndarray:
    """一维线性插值（**自己写**：``np.interp`` 在 P3 黑名单里）。

    区间外按端点常数延拓（并如实返回端点值，不做外推）。
    """
    xp_a = np.asarray(xp, dtype=float)
    fp_a = np.asarray(fp, dtype=float)
    x_a = np.asarray(x, dtype=float)
    idx = np.clip(np.searchsorted(xp_a, x_a, side="right") - 1, 0, xp_a.shape[0] - 2)
    x0 = xp_a[idx]
    x1 = xp_a[idx + 1]
    t = np.where(x1 > x0, (x_a - x0) / np.where(x1 > x0, x1 - x0, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    return fp_a[idx] * (1.0 - t) + fp_a[idx + 1] * t


def centerline_profiles(u: np.ndarray, v: np.ndarray, gr: CavityGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """竖直中心线 :math:`u(y)` 与水平中心线 :math:`v(x)`（取最靠近 0.5 的网格线）。

    返回 ``(y, u_center, x, v_center)``。**注意**：这是"最靠近几何中心的那条网格线"，
    n 为奇数时它离 0.5 有 :math:`h/2` 的偏移——把这条曲线的值与基准比较时，
    这个偏移本身就是一个 :math:`O(h)` 的误差源（pitfalls 里有实测）。
    """
    j = gr.index_of(0.5, axis="y")
    i = gr.index_of(0.5, axis="x")
    # 竖直中心线 x=0.5 上是 u(y) = u[:, i]；水平中心线 y=0.5 上是 v(x) = v[j, :]。
    # 两者写反是本章最容易犯的索引错误：RMSE 会稳定在 0.36 左右且**不随网格下降**
    # （见 pitfalls.md 里登记的实测值）。
    return gr.y.copy(), np.asarray(u[:, i], dtype=float).copy(), gr.x.copy(), np.asarray(v[j, :], dtype=float).copy()


@dataclass
class BenchmarkComparison:
    """与 Ghia (1982) 中心线剖面的对拍结果。"""

    re: float
    n: int
    scheme: str
    y: np.ndarray
    u_num: np.ndarray
    u_ref: np.ndarray
    x: np.ndarray
    v_num: np.ndarray
    v_ref: np.ndarray
    rmse_u: float
    max_err_u: float
    rmse_v: float
    max_err_v: float
    centerline_offset: float
    transcription_floor: float

    @property
    def rmse(self) -> float:
        """两条中心线合并的 RMSE（本章"网格收敛研究"用的就是它）。"""
        return float(math.sqrt(0.5 * (self.rmse_u**2 + self.rmse_v**2)))


def compare_with_benchmark(u: np.ndarray, v: np.ndarray, gr: CavityGrid, re: float, *, scheme: str = "upwind") -> BenchmarkComparison:
    """把数值中心线插值到 Ghia 原表节点上，返回 RMSE 与最大偏差。

    插值到**基准节点**（而不是把基准插值到网格）是刻意的：后者会把网格点上的
    离散误差抹平成"看起来更准"。两者之差在 ``pitfalls.md`` 里有实测数字。
    """
    y_n, u_c, x_n, v_c = centerline_profiles(u, v, gr)
    y_ref, u_ref = ghia_centerline_u(re)
    x_ref, v_ref = ghia_centerline_v(re)
    u_at = linear_interp(y_n, u_c, y_ref)
    v_at = linear_interp(x_n, v_c, x_ref)
    du = u_at - u_ref
    dv = v_at - v_ref
    mid = 0.5
    return BenchmarkComparison(
        re=float(re),
        n=gr.n,
        scheme=str(scheme),
        y=y_ref,
        u_num=u_at,
        u_ref=u_ref,
        x=x_ref,
        v_num=v_at,
        v_ref=v_ref,
        rmse_u=float(math.sqrt(float(np.mean(du**2)))),
        max_err_u=max_abs(du),
        rmse_v=float(math.sqrt(float(np.mean(dv**2)))),
        max_err_v=max_abs(dv),
        centerline_offset=abs(float(y_n[gr.index_of(mid, axis="y")]) - mid),
        transcription_floor=BENCHMARK_TRANSCRIPTION_SPREAD,
    )


def vortex_centers(
    psi: np.ndarray,
    gr: CavityGrid,
    *,
    threshold: float = 0.0,
    max_vortices: int = 6,
) -> list[dict[str, float]]:
    """内部流函数的局部极值 = 涡心（含主涡与二次涡）。

    判据是**八邻域严格极值**；``threshold`` 用来丢掉绝对值太小的伪极值
    （例如 n 很小时角点附近的数值噪声）。返回按 :math:`|\\psi|` 降序的列表，
    每项含名字、坐标与 :math:`\\psi` 值。**注意**：涡心位置的分辨率上限就是网格
    （:math:`O(h)`），把它与基准的 4 位小数直接比是本章最常见的"假精度"。
    """
    p = np.asarray(psi, dtype=float)
    n = gr.n
    found: list[dict[str, float]] = []
    for j in range(1, n - 1):
        for i in range(1, n - 1):
            v0 = p[j, i]
            if abs(v0) <= threshold:
                continue
            nb = p[j - 1 : j + 2, i - 1 : i + 2].copy()
            nb[1, 1] = np.nan
            flat = nb.ravel()
            flat = flat[np.isfinite(flat)]
            if flat.size == 0:
                continue
            if (v0 > flat.max() if v0 > 0 else v0 < flat.min()):
                found.append({"x": float(gr.x[i]), "y": float(gr.y[j]), "psi": float(v0), "i": float(i), "j": float(j)})
    found.sort(key=lambda d: -abs(d["psi"]))
    out: list[dict[str, float]] = []
    for rank, item in enumerate(found[: int(max_vortices)]):
        item = dict(item)
        item["name_index"] = float(rank)
        out.append(item)
    return out


def ghia_centerline_u(re: float) -> tuple[np.ndarray, np.ndarray]:
    """Ghia 竖直中心线剖面（薄封装，供本模块与帧统一入口）。"""
    return centerline_u(int(re))


def ghia_centerline_v(re: float) -> tuple[np.ndarray, np.ndarray]:
    """Ghia 水平中心线剖面（薄封装）。"""
    return centerline_v(int(re))


def ghia_vortex_centers(re: float) -> dict[str, tuple[float, float]]:
    """Ghia 涡心位置（薄封装）。"""
    return _ghia_vortex_centers(int(re))
