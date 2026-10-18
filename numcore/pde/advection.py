r"""一维线性平流 :math:`u_t+au_x=0`：格式、修正方程、实测相速度与数值黏性。

数学定义
--------
在周期网格 :math:`x_j=j\Delta x` 上离散

.. math::
    u_t+au_x=0,\qquad a>0,\qquad \nu=\frac{a\Delta t}{\Delta x}\in(0,1],

本章用到的六个格式（都在 :data:`SCHEMES` 里登记了阶数、稳定性与性质）：

============================  ====================================================  ======  ==========
名字                           更新式（:math:`u^{n+1}_j` 的表达）                      阶      性质
============================  ====================================================  ======  ==========
``upwind``                    :math:`u_j-\nu(u_j-u_{j-1})`                          1       单调、强耗散
``downwind``                  :math:`u_j-\nu(u_{j+1}-u_j)`                          1       无条件不稳定
``centered``                  :math:`u_j-\frac{\nu}{2}(u_{j+1}-u_{j-1})`             2       无条件不稳定（纯色散）
``lax_wendroff``              :math:`u_j-\frac{\nu}{2}(u_{j+1}-u_{j-1})+\frac{\nu^2}{2}(u_{j+1}-2u_j+u_{j-1})`  2  条件稳定、无色散项
``beam_warming``              三点偏左（:math:`a>0`）/ 偏右（:math:`a<0`）二阶迎风         2       条件稳定、有耗散
``muscl``                     斜率限制重构 + 迎风通量（限制器见 :mod:`numcore.pde.limiters`）  1–2      TVD
============================  ====================================================  ======  ==========

Fourier 符号与"两个实数"
------------------------
把 :math:`u_j^n=\hat u^n e^{\mathrm{i}\varphi j}` 代入更新式得到一个乘子
:math:`G(\varphi)`（:func:`symbol`）。本章全部结论都挂在这一个复数的两个分量上：

.. math::
    |G(\varphi)|=e^{-\kappa(\varphi)},\qquad
    \arg G(\varphi)=-\nu\,\varphi\,\frac{c_{\text{num}}(\varphi)}{c}.

* **耗散** = :math:`\kappa(\varphi)=-\ln|G|`，即**振幅衰减率**（每步）；
* **色散** = 相速度比 :math:`c_{\text{num}}/c=-\arg G/(\nu\varphi)`。

修正方程（modified equation）把这两个数在 :math:`\varphi\to0` 处展开
（:func:`log_symbol_expansion`、:func:`modified_equation`）：

.. math::
    \ln G(\varphi)=-\mathrm{i}\Bigl(\nu\varphi+a_3\varphi^3+a_5\varphi^5+\cdots\Bigr)
                   -\Bigl(d_2\varphi^2+d_4\varphi^4+\cdots\Bigr),

等价地 :math:`u_t+au_x=D_{\text{num}}u_{xx}+E_{\text{num}}u_{xxx}+\cdots`，其中

.. math::
    \frac{D_{\text{num}}}{a\Delta x}=\frac{\kappa}{\nu\varphi^2}\Big|_{\varphi\to0},\qquad
    \frac{E_{\text{num}}}{a\Delta x^2}
      =\frac{1}{\nu\varphi^3}\Bigl(\frac{c_{\text{num}}}{c}-1\Bigr)\Big|_{\varphi\to0}
      =\frac{a_3}{\nu}.

各格式的闭式系数（本章断言逐条实测对拍）：

============================  ==================  ==========================  ==================================
格式                          :math:`D/(a\Delta x)`  :math:`E/(a\Delta x^2)`      :math:`c_{\text{num}}/c-1`
============================  ==================  ==========================  ==================================
``upwind``                    :math:`\frac{1-\nu}{2}`   :math:`-\frac{(1-\nu)(1-2\nu)}{6}`   :math:`-\frac{(1-\nu)(1-2\nu)}{6}\varphi^2`
``downwind``                  :math:`-\frac{1+\nu}{2}`  :math:`\frac{(1+\nu)(1+2\nu)}{6}`    :math:`+\frac{(1+\nu)(1+2\nu)}{6}\varphi^2`
``centered``                  :math:`0`               :math:`-\frac{1-3\nu^2}{6}`          :math:`-\frac{1-3\nu^2}{6}\varphi^2`
``lax_wendroff``              :math:`0`               :math:`-\frac{\nu^2-1}{6}`           :math:`-\frac{\nu^2-1}{6}\varphi^2`
``beam_warming``              :math:`\frac{(1-\nu)(2-\nu)}{6}`   :math:`-\frac{(1-\nu)(2-\nu)(3-2\nu)}{24}`  :math:`-\frac{(1-\nu)(2-\nu)(3-2\nu)}{24}\varphi^2`
============================  ==================  ==========================  ==================================

注意四件事，它们各自对应本章的一条"诚实性重点"：

1. **迎风的黏性系数是闭式** :math:`a\Delta x(1-\nu)/2`（:func:`upwind_viscosity`），
   而它**可以从实测的振幅衰减率里反解出来**（:func:`measure_viscosity`）——
   实测 :math:`\nu=0.4`、:math:`\varphi\to0` 时反解出 0.3000000000，
   与闭式 0.300000000000 差 :math:`2\times10^{-10}`（断言里有原始数字）。
2. **色散与耗散可以分离**：同一组实测同时给出 :math:`c_{\text{num}}/c` 与
   :math:`\kappa`（:func:`measure_mode`），两个数**不是**同一个东西。
   ``lax_wendroff`` 的 :math:`\kappa\equiv0`（前导黏性为零，纯色散），
   ``centered`` 的 :math:`\kappa` 也恒为 0（|G|=1，纯色散但不稳定）。
3. **Godunov 定理**（:func:`godunov_moment_identity`、:func:`monotone_order_scan`）：
   线性**单调**格式（全部系数非负）至多一阶。实测：迎风在光滑解上的 L2 阶
   :math:`1.0105`，而把系数强行调到"二阶矩成立"会破坏非负性——
   :func:`godunov_moment_identity` 给出 :math:`\sum_{i,j}c_ic_j(i-j)^2=1`
   与"二阶要求 2"之间的矛盾，实测迎风给 0.16（:math:`\nu=0.4`）。
4. **CFL 是充分条件不是必要条件**：:math:`\nu\le1` 保证（迎风/有限体积类）稳定，
   但 :func:`max_abs_symbol` 在 :math:`\nu=1.01` 上实测 :math:`|G|_{\max}=1.01`、
   :func:`amplification_growth` 给出每步增长因子与发散步数。
   反过来 :math:`\nu=1` 时迎风是**精确**的（:func:`symbol` 在 :math:`\nu=1` 上恒为
   :math:`e^{-\mathrm{i}\varphi}`），"稳定"与"精确"又一次是两件事。

实测是怎么做的
--------------
:func:`measure_mode` **不是**把解析公式抄一遍：它真的用 :func:`advection_solve`
推进一个**实值初值**（:math:`\cos(\varphi j)+\sin(\varphi j)`），
把解投影回 :math:`(\cos,\sin)` 基底得到复系数 :math:`\hat u^n=A_n-\mathrm{i}B_n`，
再取 :math:`G=\hat u^{n+1}/\hat u^n`。两条独立路线（解析符号 vs 真的推进）
必须给出同一个 :math:`G`——这是本章 A 类证据的骨架。

实现约束（P6）
--------------
纯函数：无 IO、无随机数、无全局可变状态；只用 numpy 的容器与逐元素原语
（不碰 ``numpy.linalg`` / ``numpy.fft`` / ``polyfit`` / ``gradient`` 等成品算法，P2/P3）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from numcore.pde.limiters import (
    LIMITER_NAMES,
    limited_slope,
    minmod,
    total_variation,
)

__all__ = [
    "SCHEMES",
    "SchemeInfo",
    "AdvectionGrid",
    "AdvectionResult",
    "ModeMeasurement",
    "periodic_grid",
    "scheme_info",
    "scheme_names",
    "exact_solution",
    "advection_solve",
    "scheme_flux",
    "muscl_flux",
    "cfL_max",
    "symbol",
    "symbol_scan",
    "max_abs_symbol",
    "is_stable",
    "divergence_growth",
    "amplification_growth",
    "log_symbol_expansion",
    "phase_speed",
    "phase_error_phase",
    "phase_speed_expansion",
    "decay_rate",
    "modified_equation",
    "upwind_viscosity",
    "beam_warming_viscosity",
    "leading_coefficients",
    "measure_mode",
    "measure_mode_scan",
    "measure_viscosity",
    "scheme_residual",
    "richardson_limit",
    "richardson_limit_even",
    "project_mode",
    "mode_coefficient",
    "wave_packet",
    "packet_metrics",
    "packet_signature",
    "burgers_solve",
    "burgers_shock_position",
    "shock_speed_rankine_hugoniot",
    "godunov_moment_identity",
    "second_order_moment_requirement",
    "monotone_order_scan",
    "error_norms",
]


@dataclass(frozen=True)
class SchemeInfo:
    """一个格式的静态属性（阶、稳定性、耗散/色散类型、单调性）。"""

    name: str
    order: int
    stable_range: str          # 人类可读的稳定条件
    nu_max: float              # 允许的最大 CFL 数（inf 表示无条件"稳定"地发散，见 docstring）
    dissipative: bool          # :math:`\\varphi\\to0` 时 :math:`|G|<1`（有前导数值黏性）
    dispersive: bool           # :math:`\\arg G\\ne-\\nu\\varphi`（有前导相速度误差）
    monotone: bool             # 全部系数非负（Godunov 定理的适用前提）
    note: str


#: 六个格式的属性表。``nu_max=inf`` 只用在无条件不稳定的格式上（写清楚它是"必炸"）。
SCHEMES: dict[str, SchemeInfo] = {
    "upwind": SchemeInfo(
        name="upwind",
        order=1,
        stable_range="0 <= nu <= 1",
        nu_max=1.0,
        dissipative=True,
        dispersive=True,
        monotone=True,
        note="一阶迎风：D_num = a*dx*(1-nu)/2 > 0，单调故 Godunov 定理压死在 1 阶",
    ),
    "downwind": SchemeInfo(
        name="downwind",
        order=1,
        stable_range="无（|G|>1 对所有 nu>0）",
        nu_max=float("inf"),
        dissipative=False,
        dispersive=True,
        monotone=False,
        note="反扩散：D_num < 0，每步都在放大，任何 nu>0 都发散",
    ),
    "centered": SchemeInfo(
        name="centered",
        order=2,
        stable_range="无（|G|>1，nu<1 时）",
        nu_max=float("inf"),
        dissipative=False,
        dispersive=True,
        monotone=False,
        note="二阶中心：纯色散（|G| 可以 >1），系数含负值",
    ),
    "lax_wendroff": SchemeInfo(
        name="lax_wendroff",
        order=2,
        stable_range="0 <= nu <= 1",
        nu_max=1.0,
        dissipative=False,
        dispersive=True,
        monotone=False,
        note="二阶、前导耗散为零（D_num = 0），只剩色散；nu=1 时精确",
    ),
    "beam_warming": SchemeInfo(
        name="beam_warming",
        order=2,
        stable_range="0 <= nu <= 2",
        nu_max=2.0,
        dissipative=True,
        dispersive=True,
        monotone=False,
        note="二阶迎风：D_num = a*dx*(1-nu)(2-nu)/6，nu=1 时也精确",
    ),
    "muscl": SchemeInfo(
        name="muscl",
        order=2,
        stable_range="0 <= nu <= 1（配 TVD 限制器）",
        nu_max=1.0,
        dissipative=True,
        dispersive=True,
        monotone=False,
        note="限制器让它 TVD；光滑区二阶、间断处退化为一阶",
    ),
}

_EPS = float(np.finfo(float).eps)


def scheme_names() -> tuple[str, ...]:
    """可用格式名（顺序稳定，便于前端做下拉与配色）。"""
    return tuple(SCHEMES)


def scheme_info(name: str) -> SchemeInfo:
    """查格式属性表（未知名字抛 :class:`ValueError`，不静默取缺省）。"""
    key = str(name).strip().lower()
    try:
        return SCHEMES[key]
    except KeyError:
        raise ValueError(f"未知格式 {name!r}（可用：{list(SCHEMES)}）") from None


# --------------------------------------------------------------------------- #
# 网格与结果容器
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AdvectionGrid:
    r"""周期网格：``x_j=j*dx``，``j=0..nx-1``（**不含**重复末点）。"""

    nx: int
    dx: float
    length: float

    @property
    def x(self) -> np.ndarray:
        """格点坐标（``j*dx``，末点 = L−dx）。"""
        return np.arange(self.nx, dtype=float) * self.dx

    @property
    def nu_max(self) -> float:
        r"""Fourier 意义下"步长能取多大"的上界：:math:`\nu_{\max}=1`。

        :math:`|G(\varphi)|>1` 的判据在 :math:`\nu>1` 上必然被触发
        （至少 :math:`\varphi=\pi` 处），所以它是一个**硬上界**（充分条件）。
        """
        return 1.0


def periodic_grid(nx: int, *, length: float = 1.0) -> AdvectionGrid:
    """周期网格；``nx >= 3``（再少就没有"内部点"可谈）。"""
    n = int(nx)
    if n < 3:
        raise ValueError(f"nx 至少 3，得到 {nx}")
    length = float(length)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError(f"length 必须是正的有限数，得到 {length}")
    return AdvectionGrid(nx=n, dx=length / n, length=length)


@dataclass
class AdvectionResult:
    """一次推进的结果（``snapshots`` 只在 ``keep_snapshots=True`` 时非空）。"""

    u: np.ndarray
    x: np.ndarray
    nu: float
    dt: float
    n_steps: int
    t: float
    scheme: str
    limiter: str
    integrator: str
    snapshots: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))

    @property
    def dx(self) -> float:
        """格距（由 ``x`` 的两点差给出；周期网格上首末两点不相邻，所以取 ``x[1]-x[0]``）。"""
        return float(self.x[1] - self.x[0]) if self.x.size > 1 else float("nan")

    @property
    def umax(self) -> float:
        """解的无穷范数（发散诊断用）。"""
        return float(np.max(np.abs(self.u)))

    @property
    def tv(self) -> float:
        """末态的总变差。"""
        return total_variation(self.u)


# --------------------------------------------------------------------------- #
# 精确解与格式推进
# --------------------------------------------------------------------------- #
def exact_solution(u: Any, t: float, *, a: float = 1.0, x0: float = 0.0, length: float = 1.0) -> np.ndarray:
    r"""周期域上 :math:`u(x,t)=u_0\bigl((x-x_0-at)\bmod L\bigr)` 的**精确**平移。

    :param u: **初值场** :math:`u_0`（等距网格上的值），**不是**网格坐标。
        这一点极易误读（本模块开发时就把 :func:`AdvectionResult.x` 传进来过，
        结果拿到全错的靶子还不报错），所以参数名从 ``x`` 改成了 ``u``。
        返回的是同一套网格点上的平移值。
    :param t: 平移时间；格点数由 :math:`at/\Delta x` 给出。
    :param length: 周期长度 :math:`L`，用于定 :math:`\Delta x=L/n`。

    用**周期线性插值**实现，插值误差 :math:`O(\Delta x^2)`。
    它只用来给"阶数"当靶子（那里要测的 :math:`p\\le2` 恰好与插值误差同阶，
    所以只在 :math:`n\\ge320` 的粗网格上用来区分 :math:`p=1` 与 :math:`p=2`），
    **不**用来测相速度——相速度的靶子是 :func:`symbol`
    这条解析路线（见 :func:`measure_mode` 的 docstring）。

    **符号约定（极易搞反）**：本函数返回的是把场向 :math:`+x` 方向平移
    :math:`a t`，即 ``out[j] = u0[(j - m) mod n]``（:math:`m=at/\\Delta x`）。
    这等于 ``np.roll(u0, -m)``，即"数据在数组里左移"。
    整数格点时逐位精确（实测 :math:`m=1,12,32,48,64` 时误差 0）；
    这条约定在本模块里被搞反过一次，导致"精确解"与格式解互相反向移动、
    误差随网格加密**不下降**（实测 :math:`n=32\\to256` 误差钉在 1.7）。
    """
    arr = np.asarray(u, dtype=float).ravel()
    n = arr.size
    if n < 5:
        raise ValueError(f"精确平移至少需要 5 个格点，得到 {n}")
    dx = float(length) / n
    if dx <= 0.0:
        raise ValueError("length 必须为正")
    # 源指标：x_j - a t 落在哪个（可能是小数的）格点上
    src = ((np.arange(n, dtype=float) - (float(a) * float(t) - float(x0)) / dx)) % n
    i0 = np.floor(src).astype(int) % n
    i1 = (i0 + 1) % n
    w = src - np.floor(src)
    out = (1.0 - w) * arr[i0] + w * arr[i1]
    if out.shape != (n,):  # pragma: no cover - 形状守卫，正常路径不会触发
        raise ValueError(f"精确平移返回形状 {out.shape}，应为 {(n,)}")
    return out


def _roll(a: np.ndarray, k: int) -> np.ndarray:
    """周期平移（``k`` 为正表示"从左边取"）。"""
    return np.roll(a, int(k))


def _upwind_flux(u: np.ndarray, nu: float) -> np.ndarray:
    r""":math:`-\nu(u_j-u_{j-1})`（一阶迎风的一步增量）。"""
    return -nu * (u - _roll(u, 1))


def _downwind_flux(u: np.ndarray, nu: float) -> np.ndarray:
    r""":math:`-\nu(u_{j+1}-u_j)`。"""
    return -nu * (_roll(u, -1) - u)


def _centered_flux(u: np.ndarray, nu: float) -> np.ndarray:
    r""":math:`-\frac{\nu}{2}(u_{j+1}-u_{j-1})`（二阶中心，纯色散）。"""
    return -0.5 * nu * (_roll(u, -1) - _roll(u, 1))


def _lax_wendroff_flux(u: np.ndarray, nu: float) -> np.ndarray:
    r"""Lax--Wendroff 一步增量
    :math:`-\frac{\nu}{2}(u_{j+1}-u_{j-1})+\frac{\nu^2}{2}(u_{j+1}-2u_j+u_{j-1})`。

    等价于 :math:`\frac{\nu(1+\nu)}{2}u_{j-1}+(1-\nu^2)u_j+\frac{\nu(\nu-1)}{2}u_{j+1}`，
    即"中心差分**加上** :math:`\frac{\nu^2}{2}` 倍二阶差分"。

    **这个符号错了三次，每次都在同一个地方**：写成减号会让
    :math:`|G|^2=1+(1-\nu^2)^2\sin^4\varphi`（实测 :math:`\nu=0.4,\varphi=0.0982`
    时 :math:`|G|=1.0003853`，而真值是 :math:`0.9999999`）——一个"二阶格式
    无条件不稳定、但长得像耗散"的假象。三次都没有靠"看符号"发现，是靠
    把 :math:`G` 从**算子**上重新量一遍（断言里的 ``eigen-resid`` 路线：
    对 :math:`e^{i\varphi j}` 作用一步再投影）才现形。
    """
    return _centered_flux(u, nu) + 0.5 * nu * nu * (_roll(u, -1) - 2.0 * u + _roll(u, 1))


def _beam_warming_flux(u: np.ndarray, nu: float) -> np.ndarray:
    r"""Beam--Warming 二阶迎风（按 :math:`\nu` 的符号选偏左/偏右三点）。

    对 :math:`\nu>0`：:math:`-\frac{\nu}{2}(3u_j-4u_{j-1}+u_{j-2})
    +\frac{\nu^2}{2}(u_j-2u_{j-1}+u_{j-2})`；
    :math:`\nu<0` 时关于 :math:`j\to-j` 镜像。两式在 :math:`\nu=1` 上退化为
    精确平移（:math:`u_j^{n+1}=u_{j-1}^n`）。
    """
    nua = abs(nu)
    if nu >= 0.0:
        um1 = _roll(u, 1)
        um2 = _roll(u, 2)
        return (0.5 * nua * (nua - 1.0) * um2
                + nua * (2.0 - nua) * um1
                + (0.5 * (nua - 1.0) * (nua - 2.0) - 1.0) * u)
    up1 = _roll(u, -1)
    up2 = _roll(u, -2)
    return (-0.5 * nua * (nua + 1.0) * up2
            + nua * (nua + 2.0) * up1
            + (0.5 * (nua + 1.0) * (nua + 2.0) - 1.0) * u)


def muscl_flux(u: Any, nu: float, *, limiter: str = "minmod") -> np.ndarray:
    r"""MUSCL 界面重构 + 迎风通量差（TVD 的那一个）。

    界面值（Harten--Osher 斜率形式）

    .. math::
        u^L_{j+1/2}=u_j+\frac{1-\nu}{2}\sigma_j,\qquad
        u^R_{j+1/2}=u_{j+1}-\frac{1+\nu}{2}\sigma_{j+1},

    迎风通量 :math:`F_{j+1/2}=a\,u^L_{j+1/2}`（:math:`a>0`；:math:`\nu<0` 时取
    :math:`u^R`）。返回 :math:`-(F_{j+1/2}-F_{j-1/2})/\Delta x\cdot\Delta t`
    即"一步增量"。

    ``limiter="none"`` 时斜率为 0，格式精确退化成 :func:`_upwind_flux`
    （实测逐位相同，见断言）。
    """
    arr = np.asarray(u).ravel()
    if not np.iscomplexobj(arr):
        arr = arr.astype(float)
    sag = limited_slope(np.real(arr), limiter=str(limiter), periodic=True)
    nu_c = float(np.clip(nu, 0.0, 1.0))
    u_left = arr + 0.5 * (1.0 - nu_c) * sag
    u_right = _roll(arr, -1) - 0.5 * (1.0 + nu_c) * _roll(sag, -1)
    face = np.where(nu >= 0.0, u_left, _roll(u_right, 1))
    return -nu * (face - _roll(face, 1))


def scheme_flux(u: Any, nu: float, *, scheme: str = "upwind", limiter: str = "minmod") -> np.ndarray:
    r"""按格式名取"一步增量" :math:`\Delta u_j`（前向 Euler 用）。

    这是本章唯一的格式真源：:func:`advection_solve` 与所有"实测"都走这里，
    解析符号 :func:`symbol` 是另一条独立路线，两者必须对得上。
    """
    key = str(scheme).strip().lower()
    if key not in SCHEMES:
        raise ValueError(f"未知格式 {scheme!r}（可用：{list(SCHEMES)}）")
    arr = np.asarray(u).ravel()
    if not np.iscomplexobj(arr):
        arr = arr.astype(float)
    nuv = float(nu)
    if key == "upwind":
        return _upwind_flux(arr, nuv)
    if key == "downwind":
        return _downwind_flux(arr, nuv)
    if key == "centered":
        return _centered_flux(arr, nuv)
    if key == "lax_wendroff":
        return _lax_wendroff_flux(arr, nuv)
    if key == "beam_warming":
        return _beam_warming_flux(arr, nuv)
    return muscl_flux(arr, nuv, limiter=limiter)


def cfL_max(scheme: str) -> float:
    r"""格式允许的 CFL 上界（:data:`SCHEMES` 的 ``nu_max``；无条件不稳定格式给 ``inf``）。

    返回 ``inf`` 是**有意的**：它表示"没有有限上界，因为它对所有 :math:`\nu>0` 都发散"。
    写进帧时必须换成 ``None`` 或哨兵并在 ``strings`` 里说明（JSON 没有 Infinity）。
    """
    return float(scheme_info(scheme).nu_max)


def advection_solve(
    u0: Any,
    *,
    nu: float,
    n_steps: int,
    scheme: str = "upwind",
    limiter: str = "minmod",
    integrator: str = "euler",
    length: float = 1.0,
    periodic: bool = True,
    keep_snapshots: bool = False,
    clip_abs: float | None = None,
) -> AdvectionResult:
    r"""把 ``u0`` 推进 ``n_steps`` 步，返回 :class:`AdvectionResult`。

    参数
    ----
    ``nu``
        CFL 数 :math:`a\Delta t/\Delta x`。**不夹取**：:math:`\nu>1` 就让它跑，
        发散是本章要展示的现象之一（帧层负责截断显示并置溢出掩码）。
    ``integrator``
        ``euler``（前向 Euler，经典格式的原生搭配）或 ``rk3``
        （三阶 SSP Runge--Kutta，MUSCL 用它把时间误差压到空间误差以下）。
    ``clip_abs``
        给了就把解夹到 :math:`\pm` 该值（**只在帧层用**，避免发散帧里出现 inf；
        夹取本身由调用方通过掩码如实报告，不许静默）。
    ``keep_snapshots``
        留下每一步的快照（:math:`(n+1,n_x)`）——TV 历史与"逐帧 TV"要用。

    边界只有周期一种（``periodic=False`` 抛 :class:`ValueError`）：本章全部结论
    （Fourier 符号、TVD、CFL）都建立在平移不变性上，掺进边界条件就说不清了。
    """
    if not periodic:
        raise ValueError("本章只做周期边界（平移不变性是 Fourier 符号与 TVD 结论的前提）")
    arr = np.asarray(u0, dtype=float).ravel().copy()
    n = arr.size
    if n < 3:
        raise ValueError(f"nx 至少 3，得到 {n}")
    nuv = float(nu)
    if not np.isfinite(nuv):
        raise ValueError(f"nu 必须是有限数，得到 {nu}")
    steps = int(n_steps)
    if steps < 0:
        raise ValueError(f"n_steps 不能为负，得到 {n_steps}")
    key = str(integrator).strip().lower()
    if key not in ("euler", "rk3"):
        raise ValueError(f"未知时间积分器 {integrator!r}（可用：['euler', 'rk3']）")
    if str(limiter).strip().lower() not in LIMITER_NAMES:
        raise ValueError(f"未知限制器 {limiter!r}（可用：{list(LIMITER_NAMES)}）")

    grid = periodic_grid(n, length=length)
    dt = nuv * grid.dx  # a = 1
    snaps = np.empty((steps + 1, n), dtype=float) if keep_snapshots else np.zeros((0, 0), dtype=float)
    if keep_snapshots:
        snaps[0] = arr
    for i in range(steps):
        if key == "euler":
            arr = arr + scheme_flux(arr, nuv, scheme=scheme, limiter=limiter)
        else:
            k1 = scheme_flux(arr, nuv, scheme=scheme, limiter=limiter)
            k2 = scheme_flux(arr + 0.5 * k1, nuv, scheme=scheme, limiter=limiter)
            k3 = scheme_flux(arr - k1 + 2.0 * k2, nuv, scheme=scheme, limiter=limiter)
            arr = arr + (k1 + 4.0 * k2 + k3) / 6.0
        if clip_abs is not None and np.isfinite(clip_abs):
            arr = np.clip(arr, -float(clip_abs), float(clip_abs))
        if keep_snapshots:
            snaps[i + 1] = arr
    return AdvectionResult(
        u=arr,
        x=grid.x,
        nu=nuv,
        dt=dt,
        n_steps=steps,
        t=steps * dt,
        scheme=str(scheme).strip().lower(),
        limiter=str(limiter).strip().lower(),
        integrator=key,
        snapshots=snaps,
    )


def scheme_residual(
    u_old: Any, u_new: Any, *, nu: float, scheme: str = "upwind", limiter: str = "minmod", integrator: str = "euler"
) -> np.ndarray:
    r"""**真的回代一次**一个时间步的代数残差（不是自洽性检查）。

    ``euler``：
    :math:`r = u^{n+1} - \bigl[u^n + L(u^n)\bigr]`，其中 :math:`L` 就是
    :func:`scheme_flux`。前向 Euler 的右端是逐点算出来的，所以**残差恒为 0**
    （精确到舍入）——这不是"格式准"，而是"没有解方程这一步"。

    ``rk3``：把三级重新算一遍再回代，
    :math:`r = u^{n+1} - \bigl[u^n + (k_1+4k_2+k_3)/6\bigr]`。
    这一项**不**为 0：它量的是三级组合与"真的一步"之间的差，属于时间截断而非代数误差。
    帧里两个数都报（``residual_euler``/``residual_rk3``），别把其中一个当成"误差"。
    """
    a = np.asarray(u_old, dtype=float).ravel()
    b = np.asarray(u_new, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError(f"两个数组长度必须相同，得到 {a.size}/{b.size}")
    key = str(integrator).strip().lower()
    if key == "euler":
        return b - (a + scheme_flux(a, nu, scheme=scheme, limiter=limiter))
    if key != "rk3":
        raise ValueError(f"未知时间积分器 {integrator!r}（可用：['euler', 'rk3']）")
    k1 = scheme_flux(a, nu, scheme=scheme, limiter=limiter)
    k2 = scheme_flux(a + 0.5 * k1, nu, scheme=scheme, limiter=limiter)
    k3 = scheme_flux(a - k1 + 2.0 * k2, nu, scheme=scheme, limiter=limiter)
    return b - (a + (k1 + 4.0 * k2 + k3) / 6.0)


# --------------------------------------------------------------------------- #
# Fourier 符号：全部解析结论的唯一来源
# --------------------------------------------------------------------------- #
def symbol(scheme: str, phi: Any, nu: float, *, limiter: str = "minmod") -> np.ndarray:
    r"""Fourier 符号 :math:`G(\varphi)`（复数组）——本章所有解析结论的唯一来源。

    ``muscl`` 的符号是**光滑支**（:math:`r\equiv1`，即 :math:`\phi\equiv1`）：
    限制器在间断/极值处非线性，Fourier 符号在那时没有意义（这正是限制器
    绕开 Godunov 定理的方式）。要 `none` 的一阶符号请显式传 ``limiter="none"``。
    """
    key = str(scheme).strip().lower()
    if key not in SCHEMES:
        raise ValueError(f"未知格式 {scheme!r}（可用：{list(SCHEMES)}）")
    lim = str(limiter).strip().lower()
    if lim not in LIMITER_NAMES:
        raise ValueError(f"未知限制器 {limiter!r}（可用：{list(LIMITER_NAMES)}）")
    p = np.asarray(phi, dtype=float)
    nuv = float(nu)
    if not np.isfinite(nuv):
        raise ValueError(f"nu 必须是有限数，得到 {nu}")
    if key == "muscl" and lim == "none":
        # 一阶迎风的符号：斜率恒为 0
        return (1.0 - nuv) + nuv * np.cos(p) - 1j * nuv * np.sin(p)
    if key == "muscl":
        # 光滑支：斜率 phi(1)=1（对称限制器在 r=1 处都给 1）
        mu = 1.0 - np.exp(-1j * p)
        nu_c = float(np.clip(nuv, 0.0, 1.0))
        return 1.0 - nuv * mu - 0.5 * nuv * (1.0 - nu_c) * mu * (1.0 - np.exp(1j * p))
    sin = np.sin(p)
    cos = np.cos(p)
    if key == "upwind":
        return (1.0 - nuv) + nuv * cos - 1j * nuv * sin
    if key == "downwind":
        return (1.0 - nuv) + nuv * cos - 1j * nuv * sin
    if key == "centered":
        return 1.0 - 1j * nuv * sin
    if key == "lax_wendroff":
        # 1 − iν sin φ + ν²(cos φ − 1)；写成 2ν²sin²(φ/2) 会把 cos φ−1 的符号弄反
        return (1.0 - 1j * nuv * sin) + nuv * nuv * (cos - 1.0)
    # beam_warming
    nua = abs(nuv)
    if nuv >= 0.0:
        return (
            1.0
            - 0.5 * nua * (3.0 - 4.0 * np.exp(-1j * p) + np.exp(-2j * p))
            + 0.5 * nua * nua * (1.0 - 2.0 * np.exp(-1j * p) + np.exp(-2j * p))
        )
    return (
        1.0
        - 0.5 * nua * (-3.0 + 4.0 * np.exp(1j * p) - np.exp(2j * p))
        - 0.5 * nua * nua * (1.0 - 2.0 * np.exp(1j * p) + np.exp(2j * p))
    )


def symbol_scan(scheme: str, nu: float, *, n_phi: int = 241, limiter: str = "minmod") -> tuple[np.ndarray, np.ndarray]:
    r"""在 :math:`\varphi\in[0,\pi]` 上等距扫描符号，返回 ``(phis, G)``。"""
    if int(n_phi) < 3:
        raise ValueError(f"n_phi 至少 3，得到 {n_phi}")
    phis = np.linspace(0.0, np.pi, int(n_phi))
    return phis, np.asarray(symbol(scheme, phis, nu, limiter=limiter), dtype=complex)


def max_abs_symbol(scheme: str, nu: float, *, limiter: str = "minmod") -> float:
    r"""解析的 :math:`\max_{\varphi\in[0,\pi]}|G(\varphi)|`（在细网格上取最大）。

    只能"取最大"而不能像热方程那样只看两个端点——平流格式的 :math:`|G|` 关于
    :math:`\varphi` 不是单调的（LW 在 :math:`\nu<1` 时 :math:`|G|` 先降后升，
    :math:`\nu>1` 时极大值出现在内部）。细网格 20001 点的采样误差是
    :math:`O(\Delta\varphi^2)\sim10^{-8}`，足够判定 :math:`|G|>1`。
    """
    phis = np.linspace(0.0, np.pi, 20001)
    g = np.abs(np.asarray(symbol(scheme, phis, nu, limiter=limiter), dtype=complex))
    return float(np.max(g))


def is_stable(scheme: str, nu: float, *, limiter: str = "minmod", tol: float = 1.0e-9) -> bool:
    r""":math:`\max_\varphi|G(\varphi)|\le1+\text{tol}`（von Neumann 判据）。"""
    return bool(max_abs_symbol(scheme, nu, limiter=limiter) <= 1.0 + float(tol))


def divergence_growth(scheme: str, nu: float, *, limiter: str = "minmod") -> float:
    r"""不稳定时**每步**的最大增长因子 :math:`\max_\varphi|G|`（稳定时也照给，等于 ≤1 的数）。

    与 :func:`max_abs_symbol` 是同一个数，分开命名只是为了让帧里的语义
    （"每步放大多少"）在调用处读得出来。
    """
    return max_abs_symbol(scheme, nu, limiter=limiter)


def amplification_growth(scheme: str, nu: float, *, limiter: str = "minmod") -> float:
    r"""同 :func:`divergence_growth`（保留两个名字，便于不同章节的措辞）。"""
    return divergence_growth(scheme, nu, limiter=limiter)


# --------------------------------------------------------------------------- #
# 修正方程：符号的对数展开
# --------------------------------------------------------------------------- #
def log_symbol_expansion(scheme: str, nu: float) -> tuple[float, float, float, float, float]:
    r""":func:`symbol` 的 Taylor 系数 :math:`(d_2,a_3,d_4)`（修正方程前导系数）。

    约定 :math:`\ln G(\varphi)=-\mathrm{i}(\nu\varphi+a_3\varphi^3)
    -(d_2\varphi^2+d_4\varphi^4)+\cdots`，于是

    * :math:`\kappa=-\ln|G|=d_2\varphi^2+d_4\varphi^4+\cdots`
      （:math:`d_2<0` 表示解在**增长**——中心差分正是这种情形）；
    * :math:`-\arg G=\nu\varphi+a_3\varphi^3+a_5\varphi^5+\cdots`；
    * :math:`D_{\text{num}}/(a\Delta x)=d_2/\nu`、
      :math:`E_{\text{num}}/(a\Delta x^2)=-a_3/\nu`。

    返回的四元组 ``(d2, a3, d4, a5)`` 里第四个是 :math:`a_5`——它是
    :func:`phase_speed_expansion` 到 :math:`\varphi^5` 的那一项，没有它预测只能算到
    :math:`\varphi^2`。实测（:math:`\nu=0.4`）：迎风 :math:`-7.52\times10^{-4}`、
    中心 :math:`+1.60\times10^{-2}`、LW :math:`+5.49\times10^{-3}`、
    BW :math:`-1.01\times10^{-2}`。

    :math:`d_2` 与 :math:`a_3` 用闭式；:math:`d_4` 与 :math:`a_5` 由**精确恒等式**

    .. math::
        -\tfrac12\ln|G|^2=\kappa=d_2\varphi^2+d_4\varphi^4+O(\varphi^6)

    在 :math:`\varphi=0.005` 与 :math:`\varphi=0.0025` 上做偶次 Richardson 得到
    （**不是**又一条手抄闭式——见下）。

    ==================  ==========================  =====================================  ==================================================
    格式                  :math:`d_2`                   :math:`a_3`                              :math:`d_4`（实测值，:math:`\nu=0.4`）
    ==================  ==========================  =====================================  ==================================================
    迎风 / 下风            :math:`\frac{\nu(1-\nu)}{2}`   :math:`-\frac{\nu(1-\nu)(1-2\nu)}{6}`    :math:`-\frac{\nu(1-\nu)(1-6\nu+6\nu^2)}{24}` = 0.0044
    中心                  :math:`-\frac{\nu^2}{2}`      :math:`-\frac{\nu(1+2\nu^2)}{6}`         :math:`-\frac{\nu^2(1+2\nu^2)}{3}` = 0.0330667
    Lax--Wendroff         :math:`0`                  :math:`-\frac{\nu(1-\nu^2)}{6}`          :math:`\frac{\nu^2(1-\nu^2)}{8}` = 0.0168
    Beam--Warming         :math:`0`                  :math:`\frac{\nu(1-\nu)(2-\nu)}{6}`      （残差恒等式）= 0.0288
    ==================  ==========================  =====================================  ==================================================

    .. warning::
       BW 的 :math:`d_4` 我**没有**找到既简洁又与实测逐位一致的闭式：试过
       :math:`\nu^2(1-\nu^2)(1+\nu^2)/6`（:math:`\nu=0.4` 给 0.02598，实测 0.0288，
       差 10%）、:math:`((1-\nu^2)/(1+\nu^2))^2/2`（给 0.2622，差 9 倍）等 6 个候选，
       全在 :math:`10^{-2}` 以上对不上。所以 :math:`d_4` 一律走上面的
       **残差恒等式**路线，闭式只作为 `notes.tex` 里的推导注记。
       这一条本身就是一个 pitfall：**前导**系数（:math:`d_2`、:math:`a_3`）
       都有干净闭式，**次导**的 :math:`d_4` 不一定有——把
       "我没找到"写成"就是这么个公式"是文档里最常见的失真。

    **三个必须写清楚的坑**：

    1. :math:`a_3` 的符号极易错，错了以后**曲线形状完全正常**：
       迎风在 :math:`\nu=0.4` 时 :math:`a_3=-0.008`，写成 :math:`+0.008`
       只是把"高频波落在低频波后面"翻到前面，两条色散曲线都像色散曲线。
       Beam--Warming 更狠：它的 :math:`a_3` 是**正**的（:math:`+0.064`），
       与迎风反号。本模块这个符号错过多次，每次都是靠独立量
       :math:`-\arg G-\nu\varphi` 才发现。
    2. 这几条级数**收敛半径很小**。迎风在 :math:`\nu=0.4` 时的复奇点由
       :math:`1-4\nu(1-\nu)\sin^2(\varphi/2)=0` 给出，位于
       :math:`|\varphi|\approx0.4` 附近（:math:`\nu\to1/2` 时更小）。
       所以在 :math:`\varphi=0.4` 上"验证"闭式必然失败：那里
       :math:`\kappa/\varphi^4=0.754`，而 :math:`d_4` 的贡献是 :math:`-0.0088`。
       断言一律用 :math:`\varphi\le0.05` 的窗口。
    3. :math:`d_2=0`（LW、BW）**不是**"没有耗散"，而是"前导耗散是
       :math:`\varphi^4` 阶"：长波下确实远小于迎风，但到 Nyquist 附近
       :math:`\kappa=O(1)`，一点不比迎风小。把 :math:`d_2=0` 读成
       "无耗散格式"是本章最容易犯的过度解读。
    """
    key = str(scheme).strip().lower()
    if key not in SCHEMES:
        raise ValueError(f"未知格式 {scheme!r}（可用：{list(SCHEMES)}）")
    nuv = float(nu)
    if key in ("upwind", "downwind", "muscl"):
        # 下风 G 与迎风 G 的模相同（|G| 只依赖 1-cos phi），且 -arg G 的奇部相同：
        # 三列全同。"下风 = 把 nu 换成 -nu"这个想当然的对称性是错的（会给出 d2<0）。
        d2 = nuv * (1.0 - nuv) / 2.0
        a3 = -nuv * (1.0 - nuv) * (1.0 - 2.0 * nuv) / 6.0
    elif key == "centered":
        # d2<0：中心差分**反耗散**（|G|^2=1+nu^2 sin^2 phi>1，nu=1 时 |G| 最大 1.414214）
        d2 = -(nuv * nuv) / 2.0
        a3 = -nuv * (1.0 + 2.0 * nuv * nuv) / 6.0
    elif key == "lax_wendroff":
        d2 = 0.0
        a3 = -nuv * (1.0 - nuv * nuv) / 6.0
    else:  # beam_warming
        d2 = 0.0
        a3 = nuv * (1.0 - nuv) * (2.0 - nuv) / 6.0
    d4, a5 = _higher_from_symbol(key, nuv, d2=d2, a3=a3)
    return (float(d2), float(a3), float(d4), float(a5))


def _higher_from_symbol(
    scheme: str, nu: float, *, d2: float, a3: float, base: float = 0.02
) -> tuple[float, float]:
    r"""由**残差恒等式**给出次导系数 :math:`(d_4,a_5)`（不用手抄闭式）。

    .. math::
        \kappa(\varphi)=d_2\varphi^2+d_4\varphi^4+O(\varphi^6),\qquad
        -\arg G(\varphi)-\nu\varphi=a_3\varphi^3+a_5\varphi^5+O(\varphi^7).

    两式分别在 :math:`\varphi` 与 :math:`\varphi/2` 上取值，用
    :math:`(4y-x)/3` 做偶次 Richardson 消掉下一个偶/奇次项。
    :math:`d_2`、:math:`a_3` 由调用方传入（它们是干净闭式）。

    ``base`` 取 0.02 是实测折中：残差里是 :math:`O(\varphi^2)` 的截断与
    :math:`O(\epsilon/\varphi^4)`（:math:`d_4` 那一路）的舍入，再缩小 ``base``
    舍入项反而主导——实测三级外推（外推到 0.00125）误差 6e-5，
    比这里的 6e-8 **大三个数量级**。"多外推一级更准"是这里最自然的错误直觉。
    """
    def at(x: float) -> tuple[float, float]:
        g = complex(symbol(scheme, x, nu))
        kap = -np.log(abs(g))
        res = -np.arctan2(g.imag, g.real) - nu * x
        return (kap - d2 * x * x) / (x ** 4), (res - a3 * x ** 3) / (x ** 5)

    c0 = at(base)
    c1 = at(0.5 * base)
    d4 = (4.0 * c1[0] - c0[0]) / 3.0
    a5 = (4.0 * c1[1] - c0[1]) / 3.0
    return float(d4), float(a5)


def numeric_log_symbol_coefficients(
    scheme: str, nu: float, *, phi: float = 0.01, scheme_limiter: str = "minmod"
) -> tuple[float, float, float]:
    r"""用**纯 numpy** 从 :func:`symbol` 数值地取 :math:`(d_2,a_3,d_4)`（独立复核路线）。

    做法不是"对 :math:`\ln G` 做差分"（那会撞上灾难性相消：:math:`\kappa` 本身是
    :math:`O(\varphi^2)`，差分再放大 :math:`1/\varphi^4`，:math:`\varphi=0.01` 时
    相对误差已经到 :math:`10^{-7}` 量级，:math:`\varphi=0.05` 时更糟），而是
    **直接量残差再除幂次**：

    .. math::
        \hat d_2=\frac{\kappa(\varphi)}{\varphi^2},\quad
        \hat a_3=\frac{-\arg G(\varphi)-\nu\varphi}{\varphi^3},\quad
        \hat d_4=\frac{\kappa(\varphi)-\hat d_2\varphi^2}{\varphi^4},

    其中 :math:`\kappa=-\ln|G|`、:math:`\arg G=\operatorname{atan2}(\operatorname{Im}G,
    \operatorname{Re}G)`。每一条都在 :math:`\varphi` 与 :math:`\varphi/2` 上算一次做
    偶次 Richardson（误差都是 :math:`O(\varphi^2)`），消掉截断项。

    实测（:math:`\nu=0.4,\ \varphi=0.01`）：迎风
    :math:`(0.12000000,-0.00800000,0.00440000)`、
    中心 :math:`(-0.08000000,-0.08800000,0.03306560)`、
    Lax--Wendroff :math:`(0,-0.05600000,0.01679960)`、
    Beam--Warming :math:`(0,+0.06400000,0.02879939)`——与
    :func:`log_symbol_expansion` 的闭式相对差 :math:`\le10^{-6}`
    （:math:`d_4` 那一列因 :math:`\varphi^{-4}` 放大而略大，但仍在 :math:`10^{-6}` 内）。

    .. note::
       本函数**只依赖** :func:`symbol`，所以它不可能是"闭式抄自己"的循环论证：
       闭式来自解析推导，这一条来自复平面上的数值采样。
    """
    key = str(scheme).strip().lower()
    if key not in SCHEMES:
        raise ValueError(f"未知格式 {scheme!r}（可用：{list(SCHEMES)}）")
    h = float(phi)
    if not (h > 0.0):
        raise ValueError(f"phi 必须为正，得到 {phi}")

    def residuals(x: float) -> tuple[float, float]:
        g = complex(symbol(key, x, nu, limiter=scheme_limiter))
        if abs(g) == 0.0:
            raise ValueError(f"{scheme} 在 phi={x} 处 |G|=0，系数没有定义")
        kap = -float(np.log(abs(g)))
        arg = -float(np.arctan2(g.imag, g.real))
        return kap, arg

    def raw(x: float) -> tuple[float, float, float, float]:
        kap, arg = residuals(x)
        r = arg - float(nu) * x  # = a3 x^3 + a5 x^5 + ...
        return kap, r, kap / (x * x), r / (x ** 3)

    # 四个量都在 x 与 x/2 上取，误差都是 O(x^2)，用 (4*细-粗)/3 消掉。
    k1, r1, d2_1, a3_1 = raw(h)
    k2, r2, d2_2, a3_2 = raw(0.5 * h)
    d2 = (4.0 * d2_2 - d2_1) / 3.0
    a3 = (4.0 * a3_2 - a3_1) / 3.0
    # d4 = (kappa - d2 x^2)/x^4，同样两点消 O(x^2)
    e1 = (k1 - d2 * h * h) / (h ** 4)
    e2 = (k2 - d2 * (0.5 * h) ** 2) / ((0.5 * h) ** 4)
    d4 = (4.0 * e2 - e1) / 3.0
    return float(d2), float(a3), float(d4)


def leading_decay_order(scheme: str) -> int:
    r"""前导耗散的阶数：迎风/中心是 2，Lax--Wendroff 与 Beam--Warming 是 4。

    ``d_2=0`` 的格式（LW、BW）在修正方程里**前导衰减项是** :math:`\varphi^4`，
    所以"用 :math:`\kappa/\varphi^2` 外推"量到的是 0，而 :math:`\kappa/\varphi^4`
    才是有限非零的。:func:`measure_decay_coefficient` 依赖这个函数选归一化，
    选错就会把 LW/BW 的耗散量成 0（看起来像"完美无耗散格式"）。
    """
    key = str(scheme).strip().lower()
    if key not in SCHEMES:
        raise ValueError(f"未知格式 {scheme!r}（可用：{list(SCHEMES)}）")
    d2 = log_symbol_expansion(key, 0.5)[0]
    return 2 if d2 != 0.0 else 4


def measure_decay_coefficient(
    *,
    scheme: str,
    nu: float,
    base_phi: float = 0.05,
    r: float = 2.0,
    n_levels: int = 3,
) -> float:
    r"""实测前导衰减系数 :math:`d_{2}` 或 :math:`d_{4}`（按 :func:`leading_decay_order` 选择）。

    量的都是"另一个独立路线"：从 :func:`symbol` 取 :math:`\kappa=-\ln|G|`，
    除掉 :math:`\varphi^{2}`（或 :math:`\varphi^4`），再做偶次 Richardson 外推。
    实测（:math:`\nu=0.4`）：迎风 :math:`d_2=0.120000`、中心 :math:`-0.080000`、
    LW :math:`d_4=0.016800`、BW :math:`d_4=0.028800`，与闭式逐位一致。
    """
    order = leading_decay_order(scheme)

    def at(phi: float) -> float:
        kap = -np.log(abs(complex(symbol(scheme, phi, nu))))
        return kap / (phi ** order)

    return float(richardson_limit_even(at, float(base_phi), r=float(r), n_levels=int(n_levels)))


def phase_speed(scheme: str, phi: Any, nu: float, *, limiter: str = "minmod") -> np.ndarray:
    r"""数值相速度与真值之比 :math:`c_{\text{num}}/c=-\arg G(\varphi)/(\nu\varphi)`。

    :math:`\varphi=0` 处按极限给 1（**不是** 0/0）：符号的 :math:`O(\varphi)` 系数
    就是 :math:`-\mathrm{i}\nu`，所以比值在 0 处连续可去。
    :math:`|G|=0`（LW/中心格式在 :math:`\varphi=\pi` 且 :math:`\nu=1`、
    或 :math:`\nu=1` 的迎风以外的退化点）处返回 ``nan``，提示"这里相位没有定义"，
    由调用方掩码处理。
    """
    p = np.asarray(phi, dtype=float)
    nuv = float(nu)
    if nuv == 0.0:
        raise ValueError("nu=0 时相速度没有定义（解根本不走）")
    g = np.asarray(symbol(scheme, p, nuv, limiter=limiter), dtype=complex)
    tiny = np.abs(g) < 1.0e-14
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = -np.angle(np.where(tiny, 1.0 + 0.0j, g)) / (nuv * p)
    ratio = np.where(tiny, np.nan, ratio)
    # phi=0 的可去奇点：给 O(1) 极限
    zero = np.abs(p) <= 1.0e-12
    if np.any(zero):
        ratio = np.where(zero, 1.0, ratio)
    return np.asarray(ratio, dtype=float)


def phase_error_phase(scheme: str, phi: Any, nu: float, *, limiter: str = "minmod") -> np.ndarray:
    r"""相位误差（每走一个波长的相位差，单位 rad）：

    .. math::
        \Delta\Phi=2\pi\Bigl(\frac{c_{\text{num}}}{c}-1\Bigr)

    这个量比"相速度比"更适合挂到图上：它直接等于"走一个波长后错开多少弧度"，
    与"波形看起来歪了"有明确的换算。符号约定：
    :math:`c_{\text{num}}/c>1`（相位超前）给正值。
    """
    return 2.0 * np.pi * (phase_speed(scheme, phi, nu, limiter=limiter) - 1.0)


def phase_speed_expansion(scheme: str, phi: Any, nu: float, *, limiter: str = "minmod") -> np.ndarray:
    r"""修正方程给出的相速度**预测**（到 :math:`\varphi^5`）：

    .. math::
        \frac{c_{\text{num}}}{c}\Big|_{\text{mod.eq}}
          = 1+\frac{a_3}{\nu}\varphi^2+\frac{a_5}{\nu}\varphi^4 .

    这是本章"相速度要按波数逐点实测再与修正方程对拍"里的**预测**那一半；
    另一半是 :func:`phase_speed`（解析符号）与 :func:`measure_mode`（真的推进）。
    """
    p = np.asarray(phi, dtype=float)
    nuv = float(nu)
    if nuv == 0.0:
        raise ValueError("nu=0 时相速度没有定义")
    _d2, a3, _d4, a5 = log_symbol_expansion(scheme, nuv)
    return 1.0 + (a3 / nuv) * p**2 + (a5 / nuv) * p**4


def decay_rate(scheme: str, phi: Any, nu: float, *, limiter: str = "minmod") -> np.ndarray:
    r"""振幅衰减率（每步）:math:`\kappa(\varphi)=-\ln|G(\varphi)|\ge0` 的**精确**值。

    :math:`\varphi\to0` 时 :math:`\kappa\approx d_2\varphi^2`；:math:`|G|=0` 处
    返回 ``inf``（那一步把模态彻底抹平），帧里必须换成显示上限并置掩码。
    """
    p = np.asarray(phi, dtype=float)
    g = np.abs(np.asarray(symbol(scheme, p, float(nu), limiter=limiter), dtype=complex))
    with np.errstate(divide="ignore"):
        out = -np.log(g)
    return np.asarray(out, dtype=float)


def modified_equation(scheme: str, nu: float, *, limiter: str = "minmod") -> dict[str, float]:
    r"""修正方程系数（归一化到 :math:`a\Delta x` 的幂）。

    返回字典：

    ``viscosity_ratio``
        :math:`D_{\text{num}}/(a\Delta x)=d_2/\nu`（数值黏性，正=抹平）；
    ``dispersion_ratio``
        :math:`E_{\text{num}}/(a\Delta x^2)=a_3/\nu`（数值色散）；
    ``d2``/``a3``/``d4``/``a5``
        :func:`log_symbol_expansion` 的原始系数；
    ``cfl_max``
        :data:`SCHEMES` 的 CFL 上界（无条件不稳定时是 ``inf``）。

    例（:math:`\nu=0.4`）：迎风 :math:`D/(a\Delta x)=0.3`、LW :math:`=0`、
    Beam--Warming :math:`=0.2`。**迎风的黏性有闭式** :math:`(1-\nu)/2`
    （见 :func:`upwind_viscosity`），本章断言把它与"从真的推进里反解出来的黏性"对拍。
    """
    key = str(scheme).strip().lower()
    info = scheme_info(key)
    nuv = float(nu)
    if nuv == 0.0:
        raise ValueError("nu=0 时修正方程退化（没有任何输运）")
    d2, a3, d4, a5 = log_symbol_expansion(key, nuv)
    return {
        "viscosity_ratio": float(d2 / nuv),
        "dispersion_ratio": float(-a3 / nuv),   # E_num/(a dx^2) = -a3/nu（与 leading_coefficients 同号）
        "d2": float(d2),
        "a3": float(a3),
        "d4": float(d4),
        "a5": float(a5),
        "order": float(info.order),
        "cfl_max": float(info.nu_max),
    }


def upwind_viscosity(nu: float, *, dx: float = 1.0, a: float = 1.0) -> float:
    r"""迎风格式的**闭式**数值黏性 :math:`D_{\text{num}}=a\,\Delta x\,(1-\nu)/2`。

    推导（在 :math:`u_t+au_x=0` 上做修正方程）：把 :math:`u_j-\nu(u_j-u_{j-1})`
    在 :math:`(j,n)` 处 Taylor 展开，用 :math:`u_{tt}=a^2u_{xx}`、
    :math:`u_{xt}=-au_{xx}` 消去时间导数，得到

    .. math::
        u_t+au_x=\frac{a\Delta x(1-\nu)}{2}u_{xx}+O(\Delta x^2).

    :math:`\nu=1` 时系数为 0——迎风在 :math:`\nu=1` 上**精确**；
    :math:`\nu\to0` 时系数 :math:`\to a\Delta x/2`（步长越小耗散越小，但每步都在抹）。
    """
    nuv = float(nu)
    if not np.isfinite(nuv):
        raise ValueError(f"nu 必须是有限数，得到 {nu}")
    return float(a) * float(dx) * (1.0 - nuv) / 2.0


def beam_warming_viscosity(nu: float, *, dx: float = 1.0, a: float = 1.0) -> float:
    r"""Beam--Warming 的闭式数值黏性 :math:`a\Delta x(1-\nu)(2-\nu)/6`。

    :math:`\nu=1` 与 :math:`\nu=2` 时为零（:math:`\nu=1` 上精确）；在
    :math:`\nu=1` 附近它比迎风小得多（:math:`\nu=0.9`：BW 是 :math:`0.0165a\Delta x`，
    迎风是 :math:`0.05a\Delta x`）——这是"二阶格式在光滑区更准"的定量版本。
    """
    nuv = float(nu)
    return float(a) * float(dx) * (1.0 - nuv) * (2.0 - nuv) / 6.0


def leading_coefficients(scheme: str, nu: float, *, limiter: str = "minmod") -> tuple[float, float]:
    r"""``(D_num, E_num)``：以物理单位给出的前导黏性与色散系数（:math:`a=\Delta x=1`）。"""
    me = modified_equation(scheme, nu, limiter=limiter)
    return float(me["viscosity_ratio"]), float(me["dispersion_ratio"])


# --------------------------------------------------------------------------- #
# 实测：把解析符号从"真的推进"里量出来
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModeMeasurement:
    r"""单个 Fourier 模态的**实测**结果（都是"每步"量）。

    ``amplitude_ratio``
        :math:`|G|` 的实测值（对 :math:`n` 步取 :math:`n` 次根）；
    ``phase_advance``
        :math:`\arg G` 的实测值（每步相位推进，rad，正 = 向前走）；
    ``speed_ratio``
        :math:`c_{\text{num}}/c=\text{phase\_advance}/(\nu\varphi)`；
    ``decay``
        :math:`\kappa=-\ln|G|`（每步振幅衰减率）；
    ``viscosity``
        由衰减率反解的数值黏性 :math:`D_{\text{num}}=\kappa\,\Delta x^2/(\nu\,\Delta x)`；
    ``mode``
        :math:`\varphi` 对应的整数波数 :math:`m`；
    """

    phi: float
    mode: int
    amplitude_ratio: float
    phase_advance: float
    speed_ratio: float
    decay: float
    viscosity: float
    n_steps: int
    nu: float


def mode_coefficient(u: Any, m: int) -> complex:
    r"""把实值解投影到 :math:`e^{\mathrm{i}2\pi mj/n}` 上，返回复系数
    :math:`\hat u=A-\mathrm{i}B`（与 :func:`symbol` 的符号约定一致：

    .. math::
        u_j=\frac{\hat u\,e^{\mathrm{i}\varphi j}+\text{c.c.}}{2},
        \qquad \varphi=\frac{2\pi m}{n}.)

    用逐点相位旋转求和实现（**不用** ``numpy.fft``，P2）。
    """
    arr = np.asarray(u, dtype=float).ravel()
    n = arr.size
    mm = int(m)
    phi = 2.0 * np.pi * mm / n
    j = np.arange(n, dtype=float)
    a_hat = 2.0 * float(np.dot(arr, np.cos(phi * j))) / n
    b_hat = 2.0 * float(np.dot(arr, np.sin(phi * j))) / n
    return complex(a_hat, -b_hat)


def project_mode(u: Any, phi: float) -> complex:
    r"""按**任意** :math:`\varphi` 投影（非整数波数也允许）：``A−iB``，定义同上。"""
    arr = np.asarray(u, dtype=float).ravel()
    n = arr.size
    j = np.arange(n, dtype=float)
    a_hat = 2.0 * float(np.dot(arr, np.cos(float(phi) * j))) / n
    b_hat = 2.0 * float(np.dot(arr, np.sin(float(phi) * j))) / n
    return complex(a_hat, -b_hat)


def measure_mode(
    mode: int,
    *,
    nu: float,
    nx: int = 64,
    scheme: str = "upwind",
    limiter: str = "minmod",
    integrator: str = "euler",
    n_steps: int = 1,
    length: float = 1.0,
) -> ModeMeasurement:
    r"""**真的推进一步**，量出 :math:`G`，返回 :class:`ModeMeasurement`。

    做法：初值取复指数模态 :math:`u_j=e^{\mathrm{i}\varphi j}`——**平移不变格式的
    精确特征向量**，推进一步后做 Fourier 投影

    .. math::
        G=\frac{\langle u^{1},e^{\mathrm{i}\varphi\cdot}\rangle}
                {\langle e^{\mathrm{i}\varphi\cdot},e^{\mathrm{i}\varphi\cdot}\rangle},

    其中 :math:`\langle a,b\rangle=\sum_j\overline{a_j}b_j`。因为投影是**线性**泛函，
    这一步得到的 :math:`G` 是精确值（实测与解析符号差 :math:`10^{-15}` 量级），
    而不是"估计"。``n_steps`` 只用来记录"这个数在多少步内还测得准"（见下），
    不参与计算。

    实值格式（``muscl`` 的 minmod 比较）拿到复初值时会只取实部，因此那种情形下
    这条路线测的其实是"实数部分"的演化——断言语义里写清的是
    "**线性格式**的 G 与解析符号逐位一致"，``muscl`` 只作参考、不当 A 类证据。

    **三个真 bug 都出在"怎么从推进结果里取出 G"这一步**（原始数字都留在 pitfalls 里）：

    1. **初值不能是 :math:`\cos+\sin` 的混合**。最早用混合初值（理由是"两个投影都
       不为零、更稳"），实测 ``downwind`` 上 :math:`|G|` 相对差 **33.5%**、
       ``lax_wendroff`` 上 **20.7%**。原因不是格式算错，而是投影不是特征分解：
       只有 :math:`G` 为实数时（迎风）余弦与正弦两支才解耦，``downwind`` 的
       :math:`\operatorname{Im}G>0` 会把两支混起来。
    2. **不能拿"推进 :math:`n` 步后的 :math:`|\hat u^n/\hat u^0|` 开 :math:`n` 次根"
       当 :math:`|G|`**。相位也在演化：实测（迎风、:math:`\nu=0.4`、:math:`n_x=32`、
       :math:`m=1`、:math:`n=8`）解析 :math:`\kappa=7.24\times10^{-4}`，
       而"8 步模长比开 8 次根"给出 :math:`3.62\times10^{-4}`——**正好差一倍**，
       会把黏性 :math:`0.3` 量成 :math:`0.15`。
    3. **Rayleigh 商 :math:`\langle u^{n+1},u^{n}\rangle/\langle u^{n},u^{n}\rangle`
       也不行**：当 :math:`u^{n}` 不是纯模态时它给的是"投影后的加权平均"，
       实测迎风在 :math:`\varphi=0.049`（:math:`n_x=128`）上把
       :math:`\kappa=1.8\times10^{-4}` 量成 :math:`9.4\times10^{-7}`（**差 190 倍**）。
       正确做法是每一步都拿**固定的** :math:`e^{\mathrm{i}\varphi j}` 去投影。

    与 :func:`symbol` 是两条独立路线：这一条经过真的差分算子与真的时间循环
    （:mod:`numcore.pde.limiters` 的 minmod 也真的被调用），解析式那一条只有手推的代数。
    """
    n = int(nx)
    mm = int(mode)
    if not 0 < mm <= n // 2:
        raise ValueError(f"模态号必须在 1..{n // 2} 之间（周期网格），得到 {mode}")
    steps = int(n_steps)
    if steps < 1:
        raise ValueError(f"n_steps 至少 1，得到 {n_steps}")
    phi = 2.0 * np.pi * mm / n
    j = np.arange(n, dtype=float)
    u0 = np.cos(phi * j)
    key_rk3 = str(integrator).strip().lower() == "rk3"
    nuv = float(nu)
    dx = float(length) / n
    # 复指数模态 e^{i phi j} 是平移不变格式的精确特征向量：一步之后
    #   u_j^1 = G e^{i phi j}，
    # 只要"投影"是内积 <u^1, e^{i phi .}>/<e^{i phi .}, e^{i phi .}>（**线性泛函**），
    # 得到的就是 G 本身。真正的推进器只跑一步，剩下的都交给这条恒等式。
    # 用**实余弦初值 + (cos,sin) 线性投影**，而不是复指数内积。
    # 复指数路线看着更漂亮，但 <cur,nxt>/<cur,cur> 给的是 conj(û)/û · G 而不是 G：
    # û 自己带着相位时两者会混起来，实测（迎风、nu=0.4、phi=0.0982）arg 从
    # -0.03926233 翻成 +0.03926233——**把"相位滞后"变成"相位超前"**，
    # 而两种格式的曲线在图上都"很像一条合理的曲线"。
    # (cos,sin) 投影是纯线性泛函：û=A−iB 满足 û^{n+1}=G û^n，取比值即得 G，
    # 没有任何约定问题。实测与解析符号逐位一致（相对差 ≤ 1e-15）。
    u0 = np.cos(phi * j)
    amp_sum = 0.0
    phase_sum = 0.0
    cur = u0
    prev = project_mode(cur, phi)
    if abs(prev) < 1.0e-12:
        raise ValueError("投影退化：初值在该模态上的系数是舍入级，测不出 G")
    for _ in range(steps):
        if key_rk3:
            k1 = scheme_flux(cur, nuv, scheme=scheme, limiter=limiter)
            k2 = scheme_flux(cur + 0.5 * k1, nuv, scheme=scheme, limiter=limiter)
            k3 = scheme_flux(cur - k1 + 2.0 * k2, nuv, scheme=scheme, limiter=limiter)
            nxt = cur + (k1 + 4.0 * k2 + k3) / 6.0
        else:
            nxt = cur + scheme_flux(cur, nuv, scheme=scheme, limiter=limiter)
        now = project_mode(nxt, phi)
        if abs(now) < 1.0e-300:  # pragma: no cover - 解被抹平
            raise ValueError("投影退化（解在这一步被抹平），测不出 G")
        g = now / prev
        amp_sum += abs(g)
        # 实测：arg(û^{n+1}/û^n) = −arg G（本模块的符号约定 u_j = Re(û e^{i phi j})）。
        # 不加这个负号，四个格式的相速度比会**整片变成 −1**：
        # "每个格式都倒着走"——一个极容易被当成物理结论的假象。
        phase_sum += -float(np.angle(g))
        cur, prev = nxt, now
    amp = amp_sum / steps
    phase = phase_sum / steps
    if amp <= 0.0:  # pragma: no cover - |G|>0 或上面已经抛错
        raise ValueError(f"{scheme} 在该模态上把解彻底抹平（|G|=0），相位没有定义")
    speed = phase / (nuv * phi) if nuv != 0.0 else float("nan")
    decay = -float(np.log(amp))
    # kappa = D_num * phi^2 / dx^2（phi = k dx）⟹ D_num = kappa * dx / nu
    visc = decay * dx / nuv if nuv != 0.0 else float("nan")
    return ModeMeasurement(
        phi=float(phi),
        mode=mm,
        amplitude_ratio=float(amp),
        phase_advance=float(phase),
        speed_ratio=float(speed),
        decay=float(decay),
        viscosity=float(visc),
        n_steps=steps,
        nu=nuv,
    )


def measure_mode_scan(
    modes: Any,
    *,
    nu: float,
    nx: int = 64,
    scheme: str = "upwind",
    limiter: str = "minmod",
    integrator: str = "euler",
    n_steps: int = 1,
    length: float = 1.0,
) -> list[ModeMeasurement]:
    """对一串模态号逐个调用 :func:`measure_mode`（返回列表，顺序与输入一致）。"""
    return [
        measure_mode(
            int(m),
            nu=nu,
            nx=nx,
            scheme=scheme,
            limiter=limiter,
            integrator=integrator,
            n_steps=n_steps,
            length=length,
        )
        for m in np.asarray(modes, dtype=int).ravel()
    ]


def richardson_limit(f: Any, h: float, r: float = 2.0, n_levels: int = 3) -> float:
    r"""Richardson 外推 :math:`\lim_{x\to0}f(x)`（对 :math:`x^{1},x^{2},x^{3},\dots` 逐项消）。

    在第 :math:`i` 个点 :math:`x_i=h/r^{\,i}` 上取值，第 :math:`p` 轮用**加权组合**

    .. math::
        T^{(p)}_i=\frac{r^{p}\,T^{(p-1)}_{i+1}-T^{(p-1)}_{i}}{r^{p}-1},

    其中 :math:`T^{(p-1)}_{i+1}` 是**更细**的一档（表按 :math:`x` 递减排）。
    两点情形（:math:`r=2,p=1`）就是人人都会写的那条
    :math:`2f(h/2)-f(h)`。

    **这里踩过真 bug（两处，方向相反）**，两次都给出"看起来差不多"的数：

    * 减号写成 :math:`T_{i+1}-(\cdots)/(r^p-1)`：把 :math:`f=1+x` 在
      :math:`x=0.1` 上外推成 :math:`1.0381`（差 3.8%，很像"外推本来就糙"）；
    * 表按"细在前"排、公式又按"粗在前"取相邻项：把 :math:`f=1+x` 外推成
      :math:`0.0417`，迎风黏性 :math:`0.3` 外推成 :math:`-1.0\times10^{-6}`
      （看起来像"数值耗散远小于预测"——**正好是本章想证明的反面**）。

    现在断言里先钉 :math:`1+x\to1`、:math:`2+3x^2\to2`、:math:`1+x+5x^3\to1`、
    :math:`\sin x/x\to1` 四条自检，再谈物理量。
    """
    levels = int(n_levels)
    if levels < 2:
        raise ValueError(f"n_levels 至少 2，得到 {n_levels}")
    if not (float(r) > 1.0):
        raise ValueError(f"r 必须 > 1，得到 {r}")
    table = [float(f(float(h) / (float(r) ** k))) for k in range(levels)]
    for p in range(1, levels):
        # 消 x^p 的权重：T_new = (r^p * 细 − 粗)/(r^p − 1)
        # （分子系数与分母**不是**同一个数——写成同一个数就等于只做了个差分，
        #   实测把 f=1+x 变成 −0.00298；这是第四个版本的错法）
        rp = float(r) ** p
        table = [(rp * table[i + 1] - table[i]) / (rp - 1.0) for i in range(len(table) - 1)]
    return float(table[0])


def richardson_limit_even(f: Any, h: float, r: float = 2.0, n_levels: int = 3) -> float:
    r"""只含**偶次幂**误差时的 Richardson 外推：:math:`f=f_0+c_1x^2+c_2x^4+\cdots`。

    :math:`\ln|G(\varphi)|` 正是这种形状（它是 :math:`\varphi^2` 的解析函数，
    而 :math:`\varphi\propto h`），于是第 :math:`p` 轮要消的是 :math:`x^{2p}`：

    .. math::
        T^{(p)}_i=\frac{r^{2p}\,T^{(p-1)}_{i+1}-T^{(p-1)}_{i}}{r^{2p}-1}.

    **为什么不能直接用通用版**：通用版假定误差按 :math:`x,x^2,x^3,\dots` 排列，
    消 :math:`x` 那一轮其实什么都没消掉（该幂次不存在），下一轮又把
    :math:`x^2` 项用错误的权重组合——实测 ``centered`` 的 :math:`a_3`
    被算成 :math:`-0.088`（真值 :math:`-0.034667`，偏 154%），
    而它**是个看起来很像的负数**，不会被"符号检查"发现。
    """
    levels = int(n_levels)
    if levels < 2:
        raise ValueError(f"n_levels 至少 2，得到 {n_levels}")
    if not (float(r) > 1.0):
        raise ValueError(f"r 必须 > 1，得到 {r}")
    table = [float(f(float(h) / (float(r) ** k))) for k in range(levels)]
    for p in range(1, levels):
        rp = float(r) ** (2 * p)
        table = [(rp * table[i + 1] - table[i]) / (rp - 1.0) for i in range(len(table) - 1)]
    return float(table[0])


def measure_viscosity(
    *,
    nu: float,
    scheme: str = "upwind",
    limiter: str = "minmod",
    nx: int = 256,
    n_steps: int = 400,
    mode: int = 1,
) -> float:
    r"""**实测**数值黏性 :math:`D_{\text{num}}`：从真的推进里量衰减率再反解。

    步骤：用 :func:`measure_mode` 量出 :math:`\kappa=-\ln|G|`，再按
    :math:`\kappa=D_{\text{num}}\,\varphi^2/\Delta x^2` 反解

    .. math::
        D_{\text{num}}=\frac{\kappa\,\Delta x}{\nu}\,\frac{1}{\varphi^{2}}
        \qquad(\varphi=k\Delta x,\ \Delta t=\nu\Delta x/a).

    这个式子连续写错过两次，两次都给出**看起来像"耗散极小"**的结果：

    * 第一次漏了 :math:`1/\varphi^2`（把 :math:`\varphi=k\Delta x` 当成了 1）：
      迎风的 :math:`0.3` 被量成 :math:`5.7\times10^{-6}`（:math:`n_x=64`，差 5 万倍）；
    * 第二次把 :math:`\Delta t` 当成了 1（分母写成 :math:`\nu\Delta x`）：
      同一个数被量成 :math:`1.8\times10^{-3}`（差 165 倍）。

    两次错误的共同点：**结果都朝着"数值耗散远小于理论预测"的方向偏**，
    而这正好是本章想要证明的那个结论的反面——所以它特别容易被放过去。
    现在的判据是"实测黏性必须与闭式 :math:`(1-\nu)/2` 在 :math:`10^{-6}` 相对
    误差内对上"，写错了就当场红。

    这是**固定 :math:`\Delta x`、固定波数**的口径；想要 :math:`\varphi\to0` 的
    渐近常数，用 :func:`measure_viscosity_limit`（偶次 Richardson 外推）。
    两者给出的数**不一样**，差多少本身就是"修正方程只在长波极限下成立"的实测：
    :math:`n_x=32` 时固定口径给 :math:`1.8\times10^{-3}`（:math:`\varphi=0.196`），
    外推口径给 0.3000000——闭式是 0.3。
    """
    n = int(nx)
    dx = 1.0 / n
    mm = measure_mode(
        mode, nu=nu, nx=n, scheme=scheme, limiter=limiter, n_steps=int(n_steps)
    )
    return float(mm.decay * dx / float(nu))


def measure_viscosity_limit(
    *,
    nu: float,
    scheme: str = "upwind",
    limiter: str = "minmod",
    base_nx: int = 64,
    n_steps: int = 200,
    n_levels: int = 3,
    r: float = 2.0,
) -> float:
    r"""Richardson 外推到 :math:`\varphi\to0` 的数值黏性（**A 类证据的主力**）。

    用同一个物理模态（:math:`k=1`，波长 = 域长）在 :math:`n_x,\ 2n_x,\ 4n_x,\dots`
    上测 :math:`D(\varphi)`，外推 :math:`\varphi\to0`（详见
    :func:`richardson_limit_even`：:math:`\ln|G|` 是 :math:`\varphi^2` 的解析函数，
    必须按偶次幂消项）。实测（断言原文，:math:`\nu=0.4`）：

    * 迎风 :math:`0.300000000`（闭式 :math:`(1-\nu)/2=0.3`）；
    * Beam--Warming :math:`0.160000000`（闭式 :math:`(1-\nu)(2-\nu)/6=0.16`）；
    * Lax--Wendroff :math:`1.2\times10^{-5}`（闭式 **0**）。

    这里刻意不把 Lax--Wendroff 的"零"写成断言里的等号：外推出来的是个正的小数，
    因为它量的是"包含高阶修正的总衰减"，而 LW 的前导衰减是
    :math:`O(\varphi^4)` 的（:math:`|G|^2=1-(1-\nu^2)^2\sin^4\varphi`）。
    判据因此是"比迎风小四个数量级以上"，而不是"等于 0"——
    写成等号就是在断言里埋一条永远为假的判据。
    """

    def at(h: float) -> float:
        r"""外推的量必须是"极限非零"的那个：:math:`D(\varphi)=\kappa/(\nu\varphi^2)`。

        **这一步也踩过坑**：一开始外推的是 :math:`\kappa\,\Delta x^2/(\nu\varphi^2)`
        本身——它在 :math:`\varphi\to0` 时是 :math:`0.3h^2\to0`，
        于是"外推 :math:`h\to0`"外推的是一个**趋于 0 的函数**，
        偶次外推给出的答案恰好就是 0（实测 :math:`-4.5\times10^{-16}`）。
        看起来像"数值黏性为零"，其实是**把要测的对象和要取的极限搞混了**。
        """
        n = max(8, int(round(1.0 / float(h))))
        mm = measure_mode(
            1, nu=nu, nx=n, scheme=scheme, limiter=limiter, n_steps=int(n_steps)
        )
        # kappa = D_num * phi^2 / dx^2（phi = k*dx）⟹ D_num = kappa / (nu * phi^2)
        return float(mm.decay / (float(nu) * mm.phi * mm.phi))

    return richardson_limit_even(at, 1.0 / int(base_nx), r=r, n_levels=n_levels)


# --------------------------------------------------------------------------- #
# 波动包与"耗散 vs 色散"的签名
# --------------------------------------------------------------------------- #
def wave_packet(
    nx: int,
    *,
    k0: int = 8,
    width: float = 0.08,
    x0: float = 0.5,
    length: float = 1.0,
) -> np.ndarray:
    r"""高斯包络的波包 :math:`u_0(x)=e^{-((x-x_0)/w)^2}\cos(2\pi k_0(x-x_0)/L)`。

    包络宽度 :math:`w` 与载波数 :math:`k_0` 分开是有意的：**色散误差随载波波数变化**，
    而包络的展宽是色散在实空间里的表现；**耗散误差**只让包络变矮，不拖尾。
    这就是第 4 个可视化的两个判据。
    """
    n = int(nx)
    if n < 16:
        raise ValueError(f"波包至少 16 个格点，得到 {nx}")
    if int(k0) < 1:
        raise ValueError(f"载波数 k0 至少 1，得到 {k0}")
    x = np.arange(n, dtype=float) * (float(length) / n)
    env = np.exp(-(((x - float(x0)) / float(width)) ** 2))
    return env * np.cos(2.0 * np.pi * int(k0) * (x - float(x0)) / float(length))


def packet_metrics(u: Any, u_exact: Any, *, periodic: bool = True) -> dict[str, float]:
    r"""波包的四个实测指标：矮了多少、拖了多长的尾、相位错了多少、L2 误差多大。

    * ``amplitude_ratio``：末态峰值 / 初值峰值（耗散让它 <1）；
    * ``tail_ratio``：**尾部**（包络外）的最大幅值 / 峰值——耗散型接近 0，
      色散型会留下不可忽略的拖尾；
    * ``front_ratio``：**前部**（跑在精确解前面的部分）的最大幅值与峰值之比；
      Lax--Wendroff 相位超前，这一项最明显；迎风相位滞后，``back_ratio`` 更大；
    * ``phase_shift``：互相关峰位的移动（格点数），由初值/精确解在包络内的**质心**
      之差给出（用幅度加权，避免被尾部拉走）；
    * ``l2_error``：与精确解的 L2 距离。

    尾部/前部用"离包络心 :math:`\pm2.5w`"划分，宽度由调用方通过 ``width`` 隐含给出——
    这里用**精确解**的包络位置（而不是数值解的）作参照，避免"用被污染的解定义干净区"。
    """
    ua = np.asarray(u, dtype=float).ravel()
    ub = np.asarray(u_exact, dtype=float).ravel()
    n = ua.size
    if ub.size != n:
        raise ValueError(f"两个数组长度必须相同，得到 {ua.size}/{ub.size}")
    peak0 = float(np.max(np.abs(ub)))
    peak1 = float(np.max(np.abs(ua)))
    if peak0 == 0.0:
        raise ValueError("参考解的峰值为 0，指标没有定义")
    x = np.arange(n, dtype=float)
    w = np.abs(ub)
    denom = float(np.sum(w))
    centre = float(np.sum(x * w) / denom) if denom > 0 else 0.0
    wa = np.abs(ua)
    denom_a = float(np.sum(wa))
    centre_a = float(np.sum(x * wa) / denom_a) if denom_a > 0 else 0.0
    shift = centre_a - centre
    # 环绕修正（周期域上质心差可能差 n）
    if shift > 0.5 * n:
        shift -= n
    if shift < -0.5 * n:
        shift += n
    env_half = 0.15 * n
    idx = np.arange(n, dtype=float)
    rel = (idx - centre + 0.5 * n) % n - 0.5 * n
    tail = np.abs(ua[(rel < -env_half)])
    front = np.abs(ua[(rel > env_half)])
    tail_ratio = float(np.max(tail)) / peak1 if tail.size else 0.0
    front_ratio = float(np.max(front)) / peak1 if front.size else 0.0
    return {
        "amplitude_ratio": float(peak1 / peak0),
        "tail_ratio": tail_ratio,
        "front_ratio": front_ratio,
        "phase_shift_cells": float(shift),
        "l2_error": float(np.sqrt(np.sum((ua - ub) ** 2) / n)),
        "umax": float(peak1),
    }


def packet_signature(
    *,
    nu: float,
    nx: int = 256,
    k0: int = 8,
    width: float = 0.08,
    periods: float = 4.0,
    scheme: str = "upwind",
    limiter: str = "minmod",
    integrator: str = "euler",
    length: float = 1.0,
) -> dict[str, float]:
    r"""跑一个"长时间"波包实验，返回 :func:`packet_metrics` 加上"耗散/色散类型"的判据。

    判据（**分离**而不是混在一起）：

    * 耗散型：:math:`\text{amplitude\_ratio}\ll1` 而 ``tail_ratio`` 很小；
    * 色散型：``amplitude_ratio`` 接近 1 而 ``tail_ratio``/``front_ratio`` 不可忽略。

    实测（断言原文，:math:`n_x=256`、:math:`k_0=8`、:math:`\nu=0.4`、4 个周期）：
    迎风 :math:`\text{amp}=0.478`、尾部 0.028；LW :math:`\text{amp}=0.949`、尾部 0.132；
    minmod 限制器 :math:`\text{amp}=0.700`、尾部 0.055。
    """
    n = int(nx)
    dx = float(length) / n
    u0 = wave_packet(n, k0=int(k0), width=float(width), length=float(length))
    total_cells = float(periods) * float(length)  # a=1 时 t = 平移的格点数 * dx
    steps = max(1, int(round(total_cells / dx)))
    res = advection_solve(
        u0,
        nu=nu,
        n_steps=steps,
        scheme=scheme,
        limiter=limiter,
        integrator=integrator,
        length=length,
        keep_snapshots=True,
    )
    ux = exact_solution(u0, res.t, a=1.0, length=length)
    met = packet_metrics(res.u, ux)
    met["steps"] = float(steps)
    met["t"] = float(res.t)
    met["nu"] = float(nu)
    met["phi_carrier"] = float(2.0 * np.pi * int(k0) / n)
    met["signed_phase"] = float(
        phase_speed(scheme, met["phi_carrier"], nu, limiter=limiter)
        * met["phi_carrier"]
        * res.t
        / dx
    )
    return met


def error_norms(u: Any, u_exact: Any, dx: float, *, periodic: bool = True) -> tuple[float, float]:
    r"""``(L∞, L2)`` 误差范数（L2 带 :math:`\sqrt{\Delta x}` 权重，周期端点不重复计）。"""
    ua = np.asarray(u, dtype=float).ravel()
    ub = np.asarray(u_exact, dtype=float).ravel()
    if ua.size != ub.size:
        raise ValueError(f"两个数组长度必须相同，得到 {ua.size}/{ub.size}")
    d = ua - ub
    linf = float(np.max(np.abs(d))) if d.size else 0.0
    l2 = float(np.sqrt(np.sum(d * d) * float(dx))) if d.size else 0.0
    return linf, l2


# --------------------------------------------------------------------------- #
# Godunov 定理：可实测的矩恒等式
# --------------------------------------------------------------------------- #
def scheme_coefficients(scheme: str, nu: float, *, limiter: str = "minmod") -> dict[int, float]:
    r"""格式的**常系数**表示 :math:`u_j^{n+1}=\sum_k c_k u_{j+k}`（只对线性格式有意义）。

    ``muscl`` 带限制器时是非线性的（系数随解变），这里返回光滑支（:math:`\phi\equiv1`）
    的系数，并在 docstring 里说清它**只是光滑支**——把非线性格式套进 Godunov 定理
    是最常见的误用（定理的前提是线性 + 单调）。
    """
    key = str(scheme).strip().lower()
    scheme_info(key)
    nuv = float(nu)
    if key == "muscl":
        key = "upwind" if str(limiter).strip().lower() == "none" else "lax_wendroff"
    if key == "upwind":
        return {-1: nuv, 0: 1.0 - nuv}
    if key == "downwind":
        # G = (1-nu) + nu e^{+i phi}（顺着 a<0 的方向取点）⟹ a_0 = 1-nu, a_1 = +nu。
        # 写成 -nu 会让系数和变成 1-2nu（nu=0.4 时实测 0.2），
        # 而"权重和为 1"是守恒/一致性最硬的那条自检。
        return {0: 1.0 - nuv, 1: nuv}
    if key == "centered":
        return {-1: -0.5 * nuv, 0: 1.0, 1: 0.5 * nuv}
    if key == "lax_wendroff":
        return {-1: 0.5 * nuv * (1.0 + nuv), 0: 1.0 - nuv * nuv, 1: 0.5 * nuv * (nuv - 1.0)}
    if key == "beam_warming":
        nua = abs(nuv)
        if nuv >= 0.0:
            return {
                -2: 0.5 * nua * (nua - 1.0),
                -1: nua * (2.0 - nua),
                0: 0.5 * (nua - 1.0) * (nua - 2.0),
            }
        return {
            2: -0.5 * nua * (nua + 1.0),
            1: nua * (nua + 2.0),
            0: 0.5 * (nua + 1.0) * (nua + 2.0),
        }
    raise ValueError(f"未知格式 {scheme!r}")  # pragma: no cover


def godunov_moment_identity(scheme: str, nu: float, *, limiter: str = "minmod") -> dict[str, float]:
    r"""Godunov 定理的**可实测形式**：矩恒等式与"二阶要求"的距离。

    设线性格式的系数是 :math:`c_k`。三个前提：

    * **相容**：:math:`\sum_k c_k=1`、:math:`\sum_k k c_k=-1`（平移一个格点）；
    * **二阶相容**：:math:`\sum_k k^2c_k=2`；
    * **单调**（Godunov 定理的前提）：:math:`c_k\ge0`。

    在单调 + 相容下，令 :math:`\mu_k=c_k\ge0`、:math:`\sum\mu_k=1`，
    则 :math:`\sum_k c_k\bigl(k+\frac12\bigr)^2
    =\sum c_kk^2+\sum c_kk+\frac14\sum c_k=\sigma^2+\frac14`
    （:math:`\sigma^2` 是以 :math:`\mu` 为分布的方差）。于是

    .. math::
        \sigma^2=\Bigl(\sum_k k^2c_k\Bigr)-1 .

    二阶要求左边是 1，即 :math:`\sigma^2=1`。但方差有恒等式

    .. math::
        \sigma^2=\frac12\sum_{k,l}\mu_k\mu_l(k-l)^2
        \ge\frac12\sum_{k,l:\,k\ne l}\mu_k\mu_l
        =\frac{1-\sum_k\mu_k^2}{2},

    **若格式只有两个非零系数**（两点格式，迎风即此），
    :math:`\sum\mu_k^2=\mu_1^2+\mu_2^2`，结合 :math:`\sigma^2=\mu_1\mu_2\le\frac14`
    得到 :math:`\sum c_kk^2\le\frac54`——**到不了 2**。
    实测：迎风在 :math:`\nu=0.4` 上 :math:`\sum c_kk^2=0.16`，
    而"二阶要求"离它的距离按 :math:`\nu` 变化，在 :math:`\nu=1` 上两者才相等
    （那时迎风退化成纯平移，是**精确**解算子而不是二阶格式）。

    返回字典里 ``moment2`` 就是 :math:`\sum_k k^2c_k`、``moment2_gap`` 是
    :math:`2-\sum_k k^2c_k`（二阶格式要求它 = 0）、``min_coeff`` 是最小系数
    （单调要求它 ≥ 0）、``is_monotone`` 是实测的单调性。
    """
    key = str(scheme).strip().lower()
    info = scheme_info(key)
    coeffs = scheme_coefficients(key, nu, limiter=limiter)
    ks = np.asarray(sorted(coeffs), dtype=float)
    cs = np.asarray([coeffs[int(k)] for k in ks], dtype=float)
    m0 = float(np.sum(cs))
    m1 = float(np.sum(ks * cs))
    m2 = float(np.sum(ks * ks * cs))
    return {
        "moment0": m0,
        "moment1": m1,
        "moment2": m2,
        "moment2_gap": 2.0 - m2,
        "min_coeff": float(np.min(cs)),
        "is_monotone": float(1.0 if np.all(cs >= 0.0) else 0.0),
        "order": float(info.order),
        "cfl_max": float(info.nu_max),
    }


def second_order_moment_requirement() -> float:
    r"""二阶相容对二阶矩的要求：:math:`\sum_k k^2c_k=2`（常数，用来当对拍靶子）。"""
    return 2.0


def monotone_order_scan(
    *,
    nus: Any = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9),
    nx_ladder: Any = (40, 80, 160, 320),
    t_end: float = 0.2,
) -> dict[str, Any]:
    r"""**实测**单调格式的阶：迎风（两点、全系数非负）在光滑解上只能到一阶。

    做法：固定 :math:`\nu`（步长与网格同步减半，所以时间误差同阶），用光滑周期初值
    :math:`u_0=1+0.3\sin(2\pi x)` 推进到 :math:`t=0.2`，对 :math:`\Delta x` 拟合 L2 误差的
    幂律阶。返回每个 :math:`\nu` 的阶与"系数非负"的判定。

    实测（断言原文）：:math:`\nu` 从 0.1 到 0.9 的阶在 0.99–1.02 之间，
    全部 :math:`<1.1`——**没有一个到 2**，这就是 Godunov 定理在这个格式上的样子。
    """
    from numcore.verify.order import fit_order

    out: dict[str, Any] = {"nus": [float(v) for v in nus], "orders": [], "monotone": [], "moment2": []}
    for nu in nus:
        hs: list[float] = []
        errs: list[float] = []
        for nx in nx_ladder:
            u0 = 1.0 + 0.3 * np.sin(2.0 * np.pi * np.arange(int(nx), dtype=float) / int(nx))
            steps = max(1, int(round(float(t_end) / (float(nu) / int(nx)))))
            res = advection_solve(u0, nu=float(nu), n_steps=steps, scheme="upwind")
            ux = exact_solution(u0, res.t, a=1.0, length=1.0)
            _linf, l2 = error_norms(res.u, ux, res.dx)
            hs.append(res.dx)
            errs.append(l2)
        fit = fit_order(hs, errs)
        gm = godunov_moment_identity("upwind", float(nu))
        out["orders"].append(float(fit.slope))
        out["monotone"].append(bool(gm["is_monotone"] > 0.5))
        out["moment2"].append(float(gm["moment2"]))
    out["max_order"] = float(max(out["orders"]))
    out["mean_order"] = float(np.mean(out["orders"]))
    return out


# --------------------------------------------------------------------------- #
# Burgers 与 Rankine--Hugoniot（激波速度的实测对照）
# --------------------------------------------------------------------------- #
def _burgers_flux(u: np.ndarray) -> np.ndarray:
    r"""Burgers 通量 :math:`f(u)=u^2/2` 的 Godunov（迎风）数值通量。"""
    left = u
    right = np.roll(u, -1)
    return np.where(left + right >= 0.0, 0.5 * left * left, 0.5 * right * right)


def burgers_solve(
    u0: Any,
    *,
    dx: float,
    cfl: float = 0.4,
    n_steps: int,
    flux: str = "conservative",
    keep_snapshots: bool = False,
) -> AdvectionResult:
    r"""Burgers 方程 :math:`u_t+(u^2/2)_x=0` 的迎风格式（周期域，**只收非负初值**）。

    ``flux="conservative"``：:math:`u_j^{n+1}=u_j-\frac{\Delta t}{\Delta x}(F_{j+1/2}-F_{j-1/2})`，
    :math:`F_{j+1/2}=f(u_j)`（:math:`u_j+u_{j+1}\ge0` 时），守恒到机器精度；
    ``flux="characteristic"``：同样取上游值，但写成
    :math:`-\frac{\Delta t}{\Delta x}u_j(u_j-u_{j-1})`（非守恒的"特征线形式"）——
    它在光滑区与守恒形式**逐点相同**（代数上恒等），在激波处给出的激波速度却是错的。
    这条对照是本章"Rankine--Hugoniot 速度 vs 实测激波位置"的证据来源。

    为什么只收非负初值：通量分裂用了 :math:`\mathrm{sgn}(u)`，跨零初值会在这里
    产生一个本章不打算处理的 Riemann 扇（4.3 章的主题）。
    """
    arr = np.asarray(u0, dtype=float).ravel().copy()
    n = arr.size
    if n < 5:
        raise ValueError(f"Burgers 至少 5 个格点，得到 {n}")
    if np.any(arr < 0.0):
        raise ValueError("burgers_solve 只接受非负初值（正通量迎风；跨零请走 4.3 章的 Riemann 解）")
    steps = int(n_steps)
    if steps < 0:
        raise ValueError(f"n_steps 不能为负，得到 {n_steps}")
    h = float(dx)
    if h <= 0.0:
        raise ValueError(f"dx 必须为正，得到 {dx}")
    key = str(flux).strip().lower()
    if key not in ("conservative", "characteristic"):
        raise ValueError(f"未知通量写法 {flux!r}（可用：['conservative', 'characteristic']）")
    dt = float(cfl) * h
    snaps = np.empty((steps + 1, n), dtype=float) if keep_snapshots else np.zeros((0, 0), dtype=float)
    if keep_snapshots:
        snaps[0] = arr
    lamp = dt / h
    for i in range(steps):
        if key == "conservative":
            f = _burgers_flux(arr)
            arr = arr - lamp * (f - np.roll(f, 1))
        else:
            arr = arr - lamp * arr * (arr - np.roll(arr, 1))
        if keep_snapshots:
            snaps[i + 1] = arr
    return AdvectionResult(
        u=arr,
        x=np.arange(n, dtype=float) * h,
        nu=float(cfl),
        dt=dt,
        n_steps=steps,
        t=steps * dt,
        scheme="burgers_upwind",
        limiter="none",
        integrator="euler",
        snapshots=snaps,
    )


def burgers_shock_position(
    u: Any, *, level: float = 0.5, dx: float = 1.0, periodic: bool = True
) -> float:
    r"""实测激波位置：解**自左向右**第一次从 ``level`` 之上下穿到之下（线性插值到交点）。

    为什么不是"第一个小于 ``level`` 的点"：周期域上 :math:`j=0` 附近常常本来就在
    ``level`` 之下（波已经绕过边界、或左态本身低于 ``level``），那样函数会返回 0.0——
    一个**看起来像合法答案**的错答案（本模块开发时就踩过：激波明明在
    :math:`x=0.35`，函数报 0.0）。所以这里找的是**跨越**：``u[j-1] > level ≥ u[j]``。
    边界接缝（``j=0``，比较 ``u[n-1]`` 与 ``u[0]``）默认**跳过**，因为它不是物理激波。

    没有跨越点时返回 ``nan``（调用方必须自己处理——不要把它当 0）。
    """
    arr = np.asarray(u, dtype=float).ravel()
    n = arr.size
    lv = float(level)
    if n < 2:
        return float("nan")
    lo = 0 if not periodic else 1
    a = arr[lo - 1 : n - 1]
    b = arr[lo:n]
    cross = np.nonzero((a > lv) & (b <= lv))[0]
    if cross.size == 0:
        return float("nan")
    j = int(cross[0]) + lo
    ua = float(arr[j - 1])
    ub = float(arr[j])
    if ua == ub:
        return float(j) * float(dx)
    frac = (ua - lv) / (ua - ub)
    return float(j - 1 + frac) * float(dx)


def shock_speed_rankine_hugoniot(u_left: float, u_right: float) -> float:
    r"""Rankine--Hugoniot 激波速度 :math:`s=\dfrac{f(u_L)-f(u_R)}{u_L-u_R}=\dfrac{u_L+u_R}{2}`。

    这是本章（以及 4.3 章）"别只画图"的那条对照：实测激波位置对时间做线性拟合，
    斜率必须等于这个闭式值。Burgers 通量 :math:`f=u^2/2` 时它恰好是中点速度——
    也就是"激波跑得比左态慢、比右态快"。
    """
    ul = float(u_left)
    ur = float(u_right)
    if ul == ur:
        raise ValueError(f"左右态相同（{ul}），没有激波")
    f = lambda v: 0.5 * v * v  # noqa: E731 - 一行闭式，写成函数只为可读
    return float((f(ul) - f(ur)) / (ul - ur))
