r"""6.1 章内核之三：浅水方程（Saint--Venant）——守恒型有限体积、Rusanov 通量、干底与静水平衡。

数学定义
--------
平坦底、无摩阻的一维浅水方程（守恒量 :math:`U=(h,\,hu)`，:math:`h` 是水深、
:math:`u` 是深度平均速度）：

.. math::
    \frac{\partial}{\partial t}\begin{pmatrix} h\\ hu\end{pmatrix}
    +\frac{\partial}{\partial x}
    \begin{pmatrix} hu\\ hu^2+\frac{1}{2}gh^2\end{pmatrix}=0 ,
    \qquad
    F(U)=\begin{pmatrix} hu\\ \dfrac{(hu)^2}{h}+\dfrac{1}{2}gh^2\end{pmatrix},

特征速度 :math:`\lambda_\pm=u\pm\sqrt{gh}`（:data:`G_DEFAULT` :math:`=9.81`）。
带底高 :math:`b(x)` 时动量方程右端多一项 :math:`-g h\,b_x`
（:func:`swe_rhs_1d_bathy`）；二维是两个方向的同一套东西
（:func:`swe_step_2d`）。

离散：**守恒型有限体积 + Rusanov（局部 Lax--Friedrichs）通量**

.. math::
    F_{j+1/2}=\frac{1}{2}\bigl(F(U_j)+F(U_{j+1})\bigr)
              -\frac{1}{2}a_{j+1/2}\bigl(U_{j+1}-U_j\bigr),\qquad
    a_{j+1/2}=\max\bigl(|u_j|+\sqrt{gh_j},\,|u_{j+1}|+\sqrt{gh_{j+1}}\bigr),

时间推进显式。**边界**：:func:`swe_step_1d` / :func:`swe_solve_1d` 缺省用
``bc="outflow"``（零阶外推鬼格，等价于「界面通量取边界格自己的物理通量」，
一阶透明/无反射，边界外的波直接离开、不回来）；``bc="periodic"`` 用环绕邻居。
两者在本模块里都**只有一个真源**：:func:`swe_rhs_1d`。

这一章最容易骗人的地方（数字全部是本仓库实测，跑法见 ``tests/6.1/``）
--------------------------------------------------------------------
1. **对拍窗口决定成败，而不是格式决定成败**。干底 Ritter 解只在波扇
   :math:`-c_L\le x/t\le 2c_L` 内处处光滑；把 L1 误差算到整个域上，
   ``n=400`` 时相对误差从 :math:`3.8\times10^{-3}`（扇内）跳到
   :math:`2.0\times10^{-2}`（全域，含湿/干锋面与扇外），**差 5 倍**——
   而"全域误差"里主要装的是干锋的位置滞后，不是格式精度。
2. **"一阶格式"在间断附近会退化**。扇内是光滑的（:math:`h` 对 :math:`x/t`
   是二次多项式），实测 L1 阶 :math:`0.98`（:math:`n=100\to800`）；
   一旦把干锋包进窗口，实测阶掉到 :math:`0.55` 上下。写"理论阶 1"就当实测 1，
   是本章最容易犯的假证据。
3. **Rusanov 的波速上界让通量失去"逐点单调"的直觉**。
   :math:`F^{h}` 对 :math:`h_L` 单调不减、对 :math:`h_R` 单调不增**只在
   两态都静止**（:math:`hu\equiv0`）时逐点成立；带上动量后同一句话就不再逐点真
   （本模块的断言只测静止方向，并把这一限制写进 ``detail``）。
4. **静水（lake at rest）不是"小量"，是精确恒等式**。带底高的格式若不做
   静水重构，:math:`h+b\equiv\text{const},hu\equiv0` 会被底坡源项激发出
   :math:`O(\Delta x^2)` 的伪动量，50 步后无量纲漂移 :math:`1.5\times10^{-2}`；
   做了 Audusse 静水重构 + 界面源项后，同一算例的漂移是 :math:`7.4\times10^{-17}`
   （机器精度）。**注意**：教科书式的朴素源项
   :math:`-gh_j(b_{j+1/2}-b_{j-1/2})/\Delta x` 与 ``max`` 界面底高搭配时
   **并不**well-balanced，实测漂移 :math:`1.1\times10^{-3}`——它看着完全正确。
5. **干底上"守恒"与"正性"是对立的**。:math:`h` 地板（``min_depth``）本身注入
   质量：干底算例里它每步最多注入 :math:`\text{min\_depth}\cdot\Delta x` 的水，
   所以那些算例的质量漂移**本来就非零**，拿它当"守恒性证据"是假证据；
   真正该测的是湿域（无干格）算例与周期域。
6. **能量不是离散守恒量**。:math:`E=\int(\frac12hu^2+\frac12gh^2)\,dx` 是守恒变量的
   **非线性**泛函，有限体积格式没有与之相容的离散通量：实测周期域光滑解在
   :math:`t=0.5` 内相对能量损失 :math:`2.5\times10^{-4}`（:math:`n=200`），
   :math:`n` 加倍后按 :math:`\Delta x^{1.0}` 下降（数值耗散的一阶率），
   而不是收敛到 0 的舍入量级。**出流边界下更不守恒**：波一离开就带走能量。
7. **CFL 是"实测出来的"，不是"抄回来的"**。本模块固定 :math:`\Delta t` 的二分实测
   给出临界 CFL :math:`1.07`（湿溃坝、:math:`n=100`、200 步），
   所以 ``cfl<=0.9`` 稳、``cfl>=1.2`` 必炸；把 ``max_stable_dt`` 的理论值
   :math:`\Delta x/a_{\max}` 当"稳定上界"用会高估 :math:`\sim 7\%`。
   :func:`cfl_number` 如实报出超过 1 的实际 CFL，**不裁剪**。

模块边界与依赖
--------------
只依赖 :mod:`numpy` 的容器与逐元素原语，以及 4.2 章的
:func:`numcore.pde.limiters.minmod`（``scheme="muscl"`` 的二阶重构斜率，
斜率限制器**不重复实现**）。不 import :mod:`numcore.reference`（P10）；
不碰 ``numpy.linalg`` / ``numpy.fft`` / 任何成品算法（P2/P3）；
纯函数、无 IO、无随机数、无全局可变状态（P6）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from numcore.pde.limiters import minmod

__all__ = [
    "G_DEFAULT",
    "SCHEMES",
    "LAKE_LEVEL",
    "SWEParams",
    "SWEResult",
    "SWE2DResult",
    "dam_break_1d",
    "swe_flux",
    "rusanov_flux",
    "swe_rhs_1d",
    "swe_step_1d",
    "swe_solve_1d",
    "wave_speed",
    "cfl_number",
    "max_stable_dt",
    "ritter_dam_break",
    "lake_bottom",
    "lake_at_rest",
    "lake_at_rest_metrics",
    "lake_at_rest_drift",
    "swe_rhs_1d_bathy",
    "swe_step_1d_bathy",
    "total_mass",
    "total_momentum",
    "total_energy",
    "radial_dam_break_2d",
    "swe_step_2d",
    "swe_solve_2d",
]

#: 重力加速度（本模块所有算例共用；SI 单位）。
G_DEFAULT: float = 9.81

#: 支持的空间格式：一阶 Rusanov 与 MUSCL（minmod 斜率限制）重构 + Rusanov。
SCHEMES: tuple[str, ...] = ("rusanov", "muscl")

#: :func:`lake_at_rest` 的静水面高程（:math:`h+b=\text{const}` 里的 const）。
LAKE_LEVEL: float = 1.0

#: 速度兜底：:math:`u=(hu)/\max(h,\varepsilon)`，避免干格上的 ``0/0``。
_H_FLOOR: float = 1.0e-12

#: 全域全干（:math:`h\equiv0,hu\equiv0`）时给 :func:`max_stable_dt` 的参考水深。
_DRY_REF: float = 1.0e-12


# --------------------------------------------------------------------------- #
# 参数与结果容器
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SWEParams:
    r"""一维浅水算例的参数（平坦底、无摩阻）。

    ``n`` 是格数、``length`` 是域长（:math:`\Delta x=\text{length}/n`）、
    ``t_end`` 是终止时刻、``cfl`` 是目标 CFL 数（:math:`\Delta t=\text{cfl}\cdot
    \Delta x/a_{\max}`）、``g`` 是重力加速度、``scheme`` ∈ :data:`SCHEMES`。
    """

    n: int
    length: float
    t_end: float
    cfl: float
    g: float = G_DEFAULT
    scheme: str = "rusanov"


@dataclass
class SWEResult:
    r"""一维浅水结果：终态 + 三个守恒量的历史 + 实际用过的最大 CFL。

    ``u`` 是原始速度 :math:`hu/\max(h,\varepsilon)`（干格上为 0）；
    ``mass/momentum/energy`` 是**终态**的三个积分量，``*_history`` 是每一步后的
    同一个积分量（长度 ``n_steps+1``，第 0 项是初值）。
    ``max_cfl`` 是整段推进里 :func:`cfl_number` 的**最大值**（不裁剪、不如实折算，
    实际值可能略大于 ``params.cfl``——因为波速在一步内变了）。
    """

    x: np.ndarray
    h: np.ndarray
    hu: np.ndarray
    u: np.ndarray
    t: float
    dx: float
    dt: float
    n_steps: int
    mass: float
    momentum: float
    energy: float
    mass_history: np.ndarray
    momentum_history: np.ndarray
    energy_history: np.ndarray
    max_cfl: float
    time_history: np.ndarray = field(default_factory=lambda: np.zeros(1))
    bc: str = "outflow"
    scheme: str = "rusanov"
    h_min: float = 0.0
    reached_t_end: bool = True


@dataclass
class SWE2DResult:
    r"""二维径向溃坝结果（平底、出流边界、:math:`n\times n` 均匀网格）。

    数组都是 ``(n, n)``（``x``/``y`` 是一维坐标），第 0 轴对应 :math:`x`、
    第 1 轴对应 :math:`y`；``mom`` 是 ``(2, n, n)`` 的动量栈（``mom[0]=hu``、
    ``mom[1]=hv``），``speed`` 是 :math:`\sqrt{u^2+v^2}`。
    """

    x: np.ndarray
    y: np.ndarray
    h: np.ndarray
    hu: np.ndarray
    hv: np.ndarray
    u: np.ndarray
    v: np.ndarray
    speed: np.ndarray
    t: float
    dx: float
    dy: float
    dt: float
    n_steps: int
    mass: float
    momentum_x: float
    momentum_y: float
    energy: float
    max_cfl: float
    mass_history: np.ndarray
    energy_history: np.ndarray
    h_min: float
    reached_t_end: bool = True


# --------------------------------------------------------------------------- #
# 内部小工具
# --------------------------------------------------------------------------- #
def _check_n(n: int, *, least: int = 4) -> int:
    """格数校验（``n`` 太小的话"格心网格"没有意义，直接报错而不是猜）。"""
    k = int(n)
    if k < int(least):
        raise ValueError(f"格数至少 {int(least)}，得到 {k}")
    return k


def _check_bc(bc: str) -> str:
    """边界名规范化（只认 ``outflow`` 与 ``periodic``，拼错即报错）。"""
    key = str(bc).strip().lower()
    if key not in ("outflow", "periodic"):
        raise ValueError(f"未知边界 {bc!r}（可用：outflow / periodic）")
    return key


def _check_scheme(scheme: str) -> str:
    """格式名规范化（只认 :data:`SCHEMES`，拼错即报错）。"""
    key = str(scheme).strip().lower()
    if key not in SCHEMES:
        raise ValueError(f"未知格式 {scheme!r}（可用：{list(SCHEMES)}）")
    return key


def _velocity(h: np.ndarray, hu: np.ndarray) -> np.ndarray:
    r"""原始速度 :math:`u=(hu)/\max(h,\varepsilon)`（干格上为 0，不出 ``nan``）。"""
    return np.asarray(hu, dtype=float) / np.maximum(np.asarray(h, dtype=float), _H_FLOOR)


def _uniform_dx(x: np.ndarray) -> float:
    """从坐标数组反解均匀步长（非均匀直接报错：本模块只支持均匀格心网格）。"""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < 2:
        raise ValueError(f"坐标数组至少 2 个点，得到 {arr.size}")
    d = float(arr[1] - arr[0])
    spread = float(np.max(np.abs(arr[-1] - arr[0] - d * (arr.size - 1))))
    if spread > 1.0e-9 * max(abs(d), 1.0):
        raise ValueError("本模块只支持均匀网格（坐标间距必须相等）")
    return d


def _cell_centers(n: int, length: float) -> tuple[np.ndarray, float]:
    r"""格心坐标与步长：:math:`x_j=(j+\frac12)\Delta x`。"""
    nn = _check_n(n)
    dx = float(length) / nn
    return (np.arange(nn, dtype=float) + 0.5) * dx, dx


# --------------------------------------------------------------------------- #
# 初值与精确解
# --------------------------------------------------------------------------- #
def dam_break_1d(
    n: int,
    *,
    length: float = 1.0,
    left: float = 1.0,
    right: float = 0.125,
    x0: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""溃坝初值：:math:`h=h_L\,(x<x_0)`、:math:`h=h_R\,(x>x_0)`，:math:`hu\equiv0`。

    返回 ``(x, h, hu)``。``right=0`` 就是干底（Ritter 考题的初值，
    与 :func:`ritter_dam_break` 配套）；``right=0.125`` 是湿底标准考题
    （:math:`h_L/h_R=8`，右行激波 + 左行稀疏波）。
    """
    x, _ = _cell_centers(n, length)
    h = np.where(x < float(x0), float(left), float(right))
    hu = np.zeros(x.shape, dtype=float)
    return x, np.asarray(h, dtype=float), hu


def ritter_dam_break(
    x: np.ndarray,
    t: float,
    *,
    h_left: float = 1.0,
    g: float = G_DEFAULT,
    x0: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    r"""干底 Ritter 精确解（:math:`h_R=0`，波扇自相似变量 :math:`\xi=(x-x_0)/t`）。

    .. math::
        c_L=\sqrt{gh_L},\qquad
        h=\begin{cases}
            h_L, & \xi\le -c_L,\\[2pt]
            \dfrac{(2c_L-\xi)^2}{9g}, & -c_L<\xi<2c_L,\\[4pt]
            0, & \xi\ge 2c_L,
        \end{cases}\qquad
        u=\begin{cases}
            0, & \xi\le -c_L,\\
            \tfrac23(\xi+c_L), & -c_L<\xi<2c_L,\\
            0, & \xi\ge 2c_L.
        \end{cases}

    返回 ``(h, u)``。**对拍窗口**：只有 :math:`\xi\in(-c_L,2c_L)` 内解是光滑的
    （:math:`h` 对 :math:`\xi` 是二次多项式），湿/干锋面 :math:`\xi=2c_L`
    与头部折点 :math:`\xi=-c_L` 都在窗口外——把整域喂进 L1 误差里测出来的
    是"锋面位置误差"，不是格式精度。

    ``t<=0`` 时自相似变量无定义：这里返回 :math:`t\to0^+` 的**逐点极限**
    （即初值 :math:`h_L\cdot[x<x_0]`、:math:`u\equiv0`），不抛异常、也不返回
    ``nan``。注意这条约定本身是一个陷阱：拿 ``t=0`` 做收敛阶会得到恒等于 0 的
    误差（任何格式都"精确"），那不是证据。
    """
    xs = np.asarray(x, dtype=float)
    hl = float(h_left)
    gg = float(g)
    if hl <= 0.0:
        raise ValueError(f"h_left 必须为正，得到 {hl}")
    if gg <= 0.0:
        raise ValueError(f"g 必须为正，得到 {gg}")
    t_in = float(t)
    if t_in <= 0.0:
        h = np.where(xs < float(x0), hl, 0.0)
        return np.asarray(h, dtype=float), np.zeros(xs.shape, dtype=float)
    c_l = float(np.sqrt(gg * hl))
    xi = (xs - float(x0)) / t_in
    left = xi <= -c_l
    right = xi >= 2.0 * c_l
    fan = ~left & ~right
    h = np.where(left, hl, 0.0)
    u = np.zeros(xs.shape, dtype=float)
    h = np.where(fan, (2.0 * c_l - xi) ** 2 / (9.0 * gg), h)
    u = np.where(fan, (2.0 / 3.0) * (xi + c_l), u)
    return np.asarray(h, dtype=float), np.asarray(u, dtype=float)


def lake_bottom(
    n: int,
    *,
    length: float = 1.0,
    amplitude: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    r"""静水考题的底高：中心处一个光滑凸包
    :math:`b(x)=A\exp[-((x-L/2)/(0.1L))^2]`（:math:`A=` ``amplitude``）。

    返回 ``(x, b)``。静水面 :math:`h+b=` :data:`LAKE_LEVEL`，所以
    ``amplitude`` 必须小于 :data:`LAKE_LEVEL`（否则初值里出现负水深，
    那是另一个（干底）问题，不该混进 well-balanced 断言）。
    """
    x, _ = _cell_centers(n, length)
    amp = float(amplitude)
    if not 0.0 < amp < LAKE_LEVEL:
        raise ValueError(f"amplitude 必须在 (0, LAKE_LEVEL=({LAKE_LEVEL})) 内，得到 {amp}")
    b = amp * np.exp(-(((x - 0.5 * float(length)) / (0.1 * float(length))) ** 2))
    return x, np.asarray(b, dtype=float)


def lake_at_rest(
    n: int,
    *,
    length: float = 1.0,
    g: float = G_DEFAULT,
    amplitude: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""静水（lake at rest）初值：:math:`h+b\equiv` :data:`LAKE_LEVEL`、:math:`hu\equiv0`。

    返回 ``(x, h, hu)``。底高由 :func:`lake_bottom` 给出（同一个真源），
    所以要拿它做漂移实验必须用同一组 ``n/length/amplitude`` 再调一次
    :func:`lake_bottom` 取 ``b``：

    .. code-block:: python

        x, h, hu = lake_at_rest(n, amplitude=a)
        _, b = lake_bottom(n, amplitude=a)
        drift = lake_at_rest_drift(h, hu, b, dx, steps=50)

    这个状态是**精确**平衡解（:math:`u\equiv0` 且自由面水平），任何"看起来
    不涨"的漂移都不是证据——本模块的断言要求它是舍入量级（实测
    :math:`7.4\times10^{-17}`），并同时给出朴素源项的反例（实测
    :math:`1.1\times10^{-3}`）。
    """
    x, b = lake_bottom(n, length=length, amplitude=amplitude)
    h = LAKE_LEVEL - b
    hu = np.zeros(x.shape, dtype=float)
    return x, np.asarray(h, dtype=float), hu


# --------------------------------------------------------------------------- #
# 通量与波速
# --------------------------------------------------------------------------- #
def swe_flux(h, hu, g: float = G_DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    r"""物理通量 :math:`F=(hu,\;hu^2/h+\tfrac12gh^2)`（逐分量，支持批量）。

    :math:`h<0` 的输入对压强项取 :math:`\max(h,0)`（负水深是**非物理态**，
    不是可以静默参与运算的数）；:math:`h=0,hu=0` 时给的正是 :math:`(0,0)`。
    """
    hh = np.asarray(h, dtype=float)
    mm = np.asarray(hu, dtype=float)
    hc = np.maximum(hh, 0.0)
    u = _velocity(hh, mm)
    f_h = mm
    f_hu = mm * u + 0.5 * float(g) * hc * hc
    return np.asarray(f_h, dtype=float), np.asarray(f_hu, dtype=float)


def wave_speed(h, u, g: float = G_DEFAULT) -> np.ndarray:
    r"""特征速度上界 :math:`|u|+\sqrt{gh}`（:math:`h<0` 取 :math:`\max(h,0)`）。"""
    hh = np.asarray(h, dtype=float)
    uu = np.asarray(u, dtype=float)
    return np.asarray(np.abs(uu) + np.sqrt(float(g) * np.maximum(hh, 0.0)), dtype=float)


def rusanov_flux(
    hL, huL, hR, huR, g: float = G_DEFAULT
) -> tuple[np.ndarray, np.ndarray]:
    r"""Rusanov（局部 Lax--Friedrichs）通量
    :math:`F=\frac12(F_L+F_R)-\frac12 a_{\max}(U_R-U_L)`。

    ``a_max`` 取两态 :func:`wave_speed` 的较大者。三条性质（都在
    ``tests/6.1`` 里逐点实测）：

    * **相容**：``hL==hR`` 且 ``huL==huR`` 时逐位等于 :func:`swe_flux`（实测相对差
      0）；
    * **静止方向单调**：``huL==huR==0`` 时 :math:`F^{h}` 关于 :math:`h_L`
      单调不减、关于 :math:`h_R` 单调不增（这正是"迎风把质量往低处推"的离散版本）；
      带上动量后这句话**不保证**逐点成立，别把它当定理用；
    * **不分辨波型**：一个常系数耗散同时抹平激波与稀疏波，所以激波被抹宽
      （这正是"看起来收敛"的 L1 阶在间断上退化的原因），代价是熵条件自动满足。
    """
    hL_a = np.asarray(hL, dtype=float)
    huL_a = np.asarray(huL, dtype=float)
    hR_a = np.asarray(hR, dtype=float)
    huR_a = np.asarray(huR, dtype=float)
    fL_h, fL_hu = swe_flux(hL_a, huL_a, g)
    fR_h, fR_hu = swe_flux(hR_a, huR_a, g)
    a = np.maximum(
        wave_speed(hL_a, _velocity(hL_a, huL_a), g),
        wave_speed(hR_a, _velocity(hR_a, huR_a), g),
    )
    f_h = 0.5 * (fL_h + fR_h) - 0.5 * a * (hR_a - hL_a)
    f_hu = 0.5 * (fL_hu + fR_hu) - 0.5 * a * (huR_a - huL_a)
    return np.asarray(f_h, dtype=float), np.asarray(f_hu, dtype=float)


def cfl_number(h, hu, dx: float, dt: float, g: float = G_DEFAULT) -> float:
    r"""实际 CFL 数 :math:`\Delta t\,\max_j(|u_j|+\sqrt{gh_j})/\Delta x`。

    **如实报出**：:math:`>1` 就是 :math:`>1`，不裁剪、不取 min。所有"越界"的
    证据都必须来自这个数，而不是来自"解看起来还活着"。
    """
    hh = np.asarray(h, dtype=float)
    d = float(dx)
    if d <= 0.0:
        raise ValueError(f"dx 必须为正，得到 {d}")
    a = float(np.max(wave_speed(hh, _velocity(hh, np.asarray(hu, dtype=float)), g)))
    return float(dt) * a / d


def max_stable_dt(
    h, hu, dx: float, g: float = G_DEFAULT, *, cfl: float = 0.4
) -> float:
    r"""显式推进的稳定步长估计 :math:`\Delta t=\text{cfl}\cdot\Delta x/a_{\max}`。

    :math:`a_{\max}=\max_j(|u_j|+\sqrt{gh_j})`。全域全干（:math:`a_{\max}=0`）时
    用参考水深 :data:`_DRY_REF` 兜底，保证返回值有限且为正——**它是估计，不是定理**：
    实测临界 CFL 是 :math:`1.07`（湿溃坝、:math:`n=100`、200 步二分），
    所以 ``cfl=1.0`` 给出的 :math:`\Delta t` 已经越界约 :math:`7\%`；
    生产用 ``cfl<=0.9``。
    """
    hh = np.asarray(h, dtype=float)
    d = float(dx)
    if d <= 0.0:
        raise ValueError(f"dx 必须为正，得到 {d}")
    a = float(np.max(wave_speed(hh, _velocity(hh, np.asarray(hu, dtype=float)), g)))
    if not np.isfinite(a) or a <= 0.0:
        a = float(np.sqrt(float(g) * _DRY_REF))
    return float(cfl) * d / a


# --------------------------------------------------------------------------- #
# 界面状态（一阶与 MUSCL 重构、周期与出流边界）
# --------------------------------------------------------------------------- #
def _interface_states(h, hu, bc: str, scheme: str) -> tuple[np.ndarray, ...]:
    r"""把单元量折成界面左右态（**唯一的**界面构造点，所有格式共用）。

    * ``bc="periodic"``：界面数 :math:`n`，最后一个界面连接 :math:`n-1` 与 :math:`0`；
    * ``bc="outflow"``：界面数 :math:`n+1`，两端各加一个等于边界格的鬼格
      （零阶外推），边界界面的通量因此退化为"边界格自己的物理通量"；
    * ``scheme="muscl"``：斜率 :math:`\sigma_j=\text{minmod}(h_j-h_{j-1},h_{j+1}-h_j)`
      （:func:`numcore.pde.limiters.minmod`，不重复实现），界面值
      :math:`h^L_{j+1/2}=h_j+\tfrac12\sigma_j`、:math:`h^R_{j+1/2}=h_{j+1}-\tfrac12\sigma_{j+1}`；
      minmod 的凸组合性质保证 :math:`h^\pm\ge0`（再取一次 :math:`\max(\cdot,0)`
      只是护栏），这是干底上敢开二阶重构的理由。
    """
    hh = np.asarray(h, dtype=float)
    mm = np.asarray(hu, dtype=float)
    per = bc == "periodic"
    if scheme == "rusanov":
        if per:
            return hh, mm, np.roll(hh, -1), np.roll(mm, -1)
        # 界面 k（k=0..n）夹在单元 k-1 与单元 k 之间：左态是把单元整体后移一位
        # 再补左鬼格，右态是前移一位再补右鬼格。**写反了会让边界通量错位一格**，
        # 溃坝算例第一步就会把湿格的水抽干（这个 bug 真发生过）。
        hL = np.concatenate([hh[:1], hh])
        huL = np.concatenate([mm[:1], mm])
        hR = np.concatenate([hh, hh[-1:]])
        huR = np.concatenate([mm, mm[-1:]])
        return hL, huL, hR, huR
    if per:
        dm = hh - np.roll(hh, 1)
        dp = np.roll(hh, -1) - hh
        sig = minmod(dm, dp)
        s_m = mm - np.roll(mm, 1)
        s_p = np.roll(mm, -1) - mm
        sig_m = minmod(s_m, s_p)
        hL = np.maximum(hh + 0.5 * sig, 0.0)
        hR = np.maximum(np.roll(hh - 0.5 * sig, -1), 0.0)
        huL = mm + 0.5 * sig_m
        huR = np.roll(mm - 0.5 * sig_m, -1)
        return hL, huL, hR, huR
    e = np.concatenate([hh[:1], hh, hh[-1:]])
    em = np.concatenate([mm[:1], mm, mm[-1:]])
    sig = minmod(e[1:-1] - e[:-2], e[2:] - e[1:-1])
    sig_m = minmod(em[1:-1] - em[:-2], em[2:] - em[1:-1])
    pad = np.concatenate([[0.0], 0.5 * sig, [0.0]])      # 鬼格斜率取 0
    pad_m = np.concatenate([[0.0], 0.5 * sig_m, [0.0]])
    hL = np.maximum((e + pad)[:-1], 0.0)
    hR = np.maximum((e - pad)[1:], 0.0)
    huL = (em + pad_m)[:-1]
    huR = (em - pad_m)[1:]
    return hL, huL, hR, huR


def swe_rhs_1d(
    h, hu, dx: float, g: float = G_DEFAULT, *, scheme: str = "rusanov", bc: str = "outflow"
) -> tuple[np.ndarray, np.ndarray]:
    r"""半离散右端 :math:`\mathrm{d}U_j/\mathrm{d}t=-(F_{j+1/2}-F_{j-1/2})/\Delta x`。

    返回 ``(dh, dhu)``。**这是本模块唯一的空间离散点**：:func:`swe_step_1d`、
    :func:`swe_solve_1d` 都只是在这里加一层时间推进，因此"1 维格式"不会出现
    两份会漂移的实现。边界由 ``bc`` 决定（见 :func:`_interface_states`）。
    """
    key_s = _check_scheme(scheme)
    key_b = _check_bc(bc)
    d = float(dx)
    if d <= 0.0:
        raise ValueError(f"dx 必须为正，得到 {d}")
    hL, huL, hR, huR = _interface_states(h, hu, key_b, key_s)
    f_h, f_hu = rusanov_flux(hL, huL, hR, huR, g)
    if key_b == "periodic":
        # 界面数 = 格数：第 j 格的左界面是第 (j-1) 格的右界面（环绕）。
        d_h = f_h - np.roll(f_h, 1)
        d_hu = f_hu - np.roll(f_hu, 1)
    else:
        d_h = f_h[1:] - f_h[:-1]
        d_hu = f_hu[1:] - f_hu[:-1]
    return (
        np.asarray(-d_h / d, dtype=float),
        np.asarray(-d_hu / d, dtype=float),
    )


def swe_step_1d(
    h,
    hu,
    dx: float,
    dt: float,
    g: float = G_DEFAULT,
    *,
    scheme: str = "rusanov",
    min_depth: float = 1.0e-8,
    bc: str = "outflow",
) -> tuple[np.ndarray, np.ndarray]:
    r"""显式一步（前向 Euler + :func:`swe_rhs_1d`），带**干底安全**。

    干底安全有三条，缺一不可：

    1. 水深地板 ``h = max(h, min_depth)``（``min_depth=0`` 表示**关掉地板**，
       专门留给"无地板就出负水深"的反例）；
    2. 干格（:math:`h\le\text{min\_depth}`）上的动量清零——干格不该携带动量，
       否则 :math:`u=hu/h` 会炸；
    3. 单步结果里的 ``h`` 非负且有限（地板注入的质量是**如实存在**的，
       所以在干底算例上质量漂移本来就非零，见 :func:`total_mass` 的说明）。

    返回 ``(h_new, hu_new)``。
    """
    d = float(dx)
    step = float(dt)
    if d <= 0.0:
        raise ValueError(f"dx 必须为正，得到 {d}")
    if step < 0.0:
        raise ValueError(f"dt 必须非负，得到 {step}")
    hh = np.asarray(h, dtype=float)
    mm = np.asarray(hu, dtype=float)
    if step == 0.0:
        return hh.copy(), mm.copy()
    dh, dhu = swe_rhs_1d(hh, mm, d, g, scheme=scheme, bc=bc)
    h_new = hh + step * dh
    hu_new = mm + step * dhu
    floor = float(min_depth)
    h_new = np.maximum(h_new, floor)
    dry = h_new <= floor
    hu_new = np.where(dry, 0.0, hu_new)
    return np.asarray(h_new, dtype=float), np.asarray(hu_new, dtype=float)


# --------------------------------------------------------------------------- #
# 时间推进主循环
# --------------------------------------------------------------------------- #
def swe_solve_1d(
    params: SWEParams,
    *,
    h0=None,
    hu0=None,
    bc: str = "outflow",
    max_steps: int = 200000,
) -> SWEResult:
    r"""按 :class:`SWEParams` 推进到 ``t_end``（自适应 :math:`\Delta t`）。

    缺省初值是 :func:`dam_break_1d`（``length``/``n`` 取自 ``params``，
    :math:`h_L=1,h_R=0.125,x_0=L/2`）。``bc`` 见 :func:`swe_rhs_1d`。
    ``max_steps`` 是**诚实的刹车**：撞上它就停下，并在 ``reached_t_end`` 里
    写 ``False``（而不是假装推到了 ``t_end``）。

    ``max_cfl`` 记录整段推进里 :func:`cfl_number` 的最大值；由于波速在一步内
    会变，它可能略大于 ``params.cfl``。
    """
    n = _check_n(params.n)
    key_b = _check_bc(bc)
    key_s = _check_scheme(params.scheme)
    x, dx = _cell_centers(n, params.length)
    if h0 is None:
        _, h, hu = dam_break_1d(n, length=params.length)
    else:
        h = np.asarray(h0, dtype=float).copy()
        if h.shape != x.shape:
            raise ValueError(f"h0 形状 {h.shape} 与网格 {x.shape} 不符")
        hu = np.zeros_like(h) if hu0 is None else np.asarray(hu0, dtype=float).copy()
        if hu.shape != x.shape:
            raise ValueError(f"hu0 形状 {hu.shape} 与网格 {x.shape} 不符")
    g = float(params.g)
    t_end = float(params.t_end)
    t = 0.0
    n_steps = 0
    dt_last = 0.0
    max_cfl = 0.0
    m_hist = [total_mass(x, h)]
    p_hist = [total_momentum(x, hu)]
    e_hist = [total_energy(x, h, hu, g)]
    t_hist = [0.0]
    reached = True
    while t < t_end:
        if n_steps >= int(max_steps):
            reached = False
            break
        dt = max_stable_dt(h, hu, dx, g, cfl=params.cfl)
        dt = min(dt, t_end - t)
        if not np.isfinite(dt) or dt <= 0.0:
            reached = False
            break
        max_cfl = max(max_cfl, cfl_number(h, hu, dx, dt, g))
        h, hu = swe_step_1d(h, hu, dx, dt, g, scheme=key_s, bc=key_b)
        t += dt
        n_steps += 1
        dt_last = dt
        m_hist.append(total_mass(x, h))
        p_hist.append(total_momentum(x, hu))
        e_hist.append(total_energy(x, h, hu, g))
        t_hist.append(t)
    return SWEResult(
        x=x,
        h=h,
        hu=hu,
        u=_velocity(h, hu),
        t=float(t),
        dx=dx,
        dt=float(dt_last),
        n_steps=int(n_steps),
        mass=m_hist[-1],
        momentum=p_hist[-1],
        energy=e_hist[-1],
        mass_history=np.asarray(m_hist, dtype=float),
        momentum_history=np.asarray(p_hist, dtype=float),
        energy_history=np.asarray(e_hist, dtype=float),
        max_cfl=float(max_cfl),
        time_history=np.asarray(t_hist, dtype=float),
        bc=key_b,
        scheme=key_s,
        h_min=float(np.min(h)),
        reached_t_end=bool(reached and t >= t_end - 1.0e-15 * max(abs(t_end), 1.0)),
    )


# --------------------------------------------------------------------------- #
# 守恒量（格心网格上的中点公式）
# --------------------------------------------------------------------------- #
def total_mass(x, h) -> float:
    r"""总质量 :math:`\int h\,\mathrm{d}x\approx\Delta x\sum_j h_j`。

    格心有限体积里这就是**离散**质量（与 :func:`swe_rhs_1d` 的通量差精确相容），
    所以周期域上它的漂移只该是舍入量级。**干底算例例外**：``min_depth``
    地板每步注入 :math:`\text{min\_depth}\cdot\Delta x\cdot\#\text{干格}` 的水，
    所以那里"质量漂移"里混着地板的水，不能当守恒性证据。
    """
    d = _uniform_dx(x)
    return float(np.sum(np.asarray(h, dtype=float)) * d)


def total_momentum(x, hu) -> float:
    r"""总动量 :math:`\int hu\,\mathrm{d}x\approx\Delta x\sum_j (hu)_j`。

    平底、周期边界下是离散守恒量；**出流边界下不是**（波离开就带走动量），
    出流算例报出来的动量变化是**真实现象**，不是 bug。
    """
    d = _uniform_dx(x)
    return float(np.sum(np.asarray(hu, dtype=float)) * d)


def total_energy(x, h, hu, g: float = G_DEFAULT) -> float:
    r"""总能量 :math:`\int\bigl(\frac12hu^2+\frac12gh^2\bigr)\mathrm{d}x`
    （平坦底，:math:`u=hu/\max(h,\varepsilon)`）。

    **它不是离散守恒量**：能量是守恒量的非线性泛函，有限体积格式里没有与之相容的
    离散通量，Rusanov 的数值耗散会让它单调下降（实测周期域光滑解、:math:`n=200`、
    :math:`t=0.5` 时相对损失 :math:`2.5\times10^{-4}`，随 :math:`\Delta x` 一阶下降）。
    出流边界下它还会被波**带走**一部分。所以本模块只把它当"有界的、可解释的量"报，
    任何"能量守恒"的断言都必须同时交代边界与网格。
    """
    d = _uniform_dx(x)
    hh = np.asarray(h, dtype=float)
    mm = np.asarray(hu, dtype=float)
    hc = np.maximum(hh, 0.0)
    u = _velocity(hh, mm)
    dens = 0.5 * hc * u * u + 0.5 * float(g) * hc * hc
    return float(np.sum(dens) * d)


# --------------------------------------------------------------------------- #
# 带底高：静水重构 + 界面源项（well-balanced）
# --------------------------------------------------------------------------- #
def swe_rhs_1d_bathy(
    h,
    hu,
    b,
    dx: float,
    g: float = G_DEFAULT,
    *,
    bc: str = "periodic",
    source: str = "well_balanced",
) -> tuple[np.ndarray, np.ndarray]:
    r"""带底高 :math:`b(x)` 的半离散右端（Audusse 静水重构 + 界面源项）。

    动量方程 :math:`(hu)_t+(hu^2+\tfrac12gh^2)_x=-gh\,b_x`。
    记界面底高 :math:`b_{j+1/2}=\max(b_j,b_{j+1})`、重构水深
    :math:`h^L_{j+1/2}=\max(0,\,h_j+b_j-b_{j+1/2})`（右侧同理），
    界面通量用重构态代入 :func:`rusanov_flux`，源项取

    .. math::
        S_j=\frac{g}{2\Delta x}\Bigl[\bigl(h^L_{j+1/2}\bigr)^2-\bigl(h^R_{j-1/2}\bigr)^2\Bigr].

    这个源项有两条性质（都在 ``tests/6.1`` 实测）：

    * **静水精确**：:math:`h+b\equiv H`、:math:`hu\equiv0` 时
      :math:`h^L=h^R=H-b_{j+1/2}`，通量差与源项逐位相消，漂移 = 舍入；
    * **相容**：光滑流上 Taylor 展开给出 :math:`S_j=-gh\,b_x+O(\Delta x)`，
      与通量差里的 :math:`-gh\,h_x` 合成 :math:`-gh\,(h+b)_x`，正是动量方程。

    ``source="naive"`` 是**负对照**：换成教科书式的
    :math:`-gh_j(b_{j+1/2}-b_{j-1/2})/\Delta x`。它与同一个 ``max`` 界面底高
    搭配时**不**well-balanced（同阶项消不掉，残差 :math:`O(\Delta x)` 乘底坡平方），
    实测静水漂移从 :math:`7.4\times10^{-17}` 劣化到 :math:`1.1\times10^{-3}`——
    这正是"看着完全正确却悄悄造浪"的那类 bug。

    只支持一阶 ``scheme="rusanov"``（二阶重构要连 :math:`\eta=h+b` 一起重构，
    与本节主题无关，本模块不做）。
    """
    key_b = _check_bc(bc)
    key_src = str(source).strip().lower()
    if key_src not in ("well_balanced", "naive"):
        raise ValueError(f"未知源项写法 {source!r}（可用：well_balanced / naive）")
    d = float(dx)
    if d <= 0.0:
        raise ValueError(f"dx 必须为正，得到 {d}")
    gg = float(g)
    hh = np.asarray(h, dtype=float)
    mm = np.asarray(hu, dtype=float)
    bb = np.asarray(b, dtype=float)
    if not (hh.shape == mm.shape == bb.shape):
        raise ValueError(f"h/hu/b 形状必须一致，得到 {hh.shape}/{mm.shape}/{bb.shape}")
    u = _velocity(hh, mm)
    if key_b == "periodic":
        h_l, u_l, b_l = hh, u, bb
        h_r, u_r, b_r = np.roll(hh, -1), np.roll(u, -1), np.roll(bb, -1)
    else:
        h_l = np.concatenate([hh, hh[-1:]])
        u_l = np.concatenate([u, u[-1:]])
        b_l = np.concatenate([bb, bb[-1:]])
        h_r = np.concatenate([hh[:1], hh])
        u_r = np.concatenate([u[:1], u])
        b_r = np.concatenate([bb[:1], bb])
    b_i = np.maximum(b_l, b_r)
    hl = np.maximum(h_l + b_l - b_i, 0.0)
    hr = np.maximum(h_r + b_r - b_i, 0.0)
    f_h, f_hu = rusanov_flux(hl, hl * u_l, hr, hr * u_r, gg)
    src = np.zeros(hh.shape, dtype=float)
    if key_b == "periodic":
        src = 0.5 * gg * (hl ** 2 - np.roll(hr, 1) ** 2) / d
        naive = -gg * hh * (b_i - np.roll(b_i, 1)) / d
    else:
        src = 0.5 * gg * (hl[1:] ** 2 - hr[:-1] ** 2) / d
        naive = -gg * hh * (b_i[1:] - b_i[:-1]) / d
    if key_src == "naive":
        src = naive
    if key_b == "periodic":
        d_h = f_h - np.roll(f_h, 1)
        d_hu = f_hu - np.roll(f_hu, 1)
    else:
        d_h = f_h[1:] - f_h[:-1]
        d_hu = f_hu[1:] - f_hu[:-1]
    return (
        np.asarray(-d_h / d, dtype=float),
        np.asarray(-d_hu / d + src, dtype=float),
    )


def swe_step_1d_bathy(
    h,
    hu,
    b,
    dx: float,
    dt: float,
    g: float = G_DEFAULT,
    *,
    bc: str = "periodic",
    source: str = "well_balanced",
    min_depth: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray]:
    r"""带底高的显式一步（前向 Euler + :func:`swe_rhs_1d_bathy`，含干底地板）。"""
    d = float(dx)
    step = float(dt)
    if step < 0.0:
        raise ValueError(f"dt 必须非负，得到 {step}")
    hh = np.asarray(h, dtype=float)
    mm = np.asarray(hu, dtype=float)
    if step == 0.0:
        return hh.copy(), mm.copy()
    dh, dhu = swe_rhs_1d_bathy(hh, mm, b, d, g, bc=bc, source=source)
    h_new = np.maximum(hh + step * dh, float(min_depth))
    hu_new = np.where(h_new <= float(min_depth), 0.0, mm + step * dhu)
    return np.asarray(h_new, dtype=float), np.asarray(hu_new, dtype=float)


def lake_at_rest_metrics(
    h,
    hu,
    b,
    dx: float,
    g: float = G_DEFAULT,
    *,
    steps: int = 50,
    dt: float | None = None,
    source: str = "well_balanced",
    bc: str = "periodic",
) -> dict[str, float]:
    r"""静水算例的**分量**度量（:func:`lake_at_rest_drift` 的展开版，供诊断用）。

    返回字典：``level``（初始水面高程 :math:`\max(h+b)`）、``drift_momentum``
    （:math:`\max|hu|/(H\sqrt{gH})`，无量纲伪动量，Froude 型）、
    ``drift_surface``（:math:`\max|(h+b)-H|/H`，自由面相对起伏）、
    ``h_min``、``hu_max``、``steps``、``dt``。
    """
    hh = np.asarray(h, dtype=float).copy()
    mm = np.asarray(hu, dtype=float).copy()
    bb = np.asarray(b, dtype=float)
    d = float(dx)
    gg = float(g)
    level = float(np.max(hh + bb))
    if dt is None:
        step = max_stable_dt(hh, mm, d, gg, cfl=0.4)
    else:
        step = float(dt)
    for _ in range(int(steps)):
        hh, mm = swe_step_1d_bathy(hh, mm, bb, d, step, gg, bc=bc, source=source)
    scale = level * float(np.sqrt(gg * level))
    return {
        "level": level,
        "drift_momentum": float(np.max(np.abs(mm))) / scale,
        "drift_surface": float(np.max(np.abs((hh + bb) - level))) / level,
        "h_min": float(np.min(hh)),
        "hu_max": float(np.max(np.abs(mm))),
        "steps": float(int(steps)),
        "dt": float(step),
    }


def lake_at_rest_drift(
    h,
    hu,
    b,
    dx: float,
    g: float = G_DEFAULT,
    *,
    steps: int = 50,
    dt: float | None = None,
    source: str = "well_balanced",
    bc: str = "periodic",
) -> float:
    r"""静水初值推进 ``steps`` 步后的**无量纲漂移**（well-balanced 的实测判据）。

    返回 :func:`lake_at_rest_metrics` 里两个无量纲量的较大者：

    .. math:,
        D=\max\Bigl(\underbrace{\frac{\max|hu|}{H\sqrt{gH}}}_{\text{伪动量/Froude}},\;
        \underbrace{\frac{\max|(h+b)-H|}{H}}_{\text{自由面起伏}}\Bigr),\qquad H=\max(h+b).

    这个量在"格式真的保持静水"时应该是舍入量级（实测
    :math:`7.4\times10^{-17}`），在"忘了底坡源项/用了朴素源项"时是
    :math:`10^{-3}\sim10^{-2}` 量级。:math:`\Delta t` 缺省按 ``cfl=0.4`` 估。
    """
    return float(max(lake_at_rest_metrics(
        h, hu, b, dx, g, steps=steps, dt=dt, source=source, bc=bc
    )[k] for k in ("drift_momentum", "drift_surface")))


# --------------------------------------------------------------------------- #
# 二维：径向溃坝
# --------------------------------------------------------------------------- #
def radial_dam_break_2d(
    n: int,
    *,
    length: float = 1.0,
    inner: float = 1.0,
    outer: float = 0.1,
    radius: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""二维径向溃坝初值：:math:`r<` ``radius`` 内 :math:`h=` ``inner``，
    外面 :math:`h=` ``outer``，:math:`hu=hv=0`。

    返回 ``(x, y, h, mom)``，其中 ``h`` 是 ``(n, n)``（第 0 轴 :math:`x`、
    第 1 轴 :math:`y`），``mom`` 是 ``(2, n, n)``：``hu = mom[0]``、``hv = mom[1]``。
    ``inner > outer > 0`` 保证全域湿底，所以这个算例里没有干格、
    ``min_depth`` 地板一次都不会触发（质量漂移因此只来自边界通量）。
    """
    nn = _check_n(n)
    L = float(length)
    dx = L / nn
    x = (np.arange(nn, dtype=float) + 0.5) * dx
    X, Y = np.meshgrid(x, x, indexing="ij")
    r = np.sqrt((X - 0.5 * L) ** 2 + (Y - 0.5 * L) ** 2)
    h = np.where(r < float(radius), float(inner), float(outer))
    mom = np.zeros((2, nn, nn), dtype=float)
    return x, x.copy(), np.asarray(h, dtype=float), mom


def _flux_x_2d(h, hu, hv, g: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""二维 :math:`x` 方向通量 :math:`(hu,\;hu\,u+\tfrac12gh^2,\;hv\,u)`。"""
    hc = np.maximum(np.asarray(h, dtype=float), 0.0)
    u = _velocity(h, hu)
    v = _velocity(h, hv)
    return (
        np.asarray(hu, dtype=float),
        np.asarray(hu, dtype=float) * u + 0.5 * float(g) * hc * hc,
        np.asarray(hv, dtype=float) * u,
    )


def _flux_y_2d(h, hu, hv, g: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""二维 :math:`y` 方向通量 :math:`(hv,\;hu\,v,\;hv\,v+\tfrac12gh^2)`。"""
    hc = np.maximum(np.asarray(h, dtype=float), 0.0)
    u = _velocity(h, hu)
    v = _velocity(h, hv)
    return (
        np.asarray(hv, dtype=float),
        np.asarray(hu, dtype=float) * v,
        np.asarray(hv, dtype=float) * v + 0.5 * float(g) * hc * hc,
    )


def _ghost_pair(a: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """零阶外推的一对鬼格切片（``outflow``）：返回 ``(aL, aR)``，长度 +1。"""
    lead = np.take(a, [0], axis=axis)
    tail = np.take(a, [a.shape[axis] - 1], axis=axis)
    return (
        np.concatenate([lead, a], axis=axis),
        np.concatenate([a, tail], axis=axis),
    )


def swe_step_2d(
    h,
    hu,
    hv,
    dx,
    dy,
    dt,
    g: float = G_DEFAULT,
    *,
    scheme: str = "rusanov",
    min_depth: float = 1.0e-8,
    bc: str = "outflow",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""二维显式一步（守恒型有限体积 + 两个方向的 Rusanov 通量，出流边界）。

    .. math::
        U^{n+1}_{ij}=U^n_{ij}-\frac{\Delta t}{\Delta x}\bigl(F^x_{i+1/2,j}-F^x_{i-1/2,j}\bigr)
        -\frac{\Delta t}{\Delta y}\bigl(F^y_{i,j+1/2}-F^y_{i,j-1/2}\bigr).

    两条与一维一致的纪律：**一个波速上界两个方向共用**
    （:math:`a=|(u,v)|+\sqrt{gh}`，取界面两侧的最大值），干格上的动量清零。
    ``scheme`` 只支持 ``"rusanov"``（二阶重构在二维要连底坡与干湿一起处理，
    本模块不做，宁可不做也不做错）。返回 ``(h, hu, hv)``。
    """
    key_s = _check_scheme(scheme)
    if key_s != "rusanov":
        raise ValueError(f"二维只支持 scheme='rusanov'，得到 {scheme!r}")
    key_b = _check_bc(bc)
    if key_b != "outflow":
        raise ValueError(f"二维只支持 bc='outflow'，得到 {bc!r}")
    hh = np.asarray(h, dtype=float)
    mm = np.asarray(hu, dtype=float)
    nn = np.asarray(hv, dtype=float)
    if not (hh.shape == mm.shape == nn.shape):
        raise ValueError(f"h/hu/hv 形状必须一致，得到 {hh.shape}/{mm.shape}/{nn.shape}")
    step = float(dt)
    if step < 0.0:
        raise ValueError(f"dt 必须非负，得到 {step}")
    if step == 0.0:
        return hh.copy(), mm.copy(), nn.copy()
    gg = float(g)
    dxf = float(dx)
    dyf = float(dy)
    if dxf <= 0.0 or dyf <= 0.0:
        raise ValueError(f"dx/dy 必须为正，得到 {dxf}/{dyf}")
    # ---- x 方向 ----
    hL, hR = _ghost_pair(hh, 0)
    huL, huR = _ghost_pair(mm, 0)
    hvL, hvR = _ghost_pair(nn, 0)
    fL = _flux_x_2d(hL, huL, hvL, gg)
    fR = _flux_x_2d(hR, huR, hvR, gg)
    aL = np.sqrt(_velocity(hL, huL) ** 2 + _velocity(hL, hvL) ** 2) + np.sqrt(
        gg * np.maximum(hL, 0.0)
    )
    aR = np.sqrt(_velocity(hR, huR) ** 2 + _velocity(hR, hvR) ** 2) + np.sqrt(
        gg * np.maximum(hR, 0.0)
    )
    a = np.maximum(aL, aR)
    fx = tuple(
        0.5 * (fL[k] + fR[k]) - 0.5 * a * (hR - hL if k == 0 else (huR - huL if k == 1 else hvR - hvL))
        for k in range(3)
    )
    # ---- y 方向 ----
    gL, gR = _ghost_pair(hh, 1)
    puL, puR = _ghost_pair(mm, 1)
    pvL, pvR = _ghost_pair(nn, 1)
    qL = _flux_y_2d(gL, puL, pvL, gg)
    qR = _flux_y_2d(gR, puR, pvR, gg)
    bL = np.sqrt(_velocity(gL, puL) ** 2 + _velocity(gL, pvL) ** 2) + np.sqrt(
        gg * np.maximum(gL, 0.0)
    )
    bR = np.sqrt(_velocity(gR, puR) ** 2 + _velocity(gR, pvR) ** 2) + np.sqrt(
        gg * np.maximum(gR, 0.0)
    )
    b = np.maximum(bL, bR)
    fy = tuple(
        0.5 * (qL[k] + qR[k]) - 0.5 * b * (gR - gL if k == 0 else (puR - puL if k == 1 else pvR - pvL))
        for k in range(3)
    )
    h_new = hh - step / dxf * (fx[0][1:, :] - fx[0][:-1, :]) - step / dyf * (
        fy[0][:, 1:] - fy[0][:, :-1]
    )
    hu_new = mm - step / dxf * (fx[1][1:, :] - fx[1][:-1, :]) - step / dyf * (
        fy[1][:, 1:] - fy[1][:, :-1]
    )
    hv_new = nn - step / dxf * (fx[2][1:, :] - fx[2][:-1, :]) - step / dyf * (
        fy[2][:, 1:] - fy[2][:, :-1]
    )
    floor = float(min_depth)
    h_new = np.maximum(h_new, floor)
    dry = h_new <= floor
    hu_new = np.where(dry, 0.0, hu_new)
    hv_new = np.where(dry, 0.0, hv_new)
    return (
        np.asarray(h_new, dtype=float),
        np.asarray(hu_new, dtype=float),
        np.asarray(hv_new, dtype=float),
    )


def swe_solve_2d(
    n: int,
    *,
    t_end: float,
    cfl: float = 0.4,
    length: float = 1.0,
    inner: float = 1.0,
    outer: float = 0.1,
    radius: float = 0.25,
    g: float = G_DEFAULT,
    max_steps: int = 600,
    min_depth: float = 1.0e-8,
) -> SWE2DResult:
    r"""二维径向溃坝推进（供帧直接消费）。

    自适应步长 :math:`\Delta t=\text{cfl}\cdot\min(\Delta x,\Delta y)/a_{\max}`，
    每一步都会重新估波速。``max_steps`` 是硬上限（缺省 600）：撞上就停下并把
    ``reached_t_end`` 写成 ``False``，**不假装**推到了 ``t_end``。
    出流边界、全湿初值 ⟹ 全域没有干格，``min_depth`` 地板不会触发。

    性能（实测）：:math:`n=96`、600 步约 0.1 s（纯 numpy 数组运算，
    每步约 20 次 ``(96,96)`` 级运算），远低于帧的 1 s 预算。
    """
    nn = _check_n(n)
    L = float(length)
    dx = L / nn
    x, y, h, mom = radial_dam_break_2d(
        nn, length=L, inner=inner, outer=outer, radius=radius
    )
    hu = np.asarray(mom[0], dtype=float).copy()
    hv = np.asarray(mom[1], dtype=float).copy()
    gg = float(g)
    t = 0.0
    n_steps = 0
    dt_last = 0.0
    max_cfl = 0.0
    m_hist = [_mass_2d(h, dx)]
    e_hist = [_energy_2d(h, hu, hv, gg, dx)]
    reached = True
    while t < float(t_end):
        if n_steps >= int(max_steps):
            reached = False
            break
        sp = np.sqrt(_velocity(h, hu) ** 2 + _velocity(h, hv) ** 2) + np.sqrt(
            gg * np.maximum(h, 0.0)
        )
        a = float(np.max(sp))
        if not np.isfinite(a) or a <= 0.0:
            a = float(np.sqrt(gg * _DRY_REF))
        dt = float(cfl) * dx / a
        dt = min(dt, float(t_end) - t)
        if not np.isfinite(dt) or dt <= 0.0:
            reached = False
            break
        max_cfl = max(max_cfl, float(dt) * a / dx)
        h, hu, hv = swe_step_2d(
            h, hu, hv, dx, dx, dt, gg, min_depth=min_depth
        )
        t += dt
        n_steps += 1
        dt_last = dt
        m_hist.append(_mass_2d(h, dx))
        e_hist.append(_energy_2d(h, hu, hv, gg, dx))
    u = _velocity(h, hu)
    v = _velocity(h, hv)
    return SWE2DResult(
        x=x,
        y=y,
        h=h,
        hu=hu,
        hv=hv,
        u=u,
        v=v,
        speed=np.sqrt(u * u + v * v),
        t=float(t),
        dx=dx,
        dy=dx,
        dt=float(dt_last),
        n_steps=int(n_steps),
        mass=m_hist[-1],
        momentum_x=float(np.sum(hu) * dx * dx),
        momentum_y=float(np.sum(hv) * dx * dx),
        energy=e_hist[-1],
        max_cfl=float(max_cfl),
        mass_history=np.asarray(m_hist, dtype=float),
        energy_history=np.asarray(e_hist, dtype=float),
        h_min=float(np.min(h)),
        reached_t_end=bool(reached and t >= float(t_end) - 1.0e-15 * max(abs(float(t_end)), 1.0)),
    )


def _mass_2d(h: np.ndarray, d: float) -> float:
    r"""二维总质量 :math:`\Delta x\Delta y\sum h`（内部小工具）。"""
    return float(np.sum(h) * d * d)


def _energy_2d(h, hu, hv, g: float, d: float) -> float:
    """二维总能量（内部小工具；定义与一维一致，平坦底）。"""
    hc = np.maximum(np.asarray(h, dtype=float), 0.0)
    u = _velocity(h, hu)
    v = _velocity(h, hv)
    dens = 0.5 * hc * (u * u + v * v) + 0.5 * float(g) * hc * hc
    return float(np.sum(dens) * d * d)
