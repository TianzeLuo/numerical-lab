r"""4.3 章内核之二：有限体积法、守恒账本与熵诊断。

这一层把"通量"变成"能跑的格式"，并且**把账记清楚**：

* :func:`fv_step` —— 一阶有限体积更新 :math:`u_j^{n+1}=u_j-\frac{\Delta t}{\Delta x}(F_{j+1/2}-F_{j-1/2})`，
  通量由 :mod:`numcore.pde.riemann` 给（Godunov / Rusanov / Osher / 朴素迎风）；
* :func:`muscl_hancock_step` —— 二阶：限制斜率重构 + 半步预测 + 界面 Riemann 通量，
  限制器直接复用 :mod:`numcore.pde.limiters`（**不重复实现 4.2 的十个限制器**）；
* :func:`advective_step` —— **非守恒**的"特征线形式" :math:`u_t+f'(u)u_x=0`，
  光滑区与守恒形式逐点一致、激波处给出错误速度：守恒账本的负对照；
* :func:`solve` —— 时间推进 + 三本账（质量、边界通量累计、熵收支），
  每一步都记账，因此"守恒到机器零"不是一个说法而是一条曲线。

三条贯穿全章的判据（都是**可测**的）：

1. **守恒性**：:math:`M(t)-M(0)+\int_0^t(F_{\text{out}}-F_{\text{in}})\,dt=0`
   ——守恒格式到舍入，非守恒格式按 :math:`O(1)` 漏；
2. **熵条件**：:math:`\dot E_\eta+\Delta Q\le 0` ——正确解在激波处耗散（严格负）、
   在光滑稀疏波处趋于 0（:math:`O(\Delta x)`）；**膨胀激波**（跨声速稀疏波上
   用朴素迎风通量）会让它变成正数，而且**不随网格变小而消失**；
3. **极值原理 / TVD**：一阶 Godunov 不产生新极值、TV 不增；MUSCL 在 minmod 下同样
   （sweby 区域内的限制器都保 TVD），但二阶格式在**极值点**会退化成一阶——
   这是限制器的代价，不是 bug。

\(t\) 与 \(\Delta t\)：本模块的 :func:`solve` 用 CFL 自适应步长推进到 ``t_end``，
``t_actual`` 是**真实到达的时刻**（未必正好等于 ``t_end``）——帧里必须报 ``t_actual``，
不能把滑块的 ``t`` 当成"解在 t 处的值"（差半步会让所有误差曲线平移）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from numcore.pde import limiters as lm
from numcore.pde import riemann as rr

__all__ = [
    "SCHEMES",
    "CONSERVATIVE_SCHEMES",
    "INITIAL_DATA",
    "cell_centers",
    "initial_data",
    "max_speed",
    "cfl_dt",
    "interface_states",
    "interface_fluxes",
    "fv_step",
    "muscl_hancock_step",
    "advective_step",
    "entropy",
    "entropy_flux",
    "total_mass",
    "FVMResult",
    "solve",
    "euler_fv_step",
    "euler_solve",
    "EulerResult",
    "exact_riemann_solution",
    "smooth_profile",
    "profile_derivative",
    "breaking_time",
    "breaking_time_numeric",
    "characteristic_lines",
    "exact_smooth_burgers",
    "l1_error",
    "fit_speed",
    "shock_track",
]

#: 一阶通量格式：前三个相容且保熵，``upwind`` 是跨声速稀疏波上的熵违背负对照。
CONSERVATIVE_SCHEMES: tuple[str, ...] = ("godunov", "rusanov", "osher", "upwind")

#: 全部格式（含非守恒的 ``advective``）。
SCHEMES: tuple[str, ...] = CONSERVATIVE_SCHEMES + ("advective",)

#: 初值类型。
INITIAL_DATA: tuple[str, ...] = ("shock", "rarefaction", "sine", "gaussian")

#: 边界条件：``outflow``（零梯度，质量可以流出）与 ``periodic``。
BCS: tuple[str, ...] = ("outflow", "periodic")


def cell_centers(n: int, dx: float, x0: float = 0.0) -> np.ndarray:
    r"""单元中心坐标 :math:`x_j=x_0+(j+\tfrac12)\Delta x`（有限体积里解是单元平均）。"""
    m = int(n)
    if m < 2:
        raise ValueError(f"至少需要 2 个单元，得到 {n}")
    return float(x0) + (np.arange(m, dtype=float) + 0.5) * float(dx)


def initial_data(
    key: str,
    x: Any,
    *,
    amplitude: float = 0.5,
    width: float = 0.08,
    u_left: float = 1.0,
    u_right: float = 0.0,
    center: float = 0.5,
) -> np.ndarray:
    r"""四种初值，各自对应一种"看不见的失败"。

    * ``shock``：右行激波（:math:`u_L=1,u_R=0`，Burgers 的理论激波速度 0.5）。
      最容易被骗的地方是**周期域**：波绕过边界后与自身相互作用，表观的"激波速度"
      就不再是 RH 值；
    * ``rarefaction``：跨声速稀疏波（:math:`u_L=-1,u_R=1`）——扇面里含
      :math:`f'(u)=0` 的临界点。用朴素迎风通量它会**冻成一条驻定间断**；
    * ``sine``：光滑周期波，破散时间 :math:`t_b=1/(2\pi A)` 有闭式，用来量格式阶；
    * ``gaussian``：非周期光滑凸包，破散时间只能数值扫出来。
    """
    xs = np.asarray(x, dtype=float)
    c = float(center)
    if key == "shock":
        return np.where(xs < c, float(u_left), float(u_right)) + 0.0 * xs
    if key == "rarefaction":
        return np.where(xs < c, -1.0, 1.0) + 0.0 * xs
    if key == "sine":
        return float(amplitude) * np.sin(2.0 * np.pi * xs)
    if key == "gaussian":
        return float(amplitude) * np.exp(-(((xs - c) / float(width)) ** 2))
    raise ValueError(f"未知初值 {key!r}（可用：{list(INITIAL_DATA)}）")


def max_speed(u: Any, kind: str = "burgers") -> float:
    r"""当前解上的最大特征速度 :math:`\max_j|f'(u_j)|`（CFL 的波速）。"""
    arr = np.asarray(u, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(np.asarray(rr.flux_derivative(kind, arr), dtype=float))))


def cfl_dt(u: Any, dx: float, cfl: float, kind: str = "burgers") -> float:
    r"""CFL 步长 :math:`\Delta t=\text{cfl}\cdot\Delta x/\max|f'|`（波速为 0 时返回 ``inf``）。

    返回 ``inf`` 是刻意的：调用方必须自己处理"常值解"这个退化情形；
    悄悄换成一个有限的大数会让后续的步数估计变成假证据。
    """
    a = max_speed(u, kind)
    if a <= 0.0:
        return float("inf")
    return float(cfl) * float(dx) / a


def _boundary_values(u: np.ndarray, bc: str) -> tuple[float, float]:
    r"""两侧边界的"外部状态"：``outflow`` 取零梯度（贴边单元值），``periodic`` 取对侧。"""
    if bc == "periodic":
        return float(u[-1]), float(u[0])
    if bc == "outflow":
        return float(u[0]), float(u[-1])
    raise ValueError(f"未知边界条件 {bc!r}（可用：{list(BCS)}）")


def interface_states(u: Any, bc: str = "outflow") -> tuple[np.ndarray, np.ndarray]:
    r"""一阶界面左右状态（长度 ``n+1``，下标 ``j`` = 单元 ``j-1`` 与 ``j`` 之间的界面）。

    两个边界界面的状态**必须按物理含义给**：

    * ``outflow``（零梯度）：界面 0 的两侧都取 :math:`u_0`、界面 :math:`n` 都取
      :math:`u_{n-1}`，于是边界通量就是物理通量 :math:`f(u_{\text{edge}})`；
    * ``periodic``：界面 0 与界面 :math:`n` 是**同一条缝**，两侧状态必须是
      :math:`(u_{n-1},u_0)`——两端各写成"两侧同值"会让 :math:`F_0\ne F_n`，
      格式立刻多出一个内部源：实测周期算例的总质量漂 0.6%（0.5000 -> 0.4969），
      而账本残差只有 :math:`1.6\times10^{-16}`——账本自己平衡，质量自己在漏。
      这条是"账本平衡不等于格式守恒"的活证据。
    """
    arr = np.asarray(u, dtype=float).ravel()
    n = arr.size
    left = np.empty(n + 1, dtype=float)
    right = np.empty(n + 1, dtype=float)
    left[1:] = arr
    right[:-1] = arr
    if bc == "periodic":
        left[0] = left[n] = float(arr[-1])
        right[0] = right[n] = float(arr[0])
    elif bc == "outflow":
        left[0] = right[0] = float(arr[0])
        left[n] = right[n] = float(arr[-1])
    else:
        raise ValueError(f"未知边界条件 {bc!r}（可用：{list(BCS)}）")
    return left, right


def interface_fluxes(
    u: Any, kind: str = "burgers", scheme: str = "godunov", bc: str = "outflow"
) -> np.ndarray:
    r"""一阶数值通量（长度 ``n+1``；``F[j]`` 是单元 ``j-1`` 与 ``j`` 之间的界面通量）。"""
    left, right = interface_states(u, bc)
    return np.asarray(rr.scalar_numerical_flux(kind, left, right, scheme), dtype=float)


def fv_step(
    u: Any,
    dt: float,
    dx: float,
    *,
    kind: str = "burgers",
    scheme: str = "godunov",
    bc: str = "outflow",
) -> tuple[np.ndarray, np.ndarray]:
    r"""一阶有限体积推进一步，返回 ``(u_new, F)``（``F`` 是界面通量，用于画进出箭头）。

    :math:`u_j^{n+1}=u_j-\frac{\Delta t}{\Delta x}(F_{j+1/2}-F_{j-1/2})`——
    这个式子是**望远镜求和**的，所以内部界面通量怎么错都不会破坏守恒性；
    破坏守恒的只能是"不写成通量差"的离散（见 :func:`advective_step`）。
    """
    arr = np.asarray(u, dtype=float).ravel()
    f = interface_fluxes(arr, kind, scheme, bc)
    lam = float(dt) / float(dx)
    return arr - lam * (f[1:] - f[:-1]), f


def muscl_hancock_step(
    u: Any,
    dt: float,
    dx: float,
    *,
    kind: str = "burgers",
    scheme: str = "godunov",
    limiter: str = "minmod",
    bc: str = "outflow",
) -> tuple[np.ndarray, np.ndarray]:
    r"""MUSCL--Hancock 二阶格式一步（限制斜率 + 半步预测 + 界面 Riemann 通量）。

    三步：

    1. **重构**：:math:`\sigma_j=\phi(r_j)\Delta_+u_j`（直接调
       :func:`numcore.pde.limiters.limited_slope`，限制器清单与 4.2 章同一份）；
    2. **半步预测**：:math:`\bar u_j=u_j-\frac12\frac{\Delta t}{\Delta x}
       \bigl[f(u_j+\tfrac12\sigma_j)-f(u_j-\tfrac12\sigma_j)\bigr]`；
    3. **界面态**：:math:`\bar u^-_{j+1/2}=\bar u_j+\tfrac12\sigma_j`、
       :math:`\bar u^+_{j+1/2}=\bar u_{j+1}-\tfrac12\sigma_{j+1}`，再取 Riemann 通量。

    ``limiter="none"`` 时 :math:`\sigma\equiv0`，预测步退化成恒等，
    整个格式与 :func:`fv_step` **逐位相同**——这条恒等式是本章的一条断言
    （"二阶格式在极值处退化成一阶"这句话在这里有一个精确的、可执行的版本）。
    """
    arr = np.asarray(u, dtype=float).ravel()
    n = arr.size
    key = str(limiter).strip().lower()
    sigma = np.asarray(lm.limited_slope(arr, limiter=key, periodic=(bc == "periodic")), dtype=float)
    lam = float(dt) / float(dx)
    if key == "none" or not bool(np.any(sigma != 0.0)):
        return fv_step(arr, dt, dx, kind=kind, scheme=scheme, bc=bc)
    pred = arr - 0.5 * lam * (
        np.asarray(rr.scalar_flux(kind, arr + 0.5 * sigma), dtype=float)
        - np.asarray(rr.scalar_flux(kind, arr - 0.5 * sigma), dtype=float)
    )
    left = np.empty(n + 1, dtype=float)
    right = np.empty(n + 1, dtype=float)
    left[1:] = pred + 0.5 * sigma
    right[:-1] = pred - 0.5 * sigma
    if bc == "periodic":
        left[0] = left[n] = float(left[n])
        right[0] = right[n] = float(right[0])
    else:
        left[0] = right[0] = float(arr[0])
        left[n] = right[n] = float(arr[-1])
    f = np.asarray(rr.scalar_numerical_flux(kind, left, right, scheme), dtype=float)
    return arr - lam * (f[1:] - f[:-1]), f


def advective_step(
    u: Any,
    dt: float,
    dx: float,
    *,
    kind: str = "burgers",
    bc: str = "outflow",
) -> tuple[np.ndarray, np.ndarray]:
    r"""**非守恒**的"特征线形式" :math:`u_t+f'(u)u_x=0` 的迎风离散（负对照）。

    :math:`u_j^{n+1}=u_j-\frac{\Delta t}{\Delta x}f'(u_j)(u_j-u_{j-1})`
    （:math:`f'(u_j)<0` 时取右侧差分）。它与守恒形式在**光滑区逐点相同**
    （链式法则 :math:`(f(u))_x=f'(u)u_x`），所以任何"看光滑解"的检查都抓不住它；
    差别只在间断处：守恒形式给 RH 速度，这一支给**特征速度**（Burgers 的
    :math:`u_L`），于是激波位置越跑越偏、总质量单调漂移。

    返回的"通量"数组不是通量差（它压根没有通量差形式），只用来画箭头，
    因此守恒账本里必须用**显式的边界通量**来对账（见 :func:`solve` 的
    ``ledger_defect``）——拿它当通量差去累加就会得到假的机器零。
    """
    arr = np.asarray(u, dtype=float).ravel()
    n = arr.size
    lam = float(dt) / float(dx)
    d = np.asarray(rr.flux_derivative(kind, arr), dtype=float)
    um = np.empty(n, dtype=float)
    up = np.empty(n, dtype=float)
    um[0] = float(arr[0]) if bc == "outflow" else float(arr[-1])
    um[1:] = arr[:-1]
    up[-1] = float(arr[-1]) if bc == "outflow" else float(arr[0])
    up[:-1] = arr[1:]
    diff = np.where(d >= 0.0, arr - um, up - arr)
    return arr - lam * d * diff, np.zeros(n + 1, dtype=float)


# --------------------------------------------------------------------------- #
# 熵与守恒账本
# --------------------------------------------------------------------------- #
def entropy(u: Any) -> np.ndarray:
    r"""凸熵 :math:`\eta(u)=u^2/2`（三种通量共用；凸性只要求 :math:`\eta''>0`）。"""
    arr = np.asarray(u, dtype=float)
    return 0.5 * arr * arr


def entropy_flux(u: Any, kind: str = "burgers") -> np.ndarray:
    r"""熵通量 :math:`q(u)=\int^u\eta'(v)f'(v)\,dv`（Burgers 给 :math:`u^3/3`）。

    必须与 :func:`entropy` **成对**使用（熵对 :math:`(\eta,q)` 的选取不唯一，
    但一旦选定 :math:`q'=\eta'f'` 就唯一了）；换一个 :math:`\eta` 而忘了换 :math:`q`，
    熵收支会凭空出现一项"生产"，那是记账错误不是物理。
    """
    arr = np.asarray(u, dtype=float)
    key = str(kind).strip().lower()
    if key == "burgers":
        return arr ** 3 / 3.0
    if key == "linear":
        return 0.5 * rr.LINEAR_SPEED * arr * arr
    if key == "traffic":
        return 0.5 * arr * arr - 2.0 * arr ** 3 / 3.0
    raise ValueError(f"未知标量通量 {kind!r}（可用：{list(rr.SCALAR_FLUXES)}）")


def total_mass(u: Any, dx: float) -> float:
    r"""离散总质量 :math:`M=\sum_j u_j\Delta x`。"""
    return float(np.sum(np.asarray(u, dtype=float))) * float(dx)


@dataclass
class FVMResult:
    """一次 :func:`solve` 的全部账本（解 + 三条历史 + 边界通量累计）。"""

    x: np.ndarray
    u: np.ndarray
    dx: float
    t: float
    dt: float
    n_steps: int
    kind: str
    scheme: str
    limiter: str
    ic: str
    bc: str
    cfl: float
    time_history: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mass_history: np.ndarray = field(default_factory=lambda: np.zeros(0))
    defect_history: np.ndarray = field(default_factory=lambda: np.zeros(0))
    inflow_history: np.ndarray = field(default_factory=lambda: np.zeros(0))
    outflow_history: np.ndarray = field(default_factory=lambda: np.zeros(0))
    entropy_history: np.ndarray = field(default_factory=lambda: np.zeros(0))
    entropy_budget: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tv_history: np.ndarray = field(default_factory=lambda: np.zeros(0))
    max_history: np.ndarray = field(default_factory=lambda: np.zeros(0))
    min_history: np.ndarray = field(default_factory=lambda: np.zeros(0))
    flux: np.ndarray = field(default_factory=lambda: np.zeros(0))
    snapshots: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    snapshot_times: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def max_defect(self) -> float:
        r"""守恒账本的最大偏差 :math:`\max_n|M^n-M^0+\sum(F_{\text{out}}-F_{\text{in}})\Delta t|`。"""
        if self.mass_history.size == 0:
            return 0.0
        return float(np.max(np.abs(self.defect_history)))

    @property
    def relative_defect(self) -> float:
        r"""相对偏差（除以初始质量与累计边界通量之和的尺度，避免"除以 0 得 0"）。"""
        scale = abs(float(self.mass_history[0])) if self.mass_history.size else 0.0
        scale += float(np.max(np.abs(self.outflow_history))) if self.outflow_history.size else 0.0
        scale += float(np.max(np.abs(self.inflow_history))) if self.inflow_history.size else 0.0
        return self.max_defect / scale if scale > 0.0 else self.max_defect

    @property
    def entropy_production(self) -> float:
        r"""累计熵产 :math:`\int_0^t\dot E_\eta+\Delta Q\,dt`（熵解应当 :math:`\le 0`）。"""
        if self.entropy_budget.size == 0:
            return 0.0
        return float(self.entropy_budget[-1])


def solve(
    *,
    n: int = 200,
    length: float = 1.0,
    t_end: float = 0.2,
    cfl: float = 0.4,
    kind: str = "burgers",
    scheme: str = "godunov",
    limiter: str = "none",
    ic: str = "shock",
    bc: str = "outflow",
    amplitude: float = 0.5,
    width: float = 0.08,
    u_left: float = 1.0,
    u_right: float = 0.0,
    center: float = 0.5,
    keep_snapshots: bool = False,
    max_steps: int = 200000,
) -> FVMResult:
    r"""推进到 ``t_end``，逐步记录三本账（质量 / 边界通量 / 熵收支）。

    ``dt`` 由 CFL 自适应取（:func:`cfl_dt`），末步被截断到正好落在 ``t_end`` 上；
    ``result.t`` 是**实际到达时刻**（与 ``t_end`` 一致，除非波速为 0 而无法推进）。

    ``scheme="advective"`` 走非守恒支路：那时界面"通量"不再望远镜求和，
    账本用**显式边界通量** :math:`f(u_{\text{edge}})` 记账——于是同一套记账规则
    能同时量出"守恒格式漏不漏"和"非守恒格式漏多少"。
    """
    dx = float(length) / int(n)
    x = cell_centers(n, dx)
    u = initial_data(ic, x, amplitude=amplitude, width=width,
                     u_left=u_left, u_right=u_right, center=center)
    key = str(scheme).strip().lower()
    if key not in SCHEMES:
        raise ValueError(f"未知格式 {scheme!r}（可用：{list(SCHEMES)}）")
    lim = str(limiter).strip().lower()
    if lim != "none" and key == "advective":
        raise ValueError("非守恒的 advective 支路没有 MUSCL 版本（斜率重构只对守恒形式有意义）")
    conservative = key != "advective"
    times = [0.0]
    mass = [total_mass(u, dx)]
    inflow = [0.0]
    outflow = [0.0]
    defect = [0.0]
    ent = [float(np.sum(entropy(u))) * dx]
    ebudget = [0.0]
    tv = [lm.total_variation(u, periodic=(bc == "periodic"))]
    hi = [float(np.max(u))]
    lo = [float(np.min(u))]
    snaps: list[np.ndarray] = [u.copy()] if keep_snapshots else []
    snap_ts: list[float] = [0.0] if keep_snapshots else []
    t = 0.0
    dt_last = 0.0
    steps = 0
    flux = np.zeros(n + 1, dtype=float)
    while t < float(t_end) - 1.0e-15 and steps < int(max_steps):
        dt = cfl_dt(u, dx, cfl, kind)
        if not math.isfinite(dt):
            break
        dt = min(dt, float(t_end) - t)
        if conservative:
            if lim == "none":
                u, flux = fv_step(u, dt, dx, kind=kind, scheme=key, bc=bc)
            else:
                u, flux = muscl_hancock_step(u, dt, dx, kind=kind, scheme=key, limiter=lim, bc=bc)
            f_in = float(flux[0])
            f_out = float(flux[n])
        else:
            u, flux = advective_step(u, dt, dx, kind=kind, bc=bc)
            edge_l, edge_r = _boundary_values(u, bc)
            f_in = float(rr.scalar_flux(kind, edge_l))
            f_out = float(rr.scalar_flux(kind, edge_r))
        t += dt
        dt_last = dt
        steps += 1
        inflow.append(inflow[-1] + dt * f_in)
        outflow.append(outflow[-1] + dt * f_out)
        mass.append(total_mass(u, dx))
        defect.append(mass[-1] - mass[0] + (outflow[-1] - inflow[-1]))
        ent.append(float(np.sum(entropy(u))) * dx)
        q_in = float(np.sum(entropy_flux(_boundary_values(u, bc)[0], kind)))
        q_out = float(np.sum(entropy_flux(_boundary_values(u, bc)[1], kind)))
        eb = ebudget[-1] + (ent[-1] - ent[-2]) + dt * (q_out - q_in)
        ebudget.append(eb)
        tv.append(lm.total_variation(u, periodic=(bc == "periodic")))
        hi.append(float(np.max(u)))
        lo.append(float(np.min(u)))
        if keep_snapshots:
            snaps.append(u.copy())
            snap_ts.append(t)
        times.append(t)
    return FVMResult(
        x=x, u=u, dx=dx, t=t, dt=dt_last, n_steps=steps, kind=kind, scheme=key,
        limiter=lim, ic=ic, bc=bc, cfl=float(cfl),
        time_history=np.asarray(times, dtype=float),
        mass_history=np.asarray(mass, dtype=float),
        defect_history=np.asarray(defect, dtype=float),
        inflow_history=np.asarray(inflow, dtype=float),
        outflow_history=np.asarray(outflow, dtype=float),
        entropy_history=np.asarray(ent, dtype=float),
        entropy_budget=np.asarray(ebudget, dtype=float),
        tv_history=np.asarray(tv, dtype=float),
        max_history=np.asarray(hi, dtype=float),
        min_history=np.asarray(lo, dtype=float),
        flux=np.asarray(flux, dtype=float),
        snapshots=np.asarray(snaps, dtype=float) if snaps else np.zeros((0, 0)),
        snapshot_times=np.asarray(snap_ts, dtype=float),
    )


# --------------------------------------------------------------------------- #
# Euler 方程组的有限体积（复用同一套界面/限制器逻辑）
# --------------------------------------------------------------------------- #
@dataclass
class EulerResult:
    """Euler 方程组的有限体积结果（三本账同样齐全）。"""

    x: np.ndarray
    U: np.ndarray
    dx: float
    t: float
    dt: float
    n_steps: int
    scheme: str
    limiter: str
    bc: str
    cfl: float
    mass_history: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mass_defect: np.ndarray = field(default_factory=lambda: np.zeros(0))
    min_rho_history: np.ndarray = field(default_factory=lambda: np.zeros(0))
    min_p_history: np.ndarray = field(default_factory=lambda: np.zeros(0))
    max_history: np.ndarray = field(default_factory=lambda: np.zeros(0))
    time_history: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def rho(self) -> np.ndarray:
        return rr.euler_primitives(self.U)[0]

    @property
    def velocity(self) -> np.ndarray:
        return rr.euler_primitives(self.U)[1]

    @property
    def pressure(self) -> np.ndarray:
        return rr.euler_primitives(self.U)[2]


def _limited_slopes_system(U: np.ndarray, limiter: str, bc: str) -> np.ndarray:
    r"""逐分量限制斜率（Euler 侧与标量侧共用同一份限制器清单）。"""
    sigma = np.empty_like(U)
    for c in range(U.shape[1]):
        sigma[:, c] = lm.limited_slope(U[:, c], limiter=limiter, periodic=(bc == "periodic"))
    return sigma


def _euler_interface_states(U: np.ndarray, bc: str) -> tuple[np.ndarray, np.ndarray]:
    r"""一阶界面状态（长度 ``n+1``）。"""
    n = U.shape[0]
    left = np.empty((n + 1, 3), dtype=float)
    right = np.empty((n + 1, 3), dtype=float)
    left[1:] = U
    right[:-1] = U
    if bc == "periodic":
        # 界面 0 与界面 n 是同一条缝：两侧状态必须是 (U[n-1], U[0])。
        left[0] = left[n] = U[-1]
        right[0] = right[n] = U[0]
    else:
        left[0] = right[0] = U[0]
        left[n] = right[n] = U[-1]
    return left, right


def euler_fv_step(
    U: Any,
    dt: float,
    dx: float,
    *,
    scheme: str = "hllc",
    limiter: str = "none",
    bc: str = "outflow",
    gamma: float = rr.GAMMA,
) -> np.ndarray:
    r"""Euler 方程组推进一步（``limiter != none`` 时走 MUSCL--Hancock 二阶版本）。

    重构作用在**守恒变量**上（分量逐点限制）；预测步用物理通量
    :func:`numcore.pde.riemann.euler_flux`，界面通量走 HLLC / Rusanov / 精确 Godunov。
    守恒性同样来自望远镜求和：:math:`\rho` 的总量只受边界通量影响。
    """
    arr = np.asarray(U, dtype=float)
    n = arr.shape[0]
    lam = float(dt) / float(dx)
    lim = str(limiter).strip().lower()
    if lim == "none":
        left, right = _euler_interface_states(arr, bc)
    else:
        sigma = _limited_slopes_system(arr, lim, bc)
        pred = arr - 0.5 * lam * (
            np.asarray(rr.euler_flux(arr + 0.5 * sigma, gamma), dtype=float)
            - np.asarray(rr.euler_flux(arr - 0.5 * sigma, gamma), dtype=float)
        )
        left = np.empty((n + 1, 3), dtype=float)
        right = np.empty((n + 1, 3), dtype=float)
        left[1:] = pred + 0.5 * sigma
        right[:-1] = pred - 0.5 * sigma
        if bc == "periodic":
            left[0] = float(left[n])
            right[n] = float(right[0])
        else:
            left[0] = right[0] = arr[0]
            left[n] = right[n] = arr[-1]
    f = np.asarray(rr.euler_numerical_flux(left, right, scheme, gamma), dtype=float)
    return arr - lam * (f[1:] - f[:-1])


def euler_solve(
    w_left: Sequence[float],
    w_right: Sequence[float],
    *,
    n: int = 400,
    length: float = 1.0,
    t_end: float = 0.2,
    cfl: float = 0.4,
    scheme: str = "hllc",
    limiter: str = "minmod",
    bc: str = "outflow",
    gamma: float = rr.GAMMA,
    max_steps: int = 200000,
) -> EulerResult:
    r"""Euler Riemann 问题的有限体积推进（正的密度/压强是**硬不变量**，逐步记账）。"""
    dx = float(length) / int(n)
    x = cell_centers(n, dx)
    U = np.repeat(np.asarray(rr.euler_state(*w_left, gamma), dtype=float)[None, :], n, axis=0).copy()
    U[n // 2:] = np.asarray(rr.euler_state(*w_right, gamma), dtype=float)
    mass = [float(np.sum(U[:, 0])) * dx]
    defect = [0.0]
    min_rho = [float(np.min(U[:, 0]))]
    min_p = [float(np.min(rr.euler_primitives(U, gamma)[2]))]
    hi = [float(np.max(np.abs(U)))]
    times = [0.0]
    t = 0.0
    dt_last = 0.0
    steps = 0
    while t < float(t_end) - 1.0e-15 and steps < int(max_steps):
        rho, vel, pres = rr.euler_primitives(U, gamma)
        a = float(np.max(np.abs(vel) + rr.sound_speed(rho, pres, gamma)))
        if a <= 0.0:
            break
        dt = min(cfl * dx / a, float(t_end) - t)
        U = euler_fv_step(U, dt, dx, scheme=scheme, limiter=limiter, bc=bc, gamma=gamma)
        t += dt
        dt_last = dt
        steps += 1
        times.append(t)
        mass.append(float(np.sum(U[:, 0])) * dx)
        defect.append(mass[-1] - mass[0])
        rho, _, pres = rr.euler_primitives(U, gamma)
        min_rho.append(float(np.min(rho)))
        min_p.append(float(np.min(pres)))
        hi.append(float(np.max(np.abs(U))))
    return EulerResult(
        x=x, U=U, dx=dx, t=t, dt=dt_last, n_steps=steps, scheme=str(scheme),
        limiter=str(limiter), bc=bc, cfl=float(cfl),
        mass_history=np.asarray(mass, dtype=float),
        mass_defect=np.asarray(defect, dtype=float),
        min_rho_history=np.asarray(min_rho, dtype=float),
        min_p_history=np.asarray(min_p, dtype=float),
        max_history=np.asarray(hi, dtype=float),
        time_history=np.asarray(times, dtype=float),
    )


# --------------------------------------------------------------------------- #
# 精确解：Riemann 问题与光滑破散
# --------------------------------------------------------------------------- #
def exact_riemann_solution(
    x: Any,
    t: float,
    *,
    kind: str = "burgers",
    u_left: float = 1.0,
    u_right: float = 0.0,
    center: float = 0.5,
    t_floor: float = 1.0e-6,
) -> dict[str, Any]:
    r"""把标量 Riemann 精确解撒到坐标 ``x`` 上（``t<=0`` 时抬到 ``t_floor`` 并置掩码）。

    抬升而不是报错/返回 NaN：帧里"这一刻没有自相似解"必须是一条可显示的说明
    （``t_clamped`` + ``valid_mask``），不是白屏。
    """
    t_in = float(t)
    clamped = bool(t_in <= 0.0)
    tt = max(t_in, float(t_floor))
    xi = (np.asarray(x, dtype=float) - float(center)) / tt
    u = np.asarray(rr.scalar_sample(kind, xi, u_left, u_right), dtype=float)
    ri = rr.scalar_riemann(kind, u_left, u_right)
    return {
        "x": np.asarray(x, dtype=float),
        "u": u,
        "xi": xi,
        "t": tt,
        "t_clamped": clamped,
        "valid_mask": np.zeros(np.asarray(x).shape, dtype=float) if clamped
        else np.ones(np.asarray(x).shape, dtype=float),
        "riemann": ri,
    }


def smooth_profile(
    key: str, x: Any, *, amplitude: float = 0.5, width: float = 0.08, center: float = 0.5
) -> np.ndarray:
    r"""光滑初值 :math:`u_0(x)`（``sine`` 周期、``gaussian`` 单峰，解析可导）。"""
    xs = np.asarray(x, dtype=float)
    if key == "sine":
        return float(amplitude) * np.sin(2.0 * np.pi * xs)
    if key == "gaussian":
        return float(amplitude) * np.exp(-(((xs - float(center)) / float(width)) ** 2))
    raise ValueError(f"没有解析光滑剖面 {key!r}（可用：sine / gaussian）")


def profile_derivative(
    key: str, x: Any, *, amplitude: float = 0.5, width: float = 0.08, center: float = 0.5
) -> np.ndarray:
    r""":math:`u_0'(x)` 的解析式（破散时间的判据 :math:`\min u_0'` 用它）。"""
    xs = np.asarray(x, dtype=float)
    if key == "sine":
        return float(amplitude) * 2.0 * np.pi * np.cos(2.0 * np.pi * xs)
    if key == "gaussian":
        z = (xs - float(center)) / float(width)
        return -2.0 * float(amplitude) * z / float(width) * np.exp(-z * z)
    raise ValueError(f"没有解析导数 {key!r}（可用：sine / gaussian）")


def breaking_time_numeric(
    key: str, *, amplitude: float = 0.5, width: float = 0.08, n_scan: int = 20001
) -> float:
    r"""破散时间 :math:`t_b=-1/\min_x u_0'(x)`——**扫描**出来的，不是背下来的。

    扫描点数一并返回在调用方的 ``detail`` 里：这是"数值量出来的理论值"，
    与 ``sine`` 的闭式 :math:`1/(2\pi A)` 的差就是扫描误差（本章断言实测两者
    相对差在 :math:`10^{-5}` 量级——**够用来对拍，不够用来当"精确值"**）。
    """
    xs = np.linspace(0.0, 1.0, int(n_scan))
    d = np.asarray(profile_derivative(key, xs, amplitude=amplitude, width=width), dtype=float)
    dmin = float(np.min(d))
    if dmin >= 0.0:
        return float("inf")
    return float(-1.0 / dmin)


def breaking_time(key: str, *, amplitude: float = 0.5, width: float = 0.08) -> float:
    r"""破散时间的**闭式**（``sine`` 与 ``gaussian`` 都有）。

    * ``sine``：:math:`t_b=1/(2\pi A)`；
    * ``gaussian``：:math:`u_0'=-2Az/w\,e^{-z^2}` 在 :math:`z=1/\sqrt2` 取最大模
      :math:`\sqrt{2/e}\,A/w`，所以 :math:`t_b=w e^{1/2}/(A\sqrt2)`。
    """
    if key == "sine":
        if float(amplitude) <= 0.0:
            return float("inf")
        return float(1.0 / (2.0 * np.pi * float(amplitude)))
    if key == "gaussian":
        if float(amplitude) <= 0.0:
            return float("inf")
        return float(float(width) * math.exp(0.5) / (float(amplitude) * math.sqrt(2.0)))
    raise ValueError(f"{key!r} 没有闭式破散时间（请用 breaking_time_numeric）")


def characteristic_lines(
    key: str,
    *,
    amplitude: float = 0.5,
    width: float = 0.08,
    t_end: float = 0.4,
    n_lines: int = 40,
    center: float = 0.5,
) -> dict[str, np.ndarray]:
    r"""特征线 :math:`x(t)=x_0+t\,u_0(x_0)` 与**相邻线的交点**（相交即激波、发散即稀疏波）。

    Burgers 的特征速度就是 :math:`u` 本身，所以相邻两条线
    :math:`i,i+1`（:math:`u_i>u_{i+1}`）在
    :math:`t_*=\dfrac{x_{i+1}-x_i}{u_i-u_{i+1}}` 相交——这就是**激波形成时刻**；
    最小的那个 :math:`t_*` 是全局破散时间。:math:`u_i<u_{i+1}` 时两线发散
    （稀疏波），返回的 ``t_cross`` 是 ``inf``（用有限大数会让"取最小值"变成假证据）。
    """
    x0 = (np.arange(int(n_lines), dtype=float) + 0.5) / float(n_lines)
    u0 = np.asarray(smooth_profile(key, x0, amplitude=amplitude, width=width, center=center), dtype=float)
    x_end = x0 + float(t_end) * u0
    du = u0[:-1] - u0[1:]
    dxg = x0[1:] - x0[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_cross = np.where(du > 0.0, dxg / np.where(du > 0.0, du, 1.0), np.inf)
    x_cross = x0[:-1] + t_cross * u0[:-1]
    finite = np.isfinite(t_cross)
    first = int(np.argmin(np.where(finite, t_cross, np.inf))) if bool(np.any(finite)) else -1
    return {
        "x0": x0,
        "u0": u0,
        "x_end": x_end,
        "t_cross": t_cross,
        "x_cross": np.where(finite, x_cross, 0.0),
        "cross_mask": finite.astype(float),
        "first_pair": np.asarray(first, dtype=float),
        "t_break": float(t_cross[first]) if first >= 0 else float("inf"),
        "x_break": float(x_cross[first]) if first >= 0 else float("nan"),
    }


def exact_smooth_burgers(
    x: Any,
    t: float,
    key: str = "sine",
    *,
    amplitude: float = 0.5,
    width: float = 0.08,
    n_bracket: int = 400,
    n_bisect: int = 80,
) -> dict[str, Any]:
    r"""光滑初值下的 Burgers 精确解：沿特征线回溯 :math:`x_0+t\,u_0(x_0)=x`。

    **只在破散之前有效**：:math:`t<t_b` 时 :math:`g(x_0)=x_0+t u_0(x_0)` 严格单调，
    回溯唯一；:math:`t>t_b` 时三条特征线折到同一点上，解多值——
    函数照样返回数值（"最靠左的那一支"），但把 ``valid_mask`` 置 0、
    ``t_invalid=True``。**这是本章最容易写成假证据的地方**：破散之后
    拿这个"精确解"去算误差，误差会随 :math:`\Delta x` 收敛到 0 得漂漂亮亮，
    而它对照的根本不是熵解。

    回溯用括号 + 二分（``n_bracket`` 个候选点里找符号变化，然后 ``n_bisect`` 步二分），
    两端点数都写进返回字典——"精确解"的精度也是量出来的。
    """
    xs = np.asarray(x, dtype=float)
    t_in = float(t)
    tb = breaking_time(key, amplitude=amplitude, width=width)
    valid = bool(t_in < tb)
    tt = max(t_in, 0.0)
    u0_flat = np.zeros(xs.size, dtype=float)
    xi_out = np.zeros(xs.size, dtype=float)
    for i, xv in enumerate(xs.ravel()):
        s = np.linspace(xv - 1.0, xv + 1.0, int(n_bracket))
        h = s + tt * np.asarray(
            smooth_profile(key, np.mod(s, 1.0), amplitude=amplitude, width=width), dtype=float
        ) - xv
        idx = np.nonzero((h[:-1] <= 0.0) & (h[1:] >= 0.0))[0]
        if idx.size == 0:
            u0_flat[i] = float("nan")
            xi_out[i] = float("nan")
            continue
        a = float(s[int(idx[0])])
        b = float(s[int(idx[0]) + 1])
        for _ in range(int(n_bisect)):
            m = 0.5 * (a + b)
            hm = m + tt * float(smooth_profile(key, np.mod(m, 1.0), amplitude=amplitude, width=width)) - xv
            if hm > 0.0:
                b = m
            else:
                a = m
        root = 0.5 * (a + b)
        xi_out[i] = root
        u0_flat[i] = float(smooth_profile(key, np.mod(root, 1.0), amplitude=amplitude, width=width))
    mask = np.ones(xs.shape, dtype=float) if valid else np.zeros(xs.shape, dtype=float)
    bad = ~np.isfinite(u0_flat)
    mask[bad.reshape(xs.shape)] = 0.0
    return {
        "x": xs,
        "u": u0_flat.reshape(xs.shape),
        "x0": xi_out.reshape(xs.shape),
        "t": tt,
        "t_break": tb,
        "t_invalid": (not valid) and t_in > 0.0,
        "valid_mask": mask,
        "n_bracket": int(n_bracket),
        "n_bisect": int(n_bisect),
    }


def l1_error(u: Any, exact: Any, dx: float) -> float:
    r"""离散 :math:`L^1` 误差 :math:`\sum_j|u_j-u^{\text{exact}}_j|\Delta x`。"""
    a = np.asarray(u, dtype=float)
    b = np.asarray(exact, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"形状不一致：{a.shape} vs {b.shape}")
    return float(np.sum(np.abs(a - b)) * float(dx))


def fit_speed(times: Any, positions: Any) -> dict[str, float]:
    r"""最小二乘拟合激波位置-时间直线的斜率（**实测激波速度**）。

    只用有限的点；点数与窗口写进返回值——"激波速度"离开时间窗口没有意义
    （激波形成之前、波绕过边界之后，直线拟合的斜率都不是 RH 值）。
    """
    ts = np.asarray(times, dtype=float)
    xs = np.asarray(positions, dtype=float)
    ok = np.isfinite(ts) & np.isfinite(xs)
    ts = ts[ok]
    xs = xs[ok]
    m = ts.size
    if m < 2:
        return {"speed": float("nan"), "intercept": float("nan"), "n_used": float(m), "r2": float("nan")}
    tbar = float(np.mean(ts))
    xbar = float(np.mean(xs))
    sxx = float(np.sum((ts - tbar) ** 2))
    if sxx <= 0.0:
        return {"speed": float("nan"), "intercept": float("nan"), "n_used": float(m), "r2": float("nan")}
    slope = float(np.sum((ts - tbar) * (xs - xbar)) / sxx)
    inter = float(xbar - slope * tbar)
    resid = xs - (inter + slope * ts)
    sst = float(np.sum((xs - xbar) ** 2))
    r2 = 1.0 - float(np.sum(resid * resid)) / sst if sst > 0.0 else 1.0
    return {"speed": slope, "intercept": inter, "n_used": float(m), "r2": r2}


def shock_front(
    u: Any, *, level: float = 0.5, dx: float = 1.0, periodic: bool = False
) -> float:
    r"""激波位置：解**自左向右**第一次从 ``level`` 之上下穿到之下的位置（线性插值到交点）。

    为什么 4.3 章自己带一个：4.2 章的 :func:`numcore.pde.advection.burgers_shock_position`
    在 ``periodic=False`` 分支上取 ``arr[lo-1:n-1]``（``lo=0``）得到**空切片**，
    随即在广播处抛 ``ValueError``——本章的主场景恰好全是非周期出流边界。
    这里按同一套约定重写一遍：返回 ``(j-1+frac)*dx``、**两个分支都跳过首尾接缝那一段**
    （周期域上的接缝不是物理激波），因此 ``periodic`` 只作为接口兼容保留。
    与 4.2 的周期分支逐位一致（有断言对拍），**不去改别人的文件**。

    没有跨越点时返回 ``nan``（调用方必须自己处理——不要把它当 0）。
    """
    arr = np.asarray(u, dtype=float).ravel()
    n = arr.size
    lv = float(level)
    if n < 2:
        return float("nan")
    a = arr[0 : n - 1]
    b = arr[1:n]
    cross = np.nonzero((a > lv) & (b <= lv))[0]
    if cross.size == 0:
        return float("nan")
    j = int(cross[0]) + 1
    ua = float(arr[j - 1])
    ub = float(arr[j])
    if ua == ub:
        return float(j) * float(dx)
    frac = (ua - lv) / (ua - ub)
    return float(j - 1 + frac) * float(dx)


def shock_track(
    u_history: Any,
    dx: float,
    *,
    level: float = 0.5,
    times: Any | None = None,
    periodic: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""从快照序列里逐帧找激波位置，返回 ``(times, positions, finite_mask)``。

    找不到跨越的帧位置是 ``nan`` 并由掩码标出——**不要把 nan 换成 0**
    （那会让"激波还没形成"看起来像"激波在原点"，实测在稀疏波算例上会给出
    一条漂亮但完全假的零速度直线）。
    """
    snaps = np.asarray(u_history, dtype=float)
    if snaps.ndim != 2:
        raise ValueError(f"快照必须是二维 (n_frames, n)，得到 {snaps.shape}")
    ts = np.arange(snaps.shape[0], dtype=float) if times is None else np.asarray(times, dtype=float)
    pos = np.asarray(
        [shock_front(row, level=level, dx=dx, periodic=periodic) for row in snaps], dtype=float
    )
    return ts, pos, np.isfinite(pos).astype(float)
