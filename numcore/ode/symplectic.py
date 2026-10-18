r"""3.3 章（几何数值积分与长期行为）的方法族与哈密顿系统真源。

数学对象
--------
**可分哈密顿系统**

.. math::
    H(q,p)=T(p)+V(q),\qquad T(p)=\tfrac12|p|^2,\qquad f(q)=-\nabla V(q),
    \qquad q,p\in\mathbb R^d

（本章取单位质量）。**辛积分器**是满足

.. math::
    (\mathrm D\Phi_h)^{\mathsf T}J\,(\mathrm D\Phi_h)=J,\qquad
    J=\begin{pmatrix}0&I\\-I&0\end{pmatrix}

的映射 :math:`\Phi_h`：它保持相面积 :math:`\omega=\sum_i\mathrm dq_i\wedge\mathrm dp_i`。
等价的局部判据是"每一步的线性化映射行列式为 1"，本章用它给非辛方法做反例。

本章四个辛方法都由 kick / drift 的复合给出（:math:`f=-\nabla V`）：

.. math::
    \mathrm{kick}(\tau):\;p\leftarrow p+\tau f(q),\qquad
    \mathrm{drift}(\tau):\;q\leftarrow q+\tau p .

* ``symplectic_euler`` :math:`=\mathrm{drift}(h)\circ\mathrm{kick}(h)`——一阶、辛、**不对称**；
* ``verlet`` :math:`=\mathrm{kick}(h/2)\circ\mathrm{drift}(h)\circ\mathrm{kick}(h/2)`
  （速度 Verlet / Störmer--Verlet）——二阶、**对称**、辛；
* ``leapfrog``：交错速度记账（状态里存 :math:`v_{n-1/2}`，采样时同步到整数时刻）。
  在本章的单位质量可分系统上它与 ``verlet`` **是同一个映射**，实现里逐位相同——
  这一条由断言实测，而不是由"名字不同所以方法不同"的直觉担保；
* ``implicit_midpoint``：隐式、二阶、对称、辛；解 :math:`z=q+\tfrac h2p+\tfrac{h^2}4f(z)`
  用定点迭代（收敛因子 :math:`\approx h^2\omega^2/4`，本章帧里如实报出迭代次数）。

非辛对照**不重写**：``rk4`` 与 ``explicit_euler`` 直接借 3.1 章的
:func:`numcore.ode.rk.rk_step`（同一张 Butcher 表、同一份实现、同一个
:math:`1/\gamma` 口径）——这正是"复用但不重复实现"的落点。

为什么要专门写一条长时间快路径
------------------------------
10^6 步的长期积分是本章的招牌。若用 numpy 的 1 元素数组逐步推进，每步要付若干次
numpy 调用（实测 ~2–4 μs/步 ⟹ 10^6 步要好几秒）；把状态放在**裸 float** 上、
每个方法展开成一个闭包，实测 ~0.2 μs/步 ⟹ 10^6 步 0.2 s。
两条路径算的是**同一个映射**（``tests/3.3`` 里有逐位对拍断言），
dof ≥ 3 没有快路径，一律走通用实现——不存在"快路径悄悄换了个方法"的缝。

爆破解不撒谎（与 3.1/3.2 同款约定）
-----------------------------------
:math:`h` 越过稳定边界后 :math:`|y|` 会指数增长，10^6 步必然溢出成 ``inf``——
而 ``inf`` 进不了帧（``Frame.validate()`` 当场拒绝）。所以积分在
:math:`\max|y|>` :data:`BLOWUP` 时**显式停止**，返回 ``ok=False``/``blowup=True``/
``t_blowup``/``i_blowup``（停止发生在采样序列的第几个点），调用方据此写
"占位值 + 掩码 + 文案"三件套，而不是让非有限值流到前端。

自研边界（P2/P3）
-----------------
只做逐元素运算、手算 2×2 行列式与 3 项以内的闭式解；不解线性系统、不碰任何
numpy 成品算法。幂律斜率复用 3.1 章的 :func:`numcore.ode.rk.loglog_slope`（同一口径）。
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from numcore.ode.rk import loglog_slope, rk_step

__all__ = [
    "ForceFn",
    "System",
    "MethodSpec",
    "Run",
    "StepInfo",
    "BLOWUP",
    "METHODS",
    "SYMPLECTIC_METHODS",
    "NONSYMPLECTIC_METHODS",
    "SYSTEM_NAMES",
    "method_spec",
    "method_label",
    "is_symplectic",
    "system",
    "systems",
    "sample_indices",
    "step",
    "step_detailed",
    "integrate",
    "linear_map",
    "det_2x2",
    "harmonic_energy",
    "agm",
    "pendulum_period",
    "pendulum_system",
    "kepler_energy",
    "kepler_angular_momentum",
    "kepler_flow",
    "kepler_orbit_curve",
    "orbital_elements",
    "lj_potential",
    "loglog_slope",
]

#: 右端（力）函数：``f(q) -> -grad V(q)``，numpy 通用路径的签名。
ForceFn = Callable[[np.ndarray], np.ndarray]

#: 未爆炸的显示上限。超过它一律停止积分并如实标注（``inf`` 进不了帧）。
BLOWUP = 1.0e12


# --------------------------------------------------------------------------- #
# 1. 方法表
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MethodSpec:
    r"""一个积分方法的**分类学**：阶、是否辛、是否对称、是否隐式、每步力求值次数。

    ``symplectic`` 与 ``symmetric`` 是两件不同的事：``symplectic_euler`` 是辛的但
    **不对称**（交换 kick 与 drift 的顺序得到的是另一个方法），而隐式中点是既辛又对称的。
    本章的"能量无长期漂移"用的是辛性；"修正哈密顿量是 :math:`h` 的偶次幂级数"用的是对称性——
    两件事必须分开测。
    """

    name: str
    label: str
    order: int
    symplectic: bool
    symmetric: bool
    implicit: bool
    n_force: int
    note: str


_METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(
        name="symplectic_euler",
        label="辛 Euler",
        order=1,
        symplectic=True,
        symmetric=False,
        implicit=False,
        n_force=1,
        note=(
            "drift(h)∘kick(h)：最省的一次求值辛方法。能量误差是 O(h) 的**有界振荡**——"
            "它不是'更好的 Euler'，是'另一种错法'：误差不小，但不随时间长大。"
        ),
    ),
    MethodSpec(
        name="verlet",
        label="速度 Verlet（Störmer–Verlet）",
        order=2,
        symplectic=True,
        symmetric=True,
        implicit=False,
        n_force=2,
        note=(
            "kick(h/2)∘drift(h)∘kick(h/2)：二阶、对称、辛。对称性使修正哈密顿量只含 h 的偶次幂，"
            "能量误差带宽是 O(h²)（比辛 Euler 低一阶）。"
        ),
    ),
    MethodSpec(
        name="leapfrog",
        label="跳蛙（交错速度，与 verlet 同一映射）",
        order=2,
        symplectic=True,
        symmetric=True,
        implicit=False,
        n_force=2,
        note=(
            "状态里存 v_{n−1/2}，采样时同步到整数时刻。在可分系统上它与 verlet 是同一个映射，"
            "本章的实现逐位相同（断言实测差 0）——名字是两个，方法是一个。"
        ),
    ),
    MethodSpec(
        name="implicit_midpoint",
        label="隐式中点",
        order=2,
        symplectic=True,
        symmetric=True,
        implicit=True,
        n_force=1,
        note=(
            "二阶、对称、辛，且 A-稳定。线性问题上它与梯形法重合，|R(iy)|≡1："
            "谐振子上的能量误差只有舍入级（带宽 0），这是'辛'的另一种极端。"
        ),
    ),
    MethodSpec(
        name="rk4",
        label="经典 RK4（非辛对照，借自 3.1 章）",
        order=4,
        symplectic=False,
        symmetric=False,
        implicit=False,
        n_force=4,
        note=(
            "短期精度最高（局部 O(h⁵)），但每一步的线性化行列式 = |R|² = 1 − h⁶/72 + … ≠ 1："
            "谐振子轨道**向内**螺旋（数值耗散）。"
        ),
    ),
    MethodSpec(
        name="explicit_euler",
        label="显式 Euler（非辛对照，借自 3.1 章）",
        order=1,
        symplectic=False,
        symmetric=False,
        implicit=False,
        n_force=1,
        note="线性化行列式 = 1 + h² > 1：谐振子轨道**向外**螺旋，且无稳定步长上限。",
    ),
)

#: 全部方法名（辛方法在前，非辛对照在后）。
METHODS: tuple[str, ...] = tuple(m.name for m in _METHODS)

#: 辛方法名。
SYMPLECTIC_METHODS: tuple[str, ...] = tuple(m.name for m in _METHODS if m.symplectic)

#: 非辛对照方法名。
NONSYMPLECTIC_METHODS: tuple[str, ...] = tuple(m.name for m in _METHODS if not m.symplectic)

#: 名字 → 3.1 章 Butcher 表的名字（非辛对照走 3.1 的通用实现）。
_RK_ALIAS: dict[str, str] = {"rk4": "rk4", "explicit_euler": "euler"}


def method_spec(name: str) -> MethodSpec:
    """按名字取方法规格；不认识就抛 :class:`ValueError`（不返回默认值）。"""
    for m in _METHODS:
        if m.name == name:
            return m
    raise ValueError(f"未知方法 {name!r}（可用：{list(METHODS)}）")


def method_label(name: str) -> str:
    """方法的中文显示名。"""
    return method_spec(name).label


def is_symplectic(name: str) -> bool:
    """该方法是否辛（本章的机器判据，供 manifest/帧/断言共用一份口径）。"""
    return method_spec(name).symplectic


# --------------------------------------------------------------------------- #
# 2. 哈密顿系统
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class System:
    r"""一个可分哈密顿系统 :math:`H=T+V` 与它的全部真值。

    ``force_1d`` / ``force_2d`` 是**快路径**用的标量力（签名分别是
    ``f(q)->float`` 与 ``f(q1,q2)->(f1,f2)``），``None`` 表示该系统只能走 numpy 通用路径。
    ``exact_curve`` 给出**精确不变量曲线**（谐振子的圆、Kepler 的椭圆）——
    相空间图上的参考线必须是它，而不是"再算一遍数值解"。
    """

    name: str
    label: str
    dof: int
    q0: tuple[float, ...]
    p0: tuple[float, ...]
    force: ForceFn
    energy: Callable[[np.ndarray, np.ndarray], float]
    period: float | None
    exact_curve: Callable[[int], tuple[np.ndarray, np.ndarray]] | None
    force_1d: Callable[[float], float] | None = None
    force_2d: Callable[[float, float], tuple[float, float]] | None = None
    momentum: Callable[[np.ndarray], float] | None = None
    angular_momentum: Callable[[np.ndarray, np.ndarray], float] | None = None
    central: bool = False
    note: str = ""

    def q0_array(self) -> np.ndarray:
        return np.asarray(self.q0, dtype=float)

    def p0_array(self) -> np.ndarray:
        return np.asarray(self.p0, dtype=float)

    def energy_of(self, q: Any, p: Any) -> float:
        """:math:`H(q,p)`（标量）。"""
        return float(self.energy(np.asarray(q, dtype=float).reshape(-1), np.asarray(p, dtype=float).reshape(-1)))


# ---- 谐振子 ---------------------------------------------------------------- #
def harmonic_energy(q: Any, p: Any) -> float:
    r""":math:`H=\tfrac12(p^2+\omega^2q^2)`（本章 :math:`\omega=1`，单位质量）。"""
    qq = np.asarray(q, dtype=float).reshape(-1)
    pp = np.asarray(p, dtype=float).reshape(-1)
    return float(0.5 * np.sum(pp * pp + qq * qq))


def _harmonic_force(q: np.ndarray) -> np.ndarray:
    return -q


def _harmonic_force_1d(q: float) -> float:
    return -q


def _harmonic_curve(n: int, *, q0: float = 1.0, p0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    r"""精确轨道：相平面上的圆 :math:`q^2+p^2=2H`（半径 :math:`\sqrt{2H}`）。"""
    r = math.sqrt(2.0 * harmonic_energy(np.array([q0]), np.array([p0])))
    th = np.linspace(0.0, 2.0 * math.pi, int(n))
    return r * np.cos(th), r * np.sin(th)


# ---- 单摆 ------------------------------------------------------------------ #
def agm(a: float, b: float, *, tol: float = 1e-16, max_iter: int = 60) -> float:
    r"""算术--几何平均 :math:`\mathrm{AGM}(a,b)`（自写迭代，无第三方库）。

    :math:`a_{k+1}=(a_k+b_k)/2`、:math:`b_{k+1}=\sqrt{a_kb_k}`，二次收敛
    （每步有效位数翻倍，实测 5 步到机器精度）。
    """
    aa, bb = float(a), float(b)
    if aa <= 0.0 or bb <= 0.0:
        raise ValueError(f"AGM 需要正数，得到 {a}, {b}")
    for _ in range(int(max_iter)):
        if abs(aa - bb) <= tol * max(abs(aa), abs(bb)):
            break
        aa, bb = 0.5 * (aa + bb), math.sqrt(aa * bb)
    return 0.5 * (aa + bb)


def pendulum_period(amplitude: float) -> float:
    r"""单摆 :math:`\theta''=-\sin\theta` 从 :math:`\theta_0` 静止释放的**精确周期**

    .. math::
        T=\frac{4}{\sqrt{1}}\,\mathrm{AGM}\bigl(1,\cos(\theta_0/2)\bigr)^{-1}\pi
        =4K(k),\qquad k=\sin(\theta_0/2),

    其中 :math:`K` 用 AGM 算：:math:`K(k)=\pi/\bigl(2\,\mathrm{AGM}(1,\sqrt{1-k^2})\bigr)`。
    小角极限 :math:`T\to2\pi`；本章用它给"非线性系统的精确参考周期"，
    而不是拿数值解自己的过零点当参考（那是循环论证）。
    """
    th0 = abs(float(amplitude))
    if th0 >= math.pi:
        raise ValueError(f"振幅必须在 |θ0| < π 内（否则是旋转而非振动），得到 {amplitude}")
    if th0 == 0.0:
        return 2.0 * math.pi
    k = math.sin(0.5 * th0)
    return 2.0 * math.pi / agm(1.0, math.sqrt(1.0 - k * k))


def pendulum_system(amplitude: float = 2.0) -> System:
    r"""振幅为 ``amplitude``（:math:`\theta_0`，从静止释放）的单摆系统。

    ``period`` 填的是 **AGM 算的椭圆积分精确周期**（:func:`pendulum_period`），
    不是数值解自己的过零点——"实测周期 vs 理论周期"的断言必须分得清谁是参照物。
    振幅是参数的唯一原因：本章的"A 类"断言要验证周期**随振幅增长**
    （小角极限 :math:`2\pi`，:math:`\theta_0=2` 时 8.35）。
    """
    amp = float(amplitude)
    if not 0.0 < amp < math.pi:
        raise ValueError(f"振幅需在 (0, π) 内（否则不是摆动），得到 {amplitude}")
    e0 = 1.0 - math.cos(amp)
    if amp < 1e-6:
        per = 2.0 * math.pi
    else:
        per = pendulum_period(amp)
    return System(
        name=f"pendulum_amp{amp:g}",
        label=f"单摆（θ0={amp:g} rad）",
        dof=1,
        q0=(amp,),
        p0=(0.0,),
        force=_pendulum_force,
        energy=lambda q, p: float(0.5 * p[0] * p[0] + 1.0 - math.cos(float(q[0]))),
        period=per,
        exact_curve=None,
        force_1d=_pendulum_force_1d,
        note=(
            f"θ0={amp:g} rad，H0={e0:.6f}；精确周期 {per:.9f}（AGM 椭圆积分）。"
            "非线性 ⟹ 修正哈密顿量只是渐近级数。"
        ),
    )


def _pendulum_force(q: np.ndarray) -> np.ndarray:
    return -np.sin(q)


def _pendulum_force_1d(q: float) -> float:
    return -math.sin(q)


# ---- Kepler（二维中心力） --------------------------------------------------- #
def kepler_energy(q: Any, p: Any, *, mu: float = 1.0) -> float:
    r""":math:`H=\tfrac12|p|^2-\mu/|q|`。"""
    qq = np.asarray(q, dtype=float).reshape(-1)
    pp = np.asarray(p, dtype=float).reshape(-1)
    r = math.sqrt(float(np.sum(qq * qq)))
    return float(0.5 * np.sum(pp * pp) - mu / r)


def kepler_angular_momentum(q: Any, p: Any) -> float:
    r""":math:`L=q_xp_y-q_yp_x`（二维中心力的运动积分）。"""
    qq = np.asarray(q, dtype=float).reshape(-1)
    pp = np.asarray(p, dtype=float).reshape(-1)
    return float(qq[0] * pp[1] - qq[1] * pp[0])


def _kepler_elements(ecc: float, *, mu: float = 1.0, a: float = 1.0) -> tuple[np.ndarray, np.ndarray, float]:
    r"""近星点出发的椭圆初值 :math:`(q_0,p_0)`：:math:`r_p=a(1-e)`、:math:`v_p=\sqrt{\mu(1+e)/(a(1-e))}`。"""
    if not 0.0 <= ecc < 1.0:
        raise ValueError(f"需要 0 ≤ e < 1（椭圆），得到 {ecc}")
    rp = a * (1.0 - ecc)
    vp = math.sqrt(mu * (1.0 + ecc) / (a * (1.0 - ecc)))
    return np.array([rp, 0.0]), np.array([0.0, vp]), 2.0 * math.pi * math.sqrt(a**3 / mu)


def kepler_flow(t: Any, ecc: float, *, mu: float = 1.0, a: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    r"""Kepler 问题的**精确流**：返回 ``(q(t), p(t))``，:math:`q,p` 形状 ``(m, 2)``。

    路线是标准的三步（自写 Newton 解 Kepler 方程 :math:`E-e\sin E=M`，不用任何成品求根）：

    1. 平近点角 :math:`M=2\pi t/P`、:math:`P=2\pi\sqrt{a^3/\mu}`；
    2. :math:`E` 由 Newton 迭代 :math:`E\leftarrow E-(E-e\sin E-M)/(1-e\cos E)`；
    3. :math:`r=a(1-e\cos E)`、真近点角由
       :math:`\tan(\nu/2)=\sqrt{(1+e)/(1-e)}\tan(E/2)` 给出，
       速度用 :math:`v=(\mu/L)(-\sin\nu,\,e+\cos\nu)`、:math:`L=\sqrt{\mu a(1-e^2)}`。

    为什么值得写：本章所有"漂移"都要与真值比。没有精确流，"非辛方法线性漂移"
    这句话就只剩"两条曲线不一样"。
    """
    tt = np.atleast_1d(np.asarray(t, dtype=float)).reshape(-1)
    per = 2.0 * math.pi * math.sqrt(a**3 / mu)
    m_anom = 2.0 * math.pi * tt / per
    ecc_anom = m_anom + ecc * np.sin(m_anom)  # 初值取一阶修正，e→0.9 也只需几步
    for _ in range(60):
        f = ecc_anom - ecc * np.sin(ecc_anom) - m_anom
        fp = 1.0 - ecc * np.cos(ecc_anom)
        step = f / fp
        ecc_anom = ecc_anom - step
        if float(np.max(np.abs(step))) < 1e-16:
            break
    r = a * (1.0 - ecc * np.cos(ecc_anom))
    # 半角公式 tan(ν/2)=√((1+e)/(1−e))·tan(E/2) 的 atan2 写法（无象限歧义）
    half_e = 0.5 * ecc_anom
    nu = 2.0 * np.arctan2(math.sqrt(1.0 + ecc) * np.sin(half_e), math.sqrt(1.0 - ecc) * np.cos(half_e))
    ell = math.sqrt(mu * a * (1.0 - ecc * ecc))
    qx = r * np.cos(nu)
    qy = r * np.sin(nu)
    vx = (mu / ell) * (-np.sin(nu))
    vy = (mu / ell) * (ecc + np.cos(nu))
    return np.stack([qx, qy], axis=-1), np.stack([vx, vy], axis=-1)


def kepler_orbit_curve(ecc: float, n: int, *, a: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    r"""精确轨道曲线 :math:`r(\nu)=\dfrac{a(1-e^2)}{1+e\cos\nu}`（近星点在 :math:`+x`）。"""
    if not 0.0 <= ecc < 1.0:
        raise ValueError(f"需要 0 ≤ e < 1，得到 {ecc}")
    nu = np.linspace(0.0, 2.0 * math.pi, int(n))
    r = a * (1.0 - ecc * ecc) / (1.0 + ecc * np.cos(nu))
    return r * np.cos(nu), r * np.sin(nu)


def _kepler_make(ecc: float, name: str, label: str) -> System:
    q0, p0, per = _kepler_elements(ecc)

    def force(q: np.ndarray) -> np.ndarray:
        r2 = float(q[0] * q[0] + q[1] * q[1])
        r = math.sqrt(r2)
        s = -1.0 / (r2 * r)
        return s * q

    def force_2d(x: float, y: float) -> tuple[float, float]:
        r2 = x * x + y * y
        r = math.sqrt(r2)
        s = -1.0 / (r2 * r)
        return s * x, s * y

    def energy(q: np.ndarray, p: np.ndarray) -> float:
        return kepler_energy(q, p)

    def curve(n: int) -> tuple[np.ndarray, np.ndarray]:
        x, y = kepler_orbit_curve(ecc, n)
        return x, y

    return System(
        name=name,
        label=label,
        dof=2,
        q0=(float(q0[0]), float(q0[1])),
        p0=(float(p0[0]), float(p0[1])),
        force=force,
        energy=energy,
        period=per,
        exact_curve=curve,
        force_2d=force_2d,
        angular_momentum=kepler_angular_momentum,
        central=True,
        note=(
            f"a=1、μ=1、e={ecc:g}：近星点 r_p={1.0 - ecc:g}、远星点 r_a={1.0 + ecc:g}、"
            f"周期 P={per:.6f}。中心力 ⟹ 角动量 L=√(a(1−e²)) 是运动积分："
            "辛方法的径向 kick 逐位保它，RK4 不保（这是判别辛性最干净的一把尺子）。"
        ),
    )


# ---- Lennard-Jones 二聚体（分子动力学最小模型） ------------------------------ #
def lj_potential(r: Any, *, eps: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    r"""Lennard-Jones 势 :math:`V=4\varepsilon\bigl[(\sigma/r)^{12}-(\sigma/r)^6\bigr]`。"""
    rr = np.asarray(r, dtype=float)
    s = (sigma / rr) ** 2
    return 4.0 * eps * (s**6 - s**3)


def lj_force(r: float, *, eps: float = 1.0, sigma: float = 1.0) -> float:
    r""":math:`-V'(r)=\dfrac{24\varepsilon}{r}\bigl(2(\sigma/r)^{12}-(\sigma/r)^6\bigr)`（吸引为正）。"""
    s = (sigma / r) ** 2
    s3 = s * s * s
    return 24.0 * eps * (2.0 * s3 * s3 - s3) / r


def _lj_dimer() -> System:
    r"""两个等质量原子的一维 LJ 二聚体（质心自由、相对运动非线性）。

    :math:`q=(x_1,x_2)`、:math:`V=V_{LJ}(x_2-x_1)`——这是"分子动力学"能有的最小模型：
    平动不变 ⟹ 总动量逐位守恒（任何 RK 方法都保，**不能**用它证明辛性），
    而能量误差的有界/漂移才是方法的指纹。
    """
    sigma = 1.0
    eps = 1.0
    r_eq = 2.0 ** (1.0 / 6.0) * sigma
    r0 = 1.3 * sigma
    x1 = -0.5 * r0
    x2 = +0.5 * r0

    def force(q: np.ndarray) -> np.ndarray:
        r = float(q[1] - q[0])
        f = lj_force(r, eps=eps, sigma=sigma)
        return np.array([-f, f])

    def force_2d(a: float, b: float) -> tuple[float, float]:
        f = lj_force(b - a, eps=eps, sigma=sigma)
        return -f, f

    def energy(q: np.ndarray, p: np.ndarray) -> float:
        r = float(q[1] - q[0])
        return float(0.5 * (p[0] * p[0] + p[1] * p[1]) + float(np.sum(lj_potential(r, eps=eps, sigma=sigma))))

    def momentum(p: np.ndarray) -> float:
        return float(p[0] + p[1])

    return System(
        name="lj_dimer",
        label="LJ 二聚体（两原子分子，一维）",
        dof=2,
        q0=(x1, x2),
        p0=(0.0, 0.0),
        force=force,
        energy=energy,
        period=None,
        exact_curve=None,
        force_2d=force_2d,
        momentum=momentum,
        note=(
            f"σ=1、ε=1、平衡键长 r_eq=2^(1/6)={r_eq:.6f}；初值 r=1.3（拉伸 15.8%），"
            "两原子从静止释放。非线性力 ⟹ 没有闭式解，但能量与总动量都是精确的运动积分。"
        ),
    )


def _lj_chain(n_atoms: int = 6, amplitude: float = 0.3) -> System:
    r"""一维 LJ 链（``n_atoms`` 个等质量原子、最近邻、自由端）——dof ≥ 3 的通用路径用例。

    初值取平衡键长 + **正弦速度剖面** :math:`p_i=A\sin\bigl(\pi(i+1)/(N+1)\bigr)`
    （一个相干声子模），因此总动量 :math:`\sum_ip_i\ne0` 但**守恒**；
    平动不变性使 :math:`\sum_ip_i` 成为逐位守恒量（非线性力也挡不住这条线性不变量）。
    """
    sigma = 1.0
    eps = 1.0
    r_eq = 2.0 ** (1.0 / 6.0) * sigma
    xs = tuple(float(i) * r_eq for i in range(n_atoms))
    ps = tuple(float(amplitude) * math.sin(math.pi * (i + 1) / (n_atoms + 1)) for i in range(n_atoms))

    def force(q: np.ndarray) -> np.ndarray:
        out = np.zeros_like(q)
        for i in range(n_atoms - 1):
            f = lj_force(float(q[i + 1] - q[i]), eps=eps, sigma=sigma)
            out[i] -= f
            out[i + 1] += f
        return out

    def energy(q: np.ndarray, p: np.ndarray) -> float:
        tot = 0.5 * float(np.sum(p * p))
        for i in range(n_atoms - 1):
            tot += float(np.sum(lj_potential(float(q[i + 1] - q[i]), eps=eps, sigma=sigma)))
        return tot

    def momentum(p: np.ndarray) -> float:
        return float(np.sum(p))

    return System(
        name="lj_chain",
        label=f"一维 LJ 链（N={n_atoms}，自由端）",
        dof=n_atoms,
        q0=xs,
        p0=ps,
        force=force,
        energy=energy,
        period=None,
        exact_curve=None,
        momentum=momentum,
        note=(
            f"N={n_atoms}、最近邻、自由端、初值是平衡键长 + 正弦速度剖面（幅度 {amplitude}）。"
            "dof ≥ 3：它走 numpy 通用路径，是'快路径之外仍然正确'的活证据。"
        ),
    )


_SYSTEMS: tuple[System, ...] = (
    System(
        name="harmonic",
        label="谐振子（线性，H=(q²+p²)/2）",
        dof=1,
        q0=(1.0,),
        p0=(0.0,),
        force=_harmonic_force,
        energy=harmonic_energy,
        period=2.0 * math.pi,
        exact_curve=_harmonic_curve,
        force_1d=_harmonic_force_1d,
        note="唯一有闭式解又有闭式修正能量的系统：本章所有'精确'参照物都来自它。",
    ),
    dataclasses.replace(pendulum_system(2.0), name="pendulum", label="单摆（非线性，θ''=−sin θ，θ0=2）"),
    _kepler_make(0.0, "kepler_circular", "Kepler 圆轨道（e=0）"),
    _kepler_make(0.6, "kepler_ecc_0.6", "Kepler 椭圆（e=0.6）"),
    _kepler_make(0.9, "kepler_ecc_0.9", "Kepler 椭圆（e=0.9，近星点欠分辨）"),
    _lj_dimer(),
    _lj_chain(),
)

#: 全部系统名。
SYSTEM_NAMES: tuple[str, ...] = tuple(s.name for s in _SYSTEMS)


def systems() -> tuple[System, ...]:
    """全部系统（不可变元组）。"""
    return _SYSTEMS


def system(name: str) -> System:
    """按名字取系统；不认识就抛 :class:`ValueError`。"""
    for s in _SYSTEMS:
        if s.name == name:
            return s
    raise ValueError(f"未知系统 {name!r}（可用：{list(SYSTEM_NAMES)}）")


# --------------------------------------------------------------------------- #
# 3. 单步（numpy 通用路径）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StepInfo:
    """一次单步的全部可观测物（隐式方法的迭代信息也在这里）。"""

    q: np.ndarray
    p: np.ndarray
    n_force: int
    iterations: int
    residual: float
    converged: bool

    @property
    def state(self) -> np.ndarray:
        return np.concatenate([self.q, self.p])


def step_detailed(
    method: str,
    sys: System,
    q: Any,
    p: Any,
    h: float,
    *,
    tol: float = 1e-14,
    max_iter: int = 40,
) -> StepInfo:
    r"""走一步，返回 :class:`StepInfo`（:math:`q,p` 用 numpy 数组，任意 dof）。

    隐式方法解 :math:`z=q+\tfrac h2p+\tfrac{h^2}4f(z)`：定点迭代，收敛判据是
    :math:`\|z_{k+1}-z_k\|_\infty\le\mathrm{tol}\cdot\max(1,\|z_k\|_\infty)`；
    迭代发散（残差连续两轮变大）时**提前停止**并置 ``converged=False``——
    这正是 :math:`h` 越过稳定边界后该显示的状态，而不是让它悄悄溢出成 ``inf``。
    """
    if not h > 0.0:
        raise ValueError(f"步长必须为正，得到 {h}")
    spec = method_spec(method)
    qq = np.asarray(q, dtype=float).reshape(-1).copy()
    pp = np.asarray(p, dtype=float).reshape(-1).copy()
    if spec.name == "symplectic_euler":
        pp = pp + h * sys.force(qq)
        qq = qq + h * pp
        return StepInfo(qq, pp, 1, 0, 0.0, True)
    if spec.name in ("verlet", "leapfrog"):
        pp = pp + (0.5 * h) * sys.force(qq)
        qq = qq + h * pp
        pp = pp + (0.5 * h) * sys.force(qq)
        return StepInfo(qq, pp, 2, 0, 0.0, True)
    if spec.name == "implicit_midpoint":
        z = qq + (0.5 * h) * pp
        prev = float("inf")
        iters = 0
        converged = False
        resid = float("inf")
        for it in range(1, int(max_iter) + 1):
            iters = it
            z_new = qq + (0.5 * h) * pp + (0.25 * h * h) * sys.force(z)
            resid = float(np.max(np.abs(z_new - z)))
            z = z_new
            scale = max(1.0, float(np.max(np.abs(z))))
            if resid <= tol * scale:
                converged = True
                break
            if resid > 2.0 * prev:
                break  # 发散：不要再迭代，免得把状态推到 1e300
            prev = resid
        pp_new = pp + h * sys.force(z)
        qq_new = 2.0 * z - qq
        return StepInfo(qq_new, pp_new, iters, iters, resid, converged)
    if spec.name in _RK_ALIAS:
        tab_name = _RK_ALIAS[spec.name]
        dof = qq.size

        def rhs(_t: float, y: np.ndarray) -> np.ndarray:
            return np.concatenate([y[dof:], sys.force(y[:dof])])

        y = np.concatenate([qq, pp])
        res = rk_step(tab_name, rhs, 0.0, y, h)
        return StepInfo(res.y[:dof].copy(), res.y[dof:].copy(), res.n_f_evals, 0, 0.0, True)
    raise ValueError(f"未知方法 {method!r}")  # pragma: no cover - method_spec 已经拦住


def step(method: str, sys: System, q: Any, p: Any, h: float, **kw: Any) -> tuple[np.ndarray, np.ndarray]:
    """走一步，只要 :math:`(q,p)`（隐式方法的迭代信息见 :func:`step_detailed`）。"""
    info = step_detailed(method, sys, q, p, h, **kw)
    return info.q, info.p


# --------------------------------------------------------------------------- #
# 4. 长时间积分
# --------------------------------------------------------------------------- #
def sample_indices(n_steps: int, n_sample: int) -> np.ndarray:
    """采样下标：**必含 0 与 n_steps**，其余在 :math:`[0,n_{steps}]` 上等距取整后去重。

    "尾部重复末点"是合法的补位（``n_sample ≥ n_steps`` 时得到的就是每一步一个点），
    但"末点不在采样里"不行——本章所有漂移斜率都取到观测终点为止，
    漏掉末点等于把观测时长缩短了一格。
    """
    n = int(n_steps)
    m = int(n_sample)
    if n < 1:
        raise ValueError(f"步数至少 1，得到 {n_steps}")
    if m < 1:
        raise ValueError(f"采样数至少 1，得到 {n_sample}")
    raw = np.rint(np.linspace(0.0, float(n), m + 1)).astype(np.int64)
    idx = np.unique(np.clip(raw, 0, n))
    return idx


@dataclass(frozen=True)
class Run:
    r"""一次（可能很长的）积分的全部可观测物：采样状态 + 不变量序列 + 诊断。

    ``energy_error`` 是**相对**能量误差 :math:`(H_k-H_0)/|H_0|`（:math:`H_0=0` 时退化为绝对差，
    由 :attr:`energy_relative` 标志说明），这样不同系统、不同初值的曲线可以同框比较。
    ``ok=False`` 表示在 ``t_blowup`` 处触及 :data:`BLOWUP` 而**主动停止**：
    从 ``i_blowup`` 起的采样点没有可信数据（调用方负责写占位值 + 掩码）。
    """

    system: str
    method: str
    h: float
    n_steps: int
    t: np.ndarray
    q: np.ndarray
    p: np.ndarray
    energy: np.ndarray
    energy_error: np.ndarray
    energy_relative: bool
    momentum: np.ndarray | None
    angular_momentum: np.ndarray | None
    n_force: int
    iterations_total: int
    iterations_max: int
    residual_max: float
    converged: bool
    ok: bool
    blowup: bool
    t_blowup: float
    i_blowup: int
    fast_path: bool
    n_sample: int
    idx: np.ndarray
    n_steps_done: int

    @property
    def dof(self) -> int:
        return int(self.q.shape[1])

    def final_energy_error(self) -> float:
        """最后一个**可信**采样点的相对能量误差（爆炸后取爆点前一点）。"""
        k = self.i_blowup if self.blowup else self.energy_error.size - 1
        return float(self.energy_error[max(0, min(k, self.energy_error.size - 1))])

    def n_periods(self) -> float | None:
        """观测时长相当于多少个**精确周期**（无周期概念的系统返回 ``None``）。"""
        per = system(self.system).period
        if per is None or per <= 0.0:
            return None
        return float(self.t[-1] / per)

    def energy_span(self) -> float:
        r""":math:`\max H-\min H`（绝对带宽，取爆炸前的可信段）。"""
        e = self.energy[: self.i_blowup] if self.blowup else self.energy
        return float(np.max(e) - np.min(e))

    def energy_band(self) -> float:
        r""":math:`\sup_k|H_k-H_0|`（相对 :math:`|H_0|`）——「有界」的那个数。"""
        err = self.energy_error[: self.i_blowup] if self.blowup else self.energy_error
        return float(np.max(np.abs(err)))

    def final_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.q[-1].copy(), self.p[-1].copy()


def _fast_steppers(method: str, sys: System, h: float, dof: int) -> Callable[..., tuple]:
    """构造纯标量快路径的单步闭包（dof ≤ 2）。

    返回的闭包签名由 ``dof`` 决定：

    * dof=1：``(q, p) -> (q, p, iters, resid, converged)``；
    * dof=2：``(q1, q2, p1, p2) -> (q1, q2, p1, p2, iters, resid, converged)``。

    显式方法的 ``iters=0``、``resid=0.0``、``converged=True``；隐式方法返回定点迭代的
    真实轮数与残差。**状态全是裸 float**：这是 10^6 步能在 0.2 s 内跑完的原因，
    也是本模块唯一"为了预算而妥协写法"的地方——数学顺序与通用路径逐字对应。
    """
    spec = method_spec(method)
    conv_tol = 1.0e-14

    def _fp1(q: float, p: float) -> tuple:
        h2q = 0.25 * h * h
        f = sys.force_1d
        z = q + 0.5 * h * p
        prev = float("inf")
        it = 0
        resid = float("inf")
        for k in range(1, 41):
            it = k
            z_new = q + 0.5 * h * p + h2q * f(z)
            resid = abs(z_new - z)
            z = z_new
            if resid <= conv_tol * max(1.0, abs(z)):
                return 2.0 * z - q, p + h * f(z), it, resid, True
            if resid > 2.0 * prev:
                break
            prev = resid
        return 2.0 * z - q, p + h * f(z), it, resid, False

    def _fp2(a: float, b: float, u: float, v: float) -> tuple:
        h2q = 0.25 * h * h
        g = sys.force_2d
        za = a + 0.5 * h * u
        zb = b + 0.5 * h * v
        prev = float("inf")
        it = 0
        resid = float("inf")
        for k in range(1, 41):
            it = k
            fa, fb = g(za, zb)
            na = a + 0.5 * h * u + h2q * fa
            nb = b + 0.5 * h * v + h2q * fb
            resid = max(abs(na - za), abs(nb - zb))
            za, zb = na, nb
            if resid <= conv_tol * max(1.0, abs(za), abs(zb)):
                fa, fb = g(za, zb)
                return 2.0 * za - a, 2.0 * zb - b, u + h * fa, v + h * fb, it, resid, True
            if resid > 2.0 * prev:
                break
            prev = resid
        fa, fb = g(za, zb)
        return 2.0 * za - a, 2.0 * zb - b, u + h * fa, v + h * fb, it, resid, False

    if dof == 1:
        f1 = sys.force_1d
        if f1 is None:
            raise ValueError(f"系统 {sys.name} 没有一维快路径")

        if spec.name == "symplectic_euler":
            def st1(q: float, p: float) -> tuple:
                p = p + h * f1(q)
                q = q + h * p
                return q, p, 0, 0.0, True
        elif spec.name in ("verlet", "leapfrog"):
            hh = 0.5 * h

            def st1(q: float, p: float) -> tuple:
                p = p + hh * f1(q)
                q = q + h * p
                p = p + hh * f1(q)
                return q, p, 0, 0.0, True
        elif spec.name == "implicit_midpoint":
            def st1(q: float, p: float) -> tuple:
                return _fp1(q, p)
        elif spec.name == "explicit_euler":
            def st1(q: float, p: float) -> tuple:
                fq = f1(q)  # 力在**旧**位置求值：这才是 y_{n+1}=y_n+h·F(y_n)
                q = q + h * p
                p = p + h * fq
                return q, p, 0, 0.0, True
        elif spec.name == "rk4":
            def st1(q: float, p: float) -> tuple:
                k1q, k1p = p, f1(q)
                k2q, k2p = p + 0.5 * h * k1p, f1(q + 0.5 * h * k1q)
                k3q, k3p = p + 0.5 * h * k2p, f1(q + 0.5 * h * k2q)
                k4q, k4p = p + h * k3p, f1(q + h * k3q)
                q = q + (h / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q)
                p = p + (h / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
                return q, p, 0, 0.0, True
        else:  # pragma: no cover - method_spec 已经拦住
            raise ValueError(f"未知方法 {method!r}")
        return st1

    if dof == 2:
        g = sys.force_2d
        if g is None:
            raise ValueError(f"系统 {sys.name} 没有二维快路径")

        if spec.name == "symplectic_euler":
            def st2(a: float, b: float, u: float, v: float) -> tuple:
                fa, fb = g(a, b)
                u = u + h * fa
                v = v + h * fb
                a = a + h * u
                b = b + h * v
                return a, b, u, v, 0, 0.0, True
        elif spec.name in ("verlet", "leapfrog"):
            hh = 0.5 * h

            def st2(a: float, b: float, u: float, v: float) -> tuple:
                fa, fb = g(a, b)
                u = u + hh * fa
                v = v + hh * fb
                a = a + h * u
                b = b + h * v
                fa, fb = g(a, b)
                u = u + hh * fa
                v = v + hh * fb
                return a, b, u, v, 0, 0.0, True
        elif spec.name == "implicit_midpoint":
            def st2(a: float, b: float, u: float, v: float) -> tuple:
                return _fp2(a, b, u, v)
        elif spec.name == "explicit_euler":
            def st2(a: float, b: float, u: float, v: float) -> tuple:
                fa, fb = g(a, b)
                a = a + h * u
                b = b + h * v
                u = u + h * fa
                v = v + h * fb
                return a, b, u, v, 0, 0.0, True
        elif spec.name == "rk4":
            def st2(a: float, b: float, u: float, v: float) -> tuple:
                k1a, k1b = u, v
                k1u, k1v = g(a, b)
                k2a, k2b = u + 0.5 * h * k1u, v + 0.5 * h * k1v
                k2u, k2v = g(a + 0.5 * h * k1a, b + 0.5 * h * k1b)
                k3a, k3b = u + 0.5 * h * k2u, v + 0.5 * h * k2v
                k3u, k3v = g(a + 0.5 * h * k2a, b + 0.5 * h * k2b)
                k4a, k4b = u + h * k3u, v + h * k3v
                k4u, k4v = g(a + h * k3a, b + h * k3b)
                a = a + (h / 6.0) * (k1a + 2.0 * k2a + 2.0 * k3a + k4a)
                b = b + (h / 6.0) * (k1b + 2.0 * k2b + 2.0 * k3b + k4b)
                u = u + (h / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
                v = v + (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
                return a, b, u, v, 0, 0.0, True
        else:  # pragma: no cover
            raise ValueError(f"未知方法 {method!r}")
        return st2

    raise ValueError(f"快路径只支持 dof ≤ 2，得到 {dof}")


def integrate(
    sys_name: str | System,
    method: str,
    h: float,
    n_steps: int,
    *,
    n_sample: int | None = None,
    max_abs_state: float = BLOWUP,
) -> Run:
    r"""定步长长时间积分：返回采样后的 :class:`Run`。

    采样口径：``n_sample`` 个区间（缺省 2000，且不超过 ``n_steps``），采样点必含
    :math:`t=0` 与 :math:`t=n_{steps}h`；**积分本身走满 ``n_steps`` 步**，
    没有被采样的步照样真算——只是不存。

    快路径（dof ≤ 2 的标量闭包）与通用路径（numpy）算的是同一个映射；
    ``tests/3.3::test_fast_path_matches_generic_path`` 逐位对拍两者。

    爆破解不撒谎：``max|q|`` 或 ``max|p|`` 触到 ``max_abs_state``（缺省 :data:`BLOWUP`）
    时立即停止，``ok=False``／``blowup=True``。口径（帧与测试都按这个来）：

    * ``i_blowup`` 是**第一个含爆炸状态**的采样点下标——**它前面**（``[:i_blowup]``）才是可信段，
      帧里把它们之后的位置写成占位值并配掩码；
    * ``n_steps_done`` 是真正走完的步数；被填尾的采样点 ``t`` 与 ``idx`` 都改成 ``n_steps_done``
      （此前 ``idx`` 保留着原计划步号，于是 ``t = idx*h`` 在尾部不成立——这是真实修过的一个 bug）；
    * 未爆炸时 ``i_blowup == idx.size``、``n_steps_done == n_steps``。
    """
    s = sys_name if isinstance(sys_name, System) else system(sys_name)
    spec = method_spec(method)
    n = int(n_steps)
    if n < 1:
        raise ValueError(f"步数至少 1，得到 {n_steps}")
    if not h > 0.0:
        raise ValueError(f"步长必须为正，得到 {h}")
    m = int(n_sample) if n_sample is not None else min(n, 2000)
    idx = sample_indices(n, m)
    use_fast = (s.dof <= 2) and ((s.dof == 1 and s.force_1d is not None) or (s.dof == 2 and s.force_2d is not None))

    e0 = s.energy_of(s.q0_array(), s.p0_array())
    energy_rel = abs(e0) > 0.0
    scale_e = abs(e0) if energy_rel else 1.0

    m_pts = int(idx.size)
    ts = np.zeros(m_pts, dtype=float)
    qs = np.zeros((m_pts, s.dof), dtype=float)
    ps = np.zeros((m_pts, s.dof), dtype=float)
    es = np.zeros(m_pts, dtype=float)
    mos: np.ndarray | None = np.zeros(m_pts, dtype=float) if s.momentum is not None else None
    angs: np.ndarray | None = np.zeros(m_pts, dtype=float) if s.angular_momentum is not None else None

    n_force = 0
    iters_total = 0
    iters_max = 0
    resid_max = 0.0
    converged = True
    blowup = False
    t_blow = float("nan")
    i_blow = 0

    def _record(si: int, done: int, qa: np.ndarray, pa: np.ndarray) -> None:
        ts[si] = done * h
        qs[si] = qa
        ps[si] = pa
        es[si] = s.energy_of(qa, pa)
        if mos is not None:
            mos[si] = float(s.momentum(pa))  # type: ignore[misc]
        if angs is not None:
            angs[si] = float(s.angular_momentum(qa, pa))  # type: ignore[misc]

    if use_fast:
        stepper = _fast_steppers(method, s, h, s.dof)
        if s.dof == 1:
            qf: float = float(s.q0[0])
            pf: float = float(s.p0[0])
        else:
            qf = [float(s.q0[0]), float(s.q0[1])]  # type: ignore[assignment]
            pf = [float(s.p0[0]), float(s.p0[1])]  # type: ignore[assignment]
        done = 0
        for si in range(m_pts):
            target = int(idx[si])
            if si > 0 and not blowup:
                seg = target - done
                if s.dof == 1:
                    for _ in range(seg):
                        qf, pf, it, res, cv = stepper(qf, pf)
                        done += 1
                        if it:
                            iters_total += it
                            iters_max = max(iters_max, it)
                            resid_max = max(resid_max, res)
                            converged = converged and cv
                            n_force += it
                        else:
                            n_force += spec.n_force
                        if not (abs(qf) < max_abs_state and abs(pf) < max_abs_state):
                            blowup = True
                            break
                else:
                    for _ in range(seg):
                        qf[0], qf[1], pf[0], pf[1], it, res, cv = stepper(qf[0], qf[1], pf[0], pf[1])
                        done += 1
                        if it:
                            iters_total += it
                            iters_max = max(iters_max, it)
                            resid_max = max(resid_max, res)
                            converged = converged and cv
                            n_force += it
                        else:
                            n_force += spec.n_force
                        if not (
                            abs(qf[0]) < max_abs_state
                            and abs(qf[1]) < max_abs_state
                            and abs(pf[0]) < max_abs_state
                            and abs(pf[1]) < max_abs_state
                        ):
                            blowup = True
                            break
                if blowup:
                    t_blow = done * h
                    i_blow = si
            qa = np.array([qf], dtype=float) if s.dof == 1 else np.array([qf[0], qf[1]], dtype=float)
            pa = np.array([pf], dtype=float) if s.dof == 1 else np.array([pf[0], pf[1]], dtype=float)
            _record(si, done, qa, pa)
            if blowup:
                ts[si:] = ts[si]
                idx[si:] = done
                qs[si:] = qa
                ps[si:] = pa
                es[si:] = es[si]
                if mos is not None:
                    mos[si:] = mos[si]
                if angs is not None:
                    angs[si:] = angs[si]
                break
    else:
        qa = s.q0_array().copy()
        pa = s.p0_array().copy()
        done = 0
        for si in range(m_pts):
            target = int(idx[si])
            if si > 0 and not blowup:
                seg = target - done
                for _ in range(seg):
                    info = step_detailed(method, s, qa, pa, h, max_iter=40)
                    qa, pa = info.q, info.p
                    done += 1
                    n_force += info.n_force
                    iters_total += info.iterations
                    iters_max = max(iters_max, info.iterations)
                    resid_max = max(resid_max, info.residual)
                    if info.iterations:
                        converged = converged and info.converged
                    if float(np.max(np.abs(qa))) >= max_abs_state or float(np.max(np.abs(pa))) >= max_abs_state:
                        blowup = True
                        break
                if blowup:
                    t_blow = done * h
                    i_blow = si
            _record(si, done, qa, pa)
            if blowup:
                ts[si:] = ts[si]
                qs[si:] = qa
                ps[si:] = pa
                es[si:] = es[si]
                if mos is not None:
                    mos[si:] = mos[si]
                if angs is not None:
                    angs[si:] = angs[si]
                break

    err = (es - es[0]) / scale_e
    return Run(
        system=s.name,
        method=method,
        h=float(h),
        n_steps=n,
        t=ts,
        q=qs,
        p=ps,
        energy=es,
        energy_error=err,
        energy_relative=energy_rel,
        momentum=mos,
        angular_momentum=angs,
        n_force=int(n_force),
        iterations_total=int(iters_total),
        iterations_max=int(iters_max),
        residual_max=float(resid_max),
        converged=bool(converged),
        ok=not blowup,
        blowup=bool(blowup),
        t_blowup=float(t_blow),
        i_blowup=int(i_blow),
        fast_path=bool(use_fast),
        n_sample=int(m_pts),
        idx=idx,
        n_steps_done=int(done),
    )


# --------------------------------------------------------------------------- #
# 5. 线性映射与修正能量（谐振子专用，但结论是本章的骨架）
# --------------------------------------------------------------------------- #
def orbital_elements(q: Any, p: Any, *, mu: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""由状态矢量算 Kepler 轨道三要素 :math:`(a,e,\varpi)`（**瞬时**值，逐点可算）。

    * 比角动量 :math:`L=q_xp_y-q_yp_x`；
    * Laplace--Runge--Lenz 矢量 :math:`\mathbf A=\mathbf p\times\mathbf L-\mu\,\mathbf q/r`
      （二维：:math:`A_x=p_yL-\mu q_x/r`、:math:`A_y=-p_xL-\mu q_y/r`），
      :math:`e=\|\mathbf A\|/\mu`、近日点方向 :math:`\varpi=\mathrm{atan2}(A_y,A_x)`；
    * 半长轴由能量给：:math:`a=-\mu/(2E)`。

    为什么要它：轨道**形状**由 :math:`(E,L)` 决定、**方位**由 :math:`\mathbf A` 的方向决定，
    两者是不同性质的量。辛性只管前者（能量有界 + 角动量逐位守恒 ⟹ 形状有界），
    后者是**相位型**误差，任何固定步长方法都会进动——把两者混在一个数里，
    就会得出"辛方法也会长期失真"的错误结论（本章 pitfalls 专门讲这一条）。
    """
    qq = np.asarray(q, dtype=float)
    pp = np.asarray(p, dtype=float)
    single = qq.ndim == 1
    if single:
        qq = qq[None, :]
        pp = pp[None, :]
    qx, qy = qq[:, 0], qq[:, 1]
    px, py = pp[:, 0], pp[:, 1]
    ell = qx * py - qy * px
    r = np.sqrt(qx * qx + qy * qy)
    ax = py * ell - mu * qx / r
    ay = -px * ell - mu * qy / r
    ecc = np.sqrt(ax * ax + ay * ay) / mu
    varpi = np.arctan2(ay, ax)
    energy = 0.5 * (px * px + py * py) - mu / r
    with np.errstate(divide="ignore", invalid="ignore"):
        a = -mu / (2.0 * energy)
    if single:
        return a[:1], ecc[:1], varpi[:1]
    return a, ecc, varpi


def det_2x2(M: Any) -> float:
    r"""2×2 行列式手算（:math:`\det M=ad-bc`）——``numpy.linalg`` 属 P2 红线。"""
    A = np.asarray(M, dtype=float)
    if A.shape != (2, 2):
        raise ValueError(f"需要 2×2，得到 {A.shape}")
    return float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])


def linear_map(method: str, h: float, *, omega: float = 1.0) -> np.ndarray:
    r"""方法在谐振子上的**精确 2×2 映射** :math:`M`：:math:`y_{n+1}=My_n`、:math:`y=(q,p)`。

    构造方式是**照着定义做实验**：把通用单步函数作用在两个基向量
    :math:`(1,0)`、:math:`(0,1)` 上（线性问题里这两个像就是 :math:`M` 的两列）。
    频率 :math:`\omega` 通过时间尺度进入：:math:`M=M_{\omega=1}(\omega h)`，
    所以内部用步长 :math:`\omega h` 调用单位频率系统。

    ``symplectic_euler`` 与 ``verlet`` 满足 :math:`\det M=1`（离散辛条件在 2×2 上的全部内容），
    ``rk4`` 的 :math:`\det M=|R(ih)|^2=1-h^6/72+\cdots\ne1`——这正是它线性漂移的根源。
    """
    if not h > 0.0:
        raise ValueError(f"步长必须为正，得到 {h}")
    sys = system("harmonic")
    e1q, e1p = step(method, sys, np.array([1.0]), np.array([0.0]), float(h) * float(omega))
    e2q, e2p = step(method, sys, np.array([0.0]), np.array([1.0]), float(h) * float(omega))
    return np.array(
        [
            [float(e1q[0]), float(e2q[0])],
            [float(e1p[0]), float(e2p[0])],
        ],
        dtype=float,
    )
