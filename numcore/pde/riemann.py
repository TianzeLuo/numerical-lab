r"""4.3 章内核之一：Riemann 问题、熵条件与数值通量。

这一层只做一件事：**把"界面上的跳跃"翻译成一条速度**。所有 FVM 格式的差别，
最后都落在同一个函数签名上——给定左右两个状态，返回一个通量
:math:`F(u_L,u_R)`；而"谁对谁错"完全由 :math:`F` 在**跨声速稀疏波**与**激波**处的
取值决定。因此本模块把三件事严格分开：

1. **标量守恒律**（凸/凹/线性三种通量）：Riemann 解的结构（激波 / 稀疏波 /
   接触间断）、Rankine--Hugoniot 速度、广义 Lax 条件与 Oleinik 条件；
2. **数值通量**：``godunov``（精确解在 :math:`\xi=0` 处的通量，**定义式**）、
   ``rusanov``（局部 Lax--Friedrichs）、``osher``（通量分裂）、以及 ``upwind``
   （把 :math:`F=f(u_j)` 当通量的"想当然"写法——它在跨声速稀疏波上**违反熵条件**，
   本章用它当负对照）；
3. **Euler 方程组**的精确 Riemann 解（Toro 的 :math:`p_*` 迭代）：激波 / 接触间断 /
   稀疏波三种波同时出现，接触间断正是标量守恒律里**不存在**的那种波。

两条纪律贯穿全文件：

* **通量只能有一个真源**。``godunov_flux`` 不另写一套 if-else，而是直接取
  ``scalar_sample(kind, 0.0, uL, uR)`` 再作用通量——"Godunov 通量 = 精确解在
  :math:`x/t=0` 处的通量"是**定义**，分开写迟早会漂移；
* **熵条件是可测的**。稀疏波在跨声速处必须由 :math:`f'(u)=0` 的临界点决定通量
  （Burgers 给 :math:`F=0`），少这一步就会得到一条**驻定的膨胀激波**——它满足
  弱解定义、满足守恒、也满足所有非熵判据，只在熵产率上露馅。

Euler 侧的约定：守恒量 :math:`U=(\rho,\rho u,E)`、:math:`E=p/(\gamma-1)+\rho u^2/2`，
理想气体 :math:`\gamma=1.4`；原始变量 :math:`W=(\rho,u,p)`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "GAMMA",
    "SCALAR_FLUXES",
    "scalar_flux",
    "flux_derivative",
    "flux_curvature",
    "shock_speed",
    "lax_entropy_condition",
    "oleinik_margin",
    "entropy_ok",
    "ScalarRiemann",
    "scalar_riemann",
    "scalar_sample",
    "godunov_flux",
    "rusanov_flux",
    "osher_flux",
    "upwind_flux",
    "scalar_numerical_flux",
    "SCALAR_SCHEMES",
    "EULER_PROBLEMS",
    "euler_state",
    "euler_primitives",
    "euler_flux",
    "sound_speed",
    "pressure_function",
    "EulerStar",
    "euler_star_state",
    "euler_sample",
    "euler_exact",
    "euler_rh_residuals",
    "riemann_invariant",
    "euler_entropy",
    "hllc_flux",
    "rusanov_flux_euler",
    "euler_numerical_flux",
]

#: 理想气体比热比（本章所有 Euler 算例共用）。
GAMMA = 1.4

#: 支持标量通量的名字：凸（Burgers）、线性（接触间断）、凹（交通流）。
SCALAR_FLUXES: tuple[str, ...] = ("burgers", "linear", "traffic")

#: 标量数值通量：前三个相容、第四个在跨声速稀疏波上违反熵条件（负对照）。
SCALAR_SCHEMES: tuple[str, ...] = ("godunov", "rusanov", "osher", "upwind")

#: 线性通量 :math:`f(u)=a u` 的波速（Riemann 不变量处处的特征速度）。
LINEAR_SPEED = 1.0

_SMALL = 1.0e-300


def _signed_guard(x: Any, floor: float = 1.0e-30) -> np.ndarray:
    r"""保号的除法护垫：``|x|<floor`` 时换成同号的 ``floor``。

    为什么不能写 ``np.maximum(x, tiny)``：这里要护的多数是**负的**波速差
    （:math:`S_L-u_L<0`），``maximum`` 会把它们全抬成 ``+tiny``，于是
    :math:`O(1)` 的物理量除以 ``1e-300`` 直接溢出——HLLC 在第一次迭代就炸成
    ``inf``，而报错位置看起来却在"能量"那一行（本模块开发时真实踩过）。
    """
    arr = np.asarray(x, dtype=float)
    tiny = np.where(arr < 0.0, -float(floor), float(floor))
    return np.where(np.abs(arr) < float(floor), tiny, arr)

#: Toro 的三道标准考题（原始变量 ``(rho, u, p)`` 左右态）。
EULER_PROBLEMS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    # 激波 + 接触间断 + 稀疏波：三种波同时出现。
    "sod": ((1.0, 0.0, 1.0), (0.125, 0.0, 0.1)),
    # Lax：更强的激波，接触间断几乎贴着激波。
    "lax": ((0.445, 0.698, 3.528), (0.5, 0.0, 0.571)),
    # 双稀疏波（无真空）：u_* 恰为 0，用来测"对称性"这条不变量。
    "double_rarefaction": ((1.0, -0.5, 1.0), (1.0, 0.5, 1.0)),
}


# --------------------------------------------------------------------------- #
# 标量通量与它们的导数
# --------------------------------------------------------------------------- #
def scalar_flux(kind: str, u: Any) -> np.ndarray:
    r"""标量通量 :math:`f(u)`。

    * ``burgers``：:math:`f=u^2/2`、:math:`f'=u`（**凸**，特征速度就是解本身）；
    * ``linear``：:math:`f=a u`、:math:`a=1`（:math:`f''\equiv0`，**线性退化**——
      跳跃以 :math:`a` 传播，这就是接触间断，标量场上它是唯一可能的"接触"）；
    * ``traffic``：:math:`f=u(1-u)`、:math:`f'=1-2u`（**凹**，激波与稀疏波的角色
      与凸通量互换：:math:`u_L<u_R` 才是压缩）。
    """
    arr = np.asarray(u, dtype=float)
    key = str(kind).strip().lower()
    if key == "burgers":
        return 0.5 * arr * arr
    if key == "linear":
        return LINEAR_SPEED * arr
    if key == "traffic":
        return arr * (1.0 - arr)
    raise ValueError(f"未知标量通量 {kind!r}（可用：{list(SCALAR_FLUXES)}）")


def flux_derivative(kind: str, u: Any) -> np.ndarray:
    r"""特征速度 :math:`f'(u)`（Riemann 不变量的斜率、CFL 的波速）。"""
    arr = np.asarray(u, dtype=float)
    key = str(kind).strip().lower()
    if key == "burgers":
        return arr
    if key == "linear":
        return np.full(arr.shape, LINEAR_SPEED, dtype=float)
    if key == "traffic":
        return 1.0 - 2.0 * arr
    raise ValueError(f"未知标量通量 {kind!r}（可用：{list(SCALAR_FLUXES)}）")


def flux_curvature(kind: str, u: Any) -> np.ndarray:
    r"""二阶导数 :math:`f''(u)`：凸（+）、线性（0）、凹（-）的判据。"""
    arr = np.asarray(u, dtype=float)
    key = str(kind).strip().lower()
    if key == "burgers":
        return np.ones(arr.shape, dtype=float)
    if key == "linear":
        return np.zeros(arr.shape, dtype=float)
    if key == "traffic":
        return np.full(arr.shape, -2.0, dtype=float)
    raise ValueError(f"未知标量通量 {kind!r}（可用：{list(SCALAR_FLUXES)}）")


def _curvature_sign(kind: str) -> int:
    """整条实轴上曲率的符号（+1 凸 / 0 线性 / -1 凹）。"""
    key = str(kind).strip().lower()
    return {"burgers": 1, "linear": 0, "traffic": -1}[key]


def shock_speed(kind: str, u_left: float, u_right: float) -> float:
    r"""Rankine--Hugoniot 激波速度 :math:`s=\dfrac{f(u_L)-f(u_R)}{u_L-u_R}`。

    这是整章的对拍靶子：**数值格式自己不知道** :math:`s`，它只会搬运通量；
    把实测激波位置对时间做最小二乘，斜率必须收敛到这个闭式值。左右态相同即报错
    （那时"激波"没有定义，返回任何数都会变成一条假证据）。
    """
    ul = float(u_left)
    ur = float(u_right)
    if ul == ur:
        raise ValueError(f"左右态相同（{ul}），Riemann 问题退化，没有激波速度可言")
    return float((float(scalar_flux(kind, ul)) - float(scalar_flux(kind, ur))) / (ul - ur))


def lax_entropy_condition(kind: str, u_left: float, u_right: float, speed: float) -> bool:
    r"""广义 Lax 熵条件 :math:`f'(u_L)>s>f'(u_R)`。

    对凸通量它等价于"激波是压缩的"（:math:`u_L>u_R`）；对凹通量（``traffic``）
    同一不等式给出的方向自动反过来。**它不是"激波速度对不对"的判据**——
    熵违背的膨胀激波同样满足 RH，只是不满足这一条。
    """
    return bool(
        float(flux_derivative(kind, u_left)) > float(speed) > float(flux_derivative(kind, u_right))
    )


def oleinik_margin(kind: str, u_left: float, u_right: float, speed: float, *, n_scan: int = 257) -> float:
    r"""Oleinik 熵条件的余量 :math:`\min_{u}\frac{f(u)-f(u_L)}{u-u_L}-s`（内部扫描）。

    凸通量时它与 Lax 条件等价；非凸通量（这里指曲率变号，本章的 ``traffic`` 不变号）
    时它更强。返回负值即违背：**所有"看起来对"的守恒格式都可能在非凸通量上漏掉它**。
    扫描点数写进调用方的 ``detail``——余量是"扫描出来的"，不是解析下界。
    """
    ul = float(u_left)
    ur = float(u_right)
    if ul == ur:
        raise ValueError("左右态相同，Oleinik 条件无定义")
    lo, hi = (ur, ul) if ul > ur else (ul, ur)
    us = np.linspace(lo, hi, int(n_scan))
    f = np.asarray(scalar_flux(kind, us), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (f - float(scalar_flux(kind, ul))) / (us - ul)
    ok = np.isfinite(slope)
    if not bool(np.any(ok)):
        return float("nan")
    return float(np.min(slope[ok]) - float(speed))


def entropy_ok(kind: str, u_left: float, u_right: float) -> bool:
    r"""Riemann 解是否满足熵条件（凸/线性走 Lax，凹走同一不等式的镜像）。"""
    ul = float(u_left)
    ur = float(u_right)
    if ul == ur:
        return True
    if _curvature_sign(kind) == 0:
        return True  # 线性退化：接触间断，Lax 条件退化成等式
    if flux_derivative(kind, ul) > flux_derivative(kind, ur):
        return lax_entropy_condition(kind, ul, ur, shock_speed(kind, ul, ur))
    return True  # 稀疏波：由凸/凹性自动满足熵条件（本章不处理复合波）


# --------------------------------------------------------------------------- #
# 标量 Riemann 解
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScalarRiemann:
    """标量 Riemann 解的结构（激波 / 稀疏波 / 接触间断 + 熵条件判定）。

    ``wave`` 取 ``shock`` / ``rarefaction`` / ``contact`` / ``trivial``；
    稀疏波额外给出 ``head``、``tail``（扇面的两条边界特征速度）与 ``sonic``
    （是否有 :math:`f'(u)=0` 的临界点落在扇面**内部**——跨声速稀疏波，
    也就是所有熵违背格式翻车的地方）。
    """

    kind: str
    u_left: float
    u_right: float
    wave: str
    speed: float
    head: float
    tail: float
    sonic: bool
    entropy_ok: bool
    entropy_margin: float
    curvature: int


def scalar_riemann(kind: str, u_left: float, u_right: float) -> ScalarRiemann:
    r"""解一维标量 Riemann 问题 :math:`u_t+f(u)_x=0`，:math:`u(x,0)=u_L\,(x<0),u_R\,(x>0)`。

    判据只看特征速度的方向：:math:`f'(u_L)>f'(u_R)` 时特征线**相交**（压缩）→ 激波，
    速度由 RH 给出；:math:`f'(u_L)<f'(u_R)` 时特征线**发散** → 稀疏波，扇面边界是
    :math:`f'(u_L)` 与 :math:`f'(u_R)`；:math:`f''\equiv0`（线性通量）时两者相等，
    跳跃原地以 :math:`a` 传播——那是**接触间断**（线性退化），也是标量场上唯一
    不需要熵条件的波。
    """
    ul = float(u_left)
    ur = float(u_right)
    key = str(kind).strip().lower()
    curv = _curvature_sign(key)
    lam_l = float(flux_derivative(key, ul))
    lam_r = float(flux_derivative(key, ur))
    if ul == ur:
        return ScalarRiemann(key, ul, ur, "trivial", lam_l, lam_l, lam_l, False, True, 0.0, curv)
    if curv == 0:
        # 线性通量：整条解就是一个以 a 传播的接触间断。
        s = LINEAR_SPEED
        return ScalarRiemann(key, ul, ur, "contact", s, lam_l, lam_r, False, True, 0.0, curv)
    if lam_l > lam_r:
        s = shock_speed(key, ul, ur)
        margin = oleinik_margin(key, ul, ur, s)
        return ScalarRiemann(
            key, ul, ur, "shock", s, lam_l, lam_r, False,
            lax_entropy_condition(key, ul, ur, s), margin, curv,
        )
    sonic = bool(lam_l < 0.0 < lam_r)
    return ScalarRiemann(key, ul, ur, "rarefaction", 0.0, lam_l, lam_r, sonic, True, 0.0, curv)


def _inverse_flux_derivative(kind: str, xi: Any, lo: float, hi: float) -> np.ndarray:
    r"""在扇面 :math:`[u_L,u_R]` 内解 :math:`f'(u)=\xi`（凸/凹两种通量的闭式，兜底二分）。

    Burgers 上它给出本章最好用的一条 A 类对拍：稀疏波内部**解就是** :math:`u=x/t`
    （因为 :math:`f'(u)=u`），不是"近似等于"。
    """
    key = str(kind).strip().lower()
    xs = np.asarray(xi, dtype=float)
    if key == "burgers":
        return xs
    if key == "traffic":
        return 0.5 * (1.0 - xs)
    # 兜底：f' 在扇面内单调（凸/凹），二分 100 步足够到双精度。
    a = np.full(xs.shape, float(lo), dtype=float)
    b = np.full(xs.shape, float(hi), dtype=float)
    for _ in range(100):
        m = 0.5 * (a + b)
        below = np.asarray(flux_derivative(key, m), dtype=float) <= xs
        a = np.where(below, m, a)
        b = np.where(below, b, m)
    return 0.5 * (a + b)


def scalar_sample(kind: str, xi: Any, u_left: Any, u_right: Any) -> np.ndarray:
    r"""精确解在相似变量 :math:`\xi=x/t` 处的取值（自相似解的定义式，**逐元素向量化**）。

    * 激波：:math:`\xi<s` 取 :math:`u_L`，否则 :math:`u_R`（间断就在 :math:`\xi=s`）；
    * 稀疏波：:math:`\xi\le f'(u_L)` 取 :math:`u_L`、:math:`\xi\ge f'(u_R)` 取 :math:`u_R`，
      中间用 :math:`f'(u)=\xi` 反解——**跨声速时中间那段包含** :math:`f'=0`
      的临界点，这正是熵条件的落点；
    * 接触间断 / 常值：以 :math:`f'(u_L)` 传播的跳跃（:math:`u_L=u_R` 时两支相同）。

    波型判据是逐元素的 :math:`f'(u_L)` 与 :math:`f'(u_R)` 的大小关系——
    写成标量循环会让"每一对界面都要解一次 Riemann 问题"的 FVM 直接慢两个数量级，
    所以这里从第一天就按数组写：`:math:`xi`、:math:`u_L`、:math:`u_R` 必须同形
    （广播由调用方负责，`godunov_flux` 用 ``np.zeros`` 造 :math:`\xi=0` 的数组）。

    ``t=0`` 时相似变量没有定义，调用方必须自己处理（见 :func:`euler_exact` 与
    :func:`numcore.pde.fvm.exact_riemann_solution` 的 ``t_floor`` 与掩码约定）。
    """
    ul = np.asarray(u_left, dtype=float)
    ur = np.asarray(u_right, dtype=float)
    xs = np.asarray(xi, dtype=float)
    lam_l = np.asarray(flux_derivative(kind, ul), dtype=float)
    lam_r = np.asarray(flux_derivative(kind, ur), dtype=float)
    fl = np.asarray(scalar_flux(kind, ul), dtype=float)
    fr = np.asarray(scalar_flux(kind, ur), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_rh = np.where(ul != ur, (fl - fr) / np.where(ul != ur, ul - ur, 1.0), lam_l)
    out_shock = np.where(xs < s_rh, ul, ur)
    lo = np.minimum(ul, ur)
    hi = np.maximum(ul, ur)
    mid = _inverse_flux_derivative(kind, xs, lo, hi)
    out_rare = np.where(xs <= lam_l, ul, np.where(xs >= lam_r, ur, mid))
    out_eq = np.where(xs < lam_l, ul, ur)
    converging = lam_l > lam_r
    diverging = lam_l < lam_r
    return np.where(converging, out_shock, np.where(diverging, out_rare, out_eq))


# --------------------------------------------------------------------------- #
# 标量数值通量
# --------------------------------------------------------------------------- #
def godunov_flux(kind: str, u_left: Any, u_right: Any) -> np.ndarray:
    r"""Godunov 通量 :math:`F=f\bigl(u(\xi=0)\bigr)`——**精确解在界面上的通量**。

    实现上它只调用 :func:`scalar_sample` 一次：这条"不另写一套 if-else"的纪律
    是刻意的。跨声速稀疏波（:math:`u_L<0<u_R`，Burgers）上它给出
    :math:`f(0)=0`；把这一步写漏，得到的格式会收敛到**膨胀激波**。
    """
    ul = np.asarray(u_left, dtype=float)
    ur = np.asarray(u_right, dtype=float)
    star = scalar_sample(kind, np.zeros(np.broadcast(ul, ur).shape, dtype=float), ul, ur)
    return np.asarray(scalar_flux(kind, star), dtype=float)


def rusanov_flux(kind: str, u_left: Any, u_right: Any) -> np.ndarray:
    r"""局部 Lax--Friedrichs（Rusanov）通量
    :math:`F=\frac12(f_L+f_R)-\frac12 a_{\max}(u_R-u_L)`，
    :math:`a_{\max}=\max|f'|` 在两态之间取。

    **它不分辨波型**：一个常系数的耗散项同时抹平激波与稀疏波，所以激波被抹宽
    （误差 :math:`O(\Delta x)`、比 Godunov 宽），但**熵条件自动满足**——
    耗散太强反而"不会犯熵错误"。这条对照必须与 Godunov 并排报。
    """
    ul = np.asarray(u_left, dtype=float)
    ur = np.asarray(u_right, dtype=float)
    fl = np.asarray(scalar_flux(kind, ul), dtype=float)
    fr = np.asarray(scalar_flux(kind, ur), dtype=float)
    lo = np.minimum(ul, ur)
    hi = np.maximum(ul, ur)
    a = np.maximum(np.abs(np.asarray(flux_derivative(kind, lo), dtype=float)),
                   np.abs(np.asarray(flux_derivative(kind, hi), dtype=float)))
    return 0.5 * (fl + fr) - 0.5 * a * (ur - ul)


def _split_integrals(kind: str, u: Any) -> tuple[np.ndarray, np.ndarray]:
    r"""Osher--Solomon 通量分裂 :math:`f^\pm`（以 0 为参考点，:math:`f=f^++f^-`）。

    :math:`f^+(u)=f(0)+\int_0^u\max(f',0)`、:math:`f^-(u)=\int_0^u\min(f',0)`。
    注意这两支是**积分的值**而不是被积函数：:math:`u<0` 时 Burgers 的
    :math:`\int_0^u s\,ds=+\tfrac12u^2` 是**正的**（负的被积函数沿负方向积分）——
    第一版把符号写成 :math:`-\tfrac12\min(u,0)^2`，结果 Osher 通量在
    :math:`f'<0` 的一侧整片反号，跨声速稀疏波当场炸成 :math:`10^5`（实测），
    而"激波"算例上一切正常，因为那里两支恰好同号。
    """
    arr = np.asarray(u, dtype=float)
    key = str(kind).strip().lower()
    if key == "burgers":
        fp = 0.5 * np.maximum(arr, 0.0) ** 2
        fm = 0.5 * np.minimum(arr, 0.0) ** 2
        return fp, fm
    if key == "linear":
        return LINEAR_SPEED * arr, np.zeros_like(arr)
    if key == "traffic":
        up = arr <= 0.5
        fp = np.where(up, arr - arr * arr, 0.25 + 0.0 * arr)
        down = arr >= 0.5
        fm = np.where(down, -(arr - 0.5) ** 2, 0.0 + 0.0 * arr)
        return fp, fm
    raise ValueError(f"未知标量通量 {kind!r}（可用：{list(SCALAR_FLUXES)}）")


def osher_flux(kind: str, u_left: Any, u_right: Any) -> np.ndarray:
    r"""Osher--Solomon 通量 :math:`F=f^+(u_L)+f^-(u_R)`。

    它是**唯一**在本章里三处都对（激波、跨声速稀疏波、线性退化）又不引入额外
    耗散的通量：跨声速时上游那一支取到临界点为止。它和 Godunov 通量在
    Burgers 上**逐点相同**（本章断言实测偏差 0），在凹通量上也能用——
    这正是"通量分裂"值得单独讲的理由。
    """
    fp, _ = _split_integrals(kind, u_left)
    _, fm = _split_integrals(kind, u_right)
    return np.asarray(fp, dtype=float) + np.asarray(fm, dtype=float)


def upwind_flux(kind: str, u_left: Any, u_right: Any) -> np.ndarray:
    r"""朴素"迎风"通量 :math:`F=f(u_{\text{upwind}})`，上游由 :math:`f'(u_L)+f'(u_R)` 的符号定。

    它是本章的**负对照**：守恒、相容、单调、TVD、稳定（这些判据它全过），
    但在跨声速稀疏波上 :math:`F=f(u_L)\ne f(0)`——于是数值解收敛到一条
    **驻定的膨胀激波**：满足弱解定义、满足 RH、TV 也不增，只在熵条件上翻车。
    换句话说：**除了熵条件，所有"看起来对"的检查都抓不住它**。
    """
    ul = np.asarray(u_left, dtype=float)
    ur = np.asarray(u_right, dtype=float)
    up = (np.asarray(flux_derivative(kind, ul), dtype=float)
          + np.asarray(flux_derivative(kind, ur), dtype=float)) > 0.0
    return np.where(up, scalar_flux(kind, ul), scalar_flux(kind, ur))


def scalar_numerical_flux(kind: str, u_left: Any, u_right: Any, scheme: str = "godunov") -> np.ndarray:
    r"""按名字分发标量数值通量（``godunov`` / ``rusanov`` / ``osher`` / ``upwind``）。"""
    key = str(scheme).strip().lower()
    if key == "godunov":
        return godunov_flux(kind, u_left, u_right)
    if key == "rusanov":
        return rusanov_flux(kind, u_left, u_right)
    if key == "osher":
        return osher_flux(kind, u_left, u_right)
    if key == "upwind":
        return upwind_flux(kind, u_left, u_right)
    raise ValueError(f"未知标量通量格式 {scheme!r}（可用：{list(SCALAR_SCHEMES)}）")


# --------------------------------------------------------------------------- #
# Euler 方程组：精确 Riemann 解
# --------------------------------------------------------------------------- #
def euler_state(rho: Any, u: Any, p: Any, gamma: float = GAMMA) -> np.ndarray:
    r"""原始变量 :math:`(\rho,u,p)\to` 守恒量 :math:`(\rho,\rho u,E)`（最后一维为分量）。"""
    r = np.asarray(rho, dtype=float)
    v = np.asarray(u, dtype=float)
    pr = np.asarray(p, dtype=float)
    e = pr / (float(gamma) - 1.0) + 0.5 * r * v * v
    return np.stack([r, r * v, e], axis=-1)


def euler_primitives(U: Any, gamma: float = GAMMA) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""守恒量 :math:`(\rho,\rho u,E)\to` 原始变量 :math:`(\rho,u,p)`。"""
    arr = np.asarray(U, dtype=float)
    r = arr[..., 0]
    mom = arr[..., 1]
    e = arr[..., 2]
    v = mom / np.maximum(r, _SMALL)
    p = (float(gamma) - 1.0) * (e - 0.5 * mom * mom / np.maximum(r, _SMALL))
    return r, v, p


def euler_flux(U: Any, gamma: float = GAMMA) -> np.ndarray:
    r"""Euler 通量 :math:`F=(\rho u,\ \rho u^2+p,\ u(E+p))`（逐分量，支持批量）。"""
    arr = np.asarray(U, dtype=float)
    r, v, p = euler_primitives(arr, gamma)
    e = arr[..., 2]
    out = np.stack([r * v, r * v * v + p, v * (e + p)], axis=-1)
    return out


def sound_speed(rho: Any, p: Any, gamma: float = GAMMA) -> np.ndarray:
    r"""声速 :math:`c=\sqrt{\gamma p/\rho}`（非物理态会被调用方先挡住）。"""
    r = np.asarray(rho, dtype=float)
    pr = np.asarray(p, dtype=float)
    return np.sqrt(np.maximum(float(gamma) * pr / np.maximum(r, _SMALL), 0.0))


def pressure_function(p_star: float, w: Sequence[float], gamma: float = GAMMA) -> float:
    r"""Toro 的压力函数 :math:`f_K(p_*)`：左/右波把 :math:`p_*` 映射成速度修正。

    * 激波支（:math:`p_*>p_K`）：:math:`f_K=(p_*-p_K)\sqrt{A_K/(p_*+B_K)}`，
      :math:`A_K=2/((\gamma+1)\rho_K)`、:math:`B_K=p_K(\gamma-1)/(\gamma+1)`；
    * 稀疏波支（:math:`p_*\le p_K`）：
      :math:`f_K=\frac{2c_K}{\gamma-1}\bigl[(p_*/p_K)^{(\gamma-1)/(2\gamma)}-1\bigr]`。

    两支在 :math:`p_*=p_K` 处**一阶连续**（声学极限），所以 :math:`f=f_L+f_R+(u_R-u_L)`
    在 :math:`p_*` 上严格单调增，二分法必然收敛。
    """
    rho_k = float(w[0])
    u_k = float(w[1])
    p_k = float(w[2])
    g = float(gamma)
    ps = float(p_star)
    if ps > p_k:
        a_k = 2.0 / ((g + 1.0) * rho_k)
        b_k = p_k * (g - 1.0) / (g + 1.0)
        return float((ps - p_k) * np.sqrt(a_k / max(ps + b_k, _SMALL)))
    c_k = float(sound_speed(rho_k, p_k, g))
    return float(2.0 * c_k / (g - 1.0) * ((max(ps, 0.0) / p_k) ** ((g - 1.0) / (2.0 * g)) - 1.0))


@dataclass(frozen=True)
class EulerStar:
    """Euler Riemann 解的星区状态与三条波的位置。

    ``p_star``、``u_star`` 是接触间断上的压强与速度；``rho_star_l``/``rho_star_r``
    是接触间断两侧的密度（**它们不相等**——这正是接触间断的定义：压强与速度连续、
    密度跳跃）。``left_wave``/``right_wave`` 取 ``shock`` 或 ``rarefaction``。
    """

    p_star: float
    u_star: float
    rho_star_l: float
    rho_star_r: float
    left_wave: str
    right_wave: str
    s_left: float
    s_right: float
    head_l: float
    tail_l: float
    head_r: float
    tail_r: float
    vacuum: bool
    iterations: int
    bracket: float


def euler_star_state(
    w_left: Sequence[float],
    w_right: Sequence[float],
    gamma: float = GAMMA,
    *,
    tol: float = 1.0e-14,
    max_iter: int = 200,
) -> EulerStar:
    r"""精确 Riemann 解：二分求 :math:`p_*`，再由波型给出两侧星区状态。

    方程 :math:`f(p_*)=f_L(p_*)+f_R(p_*)+(u_R-u_L)=0` 单调，二分不动点：
    先取 :math:`p_{\text{hi}}=\max(p_L,p_R)`，不够大就翻倍；下界从
    :math:`10^{-14}\max(p_L,p_R)` 起（:math:`p_*\to0` 时 :math:`f` 有有限极限，
    若 :math:`f(0)>0` 则出现**真空**——本章的算例都避开真空，但代码要如实标记）。

    返回的 :math:`\rho_{*L}\ne\rho_{*R}` 是接触间断的**证据**；两侧波速都写成
    "头部 / 尾部"两个数：激波时两者相等（退化成一条线），稀疏波时是扇面的两条边界。
    """
    wl = tuple(float(v) for v in w_left)
    wr = tuple(float(v) for v in w_right)
    g = float(gamma)
    rho_l, u_l, p_l = wl
    rho_r, u_r, p_r = wr
    if min(rho_l, p_l, rho_r, p_r) <= 0.0:
        raise ValueError(f"Euler 左右态必须 rho>0 且 p>0，得到 {wl} / {wr}")
    c_l = float(sound_speed(rho_l, p_l, g))
    c_r = float(sound_speed(rho_r, p_r, g))
    f0 = pressure_function(0.0, wl, g) + pressure_function(0.0, wr, g) + (u_r - u_l)
    if f0 > 0.0:
        # 两侧稀疏波把中间抽成真空：p_*=0，速度区间留空。
        return EulerStar(0.0, 0.5 * (u_l + u_r), 0.0, 0.0, "rarefaction", "rarefaction",
                         u_l - c_l, u_r + c_r, u_l - c_l, u_r + c_r, u_l - c_l, u_r + c_r,
                         True, 0, 0.0)
    p_lo = 1.0e-14 * max(p_l, p_r)
    p_hi = max(p_l, p_r, 1.0e-12)
    it = 0
    for it in range(int(max_iter)):
        val = pressure_function(p_hi, wl, g) + pressure_function(p_hi, wr, g) + (u_r - u_l)
        if val > 0.0:
            break
        p_hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (p_lo + p_hi)
        val = pressure_function(mid, wl, g) + pressure_function(mid, wr, g) + (u_r - u_l)
        if abs(p_hi - p_lo) <= float(tol) * max(p_hi, 1.0):
            p_lo = p_hi = mid
            break
        if val > 0.0:
            p_hi = mid
        else:
            p_lo = mid
    p_star = 0.5 * (p_lo + p_hi)
    u_star = 0.5 * (u_l + u_r) + 0.5 * (pressure_function(p_star, wr, g) - pressure_function(p_star, wl, g))
    g1 = (g - 1.0) / (g + 1.0)
    if p_star > p_l:
        left_wave = "shock"
        rho_star_l = rho_l * (p_star / p_l + g1) / (g1 * p_star / p_l + 1.0)
        s_left = u_l - c_l * np.sqrt((g + 1.0) / (2.0 * g) * p_star / p_l + (g - 1.0) / (2.0 * g))
        head_l = tail_l = float(s_left)
    else:
        left_wave = "rarefaction"
        rho_star_l = rho_l * (p_star / p_l) ** (1.0 / g)
        c_star_l = c_l * (p_star / p_l) ** ((g - 1.0) / (2.0 * g))
        head_l = u_l - c_l
        tail_l = u_star - float(c_star_l)
    if p_star > p_r:
        right_wave = "shock"
        rho_star_r = rho_r * (p_star / p_r + g1) / (g1 * p_star / p_r + 1.0)
        s_right = u_r + c_r * np.sqrt((g + 1.0) / (2.0 * g) * p_star / p_r + (g - 1.0) / (2.0 * g))
        head_r = tail_r = float(s_right)
    else:
        right_wave = "rarefaction"
        rho_star_r = rho_r * (p_star / p_r) ** (1.0 / g)
        c_star_r = c_r * (p_star / p_r) ** ((g - 1.0) / (2.0 * g))
        head_r = u_r + c_r
        tail_r = u_star + float(c_star_r)
    return EulerStar(
        float(p_star), float(u_star), float(rho_star_l), float(rho_star_r),
        left_wave, right_wave, float(head_l if left_wave == "shock" else tail_l),
        float(head_r if right_wave == "shock" else tail_r),
        float(head_l), float(tail_l), float(head_r), float(tail_r),
        False, int(it) + 1, float(abs(p_hi - p_lo)),
    )


def euler_sample(
    xi: Any,
    w_left: Sequence[float],
    w_right: Sequence[float],
    gamma: float = GAMMA,
    *,
    star: EulerStar | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""精确解在相似变量 :math:`\xi=x/t` 处的原始变量 :math:`(\rho,u,p)`。

    采样规则（Toro §4.9）按接触间断分成左右两半；每一半再看自己的波是激波还是
    稀疏波——**稀疏波内部用等熵关系** :math:`\rho=\rho_K(c/c_K)^{2/(\gamma-1)}`，
    激波后则用 RH 给出的常状态。跨过接触间断时 :math:`u`、:math:`p` 必须**连续**，
    而 :math:`\rho` 允许跳跃：这是画"解扇"时唯一要盯住的结构。
    """
    st = euler_star_state(w_left, w_right, gamma) if star is None else star
    g = float(gamma)
    rho_l, u_l, p_l = (float(v) for v in w_left)
    rho_r, u_r, p_r = (float(v) for v in w_right)
    c_l = float(sound_speed(rho_l, p_l, g))
    c_r = float(sound_speed(rho_r, p_r, g))
    xs = np.asarray(xi, dtype=float)
    shape = xs.shape
    rho = np.empty(shape, dtype=float)
    u = np.empty(shape, dtype=float)
    p = np.empty(shape, dtype=float)
    left = xs <= st.u_star
    # ---------------- 左半边 ----------------
    if st.left_wave == "shock":
        m_wl = left & (xs < st.tail_l)
        m_star = left & ~m_wl
    else:
        m_wl = left & (xs < st.head_l)
        m_star = left & (xs > st.tail_l)
        m_fan = left & ~m_wl & ~m_star
        # 扇面内 xi = u - c（左稀疏波的 C- 特征线），沿 C+ 特征 J_+ = u + 2c/(g-1) 守恒；
        # 两式联立给出 u（含 +xi）与 c（含 -xi）——**两个式子必须分开写**，
        # 曾经把 c 也写成 u 的那一支，结果整条扇面的密度/压强全错，
        # 而"看起来仍然像一条稀疏波"（只有与细网格数值解对拍才会露馅）。
        u[m_fan] = 2.0 / (g + 1.0) * (c_l + 0.5 * (g - 1.0) * u_l + xs[m_fan])
        cc = 2.0 / (g + 1.0) * (c_l + 0.5 * (g - 1.0) * (u_l - xs[m_fan]))
        rho[m_fan] = rho_l * (cc / c_l) ** (2.0 / (g - 1.0))
        p[m_fan] = p_l * (cc / c_l) ** (2.0 * g / (g - 1.0))
    rho[m_wl], u[m_wl], p[m_wl] = rho_l, u_l, p_l
    rho[m_star], u[m_star], p[m_star] = st.rho_star_l, st.u_star, st.p_star
    # ---------------- 右半边 ----------------
    right = ~left
    if st.right_wave == "shock":
        m_wr = right & (xs > st.tail_r)
        m_star_r = right & ~m_wr
    else:
        m_wr = right & (xs > st.head_r)
        m_star_r = right & (xs < st.tail_r)
        m_fan_r = right & ~m_wr & ~m_star_r
        # 右稀疏波：xi = u + c，沿 C- 特征 J_- = u - 2c/(g-1) 守恒（与左扇面镜像）。
        u[m_fan_r] = 2.0 / (g + 1.0) * (-c_r + 0.5 * (g - 1.0) * u_r + xs[m_fan_r])
        cc = 2.0 / (g + 1.0) * (c_r + 0.5 * (g - 1.0) * (xs[m_fan_r] - u_r))
        rho[m_fan_r] = rho_r * (cc / c_r) ** (2.0 / (g - 1.0))
        p[m_fan_r] = p_r * (cc / c_r) ** (2.0 * g / (g - 1.0))
    rho[m_wr], u[m_wr], p[m_wr] = rho_r, u_r, p_r
    rho[m_star_r], u[m_star_r], p[m_star_r] = st.rho_star_r, st.u_star, st.p_star
    return rho, u, p


def euler_exact(
    x: Any,
    t: float,
    w_left: Sequence[float],
    w_right: Sequence[float],
    gamma: float = GAMMA,
    *,
    t_floor: float = 1.0e-12,
) -> dict[str, Any]:
    r"""把精确 Riemann 解撒到坐标 ``x`` 上，返回 ``rho/u/p`` 与**掩码**。

    ``t <= 0`` 时自相似变量没有定义，函数**不抛异常也不返回 NaN**：时刻被抬到
    ``t_floor``，同时 ``meta`` 里 ``t_clamped=True``、``t_floor`` 与 ``valid_mask``
    一并返回——帧里"查无此值"必须是一条可显示的说明，而不是白屏或静默的 0。
    """
    t_in = float(t)
    clamped = bool(t_in <= 0.0)
    tt = max(t_in, float(t_floor))
    x0 = 0.5 * (float(np.min(np.asarray(x))) + float(np.max(np.asarray(x))))
    xi = (np.asarray(x, dtype=float) - x0) / tt
    st = euler_star_state(w_left, w_right, gamma)
    rho, u, p = euler_sample(xi, w_left, w_right, gamma, star=st)
    valid = np.ones(np.asarray(x).shape, dtype=float)
    if clamped:
        valid = np.zeros(np.asarray(x).shape, dtype=float)
    return {
        "x": np.asarray(x, dtype=float),
        "rho": rho,
        "u": u,
        "p": p,
        "xi": xi,
        "t": tt,
        "t_clamped": clamped,
        "valid_mask": valid,
        "star": st,
        "vacuum": bool(st.vacuum),
    }


def euler_rh_residuals(
    w_left: Sequence[float], w_right: Sequence[float], speed: float, gamma: float = GAMMA
) -> dict[str, float]:
    r"""跨激波的 RH 跳跃残差（质量 / 动量 / 能量），按跳跃尺度归一。

    :math:`s[U]=[F]`：三项分别算 :math:`|s\Delta\rho-\Delta(\rho u)|`、
    :math:`|s\Delta(\rho u)-\Delta(\rho u^2+p)|`、
    :math:`|s\Delta E-\Delta(u(E+p))|`，各自除以参与项的尺度。
    一个**真实**的激波这三项都该贴舍入；非物理的"膨胀激波"里至少有一项会
    :math:`O(1)` 地破——但**注意**：膨胀激波在标量情形下 RH 也是满足的，
    所以 RH 残差不能当熵条件的判据（这是本章最容易被写成假证据的一处）。
    """
    wl = tuple(float(v) for v in w_left)
    wr = tuple(float(v) for v in w_right)
    ul = euler_state(*wl, gamma)
    ur = euler_state(*wr, gamma)
    fl = euler_flux(ul, gamma)
    fr = euler_flux(ur, gamma)
    s = float(speed)
    scale = np.maximum(np.abs(ul) + np.abs(ur), 1.0e-30)
    resid = np.abs(s * (ur - ul) - (fr - fl)) / scale
    return {
        "mass": float(resid[0]),
        "momentum": float(resid[1]),
        "energy": float(resid[2]),
        "max": float(np.max(resid)),
    }


def riemann_invariant(u: Any, c: Any, gamma: float = GAMMA, *, sign: float = 1.0) -> np.ndarray:
    r"""Riemann 不变量 :math:`J_\pm=u\pm\frac{2c}{\gamma-1}`（稀疏波内逐点守恒）。"""
    return np.asarray(u, dtype=float) + float(sign) * 2.0 * np.asarray(c, dtype=float) / (float(gamma) - 1.0)


def euler_entropy(rho: Any, p: Any, gamma: float = GAMMA) -> np.ndarray:
    r"""热力学熵（去掉常数）：:math:`p/\rho^\gamma`——跨激波**增大**、跨稀疏波不变。"""
    r = np.maximum(np.asarray(rho, dtype=float), _SMALL)
    return np.asarray(p, dtype=float) / r ** float(gamma)


def hllc_flux(U_left: Any, U_right: Any, gamma: float = GAMMA) -> np.ndarray:
    r"""HLLC 数值通量（Toro）：两条声波 + 中间接触波，接触间断两侧各补一个星区状态。

    与精确 Godunov 的差别是**可测的**：接触间断被精确捕捉（不抹宽），但稀疏波
    仍被两条常波速近似，所以与精确解的 L1 偏差比"抹宽"要小、却不为零。
    这里用 Davis 波速估计 :math:`S_L=\min(u_L-c_L,u_R-c_R)`、
    :math:`S_R=\max(u_L+c_L,u_R+c_R)`——**故意用最朴素的一档**，因为本章要把
    "波速估计的选择"留成一个可见的旋钮。
    """
    ul = np.asarray(U_left, dtype=float)
    ur = np.asarray(U_right, dtype=float)
    rl, vl, pl = euler_primitives(ul, gamma)
    rr, vr, pr = euler_primitives(ur, gamma)
    cl = sound_speed(rl, pl, gamma)
    cr = sound_speed(rr, pr, gamma)
    fl = euler_flux(ul, gamma)
    fr = euler_flux(ur, gamma)
    sl = np.minimum(vl - cl, vr - cr)
    sr = np.maximum(vl + cl, vr + cr)
    with np.errstate(divide="ignore", invalid="ignore"):
        num = pr - pl + rl * vl * (sl - vl) - rr * vr * (sr - vr)
        den = rl * (sl - vl) - rr * (sr - vr)
        sm = num / _signed_guard(den)
    out = np.empty_like(ul)
    # 左星区
    dl = _signed_guard(sl - sm)
    rho_sl = rl * (sl - vl) / dl
    e_sl = (ul[..., 2] / rl + (sm - vl) * (sm + pl / (rl * _signed_guard(sl - vl)))) * rho_sl
    u_sl = np.stack([rho_sl, rho_sl * sm, e_sl], axis=-1)
    # 右星区
    dr = _signed_guard(sr - sm)
    rho_sr = rr * (sr - vr) / dr
    e_sr = (ur[..., 2] / rr + (sm - vr) * (sm + pr / (rr * _signed_guard(sr - vr)))) * rho_sr
    u_sr = np.stack([rho_sr, rho_sr * sm, e_sr], axis=-1)
    f_sl = fl + sl[..., None] * (u_sl - ul)
    f_sr = fr + sr[..., None] * (u_sr - ur)
    out = np.where((sl >= 0.0)[..., None], fl, out)
    out = np.where(((sl < 0.0) & (sm >= 0.0))[..., None], f_sl, out)
    out = np.where(((sm < 0.0) & (sr > 0.0))[..., None], f_sr, out)
    out = np.where((sr <= 0.0)[..., None], fr, out)
    return out


def rusanov_flux_euler(U_left: Any, U_right: Any, gamma: float = GAMMA) -> np.ndarray:
    r"""Euler 的局部 Lax--Friedrichs 通量（波速取两侧 :math:`|u|+c` 的最大值）。"""
    ul = np.asarray(U_left, dtype=float)
    ur = np.asarray(U_right, dtype=float)
    rl, vl, pl = euler_primitives(ul, gamma)
    rr, vr, pr = euler_primitives(ur, gamma)
    cl = sound_speed(rl, pl, gamma)
    cr = sound_speed(rr, pr, gamma)
    a = np.maximum(np.abs(vl) + cl, np.abs(vr) + cr)
    fl = euler_flux(ul, gamma)
    fr = euler_flux(ur, gamma)
    return 0.5 * (fl + fr) - 0.5 * a[..., None] * (ur - ul)


def euler_numerical_flux(U_left: Any, U_right: Any, scheme: str = "hllc", gamma: float = GAMMA) -> np.ndarray:
    r"""按名字分发 Euler 数值通量（``hllc`` / ``rusanov`` / ``godunov``）。

    ``godunov`` 逐界面调用精确 Riemann 解（慢，但它是"精确"的参照物，
    只在测试与小网格上用）。
    """
    key = str(scheme).strip().lower()
    if key == "hllc":
        return hllc_flux(U_left, U_right, gamma)
    if key == "rusanov":
        return rusanov_flux_euler(U_left, U_right, gamma)
    if key == "godunov":
        ul = np.asarray(U_left, dtype=float)
        ur = np.asarray(U_right, dtype=float)
        flat_l = ul.reshape(-1, 3)
        flat_r = ur.reshape(-1, 3)
        out = np.empty_like(flat_l)
        for i in range(flat_l.shape[0]):
            wl = tuple(float(v) for v in euler_primitives(flat_l[i], gamma))
            wr = tuple(float(v) for v in euler_primitives(flat_r[i], gamma))
            st = euler_star_state(wl, wr, gamma)
            rho, v, p = euler_sample(np.asarray(0.0), wl, wr, gamma, star=st)
            out[i] = euler_state(float(rho), float(v), float(p), gamma)
        return euler_flux(out.reshape(np.asarray(U_left).shape), gamma)
    raise ValueError(f"未知 Euler 通量格式 {scheme!r}（可用：hllc / rusanov / godunov）")


def problem(name: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    r"""按名字取 :data:`EULER_PROBLEMS` 里的标准考题（未知名字即报错，不猜）。"""
    key = str(name).strip().lower()
    if key not in EULER_PROBLEMS:
        raise ValueError(f"未知 Euler Riemann 考题 {name!r}（可用：{sorted(EULER_PROBLEMS)}）")
    left, right = EULER_PROBLEMS[key]
    return tuple(left), tuple(right)  # type: ignore[return-value]


def problem_names() -> tuple[str, ...]:
    r"""可用的标准考题名（前端下拉框与断言共用同一份真源）。"""
    return tuple(sorted(EULER_PROBLEMS))


def _as_mapping(d: Mapping[str, Any] | None) -> dict[str, Any]:
    """小工具：把可选映射折成普通字典（供测试与 server 层共用）。"""
    return dict(d or {})
