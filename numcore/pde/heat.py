r"""一维热传导方程的时间推进（FTCS / BTCS / Crank--Nicolson / theta 族）与二维 ADI。

数学定义
--------
一维抛物型方程

.. math::
    u_t = \alpha\,u_{xx} + f(x), \qquad x \in (0, L),\ t > 0,

空间用二阶中心差分 :math:`D u_j = (u_{j+1} - 2u_j + u_{j-1})/\Delta x^2`，
时间用 theta 族（:math:`\theta \in [0,1]`）

.. math::
    \frac{u^{n+1} - u^n}{\Delta t}
    = \theta\,(\alpha D u^{n+1} + f) + (1-\theta)\,(\alpha D u^{n} + f).

三个常用成员是它的特例：`ftcs` 取 :math:`\theta = 0`（显式，条件稳定）、
`btcs` 取 :math:`\theta = 1`（隐式，无条件稳定、一阶）、
`cn` 取 :math:`\theta = 1/2`（Crank--Nicolson，无条件稳定、二阶）。
记格点比 :math:`r = \alpha \Delta t/\Delta x^2`，则一步的线性系统是

.. math::
    \bigl(I - \theta r\,\tilde D\bigr) u^{n+1} = \bigl(I + (1-\theta) r\,\tilde D\bigr) u^{n} + \Delta t\,f,

其中 :math:`\tilde D` 是"以 :math:`\Delta x^2` 为单位"的二阶差分算子
（:func:`laplacian_1d`），边界条件按下面的约定折进它。

边界条件
--------
三种边界都用**同一批格点**当未知量（Dirichlet 的端点也留在向量里），
于是"解一个 :math:`n \times n` 三对角系统"就够了，不必为每种边界写一套索引算术：

* ``dirichlet``：端点行的方程直接写成 :math:`u_0 = u_L`、:math:`u_{n-1} = u_R`（**恒等行**）。
  恒等行会带来两个恰好等于 1 的特征值，做稳定性测量时必须把它们投影掉——
  这是本章最容易造出假证据的地方，见 :func:`step_growth` 的说明。
* ``neumann``：用幽灵点消元。端点 :math:`u_x(0) = q_0` 给出
  :math:`\tilde D u|_0 = 2(u_1 - u_0) - 2\Delta x\,q_0`，
  即"对角 2、次对角 -2"的边界行；右端同理。端点行的行和为零——
  **这正是离散总热量守恒的原因**（见 :func:`conservation_defect`）。
* ``periodic``：网格取 :math:`x_j = jL/n`（不是 :math:`L/(n-1)`），
  系统变成**循环三对角**，用 Sherman--Morrison 公式化成两次追赶法（:func:`cyclic_apply`）。

实现约束
--------
三对角求解直接复用 2.2 章的追赶法 :func:`numcore.interp.spline.tridiagonal_apply`
（单一真源，长度 n 的约定：``lower[0]`` 与 ``upper[-1]`` 是矩阵外的位置，恒为 0）。
本模块只用 numpy 的容器与逐元素原语，**不调用**任何现成求解器（POLICY P2/P3）。
随机数只经 :func:`numcore.utils.rng`（P4）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from numcore.interp.spline import tridiagonal_apply
from numcore.utils import rng

BC = Literal["dirichlet", "neumann", "periodic"]
Scheme = Literal["ftcs", "btcs", "cn", "theta"]

__all__ = [
    "BC",
    "Scheme",
    "SCHEME_THETA",
    "HeatGrid",
    "HeatResult",
    "HeatTrajectory",
    "uniform_grid",
    "scheme_theta",
    "diffusivity",
    "r_from_alpha",
    "laplacian_1d",
    "theta_matrix",
    "theta_system",
    "theta_step",
    "step_residual",
    "cyclic_apply",
    "steady_state_1d",
    "heat_solve",
    "heat_trajectory",
    "initial_profile",
    "exact_dirichlet_sine",
    "exact_neumann_cos",
    "exact_periodic_cos",
    "exact_steady_source",
    "exact_source_series",
    "discrete_sine_eigenvalue",
    "exact_semidiscrete_sine",
    "semidiscrete_sine_2d",
    "error_norms",
    "conservation_defect",
    "ftcs_step",
    "explicit_growth",
    "step_growth",
    "heat2d_grid",
    "heat2d_exact_sine",
    "heat2d_explicit_step",
    "heat2d_adi_step",
    "heat2d_growth",
]


#: scheme 名 -> theta 权重（``theta`` 需要调用方另外给 :math:`\theta`）
SCHEME_THETA: dict[str, float] = {"ftcs": 0.0, "btcs": 1.0, "cn": 0.5}


# --------------------------------------------------------------------------- #
# 网格与基本算子
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HeatGrid:
    """一维均匀网格：``x`` 是格点，``dx`` 是间距，``periodic`` 决定端点是否重合。"""

    x: np.ndarray
    dx: float
    periodic: bool = False

    @property
    def nx(self) -> int:
        """格点数 :math:`n`（周期网格上 ``x[0]`` 与 ``x[n-1]`` 相差 :math:`\\Delta x`）。"""
        return int(self.x.size)

    @property
    def length(self) -> float:
        """区间长度 :math:`L`（周期网格上是 ``n·dx``）。"""
        return float(self.dx * (self.nx if self.periodic else self.nx - 1))


def uniform_grid(nx: int, length: float = 1.0, x0: float = 0.0, *, periodic: bool = False) -> HeatGrid:
    """均匀网格：非周期取 ``linspace(x0, x0+length, nx)``，周期取 ``x0 + j·length/nx``。"""
    if nx < 3:
        raise ValueError(f"nx 至少 3，得到 {nx}")
    if length <= 0.0:
        raise ValueError(f"length 必须为正，得到 {length}")
    if periodic:
        dx = length / nx
        x = x0 + dx * np.arange(nx, dtype=float)
    else:
        dx = length / (nx - 1)
        x = np.linspace(x0, x0 + length, nx)
    return HeatGrid(x=x, dx=float(dx), periodic=bool(periodic))


def scheme_theta(scheme: str, theta: float | None = None) -> float:
    """把格式名翻成 :math:`\\theta` 权重；``scheme="theta"`` 时必须显式给 ``theta``。"""
    name = str(scheme).lower()
    if name == "theta":
        if theta is None:
            raise ValueError("scheme='theta' 需要显式给出 theta")
        val = float(theta)
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"theta 必须在 [0,1] 内，得到 {val}")
        return val
    if name not in SCHEME_THETA:
        raise ValueError(f"未知格式 {scheme!r}（可用：{sorted(SCHEME_THETA)} 或 'theta'）")
    if theta is not None:
        raise ValueError(f"scheme={name!r} 的 theta 固定为 {SCHEME_THETA[name]}，不能再指定")
    return SCHEME_THETA[name]


def diffusivity(r: float, dx: float, dt: float) -> float:
    """由格点比反解扩散系数 :math:`\\alpha = r\\,\\Delta x^2/\\Delta t`。"""
    if dt <= 0.0:
        raise ValueError(f"dt 必须为正，得到 {dt}")
    return float(r) * float(dx) ** 2 / float(dt)


def r_from_alpha(alpha: float, dt: float, dx: float) -> float:
    """格点比 :math:`r = \\alpha\\,\\Delta t/\\Delta x^2`。"""
    if dx <= 0.0:
        raise ValueError(f"dx 必须为正，得到 {dx}")
    return float(alpha) * float(dt) / float(dx) ** 2


def laplacian_1d(u: Any, dx: float, *, bc: str = "dirichlet", left: float = 0.0, right: float = 0.0) -> np.ndarray:
    r"""以 :math:`\Delta x^2` 为单位的二阶中心差分 :math:`\tilde D u`。

    参数 ``left`` / ``right`` 的含义随 ``bc`` 改变（这是本模块唯一"同名不同义"的地方，
    刻意写成函数签名上的显式参数，避免调用方猜）：

    * ``dirichlet``：两个**端点值** :math:`(u_L, u_R)`，只参与内部行的差分；
      端点行本身返回 0（它们由恒等行接管，见 :func:`theta_matrix`）。
    * ``neumann``：两个**端点导数** :math:`(u_x(0), u_x(L))`，用幽灵点消元进入端点行。
    * ``periodic``：忽略。

    返回与 ``u`` 同形的数组。
    """
    arr = np.asarray(u, dtype=float)
    n = arr.size
    if n < 3:
        raise ValueError(f"laplacian_1d 需要至少 3 个格点，得到 {n}")
    if dx <= 0.0:
        raise ValueError(f"dx 必须为正，得到 {dx}")
    out = np.zeros(n, dtype=float)
    if bc == "dirichlet":
        out[1:-1] = arr[2:] - 2.0 * arr[1:-1] + arr[:-2]
    elif bc == "neumann":
        out[1:-1] = arr[2:] - 2.0 * arr[1:-1] + arr[:-2]
        out[0] = 2.0 * (arr[1] - arr[0]) - 2.0 * dx * float(left)
        out[-1] = 2.0 * (arr[-2] - arr[-1]) + 2.0 * dx * float(right)
    elif bc == "periodic":
        out = np.roll(arr, -1) - 2.0 * arr + np.roll(arr, 1)
    else:
        raise ValueError(f"未知边界条件 {bc!r}（可用：dirichlet / neumann / periodic）")
    return out


def _check_bc(bc: str, nx: int) -> str:
    name = str(bc).lower()
    if name not in ("dirichlet", "neumann", "periodic"):
        raise ValueError(f"未知边界条件 {bc!r}（可用：dirichlet / neumann / periodic）")
    if name == "periodic" and nx < 3:
        raise ValueError(f"周期边界至少 3 个格点，得到 {nx}")
    return name


def cyclic_apply(lower: Any, diag: Any, upper: Any, corner: float, rhs: Any) -> np.ndarray:
    r"""循环三对角系统 :math:`(T + \alpha(e_0e_{n-1}^{\mathsf T} + e_{n-1}e_0^{\mathsf T}))x = b`。

    用 Sherman--Morrison 公式化成**两次追赶法**（:math:`\gamma = -d_0`）：

    .. math::
        T' = T - \gamma e_0e_0^{\mathsf T} - \frac{\alpha^2}{\gamma}e_{n-1}e_{n-1}^{\mathsf T},
        \qquad u = \gamma e_0 + \alpha e_{n-1},\quad v = e_0 + \frac{\alpha}{\gamma}e_{n-1},

    于是 :math:`A = T' + uv^{\mathsf T}`，:math:`x = T'^{-1}b - \frac{v^{\mathsf T}T'^{-1}b}{1 + v^{\mathsf T}T'^{-1}u}\,T'^{-1}u`。
    分母为零（理论上只在极端构造下发生）时抛 :class:`ZeroDivisionError`，不静默返回垃圾。
    """
    lo = np.asarray(lower, dtype=float).ravel().copy()
    dg = np.asarray(diag, dtype=float).ravel().copy()
    up = np.asarray(upper, dtype=float).ravel().copy()
    b = np.asarray(rhs, dtype=float).ravel()
    n = dg.size
    if n < 3:
        raise ValueError(f"循环三对角系统至少 3 阶，得到 {n}")
    if lo.size != n or up.size != n or b.size != n:
        raise ValueError(f"三条对角线与右端项长度必须一致：{lo.size}/{dg.size}/{up.size}/{b.size}")
    alpha = float(corner)
    gamma = -dg[0]
    if gamma == 0.0:  # pragma: no cover - dg[0] > 0 在 theta 族里恒成立
        raise ZeroDivisionError("循环三对角：gamma = -diag[0] 为零，Sherman-Morrison 不可用")
    dg[0] = dg[0] - gamma
    dg[-1] = dg[-1] - alpha * alpha / gamma
    lo[0] = 0.0
    up[-1] = 0.0
    x0 = tridiagonal_apply(lo, dg, up, b)
    u_vec = np.zeros(n, dtype=float)
    u_vec[0] = gamma
    u_vec[-1] = alpha
    z = tridiagonal_apply(lo, dg, up, u_vec)
    denom = 1.0 + z[0] + alpha * z[-1] / gamma
    if denom == 0.0:  # pragma: no cover - 只在病态构造下发生
        raise ZeroDivisionError("循环三对角：Sherman-Morrison 分母为零")
    fact = (x0[0] + alpha * x0[-1] / gamma) / denom
    return x0 - fact * z


def _solve(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, corner: float, rhs: np.ndarray) -> np.ndarray:
    """按 ``corner`` 是否为零选择追赶法或循环追赶法。"""
    if corner == 0.0:
        return tridiagonal_apply(lower, diag, upper, rhs)
    return cyclic_apply(lower, diag, upper, corner, rhs)


# --------------------------------------------------------------------------- #
# theta 族一步
# --------------------------------------------------------------------------- #
def theta_matrix(nx: int, r: float, theta: float, *, bc: str = "dirichlet") -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    r"""theta 族左端矩阵 :math:`I - \theta r\tilde D` 的三条对角线（+ 循环角元）。

    返回 ``(lower, diag, upper, corner)``，长度为 ``nx``；``lower[0]`` 与 ``upper[-1]``
    是矩阵外的位置，恒为 0（与 :func:`numcore.interp.spline.tridiagonal_apply` 的约定一致）。
    :math:`\tilde D` 的对角是 -2、次/超对角是 1，于是

    * 内部行：对角 :math:`1 + 2\theta r`、次/超对角 :math:`-\theta r`；
    * Dirichlet 端点行：恒等行 ``(1, 0)``（端点值被方程直接钉住）；
    * Neumann 端点行：对角 :math:`1 + 2\theta r`、内邻 :math:`-2\theta r`（幽灵点消元的结果）；
    * 周期：角元 :math:`-\theta r`。

    注意 Dirichlet 的恒等行**不是**精度损失，但它给迭代矩阵添了两个恰好等于 1 的特征值——
    做谱半径测量时必须投影掉（:func:`step_growth`）。
    """
    n = int(nx)
    if n < 3:
        raise ValueError(f"nx 至少 3，得到 {n}")
    if r < 0.0:
        raise ValueError(f"r 不能为负，得到 {r}")
    name = _check_bc(bc, n)
    lo = np.zeros(n, dtype=float)
    dg = np.full(n, 1.0 + 2.0 * theta * r, dtype=float)
    up = np.zeros(n, dtype=float)
    lo[1:] = -theta * r
    up[:-1] = -theta * r
    corner = 0.0
    if name == "dirichlet":
        dg[0] = 1.0
        up[0] = 0.0
        dg[-1] = 1.0
        lo[-1] = 0.0
    elif name == "neumann":
        up[0] = -2.0 * theta * r
        lo[-1] = -2.0 * theta * r
    else:
        corner = -theta * r
    return lo, dg, up, float(corner)


def theta_system(
    u: Any,
    *,
    r: float,
    theta: float,
    dx: float,
    dt: float,
    bc: str = "dirichlet",
    left: float = 0.0,
    right: float = 0.0,
    source: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    r"""一步 theta 格式的线性系统 ``(lower, diag, upper, corner, rhs)``。

    ``source`` 是**同一时刻的源项** :math:`f`（数组或标量），进入右端的方式是
    :math:`\Delta t\,f`；对定常源项这与"theta 加权"完全一致，对随时间变化的源项
    则等价于取 :math:`\theta` 加权时刻的值。
    """
    arr = np.asarray(u, dtype=float).ravel()
    n = arr.size
    if n < 3:
        raise ValueError(f"theta_system 需要至少 3 个格点，得到 {n}")
    if dt < 0.0:
        raise ValueError(f"dt 不能为负，得到 {dt}")
    name = _check_bc(bc, n)
    lo, dg, up, corner = theta_matrix(n, r, theta, bc=name)
    rhs = arr + (1.0 - theta) * r * laplacian_1d(arr, dx, bc=name, left=left, right=right)
    if source is not None:
        rhs = rhs + dt * np.broadcast_to(np.asarray(source, dtype=float), (n,))
    if name == "dirichlet":
        rhs = rhs.copy()
        rhs[0] = float(left)
        rhs[-1] = float(right)
    elif name == "neumann":
        # 幽灵点上的通量项在左右两端各出现一次，theta 权重合计为 1；
        # 显式部分已由 laplacian_1d 带进 -(1-θ)·2r·dx·q，这里补上剩下的 -θ 部分。
        rhs = rhs.copy()
        rhs[0] -= 2.0 * theta * r * dx * float(left)
        rhs[-1] += 2.0 * theta * r * dx * float(right)
    return lo, dg, up, corner, rhs


def theta_step(
    u: Any,
    *,
    r: float,
    theta: float,
    dx: float,
    dt: float,
    bc: str = "dirichlet",
    left: float = 0.0,
    right: float = 0.0,
    source: Any = None,
) -> np.ndarray:
    r"""theta 族推进一步：解 :math:`(I-\theta r\tilde D)u^{n+1} = (I+(1-\theta)r\tilde D)u^n + \Delta t f`。"""
    lo, dg, up, corner, rhs = theta_system(
        u, r=r, theta=theta, dx=dx, dt=dt, bc=bc, left=left, right=right, source=source
    )
    return _solve(lo, dg, up, corner, rhs)


def step_residual(
    u_new: Any,
    u_old: Any,
    *,
    r: float,
    theta: float,
    dx: float,
    dt: float,
    bc: str = "dirichlet",
    left: float = 0.0,
    right: float = 0.0,
    source: Any = None,
) -> float:
    """把一步的解代回离散方程，返回**规范化的后向残差**（解是否满足离散方程本身）。

    与"截断误差"完全是两件事：显式格式的残差恒为舍入级（右端就是逐点算出来的），
    隐式格式的残差反映追赶法的回代误差。用它可以一眼看出"隐式格式算得准不准"与
    "隐式格式离真解有多远"是两回事。
    """
    lo, dg, up, corner, rhs = theta_system(
        u_old, r=r, theta=theta, dx=dx, dt=dt, bc=bc, left=left, right=right, source=source
    )
    x = np.asarray(u_new, dtype=float).ravel()
    if x.size != dg.size:
        raise ValueError(f"解的长度 {x.size} 与阶数 {dg.size} 不一致")
    # A x 的分量自己算（含循环角元），这样三种边界走同一条代码路径，
    # 不会出现"周期情形下残差忘了算角元"这种只在一种边界上撒谎的判据。
    ax = dg * x
    ax[1:] = ax[1:] + lo[1:] * x[:-1]
    ax[:-1] = ax[:-1] + up[:-1] * x[1:]
    if corner != 0.0:
        ax[0] = ax[0] + corner * x[-1]
        ax[-1] = ax[-1] + corner * x[0]
    num = float(np.max(np.abs(ax - rhs))) if x.size else 0.0
    row_sums = np.abs(dg) + np.abs(lo) + np.abs(up)
    if corner != 0.0:
        row_sums[0] = row_sums[0] + abs(corner)
        row_sums[-1] = row_sums[-1] + abs(corner)
    scale = float(np.max(row_sums)) * float(np.max(np.abs(x))) + float(np.max(np.abs(rhs)))
    return float(num / scale) if scale > 0.0 else float(num)


# --------------------------------------------------------------------------- #
# 稳态与显式一步（含变系数）
# --------------------------------------------------------------------------- #
def steady_state_1d(
    nx: int,
    *,
    source: Any,
    bc: str = "dirichlet",
    left: float = 0.0,
    right: float = 0.0,
    length: float = 1.0,
    dx: float | None = None,
) -> np.ndarray:
    r"""离散稳态 :math:`-u'' = f`：三对角系统 :math:`(-\tilde D)u = \Delta x^2 f`。

    与时间推进共用同一套追赶法，因此"稳态"与"长时间推进的极限"可以在同一精度口径下比较。
    Dirichlet 用恒等行；Neumann 端点行对应对角 2、内邻 -2 再加通量项；
    周期用循环追赶法。返回长度 ``nx`` 的解向量。
    """
    n = int(nx)
    if n < 3:
        raise ValueError(f"nx 至少 3，得到 {n}")
    name = _check_bc(bc, n)
    if dx is None:
        dx = length / n if name == "periodic" else length / (n - 1)
    if dx <= 0.0:
        raise ValueError(f"dx 必须为正，得到 {dx}")
    f = np.broadcast_to(np.asarray(source, dtype=float), (n,))
    lo = np.full(n, -1.0)
    dg = np.full(n, 2.0)
    up = np.full(n, -1.0)
    corner = 0.0
    rhs = dx * dx * np.array(f, dtype=float)
    if name == "dirichlet":
        lo[0] = 0.0
        up[0] = 0.0
        dg[0] = 1.0
        dg[-1] = 1.0
        lo[-1] = 0.0
        up[-1] = 0.0
        rhs[0] = float(left)
        rhs[-1] = float(right)
    elif name == "neumann":
        lo[0] = 0.0
        up[0] = -2.0
        dg[-1] = 2.0
        lo[-1] = -2.0
        up[-1] = 0.0
        rhs[0] = rhs[0] - 2.0 * dx * float(left)
        rhs[-1] = rhs[-1] + 2.0 * dx * float(right)
    else:
        lo[0] = 0.0
        up[-1] = 0.0
        corner = -1.0
    return _solve(lo, dg, up, corner, rhs)


def ftcs_step(
    u: Any,
    lam: Any,
    *,
    bc: str = "dirichlet",
    dx: float = 1.0,
    left: float = 0.0,
    right: float = 0.0,
) -> np.ndarray:
    r"""显式一步 :math:`u_j \leftarrow u_j + \lambda_j(u_{j+1}-2u_j+u_{j-1})`。

    ``lam`` 可以是标量（定常系数，:math:`\lambda = r`）或**逐点数组**
    （冻结系数的变系数问题 :math:`\lambda_j = \alpha_j \Delta t/\Delta x^2`）。
    局部稳定条件是 :math:`\max_j \lambda_j \le 1/2`——注意判据里出现的是**最大值**，
    "把平均系数代进 1/2"是本章最经典的假判据之一（变系数断言里实测过）。

    Neumann 边界下 ``left`` / ``right`` 是端点导数（需要 ``dx`` 才能折成流量项）。
    """
    arr = np.asarray(u, dtype=float).ravel()
    n = arr.size
    name = _check_bc(bc, n)
    lam_arr = np.broadcast_to(np.asarray(lam, dtype=float), (n,))
    if np.any(lam_arr < 0.0):
        raise ValueError("lam 不能为负（反扩散会无条件不稳定）")
    if name == "periodic":
        return arr + lam_arr * (np.roll(arr, -1) - 2.0 * arr + np.roll(arr, 1))
    out = arr.copy()
    out[1:-1] = arr[1:-1] + lam_arr[1:-1] * (arr[2:] - 2.0 * arr[1:-1] + arr[:-2])
    if name == "neumann":
        out[0] = arr[0] + lam_arr[0] * (2.0 * (arr[1] - arr[0]) - 2.0 * dx * float(left))
        out[-1] = arr[-1] + lam_arr[-1] * (2.0 * (arr[-2] - arr[-1]) + 2.0 * dx * float(right))
    else:
        out[0] = float(left)
        out[-1] = float(right)
    return out


# --------------------------------------------------------------------------- #
# 初值与精确解
# --------------------------------------------------------------------------- #
def initial_profile(x: Any, kind: str = "sine", k: int = 1, *, amplitude: float = 1.0) -> np.ndarray:
    r"""初值剖面（**零 Dirichlet 数据**下的常用形状，``k`` 是模态数）。

    * ``sine``：:math:`\sin(k\pi x/L)`——Dirichlet 的特征函数（两端自动为零）；
    * ``cos``：:math:`\cos(k\pi x/L)`——Neumann 的特征函数（两端导数自动为零）；
    * ``cos_periodic``：:math:`\cos(2\pi k x/L)`——周期网格的特征函数。
      注意周期网格的 ``x`` 到 ``L-Δx`` 为止，所以周期取 ``span+Δx`` 而不是 ``span``；
    * ``hat``：三角帽 :math:`1-|2x/L-1|`，两端为零、处处连续但导数有折点，
      它会**同时激发所有模态**（用来观察短波最先被耗散掉）；
    * ``noise``：确定性伪随机扰动（:func:`numcore.utils.rng`），两端强制为零。
    """
    arr = np.asarray(x, dtype=float)
    span = float(arr[-1] - arr[0]) if arr.size >= 2 else 1.0
    if span <= 0.0:
        span = 1.0
    if kind == "sine":
        return amplitude * np.sin(k * np.pi * (arr - arr[0]) / span)
    if kind == "cos":
        return amplitude * np.cos(k * np.pi * (arr - arr[0]) / span)
    if kind == "cos_periodic":
        # 周期网格不含右端点（x = j·L/n），span 要补上一个 Δx 才是周期的**周期**。
        # 少补这一项会得到 cos(2πx/span)，k=1、n=41 时波数偏差 2.5%，
        # 初值与周期边界**不相容**（u[n-1] ≠ u[0]），实测 L∞ 误差被抬到 3.6e-2
        # ——比 btcs 自身的截断误差大两个数量级，会让人误判格式写错了。
        step = float(arr[1] - arr[0]) if arr.size >= 2 else 0.0
        return amplitude * np.cos(2.0 * np.pi * k * (arr - arr[0]) / (span + step))
    if kind == "hat":
        return amplitude * (1.0 - np.abs(2.0 * (arr - arr[0]) / span - 1.0))
    if kind == "noise":
        gen = rng(20261018)
        val = gen.standard_normal(arr.size)
        val = val - float(np.mean(val))
        if arr.size >= 2:
            val[0] = 0.0
            val[-1] = 0.0
        peak = float(np.max(np.abs(val))) if val.size else 0.0
        if peak > 0.0:
            val = val / peak
        return amplitude * val
    raise ValueError(f"未知初值类型 {kind!r}（可用：sine / cos / cos_periodic / hat / noise）")


def exact_dirichlet_sine(x: Any, t: float, *, k: int = 1, alpha: float = 1.0, length: float = 1.0) -> np.ndarray:
    r""":math:`u_t = \alpha u_{xx}`、:math:`u(0)=u(L)=0`、:math:`u(x,0)=\sin(k\pi x/L)` 的精确解。

    :math:`u = e^{-\alpha (k\pi/L)^2 t}\sin(k\pi x/L)`；两个端点**显式置零**
    （边界条件精确成立，而 ``sin(kπ)`` 在浮点下是 1e-16 量级的残渣——
    留着它会把误差场的量级整个污染掉）。
    """
    arr = np.asarray(x, dtype=float)
    u = np.sin(k * np.pi * arr / length) * float(np.exp(-alpha * (k * np.pi / length) ** 2 * t))
    if arr.size >= 2:
        u[0] = 0.0
        u[-1] = 0.0
    return u


def exact_neumann_cos(
    x: Any, t: float, *, k: int = 1, alpha: float = 1.0, length: float = 1.0, mean: float = 0.0
) -> np.ndarray:
    r"""Neumann 绝缘边界下 :math:`u(x,0)=\mathrm{mean}+\cos(k\pi x/L)` 的精确解。

    :math:`u = \mathrm{mean} + e^{-\alpha(k\pi/L)^2t}\cos(k\pi x/L)`：
    常数模态（平均值）**永远不衰减**——这正是"总热量守恒"的连续版本。
    """
    arr = np.asarray(x, dtype=float)
    return mean + np.cos(k * np.pi * arr / length) * float(np.exp(-alpha * (k * np.pi / length) ** 2 * t))


def exact_periodic_cos(x: Any, t: float, *, k: int = 1, alpha: float = 1.0, length: float = 1.0) -> np.ndarray:
    r"""周期边界下 :math:`u(x,0)=\cos(2\pi k x/L)` 的精确解 :math:`e^{-4\alpha(\pi k/L)^2t}\cos(2\pi kx/L)`。"""
    arr = np.asarray(x, dtype=float)
    return np.cos(2.0 * np.pi * k * arr / length) * float(
        np.exp(-4.0 * alpha * (np.pi * k / length) ** 2 * t)
    )


def discrete_sine_eigenvalue(nx: int, k: int = 1, *, alpha: float = 1.0, length: float = 1.0) -> float:
    r"""**离散** Dirichlet 二阶差分算子的特征值 :math:`\lambda_k = 4\alpha\sin^2(k\pi\Delta x/2L)/\Delta x^2`。

    :math:`\sin(k\pi x/L)` 在均匀网格上**恰好**是 :math:`\tilde D` 的特征向量
    （二阶中心差分对三角函数是精确的），所以半离散问题
    :math:`\dot v = \alpha\tilde D v` 的解是纯指数 :math:`e^{-\lambda_k t}\sin(k\pi x/L)`，
    **没有任何空间截断误差掺杂**。把它当靶子，测出来的就是干干净净的时间阶；
    与连续特征值 :math:`(k\pi/L)^2` 的差
    :math:`\lambda_k - (k\pi/L)^2 = \frac{\alpha k^4\pi^4}{12L^4}\Delta x^2 + O(\Delta x^4)`
    才是空间误差的来源（这也是"固定 r 加密网格时 FTCS 看起来也是二阶"的原因：
    :math:`\Delta t = r\Delta x^2/\alpha`，一阶时间误差被自动压成 :math:`O(\Delta x^2)`）。
    """
    n = int(nx)
    if n < 3:
        raise ValueError(f"nx 至少 3，得到 {n}")
    if not 1 <= int(k) <= n - 2:
        raise ValueError(f"模态 k 必须在 1..{n - 2} 之间（端点固定，可分辨模态只有 {n - 2} 个），得到 {k}")
    dx = length / (n - 1)
    s = float(np.sin(k * np.pi * dx / (2.0 * length)))
    return float(4.0 * alpha * s * s / (dx * dx))


def exact_semidiscrete_sine(
    x: Any, t: float, *, nx: int | None = None, k: int = 1, alpha: float = 1.0, length: float = 1.0
) -> np.ndarray:
    r"""半离散问题的精确解 :math:`e^{-\lambda_k t}\sin(k\pi x/L)`（:math:`\lambda_k` 见上）。

    这是"只离散空间、时间不离散"的靶子——时间推进格式的误差只能来自时间离散，
    用它做收敛阶的对照物，空间误差被**解析地**扣掉了，不会出现"空间误差盖住时间误差"
    导致拟合出 0 阶的经典事故。
    """
    arr = np.asarray(x, dtype=float)
    if nx is None:
        nx = int(arr.size)
    lam = discrete_sine_eigenvalue(nx, k, alpha=alpha, length=length)
    u = np.sin(k * np.pi * arr / length) * float(np.exp(-lam * t))
    if arr.size >= 2:
        u[0] = 0.0
        u[-1] = 0.0
    return u


def semidiscrete_mode_solution(n_steps: int, *, r: float, nx: int, k: int = 1, alpha: float = 1.0, length: float = 1.0) -> float:
    r"""单模态初值下数值解在半离散精确解处的**放大链** :math:`G(\varphi_k)^n`（解析值）。

    :math:`\sin(k\pi x/L)` 是离散算子的特征向量，所以 theta 格式的数值解逐点等于
    :math:`G(\varphi_k)^n\sin(k\pi x/L)`，其中 :math:`G` 由
    :func:`numcore.pde.vonneumann.amplification_theta` 给出。这个函数只算解析的幂，
    与"真的跑一遍格式"是两条独立路线——断言把两者对拍，就等于同时钉住了
    von Neumann 因子与时间推进实现。
    """
    if n_steps < 0:
        raise ValueError(f"n_steps 不能为负，得到 {n_steps}")
    return float("nan")  # 由 vonneumann 模块提供，见 numcore.pde.vonneumann.mode_amplification_power


def exact_steady_source(x: Any, f: float, *, left: float = 0.0, right: float = 0.0, length: float = 1.0) -> np.ndarray:
    r"""定常源项下 :math:`-u''=f`（Dirichlet）的稳态解 :math:`u = u_L + (u_R-u_L)x/L + \frac{f}{2}x(L-x)`。"""
    arr = np.asarray(x, dtype=float)
    return left + (right - left) * arr / length + 0.5 * f * arr * (length - arr)


def exact_source_series(x: Any, t: float, f: float, *, terms: int = 400, alpha: float = 1.0, length: float = 1.0) -> np.ndarray:
    r""":math:`u_t=u_{xx}+f`、零初值与零 Dirichlet 数据下的精确解（分离变量级数）。

    .. math::
        u(x,t) = \frac{f}{2}x(L-x) - \frac{4fL^3}{\pi^3}\sum_{m\ \mathrm{odd}}
                 \frac{\sin(m\pi x/L)}{m^3}\,e^{-\alpha(m\pi/L)^2t}.

    奇数模态的系数是 :math:`4fL^3/(m^3\pi^3)`（由 :math:`\int_0^L x(L-x)\sin(m\pi x/L)dx` 得到），
    级数按 :math:`m^{-3}` 衰减，``terms`` 取几百项即可到双精度。
    """
    arr = np.asarray(x, dtype=float)
    out = exact_steady_source(arr, f, length=length)
    m = np.arange(1, int(terms) + 1, 2, dtype=float)
    coef = 4.0 * f * length**3 / (np.pi**3 * m**3)
    decay = np.exp(-alpha * (m * np.pi / length) ** 2 * t)
    wave = np.sin(np.outer(m, arr) * np.pi / length)
    return out - (coef * decay) @ wave


def semidiscrete_sine_2d(
    x: Any, y: Any, t: float, *, nx: int, ny: int | None = None, kx: int = 1, ky: int = 1,
    alpha: float = 1.0, length: float = 1.0
) -> np.ndarray:
    r"""二维半离散精确解 :math:`e^{-(\lambda_{k_x}+\lambda_{k_y})t}\sin(k_x\pi x/L)\sin(k_y\pi y/L)`。

    二维张量积网格上 :math:`\sin(k_x\pi x/L)\sin(k_y\pi y/L)` 是离散五点 Laplacian 的
    精确特征向量（特征值是两个方向离散特征值之和），所以它**不含任何空间截断误差**。
    测二维格式的时间阶必须用它当靶子：拿连续解当靶子时 :math:`O(\Delta x^2)` 的空间误差
    会把时间阶完全盖住——实测 ADI 用连续解做靶子时拟合出的阶是 :math:`-0.0018`
    （dt 缩小 32 倍，误差纹丝不动），换成半离散解才是干净的 2。
    """
    xx = np.asarray(x, dtype=float)[:, None]
    yy = np.asarray(y, dtype=float)[None, :]
    if ny is None:
        ny = int(np.asarray(y).size)
    lam = discrete_sine_eigenvalue(nx, kx, alpha=alpha, length=length) + discrete_sine_eigenvalue(
        ny, ky, alpha=alpha, length=length
    )
    u = (
        np.sin(kx * np.pi * xx / length)
        * np.sin(ky * np.pi * yy / length)
        * float(np.exp(-lam * t))
    )
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    return u


def error_norms(u_num: Any, u_exact: Any, dx: float) -> tuple[float, float]:
    r"""返回 :math:`(\|e\|_\infty,\ \|e\|_2)`，其中 :math:`\|e\|_2 = \sqrt{\Delta x\sum e_j^2}`。"""
    a = np.asarray(u_num, dtype=float).ravel()
    b = np.asarray(u_exact, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError(f"两个解长度不一致：{a.size} vs {b.size}")
    e = a - b
    linf = float(np.max(np.abs(e))) if e.size else 0.0
    if not e.size:
        return linf, 0.0
    # 先按最大值缩放再平方求和：解发散到 1e80 时 e² 会溢出成 inf，
    # 而 sqrt(inf) 仍然是 inf——"误差很大"会被静默换成一个非有限值，
    # 而帧契约（allow_nan=False）不允许它出现在数组里。缩放后结果永远有限（除非 e 本身溢出）。
    scale = linf if linf > 0.0 else 1.0
    l2 = float(scale * np.sqrt(dx * float(np.sum((e / scale) ** 2))))
    return linf, l2


def conservation_defect(values: Any, dx: float) -> np.ndarray:
    r"""离散总热量 :math:`H(t) = \Delta x\sum_j u_j` 相对初值的漂移 :math:`H(t)-H(0)`。

    Neumann 与周期边界下 :math:`\tilde D` 的**行和为零**，所以
    :math:`\mathbf 1^{\mathsf T}\tilde D = 0`，于是不论 :math:`\theta` 取多少都有
    :math:`\mathbf 1^{\mathsf T}u^{n+1} = \mathbf 1^{\mathsf T}u^{n}`——
    离散总热量在**每一步**都精确守恒（只在舍入级漂移）。
    Dirichlet 边界下端点行是恒等行，行和不为零，热量按边界通量流失。
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    h = dx * np.sum(arr, axis=1)
    return h - h[0]


# --------------------------------------------------------------------------- #
# 一次完整的求解
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HeatResult:
    """一次时间推进的结果（**不含精确解**：精确解由调用方按问题自己取，避免"静默给零"）。"""

    x: np.ndarray
    u: np.ndarray
    t: float
    dt: float
    dx: float
    r: float
    n_steps: int
    scheme: str
    bc: str

    @property
    def nx(self) -> int:
        return int(self.u.size)

    @property
    def mass(self) -> float:
        r"""离散总热量 :math:`\Delta x\sum_j u_j`。"""
        return float(self.dx * np.sum(self.u))

    @property
    def umax(self) -> float:
        r"""解的 :math:`\infty` 范数（发散时用它做溢出判据）。"""
        return float(np.max(np.abs(self.u))) if self.u.size else 0.0


@dataclass(frozen=True)
class HeatTrajectory:
    """沿时间轴的快照与逐步诊断量（动画与守恒账本用）。"""

    x: np.ndarray
    t: np.ndarray
    U: np.ndarray
    mass: np.ndarray
    umax: np.ndarray
    dt: float
    dx: float
    r: float
    n_steps: int
    scheme: str
    bc: str


def _step_affine(
    nx: int,
    *,
    r: float,
    theta: float,
    dx: float,
    dt: float,
    bc: str,
    left: float,
    right: float,
    source: Any,
) -> tuple[np.ndarray, np.ndarray]:
    r"""把一步写成**仿射映射** :math:`u^{n+1} = M u^n + c`，返回 ``(M, c)``。

    为什么值得这么写：矩阵 :math:`A = I-\theta r\tilde D` 与右端算子
    :math:`B = I+(1-\theta)r\tilde D` 都不随时间变，于是 :math:`n` 次追赶法
    （每次 :math:`O(n)`，共 :math:`O(n^2)`）就能把一步算子显式化，
    之后每步只剩一次矩阵-向量乘。它在数学上与逐步求解完全等价，
    :func:`heat_solve` 与逐步调用的结果在舍入级一致（断言里对拍过）。
    """
    lo, dg, up, corner = theta_matrix(nx, r, theta, bc=bc)
    basis = np.eye(nx, dtype=float)
    dmat = np.column_stack(
        [laplacian_1d(basis[:, j], dx, bc=bc, left=0.0, right=0.0) for j in range(nx)]
    )
    bmat = (1.0 - theta) * r * dmat
    bmat[np.arange(nx), np.arange(nx)] += 1.0
    if bc == "dirichlet":
        # 端点行交给仿射项 c 携带（M 的端点行必须为 0，否则会把边界值再乘一遍）
        bmat[0, :] = 0.0
        bmat[-1, :] = 0.0
    c = np.zeros(nx, dtype=float)
    if source is not None:
        c = c + dt * np.broadcast_to(np.asarray(source, dtype=float), (nx,))
    if bc == "dirichlet":
        c[0] = float(left)
        c[-1] = float(right)
    elif bc == "neumann":
        c[0] = c[0] - 2.0 * theta * r * dx * float(left)
        c[-1] = c[-1] + 2.0 * theta * r * dx * float(right)
    mat = np.column_stack([_solve(lo, dg, up, corner, bmat[:, j]) for j in range(nx)])
    vec = _solve(lo, dg, up, corner, c)
    return mat, vec


def heat_solve(
    nx: int = 41,
    *,
    scheme: str = "cn",
    r: float = 0.4,
    t_end: float | None = None,
    n_steps: int | None = None,
    alpha: float = 1.0,
    bc: str = "dirichlet",
    initial: Any = "sine",
    k: int = 1,
    left: float = 0.0,
    right: float = 0.0,
    source: Any = None,
    theta: float | None = None,
    length: float = 1.0,
) -> HeatResult:
    r"""用 theta 族求解一维热传导，返回 :class:`HeatResult`。

    时间步由格点比反解：:math:`\Delta t = r\,\Delta x^2/\alpha`；
    步数缺省 :math:`\lceil t_{end}/\Delta t\rceil`（最后一拍可以略过 ``t_end``，实到时刻写在
    ``result.t`` 里——**绝不假装正好落在请求时刻**），也可以用 ``n_steps=`` 钉死。
    ``initial`` 可以是 :func:`initial_profile` 的类型名，也可以是**数组**（长度必须是 ``nx``）。
    """
    name_bc = _check_bc(bc, nx)
    th = scheme_theta(scheme, theta)
    grid = uniform_grid(nx, length=length, periodic=(name_bc == "periodic"))
    dx = grid.dx
    if alpha <= 0.0:
        raise ValueError(f"alpha 必须为正，得到 {alpha}")
    if r < 0.0:
        raise ValueError(f"r 不能为负，得到 {r}")
    dt = r * dx * dx / alpha
    if dt <= 0.0:
        raise ValueError(f"r={r} 让 dt 为零：无法推进（显式格式至少需要 r > 0）")
    if n_steps is None:
        if t_end is None:
            raise ValueError("t_end 与 n_steps 至少要给一个")
        n_steps = max(1, int(np.ceil(float(t_end) / dt)))
    n_steps = int(n_steps)
    if n_steps < 0:
        raise ValueError(f"n_steps 不能为负，得到 {n_steps}")
    if isinstance(initial, str):
        u0 = initial_profile(grid.x, initial, k)
    else:
        u0 = np.asarray(initial, dtype=float).ravel().copy()
        if u0.size != nx:
            raise ValueError(f"初值长度 {u0.size} 与 nx={nx} 不一致")
    src = None if source is None else np.broadcast_to(np.asarray(source, dtype=float), (nx,))
    mat, vec = _step_affine(
        nx, r=r, theta=th, dx=dx, dt=dt, bc=name_bc, left=left, right=right, source=src
    )
    u = np.array(u0, dtype=float)
    for _ in range(n_steps):
        u = mat @ u + vec
    return HeatResult(
        x=grid.x,
        u=u,
        t=n_steps * dt,
        dt=float(dt),
        dx=float(dx),
        r=float(r),
        n_steps=n_steps,
        scheme=str(scheme),
        bc=name_bc,
    )


def heat_trajectory(
    nx: int = 41,
    *,
    scheme: str = "cn",
    r: float = 0.4,
    n_steps: int = 200,
    alpha: float = 1.0,
    bc: str = "dirichlet",
    initial: Any = "sine",
    k: int = 1,
    left: float = 0.0,
    right: float = 0.0,
    source: Any = None,
    theta: float | None = None,
    length: float = 1.0,
    n_record: int = 41,
) -> HeatTrajectory:
    """推进 ``n_steps`` 步并记录 ``n_record`` 个等间隔快照 + 每一步的总热量/峰值。

    "逐步诊断量"是守恒类证据的来源：只有每一步都记账，才能区分
    "总量在舍入级漂移"与"总量单调流失"。
    """
    name_bc = _check_bc(bc, nx)
    th = scheme_theta(scheme, theta)
    grid = uniform_grid(nx, length=length, periodic=(name_bc == "periodic"))
    dx = grid.dx
    dt = r * dx * dx / alpha
    if dt <= 0.0:
        raise ValueError(f"r={r} 让 dt 为零：无法推进")
    n_steps = int(n_steps)
    if n_steps < 1:
        raise ValueError(f"n_steps 至少 1，得到 {n_steps}")
    n_record = max(2, int(n_record))
    if isinstance(initial, str):
        u = np.array(initial_profile(grid.x, initial, k), dtype=float)
    else:
        u = np.asarray(initial, dtype=float).ravel().copy()
        if u.size != nx:
            raise ValueError(f"初值长度 {u.size} 与 nx={nx} 不一致")
    src = None if source is None else np.broadcast_to(np.asarray(source, dtype=float), (nx,))
    mat, vec = _step_affine(
        nx, r=r, theta=th, dx=dx, dt=dt, bc=name_bc, left=left, right=right, source=src
    )
    idx = sorted({int(round(i * n_steps / (n_record - 1))) for i in range(n_record)})
    idx[0] = 0
    idx[-1] = n_steps
    t_out = np.empty(len(idx), dtype=float)
    u_out = np.empty((len(idx), nx), dtype=float)
    mass_out = np.empty(len(idx), dtype=float)
    umax_out = np.empty(len(idx), dtype=float)
    j = 0
    for step in range(n_steps + 1):
        if j < len(idx) and step == idx[j]:
            t_out[j] = step * dt
            u_out[j] = u
            mass_out[j] = dx * float(np.sum(u))
            umax_out[j] = float(np.max(np.abs(u)))
            j += 1
        if step < n_steps:
            u = mat @ u + vec
    return HeatTrajectory(
        x=grid.x,
        t=t_out,
        U=u_out,
        mass=mass_out,
        umax=umax_out,
        dt=float(dt),
        dx=float(dx),
        r=float(r),
        n_steps=n_steps,
        scheme=str(scheme),
        bc=name_bc,
    )


# --------------------------------------------------------------------------- #
# 稳定性：实测谱半径与解析阈值
# --------------------------------------------------------------------------- #
def _project(v: np.ndarray, bc: str) -> np.ndarray:
    r"""把迭代向量投影到**非平凡子空间**，这是测准"最坏模态衰减率"的前提。

    * ``neumann`` / ``periodic``：减去均值。常数模态是**右**特征向量、放大因子恒为 1
      （守恒），它会盖住其余所有模态：实测 ``nx=41, r=0.49`` 时不减均值得到
      ``1.000000000000``，减掉之后才是真正的谱半径 ``0.996978987018``。
      对阈值二分无伤大雅（越界时 :math:`|G|>1` 的模态仍会胜出），但"最坏模态衰减多快"
      这个量会被彻底抹平——BTCS 在 :math:`r=50` 的真实值是 0.646，不投影则是 1.0。
    * ``dirichlet``：**这一步实测是空操作**。恒等行确实给矩阵添了两个等于 1 的特征值，
      但它们对应的右特征向量在端点上有非零分量，而 :func:`ftcs_step` / :func:`theta_step`
      每步都把端点硬钉成 0（``left=right=0``），迭代走不到那些方向上去。
      实测 ``nx=41, r=0.49``：投影前后都是 ``0.996978987``（差 5e-11，是幂迭代的收敛残差）。
      保留置零只是为了让语义在一个入口写清楚，并防止将来有人把端点数据改成非零。
    """
    if bc == "dirichlet":
        out = v.copy()
        out[0] = 0.0
        out[-1] = 0.0
        return out
    return v - float(np.mean(v))


def explicit_growth(
    lam: Any,
    *,
    nx: int | None = None,
    bc: str = "dirichlet",
    n_steps: int | None = None,
    seed: int = 20261018,
    rtol: float = 1e-12,
    max_iter: int = 20000,
) -> float:
    r"""**实测**显式一步算子的谱半径：对固定种子的随机扰动做归一化幂迭代。

    ``lam`` 是标量（定常系数）或逐点数组（变系数）。四个设计决定都不能省：

    1. 初值必须是**随机扰动**——单模态初值永远不激发不稳定模态，
       "拿最低模态测稳定阈值"是本项目最常见的假阴性；
    2. 每步都要**归一化**并投影（:func:`_project`），否则衰减最快的模态会把范数压下去，
       或者常数模态（Neumann/周期）把结论钉死在"稳定"；
    3. 返回**最后一步**的放大比（幂迭代已收敛到主导特征值）；
    4. ``n_steps=None`` 时迭代到相邻比值相对变化 ≤ ``rtol`` 才停：
       阈值附近两个特征值只差 1e-7 量级，固定步数根本分辨不出来。
    """
    if nx is None:
        arr = np.asarray(lam, dtype=float)
        if arr.ndim == 0:
            raise ValueError("lam 是标量时必须给 nx")
        n = int(arr.size)
    else:
        n = int(nx)
    name = _check_bc(bc, n)
    lam_arr = np.broadcast_to(np.asarray(lam, dtype=float), (n,))
    gen = rng(seed)
    v = np.asarray(gen.standard_normal(n), dtype=float)
    v = _project(v, name)
    nv = float(np.sqrt(np.mean(v * v)))
    if nv == 0.0:  # pragma: no cover - 概率为零
        raise ValueError("初始扰动投影后为零")
    v = v / nv
    ratio = 1.0
    limit = int(n_steps) if n_steps is not None else int(max_iter)
    for it in range(limit):
        w = ftcs_step(v, lam_arr, bc=name)
        w = _project(w, name)
        nw = float(np.sqrt(np.mean(w * w)))
        if nw == 0.0:
            return 0.0
        converged = it > 8 and abs(nw - ratio) <= rtol * max(nw, 1e-300)
        ratio = nw
        v = w / nw
        if n_steps is None and converged:
            break
    return float(ratio)


def step_growth(
    nx: int,
    *,
    scheme: str = "ftcs",
    r: float = 0.4,
    theta: float | None = None,
    bc: str = "dirichlet",
    n_steps: int | None = None,
    seed: int = 20261018,
    rtol: float = 1e-12,
    max_iter: int = 20000,
) -> float:
    r"""**实测** theta 族一步算子在物理子空间里的谱半径（含义同 :func:`explicit_growth`）。

    显式格式直接走 :func:`explicit_growth`；隐式格式每一步真解一次三对角系统
    （``dx = dt = 1``，边界数据为零——它们只影响仿射项，不影响算子）。
    """
    name = _check_bc(bc, nx)
    th = scheme_theta(scheme, theta)
    if th == 0.0:
        return explicit_growth(r, nx=nx, bc=name, n_steps=n_steps, seed=seed, rtol=rtol, max_iter=max_iter)
    gen = rng(seed)
    v = _project(np.asarray(gen.standard_normal(nx), dtype=float), name)
    nv = float(np.sqrt(np.mean(v * v)))
    if nv == 0.0:  # pragma: no cover
        raise ValueError("初始扰动投影后为零")
    v = v / nv
    ratio = 1.0
    limit = int(n_steps) if n_steps is not None else int(max_iter)
    for it in range(limit):
        w = theta_step(v, r=r, theta=th, dx=1.0, dt=1.0, bc=name)
        w = _project(w, name)
        nw = float(np.sqrt(np.mean(w * w)))
        if nw == 0.0:
            return 0.0
        converged = it > 8 and abs(nw - ratio) <= rtol * max(nw, 1e-300)
        ratio = nw
        v = w / nw
        if n_steps is None and converged:
            break
    return float(ratio)


# --------------------------------------------------------------------------- #
# 二维：显式与 ADI（Peaceman--Rachford）
# --------------------------------------------------------------------------- #
def heat2d_grid(nx: int, ny: int | None = None, length: float = 1.0) -> tuple[np.ndarray, np.ndarray, float, float]:
    """二维均匀网格，返回 ``(x, y, dx, dy)``（两向都是 ``length/(n-1)``，Dirichlet）。"""
    if ny is None:
        ny = nx
    if nx < 3 or ny < 3:
        raise ValueError(f"nx、ny 至少 3，得到 {nx}/{ny}")
    x = np.linspace(0.0, length, nx)
    y = np.linspace(0.0, length, ny)
    return x, y, float(length / (nx - 1)), float(length / (ny - 1))


def heat2d_exact_sine(
    x: Any, y: Any, t: float, *, kx: int = 1, ky: int = 1, alpha: float = 1.0, length: float = 1.0
) -> np.ndarray:
    r"""二维正方形上零 Dirichlet 数据、:math:`u_0=\sin(k_x\pi x/L)\sin(k_y\pi y/L)` 的精确解。

    :math:`u = e^{-\alpha\pi^2(k_x^2+k_y^2)t/L^2}\sin(k_x\pi x/L)\sin(k_y\pi y/L)`，
    边界上显式置零（理由同 :func:`exact_dirichlet_sine`）。
    """
    xx = np.asarray(x, dtype=float)[:, None]
    yy = np.asarray(y, dtype=float)[None, :]
    decay = float(np.exp(-alpha * (np.pi / length) ** 2 * (kx * kx + ky * ky) * t))
    u = np.sin(kx * np.pi * xx / length) * np.sin(ky * np.pi * yy / length) * decay
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    return u


def _laplacian_2d_dirichlet(u: np.ndarray) -> np.ndarray:
    """二维五点 Laplacian（以 :math:`\\Delta x^2` 为单位，边界行/列取 0）。"""
    out = np.zeros_like(u)
    out[1:-1, 1:-1] = (
        u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]
    )
    return out


def heat2d_explicit_step(u: Any, r: float) -> np.ndarray:
    r"""二维显式一步：:math:`u \leftarrow u + r\,(\delta_x^2+\delta_y^2)u`，边界固定为 0。

    稳定条件是 :math:`r \le 1/4`（一维的一半）：两个方向各贡献一个 :math:`4r` 的 Nyquist 模态，
    最坏情形 :math:`|1-8r|\le1` 给出 :math:`r\le1/4`。把一维的 1/2 直接搬到二维，
    是"维度改阈值"最直接的例子。
    """
    arr = np.asarray(u, dtype=float)
    if r < 0.0:
        raise ValueError(f"r 不能为负，得到 {r}")
    out = arr.copy()
    out[1:-1, 1:-1] = arr[1:-1, 1:-1] + r * _laplacian_2d_dirichlet(arr)[1:-1, 1:-1]
    return out


def heat2d_adi_step(u: Any, r: float) -> np.ndarray:
    r"""Peaceman--Rachford ADI 一步（无条件稳定、二阶、每半步只解一维三对角）。

    .. math::
        (I - \tfrac{r}{2}\delta_x^2)u^{*} = (I + \tfrac{r}{2}\delta_y^2)u^n,\qquad
        (I - \tfrac{r}{2}\delta_y^2)u^{n+1} = (I + \tfrac{r}{2}\delta_x^2)u^{*}.

    每个方向的求解是**逐行/逐列**的追赶法（端点用恒等行钉住零边界）。
    它与二维 BTCS 的差别只在"每个方向只隐式一半"，代价从二维五对角降到一维三对角——
    这正是 ADI 存在的理由。
    """
    arr = np.asarray(u, dtype=float)
    if r < 0.0:
        raise ValueError(f"r 不能为负，得到 {r}")
    ny, nx = arr.shape
    half = 0.5 * r
    lo = np.full(nx, -half)
    dg = np.full(nx, 1.0 + 2.0 * half)
    up = np.full(nx, -half)
    lo[0] = 0.0
    up[0] = 0.0
    dg[0] = 1.0
    dg[-1] = 1.0
    lo[-1] = 0.0
    up[-1] = 0.0
    # 第一半步：x 方向隐式，y 方向显式
    rhs = arr.copy()
    rhs[1:-1, :] = rhs[1:-1, :] + half * (arr[2:, :] - 2.0 * arr[1:-1, :] + arr[:-2, :])
    # 恒等行会把右端的边界位置原样搬到解上，所以先把边界位置的右端清零，
    # 免得"边界值"从右端的浮点残渣里长出来。
    rhs[0, :] = 0.0
    rhs[-1, :] = 0.0
    rhs[:, 0] = 0.0
    rhs[:, -1] = 0.0
    star = np.empty_like(arr)
    for j in range(ny):
        star[j, :] = tridiagonal_apply(lo, dg, up, rhs[j, :])
    star[0, :] = 0.0
    star[-1, :] = 0.0
    star[:, 0] = 0.0
    star[:, -1] = 0.0
    # 第二半步：y 方向隐式，x 方向显式
    rhs2 = star.copy()
    rhs2[:, 1:-1] = rhs2[:, 1:-1] + half * (star[:, 2:] - 2.0 * star[:, 1:-1] + star[:, :-2])
    lo2 = np.full(ny, -half)
    dg2 = np.full(ny, 1.0 + 2.0 * half)
    up2 = np.full(ny, -half)
    lo2[0] = 0.0
    up2[0] = 0.0
    dg2[0] = 1.0
    dg2[-1] = 1.0
    lo2[-1] = 0.0
    up2[-1] = 0.0
    out = np.empty_like(arr)
    for i in range(nx):
        out[:, i] = tridiagonal_apply(lo2, dg2, up2, rhs2[:, i])
    out[0, :] = 0.0
    out[-1, :] = 0.0
    out[:, 0] = 0.0
    out[:, -1] = 0.0
    return out


def heat2d_growth(
    nx: int,
    *,
    r: float,
    ny: int | None = None,
    adi: bool = False,
    n_steps: int | None = None,
    seed: int = 20261018,
    rtol: float = 1e-12,
    max_iter: int = 4000,
) -> float:
    r"""**实测**二维一步算子的谱半径（幂迭代，边界与常数模态都投影掉）。

    ``adi=True`` 时测的是 ADI 一步算子（两个半步合起来），它在任意 :math:`r>0` 上都应满足
    :math:`\le 1`——"无条件稳定"这句话的实测版本。
    """
    if ny is None:
        ny = nx
    gen = rng(seed)
    v = np.asarray(gen.standard_normal((nx, ny)), dtype=float)
    v[0, :] = 0.0
    v[-1, :] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0
    nv = float(np.sqrt(np.mean(v * v)))
    if nv == 0.0:  # pragma: no cover
        raise ValueError("初始扰动为零")
    v = v / nv
    ratio = 1.0
    limit = int(n_steps) if n_steps is not None else int(max_iter)
    for it in range(limit):
        w = heat2d_adi_step(v, r) if adi else heat2d_explicit_step(v, r)
        w[0, :] = 0.0
        w[-1, :] = 0.0
        w[:, 0] = 0.0
        w[:, -1] = 0.0
        nw = float(np.sqrt(np.mean(w * w)))
        if nw == 0.0:
            return 0.0
        converged = it > 8 and abs(nw - ratio) <= rtol * max(nw, 1e-300)
        ratio = nw
        v = w / nw
        if n_steps is None and converged:
            break
    return float(ratio)

