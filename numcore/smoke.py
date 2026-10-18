"""0.0 章（管线自检）的数值内核。

这一章不教新东西，它的存在是为了**证明整条流水线是真的**：
自研实现 → 后端帧 → 前端 Scene → 三类断言 → 视觉回归 → 笔记。
所以这里放两个"最小但完整"的问题：阻尼振子（有闭式解）与一维热传导显式格式（有稳定阈值）。

两者都只用 numpy 的基础 ufunc（P3：不得使用任何现成求解器）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from numcore.utils import rng

__all__ = [
    "damped_oscillator",
    "oscillator_energy",
    "oscillator_residual",
    "ftcs_amplification",
    "ftcs_critical_r",
    "ftcs_critical_r_limit",
    "ftcs_growth",
    "ftcs_stable",
    "initial_profile",
    "heat_exact_sine",
    "heat_ftcs",
    "HeatRun",
]


# --------------------------------------------------------------------------- #
# 阻尼振子 x'' + 2ζω x' + ω² x = 0
# --------------------------------------------------------------------------- #
def damped_oscillator(
    t: Any,
    omega: float = 1.0,
    zeta: float = 0.05,
    x0: float = 1.0,
    v0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """阻尼振子的**精确解**，返回 ``(x(t), v(t))``。

    方程：``x'' + 2ζω x' + ω² x = 0``，初值 ``x(0)=x0``、``x'(0)=v0``。

    * ``ζ < 1``（欠阻尼）：``x = e^{−ζωt}[x0 cos(ω_d t) + ((v0 + ζω x0)/ω_d) sin(ω_d t)]``，``ω_d = ω√(1−ζ²)``；
    * ``ζ = 1``（临界）：``x = e^{−ωt}[x0 + (v0 + ω x0) t]``；
    * ``ζ > 1``（过阻尼）：两个实根 ``λ± = ω(−ζ ± √(ζ²−1))``，按初值定系数。

    三种分支都实现了——因为它们对应完全不同的定性行为，是 3.1/3.2 章的对照基准。
    """
    tt = np.atleast_1d(np.asarray(t, dtype=float))
    if omega <= 0:
        raise ValueError(f"omega 必须为正，得到 {omega}")
    if zeta < 0:
        raise ValueError(f"zeta 不能为负，得到 {zeta}")
    if abs(zeta - 1.0) < 1e-12:
        x = np.exp(-omega * tt) * (x0 + (v0 + omega * x0) * tt)
        v = np.exp(-omega * tt) * (
            (v0 + omega * x0) - omega * (x0 + (v0 + omega * x0) * tt)
        )
        return x, v
    if zeta < 1.0:
        wd = omega * float(np.sqrt(1.0 - zeta * zeta))
        a = x0
        b = (v0 + zeta * omega * x0) / wd
        env = np.exp(-zeta * omega * tt)
        x = env * (a * np.cos(wd * tt) + b * np.sin(wd * tt))
        v = env * (
            (-zeta * omega * a + wd * b) * np.cos(wd * tt)
            + (-zeta * omega * b - wd * a) * np.sin(wd * tt)
        )
        return x, v
    root = omega * float(np.sqrt(zeta * zeta - 1.0))
    lam_p = -zeta * omega + root
    lam_m = -zeta * omega - root
    c_p = (v0 - lam_m * x0) / (lam_p - lam_m)
    c_m = x0 - c_p
    x = c_p * np.exp(lam_p * tt) + c_m * np.exp(lam_m * tt)
    v = c_p * lam_p * np.exp(lam_p * tt) + c_m * lam_m * np.exp(lam_m * tt)
    return x, v


def oscillator_energy(x: Any, v: Any, omega: float = 1.0) -> np.ndarray:
    """无阻尼振子的比能量 ``E = ½(v² + ω²x²)``（守恒量，B 类断言的靶子）。"""
    return 0.5 * (np.asarray(v, dtype=float) ** 2 + (omega**2) * np.asarray(x, dtype=float) ** 2)


def oscillator_ode_residual(
    t: Any,
    omega: float = 1.0,
    zeta: float = 0.05,
    x0: float = 1.0,
    v0: float = 0.0,
    *,
    h: float = 1e-5,
) -> np.ndarray:
    """把解代回原方程的残差 ``|x'' + 2ζω x' + ω²x|``。

    ``x''`` 用**中心差分**从解析解本身求得（而不是用解析二阶导——那样残差会因为
    推导自洽而恒等于 0，什么都测不出来）。于是这个残差里同时含有
    "解析解是否真的满足方程" 与 "差分截断误差 ``O(h²·x'''')``" 两部分，
    量级约 ``h² ≈ 1e-10``：这正是 A 类断言该有的样子。
    """
    tt = np.atleast_1d(np.asarray(t, dtype=float))
    x_c, v_c = damped_oscillator(tt, omega, zeta, x0, v0)
    x_p, _ = damped_oscillator(tt + h, omega, zeta, x0, v0)
    x_m, _ = damped_oscillator(tt - h, omega, zeta, x0, v0)
    acc = (x_p - 2.0 * x_c + x_m) / (h * h)
    return np.abs(acc + 2.0 * zeta * omega * v_c + (omega**2) * x_c)


# --------------------------------------------------------------------------- #
# 一维热传导的显式格式（FTCS）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HeatRun:
    """一次显式热传导计算的全部产物。"""

    x: np.ndarray
    u: np.ndarray
    u_exact: np.ndarray
    error: np.ndarray
    dt: float
    n_steps: int
    r: float

    @property
    def max_error(self) -> float:
        return float(np.max(np.abs(self.error))) if self.error.size else 0.0

    @property
    def l2_error(self) -> float:
        return float(np.sqrt(np.mean(self.error**2))) if self.error.size else 0.0


def ftcs_amplification(r: float, mode: int, nx: int) -> float:
    """FTCS 对第 ``mode`` 个 Fourier 模的放大因子 ``g = 1 − 4r sin²(kπΔx/2)``。

    有效模态是 ``k = 1 … nx−2``（两端固定，内部有 ``nx−2`` 个未知量）。
    ``k = nx−1`` 给的是"每格反号"的幽灵模态，网格上无法表示，不属于可分辨模态。
    """
    if r < 0.0:
        raise ValueError(f"r 不能为负，得到 {r}")
    if not 1 <= mode <= nx - 1:
        raise ValueError(f"mode 必须在 1..{nx - 1} 之间，得到 {mode}")
    theta = mode * np.pi / (nx - 1)
    return float(1.0 - 4.0 * r * float(np.sin(theta / 2.0)) ** 2)


def ftcs_critical_r(nx: int) -> float:
    """**有限网格上**的 FTCS 稳定阈值 ``r* = 1/(2 sin²(θ_max/2))``，``θ_max = (nx−2)π/(nx−1)``。

    注意它严格大于 1/2：网格分辨不出"每格反号"的那个模态，所以存在一小段
    ``1/2 < r < r*(nx)`` 其实是稳定的。``r*(nx) → 1/2``（nx → ∞）。
    这正是 C 类断言必须**实测**而不是抄"r ≤ 1/2"的原因——两者只在连续极限下相等。
    """
    if nx < 4:
        raise ValueError(f"nx 至少 4，得到 {nx}")
    theta_max = (nx - 2) * np.pi / (nx - 1)
    return float(1.0 / (2.0 * float(np.sin(theta_max / 2.0)) ** 2))


def ftcs_critical_r_limit() -> float:
    """连续极限下的稳定阈值 ``r ≤ 1/2``（理论值，用于与实测值对照）。"""
    return 0.5


def ftcs_stable(r: float, nx: int, *, margin: float = 0.0) -> bool:
    """**解析判据**：全部可分辨模态（``k = 1 … nx−2``）的放大因子是否满足 ``|g| ≤ 1``。"""
    for mode in range(1, nx - 1):
        if abs(ftcs_amplification(r, mode, nx)) > 1.0 + margin:
            return False
    return True


def ftcs_growth(
    r: float,
    nx: int = 21,
    *,
    n_steps: int | None = None,
    seed: int = 20261018,
    rtol: float = 1e-12,
    max_iter: int = 20000,
) -> float:
    """**实测** FTCS 一步算子的谱半径 |λ|max：对固定种子的随机扰动做**归一化幂迭代**。

    四个必须讲清的设计决定（否则这个"测量"会给出假的稳定性结论）：

    1. **必须用随机扰动**，不能用 ``sin(πx)``：单模态初值永远不激发不稳定模态，
       "拿最低模态测稳定性"是数值实验里最常见的假阴性；
    2. **必须每步归一化**：不归一化时衰减最快的模态会把范数压下去，即使存在一个略大于 1
       的特征值，总范数照样变小（实测：r=0.5032 时 300 步总增长比只有 0.078），
       于是阈值附近完全测不出差别；
    3. **返回最后一步的放大比**：归一化幂迭代收敛到主导特征值，取最后一步即 |λ|max；
    4. **``n_steps=None`` 时迭代到收敛**（相邻两次比值相对变化 ≤ ``rtol``，上限 ``max_iter``）：
       阈值附近 |λ₁| 与 |λ₂| 只差 1e-7 量级，固定步数根本分辨不出来——二分法要准，
       就必须让每次测量自己收敛。
    """
    if nx < 4:
        raise ValueError(f"nx 至少 4，得到 {nx}")
    gen = rng(seed)
    v = np.asarray(gen.standard_normal(nx), dtype=float)
    v[0] = 0.0
    v[-1] = 0.0
    norm = float(np.sqrt(np.mean(v * v)))
    if norm == 0.0:  # pragma: no cover - 概率为零，兜底
        raise ValueError("初始扰动为零")
    v = v / norm
    ratio = 1.0
    it = 0
    limit = int(n_steps) if n_steps is not None else int(max_iter)
    while it < limit:
        w = v.copy()
        w[1:-1] = v[1:-1] + r * (v[2:] - 2.0 * v[1:-1] + v[:-2])
        w[0] = 0.0
        w[-1] = 0.0
        nw = float(np.sqrt(np.mean(w * w)))
        if nw == 0.0:  # pragma: no cover - 只有 r=0 才可能
            return 0.0
        converged = it > 8 and abs(nw - ratio) <= rtol * max(nw, 1e-300)
        ratio = nw
        v = w / nw
        it += 1
        if n_steps is None and converged:
            break
    return ratio


def heat_exact_sine(x: Any, t: float, alpha: float = 1.0, k: int = 1) -> np.ndarray:
    """``u_t = αu_xx``、``u(0)=u(1)=0``、``u(x,0)=sin(kπx)`` 的精确解。

    两端点**显式置零**：边界条件是 ``u(0,t)=u(1,t)=0``，这是精确成立的，
    而 ``sin(kπ)`` 在 IEEE 双精度下给 ``1.2e-16``（不是 0）。留着这点浮点残渣，
    误差场就会凭空多出 12 个数量级的"误差"（实测把对数误差轴从 ``1e-8…1e-3``
    拉到 ``1e-20…1``，中段结构全被压扁），而它跟数值格式毫无关系。
    前端的模型层发现并报告了这一点（`viz/chapters/0.0/heat_ftcs.model.ts` 的作者）。
    """
    arr = np.asarray(x, dtype=float)
    u = np.sin(k * np.pi * arr) * float(np.exp(-alpha * (k * np.pi) ** 2 * t))
    if arr.size >= 2:
        if abs(arr[0]) < 1e-12:
            u[0] = 0.0
        if abs(arr[-1] - 1.0) < 1e-12:
            u[-1] = 0.0
    return u


def initial_profile(x: Any, kind: str = "sine", k: int = 1) -> np.ndarray:
    """初值剖面：``sine``（单模态，有精确解）或 ``hat``（三角帽，激发全部模态）。"""
    arr = np.asarray(x, dtype=float)
    if kind == "sine":
        return np.sin(k * np.pi * arr)
    if kind == "hat":
        return 1.0 - np.abs(2.0 * arr - 1.0)
    raise ValueError(f"未知初值类型 {kind!r}（可用 sine / hat）")


def ftcs_step(u: Any, r: float) -> np.ndarray:
    """**一步** FTCS：``u_j ← u_j + r(u_{j+1} - 2u_j + u_{j-1})``，两端 Dirichlet 置零。

    单独暴露一步是给"逐点对拍"用的：TS 侧有同名的 ``ftcsStep``（``viz/shared/src/numeric/heat.ts``），
    G7 的双实现对拍要能一对一映射。没有它的时候，对拍只能拿
    ``t_end = n_steps·Δt`` 反算步数——那是在依赖 ``ceil`` 的巧合，不是契约。
    """
    arr = np.asarray(u, dtype=float)
    out = arr.copy()
    if arr.size >= 3:
        interior = slice(1, -1)
        out[interior] = arr[interior] + r * (arr[2:] - 2.0 * arr[interior] + arr[:-2])
    if arr.size >= 2:
        out[0] = 0.0
        out[-1] = 0.0
    return out


def ftcs_run(u0: Any, r: float, n_steps: int) -> np.ndarray:
    """从 ``u0`` 走 ``n_steps`` 步 FTCS（每步都重新施加边界条件）。"""
    if n_steps < 0:
        raise ValueError(f"n_steps 不能为负，得到 {n_steps}")
    u = np.asarray(u0, dtype=float).copy()
    for _ in range(int(n_steps)):
        u = ftcs_step(u, r)
    return u


def heat_ftcs(
    nx: int = 41,
    alpha: float = 1.0,
    t_end: float = 0.05,
    r: float = 0.4,
    k: int = 1,
    *,
    initial: str = "sine",
    n_steps: int | None = None,
) -> HeatRun:
    """显式 FTCS 解一维热传导（自研，仅用切片与 ufunc）。

    步长由 ``r = αΔt/Δx²`` 反解：``Δt = r Δx² / α``；
    步数缺省 ``ceil(t_end / Δt)``（最后一拍可能略过 ``t_end``），
    也可以用 ``n_steps=`` 直接钉死——对拍/复现都靠它，免得依赖 ``ceil`` 的巧合。
    """
    if nx < 5:
        raise ValueError(f"nx 至少 5，得到 {nx}")
    if alpha <= 0:
        raise ValueError(f"alpha 必须为正，得到 {alpha}")
    if r <= 0:
        raise ValueError(f"r 必须为正，得到 {r}")
    dx = 1.0 / (nx - 1)
    dt = r * dx * dx / alpha
    if n_steps is None:
        n_steps = max(1, int(np.ceil(t_end / dt)))
    elif n_steps < 0:
        raise ValueError(f"n_steps 不能为负，得到 {n_steps}")
    x = np.linspace(0.0, 1.0, nx)
    u = ftcs_run(initial_profile(x, initial, k), r, n_steps)
    t_final = n_steps * dt
    exact = heat_exact_sine(x, t_final, alpha=alpha, k=k)
    return HeatRun(
        x=x,
        u=u.copy(),
        u_exact=exact,
        error=u - exact,
        dt=dt,
        n_steps=n_steps,
        r=r,
    )


def heat_ftcs_history(
    nx: int = 41,
    alpha: float = 1.0,
    t_end: float = 0.05,
    r: float = 0.4,
    k: int = 1,
    *,
    frames: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 ``frames`` 个等间隔时刻返回 ``(x, U, U_exact)``，``U`` 形状 ``(frames, nx)``。

    供时间轴（``mode="scrub"``）使用：帧数固定，便于前端与视觉回归对齐。
    """
    if frames < 2:
        raise ValueError(f"frames 至少 2，得到 {frames}")
    dx = 1.0 / (nx - 1)
    r_used = min(r, 0.49)  # 越界会让格式发散，显示层应当显式看到"越界"而不是 NaN
    dt = r_used * dx * dx / alpha
    total = max(1, int(np.ceil(t_end / dt)))
    per_frame = max(1, total // (frames - 1))
    x = np.linspace(0.0, 1.0, nx)
    u = np.sin(k * np.pi * x)
    interior = slice(1, -1)
    U = np.empty((frames, nx), dtype=float)
    E = np.empty((frames, nx), dtype=float)
    U[0] = u
    E[0] = u - heat_exact_sine(x, 0.0, alpha=alpha, k=k)
    t = 0.0
    for f in range(1, frames):
        for _ in range(per_frame):
            u[interior] = u[interior] + r_used * (u[2:] - 2.0 * u[interior] + u[:-2])
            u[0] = 0.0
            u[-1] = 0.0
            t += dt
        U[f] = u
        E[f] = u - heat_exact_sine(x, t, alpha=alpha, k=k)
    return x, U, E


def heat_run_meta(run: HeatRun) -> dict[str, Any]:
    """把一次计算的关键标量打包成 ``meta``（只允许标量，见 PROTOCOLS §1.1）。"""
    return {
        "dt": run.dt,
        "n_steps": run.n_steps,
        "r": run.r,
        "max_error": run.max_error,
        "l2_error": run.l2_error,
    }
