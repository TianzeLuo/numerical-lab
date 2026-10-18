r"""``numcore.inverse.sensitivity`` —— 灵敏度分析的**算例层**（5.3 章）。

:mod:`numcore.inverse.adjoint` 只认"一个右端、一个解析 Jacobian、一个终端目标"，
本模块把三组算例与四个"要量出来的结论"接上去：

============  ==========================================================================
算例           用途
============  ==========================================================================
``duffing``    非线性受迫振子（刚度 / 阻尼 / 三次项）：梯度场、V 形曲线、正反轨迹叠加。
                非线性是**故意的**——线性问题上连续伴随与离散伴随会恰好重合，
                那样"两者差 :math:`O(h^p)`"这条结论就永远测不出来。
``linear``     线性时不变 + 参数只进**常数**源项：这一组专门用来展示"两者恰好重合"
                的那条边界（对照，不是陪衬）。
``heat``       4.1 章热方程 :math:`\theta` 格式的 PDE 约束：影响场、存储/重算账。
============  ==========================================================================

四个结论各自的可测形式：

1. **灵敏度方向**（:func:`sensitivity_directions`）：归一化梯度方向与中心差分方向的夹角，
   以及各分量的"弹性" :math:`(\partial J/\partial p_j)(p_j/J)`——后者才是"哪个参数更值钱"
   的尺度无关说法。
2. **梯度场**（:func:`gradient_field`）：参数平面上逐点的目标函数等高线 + 伴随梯度方向 +
   与有限差分方向的相对误差。误差必须**逐点** < 1e-6，只报最大值不报分布等于没报。
3. **代价与参数个数无关**（:func:`cost_scaling_study`）：:math:`k=2/8/32` 三点上
   伴随的求解次数恒定、有限差分线性增长；同时把检查点版的存储/重算账摆出来。
4. **影响场的宽度律**（:func:`influence_width_law`）：反向伴随场的二阶中心矩随
   :math:`\tau` 增长，对照 :math:`\sigma_0^2+2\alpha\tau`（连续律）与半离散精确解
   （无空间截断误差的靶子）；**质心守恒**作为负面对照一起报。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from numcore.inverse.adjoint import (
    AdjointResult,
    CostRow,
    HeatAdjointProblem,
    OdeModel,
    WorkCount,
    continuous_adjoint,
    cost_comparison,
    dense_sine_coefficients,
    discrete_adjoint,
    heat_discrete_adjoint,
    influence_moments,
    relative_error,
    rk_cost,
    rk_forward,
    semidiscrete_adjoint_field,
    finite_difference_gradient,
)
from numcore.utils import rng

__all__ = [
    "scalar_decay_model",
    "linear_forcing_model",
    "duffing_model",
    "duffing_slice_model",
    "SensitivityDirections",
    "sensitivity_directions",
    "GradientField",
    "gradient_field",
    "cost_scaling_study",
    "InfluenceStudy",
    "influence_width_law",
    "linear_slope",
    "heat_adjoint_dt_errors",
    "heat_boundary_layer_ratio",
    "heat_gradient_summary",
    "adjoint_growth_law",
    "trajectory_norms",
    "adjoint_and_continuous",
    "objective_only",
]


# --------------------------------------------------------------------------- #
# 1. 算例
# --------------------------------------------------------------------------- #
def scalar_decay_model() -> OdeModel:
    r"""一维衰减 :math:`y'=-py,\ y(0)=1,\ J=y(T)`：唯一一个**两个梯度都有闭式**的算例。

    连续梯度 :math:`dJ/dp=-Te^{-pT}`；离散梯度（RK4 的稳定多项式
    :math:`R`，:math:`J_h=R(-hp)^N`）是 :math:`dJ_h/dp=-T R(-hp)^{N-1}R'(-hp)`。
    前者不依赖 :math:`h`，后者的差是 :math:`O(h^4)`——"离散伴随求的是离散问题的梯度"
    这句话在这里可以被精确称重。
    """
    def rhs(t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:  # noqa: ARG001
        return np.array([-float(p[0]) * float(y[0])])

    def jac_y(t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:  # noqa: ARG001
        return np.array([[-float(p[0])]])

    def jac_p(t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:  # noqa: ARG001
        return np.array([[-float(y[0])]])

    def y0(p: np.ndarray) -> np.ndarray:  # noqa: ARG001
        return np.array([1.0])

    def cost(y: np.ndarray) -> float:
        return float(y[0])

    def cost_grad(y: np.ndarray) -> np.ndarray:
        return np.array([1.0])

    def dy0_dp(p: np.ndarray) -> np.ndarray:  # noqa: ARG001
        return np.zeros((1, 1), dtype=float)

    return OdeModel(
        name="scalar_decay",
        dim=1,
        n_params=1,
        rhs=rhs,
        jac_y=jac_y,
        jac_p=jac_p,
        y0=y0,
        cost=cost,
        cost_grad=cost_grad,
        dy0_dp=dy0_dp,
        note="y'=-p y；连续梯度与离散梯度都有闭式",
    )


def linear_forcing_model(n_params: int, *, dim: int = 4, seed: int = 5321) -> OdeModel:
    r"""线性时不变算例：:math:`y'=Ay+Bp,\ y(0)=0,\ J=\tfrac12\|y(T)\|^2`。

    :math:`A` 对角稳定、:math:`B` 由确定性种子生成（P4）。参数**只进常数源项**，
    于是 :math:`\partial f/\partial p=B` 与状态无关——这正是连续伴随与离散伴随
    **恰好重合**的那条边界（两个递推逐位相同），也因此它适合用来做
    "代价与 :math:`k` 无关"的计数实验：梯度对每个分量都非零，有限差分没有退化。
    """
    gen = rng(int(seed))
    A = np.diag(-(0.6 + 0.8 * np.asarray(gen.random(int(dim)), dtype=float)))
    B = np.asarray(gen.standard_normal((int(dim), int(n_params))), dtype=float) / math.sqrt(float(dim))

    def rhs(t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:  # noqa: ARG001
        return A @ np.asarray(y, dtype=float) + B @ np.asarray(p, dtype=float)

    def jac_y(t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:  # noqa: ARG001
        return A

    def jac_p(t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:  # noqa: ARG001
        return B

    def y0(p: np.ndarray) -> np.ndarray:  # noqa: ARG001
        return np.zeros(int(dim), dtype=float)

    def cost(y: np.ndarray) -> float:
        yv = np.asarray(y, dtype=float)
        return 0.5 * float(np.dot(yv, yv))

    def cost_grad(y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float)

    return OdeModel(
        name=f"linear_forcing_k{n_params}",
        dim=int(dim),
        n_params=int(n_params),
        rhs=rhs,
        jac_y=jac_y,
        jac_p=jac_p,
        y0=y0,
        cost=cost,
        cost_grad=cost_grad,
        note="线性、常 Jacobian、参数只进常数源项",
    )


def _duffing_rhs(t: float, y: np.ndarray, p: np.ndarray, forcing: float, omega: float, cubic_fixed: float | None) -> np.ndarray:
    y1 = float(y[0])
    y2 = float(y[1])
    p3 = float(p[2]) if cubic_fixed is None else float(cubic_fixed)
    drive = float(forcing) * math.cos(float(omega) * float(t))
    return np.array([y2, -float(p[0]) * y1 - p3 * y1 * y1 * y1 - float(p[1]) * y2 + drive])


def duffing_model(*, forcing: float = 0.1, omega: float = 1.0) -> OdeModel:
    r"""受迫 Duffing 振子（3 参数）::

        y1' = y2
        y2' = -p1 y1 - p3 y1^3 - p2 y2 + F cos(omega t)
        J   = (y1(T)^2 + y2(T)^2) / 2

    :math:`p_1` 是线性刚度（小振幅下的共振脊在 :math:`p_1=\omega^2`）、
    :math:`p_2` 是阻尼、:math:`p_3` 是三次项。非线性体现在
    :math:`\partial f/\partial y` **沿轨迹变化**——这正是连续伴随与离散伴随分开的地方。
    """
    def rhs(t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return _duffing_rhs(t, y, p, forcing, omega, None)

    def jac_y(t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:  # noqa: ARG001
        y1 = float(y[0])
        return np.array([[0.0, 1.0], [-float(p[0]) - 3.0 * float(p[2]) * y1 * y1, -float(p[1])]])

    def jac_p(t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:  # noqa: ARG001
        y1 = float(y[0])
        y2 = float(y[1])
        return np.array([[0.0, 0.0, 0.0], [-y1, -y2, -y1 * y1 * y1]])

    def y0(p: np.ndarray) -> np.ndarray:  # noqa: ARG001
        return np.array([1.0, 0.0])

    def cost(y: np.ndarray) -> float:
        return 0.5 * float(y[0] * y[0] + y[1] * y[1])

    def cost_grad(y: np.ndarray) -> np.ndarray:
        return np.array([float(y[0]), float(y[1])])

    def dy0_dp(p: np.ndarray) -> np.ndarray:  # noqa: ARG001
        return np.zeros((2, 3), dtype=float)

    return OdeModel(
        name="duffing3",
        dim=2,
        n_params=3,
        rhs=rhs,
        jac_y=jac_y,
        jac_p=jac_p,
        y0=y0,
        cost=cost,
        cost_grad=cost_grad,
        dy0_dp=dy0_dp,
        note=f"受迫 Duffing（F={forcing}, omega={omega}）",
    )


def duffing_slice_model(cubic: float, *, forcing: float = 0.1, omega: float = 1.0) -> OdeModel:
    r"""两参数切片（:math:`p_3` 固定）：参数平面上的梯度场与轨迹播放用这一组。

    参数是 :math:`(p_1,p_2)` = （线性刚度，阻尼），三次项由 ``cubic`` 烘进去。
    切片与三参数模型共用同一个右端，所以两者的梯度可以互相印证。
    """
    def rhs(t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return _duffing_rhs(t, y, p, forcing, omega, float(cubic))

    def jac_y(t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:  # noqa: ARG001
        y1 = float(y[0])
        return np.array([[0.0, 1.0], [-float(p[0]) - 3.0 * float(cubic) * y1 * y1, -float(p[1])]])

    def jac_p(t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:  # noqa: ARG001
        return np.array([[0.0, 0.0], [-float(y[0]), -float(y[1])]])

    def y0(p: np.ndarray) -> np.ndarray:  # noqa: ARG001
        return np.array([1.0, 0.0])

    def cost(y: np.ndarray) -> float:
        return 0.5 * float(y[0] * y[0] + y[1] * y[1])

    def cost_grad(y: np.ndarray) -> np.ndarray:
        return np.array([float(y[0]), float(y[1])])

    def dy0_dp(p: np.ndarray) -> np.ndarray:  # noqa: ARG001
        return np.zeros((2, 2), dtype=float)

    return OdeModel(
        name=f"duffing_slice_c{cubic:g}",
        dim=2,
        n_params=2,
        rhs=rhs,
        jac_y=jac_y,
        jac_p=jac_p,
        y0=y0,
        cost=cost,
        cost_grad=cost_grad,
        dy0_dp=dy0_dp,
        note=f"两参数切片（cubic={cubic}）",
    )


# --------------------------------------------------------------------------- #
# 2. 灵敏度方向
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SensitivityDirections:
    r"""一次灵敏度分析的产物：伴随梯度、有限差分梯度、方向夹角、弹性、代价。"""

    p: np.ndarray
    grad_adjoint: np.ndarray
    grad_fd: np.ndarray
    rel_err: float
    direction: np.ndarray
    direction_fd: np.ndarray
    angle_deg: float
    elasticity: np.ndarray
    work: WorkCount
    n_fd_solves: int

    def dominant(self) -> int:
        r"""弹性绝对值最大的参数下标（"最值钱"的那个参数）。"""
        e = np.abs(np.asarray(self.elasticity, dtype=float))
        return int(np.argmax(e)) if e.size else -1


def sensitivity_directions(
    model: OdeModel,
    p: Any,
    *,
    T: float,
    n_steps: int,
    method: str = "rk4",
    eps: float = 1e-6,
) -> SensitivityDirections:
    r"""归一化灵敏度方向 + 弹性 + 与中心差分的对拍。

    方向是**单位向量**（"往哪儿调"），弹性 :math:`(\partial J/\partial p_j)(p_j/J)`
    才是"调多少"（尺度无关）。只报梯度向量的分量会在参数量纲不同时骗人：
    :math:`p_1\sim1` 与 :math:`p_2\sim10^{-3}` 上同样大小的梯度分量含义完全不同。
    """
    pv = np.asarray(p, dtype=float).reshape(-1)
    trace = rk_forward(model, pv, T=T, n_steps=n_steps, method=method, store="full")
    adj = discrete_adjoint(model, pv, trace)
    g_fd, fdw = finite_difference_gradient(model, pv, T=T, n_steps=n_steps, method=method, eps=eps, scheme="central")
    na = _unit(adj.grad)
    nf = _unit(g_fd)
    cosang = float(np.clip(float(np.dot(na, nf)), -1.0, 1.0)) if na.size else float("nan")
    j = float(trace.cost)
    with np.errstate(divide="ignore", invalid="ignore"):
        elas = np.where(np.abs(j) > 0.0, adj.grad * pv / j, np.nan)
    work = WorkCount(
        n_steps=int(n_steps),
        n_rhs=int(trace.work.n_rhs),
        n_jac_y=int(adj.work.n_jac_y),
        n_jac_p=int(adj.work.n_jac_p),
        n_forward_solves=1,
        n_floats_stored=int(trace.work.n_floats_stored),
    )
    return SensitivityDirections(
        p=pv,
        grad_adjoint=adj.grad,
        grad_fd=g_fd,
        rel_err=relative_error(adj.grad, g_fd),
        direction=na,
        direction_fd=nf,
        angle_deg=float(math.degrees(math.acos(cosang))) if math.isfinite(cosang) else float("nan"),
        elasticity=elas,
        work=work,
        n_fd_solves=int(fdw.n_forward_solves),
    )


def _unit(v: Any) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.sqrt(float(np.dot(a, a))))
    return a / n if n > 0.0 else np.zeros_like(a)


# --------------------------------------------------------------------------- #
# 3. 参数平面上的梯度场
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GradientField:
    r"""参数平面上逐点的目标函数、伴随方向、有限差分方向与逐点相对误差。"""

    p1: np.ndarray
    p2: np.ndarray
    objective: np.ndarray
    arrow_p1: np.ndarray
    arrow_p2: np.ndarray
    dir1: np.ndarray
    dir2: np.ndarray
    fd_dir1: np.ndarray
    fd_dir2: np.ndarray
    log10_rel_err: np.ndarray
    grad_norm: np.ndarray
    rel_err: np.ndarray
    work: WorkCount
    n_grid_solves: int
    n_arrow_solves: int

    def max_rel_err(self) -> float:
        return float(np.max(self.rel_err)) if self.rel_err.size else float("nan")

    def min_rel_err(self) -> float:
        return float(np.min(self.rel_err)) if self.rel_err.size else float("nan")

    def median_rel_err(self) -> float:
        return float(np.median(self.rel_err)) if self.rel_err.size else float("nan")


def gradient_field(
    model: OdeModel,
    *,
    p1_range: tuple[float, float],
    p2_range: tuple[float, float],
    n1: int = 13,
    n2: int = 11,
    arrow_n1: int = 4,
    arrow_n2: int = 4,
    T: float,
    n_steps: int,
    method: str = "rk4",
    eps: float = 1e-6,
) -> GradientField:
    r"""在 :math:`(p_1,p_2)` 网格上画目标函数，并在一个子网格上放伴随梯度箭头。

    两件事分开做是**刻意的**：等高线要的是密网格（便宜，只要目标函数值），
    箭头要的是伴随 + 有限差分对拍（贵，只在少数点上做）。把两者绑在一起，
    要么网格稀到看不出共振脊、要么一帧算不完。
    """
    a1 = np.linspace(float(p1_range[0]), float(p1_range[1]), int(n1))
    a2 = np.linspace(float(p2_range[0]), float(p2_range[1]), int(n2))
    Z = np.empty((int(n2), int(n1)), dtype=float)
    n_grid = 0
    n_rhs = 0
    for j, p2 in enumerate(a2):
        for i, p1 in enumerate(a1):
            tr = rk_forward(model, np.array([p1, p2]), T=T, n_steps=n_steps, method=method, store="nodes")
            Z[j, i] = tr.cost
            n_grid += 1
            n_rhs += int(tr.work.n_rhs)
    q1 = np.linspace(float(p1_range[0]), float(p1_range[1]), int(arrow_n1))
    q2 = np.linspace(float(p2_range[0]), float(p2_range[1]), int(arrow_n2))
    m = int(arrow_n1) * int(arrow_n2)
    ap1 = np.empty(m, dtype=float)
    ap2 = np.empty(m, dtype=float)
    d1 = np.empty(m, dtype=float)
    d2 = np.empty(m, dtype=float)
    f1 = np.empty(m, dtype=float)
    f2 = np.empty(m, dtype=float)
    gnorm = np.empty(m, dtype=float)
    rel = np.empty(m, dtype=float)
    n_arrow = 0
    for j, p2 in enumerate(q2):
        for i, p1 in enumerate(q1):
            idx = j * int(arrow_n1) + i
            pv = np.array([p1, p2])
            tr = rk_forward(model, pv, T=T, n_steps=n_steps, method=method, store="full")
            adj = discrete_adjoint(model, pv, tr)
            g_fd, _ = finite_difference_gradient(model, pv, T=T, n_steps=n_steps, method=method, eps=eps, scheme="central")
            u = _unit(adj.grad)
            uf = _unit(g_fd)
            ap1[idx] = p1
            ap2[idx] = p2
            d1[idx] = u[0]
            d2[idx] = u[1]
            f1[idx] = uf[0]
            f2[idx] = uf[1]
            gnorm[idx] = float(np.sqrt(float(np.dot(adj.grad, adj.grad))))
            rel[idx] = relative_error(adj.grad, g_fd)
            n_arrow += 1
            n_rhs += int(tr.work.n_rhs + adj.work.n_jac_y + adj.work.n_jac_p)
    log10err = np.log10(np.maximum(rel, np.finfo(float).tiny))
    work = WorkCount(
        n_steps=int(n_steps),
        n_rhs=int(n_rhs),
        n_forward_solves=int(n_grid + n_arrow),
        n_floats_stored=int(n_arrow * (int(n_steps) + 1) * model.dim),
    )
    return GradientField(
        p1=a1,
        p2=a2,
        objective=Z,
        arrow_p1=ap1,
        arrow_p2=ap2,
        dir1=d1,
        dir2=d2,
        fd_dir1=f1,
        fd_dir2=f2,
        log10_rel_err=log10err,
        grad_norm=gnorm,
        rel_err=rel,
        work=work,
        n_grid_solves=int(n_grid),
        n_arrow_solves=int(n_arrow),
    )


# --------------------------------------------------------------------------- #
# 4. 代价：与参数个数无关？
# --------------------------------------------------------------------------- #
def cost_scaling_study(
    param_counts: Sequence[int] = (2, 8, 32),
    *,
    dim: int = 4,
    T: float = 4.0,
    n_steps: int = 96,
    method: str = "rk4",
    block: int | None = None,
) -> list[CostRow]:
    r"""在 :math:`k=2/8/32` 三个点上测三笔账（伴随 / 检查点伴随 / 中心差分）。

    参数取值由种子固定（P4）。这里的算例是线性的——它保证每个分量的梯度都非零，
    有限差分不会因为"某一列恰好是 0"而少算一次求解，计数才是干净的。
    """
    cases: list[tuple[OdeModel, np.ndarray]] = []
    for k in param_counts:
        kk = int(k)
        gen = rng(9700 + kk)
        p = np.asarray(gen.uniform(-0.4, 0.4, kk), dtype=float)
        cases.append((linear_forcing_model(kk, dim=int(dim)), p))
    return cost_comparison(cases, T=T, n_steps=n_steps, method=method, block=block)


# --------------------------------------------------------------------------- #
# 5. 影响场：宽度律与质心守恒
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InfluenceStudy:
    r"""反向伴随影响场的一整套可测量：宽度律、质心守恒、与两个靶子的偏差。"""

    tau: np.ndarray
    width_measured: np.ndarray
    width_continuous: np.ndarray
    center_err: np.ndarray
    mu: np.ndarray
    grad: np.ndarray
    grad_fd: np.ndarray
    grad_rel_err: float
    width_ratio_at_max_tau: float
    center_drift_max: float
    semidiscrete_rel_err_max: float
    semidiscrete_window: tuple[float, float]
    boundary_contaminated: np.ndarray
    n_boundary_rows: int
    dx: float
    dt: float
    r: float
    alpha: float
    sigma0_sq: float
    var_slope_measured: float
    var_slope_theory: float
    width_ratio_clean_max: float

    def clean_mask(self) -> np.ndarray:
        """未被 Dirichlet 壁污染的时间行（这些行上的宽度律才可读）。"""
        return ~np.asarray(self.boundary_contaminated, dtype=bool)

    def var_slope_ratio(self) -> float:
        """实测宽度平方增长率与 :math:`2\\alpha` 的比值。"""
        return float(self.var_slope_measured / self.var_slope_theory) if self.var_slope_theory else float("nan")


def linear_slope(x: Any, y: Any, mask: Any = None) -> float | None:
    r"""一元最小二乘斜率（点数 < 2 或自变量无散布时返回 ``None``）。"""
    xv = np.asarray(x, dtype=float).reshape(-1)
    yv = np.asarray(y, dtype=float).reshape(-1)
    if mask is not None:
        m = np.asarray(mask).astype(bool).reshape(-1)
        xv = xv[m]
        yv = yv[m]
    ok = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[ok]
    yv = yv[ok]
    if xv.size < 2:
        return None
    xc = xv - float(np.mean(xv))
    den = float(np.dot(xc, xc))
    if den <= 0.0:
        return None
    yc = yv - float(np.mean(yv))
    return float(np.dot(xc, yc) / den)


def influence_width_law(
    problem: HeatAdjointProblem,
    amps: Any,
    *,
    n_sample: int = 61,
    fd_eps: float = 1e-6,
    boundary_tol: float = 1e-6,
) -> InfluenceStudy:
    r"""影响场的宽度律、质心守恒与靶子对照。

    三件必须分开报的事：

    * :math:`\sigma^2(\tau)=\sigma_0^2+2\alpha\tau` 是**连续**律；离散特征值
      :math:`\lambda_k< (k\pi/L)^2` 会让实测宽度**略快**于它（:math:`O(\Delta x^2)`）。
    * **半离散精确解**（用离散特征值 :math:`\lambda_k` 的正弦模叠加）是没有空间截断误差的靶子，
      实测与它的偏差才是干净的时间离散误差。
    * **质心守恒**：纯扩散不搬动影响场的中心（实测漂移应当只在舍入级）。
      "影响场随时间反向扩散"说的是**宽度**在长，不是峰值在跑——把这两件事混为一谈
      是本章最容易写出的假结论。
    """
    ops_mu, grad = heat_discrete_adjoint(problem, amps)
    x = problem.x()
    N = int(problem.n_steps)
    dx = float(problem.length) / float(int(problem.nx) - 1)
    dt = float(problem.horizon) / float(N)
    w = problem.observation()
    _, _, var0 = influence_moments(x, w)
    ks, coeffs = dense_sine_coefficients(x, w, float(problem.length))

    idx = np.unique(np.linspace(0, N, int(n_sample)).astype(int))
    tau = np.array([float(problem.horizon) - float(i) * dt for i in idx], dtype=float)
    width = np.empty(idx.size, dtype=float)
    center_err = np.empty(idx.size, dtype=float)
    semierr = np.empty(idx.size, dtype=float)
    contaminated = np.zeros(idx.size, dtype=bool)
    xc = float(np.sum(w * x) / np.sum(w))
    ref_all = semidiscrete_adjoint_field(
        x, coeffs, tau, nx=int(problem.nx), alpha=float(problem.alpha), length=float(problem.length)
    )
    for row, i in enumerate(idx):
        _, mean, var = influence_moments(x, ops_mu[i])
        width[row] = math.sqrt(max(var, 0.0))
        center_err[row] = mean - xc
        # 末端两个节点是 Dirichlet 钉住的行：伴随在那里取不到"精确 Dirichlet"的意义
        # （见 heat_adjoint_dt_errors 的说明），场对照只在内部节点上做。
        ref = ref_all[row][1:-1]
        num = float(np.sqrt(np.sum((ops_mu[i][1:-1] - ref) ** 2) * dx))
        den = float(np.sqrt(np.sum(ref * ref) * dx))
        semierr[row] = num / den if den > 0.0 else float("inf")
        peak = float(np.max(np.abs(ops_mu[i])))
        contaminated[row] = bool(peak > 0.0 and max(abs(ops_mu[i][0]), abs(ops_mu[i][-1])) / peak > float(boundary_tol))
    good = ~contaminated
    semi_max = float(np.max(semierr[good])) if bool(np.any(good)) else float("inf")
    win = (float(np.min(tau[good])), float(np.max(tau[good]))) if bool(np.any(good)) else (float("nan"), float("nan"))
    width_cont = np.sqrt(var0 + 2.0 * float(problem.alpha) * tau)
    g_fd, fdw = _heat_fd_gradient(problem, amps, eps=fd_eps)
    clean = ~contaminated
    var_meas = width * width
    slope_meas = linear_slope(tau, var_meas, clean)
    ratio_clean = float(np.max(np.abs(width[clean] / width_cont[clean] - 1.0))) if bool(np.any(clean)) else float("nan")
    return InfluenceStudy(
        tau=tau,
        width_measured=width,
        width_continuous=width_cont,
        center_err=center_err,
        mu=ops_mu[idx],
        grad=grad,
        grad_fd=g_fd,
        grad_rel_err=relative_error(grad, g_fd),
        width_ratio_at_max_tau=float(width[0] / width_cont[0]) if width_cont[0] > 0.0 else float("nan"),
        center_drift_max=float(np.max(np.abs(center_err))),
        semidiscrete_rel_err_max=semi_max,
        semidiscrete_window=win,
        boundary_contaminated=contaminated,
        n_boundary_rows=int(np.count_nonzero(contaminated)),
        dx=dx,
        dt=dt,
        r=float(problem.alpha) * dt / (dx * dx),
        alpha=float(problem.alpha),
        sigma0_sq=float(var0),
        var_slope_measured=float("nan") if slope_meas is None else float(slope_meas),
        var_slope_theory=2.0 * float(problem.alpha),
        width_ratio_clean_max=ratio_clean,
    )


def heat_adjoint_dt_errors(
    *,
    nx: int = 41,
    length: float = 1.0,
    alpha: float = 0.05,
    horizon: float = 0.2,
    n_steps_list: Sequence[int] = (50, 100, 200, 400, 800),
    centers: tuple[float, ...] = (0.35, 0.65),
    widths: tuple[float, ...] = (0.08, 0.08),
    obs_center: float = 0.5,
    obs_width: float = 0.05,
    theta: float = 0.5,
    amps: Any = (1.0, 0.7),
    interior_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    r"""伴随场对**半离散精确解**的相对偏差随 :math:`\Delta t` 的序列（返回 ``(dt, err)``）。

    两条必须分开的口径（实测差一个整阶）：

    * **内部节点**（去掉两端，``interior_only=True``）上，:math:`\theta=1/2` 的偏差是 :math:`O(\Delta t^2)`；
    * **含端点**（``interior_only=False``）时只剩 :math:`O(\Delta t)`。原因是 Dirichlet 端点用的是"恒等行"，
      :math:`M` 因此**不对称**，伴随要解 :math:`M^T`，端点那一行的耦合方向被翻了过去
      （:math:`M_{0,1}=0` 而 :math:`M_{1,0}=-\\theta r`）。这一阶不是时间格式的阶，
      只出现在紧贴边界的那一层，所以本函数默认按内部节点测。
    """
    dts: list[float] = []
    errs: list[float] = []
    for n in n_steps_list:
        prob = HeatAdjointProblem(
            nx=int(nx),
            length=float(length),
            alpha=float(alpha),
            horizon=float(horizon),
            n_steps=int(n),
            centers=tuple(centers),
            widths=tuple(widths),
            obs_center=float(obs_center),
            obs_width=float(obs_width),
            theta=float(theta),
        )
        mu, _ = heat_discrete_adjoint(prob, np.asarray(amps, dtype=float))
        x = prob.x()
        ks, coeffs = dense_sine_coefficients(x, prob.observation(), float(length))
        ref = semidiscrete_adjoint_field(x, coeffs, float(horizon), nx=int(nx), alpha=float(alpha), length=float(length))
        dx = float(length) / float(int(nx) - 1)
        if interior_only:
            num = float(np.sqrt(np.sum((mu[0][1:-1] - ref[1:-1]) ** 2) * dx))
            den = float(np.sqrt(np.sum(ref[1:-1] ** 2) * dx))
        else:
            num = float(np.sqrt(np.sum((mu[0] - ref) ** 2) * dx))
            den = float(np.sqrt(np.sum(ref * ref) * dx))
        dts.append(float(horizon) / float(n))
        errs.append(num / den if den > 0.0 else float("inf"))
    return np.asarray(dts, dtype=float), np.asarray(errs, dtype=float)


def heat_boundary_layer_ratio(
    *,
    nx: int = 41,
    length: float = 1.0,
    alpha: float = 0.05,
    horizon: float = 0.2,
    n_steps: int = 400,
    obs_center: float = 0.5,
    obs_width: float = 0.05,
    theta: float = 0.5,
    amps: Any = (1.0, 0.7),
) -> dict[str, float]:
    r"""同一算例上"含端点"与"去端点"两种相对偏差的比值（端点一阶层的体积）。

    这个比值 > 1 就是"端点那一层确实把阶拉低了"的直接证据；比值 ≈ 1 说明那一层可以忽略。
    """
    prob = HeatAdjointProblem(
        nx=int(nx), length=float(length), alpha=float(alpha), horizon=float(horizon), n_steps=int(n_steps),
        centers=(0.35, 0.65), widths=(0.08, 0.08), obs_center=float(obs_center), obs_width=float(obs_width),
        theta=float(theta),
    )
    mu, _ = heat_discrete_adjoint(prob, np.asarray(amps, dtype=float))
    x = prob.x()
    ks, coeffs = dense_sine_coefficients(x, prob.observation(), float(length))
    ref = semidiscrete_adjoint_field(x, coeffs, float(horizon), nx=int(nx), alpha=float(alpha), length=float(length))
    dx = float(length) / float(int(nx) - 1)
    den = float(np.sqrt(np.sum(ref[1:-1] ** 2) * dx))
    full = float(np.sqrt(np.sum((mu[0] - ref) ** 2) * dx)) / den
    inner = float(np.sqrt(np.sum((mu[0][1:-1] - ref[1:-1]) ** 2) * dx)) / den
    return {"full": full, "interior": inner, "ratio": (full / inner if inner > 0.0 else float("inf"))}


def _heat_fd_gradient(problem: HeatAdjointProblem, amps: Any, *, eps: float = 1e-6) -> tuple[np.ndarray, WorkCount]:
    r"""热方程目标 :math:`J=w^Tu_N` 的中心差分梯度（负面对照与精度参照都用它）。"""
    pv = np.asarray(amps, dtype=float).reshape(-1)
    grad = np.zeros(pv.size, dtype=float)
    n_solves = 0
    from numcore.inverse.adjoint import heat_forward  # 局部 import：避免顶层循环

    for j in range(pv.size):
        step = float(eps) * max(1.0, abs(float(pv[j])))
        pp = pv.copy()
        pp[j] = pv[j] + step
        pm = pv.copy()
        pm[j] = pv[j] - step
        w = problem.observation()
        jp = float(np.dot(w, heat_forward(problem, pp)[-1]))
        jm = float(np.dot(w, heat_forward(problem, pm)[-1]))
        n_solves += 2
        grad[j] = (jp - jm) / (2.0 * step)
    return grad, WorkCount(n_steps=int(problem.n_steps), n_forward_solves=n_solves)


def heat_gradient_summary(problem: HeatAdjointProblem, amps: Any, *, fd_eps: float = 1e-6) -> dict[str, Any]:
    r"""热方程伴随梯度的一次小结（供帧层直接搬用）：伴随值、差分值、相对误差与代价。"""
    _, grad = heat_discrete_adjoint(problem, amps)
    g_fd, w = _heat_fd_gradient(problem, amps, eps=fd_eps)
    return {
        "grad_adjoint": grad,
        "grad_fd": g_fd,
        "rel_err": relative_error(grad, g_fd),
        "n_fd_solves": int(w.n_forward_solves),
    }


def adjoint_growth_law(
    *,
    dampings: Any = (-0.1, -0.05, 0.0, 0.05, 0.1),
    stiffness: float = 1.0,
    T: float = 60.0,
    n_steps: int = 192,
    method: str = "rk4",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""伴随反向放大率随阻尼**过零换向**：返回 ``(p2, 实测 |\mu_0|/|\mu_N|, 理论 exp(-p2*T/2))``。

    线性振子（``cubic=0``）的雅可比是常数矩阵，其特征值 :math:`\lambda=-p_2/2\pm i\omega_d`
    （:math:`\omega=1` 时 :math:`\zeta=p_2/2`），所以正向一步的压缩率是
    :math:`e^{-p_2T/2}`。转置不改变奇异值，因此**伴随反向递推的放大率就是同一个数**：
    :math:`p_2>0` 时它 <1（往回走衰减）、:math:`p_2=0` 时 =1、:math:`p_2<0` 时 >1。
    这一列数就是"伴随往回走会发散"这句话的适用边界。
    """
    model = duffing_slice_model(0.0)
    pv = np.asarray(dampings, dtype=float).reshape(-1)
    meas = np.empty(pv.size, dtype=float)
    for i, p2 in enumerate(pv):
        pp = np.array([float(stiffness), float(p2)], dtype=float)
        trace = rk_forward(model, pp, T=T, n_steps=int(n_steps), method=method, store="full")
        adj = discrete_adjoint(model, pp, trace)
        meas[i] = adj.growth_ratio()
    theory = np.exp(-pv * float(T) / 2.0)
    return pv, meas, theory


def trajectory_norms(states: Any) -> np.ndarray:
    r"""逐时刻的欧氏范数（帧里画范数曲线用；手写以免碰 P3 的 ``norm`` 属性）。"""
    a = np.asarray(states, dtype=float)
    return np.sqrt(np.sum(a * a, axis=1))


def adjoint_arrow_scale(grad_norm: Any) -> float:
    r"""箭头长度的公共尺度：让最大箭头约为 1，同时给出**不截断到 0** 的地板。

    场里范数跨几个量级时，直接用真实长度画会让小箭头看不见、大箭头糊成一片；
    截断到 0 又会把"这个点的灵敏度接近 0"画成"没有数据"。折中是取对数尺度。
    """
    g = np.asarray(grad_norm, dtype=float).reshape(-1)
    ok = g[np.isfinite(g) & (g > 0.0)]
    if ok.size == 0:
        return 1.0
    return float(np.max(ok))


def log_scaled_length(grad_norm: Any, floor: float = 1e-12) -> np.ndarray:
    r"""把梯度范数压到 ``[0,1]`` 的对数尺度长度（0 给地板，避免对数轴上的 -inf）。"""
    g = np.maximum(np.asarray(grad_norm, dtype=float).reshape(-1), float(floor))
    lg = np.log10(g)
    lo = float(np.min(lg))
    hi = float(np.max(lg))
    return (lg - lo) / (hi - lo) if hi > lo else np.full(lg.size, 0.5)


def adjoint_and_continuous(model: OdeModel, p: Any, *, T: float, n_steps: int, method: str = "rk4") -> tuple[AdjointResult, AdjointResult]:
    r"""同时给出离散伴随与连续伴随（两者之差是本章要分开报的那个量）。"""
    trace = rk_forward(model, p, T=T, n_steps=n_steps, method=method, store="full")
    return discrete_adjoint(model, p, trace), continuous_adjoint(model, p, trace)


def objective_only(model: OdeModel, p: Any, *, T: float, n_steps: int, method: str = "rk4") -> float:
    """只要目标函数值（探针与帧层的小工具）。"""
    return rk_cost(model, p, T=T, n_steps=n_steps, method=method)
