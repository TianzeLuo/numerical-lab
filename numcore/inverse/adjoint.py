r"""``numcore.inverse.adjoint`` —— 离散伴随、连续伴随与梯度检验（5.3 章）。

这一层只做一件事：**把"目标函数对参数的导数"从 O(n) 次正演降到 O(1) 次正演**，
并且把这件事的每一条前提都用可测的数字钉住。

四条口径（写死在实现里，不是注释）：

1. **正演口径沿用 3.1 章**。时间推进一律走 :func:`numcore.ode.rk.rk_step` 与
   :func:`numcore.ode.rk.as_tableau`（Butcher 表、FSAL 复用、级时刻 :math:`c_i`
   都来自那里）；步内插值走 :func:`numcore.ode.rk.hermite_dense`（连续伴随需要
   "正向轨迹在任意时刻的值"）。本模块**不自己写 RK**。
2. **PDE 口径沿用 4.1 章**。热方程的 :math:`\theta` 格式矩阵由
   :func:`numcore.pde.heat.laplacian_1d` 逐列生成，`theta=1/2` 的 Crank--Nicolson
   与 :func:`numcore.pde.heat.theta_step` 逐步一致（有断言对拍）。
3. **解线性系统只走** :mod:`numcore.linalg.lu`（P2/P3：``numpy.linalg`` 与
   ``solve``/``inv`` 这类成品属性在 ``numcore/`` 下是硬违规）。
4. **随机数只走** :func:`numcore.utils.rng`（模型参数由种子生成，P4）。

三个"会被讲糊"的地方，这里各给一个显式接口：

* **离散伴随 vs 连续伴随**：:func:`discrete_adjoint` 给出离散目标函数
  :math:`J_h` 的**精确**梯度（与有限差分对拍到舍入水平）；:func:`continuous_adjoint`
  先把连续伴随方程 :math:`\mu'=-(\partial f/\partial y)^T\mu,\ \mu(T)=(\partial g/\partial y)^T`
  在**同一张表、同一网格**上离散，再拿它算 :math:`\int\mu^T f_p`。两者只在
  :math:`h\to0` 时重合，差是 :math:`O(h^p)`——这正是本章最容易含糊过去的一步。
* **伴随法不免费**：:func:`discrete_adjoint` 需要整条正向轨迹（含每一级的级值），
  :func:`discrete_adjoint_checkpointed` 用分块检查点把它换成"重算一遍"。
  两者的账（求值次数与浮点数存储量）都在 :class:`WorkCount` 里如实记着。
* **梯度检验的 V 形**：:func:`gradient_check_scan` 在同一个 :math:`\varepsilon`
  扫描上同时给出截断误差下降段（斜率 :math:`\approx p`）与舍入误差上升段
  （斜率 :math:`\approx -1`）、**实测最优** :math:`\varepsilon^*`，以及它与
  :math:`u^{1/3}`（中心差分）/ :math:`u^{1/2}`（单侧差分）的比值。
  "ε 越小越准"在这里被数字直接否掉。

伴随梯度的符号约定（下文全部沿用，写错符号是本章最常见的"跑通了但全反"）：

.. math::

    \mu_N=\left(\frac{\partial g}{\partial y_N}\right)^T,\qquad
    \mu_n=\left(\frac{\partial\Phi_n}{\partial y_n}\right)^T\mu_{n+1},\qquad
    \frac{dJ}{dp}=\sum_{n=0}^{N-1}\left(\frac{\partial\Phi_n}{\partial p}\right)^T\mu_{n+1}
    +\left(\frac{\partial y_0}{\partial p}\right)^T\mu_0,

其中 :math:`\Phi_n` 是第 :math:`n` 步的一步法算子。对一张显式 RK 表，
令 :math:`w_i` 是级增量 :math:`k_i` 的反向伴随，则

.. math::

    w_i=h b_i v+h\sum_{j>i}a_{ji}\,(\partial_y f)(Y_j)^Tw_j,\qquad
    \mu_n=v+\sum_i(\partial_y f)(Y_i)^Tw_i,\qquad
    \frac{dJ}{dp}\mathrel{+}=\sum_i(\partial_p f)(Y_i)^Tw_i,

:math:`Y_i` 是第 :math:`i` 级的状态（:math:`Y_i=y_n+h\sum_{j<i}a_{ij}k_j`），
:math:`w` 的递推**级序倒着走**、用的是 :math:`a_{ji}`（转置表）。
写成"连续伴随的形状" :math:`w_i=(\partial_yf)(Y_i)^T(v+h\sum_j a_{ij}w_j)` 看起来更自然，
但它是错的：转置必须作用在**整个乘积**上（实测梯度相对误差会到 0.85–1.08，
见 ``chapters/5.3/pitfalls.md`` 陷阱 1）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from numcore.linalg.lu import lu_decompose, solve_lu
from numcore.ode.rk import as_tableau, hermite_dense, rk_step
from numcore.pde.heat import discrete_sine_eigenvalue, laplacian_1d

__all__ = [
    "UNIT_ROUNDOFF",
    "OdeModel",
    "WorkCount",
    "ForwardTrace",
    "AdjointResult",
    "GradientCheck",
    "CostRow",
    "rk_forward",
    "rk_forward_from",
    "rk_cost",
    "discrete_adjoint",
    "continuous_adjoint",
    "discrete_adjoint_checkpointed",
    "finite_difference_gradient",
    "gradient_check_scan",
    "loglog_slope",
    "relative_error",
    "cost_comparison",
    "HeatAdjointProblem",
    "HeatOperators",
    "heat_operators",
    "heat_forward",
    "heat_discrete_adjoint",
    "heat_forward_matches_theta_step",
    "influence_moments",
    "dense_sine_coefficients",
    "semidiscrete_adjoint_field",
]

#: float64 的舍入单位 :math:`u=2^{-53}`（1.1 章的口径：不是 ``np.finfo`` 的 2.22e-16 提示值）
UNIT_ROUNDOFF = 2.0**-53


# --------------------------------------------------------------------------- #
# 1. 模型接口
# --------------------------------------------------------------------------- #
#: 右端函数签名 ``f(t, y, p) -> (dim,)``；``jac_y`` 返回 ``(dim, dim)``，``jac_p`` 返回 ``(dim, k)``。
RHS = Callable[[float, np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class OdeModel:
    r"""一个带参数的一步法可解问题 :math:`y'=f(t,y,p)`、:math:`y(0)=y_0(p)`、:math:`J=g(y(T))`。

    只收**解析 Jacobian**（``jac_y`` / ``jac_p``）：用有限差分去凑伴随所需的 Jacobian
    会把"伴随一次求值"变成"每步 k 次求值"，本章要测的"代价与参数个数无关"就当场作废。
    """

    name: str
    dim: int
    n_params: int
    rhs: RHS
    jac_y: RHS
    jac_p: RHS
    y0: Callable[[np.ndarray], np.ndarray]
    cost: Callable[[np.ndarray], float]
    cost_grad: Callable[[np.ndarray], np.ndarray]
    dy0_dp: Callable[[np.ndarray], np.ndarray] | None = None
    note: str = ""

    def check_shapes(self, p: Any) -> None:
        """自检：维数与参数个数对得上（造题错了要在造题处就炸，不要留到梯度检验里）。"""
        pv = np.asarray(p, dtype=float).reshape(-1)
        if pv.size != int(self.n_params):
            raise ValueError(f"{self.name}: 参数个数应为 {self.n_params}，得到 {pv.size}")
        y = np.asarray(self.y0(pv), dtype=float).reshape(-1)
        if y.size != int(self.dim):
            raise ValueError(f"{self.name}: 初值维数应为 {self.dim}，得到 {y.size}")
        f = np.asarray(self.rhs(0.0, y, pv), dtype=float)
        jy = np.asarray(self.jac_y(0.0, y, pv), dtype=float)
        jp = np.asarray(self.jac_p(0.0, y, pv), dtype=float)
        if f.shape != (self.dim,) or jy.shape != (self.dim, self.dim) or jp.shape != (self.dim, self.n_params):
            raise ValueError(
                f"{self.name}: 形状不符（f={f.shape}, jac_y={jy.shape}, jac_p={jp.shape}）"
            )
        g = np.asarray(self.cost_grad(y), dtype=float)
        if g.shape != (self.dim,):
            raise ValueError(f"{self.name}: cost_grad 形状应为 ({self.dim},)，得到 {g.shape}")


@dataclass(frozen=True)
class WorkCount:
    r"""一次计算的**确定性**代价账（挂钟时间不进这里：``numcore`` 里没有 ``time``，P6）。

    时间测量属于测试层（``tests/`` 与 ``server/`` 可以 import ``time``），
    这一层只登记"求值次数"这种可复现的量。
    """

    n_steps: int = 0
    n_rhs: int = 0
    n_jac_y: int = 0
    n_jac_p: int = 0
    n_forward_solves: int = 0
    n_linear_solves: int = 0
    n_floats_stored: int = 0

    def total_evals(self) -> int:
        """把一次"残差求值"记 1、一次 Jacobian 求值记 1 的朴素总账（口径写在名字里）。"""
        return int(self.n_rhs + self.n_jac_y + self.n_jac_p)


# --------------------------------------------------------------------------- #
# 2. 正演
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ForwardTrace:
    r"""一次定步长正演的轨迹与代价。

    ``stages`` 是**级导数** :math:`k_i`（形状 ``(n_steps, s, dim)``），
    ``node_derivs`` 是节点导数 :math:`f(t_n,y_n)`（稠密输出用）。
    ``stages=None`` 表示这次正演是"只算 J 不算梯度"的（省内存省一次求值）。
    """

    method: str
    times: np.ndarray
    states: np.ndarray
    stages: np.ndarray | None
    node_derivs: np.ndarray | None
    h: float
    cost: float
    work: WorkCount

    @property
    def n_steps(self) -> int:
        return int(self.states.shape[0] - 1)

    @property
    def dim(self) -> int:
        return int(self.states.shape[1])

    def stage_states(self) -> np.ndarray:
        r"""把级导数还原成级状态 :math:`Y_i=y_n+h\sum_{j<i}a_{ij}k_j`（形状 ``(N, s, dim)``）。"""
        if self.stages is None:
            raise ValueError("这次正演没有保存级值（store='nodes'），无法还原级状态")
        tab = as_tableau(self.method)
        A = tab.A_array()
        s = tab.stages
        ks = self.stages
        Y = np.empty_like(ks)
        for i in range(s):
            acc = np.zeros_like(ks[:, 0, :])
            for j in range(i):
                aij = float(A[i, j])
                if aij != 0.0:
                    acc = acc + aij * ks[:, j, :]
            Y[:, i, :] = self.states[: self.n_steps] + self.h * acc
        return Y


def rk_forward(
    model: OdeModel,
    p: Any,
    *,
    T: float,
    n_steps: int,
    t0: float = 0.0,
    method: str = "rk4",
    store: str = "full",
) -> ForwardTrace:
    r"""定步长正演：:math:`y_{n+1}=\Phi_n(y_n,p)`，顺带把 :math:`J=g(y_N)` 算出来。

    ``store="full"`` 保存级导数与节点导数（伴随要用）；``store="nodes"`` 只留节点
    （有限差分只要目标函数值）。FSAL 复用照 3.1 的账算：重用的那一次不重复计数。
    """
    return rk_forward_from(
        model,
        p,
        np.asarray(model.y0(np.asarray(p, dtype=float).reshape(-1)), dtype=float).reshape(-1),
        t0=t0,
        T=T,
        n_steps=n_steps,
        method=method,
        store=store,
    )


def rk_forward_from(
    model: OdeModel,
    p: Any,
    y_start: Any,
    *,
    t0: float,
    T: float,
    n_steps: int,
    method: str = "rk4",
    store: str = "nodes",
) -> ForwardTrace:
    r"""从给定状态 :math:`y(t_0)` 出发的正演（轨迹播放里的"脉冲扰动"实验要用它）。"""
    if store not in ("full", "nodes"):
        raise ValueError(f"store 只能是 'full' / 'nodes'，得到 {store!r}")
    n = int(n_steps)
    if n < 1:
        raise ValueError(f"n_steps 至少 1，得到 {n}")
    span = float(T) - float(t0)
    if not span > 0.0:
        raise ValueError(f"需要 T > t0，得到 t0={t0}, T={T}")
    tab = as_tableau(method)
    s = tab.stages
    c = tab.c_array()
    pv = np.asarray(p, dtype=float).reshape(-1)
    yy = np.asarray(y_start, dtype=float).reshape(-1).copy()
    h = span / float(n)
    dim = int(yy.size)

    times = np.empty(n + 1, dtype=float)
    states = np.empty((n + 1, dim), dtype=float)
    stages = np.empty((n, s, dim), dtype=float) if store == "full" else None
    derivs = np.empty((n + 1, dim), dtype=float) if store == "full" else None
    times[0] = float(t0)
    states[0] = yy
    n_rhs = 0
    f_start: np.ndarray | None = None
    fsal_ready = False
    for k in range(n):
        t = float(t0) + h * k
        step = rk_step(tab, lambda tt, yv: model.rhs(tt, yv, pv), t, yy, h, f_start=f_start)
        n_rhs += int(step.n_f_evals)
        if stages is not None:
            stages[k] = step.stages
            if float(c[0]) == 0.0:
                derivs[k] = step.stages[0]
            else:  # pragma: no cover - 3.1 的表全都有 c_0=0；留一条正确路径以防万一
                derivs[k] = np.asarray(model.rhs(t, yy, pv), dtype=float)
                n_rhs += 1
        yy = np.asarray(step.y, dtype=float)
        fsal_ready = bool(step.fsal_ready)
        f_start = step.stages[-1] if fsal_ready else None
        times[k + 1] = t + h
        states[k + 1] = yy
    if derivs is not None:
        if fsal_ready:
            derivs[n] = stages[n - 1, -1]
        else:
            derivs[n] = np.asarray(model.rhs(float(T), yy, pv), dtype=float)
            n_rhs += 1
    cost = float(model.cost(states[n]))
    n_floats = int(states.size) + (int(stages.size) if stages is not None else 0)
    n_floats += int(derivs.size) if derivs is not None else 0
    work = WorkCount(
        n_steps=n,
        n_rhs=n_rhs,
        n_forward_solves=1,
        n_floats_stored=n_floats,
    )
    return ForwardTrace(
        method=tab.name,
        times=times,
        states=states,
        stages=stages,
        node_derivs=derivs,
        h=h,
        cost=cost,
        work=work,
    )


def rk_cost(model: OdeModel, p: Any, *, T: float, n_steps: int, t0: float = 0.0, method: str = "rk4") -> float:
    """只要目标函数值 :math:`J_h(p)`（有限差分用；不保存级值）。"""
    return rk_forward(model, p, T=T, n_steps=n_steps, t0=t0, method=method, store="nodes").cost


# --------------------------------------------------------------------------- #
# 3. 离散伴随与连续伴随
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AdjointResult:
    r"""一次伴随解的产物：整条伴随轨迹 :math:`\mu_n`、梯度、以及代价账。"""

    kind: str
    mu: np.ndarray
    grad: np.ndarray
    work: WorkCount
    note: str = ""

    def growth_ratio(self) -> float:
        r"""反向放大率 :math:`\|\mu_0\|/\|\mu_N\|`（正向收缩得越快，反向涨得越猛）。"""
        a = _vec_norm(self.mu[0])
        b = _vec_norm(self.mu[-1])
        return float(a / b) if b > 0.0 else float("inf")


def _vec_norm(v: Any) -> float:
    """欧氏范数（手写：``numpy.linalg.norm`` 在 P2/P3 下是违规属性）。"""
    a = np.asarray(v, dtype=float).reshape(-1)
    return float(np.sqrt(float(np.dot(a, a))))


def discrete_adjoint(model: OdeModel, p: Any, trace: ForwardTrace) -> AdjointResult:
    r"""**先离散、再求伴随**：返回离散目标函数 :math:`J_h` 的精确梯度。

    "精确"是可以当场验证的：与同一 :math:`J_h` 的中心差分相比，误差只受
    :math:`\varepsilon` 的截断/舍入权衡限制（实测最优 :math:`\varepsilon` 处约 1e-9），
    而且这个误差里**没有**任何时间离散误差——离散伴随求的是离散问题的梯度，
    不是"连续梯度的近似"，这是它与 :func:`continuous_adjoint` 的分界线。
    """
    if trace.stages is None or trace.node_derivs is None:
        raise ValueError("离散伴随需要 store='full' 的正演轨迹（级值与节点导数缺一不可）")
    tab = as_tableau(trace.method)
    A = tab.A_array()
    b = tab.b_array()
    c = tab.c_array()
    s = tab.stages
    pv = np.asarray(p, dtype=float).reshape(-1)
    N = trace.n_steps
    dim = trace.dim
    Y = trace.stage_states()
    h = trace.h
    mu = np.empty((N + 1, dim), dtype=float)
    v = np.asarray(model.cost_grad(trace.states[N]), dtype=float).reshape(-1)
    if v.size != dim:
        raise ValueError(f"cost_grad 形状应为 ({dim},)，得到 {v.shape}")
    mu[N] = v
    grad = np.zeros(int(model.n_params), dtype=float)
    n_jy = 0
    n_jp = 0
    for n in range(N - 1, -1, -1):
        tn = float(trace.times[n])
        jy: list[np.ndarray] = []
        ps: list[np.ndarray] = []
        for i in range(s):
            ti = tn + float(c[i]) * h
            jy.append(np.asarray(model.jac_y(ti, Y[n, i], pv), dtype=float))
            n_jy += 1
            ps.append(np.asarray(model.jac_p(ti, Y[n, i], pv), dtype=float))
            n_jp += 1
        w: list[np.ndarray] = [np.zeros(dim, dtype=float)] * s
        for i in range(s - 1, -1, -1):
            acc = (h * float(b[i])) * v
            for j in range(i + 1, s):
                aji = float(A[j, i])
                if aji != 0.0:
                    acc = acc + (h * aji) * (jy[j].T @ w[j])
            w[i] = acc
        v_new = v.copy()
        for i in range(s):
            v_new = v_new + (jy[i].T @ w[i])
            grad = grad + (ps[i].T @ w[i])
        v = v_new
        mu[n] = v
    if model.dy0_dp is not None:
        j0 = np.asarray(model.dy0_dp(pv), dtype=float)
        grad = grad + j0.T @ mu[0]
    work = WorkCount(
        n_steps=N,
        n_rhs=0,
        n_jac_y=n_jy,
        n_jac_p=n_jp,
        n_forward_solves=0,
        n_floats_stored=int(mu.size),
    )
    return AdjointResult(kind="discrete", mu=mu, grad=grad, work=work,
                         note="精确梯度 of J_h（与 h 无关的精确）")


def continuous_adjoint(model: OdeModel, p: Any, trace: ForwardTrace) -> AdjointResult:
    r"""**先求伴随、再离散**：把连续伴随方程在同一张表、同一网格上离散。

    连续伴随（:math:`\mu` 是列向量，:math:`g` 是终端目标）::

        mu' = -(df/dy)^T mu,      mu(T) = (dg/dy)^T,      dJ/dp = int_0^T mu^T f_p dt

    实现上有两处**必须说清楚**、否则就是"看起来对"：

    * 反向积分的级时刻是 :math:`\tau_i=t_{n+1}-c_ih`（不是 :math:`t_n+c_ih`）。
      正向轨迹在 :math:`\tau_i` 的值用 3.1 的 Hermite 稠密输出取，
      所以级时刻与级状态严格配对；级时刻错配会让阶悄悄掉下来。
    * 级值递归走的是 :math:`a_{ij}` 本身（:math:`W_i=v+h\sum_j a_{ij}f_y(\tau_j)^TW_j`），
      而离散伴随走的是 :math:`a_{ij}` **转置后**的结构（:math:`f_y` 在级值外面）。
      对常 Jacobian（线性问题）两者恰好重合；非线性时差 :math:`O(h^p)`——
      这就是"连续伴随 vs 离散伴随"的全部内容。
    """
    if trace.stages is None or trace.node_derivs is None:
        raise ValueError("连续伴随需要 store='full' 的正演轨迹（稠密输出要端点导数）")
    tab = as_tableau(trace.method)
    A = tab.A_array()
    b = tab.b_array()
    c = tab.c_array()
    s = tab.stages
    pv = np.asarray(p, dtype=float).reshape(-1)
    N = trace.n_steps
    dim = trace.dim
    h = trace.h
    mu = np.empty((N + 1, dim), dtype=float)
    v = np.asarray(model.cost_grad(trace.states[N]), dtype=float).reshape(-1)
    mu[N] = v
    grad = np.zeros(int(model.n_params), dtype=float)
    n_jy = 0
    n_jp = 0
    theta = 1.0 - c  # 镜像级时刻：tau_i = t_{n+1} - c_i h = t_n + (1 - c_i) h
    for n in range(N - 1, -1, -1):
        tn = float(trace.times[n])
        Yt = hermite_dense(trace.states[n], trace.states[n + 1], trace.node_derivs[n], trace.node_derivs[n + 1], h, theta)
        taus = tn + theta * h
        jy: list[np.ndarray] = []
        ps: list[np.ndarray] = []
        for i in range(s):
            jy.append(np.asarray(model.jac_y(float(taus[i]), Yt[i], pv), dtype=float))
            n_jy += 1
            ps.append(np.asarray(model.jac_p(float(taus[i]), Yt[i], pv), dtype=float))
            n_jp += 1
        W = np.empty((s, dim), dtype=float)
        for i in range(s):
            acc = v.copy()
            for j in range(i):
                aij = float(A[i, j])
                if aij != 0.0:
                    acc = acc + (h * aij) * (jy[j].T @ W[j])
            W[i] = acc
        for i in range(s):
            bi = float(b[i])
            if bi != 0.0:
                grad = grad + (h * bi) * (ps[i].T @ W[i])
        v = v + h * _contract(jy, W, b)
        mu[n] = v
    if model.dy0_dp is not None:
        j0 = np.asarray(model.dy0_dp(pv), dtype=float)
        grad = grad + j0.T @ mu[0]
    work = WorkCount(n_steps=N, n_rhs=0, n_jac_y=n_jy, n_jac_p=n_jp, n_floats_stored=int(mu.size))
    return AdjointResult(kind="continuous", mu=mu, grad=grad, work=work,
                         note="连续伴随在同一网格上的离散近似（只与 h->0 的极限重合）")


def _contract(jy: Sequence[np.ndarray], W: np.ndarray, b: np.ndarray) -> np.ndarray:
    r""":math:`\sum_i b_i\,(\partial_yf)(\tau_i)^TW_i`（连续伴随的级求和）。"""
    s = len(jy)
    out = np.zeros_like(W[0])
    for i in range(s):
        bi = float(b[i])
        if bi != 0.0:
            out = out + bi * (jy[i].T @ W[i])
    return out


def discrete_adjoint_checkpointed(
    model: OdeModel,
    p: Any,
    *,
    T: float,
    n_steps: int,
    t0: float = 0.0,
    method: str = "rk4",
    block: int | None = None,
) -> AdjointResult:
    r"""**分块检查点**版的离散伴随：用"重算一遍正向"换掉 O(N) 的轨迹存储。

    存储从 :math:`O(N s\\,d)` 降到 :math:`O(\\sqrt N s\\,d)`（取
    :math:`\\text{block}\\approx\\sqrt N`），代价是正向求值次数从 1 倍涨到约 2 倍
    （每一块在反向时重算一次）。梯度与 :func:`discrete_adjoint` **逐位相同**
    （有断言对拍）——否则这个替换就是在偷偷换问题。
    """
    tab = as_tableau(method)
    N = int(n_steps)
    if N < 1:
        raise ValueError(f"n_steps 至少 1，得到 {N}")
    b_len = int(round(math.sqrt(N))) if block is None else int(block)
    b_len = max(1, min(b_len, N))
    pv = np.asarray(p, dtype=float).reshape(-1)
    y = np.asarray(model.y0(pv), dtype=float).reshape(-1).copy()
    dim = int(y.size)
    span = float(T) - float(t0)
    h = span / float(N)

    # --- 阶段 1：只存块边界上的检查点 ---
    n_blocks = int(math.ceil(N / b_len))
    marks = [y.copy()]
    n_rhs = 0
    f_start: np.ndarray | None = None
    for k in range(N):
        step = rk_step(tab, lambda tt, yv: model.rhs(tt, yv, pv), float(t0) + h * k, y, h, f_start=f_start)
        n_rhs += int(step.n_f_evals)
        y = np.asarray(step.y, dtype=float)
        f_start = step.stages[-1] if step.fsal_ready else None
        if (k + 1) % b_len == 0 and k + 1 < N:
            marks.append(y.copy())
    n_marks = len(marks)

    # --- 阶段 2：逐块重算 + 反向递推 ---
    s = tab.stages
    A = tab.A_array()
    bb = tab.b_array()
    cc = tab.c_array()
    mu = np.empty((N + 1, dim), dtype=float)
    v = np.asarray(model.cost_grad(y), dtype=float).reshape(-1)
    mu[N] = v
    grad = np.zeros(int(model.n_params), dtype=float)
    n_jy = 0
    n_jp = 0
    for blk in range(n_blocks - 1, -1, -1):
        start = blk * b_len
        stop = min(start + b_len, N)
        # 重算这一块：从检查点出发，把级值与级状态重新造出来
        yy = marks[blk].copy()
        states = np.empty((stop - start + 1, dim), dtype=float)
        stages = np.empty((stop - start, s, dim), dtype=float)
        states[0] = yy
        fs: np.ndarray | None = None
        for k in range(start, stop):
            step = rk_step(tab, lambda tt, yv: model.rhs(tt, yv, pv), float(t0) + h * k, yy, h, f_start=fs)
            n_rhs += int(step.n_f_evals)
            stages[k - start] = step.stages
            yy = np.asarray(step.y, dtype=float)
            fs = step.stages[-1] if step.fsal_ready else None
            states[k - start + 1] = yy
        for k in range(stop - 1, start - 1, -1):
            tn = float(t0) + h * k
            ks = stages[k - start]
            jy = []
            ps = []
            for i in range(s):
                acc = np.zeros(dim, dtype=float)
                for j in range(i):
                    aij = float(A[i, j])
                    if aij != 0.0:
                        acc = acc + aij * ks[j]
                Yi = states[k - start] + h * acc
                ti = tn + float(cc[i]) * h
                jy.append(np.asarray(model.jac_y(ti, Yi, pv), dtype=float))
                n_jy += 1
                ps.append(np.asarray(model.jac_p(ti, Yi, pv), dtype=float))
                n_jp += 1
            w = [np.zeros(dim, dtype=float)] * s
            for i in range(s - 1, -1, -1):
                acc = (h * float(bb[i])) * v
                for j in range(i + 1, s):
                    aji = float(A[j, i])
                    if aji != 0.0:
                        acc = acc + (h * aji) * (jy[j].T @ w[j])
                w[i] = acc
            v_new = v.copy()
            for i in range(s):
                v_new = v_new + (jy[i].T @ w[i])
                grad = grad + (ps[i].T @ w[i])
            v = v_new
            mu[k] = v
    if model.dy0_dp is not None:
        j0 = np.asarray(model.dy0_dp(pv), dtype=float)
        grad = grad + j0.T @ mu[0]
    floats = n_marks * dim + b_len * s * dim + int(mu.size)
    work = WorkCount(
        n_steps=N,
        n_rhs=n_rhs,
        n_jac_y=n_jy,
        n_jac_p=n_jp,
        n_floats_stored=int(floats),
    )
    return AdjointResult(kind="discrete-checkpointed", mu=mu, grad=grad, work=work,
                         note=f"分块检查点（block={b_len}, 检查点 {n_marks} 个）")


# --------------------------------------------------------------------------- #
# 4. 有限差分与梯度检验
# --------------------------------------------------------------------------- #
def relative_error(a: Any, b: Any) -> float:
    r"""对称相对误差 :math:`\|a-b\|_\infty/\max(1,\|a\|_\infty,\|b\|_\infty)`。

    分母带 1 的地板是**故意**的：目标函数尺度可以远小于 1（差分类问题里很常见），
    纯相对误差会在 :math:`|g|` 很小时把舍入噪声放大成"误差很大"的假象。
    """
    av = np.asarray(a, dtype=float).reshape(-1)
    bv = np.asarray(b, dtype=float).reshape(-1)
    if av.size != bv.size:
        raise ValueError(f"长度不一致：{av.size} vs {bv.size}")
    num = float(np.max(np.abs(av - bv))) if av.size else 0.0
    den = max(1.0, float(np.max(np.abs(av))) if av.size else 0.0, float(np.max(np.abs(bv))) if av.size else 0.0)
    return float(num / den)


def finite_difference_gradient(
    model: OdeModel,
    p: Any,
    *,
    T: float,
    n_steps: int,
    t0: float = 0.0,
    method: str = "rk4",
    eps: float = 1e-6,
    scheme: str = "central",
    scales: Any = None,
) -> tuple[np.ndarray, WorkCount]:
    r"""有限差分梯度（**同一** :math:`J_h`、同一网格、同一步长）。

    步长按分量缩放：:math:`h_j=\varepsilon\cdot s_j`，:math:`s_j=\max(1,|p_j|)`
    （``scales`` 可覆盖）。用绝对步长去扰一个 :math:`10^{-3}` 量级的参数，
    测到的其实是"扰动为 0"的假象；用相对步长去扰一个过零的参数，测到的是舍入——
    两种错法都会让 V 形曲线看起来"形状对、位置全错"。
    """
    pv = np.asarray(p, dtype=float).reshape(-1)
    if scales is None:
        sc = np.maximum(1.0, np.abs(pv))
    else:
        sc = np.asarray(scales, dtype=float).reshape(-1)
        if sc.size != pv.size:
            raise ValueError(f"scales 长度应为 {pv.size}，得到 {sc.size}")
    k = int(pv.size)
    grad = np.zeros(k, dtype=float)
    n_solves = 0
    n_rhs = 0
    if scheme == "central":
        for j in range(k):
            step = float(eps) * float(sc[j])
            pp = pv.copy()
            pp[j] = pv[j] + step
            pm = pv.copy()
            pm[j] = pv[j] - step
            tr_p = rk_forward(model, pp, T=T, n_steps=n_steps, t0=t0, method=method, store="nodes")
            tr_m = rk_forward(model, pm, T=T, n_steps=n_steps, t0=t0, method=method, store="nodes")
            n_solves += 2
            n_rhs += int(tr_p.work.n_rhs + tr_m.work.n_rhs)
            grad[j] = (tr_p.cost - tr_m.cost) / (2.0 * step)
    elif scheme == "forward":
        base = rk_forward(model, pv, T=T, n_steps=n_steps, t0=t0, method=method, store="nodes")
        n_solves += 1
        n_rhs += int(base.work.n_rhs)
        for j in range(k):
            step = float(eps) * float(sc[j])
            pp = pv.copy()
            pp[j] = pv[j] + step
            tr_p = rk_forward(model, pp, T=T, n_steps=n_steps, t0=t0, method=method, store="nodes")
            n_solves += 1
            n_rhs += int(tr_p.work.n_rhs)
            grad[j] = (tr_p.cost - base.cost) / step
    elif scheme == "backward":
        base = rk_forward(model, pv, T=T, n_steps=n_steps, t0=t0, method=method, store="nodes")
        n_solves += 1
        n_rhs += int(base.work.n_rhs)
        for j in range(k):
            step = float(eps) * float(sc[j])
            pm = pv.copy()
            pm[j] = pv[j] - step
            tr_m = rk_forward(model, pm, T=T, n_steps=n_steps, t0=t0, method=method, store="nodes")
            n_solves += 1
            n_rhs += int(tr_m.work.n_rhs)
            grad[j] = (base.cost - tr_m.cost) / step
    else:
        raise ValueError(f"未知差分格式 {scheme!r}（central / forward / backward）")
    work = WorkCount(n_steps=int(n_steps), n_rhs=n_rhs, n_forward_solves=n_solves)
    return grad, work


def loglog_slope(x: Any, y: Any, mask: Any = None) -> float | None:
    r"""双对数最小二乘斜率 :math:`p`（:math:`y\approx Cx^p`）；点数 < 2 时返回 ``None``。

    只喂**幂律**关系。对数型关系（例如"有效位数 ~ log10(ε)"）拟合出来的 p 没有意义，
    V 形曲线的两段恰好都是幂律（截断 :math:`\varepsilon^p`、舍入 :math:`u/\varepsilon`），
    所以这里能用；用之前先确认两段各自的点数与区间（窗口写进断言）。
    """
    xv = np.asarray(x, dtype=float).reshape(-1)
    yv = np.asarray(y, dtype=float).reshape(-1)
    if mask is not None:
        m = np.asarray(mask).astype(bool).reshape(-1)
        xv = xv[m]
        yv = yv[m]
    ok = np.isfinite(xv) & np.isfinite(yv) & (xv > 0.0) & (yv > 0.0)
    xv = np.log(xv[ok])
    yv = np.log(yv[ok])
    if xv.size < 2:
        return None
    xc = xv - float(np.mean(xv))
    yc = yv - float(np.mean(yv))
    den = float(np.dot(xc, xc))
    if den <= 0.0:
        return None
    return float(np.dot(xc, yc) / den)


@dataclass(frozen=True)
class GradientCheck:
    r"""一次 :math:`\varepsilon` 扫描的全部证据（V 形的两段 + 实测最优 + 代价账）。"""

    eps: np.ndarray
    err: dict[str, np.ndarray]
    best_index: dict[str, int]
    slope_down: dict[str, float | None]
    slope_up: dict[str, float | None]
    adjoint_grad: np.ndarray
    fd_best: dict[str, np.ndarray]
    work: dict[str, WorkCount]
    n_params: int
    unit_roundoff: float

    def best_eps(self, scheme: str) -> float | None:
        i = self.best_index.get(scheme, -1)
        if i is None or i < 0:
            return None
        return float(self.eps[i])

    def best_err(self, scheme: str) -> float | None:
        i = self.best_index.get(scheme, -1)
        if i is None or i < 0:
            return None
        e = self.err.get(scheme)
        return None if e is None else float(e[i])

    def optimum_ratio(self, scheme: str, exponent: str) -> float | None:
        r"""实测最优 :math:`\varepsilon^*` 与理论量级 :math:`u^{1/q}` 的比值。

        ``exponent="1/3"``（中心差分：截断 :math:`\varepsilon^2` vs 舍入 :math:`u/\varepsilon`）
        或 ``"1/2"``（单侧差分：截断 :math:`\varepsilon` vs 舍入 :math:`u/\varepsilon`）。
        """
        e = self.best_eps(scheme)
        if e is None:
            return None
        q = {"1/3": 3.0, "1/2": 2.0}.get(str(exponent))
        if q is None:
            raise ValueError(f"未知指数 {exponent!r}")
        return float(e / (self.unit_roundoff ** (1.0 / q)))


def gradient_check_scan(
    model: OdeModel,
    p: Any,
    *,
    T: float,
    n_steps: int,
    t0: float = 0.0,
    method: str = "rk4",
    eps_values: Sequence[float],
    schemes: Sequence[str] = ("central", "forward"),
    scales: Any = None,
) -> GradientCheck:
    r"""在同一个 :math:`\varepsilon` 扫描上做梯度检验（本章的招牌证据）。

    对每个 :math:`\varepsilon`：伴随梯度只算一次（它**与** :math:`\varepsilon` 无关，
    这本身就是伴随法的卖点），有限差分梯度按格式各算一次，然后记
    :math:`\|\nabla J_{fd}-\nabla J_{adj}\|_\infty` 的对称相对误差。

    返回的两段斜率分别取自"最优点的左侧"与"右侧"（各至少 2 点）：
    截断段应该看到 :math:`p`（中心差分 :math:`p=2`、单侧 :math:`p=1`），
    舍入段应该看到 :math:`-1`。**最优点必须落在扫描区间内部**——落在端点上说明
    扫描窗口没盖住 V 形，那时斜率与最优 :math:`\varepsilon` 都不该被当成结论。
    """
    epsv = np.asarray([float(e) for e in eps_values], dtype=float)
    if epsv.size < 3:
        raise ValueError(f"eps 扫描至少 3 个点，得到 {epsv.size}")
    if np.any(epsv <= 0.0) or np.any(~np.isfinite(epsv)):
        raise ValueError("eps 必须全为正的有限数")
    trace = rk_forward(model, p, T=T, n_steps=n_steps, t0=t0, method=method, store="full")
    adj = discrete_adjoint(model, p, trace)
    err: dict[str, np.ndarray] = {}
    best_index: dict[str, int] = {}
    slope_down: dict[str, float | None] = {}
    slope_up: dict[str, float | None] = {}
    fd_best: dict[str, np.ndarray] = {}
    work: dict[str, WorkCount] = {}
    for sch in schemes:
        errs = np.empty(epsv.size, dtype=float)
        grads: list[np.ndarray] = []
        wsum = WorkCount(n_steps=int(n_steps))
        for i, e in enumerate(epsv):
            g, w = finite_difference_gradient(
                model, p, T=T, n_steps=n_steps, t0=t0, method=method, eps=float(e), scheme=sch, scales=scales
            )
            errs[i] = relative_error(g, adj.grad)
            grads.append(g)
            wsum = WorkCount(
                n_steps=int(n_steps),
                n_rhs=wsum.n_rhs + int(w.n_rhs),
                n_forward_solves=wsum.n_forward_solves + int(w.n_forward_solves),
            )
        err[sch] = errs
        work[sch] = wsum
        i_star = int(np.argmin(errs))
        best_index[sch] = i_star
        fd_best[sch] = grads[i_star]
        left = np.zeros(epsv.size, dtype=bool)
        right = np.zeros(epsv.size, dtype=bool)
        left[:i_star] = True
        right[i_star + 1 :] = True
        slope_down[sch] = loglog_slope(epsv, errs, left)
        slope_up[sch] = loglog_slope(epsv, errs, right)
    return GradientCheck(
        eps=epsv,
        err=err,
        best_index=best_index,
        slope_down=slope_down,
        slope_up=slope_up,
        adjoint_grad=adj.grad,
        fd_best=fd_best,
        work=work,
        n_params=int(np.asarray(p, dtype=float).size),
        unit_roundoff=UNIT_ROUNDOFF,
    )


# --------------------------------------------------------------------------- #
# 5. 代价账：伴随 vs 有限差分，以及检查点
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CostRow:
    r"""给定参数个数的一行代价账（求解次数 / 求值次数 / 存储浮点数）。"""

    n_params: int
    adjoint_rhs: int
    adjoint_jac: int
    adjoint_solves: int
    adjoint_floats: int
    checkpoint_rhs: int
    checkpoint_floats: int
    checkpoint_block: int
    fd_rhs: int
    fd_solves: int
    grad_rel_err_checkpoint: float

    def fd_over_adjoint_evals(self) -> float:
        """有限差分求值次数 / 伴随求值次数（伴随 = 正向残差求值 + 反向 Jacobian 求值）。"""
        a = float(self.adjoint_rhs + self.adjoint_jac)
        return float(self.fd_rhs) / a if a > 0.0 else float("inf")

    def storage_reduction(self) -> float:
        """检查点相对"全存"省下的浮点数倍数。"""
        return float(self.adjoint_floats) / float(self.checkpoint_floats) if self.checkpoint_floats > 0 else float("inf")


def cost_comparison(
    cases: Sequence[tuple[OdeModel, Any]],
    *,
    T: float,
    n_steps: int,
    t0: float = 0.0,
    method: str = "rk4",
    block: int | None = None,
    fd_scheme: str = "central",
) -> list[CostRow]:
    r"""对每个 ``(model, p)`` 测三笔账：伴随、分块检查点伴随、有限差分。

    ``cases`` 里每个模型有自己的参数个数 :math:`k`；正演与网格完全相同。
    返回的每行都带"检查点梯度与全存梯度"的相对误差——这一项必须是舍入级，
    否则"省内存"省掉的是正确性。
    """
    rows: list[CostRow] = []
    for model, p in cases:
        trace = rk_forward(model, p, T=T, n_steps=n_steps, t0=t0, method=method, store="full")
        adj = discrete_adjoint(model, p, trace)
        ck = discrete_adjoint_checkpointed(model, p, T=T, n_steps=n_steps, t0=t0, method=method, block=block)
        _, fdw = finite_difference_gradient(
            model, p, T=T, n_steps=n_steps, t0=t0, method=method, eps=1e-6, scheme=fd_scheme
        )
        k = int(np.asarray(p, dtype=float).size)
        rows.append(
            CostRow(
                n_params=k,
                adjoint_rhs=int(trace.work.n_rhs),
                adjoint_jac=int(adj.work.n_jac_y + adj.work.n_jac_p),
                adjoint_solves=1,
                adjoint_floats=int(trace.work.n_floats_stored),
                checkpoint_rhs=int(ck.work.n_rhs),
                checkpoint_floats=int(ck.work.n_floats_stored),
                checkpoint_block=int(round(math.sqrt(int(n_steps)))) if block is None else int(block),
                fd_rhs=int(fdw.n_rhs),
                fd_solves=int(fdw.n_forward_solves),
                grad_rel_err_checkpoint=relative_error(ck.grad, adj.grad),
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# 6. PDE 约束：热方程的 theta 格式与它的离散伴随
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HeatAdjointProblem:
    r"""一维热方程约束的**线性**目标：:math:`J(p)=w^Tu_N`。

    :math:`u_t=\alpha u_{xx}+\sum_jp_j\varphi_j(x)`，Dirichlet 两端为零，
    :math:`u(x,0)=0`；观测是终端时刻的一条钟形权重 :math:`w`（宽度 ``obs_width``，
    中心 ``obs_center``）。参数**只进源项**：这样 :math:`\partial s/\partial p_j`
    与状态无关，离散伴随梯度是 :math:`\sum_n\mu_{n+1}^T\partial s_n/\partial p_j`
    的有限和——不是"近似"，可以逐位对拍。
    """

    nx: int
    length: float
    alpha: float
    horizon: float
    n_steps: int
    centers: tuple[float, ...]
    widths: tuple[float, ...]
    obs_center: float
    obs_width: float
    theta: float = 0.5

    @property
    def n_params(self) -> int:
        return len(self.centers)

    def x(self) -> np.ndarray:
        return np.linspace(0.0, float(self.length), int(self.nx))

    def source_basis(self) -> np.ndarray:
        r"""源项基 :math:`\varphi_j`（形状 ``(k, nx)``，两端点清零：Dirichlet 行钉住端点）。"""
        x = self.x()
        out = np.empty((self.n_params, int(self.nx)), dtype=float)
        for j, (c, wd) in enumerate(zip(self.centers, self.widths)):
            g = np.exp(-0.5 * ((x - float(c)) / float(wd)) ** 2)
            g[0] = 0.0
            g[-1] = 0.0
            out[j] = g
        return out

    def observation(self) -> np.ndarray:
        r"""观测权重 :math:`w`（含 :math:`\Delta x` 因子，把 :math:`w^Tu` 变成积分近似）。"""
        x = self.x()
        dx = float(self.length) / float(int(self.nx) - 1)
        w = np.exp(-0.5 * ((x - float(self.obs_center)) / float(self.obs_width)) ** 2) * dx
        w[0] = 0.0
        w[-1] = 0.0
        return w

    def bump_variance(self) -> float:
        r"""观测权重自身的方差 :math:`\sigma_0^2`（离散二阶中心矩，理论宽度律的起点）。"""
        x = self.x()
        w = self.observation()
        _, _, var = influence_moments(x, w)
        return float(var)


@dataclass(frozen=True)
class HeatOperators:
    r"""一个 :math:`\theta` 格式的矩阵对 :math:`M=I-\theta r\tilde D`、:math:`G`（Dirichlet 行清零）。"""

    M: np.ndarray
    G: np.ndarray
    dt: float
    dx: float
    r: float
    theta: float
    nx: int


def heat_operators(problem: HeatAdjointProblem) -> HeatOperators:
    r"""由 4.1 章的 :func:`laplacian_1d` **逐列**生成离散矩阵（保证与 ``theta_step`` 同源）。

    ``G`` 的端点行被清零：``theta_system`` 在 Dirichlet 下把右端两行**覆盖**成端点值，
    于是那两行对 :math:`u^n` 的偏导恒为 0。这一步漏掉的话，伴随会在端点凭空多出
    一条耦合，梯度检验在端点附近立刻现形。
    """
    nx = int(problem.nx)
    dx = float(problem.length) / float(nx - 1)
    dt = float(problem.horizon) / float(problem.n_steps)
    r = float(problem.alpha) * dt / (dx * dx)
    D = np.empty((nx, nx), dtype=float)
    for j in range(nx):
        e = np.zeros(nx, dtype=float)
        e[j] = 1.0
        D[:, j] = laplacian_1d(e, dx, bc="dirichlet")
    eye = np.eye(nx, dtype=float)
    M = eye - float(problem.theta) * r * D
    G = eye + (1.0 - float(problem.theta)) * r * D
    G[0, :] = 0.0
    G[-1, :] = 0.0
    return HeatOperators(M=M, G=G, dt=dt, dx=dx, r=r, theta=float(problem.theta), nx=nx)


def heat_forward(problem: HeatAdjointProblem, amps: Any) -> np.ndarray:
    r"""正演：:math:`u^{n+1}=M^{-1}(Gu^n+\Delta t\sum_jp_j\varphi_j)`（返回 ``(N+1, nx)``）。"""
    ops = heat_operators(problem)
    basis = problem.source_basis()
    pv = np.asarray(amps, dtype=float).reshape(-1)
    if pv.size != problem.n_params:
        raise ValueError(f"源项幅度个数应为 {problem.n_params}，得到 {pv.size}")
    src = ops.dt * (pv @ basis)
    src[0] = 0.0
    src[-1] = 0.0
    res = lu_decompose(ops.M)
    u = np.zeros(ops.nx, dtype=float)
    out = np.empty((int(problem.n_steps) + 1, ops.nx), dtype=float)
    out[0] = u
    for n in range(int(problem.n_steps)):
        u = solve_lu(res, ops.G @ u + src)
        out[n + 1] = u
    return out


def heat_forward_matches_theta_step(problem: HeatAdjointProblem, amps: Any, n_check: int = 5) -> float:
    r"""把本模块的矩阵推进与 4.1 的 :func:`numcore.pde.heat.theta_step` 对拍（返回最大偏差）。"""
    from numcore.pde.heat import theta_step  # 局部 import：只用在对拍函数里，避免循环依赖

    ops = heat_operators(problem)
    basis = problem.source_basis()
    pv = np.asarray(amps, dtype=float).reshape(-1)
    src = ops.dt * (pv @ basis)
    src[0] = 0.0
    src[-1] = 0.0
    res = lu_decompose(ops.M)
    u_a = np.zeros(ops.nx, dtype=float)
    u_b = np.zeros(ops.nx, dtype=float)
    worst = 0.0
    for n in range(int(problem.n_steps)):
        u_a = solve_lu(res, ops.G @ u_a + src)
        u_b = theta_step(
            u_b,
            r=ops.r,
            theta=ops.theta,
            dx=ops.dx,
            dt=ops.dt,
            bc="dirichlet",
            left=0.0,
            right=0.0,
            source=pv @ basis,
        )
        if n % max(1, int(problem.n_steps) // max(1, int(n_check))) == 0:
            worst = max(worst, float(np.max(np.abs(u_a - u_b))))
    worst = max(worst, float(np.max(np.abs(u_a - u_b))))
    return worst


def heat_discrete_adjoint(
    problem: HeatAdjointProblem,
    amps: Any,
    *,
    transposed: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    r"""热方程 :math:`\theta` 格式的离散伴随：返回 ``(mu (N+1, nx), grad (k,))``。

    递推是 :math:`\mu_n=G^TM^{-T}\mu_{n+1}`、:math:`\mu_N=w`，梯度是
    :math:`\sum_n\mu_{n+1}^T(\Delta t\,\varphi_j)`。

    ``transposed=False`` 是**负对照**：把 :math:`M^{-T}` 换成 :math:`M^{-1}`。
    看上去"矩阵本来就是对称三对角"，但 Dirichlet 的恒等行破坏了对称性
    （:math:`M_{0,1}=0` 而 :math:`M_{1,0}=-\theta r`），换掉之后伴随的端点耦合
    整体消失，梯度当场对不上有限差分——这正是本章要写进 pitfalls 的那条。
    """
    ops = heat_operators(problem)
    basis = problem.source_basis()
    pv = np.asarray(amps, dtype=float).reshape(-1)
    if pv.size != int(problem.n_params):
        raise ValueError(f"参数个数不对：给 {pv.size} 个，问题要 {problem.n_params} 个")
    src_unit = ops.dt * basis
    src_unit[:, 0] = 0.0
    src_unit[:, -1] = 0.0
    w = problem.observation()
    N = int(problem.n_steps)
    mu = np.empty((N + 1, ops.nx), dtype=float)
    mu[N] = w
    res = lu_decompose(ops.M.T if transposed else ops.M)
    Gt = ops.G.T
    grad = np.zeros(problem.n_params, dtype=float)
    v = w
    for n in range(N - 1, -1, -1):
        x = solve_lu(res, v)
        grad = grad + src_unit @ x
        v = Gt @ x
        mu[n] = v
    return mu, grad


def influence_moments(x: Any, field: Any, dx: float | None = None) -> tuple[float, float, float]:
    r"""一维场的前两阶矩 ``(mass, mean, variance)``（相对"质量"归一，不用任何统计成品）。

    方差用 :math:`\sum_i\rho_i(x_i-\bar x)^2`（:math:`\rho` 是归一化权重）。
    """
    xv = np.asarray(x, dtype=float).reshape(-1)
    fv = np.asarray(field, dtype=float).reshape(-1)
    if xv.size != fv.size:
        raise ValueError(f"网格与场长度不一致：{xv.size} vs {fv.size}")
    mass = float(np.sum(fv)) * (1.0 if dx is None else float(dx))
    tot = float(np.sum(fv))
    if tot == 0.0:
        return 0.0, float("nan"), float("nan")
    rho = fv / tot
    mean = float(np.sum(rho * xv))
    var = float(np.sum(rho * (xv - mean) ** 2))
    return mass, mean, var


def dense_sine_coefficients(x: Any, field: Any, length: float, n_modes: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    r"""把场展开成离散正弦基 :math:`v_k=\sin(k\pi x/L)`：返回 ``(k 数组, 系数)``。

    系数用离散正交归一 :math:`c_k=\langle f,v_k\rangle/\langle v_k,v_k\rangle`——
    解析归一化常数（:math:`2/L` 还是 :math:`2/(L+dx)`）写错时，展开式在**内部点**上
    依然像模像样，只有端点会露馅，所以这里不给自己留这个坑。
    """
    xv = np.asarray(x, dtype=float).reshape(-1)
    fv = np.asarray(field, dtype=float).reshape(-1)
    n = int(xv.size)
    kmax = (n - 2) if n_modes is None else int(n_modes)
    kmax = max(1, min(kmax, n - 2))
    ks = np.arange(1, kmax + 1, dtype=float)
    V = np.sin(np.pi * np.outer(ks, xv) / float(length))  # (kmax, n)
    num = V @ fv
    den = np.sum(V * V, axis=1)
    return ks, num / den


def semidiscrete_adjoint_field(
    x: Any,
    coeffs: Any,
    tau: Any,
    *,
    nx: int,
    alpha: float,
    length: float = 1.0,
) -> np.ndarray:
    r"""半离散精确伴随场 :math:`\sum_kc_k\sin(k\pi x/L)e^{-\lambda_k\tau}`。

    :math:`\lambda_k` 用 4.1 的**离散**特征值（:func:`numcore.pde.heat.discrete_sine_eigenvalue`），
    所以这个靶子里没有任何空间截断误差——拿它测时间格式，测到的才是时间阶。
    ``tau`` 可以是标量（返回 ``(nx,)``）或数组（返回 ``(len(tau), nx)``）。
    """
    xv = np.asarray(x, dtype=float).reshape(-1)
    cv = np.asarray(coeffs, dtype=float).reshape(-1)
    tv = np.asarray(tau, dtype=float).reshape(-1)
    ks = np.arange(1, cv.size + 1, dtype=float)
    V = np.sin(np.pi * np.outer(ks, xv) / float(length))  # (kmax, nx)
    lams = np.array(
        [discrete_sine_eigenvalue(int(nx), int(k), alpha=float(alpha), length=float(length)) for k in ks],
        dtype=float,
    )
    decay = np.exp(-np.outer(tv, lams))  # (ntau, kmax)
    out = (decay * cv[None, :]) @ V  # (ntau, nx)
    out[:, 0] = 0.0
    out[:, -1] = 0.0
    return out[0] if np.isscalar(tau) or np.asarray(tau).ndim == 0 else out


def heat_adjoint_work(problem: HeatAdjointProblem, n_forward_solves: int, n_adjoint_solves: int) -> WorkCount:
    """PDE 侧的代价账（线性问题：伴随不需要正向轨迹，存储只有 :math:`O(n_x)`）。"""
    ops = heat_operators(problem)
    return WorkCount(
        n_steps=int(problem.n_steps),
        n_rhs=0,
        n_forward_solves=int(n_forward_solves),
        n_linear_solves=int((n_forward_solves + n_adjoint_solves) * int(problem.n_steps)),
        n_floats_stored=int(ops.nx) * (int(problem.n_steps) + 1),
    )
