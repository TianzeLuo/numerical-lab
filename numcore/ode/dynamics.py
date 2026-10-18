r"""3.5 章（动力系统、分岔与混沌）的**系统、映射与几何**真源。

数学对象
--------

自治系统 :math:`\dot y=f(y)`、离散映射 :math:`x_{n+1}=g(x_n)`，以及围绕它们的四件事：

1. **平衡点与线性化**。:math:`f(y_*)=0` 处的 Jacobi 矩阵 :math:`J=Df(y_*)` 决定局部相型：
   :math:`\operatorname{Re}\lambda_i<0` 全负 ⟹ 渐近稳定（Hartman--Grobman 保证局部拓扑共轭），
   有正有负 ⟹ 鞍点（稳定/不稳定流形沿特征方向切触），纯虚对 ⟹ 线性化给不出结论（中心）。
   本章的判据**只用**特征值，判据本身由断言与闭式解对拍。
2. **零倾线**（nullcline）。:math:`\dot x=0` 与 :math:`\dot y=0` 两条曲线；它们的交点就是平衡点。
   本模块给两条独立路径：**数值追踪**（网格符号变化 + 二分，见 :func:`zero_curves`）与
   **闭式曲线**（见 :func:`analytic_nullclines`）——两条路径在断言里逐点对拍。
3. **Poincaré 截面**。轨道穿过超平面 :math:`\Sigma=\{y:\ h(y)=0\}` 的点集就是截面；
   首次返回映射 :math:`P` 把 :math:`\Sigma` 映到自己。本模块**不**用稠密输出，
   而是用「线性插值定位 + 从同一点出发按比例再走一小步」的割线精化
   （见 :func:`poincare_crossings`），精化后的平面残差是实测的，不是假设的。
4. **三维显示投影**。:math:`\mathbb{R}^3` 的原始数据一律照给（``*_x/_y/_z``），
   另给一份**正交投影**的屏幕坐标 :math:`(u,v)`：方位角 :math:`\alpha`、仰角 :math:`\varepsilon`

   .. math::
      u=-\sin\alpha\,x+\cos\alpha\,y,\qquad
      v=-\cos\alpha\sin\varepsilon\,x-\sin\alpha\sin\varepsilon\,y+\cos\varepsilon\,z .

   这是**纯显示空间**的变换：无透视、深度不可读（远近不改变大小），量值必须看数字。

自研边界（P2/P3）
-----------------
本模块只用 numpy 的容器与原语：**不碰**任何现成求解器。特征值是自写的
Durand--Kerner 同时迭代（:func:`poly_roots`），特征向量用二维零空间公式
（:func:`eigenvector_2d`），平面系统的非线性平衡点用 Newton + 3.2 章的自研 LU
（:func:`newton_root`）。轨迹积分**复用 3.1 章** :func:`numcore.ode.rk.integrate_fixed`
与 :func:`numcore.ode.rk.rk_step`，本章不再写一套 RK。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from numcore.linalg.lu import lu_decompose, solve_lu
from numcore.ode import rk as rkmod

__all__ = [
    "BLOWUP",
    "FlowRHS",
    "MapRHS",
    "FlowSystem",
    "EnsembleRun",
    "SectionResult",
    "FLOW_NAMES",
    "MAP_NAMES",
    "FLOW_LABELS",
    "KIND_CODES",
    "KIND_NAMES",
    "lorenz_rhs",
    "lorenz_jac",
    "lorenz_fixed_points",
    "lorenz_hopf_rho",
    "lorenz_divergence",
    "rossler_rhs",
    "rossler_jac",
    "rossler_fixed_points",
    "van_der_pol_rhs",
    "van_der_pol_jac",
    "van_der_pol_fixed_points",
    "damped_pendulum_rhs",
    "damped_pendulum_jac",
    "damped_pendulum_fixed_points",
    "hopf_rhs",
    "hopf_jac",
    "hopf_fixed_points",
    "flow_system",
    "flow_names",
    "logistic",
    "logistic_deriv",
    "henon",
    "henon_jac",
    "henon_fixed_points",
    "map_callable",
    "char_poly_coeffs",
    "poly_roots",
    "eigenvalues",
    "eigenvector_2d",
    "classify_eigenvalues",
    "fixed_point_report",
    "newton_root",
    "vector_field",
    "zero_curves",
    "nullcline_branches",
    "analytic_nullclines",
    "trajectory",
    "integrate_ensemble",
    "poincare_crossings",
    "project3d",
    "projection_matrix",
    "window_of",
]

#: 未爆炸的显示上限（与 3.1 章同口径）：超过它一律停止并如实标注，``inf`` 进不了帧。
BLOWUP = 1.0e12

#: 右端函数：``f(t, y) -> dy``，支持 ``y`` 形状 ``(dim,)`` 与 ``(n, dim)``（批量）。
FlowRHS = Callable[[float, np.ndarray], np.ndarray]

#: 映射：``x -> g(x)``，支持 ``x`` 形状 ``(dim,)`` 与 ``(n, dim)``。
MapRHS = Callable[[np.ndarray], np.ndarray]

#: 平面系统的相型编码（进帧的是整数，名字与中文说明在 :data:`KIND_NAMES` 里）。
KIND_CODES: dict[str, int] = {
    "stable_node": 0,
    "stable_focus": 1,
    "center": 2,
    "saddle": 3,
    "unstable_node": 4,
    "unstable_focus": 5,
    "degenerate": 6,
    #: 混合符号 + 复共轭对：一个稳定实方向 + 一个不稳定螺旋面。
    #: 这不是"鞍点"（鞍点的两个方向都是实的），Lorenz 在 ρ=28 的 C± 正是这一型
    #: （[−13.85, 0.094±10.19i]），文献里叫 saddle-focus。
    "saddle_focus": 7,
}

KIND_NAMES: dict[str, str] = {
    "stable_node": "稳定结点",
    "stable_focus": "稳定焦点",
    "center": "中心（线性化不判定）",
    "saddle": "鞍点",
    "unstable_node": "不稳定结点",
    "unstable_focus": "不稳定焦点",
    "saddle_focus": "鞍焦（一个稳定方向 + 一个不稳定螺旋面）",
    "degenerate": "临界（零实部/重根）",
}

FLOW_LABELS: dict[str, str] = {
    "lorenz": "Lorenz（σ=10, ρ=28, β=8/3 的经典混沌）",
    "rossler": "Rössler（a=b=0.2, c=5.7 的折叠带）",
    "van_der_pol": "van der Pol（μ 越大越张弛）",
    "damped_pendulum": "阻尼摆（x 是角、y 是角速度）",
    "hopf_normal_form": "Hopf 正规形（μ>0 出现半径 √μ 的极限环）",
}


# --------------------------------------------------------------------------- #
# 0. 批量形状约定
# --------------------------------------------------------------------------- #
def _batch(y: Any, dim: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """把状态摊成 ``(n, dim)`` 并记住原形状（返回 ``(批量数组, 原形状)``）。"""
    arr = np.asarray(y, dtype=float)
    shape = arr.shape
    flat = arr.reshape(-1, dim)
    return flat, shape


def _unbatch(out: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return out.reshape(shape)


# --------------------------------------------------------------------------- #
# 1. 连续系统：右端与 Jacobi 矩阵
# --------------------------------------------------------------------------- #
def lorenz_rhs(sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> FlowRHS:
    r""":math:`\dot x=\sigma(y-x),\ \dot y=x(\rho-z)-y,\ \dot z=xy-\beta z`（Lorenz 1963）。"""

    def f(t: float, y: Any) -> np.ndarray:  # noqa: ARG001 - 签名统一
        Y, shape = _batch(y, 3)
        x, yy, z = Y[:, 0], Y[:, 1], Y[:, 2]
        out = np.empty_like(Y)
        out[:, 0] = sigma * (yy - x)
        out[:, 1] = x * (rho - z) - yy
        out[:, 2] = x * yy - beta * z
        return _unbatch(out, shape)

    return f


def lorenz_jac(sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> FlowRHS:
    r"""Lorenz 的 Jacobi 矩阵 :math:`Df`（对每个状态点给一个 :math:`3\times3`）。"""

    def j(t: float, y: Any) -> np.ndarray:  # noqa: ARG001
        Y, shape = _batch(y, 3)
        x, yy, z = Y[:, 0], Y[:, 1], Y[:, 2]
        n = Y.shape[0]
        J = np.zeros((n, 3, 3))
        J[:, 0, 0] = -sigma
        J[:, 0, 1] = sigma
        J[:, 1, 0] = rho - z
        J[:, 1, 1] = -1.0
        J[:, 1, 2] = -x
        J[:, 2, 0] = yy
        J[:, 2, 1] = x
        J[:, 2, 2] = -beta
        if len(shape) == 1:
            return J[0]
        return J

    return j


def lorenz_divergence(sigma: float = 10.0, beta: float = 8.0 / 3.0) -> float:
    r"""Lorenz 的散度 :math:`\nabla\cdot f=-(\sigma+1+\beta)`（常数 ⟹ 相体积指数收缩）。

    这条恒等式是本章最硬的一个锚：三个 Lyapunov 指数之和**必须**等于散度的时间平均，
    于是"测出来的指数"有一个不依赖任何参考值的绝对标尺。
    """
    return -(float(sigma) + 1.0 + float(beta))


def lorenz_fixed_points(sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> list[np.ndarray]:
    r"""Lorenz 的平衡点闭式解。

    :math:`\rho\le1` 时只有原点；:math:`\rho>1` 时还有
    :math:`C_\pm=(\pm\sqrt{\beta(\rho-1)},\pm\sqrt{\beta(\rho-1)},\rho-1)`
    （两对称点，也叫"蝴蝶的两个翅根"）。
    """
    pts = [np.zeros(3)]
    if rho > 1.0:
        s = math.sqrt(beta * (rho - 1.0))
        pts.append(np.array([s, s, rho - 1.0]))
        pts.append(np.array([-s, -s, rho - 1.0]))
    return pts


def lorenz_hopf_rho(sigma: float = 10.0, beta: float = 8.0 / 3.0) -> float | None:
    r""":math:`C_\pm` 发生 Hopf 分岔的 :math:`\rho_H=\dfrac{\sigma(\sigma+\beta+3)}{\sigma-\beta-1}`。

    推导见 ``notes.tex``：在 :math:`\rho=\rho_H` 处 :math:`C_\pm` 的 Jacobi 矩阵有一对
    纯虚特征值 :math:`\pm i\omega`，:math:`\omega^2=\dfrac{2\sigma\beta(\rho_H-1)}{\sigma+\beta+1}`
    （数值上 :math:`\sigma=10,\beta=8/3` 给 :math:`\rho_H=470/19\approx24.7368`、:math:`\omega\approx9.6245`）。
    :math:`\sigma\le\beta+1` 时分母非正，此时不存在这对纯虚根，返回 ``None``。
    """
    den = float(sigma) - float(beta) - 1.0
    if den <= 0.0:
        return None
    return float(sigma) * (float(sigma) + float(beta) + 3.0) / den


def lorenz_hopf_omega(sigma: float = 10.0, beta: float = 8.0 / 3.0) -> float | None:
    r"""Hopf 点上的虚部 :math:`\omega=\sqrt{2\sigma\beta(\rho_H-1)/(\sigma+\beta+1)}`。"""
    rho_h = lorenz_hopf_rho(sigma, beta)
    if rho_h is None:
        return None
    val = 2.0 * float(sigma) * float(beta) * (rho_h - 1.0) / (float(sigma) + float(beta) + 1.0)
    return math.sqrt(val) if val > 0.0 else None


def rossler_rhs(a: float = 0.2, b: float = 0.2, c: float = 5.7) -> FlowRHS:
    r""":math:`\dot x=-y-z,\ \dot y=x+ay,\ \dot z=b+z(x-c)`（Rössler 1976）。"""

    def f(t: float, y: Any) -> np.ndarray:  # noqa: ARG001
        Y, shape = _batch(y, 3)
        x, yy, z = Y[:, 0], Y[:, 1], Y[:, 2]
        out = np.empty_like(Y)
        out[:, 0] = -yy - z
        out[:, 1] = x + a * yy
        out[:, 2] = b + z * (x - c)
        return _unbatch(out, shape)

    return f


def rossler_jac(a: float = 0.2, b: float = 0.2, c: float = 5.7) -> FlowRHS:
    r"""Rössler 的 Jacobi 矩阵（第三行含 :math:`z`，是它"折叠"的来源）。"""

    def j(t: float, y: Any) -> np.ndarray:  # noqa: ARG001
        Y, shape = _batch(y, 3)
        x, _yy, z = Y[:, 0], Y[:, 1], Y[:, 2]
        n = Y.shape[0]
        J = np.zeros((n, 3, 3))
        J[:, 0, 1] = -1.0
        J[:, 0, 2] = -1.0
        J[:, 1, 0] = 1.0
        J[:, 1, 1] = a
        J[:, 2, 0] = z
        J[:, 2, 2] = x - c
        if len(shape) == 1:
            return J[0]
        return J

    return j


def van_der_pol_rhs(mu: float = 0.5) -> FlowRHS:
    r"""van der Pol：:math:`\dot x=y,\ \dot y=\mu(1-x^2)y-x`。

    :math:`\mu=0` 时退化为谐振子（原点中心 + 半径 2 的圆是**一族**周期轨）；
    :math:`\mu>0` 时原点不稳定，出现唯一的稳定极限环（Poincaré--Bendixson）。
    """

    def f(t: float, y: Any) -> np.ndarray:  # noqa: ARG001
        Y, shape = _batch(y, 2)
        x, yy = Y[:, 0], Y[:, 1]
        out = np.empty_like(Y)
        out[:, 0] = yy
        out[:, 1] = mu * (1.0 - x * x) * yy - x
        return _unbatch(out, shape)

    return f


def van_der_pol_jac(mu: float = 0.5) -> FlowRHS:
    def j(t: float, y: Any) -> np.ndarray:  # noqa: ARG001
        Y, shape = _batch(y, 2)
        x, yy = Y[:, 0], Y[:, 1]
        n = Y.shape[0]
        J = np.zeros((n, 2, 2))
        J[:, 0, 1] = 1.0
        J[:, 1, 0] = -2.0 * mu * x * yy - 1.0
        J[:, 1, 1] = mu * (1.0 - x * x)
        if len(shape) == 1:
            return J[0]
        return J

    return j


def van_der_pol_fixed_points(mu: float = 0.5) -> list[np.ndarray]:  # noqa: ARG001 - 唯一平衡点
    r"""van der Pol 的唯一平衡点是原点（与 :math:`\mu` 无关）。"""
    return [np.zeros(2)]


def damped_pendulum_rhs(c: float = 0.3) -> FlowRHS:
    r"""阻尼摆：:math:`\dot x=y,\ \dot y=-\sin x-c\,y`（:math:`x` 是角，模 :math:`2\pi`）。"""

    def f(t: float, y: Any) -> np.ndarray:  # noqa: ARG001
        Y, shape = _batch(y, 2)
        out = np.empty_like(Y)
        out[:, 0] = Y[:, 1]
        out[:, 1] = -np.sin(Y[:, 0]) - c * Y[:, 1]
        return _unbatch(out, shape)

    return f


def damped_pendulum_jac(c: float = 0.3) -> FlowRHS:
    def j(t: float, y: Any) -> np.ndarray:  # noqa: ARG001
        Y, shape = _batch(y, 2)
        n = Y.shape[0]
        J = np.zeros((n, 2, 2))
        J[:, 0, 1] = 1.0
        J[:, 1, 0] = -np.cos(Y[:, 0])
        J[:, 1, 1] = -c
        if len(shape) == 1:
            return J[0]
        return J

    return j


def damped_pendulum_fixed_points(c: float = 0.3) -> list[np.ndarray]:  # noqa: ARG001
    r"""摆的平衡点：:math:`(0,0)`（下垂）与 :math:`(\pi,0)`（倒立，鞍点）。"""
    return [np.zeros(2), np.array([math.pi, 0.0])]


def hopf_rhs(mu: float = 0.5, omega: float = 1.0) -> FlowRHS:
    r"""Hopf 正规形（直角坐标）：:math:`\dot x=\mu x-\omega y-x r^2,\ \dot y=\omega x+\mu y-y r^2`。

    极坐标下就是 :math:`\dot r=\mu r-r^3,\ \dot\theta=\omega`：:math:`\mu>0` 时存在
    半径 :math:`\sqrt\mu` 的**稳定**极限环，:math:`\mu<0` 时原点是稳定焦点。
    这条闭式半径是本章极限环断言的解析标准件。
    """

    def f(t: float, y: Any) -> np.ndarray:  # noqa: ARG001
        Y, shape = _batch(y, 2)
        x, yy = Y[:, 0], Y[:, 1]
        r2 = x * x + yy * yy
        out = np.empty_like(Y)
        out[:, 0] = mu * x - omega * yy - x * r2
        out[:, 1] = omega * x + mu * yy - yy * r2
        return _unbatch(out, shape)

    return f


def hopf_jac(mu: float = 0.5, omega: float = 1.0) -> FlowRHS:
    def j(t: float, y: Any) -> np.ndarray:  # noqa: ARG001
        Y, shape = _batch(y, 2)
        x, yy = Y[:, 0], Y[:, 1]
        r2 = x * x + yy * yy
        n = Y.shape[0]
        J = np.zeros((n, 2, 2))
        J[:, 0, 0] = mu - 3.0 * x * x - yy * yy
        J[:, 0, 1] = -omega - 2.0 * x * yy
        J[:, 1, 0] = omega - 2.0 * x * yy
        J[:, 1, 1] = mu - x * x - 3.0 * yy * yy
        if len(shape) == 1:
            return J[0]
        return J

    return j


def hopf_fixed_points(mu: float = 0.5, omega: float = 1.0) -> list[np.ndarray]:  # noqa: ARG001
    return [np.zeros(2)]


def rossler_fixed_points(a: float = 0.2, b: float = 0.2, c: float = 5.7) -> list[np.ndarray]:
    r"""Rössler 的平衡点没有好看的闭式解：用 Newton 解 :math:`f(y)=0`（初值走多个起点）。

    第一条方程给 :math:`y=-z`，第二条给 :math:`x=-ay=az`，第三条变成关于 :math:`z` 的一元方程
    :math:`b+z(az-c)=0`；本函数仍然走通用 Newton（理由：它是本章"数值求根"那条路径的样例，
    与闭式路径对照由断言完成）。
    """
    rhs = rossler_rhs(a, b, c)
    jac = rossler_jac(a, b, c)
    seeds = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]), np.array([-4.0, 4.0, 0.05])]
    out: list[np.ndarray] = []
    for s in seeds:
        root = newton_root(lambda y: rhs(0.0, y), lambda y: jac(0.0, y), s, tol=1e-14, max_iter=80)
        if root is None:
            continue
        if not any(float(np.max(np.abs(root - p))) < 1e-9 for p in out):
            out.append(root)
    return out


# --------------------------------------------------------------------------- #
# 2. 离散映射
# --------------------------------------------------------------------------- #
def logistic(r: float = 4.0) -> MapRHS:
    r"""logistic 映射 :math:`x\mapsto rx(1-x)`（区间映射，:math:`r\in[0,4]`）。"""

    def g(x: Any) -> np.ndarray:
        X = np.asarray(x, dtype=float)
        return r * X * (1.0 - X)

    return g


def logistic_deriv(r: float = 4.0) -> MapRHS:
    r""":math:`g'(x)=r(1-2x)`。"""

    def dg(x: Any) -> np.ndarray:
        X = np.asarray(x, dtype=float)
        return r * (1.0 - 2.0 * X)

    return dg


def henon(a: float = 1.4, b: float = 0.3) -> MapRHS:
    r"""Hénon 映射 :math:`(x,y)\mapsto(1-ax^2+y,\ bx)`（耗散：:math:`|\det J|=|b|`）。"""

    def g(u: Any) -> np.ndarray:
        U, shape = _batch(u, 2)
        x, yy = U[:, 0], U[:, 1]
        out = np.empty_like(U)
        out[:, 0] = 1.0 - a * x * x + yy
        out[:, 1] = b * x
        return _unbatch(out, shape)

    return g


def henon_jac(a: float = 1.4, b: float = 0.3) -> MapRHS:
    def J(u: Any) -> np.ndarray:
        U, shape = _batch(u, 2)
        n = U.shape[0]
        out = np.zeros((n, 2, 2))
        out[:, 0, 0] = -2.0 * a * U[:, 0]
        out[:, 0, 1] = 1.0
        out[:, 1, 0] = b
        if len(shape) == 1:
            return out[0]
        return out

    return J


def henon_fixed_points(a: float = 1.4, b: float = 0.3) -> list[np.ndarray]:
    r"""Hénon 的两个不动点闭式解：:math:`x_\pm=\dfrac{-(1-b)\pm\sqrt{(1-b)^2+4a}}{2a},\ y_\pm=bx_\pm`。"""
    disc = (1.0 - b) ** 2 + 4.0 * a
    if disc < 0.0:
        return []
    sq = math.sqrt(disc)
    out = []
    for s in (1.0, -1.0):
        x = (-(1.0 - b) + s * sq) / (2.0 * a)
        out.append(np.array([x, b * x]))
    return out


def map_callable(name: str, **params: float) -> tuple[MapRHS, MapRHS]:
    """按名字取 ``(映射, Jacobi)``：``logistic`` / ``henon``。"""
    key = str(name).lower()
    if key == "logistic":
        return logistic(float(params.get("r", 4.0))), logistic_deriv(float(params.get("r", 4.0)))
    if key == "henon":
        return henon(float(params.get("a", 1.4)), float(params.get("b", 0.3))), henon_jac(
            float(params.get("a", 1.4)), float(params.get("b", 0.3))
        )
    raise KeyError(f"未知映射 {name!r}（可用：logistic、henon）")


# --------------------------------------------------------------------------- #
# 3. 系统封装
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FlowSystem:
    """一个连续系统 + 它的 Jacobi 矩阵 + 平衡点闭式解 + 显示窗口。"""

    name: str
    dim: int
    rhs: FlowRHS
    jac: FlowRHS
    params: tuple[tuple[str, float], ...]
    window: tuple[tuple[float, float], ...]
    axis_labels: tuple[str, ...]

    def param(self, key: str) -> float:
        for k, v in self.params:
            if k == key:
                return v
        raise KeyError(f"系统 {self.name} 没有参数 {key!r}（有：{[k for k, _ in self.params]}）")

    def f(self, y: Any) -> np.ndarray:
        """右端（时间无关，自动补上 ``t=0``）。"""
        return np.asarray(self.rhs(0.0, np.asarray(y, dtype=float)), dtype=float)

    def jac_at(self, y: Any) -> np.ndarray:
        return np.asarray(self.jac(0.0, np.asarray(y, dtype=float)), dtype=float)

    def label(self) -> str:
        return FLOW_LABELS.get(self.name, self.name)


#: 平面系统（viz 相图用）的显示窗口：`(xlim, ylim)`。
_PLANAR_WINDOWS: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "van_der_pol": ((-3.2, 3.2), (-4.5, 4.5)),
    "damped_pendulum": ((-2.0 * math.pi, 2.0 * math.pi), (-4.0, 4.0)),
    "hopf_normal_form": ((-2.0, 2.0), (-2.0, 2.0)),
}

FLOW_NAMES: tuple[str, ...] = ("lorenz", "rossler", "van_der_pol", "damped_pendulum", "hopf_normal_form")
MAP_NAMES: tuple[str, ...] = ("logistic", "henon")


def flow_names() -> tuple[str, ...]:
    return FLOW_NAMES


def window_of(name: str) -> tuple[tuple[float, float], ...]:
    """平面系统的固定显示窗口（``autoscaled=false`` 的依据：窗口写死在数据里）。"""
    key = str(name).lower()
    if key not in _PLANAR_WINDOWS:
        raise KeyError(f"{name!r} 不是平面系统（可用：{sorted(_PLANAR_WINDOWS)}）")
    xlim, ylim = _PLANAR_WINDOWS[key]
    return (xlim, ylim)


def flow_system(name: str, **params: float) -> FlowSystem:
    """按名字装配系统；未给的参数取经典缺省值。"""
    key = str(name).lower()
    if key == "lorenz":
        sigma = float(params.get("sigma", 10.0))
        rho = float(params.get("rho", 28.0))
        beta = float(params.get("beta", 8.0 / 3.0))
        return FlowSystem(
            name="lorenz",
            dim=3,
            rhs=lorenz_rhs(sigma, rho, beta),
            jac=lorenz_jac(sigma, rho, beta),
            params=(("sigma", sigma), ("rho", rho), ("beta", beta)),
            window=((-22.0, 22.0), (-30.0, 30.0), (0.0, 52.0)),
            axis_labels=("x", "y", "z"),
        )
    if key == "rossler":
        a = float(params.get("a", 0.2))
        b = float(params.get("b", 0.2))
        c = float(params.get("c", 5.7))
        return FlowSystem(
            name="rossler",
            dim=3,
            rhs=rossler_rhs(a, b, c),
            jac=rossler_jac(a, b, c),
            params=(("a", a), ("b", b), ("c", c)),
            window=((-14.0, 14.0), (-14.0, 14.0), (-2.0, 24.0)),
            axis_labels=("x", "y", "z"),
        )
    if key == "van_der_pol":
        mu = float(params.get("mu", 0.5))
        return FlowSystem(
            name="van_der_pol",
            dim=2,
            rhs=van_der_pol_rhs(mu),
            jac=van_der_pol_jac(mu),
            params=(("mu", mu),),
            window=_PLANAR_WINDOWS["van_der_pol"],
            axis_labels=("x", "y"),
        )
    if key == "damped_pendulum":
        c = float(params.get("c", 0.3))
        return FlowSystem(
            name="damped_pendulum",
            dim=2,
            rhs=damped_pendulum_rhs(c),
            jac=damped_pendulum_jac(c),
            params=(("c", c),),
            window=_PLANAR_WINDOWS["damped_pendulum"],
            axis_labels=("x", "y"),
        )
    if key == "hopf_normal_form":
        mu = float(params.get("mu", 0.5))
        omega = float(params.get("omega", 1.0))
        return FlowSystem(
            name="hopf_normal_form",
            dim=2,
            rhs=hopf_rhs(mu, omega),
            jac=hopf_jac(mu, omega),
            params=(("mu", mu), ("omega", omega)),
            window=_PLANAR_WINDOWS["hopf_normal_form"],
            axis_labels=("x", "y"),
        )
    raise KeyError(f"未知系统 {name!r}（可用：{list(FLOW_NAMES)}）")


# --------------------------------------------------------------------------- #
# 4. 特征值与相型
# --------------------------------------------------------------------------- #
def char_poly_coeffs(J: Any) -> np.ndarray:
    r"""特征多项式系数（**降幂**）:math:`\lambda^n+a_1\lambda^{n-1}+\cdots+a_n`。

    直接展开低阶行列式：:math:`a_1=-\operatorname{tr}J`、:math:`a_2` 是全部二阶主子式之和、
    :math:`a_3=-\det J`。:math:`n\le3` 之外抛错（本章用不到，也不假装支持）。
    """
    A = np.asarray(J, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"需要方阵，得到形状 {A.shape}")
    n = A.shape[0]
    if n == 1:
        return np.array([1.0, -A[0, 0]])
    if n == 2:
        return np.array([1.0, -(A[0, 0] + A[1, 1]), A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]])
    if n == 3:
        a1 = -(A[0, 0] + A[1, 1] + A[2, 2])
        a2 = (
            A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
            + A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]
            + A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
        )
        a3 = -float(
            A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
            - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
            + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])
        )
        return np.array([1.0, a1, a2, a3])
    raise ValueError(f"只支持 n ≤ 3 的特征多项式，得到 n={n}")


def poly_roots(coeffs: Any, *, iters: int = 200, tol: float = 1e-14) -> np.ndarray:
    r"""自写 Durand--Kerner（Weierstrass 同时迭代）求低阶多项式全部复根。

    与 3.2 章 :func:`numcore.ode.stability.lmm_root_modulus` 同一算法、独立实现：
    首一化后用半径 :math:`1+\max|c_j|` 的圆上等分布点作初值（**确定性**，不掷骰子），
    迭代 :math:`z_i\leftarrow z_i-p(z_i)/\prod_{j\ne i}(z_i-z_j)`。
    一次/二次走闭式解（少 200 次迭代，也避免重根附近的慢收敛）。

    返回按实部升序（实部相同按虚部升序）排列的复根数组。
    """
    c = np.asarray(coeffs, dtype=complex).reshape(-1)
    c = np.trim_zeros(c, "f")
    if c.size == 0:
        raise ValueError("零多项式没有根")
    deg = c.size - 1
    if deg <= 0:
        return np.zeros(0, dtype=complex)
    c = c / c[0]
    if deg == 1:
        roots = np.array([-c[1]])
    elif deg == 2:
        b, cc = c[1], c[2]
        disc = b * b - 4.0 * cc
        sq = np.sqrt(disc + 0j)
        roots = np.array([(-b + sq) / 2.0, (-b - sq) / 2.0])
    else:
        radius = 1.0 + float(np.max(np.abs(c[1:])))
        ang = 2.0 * math.pi * (np.arange(deg, dtype=float) + 0.5) / deg
        roots = radius * np.exp(1j * (ang + 0.35))
        for _ in range(max(1, int(iters))):
            vals = np.zeros(deg, dtype=complex)
            for i in range(deg):
                acc = c[0]
                for k in range(1, deg + 1):
                    acc = acc * roots[i] + c[k]
                vals[i] = acc
            diff = roots[:, None] - roots[None, :]
            np.fill_diagonal(diff, 1.0)
            prod = np.prod(diff, axis=1)
            step = vals / prod
            roots = roots - step
            if float(np.max(np.abs(step))) <= tol * max(1.0, radius):
                break
    order = np.lexsort((roots.imag, roots.real))
    return roots[order]


def eigenvalues(J: Any, **kw: Any) -> np.ndarray:
    """矩阵特征值：``char_poly_coeffs`` + ``poly_roots``（不碰任何现成求解器）。"""
    return poly_roots(char_poly_coeffs(J), **kw)


def eigenvector_2d(J: Any, lam: complex) -> np.ndarray:
    r"""二维矩阵对应 :math:`\lambda` 的特征向量（实向量，若 :math:`\lambda` 实）。

    零空间公式：:math:`(J-\lambda I)v=0` 的两行各给一个候选
    :math:`(J_{01},\lambda-J_{00})` 与 :math:`(\lambda-J_{11},J_{10})`，取模更大的那个
    （另一行在数值上可能同时接近 0，此时方向不可信）。
    """
    A = np.asarray(J, dtype=float)
    if A.shape != (2, 2):
        raise ValueError(f"只支持 2×2，得到 {A.shape}")
    v1 = np.array([A[0, 1], lam - A[0, 0]], dtype=complex)
    v2 = np.array([lam - A[1, 1], A[1, 0]], dtype=complex)
    v = v1 if float(np.max(np.abs(v1))) >= float(np.max(np.abs(v2))) else v2
    nrm = float(np.sqrt(np.sum(np.abs(v) ** 2)))
    if nrm == 0.0:
        return np.zeros(2)
    v = v / nrm
    # 复特征向量（焦点附近）没有唯一的实方向可画：取实部并在调用处只用于**实**特征值
    # （鞍点的稳定/不稳定方向）。这一点写在 docstring 里，不靠调用方自觉。
    return np.asarray(v.real, dtype=float)


def classify_eigenvalues(ev: Any, *, rel_tol: float = 1e-8) -> str:
    r"""相型判定（只用 :math:`\operatorname{Re}\lambda_i` 与虚部是否为零）。

    判据是教科书口径：全负 + 无虚部 = 稳定结点；全负 + 有虚部 = 稳定焦点；
    有正有负且全为实 = 鞍点；有正有负且含复共轭对 = 鞍焦（``saddle_focus``）；
    实部全为零 = 中心（线性化**不**判定稳定性）；有零有负（但无正）= 临界（``degenerate``，
    线性化同样不判定）。:math:`|\operatorname{Re}\lambda|\le\text{rel\_tol}\cdot\max(1,|\lambda|)`
    视为零——阈值是显式的，因为它决定了"是不是中心"这个定性结论。
    """
    w = np.asarray(ev, dtype=complex).reshape(-1)
    if w.size == 0:
        return "degenerate"
    scale = max(1.0, float(np.max(np.abs(w))))
    tol = float(rel_tol) * scale
    re = w.real
    n_pos = int(np.count_nonzero(re > tol))
    n_neg = int(np.count_nonzero(re < -tol))
    has_imag = bool(np.max(np.abs(w.imag)) > tol)
    if n_pos and n_neg:
        # 混合符号时还要看"螺旋"：只要不稳定的那一侧有复共轭对，就是鞍焦而不是鞍点。
        # （初版把所有混合符号都判成 saddle，于是 Lorenz 在 ρ=28 的 C± 被写成了鞍点。）
        return "saddle_focus" if has_imag else "saddle"
    if n_pos:
        return "unstable_focus" if has_imag else "unstable_node"
    if n_neg:
        if n_neg == w.size:
            return "stable_focus" if has_imag else "stable_node"
        return "degenerate"
    return "center"


@dataclass(frozen=True)
class FixedPointReport:
    """一个平衡点的线性化体检报告。"""

    point: np.ndarray
    jacobian: np.ndarray
    eigenvalues: np.ndarray
    kind: str
    code: int
    residual: float
    stable: bool

    def kind_label(self) -> str:
        return KIND_NAMES.get(self.kind, self.kind)


def fixed_point_report(system: FlowSystem, point: Any) -> FixedPointReport:
    """在 ``point`` 处做线性化：残差、Jacobi 矩阵、特征值、相型。"""
    p = np.asarray(point, dtype=float).reshape(-1)
    J = system.jac_at(p)
    ev = eigenvalues(J)
    kind = classify_eigenvalues(ev)
    res = float(np.max(np.abs(system.f(p)))) if p.size else float("nan")
    return FixedPointReport(
        point=p,
        jacobian=J,
        eigenvalues=ev,
        kind=kind,
        code=int(KIND_CODES.get(kind, 6)),
        residual=res,
        stable=kind in ("stable_node", "stable_focus"),
    )


def newton_root(
    f: Callable[[np.ndarray], np.ndarray],
    jac: Callable[[np.ndarray], np.ndarray],
    x0: Any,
    *,
    tol: float = 1e-12,
    max_iter: int = 60,
    damping: bool = True,
) -> np.ndarray | None:
    r"""Newton 求根（线性系统走 3.2 章的自研 LU）。

    ``tol`` 是**相对**判据 :math:`\|f\|\le\text{tol}\cdot\max(1,\|f_0\|)`；绝对判据在
    :math:`\|f\|` 本身接近 0 的地方会"零次迭代就宣布收敛"（3.2 章踩过的坑）。
    ``damping=True`` 时残差回溯：新的 :math:`\|f\|` 没下降就把步长减半（最多 20 次）。
    不收敛返回 ``None``——这是**要报告的状态**，不是崩溃。
    """
    x = np.asarray(x0, dtype=float).reshape(-1).copy()
    f0 = np.asarray(f(x), dtype=float).reshape(-1)
    scale = max(1.0, float(np.max(np.abs(f0))))
    for _ in range(max(1, int(max_iter))):
        fx = np.asarray(f(x), dtype=float).reshape(-1)
        if float(np.max(np.abs(fx))) <= tol * scale:
            return x
        A = np.asarray(jac(x), dtype=float)
        lu = lu_decompose(A, pivot="partial")
        try:
            dx = solve_lu(lu, -fx)
        except Exception:  # noqa: BLE001 - 奇异 Jacobi 是一种要报告的状态
            return None
        if not np.all(np.isfinite(dx)):
            return None
        step = 1.0
        for _ in range(20 if damping else 1):
            cand = x + step * dx
            fc = np.asarray(f(cand), dtype=float).reshape(-1)
            if not damping or float(np.max(np.abs(fc))) < float(np.max(np.abs(fx))):
                break
            step *= 0.5
        x = x + step * dx
        if float(np.max(np.abs(step * dx))) <= tol * max(1.0, float(np.max(np.abs(x)))):
            fx = np.asarray(f(x), dtype=float).reshape(-1)
            if float(np.max(np.abs(fx))) <= tol * scale:
                return x
    fx = np.asarray(f(x), dtype=float).reshape(-1)
    return x if float(np.max(np.abs(fx))) <= tol * scale else None


# --------------------------------------------------------------------------- #
# 5. 零倾线
# --------------------------------------------------------------------------- #
def zero_curves(
    g: np.ndarray,
    x: Any,
    y: Any,
    *,
    bisect: int = 60,
    gap: float = 0.35,
    min_points: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    r"""标量场 ``g``（形状 ``(nx, ny)``）在网格上的**零曲线**（零倾线的通用追踪器）。

    逐列扫描 :math:`y` 方向的符号变化，每个变化里用二分把 :math:`g=0` 的根压到机器精度；
    再按"与上一条活跃分支的**归一化二维距离**最小"把各列的点连成分支（``gap`` 是允许的最大距离：
    :math:`\sqrt{(\Delta x/W)^2+(\Delta y/H)^2}\le\text{gap}`）。返回若干 ``(xs, ys)`` 分支，
    长度小于 ``min_points`` 的丢掉。

    为什么距离要带 :math:`x` 分量：van der Pol 的 :math:`\dot y=0` 零倾线在 :math:`x=\pm1`
    有竖直渐近线，两条分支会在窗口顶/底被切断；只看 :math:`\Delta y` 会把"从顶部出去的那一支"
    与"从底部进来的那一支"接成一条穿窗而过的假曲线（实测 :math:`\Delta y=3.14`，恰好落在
    只有 :math:`y` 判据的缝隙里）。

    为什么不用等值线/等高线算法：那会把"根"降级成"格点插值"，残差就不是 O(u) 了；
    这里的每个点都是**二分出来的根**，帧里的 ``residual_null`` 才有意义。
    """
    G = np.asarray(g, dtype=float)
    xs = np.asarray(x, dtype=float).reshape(-1)
    ys = np.asarray(y, dtype=float).reshape(-1)
    if G.shape != (xs.size, ys.size):
        raise ValueError(f"g 的形状 {G.shape} 与网格 ({xs.size}, {ys.size}) 不符")
    width = float(xs[-1] - xs[0]) if xs.size > 1 else 1.0
    height = float(ys[-1] - ys[0]) if ys.size > 1 else 1.0
    branches: list[list[tuple[float, float, bool]]] = []
    for i in range(xs.size):
        col = G[i, :]
        found: list[float] = []
        for j in range(ys.size - 1):
            a, b = float(col[j]), float(col[j + 1])
            if a == 0.0:
                found.append(float(ys[j]))
                continue
            if a * b < 0.0:
                # 在**这一个**格点区间内二分；区间内的值用显式线性插值，
                # 不用 np.interp（P3 名单里就有 interp，而这里根本不需要它）。
                ya, yb = float(ys[j]), float(ys[j + 1])
                ga, gb = a, b
                for _ in range(int(bisect)):
                    mid = 0.5 * (ya + yb)
                    gm = ga + (gb - ga) * (mid - ya) / (yb - ya)
                    if ga * gm <= 0.0:
                        yb, gb = mid, gm
                    else:
                        ya, ga = mid, gm
                found.append(0.5 * (ya + yb))
        if not found:
            continue
        used = [False] * len(found)
        for br in branches:
            if br[-1][2]:  # 已经断掉的分支不再接收新点
                continue
            lx, ly, _ = br[-1]
            best, bestd = -1, float("inf")
            for k, val in enumerate(found):
                if used[k]:
                    continue
                d = math.hypot((float(xs[i]) - lx) / max(width, 1e-300), (val - ly) / max(height, 1e-300))
                if d < bestd:
                    best, bestd = k, d
            if best >= 0 and bestd <= gap:
                used[best] = True
                br.append((float(xs[i]), float(found[best]), False))
        for k, val in enumerate(found):
            if not used[k]:
                branches.append([(float(xs[i]), float(val), False)])
        for br in branches:  # 本列没能延续的分支就此封闭
            if br[-1][0] != float(xs[i]):
                br[-1] = (br[-1][0], br[-1][1], True)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for br in branches:
        if len(br) < int(min_points):
            continue
        arr = np.asarray([(p[0], p[1]) for p in br], dtype=float)
        out.append((arr[:, 0], arr[:, 1]))
    return out


def _field_on_grid(system: FlowSystem, x: Any, y: Any) -> np.ndarray:
    """把平面系统在网格上求值：返回 ``(nx, ny, 2)``。"""
    xs = np.asarray(x, dtype=float).reshape(-1)
    ys = np.asarray(y, dtype=float).reshape(-1)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)
    F = np.asarray(system.rhs(0.0, pts), dtype=float).reshape(xs.size, ys.size, 2)
    return F


def vector_field(system: FlowSystem, x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """平面系统在网格上的向量场：返回 ``(U, V)``，形状都是 ``(nx, ny)``。

    与 :func:`nullcline_branches` 共用 :func:`_field_on_grid`，所以相图上的箭头与零倾线
    **必然**来自同一套求值——不会出现"箭头往右、零倾线偏说往左"这种自相矛盾的图。
    """
    F = _field_on_grid(system, x, y)
    return F[:, :, 0], F[:, :, 1]


def nullcline_branches(
    system: FlowSystem,
    *,
    nx: int = 161,
    ny: int = 161,
    gap: float = 0.35,
    min_points: int = 4,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """平面系统两条零倾线的**数值**分支：``{"x": [...], "y": [...]}``。

    ``"x"`` 是 :math:`\\dot x=0` 的分支，``"y"`` 是 :math:`\\dot y=0` 的分支。
    """
    if system.dim != 2:
        raise ValueError(f"{system.name} 不是平面系统，画不了相图")
    (xlo, xhi), (ylo, yhi) = system.window[0], system.window[1]
    xs = np.linspace(xlo, xhi, int(nx))
    ys = np.linspace(ylo, yhi, int(ny))
    F = _field_on_grid(system, xs, ys)
    return {
        "x": zero_curves(F[:, :, 0], xs, ys, gap=gap, min_points=min_points),
        "y": zero_curves(F[:, :, 1], xs, ys, gap=gap, min_points=min_points),
    }


def analytic_nullclines(system: FlowSystem, *, n: int = 241) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    r"""两条零倾线的**闭式**分支（三个平面系统各有各的解析式）。

    * van der Pol：:math:`y=0` 与 :math:`y=\dfrac{x}{\mu(1-x^2)}`（:math:`x=\pm1` 是两条竖直渐近线，
      窗口内因此有三段分支）；
    * 阻尼摆：:math:`y=0` 与 :math:`y=-\sin x/c`（整段一条）；
    * Hopf 正规形：用极坐标参数化，:math:`r\ne0` 时
      :math:`\tan\theta=(\mu-r^2)/\omega`（:math:`\dot x=0`）与
      :math:`\tan\theta=\omega/(r^2-\mu)`（:math:`\dot y=0`），再加上原点本身。
    """
    if system.dim != 2:
        raise ValueError(f"{system.name} 不是平面系统")
    (xlo, xhi), (ylo, yhi) = system.window[0], system.window[1]
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {"x": [], "y": []}
    name = system.name
    if name == "van_der_pol":
        mu = system.param("mu")
        xs = np.linspace(xlo, xhi, int(n))
        out["x"].append((xs, np.zeros_like(xs)))
        if abs(mu) > 0.0:
            eps = 1e-6
            for lo, hi in ((xlo, -1.0 - eps), (-1.0 + eps, 1.0 - eps), (1.0 + eps, xhi)):
                if hi <= lo:
                    continue
                seg = np.linspace(lo, hi, max(3, int(n * (hi - lo) / (xhi - xlo))))
                ys = seg / (mu * (1.0 - seg * seg))
                keep = (ys >= ylo) & (ys <= yhi)
                if int(np.count_nonzero(keep)) >= 2:
                    out["y"].append((seg[keep], ys[keep]))
        else:  # μ=0：ẏ=−x ⟹ x=0 整条竖线
            ys = np.linspace(ylo, yhi, int(n))
            out["y"].append((np.zeros_like(ys), ys))
    elif name == "damped_pendulum":
        c = system.param("c")
        xs = np.linspace(xlo, xhi, int(n))
        out["x"].append((xs, np.zeros_like(xs)))
        if abs(c) > 0.0:
            ys = -np.sin(xs) / c
            keep = (ys >= ylo) & (ys <= yhi)
            out["y"].append((xs[keep], ys[keep]))
        else:
            ys = np.linspace(ylo, yhi, int(n))
            for k in range(-2, 3):
                out["y"].append((k * math.pi * np.ones_like(ys), ys))
    elif name == "hopf_normal_form":
        mu = system.param("mu")
        omega = system.param("omega")
        rmax = math.hypot(xhi, yhi)
        rr = np.linspace(1e-9, rmax, int(n))
        for key, num, den in (("x", mu - rr * rr, omega), ("y", omega, rr * rr - mu)):
            th = np.arctan2(num, den)
            xs = rr * np.cos(th)
            ys = rr * np.sin(th)
            keep = (xs >= xlo) & (xs <= xhi) & (ys >= ylo) & (ys <= yhi)
            if int(np.count_nonzero(keep)) >= 2:
                out[key].append((xs[keep], ys[keep]))
        out["x"].append((np.zeros(1), np.zeros(1)))
        out["y"].append((np.zeros(1), np.zeros(1)))
    else:
        raise KeyError(f"{name!r} 没有闭式零倾线")
    return out


# --------------------------------------------------------------------------- #
# 6. 轨迹积分（复用 3.1 章的 RK）
# --------------------------------------------------------------------------- #
def trajectory(
    system: FlowSystem,
    y0: Any,
    t_span: tuple[float, float],
    h: float,
    *,
    method: str = "rk4",
    max_steps: int = 400000,
) -> rkmod.Trajectory:
    """单条轨迹：直接转发 3.1 章的 :func:`numcore.ode.rk.integrate_fixed`。

    复用而不是重写：爆炸停止、末步截断、求值计数这三条"不撒谎的约定"都在那里实现过一遍
    （本章的帧只消费它的 ``ok`` / ``blowup`` / ``t_blowup`` / ``n_f_evals``）。
    """
    return rkmod.integrate_fixed(
        method,
        system.rhs,
        t_span,
        np.asarray(y0, dtype=float).reshape(-1),
        h,
        max_steps=max_steps,
        label=system.name,
    )


@dataclass(frozen=True)
class EnsembleRun:
    """一批初值同时推进的结果：``Y`` 形状 ``(n_t, n_traj, dim)``。"""

    t: np.ndarray
    Y: np.ndarray
    h: float
    n_steps: int
    blowup: bool
    t_blowup: float
    n_f_evals: int

    @property
    def n_traj(self) -> int:
        return int(self.Y.shape[1])

    def final(self) -> np.ndarray:
        return self.Y[-1]

    def spread(self) -> np.ndarray:
        """每时刻的**最大两两距离**（相体积扩张的直接读数）。"""
        n_t = self.Y.shape[0]
        out = np.zeros(n_t)
        for k in range(n_t):
            Z = self.Y[k]
            d = np.sqrt(np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2))
            out[k] = float(np.max(d))
        return out


def integrate_ensemble(
    system: FlowSystem,
    Y0: Any,
    t_span: tuple[float, float],
    h: float,
    *,
    method: str = "rk4",
    stride: int = 1,
    max_steps: int = 400000,
) -> EnsembleRun:
    r"""一批初值用**同一个** :func:`numcore.ode.rk.rk_step` 推进（批量形状约定见 §0）。

    为什么批量：``rk_step`` 把状态摊平后调用右端，右端按 ``(n, dim)`` 一次算完全部轨道，
    于是 Python 层的每步开销被摊到 :math:`n` 条轨道上——8 对孪生轨迹（16 条）跑 30 个时间单位
    的实际代价与 3 条轨道同量级（实测 h=0.01 时 0.10 s vs 0.06 s）。步长、方法、爆炸判据
    与 :func:`trajectory` 完全一致，两条路径在断言里对拍。
    """
    t0, t1 = float(t_span[0]), float(t_span[1])
    if not t1 > t0:
        raise ValueError(f"需要 t1 > t0，得到 {t_span}")
    if not h > 0.0:
        raise ValueError(f"步长必须为正，得到 {h}")
    Y = np.asarray(Y0, dtype=float)
    if Y.ndim != 2 or Y.shape[1] != system.dim:
        raise ValueError(f"Y0 形状应为 (n, {system.dim})，得到 {Y.shape}")
    state = Y.reshape(-1).copy()
    ts = [t0]
    rows = [Y.copy()]
    n_f = 0
    t = t0
    n_steps = 0
    blowup = False
    t_blow = float("nan")
    while t < t1 - 1e-15 and n_steps <= int(max_steps):
        hh = min(float(h), t1 - t)
        step = rkmod.rk_step(method, system.rhs, t, state, hh)
        n_f += step.n_f_evals
        state = step.y
        t += hh
        n_steps += 1
        if n_steps % max(1, int(stride)) == 0 or t >= t1 - 1e-15:
            ts.append(t)
            rows.append(state.reshape(Y.shape).copy())
        if not np.all(np.isfinite(state)) or float(np.max(np.abs(state))) > BLOWUP:
            blowup = True
            t_blow = t
            break
    return EnsembleRun(
        t=np.asarray(ts, dtype=float),
        Y=np.asarray(rows, dtype=float),
        h=float(h),
        n_steps=int(n_steps),
        blowup=blowup,
        t_blowup=t_blow,
        n_f_evals=int(n_f),
    )


# --------------------------------------------------------------------------- #
# 7. Poincaré 截面
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SectionResult:
    """一次 Poincaré 截面采样的全部可观测物。

    ``points`` 是精化后的截面点，``lin_points`` 是纯线性插值的点，
    ``plane_residual[j]`` 是精化点在平面函数上的残差 :math:`|h(y_j)|`，
    ``refine_shift[j]`` 是精化相对线性插值的位移（这**就是**线性插值的截面误差）。
    """

    t: np.ndarray
    points: np.ndarray
    lin_points: np.ndarray
    plane_residual: np.ndarray
    refine_shift: np.ndarray
    orbit_t: np.ndarray
    orbit_y: np.ndarray
    plane_index: int
    level: float
    direction: int
    n_steps: int
    n_f_evals: int
    blowup: bool
    t_blowup: float

    @property
    def n_cross(self) -> int:
        return int(self.t.size)

    def mean_return_time(self) -> float:
        if self.t.size < 2:
            return float("nan")
        return float(np.mean(np.diff(self.t)))

    def return_map(self, coord: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """首次返回映射的散点：``(u_n, u_{n+1})``，``coord`` 缺省用平面上第一个非截面坐标。"""
        idx = self.plane_index
        k = (idx + 1) % self.points.shape[1] if coord is None else int(coord)
        vals = self.points[:, k]
        return vals[:-1], vals[1:]


def poincare_crossings(
    system: FlowSystem,
    y0: Any,
    *,
    plane_index: int,
    level: float,
    direction: int = 1,
    t_span: tuple[float, float] = (0.0, 100.0),
    h: float = 0.005,
    method: str = "rk4",
    max_crossings: int = 200,
    refine: int = 3,
    max_steps: int = 400000,
) -> SectionResult:
    r"""轨道穿过超平面 :math:`y_{\text{plane}}=\text{level}` 的点集与残差。

    判定用的是**方向敏感的符号变化**：``direction=+1`` 只收 :math:`h` 由负变正的那次穿越，
    这样返回映射才是单值的（双向都收会把两次穿越混在一张散点图里，形成"重影"）。

    定位分两步，两步都留着，因为它们的差就是误差：

    1. **线性插值**：:math:`\theta=-h_n/(h_{n+1}-h_n)`，给 ``lin_points``（:math:`O(h^2)`）；
    2. **Newton 精化**：从 :math:`y_n` 出发、按 :math:`\theta h` 再走一小步（同方法同阶），
       用**局部导数** :math:`\dot h=f(y)[\text{plane}]` 做 Newton 更新
       :math:`\theta\leftarrow\theta-h(y)/(\dot h\,h)`。局部导数是必须的：改用整步平均斜率当
       割线时实测残差只到 :math:`3\times10^{-3}`（穿越点附近曲率大，收敛比只有约 1/3.5），
       换成局部导数后当场掉到舍入水平。

    这样"截面点准不准"不需要稠密输出，也不需要信任插值阶——它有一个当场算出来的残差
    （``plane_residual``），而线性插值本身的位移由 ``refine_shift`` 如实给出。
    """
    if not 0 <= int(plane_index) < system.dim:
        raise ValueError(f"平面下标越界：{plane_index}（系统 {system.name} 是 {system.dim} 维）")
    if direction not in (-1, 1):
        raise ValueError("direction 只能是 +1 或 -1")
    idx = int(plane_index)
    y = np.asarray(y0, dtype=float).reshape(-1).copy()
    t = float(t_span[0])
    t1 = float(t_span[1])
    ts: list[float] = []
    pts: list[np.ndarray] = []
    lins: list[np.ndarray] = []
    resid: list[float] = []
    shift: list[float] = []
    ot: list[float] = []
    oy: list[np.ndarray] = []
    n_f = 0
    n_steps = 0
    blowup = False
    t_blow = float("nan")
    g_prev = float(y[idx] - level)
    n_ref = max(0, int(refine))
    while t < t1 - 1e-15 and n_steps <= int(max_steps):
        hh = min(float(h), t1 - t)
        step = rkmod.rk_step(method, system.rhs, t, y, hh)
        n_f += step.n_f_evals
        y_next = step.y
        g_next = float(y_next[idx] - level)
        ot.append(t + hh)
        oy.append(y_next.copy())
        if (g_prev < 0.0 <= g_next and direction > 0) or (g_prev > 0.0 >= g_next and direction < 0):
            theta = -g_prev / (g_next - g_prev) if g_next != g_prev else 0.5
            lin = y + theta * (y_next - y)
            yy = y.copy()
            for _ in range(n_ref):
                sub = rkmod.rk_step(method, system.rhs, t, y, max(theta * hh, 1e-15))
                n_f += sub.n_f_evals
                yy = sub.y
                gy = float(yy[idx] - level)
                dg = float(np.asarray(system.rhs(t + theta * hh, yy), dtype=float).reshape(-1)[idx]) * hh
                n_f += 1
                if dg == 0.0 or not math.isfinite(gy) or not math.isfinite(dg):
                    break
                theta = min(max(theta - gy / dg, 0.0), 1.0)
            if n_ref == 0:
                sub = rkmod.rk_step(method, system.rhs, t, y, max(theta * hh, 1e-15))
                n_f += sub.n_f_evals
                yy = sub.y
            gy = float(yy[idx] - level)
            ts.append(t + theta * hh)
            pts.append(yy.copy())
            lins.append(lin.copy())
            resid.append(abs(gy))
            shift.append(float(np.sqrt(np.sum((yy - lin) ** 2))))
            if len(ts) >= int(max_crossings):
                g_prev = g_next
                y = y_next
                t += hh
                n_steps += 1
                break
        g_prev = g_next
        y = y_next
        t += hh
        n_steps += 1
        if not np.all(np.isfinite(y)) or float(np.max(np.abs(y))) > BLOWUP:
            blowup = True
            t_blow = t
            break
    return SectionResult(
        t=np.asarray(ts, dtype=float),
        points=np.asarray(pts, dtype=float).reshape(-1, system.dim),
        lin_points=np.asarray(lins, dtype=float).reshape(-1, system.dim),
        plane_residual=np.asarray(resid, dtype=float),
        refine_shift=np.asarray(shift, dtype=float),
        orbit_t=np.asarray(ot, dtype=float),
        orbit_y=np.asarray(oy, dtype=float).reshape(-1, system.dim),
        plane_index=idx,
        level=float(level),
        direction=int(direction),
        n_steps=int(n_steps),
        n_f_evals=int(n_f),
        blowup=bool(blowup),
        t_blowup=float(t_blow),
    )


# --------------------------------------------------------------------------- #
# 8. 三维显示投影
# --------------------------------------------------------------------------- #
def projection_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    r"""正交投影矩阵 :math:`2\times3`：:math:`(u,v)=P(x,y,z)`（见模块头）。

    方位角绕 :math:`z` 轴、仰角再抬头；**没有透视除法**，所以远近不改变尺寸，
    深度在图上不可读（这一点写进帧的 ``strings.projection_note``）。
    """
    az = math.radians(float(azimuth_deg))
    el = math.radians(float(elevation_deg))
    ca, sa = math.cos(az), math.sin(az)
    ce, se = math.cos(el), math.sin(el)
    return np.array([[-sa, ca, 0.0], [-ca * se, -sa * se, ce]], dtype=float)


def project3d(X: Any, azimuth_deg: float = -60.0, elevation_deg: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """把 ``(..., 3)`` 的坐标投影成屏幕坐标 ``(u, v)``（形状与输入的前导维一致）。"""
    P = projection_matrix(azimuth_deg, elevation_deg)
    A = np.asarray(X, dtype=float)
    if A.shape[-1] != 3:
        raise ValueError(f"最后一维必须是 3，得到 {A.shape}")
    flat = A.reshape(-1, 3)
    uv = flat @ P.T
    u = uv[:, 0].reshape(A.shape[:-1])
    v = uv[:, 1].reshape(A.shape[:-1])
    return u, v
