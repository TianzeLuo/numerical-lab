r"""5.1 章（无约束优化与拟牛顿）的数值内核。

数学对象
--------
无约束问题 :math:`\min_{x\in\mathbb{R}^n} f(x)`，迭代格式一律写成

.. math::
    x_{k+1} = x_k + \alpha_k p_k,\qquad p_k = -B_k^{-1}\nabla f(x_k)
    \quad\text{或}\quad p_k = -\nabla f(x_k) + \beta_k p_{k-1},

其中 :math:`B_k` 是 Hessian 或其拟牛顿近似（BFGS / L-BFGS），:math:`\alpha_k` 由线搜索或
信赖域给出。所有算法返回 :class:`Trace`——**同一种轨迹容器**，于是"迭代数 vs 条件数"、
"割线方程残差逐步是多少"、"二次收敛从哪里开始"这三类问题可以用同一套代码量。

本模块刻意不做的事
------------------
* **不提供"收敛"这个结论**。:class:`Trace` 只记录停止原因（``stop``）与三套判据各自的
  数值（梯度范数 / 步长 / 函数值下降），因为这三者在病态问题上会给出**不同**的答案
  （见 `chapters/5.1/pitfalls.md`）。是否收敛由调用方判定。
* **不把"随机扰动跑掉了"当作方法有效**。鞍点逃逸要给出负曲率方向、负特征值的大小、
  以及**沿该方向**逃逸所需的最小扰动幅度（:func:`escape_threshold` 二分实测），
  并与随机方向的阈值并列——两者的比值就是 :math:`1/|\langle u_{\min},d\rangle|`。

依赖方向
--------
``numcore.inverse`` → ``numcore.linalg`` / ``numcore.utils``（单向）。线性系统一律走
:mod:`numcore.linalg.lu`（一般矩阵）或 :mod:`numcore.linalg.chol`（正定矩阵），
特征分解走 :mod:`numcore.linalg.eigen.hermitian_jacobi_eigh`：P2/P3 在 ``numcore/`` 下
禁止 ``numpy.linalg`` 与现成求解器属性。

出处
----
Nocedal & Wright, *Numerical Optimization*, 2nd ed.（线搜索 §3、信赖域 §4、
拟牛顿 §6、非线性 CG §5）；Golub & Van Loan, *Matrix Computations*, 4th ed.（Cholesky / LU）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Sequence

import numpy as np

from numcore.linalg.chol import chol_solve, cholesky
from numcore.linalg.eigen import hermitian_jacobi_eigh
from numcore.linalg.lu import lu_partial, solve_lu
from numcore.utils import rng

__all__ = [
    # 容器
    "Quadratic",
    "Objective",
    "Trace",
    "LineSearchResult",
    "SteihaugResult",
    "EscapeMeasurement",
    # 问题构造
    "quadratic_from_spectrum",
    "anisotropic_quadratic",
    "saddle_quadratic",
    "rosenbrock_objective",
    "quartic_quadratic",
    "quadratic_objective",
    # 原语与诊断
    "inner",
    "euclidean_norm",
    "divergence_limit",
    "solve_symmetric",
    "hessian_step",
    "secant_residual",
    "quadratic_window",
    "convergence_order",
    "contour_grid",
    "min_eigenpair",
    "negative_curvature_direction",
    "escape_threshold",
    "orthonormal_matrix",
    # 线搜索与子问题
    "armijo_backtracking",
    "wolfe_line_search",
    "exact_step_quadratic",
    "steihaug_cg",
    # 收敛率（理论侧，供断言对拍）
    "gd_rate",
    "gd_iterations",
    "gd_optimal_step",
    "momentum_parameters",
    "momentum_rate",
    "momentum_iterations",
    # 算法
    "gd",
    "momentum",
    "newton",
    "bfgs",
    "lbfgs",
    "cg",
    "trust_region",
]

def _eye(n: int) -> np.ndarray:
    """单位阵。

    刻意**不**做模块级缓存：P6 要的是"没有全局可变状态"，一个每维度的缓存字典正是
    那种"今天只读、明天被人写一行就变味"的东西；``n`` 在本章是 2 到 32，:func:`numpy.eye`
    的代价可以忽略。
    """
    return np.eye(int(n), dtype=float)


# --------------------------------------------------------------------------- #
# 原语（P3：范数 / 求解都自研或走 numcore.linalg）
# --------------------------------------------------------------------------- #
def inner(a: Any, b: Any) -> float:
    r"""欧氏内积 :math:`\langle a,b\rangle = a^{\mathsf T}b`。"""
    return float(np.dot(np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()))


def euclidean_norm(v: Any) -> float:
    r""":math:`\|v\|_2`（自研：P3 不许用 ``numpy.linalg.norm``）。"""
    a = np.asarray(v, dtype=float).ravel()
    if a.size == 0:
        return 0.0
    s = float(np.dot(a, a))
    if not np.isfinite(s):
        return float("inf") if s > 0 else float("nan")
    return float(np.sqrt(s))


def _lu_solve_or_none(A: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """部分主元 LU 解 :math:`Ax=b`；奇异或非有限时返回 ``None``（不抛、不冒充）。"""
    res = lu_partial(A)
    if res.singular:
        return None
    try:
        x = solve_lu(res, b)
    except (ZeroDivisionError, ValueError):
        return None
    return x if bool(np.all(np.isfinite(x))) else None


def solve_symmetric(A: Any, b: Any) -> tuple[np.ndarray, bool]:
    """解对称系统 :math:`Ax=b`，返回 ``(x, 用了 Cholesky)``。

    正定就走 :mod:`numcore.linalg.chol`（代价减半且不会遇到零主元），否则退回部分主元 LU。
    ``used_cholesky=False`` 本身就是"这一步的 Hessian 不是正定的"的证据。
    """
    M = np.asarray(A, dtype=float)
    rhs = np.asarray(b, dtype=float).ravel()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        cres = cholesky(M)
        if cres.ok:
            x = chol_solve(cres, rhs)
            if bool(np.all(np.isfinite(x))):
                return x, True
        x = _lu_solve_or_none(M, rhs)
    if x is None:
        raise ZeroDivisionError("对称系统奇异（Cholesky 与 LU 都没有有限解）")
    return x, False


def hessian_step(
    H: Any,
    g: Any,
    *,
    shift0: float = 0.0,
    max_shift: int = 40,
) -> tuple[np.ndarray, bool, float]:
    r"""牛顿步 :math:`p=-H^{-1}g`（带 Levenberg 位移兜底），返回 ``(p, 正定, 位移量)``。

    三种情形分得很清楚，因为它们的**含义完全不同**：

    * Cholesky 成功 → :math:`H\succ0`，返回 ``(p, True, shift)``；
    * Cholesky 失败但 LU 有解 → :math:`H` 不定（例如鞍点），返回 ``(p, False, shift)``
      ——这正是"牛顿法不检查曲率符号"的机器证据；
    * 两者都失败（奇异）→ 加 :math:`\tau I` 重试，``shift`` 逐次翻倍。
    """
    Hm = np.asarray(H, dtype=float)
    gv = np.asarray(g, dtype=float).ravel()
    n = Hm.shape[0]
    shift = float(shift0)
    for _ in range(int(max_shift) + 1):
        M = Hm if shift == 0.0 else Hm + shift * _eye(n)
        try:
            p, pd = solve_symmetric(M, -gv)
        except ZeroDivisionError:
            pd, p = False, None
        if p is not None:
            return p, bool(pd), shift
        shift = 1e-8 if shift == 0.0 else 2.0 * shift
    raise ValueError(f"Hessian 在 {max_shift} 次位移后仍不可解（shift={shift:g}）")


def secant_residual(B: Any, s: Any, y: Any) -> float:
    r"""割线方程残差 :math:`\|Bs-y\|_2/\|y\|_2`（BFGS 的"合格"判据之一）。

    ``y = 0``（梯度没变）时分母没有意义——返回 ``inf`` 而不是编一个数出来，
    调用方负责把它写成帧里的 ``None``（PROTOCOLS §1.3）。
    """
    Bm = np.asarray(B, dtype=float)
    sv = np.asarray(s, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    den = euclidean_norm(yv)
    if den == 0.0:
        return float("inf")
    return euclidean_norm(Bm @ sv - yv) / den


def orthonormal_matrix(n: int, *, seed: int = 20260501) -> np.ndarray:
    r"""随机正交矩阵（对 ``rng`` 高斯样本做**两遍**改进 Gram-Schmidt）。

    为什么要两遍：经典 Gram-Schmidt 在相近列上会丢掉正交性，两遍之后
    :math:`\|Q^{\mathsf T}Q-I\|` 才落回 :math:`O(u)`（B 类断言钉住这一点）。
    """
    g = rng(seed)
    Q = np.asarray(g.standard_normal((int(n), int(n))), dtype=float)
    for _ in range(2):
        for j in range(Q.shape[1]):
            v = Q[:, j].copy()
            for i in range(j):
                v = v - float(np.dot(Q[:, i], v)) * Q[:, i]
            nrm = euclidean_norm(v)
            if nrm == 0.0:
                raise ValueError("Gram-Schmidt 退化：随机列线性相关（不应发生）")
            Q[:, j] = v / nrm
    return Q


def min_eigenpair(H: Any) -> tuple[float, np.ndarray]:
    r"""对称矩阵的最小特征对 :math:`(\lambda_{\min}, u_{\min})`（升序 Jacobi，取第 0 列）。

    非对称输入会先对称化，然后由 :func:`numpy.allclose` 级别的检查确认——这里直接
    对称化并在 docstring 里说明，不做静默的"取上三角"。
    """
    M = np.asarray(H, dtype=float)
    M = 0.5 * (M + M.T)
    w, V = hermitian_jacobi_eigh(M)
    w = np.real(np.asarray(w, dtype=complex)).ravel()
    V = np.real(np.asarray(V, dtype=complex))
    return float(w[0]), np.asarray(V[:, 0], dtype=float)


def negative_curvature_direction(H: Any) -> np.ndarray:
    r"""负曲率方向 :math:`u_{\min}`（最小特征值的特征向量），单位长度。

    鞍点上它才是"真正能下降"的方向；随机方向与它的重合度决定了扰动要多大幅度才够。
    """
    _, u = min_eigenpair(H)
    nrm = euclidean_norm(u)
    if nrm == 0.0:
        raise ValueError("最小特征向量为零（不应发生）")
    return u / nrm


# --------------------------------------------------------------------------- #
# 问题容器
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Quadratic:
    r"""二次目标 :math:`f(x)=\tfrac12x^{\mathsf T}Ax-b^{\mathsf T}x`（:math:`A` 对称）。

    字段里的 ``eigenvalues`` / ``eigenvectors`` 是**实测**的（Jacobi 迭代），
    不是把构造时想要的谱抄回来——这是本章"κ 与迭代数的关系"能当证据的前提。
    ``A`` 不定时 :math:`x_*=A^{-1}b` 仍是驻点，但那是**鞍点**，:meth:`has_minimum` 为假，
    且 :math:`f(x)-f(x_*)` 可以取负值（画图时必须给地板，见 frames 侧）。
    """

    A: np.ndarray
    b: np.ndarray
    x_star: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    f_star: float

    is_quadratic: ClassVar[bool] = True

    # -- 目标函数接口 ---------------------------------------------------- #
    def value(self, x: Any) -> float:
        xv = np.asarray(x, dtype=float).ravel()
        return float(0.5 * float(np.dot(xv, self.A @ xv)) - float(np.dot(self.b, xv)))

    def grad(self, x: Any) -> np.ndarray:
        return self.A @ np.asarray(x, dtype=float).ravel() - self.b

    def hess(self, x: Any) -> np.ndarray:  # noqa: ARG002 - 二次型与 x 无关
        return self.A

    # -- 谱与条件数 ------------------------------------------------------ #
    @property
    def mu(self) -> float:
        r"""强凸常数 :math:`\\mu=\\lambda_{\\min}`（可正可负）。"""
        return float(self.eigenvalues[0])

    @property
    def L(self) -> float:
        r"""光滑常数 :math:`L=\\lambda_{\\max}`。"""
        return float(self.eigenvalues[-1])

    @property
    def kappa(self) -> float:
        r"""条件数 :math:`\\kappa=\\lambda_{\\max}/\\lambda_{\\min}`（:math:`\\mu\\le0` 时返回 ``inf``）。"""
        if self.mu <= 0.0:
            return float("inf")
        return self.L / self.mu

    def has_minimum(self) -> bool:
        return bool(self.mu > 0.0)

    def kappa_of(self, positive: bool = True) -> float:
        r"""只取**正**特征值算条件数（鞍点上用它描述"稳定子空间有多扁"）。"""
        w = self.eigenvalues[self.eigenvalues > 0.0] if positive else np.abs(self.eigenvalues)
        if w.size < 2:
            return 1.0
        return float(w[-1] / w[0])

    # -- 与最优解相关的量 ------------------------------------------------ #
    def dist(self, x: Any) -> float:
        r"""到驻点的距离 :math:`\|x-x_*\|_2`（这是本章画误差用的量）。"""
        return euclidean_norm(np.asarray(x, dtype=float).ravel() - self.x_star)

    def ferr(self, x: Any) -> float:
        r""":math:`f(x)-f(x_*)`（鞍点上可为负；取对数前必须给地板）。"""
        return self.value(x) - self.f_star

    # -- 梯度下降的理论量 ------------------------------------------------ #
    def optimal_step(self) -> float:
        r"""二次型上的最优定步长 :math:`\alpha^\star=2/(L+\mu)`。"""
        return 2.0 / (self.L + self.mu)

    def spectral_radius(self, step: float) -> float:
        r"""定步长 GD 的误差传播谱半径 :math:`\max_i|1-\alpha\lambda_i|`。"""
        return float(np.max(np.abs(1.0 - float(step) * self.eigenvalues)))

    def relative_error(self, x: Any) -> float:
        r""":math:`\|x-x_*\|_2/\|x_*-x_0\|` 需要初值，这里只给绝对距离的别名。"""
        return self.dist(x)


@dataclass(frozen=True)
class Objective:
    """一般目标函数 :math:`f`：值、梯度、可选 Hessian，以及可选的解析最优解。

    ``is_quadratic`` 必须如实标注：``line_search="exact"`` 与"有限步终止"这类结论
    只对二次型成立，拿它去套 Rosenbrock 会得到没有意义的数。
    """

    value: Callable[[np.ndarray], float]
    grad: Callable[[np.ndarray], np.ndarray]
    hess: Callable[[np.ndarray], np.ndarray] | None = None
    x_star: np.ndarray | None = None
    f_star: float | None = None
    lipschitz: float | None = None
    strong_convex: float | None = None
    #: 如实标注"这是不是二次型"：``line_search="exact"`` 与有限步终止只对二次型成立。
    #: 这里用**普通字段**（而不是 :class:`Quadratic` 上的 ``ClassVar``），
    #: 因为一般目标函数需要在运行期决定这件事，而二次型的答案永远是 True。
    is_quadratic: bool = False

    def dist(self, x: Any) -> float:
        if self.x_star is None:
            return float("nan")
        return euclidean_norm(np.asarray(x, dtype=float).ravel() - np.asarray(self.x_star, dtype=float).ravel())

    def ferr(self, x: Any) -> float:
        if self.f_star is None:
            return float("nan")
        return float(self.value(np.asarray(x, dtype=float).ravel())) - float(self.f_star)


def quadratic_objective(Q: Quadratic) -> Objective:
    """把 :class:`Quadratic` 包成 :class:`Objective`（保留谱信息在 ``Q`` 上）。"""
    return Objective(
        value=Q.value,
        grad=Q.grad,
        hess=Q.hess,
        x_star=Q.x_star,
        f_star=Q.f_star,
        lipschitz=Q.L,
        strong_convex=Q.mu,
        is_quadratic=True,
    )


# --------------------------------------------------------------------------- #
# 轨迹容器
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Trace:
    r"""一次迭代的完整记录：:math:`x_k`、:math:`f_k`、:math:`\|\nabla f_k\|`、
    :math:`\|x_{k+1}-x_k\|`、步长 :math:`\alpha_k`、到驻点的距离、以及方法专属诊断。

    约定：``xs``/``f``/``gnorm``/``dist``/``ferr`` 长度是 ``n_iter+1``（含初值），
    ``step``/``alphas``/``ratio``/``accepted`` 长度是 ``n_iter``。
    奇异值用**空数组**表示"这个方法没有这个量"，不用 0 冒充。
    """

    xs: np.ndarray
    f: np.ndarray
    gnorm: np.ndarray
    step: np.ndarray
    alphas: np.ndarray
    dist: np.ndarray
    ferr: np.ndarray
    n_iter: int
    stop: str
    secant_residual: np.ndarray = field(default_factory=lambda: np.zeros(0))
    radius: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ratio: np.ndarray = field(default_factory=lambda: np.zeros(0))
    accepted: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    extras: tuple[tuple[str, np.ndarray], ...] = ()

    @property
    def x_final(self) -> np.ndarray:
        return self.xs[-1]

    def extra(self, name: str) -> np.ndarray:
        """按名字取方法专属诊断数组（不存在时抛 ``KeyError``，不返回空数组冒充）。"""
        for key, val in self.extras:
            if key == name:
                return val
        raise KeyError(f"轨迹里没有诊断量 {name!r}（有：{[k for k, _ in self.extras]}）")

    def has(self, name: str) -> bool:
        return any(k == name for k, _ in self.extras)

    def at(self, k: int) -> dict[str, float]:
        """第 ``k`` 步的标量快照（给帧 / 报告用）。"""
        kk = int(np.clip(k, 0, self.n_iter))
        out = {
            "k": float(kk),
            "f": float(self.f[kk]),
            "gnorm": float(self.gnorm[kk]),
            "dist": float(self.dist[kk]) if self.dist.size else float("nan"),
            "ferr": float(self.ferr[kk]) if self.ferr.size else float("nan"),
        }
        if self.step.size and kk < self.step.size:
            out["step_norm"] = float(self.step[kk])
        if self.alphas.size and kk < self.alphas.size:
            out["alpha"] = float(self.alphas[kk])
        return out


def _blank() -> np.ndarray:
    r""""这个方法没有这个量"用空数组表示，不用 0 冒充。"""
    return np.zeros(0, dtype=float)


def _build_trace(
    xs: list[np.ndarray],
    fs: list[float],
    gs: list[float],
    steps: list[float],
    alphas: list[float],
    obj: Any,
    stop: str,
    *,
    secant: list[float] | None = None,
    radius: list[float] | None = None,
    ratio: list[float] | None = None,
    accepted: list[bool] | None = None,
    extras: list[tuple[str, list[Any]]] | None = None,
) -> Trace:
    dist: list[float] = []
    ferr: list[float] = []
    has_star = getattr(obj, "x_star", None) is not None
    for x in xs:
        dist.append(float(obj.dist(x)) if has_star else float("nan"))
        ferr.append(float(obj.ferr(x)) if has_star else float("nan"))
    ex: list[tuple[str, np.ndarray]] = []
    for name, vals in extras or []:
        arr = np.asarray(vals)
        ex.append((name, arr))
    return Trace(
        xs=np.asarray(xs, dtype=float),
        f=np.asarray(fs, dtype=float),
        gnorm=np.asarray(gs, dtype=float),
        step=np.asarray(steps, dtype=float),
        alphas=np.asarray(alphas, dtype=float),
        dist=np.asarray(dist, dtype=float) if has_star else _blank(),
        ferr=np.asarray(ferr, dtype=float) if has_star else _blank(),
        n_iter=len(steps),
        stop=str(stop),
        secant_residual=np.asarray(secant if secant is not None else [], dtype=float),
        radius=np.asarray(radius if radius is not None else [], dtype=float),
        ratio=np.asarray(ratio if ratio is not None else [], dtype=float),
        accepted=np.asarray(accepted if accepted is not None else [], dtype=bool),
        extras=tuple(ex),
    )


# --------------------------------------------------------------------------- #
# 问题构造
# --------------------------------------------------------------------------- #
def _spectrum_matrix(lam: np.ndarray, Q: np.ndarray) -> np.ndarray:
    r""":math:`A=Q\,\mathrm{diag}(\lambda)\,Q^{\mathsf T}`，并对称化到 :math:`O(u)`。"""
    A = (Q * lam) @ Q.T
    return 0.5 * (A + A.T)


def _finish_quadratic(A: np.ndarray, b: np.ndarray, x_star: np.ndarray) -> Quadratic:
    w, V = hermitian_jacobi_eigh(A)
    w = np.real(np.asarray(w, dtype=complex)).ravel()
    V = np.real(np.asarray(V, dtype=complex))
    f_star = -0.5 * float(np.dot(b, x_star))
    return Quadratic(
        A=A,
        b=np.asarray(b, dtype=float),
        x_star=np.asarray(x_star, dtype=float),
        eigenvalues=np.sort(w),
        eigenvectors=V,
        f_star=f_star,
    )


def quadratic_from_spectrum(
    lam: Sequence[float],
    *,
    seed: int = 20260501,
    rotate: bool = True,
    x_star: Any = None,
) -> Quadratic:
    r"""给定正谱 :math:`\lambda_1\le\dots\le\lambda_n>0` 的随机旋转二次型。

    ``rotate=True`` 时用 :func:`orthonormal_matrix` 造随机正交 :math:`Q`——**不旋转**
    的病态二次型（坐标轴对齐）会让梯度下降恰好落在特征方向上，一步到底，
    于是"κ 大就慢"这个结论根本演示不出来（这是本章第一个差点写成假证据的地方）。
    """
    lv = np.sort(np.asarray(lam, dtype=float).ravel())
    if lv.size == 0 or not bool(np.all(lv > 0.0)):
        raise ValueError(f"谱必须是正数向量，得到 {lv!r}")
    n = int(lv.size)
    Q = orthonormal_matrix(n, seed=seed) if rotate else np.eye(n)
    A = _spectrum_matrix(lv, Q)
    xs = np.ones(n) if x_star is None else np.asarray(x_star, dtype=float).ravel()
    b = A @ xs
    return _finish_quadratic(A, b, xs)


def anisotropic_quadratic(
    kappa: float,
    *,
    angle_deg: float = 35.0,
    x_star: Any = None,
    seed: int = 20260501,
) -> Quadratic:
    r"""二维各向异性二次型：:math:`\lambda=(1,\kappa)`，长轴与 :math:`x_1` 轴成 ``angle_deg``。

    这是"下降路径赛跑"与"收敛率图"的公共问题。转动角度是**必须**的参数：
    :math:`\theta=0` 时起点若落在坐标轴上，GD 一步到位，锯齿消失。
    """
    k = float(kappa)
    if not k >= 1.0:
        raise ValueError(f"条件数必须 ≥ 1，得到 {k!r}")
    th = np.deg2rad(float(angle_deg))
    c, s = float(np.cos(th)), float(np.sin(th))
    R = np.array([[c, -s], [s, c]], dtype=float)
    A = _spectrum_matrix(np.array([1.0, k]), R)
    xs = np.zeros(2) if x_star is None else np.asarray(x_star, dtype=float).ravel()
    b = A @ xs
    q = _finish_quadratic(A, b, xs)
    if q.mu <= 0.0:  # pragma: no cover - 由上面的 k>=1 保证
        raise ValueError("anisotropic_quadratic 需要正定矩阵")
    return q


def saddle_quadratic(
    neg: float = -0.5,
    pos: float = 1.0,
    *,
    angle_deg: float = 0.0,
) -> Quadratic:
    r"""二维鞍点：:math:`f=\tfrac12(\lambda_+ x_1^2+\lambda_- x_2^2)`，:math:`\lambda_-<0<\lambda_+`。

    驻点在原点，但它**不是**极小点：:math:`\lambda_{\min}=\lambda_-<0`。
    ``x_star`` 记的是那个驻点（:meth:`Quadratic.dist` 因此是"到驻点的距离"），
    而 :meth:`Quadratic.ferr` 在负曲率方向上是负的——这不是 bug，是鞍点的定义。
    """
    if not (float(neg) < 0.0 < float(pos)):
        raise ValueError(f"需要 neg < 0 < pos，得到 neg={neg!r}, pos={pos!r}")
    th = np.deg2rad(float(angle_deg))
    c, s = float(np.cos(th)), float(np.sin(th))
    R = np.array([[c, -s], [s, c]], dtype=float)
    lam = np.array([float(neg), float(pos)])
    A = _spectrum_matrix(lam, R)
    xs = np.zeros(2)
    b = A @ xs
    return _finish_quadratic(A, b, xs)


def quartic_quadratic(
    kappa: float,
    *,
    angle_deg: float = 35.0,
    beta: float = 0.05,
) -> Objective:
    r""""各向异性四次型" :math:`f(x)=\tfrac12x^{\mathsf T}Ax+\tfrac{\beta}{4}\|x\|_2^4`。

    :math:`A` 是 :func:`anisotropic_quadratic` 的矩阵，于是

    .. math::
        \nabla f=(A+\beta\|x\|^2I)x,\qquad
        \nabla^2f=A+\beta(\|x\|^2I+2xx^{\mathsf T})\succ0 .

    **为什么造它**：二次型上牛顿法一步到底，根本看不到"二次收敛的尾巴"；
    而 Rosenbrock 的 :math:`x_*` 要另想办法验证。这个函数的极小点**精确地**是原点
    （:math:`\nabla f(0)=0` 且 Hessian 处处正定），Hessian 又随 :math:`\|x\|` 变化，
    于是三种收敛形态（线性 / 超线性 / 二次）能在同一张半对数图上同时出现，
    且"到最优解的距离"是精确可算的。
    """
    base = anisotropic_quadratic(kappa, angle_deg=angle_deg)
    A = base.A

    def value(x: np.ndarray) -> float:
        xv = np.asarray(x, dtype=float).ravel()
        r2 = float(np.dot(xv, xv))
        return 0.5 * float(np.dot(xv, A @ xv)) + 0.25 * float(beta) * r2 * r2

    def grad(x: np.ndarray) -> np.ndarray:
        xv = np.asarray(x, dtype=float).ravel()
        return A @ xv + float(beta) * float(np.dot(xv, xv)) * xv

    def hess(x: np.ndarray) -> np.ndarray:
        xv = np.asarray(x, dtype=float).ravel()
        n = xv.size
        r2 = float(np.dot(xv, xv))
        return A + float(beta) * (r2 * np.eye(n) + 2.0 * np.outer(xv, xv))

    return Objective(
        value=value,
        grad=grad,
        hess=hess,
        x_star=np.zeros(2),
        f_star=0.0,
        is_quadratic=False,
    )


def rosenbrock_objective() -> Objective:
    r"""Rosenbrock 香蕉函数 :math:`f=100(y-x^2)^2+(1-x)^2`，极小点 :math:`(1,1)`，:math:`f_*=0`。

    非凸、Hessian 在远处不定、谷底极扁——**"牛顿法总是二次收敛"最常翻车的地方**。
    """
    def value(x: np.ndarray) -> float:
        a, b = float(x[0]), float(x[1])
        return 100.0 * (b - a * a) ** 2 + (1.0 - a) ** 2

    def grad(x: np.ndarray) -> np.ndarray:
        a, b = float(x[0]), float(x[1])
        return np.array(
            [-400.0 * a * (b - a * a) - 2.0 * (1.0 - a), 200.0 * (b - a * a)],
            dtype=float,
        )

    def hess(x: np.ndarray) -> np.ndarray:
        a, b = float(x[0]), float(x[1])
        return np.array(
            [[-400.0 * (b - 3.0 * a * a) + 2.0, -400.0 * a], [-400.0 * a, 200.0]],
            dtype=float,
        )

    return Objective(
        value=value,
        grad=grad,
        hess=hess,
        x_star=np.array([1.0, 1.0]),
        f_star=0.0,
        is_quadratic=False,
    )


# --------------------------------------------------------------------------- #
# 收敛率（理论侧）
# --------------------------------------------------------------------------- #
def divergence_limit(x0: Any) -> float:
    r"""发散判据的**相对**阈值：迭代点比初值放大 :math:`1/u` 倍（≈4.5e15）即判为数值发散。

    用相对量而不是固定的 1e150：``x0`` 本身可能在 1e-8 量级，
    "长了 15 个数量级"在任何尺度下都是同一件事，而 1e150 会让
    :math:`\alpha>2/L` 的慢发散（每步 ×2.2）跑满 ``max_iter`` 才被抓住——
    那时 ``stop`` 会写成 ``max_iter``，看上去像"只是没跑完"。
    """
    return max(1.0, euclidean_norm(np.asarray(x0, dtype=float))) / float(np.finfo(float).eps)


def gd_rate(kappa: float) -> float:
    r"""最优定步长下梯度下降的误差压缩因子 :math:`\rho=\frac{\kappa-1}{\kappa+1}`。"""
    k = float(kappa)
    return (k - 1.0) / (k + 1.0)


def gd_optimal_step(L: float, mu: float) -> float:
    r""":math:`\alpha^\star=2/(L+\mu)`。"""
    return 2.0 / (float(L) + float(mu))


def gd_iterations(kappa: float, eps: float, *, step: float | None = None) -> float:
    r"""达到 :math:`\|e_k\|\le\varepsilon\|e_0\|` 的**理论**迭代数。

    :math:`N=\lceil\ln\varepsilon/\ln\rho\rceil`；
    :math:`\kappa\gg1` 时 :math:`\rho\approx1-2/\kappa`，故 :math:`N\approx\frac{\kappa}{2}\ln\frac1\varepsilon`
    ——"迭代数与条件数成正比"就是这一行。
    """
    k = float(kappa)
    rho = (k - 1.0) / (k + 1.0) if step is None else max(abs(1.0 - float(step) * 1.0), abs(1.0 - float(step) * k))
    e = float(eps)
    if not (0.0 < e < 1.0):
        raise ValueError(f"eps 必须在 (0,1)，得到 {eps!r}")
    if not (0.0 < rho < 1.0):
        return float("inf")
    return float(np.ceil(np.log(e) / np.log(rho)))


def momentum_parameters(kappa: float) -> tuple[float, float]:
    r"""重球动量的经典最优参数 :math:`\beta=\left(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\right)^2`、
    :math:`\alpha=\frac{4}{(\sqrt L+\sqrt\mu)^2}`（此处 :math:`\mu=1,L=\kappa`）。"""
    k = float(kappa)
    r = np.sqrt(k)
    beta = ((r - 1.0) / (r + 1.0)) ** 2
    step = 4.0 / (r + 1.0) ** 2
    return float(beta), float(step)


def momentum_rate(kappa: float) -> float:
    r"""重球动量的压缩因子 :math:`\rho=\frac{\sqrt\kappa-1}{\sqrt\kappa+1}`（对比 GD 的 :math:`\frac{\kappa-1}{\kappa+1}`）。"""
    r = np.sqrt(float(kappa))
    return float((r - 1.0) / (r + 1.0))


def momentum_iterations(kappa: float, eps: float) -> float:
    r"""重球动量的理论迭代数 :math:`\approx\frac{\sqrt\kappa}{2}\ln\frac1\varepsilon`。"""
    rho = momentum_rate(kappa)
    e = float(eps)
    if not (0.0 < e < 1.0):
        raise ValueError(f"eps 必须在 (0,1)，得到 {eps!r}")
    return float(np.ceil(np.log(e) / np.log(rho)))


# --------------------------------------------------------------------------- #
# 线搜索
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LineSearchResult:
    r"""一次线搜索的结果。

    ``ok`` 表示"这一步**可用**"（Armijo 充分下降成立），``wolfe`` 表示"曲率条件也成立"。
    两者分开是刻意的：拟牛顿法可以接受只有 Armijo 的步，但那时 :math:`y^{\mathsf T}s>0`
    没有保证，**更新必须被跳过**——把 ``ok`` 与 ``wolfe`` 合并成一个布尔量，
    就会把"线搜索退化"与"方法失效"混为一谈。
    """

    alpha: float
    ok: bool
    n_evals: int
    slope: float
    reason: str
    wolfe: bool = False


def armijo_backtracking(
    obj: Any,
    x: np.ndarray,
    f_x: float,
    g_x: np.ndarray,
    p: np.ndarray,
    *,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5,
    max_back: int = 60,
) -> LineSearchResult:
    r"""Armijo 回溯：找 :math:`\alpha` 使 :math:`f(x+\alpha p)\le f(x)+c_1\alpha\langle g,p\rangle`。

    只保证**充分下降**，不保证曲率条件。于是它可能接受一个极小的步长把迭代拖成
    "几乎不动"——这正是"线搜索失效"的一种可测形态（``n_evals`` 很大而 ``alpha`` 极小）。
    """
    slope = inner(g_x, p)
    if not slope < 0.0:
        return LineSearchResult(1.0, False, 0, slope, "not_a_descent_direction")
    alpha = float(alpha0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for i in range(int(max_back) + 1):
            f_try = float(obj.value(x + alpha * p))
            if np.isfinite(f_try) and f_try <= f_x + c1 * alpha * slope:
                return LineSearchResult(alpha, True, i + 1, slope, "armijo")
            alpha *= float(rho)
            if not (alpha > 1e-300):
                break
    return LineSearchResult(alpha, False, int(max_back) + 1, slope, "max_backtracking")


def wolfe_line_search(
    obj: Any,
    x: np.ndarray,
    f_x: float,
    g_x: np.ndarray,
    p: np.ndarray,
    *,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iter: int = 40,
    alpha_max: float = 1e3,
) -> LineSearchResult:
    r"""强 Wolfe 线搜索（**二分 zoom**，无插值假设）。

    强 Wolfe 条件

    .. math::
        f(x+\alpha p)\le f(x)+c_1\alpha\langle g,p\rangle,\qquad
        |\langle\nabla f(x+\alpha p),p\rangle|\le c_2|\langle g,p\rangle|

    第二条（曲率条件）正是 BFGS 的 :math:`y^{\mathsf T}s>0` 的来源。
    但**它不一定找得到**：当 :math:`f` 已经落到舍入地板、或方向 :math:`p` 快与
    :math:`g` 正交时，满足曲率条件的 :math:`\alpha` 窗口会窄到机器精度以下。
    这时本函数返回"Armijo 成立、曲率不成立"的步（``ok=True, wolfe=False,
    reason="zoom_exhausted"``），而不是假装失败或假装成功——调用方据此跳过拟牛顿更新。
    """
    slope0 = inner(g_x, p)
    n_eval = 0
    if not slope0 < 0.0:
        return LineSearchResult(1.0, False, 0, slope0, "not_a_descent_direction")

    def phi(a: float) -> float:
        return float(obj.value(x + a * p))

    def dphi(a: float) -> float:
        return inner(obj.grad(x + a * p), p)

    def armijo_ok(a: float, ph: float) -> bool:
        return bool(np.isfinite(ph)) and ph <= f_x + c1 * a * slope0

    a_prev, phi_prev = 0.0, float(f_x)
    a = float(alpha0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for i in range(int(max_iter)):
            n_eval += 1
            ph = phi(a)
            if not np.isfinite(ph) or ph > f_x + c1 * a * slope0 or (i > 0 and ph >= phi_prev):
                return _zoom(obj, x, p, f_x, slope0, a_prev, a, c1, c2, max_iter, n_eval)
            dph = dphi(a)
            if abs(dph) <= -c2 * slope0:
                return LineSearchResult(a, True, n_eval, slope0, "wolfe", wolfe=True)
            if dph >= 0.0:
                return _zoom(obj, x, p, f_x, slope0, a, a_prev, c1, c2, max_iter, n_eval)
            a_prev, phi_prev = a, ph
            a = min(2.0 * a, float(alpha_max))
        # 扩张用尽：退而求其次，只要 Armijo 成立就交回可用步（并标明曲率未验）
        if armijo_ok(a, phi(a)):
            return LineSearchResult(a, True, n_eval, slope0, "expansion_exhausted", wolfe=False)
    return LineSearchResult(a, False, n_eval, slope0, "max_iter")


def _zoom(
    obj: Any,
    x: np.ndarray,
    p: np.ndarray,
    f_x: float,
    slope0: float,
    a_lo: float,
    a_hi: float,
    c1: float,
    c2: float,
    max_iter: int,
    n_eval: int,
) -> LineSearchResult:
    r"""在 ``[a_lo, a_hi]`` 上二分（zoom），返回满足强 Wolfe 的步长。

    二分用尽仍未命中曲率条件时，返回区间内**Armijo 成立且** :math:`|\varphi'|` **最小**
    的那个点（``reason="zoom_exhausted"``）：它仍然是一个合法的下降步，
    只是曲率条件没有保证，拟牛顿更新会被调用方跳过。
    """
    best_a: float | None = None
    best_d = float("inf")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(int(max_iter)):
            a = 0.5 * (a_lo + a_hi)
            n_eval += 1
            ph = float(obj.value(x + a * p))
            if not np.isfinite(ph) or ph > f_x + c1 * a * slope0:
                a_hi = a
            else:
                dph = inner(obj.grad(x + a * p), p)
                if abs(dph) <= -c2 * slope0:
                    return LineSearchResult(a, True, n_eval, slope0, "wolfe", wolfe=True)
                if abs(dph) < best_d:
                    best_a, best_d = a, abs(dph)
                if dph * (a_hi - a_lo) >= 0.0:
                    a_hi = a_lo
                a_lo = a
            if abs(a_hi - a_lo) <= 1e-16 * max(1.0, abs(a_hi)):
                break
    if best_a is not None:
        return LineSearchResult(best_a, True, n_eval, slope0, "zoom_exhausted", wolfe=False)
    a_mid = 0.5 * (a_lo + a_hi)
    ph_mid = float(obj.value(x + a_mid * p))
    ok = bool(np.isfinite(ph_mid)) and ph_mid <= f_x + c1 * a_mid * slope0
    return LineSearchResult(a_mid, ok, n_eval, slope0, "zoom_failed")


def exact_step_quadratic(obj: Any, x: np.ndarray, p: np.ndarray) -> float:
    r"""二次型上的**精确**线搜索 :math:`\alpha^\star=-\frac{\langle g,p\rangle}{\langle p,Ap\rangle}`。

    只对 :class:`Quadratic` 有效（``is_quadratic`` 为假时抛 ``ValueError``）：
    把二次型的精确步长用到 Rosenbrock 上会得到"沿这条直线最优"的步长，
    但那不是全局最优步长，当"精确"讲就是假的。
    """
    if not bool(getattr(obj, "is_quadratic", False)):
        raise ValueError("exact_step_quadratic 只适用于二次型（obj.is_quadratic 为 False）")
    g = np.asarray(obj.grad(x), dtype=float)
    pv = np.asarray(p, dtype=float)
    den = float(np.dot(pv, obj.A @ pv))
    num = -inner(g, pv)
    if not (den > 0.0):
        return float("nan")
    return num / den


# --------------------------------------------------------------------------- #
# 信赖域子问题：Steihaug-CG
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SteihaugResult:
    r"""信赖域子问题的近似解 :math:`p`，以及三个"为什么停下"的布尔量。"""

    p: np.ndarray
    hit_boundary: bool
    negative_curvature: bool
    n_iter: int
    pnorm: float
    predicted: float
    reason: str


def _boundary_tau(p: np.ndarray, d: np.ndarray, delta: float) -> float:
    r"""求 :math:`\tau>0` 使 :math:`\|p+\tau d\|=\Delta`（解一元二次，取正根）。"""
    a = float(np.dot(d, d))
    b = 2.0 * float(np.dot(p, d))
    c = float(np.dot(p, p)) - float(delta) ** 2
    if not (a > 0.0):
        return 0.0
    disc = b * b - 4.0 * a * c
    disc = max(disc, 0.0)
    return float((-b + np.sqrt(disc)) / (2.0 * a))


def steihaug_cg(
    B_apply: Callable[[np.ndarray], np.ndarray],
    g: Any,
    delta: float,
    *,
    tol: float = 1e-12,
    max_iter: int | None = None,
) -> SteihaugResult:
    r"""Steihaug-CG 解信赖域子问题 :math:`\min_{\|p\|\le\Delta} g^{\mathsf T}p+\tfrac12p^{\mathsf T}Bp`。

    两条提前退出规则正是它处理**不定** Hessian 的方式：

    * 遇到负曲率方向（:math:`d^{\mathsf T}Bd\le0`）→ 沿该方向一路走到边界；
    * 迭代点越出球面 → 沿当前方向取边界交点。

    于是"负曲率"不是一个事后判断，而是子问题求解过程中的一个**分支**——
    这是信赖域与线搜索在非凸问题上的本质差别。
    """
    gv = np.asarray(g, dtype=float).ravel()
    n = gv.size
    dmax = n if max_iter is None else int(max_iter)
    p = np.zeros(n, dtype=float)
    r = gv.copy()
    d = -r
    gnorm0 = euclidean_norm(gv)
    if gnorm0 == 0.0:
        return SteihaugResult(p, False, False, 0, 0.0, 0.0, "zero_gradient")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        rr = float(np.dot(r, r))
        for j in range(int(dmax)):
            Bd = np.asarray(B_apply(d), dtype=float).ravel()
            dBd = float(np.dot(d, Bd))
            if not np.isfinite(dBd):
                return SteihaugResult(p, True, False, j, euclidean_norm(p), 0.0, "nonfinite")
            if dBd <= 0.0:
                tau = _boundary_tau(p, d, float(delta))
                p_new = p + tau * d
                pred = -(inner(gv, p_new) + 0.5 * float(np.dot(p_new, np.asarray(B_apply(p_new), dtype=float))))
                return SteihaugResult(p_new, True, True, j + 1, euclidean_norm(p_new), float(pred), "negative_curvature")
            alpha = rr / dBd
            p_new = p + alpha * d
            if euclidean_norm(p_new) >= float(delta):
                tau = _boundary_tau(p, d, float(delta))
                p_new = p + tau * d
                pred = -(inner(gv, p_new) + 0.5 * float(np.dot(p_new, np.asarray(B_apply(p_new), dtype=float))))
                return SteihaugResult(p_new, True, False, j + 1, euclidean_norm(p_new), float(pred), "boundary")
            r_new = r + alpha * Bd
            rr_new = float(np.dot(r_new, r_new))
            if np.sqrt(max(rr_new, 0.0)) <= tol * gnorm0:
                pred = -(inner(gv, p_new) + 0.5 * float(np.dot(p_new, np.asarray(B_apply(p_new), dtype=float))))
                return SteihaugResult(p_new, False, False, j + 1, euclidean_norm(p_new), float(pred), "converged")
            beta = rr_new / rr
            d = -r_new + beta * d
            p, r, rr = p_new, r_new, rr_new
    pred = -(inner(gv, p) + 0.5 * float(np.dot(p, np.asarray(B_apply(p), dtype=float))))
    return SteihaugResult(p, False, False, int(dmax), euclidean_norm(p), float(pred), "max_iter")


# --------------------------------------------------------------------------- #
# 停止判据（三种口径，**不要**混为一谈）
# --------------------------------------------------------------------------- #
def _stop_reason(
    x: np.ndarray,
    f: float,
    g: np.ndarray,
    step_norm: float,
    *,
    tol: float,
    tol_kind: str,
    g0: float,
    f_prev: float | None,
) -> str:
    r"""三种判据三个口径：

    * ``"grad"``：:math:`\|\nabla f_k\|_2\le\varepsilon\max(1,\|\nabla f_0\|_2)`；
    * ``"step"``：:math:`\|x_{k+1}-x_k\|_2\le\varepsilon\max(1,\|x_{k+1}\|_2)`；
    * ``"frel"``：:math:`|f_{k+1}-f_k|\le\varepsilon\max(1,|f_{k+1}|)`。

    病态问题上三者的结论可以差一个量级——本章的断言会把这件事测出来。
    """
    if tol_kind == "grad":
        return "grad" if euclidean_norm(g) <= float(tol) * max(1.0, g0) else ""
    if tol_kind == "step":
        return "step" if step_norm <= float(tol) * max(1.0, euclidean_norm(x)) else ""
    if tol_kind == "frel":
        if f_prev is not None and abs(f - f_prev) <= float(tol) * max(1.0, abs(f)):
            return "frel"
        return ""
    raise ValueError(f"未知的停止判据 {tol_kind!r}（可选 grad / step / frel）")


# --------------------------------------------------------------------------- #
# 算法
# --------------------------------------------------------------------------- #
def gd(
    obj: Any,
    x0: Any,
    *,
    step: float | None = None,
    max_iter: int = 1000,
    tol: float = 1e-10,
    tol_kind: str = "grad",
) -> Trace:
    r"""定步长梯度下降 :math:`x_{k+1}=x_k-\alpha\nabla f(x_k)`。

    ``step=None`` 时用二次型的最优步长 :math:`2/(L+\mu)`（非二次型必须显式给 ``step``，
    否则就是拿一个没有依据的数在跑）。``step > 2/L`` 时迭代**发散**——
    这个阈值由 C 类断言二分实测，不是抄来的。
    """
    x = np.asarray(x0, dtype=float).ravel()
    if step is None:
        if not bool(getattr(obj, "is_quadratic", False)):
            raise ValueError("非二次型必须显式给 step（没有 L、mu 就没有最优步长的依据）")
        if not bool(obj.has_minimum()):
            raise ValueError(
                "问题不是强凸的（λ_min ≤ 0）：2/(L+μ) 这个「最优步长」对鞍点没有意义，"
                "请显式给 step（例如 1/L 是保证 |1−αλ|<1 的安全取值）"
            )
        step = float(obj.optimal_step())
    alpha = float(step)
    if not (alpha > 0.0):
        raise ValueError(f"步长必须为正，得到 {step!r}")
    xs, fs, gs, steps, alphas = [x.copy()], [], [], [], []
    div_limit = divergence_limit(x)
    g = np.asarray(obj.grad(x), dtype=float).ravel()
    f = float(obj.value(x))
    g0 = euclidean_norm(g)
    fs.append(f)
    gs.append(g0)
    stop = "max_iter"
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for k in range(int(max_iter)):
            reason = _stop_reason(x, f, g, steps[-1] if steps else float("inf"), tol=tol, tol_kind=tol_kind, g0=g0, f_prev=fs[-2] if len(fs) > 1 else None)
            if reason:
                stop = reason
                break
            x_new = x - alpha * g
            if not bool(np.all(np.isfinite(x_new))) or euclidean_norm(x_new) > div_limit:
                stop = "diverged"
                break
            f_new = float(obj.value(x_new))
            if not np.isfinite(f_new):
                stop = "diverged"
                break
            g_new = np.asarray(obj.grad(x_new), dtype=float).ravel()
            if not bool(np.all(np.isfinite(g_new))):
                stop = "diverged"
                break
            steps.append(euclidean_norm(x_new - x))
            alphas.append(alpha)
            x, f, g = x_new, f_new, g_new
            xs.append(x.copy())
            fs.append(f)
            gs.append(euclidean_norm(g))
    return _build_trace(xs, fs, gs, steps, alphas, obj, stop)


def momentum(
    obj: Any,
    x0: Any,
    *,
    step: float | None = None,
    beta: float = 0.9,
    max_iter: int = 1000,
    tol: float = 1e-10,
    tol_kind: str = "grad",
) -> Trace:
    r"""重球动量法 :math:`v_{k+1}=\beta v_k-\alpha\nabla f(x_k),\ x_{k+1}=x_k+v_{k+1}`。

    ``step=None`` 时取经典最优参数 :math:`\beta=\left(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\right)^2`、
    :math:`\alpha=4/(\sqrt L+\sqrt\mu)^2`——同样只对二次型成立。
    """
    x = np.asarray(x0, dtype=float).ravel()
    if step is None:
        if not bool(getattr(obj, "is_quadratic", False)):
            raise ValueError("非二次型必须显式给 step")
        b_theory, s_theory = momentum_parameters(obj.kappa)
        step = s_theory
        beta = b_theory
    alpha = float(step)
    b_used = float(beta)
    v = np.zeros_like(x)
    xs, fs, gs, steps, alphas, betas = [x.copy()], [], [], [], [], []
    div_limit = divergence_limit(x)
    g = np.asarray(obj.grad(x), dtype=float).ravel()
    f = float(obj.value(x))
    g0 = euclidean_norm(g)
    fs.append(f)
    gs.append(g0)
    stop = "max_iter"
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(int(max_iter)):
            reason = _stop_reason(x, f, g, steps[-1] if steps else float("inf"), tol=tol, tol_kind=tol_kind, g0=g0, f_prev=fs[-2] if len(fs) > 1 else None)
            if reason:
                stop = reason
                break
            v = b_used * v - alpha * g
            x_new = x + v
            if not bool(np.all(np.isfinite(x_new))) or euclidean_norm(x_new) > div_limit:
                stop = "diverged"
                break
            f_new = float(obj.value(x_new))
            g_new = np.asarray(obj.grad(x_new), dtype=float).ravel()
            if not (np.isfinite(f_new) and bool(np.all(np.isfinite(g_new)))):
                stop = "diverged"
                break
            steps.append(euclidean_norm(x_new - x))
            alphas.append(alpha)
            betas.append(b_used)
            x, f, g = x_new, f_new, g_new
            xs.append(x.copy())
            fs.append(f)
            gs.append(euclidean_norm(g))
    return _build_trace(xs, fs, gs, steps, alphas, obj, stop, extras=[("beta", betas)])


def newton(
    obj: Any,
    x0: Any,
    *,
    line_search: str = "armijo",
    alpha_cap: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5,
    shift0: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-12,
    tol_kind: str = "grad",
) -> Trace:
    r"""牛顿法 :math:`x_{k+1}=x_k-\alpha_k H_k^{-1}g_k`。

    **"二次收敛"成立的两个前提都必须在返回值里看得见**：

    1. :math:`\alpha_k=1` 被线搜索接受（``alphas[k]==1``）——``alpha_cap<1`` 时永远
       拿不到整步，超线性立刻退化成线性（本章断言测了这条退化）；
    2. 迭代已经进入解附近的邻域——远处的 :math:`H_k` 可能不定或近奇异，
       此时步长会被回溯到很小，:math:`e_{k+1}/e_k^2` 那个常数根本不存在。

    ``line_search="none"`` 是无全球化的纯牛顿步（用来演示发散），
    ``"armijo"`` / ``"wolfe"`` 是两种全球化。
    """
    if getattr(obj, "hess", None) is None:
        raise ValueError("牛顿法需要 Hessian（obj.hess 为 None）")
    x = np.asarray(x0, dtype=float).ravel()
    xs, fs, gs, steps, alphas, nbacks, pds, shifts = [x.copy()], [], [], [], [], [], [], []
    div_limit = divergence_limit(x)
    g = np.asarray(obj.grad(x), dtype=float).ravel()
    f = float(obj.value(x))
    g0 = euclidean_norm(g)
    fs.append(f)
    gs.append(g0)
    stop = "max_iter"
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(int(max_iter)):
            reason = _stop_reason(x, f, g, steps[-1] if steps else float("inf"), tol=tol, tol_kind=tol_kind, g0=g0, f_prev=fs[-2] if len(fs) > 1 else None)
            if reason:
                stop = reason
                break
            H = np.asarray(obj.hess(x), dtype=float)
            try:
                p, pd, shift = hessian_step(H, g, shift0=shift0)
            except (ValueError, ZeroDivisionError):
                stop = "hessian_singular"
                break
            n_back = 0
            if line_search == "none":
                alpha, ok = float(alpha_cap), True
            else:
                a0 = float(alpha_cap)
                if line_search == "armijo":
                    ls = armijo_backtracking(obj, x, f, g, p, alpha0=a0, c1=c1, rho=rho)
                elif line_search == "wolfe":
                    ls = wolfe_line_search(obj, x, f, g, p, alpha0=a0, c1=c1)
                else:
                    raise ValueError(f"未知线搜索 {line_search!r}（none / armijo / wolfe）")
                alpha, ok, n_back = ls.alpha, ls.ok, ls.n_evals
                if not ok:
                    # 牛顿方向被线搜索否决 → 退回最速下降方向再试一次（并如实记录）
                    ls2 = armijo_backtracking(obj, x, f, g, -g, alpha0=1.0, c1=c1, rho=rho)
                    if not ls2.ok:
                        stop = "line_search_failed"
                        break
                    p, alpha, n_back = -g, ls2.alpha, n_back + ls2.n_evals
            x_new = x + alpha * p
            if not bool(np.all(np.isfinite(x_new))) or euclidean_norm(x_new) > div_limit:
                stop = "diverged"
                break
            f_new = float(obj.value(x_new))
            if not np.isfinite(f_new):
                stop = "diverged"
                break
            g_new = np.asarray(obj.grad(x_new), dtype=float).ravel()
            if not bool(np.all(np.isfinite(g_new))):
                stop = "diverged"
                break
            steps.append(euclidean_norm(x_new - x))
            alphas.append(float(alpha))
            nbacks.append(float(n_back))
            pds.append(1.0 if pd else 0.0)
            shifts.append(float(shift))
            x, f, g = x_new, f_new, g_new
            xs.append(x.copy())
            fs.append(f)
            gs.append(euclidean_norm(g))
    return _build_trace(
        xs,
        fs,
        gs,
        steps,
        alphas,
        obj,
        stop,
        extras=[
            ("n_backtracks", nbacks),
            ("hessian_pd", pds),
            ("shift", shifts),
        ],
    )


def bfgs(
    obj: Any,
    x0: Any,
    *,
    line_search: str = "wolfe",
    c1: float = 1e-4,
    c2: float = 0.9,
    alpha0: float = 1.0,
    B0: Any = None,
    curvature_factor: float = 1e-10,
    max_iter: int = 200,
    tol: float = 1e-10,
    tol_kind: str = "grad",
) -> Trace:
    r"""BFGS（**直接维护** :math:`B_k\approx\nabla^2f`，每步用自研 LU/Cholesky 解 :math:`B_kp=-g`）。

    .. math::
        B_{k+1}=B_k-\frac{B_ks_ks_k^{\mathsf T}B_k}{s_k^{\mathsf T}B_ks_k}
        +\frac{y_ky_k^{\mathsf T}}{y_k^{\mathsf T}s_k},\qquad s_k=x_{k+1}-x_k,\ y_k=g_{k+1}-g_k.

    维护 :math:`B` 而不是 :math:`H=B^{-1}` 的理由是**可测**：割线方程 :math:`B_{k+1}s_k=y_k`
    的残差可以直接算出来（:func:`secant_residual`），而 :math:`H` 版本要先求逆才能测。
    返回值里 ``secant_residual`` 是**更新后**的残差，
    ``extras["secant_before"]`` 是**更新前**的残差——前者恒为 :math:`O(u)`，
    后者一般不是，两者混着报就是假证据。
    """
    x = np.asarray(x0, dtype=float).ravel()
    n = x.size
    B = np.array(B0, dtype=float) if B0 is not None else np.array(_eye(n), dtype=float)
    xs, fs, gs, steps, alphas = [x.copy()], [], [], [], []
    secant_after: list[float] = []
    secant_before: list[float] = []
    updated: list[float] = []
    chol_ok: list[float] = []
    wolfe_ok: list[float] = []
    g = np.asarray(obj.grad(x), dtype=float).ravel()
    f = float(obj.value(x))
    g0 = euclidean_norm(g)
    fs.append(f)
    gs.append(g0)
    stop = "max_iter"
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(int(max_iter)):
            reason = _stop_reason(x, f, g, steps[-1] if steps else float("inf"), tol=tol, tol_kind=tol_kind, g0=g0, f_prev=fs[-2] if len(fs) > 1 else None)
            if reason:
                stop = reason
                break
            try:
                p, pd = solve_symmetric(B, -g)
            except ZeroDivisionError:
                stop = "hessian_singular"
                break
            chol_ok.append(1.0 if pd else 0.0)
            if inner(g, p) >= 0.0:  # 不是下降方向：用最速下降兜底（并如实标出）
                p = -g
            if line_search == "armijo":
                ls = armijo_backtracking(obj, x, f, g, p, alpha0=alpha0, c1=c1)
            elif line_search == "wolfe":
                ls = wolfe_line_search(obj, x, f, g, p, alpha0=alpha0, c1=c1, c2=c2)
            else:
                raise ValueError(f"未知线搜索 {line_search!r}（armijo / wolfe）")
            if not ls.ok:
                stop = f"line_search_failed:{ls.reason}"
                break
            alpha = float(ls.alpha)
            s = alpha * p
            x_new = x + s
            f_new = float(obj.value(x_new))
            g_new = np.asarray(obj.grad(x_new), dtype=float).ravel()
            if not (np.isfinite(f_new) and bool(np.all(np.isfinite(g_new)))):
                stop = "diverged"
                break
            y = g_new - g
            before = secant_residual(B, s, y)
            sy = inner(s, y)
            good = sy > float(curvature_factor) * euclidean_norm(s) * euclidean_norm(y)
            if good:
                Bs = B @ s
                sBs = inner(s, Bs)
                if sBs > 0.0:
                    B = B - np.outer(Bs, Bs) / sBs + np.outer(y, y) / sy
            updated.append(1.0 if good else 0.0)
            wolfe_ok.append(1.0 if ls.wolfe else 0.0)
            secant_before.append(float(before))
            secant_after.append(secant_residual(B, s, y))
            steps.append(euclidean_norm(s))
            alphas.append(alpha)
            x, f, g = x_new, f_new, g_new
            xs.append(x.copy())
            fs.append(f)
            gs.append(euclidean_norm(g))
    return _build_trace(
        xs,
        fs,
        gs,
        steps,
        alphas,
        obj,
        stop,
        secant=secant_after,
        extras=[
            ("secant_before", secant_before),
            ("updated", updated),
            ("cholesky_ok", chol_ok),
            ("wolfe_ok", wolfe_ok),
        ],
    )


def _two_loop(
    g: np.ndarray,
    S: list[np.ndarray],
    Y: list[np.ndarray],
    gamma: float,
) -> np.ndarray:
    r"""L-BFGS 双循环递归：返回 :math:`-H_kg`（:math:`H_k` 是隐式逆 Hessian 近似）。

    外边循环先乘 :math:`\prod(1-\rho_jy_js_j^{\mathsf T})`，内边循环再乘 :math:`\prod(1-\rho_js_jy_j^{\mathsf T})`，
    中间夹一个尺度 :math:`\gamma=s^{\mathsf T}y/y^{\mathsf T}y`（Nocedal & Wright 算法 7.4）。
    """
    m = len(S)
    q = np.array(g, dtype=float, copy=True)
    alpha_list: list[float] = []
    rho_list = [1.0 / inner(y, s) for s, y in zip(S, Y)]
    for i in range(m - 1, -1, -1):
        a = rho_list[i] * inner(S[i], q)
        alpha_list.append(a)
        q = q - a * Y[i]
    r = float(gamma) * q
    for i in range(m):
        b = rho_list[i] * inner(Y[i], r)
        r = r + S[i] * (alpha_list[m - 1 - i] - b)
    return -r


def _lbfgs_inverse_apply(
    v: np.ndarray,
    S: list[np.ndarray],
    Y: list[np.ndarray],
    gamma: float,
) -> np.ndarray:
    r"""把双循环当成线性算子 :math:`H_k v` 用（用于**测量**它对旧割线方程满足得如何）。

    注意符号层级：:func:`_two_loop` 返回的是 :math:`-H_kv`，所以这里再取一次负号
    才得到 :math:`H_kv`。第一版多写了一层负号，量出来的"割线残差"全是 :math:`O(1)`
    （最新那一对本来应当精确满足）——差一点就把它当成"L-BFGS 不满足割线方程"的证据。
    """
    return -_two_loop(np.asarray(v, dtype=float), S, Y, gamma)


def lbfgs(
    obj: Any,
    x0: Any,
    *,
    memory: int = 10,
    line_search: str = "wolfe",
    c1: float = 1e-4,
    c2: float = 0.9,
    alpha0: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-10,
    tol_kind: str = "grad",
) -> Trace:
    r"""L-BFGS：只存最近 ``memory`` 对 :math:`(s_j,y_j)`，用双循环递归隐式表示 :math:`H_k`。

    **它满足多少个割线方程**是可以数的：最新的那一对恒满足（构造使然），
    再往前第 :math:`j` 对只在 :math:`j<m` 时满足。返回值里
    ``extras["inverse_secant"]`` 是**终止时刻**对最近 :math:`m` 对逐对的残差
    :math:`\|H_ky_j-s_j\|/\|s_j\|`，第 0 项应当接近 :math:`u`，越老越大——
    "L-BFGS 的记忆 m 有多大用"这件事由此变成一个可读的数列，而不是"看起来收敛得不错"。
    """
    if int(memory) < 1:
        raise ValueError(f"memory 必须 ≥ 1，得到 {memory!r}")
    m_cap = int(memory)
    x = np.asarray(x0, dtype=float).ravel()
    xs, fs, gs, steps, alphas = [x.copy()], [], [], [], []
    S: list[np.ndarray] = []
    Y: list[np.ndarray] = []
    wolfe_ok: list[float] = []
    gamma = 1.0
    g = np.asarray(obj.grad(x), dtype=float).ravel()
    f = float(obj.value(x))
    g0 = euclidean_norm(g)
    fs.append(f)
    gs.append(g0)
    stop = "max_iter"
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(int(max_iter)):
            reason = _stop_reason(x, f, g, steps[-1] if steps else float("inf"), tol=tol, tol_kind=tol_kind, g0=g0, f_prev=fs[-2] if len(fs) > 1 else None)
            if reason:
                stop = reason
                break
            p = _two_loop(g, S, Y, gamma) if S else -g
            if inner(g, p) >= 0.0:
                p = -g
            if line_search == "armijo":
                ls = armijo_backtracking(obj, x, f, g, p, alpha0=alpha0, c1=c1)
            elif line_search == "wolfe":
                ls = wolfe_line_search(obj, x, f, g, p, alpha0=alpha0, c1=c1, c2=c2)
            else:
                raise ValueError(f"未知线搜索 {line_search!r}（armijo / wolfe）")
            if not ls.ok:
                stop = f"line_search_failed:{ls.reason}"
                break
            alpha = float(ls.alpha)
            s = alpha * p
            x_new = x + s
            f_new = float(obj.value(x_new))
            g_new = np.asarray(obj.grad(x_new), dtype=float).ravel()
            if not (np.isfinite(f_new) and bool(np.all(np.isfinite(g_new)))):
                stop = "diverged"
                break
            y = g_new - g
            wolfe_ok.append(1.0 if ls.wolfe else 0.0)
            sy = inner(s, y)
            if sy > 1e-10 * euclidean_norm(s) * euclidean_norm(y):
                S.append(s.copy())
                Y.append(y.copy())
                if len(S) > m_cap:
                    S.pop(0)
                    Y.pop(0)
                gamma = sy / inner(y, y)
            steps.append(euclidean_norm(s))
            alphas.append(alpha)
            x, f, g = x_new, f_new, g_new
            xs.append(x.copy())
            fs.append(f)
            gs.append(euclidean_norm(g))
        inv_sec: list[float] = []
        for j in range(len(S) - 1, -1, -1):
            Hs = _lbfgs_inverse_apply(Y[j], S, Y, gamma)
            den = euclidean_norm(S[j])
            inv_sec.append(float(euclidean_norm(Hs - S[j]) / den) if den > 0.0 else float("inf"))
    return _build_trace(
        xs,
        fs,
        gs,
        steps,
        alphas,
        obj,
        stop,
        extras=[("inverse_secant", inv_sec), ("memory", [float(m_cap)]), ("wolfe_ok", wolfe_ok)],
    )


def _beta_cg(variant: str, g_new: np.ndarray, g_old: np.ndarray) -> float:
    r"""非线性 CG 的 :math:`\beta` 公式（Fletcher-Reeves / Polak-Ribiere / Hestenes-Stiefel）。"""
    den = float(np.dot(g_old, g_old))
    if den == 0.0:
        return 0.0
    if variant == "FR":
        return float(np.dot(g_new, g_new) / den)
    if variant == "PR":
        return float(np.dot(g_new, g_new - g_old) / den)
    if variant == "PR+":
        return max(0.0, float(np.dot(g_new, g_new - g_old) / den))
    if variant == "HS":
        d = np.asarray(g_new - g_old, dtype=float)
        dd = float(np.dot(d, g_new - g_old))
        return float(np.dot(g_new, d) / dd) if dd != 0.0 else 0.0
    raise ValueError(f"未知的 beta 公式 {variant!r}（FR / PR / PR+ / HS）")


def cg(
    obj: Any,
    x0: Any,
    *,
    variant: str = "PR+",
    line_search: str = "wolfe",
    restart: int | None = None,
    c1: float = 1e-4,
    c2: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-10,
    tol_kind: str = "grad",
) -> Trace:
    r"""非线性共轭梯度 :math:`p_k=-g_k+\beta_kp_{k-1}`。

    ``line_search="exact"`` 只对二次型开放：那时 CG 有**有限步终止**性质
    （至多 :math:`n` 步精确到达），且方向两两 :math:`A`-共轭——这是本章最干净的
    A 类对拍之一。``restart`` 给定期重启周期（``None`` 表示不重启）。
    返回的 ``extras["directions"]`` 是方向历史，供断言直接验共轭性。
    """
    x = np.asarray(x0, dtype=float).ravel()
    xs, fs, gs, steps, alphas = [x.copy()], [], [], [], []
    betas: list[float] = []
    restarts: list[float] = []
    dirs: list[np.ndarray] = []
    g = np.asarray(obj.grad(x), dtype=float).ravel()
    f = float(obj.value(x))
    g0 = euclidean_norm(g)
    fs.append(f)
    gs.append(g0)
    p = -g
    stop = "max_iter"
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for k in range(int(max_iter)):
            reason = _stop_reason(x, f, g, steps[-1] if steps else float("inf"), tol=tol, tol_kind=tol_kind, g0=g0, f_prev=fs[-2] if len(fs) > 1 else None)
            if reason:
                stop = reason
                break
            dirs.append(p.copy())
            if line_search == "exact":
                alpha = exact_step_quadratic(obj, x, p)
                if not np.isfinite(alpha) or alpha <= 0.0:
                    stop = "exact_step_failed"
                    break
            elif line_search == "wolfe":
                ls = wolfe_line_search(obj, x, f, g, p, alpha0=1.0, c1=c1, c2=c2)
                if not ls.ok:
                    stop = f"line_search_failed:{ls.reason}"
                    break
                alpha = float(ls.alpha)
            elif line_search == "armijo":
                ls = armijo_backtracking(obj, x, f, g, p, alpha0=1.0, c1=c1)
                if not ls.ok:
                    stop = f"line_search_failed:{ls.reason}"
                    break
                alpha = float(ls.alpha)
            else:
                raise ValueError(f"未知线搜索 {line_search!r}（exact / wolfe / armijo）")
            x_new = x + alpha * p
            f_new = float(obj.value(x_new))
            g_new = np.asarray(obj.grad(x_new), dtype=float).ravel()
            if not (np.isfinite(f_new) and bool(np.all(np.isfinite(g_new)))):
                stop = "diverged"
                break
            beta = _beta_cg(variant, g_new, g)
            did_restart = False
            if restart is not None and (k + 1) % int(restart) == 0:
                beta = 0.0
                did_restart = True
            p_new = -g_new + beta * p
            if inner(p_new, g_new) >= 0.0:  # 非下降方向 → 重启
                p_new = -g_new
                did_restart = True
            steps.append(euclidean_norm(x_new - x))
            alphas.append(float(alpha))
            betas.append(float(beta))
            restarts.append(1.0 if did_restart else 0.0)
            x, f, g, p = x_new, f_new, g_new, p_new
            xs.append(x.copy())
            fs.append(f)
            gs.append(euclidean_norm(g))
    return _build_trace(
        xs,
        fs,
        gs,
        steps,
        alphas,
        obj,
        stop,
        extras=[("beta", betas), ("restart", restarts), ("directions", np.asarray(dirs, dtype=float))],
    )


def trust_region(
    obj: Any,
    x0: Any,
    *,
    delta0: float = 1.0,
    delta_max: float = 1e6,
    eta: float = 0.15,
    t1: float = 0.25,
    t2: float = 0.75,
    gamma1: float = 0.25,
    gamma2: float = 2.0,
    max_iter: int = 200,
    tol: float = 1e-10,
    tol_kind: str = "grad",
    cg_tol: float = 1e-12,
) -> Trace:
    r"""信赖域：每步解 :math:`\min_{\|p\|\le\Delta_k} m_k(p)`（Steihaug-CG），再按

    .. math::
        \rho_k=\frac{f(x_k)-f(x_k+p_k)}{m_k(0)-m_k(p_k)}

    决定接受与半径：

    * :math:`\rho_k<t_1` → :math:`\Delta_{k+1}=\gamma_1\Delta_k`（模型不可信，缩小）；
    * :math:`\rho_k>t_2` 且 :math:`\|p_k\|=\Delta_k` → :math:`\Delta_{k+1}=\min(\gamma_2\Delta_k,\Delta_{\max})`；
    * :math:`\rho_k>\eta` 才接受这一步（``eta`` 就是"接受阈值"）。

    **被拒绝的步也要记录**：``accepted`` 与 ``ratio`` 逐步可读，否则"半径在自适应"
    这句话在图上无从检验。负曲率由子问题求解器分支给出（``extras["neg_curv"]``），
    不是事后看 Hessian 的特征值。
    """
    if getattr(obj, "hess", None) is None:
        raise ValueError("信赖域需要 Hessian（obj.hess 为 None）")
    x = np.asarray(x0, dtype=float).ravel()
    delta = float(delta0)
    if not (delta > 0.0):
        raise ValueError(f"初始半径必须为正，得到 {delta0!r}")
    xs, fs, gs, steps, alphas = [x.copy()], [], [], [], []
    radius_hist: list[float] = [delta]
    ratios: list[float] = []
    accepted: list[bool] = []
    preds: list[float] = []
    acts: list[float] = []
    neg_curv: list[float] = []
    hit_b: list[float] = []
    g = np.asarray(obj.grad(x), dtype=float).ravel()
    f = float(obj.value(x))
    g0 = euclidean_norm(g)
    fs.append(f)
    gs.append(g0)
    stop = "max_iter"
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(int(max_iter)):
            reason = _stop_reason(x, f, g, steps[-1] if steps else float("inf"), tol=tol, tol_kind=tol_kind, g0=g0, f_prev=fs[-2] if len(fs) > 1 else None)
            if reason:
                stop = reason
                break
            H = np.asarray(obj.hess(x), dtype=float)
            sub = steihaug_cg(lambda v: H @ v, g, delta, tol=cg_tol)
            p = sub.p
            pred = float(sub.predicted)
            f_try = float(obj.value(x + p))
            actual = f - f_try
            rho = actual / pred if pred > 0.0 else float("-inf")
            preds.append(float(pred))
            acts.append(float(actual))
            neg_curv.append(1.0 if sub.negative_curvature else 0.0)
            hit_b.append(1.0 if sub.hit_boundary else 0.0)
            if np.isfinite(rho) and rho > float(eta) and actual > 0.0:
                steps.append(euclidean_norm(p))
                alphas.append(1.0)
                x, f = x + p, f_try
                g = np.asarray(obj.grad(x), dtype=float).ravel()
                xs.append(x.copy())
                fs.append(f)
                gs.append(euclidean_norm(g))
                accepted.append(True)
            else:
                accepted.append(False)
            ratios.append(float(rho))
            pnorm = euclidean_norm(p)
            if not np.isfinite(rho) or rho < float(t1):
                delta = float(gamma1) * delta
            elif rho > float(t2) and pnorm >= 0.99 * delta:
                delta = min(float(gamma2) * delta, float(delta_max))
            delta = max(delta, 1e-14)
            radius_hist.append(delta)
            if not accepted[-1] and delta <= 1e-13:
                stop = "radius_collapsed"
                break
    return _build_trace(
        xs,
        fs,
        gs,
        steps,
        alphas,
        obj,
        stop,
        radius=radius_hist,
        ratio=ratios,
        accepted=accepted,
        extras=[
            ("predicted", preds),
            ("actual", acts),
            ("neg_curv", neg_curv),
            ("hit_boundary", hit_b),
        ],
    )


# --------------------------------------------------------------------------- #
# 几何与鞍点
# --------------------------------------------------------------------------- #
def contour_grid(
    obj: Any,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    n: int = 81,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""等高线网格 :math:`(X,Y,Z)`，:math:`Z_{ij}=f(x_i,y_j)`（按行/列顺序与 meshgrid 一致）。"""
    if int(n) < 2:
        raise ValueError(f"网格点数必须 ≥ 2，得到 {n!r}")
    xs = np.linspace(float(xlim[0]), float(xlim[1]), int(n))
    ys = np.linspace(float(ylim[0]), float(ylim[1]), int(n))
    X, Y = np.meshgrid(xs, ys)
    Z = np.empty_like(X)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = float(obj.value(np.array([X[i, j], Y[i, j]], dtype=float)))
    return X, Y, Z


def convergence_order(err: Any, *, floor: float = 1e-13) -> float:
    r"""实测收敛阶 :math:`p=\ln(e_{k+1}/e_k)/\ln(e_k/e_{k-1})` 的**尾部**估计。

    用最后三个仍然大于 ``floor`` 且严格递减的误差点。线性收敛 :math:`p\to 1`、
    二次 :math:`p\to 2`；点数不够时返回 ``nan``（不编数）。
    """
    e = np.asarray(err, dtype=float).ravel()
    e = e[(e > float(floor)) & np.isfinite(e)]
    if e.size < 3:
        return float("nan")
    a, b, c = float(e[-3]), float(e[-2]), float(e[-1])
    if not (a > b > c > 0.0):
        return float("nan")
    denom = math.log(b / a)
    if denom == 0.0:
        return float("nan")
    return float(math.log(c / b) / denom)


def quadratic_window(
    dist: Any,
    *,
    factor: float = 1.5,
    spread: float = 3.0,
    min_order: float = 1.8,
    floor: float = 1e-13,
) -> tuple[int | None, float | None, np.ndarray]:
    r"""从误差序列里**实测**"进入二次区"的位置与二次收敛常数。

    判据（写进 ``theory`` 的就是这三条，不是事后挑窗口）：

    1. **尾部阶数够高**：:func:`convergence_order` 给出的实测阶 :math:`p\ge` ``min_order``。
       线性尾巴（无论每步 ×0.5 还是 ×0.9）的 :math:`p\approx 1`，这一条直接否掉它
       ——"卡住步长的牛顿法"就是被这一条挡在外面的。
    2. **常数不再漂移**：从序列末尾往前扫，相邻两个 :math:`q_k=e_{k+1}/e_k^2`
       的比值要落在 :math:`[1/\text{factor},\text{factor}]` 之内（``factor=1.5``：
       任何 :math:`e_{k+1}\le e_k/2` 的线性尾巴都会让 :math:`q` 每步翻倍，必被排除）。
    3. **整段散布有界**：窗口内 :math:`\max q/\min q\le` ``spread``。

    任一条不满足就返回 ``(None, None, ...)``：**常数没有被测出来**，
    不要用单点或漂移中的数冒充（Rosenbrock 上就是这种情况）。
    返回 ``(进入二次区的步号, 常数上界 C, 窗口内的 q 序列)``，
    其中 :math:`C=\max q`，于是 :math:`e_{k+1}\le Ce_k^2` 在窗口内逐步成立。
    """
    e = np.asarray(dist, dtype=float).ravel()
    q = np.full(max(e.size - 1, 0), np.nan)
    for k in range(e.size - 1):
        if e[k] > float(floor) and e[k + 1] > 0.0:
            q[k] = float(e[k + 1] / e[k] ** 2)
    idx = [int(k) for k in np.flatnonzero(np.isfinite(q))]
    if len(idx) < 2 or convergence_order(e, floor=floor) < float(min_order):
        return None, None, np.asarray([], dtype=float)
    start = len(idx) - 1
    while start - 1 >= 0:
        a, b = q[idx[start - 1]], q[idx[start]]
        if a > 0.0 and b > 0.0 and max(a / b, b / a) <= float(factor):
            start -= 1
        else:
            break
    window = np.asarray([q[k] for k in idx[start:]], dtype=float)
    if window.size < 2 or float(np.max(window) / np.min(window)) > float(spread):
        return None, None, window
    return int(idx[start]), float(np.max(window)), window


@dataclass(frozen=True)
class EscapeMeasurement:
    r"""沿给定方向逃出鞍点邻域所需的最小扰动幅度（以及它与负曲率方向的重合度）。

    ``escaped=False`` 表示"放大到 1e12 也没跑掉"——这时 ``threshold`` 是 ``inf``，
    帧里必须写 ``None`` 并在 ``strings`` 里说明（PROTOCOLS §1.3），不许写成 0。
    """

    threshold: float
    lo: float
    hi: float
    iterations: int
    final_distance: float
    escaped: bool
    direction_overlap: float


def escape_threshold(
    obj: Any,
    x: Any,
    direction: Any,
    *,
    method: str = "gd",
    step: float | None = None,
    max_iter: int = 400,
    radius: float = 1.0,
    hi0: float = 1e-6,
    max_expand: int = 60,
    bisect: int = 80,
    tol: float = 1e-9,
) -> EscapeMeasurement:
    r"""从 ``x`` 沿 ``direction`` 扰动 :math:`\delta` 后，需要多大的 :math:`\delta` 才"逃出"
    以 ``x`` 为心、``radius`` 为半径的邻域（二分实测，判据是**迭代终点的距离**）。

    为什么必须做成二分实测：直接报"我加了 1e-3 的随机扰动，跑掉了"是没有信息量的
    ——跑掉与否取决于扰动在**负曲率方向上**的投影，而这个投影可以任意小。
    :attr:`EscapeMeasurement.direction_overlap` 给出 :math:`|\langle u_{\min},d\rangle|`，
    它与阈值的乘积近似为常数，这才是"随机扰动有效"的充要条件。
    找不到逃逸（例如扰动方向与负曲率方向严格正交）时 ``escaped=False``、
    ``threshold=inf``，调用方据此写 ``None`` + 说明。
    """
    if method not in ("gd", "newton"):
        raise ValueError(f"method 只能是 gd / newton，得到 {method!r}")
    xv = np.asarray(x, dtype=float).ravel()
    d = np.asarray(direction, dtype=float).ravel()
    nd = euclidean_norm(d)
    if nd == 0.0:
        raise ValueError("扰动方向不能是零向量")
    u = d / nd

    def escaped_at(delta: float) -> bool:
        x0 = xv + float(delta) * u
        if method == "gd":
            try:
                tr = gd(obj, x0, step=step, max_iter=max_iter, tol=1e-300, tol_kind="grad")
            except (ValueError, ZeroDivisionError):
                return False
        else:
            try:
                tr = newton(obj, x0, line_search="none", max_iter=max_iter, tol=1e-300)
            except (ValueError, ZeroDivisionError):
                return False
        return euclidean_norm(tr.x_final - xv) >= float(radius)

    lo, hi = 0.0, float(hi0)
    if escaped_at(lo):
        return EscapeMeasurement(0.0, 0.0, 0.0, 0, 0.0, True, 0.0)
    found = False
    for _ in range(int(max_expand)):
        if escaped_at(hi):
            found = True
            break
        lo, hi = hi, 2.0 * hi
        if hi > 1e12:
            break
    if not found:
        return EscapeMeasurement(float("inf"), lo, hi, 0, 0.0, False, 0.0)
    it = 0
    for _ in range(int(bisect)):
        mid = 0.5 * (lo + hi)
        if escaped_at(mid):
            hi = mid
        else:
            lo = mid
        it += 1
        if hi - lo <= float(tol) * max(abs(hi), abs(lo), 1e-300):
            break
    thr = 0.5 * (lo + hi)
    tr = gd(obj, xv + thr * u, step=step, max_iter=max_iter, tol=1e-300, tol_kind="grad") if method == "gd" else newton(
        obj, xv + thr * u, line_search="none", max_iter=max_iter, tol=1e-300
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        u_neg = negative_curvature_direction(obj.hess(xv))
    return EscapeMeasurement(
        threshold=float(thr),
        lo=float(lo),
        hi=float(hi),
        iterations=int(it),
        final_distance=euclidean_norm(tr.x_final - xv),
        escaped=True,
        direction_overlap=abs(inner(u_neg, u)),
    )
