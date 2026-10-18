r"""求解与误差：自研 Cholesky/LU 两条路线、各种范数下的误差、Lax--Milgram 常数。

数学定义
--------
离散问题 :math:`K u=F` 里 :math:`K` 对称正定，所以理论上的解法是 Cholesky；
本模块**先试 Cholesky、失败再退到部分主元 LU**，并把"走了哪条路、最小主元是多少"
如实返回——因为本章有整整一类问题（用 :math:`P_1` 解四阶梁方程）的 :math:`K` 是**奇异**的，
那时需要的不是"解出来"，而是"说清楚它为什么解不出来"。

误差范数不是一个数，而是一族。对同一个 :math:`P_1` 解：

.. math::
    \|u-u_h\|_{L^2}=O(h^2),\qquad
    |u-u_h|_{H^1}=O(h),\qquad
    \|u-u_h\|_a=\sqrt{a(u-u_h,u-u_h)}=O(h).

"收敛阶是几"这个问题**必须先说清是哪个范数**，否则两条曲线画在一起时
会让人以为其中一个算错了。本模块把 :math:`L^2` / :math:`H^1` 半范 / 能量范数 /
（Hermite 空间的）:math:`H^2` 半范并列给出，用同一批 Gauss 求积点算，
保证它们的差别只来自被积函数、不来自实现。

Lax--Milgram 与 Céa 的常数是**可测的**：离散强制性与连续性常数就是一对广义特征值

.. math::
    \alpha_h=\min_{v_h\ne0}\frac{a(v_h,v_h)}{\|v_h\|_{H^1}^2},\qquad
    M_h=\max_{v_h\ne0}\frac{a(v_h,v_h)}{\|v_h\|_{H^1}^2},

于是 Céa 引理 :math:`\|u-u_h\|_a\le\sqrt{M_h/\alpha_h}\inf_{v_h}\|u-v_h\|_a`
前面的常数就是 :math:`\sqrt{M_h/\alpha_h}`。系数反差大时它会大到让误差界失去分辨力——
章节里给的是实测的 :math:`\alpha_h,M_h`、采样得到的解析上下界、以及真正测到的
Galerkin 效率比（:math:`\|u-u_h\|_a/\|u-I_hu\|_a`）三个数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from numcore.linalg.chol import chol_solve, cholesky
from numcore.linalg.eigen import hermitian_jacobi_eigh
from numcore.linalg.lu import lu_partial, solve_lu
from numcore.pde.fem.assemble import (
    Assembly,
    eval_field,
    assemble_1d,
    element_mass_1d,
    element_stiffness_1d,
    symmetry_defect,
)
from numcore.pde.fem.mesh import Mesh2D, Space1D
from numcore.pde.fem.shape import (
    gauss_legendre,
    hermite_second_derivs,
    hermite_values,
    lagrange_values,
    lagrange_values_and_derivs,
    quad_jacobian,
    quad_jacobian_det,
    quad_q1_values,
    quad_rule,
    tri_jacobian,
    tri_p1_values,
    tri_rule,
)

__all__ = [
    "FemSolution",
    "spd_solve",
    "system_condition",
    "generalized_extremes",
    "lax_milgram_constants",
    "solve_1d",
    "interpolant_1d",
    "error_norms_1d",
    "energy_norm_1d",
    "approximation_error_1d",
    "exact_values_1d",
    "error_norms_2d",
    "Model1D",
    "Model2D",
    "model_1d",
    "model_2d",
    "solve_model",
    "solve_model_2d",
]

_EIG_MAX = 220
"""超过这个自由度就不再跑 Jacobi 特征分解（:math:`O(n^3)` 的 Python 循环会拖垮单帧）。"""

_NQ_ERROR = 10
r"""误差积分的**最少**一维 Gauss 点数；实际点数还会随空间阶数增长（见 :func:`_err_order`）。

被积函数是 :math:`(u-u_h)^2` 或 :math:`(u'-u_h')^2`：:math:`u_h` 是 :math:`p` 次多项式，
所以最低要求是 :math:`n\ge p+1`。10 点对 :math:`P_1/P_2` 绰绰有余，对 :math:`p=12` 不够——
"误差算得准不准"和"解算得准不准"是两件事，这里必须一起管。
"""


def _err_order(space: Space1D) -> int:
    """误差积分点数：随空间阶数增长（:math:`p+4` 点给被积函数留足余量）。"""
    return max(_NQ_ERROR, int(space.degree) + 4)


@dataclass(frozen=True)
class FemSolution:
    r"""一次有限元求解的全部可观测物。

    ``method`` 是真正走通的那条路线（``"cholesky"`` / ``"lu"`` / ``"none"``）；
    ``ok=False`` 时 :math:`K` 奇异或非正定，``u`` 是零向量而 ``fail_reason`` 说明原因——
    帧里就是把这件事写成"可显示的状态"，而不是让它抛出去或者拿 NaN 冒充。
    """

    u: np.ndarray
    free: np.ndarray
    fixed: np.ndarray
    fixed_values: np.ndarray
    ok: bool
    method: str
    n_dof: int
    n_free: int
    residual: float
    symmetry_defect: float
    chol_min_pivot_sq: float
    chol_fail_at: int
    energy: float
    fail_reason: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "method": self.method,
            "n_dof": self.n_dof,
            "n_free": self.n_free,
            "residual": self.residual,
            "energy": self.energy,
        }


def spd_solve(K: Any, F: Any) -> dict[str, Any]:
    r"""解 :math:`Kx=F`：先 Cholesky，失败退到部分主元 LU。

    返回 ``{"x", "ok", "method", "min_pivot_sq", "fail_at", "residual"}``。
    LU 也失败（零主元）时 ``ok=False``、``x=0``、``method="none"``——
    这正是"用 :math:`P_1` 解梁方程"会发生的事，必须能如实报出来。
    """
    A = np.asarray(K, dtype=float)
    b = np.asarray(F, dtype=float).reshape(-1)
    if A.shape[0] == 0:
        # 零自由度不是错误：单个 P1 单元 + 两端 Dirichlet 时，解就是边界值本身。
        # 这一步必须显式处理——否则"最小网格"这个参数角点会抛 ValueError。
        return {
            "x": np.zeros(0, dtype=float),
            "ok": True,
            "method": "trivial",
            "min_pivot_sq": float("inf"),
            "max_pivot_sq": 0.0,
            "fail_at": -1,
            "residual": 0.0,
        }
    res = cholesky(A)
    if res.ok:
        x = chol_solve(res, b)
        return {
            "x": x,
            "ok": True,
            "method": "cholesky",
            "min_pivot_sq": float(res.min_pivot_sq),
            "max_pivot_sq": float(res.max_pivot_sq),
            "fail_at": -1,
            "residual": _residual(A, x, b),
        }
    try:
        lu = lu_partial(A)
        x = solve_lu(lu, b)
        return {
            "x": x,
            "ok": True,
            "method": "lu",
            "min_pivot_sq": float(res.min_pivot_sq),
            "max_pivot_sq": float(res.max_pivot_sq),
            "fail_at": int(res.fail_at),
            "residual": _residual(A, x, b),
        }
    except (ZeroDivisionError, ValueError):
        n = A.shape[0]
        return {
            "x": np.zeros(n, dtype=float),
            "ok": False,
            "method": "none",
            "min_pivot_sq": float(res.min_pivot_sq),
            "max_pivot_sq": float(res.max_pivot_sq),
            "fail_at": int(res.fail_at),
            "residual": float("inf"),
        }


def _residual(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    r"""回代残差 :math:`\max_i|(Ax-b)_i|`（绝对量；章节里同时给 :math:`\|A\|_\infty`）。"""
    r = A @ x - b
    return float(np.max(np.abs(r))) if r.size else 0.0


def system_condition(A: Any, *, with_eigen: bool = True) -> dict[str, Any]:
    r"""对称矩阵的条件数 :math:`\kappa_2=\lambda_{\max}/\lambda_{\min}` 与 Cholesky 主元比。

    两条路线都给：Jacobi 特征分解是**真值**，Cholesky 的
    :math:`(\max s_i/\min s_i)^2` 只是**估计**（它只反映主元序列的跨度）。
    章节里拿它们对照，是为了说明"生成一个漂亮的 Cholesky 不等于矩阵良态"。
    """
    A2 = np.asarray(A, dtype=float)
    n = A2.shape[0]
    res = cholesky(A2)
    est = float(res.max_pivot_sq) / float(res.min_pivot_sq) if res.ok and res.min_pivot_sq > 0.0 else float("inf")
    out: dict[str, Any] = {
        "n": n,
        "cholesky_ok": bool(res.ok),
        "pivot_ratio": est,
        "min_pivot_sq": float(res.min_pivot_sq),
        "max_pivot_sq": float(res.max_pivot_sq),
    }
    if with_eigen and 0 < n <= _EIG_MAX:
        w, _ = _jacobi_eigh_guarded(0.5 * (A2 + A2.T))
        lam = np.real(w)
        lo = float(np.min(lam))
        hi = float(np.max(lam))
        out["lambda_min"] = lo
        out["lambda_max"] = hi
        out["condition"] = hi / lo if lo > 0.0 else float("inf")
    else:
        out["lambda_min"] = None
        out["lambda_max"] = None
        out["condition"] = None
    return out


def _jacobi_eigh_guarded(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""调用 1.2 章的 Jacobi 特征分解，并屏蔽一个**良性**的中间溢出警告。

    扫描后期某些非对角元会变成次正规数，此时 ``tau = gap/(2|a_pq|)`` 的平方会溢出成
    ``inf``——但该步的旋转角本来就是 0，结果不受影响（:func:`generalized_extremes` 的返回值
    与 Cholesky 主元比、与解析上下界都对得上）。``pyproject`` 把 ``RuntimeWarning`` 当错误，
    所以这里必须显式说明"我知道它会溢出，且它无害"，而不是让它悄悄变成一条失败。
    """
    with np.errstate(over="ignore", invalid="ignore"):
        w, v = hermitian_jacobi_eigh(A)
    return w, v


def generalized_extremes(A: Any, B: Any) -> tuple[float, float]:
    r"""广义特征值问题 :math:`Av=\lambda Bv` 的最小、最大特征值（:math:`B` 对称正定）。

    做法是标准的"先 Cholesky 再对称化"：:math:`B=LL^{\mathsf T}`，
    :math:`C=L^{-1}AL^{-\mathsf T}` 对称，其特征值就是原问题的特征值。
    这里**不**用 :math:`B^{-1}A`（那会破坏对称性，也把一个好问题变成非对称问题）。
    """
    Am = np.asarray(A, dtype=float)
    Bm = np.asarray(B, dtype=float)
    n = Am.shape[0]
    res = cholesky(Bm)
    if not res.ok:
        raise ValueError(f"B 不是正定矩阵（Cholesky 在第 {res.fail_at} 个主元失败），广义特征值无定义")
    L = res.L
    Y = np.array(Am, dtype=float, copy=True)
    for i in range(n):
        if i:
            Y[i] = Y[i] - L[i, :i] @ Y[:i]
        Y[i] = Y[i] / L[i, i]
    Z = Y.T.copy()
    for i in range(n):
        if i:
            Z[i] = Z[i] - L[i, :i] @ Z[:i]
        Z[i] = Z[i] / L[i, i]
    C = 0.5 * (Z + Z.T)
    w, _ = _jacobi_eigh_guarded(C)
    lam = np.real(w)
    return float(np.min(lam)), float(np.max(lam))


def lax_milgram_constants(
    space: Space1D,
    a_field: Callable[[np.ndarray], Any] | None = None,
    c_field: Callable[[np.ndarray], Any] | None = None,
    free: Any = None,
    *,
    a_bounds: tuple[float, float] | None = None,
    c_bounds: tuple[float, float] | None = None,
    n_sample: int = 400,
) -> dict[str, Any]:
    r"""离散双线性形式的强制性与连续性常数（在**单位系数** :math:`H^1` 范数下）。

    :math:`B=K_1+M_1`（:math:`a=c=1` 的刚度阵与质量阵）定义范数
    :math:`\|v\|_{H^1}^2=v^{\mathsf T}Bv`，:math:`A=K_a+M_c` 是真实双线性形式的矩阵，
    于是 :math:`\alpha_h,\ M_h` 是 :math:`(A,B)` 的广义特征值端点。

    解析上下界 :math:`(\min a,\max a,\min c,\max c)` 默认在网格上**采样**得到——
    采样会漏掉比采样点更极端的地方，所以调用方知道真值时必须显式传 ``a_bounds``/``c_bounds``。
    返回的 ``cea_factor`` 就是 Céa 引理前面那个 :math:`\sqrt{M/\alpha}`。
    """
    k1 = element_stiffness_1d(space, None)
    m1 = element_mass_1d(space, None)
    ka = element_stiffness_1d(space, a_field)
    # 反应项的缺省必须是 0（"没有 c 项"），不是 1（"L2 内积"）：写成 1 会让
    # alpha_h 恰好等于 min(a,c) 里那个"有支撑的地方"，实测直接把 alpha_h 顶成 1.0000。
    mc = element_mass_1d(space, c_field if c_field is not None else 0.0)
    A1 = assemble_1d(space, ka, np.zeros((space.n_elem, space.npe))).K + assemble_1d(
        space, mc, np.zeros((space.n_elem, space.npe))
    ).K
    B1 = assemble_1d(space, k1, np.zeros((space.n_elem, space.npe))).K + assemble_1d(
        space, m1, np.zeros((space.n_elem, space.npe))
    ).K
    if free is not None:
        idx = np.asarray(free, dtype=np.int64).reshape(-1)
        A1 = A1[np.ix_(idx, idx)]
        B1 = B1[np.ix_(idx, idx)]
    alpha_h, m_h = generalized_extremes(A1, B1)
    if a_bounds is None or c_bounds is None:
        xg = np.linspace(0.0, 1.0, int(n_sample))
        a_s = eval_field(a_field, xg)
        c_s = eval_field(c_field, xg, default=0.0)
        a_bounds = (float(np.min(a_s)), float(np.max(a_s))) if a_bounds is None else a_bounds
        c_bounds = (float(np.min(c_s)), float(np.max(c_s))) if c_bounds is None else c_bounds
    a_lo, a_hi = float(a_bounds[0]), float(a_bounds[1])
    c_lo, c_hi = float(c_bounds[0]), float(c_bounds[1])
    ana_alpha = min(a_lo, c_lo)
    ana_m = max(a_hi, c_hi)
    return {
        "alpha_h": float(alpha_h),
        "M_h": float(m_h),
        "cea_factor": float(np.sqrt(m_h / alpha_h)) if alpha_h > 0.0 else float("inf"),
        "analytic_alpha": float(ana_alpha),
        "analytic_M": float(ana_m),
        "analytic_cea_factor": float(np.sqrt(ana_m / ana_alpha)) if ana_alpha > 0.0 else float("inf"),
        "a_bounds": (a_lo, a_hi),
        "c_bounds": (c_lo, c_hi),
        "kappa_a": float(a_hi / a_lo) if a_lo > 0.0 else float("inf"),
    }


# --------------------------------------------------------------------------- #
# 求解（带边界条件）
# --------------------------------------------------------------------------- #
def solve_1d(
    space: Space1D,
    ke: np.ndarray,
    fe: np.ndarray,
    fixed: Any | None = None,
    fixed_values: Any | None = None,
    *,
    energy_a: Callable[[np.ndarray], Any] | None = None,
    energy_c: Callable[[np.ndarray], Any] | None = None,
) -> tuple[Assembly, FemSolution]:
    r"""装配 + 加 Dirichlet 条件 + 求解，返回 ``(装配结果, 解)``。

    ``fixed=None`` 用 :meth:`Space1D.boundary_dofs` 的缺省（一维 Dirichlet 两端为零）。
    """
    asm = assemble_1d(space, ke, fe)
    if fixed is None:
        fixed = space.boundary_dofs("dirichlet")
    fx = np.asarray(fixed, dtype=np.int64).reshape(-1)
    if fixed_values is None:
        g = np.zeros(fx.size, dtype=float)
    else:
        g = np.asarray(fixed_values, dtype=float).reshape(-1)
    n = asm.K.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[fx] = False
    free = np.flatnonzero(mask)
    Kff = asm.K[np.ix_(free, free)]
    if fx.size:
        Ff = asm.F[free] - asm.K[np.ix_(free, fx)] @ g
    else:
        Ff = asm.F[free].copy()
    out = spd_solve(Kff, Ff)
    u = np.zeros(n, dtype=float)
    if fx.size:
        u[fx] = g
    u[free] = out["x"]
    energy = energy_norm_1d(space, u, energy_a, energy_c) if out["ok"] else float("inf")
    reason = ""
    if not out["ok"]:
        reason = (
            f"刚度阵奇异：Cholesky 在第 {out['fail_at']} 个主元失败"
            f"（最小主元 {out['min_pivot_sq']:.3e}），部分主元 LU 也出现零主元"
        )
    sol = FemSolution(
        u=u,
        free=free,
        fixed=fx,
        fixed_values=g,
        ok=bool(out["ok"]),
        method=str(out["method"]),
        n_dof=n,
        n_free=int(free.size),
        residual=float(out["residual"]) if np.isfinite(out["residual"]) else float("inf"),
        symmetry_defect=symmetry_defect(asm.K),
        chol_min_pivot_sq=float(out["min_pivot_sq"]),
        chol_fail_at=int(out["fail_at"]),
        energy=float(energy),
        fail_reason=reason,
    )
    return asm, sol


# --------------------------------------------------------------------------- #
# 一维误差范数
# --------------------------------------------------------------------------- #
def exact_values_1d(space: Space1D, exact: Callable[[np.ndarray], Any],
                    dexact: Callable[[np.ndarray], Any] | None = None) -> np.ndarray:
    r"""把解析解变成自由度向量：拉格朗日空间取节点值，Hermite 空间取 :math:`(w,w')`。"""
    if space.is_hermite:
        if dexact is None:
            raise ValueError("Hermite 空间需要解析解的一阶导（自由度里有 w'）")
        vals = np.empty(space.n_dof, dtype=float)
        vals[0::2] = np.asarray(exact(space.mesh.x), dtype=float)
        vals[1::2] = np.asarray(dexact(space.mesh.x), dtype=float)
        return vals
    return np.asarray(exact(space.nodes), dtype=float)


def interpolant_1d(space: Space1D, exact: Callable[[np.ndarray], Any],
                   dexact: Callable[[np.ndarray], Any] | None = None) -> np.ndarray:
    r"""节点插值 :math:`I_hu\in V_h`（Céa 引理里那个"最佳逼近"的参照物之一）。

    注意 :math:`I_hu` 一般**不是**能量范数下的最佳逼近——真正的最佳逼近就是有限元解
    :math:`u_h` 自己（Galerkin 正交性）。章节里用
    :math:`\|u-I_hu\|_a` 当"最佳逼近误差"的上界参照，并实测
    :math:`\|u-u_h\|_a\le\|u-I_hu\|_a` 以及两者的比值。
    """
    return exact_values_1d(space, exact, dexact)


def _eval_uh_1d(space: Space1D, u: np.ndarray, e: int, xq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """单元 ``e`` 上求 :math:`(u_h,u_h',u_h'')`（Hermite 才有第二项非零）。"""
    ids = space.elems[e]
    if space.is_hermite:
        h = float(space.mesh.x[space.mesh.elems[e, 1]] - space.mesh.x[space.mesh.elems[e, 0]])
        xi = (xq - float(space.mesh.x[space.mesh.elems[e, 0]])) / h
        V = hermite_values(xi, h)
        D, D2 = hermite_second_derivs(xi, h)
        ul = u[ids]
        return V.T @ ul, D.T @ ul, D2.T @ ul
    V, D = lagrange_values_and_derivs(space.nodes[ids], xq)
    ul = u[ids]
    return V.T @ ul, D.T @ ul, np.zeros_like(xq)


def error_norms_1d(
    space: Space1D,
    u: np.ndarray,
    exact: Callable[[np.ndarray], Any],
    dexact: Callable[[np.ndarray], Any] | None = None,
    *,
    a_field: Callable[[np.ndarray], Any] | None = None,
    c_field: Callable[[np.ndarray], Any] | None = None,
    want_h2: bool = False,
    d2exact: Callable[[np.ndarray], Any] | None = None,
) -> dict[str, float]:
    r"""误差的四个范数（同一批 Gauss 点，差别只来自被积函数）：

    ``l2`` :math:`=\|u-u_h\|_{L^2}`、``h1`` :math:`=|u-u_h|_{H^1}`、
    ``energy`` :math:`=\sqrt{a(e,e)}`、``h2`` :math:`=|u-u_h|_{H^2}`（仅 Hermite，
    拉格朗日空间的 :math:`u_h''` 在单元内恒为零，这个量对非多项式解析解是无穷）。
    """
    xs, ws = gauss_legendre(_err_order(space))
    l2 = 0.0
    h1 = 0.0
    en = 0.0
    h2 = 0.0
    for e in range(space.n_elem):
        xa = float(space.mesh.x[space.mesh.elems[e, 0]])
        xb = float(space.mesh.x[space.mesh.elems[e, 1]])
        h = xb - xa
        xq = xa + 0.5 * h * (xs + 1.0)
        wq = 0.5 * h * ws
        uh, duh, d2uh = _eval_uh_1d(space, u, e, xq)
        ue = np.asarray(exact(xq), dtype=float)
        due = np.asarray(dexact(xq), dtype=float) if dexact is not None else _central_diff(exact, xq)
        de = ue - uh
        dd = due - duh
        l2 += float(np.sum(wq * de * de))
        h1 += float(np.sum(wq * dd * dd))
        # a_field=None 的默认是 a≡1，c_field=None 的默认是 c≡0（与 element_bilinear_1d
        # 和模型的 c_bounds=(0,0) 一致）。这里曾经在"两个都 None"时改成 dd²+de²，
        # 于是量出来的"能量范数"其实是 H¹ **范数**——比能量半范大 0.08%（P1 光滑解），
        # 而且让 √(eᵀKe) 与它对不上。范数的定义只能有一个。
        aq = eval_field(a_field, xq)
        cq = eval_field(c_field, xq, default=0.0)
        en += float(np.sum(wq * (aq * dd * dd + cq * de * de)))
        if want_h2:
            if not space.is_hermite:
                raise ValueError("H2 半范只对 Hermite 空间有定义（拉格朗日单元内 u_h'' 恒为零）")
            d2e = np.asarray(d2exact(xq), dtype=float) if d2exact is not None else _central_diff2(exact, xq)
            h2 += float(np.sum(wq * (d2e - d2uh) ** 2))
    out = {
        "l2": float(np.sqrt(max(l2, 0.0))),
        "h1": float(np.sqrt(max(h1, 0.0))),
        "energy": float(np.sqrt(max(en, 0.0))),
    }
    if want_h2:
        out["h2"] = float(np.sqrt(max(h2, 0.0)))
    return out


def _central_diff(f: Callable[[np.ndarray], Any], x: np.ndarray) -> np.ndarray:
    r"""自研五点中心差分 :math:`f'(x)\approx\frac{f(x-2s)-8f(x-s)+8f(x+s)-f(x+2s)}{12s}`。

    ``np.gradient`` 是 P3 的禁用属性；传进来的解析导数优先，这个差分只是
    "调用方没给导数"时的兜底，精度 :math:`O(s^4)`，步长 :math:`s=10^{-3}`。
    """
    s = 1e-3
    x = np.asarray(x, dtype=float)
    return (
        np.asarray(f(x - 2.0 * s))
        - 8.0 * np.asarray(f(x - s))
        + 8.0 * np.asarray(f(x + s))
        - np.asarray(f(x + 2.0 * s))
    ) / (12.0 * s)


def _central_diff2(f: Callable[[np.ndarray], Any], x: np.ndarray) -> np.ndarray:
    r"""自研五点二阶中心差分 :math:`f''(x)\approx\frac{-f(x-2s)+16f(x-s)-30f(x)+16f(x+s)-f(x+2s)}{12s^2}`。"""
    s = 1e-3
    x = np.asarray(x, dtype=float)
    return (
        -np.asarray(f(x - 2.0 * s))
        + 16.0 * np.asarray(f(x - s))
        - 30.0 * np.asarray(f(x))
        + 16.0 * np.asarray(f(x + s))
        - np.asarray(f(x + 2.0 * s))
    ) / (12.0 * s * s)


def energy_norm_1d(
    space: Space1D,
    u: np.ndarray,
    a_field: Callable[[np.ndarray], Any] | None = None,
    c_field: Callable[[np.ndarray], Any] | None = None,
) -> float:
    r"""有限元函数自己的能量范数 :math:`\sqrt{a(u_h,u_h)}`（用同一套求积点）。"""
    xs, ws = gauss_legendre(_err_order(space))
    acc = 0.0
    for e in range(space.n_elem):
        xa = float(space.mesh.x[space.mesh.elems[e, 0]])
        xb = float(space.mesh.x[space.mesh.elems[e, 1]])
        h = xb - xa
        xq = xa + 0.5 * h * (xs + 1.0)
        wq = 0.5 * h * ws
        uh, duh, _ = _eval_uh_1d(space, u, e, xq)
        aq = eval_field(a_field, xq)
        cq = eval_field(c_field, xq, default=0.0)
        # 同上：默认 a≡1、c≡0 ⟹ 能量范数就是 H¹ 半范 |u_h|，不是 H¹ 范数。
        acc += float(np.sum(wq * (aq * duh * duh + cq * uh * uh)))
    return float(np.sqrt(max(acc, 0.0)))


def approximation_error_1d(
    space: Space1D,
    exact: Callable[[np.ndarray], Any],
    dexact: Callable[[np.ndarray], Any] | None = None,
    *,
    a_field: Callable[[np.ndarray], Any] | None = None,
    c_field: Callable[[np.ndarray], Any] | None = None,
) -> dict[str, float]:
    r"""插值误差 :math:`\|u-I_hu\|`（与 :func:`error_norms_1d` 同一套口径）。"""
    ih = interpolant_1d(space, exact, dexact)
    return error_norms_1d(space, ih, exact, dexact, a_field=a_field, c_field=c_field)


# --------------------------------------------------------------------------- #
# 二维误差范数
# --------------------------------------------------------------------------- #
def error_norms_2d(
    mesh: Mesh2D,
    u: np.ndarray,
    exact: Callable[[np.ndarray], Any],
    dexact: Callable[[np.ndarray], Any] | None = None,
    *,
    a_field: Callable[[np.ndarray], Any] | None = None,
) -> dict[str, float]:
    r"""二维 :math:`L^2` / :math:`H^1` 半范 / 能量范数误差（:math:`P_1` 三角形与 :math:`Q_1` 四边形）。

    :math:`Q_1` 的梯度在**每个求积点上**用该点的 :math:`J^{-1}` 变换——
    这与 :mod:`~numcore.pde.fem.assemble` 里"冻结 J"的错误路线形成对照。
    """
    l2 = 0.0
    h1 = 0.0
    en = 0.0
    for e in range(mesh.n_elem):
        corners = mesh.nodes[mesh.elems[e]]
        if mesh.kind == "tri":
            xi, eta, w = tri_rule(3)
            N, dxi, deta = tri_p1_values(xi, eta)
            J = tri_jacobian(corners)
            det = float(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
            inv = np.asarray([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]], dtype=float) / det
            dN = np.stack([dxi, deta], axis=0)
            grad = np.einsum("ba,biq->aiq", inv, dN)
            pts = N.T @ corners
            wdet = w * abs(det)
        else:
            xi, eta, w = quad_rule(3)
            N, dxi, deta = quad_q1_values(xi, eta)
            J = quad_jacobian(xi, eta, corners)
            det = quad_jacobian_det(xi, eta, corners)
            inv = np.empty_like(J)
            inv[:, 0, 0] = J[:, 1, 1] / det
            inv[:, 1, 1] = J[:, 0, 0] / det
            inv[:, 0, 1] = -J[:, 0, 1] / det
            inv[:, 1, 0] = -J[:, 1, 0] / det
            dN = np.stack([dxi, deta], axis=0)
            grad = np.einsum("qba,biq->aiq", inv, dN)
            pts = N.T @ corners
            wdet = w * np.abs(det)
        ul = u[mesh.elems[e]]
        uh = N.T @ ul
        guh = np.einsum("aiq,i->qa", grad, ul)
        ue = np.asarray(exact(pts), dtype=float)
        if dexact is not None:
            gex = np.asarray(dexact(pts), dtype=float)
        else:
            gex = np.zeros((pts.shape[0], 2), dtype=float)
        de = ue - uh
        gd = gex - guh
        l2 += float(np.sum(wdet * de * de))
        h1 += float(np.sum(wdet * np.sum(gd * gd, axis=1)))
        if a_field is None:
            en += float(np.sum(wdet * (np.sum(gd * gd, axis=1) + de * de)))
        else:
            aq = eval_field(a_field, pts)
            en += float(np.sum(wdet * (aq * np.sum(gd * gd, axis=1) + de * de)))
    return {
        "l2": float(np.sqrt(max(l2, 0.0))),
        "h1": float(np.sqrt(max(h1, 0.0))),
        "energy": float(np.sqrt(max(en, 0.0))),
    }


# --------------------------------------------------------------------------- #
# 标准算例：解析解、载荷与边界（帧与断言共用同一份真源）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Model1D:
    r"""一维标准算例：解析解 + 载荷 + 系数 + 边界，全部闭式给出。

    为什么把它放进内核而不是散在测试与帧里：**解析解只能有一份**。
    测试与前端帧如果各写一份 :math:`u`，两者永远"互相验证通过"却一起错。
    这里给出的每个算例都满足 :math:`-(au')'+cu=f`（弱形式意义下），
    并在 :mod:`tests/4.4` 里逐点对拍过。

    ==================  ==================================================================
    名字                 内容 / 为什么需要它
    ==================  ==================================================================
    ``smooth``          :math:`u=\sin\pi x`：光滑解，:math:`P_1` 的 :math:`L^2` 阶 2、能量阶 1
    ``interface``       :math:`a` 在两个区间上差 100 倍、:math:`f=0`：解是分段线性、
                        在界面处**折一下**（:math:`C^0` 但不 :math:`C^1`），
                        一致加密只有 :math:`O(h^{1/2})`，自适应加密能扳回几何收敛
    ``point``           :math:`-u''=\delta(x-x_0)`：解是 Green 函数（分段线性），
                        载荷落在节点上时 :math:`P_1` 解**节点精确**
    ``beam_point``      Hermite 三次梁（简支 + 集中力）：四阶问题的 :math:`C^1` 空间
    ``beam_uniform``    Hermite 三次梁（简支 + 均布载荷）
    ==================  ==================================================================
    """

    name: str
    exact: Callable[[Any], Any]
    dexact: Callable[[Any], Any] | None
    d2exact: Callable[[Any], Any] | None
    load: Callable[[Any], Any] | None
    a_field: Callable[[Any], Any] | None
    c_field: Callable[[Any], Any] | None
    a_bounds: tuple[float, float]
    c_bounds: tuple[float, float]
    bc_kind: str
    bc_values: tuple[float, float]
    breakpoints: tuple[float, ...] = ()
    point_load: tuple[float, float] | None = None
    hermite: bool = False
    note: str = ""

    @property
    def has_point_load(self) -> bool:
        return self.point_load is not None


def model_1d(name: str, **kw: Any) -> Model1D:
    r"""按名字构造标准算例（``smooth`` / ``interface`` / ``point`` / ``beam_point`` / ``beam_uniform``）。

    可调参数用关键字给出：``interface`` 的 ``xs``（界面位置，默认 :math:`1/\pi`，
    刻意取无理数——一致网格永远不会恰好把节点放在界面上）、``ratio``（系数反差，默认 100）、
    ``point`` 的 ``x0``、``beam_*`` 的 ``x0`` 与 ``P``/``q``。
    """
    pi = float(np.pi)
    if name == "smooth":
        c = float(kw.get("c", 0.0))

        def c_smooth(x: Any) -> np.ndarray:
            return np.full(np.asarray(x, dtype=float).shape, c)

        return Model1D(
            name=name,
            exact=lambda x: np.sin(pi * np.asarray(x, dtype=float)),
            dexact=lambda x: pi * np.cos(pi * np.asarray(x, dtype=float)),
            d2exact=lambda x: -pi * pi * np.sin(pi * np.asarray(x, dtype=float)),
            load=lambda x: (pi * pi + c) * np.sin(pi * np.asarray(x, dtype=float)),
            a_field=None,
            c_field=(c_smooth if c != 0.0 else None),
            a_bounds=(1.0, 1.0),
            c_bounds=(c, c),
            bc_kind="dirichlet",
            bc_values=(0.0, 0.0),
            note=f"u=sin(pi x)，-u''+{c:g}u=({pi * pi + c:.6g})sin(pi x)：光滑解基准"
                 + ("（c=1 时双线性形式就是 H1 内积，Cea 常数恰好是 1）" if c == 1.0 else ""),
        )
    if name == "interface":
        xs = float(kw.get("xs", 1.0 / pi))
        ratio = float(kw.get("ratio", 100.0))
        c_int = float(kw.get("c", 0.0))

        def a_field(x: Any) -> np.ndarray:
            xx = np.asarray(x, dtype=float)
            return np.where(xx < xs, 1.0, ratio)

        cl = 1.0 / (xs / 1.0 + (1.0 - xs) / ratio)

        def exact(x: Any) -> np.ndarray:
            xx = np.asarray(x, dtype=float)
            return np.where(xx < xs, cl * xx, cl * xs + (cl / ratio) * (xx - xs))

        def dexact(x: Any) -> np.ndarray:
            xx = np.asarray(x, dtype=float)
            return np.where(xx < xs, cl, cl / ratio)

        return Model1D(
            name=name,
            exact=exact,
            dexact=dexact,
            d2exact=lambda x: np.zeros_like(np.asarray(x, dtype=float)),
            load=lambda x: c_int * exact(x),
            a_field=a_field,
            c_field=(lambda x: np.full(np.asarray(x, dtype=float).shape, c_int)) if c_int != 0.0 else None,
            a_bounds=(1.0, ratio),
            c_bounds=(c_int, c_int),
            bc_kind="dirichlet",
            bc_values=(0.0, 1.0),
            breakpoints=(xs,),
            note=(f"a=1 (x<{xs:.6f}) / {ratio:g} (x>{xs:.6f})，f={c_int:g}u，u(0)=0,u(1)=1："
                  "解分段线性，界面处 C0 不 C1"),
        )
    if name == "point":
        x0 = float(kw.get("x0", 0.5))
        P = float(kw.get("P", 1.0))

        def exact(x: Any) -> np.ndarray:
            xx = np.asarray(x, dtype=float)
            return P * np.minimum(xx, x0) * (1.0 - np.maximum(xx, x0))

        def dexact(x: Any) -> np.ndarray:
            xx = np.asarray(x, dtype=float)
            return np.where(xx < x0, P * (1.0 - x0), -P * x0)

        return Model1D(
            name=name,
            exact=exact,
            dexact=dexact,
            d2exact=lambda x: np.zeros_like(np.asarray(x, dtype=float)),
            load=None,
            a_field=None,
            c_field=None,
            a_bounds=(1.0, 1.0),
            c_bounds=(0.0, 0.0),
            bc_kind="dirichlet",
            bc_values=(0.0, 0.0),
            point_load=(x0, P),
            note=f"u=G(x,{x0:.6f})：点载荷的 Green 函数，分段线性",
        )
    if name in ("beam_point", "beam_uniform"):
        a = float(kw.get("x0", 0.5))
        P = float(kw.get("P", 1.0))
        q = float(kw.get("q", 1.0))
        if name == "beam_point":
            b = 1.0 - a

            def exact(x: Any) -> np.ndarray:
                xx = np.asarray(x, dtype=float)
                left = P * b * xx * (1.0 - b * b - xx * xx) / 6.0
                s = 1.0 - xx
                right = P * a * s * (1.0 - a * a - s * s) / 6.0
                return np.where(xx <= a, left, right)

            def dexact(x: Any) -> np.ndarray:
                # 右段 w = P a s(2x - x² - a²)/6（s=1-x），求导后是 -P a(1 - a² - 3s²)/6。
                # 这里少写一个负号曾让"节点上精确"的 Hermite 解在 H¹ 半范下量出 0.0645 的
                # "误差"（同一批解的 H² 误差是 1.8e-15）——范数与解析导数必须同时对。
                xx = np.asarray(x, dtype=float)
                left = P * b * (1.0 - b * b - 3.0 * xx * xx) / 6.0
                s = 1.0 - xx
                right = -P * a * (1.0 - a * a - 3.0 * s * s) / 6.0
                return np.where(xx <= a, left, right)

            def d2exact(x: Any) -> np.ndarray:
                xx = np.asarray(x, dtype=float)
                return np.where(xx <= a, -P * b * xx, -P * a * (1.0 - xx))

            return Model1D(
                name=name,
                exact=exact,
                dexact=dexact,
                d2exact=d2exact,
                load=None,
                a_field=None,
                c_field=None,
                a_bounds=(1.0, 1.0),
                c_bounds=(0.0, 0.0),
                bc_kind="simply_supported",
                bc_values=(0.0, 0.0),
                point_load=(a, P),
                hermite=True,
                note=f"简支梁 + 集中力 P={P:g} 于 x={a:.6f}（EI=1，ww''''=Pδ）",
            )

        def exact_u(x: Any) -> np.ndarray:
            xx = np.asarray(x, dtype=float)
            return q * xx * (1.0 - 2.0 * xx * xx + xx ** 3) / 24.0

        def dexact_u(x: Any) -> np.ndarray:
            xx = np.asarray(x, dtype=float)
            return q * (1.0 - 6.0 * xx * xx + 4.0 * xx ** 3) / 24.0

        def d2exact_u(x: Any) -> np.ndarray:
            xx = np.asarray(x, dtype=float)
            return q * (xx * xx - xx) / 2.0

        return Model1D(
            name=name,
            exact=exact_u,
            dexact=dexact_u,
            d2exact=d2exact_u,
            load=lambda x: np.full_like(np.asarray(x, dtype=float), q),
            a_field=None,
            c_field=None,
            a_bounds=(1.0, 1.0),
            c_bounds=(0.0, 0.0),
            bc_kind="simply_supported",
            bc_values=(0.0, 0.0),
            hermite=True,
            note=f"简支梁 + 均布载荷 q={q:g}（EI=1），最大挠度 5q/384={5.0 * q / 384.0:.6g}",
        )
    raise ValueError(f"未知算例 {name!r}（可选 smooth/interface/point/beam_point/beam_uniform）")


def solve_model(model: Model1D, mesh: Any, degree: int | None = None) -> dict[str, Any]:
    r"""把一个标准算例在给定网格上解出来（帧与断言共用的唯一入口）。

    返回 ``{"space","asm","sol","error","mesh"}``。Hermite 算例走梁的刚度阵，
    其余走拉格朗日；点载荷走精确的载荷泛函；间断系数走切开跳点的求积。
    """
    from numcore.pde.fem.assemble import (
        assemble_1d,
        element_beam_load,
        element_beam_stiffness,
        element_bilinear_1d,
        element_load_1d,
        element_point_load_1d,
        element_point_load_beam,
    )
    from numcore.pde.fem.mesh import hermite_space, lagrange_space

    from numcore.pde.fem.mesh import Mesh1D

    m: Mesh1D = mesh
    if model.hermite:
        space = hermite_space(m)
        ke = element_beam_stiffness(space, model.a_field)
        if model.has_point_load:
            fe = element_point_load_beam(space, model.point_load[0], model.point_load[1])
        else:
            fe = element_beam_load(space, model.load)
        fixed = space.boundary_dofs(model.bc_kind)
        vals = np.asarray([model.bc_values[0], model.bc_values[1]], dtype=float)
    else:
        space = lagrange_space(m, 1 if degree is None else int(degree))
        ke = element_bilinear_1d(space, model.a_field, model.c_field,
                                 breakpoints=model.breakpoints or None)
        if model.has_point_load:
            fe = element_point_load_1d(space, model.point_load[0], model.point_load[1])
        else:
            fe = element_load_1d(space, model.load, breakpoints=model.breakpoints or None)
        fixed = space.boundary_dofs(model.bc_kind)
        vals = np.asarray([model.bc_values[0], model.bc_values[1]], dtype=float)
    asm = assemble_1d(space, ke, fe)
    n = space.n_dof
    mask = np.ones(n, dtype=bool)
    mask[fixed] = False
    free = np.flatnonzero(mask)
    Kff = asm.K[np.ix_(free, free)]
    Ff = asm.F[free] - (asm.K[np.ix_(free, fixed)] @ vals if fixed.size else 0.0)
    out = spd_solve(Kff, Ff)
    u = np.zeros(n, dtype=float)
    if fixed.size:
        u[fixed] = vals
    u[free] = out["x"]
    sol = FemSolution(
        u=u, free=free, fixed=np.asarray(fixed, dtype=np.int64), fixed_values=vals,
        ok=bool(out["ok"]), method=str(out["method"]), n_dof=n, n_free=int(free.size),
        residual=float(out["residual"]) if np.isfinite(out["residual"]) else float("inf"),
        symmetry_defect=symmetry_defect(asm.K),
        chol_min_pivot_sq=float(out["min_pivot_sq"]), chol_fail_at=int(out["fail_at"]),
        energy=energy_norm_1d(space, u, model.a_field, model.c_field) if out["ok"] else float("inf"),
        fail_reason="" if out["ok"] else
        f"刚度阵奇异：Cholesky 在第 {out['fail_at']} 个主元失败，LU 也出现零主元",
    )
    err = error_norms_1d(
        space, u, model.exact, model.dexact,
        a_field=model.a_field, c_field=model.c_field,
        want_h2=model.hermite, d2exact=model.d2exact,
    )
    return {"space": space, "asm": asm, "sol": sol, "error": err, "mesh": m}


@dataclass(frozen=True)
class Model2D:
    r"""二维标准算例（制造解）：边界上为零的局部凸包，载荷中心可拖动。

    :math:`u=\sin(\pi x)\sin(\pi y)\,e^{-r^2/\sigma^2}`（:math:`r^2=(x-x_0)^2+(y-y_0)^2`）
    在边界上恒为零，而

    .. math::
        f=-\Delta u=2\pi^2u-2\nabla g\cdot\nabla h-g\,\Delta h,\qquad
        \nabla h=-\frac{2h}{\sigma^2}(x-x_0,\,y-y_0),\quad
        \Delta h=h\Bigl(\frac{4r^2}{\sigma^4}-\frac{4}{\sigma^2}\Bigr),

    全部闭式。于是"拖动载荷中心"这个交互动作**不会**把解析解弄丢：
    用户可以一边拖一边看误差怎么随载荷与网格的相对位置变化。
    """

    name: str
    exact: Callable[[Any], Any]
    gradient: Callable[[Any], Any]
    load: Callable[[Any], Any]
    sigma: float
    x0: float
    y0: float
    note: str = ""


def model_2d(name: str = "bump", **kw: Any) -> Model2D:
    r"""二维标准算例（目前只有 ``bump``：可拖动的局部载荷，解析解闭式）。"""
    if name != "bump":
        raise ValueError(f"未知的二维算例 {name!r}（可选 bump）")
    pi = float(np.pi)
    x0 = float(kw.get("x0", 0.5))
    y0 = float(kw.get("y0", 0.5))
    sigma = float(kw.get("sigma", 0.18))
    s2 = sigma * sigma

    def g_of(p: np.ndarray) -> np.ndarray:
        return np.sin(pi * p[:, 0]) * np.sin(pi * p[:, 1])

    def grad_g(p: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [pi * np.cos(pi * p[:, 0]) * np.sin(pi * p[:, 1]),
             pi * np.sin(pi * p[:, 0]) * np.cos(pi * p[:, 1])]
        )

    def h_of(p: np.ndarray) -> np.ndarray:
        r2 = (p[:, 0] - x0) ** 2 + (p[:, 1] - y0) ** 2
        return np.exp(-r2 / s2)

    def exact(p: Any) -> np.ndarray:
        pts = np.asarray(p, dtype=float).reshape(-1, 2)
        return g_of(pts) * h_of(pts)

    def gradient(p: Any) -> np.ndarray:
        pts = np.asarray(p, dtype=float).reshape(-1, 2)
        g = g_of(pts)
        h = h_of(pts)
        grad_h = np.column_stack([-2.0 * h * (pts[:, 0] - x0) / s2, -2.0 * h * (pts[:, 1] - y0) / s2])
        return h[:, None] * grad_g(pts) + g[:, None] * grad_h

    def load(p: Any) -> np.ndarray:
        pts = np.asarray(p, dtype=float).reshape(-1, 2)
        g = g_of(pts)
        h = h_of(pts)
        r2 = (pts[:, 0] - x0) ** 2 + (pts[:, 1] - y0) ** 2
        d2h = h * (4.0 * r2 / (s2 * s2) - 4.0 / s2)
        cross = np.sum(grad_g(pts) * np.column_stack(
            [-2.0 * h * (pts[:, 0] - x0) / s2, -2.0 * h * (pts[:, 1] - y0) / s2]), axis=1)
        return 2.0 * pi * pi * g * h - 2.0 * cross - g * d2h

    return Model2D(
        name=name, exact=exact, gradient=gradient, load=load, sigma=sigma, x0=x0, y0=y0,
        note=f"u=sin(pi x)sin(pi y)exp(-r^2/{sigma:g}^2)，载荷中心 ({x0:g},{y0:g})",
    )


def solve_model_2d(
    model: Model2D,
    mesh: Mesh2D,
    *,
    freeze_jacobian: bool = False,
) -> dict[str, Any]:
    r"""二维算例的求解入口（:math:`P_1` 三角形或 :math:`Q_1` 四边形 + 边界 Dirichlet 零）。

    ``freeze_jacobian=True`` 把每个四边形的 :math:`J` 冻结在单元中心——
    这是"把等参当仿射"的错误路线，章节里用它对照逐点 :math:`J`。
    """
    from numcore.pde.fem.assemble import assemble_2d, element_bilinear_2d, element_load_2d

    ke = element_bilinear_2d(mesh, None, None, freeze_jacobian=freeze_jacobian)
    fe = element_load_2d(mesh, model.load)
    asm = assemble_2d(mesh, ke, fe)
    fixed = np.flatnonzero(mesh.boundary_mask())
    n = mesh.n_node
    mask = np.ones(n, dtype=bool)
    mask[fixed] = False
    free = np.flatnonzero(mask)
    Kff = asm.K[np.ix_(free, free)]
    Ff = asm.F[free]
    out = spd_solve(Kff, Ff)
    u = np.zeros(n, dtype=float)
    u[free] = out["x"]
    err = error_norms_2d(mesh, u, model.exact, model.gradient)
    sol = FemSolution(
        u=u, free=free, fixed=np.asarray(fixed, dtype=np.int64),
        fixed_values=np.zeros(fixed.size, dtype=float), ok=bool(out["ok"]),
        method=str(out["method"]), n_dof=n, n_free=int(free.size),
        residual=float(out["residual"]) if np.isfinite(out["residual"]) else float("inf"),
        symmetry_defect=symmetry_defect(asm.K), chol_min_pivot_sq=float(out["min_pivot_sq"]),
        chol_fail_at=int(out["fail_at"]), energy=float(err["energy"]),
        fail_reason="" if out["ok"] else "刚度阵奇异：Cholesky 与 LU 都无法分解",
    )
    return {"mesh": mesh, "asm": asm, "sol": sol, "error": err}
