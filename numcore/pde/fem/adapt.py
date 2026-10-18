r"""后验误差指示子与自适应加密。

数学定义
--------
先验估计说"误差 :math:`\le C h^p`"，但它给不出**这一张网格上**哪个单元该加密。
后验指示子干的就是这件事。一维二阶问题 :math:`-(au')'+cu=f` 的残差型指示子是

.. math::
    \eta_K^2=h_K^2\|R_K\|_{L^2(K)}^2+\tfrac12\sum_{F\subset\partial K}h_F\,\|J_F\|^2,
    \qquad R_K=f+ (au_h')'-cu_h,\quad J_F=[\![a u_h']\!]_F,

其中 :math:`R_K` 是**单元内部残差**、:math:`J_F` 是**面跳跃**。它的来源是
"把 :math:`u-u_h` 代进弱形式、再逐单元分部积分"——离散解在单元内部并不满足微分方程，
在节点处导数不连续，两件事各自留下一个残差项。

三条纪律，本模块都实现成可测量的量：

1. **指示子必须与真误差对拍**：效率指数 :math:`\eta_K/\|e\|_{a,K}` 是一个**分布**，
   只给总量会掩盖"某些单元差 10 倍、某些差 0.1 倍"。:func:`efficiency_stats`
   给分位数、总有效指数与对数相关；
2. **标记策略会改变结果**：Dörfler 标记（:func:`dorfler_mark`）取"累计平方和占比达
   :math:`\theta` 的最小集合"，:math:`\theta` 越小标记越多；
3. **加密要有闭包**：一维里把一个单元二分会让邻居出现悬挂节点，所以
   :meth:`~numcore.pde.fem.mesh.Mesh1D.refine_marked` 会把邻居一起分掉——
   于是"同样自由度"的比较必须用**加密后的真实自由度**，不能用标记数。

**为什么自适应在 1D 能扳回指数收敛**：尖角解的真误差集中在极少数单元里，
自适应加密只加深那一处，单元数随深度**线性**增长而误差几何下降；
一致加密则要付出 :math:`h^{1/2}` 的代数衰减。这个对照在 :func:`compare_strategies` 里实测。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from numcore.pde.fem.assemble import (
    element_bilinear_1d,
    element_load_1d,
    element_point_load_1d,
    eval_field,
)
from numcore.pde.fem.mesh import Mesh1D, Space1D, lagrange_space
from numcore.pde.fem.shape import gauss_legendre
from numcore.pde.fem.solve import FemSolution, solve_1d

__all__ = [
    "Indicator",
    "residual_indicator_1d",
    "dorfler_mark",
    "true_element_errors",
    "efficiency_stats",
    "adaptive_history",
    "uniform_history",
    "compare_strategies",
    "RefinementLevel",
    "pack_levels",
]

_NQ_IND = 6
"""指示子积分的 Gauss 点数（单元内部残差是低次多项式/光滑函数）。"""


@dataclass(frozen=True)
class Indicator:
    r"""一次后验估计的结果：逐单元指示子与它的两个来源。"""

    eta: np.ndarray
    residual: np.ndarray
    jump: np.ndarray
    eta_total: float
    h: np.ndarray
    n_elem: int

    def to_json(self) -> dict[str, Any]:
        return {"eta_total": self.eta_total, "n_elem": self.n_elem}


def _second_deriv_per_element(space: Space1D, u: np.ndarray) -> np.ndarray:
    r"""单元内 :math:`u_h''`（:math:`P_1` 恒为 0、:math:`P_2` 恒为常数 :math:`\frac{2}{h^2}(u_{i}-2u_m+u_{i+1})`）。

    Hermite 与高阶拉格朗日空间不走这条路（它们的二阶导在单元内不是常数，
    残差型指示子要另写），调用时会显式报错而不是给一个没人认领的数。
    """
    if space.family == "p1":
        return np.zeros(space.n_elem, dtype=float)
    if space.family == "p2":
        out = np.zeros(space.n_elem, dtype=float)
        for e in range(space.n_elem):
            ids = space.elems[e]
            h = float(space.mesh.x[space.mesh.elems[e, 1]] - space.mesh.x[space.mesh.elems[e, 0]])
            out[e] = 2.0 / (h * h) * (u[ids[0]] - 2.0 * u[ids[2]] + u[ids[1]])
        return out
    raise ValueError(f"残差型指示子只对 P1/P2 空间实现（得到 {space.family!r}）")


def residual_indicator_1d(
    space: Space1D,
    u: np.ndarray,
    f_field: Callable[[np.ndarray], Any] | None = None,
    a_field: Callable[[np.ndarray], Any] | None = None,
    c_field: Callable[[np.ndarray], Any] | None = None,
) -> Indicator:
    r"""残差型后验指示子（单元内部残差 + 面跳跃），逐单元返回。

    面跳跃用的是**逐点**的 :math:`a`：在间断系数问题里，界面两侧的
    :math:`a u_h'` 才是通量，用单元中心的 :math:`a` 会把跳跃算错一个因子。
    内部残差里的 :math:`(au_h')'` 取 :math:`a\,u_h''`（:math:`a` 在单元内分段常数时精确）。
    """
    u2 = _second_deriv_per_element(space, u)
    xs, ws = gauss_legendre(max(_NQ_IND, int(space.degree) + 2))
    h = space.mesh.h
    res2 = np.zeros(space.n_elem, dtype=float)
    for e in range(space.n_elem):
        xa = float(space.mesh.x[space.mesh.elems[e, 0]])
        xb = float(space.mesh.x[space.mesh.elems[e, 1]])
        hh = xb - xa
        xq = xa + 0.5 * hh * (xs + 1.0)
        wq = 0.5 * hh * ws
        fq = eval_field(f_field, xq, default=0.0)
        aq = eval_field(a_field, xq)
        cq = eval_field(c_field, xq, default=0.0)
        uh = _values_at(space, u, e, xq)
        R = fq + aq * u2[e] - cq * uh
        res2[e] = float(np.sum(wq * R * R))
    jump = _face_jumps(space, u, a_field)
    eta = np.sqrt(np.maximum(h * h * res2, 0.0) + 0.5 * jump)
    return Indicator(
        eta=eta,
        residual=np.sqrt(np.maximum(res2, 0.0)),
        jump=np.sqrt(np.maximum(jump, 0.0)),
        eta_total=float(np.sqrt(np.sum(eta * eta))),
        h=h.copy(),
        n_elem=space.n_elem,
    )


def _values_at(space: Space1D, u: np.ndarray, e: int, xq: np.ndarray) -> np.ndarray:
    """单元 ``e`` 上 :math:`u_h` 在 ``xq`` 处的值。"""
    from numcore.pde.fem.solve import _eval_uh_1d

    uh, _, _ = _eval_uh_1d(space, u, e, xq)
    return uh


def _face_jumps(space: Space1D, u: np.ndarray, a_field: Callable[[np.ndarray], Any] | None) -> np.ndarray:
    r"""逐单元累计的面跳跃项 :math:`h_F[\![au_h']\!]^2`（内部节点各贡献给左右两个单元）。

    一维里"面"就是节点；界面尺度取两侧单元长度的**平均** :math:`(h_{K}+h_{K'})/2`。
    """
    out = np.zeros(space.n_elem, dtype=float)
    from numcore.pde.fem.solve import _eval_uh_1d

    n_node = space.mesh.n_node
    for i in range(1, n_node - 1):
        left = i - 1
        right = i
        xi = float(space.mesh.x[i])
        dl = _eval_uh_1d(space, u, left, np.asarray([xi]))[1][0]
        dr = _eval_uh_1d(space, u, right, np.asarray([xi]))[1][0]
        ai = float(eval_field(a_field, np.asarray([xi]))[0])
        hf = 0.5 * (float(space.mesh.h[left]) + float(space.mesh.h[right]))
        val = hf * (ai * (dr - dl)) ** 2
        out[left] += 0.5 * val
        out[right] += 0.5 * val
    return out


def dorfler_mark(eta: Any, theta: float = 0.5) -> np.ndarray:
    r"""Dörfler 标记：取累计 :math:`\eta^2` 占比首次达到 :math:`\theta` 的最小集合。

    返回布尔数组。:math:`\theta=1` 等价于全部标记（一致加密），
    :math:`\theta\to0` 只标记最大的那一个——两者都由同一个函数给出，
    于是"标记策略"在本章里是一个可以扫的参数，而不是两段代码。
    """
    e = np.asarray(eta, dtype=float).reshape(-1)
    th = float(theta)
    if not 0.0 < th <= 1.0:
        raise ValueError(f"theta 必须在 (0,1] 内，得到 {theta!r}")
    total = float(np.sum(e * e))
    mask = np.zeros(e.size, dtype=bool)
    if total <= 0.0:
        return mask
    order = np.argsort(-e, kind="stable")
    acc = 0.0
    for idx in order:
        mask[idx] = True
        acc += float(e[idx] ** 2)
        if acc >= th * total:
            break
    return mask


def true_element_errors(
    space: Space1D,
    u: np.ndarray,
    exact: Callable[[np.ndarray], Any],
    dexact: Callable[[np.ndarray], Any],
    a_field: Callable[[np.ndarray], Any] | None = None,
    c_field: Callable[[np.ndarray], Any] | None = None,
) -> np.ndarray:
    r"""逐单元**真**能量误差 :math:`\|e\|_{a,K}`：效率指数分布的分母。"""
    from numcore.pde.fem.solve import _err_order, _eval_uh_1d

    # 求积阶必须与 error_norms_1d 一致：不然"逐单元误差之和"与"总误差"会差一个
    # 0.08% 的求积伪影（10 点 vs 12 点），而效率指数会把这个伪影算进指示子头上。
    xs, ws = gauss_legendre(_err_order(space))
    out = np.zeros(space.n_elem, dtype=float)
    for e in range(space.n_elem):
        xa = float(space.mesh.x[space.mesh.elems[e, 0]])
        xb = float(space.mesh.x[space.mesh.elems[e, 1]])
        hh = xb - xa
        xq = xa + 0.5 * hh * (xs + 1.0)
        wq = 0.5 * hh * ws
        uh, duh, _ = _eval_uh_1d(space, u, e, xq)
        de = np.asarray(exact(xq), dtype=float) - uh
        dd = np.asarray(dexact(xq), dtype=float) - duh
        aq = eval_field(a_field, xq)
        cq = eval_field(c_field, xq, default=0.0)
        out[e] = float(np.sqrt(max(np.sum(wq * (aq * dd * dd + cq * de * de)), 0.0)))
    return out


def efficiency_stats(eta: Any, true_err: Any, *, floor: float = 1e-14) -> dict[str, Any]:
    r"""指示子相对真误差的效率指数分布。

    ``effectivity_total`` :math:`=(\sum\eta_K^2)^{1/2}/(\sum\|e\|_{a,K}^2)^{1/2}`
    是常被引用的那个数；本函数另外给逐单元比值的分位数、最大最小比、
    以及 :math:`\log\eta` 与 :math:`\log\|e\|_a` 的相关系数——
    后者才回答"指示子会不会指错地方"。
    """
    e = np.asarray(eta, dtype=float).reshape(-1)
    t = np.asarray(true_err, dtype=float).reshape(-1)
    if e.size != t.size:
        raise ValueError(f"长度不一致：{e.size} vs {t.size}")
    tot_t = float(np.sqrt(np.sum(t * t)))
    tot_e = float(np.sqrt(np.sum(e * e)))
    ratio = e / np.maximum(t, floor)
    le = np.log(np.maximum(e, floor))
    lt = np.log(np.maximum(t, floor))
    ce = le - float(np.mean(le))
    ct = lt - float(np.mean(lt))
    denom = float(np.sqrt(np.sum(ce * ce) * np.sum(ct * ct)))
    corr = float(np.sum(ce * ct) / denom) if denom > 0.0 else 0.0
    return {
        "effectivity_total": float(tot_e / tot_t) if tot_t > 0.0 else float("inf"),
        "eta_total": tot_e,
        "true_total": tot_t,
        "ratio_min": float(np.min(ratio)),
        "ratio_q25": float(np.quantile(ratio, 0.25)),
        "ratio_median": float(np.median(ratio)),
        "ratio_q75": float(np.quantile(ratio, 0.75)),
        "ratio_max": float(np.max(ratio)),
        "log_corr": corr,
    }


@dataclass(frozen=True)
class RefinementLevel:
    r"""自适应/一致加密的第 :math:`\ell` 层快照（帧里的「网格演变」就是它的序列）。"""

    level: int
    n_elem: int
    n_dof: int
    mesh: Mesh1D
    u: np.ndarray
    error_energy: float
    error_l2: float
    eta: np.ndarray
    true_err: np.ndarray
    effectivity_total: float
    marked_fraction: float


def _level_snapshot(space: Space1D, sol: FemSolution, exact, dexact, f_field, a_field, c_field,
                    marked_fraction: float) -> RefinementLevel:
    from numcore.pde.fem.solve import error_norms_1d

    err = error_norms_1d(space, sol.u, exact, dexact, a_field=a_field, c_field=c_field)
    ind = residual_indicator_1d(space, sol.u, f_field, a_field, c_field)
    true_err = true_element_errors(space, sol.u, exact, dexact, a_field, c_field)
    st = efficiency_stats(ind.eta, true_err)
    return RefinementLevel(
        level=0,
        n_elem=space.n_elem,
        n_dof=space.n_dof,
        mesh=space.mesh,
        u=sol.u,
        error_energy=float(err["energy"]),
        error_l2=float(err["l2"]),
        eta=ind.eta,
        true_err=true_err,
        effectivity_total=float(st["effectivity_total"]),
        marked_fraction=float(marked_fraction),
    )


def _solve_level(mesh: Mesh1D, degree: int, f_field, a_field, c_field,
                 fixed: Any, fixed_values: Any | None,
                 point_load: tuple[float, float] | None) -> tuple[Space1D, FemSolution]:
    space = lagrange_space(mesh, degree)
    ke = element_bilinear_1d(space, a_field, c_field)
    if point_load is None:
        fe = element_load_1d(space, f_field)
    else:
        fe = element_point_load_1d(space, point_load[0], point_load[1])
    sol = solve_1d(space, ke, fe, fixed, fixed_values)[1]
    return space, sol


def adaptive_history(
    mesh0: Mesh1D,
    *,
    degree: int = 1,
    exact: Callable[[np.ndarray], Any],
    dexact: Callable[[np.ndarray], Any],
    f_field: Callable[[np.ndarray], Any] | None = None,
    a_field: Callable[[np.ndarray], Any] | None = None,
    c_field: Callable[[np.ndarray], Any] | None = None,
    fixed: Any | None = None,
    fixed_values: Any | None = None,
    point_load: tuple[float, float] | None = None,
    theta: float = 0.5,
    levels: int = 8,
) -> list[RefinementLevel]:
    r"""自适应加密历史：解一次 → 估计 → Dörfler 标记 → :meth:`Mesh1D.refine_marked` → 再解。

    ``point_load=(x0,P)`` 时用点载荷泛函代替 :math:`\int fv`（奇异解的经典入口）。
    """
    out: list[RefinementLevel] = []
    mesh = mesh0
    for lev in range(int(levels) + 1):
        space, sol = _solve_level(mesh, degree, f_field, a_field, c_field,
                                  fixed, fixed_values, point_load)
        marked_frac = 1.0
        if lev < int(levels):
            ind = residual_indicator_1d(space, sol.u, f_field, a_field, c_field)
            mark = dorfler_mark(ind.eta, theta)
            marked_frac = float(np.mean(mark)) if mark.size else 0.0
            if not bool(np.any(mark)):
                mark = np.ones(mesh.n_elem, dtype=bool)
            mesh = mesh.refine_marked(mark)
        snap = _level_snapshot(space, sol, exact, dexact, f_field, a_field, c_field, marked_frac)
        out.append(
            RefinementLevel(
                level=lev, n_elem=snap.n_elem, n_dof=snap.n_dof, mesh=snap.mesh, u=snap.u,
                error_energy=snap.error_energy, error_l2=snap.error_l2, eta=snap.eta,
                true_err=snap.true_err, effectivity_total=snap.effectivity_total,
                marked_fraction=snap.marked_fraction,
            )
        )
    return out


def uniform_history(
    mesh0: Mesh1D,
    *,
    degree: int = 1,
    exact: Callable[[np.ndarray], Any],
    dexact: Callable[[np.ndarray], Any],
    f_field: Callable[[np.ndarray], Any] | None = None,
    a_field: Callable[[np.ndarray], Any] | None = None,
    c_field: Callable[[np.ndarray], Any] | None = None,
    fixed: Any | None = None,
    fixed_values: Any | None = None,
    point_load: tuple[float, float] | None = None,
    levels: int = 8,
) -> list[RefinementLevel]:
    r"""一致加密历史（同样的层数、同样的解与估计，只是标记换成"全部"）。"""
    out: list[RefinementLevel] = []
    mesh = mesh0
    for lev in range(int(levels) + 1):
        space, sol = _solve_level(mesh, degree, f_field, a_field, c_field,
                                  fixed, fixed_values, point_load)
        snap = _level_snapshot(space, sol, exact, dexact, f_field, a_field, c_field, 1.0)
        out.append(
            RefinementLevel(
                level=lev, n_elem=snap.n_elem, n_dof=snap.n_dof, mesh=snap.mesh, u=snap.u,
                error_energy=snap.error_energy, error_l2=snap.error_l2, eta=snap.eta,
                true_err=snap.true_err, effectivity_total=snap.effectivity_total,
                marked_fraction=1.0,
            )
        )
        if lev < int(levels):
            mesh = mesh.refine_uniform()
    return out


def _log_interp(x: float, lx: np.ndarray, ly: np.ndarray) -> float:
    r"""对数-对数坐标下的**自研**分段线性插值（``np.interp`` 属 P3 禁用）。

    ``lx`` 必须严格递增；``x`` 落在区间外时夹到端点（调用方自己判范围）。
    二分找区间，段内按比例线性混合——与 ``np.interp`` 逐位一致（同样只有四则运算）。
    """
    if x <= lx[0]:
        return float(ly[0])
    if x >= lx[-1]:
        return float(ly[-1])
    lo, hi = 0, int(lx.size) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if lx[mid] <= x:
            lo = mid
        else:
            hi = mid
    t = (x - float(lx[lo])) / (float(lx[hi]) - float(lx[lo]))
    return float(ly[lo] + t * (ly[hi] - ly[lo]))


def compare_strategies(adaptive: list[RefinementLevel], uniform: list[RefinementLevel]) -> dict[str, Any]:
    r"""在**相同自由度**下比较两条路线：对每个自适应层的 :math:`n_{dof}`，
    在一致加密历史里线性插值出该自由度下的误差，给出比值。

    插值在对数-对数坐标下做（误差随自由度是幂律或指数律，端点上都不会外推：
    只有落在一致历史的自由度区间内的自适应层才参与比较）。
    """
    nd_a = np.asarray([lv.n_dof for lv in adaptive], dtype=float)
    ea = np.asarray([lv.error_energy for lv in adaptive], dtype=float)
    nd_u = np.asarray([lv.n_dof for lv in uniform], dtype=float)
    eu = np.asarray([lv.error_energy for lv in uniform], dtype=float)
    ratio = np.full(nd_a.size, np.nan)
    lx = np.log(nd_u)
    ly = np.log(eu)
    for i in range(nd_a.size):
        if nd_u[0] <= nd_a[i] <= nd_u[-1] and np.all(eu > 0.0):
            ratio[i] = ea[i] / float(np.exp(_log_interp(np.log(nd_a[i]), lx, ly)))
    return {
        "n_dof_adaptive": nd_a,
        "error_adaptive": ea,
        "n_dof_uniform": nd_u,
        "error_uniform": eu,
        "ratio_adaptive_over_uniform": ratio,
        "best_ratio": float(np.nanmin(ratio)) if np.any(np.isfinite(ratio)) else float("nan"),
        "median_ratio": float(np.nanmedian(ratio)) if np.any(np.isfinite(ratio)) else float("nan"),
    }


def pack_levels(levels: list[RefinementLevel]) -> dict[str, np.ndarray]:
    r"""把不等长的层快照打包成"拼接数组 + 偏移"（帧的数组段必须是矩形的）。

    返回 ``node_x``（所有层的节点拼在一起）、``level_offset``（每层在拼接数组里的起点）、
    ``indicator`` / ``true_error`` / ``u`` 同样拼接；``n_dof`` / ``error_energy`` 是逐层标量。
    """
    nodes = np.concatenate([lv.mesh.x for lv in levels])
    off = np.cumsum([0] + [lv.mesh.x.size for lv in levels]).astype(np.int64)
    eta = np.concatenate([_pad_to_nodes(lv) for lv in levels])
    return {
        "node_x": nodes,
        "level_offset": off,
        "level_n_dof": np.asarray([lv.n_dof for lv in levels], dtype=np.int64),
        "level_n_elem": np.asarray([lv.n_elem for lv in levels], dtype=np.int64),
        "level_error_energy": np.asarray([lv.error_energy for lv in levels], dtype=float),
        "level_error_l2": np.asarray([lv.error_l2 for lv in levels], dtype=float),
        "level_effectivity": np.asarray([lv.effectivity_total for lv in levels], dtype=float),
        "indicator": eta,
        "element_h": np.concatenate([lv.mesh.h for lv in levels]),
        "true_element_error": np.concatenate([lv.true_err for lv in levels]),
    }


def _pad_to_nodes(lv: RefinementLevel) -> np.ndarray:
    """把逐**单元**的量铺到**节点**上（前端按节点画热力条时不用再做索引算术）。"""
    x = lv.mesh.x
    val = np.zeros(x.size, dtype=float)
    for e in range(lv.mesh.n_elem):
        i, j = int(lv.mesh.elems[e, 0]), int(lv.mesh.elems[e, 1])
        val[i] = max(val[i], float(lv.eta[e]))
        val[j] = max(val[j], float(lv.eta[e]))
    return val


def indicator_correlation(levels: list[RefinementLevel]) -> float:
    r"""所有层的 :math:`\log\eta` 与 :math:`\log\|e\|_a` 合在一起的相关系数（指示子可信度的单数概括）。"""
    eta = np.concatenate([lv.eta for lv in levels])
    tru = np.concatenate([lv.true_err for lv in levels])
    return float(efficiency_stats(eta, tru)["log_corr"])
