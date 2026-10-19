r"""4.4 章：有限元方法（4 个可视化）。

帧只做搬运：网格、单元矩阵、装配、求解、误差范数、后验指示子全部来自
:mod:`numcore.pde.fem`，本文件不另写一份数学。四个可视化的分工是
"**同一套装配代码，换一个地方看它会不会骗你**"：

1. ``assembly_animation`` —— 装配动画。把单元矩阵一格格加进全局矩阵，
   时间轴是"已经装进去几个单元"。``mirror=double``（上三角镜像时把对角线搬了两遍）
   装出来的矩阵**对称残差恰好是 0.0**、Cholesky 成功、条件数还从 25.27 降到 2.72
   ——所有常规体检指标都变好了，可是矩阵不是那个矩阵（max|ΔK| = 16.0）。
2. ``solve_deform`` —— 拖动载荷位置的实时求解。梁用 Hermite 三次（4 阶问题），
   膜用 Q1/三角形等参（2 阶问题）。副图是**载荷位置扫描**：集中力落在节点上时
   能量误差是 1e-15 量级（一维 Green 函数分段线性，节点精确），落进单元内部
   立刻跳到 1e-3——误差关于载荷位置是**锯齿**，同时节点误差始终在机器精度上。
3. ``hp_convergence`` —— h-加密 vs p-加密。同一个 p 阶梯在光滑解上半对数拟合
   R2=0.970（指数收敛），在带尖角的解上掉到 0.759 而双对数拟合是 0.928（代数）。
   "谱收敛"不是方法的属性，是（方法，解的正则性）的属性；副图给出条件数随 p 的增长。
4. ``adaptive_mesh`` —— 后验指示子热力图 + 自适应网格演变。时间轴是加密层数。
   指示子的效率指数是一个**分布**（实测总量 1.24–1.92），而在同样自由度下
   自适应比一致加密好 8 倍（实测比值 0.125），代价是最小单元只有 4.9e-4。

帧契约（与 4.1 一致）：``meta`` 只放有限标量，非有限一律换成哨兵 + 显式掩码 + strings 说明；
数组走 ``add``，字符串走 ``set_strings``；每个时间轴的语义写在 ``time_semantics`` 里。
"""

from __future__ import annotations

import functools
import math
from typing import Any, Mapping

import numpy as np

from numcore.pde.fem import adapt as ad
from numcore.pde.fem import assemble as asm
from numcore.pde.fem import mesh as ms
from numcore.pde.fem import solve as sv
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

#: 误差类量的地板：对数轴上比它更小的值显示为地板并置掩码。
_ERR_FLOOR = 1.0e-17

#: 逐单元"效率指数"的地板：真误差为 0 的单元上比值没有意义，不写 NaN。
_RATIO_FLOOR = 1.0e-12

#: 逐单元效率指数的判定门槛（相对最大单元误差）：低于它的单元只报占位值 + 掩码。
_REL_ERR_FLOOR = 1.0e-3

#: 允许做精确特征分解的自由度上限（Jacobi 是稠密 O(n^3)，n=81 就要 0.6 秒）。
_EIG_MAX = 48

#: 条件数的显示上限（矩阵半装配时是奇异的，条件数是 inf）。
_COND_CEIL = 1.0e300

#: 一致加密阶梯的单元数上限：超过它就不再加倍（自适应那侧的对照仍在同一段自由度区间内）。
_UNIFORM_MAX_ELEM = 1024

#: 扫载荷位置的采样点数（每点一次求解，帧预算 ~1 秒）。
_SCAN_1D = 41
_SCAN_2D = 9

_NONFINITE_POLICY = (
    "本帧不含任何非有限值：对数轴上的 0 与无穷用 strings.floor（={floor:g}）占位并配掩码"
    "（*_at_floor / *_unmeasured）；真误差为 0 的单元上效率指数用 {ratio:g} 占位并配 "
    "ratio_undefined 掩码；奇异系统的解不存在，用 0 占位并配 singular_mask 与 meta.fail_reason，"
    "而不是写 NaN；折线不用 NaN 分隔（NaN 会让前端静默丢段）；meta 一律有限标量"
    "（.nlf 头 allow_nan=False，塞一个 NaN 整帧就写不出去）。"
)

_TIME_SEMANTICS = {
    "assembly_animation": "装配进度：t∈[0,1] 映射到「已装入第 k=floor(t·n_elem) 个单元」，t=1 是装完",
    "solve_deform": "准静态加载因子 λ=t∈[0,1]：位移与载荷成正比、能量与 λ^2 成正比（线性问题）",
    "hp_convergence": "无时间轴（一次性扫完整条 h 阶梯与 p 阶梯）",
    "adaptive_mesh": "加密层数：t∈[0,1] 映射到第 ℓ=round(t·levels) 层网格，t=1 是最细的一层",
}


def _fin(x: Any, default: float = 0.0) -> float:
    """把非有限标量（含 ``None``）换成哨兵（``.nlf`` 头用 ``allow_nan=False``）。"""
    if x is None:
        return float(default)
    v = float(x)
    return v if math.isfinite(v) else float(default)


def _floor(arr: Any, floor: float = _ERR_FLOOR) -> np.ndarray:
    """对数轴的软着陆：把 0 与非有限值抬到地板上（配掩码如实报告）。"""
    a = np.asarray(arr, dtype=float).copy()
    a[~np.isfinite(a)] = floor
    return np.maximum(a, floor)


def _cond(arr: Any) -> np.ndarray:
    """条件数数组：非正/非有限（未测量或奇异）一律换成显示上限。"""
    a = np.asarray(arr, dtype=float).copy()
    a[~np.isfinite(a)] = _COND_CEIL
    return np.clip(a, 1.0, _COND_CEIL)


def _loglog_slope(x: Any, y: Any) -> float:
    """双对数最小二乘斜率（自己算，不用被禁用的现成拟合）。"""
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.asarray(y, dtype=float))
    n = lx.size
    if n < 2 or abs(n * np.sum(lx * lx) - np.sum(lx) ** 2) < 1e-300:
        return float("nan")
    return float((n * np.sum(lx * ly) - np.sum(lx) * np.sum(ly)) / (n * np.sum(lx * lx) - np.sum(lx) ** 2))


def _fit_quality(x: Any, y: Any, logy: bool) -> tuple[float, float]:
    """``(斜率, R^2)``：``logy=True`` 时对 :math:`\\ln y` 做半对数拟合，否则双对数。"""
    xa = np.log(np.asarray(x, dtype=float)) if logy else np.asarray(x, dtype=float)
    ya = np.log(np.asarray(y, dtype=float))
    n = xa.size
    if n < 2:
        return float("nan"), float("nan")
    sx, sy = float(np.sum(xa)), float(np.sum(ya))
    den = n * float(np.sum(xa * xa)) - sx * sx
    if abs(den) < 1e-300:
        return float("nan"), float("nan")
    slope = (n * float(np.sum(xa * ya)) - sx * sy) / den
    intercept = (sy - slope * sx) / n
    ss_res = float(np.sum((ya - (slope * xa + intercept)) ** 2))
    ss_tot = float(np.sum((ya - np.mean(ya)) ** 2))
    return float(slope), (1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan"))


def _log_corr(a: Any, b: Any, floor: float) -> float:
    """:math:`\\log` 尺度上的相关系数（指示子会不会指错地方的度量）。"""
    la = np.log(np.maximum(np.asarray(a, dtype=float), floor))
    lb = np.log(np.maximum(np.asarray(b, dtype=float), floor))
    ca = la - float(np.mean(la))
    cb = lb - float(np.mean(lb))
    den = float(np.sqrt(np.sum(ca * ca) * np.sum(cb * cb)))
    return float(np.sum(ca * cb) / den) if den > 0.0 else 0.0


# --------------------------------------------------------------------------- #
# viz 1：装配动画
# --------------------------------------------------------------------------- #
def _coefficient_field(kind: str):
    """装配动画用的系数场：常数 1，或界面在 1/π 处从 1 跳到 100。"""
    if kind == "const":
        return None, None
    xs = 1.0 / float(np.pi)

    def a_jump(x: Any) -> np.ndarray:
        return np.where(np.asarray(x, dtype=float) < xs, 1.0, 100.0)

    return a_jump, (xs,)


@functools.lru_cache(maxsize=64)
def _assembly_history(n_elem: int, degree: str, coefficient: str, mirror: str) -> dict[str, Any]:
    """每个单元装一步：部分矩阵、定型时刻、残差、条件数代理与解。

    这是整章最贵的一段（O(n_elem) 次前缀装配 + 一次特征分解），所以整帧缓存：
    scrub 时间轴只是取其中一步，不该每一步都重算。
    """
    mesh = ms.uniform_mesh_1d(n_elem)
    degree_num = 1 if degree == "p1" else 2
    space = ms.lagrange_space(mesh, degree_num)
    a_field, breaks = _coefficient_field(coefficient)
    ke = asm.element_bilinear_1d(space, a_field, None, breakpoints=breaks)
    fe = asm.element_load_1d(space, lambda x: np.ones_like(np.asarray(x, dtype=float)), breakpoints=breaks)
    n_dof = space.n_dof
    full = asm.assemble_1d(space, ke, fe)
    K_full = full.K
    F_full = full.F
    K_route = K_full if mirror == "full" else asm.upper_mirror(ke, space.elems, n_dof, F_full,
                                                                 diagonal=mirror).K
    route_defect = float(np.max(np.abs(K_route - K_full)))

    partial = np.stack([full.partial(k) for k in range(n_elem + 1)], axis=0)
    res_hist = np.asarray([float(np.max(np.abs(partial[k] - K_full))) for k in range(n_elem + 1)])
    sym_hist = np.asarray([asm.symmetry_defect(partial[k]) for k in range(n_elem + 1)])

    free = np.flatnonzero(_free_mask(space))
    piv_hist = np.zeros(n_elem + 1)
    Kff = K_full[np.ix_(free, free)]
    for k in range(n_elem + 1):
        if free.size == 0:
            piv_hist[k] = _COND_CEIL
            continue
        sub = partial[k][np.ix_(free, free)]
        res = sv.spd_solve(sub, np.zeros(free.size))
        if res["ok"] and res["min_pivot_sq"] > 0.0:
            piv_hist[k] = min(res["max_pivot_sq"] / res["min_pivot_sq"], _COND_CEIL)
        else:
            piv_hist[k] = _COND_CEIL
    piv_hist[0] = _COND_CEIL

    lam = {"min": -1.0, "max": -1.0, "cond": -1.0}
    if 0 < free.size <= _EIG_MAX:
        c = sv.system_condition(Kff)
        lam = {"min": _fin(c["lambda_min"], -1.0), "max": _fin(c["lambda_max"], -1.0),
               "cond": _fin(c["condition"], -1.0)}
    sol_full = sv.spd_solve(Kff, F_full[free])
    u_full = np.zeros(n_dof)
    if sol_full["ok"]:
        u_full[free] = sol_full["x"]
    sol_route = sv.spd_solve(K_route[np.ix_(free, free)], F_full[free])
    u_route = np.zeros(n_dof)
    if sol_route["ok"]:
        u_route[free] = sol_route["x"]
    solve_defect = float(np.max(np.abs(u_route - u_full))) if sol_full["ok"] and sol_route["ok"] else -1.0

    return {
        "mesh": mesh, "space": space, "n_dof": n_dof, "ke": ke, "fe": fe,
        "K_full": K_full, "K_route": K_route, "F": F_full, "partial": partial,
        "res_hist": res_hist, "sym_hist": sym_hist, "piv_hist": piv_hist, "lam": lam,
        "route_defect": route_defect, "solve_defect": solve_defect,
        "entry_done_at": full.entry_done_at(), "free": free,
        "u_full": u_full, "u_route": u_route, "ok_route": bool(sol_route["ok"]),
    }


def _free_mask(space: ms.Space1D) -> np.ndarray:
    """Dirichlet 自由自由度掩码（两端钉住）。"""
    m = np.ones(space.n_dof, dtype=bool)
    m[space.boundary_dofs("dirichlet")] = False
    return m


_VIZ1_MIRRORS = ("full", "half", "double")


def _viz_assembly(p: Mapping[str, Any], t: float) -> Frame:
    n_elem = int(p["n_elem"])
    degree = str(p["degree"])
    coefficient = str(p["coefficient"])
    mirror = str(p["mirror"])
    info = _assembly_history(n_elem, degree, coefficient, mirror)
    space: ms.Space1D = info["space"]
    n_dof = int(info["n_dof"])
    t = max(0.0, min(float(t), 1.0))
    k_now = int(math.floor(t * n_elem + 1e-9))   # 已经装进去的单元数
    fr = Frame()

    K_partial = info["partial"][k_now]
    e_idx = max(0, min(k_now, n_elem - 1))
    local_dofs = np.asarray(space.elems[e_idx], dtype=np.int64)
    done_at = np.asarray(info["entry_done_at"], dtype=float)
    pending = (done_at < 0.0).astype(float)
    fr.add("mesh_x", space.mesh.x, unit="1", label="网格顶点 x")
    fr.add("mesh_h", space.mesh.h, unit="1", label="单元长度 h_K")
    fr.add("K_partial", K_partial, unit="1", label=f"第 {k_now}/{n_elem} 个单元装完后的 K（截断显示未装入的部分）")
    fr.add("K_route", info["K_route"], unit="1", label="所选装配路线的最终 K")
    fr.add("K_final", info["K_full"], unit="1", label="完整散射装配的最终 K（参照）")
    fr.add("K_defect", info["K_route"] - info["K_full"], unit="1",
           label="所选路线 − 完整散射（half 路线恒为 0；double 路线对角线差一倍）")
    fr.add("k_local", info["ke"][e_idx], unit="1", label="当前单元的局部矩阵 ke")
    fr.add("local_dofs", local_dofs, unit="1", label="当前单元的自由度编号（散射目标）")
    fr.add("entry_done_at", done_at, unit="1", label="该元素最终定型的单元号（-1 = 还没被任何单元碰过）")
    fr.add("entry_pending", pending, unit="1", label="1 = 该元素尚未定型（-1 的掩码）")
    fr.add("k_axis", np.arange(n_elem + 1, dtype=float), unit="1", label="已装入的单元数 k")
    fr.add("residual", _floor(info["res_hist"]), unit="1", label="max|K_k − K|（对数轴，地板 1e-17）")
    fr.add("residual_at_floor", (info["res_hist"] < _ERR_FLOOR).astype(float), unit="1",
           label="1 = 该步的残差已到地板")
    fr.add("symmetry_residual", info["sym_hist"], unit="1", label="max|K_k − K_k^T|（对轴，P1 上恒为精确 0）")
    fr.add("pivot_condition", _cond(info["piv_hist"]), unit="1",
           label="自由块 Cholesky 主元比（条件数代理；半装配时矩阵奇异 = 显示上限）")
    fr.add("pivot_unmeasured", (info["piv_hist"] >= _COND_CEIL).astype(float), unit="1",
           label="1 = 该步矩阵奇异/未测量，主元比不是数")
    fr.add("u_full", info["u_full"], unit="1", label="完整散射装配下的解 u_h")
    fr.add("u_route", info["u_route"], unit="1", label="所选装配路线下的解 u_h")
    fr.add("solution_defect", _floor(np.abs(info["u_route"] - info["u_full"])), unit="1",
           label="|u_route − u_full|（对数轴）")

    fr.set_meta(
        n_elem=n_elem, n_dof=n_dof, npe=int(space.npe), degree_index=int(degree == "p2"),
        mirror_index=_VIZ1_MIRRORS.index(mirror), coefficient_index=int(coefficient == "jump"),
        t=t, k_done=k_now, k_remaining=n_elem - k_now,
        local_dof_min=int(np.min(local_dofs)), local_dof_max=int(np.max(local_dofs)), local_elem_index=e_idx,
        residual_now=_fin(float(info["res_hist"][k_now])),
        residual_rel_now=_fin(float(info["res_hist"][k_now]) / max(float(np.max(np.abs(info["K_full"]))), 1e-300)),
        symmetry_now=_fin(float(info["sym_hist"][k_now])),
        symmetry_max_over_history=_fin(float(np.max(info["sym_hist"]))),
        entries_total=int(n_dof * n_dof), entries_pending=int(np.count_nonzero(pending > 0.5)),
        entries_pending_fraction=_fin(float(np.count_nonzero(pending > 0.5)) / float(n_dof * n_dof)),
        route_defect=_fin(info["route_defect"]), solve_defect=_fin(info["solve_defect"], -1.0),
        lambda_min_final=_fin(info["lam"]["min"], -1.0), lambda_max_final=_fin(info["lam"]["max"], -1.0),
        condition_final=_fin(info["lam"]["cond"], -1.0), eigen_exact=bool(0 < info["free"].size <= _EIG_MAX),
        route_is_exact=bool(info["route_defect"] == 0.0), route_singular=bool(not info["ok_route"]),
    )
    fr.set_strings(
        time_semantics=_TIME_SEMANTICS["assembly_animation"],
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_ERR_FLOOR, ratio=_RATIO_FLOOR),
        floor=f"{_ERR_FLOOR:g}",
        reading_note=(
            "主图看部分矩阵 K_k 的颜色累积，副图看当前单元的 ke 与它的散射目标，"
            "第三条曲线是装配一致性残差与对称性残差。"
            "后者不是合格证：换装配路线就能让它保持 0 而矩阵全错。"
        ),
        route_note=(
            "full = 每个单元把 ke 完整散射进 K；half = 只装上三角再镜像（对角只放一次）；"
            "double = 上三角里把对角线整条放进去，镜像时对角线被算了两遍。"
            "三种路线的 max|K−K^T| 都是 0.0、Cholesky 也都成功——只有 route_defect 与解分得清。"
        ),
        eigen_note=(
            f"精确特征分解只在自由度数 ≤ {_EIG_MAX} 时做（Jacobi 是稠密 O(n^3)）；"
            "对轴上画的是 Cholesky 主元比，它是条件数的廉价代理、不是条件数本身，"
            "而且半装配时矩阵本来就奇异（奇异步已用 pivot_unmeasured 掩码标出）。"
        ),
        coefficient_note=(
            "系数场 a(x) 在 x=1/π 处从 1 跳到 100：装配动画只关心「同一格被谁加过」，"
            "系数跳变让不同单元的量级差 100 倍，颜色累积因此更容易看出谁还没来。"
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 2：拖动载荷的实时求解
# --------------------------------------------------------------------------- #
def _solve_beam_cached(n_elem: int, x0: float, disc: str) -> dict[str, Any]:
    """梁：Hermite 三次（或故意用 P1 线性单元看它怎么死）。"""
    x0 = float(min(max(x0, 1e-6), 1.0 - 1e-6))
    mesh = ms.uniform_mesh_1d(n_elem)
    if disc == "p1_naive":
        space = ms.lagrange_space(mesh, 1)
        ke = np.zeros((n_elem, 2, 2), dtype=float)
        fe = asm.element_load_1d(space, lambda x: np.ones_like(np.asarray(x, dtype=float)))
        model = sv.model_1d("beam_point", x0=min(x0, 1.0 - x0))
    else:
        space = ms.hermite_space(mesh)
        ke = asm.element_beam_stiffness(space)
        fe = asm.element_point_load_beam(space, x0, 1.0)
        model = sv.model_1d("beam_point", x0=min(x0, 1.0 - x0))
    # Hermite 的简支 = 两端 w=0（转角自由，w''=0 是自然边界条件）；
    # Lagrange 空间上只有 dirichlet/neumann 两种，端点全钉就是 dirichlet。
    fixed = space.boundary_dofs("simply_supported" if space.is_hermite else "dirichlet")
    sol = sv.solve_1d(space, ke, fe, fixed, np.zeros(fixed.size, dtype=float))[1]
    if sol.ok:
        err = sv.error_norms_1d(space, sol.u, model.exact, model.dexact, d2exact=model.d2exact, want_h2=True)
    else:
        err = {k: float("inf") for k in ("l2", "h1", "energy", "h2")}
    nodes = mesh.x
    # Hermite 每个顶点两个自由度（w, w'）：取 0::2 才是节点位移。
    # 写成 sol.u[space.dof_node] 会得到 2·n_node 个数（dof→顶点是满射不是单射），
    # 与 exact(nodes) 的形状对不上——这个形状 bug 在 probe2 里被抓过一次。
    u_node = np.asarray(sol.u[0::2], dtype=float) if space.is_hermite else np.asarray(sol.u, dtype=float)
    exact_node = model.exact(nodes)
    cond = -1.0
    if sol.ok and 0 < sol.free.size <= _EIG_MAX:
        Ke = asm.assemble_1d(space, ke, fe).K
        cond = _fin(sv.system_condition(Ke[np.ix_(sol.free, sol.free)])["condition"], -1.0)
    dist = float(np.min(np.abs(nodes - x0))) if nodes.size else -1.0
    return {"space": space, "mesh": mesh, "sol": sol, "model": model, "err": err, "nodes": nodes,
            "u_node": u_node, "exact_node": exact_node, "cond": cond, "node_distance": dist,
            "on_node": bool(dist <= 1e-12), "kind": "beam"}


@functools.lru_cache(maxsize=48)
def _solve_membrane_cached(n_grid: int, element: str, distortion: float, x0: float, sigma: float) -> dict[str, Any]:
    """膜：Q1 等参四边形（可畸变）或 P1 三角形。"""
    mesh = ms.uniform_tri_mesh(n_grid, n_grid) if element == "tri" else ms.uniform_quad_mesh(n_grid, n_grid, distortion)
    model = sv.model_2d("bump", x0=float(np.clip(x0, 0.06, 0.94)), y0=0.5, sigma=sigma)
    r = sv.solve_model_2d(model, mesh)
    err_node = np.abs(r["sol"].u - model.exact(mesh.nodes))
    elem_err = np.asarray([float(np.mean(err_node[mesh.elems[e]])) for e in range(mesh.n_elem)])
    if mesh.kind == "quad":
        from numcore.pde.fem.shape import quad_jacobian_det, quad_rule

        xi, eta, _ = quad_rule(3)
        worst = 0.0
        for e in range(mesh.n_elem):
            det = quad_jacobian_det(xi, eta, mesh.nodes[mesh.elems[e]])
            worst = max(worst, float(np.max(det) / max(float(np.min(det)), 1e-300)))
        jac = worst
    else:
        jac = float(np.max(mesh.areas) / max(float(np.min(mesh.areas)), 1e-300))
    return {"mesh": mesh, "sol": r["sol"], "model": model, "err": r["error"], "err_node": err_node,
            "exact_node": model.exact(mesh.nodes), "elem_err": elem_err, "jac_ratio": jac,
            "cond": -1.0, "node_distance": -1.0, "kind": "membrane"}


@functools.lru_cache(maxsize=48)
def _load_scan_1d(n_elem: int, disc: str, n_scan: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """扫集中力位置：能量误差（锯齿）与节点最大误差（机器零）对照。"""
    xs = np.linspace(0.02, 0.98, n_scan)
    en = np.zeros(n_scan)
    nod = np.zeros(n_scan)
    for i, x0 in enumerate(xs):
        info = _solve_beam_cached(n_elem, float(x0), disc)
        # 四阶问题的能量范数是 H^2 半范  || w'' || ；error_norms_1d 的 "energy" 是 H^1 半范，
        # 对梁不对（这一条是被"节点上解精确、H^1 误差却有 0.0645"逼出来的）。
        en[i] = _fin(info["err"]["h2"], float("inf")) if info["sol"].ok else float("inf")
        nod[i] = _fin(float(np.max(np.abs(info["u_node"] - info["exact_node"]))))
    return xs, en, nod


@functools.lru_cache(maxsize=48)
def _load_scan_2d(n_grid: int, element: str, distortion: float, sigma: float, n_scan: int) -> tuple[np.ndarray, np.ndarray]:
    """扫载荷中心位置（沿 y=0.5 的横线）：二维误差同样随载荷与网格的相对位置变化。"""
    xs = np.linspace(0.1, 0.9, n_scan)
    en = np.zeros(n_scan)
    for i, x0 in enumerate(xs):
        info = _solve_membrane_cached(n_grid, element, distortion, float(x0), sigma)
        en[i] = _fin(info["err"]["energy"], float("inf"))
    return xs, en


def _curve_1d(space: ms.Space1D, u: np.ndarray, per_elem: int = 9) -> np.ndarray:
    """把 :math:`u_h` 在单元内逐点求值，得到一条能看出单元内部形状的曲线 (x, u)。"""
    from numcore.pde.fem.solve import _eval_uh_1d

    xs = space.mesh.x
    pts = []
    for e in range(space.n_elem):
        a, b = float(xs[space.mesh.elems[e, 0]]), float(xs[space.mesh.elems[e, 1]])
        xq = np.linspace(a, b, per_elem, endpoint=(e == space.n_elem - 1))
        vals = _eval_uh_1d(space, u, e, xq)[0]
        pts.append(np.column_stack([xq, np.asarray(vals, dtype=float)]))
    return np.concatenate(pts, axis=0)


def _viz_deform(p: Mapping[str, Any], t: float) -> Frame:
    model_kind = str(p["model"])
    x0 = float(p["x0"])
    n_elem = int(p["n_elem"])
    n_grid = int(p["n_grid"])
    element = str(p["element"])
    distortion = float(p["distortion"])
    sigma = float(p["sigma"])
    disc = str(p["discretization"])
    lam = max(0.0, min(float(t), 1.0))
    fr = Frame()

    if model_kind == "beam":
        info = _solve_beam_cached(n_elem, x0, disc)
        space: ms.Space1D = info["space"]
        mesh = info["mesh"]
        sol = info["sol"]
        singular = bool(not sol.ok)
        u_node = np.asarray(info["u_node"], dtype=float) * lam if not singular else np.zeros(mesh.n_node)
        exact_node = np.asarray(info["exact_node"], dtype=float) * lam
        curve = _curve_1d(space, sol.u) if not singular else np.column_stack([mesh.x, np.zeros(mesh.n_node)])
        curve = np.column_stack([curve[:, 0], curve[:, 1] * lam])
        xs_scan, en_scan, nod_scan = _load_scan_1d(n_elem, disc, _SCAN_1D)
        lam_axis = np.linspace(0.0, 1.0, 11)

        fr.add("mesh_xy", np.column_stack([mesh.x, np.zeros(mesh.n_node)]), unit="1", label="网格顶点 (x, 0)")
        fr.add("elements", np.asarray(space.elems, dtype=np.int64), unit="1", label="单元 → 自由度编号")
        fr.add("solution_curve", curve, unit="1", label="单元内插值曲线 (x, w)（单元内部形状看得见）")
        fr.add("solution", u_node, unit="1", label="节点位移 w(x_i)")
        fr.add("solution_exact", exact_node, unit="1", label="节点上的精确解 w")
        fr.add("rotation", (np.asarray(sol.u[1::2], dtype=float) * lam if space.is_hermite
                            else np.zeros(space.n_dof)), unit="1",
               label="转角 w'(x_i)（Hermite 的第二个自由度；P1 没有，恒 0）")
        fr.add("error", _floor(np.abs(u_node - exact_node)), unit="1",
               label="节点上的 |w_h − w|（对数轴；-1 未在图上时仍要如实给值）")
        fr.add("error_energy_scan", _floor(en_scan), unit="1",
               label="扫载荷位置的 H^2 半范误差（梁的能量范数，对数轴）")
        fr.add("nodal_error_scan", _floor(nod_scan), unit="1", label="扫载荷位置的节点最大误差（对数轴）")
        fr.add("scan_at_floor", (nod_scan < _ERR_FLOOR).astype(float), unit="1", label="1 = 节点误差已到地板")
        fr.add("scan_x0", xs_scan, unit="1", label="集中力位置 x0")
        fr.add("load_axis", lam_axis, unit="1", label="加载因子 λ")
        fr.add("energy_law", (lam_axis**2) * _fin(info["err"]["h2"]), unit="1",
               label="能量随 λ 的二次律 λ^2 || w−w_h || _{H^2}（线性问题的解析预期）")
        fr.add("mark_x0", np.asarray([x0]), unit="1", label="当前载荷位置")
        fr.add("mark_energy", np.asarray([_fin(info["err"]["h2"], -1.0)]), unit="1",
               label="当前能量误差（H^2 半范）")
        fr.add("singular_mask", np.full(mesh.n_node, 1.0 if singular else 0.0), unit="1",
               label="1 = 该点的解不可用（系统奇异）")
        fr.set_meta(
            model_index=0, n_elem=n_elem, n_dof=int(space.n_dof), n_free=int(sol.n_free),
            hermite=bool(space.is_hermite), disc_index=int(disc == "p1_naive"), singular=singular,
            load_factor=lam, x0=x0, node_distance=_fin(info["node_distance"], -1.0),
            load_on_node=bool(info["on_node"]), energy=lam * lam * _fin(info["err"]["h2"]),
            l2=lam * _fin(info["err"]["l2"]), h2=lam * _fin(info["err"]["h2"]),
            h1=lam * _fin(info["err"]["h1"]),
            nodal_error=lam * _fin(float(np.max(np.abs(np.asarray(info["u_node"]) - info["exact_node"]))))
            if not singular else -1.0,
            w_max=float(np.max(np.abs(u_node))),
            theta_max=(_fin(float(np.max(np.abs(sol.u[1::2]))) if space.is_hermite else -1.0, -1.0) * lam),
            condition=_fin(info["cond"], -1.0), residual=_fin(sol.residual),
            chol_min_pivot_sq=_fin(sol.chol_min_pivot_sq, -1.0),
            scan_min_error=_fin(float(np.min(en_scan))), scan_max_error=_fin(float(np.max(en_scan))),
            scan_peak_to_peak=(_fin(float(np.max(en_scan)) / max(float(np.min(en_scan)), 1e-300))
                               if np.all(np.isfinite(en_scan)) else -1.0),
            scan_at_floor_count=int(np.count_nonzero(nod_scan < _ERR_FLOOR)), scan_points=int(en_scan.size),
            scan_usable=bool(np.all(np.isfinite(en_scan))),
        )
        fr.set_strings(
            time_semantics=_TIME_SEMANTICS["solve_deform"],
            nonfinite_policy=_NONFINITE_POLICY.format(floor=_ERR_FLOOR, ratio=_RATIO_FLOOR),
            floor=f"{_ERR_FLOOR:g}",
            kind="beam",
            reading_note=(
                "副图那条上下跳的曲线是「把集中力沿梁扫一遍」的能量误差："
                "载荷正好落在节点上时误差是 1e-15 量级（Hermite 解节点精确），"
                "一落进单元内部立刻跳到 1e-3 量级；下面那条平的曲线是同一批解的节点误差，"
                "它始终在机器精度上——「节点值很准」完全推不出「解很准」。"
            ),
            singular_note=(
                "discretization=p1_naive 用 P1 线性单元解四阶梁：N''≡0 ⇒ 单元矩阵恒为零、"
                "整矩阵奇异。这里如实给出 singular=True 与 fail_reason，位移用 0 占位并置 "
                "singular_mask——不写 NaN、也不假装算出来了。"
            ),
            beam_note=(
                "梁的刚度阵是 ∫EI w''v''，边界是简支（两端 w=0，转角自由）。"
                "均布载荷下最大挠度 5q/384 且节点上精确；集中力落在节点上时位移与转角都精确。"
                "meta.load_on_node 与 meta.node_distance 把「载荷落在哪」变成可读数值。"
            ),
            fail_reason=("" if not singular else "刚度阵奇异：Cholesky 与 LU 都无法分解（P1 单元在四阶问题上恒零）"),
        )
        return fr

    info = _solve_membrane_cached(n_grid, element, distortion, x0, sigma)
    mesh: ms.Mesh2D = info["mesh"]
    sol = info["sol"]
    u_node = np.asarray(sol.u, dtype=float) * lam
    exact_node = np.asarray(info["exact_node"], dtype=float) * lam
    xs_scan, en_scan = _load_scan_2d(n_grid, element, distortion, sigma, _SCAN_2D)
    lam_axis = np.linspace(0.0, 1.0, 11)
    fr.add("mesh_xy", mesh.nodes, unit="1", label="网格顶点 (x, y)")
    fr.add("elements", np.asarray(mesh.elems, dtype=np.int64), unit="1", label="单元 → 顶点编号")
    fr.add("solution", u_node, unit="1", label="节点位移 u_h")
    fr.add("solution_exact", exact_node, unit="1", label="节点上的精确解 u")
    fr.add("error", _floor(np.abs(u_node - exact_node)), unit="1", label="节点上的 |u_h − u|（对数轴）")
    fr.add("error_element", _floor(np.asarray(info["elem_err"], dtype=float) * lam), unit="1",
           label="逐单元节点误差均值（对数轴）")
    fr.add("error_energy_scan", _floor(en_scan), unit="1", label="扫载荷中心位置的能量误差（对数轴）")
    fr.add("scan_x0", xs_scan, unit="1", label="载荷中心 x0（y 固定在 0.5）")
    fr.add("load_axis", lam_axis, unit="1", label="加载因子 λ")
    fr.add("energy_law", (lam_axis**2) * _fin(info["err"]["energy"]), unit="1",
           label="能量随 λ 的二次律 λ^2 || u−u_h || _a")
    fr.add("mark_x0", np.asarray([x0]), unit="1", label="当前载荷中心")
    fr.add("mark_energy", np.asarray([_fin(info["err"]["energy"], -1.0)]), unit="1", label="当前能量误差")
    fr.add("element_h", mesh.diameters, unit="1", label="单元直径 h_K")
    fr.add("singular_mask", np.full(mesh.n_node, 1.0 if not sol.ok else 0.0), unit="1",
           label="1 = 该点的解不可用（系统奇异）")
    fr.set_meta(
        model_index=1, n_grid=n_grid, n_node=int(mesh.n_node), n_elem=int(mesh.n_elem),
        element_index=int(element == "quad"), distortion=distortion, sigma=sigma, load_factor=lam, x0=x0,
        singular=bool(not sol.ok), energy=lam * lam * _fin(info["err"]["energy"]),
        l2=lam * _fin(info["err"]["l2"]), h1=lam * _fin(info["err"]["h1"]),
        nodal_error=lam * _fin(float(np.max(np.abs(np.asarray(sol.u) - info["exact_node"])))),
        u_max=float(np.max(np.abs(u_node))), jac_ratio=_fin(info["jac_ratio"], -1.0),
        min_area=_fin(float(mesh.min_area)), max_area=_fin(float(np.max(mesh.areas))),
        scan_min_error=_fin(float(np.min(en_scan))), scan_max_error=_fin(float(np.max(en_scan))),
        scan_peak_to_peak=_fin(float(np.max(en_scan)) / max(float(np.min(en_scan)), 1e-300)),
        scan_points=int(en_scan.size), residual=_fin(sol.residual), condition=_fin(info["cond"], -1.0),
    )
    fr.set_strings(
        time_semantics=_TIME_SEMANTICS["solve_deform"],
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_ERR_FLOOR, ratio=_RATIO_FLOOR),
        floor=f"{_ERR_FLOOR:g}",
        reading_note=(
            "拖动 x0 相当于把载荷在膜上挪：解析解跟着载荷走（制造解），"
            "所以误差曲线的变化完全来自网格与载荷的相对位置，不是靶子变了。"
            "副图把载荷沿 y=0.5 扫一遍，最大最小比就是「同一网格换一个载荷位置」的代价。"
        ),
        mesh_note=(
            "四边形是双线性等参单元：J 逐点变化，刚度阵用 2×2 Gauss 张量积；"
            "distortion>0 时内部顶点被光滑推开（边界不动），meta.jac_ratio = 单元内 "
            "max detJ / min detJ，是网格畸变的直接度量——畸变到 0.16 时它能到 109。"
        ),
        membrane_note=(
            "u=sin(pi x)sin(pi y)exp(-r^2/σ^2) 在边界上恒为零、f=-Δu 有闭式，"
            "所以「拖动载荷」这个交互不会把靶子弄丢。"
        ),
        fail_reason="" if sol.ok else "二维刚度阵奇异：Cholesky 与 LU 都无法分解",
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 3：h-加密 vs p-加密
# --------------------------------------------------------------------------- #
def _model_for(solution: str) -> sv.Model1D:
    if solution == "kink":
        return sv.model_1d("interface", xs=1.0 / 3.0, ratio=100.0)
    return sv.model_1d("smooth")


@functools.lru_cache(maxsize=32)
def _h_ladder(solution: str, hmax: int) -> dict[str, Any]:
    """h-加密阶梯：ne = 4,8,16,...（界面算例时 θ 在 1/3 与 2/3 间摆动，如实记录）。"""
    model = _model_for(solution)
    ne_list = [4 * (2**k) for k in range(hmax)]
    dof = np.zeros(len(ne_list))
    l2 = np.zeros(len(ne_list))
    en = np.zeros(len(ne_list))
    for i, ne in enumerate(ne_list):
        r = sv.solve_model(model, ms.uniform_mesh_1d(ne), 1)
        dof[i] = r["sol"].n_free
        l2[i] = _fin(r["error"]["l2"], float("inf"))
        en[i] = _fin(r["error"]["energy"], float("inf"))
    return {"model": model, "ne": np.asarray(ne_list, dtype=float), "dof": dof, "l2": l2, "energy": en}


@functools.lru_cache(maxsize=32)
def _p_ladder(solution: str, pmax: int) -> dict[str, Any]:
    """p-加密阶梯：单元素 Chebyshev–Lobatto 空间，Dirichlet 两端。"""
    model = _model_for(solution)
    ps = list(range(1, pmax + 1))
    dof = np.zeros(len(ps))
    l2 = np.zeros(len(ps))
    en = np.zeros(len(ps))
    cond = np.full(len(ps), -1.0)
    for i, p in enumerate(ps):
        space = ms.lobatto_space(p)
        ke = asm.element_bilinear_1d(space, model.a_field, model.c_field, breakpoints=model.breakpoints or None)
        fe = asm.element_load_1d(space, model.load, breakpoints=model.breakpoints or None)
        fixed = space.boundary_dofs("dirichlet")
        sol = sv.solve_1d(space, ke, fe, fixed, np.asarray([0.0, model.bc_values[1]], dtype=float))[1]
        if sol.ok:
            err = sv.error_norms_1d(space, sol.u, model.exact, model.dexact,
                                    a_field=model.a_field, c_field=model.c_field)
        else:
            err = {"l2": float("inf"), "energy": float("inf")}
        dof[i] = sol.n_free
        l2[i] = _fin(err["l2"], float("inf"))
        en[i] = _fin(err["energy"], float("inf"))
        if sol.ok and 0 < sol.n_free <= 24:
            cond[i] = _fin(sv.system_condition(ke[0][np.ix_(sol.free, sol.free)])["condition"], -1.0)
    return {"model": model, "p": np.asarray(ps, dtype=float), "dof": dof, "l2": l2, "energy": en, "cond": cond}


def _interp_at(dof: Any, err: Any, xq: float) -> float:
    """在对数-对数坐标下插值出 ``xq`` 处的误差（只在区间内插，不外推）。"""
    d = np.asarray(dof, dtype=float)
    e = np.asarray(err, dtype=float)
    if xq <= d[0]:
        return float(e[0])
    if xq >= d[-1]:
        return float(e[-1])
    return float(np.exp(np.interp(np.log(xq), np.log(d), np.log(e))))


def _viz_hp(p: Mapping[str, Any], t: float) -> Frame:
    del t
    solution = str(p["solution"])
    strategy = str(p["strategy"])
    metric = str(p["metric"])
    pmax = int(p["pmax"])
    hmax = int(p["hmax"])
    fr = Frame()
    h = _h_ladder(solution, hmax)
    pp = _p_ladder(solution, pmax)
    key = "l2" if metric == "l2" else "energy"
    eh = np.asarray(h[key], dtype=float)
    ep = np.asarray(pp[key], dtype=float)

    slope_h = _loglog_slope(h["dof"], eh)
    _, r2_h = _fit_quality(h["dof"], eh, logy=True)
    slope_alg, r2_alg = _fit_quality(pp["p"], ep, logy=True)
    slope_exp, r2_exp = _fit_quality(pp["p"], ep, logy=False)
    fit_h = eh[0] * (h["dof"] / h["dof"][0]) ** slope_h
    fit_p_exp = ep[0] * np.exp(slope_exp * (pp["p"] - pp["p"][0]))
    fit_p_alg = ep[0] * (pp["p"] / pp["p"][0]) ** slope_alg

    dof_p_last = float(pp["dof"][-1])
    dof_h_last = float(h["dof"][-1])
    # 等自由度对照：两条阶梯都取 p 阶梯的最大自由度（7）。
    # 曾经的写法把 p 的误差"插值"到 h 的末端自由度 63——那是外插（点数不够），
    # 会让图上的标记落在没有数据的横坐标上。现在两个标记都钉在同一个自由度上，
    # 另外单列 h_dof_gap = h 用满 63 个自由度仍差多少倍（这才是"省了多少自由度"）。
    err_h_at_pdof = _interp_at(h["dof"], eh, dof_p_last)
    err_p_at_hdof = _interp_at(pp["dof"], ep, dof_p_last)
    h_dof_gap = float(eh[-1]) / max(float(err_p_at_hdof), 1e-300)
    focus = h if strategy == "h" else (pp if strategy == "p" else h)

    fr.add("dof_h", h["dof"], unit="1", label="h-加密的自由度数")
    fr.add("error", _floor(eh), unit="1", label="h-加密误差（对数轴）")
    fr.add("dof_p", pp["dof"], unit="1", label="p-加密的自由度数")
    fr.add("error_p", _floor(ep), unit="1", label="p-加密误差（对数轴）")
    fr.add("error_at_floor", (eh < _ERR_FLOOR).astype(float), unit="1", label="1 = h 阶梯该点已到地板")
    fr.add("error_p_at_floor", (ep < _ERR_FLOOR).astype(float), unit="1", label="1 = p 阶梯该点已到地板")
    fr.add("p_axis", pp["p"], unit="1", label="单元阶次 p")
    fr.add("error_vs_p", _floor(ep), unit="1", label="误差 vs p（同一份数据换个横轴看）")
    fr.add("fit_h", _floor(fit_h), unit="1", label="h 阶梯的双对数拟合线")
    fr.add("fit_p_exp", _floor(fit_p_exp), unit="1", label="p 阶梯的半对数（指数）拟合线")
    fr.add("fit_p_alg", _floor(fit_p_alg), unit="1", label="p 阶梯的双对数（代数）拟合线")
    fr.add("condition_p", _cond(np.where(pp["cond"] > 0.0, pp["cond"], _COND_CEIL)), unit="1",
           label="约化刚度阵条件数 vs p（上限 = 未测量）")
    fr.add("condition_unmeasured", (pp["cond"] <= 0.0).astype(float), unit="1", label="1 = 该 p 的条件数未测量")
    fr.add("focus_dof", np.asarray(focus["dof"], dtype=float), unit="1", label="所选策略的自由度（对照用）")
    fr.add("focus_error", _floor(np.asarray(focus[key], dtype=float)), unit="1", label="所选策略的误差（对照用）")
    fr.add("mark_equal_dof_p", np.asarray([dof_p_last]), unit="1",
           label=f"等自由度对照点：{dof_p_last:g} 个自由度（p 阶梯末端；两条阶梯取同一个横坐标）")
    fr.add("mark_equal_dof_h", np.asarray([dof_p_last]), unit="1",
           label="h-加密在同一个自由度处的读数（与 p 逐点对照，不做外插）")
    fr.add("mark_h_last", np.asarray([dof_h_last]), unit="1", label=f"h 阶梯末端：{dof_h_last:g} 个自由度")
    fr.add("error_h_at_mark", _floor(np.asarray([err_h_at_pdof])), unit="1",
           label=f"等自由度（{dof_p_last:g}）处 h-加密的误差")
    fr.add("error_p_at_mark", _floor(np.asarray([err_p_at_hdof])), unit="1",
           label=f"等自由度（{dof_p_last:g}）处 p-加密的误差")
    fr.add("error_h_last_at_mark", _floor(np.asarray([eh[-1]])), unit="1",
           label=f"h 阶梯末端（{dof_h_last:g} 个自由度）的误差")

    fr.set_meta(
        solution_index=int(solution == "kink"), strategy_index=("both", "h", "p").index(strategy),
        metric_index=int(metric == "l2"), pmax=pmax, hmax=hmax, n_h=len(h["ne"]), n_p=len(pp["p"]),
        dof_h_last=dof_h_last, dof_p_last=dof_p_last, err_h_last=_fin(eh[-1]), err_p_last=_fin(ep[-1]),
        slope_h_dof=_fin(slope_h), r2_h_dof=_fin(r2_h),
        slope_p_semilog=_fin(slope_exp), r2_p_semilog=_fin(r2_exp),
        slope_p_loglog=_fin(slope_alg), r2_p_loglog=_fin(r2_alg),
        exponential_better=bool(_fin(r2_exp) > _fin(r2_alg)),
        equal_dof=dof_p_last,
        err_h_at_pdof=_fin(err_h_at_pdof), err_p_at_hdof=_fin(err_p_at_hdof),
        gain_p_over_h=_fin(err_h_at_pdof / max(err_p_at_hdof, 1e-300)),
        h_dof_gap=_fin(h_dof_gap),
        err_ratio_per_doubling_h=_fin(float(np.mean(eh[:-1] / eh[1:]))),
        cond_p_first=_fin(float(pp["cond"][0]), -1.0), cond_p_last=_fin(float(pp["cond"][-1]), -1.0),
        cond_growth=_fin(float(pp["cond"][-1]) / max(float(pp["cond"][0]), 1e-300), -1.0),
        cond_at_pmax_measured=bool(pp["cond"][-1] > 0.0),
    )
    fr.set_strings(
        time_semantics=_TIME_SEMANTICS["hp_convergence"],
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_ERR_FLOOR, ratio=_RATIO_FLOOR),
        floor=f"{_ERR_FLOOR:g}",
        reading_note=(
            "两种横轴一起看：横轴取自由度时两条阶梯可比，横轴取 p 时才能看出「指数还是代数」。"
            "同一个 p-阶梯在 smooth 上双对数拟合的 R^2 明显低于半对数，在 kink 上反过来——"
            "所以「谱收敛」这句话必须带解的正则性一起说。"
        ),
        metric_note=(
            "metric=energy 与 metric=l2 的差别不是细节：光滑解上 L2 阶 2、能量阶 1；"
            "界面算例上 L2 阶约 1、能量阶约 0.5。只报一个「收敛阶」等于没报。"
        ),
        kink_note=(
            "kink 算例的解在 x=1/3 处 C0 但不 C1：单元素 p-加密只能代数收敛。"
            "h 阶梯的 ne=4,8,16,... 会让界面在单元内的相对位置在 1/3 与 2/3 之间摆动，"
            "所以这条 h 阶梯的实测阶带一点相位噪声（断言里把 θ 固定住再测那个干净的）。"
        ),
        cond_note=(
            "p-加密不是免费的：谱条件数随 p 增长（见 condition_p 与 meta.cond_growth），"
            "自由度很少的 p=12 能到 1e-11 是靠高阶多项式，代价是条件数。"
            "高阶单元精度高、条件数差，这两句话必须同时说。"
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 4：后验指示子与自适应网格
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=32)
def _adapt_pair(ratio: float, theta: float, levels: int, n0: int) -> dict[str, Any]:
    """自适应与一致加密各跑一条阶梯（同一算例、同一指示子、同一求解器）。"""
    model = sv.model_1d("interface", xs=1.0 / 3.0, ratio=ratio)
    mesh0 = ms.uniform_mesh_1d(n0)
    kwargs = dict(degree=1, exact=model.exact, dexact=model.dexact, f_field=model.load,
                  a_field=model.a_field, fixed_values=(0.0, model.bc_values[1]))
    adap = ad.adaptive_history(mesh0, theta=theta, levels=levels, **kwargs)
    # 一致加密是指数增长的：n0=16、levels=8 时末端有 4096 个单元。
    # 截断到 _UNIFORM_MAX_ELEM（自适应那一侧的自由度远在这段区间内），
    # 并把"截断了几层"如实写进 meta —— 悄悄截断会让比较区间变得不可见。
    max_levels = max(1, int(math.floor(math.log2(_UNIFORM_MAX_ELEM / max(n0, 1)))))
    levels_u = min(levels, max_levels)
    unif = ad.uniform_history(mesh0, levels=levels_u, **kwargs)
    return {"model": model, "adaptive": adap, "uniform": unif, "levels_uniform": levels_u,
            "uniform_truncated": bool(levels_u < levels),
            "compare": ad.compare_strategies(adap, unif)}


def _viz_adaptive(p: Mapping[str, Any], t: float) -> Frame:
    theta = float(p["theta"])
    strategy = str(p["strategy"])
    ratio = float(p["ratio"])
    levels = int(p["levels"])
    n0 = int(p["n0"])
    info = _adapt_pair(ratio, theta, levels, n0)
    model = info["model"]
    adap, unif = info["adaptive"], info["uniform"]
    hist = adap if strategy == "adaptive" else unif
    t = max(0.0, min(float(t), 1.0))
    lev = int(round(t * levels))
    lv = hist[lev]
    mesh = lv.mesh
    eta = np.asarray(lv.eta, dtype=float)
    true_err = np.asarray(lv.true_err, dtype=float)
    # 逐单元效率指数只在"真误差不是舍入噪声"的单元上有意义。
    # 门槛取最大单元误差的 _REL_ERR_FLOOR 倍：低于它的单元，比值由舍入决定，
    # 报出来会是 1e-14 或 1e14 这种噪声（曾经就报出过 spread=4e13）。
    err_scale = max(float(np.max(true_err)), _RATIO_FLOOR)
    eta_scale = max(float(np.max(eta)), _RATIO_FLOOR)
    undef_thresh = max(_RATIO_FLOOR, _REL_ERR_FLOOR * err_scale)
    eta_thresh = _REL_ERR_FLOOR * eta_scale
    # 两类"没有意义"：真误差本身是舍入噪声；或者指示子在这个单元上恰好是噪声
    # （后者才是真正值得报的：指示子对它有误差却什么都没说）。两类分开记账。
    noise_err = true_err <= undef_thresh
    blind = (~noise_err) & (eta <= eta_thresh)
    undefined = (noise_err | blind).astype(float)
    ratio_elem = np.where(undefined > 0.5, _RATIO_FLOOR, eta / np.maximum(true_err, _RATIO_FLOOR))
    defined = undefined < 0.5
    ratio_pos = ratio_elem[defined]
    ratio_pos = ratio_pos[ratio_pos > 0.0]
    tot_err2 = max(float(np.sum(true_err**2)), 1e-300)
    undef_error_share = float(np.sum(true_err[noise_err] ** 2) / tot_err2)
    blind_error_share = float(np.sum(true_err[blind] ** 2) / tot_err2)
    corr = _log_corr(eta, true_err, _RATIO_FLOOR)
    fr = Frame()

    fr.add("mesh_x", mesh.x, unit="1", label="当前网格顶点")
    fr.add("elements", np.asarray(mesh.elems, dtype=np.int64), unit="1", label="单元 → 顶点编号")
    fr.add("element_h", mesh.h, unit="1", label="单元长度 h_K")
    fr.add("element_eta", _floor(eta), unit="1", label="后验指示子 η_K（对数轴）")
    fr.add("error", _floor(true_err), unit="1", label="逐单元真误差  || e || _{a,K}（对数轴，只有拿解析解才算得出）")
    fr.add("element_ratio", ratio_elem, unit="1",
           label=f"效率指数 η_K/真误差（真误差 ≤{_REL_ERR_FLOOR:g}×最大单元误差时视为噪声，用占位值）")
    fr.add("ratio_undefined", undefined, unit="1",
           label=f"1 = 该单元真误差在最大单元误差的 {_REL_ERR_FLOOR:g} 倍以下，比值由舍入决定")
    fr.add("element_eta_share", eta**2 / max(float(np.sum(eta**2)), 1e-300), unit="1",
           label="η_K^2 占 Ση^2 的份额（Doerfler 标记就是按它累计的）")
    fr.add("solution", np.asarray(lv.u, dtype=float), unit="1", label="节点解 u_h")
    fr.add("solution_exact", model.exact(mesh.x), unit="1", label="节点上的精确解 u")
    fr.add("level_axis", np.arange(levels + 1, dtype=float), unit="1", label="加密层数 ℓ")
    fr.add("dof_adaptive", np.asarray([h.n_dof for h in adap], dtype=float), unit="1", label="自适应：自由度数")
    fr.add("error_adaptive", _floor(np.asarray([h.error_energy for h in adap], dtype=float)), unit="1",
           label="自适应：能量误差（对数轴）")
    fr.add("dof_uniform", np.asarray([h.n_dof for h in unif], dtype=float), unit="1", label="一致加密：自由度数")
    fr.add("error_uniform", _floor(np.asarray([h.error_energy for h in unif], dtype=float)), unit="1",
           label="一致加密：能量误差（对数轴）")
    fr.add("effectivity", np.asarray([h.effectivity_total for h in adap], dtype=float), unit="1",
           label="自适应：总效率指数 √(Ση^2)/√(Σ || e || ^2)（对轴；等于 meta.eta_norm/error_now）")
    fr.add("error_at_floor", (np.asarray([h.error_energy for h in adap], dtype=float) < _ERR_FLOOR).astype(float),
           unit="1", label="1 = 该层误差已到地板")
    fr.add("mark_level", np.asarray([float(lev)]), unit="1", label="当前层")
    cmp_ratio = np.asarray(info["compare"]["ratio_adaptive_over_uniform"], dtype=float)
    fr.add("compare_ratio", _floor(cmp_ratio), unit="1", label="同自由度下 自适应/一致 的误差比（对数轴）")
    fr.add("compare_unmeasured", (~np.isfinite(cmp_ratio)).astype(float), unit="1",
           label="1 = 该层的自由度落在一致加密区间外，比值无法插值")
    fr.add("compare_index", np.arange(cmp_ratio.size, dtype=float), unit="1", label="同自由度比较的采样序号")

    fr.set_meta(
        strategy_index=int(strategy == "uniform"), ratio=ratio, theta=theta, levels=levels, n0=n0, level=lev,
        n_elem=int(mesh.n_elem), n_dof=int(lv.n_dof), n_node=int(mesh.n_node),
        n_free=int(lv.n_dof) - 2,
        min_h=_fin(float(mesh.min_h)), max_h=_fin(float(mesh.max_h)),
        h_ratio=_fin(float(mesh.max_h) / max(float(mesh.min_h), 1e-300)),
        eta_sum=_fin(float(np.sum(eta))), eta_norm=_fin(float(np.sqrt(np.sum(eta * eta)))),
        eta_max=_fin(float(np.max(eta))),
        eta_share_max=_fin(float(np.max(eta) ** 2 / max(float(np.sum(eta**2)), 1e-300))),
        eta_over_half=int(np.count_nonzero(eta > 0.5 * float(np.max(eta)))),
        error_now=_fin(float(lv.error_energy)), error_l2_now=_fin(float(lv.error_l2)),
        effectivity_total=_fin(float(lv.effectivity_total)),
        effectivity_min_defined=_fin(float(np.min(ratio_pos)) if ratio_pos.size else -1.0),
        effectivity_max_defined=_fin(float(np.max(ratio_pos)) if ratio_pos.size else -1.0),
        effectivity_spread=_fin(float(np.max(ratio_pos)) / max(float(np.min(ratio_pos)), 1e-300))
        if ratio_pos.size else -1.0,
        effectivity_median=_fin(float(np.median(ratio_pos)) if ratio_pos.size else -1.0),
        effectivity_q25=_fin(float(np.percentile(ratio_pos, 25.0)) if ratio_pos.size else -1.0),
        effectivity_q75=_fin(float(np.percentile(ratio_pos, 75.0)) if ratio_pos.size else -1.0),
        effectivity_defined_count=int(ratio_pos.size),
        ratio_undefined_count=int(np.count_nonzero(undefined > 0.5)),
        indicator_noise_count=int(np.count_nonzero(noise_err)),
        indicator_blind_count=int(np.count_nonzero(blind)),
        indicator_zero_count=int(np.count_nonzero((eta <= 0.0) & ~noise_err)),
        error_share_top3=_fin(float(np.sum(np.sort(true_err**2)[::-1][:3]) / tot_err2)),
        eta_share_top3=_fin(float(np.sum(np.sort(eta**2)[::-1][:3]) / max(float(np.sum(eta**2)), 1e-300))),
        undefined_error_share=_fin(undef_error_share),
        blind_error_share=_fin(blind_error_share), log_corr=_fin(corr),
        marked_fraction=_fin(float(lv.marked_fraction)), best_ratio=_fin(float(info["compare"]["best_ratio"]), -1.0),
        median_ratio=_fin(float(info["compare"]["median_ratio"]), -1.0),
        ratio_at_equal_dof=_fin(float(cmp_ratio[lev]), -1.0) if cmp_ratio.size > lev else -1.0,
        uniform_levels_used=int(info["levels_uniform"]), uniform_truncated=bool(info["uniform_truncated"]),
        uniform_dof_last=_fin(float(unif[-1].n_dof)), adaptive_dof_last=_fin(float(adap[-1].n_dof)),
        uniform_error_last=_fin(float(unif[-1].error_energy)), adaptive_error_last=_fin(float(adap[-1].error_energy)),
        kink_element_index=int(np.argmin(mesh.h)),
        kink_error_share=_fin(float(true_err[int(np.argmin(mesh.h))] ** 2 / max(float(np.sum(true_err**2)), 1e-300))),
        error_now_equals_eta_norm_ratio=_fin(float(np.sqrt(np.sum(eta * eta))) / max(float(lv.error_energy), 1e-300)),
    )
    fr.set_strings(
        time_semantics=_TIME_SEMANTICS["adaptive_mesh"],
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_ERR_FLOOR, ratio=_RATIO_FLOOR),
        floor=f"{_ERR_FLOOR:g}",
        reading_note=(
            "主图把 η_K 与逐单元真误差画在一起：指示子只用 u_h 与系数场就能算出来，"
            "真误差要用解析解——两者不成比例，自适应就会往错的地方加密（meta.log_corr 是它们的对数相关系数）。"
            "副图是同样自由度下的两条误差阶梯：一致加密是 N^{-1/2}，自适应近似几何下降。"
        ),
        indicator_note=(
            "η_K^2 = h_K^2 || R_K || ^2 + ½Σ_F h_F || [[a u_h']] || ^2，R_K 用单元内的二阶导（P1 上为 0，"
            "P2 上按 2/h^2(u_i−2u_m+u_{i+1}) 精确给），跳跃项按每个求积点上的 a 取值。"
            "效率指数是一个分布：总量逐层摆动，逐单元的 max/min 见 meta.effectivity_spread。"
        ),
        theta_note=(
            "Doerfler 标记：把 η 从大到小累计到 θ·Ση^2 为止（再补一层邻接闭包保证网格协调）。"
            "θ=1 就是一致加密——它看着最保险，其实放弃了自适应。"
        ),
        uniform_note=(
            f"一致加密阶梯在单元数超过 {_UNIFORM_MAX_ELEM} 时停止加倍（meta.uniform_levels_used 与 "
            "uniform_truncated 如实记录）；自适应的自由度远小于它，比较区间不受影响。"
        ),
        floor_note=(
            f"真误差 ≤{_RATIO_FLOOR:g} 的单元上效率指数没有意义，用 {_RATIO_FLOOR:g} 占位"
            "并置 ratio_undefined 掩码；同自由度比值在区间外无法插值时用 compare_unmeasured 掩码——"
            "都比写一个 0 或 NaN 诚实。"
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# MANIFEST
# --------------------------------------------------------------------------- #
MANIFEST: dict[str, Any] = {
    "chapter": "4.4",
    "title": "有限元方法",
    "part": "IV",
    "summary": (
        "有限元的全部结论都挂在两个对象上：装配出来的矩阵 K，和用 K 解出来的 u_h。"
        "这一章把四条证据链并排放：单元矩阵怎么变成 K（三种装配路线里只有一种是对的）、"
        "K 到底是不是对称正定（对称残差为 0 能不能当合格证）、误差在哪个范数下以什么阶衰减"
        "（同一个解，L2 给 2、能量给 1；界面算例给 1 和 0.5）、以及后验指示子到底有多准"
        "（效率指数是分布，不是常数）。三个最容易被骗的地方都标出来了："
        "对称+正定+Cholesky 成功同时成立也可能是错的矩阵、「谱收敛」在尖角解上会退化成代数、"
        "比较两种加密策略必须固定自由度而不是层数。"
    ),
    "vizzes": [
        # ------------------------------------------------------------------ #
        {
            "id": "assembly_animation",
            "title": "装配动画：单元矩阵怎么变成全局矩阵",
            "description": (
                "主图是逐单元累积出来的 K_k（时间轴是「已装入几个单元」），"
                "副图画当前单元的局部矩阵与它的自由度编号，"
                "第三条曲线是装配一致性残差 max|K_k−K| 与对称性残差 max|K_k−K_k^T|。"
                "mirror=double 用上三角镜像装配、对角线放了两遍：两条残差都看不出问题"
                "（对称残差恰好 0.0、条件数反而更小），只有 route_defect 与解能。"
            ),
            "dataSource": "backend",
            "params": [
                {"name": "n_elem", "label": "单元数", "type": "int", "min": 3, "max": 20, "step": 1,
                 "default": 8, "critical": [], "criticalNote": ""},
                {"name": "degree", "label": "单元次数", "type": "enum", "options": ["p1", "p2"],
                 "default": "p1", "critical": [], "criticalNote": ""},
                {"name": "coefficient", "label": "系数场 a(x)", "type": "enum", "options": ["const", "jump"],
                 "default": "const", "critical": [], "criticalNote": ""},
                {
                    "name": "mirror", "label": "装配路线", "type": "enum",
                    "options": ["full", "half", "double"], "default": "full", "critical": [],
                    "criticalNote": (
                        "double 是错误路线（上三角里把对角线整条放进去了，镜像时对角线被加了两遍）："
                        "它仍然对称、仍然正定、Cholesky 仍然成功，但矩阵不是那个矩阵。"
                    ),
                },
            ],
            "time": {"mode": "scrub", "tMax": 1.0, "dt": 0.05, "label": "装配进度（已装入的单元比例）"},
            "requiredAux": ["residual", "symmetry_residual"],
            "axes": [
                {"id": "matrix", "autoscaled": False,
                 "title": "全局刚度阵 K_k（固定窗口，灰=未定型）", "ylabel": "自由度 i"},
                {"id": "local", "autoscaled": True, "title": "当前单元矩阵 ke（色标自适应）",
                 "ylabel": "局部自由度"},
                {"id": "residuals", "autoscaled": False,
                 "title": "残差与主元比（对数，地板 1e-17）", "ylabel": "max|ΔK|",
                 "scale": "log"},
            ],
            "pitfalls": (
                "第一，对称性残差不是合格证：P1 完整装配与两种镜像路线的 max|K−K^T| 都是 0.0，"
                "但 double 路线的 max|K−K_full| 是 16.0、λ_min 从 1.217927 变成 17.217927、"
                "条件数还从 25.274 降到 2.717——所有「体检指标」都变好了，矩阵却是错的。"
                "第二，装配进度不能用「非零元个数」看：离对角线远的格子很早就定型，"
                "对角线的最后一个贡献者最晚到（entry_done_at 会告诉你谁在等）。"
                "第三，P2 的对称残差不是 0 而是 3.6e-15，来源是单元矩阵自己的矩阵乘，"
                "不是散射顺序——把散射反序做一遍，残差一模一样。"
            ),
            "theory": [
                "K = Σ_K R_K^T ke_K R_K（R_K 是单元自由度到全局自由度的散射矩阵）；装配是线性累加，与顺序无关",
                "v^TKv = Σ_K ∫_K a|∇v_h|^2 ≥ 0；等号当且仅当 v_h 在每个连通片上为常数，Dirichlet 消元后严格正定",
                "上三角镜像装配 K ← K + K^T 只在「对角线只放一次」时等于完整散射；对角线放两遍等于给 K 加了一个对角阵",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "solve_deform",
            "title": "拖动载荷：梁与膜的实时求解与误差锯齿",
            "description": (
                "主图是当前载荷位置下的变形（时间轴是准静态加载因子 λ）。"
                "副图把载荷沿梁/膜扫一遍，画出能量误差关于载荷位置的曲线："
                "集中力正好落在节点上时误差是 1e-15 量级，落进单元内部立刻跳到 1e-3 量级。"
                "同一副图上还有一条始终贴着地板的节点误差曲线——它证明「节点值准」"
                "完全推不出「解准」。"
            ),
            "dataSource": "backend",
            "params": [
                {"name": "model", "label": "模型", "type": "enum", "options": ["beam", "membrane"],
                 "default": "beam", "critical": [], "criticalNote": ""},
                {
                    "name": "x0", "label": "载荷位置 x0", "type": "float", "min": 0.02, "max": 0.98,
                    "step": 0.01, "default": 0.5, "critical": [0.5], "beyond": "above",
                    "criticalNote": (
                        "x0 落在节点上还是单元内部，会让集中力算例的能量误差差 11 个数量级"
                        "（实测 5.7e-15 对 1.25e-1）。拖动这条滑杆就是在扫这个锯齿。"
                    ),
                },
                {"name": "n_elem", "label": "梁的单元数", "type": "int", "min": 2, "max": 32, "step": 1,
                 "default": 8, "critical": [], "criticalNote": ""},
                {
                    "name": "discretization", "label": "梁的离散", "type": "enum",
                    "options": ["hermite", "p1_naive"], "default": "hermite", "critical": [],
                    "criticalNote": (
                        "p1_naive 用 P1 线性单元解四阶梁：N''≡0 让单元矩阵恒为零，整矩阵奇异。"
                        "帧会如实给出 singular=True，而不是写 NaN。"
                    ),
                },
                {"name": "n_grid", "label": "膜的网格 n×n", "type": "int", "min": 4, "max": 20, "step": 2,
                 "default": 12, "critical": [], "criticalNote": ""},
                {"name": "element", "label": "膜单元", "type": "enum", "options": ["tri", "quad"],
                 "default": "quad", "critical": [], "criticalNote": ""},
                {
                    "name": "distortion", "label": "四边形畸变幅度", "type": "float", "min": 0.0, "max": 0.2,
                    "step": 0.01, "default": 0.0, "critical": [0.16], "beyond": "above",
                    "criticalNote": (
                        "畸变到 0.16 时单元内 max detJ / min detJ 最大到 109.17、最小面积 7.94e-5，"
                        "网格接近自交——误差曲线仍然平滑下降，但「冻结 J」与逐点 J 的解已经差 49%。"
                    ),
                },
                {"name": "sigma", "label": "膜载荷宽度 σ", "type": "float", "min": 0.06, "max": 0.4,
                 "step": 0.01, "default": 0.18, "critical": [], "criticalNote": ""},
            ],
            "time": {"mode": "scrub", "tMax": 1.0, "dt": 0.05, "label": "加载因子 λ（准静态）"},
            "requiredAux": ["error", "error_energy_scan"],
            "axes": [
                {"id": "deformed", "autoscaled": True,
                 "title": "变形后的构型（自动缩放）", "ylabel": "位移"},
                {"id": "scan", "autoscaled": False,
                 "title": "扫载荷位置的误差（对数，地板 1e-17）", "ylabel": "误差",
                 "scale": "log"},
            ],
            "pitfalls": (
                "第一，「加密网格误差一定下降」在点载荷算例上是错的：误差随载荷位置呈锯齿，"
                "载荷落在节点上时 P1/Hermite 的节点值精确（实测 5.7e-15），"
                "落进单元内部立刻回到 1e-3 量级——换网格只是换了锯齿的相位。"
                "第二，节点误差始终 ≤4.4e-16（一维 Green 函数分段线性），"
                "所以任何只看节点的误差检查都会得出「解完全正确」的结论。"
                "第三，四阶问题用 P1 单元时单元矩阵恒零、系统奇异，"
                "这时该如实报告奇异，而不是给一个被 NaN 污染的解。"
            ),
            "theory": [
                "梁的能量范数是 H^2 半范  || w'' || _{L^2}：Hermite 三次是 C^1 协调元，P1 不协调（且 N''≡0）",
                "一维 Green 函数分段线性 ⟹ 节点上 u_h=u（对任意网格），误差全部关在单元内部",
                "等参四边形：∇φ=J^{-⊤}∇_ξφ、∫_K=∫_K̂|det J|；J 逐点取还是冻结在中心，是两个不同的变分问题",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "hp_convergence",
            "title": "h-加密与 p-加密：阶是（方法，解的正则性）的函数",
            "description": (
                "两条误差阶梯画在一起：h-加密（一致细分，P1）与 p-加密（单元素高阶）。"
                "横轴可以换成自由度，也可以换成 p。同一份 p-加密代码，"
                "在光滑解上半对数拟合 R^2=0.970（指数收敛），"
                "在带尖角的解上掉到 0.759 而双对数拟合是 0.928（退化成代数收敛）。"
                "副图是 p-阶梯的条件数——高阶单元不是免费的。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "solution", "label": "解的正则性", "type": "enum",
                    "options": ["smooth", "kink"], "default": "smooth", "critical": [],
                    "criticalNote": (
                        "kink 是系数在 x=1/3 处跳 100 倍的分段线性解（C0 不 C1）："
                        "p-加密在它上面不再指数收敛，这正是「谱收敛」被过度推广的地方。"
                    ),
                },
                {"name": "strategy", "label": "聚焦的策略", "type": "enum", "options": ["both", "h", "p"],
                 "default": "both", "critical": [], "criticalNote": ""},
                {
                    "name": "metric", "label": "误差范数", "type": "enum", "options": ["energy", "l2"],
                    "default": "energy", "critical": [],
                    "criticalNote": (
                        "同一个解、同一张网格，换范数就换阶：光滑解 L2 是 2、能量是 1；"
                        "界面算例是 1 与 0.5。范数必须和阶一起报。"
                    ),
                },
                {"name": "pmax", "label": "p 阶梯上限", "type": "int", "min": 2, "max": 12, "step": 1,
                 "default": 8, "critical": [], "criticalNote": ""},
                {"name": "hmax", "label": "h 阶梯级数", "type": "int", "min": 2, "max": 7, "step": 1,
                 "default": 5, "critical": [], "criticalNote": ""},
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error", "error_p"],
            "axes": [
                {"id": "ladder", "autoscaled": False, "title": "误差 vs 自由度（双对数）",
                 "ylabel": "误差", "scale": "log"},
                {"id": "vs_p", "autoscaled": False, "title": "误差 vs p（半对数）",
                 "ylabel": "误差", "scale": "log"},
                {"id": "condition", "autoscaled": True, "title": "条件数 vs p（自动缩放）",
                 "ylabel": "cond", "scale": "log"},
            ],
            "pitfalls": (
                "第一，「p-加密指数收敛」只在解足够光滑时成立：光滑解半对数 R^2=0.970、"
                "p=12 时误差 2.6e-11；同一个代码在尖角解上 p=12 还有 0.59，"
                "双对数拟合（R^2=0.928）反而优于半对数（0.759）——那是代数收敛。"
                "第二，比较必须在同样自由度下：p-加密用 13 个自由度到 1e-11，"
                "按「加密层数」比较会让它显得免费；副图的条件数说明它换来的代价。"
                "第三，只报一个范数等于没报：同一条阶梯换成 L2 范数，阶会翻倍。"
            ),
            "theory": [
                "Cea + Bramble–Hilbert： || u−u_h || _{H^1} ≤ C h^{min(p,s−1)} || u || _{H^s}，s 是解的正则性",
                "光滑解（s 很大）上 p-加密是 e^{−cp}；尖角解（s=3/2−ε）上退化为代数收敛",
                "Aubin–Nitsche 对偶：L2 阶 = 2×能量阶，但要求对偶问题有 H^2 正则性——系数跳变时只有 H^{3/2}",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "adaptive_mesh",
            "title": "后验指示子与自适应网格：η 是分布，不是常数",
            "description": (
                "时间轴是加密层数。主图是当前网格上的逐单元指示子 η_K 与逐单元真误差"
                "（真误差只有拿解析解才算得出，指示子只用 u_h 与系数场）；"
                "副图是自适应与一致加密在同样自由度下的能量误差，以及效率指数随层数的摆动。"
                "界面算例上自适应网格自己找到了 x=1/3 并把 h 压到 4.9e-4，"
                "同自由度误差是一致加密的 0.125 倍。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "theta", "label": "Doerfler 标记比例 θ", "type": "float", "min": 0.1, "max": 1.0,
                    "step": 0.05, "default": 0.5, "critical": [1.0], "beyond": "above",
                    "criticalNote": (
                        "θ=1 时所有单元都被标记，自适应退化成一致加密——用它做对照，"
                        "同样的自由度下误差会差 8 倍。"
                    ),
                },
                {"name": "strategy", "label": "展示的策略", "type": "enum",
                 "options": ["adaptive", "uniform"], "default": "adaptive", "critical": [], "criticalNote": ""},
                {"name": "ratio", "label": "系数反差 a₂/a₁", "type": "float", "min": 10.0, "max": 10000.0,
                 "step": 10.0, "default": 100.0, "critical": [], "criticalNote": ""},
                {"name": "levels", "label": "加密层数", "type": "int", "min": 1, "max": 8, "step": 1,
                 "default": 6, "critical": [], "criticalNote": ""},
                {"name": "n0", "label": "初始单元数", "type": "int", "min": 2, "max": 16, "step": 2,
                 "default": 4, "critical": [], "criticalNote": ""},
            ],
            "time": {"mode": "scrub", "tMax": 1.0, "dt": 0.1, "label": "加密层数 ℓ"},
            "requiredAux": ["element_eta", "error"],
            "axes": [
                {"id": "mesh", "autoscaled": False, "title": "当前网格与指示子 η_K（固定 0–1）",
                 "ylabel": "η_K 归一"},
                {"id": "indicator", "autoscaled": False,
                 "title": "η_K 与真误差（对数，地板 1e-17）", "ylabel": "误差/指示子", "scale": "log"},
                {"id": "history", "autoscaled": False,
                 "title": "能量误差 vs 自由度（双对数）", "ylabel": "能量误差", "scale": "log"},
            ],
            "pitfalls": (
                "第一，指示子不是一个常数：效率指数总量在 1.24–1.92 之间逐层摆动，"
                "逐单元的 max/min 比值更大——拿一次测量的效率指数当「可靠常数」是过度推广。"
                "第二，误差小不等于网格好看：自适应最终网格的 max h/min h 在本帧能到 256"
                "（最小单元 9.8e-4，正好压在界面上；断言里 levels=9 时是 512），"
                "按「网格质量」的常规指标看它很差，但同自由度误差是一致加密的 0.125 倍。"
                "第三，θ=1 看着像「最保险的加密」其实是放弃自适应："
                "θ=1 就是一致加密，在奇点算例上自由度要多花 60 倍。"
            ),
            "theory": [
                "η_K^2 = h_K^2 || R_K || ^2 + ½Σ_F h_F || [[a u_h']] || ^2，R_K = f + (a u_h')' − c u_h（P1 上 R_K=f−cu_h）",
                "Doerfler 标记：Σ_{K∈M} η_K^2 ≥ θ^2Σ_K η_K^2，再取一层邻接闭包保证网格协调",
                "界面算例的能量误差 ~ h_K^{1/2}：自适应每两级减半（几何），一致加密是 N^{−1/2}（代数）",
            ],
        },
    ],
}


# --------------------------------------------------------------------------- #
# 调度
# --------------------------------------------------------------------------- #
_DISPATCH = {
    "assembly_animation": _viz_assembly,
    "solve_deform": _viz_deform,
    "hp_convergence": _viz_hp,
    "adaptive_mesh": _viz_adaptive,
}


def compute(viz_id: str, params: Mapping[str, Any], t: float = 0.0) -> Frame:
    """按 ``viz_id`` 分发到四个可视化之一（帧里的数学全部来自 numcore）。"""
    try:
        fn = _DISPATCH[str(viz_id)]
    except KeyError:
        raise KeyError(f"4.4 章没有可视化 {viz_id!r}（可用：{sorted(_DISPATCH)}）") from None
    return fn(params, float(t))
