r"""4.1 章：抛物型方程与时间推进（4 个可视化）。

帧只做搬运：网格、算子、theta 族推进、精确解、谱半径测量都来自 :mod:`numcore.pde.heat`，
放大因子与稳定域来自 :mod:`numcore.pde.vonneumann`，本文件不另写一份数学。
四个可视化的分工是"**同一段代码，换一个地方看它会不会骗你**"：

1. ``stability_threshold`` —— 招牌。主图是当前参数下**真跑出来**的解曲线，副图是放大因子
   :math:`|G(\varphi)|` 与实测谱半径。同一条 :math:`r` 线上，显式格式可以"看起来好好的"
   （纯正弦初值根本不激发高频模态），换个噪声初值就炸；而放大因子曲线一眼就能看出越界。
   时间轴是真实时间 :math:`t`，向右拖解跟着往前走。
2. ``convergence_order`` —— 双对数误差阶梯。``target=space`` 固定 :math:`r` 加密网格，
   ``target=time`` 固定网格缩小 :math:`\Delta t`——后者必须换靶子：拿连续解当靶子，
   CN 的实测阶是 :math:`-0.0004`（空间误差把它盖住了），换成半离散解才是 2.0000。
   两个数字都写在 meta 里，谁都能自己验。
3. ``numerical_dissipation`` —— 数值耗散。主图（对数纵轴）把三条曲线画在一起：
   真解的 :math:`e^{-r\varphi^2n}`、半离散的 :math:`e^{-\lambda_k t}`、以及**从真的解里
   分解出来的**逐模态存活率。后者与 :math:`|G|^n` 逐位重合（实测差 ≤ 1e-14），
   与真解的差就是耗散误差；副图是这条差随 :math:`\varphi` 的走向，符号由
   :math:`r/12+r^2(\theta-\frac12)` 决定。
4. ``boundary_conservation`` —— 边界条件。三种边界的总热量曲线画在同一副轴上：
   Neumann/周期是**贴着 0 的一条直线**（实测漂移 1e-15，舍入级），
   Dirichlet 是斜着往下掉的线（解析流失量 :math:`\frac2\pi(1-e^{-\pi^2t})`）。

帧契约（与 1.2 / 3.1 / 3.2 章一致）：``meta`` 只放有限标量（``allow_nan=False``，非有限值
一律换成哨兵 + 显式掩码），字符串走 ``strings``，数组走 ``add``；每个时间轴的语义写在
``set_strings(time_semantics=...)`` 里。对数轴的地板、被截断的巨值都在 ``strings`` 里说明。
"""

from __future__ import annotations

import functools
import math
from typing import Any, Mapping

import numpy as np

from numcore.pde import heat
from numcore.pde import vonneumann as vn
from numcore.verify.order import fit_order
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

#: 单位舍入（= eps/2）：对数轴地板与"精确 0"的显示约定都用它。
_ROUNDOFF = float(np.finfo(float).eps) / 2.0

#: 对数轴地板：比这更小的值一律显示为地板，掩码同时置位。
_ERR_FLOOR = 1.0e-17

#: 发散时的显示上限：解超过它就地截断显示，**不隐藏**——meta.overflow 与掩码同时置位。
_U_CLIP = 1.0e3

#: 细步参考的步长比：初值不是单模态时，用同格式 r/16 的解当参考解。
_REF_FACTOR = 16.0

_T_MAX = {"stability_threshold": 0.01, "numerical_dissipation": 0.01, "boundary_conservation": 0.05}

_NONFINITE_POLICY = (
    "本帧不含任何非有限值：对数轴上的 0 与无穷用 strings.floor（={floor:g}）占位并配掩码，"
    "解超过显示上限 {uc:g} 的位置截断并配 *_clipped 掩码、meta.overflow 置位；"
    "折线不用 NaN 分隔（NaN 会让前端静默丢段）；meta 一律是有限标量"
    "（.nlf 头用 allow_nan=False 编码，塞一个 NaN 整帧就写不出去）"
)

_TIME_SEMANTICS = {
    "stability_threshold": "真实时间 t：从初值推进到当前 t（meta.t_actual 是实际到达的时刻，未必正好等于 t）",
    "convergence_order": "无时间轴（一次性扫描整条误差阶梯）",
    "numerical_dissipation": "真实时间 t：模态存活率 |a_k(t)/a_k(0)| 随时间衰减，t=0 时全为 1",
    "boundary_conservation": "真实时间 t：三条总热量曲线 H(t) 从 0 漂移出去（或原地不动）",
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


def _clip(arr: Any, ceil: float = _U_CLIP) -> tuple[np.ndarray, np.ndarray]:
    """把超过显示上限（或非有限）的元换成 ``ceil``，返回 ``(数组, 截断掩码)``。"""
    a = np.asarray(arr, dtype=float).copy()
    bad = ~np.isfinite(a) | (np.abs(a) > ceil)
    return np.where(bad, ceil, a), bad.astype(float)


def _single_mode_exact(bc: str, initial: str):
    """单模态初值下返回解析解函数；不是单模态（噪声/帽子）时返回 ``None``。

    这里刻意**不**给"看起来差不多"的近似解：参考解不存在的场合就必须用细步参考，
    并且如实写进 ``strings.error_reference``。混用会让误差曲线变成一条讲不清来源的线。
    """
    if bc == "dirichlet" and initial == "sine":
        return lambda x, t: heat.exact_dirichlet_sine(x, t)
    if bc == "neumann" and initial == "cos":
        return lambda x, t: heat.exact_neumann_cos(x, t, mean=0.0)
    if bc == "periodic" and initial == "cos_periodic":
        return lambda x, t: heat.exact_periodic_cos(x, t)
    return None


def _profile(key: str, nx: int, bc: str = "dirichlet") -> Any:
    """把初值名变成真正的数组（``modes`` 是多模态叠加，由数组直接给）。"""
    grid = heat.uniform_grid(nx, periodic=(bc == "periodic"))
    if key == "modes":
        u = np.zeros(nx, dtype=float)
        for k in range(1, 9):
            u = u + heat.initial_profile(grid.x, "sine", k=k) / k
        return u
    if key == "cos_mean1":
        return 1.0 + heat.initial_profile(grid.x, "cos")
    if key == "cos_periodic_mean1":
        return 1.0 + heat.initial_profile(grid.x, "cos_periodic")
    return heat.initial_profile(grid.x, key)


def _step_count(r: float, dx: float, t: float) -> int:
    """把请求时刻换成整数步数（取最近的一步；实际到达时刻由调用方如实写进 meta）。"""
    dt = r * dx * dx
    if dt <= 0.0:
        return 0
    return max(0, int(round(float(t) / dt)))


@functools.lru_cache(maxsize=256)
def _run(nx: int, scheme: str, r: float, n_steps: int, bc: str, initial_key: str, r_scale: float) -> heat.HeatResult:
    """带缓存的推进（scrub 时同一组参数会被反复请求）。"""
    return heat.heat_solve(
        nx,
        scheme=scheme,
        r=r * r_scale,
        n_steps=n_steps,
        bc=bc,
        initial=_profile(initial_key, nx, bc),
    )


#: 幂迭代预算：nx=101 时 20000 步要 200–300 ms（相邻特征值只差 1e-4，判据几乎不触发）。
#: 3000 步 / rtol=1e-9 在全部参数角上把实测值控制在 1.4e-6 以内（退化最厉害的
#: nx=101、r=0.05 除外，那里顶部特征值挤在一起，估计量只能到 1e-4——如实写进 radius_note）。
_GROWTH_MAX_ITER = 3000
_GROWTH_RTOL = 1.0e-9


@functools.lru_cache(maxsize=256)
def _growth(nx: int, scheme: str, r: float, bc: str) -> float:
    """实测谱半径（幂迭代，缓存）——**必须是这个 scheme 的算子**。

    这里踩过一次真坑：FTCS 有现成的 `heat.explicit_growth`，于是早期版本无条件调它，
    结果 btcs/cn 的帧把 FTCS 的谱半径当成"实测值"报出去（btcs r=50 报 198.69，
    真值 0.76437；cn r=500 报 1995.9，真值 0.99793），`radius_note` 还把差异归错原因。
    现在按 `scheme_theta` 取 θ，用 `theta_step` 做同一种归一化幂迭代（隐式那一步是
    三对角求解，代价与显式同阶），并缓存。
    """
    if scheme == "ftcs":
        return heat.explicit_growth(r * np.ones(nx), nx=nx, bc=bc, rtol=_GROWTH_RTOL, max_iter=_GROWTH_MAX_ITER)
    theta = heat.scheme_theta(scheme, None)
    dx = 1.0 / nx
    dt = r * dx * dx  # 定常系数下 alpha=1，r = dt/dx^2
    # 起始向量取**确定性**的非对称剖面（不用随机数：P4 禁止在这里直接抽随机，
    # 而且确定性起点让"同参数同结果"这件事本身可断言）。它必须与主特征向量有非零投影。
    k = np.arange(nx)
    x = 1.0 + 0.3 * np.sin(2.0 * np.pi * (k + 0.5) / nx) + 0.1 * np.cos(4.0 * np.pi * (k + 0.5) / nx)
    x = x / float(np.sqrt(np.dot(x, x)))
    lam = 1.0
    for _ in range(_GROWTH_MAX_ITER):
        y = heat.theta_step(x, r=r, theta=theta, dx=dx, dt=dt, bc=bc)
        ny = float(np.sqrt(np.dot(y, y)))
        if ny == 0.0:
            return 0.0
        x = y / ny
        lam_new = ny
        if abs(lam_new - lam) <= _GROWTH_RTOL * max(1.0, abs(lam_new)):
            lam = lam_new
            break
        lam = lam_new
    return float(lam)


def _analytic_radius(scheme: str, r: float, nx: int, bc: str) -> float:
    r"""解析的有限网格谱半径——**与幂迭代同一个子空间**：非 Dirichlet 要排掉常数模态。

    不排掉的话，Neumann/周期的解析值恒等于 1（常数模态 |G|=1），
    而幂迭代测的是投影后的值，两个数根本不在比同一件事。
    """
    modes = vn.mode_wavenumbers(nx, bc=bc)
    if bc != "dirichlet":
        modes = modes[1:]
    return float(np.max(np.abs(vn.amplification(scheme, modes, r))))


# --------------------------------------------------------------------------- #
# viz 1：稳定性阈值
# --------------------------------------------------------------------------- #
def _viz_stability_threshold(p: Mapping[str, Any], t: float) -> Frame:
    scheme = str(p["scheme"])
    r = float(p["r"])
    nx = int(p["nx"])
    bc = str(p["bc"])
    initial = str(p["initial"])
    t = max(0.0, min(float(t), _T_MAX["stability_threshold"]))
    fr = Frame()

    dx = heat.uniform_grid(nx, periodic=(bc == "periodic")).dx
    n_steps = _step_count(r, dx, t)
    res = _run(nx, scheme, r, n_steps, bc, initial, 1.0)

    exact_fn = _single_mode_exact(bc, initial)
    if exact_fn is not None:
        ref = exact_fn(res.x, res.t)
        ref_kind = "analytic"
        r_ref = r
    else:
        ref = _run(nx, scheme, r, n_steps, bc, initial, 1.0 / _REF_FACTOR).u
        ref_kind = f"同格式 r/{_REF_FACTOR:g} 的细步解"
        r_ref = r / _REF_FACTOR

    u_disp, u_clip_mask = _clip(res.u)
    ref_disp, _ = _clip(ref)
    err = np.abs(res.u - ref)
    err_linf = float(np.max(err)) if err.size else 0.0
    err_l2 = float(np.sqrt(np.sum(err**2) * dx)) if err.size else 0.0
    overflow = bool((not np.all(np.isfinite(res.u))) or res.umax > _U_CLIP)

    phis = np.linspace(0.0, np.pi, 241)
    others = [s for s in ("ftcs", "btcs", "cn") if s != scheme]
    g_abs_cur = np.abs(np.asarray(vn.amplification(scheme, phis, r), dtype=float))
    g_abs_a = np.abs(np.asarray(vn.amplification(others[0], phis, r), dtype=float))
    g_abs_b = np.abs(np.asarray(vn.amplification(others[1], phis, r), dtype=float))
    unstable = (g_abs_cur > 1.0 + 1e-15).astype(float)

    theta = heat.scheme_theta(scheme)
    measured_radius = _growth(nx, scheme, r, bc)
    modes = vn.mode_wavenumbers(nx, bc=bc)
    g_modes = np.abs(np.asarray(vn.amplification(scheme, modes, r), dtype=float))
    analytic_radius = _analytic_radius(scheme, r, nx, bc)
    nontrivial = g_modes[1:] if bc != "dirichlet" else g_modes
    n_unstable = int(np.count_nonzero(nontrivial > 1.0 + 1e-15))
    r_crit = vn.critical_r_finite(nx, bc=bc, theta=theta)
    flip = vn.sign_flip_wavenumber(r, theta=theta)

    # 最后一步的代数残差（真的一步，不是自洽性检查）
    prev = _run(nx, scheme, r, max(0, n_steps - 1), bc, initial, 1.0).u
    resid = 0.0 if n_steps == 0 else heat.step_residual(res.u, prev, r=r, theta=theta, dx=dx, dt=r * dx * dx, bc=bc)

    fr.add("x", res.x, unit="1", label="格点 x")
    fr.add("u", u_disp, unit="1", label=f"{scheme} 的解 u(x,t)（超过 ±{_U_CLIP:g} 已截断）")
    fr.add("u_clipped", u_clip_mask, unit="1", label="1 = 该点被截断显示")
    fr.add("u_ref", ref_disp, unit="1", label="参考解（解析解或细步解）")
    fr.add("error_curve", _floor(err), unit="1", label="|u − 参考解|（对数轴，地板 1e-17）")
    fr.add("error_at_floor", (err < _ERR_FLOOR).astype(float), unit="1", label="1 = 该点误差已到地板")
    fr.add("phi", phis, unit="rad", label="无量纲波数 φ=kΔx")
    fr.add("g_abs_cur", g_abs_cur, unit="1", label=f"|G(φ)|（当前格式 {scheme}）")
    fr.add("g_abs_other_a", g_abs_a, unit="1", label=f"|G(φ)|（{others[0]}，对照）")
    fr.add("g_abs_other_b", g_abs_b, unit="1", label=f"|G(φ)|（{others[1]}，对照）")
    fr.add("g_one", np.ones_like(phis), unit="1", label="稳定界线 |G| = 1")
    fr.add("g_unstable_region", unstable, unit="1", label="1 = 该波数上 |G| > 1（会被放大）")
    fr.add("mark_growth", np.asarray([measured_radius]), unit="1", label="实测谱半径（幂迭代）")
    fr.add("mark_phi_flip", np.asarray([flip if flip is not None else -1.0]), unit="rad", label="符号翻转波数 φ*（-1 = 不存在）")
    fr.add("algebraic_residual", np.asarray([_fin(resid)]), unit="1", label="最后一步的回代残差 |A u^{n+1} − b(u^n)|")

    fr.set_meta(
        scheme_index=int(("ftcs", "btcs", "cn").index(scheme)),
        r=r,
        theta=theta,
        nx=nx,
        t_requested=t,
        t_actual=res.t,
        dt=res.dt,
        n_steps=res.n_steps,
        dx=dx,
        umax=_fin(res.umax),
        overflow=overflow,
        clipped_points=int(np.count_nonzero(u_clip_mask > 0.5)),
        measured_radius=_fin(measured_radius),
        analytic_radius=_fin(analytic_radius),
        radius_gap=_fin(abs(measured_radius - analytic_radius)),
        radius_rel_gap=_fin(abs(measured_radius - analytic_radius) / analytic_radius) if analytic_radius > 0 else -1.0,
        r_critical_finite=_fin(r_crit, -1.0),
        r_critical_continuous=0.5 if scheme == "ftcs" else -1.0,
        unstable_modes=n_unstable,
        total_modes=int(nontrivial.size),
        unstable_fraction=_fin(n_unstable / nontrivial.size),
        phi_flip=_fin(flip, -1.0),
        has_flip=bool(flip is not None),
        reference_r=r_ref,
        err_linf=_fin(err_linf),
        err_l2=_fin(err_l2),
        mass=res.mass,
    )
    fr.set_strings(
        time_semantics=_TIME_SEMANTICS["stability_threshold"],
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_ERR_FLOOR, uc=_U_CLIP),
        floor=f"{_ERR_FLOOR:g}",
        error_reference="解析解" if ref_kind == "analytic" else ref_kind,
        scheme=scheme,
        bc=bc,
        initial=initial,
        reading_note=(
            "副图看三条 |G(φ)| 曲线：只要有一条越过 1 就是不稳定，与主图「看起来好不好」无关。"
            "主图的解是同一段代码真跑出来的——初值换成正弦时，越界的高频模态可能根本没被激发。"
        ),
        radius_note=(
            "实测谱半径是幂迭代估计（投影掉常数模态），解析值是同一子空间上的精确最大值。"
            "两者之差随顶部特征值的间隙变化：小 r 时所有模态衰减率挤在一起，"
            "估计量只能到 1e-4 量级；阈值附近（间隙大）能到 1e-7 以内。"
        ),
        stable_note=(
            f"实测谱半径 {measured_radius:.6f}，解析有限网格值 {analytic_radius:.6f}；"
            f"该网格上 {n_unstable}/{nontrivial.size} 个非平凡模态会被放大；"
            f"有限网格阈值 {r_crit:.6f}（连续极限 0.5 只对显式格式且 Δx→0 成立）"
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 2：收敛阶
# --------------------------------------------------------------------------- #
_SPACE_LADDER = (11, 22, 44, 88, 176)


def _ladder_steps(r: float, nx: int, t_end: float) -> int:
    """与 ``heat_solve(t_end=...)`` 同口径的步数（ceil，最后一拍可以越过 t_end）。"""
    dx = heat.uniform_grid(nx).dx
    return max(1, int(math.ceil(float(t_end) / (r * dx * dx))))


def _viz_convergence_order(p: Mapping[str, Any], t: float) -> Frame:
    scheme = str(p["scheme"])
    target = str(p["target"])
    nx = int(p["nx"])
    levels = int(p["levels"])
    t_end = float(p["t_end"])
    fr = Frame()
    del t  # 无时间轴

    if target == "space":
        sizes = list(_SPACE_LADDER[:levels])
        hs: list[float] = []
        errs: list[float] = []
        conts: list[float] = []
        for n in sizes:
            res = _run(n, scheme, 0.4, _ladder_steps(0.4, n, t_end), "dirichlet", "sine", 1.0)
            exact = heat.exact_dirichlet_sine(res.x, res.t)
            hs.append(res.dx)
            errs.append(heat.error_norms(res.u, exact, res.dx)[1])
            conts.append(errs[-1])
        theory = 2.0
        h_label = "Δx"
        ref_name = "连续精确解"
        ladder_note = "固定 r=0.4 加密网格（11/22/44/88/176）：dt=r·Δx² 跟着变小，时间误差被压成同阶"
    else:
        hs = [0.4 / (2.0**i) for i in range(levels)]
        errs = []
        conts = []
        for r in hs:
            res = _run(nx, scheme, r, _ladder_steps(r, nx, t_end), "dirichlet", "sine", 1.0)
            sd = heat.exact_semidiscrete_sine(res.x, res.t, nx=nx, k=1)
            exact = heat.exact_dirichlet_sine(res.x, res.t)
            errs.append(heat.error_norms(res.u, sd, res.dx)[1])
            conts.append(heat.error_norms(res.u, exact, res.dx)[1])
        theory = 1.0 if scheme in ("ftcs", "btcs") else 2.0
        h_label = "Δt"
        ref_name = "半离散精确解"
        ladder_note = "固定网格把 Δt 每次减半（r=0.4→0.05）：靶子换成半离散解，空间误差被解析扣掉"

    fit = fit_order(hs, errs)
    slope_cont = fit_order(hs, [max(v, 1e-300) for v in conts]).slope
    i_ref = int(np.argmin(hs))
    theory_line = errs[i_ref] * (np.asarray(hs) / hs[i_ref]) ** (-theory)

    if target == "space":
        fine = _run(sizes[-1], scheme, 0.4, _ladder_steps(0.4, sizes[-1], t_end), "dirichlet", "sine", 1.0)
        prof = np.abs(fine.u - heat.exact_dirichlet_sine(fine.x, fine.t))
        x_fine = fine.x
    else:
        fine = _run(nx, scheme, hs[-1], _ladder_steps(hs[-1], nx, t_end), "dirichlet", "sine", 1.0)
        prof = np.abs(fine.u - heat.exact_semidiscrete_sine(fine.x, fine.t, nx=nx, k=1))
        x_fine = fine.x

    fr.add("h_ladder", np.asarray(hs), unit=h_label, label=f"步长 {h_label}（{len(hs)} 级）")
    fr.add("error_ladder", _floor(errs), unit="1", label=f"L2 误差（对{ref_name}）")
    fr.add("error_ladder_continuous", _floor(conts), unit="1", label="L2 误差（对连续精确解，作对照）")
    fr.add("theory_line", _floor(theory_line), unit="1", label=f"理论斜率 −{theory:g}（过最细一级）")
    fr.add("mark_h", np.asarray([hs[i_ref]]), unit=h_label, label="最细一级的步长")
    fr.add("mark_error", np.asarray([errs[i_ref]]), unit="1", label="最细一级的误差")
    fr.add("x_profile", x_fine, unit="1", label="x（最细一级）")
    fr.add("error_profile", _floor(prof), unit="1", label="最细一级的逐点误差（对数轴，地板 1e-17）")

    fr.set_meta(
        target_is_space=bool(target == "space"),
        theory_order=theory,
        measured_order=_fin(fit.slope),
        order_stderr=_fin(fit.stderr),
        r2=_fin(fit.r2),
        n_levels=len(hs),
        n_used=fit.n_used,
        n_dropped=fit.n_dropped,
        constant=_fin(fit.constant),
        order_vs_continuous=_fin(slope_cont),
        err_finest=_fin(errs[-1]),
        err_coarsest=_fin(errs[0]),
        err_ratio=_fin(errs[0] / errs[-1]) if errs[-1] > 0 else -1.0,
        err_ratio_continuous=_fin(conts[0] / conts[-1]) if conts[-1] > 0 else -1.0,
        order_gap=_fin(abs(fit.slope - theory)),
        fixed_r=0.4,
        nx=nx,
        t_end=t_end,
        fine_points=int(x_fine.size),
        err_floor=_ERR_FLOOR,
    )
    fr.set_strings(
        time_semantics=_TIME_SEMANTICS["convergence_order"],
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_ERR_FLOOR, uc=_U_CLIP),
        floor=f"{_ERR_FLOOR:g}",
        scheme=scheme,
        target=target,
        reference=ref_name,
        ladder_note=ladder_note,
        reading_note=(
            "主图双对数：曲线斜率就是实测阶，虚线的斜率是理论阶。"
            "副图是同一次最细一级的逐点误差——误差在空间上不是常数，"
            "用一个范数代表全部信息本身就是一次压缩。"
        ),
        trap_note=(
            f"同一批数据改用连续解当靶子，实测斜率变成 {slope_cont:.4f}"
            + ("（空间误差主导，时间阶被完全盖住——这不是格式的问题，是靶子的问题）" if target == "time" else "（本来就是连续解当靶子，两者应当一致）")
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 3：数值耗散
# --------------------------------------------------------------------------- #
def _viz_numerical_dissipation(p: Mapping[str, Any], t: float) -> Frame:
    scheme = str(p["scheme"])
    r = float(p["r"])
    nx = int(p["nx"])
    n_modes = int(p["n_modes"])
    t = max(0.0, min(float(t), _T_MAX["numerical_dissipation"]))
    fr = Frame()

    dx = heat.uniform_grid(nx).dx
    u0 = _profile("modes", nx)
    n_steps = _step_count(r, dx, t)
    res = heat.heat_solve(nx, scheme=scheme, r=r, n_steps=n_steps, initial=u0)

    phis, ratio, alive = vn.mode_decay_resolved(u0, res.u, bc="dirichlet", k_max=n_modes)
    _, _, present = vn.mode_decay_resolved(u0, u0, bc="dirichlet", k_max=n_modes)
    n_absent = int(np.count_nonzero(~present))
    n_masked = int(np.count_nonzero(present & ~alive))
    ratio_f = _floor(np.where(alive, ratio, _ERR_FLOOR))

    theta = heat.scheme_theta(scheme)
    g_abs = np.abs(np.asarray(vn.amplification(scheme, phis, r), dtype=float))
    formula = g_abs**n_steps
    exact = np.exp(-r * phis**2 * n_steps)
    lam = np.asarray([heat.discrete_sine_eigenvalue(nx, int(k)) for k in range(1, phis.size + 1)])
    semidiscrete = np.exp(-lam * res.t)

    resid = np.where(alive, ratio - formula, 0.0)
    diss = np.where(alive, ratio - semidiscrete, 0.0)
    flip = vn.sign_flip_wavenumber(r, theta=theta)
    k_worst = int(np.argmax(np.abs(diss))) if diss.size else 0

    fr.add("phi", phis, unit="rad", label="模态波数 φ=kπ/(nx−1)")
    fr.add("decay_measured", ratio_f, unit="1", label="实测存活率 |a_k(t)/a_k(0)|（对数轴）")
    fr.add("decay_alive", alive.astype(float), unit="1", label="1 = 初值里这个模态存在（否则比值是噪声）")
    fr.add("decay_formula", _floor(formula), unit="1", label="|G(φ)|ⁿ（放大因子的解析值）")
    fr.add("decay_semidiscrete", _floor(semidiscrete), unit="1", label="半离散精确解 e^(−λ_k t)")
    fr.add("decay_exact", _floor(exact), unit="1", label="真解 e^(−rφ²n)")
    fr.add("amplitude_residual", resid, unit="1", label="实测存活率 − |G|ⁿ（应精确为 0）")
    fr.add("dissipation_error", diss, unit="1", label="实测存活率 − 半离散精确解（负 = 数值衰减更快）")
    fr.add("phi_flip", np.asarray([flip if flip is not None else -1.0]), unit="rad", label="符号翻转波数 φ*（-1 = 不存在）")

    fr.set_meta(
        scheme_index=int(("ftcs", "btcs", "cn").index(scheme)),
        r=r,
        theta=theta,
        nx=nx,
        t_requested=t,
        t_actual=res.t,
        dt=res.dt,
        n_steps=res.n_steps,
        n_modes=n_modes,
        n_alive=int(np.count_nonzero(alive)),
        n_absent=n_absent,
        n_masked_by_roundoff=n_masked,
        mask_floor_rel=1e-8,
        max_amplitude_residual=_fin(float(np.max(np.abs(resid)))),
        worst_phi=_fin(phis[k_worst]),
        worst_dissipation=_fin(float(diss[k_worst])),
        mean_dissipation=_fin(float(np.mean(diss[alive]))) if np.any(alive) else 0.0,
        dissipation_coefficient=_fin(vn.dissipation_coefficient(scheme, r, theta=theta)),
        c4_numeric=_fin(vn.symbol_expansion(scheme, r, theta=theta)[2]),
        c4_exact=_fin(r * r / 2.0),
        phi_flip=_fin(flip, -1.0),
        has_flip=bool(flip is not None),
        decay_min=_fin(float(np.min(ratio_f))),
        decay_max=_fin(float(np.max(ratio_f))),
        err_floor=_ERR_FLOOR,
    )
    fr.set_strings(
        time_semantics=_TIME_SEMANTICS["numerical_dissipation"],
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_ERR_FLOOR, uc=_U_CLIP),
        floor=f"{_ERR_FLOOR:g}",
        scheme=scheme,
        initial="Σ_{k=1..8} sin(kπx)/k（多模态，前 8 个模态都有能量）",
        reading_note=(
            "三条线：真解、半离散解、实测存活率（自上而下）。"
            "实测线与 |G|ⁿ 的重合度就是对拍精度（meta.max_amplitude_residual）；"
            "实测线与真解的距离就是这一步的数值耗散，短波（右侧）掉得最快。"
        ),
        mask_note=(
            "decay_alive=0 有两种原因，都写在 meta 里：初值里没有这个模态（分母是舍入级，"
            "实测过纯 sin(πx) 初值下 k≥2 的比值给出 22.1 这种假放大率），"
            "或者解里这个模态已经低于 ‖u‖∞×1e-8（分子被别的模态的舍入淹没）。"
            "被掩码的位置显示为地板值——不是 0，是「量不出来」。"
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 4：边界条件与守恒
# --------------------------------------------------------------------------- #
_BC_KEYS = {"dirichlet": "sine", "neumann": "cos_mean1", "periodic": "cos_periodic_mean1"}

#: 总热量历史曲线的采样点数（副轴）。
_N_RECORD = 21


@functools.lru_cache(maxsize=128)
def _mass_history(
    nx: int, scheme: str, r: float, n_steps: int, bc: str, n_max: int
) -> tuple[np.ndarray, np.ndarray]:
    """总热量漂移 ``H(t)−H(0)`` 的历史曲线，**裁到当前时刻**。

    ``heat_trajectory`` 要求 ``n_steps ≥ 1``，所以 ``t=0`` 时这里直接给出
    「21 个 0」的退化历史：图上是一个位于原点的点，而不是"提前跑了一步"的曲线
    （这个坑真踩过：``heat_solve(t_end=0)`` 会按 ``max(1, ceil(0/dt))`` 走一步）。
    """
    if n_steps < 1:
        return np.zeros(_N_RECORD, dtype=float), np.zeros(_N_RECORD, dtype=float)
    traj = _traj(nx, scheme, r, min(int(n_steps), int(n_max)), bc, _BC_KEYS[bc], _N_RECORD)
    return traj.mass - traj.mass[0], traj.t


@functools.lru_cache(maxsize=128)
def _traj(nx: int, scheme: str, r: float, n_steps: int, bc: str, initial_key: str, n_record: int):
    """带缓存的总热量轨迹（scrub 时同一组参数会被反复请求）。"""
    return heat.heat_trajectory(
        nx, scheme=scheme, r=r, n_steps=n_steps, bc=bc, initial=_profile(initial_key, nx, bc), n_record=n_record
    )


def _bc_exact(bc: str):
    if bc == "dirichlet":
        return lambda x, t: heat.exact_dirichlet_sine(x, t)
    if bc == "neumann":
        return lambda x, t: heat.exact_neumann_cos(x, t, mean=1.0)
    return lambda x, t: 1.0 + heat.exact_periodic_cos(x, t)


def _viz_boundary_conservation(p: Mapping[str, Any], t: float) -> Frame:
    bc = str(p["bc"])
    scheme = str(p["scheme"])
    r = float(p["r"])
    nx = int(p["nx"])
    t = max(0.0, min(float(t), _T_MAX["boundary_conservation"]))
    fr = Frame()

    dx = heat.uniform_grid(nx, periodic=(bc == "periodic")).dx
    n_total = max(1, _step_count(r, dx, _T_MAX["boundary_conservation"]))
    n_steps = _step_count(r, dx, t)
    masses: dict[str, np.ndarray] = {}
    t_hist = np.zeros(_N_RECORD, dtype=float)
    for name in ("dirichlet", "neumann", "periodic"):
        shift, hist_t = _mass_history(nx, scheme, r, n_steps, name, n_total)
        masses[name] = shift
        if name == bc:
            t_hist = hist_t

    u0 = _profile(_BC_KEYS[bc], nx, bc)
    res = heat.heat_solve(nx, scheme=scheme, r=r, n_steps=n_steps, bc=bc, initial=u0)
    u_exact = _bc_exact(bc)(res.x, res.t)
    err = np.abs(res.u - u_exact)
    drift_now = float(res.mass - float(np.sum(u0)) * dx)
    leak_now = -drift_now  # 流失量取正值：ΔH 是**掉下去**的热量
    ref_leak = (2.0 / np.pi) * (1.0 - np.exp(-np.pi**2 * res.t)) if bc == "dirichlet" else 0.0
    others = [k for k in masses if k != bc]

    fr.add("x", res.x, unit="1", label="格点 x")
    fr.add("u", res.u, unit="1", label=f"{bc} 边界的数值解")
    fr.add("u_exact", u_exact, unit="1", label="解析解")
    fr.add("error_curve", _floor(err), unit="1", label="|u − 解析解|（对数轴，地板 1e-17）")
    fr.add("t_history", t_hist, unit="s", label="H(t) 的历史时刻")
    fr.add("mass_shift_dirichlet", masses["dirichlet"], unit="1", label="H(t)−H(0)（Dirichlet：边界流失）")
    fr.add("mass_shift_neumann", masses["neumann"], unit="1", label="H(t)−H(0)（Neumann：绝缘）")
    fr.add("mass_shift_periodic", masses["periodic"], unit="1", label="H(t)−H(0)（周期）")
    fr.add("mass_zero", np.zeros_like(t_hist), unit="1", label="守恒基线 0")
    fr.add(
        "mass_residual",
        np.asarray([masses["neumann"][-1], masses["periodic"][-1], drift_now]),
        unit="1",
        label="末刻总热量漂移（Neumann / 周期 / 当前边界）",
    )
    fr.add("mark_t", np.asarray([res.t]), unit="s", label="当前时刻")

    fr.set_meta(
        bc_index=int(("dirichlet", "neumann", "periodic").index(bc)),
        scheme_index=int(("ftcs", "btcs", "cn").index(scheme)),
        r=r,
        nx=nx,
        t_requested=t,
        t_actual=res.t,
        dt=res.dt,
        n_steps=res.n_steps,
        n_total=n_total,
        mass0=_fin(float(np.sum(u0)) * dx),
        mass_now=_fin(res.mass),
        drift_now=_fin(drift_now),
        drift_dirichlet_max=_fin(float(np.max(np.abs(masses["dirichlet"])))),
        drift_neumann_max=_fin(float(np.max(np.abs(masses["neumann"])))),
        drift_periodic_max=_fin(float(np.max(np.abs(masses["periodic"])))),
        drift_other_max=_fin(max(float(np.max(np.abs(masses[k]))) for k in others)),
        analytic_leak=_fin(ref_leak),
        leak_now=_fin(leak_now),
        leak_ratio=_fin(leak_now / ref_leak) if ref_leak > 0.0 else -1.0,
        err_linf=_fin(float(np.max(err))),
        err_l2=_fin(heat.error_norms(res.u, u_exact, dx)[1]),
        conservation_holds=bool(bc != "dirichlet"),
        roundoff=_ROUNDOFF,
        err_floor=_ERR_FLOOR,
    )
    fr.set_strings(
        time_semantics=_TIME_SEMANTICS["boundary_conservation"],
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_ERR_FLOOR, uc=_U_CLIP),
        floor=f"{_ERR_FLOOR:g}",
        bc=bc,
        scheme=scheme,
        reading_note=(
            "副图三条曲线共用一根轴：Neumann 与周期是贴着 0 的直线（离散算子行和为零，"
            "与 θ、与 r 都无关），Dirichlet 是斜着下掉的线。"
            "「守恒」不是「数很小」，而是「结构性为零」。"
        ),
        leak_note=(
            f"Dirichlet 的流失量有解析值 (2/π)(1−e^(−π²t)) = {ref_leak:.6f}，"
            f"实测 {leak_now:.6f}（比 {leak_now / ref_leak if ref_leak else 0.0:.4f}）"
        ),
        drift_note=(
            f"Neumann/周期末刻漂移 {masses['neumann'][-1]:.3e} / {masses['periodic'][-1]:.3e}，"
            f"量级是舍入（{_ROUNDOFF:.1e} × 步数 {n_steps}）；"
            f"三条历史曲线都裁到当前时刻（t=0 时是一个位于 0 的点）"
        ),
        initial_note=(
            "三种边界用三种相容初值：Dirichlet 用 sin(πx)、Neumann 用 1+cos(πx)、"
            "周期用 1+cos(2πx)；比较的应当是漂移量，不是解的绝对大小"
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# MANIFEST
# --------------------------------------------------------------------------- #
MANIFEST: dict[str, Any] = {
    "chapter": "4.1",
    "title": "抛物型方程与时间推进",
    "part": "IV",
    "summary": (
        "热方程 $\\theta$ 族格式的全部结论都挂在一个数上：放大因子 $G(\\varphi)$。"
        "显式的 $r \\le 1/2$、隐式的无条件稳定、CN 的二阶、数值耗散掉得最快的是短波——"
        "都是 $G(\\varphi)$ 的不同侧面。这一章把三条证据链并排放：解析的 $G(\\varphi)$、"
        "从真的解里分解出来的逐模态存活率、以及二分实测的稳定阈值。"
        "三个最容易被骗的地方也都标出来了：$r \\le 1/2$ 不是普适常数、稳定不等于精确、"
        "解看起来光滑不等于收敛。"
    ),
    "vizzes": [
        # ------------------------------------------------------------------ #
        {
            "id": "stability_threshold",
            "title": "稳定性阈值：显式 vs 隐式 vs Crank–Nicolson",
            "description": (
                "主图是当前参数下真跑出来的解曲线与参考解（时间轴是真实时间 $t$，向右拖推进）。"
                "副图是同一 $r$ 上三种格式的 $|G(\\varphi)|$ 曲线、稳定界线 1、以及幂迭代实测的谱半径。"
                "把 $r$ 从 0.4 拖到 0.6，显式格式的 $|G|$ 在右端越过 1——但主图的解是否现形，"
                "完全取决于初值里有没有那个模态：这正是换 initial 能看见的东西。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "scheme",
                    "label": "格式",
                    "type": "enum",
                    "options": ["ftcs", "btcs", "cn"],
                    "default": "ftcs",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "r",
                    "label": "格点比 r = αΔt/Δx²",
                    "type": "float",
                    "min": 0.05,
                    "max": 0.8,
                    "step": 0.01,
                    "default": 0.45,
                    "critical": [0.5],
                    "criticalNote": (
                        "$r=1/2$ 是显式格式的稳定阈值（连续极限），不适用于 BTCS/CN（它们无条件稳定）。"
                        "有限网格上也各不相同：Dirichlet 略大于 $1/2$（端点钉住后 Nyquist 模态不存在），"
                        "Neumann 与偶数周期网格恰好等于 $1/2$。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "nx",
                    "label": "格点数 nx",
                    "type": "int",
                    "min": 21,
                    "max": 101,
                    "step": 10,
                    "default": 41,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "bc",
                    "label": "边界条件",
                    "type": "enum",
                    "options": ["dirichlet", "neumann", "periodic"],
                    "default": "dirichlet",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "initial",
                    "label": "初值",
                    "type": "enum",
                    "options": ["sine", "noise", "hat"],
                    "default": "sine",
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 0.01, "dt": 0.00025, "label": "t（真实时间）"},
            "requiredAux": ["error", "residual"],
            "axes": [
                {
                    "id": "solution",
                    "autoscaled": False,
                    "title": "解曲线（窗口固定 ±1.5：发散时冲出去，而不是被自动缩放藏起来）",
                    "ylabel": "u(x,t)",
                },
                {
                    "id": "amplification",
                    "autoscaled": False,
                    "title": "放大因子 |G(φ)| 与稳定界线 1（窗口固定 0–2.4）",
                    "ylabel": "|G|",
                },
            ],
            "pitfalls": (
                "第一，换初值就能换结论：$r=0.51$ 已经越界，但 $\\sin(\\pi x)$ 初值下解逐点光滑、峰值 0.61、"
                "误差 3.2e-4（比稳定的 $r=0.45$ 还好看）；同一个 $r$ 换成噪声初值，峰值 42.9。"
                "看到「光滑」就下结论是被初值骗了，判据只有 $|G|>1$。"
                "第二，副图的 $|G|$ 是解析公式画的、实测谱半径是幂迭代测的；Neumann/周期下不把常数模态"
                "投影掉的话，实测值会永远停在 1.000000000000。第三，主图窗口固定 $\\pm 1.5$，"
                "超过 $\\pm 10^3$ 的点被截断显示并置溢出掩码：截断不等于没有发生，meta.umax 是真值。"
            ),
            "theory": [
                "$G_\\theta(\\varphi) = \\frac{1-4(1-\\theta)r\\sin^2(\\varphi/2)}{1+4\\theta r\\sin^2(\\varphi/2)}$；$|G|$ 的极值只在 $\\varphi=0$ 与 $\\varphi=\\pi$ 取到",
                "FTCS：$|G|\\le 1 \\Leftrightarrow r \\le 1/2$（连续极限）；有限网格上是 $1/(2\\sin^2(\\varphi_{\\max}/2))$，$\\varphi_{\\max}$ 由边界与网格奇偶性决定",
                "BTCS/CN 无条件稳定（$\\theta \\ge 1/2$）；但 $G(\\pi)\\to -1$ 意味着不衰减而每步翻符号",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "convergence_order",
            "title": "收敛阶：空间细化与时间细化（以及选错靶子的后果）",
            "description": (
                "双对数误差阶梯。target=space 固定 $r=0.4$ 加密网格（11/22/44/88/176），"
                "target=time 固定网格把 $\\Delta t$ 每次减半。主图画实测误差、理论斜率参考线与"
                "「对连续解」的对照曲线；副图画最细一级的逐点误差分布。"
                "meta.order_vs_continuous 是同一批数据换靶子后的斜率——时间细化时会塌到 0 附近。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "scheme",
                    "label": "格式",
                    "type": "enum",
                    "options": ["ftcs", "btcs", "cn"],
                    "default": "cn",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "target",
                    "label": "细化方向",
                    "type": "enum",
                    "options": ["space", "time"],
                    "default": "time",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "nx",
                    "label": "时间细化的固定 nx",
                    "type": "int",
                    "min": 21,
                    "max": 81,
                    "step": 20,
                    "default": 41,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "levels",
                    "label": "阶梯级数",
                    "type": "int",
                    "min": 3,
                    "max": 5,
                    "step": 1,
                    "default": 4,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "t_end",
                    "label": "终止时刻 T",
                    "type": "float",
                    "min": 0.01,
                    "max": 0.1,
                    "step": 0.01,
                    "default": 0.05,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "ladder",
                    "autoscaled": False,
                    "title": "误差 vs 步长（双对数，固定窗口）",
                    "ylabel": "L2 误差",
                    "scale": "log",
                },
                {
                    "id": "profile",
                    "autoscaled": True,
                    "title": "最细一级的逐点误差（对数轴）",
                    "ylabel": "|误差|",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "第一，选错靶子会得到相反的结论：时间细化必须对半离散解测。对连续解测时 CN 的实测斜率是"
                "$-0.0004$（空间误差 1.09e-4 完全盖住时间误差 1e-8），BTCS 是 0.46——两个都荒谬，"
                "但看上去像「格式不收敛」。第二，固定 $r$ 加密网格会让一阶格式也测出二阶"
                "（$dt=r\\Delta x^2$ 把时间误差压成同阶），所以空间细化那一列不代表时间精度。"
                "第三，副图的逐点误差在对数轴上有 1e-17 的地板：曲线两端贴到地板不是"
                "「边界处特别准」，而是那里本来就接近 0（正弦型误差在端点必为 0）。"
            ),
            "theory": [
                "$\\theta$ 族的时间阶：$\\theta=1/2$ 是 2 阶，$\\theta\\ne 1/2$ 是 1 阶；空间恒为 2 阶（二阶中心差分）",
                "单模态下数值解精确等于半离散解乘 $G^n$：时间误差 $= |G^n - e^{-\\lambda_k t}|$，与空间误差可分离",
                "固定 $r$ 时 $dt=O(\\Delta x^2)$，故三种格式在空间细化的阶梯上都表现为 2 阶",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "numerical_dissipation",
            "title": "数值耗散：解析衰减、半离散衰减与实测存活率",
            "description": (
                "主图纵轴取对数：三条曲线分别是真解 $e^{-r\\varphi^2 n}$、半离散解 $e^{-\\lambda_k t}$、"
                "以及从真的解里按正交模态分解出来的存活率 $|a_k(t)/a_k(0)|$。"
                "实测线与 $|G|^n$ 逐位重合（meta.max_amplitude_residual），与半离散解的差"
                "就是这一步的耗散误差——短波（右侧）掉得最快，这就是数值耗散。"
                "时间轴是真实时间：向右拖，整条曲线往下沉。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "scheme",
                    "label": "格式",
                    "type": "enum",
                    "options": ["ftcs", "btcs", "cn"],
                    "default": "cn",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "r",
                    "label": "格点比 r",
                    "type": "float",
                    "min": 0.05,
                    "max": 2.0,
                    "step": 0.05,
                    "default": 0.4,
                    "critical": [0.5],
                    "criticalNote": (
                        "对显式格式，$r>1/2$ 会让短波放大（图上右侧翘起来）；BTCS/CN 没有这条线。"
                        "另外 $r=1/6$ 是耗散方向的翻转点：FTCS 在 $r<1/6$ 时衰减得比真解慢、$r>1/6$ 时比真解快。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "nx",
                    "label": "格点数 nx",
                    "type": "int",
                    "min": 21,
                    "max": 101,
                    "step": 20,
                    "default": 41,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n_modes",
                    "label": "画出的模态数",
                    "type": "int",
                    "min": 4,
                    "max": 16,
                    "step": 2,
                    "default": 8,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 0.01, "dt": 0.001, "label": "t（真实时间）"},
            "requiredAux": ["error", "residual"],
            "axes": [
                {
                    "id": "decay",
                    "autoscaled": False,
                    "title": "逐模态存活率（对数纵轴，窗口固定 1e-17–2）",
                    "ylabel": "|a_k(t)/a_k(0)|",
                    "scale": "log",
                },
                {
                    "id": "dissipation",
                    "autoscaled": True,
                    "title": "耗散误差（实测 − 半离散；负 = 数值衰减更快）",
                    "ylabel": "Δ 存活率",
                },
            ],
            "pitfalls": (
                "第一，模态分解的分母可能是舍入级：初值是纯 $\\sin(\\pi x)$ 时 $k\\ge 2$ 的系数精确为 0，"
                "直接相除会得到 22.1、18.5 这种假放大率。本帧把这类模态判为 nan 并置掩码"
                "（decay_alive=0），存活率在这些位置显示为地板值。"
                "第二，Neumann 边界的余弦族对普通内积不正交——实测 $\\cos(\\pi x)$ 会在 $\\cos(3\\pi x)$ 上"
                "漏出 $1/21=0.0476$ 的假分量，必须用端点半权重；本帧用 Dirichlet 正弦族正是为了避开它。"
                "第三，纵轴地板 1e-17：贴到地板意味着双精度量不出来了，不代表存活率真的等于 0。"
            ),
            "theory": [
                "$G(\\varphi) = 1 - r\\varphi^2 + (r/12 + \\theta r^2)\\varphi^4 + O(\\varphi^6)$；真解 $e^{-r\\varphi^2} = 1 - r\\varphi^2 + (r^2/2)\\varphi^4 + \\ldots$",
                "耗散误差系数 $= r/12 + r^2(\\theta-1/2)$：CN 恒为正（欠耗散），FTCS 在 $r=1/6$ 变号",
                "短波衰减最快 $\\Rightarrow$ 数值解总比真解更光滑；看起来平滑本身就是耗散的证据",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "boundary_conservation",
            "title": "边界条件：总热量守恒 vs 边界流失",
            "description": (
                "主图是当前边界条件下真跑出来的解与解析解；副图把三种边界的总热量漂移"
                "$H(t)-H(0)$ 画在同一根轴上：Neumann 与周期是贴着 0 的直线"
                "（离散算子行和为零，与格式、与 $r$ 都无关），Dirichlet 是斜着下掉的线，"
                "流失量的解析值是 $\\frac{2}{\\pi}(1-e^{-\\pi^2 t})$。时间轴是真实时间 $t$。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "bc",
                    "label": "边界条件",
                    "type": "enum",
                    "options": ["dirichlet", "neumann", "periodic"],
                    "default": "neumann",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "scheme",
                    "label": "格式",
                    "type": "enum",
                    "options": ["ftcs", "btcs", "cn"],
                    "default": "cn",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "r",
                    "label": "格点比 r",
                    "type": "float",
                    "min": 0.05,
                    "max": 0.8,
                    "step": 0.05,
                    "default": 0.4,
                    "critical": [0.5],
                    "criticalNote": "显式格式在 $r>1/2$ 时失稳；守恒结论与 $r$ 无关，但解的准确性有关。",
                    "beyond": "above",
                },
                {
                    "name": "nx",
                    "label": "格点数 nx",
                    "type": "int",
                    "min": 21,
                    "max": 101,
                    "step": 20,
                    "default": 41,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 0.05, "dt": 0.0025, "label": "t（真实时间）"},
            "requiredAux": ["error", "residual"],
            "axes": [
                {
                    "id": "solution",
                    "autoscaled": False,
                    "title": "解曲线（窗口固定 −0.5–2.2：三种边界的初值都在范围内）",
                    "ylabel": "u(x,t)",
                },
                {
                    "id": "mass",
                    "autoscaled": False,
                    "title": "总热量漂移 H(t)−H(0)（窗口固定 −0.7–0.1）",
                    "ylabel": "ΔH",
                },
            ],
            "pitfalls": (
                "第一，「守恒」在图上长得像「一条很小的曲线」，但它是结构性的：离散算子的行和恒为 0，"
                "所以无论 $\\theta$ 取多少、$r$ 取多少，Neumann/周期都精确守恒（实测漂移 1e-15，舍入级）；"
                "Dirichlet 的端点行不是零行，热量必然按边界通量流失。"
                "第二，别把总热量不变当成解不变：Neumann 下常数模态永远不衰减，"
                "数值解会冻结在一个与真解不同的常数上，误差不随时间减小。"
                "第三，本帧三种边界用三种相容初值（$\\sin$ / $1+\\cos$ / $1+\\cos 2\\pi x$），"
                "该比较的是漂移量，不是解的绝对大小。"
            ),
            "theory": [
                "Neumann 用幽灵点消元后端点行为 $[2,-2]$，行和为零 $\\Rightarrow 1^{T}u$ 精确守恒（任意 $\\theta$）",
                "Dirichlet 的总热量按边界通量流失：$\\sin(\\pi x)$ 初值下 $H(t)=\\frac{2}{\\pi}e^{-\\pi^2 t}$",
                "周期循环矩阵每行和也为零 $\\Rightarrow$ 同样守恒；Sherman–Morrison 求解不改变这个性质",
            ],
        },
    ],
}


# --------------------------------------------------------------------------- #
# 调度
# --------------------------------------------------------------------------- #
_DISPATCH = {
    "stability_threshold": _viz_stability_threshold,
    "convergence_order": _viz_convergence_order,
    "numerical_dissipation": _viz_numerical_dissipation,
    "boundary_conservation": _viz_boundary_conservation,
}


def compute(viz_id: str, params: Mapping[str, Any], t: float = 0.0) -> Frame:
    """按 ``viz_id`` 分发到四个可视化之一（帧里的数学全部来自 numcore）。"""
    try:
        fn = _DISPATCH[str(viz_id)]
    except KeyError:
        raise KeyError(f"4.1 章没有可视化 {viz_id!r}（可用：{sorted(_DISPATCH)}）") from None
    return fn(params, float(t))
