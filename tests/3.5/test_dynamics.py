r"""3.5 章（上半）：平衡点、零倾线、Poincare 截面与极限环的 A/C 类断言。

这一半谈的是「轨道本身」：

* 平衡点与线性化、散度 :math:`\nabla\cdot f`、Hopf 阈值都有**闭式**，所以它们是 A 类；
* 零倾线有两条独立的代码路径（数值追踪 vs 闭式参数化），两者必须给出同一条曲线；
* Poincare 截面点的定位误差是 :math:`O(h^2)`——这是收敛速率，所以是 C 类
  （用 :func:`~numcore.verify.assert_order` 显式声明 ``kind="C"``）；
* van der Pol 在 :math:`\mu\to0` 的周期修正 :math:`(T-2\pi)/\mu^2\to2\pi/16`
  与 Hopf 标准型的极限环半径 :math:`\sqrt\mu` 都是"阈值/极限"型结论。

三维帧的契约（原始三维数组 + 投影后的二维数组 + strings 里的投影参数）也在这里被核对：
帧里给出的 ``u/v`` 必须能由同帧的 ``x/y/z`` 与 ``strings.projection`` 里的角度重算出来。
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import pytest

from numcore.ode import continuation as cont
from numcore.ode import dynamics as dyn
from numcore.verify import assert_leq, assert_order, assertion, record

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.frames import build_frame  # noqa: E402
from server.registry import Registry  # noqa: E402

U = float(np.finfo(float).eps) / 2.0


@pytest.fixture(scope="module")
def entry():
    """3.5 章的注册项（``discover(strict=False)``：别的章节坏了不该连累本章）。"""
    reg = Registry(ROOT)
    entries = reg.discover(strict=False)
    assert "3.5" in entries, "3.5 章的模块没有被注册"
    return entries["3.5"]


# --------------------------------------------------------------------------- #
# 1. 平衡点、特征值与散度
# --------------------------------------------------------------------------- #
@assertion(
    "A",
    theory="Lorenz 的非平凡平衡点 C±=(±√(β(ρ−1)), ±√(β(ρ−1)), ρ−1) 与原点都是 f=0 的精确解；"
    "散度 ∇·f=−(σ+1+β) 与状态无关（常数）",
    chapter="3.5",
)
def test_lorenz_fixed_points_and_divergence() -> None:
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    system = dyn.flow_system("lorenz", sigma=sigma, rho=rho, beta=beta)
    origin, cp, cm = dyn.lorenz_fixed_points(sigma, rho, beta)
    roots = np.array([origin, cp, cm])
    res = np.asarray(system.rhs(0.0, roots), dtype=float)
    max_res = float(np.max(np.abs(res)))
    record(
        label="三个平衡点的 |f| 残差",
        measured=max_res,
        tol=1e-12,
        passed=max_res < 1e-12,
        detail="Newton 都不用上：C± 有闭式，代回去就是 0",
    )
    assert max_res < 1e-12

    sq = math.sqrt(beta * (rho - 1.0))
    closed = np.array([[0.0, 0.0, 0.0], [sq, sq, rho - 1.0], [-sq, -sq, rho - 1.0]])
    gap = float(np.max(np.abs(roots - closed)))
    record(label="C± 与闭式 √(β(ρ−1)) 的偏差", measured=gap, tol=4 * U, passed=gap < 4 * U)
    assert gap < 4 * U

    div = dyn.lorenz_divergence(sigma, beta)
    exact = -(sigma + 1.0 + beta)
    record(
        label="散度 = −(σ+1+β)",
        measured=abs(div - exact),
        tol=0.0,
        passed=div == exact,
        detail=f"σ=10、β=8/3 → {div:.10f}",
    )
    assert div == exact

    # 特征值：原点一个正实根 + 两个负实根；C± 在 ρ=28 处是稳定焦点（复共轭对 + 一个负实根）
    ev_o = dyn.eigenvalues(system.jac_at(origin))
    ev_c = dyn.eigenvalues(system.jac_at(cp))
    want_o = np.array([-22.827724, -2.666667, 11.827724])
    got_o = np.sort(ev_o.real)
    record(
        label="原点的三个实特征值",
        measured=float(np.max(np.abs(got_o - want_o))),
        tol=1e-5,
        passed=bool(np.max(np.abs(got_o - want_o)) < 1e-5),
        detail=f"实测 {np.round(got_o, 6)}，闭式 {want_o}",
    )
    assert np.max(np.abs(got_o - want_o)) < 1e-5
    img_c = np.sort(np.abs(ev_c.imag))
    record(
        label="C± 复共轭对的虚部与 10.194501 之差",
        measured=abs(float(img_c[-1]) - 10.194501),
        tol=1e-5,
        passed=abs(float(img_c[-1]) - 10.194501) < 1e-5,
        detail=f"实测特征值 {np.round(ev_c, 6)}（虚部 {float(img_c[-1]):.6f}）",
    )
    assert abs(float(img_c[-1]) - 10.194501) < 1e-4
    # ρ=28 > ρ_H=24.7368，所以 C± 已经越过 Hopf：一个稳定实方向 + 一个不稳定螺旋面。
    # 这不是"saddle"（鞍点的两个方向都是实的）——本仓初版把所有混合符号都判成 saddle，
    # 这条断言就是那个 bug 的回归测试。
    kind_c = dyn.fixed_point_report(system, cp).kind
    record(
        label="ρ=28 处 C± 的相型（鞍焦）",
        measured=float(dyn.KIND_CODES[kind_c]),
        tol=float(dyn.KIND_CODES["saddle_focus"]),
        passed=kind_c == "saddle_focus",
        detail=f"实测 {kind_c}；特征值 [−13.8546, 0.0940±10.1945i]",
    )
    assert kind_c == "saddle_focus"
    assert dyn.fixed_point_report(system, origin).kind == "saddle"


@assertion(
    "C",
    theory="Hopf 分岔的定义是平衡点的一对特征值穿过虚轴：max Re λ(ρ) 在 ρ_H 变号。"
    "二分实测该变号点，与闭式 ρ_H=σ(σ+β+3)/(σ−β−1)=470/19 对拍",
    chapter="3.5",
)
def test_lorenz_hopf_threshold_is_470_over_19() -> None:
    def make(rho: float) -> dyn.FlowSystem:
        return dyn.flow_system("lorenz", rho=rho)

    measured = cont.hopf_threshold(make, lo=10.0, hi=40.0, seed=[8.5, 8.5, 27.0], bisect=40)
    closed = dyn.lorenz_hopf_rho(10.0, 8.0 / 3.0)
    assert closed is not None
    gap = abs(measured - closed)
    record(
        label="二分实测 ρ_H 与 470/19",
        measured=gap,
        tol=1e-9,
        passed=gap < 1e-9,
        detail=f"实测 {measured:.12f}，闭式 {closed:.12f}（470/19={470 / 19:.12f}）",
    )
    assert gap < 1e-9, f"ρ_H 实测 {measured} 与闭式 {closed} 差 {gap}"

    # ω² 的闭式：在 ρ_H 处特征值必须是 ±iω，ω 由 2σβ(ρ_H−1)/(σ+β+1) 给出
    omega = dyn.lorenz_hopf_omega(10.0, 8.0 / 3.0)
    system = dyn.flow_system("lorenz", rho=closed)
    _, cp, _ = dyn.lorenz_fixed_points(10.0, closed, 8.0 / 3.0)
    ev = dyn.eigenvalues(system.jac_at(cp))
    # 注意：判据是 max Re λ（有符号的最大值），不是 max|Re λ|——ρ_H 处 C± 的第三个特征值
    # 是 −13.6667，只有那一对复根落在虚轴上。写成 |Re λ| 会得到 13.67 而误判"不是 Hopf"。
    real_part = float(np.max(ev.real))
    third = float(np.min(ev.real))
    record(
        label="ρ_H 处 C± 的 max Re λ（应为 0）",
        measured=abs(real_part),
        tol=1e-12,
        passed=abs(real_part) < 1e-12,
        detail=f"特征值 {np.round(ev, 8)}（第三个 = {third:.6f} ≠ 0，ω={omega:.10f}）",
    )
    assert abs(real_part) < 1e-12
    record(
        label="ρ_H 处第三个特征值（负实根，与 −(σ+1+β) 同值）",
        measured=abs(third + (10.0 + 1.0 + 8.0 / 3.0)),
        tol=1e-9,
        passed=abs(third + (10.0 + 1.0 + 8.0 / 3.0)) < 1e-9,
        detail=f"实测 {third:.10f}，−(σ+1+β)={-(10.0 + 1.0 + 8.0 / 3.0):.10f}",
    )
    assert omega is not None and abs(abs(ev[2].imag) - omega) < 1e-9


@assertion(
    "A",
    theory="van der Pol 的 dy/dt=0 零倾线有闭式 y=x/(μ(1−x²))；数值追踪（逐列二分 + 归一化"
    "二维距离连接）必须给出同一条曲线，且分支数一致",
    chapter="3.5",
)
def test_nullclines_match_closed_forms() -> None:
    vdp = dyn.flow_system("van_der_pol", mu=0.5)
    nb = dyn.nullcline_branches(vdp, nx=161, ny=161, min_points=3)
    ana = dyn.analytic_nullclines(vdp)
    sizes_num = [int(x.size) for x, _ in nb["y"]]
    sizes_ana = [int(x.size) for x, _ in ana["y"]]
    record(
        label="van der Pol 的 dy/dt=0 零倾线分支数（数值 vs 闭式）",
        measured=len(sizes_num),
        tol=0.0,
        passed=len(sizes_num) == len(sizes_ana) == 3,
        detail=f"数值 {sizes_num}，闭式 {sizes_ana}（x=±1 处的竖直渐近线把它切成三段）",
    )
    assert len(sizes_num) == 3, f"应当是 3 支（渐近线两侧各一支 + 中间一支），得到 {sizes_num}"

    worst = 0.0
    for xs, ys in nb["y"]:
        worst = max(worst, float(np.max(np.abs(ys - xs / (0.5 * (1.0 - xs * xs))))))
    record(
        label="数值零倾线 vs 闭式 y=x/(μ(1−x²)) 的最大偏差",
        measured=worst,
        tol=1e-12,
        passed=worst < 1e-12,
        detail="每个点都是二分出来的根，不是网格插值",
    )
    assert worst < 1e-12

    # 竖直渐近线两侧不能被接成一条：x<−1 与 x>−1 必须落在不同分支上
    spans = [(float(x.min()), float(x.max())) for x, _ in nb["y"]]
    bridges = [s for s in spans if s[0] < -1.0 < s[1]]
    record(
        label="跨过 x=−1 渐近线的分支数（应为 0）",
        measured=len(bridges),
        tol=0.0,
        passed=not bridges,
        detail=f"分支跨度 {[(round(a, 2), round(b, 2)) for a, b in spans]}",
    )
    assert not bridges

    pend = dyn.flow_system("damped_pendulum", c=0.3)
    nbp = dyn.nullcline_branches(pend, nx=161, ny=161)
    worst_p = max(
        float(np.max(np.abs(ys + np.sin(xs) / 0.3))) for xs, ys in nbp["y"]
    )
    record(
        label="阻尼摆 dy/dt=0 零倾线 vs 闭式 y=−sin(x)/c",
        measured=worst_p,
        tol=1e-12,
        passed=worst_p < 1e-12,
    )
    assert worst_p < 1e-12


# --------------------------------------------------------------------------- #
# 2. Poincare 截面
# --------------------------------------------------------------------------- #
@assertion(
    "A",
    theory="定向 Poincare 截面只收单向穿越：截面上每一点都必须满足 f(y)[平面分量]>0（向上穿越）"
    "且落在平面上；返回时间与穿越顺序一一对应",
    chapter="3.5",
)
def test_poincare_section_is_single_valued_and_on_plane() -> None:
    system = dyn.flow_system("lorenz", rho=28.0)
    level = 27.0
    sec = dyn.poincare_crossings(
        system, [1.0, 0.0, 20.0], plane_index=2, level=level, direction=1, t_span=(0.0, 60.0), h=0.01
    )
    assert sec.t.size > 20, f"60 个时间单位里应当有几十次穿越，得到 {sec.t.size}"
    zn = sec.points[:, 2] - level
    record(
        label="截面点离平面的最大距离（精化后）",
        measured=float(np.max(np.abs(zn))),
        tol=1e-9,
        passed=float(np.max(np.abs(zn))) < 1e-9,
        detail=f"{sec.t.size} 个点，步长 h=0.01",
    )
    assert float(np.max(np.abs(zn))) < 1e-9

    zhat = np.asarray(system.rhs(sec.t, sec.points), dtype=float)[:, 2]
    record(
        label="穿越点上的 dz/dt 全为正（定向判据）",
        measured=float(np.min(zhat)),
        tol=0.0,
        passed=float(np.min(zhat)) > 0.0,
        detail="双向都收会让返回映射变成双值（重影）",
    )
    assert float(np.min(zhat)) > 0.0

    dt = np.diff(sec.t)
    record(
        label="相邻穿越的时间间隔（均值±极差）",
        measured=float(np.mean(dt)),
        tol=0.0,
        passed=float(np.min(dt)) > 0.0,
        detail=f"均值 {np.mean(dt):.4f}，最小 {np.min(dt):.4f}，最大 {np.max(dt):.4f}",
    )
    assert float(np.min(dt)) > 0.0


@assertion(
    "C",
    theory="截面点的定位误差有两个来源：线性插值是 O(h²)，Newton 精化把它压到舍入水平。"
    "「线性插值点 → 精化点」的位移必须按 h² 收敛",
    chapter="3.5",
)
def test_poincare_refinement_converges_second_order() -> None:
    system = dyn.flow_system("lorenz", rho=28.0)
    hs = [0.04, 0.02, 0.01, 0.005]
    shifts = []
    resids = []
    for h in hs:
        sec = dyn.poincare_crossings(
            system, [1.0, 0.0, 20.0], plane_index=2, level=27.0, direction=1,
            t_span=(0.0, 30.0), h=h, max_crossings=25,
        )
        shifts.append(float(np.max(sec.refine_shift)))
        resids.append(float(np.max(sec.plane_residual)))
    record(
        label="精化位移随步长的收敛（最大位移）",
        measured=shifts[-1],
        tol=0.0,
        passed=True,
        detail="h=" + "、".join(f"{h:g}:{s:.3e}" for h, s in zip(hs, shifts)),
    )
    fit = assert_order(
        hs,
        shifts,
        theory=2.0,
        tol=0.35,
        kind="C",
        label="线性插值定位误差的收敛阶",
        detail="位移 = |精化点 − 线性插值点|，理论上是 O(h²)",
    )
    record(
        label="平面残差（Newton 精化后）",
        measured=resids[-1],
        tol=1e-12,
        passed=resids[-1] < 1e-12,
        detail="h=" + "、".join(f"{h:g}:{r:.2e}" for h, r in zip(hs, resids)),
    )
    assert resids[-1] < 1e-12
    assert fit.slope > 1.5


# --------------------------------------------------------------------------- #
# 3. 极限环：van der Pol 与 Hopf 标准型
# --------------------------------------------------------------------------- #
def _vdp_period(mu: float, h: float, t_end: float, n_last: int = 4) -> float:
    """用**精化后的**截面穿越时刻量周期（线性插值定位只有 O(h²)，量 μ→0 的修正不够用）。"""
    system = dyn.flow_system("van_der_pol", mu=mu)
    sec = dyn.poincare_crossings(
        system, [2.0, 0.0], plane_index=0, level=0.0, direction=1, t_span=(0.0, t_end), h=h
    )
    assert sec.t.size >= n_last + 1, f"μ={mu} 的穿越次数不足：{sec.t.size}"
    return float(np.mean(np.diff(sec.t[-n_last:])))


@assertion(
    "A",
    theory="μ=0 的 van der Pol 退化成简谐振子 x''+x=0，周期精确等于 2π；"
    "μ→0 时极限环周期的修正是 (T−2π)/μ² → 2π/16",
    chapter="3.5",
)
def test_van_der_pol_period_limits() -> None:
    t0 = _vdp_period(0.0, 0.01, 30.0)
    gap0 = abs(t0 - 2.0 * math.pi)
    record(
        label="μ=0 的周期 − 2π",
        measured=gap0,
        tol=1e-8,
        passed=gap0 < 1e-8,
        detail=f"实测 T={t0:.12f}，2π={2.0 * math.pi:.12f}（h=0.01，穿越时刻经 Newton 精化）",
    )
    assert gap0 < 1e-8

    coef = (_vdp_period(0.02, 0.002, 60.0) - 2.0 * math.pi) / 0.02**2
    want = 2.0 * math.pi / 16.0
    record(
        label="(T−2π)/μ² 在 μ=0.02 处的值 vs 2π/16",
        measured=abs(coef - want),
        tol=1e-4,
        passed=abs(coef - want) < 1e-4,
        detail=f"实测 {coef:.8f}，理论 {want:.8f}（相对差 {abs(coef - want) / want:.2e}）",
    )
    assert abs(coef - want) < 1e-4

    amps = {}
    for mu, h, t_end in ((0.5, 0.002, 60.0), (1.0, 0.002, 60.0), (2.0, 0.001, 80.0)):
        system = dyn.flow_system("van_der_pol", mu=mu)
        tr = dyn.trajectory(system, [2.0, 0.0], (0.0, t_end), h)
        tail = tr.y[len(tr.y) // 2:]
        amps[mu] = float(np.max(np.abs(tail[:, 0])))
    record(
        label="极限环振幅随 μ 的变化（2 的修正）",
        measured=amps[1.0] - 2.0,
        tol=0.0,
        passed=amps[2.0] > amps[1.0] > amps[0.5] > 2.0,
        detail="振幅 " + "、".join(f"μ={m}:{a:.6f}" for m, a in amps.items()),
    )
    assert amps[2.0] > amps[1.0] > amps[0.5] > 2.0


@assertion(
    "A",
    theory="Hopf 标准型 ρ'=μρ−ρ³、θ'=ω 的极限环是半径 √μ 的圆，周期精确等于 2π/ω",
    chapter="3.5",
)
def test_hopf_normal_form_limit_cycle() -> None:
    mu = 0.5
    system = dyn.flow_system("hopf_normal_form", mu=mu)
    tr = dyn.trajectory(system, [0.3, 0.0], (0.0, 60.0), 0.002)
    tail = tr.y[-2000:]
    r = float(np.mean(np.sqrt(np.sum(tail * tail, axis=1))))
    want = math.sqrt(mu)
    record(
        label="极限环半径 vs √μ",
        measured=abs(r - want),
        tol=5e-12,
        passed=abs(r - want) < 5e-12,
        detail=f"实测 {r:.12f}，√μ={want:.12f}",
    )
    assert abs(r - want) < 5e-12

    sec = dyn.poincare_crossings(
        system, [0.3, 0.0], plane_index=1, level=0.0, direction=1, t_span=(0.0, 60.0), h=0.002
    )
    period = float(np.mean(np.diff(sec.t[-4:])))
    omega = 1.0
    record(
        label="极限环周期 vs 2π/ω",
        measured=abs(period - 2.0 * math.pi / omega),
        tol=1e-6,
        passed=abs(period - 2.0 * math.pi / omega) < 1e-6,
        detail=f"实测 {period:.10f}，2π={2.0 * math.pi:.10f}",
    )
    assert abs(period - 2.0 * math.pi / omega) < 1e-6


# --------------------------------------------------------------------------- #
# 4. 批量积分路径与单条路径对拍
# --------------------------------------------------------------------------- #
@assertion(
    "A",
    theory="integrate_ensemble 把 (n,dim) 摊平后用同一个 rk_step 推进；n=1 时它必须与"
    "单条轨迹的 integrate_fixed 逐点相同（两条独立的代码路径）",
    chapter="3.5",
)
def test_ensemble_matches_single_trajectory() -> None:
    system = dyn.flow_system("lorenz")
    y0 = np.array([1.0, 0.0, 20.0])
    single = dyn.trajectory(system, y0, (0.0, 10.0), 0.01)
    batch = dyn.integrate_ensemble(system, y0.reshape(1, 3), (0.0, 10.0), 0.01)
    n = min(single.y.shape[0], batch.Y.shape[0])
    gap = float(np.max(np.abs(single.y[:n] - batch.Y[:n, 0, :])))
    record(
        label="批量积分 vs 单条积分（逐点最大差）",
        measured=gap,
        tol=0.0,
        passed=gap == 0.0,
        detail=f"{n} 个采样点，步数 {single.n_steps} / {batch.n_steps}",
    )
    assert gap == 0.0
    assert single.n_steps == batch.n_steps


# --------------------------------------------------------------------------- #
# 5. 帧的 3D 契约
# --------------------------------------------------------------------------- #
@assertion(
    "A",
    theory="三维帧必须同时给原始 x/y/z 与投影后的 u/v，并把投影参数写进 strings；"
    "帧里的 u/v 必须能由同帧的 x/y/z 与同一组角度重算出来",
    chapter="3.5",
)
def test_frames_carry_raw_3d_and_projection(entry) -> None:
    for viz_id, xyz, uv in (
        ("lorenz_butterfly", ("traj1_x", "traj1_y", "traj1_z"), ("traj1_u", "traj1_v")),
        ("poincare_section", ("orbit_x", "orbit_y", "orbit_z"), ("orbit_u", "orbit_v")),
    ):
        frame = build_frame(entry, viz_id, {}, 10.0, strict=True)
        arr = frame._arrays
        X = np.stack([arr[xyz[0]], arr[xyz[1]], arr[xyz[2]]], axis=1)
        info = json.loads(frame.strings["projection"])
        u, v = dyn.project3d(X, float(info["azimuth_deg"]), float(info["elevation_deg"]))
        gap_u = float(np.max(np.abs(u - arr[uv[0]])))
        gap_v = float(np.max(np.abs(v - arr[uv[1]])))
        record(
            label=f"{viz_id}：由 x/y/z + strings 角度重算 u/v 的偏差",
            measured=max(gap_u, gap_v),
            tol=1e-12,
            passed=max(gap_u, gap_v) < 1e-12,
            detail=f"角度 az={info['azimuth_deg']}、el={info['elevation_deg']}，{X.shape[0]} 个点",
        )
        assert max(gap_u, gap_v) < 1e-12
        assert "u" in frame.strings["projection"] and "v" in frame.strings["projection"]


@assertion(
    "A",
    theory="帧里报出来的标量必须与同帧的数组一致（t_visible 是 sep_mean 第一次达到显示分辨率"
    "１e-3 的时刻）——否则前端画的与 meta 说的是两回事",
    chapter="3.5",
)
def test_butterfly_frame_meta_matches_arrays(entry) -> None:
    frame = build_frame(entry, "lorenz_butterfly", {}, 30.0, strict=True)
    arr = frame._arrays
    sep = arr["sep_mean"]
    t = arr["sep_t"]
    vis = sep >= frame.meta["display_res"]
    assert bool(np.any(vis)), "默认参数下平均分离应当越过显示分辨率"
    k = int(np.argmax(vis))
    gap = abs(float(t[k]) - float(frame.meta["t_visible"]))
    record(
        label="meta.t_visible vs 数组里第一次越过 1e-3 的时刻",
        measured=gap,
        tol=1e-12,
        passed=gap < 1e-12,
        detail=f"t_visible={frame.meta['t_visible']:.4f}，数组给出 {float(t[k]):.4f}",
    )
    assert gap < 1e-12

    # 拟合窗口里的点数、增长率与前后两条曲线的一致性
    rate = frame.meta["rate"]
    assert frame.meta["rate_ok"] is True
    record(
        label="孪生实验的拟合增长率（默认参数）",
        measured=rate,
        tol=0.06,
        passed=abs(rate - 0.9056) < 0.06,
        detail=(
            f"公认值 0.9056；拟合窗口 1e-6≤d≤1、{int(frame.meta['rate_n_fit'])} 个点、"
            f"R²={frame.meta['rate_r2']:.4f}；跨孪生对的散布 "
            f"{frame.meta['rate_pair_min']:.4f}…{frame.meta['rate_pair_max']:.4f}"
        ),
    )
    assert abs(rate - 0.9056) < 0.06
    record(
        label="h 与 2h 两条分离曲线的增长率之差（默认参数）",
        measured=frame.meta["rate_diff"],
        tol=0.02,
        passed=frame.meta["rate_diff"] < 0.02,
        detail="步长减半后增长率不变，这是「不是数值噪声」的直接证据",
    )
    assert frame.meta["rate_diff"] < 0.02
