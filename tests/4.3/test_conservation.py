r"""4.3 章断言（一）：有限体积的守恒性、熵条件与激波速度。

这一组只问「账本能不能被独立地量出来」：

* **守恒**：总质量变化必须只由边界通量解释。内部通量怎么错都不破坏这条——
  所以非守恒的 ``advective`` 支路是同一把尺子下的**负对照**。
* **RH**：激波速度由位置-时间线性拟合实测，再与 Rankine–Hugoniot 闭式对比。
* **熵**：跨声速稀疏波上，朴素迎风通量满足 RH、TV 不增、格式单调，
  只有熵收支翻正——这是本章唯一能抓住它的判据。
* **特征线**：破散时间有三个来源（闭式 / 网格扫描 / 相邻线交点），
  三条路线必须互相吻合到扫描误差。

所有「实测」都是真推进出来的，没有把理论值抄进 measured。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcore.pde import fvm  # noqa: E402
from numcore.pde import riemann as rr  # noqa: E402
from numcore.verify import assertion, record  # noqa: E402

CHAPTER = "4.3"


# --------------------------------------------------------------------------- #
# A 类：解析对拍
# --------------------------------------------------------------------------- #
@assertion("A", theory="Burgers 激波速度 s=(f(u_L)-f(u_R))/(u_L-u_R)=(u_L+u_R)/2=0.5",
           chapter=CHAPTER)
def test_burgers_shock_speed_matches_rankine_hugoniot() -> None:
    """实跑 n=400、t=0.5，追踪 0.5 等值线位置，线性拟合斜率。"""
    res = fvm.solve(n=400, t_end=0.5, cfl=0.4, ic="shock", scheme="godunov",
                    limiter="none", bc="outflow", keep_snapshots=True)
    ts, pos, ok = fvm.shock_track(res.snapshots, res.dx, level=0.5, times=res.snapshot_times)
    assert np.count_nonzero(ok) >= 20, "有效激波位置太少，拟合没有意义"
    fit = fvm.fit_speed(ts[ok > 0.5], pos[ok > 0.5])
    rel = abs(fit["speed"] - 0.5) / 0.5
    rec = record(label="Godunov 实测激波速度（n=400）", theory="RH: s=0.5",
                 measured=fit["speed"], tol="相对误差 <= 2e-3",
                 passed=(rel <= 2e-3 and fit["r2"] > 0.999), detail=f"r2={fit['r2']:.6f}, 相对误差={rel:.2e}")
    assert rec.passed, rec.measured


@assertion("A", theory="精确 Riemann 解是自相似的：u(x,t)=U((x-0.5)/t)，不同时刻采样到同一 xi 必须逐点一致",
           chapter=CHAPTER)
def test_riemann_solution_is_self_similar() -> None:
    """在两个时刻把精确解采样到相同的 xi 网格上（不是抄公式，是查询同一个解）。"""
    xi = np.linspace(-1.2, 1.2, 241)
    x1 = 0.5 + xi * 0.2
    x2 = 0.5 + xi * 0.45
    u1 = fvm.exact_riemann_solution(x1, 0.2, kind="burgers", u_left=1.0, u_right=0.0)["u"]
    u2 = fvm.exact_riemann_solution(x2, 0.45, kind="burgers", u_left=1.0, u_right=0.0)["u"]
    # 激波处（xi=0.5）采样点恰好落在间断上：剔除两侧一个点，比较连续区。
    keep = np.abs(xi - 0.5) > 1e-9
    err = float(np.max(np.abs(u1[keep] - u2[keep])))
    rec = record(label="同一 xi 在两个时刻的精确解最大差", theory="自相似 -> 0",
                 measured=err, tol=0.0, passed=(err == 0.0), detail=f"比较 {int(np.sum(keep))} 个点")
    assert rec.passed, rec.measured


@assertion("A", theory="Sod 星区值：p*=0.30313018、u*=0.92745262、rho_*L=0.42631943、rho_*R=0.26557371",
           chapter=CHAPTER)
def test_sod_star_values_match_the_textbook() -> None:
    wl, wr = rr.problem("sod")
    st = rr.euler_star_state(wl, wr)
    reference = {
        "p_star": 0.30313018,
        "u_star": 0.92745262,
        "rho_star_l": 0.42631943,
        "rho_star_r": 0.26557371,
    }
    worst = 0.0
    rows = []
    for key, want in reference.items():
        got = float(getattr(st, key))
        rel = abs(got - want) / abs(want)
        worst = max(worst, rel)
        rows.append(f"{key}={got:.8f}")
    rec = record(label="Sod 星区值 vs Toro 教科书", theory="p*=0.30313018 等",
                 measured=worst, tol=2e-5,
                 passed=(worst <= 2e-5 and not st.vacuum),
                 detail=f"最大相对差 {worst:.2e}；" + ", ".join(rows))
    assert rec.passed, rec.measured


@assertion("A", theory="跨激波的 Rankine–Hugoniot 跳跃条件 s[U]=[F] 必须贴舍入（按跳跃尺度归一）",
           chapter=CHAPTER)
def test_euler_shock_satisfies_all_three_rankine_hugoniot_conditions() -> None:
    wl, wr = rr.problem("sod")
    st = rr.euler_star_state(wl, wr)
    # 右激波：星区右态 -> 右初值，速度 s_right。
    resid = rr.euler_rh_residuals((st.rho_star_r, st.u_star, st.p_star), wr, st.s_right)
    rec = record(label="Sod 右激波三条 RH 残差的最大值", theory="<= 1e-13（双向归一）",
                 measured=resid["max"], tol=1e-13,
                 passed=(resid["max"] <= 1e-13),
                 detail=f"mass={resid['mass']:.2e}, momentum={resid['momentum']:.2e}, energy={resid['energy']:.2e}")
    assert rec.passed, rec.measured


@assertion("A", theory="接触间断：压强与速度连续、密度跳跃；euler_sample 在 xi=u* 两侧必须体现这一点",
           chapter=CHAPTER)
def test_euler_contact_is_continuous_in_pressure_and_velocity_only() -> None:
    wl, wr = rr.problem("sod")
    st = rr.euler_star_state(wl, wr)
    eps = 1e-8
    rho_l, u_l, p_l = rr.euler_sample(np.asarray([st.u_star - eps]), wl, wr, star=st)
    rho_r, u_r, p_r = rr.euler_sample(np.asarray([st.u_star + eps]), wl, wr, star=st)
    dp = float(abs(p_l[0] - p_r[0]))
    du = float(abs(u_l[0] - u_r[0]))
    drho = float(abs(rho_l[0] - rho_r[0]))
    ok = dp <= 1e-12 and du <= 1e-12 and drho > 0.1
    rec = record(label="接触间断两侧：|Δp|, |Δu|, |Δrho|", theory="dp=du=0，drho>0.1",
                 measured=f"dp={dp:.1e}, du={du:.1e}, drho={drho:.3f}", tol="dp,du<=1e-12",
                 passed=ok)
    assert rec.passed, rec.measured


@assertion("A", theory="HLLC 精确分辨静止接触间断：u=0,p=1 时通量必须是 [0,p,0]（与密度无关）",
           chapter=CHAPTER)
def test_hllc_preserves_a_stationary_contact_exactly() -> None:
    ul = rr.euler_state(1.0, 0.0, 1.0)
    ur = rr.euler_state(0.125, 0.0, 1.0)
    flux = rr.hllc_flux(ul, ur)
    pressure_flux = float(flux[1])
    rec = record(label="静止接触的 HLLC 质量/动量/能量通量", theory="[0, p, 0]",
                 measured=f"[{flux[0]:.1e}, {pressure_flux:.12f}, {flux[2]:.1e}]",
                 tol="|F0|,|F2|<=1e-14，|F1-1|<=1e-14",
                 passed=(abs(float(flux[0])) <= 1e-14 and abs(pressure_flux - 1.0) <= 1e-14
                         and abs(float(flux[2])) <= 1e-14))
    assert rec.passed, rec.measured


@assertion("A", theory="标量 Godunov 通量跨声速稀疏波取临界点 f'(u_s)=0；朴素迎风取 f(u_L)（两者不同）",
           chapter=CHAPTER)
def test_godunov_and_upwind_disagree_exactly_on_a_transonic_rarefaction() -> None:
    ul = np.asarray([-1.0])
    ur = np.asarray([1.0])
    fg = float(rr.godunov_flux("burgers", ul, ur)[0])
    fu = float(rr.upwind_flux("burgers", ul, ur)[0])
    # Burgers：f(u)=u^2/2，临界点 u_s=0 -> f=0；迎风取 max|f'| 一侧 -> f(1)=0.5。
    ok = abs(fg - 0.0) <= 1e-15 and abs(fu - 0.5) <= 1e-15
    rec = record(label="跨声速稀疏波的 Godunov vs upwind 通量", theory="Godunov=0, upwind=0.5",
                 measured=f"godunov={fg:.1f}, upwind={fu:.1f}", tol=1e-15, passed=ok)
    assert rec.passed, rec.measured


@assertion("A", theory="破散时间三条路线一致：闭式 1/(2πA)、扫描 -1/min u0'、相邻特征线首个交点",
           chapter=CHAPTER)
def test_breaking_time_three_routes_agree() -> None:
    amp = 0.5
    closed = fvm.breaking_time("sine", amplitude=amp)
    scan = fvm.breaking_time_numeric("sine", amplitude=amp, n_scan=20001)
    lines = fvm.characteristic_lines("sine", amplitude=amp, t_end=0.8, n_lines=40)
    rel_scan = abs(scan - closed) / closed
    rel_lines = abs(float(lines["t_break"]) - scan) / scan
    ok = rel_scan <= 1e-3 and rel_lines <= 2e-3
    rec = record(label="破散时间闭式/扫描/特征线交点", theory=f"闭式 {closed:.12f}",
                 measured=f"scan={scan:.9f}, lines={float(lines['t_break']):.9f}",
                 tol="相对差 <=2e-3", passed=ok,
                 detail=f"scan 相对 {rel_scan:.2e}，lines 相对 {rel_lines:.2e}")
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# B 类：不变量
# --------------------------------------------------------------------------- #
@assertion("B", theory="守恒格式总质量变化只由边界通量解释：|M-M0+(out-in)| ~ 舍入；非守恒支路漏 O(1)",
           chapter=CHAPTER)
def test_conservation_ledger_separates_conservative_from_advective() -> None:
    runs = {
        "godunov": fvm.solve(n=160, t_end=0.6, cfl=0.4, ic="shock", scheme="godunov",
                             limiter="none", bc="outflow"),
        "muscl": fvm.solve(n=160, t_end=0.6, cfl=0.4, ic="shock", scheme="godunov",
                           limiter="minmod", bc="outflow"),
        "advective": fvm.solve(n=160, t_end=0.6, cfl=0.4, ic="shock", scheme="advective",
                               limiter="none", bc="outflow"),
    }
    defects = {k: abs(float(v.max_defect)) for k, v in runs.items()}
    ok = defects["godunov"] <= 1e-13 and defects["muscl"] <= 1e-13 and defects["advective"] > 0.1
    rec = record(label="三本账的最大残差（godunov / muscl / advective）",
                 theory="守恒两条 ~1e-16，非守恒 ~0.3",
                 measured=f"{defects['godunov']:.2e} / {defects['muscl']:.2e} / {defects['advective']:.2e}",
                 tol="守恒 <=1e-13，非守恒 >0.1", passed=ok)
    assert rec.passed, rec.measured


@assertion("B", theory="TVD：单调格式的总变差不增（tv_history 逐点非增）",
           chapter=CHAPTER)
def test_total_variation_is_non_increasing() -> None:
    res = fvm.solve(n=200, t_end=0.4, cfl=0.4, ic="shock", scheme="godunov",
                    limiter="none", bc="outflow")
    tv = np.asarray(res.tv_history, dtype=float)
    rise = float(np.max(np.diff(tv))) if tv.size > 1 else 0.0
    tv0 = float(tv[0])
    tvend = float(tv[-1])
    # 末态的总变差会因激波被抹宽而下降，但**不许上升**。
    rec = record(label="Godunov 的总变差最大单步增量", theory="<= 1e-12（TVD）",
                 measured=rise, tol=1e-12, passed=(rise <= 1e-12),
                 detail=f"TV: {tv0:.6f} -> {tvend:.6f}，步数 {tv.size - 1}")
    assert rec.passed, rec.measured


@assertion("B", theory="熵收支：正确格式在稀疏波上熵产 <= 0；跨声速稀疏波上的迎风通量熵产 > 0",
           chapter=CHAPTER)
def test_entropy_budget_separates_godunov_from_upwind() -> None:
    god = fvm.solve(n=200, t_end=0.3, cfl=0.4, ic="rarefaction", scheme="godunov",
                    limiter="none", bc="outflow")
    upw = fvm.solve(n=200, t_end=0.3, cfl=0.4, ic="rarefaction", scheme="upwind",
                    limiter="none", bc="outflow")
    eg = float(god.entropy_production)
    eu = float(upw.entropy_production)
    ok = eg <= 1e-12 and eu > 1e-3
    rec = record(label="跨声速稀疏波末态熵产（Godunov / upwind）",
                 theory="Godunov <= 0，upwind > 0",
                 measured=f"godunov={eg:.3e}, upwind={eu:.3e}", tol="符号判据", passed=ok)
    assert rec.passed, rec.measured


@assertion("B", theory="Euler 的密度与压强保持为正（有限体积的硬不变量）",
           chapter=CHAPTER)
def test_euler_positivity_and_mass_ledger() -> None:
    wl, wr = rr.problem("sod")
    res = fvm.euler_solve(wl, wr, n=300, t_end=0.2, cfl=0.4, scheme="hllc", limiter="minmod")
    min_rho = float(np.min(res.min_rho_history))
    min_p = float(np.min(res.min_p_history))
    defect = float(np.max(np.abs(res.mass_defect)))
    ok = min_rho > 0.0 and min_p > 0.0 and defect <= 1e-12
    rec = record(label="Euler 最小密度/最小压强/质量残差",
                 theory="rho>0, p>0, |ΔM| <= 1e-12",
                 measured=f"min rho={min_rho:.6f}, min p={min_p:.6f}, defect={defect:.2e}",
                 tol="见 theory", passed=ok)
    assert rec.passed, rec.measured


@assertion("B", theory="周期边界下守恒格式的质量严格不变；质量变化只可能来自边界通量",
           chapter=CHAPTER)
def test_periodic_boundary_conserves_mass_exactly() -> None:
    res = fvm.solve(n=160, t_end=0.4, cfl=0.4, ic="sine", scheme="godunov",
                    limiter="minmod", bc="periodic")
    mass = np.asarray(res.mass_history, dtype=float)
    drift = float(np.max(np.abs(mass - mass[0])))
    rec = record(label="周期边界下总质量的最大漂移", theory="<= 1e-13",
                 measured=drift, tol=1e-13, passed=(drift <= 1e-13),
                 detail=f"M0={mass[0]:.12f}, Mend={mass[-1]:.12f}")
    assert rec.passed, rec.measured
