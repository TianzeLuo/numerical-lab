r"""6.1 章：浅水方程（Rusanov 有限体积）的精确解、守恒与失效边界。

四组证据：

* **A 精确解**：Ritter 干底解的扇区公式与 t→0 极限；Rusanov 对相同左右态退化为
  物理通量；波速公式 :math:`|u|+\\sqrt{gh}`。
* **A 误差窗口**：n=100/200/400 的干底溃坝 L1 误差在扇区/全域两个窗口上的差别；
  全域多出来的是干锋位置误差，不是干净的格式阶。
* **B 守恒**：周期湿域上质量到舍入量级守恒；干底质量漂移来自正性地板；
  静水 well-balanced 漂移 1e-16 量级，而朴素源项是 1e-3 量级。
* **B 稳定性**：cfl=0.8 稳定；cfl=2.0 出非有限值/把水抽干。临界值在 1.2-1.5 之间，
  不是某些教科书抄来的 1.07——本仓库只报实测。
"""

from __future__ import annotations

import math
import pathlib
import sys
import warnings

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcore.apps import shallow_water as sw  # noqa: E402
from numcore.verify import assertion, record  # noqa: E402


@assertion("A", theory="Rusanov 对相同左右态退化为物理通量；波速=|u|+sqrt(gh)；Ritter t→0 回到初始阶跃", chapter="6.1")
def test_flux_consistency_wave_speed_and_ritter_limit() -> None:
    h = np.array([0.8, 1.2, 2.0])
    hu = np.array([0.1, -0.3, 0.5])
    u = hu / h
    fh, fhu = sw.swe_flux(h, hu)
    rh, rhu = sw.rusanov_flux(h, hu, h, hu)
    a = sw.wave_speed(h, u)
    x = np.linspace(0.0, 1.0, 101)
    h0, u0 = sw.ritter_dam_break(x, 0.0)
    step_ok = bool(np.all(h0[x < 0.5] == 1.0) and np.all(h0[x > 0.5] == 0.0) and np.all(u0 == 0.0))
    ok = (
        np.allclose(fh, rh, atol=0.0)
        and np.allclose(fhu, rhu, atol=0.0)
        and np.allclose(a, np.abs(u) + np.sqrt(sw.G_DEFAULT * h), atol=1e-15)
        and step_ok
    )
    rec = record(
        label="Rusanov 一致性、波速与 t=0 极限",
        theory="相同态通量逐位相等；a=|u|+sqrt(gh)；t=0 时 h=1[x<0.5]，u=0",
        measured=f"max|ΔF_h|={np.max(np.abs(fh - rh)):.1e}，max|ΔF_hu|={np.max(np.abs(fhu - rhu)):.1e}，波速 {a.tolist()}",
        tol="逐位一致（0）；阶跃精确",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


@assertion("A", theory="Ritter 扇区 -cL<ξ<2cL 内 h=(2cL-ξ)²/(9g)、u=2/3(ξ+cL)；h>=0；扇外为 1/0", chapter="6.1")
def test_ritter_solution_formula_and_positivity() -> None:
    g = sw.G_DEFAULT
    c_l = math.sqrt(g)
    x = np.linspace(0.0, 1.0, 2001)
    t = 0.05
    h, u = sw.ritter_dam_break(x, t)
    xi = (x - 0.5) / t
    fan = (xi > -c_l) & (xi < 2.0 * c_l)
    h_want = (2.0 * c_l - xi[fan]) ** 2 / (9.0 * g)
    u_want = (2.0 / 3.0) * (xi[fan] + c_l)
    outside_left = xi <= -c_l
    outside_right = xi >= 2.0 * c_l
    ok = (
        np.allclose(h[fan], h_want, rtol=0.0, atol=1e-14)
        and np.allclose(u[fan], u_want, rtol=0.0, atol=1e-14)
        and np.all(h[outside_left] == 1.0)
        and np.all(u[outside_right] == 0.0)
        and np.all(h >= 0.0)
    )
    rec = record(
        label="Ritter 干底解的扇区公式与正性",
        theory="扇区公式逐点成立；扇外 h=1/0；h>=0",
        measured=f"扇内 max|Δh|={np.max(np.abs(h[fan] - h_want)):.2e}，max|Δu|={np.max(np.abs(u[fan] - u_want)):.2e}；h_min={h.min():.1e}",
        tol="扇内逐点 1e-14；正性",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


@assertion("A", theory="干底溃坝 L1 误差随 n 下降；扇区内阶约 0.7-0.9，全域含干锋误差更大", chapter="6.1")
def test_dam_break_error_windows_and_convergence() -> None:
    g = sw.G_DEFAULT
    c_l = math.sqrt(g)
    t_end = 0.05
    rows = []
    for n in (100, 200, 400):
        x, h0, hu0 = sw.dam_break_1d(n, right=0.0)
        p = sw.SWEParams(n=n, length=1.0, t_end=t_end, cfl=0.8, g=g, scheme="rusanov")
        res = sw.swe_solve_1d(p, h0=h0, hu0=hu0, bc="outflow")
        h_exact, _ = sw.ritter_dam_break(x, res.t)
        xi = (x - 0.5) / res.t
        tight = (xi > -0.8 * c_l) & (xi < 1.2 * c_l)
        err = np.abs(res.h - h_exact)
        base = max(float(np.sum(h_exact) * (x[1] - x[0])), 1e-30)
        l1_tight = float(np.sum(err[tight]) * (x[1] - x[0]) / base)
        l1_all = float(np.sum(err) * (x[1] - x[0]) / base)
        rows.append((n, l1_tight, l1_all, res.max_cfl, res.h_min))
    l1 = np.array([r[1] for r in rows])
    order = [math.log(l1[i - 1] / l1[i]) / math.log(2.0) for i in range(1, len(l1))]
    ratio = rows[-1][2] / rows[-1][1]
    ok = bool(np.all(np.diff(l1) < 0.0) and all(0.4 < o < 1.2 for o in order) and ratio > 1.3)
    rec = record(
        label="干底溃坝的窗口误差与阶",
        theory="误差单调下降；扇区阶约 0.7-0.9；全域/扇区比 >1.3（干锋误差）",
        measured="；".join(f"n={n}: tight={lt:.2e}, all={la:.2e}, CFL={mc:.2f}" for n, lt, la, mc, _ in rows)
        + f"；阶 {order}；全域/扇区={ratio:.2f}",
        tol="严格下降、0.4<阶<1.2、比值>1.3",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


@assertion("B", theory="周期湿域上质量守恒到舍入量级（通量差逐项相消）；干底漂移来自 min_depth 注入", chapter="6.1")
def test_periodic_mass_is_conserved() -> None:
    n = 100
    x = (np.arange(n, dtype=float) + 0.5) / n
    h0 = 1.0 + 0.1 * np.sin(2.0 * np.pi * x)
    hu0 = 0.2 * h0 * np.cos(2.0 * np.pi * x)
    p = sw.SWEParams(n=n, length=1.0, t_end=0.05, cfl=0.3, g=sw.G_DEFAULT, scheme="rusanov")
    res = sw.swe_solve_1d(p, h0=h0, hu0=hu0, bc="periodic")
    m0 = sw.total_mass(x, h0)
    drift = abs(res.mass - m0) / abs(m0)
    dry_x, dry_h, _ = sw.dam_break_1d(200, right=0.0)
    p_dry = sw.SWEParams(n=200, length=1.0, t_end=0.05, cfl=0.8, g=sw.G_DEFAULT, scheme="rusanov")
    dry = sw.swe_solve_1d(p_dry, h0=dry_h, hu0=np.zeros_like(dry_h), bc="outflow")
    dry_drift = abs(dry.mass - sw.total_mass(dry_x, dry_h)) / sw.total_mass(dry_x, dry_h)
    ok = drift <= 1e-12 and dry_drift > 1e-10
    rec = record(
        label="周期湿域与干底的质量漂移",
        theory="周期：<=1e-12；干底：>1e-10（正性地板注入，不是不守恒）",
        measured=f"periodic drift={drift:.2e}；dry drift={dry_drift:.2e}",
        tol="periodic<=1e-12 且 dry>1e-10",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


@assertion("B", theory="静水 well-balanced 漂移 1e-16 量级；朴素源项 1e-3 量级——差 12 个数量级", chapter="6.1")
def test_lake_at_rest_well_balanced_vs_naive() -> None:
    n = 100
    x, h, hu = sw.lake_at_rest(n, amplitude=0.1)
    _, b = sw.lake_bottom(n, amplitude=0.1)
    dx = x[1] - x[0]
    well = sw.lake_at_rest_metrics(h, hu, b, dx, steps=50, source="well_balanced")
    naive = sw.lake_at_rest_metrics(h, hu, b, dx, steps=50, source="naive")
    d_well = max(well["drift_momentum"], well["drift_surface"])
    d_naive = max(naive["drift_momentum"], naive["drift_surface"])
    ok = d_well < 1e-14 and d_naive > 1e-4
    rec = record(
        label="静水漂移：well-balanced vs 朴素源项",
        theory="well-balanced <1e-14；naive >1e-4；两者应差多个数量级",
        measured=f"well-balanced={d_well:.2e}，naive={d_naive:.2e}，比值 {d_naive / max(d_well, 1e-300):.1e}",
        tol="1e-14 / 1e-4",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


@assertion("B", theory="cfl=0.8 时湿溃坝稳定；cfl=2.0 时出非有限值或把水抽干——CFL 不是装饰", chapter="6.1")
def test_cfl_stability_bracket() -> None:
    x, h0, hu0 = sw.dam_break_1d(100, right=0.125)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stable = sw.swe_solve_1d(
            sw.SWEParams(n=100, length=1.0, t_end=0.2, cfl=0.8, scheme="rusanov"),
            h0=h0,
            hu0=hu0,
            bc="outflow",
        )
        bad = sw.swe_solve_1d(
            sw.SWEParams(n=100, length=1.0, t_end=0.2, cfl=2.0, scheme="rusanov"),
            h0=h0,
            hu0=hu0,
            bc="outflow",
            max_steps=3000,
        )
    stable_ok = bool(stable.reached_t_end and np.all(np.isfinite(stable.h)) and stable.h_min > 0.2)
    bad_failed = bool((not np.all(np.isfinite(bad.h))) or float(np.max(bad.h)) < 1.0e-6)
    rec = record(
        label="CFL 稳定/失稳的实测边界",
        theory="cfl=0.8：ht_end 到达、h_min>0.2；cfl=2.0：非有限或 h_max<1e-6",
        measured=(
            f"stable: reached={stable.reached_t_end}, h_min={stable.h_min:.3f}, maxCFL={stable.max_cfl:.2f}；"
            f"bad: finite={bool(np.all(np.isfinite(bad.h)))}, h_max={float(np.nanmax(bad.h)):.3e}, steps={bad.n_steps}"
        ),
        tol="cfl=0.8 通过；cfl=2.0 失败",
        passed=bool(stable_ok and bad_failed),
    )
    assert rec.passed, rec.measured


@assertion("A", theory="二维径向溃坝推进有限、质量与能量非负；出流边界下质量应减少而不是增加", chapter="6.1")
def test_radial_dam_break_2d_is_finite_and_physical() -> None:
    res = sw.swe_solve_2d(64, t_end=0.05, cfl=0.3, max_steps=300)
    m0 = float(res.mass_history[0])
    e0 = float(res.energy_history[0])
    ok = (
        np.all(np.isfinite(res.h))
        and np.all(np.isfinite(res.speed))
        and res.h_min > 0.0
        and res.mass > 0.0
        and res.mass <= m0 * 1.000001
        and res.energy > 0.0
        and res.energy <= e0 * 1.000001
        and res.reached_t_end
    )
    rec = record(
        label="二维径向溃坝的有限性、正性与出流质量",
        theory="h_min>0；出流边界质量/能量不增（允许舍入 1e-6）；推到 t_end",
        measured=(
            f"t={res.t:.4f}, steps={res.n_steps}, h_min={res.h_min:.3e}, "
            f"mass={res.mass:.6f} (m0={m0:.6f}), energy={res.energy:.6f} (e0={e0:.6f}), maxCFL={res.max_cfl:.2f}"
        ),
        tol="有限、正、非增、到达 t_end",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured
