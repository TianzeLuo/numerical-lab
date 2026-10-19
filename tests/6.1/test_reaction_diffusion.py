r"""6.1 章：Gray--Scott 反应扩散与图灵判据。

这一文件钉住三件事：

* **底态选对**：非平凡稳态有两个分支，只有 u_-（小 u、大 v）是稳定支；
  u_+ 的 det J<0，是鞍点。经典 spots 预设根本没有非平凡稳态——它的平凡态
  永远不会扩散失稳。
* **判据不是"图案好看"**：turing 预设 λ*=+0.015>0 且底态稳定；spots/stripes/waves
  的最快增长模都在 φ=0，没有正波数增长带：spots/waves 没有非平凡稳态，
  stripes 的底态存在但均匀模已失稳（λ(0)=+0.0054）。det 的顶点不是增长率最大点
  （φ*=53.90 vs det 顶点 62.75，λ* 高 12%）。
* **离散化**：显式 dt 上限只由最大扩散系数决定，越界必须夹紧并如实报告；
  五点模板的有效波数给出 O(Δx²) 的增长率误差；质量只能被反应项改变。
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcore.apps import reaction_diffusion as rd  # noqa: E402
from numcore.verify import assertion, record  # noqa: E402


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    mx, my = float(lx.mean()), float(ly.mean())
    return float(np.sum((lx - mx) * (ly - my)) / np.sum((lx - mx) ** 2))


@assertion("A", theory="Gray--Scott 非平凡稳态的稳定支是 u_-；turing 预设底态 (0.25,0.3)，det J>0", chapter="6.1")
def test_turing_base_state_selects_stable_branch() -> None:
    du, dv, feed, kill = rd.RD_PRESETS["turing"]
    base = rd.turing_base_state(feed, kill)
    assert base is not None
    u_star, v_star = base
    disc = math.sqrt(1.0 - 4.0 * (feed + kill) ** 2 / feed)
    u_plus = 0.5 * (1.0 + disc)
    det_stable = (feed + kill) * (v_star * v_star - feed)
    v_plus = (feed + kill) / u_plus
    det_unstable = (feed + kill) * (v_plus * v_plus - feed)
    classic = rd.turing_base_state(*rd.RD_PRESETS["spots"][2:])
    ok = (
        abs(u_star - 0.25) < 1e-12
        and abs(v_star - 0.3) < 1e-12
        and det_stable > 0.0
        and det_unstable < 0.0
        and classic is None
    )
    rec = record(
        label="图灵底态的两支与稳定性",
        theory="u_-=(0.25,0.3)，det J=+0.0045；u_+ 分支 det<0；spots 无非平凡稳态",
        measured=(
            f"u_-=({u_star:.6f},{v_star:.6f}), det={det_stable:.6f}；"
            f"u_+=({u_plus:.6f},{v_plus:.6f}), det={det_unstable:.6f}；spots base={classic}"
        ),
        tol="u_/v_ 精确、det 符号正确",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


@assertion("A", theory="turing 预设 φ*=53.90、λ*=+0.0150；det 顶点 62.75 给 0.0134，低 12%", chapter="6.1")
def test_most_unstable_mode_beats_det_vertex() -> None:
    du, dv, feed, kill = rd.RD_PRESETS["turing"]
    phi, lam = rd.most_unstable_mode(du, dv, feed, kill)
    u, v = rd.turing_base_state(feed, kill)  # type: ignore[misc]
    fu, fv = -v * v - feed, -2.0 * u * v
    gu, gv = v * v, 2.0 * u * v - (feed + kill)
    a2 = du * gv + dv * fu
    z_det = a2 / (2.0 * du * dv)
    phi_det = math.sqrt(z_det)
    lam_det = float(rd.turing_growth_rate(phi_det, du, dv, feed, kill))
    classic = [rd.most_unstable_mode(*rd.RD_PRESETS[name]) for name in ("spots", "stripes", "waves")]
    stripes_base = rd.turing_base_state(*rd.RD_PRESETS["stripes"][2:])
    stripes_lam0 = float(rd.turing_growth_rate(0.0, *rd.RD_PRESETS["stripes"]))
    ok = (
        abs(phi - 53.898800177) < 1e-5
        and abs(lam - 0.0150170086) < 1e-8
        and phi < phi_det
        and lam > lam_det
        and (lam - lam_det) / lam > 0.1
        and all(p <= 1e-12 for p, _ in classic)
        and stripes_base is not None
        and stripes_lam0 > 0.0
    )
    rec = record(
        label="最不稳定波数与 det 顶点",
        theory=(
            "φ*=53.899、λ*=0.015017；det 顶点 φ=62.75、λ=0.013392；经典三预设最快模都在 "
            "φ=0，其中 stripes 的 λ(0)=+0.0054 是均匀失稳，不是图灵带"
        ),
        measured=(
            f"φ*={phi:.6f}，λ*={lam:.7f}；det 顶点 φ={phi_det:.4f}，λ={lam_det:.7f}，"
            f"低 {(1.0 - lam_det / lam) * 100:.1f}%；经典预设 "
            + "、".join(f"({p:.2f},{l:.4f})" for p, l in classic)
            + f"；stripes λ(0)={stripes_lam0:+.4f}"
        ),
        tol="φ/λ 对拍 1e-5 内；λ*>λ_det 且高 >10%；经典预设无 φ>0 增长模",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


@assertion("B", theory="显式 dt 超过 Δx²/(4 max(Du,Dv))/2 时被夹紧并报告；clamp=False 必须报错", chapter="6.1")
def test_dt_limit_clamps_instead_of_silently_diverging() -> None:
    du, dv, feed, kill = rd.RD_PRESETS["turing"]
    n = 128
    limit = rd.diffusion_dt_limit(n, du, dv)
    params = rd.RDParams(n=n, du=du, dv=dv, feed=feed, kill=kill, dt=0.25, steps=1)
    res = rd.gray_scott_solve(params, seed=20261019)
    raised = False
    with pytest.raises(ValueError):
        rd.gray_scott_solve(params, seed=20261019, clamp=False)
    raised = True
    ok = res.dt_clamped and abs(res.dt - limit) < 1e-15 and res.dt_requested == 0.25 and raised and res.finite
    rec = record(
        label="dt=0.25 在 n=128 的 turing 预设上被夹紧",
        theory="limit=0.0953674；实际 dt=limit，dt_requested=0.25；clamp=False 抛 ValueError",
        measured=f"limit={limit:.7f}，dt={res.dt:.7f}，clamped={res.dt_clamped}，requested={res.dt_requested}",
        tol="dt==limit 且 clamped、有限；拒绝越界请求",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


@assertion("B", theory="周期域上 Σu 的变化率只由反应项决定（扩散项逐点和为 0）；每步质量差=dt·Σ反应", chapter="6.1")
def test_mass_changes_only_by_reaction() -> None:
    n = 32
    du, dv, feed, kill = rd.RD_PRESETS["turing"]
    u, v = rd.initial_condition(n, kind="random", seed=20261019)
    dt = 0.2
    dx = 1.0 / n
    mass0 = float(u.sum())
    predicted = mass0
    detail = []
    for _ in range(5):
        predicted += dt * float(rd.reaction_balance(u, v, feed, kill))
        u, v = rd.gray_scott_step(u, v, du, dv, feed, kill, dx, dt)
        detail.append(f"{float(u.sum()) - predicted:.2e}")
    actual = float(u.sum())
    err = abs(actual - predicted) / max(abs(predicted), 1.0)
    rec = record(
        label="五步内 Σu 变化与 dt·Σ反应 的逐位对账",
        theory="每步差 <= 1e-12（相对）；扩散项的周期和恒为 0",
        measured=f"predicted={predicted:.12f}，actual={actual:.12f}，相对差={err:.2e}；逐步 {detail}",
        tol="相对差 <= 1e-12",
        passed=bool(err <= 1e-12),
    )
    assert rec.passed, rec.measured


@assertion("A", theory="五点模板有效波数 φ_eff=(2/Δx)sin(φΔx/2) 的增长率相对误差随 n 按 O(Δx²) 收敛", chapter="6.1")
def test_stencil_dispersion_error_converges_second_order() -> None:
    du, dv, feed, kill = rd.RD_PRESETS["turing"]
    phi_star, _ = rd.most_unstable_mode(du, dv, feed, kill)
    m = max(1, int(round(phi_star / (2.0 * math.pi))))
    phi = 2.0 * math.pi * m
    lam = float(rd.turing_growth_rate(phi, du, dv, feed, kill))
    ns = np.array([64.0, 128.0, 256.0, 512.0])
    err = []
    for n in ns:
        dx = 1.0 / n
        phi_eff = 2.0 / dx * math.sin(phi * dx / 2.0)
        err.append(abs(float(rd.turing_growth_rate(phi_eff, du, dv, feed, kill)) - lam) / abs(lam))
    err_a = np.asarray(err)
    slope = _slope(ns, err_a)
    ok = bool(np.all(np.diff(err_a) < 0.0) and 1.5 < -slope < 2.3)
    rec = record(
        label="五点模板增长率误差的网格收敛",
        theory="误差按 n^-2 下降；n=64/128/256/512 的 log-log 斜率约 -2",
        measured="、".join(f"{n:.0f}:{e:.2e}" for n, e in zip(ns, err_a)) + f"；斜率 {slope:.2f}",
        tol="-slope ∈ (1.5, 2.3)",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


@assertion("B", theory="经典 spots 的 λ*=-0.035<0 却仍长出振幅 0.7 的斑图：图案不是图灵判据", chapter="6.1")
def test_classic_pattern_grows_without_turing_instability() -> None:
    du, dv, feed, kill = rd.RD_PRESETS["spots"]
    phi, lam = rd.most_unstable_mode(du, dv, feed, kill)
    u0, v0 = rd.initial_condition(64, kind="spots", seed=20261019)
    res = rd.gray_scott_solve(rd.RDParams(n=64, du=du, dv=dv, feed=feed, kill=kill, dt=0.25, steps=4000), u0=u0, v0=v0, seed=20261019)
    ok = phi <= 1e-12 and lam < 0.0 and res.amplitude > 0.5 and res.finite
    rec = record(
        label="经典 spots 的 λ* 与饱和振幅",
        theory="λ*=-0.035<0，但 4000 步后振幅>0.5：自复制瞬态，不是图灵失稳",
        measured=f"φ*={phi:.3f}，λ*={lam:.4f}，振幅={res.amplitude:.3f}，斑点={res.spot_count}，t={res.time:.1f}",
        tol="λ*<0 且 amplitude>0.5",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


@assertion("A", theory="图灵预设饱和图案的主波长接近 2π/φ*=0.1166；显式 dt 夹紧后仍有限", chapter="6.1")
def test_turing_pattern_saturates_near_analytic_wavelength() -> None:
    du, dv, feed, kill = rd.RD_PRESETS["turing"]
    phi, _ = rd.most_unstable_mode(du, dv, feed, kill)
    u0, v0 = rd.initial_condition(128, kind="spots", seed=20261019)
    res = rd.gray_scott_solve(rd.RDParams(n=128, du=du, dv=dv, feed=feed, kill=kill, dt=0.25, steps=10000), u0=u0, v0=v0, seed=20261019)
    lam_theory = 2.0 * math.pi / phi
    measured = rd.dominant_wavelength(res.u, res.dx)
    rel = abs(measured - lam_theory) / lam_theory
    ok = res.finite and rel < 0.12 and res.amplitude > 0.2 and res.spot_count >= 40
    rec = record(
        label="图灵饱和图案的波长与振幅",
        theory="理论波长 2π/φ*=0.1166；实测应在 12% 内，振幅>0.2，局部极大>=40",
        measured=f"实测波长={measured:.4f}，理论={lam_theory:.4f}，相对差={rel * 100:.1f}%；振幅={res.amplitude:.3f}，斑点={res.spot_count}",
        tol="相对差<12%，amplitude>0.2，spots>=40",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured
