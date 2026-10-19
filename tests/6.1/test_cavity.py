r"""6.1 章：顶盖驱动方腔的求解器与基准证据。

四类断言：

* **A 解析/代数**：DST 直接 Poisson 的残差到舍入量级；Helmholtz(a=0) 与它逐位一致；
  速度场由流函数差分得到，散度到舍入量级；线性插值在区间外是常数延拓。
* **A 基准**：Ghia Re=100 的中心线 RMSE 随 n 下降，实测阶约 1；
  Re=1000 粗网格上的 RMSE 明显大于抄写分歧地板——这不是"代码坏了"。
* **B 判据 vs 恒等式**："残差 1e-7"与"基准 RMSE 0.2"可以同时成立；
  残差小只说明离散方程解完了，不说明离散误差小。
* **B 稳定性边界**：dt 的扩散限制与 CFL 限制在 Re_x=4 c_cfl(n-1) 交叉；
  中心格式的格点雷诺数 Re_h>2 时失去对角占优。两个阈值都自己算，不查表。
"""

from __future__ import annotations

import functools
import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcore.apps import cavity  # noqa: E402
from numcore.verify import assertion, record  # noqa: E402


@functools.lru_cache(maxsize=8)
def _steady(n: int, re: float) -> cavity.SteadyResult:
    return cavity.steady_solve(int(n), float(re), scheme="upwind", max_sweeps=12000, tol=1e-7, record_every=25)


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    mx, my = float(lx.mean()), float(ly.mean())
    return float(np.sum((lx - mx) * (ly - my)) / np.sum((lx - mx) ** 2))


@assertion("A", theory="DST 直接 Poisson 残差到舍入量级；Helmholtz 对制造解 (I-aΔ)ψ=f 恢复 ψ 到舍入量级", chapter="6.1")
def test_dst_poisson_residual_and_helmholtz_manufactured_solution() -> None:
    gr = cavity.cavity_grid(33)
    h = gr.h
    x, y = gr.x[1:-1], gr.y[1:-1]
    omega = np.zeros((gr.n, gr.n), dtype=float)
    omega[1:-1, 1:-1] = (
        np.sin(np.pi * x)[None, :] * np.sin(np.pi * y)[:, None]
        + 0.4 * np.sin(2.0 * np.pi * x)[None, :] * np.sin(np.pi * y)[:, None]
    )
    psi = cavity.poisson_solve_dst(omega, gr)
    res = cavity.poisson_residual(psi, omega, gr)
    max_res = float(np.max(np.abs(res)))

    # Helmholtz 的算子不是 Poisson：给定 f=(I-aΔ)ψ，解出来必须是 ψ。
    psi_exact = np.zeros((gr.n, gr.n), dtype=float)
    psi_exact[1:-1, 1:-1] = np.sin(np.pi * x)[None, :] * np.sin(np.pi * y)[:, None]
    lap = np.zeros((gr.n, gr.n), dtype=float)
    lap[1:-1, 1:-1] = (
        psi_exact[1:-1, 2:] + psi_exact[1:-1, :-2] + psi_exact[2:, 1:-1] + psi_exact[:-2, 1:-1]
        - 4.0 * psi_exact[1:-1, 1:-1]
    ) / h**2
    a = 0.25
    f = psi_exact - a * lap
    psi_h = cavity.helmholtz_solve_dst(f, a, gr)
    h_err = float(np.max(np.abs(psi_h - psi_exact)))
    rec = record(
        label="DST Poisson 残差与 Helmholtz 制造解",
        theory="max|Δ_hψ+ω| <= 1e-10；max|ψ_H-ψ_exact| <= 1e-10",
        measured=f"max|res|={max_res:.3e}，max|ψ_H-ψ_exact|={h_err:.3e}",
        tol="1e-10 / 1e-10",
        passed=bool(max_res <= 1e-10 and h_err <= 1e-10),
    )
    assert rec.passed, rec.measured


@assertion("A", theory="速度场 (u,v)=(ψ_y,-ψ_x) 的离散散度到舍入量级；线性插值在两端常数延拓", chapter="6.1")
def test_velocity_is_divergence_free_and_interp_is_conservative() -> None:
    res = _steady(33, 1000)
    gr = cavity.cavity_grid(33)
    div = cavity.divergence(res.u, res.v, gr)
    max_div = float(np.max(np.abs(div)))
    xp = np.array([0.0, 0.25, 0.75, 1.0])
    fp = np.array([1.0, 3.0, 5.0, 7.0])
    q = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
    got = np.asarray(cavity.linear_interp(xp, fp, q))
    want = np.array([1.0, 1.0, 4.0, 7.0, 7.0])
    interp_ok = bool(np.allclose(got, want, atol=1e-15))
    rec = record(
        label="由流函数得到的速度场散度；区间外插值",
        theory="离散散度 <= 1e-12（五点/中心差分同源）；区间外取端点常数",
        measured=f"max|div|={max_div:.3e}；插值 {got.tolist()} vs {want.tolist()}",
        tol="div<=1e-12 且插值逐点相等",
        passed=bool(max_div <= 1e-12 and interp_ok),
    )
    assert rec.passed, rec.measured


@assertion("B", theory="残差 1e-7 与基准 RMSE 0.2 可以同时成立：残差量离散方程，RMSE 量离散误差", chapter="6.1")
def test_small_residual_does_not_imply_benchmark_accuracy() -> None:
    coarse = _steady(33, 1000)
    fine = _steady(49, 100)
    gr_c = cavity.cavity_grid(33)
    gr_f = cavity.cavity_grid(49)
    cmp_c = cavity.compare_with_benchmark(coarse.u, coarse.v, gr_c, 1000)
    cmp_f = cavity.compare_with_benchmark(fine.u, fine.v, gr_f, 100)
    ok = (
        coarse.converged
        and fine.converged
        and coarse.residual < 1e-6
        and fine.residual < 1e-6
        and cmp_c.rmse > 0.2
        and cmp_f.rmse < 0.01
    )
    rec = record(
        label="Re=1000/n=33 与 Re=100/n=49 的残差 vs Ghia RMSE",
        theory="两组残差都 <1e-6；Re=1000 粗网格 RMSE>0.2，而 Re=100 细网格 RMSE<1e-2",
        measured=(
            f"Re=1000 n=33：res={coarse.residual:.2e}，RMSE={cmp_c.rmse:.4f}；"
            f"Re=100 n=49：res={fine.residual:.2e}，RMSE={cmp_f.rmse:.4f}"
        ),
        tol="残差<1e-6；RMSE 0.2 / 0.01",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


@assertion("A", theory="Ghia Re=100 的中心线 RMSE 随 n=17/25/33/49 单调下降，双对数斜率约 1", chapter="6.1")
def test_ghia_grid_convergence_re100() -> None:
    ns = np.array([17.0, 25.0, 33.0, 49.0])
    rmse = []
    detail = []
    for n in ns.astype(int):
        res = _steady(int(n), 100)
        gr = cavity.cavity_grid(int(n))
        cmp = cavity.compare_with_benchmark(res.u, res.v, gr, 100)
        rmse.append(cmp.rmse)
        detail.append(f"n={n}: {cmp.rmse:.5f}")
    rmse_a = np.asarray(rmse)
    decreasing = bool(np.all(np.diff(rmse_a) < 0.0))
    slope = _slope(1.0 / (ns - 1.0), rmse_a)
    ok = decreasing and 0.5 < slope < 2.5
    rec = record(
        label="Re=100 的 Ghia RMSE 网格收敛",
        theory="RMSE 单调下降；log(RMSE) vs log(h) 的斜率在 (0.5,2.5)（一阶迎风约 1）",
        measured="；".join(detail) + f"；实测阶 {slope:.2f}",
        tol="阶 ∈ (0.5, 2.5)",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


@assertion("B", theory="dt 的扩散/CFL 限制在 Re_x=4 c_cfl(n-1) 交叉；低于它由扩散限制，高于它由 CFL 限制", chapter="6.1")
def test_dt_limits_cross_over() -> None:
    n = 33
    low = cavity.dt_limits(n, 50.0, cfl=0.8)
    high = cavity.dt_limits(n, 400.0, cfl=0.8)
    expected_cross = 4.0 * 0.8 * (n - 1) / 1.0
    ok = (
        low["dt_diffusion"] < low["dt_cfl"]
        and high["dt_diffusion"] > high["dt_cfl"]
        and abs(low["re_cross"] - expected_cross) <= 1e-12
        and low["binding"] == 0.0
        and high["binding"] == 1.0
    )
    rec = record(
        label="Re=50/400 的 dt 限制归属与交叉点",
        theory=f"Re_cross={expected_cross:g}；Re=50 属扩散侧、Re=400 属 CFL 侧",
        measured=(
            f"Re=50: dt_diff={low['dt_diffusion']:.5f}, dt_cfl={low['dt_cfl']:.5f}, binding={low['binding']:g}；"
            f"Re=400: dt_diff={high['dt_diffusion']:.5f}, dt_cfl={high['dt_cfl']:.5f}, binding={high['binding']:g}"
        ),
        tol="严格不等式、交叉点相等",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


@assertion("B", theory="中心格式的格点雷诺数 Re_h=Re·|u|·h>2 时失去对角占优；阈值 Re=2(n-1)/u_max", chapter="6.1")
def test_central_scheme_threshold_and_cell_reynolds() -> None:
    detail = []
    ok = True
    for n in (33, 41, 49):
        threshold = cavity.central_scheme_threshold_re(n)
        cell = cavity.cell_reynolds(1000.0, n)
        expected = 2.0 * (n - 1)
        own = cavity.cell_reynolds(threshold, n)
        ok = ok and abs(threshold - expected) <= 1e-12 and cell > 2.0 and abs(own / 2.0 - 1.0) < 0.02
        detail.append(f"n={n}: Re_th={threshold:g}, Re_h(Re=1000)={cell:.2f}, Re_h(Re_th)={own:.3f}")
    rec = record(
        label="中心格式阈值与格点雷诺数",
        theory="Re_th=2(n-1)；Re=1000 在 n=33/41/49 上都已超过阈值；阈值处 Re_h≈2",
        measured="；".join(detail),
        tol="Re_th 相等、cell_re>2、阈值处 cell_re≈2",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured
