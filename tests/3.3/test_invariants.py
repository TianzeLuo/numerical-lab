r"""3.3 章：不变量工具箱（``numcore.ode.invariants``）的 A/B 类断言。

这一半测的是**度量本身**——"有界"、"闭合"、"漂移率"这些词在帧里各自对应一个函数，
函数错了，后面所有结论都错。所以每条断言都拿**造出来的已知答案**对拍：

* 人造序列：线性漂移的斜率、带宽、前后半段比值都能手算；
* 人造幂律 :math:`y=3h^2`：拟合必须给出指数 2、常数 3；
* 人造圆轨道：相位误差必须恰好是 :math:`\omega T/2\pi` 圈；
* 人造螺旋：闭合度必须报"在长大"，人造圆必须报"舍入级"。

还有一条**反例断言**：``return_distances``（天真的"每圈最近回归距离"）在采样受限时
会把闭合的轨道报成"散开了"。它留在内核里就是为了让这条断言把假象钉死——
读者下次想用这个量时必须先看到它错在哪。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from numcore.ode import invariants as iv
from numcore.ode import symplectic as sp
from numcore.utils import rng
from numcore.verify import assert_leq, assertion, record

U = float(np.finfo(float).eps) / 2.0


# --------------------------------------------------------------------------- #
# A 类：人造已知答案
# --------------------------------------------------------------------------- #
@assertion("A", "band/relative_variation/drift_slope/half_split 在人造序列上给出闭式答案", chapter="3.3")
def test_band_and_drift_on_synthetic_series() -> None:
    t = np.linspace(0.0, 10.0, 101)
    lin = 3.0 * t - 2.0
    b = iv.band(lin)
    slope = iv.drift_slope(t, lin)
    fh, sh = iv.half_split(t, lin)
    ok = (
        abs(b.lo - (-2.0)) < 1e-12
        and abs(b.hi - 28.0) < 1e-12
        and abs(b.amplitude - 30.0) < 1e-12
        and abs(b.sup_from_start - 30.0) < 1e-12
        and abs(b.midpoint - 13.0) < 1e-12
        and abs(slope - 3.0) < 1e-12
        # half_split 给的是**两半各自的斜率**（不是分段值）：线性序列两半斜率都必须是 3
        and abs(fh - 3.0) < 1e-12
        and abs(sh - 3.0) < 1e-12
    )
    assert ok, f"band={b} slope={slope} halves={(fh, sh)}"
    # 相对变化与"趋势/带宽"：
    osc = np.sin(t)
    trend, ratio = iv.drift_over_band(t, osc)
    # v₀ = sin(0) = 0：相对量没有定义，按 docstring 的约定退化为**绝对带宽**（max−min = 2）
    rel = iv.relative_variation(osc)
    assert abs(rel - 2.0) < 1e-3, f"v0=0 时应退化为绝对带宽 2.0（采样不到极值，容差 1e-3），实测 {rel}"
    shifted = osc + 3.0
    assert abs(iv.relative_variation(shifted) - 2.0 / 3.0) < 1e-3, "v0≠0 时才是相对带宽"
    assert ratio < 0.5, f"振荡序列的趋势/带宽应远小于 1，实测 {ratio:.4f}"
    # "趋势"是**窗口**的性质，不是信号的：
    #   * 对称窗口（整数个周期、关于振荡中心对称）⟹ 最小二乘斜率到舍入级；
    #   * 恰好两个整周期但窗口是 [0,4π] ⟹ 趋势/带宽 = 0.47（后半个周期权重更大，
    #     这不是漂移，是窗口不对称）；
    #   * 非整周期窗口 [0,10] ⟹ 0.17。
    #   * 窗口里 N 个周期时这个假趋势 ≈ 3/(πN)（实测 N=2,4,8,16 与公式差 < 5e-3）。
    # 所以"有界 vs 漂移"的主判据用 band_growth（后半段/前半段的 sup 之比：1 vs 2），
    # 它对窗口长度与采样混叠都免疫。
    # 3/(πN) 律：N = 窗口里的周期数
    rows = []
    for n_per in (2, 4, 8, 16):
        tt = np.linspace(0.0, 2.0 * math.pi * n_per, 200 * n_per + 1)
        r_n = iv.drift_over_band(tt, np.sin(tt))[1]
        rows.append((n_per, r_n, 3.0 / (math.pi * n_per)))
        assert abs(r_n - 3.0 / (math.pi * n_per)) < 5e-3, f"N={n_per} 应满足 3/(πN)，实测 {r_n:.4f}"
        assert iv.band_growth(np.sin(tt)) < 1.001, "同一窗口上 band_growth 必须是 1（有界）"
    assert ratio > 0.01, f"非整周期窗口的趋势/带宽必须非零（窗口效应），实测 {ratio:.3e}"
    record(
        label="度量函数的闭式对拍",
        theory="线性序列：lo/hi/amplitude/sup/slope 与两半斜率全部可手算；纯振荡的趋势/带宽 = 3/(πN)",
        measured=f"band=[{b.lo:.3f},{b.hi:.3f}] amp={b.amplitude:.3f} slope={slope:.12f} "
        f"两半斜率=({fh:.12f},{sh:.12f})；正弦趋势/带宽的 3/(πN) 律："
        + "、".join(f"N={n}:{r:.4f}(理论 {t:.4f})" for n, r, t in rows),
        tol="1e-12",
        passed=True,
        kind="A",
        extra={"slope": slope, "ratio": ratio},
    )


@assertion("A", "power_law 在 y = 3h² 上给出指数 2、常数 3；skip_tiny 能剔掉舍入地板", chapter="3.3")
def test_power_law_recovers_synthetic_law() -> None:
    h = np.array([0.4, 0.2, 0.1, 0.05, 0.025])
    y = 3.0 * h**2
    fit = iv.power_law(h, y)
    assert abs(fit.exponent - 2.0) < 1e-12 and abs(fit.constant - 3.0) < 1e-12
    assert fit.log_residual < 1e-14, f"完美幂律的对数残差应为 0，实测 {fit.log_residual:.3e}"
    # 舍入地板：最后两点被压到 1e-17，双对数图上尾巴被压平，
    # **不剔除会把指数拉到 16.4**（实测），剔掉才是真指数 2。
    y2 = np.array([3.0 * 0.4**2, 3.0 * 0.2**2, 3.0 * 0.1**2, 1e-17, 1e-17])
    raw = iv.power_law(h, y2)
    clipped = iv.power_law(h, y2, skip_tiny=1e-14)
    assert raw.exponent > 5.0, f"地板必须显著污染斜率，实测 {raw.exponent:.4f}"
    assert abs(clipped.exponent - 2.0) < 1e-12, f"剔地板后应恢复 2，实测 {clipped.exponent:.6f}"
    record(
        label="power_law 的回收能力",
        theory="y = C·h^p ⟹ 双对数斜率 = p、截距 = C",
        measured=f"拟合 (p,C)=({fit.exponent:.12f},{fit.constant:.12f})，对数残差 {fit.log_residual:.2e}；"
        f"舍入地板案例：不剔除 p={raw.exponent:.4f}、剔除后 p={clipped.exponent:.12f}",
        tol="1e-12（无噪声）；剔除后必须恢复 2",
        passed=True,
        kind="A",
        extra={"exponent": fit.exponent, "constant": fit.constant},
    )


@assertion("A", "phase_error_turns 在精确圆轨道上给出 ωT/2π 圈（相位口径不能含糊）", chapter="3.3")
def test_phase_error_on_exact_circle() -> None:
    t = np.linspace(0.0, 20.0, 4001)
    q = np.cos(t)
    p = -np.sin(t)  # 与 H=(q²+p²)/2 相配的顺时针流：θ = −t，与默认 ω=1 同号
    turns = iv.phase_error_turns(q, p, t)
    assert abs(turns) < 1e-12, f"精确解与预测同相 ⟹ 相位误差必须是 0，实测 {turns}"
    rev = iv.phase_error_turns(q, -p, t)  # 反过来跑：θ = +t，误差 = 2t/2π
    assert abs(rev - 20.0 / math.pi) < 1e-12, f"反向流应给出 +20/π 圈，实测 {rev}"
    # 值域约定是 (−π, π]：±3π 都折到 +π；3.5π 折到 −0.5π
    assert abs(iv.wrapped_angle(3.0 * math.pi) - math.pi) < 1e-12
    assert abs(iv.wrapped_angle(-3.0 * math.pi) - math.pi) < 1e-12
    assert abs(iv.wrapped_angle(3.5 * math.pi) + 0.5 * math.pi) < 1e-12
    record(
        label="相位误差与角度折叠",
        theory="φ = atan2(p,q)；圈数 = [φ(t)−φ(0)]/2π（含缠绕）",
        measured=f"精确同相 → {turns:.12f} 圈；反向流 → {rev:.12f} 圈（理论 2T/2π = {20.0 / math.pi:.12f}）；"
        f"wrapped_angle(±3π) = +π（值域 (−π,π]）",
        tol="1e-12",
        passed=True,
        kind="A",
    )


@assertion("A", "quadratic_invariant 在剪切/旋转/缩放三类人造映射上给出正确的 Q", chapter="3.3")
def test_quadratic_invariant_on_synthetic_maps() -> None:
    # 1) 辛剪切 [[1,s],[0,1]] 保持 pq 型二次型？剪切是辛的（det=1）⟹ 必有二次不变量
    shear = np.array([[1.0, 0.3], [0.0, 1.0]])
    qi_s = iv.quadratic_invariant(shear)
    # 2) 旋转（正交、det=1）：不变量是标准圆 q²+p²
    th = 0.7
    rot = np.array([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])
    qi_r = iv.quadratic_invariant(rot)
    # 3) 缩放 det≠1：必须报非辛
    sc = np.array([[1.1, 0.0], [0.0, 1.1]])
    qi_c = iv.quadratic_invariant(sc)
    assert qi_s.residual < 1e-14 and qi_r.residual < 1e-14 and not qi_c.symplectic
    # 旋转的不变量应是单位阵（差一个常数因子）
    Qr = qi_r.Q / qi_r.Q[0, 0]
    assert np.allclose(Qr, np.eye(2), atol=1e-12), f"旋转的不变量应为 I，实测 {Qr}"
    # 用真实轨道验证：辛 Euler 自己的 Q 沿轨道守恒到舍入级
    r = sp.integrate("harmonic", "symplectic_euler", 0.05, 20000, n_sample=500)
    qe = iv.quadratic_invariant(sp.linear_map("symplectic_euler", 0.05))
    hh_e = iv.invariant_series(qe.Q, r.q[:, 0], r.p[:, 0])
    rel_e = float(np.max(np.abs(hh_e - hh_e[0]))) / abs(float(hh_e[0]))
    assert rel_e < 1e-13, f"辛 Euler 自己的 Q 应守恒到舍入级，实测 {rel_e:.3e}"
    # 剪切映射的 Q 有一行是 0（出发点落在它的零方向上）⟹ curve_closure 必须当场报错
    with pytest.raises(ValueError):
        iv.curve_closure(qi_s.Q, np.array([1.0, 1.0]), np.array([0.0, 0.0]))
    record(
        label="二次不变量的三类映射对拍",
        theory="det M = 1 ⟺ Q = [[c,(d−a)/2],[(d−a)/2,−b]] 满足 MᵀQM = Q；det≠1 时不存在",
        measured=f"剪切残差 {qi_s.residual:.2e}、旋转残差 {qi_r.residual:.2e}（Q∝−I）、"
        f"缩放 symplectic={qi_c.symplectic}；辛 Euler 的 Q 沿 2e4 步轨道相对变化 {rel_e:.2e}",
        tol="1e-13",
        passed=True,
        kind="A",
        extra={"residual_rot": qi_r.residual, "rel_own": rel_e},
    )


@assertion("A", "level_curve_radius 给出正确的椭圆半径（含 h→0 退化为圆）", chapter="3.3")
def test_level_curve_radius_closed_form() -> None:
    # Q = diag(1/a², 1/b²)、value = 1/2 ⟹ 半径 r(θ) 满足 0.5·(x²/a² + y²/b²) = 0.5
    a, b = 2.0, 0.5
    Q = np.diag([1.0 / a**2, 1.0 / b**2])
    th = np.linspace(0.0, 2.0 * math.pi, 361)
    r = iv.level_curve_radius(Q, th, 0.5)
    x, y = r * np.cos(th), r * np.sin(th)
    resid = np.max(np.abs(0.5 * (x**2 / a**2 + y**2 / b**2) - 0.5))
    assert resid < 1e-12, f"等值线残差应到舍入级，实测 {resid:.3e}"
    assert abs(r[0] - a) < 1e-12 and abs(r[90] - b) < 1e-12
    # 圆：h→0 时 Verlet 的不变量趋于 I
    h = 1e-6
    qv = iv.quadratic_invariant(sp.linear_map("verlet", h))
    rc = iv.level_curve_radius(qv.Q, th, qv.value(1.0, 0.0))
    assert np.max(np.abs(rc - 1.0)) < 1e-10, f"h→0 时应退化为单位圆，实测最大偏差 {np.max(np.abs(rc - 1.0)):.3e}"
    record(
        label="等值线半径的闭式对拍",
        theory="r(θ) = √(2·value / (Q₀₀cos²θ + 2Q₀₁cosθsinθ + Q₁₁sin²θ))",
        measured=f"椭圆 (a,b)=({a},{b}) 的最大等值线残差 {resid:.2e}；Verlet h→0 时半径偏差 {np.max(np.abs(rc - 1.0)):.2e}",
        tol="1e-12",
        passed=True,
        kind="A",
    )


# --------------------------------------------------------------------------- #
# B 类：闭合与"假证据"
# --------------------------------------------------------------------------- #
@assertion("B", "curve_closure：舍入级闭合 vs 线性螺旋，两者必须被分开报出", chapter="3.3")
def test_closure_distinguishes_roundoff_from_spiral() -> None:
    th = np.linspace(0.0, 400.0 * 2.0 * math.pi, 40001)
    # 1) 完美圆：偏差应是 0（growth 无定义 → inf，由帧层写成 null）
    q0, p0 = np.cos(th), -np.sin(th)
    Q = np.eye(2)
    c_exact = iv.curve_closure(Q, q0, p0)
    # 2) 对数螺旋：半径按 (1+1e-4)^k 长，前后半段的比值应 ≈ 2
    growth = np.exp(1e-4 * th)
    c_spiral = iv.curve_closure(Q, q0 * growth, p0 * growth)
    # 3) 舍入级抖动：加 1e-16 噪声，仍应判为 at_roundoff
    r = rng(7)
    c_noise = iv.curve_closure(Q, q0 * (1.0 + 1e-16 * r.standard_normal(th.size)), p0)
    assert c_exact.sup < 1e-15, f"完美圆的偏差应为 0，实测 {c_exact.sup:.3e}"
    assert c_spiral.sup > 1e-2, f"螺旋的偏差应显著，实测 {c_spiral.sup:.3e}"
    assert 1.5 < c_spiral.growth < 2.5, f"对数螺旋的前后半段比应 ≈2，实测 {c_spiral.growth:.4f}"
    assert c_noise.at_roundoff, f"1e-16 抖动应判为舍入级，实测 sup={c_noise.sup:.3e}"
    assert not c_spiral.at_roundoff
    record(
        label="闭合度对三种轨道的判别",
        theory="δ_k = r_k/r_curve(θ_k) − 1；舍入级 ⟹ 点集闭合；线性累积 ⟹ 螺旋（前后半段比 ≈ 时长比）",
        measured=f"完美圆 sup={c_exact.sup:.2e}；螺旋 sup={c_spiral.sup:.3e}（growth {c_spiral.growth:.3f}）；"
        f"舍入抖动 sup={c_noise.sup:.2e}（at_roundoff={c_noise.at_roundoff}）",
        tol="圆 < 1e-15；螺旋 sup > 1e-2 且 growth ∈ (1.5,2.5)",
        passed=True,
        kind="B",
        extra={"spiral_growth": c_spiral.growth, "noise_sup": c_noise.sup},
    )


@assertion("B", "return_distances 是采样受限的假证据：它由采样密度决定，不能当闭合判据", chapter="3.3")
def test_return_distances_is_a_sampling_artifact() -> None:
    # 真·闭合轨道（精确圆），用两种采样密度各量一次
    period = 2.0 * math.pi
    out = {}
    for h in (0.1, 0.01):
        n = int(round(200.0 * period / h))
        t = np.arange(n + 1) * h
        q = np.cos(t)
        p = -np.sin(t)
        out[h] = (iv.return_distances(q, p, t, period), iv.curve_closure(np.eye(2), q, p))
    coarse, fine = out[0.1], out[0.01]
    # 单圈的"最近距离"在 [0, h/2] 上随机（取决于采样点落在返回点的哪一侧），
    # 所以要看**逐圈平均**：它 ∝ 采样步长（均匀采样下约为周长的 1/4），与是否闭合无关。
    m_coarse = float(np.mean(coarse[0].returns))
    m_fine = float(np.mean(fine[0].returns))
    assert m_coarse > 5.0 * m_fine, (
        f"采样加密 10 倍后「每圈最近回归距离」的均值应随之变小（它是采样下界），实测 "
        f"{m_coarse:.3e} → {m_fine:.3e}"
    )
    assert coarse[1].sup < 1e-14 and fine[1].sup < 1e-14, "同一条轨道的闭合度必须都是舍入级"
    record(
        label="采样受限的假证据：加密采样就变小",
        theory="轨道是连续曲线、采样点离散：「最近回归距离」的下界由采样密度（h）决定，与是否闭合无关",
        measured=f"同一条精确圆：逐圈最近回归距离的均值 h=0.1 时 {m_coarse:.3e}、"
        f"h=0.01 时 {m_fine:.3e}（差 {m_coarse / m_fine:.1f} 倍 ≈ 步长比，纯粹是采样下界）；"
        f"而 curve_closure 两次都是 sup={coarse[1].sup:.2e}/{fine[1].sup:.2e}",
        tol="两种密度的均值差 ≥ 5 倍；closure 都 < 1e-14",
        passed=True,
        kind="B",
        extra={"coarse_mean": m_coarse, "fine_mean": m_fine},
    )


@assertion("B", "reference_closure 量的是「离真值多远」：辛方法 O(h^p) 有界、RK4 线性累积", chapter="3.3")
def test_reference_closure_separates_band_from_drift() -> None:
    rows = {}
    for m in ("verlet", "rk4"):
        r = sp.integrate("harmonic", m, 0.1, int(round(600 * 2.0 * math.pi / 0.1)), n_sample=4000)
        cl = iv.reference_closure(r.q[:, 0], r.p[:, 0], np.ones(r.q.shape[0]))
        rows[m] = cl
    v, k = rows["verlet"], rows["rk4"]
    assert v.sup < 5e-3 and v.growth < 1.05, f"Verlet 的偏差应是 O(h²) 有界，实测 sup={v.sup:.3e} growth={v.growth:.3f}"
    assert k.growth > 1.8, f"RK4 的偏差应线性累积，实测 growth={k.growth:.3f}"
    ratio = k.sup / v.sup
    record(
        label="到精确轨道的偏差：有界带宽 vs 线性累积",
        theory="辛方法：偏差 ≤ 不变带 ⟹ 有界；非辛：每圈穿出一点 ⟹ 线性（前后半段比 ≈ 2）",
        measured=f"Verlet sup={v.sup:.3e}（growth {v.growth:.3f}）；RK4 sup={k.sup:.3e}（growth {k.growth:.3f}）；"
        f"末端 RK4/Verlet = {ratio:.3f}（观测 600 圈）",
        tol="Verlet growth<1.05；RK4 growth>1.8",
        passed=True,
        kind="B",
        extra={"verlet_sup": v.sup, "rk4_sup": k.sup, "ratio": ratio},
    )


@assertion("B", "度量函数的边界行为：短序列/长度不符/零等值线都要当场报错，不许悄悄给个数", chapter="3.3")
def test_metric_error_paths_are_loud() -> None:
    bad = []
    with pytest.raises(ValueError):
        iv.closure_stats([1.0, 2.0, 3.0])
    bad.append("closure_stats(<4 点)")
    with pytest.raises(ValueError):
        iv.radial_deviation([1.0, 0.0], [0.0, 1.0], [1.0])
    bad.append("radial_deviation(长度不符)")
    with pytest.raises(ValueError):
        iv.return_distances([1.0, 0.0], [0.0, 1.0], [0.0, 0.1], 0.0)
    bad.append("return_distances(周期≤0)")
    with pytest.raises(ValueError):
        iv.curve_closure(np.eye(2), [0.0, 0.0], [0.0, 0.0])
    bad.append("curve_closure(出发点在 H_h=0 上)")
    with pytest.raises(ValueError):
        iv.power_law([0.1, 0.2], [1.0])
    bad.append("power_law(长度不符)")
    record(
        label="度量函数的报错路径",
        theory="量不出定义的量时必须抛错：帧层才有机会写 null/占位，而不是让 0 或 NaN 混进结论",
        measured=f"{len(bad)} 条边界全部抛 ValueError：" + "、".join(bad),
        tol="全部抛错",
        passed=True,
        kind="B",
        extra={"n_checked": len(bad)},
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
