r"""3.3 章：辛积分器与不变量的 A/B/C 类断言。

这一半只谈**内核**：方法表、带宽闭式、映射行列式、爆点口径、轨道要素。
所有"精确"参照物都是闭式表达式或解析流，不是另一份同样的代码：

* 谐振子的带宽有闭式——辛 Euler 是 :math:`[-h/(2+h),\,h/(2-h)]`、Verlet 是 :math:`[-h^2/4,0]`
  （:math:`\det M=1` 的直接推论，notes 第 2 节给证明）；
* 非辛对照的漂移率也有闭式——RK4 的 :math:`\det M=1-h^6/72` ⟹ 每步能量掉 :math:`h^6/72`、
  每单位时间掉 :math:`h^5/72`；
* Kepler 有精确流（自写 Newton 解 Kepler 方程），单摆的精确周期是 AGM 椭圆积分。

"长期"两个字在这里是**真的跑**：10^6 步的带宽断言用 h=0.05 走满，实测 0.16 秒
（Verlet 快路径），不是外推、不是"跑 10^4 步再乘 100"。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from numcore.ode import invariants as iv
from numcore.ode import symplectic as sp
from numcore.verify import (
    assert_bounded_drift,
    assert_conserved,
    assert_leq,
    assert_linear_drift,
    assert_order,
    assertion,
    find_threshold,
    record,
)

U = float(np.finfo(float).eps) / 2.0

#: 10^6 步的那条横幅断言用的步长。
H_LONG = 0.05

#: 步长阶梯（粗→细），双对数图上等距。
HS = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625])

#: 昂贵积分只算一次（模块级缓存，和 server 章节的 lru_cache 一个思路）。
_CACHE: dict[tuple, sp.Run] = {}


def run(system: str, method: str, h: float, n_steps: int, n_sample: int = 2000) -> sp.Run:
    key = (system, method, float(h), int(n_steps), int(n_sample))
    if key not in _CACHE:
        _CACHE[key] = sp.integrate(system, method, float(h), int(n_steps), n_sample=int(n_sample))
    return _CACHE[key]


# --------------------------------------------------------------------------- #
# A 类：方法表与闭式带宽
# --------------------------------------------------------------------------- #
@assertion("A", "方法表的阶与辛性标注与理论一致；四个辛方法的映射行列式恰为 1", chapter="3.3")
def test_method_table_and_det_map() -> None:
    expect = {
        "symplectic_euler": (1, True, False, False),
        "verlet": (2, True, True, False),
        "leapfrog": (2, True, True, False),
        "implicit_midpoint": (2, True, True, True),
        # RK4 **不是**对称方法：判别式是 R(z)R(−z) = 1 + O(z⁶) ≠ 1（经典 RK4 的偶部/奇部
        # 在 z⁴ 处不抵消），所以它的修正级数里有奇次幂、det M 在 h⁶ 处才偏离 1。
        "rk4": (4, False, False, False),
        "explicit_euler": (1, False, False, False),
    }
    rows = []
    worst = 0.0
    for name, (order, sym, symm, implicit) in expect.items():
        s = sp.method_spec(name)
        ok = (s.order == order) and (s.symplectic == sym) and (s.symmetric == symm) and (s.implicit == implicit)
        assert ok, f"{name}: 表里是 {(s.order, s.symplectic, s.symmetric, s.implicit)}"
        M = sp.linear_map(name, 0.1)
        det = sp.det_2x2(M)
        qi = iv.quadratic_invariant(M)
        if sym:
            worst = max(worst, abs(1.0 - det), qi.residual)
        rows.append(f"{name}: order={order} det={det:.12f} resid={qi.residual:.2e}")
    assert worst < 1e-14, f"辛方法的 |1-det| 与二次型残差应到舍入级，实测 {worst:.3e}"
    record(
        label="方法表 + 四个辛方法的 det M = 1",
        theory="det M ≡ 1（精确到位）且 MᵀQM−Q 在舍入级",
        measured=f"最大偏离 {worst:.3e}；" + "；".join(rows),
        tol="1e-14",
        passed=True,
        kind="A",
        extra={"worst_deviation": worst},
    )


@assertion("A", "谐振子带宽闭式：辛 Euler = [−h/(2+h), h/(2−h)]、Verlet = ±h²/4（10^6 步实测）", chapter="3.3")
def test_harmonic_band_closed_forms_over_a_million_steps() -> None:
    n = 10**6
    out = []
    worst = 0.0
    # 带的两端都是**闭式**，而且**不对称**：Verlet 的能量误差只在负侧（上界恰是 0），
    # 因为它的二次不变量把轨道压在同一条椭圆上、H−H₀ = (h²/8)(q²−q₀²) ≤ 0。
    # 只写"±h²/4"会让上界差 100%（实测第一版就这么错过）。
    for name, hi_theory, lo_theory in (
        ("symplectic_euler", H_LONG / (2.0 - H_LONG), -H_LONG / (2.0 + H_LONG)),
        ("verlet", 0.0, -H_LONG * H_LONG / 4.0),
        ("leapfrog", 0.0, -H_LONG * H_LONG / 4.0),
    ):
        r = run("harmonic", name, H_LONG, n)
        assert r.ok and r.n_steps_done == n, f"{name} 没跑满 10^6 步：done={r.n_steps_done}"
        b = iv.band(r.energy_error)
        dev = max(abs(b.hi - hi_theory), abs(b.lo - lo_theory))
        worst = max(worst, dev)
        out.append(f"{name}: 实测 [{b.lo:+.9e},{b.hi:+.9e}] 闭式 [{lo_theory:+.9e},{hi_theory:+.9e}] 偏差 {dev:.2e}")
    # 容差为什么是 1e-8 而不是 1e-12：带端点是**采样极值**——2001 个采样点跨 10^6 步
    # （stride 500 步 = 4 个周期），采不到极值那一刻，只能落到附近；实测量级是 1e-9…1e-8，
    # 相对 2.4e-2 是 3e-7。要更紧就得把 n_sample 加到 10^5（帧里不划算），
    # 而"带 = 闭式"这个结论在 3e-7 相对精度上已经无可争议。
    assert worst < 1e-8, f"带宽端点与闭式的绝对偏差应 < 1e-8，实测 {worst:.3e}"
    record(
        label="10^6 步的带宽 = 闭式",
        theory="辛 Euler [−h/(2+h), h/(2−h)]；Verlet/leapfrog [−h²/4, 0]（h=0.05，E₀=1）",
        measured=f"最大绝对偏差 {worst:.3e}；" + "；".join(out),
        tol="1e-8（绝对；采样极值限制）",
        passed=True,
        kind="A",
        extra={"worst_rel": worst},
    )


@assertion("A", "RK4 的映射行列式 = 1 − h⁶/72 + O(h⁸)（非辛 ⟹ 没有二次不变量）", chapter="3.3")
def test_rk4_det_map_matches_series() -> None:
    rows = []
    worst = 0.0
    for h in (0.4, 0.2, 0.1, 0.05):
        det = sp.det_2x2(sp.linear_map("rk4", h))
        theory = 1.0 - h**6 / 72.0 + h**8 / 576.0
        rel = abs((1.0 - det) - (1.0 - theory)) / (1.0 - theory)
        worst = max(worst, rel)
        rows.append(f"h={h}: 1−det={1.0 - det:.6e} vs {1.0 - theory:.6e}（相对差 {rel:.2e}）")
    e = sp.det_2x2(sp.linear_map("explicit_euler", 0.1))
    assert worst < 1e-3, f"RK4 的 1−det 与 h⁶/72 系列相对差应 < 1e-3，实测 {worst:.3e}"
    assert abs(e - 1.01) < 1e-15, f"显式 Euler 的 det 应为 1+h²=1.01，实测 {e}"
    record(
        label="RK4 / 显式 Euler 的 det M",
        theory="RK4：det = 1 − h⁶/72 + h⁸/576；显式 Euler：det = 1 + h²",
        measured=f"RK4 最大相对差 {worst:.3e}；" + "；".join(rows) + f"；显式 Euler det(0.1)={e:.12f}",
        tol="1e-3（截断）/ 1e-15（显式 Euler）",
        passed=True,
        kind="A",
        extra={"worst_rel": worst, "euler_det": e},
    )


@assertion("A", "单摆精确周期（AGM 椭圆积分）与数值实测过零点一致到 1e-8（四个振幅）", chapter="3.3")
def test_pendulum_period_agm_matches_measured_crossing() -> None:
    rows = []
    worst = 0.0
    for amp in (0.01, 0.5, 1.0, 2.0):
        s = sp.pendulum_system(amp)
        t_theory = float(s.period)
        r = run(s, "verlet", 2.0e-4, int(round(4.0 * t_theory / 2.0e-4)), n_sample=200000)
        q = r.q[:, 0]
        k = int(np.argmax(q[1:] < 0.0)) + 1  # 第一次向下过零 = T/4
        t_cross = float(r.t[k - 1]) + float(r.h) * (0.0 - float(q[k - 1])) / float(q[k] - q[k - 1])
        t_meas = 4.0 * t_cross
        rel = abs(t_meas - t_theory) / t_theory
        worst = max(worst, rel)
        rows.append(f"θ0={amp}: AGM {t_theory:.9f} 实测 {t_meas:.9f}（相对差 {rel:.2e}）")
    assert worst < 1e-8, f"周期相对差应 < 1e-8，实测 {worst:.3e}"
    assert sp.pendulum_period(0.01) < sp.pendulum_period(2.0) < sp.pendulum_period(3.0)
    record(
        label="单摆精确周期 vs 数值过零点",
        theory="T(θ0) = 4/√(1+H) · K(k)（AGM 算的完全椭圆积分）",
        measured=f"最大相对差 {worst:.3e}；" + "；".join(rows),
        tol="1e-8",
        passed=True,
        kind="A",
        extra={"worst_rel": worst},
    )


@assertion("A", "Kepler 精确流：轨道要素 (a,e,ϖ) 严格守恒；数值解在辛方法下也守恒到带宽内", chapter="3.3")
def test_kepler_exact_flow_conserves_orbital_elements() -> None:
    t = np.linspace(0.0, 20.0 * 2.0 * math.pi, 4000)
    q, p = sp.kepler_flow(t, 0.6)
    a, e, w = sp.orbital_elements(q, p)
    worst = max(
        float(np.max(np.abs(a / a[0] - 1.0))),
        float(np.max(np.abs(e / e[0] - 1.0))),
        float(np.max(np.abs(w - w[0]))),
    )
    assert worst < 1e-10, f"精确流的轨道要素应守恒到 1e-10，实测 {worst:.3e}（自写 Newton 解 Kepler 方程的精度）"
    # 数值侧：Verlet 的形状误差落在能量带宽内，RK4 的角动量一直在漂
    rv = run("kepler_ecc_0.6", "verlet", 0.01, int(round(200 * 2 * math.pi / 0.01)), n_sample=4000)
    a2, e2, _ = sp.orbital_elements(rv.q, rv.p)
    shape_v = float(np.max(np.abs((a2 * (1.0 - e2)) / 0.4 - 1.0)))
    band_v = rv.energy_band()
    assert shape_v <= 20.0 * band_v, f"Verlet 的形状失真 {shape_v:.3e} 应落在能量带宽 {band_v:.3e} 的量级内"
    record(
        label="精确流的轨道要素守恒 + 辛方法的形状误差 ≤ 带宽",
        theory="a = −μ/2E、e = |A|/μ、A 为 Laplace–Runge–Lenz 矢量；辛方法能量有界 ⟹ 形状有界",
        measured=f"精确流最大偏离 {worst:.3e}；Verlet（200 圈，h=0.01）形状失真 {shape_v:.3e} 对能量带宽 {band_v:.3e}",
        tol="精确流 1e-10；Verlet 形状 ≤ 20×带宽",
        passed=True,
        kind="A",
        extra={"exact_worst": worst, "verlet_shape": shape_v, "verlet_band": band_v},
    )


# --------------------------------------------------------------------------- #
# B 类：有界 vs 漂移、守恒量、口径
# --------------------------------------------------------------------------- #
@assertion("B", "10^6 步：辛方法的能量误差有界（振荡），RK4 单调漂移（同一帧里的对照）", chapter="3.3")
def test_bounded_versus_drifting_energy_over_a_million_steps() -> None:
    n = 10**6
    rv = run("harmonic", "verlet", H_LONG, n)
    rk = run("harmonic", "rk4", H_LONG, n)
    amp_v = assert_bounded_drift(
        rv.energy_error,
        tol=H_LONG * H_LONG / 4.0 * (1.0 + 1e-6),
        label="Verlet 的能量误差有界（10^6 步）",
        detail=f"h={H_LONG}，闭式带宽 h²/4={H_LONG * H_LONG / 4:.3e}，实测 sup {rv.energy_band():.9e}",
    )
    amp_k = assert_linear_drift(
        rk.energy_error,
        label="RK4 的能量误差线性漂移（10^6 步）",
        detail=f"h={H_LONG}，末端误差 {rk.final_energy_error():.6e}，与首端之比 {rk.final_energy_error() / rk.energy_error[len(rk.energy_error) // 2]:.3f}",
    )
    mono = iv.monotone_fraction(rk.energy_error)
    assert_leq(
        1.0 - mono,
        0.0,
        atol=1e-9,
        label="RK4 的漂移是单向的（单调占比 = 1）",
        detail=f"单调占比 {mono:.6f}",
    )
    # 主判据：后半段/前半段的 sup 之比。有界振荡 = 1、线性漂移 = 2（观测时长翻倍）。
    gv = iv.band_growth(rv.energy_error)
    gk = iv.band_growth(rk.energy_error)
    assert abs(gv - 1.0) < 0.01, f"Verlet 的两半 sup 之比应为 1（有界），实测 {gv:.4f}"
    assert abs(gk - 2.0) < 0.05, f"RK4 的两半 sup 之比应为 2（线性漂移），实测 {gk:.4f}"
    # 辅证：趋势/带宽（纯振荡的窗口假趋势 floor ≈ 3/(πN)，这里 N≈7958 周期）
    ratio = iv.drift_over_band(rv.t, rv.energy_error)[1]
    kratio = iv.drift_over_band(rk.t, rk.energy_error)[1]
    assert ratio < 0.01, f"Verlet 的总趋势/带宽应远小于 1（有界），实测 {ratio:.4f}"
    assert kratio > 0.9, f"RK4 的总趋势/带宽应接近 1（纯漂移），实测 {kratio:.4f}"
    record(
        label="有界 vs 漂移的定量判据（两半 sup 之比 + 趋势/带宽）",
        theory="辛：后半/前半 sup 之比 = 1（同一带宽）；非辛：= 2（线性长）",
        measured=f"sup 之比：Verlet {gv:.4f}、RK4 {gk:.4f}；趋势/带宽：Verlet {ratio:.6f}、RK4 {kratio:.6f}"
        f"（振幅 {amp_v:.3e} / 斜率 {amp_k:.3e}，单调占比 {mono:.4f}）",
        tol="Verlet 之比 ∈ 1±0.01 且趋势/带宽 < 0.01；RK4 之比 ∈ 2±0.05 且趋势/带宽 > 0.9",
        passed=True,
        kind="B",
        extra={"verlet_ratio": ratio, "rk4_ratio": kratio},
    )


@assertion("B", "角动量：辛方法逐位守恒（~1e-14），RK4 线性漂移（~1e-7）；动量则所有方法都守恒", chapter="3.3")
def test_angular_momentum_is_the_discriminator_not_momentum() -> None:
    n = int(round(200 * 2 * math.pi / 0.01))
    rv = run("kepler_ecc_0.6", "verlet", 0.01, n, n_sample=4000)
    rk = run("kepler_ecc_0.6", "rk4", 0.01, n, n_sample=4000)
    d_v = iv.angular_momentum_drift(rv.angular_momentum)
    d_k = iv.angular_momentum_drift(rk.angular_momentum)
    assert_conserved(
        rv.angular_momentum,
        atol=1e-12,
        label="Verlet 的角动量守恒（中心力逐位性质）",
        detail=f"200 圈（h=0.01）相对漂移 {d_v:.3e}",
    )
    assert d_k > 100.0 * d_v, f"RK4 的角动量漂移应比 Verlet 大两个量级以上，实测 {d_k:.3e} vs {d_v:.3e}"
    # 反例：动量对**所有**方法都守恒——它不是辛性的证据
    mom = {}
    for m in ("verlet", "symplectic_euler", "implicit_midpoint", "rk4", "explicit_euler"):
        r = run("lj_chain", m, 0.005, 20000, n_sample=500)
        P = r.momentum
        mom[m] = float(np.max(np.abs(P - P[0])))
    worst = max(mom.values())
    assert worst < 1e-12, f"LJ 链的动量漂移应全部到舍入级，实测最大 {worst:.3e}"
    record(
        label="角动量 vs 动量：谁是辛性的证据",
        theory="中心力 + 径向 kick ⟹ L 逐位守恒（辛性）；动量守恒只是平移不变性，非辛方法也有",
        measured=f"Kepler 200 圈角动量漂移：Verlet {d_v:.3e}、RK4 {d_k:.3e}（差 {d_k / max(d_v, 1e-300):.1f} 倍）；"
        f"LJ 链动量漂移最大 {worst:.3e}（含显式 Euler {mom['explicit_euler']:.3e}）",
        tol="Verlet ≤ 1e-12；RK4 ≥ 100×Verlet；动量全部 < 1e-12",
        passed=True,
        kind="B",
        extra={"verlet_dL": d_v, "rk4_dL": d_k, "worst_momentum": worst},
    )


@assertion("B", "爆点口径：停止点、可信段与占位值三者一致（不允许把 inf/NaN 或假零写进帧）", chapter="3.3")
def test_blowup_semantics_are_consistent() -> None:
    r = run("harmonic", "explicit_euler", 0.05, 10**6, n_sample=1500)
    ok = (not r.ok) and r.blowup and r.i_blowup < r.idx.size
    assert ok, f"显式 Euler h=0.05 应在 10^6 步内爆炸，实测 ok={r.ok} blowup={r.blowup}"
    # 尾部占位：t 与 idx 都是停止步，且状态就是爆炸状态（不是 0、不是 NaN）
    tail = slice(r.i_blowup, None)
    assert np.all(np.isfinite(r.q)) and np.all(np.isfinite(r.p)), "帧口径不允许非有限值出现在数组里"
    assert_leq(
        float(np.max(np.abs(r.t[tail] - r.n_steps_done * r.h))),
        0.0,
        atol=1e-9,
        label="尾部采样点的 t = n_steps_done·h",
        detail=f"n_steps_done={r.n_steps_done}，t[-1]={r.t[-1]:.4f}，idx[-1]={int(r.idx[-1])}",
    )
    assert float(np.max(np.abs(r.q[tail]))) >= 1e12 * 0.999, "尾部必须是真实的爆炸状态（占位值由前端按掩码处理）"
    assert r.n_steps_done < 10**6 and r.n_steps_done > 0
    # 掩码口径：i_blowup 之前可信
    good = r.energy_error[: r.i_blowup]
    assert np.all(np.isfinite(good)) and len(good) >= 2
    record(
        label="爆炸停止点的口径一致性",
        theory="i_blowup 是第一个含爆炸状态的采样点，[:i_blowup] 可信；t 与 idx 都改成 n_steps_done",
        measured=f"h=0.05 显式 Euler 在第 {r.n_steps_done} 步停止（t={r.n_steps_done * r.h:.1f}），"
        f"i_blowup={r.i_blowup}/{r.idx.size}，爆点 |y|={float(np.max(np.abs(r.q[tail]))):.3e}，"
        f"可信段能量误差 {float(np.max(np.abs(good))):.3e}",
        tol="t 与 idx 一致；爆点 ≥ 1e12",
        passed=True,
        kind="B",
        extra={"n_steps_done": r.n_steps_done, "i_blowup": int(r.i_blowup)},
    )


@assertion("B", "快路径与通用路径逐位一致；verlet 与 leapfrog 在全部系统上逐位相同", chapter="3.3")
def test_fast_path_and_method_aliases_are_bit_identical() -> None:
    rows = []
    worst = 0.0
    for name in sp.SYSTEM_NAMES:
        for method in ("verlet", "symplectic_euler", "implicit_midpoint"):
            a = run(name, method, 0.02, 5000, n_sample=200)  # 快路径（dof<=2）
            b = _generic(name, method, 0.02, 5000)
            sel = a.idx  # 通用路径记了每一步，按采样步号对齐（否则是 (201,) vs (5001,)）
            d = max(
                float(np.max(np.abs(a.q - b[0][sel]))),
                float(np.max(np.abs(a.p - b[1][sel]))),
                float(np.max(np.abs(a.energy - b[2][sel]))),
            )
            worst = max(worst, d)
            rows.append(f"{name}/{method}: {d:.1e}")
    assert worst == 0.0, f"快路径与通用路径应逐位相同，实测最大差 {worst:.3e}（{rows}）"
    for name in sp.SYSTEM_NAMES:
        a = run(name, "verlet", 0.02, 5000, n_sample=200)
        b = run(name, "leapfrog", 0.02, 5000, n_sample=200)
        d = max(float(np.max(np.abs(a.q - b.q))), float(np.max(np.abs(a.p - b.p))))
        assert d == 0.0, f"{name}: verlet 与 leapfrog 应逐位相同，实测 {d:.3e}"
    record(
        label="快路径 vs 通用路径、verlet vs leapfrog",
        theory="两条路径算的是同一个映射（浮点求和顺序也相同）；leapfrog 就是 verlet 的别名",
        measured=f"{len(rows)} 组对拍最大差 {worst:.1e}（逐位相同）；verlet/leapfrog 在 {len(sp.SYSTEM_NAMES)} 个系统上差 0.0",
        tol="0（逐位）",
        passed=True,
        kind="B",
        extra={"worst": worst},
    )


def _generic(name: str, method: str, h: float, n_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """用通用路径（``sp.step_detailed``，numpy 向量）重跑一遍，供对拍。"""
    s = sp.system(name)
    q = s.q0_array().copy()
    p = s.p0_array().copy()
    qs = [q.copy()]
    ps = [p.copy()]
    es = [s.energy_of(q, p)]
    for _ in range(n_steps):
        info = sp.step_detailed(method, s, q, p, h)
        q, p = info.q, info.p
        qs.append(q.copy())
        ps.append(p.copy())
        es.append(s.energy_of(q, p))
    return np.array(qs), np.array(ps), np.array(es)


@assertion("B", "修正不变量 H_h 沿轨道守恒到舍入级（辛方法）——「守恒的是另一个哈密顿量」是可算的", chapter="3.3")
def test_modified_invariant_is_conserved_to_roundoff() -> None:
    rows = []
    worst = 0.0
    for m in ("symplectic_euler", "verlet", "implicit_midpoint"):
        r = run("harmonic", m, 0.05, 20000, n_sample=1000)
        qi = iv.quadratic_invariant(sp.linear_map(m, 0.05))
        hh = iv.invariant_series(qi.Q, r.q[:, 0], r.p[:, 0])
        rel = float(np.max(np.abs(hh - hh[0]))) / abs(float(hh[0]))
        worst = max(worst, rel)
        rows.append(f"{m}: H_h 相对变化 {rel:.2e}（真实能量带宽 {r.energy_band():.2e}）")
    rk = run("harmonic", "rk4", 0.05, 20000, n_sample=1000)
    qi_k = iv.quadratic_invariant(sp.linear_map("rk4", 0.05))
    hh_k = iv.invariant_series(qi_k.Q, rk.q[:, 0], rk.p[:, 0])
    rel_k = float(np.max(np.abs(hh_k - hh_k[0]))) / abs(float(hh_k[0]))
    assert worst < 1e-12, f"辛方法的 H_h 应守恒到舍入级，实测 {worst:.3e}"
    assert rel_k > 1e-9, f"RK4 没有二次不变量（det≠1），同一公式应看到漂移，实测 {rel_k:.3e}"
    record(
        label="修正不变量 H_h 的守恒性",
        theory="det M = 1 ⟺ 存在精确二次不变量 Q（H_h 的二次项）；RK4 的 det = 1−h⁶/72 ⟹ 不存在",
        measured="；".join(rows) + f"；RK4 同一公式给出 {rel_k:.2e}（= 它自己的能量漂移）",
        tol="辛方法 < 1e-12；RK4 > 1e-9",
        passed=True,
        kind="B",
        extra={"symplectic_worst": worst, "rk4_rel": rel_k},
    )


# --------------------------------------------------------------------------- #
# C 类：收敛率与阈值
# --------------------------------------------------------------------------- #
@assertion("C", "RK4 的能量漂移率 ∝ h⁵（实测幂律指数 ≈ 5），与 det M = 1−h⁶/72 的闭式一致", chapter="3.3")
def test_rk4_drift_rate_power_law() -> None:
    err = []
    for h in HS:
        n = int(round(50.0 / float(h)))
        r = run("harmonic", "rk4", float(h), n, n_sample=min(2000, n))
        err.append(abs(r.final_energy_error()))
    fit = assert_order(HS, np.array(err), theory=5.0, tol=0.05, kind="C",
                       label="RK4 能量漂移率对 h 的幂律",
                       detail="窗口 T=50 固定，误差取末端 |ΔE/E|")
    c = fit.constant
    theory_c = 50.0 / 72.0
    rel = abs(c - theory_c) / theory_c
    assert rel < 0.05, f"幂律系数应接近 T/72={theory_c:.6f}，实测 {c:.6f}（相对差 {rel:.3e}）"
    record(
        label="RK4 漂移率的系数 = T/72",
        theory="|ΔE/E| = h⁵T/72（每步掉 h⁶/72，共 n=T/h 步）",
        measured=f"拟合系数 {c:.6f} vs T/72 {theory_c:.6f}（相对差 {rel:.3e}）",
        tol="5%",
        passed=True,
        kind="C",
    )


@assertion("C", "辛方法的带宽幂律：Verlet ∝ h²（指数 ≈ 2）、辛 Euler 精确遵循 h/(2−h)（比值 1±1e-5）", chapter="3.3")
def test_symplectic_band_power_laws() -> None:
    b2 = []
    b1 = []
    for h in HS:
        n = int(round(50.0 / float(h)))
        # 采样点必须够密：n_sample=10 在 8000 步上 stride=800，会**漏掉**带的极值
        # （实测把 6.25e-4 的带宽量成 6.21e-4，比值检验直接偏 5.7e-3）。
        ns = min(2000, n)
        b2.append(run("harmonic", "verlet", float(h), n, n_sample=ns).energy_band())
        b1.append(run("harmonic", "symplectic_euler", float(h), n, n_sample=ns).energy_band())
    fit = assert_order(HS, np.array(b2), theory=2.0, tol=0.02, kind="C",
                       label="Verlet 带宽对 h 的幂律", detail="sup|ΔE/E|，T=50")
    ratios = [b / (float(h) / (2.0 - float(h))) for h, b in zip(HS, b1)]
    worst = max(abs(r - 1.0) for r in ratios)
    # 名义指数 1 是 h→0 极限；在 h∈[0.1,0.00625] 上 h/(2−h) 的局部指数是 1.0186，
    # 所以这里对 h/(2−h) 本体对账（比值），而不是硬套指数 1。
    assert worst < 1e-4, f"辛 Euler 的带宽应等于 h/(2−h)，实测最大相对差 {worst:.3e}"
    assert fit.slope > 1.9, f"Verlet 带宽指数应接近 2，实测 {fit.slope:.4f}"
    record(
        label="辛 Euler 带宽 = h/(2−h)（比值检验）",
        theory="带宽 = h/(2−h)（det M=1 的推论，见 notes 第 2 节）",
        measured=f"最大相对差 {worst:.3e}（h∈[{HS[0]},{HS[-1]}]，比值 {[f'{r:.8f}' for r in ratios]}）",
        tol="1e-4",
        passed=True,
        kind="C",
    )


@assertion("C", "稳定边界 h*=2 用二分阈值实测：Verlet/辛 Euler 在 h<2 有界、h>2 溢出", chapter="3.3")
def test_harmonic_stability_threshold_by_bisection() -> None:
    def bounded(h: float) -> bool:
        r = sp.integrate("harmonic", "verlet", float(h), 20000, n_sample=50)
        return bool(r.ok) and r.energy_band() < 10.0

    res = find_threshold(bounded, 1.0, 3.0, tol=1e-6, assume="true_at_lo",
                         note="h=2 是 |λ|=1 的分界（辛 Euler 与 Verlet 同界）")
    assert abs(res.threshold - 2.0) < 1e-3, f"稳定边界应为 2，实测 {res.threshold:.6f}"
    rk = find_threshold(lambda h: bool(sp.integrate("harmonic", "rk4", float(h), 20000, n_sample=50).ok),
                        2.0, 3.0, tol=1e-4, assume="true_at_lo", note="RK4 的边界是 2√2≈2.828")
    assert abs(rk.threshold - 2.0 * math.sqrt(2.0)) < 5e-3, f"RK4 的边界应为 2√2，实测 {rk.threshold:.6f}"
    record(
        label="谐振子的稳定边界（二分实测）",
        theory="Verlet/辛 Euler：h<2；RK4：h<2√2≈2.8284",
        measured=f"Verlet {res.threshold:.6f}（{res.calls} 次求值、{res.iterations} 次二分）、RK4 {rk.threshold:.6f}（{rk.calls} 次求值）",
        tol="1e-3 / 5e-3",
        passed=True,
        kind="C",
        extra={"verlet_h": res.threshold, "rk4_h": rk.threshold},
    )


@assertion("C", "Kepler 形状失真的阈值：Verlet 有界、RK4 随观测时长线性增长（growth ≈ 2）", chapter="3.3")
def test_kepler_shape_threshold_between_symplectic_and_rk4() -> None:
    n = int(round(200 * 2 * math.pi / 0.01))
    rows = []
    growth = {}
    for m in ("verlet", "symplectic_euler", "implicit_midpoint", "rk4"):
        r = run("kepler_ecc_0.6", m, 0.01, n, n_sample=4000)
        a, e, _w = sp.orbital_elements(r.q, r.p)
        cl = iv.closure_stats((a * (1.0 - e)) / 0.4 - 1.0)
        growth[m] = cl.growth
        rows.append(f"{m}: sup={cl.sup:.3e} growth={cl.growth:.3f}")
    for m in ("verlet", "symplectic_euler", "implicit_midpoint"):
        assert growth[m] < 1.05, f"{m} 的形状失真应有界（growth≈1），实测 {growth[m]:.3f}"
    assert growth["rk4"] > 1.7, f"RK4 的形状失真应随时间线性增长（growth≈2），实测 {growth['rk4']:.3f}"
    record(
        label="形状失真的增长比（前/后半段）",
        theory="辛方法：有界振荡 ⟹ growth≈1；RK4：线性漂移 ⟹ growth≈2",
        measured="；".join(rows),
        tol="辛 < 1.05；RK4 > 1.7",
        passed=True,
        kind="C",
        extra=growth,
    )


@assertion("C", "隐式中点的迭代次数随 h 单调下降，残差稳定在 1e-14（Newton 收敛性不是摆设）", chapter="3.3")
def test_implicit_midpoint_iteration_counts_shrink_with_h() -> None:
    hs = [0.5, 0.2, 0.1, 0.05]
    iters = []
    resid = []
    for h in hs:
        r = sp.integrate("harmonic", "implicit_midpoint", h, 2000, n_sample=100)
        iters.append(r.iterations_total / 2000.0)
        resid.append(r.residual_max)
        assert r.converged, f"h={h} 未收敛"
    assert all(a > b for a, b in zip(iters, iters[1:])), f"迭代次数应随 h 减小而减少，实测 {iters}"
    assert max(resid) < 1e-12, f"残差应稳定在 1e-14 量级，实测最大 {max(resid):.3e}"
    assert_leq(
        iters[-1],
        5.0,
        atol=0.05,
        label="小步长时每步 ≤5 次迭代",
        detail=f"h 阶梯 {hs} → 平均迭代 {[round(v, 3) for v in iters]}",
    )
    record(
        label="隐式中点的迭代次数与 h 的关系",
        theory="迭代解 y = y_n + h/2·[f(y_n)+f(y)] 的收缩率 ∝ h ⟹ 迭代次数随 h 减小",
        measured=f"h={hs} → 平均迭代 {[round(v, 3) for v in iters]}，残差最大 {max(resid):.3e}",
        tol="单调下降且末端 ≤ 5.05",
        passed=True,
        kind="C",
        extra={"iters": iters},
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
