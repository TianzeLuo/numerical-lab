"""3.2 章（上半）：稳定域、稳定函数与 A/L/A(α)-稳定性的 A/C 类断言。

这一半只谈「一个方法稳不稳」，不谈它解得多准。所有结论都从两样东西来：
**Butcher 表**（显式 RK 的 :math:`R(z)` 由 :math:`b^TA^{j-1}e` 手算出来）与
**特征多项式**（线性多步法的 :math:`\\rho/\\sigma`）。因此这里的对拍对象都是
闭式表达式或几何事实：圆、虚轴、三次方程的根——**不是**另一份同样的代码。

数值上最值得盯的两条：

* 多步法的 :math:`|R|` 要用自写的 Durand–Kerner 求根（P3 不许 ``np.roots``），
  所以专门有一条断言把它与 BDF2 的闭式根 :math:`1/(2+\\sqrt{1+2z})` 逐点对拍；
* 边界用参数化公式 :math:`z(\\theta)=\\rho/\\sigma` 给出，再回代验算
  :math:`\\bigl||R(z_\\theta)|-1\\bigr|` —— "边界画对了"必须自己证明。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from numcore.ode import stability as stab
from numcore.verify import (
    assert_leq,
    assert_order,
    assert_threshold,
    assertion,
    find_threshold,
    record,
)

U = float(np.finfo(float).eps) / 2.0
PI = math.pi

#: 几个有代表性的 :math:`z`（正实、负实、虚轴、圆外、大模）
Z_SAMPLES = np.array(
    [0.0, 0.5, -1.0, -2.0, 2.0, -3.5 + 0.0j, 0.0 + 2.0j, -1.0 + 1.5j, -4.0 - 3.0j, 0.3 - 6.0j, -30.0 + 12.0j]
)


# --------------------------------------------------------------------------- #
# A 类：稳定函数本身
# --------------------------------------------------------------------------- #
@assertion(
    "A",
    "4 级 4 阶显式 RK（经典 RK4）的稳定多项式 = 截断指数 1+z+z²/2+z³/6+z⁴/24",
    chapter="3.2",
)
def test_rk4_stability_polynomial_is_truncated_exponential() -> None:
    coeffs = np.asarray(stab.rk_stability_coefficients("rk4"), dtype=float)
    exact = np.array([1.0 / math.factorial(j) for j in range(5)])
    worst = float(np.max(np.abs(coeffs - exact)))
    # 另一条独立路径：直接对复点求值，与手写 Horner 的截断指数对拍
    ref = np.array([sum(z**j / math.factorial(j) for j in range(5)) for z in Z_SAMPLES])
    got = np.asarray(stab.stability_function("rk4", 4, Z_SAMPLES), dtype=complex)
    scale = float(np.max(np.abs(ref)))
    worst_eval = float(np.max(np.abs(got - ref))) / scale  # 相对：|R| 在样本上到 1e6
    assert_leq(worst, 8 * U, label="RK4 稳定多项式系数 vs 1/j!")
    assert_leq(worst_eval, 64 * U, label="RK4 R(z) 求值 vs 截断指数（相对）")
    record(
        label="RK4 系数最大偏差 / 求值最大偏差",
        theory="0",
        measured=f"{worst:.3e} / {worst_eval:.3e}",
        tol=f"{8 * U:.1e}（求值按 |R| 的相对量）",
        passed=worst <= 8 * U and worst_eval <= 64 * U,
        kind="A",
    )


@assertion(
    "A",
    "隐式 Euler R=1/(1−z)；梯形 R=(1+z/2)/(1−z/2)；BDF1 与隐式 Euler 同；BDF2 R=1/(2+sqrt(1+2z))",
    chapter="3.2",
)
def test_rational_stability_functions_match_closed_forms() -> None:
    # z=2 是梯形法的极点（1−z/2=0）：闭式在那里也是 1/0，单独在最后查
    z = Z_SAMPLES[Z_SAMPLES != 2.0].astype(complex)
    ie = np.asarray(stab.stability_function("implicit_euler", 1, z), dtype=complex)
    tr = np.asarray(stab.stability_function("trapezoidal", 2, z), dtype=complex)
    b1 = np.asarray(stab.stability_function("bdf", 1, z), dtype=complex)
    d_ie = float(np.max(np.abs(ie - 1.0 / (1.0 - z))))
    d_tr = float(np.max(np.abs(tr - (1.0 + 0.5 * z) / (1.0 - 0.5 * z))))
    d_b1 = float(np.max(np.abs(b1 - ie)))
    # BDF2：特征方程 (3/2 − z)r² − 2r + 1/2 = 0 的根是 [2 ± sqrt(1+2z)]/(3 − 2z)，
    # 有理化后 = 1/(2 ∓ sqrt(1+2z))；其中 r=1 的那支（z=0 时）正是 1/(2 − sqrt(1+2z))，
    # 它在左半平面里模不小于另一支，所以"最大根模"取的就是它。
    zz = z[np.abs(z) < 20.0]
    b2 = np.asarray(stab.stability_function("bdf", 2, zz), dtype=complex)
    d_b2 = float(np.max(np.abs(b2 - 1.0 / (2.0 - np.sqrt(1.0 + 2.0 * zz)))))
    worst = max(d_ie, d_tr, d_b1, d_b2)
    assert_leq(worst, 1e4 * U, label="有理稳定函数 vs 闭式")
    with np.errstate(divide="ignore", invalid="ignore"):
        pole_mag = float(np.max(stab.stability_modulus("trapezoidal", 2, np.array([2.0]))))
    assert math.isinf(pole_mag), f"梯形法在 z=2 的极点处应当是 +∞，实测 {pole_mag}"
    record(
        label="四个闭式对拍的最大偏差",
        theory="0",
        measured=f"隐式Euler {d_ie:.2e}，梯形 {d_tr:.2e}，BDF1 {d_b1:.2e}，BDF2 {d_b2:.2e}；极点 |R(2)|={pole_mag}",
        tol=f"{1e4 * U:.1e}",
        passed=worst <= 1e4 * U,
        kind="A",
        extra={"d_implicit_euler": d_ie, "d_trapezoidal": d_tr, "d_bdf1": d_b1, "d_bdf2": d_b2},
    )


@assertion(
    "A",
    "多步法求根路线（自写 Durand–Kerner）与闭式根一致；BDF2 的最大根模 = |1/(2−sqrt(1+2z))|（z=0 处 = 1）",
    chapter="3.2",
)
def test_durand_kerner_matches_closed_form_bdf2() -> None:
    alpha, beta_k = stab.bdf_coefficients(2)
    # 两支根是 1/(2∓sqrt(1+2z))：z=0 时分别给出 1（物理根）与 1/3，
    # 而在整个左半平面 + 右半平面的稳定侧，物理根 1/(2−sqrt(1+2z)) 的模都不小于另一支，
    # 所以"最大根模"就是它。唯一的例外是方法自身的极点 z=1.5（3−2z=0），那里两支都发散，
    # 相对比较失去意义——把它挖掉，再单独查极点位置。
    z = np.linspace(-4.0, 4.0, 161).astype(complex)
    z = z[np.abs(z - 1.5) > 0.05]
    got = np.asarray(stab.lmm_root_modulus(alpha, beta_k, z), dtype=float)
    s = np.sqrt(1.0 + 2.0 * z)
    ref = np.abs(1.0 / (2.0 - s))
    d = float(np.max(np.abs(got - ref) / np.maximum(ref, 1.0)))
    d_two = float(np.max(np.abs(got - np.maximum(np.abs(1.0 / (2.0 - s)), np.abs(1.0 / (2.0 + s)))) / np.maximum(ref, 1.0)))
    # 迭代次数对精度的影响：48 次与 240 次之差应当埋在舍入里
    got240 = np.asarray(stab.lmm_root_modulus(alpha, beta_k, z, iters=240), dtype=float)
    d_iter = float(np.max(np.abs(got - got240)))
    # 极点：z→1.5 时 |R| 发散（3−2z→0），这是方法的极点而不是"求根失败"
    near = float(np.max(stab.stability_modulus("bdf", 2, np.array([1.5 - 1e-6, 1.5 + 1e-6]))))
    assert_leq(max(d, d_two, d_iter), 1e-12, label="DK 求根 vs 闭式（并检查迭代次数收敛）")
    assert near > 1e5, f"BDF2 在 z=1.5 附近应当发散，实测 {near:.3e}"
    record(
        label="BDF2 最大根模：DK vs 闭式 / 迭代 48 vs 240",
        theory="0",
        measured=f"物理根（相对）{d:.3e} / 双根取最大（相对）{d_two:.3e} / 迭代差 {d_iter:.3e} / 极点附近 |R|={near:.3e}",
        tol="1e-12（极点 1.5 附近挖去 ±0.05）",
        passed=max(d, d_two, d_iter) <= 1e-12 and near > 1e5,
        kind="A",
        extra={"dk_vs_closed": d, "dk_vs_maxroot": d_two, "iter_gap": d_iter, "pole_modulus": near},
    )


@assertion(
    "A",
    "两条独立路径给出同一个 |R|：多项式的 Horner 求值 vs 共轭配对求模；多步法则用根模",
    chapter="3.2",
)
def test_modulus_second_route_agrees() -> None:
    worst = 0.0
    detail = []
    poles_ok = True
    for name, order in (("forward_euler", 1), ("rk4", 4), ("implicit_euler", 1), ("trapezoidal", 2), ("bdf", 2), ("bdf", 5)):
        a = np.asarray(stab.stability_modulus(name, order, Z_SAMPLES), dtype=float)
        b = np.asarray(stab.modulus_second_route(name, order, Z_SAMPLES), dtype=float)
        # z=2 是梯形法的极点（|R|=∞）：两条路径必须**同时**报非有限，
        # 但非有限点本身不能拿来做减法（inf−inf=nan）。
        poles_ok = poles_ok and bool(np.array_equal(np.isfinite(a), np.isfinite(b)))
        finite = np.isfinite(a) & np.isfinite(b)
        d = float(np.max(np.abs(a[finite] - b[finite]))) if bool(np.any(finite)) else 0.0
        poles = int(np.count_nonzero(~finite))
        worst = max(worst, d)
        detail.append(f"{name}{order}:{d:.1e}" + (f"(极点{poles})" if poles else ""))
    assert_leq(worst, 1e-12, label="双路径 |R| 之差")
    assert poles_ok, "两条路径对极点位置的判断不一致"
    record(
        label="双路径最大差（极点位置另查）",
        theory="0；极点处两条路径都必须是 +∞（梯形法 z=2）",
        measured=f"{worst:.3e}（{', '.join(detail)}）；极点判定一致={poles_ok}",
        tol="1e-12",
        passed=worst <= 1e-12 and poles_ok,
        kind="A",
    )


# --------------------------------------------------------------------------- #
# A 类：稳定域边界（几何事实）
# --------------------------------------------------------------------------- #
@assertion(
    "A",
    "边界回代必须满足 |R(z_θ)|=1（这正是「边界画对了」的定义）；四条已知边界：前向 Euler 是 |z+1|=1 圆、隐式 Euler 是 |z−1|=1 圆、梯形是虚轴、BDF1 与隐式 Euler 重合",
    chapter="3.2",
)
def test_boundary_lies_on_unit_modulus_circle() -> None:
    rows = []
    worst = 0.0
    for name, order in (("forward_euler", 1), ("heun", 2), ("rk4", 4), ("implicit_euler", 1), ("trapezoidal", 2), ("bdf", 1), ("bdf", 2), ("bdf", 3), ("bdf", 4)):
        z = stab.stability_boundary(name, order, n_theta=1025)
        m = np.asarray(stab.stability_modulus(name, order, z), dtype=float)
        d = float(np.max(np.abs(m - 1.0))) if m.size else float("nan")
        worst = max(worst, d)
        rows.append(f"{name}{order}:{d:.1e}({z.size}点)")
    assert_leq(worst, 1e-11, label="边界 |R|=1 回代残差")
    # 几何对拍
    fe = stab.stability_boundary("forward_euler", 1, n_theta=513)
    ie = stab.stability_boundary("implicit_euler", 1, n_theta=513)
    tr = stab.stability_boundary("trapezoidal", 2, n_theta=513)
    b1 = stab.stability_boundary("bdf", 1, n_theta=513)
    d_fe = float(np.max(np.abs(np.abs(fe + 1.0) - 1.0)))
    d_ie = float(np.max(np.abs(np.abs(ie - 1.0) - 1.0)))
    d_tr = float(np.max(np.abs(np.real(tr)))) if tr.size else 0.0
    # 圆 vs 外凸曲线：把两条曲线按最近点距离比较（参数化不同，不能逐点相减）
    def _hausdorff(a: np.ndarray, b: np.ndarray) -> float:
        d = np.abs(a[:, None] - b[None, :])
        return float(max(np.max(np.min(d, axis=1)), np.max(np.min(d, axis=0))))

    d_b1 = _hausdorff(b1, ie)
    worst_geo = max(d_fe, d_ie, d_tr, d_b1)
    assert_leq(worst_geo, 1e-3, label="已知边界的几何对拍（圆 / 虚轴 / 重合）")
    record(
        label="边界回代残差与几何事实",
        theory="|R|=1；|z+1|=1；|z−1|=1；Re z=0",
        measured=f"回代最大 {worst:.2e}（{'; '.join(rows)}）；几何最大 {worst_geo:.2e}",
        tol="1e-11 / 1e-3",
        passed=worst <= 1e-11 and worst_geo <= 1e-3,
        kind="A",
        extra={"residual": worst, "geometry": worst_geo, "d_circle_fe": d_fe, "d_circle_ie": d_ie, "d_axis_tr": d_tr, "d_bdf1_ie": d_b1},
    )


@assertion(
    "A",
    "显式方法在负实轴上的稳定区间端点：前向 Euler 是 −2（精确），RK4 是 x³+4x²+12x+24=0 的实根 −2.785293…；p 阶 Taylor 多项式族的端点是 R_p(x)=1 的负根",
    chapter="3.2",
)
def test_negative_real_axis_endpoints() -> None:
    fe = stab.negative_real_endpoint("forward_euler", 1)
    rk4 = stab.negative_real_endpoint("rk4", 4)
    # 独立算一遍三次方程的根（自写二分，不用 numpy.roots——P3）
    lo, hi = -3.0, -2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid**3 + 4.0 * mid**2 + 12.0 * mid + 24.0 < 0.0:
            lo = mid
        else:
            hi = mid
    cubic = 0.5 * (lo + hi)
    d_fe = abs(fe + 2.0)
    d_rk4 = abs(rk4 - cubic)
    d_identity = abs(float(np.max(stab.stability_modulus("rk4", 4, np.array([rk4])))) - 1.0)
    assert_leq(max(d_fe, d_rk4, d_identity), 1e-10, label="实轴端点 vs 三次方程实根")
    tay = {p: stab.negative_real_endpoint("explicit_taylor", p) for p in (2, 3, 4, 5, 6)}
    tay_res = float(max(abs(float(np.max(stab.stability_modulus("explicit_taylor", p, np.array([tay[p]])))) - 1.0) for p in tay))
    assert_leq(tay_res, 1e-9, label="Taylor 族端点处 |R|=1")
    record(
        label="负实轴端点",
        theory="−2（Euler）；三次根 −2.785293563405282（RK4）",
        measured=f"Euler {fe:.15f}（差 {d_fe:.1e}），RK4 {rk4:.15f}（与独立二分之差 {d_rk4:.1e}），"
        f"|R| 回代差 {d_identity:.1e}；Taylor p=2..6 端点 {[round(tay[p], 4) for p in sorted(tay)]}（回代差 {tay_res:.1e}）",
        tol="1e-10",
        passed=max(d_fe, d_rk4, d_identity) <= 1e-10 and tay_res <= 1e-9,
        kind="A",
        extra={"euler": fe, "rk4": rk4, "cubic_root": cubic, "taylor": tay},
    )


@assertion(
    "A",
    "RK4 的稳定域在虚轴上恰好触及 ±2√2 i（|R(iy)|≤1 的边界），所以它对振荡问题要求 hω<2√2",
    chapter="3.2",
)
def test_rk4_imaginary_axis_crossing_is_two_sqrt_two() -> None:
    target = 2.0 * math.sqrt(2.0)
    mod = lambda y: float(np.max(stab.stability_modulus("rk4", 4, complex(0.0, y))))  # noqa: E731
    # 第一次穿越：|R(iy)|≤1 在 [0, 2√2] 上成立，越过 2√2 立刻 >1
    lo, hi = 0.0, 8.0
    assert mod(lo) <= 1.0 and mod(hi) > 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mod(mid) <= 1.0:
            lo = mid
        else:
            hi = mid
    crossing = 0.5 * (lo + hi)
    d = abs(crossing - target)
    interior = max(mod(y) for y in np.linspace(0.0, target, 501))
    over = interior - 1.0
    beyond = mod(target + 1e-3) - 1.0
    assert_leq(abs(over), 1e-9, label="[0,2√2] 上 |R(iy)|≤1")
    assert_leq(d, 1e-9, label="RK4 虚轴第一次穿越 = 2√2")
    assert beyond > 0.0, f"越过 2√2 后应当 >1，实测 {1 + beyond:.9f}"
    record(
        label="RK4 虚轴穿越点",
        theory=f"2√2 = {target:.12f}（|R(iy)|=1 的第一个正根）",
        measured=f"二分得到 {crossing:.12f}（差 {d:.2e}）；区间内最大超出 {over:+.2e}；"
        f"越过 1e-3 后 |R|={1 + beyond:.9f}",
        tol="1e-9",
        passed=abs(over) <= 1e-9 and d <= 1e-9 and beyond > 0.0,
        kind="A",
        extra={"crossing": crossing, "two_sqrt_two": target},
    )


# --------------------------------------------------------------------------- #
# A 类：A-稳定 / L-稳定 / Dahlquist
# --------------------------------------------------------------------------- #
@assertion(
    "A",
    "A-稳定：隐式 Euler、梯形、BDF1、BDF2 是；前向 Euler、Heun、中点、RK4 都不是（显式方法 R 是多项式，|z|→∞ 时无界）",
    chapter="3.2",
)
def test_a_stability_classification() -> None:
    yes = ["implicit_euler", "trapezoidal", "bdf"]
    rows = []
    ok = True
    for name in ("forward_euler", "midpoint", "heun", "rk4", "explicit_taylor"):
        rep = stab.stability_report(name, 4 if name == "rk4" else 2)
        rows.append(f"{name}:A={rep.a_stable}")
        ok = ok and (not rep.a_stable)
    for name in ("implicit_euler", "trapezoidal", "bdf"):
        for order in ((1,) if name == "implicit_euler" else (2,) if name == "trapezoidal" else (1, 2)):
            rep = stab.stability_report(name, order)
            rows.append(f"{name}{order}:A={rep.a_stable},角={rep.angle_deg:.4f}")
            ok = ok and rep.a_stable
    # 独立判据（单步法）：虚轴上的最大模 ≤ 1
    max_im = {}
    for name, order in (("implicit_euler", 1), ("trapezoidal", 2)):
        m, _ = stab.max_on_imaginary_axis(name, order, y_max=200.0, n=200001)
        max_im[f"{name}{order}"] = m
    assert_leq(max(abs(v - 1.0) for v in max_im.values()), 1e-9, label="A-稳定单步法的虚轴最大模 = 1")
    assert ok, rows
    record(
        label="A-稳定性分类（含虚轴独立判据）",
        theory="隐式 Euler/梯形/BDF1/BDF2 为真，显式方法为假；隐式与梯形 max|R(iy)|=1",
        measured="；".join(rows) + "；" + "，".join(f"{k}={v:.12f}" for k, v in max_im.items()),
        tol="分类一致且虚轴最大模 = 1±1e-9",
        passed=ok and max(abs(v - 1.0) for v in max_im.values()) <= 1e-9,
        kind="A",
        extra={"max_on_imaginary": max_im},
    )


@assertion(
    "A",
    "L-稳定：隐式 Euler 与 BDF2 在 z→−∞ 时放大因子→0；梯形法 A-稳定但不 L-稳定（→1）",
    chapter="3.2",
)
def test_l_stability_and_trapezoidal_failure() -> None:
    ie = stab.stiff_limit_amplification("implicit_euler", 1)
    tr = stab.stiff_limit_amplification("trapezoidal", 2)
    b2 = stab.stiff_limit_amplification("bdf", 2)
    ok = ie <= 1e-6 and b2 <= 1e-6 and abs(tr - 1.0) <= 1e-9
    assert ok, f"隐式 Euler {ie:.3e} / BDF2 {b2:.3e} / 梯形 {tr:.12f}"
    record(
        label="z=−1e14 处的放大因子",
        theory="隐式 Euler 与 BDF2 → 0；梯形 → 1（A-稳定但非 L-稳定）",
        measured=f"隐式 Euler {ie:.3e}，BDF2 {b2:.3e}，梯形 {tr:.12f}",
        tol="≤1e-6（前两者），|·−1|≤1e-9（梯形）",
        passed=ok,
        kind="A",
        extra={"implicit_euler": ie, "trapezoidal": tr, "bdf2": b2},
    )


@assertion(
    "A",
    "Dahlquist 第二障碍：A-稳定的线性多步法阶 ≤ 2，所以 BDF 里只有 k=1,2 是 A-稳定的",
    chapter="3.2",
)
def test_dahlquist_second_barrier() -> None:
    stable_orders = stab.a_stable_orders_bdf()
    ev = stab.dahlquist_barrier_evidence()
    rec = record(
        label="BDF 的 A-稳定阶集合",
        theory="(1, 2)",
        measured=f"{stable_orders}；证据：{ev}",
        tol="集合相等",
        passed=tuple(stable_orders) == (1, 2),
        kind="A",
        extra={"stable_orders": list(stable_orders)},
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# A 类：A(α) 角与理论值
# --------------------------------------------------------------------------- #
@assertion(
    "A",
    "BDF-k 的 A(α) 角实测值 = 理论值 90/90/86.03/73.35/51.84/17.84 度（k=1..6），误差 < 0.002 度",
    chapter="3.2",
)
def test_bdf_angles_match_theory() -> None:
    rows = []
    worst = 0.0
    for k in range(1, 7):
        ang, a_ok = stab.bdf_stability_angle_deg(k)
        theory = stab.BDF_ANGLE_THEORY_DEG[k]
        d = abs(ang - theory)
        worst = max(worst, d)
        rows.append(f"k={k}:{ang:.4f}/{theory:.2f}(A={a_ok})")
    # 理论常数在文献里只报两位小数（86.03/73.35/51.84/17.84），所以对拍的极限
    # 就是那半个末位的舍入（≈0.005°）；实测偏差全部来自这里，不是算法误差。
    assert_leq(worst, 3e-3, label="BDF 稳定角 vs 理论（理论值只有两位小数）")
    record(
        label="BDF 稳定角实测 vs 理论",
        theory="90, 90, 86.03, 73.35, 51.84, 17.84",
        measured=f"最大偏差 {worst:.2e} 度；{'; '.join(rows)}",
        tol="3e-3 度（理论值 2 位小数的舍入上限 0.005 度以内）",
        passed=worst <= 3e-3,
        kind="A",
        extra={"worst": worst},
    )


# --------------------------------------------------------------------------- #
# C 类：实测阈值
# --------------------------------------------------------------------------- #
@assertion(
    "C",
    "BDF 失去 A-稳定性的阶数阈值：角度 ≥ 90° 的最大整数 k = 2（二分实测）",
    chapter="3.2",
)
def test_bdf_a_stability_order_threshold() -> None:
    cache: dict[int, float] = {}

    def angle(k: int) -> float:
        if k not in cache:
            cache[k] = stab.bdf_stability_angle_deg(int(k))[0]
        return cache[k]

    res = find_threshold(lambda k: angle(int(round(k))) >= 90.0, 1.0, 6.0, tol=1e-6, note="角度 ≥ 90° 视为 A-稳定")
    # 二分落在 2 与 3 之间，取整后应当正好是 2
    measured_k = math.floor(res.threshold)
    assert_threshold(measured_k, 2.0, rtol=1e-12, label="BDF 保持 A-稳定的最大阶", detail=f"二分区 [{res.lo:.6f}, {res.hi:.6f}]")
    record(
        label="角度随 k 的收缩",
        theory="k=1,2 为 90°，k=3 掉到 86.03°",
        measured=f"阈值落在 k∈[{res.lo:.4f},{res.hi:.4f}]，取整后 {measured_k}；角度序列 "
        + ", ".join(f"{angle(k):.4f}" for k in range(1, 7)),
        tol="阈值区间含 2.5 且整数部分为 2",
        passed=res.lo <= 2.5 <= res.hi and measured_k == 2,
        kind="C",
        extra={"threshold": res.threshold, "iterations": res.iterations},
    )


@assertion(
    "C",
    "第一次失去 A-稳定性时角度只掉 3.97°（86.0324 vs 90），但到 k=6 只剩 17.84°——收缩是加速的",
    chapter="3.2",
)
def test_bdf_angle_shrinkage_is_measured() -> None:
    ang = np.array([stab.bdf_stability_angle_deg(k)[0] for k in range(1, 7)])
    drops = -np.diff(ang)
    first = float(drops[1])  # 90 → 86.03
    last = float(drops[-1])  # 51.84 → 17.84
    ratio = last / max(first, 1e-300)
    # 单调收缩（允许 1e-9 的平台：k=1,2 都是 90°）
    mono = bool(np.all(drops >= -1e-9))
    assert_threshold(first, 90.0 - stab.BDF_ANGLE_THEORY_DEG[3], rtol=2e-3, label="k=2→3 的角度跌幅")
    assert mono and ratio > 5.0, f"drops={drops}, ratio={ratio}"
    record(
        label="角度跌幅序列与加速比",
        theory="第一次跌 3.968°，最后一次跌 34.0°，加幅 > 5 倍",
        measured=f"跌幅 {np.array2string(drops, precision=3)}，首跌 {first:.4f}°，末跌 {last:.4f}°，比 {ratio:.2f}",
        tol="单调 + 比 > 5",
        passed=mono and ratio > 5.0,
        kind="C",
        extra={"drops": drops.tolist(), "ratio": ratio},
    )


@assertion(
    "C",
    "显式方法的稳定域面积随阶数增大：实测边界围成的面积 RK4 > Heun > 前向 Euler",
    chapter="3.2",
)
def test_explicit_region_area_grows_with_order() -> None:
    def area(name: str, order: int) -> float:
        z = stab.stability_boundary(name, order, n_theta=4097)
        return float(abs(np.sum(np.real(z) * np.roll(np.imag(z), -1) - np.roll(np.real(z), -1) * np.imag(z))) / 2.0)

    a1 = area("forward_euler", 1)
    a2 = area("heun", 2)
    a4 = area("rk4", 4)
    assert a1 < a2 < a4, (a1, a2, a4)
    record(
        label="显式稳定域面积",
        theory="Euler 的域是圆心 −1、半径 1 的圆，面积 π=3.14159…；Heun 更大；RK4 最大",
        measured=f"Euler {a1:.6f}（圆面积 π={PI:.6f}），Heun {a2:.6f}，RK4 {a4:.6f}",
        tol="严格递增；Euler 与 π 相对差 < 1e-3",
        passed=a1 < a2 < a4 and abs(a1 - PI) < 1e-3 * PI,
        kind="C",
        extra={"area_forward_euler": a1, "area_heun": a2, "area_rk4": a4},
    )


@assertion(
    "C",
    "梯形法的稳定域 = 整个左半平面：沿任意角度 α∈(0°,90°) 的射线取 r→∞ 时 |R|→1 且恒 ≤1",
    chapter="3.2",
)
def test_trapezoidal_region_is_left_half_plane() -> None:
    r = np.logspace(-3.0, 6.0, 200)
    worst = 0.0
    rows = []
    for deg in (1.0, 15.0, 45.0, 75.0, 89.9):
        th = math.radians(deg)
        z = -r * (math.cos(th) - 1j * math.sin(th))
        m = np.asarray(stab.stability_modulus("trapezoidal", 2, z), dtype=float)
        worst = max(worst, float(np.max(m)) - 1.0)
        rows.append(f"{deg:g}°:max={float(np.max(m)):.12f}")
    assert_leq(abs(worst), 1e-12, label="梯形法在左半平面 |R|≤1")
    record(
        label="梯形法左半平面射线的 |R| 上确界",
        theory="1（A-稳定）",
        measured=f"超出 1 的最大量 {worst:+.3e}；{'; '.join(rows)}",
        tol="1e-12",
        passed=abs(worst) <= 1e-12,
        kind="C",
    )
